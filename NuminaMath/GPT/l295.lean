import Mathlib

namespace max_min_f_fraction_f_zero_l295_295779

section
variables {x : ℝ}
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x - 2 * cos x

theorem max_min_f : 
  (x ∈ set.Icc (0 : ℝ) π) → (∀ y ∈ set.image f (set.Icc 0 π), y ≤ 4 ∧ y ≥ -2) := by sorry

theorem fraction_f_zero :
  f x = 0 → 
  (2 * cos (x / 2) ^ 2 - sin x - 1) / (sqrt 2 * sin (x + π / 4)) = 2 - sqrt 3 := by sorry
end

end max_min_f_fraction_f_zero_l295_295779


namespace common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295102

def is_geometric_sequence (a : ℕ → ℂ) (q : ℂ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_mean (a1 a2 a3 : ℂ) : Prop :=
  2 * a1 = a2 + a3

noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  (1 - (1 + 3 * n) * (-2 : ℤ)^n) / 9

theorem common_ratio_of_geometric_sequence (a : ℕ → ℂ) (q : ℂ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : q ≠ 1) 
  (h3 : arithmetic_mean (a 1) (a 2) (a 3)) : 
  q = -2 := 
sorry

theorem sum_of_first_n_terms_of_sequence (n : ℕ) 
  (a : ℕ → ℂ) 
  (h1 : is_geometric_sequence a (-2)) 
  (h2 : a 1 = 1) : 
  ∑ k in finset.range n, k * a k = sum_first_n_terms n := 
sorry

end common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295102


namespace B_M_N_collinear_l295_295833

variable 
  (A B C D E F G H M N : Point)
  (ABCD : Quadrilateral)
  (AngleB : RightAngle (angle B))
  (diags_eq : distance A C = distance B D)
  (E_mid_AB : midpoint E A B)
  (G_mid_CD : midpoint G C D)
  (N_perp_bisect_EG : perpendicular_bisector_through N E G)
  (F_mid_BC : midpoint F B C)
  (H_mid_AD : midpoint H A D)
  (M_perp_bisect_FH : perpendicular_bisector_through M F H)

theorem B_M_N_collinear :
  collinear B M N := 
sorry

end B_M_N_collinear_l295_295833


namespace angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l295_295229

theorem angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees 
  (cube : Type*)
  [measure_theory.measure_space cube]
  (adjacent_sides_diagonals_perpendicular : ∀ (x y z : cube), ∃ (d1 d2 : cube),
    d1 ≠ d2 ∧ is_diagonal_of_adjacent_sides_of_cube d1 d2 ∧ ∠ d1 d2 = 90) :
  ∀ (W : Type*) [measure_theory.measure_space W], is_right_angle W :=
by
  sorry

noncomputable def is_diagonal_of_adjacent_sides_of_cube (d1 d2 : Type*) : Prop :=
  sorry

noncomputable def is_right_angle (W : Type*) : Prop :=
  W = 90

end angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l295_295229


namespace area_ratio_of_region_R_to_square_l295_295610

theorem area_ratio_of_region_R_to_square (s : ℝ) (h : s > 0) :
  let A := (0, 0 : ℝ × ℝ),
      B := (s, 0 : ℝ × ℝ),
      C := (s, s : ℝ × ℝ),
      D := (0, s : ℝ × ℝ),
      M := (s / 2, s : ℝ × ℝ),
      center := (s / 2, s / 2 : ℝ × ℝ),
      square_area := s * s,
      smaller_square_area := (s / 2) * (s / 2),
      ratio := smaller_square_area / square_area
  in ratio = 1 / 4 := 
by
  sorry

end area_ratio_of_region_R_to_square_l295_295610


namespace lateral_surface_area_cylinder_l295_295172

-- Given: the base area S and the lateral surface of the cylinder unfolds into a square
variables (S : ℝ)

-- The lateral surface area of the cylinder
theorem lateral_surface_area_cylinder (h_base_area : ∀ (r : ℝ), r^2 * π = S) 
    (h_lateral_surface_square : ∃ (h : ℝ), h = 2 * π * sqrt (S / π)) :
    ∃ A : ℝ, A = 4 * π * S :=
by
  sorry

end lateral_surface_area_cylinder_l295_295172


namespace sum_f_1_to_8_eq_9_to_16_sum_f_1_to_2020_l295_295367

-- Define the function f
def f (k : ℤ) : ℝ := Real.sin (k * Real.pi / 4)

-- Theorem 1: Prove that sum of f from 1 to 8 equals the sum of f from 9 to 16
theorem sum_f_1_to_8_eq_9_to_16 : 
  (∑ k in Finset.range 8, f (k + 1)) = (∑ k in Finset.range 8, f (k + 9)) :=
by
  sorry

-- Theorem 2: Find the value of the sum of f from 1 to 2020
theorem sum_f_1_to_2020 : 
  (∑ k in Finset.range 2020, f (k + 1)) = 1 + Real.sqrt 2 :=
by
  sorry

end sum_f_1_to_8_eq_9_to_16_sum_f_1_to_2020_l295_295367


namespace triangle_inequalities_l295_295204

-- Definitions for the problem conditions
variables {A B C D E F : Type} -- Vertices of triangles
variables {a b c S : ℝ} -- Side lengths and area of triangle
variables {AT BT CT DE : ℝ} -- Lengths of specific segments

-- Hypotheses corresponding to the conditions
-- Assume vertices D, E, F are on sides BC, CA, AB respectively
variables (hDEF_triangle : D ∈ line_segment B C ∧ E ∈ line_segment C A ∧ F ∈ line_segment A B)
-- Assume lengths of sides of triangle ABC
variables (ha : a = distance B C)
variables (hb : b = distance C A)
variables (hc : c = distance A B)
-- Assume the area of triangle ABC
variables (hS_area : S = triangle_area A B C)
-- Assume lengths of specific segments
variables (hAT : AT = distance A T)
variables (hBT : BT = distance B T)
variables (hCT : CT = distance C T)
variables (hDE : DE = distance D E)

-- Statement of the problem in Lean 
theorem triangle_inequalities
  (hDEF_triangle : D ∈ line_segment B C ∧ E ∈ line_segment C A ∧ F ∈ line_segment A B)
  (ha : a = distance B C) 
  (hb : b = distance C A)
  (hc : c = distance A B)
  (hS_area : S = triangle_area A B C) 
  (hAT : AT = distance A T)
  (hBT : BT = distance B T)
  (hCT : CT = distance C T)
  (hDE : DE = distance D E) :
  2 * S ≤ (AT + BT + CT) * DE ∧ 
  ∃ T : Point, (AT + BT + CT)^2 ≥ (a^2 + b^2 + c^2 + 4 * S * real.sqrt 3) / 2 :=
sorry

end triangle_inequalities_l295_295204


namespace chord_length_l295_295927

/-- Given a circle of radius 1 that touches three sides of a 2 by 4 rectangle,
prove that the length of the chord PQ, where P and Q are the intersections of
the circle and the diagonal of the rectangle, is 4 / sqrt(5). -/
theorem chord_length (P Q : Point) (r : ℝ) (rect_diagonal : Line) :
    r = 1 ∧ in_rectangle P 2 4 ∧ in_rectangle Q 2 4 ∧ intersects_circle rect_diagonal (circle r) P Q →
    chord_length (circle r) P Q = 4 / Real.sqrt 5 := 
sorry

end chord_length_l295_295927


namespace sum_q_p_is_minus_12_l295_295696

noncomputable def p (x : ℝ) : ℝ := x^2 - 3 * x + 2

noncomputable def q (x : ℝ) : ℝ := -x^2

theorem sum_q_p_is_minus_12 :
  (q (p 0) + q (p 1) + q (p 2) + q (p 3) + q (p 4)) = -12 :=
by
  sorry

end sum_q_p_is_minus_12_l295_295696


namespace count_valid_numbers_l295_295429

def is_valid_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ b = a + c

theorem count_valid_numbers : ∃ n : ℕ, n = 45 ∧
  (∀ (a b c : ℕ), is_valid_number a b c → 
    (a = 1 ∧ c ≤ 8) ∨ (a = 2 ∧ c ≤ 7) ∨ 
    (a = 3 ∧ c ≤ 6) ∨ (a = 4 ∧ c ≤ 5) ∨ 
    (a = 5 ∧ c ≤ 4) ∨ (a = 6 ∧ c ≤ 3) ∨ 
    (a = 7 ∧ c ≤ 2) ∨ (a = 8 ∧ c ≤ 1) ∨ 
    (a = 9 ∧ c = 0)) :=
begin
  use 45,
  split,
  { refl, },
  sorry
end

end count_valid_numbers_l295_295429


namespace matrix_pow_A4_l295_295310

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![1, 1]]

-- State the theorem
theorem matrix_pow_A4 :
  A^4 = ![![0, -9], ![9, -9]] :=
by
  sorry -- Proof is omitted

end matrix_pow_A4_l295_295310


namespace probability_even_product_l295_295217

theorem probability_even_product (s : Finset ℕ) (hs : s = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ p : ℚ, p = 5 / 7 ∧ 
    let pairs := (s.product s).filter (λ x, x.1 ≠ x.2) in
    let even_product := pairs.count (λ x, (x.1 * x.2) % 2 = 0) in 
    p = even_product / pairs.card) :=
begin
  sorry
end

end probability_even_product_l295_295217


namespace average_bacterial_count_closest_to_true_value_l295_295967

-- Define the conditions
variables (dilution_spread_plate_method : Prop)
          (count_has_randomness : Prop)
          (count_not_uniform : Prop)

-- State the theorem
theorem average_bacterial_count_closest_to_true_value
  (h1: dilution_spread_plate_method)
  (h2: count_has_randomness)
  (h3: count_not_uniform)
  : true := sorry

end average_bacterial_count_closest_to_true_value_l295_295967


namespace max_fuel_needed_from_D_l295_295940

-- Definitions of the given conditions
def max_fuel_from_D 
  (distance_A_to_BC : ℝ)
  (sum_distances_B_to_AC_C_to_AB : ℝ)
  (fuel_consumption_rate : ℝ)
  : ℝ :=
  -- Distance from A to the road BC
  distance_A_to_BC = 100 ∧ 
  -- Sum of distances from B to the road AC and C to the road AB
  sum_distances_B_to_AC_C_to_AB = 300 ∧ 
  -- Fuel consumption rate in liters per km
  fuel_consumption_rate = 0.1 
  -- The maximum amount of fuel needed by a motorist who needs to get from settlement D to any of the roads
  → 30

-- The theorem to prove
theorem max_fuel_needed_from_D 
  (distance_A_to_BC : ℝ)
  (sum_distances_B_to_AC_C_to_AB : ℝ)
  (fuel_consumption_rate : ℝ)
  : max_fuel_from_D distance_A_to_BC sum_distances_B_to_AC_C_to_AB fuel_consumption_rate = 30 :=
sorry

end max_fuel_needed_from_D_l295_295940


namespace probability_at_least_one_multiple_of_3_l295_295297

/-- Ben twice chooses a random integer between 1 and 50, inclusive, 
and he may choose the same integer both times. 
The probability that at least one of the numbers Ben chooses is a multiple of 3 is 336/625 -/
theorem probability_at_least_one_multiple_of_3 : 
  (2 * 34 / 50) ^ 2 + (2 * 16 / 50) ^ 2 = 336 / 625 :=
sorry

end probability_at_least_one_multiple_of_3_l295_295297


namespace projection_problem_l295_295094

open Real

variables (a b v : ℝ × ℝ)

-- These are the provided conditions
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let s := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2) in
  (s * v.1, s * v.2)

theorem projection_problem 
  (h1 : orthogonal a b) 
  (h2 : proj (4, 2) a = (1, 2)) :
  proj (4, 2) b = (3, 0) :=
sorry

end projection_problem_l295_295094


namespace sufficient_condition_for_parallel_l295_295379

-- Definitions of the planes and lines
variables {α β : Plane} {m : Line}

-- Sufficient condition for m to be parallel to α
theorem sufficient_condition_for_parallel (h1 : α ∥ β) (h2 : ¬(m ⊆ β)) : m ∥ α :=
sorry

end sufficient_condition_for_parallel_l295_295379


namespace percent_calculation_l295_295253

theorem percent_calculation (Part Whole : ℝ) (hPart : Part = 14) (hWhole : Whole = 70) : 
  (Part / Whole) * 100 = 20 := 
by 
  sorry

end percent_calculation_l295_295253


namespace max_ab_l295_295396

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end max_ab_l295_295396


namespace top_card_yellow_second_card_not_yellow_l295_295273

-- Definitions based on conditions
def total_cards : Nat := 65

def yellow_cards : Nat := 13

def non_yellow_cards : Nat := total_cards - yellow_cards

-- Total combinations of choosing two cards
def total_combinations : Nat := total_cards * (total_cards - 1)

-- Numerator for desired probability 
def desired_combinations : Nat := yellow_cards * non_yellow_cards

-- Target probability
def desired_probability : Rat := Rat.ofInt (desired_combinations) / Rat.ofInt (total_combinations)

-- Mathematical proof statement
theorem top_card_yellow_second_card_not_yellow :
  desired_probability = Rat.ofInt 169 / Rat.ofInt 1040 :=
by
  sorry

end top_card_yellow_second_card_not_yellow_l295_295273


namespace waiter_gratuity_l295_295523

def price_leticia : ℕ := 10
def price_scarlett : ℕ := 13
def price_percy : ℕ := 17

def total_cost := price_leticia + price_scarlett + price_percy
def tip_percentage := 0.10
def gratuity := (tip_percentage * total_cost.toReal).toNat

theorem waiter_gratuity : gratuity = 4 :=
sorry

end waiter_gratuity_l295_295523


namespace solve_for_x_l295_295813

theorem solve_for_x (x : ℝ) (h : 9 / (5 + x / 0.75) = 1) : x = 3 :=
by {
  sorry
}

end solve_for_x_l295_295813


namespace face_opposite_y_is_D_l295_295182

-- Define the faces of the cube and declare the folding condition
inductive Face
| A | B | C | D | E | y

-- Hypothesize the configuration of the cube
variable (is_cube_form : Prop)
variable (opposite_face : Face → Face)

-- Specific condition: the face marked 'y' and face 'D' are opposite
axiom cube_condition : is_cube_form → (opposite_face Face.y = Face.D)

theorem face_opposite_y_is_D : is_cube_form → (opposite_face Face.y = Face.D) :=
begin
  exact cube_condition,
end

end face_opposite_y_is_D_l295_295182


namespace average_age_increase_l295_295923

variable (A B C : ℕ)

theorem average_age_increase (A : ℕ) (B : ℕ) (C : ℕ) (h1 : 21 < B) (h2 : 23 < C) (h3 : A + B + C > A + 21 + 23) :
  (B + C) / 2 > 22 := by
  sorry

end average_age_increase_l295_295923


namespace temperature_difference_l295_295203

theorem temperature_difference 
    (freezer_temp : ℤ) (room_temp : ℤ) (temperature_difference : ℤ) 
    (h1 : freezer_temp = -4) 
    (h2 : room_temp = 18) : 
    temperature_difference = room_temp - freezer_temp := 
by 
  sorry

end temperature_difference_l295_295203


namespace tan_pi_over_3_plus_cos_19_pi_over_6_l295_295347

theorem tan_pi_over_3_plus_cos_19_pi_over_6 :
  tan (Real.pi / 3) + cos (19 * Real.pi / 6) = sqrt 3 / 2 :=
by
  sorry

end tan_pi_over_3_plus_cos_19_pi_over_6_l295_295347


namespace problem_1_solution_problem_2_solution_problem_3_solution_l295_295992

noncomputable theory

-- Proof for Problem 1: Solving the system of equations
def system_of_equations_soln (x y : ℝ) : Prop :=
  (5 * x + 2 * y = 25) ∧ (3 * x + 4 * y = 15)

theorem problem_1_solution : system_of_equations_soln 5 0 :=
by sorry

-- Proof for Problem 2: Value of the expression
def calculate_expression : ℝ :=
  2 * (Real.sqrt 3 - 1) - abs (Real.sqrt 3 - 2) - Real.cbrt (-64)

theorem problem_2_solution : calculate_expression = 3 * Real.sqrt 3 :=
by sorry

-- Proof for Problem 3: Solving the equation
def equation_solution (x : ℝ) : Prop :=
  2 * (x - 1)^2 - 49 = 1

theorem problem_3_solution : equation_solution (-4) ∨ equation_solution 6 :=
by sorry

end problem_1_solution_problem_2_solution_problem_3_solution_l295_295992


namespace factorization_correct_l295_295980

theorem factorization_correct : ∀ y : ℝ, y^2 - 4*y + 4 = (y - 2)^2 := by
  intro y
  sorry

end factorization_correct_l295_295980


namespace sum_of_constants_l295_295647

variables (a b c : ℝ)

-- Conditions
def equation (y : ℝ) : ℝ := a * y^2 + b * y + c

def vertex_condition : Prop :=
  equation (-6) = 10

def point_condition : Prop :=
  equation (-4) = 8

-- Theorem
theorem sum_of_constants :
  vertex_condition a b c →
  point_condition a b c →
  a + b + c = -39 :=
by
  sorry

end sum_of_constants_l295_295647


namespace smallest_positive_solution_l295_295721

theorem smallest_positive_solution :
  ∃ (x : ℝ), 0 < x ∧ tan x + tan (4 * x) = sec (4 * x) ∧ x = π / 14 :=
by
  sorry

end smallest_positive_solution_l295_295721


namespace ratio_of_breadth_to_length_l295_295173

-- given conditions
def playground_area : ℕ := 3200
def playground_area_ratio : ℚ := 1 / 9
def breadth : ℕ := 480

-- statement to prove
theorem ratio_of_breadth_to_length :
  let total_area := playground_area * 9 in
  let length := total_area / breadth in
  breadth / length = 8 := by
  sorry

end ratio_of_breadth_to_length_l295_295173


namespace average_percentage_15_students_l295_295642

-- Define the average percentage of the 15 students
variable (x : ℝ)

-- Condition 1: Total percentage for the 15 students is 15 * x
def total_15_students : ℝ := 15 * x

-- Condition 2: Total percentage for the 10 students who averaged 88%
def total_10_students : ℝ := 10 * 88

-- Condition 3: Total percentage for all 25 students who averaged 79%
def total_all_students : ℝ := 25 * 79

-- Mathematical problem: Prove that x = 73 given the conditions.
theorem average_percentage_15_students (h : total_15_students x + total_10_students = total_all_students) : x = 73 := 
by
  sorry

end average_percentage_15_students_l295_295642


namespace area_bounded_by_arctan_y0_xsqrt3_l295_295683

noncomputable def area_under_curve : ℝ :=
  ∫ x in 0..(Real.sqrt 3), Real.arctan x

theorem area_bounded_by_arctan_y0_xsqrt3 :
  area_under_curve = (Real.pi / Real.sqrt 3) - Real.log 2 :=
by
  sorry

end area_bounded_by_arctan_y0_xsqrt3_l295_295683


namespace PR_PS_AF_l295_295091

variables {A B C D P S R Q F : Type}

-- Given conditions
-- ABCD is a parallelogram
axiom is_parallelogram (ABCD : A ↔ B ↔ C ↔D) : Prop

-- P is a point on diagonal AC
axiom point_on_diagonal (P : A → C) : Prop

-- PS is perpendicular to AB and PR is perpendicular to CD
axiom perp_PS_AB (PS : P → S) (AB : A ↔ B) : Prop
axiom perp_PR_CD (PR : P → R) (CD : C ↔ D) : Prop

-- AF is perpendicular to CD from point A
axiom perp_AF_CD (AF : A → F) (CD : C ↔ D) : Prop

-- PQ is perpendicular to AF from point P
axiom perp_PQ_AF (PQ : P → Q) (AF : A → F) : Prop

-- Conjecture to prove
theorem PR_PS_AF (PR PS AF : P → S → F) : PR + PS = AF := sorry

end PR_PS_AF_l295_295091


namespace find_coefficients_l295_295025

theorem find_coefficients (a b : ℚ) (h_a_nonzero : a ≠ 0)
  (h_prod : (3 * b - 2 * a = 0) ∧ (-2 * b + 3 = 0)) : 
  a = 9 / 4 ∧ b = 3 / 2 :=
by
  sorry

end find_coefficients_l295_295025


namespace angle_P_in_regular_octagon_l295_295549

noncomputable theory

def is_regular_octagon (sides : list ℝ) (angles : list ℝ) : Prop :=
  sides.length = 8 ∧ angles.length = 8 ∧ 
  ∀ (n : ℕ), n < 8 → sides.nth_le n (by linarith) = sides.nth_le 0 (by linarith) ∧
  ∀ (n : ℕ), n < 8 → angles.nth_le n (by linarith) = angles.nth_le 0 (by linarith) ∧
  angles.nth_le 0 (by linarith) = 135

theorem angle_P_in_regular_octagon (sides : list ℝ) (angles : list ℝ) (P : EuclideanGeometry.Point) 
  (A B G H : EuclideanGeometry.Point) :
  is_regular_octagon sides angles →
  EuclideanGeometry.collinear {A, G, P} →
  EuclideanGeometry.collinear {B, H, P} →
  EuclideanGeometry.angle (A - G) (A - P) = 45 →
  EuclideanGeometry.angle (H - A) (H - P) = 45 →
  EuclideanGeometry.angle (G - P) (H - P) = 90 :=
by
  intros h_regular_octagon h_collinear_AGP h_collinear_BHP h_angle_AGP h_angle_HAP
  -- The proof is omitted
  sorry

end angle_P_in_regular_octagon_l295_295549


namespace analytical_expression_minimum_c_range_of_m_l295_295006
noncomputable theory

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Q1: Proving the analytical expression of the function
theorem analytical_expression :
  (∀ (a b : ℝ), f(1) = a + b - 3 → f'(1) = 3*a + 2*b - 3 → f = λ x, x^3 - 3*x) := sorry 

-- Q2: Minimum value of c such that |f(x1) - f(x2)| ≤ c for x1, x2 in [-2, 2]
theorem minimum_c (c : ℝ) :
  (∀ x1 x2 ∈ Icc (-2 : ℝ) 2, abs (f x1 - f x2) ≤ c) → c = 4 := sorry
  
-- Q3: Range of m for three tangents passing through M(2, m)
theorem range_of_m (m : ℝ) :
  (∃ m : ℝ, ∃ x0 : ℝ, f x = x0^3 - 3*x ∧ 3*x0^2 - 3 = m ∧ -6 < m ∧ m < 2) := sorry

end analytical_expression_minimum_c_range_of_m_l295_295006


namespace division_pairs_l295_295338

theorem division_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (ab^2 + b + 7) ∣ (a^2 * b + a + b) →
  (∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k) ∨ (a, b) = (11, 1) ∨ (a, b) = (49, 1) :=
sorry

end division_pairs_l295_295338


namespace tangent_line_at_a_eq_1_range_of_a_with_two_zeros_l295_295031

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * real.exp (a * x) + real.log x - real.exp 1

noncomputable def g (x : ℝ) : ℝ := real.log x + 1 / x - real.exp 1

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a - g x

theorem tangent_line_at_a_eq_1 :
  let fx := f 1 1 in
  fx = 0 →
  ∃ k : ℝ, k = 2 * real.exp 1 + 1 ∧ ∀ x, f x 1 = k * (x - 1) + fx :=
sorry

theorem range_of_a_with_two_zeros :
  (∃ x : ℝ, 0 < x ∧ h x a = 0) ∧ (∃ y : ℝ, 0 < y ∧ y ≠ x ∧ h y a = 0) ↔ -2 / real.exp 1 < a ∧ a < 0 :=
sorry

end tangent_line_at_a_eq_1_range_of_a_with_two_zeros_l295_295031


namespace find_A_l295_295623

theorem find_A (A : ℕ) (h : 10 * A + 2 - 23 = 549) : A = 5 :=
by sorry

end find_A_l295_295623


namespace straight_line_cannot_intersect_all_segments_l295_295687

/-- A broken line in the plane with 11 segments -/
structure BrokenLine :=
(segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
(closed_chain : ∀ i : Fin 11, i.val < 10 → (segments ⟨i.val + 1, sorry⟩).fst = (segments i).snd)

/-- A straight line that doesn't contain the vertices of the broken line -/
structure StraightLine :=
(is_not_vertex : (ℝ × ℝ) → Prop)

/-- The main theorem stating the impossibility of a straight line intersecting all segments -/
theorem straight_line_cannot_intersect_all_segments (line : StraightLine) (brokenLine: BrokenLine) :
  ∃ i : Fin 11, ¬∃ t : ℝ, ∃ x y : ℝ, 
    brokenLine.segments i = ((x, y), (x + t, y + t)) ∧ 
    ¬line.is_not_vertex (x, y) ∧ 
    ¬line.is_not_vertex (x + t, y + t) :=
sorry

end straight_line_cannot_intersect_all_segments_l295_295687


namespace find_a_l295_295178

theorem find_a (a : ℝ) (p : ℕ → ℝ) (h : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 → p k = a * (1 / 2) ^ k)
  (prob_sum : a * (1 / 2 + (1 / 2) ^ 2 + (1 / 2) ^ 3) = 1) : a = 8 / 7 :=
sorry

end find_a_l295_295178


namespace log3_interval_sum_l295_295319

theorem log3_interval_sum (a b : ℤ) (h₁ : 3^4 = 81) (h₂ : 3^5 = 243)
  (h₃ : 81 < 200) (h₄ : 200 < 243) (h₅ : 4 < real.log 200 / real.log 3) (h₆ : real.log 200 / real.log 3 < 5) :
  a = 4 ∧ b = 5 → a + b = 9 := 
by {
  intro h,
  cases h with ha hb,
  rw [ha, hb],
  exact nat.add_eq_add ha hb,
  sorry
}

end log3_interval_sum_l295_295319


namespace probability_male_female_ratio_l295_295359

theorem probability_male_female_ratio :
  let total_possibilities := Nat.choose 9 5
  let specific_scenarios := Nat.choose 5 2 * Nat.choose 4 3 + Nat.choose 5 3 * Nat.choose 4 2
  let probability := specific_scenarios / (total_possibilities : ℚ)
  probability = 50 / 63 :=
by 
  sorry

end probability_male_female_ratio_l295_295359


namespace positive_expressions_l295_295903

variables (U V W X Y : ℝ)

theorem positive_expressions
  (hU : U ≈ -2)
  (hV : V ≈ -1)
  (hW : W ≈ 2)
  (hX : X ≈ 1)
  (hY : Y ≈ -1) :
  (U * V > 0) ∧ (X / V * U > 0) ∧ (W / (U * V) > 0) ∧ ((X - Y) / W > 0) :=
begin
  sorry
end

end positive_expressions_l295_295903


namespace train_speed_before_acceleration_l295_295262

variable (v s : ℝ) -- defining the variables v and s as real numbers

theorem train_speed_before_acceleration :
  ∃ x : ℝ, ∀ v s : ℝ, v ≠ 0 → s ≠ 0 → x = s * v / 50 :=
by
  intro v s 
  assume hv : v ≠ 0
  assume hs : s ≠ 0
  use (s * v / 50)
  sorry -- proof is omitted

end train_speed_before_acceleration_l295_295262


namespace max_sum_of_edge_products_l295_295900

-- Define the set of numbers and their squares
def num_squares := { n : ℕ | n ∈ (finset.range 8).image (λ x, (x + 1) ^ 2) }

-- Define the vertices of a cube, where each vertex is assigned one of the squared numbers
def cube_vertices := (fin 8) → ℕ

-- Define the condition that each vertex should contain a unique number from num_squares.
axiom vertex_condition {v : cube_vertices} : ∀ i j : fin 8, i ≠ j → v i ≠ v j ∧ v i ∈ num_squares

-- Prove that the maximum possible sum of all edge products is 9420.
theorem max_sum_of_edge_products : ∃ v : cube_vertices, 
  (∑ i in (finset.univ : finset (fin 8)), ∑ j in (finset.filter (λ k, k ≠ i) finset.univ), (v i) * (v j)) / 2 = 9420 :=
begin
  sorry
end

end max_sum_of_edge_products_l295_295900


namespace complex_inequality_l295_295753

namespace ComplexInequality
open Complex

theorem complex_inequality (x y z : ℂ) (h : ∥x∥^2 + ∥y∥^2 + ∥z∥^2 = 1) :
  ∥x^3 + y^3 + z^3 - 3 * x * y * z∥ ≤ 1 :=
sorry

end ComplexInequality

end complex_inequality_l295_295753


namespace number_of_divisors_of_M_l295_295702

-- Define the number M
def M : ℕ := 2^6 * 3^4 * 5^2 * 7^2 * 11^1

-- Statement of the proof problem
theorem number_of_divisors_of_M : nat.divisors_count M = 630 :=
by
  sorry

end number_of_divisors_of_M_l295_295702


namespace range_of_a_for_isosceles_triangle_l295_295752

theorem range_of_a_for_isosceles_triangle :
  ∀ a : ℝ, 0 < a ∧ a ≤ 9 ↔ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 6*x1 + a = 0 ∧ x2^2 - 6*x2 + a = 0 ∧ 
  ∃ triangle : ℝ × ℝ × ℝ, 
    (triangle.1 = x1 ∨ triangle.2 = x1 ∨ triangle.3 = x1) ∧ 
    (triangle.1 = x2 ∨ triangle.2 = x2 ∨ triangle.3 = x2) ∧ 
    (triangle.1 = triangle.2 ∨ triangle.2 = triangle.3 ∨ triangle.1 = triangle.3) ∧ 
    (triangle.1 + triangle.2 > triangle.3 ∧ triangle.2 + triangle.3 > triangle.1 ∧ triangle.1 + triangle.3 > triangle.2)) :=
begin
  sorry
end

end range_of_a_for_isosceles_triangle_l295_295752


namespace total_expenditure_proof_l295_295459

-- Define the conditions
def pricePerGallonNC := 2.00
def gallonsBoughtNC := 10
def priceIncrementVA := 1.00
def gallonsBoughtVA := 10

-- Define the function to calculate total expenditure
def totalExpenditure (priceNC : ℝ) (gallonsNC : ℕ) (incrementVA : ℝ) (gallonsVA : ℕ) : ℝ :=
  (priceNC * gallonsNC) + ((priceNC + incrementVA) * gallonsVA)

-- Theorem statement
theorem total_expenditure_proof : 
  totalExpenditure pricePerGallonNC gallonsBoughtNC priceIncrementVA gallonsBoughtVA = 50.00 :=
by
  -- Proof will be provided here
  sorry

end total_expenditure_proof_l295_295459


namespace find_f8_solution_l295_295874

def monotonic_function (f : ℝ → ℝ) :=
  ∀ x y : ℝ, (0 < x → 0 < y → x < y → f(x) ≤ f(y)) ∨ (0 < x → 0 < y → x < y → f(x) ≥ f(y))

theorem find_f8_solution (f : ℝ → ℝ) (h_monotonic : monotonic_function f) (h_cond1 : ∀ x : ℝ, 0 < x → f(x) > -4 / x) (h_cond2 : ∀ x : ℝ, 0 < x → f(f(x) + 4 / x) = 3) :
  f 8 = 7 / 2 :=
sorry

end find_f8_solution_l295_295874


namespace bob_rope_sections_l295_295301

/-- Given a 50-foot rope, where 1/5 is used for art, half of the remaining is given to a friend,
     and the rest is cut into 2-foot sections, prove that the number of sections Bob gets is 10. -/
theorem bob_rope_sections :
  ∀ (total_rope art_fraction remaining_fraction section_length : ℕ),
    total_rope = 50 →
    art_fraction = 5 →
    remaining_fraction = 2 →
    section_length = 2 →
    (total_rope / art_fraction / remaining_fraction / section_length) = 10 :=
by
  intros total_rope art_fraction remaining_fraction section_length
  assume h_total_rope h_art_fraction h_remaining_fraction h_section_length
  rw [h_total_rope, h_art_fraction, h_remaining_fraction, h_section_length]
  have h1 : 50 / 5 = 10 := by norm_num
  have h2 : (50 - 10) / 2 = 20 := by norm_num
  have h3 : 20 / 2 = 10 := by norm_num
  exact h3

end bob_rope_sections_l295_295301


namespace categorization_using_structure_chart_l295_295589

open Complex

-- Definitions for the sets and their subdivisions
def set_of_complex_numbers := ℂ
def set_of_real_numbers := ℝ
def set_of_rational_numbers := {x : ℝ // ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}
def set_of_irrational_numbers := {x : ℝ // ∀ a b : ℤ, b ≠ 0 → x ≠ a / b}
def set_of_imaginary_numbers := {x : ℂ // x.im ≠ 0}
def set_of_pure_imaginary_numbers := {x : ℂ // x.re = 0 ∧ x.im ≠ 0}
def set_of_non_pure_imaginary_numbers := {x : ℂ // x.re ≠ 0 ∧ x.im ≠ 0}

-- The theorem that proves the appropriate description is a structure chart
theorem categorization_using_structure_chart :
  appropriate_description set_of_complex_numbers set_of_real_numbers set_of_rational_numbers set_of_irrational_numbers set_of_imaginary_numbers set_of_pure_imaginary_numbers set_of_non_pure_imaginary_numbers = "structure_chart" := 
  sorry

end categorization_using_structure_chart_l295_295589


namespace starting_positions_count_l295_295864

theorem starting_positions_count :
  let P_0 : ℝ → Prop := λ y_0, ∃ (seq : ℕ → ℝ), 
    seq 0 = y_0 ∧ 
    ∀ n, seq (n + 1) = seq n - 2 * real.sqrt (seq n^2 - 1) ∧ 
    seq 2008 = y_0
  in
  (set.univ.filter P_0).nonempty.count = 2^2008 - 2 :=
sorry

end starting_positions_count_l295_295864


namespace num_sampled_students_in_interval_l295_295469

theorem num_sampled_students_in_interval (total_students : ℕ) (sample_size : ℕ) (lower_bound : ℕ) (upper_bound : ℕ) 
  (sampling_interval : ℕ) (elements_in_interval : ℕ) : 
  total_students = 900 →
  sample_size = 45 →
  lower_bound = 481 →
  upper_bound = 720 →
  sampling_interval = 20 →
  elements_in_interval = (upper_bound - lower_bound + 1) →
  (sampled_students_in_interval : ℕ) :=
  by
  intro h1 h2 h3 h4 h5 h6,
  let target := ⌊ elements_in_interval / sampling_interval ⌋,
  have : target = 12, sorry

-- number of sampled students whose numbers fall within the interval [481, 720]
-- sorry is a placeholder for the actual proof.

end num_sampled_students_in_interval_l295_295469


namespace parallel_lines_l295_295414

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l295_295414


namespace fraction_decomposition_l295_295543

theorem fraction_decomposition (a b : ℕ) (h : a < b) :
  ∃ (n : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → n i ≠ n j) ∧ (∃ m : ℕ, a / b = ∑ i in Finset.range m, 1 / n i) :=
by sorry

end fraction_decomposition_l295_295543


namespace average_marks_l295_295636

theorem average_marks (avg1 : ℝ) (n1 : ℕ) (avg2 : ℝ) (n2 : ℕ) : 
  avg1 = 40 → n1 = 24 → avg2 = 60 → n2 = 50 → (avg1 * n1 + avg2 * n2) / (n1 + n2) = 53.51 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have htotal1 : 24 * 40 = 960 := by norm_num
  have htotal2 : 50 * 60 = 3000 := by norm_num
  have htotalmarks : 960 + 3000 = 3960 := by norm_num
  have htotalstudents : 24 + 50 = 74 := by norm_num
  rw [htotal1, htotal2, htotalmarks, htotalstudents]
  norm_num
  sorry

end average_marks_l295_295636


namespace gcd_lcm_ratio_l295_295449

theorem gcd_lcm_ratio (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 200) (h2 : 2 * k = A) (h3 : 5 * k = B) : Nat.gcd A B = k :=
by
  sorry

end gcd_lcm_ratio_l295_295449


namespace mary_stickers_left_l295_295137

def initial_stickers : ℕ := 50
def stickers_per_friend : ℕ := 4
def number_of_friends : ℕ := 5
def total_students_including_mary : ℕ := 17
def stickers_per_other_student : ℕ := 2

theorem mary_stickers_left :
  let friends_stickers := stickers_per_friend * number_of_friends
  let other_students := total_students_including_mary - 1 - number_of_friends
  let other_students_stickers := stickers_per_other_student * other_students
  let total_given_away := friends_stickers + other_students_stickers
  initial_stickers - total_given_away = 8 :=
by
  sorry

end mary_stickers_left_l295_295137


namespace collinear_X_Y_Mc_l295_295582

variables (A B C M M_a M_b M_c X Y : Point)
variables (ω_a ω_b : Circle)

-- Conditions
axiom triangle_medians : MediansIntersectAt (△ A B C) M
axiom circle_omega_a : CircleThroughMidpointAndTangent ω_a (segment A M) (side B C) M_a
axiom circle_omega_b : CircleThroughMidpointAndTangent ω_b (segment B M) (side C A) M_b
axiom points_XY_are_intersections : Intersections ω_a ω_b X Y

-- Proof statement
theorem collinear_X_Y_Mc :
  Collinear X Y M_c :=
sorry

end collinear_X_Y_Mc_l295_295582


namespace f_even_implies_g_odd_g_has_two_distinct_roots_implies_f_monotonicity_g_x_equals_x_f_zero_roots_ordering_l295_295037

-- Definitions for f(x) and g(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1
def g (a b : ℝ) (x : ℝ) : ℝ := (b * x - 1) / (a^2 * x + 2 * b)

-- (1) f is even function and g(x) parity
theorem f_even_implies_g_odd (a b : ℝ) (h : ∀ x : ℝ, f a b x = f a b (-x)) : b = 0 → ∀ x : ℝ, g a b (-x) = -g a b x := by
  sorry

-- (2) g(x) = x with two distinct real roots and monotonicity on (-1, 1)
theorem g_has_two_distinct_roots_implies_f_monotonicity (a b c : ℝ) (h1 : ∀ x : ℝ, g a b x = x)
    (h2 : ∀ z : ℝ, (a^2 * z^2 + b * z + c) = 0)
    (h3 : a > 0) : 
    (f a b).MonotonicOn_Ioo (-1) 1 :=
by 
  sorry

-- (3) g(x) = x and f(x) = 0 roots ordering condition on a
theorem g_x_equals_x_f_zero_roots_ordering (a b : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
    (h1 : ∀ x : ℝ, g a b x = x)
    (h2 : f a b x₃ = 0 ∧ f a b x₄ = 0)
    (h3 : x₃ < x₁ ∧ x₁ < x₂ ∧ x₂ < x₄)
    (h4 : a > 0) :
    1 < a := 
by 
  sorry

end f_even_implies_g_odd_g_has_two_distinct_roots_implies_f_monotonicity_g_x_equals_x_f_zero_roots_ordering_l295_295037


namespace find_F_l295_295057

theorem find_F (C F : ℝ) (h1 : C = (4 / 7) * (F - 40)) (h2 : C = 35) : F = 101.25 :=
  sorry

end find_F_l295_295057


namespace range_of_a_l295_295994

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

def log_base (b x : ℝ) : ℝ := log x / log b

theorem range_of_a (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing (λ x, log_base (a - 1) x) :=
by
  sorry

end range_of_a_l295_295994


namespace inequality_proof_l295_295090

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / (x + y)^2 :=
sorry

end inequality_proof_l295_295090


namespace find_common_ratio_sum_first_n_terms_l295_295097

variable (a : ℕ → ℝ) (q : ℝ) 

-- Condition: {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: a₁ is the arithmetic mean of a₂ and a₃
def arithmetic_mean (a : ℕ → ℝ) :=
  a 1 = (a 2 + a 3) / 2

-- Proposition 1: Find the common ratio q
theorem find_common_ratio (h1 : is_geometric_sequence a q) (h2 : q ≠ 1) (h3 : arithmetic_mean a) : 
  q = -2 :=
by sorry

-- Proposition 2: Find the sum of the first n terms of the sequence {n * a_n}, given a₁ = 1
def sequence_n_times_a (a : ℕ → ℝ) :=
  λ n, n * a n

def sum_of_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) f

def geom_sum (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n else (1 - q ^ n) / (1 - q)

theorem sum_first_n_terms (h1 : is_geometric_sequence a q) (h2 : q = -2) (h3 : a 1 = 1) (n : ℕ) :
  sum_of_sequence (sequence_n_times_a a) n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
by sorry

end find_common_ratio_sum_first_n_terms_l295_295097


namespace cubic_no_negative_roots_l295_295316

noncomputable def cubic_eq (x : ℝ) : ℝ := x^3 - 9 * x^2 + 23 * x - 15

theorem cubic_no_negative_roots {x : ℝ} : cubic_eq x = 0 → 0 ≤ x := sorry

end cubic_no_negative_roots_l295_295316


namespace min_distance_racetracks_l295_295546

theorem min_distance_racetracks : 
  ∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (((B.1 - 1) ^ 2) / 16 + (B.2 ^ 2) / 4 = 1) → 
  dist A B ≥ (Real.sqrt 33 - 3) / 3 := by
  sorry

end min_distance_racetracks_l295_295546


namespace find_number_l295_295643

-- Define the condition
def is_number (x : ℝ) : Prop :=
  0.15 * x = 0.25 * 16 + 2

-- The theorem statement: proving the number is 40
theorem find_number (x : ℝ) (h : is_number x) : x = 40 :=
by
  -- We would insert the proof steps here
  sorry

end find_number_l295_295643


namespace Problem_8_451_l295_295241

theorem Problem_8_451 (x : ℝ) (h : |cos (2 * x)| ≠ 1) : 
  (1 - cos (2 * x) + cos (2 * x)^2 - cos (2 * x)^3 + ...) / 
  (1 + cos (2 * x) + cos (2 * x)^2 + cos (2 * x)^3 + ...) = (1 / 3) * (tan x)^4 ↔ 
  cos (2 * x) = -1 / 2 :=
sorry

end Problem_8_451_l295_295241


namespace arithmetic_sequence_common_difference_l295_295947

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n * a₁ + (n * (n - 1) / 2) * d

/-- Condition for the sum of the first 5 terms -/
def S5 (a₁ d : ℚ) : Prop := S 5 a₁ d = 6

/-- Condition for the second term of the sequence -/
def a2 (a₁ d : ℚ) : Prop := a₁ + d = 1

/-- The main theorem to be proved -/
theorem arithmetic_sequence_common_difference (a₁ d : ℚ) (hS5 : S5 a₁ d) (ha2 : a2 a₁ d) : d = 1 / 5 :=
sorry

end arithmetic_sequence_common_difference_l295_295947


namespace equation_of_BC_l295_295764

def point := (ℝ × ℝ)
def line_eqn (m b : ℝ) (p : point) : Prop := p.snd = m * p.fst + b

theorem equation_of_BC
  (A : point) (A_eq : A = (3, -1))
  (B : point) (C : point)
  (B_eqn : ∀ (x : ℝ), (x, B.snd) ∈ B :: x = 0)
  (C_eqn : ∀ (x : ℝ), (x, C.snd) ∈ C :: x = y)
  (BC_eq : ∀ (BC_eqn : line_eqn 2 5)) :
  BC_eq :=
sorry

end equation_of_BC_l295_295764


namespace ineq_sum_squares_l295_295049

variable (n : ℕ) 
variable (x : Fin n → ℝ)

theorem ineq_sum_squares (hx: ∀ k : Fin n, x k > 0) :
  (∑ k, (x k / (1 + ∑ i in Finset.range (k.1+1), (x ⟨i, sorry⟩)^2))^2) 
  ≤ (∑ k, (x k)^2) / (1 + ∑ k, (x k)^2) :=
by
  sorry

end ineq_sum_squares_l295_295049


namespace total_cleaning_time_is_100_l295_295893

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end total_cleaning_time_is_100_l295_295893


namespace ratio_of_areas_l295_295446

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  -- The problem is to prove the ratio of the areas is 4/9
  sorry

end ratio_of_areas_l295_295446


namespace iced_coffee_cost_is_2_l295_295136

def weekly_latte_cost := 4 * 5
def annual_latte_cost := weekly_latte_cost * 52
def weekly_iced_coffee_cost (x : ℝ) := x * 3
def annual_iced_coffee_cost (x : ℝ) := weekly_iced_coffee_cost x * 52
def total_annual_coffee_cost (x : ℝ) := annual_latte_cost + annual_iced_coffee_cost x
def reduced_spending_goal (x : ℝ) := 0.75 * total_annual_coffee_cost x
def saved_amount := 338

theorem iced_coffee_cost_is_2 :
  ∃ x : ℝ, (total_annual_coffee_cost x - reduced_spending_goal x = saved_amount) → x = 2 :=
by
  sorry

end iced_coffee_cost_is_2_l295_295136


namespace a_eq_b_pow_n_l295_295640

variables (a b n : ℕ)
variable (h : ∀ (k : ℕ), k ≠ b → b - k ∣ a - k^n)

theorem a_eq_b_pow_n : a = b^n := 
by
  sorry

end a_eq_b_pow_n_l295_295640


namespace all_identical_digits_count_two_different_digits_count_three_different_digits_count_four_different_digits_count_five_different_digits_count_l295_295428

-- Definitions and conditions set
def digit_set := Fin 10 -- 0 through 9
def non_zero_digit_set := {d : digit_set // d ≠ 0}
def is_valid_five_digit_number (ns : List non_zero_digit_set) : Prop := ns.length = 5

-- Theorems for each proof problem
theorem all_identical_digits_count : (∃ f : digit_set, ∀ i : Fin 5, digit_set i = f) → nat :=
  begin
  sorry -- proof not required
  end

theorem two_different_digits_count : (∃ d1 d2 : digit_set, d1 ≠ d2 ∧ ∀ i : Fin 5, (digit_set i = d1 ∨ digit_set i = d2)) → nat :=
  begin
  sorry -- proof not required
  end

theorem three_different_digits_count : (∃ d1 d2 d3 : digit_set, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ ∀ i : Fin 5, (digit_set i = d1 ∨ digit_set i = d2 ∨ digit_set i = d3)) → nat :=
  begin
  sorry -- proof not required
  end

theorem four_different_digits_count : (∃ d1 d2 d3 d4 : digit_set, distinct [d1, d2, d3, d4] ∧ ∀ i : Fin 5, (digit_set i = d1 ∨ digit_set i = d2 ∨ digit_set i = d3 ∨ digit_set i = d4)) → nat :=
  begin
  sorry -- proof not required
  end

theorem five_different_digits_count : (∃ d1 d2 d3 d4 d5 : digit_set, distinct [d1, d2, d3, d4, d5] ∧ ∀ i : Fin 5, (digit_set i = d1 ∨ digit_set i = d2 ∨ digit_set i = d3 ∨ digit_set i = d4 ∨ digit_set i = d5)) → nat :=
  begin
  sorry -- proof not required
  end

end all_identical_digits_count_two_different_digits_count_three_different_digits_count_four_different_digits_count_five_different_digits_count_l295_295428


namespace complement_intersection_l295_295519

theorem complement_intersection (A B U : Set ℕ) (hA : A = {4, 5, 7}) (hB : B = {3, 4, 7, 8}) (hU : U = A ∪ B) :
  U \ (A ∩ B) = {3, 5, 8} :=
by
  sorry

end complement_intersection_l295_295519


namespace math_sign_white_area_l295_295205

theorem math_sign_white_area :
  ∀ (length width : ℕ) (black_area_M black_area_A black_area_T black_area_H : ℕ),
  length = 6 →
  width = 18 →
  black_area_M = 16 →
  black_area_A = 10 →
  black_area_T = 10 →
  black_area_H = 16 →
  let total_area := length * width in
  let black_area := black_area_M + black_area_A + black_area_T + black_area_H in
  let white_area := total_area - black_area in
  white_area = 56 :=
by
  intros length width black_area_M black_area_A black_area_T black_area_H
  assume h1 : length = 6
  assume h2 : width = 18
  assume h3 : black_area_M = 16
  assume h4 : black_area_A = 10
  assume h5 : black_area_T = 10
  assume h6 : black_area_H = 16
  let total_area := length * width
  let black_area := black_area_M + black_area_A + black_area_T + black_area_H
  let white_area := total_area - black_area
  sorry

end math_sign_white_area_l295_295205


namespace quadratic_inequality_solution_l295_295339

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 - 8 * x + c > 0) ↔ (0 < c ∧ c < 16) := 
sorry

end quadratic_inequality_solution_l295_295339


namespace percentage_of_b_l295_295645

variable (a b c p : ℝ)

-- Conditions
def condition1 : Prop := 0.02 * a = 8
def condition2 : Prop := c = b / a
def condition3 : Prop := p * b = 2

-- Theorem statement
theorem percentage_of_b (h1 : condition1 a)
                        (h2 : condition2 b a c)
                        (h3 : condition3 p b) :
  p = 0.005 := sorry

end percentage_of_b_l295_295645


namespace partitioning_of_seven_hexagons_l295_295695

theorem partitioning_of_seven_hexagons :
  ∀ (num_hexagons : ℕ) (partitioning : ℕ → ℕ), 
    num_hexagons = 7 
    ∧ partitioning num_hexagons = (2 ^ num_hexagons) := 
begin
  intro num_hexagons,
  intro partitioning,
  split,
  { sorry },  -- Here we'd show num_hexagons = 7 is given
  { sorry }   -- Here we'd show partitioning 7 = 2^7
end

end partitioning_of_seven_hexagons_l295_295695


namespace polynomial_identity_eq_l295_295431

theorem polynomial_identity_eq (a a1 a2 a3 a4 : ℤ) :
  (λ x : ℤ, (x - 1)^4) = (λ x, a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) →
  (a + a2 + a4 = 8) :=
by
  sorry

end polynomial_identity_eq_l295_295431


namespace hyperbola_eccentricity_l295_295391

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : 2 * a = 16) (h₂ : 2 * b = 12) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  (c / a) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l295_295391


namespace ice_cream_to_afford_games_l295_295153

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l295_295153


namespace total_cleaning_time_l295_295889

theorem total_cleaning_time (time_outside : ℕ) (fraction_inside : ℚ) (time_inside : ℕ) (total_time : ℕ) :
  time_outside = 80 →
  fraction_inside = 1 / 4 →
  time_inside = fraction_inside * time_outside →
  total_time = time_outside + time_inside →
  total_time = 100 :=
by
  intros hto hfi htinside httotal
  rw [hto, hfi] at htinside
  norm_num at htinside
  rw [hto, htinside] at httotal
  norm_num at httotal
  exact httotal

end total_cleaning_time_l295_295889


namespace range_of_a_l295_295116

noncomputable def f (a x : ℝ) :=
  if x < a then cos(2 * π * x - 2 * π * a)
  else x^2 - 2 * (a + 1) * x + a^2 + 5

theorem range_of_a (a : ℝ) (h : ∃ x, f a x = 0 ∧ 0 < x) :
  (5 / 2 < a ∧ a ≤ 11 / 4) ∨ (2 < a ∧ a ≤ 9 / 4) :=
sorry

end range_of_a_l295_295116


namespace general_term_a_n_sum_first_n_terms_T_n_l295_295126

noncomputable def sequence_sum_relation (S_n a_n : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → S_n n = -a_n n + 1 - (1 / 2^n) 

theorem general_term_a_n (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (sequence_sum_relation S_n a_n) → (a_n = λ n, n / 2^(n + 1)) :=
by
  intro h
  funext n
  sorry

theorem sum_first_n_terms_T_n (S_n : ℕ → ℝ) (T_n : ℕ → ℝ) :
  (sequence_sum_relation S_n (λ n, n / 2^(n + 1))) → (T_n = λ n, n - 2 + (n + 4) / 2^(n + 1)) :=
by
  intro h
  funext n
  sorry

end general_term_a_n_sum_first_n_terms_T_n_l295_295126


namespace probability_no_adjacent_stand_10_people_l295_295559

theorem probability_no_adjacent_stand_10_people :
  let b : Nat -> Nat
    | 2 => 3
    | 3 => 4
    | n => b (n - 1) + b (n - 2),
  probability_no_adjacent_stand (n : Nat) (h : n = 10) : 
    (b n) / (2^10) = 123 / 1024 := 
by
  sorry

end probability_no_adjacent_stand_10_people_l295_295559


namespace box_width_l295_295257

theorem box_width (W S : ℕ) (h1 : 30 * W * 12 = 80 * S^3) (h2 : S ∣ 30 ∧ S ∣ 12) : W = 48 :=
by
  sorry

end box_width_l295_295257


namespace hands_opposite_22_times_in_day_l295_295632

def clock_hands_opposite_in_day : ℕ := 22

def minute_hand_speed := 12
def opposite_line_minutes := 30

theorem hands_opposite_22_times_in_day (minute_hand_speed: ℕ) (opposite_line_minutes : ℕ) : 
  minute_hand_speed = 12 →
  opposite_line_minutes = 30 →
  clock_hands_opposite_in_day = 22 :=
by
  intros h1 h2
  sorry

end hands_opposite_22_times_in_day_l295_295632


namespace cats_distribution_probabilites_l295_295209

noncomputable def prob_X_k (n k : ℕ) := (Nat.choose n k) * (Nat.choose (n - k) (k - 1))

theorem cats_distribution_probabilites :
  let total_arrangements := Nat.choose 10 3 in
  let prob := λ k, prob_X_k 4 k * prob_X_k 6 (k - 1) / total_arrangements in
    (prob 1 = 1 / 30) ∧
    (prob 2 = 9 / 30) ∧
    (prob 3 = 15 / 30) ∧
    (prob 4 = 5 / 30) := by
  sorry

end cats_distribution_probabilites_l295_295209


namespace maximum_guarded_apples_l295_295133

theorem maximum_guarded_apples (a1 a2 a3 a4 a5 a6 : ℤ) (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6) :
  ∃ (x1 x2 x3 x4 : ℤ), 
  let 
    x1 := (a1 + a2 - a3) / 2,
    x2 := (a1 + a3 - a2) / 2,
    x3 := (a2 + a3 - a1) / 2,
    x4 := a4 - x3,
    sums := [x1 + x2, x1 + x3, x1 + x4, x2 + x3, x2 + x4, x3 + x4].toFinset
  in 
  let equal_sums := {x ∈ sums | x = a1 ∨ x = a2 ∨ x = a3 ∨ x = a4}.card,
      greater_sums := {x ∈ sums | x ≥ a5 ∨ x ≥ a6 ∨ x ∉ {a1, a2, a3, a4}}.card 
  in equal_sums * 3 + greater_sums = 14 :=
begin
  -- Proof skipped.
  sorry,
end

end maximum_guarded_apples_l295_295133


namespace find_box_length_l295_295649

theorem find_box_length (width depth : ℕ) (num_cubes : ℕ) (cube_side length : ℕ) 
  (h1 : width = 20)
  (h2 : depth = 10)
  (h3 : num_cubes = 56)
  (h4 : cube_side = 10)
  (h5 : length * width * depth = num_cubes * cube_side * cube_side * cube_side) :
  length = 280 :=
sorry

end find_box_length_l295_295649


namespace measure_distance_l295_295377

-- Assume the square sheet has side length 1
def square_sheet (side_length : ℝ) := square with each side of length 1 

-- Define the foldability condition where folds can be made along any segment with endpoints on the edges, leaving a crease mark
def foldable (fold : (ℝ × ℝ) → (ℝ × ℝ) → Prop) :=
∀ (a b : ℝ × ℝ), fold a b

-- Target Distance to be measured
def target_distance := 5 / 6

-- Prove that the sheet can be folded to measure the target distance
theorem measure_distance (side_length : ℝ) (fold : ((ℝ × ℝ) → (ℝ × ℝ) → Prop)) :
  side_length = 1 → foldable fold →
  ∃ (d : ℝ), d = target_distance := 
by
  sorry

end measure_distance_l295_295377


namespace competition_winner_l295_295729

-- Definitions of the singers
inductive Singer
| A | B | C | D

open Singer

-- Definition of what each singer said
def A_says (winner : Singer) : Prop := winner = B ∨ winner = C
def B_says (winner : Singer) : Prop := winner ≠ A ∧ winner ≠ C
def C_says (winner : Singer) : Prop := winner = D
def D_says (C_is_lying : Prop) : Prop := C_is_lying

-- Definition of the condition that exactly one statement is true
def one_truth (winner : Singer) : Prop :=
  (if A_says winner then 1 else 0) +
  (if B_says winner then 1 else 0) +
  (if C_says winner then 1 else 0) +
  (if D_says ¬C_says winner then 1 else 0) = 1

-- The proof statement
theorem competition_winner : ∃ winner : Singer, one_truth winner ∧ winner = D :=
by
  -- Here we state the proof is required
  sorry

end competition_winner_l295_295729


namespace sum_of_all_valid_n_l295_295855

def euler_totient (n : ℕ) : ℕ :=
  if h : n > 0 then
    (List.range n).filter (fun m => Nat.gcd m n = 1).length
  else 0

def is_valid_n (n : ℕ) : Prop := 
  n > 1 ∧ n < 100 ∧ n % (euler_totient n) = 0

def sum_valid_ns : ℕ := 
  (List.range 99).filter is_valid_n |>.sum

theorem sum_of_all_valid_n : sum_valid_ns = 396 := by
  sorry

end sum_of_all_valid_n_l295_295855


namespace general_formula_a_n_general_formula_b_n_sum_T_n_smallest_n_for_T_n_gt_2023_l295_295770

-- Definitions based on given conditions
def S : ℕ → ℕ
| n := n * n

def A : ℕ → ℕ

-- Define a sequence with initial term and recurrence based on given conditions
def a : ℕ → ℕ
| 1 := 1
| (n + 1) := S (n + 1) - S n

def b : ℕ → ℕ
| 1 := a 1
| (n + 1) := A n + 1

-- Ensure definitions for the sum of first n terms
def T (n : ℕ) : ℕ :=
∑ i in finset.range n, (a (i + 1)) * (b (i + 1))

-- Main theorem statements
theorem general_formula_a_n (n : ℕ) : a n = 2 * n - 1 := sorry
theorem general_formula_b_n (n : ℕ) : b n = 2 ^ (n - 1) := sorry
theorem sum_T_n (n : ℕ) : T n = (2 * n - 3) * 2^(n - 1) + 3 := sorry
theorem smallest_n_for_T_n_gt_2023 : ∃ n : ℕ, T n > 2023 ∧ ∀ m < n, T m ≤ 2023 :=
⟨8, by sorry, by sorry⟩

end general_formula_a_n_general_formula_b_n_sum_T_n_smallest_n_for_T_n_gt_2023_l295_295770


namespace g_three_eighths_l295_295186

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l295_295186


namespace positive_even_representation_l295_295845

theorem positive_even_representation (k : ℕ) (h : k > 0) :
  ∃ (a b : ℤ), (2 * k : ℤ) = a * b ∧ a + b = 0 := 
by
  sorry

end positive_even_representation_l295_295845


namespace simplify_and_evaluate_expr_l295_295550

/-- Simplification of the expression and evaluation -/
theorem simplify_and_evaluate_expr (x : ℝ) (hx : x ≠ -1) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  let expr := (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) in
  x = real.sqrt 3 → expr = (real.sqrt 3 - 1) / 2 :=
by
  intros
  -- Proof will be placed here.
  sorry

end simplify_and_evaluate_expr_l295_295550


namespace inscribed_sphere_radius_l295_295011

theorem inscribed_sphere_radius (e h r : ℝ) (tetrahedron_regular : e = 1) (tetrahedron_height : h = real.sqrt 2) : 
    r = real.sqrt 2 / 6 :=
    sorry

end inscribed_sphere_radius_l295_295011


namespace false_proposition_C_l295_295287

theorem false_proposition_C :
  (∀ x : ℝ, 2 * x - 1 > 0) →
  (∃ x : ℝ, real.log x < 1) →
  (∀ x : ℝ, x^2 > 0) →
  (∃ x : ℝ, real.tan x = 2) →
  false :=
by {
  intros h1 h2 h3 h4,
  have h5 : (0 : ℝ)^2 = 0 := by norm_num,
  have h6 : 0 ≠ 0 := h3 0,
  exact h6 h5,
}

end false_proposition_C_l295_295287


namespace difference_between_scores_l295_295998

noncomputable def total_runs := 60 * 46
noncomputable def total_runs_excl := 58 * 44
def highest_score := 179
def sum_of_scores := total_runs - total_runs_excl
def lowest_score := sum_of_scores - highest_score
def difference := highest_score - lowest_score

theorem difference_between_scores :
  difference = 150 :=
begin
  -- The proofs steps are omitted as per the instructions
  sorry
end

end difference_between_scores_l295_295998


namespace directed_line_points_relation_l295_295510

-- Definitions
variables {A B C D : Type} [linear_ordered_field A]
variables {a b c d : A}
variables (AC CB AD DB AB : A)
variables (h1 : AC = c) (h2 : CB = b - c) (h3 : AD = d) (h4 : DB = b - d)
variables (h_cond : AC / CB + AD / DB = 0)

-- Theorem statement
theorem directed_line_points_relation
  (h1 : ∀ x : A, AC = x) (h2 : ∀ y : A, CB = y - AC)
  (h3 : ∀ z : A, AD = z) (h4 : ∀ w : A, DB = w - AD)
  (h_cond : AC / CB + AD / DB = 0) :
  1 / AC + 1 / AD = 2 / AB :=
sorry

end directed_line_points_relation_l295_295510


namespace circle_segment_area_l295_295221

theorem circle_segment_area :
  let r := 10
  let d := 12
  let theta := 2 * real.arccos ((r - d/2) / r)
  let sector_area := (theta / (2 * real.pi)) * real.pi * r^2
  let triangle_area := d / 2 * real.sqrt(r^2 - (r - d/2)^2)
  (2 * sector_area - 2 * triangle_area) = 20 * 12 / 25 * real.pi - 36 :=
by
  sorry

end circle_segment_area_l295_295221


namespace pattern_B_cannot_form_tetrahedron_l295_295236

-- Definitions of patterns (A), (B), (C), (D) could be understood as different configurations of squares
-- considering their ability to form a regular tetrahedron.

inductive Pattern
| A
| B
| C
| D

-- Definition of regular tetrahedron properties
structure Tetrahedron :=
(faces : Finset (Finset ℕ))
(edges : Finset (ℕ × ℕ))
(vertices : Finset ℕ)
(valid : 
  faces.card = 4 ∧ 
  edges.card = 6 ∧ 
  vertices.card = 4 ∧
  ∀ v ∈ vertices, (∃ e1 e2 e3 ∈ edges, v ∈ [e1.1, e1.2, e2.1, e2.2, e3.1, e3.2]) ∧
  ∀ face ∈ faces, (∃ v1 v2 v3 ∈ vertices, face = {v1, v2, v3})
)

-- Functions to determine foldability into a regular tetrahedron
def can_form_tetrahedron (p : Pattern) : Prop := sorry

-- Equivalent proof that pattern B cannot form a tetrahedron
theorem pattern_B_cannot_form_tetrahedron : ¬ can_form_tetrahedron Pattern.B := 
sorry

end pattern_B_cannot_form_tetrahedron_l295_295236


namespace no_such_arrangement_l295_295492

open Matrix

-- Define the dimensions of the table
def n := 2002
def N := n * n

-- Define the property of a matrix satisfying the triplet product condition
def satisfies_triplet_condition (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i j : Fin n, 
  ∃ a b c : Fin n, (a ≠ i ∨ b ≠ j ∨ c ≠ j) ∧ (M i j = M a j * M b j ∨ M i j = M i c * M i b)

-- Define the proof problem statement
theorem no_such_arrangement :
  ¬ (∃ M : Matrix (Fin n) (Fin n) ℕ, 
    (∀ entries : Fin n × Fin n, M entries.fst entries.snd ∈ Fin.range N.succ ∧ 
    ∀ i j : Fin n, ∃ a b c : Fin n, (a ≠ i ∨ b ≠ j ∨ c ≠ j) ∧ (M i j = M a j * M b j ∨ M i j = M i c * M i b))) :=
sorry

end no_such_arrangement_l295_295492


namespace blue_spot_percentage_l295_295139

def total_cows : ℕ := 140
def red_spot_percentage : ℕ := 40
def no_spot_cows : ℕ := 63

theorem blue_spot_percentage :
  let red_spot_cows := (red_spot_percentage * total_cows) / 100
  let no_red_spot_cows := total_cows - red_spot_cows
  let blue_spot_cows := no_red_spot_cows - no_spot_cows in
  (blue_spot_cows * 100) / no_red_spot_cows = 25 :=
by
  sorry

end blue_spot_percentage_l295_295139


namespace A_plus_B_l295_295822

theorem A_plus_B:
  (A B: ℝ) (radius1 radius2: ℝ) (small_circles large_circles: ℕ) 
  (grid_side small_square_side: ℝ)
  (A_eq: A = grid_side^2)
  (B_eq: B = small_circles * π * radius1^2 + large_circles * π * radius2^2)
  (total_area: small_circles = 4)
  (radius1_val: radius1 = 1)
  (large_circles_val: large_circles = 2)
  (radius2_val: radius2 = 1.5)
  (grid_side_val: grid_side = 8)
  (visible_area_eq: A - B * π = 64 - 8.5 * π)
  : A + B = 72.5
  := by
  sorry

end A_plus_B_l295_295822


namespace largest_angle_of_triangle_l295_295842

theorem largest_angle_of_triangle (d e f : ℝ) (h1 : d + 2 * e + 2 * f = d ^ 2) (h2 : d + 2 * e - 2 * f = -9) : 
  ∃ (F : ℝ), F = 120 ∧ cos (F.toRadians) = -1 / 2 :=
by
  sorry

end largest_angle_of_triangle_l295_295842


namespace point_P_reaches_10_l295_295654

theorem point_P_reaches_10 (tosses : ℕ) (heads : ℕ) (tails : ℕ) 
  (H_tosses : tosses ≤ 12)
  (H_heads_tails : heads + tails = tosses)
  (H_heads_at_10 : heads = 10) :
  (∑ k in finset.range 3, if k = 0 then 1 else nat.choose (10 + k - 1) (k - 1)) = 66 :=
by
  sorry

end point_P_reaches_10_l295_295654


namespace value_of_v5_when_x_is_3_l295_295966

noncomputable def polynomial : ℕ → ℤ := 
  λ x, 3 * x ^ 9 + 3 * x ^ 6 + 5 * x ^ 4 + x ^ 3 + 7 * x ^ 2 + 3 * x + 1

noncomputable def horner_evaluation (x : ℕ) : ℕ := 
  let v0 := x in
  let v1 := v0 * x in
  let v2 := v1 * x in
  let v3 := v2 * x + 3 in
  let v4 := v3 * x in
  let v5 := v4 * x + 5 in
  v5

theorem value_of_v5_when_x_is_3 : horner_evaluation 3 = 761 :=
by
  sorry

end value_of_v5_when_x_is_3_l295_295966


namespace prove_collinear_prove_perpendicular_l295_295363

noncomputable def vec_a : ℝ × ℝ := (1, 3)
noncomputable def vec_b : ℝ × ℝ := (3, -4)

def collinear (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.2 = v1.2 * v2.1

def perpendicular (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem prove_collinear : collinear (-1) :=
by
  sorry

theorem prove_perpendicular : perpendicular (16) :=
by
  sorry

end prove_collinear_prove_perpendicular_l295_295363


namespace geometric_sequence_sum_l295_295109

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = a 1 * q) (h3 : a 3 = a 1 * q^2) 
  (h4 : 2 * a 1 = a 2 + a 3) :
  q = -2 ∧ (∑ i in Finset.range n, (i + 1) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9 := 
by
  sorry

end geometric_sequence_sum_l295_295109


namespace fourth_jeweler_bags_proof_l295_295462

-- Definitions representing the given conditions
def total_bags : ℕ := 13
def total_gold_bars : ℕ := (range (total_bags + 1)).sum
def lost_bag_bars : ℕ := 7 -- As determined in the solution process
def remaining_bars : ℕ := total_gold_bars - lost_bag_bars
def each_jeweler_bags : ℕ := (total_bags - 1) / 4
def each_jeweler_bars : ℕ := remaining_bars / 4

-- Bags assigned based on the problem statement
def first_jeweler_bags : finset ℕ := {1, 8, 12}
def second_jeweler_bags : finset ℕ := {3, 5, 13}
def third_jeweler_bags : finset ℕ := {4, 6, 11}

-- The proposition to prove
theorem fourth_jeweler_bags_proof :
  finset.sum (first_jeweler_bags \ {7}) id = each_jeweler_bars ∧
  finset.sum (second_jeweler_bags \ {7}) id = each_jeweler_bars ∧
  finset.sum (third_jeweler_bags \ {7}) id = each_jeweler_bars ∧
  ∃ fourth_jeweler_bags : finset ℕ, 
  fourth_jeweler_bags = ({2, 9, 10} : finset ℕ) ∧
  finset.sum fourth_jeweler_bags id = each_jeweler_bars ∧ 
  finsets.card fourth_jeweler_bags = each_jeweler_bags :=
sorry

-- The relation between remaining bars and each jeweler's received bars
lemma each_jeweler_receives_equal_bars : remaining_bars % 4 = 0 :=
by sorry

-- The sum of all bags should be equal to remaining bars
lemma sum_of_remaining_bags : (finset.sum (insert 2 (insert 9 (insert 10 ∅))) id) = remaining_bars :=
by sorry

end fourth_jeweler_bags_proof_l295_295462


namespace computer_literate_female_employees_l295_295831

theorem computer_literate_female_employees
  (total_employees : ℕ)
  (percent_female : ℚ)
  (percent_male_literate : ℚ)
  (percent_total_literate : ℚ)
  (total_employees = 1300)
  (percent_female = 60/100)
  (percent_male_literate = 50/100)
  (percent_total_literate = 62/100) :
  let female_employees := percent_female * total_employees,
      male_employees := total_employees - female_employees,
      male_literate := percent_male_literate * male_employees,
      total_literate := percent_total_literate * total_employees,
      female_literate := total_literate - male_literate
  in
    female_literate = 546 := 
by
  sorry

end computer_literate_female_employees_l295_295831


namespace total_cost_expression_minimum_average_cost_monthly_payment_approx_l295_295259

section CarCostProof

variable (n : ℕ+)

-- (1) Establishing the expression for the total cost of using the car for n years
def f (n : ℕ) : ℝ := 0.1 * n^2 + 1.35 * n + 14.4

theorem total_cost_expression (n : ℕ+) : 
  f n = 0.1 * n^2 + 1.35 * n + 14.4 := by sorry

-- (2) Proving the most economical scrapping time is after 12 years
def annual_average_cost (n : ℕ+) : ℝ := (0.1 * n^2 + 1.35 * n + 14.4) / n

theorem minimum_average_cost : 
  (∃ n : ℕ+, n = 12) → ∀ k : ℕ+, annual_average_cost n ≤ annual_average_cost k := by sorry

-- (3) Proving the monthly installment payment is approximately 6773 yuan
def monthly_payment : ℝ := 14.4 * 1.27 * 0.01 / (1.27 - 1)

theorem monthly_payment_approx : 
  abs (monthly_payment - 6773) < 1 := by sorry

end CarCostProof

end total_cost_expression_minimum_average_cost_monthly_payment_approx_l295_295259


namespace simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l295_295306

-- Problem (1)
theorem simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth :
  (Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1 / 5) = 6 * Real.sqrt 5 / 5) :=
by
  sorry

-- Problem (2)
theorem simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3 :
  (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1 / 2) * Real.sqrt 3 = 2 :=
by
  sorry

end simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l295_295306


namespace min_value_2a_plus_b_l295_295364

theorem min_value_2a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + b = a^2 + a * b) :
  2 * a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_2a_plus_b_l295_295364


namespace coin_ratio_l295_295255

theorem coin_ratio (total_value : ℕ) (one_rupee_coins : ℕ) 
  (fifty_paise_coins : ℕ) (twentyfive_paise_coins : ℕ) :
  total_value = 140 →
  one_rupee_coins = 80 →
  fifty_paise_coins = 80 →
  twentyfive_paise_coins = 80 →
  (one_rupee_coins : fifty_paise_coins : twentyfive_paise_coins) = (1 : 1 : 1) :=
by
  intros h_total h_one h_fifty h_twentyfive
  rw [←h_one, ←h_fifty, ←h_twentyfive]
  -- prove the ratio 80:80:80 simplifies to 1:1:1
  sorry

end coin_ratio_l295_295255


namespace inequality_abc_l295_295444

theorem inequality_abc {a b c : ℝ} {n : ℕ} 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) (hn : 0 < n) :
  (1 / (1 + a)^(1 / n : ℝ)) + (1 / (1 + b)^(1 / n : ℝ)) + (1 / (1 + c)^(1 / n : ℝ)) 
  ≤ 3 / (1 + (a * b * c)^(1 / 3 : ℝ))^(1 / n : ℝ) := sorry

end inequality_abc_l295_295444


namespace bookseller_fiction_books_count_l295_295648

theorem bookseller_fiction_books_count (n : ℕ) (h1 : n.factorial * 6 = 36) : n = 3 :=
sorry

end bookseller_fiction_books_count_l295_295648


namespace intersection_M_N_l295_295794

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by
  sorry

end intersection_M_N_l295_295794


namespace inequality_abc_l295_295544

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := 
by
  sorry

end inequality_abc_l295_295544


namespace y_affected_by_other_factors_l295_295484

-- Given the linear regression model
def linear_regression_model (b a e x : ℝ) : ℝ := b * x + a + e

-- Theorem: Prove that the dependent variable \( y \) may be affected by factors other than the independent variable \( x \)
theorem y_affected_by_other_factors (b a e x : ℝ) :
  ∃ y, (y = linear_regression_model b a e x ∧ e ≠ 0) :=
sorry

end y_affected_by_other_factors_l295_295484


namespace original_amount_of_the_bill_l295_295949

-- Definitions used in the Lean statement
def TD : ℝ := 360
def BD : ℝ := 432

-- Lean 4 statement for the proof problem
theorem original_amount_of_the_bill : 
  ∃ (FV : ℝ), FV = 1800 ∧ (BD - TD = 72) ∧ (72 = (TD^2) / FV) :=
by
  let FV := 1800
  use FV
  split
  { 
    refl
  }
  split
  {
    simp [BD, TD]
  }
  {
    field_simp
    sorry
  }

end original_amount_of_the_bill_l295_295949


namespace ellipse_condition_l295_295565

theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 3) → ((m > 1 ∧ m < 3 ∧ m ≠ 2) ∨ (m = 2)) :=
by
  sorry

end ellipse_condition_l295_295565


namespace inscribed_sphere_l295_295279

theorem inscribed_sphere (r_base height : ℝ) (r_sphere b d : ℝ)
  (h_base : r_base = 15)
  (h_height : height = 20)
  (h_sphere : r_sphere = b * Real.sqrt d - b)
  (h_rsphere_eq : r_sphere = 120 / 11) : 
  b + d = 12 := 
sorry

end inscribed_sphere_l295_295279


namespace plates_difference_l295_295593

def num_plates_sunshine := 26^3 * 10^3
def num_plates_prairie := 26^2 * 10^4
def difference := num_plates_sunshine - num_plates_prairie

theorem plates_difference :
  difference = 10816000 := by sorry

end plates_difference_l295_295593


namespace tangent_circle_parallel_collinear_l295_295571

theorem tangent_circle_parallel_collinear
  {CD AB : Line}
  {A B C D E F G H : Point}
  (h_tangent_CD : Tangent CD CircleLarger ∧ Tangent CD CircleSmaller)
  (h_parallel : Parallel CD AB)
  (h_points_AC : OnCircle CircleLarger A ∧ OnCircle CircleLarger C)
  (h_tangent_AB : Tangent AB CircleLarger ∧ Tangent AB CircleSmaller)
  (points_of_tangency : OnCircle CircleLarger F ∧ 
                        OnCircle CircleSmaller E ∧ 
                        OnLine CD F ∧ 
                        OnLine CD E ∧ 
                        OnCircle CircleLarger G ∧
                        OnCircle CircleSmaller H ∧
                        OnLine AB G ∧ 
                        OnLine AB H) :
  CFH = CDH :=
begin
  sorry
end

end tangent_circle_parallel_collinear_l295_295571


namespace alice_real_estate_investment_l295_295284

theorem alice_real_estate_investment :
  ∃ (m : ℝ), m + 5 * m = 200000 → 5 * m = 166666.65 :=
begin
  sorry
end

end alice_real_estate_investment_l295_295284


namespace bees_directions_l295_295215

def point := (ℝ × ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def next_position_A (n : ℕ) : point :=
  ((n / 3) - (2 * n / 3), (n / 3) - (2 * n / 3), (n / 3) - (2 * n / 3))

def next_position_B (n : ℕ) : point :=
  (-(n / 3), -(n / 3), -(n / 3))

theorem bees_directions :
  ∃ n : ℕ, distance (next_position_A n) (next_position_B n) = 15 ∧
  n % 3 = 0 ∧ -- Additional conditions to match the exact position and movement pattern
  (next_position_A (n + 1)).1 > (next_position_A n).1 ∧
  (next_position_B (n + 1)).1 < (next_position_B n).1 :=
by sorry

end bees_directions_l295_295215


namespace symmetric_line_equation_l295_295192

theorem symmetric_line_equation (P : ℚ × ℚ) (slope : ℚ) (symmetric : P = (1, 2) ∧ slope = 3 ∧ is_symmetric (3 * 1 - 1)) : 
  ∃ l2 : ℚ × ℚ → Prop, l2 = fun (x, y) => 3 * x + y + 1 = 0 :=
by
  intro H
  sorry

end symmetric_line_equation_l295_295192


namespace second_amount_is_400_l295_295454

theorem second_amount_is_400 (A : ℝ) : 
  let Interest1 := 200 * 0.1 * 12,
      Interest2 := A * 0.12 * 5
  in Interest1 = Interest2 → A = 400 := 
by
  let Interest1 := 200 * 0.1 * 12
  let Interest2 := A * 0.12 * 5
  intro h
  have eq1 : Interest1 = 240 := by sorry
  have eq2 : Interest2 = 0.6 * A := by sorry
  have h' : 240 = 0.6 * A := by sorry
  rw eq1 at h
  rw eq2 at h
  rw eq2
  symmetry
  exact eq_div_of_mul_eq (by norm_num) h'

end second_amount_is_400_l295_295454


namespace find_orange_shells_l295_295165

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end find_orange_shells_l295_295165


namespace julia_height_in_cm_l295_295086

def height_in_feet : ℕ := 5
def height_in_inches : ℕ := 4
def feet_to_inches : ℕ := 12
def inch_to_cm : ℝ := 2.54

theorem julia_height_in_cm : (height_in_feet * feet_to_inches + height_in_inches) * inch_to_cm = 162.6 :=
sorry

end julia_height_in_cm_l295_295086


namespace pf_length_l295_295077

noncomputable def point (x y : ℝ) := (x, y)

variables {A B O F P : (ℝ × ℝ)}

-- Define the parabola y^2 = 4x and point F
def parabola (p : ℝ × ℝ) : Prop := p.2 ^ 2 = 4 * p.1
def F := point 1 0

-- Define the conditions:
-- 1. AB is a chord of the parabola y^2 = 4x passing through F.
-- 2. The circumcircle of triangle AOB intersects the parabola at P.
-- 3. P is distinct from points O, A, and B.
-- 4. PF bisects angle APB.
def chord (A B : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  parabola A ∧ parabola B ∧ parabola P ∧
  (P ≠ (0,0)) ∧ 
  (P.2 = 2 * (F.1 - P.1))

-- Prove |PF| = sqrt(13) - 1
theorem pf_length (P : point ℝ ℝ) (h : chord A B F P) :
  ∃ y_P, point P.1 y_P ∧
  abs (dist P F) = sqrt 13 - 1 :=
begin
  sorry
end

end pf_length_l295_295077


namespace probability_even_product_l295_295218

theorem probability_even_product (s : Finset ℕ) (hs : s = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ p : ℚ, p = 5 / 7 ∧ 
    let pairs := (s.product s).filter (λ x, x.1 ≠ x.2) in
    let even_product := pairs.count (λ x, (x.1 * x.2) % 2 = 0) in 
    p = even_product / pairs.card) :=
begin
  sorry
end

end probability_even_product_l295_295218


namespace P_512_value_l295_295863

noncomputable def x_sequence : ℕ → ℝ
| 0       := 2
| (n + 1) := x_sequence n - (1 / (4 * x_sequence n)) - 2^(-n : ℝ)

theorem P_512_value : (x_sequence 512, 0) = (some_value, 0) := sorry

end P_512_value_l295_295863


namespace Rudolph_stop_signs_l295_295901

def distance : ℕ := 5 + 2
def stopSignsPerMile : ℕ := 2
def totalStopSigns : ℕ := distance * stopSignsPerMile

theorem Rudolph_stop_signs :
  totalStopSigns = 14 := 
  by sorry

end Rudolph_stop_signs_l295_295901


namespace probability_of_at_least_one_3_l295_295624

noncomputable def probability_at_least_one_3 (d1 d2 : ℕ) (h : d1 ≠ d2) : ℚ :=
let total_outcomes := 30 in
let favorable_outcomes := 10 in
favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_3 (d1 d2 : ℕ) (h : d1 ≠ d2) :
  probability_at_least_one_3 d1 d2 h = 1 / 3 :=
sorry

end probability_of_at_least_one_3_l295_295624


namespace range_of_m_l295_295731

noncomputable def proposition_p (x m : ℝ) := (x - m) ^ 2 > 3 * (x - m)
noncomputable def proposition_q (x : ℝ) := x ^ 2 + 3 * x - 4 < 0

theorem range_of_m (m : ℝ) : 
  (∀ x, proposition_p x m → proposition_q x) → 
  (1 ≤ m ∨ m ≤ -7) :=
sorry

end range_of_m_l295_295731


namespace problem_l295_295337

variable (a : ℝ)

def condition : Prop := (a / 3) - (3 / a) = 4

theorem problem (h : condition a) : ((a ^ 8 - 6561) / (81 * a ^ 4)) * (3 * a / (a ^ 2 + 9)) = 72 :=
by
  sorry

end problem_l295_295337


namespace range_of_function_l295_295936

-- Given conditions 
def independent_variable_range (x : ℝ) : Prop := x ≥ 2

-- Proof statement (no proof only statement with "sorry")
theorem range_of_function (x : ℝ) (y : ℝ) (h : y = Real.sqrt (x - 2)) : independent_variable_range x :=
by sorry

end range_of_function_l295_295936


namespace problem_sqrt_inequality_l295_295435

theorem problem_sqrt_inequality :
  { m : ℝ | sqrt (2 * m + 1) > sqrt (m^2 + m - 1) } = set.Ici ((sqrt 5 - 1) / 2) ∩ set.Iio 2 :=
by
  sorry

end problem_sqrt_inequality_l295_295435


namespace sum_of_perimeters_of_triangle_ACD_l295_295542

variable {A B C D : Type}
variables (AB BC AD CD BD : ℕ) (s : ℕ)

-- Encoding the conditions
def conditions := AB = 9 ∧ BC = 21 ∧ AD = CD ∧ AD ∈ ℕ ∧ BD ∈ ℕ

-- Main statement: Proving the sum of all possible perimeters of triangle ACD is 380
theorem sum_of_perimeters_of_triangle_ACD (h : conditions) : s = 380 :=
by sorry

end sum_of_perimeters_of_triangle_ACD_l295_295542


namespace y_value_solution_l295_295322

theorem y_value_solution (y : ℝ) (h : (3 / y) - ((4 / y) * (2 / y)) = 1.5) : 
  y = 1 + Real.sqrt (19 / 3) := 
sorry

end y_value_solution_l295_295322


namespace tangent_line_at_a_half_at_1_max_interval_length_l295_295775

noncomputable def f (x : ℝ) (a : ℝ) := log x + a*x - x^2

noncomputable def derivative_f (a : ℝ) (x : ℝ) := 1/x + a - 2 * x

theorem tangent_line_at_a_half_at_1 :
  let a := 1/2,
      x := 1,
      f1 := (f x a) 
  in (derivative_f a x = -1/2) → (f1 = -1/2) → ∀ y, y - (-1/2) = (-1/2) * (x - 1) → y = -1/2 * x := sorry

theorem max_interval_length :
  ∀ a t: ℝ, (0 < a ∧ a ≤ 1) → (t = (a + sqrt (a^2 + 8)) / 4) → t - 0 ≤ 1 := sorry

end tangent_line_at_a_half_at_1_max_interval_length_l295_295775


namespace rectangle_area_is_48_l295_295671

-- Defining the square's area
def square_area : ℝ := 16

-- Defining the rectangle's width which is the same as the square's side length
def rectangle_width : ℝ := Real.sqrt square_area

-- Defining the rectangle's length which is three times its width
def rectangle_length : ℝ := 3 * rectangle_width

-- The theorem to state that the area of the rectangle is 48
theorem rectangle_area_is_48 : rectangle_width * rectangle_length = 48 :=
by
  -- Placeholder for the actual proof
  sorry

end rectangle_area_is_48_l295_295671


namespace circle_standard_eq_l295_295944

-- Definitions based on the given conditions
def center : ℝ × ℝ := (-3, 4)
def radius : ℝ := 2

-- The standard equation of a circle
def standard_circle_eq (h k r : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x - h)^2 + (y - k)^2 = r^2

-- The main statement to prove
theorem circle_standard_eq :
  standard_circle_eq (-3) 4 2 = (λ x y, (x + 3)^2 + (y - 4)^2 = 4) :=
by
  sorry

end circle_standard_eq_l295_295944


namespace cat_finishes_on_second_friday_l295_295158

def daily_consumption := (1/4 : ℚ) + (1/6 : ℚ)
def initial_food_supply := 8

theorem cat_finishes_on_second_friday :
  ∃ (days : ℕ), (days > 0 ∧ days <= 14) ∧ 
  (days * daily_consumption = initial_food_supply) ∧
  (days mod 7 = 5) :=
  sorry

end cat_finishes_on_second_friday_l295_295158


namespace trains_encountered_l295_295948

def TravelTime : ℕ := 5
def DepartureInterval : ℕ := 1  -- in hours
def DepartureTime : Nat := 5  -- 5 minutes past the hour

theorem trains_encountered
  (travel_time : ℕ := TravelTime)
  (dep_interval : ℕ := DepartureInterval)
  (dep_time : Nat := DepartureTime)
  : number of trains encountered = 9 := 
sorry

end trains_encountered_l295_295948


namespace sum_first_99_terms_l295_295483

variable {a : ℕ → ℕ}

def geometric_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem sum_first_99_terms 
  (hgeo: geometric_seq a 2)
  (h_sum: ∑ k in Finset.range 33, a (3 * k + 1) = 11) :
  ∑ k in Finset.range 99, a (k + 1) = 77 := 
sorry

end sum_first_99_terms_l295_295483


namespace pow_calculation_l295_295303

-- We assume a is a non-zero real number or just a variable
variable (a : ℝ)

theorem pow_calculation : (2 * a^2)^3 = 8 * a^6 := 
by
  sorry

end pow_calculation_l295_295303


namespace problem_1_problem_2_problem_3_l295_295030

noncomputable def f (a x : ℝ) : ℝ := a^(x-1)

theorem problem_1 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a 3 = 4 → a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a (Real.log a) = 100 → (a = 100 ∨ a = 1 / 10) :=
sorry

theorem problem_3 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (a > 1 → f a (Real.log (1 / 100)) > f a (-2.1)) ∧
  (0 < a ∧ a < 1 → f a (Real.log (1 / 100)) < f a (-2.1)) :=
sorry

end problem_1_problem_2_problem_3_l295_295030


namespace g_negative_l295_295514

def g (a : ℚ) : ℚ := sorry

theorem g_negative {a b : ℚ} (h₁ : ∀ a b, g (a * b) = g a + g b)
                    (h₂ : ∀ p : ℚ, nat.prime p.natAbs → g p = p)
                    (x : ℚ) : 
                    x = 23/30 → g x < 0 :=
by
  intros hx
  sorry

end g_negative_l295_295514


namespace f_derivative_eq_l295_295409

noncomputable def f₀ (x : ℝ) : ℝ := x * Real.sin x

noncomputable def f_n : ℕ+ → ℝ → ℝ
| ⟨1, _⟩, x => Real.sin x + x * Real.cos x
| ⟨(n + 1 : ℕ), h⟩, x => 
  let fn := f_n ⟨n, Nat.succ_pos n⟩ x
  (n + 1) * Real.sin (x + (↑n - 1) * Real.pi / 2) + x * Real.cos (x + (↑n - 1) * Real.pi / 2)

theorem f_derivative_eq (n : ℕ+) :
  f_n n = 
  fun x => n * Real.sin (x + (n - 1) / 2 * Real.pi) + x * Real.cos (x + (n - 1) / 2 * Real.pi) :=
sorry

end f_derivative_eq_l295_295409


namespace area_of_region_l295_295705

theorem area_of_region :
  (∃ (x y: ℝ), x^2 + y^2 = 5 * |x - y| + 2 * |x + y|) → 
  (∃ (A : ℝ), A = 14.5 * Real.pi) :=
sorry

end area_of_region_l295_295705


namespace required_run_rate_l295_295838

theorem required_run_rate (target : ℝ) (initial_run_rate : ℝ) (initial_overs : ℕ) (remaining_overs : ℕ) :
  target = 282 → initial_run_rate = 3.8 → initial_overs = 10 → remaining_overs = 40 →
  (target - initial_run_rate * initial_overs) / remaining_overs = 6.1 :=
by
  intros
  sorry

end required_run_rate_l295_295838


namespace school_club_profit_l295_295668

def price_per_bar_buy : ℚ := 5 / 6
def price_per_bar_sell : ℚ := 2 / 3
def total_bars : ℕ := 1200
def total_cost : ℚ := total_bars * price_per_bar_buy
def total_revenue : ℚ := total_bars * price_per_bar_sell
def profit : ℚ := total_revenue - total_cost

theorem school_club_profit : profit = -200 := by
  sorry

end school_club_profit_l295_295668


namespace proof_main_l295_295870

noncomputable def main : Prop :=
  let p : ℝ
  let q : ℝ
  let r : ℝ
  let t := Real.sqrt p + Real.sqrt q + Real.sqrt r
  (Polynomial.eval (x ^ 3 - 7 * x ^ 2 + 8 * x - 1) p = 0) ∧
  (Polynomial.eval (x ^ 3 - 7 * x ^ 2 + 8 * x - 1) q = 0) ∧
  (Polynomial.eval (x ^ 3 - 7 * x ^ 2 + 8 * x - 1) r = 0) →
  t^4 - 14 * t^2 - 8 * t = -18

theorem proof_main : main := sorry

end proof_main_l295_295870


namespace temperature_conversion_l295_295808

theorem temperature_conversion (C F F_new C_new : ℚ) 
  (h_formula : C = (5/9) * (F - 32))
  (h_C : C = 30)
  (h_F_new : F_new = F + 15)
  (h_F : F = 86)
: C_new = (5/9) * (F_new - 32) ↔ C_new = 38.33 := 
by 
  sorry

end temperature_conversion_l295_295808


namespace road_length_in_km_l295_295537

theorem road_length_in_km (scale : ℕ) (map_length_cm : ℕ) (actual_length_cm : ℕ) (conversion_factor : ℕ) :
  scale = 2500000 → map_length_cm = 6 → conversion_factor = 100000 → actual_length_cm = map_length_cm * scale →
  actual_length_cm / conversion_factor = 150 :=
by
  intros hscale hmap_length hconversion_factor hactual_length
  rw [hscale, hmap_length, hconversion_factor, hactual_length]
  norm_num
  sorry

end road_length_in_km_l295_295537


namespace find_all_desirable_numbers_l295_295718

def is_multiple_of_30 (n : ℕ) : Prop :=
  n % 30 = 0

def replace_greatest_with_one (n : ℕ) : ℕ :=
  let digits := to_digits n in
  if (digits.head! = digits.max!) then
    1 * 10^(digits.length' - 1) + (digits.tail.join_digits)
  else if (digits.tail.head! = digits.max!) then
    digits.head * 10^(digits.length' - 1) + 1 * 10^(digits.tail.length' - 1) + (digits.tail.tail.join_digits)
  else
    digits.head.join_digits

def is_desirable_number (n : ℕ) : Prop :=
  is_three_digit n ∧ distinct_digits n ∧ is_multiple_of_30 (replace_greatest_with_one n)

theorem find_all_desirable_numbers :
  {n : ℕ | is_desirable_number n} = {230, 560, 890, 320, 650, 980} :=
sorry

end find_all_desirable_numbers_l295_295718


namespace scientific_notation_686000_l295_295325

-- Defining the scientific notation criteria
structure SciNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

-- Given in the problem
def x : ℝ := 686000

-- Equivalent proof problem
theorem scientific_notation_686000 :
  ∃ (a : ℝ) (n : ℤ) (hn1 : 1 ≤ |a|) (hn2 : |a| < 10), x = a * (10:ℝ)^n := by
  use 6.86
  use 5
  simp
  split
  exact le_of_lt (lt_abs_self 6.86)
  exact lt_of_le_of_lt (le_abs_self 6.86) (by norm_num)
  sorry

end scientific_notation_686000_l295_295325


namespace hypotenuse_length_l295_295069

theorem hypotenuse_length (a b c : ℝ) (h1 : b = 2 * a) (h2 : c^2 = a^2 + b^2) (h3 : a^2 + b^2 + c^2 = 1450) : 
  c = 5 * real.sqrt 29 :=
begin
  sorry
end

end hypotenuse_length_l295_295069


namespace intervals_of_monotonicity_is_decreasing_in_interval_l295_295406

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + x + 1

theorem intervals_of_monotonicity (a : ℝ) :
  (a^2 ≤ 3 -> ∀ x y : ℝ, x < y -> f x a < f y a) ∧
  (a^2 > 3 -> (∀ x y : ℝ, x < y ∧ x < (-a - real.sqrt (a^2 - 3)) / 3 -> f x a < f y a) ∧ 
  (∀ x y : ℝ, x < y ∧ (-a - real.sqrt (a^2 - 3)) / 3 < x ∧ y < (-a + real.sqrt (a^2 - 3)) / 3 -> f x a > f y a) ∧ 
  (∀ x y : ℝ, x < y ∧ (-a + real.sqrt (a^2 - 3)) / 3 < x -> f x a < f y a)) := 
sorry

theorem is_decreasing_in_interval (a : ℝ) :
  (∀ x : ℝ, -2/3 < x ∧ x < -1/3 -> f' x a ≤ 0) -> a ≥ 2 :=
sorry

end intervals_of_monotonicity_is_decreasing_in_interval_l295_295406


namespace years_later_l295_295335

variables (R F Y : ℕ)

-- Conditions
def condition1 := F = 4 * R
def condition2 := F + Y = 5 * (R + Y) / 2
def condition3 := F + Y + 8 = 2 * (R + Y + 8)

-- The result to be proved
theorem years_later (R F Y : ℕ) (h1 : condition1 R F) (h2 : condition2 R F Y) (h3 : condition3 R F Y) : 
  Y = 8 := by
  sorry

end years_later_l295_295335


namespace sum_valid_fractions_eq_400_l295_295568

open Nat

def isCoprimeWith30 (n r : ℕ) : Prop := Nat.gcd (30 * n + r) 30 = 1

def validNumerators : List ℕ := [1, 7, 11, 13, 17, 19, 23, 29]

-- We define the fractions we are interested in.
def validFractions : List ℚ :=
List.filter (λ f : ℚ, f < 10)
  ((List.range 10).bind (λ n, (validNumerators.map (λ r, (30 * n + r) / 30))))

-- The main proof statement
theorem sum_valid_fractions_eq_400 : (validFractions.sum = 400) :=
sorry

end sum_valid_fractions_eq_400_l295_295568


namespace det_trace_matrix_l295_295545

noncomputable def det_formula (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  1 / (n.factorial) * Matrix.det (Matrix.of (λ i j, if i = j then (Matrix.trace (Matrix.mul A A)) else  Matrix.trace (Matrix.pow A (i + 1))))

theorem det_trace_matrix (A : Matrix (Fin n) (Fin n) ℝ) : Matrix.det A =
  det_formula A :=
  by sorry

end det_trace_matrix_l295_295545


namespace goldfish_distribution_count_l295_295906

-- Definitions based on conditions
def num_goldfish : ℕ := 7
def num_tanks : ℕ := 3
def tank_labels : Fin num_tanks → ℕ := λ i => i + 1
def valid_distribution (distribution : Fin num_tanks → ℕ) : Prop :=
  (∀ i : Fin num_tanks, distribution i ≥ tank_labels i) ∧
  (∑ i, distribution i = num_goldfish)

-- Main theorem to prove
theorem goldfish_distribution_count : 
  ∃ (distributions : Fin num_tanks → ∑ n in finset.range num_tanks, ℕ), 
  valid_distribution distributions ∧ 
  finset.card (finset.filter valid_distribution (finset.range finset.card)) = 455 := sorry

end goldfish_distribution_count_l295_295906


namespace slower_speed_walked_l295_295667

theorem slower_speed_walked (x : ℝ) : 
  (∀ x, (50 / x = 5) ↔ x = 10) → (x = 10) :=
begin
  intro h,
  specialize h 10,
  exact h.mpr (by linarith),
end

end slower_speed_walked_l295_295667


namespace find_n_mod_60_l295_295438

theorem find_n_mod_60 {x y : ℤ} (hx : x ≡ 45 [ZMOD 60]) (hy : y ≡ 98 [ZMOD 60]) :
  ∃ n, 150 ≤ n ∧ n ≤ 210 ∧ (x - y ≡ n [ZMOD 60]) ∧ n = 187 := by
  sorry

end find_n_mod_60_l295_295438


namespace find_angle_C_find_side_a_l295_295384

namespace TriangleProof

-- Declare the conditions and the proof promises
variables {A B C : ℝ} {a b c S : ℝ}

-- First part: Prove angle C
theorem find_angle_C (h1 : c^2 = a^2 + b^2 - a * b) : C = 60 :=
sorry

-- Second part: Prove the value of a
theorem find_side_a (h2 : b = 2) (h3 : S = (3 * Real.sqrt 3) / 2) : a = 3 :=
sorry

end TriangleProof

end find_angle_C_find_side_a_l295_295384


namespace line_intersects_circle_l295_295933

theorem line_intersects_circle (a : ℝ) :
  let line := λ p : ℝ × ℝ, p.1 * a - p.2 + 2 * a = 0
  let circle := λ p : ℝ × ℝ, p.1 ^ 2 + p.2 ^ 2 = 5
  ∃ p : ℝ × ℝ, line p ∧ circle p :=
sorry

end line_intersects_circle_l295_295933


namespace no_four_consecutive_subsets_l295_295048

def f : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 4
| 3       := 8
| n + 4   := f n + f (n + 1) + f (n + 2) + f (n + 3)

theorem no_four_consecutive_subsets :
  f 10 = 773 :=
sorry

end no_four_consecutive_subsets_l295_295048


namespace find_value_of_f_log2_6_l295_295402

def f : ℝ → ℝ
| x := if x ≥ 3 then 2^x else f (x + 1)

lemma log2_6_pos : 2 < real.log 6 / real.log 2 ∧ real.log 6 / real.log 2 < 3 :=
begin
  -- Using properties of logarithms, we can derive that log2(6) is between 2 and 3.
  have h1 : 2 < real.log 6 / real.log 2,
  { sorry }, -- You can fill in the proof step here
  have h2 : real.log 6 / real.log 2 < 3,
  { sorry }, -- You can fill in the proof step here
  exact ⟨h1, h2⟩,
end

theorem find_value_of_f_log2_6 : f (real.log 6 / real.log 2) = 12 :=
begin
  have h_log2_6 : 2 < real.log 6 / real.log 2 ∧ real.log 6 / real.log 2 < 3 := log2_6_pos,
  have h_f : f (real.log 6 / real.log 2) = f (real.log 12 / real.log 2),
  { sorry }, -- You can fill in the proof step here
  have h_f_log2_12 : f (real.log 12 / real.log 2) = 12,
  { sorry }, -- You can fill in the proof step here using the information that real.log 12 / real.log 2 = log2(12)
  rw h_f,
  exact h_f_log2_12,
end

end find_value_of_f_log2_6_l295_295402


namespace problem_statement_l295_295785

noncomputable def f : ℝ → ℝ := sorry -- since only the condition is required, we don't define f explicitly

theorem problem_statement (x : ℝ) :
  (∀ x, f (Real.tan (2 * x)) = Real.tan(x)^4 + (1 / Real.tan(x))^4) →
  f (Real.sin x) + f (Real.cos x) ≥ 196 :=
by
  intros h
  sorry

end problem_statement_l295_295785


namespace simon_treasures_l295_295915

variable (sand_dollars sea_glass seashells total_treasures : Nat)

def collected_sand_dollars : sand_dollars = 10 := 
    by sorry

def collected_sea_glass : sea_glass = 3 * sand_dollars := 
    by sorry

def collected_seashells : seashells = 5 * sea_glass := 
    by sorry

def collected_total_treasures : total_treasures = sand_dollars + sea_glass + seashells := 
    by sorry

theorem simon_treasures : total_treasures = 190 := 
    by
        rw [collected_total_treasures, collected_seashells, collected_sea_glass, collected_sand_dollars]
        simp
        sorry

end simon_treasures_l295_295915


namespace circumscribed_sphere_radius_l295_295941

theorem circumscribed_sphere_radius (a b : ℝ) (h : b^2 > a^2) : 
  let R := 1 / 2 * sqrt (b^2 - a^2) in 
  R = 1 / 2 * sqrt (b^2 - a^2) :=
by 
  sorry

end circumscribed_sphere_radius_l295_295941


namespace area_kappa_l295_295314

open Real

def regular_ngon_area (n : ℕ) (s : ℝ) : ℝ := sorry
def regular_ngon_circumradius_area (n : ℕ) (c : ℝ) : ℝ := sorry
def kappa_enclosed_area (n : ℕ) (s : ℝ) (c : ℝ) : ℝ := sorry

theorem area_kappa (n : ℕ) (A : ℝ) (B : ℝ)
  (hA : A = regular_ngon_area n 1)
  (hB : B = regular_ngon_circumradius_area n 1) :
  kappa_enclosed_area n 1 1 = 6 * A - 2 * B :=
sorry

end area_kappa_l295_295314


namespace determine_p_q_l295_295869

theorem determine_p_q (p q : ℝ) :
  (∀ (z : ℂ), (z ^ 2 + (6 + complex.I * p) * z + (13 + complex.I * q) = 0) → (IM (6 + complex.I * p) = 0 ∧ IM (13 + complex.I * q) = 0)) →
  (p = 0 ∧ q = 0) :=
by
  intros hroots
  have h1 := hroots 0 -- To establish a contradiction highlighting imaginary parts must be zero
  sorry

end determine_p_q_l295_295869


namespace count_consecutive_sum_255_l295_295832

theorem count_consecutive_sum_255 :
  ∃ f : ℕ → ℕ → ℕ, (∀ k n : ℕ, k ≥ 3 → 255 = k * n + k * (k - 1) / 2 → n = (255 - (k * (k - 1)) / 2) / k) ∧
    ((∃ g : ℕ → ℕ, ∀ k : ℕ, 255 % k = 0 ∧ k ≥ 3 → g k = (255 / k) - (k - 1) / 2 ∧ g k > 0) → cardinal.mk (set_of (λ k, k ≥ 3 ∧ ∃ n : ℕ, 255 = k * n + k * (k - 1) / 2))) = 4 := sorry

end count_consecutive_sum_255_l295_295832


namespace quadrilateral_perimeter_l295_295956

-- Define the basic conditions
variables (a b : ℝ)

-- Let's define what happens when Xiao Ming selected 2 pieces of type A, 7 pieces of type B, and 3 pieces of type C
theorem quadrilateral_perimeter (a b : ℝ) : 2 * (a + 3 * b + 2 * a + b) = 6 * a + 8 * b :=
by sorry

end quadrilateral_perimeter_l295_295956


namespace luke_rounds_played_l295_295884

theorem luke_rounds_played (points_per_round total_points : ℕ) (h1 : points_per_round = 146) (h2 : total_points = 22922) :
  total_points / points_per_round = 157 :=
by
  rw [h1, h2]
  norm_num

end luke_rounds_played_l295_295884


namespace segment_through_D_l295_295275

variable (A B C D : Point)
variable (l : Line)
variable (a b c : ℝ)

-- Assume D is given on the angle bisector of ∠BAC
axiom angle_bisector (A B C D : Point) : IsAngleBisector A B C D

-- There exists a line l through D such that the segment inside ∠BAC has a given length
theorem segment_through_D (h : ∃ (l : Line), ∀ (P Q : Point), 
  P ∈ l ∧ Q ∈ l ∧ SegmentInsideAngle A B C P Q ∧ Length P Q = a) :
  ∃ x : ℝ, x = ( -b + sqrt (b^2 + 4*c^2) ) / 2 ∨ x = ( -b - sqrt (b^2 + 4*c^2) ) / 2 :=
sorry

end segment_through_D_l295_295275


namespace find_angle_B_l295_295063

-- Definition of the sides and angle A
def a : ℝ := Real.sqrt 3
def b : ℝ := Real.sqrt 2
def A : ℝ := π / 3  -- 60 degrees in radians

-- Sine rule relation and the conclusion that angle B is 45 degrees
theorem find_angle_B (A_is_60 : A = π / 3) (a_is_sqrt3 : a = Real.sqrt 3) (b_is_sqrt2 : b = Real.sqrt 2) : 
  let sin_B := (b * Real.sin A) / a,
  Real.arcsin sin_B = π / 4 := 
  by
    sorry

end find_angle_B_l295_295063


namespace total_cartons_needed_l295_295529

theorem total_cartons_needed (strawberries blueberries purchased : ℕ) : strawberries = 4 → blueberries = 8 → purchased = 9 → (strawberries + blueberries + purchased = 21) :=
by 
  intros h_straw h_blue h_purchase
  rw [h_straw, h_blue, h_purchase]
  norm_num

end total_cartons_needed_l295_295529


namespace final_tally_l295_295830

def total_votes : ℕ := 4000000

def invalid_votes_percentage : ℝ := 0.15
def valid_votes_percentage : ℝ := 1 - invalid_votes_percentage

def valid_votes : ℕ := (valid_votes_percentage * total_votes).to_nat

def first_preference_A_percentage : ℝ := 0.4
def first_preference_B_percentage : ℝ := 0.3
def first_preference_C_percentage : ℝ := 0.2
def first_preference_D_percentage : ℝ := 0.1

def first_preference_A : ℕ := (first_preference_A_percentage * valid_votes).to_nat
def first_preference_B : ℕ := (first_preference_B_percentage * valid_votes).to_nat
def first_preference_C : ℕ := (first_preference_C_percentage * valid_votes).to_nat
def first_preference_D : ℕ := (first_preference_D_percentage * valid_votes).to_nat

def redistribution_A_percentage : ℝ := 0.25
def redistribution_B_percentage : ℝ := 0.35
def redistribution_C_percentage : ℝ := 0.4

def redistributed_A : ℕ := (redistribution_A_percentage * first_preference_D).to_nat
def redistributed_B : ℕ := (redistribution_B_percentage * first_preference_D).to_nat
def redistributed_C : ℕ := (redistribution_C_percentage * first_preference_D).to_nat

def total_A : ℕ := first_preference_A + redistributed_A
def total_B : ℕ := first_preference_B + redistributed_B
def total_C : ℕ := first_preference_C + redistributed_C
def total_D : ℕ := 0

theorem final_tally :
  total_A = 1445000 ∧
  total_B = 1139000 ∧
  total_C = 816000 ∧
  total_D = 0 :=
by
  -- Proof goes here
  sorry

end final_tally_l295_295830


namespace coffee_price_l295_295293

theorem coffee_price (C : ℝ) :
  (7 * C) + (8 * 4) = 67 → C = 5 :=
by
  intro h
  sorry

end coffee_price_l295_295293


namespace total_students_in_grade_l295_295330

-- Define the conditions as constants and assumptions
constant n_students_better_than_ella : ℕ := 59
constant n_students_worse_than_ella : ℕ := 59

-- Theorem statement: total number of students
theorem total_students_in_grade : (n_students_better_than_ella + 1 + n_students_worse_than_ella) = 119 :=
by
  -- This is a place holder proof
  sorry

end total_students_in_grade_l295_295330


namespace prime_count_from_set_l295_295430

def is_prime (n : ℕ) : Bool :=
  n > 1 ∧ ¬(∃ m ∈ list.range (n - 2), n % (m + 2) = 0)

def two_digit_primes_from_digit_set (s : List ℕ) : List ℕ :=
  let pairs := s.bind (λ x, s.filter (λ y, y ≠ x).map (λ y, 10 * x + y))
  pairs.filter is_prime

theorem prime_count_from_set :
  two_digit_primes_from_digit_set [3, 5, 6, 7] = [37, 53, 67, 73] ∧
  (two_digit_primes_from_digit_set [3, 5, 6, 7]).length = 4 := by
  sorry

end prime_count_from_set_l295_295430


namespace sum_of_intersections_l295_295765

noncomputable def f (x : ℝ) : ℝ := sorry -- We assume the function f is given

def g (x : ℝ) : ℝ := (x - 1)^3 + 1

def symmetric_about (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f (2 * c.1 - x) = 2 * c.2 - f x

axiom f_symmetric : symmetric_about f (1, 1)
axiom f_g_intersections : ∃ points : list (ℝ × ℝ), 
  points.length = 2019 ∧ 
  ∀ p ∈ points, f p.1 = g p.1

theorem sum_of_intersections :
  (∑ p : ℝ × ℝ in (f_g_intersections.some : list (ℝ × ℝ)), (p.1 + p.2)) = 4038 := by
  sorry

end sum_of_intersections_l295_295765


namespace cost_per_ice_cream_l295_295285

theorem cost_per_ice_cream (chapati_count : ℕ)
                           (rice_plate_count : ℕ)
                           (mixed_vegetable_plate_count : ℕ)
                           (ice_cream_cup_count : ℕ)
                           (cost_per_chapati : ℕ)
                           (cost_per_rice_plate : ℕ)
                           (cost_per_mixed_vegetable : ℕ)
                           (amount_paid : ℕ)
                           (total_cost_chapatis : ℕ)
                           (total_cost_rice : ℕ)
                           (total_cost_mixed_vegetable : ℕ)
                           (total_non_ice_cream_cost : ℕ)
                           (total_ice_cream_cost : ℕ)
                           (cost_per_ice_cream_cup : ℕ) :
    chapati_count = 16 →
    rice_plate_count = 5 →
    mixed_vegetable_plate_count = 7 →
    ice_cream_cup_count = 6 →
    cost_per_chapati = 6 →
    cost_per_rice_plate = 45 →
    cost_per_mixed_vegetable = 70 →
    amount_paid = 961 →
    total_cost_chapatis = chapati_count * cost_per_chapati →
    total_cost_rice = rice_plate_count * cost_per_rice_plate →
    total_cost_mixed_vegetable = mixed_vegetable_plate_count * cost_per_mixed_vegetable →
    total_non_ice_cream_cost = total_cost_chapatis + total_cost_rice + total_cost_mixed_vegetable →
    total_ice_cream_cost = amount_paid - total_non_ice_cream_cost →
    cost_per_ice_cream_cup = total_ice_cream_cost / ice_cream_cup_count →
    cost_per_ice_cream_cup = 25 :=
by
    intros; sorry

end cost_per_ice_cream_l295_295285


namespace contrapositive_sin_l295_295566

theorem contrapositive_sin (A B : ℝ) : (A = B → sin A = sin B) → (sin A ≠ sin B → A ≠ B) :=
by
  sorry

end contrapositive_sin_l295_295566


namespace find_angle_A_range_of_perimeter_l295_295820

-- Definitions based on the problem conditions
variables {A B C : ℝ} {a b c : ℝ}
variables (triangle_ABC : a^2 - 2 * b * c * (Math.cos A) = (b + c)^2)

-- Proving the measure of angle A
theorem find_angle_A (h : a^2 - 2 * b * c * (Real.cos A) = (b + c)^2) :
  A = 120 :=
sorry

-- Proving the range of the perimeter given a = 3
theorem range_of_perimeter (h : a = 3) : (6 < a + b + c) ∧ (a + b + c ≤ 2 * Real.sqrt 3 + 3) :=
sorry

end find_angle_A_range_of_perimeter_l295_295820


namespace find_a_l295_295839

theorem find_a {S : ℕ → ℤ} (a : ℤ)
  (hS : ∀ n : ℕ, S n = 5 ^ (n + 1) + a) : a = -5 :=
sorry

end find_a_l295_295839


namespace average_age_women_l295_295561

variables (A : ℝ) (n : ℕ)

-- Conditions
def average_age_10_men (A : ℝ) : ℝ := A
def total_age_replaced_men : ℝ := 15 + 20 + 25 + 30
def new_average_age (A : ℝ) : ℝ := A + 4
def total_age_before_replacement (A : ℝ) : ℝ := 10 * A
def total_age_after_replacement (A : ℝ) : ℝ := 10 * (A + 4)
def total_age_non_replaced_men (A : ℝ) : ℝ := 6 * A

-- Proof of the main problem
theorem average_age_women : average_age_10_men A − total_age_replaced_men = 6 * A →
                              total_age_after_replacement A − total_age_non_replaced_men A = 130 →
                              (total_age_after_replacement A − total_age_non_replaced_men A) / 4 = 32.5 :=
by
  sorry

end average_age_women_l295_295561


namespace exists_zero_in_interval_l295_295596

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 3

theorem exists_zero_in_interval :
  ∃ c ∈ Ioo (1 / 2 : ℝ) 1, f c = 0 :=
sorry

end exists_zero_in_interval_l295_295596


namespace percent_increase_is_correct_l295_295289

def increase : ℝ := 25000
def new_salary : ℝ := 90000
def original_salary : ℝ := new_salary - increase
def percent_increase : ℝ := (increase / original_salary) * 100

theorem percent_increase_is_correct : percent_increase = 38.46 :=
by
  -- Proof goes here
  sorry

end percent_increase_is_correct_l295_295289


namespace average_fixed_points_of_permutation_l295_295666

open Finset

noncomputable def average_fixed_points (n : ℕ) : ℕ :=
  1

theorem average_fixed_points_of_permutation (n : ℕ) :
  ∀ (σ : (Fin n) → (Fin n)), 
  (1: ℚ) = (1: ℕ) :=
by
  sorry

end average_fixed_points_of_permutation_l295_295666


namespace truncated_cone_volume_correct_l295_295952

-- Definition of given conditions
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height : ℝ := 8

-- Definition of the formula for the volume of a truncated cone
def truncated_cone_volume (R r h : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

-- The theorem that we need to prove
theorem truncated_cone_volume_correct :
  truncated_cone_volume large_base_radius small_base_radius height = 466.67 * Real.pi :=
by 
  sorry

end truncated_cone_volume_correct_l295_295952


namespace weight_of_one_serving_l295_295350

theorem weight_of_one_serving
  (total_servings : ℕ)
  (chicken_weight_pounds : ℝ)
  (stuffing_weight_ounces : ℝ)
  (ounces_per_pound : ℝ)
  (total_servings = 12)
  (chicken_weight_pounds = 4.5)
  (stuffing_weight_ounces = 24)
  (ounces_per_pound = 16) :
  (chicken_weight_pounds * ounces_per_pound + stuffing_weight_ounces) / total_servings = 8 :=
by
  sorry

end weight_of_one_serving_l295_295350


namespace equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l295_295613

-- Definition of a cute triangle
def is_cute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

-- 1. Prove an equilateral triangle is a cute triangle
theorem equilateral_is_cute (a : ℝ) : is_cute_triangle a a a :=
by
  sorry

-- 2. Prove the triangle with sides 4, 2√6, and 2√5 is a cute triangle
theorem specific_triangle_is_cute : is_cute_triangle 4 (2*Real.sqrt 6) (2*Real.sqrt 5) :=
by
  sorry

-- 3. Prove the length of AB for the given right triangle is 2√6 or 2√3
theorem find_AB_length (AB BC : ℝ) (AC : ℝ := 2*Real.sqrt 2) (h_cute : is_cute_triangle AB BC AC) : AB = 2*Real.sqrt 6 ∨ AB = 2*Real.sqrt 3 :=
by
  sorry

end equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l295_295613


namespace field_trip_students_l295_295563

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end field_trip_students_l295_295563


namespace common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295104

def is_geometric_sequence (a : ℕ → ℂ) (q : ℂ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_mean (a1 a2 a3 : ℂ) : Prop :=
  2 * a1 = a2 + a3

noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  (1 - (1 + 3 * n) * (-2 : ℤ)^n) / 9

theorem common_ratio_of_geometric_sequence (a : ℕ → ℂ) (q : ℂ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : q ≠ 1) 
  (h3 : arithmetic_mean (a 1) (a 2) (a 3)) : 
  q = -2 := 
sorry

theorem sum_of_first_n_terms_of_sequence (n : ℕ) 
  (a : ℕ → ℂ) 
  (h1 : is_geometric_sequence a (-2)) 
  (h2 : a 1 = 1) : 
  ∑ k in finset.range n, k * a k = sum_first_n_terms n := 
sorry

end common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295104


namespace extra_bananas_l295_295140

theorem extra_bananas (B : ℝ) : ∃ (E : ℝ), 
  (∀ (n_total n_absent : ℕ), 
  n_total = 700 ∧ n_absent = 350 → 
  let n_present := n_total - n_absent in
  let B_share_if_all_present := B / 700 in
  let B_share_if_absent := B / 350 in
  B_share_if_absent = B_share_if_all_present + E → 
  E = 350) :=
begin
  use 350,
  intros n_total n_absent h,
  cases h with ht ha,
  rw [ht, ha],
  let n_present := 700 - 350,
  let B_share_if_all_present := B / 700,
  let B_share_if_absent := B / 350,
  suffices : B / 350 = B / 700 + 350,
  exact this
end

end extra_bananas_l295_295140


namespace evolution_of_information_l295_295840

noncomputable def seqs (α : Type*) [OrderedField α] :=
  Σ' (a b : ℕ → α), 
      ∃ (a1 b1 : α), (a 1 > b 1) ∧ 
          (∀ n : ℕ, a (n + 1) = 2 * a n + b n ∧ b (n + 1) = a n + 2 * b n ∧ a 0 = a1 ∧ b 0 = b1)

theorem evolution_of_information (α : Type*) [OrderedField α] (σ : seqs α) :
  let ⟨a, b, h⟩ := σ in
  (∀ n : ℕ, 0 < n → a n > b n) ∧
  (∀ n : ℕ, 0 < n → a (n + 1) > a n ∧ b (n + 1) > b n) ∧
  ∃ k : ℕ, ∀ n : ℕ, k < n → abs ((a n) / (b n) - 1) < (10 : α)^(-10) :=
by
  sorry

end evolution_of_information_l295_295840


namespace sum_real_imag_parts_l295_295389

def complex_condition (z : ℂ) : Prop := (1 - complex.i)^2 = complex.norm_sq (1 + complex.i) * z

theorem sum_real_imag_parts (z : ℂ) (h : complex_condition z) : z.re + z.im = -1 := sorry

end sum_real_imag_parts_l295_295389


namespace inscribed_circle_radius_eq_l295_295978

noncomputable def triangle_radius_inscribed {ABC : Type} (AB AC BC : ℝ) (hAB : AB = 7) (hAC : AC = 7) (hBC : BC = 6) : ℝ :=
  let s := (AB + AC + BC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  area / s

theorem inscribed_circle_radius_eq {ABC : Type} (AB AC BC : ℝ) (hAB : AB = 7) (hAC : AC = 7) (hBC : BC = 6) :
  triangle_radius_inscribed AB AC BC hAB hAC hBC = 3 * Real.sqrt 10 / 5 :=
by
  sorry

end inscribed_circle_radius_eq_l295_295978


namespace geometric_sequence_sum_l295_295108

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = a 1 * q) (h3 : a 3 = a 1 * q^2) 
  (h4 : 2 * a 1 = a 2 + a 3) :
  q = -2 ∧ (∑ i in Finset.range n, (i + 1) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9 := 
by
  sorry

end geometric_sequence_sum_l295_295108


namespace total_cost_is_correct_l295_295849

noncomputable def jacket_cost : ℝ := 54.74
noncomputable def sweater_cost : ℝ := 22.99
noncomputable def jeans_cost : ℝ := 32.36
noncomputable def boots_cost : ℝ := 76.45
noncomputable def discount_percentage : ℝ := 0.12
noncomputable def total_cost_before_discount : ℝ := jacket_cost + sweater_cost + jeans_cost + boots_cost
noncomputable def discount_amount : ℝ := total_cost_before_discount * discount_percentage
noncomputable def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem total_cost_is_correct :
  total_cost_after_discount ≈ 164.16 := -- Using ≈ to denote approximate equality
by
  -- The proof will go here
  sorry

end total_cost_is_correct_l295_295849


namespace equation_of_line_through_midpoint_of_chord_l295_295143

-- Define the problem
def midpoint_of_chord (P : ℝ × ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  let l : ℝ → ℝ := λ x, k * (x - P.1) + P.2
  let ellipse : ℝ × ℝ → Prop := λ pt, (pt.1 ^ 2) / (a ^ 2) + (pt.2 ^ 2) / (b ^ 2) = 1
  let chord_points : set (ℝ × ℝ) := { pt | ellipse pt ∧ pt.2 = l pt.1 }
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ ∀ pt ∈ chord_points, pt.1 = x1 ∨ pt.1 = x2 ∧ (x1 + x2) / 2 = P.1

theorem equation_of_line_through_midpoint_of_chord :
  ∀ (P : ℝ × ℝ) (a b : ℝ), 
  P = (1, 1) ∧ a = 3 ∧ b = 2 →
  (midpoint_of_chord P (-4/9) a b →
  ∃ (A B C : ℝ), A * P.1 + B * P.2 = C ∧ A = 4 ∧ B = 9 ∧ C = 13) :=
by
  intros
  sorry

end equation_of_line_through_midpoint_of_chord_l295_295143


namespace savings_percentage_l295_295662

variable (I : ℝ) -- First year's income
variable (S : ℝ) -- Amount saved in the first year

-- Conditions
axiom condition1 (h1 : S = 0.05 * I) : Prop
axiom condition2 (h2 : S + 0.05 * I = 2 * S) : Prop
axiom condition3 (h3 : (I - S) + 1.10 * (I - S) = 2 * (I - S)) : Prop

-- Theorem that proves the man saved 5% of his income in the first year
theorem savings_percentage : S = 0.05 * I :=
by
  sorry -- Proof goes here

end savings_percentage_l295_295662


namespace triangle_inequalities_l295_295250

theorem triangle_inequalities (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) →
  (a = b ∧ a > c) ∨ (a = b ∧ b = c) :=
by
  sorry

end triangle_inequalities_l295_295250


namespace relationship_A_B_l295_295035

variable (n : ℕ)

def A_n (n : ℕ) : ℝ := ∑ k in finset.range (n+1) \ {0}, (3 * k) / (1 + k^2 + k^4)
def B_n (n : ℕ) : ℝ := ∏ k in finset.range (n+1) \ {0, 1}, (k^3 + 1) / (k^3 - 1)

theorem relationship_A_B (h : n ≥ 2) : A_n n = B_n n :=
sorry

end relationship_A_B_l295_295035


namespace sum_y_coords_on_y_axis_l295_295308

theorem sum_y_coords_on_y_axis {C : Type} [normed_group C] [normed_space ℝ C]
  (center : C) (radius : ℝ) (x y : ℝ) (h₁ : center = (-4, 5))
  (h₂ : radius = 13) (h₃ : ∀ x y: ℝ, (x + 4)^2 + (y - 5)^2 = radius^2) :
  let y₁ := 5 + sqrt 153,
      y₂ := 5 - sqrt 153 in
  y₁ + y₂ = 10 :=
by
  sorry

end sum_y_coords_on_y_axis_l295_295308


namespace distance_second_day_l295_295171

theorem distance_second_day 
  (total_distance : ℕ)
  (a1 : ℕ)
  (n : ℕ)
  (r : ℚ)
  (hn : n = 6)
  (htotal : total_distance = 378)
  (hr : r = 1 / 2)
  (geo_sum : a1 * (1 - r^n) / (1 - r) = total_distance) :
  a1 * r = 96 :=
by
  sorry

end distance_second_day_l295_295171


namespace complex_magnitude_problem_l295_295740

theorem complex_magnitude_problem :
  let z := (1 - 2 * Complex.i) * Complex.i
  in Complex.abs z = Real.sqrt 5 :=
by
  let z := (1 - 2 * Complex.i) * Complex.i
  -- sorry is used here to skip detailed proof steps
  sorry

end complex_magnitude_problem_l295_295740


namespace parallelogram_circle_intersection_l295_295009

-- Defining the setup for the problem
variables {A B C D M K N : Point}
variables (parallelogram_ABCD : Parallelogram A B C D)
variables (circle_omega : Circle ω A M K N)
variables (intersects_AB_M : Intersects circle_omega (Segment A B) M)
variables (intersects_AC_K : Intersects circle_omega (Segment A C) K)
variables (intersects_AD_N : Intersects circle_omega (Segment A D) N)

theorem parallelogram_circle_intersection
  (h₀ : parallelogram_ABCD)
  (h₁ : circle_omega)
  (h₂ : intersects_AB_M)
  (h₃ : intersects_AC_K)
  (h₄ : intersects_AD_N) :
  dist A B * dist A M + dist A D * dist A N = dist A K * dist A C :=
sorry

end parallelogram_circle_intersection_l295_295009


namespace remainder_zero_when_divided_by_condition_l295_295321

noncomputable def remainder_problem (x : ℂ) : ℂ :=
  (2 * x^5 - x^4 + x^2 - 1) * (x^3 - 1)

theorem remainder_zero_when_divided_by_condition (x : ℂ) (h : x^2 - x + 1 = 0) :
  remainder_problem x % (x^2 - x + 1) = 0 := by
  sorry

end remainder_zero_when_divided_by_condition_l295_295321


namespace number_of_pairs_of_skew_lines_l295_295798

noncomputable def cube_vertices : set (set (ℝ × ℝ × ℝ)) := sorry

noncomputable def is_skew (l1 l2 : set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem number_of_pairs_of_skew_lines : 
  let lines := {l : set (ℝ × ℝ × ℝ) | ∃ v1 v2 ∈ cube_vertices, l = {v1, v2}} in
  (lines.toList.combinations 2).count (λ p, is_skew p.head p.tail.head) = 174 :=
sorry

end number_of_pairs_of_skew_lines_l295_295798


namespace years_of_school_eq_13_l295_295085

/-- Conditions definitions -/
def cost_per_semester : ℕ := 20000
def semesters_per_year : ℕ := 2
def total_cost : ℕ := 520000

/-- Derived definitions from conditions -/
def cost_per_year := cost_per_semester * semesters_per_year
def number_of_years := total_cost / cost_per_year

/-- Proof that number of years equals 13 given the conditions -/
theorem years_of_school_eq_13 : number_of_years = 13 :=
by sorry

end years_of_school_eq_13_l295_295085


namespace largest_area_triangle_inscribed_l295_295146

-- Definitions for necessary conditions
variable {M : Type} [convex_polygon M]

-- Mathematically equivalent proof problem
theorem largest_area_triangle_inscribed (M : convex_polygon) :
  ∀ (XYZ : triangle), inscribed_in_polygon XYZ M → 
  ∃ (P Q R : vertex M), area (triangle.mk P Q R) ≥ area XYZ :=
sorry

end largest_area_triangle_inscribed_l295_295146


namespace equivalent_expression_l295_295536

theorem equivalent_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x⁻² * y⁻²) / (x⁻⁴ - y⁻⁴) = (x² * y²) / (y⁴ - x⁴) :=
by { sorry }

end equivalent_expression_l295_295536


namespace part1_part2_l295_295365

theorem part1 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : a^2 + b^2 = 22 :=
sorry

theorem part2 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : (a - 2) * (b + 2) = 7 :=
sorry

end part1_part2_l295_295365


namespace molly_flips_two_tails_consecutively_l295_295138

theorem molly_flips_two_tails_consecutively :
  (prob_T : ℚ) (prob_T = 1/2) → (prob_TT : ℚ) (prob_TT = prob_T * prob_T) → prob_TT = 1/4 :=
by intros prob_T h_prob_1 prob_TT h_prob_2; rw [h_prob_1, h_prob_2]; norm_num

#check molly_flips_two_tails_consecutively

end molly_flips_two_tails_consecutively_l295_295138


namespace volume_truncated_cone_l295_295950

-- Define the geometric constants
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height_truncated_cone : ℝ := 8

-- The statement to prove the volume of the truncated cone
theorem volume_truncated_cone :
  let V_large := (1/3) * Real.pi * (large_base_radius^2) * (height_truncated_cone + height_truncated_cone)
  let V_small := (1/3) * Real.pi * (small_base_radius^2) * height_truncated_cone
  V_large - V_small = (1400/3) * Real.pi :=
by
  sorry

end volume_truncated_cone_l295_295950


namespace standard_equation_hyperbola_line_through_fixed_point_l295_295741

-- Define the conditions of the hyperbola and line
variables {C : Type} {e : ℝ} {a b : ℝ} {k m : ℝ} (A B D : Type) [topological_space C]
variables (A : {x1 : ℝ × ℝ | y = k * x1.1 + m ∧ (x1.1 ^ 2) / 4 - (x1.2 ^ 2) = 1})
          (B : {x2 : ℝ × ℝ | y = k * x2.1 + m ∧ (x2.1 ^ 2) / 4 - (x2.2 ^ 2) = 1})
          (D : {D : ℝ × ℝ | D = (-2, 0)})

-- State the standard equation of the hyperbola
theorem standard_equation_hyperbola (h_center : ∀ c ∈ C, c = (0, 0))
                                    (h_foci : ∃ c1 c2 ∈ C, c1.2 = 0 ∧ c2.2 = 0 ∧ c1.1 = -c2.1)
                                    (h_eccen : e = sqrt 5 / 2)
                                    (h_conj_len : 2 * b = 2) :
  (∃ a b, a = 2 ∧ b = 1 ∧ (x ^ 2) / a ^ 2 - y ^ 2 / b ^ 2 = 1) := 
sorry

-- Prove that the line passes through the fixed point
theorem line_through_fixed_point (h_intersection : ∃ A B, A ≠ B ∧ l ∩ C = {A, B})
                                 (h_circle : (x1 = -2 ∧ y1 = 0) ∨ (x2 = -2 ∧ y2 = 0))
                                 (h_diameter_cond : 3 * m^2 - 16 * m * k + 20 * k^2 = 0) :
  ( ∃ fixed_point : ℝ × ℝ, fixed_point = (-10 / 3, 0) ∧ 
    (y = k * x + m) passes_through fixed_point) :=
sorry

end standard_equation_hyperbola_line_through_fixed_point_l295_295741


namespace expected_value_is_3_50_l295_295657

noncomputable def expected_value_winnings : ℚ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let prob := 1 / 8
  let winnings (x : ℕ) : ℚ :=
    if x % 2 = 0 then
      if x = 8 then 16 else x
    else
      0
  ∑ x in outcomes, prob * winnings x

theorem expected_value_is_3_50 : expected_value_winnings = 3.5 := by
  sorry

end expected_value_is_3_50_l295_295657


namespace cone_diameter_height3_slant5_l295_295061

/--
Given a right cone with height 3 and slant height 5, the diameter of the base of the cone is 8.
-/
theorem cone_diameter_height3_slant5 
  (r h l : ℝ) (height_cond : h = 3) (slant_height_cond : l = 5) (pythagorean_cond : r^2 + h^2 = l^2) : 
  2 * r = 8 :=
by
  rw [height_cond, slant_height_cond, pythagorean_cond]
  sorry

end cone_diameter_height3_slant5_l295_295061


namespace evaluate_expression_at_x_l295_295552

theorem evaluate_expression_at_x (x : ℝ) (h : x = Real.sqrt 2 - 3) : 
  (3 * x / (x^2 - 9)) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end evaluate_expression_at_x_l295_295552


namespace louise_hip_size_conversion_l295_295129

theorem louise_hip_size_conversion :
  (let inches_per_foot := 12,
       millimeters_per_foot := 305,
       louise_hip_size_in_inches := 42 in
   louise_hip_size_in_inches / inches_per_foot * millimeters_per_foot = 1067.5) :=
by
  -- The proof details are omitted as per instructions
  sorry

end louise_hip_size_conversion_l295_295129


namespace find_a_and_b_range_of_m_l295_295881

section
variable (a b m : ℝ)

def f (x : ℝ) : ℝ := 2 * x ^ 3 + a * x ^ 2 + b * x + m
def f' (x : ℝ) : ℝ := 6 * x ^ 2 + 2 * a * x + b

theorem find_a_and_b :
  (∀ x : ℝ, f' x = 6 * x ^ 2 + 2 * a * x + b) →
  (∀ x : ℝ, (f' (-x - 1/2)) = f' (x - 1/2)) →
  f' 1 = 0 →
  a = 3 ∧ b = -12 :=
begin
  intros h1 h2 h3,
  sorry
end

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x = 2 * x ^ 3 + 3 * x ^ 2 - 12 * x + m) →
  (∃ x : ℝ, f x = 0) → 
  (∃ y : ℝ, f y = 0) → 
  (∃ z : ℝ, f z = 0) → 
  -20 < m ∧ m < 7 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end

end find_a_and_b_range_of_m_l295_295881


namespace stock_worth_is_100_l295_295274

-- Define the number of puppies and kittens
def num_puppies : ℕ := 2
def num_kittens : ℕ := 4

-- Define the cost per puppy and kitten
def cost_per_puppy : ℕ := 20
def cost_per_kitten : ℕ := 15

-- Define the total stock worth function
def stock_worth (num_puppies num_kittens cost_per_puppy cost_per_kitten : ℕ) : ℕ :=
  (num_puppies * cost_per_puppy) + (num_kittens * cost_per_kitten)

-- The theorem to prove that the stock worth is $100
theorem stock_worth_is_100 :
  stock_worth num_puppies num_kittens cost_per_puppy cost_per_kitten = 100 :=
by
  sorry

end stock_worth_is_100_l295_295274


namespace pairs_of_mittens_correct_l295_295802

variables (pairs_of_plugs_added pairs_of_plugs_original plugs_total pairs_of_plugs_current pairs_of_mittens : ℕ)

theorem pairs_of_mittens_correct :
  pairs_of_plugs_added = 30 →
  plugs_total = 400 →
  pairs_of_plugs_current = plugs_total / 2 →
  pairs_of_plugs_current = pairs_of_plugs_original + pairs_of_plugs_added →
  pairs_of_mittens = pairs_of_plugs_original - 20 →
  pairs_of_mittens = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pairs_of_mittens_correct_l295_295802


namespace percentage_increase_first_year_l295_295932

variable (P : ℝ) -- The percentage increase in the first year
variable (initial_population end_of_second_year_population : ℝ)
variable (decrease_rate : ℝ)

-- Given conditions
def condition1 := initial_population = 415600
def condition2 := decrease_rate = 0.3
def condition3 := end_of_second_year_population = 363650

-- The equation derived from the conditions and steps in the solution
def population_equation := 
  let first_year_population := initial_population + (P / 100) * initial_population in
  let second_year_population := first_year_population * (1 - decrease_rate) in
  second_year_population = end_of_second_year_population

-- The proof statement in Lean 4: Prove P = 25 given the conditions
theorem percentage_increase_first_year (h1 : condition1) (h2 : condition2) (h3 : condition3) : P = 25 :=
by
  sorry

end percentage_increase_first_year_l295_295932


namespace p_or_q_is_false_implies_p_and_q_is_false_l295_295251

theorem p_or_q_is_false_implies_p_and_q_is_false (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ((¬ (p ∧ q) → (p ∨ q ∨ ¬ (p ∨ q)))) := sorry

end p_or_q_is_false_implies_p_and_q_is_false_l295_295251


namespace cos_tan_combination_l295_295710

theorem cos_tan_combination :
  cos (25 * π / 6) + cos (25 * π / 3) + tan (-25 * π / 4) =
  (sqrt 3 / 2) - (1 / 2) :=
by sorry

end cos_tan_combination_l295_295710


namespace max_attempts_four_digit_password_l295_295211

/-- 
Given a 4-digit password where each digit is from 0 to 9 and the sum of the digits is 20,
prove that the maximum number of attempts to unlock the phone is 417.
-/
theorem max_attempts_four_digit_password : 
  ∃ (passwords : Finset (Fin 10 × Fin 10 × Fin 10 × Fin 10)), 
    (∀ passwd ∈ passwords, (passwd.1 + passwd.2 + passwd.3 + passwd.4) = 20) 
    ∧ passwords.card = 417 :=
sorry

end max_attempts_four_digit_password_l295_295211


namespace total_children_l295_295208

variables (happy sad neither happy_boys sad_girls neither_boys boys girls : ℕ)

-- Conditions
def h_cond : happy = 30 :=
-- This condition asserts that there are 30 happy children.
-- sorry

def s_cond : sad = 10 :=
-- This condition asserts that there are 10 sad children.
-- sorry

def n_cond : neither = 20 :=
-- This condition asserts that there are 20 children who are neither happy nor sad.
-- sorry

def b_cond : boys = 16 :=
-- This condition asserts that there are 16 boys.
-- sorry

def g_cond : girls = 44 :=
-- This condition asserts that there are 44 girls.
-- sorry

def hb_cond : happy_boys = 6 :=
-- This condition asserts that there are 6 happy boys.
-- sorry

def sg_cond : sad_girls = 4 :=
-- This condition asserts that there are 4 sad girls.
-- sorry

def nb_cond : neither_boys = 4 :=
-- This condition asserts that there are 4 boys who are neither happy nor sad.
-- sorry

-- Proof Problem
theorem total_children : boys + girls = 60 :=
-- Goal is to prove that the total number of children is 60 given the conditions.
-- Sorry is placed to skip the proof.
sorry

end total_children_l295_295208


namespace mean_value_of_interior_angles_pentagon_l295_295976

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem mean_value_of_interior_angles_pentagon :
  sum_of_interior_angles 5 / 5 = 108 :=
by
  sorry

end mean_value_of_interior_angles_pentagon_l295_295976


namespace determine_a_and_b_l295_295701

variable (a b : ℕ)
theorem determine_a_and_b 
  (h1: 0 ≤ a ∧ a ≤ 9) 
  (h2: 0 ≤ b ∧ b ≤ 9)
  (h3: (a + b + 45) % 9 = 0)
  (h4: (b - a) % 11 = 3) : 
  a = 3 ∧ b = 6 :=
sorry

end determine_a_and_b_l295_295701


namespace ice_cream_to_afford_games_l295_295154

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l295_295154


namespace right_triangles_count_l295_295720

theorem right_triangles_count (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n, (∀ A B C : ℤ × ℤ, 
    A = (0, 0) → 
    B = (xB, yB) ∧ C = (xC, yC) →
    (xB, yB) ≠ (xC, yC) →
    ∃ incenter : ℚ × ℚ, incenter = (96p, 672p) ∧ 
                      -- x-coordinates and y-coordinates of incenter
                      (96 * p = (OB + OC) - AB (both integer coordinates) ∧
                      (672 * p = (OA + OB) - AB (both integer coordinates)) ∧
      xB * yC - xC * yB = 0) ∧
  n = 108) sorry

end right_triangles_count_l295_295720


namespace range_of_g_l295_295784

open Set

def f (x : ℝ) := 1 / x

def is_range (y : ℝ) : Prop :=
  1 ≤ x ∧ x ≤ 2 →
  y = 2 * f x + f (x ^ 2)

theorem range_of_g :
  (range (λ x : ℝ, 2 * f x + f (x ^ 2)) ∩ Icc 1 2) = Icc (1 / 2 + sqrt 2) 3 :=
begin
  sorry
end

end range_of_g_l295_295784


namespace sum_of_c_n_l295_295375

-- Definition of sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Definition of sequence b_n following given recurrence
def b : ℕ → ℝ
| 0       := 1  -- not used as n >= 1
| 1       := 1
| (n + 2) := b (n + 1) / 2

-- Definition of sequence c_n = a_n * b_n
def c (n : ℕ) : ℝ := (a n : ℝ) * b n

-- Partial sum of the sequence c_n
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, c (k + 1))

-- Proving the statement
theorem sum_of_c_n (n : ℕ) (hn : 0 < n) :
  T n = 4 - (n + 2) * (1 / 2)^(n - 1) :=
sorry

end sum_of_c_n_l295_295375


namespace max_n_for_positive_sum_l295_295746
-- Import the necessary library

-- Define the core definitions and the theorem
noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def has_maximum_sum_sn (S : ℕ → ℝ) : Prop :=
  ∃ n, S n = ⊤

theorem max_n_for_positive_sum {a : ℕ → ℝ} {S : ℕ → ℝ} 
  (h_arithmetic : arithmetic_sequence a)
  (h_ratio : a 11 / a 10 < -1)
  (h_max_sum : has_maximum_sum_sn S) :
  ∃ n, S n > 0 ∧ ∀ m > n, S m ≤ 0 :=
begin
  use 19,
  sorry -- Proof is omitted as per the instructions
end

end max_n_for_positive_sum_l295_295746


namespace find_a_and_b_l295_295866

-- Define the given constants and functions.
variables (a b : ℝ)
def f (x : ℝ) : ℝ := (a - 3) * Real.sin x + b
def g (x : ℝ) : ℝ := a + b * Real.cos x

-- Conditions
axiom h1 : ∀ x : ℝ, f x = f (-x)  -- f(x) is an even function
axiom h2 : ∀ t : ℝ, (-1) ≤ t ∧ t ≤ 1 -> g t ≥ -1
axiom h3 : Real.sin b > 0  -- sin b > 0
axiom h4 : ∀ t, -1 ≤ t ∧ t ≤ 1 -> g t ≥ g (-t)

-- Define the theorem to be proven
theorem find_a_and_b : a = 3 ∧ b = -4 :=
  sorry

end find_a_and_b_l295_295866


namespace slope_of_line_l295_295675

theorem slope_of_line
  (m : ℝ)
  (b : ℝ)
  (h1 : b = 4)
  (h2 : ∀ x y : ℝ, y = m * x + b → (x = 199 ∧ y = 800) → True) :
  m = 4 :=
by
  sorry

end slope_of_line_l295_295675


namespace range_of_a_l295_295041

-- Define the set A
def A (a x : ℝ) := 6 * x + a > 0

-- Theorem stating the range of a given the conditions
theorem range_of_a (a : ℝ) (h : ¬ A a 1) : a ≤ -6 :=
by
  -- Here we would provide the proof
  sorry

end range_of_a_l295_295041


namespace fifth_term_of_sequence_l295_295039

theorem fifth_term_of_sequence :
  ∀ (a : ℕ → ℤ), a 1 = 3 ∧ (∀ n : ℕ, a (n + 1) + 2 * a n = 0) → a 5 = 48 :=
begin
  assume a,
  sorry
end

end fifth_term_of_sequence_l295_295039


namespace Nathan_daily_hours_l295_295899

theorem Nathan_daily_hours (x : ℝ) 
  (h1 : 14 * x + 35 = 77) : 
  x = 3 := 
by 
  sorry

end Nathan_daily_hours_l295_295899


namespace question_count_and_score_l295_295659

namespace HomeworkProblem

variables (x y z : ℕ)

theorem question_count_and_score :
  (x + y + z = 100) ∧ (0.5 * ↑x + 3 * ↑y + 10 * ↑z = 100) →
  (x = 80 ∧ y = 20 ∧ z = 0) :=
begin
  intros h,
  let h1 := h.1,
  let h2 := h.2,
  sorry
end

end HomeworkProblem

end question_count_and_score_l295_295659


namespace population_96_percent_l295_295254

theorem population_96_percent (total_population : ℕ) (percent : ℚ) (result : ℕ)
  (H1 : total_population = 24000)
  (H2 : percent = 0.96) :
  result = total_population * percent :=
by
  have H3 : result = total_population * percent,
    by sorry,
  exact H3

end population_96_percent_l295_295254


namespace g_of_3_over_8_l295_295191

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l295_295191


namespace total_splash_width_l295_295604

theorem total_splash_width :
  let pebbles_splash_width := 1 / 4
  let rocks_splash_width := 1 / 2
  let boulders_splash_width := 2
  let num_pebbles := 6
  let num_rocks := 3
  let num_boulders := 2
  let total_width := num_pebbles * pebbles_splash_width + num_rocks * rocks_splash_width + num_boulders * boulders_splash_width
  in total_width = 7 := by
  sorry

end total_splash_width_l295_295604


namespace probability_OC_less_than_half_AB_l295_295965

-- Definition of the problem's conditions
def r : ℝ := 1  -- Assume r is given as 1 for simplicity

def uniformly_random_points_on_circle (A B : ℝ) : Prop := 
  A ∈ set.Icc (0:ℝ) (2*real.pi) ∧ 
  B ∈ set.Icc (0:ℝ) (2*real.pi)

def uniformly_random_point_in_circle (C: ℝ × ℝ) (r: ℝ) : Prop :=
  C.1 * C.1 + C.2 * C.2 ≤ r * r

-- Definition of the distance function
def distance_from_center_O (C: ℝ × ℝ) : ℝ := 
  real.sqrt (C.1 * C.1 + C.2 * C.2)

def chord_length (θ: ℝ) (r: ℝ) : ℝ :=
  2 * r * real.sin(θ / 2)

def probability_density_distance_OC (s: ℝ) (r: ℝ) : ℝ :=
  (2 * s) / (r * r)

-- Main theorem statement
theorem probability_OC_less_than_half_AB {r: ℝ} 
  (hA: ∀ θ_A, θ_A ∈ set.Icc (0:ℝ) (2*real.pi))
  (hB: ∀ θ_B, θ_B ∈ set.Icc (0:ℝ) (2*real.pi))
  (hC: ∀ C, uniformly_random_point_in_circle C r) :
  (let θ_A := (2 * real.pi : ℝ)
       θ_B := (2 * real.pi : ℝ)
       AB := chord_length (θ_A - θ_B) r
       OC := distance_from_center_O in 
        ∀ (OC < AB / 2), true {prob = 0.5}
       ) :=
  sorry

end probability_OC_less_than_half_AB_l295_295965


namespace triangle_angle_third_angle_l295_295073

def is_semi_prime (n : ℕ) : Prop :=
  ∃ p q : ℕ, (nat.prime p) ∧ (nat.prime q) ∧ (n = p * q)

def is_triangle_angle (x : ℕ) : Prop :=
  x > 0 ∧ x < 180

theorem triangle_angle_third_angle
  (p q : ℕ)
  (h_semi_prime_p : is_semi_prime p)
  (h_semi_prime_q : is_semi_prime q)
  (h_smallest_semi_prime_q : q = 4)
  (h_p_eq_2q : p = 2 * q)
  (h_triangle_angle_p : is_triangle_angle p)
  (h_triangle_angle_q : is_triangle_angle q) :
  ∃ x : ℕ, is_triangle_angle x ∧ p + q + x = 180 ∧ x = 168 :=
by {
  sorry
}

end triangle_angle_third_angle_l295_295073


namespace gratuity_is_four_l295_295524

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end gratuity_is_four_l295_295524


namespace prove_f_5_eq_ln_5_l295_295366

variable (f : ℝ → ℝ)
variable (e : ℝ)
variable [exp : 𝓒 2 ℝ]

theorem prove_f_5_eq_ln_5 (h : ∀ (x : ℝ), f (exp x) = x) : f 5 = Real.log 5 :=
by
  sorry

end prove_f_5_eq_ln_5_l295_295366


namespace triangle_radius_relation_l295_295013

-- Parameters for triangle ABC
variables {A B C K L M X Y Z : Point}
variable {R r : ℝ}

-- Definitions based on the given conditions
def is_midpoint (P Q R : Point) : Prop := dist P R = dist P Q
def is_arc_midpoint (O : Point) (P Q R : Point) : Prop := dist O Q = dist O R

-- Assuming triangle ABC has the specified midpoints and arc midpoints
axiom midpoint_K : is_midpoint B C K
axiom midpoint_L : is_midpoint C A L
axiom midpoint_M : is_midpoint A B M

axiom arc_midpoint_X : is_arc_midpoint circumcenter B C X
axiom arc_midpoint_Y : is_arc_midpoint circumcenter C A Y
axiom arc_midpoint_Z : is_arc_midpoint circumcenter A B Z

-- Assuming circumradius and inradius
axiom circumradius : ℝ
axiom inradius : ℝ

-- The statement to be proven
theorem triangle_radius_relation : 
  r + dist K X + dist L Y + dist M Z = 2 * R := 
sorry

end triangle_radius_relation_l295_295013


namespace problem_statement_l295_295811

theorem problem_statement (a b : ℝ) (h : 3 * a - 2 * b = -1) : 3 * a - 2 * b + 2024 = 2023 :=
by
  sorry

end problem_statement_l295_295811


namespace problem_l295_295724

def f (z : ℂ) : ℂ :=
if z.im ≠ 0 then z^2 else -z^2

theorem problem:
  f (f (f (f (1 + 1 * Complex.I)))) = -256 :=
by {
  -- The steps to show the proof would go here, but in this case we leave it as sorry.
  sorry
}

end problem_l295_295724


namespace JillTotalTaxPercentage_l295_295904

noncomputable def totalTaxPercentage : ℝ :=
  let totalSpending (beforeDiscount : ℝ) : ℝ := 100
  let clothingBeforeDiscount : ℝ := 0.4 * totalSpending 100
  let foodBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let electronicsBeforeDiscount : ℝ := 0.1 * totalSpending 100
  let cosmeticsBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let householdBeforeDiscount : ℝ := 0.1 * totalSpending 100

  let clothingDiscount : ℝ := 0.1 * clothingBeforeDiscount
  let foodDiscount : ℝ := 0.05 * foodBeforeDiscount
  let electronicsDiscount : ℝ := 0.15 * electronicsBeforeDiscount

  let clothingAfterDiscount := clothingBeforeDiscount - clothingDiscount
  let foodAfterDiscount := foodBeforeDiscount - foodDiscount
  let electronicsAfterDiscount := electronicsBeforeDiscount - electronicsDiscount
  
  let taxOnClothing := 0.06 * clothingAfterDiscount
  let taxOnFood := 0.0 * foodAfterDiscount
  let taxOnElectronics := 0.1 * electronicsAfterDiscount
  let taxOnCosmetics := 0.08 * cosmeticsBeforeDiscount
  let taxOnHousehold := 0.04 * householdBeforeDiscount

  let totalTaxPaid := taxOnClothing + taxOnFood + taxOnElectronics + taxOnCosmetics + taxOnHousehold
  (totalTaxPaid / totalSpending 100) * 100

theorem JillTotalTaxPercentage :
  totalTaxPercentage = 5.01 := by
  sorry

end JillTotalTaxPercentage_l295_295904


namespace cauchy_schwarz_inequality_l295_295879

theorem cauchy_schwarz_inequality 
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (b : Fin n → ℝ)
  (h_pos_a : ∀ i, 0 < a i)
  (h_pos_b : ∀ i, 0 < b i)
  (h_sum : ∑ i, a i = ∑ i, b i) :

  (∑ i, (a i)^2 / (a i + b i)) ≥ (∑ i, a i) / 2 :=
sorry

end cauchy_schwarz_inequality_l295_295879


namespace total_cleaning_time_l295_295891

theorem total_cleaning_time (time_outside : ℕ) (fraction_inside : ℚ) (time_inside : ℕ) (total_time : ℕ) :
  time_outside = 80 →
  fraction_inside = 1 / 4 →
  time_inside = fraction_inside * time_outside →
  total_time = time_outside + time_inside →
  total_time = 100 :=
by
  intros hto hfi htinside httotal
  rw [hto, hfi] at htinside
  norm_num at htinside
  rw [hto, htinside] at httotal
  norm_num at httotal
  exact httotal

end total_cleaning_time_l295_295891


namespace problem1_problem2_binomial_distribution_X_expected_value_X_l295_295646

open ProbabilityTheory

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Definitions for conditions
def bag_initial : set (Fin 6) := {0, 1, 2, 3, 4, 5} -- 4 white balls (0-3), 2 black balls (4-5)
def is_black (b : Fin 6) : Prop := b = 4 ∨ b = 5

-- Problem 1: Ball is replaced
def draw_with_replacement (n : ℕ) : Measure (Fin 6) := MeasureTheory.Measure.map (λ (i : Fin n), i % 6) (MeasureTheory.Measure.prodMeasure (bag_initial n))

theorem problem1 : (draw_with_replacement 2).prob {b | is_black b = 4 ∨ is_black b = 5} = 1 / 3 := sorry

-- Problem 2: Ball is not replaced
def draw_without_replacement (draws : list (Fin 6)) (n : ℕ) : Measure (Fin n) :=
  MeasureTheory.Measure.map (λ (i : Fin n), i % (6 - draws.length)) (MeasureTheory.Measure.prodMeasure (bag_initial n))

theorem problem2 : (draw_without_replacement [0, 1, 2, 3] 2).prob {b | is_black b = 4 ∨ is_black b = 5} = 2 / 5 := sorry

-- Problem 3: Binomial distribution of black balls in three draws with replacement
def X_binomial_with_replacement : measure_theory.measure (fin (ℕ → fin 6)) := measure.map (λ xs, finset.filter (λ x, is_black x) (finset.univ.map ↑xs)).card (measure.prod (finset.pi finset.univ (λ _, measure.prodMeasure (bag_initial 6))))

theorem binomial_distribution_X : X_binomial_with_replacement = probability_theory.binomial 3 (1 / 3) := sorry

theorem expected_value_X : (expected_value X_binomial_with_replacement) = 1 := sorry

end problem1_problem2_binomial_distribution_X_expected_value_X_l295_295646


namespace odd_function_f_neg_x_l295_295762

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x else -(x^2 + 2 * x)

theorem odd_function_f_neg_x (x : ℝ) (hx : x < 0) :
  f x = -x^2 - 2 * x :=
by
  sorry

end odd_function_f_neg_x_l295_295762


namespace julia_james_same_number_l295_295853

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}

def probability_same_number : ℚ := (10:ℚ) / (100:ℚ)

lemma relatively_prime : Nat.gcd 1 10 = 1 := by
  sorry

theorem julia_james_same_number : ∃ (m n : ℕ), probability_same_number = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 11 :=
by
  use 1, 10
  split
  · exact rfl
  split
  · exact relatively_prime
  · exact rfl

end julia_james_same_number_l295_295853


namespace tom_age_l295_295602

theorem tom_age (c : ℕ) (h1 : 2 * c - 1 = tom) (h2 : c + 3 = dave) (h3 : c + (2 * c - 1) + (c + 3) = 30) : tom = 13 :=
  sorry

end tom_age_l295_295602


namespace highest_power_of_3_divides_consecutive_integers_l295_295168

def form_integer_from_range (start : Nat) (end : Nat) : Nat :=
  let rec aux (n : Nat) (acc : Nat) :=
    if n > end then 
      acc 
    else 
      aux (n + 1) (acc * 10^(Nat.log10 n + 1) + n)
  aux start 0

theorem highest_power_of_3_divides_consecutive_integers :
  ∃ k : Nat, form_integer_from_range 19 92 = N ∧ 3^k | N ∧ ¬ ∃ l > k, 3^l | N :=
sorry

end highest_power_of_3_divides_consecutive_integers_l295_295168


namespace olivia_spent_at_showroom_l295_295958

theorem olivia_spent_at_showroom :
  ∃ (x : ℕ), 
    let initial := 106 in
    let supermarket := 31 in
    let remaining := 26 in
    initial - supermarket - x = remaining ∧ x = 49 :=
begin
  sorry,
end

end olivia_spent_at_showroom_l295_295958


namespace horner_eval_f_at_5_eval_f_at_5_l295_295368

def f (x: ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_eval_f_at_5 :
  f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
  sorry

theorem eval_f_at_5 : f 5 = 2015 := by 
  have h : f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
    apply horner_eval_f_at_5
  rw [h]
  norm_num

end horner_eval_f_at_5_eval_f_at_5_l295_295368


namespace correct_statements_l295_295739

variable {ℝ : Type*} [LinearOrderedField ℝ]

def f (a b c x : ℝ) : ℝ := a * (x + b) * (x + c)
def g (a b c x : ℝ) : ℝ := x * (f a b c x)

theorem correct_statements (h : a ≠ 0) :
  (∃ (a b c : ℝ), (∀ x, (f a b c x = f (2*x) b c x) ∧ (g a b c x ≠ g (2*x) b c x))) ∧
  (∃ (a b c : ℝ), (∀ x, (f a b c x + g a b c x = 0) ∧ (f a b c x - g a b c x = 0))) :=
by
  sorry

end correct_statements_l295_295739


namespace total_female_officers_l295_295635

noncomputable def female_officers (total_officers_on_duty: ℕ) (percentage_female_on_duty: ℝ) (half_on_duty_female: ℕ) :=
  (half_on_duty_female : ℝ) / percentage_female_on_duty

theorem total_female_officers 
    (total_officers_on_duty: ℕ := 180)
    (percentage_female_on_duty: ℝ := 0.18)
    (half_on_duty_female: ℕ := 90) :
  female_officers total_officers_on_duty percentage_female_on_duty half_on_duty_female = 500 := 
by
  -- Constants from conditions
  have total_officers_on_duty := 180
  have percentage_female_on_duty := 0.18
  have half_on_duty_female := 90
  
  -- Equation to calculate total female officers
  have female_officers_calculated : ℝ := (half_on_duty_female : ℝ) / percentage_female_on_duty
  
  -- Assert the solution
  show female_officers_calculated = 500
  sorry

end total_female_officers_l295_295635


namespace problem_1_problem_2_l295_295744

def seq_a : ℕ → ℕ
| 0       := 0
| (n + 1) := seq_a 0 + (finset.range n).sum (λ i, seq_a i) + (n + 1)

def seq_b : ℕ → ℕ
| 0       := 1
| (n + 1) := (finset.range (n + 1)).sum (λ i, seq_b i) - (finset.range n).sum (λ i, seq_b i)

def T (n : ℕ) : ℕ := (finset.range n).sum (λ i, seq_b i)

def a_n := λ n, if n = 0 then 0 else 2^(n-1) - 1

def geometric_sequence (u : ℕ → ℕ) : Prop :=
∃ a r : ℕ, ∀ n, u n = a * r ^ n

theorem problem_1 : geometric_sequence (λ n, seq_a n + 1) := sorry

theorem problem_2 :
  ∀ m : ℝ,
    (∀ n : ℕ, (finset.range n).sum (λ k, seq_b k / (seq_a k + 1)) ≥ m - 9 / (2 + 2 * seq_a n))
    → m ≤ 61 / 16 := sorry

end problem_1_problem_2_l295_295744


namespace geometric_sequence_sum_l295_295107

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = a 1 * q) (h3 : a 3 = a 1 * q^2) 
  (h4 : 2 * a 1 = a 2 + a 3) :
  q = -2 ∧ (∑ i in Finset.range n, (i + 1) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9 := 
by
  sorry

end geometric_sequence_sum_l295_295107


namespace find_vector_P_l295_295488

noncomputable def vector_combinations (A B C D E P: Type) [AddCommGroup P] [Module ℝ P] :=
  let D := (2 • C + B) / 3
  let E := (2 / 3) • A + (1 / 3) • C
  ∃ (x y z : ℝ), 
    x • A + y • B + z • C = P ∧
    x + y + z = 1 ∧ 
    x = 1 / 2 ∧ y = 1 / 6 ∧ z = 1 / 3

theorem find_vector_P (A B C P: Type) [AddCommGroup P] [Module ℝ A]
  (D : P) (E : P):
  vector_combinations A B C D E P :=
by
  unfold vector_combinations
  sorry

end find_vector_P_l295_295488


namespace symmetric_center_monotonically_increasing_on_interval_l295_295028

def tan_fn (ω : ℝ) : ℝ → ℝ := λ x, Real.tan (2 * ω * x - Real.pi / 6)

noncomputable def sym_center (k : ℤ) : ℝ × ℝ := (k * Real.pi / 4 + Real.pi / 12, 0)

theorem symmetric_center (k : ℤ) 
  (hw : ω = 1) 
  : ∃ k ∈ Int, sym_center k = (k * Real.pi / 4 + Real.pi / 12, 0) :=
begin
  use k,
  split,
  exact Int.mem_set_of_eq k,
  rw [hw],
  exact rfl,
end

theorem monotonically_increasing_on_interval 
  (hω : ω = 1) 
  : ∀ x y, (Real.pi / 12 < x) ∧ (x < Real.pi / 3) ∧ (x ≤ y) ∧ (y < Real.pi / 3) →
  (tan_fn ω) x ≤ (tan_fn ω) y :=
begin
  rw hω,
  intros x y h,
  dsimp [tan_fn],
  sorry
end

end symmetric_center_monotonically_increasing_on_interval_l295_295028


namespace directed_line_points_relation_l295_295511

-- Definitions
variables {A B C D : Type} [linear_ordered_field A]
variables {a b c d : A}
variables (AC CB AD DB AB : A)
variables (h1 : AC = c) (h2 : CB = b - c) (h3 : AD = d) (h4 : DB = b - d)
variables (h_cond : AC / CB + AD / DB = 0)

-- Theorem statement
theorem directed_line_points_relation
  (h1 : ∀ x : A, AC = x) (h2 : ∀ y : A, CB = y - AC)
  (h3 : ∀ z : A, AD = z) (h4 : ∀ w : A, DB = w - AD)
  (h_cond : AC / CB + AD / DB = 0) :
  1 / AC + 1 / AD = 2 / AB :=
sorry

end directed_line_points_relation_l295_295511


namespace mass_percentage_Al_in_AlBr3_l295_295344

-- Define molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_Br : ℝ := 79.90

-- Calculate the molar mass of AlBr3
def molar_mass_AlBr3 : ℝ := molar_mass_Al + 3 * molar_mass_Br

-- Calculate the mass percentage of Al in AlBr3
def mass_percentage_Al := (molar_mass_Al / molar_mass_AlBr3) * 100

-- Prove that the mass percentage is approximately 10.11%
theorem mass_percentage_Al_in_AlBr3 : abs (mass_percentage_Al - 10.11) < 0.001 := 
by sorry

end mass_percentage_Al_in_AlBr3_l295_295344


namespace abs_lt_pairs_count_l295_295735

def A : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}

def num_abs_lt_pairs (s : Set ℤ) : ℕ :=
  Set.toFinset s
    .sum (λ a => Set.toFinset s
         .count (λ b => |a| < |b|))

theorem abs_lt_pairs_count :
  num_abs_lt_pairs A = 18 :=
by
  -- Problem conditions
  let A := {-3, -2, -1, 0, 1, 2, 3}
  have len_A : A.card = 7 := rfl

  -- Transforms the set into a finset for counting
  let finA := A.to_finset

  -- Count pairs (a, b) such that |a| < |b|
  let count := finA.sum (λ a => finA.count (λ b => |a| < |b|))

  -- Prove the count is 18
  show count = 18
  sorry

end abs_lt_pairs_count_l295_295735


namespace journey_time_l295_295258

-- Define the conditions
def total_distance := 250 -- km
def speed_40 := 40 -- kmph
def distance_40 := 148 -- km
def speed_60 := 60 -- kmph

-- Define the problem to be proved
theorem journey_time :
  (distance_40 / speed_40 + (total_distance - distance_40) / speed_60) = 5.4 :=
by
  sorry

end journey_time_l295_295258


namespace slope_condition_l295_295397

theorem slope_condition {m : ℝ} : 
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end slope_condition_l295_295397


namespace sum_of_roots_eqn_l295_295619

theorem sum_of_roots_eqn (a b c : ℝ) (h : a = 1 ∧ b = -16 ∧ c = 36) :
  let α := -b / a
  in α = 16 :=
by
  cases h with
  | intro ha hb_hc =>
    cases hb_hc with
    | intro hb hc =>
      rw [ha, hb]
      simp
      sorry

end sum_of_roots_eqn_l295_295619


namespace population_in_two_years_l295_295476

def initial_population : ℕ := 10000
def children_0 : ℕ := 3500
def adults_0 : ℕ := 5000
def seniors_0 : ℕ := 1500

def children_rate : ℚ := 0.10
def adults_rate : ℚ := 0.15
def seniors_rate : ℚ := 0.05

def population_after_two_years (initial: ℕ) (rate: ℚ) : ℕ :=
  (initial : ℚ) * (1 + rate)^2

def total_population (c a s : ℕ) : ℕ :=
  c + a + s

theorem population_in_two_years : 
  total_population (population_after_two_years children_0 children_rate) 
                   (population_after_two_years adults_0 adults_rate)
                   (population_after_two_years seniors_0 seniors_rate) = 12501 :=
by
  sorry

end population_in_two_years_l295_295476


namespace find_b_l295_295127

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def p1 : Point := ⟨-1, 3⟩
def p2 : Point := ⟨2, 7⟩

-- Define the direction vector calculation
def direction_vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

-- Define the scaling of the direction vector
def scale_vector (v : Point) (k : ℝ) : Point :=
  ⟨k * v.x, k * v.y⟩

-- Define the desired result
def desired_vector : Point := ⟨2, 8 / 3⟩

-- The theorem stating our proof problem
theorem find_b :
  let v := direction_vector p1 p2 in
  let scaled_v := scale_vector v (2 / v.x) in
  scaled_v = desired_vector :=
by
  -- Proof logic here (we use sorry to skip the proof step)
  sorry

end find_b_l295_295127


namespace find_a_find_tan_A_l295_295452

noncomputable def triangle_ABC (a b c A B C : ℝ) : Prop :=
  (a * Math.cos B = 1) ∧
  (b * Math.sin A = Math.sqrt 2) ∧
  (A - B = Real.pi / 4)

theorem find_a (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) : a = Math.sqrt 3 :=
by sorry

theorem find_tan_A (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) : Math.tan A = -3 - 2 * Math.sqrt 2 :=
by sorry

end find_a_find_tan_A_l295_295452


namespace smallest_positive_period_of_f_range_of_f_intervals_of_increase_of_f_l295_295776

noncomputable def f (x : ℝ) := (Real.sqrt 3) * (Real.sin x ^ 2 - Real.cos x ^ 2) + 2 * Real.sin x * Real.cos x

-- The smallest positive period of f(x) is π
theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

-- The range of f(x) in the interval [-π/3, π/3] is [-2, √3]
theorem range_of_f : ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3), f x ∈ Set.Icc (-2) (Real.sqrt 3) := sorry

-- The intervals of monotonic increase of f(x) are [kπ - π/12, kπ + 5π/12] for k ∈ ℤ
theorem intervals_of_increase_of_f : ∃ k : ℤ, Set.Icc (↑k * Real.pi - Real.pi / 12) (↑k * Real.pi + 5 * Real.pi / 12) ⊆ Set.preimage (f '' Set.univ) Set.Ioi := sorry

end smallest_positive_period_of_f_range_of_f_intervals_of_increase_of_f_l295_295776


namespace negation_of_square_positive_l295_295194

-- Conditions
def nat := ℕ
def square_positive (n : nat) : Prop := n^2 > 0

-- Question and correct answer
theorem negation_of_square_positive :
  ¬ (∀ n : nat, square_positive n) ↔ ∃ n : nat, n^2 ≤ 0 :=
begin
  sorry -- proof to be filled in
end

end negation_of_square_positive_l295_295194


namespace find_numbers_l295_295601

variables {x y z k l : ℝ}

theorem find_numbers (h1 : (x + y + z)^2 = x^2 + y^2 + z^2) (h2 : xy = z^2) : 
  (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0) :=
begin
  -- Proof is intentionally left out.
  sorry
end

end find_numbers_l295_295601


namespace Simon_total_treasures_l295_295917

-- Definitions of quantities as per provided conditions.
def sand_dollars : ℕ := 10
def glass (sand_dollars: ℕ) := 3 * sand_dollars
def seashells (glass: ℕ) := 5 * glass
def total_treasures (sand_dollars glass seashells: ℕ) := sand_dollars + glass + seashells

-- The main theorem statement proving the total number of treasures
theorem Simon_total_treasures :
  ∃ sand_dollars glass seashells,
    glass = 3 * sand_dollars ∧
    seashells = 5 * glass ∧
    total_treasures sand_dollars glass seashells = 190 :=
by
  use sand_dollars
  use glass sand_dollars
  use seashells (glass sand_dollars)
  simp [sand_dollars, glass, seashells, total_treasures]
  sorry

end Simon_total_treasures_l295_295917


namespace probability_fourth_term_integer_l295_295848

-- Define the initial conditions and rules for the sequence
def initial_term : ℕ := 8

def heads_step (n : ℕ) : ℕ :=
2 * n - 1

def tails_step (n : ℕ) (tails_count : ℕ) : ℕ :=
if tails_count = 1 ∨ tails_count = 3 then
  n / 2 - 1
else
  3 * n - 2

-- Define a function to calculate the fourth term under a sequence of flips
def fourth_term (flips : List Bool) : Rat :=
match flips with
| [f1, f2, f3] =>
  let a2 := if f1 then heads_step initial_term else tails_step initial_term 1
  let a3 := if f2 then heads_step a2 else tails_step a2 (if f1 = false then 2 else 1)
  if f3 then heads_step a3 else tails_step a3 (if f2 = false ∧ f1 = false then 3 else 1)
| _ => 0 -- Default case: incorrect number of flips provided

-- Define the main theorem
theorem probability_fourth_term_integer : 
  let outcomes := [fourth_term [true, true, true], fourth_term [true, true, false],
                    fourth_term [true, false, true], fourth_term [true, false, false],
                    fourth_term [false, true, true], fourth_term [false, true, false],
                    fourth_term [false, false, true]] in
  (outcomes.filter (λ x, x.denom = 1)).length = 4 ∧
  outcomes.length = 7 →
  (outcomes.filter (λ x, x.denom = 1)).length / outcomes.length = Rat.ofInt 4 / 7 := by
  sorry

end probability_fourth_term_integer_l295_295848


namespace maximal_cards_taken_l295_295206

theorem maximal_cards_taken (cards : Finset ℕ) (h_cards : ∀ n, n ∈ cards ↔ 1 ≤ n ∧ n ≤ 100)
                            (andriy_cards nick_cards : Finset ℕ)
                            (h_card_count : andriy_cards.card = nick_cards.card)
                            (h_card_relation : ∀ n, n ∈ andriy_cards → (2 * n + 2) ∈ nick_cards) :
                            andriy_cards.card + nick_cards.card ≤ 50 := 
sorry

end maximal_cards_taken_l295_295206


namespace common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295100

def is_geometric_sequence (a : ℕ → ℂ) (q : ℂ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_mean (a1 a2 a3 : ℂ) : Prop :=
  2 * a1 = a2 + a3

noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  (1 - (1 + 3 * n) * (-2 : ℤ)^n) / 9

theorem common_ratio_of_geometric_sequence (a : ℕ → ℂ) (q : ℂ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : q ≠ 1) 
  (h3 : arithmetic_mean (a 1) (a 2) (a 3)) : 
  q = -2 := 
sorry

theorem sum_of_first_n_terms_of_sequence (n : ℕ) 
  (a : ℕ → ℂ) 
  (h1 : is_geometric_sequence a (-2)) 
  (h2 : a 1 = 1) : 
  ∑ k in finset.range n, k * a k = sum_first_n_terms n := 
sorry

end common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295100


namespace divisible_by_5_l295_295357

-- Problem statement: For which values of \( x \) is \( 2^x - 1 \) divisible by \( 5 \)?
-- Equivalent Proof Problem in Lean 4.

theorem divisible_by_5 (x : ℕ) : 
  (∃ t : ℕ, x = 6 * t + 1) ∨ (∃ t : ℕ, x = 6 * t + 4) ↔ (5 ∣ (2^x - 1)) :=
by sorry

end divisible_by_5_l295_295357


namespace least_value_a_plus_b_l295_295447

theorem least_value_a_plus_b (a b : ℝ) (h : log 2 a + log 2 b ≥ 7) : a + b ≥ 16 * sqrt 2 :=
sorry

end least_value_a_plus_b_l295_295447


namespace y_minus_x_is_7_l295_295836

theorem y_minus_x_is_7 (x y : ℕ) (hx : x ≠ y) (h1 : 3 + y = 10) (h2 : 0 + x + 1 = 1) (h3 : 3 + 7 = 10) :
  y - x = 7 :=
by
  sorry

end y_minus_x_is_7_l295_295836


namespace percentage_increase_is_30_l295_295531

-- Define the conditions
def old_plan_cost : ℝ := 150
def new_plan_cost : ℝ := 195

-- Define the calculation for percentage increase
def calculate_increase (old_cost new_cost : ℝ) : ℝ :=
  ((new_cost - old_cost) / old_cost) * 100

-- State the theorem we want to prove
theorem percentage_increase_is_30 :
  calculate_increase old_plan_cost new_plan_cost = 30 :=
by
  -- This term will get the proof done.
  sorry

end percentage_increase_is_30_l295_295531


namespace solution_l295_295320

-- Define the vectors and their conditions
variables {u v : ℝ}

def vec1 := (3, -2)
def vec2 := (9, -7)
def vec3 := (-1, 2)
def vec4 := (-3, 4)

-- Condition: The linear combination of vec1 and u*vec2 equals the linear combination of vec3 and v*vec4.
axiom H : (3 + 9 * u, -2 - 7 * u) = (-1 - 3 * v, 2 + 4 * v)

-- Statement of the proof problem:
theorem solution : u = -4/15 ∧ v = -8/15 :=
by {
  sorry
}

end solution_l295_295320


namespace triangle_area_l295_295453

theorem triangle_area (a : ℝ) (angleC : ℝ) (tan_half_B : ℝ) 
  (h1 : a = 2) 
  (h2 : angleC = Real.pi / 4) 
  (h3 : tan_half_B = 1 / 2) :
  let b := 2 * tan (Real.arctan (1 / 2)) / (Real.sqrt 2 / 2) in
  let sinC := Real.sin (Real.pi / 4) in
  let area_ABC := (1 / 2) * a * b * sinC in
  area_ABC = 8 / 7 := 
sorry

end triangle_area_l295_295453


namespace solve_apples_problem_l295_295530

def apples_problem (marin_apples donald_apples total_apples : ℕ) : Prop :=
  marin_apples = 9 ∧ total_apples = 11 → donald_apples = 2

theorem solve_apples_problem : apples_problem 9 2 11 := by
  sorry

end solve_apples_problem_l295_295530


namespace isosceles_triangle_PQT_l295_295872

noncomputable def Circle (α : Type) := {p : α // true} -- Placeholder for circle definition

variables 
  {α : Type} [MetricSpace α]
  (A B C D E P Q T : α)
  (h_circle : Circle α)
  (h_on_circle : ∀ (X : α), X ∈ {A, B, C, D, E} → X ∈ h_circle)
  (h_eq_AB_BC : dist A B = dist B C)
  (h_eq_CD_DE : dist C D = dist D E)
  (h_intersections : ∃ P Q T,
    (AreIntersect (LineThrough A D) (LineThrough B E) = P) ∧
    (AreIntersect (LineThrough A C) (LineThrough B D) = Q) ∧
    (AreIntersect (LineThrough B D) (LineThrough C E) = T))

theorem isosceles_triangle_PQT :
  IsIsoscelesTriangle P Q T :=
by
  -- Proof goes here
  sorry

end isosceles_triangle_PQT_l295_295872


namespace cube_game_strategy_l295_295327

theorem cube_game_strategy (a : Fin 8 → ℝ) (h_nonneg : ∀ i, 0 ≤ a i) (h_sum : ∑ i, a i = 1) :
  ∃ i : Fin 8, ∃ F1 F2 F3 : Fin 6, i ∈ vertices_of_face F1 ∧ i ∈ vertices_of_face F2 ∧ i ∈ vertices_of_face F3 ∧ a i ≤ 1/6 :=
by
  sorry

end cube_game_strategy_l295_295327


namespace find_common_ratio_sum_first_n_terms_l295_295112

-- Definitions and conditions for Part (1)
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_mean_condition (a : ℕ → ℝ) : Prop :=
  2 * a 0 = a 1 + a 2

-- Theorem for Part (1)
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) (ham : arithmetic_mean_condition a) (hq : q ≠ 1) :
  q = -2 :=
sorry

-- Definitions and conditions for Part (2)
def sum_first_n_na_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) * a i

-- Theorem for Part (2)
theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) 
  (hg : is_geometric_sequence a (-2)) (ha1 : a 0 = 1) :
  sum_first_n_na_n a n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
sorry

end find_common_ratio_sum_first_n_terms_l295_295112


namespace sum_product_leq_l295_295003

variable {n : ℕ} [Fact (0 < n)]
variable {a b : ℝ} (x : Fin n → ℝ) [Fact (0 < a)] [Fact (a < b)]
variable (hx : ∀ i, a ≤ x i ∧ x i ≤ b)

theorem sum_product_leq : 
  (∑ i, x i) * (∑ i, (x i)⁻¹) ≤ (a + b)^2 * n^2 / (4 * a * b) := by
sorry

end sum_product_leq_l295_295003


namespace projection_vector_c_l295_295934

theorem projection_vector_c (c : ℝ) 
  (h : ((5:c) •ᵥ ⟨3, 2⟩ : ℝ × ℝ) •ᵥ 𝓁⟨3, 2⟩ = (1/13) •ᵥ ⟨3 ,2⟩) : 
  c = -7 := 
by
  sorry

end projection_vector_c_l295_295934


namespace sum_of_powers_twice_square_l295_295145

theorem sum_of_powers_twice_square (x y : ℤ) : 
  ∃ z : ℤ, x^4 + y^4 + (x + y)^4 = 2 * z^2 := by
  let z := x^2 + x * y + y^2
  use z
  sorry

end sum_of_powers_twice_square_l295_295145


namespace license_plates_count_l295_295169

-- Define Rotokas alphabet without repeating any dependencies or requiring the solution steps.
def rotokas_alphabet := ['A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U', 'V']

-- Define the problem-specific conditions.
def starts_with_I_or_U (plate: List Char) : Prop :=
  plate.head? = some 'I' ∨ plate.head? = some 'U'

def ends_with_A (plate: List Char) : Prop :=
  plate.reverse.head? = some 'A'

def does_not_contain_both_E_and_S (plate: List Char) : Prop :=
  ¬(plate.contains 'E' ∧ plate.contains 'S')

def no_repeats (plate: List Char) : Prop :=
  plate.nodup

-- Combine all the conditions
def valid_license_plate (plate: List Char) : Prop :=
  starts_with_I_or_U plate ∧ 
  ends_with_A plate ∧ 
  does_not_contain_both_E_and_S plate ∧ 
  no_repeats plate

/-- There are 672 possible valid license plates from the Rotokas alphabet. -/
theorem license_plates_count : ∃ (plates : Finset (List Char)), 
  plates.card = 672 ∧ 
  ∀ plate ∈ plates, valid_license_plate plate := 
sorry

end license_plates_count_l295_295169


namespace max_sets_produced_in_7_days_l295_295270

structure GroupProduction :=
  (shirts : ℕ)
  (trousers : ℕ)

def daily_production_capacity : List GroupProduction :=
  [{ shirts := 8, trousers := 10 },
   { shirts := 9, trousers := 12 },
   { shirts := 7, trousers := 11 },
   { shirts := 6, trousers := 7 }]

def total_sets (production : GroupProduction) (days : ℕ) : ℕ :=
  min (production.shirts * days) (production.trousers * days)

theorem max_sets_produced_in_7_days (productions : List GroupProduction) :
  (total_sets productions[3] 7 + total_sets productions[0] 0 + total_sets productions[1] 3 + total_sets productions[2] 7) = 125 :=
by
  sorry

end max_sets_produced_in_7_days_l295_295270


namespace circle_properties_l295_295570

/-- The endpoints of a diameter of a circle are (2, 1) and (8, 7). -/
variable (A B : ℝ × ℝ) 
          (hA : A = (2, 1)) 
          (hB : B = (8, 7))

/-- The midpoint of a segment with endpoints A and B. -/
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The distance between two points A and B. -/
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

/-- Prove that the center of the circle is (5, 4) and the length of the diameter is 6√2. -/
theorem circle_properties :
  midpoint A B = (5, 4) ∧ distance A B = 6 * real.sqrt 2 :=
by
  sorry

end circle_properties_l295_295570


namespace simplifed_trig_expression_is_zero_l295_295918

open Real

noncomputable def simplify_trig_expression : ℝ → ℝ :=
  λ x, sin (x + 60 * (π / 180)) + 2 * sin (x - 60 * (π / 180)) - sqrt 3 * cos (120 * (π / 180) - x)

theorem simplifed_trig_expression_is_zero (x : ℝ) : simplify_trig_expression x = 0 :=
by
  sorry

end simplifed_trig_expression_is_zero_l295_295918


namespace fraction_equality_l295_295051

variables {R : Type*} [Field R] {m n p q : R}

theorem fraction_equality 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 :=
sorry

end fraction_equality_l295_295051


namespace correct_negation_statement_l295_295318

def Person : Type := sorry

def is_adult (p : Person) : Prop := sorry
def is_teenager (p : Person) : Prop := sorry
def is_responsible (p : Person) : Prop := sorry
def is_irresponsible (p : Person) : Prop := sorry

axiom all_adults_responsible : ∀ p, is_adult p → is_responsible p
axiom some_adults_responsible : ∃ p, is_adult p ∧ is_responsible p
axiom no_teenagers_responsible : ∀ p, is_teenager p → ¬is_responsible p
axiom all_teenagers_irresponsible : ∀ p, is_teenager p → is_irresponsible p
axiom exists_irresponsible_teenager : ∃ p, is_teenager p ∧ is_irresponsible p
axiom all_teenagers_responsible : ∀ p, is_teenager p → is_responsible p

theorem correct_negation_statement
: (∃ p, is_teenager p ∧ ¬is_responsible p) ↔ 
  (∃ p, is_teenager p ∧ is_irresponsible p) :=
sorry

end correct_negation_statement_l295_295318


namespace find_common_ratio_sum_first_n_terms_l295_295095

variable (a : ℕ → ℝ) (q : ℝ) 

-- Condition: {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: a₁ is the arithmetic mean of a₂ and a₃
def arithmetic_mean (a : ℕ → ℝ) :=
  a 1 = (a 2 + a 3) / 2

-- Proposition 1: Find the common ratio q
theorem find_common_ratio (h1 : is_geometric_sequence a q) (h2 : q ≠ 1) (h3 : arithmetic_mean a) : 
  q = -2 :=
by sorry

-- Proposition 2: Find the sum of the first n terms of the sequence {n * a_n}, given a₁ = 1
def sequence_n_times_a (a : ℕ → ℝ) :=
  λ n, n * a n

def sum_of_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) f

def geom_sum (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n else (1 - q ^ n) / (1 - q)

theorem sum_first_n_terms (h1 : is_geometric_sequence a q) (h2 : q = -2) (h3 : a 1 = 1) (n : ℕ) :
  sum_of_sequence (sequence_n_times_a a) n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
by sorry

end find_common_ratio_sum_first_n_terms_l295_295095


namespace tom_monthly_fluid_intake_l295_295713

-- Define the daily fluid intake amounts
def daily_soda_intake := 5 * 12
def daily_water_intake := 64
def daily_juice_intake := 3 * 8
def daily_sports_drink_intake := 2 * 16
def additional_weekend_smoothie := 32

-- Define the weekdays and weekend days in a month
def weekdays_in_month := 5 * 4
def weekend_days_in_month := 2 * 4

-- Calculate the total daily intake
def daily_intake := daily_soda_intake + daily_water_intake + daily_juice_intake + daily_sports_drink_intake
def weekend_daily_intake := daily_intake + additional_weekend_smoothie

-- Calculate the total monthly intake
def total_fluid_intake_in_month := (daily_intake * weekdays_in_month) + (weekend_daily_intake * weekend_days_in_month)

-- Statement to prove
theorem tom_monthly_fluid_intake : total_fluid_intake_in_month = 5296 :=
by
  unfold total_fluid_intake_in_month
  unfold daily_intake weekend_daily_intake
  unfold weekdays_in_month weekend_days_in_month
  unfold daily_soda_intake daily_water_intake daily_juice_intake daily_sports_drink_intake additional_weekend_smoothie
  sorry

end tom_monthly_fluid_intake_l295_295713


namespace fraction_difference_l295_295247

theorem fraction_difference:
  let f1 := 2 / 3
  let f2 := 3 / 4
  let f3 := 4 / 5
  let f4 := 5 / 7
  (max f1 (max f2 (max f3 f4)) - min f1 (min f2 (min f3 f4))) = 2 / 15 :=
by
  sorry

end fraction_difference_l295_295247


namespace heat_needed_l295_295591

noncomputable def specificHeat (c0 : ℝ) (alpha : ℝ) (t : ℝ) : ℝ :=
  c0 * (1 + alpha * t)

def mass : ℝ := 3 -- kg
def c0 : ℝ := 200 -- J/(kg·°C)
def alpha : ℝ := 0.05 -- °C⁻¹
def t_initial : ℝ := 30 -- °C
def t_final : ℝ := 80 -- °C

def heat_transfer (m : ℝ) (c0 : ℝ) (alpha : ℝ) (t_initial : ℝ) (t_final : ℝ) : ℝ :=
  let deltaT := t_final - t_initial
  let c30 := specificHeat c0 alpha t_initial
  let c80 := specificHeat c0 alpha t_final
  let c_avg := (c30 + c80) / 2
  c_avg * m * deltaT / 1000 -- Convert J to kJ
  
theorem heat_needed :
  heat_transfer mass c0 alpha t_initial t_final = 112.5 := by
  sorry

end heat_needed_l295_295591


namespace sophie_saves_money_l295_295920

variable (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ)
variable (given_on_birthday : Bool)

noncomputable def money_saved_per_year (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ) : ℝ :=
  (loads_per_week * dryer_sheets_per_load * weeks_per_year / sheets_per_box) * cost_per_box

theorem sophie_saves_money (h_loads_per_week : loads_per_week = 4) (h_dryer_sheets_per_load : dryer_sheets_per_load = 1)
                           (h_weeks_per_year : weeks_per_year = 52) (h_cost_per_box : cost_per_box = 5.50)
                           (h_sheets_per_box : sheets_per_box = 104) (h_given_on_birthday : given_on_birthday = true) :
  money_saved_per_year 4 1 52 5.50 104 = 11 :=
by
  have h1 : loads_per_week = 4 := h_loads_per_week
  have h2 : dryer_sheets_per_load = 1 := h_dryer_sheets_per_load
  have h3 : weeks_per_year = 52 := h_weeks_per_year
  have h4 : cost_per_box = 5.50 := h_cost_per_box
  have h5 : sheets_per_box = 104 := h_sheets_per_box
  have h6 : given_on_birthday = true := h_given_on_birthday
  sorry

end sophie_saves_money_l295_295920


namespace tank_capacity_is_823_l295_295660

-- Define the conditions
def leak_empty_time := 6 -- hours
def inlet_rate_per_minute := 4 -- liters per minute
def tank_empty_time_with_inlet_open := 8 -- hours

noncomputable def capacity_of_tank : ℝ :=
  let C := 823 in
  C

-- Define the rates based on the conditions
def leak_rate (C : ℝ) : ℝ := C / leak_empty_time
def inlet_rate : ℝ := inlet_rate_per_minute * 60
def net_emptying_rate (C : ℝ) : ℝ := C / tank_empty_time_with_inlet_open

-- The main theorem to prove
theorem tank_capacity_is_823 : capacity_of_tank = 823 :=
by {
  -- define initial capacity
  let C := capacity_of_tank,
  -- set up equation: inlet rate - leak rate = net emptying rate
  have eq1 : inlet_rate - leak_rate C = net_emptying_rate C,
  -- expand the definitions
  from rfl,
  have calc1 : 240 - C / 6 = C / 8,
  -- solving the equation step-by-step
  sorry
}

end tank_capacity_is_823_l295_295660


namespace cosine_sum_identity_l295_295382

theorem cosine_sum_identity 
  (α : ℝ) 
  (h_sin : Real.sin α = 3 / 5) 
  (h_alpha_first_quad : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (Real.pi / 3 + α) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end cosine_sum_identity_l295_295382


namespace range_of_a_l295_295817

theorem range_of_a (t a : ℝ) : 
  (∃ x : ℝ, 4^x + a * 2^x + a + 1 = 0 ∧ x ∈ ℝ) ↔ 
  a ∈ set.Iic (2 - 2 * real.sqrt 2) :=
sorry

end range_of_a_l295_295817


namespace probability_odd_number_5_of_6_rolls_l295_295614

-- Prove that the probability of getting an odd number in exactly 5 out of 6 rolls of an 8-sided die is 3/32.
theorem probability_odd_number_5_of_6_rolls : 
  ∀ (n k : ℕ) (p : ℚ), 
  n = 6 → 
  k = 5 → 
  p = 1/2 → 
  (p ^ k * (1 - p) ^ (n - k) * nat.choose n k) = 3 / 32 := 
begin
  intros,
  rw [←nat.cast_choose, nat.choose_eq_factorial_div_factorial],
  have h1 : nat.factorial 6 = 720, by norm_num,
  have h2 : nat.factorial 5 = 120, by norm_num,
  have h3 : nat.factorial 1 = 1, by norm_num,
  rw [h1, h2, h3, nat.factorial_succ],
  norm_num,
end

end probability_odd_number_5_of_6_rolls_l295_295614


namespace factorization_a_squared_minus_3a_l295_295334

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3 * a = a * (a - 3) := 
by 
  sorry

end factorization_a_squared_minus_3a_l295_295334


namespace parallel_planes_condition_l295_295926

-- Define the properties and parallelism condition for planes
def plane (α : Type) := set α
def parallel (α β : Type) [plane α] [plane β] : Prop := ∀ (l : set α), l ∈ α → ∀ (m : set β), m ∈ β → parallel_line l m

theorem parallel_planes_condition
  (α β : Type) [plane α] [plane β] :
  (∀ (l : set α), l ∈ α → ∀ (m : set β), m ∈ β → parallel_line l m) ↔ any_line_parallel α β :=
sorry

end parallel_planes_condition_l295_295926


namespace pentagon_area_l295_295465

/-- Given a convex pentagon ABCDE where BE and CE are angle bisectors at vertices B and C 
respectively, with ∠A = 35 degrees, ∠D = 145 degrees, and the area of triangle BCE is 11, 
prove that the area of the pentagon ABCDE is 22. -/
theorem pentagon_area (ABCDE : Type) (angle_A : ℝ) (angle_D : ℝ) (area_BCE : ℝ)
  (h_A : angle_A = 35) (h_D : angle_D = 145) (h_area_BCE : area_BCE = 11) :
  ∃ (area_ABCDE : ℝ), area_ABCDE = 22 :=
by
  sorry

end pentagon_area_l295_295465


namespace propositions_analysis_l295_295773

theorem propositions_analysis :
  (¬ (∀ l α, (∀ x, (x ∈ α) ∧ (x ∈ l)) → l ∥ α)) ∧
  (∀ α β l P, (α ∩ β = l) ∧ (∀ Q, Q ∈ α → Q ∈ β = l) → (∀ x, x ∈ P → x.perpendicular l) → x.perpendicular β) ∧
  (¬ (∃ x, x > 3 ∧ ¬(x < 2))) ∧
  (∀ a : ℝ, (a < 2) → a² < 2 * a ∧ (∃ b, b < 2 ∧ ¬(b² < 2 * b))) :=
by 
  sorry

end propositions_analysis_l295_295773


namespace monotonicity_range_of_a_l295_295782

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * (1 - x)
noncomputable def f' (x a : ℝ) : ℝ := 1 / x - a

-- 1. Monotonicity discussion
theorem monotonicity (a x : ℝ) (h : 0 < x) : 
  (a ≤ 0 → ∀ x, 0 < x → f' x a > 0) ∧
  (a > 0 → (∀ x, 0 < x ∧ x < 1 / a → f' x a > 0) ∧ (∀ x, x > 1 / a → f' x a < 0)) :=
sorry

-- 2. Range of a for maximum value condition
noncomputable def g (a : ℝ) : ℝ := Real.log a + a - 1

theorem range_of_a (a : ℝ) : 
  (0 < a) ∧ (a < 1) ↔ g a < 0 :=
sorry

end monotonicity_range_of_a_l295_295782


namespace count_multiples_l295_295806

theorem count_multiples (n : ℕ) :
  let multiples_of_5 := n / 5
  let multiples_of_6 := n / 6
  let multiples_of_15 := n / 15
  let total := multiples_of_5 + multiples_of_6 - multiples_of_15
  total = 900 ↔ n = 3000 :=
begin
  intros, 
  let multiples_of_5 := 3000 / 5,
  let multiples_of_6 := 3000 / 6,
  let multiples_of_15 := 3000 / 15,
  let total := multiples_of_5 + multiples_of_6 - multiples_of_15,
  exact total = 900,
  sorry
end

end count_multiples_l295_295806


namespace fib_identity_1_fib_identity_2_l295_295630

def fib : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+3) := fib(n+2) + fib(n+1)

theorem fib_identity_1 (n : ℕ) : fib (2*n + 1) * fib (2*n - 1) = fib (2*n)^2 + 1 := 
sorry

theorem fib_identity_2 (n : ℕ) : fib (2*n + 1)^2 + fib (2*n - 1)^2 + 1 = 3 * fib (2*n + 1) * fib (2*n - 1) := 
sorry

end fib_identity_1_fib_identity_2_l295_295630


namespace gratuity_is_four_l295_295526

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end gratuity_is_four_l295_295526


namespace parabola_vertex_intercept_l295_295931

variable (a b c p : ℝ)

theorem parabola_vertex_intercept (h_vertex : ∀ x : ℝ, (a * (x - p) ^ 2 + p) = a * x^2 + b * x + c)
                                  (h_intercept : a * p^2 + p = 2 * p)
                                  (hp : p ≠ 0) : b = -2 :=
sorry

end parabola_vertex_intercept_l295_295931


namespace range_of_vector_length_l295_295757

open Real EuclideanGeometry

noncomputable def point (x y : ℝ) := (x, y)

def ellipse (a b : ℝ) (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def circle (h k r : ℝ) (x y : ℝ) := (x - h)^2 + (y - k)^2 = r^2

def moving_point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse 4 2sqrt3 P.1 P.2

def points_on_circle (M N : ℝ × ℝ) : Prop :=
  circle 2 0 1 M.1 M.2 ∧ circle 2 0 1 N.1 N.2 ∧ dist M N = sqrt(3)

theorem range_of_vector_length (P M N : ℝ × ℝ) (hP : moving_point_on_ellipse P) (hMN : points_on_circle M N) :
  ∃ L : set ℝ, L = { x | x ≥ 3 ∧ x ≤ 13 } ∧ abs (dist P M + dist P N) ∈ L :=
by sorry

end range_of_vector_length_l295_295757


namespace part1_part2_l295_295693

def A (x : ℝ) : Prop := x^2 + 2*x - 3 < 0
def B (x : ℝ) (a : ℝ) : Prop := abs (x + a) < 1

theorem part1 (a : ℝ) (h : a = 3) : (∃ x : ℝ, (A x ∨ B x a)) ↔ (∃ x : ℝ, -4 < x ∧ x < 1) :=
by {
  sorry
}

theorem part2 : (∀ x : ℝ, B x a → A x) ∧ (¬ ∀ x : ℝ, A x → B x a) ↔ 0 ≤ a ∧ a ≤ 2 :=
by {
  sorry
}

end part1_part2_l295_295693


namespace car_trip_eq_560_miles_l295_295651

noncomputable def car_trip_length (v L : ℝ) :=
  -- Conditions from the problem
  -- 1. Car travels for 2 hours before the delay
  let pre_delay_time := 2
  -- 2. Delay time is 1 hour
  let delay_time := 1
  -- 3. Post-delay speed is 2/3 of the initial speed
  let post_delay_speed := (2 / 3) * v
  -- 4. Car arrives 4 hours late under initial scenario:
  let late_4_hours_time := 2 + 1 + (3 * (L - 2 * v)) / (2 * v)
  -- Expected travel time without any delays is 2 + (L / v)
  -- Difference indicates delay of 4 hours
  let without_delay_time := (L / v)
  let time_diff_late_4 := (late_4_hours_time - without_delay_time = 4)
  -- 5. Delay 120 miles farther, car arrives 3 hours late
  let delay_120_miles_farther := 120
  let late_3_hours_time := 2 + delay_120_miles_farther / v + 1 + (3 * (L - 2 * v - 120)) / (2 * v)
  let time_diff_late_3 := (late_3_hours_time - without_delay_time = 3)

  -- Combining conditions to solve for L
  -- Goal: Prove L = 560
  L = 560 -> time_diff_late_4 ∧ time_diff_late_3

theorem car_trip_eq_560_miles (v : ℝ) : ∃ (L : ℝ), car_trip_length v L := 
by 
  sorry

end car_trip_eq_560_miles_l295_295651


namespace find_m_values_l295_295029

theorem find_m_values (m : ℝ) :
  (m^2 - 2 * m - 2 > 0) ∧ (m^2 + m - 1 > 0) → 
  m ∈ {x : ℝ | (x^2 - 2 * x - 2 > 0) ∧ (x^2 + x - 1 > 0) } :=
by {
  intros h,
  exact h,
  sorry
}

end find_m_values_l295_295029


namespace intersection_complementA_setB_l295_295882

noncomputable def setA : Set ℝ := { x | abs x > 1 }

noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

noncomputable def complementA : Set ℝ := { x | abs x ≤ 1 }

theorem intersection_complementA_setB : 
  (complementA ∩ setB) = { x | 0 ≤ x ∧ x ≤ 1 } := by
  sorry

end intersection_complementA_setB_l295_295882


namespace calculate_expression_l295_295616

theorem calculate_expression : 12 * ((1/3) + (1/4) + (1/6))⁻¹ = 16 := by 
  sorry

end calculate_expression_l295_295616


namespace Lou_shoes_price_l295_295527

/-- Lou's Fine Shoes pricing strategy applied over a week:
1. Start with the price on Tuesday.
2. Increase by 20% on Wednesday.
3. Decrease by 15% on Thursday.
4. Decrease by 10% on Friday.
Given the initial price on Tuesday is $50, the final price on Friday is $45.90. -/
theorem Lou_shoes_price: 
  let price_tuesday := 50 in
  let price_wednesday := price_tuesday * 1.20 in
  let price_thursday := price_wednesday * 0.85 in
  let price_friday := price_thursday * 0.90 in
  price_friday = 45.90 := 
by
  sorry

end Lou_shoes_price_l295_295527


namespace solve_notes_problem_l295_295271

def notes_problem : Prop :=
  ∃ (remaining_denomination : ℕ),
  let total_amount := 10350 in
  let total_notes := 36 in
  let notes_50 := 17 in
  let amount_from_50 := 50 * notes_50 in
  let remaining_amount := total_amount - amount_from_50 in
  let remaining_notes := total_notes - notes_50 in
  remaining_denomination = remaining_amount / remaining_notes

theorem solve_notes_problem : notes_problem :=
  sorry

end solve_notes_problem_l295_295271


namespace find_values_x_y_z_square_l295_295004

theorem find_values_x_y_z_square (x y z : ℤ) (h1 : x + y + z = 3) (h2 : x^3 + y^3 + z^3 = 3) : 
  ∃ (n ∈ ({3, 57} : set ℤ)), x^2 + y^2 + z^2 = n := by
  sorry

end find_values_x_y_z_square_l295_295004


namespace sin_five_pi_over_two_minus_alpha_l295_295810

theorem sin_five_pi_over_two_minus_alpha (α : ℝ) (h : cos (π + α) = -1/3) : sin (5 * π / 2 - α) = 1/3 :=
by
  sorry

end sin_five_pi_over_two_minus_alpha_l295_295810


namespace largest_pencil_package_l295_295804

theorem largest_pencil_package (a b c : ℕ) (ha : a = 36) (hb : b = 54) (hc : c = 72) : Int.gcd (Int.gcd a b) c = 18 :=
by
  rw [ha, hb, hc]
  -- Proof needed to resolve the GCD calculation
    sorry

end largest_pencil_package_l295_295804


namespace largest_prime_factor_1764_l295_295228

theorem largest_prime_factor_1764 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1764 ∧ ∀ q, Nat.Prime q → Nat.dvd q 1764 → q ≤ p :=
by
  have h : Nat.factors 1764 = [2, 2, 3, 3, 7, 7] := sorry  -- This is where we assume the factorization.
  have primes := List.map Nat.Prime (Nat.factors 1764)
  have max_prime := List.max primes
  use 7
  split
  · -- Prove that 7 is prime
    sorry
  split
  · -- Prove that 7 divides 1764
    sorry
  · -- Prove that 7 is the largest prime divisor
    sorry

end largest_prime_factor_1764_l295_295228


namespace projection_length_equal_l295_295018

noncomputable def length_of_projections (OA OB OC OP : ℝ) (proj_eq : OA = OB ∧ OB = OC ∧ OC = OP) : ℝ :=
  sorry

theorem projection_length_equal (OA OB OC : ℝ) (h1 : OA = OB) (h2 : OB = OC) (h3 : ∀ P, proj_eq O P = length_of_projections OA OB OC P):
  OA = 1 → OB = 1 → OC = 2 → length_of_projections 1 1 2 1 = 2 / 3 :=
sorry

end projection_length_equal_l295_295018


namespace count_valid_c_values_l295_295703

def valid_c_interval (c : ℤ) : Prop :=
  0 ≤ c ∧ c ≤ 2000

def is_valid_c (c : ℤ) : Prop :=
  ∃ (x : ℚ), 8 * ⌊x⌋ + 3 * ⌈x⌉ = c

theorem count_valid_c_values : 
  (∃ (count : ℤ), count = 364) ∧ 
  (∀ c, valid_c_interval c → is_valid_c c ↔ 
        (c % 11 = 0 ∨ (c - 3) % 11 = 0)) :=
by sorry

end count_valid_c_values_l295_295703


namespace find_x_l295_295723

theorem find_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end find_x_l295_295723


namespace gcd_factorial_l295_295930

theorem gcd_factorial (b : ℕ) : gcd ((b - 4)!) (gcd ((b - 1)!) (gcd ((b + 1)!) ((b + 6)!))) = 392040 ↔ b = 17 :=
sorry

end gcd_factorial_l295_295930


namespace alice_savings_third_month_l295_295847

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end alice_savings_third_month_l295_295847


namespace sofa_love_seat_ratio_l295_295278

theorem sofa_love_seat_ratio (L S: ℕ) (h1: L = 148) (h2: S + L = 444): S = 2 * L := by
  sorry

end sofa_love_seat_ratio_l295_295278


namespace smallest_positive_integer_has_properties_l295_295618

def contains_digit_3 (n : ℕ) : Prop :=
  n.to_digits.contains 3

def is_terminating (n : ℕ) : Prop :=
  ∃ a b, n = 2^a * 5^b

noncomputable def smallest_n : ℕ :=
  Nat.find (λ n, n > 0 ∧ is_terminating n ∧ contains_digit_3 n ∧ ¬(n % 3 = 0))

theorem smallest_positive_integer_has_properties : smallest_n = 32 :=
  by {
    -- The proof would be provided here.
    sorry
  }

end smallest_positive_integer_has_properties_l295_295618


namespace sum_first_two_b_l295_295016

variable {a : ℕ → ℤ} {b : ℕ → ℤ}
variable (a_2 : a 2 = 5)
variable (S_4 : ∑ i in Finset.range (4 + 1), a (i + 1) = 28)
variable (b_def : ∀ n, b n = (-1)^n * a n)

theorem sum_first_two_b (T_2 : b 1 + b 2 = 3) : T_2 = 3 :=
by
  sorry

end sum_first_two_b_l295_295016


namespace algebraic_expression_transformation_l295_295448

theorem algebraic_expression_transformation (a b : ℝ) (h : ∀ x : ℝ, x^2 - 6*x + b = (x - a)^2 - 1) : b - a = 5 :=
by
  sorry

end algebraic_expression_transformation_l295_295448


namespace a_1_S_2_recurrence_gen_formula_a_T_n_bounds_l295_295883

-- Define the sequence and its properties
def seq_a (n : ℕ) : ℕ := if n = 0 then 0 else 2 ^ n
def sum_a (n : ℕ) : ℕ := Finset.sum (Finset.range n) seq_a

-- Conditions
theorem a_1 : seq_a 1 = 2 := by rfl
theorem S_2 : sum_a 2 = 6 := by simp [sum_a, seq_a]; norm_num

-- Recurrence relation
theorem recurrence (n : ℕ) (hn : 2 ≤ n) :
  sum_a (n + 1) = 3 * sum_a n - 2 * sum_a (n - 1) := sorry

-- Part (1): Prove the general formula for a_n is 2^n
theorem gen_formula_a (n : ℕ) : seq_a n = 2 ^ n := by rfl

-- Part (2): Define sequence b_n and its sum T_n
def seq_b (n : ℕ) : ℝ := (Real.log (seq_a n) / Real.log 2) / seq_a n
def sum_b (n : ℕ) : ℝ := Finset.sum (Finset.range n) seq_b

-- Prove the inequalities for T_n
theorem T_n_bounds (n : ℕ) : 1 / 2 ≤ sum_b n ∧ sum_b n < 2 := sorry

end a_1_S_2_recurrence_gen_formula_a_T_n_bounds_l295_295883


namespace lambda_value_l295_295380

universe u
variables {V : Type u} [inner_product_space ℝ V]

theorem lambda_value
  {a b : V}
  (h_a : ∥a∥ = 2)
  (h_b : ∥b∥ = 3)
  (angle_ab : real.angle_between a b = real.pi / 3)
  (perpendicular : inner (2 • a - b) (λ • a + 2 • b) = 0)
  : λ = 6 / 5 :=
sorry

end lambda_value_l295_295380


namespace river_current_speed_l295_295089

def karen_still_pond_speed := 10 -- Karen's speed in miles per hour on a still pond
def river_length := 12 -- River length in miles
def time_to_paddle_up := 2 -- Time to paddle up the river in hours

def effective_speed_against_current : ℝ :=
  river_length / time_to_paddle_up -- Effective speed in miles per hour

theorem river_current_speed :
  (effective_speed_against_current = karen_still_pond_speed - 4) :=
  by sorry

noncomputable def river_speed := karen_still_pond_speed - effective_speed_against_current

#eval river_speed -- This should evaluate to 4

end river_current_speed_l295_295089


namespace leo_kept_packs_l295_295501

axiom marbles_total : Nat := 8000
axiom marbles_per_pack : Nat := 20
axiom packs_total : Nat := marbles_total / marbles_per_pack
axiom packs_Manny : Nat := packs_total / 4
axiom packs_Neil : Nat := packs_total / 8
axiom packs_Paula : Nat := packs_total / 5
axiom packs_Bruce : Nat := packs_total / 10
axiom packs_Olivia : Nat := packs_total / 20
axiom packs_distributed : Nat := packs_Manny + packs_Neil + packs_Paula + packs_Bruce + packs_Olivia

theorem leo_kept_packs : (packs_total - packs_distributed) = 110 :=
by
  sorry

end leo_kept_packs_l295_295501


namespace min_matches_for_shortest_side_l295_295969

noncomputable def min_shortest_side (total_matches : ℕ) : ℕ :=
  let a := (total_matches / 7) in
  if 4 * a + (total_matches - 4 * a) = total_matches ∧ 2 * a < (total_matches - 4 * a) ∧ (total_matches - 4 * a) < 3 * a 
  then a else sorry

theorem min_matches_for_shortest_side 
    (a b : ℕ) (total_matches : ℕ)
    (h1 : total_matches = 120)
    (h2 : a < b ∧ b < 3 * a)
    (h3 : 2 * a < b ∧ b < 4 * a)
    (h4 : 4 * a + b = total_matches) :
    min_shortest_side total_matches = 18 :=
sorry

end min_matches_for_shortest_side_l295_295969


namespace sterling_auto_store_sales_time_l295_295294

theorem sterling_auto_store_sales_time :
  ∀ (cars_for_sale : ℕ) 
    (sales_professionals : ℕ) 
    (cars_per_salesperson_per_month : ℕ),
  cars_for_sale = 500 →
  sales_professionals = 10 →
  cars_per_salesperson_per_month = 10 →
  cars_for_sale / (sales_professionals * cars_per_salesperson_per_month) = 5 :=
by
  intros cars_for_sale sales_professionals cars_per_salesperson_per_month
  intro h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end sterling_auto_store_sales_time_l295_295294


namespace num_results_l295_295902

def numHoles : Nat := 8
def numDisplay : Nat := 3
def colors : Fin numHoles → Fin 2 := 
sorry -- This function assigns colors to holes and is to be defined as per conditions.

theorem num_results (h1 : numHoles = 8) 
                    (h2 : numDisplay = 3) 
                    (adj : ∀ i j, ¬ adjacent i j ) :
                    ∃ results, results = 160 :=
by
    sorry

end num_results_l295_295902


namespace starting_elevation_l295_295498

variable (rate time final_elevation : ℝ)
variable (h_rate : rate = 10)
variable (h_time : time = 5)
variable (h_final_elevation : final_elevation = 350)

theorem starting_elevation (start_elevation : ℝ) :
  start_elevation = 400 :=
  by
    sorry

end starting_elevation_l295_295498


namespace sophia_pushups_last_day_l295_295556

theorem sophia_pushups_last_day :
  ∃ a : ℕ, (∑ n in finset.range 7, a + 5 * n = 175) ∧ (a + 6 * 5 = 40) :=
by
  sorry

end sophia_pushups_last_day_l295_295556


namespace chord_length_l295_295068

noncomputable def polar_curve (theta : ℝ) : ℝ := 2 * √2 * Real.sin (theta - π / 4)

def parametric_line (t : ℝ) : ℝ × ℝ :=
(1 + 4 / 5 * t, -1 - 3 / 5 * t)

def cartesian_line (x y : ℝ) : Prop :=
3 * x + 4 * y + 1 = 0

theorem chord_length : 
  let center := (-1 : ℝ, 1 : ℝ) in
  let radius := √2 in
  let distance := (2 : ℝ) / 5 in
  2 * √(radius ^ 2 - distance ^ 2) = (2 : ℝ) * √46 / 5 :=
by sorry

end chord_length_l295_295068


namespace simplify_expression_l295_295553

theorem simplify_expression : |(-4 : Int)^2 - (3 : Int)^2 + 2| = 9 := by
  sorry

end simplify_expression_l295_295553


namespace total_expenditure_proof_l295_295458

-- Define the conditions
def pricePerGallonNC := 2.00
def gallonsBoughtNC := 10
def priceIncrementVA := 1.00
def gallonsBoughtVA := 10

-- Define the function to calculate total expenditure
def totalExpenditure (priceNC : ℝ) (gallonsNC : ℕ) (incrementVA : ℝ) (gallonsVA : ℕ) : ℝ :=
  (priceNC * gallonsNC) + ((priceNC + incrementVA) * gallonsVA)

-- Theorem statement
theorem total_expenditure_proof : 
  totalExpenditure pricePerGallonNC gallonsBoughtNC priceIncrementVA gallonsBoughtVA = 50.00 :=
by
  -- Proof will be provided here
  sorry

end total_expenditure_proof_l295_295458


namespace rectangle_area_from_square_area_and_proportions_l295_295673

theorem rectangle_area_from_square_area_and_proportions :
  ∃ (a b w : ℕ), a = 16 ∧ b = 3 * w ∧ w = Int.natAbs (Int.sqrt a) ∧ w * b = 48 :=
by
  sorry

end rectangle_area_from_square_area_and_proportions_l295_295673


namespace find_orange_shells_l295_295164

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end find_orange_shells_l295_295164


namespace parabola_focal_ratio_correct_l295_295372

noncomputable def parabola_focal_ratio (p : ℝ) (hp : p > 0) : ℝ :=
  let y_sq := λ x : ℝ, 2 * p * x in
  let line := λ x : ℝ, x - p / 2 in
  let roots := -- Solve for roots of the combined equation
    [3 + 2 * Real.sqrt 2 / 2, 3 - 2 * Real.sqrt 2 / 2] in -- x_A, x_B solved
  let x_A := roots.head in
  let x_B := roots.tail.head in
  (x_A + p / 2) / (x_B + p / 2)

theorem parabola_focal_ratio_correct (p : ℝ) (hp : p > 0) :
  parabola_focal_ratio p hp = 3 + 2 * Real.sqrt 2 :=
by sorry

end parabola_focal_ratio_correct_l295_295372


namespace factor_polynomial_l295_295333

theorem factor_polynomial :
  (λ x : ℝ, (x^2 + 3 * x + 2) * (x^2 + 7 * x + 12) + (x^2 + 5 * x - 6)) =
  (λ x : ℝ, (x^2 + 5 * x + 2) * (x^2 + 5 * x + 9)) :=
by sorry

end factor_polynomial_l295_295333


namespace parabola_line_intersection_k_l295_295036

theorem parabola_line_intersection_k (k : ℝ) :
  (∃ A B M N : ℝ × ℝ,
    (A.2 = 2 * A.1^2) ∧ (B.2 = 2 * B.1^2) ∧
    (A.2 = k * A.1 + 2) ∧ (B.2 = k * B.1 + 2) ∧
    (M = ( (A.1 + B.1)/ 2, (A.2 + B.2)/ 2 )) ∧
    (N = (M.1, 0)) ∧
    let NA := (A.1 - N.1, A.2) in
    let NB := (B.1 - N.1, B.2) in
    ((NA.1 * NB.1 + NA.2 * NB.2) = 0)) ↔ (k = 4 * real.sqrt 3 ∨ k = -4 * real.sqrt 3) :=
sorry

end parabola_line_intersection_k_l295_295036


namespace sixDigitIntegersCount_l295_295805

-- Define the digits to use.
def digits : List ℕ := [1, 2, 2, 5, 9, 9]

-- Define the factorial function as it might not be pre-defined in Mathlib.
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Calculate the number of unique permutations accounting for repeated digits.
def numberOfUniquePermutations : ℕ :=
  factorial 6 / (factorial 2 * factorial 2)

-- State the theorem proving that we can form exactly 180 unique six-digit integers.
theorem sixDigitIntegersCount : numberOfUniquePermutations = 180 :=
  sorry

end sixDigitIntegersCount_l295_295805


namespace loss_per_metre_eq_12_l295_295669

-- Definitions based on the conditions
def totalMetres : ℕ := 200
def totalSellingPrice : ℕ := 12000
def costPricePerMetre : ℕ := 72

-- Theorem statement to prove the loss per metre of cloth
theorem loss_per_metre_eq_12 : (costPricePerMetre * totalMetres - totalSellingPrice) / totalMetres = 12 := 
by sorry

end loss_per_metre_eq_12_l295_295669


namespace value_of_x_when_y_is_neg_4_l295_295050

theorem value_of_x_when_y_is_neg_4 : 
  ∀ (x y : ℤ), (4 * 2^x = 5^(y + 4)) → (y = -4) → (x = -2) :=
by
  intros x y h₁ h₂
  sorry

end value_of_x_when_y_is_neg_4_l295_295050


namespace find_common_ratio_sum_first_n_terms_l295_295098

variable (a : ℕ → ℝ) (q : ℝ) 

-- Condition: {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: a₁ is the arithmetic mean of a₂ and a₃
def arithmetic_mean (a : ℕ → ℝ) :=
  a 1 = (a 2 + a 3) / 2

-- Proposition 1: Find the common ratio q
theorem find_common_ratio (h1 : is_geometric_sequence a q) (h2 : q ≠ 1) (h3 : arithmetic_mean a) : 
  q = -2 :=
by sorry

-- Proposition 2: Find the sum of the first n terms of the sequence {n * a_n}, given a₁ = 1
def sequence_n_times_a (a : ℕ → ℝ) :=
  λ n, n * a n

def sum_of_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) f

def geom_sum (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n else (1 - q ^ n) / (1 - q)

theorem sum_first_n_terms (h1 : is_geometric_sequence a q) (h2 : q = -2) (h3 : a 1 = 1) (n : ℕ) :
  sum_of_sequence (sequence_n_times_a a) n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
by sorry

end find_common_ratio_sum_first_n_terms_l295_295098


namespace line_perpendicular_proof_l295_295151

variable {Line Plane : Type}
variable (a b : Line) (α β : Plane)
variable [IsParallel : Line → Plane → Prop]
variable [IsPerpendicular : Line → Plane → Prop]
variable [PlanePerpendicular : Plane → Plane → Prop]
variable [LinePerpendicular : Line → Line → Prop]

theorem line_perpendicular_proof
  (h1 : IsPerpendicular a α)
  (h2 : IsPerpendicular b β)
  (h3 : PlanePerpendicular α β) :
  LinePerpendicular a b := 
  sorry

end line_perpendicular_proof_l295_295151


namespace thirteenth_result_l295_295562

theorem thirteenth_result (a : ℕ → ℝ) (h_avg25: (∑ i in finset.range 25, a i) / 25 = 18)
  (h_avg12_first: (∑ i in finset.range 12, a i) = 120)
  (h_avg12_last: (∑ i in finset.range (24 - 11) 24, a i) = 240) :  
  a 12 = 90 :=
by
  sorry

end thirteenth_result_l295_295562


namespace perimeter_of_abcdefg_l295_295214

theorem perimeter_of_abcdefg
  (ABC_equilateral : Equilateral ABC)
  (ADE_equilateral : Equilateral ADE)
  (EFG_equilateral : Equilateral EFG)
  (D_divides_AC : divides AC D 1 3)
  (G_divides_AE : divides AE G 2 1)
  (AB_length : AB = 6) :
  perimeter_of_figure ABCDEFG = 20 := 
sorry

end perimeter_of_abcdefg_l295_295214


namespace intersection_point_on_circumcircle_l295_295745

-- Definitions of the conditions
variables {A B C : Type} [EuclideanGeometry A] {Γ : Circle} {S : Point}

-- Assume triangle ABC, circumcircle Γ, bisectors, and intersection point S
def is_triangle (A B C : Point) : Prop := 
  ¬(A = B) ∧ ¬(B = C) ∧ ¬(A = C)

def is_circumcircle (Γ : Circle) (A B C : Point) : Prop := 
  Γ.contains A ∧ Γ.contains B ∧ Γ.contains C

def is_internal_angle_bisector (S A B C : Point) : Prop := 
  S lies on the bisector of ∠BAC

def is_perpendicular_bisector (S B C : Point) : Prop := 
  S lies on the perpendicular bisector of segment BC

def lies_on_circumcircle (S : Point) (Γ : Circle) : Prop :=
  Γ.contains S

-- Main theorem statement
theorem intersection_point_on_circumcircle
  (A B C : Point) (Γ : Circle) (S : Point)
  (h_triangle : is_triangle A B C)
  (h_circumcircle : is_circumcircle Γ A B C)
  (h_internal_bisector : is_internal_angle_bisector S A B C)
  (h_perpendicular_bisector : is_perpendicular_bisector S B C) :
  lies_on_circumcircle S Γ :=
sorry

end intersection_point_on_circumcircle_l295_295745


namespace each_child_receives_1680_l295_295266

-- Definitions for conditions
def husband_weekly_savings : ℕ := 335
def wife_weekly_savings : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 6
def children : ℕ := 4

-- Total savings calculation
def husband_monthly_savings := husband_weekly_savings * weeks_in_month
def wife_monthly_savings := wife_weekly_savings * weeks_in_month
def total_monthly_savings := husband_monthly_savings + wife_monthly_savings
def total_savings := total_monthly_savings * months_saving
def half_savings := total_savings / 2
def amount_per_child := half_savings / children

-- The theorem to prove
theorem each_child_receives_1680 : amount_per_child = 1680 := 
by 
sorriesorry

end each_child_receives_1680_l295_295266


namespace angle_CAB_l295_295212

noncomputable def isosceles_triangle (A B C I : Type) [plane_geom.points A] [plane_geom.points B] [plane_geom.points C] 
  [eq (dist A B) (dist A C)] : Prop := 
isosceles A B C

theorem angle_CAB (A B C I : Type) [plane_geom.points A] [plane_geom.points B] [plane_geom.points C] 
  (isosceles_triangle A B C I) 
  (incenter A B C I)
  (angle_CIA_eq_130 : plane_geom.angle A I C = 130) 
  : plane_geom.angle A B C = 80 :=
sorry

end angle_CAB_l295_295212


namespace straight_line_cannot_intersect_all_segments_l295_295688

/-- A broken line in the plane with 11 segments -/
structure BrokenLine :=
(segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
(closed_chain : ∀ i : Fin 11, i.val < 10 → (segments ⟨i.val + 1, sorry⟩).fst = (segments i).snd)

/-- A straight line that doesn't contain the vertices of the broken line -/
structure StraightLine :=
(is_not_vertex : (ℝ × ℝ) → Prop)

/-- The main theorem stating the impossibility of a straight line intersecting all segments -/
theorem straight_line_cannot_intersect_all_segments (line : StraightLine) (brokenLine: BrokenLine) :
  ∃ i : Fin 11, ¬∃ t : ℝ, ∃ x y : ℝ, 
    brokenLine.segments i = ((x, y), (x + t, y + t)) ∧ 
    ¬line.is_not_vertex (x, y) ∧ 
    ¬line.is_not_vertex (x + t, y + t) :=
sorry

end straight_line_cannot_intersect_all_segments_l295_295688


namespace function_monotonic_a_geq_1_4_extreme_points_inequality_l295_295780

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - x + a * (Real.log x)

def derivative_f (x : ℝ) (a : ℝ) : ℝ := x - 1 + a / x

theorem function_monotonic_a_geq_1_4 (a : ℝ) :
  (∀ x > 0, derivative_f x a = 0 → False) → a ≥ 1 / 4 :=
  by sorry

theorem extreme_points_inequality (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 2 / 9)
  (h3 : x1 + x2 = 1) (h4 : x1 * x2 = a) (h5 : x1 < x2) :
  f x1 a / x2 > - 5 / 12 - 1 / 3 * Real.log 3 :=
  by sorry

end function_monotonic_a_geq_1_4_extreme_points_inequality_l295_295780


namespace orange_pill_cost_l295_295296

-- Define the constants from the problem
def total_days : ℕ := 21
def total_cost : ℕ := 804
def daily_pill_count : ℕ := 3
def two_pill_diff : ℕ := 2

-- Define the daily cost constraint
def daily_cost (y : ℝ) (b : ℝ) : Prop := 2 * b + y = (total_cost: ℝ) / total_days

-- Lean uses real numbers for prices and costs
theorem orange_pill_cost (y b : ℝ) (h_daily : daily_cost y b) (h_diff : y = b + two_pill_diff):
  y = 14.1 :=
by
  -- We know that the daily cost must satisfy the equation
  -- 3y - 4 = total_cost / total_days
  -- From the conditions we know that daily_cost holds
  have h_decimal : 2 * b + y = (total_cost : ℝ) / total_days, from h_daily,
  -- which simplifies from the problem we derived
  -- Let's solve for y
  sorry

end orange_pill_cost_l295_295296


namespace sin_double_angle_l295_295732

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Ioo 0 (↑(Real.pi) / 2)) (h2 : Real.sin α = 4 / 5) :
  Real.sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l295_295732


namespace a1_value_a2_value_arithmetic_sequence_l295_295398

namespace ProofProblem

-- Define the sequences S_n and a_n
def S (n : ℕ) : ℕ
def a (n : ℕ) : ℕ

axiom Sn_def (n : ℕ) (h : 0 < n) : S n = 2 * a n - 2 ^ n
axiom Sn_sum_def (n : ℕ) (h : 0 < n) : S n = ∑ i in finset.range n, a (i + 1)

-- Prove the necessary statements
theorem a1_value : a 1 = 2 := 
sorry

theorem a2_value : a 2 = 6 := 
sorry

theorem arithmetic_sequence : ∃ (d : ℕ), ∀ (n : ℕ) (h : 0 < n), (a (n + 1) / (2 ^ (n + 1))) - (a n / (2 ^ n)) = d :=
sorry

end ProofProblem

end a1_value_a2_value_arithmetic_sequence_l295_295398


namespace equilateral_triangle_max_side_length_l295_295991

theorem equilateral_triangle_max_side_length {A B C : Type} (ABC : equilateral_triangle A B C)
  (l : line) (dA dB dC : ℝ) (h₁ : dA = 39) (h₂ : dB = 35) (h₃ : dC = 13) :
  let s := side_length ABC
  in s = 58 * real.sqrt 3 :=
by
  let sum_distances := dA + dB + dC
  have h : sum_distances = 87, by simp [h₁, h₂, h₃]
  have altitude_eq_sum_distances : (real.sqrt 3 / 2) * s = sum_distances, by sorry -- from geometry properties
  sorry

end equilateral_triangle_max_side_length_l295_295991


namespace find_b_l295_295403

noncomputable def f (x : ℝ) : ℝ := x^3 + real.log(x + real.sqrt(x^2 + 1))

theorem find_b (h : ∀ a : ℝ, f(1 + a) + 1 + real.log(real.sqrt(2) + 1) < 0 ↔ a < -2) : b = -2 :=
begin
  sorry
end

end find_b_l295_295403


namespace bisect_angle_l295_295993

-- Definitions of the north pole, equidistant points on a great circle through the pole, and a point on the equator
def north_pole := (0, 0, 1 : ℝ)

def equidistant_points_on_great_circle_through_N (φ : ℝ) : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) :=
  let A := (Real.cos φ, 0, Real.sin φ)
  let B := (-Real.cos φ, 0, Real.sin φ)
  (A, B)

def point_on_equator (θ : ℝ) : ℝ × ℝ × ℝ :=
  (Real.cos θ, Real.sin θ, 0)

-- The proof problem statement in Lean 4
theorem bisect_angle (A B : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) :
  ∀ (N : ℝ × ℝ × ℝ), 
    N = north_pole →
    ∃ φ θ : ℝ, 
      A = (Real.cos φ, 0, Real.sin φ) ∧ B = (-Real.cos φ, 0, Real.sin φ) ∧ 
      C = (Real.cos θ, Real.sin θ, 0) ∧ 
        (∃ P : ℝ × ℝ × ℝ,
          P = point_on_equator θ ∧
          ∃ Q : (P.1 = C.1 ∧ P.2 = C.2 ∧ P.3 = 0),
          let angle_ACN := ∠(A, C, N),
            let angle_BCN := ∠(B, C, N)
          in angle_ACN = angle_BCN)
sorry

end bisect_angle_l295_295993


namespace angle_CMB_eq_126_l295_295608

-- Definitions based on the conditions provided
variable (A B C M : Type)
variables [IsoscelesTriangle A B C E AC BC] [AngleEq A C B 120] [AngleEq M A C 15] [AngleEq M C A 9]

-- Statement to prove
theorem angle_CMB_eq_126 : ∀ (A B C M : Type) [IsoscelesTriangle A B C E AC BC] [AngleEq A C B 120] [AngleEq M A C 15] [AngleEq M C A 9], 
  AngleEq B M C 126 := 
sorry

end angle_CMB_eq_126_l295_295608


namespace exist_nat_not_in_T1987_l295_295939

def T0 : Set ℕ := { n | ∃ k : ℕ, n = 2^k }
def Tp (p : ℕ) : Set ℕ → Set ℕ
| T_prev := T_prev ∪ { n | ∃ S ⊆ T_prev, S.Finite ∧ S.Nonempty ∧ n = S.Sum id }

def T : ℕ → Set ℕ
| 0     := T0
| (p+1) := Tp p (T p)

theorem exist_nat_not_in_T1987 : ∃ (n : ℕ), n ∉ T 1987 :=
sorry

end exist_nat_not_in_T1987_l295_295939


namespace categorize_numbers_l295_295715

-- Define the given set of numbers
def given_numbers : Set ℚ := {-3.5, | -0.4 |, - ( -3 : ℤ), 7/4, 0, -30, -0.15, -128, 20, -8 / 3}

-- Define the sets to be checked
def integer_set : Set ℤ := { (3 : ℤ), 0, -30, -128, 20 }
def negative_rational_numbers : Set ℚ := { -3.5, -30, -0.15, -128, -8 / 3 }
def positive_fractions : Set ℚ := { 0.4, 7 / 4 }

-- Prove the given numbers belong to the corresponding sets
theorem categorize_numbers :
  (∀ x ∈ {-3.5, -0.4, 3, 7/4, 0, -30, -0.15, -128, 20, -8 / 3},
   (x = 3 ∨ x = 0 ∨ x = -30 ∨ x = -128 ∨ x = 20 → x ∈ integer_set) ∧
   (x = -3.5 ∨ x = -30 ∨ x = -0.15 ∨ x = -128 ∨ x = -8 / 3 → x ∈ negative_rational_numbers) ∧
   (x = 0.4 ∨ x = 7 / 4 → x ∈ positive_fractions)) :=
sorry

end categorize_numbers_l295_295715


namespace monotonic_intervals_cos_2α_l295_295362

open Real

-- Defining the function f(x)
def f (x : ℝ) (ω : ℝ) : ℝ := sin (ω * x + π / 3) * sin (π / 3 - ω * x)

-- Given conditions
variable (ω : ℝ) (hω : 0 < ω)
variable (α : ℝ) (hα1 : π / 8 < α) (hα2 : α < 5 * π / 8)
variable (h_val : f (α - π / 8) ω = 11 / 20)

-- The proof statements

-- Statement 1: Monotonic increasing intervals
theorem monotonic_intervals : ∀ k : ℤ, f x 1 is monotonically increasing in the intervals [k * π - π / 2, k * π] sorry

-- Statement 2: Value of cos 2α
theorem cos_2α : cos (2 * α) = - (sqrt 2) / 10 := sorry

end monotonic_intervals_cos_2α_l295_295362


namespace find_k_l295_295860

variables (O A B C D : Type)
variables [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

def is_origin (v : O) : Prop := v = 0

variables (OA OB OC OD : O → A) (k : ℝ)

-- Given condition
def condition (O A B C D : Type) [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
  (OA : O → A) (OB : O → B) (OC : O → C) (OD : O → D) (k : ℝ) : Prop :=
  is_origin (2 • OA O + 3 • OB O - 4 • OC O + k • OD O)

-- Our target is to find k such that points are coplanar
theorem find_k (O A B C D : Type) [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
  (OA : O → A) (OB : O → B) (OC : O → C) (OD : O → D) (k : ℝ) (h : condition O A B C D OA OB OC OD k) :
  k = -1 :=
sorry

end find_k_l295_295860


namespace solve_for_y_l295_295433

-- Given conditions expressed as a Lean definition
def given_condition (y : ℝ) : Prop :=
  (y / 5) / 3 = 15 / (y / 3)

-- Prove the equivalent statement
theorem solve_for_y (y : ℝ) (h : given_condition y) : y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 :=
sorry

end solve_for_y_l295_295433


namespace odd_power_sum_divisible_l295_295612

theorem odd_power_sum_divisible (x y : ℤ) (n : ℕ) (h_odd : ∃ k : ℕ, n = 2 * k + 1) :
  (x ^ n + y ^ n) % (x + y) = 0 := 
sorry

end odd_power_sum_divisible_l295_295612


namespace min_turns_for_route_l295_295461

-- Define the number of parallel and intersecting streets
def num_parallel_streets := 10
def num_intersecting_streets := 10

-- Define the grid as a product of these two numbers
def num_intersections := num_parallel_streets * num_intersecting_streets

-- Define the minimum number of turns necessary for a closed bus route passing through all intersections
def min_turns (grid_size : Nat) : Nat :=
  if grid_size = num_intersections then 20 else 0

-- The main theorem statement
theorem min_turns_for_route : min_turns num_intersections = 20 :=
  sorry

end min_turns_for_route_l295_295461


namespace unique_two_digit_number_l295_295708

theorem unique_two_digit_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 99) : 
  (13 * n) % 100 = 42 → n = 34 :=
by
  sorry

end unique_two_digit_number_l295_295708


namespace g_three_eighths_l295_295188

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l295_295188


namespace perimeter_of_one_face_of_cube_l295_295198

def volume_of_cube := 125 -- volume in cm³
def side_length := real.root 3 volume_of_cube   -- cube root of the volume
def perimeter_of_face := 4 * side_length        -- perimeter of one face

theorem perimeter_of_one_face_of_cube : 
  perimeter_of_face = 20 := 
begin
  sorry
end

end perimeter_of_one_face_of_cube_l295_295198


namespace max_ab_l295_295395

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end max_ab_l295_295395


namespace problem_l295_295871

theorem problem (w x y z : ℕ) (h : 3^w * 5^x * 7^y * 11^z = 2310) : 3 * w + 5 * x + 7 * y + 11 * z = 26 :=
sorry

end problem_l295_295871


namespace office_assignment_l295_295996

/-- Assign four people to clean three offices, with at least one person assigned to each office. 
There are exactly 36 different possible assignments. -/
theorem office_assignment : 
  let people := {1, 2, 3, 4}
  let offices := {A, B, C}
  ∃ (f : people → offices), (∀ o ∈ offices, ∃ p ∈ people, f p = o) ∧ (f.range.card = 3) → 
  (36 = 6 * 3 * 2) :=
by 
  let people := {1, 2, 3, 4}
  let offices := {A, B, C}
  have h_num_ways : (36 = 6 * 3 * 2) := by sorry
  exact ⟨f, fun H => sorry, rfl, h_num_ways⟩

end office_assignment_l295_295996


namespace total_dogs_barking_l295_295825

theorem total_dogs_barking 
  (initial_dogs : ℕ)
  (new_dogs : ℕ)
  (h1 : initial_dogs = 30)
  (h2 : new_dogs = 3 * initial_dogs) :
  initial_dogs + new_dogs = 120 :=
by
  sorry

end total_dogs_barking_l295_295825


namespace possible_values_of_C_l295_295799

variable {α : Type} [LinearOrderedField α]

-- Definitions of points A, B and C
def pointA (a : α) := a
def pointB (b : α) := b
def pointC (c : α) := c

-- Given condition
def given_condition (a b : α) : Prop := (a + 3) ^ 2 + |b - 1| = 0

-- Function to determine if the folding condition is met
def folding_number_line (A B C : α) : Prop :=
  (C = 2 * A - B ∨ C = 2 * B - A ∨ (A + B) / 2 = C)

-- Theorem to prove the possible values of C
theorem possible_values_of_C (a b : α) (h : given_condition a b) :
  ∃ C : α, folding_number_line (pointA a) (pointB b) (pointC C) ∧ (C = -7 ∨ C = 5 ∨ C = -1) :=
sorry

end possible_values_of_C_l295_295799


namespace isosceles_triangles_sum_of_angles_aligned_l295_295964

theorem isosceles_triangles_sum_of_angles_aligned :
  ∀ (X Y Z U V W : Point), 
  isosceles_triangle X Y Z → isosceles_triangle U V W →
  angle XYZ = 40 → angle UVW = 60 →
  base_angles_equal XY XZ → base_angles_equal UV UW →
  aligned_base_to_base X Y Z U V W →
  sum_of_angles VXZ XUV = 130 :=
by 
  intro X Y Z U V W h_XYZ_iso h_UVW_iso h_angle_XYZ h_angle_UVW h_base_eq_XYZ h_base_eq_UVW h_aligned
  sorry

end isosceles_triangles_sum_of_angles_aligned_l295_295964


namespace total_cleaning_time_is_100_l295_295894

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end total_cleaning_time_is_100_l295_295894


namespace part_a_l295_295249

theorem part_a (n : ℕ) : (∑ k in Finset.range n, (k + 1) * (n - k)) = n * (n + 1) * (n + 2) / 6 :=
sorry

end part_a_l295_295249


namespace find_angle_DP_CC1_l295_295081

-- Define the structure of the cube and points in it
structure Cube :=
  (A B C D A1 B1 C1 D1 : Point)
  (diagonal_BD1 : Line)
  (D1_on_diagonal : D1 ∈ diagonal_BD1)
  (P_on_diagonal : Point)
  (P ∈ diagonal_BD1)

-- Define the angle cos theta
def angle_cos (u v : Vector) : Real :=
  Vector.dot u v / (Vector.norm u * Vector.norm v)

def geometry_problem (cube : Cube) (angle_DPA : Real) : Prop :=
  -- Given conditions
  (P ∈ cube.diagonal_BD1 ∧ angle_DPA = 60)
  -- Conclusion
  → angle_cos (cube.P - cube.D) (cube.C1 - cube.C) = 45

-- Cube Geometry
def cube : Cube := sorry

-- The main theorem
theorem find_angle_DP_CC1 (cube : Cube) : geometry_problem cube 60 := 
by sorry

end find_angle_DP_CC1_l295_295081


namespace simplify_and_evaluate_l295_295551

theorem simplify_and_evaluate :
  ∀ (x : ℝ), x = -3 → 7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 :=
by
  intros x hx
  rw [hx]
  sorry

end simplify_and_evaluate_l295_295551


namespace cos_minus_sin_of_tan_l295_295737

theorem cos_minus_sin_of_tan (α : ℝ) (h1 : Real.tan α = 2) (h2 : π < α ∧ α < 3/2 * π) :
  Real.cos α - Real.sin α = √5 / 5 :=
sorry

end cos_minus_sin_of_tan_l295_295737


namespace marching_band_formations_l295_295272

theorem marching_band_formations :
  ∃ (s t : ℕ), s * t = 500 ∧ 10 ≤ t ∧ t ≤ 35 ∧ 25 ≤ s ∧ s ≤ 35 ∧ s.toReal / t.toReal ≤ 1 / 5 ∧ (∃! (s t : ℕ), s * t = 500 ∧ 10 ≤ t ∧ t ≤ 35 ∧ 25 ≤ s ∧ s ≤ 35 ∧ s.toReal / t.toReal ≤ 1 / 5) :=
begin
  sorry
end

end marching_band_formations_l295_295272


namespace volume_truncated_cone_l295_295951

-- Define the geometric constants
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height_truncated_cone : ℝ := 8

-- The statement to prove the volume of the truncated cone
theorem volume_truncated_cone :
  let V_large := (1/3) * Real.pi * (large_base_radius^2) * (height_truncated_cone + height_truncated_cone)
  let V_small := (1/3) * Real.pi * (small_base_radius^2) * height_truncated_cone
  V_large - V_small = (1400/3) * Real.pi :=
by
  sorry

end volume_truncated_cone_l295_295951


namespace clean_car_time_l295_295888

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l295_295888


namespace proof_problem_l295_295777

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 - 9*x - 1

-- Condition: x = -1 is an extremum point of f(x)
def extremum_point (a : ℝ) : Prop := 
  let f_prime := (λ x, 3*x^2 + 2*a*x - 9)
  f_prime (-1) = 0

-- Condition: function evaluation on the interval [-2, 5]
def eval_f_on_interval (a : ℝ) : Prop :=
  f (-2) -3 = 0 ∧ f (-1) = 4 ∧ f (3) = -28 ∧ f (5) = 4

-- Condition: no a exists such that f(x) is monotonically increasing on ℝ
def no_monotonic_increase (a : ℝ) : Prop :=
  ¬ ∃ a : ℝ, ∀ x : ℝ, (3*x^2 + 2*a*x - 9) > 0

-- Main theorem containing all conditions to prove each part.
theorem proof_problem :
  ∃ a : ℝ, extremum_point a ∧ eval_f_on_interval (-3) ∧ no_monotonic_increase a :=
by
  sorry

end proof_problem_l295_295777


namespace area_parallelogram_EFGH_l295_295373

structure Point :=
  (x : ℕ)
  (y : ℕ)

def E : Point := {x := 1, y := 3}
def F : Point := {x := 5, y := 3}
def G : Point := {x := 6, y := 1}
def H : Point := {x := 2, y := 1}

noncomputable def area_parallelogram (E F G H : Point) : ℕ :=
  let base := |F.x - E.x| in
  let height := |E.y - H.y| in
  base * height

theorem area_parallelogram_EFGH : area_parallelogram E F G H = 8 := by
  sorry

end area_parallelogram_EFGH_l295_295373


namespace find_common_ratio_sum_first_n_terms_l295_295114

-- Definitions and conditions for Part (1)
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_mean_condition (a : ℕ → ℝ) : Prop :=
  2 * a 0 = a 1 + a 2

-- Theorem for Part (1)
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) (ham : arithmetic_mean_condition a) (hq : q ≠ 1) :
  q = -2 :=
sorry

-- Definitions and conditions for Part (2)
def sum_first_n_na_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) * a i

-- Theorem for Part (2)
theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) 
  (hg : is_geometric_sequence a (-2)) (ha1 : a 0 = 1) :
  sum_first_n_na_n a n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
sorry

end find_common_ratio_sum_first_n_terms_l295_295114


namespace mod_57846_23_l295_295971

theorem mod_57846_23 : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ 57846 % 23 = n :=
by {
  use 1,
  split,
  { norm_num },  -- 0 ≤ 1
  split,
  { norm_num },  -- 1 < 23
  { norm_num },  -- 57846 % 23 = 1
  sorry
}

end mod_57846_23_l295_295971


namespace cos_alpha_beta_half_l295_295361

variables (α β : ℝ)

-- Conditions
axiom alpha_range : 0 < α ∧ α < π / 2
axiom beta_range : -π / 2 < β ∧ β < 0
axiom cos_alpha : cos (π / 4 + α) = 1 / 3
axiom cos_beta : cos (π / 4 - β / 2) = sqrt 3 / 3 

-- Proof Problem
theorem cos_alpha_beta_half :
  cos (α + β / 2) = 5 * sqrt 3 / 9 :=
sorry

end cos_alpha_beta_half_l295_295361


namespace sum_of_distances_constant_l295_295911

-- Define the triangle with vertex C, base AB, and sides AC and BC being equal
variables {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]

-- Height h from vertex C to base AB
def height (C : C) (AB : Set A) : ℝ := sorry

-- Distances h_a and h_b from point M on base AB to the lateral sides BC and AC
def distance_to_sides (M : M) (BC : Set B) (AC : Set B) : ℝ × ℝ := sorry

-- Prove that the sum of the distances from M to the sides is constant
theorem sum_of_distances_constant (C : C) (A B : A) (M : A)
    (is_isosceles : ∀ (AC BC : ℝ), AC = BC)
    (h : ∀ (AB : Set A), height C AB = height C AB)
    (dists : ∀ (BC AC : Set B), distance_to_sides M BC AC = distance_to_sides M BC AC) :
  ∃ const : ℝ, ∀ M : A, let (h_a, h_b) := distance_to_sides M BC AC in h_a + h_b = const := 
sorry

end sum_of_distances_constant_l295_295911


namespace real_solution_count_l295_295707

open Real

theorem real_solution_count :
    ∀ x : ℝ, (x ^ 2010 + 1) * (finset.range (1004 + 1)).sum (λ i, x ^ (2008 - 2 * i)) = 2010 * x ^ 2009 → x = 1 := 
by
  sorry

end real_solution_count_l295_295707


namespace new_rectangle_perimeters_l295_295223

theorem new_rectangle_perimeters {l w : ℕ} (h_l : l = 4) (h_w : w = 2) :
  (∃ P, P = 2 * (8 + 2) ∨ P = 2 * (4 + 4)) ∧ (P = 20 ∨ P = 16) :=
by
  sorry

end new_rectangle_perimeters_l295_295223


namespace ellipse_standard_form_line_through_point_E_intersects_ellipse_l295_295748

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def line_slope (k x : ℝ) : ℝ :=
  k * (x + 1)

noncomputable def line (k : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.2 = line_slope k p.1

theorem ellipse_standard_form :
  ∀ (x y : ℝ), ellipse 2 1 x y ↔ (x^2 / 4 + y^2 = 1) := by
  sorry

theorem line_through_point_E_intersects_ellipse :
  ∀ (l : ℝ → ℝ × ℝ → Prop) (k : ℝ),
    l k (-1, 0) ∧ 
    (∀ {A B : ℝ × ℝ}, A ≠ B → l k A ∧ l k B ∧ ellipse 2 1 A.fst A.snd ∧ ellipse 2 1 B.fst B.snd → |A.1 + 1| = 2 * |B.1 + 1|) →
    (l k = λ p, (sqrt 15 * p.1 + 6 * p.2 + sqrt 15 = 0) ∨ (sqrt 15 * p.1 - 6 * p.2 + sqrt 15 = 0)) :=
by
  sorry

end ellipse_standard_form_line_through_point_E_intersects_ellipse_l295_295748


namespace max_ab_value_l295_295393

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end max_ab_value_l295_295393


namespace meetings_percent_40_l295_295885

def percent_of_workday_in_meetings (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ) : ℕ :=
  (first_meeting_min + second_meeting_min + third_meeting_min) * 100 / (workday_hours * 60)

theorem meetings_percent_40 (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ)
  (h_workday : workday_hours = 10) 
  (h_first_meeting : first_meeting_min = 40) 
  (h_second_meeting : second_meeting_min = 2 * first_meeting_min) 
  (h_third_meeting : third_meeting_min = first_meeting_min + second_meeting_min) : 
  percent_of_workday_in_meetings workday_hours first_meeting_min second_meeting_min third_meeting_min = 40 :=
by
  sorry

end meetings_percent_40_l295_295885


namespace determine_n_sin_l295_295706

theorem determine_n_sin : ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ sin (n * real.pi / 180) = sin (-474 * real.pi / 180) ∧ n = 66 :=
begin
  sorry
end

end determine_n_sin_l295_295706


namespace log_product_eq_l295_295814

theorem log_product_eq (x y : ℝ) (log_x log_y : ℝ)
  (h1 : 2 * log_x + 5 * log_y = 2)
  (h2 : 3 * log_x + 2 * log_y = 2)
  (hx : log_x = log x)
  (hy : log_y = log y) :
  log (x * y) = 8 / 11 :=
by sorry

end log_product_eq_l295_295814


namespace smallest_number_contains_sequences_l295_295990

noncomputable def n : ℕ := 1212211122

def contains_substring (num seq : ℕ) : Prop :=
  num.toString.contains seq.toString

theorem smallest_number_contains_sequences :
  contains_substring n 121 ∧
  contains_substring n 1122 ∧
  contains_substring n 2122 ∧
  contains_substring n 2111 ∧
  contains_substring n 221 ∧
  contains_substring n 1212 ∧
  (∀ m : ℕ, (contains_substring m 121 ∧
             contains_substring m 1122 ∧
             contains_substring m 2122 ∧
             contains_substring m 2111 ∧
             contains_substring m 221 ∧
             contains_substring m 1212) → m ≥ n) :=
by sorry

end smallest_number_contains_sequences_l295_295990


namespace value_of_f7_l295_295021

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom symmetric_condition : ∀ x : ℝ, f(4 - x) = f(x)
axiom specific_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → f(x) = 2 * x^2

theorem value_of_f7 : f 7 = 2 := sorry

end value_of_f7_l295_295021


namespace sin_theta_value_l295_295792

theorem sin_theta_value (θ : ℝ) 
  (h1 : ∀ a b c d : ℝ, ∃ det : ℝ, det = a * d - b * c)
  (h2 : (sin (θ / 2) * sin (3 * θ / 2) - cos (θ / 2) * cos (3 * θ / 2)) = 1 / 2) :
  sin θ = sqrt 3 / 2 ∨ sin θ = - sqrt 3 / 2 := 
by  sorry

end sin_theta_value_l295_295792


namespace collinear_probability_in_grid_l295_295475

theorem collinear_probability_in_grid :
  let grid_size : ℕ := 5
  let total_dots : ℕ := grid_size * grid_size
  let number_of_lines : ℕ := 2 * grid_size -- 5 horizontal + 5 vertical lines
  let collinear_combinations : ℕ := number_of_lines * (grid_size - 1).choose 1
  let total_combinations : ℕ := total_dots.choose 4
  let probability : ℚ := collinear_combinations / total_combinations
  probability = 1 / 1265 := 
by
  let grid_size := 5
  let total_dots := grid_size * grid_size
  let number_of_lines := 2 * grid_size
  let collinear_combinations := number_of_lines * (grid_size - 1).choose 1
  let total_combinations := total_dots.choose 4
  let probability := collinear_combinations / total_combinations
  have : total_dots.choose 4 = 12650 := sorry -- Calculation step
  have : collinear_combinations = 10 := sorry -- Calculation step
  have : probability = 10 / 12650 := by
    rw [this, this]
  apply congr_arg
  norm_num

end collinear_probability_in_grid_l295_295475


namespace smallest_divisor_l295_295938
Import Mathlib

theorem smallest_divisor :
  ∃ d : ℕ, n > 1 ∧ d > 1 ∧ n = 175 ∧ n % d = 1 ∧ n % 7 = 1 ∧
      ∀ e : ℕ, (e > 1 ∧ n % e = 1) → (d ≤ e) :=
begin
  sorry
end

end smallest_divisor_l295_295938


namespace total_debt_l295_295084

theorem total_debt (paid_two_months_ago last_payment still_owes total_paid: ℕ) :
  paid_two_months_ago = 12 →
  last_payment = paid_two_months_ago + 3 →
  total_paid = paid_two_months_ago + last_payment →
  still_owes = 23 →
  (total_paid + still_owes) = 50 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  rfl

end total_debt_l295_295084


namespace solve_for_x_l295_295442

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end solve_for_x_l295_295442


namespace standard_circle_equation_of_tangent_circle_l295_295202

-- Define the center of the circle
def center : ℝ × ℝ := (-2, 3)

-- Define the tangent condition to the x-axis
def is_tangent_to_x_axis (c : ℝ × ℝ) (radius : ℝ) : Prop :=
  c.snd = radius

-- Define the standard equation of a circle
def circle_equation (c : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : ℝ :=
  (x + c.fst)^2 + (y - c.snd)^2 - radius^2

-- Proof problem: Given the conditions, prove the standard equation of the circle
theorem standard_circle_equation_of_tangent_circle : 
  ∃ r : ℝ, is_tangent_to_x_axis center r ∧ circle_equation center r x y = 0 :=
by
  -- Step 1: Given the center of the circle
  let c := center

  -- Step 2: The circle is tangent to the x-axis, thus the radius is 3
  exists 3

  -- Step 3: State the tangent condition 
  exact ⟨by simp [is_tangent_to_x_axis, center], by simp [circle_equation, center]⟩

  -- Sorry to skip the proof
  sorry

end standard_circle_equation_of_tangent_circle_l295_295202


namespace ratio_CD_AB_l295_295010

-- Definitions and conditions based on problem statement
def hyperbola1 (x : ℝ) : ℝ := 5 / x
def hyperbola2 (x : ℝ) : ℝ := 3 / x

def point_M : ℝ × ℝ := (1, 5) -- M lies on y = 5 / x, given as M(1, 5)
def point_A : ℝ × ℝ := (1, 0) -- A is the projection of M onto x-axis
def point_B : ℝ × ℝ := (0, 5) -- B is the projection of M onto y-axis
def point_C : ℝ × ℝ := (1, 3) -- C lies on y = 3 / x, sharing x-coordinate with A
def point_D : ℝ × ℝ := (3 / 5, 5) -- D lies on y = 3 / x, sharing y-coordinate with B

-- Distance calculations (geometrical setup is guaranteed by the problem conditions)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def CD : ℝ := distance point_C point_D
def AB : ℝ := distance point_A point_B

-- Theorem stating the ratio CD / AB is 2 / 5
theorem ratio_CD_AB : CD / AB = 2 / 5 :=
by {
  -- Proof omitted
  sorry
}

end ratio_CD_AB_l295_295010


namespace area_of_curves_l295_295342

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.sqrt x - x^2)

theorem area_of_curves :
  enclosed_area = 1 / 3 :=
sorry

end area_of_curves_l295_295342


namespace number_of_subsets_of_set_l295_295195

theorem number_of_subsets_of_set : 
  (set.powerset ({-1,0,1} : set ℤ)).to_finset.card = 8 := 
by
  sorry

end number_of_subsets_of_set_l295_295195


namespace pr_positive_given_mutation_l295_295071

variable (M T : Prop) [Probability M] [Probability T]

-- Conditions as Hypotheses/Assumptions
axiom Pr_M : ℝ := 1 / 200
axiom Pr_Mc : ℝ := 1 - Pr_M
axiom Pr_T_given_M : ℝ := 1
axiom Pr_T_given_Mc : ℝ := 0.05

-- Probabilities
def Pr_T : ℝ := Pr_T_given_M * Pr_M + Pr_T_given_Mc * Pr_Mc

-- Target
def Pr_M_given_T : ℝ := (Pr_T_given_M * Pr_M) / Pr_T

-- Proof Goal
theorem pr_positive_given_mutation :
  Pr_M_given_T = 20 / 219 := 
sorry

end pr_positive_given_mutation_l295_295071


namespace interval_of_increase_l295_295000

variable (a : ℝ) (x : ℝ)

-- Definition of the condition a ∈ {x | log₂(x) + x = 0}
def condition : Prop := Real.log 2 a + a = 0

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := Real.log a (x^2 - 2 * x - 3)

-- Problem Statement: Prove that the interval of increase for f(x) is (-∞, -1)
theorem interval_of_increase (h : condition a) : 
  ∀ x1 x2, x1 < x2 → x1 < -1 → x2 < -1 → f a x1 < f a x2 :=
by
  sorry

end interval_of_increase_l295_295000


namespace problem_solution_set_l295_295942

theorem problem_solution_set :
  {x : ℝ | (x - 1) ^ 2 * (x + 2) * (x - 3) ≤ 0} = set.Icc (-2 : ℝ) 3 :=
by
  sorry

end problem_solution_set_l295_295942


namespace ellipse_equation_emfn_area_l295_295771

-- Definitions
def ellipse_eq (a b x y : ℝ) : Prop := (a > b ∧ b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def right_focus (x y : ℝ) : Prop := (x = 1 ∧ y = 0)

def point_on_ellipse (x y : ℝ) : Prop := (x = 1 ∧ y = sqrt 2 / 2)

def product_of_slopes (k1 k2 : ℝ) : Prop := k1 * k2 = -1 / 2

def area_of_quadrilateral (area : ℝ) : Prop := area = 2 * sqrt 2

-- Theorem statements
theorem ellipse_equation (a b : ℝ) (P : ℝ × ℝ) :
  (ellipse_eq a b P.1 P.2) →
  (right_focus 1 0) →
  (point_on_ellipse P.1 P.2) →
  (a^2 = 2 ∧ b^2 = 1) :=
sorry

theorem emfn_area (area : ℝ) :
  (ellipse_eq (√2) 1 1 (√2 / 2)) →
  (right_focus 1 0) →
  (point_on_ellipse 1 (√2 / 2)) →
  (product_of_slopes k1 k2) →
  (area_of_quadrilateral 2√2) :=
sorry

end ellipse_equation_emfn_area_l295_295771


namespace probability_of_head_equal_half_l295_295828

def fair_coin_probability : Prop :=
  ∀ (H T : ℕ), (H = 1 ∧ T = 1 ∧ (H + T = 2)) → ((H / (H + T)) = 1 / 2)

theorem probability_of_head_equal_half : fair_coin_probability :=
sorry

end probability_of_head_equal_half_l295_295828


namespace find_common_ratio_sum_first_n_terms_l295_295113

-- Definitions and conditions for Part (1)
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_mean_condition (a : ℕ → ℝ) : Prop :=
  2 * a 0 = a 1 + a 2

-- Theorem for Part (1)
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) (ham : arithmetic_mean_condition a) (hq : q ≠ 1) :
  q = -2 :=
sorry

-- Definitions and conditions for Part (2)
def sum_first_n_na_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) * a i

-- Theorem for Part (2)
theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) 
  (hg : is_geometric_sequence a (-2)) (ha1 : a 0 = 1) :
  sum_first_n_na_n a n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
sorry

end find_common_ratio_sum_first_n_terms_l295_295113


namespace area_of_quadrilateral_ABCD_l295_295149

noncomputable def angle_ABC : ℝ := 90
noncomputable def angle_ACD : ℝ := 60
noncomputable def length_AC : ℝ := 24
noncomputable def length_CD : ℝ := 30
noncomputable def length_AE : ℝ := 6

theorem area_of_quadrilateral_ABCD 
  (angle_ABC : ℝ = 90)
  (angle_ACD : ℝ = 60)
  (length_AC : ℝ = 24)
  (length_CD : ℝ = 30)
  (length_AE : ℝ = 6) : 
  area_of_quadrilateral ABCD = 600 :=
sorry

end area_of_quadrilateral_ABCD_l295_295149


namespace minimum_abs_ab_l295_295768

theorem minimum_abs_ab (a b : ℝ) (h : (a^2) * (b / (a^2 + 1)) = 1) : abs (a * b) = 2 := 
  sorry

end minimum_abs_ab_l295_295768


namespace base_conversion_l295_295924

noncomputable def b_value : ℝ := Real.sqrt 21

theorem base_conversion (b : ℝ) (h : b = Real.sqrt 21) : 
  (1 * b^2 + 0 * b + 2) = 23 := 
by
  rw [h]
  sorry

end base_conversion_l295_295924


namespace shaded_area_half_l295_295854

section Regular13Gon

variables {A : Fin 13 → ℝ × ℝ}
variables (is_regular_13gon : (∀ i j : Fin 13, i ≠ j → dist (A i) (A j) = dist (A 0) (A 1)))

def intersects (p1 p2 q1 q2 : ℝ × ℝ) : ℝ × ℝ := sorry

parameter (A6A7_intersects_A8A9_at_B : ∃ B : ℝ × ℝ, B = intersects (A 5) (A 6) (A 7) (A 8))

theorem shaded_area_half : 
  ∃ B : ℝ × ℝ, A6A7_intersects_A8A9_at_B ∧ 
  (area_of_shaded_region A B = (1 / 2) * area_of_polygon A) :=
sorry

end Regular13Gon

end shaded_area_half_l295_295854


namespace each_child_receives_1680_l295_295267

-- Definitions for conditions
def husband_weekly_savings : ℕ := 335
def wife_weekly_savings : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 6
def children : ℕ := 4

-- Total savings calculation
def husband_monthly_savings := husband_weekly_savings * weeks_in_month
def wife_monthly_savings := wife_weekly_savings * weeks_in_month
def total_monthly_savings := husband_monthly_savings + wife_monthly_savings
def total_savings := total_monthly_savings * months_saving
def half_savings := total_savings / 2
def amount_per_child := half_savings / children

-- The theorem to prove
theorem each_child_receives_1680 : amount_per_child = 1680 := 
by 
sorriesorry

end each_child_receives_1680_l295_295267


namespace number_of_snowboarders_l295_295280

noncomputable def total_athletes (x : ℕ) : ℕ := 3 * x

noncomputable def cable_car_capacity : ℕ := 10

def athletes_with_cable_car (y : ℕ) (x : ℕ) : Prop := y ≤ cable_car_capacity ∧ y > 1.5 * x ∧ y < 1.65 * x

def athletes_on_own (x y : ℕ) : Prop := 3 * x - y > 0.45 * (3 * x) ∧ 3 * x - y < 0.5 * (3 * x)

theorem number_of_snowboarders : ∃ x : ℕ, (3 * x = total_athletes x) ∧ (x < 7) 
  ∧ ∃ y : ℕ, athletes_with_cable_car y x ∧ athletes_on_own x y ∧ x = 5 := 
begin
  sorry  
end

end number_of_snowboarders_l295_295280


namespace find_f8_l295_295573

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x * f y
axiom initial_condition : f 2 = 4

theorem find_f8 : f 8 = 256 := by
  sorry

end find_f8_l295_295573


namespace magnitude_of_angle_A_range_of_b_plus_c_l295_295064

--- Definitions for the conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition a / (sqrt 3 * cos A) = c / sin C
axiom condition1 : a / (Real.sqrt 3 * Real.cos A) = c / Real.sin C

-- Given a = 6
axiom condition2 : a = 6

-- Conditions for sides b and c being positive
axiom condition3 : b > 0
axiom condition4 : c > 0
-- Condition for triangle inequality
axiom condition5 : b + c > a

-- Part (I) Find the magnitude of angle A
theorem magnitude_of_angle_A : A = Real.pi / 3 :=
by
  sorry

-- Part (II) Determine the range of values for b + c given a = 6
theorem range_of_b_plus_c : 6 < b + c ∧ b + c ≤ 12 :=
by
  sorry

end magnitude_of_angle_A_range_of_b_plus_c_l295_295064


namespace measure_of_angle_y_is_90_degrees_l295_295078

-- Definitions
def parallel_lines (k l : line) : Prop := ∀ points x y, x ∈ k ∧ y ∈ l → ∃ (z : point), z ∈ k ∧ z ∈ l

def angle (A B C : point) (θ : ℝ) : Prop := ∀ (r : ray), r = ray.mk B C → measure_angle A B r = θ

-- Given conditions
variables {k l : line}
variable {A B C D F G H : point}
variable (θ1 θ2 θ3 : ℝ)

axiom parallel_k_l : parallel_lines k l
axiom angle_A_B_D_45_degrees : angle A B D 45
axiom angle_B_H_C_θ : angle B H C θ2
axiom angle_B_C_D_90_degrees : angle B C D 90

-- Statement to prove
theorem measure_of_angle_y_is_90_degrees : θ2 = 90 := sorry

end measure_of_angle_y_is_90_degrees_l295_295078


namespace shaded_area_of_larger_circle_l295_295066

theorem shaded_area_of_larger_circle (R r : ℝ) (A_larger A_smaller : ℝ)
  (hR : R = 9)
  (hr : r = 4.5)
  (hA_larger : A_larger = Real.pi * R^2)
  (hA_smaller : A_smaller = 3 * Real.pi * r^2) :
  A_larger - A_smaller = 20.25 * Real.pi := by
  sorry

end shaded_area_of_larger_circle_l295_295066


namespace emma_coins_l295_295712

theorem emma_coins (x : ℕ) (lost_fraction : ℚ) (found_coins : ℕ) (original_coins : ℕ) (final_coins : ℚ) :
  x = 60 →
  lost_fraction = 1 / 3 →
  found_coins = 28 →
  original_coins = 60 →
  final_coins = 68 →
  (original_coins - (original_coins * lost_fraction : ℚ) + found_coins) = final_coins ∧
  (final_coins / original_coins) = (17 / 15) :=
by
  intro H1 H2 H3 H4 H5
  rw [←H1, ←H4]
  have H6 : (original_coins * lost_fraction : ℚ) = 20 := 
    calc 
      original_coins * lost_fraction
      = 60 * (1 / 3) : by simp
      ... = 20 : by norm_num
  rw [H6] at H5
  rw [H1, H3, H4, H6]
  split
  { norm_num }
  { norm_num }

end emma_coins_l295_295712


namespace incenter_inside_BOH_l295_295477

variables {A B C H O I : Type}
variables [IsAcuteTriangle A B C] -- Assuming a typeclass or condition for an acute triangle
variables [IsOrthocenter H A B C] -- Assuming a typeclass or condition for H being the orthocenter
variables [IsCircumcenter O A B C] -- Assuming a typeclass or condition for O being the circumcenter
variables [IsIncenter I A B C] -- Assuming a typeclass or condition for I being the incenter
variables (angleOrder : ∠ C > ∠ B > ∠ A)

theorem incenter_inside_BOH (h : IsAcuteTriangle A B C) 
                            (ho : IsOrthocenter H A B C)
                            (hc : IsCircumcenter O A B C)
                            (hi : IsIncenter I A B C)
                            (angleOrder : ∠ C > ∠ B > ∠ A) :
  I ∈ triangle_interior O B H :=
sorry

end incenter_inside_BOH_l295_295477


namespace max_apples_guaranteed_l295_295131

theorem max_apples_guaranteed (a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) (h : list.pairwise (>) [a_1, a_2, a_3, a_4, a_5, a_6]) :
  ∃ (x_1 x_2 x_3 x_4 : ℤ), (∀ (s : ℤ), s ∈ [x_1 + x_2, x_1 + x_3, x_1 + x_4, x_2 + x_3, x_2 + x_4, x_3 + x_4] →
                          s = a_1 ∨ s = a_2 ∨ s = a_3 ∨ s = a_4 ∨
                          s ≥ a_5 ∨ s ≥ a_6) ∧ 14 = 
  (∑ s in [x_1 + x_2, x_1 + x_3, x_1 + x_4, x_2 + x_3, x_2 + x_4, x_3 + x_4], 
    if s = a_1 then 3 else if s = a_2 then 3 else if s = a_3 then 3 else if s = a_4 then 3 else if s ≥ a_5 ∨ s ≥ a_6 then 1 else 0) :=
by sorry

end max_apples_guaranteed_l295_295131


namespace M_minus_m_expression_l295_295786

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / a) * x^2 - 4 * x + 1

theorem M_minus_m_expression (a : ℝ) (h : a ≠ 0) :
  (∃ x_min x_max ∈ set.Icc (0 : ℝ) (1 : ℝ), f a x_max = f a x_min ∧ 
   ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), f a x_min ≤ f a x ∧ f a x ≤ f a x_max → 
  ((if a < 0 ∨ a > (1/2) then 4 - 1/a else (if 0 < a ∧ a ≤ (1/4) then 1/a + 4*a - 4 else 4*a)) = 
  ((f a x_max) - (f a x_min))) :=
  sorry

end M_minus_m_expression_l295_295786


namespace shortest_distance_proof_l295_295470

noncomputable def shortest_distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem shortest_distance_proof : 
  let A : ℝ × ℝ := (0, 250)
  let B : ℝ × ℝ := (800, 1050)
  shortest_distance A B = 1131 :=
by
  sorry

end shortest_distance_proof_l295_295470


namespace problem1_problem2_l295_295421

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def vector_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem problem1 (α β : ℝ) (h₁ : β = Real.pi / 4) :
  perpendicular (vector_a α + vector_b β) (vector_a α - vector_b β) := 
sorry

theorem problem2 (α : ℝ) (h₁ : α ∈ Set.Ioo (-Real.pi/4) (Real.pi/4)) (h₂ : β = Real.pi / 4)
  (h₃ : magnitude (vector_a α + vector_b β) = Real.sqrt (16/5)) :
  Real.sin α = -Real.sqrt(2) / 10 :=
sorry

end problem1_problem2_l295_295421


namespace zero_points_range_of_fB_l295_295797

-- Define the vectors and the function f(x)
def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
def f (x : ℝ) : ℝ := vec_m x.1 * vec_n x.1 - vec_m x.2 * vec_n x.2 

-- (I) Assert the zero points of f(x) in the interval [0, π]
theorem zero_points (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ Real.pi) : f x = 0 ↔ (x = Real.pi / 3) ∨ (x = Real.pi) := sorry

-- Define the cosine rule for the triangle and angles
def triangle_ABC (a b c : ℝ) (h : b ^ 2 = a * c) : ℝ → Prop := 
  λ B, Real.cos B >= 1 / 2

-- (II) Assert the range of f(B) given b^2 = ac in triangle ABC
theorem range_of_fB (a b c B : ℝ) (h₁ : b ^ 2 = a * c) (h₂ : 0 < B) (h₃ : B < Real.pi) : 
    f B ∈ Set.Icc (-1 : ℝ) 0 := sorry

end zero_points_range_of_fB_l295_295797


namespace abc_equilateral_if_a1b1c1_equilateral_l295_295490

-- Definition and conditions
variables (A B C A1 B1 C1 : Type) [field A] [field B] [field C] [field A1] [field B1] [field C1]

def is_triangle (A B C : Type) : Prop := sorry  -- Represents the condition that ABC forms a triangle

def altitude_from_A_intersects_BC_at_A1 (A B C A1 : Type) : Prop := sorry  -- Represents the altitude condition
def angle_bisector_from_B_intersects_AC_at_B1 (A B C B1 : Type) : Prop := sorry  -- Represents the angle bisector condition
def median_from_C_intersects_AB_at_C1 (A B C C1 : Type) : Prop := sorry  -- Represents the median condition

-- Equilateral condition for triangle A1B1C1
def equilateral_triangle (A1 B1 C1 : Type) : Prop := sorry

theorem abc_equilateral_if_a1b1c1_equilateral
  (hTriangle : is_triangle A B C)
  (hAltitude : altitude_from_A_intersects_BC_at_A1 A B C A1)
  (hAngleBisector : angle_bisector_from_B_intersects_AC_at_B1 A B C B1)
  (hMedian : median_from_C_intersects_AB_at_C1 A B C C1)
  (hEquilateralA1B1C1 : equilateral_triangle A1 B1 C1):
  equilateral_triangle A B C :=
sorry

end abc_equilateral_if_a1b1c1_equilateral_l295_295490


namespace number_of_sections_l295_295298

noncomputable def initial_rope : ℕ := 50
noncomputable def rope_for_art := initial_rope / 5
noncomputable def remaining_rope_after_art := initial_rope - rope_for_art
noncomputable def rope_given_to_friend := remaining_rope_after_art / 2
noncomputable def remaining_rope := remaining_rope_after_art - rope_given_to_friend
noncomputable def section_size : ℕ := 2
noncomputable def sections := remaining_rope / section_size

theorem number_of_sections : sections = 10 :=
by
  sorry

end number_of_sections_l295_295298


namespace angle_agh_90_deg_l295_295479

variables (A B C H G : Type)
noncomputable theory

structure Triangle (A B C : Type) :=
  (acute : True) -- To represent that triangle is acute-angled
  (AB_ne_AC : A ≠ C) -- AB != AC
  (orthocenter : H) -- H is the orthocenter
  (centroid : G) -- G is the centroid

variables [Triangle A B C] (S : Type → ℝ) -- Function to compute area

-- Given condition
axiom area_condition 
  (S_HAB S_HAC S_HBC : ℝ)
  (HAB_S : S_HAB = S (λ x : Type, True)) -- representing area S_HAB
  (HAC_S : S_HAC = S (λ x : Type, True)) -- representing area S_HAC
  (HBC_S : S_HBC = S (λ x : Type, True)) -- representing area S_HBC
  : (1 / S_HAB + 1 / S_HAC = 1 / S_HBC)

-- Theorem to prove
theorem angle_agh_90_deg
  (S_HAB S_HAC S_HBC : ℝ)
  (HAB_S : S_HAB = S (λ x : Type, True)) -- representing area S_HAB
  (HAC_S : S_HAC = S (λ x : Type, True)) -- representing area S_HAC
  (HBC_S : S_HBC = S (λ x : Type, True)) -- representing area S_HBC
  (area_cond : (1 / S_HAB + 1 / S_HAC = 1 / S_HBC)) :
  ∃(angle : ℝ), angle = 90 := 
sorry

end angle_agh_90_deg_l295_295479


namespace polynomial_coeffs_l295_295586

-- Define polynomial h
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4

-- Assume α, β, γ are roots of h
def α := some_root_of h
def β := some_root_of h
def γ := some_root_of h

-- Roots condition on α, β, γ from Vieta's formulas
axiom root_condition_1 : α + β + γ = 2
axiom root_condition_2 : α * β + β * γ + γ * α = 3
axiom root_condition_3 : α * β * γ = 4

-- Define polynomial p with squared roots of h
def p (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Final goal: Prove (a, b, c) = (-7, 14, -8)
theorem polynomial_coeffs : 
  ∃ (a b c : ℝ), 
    ({α^2, β^2, γ^2} = {root : ℝ | p root = 0}) ∧ 
    (a, b, c) = (-7, 14, -8) :=
by 
  sorry

end polynomial_coeffs_l295_295586


namespace range_f_not_R_g_has_odd_zeros_l295_295781

def f (x : ℝ) : ℝ :=
  if irrational x then x^2 else x

theorem range_f_not_R :
  ¬ ( ∀ y : ℝ, ∃ x : ℝ, f x = y ) := sorry

theorem g_has_odd_zeros (a : ℚ) (h : 0 < (a : ℝ)) :
  ∃ k : ℕ, k % 2 = 1 ∧ ∃! x : ℝ, f x - (a : ℝ) = 0 := sorry

end range_f_not_R_g_has_odd_zeros_l295_295781


namespace probability_2_4_8_before_odd_l295_295288

-- Definitions of probabilities and conditions for the die
def probability_odd : ℚ := 1 / 2
def probability_even : ℚ := 1 / 2
def probability_2_4_8_given_even : ℚ := 3 / 4

-- Main theorem to prove
theorem probability_2_4_8_before_odd :
  (∑ n in (finset.range 100 \ finset.range 4) ∋ n ≥ 4, (probability_odd ^ n * (3 ^ (n - 1) - 3 * 2 ^ (n - 1) + 3) / (4 ^ (n - 1)))) = 1 / 120 :=
by
  sorry

end probability_2_4_8_before_odd_l295_295288


namespace percentageMuslims_l295_295474

-- The conditions
def totalBoys : Nat := 850
def percentageHindus : ℝ := 0.28
def percentageSikhs : ℝ := 0.10
def otherCommunitiesBoys : Nat := 187

-- The theorem to prove
theorem percentageMuslims :
  let hindus := percentageHindus * totalBoys
  let sikhs := percentageSikhs * totalBoys
  let nonMuslimBoys := hindus + sikhs + otherCommunitiesBoys
  let muslimBoys := totalBoys - nonMuslimBoys
  let percentageMuslims := (muslimBoys / totalBoys) * 100
  percentageMuslims ≈ 40 :=
by
  sorry

end percentageMuslims_l295_295474


namespace gcd_three_digit_numbers_l295_295600

theorem gcd_three_digit_numbers (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) :
  ∃ k, (∀ n, n = 100 * a + 10 * b + c + 100 * c + 10 * b + a → n = 212 * k) :=
by
  sorry

end gcd_three_digit_numbers_l295_295600


namespace period_of_cos_3x_l295_295977

-- Define the function y = cos(3x)
def y (x : ℝ) : ℝ := Real.cos (3 * x)

-- Define the expected period
def expected_period : ℝ := 2 * Real.pi / 3

-- Prove that the period of the function y = cos(3x) is 2π/3
theorem period_of_cos_3x : ∃ P : ℝ, (P = expected_period) :=
by
  use expected_period
  sorry

end period_of_cos_3x_l295_295977


namespace find_y_l295_295369

-- Given the conditions
def x : ℤ := -2272
def z : ℤ := 1
def y (a b c : ℕ) : ℤ := 1000 + 100 * c + 10 * b + a

-- The hypothesis
def equation (a b c : ℕ) : Prop := a * x + b * y a b c + c * z = 1
def abc_condition (a b c : ℕ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b < c

-- The statement to be proved
theorem find_y (a b c : ℕ) (h_eq : equation a b c) (h_cond: abc_condition a b c) : y a b c = 1987 := 
sorry

end find_y_l295_295369


namespace integer_multiplication_l295_295972

theorem integer_multiplication :
  ∃ A : ℤ, (999999999 : ℤ) * A = (111111111 : ℤ) :=
by {
  sorry
}

end integer_multiplication_l295_295972


namespace number_of_cases_lt_abs_vals_l295_295733

open Set

def A : Set Int := {-3, -2, -1, 0, 1, 2, 3}

def countPairs (s : Set Int) : Int :=
  (s.toFinset.product s.toFinset).filter (λ p, Int.natAbs p.1 < Int.natAbs p.2).card

theorem number_of_cases_lt_abs_vals : countPairs A = 18 := 
  by
  sorry

end number_of_cases_lt_abs_vals_l295_295733


namespace range_of_a_plus_b_l295_295772

variable {a b : ℝ}

def has_two_real_roots (a b : ℝ) : Prop :=
  let discriminant := b^2 - 4 * a * (-4)
  discriminant ≥ 0

def has_root_in_interval (a b : ℝ) : Prop :=
  (a + b - 4) * (4 * a + 2 * b - 4) < 0

theorem range_of_a_plus_b 
  (h1 : has_two_real_roots a b) 
  (h2 : has_root_in_interval a b) 
  (h3 : a > 0) : 
  a + b < 4 :=
sorry

end range_of_a_plus_b_l295_295772


namespace A_finishes_work_in_9_days_l295_295650

noncomputable def B_work_rate : ℝ := 1 / 15
noncomputable def B_work_10_days : ℝ := 10 * B_work_rate
noncomputable def remaining_work_by_A : ℝ := 1 - B_work_10_days

theorem A_finishes_work_in_9_days (A_days : ℝ) (B_days : ℝ) (B_days_worked : ℝ) (A_days_worked : ℝ) :
  (B_days = 15) ∧ (B_days_worked = 10) ∧ (A_days_worked = 3) ∧ 
  (remaining_work_by_A = (1 / 3)) → A_days = 9 :=
by sorry

end A_finishes_work_in_9_days_l295_295650


namespace sufficient_condition_for_q_implies_a_range_l295_295118

theorem sufficient_condition_for_q_implies_a_range (a : ℝ)
  (p : Π (x : ℝ), |x - a| ≤ 1)
  (q : Π (x : ℝ), x^2 - 5 * x + 4 ≤ 0)
  (hpq : ∀ x, p x → q x) :
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end sufficient_condition_for_q_implies_a_range_l295_295118


namespace circle_diameter_given_area_l295_295974

theorem circle_diameter_given_area (A : ℝ) (hA : A = 196 * π) : ∃ D : ℝ, D = 28 ∧ 2 * (sqrt (A / π)) = D :=
by {
  sorry
}

end circle_diameter_given_area_l295_295974


namespace vertical_asymptote_exists_l295_295356

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (6 * x - 9)

theorem vertical_asymptote_exists : (∃ x : ℝ, 6 * x - 9 = 0 ∧ x = 3 / 2) :=
by 
  have h : 6 * (3 / 2) - 9 = 0 :=
    by norm_num
  use 3 / 2
  constructor
  · exact h
  · exact rfl

end vertical_asymptote_exists_l295_295356


namespace chocolate_cost_is_3_l295_295698

-- Definitions based on the conditions
def dan_has_5_dollars : Prop := true
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := cost_candy_bar + 1

-- Theorem to prove
theorem chocolate_cost_is_3 : cost_chocolate = 3 :=
by {
  -- This is where the proof steps would go
  sorry
}

end chocolate_cost_is_3_l295_295698


namespace sum_q_t_12_l295_295512

open Polynomial

/-- Let T be the set of 12-tuples (b_0, b_1, ..., b_11), where each entry is either 0 or 1.
For each 12-tuple t = (b_0, b_1, ..., b_11) in T, let q_t(x) be the polynomial of degree at most 11
such that q_t(n) = b_n for 0 ≤ n ≤ 11. We want to find the sum of q_t(12) for all t ∈ T. -/
theorem sum_q_t_12 : 
  let T := {t : fin 12 → bool | true},
      q_t (t : fin 12 → bool) : polynomial ℚ :=
        polynomial.interpolate (finset.univ.image (λ i : fin 12, (i.val, if t i then 1 else 0))) in
  ∑ t in T, (q_t t).eval 12 = 2048 :=
by 
  let T := {t : fin 12 → bool | true},
  let q_t (t : fin 12 → bool) : polynomial ℚ :=
    polynomial.interpolate (finset.univ.image (λ i : fin 12, (i.val, if t i then 1 else 0))),
  have h1 : ∀ t : fin 12 → bool, (q_t t).eval 12 = 2048, -- Proof omitted
  sorry

end sum_q_t_12_l295_295512


namespace liquid_X_percentage_l295_295128

theorem liquid_X_percentage
  (wA wB wC : ℝ) -- weights of solutions A, B, and C
  (pA pB pC : ℝ) -- percentages of liquid X in solutions A, B, and C
  (wA_val : wA = 500)
  (wB_val : wB = 700)
  (wC_val : wC = 300)
  (pA_val : pA = 0.008) -- 0.8% as a decimal
  (pB_val : pB = 0.018) -- 1.8% as a decimal
  (pC_val : pC = 0.025) -- 2.5% as a decimal) : 
  (total_weight : ℝ := wA + wB + wC)
  (total_X : ℝ := pA * wA + pB * wB + pC * wC)
  (percentage_X : ℝ := (total_X / total_weight) * 100) : 
  percentage_X = 1.61 := by
  rw [wA_val, wB_val, wC_val, pA_val, pB_val, pC_val]
  have h1 : total_weight = 1500 := by norm_num
  have h2 : total_X = 24.1 := by norm_num
  have h3 : total_X / total_weight = 0.0160666667 := by norm_num
  have h4 : (0.0160666667 * 100) = 1.61 := by norm_num
  exact h4
  sorry

end liquid_X_percentage_l295_295128


namespace k_less_than_half_plus_sqrt_two_n_l295_295122

theorem k_less_than_half_plus_sqrt_two_n
  (n k : ℕ) (S : set (ℝ × ℝ))
  (hS_card : S.card = n)
  (hS_no_collinear : ¬ ∃ a b c ∈ S, collinear {a, b, c})
  (hS_distances : ∀ P ∈ S, ∃ t : finset (ℝ × ℝ), t.card ≥ k ∧ ∀ Q ∈ t, dist P Q = dist P (some_elem t))
  :
  k < (1 / 2) + real.sqrt (2 * n) :=
sorry

end k_less_than_half_plus_sqrt_two_n_l295_295122


namespace tank_min_cost_l295_295656

/-- A factory plans to build an open-top rectangular tank with one fixed side length of 8m and a maximum water capacity of 72m³. The cost 
of constructing the bottom and the walls of the tank are $2a$ yuan per square meter and $a$ yuan per square meter, respectively. 
We need to prove the optimal dimensions and the minimum construction cost.
-/
theorem tank_min_cost 
  (a : ℝ)   -- cost multiplier
  (b h : ℝ) -- dimensions of the tank
  (volume_constraint : 8 * b * h = 72) : 
  (b = 3) ∧ (h = 3) ∧ (16 * a * (b + h) + 18 * a = 114 * a) :=
by
  sorry

end tank_min_cost_l295_295656


namespace correct_divisor_l295_295466

-- Definitions of variables and conditions
variables (X D : ℕ)

-- Stating the theorem
theorem correct_divisor (h1 : X = 49 * 12) (h2 : X = 28 * D) : D = 21 :=
by
  sorry

end correct_divisor_l295_295466


namespace intersection_complement_l295_295641

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement (U A : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The main theorem to be proved
theorem intersection_complement :
  B ∩ (complement U A) = {3, 4} := by
  sorry

end intersection_complement_l295_295641


namespace angle_B_and_side_b_in_triangle_l295_295062

theorem angle_B_and_side_b_in_triangle
  (A B C : ℝ) (a b c: ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_opposite_sides : a = b * sin C / sin B)
  (h_equation : 2 * c = sqrt 3 * a + 2 * b * cos A)
  (h_angle_sum : A + B + C = π)
  (h_c_val : c = 7)
  (h_b_sin : b * sin A = sqrt 3) :
  B = π / 6 ∧ b = sqrt 19 :=
by
  sorry

end angle_B_and_side_b_in_triangle_l295_295062


namespace part1_tangent_line_part2_inequality_l295_295778

noncomputable def f (x a : ℝ) := x * Real.exp (x - a) - Real.log x - Real.log a

def tangent_perpendicular_condition (a : ℝ) := a > 0 ∧ f 1 a = 1 ∧ ∀ (l : ℝ → ℝ), (l 1 = 1 ∧ ∃ m, l = λ x => m * (x - 1) + 1 ∧ m = 1)

def inequality_condition (a : ℝ) := 0 < a ∧ a < (Real.sqrt 5 - 1) / 2

theorem part1_tangent_line (a : ℝ) : tangent_perpendicular_condition a → ∀ y x, y = x :=
by
  sorry

theorem part2_inequality (a x : ℝ) (h : inequality_condition a) : f x a > a / (a + 1) :=
by
  sorry

end part1_tangent_line_part2_inequality_l295_295778


namespace weight_of_one_serving_l295_295349

theorem weight_of_one_serving
  (total_servings : ℕ)
  (chicken_weight_pounds : ℝ)
  (stuffing_weight_ounces : ℝ)
  (ounces_per_pound : ℝ)
  (total_servings = 12)
  (chicken_weight_pounds = 4.5)
  (stuffing_weight_ounces = 24)
  (ounces_per_pound = 16) :
  (chicken_weight_pounds * ounces_per_pound + stuffing_weight_ounces) / total_servings = 8 :=
by
  sorry

end weight_of_one_serving_l295_295349


namespace largest_angle_triangl_DEF_l295_295843

theorem largest_angle_triangl_DEF (d e f : ℝ) (h1 : d + 3 * e + 3 * f = d^2)
  (h2 : d + 3 * e - 3 * f = -8) : 
  ∃ (F : ℝ), F = 109.47 ∧ (F > 90) := by sorry

end largest_angle_triangl_DEF_l295_295843


namespace correct_propositions_l295_295504

def f : ℝ → ℝ := sorry
axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom shift_property (x : ℝ) : f (x - 2) = -f (x)
axiom cubic_definition (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : f(x) = x^3

theorem correct_propositions :
  (∀ x, f (x - 4) = f x) ∧
  (∀ x, 1 ≤ x → x ≤ 3 → f x = (2 - x)^3) ∧
  (∀ x, f (1 + x) = f (1 - x)) :=
by
  sorry

end correct_propositions_l295_295504


namespace draw_lines_separating_points_l295_295691

noncomputable def points := Finset (EuclideanSpace ℝ (Fin 2))
noncomputable def lines := Finset (AffineSubspace ℝ (Fin 2))

theorem draw_lines_separating_points (S : points) (hS1 : S.card = 4031) (hS2 : 
  (S.filter (λ p : ℝ ^ 2, p.isBlue)).card = 2015) (hS3 : (S.filter (λ p : ℝ ^ 2, ¬ p.isBlue)).card = 2016) (hS4 : ¬ collinear S) :
  ∃ L : lines, L.card = 2015 ∧ ∀ l ∈ L, ∀ p ∈ S, p ∉ l ∧ 
  ∀ region ∈ regions_of_lines L, (∀ p1 ∈ region, ∀ p2 ∈ region, p1.color = p2.color) :=
sorry

end draw_lines_separating_points_l295_295691


namespace find_a_l295_295404

noncomputable def f : ℝ → ℝ := λ x, if x < 1 then 2^x else Real.log x / Real.log 2

theorem find_a (a : ℝ) (h : f a = 2) : a = 4 := by
  sorry

end find_a_l295_295404


namespace number_of_zeros_f_l295_295880

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else log a x

theorem number_of_zeros_f (a : ℝ) (h₀ : 0 < a ∧ a ≠ 1) :
  ∃! x, f a x - a = 0 :=
sorry

end number_of_zeros_f_l295_295880


namespace circle_equation_line_intersect_circle_l295_295835

theorem circle_equation (x y : ℝ) : 
  y = x^2 - 4*x + 3 → (x = 0 ∧ y = 3) ∨ (y = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

theorem line_intersect_circle (m : ℝ) :
  (∀ x y : ℝ, (x + y + m = 0) ∨ ((x - 2)^2 + (y - 2)^2 = 5)) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + y₁ + m = 0) → ((x₁ - 2)^2 + (y₁ - 2)^2 = 5) →
    (x₂ + y₂ + m = 0) → ((x₂ - 2)^2 + (y₂ - 2)^2 = 5) →
    ((x₁ * x₂ + y₁ * y₂ = 0) → (m = -1 ∨ m = -3))) :=
sorry

end circle_equation_line_intersect_circle_l295_295835


namespace digit_of_fraction_l295_295617

theorem digit_of_fraction (n : ℕ) (a b: ℚ) (repeat_length : ℕ) (d : ℕ) (h : a / b = d) 
(repeat_length_proof : ∀ k, (a / b).to_real * 10^k ≠ d) : 
  n % repeat_length = d → nat.nth_digit_after_decimal (a / b) n = 0 :=
by sorry

end digit_of_fraction_l295_295617


namespace ice_creams_needed_l295_295155

theorem ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) (games_to_buy : ℕ) 
    (h1 : game_cost = 60) (h2 : ice_cream_price = 5) (h3 : games_to_buy = 2) : 
    (games_to_buy * game_cost) / ice_cream_price = 24 :=
by
  rw [h1, h2, h3]
  sorry

end ice_creams_needed_l295_295155


namespace part1_part2_l295_295763

open Real

variables {a b c : ℝ}
variables {A B C : ℝ}
variables {triangle_ABC : a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π}

-- Prove b^2 = ac given the trigonometric condition
theorem part1 (h1 : sin B * (tan A + tan C) = tan A * tan C) : b^2 = a * c :=
sorry 

-- Find the area of the triangle given a = 2, c = 1, and using the result of part 1
theorem part2 (h2 : a = 2) (h3 : c = 1) (h4 : b^2 = a * c) : 
  let b := sqrt (a * c),
      S := (1 / 2) * a * c * sin B
  in  S = (sqrt 7) / 4 :=
sorry

end part1_part2_l295_295763


namespace mixtape_total_length_l295_295680

theorem mixtape_total_length :
  ∀ (first_side_songs second_side_songs song_length : ℕ),
  first_side_songs = 6 →
  second_side_songs = 4 →
  song_length = 4 →
  (first_side_songs + second_side_songs) * song_length = 40 :=
by
  intros first_side_songs second_side_songs song_length
  intros h_first h_second h_length
  rw [h_first, h_second, h_length]
  sorry

end mixtape_total_length_l295_295680


namespace sum_of_first_45_natural_numbers_l295_295620
-- Step d): Rewrite the math proof problem in Lean 4 statement

theorem sum_of_first_45_natural_numbers : 
  (∑ k in Finset.range 45.succ, k) = 1035 :=
by
  sorry

end sum_of_first_45_natural_numbers_l295_295620


namespace arithmetic_seq_general_term_geometric_seq_general_term_l295_295400

theorem arithmetic_seq_general_term (a : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2) :
  ∀ n, a n = 2 * n + 2 :=
by sorry

theorem geometric_seq_general_term (a b : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2)
  (h3 : b 2 = a 3) (h4 : b 3 = a 7) :
  ∀ n, b n = 2 ^ (n + 1) :=
by sorry

end arithmetic_seq_general_term_geometric_seq_general_term_l295_295400


namespace fraction_habitable_l295_295058

theorem fraction_habitable : (1 / 3) * (1 / 3) = 1 / 9 := 
by 
  sorry

end fraction_habitable_l295_295058


namespace ellipse_standard_equation_proof_l295_295945

-- Define the conditions of the ellipse
def focus_on_x_axis : Prop := True -- This is trivial in the given context
def distance_focus_to_minor_axis_endpoint : ℝ := 2
def distance_focus_to_left_vertex : ℝ := 3

-- Define the standard equation of the ellipse to be proved
def standard_equation_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

-- The main theorem to be proved
theorem ellipse_standard_equation_proof :
  ∃ a b : ℝ, focus_on_x_axis ∧ distance_focus_to_minor_axis_endpoint = 2 ∧ 
  distance_focus_to_left_vertex = 3 ∧ standard_equation_ellipse 2 (sqrt 3) :=
begin
  sorry -- Proof to be provided
end

end ellipse_standard_equation_proof_l295_295945


namespace intersection_condition_l295_295867

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 - x + a

theorem intersection_condition (a : ℝ) :
  (∃! x, f a x = 0) ↔ a ∈ set.Iic (-5 / 27) ∪ set.Ioi 1 := sorry

end intersection_condition_l295_295867


namespace triangle_identity_l295_295487

theorem triangle_identity (a b c : ℝ) (A B C : ℝ) (s : ℝ) 
  (h₁ : s = (a + b + c) / 2) 
  (h₂ : cos (A / 2) = Real.sqrt ((s - b) * (s - c) / (b * c)))
  (h₃ : cos (B / 2) = Real.sqrt ((s - a) * (s - c) / (a * c)))
  (h₄ : cos (C / 2) = Real.sqrt ((s - a) * (s - b) / (a * b))) :
  (b - c) / a * (cos (A / 2))^2 + (c - a) / b * (cos (B / 2))^2 + (a - b) / c * (cos (C / 2))^2 = 0 := sorry

end triangle_identity_l295_295487


namespace translation_of_segment_l295_295075

structure Point where
  x : ℝ
  y : ℝ

variables (A B A' : Point)

def translation_vector (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y }

def translate (P Q : Point) : Point :=
  { x := P.x + Q.x,
    y := P.y + Q.y }

theorem translation_of_segment (hA : A = {x := -2, y := 0})
                                (hB : B = {x := 0, y := 3})
                                (hA' : A' = {x := 2, y := 1}) :
  translate B (translation_vector A A') = {x := 4, y := 4} := by
  sorry

end translation_of_segment_l295_295075


namespace one_serving_weight_l295_295352

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end one_serving_weight_l295_295352


namespace sum_g_1_to_24_l295_295024

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, x ∈ set.univ  -- The domain of f is ℝ
axiom domain_g : ∀ x : ℝ, x ∈ set.univ  -- The domain of g is ℝ
axiom f_odd : ∀ x : ℝ, f(x + 1) = -f(-x - 1)
axiom eq1 : ∀ x : ℝ, f(1 - x) + g(x) = 2
axiom eq2 : ∀ x : ℝ, f(x) + g(x - 3) = 2

theorem sum_g_1_to_24 : ∑ k in finset.range 24, g (k + 1) = 48 := by
  sorry

end sum_g_1_to_24_l295_295024


namespace min_M_for_inequality_l295_295346

noncomputable def M := (9 * Real.sqrt 2) / 32

theorem min_M_for_inequality (a b c : ℝ) : 
  abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) 
  ≤ M * (a^2 + b^2 + c^2)^2 := 
sorry

end min_M_for_inequality_l295_295346


namespace pascal_triangle_ratio_l295_295821

theorem pascal_triangle_ratio :
  ∃ n r : ℕ, (binomial n r) * 3 == (binomial n (r + 1)) * 2 ∧
             (binomial n (r + 1)) * 4 == (binomial n (r + 2)) * 3 ∧
             n == 34 := by
  sorry

end pascal_triangle_ratio_l295_295821


namespace solution_set_inequality_l295_295371

def f (x : ℝ) : ℝ := sorry
def f' (x : ℝ) : ℝ := sorry

axiom condition1 : ∀ (x : ℝ), f(x) - f(-x) = 2 * x^3
axiom condition2 : ∀ (x : ℝ), x ≥ 0 → f'(x) > 3*x^2

theorem solution_set_inequality : {x : ℝ | f(x) - f(x-1) > 3*x^2 - 3*x + 1} = {x : ℝ | x > 1/2} :=
sorry

end solution_set_inequality_l295_295371


namespace gain_percent_l295_295242

theorem gain_percent (CP SP : ℝ) (hCP : CP = 100) (hSP : SP = 115) : 
  ((SP - CP) / CP) * 100 = 15 := 
by 
  sorry

end gain_percent_l295_295242


namespace region_area_l295_295226

noncomputable def area_of_region := 4 * Real.pi

theorem region_area :
  (∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0) →
  Real.pi * 4 = area_of_region :=
by
  sorry

end region_area_l295_295226


namespace total_splash_width_l295_295607

theorem total_splash_width :
  let pebble_splash := 1 / 4
  let rock_splash := 1 / 2
  let boulder_splash := 2
  let pebbles := 6
  let rocks := 3
  let boulders := 2
  let total_pebble_splash := pebbles * pebble_splash
  let total_rock_splash := rocks * rock_splash
  let total_boulder_splash := boulders * boulder_splash
  let total_splash := total_pebble_splash + total_rock_splash + total_boulder_splash
  total_splash = 7 := by
  sorry

end total_splash_width_l295_295607


namespace nesbitt_inequality_l295_295548

theorem nesbitt_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c → a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2) :=
sorry

end nesbitt_inequality_l295_295548


namespace good_pair_count_l295_295317

theorem good_pair_count (L1 L2 L3 L4 L5 : ℝ → ℝ) :
  L1 = (λ x, 4 * x + 2) ∧
  L2 = (λ x, (9 * x + 3) / 3) ∧
  L3 = (λ x, (-8 * x + 10) / 2) ∧
  L4 = (λ x, (8 * x + 5) / 3) ∧
  L5 = (λ x, -1 / 4 * x + 6) →
  (∃ (pairs : ℕ), pairs = 1) :=
by
  sorry

end good_pair_count_l295_295317


namespace y_intercept_of_line_eq_l295_295954

theorem y_intercept_of_line_eq (x y : ℝ) (h : x + y - 1 = 0) : y = 1 :=
by
  sorry

end y_intercept_of_line_eq_l295_295954


namespace probability_task1_on_time_l295_295238

-- Definitions representing the probabilities of the tasks
def P_task2 : ℚ := 3 / 5
def P_task1_and_not_task2 : ℚ := 0.15

-- Hypothesis of task independence
axiom independent_tasks : ∀ (P_task1 P_task2 : Prop), P_task1 ∧ ¬P_task2 → P_task1 * (1 - P_task2)

-- Theorem to prove the probability of task 1 completion
theorem probability_task1_on_time : ∃ (P_task1 : ℚ), P_task1 = 0.375 := by
  have P_not_task2 : ℚ := 1 - P_task2
  have h : P_task1_and_not_task2 = P_task1 * P_not_task2,
  exact independent_tasks _ _
  sorry

end probability_task1_on_time_l295_295238


namespace perfect_square_trinomial_l295_295054

theorem perfect_square_trinomial (k : ℤ) : (∃ a : ℤ, (x : ℤ) → x^2 - k * x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_l295_295054


namespace find_a_l295_295760

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then x * 2^(x + a) - 1 else - (x * 2^(-x + a) - 1)

theorem find_a (a : ℝ) (h_odd: ∀ x : ℝ, f x a = -f (-x) a)
  (h_pos : ∀ x : ℝ, x > 0 → f x a = x * 2^(x + a) - 1)
  (h_neg : f (-1) a = 3 / 4) :
  a = -3 :=
by
  sorry

end find_a_l295_295760


namespace system_solutions_l295_295595

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x ^ 2 - y ^ 2 = 0 ∧ (x - a) ^ 2 + y ^ 2 = 1

-- State the theorem
theorem system_solutions (a : ℝ) :
  let num_solutions := if a = 1 ∨ a = -1 then 3 else if a = Real.sqrt 2 ∨ a = -Real.sqrt 2 then 2 else 4
  in num_solutions = if a = 1 ∨ a = -1 then 3 else if a = Real.sqrt 2 ∨ a = -Real.sqrt 2 then 2 else 4 :=
by
  sorry

end system_solutions_l295_295595


namespace number_of_friends_l295_295905

/-
  Definitions based on the given conditions:
  1. Single carnation cost = $0.50
  2. Cost of a dozen carnations = $4.00
  3. Georgia sent a dozen carnations to 5 teachers
  4. Total spent by Georgia = $25.00
-/

def cost_single_carnation : ℕ → ℕ := λ n => n * 50
def cost_dozen_carnation : ℕ → ℕ := λ n => n * 400
def total_spent_by_georgia := 2500

theorem number_of_friends :
  ∃ (friends : ℕ), cost_single_carnation friends = total_spent_by_georgia - cost_dozen_carnation 5 :=
begin
  sorry
end

end number_of_friends_l295_295905


namespace find_m_n_pairs_l295_295716

theorem find_m_n_pairs (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∀ᶠ a in Filter.atTop, (a^m + a - 1) % (a^n + a^2 - 1) = 0) → m = n + 2 :=
by
  sorry

end find_m_n_pairs_l295_295716


namespace Ann_skating_speed_l295_295290

-- Given conditions
variables {A : ℝ} -- Ann's skating speed
def Glenda_speed : ℝ := 8 -- Glenda's skating speed
def time_skated : ℝ := 3 -- Time skated in hours
def distance_apart : ℝ := 42 -- Distance apart after 3 hours

-- The statement to be proved
theorem Ann_skating_speed :
  (A * time_skated) + (Glenda_speed * time_skated) = distance_apart →
  A = 6 :=
by
  intros h,
  sorry

end Ann_skating_speed_l295_295290


namespace solve_for_x_l295_295441

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end solve_for_x_l295_295441


namespace bags_with_chocolate_hearts_l295_295495

-- Definitions for given conditions
def total_candies : ℕ := 63
def total_bags : ℕ := 9
def candies_per_bag : ℕ := total_candies / total_bags
def chocolate_kiss_bags : ℕ := 3
def not_chocolate_candies : ℕ := 28
def bags_not_chocolate : ℕ := not_chocolate_candies / candies_per_bag
def remaining_bags : ℕ := total_bags - chocolate_kiss_bags - bags_not_chocolate

-- Statement to be proved
theorem bags_with_chocolate_hearts :
  remaining_bags = 2 := by 
  sorry

end bags_with_chocolate_hearts_l295_295495


namespace f_deriv_at_2_f_monotonic_intervals_l295_295407

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem f_deriv_at_2 : (deriv f 2 = 9) :=
by sorry

theorem f_monotonic_intervals : 
  (∀ x, x ∈ Ioo -∞ -1 → deriv f x > 0) ∧ 
  (∀ x, x ∈ Ioo 1 ∞ → deriv f x > 0) ∧ 
  (∀ x, x ∈ Ioo -1 1 → deriv f x < 0) :=
by sorry

end f_deriv_at_2_f_monotonic_intervals_l295_295407


namespace max_plus_min_eq_two_l295_295787

noncomputable def f (x : ℝ) : ℝ := (sqrt 2 * sin (x + π / 4) + 2 * x^2 + x) / (2 * x^2 + cos x)

def maxValue (f : ℝ → ℝ) : ℝ := sorry -- Implement maximum calculation
def minValue (f : ℝ → ℝ) : ℝ := sorry -- Implement minimum calculation

def M : ℝ := maxValue f
def m : ℝ := minValue f

theorem max_plus_min_eq_two : M + m = 2 := sorry

end max_plus_min_eq_two_l295_295787


namespace quadratic_roots_shift_l295_295506

theorem quadratic_roots_shift (b c : ℝ) (h : ∀ x : ℝ, x^2 + b * x + c = 0) :
  (∃ r s : ℝ, 2 * x^2 - 4 * x - 8 = 0 ∧ (r, s) ∈ RootsOfQuadratic 2 (-4) (-8) ∧
  (r + 3, s + 3) ∈ RootsOfQuadratic 1 b c) →
  c = 11 :=
by
  sorry

end quadratic_roots_shift_l295_295506


namespace polygon_area_correct_l295_295302

noncomputable def polygonArea : ℝ :=
  let x1 := 1
  let y1 := 1
  let x2 := 4
  let y2 := 3
  let x3 := 5
  let y3 := 1
  let x4 := 6
  let y4 := 4
  let x5 := 3
  let y5 := 6
  (1 / 2 : ℝ) * 
  abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y5 + x5 * y1) -
       (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x5 + y5 * x1))

theorem polygon_area_correct : polygonArea = 11.5 := by
  sorry

end polygon_area_correct_l295_295302


namespace curve_cartesian_equation_minimum_AB_distance_l295_295823

noncomputable def curve_polar_to_cartesian (θ : ℝ) : ℝ := 
  (2 * Real.cos θ) / (Real.sin θ)^2

def line_parametric (t α : ℝ) : ℝ × ℝ :=
  (1/2 + t * Real.cos α, t * Real.sin α)

theorem curve_cartesian_equation :
  ∀ x y : ℝ, (∃ θ : ℝ, curve_polar_to_cartesian θ = Real.sqrt (x^2 + y^2) ∧ y = x * Real.tan θ) ↔ y^2 = 2 * x :=
by sorry

theorem minimum_AB_distance :
  ∀ (α : ℝ), 0 < α ∧ α < Real.pi → ∃ A B : ℝ × ℝ, A ≠ B ∧ ∀ t : ℝ, 
  let P := line_parametric t α in
  field_theory.Polynomial.dist P (A,B) ≥ 2 := 
by sorry

end curve_cartesian_equation_minimum_AB_distance_l295_295823


namespace distinct_integers_count_l295_295046

def f (n : ℕ) : ℕ := n * n / 1998

theorem distinct_integers_count :
  (Finset.image f (Finset.range 1997).succ).card = 1498 :=
sorry

end distinct_integers_count_l295_295046


namespace proof_problem_l295_295509

variable {Point : Type} [LinearOrder Point]
variables (A B C D : Point) (AC CB AD DB AB : ℝ)
variable (hCond : AC / CB + AD / DB = 0)
variable (h1 : AC = abs (C - A))
variable (h2 : CB = abs (B - C))
variable (h3 : AD = abs (D - A))
variable (h4 : DB = abs (B - D))
variable (h5 : AB = abs (B - A))

theorem proof_problem
  (hCond : AC / CB + AD / DB = 0)
  (h1 : AC = abs (C - A))
  (h2 : CB = abs (B - C))
  (h3 : AD = abs (D - A))
  (h4 : DB = abs (B - D))
  (h5 : AB = abs (B - A)) :
  1 / AC + 1 / AD = 2 / AB :=
sorry

end proof_problem_l295_295509


namespace sum_of_x_values_for_g_l295_295034

noncomputable def g : ℝ → ℝ := sorry  -- Assume g(x) is a quadratic function

def sum_of_distinct_x_values (f : ℝ → ℝ) (y : ℝ) : ℝ :=
  -- To be implemented: This function should return the sum of distinct x-values such that f(f(f(x))) = y
  sorry

theorem sum_of_x_values_for_g (h : g) :
  sum_of_distinct_x_values g 2 = -8 :=
sorry

end sum_of_x_values_for_g_l295_295034


namespace line_l_prime_eq_2x_minus_3y_plus_5_l295_295791

theorem line_l_prime_eq_2x_minus_3y_plus_5 (m : ℝ) (x y : ℝ) : 
  (2 * m + 1) * x + (m + 1) * y + m = 0 →
  (2 * -1 + 1) * (-1) + (1 + 1) * 1 + m = 0 →
  ∀ a b : ℝ, (3 * b, 2 * b) = (3 * 1, 2 * 1) → (a, b) = (-1, 1) → 
  2 * x - 3 * y + 5 = 0 :=
by
  intro h1 h2 a b h3 h4
  sorry

end line_l_prime_eq_2x_minus_3y_plus_5_l295_295791


namespace number_of_women_l295_295260

theorem number_of_women (w1 w2: ℕ) (m1 m2 d1 d2: ℕ)
    (h1: w2 = 5) (h2: m2 = 100) (h3: d2 = 1) 
    (h4: d1 = 3) (h5: m1 = 360)
    (h6: w1 * d1 = m1 * d2 / m2 * w2) : w1 = 6 :=
by
  sorry

end number_of_women_l295_295260


namespace find_t_l295_295117

-- Given: (1) g(x) = x^5 + px^4 + qx^3 + rx^2 + sx + t with all roots being negative integers
--        (2) p + q + r + s + t = 3024
-- Prove: t = 1600

noncomputable def poly (x : ℝ) (p q r s t : ℝ) := 
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

theorem find_t
  (p q r s t : ℝ)
  (roots_neg_int : ∀ root, root ∈ [-s1, -s2, -s3, -s4, -s5] → (root : ℤ) < 0)
  (sum_coeffs : p + q + r + s + t = 3024)
  (poly_1_eq : poly 1 p q r s t = 3025) :
  t = 1600 := 
sorry

end find_t_l295_295117


namespace symmetry_properties_l295_295032

noncomputable def f (a b c p q : ℝ) (x : ℝ) : ℝ :=
  (a * 4^x + b * 2^x + c) / (p * 2^x + q)

theorem symmetry_properties (a b c p q : ℝ) :
  (p = 0 ∧ q ≠ 0 ∧ a^2 + b^2 ≠ 0 → ¬(∀ t x, f(a, b, c, p, q, t + x) = f(a, b, c, p, q, t - x)) ∧ 
                             ¬(∀ t x, f(a, b, c, p, q, t + x) + f(a, b, c, p, q, t - x) = k)) ∧
  (p ≠ 0 ∧ q = 0 ∧ ac ≠ 0 → 
                             (∃ t x, f(a, b, c, p, q, t + x) = f(a, b, c, p, q, t - x)) ∨ 
                             (∠ t x, f(a, b, c, p, q, t + x) + f(a, b, c, p, q, t - x) = k)) ∧
  (pq ≠ 0 ∧ a = 0 ∧ b^2 + c^2 ≠ 0 → 
                              ∀ t x, f(a, b, c, p, q, t + x) + f(a, b, c, p, q, t - x) = k) ∧
  (pq ≠ 0 ∧ a ≠ 0 →
                              ∃ t x, f(a, b, c, p, q, t + x) = f(a, b, c, p, q, t - x)) :=
sorry

end symmetry_properties_l295_295032


namespace opposite_of_two_is_neg_two_l295_295584

theorem opposite_of_two_is_neg_two : ∃ y : ℝ, (2 + y = 0) ∧ (y = -2) :=
by
  use -2
  split
  · exact add_right_neg 2
  · refl

end opposite_of_two_is_neg_two_l295_295584


namespace incorrect_condition_B_l295_295399

variable {α β : Type} [Plane α] [Plane β]
variable {m n : Type} [Line m] [Line n]

theorem incorrect_condition_B (h_parallel_m_α : m ∥ α) (h_intersection : α ∩ β = n) : ¬ (m ∥ n) :=
sorry

end incorrect_condition_B_l295_295399


namespace election_votes_l295_295598

-- Define the given conditions and the goal in a Lean code statement.
theorem election_votes (V: ℝ) (h1: V = 1200) (h2: 0.62 * 1200 = 744) (h3: 0.24 * 1200 = 288) : 
  ∃ w : ℝ, w = 0.62 * V ∧ w = 744 := by
  -- Assume the necessary conditions
  have hV : V = 1200 := h1
  have hw : 0.62 * V = 744 := h2
  existsi (0.62 * V)
  show 0.62 * V = 744 from hw

end election_votes_l295_295598


namespace exponent_to_match_decimal_places_l295_295439

theorem exponent_to_match_decimal_places :
  ∃ (x : ℝ), (10^x * 3.456789) ^ 14 = 10^56 * (3.456789 ^ 14) ∧ x = 4 :=
begin
  use 4,
  split,
  sorry,
  refl,
end

end exponent_to_match_decimal_places_l295_295439


namespace max_sum_of_segments_l295_295857

theorem max_sum_of_segments (A B C D : ℝ × ℝ × ℝ)
    (h : (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D ≤ 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1)
      ∨ (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D > 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1))
    : dist A B + dist A C + dist A D + dist B C + dist B D + dist C D ≤ 5 + Real.sqrt 3 := sorry

end max_sum_of_segments_l295_295857


namespace hexagon_side_length_l295_295960

theorem hexagon_side_length (a b c : ℝ) :
  let x := (a * b * c) / (a * b + b * c + a * c) in
  x = (a * b * c) / (a * b + b * c + a * c) :=
by
  sorry

end hexagon_side_length_l295_295960


namespace triangle_angles_l295_295072

def is_isosceles_right_triangle {α : Type*} [linear_ordered_field α] (a b : α) : Prop :=
  a = b

theorem triangle_angles (a b c : ℝ) (h_a h_b h_c : ℝ)
  (h1 : h_a = b * real.sin (real.arcsin ((a / b))) )
  (h2 : h_b = a * real.sin (real.arcsin ((b / a))) )
  (h3 : h_a ≥ a)
  (h4 : h_b ≥ b) :
  is_isosceles_right_triangle a b :=

  -- Placeholder proof
  sorry

end triangle_angles_l295_295072


namespace tangents_to_circle_parallel_to_line_l295_295005

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 12 * y + 45 = 0

-- Define the equation of the given line
def line_eq (x y : ℝ) : Prop := y = 3 * x

-- Define the equations of the tangent lines to be proven
def tangent_line1 (x y : ℝ) : Prop := y = 3 * x - 9 + 4 * sqrt 10
def tangent_line2 (x y : ℝ) : Prop := y = 3 * x - 9 - 4 * sqrt 10

-- The theorem statement
theorem tangents_to_circle_parallel_to_line :
  ∀ x y : ℝ, 
  circle_eq x y → 
  (tangent_line1 x y ∨ tangent_line2 x y) :=
by 
  sorry

end tangents_to_circle_parallel_to_line_l295_295005


namespace shaded_region_area_l295_295670

theorem shaded_region_area :
  let side_length := 20
  let radius := 10
  let square_area := side_length * side_length
  let circle_area := π * radius * radius
  let quarter_circle_area := circle_area / 4
  let total_quarter_circles_area := 4 * quarter_circle_area
  shaded_region_area = square_area - total_quarter_circles_area :=
by
  sorry

end shaded_region_area_l295_295670


namespace bacteria_colony_growth_l295_295460

theorem bacteria_colony_growth (n : ℕ) : 
  (∀ m: ℕ, 4 * 3^m ≤ 500 → m < n) → n = 5 :=
by
  sorry

end bacteria_colony_growth_l295_295460


namespace xiaoming_waiting_probability_l295_295239

-- Define the length of the interval between bus arrivals.
def bus_arrival_interval : ℝ := 15

-- Define the maximum waiting time (the interval length of favorable event).
def max_waiting_time : ℝ := 10

-- Define the probability as a ratio of the favorable event's length to the interval length.
def probability_waiting_less_than_10_minutes (interval_length favorable_length : ℝ) : ℝ :=
  favorable_length / interval_length

-- State the theorem about Xiaoming's waiting probability
theorem xiaoming_waiting_probability : probability_waiting_less_than_10_minutes bus_arrival_interval max_waiting_time = 2 / 3 :=
  sorry

end xiaoming_waiting_probability_l295_295239


namespace external_tangent_length_eq_geom_mean_of_diameters_l295_295819

theorem external_tangent_length_eq_geom_mean_of_diameters (r1 r2 : ℝ) (h1 : 0 ≤ r1) (h2 : 0 ≤ r2) (h3 : r1 ≤ r2) (h4 : ∃ P Q : ℝ^2, ∃ l : Set ℝ^2, is_tangent r1 r2 P Q l) : 
  length_of_tangent r1 r2 = 2 * real.sqrt (r1 * r2) := sorry

end external_tangent_length_eq_geom_mean_of_diameters_l295_295819


namespace total_loss_l295_295679

variables (A P L : ℝ)

-- Conditions
def condition1 : Prop := A = (1/9) * P
def condition2 : Prop := 1080 = (P / (A + P)) * L

-- The problem is to prove that
theorem total_loss (h1 : condition1) (h2 : condition2) : L = 1200 :=
sorry

end total_loss_l295_295679


namespace coefficient_of_x_squared_in_expansion_l295_295925

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n, k := if h : k ≤ n then Nat.choose n k else 0

def expansion_coefficient : ℕ := 
  binomial_coefficient 5 1 - 2 * binomial_coefficient 5 2

theorem coefficient_of_x_squared_in_expansion:
  expansion_coefficient = -15 := 
by
  sorry

end coefficient_of_x_squared_in_expansion_l295_295925


namespace quadratic_solution_sum_l295_295590

theorem quadratic_solution_sum
  (x : ℚ)
  (m n p : ℕ)
  (h_eq : (5 * x - 11) * x = -6)
  (h_form : ∃ m n p, x = (m + Real.sqrt n) / p ∧ x = (m - Real.sqrt n) / p)
  (h_gcd : Nat.gcd (Nat.gcd m n) p = 1) :
  m + n + p = 22 := 
sorry

end quadratic_solution_sum_l295_295590


namespace trader_profit_loss_l295_295633

noncomputable def profit_loss_percentage (sp1 sp2: ℝ) (gain_loss_rate1 gain_loss_rate2: ℝ) : ℝ :=
  let cp1 := sp1 / (1 + gain_loss_rate1)
  let cp2 := sp2 / (1 - gain_loss_rate2)
  let tcp := cp1 + cp2
  let tsp := sp1 + sp2
  let profit_or_loss := tsp - tcp
  profit_or_loss / tcp * 100

theorem trader_profit_loss : 
  profit_loss_percentage 325475 325475 0.15 0.15 = -2.33 := 
by 
  sorry

end trader_profit_loss_l295_295633


namespace sphere_radius_l295_295175

theorem sphere_radius (A : ℝ) (r : ℝ) (h : A = 64 * π) : 4 * π * r^2 = 64 * π → r = 4 :=
by
  intro h2
  have h_eq: (4 * π * r^2) / (4 * π) = 16 := by
    rw h2
    rw h
    ring
  have h_sq: r^2 = 16 := by
    field_simp at h_eq 
    exact h_eq
  have h_r: r = 4 := by
    exact eq_of_pow_eq_pow 2 (by norm_num) h_sq
  exact h_r

end sphere_radius_l295_295175


namespace alice_savings_third_month_l295_295846

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end alice_savings_third_month_l295_295846


namespace jakes_weight_l295_295634

theorem jakes_weight
  (J K : ℝ)
  (h1 : J - 8 = 2 * K)
  (h2 : J + K = 290) :
  J = 196 :=
by
  sorry

end jakes_weight_l295_295634


namespace flour_needed_for_two_loaves_l295_295535

-- Define the amount of flour needed for one loaf.
def flour_per_loaf : ℝ := 2.5

-- Define the number of loaves.
def number_of_loaves : ℕ := 2

-- Define the total amount of flour needed for the given number of loaves.
def total_flour_needed : ℝ := flour_per_loaf * number_of_loaves

-- The theorem statement: Prove that the total amount of flour needed is 5 cups.
theorem flour_needed_for_two_loaves : total_flour_needed = 5 := by
  sorry

end flour_needed_for_two_loaves_l295_295535


namespace g_of_3_over_8_l295_295189

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l295_295189


namespace China_NTT_2009_problem_l295_295875

theorem China_NTT_2009_problem
  (n : ℕ) (a b : ℤ)
  (h_pos : 0 < n)
  (h_ineq : a > b ∧ b > 1)
  (h_odd : b % 2 = 1)
  (h_div : b^n ∣ (a^n - 1)) :
  a^b > (3^n) / n := sorry

end China_NTT_2009_problem_l295_295875


namespace total_cleaning_time_l295_295890

theorem total_cleaning_time (time_outside : ℕ) (fraction_inside : ℚ) (time_inside : ℕ) (total_time : ℕ) :
  time_outside = 80 →
  fraction_inside = 1 / 4 →
  time_inside = fraction_inside * time_outside →
  total_time = time_outside + time_inside →
  total_time = 100 :=
by
  intros hto hfi htinside httotal
  rw [hto, hfi] at htinside
  norm_num at htinside
  rw [hto, htinside] at httotal
  norm_num at httotal
  exact httotal

end total_cleaning_time_l295_295890


namespace Tonya_spent_on_brushes_l295_295603

section
variable (total_spent : ℝ)
variable (cost_canvases : ℝ)
variable (cost_paints : ℝ)
variable (cost_easel : ℝ)
variable (cost_brushes : ℝ)

def Tonya_total_spent : Prop := total_spent = 90.0
def Cost_of_canvases : Prop := cost_canvases = 40.0
def Cost_of_paints : Prop := cost_paints = cost_canvases / 2
def Cost_of_easel : Prop := cost_easel = 15.0
def Cost_of_brushes : Prop := cost_brushes = total_spent - (cost_canvases + cost_paints + cost_easel)

theorem Tonya_spent_on_brushes : Tonya_total_spent total_spent →
  Cost_of_canvases cost_canvases →
  Cost_of_paints cost_paints cost_canvases →
  Cost_of_easel cost_easel →
  Cost_of_brushes cost_brushes total_spent cost_canvases cost_paints cost_easel →
  cost_brushes = 15.0 := by
  intro h_total_spent h_cost_canvases h_cost_paints h_cost_easel h_cost_brushes
  rw [Tonya_total_spent, Cost_of_canvases, Cost_of_paints, Cost_of_easel, Cost_of_brushes] at *
  sorry
end

end Tonya_spent_on_brushes_l295_295603


namespace find_line_equation_l295_295755

theorem find_line_equation
  (l1 : ∀ x y : ℝ, 3 * x + 2 * y - 1 = 0)
  (l2 : ∀ x y : ℝ, 5 * x + 2 * y + 1 = 0)
  (l3 : ∀ x y : ℝ, 3 * x - 5 * y + 6 = 0)
  (L : ∀ x y : ℝ, L x y ↔ (L passes through intersection of l1 and l2) ∧ (L is perpendicular to l3))
  : (∀ x y : ℝ, 5 * x + 3 * y - 1 = 0) := sorry

end find_line_equation_l295_295755


namespace graph_not_pass_through_first_quadrant_l295_295661

theorem graph_not_pass_through_first_quadrant (k b : ℝ) (h1 : k < 0) (h2 : k * b > 0) : ¬ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + b :=
begin
  sorry
end

end graph_not_pass_through_first_quadrant_l295_295661


namespace ellipse_parabola_intersection_l295_295060

open Real

theorem ellipse_parabola_intersection (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) ↔ (-1 ≤ a ∧ a ≤ 17 / 8) :=
by
  sorry

end ellipse_parabola_intersection_l295_295060


namespace wardrobe_probability_l295_295829

theorem wardrobe_probability :
  let total_articles := 5 + 6 + 7 in
  let total_ways := choose total_articles 4 in
  let ways_choose_shirts := choose 5 2 in
  let ways_choose_shorts := choose 6 1 in
  let ways_choose_socks := choose 7 1 in
  let favorable_ways := ways_choose_shirts * ways_choose_shorts * ways_choose_socks in
  (favorable_ways : ℚ) / (total_ways : ℚ) = 7 / 51 :=
by
  let total_articles := 18
  let total_ways := choose total_articles 4
  let ways_choose_shirts := choose 5 2
  let ways_choose_shorts := choose 6 1
  let ways_choose_socks := choose 7 1
  let favorable_ways := ways_choose_shirts * ways_choose_shorts * ways_choose_socks
  have h1 : total_articles = 18 := rfl
  have h2 : total_ways = 3060 := by norm_num [total_ways]
  have h3 : ways_choose_shirts = 10 := by norm_num [ways_choose_shirts]
  have h4 : ways_choose_shorts = 6 := by norm_num [ways_choose_shorts]
  have h5 : ways_choose_socks = 7 := by norm_num [ways_choose_socks]
  have h_fav_ways : favorable_ways = 420 := by norm_num [favorable_ways, h3, h4, h5]
  show (favorable_ways : ℚ) / (total_ways : ℚ) = 7 / 51
  sorry -- Full proof omitted

end wardrobe_probability_l295_295829


namespace women_in_first_class_l295_295160

noncomputable def percentage_of (portion total: ℕ) : ℝ :=
  (portion : ℝ) * total / 100

theorem women_in_first_class (total_passengers : ℕ)
  (percent_women percent_first_class : ℝ) (total_women total_first_class: ℝ)
  (h1 : total_passengers = 150) 
  (h2 : percent_women = 70)
  (h3 : percent_first_class = 15)
  (h4 : total_women = percentage_of percent_women total_passengers)
  (h5 : total_first_class = percentage_of percent_first_class total_women) :
  total_first_class.round = 16 :=
by
  rw [h1, h2, h3, h4, h5]
  -- further calculations would go here
  admit

end women_in_first_class_l295_295160


namespace sum_of_consecutive_integers_exists_l295_295234

theorem sum_of_consecutive_integers_exists : 
  ∃ k : ℕ, 150 * k + 11325 = 5827604250 :=
by
  sorry

end sum_of_consecutive_integers_exists_l295_295234


namespace segments_equal_length_l295_295858

variable (A B C I X Y : Type) [Geometry] [Incenter I A B C]
variable (H1 : Circle (A, C, I)) (H2 : Circle (B, C, I))
variable (H3 : H1.intersect_line (Line (B, C)) ∧ H1.second_intersection_line (Line (B, C)) X)
variable (H4 : H2.intersect_line (Line (A, C)) ∧ H2.second_intersection_line (Line (A, C)) Y)

theorem segments_equal_length : dist A Y = dist B X := sorry

end segments_equal_length_l295_295858


namespace vasya_filling_time_l295_295597

-- Definition of conditions
def hose_filling_time (x : ℝ) : Prop :=
  ∀ (first_hose_mult second_hose_mult : ℝ), 
    first_hose_mult = x ∧
    second_hose_mult = 5 * x ∧
    (5 * second_hose_mult - 5 * first_hose_mult) = 1

-- Conclusion
theorem vasya_filling_time (x : ℝ) (first_hose_mult second_hose_mult : ℝ) :
  hose_filling_time x → 25 * x = 1 * (60 + 15) := sorry

end vasya_filling_time_l295_295597


namespace students_per_section_after_new_admission_l295_295583

theorem students_per_section_after_new_admission : 
  ∀ (S : ℕ), 24 * (S + 1) / (S + 4) = 21 :=
by
  -- given that S + 3 sections is equal to 16
  have h1 : S + 3 = 16 := by sorry
  -- solving for S from the above equation
  have h2 : S = 13 := by sorry
  -- calculate the final number of students per section
  calc
    24 * (S + 1) / (S + 4) = 24 * (13 + 1) / (13 + 4) : by rw [←h2]
                     ... = 24 * 14 / 17 : by simp
                     ... = 336 / 16 : by simp
                     ... = 21 : by norm_num

end students_per_section_after_new_admission_l295_295583


namespace shifted_graph_is_C_l295_295929

def f (x : ℝ) : ℝ :=
  if h1 : -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if h2 : 0 ≤ x ∧ x ≤ 2 then (real.sqrt (4 - (x - 2)^2)) - 2
  else if h3 : 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

def shiftedGraphLetter : char := 'C'

theorem shifted_graph_is_C : ∀ x : ℝ, f (x - 2) = f x := 
sorry

end shifted_graph_is_C_l295_295929


namespace lines_perpendicular_l295_295878

variable {P : Type} [inner_product_space ℝ P] -- Assuming an inner product space setup.

-- Definitions for points and triangles
variables (A B C D E : P)
variables (H1 H3 : P) -- Orthocenters
variables (S2 S4 : P) -- Centroids

-- Conditions
def Quadrilateral (A B C D : P) : Prop :=
  convex_hull ℝ ({A, B, C, D}).to_set = set_of_points_inside_convex_hull

def intersects_at (A C B D : P) (E: P) : Prop := 
  line [A, C] ⋂ line [B, D] = E

def orthocenter (A B C : P) (H : P) : Prop :=
  is_orthocenter_of_the_triangle A B C H

def centroid (A B C : P) (S : P) : Prop := 
  is_centroid_of_the_triangle A B C S
  
-- Problem Statement
theorem lines_perpendicular
  (quad : Quadrilateral A B C D)
  (intersect_E : intersects_at A C B D E)
  (orthocenter_EAB : orthocenter E A B H1)
  (orthocenter_ECD : orthocenter E C D H3)
  (centroid_EBC : centroid E B C S2)
  (centroid_EDA : centroid E D A S4) :
  ∠ ((H1:H3).vector) ⋂ ((S2:S4).vector) = π / 2 :=
begin
  sorry,
end

end lines_perpendicular_l295_295878


namespace measure_NMO_l295_295265

-- Definitions based on conditions in part a)
structure Triangle :=
  (P Q R : Type)
  (angle_P : ℝ)
  (angle_Q : ℝ)
  (angle_R : ℝ)
  (angle_sum_eq : angle_P + angle_Q + angle_R = 180)

def Circle (T : Triangle) := Type

structure SupportsOn {T : Triangle} (Γ : Circle T) (M N O : Type) :=
  (M_on_QR : M ∈ Seq (T.Q, T.R))
  (N_on_PQ : N ∈ Seq (T.P, T.Q))
  (O_on_PR : O ∈ Seq (T.P, T.R))

-- Proof statement based on part c)
theorem measure_NMO (T : Triangle) (Γ : Circle T) (M N O : Type)
  [SupportsOn Γ M N O]
  (P_angle : T.angle_P = 50)
  (Q_angle : T.angle_Q = 70)
  (R_angle : T.angle_R = 60) :
  measure (angle M N O) = 60 :=
sorry

end measure_NMO_l295_295265


namespace value_in_half_dollars_percentage_l295_295628

theorem value_in_half_dollars_percentage (n h q : ℕ) (hn : n = 75) (hh : h = 40) (hq : q = 30) : 
  (h * 50 : ℕ) / (n * 5 + h * 50 + q * 25 : ℕ) * 100 = 64 := by
  sorry

end value_in_half_dollars_percentage_l295_295628


namespace original_population_multiple_of_4_l295_295142

theorem original_population_multiple_of_4 :
  ∃ a b c : ℤ,
  a^2 + 150 = b^2 + 1 ∧
  a^2 + 301 = c^2 ∧
  (a^2) % 4 = 0 :=
begin
  sorry
end

end original_population_multiple_of_4_l295_295142


namespace minimum_guests_economical_option_l295_295540

theorem minimum_guests_economical_option :
  ∀ (x : ℕ), (150 + 20 * x > 300 + 15 * x) → x > 30 :=
by 
  intro x
  sorry

end minimum_guests_economical_option_l295_295540


namespace bella_earrings_l295_295295

theorem bella_earrings (B M R : ℝ) 
  (h1 : B = 0.25 * M) 
  (h2 : M = 2 * R) 
  (h3 : B + M + R = 70) : 
  B = 10 := by 
  sorry

end bella_earrings_l295_295295


namespace relations_AC_BD_skew_l295_295587

-- Define the concept of skew lines within a space.
def are_skew_lines (l1 l2 : Line) : Prop :=
  ¬∃ (p : Point), p ∈ l1 ∧ p ∈ l2 ∧ ¬∃ (π : Plane), l1 ∈ π ∧ l2 ∈ π

-- Define the relation of lines intersecting another line.
def intersect_line (l : Line) (intersecting_lines : Set Line) : Prop := 
  ∀ (l' : Line), l' ∈ intersecting_lines → (∃ (p : Point), p ∈ l ∧ p ∈ l')

-- Given conditions and theorem.
theorem relations_AC_BD_skew (A B C D : Point) 
  (AB CD AC BD : Line)
  (h1 : are_skew_lines AB CD)
  (h2 : intersect_line AC {AB, CD})
  (h3 : intersect_line BD {AB, CD}) : 
  are_skew_lines AC BD :=
sorry

end relations_AC_BD_skew_l295_295587


namespace proof_problem_l295_295386

noncomputable def question (x y : ℝ) (i : ℂ) := complex.abs (x - y * i) = √10

theorem proof_problem (x y : ℝ) (i : ℂ) (h1: i = complex.I)
  (h2: (x + 3 * i) * i = y - i) : complex.abs (x - y * i) = √10 :=
sorry

end proof_problem_l295_295386


namespace unique_solution_a_l295_295341

theorem unique_solution_a (a : ℝ) :
  (a ∈ set.Icc (-1 : ℝ) (-1 / 3) ∪ set.Icc (-1 / 3) 1) →
  ∃! x : ℝ, a * |x + 1| + (x^2 - 5 * x + 6) / (2 - x) = 0 :=
by
  sorry

end unique_solution_a_l295_295341


namespace part1_part2_l295_295378

-- Definitions of the entities involved
structure Ellipse (a b : ℝ) (e : ℝ) where
  a_pos : a > 0
  b_pos : b > 0
  eccentricity : e = sqrt 3 / 2

def point_on_ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions
def conditions_part1 (a b e : ℝ) (P F1 F2 : ℝ × ℝ) (perimeter : ℝ) : Prop :=
  Ellipse a b e ∧
  point_on_ellipse 2 1 a b ∧
  perimeter = 4 * sqrt 2 + 2 * sqrt 6

-- The equation of the ellipse we need to prove
def ellipse_equation_part1 : Prop :=
  ∃ a b : ℝ, a = 2 * sqrt 2 ∧ b = sqrt 2 ∧ point_on_ellipse x y 8 2

-- Statement of the first part
theorem part1 (a b : ℝ) (P F1 F2 : ℝ × ℝ) (perimeter : ℝ) :
  conditions_part1 a b (sqrt 3 / 2) P F1 F2 perimeter -> ellipse_equation_part1 := by
  sorry

-- Conditions for the second part
def conditions_part2 (a b : ℝ) (kPM kPN : ℝ) (P M N : ℝ × ℝ) : Prop :=
  Ellipse a b (sqrt 3 / 2) ∧
  P = (2, 1) ∧
  point_on_ellipse P.1 P.2 a b ∧
  point_on_ellipse M.1 M.2 a b ∧
  point_on_ellipse N.1 N.2 a b ∧
  kPM * kPN = -1/4

-- Statement of the second part
theorem part2 (a b : ℝ) (kPM kPN : ℝ) (P M N : ℝ × ℝ) :
  conditions_part2 a b kPM kPN P M N ->
  ∃ Q : ℝ × ℝ, point_on_ellipse Q.1 Q.2 a b ∧ (P, M, Q, N ≠ P) :=
by
  sorry

end part1_part2_l295_295378


namespace complex_number_solution_l295_295059

variable {Z : ℂ}

theorem complex_number_solution (h : Z = (2 - Z) * complex.I) : Z = 1 + complex.I := by
  sorry

end complex_number_solution_l295_295059


namespace general_formulas_sum_S_n_l295_295040

noncomputable def a : ℕ → ℝ 
| 0 => 2 
| n+1 => (3 / 4) * a n + (1 / 4) * b n

noncomputable def b : ℕ → ℝ 
| 0 => 1 
| n+1 => (3 / 4) * b n + (1 / 4) * a n

noncomputable def c (n : ℕ) := 3 * a n + b n

noncomputable def S : ℕ → ℝ
| 0 => 0
| n+1 => S n + (↑n+1) * c (↑n+1)

theorem general_formulas (n : ℕ) :
  a n = (3 + (1 / 2)^(n - 1)) / 2 ∧
  b n = (3 - (1 / 2)^(n - 1)) / 2 :=
sorry

theorem sum_S_n (n : ℕ) :
  S n = 3 * n * (n + 1) + 4 * (1 - (1 / 2)^n) - 2 * n * (1 / 2)^n :=
sorry

end general_formulas_sum_S_n_l295_295040


namespace review_organization_possible_l295_295989

-- Define the problem conditions
variables {Students Problems : Type} 
variable [Fintype Students]
variable [Fintype Problems]
variable (n : ℕ) -- number of students and problems

-- Assume six students and six problems
axiom Students_eq_six : Fintype.card Students = 6
axiom Problems_eq_six : Fintype.card Problems = 6

-- Each student solves exactly two problems
axiom student_solves_two : ∀ s : Students, Finset.card (Finset.filter (λ p : Problems, p ∈ s.problems) (Finset.univ : Finset Problems)) = 2

-- Each problem is solved by exactly two students
axiom problem_solved_by_two : ∀ p : Problems, Finset.card (Finset.filter (λ s : Students, s ∈ p.students) (Finset.univ : Finset Students)) = 2

-- Proposition: Each student can present one of the problems they solved such that all problems are reviewed.
theorem review_organization_possible :
  ∃ (presentation : Students → Problems), 
  (∀ s : Students, presentation s ∈ s.problems) ∧ 
  (Finset.univ.image presentation = Finset.univ) :=
sorry

end review_organization_possible_l295_295989


namespace problem_1_problem_2_problem_3_l295_295962

noncomputable def f : ℝ → ℝ
| x := if 0 < x ∧ x ≤ 10 then -0.1 * x ^ 2 + 2.6 * x + 43
       else if 10 < x ∧ x ≤ 16 then 59
       else if 16 < x ∧ x ≤ 30 then -3 * x + 107
       else 0

theorem problem_1 :
  (∀ x, 0 < x ∧ x ≤ 10 → f x ≤ 59) ∧
  (∀ x, 10 < x ∧ x ≤ 16 → f x = 59) ∧
  (∀ x, 16 < x ∧ x ≤ 30 → f x < 59) ∧
  (f 10 = 59) :=
sorry

theorem problem_2 : f 5 > f 20 :=
sorry

theorem problem_3 :
  ¬(∃ start : ℝ, 0 < start ∧ start ≤ 30 ∧
    (∀ t, start ≤ t ∧ t ≤ start + 13 → f t ≥ 55)) :=
sorry

end problem_1_problem_2_problem_3_l295_295962


namespace quadratic_equation_case_1_quadratic_equation_case_2_quadratic_inequality_ge_case_1_quadratic_inequality_ge_case_2_quadratic_inequality_lt_case_1_quadratic_inequality_lt_case_2_l295_295315

variable {a x : ℝ}

theorem quadratic_equation_case_1 (h : x ≥ a) : x^2 + 4x - 2|x - a| + 2 - a = 0 ↔ x^2 + 2x + a + 2 = 0 := by
  sorry

theorem quadratic_equation_case_2 (h : x < a) : x^2 + 4x - 2|x - a| + 2 - a = 0 ↔ x^2 + 6x - 3a + 2 = 0 := by
  sorry

theorem quadratic_inequality_ge_case_1 (h : x ≥ a) : x^2 + 4x - 2|x - a| + 2 - a ≥ 0 ↔ x^2 + 2x + a + 2 ≥ 0 := by
  sorry

theorem quadratic_inequality_ge_case_2 (h : x < a) : x^2 + 4x - 2|x - a| + 2 - a ≥ 0 ↔ x^2 + 6x - 3a + 2 ≥ 0 := by
  sorry

theorem quadratic_inequality_lt_case_1 (h : x ≥ a) : x^2 + 4x - 2|x - a| + 2 - a < 0 ↔ x^2 + 2x + a + 2 < 0 := by
  sorry

theorem quadratic_inequality_lt_case_2 (h : x < a) : x^2 + 4x - 2|x - a| + 2 - a < 0 ↔ x^2 + 6x - 3a + 2 < 0 := by
  sorry

end quadratic_equation_case_1_quadratic_equation_case_2_quadratic_inequality_ge_case_1_quadratic_inequality_ge_case_2_quadratic_inequality_lt_case_1_quadratic_inequality_lt_case_2_l295_295315


namespace coordinates_of_foci_l295_295174

theorem coordinates_of_foci
  (m n : ℝ) (h_mn : m < n) (h_m : m < 0) (h_n : n < 0) :
  ∃ f : ℝ, (f = sqrt (n - m)) ∧ (set_of (λ p : ℝ × ℝ, p = (0, f) ∨ p = (0, -f))) ∈
  { (x, y) : ℝ × ℝ | mx^2 + ny^2 + mn = 0} :=
sorry

end coordinates_of_foci_l295_295174


namespace vec_perpendicular_l295_295044

def vec_perp : Vector (ℝ × ℝ) :=
  let a := (3, 2)
  let b := (2, 3)
  let a_minus_b := (3 - 2, 2 - 3)
  let a_plus_b := (3 + 2, 2 + 3)
  a_minus_b × a_plus_b = (0)

theorem vec_perpendicular :
  ∀ (a b : ℝ × ℝ), a = (3,2) → b = (2,3) → let a_minus_b := (a.1 - b.1, a.2 - b.2)
                                            let a_plus_b := (a.1 + b.1, a.2 + b.2)
                                            a_minus_b × a_plus_b = (0) :=
by
  sorry

end vec_perpendicular_l295_295044


namespace sum_of_seven_primes_with_units_digit_3_eq_291_l295_295722

-- Define the list of the first seven prime numbers with a units digit of 3
def primes_with_units_digit_3 : List ℕ := [3, 13, 23, 43, 53, 73, 83]

-- Assert that all elements in the list are prime
def primes_check (l : List ℕ) : Prop :=
  ∀ n ∈ l, Prime n

-- Declare the main theorem asserting the sum of the first seven prime numbers
theorem sum_of_seven_primes_with_units_digit_3_eq_291 :
  primes_check primes_with_units_digit_3 ∧ (sum primes_with_units_digit_3 = 291) :=
by
  sorry

end sum_of_seven_primes_with_units_digit_3_eq_291_l295_295722


namespace goldfish_distribution_l295_295909

theorem goldfish_distribution : 
    let tanks := [1, 2, 3]
    let goldfish_count := 7
    let condition1 := (2, 2, 3)
    let condition2 := (1, 3, 3)
    let condition3 := (1, 2, 4)
    let combinations :=
        (Nat.choose 7 2) * (Nat.choose 5 2) +
        (Nat.choose 7 1) * (Nat.choose 6 3) +
        (Nat.choose 7 1) * (Nat.choose 6 2)
    in combinations = 455 :=
by
  sorry

end goldfish_distribution_l295_295909


namespace problem_a_problem_b_l295_295281

noncomputable theory
open Classical

variables {A B C D K L M N Q : Point} (trapezoid : Trapezoid ABCD)
(h1 : Parallel (BC) (AD)) (inscribed_circle : InscribedCircle trapezoid)
(touching_at_K : TangentAt inscribed_circle AB K)
(touching_at_L : TangentAt inscribed_circle CD L)
(touching_at_M : TangentAt inscribed_circle AD M)
(touching_at_N : TangentAt inscribed_circle BC N)
(intersect_Q : IntersectionPoint (BM) (AN) Q)

theorem problem_a : Parallel (KQ) (AD) :=
by
  sorry

theorem problem_b : length (AK) * length (KB) = length (CL) * length (LD) :=
by
  sorry

end problem_a_problem_b_l295_295281


namespace each_child_receive_amount_l295_295268

def husband_weekly_contribution : ℕ := 335
def wife_weekly_contribution : ℕ := 225
def weeks_in_month : ℕ := 4
def months : ℕ := 6
def children : ℕ := 4

noncomputable def total_weekly_contribution : ℕ := husband_weekly_contribution + wife_weekly_contribution
noncomputable def total_savings : ℕ := total_weekly_contribution * (weeks_in_month * months)
noncomputable def half_savings : ℕ := total_savings / 2
noncomputable def amount_per_child : ℕ := half_savings / children

theorem each_child_receive_amount :
  amount_per_child = 1680 :=
by
  sorry

end each_child_receive_amount_l295_295268


namespace systematic_sampling_employee_l295_295678

theorem systematic_sampling_employee {x : ℕ} (h1 : 1 ≤ 6 ∧ 6 ≤ 52) (h2 : 1 ≤ 32 ∧ 32 ≤ 52) (h3 : 1 ≤ 45 ∧ 45 ≤ 52) (h4 : 6 + 45 = x + 32) : x = 19 :=
  by
    sorry

end systematic_sampling_employee_l295_295678


namespace triangle_inradius_exradii_relation_l295_295252

theorem triangle_inradius_exradii_relation
  (a b c : ℝ) (S : ℝ) (r r_a r_b r_c : ℝ)
  (h_inradius : S = (1/2) * r * (a + b + c))
  (h_exradii_a : r_a = 2 * S / (b + c - a))
  (h_exradii_b : r_b = 2 * S / (c + a - b))
  (h_exradii_c : r_c = 2 * S / (a + b - c))
  (h_area : S = (1/2) * (a * r_a + b * r_b + c * r_c - a * r - b * r - c * r)) :
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  by sorry

end triangle_inradius_exradii_relation_l295_295252


namespace price_of_pendants_min_cost_of_pendants_l295_295984

theorem price_of_pendants :
  ∃ (a b : ℕ), 2 * a + b = 26 ∧ 4 * a + 3 * b = 62 ∧ a = 8 ∧ b = 10 :=
begin
  sorry
end

theorem min_cost_of_pendants :
  ∀ (x y : ℕ), x + y = 100 ∧ y ≥ x / 3 → ∃ w, w = 8 * x + 10 * y ∧ w = 850 :=
begin
  sorry
end

end price_of_pendants_min_cost_of_pendants_l295_295984


namespace find_common_ratio_sum_first_n_terms_l295_295111

-- Definitions and conditions for Part (1)
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_mean_condition (a : ℕ → ℝ) : Prop :=
  2 * a 0 = a 1 + a 2

-- Theorem for Part (1)
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) (ham : arithmetic_mean_condition a) (hq : q ≠ 1) :
  q = -2 :=
sorry

-- Definitions and conditions for Part (2)
def sum_first_n_na_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) * a i

-- Theorem for Part (2)
theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) 
  (hg : is_geometric_sequence a (-2)) (ha1 : a 0 = 1) :
  sum_first_n_na_n a n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
sorry

end find_common_ratio_sum_first_n_terms_l295_295111


namespace height_of_vertex_after_rotation_l295_295961

theorem height_of_vertex_after_rotation :
  ∀ (side_length : ℝ) (θ : ℝ) (base_height : ℝ),
    side_length = 2 →
    θ = 30 →
    base_height = (2 * Real.sqrt 2) / 2 →
    let additional_height := (Real.sqrt 2) / 2 in
    (base_height + additional_height) = (3 * Real.sqrt 2) / 2 :=
begin
  intros side_length θ base_height h1 h2 h3,
  let additional_height := (Real.sqrt 2) / 2,
  sorry,
end

end height_of_vertex_after_rotation_l295_295961


namespace gpa_of_entire_class_l295_295065

def students : ℕ := 200

def gpa1_num : ℕ := 18 * students / 100
def gpa2_num : ℕ := 27 * students / 100
def gpa3_num : ℕ := 22 * students / 100
def gpa4_num : ℕ := 12 * students / 100
def gpa5_num : ℕ := students - (gpa1_num + gpa2_num + gpa3_num + gpa4_num)

def gpa1 : ℕ := 58
def gpa2 : ℕ := 63
def gpa3 : ℕ := 69
def gpa4 : ℕ := 75
def gpa5 : ℕ := 85

def total_points : ℕ :=
  (gpa1_num * gpa1) + (gpa2_num * gpa2) + (gpa3_num * gpa3) + (gpa4_num * gpa4) + (gpa5_num * gpa5)

def class_gpa : ℚ := total_points / students

theorem gpa_of_entire_class :
  class_gpa = 69.48 := 
  by
  sorry

end gpa_of_entire_class_l295_295065


namespace S_finite_l295_295348

noncomputable def sequence (a₀ : ℕ) : ℕ → ℕ
| 0       := a₀
| (k + 1) := minimal (λ x, x > sequence k ∧ ∃ b : ℕ, b^2 = sequence k + x)

def S (a₀ : ℕ) : Set ℕ :=
{ n | ∀ i j, i ≠ j → n ≠ sequence a₀ i - sequence a₀ j }

theorem S_finite (a₀ : ℕ) : (S a₀).to_finset.finite :=
sorry

end S_finite_l295_295348


namespace minimize_y_l295_295876

def y (x a b : ℝ) : ℝ := (x-a)^2 * (x-b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, y x a b = 0 := by
  use a
  sorry

end minimize_y_l295_295876


namespace parallel_lines_necessary_and_sufficient_l295_295416

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l295_295416


namespace prob_second_shot_l295_295676

theorem prob_second_shot (P_A : ℝ) (P_AB : ℝ) (p : ℝ) : 
  P_A = 0.75 → 
  P_AB = 0.6 → 
  P_A * p = P_AB → 
  p = 0.8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end prob_second_shot_l295_295676


namespace correct_statements_l295_295767

variable {f : ℝ → ℝ}
variable (hf_even : ∀ x, f x = f (-x)) 
variable (hf_period : ∀ x, f (x + 6) = f x + f 3)
variable (hf_increasing : ∀ x1 x2 ∈ set.Icc 0 3, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem correct_statements :
  (f 3 = 0) ∧
  (∀ x, f (-6 - x) = f (-6 + x)) ∧
  (f (-9) = 0 ∧ f (-3) = 0 ∧ f 3 = 0 ∧ f 9 = 0) :=
by
  sorry

end correct_statements_l295_295767


namespace probability_twice_as_large_l295_295307

noncomputable def probability_geq_twice_as_large : ℝ :=
  let f := λ (y : ℝ), (if y ∈ Set.Icc 0 3000 then Set.Icc 0 (y/2) else ∅) in
  let area_f := ∫ (y : ℝ) in 0..3000, volume (f y) in
  let total_area := ∫ (y : ℝ) in 0..3000, volume (Set.Icc 0 y) in
  area_f / total_area

theorem probability_twice_as_large : probability_geq_twice_as_large = 1/2 :=
sorry

end probability_twice_as_large_l295_295307


namespace max_similar_triangles_l295_295478

variables {A B C L M H D : Type}
variables [triangle ABC : geom.triangle A B C]
variables [altitude AL : geom.altitude A L]
variables [altitude BM : geom.altitude B M]

theorem max_similar_triangles (ABC_acute : geom.acute_triangle A B C) (altitude1 : geom.altitude A L) (altitude2 : geom.altitude B M) (LM_extends_AB : geom.extended_intersection (geom.line_through L M) (geom.extended_side A B) D) :
  geom.max_similar_triangles ABC = 10 := 
sorry

end max_similar_triangles_l295_295478


namespace surface_area_correct_l295_295539

-- Definitions of the problem conditions
def volume_of_solid : ℝ := 875
def number_of_cubes : ℕ := 7
def surface_area_of_solid : ℝ := 750

-- The proof problem statement
theorem surface_area_correct : 
  ∀ (vol : ℝ) (n : ℕ), 
  (vol = volume_of_solid) → 
  (n = number_of_cubes) → 
  let volume_of_each_cube := vol / n in 
  let side_length := real.sqrt (volume_of_each_cube ^ (1/3)) in 
  let area_of_one_face := side_length^2 in 
  let total_surface_area := (n - 1) * area_of_one_face * 5 in
  total_surface_area = surface_area_of_solid :=
by
  intros vol n hvol hn
  -- Definitions
  let volume_of_each_cube := vol / n
  let side_length := real.sqrt (volume_of_each_cube ^ (1/3))
  let area_of_one_face := side_length^2
  let total_surface_area := (n - 1) * area_of_one_face * 5
  -- Proof will be provided here
  sorry

end surface_area_correct_l295_295539


namespace field_trip_students_l295_295564

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end field_trip_students_l295_295564


namespace turn_all_black_l295_295467

def invertColor (v : Vertex) (G : Graph) : Graph := sorry

theorem turn_all_black (G : Graph) (n : ℕ) (whiteBlack : Vertex → Bool) :
  (∀ v : Vertex, whiteBlack v = false) :=
by
 -- Providing the base case for induction
  induction n with 
  | zero => sorry -- The base case for graphs with one vertex
  | succ n ih =>
    -- Inductive step: assume true for graph with n vertices and prove for graph with n+1 vertices
    sorry

end turn_all_black_l295_295467


namespace largest_diff_of_set_l295_295227

def largest_diff (s : set ℤ) : ℤ :=
  s.sup id - s.inf id

theorem largest_diff_of_set :
  largest_diff ({-20, -5, 0, 3, 7, 15} : set ℤ) = 35 :=
by
  sorry

end largest_diff_of_set_l295_295227


namespace prove_cos_C_l295_295491

variables (a b c : ℝ) (A B C : ℝ)

-- a = 2, b = 3, C = 2A
def conditions : Prop := a = 2 ∧ b = 3 ∧ C = 2 * A

-- Question: cos C
def cos_C := (a^2 + b^2 - c^2) / (2 * a * b)

theorem prove_cos_C (h : conditions) : cos_C a b c = 1 / 4 :=
by
  obtain ⟨ha, hb, hC⟩ := h
  sorry

end prove_cos_C_l295_295491


namespace expected_value_binomial_l295_295418

-- Define the parameters for the binomial distribution
def X : Binomial 6 (1/2)

-- Statement of the theorem
theorem expected_value_binomial : E(X) = 3 := by
  sorry

end expected_value_binomial_l295_295418


namespace false_statement_l295_295502

open Set

def is_good_set (S : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ^ 2 - b ^ 2 ∈ S

theorem false_statement (S : Set ℤ) (hS : is_good_set S) (h_nonempty : S.nonempty) :
  ¬ (∀ x y : ℤ, x ∈ S → y ∈ S → x * y ∈ S) :=
by
  sorry

end false_statement_l295_295502


namespace certain_number_power_l295_295055

theorem certain_number_power (k : ℕ) (hk : k = 11): (1/2)^22 * (1/81)^k = (1/354294)^22 :=
by
  rw hk
  sorry

end certain_number_power_l295_295055


namespace backpack_prices_purchasing_plans_backpacks_given_away_l295_295261

-- Part 1: Prices of Type A and Type B backpacks
theorem backpack_prices (x y : ℝ) (h1 : x = 2 * y - 30) (h2 : 2 * x + 3 * y = 255) : x = 60 ∧ y = 45 :=
sorry

-- Part 2: Possible purchasing plans
theorem purchasing_plans (m : ℕ) (h1 : 8900 ≥ 50 * m + 40 * (200 - m)) (h2 : m > 87) : 
  m = 88 ∨ m = 89 ∨ m = 90 :=
sorry

-- Part 3: Number of backpacks given away
theorem backpacks_given_away (m n : ℕ) (total_A : ℕ := 89) (total_B : ℕ := 111) 
(h1 : m + n = 4) 
(h2 : 1250 = (total_A - if total_A > 10 then total_A / 10 else 0) * 60 + (total_B - if total_B > 10 then total_B / 10 else 0) * 45 - (50 * total_A + 40 * total_B)) :
m = 1 ∧ n = 3 := 
sorry

end backpack_prices_purchasing_plans_backpacks_given_away_l295_295261


namespace find_fg_of_3_l295_295788

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 4 * x - 5

theorem find_fg_of_3 : f (g 3) = 31 := by
  sorry

end find_fg_of_3_l295_295788


namespace paths_A_to_B_valid_count_l295_295897

def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

def forbidden_paths (m n : ℕ) (x y : ℕ) (paths_after : ℕ) : ℕ :=
  grid_paths x y * paths_after

def valid_paths (total_paths forbidden : ℕ) : ℕ :=
  total_paths - forbidden

theorem paths_A_to_B_valid_count :
  let total_paths := grid_paths 6 4 in
  let forbidden_1 := forbidden_paths 2 4 2 (grid_paths 4 4) in
  let forbidden_2 := forbidden_paths 4 1 4 (grid_paths 2 1) in
  valid_paths total_paths (forbidden_1 + forbidden_2) = 35 :=
by 
  let total_paths := grid_paths 6 4
  let forbidden_1 := forbidden_paths 2 4 2 (grid_paths 4 4)
  let forbidden_2 := forbidden_paths 4 1 4 (grid_paths 2 1)
  have h1 : total_paths = 210 := sorry
  have h2 : forbidden_1 = 70 := sorry
  have h3 : forbidden_2 = 105 := sorry
  have h4 : valid_paths total_paths (forbidden_1 + forbidden_2) = 35 := sorry
  exact h4

end paths_A_to_B_valid_count_l295_295897


namespace sum_of_xyz_l295_295761

theorem sum_of_xyz (x y z : ℝ) (h : (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l295_295761


namespace express_y_in_terms_of_x_x_range_for_y_l295_295411

theorem express_y_in_terms_of_x (x y : ℝ) (h : x + 2*y = -6) : y = -3 - x / 2 := 
  sorry

theorem x_range_for_y (x : ℝ) (hx_bounds : -6 < x ∧ x < -2) : 
  let y := -3 - x / 2 
  in -2 < y ∧ y < 0 :=
  sorry

end express_y_in_terms_of_x_x_range_for_y_l295_295411


namespace total_splash_width_l295_295606

theorem total_splash_width :
  let pebble_splash := 1 / 4
  let rock_splash := 1 / 2
  let boulder_splash := 2
  let pebbles := 6
  let rocks := 3
  let boulders := 2
  let total_pebble_splash := pebbles * pebble_splash
  let total_rock_splash := rocks * rock_splash
  let total_boulder_splash := boulders * boulder_splash
  let total_splash := total_pebble_splash + total_rock_splash + total_boulder_splash
  total_splash = 7 := by
  sorry

end total_splash_width_l295_295606


namespace find_g_3_8_l295_295183

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l295_295183


namespace interval_of_monotonic_increase_sum_of_sequence_l295_295801

-- Condition definitions
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1/2)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos (2*x + Real.pi/6))
def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

-- Problem statements (without proofs)
theorem interval_of_monotonic_increase (x : ℝ) (k : ℤ) :
  ((k : ℝ) * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + Real.pi / 3) ↔ 
  (f' x = 0 ∧ (∀ y, ((k : ℝ) * Real.pi - Real.pi / 6 < y ∧ y < x) → f' y ≥ 0) ∧ (∀ y, (x < y ∧ y < (k : ℝ) * Real.pi + Real.pi / 3) → f' y ≤ 0)) :=
sorry

noncomputable def local_max_points_seq (n : ℕ) : ℝ := (n : ℝ) * Real.pi + Real.pi / 3

theorem sum_of_sequence (n : ℕ) :
  (∑ i in Finset.range n, Real.pi^2 / (local_max_points_seq i * local_max_points_seq (i + 1))) = 9 * (n : ℝ) / (3 * (n : ℝ) + 1) :=
sorry

end interval_of_monotonic_increase_sum_of_sequence_l295_295801


namespace ccamathbonanza_2016_2_1_l295_295681

-- Definitions of the speeds of the runners
def bhairav_speed := 28 -- in miles per hour
def daniel_speed := 15 -- in miles per hour
def tristan_speed := 10 -- in miles per hour

-- Distance of the race
def race_distance := 15 -- in miles

-- Time conversion from hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- Time taken by each runner to complete the race (in hours)
def time_bhairav := race_distance / bhairav_speed
def time_daniel := race_distance / daniel_speed
def time_tristan := race_distance / tristan_speed

-- Time taken by each runner to complete the race (in minutes)
def time_bhairav_minutes := hours_to_minutes time_bhairav
def time_daniel_minutes := hours_to_minutes time_daniel
def time_tristan_minutes := hours_to_minutes time_tristan

-- Time differences between consecutive runners' finishes (in minutes)
def time_diff_bhairav_daniel := time_daniel_minutes - time_bhairav_minutes
def time_diff_daniel_tristan := time_tristan_minutes - time_daniel_minutes

-- Greatest length of time between consecutive runners' finishes
def greatest_time_diff := max time_diff_bhairav_daniel time_diff_daniel_tristan

-- The theorem we need to prove
theorem ccamathbonanza_2016_2_1 : greatest_time_diff = 30 := by
  sorry

end ccamathbonanza_2016_2_1_l295_295681


namespace intersection_complement_eq_l295_295809

-- Definitions as per given conditions
def U : Set ℕ := { x | x > 0 ∧ x < 9 }
def A : Set ℕ := { 1, 2, 3, 4 }
def B : Set ℕ := { 3, 4, 5, 6 }

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Statement of the theorem to be proved
theorem intersection_complement_eq : A ∩ C_U_B = { 1, 2 } :=
by
  sorry

end intersection_complement_eq_l295_295809


namespace perpendicular_AK_MN_l295_295877

-- Definitions and facts about the geometric entities involved

variables {A B C D M N K : Point} -- Define points A, B, C, D, M, N, K

-- Precondition: Quadrilateral ABCD is convex
def is_convex (A B C D : Point) : Prop := sorry 

-- Precondition: Angles ABC and ADC are equal
def angle_eq (A B C D : Point) : Prop := sorry 

-- Define M as the foot of the perpendicular from A to BC
def is_foot_of_perpendicular (M A B C : Point) : Prop := sorry 

-- Define N as the foot of the perpendicular from A to CD
def is_foot_of_perpendicular (N A C D : Point) : Prop := sorry 

-- K is the intersection of lines MD and NB
def is_intersection (K M D N B : Point) : Prop := sorry 

-- We need to prove AK and MN are perpendicular
theorem perpendicular_AK_MN
  (h1 : is_convex A B C D)
  (h2 : angle_eq A B C D)
  (h3 : is_foot_of_perpendicular M A B C)
  (h4 : is_foot_of_perpendicular N A C D)
  (h5 : is_intersection K M D N B) :
  is_perpendicular (line_through A K) (line_through M N) :=
sorry

end perpendicular_AK_MN_l295_295877


namespace new_acute_angle_ACB_l295_295578

-- Define the initial condition: the measure of angle ACB is 50 degrees.
def measure_ACB_initial : ℝ := 50

-- Define the rotation: ray CA is rotated by 540 degrees clockwise.
def rotation_CW_degrees : ℝ := 540

-- Theorem statement: The positive measure of the new acute angle ACB.
theorem new_acute_angle_ACB : 
  ∃ (new_angle : ℝ), new_angle = 50 ∧ new_angle < 90 := 
by
  sorry

end new_acute_angle_ACB_l295_295578


namespace harmonic_mean_of_1_3_6_12_l295_295574

theorem harmonic_mean_of_1_3_6_12 : 
  let H : ℚ := harmonic_mean [1, 3, 6, 12] in H = 48 / 19 :=
by
  sorry

end harmonic_mean_of_1_3_6_12_l295_295574


namespace tangent_from_origin_l295_295690

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (6, 14)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function that computes the length of the tangent from O to the circle passing through A, B, and C
noncomputable def tangent_length : ℝ :=
 sorry -- Placeholder for the actual calculation

-- The theorem we need to prove: The length of the tangent from O to the circle passing through A, B, and C is as calculated
theorem tangent_from_origin (L : ℕ) : 
  tangent_length = L := 
 sorry -- Placeholder for the proof

end tangent_from_origin_l295_295690


namespace min_value_f_x_min_value_a2_b2_c2_l295_295383

theorem min_value_f_x (x : ℝ) : ∃ (m : ℝ), (m = 1 ∧ ∀ y : ℝ, |y - 1| + |y - 2| ≥ m) :=
begin
  use 1,
  split,
  { refl },
  { intro y,
    by_cases (y ≤ 1),
    { have h1 : |y - 1| + |y - 2| = -2 * y + 3 := by {
        rw [abs_of_neg _ (lt_of_le_of_ne h (ne_of_lt (lt_trans (by linarith) (by linarith)))),
           abs_of_nonpos (by linarith : y - 2 ≤ 0)],
        linarith
      },
      linarith },
    { by_cases (y < 2),
      { have h2 : |y - 1| + |y - 2| = 1 := by {
          rw [abs_of_pos (lt_of_le_of_ne (le_of_not_lt h) (by rfl : 1 ≠ y)),
             abs_of_nonpos (by linarith)],
          linarith
        },
        linarith },
      { have h3 : |y - 1| + |y - 2| = 2 * y - 3 := by {
          rw [abs_of_pos (by linarith : y - 1 ≥ 0),
             abs_of_nonneg (by linarith : y - 2 ≥ 0)],
          linarith
        },
        linarith }
    }
  }
end

theorem min_value_a2_b2_c2 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : ∃ (m : ℝ), m = 1/3 ∧ (a^2 + b^2 + c^2 ≥ m) :=
begin
  use 1 / 3,
  split,
  { refl },
  {
    calc
      a^2 + b^2 + c^2 ≥ 1 / 3 : by {
        let s := a^2 + b^2 + c^2,
        have h1 : s + 2 * (a * b + b * c + c * a) = 1 := by {
           calc
             s + 2 * (a * b + b * c + c * a) = (a + b + c)^2 : by ring
                                      ... = 1 : by rw [h_sum, one_mul]
        },
        have h2 : 2 * (a * b + b * c + c * a) ≤ 2 * (a^2 + b^2 + c^2) : by {
          linarith [mul_le_mul_of_nonneg_left (add_le_add (add_le_add (mul_self_le_mul_self_of_nonneg a) (mul_self_le_mul_self_of_nonneg b)) (mul_self_le_mul_self_of_nonneg c)) zero_le_two]
        },
        linarith,
      }
  }
end

end min_value_f_x_min_value_a2_b2_c2_l295_295383


namespace largest_area_among_given_shapes_l295_295625

/-- Definition of the geometrical problem -/
def triangle_A_area (A B C : ℝ) (angle_A angle_B : ℝ) (side_AC : ℝ): ℝ := 
-- Circumradius formula (place holder, real area calculation is setup but not necessary for proof placeholder)
1

def trapezoid_area (d1 d2 : ℝ) (angle : ℝ): ℝ :=
0.5 * d1 * d2 * (Real.sin angle)

def circle_area (r : ℝ): ℝ :=
real.pi * r^2

def squared_diagonal_to_area (d : ℝ): ℝ :=
let a := d / Real.sqrt 2 in a^2

theorem largest_area_among_given_shapes :
  let area_triangle := triangle_A_area 1 1 1 (real.pi / 3) (real.pi / 4) (Real.sqrt 2),
      area_trapezoid := trapezoid_area (Real.sqrt 2) (Real.sqrt 3) (75 * real.pi / 180),
      area_circle := circle_area 1,
      area_square := squared_diagonal_to_area 2.5
  in area_circle > max area_triangle (max area_trapezoid area_square) :=
by
  sorry

end largest_area_among_given_shapes_l295_295625


namespace total_surface_area_correct_l295_295655

-- Define the heights of the pieces
def height_A : ℝ := 1 / 3
def height_B : ℝ := 1 / 4
def height_C : ℝ := 1 / 6
def height_D : ℝ := 1 - (height_A + height_B + height_C)

-- Define the total surface area calculation
def total_surface_area : ℝ :=
  let top_bottom_area : ℝ := 2
  let side_area : ℝ := (height_A + height_B + height_C + height_D) * 4
  top_bottom_area + side_area

-- Theorem statement
theorem total_surface_area_correct : total_surface_area = 6 := by
  sorry

end total_surface_area_correct_l295_295655


namespace correlation_coefficient_is_one_l295_295376

noncomputable def sample_data (n : ℕ) (h : n ≥ 2) : Type :=
  { xy : Fin n → ℝ × ℝ // ∃ (f : ℝ → ℝ), (∀ i, (xy i).2 = f ((xy i).1)) ∧ (∀ i j, (xy i).1 = (xy j).1 → i = j) }

theorem correlation_coefficient_is_one :
  ∀ {n : ℕ} (h : n ≥ 2) (data : sample_data n h),
  (∀ (i : Fin n), (data.val i).2 = 2 * (data.val i).1 + 1) →
  ∃ r, r = 1 :=
by
  intros n h data hline
  use 1
  sorry

end correlation_coefficient_is_one_l295_295376


namespace part_a_part_b_l295_295946

-- Define the structure and conditions of the problem
structure FootballChampionship :=
(teams : Finset ℕ)
(games : list (ℕ × ℕ))
(alternates_side_per_game : ℕ → bool)

-- Assume 10 teams
def num_teams : ℕ := 10
def teams : Finset ℕ := Finset.range num_teams

-- Helper function to determine if a schedule is valid
def valid_schedule (games : list (ℕ × ℕ)) (alternates : ℕ → bool) :=
  ∀ (i < num_teams) (j < num_teams), i ≠ j → ((i, j) ∈ games ∨ (j, i) ∈ games)
  ∧ (∀ i < num_teams, alternating_games_condition i alternates)
  ∧ (no_team_play_more_than_once_in_a_day games)

-- Define the alternating games condition
def alternating_games_condition (team : ℕ) (alternates : ℕ → bool) := sorry
def no_team_play_more_than_once_in_a_day (games : list (ℕ × ℕ)) := sorry

-- Formal statements for part (a) and part (b)
theorem part_a: ∃ games alternates, valid_schedule games alternates := sorry

theorem part_b: ¬∃ games alternates, valid_schedule (one_game_per_day games) alternates := sorry

end part_a_part_b_l295_295946


namespace sequence_has_11th_term_l295_295080

theorem sequence_has_11th_term :
  ∃ n : ℕ, n = 11 ∧ (3 * n - 1 = 32) :=
by
  have h : 3 * 11 - 1 = 32 := by norm_num
  use 11
  exact ⟨rfl, h⟩

end sequence_has_11th_term_l295_295080


namespace k_for_circle_radius_7_l295_295728

theorem k_for_circle_radius_7 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0) →
  (∃ x y : ℝ, (x + 4)^2 + (y + 2)^2 = 49) →
  k = 29 :=
by
  sorry

end k_for_circle_radius_7_l295_295728


namespace Margie_can_drive_400_miles_l295_295135

/-- Margie's car can travel 40 miles per gallon, and gas costs $5 per gallon.
    Prove that Margie can drive 400 miles with $50 worth of gas. -/
theorem Margie_can_drive_400_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (money : ℕ) :
  miles_per_gallon = 40 → cost_per_gallon = 5 → money = 50 → (money / cost_per_gallon) * miles_per_gallon = 400 :=
begin
  intros mpg cpg m hmpg hcpg hm,
  rw [hmpg, hcpg, hm],
  norm_num,
  sorry
end

end Margie_can_drive_400_miles_l295_295135


namespace range_of_a_l295_295518

noncomputable def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (1 < a ∨ -1 < a ∧ a < 1) :=
by sorry

end range_of_a_l295_295518


namespace ice_creams_needed_l295_295156

theorem ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) (games_to_buy : ℕ) 
    (h1 : game_cost = 60) (h2 : ice_cream_price = 5) (h3 : games_to_buy = 2) : 
    (games_to_buy * game_cost) / ice_cream_price = 24 :=
by
  rw [h1, h2, h3]
  sorry

end ice_creams_needed_l295_295156


namespace part_a_part_b_l295_295637

noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def psi := classical.some (exists_unique (λ x : ℝ, x^3 - x^2 - x - 1 = 0))

def binary_string_set (s : String) : Prop :=
  s = "1" ∨
  (∃ t, binary_string_set t ∧ String.length t % 2 = 1 ∧ ((s = t ++ "0") ∨ (s = "0" ++ t))) ∨
  (∃ t, binary_string_set t ∧ String.length t % 2 = 0 ∧ ((s = t ++ "1") ∨ (s = "1" ++ t)))

def valid_binary_strings (n : ℕ) : ℕ :=
  Finset.card (Finset.filter (λ s, binary_string_set s = true) (Finset.range n))

theorem part_a :
  ∃ (c1 c2 : ℝ) (λ1 λ2 : ℝ),
    0 < c1 ∧ 0 < c2 ∧ 1.6 < λ1 ∧ λ1 < 1.9 ∧ 1.6 < λ2 ∧ λ2 < 1.9 ∧
    ∀ n : ℕ, c1 * λ1 ^ n < valid_binary_strings n ∧ valid_binary_strings n < c2 * λ2 ^ n :=
sorry

theorem part_b :
  Real.limsup (λ n, Real.sqrt (valid_binary_strings n)) = psi ∧
  Real.liminf (λ n, Real.sqrt (valid_binary_strings n)) = phi :=
sorry

end part_a_part_b_l295_295637


namespace seating_arrangement_l295_295631

theorem seating_arrangement (students : ℕ) (desks : ℕ) (empty_desks : ℕ) 
  (h_students : students = 2) (h_desks : desks = 5) 
  (h_empty : empty_desks ≥ 1) :
  ∃ ways, ways = 12 := by
  sorry

end seating_arrangement_l295_295631


namespace abs_lt_pairs_count_l295_295736

def A : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}

def num_abs_lt_pairs (s : Set ℤ) : ℕ :=
  Set.toFinset s
    .sum (λ a => Set.toFinset s
         .count (λ b => |a| < |b|))

theorem abs_lt_pairs_count :
  num_abs_lt_pairs A = 18 :=
by
  -- Problem conditions
  let A := {-3, -2, -1, 0, 1, 2, 3}
  have len_A : A.card = 7 := rfl

  -- Transforms the set into a finset for counting
  let finA := A.to_finset

  -- Count pairs (a, b) such that |a| < |b|
  let count := finA.sum (λ a => finA.count (λ b => |a| < |b|))

  -- Prove the count is 18
  show count = 18
  sorry

end abs_lt_pairs_count_l295_295736


namespace find_ratio_l295_295020

variable {a b c A B C : ℝ}
variable (h_triangle : True) -- Assume a, b, c are sides of triangle ABC opposite to angles A, B, C respectively
variable (h_sin : sin B ^ 2 = 2 * sin A * sin C)
variable (h_cos : cos B = 1 / 4)
variable (h_ineq : a > c)

theorem find_ratio (h_triangle : True) 
    (h_sin : sin B ^ 2 = 2 * sin A * sin C)
    (h_cos : cos B = 1 / 4)
    (h_ineq : a > c) : 
    a / c = 2 := by
  sorry

end find_ratio_l295_295020


namespace log_eq_and_pos_implies_gt_l295_295002

theorem log_eq_and_pos_implies_gt {x y : ℝ} (hx : x > 0) (hy : y > 0) 
  (h : log 2 x + 2 * x = log 2 y + 3 * y) : x > y :=
sorry

end log_eq_and_pos_implies_gt_l295_295002


namespace scores_double_impossible_l295_295485

theorem scores_double_impossible (f : Fin 10 → ℤ)  (h_distinct : Pairwise (λ i j : Fin 10, i ≠ j → (f i) % 10 ≠ (f j) % 10)) :
  ¬ ∃ g : Fin 10 → ℤ, (∑ i, g i) = 2 * (∑ i, f i) :=
by
  sorry

end scores_double_impossible_l295_295485


namespace analytical_expression_of_f_l295_295170

def f : ℝ → ℝ := sorry

axiom condition : ∀ (x y : ℝ), f(x * y) = f(x) + f(y) + 1

theorem analytical_expression_of_f : f = (λ x : ℝ, -1) :=
by {
  sorry
}

end analytical_expression_of_f_l295_295170


namespace ice_creams_needed_l295_295157

theorem ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) (games_to_buy : ℕ) 
    (h1 : game_cost = 60) (h2 : ice_cream_price = 5) (h3 : games_to_buy = 2) : 
    (games_to_buy * game_cost) / ice_cream_price = 24 :=
by
  rw [h1, h2, h3]
  sorry

end ice_creams_needed_l295_295157


namespace count_unbroken_matches_l295_295494

theorem count_unbroken_matches :
  let n_1 := 5 * 12  -- number of boxes in the first set
  let matches_1 := n_1 * 20  -- total matches in first set of boxes
  let broken_1 := n_1 * 3  -- total broken matches in first set of boxes
  let unbroken_1 := matches_1 - broken_1  -- unbroken matches in first set of boxes

  let n_2 := 4  -- number of extra boxes
  let matches_2 := n_2 * 25  -- total matches in extra boxes
  let broken_2 := (matches_2 / 5)  -- total broken matches in extra boxes (20%)
  let unbroken_2 := matches_2 - broken_2  -- unbroken matches in extra boxes

  let total_unbroken := unbroken_1 + unbroken_2  -- total unbroken matches

  total_unbroken = 1100 := 
by
  sorry

end count_unbroken_matches_l295_295494


namespace probability_of_PRAZDNIK_l295_295653

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem probability_of_PRAZDNIK : 
  let letters := ['K', 'I', 'R', 'D', 'A', 'N', 'Z', 'P'],
      n := factorial 8,
      m := 1 in
  n = 40320 ∧ (m / n : ℚ) = 1 / 40320 := 
by
  let letters := ['K', 'I', 'R', 'D', 'A', 'N', 'Z', 'P']
  let n := factorial 8
  have h_factorial_8 : n = 40320 := rfl
  have h_probability: (m : ℚ) / n = 1 / 40320 :=
    have : (m / n : ℚ) = (1 / 40320 : ℚ) := by
      rw [Nat.cast_one, h_factorial_8, Nat.cast_mul, Nat.cast_succ, Nat.cast_zero]
    this
  exact ⟨h_factorial_8, h_probability⟩

end probability_of_PRAZDNIK_l295_295653


namespace min_value_pt_qu_rv_sw_l295_295123

open Real

theorem min_value_pt_qu_rv_sw (p q r s t u v w : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) 
  (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hpqrs : p * q * r * s = 16) 
  (htuvw : t * u * v * w = 25) :
  (pt² + qu² + rv² + sw²) ≥ 40 :=
sorry

end min_value_pt_qu_rv_sw_l295_295123


namespace imaginary_part_of_conjugate_l295_295816

open Complex

noncomputable def complex_number_z (z : ℂ) : Prop :=
  z * (1 - ⟨0, 1⟩) = Complex.norm (1 - ⟨0, 1⟩) + ⟨0, 1⟩

theorem imaginary_part_of_conjugate {z : ℂ} (h : complex_number_z z) :
  Im (conj z) = - (Real.sqrt 2 + 1) / 2 :=
sorry

end imaginary_part_of_conjugate_l295_295816


namespace max_sqrt_sum_l295_295120

theorem max_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 7) :
    sqrt (3 * x + 4) + sqrt (3 * y + 4) + sqrt (3 * z + 4) ≤ 6 * sqrt 5 := 
sorry

end max_sqrt_sum_l295_295120


namespace new_profit_percentage_l295_295697

theorem new_profit_percentage (initial_cost new_cost : ℝ) (initial_profit_percent : ℝ) (P : ℝ)
  (h1 : initial_cost = 70)
  (h2 : new_cost = 50)
  (h3 : initial_profit_percent = 0.30)
  (h4 : P = initial_cost + initial_profit_percent * P) :
  (new_cost = 50) →
  ((P - new_cost) / P * 100 = 50) :=
by
  intros
  rw [h1, h2, h3, h4]
  -- sorry will leave the proof incomplete.
  sorry

end new_profit_percentage_l295_295697


namespace milk_needed_l295_295528

-- Defining the conditions
def flour : ℕ := 1200
def milk_ratio : ℕ := 50
def flour_per_portion : ℕ := 250

-- Defining the statement to prove
theorem milk_needed : (flour / flour_per_portion) * milk_ratio = 240 :=
by {
  -- Adding the ratio into equation
  refine eq.trans _ rfl,
  -- LHS of the proof
  refine congr_arg2 nat.mul rfl rfl,
  sorry
}

end milk_needed_l295_295528


namespace one_serving_weight_l295_295354

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end one_serving_weight_l295_295354


namespace Simon_total_treasures_l295_295916

-- Definitions of quantities as per provided conditions.
def sand_dollars : ℕ := 10
def glass (sand_dollars: ℕ) := 3 * sand_dollars
def seashells (glass: ℕ) := 5 * glass
def total_treasures (sand_dollars glass seashells: ℕ) := sand_dollars + glass + seashells

-- The main theorem statement proving the total number of treasures
theorem Simon_total_treasures :
  ∃ sand_dollars glass seashells,
    glass = 3 * sand_dollars ∧
    seashells = 5 * glass ∧
    total_treasures sand_dollars glass seashells = 190 :=
by
  use sand_dollars
  use glass sand_dollars
  use seashells (glass sand_dollars)
  simp [sand_dollars, glass, seashells, total_treasures]
  sorry

end Simon_total_treasures_l295_295916


namespace compute_path_length_l295_295569

-- Definitions based on conditions
def diameter_AB : ℝ := 12
def length_AC : ℝ := 3
def length_BD : ℝ := 5
def length_CD := diameter_AB - length_AC - length_BD
def P_is_B := true

-- Theorem statement
theorem compute_path_length 
  (diameter_AB = 12) 
  (length_AC = 3)
  (length_BD = 5)
  (P_is_B : P_is_B)
  : length_AC + diameter_AB + length_BD = 20 := 
sorry

end compute_path_length_l295_295569


namespace concyclic_points_in_rhombus_l295_295638

theorem concyclic_points_in_rhombus
  (A B C D E S : Point) 
  (h_rhombus : rhombus A B C D)
  (h_angle : ∠ BAD < 90)
  (h_circle : ∀ P, circle P A D ↔ (P = D ∨ P = E))
  (h_intersection : line (BE) ∩ line (AC) = S) :
  cyclic A S D E :=
sorry

end concyclic_points_in_rhombus_l295_295638


namespace minimum_lambda_l295_295019

-- Definitions of the geometric entities
structure Point where
  x : ℝ
  y : ℝ

-- Parabola definition and properties
def parabola (p : Point) : Prop := p.y = 4 * p.x * p.x

-- Function to compute angle in radians
def angle_MFN (M F N : Point) : ℝ :=
  -- Placeholder for angle calculation, assume correct implementation is here
  sorry

-- Function to compute distance between two points
def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Midpoint of two points
def midpoint (M N : Point) : Point :=
  { x := (M.x + N.x) / 2, y := (M.y + N.y) / 2 }

-- Distance from midpoint to a horizontal line
def distance_to_line (P : Point) (y_line : ℝ) : ℝ :=
  Real.abs (P.y - y_line)

-- The main theorem
theorem minimum_lambda (M N F : Point) 
  (hM_on_parabola : parabola M)
  (hN_on_parabola : parabola N)
  (F_focus : F.y = 1 / 16 ∧ F.x = 0)
  (angle_condition : angle_MFN M F N = 2 * Real.pi / 3)
  (d : ℝ := distance_to_line (midpoint M N) (-1 / 16))
  (h_length : distance M N ^ 2 = λ * d ^ 2) :
  λ ≥ 3 := 
sorry

end minimum_lambda_l295_295019


namespace waiter_gratuity_l295_295521

def price_leticia : ℕ := 10
def price_scarlett : ℕ := 13
def price_percy : ℕ := 17

def total_cost := price_leticia + price_scarlett + price_percy
def tip_percentage := 0.10
def gratuity := (tip_percentage * total_cost.toReal).toNat

theorem waiter_gratuity : gratuity = 4 :=
sorry

end waiter_gratuity_l295_295521


namespace smallest_n_l295_295689

theorem smallest_n (j c g : ℕ) (n : ℕ) (total_cost : ℕ) 
  (h_condition : total_cost = 10 * j ∧ total_cost = 16 * c ∧ total_cost = 18 * g ∧ total_cost = 24 * n) 
  (h_lcm : Nat.lcm (Nat.lcm 10 16) 18 = 720) : n = 30 :=
by
  sorry

end smallest_n_l295_295689


namespace asymptote_of_hyperbola_l295_295759

theorem asymptote_of_hyperbola
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h4 : (sqrt(a^2 - b^2) / a) * (sqrt(a^2 + b^2) / a) = sqrt(3) / 2) :
  (∀ x y : ℝ, x = -sqrt(2) * y → x * x / a^2 - y * y / b^2 = 0) ∨
  (∀ x y : ℝ, x = sqrt(2) * y → x * x / a^2 - y * y / b^2 = 0) :=
sorry

end asymptote_of_hyperbola_l295_295759


namespace sum_binom_series_closed_form_l295_295730

theorem sum_binom_series_closed_form (k n : ℕ) :
  (∑ j in finset.range (k-1), binomial k (j+1) * ∑ i in finset.range (n+1), (i : ℕ)^(j+1)) = (n + 1)^k - (n + 1) :=
sorry

end sum_binom_series_closed_form_l295_295730


namespace max_value_cos2x_plus_2sinx_l295_295577

theorem max_value_cos2x_plus_2sinx :
  ∃ x : ℝ, -1 ≤ real.sin x ∧ real.sin x ≤ 1 ∧ 
  (∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 → (real.cos (2 * x) + 2 * real.sin x) ≤ (1 - 2 * t ^ 2 + 2 * t) ∧
  (real.cos (2 * x) + 2 * real.sin x) =  -2 * (real.sin x - 1 / 2) ^ 2 + 3 / 2) :=
sorry

end max_value_cos2x_plus_2sinx_l295_295577


namespace parallel_lines_necessary_and_sufficient_l295_295415

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l295_295415


namespace sum_area_DQiPi_l295_295859

def area_DQiPi (i : ℕ) : ℝ := (√2 / 6) / (2 ^ i)

theorem sum_area_DQiPi :
  ∑' (i : ℕ), area_DQiPi i = √2 / 6 :=
by
  sorry

end sum_area_DQiPi_l295_295859


namespace fraction_of_students_with_mentor_l295_295070

theorem fraction_of_students_with_mentor (s n : ℕ) (h : n / 2 = s / 3) :
  (n / 2 + s / 3 : ℚ) / (n + s : ℚ) = 2 / 5 := by
  sorry

end fraction_of_students_with_mentor_l295_295070


namespace two_trains_clear_time_l295_295224

noncomputable def time_to_clear_trains (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_length := length1 + length2
  total_length / relative_speed

theorem two_trains_clear_time : 
  time_to_clear_trains 120 280 42 36 ≈ 18.46 :=
by
  sorry

end two_trains_clear_time_l295_295224


namespace waiter_gratuity_l295_295522

def price_leticia : ℕ := 10
def price_scarlett : ℕ := 13
def price_percy : ℕ := 17

def total_cost := price_leticia + price_scarlett + price_percy
def tip_percentage := 0.10
def gratuity := (tip_percentage * total_cost.toReal).toNat

theorem waiter_gratuity : gratuity = 4 :=
sorry

end waiter_gratuity_l295_295522


namespace g_three_eighths_l295_295187

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l295_295187


namespace pens_solution_exists_l295_295627

-- Definition of the conditions
def pen_cost_eq (x y : ℕ) : Prop :=
  17 * x + 12 * y = 150

-- Proof problem statement that follows from the conditions
theorem pens_solution_exists :
  ∃ x y : ℕ, pen_cost_eq x y :=
by
  existsi (6 : ℕ)
  existsi (4 : ℕ)
  -- Normally the proof would go here, but as stated, we use sorry.
  sorry

end pens_solution_exists_l295_295627


namespace number_of_cases_lt_abs_vals_l295_295734

open Set

def A : Set Int := {-3, -2, -1, 0, 1, 2, 3}

def countPairs (s : Set Int) : Int :=
  (s.toFinset.product s.toFinset).filter (λ p, Int.natAbs p.1 < Int.natAbs p.2).card

theorem number_of_cases_lt_abs_vals : countPairs A = 18 := 
  by
  sorry

end number_of_cases_lt_abs_vals_l295_295734


namespace profit_calculation_l295_295263

variable (x y : ℝ)

-- Conditions
def fabric_constraints_1 : Prop := (0.5 * x + 0.9 * (50 - x) ≤ 38)
def fabric_constraints_2 : Prop := (x + 0.2 * (50 - x) ≤ 26)
def x_range : Prop := (17.5 ≤ x ∧ x ≤ 20)

-- Goal
def profit_expression : ℝ := 15 * x + 1500

theorem profit_calculation (h1 : fabric_constraints_1 x) (h2 : fabric_constraints_2 x) (h3 : x_range x) : y = profit_expression x :=
by
  sorry

end profit_calculation_l295_295263


namespace red_peppers_weight_correct_l295_295803

def weight_of_red_peppers : Prop :=
  ∀ (T G : ℝ), (T = 0.66) ∧ (G = 0.33) → (T - G = 0.33)

theorem red_peppers_weight_correct : weight_of_red_peppers :=
  sorry

end red_peppers_weight_correct_l295_295803


namespace sum_of_first_10_terms_is_55_b_sequence_relation_S_n_formula_l295_295374

-- Define the sequences and their relationships
def a_seq : ℕ → ℚ
def b_seq (n : ℕ) : ℚ := n * (a_seq n)
def S_n (n : ℕ) : ℚ := ∑ k in range (n + 1), a_seq k

-- Given conditions
axiom a_relation : ∀ n : ℕ, n > 0 → (a_seq (n + 1)) / (n + 1) - (a_seq n) / n = 1
axiom a_initial : a_seq 1 = 1
axiom sum_10_terms : ∑ k in range 1 11, (a_seq k) / k = 55
axiom b_relation : ∀ n : ℕ, n ≥ 2 → b_seq n - b_seq (n - 1) = 3 * a_seq n - 3 * n + 1
axiom S_n_relation : ∀ n : ℕ, S_n n = n * (n + 1) * (2 * n + 1) / 6

-- Proof goals
theorem sum_of_first_10_terms_is_55 : ∑ k in range 1 11, (a_seq k) / k = 55 := by
  exact sum_10_terms

theorem b_sequence_relation (n : ℕ) (h : n ≥ 2) : b_seq n - b_seq (n - 1) = 3 * a_seq n - 3 * n + 1 := by
  exact b_relation n h

theorem S_n_formula (n : ℕ) : S_n n = n * (n + 1) * (2 * n + 1) / 6 := by
  exact S_n_relation n

end sum_of_first_10_terms_is_55_b_sequence_relation_S_n_formula_l295_295374


namespace grassy_area_percentage_l295_295922

noncomputable def percentage_grassy_area (park_area path1_area path2_area intersection_area : ℝ) : ℝ :=
  let covered_by_paths := path1_area + path2_area - intersection_area
  let grassy_area := park_area - covered_by_paths
  (grassy_area / park_area) * 100

theorem grassy_area_percentage (park_area : ℝ) (path1_area : ℝ) (path2_area : ℝ) (intersection_area : ℝ) 
  (h1 : park_area = 4000) (h2 : path1_area = 400) (h3 : path2_area = 250) (h4 : intersection_area = 25) : 
  percentage_grassy_area park_area path1_area path2_area intersection_area = 84.375 :=
by
  rw [percentage_grassy_area, h1, h2, h3, h4]
  simp
  sorry

end grassy_area_percentage_l295_295922


namespace abby_potatoes_peeled_l295_295427

theorem abby_potatoes_peeled (total_potatoes : ℕ) (homers_rate : ℕ) (abbys_rate : ℕ) (time_alone : ℕ) (potatoes_peeled : ℕ) :
  (total_potatoes = 60) →
  (homers_rate = 4) →
  (abbys_rate = 6) →
  (time_alone = 6) →
  (potatoes_peeled = 22) :=
  sorry

end abby_potatoes_peeled_l295_295427


namespace problem1_problem2_l295_295766

-- Definitions
variable {f : ℝ → ℝ}
variable {x₁ x₂ a : ℝ}

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f(x) > f(y)

-- (1) Prove that for any \( x₁, x₂ \in [-1, 1] \), \([f(x₁) + f(x₂)] \cdot (x₁ + x₂) \leq 0\)
theorem problem1 (h_odd : odd_function f) (h_decreasing : decreasing_function f) (h_x₁ : -1 ≤ x₁ ∧ x₁ ≤ 1) (h_x₂ : -1 ≤ x₂ ∧ x₂ ≤ 1) :
  (f(x₁) + f(x₂)) * (x₁ + x₂) ≤ 0 :=
sorry

-- (2) Prove that if \( f(1 - a) + f((1 - a)^2) < 0 \), then \( 0 \leq a < 1 \)
theorem problem2 (h_odd : odd_function f) (h_decreasing : decreasing_function f) (h_cond : f(1 - a) + f((1 - a)^2) < 0) :
  0 ≤ a ∧ a < 1 :=
sorry

end problem1_problem2_l295_295766


namespace find_x_l295_295432

def F (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ := a^b + c * d

theorem find_x (x : ℕ) : F 3 x 5 9 = 500 → x = 6 := 
by 
  sorry

end find_x_l295_295432


namespace clean_car_time_l295_295887

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l295_295887


namespace range_of_setA_l295_295245

def setA := {x : ℕ | x > 15 ∧ x < 36 ∧ Nat.prime x}

theorem range_of_setA : (∀ a b : ℕ, a ∈ setA ∧ b ∈ setA → max a b - min a b = 31 - 17) :=
begin
  sorry
end

end range_of_setA_l295_295245


namespace length_CD_l295_295677

/-- Define a structure for triangle and use specific conditions -/
structure Triangle :=
  (A B C : Point)
  (AB : ℝ)
  (area : ℝ)

/-- Define constants for this particular problem -/
constant A : Point
constant B : Point
constant C : Point
constant AB_length : ℝ := 4.9
constant area_triangle : ℝ := 9.31

/-- Define the main proof problem -/
theorem length_CD 
  (T : Triangle)
  (h1 : T.A = A)
  (h2 : T.B = B)
  (h3 : T.C = C)
  (h4 : T.AB = AB_length)
  (h5 : T.area = area_triangle)
  : ∃ h : ℝ, h = 3.8 :=
  sorry

end length_CD_l295_295677


namespace matrix_power_4_l295_295313

def matrix_exp := λ (A : Matrix (Fin 2) (Fin 2) ℤ) (n : ℕ), A ^ n

theorem matrix_power_4 :
  let A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -1], ![1, 1]]
  matrix_exp A 4 = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_4_l295_295313


namespace car_speed_second_hour_l295_295201

theorem car_speed_second_hour (s1 : ℝ) (avg_speed : ℝ) : 
  s1 = 145 → avg_speed = 102.5 → (∃ s2, (s2 = 60 ∧ (s1 + s2) / 2 = avg_speed)) :=
by intros h1 h2
   use 60
   split
   . exact rfl
   . rw [h1, h2]
     norm_num

end car_speed_second_hour_l295_295201


namespace dot_product_values_probability_reviewing_history_probability_reviewing_geography_l295_295987

open Real

def P1 : ℝ × ℝ := (-1, 0)
def P2 : ℝ × ℝ := (-1, 1)
def P3 : ℝ × ℝ := (0, 1)
def P4 : ℝ × ℝ := (1, 1)
def P5 : ℝ × ℝ := (1, 0)

def vectors := [P1, P2, P3, P4, P5]

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_values : ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vectors → v2 ∈ vectors → v1 ≠ v2 → 
  dot_product v1 v2 ∈ {-1, 0, 1} := 
sorry

theorem probability_reviewing_history : 
  (card {pair : (ℝ × ℝ) × (ℝ × ℝ) | pair.1 ∈ vectors ∧ pair.2 ∈ vectors ∧ pair.1 ≠ pair.2 ∧ dot_product pair.1 pair.2 > 0}) =
  2 * (card {pair : (ℝ × ℝ) × (ℝ × ℝ) | pair.1 ∈ vectors ∧ pair.2 ∈ vectors ∧ pair.1 ≠ pair.2}) / 5 := 
sorry

theorem probability_reviewing_geography : 
  (card {pair : (ℝ × ℝ) × (ℝ × ℝ) | pair.1 ∈ vectors ∧ pair.2 ∈ vectors ∧ pair.1 ≠ pair.2 ∧ dot_product pair.1 pair.2 = 0}) =
  3 * (card {pair : (ℝ × ℝ) × (ℝ × ℝ) | pair.1 ∈ vectors ∧ pair.2 ∈ vectors ∧ pair.1 ≠ pair.2}) / 10 := 
sorry

end dot_product_values_probability_reviewing_history_probability_reviewing_geography_l295_295987


namespace limit_proof_l295_295639

theorem limit_proof (f : ℝ → ℝ) (x₀ L : ℝ)
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → |f x - L| < ε) :
  f = (λ x, (4 * x ^ 2 - 14 * x + 6) / (x - 3)) → x₀ = 3 → L = 10 → 
  by { sorry } :=
begin
  sorry
end

end limit_proof_l295_295639


namespace exponents_divisible_by_8_l295_295611

theorem exponents_divisible_by_8 (n : ℕ) : 8 ∣ (3^(4 * n + 1) + 5^(2 * n + 1)) :=
by
-- Base case and inductive step will be defined here.
sorry

end exponents_divisible_by_8_l295_295611


namespace no_line_can_intersect_all_segments_of_11_segment_polygonal_chain_l295_295685

theorem no_line_can_intersect_all_segments_of_11_segment_polygonal_chain
  (vertices : Fin 11 → ℝ × ℝ)
  (segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
  (closed_chain : segments 10.2.2 = segments 0.1 ∧ ∀ i, (segments i).2 = (segments (i + 1) % 11).1)
  (line : (ℝ × ℝ) → Prop)
  (line_no_vertex : ∀ v, v ∈ set.range vertices → ¬line v) :
  ¬ ∀ i, ∃ x, line x ∧ segments i.1 ≤ x ∧ x ≤ segments i.2 := sorry

end no_line_can_intersect_all_segments_of_11_segment_polygonal_chain_l295_295685


namespace exists_divisor_l295_295515

theorem exists_divisor (n k : ℕ) 
  (hn : n % 2 = 1) 
  (hk : 0 < k) 
  (hodd : (finset.filter (λ d, d ≤ k) (nat.divisors (2 * n))).card % 2 = 1) :
  ∃ d, d ∣ (2 * n) ∧ k < d ∧ d ≤ 2 * k := 
begin
  sorry
end

end exists_divisor_l295_295515


namespace inequality_bounds_l295_295343

theorem inequality_bounds (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  1 < (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) ∧
  (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) < 4 :=
sorry

end inequality_bounds_l295_295343


namespace total_splash_width_l295_295605

theorem total_splash_width :
  let pebbles_splash_width := 1 / 4
  let rocks_splash_width := 1 / 2
  let boulders_splash_width := 2
  let num_pebbles := 6
  let num_rocks := 3
  let num_boulders := 2
  let total_width := num_pebbles * pebbles_splash_width + num_rocks * rocks_splash_width + num_boulders * boulders_splash_width
  in total_width = 7 := by
  sorry

end total_splash_width_l295_295605


namespace find_b_l295_295199

-- Definitions based on the conditions provided.
def poly (b : ℝ) : polynomial ℝ :=
  2 * polynomial.X^3 + b * polynomial.X + 13

def factor1 (p : ℝ) : polynomial ℝ :=
  polynomial.X^2 + p * polynomial.X + 1

def factor2 : polynomial ℝ :=
  2 * polynomial.X + 13

-- Theorem stating that the polynomial has given factors and the value of b derived.
theorem find_b (b p : ℝ) (H : poly b = factor1 p * factor2) : b = 86.5 :=
  sorry

end find_b_l295_295199


namespace find_a6_l295_295481

variable {a : ℕ → ℝ}

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def given_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem find_a6 (d : ℝ) :
  is_arithmetic_sequence a d →
  given_condition a d →
  a 6 = 3 :=
by
  -- The proof would go here
  sorry

end find_a6_l295_295481


namespace rectangle_area_from_square_area_and_proportions_l295_295674

theorem rectangle_area_from_square_area_and_proportions :
  ∃ (a b w : ℕ), a = 16 ∧ b = 3 * w ∧ w = Int.natAbs (Int.sqrt a) ∧ w * b = 48 :=
by
  sorry

end rectangle_area_from_square_area_and_proportions_l295_295674


namespace weight_loss_percentage_l295_295240

theorem weight_loss_percentage 
  (W : ℝ) 
  (measured_percentage : ℝ) 
  (actual_loss_with_clothes : ℝ) 
  (percentage_addition : ℝ) 
  (measured_loss : ℝ) :
  measured_loss = 14.28 :=
by
  let x := measured_percentage
  let added_weight := percentage_addition / 100 * W
  let actual_loss := x / 100 * W
  let final_weight := W - actual_loss + added_weight
  have h1 : final_weight = 0.8772 * W := sorry
  have h2 : x = 14.28 := sorry
  exact h2

end weight_loss_percentage_l295_295240


namespace orange_shells_correct_l295_295162

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end orange_shells_correct_l295_295162


namespace f_2008_eq_11_l295_295812

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k n : ℕ) : ℕ :=
  Nat.recOn k (f n) (λ k' r, f r)

theorem f_2008_eq_11 : f_k 2008 8 = 11 :=
by
  sorry

end f_2008_eq_11_l295_295812


namespace cone_base_radius_l295_295225

/-- Given a semicircular piece of paper with a diameter of 2 cm is used to construct the 
  lateral surface of a cone, prove that the radius of the base of the cone is 0.5 cm. --/
theorem cone_base_radius (d : ℝ) (arc_length : ℝ) (circumference : ℝ) (r : ℝ)
  (h₀ : d = 2)
  (h₁ : arc_length = (1 / 2) * d * Real.pi)
  (h₂ : circumference = arc_length)
  (h₃ : r = circumference / (2 * Real.pi)) :
  r = 0.5 :=
by
  sorry

end cone_base_radius_l295_295225


namespace tiffany_blocks_l295_295896

def tiffany_time := 3 -- Tiffany ran for 3 minutes.
def moses_blocks := 12 -- Moses ran 12 blocks.
def moses_time := 8 -- Moses ran for 8 minutes.
def higher_speed := 2 -- The higher average speed is 2 blocks per minute.

theorem tiffany_blocks : 
  ∃ (blocks : ℕ), blocks = higher_speed * tiffany_time :=
by
  use higher_speed * tiffany_time
  sorry

end tiffany_blocks_l295_295896


namespace goldfish_distribution_l295_295908

theorem goldfish_distribution : 
    let tanks := [1, 2, 3]
    let goldfish_count := 7
    let condition1 := (2, 2, 3)
    let condition2 := (1, 3, 3)
    let condition3 := (1, 2, 4)
    let combinations :=
        (Nat.choose 7 2) * (Nat.choose 5 2) +
        (Nat.choose 7 1) * (Nat.choose 6 3) +
        (Nat.choose 7 1) * (Nat.choose 6 2)
    in combinations = 455 :=
by
  sorry

end goldfish_distribution_l295_295908


namespace range_of_b_l295_295789

noncomputable def f (x : ℝ) : ℝ := ln x - (1 / 4) * x + (3 / 4) / x - 1

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := x ^ 2 - 2 * b * x + 4

theorem range_of_b :
  (∀ x1, 0 < x1 ∧ x1 < 2 →
    ∃ x2, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f(x1) ≥ g(x2, b)) ↔ (b ∈ Set.Ici (17 / 8)) :=
sorry

end range_of_b_l295_295789


namespace solve_equation_solve_equation_any_x_l295_295919

noncomputable def harmonic (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, 1 / (k + 1)

theorem solve_equation (x b n : ℕ) 
  (hb : b > n) (hbn : b ≠ n + 1) : 
  (∑ k in Finset.range n, (x - (b - (k + 1))) / (k + 1)) = 
  (∑ k in Finset.range n, (x - (k + 1)) / (b - (k + 1))) ↔ x = b := 
begin
  sorry
end

theorem solve_equation_any_x (x b n : ℕ) 
  (hb : b > n) (hbn : b = n + 1) : 
  (∑ k in Finset.range n, (x - (b - (k + 1))) / (k + 1)) = 
  (∑ k in Finset.range n, (x - (k + 1)) / (b - (k + 1))) := 
begin
  sorry
end

end solve_equation_solve_equation_any_x_l295_295919


namespace prove_correct_cost_prices_and_total_price_l295_295197

def cost_price_of_furniture (C_table C_chair S_table S_chair : ℕ) (tax_percentage chair_discount markup_percentage : ℝ) : ℕ :=
  let marked_up_table := (S_table : ℝ) / (1 + markup_percentage)
  let discounted_chair := (S_chair : ℝ) / ((1 + markup_percentage) * (1 - chair_discount))
  let total_before_tax := S_table + S_chair
  let total_with_tax := ((total_before_tax : ℝ) * (1 + tax_percentage)).to_nat
  if marked_up_table.to_nat = C_table ∧ discounted_chair.to_nat = C_chair then total_with_tax else 0

theorem prove_correct_cost_prices_and_total_price :
  ∃ (C_table C_chair : ℕ), 
    let S_table := 4800
    let S_chair := 2700
    let tax_percentage := 0.07
    let chair_discount := 0.10
    let markup_percentage := 0.25
    cost_price_of_furniture 3840 2400 S_table S_chair tax_percentage chair_discount markup_percentage = 8025 :=
by
  sorry

end prove_correct_cost_prices_and_total_price_l295_295197


namespace range_of_fx_l295_295769

variable (a b x : ℝ)
def curve := λ x : ℝ, a * x^3 + b * x

variable (P : ℝ × ℝ := (2, 2))
variable (slope_at_P : ℝ := 9)

lemma curve_p_value : curve a b 2 = 2 := by sorry

lemma slope_at_p : deriv (curve a b) 2 = 9 := by sorry

theorem range_of_fx (x : ℝ) : (a = 2) → (b = -7) → 
  (∀ y, ∃ x, y = 2 * x^3 - 7 * x) ↔ (-2 ≤ x) ∧ (x ≤ 18) := by
  sorry

end range_of_fx_l295_295769


namespace ice_cream_to_afford_games_l295_295152

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l295_295152


namespace find_common_ratio_sum_first_n_terms_l295_295099

variable (a : ℕ → ℝ) (q : ℝ) 

-- Condition: {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: a₁ is the arithmetic mean of a₂ and a₃
def arithmetic_mean (a : ℕ → ℝ) :=
  a 1 = (a 2 + a 3) / 2

-- Proposition 1: Find the common ratio q
theorem find_common_ratio (h1 : is_geometric_sequence a q) (h2 : q ≠ 1) (h3 : arithmetic_mean a) : 
  q = -2 :=
by sorry

-- Proposition 2: Find the sum of the first n terms of the sequence {n * a_n}, given a₁ = 1
def sequence_n_times_a (a : ℕ → ℝ) :=
  λ n, n * a n

def sum_of_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) f

def geom_sum (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n else (1 - q ^ n) / (1 - q)

theorem sum_first_n_terms (h1 : is_geometric_sequence a q) (h2 : q = -2) (h3 : a 1 = 1) (n : ℕ) :
  sum_of_sequence (sequence_n_times_a a) n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
by sorry

end find_common_ratio_sum_first_n_terms_l295_295099


namespace log_function_range_all_real_l295_295027

def f (a x : ℝ) : ℝ := log a ((-2) * x + 1)

theorem log_function_range_all_real (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f a x > 0) ↔ true :=
sorry

end log_function_range_all_real_l295_295027


namespace expected_plain_zongzi_picked_l295_295328

-- Definitions and conditions:
def total_zongzi := 10
def red_bean_zongzi := 3
def meat_zongzi := 3
def plain_zongzi := 4

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probabilities
def P_X_0 : ℚ := (choose 6 2 : ℚ) / choose 10 2
def P_X_1 : ℚ := (choose 6 1 * choose 4 1 : ℚ) / choose 10 2
def P_X_2 : ℚ := (choose 4 2 : ℚ) / choose 10 2

-- Expected value of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

theorem expected_plain_zongzi_picked : E_X = 4 / 5 := by
  -- Using the definition of E_X and the respective probabilities
  unfold E_X P_X_0 P_X_1 P_X_2
  -- Use the given formula to calculate the values
  -- Remaining steps would show detailed calculations leading to the answer
  sorry

end expected_plain_zongzi_picked_l295_295328


namespace max_pancake_pieces_3_cuts_l295_295975

open Nat

def P : ℕ → ℕ
| 0 => 1
| n => n * (n + 1) / 2 + 1

theorem max_pancake_pieces_3_cuts : P 3 = 7 := by
  have h0: P 0 = 1 := by rfl
  have h1: P 1 = 2 := by rfl
  have h2: P 2 = 4 := by rfl
  show P 3 = 7
  calc
    P 3 = 3 * (3 + 1) / 2 + 1 := by rfl
    _ = 3 * 4 / 2 + 1 := by rfl
    _ = 6 + 1 := by norm_num
    _ = 7 := by norm_num

end max_pancake_pieces_3_cuts_l295_295975


namespace max_sum_value_correct_l295_295119

noncomputable def max_sum_value (x y z v w : ℝ) (h_cond : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) : ℝ :=
  let N := x * z + 3 * y * z + 5 * z * v + 2 * z * w in
  let x_N := 5 in
  let y_N := 15 in
  let z_N := 5 * real.sqrt 202.5 in
  let v_N := 25 in
  let w_N := 10 in
  let maxN := 3037.5 * real.sqrt 13 in
  N + x_N + y_N + z_N + v_N + w_N

theorem max_sum_value_correct (x y z v w : ℝ) 
  (h_pos : 0 < x) (h_pos : 0 < y) (h_pos : 0 < z) (h_pos : 0 < v) (h_pos : 0 < w)
  (h_cond : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) : 
  max_sum_value x y z v w h_cond = 55 + 3037.5 * real.sqrt 13 + 5 * real.sqrt 202.5 := 
sorry

end max_sum_value_correct_l295_295119


namespace original_number_is_1200_l295_295664

theorem original_number_is_1200 (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by
  sorry

end original_number_is_1200_l295_295664


namespace distance_from_point_P_to_origin_l295_295177

theorem distance_from_point_P_to_origin : 
  let P := (1, 2, 2) in
  let d := Math.sqrt (1^2 + 2^2 + 2^2) in
  d = 3 :=
by
  sorry

end distance_from_point_P_to_origin_l295_295177


namespace parking_lot_wheels_l295_295067

-- definitions for the conditions
def num_cars : ℕ := 10
def num_bikes : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- statement of the theorem
theorem parking_lot_wheels : (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 44 := by
  sorry

end parking_lot_wheels_l295_295067


namespace tan_beta_eq_neg_inv_3_l295_295390

theorem tan_beta_eq_neg_inv_3 (α β : ℝ) (h₁ : α + β = π / 4) (h₂ : tan α = 2) : tan β = -1 / 3 :=
by
  sorry

end tan_beta_eq_neg_inv_3_l295_295390


namespace fewer_vip_tickets_sold_l295_295243

-- Definitions based on the conditions
variables (V G : ℕ)
def tickets_sold := V + G = 320
def total_cost := 40 * V + 10 * G = 7500

-- The main statement to prove
theorem fewer_vip_tickets_sold :
  tickets_sold V G → total_cost V G → G - V = 34 := 
by
  intros h1 h2
  sorry

end fewer_vip_tickets_sold_l295_295243


namespace rectangle_area_is_48_l295_295672

-- Defining the square's area
def square_area : ℝ := 16

-- Defining the rectangle's width which is the same as the square's side length
def rectangle_width : ℝ := Real.sqrt square_area

-- Defining the rectangle's length which is three times its width
def rectangle_length : ℝ := 3 * rectangle_width

-- The theorem to state that the area of the rectangle is 48
theorem rectangle_area_is_48 : rectangle_width * rectangle_length = 48 :=
by
  -- Placeholder for the actual proof
  sorry

end rectangle_area_is_48_l295_295672


namespace minimize_surface_area_ratio_correct_l295_295370

noncomputable def minimize_surface_area_ratio (r h : ℝ) (V : ℝ) : ℝ := 
  if V = 3 * π * r^2 * h then h / r else 0

theorem minimize_surface_area_ratio_correct {V : ℝ} {r h : ℝ} 
    (hv : V = 3 * π * r^2 * h) 
    (A : ℝ) (ha : A = 2 * π * r^2 + 2 * π * r * h) : 
  minimize_surface_area_ratio r h V = 2 := 
by 
  -- Proof steps skipped
  sorry

end minimize_surface_area_ratio_correct_l295_295370


namespace group_size_l295_295468

-- Define the conditions
variables (N : ℕ)
variable (h1 : (1 / 5 : ℝ) * N = (N : ℝ) * 0.20)
variable (h2 : 128 ≤ N)
variable (h3 : (1 / 5 : ℝ) * N - 128 = 0.04 * (N : ℝ))

-- Prove that the number of people in the group is 800
theorem group_size : N = 800 :=
by
  sorry

end group_size_l295_295468


namespace matrix_pow_A4_l295_295311

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![1, 1]]

-- State the theorem
theorem matrix_pow_A4 :
  A^4 = ![![0, -9], ![9, -9]] :=
by
  sorry -- Proof is omitted

end matrix_pow_A4_l295_295311


namespace david_more_pushups_than_zachary_l295_295699

theorem david_more_pushups_than_zachary :
  ∀ (zachary_pushups zachary_crunches david_crunches : ℕ),
    zachary_pushups = 34 →
    zachary_crunches = 62 →
    david_crunches = 45 →
    david_crunches + 17 = zachary_crunches →
    david_crunches + 17 - zachary_pushups = 17 :=
by
  intros zachary_pushups zachary_crunches david_crunches
  intros h1 h2 h3 h4
  sorry

end david_more_pushups_than_zachary_l295_295699


namespace correct_option_is_C_l295_295982

def option_A (x : ℝ) : Prop := (-x^2)^3 = -x^5
def option_B (x : ℝ) : Prop := x^2 + x^3 = x^5
def option_C (x : ℝ) : Prop := x^3 * x^4 = x^7
def option_D (x : ℝ) : Prop := 2 * x^3 - x^3 = 1

theorem correct_option_is_C (x : ℝ) : ¬ option_A x ∧ ¬ option_B x ∧ option_C x ∧ ¬ option_D x :=
by
  sorry

end correct_option_is_C_l295_295982


namespace perimeter_expressed_l295_295861

open Real

-- Define the points
def P := (1 : ℝ, 2 : ℝ)
def Q := (3 : ℝ, 6 : ℝ)
def R := (6 : ℝ, 3 : ℝ)
def S := (8 : ℝ, 1 : ℝ)

-- Define the distance function
noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Calculate individual distances
noncomputable def PQ := dist P Q
noncomputable def QR := dist Q R
noncomputable def RS := dist R S
noncomputable def SP := dist S P

-- Calculate the perimeter
noncomputable def perimeter := PQ + QR + RS + SP

-- Prove that the perimeter can be expressed as 10√2 + √10
theorem perimeter_expressed :
  perimeter = 10 * sqrt 2 + sqrt 10 ∧ (10 + 1 = 11) := 
by 
  sorry

end perimeter_expressed_l295_295861


namespace term_in_arithmetic_sequence_l295_295420

theorem term_in_arithmetic_sequence (a1 a_m a_n d : ℤ) 
  (h₁ : a1 = 13) (h₂ : a_m = 25) (h₃ : a_n = 41) (hm : m > 1) (hnm : n > m > 1) :
  (∃ k : ℤ, 2013 = 13 + (k - 1) * d) :=
by
  use 77
  -- Additional proof steps are omitted and need to be completed.
  sorry

end term_in_arithmetic_sequence_l295_295420


namespace complement_A_in_U_l295_295796

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by
  sorry

end complement_A_in_U_l295_295796


namespace popsicle_count_l295_295482

-- Define the number of each type of popsicles
def num_grape_popsicles : Nat := 2
def num_cherry_popsicles : Nat := 13
def num_banana_popsicles : Nat := 2

-- Prove the total number of popsicles
theorem popsicle_count : num_grape_popsicles + num_cherry_popsicles + num_banana_popsicles = 17 := by
  sorry

end popsicle_count_l295_295482


namespace value_of_x_l295_295837

theorem value_of_x (x : ℝ) (h : 4 * x + 5 * x + x + 2 * x = 360) : x = 30 := 
by
  sorry

end value_of_x_l295_295837


namespace smaller_value_of_C_and_D_l295_295336

-- We define the condition set {1,2,3,4,5,6}
def is_digit_set (s : Finset ℕ) : Prop :=
  s = {1, 2, 3, 4, 5, 6}

-- We define that A + B is a multiple of 2
def multiple_of_2 (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

-- We define that C + D is a multiple of 3
def multiple_of_3 (c d : ℕ) : Prop :=
  (c + d) % 3 = 0

-- We define that E + F is a multiple of 5
def multiple_of_5 (e f : ℕ) : Prop :=
  (e + f) % 5 = 0

-- The main theorem statement which incorporates all our definitions and the condition of each digit being unique from the digit set.
theorem smaller_value_of_C_and_D (a b c d e f : ℕ) (s : Finset ℕ) :
  is_digit_set s →
  a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ e ∈ s ∧ f ∈ s →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  multiple_of_2 a b →
  multiple_of_3 c d →
  multiple_of_5 e f →
  (c < d) :=
begin
  sorry
end

end smaller_value_of_C_and_D_l295_295336


namespace metals_inductive_reasoning_l295_295588

def conducts_electricity (metal : String) : Prop :=
  metal = "Gold" ∨ metal = "Silver" ∨ metal = "Copper" ∨ metal = "Iron"

def all_metals_conduct_electricity (metals : List String) : Prop :=
  ∀ metal, metal ∈ metals → conducts_electricity metal

theorem metals_inductive_reasoning 
  (h1 : conducts_electricity "Gold")
  (h2 : conducts_electricity "Silver")
  (h3 : conducts_electricity "Copper")
  (h4 : conducts_electricity "Iron") :
  (all_metals_conduct_electricity ["Gold", "Silver", "Copper", "Iron"] → 
  all_metals_conduct_electricity ["All metals"]) :=
  sorry -- Proof skipped, as per instructions.

end metals_inductive_reasoning_l295_295588


namespace polynomial_root_product_l295_295161

theorem polynomial_root_product
  (P : Polynomial ℝ)
  (Hroots : P.roots = [ ( real.cos ( 2 * Real.pi / 7 )
                     , real.cos ( 4 * Real.pi / 7 )
                     , real.cos ( 6 * Real.pi / 7 ) ] ) :
  ∃ (a b c : ℝ), P = Polynomial.C (4) * X ^ 3
    + Polynomial.C (2) * X ^ 2
    + Polynomial.C (-2) * X
    + Polynomial.C (-1 / 2) :=
begin
  sorry
end

end polynomial_root_product_l295_295161


namespace cyclic_quadrilateral_iff_ratios_l295_295148

variables {Point : Type} [metric_space Point]

-- Definitions of points and segments
variables (A B C D I M N : Point)
variable (AC BD : ℝ)

-- Hypotheses:
-- Quadrilateral ABCD is circumscribed around a circle with center I
-- M and N are midpoints of diagonals AC and BD respectively
-- Given: IM/AC = IN/BD
def is_circumscribed (A B C D I : Point) : Prop := sorry
def is_midpoint (X Y Z : Point) : Prop := sorry

theorem cyclic_quadrilateral_iff_ratios (h1 : is_circumscribed A B C D I)
                                        (h2 : is_midpoint I A C M)
                                        (h3 : is_midpoint I B D N) :
  AC ≠ 0 →
  BD ≠ 0 →
  (dist I M / dist A C = dist I N / dist B D) ↔
  cyclic_quadrilateral A B C D :=
sorry

end cyclic_quadrilateral_iff_ratios_l295_295148


namespace b_8_result_l295_295121

def a₀ : ℚ := 4
def b₀ : ℚ := 5
def a (n : ℕ) : ℚ := Nat.recOn n a₀ (λ n a_n, (a_n^2 / b n))
def b (n : ℕ) : ℚ := Nat.recOn n b₀ (λ n b_n, (b_n^2 / a n))

theorem b_8_result :
  b 8 = 5^3281 / 4^3280 := 
sorry

end b_8_result_l295_295121


namespace each_child_receive_amount_l295_295269

def husband_weekly_contribution : ℕ := 335
def wife_weekly_contribution : ℕ := 225
def weeks_in_month : ℕ := 4
def months : ℕ := 6
def children : ℕ := 4

noncomputable def total_weekly_contribution : ℕ := husband_weekly_contribution + wife_weekly_contribution
noncomputable def total_savings : ℕ := total_weekly_contribution * (weeks_in_month * months)
noncomputable def half_savings : ℕ := total_savings / 2
noncomputable def amount_per_child : ℕ := half_savings / children

theorem each_child_receive_amount :
  amount_per_child = 1680 :=
by
  sorry

end each_child_receive_amount_l295_295269


namespace solve_problem1_solve_problem2_l295_295554

-- Problem 1
theorem solve_problem1 (x : ℚ) : (3 * x - 1) ^ 2 = 9 ↔ x = 4 / 3 ∨ x = -2 / 3 := 
by sorry

-- Problem 2
theorem solve_problem2 (x : ℚ) : x * (2 * x - 4) = (2 - x) ^ 2 ↔ x = 2 ∨ x = -2 :=
by sorry

end solve_problem1_solve_problem2_l295_295554


namespace hyperbola_eq_l295_295742

noncomputable def hyperbola_equation (a b : ℝ) := (x y : ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eq 
  (a b : ℝ)
  (hpos_a : a > 0)
  (hpos_b : b > 0)
  (asymptote : ℝ × ℝ → Prop)
  (common_focus : ℝ × ℝ → Prop)
  (heq_asymptote : ∀ xy:ℝ × ℝ, asymptote xy ↔ xy.2 = (sqrt 5 / 2) * xy.1)
  (heq_ellipse : ∀ xy:ℝ × ℝ, common_focus xy ↔ (xy.1^2 / 12) + (xy.2^2 / 3) = 1):
  hyperbola_equation 2 (sqrt 5) :=
  sorry

end hyperbola_eq_l295_295742


namespace average_death_rate_l295_295472

-- Definitions corresponding to the conditions given in the problem.
def birth_rate_per_two_seconds : ℕ := 8
def net_increase_per_day : ℕ := 86400
def seconds_in_a_day : ℕ := 86400

-- The goal is to prove the death rate every two seconds.
theorem average_death_rate :
  let net_increase_per_second := net_increase_per_day / seconds_in_a_day,
      birth_rate_per_second := birth_rate_per_two_seconds / 2,
      death_rate_per_second := birth_rate_per_second - net_increase_per_second
  in death_rate_per_second * 2 = 6 :=
by
  sorry

end average_death_rate_l295_295472


namespace min_edges_after_operations_l295_295572

theorem min_edges_after_operations (n : ℕ) (h : n ≥ 4) : 
  ∃ G : SimpleGraph (Fin n), (G.edgeSet.card = n) ∧ (∀ H : SimpleGraph (Fin n), (H.edgeSet.card < n) → (¬is_k4_free H)) :=
sorry

end min_edges_after_operations_l295_295572


namespace prime_factors_product_inequality_l295_295844

theorem prime_factors_product_inequality
  {a : ℕ → ℕ} (n : ℕ) (m : ℕ)
  (h1 : 1 < a 1)
  (h2 : ∀ i : ℕ, i < n → a i < a (i + 1))
  (h3 : ∀ i : ℕ, i < n → a i < 2 * a 1)
  (h4 : m = (a_1 * a_2 * ... * a_n).distinct_prime_factors) :
  (a_1 * a_2 * ... * a_n)^(m-1) ≥ (n!)^m := 
sorry

end prime_factors_product_inequality_l295_295844


namespace initial_capacity_of_bottle_l295_295500

theorem initial_capacity_of_bottle 
  (C : ℝ)
  (h1 : 1/3 * 3/4 * C = 1) : 
  C = 4 :=
by
  sorry

end initial_capacity_of_bottle_l295_295500


namespace compare_abc_l295_295001

noncomputable def a : ℝ := 2^(1/2)
noncomputable def b : ℝ := 3^(1/3)
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > a ∧ a > c :=
by
  sorry

end compare_abc_l295_295001


namespace alpha_half_quadrant_l295_295758

theorem alpha_half_quadrant (k : ℤ) (α : ℝ)
  (h : 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi) :
  (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < 2 * n * Real.pi) ∨
  (∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < (2 * n + 1) * Real.pi) :=
sorry

end alpha_half_quadrant_l295_295758


namespace find_p_q_r_sum_l295_295093

noncomputable def Q (p q r : ℝ) (v : ℂ) : Polynomial ℂ :=
  (Polynomial.C v + 2 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C v + 8 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C (3 * v - 5)).comp Polynomial.X

theorem find_p_q_r_sum (p q r : ℝ) (v : ℂ)
  (h_roots : ∃ v : ℂ, Polynomial.roots (Q p q r v) = {v + 2 * Complex.I, v + 8 * Complex.I, 3 * v - 5}) :
  (p + q + r) = -82 :=
by
  sorry

end find_p_q_r_sum_l295_295093


namespace correct_answer_l295_295983

-- Define propositions p and q
variable (p q : Prop)

-- Define real numbers x and y
variable (x y : ℝ)

-- Given conditions
axiom true_p : p
axiom true_not_q : ¬ q
axiom forall_x_real_2_pow_x_pos : ∀ x : ℝ, 2^x > 0
axiom if_xy_eq_0_then_x_eq_0_or_y_eq_0 : ∀ x y : ℝ, (x * y = 0) → (x = 0 ∨ y = 0)
axiom polynomial_equation_has_roots : ∃ x : ℝ, x^2 - 5*x - 6 = 0

-- Prove that the negation of ∀x ∈ ℝ, 2^x > 0 is ∃x₀ ∈ ℝ, 2^x₀ ≤ 0
theorem correct_answer : ∃ x₀ : ℝ, 2^x₀ ≤ 0 :=
sorry

end correct_answer_l295_295983


namespace sum_of_valid_numbers_l295_295231

def isSquareOfPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 4 = n

def isPerfectFifthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def validNumbersLessThan200 : List ℕ :=
  [1, 16, 81, 32]

def sumValidNumbers : ℕ :=
  130

theorem sum_of_valid_numbers :
  (∑ n in validNumbersLessThan200, id n) = sumValidNumbers :=
begin
  sorry
end

end sum_of_valid_numbers_l295_295231


namespace probability_of_all_events_at_least_pn_l295_295507

theorem probability_of_all_events_at_least_pn (α : ℝ) (n : ℕ)
  (hα : 0 < α ∧ α < 1/4)
  (pn : ℕ → ℝ)
  (hp0 : pn 0 = 1)
  (hp1 : pn 1 = 1 - α)
  (hp_rec : ∀ n, pn (n + 1) = pn n - α * pn (n - 1))
  (hp_pos : ∀ n, 0 < pn n)
  (prob_Ai : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ A_i : Prop, probability A_i ≥ 1 - α))
  (indep : ∀ i j, |i - j| > 1 → independent (A i) (A j)) :
  (∃ qn : ℕ → ℝ, qn n ≥ pn n) := 
sorry

end probability_of_all_events_at_least_pn_l295_295507


namespace find_common_ratio_sum_first_n_terms_l295_295110

-- Definitions and conditions for Part (1)
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_mean_condition (a : ℕ → ℝ) : Prop :=
  2 * a 0 = a 1 + a 2

-- Theorem for Part (1)
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) (ham : arithmetic_mean_condition a) (hq : q ≠ 1) :
  q = -2 :=
sorry

-- Definitions and conditions for Part (2)
def sum_first_n_na_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) * a i

-- Theorem for Part (2)
theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) 
  (hg : is_geometric_sequence a (-2)) (ha1 : a 0 = 1) :
  sum_first_n_na_n a n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
sorry

end find_common_ratio_sum_first_n_terms_l295_295110


namespace probability_of_six_on_fair_six_sided_die_l295_295979

def fair_die_probability (n : ℕ) (outcome : ℕ) : ℚ :=
if h : 1 ≤ outcome ∧ outcome ≤ n then (1 : ℚ) / n else 0

theorem probability_of_six_on_fair_six_sided_die : 
  fair_die_probability 6 6 = 1 / 6 :=
by {
  unfold fair_die_probability;
  split_ifs;
  { refl },
  { exfalso, cases h, exact h_1 (by norm_num) }
}

end probability_of_six_on_fair_six_sided_die_l295_295979


namespace cubing_inequality_l295_295968

theorem cubing_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by {
  -- assume the negation of the conclusion
  by_contradiction h₁,
  -- Assumption: a^3 <= b^3
  have h₂ : a^3 ≤ b^3 := le_of_not_gt h₁,
  -- derive contradiction
  sorry
}

end cubing_inequality_l295_295968


namespace balance_condition_l295_295538

theorem balance_condition (k : ℕ) 
  (a b : Fin k → ℕ)
  (h_initial : ∑ i, (a i - b i) > 0) -- Initial condition
  (h_swap : ∀ i, ∑ j, (a j - b j) - 2 * (a i - b i) ≤ 0) : 
  k ≤ 2 :=
sorry

end balance_condition_l295_295538


namespace probability_of_xj_as_sum_of_others_probability_of_even_partition_l295_295970

noncomputable def number_of_valid_combinations (S : ℕ) : ℕ :=
  sorry

theorem probability_of_xj_as_sum_of_others :
  let total_outcomes := 6 * 6 * 6 * 6 in
  ∃ (x : ℕ) (xi₁ xi₂ xi₃ xi₄ : ℕ), (xi₁ + xi₂ + xi₃ + xi₄ = x ∧ x = 2 * xi₁ ∨ x = 2 * xi₂ ∨ x = 2 * xi₃ ∨ x = 2 * xi₄)
  ∧ xi₁ ∈ {1, 2, 3, 4, 5, 6} ∧ xi₂ ∈ {1, 2, 3, 4, 5, 6} ∧ xi₃ ∈ {1, 2, 3, 4, 5, 6} ∧ xi₄ ∈ {1, 2, 3, 4, 5, 6} →
  (number_of_valid_combinations 4 + number_of_valid_combinations 6 + number_of_valid_combinations 8 + 
   number_of_valid_combinations 10 + number_of_valid_combinations 12 + number_of_valid_combinations 14 + 
   number_of_valid_combinations 16 + number_of_valid_combinations 18 + number_of_valid_combinations 20 + 
   number_of_valid_combinations 22 + number_of_valid_combinations 24) / total_outcomes =
  let successful_outcomes := sorry in
  successful_outcomes / total_outcomes :=
sorry

theorem probability_of_even_partition :
  let total_outcomes := 6 * 6 * 6 * 6 in
  ∃ (S : ℕ), (S ∈ {4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}) ∧ 
  ∃ (x₁ x₂ x₃ x₄ : ℕ), x₁ + x₂ + x₃ + x₄ = S ∧ 
  (∃ (A B : List ℕ), A ++ B = [x₁, x₂, x₃, x₄] ∧ 
  (A.Sum = B.Sum)) → 
  (number_of_valid_combinations 4 + number_of_valid_combinations 6 + number_of_valid_combinations 8 + 
  number_of_valid_combinations 10 + number_of_valid_combinations 12 + number_of_valid_combinations 14 + 
  number_of_valid_combinations 16 + number_of_valid_combinations 18 + number_of_valid_combinations 20 + 
  number_of_valid_combinations 22 + number_of_valid_combinations 24) / total_outcomes =
  let successful_outcomes := sorry in 
  successful_outcomes / total_outcomes :=
sorry

end probability_of_xj_as_sum_of_others_probability_of_even_partition_l295_295970


namespace marked_price_each_article_l295_295665

theorem marked_price_each_article (discounted_price total_discount percent_discount : ℝ) 
  (h1 : total_discount = 50)
  (h2 : percent_discount = 0.10) 
  (h3 : discounted_price = 50) :
  let marked_price := 50 / (1 - percent_discount) in
  marked_price / 2 = 27.78 := 
by 
  let marked_price := 50 / (1 - percent_discount)
  have h4 : marked_price = 27.78 * 2, from sorry,
  exact sorry

end marked_price_each_article_l295_295665


namespace intersection_complement_l295_295419

open Set

noncomputable def N := {x : ℕ | true}

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def C_N (B : Set ℕ) : Set ℕ := {n ∈ N | n ∉ B}

theorem intersection_complement :
  A ∩ (C_N B) = {1} :=
by
  sorry

end intersection_complement_l295_295419


namespace probability_C_l295_295999

theorem probability_C :
  let P_A : ℚ := 1 / 4,
      P_B : ℚ := 1 / 3,
      P_D : ℚ := 1 / 6,
      total_prob : ℚ := 1,
      P_C := total_prob - (P_A + P_B + P_D)
  in P_C = 1 / 4 :=
by
  sorry

end probability_C_l295_295999


namespace div_by_16_l295_295910

theorem div_by_16 (n : ℕ) : 
  ((2*n - 1)^3 - (2*n)^2 + 2*n + 1) % 16 = 0 :=
sorry

end div_by_16_l295_295910


namespace total_songs_isabel_bought_l295_295237

theorem total_songs_isabel_bought
  (country_albums pop_albums : ℕ)
  (songs_per_album : ℕ)
  (h1 : country_albums = 6)
  (h2 : pop_albums = 2)
  (h3 : songs_per_album = 9) : 
  (country_albums + pop_albums) * songs_per_album = 72 :=
by
  -- We provide only the statement, no proof as per the instruction
  sorry

end total_songs_isabel_bought_l295_295237


namespace surface_area_ratio_l295_295988

-- Defining the radii of spheres A and B
def radius_a : ℝ := 40
def radius_b : ℝ := 10

-- Defining the surface areas of spheres A and B
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

-- Proving the ratio of the surface area of sphere A to the surface area of sphere B is 16
theorem surface_area_ratio : surface_area radius_a / surface_area radius_b = 16 := 
by 
  -- We introduce surface_area definitions internally.
  let area_a := surface_area radius_a
  let area_b := surface_area radius_b
  -- Calculation should confirm the ratio.
  have h1 : area_a = 4 * real.pi * (radius_a)^2 := rfl
  have h2 : area_b = 4 * real.pi * (radius_b)^2 := rfl
  -- We need to calculate and simplify the division ratio properly.
  calc 
    area_a / area_b = (4 * real.pi * (radius_a ^ 2)) / (4 * real.pi * (radius_b ^ 2)) : by rw [h1, h2]
    ... = (radius_a ^ 2) / (radius_b ^ 2) : by ring
    ... = (40 ^ 2) / (10 ^ 2) : rfl
    ... = 1600 / 100 : by norm_num
    ... = 16 : by norm_num

end surface_area_ratio_l295_295988


namespace find_dads_dimes_l295_295895

variable (original_dimes mother_dimes total_dimes dad_dimes : ℕ)

def proof_problem (original_dimes mother_dimes total_dimes dad_dimes : ℕ) : Prop :=
  original_dimes = 7 ∧
  mother_dimes = 4 ∧
  total_dimes = 19 ∧
  total_dimes = original_dimes + mother_dimes + dad_dimes

theorem find_dads_dimes (h : proof_problem 7 4 19 8) : dad_dimes = 8 :=
sorry

end find_dads_dimes_l295_295895


namespace largest_number_l295_295235

theorem largest_number :
  let a := 0.993 
  let b := 0.9931 
  let c := 0.9929 
  let d := 0.939 
  let e := 0.99 
  in ∀ x, x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e → b ≥ x :=
by
  sorry

end largest_number_l295_295235


namespace triangle_side_length_l295_295451

variable (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]

theorem triangle_side_length (BC AC : ℝ) (angleC : Real.Angle) (hBC : BC = Real.sqrt 2) (hAC : AC = 1)
  (hAngleC : angleC = Real.pi / 4) :
  (∃ AB : ℝ, AB = 1) := by
  sorry

end triangle_side_length_l295_295451


namespace find_a_plus_b_l295_295450

theorem find_a_plus_b (a b : ℝ) (h₁ : ∀ x, x - b < 0 → x < b) 
  (h₂ : ∀ x, x + a > 0 → x > -a) 
  (h₃ : ∀ x, 2 < x ∧ x < 3 → -a < x ∧ x < b) : 
  a + b = 1 :=
by
  sorry

end find_a_plus_b_l295_295450


namespace no_line_can_intersect_all_segments_of_11_segment_polygonal_chain_l295_295686

theorem no_line_can_intersect_all_segments_of_11_segment_polygonal_chain
  (vertices : Fin 11 → ℝ × ℝ)
  (segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
  (closed_chain : segments 10.2.2 = segments 0.1 ∧ ∀ i, (segments i).2 = (segments (i + 1) % 11).1)
  (line : (ℝ × ℝ) → Prop)
  (line_no_vertex : ∀ v, v ∈ set.range vertices → ¬line v) :
  ¬ ∀ i, ∃ x, line x ∧ segments i.1 ≤ x ∧ x ≤ segments i.2 := sorry

end no_line_can_intersect_all_segments_of_11_segment_polygonal_chain_l295_295686


namespace find_a_l295_295594

variable {n : ℕ}
variable {a : ℝ}
variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def S_n (n : ℕ) := 3 ^ n + a
def a_2 := 6

-- Theorem statement
theorem find_a (ha2 : a_2 = 6) (hSn : S = S_n) : a = -1 := sorry

end find_a_l295_295594


namespace new_acute_angle_ACB_l295_295579

-- Define the initial condition: the measure of angle ACB is 50 degrees.
def measure_ACB_initial : ℝ := 50

-- Define the rotation: ray CA is rotated by 540 degrees clockwise.
def rotation_CW_degrees : ℝ := 540

-- Theorem statement: The positive measure of the new acute angle ACB.
theorem new_acute_angle_ACB : 
  ∃ (new_angle : ℝ), new_angle = 50 ∧ new_angle < 90 := 
by
  sorry

end new_acute_angle_ACB_l295_295579


namespace problem_a_problem_b_l295_295423

-- Definitions of vectors and conditions
def vec_a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, Real.sqrt 3)
def vec_b (θ : ℝ) : ℝ × ℝ := (1, Real.cos θ)
def θ_domain (θ : ℝ) : Prop := -Real.pi / 2 < θ ∧ θ < Real.pi / 2

-- The Lean 4 statement for the proof problem
theorem problem_a (θ : ℝ) (h₁ : θ_domain θ) (h₂ : (vec_a θ).fst * (vec_b θ).fst + (vec_a θ).snd * (vec_b θ).snd = 0) : θ = -Real.pi / 3 := 
sorry

theorem problem_b (θ : ℝ) (h₁ : θ_domain θ) : 
  ∃ θ_max, θ_domain θ_max ∧ |vec_a θ_max.1 + vec_b θ_max.2| = 3 := 
sorry

end problem_a_problem_b_l295_295423


namespace bob_rope_sections_l295_295300

/-- Given a 50-foot rope, where 1/5 is used for art, half of the remaining is given to a friend,
     and the rest is cut into 2-foot sections, prove that the number of sections Bob gets is 10. -/
theorem bob_rope_sections :
  ∀ (total_rope art_fraction remaining_fraction section_length : ℕ),
    total_rope = 50 →
    art_fraction = 5 →
    remaining_fraction = 2 →
    section_length = 2 →
    (total_rope / art_fraction / remaining_fraction / section_length) = 10 :=
by
  intros total_rope art_fraction remaining_fraction section_length
  assume h_total_rope h_art_fraction h_remaining_fraction h_section_length
  rw [h_total_rope, h_art_fraction, h_remaining_fraction, h_section_length]
  have h1 : 50 / 5 = 10 := by norm_num
  have h2 : (50 - 10) / 2 = 20 := by norm_num
  have h3 : 20 / 2 = 10 := by norm_num
  exact h3

end bob_rope_sections_l295_295300


namespace base_five_sum_l295_295973

theorem base_five_sum (a b : Nat) (ha : a = 2 * 5^2 + 1 * 5^1 + 2 * 5^0) (hb : b = 1 * 5^1 + 2 * 5^0) :
  Nat.toDigits 5 (a + b) = [2, 2, 4] :=
by
  sorry

end base_five_sum_l295_295973


namespace geometric_sequence_sum_l295_295105

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = a 1 * q) (h3 : a 3 = a 1 * q^2) 
  (h4 : 2 * a 1 = a 2 + a 3) :
  q = -2 ∧ (∑ i in Finset.range n, (i + 1) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9 := 
by
  sorry

end geometric_sequence_sum_l295_295105


namespace bears_win_probability_l295_295560

noncomputable def prob_bears_win_championship : ℚ :=
  ((Nat.choose 4 0) * (2/3)^5 * (1/3)^0) +
  ((Nat.choose 5 1) * (2/3)^5 * (1/3)^1) +
  ((Nat.choose 6 2) * (2/3)^5 * (1/3)^2) +
  ((Nat.choose 7 3) * (2/3)^5 * (1/3)^3) +
  ((Nat.choose 8 4) * (2/3)^5 * (1/3)^4)

theorem bears_win_probability : (prob_bears_win_championship * 100).round = 82 := by
  sorry

end bears_win_probability_l295_295560


namespace parabola_equation_pq_fixed_point_l295_295392

-- Conditions
def ellipse : set (ℝ × ℝ) := {p | (p.1^2 / 9) + (p.2^2 / 8) = 1}
def right_focus : ℝ × ℝ := (1, 0)
def directrix : ℝ → Prop := λ x, x = -1

-- Parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Question I: Find the equation of the parabola C
theorem parabola_equation (focus_eq : right_focus = (1, 0)) (dir_eq : directrix (-1)):
  ∃ C : set (ℝ × ℝ), ∀ p, p ∈ C ↔ parabola p :=
sorry

-- Question II: Prove line PQ passes through a fixed point if OP ⟂ OQ
theorem pq_fixed_point (focus_eq : right_focus = (1, 0)) (dir_eq : directrix (-1))
  (P Q : ℝ × ℝ) (hP : parabola P) (hQ : parabola Q) (hne : P ≠ O ∧ Q ≠ O) (ortho : (P.1 * Q.1 + P.2 * Q.2) = 0) :
  ∃ F, ∀ P Q, (hP : parabola P) → (hQ : parabola Q) → (hne : P ≠ O ∧ Q ≠ O) → (ortho : (P.1 * Q.1 + P.2 * Q.2) = 0) → (∃ line, line P Q = F) :=
sorry

end parabola_equation_pq_fixed_point_l295_295392


namespace distinct_selections_sum_to_seventeen_l295_295159

theorem distinct_selections_sum_to_seventeen : 
  (finset.univ.filter (λ (s : finset ℕ), s.sum = 17 ∧ (s : set ℕ).pairwise (≠) ∧ s ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9})).card = 5 := 
sorry

end distinct_selections_sum_to_seventeen_l295_295159


namespace choir_grouping_l295_295264

theorem choir_grouping (sopranos altos tenors basses : ℕ)
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18)
  (ratio : ℕ) :
  ratio = 1 →
  ∃ G : ℕ, G ≤ 10 ∧ G ≤ 15 ∧ G ≤ 12 ∧ 2 * G ≤ 18 ∧ G = 9 :=
by sorry

end choir_grouping_l295_295264


namespace range_of_a_l295_295575

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : ℝ := (a-1)*x^2 + (a-1)*x + 1

theorem range_of_a :
  (∀ x : ℝ, quadratic_inequality a x > 0) ↔ (1 ≤ a ∧ a < 5) :=
by
  sorry

end range_of_a_l295_295575


namespace goldfish_distribution_count_l295_295907

-- Definitions based on conditions
def num_goldfish : ℕ := 7
def num_tanks : ℕ := 3
def tank_labels : Fin num_tanks → ℕ := λ i => i + 1
def valid_distribution (distribution : Fin num_tanks → ℕ) : Prop :=
  (∀ i : Fin num_tanks, distribution i ≥ tank_labels i) ∧
  (∑ i, distribution i = num_goldfish)

-- Main theorem to prove
theorem goldfish_distribution_count : 
  ∃ (distributions : Fin num_tanks → ∑ n in finset.range num_tanks, ℕ), 
  valid_distribution distributions ∧ 
  finset.card (finset.filter valid_distribution (finset.range finset.card)) = 455 := sorry

end goldfish_distribution_count_l295_295907


namespace expected_value_of_B_is_272_l295_295532

theorem expected_value_of_B_is_272 :
  ∀ (S : finset ℕ) (hS : S.card = 100) (f : fin 10 → fin 10 → ℕ),
  (∀ x ∈ S, 1 ≤ x ∧ x ≤ 2019) →
  let A := 200 + ∑ j in finset.range 10, (f 0 j + ∑ i in finset.range 9, abs (f (i + 1) j - f i j) + f 9 j) +
                   ∑ i in finset.range 10, (f i 0 + ∑ j in finset.range 9, abs (f i (j + 1) - f i j) + f i 9) in
  A ≤ 272 :=
begin
  intros S hS f hS_vals,
  sorry
end

end expected_value_of_B_is_272_l295_295532


namespace intersection_A_B_l295_295795

-- Define the sets
def A := {1, 2, 4}
def B := {2, 4, 6}

-- Define the proof problem
theorem intersection_A_B : A ∩ B = {2, 4} :=
by
  sorry

end intersection_A_B_l295_295795


namespace ratio_of_volumes_l295_295345

variables (a b : ℝ)

noncomputable def volume_ratio : ℝ :=
  let V1 := π * (a / 2) ^ 2 * b
  let V2 := π * (b / 2) ^ 2 * a
  V1 / V2

theorem ratio_of_volumes (a b : ℝ) (hb : b ≠ 0) : volume_ratio a b = a / b :=
by
  unfold volume_ratio
  sorry

end ratio_of_volumes_l295_295345


namespace number_of_sections_l295_295299

noncomputable def initial_rope : ℕ := 50
noncomputable def rope_for_art := initial_rope / 5
noncomputable def remaining_rope_after_art := initial_rope - rope_for_art
noncomputable def rope_given_to_friend := remaining_rope_after_art / 2
noncomputable def remaining_rope := remaining_rope_after_art - rope_given_to_friend
noncomputable def section_size : ℕ := 2
noncomputable def sections := remaining_rope / section_size

theorem number_of_sections : sections = 10 :=
by
  sorry

end number_of_sections_l295_295299


namespace second_player_can_ensure_symmetry_l295_295222

def is_symmetric (seq : List ℕ) : Prop :=
  seq.reverse = seq

def swap_digits (seq : List ℕ) (i j : ℕ) : List ℕ :=
  if h : i < seq.length ∧ j < seq.length then
    seq.mapIdx (λ k x => if k = i then seq.get ⟨j, h.2⟩ 
                        else if k = j then seq.get ⟨i, h.1⟩ 
                        else x)
  else seq

theorem second_player_can_ensure_symmetry (seq : List ℕ) (h : seq.length = 1999) :
  (∃ swappable_seq : List ℕ, is_symmetric swappable_seq) :=
by
  sorry

end second_player_can_ensure_symmetry_l295_295222


namespace sum_of_positive_k_l295_295180

theorem sum_of_positive_k (k : ℕ) (α β : ℤ) (h1 : α * β = 16) (h2 : k = α + β) (h3 : x^2 - k * x + 16 = 0) : 
  let possible_ks := list.filter (λ k, ∃ α β : ℤ, α * β = 16 ∧ k = α + β ∧ α * β % 8 = 0) [17, 10, 8] in
  list.sum possible_ks = 35 :=
sorry

end sum_of_positive_k_l295_295180


namespace Jake_has_8_peaches_l295_295083

variables (Jake Steven Jill : ℕ)

-- The conditions
def condition1 : Steven = 15 := sorry
def condition2 : Steven = Jill + 14 := sorry
def condition3 : Jake = Steven - 7 := sorry

-- The proof statement
theorem Jake_has_8_peaches 
  (h1 : Steven = 15) 
  (h2 : Steven = Jill + 14) 
  (h3 : Jake = Steven - 7) : Jake = 8 :=
by
  -- The proof will go here
  sorry

end Jake_has_8_peaches_l295_295083


namespace common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295103

def is_geometric_sequence (a : ℕ → ℂ) (q : ℂ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_mean (a1 a2 a3 : ℂ) : Prop :=
  2 * a1 = a2 + a3

noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  (1 - (1 + 3 * n) * (-2 : ℤ)^n) / 9

theorem common_ratio_of_geometric_sequence (a : ℕ → ℂ) (q : ℂ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : q ≠ 1) 
  (h3 : arithmetic_mean (a 1) (a 2) (a 3)) : 
  q = -2 := 
sorry

theorem sum_of_first_n_terms_of_sequence (n : ℕ) 
  (a : ℕ → ℂ) 
  (h1 : is_geometric_sequence a (-2)) 
  (h2 : a 1 = 1) : 
  ∑ k in finset.range n, k * a k = sum_first_n_terms n := 
sorry

end common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295103


namespace min_value_t_l295_295841

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 1 / 4 else a_seq (n - 1) / 4

theorem min_value_t {t : ℝ} (h : ∀ n : ℕ, ∑ i in range n, a_seq i < t) : t = 1 / 3 :=
  sorry

end min_value_t_l295_295841


namespace orange_shells_correct_l295_295163

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end orange_shells_correct_l295_295163


namespace solve_ratio_l295_295210

noncomputable theory

open Real

def hyperbola (x : ℝ) : ℝ := 1 / x

def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

variables {A B K L M N : ℝ × ℝ} {k : ℝ}

-- Points A and B
axiom A_def : A = (0, 14)
axiom B_def : B = (0, 4)

-- Parallel lines passing through A and B respectively
noncomputable def line_A : ℝ → ℝ := line k 14
noncomputable def line_B : ℝ → ℝ := line k 4

-- Intersections with the hyperbola
axiom K_def : K.2 = hyperbola K.1 ∧ K.2 = line_A K.1
axiom L_def : L.2 = hyperbola L.1 ∧ L.2 = line_A L.1
axiom M_def : M.2 = hyperbola M.1 ∧ M.2 = line_B M.1
axiom N_def : N.2 = hyperbola N.1 ∧ N.2 = line_B N.1

-- The lengths AL, AK, BN, BM
noncomputable def AL : ℝ := (A.1 - L.1).abs
noncomputable def AK : ℝ := (A.1 - K.1).abs
noncomputable def BN : ℝ := (B.1 - N.1).abs
noncomputable def BM : ℝ := (B.1 - M.1).abs

-- The required ratio
noncomputable def ratio : ℝ := (AL - AK) / (BN - BM)

theorem solve_ratio : ratio = 3.5 :=
sorry

end solve_ratio_l295_295210


namespace find_c_l295_295053

theorem find_c (a b c : ℝ) (h : (λ x : ℝ, (x + 3) * (x + b)) = (λ x : ℝ, x^2 + c * x + 15)) : c = 8 :=
by
  sorry

end find_c_l295_295053


namespace max_apples_guaranteed_l295_295132

theorem max_apples_guaranteed (a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) (h : list.pairwise (>) [a_1, a_2, a_3, a_4, a_5, a_6]) :
  ∃ (x_1 x_2 x_3 x_4 : ℤ), (∀ (s : ℤ), s ∈ [x_1 + x_2, x_1 + x_3, x_1 + x_4, x_2 + x_3, x_2 + x_4, x_3 + x_4] →
                          s = a_1 ∨ s = a_2 ∨ s = a_3 ∨ s = a_4 ∨
                          s ≥ a_5 ∨ s ≥ a_6) ∧ 14 = 
  (∑ s in [x_1 + x_2, x_1 + x_3, x_1 + x_4, x_2 + x_3, x_2 + x_4, x_3 + x_4], 
    if s = a_1 then 3 else if s = a_2 then 3 else if s = a_3 then 3 else if s = a_4 then 3 else if s ≥ a_5 ∨ s ≥ a_6 then 1 else 0) :=
by sorry

end max_apples_guaranteed_l295_295132


namespace max_min_values_l295_295193

def func (x : ℝ) : ℝ := x^3 - 2 * x^2 + 1

theorem max_min_values :
  (∀ x ∈ set.interval (-1 : ℝ) 2, func x ≤ 1) ∧
  (∀ x ∈ set.interval (-1 : ℝ) 2, func x ≥ -2) ∧
  (∃ x ∈ set.interval (-1 : ℝ) 2, func x = 1) ∧
  (∃ x ∈ set.interval (-1 : ℝ) 2, func x = -2) :=
by
  sorry

end max_min_values_l295_295193


namespace solve_system_of_equations_l295_295555

theorem solve_system_of_equations :
  ∃ x y : ℚ, (4 * x - 3 * y = -7) ∧ (5 * x + 4 * y = -6) ∧ (x = -46 / 31) ∧ (y = 11 / 31) :=
begin
  sorry
end

end solve_system_of_equations_l295_295555


namespace boat_journey_time_l295_295473

noncomputable def river_speed : ℝ := 4.7 -- Speed of the river in km/hr
noncomputable def boat_speed_still_water : ℝ := 9.85 -- Speed of boat in still water in km/hr
noncomputable def distance_upstream : ℝ := 86.3 -- Distance traveled upstream in km
noncomputable def distance_downstream : ℝ := 86.3 -- Distance traveled downstream in km

noncomputable def upstream_speed : ℝ := boat_speed_still_water - river_speed -- Effective speed going upstream
noncomputable def downstream_speed : ℝ := boat_speed_still_water + river_speed -- Effective speed going downstream

noncomputable def time_upstream : ℝ := distance_upstream / upstream_speed -- Time taken to travel upstream
noncomputable def time_downstream : ℝ := distance_downstream / downstream_speed -- Time taken to travel downstream

noncomputable def total_journey_time : ℝ := time_upstream + time_downstream -- Total journey time

theorem boat_journey_time :
  total_journey_time ≈ 22.6913 :=
by
  sorry

end boat_journey_time_l295_295473


namespace simple_interest_rate_l295_295497

theorem simple_interest_rate :
  ∀ (P T F : ℝ), P = 1000 → T = 3 → F = 1300 → (F - P) = P * 0.1 * T :=
by
  intros P T F hP hT hF
  sorry

end simple_interest_rate_l295_295497


namespace shooter_probability_l295_295493

def is_hit (digit : Nat) : Bool :=
  digit ≠ 0 ∧ digit ≠ 1

def count_hits (group : List Nat) : Nat :=
  group.countp is_hit

def favorable_group (group : List Nat) : Bool :=
  count_hits group ≥ 3

def estimate_probability (groups : List (List Nat)) : Real :=
  (groups.countp favorable_group) / groups.length.toReal

theorem shooter_probability : estimate_probability [
  [5, 7, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7], [0, 3, 4, 7],
  [4, 3, 7, 3], [8, 6, 3, 6], [9, 6, 4, 7], [1, 4, 1, 7], [4, 6, 9, 8],
  [0, 3, 7, 1], [6, 2, 3, 3], [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1],
  [3, 6, 6, 1], [9, 5, 9, 7], [7, 4, 2, 4], [6, 7, 1, 0], [4, 2, 8, 1]
] = 0.75 := by
  sorry

end shooter_probability_l295_295493


namespace cyclic_quadrilateral_iff_angle_equality_l295_295015

open EuclideanGeometry

variables {A B C D E M P Q : Point}

/-- Given a triangle ABC, with M being the midpoint of AB, P a point inside the triangle, 
   and Q the reflection of P with respect to M. Let D and E be the intersections of line AP and BP
   with sides BC and AC, respectively.
   
   This theorem states that the quadrilateral A, B, D, and E lie on a circle if and only if ∠ACP = ∠QCB. -/
theorem cyclic_quadrilateral_iff_angle_equality 
  (hM : is_midpoint M A B)
  (hP_inside : is_inside P (triangle A B C))
  (hQ_reflection : is_reflection Q P M)
  (hD_intersection : intersects (line A P) (line B C) D)
  (hE_intersection : intersects (line B P) (line A C) E) :
  (is_cyclic_quad A B D E) ↔ (angle A C P = angle Q C B) :=
sorry

end cyclic_quadrilateral_iff_angle_equality_l295_295015


namespace q_can_complete_work_in_30_days_l295_295244

theorem q_can_complete_work_in_30_days (W_p W_q W_r : ℝ)
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = 1/10)
  (h3 : W_r = 1/30) :
  1 / W_q = 30 :=
by
  -- Note: You can add proof here, but it's not required in the task.
  sorry

end q_can_complete_work_in_30_days_l295_295244


namespace length_of_adult_bed_is_20_decimeters_l295_295576

-- Define the length of an adult bed as per question context
def length_of_adult_bed := 20

-- Prove that the length of an adult bed in decimeters equals 20
theorem length_of_adult_bed_is_20_decimeters : length_of_adult_bed = 20 :=
by
  -- Proof goes here
  sorry

end length_of_adult_bed_is_20_decimeters_l295_295576


namespace parabola_focus_and_distance_l295_295585

theorem parabola_focus_and_distance :
  (∃ p : ℝ, y^2 = 8 * x → y^2 = 4 * p * x ∧ (2, 0) is the focus of y^2 = 8 * x)
  ∧
  (∃ m : ℝ, y^2 = 8 * x → P (sqrt 3, m) lies on y^2 = 8 * x → 
              distance PF = sqrt 3 + 2) :=
by {
  sorry
}

end parabola_focus_and_distance_l295_295585


namespace athletes_race_l295_295358

-- Denote four athletes
inductive Place
| first
| second
| third
| last
deriving Repr, DecidableEq

open Place

-- Athlete statements
def athleteA_statement (p : Place) : Prop :=
  p ≠ first ∧ p ≠ last

def athleteB_statement (p : Place) : Prop :=
  p ≠ first

def athleteC_statement (p : Place) : Prop :=
  p = first

def athleteD_statement (p : Place) : Prop :=
  p = last

-- Only three statements are true
def exactly_three_true (statements : List Prop) : Prop :=
  (statements.count (λ s => s)) = 3

-- Main theorem
theorem athletes_race :
  ∃ (pA pB pC pD : Place),
  exactly_three_true [
    athleteA_statement pA,
    athleteB_statement pB,
    athleteC_statement pC,
    athleteD_statement pD
  ] ∧
  (athleteD_statement pD = false) ∧
  (athleteC_statement pC = true) :=
by
  sorry

end athletes_race_l295_295358


namespace dot_product_ab_bc_l295_295023

-- Define the conditions
def point_P := (-2, 5)
def circle_eq (x y F : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + F = 0
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 8 = 0

-- Define the theorem to be proven
theorem dot_product_ab_bc (
  F : ℝ, 
  P_on_circle : circle_eq (-2) 5 F,
  intersects_A_B (x y : ℝ) : line_eq x y ∧ circle_eq x y F
) : (let C := (1, 1); let r := 5 -- center and radius of the circle
          let dist_center_to_line := (abs (3 * 1 + 4 * 1 + 8)) / sqrt (3^2 + 4^2)
          let length_AB := 2 * sqrt (r^2 - dist_center_to_line^2)
          let AB := (x - 1, y - 1) 
          let BC := (x - 1, y - 1)
        in AB.1 * BC.1 + AB.2 * BC.2) = -32 := sorry

end dot_product_ab_bc_l295_295023


namespace probability_of_at_least_one_red_ball_l295_295827

theorem probability_of_at_least_one_red_ball :
  let total_ways := Nat.choose 5 2 in
  let white_ways := Nat.choose 3 2 in
  total_ways = 10 ->
  white_ways = 3 ->
  (1 - (white_ways / total_ways : ℚ)) = 7 / 10 := 
by
  intros total_ways white_ways h1 h2
  sorry

end probability_of_at_least_one_red_ball_l295_295827


namespace circular_field_area_l295_295076

theorem circular_field_area (a d : ℝ) (h1 : a = 30) (h2 : d = 16) : 
  let r := d / 2 in
  let S := (1 / 2) * a * r in
  S = 120 := by
  sorry

end circular_field_area_l295_295076


namespace cows_increased_by_24_l295_295663

theorem cows_increased_by_24 (initial_cows last_year_died last_year_sold bought_this_year gifted_this_year total_cows_now : ℕ) :
  initial_cows = 39 →
  last_year_died = 25 →
  last_year_sold = 6 →
  bought_this_year = 43 →
  gifted_this_year = 8 →
  total_cows_now = 83 →
  let remaining_cows_end_of_last_year := initial_cows - last_year_died - last_year_sold in
  let remaining_cows_this_year := remaining_cows_end_of_last_year + (bought_this_year + gifted_this_year) in
  total_cows_now = remaining_cows_this_year + (remaining_cows_end_of_last_year - 8) →
  (remaining_cows_this_year - 8) = 24 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  let remaining_cows_end_of_last_year := 39 - 25 - 6,
  let remaining_cows_this_year := remaining_cows_end_of_last_year + 43 + 8,
  have h7 : 83 = remaining_cows_this_year + (remaining_cows_end_of_last_year - 8) := by rw [h6],
  have H: remaining_cows_this_year - 8 = 24 := sorry,
  exact H,
}

end cows_increased_by_24_l295_295663


namespace intersection_A_B_eq_set_l295_295042

open Set

theorem intersection_A_B_eq_set :
  let U := ℝ
  let A := { x : ℝ | 2^x > 1 }
  let B := { x : ℝ | x^2 - 3 * x - 4 > 0 }
  A ∩ B = { x : ℝ | x > 4 } := 
by
  have hA : A = { x : ℝ | x > 0 }
  have hB : B = { x : ℝ | x < -1 ∨ x > 4 }
  sorry

end intersection_A_B_eq_set_l295_295042


namespace total_volume_of_drink_l295_295629

theorem total_volume_of_drink :
  ∀ (total_ounces : ℝ),
    (∀ orange_juice watermelon_juice grape_juice : ℝ,
      orange_juice = 0.25 * total_ounces →
      watermelon_juice = 0.4 * total_ounces →
      grape_juice = 0.35 * total_ounces →
      grape_juice = 105 →
      total_ounces = 300) :=
by
  intros total_ounces orange_juice watermelon_juice grape_juice ho hw hg hg_eq
  sorry

end total_volume_of_drink_l295_295629


namespace positive_integer_solution_exists_l295_295717

theorem positive_integer_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h_eq : x^2 = y^2 + 7 * y + 6) : (x, y) = (6, 3) := 
sorry

end positive_integer_solution_exists_l295_295717


namespace minimum_value_inequality_l295_295517

theorem minimum_value_inequality (x : Fin 100 → ℝ) (h_positive : ∀ i, 0 < x i) (h_sum_cubes : ∑ i, (x i)^3 = 1) :
  ( ∑ i, x i / (1 - (x i)^2) ) ≥ (3 * Real.sqrt 3) / (200 ^ (1 / 3)) :=
sorry

end minimum_value_inequality_l295_295517


namespace f_of_2023_l295_295092

def B := { x : ℚ // x ≠ -1 ∧ x ≠ 1 }

def h (x : B) : B :=
⟨1 / (1 - x.1), 
  by {
    have h1 : (1 : ℚ) ≠ 0 := by norm_num,
    have h2 : (1 - x.1) ≠ 0,
    { by_contra h3,
      simp at h3,
      cases x.2; exact h3 },
    split;
    { intro h3; 
      have := congr_arg (λ y, (y * (1 - x.1))) h3;
      simp [h1, h2] at this;
      exact x.2 h3 }
  }⟩

noncomputable def f (x : B) : ℝ :=
(1 / 2 : ℝ) * (Real.log |x.1 + 1| - Real.log |(h x).1 + 1| + Real.log |(h (h x)).1 + 1|)

theorem f_of_2023 :
  f ⟨2023, by norm_num⟩ = Real.log ((2024 * 2023) / (2021 * 2022)).sqrt :=
by sorry

end f_of_2023_l295_295092


namespace rectangle_R2_area_l295_295756

theorem rectangle_R2_area
  (side1_R1 : ℝ) (area_R1 : ℝ) (diag_R2 : ℝ)
  (h_side1_R1 : side1_R1 = 4)
  (h_area_R1 : area_R1 = 32)
  (h_diag_R2 : diag_R2 = 20) :
  ∃ (area_R2 : ℝ), area_R2 = 160 :=
by
  sorry

end rectangle_R2_area_l295_295756


namespace solve_for_x_l295_295440

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end solve_for_x_l295_295440


namespace smallest_number_of_pencils_l295_295621

theorem smallest_number_of_pencils
  (P : ℕ)
  (h5 : P % 5 = 2)
  (h9 : P % 9 = 2)
  (h11 : P % 11 = 2)
  (hP_gt2 : P > 2) :
  P = 497 :=
by
  sorry

end smallest_number_of_pencils_l295_295621


namespace g_of_3_over_8_l295_295190

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l295_295190


namespace lunks_needed_for_apples_l295_295807

theorem lunks_needed_for_apples :
  (∀ l k a : ℕ, (4 * k = 2 * l) ∧ (3 * a = 5 * k ) → ∃ l', l' = (24 * l / 4)) :=
by
  intros l k a h
  obtain ⟨h1, h2⟩ := h
  have k_for_apples := 3 * a / 5
  have l_for_kunks := 4 * k / 2
  sorry

end lunks_needed_for_apples_l295_295807


namespace find_x_for_f_eq_inv_f_l295_295700

theorem find_x_for_f_eq_inv_f :
  ∀ x : ℝ, (x = 5/3) ↔ (f x = 4 * x - 5 ∧ f⁻¹ x = (x + 5) / 4 ∧ f x == f⁻¹ x) :=
by sorry

end find_x_for_f_eq_inv_f_l295_295700


namespace expenditure_on_house_rent_l295_295682

theorem expenditure_on_house_rent
  (income petrol house_rent remaining_income : ℝ)
  (h1 : petrol = 0.30 * income)
  (h2 : petrol = 300)
  (h3 : remaining_income = income - petrol)
  (h4 : house_rent = 0.30 * remaining_income) :
  house_rent = 210 :=
by
  sorry

end expenditure_on_house_rent_l295_295682


namespace perfect_square_identification_l295_295981

theorem perfect_square_identification :
  let A := (14! * 15!) / 2
  let B := (15! * 16!) / 2
  let C := (16! * 17!) / 2
  let D := (17! * 18!) / 2
  let E := (18! * 19!) / 2
  D ∈ {n : ℤ | ∃ k : ℤ, n = k^2} ∧ ¬ (A ∈ {n : ℤ | ∃ k : ℤ, n = k^2}) ∧
  ¬ (B ∈ {n : ℤ | ∃ k : ℤ, n = k^2}) ∧ ¬ (C ∈ {n : ℤ | ∃ k : ℤ, n = k^2}) ∧
  ¬ (E ∈ {n : ℤ | ∃ k : ℤ, n = k^2}) :=
by
  sorry

end perfect_square_identification_l295_295981


namespace smallest_six_divisible_over_2000_is_2016_l295_295355

-- Define the concept of a "six-divisible number":
def is_six_divisible (N : ℕ) : Prop :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9}.to_finset.filter (λ d, N % d = 0).card ≥ 6

-- Statement to prove:
theorem smallest_six_divisible_over_2000_is_2016 :
  ∃ (N : ℕ), 2000 < N ∧ is_six_divisible N ∧ ∀ (M : ℕ), 2000 < M ∧ is_six_divisible M → N ≤ M :=
begin
  use 2016,
  split,
  { exact dec_trivial }, -- Proof that 2000 < 2016
  split,
  { sorry }, -- Proof that 2016 is six-divisible
  { intros M hM,
    sorry } -- Proof that 2016 is the smallest such number
end

end smallest_six_divisible_over_2000_is_2016_l295_295355


namespace g_of_five_eq_one_l295_295928

variable (g : ℝ → ℝ)

theorem g_of_five_eq_one (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
    (h2 : ∀ x : ℝ, g x ≠ 0) : g 5 = 1 :=
sorry

end g_of_five_eq_one_l295_295928


namespace dot_product_parallel_l295_295424

theorem dot_product_parallel (m : ℝ) (h : (m / 1) = (3 / 2)) :
  let a := (m, 3)
  let b := (1, 2)
  a.1 * b.1 + a.2 * b.2 = 7.5 :=
by
  have h1 : m = 1.5, from sorry
  sorry

end dot_product_parallel_l295_295424


namespace inequality_solution_set_l295_295200

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := 
by 
  sorry

end inequality_solution_set_l295_295200


namespace circle_tangent_radius_l295_295609

theorem circle_tangent_radius :
  ∀ (r1 r2 : ℝ) (x : ℝ), r1 = 2 → r2 = 3 → 
  (let d := r1 + r2 in let eq1 := x^2 + (3 + x - (3 - x))^2 = d^2 in
  let eq2 := x^2 + (2 + x - (2 - x))^2 = d^2 in 
  eq1 ∧ eq2 → x = 1/9) := 
by
  intros r1 r2 x hr1 hr2 h,
  sorry

end circle_tangent_radius_l295_295609


namespace vector_orthogonal_if_straight_line_l295_295115

variables {n : Type*} [normed_group n] [normed_space ℝ n]

def is_straight_line (f : ℝ → ℝ) : Prop := 
  ∃ (m c : ℝ), ∀ x, f x = m * x + c

noncomputable def f (x : ℝ) (a b : n) : ℝ :=
  (x • a + b) ⬝ (a - x • b)

theorem vector_orthogonal_if_straight_line {a b : n} (ha : a ≠ 0) (hb : b ≠ 0) :
  is_straight_line (λ x, f x a b) → (a ⬝ b = 0) :=
sorry

end vector_orthogonal_if_straight_line_l295_295115


namespace parallel_lines_l295_295413

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l295_295413


namespace family_functions_correct_l295_295445

theorem family_functions_correct :
  (∀ f : ℝ → ℝ, f = (λ x, |x - 3|) → ∃ d₁ d₂ : set ℝ, d₁ ≠ d₂ ∧ (∀ x ∈ d₁, ∀ y ∈ d₂, f x = f y))
  ∧ ¬ (∃ f : ℝ → ℝ, (f = (λ x, x)) ∨ (f = (λ x, 2^x)) ∨ (f = (λ x, log 0.5 x)) →
  ∃ d₁ d₂ : set ℝ, d₁ ≠ d₂ ∧ (∀ x ∈ d₁, ∀ y ∈ d₂, f x = f y)) :=
by
  split
  sorry

end family_functions_correct_l295_295445


namespace find_f_neg3_l295_295385

variable (f : ℝ → ℝ)

-- Conditions
def odd_function : Prop := ∀ x : ℝ, f(-x) = -f(x)
def function_definition (h : ℝ) : Prop := h > 0 → f(h) = h^2 - 2*h + 4

-- Question and answer
theorem find_f_neg3 (H1 : odd_function f) (H2 : function_definition f) : f (-3) = -7 := sorry

end find_f_neg3_l295_295385


namespace min_equal_area_triangles_l295_295141

theorem min_equal_area_triangles (chessboard_area missing_area : ℕ) (total_area : ℕ := chessboard_area - missing_area) 
(H1 : chessboard_area = 64) (H2 : missing_area = 1) : 
∃ n : ℕ, n = 18 ∧ (total_area = 63) → total_area / ((7:ℕ)/2) = n := 
sorry

end min_equal_area_triangles_l295_295141


namespace oak_grove_total_books_l295_295207

theorem oak_grove_total_books (public_library_books : ℕ) (school_library_books : ℕ)
  (h1 : public_library_books = 1986) (h2 : school_library_books = 5106) :
  public_library_books + school_library_books = 7092 := by
  sorry

end oak_grove_total_books_l295_295207


namespace max_ab_value_l295_295394

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end max_ab_value_l295_295394


namespace total_amount_correct_l295_295850

noncomputable def total_amount (p_a r_a t_a p_b r_b t_b p_c r_c t_c : ℚ) : ℚ :=
  let final_price (p r t : ℚ) := p - (p * r / 100) + ((p - (p * r / 100)) * t / 100)
  final_price p_a r_a t_a + final_price p_b r_b t_b + final_price p_c r_c t_c

theorem total_amount_correct :
  total_amount 2500 6 10 3150 8 12 1000 5 7 = 6847.26 :=
by
  sorry

end total_amount_correct_l295_295850


namespace sum_of_areas_of_disks_l295_295329

theorem sum_of_areas_of_disks : 
  ∀ (r : ℝ) (A₁ : ℝ) (A_total : ℝ),
    let r := Real.tan (Real.pi / 18) in
    let A₁ := Real.pi * r^2 in
    let A_total := 18 * A₁ in
    circle_radius = 1 →
    A_total = 18 * Real.pi * (Real.tan (Real.pi / 18))^2 := 
by
  intros r A₁ A_total circle_radius hc
  sorry

end sum_of_areas_of_disks_l295_295329


namespace range_of_a_l295_295557

-- Definitions related to the conditions in the problem
def polynomial (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x ^ 5 - 4 * a * x ^ 3 + 2 * b ^ 2 * x ^ 2 + 1

def v_2 (x : ℝ) (a : ℝ) : ℝ := (3 * x + 0) * x - 4 * a

def v_3 (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (((3 * x + 0) * x - 4 * a) * x + 2 * b ^ 2)

-- The main statement to prove
theorem range_of_a (x a b : ℝ) (h1 : x = 2) (h2 : ∀ b : ℝ, (v_2 x a) < (v_3 x a b)) : a < 3 :=
by
  sorry

end range_of_a_l295_295557


namespace problem_correct_option_l295_295422

variables {a b : ℕ → ℝ}

-- Conditions
def sequence_condition_1 (n : ℕ) : Prop :=
  a n + b n = 700

def sequence_condition_2 (n : ℕ) : Prop :=
  a (n + 1) = (7 / 10) * a n + (2 / 5) * b n

def initial_condition : Prop :=
  a 6 = 400

-- Theorem to prove
theorem problem_correct_option (h1 : ∀ n, sequence_condition_1 n)
                               (h2 : ∀ n, sequence_condition_2 n)
                               (h3 : initial_condition) : a 3 > b 3 :=
sorry

end problem_correct_option_l295_295422


namespace equal_sides_pentagon_angle_l295_295464

theorem equal_sides_pentagon_angle {P Q R S T : Type*} 
(angle_PRT angle_QRS : ℝ) 
(equal_sides : ∀ (A B : Type*), A ≠ B → A = Q → B = S → (P ≠ Q ∧ R = S)) 
(convex : ∀ (P Q R S T : Type*), true) 
(h1 : angle_PRT = angle_QRS / 2) 
(h2 : equal_sides P Q R S T) : 
angle_PRT = 30 :=
by
  sorry

end equal_sides_pentagon_angle_l295_295464


namespace curve_and_line_l295_295248

theorem curve_and_line (a : ℝ) (h : a > 0) :
  (∀ P M N : ℝ × ℝ, -- Given points P, M, N 
     let curve := ∀ x y : ℝ, y^2 = 2 * a * x,
         line := ∀ t : ℝ, 
           (x = -2 + (sqrt 2) / 2 * t) ∧ (y = -4 + (sqrt 2) / 2 * t)
     in ((M, N).1 - P.1) * ((M, N).2 - P.2) = ((M, N).2 - P.2) * ((M, N).2 - P.2)
     → a = 1) :=
sorry

end curve_and_line_l295_295248


namespace charts_per_associate_professor_l295_295292

-- Define the number of people
def num_people : ℕ := 5

-- Each associate professor brings 2 pencils and let's denote num_charts each brings as C
-- Each assistant professor brings 1 pencil and 2 charts

-- Define the total number of pencils and charts
def total_pencils : ℕ := 10
def total_charts : ℕ := 5

-- Variables for number of associate and assistant professors
variables (A B C : ℕ)

-- Condition equations
def eq1 : A + B = num_people
def eq2 : 2 * A + B = total_pencils
def eq3 : A * C = total_charts

-- Theorem statement we need to prove
theorem charts_per_associate_professor : eq1 ∧ eq2 ∧ eq3 → C = 1 := by
  sorry

end charts_per_associate_professor_l295_295292


namespace sum_of_first_n_terms_l295_295012

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 2^(n-1)

noncomputable def c (n : ℕ) : ℕ := a n * b n

noncomputable def T (n : ℕ) : ℕ := (2 * n - 3) * 2^n + 3

theorem sum_of_first_n_terms (n : ℕ) : 
  (∑ i in Finset.range n, c i.succ) = T n := by
  sorry

end sum_of_first_n_terms_l295_295012


namespace proof_of_problem_l295_295489

noncomputable def problem_statement (a b c A B C : ℝ) : Prop :=
  a / b = 4 / 5 ∧ a / c = 4 / 6 ∧
  (A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π) ∧
  a = 4 * (b / 5) ∧ b = 5 * (b / 5) ∧ c = 6 * (b / 5) → 
  (sin A + sin C = 2 * sin B) ∧ (cos C = 1 / 8) ∧ (3 * sin A = 8 * sin (2 * C))

-- Theorem stating the problem
theorem proof_of_problem : ∀ (a b c A B C : ℝ),
  problem_statement a b c A B C :=
begin
  sorry
end

end proof_of_problem_l295_295489


namespace angle_rotation_acute_l295_295581

theorem angle_rotation_acute (angle_ACB : ℝ) (h : angle_ACB = 50) : 
  let new_angle := (angle_ACB + 540) % 360 - 180 in 
  if new_angle < 0 then new_angle + 360 else new_angle = 50 :=
by
  -- Proof goes here
  sorry

end angle_rotation_acute_l295_295581


namespace probability_cube_product_l295_295714
open Finset Nat

-- Definitions from conditions
def tiles : Finset ℕ := (finRange 15).map (Fin.val ∘ Fin.cast 15)
def die_faces : Finset ℕ := (finRange 8).map (Fin.val ∘ Fin.cast 8)

-- Proof statement
theorem probability_cube_product :
  let outcomes := tiles.product die_faces
  let cubes := filter (λ (p : ℕ × ℕ), isCube (p.1 * p.2)) outcomes
  (cubes.card : ℚ) / (outcomes.card : ℚ) = 1 / 15 :=
by
  sorry

def isCube (n : ℕ) : Prop := ∃ m, m ^ 3 = n

-- Sample checking for cubes
lemma example_cube_64 : isCube 64 := by use 4; norm_num
lemma example_cube_27 : isCube 27 := by use 3; norm_num
lemma example_cube_8 : isCube 8 := by use 2; norm_num
lemma example_cube_1 : isCube 1 := by use 1; norm_num


end probability_cube_product_l295_295714


namespace common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295101

def is_geometric_sequence (a : ℕ → ℂ) (q : ℂ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_mean (a1 a2 a3 : ℂ) : Prop :=
  2 * a1 = a2 + a3

noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  (1 - (1 + 3 * n) * (-2 : ℤ)^n) / 9

theorem common_ratio_of_geometric_sequence (a : ℕ → ℂ) (q : ℂ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : q ≠ 1) 
  (h3 : arithmetic_mean (a 1) (a 2) (a 3)) : 
  q = -2 := 
sorry

theorem sum_of_first_n_terms_of_sequence (n : ℕ) 
  (a : ℕ → ℂ) 
  (h1 : is_geometric_sequence a (-2)) 
  (h2 : a 1 = 1) : 
  ∑ k in finset.range n, k * a k = sum_first_n_terms n := 
sorry

end common_ratio_of_geometric_sequence_sum_of_first_n_terms_of_sequence_l295_295101


namespace rectangle_width_l295_295246

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 300)
  (h2 : 2 * L + 2 * W = 70) : 
  W = 15 :=
by 
  -- We prove the width W of the rectangle is 15 meters.
  sorry

end rectangle_width_l295_295246


namespace probability_below_curve_l295_295793

noncomputable def region : set (real × real) :=
  { p | 0 ≤ p.1 ∧ p.1 ≤ real.pi ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

noncomputable def curve (x : real) : real :=
  real.cos x ^ 2

theorem probability_below_curve : 
  ∃ (A : ℝ), A = (∫ x in 0..real.pi, curve x) / real.pi ∧ A = 1 / 2 := 
sorry

end probability_below_curve_l295_295793


namespace similarity_of_intersection_triangle_l295_295179

variables (A B C D M N : Point)
variables (AB BC BD AC : Line)
variables (Plane_ADC Plane_DMN : Plane)

-- Conditions
-- 1. The edge BD is perpendicular to the plane ADC
axiom edge_BD_perpendicular_plane_ADC : BD.perpendicular Plane_ADC

-- 2. M and N are midpoints of edges AB and BC respectively.
axiom M_midpoint_AB : M.is_midpoint_of AB
axiom N_midpoint_BC : N.is_midpoint_of BC

theorem similarity_of_intersection_triangle :
  ∃ k : ℝ, k = 1 / 2 ∧ similar (MDN : Triangle) (ABC : Triangle) k :=
sorry

end similarity_of_intersection_triangle_l295_295179


namespace matrix_power_4_l295_295312

def matrix_exp := λ (A : Matrix (Fin 2) (Fin 2) ℤ) (n : ℕ), A ^ n

theorem matrix_power_4 :
  let A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -1], ![1, 1]]
  matrix_exp A 4 = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_4_l295_295312


namespace proof_problem_l295_295558

theorem proof_problem (a b : ℤ) (h1 : ∃ k, a = 5 * k) (h2 : ∃ m, b = 10 * m) :
  (∃ n, b = 5 * n) ∧ (∃ p, a - b = 5 * p) :=
by
  sorry

end proof_problem_l295_295558


namespace percent_of_x_l295_295232

variable (x : ℝ) (h : x > 0)

theorem percent_of_x (p : ℝ) : 
  (p * x = 0.21 * x + 10) → 
  p = 0.21 + 10 / x :=
sorry

end percent_of_x_l295_295232


namespace traveler_distance_l295_295047

theorem traveler_distance (h_total1 : 29 * 7 = 203) (h_total2 : 17 * 10 = 170) :
  let travel_rate := (112 : ℝ) / 203 in
  let new_hours := 170 in
  (travel_rate * new_hours) = 97 :=
by
  let travel_rate := (112 : ℝ) / 203
  let new_hours := 170
  have travel_distance := travel_rate * new_hours
  show travel_distance = 97
  sorry

end traveler_distance_l295_295047


namespace complex_number_equation_l295_295436

noncomputable def w : ℂ := sorry

theorem complex_number_equation (hw : w + w⁻¹ = complex.sqrt 2) :
  w^12 + w^(-12) = -2 :=
sorry

end complex_number_equation_l295_295436


namespace truncated_cone_volume_correct_l295_295953

-- Definition of given conditions
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height : ℝ := 8

-- Definition of the formula for the volume of a truncated cone
def truncated_cone_volume (R r h : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

-- The theorem that we need to prove
theorem truncated_cone_volume_correct :
  truncated_cone_volume large_base_radius small_base_radius height = 466.67 * Real.pi :=
by 
  sorry

end truncated_cone_volume_correct_l295_295953


namespace circle_equation_standard_l295_295943

def center : ℝ × ℝ := (-1, 1)
def radius : ℝ := 2

theorem circle_equation_standard:
  (∀ x y : ℝ, ((x + 1)^2 + (y - 1)^2 = 4) ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by 
  intros x y
  rw [center, radius]
  simp
  sorry

end circle_equation_standard_l295_295943


namespace number_of_kids_who_went_to_camp_l295_295711

variable (H C : ℕ)

def stayed_home (H : ℕ) : Prop :=
  H = 668278 - C

def more_than_stayed_home (C H : ℕ) : Prop := 
  C = H + 150780

theorem number_of_kids_who_went_to_camp (H C : ℕ) : 
  stayed_home H → more_than_stayed_home C H → C = 409529 :=
by
  intro h1 h2
  sorry

end number_of_kids_who_went_to_camp_l295_295711


namespace distance_between_skew_lines_l295_295743

-- Definitions of the rectangular prism
variables (A B C D A₁ B₁ C₁ D₁ M N : Type) [CoordSpace A] [CoordSpace B] [CoordSpace C] [CoordSpace D] [CoordSpace A₁] [CoordSpace B₁] [CoordSpace C₁] [CoordSpace D₁] [CoordSpace M] [CoordSpace N]

-- Conditions given in the problem
def prism_conditions : Prop :=
  (AB = 4) ∧ (AD = 3) ∧ (AA₁ = 2) ∧ 
  (M = midpoint D C) ∧ (N = midpoint B B₁)

-- Prove the distance between the skew lines MN and A₁B
theorem distance_between_skew_lines (h : prism_conditions) : 
  distance_skew_lines M N A₁ B = (6 * sqrt 61) / 61 :=
by sorry

end distance_between_skew_lines_l295_295743


namespace maximize_S_n_l295_295747

def a1 : ℚ := 5
def d : ℚ := -5 / 7

def S_n (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem maximize_S_n :
  (∃ n : ℕ, (S_n n ≥ S_n (n - 1)) ∧ (S_n n ≥ S_n (n + 1))) →
  (n = 7 ∨ n = 8) :=
sorry

end maximize_S_n_l295_295747


namespace martian_angles_l295_295533

-- Definitions for the problem conditions
def full_circle_clerts : ℕ := 600
def right_angle_degrees : ℕ := 90
def obtuse_angle_degrees : ℕ := 135
def full_circle_degrees : ℕ := 360

-- Theorem statement to prove the equivalence
theorem martian_angles :
  (right_angle_degrees.to_rat / full_circle_degrees.to_rat) * full_circle_clerts.to_rat = 150 ∧
  (obtuse_angle_degrees.to_rat / full_circle_degrees.to_rat) * full_circle_clerts.to_rat = 225 :=
by
  sorry

end martian_angles_l295_295533


namespace part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l295_295125

def set_A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def set_B (m : ℝ) : Set ℝ := {x | x < m}

-- Problem 1
theorem part1_A_complement_B_intersection_eq (m : ℝ) (h : m = 3) :
  set_A ∩ {x | x >= 3} = {x | 3 <= x ∧ x < 4} :=
sorry

-- Problem 2
theorem part2_m_le_neg2 (m : ℝ) (h : set_A ∩ set_B m = ∅) :
  m <= -2 :=
sorry

-- Problem 3
theorem part3_m_ge_4 (m : ℝ) (h : set_A ∩ set_B m = set_A) :
  m >= 4 :=
sorry

end part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l295_295125


namespace minimum_shoeing_time_l295_295644

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) (horses : ℕ) (hooves_per_horse : ℕ) (time_per_hoof : ℕ) 
  (total_hooves : ℕ := horses * hooves_per_horse) 
  (time_for_one_blacksmith : ℕ := total_hooves * time_per_hoof) 
  (total_parallel_time : ℕ := time_for_one_blacksmith / blacksmiths)
  (h : blacksmiths = 48)
  (h' : horses = 60)
  (h'' : hooves_per_horse = 4)
  (h''' : time_per_hoof = 5) : 
  total_parallel_time = 25 :=
by
  sorry

end minimum_shoeing_time_l295_295644


namespace apples_per_person_l295_295425

-- Define conditions
def total_apples : ℝ := 45
def number_of_people : ℝ := 3.0

-- Theorem statement: Calculate how many apples each person received.
theorem apples_per_person : 
  (total_apples / number_of_people) = 15 := 
by
  sorry

end apples_per_person_l295_295425


namespace even_product_probability_l295_295220

theorem even_product_probability (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7}) :
  let n := (s.card.choose 2),
      odd_set := s.filter (λ x, x % 2 = 1),
      odd_pairs := (odd_set.card.choose 2),
      odd_prob := odd_pairs / n,
      even_prob := 1 - odd_prob in
  even_prob = 6 / 7 :=
by
  sorry

end even_product_probability_l295_295220


namespace basketball_team_lineup_l295_295256

noncomputable def choose : ℕ → ℕ → ℕ
| n, 0       := 1
| 0, k       := 0
| n+1, k+1 := choose n k + choose n (k+1)

theorem basketball_team_lineup :
  let total_players := 12 in
  let quadruplets := 4 in
  let non_quadruplets := 8 in
  let starters := 5 in
  let at_most_two_quadruplets := 2 in
  (choose non_quadruplets starters) + 
  (choose quadruplets 1 * choose non_quadruplets (starters - 1)) + 
  (choose quadruplets 2 * choose non_quadruplets (starters - 2)) = 
  672 :=
by
  sorry

end basketball_team_lineup_l295_295256


namespace min_value_frac_l295_295022

theorem min_value_frac (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2 * y = 2) : 
  ∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  (∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  ∀ (f : ℝ), f = (1 / (x + 1) + 2 / y) → f ≥ L))) :=
sorry

end min_value_frac_l295_295022


namespace parallel_lines_l295_295412

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l295_295412


namespace incorrect_expression_l295_295052

theorem incorrect_expression : 
  ∀ (x y : ℚ), (x / y = 2 / 5) → (x + 3 * y) / x ≠ 17 / 2 :=
by
  intros x y h
  sorry

end incorrect_expression_l295_295052


namespace cross_section_not_octagon_l295_295233

def is_cross_section_possible (shape : Type) : Prop :=
  shape = EquilateralTriangle ∨ shape = Square ∨ shape = Rectangle

theorem cross_section_not_octagon :
  ¬is_cross_section_possible RegularOctagon := by
  sorry

end cross_section_not_octagon_l295_295233


namespace minimal_face_sum_of_cube_numbers_l295_295196

theorem minimal_face_sum_of_cube_numbers : 
  ∀ (cube : Fin₈ → ℕ), 
  (∀ i, cube i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) → 
  (∀ (f : Fin₄ → Fin₈), (cube (f 0) + cube (f 1) + cube (f 2) ≥ 10) ∧ 
                      (cube (f 1) + cube (f 2) + cube (f 3) ≥ 10) ∧ 
                      (cube (f 0) + cube (f 2) + cube (f 3) ≥ 10) ∧ 
                      (cube (f 0) + cube (f 1) + cube (f 3) ≥ 10)) → 
  (∃ (f : Fin₄ → Fin₈), (cube (f 0) + cube (f 1) + cube (f 2) + cube (f 3) = 16)) :=
sorry

end minimal_face_sum_of_cube_numbers_l295_295196


namespace games_bought_from_friend_is_21_l295_295499

-- Definitions from the conditions
def games_bought_at_garage_sale : ℕ := 8
def non_working_games : ℕ := 23
def good_games : ℕ := 6

-- The total number of games John has is the sum of good and non-working games
def total_games : ℕ := good_games + non_working_games

-- The number of games John bought from his friend
def games_from_friend : ℕ := total_games - games_bought_at_garage_sale

-- Statement to prove
theorem games_bought_from_friend_is_21 : games_from_friend = 21 := by
  sorry

end games_bought_from_friend_is_21_l295_295499


namespace certain_number_approximation_l295_295056

theorem certain_number_approximation (h1 : 2994 / 14.5 = 177) (h2 : 29.94 / x = 17.7) : x = 2.57455 := by
  sorry

end certain_number_approximation_l295_295056


namespace cars_pass_same_order_l295_295166

theorem cars_pass_same_order (num_cars : ℕ) (num_points : ℕ)
    (cities_speeds speeds_outside_cities : Fin num_cars → ℝ) :
    num_cars = 10 → num_points = 2011 → 
    ∃ (p1 p2 : Fin num_points), p1 ≠ p2 ∧ (∀ i j : Fin num_cars, (i < j) → 
    (cities_speeds i) / (cities_speeds i + speeds_outside_cities i) = 
    (cities_speeds j) / (cities_speeds j + speeds_outside_cities j) → p1 = p2 ) :=
by
  sorry

end cars_pass_same_order_l295_295166


namespace number_of_incorrect_propositions_l295_295480

-- Definitions of Propositions
def prop1 : Prop :=
  ∀ (L1 L2 : Line) (P : Plane), parallel L1 L2 → parallel (P.proj L1) (P.proj L2)

def prop2 : Prop :=
  ∀ (α β : Plane) (m : Line), parallel α β → (m ∈ α → parallel m β)

def prop3 : Prop :=
  ∀ (α β : Plane) (m : Line) (n : Line), (α ∩ β = m) ∧ (n ∈ α) ∧ (perpendicular n m) → perpendicular n β

-- Statement of the problem
theorem number_of_incorrect_propositions : 
  (¬ prop1 ∧ prop2 ∧ ¬ prop3) → (count_incorrect (prop1, prop2, prop3) = 2) :=
by
  sorry

end number_of_incorrect_propositions_l295_295480


namespace problem_statement_l295_295437

-- Define the given condition
def cond_1 (x : ℝ) := x + 1/x = 5

-- State the theorem that needs to be proven
theorem problem_statement (x : ℝ) (h : cond_1 x) : x^3 + 1/x^3 = 110 :=
sorry

end problem_statement_l295_295437


namespace part_i_part_ii_l295_295774

noncomputable def f (x : ℝ) : ℝ := (x / (x + 4)) * Real.exp (x + 2)

theorem part_i (x : ℝ) (hx : x > -2) : (x * Real.exp (x + 2) + x + 4 > 0) :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (Real.exp (x + 2) - a * x - 3 * a) / (x + 2)^2

theorem part_ii (a : ℝ) (ha : a ∈ set.Ico 0 1) : 
  ∃ (h : ℝ), (∀ x > -2, g x a ≥ h) ∧ set.Icc (1 / 2) (Real.exp 2 / 4) = 
  { h' | ∃ x, h = g x a } :=
sorry

end part_i_part_ii_l295_295774


namespace acute_angle_measure_l295_295959

/-
  Given:
  1. Three distinct concentric circles with radii 4, 3, and 2.
  2. Two lines passing through the center divide the circles into shaded and unshaded regions.
  3. The area of the shaded region is \(\frac{5}{8}\) of the area of the unshaded region.
  Prove that the radian measure of the acute angle formed by the two lines is \(\frac{28\pi}{143}\).
-/
 
theorem acute_angle_measure (r1 r2 r3 : ℝ) (S U θ : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2) 
  (hS_U : S = (5 / 8) * U) (H_total_area : S + U = (real.pi * (r1^2 + r2^2 + r3^2))):
  θ = (28 * real.pi) / 143 :=
by
  -- Proof required
  sorry

end acute_angle_measure_l295_295959


namespace midpoint_in_closed_set_l295_295912

variables {Point : Type} [MetricSpace Point]

noncomputable def isMidpoint (P Q R : Point) : Prop :=
  dist P Q = dist P R ∧ dist P Q + dist P R = dist Q R

noncomputable def isInteriorPoint (P : Point) (D : Set Point) : Prop :=
  ∃ r > 0, ∀ (Q : Point), dist P Q < r → Q ∈ D

theorem midpoint_in_closed_set 
  (S : Set Point) 
  (D : Set Point) 
  (P : Point)
  (hS_non_empty : S.nonempty)
  (hS_closed : is_closed S)
  (hD_contains_S : S ⊆ D) 
  (hD_minimal : ∀ (D' : Set Point), (S ⊆ D') → (D ⊆ D')) 
  (hP_interior_D : isInteriorPoint P D) :
  ∃ (Q R : Point), Q ≠ R ∧ Q ∈ S ∧ R ∈ S ∧ isMidpoint P Q R := 
sorry

end midpoint_in_closed_set_l295_295912


namespace smallest_positive_period_of_f_f_geq_neg_one_half_in_interval_l295_295783

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.cos (2 * x - (Real.pi / 3)) - 2 * Real.sin x * Real.cos x

theorem smallest_positive_period_of_f : ∀ t > 0, (∀ x, f (x + t) = f x) → t = Real.pi := 
by
  sorry

theorem f_geq_neg_one_half_in_interval : ∀ x ∈ Icc (-(Real.pi / 4)) (Real.pi / 4), f x ≥ -0.5 := 
by
  sorry

end smallest_positive_period_of_f_f_geq_neg_one_half_in_interval_l295_295783


namespace wed_production_correct_weekly_production_correct_total_wage_correct_l295_295985

def daily_wage : ℕ := 200
def daily_task : ℕ := 40
def reward : ℕ := 7
def penalty : ℕ := 8
def deviations : List ℤ := [5, -2, -1, 0, 4]

def wed_production := daily_task - 1

def planned_weekly_production := daily_task * 5

def actual_weekly_production :=
  planned_weekly_production + deviations.sum_nat_abs

def total_wage :=
  let basic_wage := daily_wage * 5
  let total_reward := (5 + 4) * reward
  let total_penalty := (2 + 1) * penalty
  basic_wage + total_reward - total_penalty

theorem wed_production_correct : wed_production = 39 := by
  sorry

theorem weekly_production_correct : actual_weekly_production = 206 := by
  sorry

theorem total_wage_correct : total_wage = 1039 := by
  sorry

end wed_production_correct_weekly_production_correct_total_wage_correct_l295_295985


namespace simplify_expression_l295_295995

theorem simplify_expression (x y : ℝ) : 
  (5 * x ^ 2 - 3 * x + 2) * (107 - 107) + (7 * y ^ 2 + 4 * y - 1) * (93 - 93) = 0 := 
by 
  sorry

end simplify_expression_l295_295995


namespace cos_double_angle_l295_295381

def tan_relation (α : ℝ) : Prop := (tan α + 1) / (tan α - 1) = 2

#check tan_relation

theorem cos_double_angle (α : ℝ) (h : tan_relation α) : cos (2 * α) = -4 / 5 :=
by sorry

end cos_double_angle_l295_295381


namespace geometric_sequence_sum_l295_295106

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = a 1 * q) (h3 : a 3 = a 1 * q^2) 
  (h4 : 2 * a 1 = a 2 + a 3) :
  q = -2 ∧ (∑ i in Finset.range n, (i + 1) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9 := 
by
  sorry

end geometric_sequence_sum_l295_295106


namespace mangoes_total_l295_295283

theorem mangoes_total (Dilan Ashley Alexis : ℕ) (h1 : Alexis = 4 * (Dilan + Ashley)) (h2 : Ashley = 2 * Dilan) (h3 : Alexis = 60) : Dilan + Ashley + Alexis = 75 :=
by
  sorry

end mangoes_total_l295_295283


namespace find_remaining_rectangle_area_l295_295277

-- Definitions of given areas
def S_DEIH : ℝ := 20
def S_HILK : ℝ := 40
def S_ABHG : ℝ := 126
def S_GHKJ : ℝ := 63
def S_DFMK : ℝ := 161

-- Definition of areas of the remaining rectangle
def S_EFML : ℝ := 101

-- Theorem statement to prove the area of the remaining rectangle
theorem find_remaining_rectangle_area :
  S_DFMK - S_DEIH - S_HILK = S_EFML :=
by
  -- This is where the proof would go
  sorry

end find_remaining_rectangle_area_l295_295277


namespace range_k_l295_295401

def curveC (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 - 4 * x + 1

def lineL (x : ℝ) (k : ℝ) : ℝ := -x - 2*k + 1

def condition (x : ℝ) (k : ℝ) : Prop :=
  lineL x k > curveC x

theorem range_k (k : ℝ) : (∀ x ∈ Ico (-3:ℝ) 3, condition x k) → k < -5/6 :=
by 
  intro h
  sorry

end range_k_l295_295401


namespace students_exceed_guinea_pigs_and_teachers_l295_295455

def num_students_per_classroom : Nat := 25
def num_guinea_pigs_per_classroom : Nat := 3
def num_teachers_per_classroom : Nat := 1
def num_classrooms : Nat := 5

def total_students : Nat := num_students_per_classroom * num_classrooms
def total_guinea_pigs : Nat := num_guinea_pigs_per_classroom * num_classrooms
def total_teachers : Nat := num_teachers_per_classroom * num_classrooms
def total_guinea_pigs_and_teachers : Nat := total_guinea_pigs + total_teachers

theorem students_exceed_guinea_pigs_and_teachers :
  total_students - total_guinea_pigs_and_teachers = 105 :=
by
  sorry

end students_exceed_guinea_pigs_and_teachers_l295_295455


namespace binary_sequence_inequality_l295_295326

theorem binary_sequence_inequality (n : ℕ)  (x : ℕ → ℝ) (h : ∀ i, x i = 0 ∨ x i = 1) :
  (finset.range (n + 1)).sum (λ i, x i / (real.sqrt 2) ^ i) ≤
  (1 + real.sqrt 2) * real.sqrt ((finset.range (n + 1)).sum (λ i, x i / 2 ^ i)) :=
sorry

end binary_sequence_inequality_l295_295326


namespace equation_of_ellipse_om_dot_op_constant_fixed_point_q_exists_l295_295749

noncomputable def problem_conditions : Prop :=
  ∃ (F1 F2 A B C D : ℝ × ℝ) (M O P : ℝ × ℝ)
    (a b : ℝ),
    (a > b ∧ b > 0) ∧ 
    (F1 ≠ F2 ∧ A ≠ B ∧ C = (-2, 0) ∧ D = (2, 0)) ∧
    ((F1, A, F2, B) form_square_with_side_length 2) ∧
    (MD_perp_to_CD M D C) ∧
    (O = (0, 0)) ∧
    (P = intersection_point_of_line_CM_and_ellipse C (M D) (ellipse_eq a b))

theorem equation_of_ellipse : problem_conditions →
  (∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
    (ellipse_eq a b = (λ x y, x^2 / 4 + y^2 / 2 = 1)) :=
sorry

theorem om_dot_op_constant : problem_conditions →
  ∃ k : ℝ, ∀ M P : ℝ × ℝ,
    M_on_ellipse M ∧ P_on_ellipse P → 
    (om_dot_op O M P = 4) :=
sorry

theorem fixed_point_q_exists : problem_conditions →
  ∃ Q : ℝ × ℝ,
    Q_on_x_axis Q ∧ 
    Q ≠ (-2, 0) ∧
    circle_condition Q (M P) D :=
sorry

end equation_of_ellipse_om_dot_op_constant_fixed_point_q_exists_l295_295749


namespace total_cleaning_time_is_100_l295_295892

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end total_cleaning_time_is_100_l295_295892


namespace angle_rotation_acute_l295_295580

theorem angle_rotation_acute (angle_ACB : ℝ) (h : angle_ACB = 50) : 
  let new_angle := (angle_ACB + 540) % 360 - 180 in 
  if new_angle < 0 then new_angle + 360 else new_angle = 50 :=
by
  -- Proof goes here
  sorry

end angle_rotation_acute_l295_295580


namespace problem1_problem2_l295_295304

-- Problem 1
theorem problem1 : ∀ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 → x = 8 :=
by
  intro x
  intro h
  sorry

-- Problem 2
theorem problem2 : ∀ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 → x = 1 :=
by
  intro x
  intro h
  sorry

end problem1_problem2_l295_295304


namespace nancy_books_count_l295_295286

-- Define the number of books that Alyssa has  
def alyssa_books : ℕ := 36

-- Define the factor by which Nancy's books exceed Alyssa's books  
def nancy_factor : ℕ := 7

-- Define Nancy's books based on the given conditions
def nancy_books (a : ℕ) (f : ℕ) : ℕ := f * a

-- The theorem we want to prove that computes Nancy's books based on the given conditions
theorem nancy_books_count (a : ℕ) (f : ℕ) : nancy_books a f = 252 := by
  -- Plugging in the values we have
  let result := nancy_books alyssa_books nancy_factor
  -- Show that the calculation result matches the expected number of books Nancy has
  show nancy_books alyssa_books nancy_factor = 252
  sorry

end nancy_books_count_l295_295286


namespace locus_of_M_is_circle_with_diameter_NB_l295_295014

-- Define the geometric problem
variables {A B C E M N : Type} [AffinePlane A B C E M N]
variables (triangle : Triangle A B C) (Midpoint_E : Midpoint E (A, C)) (Perpendicular_E_M : Perpendicular (E, M) (B, C))

-- The statement to be proved
theorem locus_of_M_is_circle_with_diameter_NB :
  ∃ (circle : Circle B N), ∀ M, M ∈ circle ↔ Locus_of_M (B, N) :=
sorry

end locus_of_M_is_circle_with_diameter_NB_l295_295014


namespace eagles_points_l295_295826

theorem eagles_points (s e : ℕ) (h1 : s + e = 52) (h2 : s - e = 6) : e = 23 :=
by
  sorry

end eagles_points_l295_295826


namespace minimum_value_l295_295033

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem minimum_value (a m n : ℝ)
    (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
    (h_a_on_graph : ∀ x, log_a a (x + 3) - 1 = 0 → x = -2)
    (h_on_line : 2 * m + n = 2)
    (h_mn_pos : m * n > 0) :
    (1 / m) + (2 / n) = 4 :=
by
  sorry

end minimum_value_l295_295033


namespace maximum_non_intersecting_chords_l295_295955

theorem maximum_non_intersecting_chords :
  ∀ (n : ℕ) (c : ℕ), n = 2006 ∧ c = 17 → 
  ∃ k : ℕ, k = 117 ∧
  ∀ (coloring : Fin n → Fin c), -- coloring function from points to colors
    ∃ (chords : Finset (Fin n × Fin n)), 
    (∀ (chord ∈ chords), (coloring chord.1) = (coloring chord.2)) ∧ 
    (∀ (c1 c2 : Fin n × Fin n), c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
      (c1.1 = c2.1 ∨ c1.1 = c2.2 ∨ c1.2 = c2.1 ∨ c1.2 = c2.2) → False) ∧
    (chords.card = k) :=
by
  sorry

end maximum_non_intersecting_chords_l295_295955


namespace total_points_scored_l295_295130

theorem total_points_scored (points_per_round : ℕ) (rounds : ℕ) (h1 : points_per_round = 42) (h2 : rounds = 2) : 
  points_per_round * rounds = 84 :=
by
  sorry

end total_points_scored_l295_295130


namespace shooting_competition_l295_295986

variable (x y : ℕ)

theorem shooting_competition (H1 : 20 * x - 12 * (10 - x) + 20 * y - 12 * (10 - y) = 208)
                             (H2 : 20 * x - 12 * (10 - x) = 20 * y - 12 * (10 - y) + 64) :
  x = 8 ∧ y = 6 := 
by 
  sorry

end shooting_competition_l295_295986


namespace ratio_x_y_z_l295_295692

theorem ratio_x_y_z (x y z : ℝ) (h1 : 0.75 * y = 0.50 * x) (h2 : 0.30 * x = 0.20 * z) :
  x : y : z = 6 : 4 : 9 :=
sorry

end ratio_x_y_z_l295_295692


namespace sequence_relation_general_formula_sum_inequality_l295_295790

def sequence_x (n : ℕ) : ℚ :=
  if n = 0 then 0 else (1/2 : ℚ) ^ n - 1

def sequence_a (n : ℕ) : ℚ :=
  2 / (sequence_x n + 1) + (-1) ^ n

theorem sequence_relation :
  ∀ n : ℕ, sequence_x (n + 1) = (sequence_x n - 1) / 2 :=
by intros; sorry

theorem general_formula :
  ∀ n : ℕ, sequence_x n = (1/2 ^ n) - 1 :=
by intros; sorry

theorem sum_inequality :
  ∀ n : ℕ, (∑ k in Finset.range (n-1), 1 / sequence_a (k+2)) < 1/2 :=
by intros; sorry

end sequence_relation_general_formula_sum_inequality_l295_295790


namespace sunny_subsets_count_l295_295856

open Set Nat

theorem sunny_subsets_count (m n : ℕ) (hm : 2 ≤ m) (hn : m ≤ n) (S : Finset ℕ) (hS : S.card = n) :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, t ⊆ S ∧ ((t.sum id) % m = 0)) ∧ T.card ≥ 2^(n - m + 1) := sorry

end sunny_subsets_count_l295_295856


namespace even_product_probability_l295_295219

theorem even_product_probability (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7}) :
  let n := (s.card.choose 2),
      odd_set := s.filter (λ x, x % 2 = 1),
      odd_pairs := (odd_set.card.choose 2),
      odd_prob := odd_pairs / n,
      even_prob := 1 - odd_prob in
  even_prob = 6 / 7 :=
by
  sorry

end even_product_probability_l295_295219


namespace div_val_is_2_l295_295898

theorem div_val_is_2 (x : ℤ) (h : 5 * x = 100) : x / 10 = 2 :=
by 
  sorry

end div_val_is_2_l295_295898


namespace exists_quadratic_function_not_obtained_by_transforming_y_eq_3x_sq_plus_1_l295_295486

theorem exists_quadratic_function_not_obtained_by_transforming_y_eq_3x_sq_plus_1 :
  ∃ (a b : ℝ), (a ≠ 3) ∧ (a ≠ -3) ∧ (b = 0) ∧ ∀ x : ℝ, (y = 3 * x^2 + 1 → y ≠ a * x^2 + b) := by
  sorry

end exists_quadratic_function_not_obtained_by_transforming_y_eq_3x_sq_plus_1_l295_295486


namespace contrapositive_equiv_l295_295935

structure Triangle :=
  (A B C : Point)
  (is_isosceles : Bool)
  (interior_angles_equal : Bool)

def contrapositive_proposition (T : Triangle) : Prop :=
  if T.is_isosceles = false then T.interior_angles_equal = false

theorem contrapositive_equiv (T : Triangle) : 
  (¬ T.is_isosceles → ¬ T.interior_angles_equal) ↔ (T.interior_angles_equal → T.is_isosceles) := by
  sorry

end contrapositive_equiv_l295_295935


namespace geometric_sequence_l295_295862

theorem geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) 
  (h2 : (3 * S 1, 2 * S 2, S 3) = (3 * S 1, 2 * S 2, S 3) ∧ (4 * S 2 = 3 * S 1 + S 3)) 
  (hq_pos : q ≠ 0) 
  (hq : ∀ n, a (n + 1) = a n * q):
  ∀ n, a n = 3^(n-1) :=
by
  sorry

end geometric_sequence_l295_295862


namespace zeros_in_15_factorial_base_18_l295_295704

theorem zeros_in_15_factorial_base_18 : ∃ k : ℕ, (∃ max_k : ℕ, 
  (∀ t : ℕ, nat.divisors_count (15!) 18 t < k ↔ t ≤ max_k) ∧ max_k = 3) := sorry

end zeros_in_15_factorial_base_18_l295_295704


namespace average_of_selected_terms_exceeds_1988_l295_295516

def selected_avg_exceeds_1988 (n : ℕ) (x : Fin n → ℝ) : Prop :=
  ∃ (selected : Finset (Fin n)), 
    (∃ i, x i > 1988) ∧
    (∀ i ∈ selected, ∃ j, i ≤ j ∧
     (∑ i in Finset.range (j - i + 1), x (Fin.ofNat (i + Finset.range (i + (j - i + 1)).min(n))).val) / (j - i + 1) > 1988) ∧
    (selected.nonempty) ∧
    (selected.sum (λ i, x i) / selected.card > 1988)

theorem average_of_selected_terms_exceeds_1988 (n : ℕ) (x : Fin n → ℝ) :
  selected_avg_exceeds_1988 n x → 
  ∀ (selected : Finset (Fin n)), 
    (∃ i, x i > 1988) →
    (∀ i ∈ selected, ∃ j, i ≤ j ∧
     (∑ i in Finset.range (j - i + 1), x (Fin.ofNat (i + Finset.range (i + (j - i + 1)).min(n))).val) / (j - i + 1) > 1988) → 
    selected.nonempty →
    (selected.sum x / selected.card > 1988) := 
sorry

end average_of_selected_terms_exceeds_1988_l295_295516


namespace lines_parallel_to_same_line_are_parallel_l295_295230

theorem lines_parallel_to_same_line_are_parallel
  {l l1 l2 : ℝ → ℝ → ℝ → Prop}
  (hl1 : ∀ (x y z : ℝ), l1 x y z → l x y z)
  (hl2 : ∀ (x y z : ℝ), l2 x y z → l x y z) :
  ∀ (x y z : ℝ), l1 x y z → l2 x y z :=
begin
  sorry
end

end lines_parallel_to_same_line_are_parallel_l295_295230


namespace corrected_mean_and_variance_l295_295824

-- Define the initial conditions and assumptions
variable {scores : Fin 50 → ℕ} 
variable (incorrect1 incorrect2 : Fin 50) 
variable (correct1 correct2 : ℕ)
variable (recorded1 : ℕ := 50)
variable (recorded2 : ℕ := 90)
acknowledge (standard_mean : (∑ i, (scores i : ℝ)) / 50 = 70)
acknowledge (standard_variance : (∑ i, ((scores i : ℝ) - 70) ^ 2) / 50 = 102)
acknowledge (incorrect_records : scores incorrect1 = recorded1 ∧ scores incorrect2 = recorded2)
acknowledge (correct_records : correct1 = 80 ∧ correct2 = 60)

-- Prove the corrected means and variances
theorem corrected_mean_and_variance (new_scores : Fin 50 → ℝ) :
  (new_scores = λ i, if i == incorrect1 then correct1 else if i == incorrect2 then correct2 else scores i) →
  ( (∑ i, new_scores i) / 50 = 70 
    ∧
    (∑ i, (new_scores i - 70) ^ 2) / 50 = 90) := by
  sorry

end corrected_mean_and_variance_l295_295824


namespace backpack_original_price_l295_295087

-- Define original price of a ring-binder
def original_ring_binder_price : ℕ := 20

-- Define the number of ring-binders bought
def number_of_ring_binders : ℕ := 3

-- Define the new price increase for the backpack
def backpack_price_increase : ℕ := 5

-- Define the new price decrease for the ring-binder
def ring_binder_price_decrease : ℕ := 2

-- Define the total amount spent
def total_amount_spent : ℕ := 109

-- Define the original price of the backpack variable
variable (B : ℕ)

-- Theorem statement: under these conditions, the original price of the backpack must be 50
theorem backpack_original_price :
  (B + backpack_price_increase) + ((original_ring_binder_price - ring_binder_price_decrease) * number_of_ring_binders) = total_amount_spent ↔ B = 50 :=
by 
  sorry

end backpack_original_price_l295_295087


namespace area_of_bat_wings_l295_295150

def point := (ℝ × ℝ)

def rectangle_JKLM (J K L M : point) : Prop :=
  J = (0, 0) ∧
  K = (2, 5) ∧
  L = (4, 3) ∧
  M = (0, 5) ∧
  dist J K = 2 ∧
  dist K L = 2 ∧
  dist L M = 2

theorem area_of_bat_wings : ∀ (J K L M : point), rectangle_JKLM J K L M → true := by
  intros J K L M h
  sorry

end area_of_bat_wings_l295_295150


namespace find_x_satisfying_sinx_plus_cosx_eq_one_l295_295340

theorem find_x_satisfying_sinx_plus_cosx_eq_one :
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0) := by
  sorry

end find_x_satisfying_sinx_plus_cosx_eq_one_l295_295340


namespace possible_integer_values_l295_295434

theorem possible_integer_values (x : ℕ) (h : (Real.ceil (Real.sqrt x) = 16)) : 
  33 = Finset.card (Finset.filter (fun n => 256 ≤ n ∧ n ≤ 288) (Finset.range 289)) :=
by sorry

end possible_integer_values_l295_295434


namespace scientific_notation_of_4600000000_l295_295921

theorem scientific_notation_of_4600000000 :
  4.6 * 10^9 = 4600000000 := 
by
  sorry

end scientific_notation_of_4600000000_l295_295921


namespace jack_pays_back_expected_amount_l295_295082

-- Definitions from the conditions
def principal : ℝ := 1200
def interest_rate : ℝ := 0.10

-- Definition for proof
def interest : ℝ := principal * interest_rate
def total_amount : ℝ := principal + interest

-- Lean statement for the proof problem
theorem jack_pays_back_expected_amount : total_amount = 1320 := by
  sorry

end jack_pays_back_expected_amount_l295_295082


namespace arithmetic_calculation_l295_295684

theorem arithmetic_calculation : 3 - (-5) + 7 = 15 := by
  sorry

end arithmetic_calculation_l295_295684


namespace parallel_lines_necessary_and_sufficient_l295_295417

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l295_295417


namespace complex_sum_equality_l295_295505

theorem complex_sum_equality (x : ℂ) (h1 : x^3013 = 1) (h2 : x ≠ 1) :
  ∑ k in Finset.range 1004, ((x^(3 * (k + 1))) / (x^(k + 1) - 1) + (x^(6 * (k + 1))) / (x^(2 * (k + 1)) - 1)) = -2008 := 
by
  sorry

end complex_sum_equality_l295_295505


namespace participant_A_enters_third_round_participant_B_receives_commendation_l295_295652

-- Conditions for Part 1
def success_prob_first (p : ℝ) := (4/5)
def success_prob_second (p : ℝ) := (3/4)

-- Question and Goal statement for Part 1
theorem participant_A_enters_third_round :
  ∀ (p : ℝ), p = (success_prob_first p + (1 - success_prob_first p) * success_prob_first p) *
                  (success_prob_second p + (1 - success_prob_second p) * success_prob_second p) →
  p = 9/10 :=
sorry

-- Conditions for Part 2
def mean_score : ℝ := 212
def std_dev (μ σ : ℝ) : ℝ := 29
def total_participants : ℕ := 2000
def top_participants : ℕ := 317
def elevated_score_participants : ℕ := 46
def participant_b_score : ℝ := 231

-- Question and Goal statement for Part 2
theorem participant_B_receives_commendation :
  ∀ (x : ℝ), x = participant_b_score →
  231 < (mean_score + std_dev mean_score std_dev) →
  False :=
sorry

end participant_A_enters_third_round_participant_B_receives_commendation_l295_295652


namespace sphere_properties_l295_295658

/-- Given the diameter of the sphere is (19/2) inches, -/
def diameter : ℚ := 19 / 2

/-- The radius is half of the diameter -/
def radius : ℚ := diameter / 2

/-- Surface area of the sphere -/
def surface_area (r : ℚ) : ℚ := 4 * Real.pi * (r ^ 2)

/-- Volume of the sphere -/
def volume (r : ℚ) : ℚ := (4 / 3) * Real.pi * (r ^ 3)

/-- We state the expected surface area and volume for the given radius -/
theorem sphere_properties :
  surface_area radius = 361 * Real.pi / 4 ∧
  volume radius = 6859 * Real.pi / 48 :=
by
  sorry

end sphere_properties_l295_295658


namespace find_ellipse_equation_l295_295750

noncomputable def ellipse_center_origin (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def foci_on_x_axis (f1 f2 : ℝ × ℝ) : Prop :=
  f1.2 = 0 ∧ f2.2 = 0

noncomputable def minor_axis_length (b : ℝ) : Prop :=
  b = 4 * real.sqrt 2

def eccentricity (c a : ℝ) : Prop :=
  c / a = 1 / 3

def ellipse_equation (a b : ℝ) : Prop :=
  (λ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1)

theorem find_ellipse_equation
  (center : ℝ × ℝ)
  (f1 f2 : ℝ × ℝ)
  (b c a : ℝ)
  (h1 : ellipse_center_origin center)
  (h2 : foci_on_x_axis f1 f2)
  (h3 : minor_axis_length b)
  (h4 : eccentricity c a)
  (h5 : a^2 = c^2 + b^2)
  : ellipse_equation (real.sqrt 36) (real.sqrt 32) :=
sorry

end find_ellipse_equation_l295_295750


namespace solve_for_x_l295_295443

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end solve_for_x_l295_295443


namespace f_eq_nine_for_k_eq_three_l295_295868

def g (k x : ℕ) : ℕ := x^2 / 10^(k-1)

def f (k : ℕ) : ℕ := 
  let start := Nat.ceil (Real.sqrt (10^k)) in
  let rec helper (n : ℕ) (prev : ℕ) : ℕ :=
    let current := g k n in
    if current - prev ≥ 3 then
      (List.range current).filter (λ m => m != prev ∧ m != current).headI
    else
      helper (n + 1) current
  helper start (g k (start - 1))

theorem f_eq_nine_for_k_eq_three : f 3 = 9 :=
by sorry

end f_eq_nine_for_k_eq_three_l295_295868


namespace linear_function_not_in_second_quadrant_l295_295387

theorem linear_function_not_in_second_quadrant (m : ℤ) (h1 : m + 4 > 0) (h2 : m + 2 ≤ 0) : 
  m = -3 ∨ m = -2 := 
sorry

end linear_function_not_in_second_quadrant_l295_295387


namespace sufficient_but_not_necessary_sin_condition_l295_295865

theorem sufficient_but_not_necessary_sin_condition (θ : ℝ) :
  (|θ - π/12| < π/12) → (sin θ < 1/2) :=
sorry

end sufficient_but_not_necessary_sin_condition_l295_295865


namespace circle_diameter_theorem_l295_295144

noncomputable def proof_problem (Q A B C D K : Point) (ω₁ ω₂ : Circle) : Prop :=
  (Q ∉ ω₁ ∧ Tangent Q A ω₁ ∧ Tangent Q B ω₁ ∧ 
   Center Q ω₂ ∧ PointOnCircle A ω₂ ∧ PointOnCircle B ω₂ ∧
   PointOnArc K A B ω₂ ∧ Inside K ω₁ ∧
   SecondIntersection (LineThrough A K) ω₁ C ∧
   SecondIntersection (LineThrough B K) ω₁ D) →
  Diameter C D ω₁

theorem circle_diameter_theorem (Q A B C D K : Point) (ω₁ ω₂ : Circle) :
  proof_problem Q A B C D K ω₁ ω₂ :=
sorry

end circle_diameter_theorem_l295_295144


namespace circle_tangent_condition_l295_295216

noncomputable def circle_tangent_range (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 = 1) ∧ (x^2 + y^2 - 6*x + 8*y + 25 - m^2 = 0) → (m ∈ set.Ioo (-4 : ℝ) 0 ∪ set.Ioo 0 4)

theorem circle_tangent_condition (m : ℝ) :
  circle_tangent_range m ↔ m ∈ set.Ioo (-4 : ℝ) 0 ∪ set.Ioo 0 4 :=
by
  sorry

end circle_tangent_condition_l295_295216


namespace problem1_problem2_problem3_l295_295997

theorem problem1 (x y : ℝ) (p q : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) 
  (h₀ : 1 ≤ p) (h₁ : 1 ≤ q) (hpq : 1 / p + 1 / q = 1) :
  x * y ≤ x^p / p + y^q / q := 
sorry

theorem problem2 (n : ℕ) (x y : Fin n → ℝ) (p q : ℝ) 
  (hp : 1 ≤ p) (hq : 1 ≤ q) (hsum : 1 / p + 1 / q = 1) 
  (hpx : ∀ i, 0 < x i) (hqy : ∀ i, 0 < y i) :
  ∑ i, x i * y i ≤ (∑ i, (x i) ^ p) ^ (1 / p) * (∑ i, (y i) ^ q) ^ (1 / q) := 
sorry

theorem problem3 (x y : ℝ) 
  (hp : is_R_or_C.of_real (3 / 2)) (hq : is_R_or_C.of_real 3) :
  (1 + x^2 * y + x^4 * y^2)^3 ≤ (1 + x^3 + x^6)^2 * (1 + y^3 + y^6) :=
sorry

end problem1_problem2_problem3_l295_295997


namespace total_gas_spent_l295_295457

theorem total_gas_spent : 
  let nc_price_per_gallon := 2.00
  let nc_gallons := 10
  let va_price_per_gallon := nc_price_per_gallon + 1.00
  let va_gallons := 10
  let nc_total := nc_price_per_gallon * nc_gallons
  let va_total := va_price_per_gallon * va_gallons
  let total_spent := nc_total + va_total
  in 
  total_spent = 50 := by
  sorry

end total_gas_spent_l295_295457


namespace inequality_abc_l295_295147

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (1 / a + 1 / b + 1 / c) ≥ (9 / (a + b + c)) :=
begin
  sorry
end

end inequality_abc_l295_295147


namespace find_common_ratio_sum_first_n_terms_l295_295096

variable (a : ℕ → ℝ) (q : ℝ) 

-- Condition: {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: a₁ is the arithmetic mean of a₂ and a₃
def arithmetic_mean (a : ℕ → ℝ) :=
  a 1 = (a 2 + a 3) / 2

-- Proposition 1: Find the common ratio q
theorem find_common_ratio (h1 : is_geometric_sequence a q) (h2 : q ≠ 1) (h3 : arithmetic_mean a) : 
  q = -2 :=
by sorry

-- Proposition 2: Find the sum of the first n terms of the sequence {n * a_n}, given a₁ = 1
def sequence_n_times_a (a : ℕ → ℝ) :=
  λ n, n * a n

def sum_of_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) f

def geom_sum (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n else (1 - q ^ n) / (1 - q)

theorem sum_first_n_terms (h1 : is_geometric_sequence a q) (h2 : q = -2) (h3 : a 1 = 1) (n : ℕ) :
  sum_of_sequence (sequence_n_times_a a) n = (1 - (1 + 3 * n) * (-2)^n) / 9 :=
by sorry

end find_common_ratio_sum_first_n_terms_l295_295096


namespace initial_zeros_of_1_over_25_pow_25_l295_295567

theorem initial_zeros_of_1_over_25_pow_25 :
  let n := 25
  let k := 25
  let base5_exp := nat.pow n k
  let base10_exp := 2 * (nat.pow 5 k)
  n = 5 → k = 25 → 10 = 2 * 5 → 
  base10_exp ≥ nat.pow (2 * k) 2 →
  decimal_initial_zeros (1 / cast base5_exp) = 33 :=
by
  intros n k base5_exp base10_exp n_eq_5 k_eq_25 ten_eq_2_5 base10_exp_bound
  let one_twenty_five := base5_exp
  let two_fifty := nat.pow 2 (2 * k)
  have calculation : decimal_initial_zeros (1 / ↓base5_exp) = decimal_initial_zeros (one / two_fifty) := sorry
  rw calculation
  exact (sorry : decimal_initial_zeros (one / two_fifty) = 33)

end initial_zeros_of_1_over_25_pow_25_l295_295567


namespace no_real_or_imaginary_t_satisfies_equation_l295_295323

theorem no_real_or_imaginary_t_satisfies_equation :
  ¬ ∃ t : ℂ, sqrt(49 - 2 * t^2) + 7 = 0 := by
  sorry

end no_real_or_imaginary_t_satisfies_equation_l295_295323


namespace gratuity_is_four_l295_295525

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end gratuity_is_four_l295_295525


namespace f_sin_x_l295_295738

noncomputable def f : ℝ → ℝ := sorry

def cos3x (x : ℝ) : ℝ := 4 * (Real.cos x) ^ 3 - 3 * (Real.cos x)

theorem f_sin_x (x : ℝ) (h : f (Real.cos x) = cos3x x) : f (Real.sin x) = - (Real.sin (3 * x)) :=
by
  sorry

end f_sin_x_l295_295738


namespace hyperbola_eccentricity_range_l295_295410

-- Defining the hyperbola and its properties
structure Hyperbola (a b : ℝ) :=
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (equation : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1))

-- Definitions from the problem
def foci_condition (a : ℝ) (λ : ℝ) (PF_1 : ℝ) : Prop :=
  λ > 0 ∧ PF_1 > 0 ∧ PQ / PF_1 = λ ∧ (5/12<= λ <= 4/3)

def eccentricity_range (e : ℝ) : Prop :=
  (sqrt(37) / 5 ≤ e ∧ e ≤ sqrt(10) / 2)

-- The Lean statement that expresses the proof problem
theorem hyperbola_eccentricity_range (a b λ PF_1 e : ℝ) (h : Hyperbola a b) :
  foci_condition a λ PF_1 → eccentricity_range e :=
sorry

end hyperbola_eccentricity_range_l295_295410


namespace total_gas_spent_l295_295456

theorem total_gas_spent : 
  let nc_price_per_gallon := 2.00
  let nc_gallons := 10
  let va_price_per_gallon := nc_price_per_gallon + 1.00
  let va_gallons := 10
  let nc_total := nc_price_per_gallon * nc_gallons
  let va_total := va_price_per_gallon * va_gallons
  let total_spent := nc_total + va_total
  in 
  total_spent = 50 := by
  sorry

end total_gas_spent_l295_295456


namespace sufficient_condition_l295_295615

theorem sufficient_condition (A B : Prop) : (A → B) ↔ (A is a sufficient condition for B) :=
sorry

end sufficient_condition_l295_295615


namespace sqrt_eq_self_l295_295815

theorem sqrt_eq_self (x : ℝ) (h : sqrt x = x) : x = 0 ∨ x = 1 := 
sorry

end sqrt_eq_self_l295_295815


namespace denominator_or_divisor_cannot_be_zero_l295_295176

theorem denominator_or_divisor_cannot_be_zero (a b c : ℝ) : b ≠ 0 ∧ c ≠ 0 → (a / b ≠ a ∨ a / c ≠ a) :=
by
  intro h
  sorry

end denominator_or_divisor_cannot_be_zero_l295_295176


namespace range_of_a_l295_295818

theorem range_of_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 1 < 0) : a < -2 ∨ a > 2 :=
sorry

end range_of_a_l295_295818


namespace find_g_3_8_l295_295185

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l295_295185


namespace partition_set_l295_295694

open Finset

theorem partition_set (S : Finset ℕ) (hS : S = (range 1989).image (λ n, n + 1)) :
  ∃ (A: Fin 118 → Finset ℕ),
    (∀ i : Fin 118, (A i).card = 17) ∧ 
    (∀ i : Fin 118, ∑ x in A i, x = ∑ x in A 0, x) ∧ 
    (∀ i j : Fin 118, i ≠ j → disjoint (A i) (A j)) :=
by
  sorry

end partition_set_l295_295694


namespace calc_expression_l295_295305

theorem calc_expression :
  (2014 * 2014 + 2012) - 2013 * 2013 = 6039 :=
by
  -- Let 2014 = 2013 + 1 and 2012 = 2013 - 1
  have h2014 : 2014 = 2013 + 1 := by sorry
  have h2012 : 2012 = 2013 - 1 := by sorry
  -- Start the main proof
  sorry

end calc_expression_l295_295305


namespace gilda_marbles_remaining_percentage_l295_295360

-- Define the initial amount of marbles
def initial_marbles : ℝ := 1 -- normalize to 1 for simplicity of percentage calculation

-- Define the percentage calculations after each person
def after_pedro (m : ℝ) : ℝ := m * 0.70
def after_ebony (m : ℝ) : ℝ := m * 0.95
def after_jimmy (m : ℝ) : ℝ := m * 0.70
def after_tina (m : ℝ) : ℝ := m * 0.90

-- Define the final calculation for the percentage remaining
def final_percentage (m : ℝ) : ℝ :=
  after_tina (after_jimmy (after_ebony (after_pedro m))) / initial_marbles * 100

-- The theorem to be proved
theorem gilda_marbles_remaining_percentage :
  final_percentage initial_marbles = 41.895 :=
by
  sorry

end gilda_marbles_remaining_percentage_l295_295360


namespace second_race_distance_l295_295471

theorem second_race_distance (Va Vb Vc : ℝ) (D : ℝ)
  (h1 : Va / Vb = 10 / 9)
  (h2 : Va / Vc = 80 / 63)
  (h3 : Vb / Vc = D / (D - 100)) :
  D = 800 :=
sorry

end second_race_distance_l295_295471


namespace maximum_guarded_apples_l295_295134

theorem maximum_guarded_apples (a1 a2 a3 a4 a5 a6 : ℤ) (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6) :
  ∃ (x1 x2 x3 x4 : ℤ), 
  let 
    x1 := (a1 + a2 - a3) / 2,
    x2 := (a1 + a3 - a2) / 2,
    x3 := (a2 + a3 - a1) / 2,
    x4 := a4 - x3,
    sums := [x1 + x2, x1 + x3, x1 + x4, x2 + x3, x2 + x4, x3 + x4].toFinset
  in 
  let equal_sums := {x ∈ sums | x = a1 ∨ x = a2 ∨ x = a3 ∨ x = a4}.card,
      greater_sums := {x ∈ sums | x ≥ a5 ∨ x ≥ a6 ∨ x ∉ {a1, a2, a3, a4}}.card 
  in equal_sums * 3 + greater_sums = 14 :=
begin
  -- Proof skipped.
  sorry,
end

end maximum_guarded_apples_l295_295134


namespace distance_P_to_center_C_l295_295008

/-- Define the given line l: y = 2x --/
def line_l (x y : ℝ) : Prop := y = 2 * x

/-- Define the given circle C: (x-8)^2 + (y-1)^2 = 2 --/
def circle_C (x y : ℝ) : Prop := (x - 8)^2 + (y - 1)^2 = 2

/-- Define the center of circle C --/
def center_C : (ℝ × ℝ) := (8, 1)

/-- Define the distance function between a point and a line --/
def distance_point_to_line (A B C x1 y1 : ℝ) : ℝ := 
  abs (A * x1 + B * y1 + C) / real.sqrt (A^2 + B^2)

/-- Prove the distance from point P on line l to the center of circle C is 3*sqrt(5) --/
theorem distance_P_to_center_C (P : ℝ × ℝ) (hP : line_l P.1 P.2) : 
  (distance_point_to_line 2 (-1) 0 8 1) = 3 * real.sqrt 5 := by
    sorry

end distance_P_to_center_C_l295_295008


namespace tommy_money_situation_l295_295963

theorem tommy_money_situation :
  let books := 8
  let cost_per_book := 5
  let amount_to_save := 27
  let total_cost := books * cost_per_book
  let initial_money := total_cost - amount_to_save
  initial_money = 13 :=
by
  unfold books cost_per_book amount_to_save total_cost initial_money
  sorry

end tommy_money_situation_l295_295963


namespace clean_car_time_l295_295886

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l295_295886


namespace percentage_of_boys_playing_soccer_l295_295079

theorem percentage_of_boys_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (students_playing_soccer : ℕ)
  (girl_students_not_playing_soccer : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : students_playing_soccer = 250)
  (h4 : girl_students_not_playing_soccer = 89) :
  (students_playing_soccer - (total_students - boys - girl_students_not_playing_soccer)) * 100 / students_playing_soccer = 86 :=
by
  sorry

end percentage_of_boys_playing_soccer_l295_295079


namespace initial_population_l295_295074

variable (P : ℝ) -- Initial population of the village.

-- Conditions as definitions:
def population_after_bombardment (P : ℝ) : ℝ := 0.85 * P
def population_after_disaster (P : ℝ) : ℝ := 0.75 * population_after_bombardment P
def final_population (P : ℝ) : ℝ := 1.10 * population_after_disaster P

-- The proof statement:
theorem initial_population (h : final_population P = 4555) : P ≈ 6496 := by
  sorry

end initial_population_l295_295074


namespace find_a_1_l295_295503

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n ≥ 2, a n = a (n - 1) + 2

def sum_n_terms (a : ℕ → ℤ) : ℕ → ℤ 
| 0 => 0
| (n + 1) => a (n + 1) + sum_n_terms n

theorem find_a_1 (h1 : is_arithmetic_sequence a) (h2 : sum_n_terms a 3 = 9) : 
  a 1 = 1 :=
sorry

end find_a_1_l295_295503


namespace min_val_z_is_7_l295_295622

noncomputable def min_val_z (x y : ℝ) (h : x + 3 * y = 2) : ℝ := 3^x + 27^y + 1

theorem min_val_z_is_7  : ∃ x y : ℝ, x + 3 * y = 2 ∧ min_val_z x y (by sorry) = 7 := sorry

end min_val_z_is_7_l295_295622


namespace planeThroughPointAndLine_l295_295719

theorem planeThroughPointAndLine :
  ∃ A B C D : ℤ, (A = -3 ∧ B = -4 ∧ C = -4 ∧ D = 14) ∧ 
  (∀ x y z : ℝ, x = 2 ∧ y = -3 ∧ z = 5 ∨ (∃ t : ℝ, x = 4 * t + 2 ∧ y = -5 * t - 1 ∧ z = 2 * t + 3) → A * x + B * y + C * z + D = 0) :=
sorry

end planeThroughPointAndLine_l295_295719


namespace distance_from_start_l295_295913

-- Define the initial conditions
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (5 + 4 * Real.sqrt 2, 4 * Real.sqrt 2)

-- Prove the distance from A to C
theorem distance_from_start (A B C : ℝ × ℝ) : 
  let d := Real.sqrt ((5 + 4 * Real.sqrt 2)^2 + (4 * Real.sqrt 2)^2)
  d = Real.sqrt (89 + 40 * Real.sqrt 2) :=
by
  sorry

end distance_from_start_l295_295913


namespace max_sphere_radius_in_cone_l295_295463

theorem max_sphere_radius_in_cone (c : ℝ) (r : ℝ) :
  ∀(a b : ℝ), a = 2 → b = sqrt 3 - 1 → r ≤ b := sorry

end max_sphere_radius_in_cone_l295_295463


namespace units_digit_7_pow_2023_l295_295709

-- We start by defining a function to compute units digit of powers of 7 modulo 10.
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  (7 ^ n) % 10

-- Define the problem statement: the units digit of 7^2023 is equal to 3.
theorem units_digit_7_pow_2023 : units_digit_of_7_pow 2023 = 3 := sorry

end units_digit_7_pow_2023_l295_295709


namespace tenisha_total_remaining_pets_l295_295167

-- Define the initial counts of dogs and cats
def total_dogs : ℕ := 40
def total_cats : ℕ := 30

-- Define the fractions of female dogs and cats
def fraction_female_dogs : ℚ := 5 / 8
def fraction_female_cats : ℚ := 4 / 6

-- Define the fractions of female dogs and cats that give birth
def fraction_birth_dogs : ℚ := 2 / 3
def fraction_birth_cats : ℚ := 5 / 8

-- Define the number of offspring per female dog and cat that gives birth
def puppies_per_dog : ℕ := 14
def kittens_per_cat : ℕ := 6

-- Define the donation fractions for puppies and kittens
def donation_fraction_puppies : ℚ := 25 / 100
def donation_fraction_kittens : ℚ := 30 / 100

-- Prove that the total number of puppies and kittens remaining after donation is 219
theorem tenisha_total_remaining_pets : 
  let female_dogs := fraction_female_dogs * total_dogs,
      birth_dogs := fraction_birth_dogs * female_dogs,
      puppies_born := birth_dogs * puppies_per_dog,
      puppies_donated := puppies_born * donation_fraction_puppies,
      puppies_remaining := puppies_born - puppies_donated,

      female_cats := fraction_female_cats * total_cats,
      birth_cats := fraction_birth_cats * female_cats,
      kittens_born := birth_cats * kittens_per_cat,
      kittens_donated := kittens_born * donation_fraction_kittens,
      kittens_remaining := kittens_born - kittens_donated,

      total_remaining_pets := puppies_remaining + kittens_remaining
  in total_remaining_pets = 219 := sorry

end tenisha_total_remaining_pets_l295_295167


namespace line_parallel_xaxis_l295_295181

theorem line_parallel_xaxis (x y : ℝ) : y = 2 ↔ (∃ a b : ℝ, a = 4 ∧ b = 2 ∧ y = 2) :=
by 
  sorry

end line_parallel_xaxis_l295_295181


namespace true_statements_l295_295007

variable (f : ℝ → ℝ)

-- If f(x+1) + f(1-x) = 0, then the graph of y = f(x) is symmetric about the point (1,0)
def symmetric_about_point (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 1) + f(1 - x) = 0 → ∀ y, f(x) = f(2 - x - y)

-- If f(1 + x) + f(x - 1) = 0, then the function y = f(x) has a period of 4
def period_four (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(1 + x) + f(x - 1) = 0 → f(x + 4) = f(x)

theorem true_statements (f : ℝ → ℝ) :
  symmetric_about_point f ∧ period_four f := sorry

end true_statements_l295_295007


namespace distance_between_parallel_lines_l295_295800

theorem distance_between_parallel_lines (a : ℝ)
  (h₁ : 3 * x + 4 * y - 4 = 0)
  (h₂ : a * x + 8 * y + 2 = 0)
  (parallel : -3 / 4 = -a / 8) : distance l₁ l₂ = 1 := 
sorry

end distance_between_parallel_lines_l295_295800


namespace perpendicular_intersection_parallel_distance_l295_295043

-- Define the two lines l1 and l2 as given in the problem
def l1 (x y : ℝ) : Prop := 2 * x + y + 4 = 0
def l2 (a x y : ℝ) : Prop := a * x + 4 * y + 1 = 0

-- First theorem: If l1 and l2 are perpendicular, then they intersect at the point (-3/2, -1)
theorem perpendicular_intersection (a : ℝ) (x y : ℝ) :
  (2 * x + y + 4 = 0) ∧ (a * x + 4 * y + 1 = 0) ∧ (2 * (-a/4) = -1) →
  (x = -3/2 ∧ y = -1) := 
sorry

-- Second theorem: If l1 and l2 are parallel, then the distance between them is 3√5 / 4
theorem parallel_distance (a : ℝ) :
  (2 = a / 4) →
  (a ≠ 8) → -- The restriction is needed to ensure we are dealing with two distinct lines.
  (∀ (x y : ℝ), (l1 x y → (8 * x + 4 * y + 16 = 0)) ∧ (l2 a x y → (8 * x + 4 * y +  1 = 0))) →
  distance (a : ℝ) : (ℝ) :=
  distance = (3 * real.sqrt 5 / 4)
 := 
sorry

end perpendicular_intersection_parallel_distance_l295_295043


namespace weight_of_one_serving_l295_295351

theorem weight_of_one_serving
  (total_servings : ℕ)
  (chicken_weight_pounds : ℝ)
  (stuffing_weight_ounces : ℝ)
  (ounces_per_pound : ℝ)
  (total_servings = 12)
  (chicken_weight_pounds = 4.5)
  (stuffing_weight_ounces = 24)
  (ounces_per_pound = 16) :
  (chicken_weight_pounds * ounces_per_pound + stuffing_weight_ounces) / total_servings = 8 :=
by
  sorry

end weight_of_one_serving_l295_295351


namespace All_China_Women_Math_Olympiad_2003_l295_295834

variable {R : Type*} [OrderedField R]

theorem All_China_Women_Math_Olympiad_2003
  (a b c : R)
  (A B C : R)
  (α : R)
  (AB : R → R → R)
  (BC : R → R → R)
  (CA : R → R → R)
  (AD BE CF DE DF : R)
  (AD_internal_bisector BE_internal_bisector CF_internal_bisector : R)
  (internal_angle_bisectors_eq : DE = DF) :
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  triangle_side_lengths := (AB = c) ∧ (BC = a) ∧ (CA = b) →
  internal_angle_bisectors := (AD_internal_bisector) ∧ (BE_internal_bisector) ∧ (CF_internal_bisector) →
  angle_bisectors_eq := (internal_angle_bisectors_eq) →
  (fraction_property : a / (b + c) = b / (c + a) + c / (a + b)) ∧ (angle_property : A > 90) :=
sorry

end All_China_Women_Math_Olympiad_2003_l295_295834


namespace problem1_problem2_l295_295309

-- Problem 1: Prove 2√6 < 5
theorem problem1 : 2 * Real.sqrt 6 < 5 :=
by
  have h1 : Real.sqrt 24 = 2 * Real.sqrt 6 := sorry
  have h2 : 5 = Real.sqrt 25 := sorry
  have h3 : 24 < 25 := by norm_num
  have h4 : Real.sqrt 24 < Real.sqrt 25 := Real.sqrt_lt_sqrt_iff.mpr h3 -- Using the increasing property of sqrt
  rw [←h1, ←h2] at h4
  exact h4

-- Problem 2: Prove -√5 < -√2
theorem problem2 : -Real.sqrt 5 < -Real.sqrt 2 :=
by
  have h1 : Real.sqrt 5 > Real.sqrt 2 := Real.sqrt_lt_sqrt_iff.mpr (by norm_num : 2 < 5)
  -- Negate both sides
  exact neg_lt_neg h1

end problem1_problem2_l295_295309


namespace three_friends_visit_exactly_27_days_l295_295426

theorem three_friends_visit_exactly_27_days
  (A B C D : ℕ) (hA : A = 6) (hB : B = 8) (hC : C = 10) (hD : D = 12) :
  let L := Nat.lcm (Nat.lcm A B) (Nat.lcm C D) in
  360 / L * (1 + 360 / (24 * 6)) = 27 := sorry

end three_friends_visit_exactly_27_days_l295_295426


namespace nina_walking_distance_l295_295851

def distance_walked_by_john : ℝ := 0.7
def distance_john_further_than_nina : ℝ := 0.3

def distance_walked_by_nina : ℝ := distance_walked_by_john - distance_john_further_than_nina

theorem nina_walking_distance :
  distance_walked_by_nina = 0.4 :=
by
  sorry

end nina_walking_distance_l295_295851


namespace regression_line_passes_through_center_l295_295038

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 1.5 * x - 15

-- Define the condition of the sample center point
def sample_center (x_bar y_bar : ℝ) : Prop :=
  y_bar = regression_eq x_bar

-- The proof goal
theorem regression_line_passes_through_center (x_bar y_bar : ℝ) (h : sample_center x_bar y_bar) :
  y_bar = 1.5 * x_bar - 15 :=
by
  -- Using the given condition as hypothesis
  exact h

end regression_line_passes_through_center_l295_295038


namespace one_serving_weight_l295_295353

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end one_serving_weight_l295_295353


namespace finite_new_zeros_insertion_l295_295513

theorem finite_new_zeros_insertion (a b : ℕ) (h_distinct : a ≠ b) :
  ∀ f, ∃ N : ℕ, ∀ k, k ≥ N → f(k) ≤ 2 * N - (ab - a - b) :=
begin
  sorry
end

end finite_new_zeros_insertion_l295_295513


namespace length_to_width_ratio_l295_295852

/-- Let the perimeter of the rectangular sandbox be 30 feet,
    the width be 5 feet, and the length be some multiple of the width.
    Prove that the ratio of the length to the width is 2:1. -/
theorem length_to_width_ratio (P w : ℕ) (h1 : P = 30) (h2 : w = 5) (h3 : ∃ k, l = k * w) : 
  ∃ l, (P = 2 * (l + w)) ∧ (l / w = 2) := 
sorry

end length_to_width_ratio_l295_295852


namespace henry_initial_money_l295_295045

variable (x : ℤ)

theorem henry_initial_money : (x + 18 - 10 = 19) → x = 11 :=
by
  intro h
  sorry

end henry_initial_money_l295_295045


namespace part1_part2_l295_295408

def f (x m : ℝ) : ℝ := |x - 1| - |2 * x + m|

theorem part1 (x : ℝ) (m : ℝ) (h : m = -4) : 
    f x m < 0 ↔ x < 5 / 3 ∨ x > 3 := 
by 
  sorry

theorem part2 (x : ℝ) (h : 1 < x) (h' : ∀ x, 1 < x → f x m < 0) : 
    m ≥ -2 :=
by 
  sorry

end part1_part2_l295_295408


namespace ellipse_area_fixed_l295_295017

open Real

theorem ellipse_area_fixed 
  (C : Set (ℝ × ℝ))
  (hC : ∀ x y, (x, y) ∈ C ↔ (x^2 / 4 + y^2 / 2 = 1))
  (a b : ℝ)
  (ha : a = 2)
  (hb : b = sqrt 2)
  (ecc : sqrt 2 / 2 * a = sqrt 2)
  (hline : ∀ x y, y = 1 → (x^2) / 4 + 1 / 2 = 1 → 2sqrt 2) :
  ∃ S : ℝ, ∀ (A B D O : ℝ × ℝ), 
    (A ∈ C ∧ B ∈ C ∧ D ∈ C ∧ O = (0, 0)) →
    (vector_add O A + vector_add O B = vector_add O D) →
    S = sqrt 6 := 
sorry

end ellipse_area_fixed_l295_295017


namespace consecutive_reposition_transformation_fixed_point_l295_295541

def num_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.length

def num_of_odd_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.countp (λ d, d % 2 = 1)

def num_of_even_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.countp (λ d, d % 2 = 0)

def reposition_transformation (n : ℕ) : ℕ :=
  let a := num_of_digits n
  let b := num_of_odd_digits n
  let c := num_of_even_digits n
  a * 100 + b * 10 + c

theorem consecutive_reposition_transformation_fixed_point (n : ℕ) (h : 1000 ≤ n ∧ n < 10000) :
  ∃ k : ℕ, ∀ m : ℕ, reposition_transformation^[m] n = k :=
sorry

end consecutive_reposition_transformation_fixed_point_l295_295541


namespace geom_sum_3m_l295_295026

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (m : ℕ)

axiom geom_sum_m : S m = 10
axiom geom_sum_2m : S (2 * m) = 30

theorem geom_sum_3m : S (3 * m) = 70 :=
by
  sorry

end geom_sum_3m_l295_295026


namespace proof_problem_l295_295508

variable {Point : Type} [LinearOrder Point]
variables (A B C D : Point) (AC CB AD DB AB : ℝ)
variable (hCond : AC / CB + AD / DB = 0)
variable (h1 : AC = abs (C - A))
variable (h2 : CB = abs (B - C))
variable (h3 : AD = abs (D - A))
variable (h4 : DB = abs (B - D))
variable (h5 : AB = abs (B - A))

theorem proof_problem
  (hCond : AC / CB + AD / DB = 0)
  (h1 : AC = abs (C - A))
  (h2 : CB = abs (B - C))
  (h3 : AD = abs (D - A))
  (h4 : DB = abs (B - D))
  (h5 : AB = abs (B - A)) :
  1 / AC + 1 / AD = 2 / AB :=
sorry

end proof_problem_l295_295508


namespace slope_absolute_value_l295_295599

noncomputable def point := (ℝ × ℝ)
noncomputable def circle := Σ (c: point), ℝ -- center and radius

def line_slope (p1 p2 : point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def circle1 : circle := ⟨(14, 92), 3⟩
def circle2 : circle := ⟨(17, 76), 3⟩
def circle3 : circle := ⟨(19, 84), 3⟩

def line := (p: point) -> ℝ 

theorem slope_absolute_value : 
  ∃ (m: ℝ), ∃ (b: ℝ), (∀ x: ℝ, x * m + b = 0 ∧ (x, x * m + b) = (17, 76)) ∧ (|m| = 24) := sorry

end slope_absolute_value_l295_295599


namespace sum_of_four_primes_divisible_by_60_l295_295754

theorem sum_of_four_primes_divisible_by_60
  (p q r s : ℕ) 
  (hp : p.prime) 
  (hq : q.prime) 
  (hr : r.prime) 
  (hs : s.prime)
  (h1 : 5 < p) 
  (h2 : p < q) 
  (h3 : q < r) 
  (h4 : r < s) 
  (h5 : s < p + 10) : 
  ∃ k : ℕ, p + q + r + s = 60 * k :=
by
  sorry

end sum_of_four_primes_divisible_by_60_l295_295754


namespace math_proof_problem_l295_295725

noncomputable def floorRootSum (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum (λ k, int.floor (n ^ (1 / (k+1) : ℝ)))

noncomputable def floorLogSum (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum (λ k, int.floor (Real.log n / Real.log (k + 1)))

theorem math_proof_problem (n : ℕ) (hn : 1 ≤ n): 
  floorRootSum n = floorLogSum n :=
by sorry

end math_proof_problem_l295_295725


namespace number_of_solutions_l295_295726

def is_prime (p : ℕ) : Prop := Nat.Prime p

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n = 1 then 1 else (Nat.factors n).last!

def satisfies_conditions (n : ℕ) : Prop :=
  let pn := greatest_prime_factor n
  let pn72 := greatest_prime_factor (n + 72)
  pn * pn = n ∧ pn72 * pn72 = n + 72

theorem number_of_solutions : ∃! n : ℕ, satisfies_conditions n := 
by sorry

end number_of_solutions_l295_295726


namespace p_implies_q_p_not_necessary_for_q_p_sufficient_but_not_necessary_l295_295291

noncomputable def f (x m : ℝ) : ℝ := exp x + 2 * x ^ 2 + m * x + 1

theorem p_implies_q (m : ℝ) :
  (∀ x ≥ 0, deriv (λ y, exp y + 2 * y ^ 2 + m * y + 1) x ≥ 0) →
  m ≥ -5 :=
begin
  sorry
end

theorem p_not_necessary_for_q (m : ℝ) :
  ¬ ((∀ x ≥ 0, deriv (λ y, exp y + 2 * y ^ 2 + m * y + 1) x ≥ 0) ↔ m ≥ -5) :=
begin
  sorry
end

theorem p_sufficient_but_not_necessary (m : ℝ) :
  (∀ x ≥ 0, deriv (λ y, exp y + 2 * y ^ 2 + m * y + 1) x ≥ 0) → (m ≥ -5) ∧
  ¬ ((∀ x ≥ 0, deriv (λ y, exp y + 2 * y ^ 2 + m * y + 1) x ≥ 0) ↔ m ≥ -5) := 
begin
  split,
  { apply p_implies_q },
  { apply p_not_necessary_for_q },
end

end p_implies_q_p_not_necessary_for_q_p_sufficient_but_not_necessary_l295_295291


namespace cube_root_fraction_eq_l295_295332

theorem cube_root_fraction_eq :
  let a := 16
  let b := 33 / 2
  (∛(a / b) = 2 * ∛4 / ∛33) := by
  sorry

end cube_root_fraction_eq_l295_295332


namespace pyramid_volume_proof_l295_295547

-- Define the given conditions as a Lean structure
structure Pyramid (EFGH : Type) (Q : Type) :=
(EF FG : ℝ)
(QF : ℝ)
(h1 : EF = 10)
(h2 : FG = 5)
(h3 : QF = 20)

-- Define the volume calculation
def volume_of_pyramid (p : Pyramid ℝ ℝ) : ℝ :=
  let base_area := p.EF * p.FG
  let altitude := Real.sqrt (p.QF^2 - p.EF^2)
  (1 / 3) * base_area * altitude

-- State the theorem
theorem pyramid_volume_proof (p : Pyramid ℝ ℝ) (h1 : p.EF = 10) (h2 : p.FG = 5) (h3 : p.QF = 20) : 
  volume_of_pyramid p = (500 * Real.sqrt 3) / 3 :=
by
  sorry

end pyramid_volume_proof_l295_295547


namespace independence_test_is_most_convincing_method_l295_295324

theorem independence_test_is_most_convincing_method
  (male_participants : ℕ) (male_opposing_views : ℕ)
  (female_participants : ℕ) (female_opposing_views : ℕ) :
  male_participants = 2548 →
  male_opposing_views = 1560 →
  female_participants = 2452 →
  female_opposing_views = 1200 →
  most_convincing_method male_participants male_opposing_views female_participants female_opposing_views = "Independence test" :=
by
  intros h1 h2 h3 h4
  sorry

end independence_test_is_most_convincing_method_l295_295324


namespace green_more_than_blue_l295_295331

def disks_in_bag : Type := Σ n : ℕ, n

def blue_disks (x : disks_in_bag.1) : ℕ := 3 * x
def yellow_disks (x : disks_in_bag.1) : ℕ := 7 * x
def green_disks (x : disks_in_bag.1) : ℕ := 8 * x

theorem green_more_than_blue (total_disks : ℕ) (h : total_disks = 54) :
  let x := total_disks / 18 in green_disks x - blue_disks x = 15 :=
begin
  sorry
end

end green_more_than_blue_l295_295331


namespace running_yardage_l295_295592

theorem running_yardage (passes_yardage total_yardage : ℕ) : passes_yardage = 60 → total_yardage = 150 → total_yardage - passes_yardage = 90 :=
by
  intros h_passes h_total
  rw [h_passes, h_total]
  sorry

end running_yardage_l295_295592


namespace arithmetic_mean_of_a_X_is_2015_l295_295873

open Set

noncomputable def A := {1..2014}
def a_X (X : Set ℕ) (hX : X ⊆ A) (h_nonempty : X.nonempty) : ℕ := X.min' (Set.nonempty_def.1 h_nonempty) + X.max' (Set.nonempty_def.1 h_nonempty)

theorem arithmetic_mean_of_a_X_is_2015 :
  (∑ X in (powerset A).to_finset, if X.nonempty then a_X X (mem_powerset.mp (mem_to_finset.mp (mem_finset.mp _))) X.nonempty else 0) / ((powerset A).to_finset.card - 1) = 2015 :=
sorry

end arithmetic_mean_of_a_X_is_2015_l295_295873


namespace triangle_CD_length_l295_295213

theorem triangle_CD_length :
  ∀ (A B C D : Type) [Triangle A B C],
  AB = 1 →
  ∠ABC = 28 * Real.pi / 180 →
  ∠BCA = 47 * Real.pi / 180 →
  AD = BD →
  CD = sin (47 * Real.pi / 180) * sin (19 * Real.pi / 180) / sin (75 * Real.pi / 180) :=
by sorry

end triangle_CD_length_l295_295213


namespace find_g_3_8_l295_295184

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l295_295184


namespace simon_treasures_l295_295914

variable (sand_dollars sea_glass seashells total_treasures : Nat)

def collected_sand_dollars : sand_dollars = 10 := 
    by sorry

def collected_sea_glass : sea_glass = 3 * sand_dollars := 
    by sorry

def collected_seashells : seashells = 5 * sea_glass := 
    by sorry

def collected_total_treasures : total_treasures = sand_dollars + sea_glass + seashells := 
    by sorry

theorem simon_treasures : total_treasures = 190 := 
    by
        rw [collected_total_treasures, collected_seashells, collected_sea_glass, collected_sand_dollars]
        simp
        sorry

end simon_treasures_l295_295914


namespace possible_values_of_reciprocal_l295_295124

theorem possible_values_of_reciprocal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ S, S = { x : ℝ | x >= 9 } ∧ (∃ x, x = (1/a + 1/b) ∧ x ∈ S) :=
sorry

end possible_values_of_reciprocal_l295_295124


namespace intercepts_equal_not_in_second_quadrant_l295_295520

-- Part I
theorem intercepts_equal (a : ℝ) :
  let l := (a + 1) * x + y + 2 + a = 0 in
  (∃ a, (-a - 2) = (-a - 2) / (a + 1)) ↔ (a = 0 ∨ a = -2) :=
sorry

-- Part II
theorem not_in_second_quadrant (a : ℝ) :
  let l := y = -(a + 1) * x - a - 2 in
  (∃ a, l does not pass through second quadrant) ↔ (-2 ≤ a ∧ a ≤ -1) :=
sorry

end intercepts_equal_not_in_second_quadrant_l295_295520


namespace fixed_stable_points_eq_nonempty_l295_295727

def isFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def isStablePoint (f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x

def fixedPoints (f : ℝ → ℝ) : Set ℝ := { x | isFixedPoint f x }
def stablePoints (f : ℝ → ℝ) : Set ℝ := { x | isStablePoint f x }

theorem fixed_stable_points_eq_nonempty (a : ℝ) (h : fixedPoints (λ x, a * x^2 - 1) = stablePoints (λ x, a * x^2 - 1) ∧ fixedPoints (λ x, a * x^2 - 1) ≠ ∅) :
  -1 / 4 ≤ a ∧ a ≤ 3 / 4 := 
sorry

end fixed_stable_points_eq_nonempty_l295_295727


namespace part_one_part_two_l295_295405

section
variables (x p : ℝ)

def f (x : ℝ) : ℝ := x - log x - 1

-- Statement for (1)
theorem part_one (hx : x > 0) : f x ≥ 0 := sorry

-- Statement for (2)
theorem part_two (h_cond : ∀ (x : ℝ), 1 ≤ x → f (1 / x) ≥ (log x)^2 / (p + log x)) : 
  2 ≤ p := sorry

end

end part_one_part_two_l295_295405


namespace jill_arrives_before_jack_by_30_minutes_l295_295496

def jill_speed : ℝ := 6 -- Jill's speed in miles per hour
def jack_speed : ℝ := 3 -- Jack's speed in miles per hour
def distance_to_campsite : ℝ := 3 -- Distance to the campsite in miles

theorem jill_arrives_before_jack_by_30_minutes : 
  (distance_to_campsite / jack_speed * 60) - (distance_to_campsite / jill_speed * 60) = 30 :=
by 
  sorry

end jill_arrives_before_jack_by_30_minutes_l295_295496


namespace find_m_l295_295751

-- Define the ellipse with given equation and conditions
def ellipse (x y a b : ℝ) : Prop :=
  a > b ∧ a > 0 ∧ b > 0 ∧ ((x^2) / (a^2) + (y^2) / (b^2) = 1)

-- Define the conditions that the ellipse passes through a specific point
def passes_through (x y : ℝ) (a b : ℝ) : Prop :=
  (ellipse 1 (3 / 2) a b)

-- Define the line that intersects the ellipse
def intersects (x y m : ℝ) : Prop :=
  y = (3 / 2) * x + m

-- Define the slopes k1 and k2, and their ratio condition
def slopes_and_ratio (k1 k2 m : ℝ) : Prop :=
  2 * k2 = k1 ∧ k1 ≠ 0

theorem find_m :
  ∃ a b m : ℝ,
  a > b ∧ a > 0 ∧ b > 0 ∧
  ellipse 1 (3 / 2) a b ∧
  (∑ (x y : ℝ), passes_through x y a b → intersects x y m) ∧
  slopes_and_ratio 2 1 m ∧
  (m = 1) := 
sorry

end find_m_l295_295751


namespace only_constant_prime_polynomial_l295_295626

def is_fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := is_fibonacci n + is_fibonacci (n+1)

def assigns_positive_prime (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ f (is_fibonacci n) = p

theorem only_constant_prime_polynomial (f : ℕ → ℕ) :
  assigns_positive_prime f →
  ∃ p : ℕ, Nat.Prime p ∧ ∀ x : ℕ, f x = p :=
  sorry

end only_constant_prime_polynomial_l295_295626


namespace minimum_value_of_a_minus_2b_l295_295276

-- Define the conditions
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 - a * x + b

-- Problem: Prove the minimum value of a - 2b is -1 given these conditions
theorem minimum_value_of_a_minus_2b (a b : ℝ) 
  (h1 : ∃ x, (-1 <= x ∧ x <= 1 ∧ quadratic_function a b x = 0))
  (h2 : ∃ y, (1 <= y ∧ y <= 2 ∧ quadratic_function a b y = 0))
  (h3 : a + b + 1 >= 0)
  (h4 : a - b - 1 >= 0)
  (h5 : 2 * a - b - 4 <= 0) :
  (a - 2 * b) >= -1 :=
begin
  sorry
end

end minimum_value_of_a_minus_2b_l295_295276


namespace min_links_to_measure_all_weights_l295_295957

theorem min_links_to_measure_all_weights (n : ℕ) (h1 : n = 150) :
  ∃ k, k = 4 ∧ ∀ w : ℕ, 1 ≤ w ∧ w ≤ n → ∃ (parts : list ℕ), (list.sum parts = w ∧ ∀ p ∈ parts, p = 1) :=
by
  use 4
  split
  exact rfl
  intros w hw
  cases hw with hw1 hw2
  exact sorry

end min_links_to_measure_all_weights_l295_295957


namespace june_rides_to_bernard_l295_295088

-- Define conditions
def distance_june_julia : ℝ := 1.5
def time_june_julia : ℝ := 6
def break_time : ℝ := 5
def distance_june_bernard : ℝ := 4.2

-- Calculate the biking rate
def biking_rate : ℝ := distance_june_julia / time_june_julia

-- Time to bike to Bernard's house
def time_to_bernard : ℝ := distance_june_bernard / biking_rate

-- The total time including the break
def total_time : ℝ := if distance_june_bernard > distance_june_julia then time_to_bernard + break_time else time_to_bernard

-- Main theorem statement
theorem june_rides_to_bernard : total_time = 21.8 := by
  sorry

end june_rides_to_bernard_l295_295088


namespace tyre_flattening_time_l295_295282

theorem tyre_flattening_time (R1 R2 : ℝ) (hR1 : R1 = 1 / 9) (hR2 : R2 = 1 / 6) : 
  1 / (R1 + R2) = 3.6 :=
by 
  sorry

end tyre_flattening_time_l295_295282


namespace probability_unequal_gender_l295_295534

theorem probability_unequal_gender (n : ℕ) (h_n : n = 12) :
  let p := 1 / 2 in
  (∑ k in Finset.range (n + 1), (Nat.choose n k) * p ^ k * (1 - p) ^ (n - k)) - (Nat.choose n (n / 2)) * p ^ (n / 2) * (1 - p) ^ (n / 2) = 793 / 1024 :=
by
  sorry

end probability_unequal_gender_l295_295534


namespace isosceles_triangle_perimeter_l295_295388

-- Definitions of conditions
def is_isosceles (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ (x = y ∨ y = 1)

def valid_equation (x y : ℝ) : Prop :=
  y = real.sqrt (2 - x) + real.sqrt (3 * x - 6) + 3

-- Theorem statement encapsulating the problem
theorem isosceles_triangle_perimeter (x y : ℝ) (h_iso: is_isosceles x y) (h_eq : valid_equation x y) :
  ∃ P : ℝ, P = 7 ∨ P = 8 :=
sorry

end isosceles_triangle_perimeter_l295_295388


namespace max_tiles_accommodated_l295_295937

/-- 
The rectangular tiles, each of size 40 cm by 28 cm, must be laid horizontally on a rectangular floor
of size 280 cm by 240 cm, such that the tiles do not overlap, and they are placed in an alternating
checkerboard pattern with edges jutting against each other on all edges. A tile can be placed in any
orientation so long as its edges are parallel to the edges of the floor, and it follows the required
checkerboard pattern. No tile should overshoot any edge of the floor. Determine the maximum number 
of tiles that can be accommodated on the floor while adhering to the placement pattern.
-/
theorem max_tiles_accommodated (tile_len tile_wid floor_len floor_wid : ℕ)
  (h_tile_len : tile_len = 40)
  (h_tile_wid : tile_wid = 28)
  (h_floor_len : floor_len = 280)
  (h_floor_wid : floor_wid = 240) :
  tile_len * tile_wid * 12 ≤ floor_len * floor_wid :=
by 
  sorry

end max_tiles_accommodated_l295_295937
