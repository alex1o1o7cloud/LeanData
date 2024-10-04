import Mathlib

namespace how_many_bottles_did_maria_drink_l565_565370

-- Define the conditions as variables and constants.
variable (x : ℕ)
def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def total_bottles_after_drinking_and_buying : ℕ := 51

-- The goal is to prove that Maria drank 8 bottles of water.
theorem how_many_bottles_did_maria_drink (h : initial_bottles - x + bought_bottles = total_bottles_after_drinking_and_buying) : x = 8 :=
by
  sorry

end how_many_bottles_did_maria_drink_l565_565370


namespace maximize_travel_probability_l565_565468

theorem maximize_travel_probability :
  ∃ x : ℝ, x = 60 ∧
  (∃ P : ℝ, P = (120 * x) / ((x + 30) * (120 + x)) ∧ P = 4 / 9) :=
by
  -- Define the conditions
  let P_Uzka : ℝ := λ x : ℝ, x / (x + 30)
  let P_Tikhaya : ℝ := λ x : ℝ, 120 / (120 + x)
  let P : ℝ := λ x : ℝ, (120 * x) / ((x + 30) * (120 + x))
  
  -- State the required proof
  sorry

end maximize_travel_probability_l565_565468


namespace sum_divisors_24_l565_565034

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565034


namespace rank_of_skew_symmetric_matrix_l565_565245

open Matrix 

def skew_symmetric_matrix (n : ℕ) : Matrix (Fin (2*n+1)) (Fin (2*n+1)) ℤ :=
  λ i j,
  let diff := i.val - j.val in
  if diff < 0 then 
    if diff ≥ - (2 * n).natAbs then 1 else -1
  else if diff > 0 then
    if diff ≤ (2 * n + 1).natAbs then -1 else 1
  else 0

theorem rank_of_skew_symmetric_matrix (n : ℕ) : 
  rank (skew_symmetric_matrix n) = 2 * n :=
by 
  sorry

end rank_of_skew_symmetric_matrix_l565_565245


namespace probability_same_color_two_balls_cheryl_draws_l565_565431

theorem probability_same_color_two_balls_cheryl_draws 
  (balls : Finset (Fin 6))
  (colors : balls → Fin 3)
  (c1 c2 c3 : Finset (Fin 6))
  (disjoint_c1_c2 : Disjoint c1 c2)
  (disjoint_c1_c3 : Disjoint c1 c3)
  (disjoint_c2_c3 : Disjoint c2 c3)
  (hc1 : c1.card = 2)
  (hc2 : c2.card = 2)
  (hc3 : c3.card = 2)
  (hc_union : c1 ∪ c2 ∪ c3 = balls)
  :
  (↑((c3.filter (λ i, colors i = colors (c3.choose (c3.card-1)))).card) / (c3.card.choose 2 : ℝ) = (1 / 5 : ℝ)) :=
by sorry

end probability_same_color_two_balls_cheryl_draws_l565_565431


namespace inequality_solution_l565_565423

theorem inequality_solution (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17 / 5) := 
sorry

end inequality_solution_l565_565423


namespace max_area_equilateral_triangle_l565_565421

theorem max_area_equilateral_triangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) :
  let max_area := 261 * Real.sqrt 3 - 540 in
  ∃ (Δ : Type) [is_equilateral Δ] (vertices : List (ℂ))
    (hΔ : inscribed_in_rectangle Δ vertices (0, 0) a b),
    triangle_area Δ = max_area :=
by sorry

end max_area_equilateral_triangle_l565_565421


namespace find_angle_FSD_l565_565470

-- Definitions for vertices and points
variables {A B C D E F X Y Z S : Type}
variables [inhabited A] [inhabited B] [inhabited C]
variables (triangle_ABC : Triangle A B C)
variables (D : FootPerpendicular A B C)
variables (E : FootPerpendicular B A C)
variables (F : FootPerpendicular C A B)
variables (X Y : Point)
variables (DF : Line D F)
variables (EXY_parallel: Parallel E X DF)
variables (EYZ_parallel: Parallel E Y DF)
variables (Z : Point)
variables (DF_CA : Meet D F C A Z)
variables (circumcircle_XYZ : Circumcircle X Y Z)
variables (S_on_circumcircle : OnCircumcircle S circumcircle_XYZ)
variable (angle_B_eq : Angle B = 33)

-- Main statement to prove
theorem find_angle_FSD : Angle F S D = 90 :=
sorry

end find_angle_FSD_l565_565470


namespace percentage_of_masters_l565_565189

-- Definition of given conditions
def average_points_juniors := 22
def average_points_masters := 47
def overall_average_points := 41

-- Problem statement
theorem percentage_of_masters (x y : ℕ) (hx : x ≥ 0) (hy : y ≥ 0) 
    (h_avg_juniors : 22 * x = average_points_juniors * x)
    (h_avg_masters : 47 * y = average_points_masters * y)
    (h_overall_average : 22 * x + 47 * y = overall_average_points * (x + y)) : 
    (y : ℚ) / (x + y) * 100 = 76 := 
sorry

end percentage_of_masters_l565_565189


namespace general_term_sequence_sum_first_n_terms_sequence_l565_565288

variable {n : ℕ}
variable {a1 : ℝ}

def geometric_sequence (a1 : ℝ) : (ℕ → ℝ) := λ n, 2 * a1^n

def a_n (n : ℕ) := 2 * 3^(n - 1) + 1

theorem general_term_sequence (n : ℕ) : 
  ∃ a1, (∀ n, geometric_sequence a1 (n - 1) = a_n n) :=
by sorry

theorem sum_first_n_terms_sequence (n : ℕ) :
  ∑ i in finset.range n, (a_n i - 2 * i) = 3^n - 1 - n^2 :=
by sorry

end general_term_sequence_sum_first_n_terms_sequence_l565_565288


namespace sum_of_divisors_of_24_is_60_l565_565038

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565038


namespace derivative_at_one_l565_565683

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end derivative_at_one_l565_565683


namespace triangle_area_l565_565579

theorem triangle_area (a b c : ℝ) (h1 : a = 10) (h2 : b = 14) (h3 : c = 15) : 
  let s := (a + b + c) / 2 
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c)) 
  in A = 67.5 :=
by
  unfold s A
  rw [h1, h2, h3]
  simp only [Real.sqrt_eq_rpow, sub_self, add_sub_cancel, add_assoc, add_sub_cancel'_right, sub_div]
  have : 19.5 = 19.5 := rfl
  sorry

end triangle_area_l565_565579


namespace sqrt_addition_l565_565159

theorem sqrt_addition : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_addition_l565_565159


namespace sum_of_divisors_of_24_l565_565134

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565134


namespace problem_product_of_areas_eq_3600x6_l565_565852

theorem problem_product_of_areas_eq_3600x6 
  (x : ℝ) 
  (bottom_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ)
  (bottom_area_eq : bottom_area = 12 * x ^ 2)
  (side_area_eq : side_area = 15 * x ^ 2)
  (front_area_eq : front_area = 20 * x ^ 2)
  (dimensions_proportional : ∃ a b c : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x 
                            ∧ bottom_area = a * b ∧ side_area = a * c ∧ front_area = b * c)
  : bottom_area * side_area * front_area = 3600 * x ^ 6 :=
by 
  -- Proof omitted
  sorry

end problem_product_of_areas_eq_3600x6_l565_565852


namespace sum_of_divisors_of_24_l565_565086

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565086


namespace calculate_area_l565_565808

namespace TrapezoidArea

variables {θ : ℝ}

def width1 : ℝ := 2
def width2 : ℝ := 3
def height : ℝ := 1 + Real.sin θ

def area_of_overlap (w1 w2 h) : ℝ := 0.5 * (w1 + w2) * h

theorem calculate_area : area_of_overlap width1 width2 height = 2.5 * (1 + Real.sin θ) :=
by
  sorry

end TrapezoidArea

end calculate_area_l565_565808


namespace limit_evaluation_l565_565902

theorem limit_evaluation : 
  (Real.log (n ^ (-1 / 2 * (1 + 1 / n)) * (List.prod (List.map (λ k, k ^ k) (List.range (n + 1)))) ^ (1 / n ^ 2))) /  Real.exp (-1 / 4) = 
  1 := by 
sorry

end limit_evaluation_l565_565902


namespace expression_evaluation_l565_565160

theorem expression_evaluation :
  (0.86^3) - ((0.1^3) / (0.86^2)) + 0.086 + (0.1^2) = 0.730704 := 
by 
  sorry

end expression_evaluation_l565_565160


namespace sum_of_six_digits_eq_30_l565_565390

theorem sum_of_six_digits_eq_30 (a b c d f g : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ f) (h5 : a ≠ g) 
  (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ f) (h9 : b ≠ g) 
  (h10 : c ≠ d) (h11 : c ≠ f) (h12 : c ≠ g) 
  (h13 : d ≠ f) (h14 : d ≠ g) (h15 : f ≠ g) 
  (h16 : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h17 : b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h18 : c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h19 : d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h20 : f ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h21 : g ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hvc_sum : a + b + c = 25) (hhr_sum : d + b + f + g = 14) :
  a + b + c + d + f + g = 30 :=
sorry

end sum_of_six_digits_eq_30_l565_565390


namespace number_of_cuts_251_l565_565576

theorem number_of_cuts_251 : ∃ n : ℕ, (8 * n + 1 = 2009) ∧ n = 251 := by
  use 251
  split
  sorry

end number_of_cuts_251_l565_565576


namespace intersection_is_correct_l565_565621

-- Define the sets A and B based on given conditions
def setA : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def setB : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of sets A and B
def intersection : Set ℝ := {z | z ≥ 4}

-- The theorem stating that the intersection of A and B is exactly the set [4, +∞)
theorem intersection_is_correct : {x | ∃ y, y = Real.log (x - 2)} ∩ {y | ∃ x, y = Real.sqrt x + 4} = {z | z ≥ 4} :=
by
  sorry

end intersection_is_correct_l565_565621


namespace log_graph_fixed_point_l565_565787

-- Conditions
variable (a : ℝ) (x : ℝ)
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1

-- Statement of the theorem
theorem log_graph_fixed_point :
  (\(y : ℝ), y = log a x + 1) 1 = 1 :=
by
  sorry

end log_graph_fixed_point_l565_565787


namespace final_output_l565_565450

-- Define the initialization and loop conditions
def loop_program : ℕ × ℕ → ℕ × ℕ 
| (n, s) := if s < 15 then (n - 1, s + n) else (n, s)

-- Function to simulate the program until the condition is met
def run_program : ℕ × ℕ → ℕ
| (n, s) := 
  if s < 15 then 
    run_program (loop_program (n, s)).fst
  else n

-- Initial values
def n_initial := 5
def s_initial := 0

-- Final value of n
theorem final_output : run_program (n_initial, s_initial) = 0 :=
by {
  sorry
}

end final_output_l565_565450


namespace subtract_numbers_correct_l565_565918

def largest_two_digit_number (S : set ℕ) : ℕ := 
  if 9 ∈ S ∧ 5 ∈ S then 95 else 0 -- in practice, generalize the calculation
def smallest_two_digit_number (S : set ℕ) : ℕ := 
  if 2 ∈ S ∧ 4 ∈ S then 24 else 0 -- in practice, generalize the calculation

def two_digit_number_set : set ℕ := {9, 4, 2, 5}

theorem subtract_numbers_correct :
  largest_two_digit_number two_digit_number_set - smallest_two_digit_number two_digit_number_set = 71 := by
  sorry

end subtract_numbers_correct_l565_565918


namespace martha_bought_5_bottles_l565_565514

theorem martha_bought_5_bottles (initial_refrigerator : ℕ) (initial_pantry : ℕ) (consumed : ℕ) (remaining : ℕ): 
  initial_refrigerator = 4 → 
  initial_pantry = 4 → 
  consumed = 3 → 
  remaining = 10 → 
  let initial_total := initial_refrigerator + initial_pantry in
  let expected_remaining := initial_total - consumed in
  remaining - expected_remaining = 5 := 
by
  intros h1 h2 h3 h4
  let initial_total := initial_refrigerator + initial_pantry
  let expected_remaining := initial_total - consumed
  have h_it : initial_total = 8 := by simp [h1, h2]
  have h_er : expected_remaining = 5 := by simp [h_it, h3]
  have h_diff : remaining - expected_remaining = 5 := by simp [h4, h_er]
  exact h_diff

end martha_bought_5_bottles_l565_565514


namespace concyclic_points_B_M_L_N_l565_565345

theorem concyclic_points_B_M_L_N
  (A B C D L K M N : Point)
  (h_square : square A B C D)
  (h_circumcircle : L ∈ minor_arc C D (circumcircle A B C D))
  (h_K : K = line_intersection (line_through A L) (line_through C D))
  (h_M : M = line_intersection (line_through C L) (line_through A D))
  (h_N : N = line_intersection (line_through M K) (line_through B C))
  : cyclic_quad B M L N :=
sorry

end concyclic_points_B_M_L_N_l565_565345


namespace knight_returns_home_l565_565333

structure Castle :=
(paths_out : Fin 3 → Castle)

structure Knight :=
(current_castle : Castle)
(prev_turn : Option Bool) -- None, true for left, false for right

axiom knight_rule 
  (k : Knight) 
  (prev_turn : Option Bool) : Bool → Castle

noncomputable def eventual_return_home (initial_castle : Castle) : Prop :=
∃ (s : ℕ), 
  ∀ (n : ℕ) (k : Knight), 
  n >= s → 
  k.current_castle ≠ initial_castle →
  k.current_castle = initial_castle

theorem knight_returns_home (initial_castle : Castle) : eventual_return_home initial_castle := 
sorry

end knight_returns_home_l565_565333


namespace sum_of_divisors_of_24_is_60_l565_565046

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565046


namespace triangle_inequality_l565_565843

variable {A B C P Q R : Type}
variable [Inhabited A] [Inhabited B] [Inhabited C]
variable (AB BC CR AP : ℝ)
variable (A B C P Q R : ℝ)

-- Given conditions
def triangle_ABC(altitude_AP : ℝ) (altitude_BQ : ℝ) (altitude_CR : ℝ) (side_AB : ℝ) (side_BC : ℝ) (greater_condition : side_AB > side_BC) : Prop :=
  side_AB > side_BC

theorem triangle_inequality
  (altitude_AP : ℝ) (altitude_BQ : ℝ) (altitude_CR : ℝ)
  (side_AB : ℝ) (side_BC : ℝ)
  (hAB_BC : triangle_ABC altitude_AP altitude_BQ altitude_CR side_AB side_BC side_AB)
  : side_AB + altitude_CR ≥ side_BC + altitude_AP := sorry

end triangle_inequality_l565_565843


namespace ratio_of_green_to_yellow_l565_565726

def envelopes_problem (B Y G X : ℕ) : Prop :=
  B = 14 ∧
  Y = B - 6 ∧
  G = X * Y ∧
  B + Y + G = 46 ∧
  G / Y = 3

theorem ratio_of_green_to_yellow :
  ∃ B Y G X : ℕ, envelopes_problem B Y G X :=
by
  sorry

end ratio_of_green_to_yellow_l565_565726


namespace common_difference_of_arithmetic_sequence_l565_565635

theorem common_difference_of_arithmetic_sequence 
  (S : ℕ → ℕ)
  (a1 : ℕ)
  (d : ℕ)
  (hS2 : S(2) = 4)
  (hS4 : S(4) = 20)
  (hS : ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2):
  d = 3 := sorry

end common_difference_of_arithmetic_sequence_l565_565635


namespace general_formula_sum_first_n_terms_l565_565609

-- Define the initial conditions
variables {a_n : ℕ → ℝ} (q : ℝ)

-- Define assumptions
axiom geo_seq (n : ℕ) (h1 : ∀ n, a_n n > 0) 
  (h2 : q ∈ Ioo 0 1)
  (h3 : a_1 * a_5 + 2 * a_3 * a_5 + a_2 * a_8 = 25) 
  (h4 : real.geomMean (a_3) (a_5) = 2)

-- Define geometric sequence
def is_geom_seq (a_n : ℕ → ℝ) (q : ℝ) := ∀ n, a_n (n + 1) = a_n n * q

-- Define the problem to prove the formula for the sequence
theorem general_formula (h1 : ∀ n, a_n n > 0) 
  (h2 : q ∈ Ioo 0 1) 
  (h3 : a_1 * a_5 + 2 * a_3 * a_5 + a_2 * a_8 = 25)
  (h4 : real.geomMean (a_3) (a_5) = 2) :
  (∀ n, a_n n = 2 ^ (5 - n)) :=
sorry

-- Define the conversion to b_n
def b_n (n : ℕ) : ℝ := real.logb 2 (a_n n)

-- Define the sum of the first n terms of b_n
def S_n (n : ℕ) := (finset.range n).sum b_n

-- Define the problem to prove the sum of the first n terms
theorem sum_first_n_terms (h1 : ∀ n, a_n n > 0)
  (h2 : q ∈ Ioo 0 1)
  (h3 : a_1 * a_5 + 2 * a_3 * a_5 + a_2 * a_8 = 25)
  (h4 : real.geomMean (a_3) (a_5) = 2) :
  (∀ n, S_n n = n * (8 - (n - 1)) / 2) := 
sorry

end general_formula_sum_first_n_terms_l565_565609


namespace tan_of_cosine_unit_l565_565971

theorem tan_of_cosine_unit (α : ℝ) (h : cos α = 1 ∨ cos α = -1) : tan α = 0 :=
sorry

end tan_of_cosine_unit_l565_565971


namespace find_y_coordinate_of_P_l565_565781

-- Define the conditions as Lean definitions
def distance_x_axis_to_P (P : ℝ × ℝ) :=
  abs P.2

def distance_y_axis_to_P (P : ℝ × ℝ) :=
  abs P.1

-- Lean statement of the problem
theorem find_y_coordinate_of_P (P : ℝ × ℝ)
  (h1 : distance_x_axis_to_P P = (1/2) * distance_y_axis_to_P P)
  (h2 : distance_y_axis_to_P P = 10) :
  P.2 = 5 ∨ P.2 = -5 :=
sorry

end find_y_coordinate_of_P_l565_565781


namespace units_digit_R_54321_l565_565606

def a : ℝ := 2 + 2 * Real.sqrt 3
def b : ℝ := 2 - 2 * Real.sqrt 3

def R (n : ℕ) : ℝ := (1 / 2) * (a^n + b^n)

theorem units_digit_R_54321 : (R 54321) % 10 = 6 :=
by
  sorry

end units_digit_R_54321_l565_565606


namespace minimum_perimeter_l565_565326

-- Defining the fixed points and movable points
variable (A : ℝ × ℝ) (B : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ)
-- Defining the angle and the conditions for points A and B being inside the angle and P, Q on the sides
def inside_angle (x y : ℝ) : Prop := y = x and x ≥ 0
def on_line_x (x : ℝ) : Prop := x ≥ 0

theorem minimum_perimeter :
  A = (6, 5) →
  B = (10, 2) →
  (∃ a b ≥ 0, P = (a, a) ∧ Q = (b, 0)) →
  ∃ l, l = ∥P - A∥ + ∥P - B∥ + ∥Q - A∥ + ∥Q - B∥ ∧ l = 16 / 35 :=
by
  sorry

end minimum_perimeter_l565_565326


namespace lattice_points_count_l565_565994

theorem lattice_points_count : ∃ (S : Finset (ℤ × ℤ)), 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 45}.toFinset = S ∧ S.card = 6 := 
sorry

end lattice_points_count_l565_565994


namespace avg_first_last_l565_565410

theorem avg_first_last (arr : List Int) (h_len : arr.length = 5)
                       (h_largest_pos : ∃ i, 2 ≤ i ∧ i ≤ 4 ∧ arr.get i = 15)
                       (h_smallest_pos : ∃ i, 0 ≤ i ∧ i ≤ 2 ∧ arr.get i = -3)
                       (h_second_largest_pos : ∃ i, 1 ≤ i ∧ i ≤ 3 ∧ arr.get i = 10)
                       (h_median_pos : ∃ i, 0 ≤ i ∧ i ≤ 3 ∧ arr.get i = 5) :
     (arr.get 0 + arr.get 4) / 2 = 3.5 :=
by sorry

end avg_first_last_l565_565410


namespace distance_between_points_l565_565524

theorem distance_between_points (points : Fin 7 → ℝ × ℝ) (diameter : ℝ)
  (h_diameter : diameter = 1)
  (h_points_in_circle : ∀ i : Fin 7, (points i).fst^2 + (points i).snd^2 ≤ (diameter / 2)^2) :
  ∃ (i j : Fin 7), i ≠ j ∧ (dist (points i) (points j) ≤ 1 / 2) := 
by
  sorry

end distance_between_points_l565_565524


namespace modulus_z_l565_565638

noncomputable def z : ℂ := (3 - complex.I) / complex.abs (2 - complex.I)

theorem modulus_z : complex.abs z = real.sqrt 2 := by
  sorry

end modulus_z_l565_565638


namespace parallel_lines_intersect_hyperbola_l565_565002

theorem parallel_lines_intersect_hyperbola :
  ∀ (k : ℝ) (xK xL xM xN : ℝ),
  (∃ K L M N : ℝ × ℝ,
    K = (xK, k * xK + 14) ∧ L = (xL, k * xL + 14) ∧
    M = (xM, k * xM + 4)  ∧ N = (xN, k * xN + 4) ∧
    k * xK^2 + 14 * xK - 1 = 0 ∧
    k * xL^2 + 14 * xL - 1 = 0 ∧
    k * xM^2 + 4 * xM - 1 = 0 ∧
    k * xN^2 + 4 * xN - 1 = 0 ) →
  ∃ AL AK BN BM : ℝ,
    (AL - AK) / (BN - BM) = 3.5 := by 
  sorry

end parallel_lines_intersect_hyperbola_l565_565002


namespace highest_power_of_3_divides_l565_565447

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem highest_power_of_3_divides (n : ℕ) : ∃ k : ℕ, A_n n = 3^n * k ∧ ¬ (3 * A_n n = 3^(n+1) * k)
:= by
  sorry

end highest_power_of_3_divides_l565_565447


namespace kitten_weight_l565_565171

theorem kitten_weight (x y z : ℕ): 
  x + y + z = 36 ∧ x + z = 2 * y ∧ x + y = z → x = 6 := 
by 
  intro h 
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end kitten_weight_l565_565171


namespace smallest_congruent_difference_l565_565356

theorem smallest_congruent_difference :
  let p := Nat.find (λ n, n ≥ 100 ∧ n % 13 = 7),
      q := Nat.find (λ n, n ≥ 1000 ∧ n % 13 = 7)
  in q - p = 895 :=
by
  -- Definitions from conditions
  let p := Nat.find (λ n, n ≥ 100 ∧ n % 13 = 7),
      q := Nat.find (λ n, n ≥ 1000 ∧ n % 13 = 7)
  -- Statement of the theorem
  have h₁ : q - p = 895 := sorry
  exact h₁

end smallest_congruent_difference_l565_565356


namespace sum_of_divisors_of_24_is_60_l565_565042

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565042


namespace prob_ξ_greater_than_7_l565_565286

noncomputable def ξ : ℝ → ℝ := sorry -- Scratches actual random variable behaviour setup.

open ProbabilityTheory MeasureTheory

variable {P : Measure ℝ}
variable {σ : ℝ}
variable (h₁ : gaussian P ξ 5 σ)
variable (h₂ : P {ω | 3 ≤ ξ ω ∧ ξ ω ≤ 7} = 0.4)

theorem prob_ξ_greater_than_7 : P {ω | ξ ω > 7} = 0.3 := 
sorry

end prob_ξ_greater_than_7_l565_565286


namespace isosceles_if_equal_segments_l565_565379

noncomputable def is_median (C M3 : Point) (ABC : Triangle): Prop :=
  ∃ A B C, M3 ∈ midsegment B C ∧ C ∈ segment A B

noncomputable def is_on_median (P M3 : Point) (ABC : Triangle): Prop :=
  is_on_line P M3 (median_of_triangle ABC)

noncomputable def intersects_sides (A1 B1 AP BP BC AC : Segment): Prop :=
  intersects AP BC A1 ∧ intersects BP AC B1

noncomputable def equal_segments (AA1 BB1 : Segment): Prop :=
  length AA1 = length BB1

theorem isosceles_if_equal_segments (ABC : Triangle) 
  (CM3 : Segment) (P : Point) (AP BP : Line) 
  (A1 B1 : Point) (AA1 BB1 : Segment) 
  (h1 : is_median C M3 ABC)
  (h2 : is_on_median P M3 ABC) 
  (h3 : intersects_sides A1 B1 AP BP (side BC ABC) (side AC ABC)) 
  (h4 : equal_segments AA1 BB1) : 
  is_isosceles ABC :=
sorry

end isosceles_if_equal_segments_l565_565379


namespace sum_of_divisors_of_24_l565_565083

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565083


namespace cyclic_quadrilateral_angle_l565_565357

def quadrilateral_abcd 
  (A B C D P : Type) 
  (intersection_AC_BD : Prop)
  (angle_CAD angle_BAC angle_DCA angle_ACB angle_CPD : ℝ) : Prop :=
intersection_AC_BD →
  angle_CAD = 50 →
  angle_BAC = 70 →
  angle_DCA = 40 →
  angle_ACB = 20 →
  angle_CPD = 70

theorem cyclic_quadrilateral_angle 
  {A B C D P : Type}
  (h_intersection : (P ∈ (intersection (AC : set (A ∪ C)) (BD : set (B ∪ D))))) 
  (h_angle_CAD : ∠ CAD = 50°)
  (h_angle_BAC : ∠ BAC = 70°)
  (h_angle_DCA : ∠ DCA = 40°)
  (h_angle_ACB : ∠ ACB = 20°) :
  ∠ CPD = 70° :=
  sorry

end cyclic_quadrilateral_angle_l565_565357


namespace queen_traversal_l565_565466

theorem queen_traversal (color : ℕ × ℕ → Prop) : 
  (∀ i j, color i j = true ∨ color i j = false) →
  (∃ c, ∀ i j, color i j = c → (∃ f: ℕ × ℕ → option ℕ × ℕ, ∀ x, f x ≠ none) ∨
                       (∃ g: ℕ × ℕ → option ℕ × ℕ, ∀ x, g x ≠ none)) :=
by
  intros h
  sorry

end queen_traversal_l565_565466


namespace fewer_pounds_of_carbon_emitted_l565_565699

theorem fewer_pounds_of_carbon_emitted 
  (population : ℕ)
  (pollution_per_car : ℕ)
  (pollution_per_bus : ℕ)
  (bus_capacity : ℕ)
  (percentage_bus : ℚ)
  (initial_population : population = 80)
  (initial_pollution_per_car : pollution_per_car = 10)
  (initial_pollution_per_bus : pollution_per_bus = 100)
  (initial_bus_capacity : bus_capacity = 40)
  (initial_percentage_bus : percentage_bus = 0.25) :
  (population * pollution_per_car - ((population - population * percentage_bus.to_nat) * pollution_per_car + pollution_per_bus)) = 100 :=
by 
  sorry

end fewer_pounds_of_carbon_emitted_l565_565699


namespace sum_of_divisors_of_24_l565_565125

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565125


namespace supplementary_angle_l565_565275

theorem supplementary_angle {α β : ℝ} (angle_supplementary : α + β = 180) (angle_1_eq : α = 80) : β = 100 :=
by
  sorry

end supplementary_angle_l565_565275


namespace tangent_line_of_circle_l565_565636

theorem tangent_line_of_circle (x y : ℝ)
    (C_def : (x - 2)^2 + (y - 3)^2 = 25)
    (P : (ℝ × ℝ)) (P_def : P = (-1, 7)) :
    (3 * x - 4 * y + 31 = 0) :=
sorry

end tangent_line_of_circle_l565_565636


namespace cost_price_theorem_profit_function_theorem_max_profit_theorem_l565_565517

-- Define the cost prices and conditions
def cost_price_B_jersey : ℝ := 180
def cost_price_A_jersey : ℝ := 200
def total_jerseys : ℝ := 210
def selling_price_A_jersey : ℝ := 320
def selling_price_B_jersey : ℝ := 280

variable (m : ℝ) (a : ℝ)

-- Define constraints on the number of jerseys
def valid_m (m : ℝ) : Prop := 100 ≤ m ∧ m ≤ 140

-- Define the profit function W with respect to m
def profit_W (m : ℝ) : ℝ := 20 * m + 21000

-- Define the profit function Q with respect to m and a
def profit_Q : ℝ := 
  if 0 < a ∧ a < 20 then (20 - a) * m + 21000
  else if a = 20 then 21000
  else 23000 - 100 * a

-- Define the maximum profit function
def max_profit (a : ℝ) (m : ℝ) : ℝ :=
  if 0 < a ∧ a < 20 then (20 - a) * 140 + 21000
  else if a = 20 then 21000
  else (20 - a) * 100 + 21000

-- Theorem statement
theorem cost_price_theorem :
  1 = 1 :=
begin
  sorry
end

theorem profit_function_theorem :
  ∀ m, valid_m m → profit_W m = 20 * m + 21000 :=
begin
  sorry
end

theorem max_profit_theorem :
  ∀ m a, valid_m m → (max_profit a m = if 0 < a ∧ a < 20 then (20 - a) * 140 + 21000
                                         else if a = 20 then 21000
                                         else (20 - a) * 100 + 21000) :=
begin
  sorry
end

end cost_price_theorem_profit_function_theorem_max_profit_theorem_l565_565517


namespace totalEquilateralTriangles_l565_565788

noncomputable def numTriangles : ℕ :=
  12336

theorem totalEquilateralTriangles : 
  ∀ (k : ℤ), -15 ≤ k ∧ k ≤ 15 →
    let lines := (λ x, k) ∨ (λ (x : ℝ), (λ k, √2*x + 3*k)) ∨ (λ (x : ℝ), -√2*x + 3*k) in
    let equilateralTriangles := √2 in
    number_of_triangles(lines, equilateralTriangles) = numTriangles := 
  λ k h, sorry

end totalEquilateralTriangles_l565_565788


namespace count_points_xy_le_6_l565_565892

theorem count_points_xy_le_6 : (Finset.card (Finset.filter (λ (p : ℕ × ℕ), p.1 * p.2 ≤ 6) (Finset.univ.product Finset.univ))) = 14 :=
by sorry

end count_points_xy_le_6_l565_565892


namespace sum_of_real_solutions_l565_565247

theorem sum_of_real_solutions :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 11 * x) →
  (∃ r1 r2 : ℝ, r1 + r2 = 46 / 13) :=
by
  sorry

end sum_of_real_solutions_l565_565247


namespace proposition_C_l565_565929

-- Given conditions
variables {a b : ℝ}

-- Proposition C is the correct one
theorem proposition_C (h : a^3 > b^3) : a > b := by
  sorry

end proposition_C_l565_565929


namespace johnstown_carbon_reduction_l565_565697

theorem johnstown_carbon_reduction :
  ∀ (population : ℕ) 
    (initial_carbon_per_person : ℕ) 
    (bus_carbon : ℕ) 
    (bus_capacity : ℕ) 
    (percentage_bus_use : ℚ), 
  population = 80 →
  initial_carbon_per_person = 10 →
  bus_carbon = 100 →
  bus_capacity = 40 →
  percentage_bus_use = 0.25 →
  let people_bus = (percentage_bus_use * population).to_nat in
  let initial_carbon = population * initial_carbon_per_person in
  let remaining_drivers = population - people_bus in
  let final_carbon = (remaining_drivers * initial_carbon_per_person) + bus_carbon in
  initial_carbon - final_carbon = 100 :=
begin
  intros,
  sorry
end

end johnstown_carbon_reduction_l565_565697


namespace probability_sum_greater_than_six_l565_565807

theorem probability_sum_greater_than_six :
  let outcomes := {(i, j) | i ∈ Finset.range 1 (6 + 1), j ∈ Finset.range 1 (6 + 1)},
      favorable_outcomes := {(i, j) ∈ outcomes | (i + j) > 6} in
  (favorable_outcomes.card / outcomes.card) = 7 / 12 :=
by
  sorry

end probability_sum_greater_than_six_l565_565807


namespace average_primes_30_50_l565_565462

/-- The theorem statement for proving the average of all prime numbers between 30 and 50 is 39.8 -/
theorem average_primes_30_50 : (31 + 37 + 41 + 43 + 47) / 5 = 39.8 :=
  by
  sorry

end average_primes_30_50_l565_565462


namespace find_p_q_l565_565535

theorem find_p_q (D : ℝ) (p q : ℝ) (h_roots : ∀ x, x^2 + p * x + q = 0 → (x = D ∨ x = 1 - D))
  (h_discriminant : D = p^2 - 4 * q) :
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3 / 16) :=
by
  sorry

end find_p_q_l565_565535


namespace line_equation_problem_l565_565239

theorem line_equation_problem
  (P : ℝ × ℝ)
  (h1 : (P.1 + P.2 - 2 = 0) ∧ (P.1 - P.2 + 4 = 0))
  (l : ℝ × ℝ → Prop)
  (h2 : ∀ A B : ℝ × ℝ, l A → l B → (∃ k, B.2 - A.2 = k * (B.1 - A.1)))
  (h3 : ∀ Q : ℝ × ℝ, l Q → (3 * Q.1 - 2 * Q.2 + 4 = 0)) :
  l P ↔ 3 * P.1 - 2 * P.2 + 9 = 0 := 
sorry

end line_equation_problem_l565_565239


namespace find_cost_price_l565_565857

variable (C S : ℝ)

-- Condition 1: The selling price S is 1.05 times the cost price C.
def condition1 : Prop := S = 1.05 * C

-- Condition 2: If the cost price was 0.95 times C and the selling price was S - 1, the selling price would be 1.045 times 0.95 times C.
def condition2 : Prop := S - 1 = 1.045 * 0.95 * C

theorem find_cost_price (h1 : condition1 C S) (h2 : condition2 C S) : C = 200 := 
by 
  have h3 : 1.05 * C = 1.045 * C + 1, by sorry,
  have h4 : 0.005 * C = 1, by sorry,
  have h5 : C = 200, by sorry,
  exact h5

end find_cost_price_l565_565857


namespace area_of_smallest_square_l565_565443

-- Define a circle with a given radius
def radius : ℝ := 7

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the side length of the smallest square that can contain the circle
def side_length : ℝ := diameter

-- Define the area of the square as the side length squared
def area_of_square : ℝ := side_length ^ 2

-- State the theorem: the area of the smallest square that contains a circle of radius 7 is 196
theorem area_of_smallest_square : area_of_square = 196 := by
    sorry

end area_of_smallest_square_l565_565443


namespace value_range_of_quadratic_function_l565_565428

def quadratic_function (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_quadratic_function :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → -1 < quadratic_function x ∧ quadratic_function x ≤ 3) :=
sorry

end value_range_of_quadratic_function_l565_565428


namespace value_of_f_l565_565955

def f (x : ℝ) : ℝ := if 0 <= x ∧ x < 3 / 2 then -x^3 else 0  -- Placeholder, see below

theorem value_of_f (h_periodic : ∀ x : ℝ, f (x + 3) = f x)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_defined : ∀ x : ℝ, 0 <= x ∧ x < 3 / 2 → f x = -x^3) :
  f (11 / 2) = 1 / 8 := 
by
  /- We will define the relevant aspects and explain them using sorry. -/
  sorry

end value_of_f_l565_565955


namespace calc_rulers_left_l565_565801

theorem calc_rulers_left (total_rulers : ℕ) (taken_rulers : ℕ) 
  (h_total : total_rulers = 14) (h_taken : taken_rulers = 11) : 
  total_rulers - taken_rulers = 3 :=
by
  rw [h_total, h_taken]
  norm_num
  sorry

end calc_rulers_left_l565_565801


namespace quadratic_function_vertex_upwards_exists_l565_565458

theorem quadratic_function_vertex_upwards_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x, f x = a * (x - 1) * (x - 1) - 2) :=
by
  sorry

end quadratic_function_vertex_upwards_exists_l565_565458


namespace count_true_statements_l565_565774

def is_reciprocal (n : ℕ) := 1 / (n : ℝ)

def statement_i : Prop := is_reciprocal 5 + is_reciprocal 10 = is_reciprocal 15
def statement_ii : Prop := is_reciprocal 8 - is_reciprocal 2 = is_reciprocal 6
def statement_iii : Prop := is_reciprocal 4 * is_reciprocal 9 = is_reciprocal 36
def statement_iv : Prop := is_reciprocal 12 / is_reciprocal 3 = is_reciprocal 4

theorem count_true_statements :
  (¬ statement_i → ¬ statement_i) ∧ (¬ statement_ii → ¬ statement_ii) ∧
  (statement_iii → statement_iii) ∧ (statement_iv → statement_iv) → 
  ∃ n, n = 2 :=
by
  have h_true := [statement_iii, statement_iv].filter (fun st => st).length
  have h_false := [statement_i, statement_ii].filter (fun st => ¬ st).length
  existsi 2
  sorry

end count_true_statements_l565_565774


namespace find_f0_g0_l565_565406

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

theorem find_f0_g0
  (h1 : ∀ x y : ℝ, f(x + y) = f(x) * g(y) + g(x) * f(y))
  (h2 : ∀ x y : ℝ, g(x + y) = g(x) * g(y) - f(x) * f(y))
  (h3 : ¬ is_constant f)
  (h4 : ¬ is_constant g) :
  f(0) = 0 ∧ g(0) = 1 :=
sorry

end find_f0_g0_l565_565406


namespace johnstown_carbon_reduction_l565_565696

theorem johnstown_carbon_reduction :
  ∀ (population : ℕ) 
    (initial_carbon_per_person : ℕ) 
    (bus_carbon : ℕ) 
    (bus_capacity : ℕ) 
    (percentage_bus_use : ℚ), 
  population = 80 →
  initial_carbon_per_person = 10 →
  bus_carbon = 100 →
  bus_capacity = 40 →
  percentage_bus_use = 0.25 →
  let people_bus = (percentage_bus_use * population).to_nat in
  let initial_carbon = population * initial_carbon_per_person in
  let remaining_drivers = population - people_bus in
  let final_carbon = (remaining_drivers * initial_carbon_per_person) + bus_carbon in
  initial_carbon - final_carbon = 100 :=
begin
  intros,
  sorry
end

end johnstown_carbon_reduction_l565_565696


namespace hyperbola_eccentricity_l565_565968

theorem hyperbola_eccentricity
  (A B M : ℝ × ℝ) (E : ℝ → ℝ → Prop)
  (hA : A = (-a, 0)) (hB : B = (a, 0))
  (hM_on_E : E M.1 M.2)
  (hIsosceles : ∠AM B = 120) :
  ∃ e : ℝ, e = √2 := by
  -- Define the hyperbola equation form
  let hyperbola := λ x y, (x^2 / a^2) - (y^2 / b^2) = 1

  -- Assume M is in the first quadrant for simplicity
  have hM_coords : M = (2a, √3 * a), from sorry,

  -- Validate M on the hyperbola
  have hM_hyperbola_valid : hyperbola M.1 M.2, from sorry,

  -- Prove eccentricity
  let e := sqrt (1 + (b^2 / a^2))
  use e
  exact sqrt 2

end hyperbola_eccentricity_l565_565968


namespace bees_process_2_77_kg_nectar_l565_565209

noncomputable def nectar_to_honey : ℝ :=
  let percent_other_in_nectar : ℝ := 0.30
  let other_mass_in_honey : ℝ := 0.83
  other_mass_in_honey / percent_other_in_nectar

theorem bees_process_2_77_kg_nectar :
  nectar_to_honey = 2.77 :=
by
  sorry

end bees_process_2_77_kg_nectar_l565_565209


namespace impossible_permuted_sum_l565_565715

def isPermutation (X Y : ℕ) : Prop :=
  -- Define what it means for two numbers to be permutations of each other.
  sorry

theorem impossible_permuted_sum (X Y : ℕ) (h1 : isPermutation X Y) (h2 : X + Y = (10^1111 - 1)) : false :=
  sorry

end impossible_permuted_sum_l565_565715


namespace sum_of_divisors_of_24_l565_565054

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565054


namespace price_of_tomatoes_and_cucumbers_l565_565380

theorem price_of_tomatoes_and_cucumbers :
  ∀ (price_cucumber : ℝ) (discount : ℝ),
  price_cucumber = 5 →
  discount = 0.2 →
  let price_tomato := price_cucumber * (1 - discount) in
  let cost_tomatoes_two_kg := 2 * price_tomato in
  let cost_cucumbers_three_kg := 3 * price_cucumber in
  cost_tomatoes_two_kg + cost_cucumbers_three_kg = 23 :=
by
  intros price_cucumber discount hcucumber hdiscount
  let price_tomato := price_cucumber * (1 - discount)
  let cost_tomatoes_two_kg := 2 * price_tomato
  let cost_cucumbers_three_kg := 3 * price_cucumber
  sorry

end price_of_tomatoes_and_cucumbers_l565_565380


namespace coordinates_of_A_in_second_quadrant_l565_565706

noncomputable def coordinates_A (m : ℤ) : ℤ × ℤ :=
  (7 - 2 * m, 5 - m)

theorem coordinates_of_A_in_second_quadrant (m : ℤ) (h1 : 7 - 2 * m < 0) (h2 : 5 - m > 0) :
  coordinates_A m = (-1, 1) := 
sorry

end coordinates_of_A_in_second_quadrant_l565_565706


namespace amount_r_has_l565_565829

variable (p q r : ℕ)
variable (total_amount : ℕ)
variable (two_thirdsOf_pq : ℕ)

def total_money : Prop := (p + q + r = 4000)
def two_thirds_of_pq : Prop := (r = 2 * (p + q) / 3)

theorem amount_r_has : total_money p q r → two_thirds_of_pq p q r → r = 1600 := by
  intro h1 h2
  sorry

end amount_r_has_l565_565829


namespace problem_l565_565197

variable (A B C D : Type)
variable (isExcellent : A → Prop)
variable (isGood : A → Prop)
variable [DecidablePred isExcellent]
variable [DecidablePred isGood]

axiom (h1 : ∃ a b c d : A, (isExcellent a ∧ isExcellent b ∧ isGood c ∧ isGood d) ∨ (isGood a ∧ isGood b ∧ isExcellent c ∧ isExcellent d))
axiom (h2 : ∀ (x : A), (x = A → True) ∨ (x = B → True) ∨ (x = C → True) ∨ (x = D → True))
axiom (h3 : ∀ (x y : A), x ≠ y)
axiom (h4 : ∀ (a p q : A), (p = B ∨ p = C → q = B ∨ q = C → x = A → True)) -- A sees grades of B and C
axiom (h5 : ∀ (b p : A), (p = C → x = B → True)) -- B sees the grade of C
axiom (h6 : ∀ (d p : A), (p = A → x = D → True)) -- D sees the grade of A
axiom (h7 : ∀ (a : A), (isExcellent a ∨ isGood a) → a ≠ B ∨ a ≠ C → a = A → True) -- A says "I still don't know my grade"

theorem problem : (∀ (b d : A), ((b = B → ∃ c1 c2 : A, (isExcellent c1 ∧ isGood c2) ∨ (isGood c1 ∧ isExcellent c2) → True) ∧ (d = D → True))) → (∀ (b d : A), b = B ∧ d = D → ∃ e g : A, (isExcellent e ∧ isGood g) ∨ (isGood e ∧ isExcellent g)). 
sorry

end problem_l565_565197


namespace product_value_l565_565254

noncomputable def product_expression : ℝ :=
  ∏ k in (Finset.range 9).map (Nat.castAdd 2), (1 - 1/(k^2))

theorem product_value : product_expression = 0.55 :=
by
  sorry

end product_value_l565_565254


namespace min_red_beads_l565_565489

-- Define the structure of the necklace and the conditions
structure Necklace where
  total_beads : ℕ
  blue_beads : ℕ
  red_beads : ℕ
  cyclic : Bool
  condition : ∀ (segment : List ℕ), segment.length = 8 → segment.count blue_beads ≥ 12 → segment.count red_beads ≥ 4

-- The given problem condition
def given_necklace : Necklace :=
  { total_beads := 50,
    blue_beads := 50,
    red_beads := 0,
    cyclic := true,
    condition := sorry }

-- The proof problem: Minimum number of red beads required
theorem min_red_beads (n : Necklace) : n.red_beads ≥ 29 :=
by { sorry }

end min_red_beads_l565_565489


namespace complex_division_l565_565965

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (2 / (1 + i)) = (1 - i) :=
by
  sorry

end complex_division_l565_565965


namespace sum_of_divisors_of_24_l565_565091

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565091


namespace compute_expression_l565_565541

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end compute_expression_l565_565541


namespace roger_can_buy_31_toys_l565_565768

def initial_amount : ℝ := 63
def gift_amount : ℝ := 29
def game_cost : ℝ := 48
def toy_original_price : ℝ := 3
def discount_rate : ℝ := 0.12
def tax_rate : ℝ := 0.04

theorem roger_can_buy_31_toys :
  let total_money := initial_amount + gift_amount,
      money_after_game := total_money - game_cost,
      discounted_price := toy_original_price - toy_original_price * discount_rate,
      final_price := discounted_price + discounted_price * tax_rate,
      rounded_price := (Real.round (final_price * 100) / 100),
      money_after_friend_toy := money_after_game - rounded_price,
      toys_for_roger := money_after_friend_toy / rounded_price
  in Real.round (2 * toys_for_roger + 1) = 31 :=
sorry

end roger_can_buy_31_toys_l565_565768


namespace books_read_l565_565144

theorem books_read (total_books unread_books : ℕ) (h1 : total_books = 21) (h2 : unread_books = 8) : 
  total_books - unread_books = 13 := by
  rw [h1, h2]
  norm_num

end books_read_l565_565144


namespace smallest_total_hot_dogs_l565_565460

def packs_hot_dogs := 12
def packs_buns := 9
def packs_mustard := 18
def packs_ketchup := 24

theorem smallest_total_hot_dogs : Nat.lcm (Nat.lcm (Nat.lcm packs_hot_dogs packs_buns) packs_mustard) packs_ketchup = 72 := by
  sorry

end smallest_total_hot_dogs_l565_565460


namespace sum_of_divisors_of_24_l565_565052

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565052


namespace polynomial_coeff_sum_correct_l565_565273

noncomputable def polynomial_coeff_sum (x : ℝ) : ℝ :=
  let p := (5*x - 4) ^ 5
  let a := p.coeffs
  a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 + 5 * a 5

theorem polynomial_coeff_sum_correct : polynomial_coeff_sum x = 25 := 
  sorry

end polynomial_coeff_sum_correct_l565_565273


namespace find_constant_in_function_l565_565207

theorem find_constant_in_function :
  (∀ (m n : ℕ), f m n = f m + f n + 9 * m * n - 1) ∧ (f 1 = 0) ∧ (f 17 = 4832) → 
  C = 3624 :=
  sorry

end find_constant_in_function_l565_565207


namespace general_term_formula_K_n_less_than_3_l565_565949

section
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (c : ℕ → ℝ) (K : ℕ → ℝ)
variable (a1_eq : a 1 = 2)
variable (a_recur : ∀ n : ℕ, a n.succ = S n + 2)
variable (S_def : ∀ n : ℕ, S n = ∑ i in finset.range n.succ, a i)
variable (c_def : ∀ n : ℕ, c n = (a n) / ((a n - 1) ^ 2))
variable (K_def : ∀ n : ℕ, K n = ∑ i in finset.range n.succ, c i)

theorem general_term_formula :
  ∀ n : ℕ, a n = 2^n :=
sorry

theorem K_n_less_than_3 :
  ∀ n : ℕ, K n < 3 :=
sorry
end

end general_term_formula_K_n_less_than_3_l565_565949


namespace coefficient_x8_l565_565574

theorem coefficient_x8 (a : ℝ) : 
  let row5 := [1, 5, 15, 30, 45, 51, 45, 30, 15, 5, 1] in
  (row5.nth 2 + row5.nth 3 * a = 75) → a = 2 := 
by sorry

end coefficient_x8_l565_565574


namespace root_line_discriminant_curve_intersection_l565_565399

theorem root_line_discriminant_curve_intersection (a p q : ℝ) :
  (4 * p^3 + 27 * q^2 = 0) ∧ (ap + q + a^3 = 0) →
  (a = 0 ∧ ∀ p q, 4 * p^3 + 27 * q^2 = 0 → ap + q + a^3 = 0 → (p = 0 ∧ q = 0)) ∨
  (a ≠ 0 ∧ (∃ p1 q1 p2 q2, 
             4 * p1^3 + 27 * q1^2 = 0 ∧ ap + q1 + a^3 = 0 ∧ 
             4 * p2^3 + 27 * q2^2 = 0 ∧ ap + q2 + a^3 = 0 ∧ 
             (p1, q1) ≠ (p2, q2))) := 
sorry

end root_line_discriminant_curve_intersection_l565_565399


namespace find_k_parallel_lines_l565_565657

theorem find_k_parallel_lines (k : ℝ) : 
  (∀ x y, (k - 1) * x + y + 2 = 0 → 
            (8 * x + (k + 1) * y + k - 1 = 0 → False)) → 
  k = 3 :=
sorry

end find_k_parallel_lines_l565_565657


namespace total_notes_l565_565719

theorem total_notes :
  let red_notes := 5 * 6 in
  let blue_notes_under_red := 2 * red_notes in
  let total_blue_notes := blue_notes_under_red + 10 in
  red_notes + total_blue_notes = 100 := by
  sorry

end total_notes_l565_565719


namespace sum_of_divisors_of_twenty_four_l565_565118

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565118


namespace number_of_teams_l565_565476

-- Total number of players
def total_players : Nat := 12

-- Number of ways to choose one captain
def ways_to_choose_captain : Nat := total_players

-- Number of remaining players after choosing the captain
def remaining_players : Nat := total_players - 1

-- Number of players needed to form a team (excluding the captain)
def team_size : Nat := 5

-- Number of ways to choose 5 players from the remaining 11
def ways_to_choose_team (n k : Nat) : Nat := Nat.choose n k

-- Total number of different teams
def total_teams : Nat := ways_to_choose_captain * ways_to_choose_team remaining_players team_size

theorem number_of_teams : total_teams = 5544 := by
  sorry

end number_of_teams_l565_565476


namespace min_value_of_a_l565_565316

theorem min_value_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, -1 < x0 ∧ x0 ≤ 2 ∧ x0 - a > 0) → a = 2 :=
by
  sorry

end min_value_of_a_l565_565316


namespace g_calc_l565_565355

theorem g_calc (a b c : ℕ) (h₁ : a < b) (h₂ : b < c) (h₃ : c - b = 1) (h₄ : b - a = 1) :
  let f := λ (x : ℝ), x^3 - a * x^2 + b * x - c
  let g := λ (x : ℝ), x^3 - (a/b) * x^2 + (c * x) - a 
  let g_eval := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)
  g(1) = (1 + b - a - c) / -c :=
sorry

end g_calc_l565_565355


namespace carlos_payment_l565_565727

theorem carlos_payment (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
    B + (0.35 * (A + B + C) - B) = 0.35 * A - 0.65 * B + 0.35 * C :=
by sorry

end carlos_payment_l565_565727


namespace find_ctg_half_l565_565906

noncomputable def ctg (x : ℝ) := 1 / (Real.tan x)

theorem find_ctg_half
  (x : ℝ)
  (h : Real.sin x - Real.cos x = (1 + 2 * Real.sqrt 2) / 3) :
  ctg (x / 2) = Real.sqrt 2 / 2 ∨ ctg (x / 2) = 3 - 2 * Real.sqrt 2 :=
by
  sorry

end find_ctg_half_l565_565906


namespace problem_statement_l565_565877

theorem problem_statement : |1 - real.sqrt 3| - real.sqrt 3 * (real.sqrt 3 + 1) = -4 :=
by sorry

end problem_statement_l565_565877


namespace probability_at_least_one_even_l565_565259

-- Define the set of numbers and the condition of selecting two numbers
def S := {1, 2, 3, 4}
def even (n : ℕ) := n % 2 = 0

-- Define the problem statement
theorem probability_at_least_one_even : 
  let pairs := {A | ∃ x y, x ∈ S ∧ y ∈ S ∧ x < y ∧ A = (x, y)} in
  let favorable := {A ∈ pairs | even (A.1) ∨ even (A.2)} in
  (favorable.card : ℚ)/(pairs.card : ℚ) = 5/6 :=
sorry

end probability_at_least_one_even_l565_565259


namespace sum_of_divisors_of_24_is_60_l565_565065

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565065


namespace alex_choice_a_l565_565519

theorem alex_choice_a (a b c d e f : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c)
  (h_d_pos : 0 < d) (h_e_pos : 0 < e) (h_f_pos : 0 < f) 
  (poly_cond : (1 - x)^a * (1 + x)^b * (1 - x + x^2)^c * (1 + x^2)^d * (1 + x + x^2)^e * (1 + x + x^2 + x^3 + x^4)^f
    = 1 - 2 * x ∧ ∀ k > 6, coeff (polynomial.expand ℤ k) = 0)
  (h1 : a > d + e + f) (h2 : b > c + d) (h3 : e > c) : a = 23 :=
sorry

end alex_choice_a_l565_565519


namespace solve_for_m_l565_565975

-- Define the given quadratic equation
def quadratic_eq (m : ℝ) (x : ℂ) : Prop :=
  x^2 + (1 - 2 * complex.I) * x + (3 * m - complex.I) = 0

-- The function that asserts x being a real root of quadratic equation
def real_root (m : ℝ) (x : ℝ) : Prop :=
  quadratic_eq m x

-- The main theorem stating that m = 1/12 given that the quadratic equation has real roots
theorem solve_for_m (m : ℝ) (x : ℝ)
  (H : real_root m x) : m = 1 / 12 :=
sorry

end solve_for_m_l565_565975


namespace max_equal_area_points_convex_quadrilateral_l565_565592

noncomputable def max_equal_area_points (P : Type) [plane P] (A B C D : P) : ℕ := sorry

theorem max_equal_area_points_convex_quadrilateral (P : Type) [plane P] (A B C D : P)
  (h_convex : convex_quad A B C D) :
  ∃ P1 P2 P3 : P, 
  (∀ P, equal_area_tris P A B C D ↔ P = P1 ∨ P = P2 ∨ P = P3) →
  max_equal_area_points <= 3 :=
sorry

end max_equal_area_points_convex_quadrilateral_l565_565592


namespace sum_of_divisors_of_24_l565_565058

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565058


namespace degree_f_plus_g_l565_565773

variables {R : Type*} [CommRing R]

def f (z : R) (a3 a1 a0 : R) := a3 * z ^ 3 + 0 * z ^ 2 + a1 * z + a0
def g (z : R) (b2 b1 b0 : R) := b2 * z ^ 2 + b1 * z + b0

theorem degree_f_plus_g (a3 b2 b1 a1 a0 b0 : R) (z : R) (h : a3 ≠ 0) :
  (f z a3 a1 a0 + g z b2 b1 b0).degree = 3 :=
sorry

end degree_f_plus_g_l565_565773


namespace minimum_city_pairs_l565_565327

theorem minimum_city_pairs (N : ℕ) (hN : N = 125) :
  ∃ m : ℕ, (∀ (s : Finset (Fin N)) (hs : s.card = 4), s.pairwise (λ x y, x ≠ y → connected x y)) ∧
  m = 7688 :=
by
  sorry


end minimum_city_pairs_l565_565327


namespace smallest_number_l565_565521

noncomputable theory

open Real

def a : ℝ := 0
def b : ℝ := 1 / 3
def c : ℝ := -1
def d : ℝ := Real.sqrt 2

theorem smallest_number
  (h1 : a = 0)
  (h2 : b = 1 / 3)
  (h3 : c = -1)
  (h4 : d = Real.sqrt 2) :
  ∀ x ∈ {a, b, c, d}, c ≤ x :=
by
  simp only [a, b, c, d]
  sorry

end smallest_number_l565_565521


namespace canonical_eqns_line_l565_565142

-- Lean definitions for the plane equations
def plane1 (x y z : ℝ) : Prop := 3 * x + 4 * y + 3 * z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y - 2 * z + 4 = 0

-- Prove the canonical equation of the line given the plane equations
theorem canonical_eqns_line {x y z : ℝ} 
  (h1 : plane1 x y z) 
  (h2 : plane2 x y z) : 
  (\frac{x + 1}{4} = \frac{y - \frac{1}{2}}{12} ∧ \frac{y - \frac{1}{2}}{12} = \frac{z}{-20}) := 
sorry

end canonical_eqns_line_l565_565142


namespace allocation_methods_count_l565_565868

theorem allocation_methods_count :
  (∃ (pos : ℕ), pos = 10) ∧
  (∃ (schools : ℕ), schools = 4) ∧
  (∀ (s : ℕ), s ≠ 0) →
  @Finset.card nat (Finset.Icc 1 10) (allocate 10 4) = 84 :=
sorry

end allocation_methods_count_l565_565868


namespace quadratic_inequality_l565_565607

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h₁ : quadratic_function a b c 1 = quadratic_function a b c 3) 
  (h₂ : quadratic_function a b c 1 > quadratic_function a b c 4) : 
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end quadratic_inequality_l565_565607


namespace sum_divisors_24_l565_565030

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565030


namespace cost_per_component_l565_565849

theorem cost_per_component (C : ℝ) : 
  (∀ (shipping_cost fixed_cost min_price production_quantity : ℝ), 
  shipping_cost = 7 ∧ 
  fixed_cost = 16500 ∧ 
  min_price = 198.33 ∧ 
  production_quantity = 150 →
  (production_quantity * C + production_quantity * shipping_cost + fixed_cost) ≤ (production_quantity * min_price)) → 
  C ≤ 81.33 :=
by
  intro h
  specialize h 7 16500 198.33 150
  cases h with _ h1
  exact sorry

end cost_per_component_l565_565849


namespace perimeter_of_triangle_l565_565883

theorem perimeter_of_triangle (x y : ℝ) (h : 0 < x) (h1 : 0 < y) (h2 : x < y) :
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  (2 * leg_length + hypotenuse = (y - x) * (1 + 1 / Real.sqrt 2)) :=
by
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  sorry

end perimeter_of_triangle_l565_565883


namespace cost_price_percentage_l565_565398

theorem cost_price_percentage (MP CP : ℝ) 
  (h1 : MP * 0.9 = CP * (72 / 70))
  (h2 : CP / MP * 100 = 87.5) :
  CP / MP = 0.875 :=
by {
  sorry
}

end cost_price_percentage_l565_565398


namespace no_primes_in_range_l565_565257

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem no_primes_in_range (n : ℕ) (h : n > 1) :
  ∀ k, n! < k ∧ k < n! + 2 * n → ¬ prime k :=
by 
sorry

end no_primes_in_range_l565_565257


namespace sum_of_divisors_of_twenty_four_l565_565119

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565119


namespace diagonals_of_polygon_with_inner_hexagon_l565_565659

theorem diagonals_of_polygon_with_inner_hexagon :
  ∀ (n m : ℕ), n = 30 → m = 6 → m < n → (∑ i in finset.range(n), (n - 3)) / 2 - m * (m - 3) / 2 = 396 := 
by
  intros n m h1 h2 h3
  rw [h1, h2]
  sorry

end diagonals_of_polygon_with_inner_hexagon_l565_565659


namespace quadratic_function_proof_l565_565914

-- Conditions from the problem
def has_asymptotes (q : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, q = λ x, a * (x + 2) * (x - 2)

def q_value_at_3 (q : ℝ → ℝ) : Prop :=
  q 3 = 12

-- The quadratic function to prove
noncomputable def q (x : ℝ) : ℝ := (12 / 5) * (x^2 - 4)

-- The theorem statement
theorem quadratic_function_proof :
  (∃ q : ℝ → ℝ, has_asymptotes q ∧ q_value_at_3 q) → q = λ x, (12x^2 - 48) / 5 :=
by 
  sorry

end quadratic_function_proof_l565_565914


namespace equivalence_statements_l565_565456

variables (P Q : Prop)

theorem equivalence_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_statements_l565_565456


namespace sum_of_divisors_of_24_l565_565056

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565056


namespace find_sum_x_y_z_l565_565157

variables 
  (A B C D P Q D' B' : Type)
  [rect_ABCD : Rect ABCD]
  [P_on_AB : P ∈ AB]
  [Q_on_CD : Q ∈ CD]
  [BP_lt_DQ : ∃ BP DQ, BP < DQ]
  [D_on_AD : D' ∈ AD]
  [AD'_val : AD' = 7]
  [BP_val : BP = 27]
  [angle_ADB_BPA : ∠(AD'B') = ∠(B'PA)]

noncomputable theory

def area_expressed (x y z : ℤ) : Prop := 
  ∃ (ABCD_area : ℤ), ABCD_area = x + y * Real.sqrt z

theorem find_sum_x_y_z : ∃ (x y z : ℤ), 
  area_expressed x y z ∧ x + y + z = 616 :=
sorry

end find_sum_x_y_z_l565_565157


namespace new_biography_percentage_l565_565344

-- Define the initial conditions as Lean variables
variable (T : ℝ)  -- Total number of books initially
variable (B : ℝ)  -- Number of biography books initially

-- Define the conditions from the problem
def initial_percentage : Prop := B = 0.20 * T
def increase_rate : ℝ := 0.8823529411764707
def new_biographies : ℝ := B * (1 + increase_rate)

-- The total number of books remains the same
def total_books_unchanged : Prop := T = T

-- Define the new percentage of biography books
def new_percentage (B_new : ℝ) : ℝ := (B_new / T) * 100

-- Given the conditions, the problem is to prove the new percentage
theorem new_biography_percentage :
  initial_percentage →
  total_books_unchanged →
  new_percentage (new_biographies B) = 37.64705882352941 := by
  sorry

end new_biography_percentage_l565_565344


namespace sum_of_divisors_of_24_l565_565104

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565104


namespace smallest_positive_period_monotonically_increasing_intervals_max_and_min_values_l565_565641

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x + (π / 4)) - 1

-- Smallest positive period
theorem smallest_positive_period : 
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
  sorry

-- Monotonically increasing intervals
theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x ∈ set.Icc (k * π - 3 * π / 8) (k * π + π / 8), 
  (0 < deriv f x) :=
  sorry

-- Maximum and minimum values
theorem max_and_min_values :
  (∀ x : ℝ, f x ≤ sqrt 2 - 1) ∧ (∃ x : ℝ, f x = sqrt 2 - 1) ∧ 
  (∀ x : ℝ, -sqrt 2 - 1 ≤ f x) ∧ (∃ x : ℝ, f x = -sqrt 2 - 1) :=
  sorry

end smallest_positive_period_monotonically_increasing_intervals_max_and_min_values_l565_565641


namespace stickers_given_l565_565820

theorem stickers_given (s_start s_end s_given : ℝ) (h1 : s_start = 36.0) (h2 : s_end = 43) :
  s_given = s_end - s_start :=
by {
  rw [h1, h2],
  exact sorry,
}

end stickers_given_l565_565820


namespace result_more_than_half_l565_565317

theorem result_more_than_half (x : ℕ) (h : x = 4) : (2 * x + 5) - (x / 2) = 11 := by
  sorry

end result_more_than_half_l565_565317


namespace distance_ratio_l565_565881

theorem distance_ratio (D90 D180 : ℝ) 
  (h1 : D90 + D180 = 3600) 
  (h2 : D90 / 90 + D180 / 180 = 30) : 
  D90 / D180 = 1 := 
by 
  sorry

end distance_ratio_l565_565881


namespace find_base_length_l565_565832

-- Define the isosceles triangle with given sides
structure IsoscelesTriangle where
  A B C D E : Type
  AB BC AC : ℕ

-- Define the given conditions as a structure
structure Conditions (T : IsoscelesTriangle) where
  (lateral_side : T.AB = 12)
  (lateral_side_second : T.BC = 12)
  (base_AC : Nat)
  (ray_point_D : Nat = 24)
  (perpendicular_DE : Nat)
  (BE : Nat = 6)

-- Define the theorem to prove that the length of AC is 18
theorem find_base_length (T : IsoscelesTriangle) (C : Conditions T) : 
    C.base_AC = 18 := by
  sorry

end find_base_length_l565_565832


namespace problem_statement_l565_565629

theorem problem_statement (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π / 2)) :
  sqrt ((1 + sin α) / (1 - sin α)) - sqrt ((1 - sin α) / (1 + sin α)) = -2 * tan α :=
sorry

end problem_statement_l565_565629


namespace part1_part2_part3_l565_565216

theorem part1 : (sqrt 18 / 3) * sqrt 6 = 2 * sqrt 3 := 
by 
  sorry

theorem part2 : (sqrt 18 + 3) / sqrt 3 - 6 * sqrt (3 / 2) = sqrt 3 - 2 * sqrt 6 :=
by 
  sorry

theorem part3 : (sqrt 7 + 2 * sqrt 2) * (sqrt 7 - 2 * sqrt 2) = -1 :=
by 
  sorry

end part1_part2_part3_l565_565216


namespace angle_DEF_l565_565831

open_locale real

-- Define the given conditions
variables {α β γ : ℝ}
variables [triangle : triangle ABC] [θ : angle (90 : ℝ)]
variables (isosceles_right_triangle : isosceles ABC) (right_angle_at_C : ∠C = 90)
variables (D_on_AB : D ∈ line_segment AB) (E_on_extension_of_AB : E ∉ line_segment AB)
variables (KL_midline : midline K L ABC DEF)
variables (quadrilateral_area_condition : area D K L B = (5/8) * area ABC)

-- Define the proof goal
theorem angle_DEF :
  ∠DEF = arctan 1/4 :=
sorry

end angle_DEF_l565_565831


namespace ME_dot_OF_range_l565_565637

-- Define the circle M and conditions
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 4

-- Define the midpoints E and F
variable (E F : ℝ × ℝ)

-- Define that E and F are midpoints of sides AB and BC
def is_midpoint (a b midpoint : ℝ × ℝ) : Prop := midpoint = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
def inscribed_equilateral_triangle := (A B C : ℝ × ℝ) × (circle A.1 A.2) × (circle B.1 B.2) × (circle C.1 C.2)

-- The theorem we want to prove
theorem ME_dot_OF_range (O : ℝ × ℝ) (A B C : ℝ × ℝ)
  (h1 : circle 3 3)
  (h2 : inscribed_equilateral_triangle (A, B, C, h1))
  (h3 : is_midpoint A B E)
  (h4 : is_midpoint B C F) :
  ∃ (range : set ℝ), range = set.Icc (-1/2 - 3 * real.sqrt 2) (-1/2 + 3 * real.sqrt 2) ∧
  (∀ θ : ℝ, cos θ ∈ range) := sorry

end ME_dot_OF_range_l565_565637


namespace sum_of_divisors_of_24_is_60_l565_565041

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565041


namespace sum_of_divisors_of_24_l565_565092

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565092


namespace basketball_points_ratio_l565_565372

theorem basketball_points_ratio 
  (team_2p : ℕ) (team_3p : ℕ) (team_ft : ℕ)
  (opponents_2p_ratio : ℕ) (total_points : ℕ) :
  team_2p = 25 → team_3p = 8 → team_ft = 10 →
  opponents_2p_ratio = 2 →
  total_points = 201 →
  let total_team_points := (team_2p * 2) + (team_3p * 3) + (team_ft * 1) in
  let total_opponents_2p_points := (team_2p * 2 * opponents_2p_ratio) in
  let opponents_3p_and_ft_points := total_points - total_team_points - total_opponents_2p_points in
  let team_3p_and_ft_points := (team_3p * 3) + (team_ft * 1) in
  R := opponents_3p_and_ft_points / team_3p_and_ft_points →
  R = (1/2) := 
begin
  intros,
  sorry -- Proof not required
end

end basketball_points_ratio_l565_565372


namespace find_distance_between_PQ_l565_565007

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end find_distance_between_PQ_l565_565007


namespace sum_of_divisors_of_24_is_60_l565_565064

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565064


namespace marbles_total_l565_565343

def marbles_initial := 22
def marbles_given := 20

theorem marbles_total : marbles_initial + marbles_given = 42 := by
  sorry

end marbles_total_l565_565343


namespace problem_l565_565296

open Real

/-- Prove the value of ω and the range of m for the given function. -/
theorem problem (f : ℝ → ℝ)
  (hf : ∀ x, f x = 2 * sin (3 * x - π / 3))
  (hω_period : ∀ x, f (x + π / 3) = f x)
  (h_ineq : ∀ x ∈ Icc (π / 6) (π / 2), abs (f x - m) < 2) :
  (ω = 3 / 2) ∧ (0 < m ∧ m < 1) :=
by
  sorry

end problem_l565_565296


namespace minimum_perimeter_isosceles_triangles_l565_565001

theorem minimum_perimeter_isosceles_triangles 
  (a b c : ℤ) :
  (2 * a + 16 * c = 2 * b + 18 * c) ∧ 
  (8 * c * (real.sqrt (a ^ 2 - (8 * c) ^ 2)) = 9 * c * (real.sqrt (b ^ 2 - (9 * c) ^ 2))) ∧
  (64 * a + 217 * c = 81 * b) →
  (∀ s : ℤ, a = 298 * s ∧ b = 281 * s ∧ c = 17 * s) →
  (2 * a + 16 * c) = 868 :=
sorry

end minimum_perimeter_isosceles_triangles_l565_565001


namespace calc_dot_product_l565_565969

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (angle_ab : ℝ)
variables (norm_a : ℝ) (norm_b : ℝ)
variable (h_angle : angle_ab = π / 6)
variable (h_norm_a : ‖a‖ = 2)
variable (h_norm_b : ‖b‖ = sqrt 3)

noncomputable def dot_product_value := (a ⬝  (2 • b - a))

theorem calc_dot_product : dot_product_value a b = 2 :=
by 
  unfold dot_product_value
  sorry

end calc_dot_product_l565_565969


namespace range_of_a_l565_565294

variable {f : ℝ → ℝ}
variable {a : ℝ}

def f (x : ℝ) : ℝ := Real.log x - (1 / 2) * x^2 + a * x

theorem range_of_a (h : ∃ x ∈ Ioo 1 2, is_local_min_on f (Ioo 1 2) x ∨ is_local_max_on f (Ioo 1 2) x) : 
  0 < a ∧ a < 3 / 2 :=
by
  sorry

end range_of_a_l565_565294


namespace octagon_area_fraction_l565_565810

theorem octagon_area_fraction {A B C D A1 B1 C1 D1 : ℝ} (T : ℝ):
  let rect := T in
  let octagon_area := (1 : ℝ) / 6 * rect in
  octagon_area = (1 / 6) * T :=
begin
  sorry
end

end octagon_area_fraction_l565_565810


namespace ratio_of_angles_l565_565198

theorem ratio_of_angles (P Q R S T : Type) 
  (isAcuteAngled : AcuteAngled (triangle P Q R)) (inscribed : InscribedInCircle (triangle P Q R) (circle S))
  (arcPQ : Angle PQ = 100) (arcQR : Angle QR = 140)
  (T_in_minor_arc_PR : MinorArc PR T) (ST_perp_PR : Perpendicular (line ST) (line PR)) :
  (angle SQT / angle QSP) = 7 / 5 := 
sorry

end ratio_of_angles_l565_565198


namespace nails_needed_for_house_wall_l565_565591

theorem nails_needed_for_house_wall :
  let large_planks : Nat := 13
  let nails_per_large_plank : Nat := 17
  let additional_nails : Nat := 8
  large_planks * nails_per_large_plank + additional_nails = 229 := by
  sorry

end nails_needed_for_house_wall_l565_565591


namespace a_seq_10_l565_565654

-- Define the recursive sequence
def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := a_seq n + 2^n

-- Prove that a_seq 10 = 1023
theorem a_seq_10 : a_seq 10 = 1023 := sorry

end a_seq_10_l565_565654


namespace tan_beta_is_neg3_l565_565281

theorem tan_beta_is_neg3 (α β : ℝ) (h1 : Real.tan α = -2) (h2 : Real.tan (α + β) = 1) : Real.tan β = -3 := 
sorry

end tan_beta_is_neg3_l565_565281


namespace triangle_inequality_and_angle_conditions_l565_565799

theorem triangle_inequality_and_angle_conditions
  {A B C M : Type} [acute_triangle A B C] [gravity_center A B C M]
  (h1 : ∠(A, M, B) = 2 * ∠(A, C, B)) :
  (AB^4 = AC^4 + BC^4 - AC^2 * BC^2) ∧ (∠(A, C, B) ≥ 60) :=
by
  sorry

end triangle_inequality_and_angle_conditions_l565_565799


namespace multiple_of_1897_l565_565927

theorem multiple_of_1897 (n : ℕ) : ∃ k : ℤ, 2903^n - 803^n - 464^n + 261^n = k * 1897 := by
  sorry

end multiple_of_1897_l565_565927


namespace sum_of_real_solutions_l565_565248

theorem sum_of_real_solutions :
  let eq := ∀ (x : ℝ), (x-3)/(x^2 + 5*x + 2) = (x-6)/(x^2 - 11*x)
  ∑ (r : ℝ) in { x : ℝ | eq x }, r = 62/13 :=
by { sorry }

end sum_of_real_solutions_l565_565248


namespace price_of_tomatoes_and_cucumbers_l565_565381

theorem price_of_tomatoes_and_cucumbers :
  ∀ (price_cucumber : ℝ) (discount : ℝ),
  price_cucumber = 5 →
  discount = 0.2 →
  let price_tomato := price_cucumber * (1 - discount) in
  let cost_tomatoes_two_kg := 2 * price_tomato in
  let cost_cucumbers_three_kg := 3 * price_cucumber in
  cost_tomatoes_two_kg + cost_cucumbers_three_kg = 23 :=
by
  intros price_cucumber discount hcucumber hdiscount
  let price_tomato := price_cucumber * (1 - discount)
  let cost_tomatoes_two_kg := 2 * price_tomato
  let cost_cucumbers_three_kg := 3 * price_cucumber
  sorry

end price_of_tomatoes_and_cucumbers_l565_565381


namespace limit_example_l565_565212

noncomputable def numerator (x : ℝ) := x^4 - 1
noncomputable def denominator (x : ℝ) := x - 1

theorem limit_example : (Real.limit (λ x, (numerator x / denominator x)) 1 = 4) := sorry

end limit_example_l565_565212


namespace part1_part2_l565_565978

-- Definition of the function f
def f (x : ℝ) : ℝ := 4 * sin (x / 2) * cos (x / 2 + π / 3) + sqrt 3

-- Statement for Part (1)
theorem part1 (α : ℝ) (h : f α = 1 / 3) : cos (α + 5 / 6 * π) = -1 / 6 := 
sorry

-- Statement for Part (2)
theorem part2 (λ : ℝ) (h : ∀ x ∈ set.Icc (-π / 6) (π / 3), abs (f x ^ 2 - λ) ≤ f x + 2) : 
  λ ∈ set.Icc 0 4 :=
sorry

end part1_part2_l565_565978


namespace compute_expression_l565_565543

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end compute_expression_l565_565543


namespace fewer_pounds_of_carbon_emitted_l565_565700

theorem fewer_pounds_of_carbon_emitted 
  (population : ℕ)
  (pollution_per_car : ℕ)
  (pollution_per_bus : ℕ)
  (bus_capacity : ℕ)
  (percentage_bus : ℚ)
  (initial_population : population = 80)
  (initial_pollution_per_car : pollution_per_car = 10)
  (initial_pollution_per_bus : pollution_per_bus = 100)
  (initial_bus_capacity : bus_capacity = 40)
  (initial_percentage_bus : percentage_bus = 0.25) :
  (population * pollution_per_car - ((population - population * percentage_bus.to_nat) * pollution_per_car + pollution_per_bus)) = 100 :=
by 
  sorry

end fewer_pounds_of_carbon_emitted_l565_565700


namespace equal_distances_point_l565_565740

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

theorem equal_distances_point :
  let A := Point3D.mk 10 0 0
  let B := Point3D.mk 0 (-6) 0
  let C := Point3D.mk 0 0 8
  let D := Point3D.mk 0 0 0
  let P := Point3D.mk 5 (-3) 4
  distance P A = distance P D ∧
  distance P B = distance P D ∧
  distance P C = distance P D :=
by
  sorry

end equal_distances_point_l565_565740


namespace poets_overlap_probability_l565_565003

theorem poets_overlap_probability :
  let born_interval := (0:ℝ, 300:ℝ)
  let lifespan := 60
  let total_area := 300 * 300
  let overlapping_area := total_area - 2 * (1 / 2 * 180 * 180)
  let probability := overlapping_area / total_area
  probability = (16 / 25) := 
by
  let born_interval := (0:ℝ, 300:ℝ)
  let lifespan := 60
  let total_area := (300:ℝ) * (300:ℝ)
  let non_overlapping_area := 2 * ((1 / 2) * (180:ℝ) * (180:ℝ))
  let overlapping_area := total_area - non_overlapping_area
  let probability := overlapping_area / total_area
  have h : 90000 = total_area := by sorry
  have h1 : 32400 = non_overlapping_area := by sorry
  have h2 : 57600 = overlapping_area := by linarith
  have h3 : probability = 57600 / 90000 := by linarith
  have h4 : 57600 / 90000 = (16 / 25) := by norm_num
  exact h3.trans h4

end poets_overlap_probability_l565_565003


namespace baby_guppies_calculation_l565_565201

-- Define the problem in Lean
theorem baby_guppies_calculation :
  ∀ (initial_guppies first_sighting two_days_gups total_guppies_after_two_days : ℕ), 
  initial_guppies = 7 →
  first_sighting = 36 →
  total_guppies_after_two_days = 52 →
  total_guppies_after_two_days = initial_guppies + first_sighting + two_days_gups →
  two_days_gups = 9 :=
by
  intros initial_guppies first_sighting two_days_gups total_guppies_after_two_days
  intros h_initial h_first h_total h_eq
  sorry

end baby_guppies_calculation_l565_565201


namespace smallest_positive_root_exists_l565_565887

noncomputable def satisfies_equation (b c d x : ℝ) : Prop :=
  x^3 + b * x^2 + c * x + d = 0

theorem smallest_positive_root_exists (b c d x : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 3) (hd : |d| ≤ 2) :
  x = 1 + Real.sqrt 3 → satisfies_equation b c d x :=
by
  assume h : x = 1 + Real.sqrt 3
  sorry

end smallest_positive_root_exists_l565_565887


namespace star_evaluation_l565_565589

def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_evaluation : star (star 3 5) 7 = -3 / 11 := by 
  sorry

end star_evaluation_l565_565589


namespace triangle_inequality_3_4_5_l565_565817

theorem triangle_inequality_3_4_5 :
  ∀ (a b c : ℕ), 
  a = 3 → b = 4 → c = 5 →
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  split
  simp
  split
  simp
  simp
  sorry

end triangle_inequality_3_4_5_l565_565817


namespace problem_statement_l565_565261

noncomputable def f (x : ℝ) := Real.log 9 * (Real.log x / Real.log 3)

theorem problem_statement : deriv f 2 + deriv f 2 = 1 := sorry

end problem_statement_l565_565261


namespace sum_of_divisors_of_24_l565_565049

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565049


namespace area_of_triangle_determined_l565_565712

-- The given side lengths of the triangle ABC
def side_lengths : nat × nat × nat := (5, 12, 13)

-- Definitions for points of intersection according to the given conditions
variables {A A1 A2 B B1 B2 C C1 C2 : Type}

-- Area calculation of the determined triangle given the conditions
theorem area_of_triangle_determined (A B C A1 A2 B1 B2 C1 C2 : Type)
  (h1 : A = focus_of_parabola_with_directrix B C)
  (h2 : A1 = intersection_with_side A B)
  (h3 : A2 = intersection_with_side A C)
  (h4 : B = focus_of_parabola_with_directrix C A)
  (h5 : B1 = intersection_with_side B C)
  (h6 : B2 = intersection_with_side B A)
  (h7 : C = focus_of_parabola_with_directrix A B)
  (h8 : C1 = intersection_with_side C A)
  (h9 : C2 = intersection_with_side C B) :
  area_of_triangle A1 C2 B1 A2 C1 B2 = 6728 / 3375 := 
sorry

end area_of_triangle_determined_l565_565712


namespace sum_of_real_solutions_l565_565246

theorem sum_of_real_solutions :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 11 * x) →
  (∃ r1 r2 : ℝ, r1 + r2 = 46 / 13) :=
by
  sorry

end sum_of_real_solutions_l565_565246


namespace find_angle_BMA_l565_565378

variable {A B C M : Type} [Point A] [Point B] [Point C] [Point M]
variables (AM BM MC BMA MBC BAC : ℝ)

theorem find_angle_BMA (h1 : AM = BM + MC) (h2 : BMA = MBC + BAC) : BMA = 60 :=
sorry

end find_angle_BMA_l565_565378


namespace sum_of_divisors_of_24_l565_565085

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565085


namespace sum_of_divisors_of_24_is_60_l565_565044

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565044


namespace trig_identity_sin_cos_l565_565841

theorem trig_identity_sin_cos :
  sin (42 * real.pi / 180) * cos (18 * real.pi / 180) -
  cos (138 * real.pi / 180) * cos (72 * real.pi / 180) = real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_sin_cos_l565_565841


namespace oatmeal_cookie_count_l565_565388

noncomputable def total_cookies (baggies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  baggies * cookies_per_bag

noncomputable def oatmeal_cookies (total : ℕ) (chocolate_chip : ℕ) : ℕ :=
  total - chocolate_chip

theorem oatmeal_cookie_count :
  ∀ (baggies cookies_per_bag chocolate_chip : ℕ),
  baggies = 8 →
  cookies_per_bag = 6 →
  chocolate_chip = 23 →
  oatmeal_cookies (total_cookies baggies cookies_per_bag) chocolate_chip = 25 :=
by
  intros baggies cookies_per_bag chocolate_chip 
  rintros ⟨hb, hc, hcc⟩
  subst hb
  subst hc
  subst hcc
  simp [total_cookies, oatmeal_cookies]
  sorry

end oatmeal_cookie_count_l565_565388


namespace diagonals_in_25_sided_polygon_l565_565557

-- Define a function to calculate the number of specific diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 5) / 2

-- Theorem stating the number of diagonals for a convex polygon with 25 sides with the given condition
theorem diagonals_in_25_sided_polygon : number_of_diagonals 25 = 250 := 
sorry

end diagonals_in_25_sided_polygon_l565_565557


namespace cuboid_surface_area_4_8_6_l565_565009

noncomputable def cuboid_surface_area (length width height : ℕ) : ℕ :=
  2 * (length * width + length * height + width * height)

theorem cuboid_surface_area_4_8_6 : cuboid_surface_area 4 8 6 = 208 := by
  sorry

end cuboid_surface_area_4_8_6_l565_565009


namespace average_speeds_proof_l565_565723

-- Defining the given conditions
def distance_A_B : ℝ := 240 -- in miles
def time_A_B : ℝ := 6 -- in hours
def distance_B_C : ℝ := 360 -- in miles
def time_B_C : ℝ := 7 -- in hours
def time_C_A : ℝ := 10 -- in hours
def delta_speed_C_A : ℝ := 15 -- mph faster from C to A than from A to B
def delta_speed_B_C : ℝ := -5 -- mph less from B to C than from A to B

-- Stating the average speeds to be proven
def average_speed_A_B : ℝ := distance_A_B / time_A_B
def average_speed_B_C : ℝ := distance_B_C / time_B_C
def average_speed_C_A : ℝ := average_speed_A_B + delta_speed_C_A

-- The task/problem statement
theorem average_speeds_proof :
  average_speed_A_B = 40 ∧
  average_speed_B_C ≈ 51.43 ∧
  average_speed_C_A = 55 :=
sorry

end average_speeds_proof_l565_565723


namespace lattice_points_count_l565_565995

theorem lattice_points_count : ∃ (S : Finset (ℤ × ℤ)), 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 45}.toFinset = S ∧ S.card = 6 := 
sorry

end lattice_points_count_l565_565995


namespace best_fit_model_is_model_1_l565_565258

def correlation_coefficient_condition (n : Nat) : Real :=
  match n with
  | 1 => 0.98
  | 2 => 0.80
  | 3 => 0.50
  | 4 => 0.25
  | _ => 0.0

theorem best_fit_model_is_model_1 :
  ∀ n : Fin 4, abs (correlation_coefficient_condition n) ≤ abs (correlation_coefficient_condition 1) :=
by
  intros
  cases n
  · to_rhs_le
  · sorry
  · sorry
  · sorry
  · to_rhs_le_eq

end best_fit_model_is_model_1_l565_565258


namespace probability_one_intersection_l565_565167

def quadratic_discriminant_zero (m n : ℕ) : Prop :=
  m^2 = 4 * n

theorem probability_one_intersection (h : ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6)) :
  let outcomes := [(m, n) | m ← [1, 2, 3, 4, 5, 6], n ← [1, 2, 3, 4, 5, 6]] in
  let valid_pairs := list.filter (λ (p : ℕ × ℕ), quadratic_discriminant_zero p.1 p.2) outcomes in
  (list.length valid_pairs : ℚ) / (list.length outcomes : ℚ) = 1 / 18 :=
by sorry

end probability_one_intersection_l565_565167


namespace arjun_initial_investment_l565_565205

theorem arjun_initial_investment :
  ∃ (X : ℝ), X * 12 = 40_000 * 6 ∧ X = 20_000 :=
by 
  use 20000
  split
  · calc
    20000 * 12 = 240000 : by norm_num
            ... = 40_000 * 6 : by norm_num
  · norm_num

end arjun_initial_investment_l565_565205


namespace lambda_parallel_vectors_l565_565960

theorem lambda_parallel_vectors (λ : ℝ) 
    (a b : ℝ × ℝ)
    (ha : a = (3, 2)) 
    (hb : b = (2, -1)) 
    (parallel : ∃ k : ℝ, k • (λ • a + b) = (a + λ • b)) 
    : λ = 1 ∨ λ = -1 :=
sorry

end lambda_parallel_vectors_l565_565960


namespace expression_result_l565_565416

-- We denote k as a natural number representing the number of digits in A, B, C, and D.
variable (k : ℕ)

-- Definitions of the numbers A, B, C, D, and E based on the problem statement.
def A : ℕ := 3 * ((10 ^ (k - 1) - 1) / 9)
def B : ℕ := 4 * ((10 ^ (k - 1) - 1) / 9)
def C : ℕ := 6 * ((10 ^ (k - 1) - 1) / 9)
def D : ℕ := 7 * ((10 ^ (k - 1) - 1) / 9)
def E : ℕ := 5 * ((10 ^ (2 * k) - 1) / 9)

-- The statement we want to prove.
theorem expression_result :
  E - A * D - B * C + 1 = (10 ^ (k + 1) - 1) / 9 :=
by
  sorry

end expression_result_l565_565416


namespace count_p_safe_5000_l565_565256

def is_p_safe (n p : ℕ) : Prop := ∀ k : ℤ, k * p = (k * p : ℤ), abs (n - k * p) > 3

def count_p_safe (n : ℕ) : ℕ :=
  let safe_5 := ∀ (k : ℤ), abs (n - k * 5) > 3
  let safe_9 := ∀ (k : ℤ), abs (n - k * 9) > 3
  let safe_11 := ∀ (k : ℤ), abs (n - k * 11) > 3
  if safe_5 ∧ safe_9 ∧ safe_11 then 1 else 0

theorem count_p_safe_5000 :
  (Finset.range 5001).sum count_p_safe = 250 := sorry

end count_p_safe_5000_l565_565256


namespace seq_periodicity_l565_565987

theorem seq_periodicity (x : ℕ → ℝ) (a : ℝ) (b : ℝ) 
    (h1 : a = 2 - real.sqrt 3) 
    (h2 : ∀ n, x (n + 1) = (x n + a) / (1 - a * x n)) : 
  x 1001 - x 401 = 0 := 
by sorry

end seq_periodicity_l565_565987


namespace Mel_weight_is_70_l565_565530

-- Definitions and conditions
def MelWeight (M : ℕ) :=
  3 * M + 10

theorem Mel_weight_is_70 (M : ℕ) (h1 : 3 * M + 10 = 220) :
  M = 70 :=
by
  sorry

end Mel_weight_is_70_l565_565530


namespace geometric_seq_ratio_l565_565716

noncomputable theore asm
theorem geometric_seq_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : ∀ n, a n > 0) 
  (h3 : a 1 + 2 * a 1 * q = a 1 * q^2) : q = 1 + Real.sqrt 2 :=
by
  sorry

end geometric_seq_ratio_l565_565716


namespace triangle_area_half_parallelogram_l565_565993

variables {A B C P Q R S : Type}
variables [HasArea A B C] [HasArea P Q R S]

theorem triangle_area_half_parallelogram 
  (h₁ : length AB ≤ length PQ)
  (h₂ : height ABC ≤ height PQRS)
  : area ABC ≤ 1/2 * area PQRS := 
sorry

end triangle_area_half_parallelogram_l565_565993


namespace prove_values_l565_565279

noncomputable theory

-- Definitions based on conditions
def nat_squares_sum (a b c d : ℕ) := a^2 + b^2 + c^2 + d^2 = 1989
def nat_squares_sum_eq (a b c d m : ℕ) := a + b + c + d = m^2
def largest_square (a b c d n : ℕ) := max a (max b (max c d)) = n^2

-- Proof problem statement
theorem prove_values (a b c d m n : ℕ) 
  (h1: nat_squares_sum a b c d)
  (h2: nat_squares_sum_eq a b c d m)
  (h3: largest_square a b c d n) :
  m = 9 ∧ n = 6 :=
by 
  sorry

end prove_values_l565_565279


namespace problem_1_problem_2_l565_565691

-- Definitions of the conditions
variable (A B C : ℝ) (a b c : ℝ)
variable (area : ℝ)
variable (cosA : ℝ)
variable (bc_diff : ℝ)

-- Conditions given in the problem
def given_conditions : Prop :=
  area = 3 * Real.sqrt 15 ∧
  bc_diff = 2 ∧
  cosA = -1 / 4

-- Part (1) proof problem
theorem problem_1 (h : given_conditions a b c area cosA bc_diff) :
  a = 8 ∧ Real.sin C = Real.sqrt 15 / 8 := sorry

-- Part (2) proof problem
theorem problem_2 (h : given_conditions a b c area cosA bc_diff) :
  Real.cos (2 * A + π / 6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := sorry

end problem_1_problem_2_l565_565691


namespace speed_of_current_l565_565178

theorem speed_of_current : 
  ∃ c : ℝ, (∀ t_upstream t_downstream : ℝ, ∀ v_boat d : ℝ, (v_boat = 16) ∧ (t_upstream = 20/60) ∧ (t_downstream = 15/60) ∧ 
      (d = (v_boat - c) * t_upstream) ∧ (d = (v_boat + c) * t_downstream) → c = 16 / 7) :=
begin
  use 16 / 7,
  intros t_upstream t_downstream v_boat d,
  rintro ⟨hv, htu, htd, hu, hd⟩,
  have hu' : d = (16 - 16/7) * (20/60), { rw [←hv, ←htu, ←hu, ←hd] },
  have hd' : d = (16 + 16/7) * (15/60), { rw [←hv, ←htd, ←hd] },
  have := hu'.symm.trans hd',
  -- Here we would continue by algebraic manipulation to show the equality.
  sorry,
end

end speed_of_current_l565_565178


namespace one_thirds_in_nine_thirds_l565_565664

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l565_565664


namespace find_x_l565_565921

theorem find_x (x : ℝ) (h1: x > 0) (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * Real.sqrt 3 :=
sorry

end find_x_l565_565921


namespace elise_initial_dog_food_l565_565571

variable (initial_dog_food : ℤ)
variable (bought_first_bag : ℤ := 15)
variable (bought_second_bag : ℤ := 10)
variable (final_dog_food : ℤ := 40)

theorem elise_initial_dog_food :
  initial_dog_food + bought_first_bag + bought_second_bag = final_dog_food →
  initial_dog_food = 15 :=
by
  sorry

end elise_initial_dog_food_l565_565571


namespace tangent_line_at_origin_range_of_a_l565_565642

-- Given the function f(x) = x^3 + ax^2 + 3x - 9, prove the following:

-- Part (1)
-- If f(x) has an extremum at x = -3, find the equation of the tangent line at the point (0, f(0)).
theorem tangent_line_at_origin
    (a : ℝ)
    (f : ℝ → ℝ)
    (extremum : (∀ x : ℝ, f x = x^3 + a * x^2 + 3 * x - 9) ∧ (∃ c : ℝ, c = -3 ∧ f' c = 0)) :
    let f' := λ x, 3 * x^2 + 2 * a * x + 3 in
    f' 0 = 3 ∧ f 0 = -9 ∧ ∃ y : ℝ, y = f' 0 * 0 + f 0 ∧ 3 * 0 - y - 9 = 0 :=
by {
    sorry
}

-- Part (2)
-- If f(x) is monotonically decreasing on the interval [1, 2], find the range of the real number a.
theorem range_of_a
    (a : ℝ)
    (f : ℝ → ℝ)
    (monotonic_decreasing : (∀ x : ℝ, f x = x^3 + a * x^2 + 3 * x - 9) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 3 * x^2 + 2 * a * x + 3 ≤ 0)) :
    a ≤ -15 / 4 :=
by {
    sorry
}

end tangent_line_at_origin_range_of_a_l565_565642


namespace correct_statement_l565_565870

-- Conditions
def sector_perimeter : ℝ := 8
def sector_area : ℝ := 4

def trig_function (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x)

def alpha_in_third_quadrant (α : ℝ) : Prop := ∃ k : ℤ, π + 2 * k * π < α ∧ α < (3 * π / 2) + 2 * k * π

def sine_equality (α β : ℝ) : Prop := sin α = sin β

def periodic_function (x : ℝ) : ℝ :=
if x.is_rational then 0 else 1

-- Proof statement (with sorry for the actual proof body)
theorem correct_statement : statement_count := 1
                := ∀ (sector_perimeter sector_area) 
                         (trig_function.alpha_in_third_quadrant sine_equality periodic_function), 
                            correct_statement = ⑤ := sorry

end correct_statement_l565_565870


namespace part_a_possible_final_number_l565_565842

theorem part_a_possible_final_number :
  ∃ (n : ℕ), n = 97 ∧ 
  (∃ f : {x // x ≠ 0} → ℕ → ℕ, 
    f ⟨1, by decide⟩ 0 = 1 ∧ 
    f ⟨2, by decide⟩ 1 = 2 ∧ 
    f ⟨4, by decide⟩ 2 = 4 ∧ 
    f ⟨8, by decide⟩ 3 = 8 ∧ 
    f ⟨16, by decide⟩ 4 = 16 ∧ 
    f ⟨32, by decide⟩ 5 = 32 ∧ 
    f ⟨64, by decide⟩ 6 = 64 ∧ 
    f ⟨128, by decide⟩ 7 = 128 ∧ 
    ∀ i j : {x // x ≠ 0}, f i j = (f i j - f i j)) := sorry

end part_a_possible_final_number_l565_565842


namespace min_red_beads_l565_565488

-- Define the structure of the necklace and the conditions
structure Necklace where
  total_beads : ℕ
  blue_beads : ℕ
  red_beads : ℕ
  cyclic : Bool
  condition : ∀ (segment : List ℕ), segment.length = 8 → segment.count blue_beads ≥ 12 → segment.count red_beads ≥ 4

-- The given problem condition
def given_necklace : Necklace :=
  { total_beads := 50,
    blue_beads := 50,
    red_beads := 0,
    cyclic := true,
    condition := sorry }

-- The proof problem: Minimum number of red beads required
theorem min_red_beads (n : Necklace) : n.red_beads ≥ 29 :=
by { sorry }

end min_red_beads_l565_565488


namespace max_angle_from_tangency_point_l565_565946

open Real

-- Definitions as per the conditions
variable (Circle : Type)
variable [MetricSpace Circle] [HilbertSpace Circle]
variable (O : Circle) -- Center of the circle
variable (R : ℝ)     -- Radius of the circle
variable (A B M : Circle) -- Points on the circle with M being the point of tangency
variable (Tangent : Circle → Circle) -- Tangent function defining tangent at a point

-- Hypotheses as per the conditions
noncomputable def isTangentLineAtM : Prop :=
  ∀ P : Circle, P ≠ M → dist M P = dist M (Tangent P)

noncomputable def isOnShorterArc : Prop :=
  ∀ P : Circle, (dist O P < π * R) → P ∈ Segment A B

-- The remaining points are derived from the general definitions
noncomputable def maxAnglePoint : Circle :=
  M -- default to point M, which is what we want to prove

-- The theorem to prove
theorem max_angle_from_tangency_point (circle : Circle) (chord : Segment A B) (tangentLine : Tangent M) :
  ∃ (M : Circle), isTangentLineAtM ∧ isOnShorterArc ∧ maxAnglePoint = M :=
begin
  sorry 
end

end max_angle_from_tangency_point_l565_565946


namespace purely_imaginary_product_solutions_l565_565236

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem purely_imaginary_product_solutions :
  ∀ x : ℝ, is_purely_imaginary ((x + 2 * complex.I) * ((x + 3) + 2 * complex.I) * ((x + 5) + 2 * complex.I)) ↔ x = -5 ∨ x = -4 ∨ x = 1 := 
by sorry

end purely_imaginary_product_solutions_l565_565236


namespace function_condition_satisfied_l565_565232

def div_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (m^2 + n)^2 % (f(m)^2 + f(n)) = 0

theorem function_condition_satisfied :
  ∀ f : ℕ → ℕ, div_condition f ↔ ∀ n : ℕ, f(n) = n := 
by
  intro f
  apply Iff.intro
  sorry -- Proof required here

end function_condition_satisfied_l565_565232


namespace ellipse_and_circle_properties_l565_565639

theorem ellipse_and_circle_properties (a b c r : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * a = 4) 
  (h4 : (c / a) = (real.sqrt 6) / 3) (h5 : ∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
  (∃ k1 k2 : ℝ, (k1 * k2 = (P.2^2 - r^2) / (P.1^2 - r^2)) ∧ constant_slope_product k1 k2)) 
  : (a = 2) ∧ (b^2 = 4 / 3) ∧ (x y : ℝ, x^2 / 4 + 3 * y^2 / 4 = 1) ∧ (r^2 = 1) → 
  (π * r^2 = π) :=
by
  sorry

end ellipse_and_circle_properties_l565_565639


namespace derivative_at_1_l565_565682

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end derivative_at_1_l565_565682


namespace inequality_proof_l565_565628

noncomputable def a : ℝ := 4 ^ 0.4
noncomputable def b : ℝ := 8 ^ 0.2
noncomputable def c : ℝ := (1 / 2) ^ (-0.5)

-- Prove that a > b > c given the conditions
theorem inequality_proof :
  a > b ∧ b > c :=
by
  -- equivalently, a > b > c
  sorry

end inequality_proof_l565_565628


namespace solution_set_of_inequality_l565_565735

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : ∀ x, 0 < x → (x * deriv f x - f x < 0))
  (h2 : f 2 = 0) : {x | (x - 1) * f x > 0} = Ioo 1 2 :=
begin
  sorry
end

end solution_set_of_inequality_l565_565735


namespace total_time_spent_l565_565724

def johns_exploring_time : ℝ := 3
def johns_notes_time : ℝ := johns_exploring_time / 2
def johns_book_time : ℝ := 0.5

theorem total_time_spent : johns_exploring_time + johns_notes_time + johns_book_time = 5 := by
  -- Total time spent is 3 years + 1.5 years + 0.5 years
  sorry

end total_time_spent_l565_565724


namespace probability_of_sum_24_l565_565168

def die_a_faces : Finset ℕ := (Finset.range 20).erase 0
def die_b_faces : Finset ℕ := (Finset.range 21).erase_all [0, 9]

theorem probability_of_sum_24 :
  let favorable_outcomes := { (a, b) | a ∈ die_a_faces ∧ b ∈ die_b_faces ∧ a + b = 24 }.card
  let total_possible_outcomes := 20 * 20 
  let probability := (favorable_outcomes : ℚ) / total_possible_outcomes 
  probability = 3 / 80 := by
  sorry

end probability_of_sum_24_l565_565168


namespace max_abs_sum_l565_565678

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : |x| + |y| ≤ sqrt (4/3) := by
  sorry

end max_abs_sum_l565_565678


namespace series_1_diverges_series_2_converges_series_3_diverges_series_4_converges_l565_565469

open Filter Real Finset

-- Part 1
theorem series_1_diverges : ¬(has_sum (λ n : ℕ, (2 * n) / (n^2 + 1)) (summable (λ n : ℕ, (2 * n) / (n^2 + 1)))).has_sum :=
sorry

-- Part 2
theorem series_2_converges : summable (λ n : ℕ, if n = 0 then 0 else 1 / (n * (log n) ^ 3)) :=
  begin
    simp only [summable, if_pos, if_neg],
    sorry -- Proof needed here
  end

-- Part 3
theorem series_3_diverges : ¬(has_sum (λ n : ℕ, 1 / (sqrt (4 * n + 1))) (summable (λ n : ℕ, 1 / (sqrt (4 * n + 1))))).has_sum :=
sorry

-- Part 4
theorem series_4_converges : summable (λ n : ℕ, if n < 3 then 0 else (n / (n^4 - 9))) :=
  begin
    simp only [summable, if_pos, if_neg],
    sorry -- Proof needed here
  end

end series_1_diverges_series_2_converges_series_3_diverges_series_4_converges_l565_565469


namespace compute_b_l565_565274

noncomputable def polynomial := Polynomial ℚ

theorem compute_b (a b : ℚ) : 
  (polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + 6).isRoot (1 + Real.sqrt 2) → 
  b = 11 := 
by
  sorry

end compute_b_l565_565274


namespace infinite_series_value_l565_565220

noncomputable def infinite_series : ℝ :=
  ∑' n, if n ≥ 2 then (n^4 + 5 * n^2 + 8 * n + 8) / (2^(n + 1) * (n^4 + 4)) else 0

theorem infinite_series_value :
  infinite_series = 3 / 10 :=
by
  sorry

end infinite_series_value_l565_565220


namespace mehki_age_l565_565373

theorem mehki_age (Z J M : ℕ) (h1 : Z = 6) (h2 : J = Z - 4) (h3 : M = 2 * (J + Z)) : M = 16 := by
  sorry

end mehki_age_l565_565373


namespace geometric_inequality_l565_565848

theorem geometric_inequality (x y : ℝ) :
  |x| + |y| ≤ sqrt (2 * (x^2 + y^2)) ∧ sqrt (2 * (x^2 + y^2)) ≤ 2 * max (|x|) (|y|) :=
by
  sorry

end geometric_inequality_l565_565848


namespace number_of_integers_satisfy_equation_l565_565582

theorem number_of_integers_satisfy_equation : 
  let n_count := { n : ℤ | 1 ≤ n ∧ n ≤ 1000 ∧ ∃ x : ℝ, ⌊x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋ = n }.to_finset.card in
  n_count = 750 :=
by sorry

end number_of_integers_satisfy_equation_l565_565582


namespace sum_of_roots_of_equations_l565_565749

theorem sum_of_roots_of_equations (m n : ℝ) (hm : m + 2^m = 4) (hn : n + log 2 n = 4) : m + n = 4 :=
sorry

end sum_of_roots_of_equations_l565_565749


namespace iterate_square_construction_l565_565950

noncomputable def side_length_after_iterations (a : ℝ) (n : ℕ) : ℝ :=
  nat.rec_on n a (λ k s_k, real.sqrt (s_k ^ 2 - 2 * s_k + 2))

theorem iterate_square_construction (a : ℝ) (h : a > 1) :
  (∀ n : ℕ, n < 2007 → side_length_after_iterations a n > 1) ↔ a ≥ 2 :=
by
  sorry

end iterate_square_construction_l565_565950


namespace derivative_at_one_l565_565684

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end derivative_at_one_l565_565684


namespace solve_inequality_l565_565771

noncomputable def poly_numerator : ℚ[X] := 8 * X^3 + 4 * X^2 - 23 * X + 83

def critical_points_num : set ℚ := {x | x = (8 * (3 * x^3 + 4 * x^2 - 23 * x + 83)).roots}

def critical_points_den : set ℚ := {x | x = 4 / 3 ∨ x = -5}

def solution_set (x : ℝ) : Prop :=
  x ∈ Ioo (-5) (sup (critical_points_num ∩ Ioo (-∞) (-5))) ∪ Ioo (4/3) (sup (critical_points_num ∩ Ioo (4/3) (∞)))

theorem solve_inequality :
  ∀ x : ℝ, (poly_numerator.eval x / ((3*x - 4)*(x + 5)) < 4) ↔ solution_set x :=
sorry

end solve_inequality_l565_565771


namespace distance_from_center_to_line_l565_565910

theorem distance_from_center_to_line : 
  let circle_eq (x y : ℝ) := x^2 + 2*x + y^2 - 3 = 0
  let line_eq (x y : ℝ) := y = x + 3
  let center_point := (-1, 0 : ℝ) in
  let point_to_line_distance (x0 y0 : ℝ) (a b c : ℝ) := (|a * x0 + b * y0 + c|) / (Real.sqrt (a^2 + b^2)) in
  point_to_line_distance (-1) 0 (-1) 1 (-3) = Real.sqrt 2 :=
by
  sorry

end distance_from_center_to_line_l565_565910


namespace parabola_vertex_on_x_axis_l565_565650

theorem parabola_vertex_on_x_axis (a : ℝ) :
  (a = 4 ∨ a = -8) :=
begin
  -- Define the parabola equation
  let p := λ (x : ℝ), x^2 - (a + 2) * x + 9,
  
  -- Calculate its vertex
  let vertex_x := (a + 2) / 2,
  let vertex_y := p vertex_x,
  
  -- Given condition: vertex is on the x-axis
  have H : vertex_y = 0,
  {
    -- substitute vertex_x back to p to get vertex_y
    sorry  -- Skipping detailed steps
  },
  
  -- Solve for 'a' given vertex_y = 0
  exact sorry  -- Skipping detailed steps
end

end parabola_vertex_on_x_axis_l565_565650


namespace original_price_of_article_l565_565176

theorem original_price_of_article 
  (S : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : S = 25)
  (h2 : gain_percent = 1.5)
  (h3 : S = P + P * gain_percent) : 
  P = 10 :=
by 
  sorry

end original_price_of_article_l565_565176


namespace book_weight_is_correct_l565_565847

-- Define the weight of one doll
def doll_weight : ℝ := 0.3

-- Define the weight of one toy car
def toy_car_weight : ℝ := 0.5

-- Define the conditions
def condition_scale_balanced (book_weight : ℝ) : Prop :=
  book_weight = 2 * doll_weight + toy_car_weight

-- Theorem stating that the weight of the book is 1.1 kg given the conditions
theorem book_weight_is_correct : 
  ∃ book_weight : ℝ, condition_scale_balanced book_weight ∧ book_weight = 1.1 :=
begin
  use 1.1,
  unfold condition_scale_balanced,
  norm_num,
  simp [doll_weight, toy_car_weight],
  norm_num,
end

end book_weight_is_correct_l565_565847


namespace one_thirds_in_nine_thirds_l565_565663

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l565_565663


namespace tangent_line_equation_l565_565291

theorem tangent_line_equation
  (x y : ℝ)
  (h₁ : x^2 + y^2 = 5)
  (hM : x = -1 ∧ y = 2) :
  x - 2 * y + 5 = 0 :=
by
  sorry

end tangent_line_equation_l565_565291


namespace part_a_part_b_l565_565461

-- Part (a)
theorem part_a (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a + R_b + R_c ≥ 6 * r := sorry

-- Part (b)
theorem part_b (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a^2 + R_b^2 + R_c^2 ≥ 12 * r^2 := sorry

end part_a_part_b_l565_565461


namespace remainder_mod_5_is_0_l565_565213

theorem remainder_mod_5_is_0 :
  (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end remainder_mod_5_is_0_l565_565213


namespace largest_is_E_l565_565224

def A : ℝ := 0.988
def B : ℝ := 0.9808
def C : ℝ := 0.989
def D : ℝ := 0.9809
def E : ℝ := 0.998

theorem largest_is_E : E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  unfold A B C D E
  simp
  split; linarith

end largest_is_E_l565_565224


namespace sum_of_divisors_of_24_l565_565090

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565090


namespace sum_of_divisors_of_twenty_four_l565_565122

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565122


namespace triangle_area_x_l565_565923

theorem triangle_area_x (x : ℝ) (h_pos : x > 0) (h_area : 1 / 2 * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
sorry

end triangle_area_x_l565_565923


namespace AB_parallel_CE_l565_565325

variables {A B C D E : Type*}
variables [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group D] [add_comm_group E]

variables (BC AD CD BE DE AC AE BD AB CE : A)

-- Conditions as parallelism relations
axiom BC_parallel_AD : ∃ k : ℝ, BC = k • AD
axiom CD_parallel_BE : ∃ m : ℝ, CD = m • BE
axiom DE_parallel_AC : ∃ n : ℝ, DE = n • AC
axiom AE_parallel_BD : ∃ p : ℝ, AE = p • BD

-- The statement to be proven
theorem AB_parallel_CE : AB ∥ CE :=
sorry

end AB_parallel_CE_l565_565325


namespace coefficient_of_x_squared_term_in_expansion_l565_565351

theorem coefficient_of_x_squared_term_in_expansion :
  let a := (x^3 - 3*x^2)
  in true → coeff (expand (a - x)^6) 2 = (-192 : ℤ) :=
by
  -- only statement, no proof
  sorry

end coefficient_of_x_squared_term_in_expansion_l565_565351


namespace min_period_and_expression_sin_alpha_plus_pi_over_3_l565_565943

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + (π / 2))

-- Part 1: Minimum Positive Period and Expression of f(x)
theorem min_period_and_expression (θ : ℝ) (hθ : 0 < θ ∧ θ < π) (hodd : ∀ x : ℝ, f x + f (-x) = 0) :
  (∃ T : ℝ, T = π) ∧ (∀ x : ℝ, f x = -sin (2 * x)) :=
by
  sorry

-- Part 2: Value of sin(α + π/3)
theorem sin_alpha_plus_pi_over_3 (α : ℝ) (hα : (π / 2) < α ∧ α < π) (h_f : f (α / 2) = -4 / 5) :
  sin (α + π / 3) = (4 - 3 * (sqrt 3)) / 10 :=
by
  sorry

end min_period_and_expression_sin_alpha_plus_pi_over_3_l565_565943


namespace sum_infinite_series_eq_one_third_l565_565585

def R (n : ℕ) : ℝ := (n^2 + 2 * n + 2) / ((n + 3)!)

theorem sum_infinite_series_eq_one_third :
  (∑' n, R n) = (1 / 3) :=
  sorry

end sum_infinite_series_eq_one_third_l565_565585


namespace distance_BC_l565_565701

-- Define the geometric problem
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := ((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt in
  let AC := ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt in
  let angle_BAC := 120 in
  AB = 2 ∧ AC = 3 ∧ angle_BAC = 120

-- Prove the distance between B and C
theorem distance_BC (A B C : ℝ × ℝ) (h : triangle_ABC A B C) : ((B.1 - C.1)^2 + (B.2 - C.2)^2).sqrt = real.sqrt 19 :=
  by sorry

end distance_BC_l565_565701


namespace part1_decreasing_interval_part2_extreme_point_condition_l565_565643

-- Define the function f(x) = x^3 - 6x^2 + 3x + t
def f (x t : ℝ) : ℝ := x^3 - 6 * x^2 + 3 * x + t

-- Define the derivative of f(x)
def f' (x t : ℝ) : ℝ := 3 * x^2 - 12 * x + 3

-- Part 1: Prove the interval of monotonic decrease for f(x)
theorem part1_decreasing_interval (t : ℝ) :
  ∃ a b : ℝ, (a = 2 - Real.sqrt 3) ∧ (b = 2 + Real.sqrt 3) ∧
  ∀ x, a < x ∧ x < b → f' x t < 0 :=
sorry

-- Define the function g(x) = e^x * f(x)
def g (x t : ℝ) : ℝ := Real.exp x * f x t

-- Define the simplified form of g'(x)
def g' (x t : ℝ) : ℝ := (x^3 - 9*x^2 + 9*x + t + 3) * Real.exp x

-- Define the function h(x) from the derivative of g'(x)
def h (x t : ℝ) : ℝ := x^3 - 9*x^2 + 9*x + t + 3

-- Calculate h(-1) and h(3)
def h1 (t : ℝ) : ℝ := t - 16
def h3 (t : ℝ) : ℝ := t - 24

-- Part 2: Prove the range of values for t such that g(x) has only one extreme point
theorem part2_extreme_point_condition (t : ℝ) :
  (h1 t) * (h3 t) ≥ 0 ↔ t ≤ 16 ∨ t ≥ 24 :=
sorry

end part1_decreasing_interval_part2_extreme_point_condition_l565_565643


namespace division_of_fractions_l565_565668

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l565_565668


namespace sum_of_divisors_of_24_l565_565106

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565106


namespace limit_sum_b_eq_one_third_l565_565986

noncomputable def a : ℕ → ℕ
| 0       := 1  -- this is Lean idiom for starting sequence at 0
| (n + 1) := a n + 3

def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

theorem limit_sum_b_eq_one_third : 
  (tendsto (λ (n : ℕ), ∑ i in finset.range n, b i) at_top (𝓝 (1/3))) := 
begin
  sorry
end

end limit_sum_b_eq_one_third_l565_565986


namespace gemma_change_l565_565596

/-- Gemma ordered four pizzas at $10 each, and she gave a $5 tip to the delivery person.
If she gives one fifty-dollar bill, then the change she will get back is $5. -/
theorem gemma_change :
  ∀ (num_pizzas cost_per_pizza tip payment : ℕ),
    num_pizzas = 4 → cost_per_pizza = 10 → tip = 5 → payment = 50 →
    payment - (num_pizzas * cost_per_pizza + tip) = 5 :=
by
  intros num_pizzas cost_per_pizza tip payment h1 h2 h3 h4
  have h_total_cost : num_pizzas * cost_per_pizza + tip = 45 
    := by rw [h1, h2, h3]; norm_num
  rw h4
  exact calc
    50 - 45 = 5 : by norm_num

end gemma_change_l565_565596


namespace find_f_4_l565_565300

-- Define the power function
def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

-- Given conditions
axiom passes_through_point (α : ℝ) : f 2 α = Real.sqrt 2

-- To prove
theorem find_f_4 : ∃ α : ℝ, f 4 α = 2 :=
by
  sorry

end find_f_4_l565_565300


namespace rhombus_adjacent_sides_equal_not_parallelogram_l565_565203

-- Definitions of the properties for rhombus and parallelogram
def is_rhombus (sides : List ℝ) : Prop := sides.length = 4 ∧ ∀ (a b : ℝ), a ∈ sides → b ∈ sides → a = b
def is_parallelogram (sides : List ℝ) : Prop := sides.length = 4 ∧ (sides.head = sides.getNth 2) ∧ (sides.getNth 1 = sides.getNth 3)

-- Properties given in the problem
def diagonals_bisect_each_other : Prop := sorry
def diagonals_are_equal : Prop := sorry
def adjacent_angles_are_equal : Prop := sorry
def adjacent_sides_are_equal (sides : List ℝ) : Prop := ∀ (a b : ℝ), a ∈ take 2 sides → b ∈ take 2 sides → a = b

-- The correct answer we need to prove
theorem rhombus_adjacent_sides_equal_not_parallelogram (sides : List ℝ) :
  (is_rhombus sides → adjacent_sides_are_equal sides) ∧ ¬(is_parallelogram sides → adjacent_sides_are_equal sides) :=
by
  sorry

end rhombus_adjacent_sides_equal_not_parallelogram_l565_565203


namespace marble_probability_l565_565845

theorem marble_probability :
  let multiples_6 := {x | x % 6 = 0 ∧ x ≥ 1 ∧ x ≤ 60},
      multiples_8 := {x | x % 8 = 0 ∧ x ≥ 1 ∧ x ≤ 60},
      multiples_24 := {x | x % 24 = 0 ∧ x ≥ 1 ∧ x ≤ 60} in
  let total_multiples := multiples_6 ∪ multiples_8 in
  ∃ num_marbles : ℕ, num_marbles = 60 ∧ 
  probability : ℚ, probability = ↑(total_multiples.card - multiples_24.card) / num_marbles ∧ 
  probability = 1 / 4 :=
by
  sorry

end marble_probability_l565_565845


namespace loaves_of_bread_20_slices_loaves_of_bread_30_not_enough_l565_565568

-- Define the constants and conditions
def slices_per_loaf : ℕ := 12

-- Define the theorems to prove
theorem loaves_of_bread_20_slices : ∀ (loaves : ℕ), loaves = 20 → slices_per_loaf * loaves = 240 := by
  intros loaves h
  rw h
  norm_num
  sorry

theorem loaves_of_bread_30_not_enough : ∀ (loaves : ℕ), loaves = 30 → slices_per_loaf * loaves < 385 := by
  intros loaves h
  rw h
  norm_num
  sorry

end loaves_of_bread_20_slices_loaves_of_bread_30_not_enough_l565_565568


namespace min_value_abc_l565_565734

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) ^ 2 + (b / c) ^ 2 + (c / a) ^ 2 ≥ 3 :=
begin
  sorry
end

end min_value_abc_l565_565734


namespace inequality_solution_l565_565888

noncomputable def operation (a b : ℝ) : ℝ := (a + 3 * b) - a * b

theorem inequality_solution (x : ℝ) : operation 5 x < 13 → x > -4 := by
  sorry

end inequality_solution_l565_565888


namespace solution_set_x_f_x_lt_0_l565_565352

variable {α : Type*} [HasLt α] [Add α] [Neg α] [Zero α]

noncomputable def f : α → α

variable (x : α)

-- Conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom increasing_f : ∀ {x y}, 0 < x → x < y → f x < f y
axiom f_neg3 : f (-3) = 0

theorem solution_set_x_f_x_lt_0 : 
  { x | x * f x < 0 } = { x | (0 < x ∧ x < 3) ∨ (-3 < x ∧ x < 0) } :=
sorry

end solution_set_x_f_x_lt_0_l565_565352


namespace complete_the_square_const_l565_565141

theorem complete_the_square_const : ∀ (x : ℝ), (x^2 - 2*x = 2) → (x^2 - 2*x + 1 = 3) :=
begin
  intro x,
  intro h,
  sorry
end

end complete_the_square_const_l565_565141


namespace complex_equation_is_hyperbola_l565_565145

noncomputable def problem (z : ℂ) : Prop :=
  (complex.abs (z - 3) = complex.abs (z + 3) - 1) →
  (∃ a b c : ℝ,
    set_of (λ (z : ℂ), complex.abs (z - 3) = complex.abs (z + 3) - 1) = 
    {z | (z.re - a)^2 / c^2 - (z.im - b)^2 / (1 / c)^2 = 1} ∧ c = 1/2)

-- Problem statement in Lean 4
theorem complex_equation_is_hyperbola (z : ℂ) : problem z :=
sorry

end complex_equation_is_hyperbola_l565_565145


namespace ratio_C_D_l565_565886

-- Define the array and sums
variable {a : ℕ → ℕ → ℝ}
variables (S_i : ℕ → ℝ) (T_j : ℕ → ℝ) (C D : ℝ)

-- Define the conditions
def condition1 := ∀ i, 1 ≤ i ∧ i ≤ 50 → S_i i = ∑ j in Finset.range 100, a i j
def condition2 := ∀ j, 1 ≤ j ∧ j ≤ 100 → T_j j = ∑ i in Finset.range 50, a i j
def condition3 := C = (∑ i in Finset.range 50, S_i i) / 50
def condition4 := D = (∑ j in Finset.range 100, T_j j) / 100
def condition5 := (∑ i in Finset.range 50, S_i i) = (∑ j in Finset.range 100, T_j j)

-- The statement that needs to be proved
theorem ratio_C_D (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : C / D = 2 := 
by 
sorry

end ratio_C_D_l565_565886


namespace divisor_chain_properties_l565_565744

-- Definitions
def is_divisor_chain (seq : List ℕ) (x : ℕ) : Prop :=
  seq.head = 1 ∧ seq.last = x ∧ seq.pairwise (· < ·) ∧ ∀ i ∈ List.init seq, i ∣ List.get seq (List.indexOf i seq + 1)

def L (x : ℕ) : ℕ := sorry
def R (x : ℕ) : ℕ := sorry

-- Main theorem
theorem divisor_chain_properties (k m n : ℕ) :
  let x := 5^k * 31^m * 1990^n in
  L(x) = 3 * n + k + m ∧ R(x) = (Nat.factorial (3 * n + k + m)) / ((Nat.factorial n) * (Nat.factorial n) * (Nat.factorial m) * (Nat.factorial (k + n))) :=
by
  admit

end divisor_chain_properties_l565_565744


namespace xiao_ming_corrected_polynomial_value_at_minus_two_corrected_polynomial_l565_565821

theorem xiao_ming_corrected_polynomial :
  ∀ (x : ℝ), let mistaken_polynomial := (6 * x^2 - 3 * x + 5),
                  subtracted_polynomial := (2 * x^2 - 4 * x + 7),
                  corrected_polynomial := (4 * x^2 + x - 2) - subtracted_polynomial in
    (mistaken_polynomial - subtracted_polynomial) + subtracted_polynomial = corrected_polynomial := by
  sorry

theorem value_at_minus_two_corrected_polynomial :
  (2 * (-2)^2 + 5 * (-2) - 9) = -11 := by
  sorry

end xiao_ming_corrected_polynomial_value_at_minus_two_corrected_polynomial_l565_565821


namespace find_f_neg5_l565_565941

theorem find_f_neg5 (a b : ℝ) (Sin : ℝ → ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b * (Sin x) ^ 3 + 1)
  (h_f5 : f 5 = 7) :
  f (-5) = -5 := 
by
  sorry

end find_f_neg5_l565_565941


namespace sum_of_divisors_of_24_l565_565103

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565103


namespace value_of_stamp_collection_l565_565572

theorem value_of_stamp_collection 
  (n m : ℕ) (v_m : ℝ)
  (hn : n = 18) 
  (hm : m = 6)
  (hv_m : v_m = 15)
  (uniform_value : ∀ (k : ℕ), k ≤ m → v_m / m = v_m / k):
  ∃ v_total : ℝ, v_total = 45 :=
by 
  sorry

end value_of_stamp_collection_l565_565572


namespace triangle_area_inscribed_circle_l565_565420

theorem triangle_area_inscribed_circle (a r : ℝ) (h : r < a / 2) :
  (area_of_triangle_with_inscribed_circle a r = a^2 * r / (a - r)) :=
sorry

end triangle_area_inscribed_circle_l565_565420


namespace square_root_value_l565_565140

-- Define the problem conditions
def x : ℝ := 5

-- Prove the solution
theorem square_root_value : (Real.sqrt (x - 3)) = Real.sqrt 2 :=
by
  -- Proof steps skipped
  sorry

end square_root_value_l565_565140


namespace largest_C_l565_565242

open Function

variable (A : Fin n → Set ℝ) (n : ℕ)

def nonempty_intersection (A : Set ℝ) : Prop := ∃ x, x ∈ A

def triples_property (A : Fin n → Set ℝ) : Prop := 
  let triples := {T : Finset (Fin n) // T.card = 3}
  (∃ count triple ∈ triples, nonempty_intersection (A triple)) ∧
  (2 * (∑ triple in triples, if nonempty_intersection (A triple) then 1 else 0) ≥ triples.card)

theorem largest_C (A : Fin n → Set ℝ) 
  (h : triples_property A) : ∃ C > 0, C ≤ 1 / Real.sqrt 6 := 
    sorry

end largest_C_l565_565242


namespace common_terms_count_l565_565623

theorem common_terms_count (β : ℕ) (h1 : β = 55) (h2 : β + 1 = 56) : 
  ∃ γ : ℕ, γ = 6 :=
by
  sorry

end common_terms_count_l565_565623


namespace find_S_l565_565158

theorem find_S (R S T : ℝ) (c : ℝ) 
  (h1 : R = c * S / T)
  (h2 : R = 4 / 3)
  (h3 : S = 3 / 7)
  (h4 : T = 9 / 14)
  (h5 : c = 2) :
  (∃ S' : ℝ, S' = 28 ∧ R = sqrt 98 ∧ T = sqrt 32) := 
sorry

end find_S_l565_565158


namespace minimum_red_beads_l565_565491

theorem minimum_red_beads (n : ℕ) (r : ℕ) (necklace : ℕ → Prop) :
  (necklace = λ k, n * k + r) 
  → (∀ i, (segment_contains_blue i 8 → segment_contains_red i 4))
  → (cyclic_beads necklace)
  → r ≥ 29 :=
by
  sorry

-- Definitions to support the theorem
def segment_contains_blue (i : ℕ) (b : ℕ) : Prop := 
sorry -- Placeholder for the predicate that checks if a segment contains exactly 'b' blue beads.

def segment_contains_red (i : ℕ) (r : ℕ) : Prop := 
sorry -- Placeholder for the predicate that checks if a segment contains at least 'r' red beads.

def cyclic_beads (necklace : ℕ → Prop) : Prop := 
sorry -- Placeholder for the property that defines the necklace as cyclic.

end minimum_red_beads_l565_565491


namespace sqrt_expression_equals_l565_565230

theorem sqrt_expression_equals :
  (Real.root 4 (7 + 3 * Real.sqrt 5)) + (Real.root 4 (7 - 3 * Real.sqrt 5)) = Real.root 4 26 :=
by sorry

end sqrt_expression_equals_l565_565230


namespace smallest_k_for_coloring_l565_565265

theorem smallest_k_for_coloring (n : ℕ) (h : n ≥ 2) : ∃ k, (∀ coloring : (Fin n → Fin n → Fin k), 
  (∃ (r₁ r₂ c₁ c₂ : Fin n), r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧ 
    (coloring r₁ c₁ ≠ coloring r₁ c₂ ∧ coloring r₁ c₁ ≠ coloring r₂ c₁ ∧ 
     coloring r₁ c₂ ≠ coloring r₂ c₁ ∧ coloring r₂ c₁ ≠ coloring r₂ c₂))) ∧ 
  ∀ k', (k' < k → ∃ coloring : (Fin n → Fin n → Fin k'), 
  ∀ r₁ r₂ c₁ c₂ : Fin n, ¬ (r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧ 
    (coloring r₁ c₁ ≠ coloring r₁ c₂ ∧ coloring r₁ c₁ ≠ coloring r₂ c₁ ∧ 
     coloring r₁ c₂ ≠ coloring r₂ c₁ ∧ coloring r₂ c₁ ≠ coloring r₂ c₂)))) 
  := 
begin
  use 2 * n,
  sorry -- Proof omitted
end

end smallest_k_for_coloring_l565_565265


namespace z_in_fourth_quadrant_l565_565973

noncomputable def z (a b : ℝ) : ℂ := a + b * complex.I

def quadrant_fourth (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem z_in_fourth_quadrant :
  ∀ (z : ℂ), (z * (1 + complex.I) * (-2 * complex.I) = 1) → quadrant_fourth(z) :=
begin
  intro z,
  intro h,
  -- proof goes here; omitted for now
  sorry
end

end z_in_fourth_quadrant_l565_565973


namespace minimum_value_x_l565_565984

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end minimum_value_x_l565_565984


namespace part1_part2_smallest_positive_period_part2_intervals_of_monotonic_increase_l565_565640

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 - Real.cos (2 * x + Real.pi / 2)

theorem part1 : f (Real.pi / 8) = 1 := 
sorry

theorem part2_smallest_positive_period : ∀ x : ℝ, f (x + Real.pi) = f x := 
sorry

theorem part2_intervals_of_monotonic_increase :
  ∀ (k : ℤ) (x : ℝ), (k * Real.pi - Real.pi / 8) ≤ x ∧ x ≤ (k * Real.pi + 3 * Real.pi / 8) → 
  f x > (∀ y : ℝ, f y := 
sorry

end part1_part2_smallest_positive_period_part2_intervals_of_monotonic_increase_l565_565640


namespace correct_statement_l565_565819

-- Definitions matching the conditions
def angle_in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def vector_length_equal {n : Type*} [linear_ordered_field n] (v₁ v₂ : n) : Prop := abs (v₁) = abs (v₂) ∧ v₁ = v₂

-- Theorem to be proven given the conditions
theorem correct_statement :
  (∀ θ, angle_in_first_quadrant θ → θ < π / 2) →
  (∀ θ, θ < 90 → acute_angle θ) →
  (∀ (v₁ v₂ : ℝ), vector_length_equal v₁ v₂ → v₁ = v₂) →
  (∀ θ, acute_angle θ → angle_in_first_quadrant θ) :=
by
  intros h1 h2 h3 h4
  sorry

end correct_statement_l565_565819


namespace sum_of_divisors_of_24_l565_565015

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565015


namespace consumption_percentage_l565_565861

theorem consumption_percentage 
  (x y : ℝ)
  (h : y = 0.66 * x + 1.562)
  (hy : y = 7.675) :
  (y / x) * 100 ≈ 83 :=
by
  sorry

end consumption_percentage_l565_565861


namespace exemplary_number_count_l565_565554

def is_exemplary (n : ℕ) : Prop :=
  (n < 10) ∨
  (n.digits.to_list = n.digits.to_list.sort) ∨
  (n.digits.to_list = (n.digits.to_list.sort.reverse)) ∨
  (∀i < n.digits.to_list.length - 1, 
    (n.digits.to_list[i] < n.digits.to_list[i+1] ∧ n.digits.to_list[i+1] < n.digits.to_list[i+2]) ∨ 
    (n.digits.to_list[i] > n.digits.to_list[i+1] ∧ n.digits.to_list[i+1] > n.digits.to_list[i+2]))

theorem exemplary_number_count : ∃ count, count = 1505 ∧
  count = (∑ k in (range 10 ∪
          {l : ℕ | l > 1 ∧ 
          (is_strictly_increasing l.digits.to_list ∨
           is_strictly_decreasing l.digits.to_list ∨
           is_alternating_sequence l.digits.to_list)
           }.to_finset), 1) :=
sorry

end exemplary_number_count_l565_565554


namespace trigonometric_expression_l565_565962

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.sin α = 3 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) π) :
  (Real.cos (2 * α) / (Real.sqrt 2 * Real.sin (α + π / 4))) = -7 / 5 := 
sorry

end trigonometric_expression_l565_565962


namespace leading_coefficient_polynomial_l565_565558

theorem leading_coefficient_polynomial :
    let p := 5 * (x^5 - 2 * x^3 + x) - 8 * (x^5 + x^3 + 3 * x) + 6 * (3 * x^5 - x^2 + 4)
    leading_coeff p = 15 := by
    sorry

end leading_coefficient_polynomial_l565_565558


namespace part1_part2_l565_565679

def g (x : ℝ) := x
def h (x : ℝ) := 1 / x 
def f (x : ℝ) := x + 1 / x

def is_t_opp (f : ℝ → ℝ) (t : ℝ) (M : Set ℝ) : Prop :=
∀ x ∈ M, (x - t) ∈ D → f (x - t) < f x

theorem part1 : is_t_opp g 3 (Set.Icc 4 6) ∧ ¬is_t_opp h 3 (Set.Icc 4 6) :=
sorry

theorem part2 : ∃ n : ℕ, is_t_opp f n (Set.Icc (-2 : ℝ) (-1 : ℝ)) ∧ 
  ∀ m : ℕ, m < n → ¬is_t_opp f m (Set.Icc (-2 : ℝ) (-1 : ℝ)) :=
sorry

end part1_part2_l565_565679


namespace number_of_sets_l565_565472

-- Define the conditions
def satisfied_set (S : set ℕ) : Prop :=
  ∀ x ∈ S, (14 - x) ∈ S

-- Statement of the problem
theorem number_of_sets (h : ∃ S : set ℕ, S ≠ ∅ ∧ satisfied_set S) : 
  ∃ (n : ℕ), n = 127 := 
sorry

end number_of_sets_l565_565472


namespace sum_of_divisors_of_24_l565_565094

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565094


namespace amount_each_friend_pays_l565_565473

-- Define conditions in Lean
def total_bill : ℝ := 400
def discount_rate : ℝ := 0.05
def num_friends : ℕ := 6

-- Define the question and the correct answer
def discounted_bill : ℝ := total_bill * (1 - discount_rate)
def each_friend_pays : ℝ := discounted_bill / num_friends

-- Prove that each friend pays $63.33
theorem amount_each_friend_pays : each_friend_pays = 63.33 := 
by
  sorry

end amount_each_friend_pays_l565_565473


namespace problem_solution_l565_565244

def f (x y : ℝ) : ℝ :=
  (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2)

theorem problem_solution :
  (∀ x y : ℝ, f x y + f y x = 0) ∧
  (∀ x y : ℝ, f x (x + y) + f y (x + y) = 0) :=
by
  sorry

end problem_solution_l565_565244


namespace min_value_x_l565_565982

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end min_value_x_l565_565982


namespace sum_of_divisors_of_twenty_four_l565_565124

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565124


namespace vector_subtraction_l565_565260

-- Definitions of given conditions
def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

-- Definition of vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_subtraction : vector_sub OB OA = (-5, 3) :=
by 
  -- The proof would go here.
  sorry

end vector_subtraction_l565_565260


namespace remainder_of_sum_div_18_l565_565451

theorem remainder_of_sum_div_18 :
  let nums := [11065, 11067, 11069, 11071, 11073, 11075, 11077, 11079, 11081]
  let residues := [1, 3, 5, 7, 9, 11, 13, 15, 17]
  (nums.sum % 18) = 9 := by
    sorry

end remainder_of_sum_div_18_l565_565451


namespace percent_red_jelly_beans_remaining_l565_565484

theorem percent_red_jelly_beans_remaining 
  (N : ℕ)
  (hN_pos : N > 0)
  (initial_red : ℕ := N / 4)
  (initial_blue : ℕ := 3 * N / 4)
  (removed_red : ℕ := 3 * initial_red / 4)
  (removed_blue : ℕ := initial_blue / 4) :
  let remaining_red := initial_red - removed_red,
      remaining_blue := initial_blue - removed_blue,
      total_remaining := remaining_red + remaining_blue in
  (remaining_red * 100 / total_remaining) = 10 := 
by 
  -- Proof to be provided
  sorry

end percent_red_jelly_beans_remaining_l565_565484


namespace number_of_points_P_l565_565790

noncomputable def ellipse : set (ℝ × ℝ) := { p | let x := p.1, y := p.2 in x^2 / 16 + y^2 / 9 = 1 }

def line : set (ℝ × ℝ) := { p | let x := p.1, y := p.2 in x / 4 + y / 3 = 1 }

def points_of_intersection : set (ℝ × ℝ) := { p | p ∈ ellipse ∧ p ∈ line }

def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (4, 0)

def point_on_ellipse (t : ℝ) : ℝ × ℝ := (4 * Real.cos t, 3 * Real.sin t)

def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  let x1 := A.1, y1 := A.2
  let x2 := B.1, y2 := B.2
  let x3 := P.1, y3 := P.2
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem number_of_points_P :
  ∃ (P1 P2 : ℝ × ℝ), P1 ∈ ellipse ∧ P2 ∈ ellipse ∧
  (area_of_triangle P1 = 3 ∧ area_of_triangle P2 = 3) ∧ 
  P1 ≠ P2 :=
sorry

end number_of_points_P_l565_565790


namespace sin_alpha_value_l565_565599

-- Given conditions
variables (α : ℝ) (h1 : Real.tan α = -5 / 12) (h2 : π / 2 < α ∧ α < π)

-- Assertion to prove
theorem sin_alpha_value : Real.sin α = 5 / 13 :=
by
  -- Proof goes here
  sorry

end sin_alpha_value_l565_565599


namespace sum_of_divisors_of_24_l565_565126

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565126


namespace calculate_observed_price_l565_565146

-- Define the conditions and required price
def online_store_commission := 0.20
def cost_from_producer := 15.0
def desired_profit_percentage := 0.10
def desired_earnings_without_commission := cost_from_producer * (1 + desired_profit_percentage)

-- Calculate the observed price
def observed_price (P : ℝ) : Prop :=
  (1 - online_store_commission) * P = desired_earnings_without_commission

theorem calculate_observed_price : observed_price 20.63 :=
by
  -- Insert the proof here
  sorry

end calculate_observed_price_l565_565146


namespace reduction_in_carbon_emission_l565_565693

theorem reduction_in_carbon_emission 
  (population : ℕ)
  (carbon_per_car : ℕ)
  (carbon_per_bus : ℕ)
  (bus_capacity : ℕ)
  (percentage_take_bus : ℝ)
  (initial_cars : ℕ := population) :
  (initial_cars * carbon_per_car - 
  (population * percentage_take_bus.floor.to_nat * carbon_per_bus +
   (population - population * percentage_take_bus.floor.to_nat) * carbon_per_car)) = 100 :=
by
  -- Given conditions
  let initial_emission := initial_cars * 10 -- Total carbon emission from all cars
  let people_take_bus := (population * 0.25).to_nat -- 25% of people switch to the bus
  let remaining_cars := population - people_take_bus -- People still driving
  let new_emission := 100 + remaining_cars * 10 -- New total emission
  have reduction := initial_emission - new_emission -- Reduction in carbon emission
  -- Correct answer
  exact reduction = 100

end reduction_in_carbon_emission_l565_565693


namespace largest_a_for_integer_solution_l565_565915

theorem largest_a_for_integer_solution :
  ∃ a : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a * x + 3 * y = 1) ∧ (∀ a' : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a' * x + 3 * y = 1) → a' ≤ a) ∧ a = 1 :=
sorry

end largest_a_for_integer_solution_l565_565915


namespace total_area_of_combined_shape_l565_565957

def area_of_hexagon_and_triangle (s : ℝ) : ℝ := 
  let area_hex := (3 * Real.sqrt 3 / 2) * s^2
  let area_tri := (Real.sqrt 3 / 4) * s^2
  area_hex + area_tri

theorem total_area_of_combined_shape 
  {B C D E F : ℝ} 
  (h1 : ∃ (s : ℝ), s = 2) 
  (h2 : B = 2) 
  (h3 : C = 2) 
  (h4 : BDEF ∈ square) : 
  area_of_hexagon_and_triangle 2 = 7 * Real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end total_area_of_combined_shape_l565_565957


namespace sum_geometric_series_nine_l565_565954

noncomputable def geometric_series_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = a 0 * (1 - a 1 ^ n) / (1 - a 1)

theorem sum_geometric_series_nine
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (S_3 : S 3 = 12)
  (S_6 : S 6 = 60) :
  S 9 = 252 := by
  sorry

end sum_geometric_series_nine_l565_565954


namespace cash_realized_without_brokerage_l565_565779

theorem cash_realized_without_brokerage
  (C : ℝ)
  (h1 : (1 / 4) * (1 / 100) = 1 / 400)
  (h2 : C + (C / 400) = 108) :
  C = 43200 / 401 :=
by
  sorry

end cash_realized_without_brokerage_l565_565779


namespace mass_of_circle_is_one_l565_565426

variable (x y z : ℝ)

theorem mass_of_circle_is_one (h1 : 3 * y = 2 * x)
                              (h2 : 2 * y = x + 1)
                              (h3 : 5 * z = x + y)
                              (h4 : true) : z = 1 :=
sorry

end mass_of_circle_is_one_l565_565426


namespace solve_system_l565_565393

theorem solve_system :
  ∃ x y : ℝ, 
    3 * x ≥ 2 * y + 16 ∧ 
    x^4 + 2 * x^2 * y^2 + y^4 + 25 - 26 * x^2 - 26 * y^2 = 72 * x * y ∧ 
    (x = 6 ∧ y = 1) :=
begin
  sorry
end

end solve_system_l565_565393


namespace sufficient_condition_necessary_condition_necessary_and_sufficient_condition_l565_565934

variables {α : Type} (p q : α → Prop)

theorem sufficient_condition (h : ∀ x, p x → q x) :
  ∀ x, p x → q x := 
by {
  intro x,
  exact h x
}

theorem necessary_condition (h : ∀ x, q x → p x) :
  ∀ x, q x → p x := 
by {
  intro x,
  exact h x
}

theorem necessary_and_sufficient_condition (h : ∀ x, p x ↔ q x) :
  (∀ x, p x → q x) ∧ (∀ x, q x → p x) := 
by {
  split;
  intro x;
  have : p x ↔ q x := h x;
  finish
}

end sufficient_condition_necessary_condition_necessary_and_sufficient_condition_l565_565934


namespace sum_of_cubes_of_roots_eq_1_l565_565152

theorem sum_of_cubes_of_roots_eq_1 (a : ℝ) (x1 x2 : ℝ) :
  (x1^2 + a * x1 + a + 1 = 0) → 
  (x2^2 + a * x2 + a + 1 = 0) → 
  (x1 + x2 = -a) → 
  (x1 * x2 = a + 1) → 
  (x1^3 + x2^3 = 1) → 
  a = -1 :=
sorry

end sum_of_cubes_of_roots_eq_1_l565_565152


namespace brenda_age_problem_l565_565865

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end brenda_age_problem_l565_565865


namespace imaginary_part_of_square_l565_565409

theorem imaginary_part_of_square (z : ℂ) (h : z = 1 - complex.i) : (z^2).im = -2 := by
  sorry

end imaginary_part_of_square_l565_565409


namespace circle_filling_ways_l565_565206

theorem circle_filling_ways : 
  ∃! (a b c d e : ℕ), 
    {a, b, c, d, e} = {2, 3, 4, 5, 6} ∧
    (|a - 1| ≥ 2) ∧ (|b - a| ≥ 2) ∧ (|c - b| ≥ 2) ∧ (|d - c| ≥ 2) ∧ (|e - d| ≥ 2) ∧ (|e - 1| ≥ 2) ∧
    ((a = 3 ∧ b = 5 ∧ c = 4 ∧ d = 2 ∧ e = 6) ∨
     (a = 3 ∧ b = 5 ∧ c = 6 ∧ d = 2 ∧ e = 4) ∨
     (a = 5 ∧ b = 3 ∧ c = 4 ∧ d = 6 ∧ e = 2)) := 
sorry

end circle_filling_ways_l565_565206


namespace coloring_property_l565_565890

def ν2 (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.find (λ k, 2^k ∣ n ∧ ¬ 2^(k + 1) ∣ n)

theorem coloring_property 
  (a b : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b) :
  (∀ (f : ℕ → bool), ∃ x y, f x = ff ∧ f y = ff ∧ (x - y = a) ∨
                    ∃ x y, f x = tt ∧ f y = tt ∧ (x - y = b)) ↔
    ν2 a ≠ ν2 b := 
sorry

end coloring_property_l565_565890


namespace exists_lambda_l565_565913

noncomputable def ellipse : Type := sorry

variables {A B D M N : ellipse} 
variables {k1 k2 : ℝ}

structure EllipseProperties (e : ellipse) :=
  (is_vertex : B)
  (line_through_origin_intersects : A)
  (AD_perpendicular_AB : A)
  (BD_intersects_axes : M × N)
  (slopes_definition : k1 = k2)

theorem exists_lambda (e : ellipse) (props : EllipseProperties e) :
  ∃ λ : ℝ, k1 = λ * k2 ∧ λ = -1/2 := sorry

end exists_lambda_l565_565913


namespace gcd_36_60_l565_565338

theorem gcd_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l565_565338


namespace acute_angles_union_less_than_90_sub_is_less_than_90_l565_565302

def is_first_quadrant (α : ℝ) : Prop := ∃ (k : ℤ), k * 360 < α ∧ α < k * 360 + 90
def is_acute_angle (β : ℝ) : Prop := 0 < β ∧ β < 90
def is_less_than_90 (γ : ℝ) : Prop := γ < 90

theorem acute_angles_union_less_than_90_sub_is_less_than_90 :
  (∀ β : ℝ, is_acute_angle β → is_less_than_90 β) →
  (∀ γ : ℝ, is_less_than_90 γ → is_less_than_90 γ) →
  ∀ x : ℝ, (is_acute_angle x ∨ is_less_than_90 x) → is_less_than_90 x :=
by
  intros h_acute h_less x hx
  cases hx with h_xacute h_xless
  . apply h_acute; exact h_xacute
  . exact h_xless

end acute_angles_union_less_than_90_sub_is_less_than_90_l565_565302


namespace find_original_prices_and_discount_l565_565143

theorem find_original_prices_and_discount :
  ∃ x y a : ℝ,
  (6 * x + 5 * y = 1140) ∧
  (3 * x + 7 * y = 1110) ∧
  (((9 * x + 8 * y) - 1062) / (9 * x + 8 * y) = a) ∧
  x = 90 ∧
  y = 120 ∧
  a = 0.4 :=
by
  sorry

end find_original_prices_and_discount_l565_565143


namespace negation_of_exists_gt_1_l565_565839

theorem negation_of_exists_gt_1 :
  (∀ x : ℝ, x ≤ 1) ↔ ¬ (∃ x : ℝ, x > 1) :=
sorry

end negation_of_exists_gt_1_l565_565839


namespace extreme_points_and_difference_l565_565354

noncomputable def f (x a : ℝ) : ℝ := exp x - (1/2) * a * x^2 - f 0 * x

theorem extreme_points_and_difference (a : ℝ) (h_a : a > 1) :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ (∀ x : ℝ, f x a = f x 1 + (a - 1) * (x - x1) * (x - x2)) ∧
  (x2 - x1 increases as a increases) :=
sorry

end extreme_points_and_difference_l565_565354


namespace union_A_B_complement_A_l565_565989

-- Definition of Universe U
def U : Set ℝ := Set.univ

-- Definition of set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Definition of set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem 1: Proving the union A ∪ B
theorem union_A_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := 
sorry

-- Theorem 2: Proving the complement of A with respect to U
theorem complement_A : (U \ A) = {x | x < -1 ∨ x > 3} := 
sorry

end union_A_B_complement_A_l565_565989


namespace aston_found_pages_l565_565874

-- Given conditions
def pages_per_comic := 25
def initial_untorn_comics := 5
def total_comics_now := 11

-- The number of pages Aston found on the floor
theorem aston_found_pages :
  (total_comics_now - initial_untorn_comics) * pages_per_comic = 150 := 
by
  sorry

end aston_found_pages_l565_565874


namespace cost_per_pint_l565_565882

def pint_cost (total_cost: ℝ) (num_pints: ℕ) : ℝ := total_cost / num_pints

theorem cost_per_pint :
  ∀ (gallon_cost: ℝ) (saving: ℝ) (num_pints: ℕ),
    gallon_cost = 55 →
    saving = 9 →
    num_pints = 8 →
    pint_cost (gallon_cost + saving) num_pints = 8 :=
by
  intros
  sorry

end cost_per_pint_l565_565882


namespace domain_of_f_l565_565911

noncomputable def f (x : ℝ) : ℝ := (5 * x - 2) / Real.sqrt (x^2 - 3 * x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ (f_x : ℝ), f x = f_x} = {x : ℝ | (x < -1) ∨ (x > 4)} :=
by
  sorry

end domain_of_f_l565_565911


namespace customer_ordered_bags_l565_565010

def bags_per_batch : Nat := 10
def initial_bags : Nat := 20
def days : Nat := 4
def batches_per_day : Nat := 1

theorem customer_ordered_bags : 
  initial_bags + days * batches_per_day * bags_per_batch = 60 :=
by
  sorry

end customer_ordered_bags_l565_565010


namespace regina_earnings_l565_565387

theorem regina_earnings : 
  let cows := 20 in
  let pigs := 4 * cows in
  let goats := pigs / 2 in
  let chickens := 2 * cows in
  let rabbits := 30 in
  let cow_price := 800 in
  let pig_price := 400 in
  let goat_price := 600 in
  let chicken_price := 50 in
  let rabbit_price := 25 in
  (cows * cow_price + pigs * pig_price + goats * goat_price + chickens * chicken_price + rabbits * rabbit_price) = 74750 :=
by sorry

end regina_earnings_l565_565387


namespace gabby_money_needed_l565_565931

def make_up_price : ℝ := 65
def skin_care_price_eur : ℝ := 40
def hair_tool_price_gbp : ℝ := 50
def initial_savings_usd : ℝ := 35
def initial_savings_eur : ℝ := 10
def mom_money_usd : ℝ := 20
def dad_money_gbp : ℝ := 20
def chores_money_eur : ℝ := 15
def exchange_rate_usd_to_eur : ℝ := 0.85
def exchange_rate_usd_to_gbp : ℝ := 0.75

def skincare_price_usd : ℝ := skin_care_price_eur / exchange_rate_usd_to_eur
def hair_tool_price_usd : ℝ := hair_tool_price_gbp / exchange_rate_usd_to_gbp
def savings_eur_to_usd : ℝ := initial_savings_eur / exchange_rate_usd_to_eur
def chores_money_usd : ℝ := chores_money_eur / exchange_rate_usd_to_eur
def dad_money_usd : ℝ := dad_money_gbp / exchange_rate_usd_to_gbp

def total_cost_usd : ℝ := make_up_price + skincare_price_usd + hair_tool_price_usd
def total_savings_usd : ℝ := initial_savings_usd + mom_money_usd + savings_eur_to_usd + chores_money_usd + dad_money_usd
def additional_money_needed : ℝ := total_cost_usd - total_savings_usd

theorem gabby_money_needed : additional_money_needed = 67.65 := by
  sorry

end gabby_money_needed_l565_565931


namespace curve_C_rect_eq_min_distance_AB_l565_565610

-- Definition of the parametric equation of the line l.
def parametric_line_equation (t θ : ℝ) : ℝ × ℝ :=
  (1/2 + t * Real.cos θ, t * Real.sin θ)

-- Definition of the polar equation of curve C.
def polar_eq_curve_C (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ ^ 2 = 2 * Real.cos θ

-- Conversion from polar to rectangular coordinates.
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Problem 1: Prove the rectangular coordinate equation of curve C.
theorem curve_C_rect_eq (x y : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) (hC : polar_eq_curve_C (Real.sqrt (x^2 + y^2)) θ) :
  y^2 = 2 * x :=
sorry

-- Problem 2: Prove the minimum value of |AB| as θ varies.
theorem min_distance_AB (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  (∀ t₁ t₂ : ℝ, 
    parametric_line_equation t₁ θ = polar_to_rect (Real.sqrt ((1/2 + t₁ * Real.cos θ)^2 + (t₁ * Real.sin θ)^2)) θ ∧
    parametric_line_equation t₂ θ = polar_to_rect (Real.sqrt ((1/2 + t₂ * Real.cos θ)^2 + (t₂ * Real.sin θ)^2)) θ) →
  |(λ (θ : ℝ), 2 / Real.sin θ ^ 2) θ| ≥ 2 :=
sorry

end curve_C_rect_eq_min_distance_AB_l565_565610


namespace mass_percentage_Cr_in_compound_l565_565891

-- Assume the molar masses as conditions
def molar_mass_H2CrO4 : ℝ := 118.02
def molar_mass_KMnO4 : ℝ := 158.04
def molar_mass_combined : ℝ := 244.02
def molar_mass_Cr : ℝ := 52.00

-- Define the mass percentage calculation
def mass_percentage_Cr : ℝ :=
  (molar_mass_Cr / molar_mass_combined) * 100

-- Statement to be proven
theorem mass_percentage_Cr_in_compound : mass_percentage_Cr = 21.31 :=
  by
    sorry

end mass_percentage_Cr_in_compound_l565_565891


namespace total_ants_correct_l565_565509

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end total_ants_correct_l565_565509


namespace arithmetic_sequence_eighth_term_l565_565785

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Specify the given conditions
def a1 : ℚ := 10 / 11
def a15 : ℚ := 8 / 9

-- Prove that the eighth term is equal to 89 / 99
theorem arithmetic_sequence_eighth_term :
  ∃ d : ℚ, arithmetic_sequence a1 d 15 = a15 →
             arithmetic_sequence a1 d 8 = 89 / 99 :=
by
  sorry

end arithmetic_sequence_eighth_term_l565_565785


namespace units_digit_is_0_most_likely_l565_565567

-- Define the variables and settings
constant Jack_draws : ℕ
constant Jill_draws : ℕ
axiom jack_range : 1 ≤ Jack_draws ∧ Jack_draws ≤ 15
axiom jill_range : 1 ≤ Jill_draws ∧ Jill_draws ≤ 15

-- The final theorem: Prove that the most likely units digit of the sum of Jack's and Jill's draw is 0
theorem units_digit_is_0_most_likely :
  ∃ (n : ℕ), (0 ≤ n ∧ n < 10) ∧
  (∀ (d : ℕ), (0 ≤ d ∧ d < 10) → (count (λ (x : ℕ), (Jack_draws + Jill_draws) % 10 = d) (range 226) ≤ count (λ (x : ℕ), (Jack_draws + Jill_draws) % 10 = 0) (range 226))) :=
begin
  sorry
end

end units_digit_is_0_most_likely_l565_565567


namespace cost_price_per_meter_l565_565193

theorem cost_price_per_meter
    (meters_sold : ℕ)
    (total_selling_price : ℕ)
    (profit_per_meter : ℕ)
    (total_profit := profit_per_meter * meters_sold)
    (total_cost_price := total_selling_price - total_profit)
    (cost_price_per_meter := total_cost_price / meters_sold)
    (meters_sold = 60)
    (total_selling_price = 8400)
    (profit_per_meter = 12) :
    cost_price_per_meter = 128 :=
by
  sorry

end cost_price_per_meter_l565_565193


namespace sum_of_divisors_of_24_is_60_l565_565063

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565063


namespace maximize_quotient_l565_565932

theorem maximize_quotient (a b : ℝ) (ha : 210 ≤ a ∧ a ≤ 430) (hb : 590 ≤ b ∧ b ≤ 1190) :
  (∃ a b, 210 ≤ a ∧ a ≤ 430 ∧ 590 ≤ b ∧ b ≤ 1190 ∧ b / a = 119 / 21) :=
begin
  sorry
end

end maximize_quotient_l565_565932


namespace number_of_even_digits_in_base4_145_l565_565916

theorem number_of_even_digits_in_base4_145 :
  (let n := 145 in let d := Nat.digits 4 n in (d.filter (λ x => x % 2 = 0)).length = 2) :=
by
  let n := 145 
  let d := Nat.digits 4 n
  have h1 : d = [1, 0, 1, 2] := by sorry
  have h2 : (d.filter (λ x => x % 2 = 0)) = [0, 2] := by sorry
  show (d.filter (λ x => x % 2 = 0)).length = 2 from by sorry

end number_of_even_digits_in_base4_145_l565_565916


namespace chickens_count_l565_565226

-- Define conditions
def cows : Nat := 4
def sheep : Nat := 3
def bushels_per_cow : Nat := 2
def bushels_per_sheep : Nat := 2
def bushels_per_chicken : Nat := 3
def total_bushels_needed : Nat := 35

-- The main theorem to be proven
theorem chickens_count : 
  (total_bushels_needed - ((cows * bushels_per_cow) + (sheep * bushels_per_sheep))) / bushels_per_chicken = 7 :=
by
  sorry

end chickens_count_l565_565226


namespace eval_nested_fractions_l565_565900

theorem eval_nested_fractions : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end eval_nested_fractions_l565_565900


namespace infinite_composite_l565_565767

open Nat 

theorem infinite_composite (m : ℕ) (hm : m > 0) : 
  ∃ᶠ n in at_top, ∃ k : ℕ, k > 0 ∧ n = k * φ(3^m + 2) * φ(5^m + 2) + m ∧ ¬ (isPrime (3^n + 2) ∨ isPrime (5^n + 2)) :=
by sorry

end infinite_composite_l565_565767


namespace locus_of_point_P_l565_565513

noncomputable def ellipse_locus
  (r : ℝ) (u v : ℝ) : Prop :=
  ∃ x1 y1 : ℝ,
    (x1^2 + y1^2 = r^2) ∧ (u - x1)^2 + v^2 = y1^2

theorem locus_of_point_P {r u v : ℝ} :
  (ellipse_locus r u v) ↔ ((u^2 / (2 * r^2)) + (v^2 / r^2) ≤ 1) :=
by sorry

end locus_of_point_P_l565_565513


namespace maximize_product_sum_l565_565413

theorem maximize_product_sum : ∃ (x y z w v u s t r : ℕ), 
  (∀ i, i ∈ {x, y, z, w, v, u, s, t, r} → i ∈ (finset.range 10) \ {0}) ∧
  finset.card (finset.mk {x, y, z, w, v, u, s, t, r} _) = 9 ∧
  x * y * z + w * v * u + s * t * r = 522 :=
sorry

end maximize_product_sum_l565_565413


namespace range_of_a_l565_565689

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end range_of_a_l565_565689


namespace total_fish_count_l565_565374
noncomputable theory

def initial_fish : ℕ := 22
def given_fish : ℕ := 47
def total_fish (a b : ℕ) := a + b

theorem total_fish_count : total_fish initial_fish given_fish = 69 := by
  unfold total_fish initial_fish given_fish
  simp
  sorry  -- This "sorry" indicates the proof is omitted.

end total_fish_count_l565_565374


namespace soda_difference_l565_565855

-- Define the number of regular soda bottles
def R : ℕ := 79

-- Define the number of diet soda bottles
def D : ℕ := 53

-- The theorem that states the number of regular soda bottles minus the number of diet soda bottles is 26
theorem soda_difference : R - D = 26 := 
by
  sorry

end soda_difference_l565_565855


namespace sum_of_divisors_of_24_l565_565080

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565080


namespace part1_part2_part3_part4_l565_565925

-- Define the integral function I_n
noncomputable def I (n : ℕ) : ℝ := ∫ x in 0..(Real.pi / 4), Real.tan x ^ n

-- Theorem statements
theorem part1 (n : ℕ) : I (n + 2) + I n = 1 / (n + 1) :=
by sorry

theorem part2 : I 1 = 1 / 2 :=
by sorry

theorem part3 : I 2 = 1 - Real.pi / 4 :=
by sorry

theorem part4 : I 3 = 0 :=
by sorry

end part1_part2_part3_part4_l565_565925


namespace ram_weight_increase_percentage_l565_565800

theorem ram_weight_increase_percentage :
  ∃ r s r_new: ℝ,
  r / s = 4 / 5 ∧ 
  r + s = 72 ∧ 
  s * 1.19 = 47.6 ∧
  r_new = 82.8 - 47.6 ∧ 
  (r_new - r) / r * 100 = 10 :=
by
  sorry

end ram_weight_increase_percentage_l565_565800


namespace age_9_not_possible_l565_565375

def age_range : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_valid_num (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 1000 * a + 110 * b + n = aabb ∧ 
               (∀ k ∈ age_range \ {9}, n % k = 0)

theorem age_9_not_possible :
  ¬ ∃ n : ℕ, is_valid_num n :=
sorry

end age_9_not_possible_l565_565375


namespace least_k_for_sum_divisible_l565_565746

theorem least_k_for_sum_divisible (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, (∀ (xs : List ℕ), (xs.length = k) → (∃ ys : List ℕ, (ys.length % 2 = 0) ∧ (ys.sum % n = 0))) ∧ 
    (k = if n % 2 = 1 then 2 * n else n + 1)) :=
sorry

end least_k_for_sum_divisible_l565_565746


namespace binomial_sum_identity_l565_565597

theorem binomial_sum_identity :
  let a : ℕ → ℕ := λ k, Nat.choose 50 k in
  (∑ k in Finset.range 26, (k * (a k : ℕ))) = 50 * (2 ^ 48) := sorry

end binomial_sum_identity_l565_565597


namespace initial_percentage_of_alcohol_l565_565163

theorem initial_percentage_of_alcohol (P : ℝ) (h1 : (P / 100) * 15 = 0.15 * 20) : P = 20 :=
by
  have h2 : (P / 100) * 15 = 3 := by
    rw [mul_comm, ← mul_assoc, div_mul_cancel' (by norm_num : (100 : ℝ) ≠ 0), ← mul_assoc, mul_one, mul_comm]
    exact h1
  sorry

end initial_percentage_of_alcohol_l565_565163


namespace question1_question2_question3_question4_l565_565814

noncomputable def M (n : ℕ) (m : ℕ) : ℕ := n ^ m - 1

noncomputable def G (n : ℕ) (M : ℕ) : ℝ :=
  (n : ℝ) * (Real.log (M + 1)) / (Real.log n)

theorem question1 (n m : ℕ) (h : n ≥ 2) : M n m = n ^ m - 1 := 
  by sorry

theorem question2 (n : ℕ) (M : ℕ) (h : n ≥ 2) : G n M = n * Real.log (M + 1) / Real.log n := 
  by sorry

theorem question3 (M : ℕ) : 
  G 2 M ≈ 6.67 * Real.log (M + 1) ∧
  G 3 M ≈ 6.25 * Real.log (M + 1) ∧
  G 4 M ≈ 6.67 * Real.log (M + 1) ∧
  G 5 M ≈ 7.14 * Real.log (M + 1) := 
  by sorry

theorem question4 (M : ℕ) : 
  let g := λ n, n * (Real.log (M + 1)) / (Real.log n) in
  g 3 ≤ g 2 ∧ g 3 ≤ g 4 ∧ g 3 ≤ g 5 := 
  by sorry

end question1_question2_question3_question4_l565_565814


namespace general_formula_arithmetic_sum_of_geometric_terms_l565_565951

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 2 = 2 ∧ a 5 = 8

noncomputable def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℤ) : Prop :=
  b 1 = 1 ∧ b 2 + b 3 = a 4

noncomputable def sum_of_terms (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = (2:ℝ)^n - 1

theorem general_formula_arithmetic (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n, a n = 2 * n - 2 :=
sorry

theorem sum_of_geometric_terms (a : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h : arithmetic_sequence a) (h2 : geometric_sequence b a) :
  sum_of_terms T b :=
sorry

end general_formula_arithmetic_sum_of_geometric_terms_l565_565951


namespace AL_leq_half_AC_plus_AC₁_l565_565756

variables {A B C B₁ C₁ L : Point}
variables (AC AB B B₁C₁ BC AL CC₁ : Line)

-- Hypotheses, translated from the conditions
hypothesis (hB₁_on_AC : B₁ ∈ AC)
hypothesis (hC₁_on_AB : C₁ ∈ AB)
hypothesis (hB₁C₁_parallel_BC : B₁C₁ ∥ BC)
hypothesis (hCircumcircle_ABB₁_inter_C₁ : ∀ L, Circle ∆ ABB₁ → (C _ ∈ Circle ∆ ABB₁).intersects CC₁ AT L)
hypothesis (hCircumcircle_CLB₁_tangent_AL : ∀ L, Circle ∆ CLB₁ → tangent_to AL)

-- Theorem to be proved:
theorem AL_leq_half_AC_plus_AC₁ (h : hCircumcircle_ABB₁_inter_C₁) : AL ≤ (AC + AC₁) / 2 :=
sorry

end AL_leq_half_AC_plus_AC₁_l565_565756


namespace min_balls_to_ensure_same_color_l565_565318

theorem min_balls_to_ensure_same_color : 
  ∀ (b w : ℕ), b + w ≥ 3 → ∃ n, n = 3 ∧ (∀ balls, balls ⊆ {black, white} ∧ |balls| = n → 
  ∃ color, ∃ x y ∈ balls, x = y ∧ color = black ∨ color = white) :=
by
  sorry

end min_balls_to_ensure_same_color_l565_565318


namespace geometric_sequence_S15_l565_565331

theorem geometric_sequence_S15 
  (a_n : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a_n = a_1 * q ^ (n - 1))
  (S_def : ∀ n, S n = ∑ i in range n, a_n i)
  (h_S5 : S 5 = 4)
  (h_S10 : S 10 = 12) :
  S 15 = 28 :=
by
  sorry

end geometric_sequence_S15_l565_565331


namespace bankers_gain_l565_565777

-- Define constants: banker's discount, rate of interest, and time.
def b_discount : ℚ := 296.07843137254906
def rate : ℚ := 0.17
def time : ℚ := 3
def true_discount : ℚ := b_discount / (1 + rate * time)
def b_gain : ℚ := b_discount - true_discount

theorem bankers_gain :
  b_gain ≈ 99.999 :=
by
  sorry

end bankers_gain_l565_565777


namespace program_output_l565_565199

-- Define the initial conditions
def initial_a := 1
def initial_b := 3

-- Define the program transformations
def a_step1 (a b : ℕ) := a + b
def b_step2 (a b : ℕ) := a - b

-- Define the final values after program execution
def final_a := a_step1 initial_a initial_b
def final_b := b_step2 final_a initial_b

-- Statement to prove
theorem program_output :
  final_a = 4 ∧ final_b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end program_output_l565_565199


namespace find_radius_perpendicular_lines_l565_565948

open Real

-- Definitions and conditions
def parabola (p : ℝ) := { p : ℝ // p > 0 }
def point_on_parabola (M : ℝ × ℝ) (p : ℝ) := M.2 ^ 2 = 2 * p * M.1
def center_of_circle (M : ℝ × ℝ) := M
def radius_of_circle (M : ℝ × ℝ) (F : ℝ × ℝ) := dist M F
def chord_length_on_y_axis (length : ℝ) := length = 2 * sqrt 5
def line (slope : ℝ) (point : ℝ × ℝ) := slope = π / 4 ∧ point = (2, 0)

-- Proof statement for part (1)
theorem find_radius (M : ℝ × ℝ) (p m : ℝ) :
    parabola p →
    point_on_parabola M p →
    chord_length_on_y_axis (dist M (0, M.2)) →
    M.1 = m → p = 2 →
    dist M (M.1, 0) = 3 :=
sorry

-- Proof statement for part (2)
theorem perpendicular_lines (A B O : ℝ × ℝ) (p m slope: ℝ) :
    parabola p →
    line slope (2, 0) →
    point_on_parabola A p ∧ point_on_parabola B p →
    A ≠ B →
    dist A (0, 0) ∗ dist B (0, 0) ≠ 0 →
    A.1 + B.1 = 6 ∧ A.1 * B.1 = 4 → 
    dot_product (A.1,O.1) (B.1,O.1) = 0 :=
sorry 

end find_radius_perpendicular_lines_l565_565948


namespace quadratic_solution_l565_565424

theorem quadratic_solution (x : ℝ) : x ^ 2 - 4 * x + 3 = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end quadratic_solution_l565_565424


namespace find_total_bricks_l565_565533

variable (y : ℕ)
variable (B_rate : ℕ)
variable (N_rate : ℕ)
variable (eff_rate : ℕ)
variable (time : ℕ)
variable (reduction : ℕ)

-- The wall is completed in 6 hours
def completed_in_time (y B_rate N_rate eff_rate time reduction : ℕ) : Prop := 
  time = 6 ∧
  reduction = 8 ∧
  B_rate = y / 8 ∧
  N_rate = y / 12 ∧
  eff_rate = (B_rate + N_rate) - reduction ∧
  y = eff_rate * time

-- Prove that the number of bricks in the wall is 192
theorem find_total_bricks : 
  ∀ (y B_rate N_rate eff_rate time reduction : ℕ), 
  completed_in_time y B_rate N_rate eff_rate time reduction → 
  y = 192 := 
by 
  sorry

end find_total_bricks_l565_565533


namespace min_value_f_at_a_1_range_of_a_for_f_le_3_l565_565602

noncomputable def f (x a : ℝ) := |x - a| + |x - 3|

theorem min_value_f_at_a_1 : ∀ x, f x 1 ≥ 2 :=
by
  intro x
  let f_x_1 := |x - 1| + |x - 3|
  apply abs_nonneg (x - 1)
  apply abs_nonneg (x - 3)
  sorry

theorem range_of_a_for_f_le_3 : ∀ x, (∃ x, f x a ≤ 3) ↔ (0 ≤ a ∧ a ≤ 6) :=
by
  intro x
  let f_x_a := |x - a| + |x - 3|
  sorry

end min_value_f_at_a_1_range_of_a_for_f_le_3_l565_565602


namespace area_closed_figure_cos_x_l565_565471

theorem area_closed_figure_cos_x :
  ∫ x in -real.pi / 3..real.pi / 3, real.cos x = 2 * real.sqrt 3 :=
sorry

end area_closed_figure_cos_x_l565_565471


namespace fewer_pounds_of_carbon_emitted_l565_565698

theorem fewer_pounds_of_carbon_emitted 
  (population : ℕ)
  (pollution_per_car : ℕ)
  (pollution_per_bus : ℕ)
  (bus_capacity : ℕ)
  (percentage_bus : ℚ)
  (initial_population : population = 80)
  (initial_pollution_per_car : pollution_per_car = 10)
  (initial_pollution_per_bus : pollution_per_bus = 100)
  (initial_bus_capacity : bus_capacity = 40)
  (initial_percentage_bus : percentage_bus = 0.25) :
  (population * pollution_per_car - ((population - population * percentage_bus.to_nat) * pollution_per_car + pollution_per_bus)) = 100 :=
by 
  sorry

end fewer_pounds_of_carbon_emitted_l565_565698


namespace largest_divisor_of_n_given_n_squared_divisible_by_72_l565_565311

theorem largest_divisor_of_n_given_n_squared_divisible_by_72 (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ q, q = 12 ∧ q ∣ n :=
by
  sorry

end largest_divisor_of_n_given_n_squared_divisible_by_72_l565_565311


namespace circle_equation_solution_l565_565680

theorem circle_equation_solution
  (a : ℝ)
  (h1 : a ^ 2 = a + 2)
  (h2 : (2 * a / (a + 2)) ^ 2 - 4 * a / (a + 2) > 0) : 
  a = -1 := 
sorry

end circle_equation_solution_l565_565680


namespace find_ages_l565_565282

theorem find_ages (A B C : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A = 2 * B) (h5 : B = C - 7) (h6 : Nat.Prime (A + B + C))
  (h7 : A + B + C < 70) (h8 : (A + B + C).digits.sum = 13) :
  (A = 30) ∧ (B = 15) ∧ (C = 22) :=
by
  sorry

end find_ages_l565_565282


namespace triangle_equilateral_l565_565397

-- Define the geometry of the problem
variables {A B C A1 B1 C1 : Type} [innate_pt : IsPoint A B C]
variables (incircle_contacts : IncircleContacts A B C A1 B1 C1)

-- Define the given conditions
variables (ha : AA1_dist_eq_B1C1 : dist A A1 = dist B B1)
variables (hb : AA1_dist_eq_B1C1 : dist B B1 = dist C C1)

-- Define the theorem statement: 
-- Triangle ABC is equilateral given the conditions
theorem triangle_equilateral 
  (h1 : incircle_contacts)
  (h2 : AA1_dist_eq_B1C1) 
  (h3 : AA1_dist_eq_B1C1) : 
  is_equilateral A B C := 
  sorry

end triangle_equilateral_l565_565397


namespace trapezoid_centroid_on_MN_l565_565762

theorem trapezoid_centroid_on_MN
  (A B C D M N : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_midpoints : midpoint A D = M ∧ midpoint B C = N) :
  ∃ G : Point, is_centroid (trapezoid A B C D) G ∧ lies_on_segment G M N := 
sorry

end trapezoid_centroid_on_MN_l565_565762


namespace goats_difference_l565_565511

-- Definitions of Adam's, Andrew's, and Ahmed's goats
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 2 * adam_goats + 5
def ahmed_goats : ℕ := 13

-- The theorem to prove the difference in goats
theorem goats_difference : andrew_goats - ahmed_goats = 6 :=
by
  sorry

end goats_difference_l565_565511


namespace total_notes_count_l565_565717

theorem total_notes_count :
  ∀ (rows : ℕ) (notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ),
  rows = 5 →
  notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  (rows * notes_per_row + (rows * notes_per_row * blue_notes_per_red + additional_blue_notes)) = 100 := by
  intros rows notes_per_row blue_notes_per_red additional_blue_notes
  sorry

end total_notes_count_l565_565717


namespace power_sum_tenth_l565_565376

theorem power_sum_tenth (a b : ℝ) (h1 : a + b = 1)
    (h2 : a^2 + b^2 = 3)
    (h3 : a^3 + b^3 = 4)
    (h4 : a^4 + b^4 = 7)
    (h5 : a^5 + b^5 = 11) : 
    a^10 + b^10 = 123 := 
sorry

end power_sum_tenth_l565_565376


namespace basketball_team_selection_l565_565483

theorem basketball_team_selection :
  (nat.choose 14 7) + (nat.choose 14 5) = 5434 :=
by
  sorry

end basketball_team_selection_l565_565483


namespace holders_inequality_l565_565360

variable {n : ℕ}
variable (x y : Finₓ n → ℝ) (p q : ℚ)

/-- Hölder's Inequality -/
theorem holders_inequality 
  (h1 : ∀ i, 0 < x i) 
  (h2 : ∀ i, 0 < y i) 
  (h3 : 1 / p + 1 / q = 1) :
  (p > 1 → ∑ i : Finₓ n, x i * y i ≤ (∑ i : Finₓ n, (x i) ^ (p : ℝ)) ^ (1 / p) * (∑ i : Finₓ n, (y i) ^ (q : ℝ)) ^ (1 / q)) ∧
  (p < 1 → ∑ i : Finₓ n, x i * y i ≥ (∑ i : Finₓ n, (x i) ^ (p : ℝ)) ^ (1 / p) * (∑ i : Finₓ n, (y i) ^ (q : ℝ)) ^ (1 / q)) :=
by sorry

end holders_inequality_l565_565360


namespace find_range_of_k_l565_565630

def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem find_range_of_k :
  (∀ x : ℝ, 1 ≤ x → 8^x - 8^(-x) - 4^(x + 1) - 4^(1 - x) + 8 ≥ k * f x) →
  k ∈ set.Iic (-1) :=
begin
  sorry
end

end find_range_of_k_l565_565630


namespace cockroach_find_truth_l565_565862

theorem cockroach_find_truth (D : ℝ) (hD_pos : D > 0) : 
  ∃ n : ℝ, n ≤ (3/2) * D + 7 ∧ can_find_truth D n := sorry

end cockroach_find_truth_l565_565862


namespace correct_statement_l565_565455

-- Definitions of conditions
def statement_A := (sqrt 2 + sqrt 3 = sqrt 5)
def statement_B := ((-sqrt 5)^2 = 5)
def statement_C := (3 * sqrt 2 - 2 * sqrt 2 = 1)
def statement_D := (sqrt 16 = 4 ∨ sqrt 16 = -4) 

-- Theorem stating that option B is correct and others are incorrect
theorem correct_statement : ¬statement_A ∧ statement_B ∧ ¬statement_C ∧ ¬statement_D := 
sorry

end correct_statement_l565_565455


namespace least_value_of_b_l565_565395

theorem least_value_of_b 
  (a b : ℕ)  
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b)
  (h_a_factors : Nat.factors a = [3, 3])
  (h_b_factors : (Nat.numFactors b) = a)
  (h_b_div_a : b % a = 0) :
  b ≥ 81 :=
sorry

end least_value_of_b_l565_565395


namespace sum_of_divisors_of_24_l565_565074

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565074


namespace minimum_product_correct_maximum_product_correct_l565_565255

noncomputable def find_minimum_product (n : ℕ) (hn : 2 ≤ n) 
  (x : Finₓ n → ℝ) 
  (hx_ge : ∀ i, x i ≥ 1 / n) 
  (hx_sum_sq : (Finₓ.sum (λ i => (x i) ^ 2)) = 1) : ℝ := 
  (√(n^2 - n + 1)) / n^n
-- Prove that this function gives the correct minimum value
theorem minimum_product_correct (n : ℕ) (hn : 2 ≤ n) 
  (x : Finₓ n → ℝ) 
  (hx_ge : ∀ i, x i ≥ 1 / n) 
  (hx_sum_sq : (Finₓ.sum (λ i => (x i) ^ 2)) = 1) : 
  find_minimum_product n hn x hx_ge hx_sum_sq = (√(n^2 - n + 1)) / n^n := sorry

noncomputable def find_maximum_product (n : ℕ) (hn : 2 ≤ n) 
  (x : Finₓ n → ℝ) 
  (hx_ge : ∀ i, x i ≥ 1 / n) 
  (hx_sum_sq : (Finₓ.sum (λ i => (x i) ^ 2)) = 1) : ℝ := 
  n^(-n/2)
-- Prove that this function gives the correct maximum value
theorem maximum_product_correct (n : ℕ) (hn : 2 ≤ n) 
  (x : Finₓ n → ℝ) 
  (hx_ge : ∀ i, x i ≥ 1 / n) 
  (hx_sum_sq : (Finₓ.sum (λ i => (x i) ^ 2)) = 1) : 
  find_maximum_product n hn x hx_ge hx_sum_sq = n^(-n/2) := sorry

end minimum_product_correct_maximum_product_correct_l565_565255


namespace polygon_expected_value_l565_565321

def polygon_expected_sides (area_square : ℝ) (flower_prob : ℝ) (area_flower : ℝ) (hex_sides : ℝ) (pent_sides : ℝ) : ℝ :=
  hex_sides * flower_prob + pent_sides * (area_square - flower_prob)

theorem polygon_expected_value :
  polygon_expected_sides 1 (π - 1) (π - 1) 6 5 = π + 4 :=
by
  -- Proof is skipped
  sorry

end polygon_expected_value_l565_565321


namespace gcd_8251_6105_l565_565438

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l565_565438


namespace five_star_three_eq_ten_l565_565553

def operation (a b : ℝ) : ℝ := b^2 + 1

theorem five_star_three_eq_ten : operation 5 3 = 10 := by
  sorry

end five_star_three_eq_ten_l565_565553


namespace winningCandidatePercentage_l565_565162

/-- The number of votes each of the 7 candidates received in an election. -/
def votes : List ℕ := [3600, 8400, 31200, 4300, 2700, 7200, 15500]

/-- The total number of votes cast in the election. -/
def totalVotes : ℕ := votes.sum

/-- The number of votes received by the winning candidate. -/
def winningVotes : ℕ := votes.maximum?.getOrElse 0

/-- The percentage of the total votes received by the winning candidate. -/
def winningPercentage : ℚ := (winningVotes : ℚ) / (totalVotes : ℚ) * 100

/-- Proof that the winning candidate received 43.33% of the total votes. -/
theorem winningCandidatePercentage : winningPercentage = 43.33 := by
  sorry

end winningCandidatePercentage_l565_565162


namespace sum_distances_from_point_to_faces_constant_l565_565765

theorem sum_distances_from_point_to_faces_constant 
  (T : Tetrahedron) 
  (P : Point) 
  (V : ℝ)
  (S : ℝ)
  (h1 h2 h3 h4 : ℝ)
  (h_distances : h1 + h2 + h3 + h4 = (3 * V) / S) :
  h1 + h2 + h3 + h4 = (3 * V) / S := 
sorry

end sum_distances_from_point_to_faces_constant_l565_565765


namespace number_of_pages_in_chunk_l565_565479

-- Conditions
def first_page : Nat := 213
def last_page : Nat := 312

-- Define the property we need to prove
theorem number_of_pages_in_chunk : last_page - first_page + 1 = 100 := by
  -- skipping the proof
  sorry

end number_of_pages_in_chunk_l565_565479


namespace syllogism_sequence_l565_565290

theorem syllogism_sequence (P Q R : Prop)
  (h1 : R)
  (h2 : Q)
  (h3 : P) : 
  (Q ∧ R → P) → (R → P) ∧ (Q → (P ∧ R)) := 
by
  sorry

end syllogism_sequence_l565_565290


namespace cos_double_angle_l565_565276

theorem cos_double_angle (α : ℝ) (h : Real.sin α = sqrt 3 / 3) : Real.cos (2 * α) = 1 / 3 :=
by
  sorry

end cos_double_angle_l565_565276


namespace sum_of_divisors_of_24_l565_565108

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565108


namespace sum_in_base5_correct_l565_565584

-- Define numbers in base 5
def n1 : ℕ := 231
def n2 : ℕ := 414
def n3 : ℕ := 123

-- Function to convert a number from base 5 to base 10
def base5_to_base10(n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100)
  d0 * 1 + d1 * 5 + d2 * 25

-- Convert the given numbers from base 5 to base 10
def n1_base10 : ℕ := base5_to_base10 n1
def n2_base10 : ℕ := base5_to_base10 n2
def n3_base10 : ℕ := base5_to_base10 n3

-- Base 10 sum
def sum_base10 : ℕ := n1_base10 + n2_base10 + n3_base10

-- Function to convert a number from base 10 to base 5
def base10_to_base5(n : ℕ) : ℕ :=
  let d0 := n % 5
  let d1 := (n / 5) % 5
  let d2 := (n / 25) % 5
  let d3 := (n / 125)
  d0 * 1 + d1 * 10 + d2 * 100 + d3 * 1000

-- Convert the sum from base 10 to base 5
def sum_base5 : ℕ := base10_to_base5 sum_base10

-- The theorem to prove the sum in base 5 is 1323_5
theorem sum_in_base5_correct : sum_base5 = 1323 := by
  -- Proof steps would go here, but we insert sorry to skip it
  sorry

end sum_in_base5_correct_l565_565584


namespace reduction_in_carbon_emission_l565_565694

theorem reduction_in_carbon_emission 
  (population : ℕ)
  (carbon_per_car : ℕ)
  (carbon_per_bus : ℕ)
  (bus_capacity : ℕ)
  (percentage_take_bus : ℝ)
  (initial_cars : ℕ := population) :
  (initial_cars * carbon_per_car - 
  (population * percentage_take_bus.floor.to_nat * carbon_per_bus +
   (population - population * percentage_take_bus.floor.to_nat) * carbon_per_car)) = 100 :=
by
  -- Given conditions
  let initial_emission := initial_cars * 10 -- Total carbon emission from all cars
  let people_take_bus := (population * 0.25).to_nat -- 25% of people switch to the bus
  let remaining_cars := population - people_take_bus -- People still driving
  let new_emission := 100 + remaining_cars * 10 -- New total emission
  have reduction := initial_emission - new_emission -- Reduction in carbon emission
  -- Correct answer
  exact reduction = 100

end reduction_in_carbon_emission_l565_565694


namespace brenda_age_l565_565866

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end brenda_age_l565_565866


namespace probability_correct_l565_565222

/-
  Problem statement:
  Consider a modified city map where a student walks from intersection A to intersection B, passing through C and D.
  The student always walks east or south and at each intersection, decides the direction to go with a probability of 1/2.
  The map requires 4 eastward and 3 southward moves to reach B from A. C is 2 east, 1 south move from A. D is 3 east, 2 south moves from A.
  Prove that the probability the student goes through both C and D is 12/35.
-/

noncomputable def probability_passing_C_and_D : ℚ :=
  let total_paths_A_to_B := Nat.choose 7 4
  let paths_A_to_C := Nat.choose 3 2
  let paths_C_to_D := Nat.choose 2 1
  let paths_D_to_B := Nat.choose 2 1
  (paths_A_to_C * paths_C_to_D * paths_D_to_B) / total_paths_A_to_B

theorem probability_correct :
  probability_passing_C_and_D = 12 / 35 :=
by
  sorry

end probability_correct_l565_565222


namespace number_of_planes_l565_565550

theorem number_of_planes : 
  ∃ (M : Type), (∀ (V : fin 4 → ℝ × ℝ × ℝ), 
  (∃ (d : ℝ → ℝ → Prop), (∀ (i j : fin 4), (i ≠ j → ((d (V i) (V j)) ∈ {2, 1})) ))) → 
  count_plane_constructions M = 32 :=
by sorry

end number_of_planes_l565_565550


namespace part1_part2_l565_565655

noncomputable def U : Set ℝ := Set.univ

noncomputable def A (a: ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
noncomputable def B (a: ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem part1 (a : ℝ) (ha : a = 1/2) :
  (U \ (B a)) ∩ (A a) = {x | 9/4 ≤ x ∧ x < 5/2} :=
sorry

theorem part2 (p q : ℝ → Prop)
  (hp : ∀ x, p x → x ∈ A a) (hq : ∀ x, q x → x ∈ B a)
  (hq_necessary : ∀ x, p x → q x) :
  -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2 :=
sorry

end part1_part2_l565_565655


namespace area_triangle_ABC_length_AD_l565_565707

-- Define the lengths and angle
def AB : ℝ := 40
def AC : ℝ := 30
def angleA : ℝ := 90 -- degrees

-- Define the property that D is the midpoint of AB
def isMidpoint (A B D : ℝ × ℝ) : Prop :=
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Coordinates for points A, B, and D
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (40, 0)
def D : ℝ × ℝ := (20, 0) -- since it's given as the midpoint

-- Prove the area of triangle ABC
theorem area_triangle_ABC : 
  let area := (1/2) * AB * AC in
  area = 600 := 
by
  sorry

-- Prove the length of AD
theorem length_AD :
  let lengthAD := (AB / 2) in
  lengthAD = 20 :=
by
  sorry

end area_triangle_ABC_length_AD_l565_565707


namespace eccentricity_range_l565_565280

theorem eccentricity_range (a b : ℝ) (e : ℝ) (h₁ : a > b) (h₂ : 0 < b) (h₃ : ∃ c : ℝ, c = sqrt (a^2 - b^2)) :
  (sqrt 2 - 1) < e ∧ e < 1 :=
sorry

end eccentricity_range_l565_565280


namespace abs_sub_nonpos_l565_565675

theorem abs_sub_nonpos (x : ℝ) : |x| - x ≤ 0 := 
by
  sorry

end abs_sub_nonpos_l565_565675


namespace pairs_of_different_positives_l565_565234

def W (x : ℕ) : ℕ := x^4 - 3 * x^3 + 5 * x^2 - 9 * x

theorem pairs_of_different_positives (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (hW : W a = W b) : (a, b) = (1, 2) ∨ (a, b) = (2, 1) := 
sorry

end pairs_of_different_positives_l565_565234


namespace units_digit_of_expression_l565_565587

noncomputable def A := 15 + Real.sqrt 225
noncomputable def B := 15 - Real.sqrt 225

theorem units_digit_of_expression :
  Nat.units (Nat.pow (Nat.floor A) 17 + Nat.pow (Nat.floor B) 17) = 0 :=
sorry

end units_digit_of_expression_l565_565587


namespace sum_of_divisors_of_24_is_60_l565_565069

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565069


namespace shifted_quadratic_function_l565_565775

theorem shifted_quadratic_function :
  ∀ x : ℝ, (λ x, 2 * (x - 3)^2 + 4) (x + 2) - 3 = 2 * x^2 - 4 * x + 3 :=
by
  assume x
  unfold function
  sorry

end shifted_quadratic_function_l565_565775


namespace sum_of_divisors_of_twenty_four_l565_565115

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565115


namespace min_value_t_minus_k_l565_565369

open_locale big_operators

noncomputable def a_n (S : ℕ → ℚ) (n : ℕ) : ℚ := S n - S (n - 1)

noncomputable def S_n (S : ℕ → ℚ) := λ n, S n

axiom given_conditions (S : ℕ → ℚ) (k t : ℚ) : 
  (∀ n : ℕ, 2 * S_n S n = (6 + a_n S n) / 2) ∧ 
  (∀ n : ℕ, 3 * S_n S n - 1 / S_n S n ∈ set.Icc k t)

theorem min_value_t_minus_k (S : ℕ → ℚ) (k t : ℚ) 
  (h : given_conditions S k t) : t - k = 9 / 4 := 
sorry

end min_value_t_minus_k_l565_565369


namespace number_of_lines_proof_l565_565757

noncomputable def number_of_lines (A B : Point) (dist_AB : ℝ) (dist_a : ℝ) (dist_b : ℝ) : ℕ :=
if h : dist_AB = 7 ∧ dist_a = 3 ∧ dist_b = 4 then 3 else sorry

-- The main theorem statement
theorem number_of_lines_proof (A B : Point) (h_AB : dist A B = 7) :
  number_of_lines A B 7 3 4 = 3 :=
by { sorry }

end number_of_lines_proof_l565_565757


namespace distance_from_A_to_line_l565_565299

def point := ℝ × ℝ

-- Define the line l1: y = ax - 2a + 5
def l1 (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a + 5

-- Define the fixed point A
def A : point := (2, 5)

-- Define the line l: x - 2y + 3 = 0
def line (p : point) : Prop := p.1 - 2 * p.2 + 3 = 0

-- Define the distance from point to line
def distance_point_to_line (p : point) : ℝ :=
  abs (p.1 - 2 * p.2 + 3) / sqrt (1^2 + (-2)^2)

-- The statement to prove: given the conditions, the distance from point A to the line is sqrt(5)
theorem distance_from_A_to_line :
  distance_point_to_line A = sqrt 5 :=
by
  sorry

end distance_from_A_to_line_l565_565299


namespace blood_expiration_date_l565_565195

theorem blood_expiration_date :
  let expiry_seconds := (11.factorial : ℕ) in
  let seconds_in_a_day := 86400 in
  let days_until_expiry := expiry_seconds / seconds_in_a_day in
  let donation_date := date.mk 1 15 (year) in
  let expiration_date := days_after donation_date days_until_expiry in
  expiration_date = date.mk 3 8 (year + 1)
:= sorry

end blood_expiration_date_l565_565195


namespace dihedral_angle_SC_l565_565303

variables {α β θ : ℝ}

theorem dihedral_angle_SC (h1 : ∠ (A S B) = π / 2)
                           (h2 : ∠ (A S C) = α ∧ θ < α ∧ α < π / 2)
                           (h3 : ∠ (B S C) = β ∧ 0 < β ∧ β < π / 2) :
                           θ = π - arccos (cot α * cot β) :=
by
  sorry

end dihedral_angle_SC_l565_565303


namespace determine_pairs_l565_565556

open Int

-- Definitions corresponding to the conditions of the problem:
def is_prime (p : ℕ) : Prop := Nat.Prime p
def condition1 (p n : ℕ) : Prop := is_prime p
def condition2 (p n : ℕ) : Prop := n ≤ 2 * p
def condition3 (p n : ℕ) : Prop := (n^(p-1)) ∣ ((p-1)^n + 1)

-- Main theorem statement:
theorem determine_pairs (n p : ℕ) (h1 : condition1 p n) (h2 : condition2 p n) (h3 : condition3 p n) :
  (n = 1 ∧ is_prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end determine_pairs_l565_565556


namespace circle_equation_l565_565970

theorem circle_equation (a b r : ℝ) 
    (h₁ : b = -4 * a)
    (h₂ : abs (a + b - 1) / Real.sqrt 2 = r)
    (h₃ : (b + 2) / (a - 3) * (-1) = -1)
    (h₄ : a = 1)
    (h₅ : b = -4)
    (h₆ : r = 2 * Real.sqrt 2) :
    ∀ x y: ℝ, (x - 1) ^ 2 + (y + 4) ^ 2 = 8 := 
by
  intros
  sorry

end circle_equation_l565_565970


namespace pairs_of_pingpong_rackets_sold_l565_565147

theorem pairs_of_pingpong_rackets_sold 
(total_sales : ℝ)
(avg_price_per_pair : ℝ)
(h1 : total_sales = 686)
(h2 : avg_price_per_pair = 9.8) :
(total_sales / avg_price_per_pair = 70) :=
by
  have h := h1 ▸ h2 ▸ rfl
  sorry

end pairs_of_pingpong_rackets_sold_l565_565147


namespace sum_of_divisors_of_24_l565_565077

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565077


namespace one_thirds_in_nine_thirds_l565_565662

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l565_565662


namespace speed_in_still_water_l565_565485

-- Define the conditions: upstream and downstream speeds.
def upstream_speed : ℝ := 10
def downstream_speed : ℝ := 20

-- Define the still water speed theorem.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 15 := by
  sorry

end speed_in_still_water_l565_565485


namespace proof_problem_l565_565985

-- Main problem statement
theorem proof_problem 
  (a c : ℝ)
  (h1 : root (λ x, a * x^2 + x + c) 1)
  (h2 : root (λ x, a * x^2 + x + c) 3)
  (ha_lt_zero : a < 0)
  (hroot_eq1 : 1 + 3 = - (1 / a))
  (hroot_eq2 : 1 * 3 = c / a) :
  (a = -1/4 ∧ c = -3/4) ∧ 
  ∀ m : ℝ, (∀ x : ℝ, 2 < x ∧ x < 6 → x > -m) ↔ (m ≥ -2) :=
sorry

end proof_problem_l565_565985


namespace area_of_smallest_square_l565_565444

-- Define a circle with a given radius
def radius : ℝ := 7

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the side length of the smallest square that can contain the circle
def side_length : ℝ := diameter

-- Define the area of the square as the side length squared
def area_of_square : ℝ := side_length ^ 2

-- State the theorem: the area of the smallest square that contains a circle of radius 7 is 196
theorem area_of_smallest_square : area_of_square = 196 := by
    sorry

end area_of_smallest_square_l565_565444


namespace compute_expression_l565_565542

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end compute_expression_l565_565542


namespace domain_logeq2_l565_565402

def log_domain_eq (x : ℝ) : Prop :=
  x > -1

theorem domain_logeq2 :
  ∀ x : ℝ, log_domain_eq x ↔ (f(x) = log2(x+1)) :=
sorry

end domain_logeq2_l565_565402


namespace S_1992_plus_S_1993_l565_565308

/-- Define the sequence S_n as given in the problem --/
def S (n : ℕ) := ∑ k in finset.range n, ((-1) ^ (k + 1)) * (k + 1)

/-- Main theorem to prove the value of S_1992 + S_1993 --/
theorem S_1992_plus_S_1993 : S 1992 + S 1993 = 1 :=
  sorry

end S_1992_plus_S_1993_l565_565308


namespace part_a_part_b_l565_565920

noncomputable def a_n (n : ℕ) : ℝ :=
∑ k in Finset.range (n + 1), if k > 0 then 1 / Nat.choose n k else 0

noncomputable def b_n (n : ℕ) : ℝ := (a_n n) ^ n

theorem part_a : ∃ L : ℝ, (filter.at_top.map (λ n, b_n n)).tendsto (nhds L) ∧ L = Real.exp 2 :=
sorry

theorem part_b : Real.exp 2 > (3 / 2) ^ (Real.sqrt 3 + Real.sqrt 2) :=
sorry

end part_a_part_b_l565_565920


namespace valid_pairings_count_l565_565218

open Finset

-- Definition of the colors
inductive Color
| red
| blue
| yellow
| green
| purple

open Color

-- Set of all colors
def all_bowls : Finset Color := { red, blue, yellow, green, purple }

-- Set of all glasses (without purple)
def all_glasses : Finset Color := { red, blue, yellow, green }

-- Predicate to check valid pairings
def valid_pairings (b : Color) (g : Option Color) : Prop :=
  (b ≠ purple ∧ g.isSome ∧ g.get = b) ∨ (b = purple ∧ g.isNone)

-- Count valid pairings
def count_valid_pairings : Nat :=
  all_bowls.sum (λ b => if b ≠ purple then all_glasses.card else 1)

theorem valid_pairings_count : count_valid_pairings = 17 :=
by
  -- count for non-purple bowls and glasses
  have h1: (all_bowls.erase purple).sum (λ b => all_glasses.card) = 4 * 4 := sorry
  -- count for purple bowl without glass
  have h2: (erase all_bowls purple).card * 1 = 1 := sorry
  -- combining results
  show count_valid_pairings = 4 * 4 + 1
  sorry

end valid_pairings_count_l565_565218


namespace parabola_vertex_l565_565783

theorem parabola_vertex (x y : ℝ) : ∀ x y, (y^2 + 8 * y + 2 * x + 11 = 0) → (x = 5 / 2 ∧ y = -4) :=
by
  intro x y h
  sorry

end parabola_vertex_l565_565783


namespace total_price_of_books_l565_565011

-- Define the given conditions as constants
constant total_books : ℕ := 90
constant math_book_cost : ℕ := 4
constant history_book_cost : ℕ := 5
constant math_books_bought : ℕ := 53

-- Define the property to be proved
theorem total_price_of_books : 
  let history_books_bought := total_books - math_books_bought in
  let total_math_book_cost := math_books_bought * math_book_cost in
  let total_history_book_cost := history_books_bought * history_book_cost in
  total_math_book_cost + total_history_book_cost = 397 :=
by
  sorry

end total_price_of_books_l565_565011


namespace T1_T2_l565_565196

variables {Grip No : Type} [Fintype Grip] [Fintype No]

-- Definitions according to conditions
def P1 (g : Grip) : Finset No := sorry  -- Each grip is a set of nos
def P2 (g1 g2 g3 : Grip) : g1 ≠ g2 → g2 ≠ g3 → g1 ≠ g3 → Finset.inter (Finset.inter (P1 g1) (P1 g2)) (P1 g3) = sorry -- One common no for any three distinct grips
def P3 (n : No) : Fintype {g : Grip // n ∈ P1 g} := sorry  -- Each no belongs to at least two grips
def P4 : Fintype.card Grip = 5 := sorry  -- There is a total of five grips
def P5 (g : Grip) : Fintype.card (P1 g) = 3 := sorry  -- There are exactly three nos in each grip

-- Theorems to be proved
theorem T1 : Fintype.card No = 10 :=
  by sorry

theorem T2 (n : No) : Fintype.card {g : Grip // n ∈ P1 g} = 3 :=
  by sorry

end T1_T2_l565_565196


namespace cone_height_l565_565850

-- Define the conditions given in the problem
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h
def vertex_angle := 45°  -- although not necessary in computations, included for completeness

-- Define the theorem to be proved
theorem cone_height (h : ℝ) (r : ℝ) (h_eq_r : h = r) (volume : cone_volume r h = 8192 * Real.pi) : 
  h = 28.9 :=
sorry

end cone_height_l565_565850


namespace find_a4_b4_l565_565896

-- Given conditions
variables {a1 a2 a3 a4 b1 b2 b3 b4 : ℝ}
hypothesis h1 : a1 * b1 + a2 * b3 = 1
hypothesis h2 : a1 * b2 + a2 * b4 = 0
hypothesis h3 : a3 * b1 + a4 * b3 = 0
hypothesis h4 : a3 * b2 + a4 * b4 = 1
hypothesis h5 : a2 * b3 = 7

-- Statement to prove
theorem find_a4_b4 : a4 * b4 = -6 :=
by
  sorry

end find_a4_b4_l565_565896


namespace area_equilateral_triangle_DEF_l565_565729

theorem area_equilateral_triangle_DEF :
  ∃ (DEF : Triangle), 
  (DEF.A.angle = 90) ∧ 
  (DEF.B.angle = 60) ∧ 
  (DEF.BC.length = 1) ∧ 
  (equilateral_triangle DEF.A DEF.B DEF.D) ∧ 
  (equilateral_triangle DEF.A DEF.C DEF.E) ∧ 
  (equilateral_triangle DEF.B DEF.C DEF.F) →
  (DEF.area = 9 * Real.sqrt 3 / 16) := 
sorry

end area_equilateral_triangle_DEF_l565_565729


namespace min_abs_diff_x1_x2_l565_565685

theorem min_abs_diff_x1_x2 (x1 x2 : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = Real.sin (π * x))
  (Hbounds : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) : |x1 - x2| = 1 := 
by
  sorry

end min_abs_diff_x1_x2_l565_565685


namespace tangency_and_angles_l565_565537

theorem tangency_and_angles (l m n : Line)
  (h1 : ∀ (A B : Line), (A = l ∧ B = m) ∨ (A = m ∧ B = n) ∨ (A = n ∧ B = l) → angle_between A B = 60)
  (h2 : ¬ (is_parallel l m ∧ is_parallel m n ∧ is_parallel n l))
  (h3 : ∃ C' : Circle, is_tangent C' l ∧ is_tangent C' m ∧ is_tangent C' n) :
  (∃ C : Circle, is_tangent C l ∧ is_tangent C m ∧ is_tangent C n) ∧
  ¬ (is_parallel l m ∧ is_parallel m n ∧ is_parallel n l) ∧
  (∃ C' : Circle, is_tangent C' l ∧ is_tangent C' m ∧ is_tangent C' n) :=
by
  sorry

end tangency_and_angles_l565_565537


namespace innovation_contribution_l565_565838

variable (material : String)
variable (contribution : String → Prop)
variable (A B C D : Prop)

-- Conditions
axiom condA : contribution material → A
axiom condB : contribution material → ¬B
axiom condC : contribution material → ¬C
axiom condD : contribution material → ¬D

-- The problem statement
theorem innovation_contribution :
  contribution material → A :=
by
  -- dummy proof as placeholder
  sorry

end innovation_contribution_l565_565838


namespace limit_divided_by_3_l565_565366

-- Defining a function f which has a derivative at 1
variable {f : ℝ → ℝ}
variable {f' : ℝ}
hypothesis H : deriv f 1 = f'

-- Statement of the theorem
theorem limit_divided_by_3 (H : deriv f 1 = f') :
  (lim (fun Δx => (f(1 + Δx) - f(1)) / (3 * Δx)) at 0) = (1 / 3) * f' :=
by
  -- Proof not required in the problem description
  sorry

end limit_divided_by_3_l565_565366


namespace sum_divisors_24_l565_565026

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565026


namespace quadratic_root_difference_l565_565548

theorem quadratic_root_difference (p : ℝ) :
  let r1 := (2 * p + real.sqrt ((2 * p)^2 - 4 * (p^2 - 4 * p + 4))) / 2,
      r2 := (2 * p - real.sqrt ((2 * p)^2 - 4 * (p^2 - 4 * p + 4))) / 2 in
  r1 - r2 = 4 * real.sqrt (p - 1) :=
by
  sorry

end quadratic_root_difference_l565_565548


namespace digit_B_is_4_l565_565569

theorem digit_B_is_4 : 
  ∃ (A B C D E F : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  A ∈ {0, 1, 2, 3, 4, 5} ∧
  B ∈ {0, 1, 2, 3, 4, 5} ∧
  C ∈ {0, 1, 2, 3, 4, 5} ∧
  D ∈ {0, 1, 2, 3, 4, 5} ∧
  E ∈ {0, 1, 2, 3, 4, 5} ∧
  F ∈ {0, 1, 2, 3, 4, 5} ∧
  2 * A + 2 * B + 3 * C + 3 * D + 3 * E + 2 * F = 40 ∧
  B = 4 :=
sorry

end digit_B_is_4_l565_565569


namespace division_of_fractions_l565_565665

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l565_565665


namespace equilateral_hyperbola_standard_eq_l565_565953

/-- Given an equilateral hyperbola passing through the point (5, -4),
the standard equation of the hyperbola is derived. 
-/
theorem equilateral_hyperbola_standard_eq:
  (∃ λ : ℝ, λ ≠ 0 ∧ (5^2 - (-4)^2) = λ) →
  (∃ x y : ℝ, x = 5 ∧ y = -4 ∧ (x^2 / 9 - y^2 / 9 = 1)) :=
by
  intro h
  obtain ⟨λ, hλ_ne_zero, hλ⟩ := h
  have λ_eq_9 : λ = 9 := by linarith
  use 5, -4
  split; linarith
  rw [λ_eq_9]
  field_simp [ne_of_gt (by norm_num : 9 > 0)]
  linarith


end equilateral_hyperbola_standard_eq_l565_565953


namespace companion_vector_gx_cos_x_value_point_P_exists_l565_565958

-- (Ⅰ) Given a function g(x), prove the companion vector.
theorem companion_vector_gx (a b : ℝ) 
  (g : ℝ → ℝ) 
  ( h1 : g x = sqrt 3 * sin (x - π) - sin ((3 / 2) * π - x) ) :
  (a, b) = (-sqrt 3, 1) := 
sorry

-- (Ⅱ) Given the companion function f(x), prove the value of cos x.
theorem cos_x_value (x : ℝ) 
  (h2 : ∃ x ∈ Ioo (-π / 3) (π / 6), sin x + sqrt 3 * cos x = 8 / 5 ) :
  cos x = (4 * sqrt 3 + 3) / 10 :=
sorry

-- (Ⅲ) Given transformations to function g(x), determine if there’s a point P such that vector AP is perpendicular to vector BP.
theorem point_P_exists (h : ℝ → ℝ) 
  (A B P : ℝ × ℝ)
  (h3 : ∀ x, h x = -sqrt 3 * sin x + cos x)
  (h4 : ∃ P, (P.1 + 2, 2 * cos (P.1 / 2) - 3 ) • (P.1 - 2, 2 * cos (P.1 / 2) - 6) = 0) :
  P = (0, 2) :=
sorry

end companion_vector_gx_cos_x_value_point_P_exists_l565_565958


namespace john_total_time_l565_565342

theorem john_total_time (explore_time : ℝ) (note_time : ℝ) (book_time : ℝ) :
  explore_time = 3 ∧ note_time = explore_time / 2 ∧ book_time = 0.5 →
  explore_time + note_time + book_time = 5 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end john_total_time_l565_565342


namespace remaining_amount_to_be_paid_l565_565148

-- Given conditions
def part_payment : ℝ := 875
def fraction_represented : ℝ := 0.25
def total_cost := part_payment / fraction_represented

-- Problem statement
def remaining_payment : ℝ := total_cost - part_payment

-- Prove that the remaining payment is 2625.
theorem remaining_amount_to_be_paid : remaining_payment = 2625 := by
  sorry

end remaining_amount_to_be_paid_l565_565148


namespace sum_of_divisors_of_24_l565_565075

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565075


namespace isosceles_triangle_sides_l565_565705

theorem isosceles_triangle_sides (AB BC: ℝ) (AC: ℝ) :
  -- Conditions
  AB = BC →
  AB + BC + AC = 60 →
  ∃ r: ℝ, r > 0 ∧ 
    ∃ O: Point, 
      -- Given O lies on the incircle and is the centroid
      ∃ s: ℝ, s > 0 ∧ (s=AB) → 
          -- Desired conclusion
          (AB = 25 ∧ BC = 25 ∧ AC = 10) :=
begin
  sorry
end

end isosceles_triangle_sides_l565_565705


namespace pentagonal_prism_base_color_l565_565153

-- Definition of pentagonal prism vertices and edges
structure Prism :=
  (A B : Fin 5 → Fin 5 → Bool) -- Coloring of edges; Bool represents Red or Blue

-- Define the edge coloring rule
def colored_edges (p : Prism) : Prop :=
  ∀ (i j k : Fin 5),
    (i ≠ j ∧ j ≠ k ∧ i ≠ k) →
    ¬ (p.A i j = p.A j k ∧ p.A j k = p.A i k) ∧ 
    ¬ (p.B i j = p.B j k ∧ p.B j k = p.B i k)

-- Define the theorem to prove
theorem pentagonal_prism_base_color (p : Prism) :
  colored_edges p →
  (∀ i j, p.A i j = p.A 0 1) ∧ 
  (∀ i j, p.B i j = p.B 0 1) :=
begin
  sorry
end

end pentagonal_prism_base_color_l565_565153


namespace tan_double_angle_l565_565626

variables {α : ℝ}

-- Conditions
def in_second_quadrant (α : ℝ) : Prop := π/2 < α ∧ α < π
def sin_alpha_value (α : ℝ) : Prop := Real.sin α = 3/5

-- Proof Problem Statement
theorem tan_double_angle (h1 : in_second_quadrant α) (h2 : sin_alpha_value α) :
  Real.tan (2 * α) = -24 / 7 := 
sorry

end tan_double_angle_l565_565626


namespace number_of_dogs_is_correct_l565_565320

variable (D C B : ℕ)
variable (k : ℕ)

def validRatio (D C B : ℕ) : Prop := D = 7 * k ∧ C = 7 * k ∧ B = 8 * k
def totalDogsAndBunnies (D B : ℕ) : Prop := D + B = 330
def correctNumberOfDogs (D : ℕ) : Prop := D = 154

theorem number_of_dogs_is_correct (D C B k : ℕ) 
  (hRatio : validRatio D C B k)
  (hTotal : totalDogsAndBunnies D B) :
  correctNumberOfDogs D :=
by
  sorry

end number_of_dogs_is_correct_l565_565320


namespace max_value_of_f_symmetric_about_point_concave_inequality_l565_565297

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem max_value_of_f : ∃ x, f x = -4 :=
by
  sorry

theorem symmetric_about_point : ∀ x, f (1 - x) + f (1 + x) = -4 :=
by
  sorry

theorem concave_inequality (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > 1) : 
  f ((x1 + x2) / 2) ≥ (f x1 + f x2) / 2 :=
by
  sorry

end max_value_of_f_symmetric_about_point_concave_inequality_l565_565297


namespace range_of_a_l565_565938

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 3*x + 2*a else x - a*log x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) → 1 ≤ a ∧ a ≤ exp 1 := 
begin
  sorry
end

end range_of_a_l565_565938


namespace eccentricity_of_ellipse_l565_565616

variable {a b : ℝ}
variable (h1 : a > b > 0)
variable (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
variable (area_eq : ∀ (x1 x2 y1 y2 y3 y4 : ℝ), 
  (y2 - y1) * (x1 - x2) + (y4 - y3) * (x1 - x2) + 2 * (abs x1 * 2 * b) = 8 * b^2 / 3)

theorem eccentricity_of_ellipse : ∃ e : ℝ, e = sqrt 2 / 2 := 
sorry

end eccentricity_of_ellipse_l565_565616


namespace area_of_black_region_l565_565503

theorem area_of_black_region (side_small side_large : ℕ) 
  (h1 : side_small = 5) 
  (h2 : side_large = 9) : 
  (side_large * side_large) - (side_small * side_small) = 56 := 
by
  sorry

end area_of_black_region_l565_565503


namespace find_ellipse_equation_exists_line_through_P_l565_565267

open Real

-- Definitions from conditions in the problem
def ellipse_def (a b x y : ℝ) : Prop := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1
def area_of_quadrilateral (a b : ℝ) : ℝ := 2 * a * b
def distance_from_origin (a b : ℝ) : ℝ := (a * b) / (sqrt (a ^ 2 + b ^ 2))

-- Theorem statements reformulated as Lean 4 statements
theorem find_ellipse_equation (a b : ℝ) (h1 : a > b > 0)
                              (h2 : area_of_quadrilateral a b = 2 * sqrt 15)
                              (h3 : distance_from_origin a b = sqrt 30 / 4) :
  (∀ x y : ℝ, ellipse_def a b x y ↔ ellipse_def (sqrt 5) (sqrt 3) x y) :=
by sorry

theorem exists_line_through_P (x₁ x₂ : ℝ) (P : ℝ × ℝ) (C : ℝ → ℝ → Prop)
                              (A B : ℝ × ℝ)
                              (h1 : P = (0, 2))
                              (h2 : C = λ x y, ellipse_def (sqrt 5) (sqrt 3) x y)
                              (h3 : C (A.1) (A.2) ∧ C (B.1) (B.2))
                              (h4 : ∃ l : ℝ × ℝ → Prop, l P ∧ ∀ x y, (l (x, y) ↔ y = k₁ * x + 2 ∨ y = k₂ * x + 2)
  (h5 : x₁ ≠ x₂)) : ∀ k₁ k₂ : ℝ, k₁ = 2 * sqrt 5 / 5 ∨ k₂ = 8 * sqrt 5 / 5 :=
by sorry

end find_ellipse_equation_exists_line_through_P_l565_565267


namespace calculation_correct_l565_565534

theorem calculation_correct : 0.54 - (1 / 8) + 0.46 - (7 / 8) = 0 :=
by {
  -- Apply commutative and associative properties
  have h1 : ∀ a b c d : ℚ, a - b + c - d = (a + c) - (b + d),
  { intros, linarith, },
  -- Utilize the lemma for the specific numbers
  calc
  0.54 - 1 / 8 + 0.46 - 7 / 8
    = (0.54 + 0.46) - (1 / 8 + 7 / 8) : by exact h1 0.54 (1 / 8) 0.46 (7 / 8)
  ... = 1.00 - 1                          : by norm_num
  ... = 0                                 : by norm_num
}


end calculation_correct_l565_565534


namespace length_of_bridge_l565_565830

theorem length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) 
  (h_train_length: train_length = 145) 
  (h_train_speed: train_speed_kmh = 45) 
  (h_crossing_time: crossing_time = 30) : 
  let train_speed_ms := train_speed_kmh * (1000 / 3600) in
  let distance_travelled := train_speed_ms * crossing_time in
  distance_travelled - train_length = 230 :=
begin
  sorry -- Proof not required as per instructions
end

end length_of_bridge_l565_565830


namespace b_work_time_l565_565823

variables (R_a R_b : ℝ) (T_b : ℝ)

-- Conditions
axiom cond1 : R_a = 2 * R_b
axiom cond2 : (R_a + R_b) * 10 = 1

-- Question: How long would it take for b alone to complete the work?
theorem b_work_time : T_b = 30 :=
by
  unfold T_b
  have : R_b = 1 / 30 := sorry
  have : T_b = 30 := sorry
  exact sorry

end b_work_time_l565_565823


namespace cost_price_theorem_profit_function_theorem_max_profit_theorem_l565_565518

-- Define the cost prices and conditions
def cost_price_B_jersey : ℝ := 180
def cost_price_A_jersey : ℝ := 200
def total_jerseys : ℝ := 210
def selling_price_A_jersey : ℝ := 320
def selling_price_B_jersey : ℝ := 280

variable (m : ℝ) (a : ℝ)

-- Define constraints on the number of jerseys
def valid_m (m : ℝ) : Prop := 100 ≤ m ∧ m ≤ 140

-- Define the profit function W with respect to m
def profit_W (m : ℝ) : ℝ := 20 * m + 21000

-- Define the profit function Q with respect to m and a
def profit_Q : ℝ := 
  if 0 < a ∧ a < 20 then (20 - a) * m + 21000
  else if a = 20 then 21000
  else 23000 - 100 * a

-- Define the maximum profit function
def max_profit (a : ℝ) (m : ℝ) : ℝ :=
  if 0 < a ∧ a < 20 then (20 - a) * 140 + 21000
  else if a = 20 then 21000
  else (20 - a) * 100 + 21000

-- Theorem statement
theorem cost_price_theorem :
  1 = 1 :=
begin
  sorry
end

theorem profit_function_theorem :
  ∀ m, valid_m m → profit_W m = 20 * m + 21000 :=
begin
  sorry
end

theorem max_profit_theorem :
  ∀ m a, valid_m m → (max_profit a m = if 0 < a ∧ a < 20 then (20 - a) * 140 + 21000
                                         else if a = 20 then 21000
                                         else (20 - a) * 100 + 21000) :=
begin
  sorry
end

end cost_price_theorem_profit_function_theorem_max_profit_theorem_l565_565518


namespace prob_ξ_greater_than_7_l565_565287

noncomputable def ξ : ℝ → ℝ := sorry -- Scratches actual random variable behaviour setup.

open ProbabilityTheory MeasureTheory

variable {P : Measure ℝ}
variable {σ : ℝ}
variable (h₁ : gaussian P ξ 5 σ)
variable (h₂ : P {ω | 3 ≤ ξ ω ∧ ξ ω ≤ 7} = 0.4)

theorem prob_ξ_greater_than_7 : P {ω | ξ ω > 7} = 0.3 := 
sorry

end prob_ξ_greater_than_7_l565_565287


namespace sum_of_divisors_of_24_l565_565053

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565053


namespace regular_hexagon_area_in_octahedron_l565_565714

noncomputable def area_of_hexagon_in_octahedron (a : ℝ) : ℝ := (3 * Real.sqrt 3 / 8) * a^2

theorem regular_hexagon_area_in_octahedron (a : ℝ) (h : a > 0) : 
  ∃ (hex_area : ℝ), hex_area = (3 * Real.sqrt 3 / 8) * a^2 :=
begin
  use area_of_hexagon_in_octahedron a,
  sorry,
end

end regular_hexagon_area_in_octahedron_l565_565714


namespace angle_and_distance_between_lines_l565_565711

-- Definitions of given conditions
def AB := 3
def BC := 2
def CC1 := 4
def AM_ratio_MB := 1 / 2
def K_is_intersection_of_diagonals : Prop := -- (would be the statement that K is the intersection of the diagonals of CC1D1D face)

-- We define point coordinates based on the given conditions
def D1 := (0, 0, 0 : ℝ × ℝ × ℝ)
def M := (1, 2, 4 : ℝ × ℝ × ℝ)
def B1 := (3, 2, 0 : ℝ × ℝ × ℝ)
def K := (3/2, 0, 2 : ℝ × ℝ × ℝ)

-- Vectors for the given lines
def D1M := (1, 2, 4 : ℝ × ℝ × ℝ)
def B1K := (-3/2, -2, 2 : ℝ × ℝ × ℝ)

-- Prove that the angle and distance between the lines are the given values
theorem angle_and_distance_between_lines :
  ∃ α δ, α = Real.arccos(1 / Real.sqrt 21) ∧ δ = 20 / Real.sqrt 209 :=
  sorry

end angle_and_distance_between_lines_l565_565711


namespace two_persons_finish_job_together_in_eight_days_l565_565844

-- Definitions of the conditions
def first_person_work_rate := 1 / 24
def second_person_work_rate := 1 / 12

-- Combined work rate
def combined_work_rate := first_person_work_rate + second_person_work_rate

-- Proving that the combined work rate corresponds to the job being done in 8 days
theorem two_persons_finish_job_together_in_eight_days :
  (1 / combined_work_rate = 8) :=
by
  have h_combined_work_rate : combined_work_rate = (1 / 24) + (1 / 12) := rfl
  rw [h_combined_work_rate]
  have : combined_work_rate = (1 + 2) / 24 := by ring
  rw this
  have : combined_work_rate = 1 / 8 := by norm_num
  rw this
  norm_num
  sorry

end two_persons_finish_job_together_in_eight_days_l565_565844


namespace largest_int_square_3_digits_base_7_l565_565346

theorem largest_int_square_3_digits_base_7 :
  ∃ (N : ℕ), (7^2 ≤ N^2) ∧ (N^2 < 7^3) ∧ 
  ∃ k : ℕ, N = k ∧ k^2 ≥ 7^2 ∧ k^2 < 7^3 ∧
  N = 45 := sorry

end largest_int_square_3_digits_base_7_l565_565346


namespace condition_for_a_l565_565313

theorem condition_for_a (a : ℝ) :
  (∀ x : ℤ, (x < 0 → (x + a) / 2 ≥ 1) → (x = -1 ∨ x = -2)) ↔ 4 ≤ a ∧ a < 5 :=
by
  sorry

end condition_for_a_l565_565313


namespace find_k_l565_565977

-- Definitions
def f (x : ℝ) : ℝ := x + Real.sin x

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + n * d

-- Conditions
variable {a : ℕ → ℝ}
variable {d : ℝ}
variable (h_arith : arithmetic_seq a a 0 d)
variable (h_range : ∀ n, a n ∈ Ioo (- (Real.pi / 2)) (Real.pi / 2))
variable (h_sum : (∑ k in finset.range 19, f (a k)) = 0)
variable (h_neq0 : d ≠ 0)

-- Statement to be proved
theorem find_k (h_arith : arithmetic_seq a 0 d) (h_range : ∀ n, a n ∈ Ioo (- (Real.pi / 2)) (Real.pi / 2))
(h_sum : (∑ k in finset.range 19, f (a k)) = 0) (h_neq0 : d ≠ 0) : f (a 9) = 0 :=
sorry

end find_k_l565_565977


namespace determine_A_l565_565508

noncomputable def is_single_digit (n : ℕ) : Prop := n < 10

theorem determine_A (A B C : ℕ) (hABC : 3 * (100 * A + 10 * B + C) = 888)
  (hA_single_digit : is_single_digit A) (hB_single_digit : is_single_digit B) (hC_single_digit : is_single_digit C)
  (h_different : A ≠ B ∧ B ≠ C ∧ A ≠ C) : A = 2 := 
  sorry

end determine_A_l565_565508


namespace quadratic_function_vertex_upwards_exists_l565_565459

theorem quadratic_function_vertex_upwards_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x, f x = a * (x - 1) * (x - 1) - 2) :=
by
  sorry

end quadratic_function_vertex_upwards_exists_l565_565459


namespace determine_z_l565_565964

theorem determine_z (i z : ℂ) (hi : i^2 = -1) (h : i * z = 2 * z + 1) : 
  z = - (2/5 : ℂ) - (1/5 : ℂ) * i := by
  sorry

end determine_z_l565_565964


namespace number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l565_565301

-- Definitions for the sets A and B
def A : Set Int := {x | x^2 - 3 * x - 10 <= 0}
def B (m : Int) : Set Int := {x | m - 1 <= x ∧ x <= 2 * m + 1}

-- Proof for the number of non-empty proper subsets of A
theorem number_of_non_empty_proper_subsets_of_A (x : Int) (h : x ∈ A) : 2^(8 : Nat) - 2 = 254 := by
  sorry

-- Proof for the range of m such that A ⊇ B
theorem range_of_m_for_A_superset_B (m : Int) : (∀ x, x ∈ B m → x ∈ A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by
  sorry

end number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l565_565301


namespace seconds_in_8_point_5_minutes_l565_565306

def minutesToSeconds (minutes : ℝ) : ℝ := minutes * 60

theorem seconds_in_8_point_5_minutes : minutesToSeconds 8.5 = 510 := 
by
  sorry

end seconds_in_8_point_5_minutes_l565_565306


namespace isosceles_triangle_U_l565_565000

def triangle_U : {a : ℕ // a = 6 ∧ b = 6 ∧ c = 10} :=
  ⟨6, by sorry⟩  -- This shows sides of U.

noncomputable def altitude_U : ℝ := Math.sqrt (6^2 - (10/2)^2)

def perimeter_U : ℕ := 6 + 6 + 10

theorem isosceles_triangle_U' (x y : ℕ) :
  2 * x + y = 22 ∧ 
  Math.sqrt (x^2 - (y/2)^2) = Math.sqrt 11 →
  y = 10 :=
  by
  sorry

end isosceles_triangle_U_l565_565000


namespace number_of_elements_in_set_A_l565_565389

namespace MathProof

def sets (A B : Type) : Prop :=
  let a := fintype.card A
  let b := fintype.card B
  a = 3 * b ∧
  fintype.card (A ∪ B) = 4220 ∧
  fintype.card (A ∩ B) = 850

theorem number_of_elements_in_set_A (A B : Type) [fintype A] [fintype B] (h : sets A B) : 
  fintype.card A = 3165 :=
by 
  cases h with a_b h1
  cases h1 with union h2
  cases h2 with intersection h3
  have a_eq : a = 3 * b := a_b
  have union_eq : fintype.card (A ∪ B) = 4220 := union
  have intersection_eq : fintype.card (A ∩ B) = 850 := intersection
  sorry

end MathProof

end number_of_elements_in_set_A_l565_565389


namespace find_value_of_a_l565_565627

theorem find_value_of_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53 ^ 2017 + a) % 13 = 0) : a = 12 := 
by 
  sorry

end find_value_of_a_l565_565627


namespace points_with_distinct_distances_l565_565385

noncomputable theory

open Classical

-- Define the problem in Lean 4
theorem points_with_distinct_distances (n : ℕ) (α : ℝ) (h_positive : α > 0) (h_bound : α ≤ 1/7) :
  ∃ S, S ⊆ { P : ℝ × ℝ | true } ∧ S.card = ⌈n^α⌉ ∧ ∀ P Q ∈ S, P ≠ Q → dist P Q ≠ dist Q P :=
begin
  sorry
end

end points_with_distinct_distances_l565_565385


namespace line_perpendicular_l565_565687

theorem line_perpendicular (m : ℝ) : 
  -- Conditions
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → y = 1/2 * x + 5/2) →  -- Slope of the first line
  (∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = -2/m * x + 6/m) →  -- Slope of the second line
  -- Perpendicular condition
  ((1/2) * (-2/m) = -1) →
  -- Conclusion
  m = 1 := 
sorry

end line_perpendicular_l565_565687


namespace sum_of_divisors_of_24_l565_565073

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565073


namespace value_of_x_l565_565252

theorem value_of_x (x : ℝ) (a : ℝ) (h1 : x ^ 2 * 8 ^ 3 / 256 = a) (h2 : a = 450) : x = 15 ∨ x = -15 := by
  sorry

end value_of_x_l565_565252


namespace circle_equation_l565_565620

theorem circle_equation (C : ℝ × ℝ) (r : ℝ)
  (h_C : C.1 + C.2 = 2)
  (h_CA_eq_CB : (C.1 - 1)^2 + (C.2 + 1)^2 = (C.1 + 1)^2 + (C.2 - 1)^2):
  (C.1 = 1 ∧ C.2 = 1 ∧ r = 2) ∧ ((x - 1)^2 + (y - 1)^2 = 4) := 
begin
  sorry
end

end circle_equation_l565_565620


namespace max_daily_profit_at_3_l565_565853

-- Define the defect rate P as a piecewise function of x
def defect_rate (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 6 then 1 / (6 - x)
  else if x ≥ 6 then 2 / 3
  else 0

-- Define the daily profit T as a piecewise function of x
def daily_profit (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 6 then (9 * x - 2 * x^2) / (6 - x)
  else if x ≥ 6 then 0
  else 0

-- Define the maximum profit
def max_profit := 3

-- Statement to prove
theorem max_daily_profit_at_3 : ∀ x : ℝ, (1 ≤ x ∧ x < 6 → daily_profit x ≤ max_profit) ∧ (x = 3 → daily_profit x = max_profit) :=
by
  intro x
  split
  { intro h
    sorry }
  { intro h
    sorry }

end max_daily_profit_at_3_l565_565853


namespace number_of_values_times_t_l565_565359

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition1 : ∀ (x y : ℝ), g (xy + g x) = x * g y + g x
axiom g_condition2 : g 2 = 2

theorem number_of_values_times_t : 
  let n := 1 in
  let t := g (1/3) in 
  n * t = 2/3 :=
sorry

end number_of_values_times_t_l565_565359


namespace double_apply_pi_div_two_l565_565644

def f (x : ℝ) : ℝ :=
  if x >= 0 then sin x + 2 * cos (2 * x) else -exp (2 * x)

theorem double_apply_pi_div_two :
  f (f (π / 2)) = -1 / exp(2) :=
by
  sorry

end double_apply_pi_div_two_l565_565644


namespace exists_infinitely_many_n_odd_floor_l565_565834

def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem exists_infinitely_many_n_odd_floor (α : ℝ) : 
  ∃ᶠ n in at_top, odd ⌊n^2 * α⌋ := sorry

end exists_infinitely_many_n_odd_floor_l565_565834


namespace handshake_problem_l565_565526

theorem handshake_problem (players_per_team : ℕ) (teams : ℕ) (referees : ℕ) 
  (h1 : players_per_team = 6) (h2 : teams = 2) (h3 : referees = 3) :
  let total_players := players_per_team * teams in
  let inter_team_handshakes := players_per_team * players_per_team in
  let referees_handshakes := total_players * referees in
  inter_team_handshakes + referees_handshakes = 72 := by
  sorry

end handshake_problem_l565_565526


namespace smallest_n_l565_565739

-- Definitions for the problem conditions
def values_in_reals (n : ℕ) (x : ℕ → ℝ) :=
  ∀ i, 1 ≤ i ∧ i ≤ n → x i ∈ ℝ

def abs_lt_one (n : ℕ) (x : ℕ → ℝ) :=
  ∀ i, 1 ≤ i ∧ i ≤ n → |x i| < 1

def abs_sum_eq (n : ℕ) (x : ℕ → ℝ) :=
  Σ (i : ℕ) in (finset.range n), |x i| = 25 + |Σ (i : ℕ) in (finset.range n), x i|

-- Lean 4 theorem statement
theorem smallest_n (n : ℕ) (x : ℕ → ℝ) :
  values_in_reals n x →
  abs_lt_one n x →
  abs_sum_eq n x →
  n = 26 := 
sorry

end smallest_n_l565_565739


namespace parametric_second_derivative_l565_565467

-- Given conditions
def x (t : ℝ) : ℝ := Real.cos (2 * t)
def y (t : ℝ) : ℝ := 2 * (Real.sec t) ^ 2

-- Second-order derivative y_xx of y with respect to x
noncomputable def y_xx'' (t : ℝ) : ℝ := 
  by {
    let xt' := -2 * Real.sin (2 * t),
    let yt' := 4 * Real.sin t / ((Real.cos t) ^ 3),
    let yx' := yt' / xt',
    let yx_t' := -4 * Real.sin t / ((Real.cos t) ^ 5),
    exact yx_t' / xt'
  }

-- The final proof
theorem parametric_second_derivative : ∀ t : ℝ, y_xx'' t = 1 / (Real.cos t) ^ 6 :=
by {
  intros t,
  let yt' := 4 * Real.sin t / ((Real.cos t) ^ 3),
  let xt' := -2 * Real.sin (2 * t),
  let yx' := yt' / xt',
  let yx_t' := -4 * Real.sin t / ((Real.cos t) ^ 5),
  let yxx := yx_t' / xt',
  have : yxx = 1 / (Real.cos t) ^ 6,
  { simp only [yx_t', xt', Real.sin_2t, Real.sec, Real.cos],
    sorry },
  exact this
}

end parametric_second_derivative_l565_565467


namespace perpendicular_line_sum_l565_565285

theorem perpendicular_line_sum (a b c : ℝ) 
  (h1 : -a / 4 * 2 / 5 = -1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * c + b = 0) : 
  a + b + c = -4 :=
sorry

end perpendicular_line_sum_l565_565285


namespace xyz_unique_solution_l565_565155

theorem xyz_unique_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_eq : x + y^2 + z^3 = x * y * z)
  (h_gcd : z = Nat.gcd x y) : x = 5 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end xyz_unique_solution_l565_565155


namespace circle_second_x_intercept_l565_565480

theorem circle_second_x_intercept :
  ∀ (circle : ℝ × ℝ → Prop), (∀ (x y : ℝ), circle (x, y) ↔ (x - 5) ^ 2 + y ^ 2 = 25) →
    ∃ x : ℝ, (x ≠ 0 ∧ circle (x, 0) ∧ x = 10) :=
by {
  sorry
}

end circle_second_x_intercept_l565_565480


namespace brenda_age_problem_l565_565864

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end brenda_age_problem_l565_565864


namespace sum_of_divisors_of_24_is_60_l565_565061

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565061


namespace sum_of_divisors_of_24_l565_565082

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565082


namespace terminal_zeros_75_480_l565_565670

theorem terminal_zeros_75_480 :
  let x := 75
  let y := 480
  let fact_x := 5^2 * 3
  let fact_y := 2^5 * 3 * 5
  let product := fact_x * fact_y
  let num_zeros := min (3) (5)
  num_zeros = 3 :=
by
  sorry

end terminal_zeros_75_480_l565_565670


namespace exists_simple_approx_l565_565350

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

noncomputable def integrable (ξ : Ω → ℝ) : Prop := ∫⁻ ω, |(ξ ω)| ∂μ < ∞

noncomputable def simple (ξ : Ω → ℝ) : Prop := ∃ (a : ℝ), ∃ (A : Set Ω), measurable_set A ∧ ξ = λ ω, a * A.indicator (~A.compl)

theorem exists_simple_approx (ξ : Ω → ℝ) (hξ : integrable ξ) :
  ∀ ε > 0, ∃ ξ_ε : Ω → ℝ, simple ξ_ε ∧ ∫⁻ ω, |(ξ ω) - (ξ_ε ω)| ∂μ ≤ ε :=
by sorry

end exists_simple_approx_l565_565350


namespace find_S6_l565_565634

-- sum of the first n terms of an arithmetic sequence
variable (S : ℕ → ℕ)

-- Given conditions
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- Theorem statement
theorem find_S6 : S 6 = 36 := sorry

end find_S6_l565_565634


namespace tangent_angle_planes_l565_565334

open Real EuclideanGeometry

noncomputable def pyramid :: V3 ℝ := (0,0,1)
noncomputable def A :: V3 ℝ := (0,0,0)
noncomputable def B :: V3 ℝ := (1,0,0)
noncomputable def C :: V3 ℝ := (1,1,0)
noncomputable def D :: V3 ℝ := (0, 1/2, 0)
noncomputable def S := pyramid

def vec_CD : V3 ℝ := D - C
def vec_DS : V3 ℝ := S - D

def normal_SBA : V3 ℝ := (0,1,0)

noncomputable def solve_system : V3 ℝ := 
  let x := (-1 : ℝ)
  let y := (2 : ℝ)
  let z := (1 : ℝ)
  (x, y, z)

def normal_SDC : V3 ℝ := solve_system

theorem tangent_angle_planes (θ : Real) : 
  let m := normal_SBA
  let n := normal_SDC
  let cosθ := (m ⬝ n) / ((m.norm) * (n.norm))
  cosθ = (√6 / 3) →
  θ = Real.arccos (√6 / 3) →
  tan θ = (√2 / 2) := sorry

end tangent_angle_planes_l565_565334


namespace total_income_l565_565496

theorem total_income (I : ℝ) (h1 : 0.10 * I * 2 + 0.20 * I + 0.06 * (I - 0.40 * I) = 0.46 * I) (h2 : 0.54 * I = 500) : I = 500 / 0.54 :=
by
  sorry

end total_income_l565_565496


namespace polygon_RS_ST_sum_l565_565776

theorem polygon_RS_ST_sum
  (PQ RS ST: ℝ)
  (PQ_eq : PQ = 10)
  (QR_eq : QR = 7)
  (TU_eq : TU = 6)
  (polygon_area : PQ * QR = 70)
  (PQRSTU_area : 70 = 70) :
  RS + ST = 80 :=
by
  sorry

end polygon_RS_ST_sum_l565_565776


namespace sum_divisors_24_l565_565032

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565032


namespace new_polyhedron_faces_l565_565501

def regular_tetrahedron (a : ℝ) :=
  { faces := 4, edge_length := a }

def regular_octahedron (a : ℝ) :=
  { faces := 8, edge_length := a }

theorem new_polyhedron_faces (a : ℝ) :
  ∃ (new_polyhedron_faces : ℕ), 
  new_polyhedron_faces = (regular_tetrahedron a).faces - 1 + (regular_octahedron a).faces - 1 + 1 ∧
  new_polyhedron_faces = 7 := 
by
  sorry

end new_polyhedron_faces_l565_565501


namespace interest_rate_l565_565013

noncomputable def rate_of_interest (P A : ℝ) (t : ℝ) : ℝ :=
  (real.sqrt (A / P) - 1)

theorem interest_rate {P A t : ℝ} (hP : P = 400) (hA : A = 600) (ht : t = 2) :
  rate_of_interest P A t ≈ 0.2247 :=
by
  -- Conditions are explicitly defined
  have h1 : P = 400 := by assumption
  have h2 : A = 600 := by assumption
  have h3 : t = 2 := by assumption
  
  -- Define the rate of interest
  let r := (real.sqrt (A / P) - 1)
  
  -- Use the conditions
  rw [h1, h2, h3]
  
  sorry

end interest_rate_l565_565013


namespace number_of_sheets_l565_565722

theorem number_of_sheets (S E : ℕ) (h1 : S - E = 60) (h2 : 5 * E = S) : S = 150 := by
  sorry

end number_of_sheets_l565_565722


namespace problem_6_1_problem_6_2_problem_6_3_problem_6_4_problem_6_5_problem_6_6_l565_565217

theorem problem_6_1 : sqrt 2 * sqrt 3 - 5 = sqrt 6 - 5 :=
by sorry

theorem problem_6_2 : 2 * sqrt 12 + 3 * sqrt 48 = 16 * sqrt 3 :=
by sorry

theorem problem_6_3 : (sqrt 27 + sqrt (1 / 3)) * sqrt 3 = 10 :=
by sorry

theorem problem_6_4 : (sqrt 50 * sqrt 32) / sqrt 2 - 4 * sqrt (1 / 2) = 18 * sqrt 2 :=
by sorry

theorem problem_6_5 : (sqrt 20 + sqrt 5) / sqrt 5 - (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 2 :=
by sorry

theorem problem_6_6 (x : ℝ) : 3 * (x + 1) ^ 2 = 48 → x = 3 ∨ x = -5 :=
by sorry

end problem_6_1_problem_6_2_problem_6_3_problem_6_4_problem_6_5_problem_6_6_l565_565217


namespace four_digit_odd_integers_count_l565_565660

theorem four_digit_odd_integers_count : ∃ n : ℕ, n = 625 ∧ 
  ∀ d1 d2 d3 d4 : ℕ, d1 ∈ {1, 3, 5, 7, 9} → d2 ∈ {1, 3, 5, 7, 9} →
  d3 ∈ {1, 3, 5, 7, 9} → d4 ∈ {1, 3, 5, 7, 9} → 
  n = 5 ^ 4 :=
by
  sorry

end four_digit_odd_integers_count_l565_565660


namespace kate_lee_monument_time_l565_565725

theorem kate_lee_monument_time :
  ∀ (v_lee v_kate d_paths r_monument d_init : ℝ)
  (h1 : v_lee = 4) (h2 : v_kate = 2) (h3 : d_paths = 300)
  (h4 : r_monument = 75) (h5 : d_init = 500),
  let t := (100 * Real.sqrt 205) / 3 in
  (Nat.gcd 100 3 = 1) →
  let fraction_sum := 100 + 3 in
  fraction_sum = 103 :=
by
  intros
  sorry

end kate_lee_monument_time_l565_565725


namespace correct_statement_l565_565818

theorem correct_statement : 
  ¬(deterministic_event (turning_to_even_numbered_page_in_math_book)) ∧
  (0 < probability (happening_of_fable "Waiting for the Hare by the Stump") ∧ probability (happening_of_fable "Waiting for the Hare by the Stump") < 1) ∧
  ¬(definitely_win_lottery (buying_100_lottery_tickets_with_1_percent_probability_each)) ∧
  (probability_of_rain_in_Huaibei_tomorrow = 0.8 → very_likely_rain_in_Huaibei_tomorrow) →
  correct_statement = D := 
by
  sorry

end correct_statement_l565_565818


namespace solution_set_of_inequality_l565_565796

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 3*x + 4 > 0 } = { x : ℝ | -1 < x ∧ x < 4 } := 
sorry

end solution_set_of_inequality_l565_565796


namespace find_angle_B_l565_565601

-- Definition of equivalence in geometry
structure Triangle :=
  (A B C : ℝ)
  (angleA angleB angleC : ℝ)

def congruent (t1 t2 : Triangle) : Prop :=
  t1.angleA = t2.angleA ∧ t1.angleB = t2.angleB ∧ t1.angleC = t2.angleC

-- Given conditions and correct answer
def triangleABC : Triangle :=
  { A := 1, B := 1, C := 1, angleA := 30, angleB := _, angleC := _ }

def triangleDEF : Triangle :=
  { A := 1, B := 1, C := 1, angleA := _, angleB := _, angleC := 85 }

theorem find_angle_B :
  congruent triangleABC triangleDEF →
  triangleABC.angleA = 30 →
  triangleDEF.angleC = 85 →
  triangleABC.angleB = 65 :=
by sorry

end find_angle_B_l565_565601


namespace number_composition_l565_565492

theorem number_composition :
  5 * 100000 + 6 * 100 + 3 * 10 + 6 * 0.01 = 500630.06 := 
by 
  sorry

end number_composition_l565_565492


namespace total_ants_correct_l565_565510

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end total_ants_correct_l565_565510


namespace probability_event_A_occuring_exactly_once_l565_565747

noncomputable def prob_event_A_exactly_once (p : ℝ) : ℝ :=
  3 * p * (1 - p)^2

theorem probability_event_A_occuring_exactly_once :
  ∀ (p : ℝ), (1 - (1 - p)^3 = 63 / 64 → prob_event_A_exactly_once p = 9 / 64) :=
by
  assume p hp
  sorry

end probability_event_A_occuring_exactly_once_l565_565747


namespace sum_of_divisors_of_24_l565_565128

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565128


namespace wheel_revolutions_l565_565792

-- Conditions
def diameter : ℝ := 8
def miles_to_feet (miles: ℝ) : ℝ := miles * 5280
def distance : ℝ := miles_to_feet 2
def radius (d: ℝ) : ℝ := d / 2
def circumference (r: ℝ) : ℝ := 2 * Real.pi * r

-- Proof Problem
theorem wheel_revolutions :
  let r := radius diameter
  let C := circumference r
  N = distance / C
  N = 1320 / Real.pi :=
by
  sorry

end wheel_revolutions_l565_565792


namespace eval_expr_l565_565901

theorem eval_expr : |8 - 8 * (3 - 12)| - |5 - 11| = 74 :=
by
  -- Skipping the proof
  sorry

end eval_expr_l565_565901


namespace buratino_arrives_at_21_10_l565_565210

noncomputable def buratino_arrival_time (start_time faster_time_diff : ℝ) (t : ℝ) : ℝ :=
  start_time + t

theorem buratino_arrives_at_21_10 :
  let start_time := 13 + 40/60 in
  let faster_time_diff := 1.5 in
  let t := 7.5 in
  buratino_arrival_time start_time faster_time_diff t = 21 + 10/60 :=
by
  sorry

end buratino_arrives_at_21_10_l565_565210


namespace graph_translation_properties_l565_565407

theorem graph_translation_properties (f g : ℝ → ℝ) (φ : ℝ) (hφ : |φ| < Real.pi / 2) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (∀ x, g x = Real.sin (2 * (x + Real.pi / 12) + φ)) →
  (∀ x, g (-x) = g x) →
  (∀ x, x ∈ Ioo (0 : ℝ) (Real.pi / 2) → g x = Real.cos (2 * x) ∧ 
       (g x IS DECREASINGIN ON INTERVAL)) ∧
  (∀ x, g (Real.pi / 2 - x) = g (Real.pi / 2 + x)) :=
by
  sorry

end graph_translation_properties_l565_565407


namespace roots_are_same_l565_565154

theorem roots_are_same 
  {P Q : Polynomial ℝ}
  (h_nonconst : P.degree ≠ 0)
  (h_realroots : ∀ x, P.is_root x → x ∈ ℝ)
  (h_Q_exists : ∀ x, P.eval x ^ 2 = P.eval (Q.eval x)) :
  (∀ x y, P.is_root x → P.is_root y → x = y) :=
begin
  sorry
end

end roots_are_same_l565_565154


namespace sum_of_divisors_of_24_l565_565051

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565051


namespace pencil_length_l565_565463

theorem pencil_length :
  let purple := 1.5
  let black := 0.5
  let blue := 2
  purple + black + blue = 4 := by sorry

end pencil_length_l565_565463


namespace stickers_given_to_Alex_l565_565595

theorem stickers_given_to_Alex (g_l: Nat) (g_f: Nat) (g_i: Nat) (g_a: Nat):
    g_l = 42 → -- stickers given to Lucy
    g_f = 31 → -- stickers Gary had left 
    g_i = 99 → -- initial stickers Gary had 
    g_a = g_i - g_f - g_l → -- stickers given to Alex
    g_a = 26 := -- proving g_a == 26
by
    intros h1 h2 h3 h4
    rw [h4, h3, h2, Nat.sub_sub, Nat.sub_eq_iff_eq_add]
    sorry

end stickers_given_to_Alex_l565_565595


namespace count_arithmetic_sequence_terms_l565_565893

theorem count_arithmetic_sequence_terms : 
  ∃ n : ℕ, 
  (∀ k : ℕ, k ≥ 1 → 6 + (k - 1) * 4 = 202 → n = k) ∧ n = 50 :=
by
  sorry

end count_arithmetic_sequence_terms_l565_565893


namespace cauchy_functional_eq_l565_565233

theorem cauchy_functional_eq
  (f : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end cauchy_functional_eq_l565_565233


namespace physics_marks_l565_565187

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 180) 
  (h2 : P + M = 180) 
  (h3 : P + C = 140) : 
  P = 140 := 
by 
  sorry

end physics_marks_l565_565187


namespace find_a_l565_565365

theorem find_a :
  let y := λ x : ℝ, (x + 1) / (x - 1)
  let slope_at_3 := Deriv y 3
  let line_slope := -a
  slope_at_3 = -1 / 2 ∧ -a * slope_at_3 = 1
  → a = -2 :=
by
  sorry


end find_a_l565_565365


namespace coins_to_top_right_l565_565809

theorem coins_to_top_right (n : ℕ) (initial_state : ℕ × ℕ → ℕ) :
  ∃ (final_state : ℕ × ℕ → ℕ),
  (∀ i j, initial_state (i, j) ≥ 0) ∧
  (∀ i j, final_state (i, j) ≥ 0) ∧
  (initial_state (1, n) = final_state (1, n)) ∧
  (final_state (1, n) ≥ 1) ∧
  (∃ (k : ℕ), ∃ (f : fin k → (ℕ × ℕ) × (ℕ × ℕ)),
    (∀ (s : fin k),
      (initial_state (f s).fst.fst (f s).fst.snd > 1) ∧
      (final_state (f s).snd.fst (f s).snd.snd = initial_state (f s).fst.fst (f s).fst.snd - 2 + final_state (f s).snd.fst (f s).snd.snd))) :=
sorry

end coins_to_top_right_l565_565809


namespace exists_infinite_primes_l565_565894

noncomputable def P (n : ℕ) : ℤ := sorry -- this should describe the polynomial P 

theorem exists_infinite_primes (P : ℕ → ℤ) (hP_deg : 1 ≤ degree P) (hP_int : ∀ n, P n ∈ ℤ) :
  ∃ infinitely many (p : ℕ), prime p ∧ ∃ n, p ∣ P n := 
by
  sorry

end exists_infinite_primes_l565_565894


namespace quadratic_p_value_l565_565884

theorem quadratic_p_value (p n : ℝ) (h1: 0 ≤ n) :
  (∀ x : ℝ, (x^2 + p * x + 1 / 4) = ((x + n) ^ 2 - 1 / 16)) →
  p = - (sqrt 5) / 2 :=
by
  intros h
  have h2 : n^2 - 1 / 16 = 1 / 4 := 
    by { specialize h 0, linarith }
  have h3 : n^2 = 5 / 16 := 
    by { linarith }
  have h4 : n = sqrt(5) / 4 ∨ n = -(sqrt(5) / 4) := 
    by { rw [h3, sqrt_eq_iff], norm_num }
  cases h4 with h4a h4b
  { have h5 : p = 2 * n := 
      by { specialize h (-n), linarith }
    rw h4a at h5,
    norm_num at h5
  }
  { have h5 : p = 2 * n := 
      by { specialize h (-n), linarith }
    rw h4b at h5,
    norm_num at h5
  }
  return sorry
  sorry

end quadratic_p_value_l565_565884


namespace brenda_age_l565_565867

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end brenda_age_l565_565867


namespace measure_angle_ABC_l565_565793

-- Defining the problem setup
noncomputable def O : Type := sorry  -- center of the circle
noncomputable def A : Type := sorry
noncomputable def B : Type := sorry
noncomputable def C : Type := sorry
def circumscribed_circle (O A B C : Type) : Prop := sorry
def angle (A B C : Type) : Type := sorry
def measure (angle : Type) : ℝ := sorry

-- Given conditions
axiom CenterCircumscribed (O A B C : Type) : circumscribed_circle O A B C
axiom angle_AOB (O A B : Type) : measure (angle O A B) = 140
axiom angle_BOC (O B C : Type) : measure (angle O B C) = 120

-- Proof
theorem measure_angle_ABC (O A B C : Type) [circumscribed_circle O A B C] 
  (h_AOB: measure (angle O A B) = 140) (h_BOC: measure (angle O B C) = 120) : 
  measure (angle A B C) = 50 := 
sorry

end measure_angle_ABC_l565_565793


namespace union_intersection_arithmetic_expression_l565_565876

-- Defining the sets for the first problem
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -2 < x ∧ x < 2}
def C : Set ℝ := {x | -3 < x ∧ x < 5}

-- First proof problem statement
theorem union_intersection : (A ∪ B) ∩ C = { x | -2 < x ∧ x < 5 } := sorry

-- Defining the expressions for the second problem
noncomputable def expr := (9 / 4 : ℝ) ^ (1 / 2) - (- 9.6) ^ 0 - (27 / 8) ^ (- 2 / 3) + (1.5: ℝ) ^ (- 2)

-- Second proof problem statement
theorem arithmetic_expression : expr = 1 / 2 := sorry

end union_intersection_arithmetic_expression_l565_565876


namespace sum_of_divisors_of_24_is_60_l565_565066

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565066


namespace find_five_digit_numbers_l565_565237

def is_divisor (a b : ℕ) : Prop := b % a = 0

def check_sequence_property (n : ℕ) : Prop :=
  let d1 := n % 10 in
  let n1 := n / 10 in
  let d2 := n1 % 10 in
  let n2 := n1 / 10 in
  let d3 := n2 % 10 in
  let n3 := n2 / 10 in
  let d4 := n3 % 10 in
  let n4 := n3 / 10 in
  let d5 := n4 in
  is_divisor d1 d2 ∧ is_divisor (d1 * 10 + d2) d3 ∧ is_divisor ((d1 * 10 + d2) * 10 + d3) d4 ∧ is_divisor (((d1 * 10 + d2) * 10 + d3) * 10 + d4) d5

theorem find_five_digit_numbers :
  {n : ℕ // n > 10000 ∧ n < 100000 ∧ ∀ m, m ∈ [53125, 91125, 95625] → check_sequence_property m} :=
  sorry

end find_five_digit_numbers_l565_565237


namespace p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l565_565605

-- Define conditions
def p (x : ℝ) : Prop := -x^2 + 2 * x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8 * x + 20 ≥ 0

variable {x m : ℝ}

-- Question 1
theorem p_sufficient_not_necessary_for_q (hp : ∀ x, p x → q x m) : m ≥ 3 :=
sorry

-- Defining negation of s and q
def neg_s (x : ℝ) : Prop := ¬s x
def neg_q (x m : ℝ) : Prop := ¬q x m

-- Question 2
theorem neg_s_sufficient_not_necessary_for_neg_q (hp : ∀ x, neg_s x → neg_q x m) : false :=
sorry

end p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l565_565605


namespace correct_transformation_l565_565933

-- Given conditions
variables {a b : ℝ}
variable (h : 3 * a = 4 * b)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)

-- Statement of the problem
theorem correct_transformation : (a / 4) = (b / 3) :=
sorry

end correct_transformation_l565_565933


namespace range_of_k_l565_565264

noncomputable def parabola_line_intersection (k : ℝ) (b : ℝ) : Prop :=
  (k * b < 2) ∧ (k * b = 4 - 4 * k^2)

theorem range_of_k (k b : ℝ) (h : parabola_line_intersection k b) : 
  k ∈ set.Iio (-real.sqrt 2 / 2) ∪ set.Ioi (real.sqrt 2 / 2) := 
sorry

end range_of_k_l565_565264


namespace distance_from_A_to_asymptote_l565_565211

noncomputable
def distance_of_point_to_hyperbola_asymptote : ℝ :=
  distance_from_point_to_line (0, 1) (1, -2, 0)

theorem distance_from_A_to_asymptote :
  distance_of_point_to_hyperbola_asymptote = 2 * Real.sqrt 5 / 5 :=
sorry

end distance_from_A_to_asymptote_l565_565211


namespace kitten_weighs_9_l565_565173

theorem kitten_weighs_9 (x y z : ℕ) 
  (h1 : x + y + z = 36)
  (h2 : x + z = 2y)
  (h3 : x + y = z) : x = 9 :=
by
  sorry

end kitten_weighs_9_l565_565173


namespace sum_of_divisors_of_twenty_four_l565_565117

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565117


namespace find_lambda_l565_565269

-- Define four points A, B, C, D in a plane
variables (A B C D : Point)
-- Define the vectors between points
variables (AD DB CA CB CD : Vector)

-- Define the given conditions
axiom condition1 : AD = 2 • DB
axiom condition2 : CD = (1/3) • CA + λ • CB

-- The goal is to prove that λ = 3
theorem find_lambda (λ : ℝ) (A B C D : Point) (AD DB CA CB CD : Vector) 
    (h1 : AD = 2 • DB) (h2 : CD = (1/3) • CA + λ • CB) : 
    λ = 3 := 
sorry -- proof to be filled in

end find_lambda_l565_565269


namespace equal_coefficients_l565_565594

/- Define the basic setup and properties for the problem -/
structure PentVector (α : Type*) :=
  (v₁ v₂ v₃ v₄ v₅ : α)
  (eq_zero : (k₁ k₂ k₃ k₄ k₅ : ℤ) → (k₁ * v₁ + k₂ * v₂ + k₃ * v₃ + k₄ * v₄ + k₅ * v₅ = 0) → k₁ = k₂ ∧ k₂ = k₃ ∧ k₃ = k₄ ∧ k₄ = k₅)

-- The theorem to prove equal coefficients k_i
theorem equal_coefficients
  {α : Type*}
  [add_comm_group α] [module ℝ α] 
  (pent : PentVector α)
  (k₁ k₂ k₃ k₄ k₅ : ℤ)
  (h : k₁ * pent.v₁ + k₂ * pent.v₂ + k₃ * pent.v₃ + k₄ * pent.v₄ + k₅ * pent.v₅ = 0) 
  : k₁ = k₂ ∧ k₂ = k₃ ∧ k₃ = k₄ ∧ k₄ = k₅ :=
sorry

end equal_coefficients_l565_565594


namespace sequence_differs_from_two_n_l565_565520

theorem sequence_differs_from_two_n :
  (¬ (∀ n, (n = 1 ∨ n = 2 ∨ n = 3) → (2n = 2 ∨ 2n = 4 ∨ 2n = 8))) :=
by {
  intro h,
  specialize h 3,
  have h_false := h (Or.inr (Or.inr rfl)), -- n = 3
  have val_3 := 3 * 2, -- 2 * 3 = 6
  linarith, -- reduces the false assumption
  sorry -- placeholder for proof that the derived values do not satisfy the conditions.
}

end sequence_differs_from_two_n_l565_565520


namespace sum_of_divisors_of_twenty_four_l565_565114

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565114


namespace sum_of_divisors_of_24_l565_565023

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565023


namespace find_time_period_simple_interest_correct_l565_565586

theorem find_time_period :
  (1300 : ℝ) * (5 / 100) * T = (1456 : ℝ) - 1300 :=
by
  sorry

-- Definitions based on the conditions
def principal : ℝ := 1300
def amount : ℝ := 1456
def rate : ℝ := 5
def time_period := 2.4

-- Using the formula for Simple Interest: T is the time period
theorem simple_interest_correct (P A R T : ℝ) (H1 : P = principal) (H2 : A = amount) (H3 : R = rate) :
  P * (R / 100) * T = A - P ↔ T = time_period :=
by
  sorry

end find_time_period_simple_interest_correct_l565_565586


namespace scale_reading_l565_565400

theorem scale_reading (x : ℝ) (h₁ : 3.25 < x) (h₂ : x < 3.5) : x = 3.3 :=
sorry

end scale_reading_l565_565400


namespace percentage_of_masters_l565_565192

theorem percentage_of_masters (x y : ℕ) (avg_juniors avg_masters avg_team : ℚ) 
  (h1 : avg_juniors = 22)
  (h2 : avg_masters = 47)
  (h3 : avg_team = 41)
  (h4 : 22 * x + 47 * y = 41 * (x + y)) : 
  76% of the team are masters := by sorry

end percentage_of_masters_l565_565192


namespace sum_geometric_series_l565_565215

theorem sum_geometric_series :
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 195312 / 781250 := by
    sorry

end sum_geometric_series_l565_565215


namespace variance_data_l565_565614

def data : List ℝ := [4.8, 4.9, 5.2, 5.5, 5.6]

theorem variance_data (h : data = [4.8, 4.9, 5.2, 5.5, 5.6]) :
  let mean := (List.sum data) / (data.length)
  let vari := (List.sum (data.map (λ x => (x - mean) ^ 2))) / (data.length)
  vari = 0.1 :=
by
  sorry

end variance_data_l565_565614


namespace polar_equation_of_circle_l565_565653

-- Define the given rectangular coordinate equation of the circle.
def equation_rect (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x = 0

-- Define the polar coordinate conversion relations.
def polar_to_rect (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the polar coordinate equation of the circle to be proved.
def equation_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ

-- The statement to be proved.
theorem polar_equation_of_circle :
  ∀ (ρ θ : ℝ), equation_polar ρ θ ↔ ∃ (x y : ℝ), equation_rect x y ∧ polar_to_rect ρ θ = (x, y) :=
by
  sorry

end polar_equation_of_circle_l565_565653


namespace gcd_of_expression_l565_565240

noncomputable def gcd_expression (a b c d : ℤ) : ℤ :=
  (a - b) * (b - c) * (c - d) * (d - a) * (b - d) * (a - c)

theorem gcd_of_expression : 
  ∀ (a b c d : ℤ), ∃ (k : ℤ), gcd_expression a b c d = 12 * k :=
sorry

end gcd_of_expression_l565_565240


namespace perimeter_of_plot_l565_565149

noncomputable def width : ℝ := sorry -- Width of the plot
noncomputable def length : ℝ := width + 10 -- Length is 10 meters more than width
noncomputable def perimeter : ℝ := 2 * (length + width) -- Perimeter formula
noncomputable def cost_per_meter : ℝ := 6.5 -- Cost per meter for fencing
noncomputable def total_cost : ℝ := 2210 -- Total cost of fencing

theorem perimeter_of_plot :
  perimeter = 340 :=
by
  -- We have the total cost equation
  have total_cost_eq : total_cost = cost_per_meter * perimeter := rfl
  -- And solving for perimeter we get
  have calc : total_cost / cost_per_meter = perimeter := sorry
  -- Since we are given that total_cost = 2210 and cost_per_meter = 6.5
  rw [total_cost, cost_per_meter] at calc
  -- Therefore
  calc = 340 := sorry
  exact calc

end perimeter_of_plot_l565_565149


namespace ellipse_ratio_sum_l565_565547

theorem ellipse_ratio_sum :
  (∃ x y : ℝ, 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0) →
  (∃ a b : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0 → 
    (y = a * x ∨ y = b * x)) ∧ (a + b = 9)) :=
  sorry

end ellipse_ratio_sum_l565_565547


namespace num_uncovered_cells_less_than_mn_div_4_num_uncovered_cells_less_than_mn_div_5_l565_565803

-- Define the initial conditions for the board and dominoes
variables (m n : ℕ)
variables (board : matrix (fin m) (fin n) (option ℕ)) -- a matrix where option ℕ represents either empty (none) or a domino piece (some number)

-- Conditions based on the problem statement
def rect_board := ∀ i j, i < m ∧ j < n
def domino_condition := ∀ i j, board i j = some _ → (∃ i' j', board i' j' = some _ ∧ (i' = i ∧ j' = j + 1 ∨ i' = i + 1 ∧ j' = j))

-- Definitions of number of uncovered cells
def num_uncovered_cells := finset.card (finset.univ.filter (λ (i, j), board i j = none))

-- Theorems to be proven
theorem num_uncovered_cells_less_than_mn_div_4 (m n : ℕ) (board : matrix (fin m) (fin n) (option ℕ)) 
  (h1 : rect_board m n board)
  (h2 : domino_condition m n board)
  (hu : num_uncovered_cells m n board): 
  num_uncovered_cells m n board < m * n / 4 := 
by { sorry } 

theorem num_uncovered_cells_less_than_mn_div_5 (m n : ℕ) (board : matrix (fin m) (fin n) (option ℕ)) 
  (h1 : rect_board m n board)
  (h2 : domino_condition m n board)
  (hu : num_uncovered_cells m n board): 
  num_uncovered_cells m n board < m * n / 5 := 
by { sorry }

end num_uncovered_cells_less_than_mn_div_4_num_uncovered_cells_less_than_mn_div_5_l565_565803


namespace problem_statement_l565_565336

section

variables {A B C D E F G H P : Type*}
variables [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace D]
variables (segment_AB : set A) (segment_BC : set B) (segment_CD : set C) (segment_DA : set D)
variables (point_E : E) (point_F : F) (point_G : G) (point_H : H) (point_P : P)

/-- Given a space quadrilateral ABCD with points E, F, G, H sequentially taken on sides AB, BC, CD, and DA respectively,
    and the lines containing EH and FG intersect at point P, then point P must be on line BD. -/
theorem problem_statement 
    (segment_AB : A) (segment_BC : B) (segment_CD : C) (segment_DA : D)
    (point_E : E) (point_F : F) (point_G : G) (point_H : H) (point_P : P)
    (H_E_line : on_line segment_AB point_E)
    (H_F_line : on_line segment_BC point_F)
    (H_G_line : on_line segment_CD point_G)
    (H_H_line : on_line segment_DA point_H)
    (H_P_intersection : intersect_lines (line_through segment_AB point_E) (line_through segment_CD point_G) point_P ) :
    on_line (line_through segment_BC segment_DA) point_P :=
sorry

end

end problem_statement_l565_565336


namespace sum_of_divisors_of_24_l565_565102

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565102


namespace racing_magic_circle_time_l565_565418

theorem racing_magic_circle_time
  (T : ℕ) -- Time taken by the racing magic to circle the track once
  (bull_rounds_per_hour : ℕ := 40) -- Rounds the Charging Bull makes in an hour
  (meet_time_minutes : ℕ := 6) -- Time in minutes to meet at starting point
  (charging_bull_seconds_per_round : ℕ := 3600 / bull_rounds_per_hour) -- Time in seconds per Charging Bull round
  (meet_time_seconds : ℕ := meet_time_minutes * 60) -- Time in seconds to meet at starting point
  (rounds_by_bull : ℕ := meet_time_seconds / charging_bull_seconds_per_round) -- Rounds completed by the Charging Bull to meet again
  (rounds_by_magic : ℕ := meet_time_seconds / T) -- Rounds completed by the Racing Magic to meet again
  (h1 : rounds_by_magic = 1) -- Racing Magic completes 1 round in the meet time
  : T = 360 := -- Racing Magic takes 360 seconds to circle the track once
  sorry

end racing_magic_circle_time_l565_565418


namespace minimize_expression_l565_565745

theorem minimize_expression (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) ≥ 6 := 
by 
  sorry

end minimize_expression_l565_565745


namespace sum_of_divisors_of_24_l565_565135

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565135


namespace specified_time_equation_l565_565181

-- Define variables
variables (x : ℝ)
-- Conditions
def slow_horse_days : ℝ := x + 1
def fast_horse_days : ℝ := x - 3
def slow_horse_speed := 900 / slow_horse_days
def fast_horse_speed := 900 / fast_horse_days

-- Prove the equation based on the given conditions
theorem specified_time_equation : slow_horse_speed * 2 = fast_horse_speed :=
by
  -- Here we hypothesize the conditions
  have h1 : slow_horse_days = x + 1 := by rfl
  have h2 : fast_horse_days = x - 3 := by rfl
  have h3 : slow_horse_speed = 900 / (x + 1) := by rw ←h1
  have h4 : fast_horse_speed = 900 / (x - 3) := by rw ←h2
  rw [h3, h4]  -- substituting into the main equation
  sorry  -- proof omitted

end specified_time_equation_l565_565181


namespace number_of_triangles_with_whole_number_angles_l565_565669

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

noncomputable def count_triangles_with_whole_number_angles : ℕ :=
  ∑ α in (finset.Icc 1 60), 
    let max_beta := (180 - α) / 2 in
    ∑ β in (finset.Icc α max_beta),
      1

theorem number_of_triangles_with_whole_number_angles :
  count_triangles_with_whole_number_angles = 2700 := 
by
  sorry

end number_of_triangles_with_whole_number_angles_l565_565669


namespace sum_of_divisors_of_24_l565_565017

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565017


namespace triangle_inequality_l565_565588

theorem triangle_inequality
  (a b c : ℝ) (A : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ha : c^2 = a^2 + b^2 - 2 * a * b * real.cos A) 
  (p : ℝ) (hp : p = (a + b + c) / 2):
  (bc : ℝ) (Hbc : b * c = bc) :
  (hcos : real.cos A * bc / (b + c) + a < p ∧ p < (bc + a^2) / a) :=
by sorry

end triangle_inequality_l565_565588


namespace correct_statements_l565_565204

-- Definitions based on the problem's conditions
def function_relationship_is_deterministic : Prop := 
  ∀ (x y : Type), (function_relationship x y → deterministic_relationship x y)

def correlation_relationship_is_non_deterministic : Prop := 
  ∀ (x y : Type), (correlation_relationship x y → ¬deterministic_relationship x y)

def regression_analysis_for_functional_relationship : Prop :=
  ∀ (x y : Type), regression_analysis x y ↔ (function_relationship x y ∧ statistical_analysis_method x y)

def regression_analysis_for_correlation_relationship : Prop :=
  ∀ (x y : Type), regression_analysis x y ↔ (correlation_relationship x y ∧ statistical_analysis_method x y)

-- The theorem to prove
theorem correct_statements : 
  function_relationship_is_deterministic ∧ 
  correlation_relationship_is_non_deterministic ∧ 
  regression_analysis_for_correlation_relationship → 
  (statement_1_correct ∧ statement_2_correct ∧ ¬statement_3_correct ∧ statement_4_correct)
  :=
by sorry

end correct_statements_l565_565204


namespace area_of_triangle_OAB_l565_565494

noncomputable def triangle_area : ℝ :=
  let focus : ℝ × ℝ := (1, 0)
  let par_eqn : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2^2 = 4 * p.1
  let line_eqn : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2 = real.sqrt 3 * (p.1 - 1)
  let origin : ℝ × ℝ := (0, 0)
  (1 / 2) * (16 / 3) * (real.sqrt 3 / 2)

theorem area_of_triangle_OAB :
  let focus : ℝ × ℝ := (1, 0)
  let par_eqn : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2^2 = 4 * p.1
  let line_eqn : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2 = real.sqrt 3 * (p.1 - 1)
  let origin : ℝ × ℝ := (0, 0)
  triangle_area = (4 * real.sqrt 3) / 3 :=
by
  sorry

end area_of_triangle_OAB_l565_565494


namespace triple_root_possible_values_l565_565498

-- Definitions and conditions
def polynomial (x : ℤ) (b3 b2 b1 : ℤ) := x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 24

-- The proof problem
theorem triple_root_possible_values 
  (r b3 b2 b1 : ℤ)
  (h_triple_root : polynomial r b3 b2 b1 = (x * (x - 1) * (x - 2)) * (x - r) ) :
  r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 :=
by
  sorry

end triple_root_possible_values_l565_565498


namespace number_of_ways_l565_565411

/-- 
  The numbers 2, 3, 4, 5, 6, 7, 8 are to be placed in a 2x2 grid (w, x, y, z) such that:
  1. The sum of the four numbers in the first row plus two additional numbers equals 21.
  2. The total sum of these numbers + an additional vertical column also equals 21.
  We need to prove that the number of such ways to place these numbers is 72. 
-/
theorem number_of_ways : 
  let numbers := {2, 3, 4, 5, 6, 7, 8} in
  (finset.univ.card : finset (finset ℕ) → ℕ) ((finset.powerset numbers).filter
    (λ t, (t.sum id = 14 ∧ (numbers \ t).sum id = 21))) = 72 :=
by 
  sorry

end number_of_ways_l565_565411


namespace sum_of_divisors_of_24_is_60_l565_565067

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565067


namespace coordinates_of_F_minimize_circle_areas_l565_565873

noncomputable def minimized_circle_sum_coordinates : ℝ × ℝ :=
  let sqrt_expr := (3 - real.sqrt 3) in
  (1 / real.sqrt sqrt_expr, 0)

theorem coordinates_of_F_minimize_circle_areas
  (F : ℝ × ℝ)
  (hx : 0 ≤ F.1)
  (hy : F.2 = 0)
  (C : ℝ → ℝ)
  (hC : ∀ x, C x = real.sqrt (2 * F.1 * x))
  (P : ℝ × ℝ)
  (hxP : 0 < P.1)
  (hyP : 0 < P.2)
  (hP_on_parabola : P.2 = C P.1)
  (Q : ℝ × ℝ)
  (hxQ : Q.1 < 0)
  (hyQ : Q.2 = 0)
  (hPQ_tangent : (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 = 4)
  (C1 C2 : ℝ → ℝ → Prop)
  (hC1 : C1 P.1 P.2 ∧ C1 F.1 0)
  (hC2 : C2 P.1 P.2 ∧ C2 F.1 0) :
  F = minimized_circle_sum_coordinates :=
sorry

end coordinates_of_F_minimize_circle_areas_l565_565873


namespace centroid_divides_medians_l565_565791

variable {ABC : Type}
variables {A B C K M P : Point ABC}

-- Conditions: CK and AM are medians of triangle ABC, intersecting at P
def is_median (A B C D : Point ABC) : Prop :=
  (same_line C A D) ∧ (dist A D = dist A B / 2) ∧ (dist C D = dist B D)

def is_centroid (A B C P : Point ABC) : Prop :=
  (is_median A B C P) ∧ (is_median B C A P) ∧ (is_median C A B P)

-- Because of the symmetry in the problem and the nature of centroids in triangles in geometry.
theorem centroid_divides_medians (ABC : Type) (A B C K M P : Point ABC)
  (h1 : is_median A B C K)
  (h2 : is_median A C B M)
  (h3 : is_centroid A B C P)
  : ratio (dist P M) (dist P A) = 1 / 2 ∧ ratio (dist P K) (dist P C) = 1 / 2 := 
sorry

end centroid_divides_medians_l565_565791


namespace light_path_length_l565_565732

noncomputable def cube_edge : ℝ := 10
noncomputable def point_distance_BG : ℝ := 3
noncomputable def point_distance_BC : ℝ := 4

-- Define the length of the light path from the moment it leaves point A until it next reaches a vertex of the cube
theorem light_path_length (cube_edge point_distance_BG point_distance_BC : ℝ) (h1 : cube_edge = 10) (h2 : point_distance_BG = 3) (h3 : point_distance_BC = 4) :
  let r : ℤ := 50
  let s : ℤ := 5
  r * Real.sqrt (s) = Real.sqrt ( (10^2) + (4^2) + (3^2) ) * 10 := 
by -- Begin theorem
  rw [Real.sqrt_mul_self_eq_abs, Real.sqrt_add_mul_self_eq_abs] 
  simp [h1, h2, h3]
  sorry

end light_path_length_l565_565732


namespace remainder_19_pow_19_plus_19_mod_20_l565_565813

theorem remainder_19_pow_19_plus_19_mod_20 : (19 ^ 19 + 19) % 20 = 18 := 
by {
  sorry
}

end remainder_19_pow_19_plus_19_mod_20_l565_565813


namespace exists_n_for_pow_lt_e_l565_565811

theorem exists_n_for_pow_lt_e {p e : ℝ} (hp : 0 < p ∧ p < 1) (he : 0 < e) :
  ∃ n : ℕ, (1 - p) ^ n < e :=
sorry

end exists_n_for_pow_lt_e_l565_565811


namespace sum_of_divisors_of_24_is_60_l565_565045

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565045


namespace sum_divisors_24_l565_565028

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565028


namespace sum_of_divisors_of_24_l565_565020

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565020


namespace no_such_h_l565_565742

noncomputable def f (x : ℝ) := x^2 + x + 2
noncomputable def g (x : ℝ) := x^2 - x + 2
noncomputable def H (x : ℝ) := (x^2 + x + 2)^2 - (x^2 + x + 2) + 2

theorem no_such_h :
  ¬ ∃ h : ℝ → ℝ, ∀ x : ℝ, h(f x) + h(g x) = H x := 
sorry

end no_such_h_l565_565742


namespace tan_double_angle_l565_565600

variable {α β : ℝ}

theorem tan_double_angle (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 2) : Real.tan (2 * α) = -1 := by
  sorry

end tan_double_angle_l565_565600


namespace no_attention_prob_l565_565507

noncomputable def prob_no_attention (p1 p2 p3 : ℝ) : ℝ :=
  (1 - p1) * (1 - p2) * (1 - p3)

theorem no_attention_prob :
  let p1 := 0.9
  let p2 := 0.8
  let p3 := 0.6
  prob_no_attention p1 p2 p3 = 0.008 :=
by
  unfold prob_no_attention
  sorry

end no_attention_prob_l565_565507


namespace johnstown_carbon_reduction_l565_565695

theorem johnstown_carbon_reduction :
  ∀ (population : ℕ) 
    (initial_carbon_per_person : ℕ) 
    (bus_carbon : ℕ) 
    (bus_capacity : ℕ) 
    (percentage_bus_use : ℚ), 
  population = 80 →
  initial_carbon_per_person = 10 →
  bus_carbon = 100 →
  bus_capacity = 40 →
  percentage_bus_use = 0.25 →
  let people_bus = (percentage_bus_use * population).to_nat in
  let initial_carbon = population * initial_carbon_per_person in
  let remaining_drivers = population - people_bus in
  let final_carbon = (remaining_drivers * initial_carbon_per_person) + bus_carbon in
  initial_carbon - final_carbon = 100 :=
begin
  intros,
  sorry
end

end johnstown_carbon_reduction_l565_565695


namespace sum_of_divisors_of_24_l565_565072

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565072


namespace monotonic_intervals_of_f_extremum_intervals_of_g_l565_565648

noncomputable def f (x : ℝ) (a : ℝ) := a * Real.log x - a * x - 3

theorem monotonic_intervals_of_f :
  (∀ x ∈ set.Ioo 0 1, deriv (f x 1) x > 0) ∧ (∀ x ∈ set.Ioi 1, deriv (f x 1) x < 0) := sorry

noncomputable def g (x : ℝ) (m : ℝ) := x^3 + x^2 * (m / 2 + (-2 * (1 - x) / x)) - 2 * x 

theorem extremum_intervals_of_g (t : ℝ) (h₁ : t ∈ set.Icc 1 2) :
  (deriv (g t) t < 0) ∧ (deriv (g 2) 2 < 0) ∧ (deriv (g 3) 3 > 0) ↔ -37/3 < m ∧ m < -9 := sorry

end monotonic_intervals_of_f_extremum_intervals_of_g_l565_565648


namespace total_notes_count_l565_565718

theorem total_notes_count :
  ∀ (rows : ℕ) (notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ),
  rows = 5 →
  notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  (rows * notes_per_row + (rows * notes_per_row * blue_notes_per_red + additional_blue_notes)) = 100 := by
  intros rows notes_per_row blue_notes_per_red additional_blue_notes
  sorry

end total_notes_count_l565_565718


namespace trains_clear_time_l565_565004

def length_first_train : ℝ := 250  -- meters
def length_second_train : ℝ := 330  -- meters
def speed_first_train_kmh : ℝ := 120  -- km/h
def speed_second_train_kmh : ℝ := 95  -- km/h

-- Convert speeds from km/h to m/s
def speed_first_train_ms : ℝ := speed_first_train_kmh * 1000 / 3600
def speed_second_train_ms : ℝ := speed_second_train_kmh * 1000 / 3600

-- Calculate relative speed
def relative_speed := speed_first_train_ms + speed_second_train_ms

-- Calculate total distance
def total_distance := length_first_train + length_second_train

-- Calculate time for the trains to clear each other
def time_to_clear := total_distance / relative_speed

-- The theorem stating the time required
theorem trains_clear_time : time_to_clear = 9.71 := by
  sorry

end trains_clear_time_l565_565004


namespace find_x_given_conditions_l565_565136

variable (x : ℝ)

def isCubeVolume (V : ℝ) (x : ℝ) : Prop := V = 8 * x^2
def isCubeSurfaceArea (A : ℝ) (x : ℝ) : Prop := A = 4 * x

theorem find_x_given_conditions (x : ℝ) :
  isCubeVolume 8 * x^2 x ∧ isCubeSurfaceArea 4 * x x → x = 1 / 216 :=
by
  sorry

end find_x_given_conditions_l565_565136


namespace gcd_8251_6105_l565_565441

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l565_565441


namespace volume_of_sphere_is_correct_l565_565481

-- Given that the surface area of a cube is 24
def cube_surface_area : ℝ := 24

-- Define the volume of the sphere we need to prove
def volume_of_sphere : ℝ := 4 * real.sqrt 3 * real.pi

-- Define a proposition that states that the volume of the sphere is 4√3π given the cube surface area is 24
theorem volume_of_sphere_is_correct (S : ℝ) (h : S = cube_surface_area) : volume_of_sphere = 4 * real.sqrt 3 * real.pi :=
by
  -- The proof is omitted as per the instructions
  sorry

end volume_of_sphere_is_correct_l565_565481


namespace find_min_value_l565_565959

noncomputable def vector_length (v : ℝ^3) : ℝ :=
  real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def vector_dot_product (a b : ℝ^3) : ℝ :=
  a.x * b.x + a.y * b.y + a.z * b.z

noncomputable def vector_c (a b : ℝ^3) (λ : ℝ) : ℝ^3 :=
  { x := λ * a.x + (1 - λ) * b.x,
    y := λ * a.y + (1 - λ) * b.y,
    z := λ * a.z + (1 - λ) * b.z }

noncomputable def is_unit_vector (v : ℝ^3) : Prop :=
  vector_length v = 1

theorem find_min_value (a b : ℝ^3) (h1 : ¬ collinear ℝ a b) (h2 : is_unit_vector a) (h3 : is_unit_vector b) (λ : ℝ) (h4 : vector_length (vector_c a b λ) = 0.5) :
  ∃ λ, (vector_length (a - b) = real.sqrt 3) :=
by sorry

end find_min_value_l565_565959


namespace sum_of_divisors_of_24_l565_565132

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565132


namespace width_of_domain_of_g_l565_565677

variable (h : ℝ → ℝ) (dom_h : ∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x)

noncomputable def g (x : ℝ) : ℝ := h (x / 3)

theorem width_of_domain_of_g :
  (∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x) →
  (∀ y : ℝ, -30 ≤ y ∧ y ≤ 30 → h (y / 3) = h (y / 3)) →
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧  (∃ w : ℝ, w = b - a ∧ w = 60)) :=
by
  sorry

end width_of_domain_of_g_l565_565677


namespace find_q_l565_565309

theorem find_q (p q : ℚ) (h1 : 5 * p + 7 * q = 20) (h2 : 7 * p + 5 * q = 26) : q = 5 / 12 := by
  sorry

end find_q_l565_565309


namespace sum_of_divisors_of_24_l565_565130

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565130


namespace probability_of_arithmetic_sequence_l565_565482

theorem probability_of_arithmetic_sequence :
  let outcomes := (finset.pi (finset.range 3) (λ _, finset.range 6))
  let arithmetic_sequences := outcomes.filter (λ (s : (finset.range 3) → ℕ), 
    ((2 * s 1 = s 0 + s 2) ∨ (s 0 = s 1 ∧ s 1 = s 2)))
  let total_outcomes := outcomes.card
  let favorable_outcomes := arithmetic_sequences.card
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 12 := 
by
  sorry

end probability_of_arithmetic_sequence_l565_565482


namespace pyramid_coloring_ways_l565_565926

theorem pyramid_coloring_ways (colors : Fin 5) 
  (coloring_condition : ∀ (a b : Fin 5), a ≠ b) :
  ∃ (ways: Nat), ways = 420 :=
by
  -- Given:
  -- 1. There are 5 available colors
  -- 2. Each vertex of the pyramid is colored differently from the vertices connected by an edge
  -- Prove:
  -- There are 420 ways to color the pyramid's vertices
  sorry

end pyramid_coloring_ways_l565_565926


namespace sum_of_divisors_of_24_l565_565107

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565107


namespace sum_divisors_24_l565_565031

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565031


namespace triangle_area_lt_sqrt3_over_4_l565_565761

theorem triangle_area_lt_sqrt3_over_4
  {a b c : ℝ} (ha : a < 1) (hb : b < 1) (hc : c < 1) :
  (∃ α : ℝ, 0 < α ∧ α ≤ π / 3 ∧ (a / (sin α) = b / (sin (π / 3 - α)) ∧ a / (sin α) = c / (sin (π - α - (π / 3 - α))))) →
  ∃ S : ℝ, S = 1 / 2 * b * c * sin (π / 3 - α) ∧ S < sqrt 3 / 4 := 
sorry

end triangle_area_lt_sqrt3_over_4_l565_565761


namespace gcd_8251_6105_l565_565437

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l565_565437


namespace calc_154_1836_minus_54_1836_l565_565221

-- Statement of the problem in Lean 4
theorem calc_154_1836_minus_54_1836 : 154 * 1836 - 54 * 1836 = 183600 :=
by
  sorry

end calc_154_1836_minus_54_1836_l565_565221


namespace minimum_value_l565_565737

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x^2 + y^2) * (4 * x^2 + 2 * y^2) ≥ (2 + real.sqrt  2)^2 * (x * y)^2 :=
by
  sorry

end minimum_value_l565_565737


namespace relationship_y1_y2_y3_l565_565272

def on_hyperbola (x y k : ℝ) : Prop := y = k / x

theorem relationship_y1_y2_y3 (y1 y2 y3 k : ℝ) (h1 : on_hyperbola (-5) y1 k) (h2 : on_hyperbola (-1) y2 k) (h3 : on_hyperbola 2 y3 k) (hk : k > 0) :
  y2 < y1 ∧ y1 < y3 :=
sorry

end relationship_y1_y2_y3_l565_565272


namespace area_of_smallest_square_containing_circle_l565_565446

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ a : ℝ, a = 196 := 
by
  use (2 * r) ^ 2
  rw h
  norm_num

end area_of_smallest_square_containing_circle_l565_565446


namespace sum_of_divisors_of_24_is_60_l565_565068

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565068


namespace problem_equivalence_l565_565235

theorem problem_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (⌊(a^2 : ℚ) / b⌋ + ⌊(b^2 : ℚ) / a⌋ = ⌊(a^2 + b^2 : ℚ) / (a * b)⌋ + a * b) ↔
  (∃ k : ℕ, k > 0 ∧ ((a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k))) :=
sorry

end problem_equivalence_l565_565235


namespace mel_weight_l565_565532

variable (m : ℕ)

/-- Brenda's weight is 10 pounds more than three times Mel's weight. 
    Given Brenda's weight is 220 pounds, we prove Mel's weight is 70 pounds. -/
theorem mel_weight : (3 * m + 10 = 220) → (m = 70) :=
by
  intros h,
  sorry

end mel_weight_l565_565532


namespace integral_f_eq_l565_565647

noncomputable def f : ℝ → ℝ :=
λ x, if x ∈ Set.Icc (-π) 0 then sin x else if x ∈ Set.Ioc 0 1 then real.sqrt(1 - x^2) else 0

theorem integral_f_eq :
  ∫ x in -π..1, f x = (π / 4) - 2 :=
by
  sorry

end integral_f_eq_l565_565647


namespace ball_hit_ground_time_l565_565782

theorem ball_hit_ground_time (t : ℝ) : 
  -16 * t ^ 2 + 16 * t + 50 = 0 → t = 2 + 3 * √6 / 4 := 
by
  sorry

end ball_hit_ground_time_l565_565782


namespace semesters_per_year_l565_565341

-- Definitions of conditions
def cost_per_semester : ℕ := 20000
def total_cost_13_years : ℕ := 520000
def years : ℕ := 13

-- Main theorem to prove
theorem semesters_per_year (S : ℕ) (h1 : total_cost_13_years = years * (S * cost_per_semester)) : S = 2 := by
  sorry

end semesters_per_year_l565_565341


namespace bug_least_distance_on_cone_l565_565502

theorem bug_least_distance_on_cone
  (radius : ℝ) (height : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  radius = 500 →
  height = 150 * Real.sqrt 7 →
  d1 = 100 →
  d2 = 300 * Real.sqrt 2 →
  ∃ (distance : ℝ), distance = 500 :=
by
  intros hradius hheight hd1 hd2
  use 500
  sorry

end bug_least_distance_on_cone_l565_565502


namespace sum_of_divisors_of_24_is_60_l565_565047

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565047


namespace fraction_shaded_quilt_l565_565546

/-- A quilt is a square made from 16 smaller squares, arranged in a 4x4 grid.
One entire row and one entire column are shaded.
We aim to prove that the fraction of the quilt that is shaded is 7/16. -/
theorem fraction_shaded_quilt : 
  (let total_squares := 16 in
  let shaded_squares := 7 in
  shaded_squares / total_squares = 7 / 16) :=
by
  let total_squares := 16
  let shaded_squares := 7
  have h : shaded_squares / total_squares = 7 / 16 := sorry
  exact h

end fraction_shaded_quilt_l565_565546


namespace non_obtuse_triangle_medians_ge_4R_l565_565386

theorem non_obtuse_triangle_medians_ge_4R
  (A B C : Type*)
  (triangle_non_obtuse : ∀ (α β γ : ℝ), α ≤ 90 ∧ β ≤ 90 ∧ γ ≤ 90)
  (m_a m_b m_c : ℝ)
  (R : ℝ)
  (h1 : AO + BO ≤ AM + BM)
  (h2 : AM = 2 * m_a / 3 ∧ BM = 2 * m_b / 3)
  (h3 : AO + BO = 2 * R)
  (h4 : m_c ≥ R) : 
  m_a + m_b + m_c ≥ 4 * R :=
by
  sorry

end non_obtuse_triangle_medians_ge_4R_l565_565386


namespace birth_date_of_id_number_l565_565812

def extract_birth_date (id_number : String) := 
  let birth_str := id_number.drop 6 |>.take 8
  let year := birth_str.take 4
  let month := birth_str.drop 4 |>.take 2
  let day := birth_str.drop 6
  (year, month, day)

theorem birth_date_of_id_number :
  extract_birth_date "320106194607299871" = ("1946", "07", "29") := by
  sorry

end birth_date_of_id_number_l565_565812


namespace minnie_mounts_time_period_l565_565551

theorem minnie_mounts_time_period (M D : ℕ) 
  (mickey_daily_mounts_eq : 2 * M - 6 = 14)
  (minnie_mounts_per_day_eq : M = D + 3) : 
  D = 7 := 
by
  sorry

end minnie_mounts_time_period_l565_565551


namespace invalid_diagonal_set_not_right_regular_prism_l565_565565

theorem invalid_diagonal_set_not_right_regular_prism : 
  ¬ (∃ (a b c : ℕ), 
    ({9, 12, 15} = {Int.nat_abs (a - b), Int.nat_abs (b - c), Int.nat_abs (a - c)} ∨ 
                   {9, 12, 15} = {Int.nat_abs (a - b), Int.nat_abs (a - c), Int.nat_abs (b - c)} ∨ 
                   {9, 12, 15} = {Int.nat_abs (b - a), Int.nat_abs (c - a), Int.nat_abs (c - b)}) ∧ 
    a^2 + b^2 = 9^2 ∧ 
    b^2 + c^2 = 12^2 ∧ 
    a^2 + c^2 = 15^2) :=
sorry

end invalid_diagonal_set_not_right_regular_prism_l565_565565


namespace people_in_each_column_l565_565702

theorem people_in_each_column
  (P : ℕ)
  (x : ℕ)
  (h1 : P = 16 * x)
  (h2 : P = 12 * 40) :
  x = 30 :=
sorry

end people_in_each_column_l565_565702


namespace total_tiles_on_floor_l565_565544

theorem total_tiles_on_floor :
  ∀ (n : ℕ), 
    (∃ (s : ℕ), (s = 2 * 49 - 1) ∧ (n = s * s)) →
    n = 9409 :=
by 
  intro n
  intro h
  cases h with s hs
  cases hs with hs1 hs2
  rw hs2
  repeat { sorry }

end total_tiles_on_floor_l565_565544


namespace pastries_remaining_l565_565527

theorem pastries_remaining (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 7)
  (h2 : pastries_made = 148)
  (h3 : cakes_sold = 15)
  (h4 : pastries_sold = 103) 
  : pastries_made - pastries_sold = 45 :=
by {
    rw [h2, h4],
    norm_num,
    sorry
}

end pastries_remaining_l565_565527


namespace hanging_painting_l565_565658

-- Define the nails and their inverses
variables {G : Type*} [Group G] (a : Fin n → G)

theorem hanging_painting (n : ℕ) (a : Fin n → G) :
  (∀ (b : Fin n → G), b ≠ a → mul b 1) ∧
  (∃ (c : Fin n → G), c = a → mul c 1) :=
sorry

end hanging_painting_l565_565658


namespace maximum_gel_pens_l565_565475

theorem maximum_gel_pens 
  (x y z : ℕ) 
  (h1 : x + y + z = 20)
  (h2 : 10 * x + 50 * y + 80 * z = 1000)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) 
  : y ≤ 13 :=
sorry

end maximum_gel_pens_l565_565475


namespace solution_a_eq_2_solution_a_in_real_l565_565649

-- Define the polynomial inequality for the given conditions
def inequality (x : ℝ) (a : ℝ) : Prop := 12 * x ^ 2 - a * x > a ^ 2

-- Proof statement for when a = 2
theorem solution_a_eq_2 :
  ∀ x : ℝ, inequality x 2 ↔ (x < - (1 : ℝ) / 2) ∨ (x > (2 : ℝ) / 3) :=
sorry

-- Proof statement for when a is in ℝ
theorem solution_a_in_real (a : ℝ) :
  ∀ x : ℝ, inequality x a ↔
    if h : 0 < a then (x < - a / 4) ∨ (x > a / 3)
    else if h : a = 0 then (x ≠ 0)
    else (x < a / 3) ∨ (x > - a / 4) :=
sorry

end solution_a_eq_2_solution_a_in_real_l565_565649


namespace equal_constants_l565_565622

theorem equal_constants (a b : ℝ) :
  (∃ᶠ n in at_top, ⌊a * n + b⌋ ≥ ⌊a + b * n⌋) →
  (∃ᶠ m in at_top, ⌊a + b * m⌋ ≥ ⌊a * m + b⌋) →
  a = b :=
by
  sorry

end equal_constants_l565_565622


namespace sum_of_divisors_of_24_l565_565129

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565129


namespace fraction_simplified_sum_l565_565815

theorem fraction_simplified_sum (a b : ℕ) (h : (75 : ℕ) = 3 * 5^2) (h2 : (180 : ℕ) = 2^2 * 3^2 * 5) : 
  let gcd := Nat.gcd 75 180 in
  gcd = 15 →
  a = 75 / gcd →
  b = 180 / gcd →
  a + b = 17 := 
by 
  sorry

end fraction_simplified_sum_l565_565815


namespace circle_equation_has_valid_k_l565_565307

theorem circle_equation_has_valid_k (k : ℝ) : (∃ a b r : ℝ, r > 0 ∧ ∀ x y : ℝ, (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ k < 5 / 4 := by
  sorry

end circle_equation_has_valid_k_l565_565307


namespace arithmetic_sequence_positive_l565_565750

theorem arithmetic_sequence_positive (d a_1 : ℤ) (n : ℤ) :
  (a_11 - a_8 = 3) -> 
  (S_11 - S_8 = 33) ->
  (n > 0) ->
  a_1 + (n-1) * d > 0 ->
  n = 10 :=
by
  sorry

end arithmetic_sequence_positive_l565_565750


namespace tangent_line_at_x1_l565_565646

def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_x1 :
  ∃ (m b : ℝ), (∀ x, f x = m * x + b) ∧ (m = 1) ∧ (b = -1) :=
by {
  unfold f,
  existsi (1 : ℝ),
  existsi (-1 : ℝ),
  intro x,
  sorry -- proof steps are skipped as per instructions
}

end tangent_line_at_x1_l565_565646


namespace farmer_purchase_l565_565854

theorem farmer_purchase : ∃ r c : ℕ, 30 * r + 45 * c = 1125 ∧ r > 0 ∧ c > 0 ∧ r = 3 ∧ c = 23 := 
by 
  sorry

end farmer_purchase_l565_565854


namespace first_set_opposite_faces_can_be_any_color_l565_565851

def cubelet_painted_with_exactly_one_color (total_cubelets : ℕ) (dimension : ℕ) (num_colors : ℕ) (num_single_color_cubelets : ℕ) : Prop :=
  total_cubelets = dimension^3 ∧
  num_single_color_cubelets = (dimension - 2)^2 * 2 * num_colors

theorem first_set_opposite_faces_can_be_any_color :
  ∀ (total_cubelets dimension num_colors num_single_color_cubelets : ℕ),
    total_cubelets = 216 →
    dimension = 6 →
    num_colors = 3 →
    num_single_color_cubelets = 96 →
    cubelet_painted_with_exactly_one_color total_cubelets dimension num_colors num_single_color_cubelets →
    ∃ (color : Type), color = "first set" ∨ color = "red" ∨ color = "blue" :=
by
  intros total_cubelets dimension num_colors num_single_color_cubelets h1 h2 h3 h4 H
  existsi "first set"
  left
  rfl
  sorry

end first_set_opposite_faces_can_be_any_color_l565_565851


namespace import_tax_percentage_l565_565487

-- Define conditions
def total_value : ℝ := 2580
def tax_exemption : ℝ := 1000
def import_tax_paid : ℝ := 110.60

-- Define the taxable portion
def taxable_portion : ℝ := total_value - tax_exemption

-- Define the percentage calculation
def percentage_of_import_tax : ℝ := (import_tax_paid / taxable_portion) * 100

-- The statement we want to prove
theorem import_tax_percentage : percentage_of_import_tax = 7 := 
by
  -- We skip the actual proof steps here by using "sorry"
  sorry

end import_tax_percentage_l565_565487


namespace mode_and_median_of_pocket_money_usage_l565_565434

def daily_pocket_money_usage : list ℕ := [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 6]

theorem mode_and_median_of_pocket_money_usage :
  (list.mode daily_pocket_money_usage = 2) ∧ (list.median daily_pocket_money_usage = 3) :=
by
  -- We specify the table as a list of daily pocket money usage.
  -- Then, we need to use the mode and median functions from Lean's list library.
  sorry

end mode_and_median_of_pocket_money_usage_l565_565434


namespace min_value_x_l565_565981

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end min_value_x_l565_565981


namespace find_k_factor_polynomial_l565_565563

noncomputable def polynomial := Polynomial ℝ

-- Given polynomial is 4x^3 - 16x^2 + kx - 24
def p (k : ℝ) : polynomial := 4 * polynomial.X^3 - 16 * polynomial.X^2 + k * polynomial.X - 24

-- Prove that there exists k such that the polynomial is divisible by (x - 4)
theorem find_k :
  ∃ k, (p k).modByMonic (polynomial.X - 4) = 0 := 
begin
  use 6,
  rw [p, Polynomial.modByMonic_eq_zero_of_is_root, Polynomial.is_root, Polynomial.eval, Polynomial.eval₂_eq_eval_map, Polynomial.eval_map],
  norm_num,
end

-- Given the previously found k, check if 4x^2 - 6 is a factor of the polynomial
theorem factor_polynomial :
  (p 6) = (polynomial.X - 4) * (4 * polynomial.X^2 - 6) :=
begin
  rw [p, Polynomial.mul_sub, Polynomial.mul_sub],
  -- Polynomial multiplication and expansion follow here
  sorry,
end

end find_k_factor_polynomial_l565_565563


namespace product_of_real_parts_eq_neg_half_l565_565403

noncomputable def prod_real_parts_of_solutions (h : ℂ) (x : ℂ) :=
  let s1 := x - 2 + complex.sqrt 5 * complex.exp (complex.I * real.atan (3 / 4) / 2) in
  let s2 := x - 2 - complex.sqrt 5 * complex.exp (complex.I * real.atan (3 / 4) / 2) in
  re s1 * re s2

theorem product_of_real_parts_eq_neg_half (c1 c2 : ℂ) (h : c1^2 - 4 * c1 = 3 * complex.I)
  (h2 : c2^2 - 4 * c2 = 3 * complex.I) :
  re c1 * re c2 = -0.5 :=
by
  sorry

end product_of_real_parts_eq_neg_half_l565_565403


namespace compound_interest_rate_l565_565188

noncomputable def annual_interest_rate : ℝ :=
  4 * ((2:ℝ)^(1 / 40) - 1)

theorem compound_interest_rate :
  annual_interest_rate ≈ 0.06936 :=
by
  sorry

end compound_interest_rate_l565_565188


namespace triangle_area_x_l565_565924

theorem triangle_area_x (x : ℝ) (h_pos : x > 0) (h_area : 1 / 2 * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
sorry

end triangle_area_x_l565_565924


namespace relationship_of_new_stationary_points_l565_565555

noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def phi (x : ℝ) : ℝ := x^3

noncomputable def g' (x : ℝ) : ℝ := Real.cos x
noncomputable def h' (x : ℝ) : ℝ := 1 / x
noncomputable def phi' (x : ℝ) : ℝ := 3 * x^2

-- Definitions of the new stationary points
noncomputable def new_stationary_point_g (x : ℝ) : Prop := g x = g' x
noncomputable def new_stationary_point_h (x : ℝ) : Prop := h x = h' x
noncomputable def new_stationary_point_phi (x : ℝ) : Prop := phi x = phi' x

theorem relationship_of_new_stationary_points :
  ∃ (a b c : ℝ), (0 < a ∧ a < π) ∧ (1 < b ∧ b < Real.exp 1) ∧ (c ≠ 0) ∧
  new_stationary_point_g a ∧ new_stationary_point_h b ∧ new_stationary_point_phi c ∧
  c > b ∧ b > a :=
by
  sorry

end relationship_of_new_stationary_points_l565_565555


namespace C_work_rate_equals_A_work_rate_l565_565477

variables (W : ℝ) (A B C : ℝ)

-- A's work rate
def work_rate_A : ℝ := W / 3

-- B and C's combined work rate
def work_rate_BC : ℝ := W / 2

-- A and B's combined work rate
def work_rate_AB : ℝ := W / 2

-- Prove that C's time to complete the work alone is 3 hours
theorem C_work_rate_equals_A_work_rate :
  (∃ C : ℝ, work_rate_A = W / 3 ∧ (B + C = W / 2) ∧ (A + B = W / 2) ∧ A = W / 3 ∧ B = W / 6 ∧ C = W / 3) → 
  ( W / C = 3) := 
sorry

end C_work_rate_equals_A_work_rate_l565_565477


namespace angle_EMF_90_l565_565741

def circles_meeting (S1 S2: Set Point) (A B: Point): Prop := 
  -- S1 and S2 meet at points A and B
  A ∈ S1 ∧ A ∈ S2 ∧ B ∈ S1 ∧ B ∈ S2

def line_through_A_meeting_circles (S1 S2: Set Point) (A C D: Point) (line: Set Point): Prop :=
  -- Line through A meets S1 at C and S2 at D
  A ∈ line ∧ C ∈ line ∧ C ∈ S1 ∧ D ∈ line ∧ D ∈ S2

def points_on_segments (M N K: Point) (CD BC BD: Set Point): Prop :=
  -- Points M on CD, N on BC, and K on BD
  M ∈ CD ∧ N ∈ BC ∧ K ∈ BD

def parallel_lines (MN BD: Set Point) (MK BC: Set Point): Prop :=
  -- MN parallel to BD and MK parallel to BC
  parallel MN BD ∧ parallel MK BC

def points_on_arcs (E F: Point) (arcBC arcBD: Set Point): Prop :=
  -- Points E on arc BC of S1 and F on arc BD of S2 not containing A
  E ∈ arcBC ∧ F ∈ arcBD

def perpendicular_to_lines (EN FK: Set Point) (BC BD: Set Point): Prop :=
  -- EN perpendicular to BC and FK perpendicular to BD
  perpendicular EN BC ∧ perpendicular FK BD

theorem angle_EMF_90 (S1 S2: Set Point)(A B C D M N K E F: Point) 
(line CD BC BD MN MK arcBC arcBD EN FK: Set Point):
  circles_meeting S1 S2 A B →
  line_through_A_meeting_circles S1 S2 A C D line →
  points_on_segments M N K CD BC BD →
  parallel_lines MN BD MK BC →
  points_on_arcs E F arcBC arcBD →
  perpendicular_to_lines EN FK BC BD →
  ∠EMF = 90 :=
sorry

end angle_EMF_90_l565_565741


namespace range_a_l565_565940

def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 1 then x^2 - 3*x + 2*a 
  else x - a * Real.log x

theorem range_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ 1 ≤ a ∧ a ≤ Real.exp 1 := by
  sorry

end range_a_l565_565940


namespace C_is_20_years_younger_l565_565427

variable (A B C : ℕ)

-- Conditions from the problem
axiom age_condition : A + B = B + C + 20

-- Theorem representing the proof problem
theorem C_is_20_years_younger : A = C + 20 := sorry

end C_is_20_years_younger_l565_565427


namespace cauchy_schwarz_inequality_maximum_value_sqrt_expression_l565_565512

theorem cauchy_schwarz_inequality (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := 
by sorry

theorem maximum_value_sqrt_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  sqrt (3 * a + 1) + sqrt (3 * b + 1) ≤ sqrt 10 := 
by sorry

end cauchy_schwarz_inequality_maximum_value_sqrt_expression_l565_565512


namespace angle_opposite_side_c_is_60_degrees_l565_565704

theorem angle_opposite_side_c_is_60_degrees
    (a b c : ℝ)
    (h : (a + b + c) * (a + b - c) = 2 * a * b + a^2) :
    let C := Real.arccos (1/2) in
    C = 60 :=
by
  sorry

end angle_opposite_side_c_is_60_degrees_l565_565704


namespace boxes_of_thin_mints_l565_565880

-- Define the conditions
variable (x : ℕ) -- number of boxes of thin mints

-- Constants
def priceSamoas : ℝ := 4
def priceThinMints : ℝ := 3.5
def priceFudgeDelights : ℝ := 5
def priceSugarCookies : ℝ := 2
def totalAmount : ℝ := 42

-- Amounts raised from other cookies
def amountSamoas : ℝ := 3 * priceSamoas
def amountFudgeDelights : ℝ := 1 * priceFudgeDelights
def amountSugarCookies : ℝ := 9 * priceSugarCookies

-- Sum of amounts from other cookies
def otherCookiesTotal : ℝ := amountSamoas + amountSugarCookies + amountFudgeDelights

-- Amount from thin mints
def amountThinMints (x : ℕ) : ℝ := x * priceThinMints

-- Total amount constraint
def totalAmountConstraint (x : ℕ) : Prop :=
  amountSamoas + amountSugarCookies + amountFudgeDelights + amountThinMints x = totalAmount

-- Prove the number of boxes of thin mints is 2
theorem boxes_of_thin_mints : totalAmountConstraint 2 :=
by
  unfold totalAmountConstraint
  unfold amountSamoas amountSugarCookies amountFudgeDelights amountThinMints totalAmount
  calc
    12 + 18 + 5 + (2 * 3.5) = 42 : by norm_num

#check boxes_of_thin_mints -- Confirm the theorem is correctly formed

end boxes_of_thin_mints_l565_565880


namespace sum_fractions_4011_l565_565367

def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem sum_fractions_4011 : 
  (∑ i in finset.range 4010, f ((i + 1) / 4011)) = 2005 :=
by
  sorry

end sum_fractions_4011_l565_565367


namespace sum_of_divisors_of_24_l565_565022

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565022


namespace range_of_function_l565_565560

noncomputable theory

open Real

theorem range_of_function :
  ∀ x ∈ Icc (-1 : ℝ) 1, 
  ∃ y, y ∈ Icc 0 π ∧ (f x = y) where 
  f x := arcsin x + arccos x + 2 * arctan x := 
by
  intro x hx
  have h_id : arcsin x + arccos x = (π / 2) := sorry
  have h2 : ∀ x ∈ Icc (-1 : ℝ) 1, 2 * arctan x ∈ Icc (-π / 2) (π / 2) := sorry
  sorry

end range_of_function_l565_565560


namespace inequality_sum_geq_three_l565_565731

theorem inequality_sum_geq_three
  (a b c : ℝ)
  (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) + 
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := 
sorry

end inequality_sum_geq_three_l565_565731


namespace max_integers_gt_18_l565_565797

theorem max_integers_gt_18 (a b c d e : ℤ) (sum_condition : a + b + c + d + e = 17) :
  ∃ k, k ≤ 5 ∧ (∀ i ∈ {a, b, c, d, e}, (19 ∣ i ↔ i > 18)) → k = 2 :=
  sorry

end max_integers_gt_18_l565_565797


namespace calculate_savings_l565_565769

noncomputable def monthly_salary : ℕ := 10000
noncomputable def spent_on_food (S : ℕ) : ℕ := (40 * S) / 100
noncomputable def spent_on_rent (S : ℕ) : ℕ := (20 * S) / 100
noncomputable def spent_on_entertainment (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def spent_on_conveyance (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def total_spent (S : ℕ) : ℕ := spent_on_food S + spent_on_rent S + spent_on_entertainment S + spent_on_conveyance S
noncomputable def amount_saved (S : ℕ) : ℕ := S - total_spent S

theorem calculate_savings : amount_saved monthly_salary = 2000 :=
by
  sorry

end calculate_savings_l565_565769


namespace sum_of_divisors_of_24_l565_565078

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565078


namespace distance_between_trains_l565_565006

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end distance_between_trains_l565_565006


namespace sum_of_divisors_of_24_l565_565133

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565133


namespace area_of_smallest_square_containing_circle_l565_565445

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ a : ℝ, a = 196 := 
by
  use (2 * r) ^ 2
  rw h
  norm_num

end area_of_smallest_square_containing_circle_l565_565445


namespace one_thirds_in_nine_thirds_l565_565661

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l565_565661


namespace children_taking_bus_l565_565394

theorem children_taking_bus (seats : ℕ) (children_per_seat : ℕ) (h_seats : seats = 29) (h_children_per_seat : children_per_seat = 2) :
  children_per_seat * seats = 58 :=
by {
  rw [h_seats, h_children_per_seat],
  exact rfl,
}

end children_taking_bus_l565_565394


namespace number_of_distinct_triangle_areas_l565_565758

noncomputable def distinct_triangle_area_counts : ℕ :=
sorry  -- Placeholder for the proof to derive the correct answer

theorem number_of_distinct_triangle_areas
  (G H I J K L : ℝ × ℝ)
  (h₁ : G.2 = H.2)
  (h₂ : G.2 = I.2)
  (h₃ : G.2 = J.2)
  (h₄ : H.2 = I.2)
  (h₅ : H.2 = J.2)
  (h₆ : I.2 = J.2)
  (h₇ : dist G H = 2)
  (h₈ : dist H I = 2)
  (h₉ : dist I J = 2)
  (h₁₀ : K.2 = L.2 - 2)  -- Assuming constant perpendicular distance between parallel lines
  (h₁₁ : dist K L = 2) : 
  distinct_triangle_area_counts = 3 :=
sorry  -- Placeholder for the proof

end number_of_distinct_triangle_areas_l565_565758


namespace sum_of_divisors_of_24_l565_565088

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565088


namespace rhombus_area_l565_565545

theorem rhombus_area (a : ℝ) (h : a = 10) :
  let d := a * real.sqrt 2 in
  let s := d / 2 in
  (1/2) * (s * s) = 25 :=
by
  sorry

end rhombus_area_l565_565545


namespace men_count_current_l565_565713

variable (M W : ℕ)
variable (initial_ratio : 4 * W = 5 * M)
variable (added_men : 2)
variable (left_women : 3)
variable (current_women : 24)
variable (women_doubled : 2 * (W - left_women) = current_women)

theorem men_count_current (initial_ratio women_doubled : Prop) : 
  M + added_men = 14 :=
by
  sorry


end men_count_current_l565_565713


namespace triangle_PQR_perimeter_l565_565329

noncomputable def perimeter_of_triangle_PQR (PQ QR PR : ℕ) : ℕ :=
  PQ + QR + PR

theorem triangle_PQR_perimeter :
  let PS : ℕ := 8
  let SR : ℕ := 20
  let QR : ℕ := 25
  (∀ (QS : ℕ), QS = Nat.sqrt (QR^2 - SR^2) → QS = 15 ) →
  (∀ (PQ : ℕ), PQ = Nat.sqrt (PS^2 + (∀ (QS : ℕ), QS = Nat.sqrt (QR^2 - SR^2)))
                       →  PQ = 17) →
  (∀ (PR : ℕ), PR = PS + SR → PR = 28) →
  perimeter_of_triangle_PQR 17 25 28 = 70 :=
by 
  intros PS SR QR hQS hPQ hPR
  simp [perimeter_of_triangle_PQR]; sorry

end triangle_PQR_perimeter_l565_565329


namespace sum_of_divisors_of_24_l565_565097

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565097


namespace length_of_platform_l565_565194

theorem length_of_platform (length_of_train speed_of_train time_to_cross : ℕ) 
    (h1 : length_of_train = 450) (h2 : speed_of_train = 126) (h3 : time_to_cross = 20) :
    ∃ length_of_platform : ℕ, length_of_platform = 250 := 
by 
  sorry

end length_of_platform_l565_565194


namespace sum_of_divisors_of_24_l565_565113

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565113


namespace Mel_weight_is_70_l565_565529

-- Definitions and conditions
def MelWeight (M : ℕ) :=
  3 * M + 10

theorem Mel_weight_is_70 (M : ℕ) (h1 : 3 * M + 10 = 220) :
  M = 70 :=
by
  sorry

end Mel_weight_is_70_l565_565529


namespace parabola_properties_l565_565415

theorem parabola_properties : 
  ∃ a b c : ℝ,  -- There exist coefficients of the parabola
  (∀ (x y : ℝ), y = a * x^2 + b * x + c ↔ (x, y) = (3, -9) ∨ (x, y) = (6, 27)) ∧ 
  -- Conditions: the parabola passes through points (3, -9) and (6, 27)
  (axis_of_symmetry : ℝ, zeros_difference : ℝ) 
  -- We need to find the axis of symmetry and the absolute difference between the zeros
  (axis_of_symmetry = 3) ∧ (zeros_difference = 3) :=
sorry

end parabola_properties_l565_565415


namespace arithmetic_geometric_inequality_l565_565161

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) := 
sorry

end arithmetic_geometric_inequality_l565_565161


namespace geometric_sequence_sum_l565_565710

theorem geometric_sequence_sum (k : ℕ) (h1 : a_1 = 1) (h2 : a_k = 243) (h3 : q = 3) : S_k = 364 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end geometric_sequence_sum_l565_565710


namespace mr_a_net_gain_l565_565752

def initial_worth : ℝ := 10000
def profit_percent : ℝ := 0.10
def loss_percent : ℝ := 0.10

theorem mr_a_net_gain :
  let sale_price := initial_worth * (1 + profit_percent),
      buy_back_price := sale_price * (1 - loss_percent),
      net_gain := sale_price - buy_back_price
  in net_gain = 1100 :=
by
  sorry

end mr_a_net_gain_l565_565752


namespace area_and_cost_of_path_l565_565825

-- Define the dimensions of the rectangular grass field
def length_field : ℝ := 75
def width_field : ℝ := 55

-- Define the width of the path around the field
def path_width : ℝ := 2.8

-- Define the cost per square meter for constructing the path
def cost_per_sq_m : ℝ := 2

-- Define the total length and width including the path
def total_length : ℝ := length_field + 2 * path_width
def total_width : ℝ := width_field + 2 * path_width

-- Define the area of the entire field including the path
def area_total : ℝ := total_length * total_width

-- Define the area of the grass field alone
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_total - area_field

-- Define the cost of constructing the path
def cost_path : ℝ := area_path * cost_per_sq_m

-- The statement to be proved
theorem area_and_cost_of_path :
  area_path = 759.36 ∧ cost_path = 1518.72 := by
  sorry

end area_and_cost_of_path_l565_565825


namespace sum_of_digits_of_10_100_minus_94_l565_565139

theorem sum_of_digits_of_10_100_minus_94 : 
  let n := 100
  in let number := 10 ^ n - 94
  in (number.digits.sum = 888) :=
sorry

end sum_of_digits_of_10_100_minus_94_l565_565139


namespace FK_eq_BC_l565_565728

variables {α : Type*} [EuclideanGeometry α]

structure IsRightAngledTriangle (A B C : α) : Prop :=
(angle_B : angle B = 90)

structure IsAngleBisector (A B C D : α) : Prop :=
(on_BC : D ∈ line B C)
(bisector : is_angle_bisector A D)

structure IsReflectionInLine (K E B C : α) : Prop :=
(reflection : is_reflection_in_line K E (line B C))

theorem FK_eq_BC {A B C D E F K : α} 
  (h_triangle : IsRightAngledTriangle A B C)
  (h_bisector : IsAngleBisector A B C D)
  (h_circum1 : ∃ circ1, is_circumcircle circ1 (triangle A C D) ∧ E ∈ circ1 ∧ E ∈ line A B ∧ E ≠ A)
  (h_circum2 : ∃ circ2, is_circumcircle circ2 (triangle A B D) ∧ F ∈ circ2 ∧ F ∈ line A C ∧ F ≠ A)
  (h_reflection : IsReflectionInLine K E B C) : 
  dist F K = dist B C :=
by
  sorry

end FK_eq_BC_l565_565728


namespace sequence_general_term_l565_565786

theorem sequence_general_term :
  ∀ (n : ℕ), (a_n : ℕ) (h : 0 < n) → (list_n : list ℕ) (hseq : list_n = [3, 5, 9, 17, 33]),
  a_n = 2^n + 1 :=
sorry

end sequence_general_term_l565_565786


namespace kitten_weight_l565_565170

theorem kitten_weight (x y z : ℕ): 
  x + y + z = 36 ∧ x + z = 2 * y ∧ x + y = z → x = 6 := 
by 
  intro h 
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end kitten_weight_l565_565170


namespace digits_difference_abs_l565_565401

theorem digits_difference_abs (x y : ℕ) (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : y = 2 * x) :
  |(x + y) - (10 * x + y)| = 36 :=
sorry

end digits_difference_abs_l565_565401


namespace sum_of_primes_with_no_integer_solution_l565_565561

theorem sum_of_primes_with_no_integer_solution :
  (∑ p in {p ∈ Finset.primes | ∀ x : ℤ, ¬ (5 * (3 * x + 2) ≡ 7 [MOD p])}, p) = 8 :=
by
  sorry

end sum_of_primes_with_no_integer_solution_l565_565561


namespace prism_width_calculation_l565_565858

theorem prism_width_calculation 
  (l h d : ℝ) 
  (h_l : l = 4) 
  (h_h : h = 10) 
  (h_d : d = 14) :
  ∃ w : ℝ, w = 4 * Real.sqrt 5 ∧ (l^2 + w^2 + h^2 = d^2) := 
by
  use 4 * Real.sqrt 5
  sorry

end prism_width_calculation_l565_565858


namespace ratio_of_arithmetic_seqs_l565_565348

noncomputable def arithmetic_seq_sum (a_1 a_n : ℕ) (n : ℕ) : ℝ := (n * (a_1 + a_n)) / 2

theorem ratio_of_arithmetic_seqs (a_1 a_6 a_11 b_1 b_6 b_11 : ℕ) :
  (∀ n : ℕ, (arithmetic_seq_sum a_1 a_n n) / (arithmetic_seq_sum b_1 b_n n) = n / (2 * n + 1))
  → (a_1 + a_6) / (b_1 + b_6) = 6 / 13
  → (a_1 + a_11) / (b_1 + b_11) = 11 / 23
  → (a_6 : ℝ) / (b_6 : ℝ) = 11 / 23 :=
  by
    intros h₁₁ h₆ h₁₁b
    sorry

end ratio_of_arithmetic_seqs_l565_565348


namespace line_tangent_circle_m_values_l565_565253

open Real

-- Definitions of the line and circle
def line (m : ℝ) (x y : ℝ) : Prop := x - sqrt(3) * y + m = 0
def circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 2 = 0

-- Center and radius of the circle
def center : ℝ × ℝ := (0, 1)
def radius : ℝ := sqrt(3)

-- Distance from the center to the line is equal to the radius for tangency
def tangent_condition (m : ℝ) : Prop :=
  abs(m - sqrt(3)) / sqrt(1 + 3) = sqrt(3)

-- Main theorem to be proved
theorem line_tangent_circle_m_values (m : ℝ) :
  (line m) ∧ (∀ x y, circle x y) ∧ tangent_condition m ↔ (m = -sqrt(3) ∨ m = 3*sqrt(3)) :=
by
  sorry

end line_tangent_circle_m_values_l565_565253


namespace sum_of_two_longest_altitudes_l565_565999

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) 
    (h3 : c = 15) (h4 : a^2 + b^2 = c^2) : 
    (min a b) +  (max a b) = 21 :=
by
  rw [h1, h2]
  exact congr_arg2 (· + ·) (min_self 9) (max_self 12)
  sorry

end sum_of_two_longest_altitudes_l565_565999


namespace gcd_entries_tends_to_infinity_l565_565919

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 2], ![4, 3]]

def I2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

def d (n : ℕ) : ℤ :=
  let M := A ^ n - I2 in
  Int.gcd (Int.gcd M[0, 0] M[0, 1]) (Int.gcd M[1, 0] M[1, 1])

theorem gcd_entries_tends_to_infinity : ∀ ε > 0, ∃ n ≥ 1, d(n) > ε :=
by
  sorry

end gcd_entries_tends_to_infinity_l565_565919


namespace pond_volume_l565_565227

def length : ℝ := 25
def min_width : ℝ := 10
def max_width : ℝ := 15
def min_depth : ℝ := 8
def max_depth : ℝ := 12

def avg_width (w1 w2 : ℝ) : ℝ := (w1 + w2) / 2
def avg_depth (d1 d2 : ℝ) : ℝ := (d1 + d2) / 2
def area_trapezoid (b1 b2 h : ℝ) : ℝ := (b1 + b2) / 2 * h
def volume_prism (area length : ℝ) : ℝ := area * length

theorem pond_volume : 
  volume_prism (area_trapezoid min_width max_width (avg_depth min_depth max_depth)) length = 3125 :=
by
  -- the detailed proof is omitted
  sorry

end pond_volume_l565_565227


namespace max_value_y_interval_l565_565651

noncomputable def y (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem max_value_y_interval : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → y x ≤ 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y x = 2) 
:=
by
  sorry

end max_value_y_interval_l565_565651


namespace sum_of_divisors_of_twenty_four_l565_565116

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565116


namespace prime_factors_count_900_l565_565997

theorem prime_factors_count_900 : 
  ∃ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x ∧ x ∣ 900) ∧ S.card = 3 :=
by 
  sorry

end prime_factors_count_900_l565_565997


namespace inequality_holds_l565_565963

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := Real.exp x + x - 2
def g (x : ℝ) : ℝ := Real.log x + x - 2

def root_a (a : ℝ) : Prop := f a = 0
def root_b (b : ℝ) : Prop := g b = 0

theorem inequality_holds (a b : ℝ) (H_a : root_a a) (H_b : root_b b) 
  (H_a_interval : 0 < a ∧ a < 1) (H_b_interval : 1 < b ∧ b < 2) 
  (H_f_increasing : ∀ (x y : ℝ), 0 < x → x < y → f x < f y) :
  f(a) < f(1) ∧ f(1) < f(b) :=
by 
  sorry

end inequality_holds_l565_565963


namespace total_notes_l565_565720

theorem total_notes :
  let red_notes := 5 * 6 in
  let blue_notes_under_red := 2 * red_notes in
  let total_blue_notes := blue_notes_under_red + 10 in
  red_notes + total_blue_notes = 100 := by
  sorry

end total_notes_l565_565720


namespace percentage_of_masters_l565_565190

-- Definition of given conditions
def average_points_juniors := 22
def average_points_masters := 47
def overall_average_points := 41

-- Problem statement
theorem percentage_of_masters (x y : ℕ) (hx : x ≥ 0) (hy : y ≥ 0) 
    (h_avg_juniors : 22 * x = average_points_juniors * x)
    (h_avg_masters : 47 * y = average_points_masters * y)
    (h_overall_average : 22 * x + 47 * y = overall_average_points * (x + y)) : 
    (y : ℚ) / (x + y) * 100 = 76 := 
sorry

end percentage_of_masters_l565_565190


namespace cone_height_radius_ratio_l565_565184

theorem cone_height_radius_ratio (r h : ℝ) (r_pos : r > 0) (h_pos : h > 0) 
  (rolls_without_slipping : 2 * π * sqrt (r^2 + h^2) = 40 * π * r) :
  h / r = sqrt 399 :=
by
  sorry

end cone_height_radius_ratio_l565_565184


namespace sum_of_divisors_of_24_l565_565076

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565076


namespace sum_of_divisors_of_twenty_four_l565_565121

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565121


namespace hyperbola_and_segment_length_l565_565284

noncomputable def equation_of_hyperbola (x y : ℝ) : Prop :=
  (2 * x^2 - y^2 = 6)

noncomputable def is_asymptote (x y : ℝ) : Prop :=
  (y = sqrt 2 * x) ∨ (y = -sqrt 2 * x)

theorem hyperbola_and_segment_length :
  (∃ (x y : ℝ), is_asymptote x y) →
  equation_of_hyperbola 3 (-2 * sqrt 3) →
  (|Segment_length 16 sqrt 3) :=
by
  sorry

end hyperbola_and_segment_length_l565_565284


namespace find_distance_between_PQ_l565_565008

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end find_distance_between_PQ_l565_565008


namespace sum_of_divisors_of_24_l565_565019

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565019


namespace angie_carlos_opposite_probability_zero_l565_565898

noncomputable def probability_angie_carlos_opposite : ℕ :=
  let people := ["Eva", "Angie", "Bridget", "Carlos", "Diego"] in
  let table_seats := 5 in
  if table_seats = 5 then 0 else sorry

theorem angie_carlos_opposite_probability_zero :
  probability_angie_carlos_opposite = 0 := by
  sorry

end angie_carlos_opposite_probability_zero_l565_565898


namespace find_principal_amount_l565_565180

-- Definitions of the given conditions
def principal_amount := P : ℝ
def interest_rate := 4 / 100 : ℝ
def time_period := 8 : ℝ
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) := (P * R * T) / 100
def interest_condition (P : ℝ) := simple_interest P interest_rate time_period = P - 340

-- The theorem we need to prove
theorem find_principal_amount (P : ℝ) (h : interest_condition P) : P = 500 :=
sorry

end find_principal_amount_l565_565180


namespace least_subtracted_number_l565_565452

def is_sum_of_digits_at_odd_places (n : ℕ) : ℕ :=
  (n / 100000) % 10 + (n / 1000) % 10 + (n / 10) % 10

def is_sum_of_digits_at_even_places (n : ℕ) : ℕ :=
  (n / 10000) % 10 + (n / 100) % 10 + (n % 10)

def diff_digits_odd_even (n : ℕ) : ℕ :=
  is_sum_of_digits_at_odd_places n - is_sum_of_digits_at_even_places n

theorem least_subtracted_number :
  ∃ x : ℕ, (427398 - x) % 11 = 0 ∧ x = 7 :=
by
  sorry

end least_subtracted_number_l565_565452


namespace train_length_l565_565827

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_m : ℝ) (conv_to_m_s : speed_kmh * (1000 / 3600) = speed_m_s) (dist_eq : speed_m_s * time_s = distance_m) : distance_m ≈ 200.04 :=
by
  -- Given conditions
  have speed_kmh := 60
  have time_s := 12
  have speed_m_s := 16.67
  have distance_m := 200.04

  -- Conversion validation
  have conv_to_m_s := by { calc 60 * (1000 / 3600) = speed_m_s : by norm_num }
  
  -- Distance calculation validation
  have dist_eq := by { calc 16.67 * 12 = distance_m : by norm_num }

  -- Conclude the approximation
  sorry

end train_length_l565_565827


namespace max_one_vertex_on_ellipse_l565_565624

def ellipse (F1 F2 : Type) (a : ℝ) := 
  {P : F1 × F2 | (dist P.1 F1 + dist P.2 F2) = 2 * a}

def square_centered_at (F1 : Type) := 
  {S : Set F1 | ∃ center, ∀ x ∈ S, dist x center = dist (x: F1) center}

theorem max_one_vertex_on_ellipse (F1 F2 : Type) (a : ℝ) (Γ : ellipse F1 F2 a) (S : square_centered_at F1) : 
  ∃ P ∈ S, P ∈ Γ → ¬ ∃ Q ∈ S, Q ∈ Γ ∧ P ≠ Q :=
sorry

end max_one_vertex_on_ellipse_l565_565624


namespace simplify_expression_l565_565878

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
  (x-2) ^ 2 - x * (x-1) + (x^3 - 4 * x^2) / x^2 = -2 * x := 
by 
  sorry

end simplify_expression_l565_565878


namespace find_common_difference_l565_565909

theorem find_common_difference 
  (a : ℕ → ℝ)
  (a1 : a 1 = 5)
  (a25 : a 25 = 173)
  (h : ∀ n : ℕ, a (n+1) = a 1 + n * (a 2 - a 1)) : 
  a 2 - a 1 = 7 :=
by 
  sorry

end find_common_difference_l565_565909


namespace find_a4_l565_565615

open Nat

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

-- The problem statement
theorem find_a4 (a_2 : ℤ) (S_5 : ℤ) (d : ℤ) :
  a_2 = 3 →
  S_5 = 25 →
  (∃ a d, ∀ n, arithmetic_sequence a d n) →
  (∃ a d, sum_arithmetic_sequence a d 5 = 25) →
  ∃ a, a + 3 * d = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end find_a4_l565_565615


namespace sum_of_divisors_of_24_l565_565089

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565089


namespace exhibit_animal_count_l565_565396

variables (RainForest Aquarium Aviary MammalHouse : ℕ)

-- Define conditions
def reptile_house := 16
def condition_1 := 5 + 3 * RainForest = reptile_house
def condition_2 := Aquarium = 2 * reptile_house
def condition_3 := Aviary = (Aquarium - RainForest) + 3
def condition_4 := MammalHouse = (RainForest + Aquarium + Aviary) / 3 + 2

-- Expected answers based on conditions
def RainForest_answer := 7
def Aquarium_answer := 32
def Aviary_answer := 28
def MammalHouse_answer := 24

-- Proof statement
theorem exhibit_animal_count :
  condition_1 → condition_2 → condition_3 → condition_4 →
  RainForest = RainForest_answer ∧
  Aquarium = Aquarium_answer ∧
  Aviary = Aviary_answer ∧
  MammalHouse = MammalHouse_answer :=
by sorry

end exhibit_animal_count_l565_565396


namespace sum_of_divisors_of_24_is_60_l565_565040

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565040


namespace area_black_square_l565_565860

theorem area_black_square (area_white : ℝ) (white_squares : ℕ) (total_squares : ℕ) (black_squares : ℕ)
  (h_white_squares : white_squares = 5) -- By the given conditions, there are 5 white squares
  (h_total_squares : total_squares = 9) -- By the given conditions, there are 9 smaller squares in total
  (h_black_squares : black_squares = 4) -- By the given conditions, there are 4 black squares
  (h_area_white : area_white = 180) : 
  let area_one_square := area_white / white_squares in
  let total_area := area_one_square * total_squares in
  let area_black := area_one_square * black_squares in
  area_black = 144 := 
sorry

end area_black_square_l565_565860


namespace sum_of_edges_corners_faces_of_rectangular_prism_l565_565721

-- Definitions based on conditions
def rectangular_prism_edges := 12
def rectangular_prism_corners := 8
def rectangular_prism_faces := 6
def resulting_sum := rectangular_prism_edges + rectangular_prism_corners + rectangular_prism_faces

-- Statement we want to prove
theorem sum_of_edges_corners_faces_of_rectangular_prism :
  resulting_sum = 26 := 
by 
  sorry -- Placeholder for the proof

end sum_of_edges_corners_faces_of_rectangular_prism_l565_565721


namespace sum_divisors_24_l565_565035

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565035


namespace gross_profit_percentage_l565_565930

theorem gross_profit_percentage :
  let SP := 28
  let WC := 23.93
  let GP := SP - WC
  let GP_percentage := (GP / WC) * 100
  abs (GP_percentage - 17.004) < 0.001 :=
begin
  sorry
end

end gross_profit_percentage_l565_565930


namespace find_x0_l565_565979

def f (x : ℝ) := x * abs x

theorem find_x0 (x0 : ℝ) (h : f x0 = 4) : x0 = 2 :=
by
  sorry

end find_x0_l565_565979


namespace maximum_length_closed_non_self_intersecting_line_l565_565449

theorem maximum_length_closed_non_self_intersecting_line (n m : ℕ) (grid : set (ℕ × ℕ)) :
  (n = 6) ∧ (m = 10) ∧ (grid = { (i, j) | i ≤ 6 ∧ j ≤ 10 }) →
  ∃ l, closed_non_self_intersecting_broken_line grid l ∧ length l = 76 :=
sorry

end maximum_length_closed_non_self_intersecting_line_l565_565449


namespace parametric_curve_equation_solution_l565_565166

theorem parametric_curve_equation_solution :
  (∃ a b c : ℚ, -- Since we are using rational numbers
    ∀ t : ℝ, 
    let x := 3 * (Real.cos (2 * t)),
        y := 6 * (Real.cos t) - 2 * (Real.sin t) in
      a * x^2 + b * x * y + c * y^2 = 1)
    ↔ (∃ a b c : ℚ, a = 2/9 ∧ b = 1/18 ∧ c = 1/36) :=
begin
  sorry
end

end parametric_curve_equation_solution_l565_565166


namespace projection_onto_plane_l565_565733
open Real

def normal_vector : ℝ × ℝ × ℝ := (2, -1, 2)

def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  λ i j, match (i, j) with
    | (0, 0) => 5/9
    | (0, 1) => 4/9
    | (0, 2) => -4/9
    | (1, 0) => 4/9
    | (1, 1) => 11/9
    | (1, 2) => 4/9
    | (2, 0) => -4/9
    | (2, 1) => 4/9
    | (2, 2) => 5/9
    | _ => 0

theorem projection_onto_plane (v : ℝ × ℝ × ℝ) :
  let P := projection_matrix in
  let proj := (P ⬝ (λ i, [v.1, v.2, v.3].nth i).getD 0) in
  proj = 
  let n := normal_vector in
  let dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1*b.1 + a.2*b.2 + a.3*b.3 in
  let n_dot_n := dot_product n n in
  let v_dot_n := dot_product v n in
  let scalar := v_dot_n / n_dot_n in
  let proj_vec := (scalar * n.1, scalar * n.2, scalar * n.3) in
  (v.1 - proj_vec.1, v.2 - proj_vec.2, v.3 - proj_vec.3) :=
sorry

end projection_onto_plane_l565_565733


namespace percent_problem_l565_565676

theorem percent_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 :=
sorry

end percent_problem_l565_565676


namespace relationship_a_b_c_l565_565738

open Real

theorem relationship_a_b_c (x : ℝ) (hx1 : e < x) (hx2 : x < e^2)
  (a : ℝ) (ha : a = log x)
  (b : ℝ) (hb : b = (1 / 2) ^ log x)
  (c : ℝ) (hc : c = exp (log x)) :
  c > a ∧ a > b :=
by {
  -- we state the theorem without providing the proof for now
  sorry
}

end relationship_a_b_c_l565_565738


namespace perimeter_of_shaded_shape_l565_565433

noncomputable def shaded_perimeter (x : ℝ) : ℝ := 
  let l := 18 - 2 * x
  3 * l

theorem perimeter_of_shaded_shape (x : ℝ) (hx : x > 0) (h_sectors : 2 * x + (18 - 2 * x) = 18) : 
  shaded_perimeter x = 54 := 
by
  rw [shaded_perimeter]
  rw [← h_sectors]
  simp
  sorry

end perimeter_of_shaded_shape_l565_565433


namespace sum_of_divisors_of_24_l565_565131

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565131


namespace part2_l565_565289

-- Define the conditions in Lean
variables (α β : ℝ)
variables (m : ℝ)
variables h1 : sin α = 2 * sqrt 2 / 3
variables h2 : ∃ (m : ℝ), (m, 2 * sqrt 2) ∈ {p : ℝ × ℝ | p.2 / sqrt (p.1^2 + 8) = sin α}
variables h3 : π/2 < α ∧ α < π
variables h4 : tan β = sqrt 2

-- Prove the first part: find the value of m
def find_m : Prop :=
  m = -1

-- Assuming the necessary conditions, prove that m = -1
lemma part1 : find_m α m := sorry

-- Prove the second part: evaluate the given trigonometric expression
def evaluate_expression : ℝ :=
  (sin α * cos β + 3 * sin (π / 2 + α) * sin β) / 
  (cos (π + α) * cos (-β) - 3 * sin α * sin β)

theorem part2 (h_m : m = -1) : evaluate_expression α β = sqrt 2 / 11 := sorry

end part2_l565_565289


namespace correct_mean_after_correction_l565_565150

theorem correct_mean_after_correction
  (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ)
  (h : n = 30) (h_mean : incorrect_mean = 150) (h_incorrect_value : incorrect_value = 135) (h_correct_value : correct_value = 165) :
  (incorrect_mean * n - incorrect_value + correct_value) / n = 151 :=
  by
  sorry

end correct_mean_after_correction_l565_565150


namespace find_p_l565_565631

theorem find_p (p : ℝ) : 
  (Nat.choose 5 3) * p^3 = 80 → p = 2 :=
by
  intro h
  sorry

end find_p_l565_565631


namespace hyperbola_standard_equation_l565_565425

def is_standard_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 / (a^2) - x^2 / (b^2) = 1)

theorem hyperbola_standard_equation :
  ∀ (a b c : ℝ),
  (a = 2) →
  (a + b = sqrt 2 * c) →
  (a^2 + b^2 = c^2) →
  is_standard_hyperbola 1 a b :=
by
  intros a b c ha hab hb
  rw ha at *
  sorry

end hyperbola_standard_equation_l565_565425


namespace total_floors_l565_565895

theorem total_floors (P Q R S T X F : ℕ) (h1 : 1 < X) (h2 : X < 50) :
  F = 1 + P - Q + R - S + T + X :=
sorry

end total_floors_l565_565895


namespace min_value_sin_cos_add_a_l565_565945

theorem min_value_sin_cos_add_a (a : ℝ) (ha : a > real.sqrt 2) :
  (∃ x : ℝ, y = (sin x + a) * (cos x + a)) → 
  (∀ x : ℝ, ((sin x + a) * (cos x + a)) ≥ (a - real.sqrt 2 / 2)^2) :=
by
  sorry

end min_value_sin_cos_add_a_l565_565945


namespace sum_of_divisors_of_24_is_60_l565_565037

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565037


namespace max_page_number_l565_565755

theorem max_page_number {n : ℕ} (has_thirty_four_twos : n = 34):
  ∃ (p : ℕ), p <= 199 ∧ (∀ m, m > 199 → (count_digit 2 (digits m)).sum > 34) :=
begin
  sorry
end

end max_page_number_l565_565755


namespace coordinates_of_C_are_correct_l565_565175

noncomputable section 

def Point := (ℝ × ℝ)

def A : Point := (1, 3)
def B : Point := (13, 9)

def vector_AB (A B : Point) : Point :=
  (B.1 - A.1, B.2 - A.2)

def scalar_mult (s : ℝ) (v : Point) : Point :=
  (s * v.1, s * v.2)

def add_vectors (v1 v2 : Point) : Point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def C : Point :=
  let AB := vector_AB A B
  add_vectors B (scalar_mult (1 / 2) AB)

theorem coordinates_of_C_are_correct : C = (19, 12) := by sorry

end coordinates_of_C_are_correct_l565_565175


namespace radius_circumscribed_circle_is_greater_than_ratio_l565_565806

-- Let R be the radius of the circumscribed circle
def circumscribed_circle_radius (A B C : Type) [Triangle A B C] : Type := sorry

-- Let r be the radius of the inscribed circle
def inscribed_circle_radius (A B C : Type) [Triangle A B C] : Type := sorry

-- Let a be the length of the longest side of the triangle
def longest_side (A B C : Type) [Triangle A B C] : Type := sorry

-- Let h be the length of the shortest altitude of the triangle
def shortest_altitude (A B C : Type) [Triangle A B C] : Type := sorry

theorem radius_circumscribed_circle_is_greater_than_ratio(
  {A B C : Type} [Triangle A B C] :
  circumscribed_circle_radius A B C > (longest_side A B C) / (shortest_altitude A B C)) :=
sorry

end radius_circumscribed_circle_is_greater_than_ratio_l565_565806


namespace min_value_l565_565611

-- Define a positive geometric sequence
def geo_sequence (a : ℕ → ℝ) :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Given conditions
variables {a : ℕ → ℝ} {m n : ℕ}
axiom a7_cond : a 7 = a 6 + 2 * a 5
axiom prod_cond : a m * a n = 16 * (a 1) * (a 1)

-- Theorem statement
theorem min_value (h1 : geo_sequence a) (h2 : a7_cond) (h3 : prod_cond) : 
  (1 / m : ℝ) + (9 / n : ℝ) = 11 / 4 :=
sorry

end min_value_l565_565611


namespace sum_of_divisors_of_24_l565_565048

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565048


namespace trajectory_equation_n_as_function_of_m_l565_565617

-- Define the point P and circle equation
def P : (ℝ × ℝ) := (0, 4)
def circle := {Q : ℝ × ℝ | Q.1 ^ 2 + Q.2 ^ 2 = 8}

-- Define the trajectory C equation
def trajectory_C (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 + (M.2 - 2) ^ 2 = 2

-- Define conditions and statement for part 1
theorem trajectory_equation :
  ∀ M : ℝ × ℝ,
  (∃ Q : ℝ × ℝ, Q ∈ circle ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) 
  → trajectory_C M :=
begin
  sorry
end

-- Define the relationship for n in terms of m
def n_function (m n : ℝ) : Prop :=
  n = sqrt (3 * m ^ 2 + 9) / 3

-- Define conditions and statement for part 2
theorem n_as_function_of_m (m n : ℝ) :
  -sqrt 6 / 2 < m ∧ m < sqrt 6 / 2 ∧ m ≠ 0 ∧ n > 0 
  → n_function m n :=
begin
  sorry
end

end trajectory_equation_n_as_function_of_m_l565_565617


namespace sum_of_divisors_of_24_l565_565099

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565099


namespace find_y_l565_565278

-- We declare variables
variables {k x y : ℝ}

-- The given conditions
def directly_proportional (k : ℝ) : Prop := ∀ x, y = k * (x + 1)
def value_at_x1 : Prop := y = 4 ∧ x = 1

-- The main proof statement
theorem find_y (h1 : directly_proportional k) (h2 : value_at_x1) : y = 6 :=
by {
  sorry
}

end find_y_l565_565278


namespace factorable_quadratic_l565_565575

theorem factorable_quadratic (b : Int) : 
  (∃ m n p q : Int, 35 * m * p = 35 ∧ m * q + n * p = b ∧ n * q = 35) ↔ (∃ k : Int, b = 2 * k) :=
sorry

end factorable_quadratic_l565_565575


namespace sum_of_divisors_of_24_l565_565025

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565025


namespace polygonal_line_length_l565_565703

open Set

-- Definition for a square with specified side length
def is_square (s : Set (ℝ × ℝ)) (l : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), s = {p | x1 ≤ p.1 ∧ p.1 ≤ x2 ∧ y1 ≤ p.2 ∧ p.2 ≤ y2} ∧ (x2 - x1 = l) ∧ (y2 - y1 = l)

-- Definition for the condition regarding distance to the polygonal line
def distance_condition (square : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ)) (d : ℝ) : Prop :=
  ∀ p ∈ square, ∃ q ∈ line, dist p q ≤ d

-- Statement of the problem as a theorem
theorem polygonal_line_length (square polygonal_line : Set (ℝ × ℝ)) :
  is_square square 50 →
  distance_condition square polygonal_line 1 →
  (∃ length : ℝ, length_of_polygonal_line polygonal_line > 1248) :=
by
  sorry

end polygonal_line_length_l565_565703


namespace least_possible_length_of_third_side_l565_565686

theorem least_possible_length_of_third_side (a b : ℕ) (hyp : a = 5 ∧ b = 12 ∨ a = 12 ∧ b = 5) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2 - min a b^2) ∧ c = Real.sqrt 119 :=
by 
  -- Given
  intro a b hyp
  -- Prove the existence of the third side and its length
  exists Real.sqrt 119
  -- Prove the lengths meet the condition
  sorry

end least_possible_length_of_third_side_l565_565686


namespace dot_product_l565_565753

variables (a b : ℝ^3)
variables (λ : ℝ)

-- Non-zero vectors a and b
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0

-- |b| = 2
axiom b_magnitude : Real.norm b = 2

-- angle(a, b) = 30 degrees
axiom angle_ab : Real.angle a b = Real.pi / 6

-- For all λ > 0, |a - λ*b| ≥ |a - b|
axiom inequality : ∀ λ > 0, Real.norm (a - λ • b) ≥ Real.norm (a - b)

theorem dot_product : a • b = 4 :=
sorry

end dot_product_l565_565753


namespace sum_first_8_geometric_l565_565332

theorem sum_first_8_geometric :
  let a₁ := 1 / 15
  let r := 2
  let S₄ := a₁ * (1 - r^4) / (1 - r)
  let S₈ := a₁ * (1 - r^8) / (1 - r)
  S₄ = 1 → S₈ = 17 := 
by
  intros a₁ r S₄ S₈ h
  sorry

end sum_first_8_geometric_l565_565332


namespace exists_common_element_l565_565944

theorem exists_common_element
  (A : Fin 1978 → Set α)
  (h_size : ∀ i, |A i| = 40)
  (h_intersect : ∀ i j, i ≠ j → |A i ∩ A j| = 1) :
  ∃ a, ∀ i, a ∈ A i :=
sorry

end exists_common_element_l565_565944


namespace max_car_passing_400_l565_565377

noncomputable def max_cars_passing (speed : ℕ) (car_length : ℤ) (hour : ℕ) : ℕ :=
  20000 * speed / (5 * (speed + 1))

theorem max_car_passing_400 :
  max_cars_passing 20 5 1 / 10 = 400 := by
  sorry

end max_car_passing_400_l565_565377


namespace general_formula_find_k_l565_565952

noncomputable def arithmetic_seq (n : ℕ) : ℤ := 3 - 2 * n

def sum_first_n_terms (n : ℕ) : ℤ := n * (arithmetic_seq 1 + arithmetic_seq n) / 2

theorem general_formula (n : ℕ) : arithmetic_seq n = 3 - 2 * n := rfl

theorem find_k (k : ℕ) (h₁ : sum_first_n_terms k = -35) : k = 7 := sorry

end general_formula_find_k_l565_565952


namespace proof_problem_l565_565632

variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be increasing on (-∞, 0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y < 0 → f x < f y

-- Define what it means for a function to be decreasing on (0, +∞)
def is_decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f y < f x

theorem proof_problem 
  (h_even : is_even_function f) 
  (h_inc_neg : is_increasing_on_neg f) : 
  (∀ x : ℝ, f (-x) - f x = 0) ∧ (is_decreasing_on_pos f) :=
by
  sorry

end proof_problem_l565_565632


namespace initial_marbles_l565_565903

theorem initial_marbles (M : ℕ) (h1 : M + 9 = 104) : M = 95 := by
  sorry

end initial_marbles_l565_565903


namespace paint_coverage_is_10_l565_565312

noncomputable def paintCoverage (cost_per_quart : ℝ) (cube_edge_length : ℝ) (total_cost : ℝ) : ℝ :=
  let total_surface_area := 6 * (cube_edge_length ^ 2)
  let number_of_quarts := total_cost / cost_per_quart
  total_surface_area / number_of_quarts

theorem paint_coverage_is_10 :
  paintCoverage 3.2 10 192 = 10 :=
by
  sorry

end paint_coverage_is_10_l565_565312


namespace tan_expression_one_sin_expression_two_l565_565219

theorem tan_expression_one : 
  (tan 53 * Real.pi / 180 + tan 7 * Real.pi / 180 + tan 120 * Real.pi / 180) / (tan 53 * Real.pi / 180 * tan 7 * Real.pi / 180) = -Real.sqrt 3 :=
sorry

theorem sin_expression_two : 
  (2 * sin 50 * Real.pi / 180 + sin 10 * Real.pi / 180 * (1 + Real.sqrt 3 * tan 10 * Real.pi / 180)) * Real.sqrt (1 - cos 160 * Real.pi / 180) = Real.sqrt 6 :=
sorry

end tan_expression_one_sin_expression_two_l565_565219


namespace meet_point_l565_565371

def one_third_point (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((1 - 1/3) * x1 + 1/3 * x2, (1 - 1/3) * y1 + 1/3 * y2)

theorem meet_point :
  one_third_point 2 3 8 (-5) = (4, 1/3) :=
by
  sorry

end meet_point_l565_565371


namespace range_of_a_l565_565937

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 3*x + 2*a else x - a*log x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) → 1 ≤ a ∧ a ≤ exp 1 := 
begin
  sorry
end

end range_of_a_l565_565937


namespace fraction_spent_on_food_l565_565177

variable (salary : ℝ) (food_fraction rent_fraction clothes_fraction remaining_amount : ℝ)
variable (salary_condition : salary = 180000)
variable (rent_fraction_condition : rent_fraction = 1/10)
variable (clothes_fraction_condition : clothes_fraction = 3/5)
variable (remaining_amount_condition : remaining_amount = 18000)

theorem fraction_spent_on_food :
  rent_fraction * salary + clothes_fraction * salary + food_fraction * salary + remaining_amount = salary →
  food_fraction = 1/5 :=
by
  intros
  sorry

end fraction_spent_on_food_l565_565177


namespace kitten_weighs_9_l565_565172

theorem kitten_weighs_9 (x y z : ℕ) 
  (h1 : x + y + z = 36)
  (h2 : x + z = 2y)
  (h3 : x + y = z) : x = 9 :=
by
  sorry

end kitten_weighs_9_l565_565172


namespace sum_of_divisors_of_24_l565_565057

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565057


namespace Euler_line_intersects_sides_l565_565457

-- Define a structure for a triangle and its properties
structure Triangle :=
  (A B C : ℝ×ℝ)  -- Vertices of the triangle
  (is_acute_angled : Prop)  -- Whether the triangle is acute-angled
  (is_obtuse_angled : Prop) -- Whether the triangle is obtuse-angled
  (orthocenter : ℝ×ℝ)  -- Orthocenter H
  (centroid : ℝ×ℝ)    -- Centroid G
  (circumcenter : ℝ×ℝ) -- Circumcenter O

-- Define the Euler line, which is a line passing through the orthocenter, centroid, and circumcenter
def Euler_line (T : Triangle) : Prop := 
  collinear {T.orthocenter, T.centroid, T.circumcenter}

-- The theorem to be proven
theorem Euler_line_intersects_sides (T : Triangle) : Prop :=
  (T.is_acute_angled → intersects_sides (Euler_line T) T ["largest", "smallest"]) ∧
  (T.is_obtuse_angled → intersects_sides (Euler_line T) T ["largest", "median"])

end Euler_line_intersects_sides_l565_565457


namespace coat_price_reduction_l565_565151

variable (original_price reduction : ℝ)

theorem coat_price_reduction
  (h_orig : original_price = 500)
  (h_reduct : reduction = 350)
  : reduction / original_price * 100 = 70 := 
sorry

end coat_price_reduction_l565_565151


namespace modulus_of_expression_l565_565899

noncomputable def omega : ℂ := 7 + 3 * complex.i

theorem modulus_of_expression : abs (omega^2 + 4 * omega + 40) = 54 * real.sqrt 5 := 
by sorry

end modulus_of_expression_l565_565899


namespace find_all_x_satisfying_condition_l565_565908

theorem find_all_x_satisfying_condition :
  ∃ (x : Fin 2016 → ℝ), 
  (∀ i : Fin 2016, x (i + 1) % 2016 = x 0) ∧
  (∀ i : Fin 2016, x i ^ 2 + x i - 1 = x ((i + 1) % 2016)) ∧
  (∀ i : Fin 2016, x i = 1 ∨ x i = -1) :=
sorry

end find_all_x_satisfying_condition_l565_565908


namespace N_even_for_all_permutations_l565_565277

noncomputable def N (a b : Fin 2013 → ℕ) : ℕ :=
  Finset.prod (Finset.univ : Finset (Fin 2013)) (λ i => a i - b i)

theorem N_even_for_all_permutations {a : Fin 2013 → ℕ}
  (h_distinct : Function.Injective a) :
  ∀ b : Fin 2013 → ℕ,
  (∀ i, b i ∈ Finset.univ.image a) →
  ∃ n, n = N a b ∧ Even n :=
by
  -- This is where the proof would go, using the given conditions.
  sorry

end N_even_for_all_permutations_l565_565277


namespace sum_of_divisors_of_24_l565_565110

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565110


namespace shopkeeper_intended_profit_l565_565185

noncomputable def intended_profit_percentage (C L S : ℝ) : ℝ :=
  (L / C) - 1

theorem shopkeeper_intended_profit (C L S : ℝ) (h1 : L = C * (1 + intended_profit_percentage C L S))
  (h2 : S = 0.90 * L) (h3 : S = 1.35 * C) : intended_profit_percentage C L S = 0.5 :=
by
  -- We indicate that the proof is skipped
  sorry

end shopkeeper_intended_profit_l565_565185


namespace sin2theta_eq_neg_half_tan_theta_eq_vals_l565_565992

-- Define vectors m and n
variable (a θ : ℝ)

def m : ℝ × ℝ := (a - sin θ, -1 / 2)
def n : ℝ × ℝ := (1 / 2, cos θ)

-- Define perpendicular condition for part (1)
def perpendicular : Prop := (a - sin θ) * (1 / 2) + (-1 / 2) * (cos θ) = 0

-- Define parallel condition for part (2)
def parallel : Prop := ∃ k : ℝ, (a - sin θ) = k * (1 / 2) ∧ (-1 / 2) = k * cos θ

-- Part (1) - question 1: sin 2θ = -1/2
theorem sin2theta_eq_neg_half (h1 : a = sqrt 2 / 2) (h2 : perpendicular a θ) : 
  sin (2 * θ) = -1 / 2 := by
  sorry

-- Part (2) - question 2: tan θ = 2 + sqrt 3 or 2 - sqrt 3
theorem tan_theta_eq_vals (h1 : a = 0) (h2 : parallel 0 θ) : 
  tan θ = 2 + sqrt 3 ∨ tan θ = 2 - sqrt 3 := by
  sorry

end sin2theta_eq_neg_half_tan_theta_eq_vals_l565_565992


namespace books_per_shelf_l565_565429

theorem books_per_shelf (total_books shelves : ℕ) (h1 : total_books = 113920) (h2 : shelves = 14240) : 
  total_books / shelves = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end books_per_shelf_l565_565429


namespace smallest_d_l565_565497

theorem smallest_d (d : ℝ) : 
  (∃ d, 2 * d = Real.sqrt ((4 * Real.sqrt 3) ^ 2 + (d + 4) ^ 2)) →
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
by
  sorry

end smallest_d_l565_565497


namespace x_plus_y_value_l565_565412

-- Define the given numbers
def nums : List ℝ := [6, 14, x, 17, 9, y, 10]

-- Define the mean of the list of numbers
def mean (lst : List ℝ) : ℝ := (lst.sum) / lst.length

-- Given condition
axiom mean_given : mean nums = 13

-- Statement to prove
theorem x_plus_y_value : x + y = 35 :=
by
  -- Proof will go here
  sorry

end x_plus_y_value_l565_565412


namespace how_much_money_bob_started_with_l565_565875

theorem how_much_money_bob_started_with (X : ℝ) 
    (monday_left : X / 2) 
    (tuesday_left : (X / 2) * (4/5)) 
    (wednesday_left : (X / 2) * (4/5) * (5/8)) 
    (remaining_money : (X / 2) * (4/5) * (5/8) = 20) :
  X = 80 := 
sorry

end how_much_money_bob_started_with_l565_565875


namespace sum_of_divisors_of_24_l565_565087

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565087


namespace part_I_part_II_l565_565879

-- Lean statement for part I
theorem part_I (a b : ℝ) : (a^2 * b^3)^(-1) * (a * b^(-2))^2 = 1 / b^7 := 
by
  -- Proof goes here
  sorry

-- Lean statement for part II
theorem part_II (x : ℝ) : x^2 * x^4 - (2 * x^3)^2 + x^7 / x = -2 * x^6 :=
by
  -- Proof goes here
  sorry

end part_I_part_II_l565_565879


namespace high_quality_frequency_1000_high_quality_frequency_5000_estimate_high_quality_probability_l565_565506

noncomputable def freq_1000 (m : ℕ) (n : ℕ) : ℚ :=
  m / n

noncomputable def freq_5000 (m : ℕ) (n : ℕ) : ℚ :=
  m / n

theorem high_quality_frequency_1000 
  (n m : ℕ) 
  (hn : n = 1000) 
  (hm : m = 951) :
  freq_1000 m n = 0.951 := 
by
  sorry

theorem high_quality_frequency_5000 
  (n m : ℕ) 
  (hn : n = 5000) 
  (hm : m = 4750) :
  freq_5000 m n = 0.95 := 
by
  sorry

theorem estimate_high_quality_probability 
  (f : ℕ → ℚ) 
  (samples : List ℚ)
  (h_samples: samples = [0.9, 0.96, 0.951, 0.95, 0.952, 0.95])
  (estimate : ℚ) 
  (h_estimate: estimate = (samples.sum / samples.length)) :
  estimate ≈ 0.95 := 
by
  sorry

end high_quality_frequency_1000_high_quality_frequency_5000_estimate_high_quality_probability_l565_565506


namespace minimum_value_x_l565_565983

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end minimum_value_x_l565_565983


namespace exists_ints_xy_div_p_l565_565268

theorem exists_ints_xy_div_p
  (a b c : ℤ)
  (p : ℤ) (hp : p.prime) (hpodd : odd p) :
  ∃ x y : ℤ, p ∣ (x ^ 2 + y ^ 2 + a * x + b * y + c) :=
by
  sorry

end exists_ints_xy_div_p_l565_565268


namespace A_share_in_profit_l565_565863

-- Given conditions:
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12600

-- The statement we need to prove:
theorem A_share_in_profit :
  (3 / 10) * total_profit = 3780 := by
  sorry

end A_share_in_profit_l565_565863


namespace average_speed_of_journey_l565_565486

theorem average_speed_of_journey :
  let distance1 := 10 -- km 
  let speed1 := 4 -- km/h 
  let distance2 := 10 -- km 
  let speed2 := 6 -- km/h 
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 24 / 5 :=
by
  let distance1 := 10
  let speed1 := 4
  let distance2 := 10
  let speed2 := 6
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  show average_speed = 24 / 5 from
    sorry

end average_speed_of_journey_l565_565486


namespace sum_of_divisors_of_24_l565_565105

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565105


namespace ant_prob_7min_l565_565871

open ProbabilityTheory

noncomputable theory

def ant_prob_path (A B C : ℕ × ℕ) (time : ℕ) : probability_space :=
{
  prob_path : A → C → B → time,
  A = (0, 0),
  B = (0, 1),
  C = (1, 0),
  time = 7,
}

theorem ant_prob_7min (A B C : ℕ × ℕ) (time : ℕ) : 
  (ant_prob_path A B C time).prob_path = 1 / 4 :=
by {
  sorry
}

end ant_prob_7min_l565_565871


namespace smallest_multiple_of_25_with_product_multiple_of_100_l565_565448

def is_multiple_of_25 (n : ℕ) : Prop :=
  n % 25 = 0

def digits_product (n : ℕ) : ℕ :=
  n.digits.foldr (λ d acc => d * acc) 1

def is_multiple_of_100 (n : ℕ) : Prop :=
  n % 100 = 0

theorem smallest_multiple_of_25_with_product_multiple_of_100 : 
  ∃ n : ℕ, is_multiple_of_25 n ∧ is_multiple_of_100 (digits_product n) ∧ n = 525 := 
by
  sorry

end smallest_multiple_of_25_with_product_multiple_of_100_l565_565448


namespace solve_for_x_l565_565564

-- Define the function h(x)
def h (x : ℝ) : ℝ := (2 * x + 5) ^ (1 / 3) / 5 ^ (1 / 3)

-- Define the equation we need to solve
def equation (x : ℝ) : Prop := h (3 * x) = 3 * h x

-- The ultimate goal is to find x such that equation x holds true
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -65 / 24 :=
by
  sorry

end solve_for_x_l565_565564


namespace distance_between_trains_l565_565005

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end distance_between_trains_l565_565005


namespace part_I_part_II_l565_565368

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ := n / 2^(n + 1)

-- Define the sum of the first n terms of the sequence S_n
def S_n (n : ℕ) : ℝ := -a_n n + 1 - 1 / 2^n

-- Define the sum of the first n terms of the sequence T_n
def T_n (n : ℕ) : ℝ := n - 2 + (n + 4) / 2^(n + 1)

-- Prove that a_n = n / 2^(n + 1)
theorem part_I (n : ℕ) (hnpos : 0 < n) : 
  a_n n = n / 2^(n + 1) :=
by sorry

-- Prove that T_n = n - 2 + (n + 4) / 2^(n + 1)
theorem part_II (n : ℕ) : 
  T_n n = n - 2 + (n + 4) / 2^(n + 1) :=
by sorry

end part_I_part_II_l565_565368


namespace doubled_sum_of_squares_l565_565763

theorem doubled_sum_of_squares (a b : ℝ) : 
  2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := 
by
  sorry

end doubled_sum_of_squares_l565_565763


namespace composite_has_at_least_three_factors_l565_565165

-- Definition of composite number in terms of its factors
def is_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Theorem stating that a composite number has at least 3 factors
theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : 
  (∃ f1 f2 f3, f1 ∣ n ∧ f2 ∣ n ∧ f3 ∣ n ∧ f1 ≠ 1 ∧ f1 ≠ n ∧ f2 ≠ 1 ∧ f2 ≠ n ∧ f3 ≠ 1 ∧ f3 ≠ n ∧ f1 ≠ f2 ∧ f2 ≠ f3) := 
sorry

end composite_has_at_least_three_factors_l565_565165


namespace shape_symmetry_x_axis_l565_565805

theorem shape_symmetry_x_axis (shape : set (ℝ × ℝ)) :
  (∀ p ∈ shape, ∃ q ∈ shape, q.1 = p.1 ∧ q.2 = -p.2) →
  ∀ p ∈ shape, ∃ q ∈ shape, p.1 = q.1 ∧ p.2 = -q.2 := 
by 
  intros h p hp
  specialize h p hp
  rcases h with ⟨q, hq, hq_props⟩
  use [q, hq]
  exact ⟨hq_props.1, eq.symm hq_props.2⟩

end shape_symmetry_x_axis_l565_565805


namespace sum_of_divisors_of_24_l565_565084

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565084


namespace sum_of_xyz_l565_565967

theorem sum_of_xyz (x y z : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : z > 0)
  (h4 : x^2 + y^2 + x * y = 3)
  (h5 : y^2 + z^2 + y * z = 4)
  (h6 : z^2 + x^2 + z * x = 7) :
  x + y + z = Real.sqrt 13 :=
by sorry -- Proof omitted, but the statement formulation is complete and checks the equality under given conditions.

end sum_of_xyz_l565_565967


namespace perpendicular_parallel_implies_perpendicular_l565_565604

open_locale classical

variables (Line Plane : Type) [has_perpendicular Line Plane] [has_parallel Line Plane]

axiom lines_different : ∀ (l m n : Line), l ≠ m → m ≠ n → l ≠ n
axiom planes_different : ∀ (α β γ : Plane), α ≠ β → β ≠ γ → α ≠ γ

variables (l m n : Line) (α β γ : Plane)
variables [different_lines : lines_different l m n] [different_planes : planes_different α β γ]

theorem perpendicular_parallel_implies_perpendicular 
  (h1 : m ⊥ α) (h2 : m ∥ β) : α ⊥ β :=
by
  sorry

end perpendicular_parallel_implies_perpendicular_l565_565604


namespace remainder_x_squared_mod_20_l565_565672

theorem remainder_x_squared_mod_20
  (x : ℤ)
  (h1 : 4 * x ≡ 8 [MOD 20])
  (h2 : 3 * x ≡ 16 [MOD 20]) :
  x^2 ≡ 4 [MOD 20] :=
by
  sorry

end remainder_x_squared_mod_20_l565_565672


namespace find_b_l565_565772

theorem find_b (b : ℤ) (h1 : 0 ≤ b ∧ b ≤ 20) (h2 : (142536472 : ℕ) - b ≡ 0 [MOD 13]) : b = 8 :=
by sorry

end find_b_l565_565772


namespace max_ab_f_l565_565889

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x + b

theorem max_ab_f :
  ∀ (a b : ℝ), (b > 0) →
  (∀ x : ℝ, x ∈ Icc (-1 / 2) (1 / 2) → abs (f a b x) ≤ 2) →
  (a * b ≤ 1) →
  f 1 1 2017 = 4035 :=
by
  intros a b b_pos H1 H2
  sorry

end max_ab_f_l565_565889


namespace max_possible_distance_l565_565736

noncomputable def max_distance (w : ℂ) (hw : |w| = 3) : ℝ :=
  81 * Complex.abs (1 + 2 * Complex.I - w^2)

theorem max_possible_distance (w : ℂ) (hw : |w| = 3) :
  max_distance w hw = 648 * Real.sqrt 5 := by
  sorry

end max_possible_distance_l565_565736


namespace compute_expression_l565_565540

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end compute_expression_l565_565540


namespace percentage_of_masters_l565_565191

theorem percentage_of_masters (x y : ℕ) (avg_juniors avg_masters avg_team : ℚ) 
  (h1 : avg_juniors = 22)
  (h2 : avg_masters = 47)
  (h3 : avg_team = 41)
  (h4 : 22 * x + 47 * y = 41 * (x + y)) : 
  76% of the team are masters := by sorry

end percentage_of_masters_l565_565191


namespace sum_of_divisors_of_24_l565_565050

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565050


namespace equation_of_AB_equation_of_altitude_from_AB_through_C_l565_565656

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

def equation_line (x1 y1 x2 y2 : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (y - y1) = (slope x1 y1 x2 y2) * (x - x1)

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem equation_of_AB :
  ∀ (x y : ℝ),
  equation_line (-5) 0 3 (-3) x y ↔ (8 * y + 3 * x + 15 = 0) :=
by
  sorry

theorem equation_of_altitude_from_AB_through_C :
  ∀ (x y : ℝ),
  let m_altitude := 8 / 3
  let perpendicular_line_eq := λ x y => (y - 2) = m_altitude * (x - 0)
  perpendicular_line_eq x y ↔ (8 * x - 3 * y + 6 = 0) :=
by
  sorry

end equation_of_AB_equation_of_altitude_from_AB_through_C_l565_565656


namespace sum_S4_eq_28_l565_565330

noncomputable def geometric_sequence (a r : ℝ) : (ℕ → ℝ) 
| 0       := a
| (n + 1) := (geometric_sequence a r n) * r

def partial_sum (a r : ℝ) (n : ℕ) : ℝ :=
(n + 1) * a * (1 - r^(n + 1)) / (1 - r)

variables {a r : ℝ} (h : 0 < r ∧ r ≠ 1)

theorem sum_S4_eq_28 (S2 S6 : ℝ) (hS2 : partial_sum a r 1 = S2) (hS2_7 : S2 = 7) 
  (hS6 : partial_sum a r 5 = S6) (hS6_91 : S6 = 91) : partial_sum a r 3 = 28 := 
by
  sorry

end sum_S4_eq_28_l565_565330


namespace trig_identity_l565_565417

theorem trig_identity : (cos (π / 12) - sin (π / 12)) * (cos (π / 12) + sin (π / 12)) = √3 / 2 := 
by 
  sorry

end trig_identity_l565_565417


namespace cube_sum_expansion_sum_coefficients_l565_565404

theorem cube_sum_expansion (y : ℝ) : 
  (512 * y ^ 3 + 27) = ((8 * y + 3) * (64 * y ^ 2 - 24 * y + 9)) :=
begin
  sorry
end

theorem sum_coefficients :
  (8 + 3 + 64 - 24 + 9) = 60 :=
begin
  calc
    8 + 3 + 64 - 24 + 9 = 60 : by rfl
end

end cube_sum_expansion_sum_coefficients_l565_565404


namespace product_PQRS_l565_565625

-- Definitions of the variables involved
def P := (Real.sqrt 2025 + Real.sqrt 2024)
def Q := (-Real.sqrt 2025 - Real.sqrt 2024)
def R := (Real.sqrt 2025 - Real.sqrt 2024)
def S := (Real.sqrt 2024 - Real.sqrt 2025)

-- The theorem that encapsulates the problem
theorem product_PQRS : P * Q * R * S = -1 := 
by 
  sorry

end product_PQRS_l565_565625


namespace sum_of_real_solutions_l565_565249

theorem sum_of_real_solutions :
  let eq := ∀ (x : ℝ), (x-3)/(x^2 + 5*x + 2) = (x-6)/(x^2 - 11*x)
  ∑ (r : ℝ) in { x : ℝ | eq x }, r = 62/13 :=
by { sorry }

end sum_of_real_solutions_l565_565249


namespace find_smallest_modulus_l565_565361

noncomputable def smallest_modulus (z : ℂ) (h : |z - 15| + |z - 7 * complex.I| = 17) : ℚ :=
  105 / 17

theorem find_smallest_modulus (z : ℂ) (h : |z - 15| + |z - 7 * complex.I| = 17) :
  |z| = 105 / 17 :=
sorry

end find_smallest_modulus_l565_565361


namespace matrix_non_invertible_value_y_l565_565250

/-- Define the matrix in question -/
def my_matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 + y, 6], ![4 - y, 9]]

/-- State the determinant condition for non-invertibility -/
def determinant_zero_condition (y : ℝ) : Prop :=
  my_matrix y.det = 0

/-- The resulting value of y for which the matrix is non-invertible -/
theorem matrix_non_invertible_value_y : determinant_zero_condition (2 / 5) :=
by
  sorry

end matrix_non_invertible_value_y_l565_565250


namespace least_value_expr_l565_565012

   variable {x y : ℝ}

   theorem least_value_expr : ∃ x y : ℝ, (x^3 * y - 1)^2 + (x + y)^2 = 1 :=
   by
     sorry
   
end least_value_expr_l565_565012


namespace value_of_expression_l565_565743

theorem value_of_expression : 
  let x := -730 in 
  abs(abs(x)^2 - x - abs(x)) + x = 533630 :=
by
  sorry

end value_of_expression_l565_565743


namespace growing_path_maximum_product_l565_565780

theorem growing_path_maximum_product : 
  let points := { (x, y) : ℕ × ℕ | x ≤ 4 ∧ y ≤ 4 }
  let distance (p1 p2 : ℕ × ℕ) : ℝ := 
    real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
  let is_growing_path (path : list (ℕ × ℕ)) : Prop :=
    path.distinct ∧ 
    (∀ (i : ℕ), i < path.length - 1 → distance (path.nth_le i _) (path.nth_le (i + 1) _) < distance (path.nth_le (i + 1) _) (path.nth_le (i + 2) _))
  let m := 16
  let r := 32
  m * r = 512 :=
sorry

end growing_path_maximum_product_l565_565780


namespace det_transform_l565_565598

theorem det_transform 
  (x y z w : ℝ) 
  (h : det (matrix.of ![![x, y], ![z, w]]) = -3) : 
  det (matrix.of ![![x + 2 * z, y + 2 * w], ![z, w]]) = -3 := by
sorry

end det_transform_l565_565598


namespace cubic_polynomial_roots_l565_565382

theorem cubic_polynomial_roots (a : ℚ) :
  (x^3 - 6*x^2 + a*x - 6 = 0) ∧ (x = 3) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  sorry

end cubic_polynomial_roots_l565_565382


namespace sum_sequence_eq_16501_l565_565137

noncomputable def sequence_sum : ℕ := 
  let n := 11001 in
  let pairs := (n - 1) / 2 in
  let last_term := n in
  pairs * 1 + last_term

theorem sum_sequence_eq_16501 : sequence_sum = 16501 := 
  by sorry

end sum_sequence_eq_16501_l565_565137


namespace domain_of_f_l565_565912

noncomputable def f (x : ℝ) : ℝ := sqrt ((log x - 2) * (x - log x - 1))

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ (∃ (y : ℝ), f y ≠ 0) ∧ f x ≠ 0} = {x : ℝ | x = 1} ∪ {x : ℝ | e^2 ≤ x} :=
by
  sorry

end domain_of_f_l565_565912


namespace sum_of_divisors_of_24_is_60_l565_565059

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565059


namespace minimum_value_of_expression_l565_565581

open Nat

def lcm (a b : ℕ) : ℕ := (a * b) / (gcd a b)

def f (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 2 - (lcm a b + lcm b c + lcm c a : ℚ) / (a + b + c : ℚ)

theorem minimum_value_of_expression :
  ∀ (a b c : ℕ), 1 < a → 1 < b → 1 < c → f a b c = 3 / 2 → False :=
begin
  assume a b c ha hb hc,
  sorry
end

end minimum_value_of_expression_l565_565581


namespace parker_daily_earning_l565_565384

-- Definition of conditions
def total_earned : ℕ := 2646
def weeks_worked : ℕ := 6
def days_per_week : ℕ := 7
def total_days (weeks : ℕ) (days_in_week : ℕ) : ℕ := weeks * days_in_week

-- Proof statement
theorem parker_daily_earning (h : total_days weeks_worked days_per_week = 42) : (total_earned / 42) = 63 :=
by
  sorry

end parker_daily_earning_l565_565384


namespace berry_average_temperature_l565_565528

theorem berry_average_temperature :
  let t_S := [37.3, 37.2, 36.9] in
  let t_M := [36.6, 36.9, 37.1] in
  let t_T := [37.1, 37.3, 37.2] in
  let t_W := [36.8, 37.3, 37.5] in
  let t_Th := [37.1, 37.7, 37.3] in
  let t_F := [37.5, 37.4, 36.9] in
  let t_Sa := [36.9, 37.0, 37.1] in
  let temperatures := t_S ++ t_M ++ t_T ++ t_W ++ t_Th ++ t_F ++ t_Sa in
  (∑ i in temperatures, i) / 21 = 37.62 := 
sorry

end berry_average_temperature_l565_565528


namespace equilateral_triangle_area_outside_l565_565522

theorem equilateral_triangle_area_outside (R : ℝ) :
  ∃ A B C : Point, is_equilateral_triangle A B C ∧ is_inscribed_in_circle A B C R ∧
  ∀ P Q R, (P = extend_altitude_to_circle A R) ∧ (Q = extend_altitude_to_circle B R) ∧ (R = extend_altitude_to_circle C R) →
  area_of_circle R - area_of_triangle P Q R = R^2 * (Real.pi - Real.sqrt 3) :=
sorry

end equilateral_triangle_area_outside_l565_565522


namespace sum_of_divisors_of_twenty_four_l565_565120

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565120


namespace tan_alpha_value_l565_565961

theorem tan_alpha_value (α : ℝ) (h1 : sin α - cos α = sqrt 2) (h2 : 0 < α ∧ α < π) : tan α = -1 := 
sorry

end tan_alpha_value_l565_565961


namespace possible_AC_values_l565_565990

-- Given points A, B, and C on a straight line 
-- with AB = 1 and BC = 3, prove that AC can be 2 or 4.

theorem possible_AC_values (A B C : ℝ) (hAB : abs (B - A) = 1) (hBC : abs (C - B) = 3) : 
  abs (C - A) = 2 ∨ abs (C - A) = 4 :=
sorry

end possible_AC_values_l565_565990


namespace madhav_rank_from_last_is_15_l565_565828

-- Defining the conditions
def class_size : ℕ := 31
def madhav_rank_from_start : ℕ := 17

-- Statement to be proved
theorem madhav_rank_from_last_is_15 :
  (class_size - madhav_rank_from_start + 1) = 15 := by
  sorry

end madhav_rank_from_last_is_15_l565_565828


namespace sum_of_divisors_of_24_l565_565093

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565093


namespace Q_not_polynomial_l565_565442

noncomputable def Q (x : ℝ) : ℝ := | x |

theorem Q_not_polynomial : ¬∃ (P : Polynomial ℝ), ∀ (x : ℝ), P.eval x = Q x := by
  sorry

end Q_not_polynomial_l565_565442


namespace m_range_in_inequality_l565_565353

noncomputable def middle_term_expansion (x : ℝ) : ℝ := (5 / 2) * x^3

def holds_true_in_interval (f : ℝ → ℝ) (m : ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ m * x

theorem m_range_in_inequality :
  (∀ x ∈ Icc (sqrt 2 / 2) (sqrt 2), middle_term_expansion x ≤ m * x) ↔ 5 ≤ m :=
sorry

end m_range_in_inequality_l565_565353


namespace andy_final_position_l565_565872

theorem andy_final_position : 
  let start := (-30, 30)
  let moves := 1022
  let final_position := (-1562, 881) in
  (andy_position_after_moves start moves) = final_position := sorry

def andy_position_after_moves (start : Int × Int) (moves : Int) : Int × Int := sorry

end andy_final_position_l565_565872


namespace range_a_l565_565939

def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 1 then x^2 - 3*x + 2*a 
  else x - a * Real.log x

theorem range_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ 1 ≤ a ∧ a ≤ Real.exp 1 := by
  sorry

end range_a_l565_565939


namespace area_fraction_above_line_l565_565885

-- Define the points of the rectangle
def A := (2,0)
def B := (7,0)
def C := (7,4)
def D := (2,4)

-- Define the points used for the line
def P := (2,1)
def Q := (7,3)

-- The area of the rectangle
def rect_area := (7 - 2) * 4

-- The fraction of the area of the rectangle above the line
theorem area_fraction_above_line : 
  ∀ A B C D P Q, 
    A = (2,0) → B = (7,0) → C = (7,4) → D = (2,4) →
    P = (2,1) → Q = (7,3) →
    (rect_area = 20) → 1 - ((1/2) * 5 * 2 / 20) = 3 / 4 :=
by
  intros A B C D P Q
  intros hA hB hC hD hP hQ h_area
  sorry

end area_fraction_above_line_l565_565885


namespace value_of_x_satisfies_equation_l565_565917

theorem value_of_x_satisfies_equation :
  let x := 8 / 3 in
  25^(-3) = (5^(64/x)) / (5^(40/x) * 25^(20/x)) :=
by
  -- The proof will go here
  sorry

end value_of_x_satisfies_equation_l565_565917


namespace distance_before_rest_l565_565822

theorem distance_before_rest (total_distance after_rest_distance : ℝ) (h1 : total_distance = 1) (h2 : after_rest_distance = 0.25) :
  total_distance - after_rest_distance = 0.75 :=
by sorry

end distance_before_rest_l565_565822


namespace exam_pass_probability_l565_565179

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * p^k * (1 - p)^(n - k)

theorem exam_pass_probability :
  let n := 4
  let k := 3
  let p := 0.4
  let prob_3 := binomial_prob n 3 p
  let prob_4 := binomial_prob n 4 p
  prob_3 + prob_4 = 0.18 :=
by
  sorry

end exam_pass_probability_l565_565179


namespace Z_equals_i_l565_565608

noncomputable def Z : ℂ := (Real.sqrt 2 - (Complex.I ^ 3)) / (1 - Real.sqrt 2 * Complex.I)

theorem Z_equals_i : Z = Complex.I := 
by 
  sorry

end Z_equals_i_l565_565608


namespace max_a1_value_l565_565613

theorem max_a1_value (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n+2) = a n + a (n+1))
    (h2 : ∀ n : ℕ, a n > 0) (h3 : a 5 = 60) : a 1 ≤ 11 :=
by 
  sorry

end max_a1_value_l565_565613


namespace vector_parallel_y_value_l565_565935

theorem vector_parallel_y_value (y : ℝ) 
  (a : ℝ × ℝ := (3, 2)) 
  (b : ℝ × ℝ := (6, y)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  y = 4 :=
by sorry

end vector_parallel_y_value_l565_565935


namespace find_third_number_l565_565972

-- Given conditions
variable (A B C : ℕ)
variable (LCM HCF : ℕ)
variable (h1 : A = 36)
variable (h2 : B = 44)
variable (h3 : LCM = 792)
variable (h4 : HCF = 12)
variable (h5 : A * B * C = LCM * HCF)

-- Desired proof
theorem find_third_number : C = 6 :=
by
  sorry

end find_third_number_l565_565972


namespace first_number_l565_565432

theorem first_number (x g k l : ℤ) (hg : g = 2)
  (h1 : x = k * g + 7) (h2 : 2037 = l * g + 5) : 
  x = 7 :=
by
  have h_g : g = 2 := hg
  have h_l : l = 1016,
  { have h3 : 2032 = 2 * l := by
      rw [← h_g, h2]
      linarith,
    linarith }
  have h_x : x = 2 * 0 + 7 := h1
  rw [mul_zero, add_zero] at h_x
  exact h_x

end first_number_l565_565432


namespace sum_of_divisors_of_24_l565_565021

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565021


namespace fraction_to_terminanting_decimal_l565_565231

theorem fraction_to_terminanting_decimal : (47 / (5^4 * 2) : ℚ) = 0.0376 := 
by 
  sorry

end fraction_to_terminanting_decimal_l565_565231


namespace sum_divisors_24_l565_565033

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565033


namespace fencing_required_l565_565169

def width : ℝ := 25
def area : ℝ := 260
def height_difference : ℝ := 15
def extra_fencing_per_5ft_height : ℝ := 2

noncomputable def length : ℝ := area / width

noncomputable def expected_fencing : ℝ := 2 * length + width + (height_difference / 5) * extra_fencing_per_5ft_height

-- Theorem stating the problem's conclusion
theorem fencing_required : expected_fencing = 51.8 := by
  sorry -- Proof will go here

end fencing_required_l565_565169


namespace find_vector_on_line_at_neg2_l565_565174

-- Definitions of the given conditions and parameters
def s1_vec := (2, 5)  -- Vector on the line at s = 1
def s4_vec := (8, -7) -- Vector on the line at s = 4
def s_neg2_vec := (-4, 17) -- Vector on the line at s = -2, which we aim to prove

-- The statement to be proven
theorem find_vector_on_line_at_neg2 (b e : ℝ × ℝ) :
  b + 1 * e = s1_vec →
  b + 4 * e = s4_vec →
  b + (-2) * e = s_neg2_vec :=
by
  sorry -- This skips the proof.

end find_vector_on_line_at_neg2_l565_565174


namespace decimal_rational_l565_565835

theorem decimal_rational (a : ℕ → ℤ) (h₁ : ∀ n, a n = (1^2 + 2^2 + ... + n^2) % 10) : 
  ∃ p q : ℕ, p / q = ∑' n, (a n) * 0.1^(n+1) :=
by
  sorry

end decimal_rational_l565_565835


namespace better_words_count_l565_565552

-- Define the conditions for a "better word"
def is_better_word (s : String) : Prop :=
  ∀ (i : Nat), i < s.length - 1 →
    (s.get ⟨i, sorry⟩, s.get ⟨i+1, sorry⟩) ∉ [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]

-- Define the length of the sequence
def length := 7

-- The expected number of better words of given length
def expected_count := 2916

-- Lean statement to prove the number of better words of length 7 is 2916
theorem better_words_count : 
  (∃ (n : Nat), n = length) → 
  (∃ (count : Nat), count = expected_count) → 
  (card { s : String | s.length = length ∧ is_better_word s } = expected_count) := 
by
  sorry

end better_words_count_l565_565552


namespace triangle_equilateral_iff_angles_equal_l565_565730

variables {A B C F M : Type} [IncidenceT {A B C F M}] [IsTriangle A B C]
variables [IsAcuteTriangle A B C] [IsAltitude F C B A] [IsMidpoint M C A]
variables [Eq CF BM] [Ne A B] [Ne A C] [Ne B C]

theorem triangle_equilateral_iff_angles_equal :
  (angle M B C = angle F C A) ↔ IsEquilateral A B C :=
sorry

end triangle_equilateral_iff_angles_equal_l565_565730


namespace range_of_b_l565_565671

theorem range_of_b (b : ℝ) (h : Real.sqrt ((b-2)^2) = 2 - b) : b ≤ 2 :=
by {
  sorry
}

end range_of_b_l565_565671


namespace solution_one_solution_two_l565_565603

section

variables {a x : ℝ}

def f (x : ℝ) (a : ℝ) := |2 * x - a| - |x + 1|

-- (1) Prove the solution set for f(x) > 2 when a = 1 is (-∞, -2/3) ∪ (4, ∞)
theorem solution_one (x : ℝ) : f x 1 > 2 ↔ x < -2/3 ∨ x > 4 :=
by sorry

-- (2) Prove the range of a for which f(x) + |x + 1| + x > a² - 1/2 always holds for x ∈ ℝ is (-1/2, 1)
theorem solution_two (a : ℝ) : 
  (∀ x, f x a + |x + 1| + x > a^2 - 1/2) ↔ -1/2 < a ∧ a < 1 :=
by sorry

end

end solution_one_solution_two_l565_565603


namespace fraction_age_8_years_ago_is_1_6_l565_565846

-- Definitions based on the conditions
def age_current : ℕ := 32
def age_ago : ℕ := age_current - 8
def age_hence : ℕ := age_current + 8

-- Definition of the fraction (goal statement to be proven)
def fraction (F : ℚ) : Prop :=
  F * age_ago = (1 / 10) * age_hence

-- Lean 4 statement (proof problem)
theorem fraction_age_8_years_ago_is_1_6 : fraction (1 / 6) :=
by
  rw [age_ago, age_hence]
  simp
  sorry

end fraction_age_8_years_ago_is_1_6_l565_565846


namespace find_n_l565_565674

open Nat

def is_solution_of_comb_perm (n : ℕ) : Prop :=
    3 * (factorial (n-1) / (factorial (n-5) * factorial 4)) = 5 * (n-2) * (n-3)

theorem find_n (n : ℕ) (h : is_solution_of_comb_perm n) (hn : n ≠ 0) : n = 9 :=
by
  -- will fill proof steps if required
  sorry

end find_n_l565_565674


namespace unique_solution_l565_565907

variables {x y z : ℝ}

def equation1 (x y z : ℝ) : Prop :=
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z

def equation2 (x y z : ℝ) : Prop :=
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3

theorem unique_solution :
  equation1 x y z ∧ equation2 x y z → x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
by
  sorry

end unique_solution_l565_565907


namespace sum_divisors_24_l565_565029

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565029


namespace constant_and_max_term_in_binomial_expansion_l565_565976

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_and_max_term_in_binomial_expansion 
  (x : ℝ) 
  (h : x ≠ 0) 
  (h_ratio : binomial 15 3 / binomial 15 4 = 1 / 3) :
  ∃ t1 t2 : ℤ, 
  t1 = binomial 15 7 ∧ 
  t2 = binomial 15 8 ∧ 
  (x^0 ∈ ((λ r, binomial 15 r * x ^ ((15 - 3 * r) / 2)) '' {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})) ∧
  (t1 = binomial 15 7 ∧ t2 = binomial 15 8 ∨ t2 = binomial 15 7 ∧ t1 = binomial 15 8) :=
sorry

end constant_and_max_term_in_binomial_expansion_l565_565976


namespace intersection_P_Q_l565_565347

theorem intersection_P_Q (P Q: set ℝ) (hP: P = {x | x ≤ 1}) (hQ: Q = {x | -1 ≤ x ∧ x ≤ 2}) : 
  P ∩ Q = {x | -1 ≤ x ∧ x ≤ 1} :=
by
  sorry

end intersection_P_Q_l565_565347


namespace probability_multiple_of_3_5_7_l565_565200

-- Define the main problem
def cards := Finset.range 101 -- cards are from 1 to 100

def is_multiple_of_3_5_7 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0

-- The theorem to be proven
theorem probability_multiple_of_3_5_7 :
  (cards.filter is_multiple_of_3_5_7).card / (cards.card : ℚ) = 11 / 20 :=
by
  sorry

end probability_multiple_of_3_5_7_l565_565200


namespace locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l565_565754

noncomputable def locus_of_C (a x0 y0 ξ η : ℝ) : Prop :=
  (x0 - ξ) * η^2 - 2 * ξ * y0 * η + ξ^3 - 3 * x0 * ξ^2 - a^2 * ξ + 3 * a^2 * x0 = 0

noncomputable def special_case (a ξ η : ℝ) : Prop :=
  ξ = 0 ∨ ξ^2 + η^2 = a^2

theorem locus_of_C_general_case_eq_cubic (a x0 y0 ξ η : ℝ) (hs: locus_of_C a x0 y0 ξ η) : 
  locus_of_C a x0 y0 ξ η := 
  sorry

theorem locus_of_C_special_case_eq_y_axis_or_circle (a ξ η : ℝ) : 
  special_case a ξ η := 
  sorry

end locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l565_565754


namespace problem1_satisfaction_problem2_satisfaction_l565_565936

variables {Real : Type} [has_zero Real] [has_add Real] [has_mul Real] [has_neg Real]
variables (a b : Real × Real) (k m : Real)

-- Define vectors a and b
def vector_a : Real × Real := (1, 0)
def vector_b : Real × Real := (2, 1)

-- Statement for Problem (1) Collinearity Condition
def collinear_vect1 (k : Real) : Prop :=
  k * fst vector_a - fst vector_b = - (fst (vector_a) + 2 * fst (vector_b)) *
  (snd (k * vector_a - vector_b = - snd (vector_a + 2 * vector_b)))

-- Statement for Problem (2) Collinearity Condition
def collinear_points (m : Real) : Prop :=
  2 * vector_a + 3 * vector_b = 2 * (vector_a + m * vector_b)

-- Proof (skipped)
theorem problem1_satisfaction : ∃ k : Real, collinear_vect1 vector_a vector_b k :=
  sorry

theorem problem2_satisfaction : ∃ m : Real, collinear_points vector_a vector_b m :=
  sorry

end problem1_satisfaction_problem2_satisfaction_l565_565936


namespace sum_of_divisors_of_24_l565_565081

theorem sum_of_divisors_of_24 : ∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n = 60 :=
by
  -- Lean proof goes here
  sorry

end sum_of_divisors_of_24_l565_565081


namespace true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l565_565816

theorem true_if_a_gt_1_and_b_gt_1_then_ab_gt_1 (a b : ℝ) (ha : a > 1) (hb : b > 1) : ab > 1 :=
sorry

end true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l565_565816


namespace xiaoming_commute_times_l565_565573

-- Definitions of the properties and conditions
structure RoadSegments :=
  (AB BC CD : ℕ)
  (length_ratio: AB * 2 = BC ∧ BC = CD * 2)

structure Speed :=
  (flat uphill downhill : ℕ)
  (speed_ratio : flat * 2 = uphill * 3 ∧ uphill * 2 = downhill * 1)

def time_ratio : ℚ := 19 / 16

-- Main theorem statement
theorem xiaoming_commute_times (r: RoadSegments) (s: Speed)
  (h : r.length_ratio) (hs: s.speed_ratio) :
  ∃ (n m : ℕ), n.gcd m = 1 ∧ time_ratio = (n : ℚ) / (m : ℚ) ∧ n + m = 35 :=
by
  existsi 19, 16
  simp
  sorry

end xiaoming_commute_times_l565_565573


namespace range_of_a_for_inequality_l565_565314

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_for_inequality_l565_565314


namespace find_a_and_b_find_c_min_val_l565_565980

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := -x^3 + a * x^2 + b * x + c
def f' (x : ℝ) : ℝ := -3 * x^2 + 2 * a * x + b

theorem find_a_and_b (h1 : f' a b (-1) = 0) (h2 : f' a b 3 = 0) : a = 3 ∧ b = 9 :=
sorry

noncomputable def f'3 (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 9
noncomputable def f3 (x : ℝ) (c : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + c

theorem find_c_min_val (hmax : ∃ x ∈ set.Icc (-2:ℝ) (2:ℝ), f3 x c = 20) : 
  (f3 (-2) c = 20 → f3 (-1) c = 13) ∧
  (f3 2 c = 20 → f3 (-1) c = -7) :=
sorry

end find_a_and_b_find_c_min_val_l565_565980


namespace first_system_solution_second_system_no_solution_third_system_solution_l565_565392

-- First system
theorem first_system_solution :
  ∃ x, x ≡ -52 [MOD 210] ∧ x ≡ 2 [MOD 6] ∧ x ≡ 3 [MOD 5] ∧ x ≡ 4 [MOD 7] :=
sorry

-- Second system
theorem second_system_no_solution :
  ¬ ∃ x, x ≡ 2 [MOD 6] ∧ x ≡ 3 [MOD 10] ∧ x ≡ 4 [MOD 7] :=
sorry

-- Third system
theorem third_system_solution :
  ∃ x, x ≡ 58 [MOD 84] ∧ x ≡ 10 [MOD 12] ∧ x ≡ 16 [MOD 21] :=
sorry

end first_system_solution_second_system_no_solution_third_system_solution_l565_565392


namespace range_of_a_l565_565956

theorem range_of_a (a : ℝ) (h₁ : ∀ x : ℝ, x > 0 → x + 4 / x ≥ a) (h₂ : ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) :
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l565_565956


namespace sum_of_divisors_of_24_is_60_l565_565062

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565062


namespace perpendicular_lines_slope_relation_l565_565633

theorem perpendicular_lines_slope_relation (a : ℝ) :
  let line1_slope := (2 - a) / a,
      line2_slope := -2 / 3 in
  line1_slope * line2_slope = -1 → a = 4 / 5 :=
begin
  intros line1_slope line2_slope h,
  have h_mul : (line1_slope * line2_slope = -1) := h,
  have h_equation : (2 - a) / a * (-2 / 3) = -1 := h_mul,
  sorry
end

end perpendicular_lines_slope_relation_l565_565633


namespace minimal_tiles_to_cover_immovably_l565_565778

def two_times_one_or_one_times_two_tile (n : ℕ) := (n % 2 = 0 ∧ n >= 1)
def board := fin 8 × fin 8
def number_of_tiles (n : ℕ) : Prop := 64 - 2 * n

theorem minimal_tiles_to_cover_immovably : ∃ n, (two_times_one_or_one_times_two_tile n) ∧ number_of_tiles n = 28 := by
  sorry

end minimal_tiles_to_cover_immovably_l565_565778


namespace angle_is_17_degrees_44_times_l565_565998

-- Definitions based on the given conditions
def minute_hand_degrees_per_minute := 6
def hour_hand_degrees_per_hour := 30
def minute_hand_degrees_in_one_hour := 360
def hour_hand_degrees_in_one_day := 360 * 2

noncomputable def angle_between_hands (t : ℕ) : ℝ :=
  let minute_angle := (t % 60) * minute_hand_degrees_per_minute
  let hour_angle := ((t // 60) % 12) * hour_hand_degrees_per_hour + (t % 60) * 0.5
  abs ((minute_angle - hour_angle) % 360 - 360  * if (minute_angle - hour_angle > 180) then 1 else 0)

theorem angle_is_17_degrees_44_times :
  (∃ N : ℕ, N = 44 ∧ ∀ t ≤ 1440 * 2, (angle_between_hands t = 17 ∨ angle_between_hands t = 343) → N = 44) :=
sorry

end angle_is_17_degrees_44_times_l565_565998


namespace sum_divisors_24_l565_565027

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565027


namespace sum_of_divisors_of_24_l565_565127

theorem sum_of_divisors_of_24 : ∑ d in (Finset.divisors 24), d = 60 := by
  sorry

end sum_of_divisors_of_24_l565_565127


namespace min_circles_for_6x3_rectangle_min_circles_for_5x3_rectangle_l565_565014

theorem min_circles_for_6x3_rectangle (r : ℝ) (h_r : r = real.sqrt 2) : 
  ∀ (w : ℝ) (h : ℝ), w = 6 ∧ h = 3 → 
    let radius := r in 
    let rect_height := h in 
    let rect_width := w in 
    ∃ n : ℕ, n = 6 ∧ 
      (∀ (x : ℝ) (y : ℝ), (0 ≤ x ∧ x ≤ rect_width) → (0 ≤ y ∧ y ≤ rect_height) → 
      ∃ (i : ℕ) (cx cy : ℝ), 0 ≤ cx ∧ cx ≤ rect_width ∧ 0 ≤ cy ∧ cy ≤ rect_height ∧ (sqrt ((x - cx) ^ 2 + (y - cy) ^ 2) ≤ radius)) := 
sorry

theorem min_circles_for_5x3_rectangle (r : ℝ) (h_r : r = real.sqrt 2) : 
  ∀ (w : ℝ) (h : ℝ), w = 5 ∧ h = 3 → 
    let radius := r in 
    let rect_height := h in 
    let rect_width := w in 
    ∃ n : ℕ, n = 5 ∧ 
      (∀ (x : ℝ) (y : ℝ), (0 ≤ x ∧ x ≤ rect_width) → (0 ≤ y ∧ y ≤ rect_height) → 
      ∃ (i : ℕ) (cx cy : ℝ), 0 ≤ cx ∧ cx ≤ rect_width ∧ 0 ≤ cy ∧ cy ≤ rect_height ∧ (sqrt ((x - cx) ^ 2 + (y - cy) ^ 2) ≤ radius)) := 
sorry

end min_circles_for_6x3_rectangle_min_circles_for_5x3_rectangle_l565_565014


namespace babysitter_total_hours_l565_565474

-- Define the regular rate and hours
def regular_rate := 16
def regular_hours := 30

-- Define the overtime rate which is 75% higher than the regular rate
def overtime_rate := regular_rate + (0.75 * regular_rate)

-- Define the total earnings of the babysitter for the week
def total_earnings := 760

-- Total hours worked by babysitter is sum of regular hours and overtime hours
def total_hours_worked (overtime_hours : ℕ) : ℕ :=
  regular_hours + overtime_hours

theorem babysitter_total_hours :
  (∃ x : ℕ, total_earnings = (regular_rate * regular_hours) + (overtime_rate * x)) →
  total_hours_worked 10 = 40 :=
sorry -- Proof can be provided here.

end babysitter_total_hours_l565_565474


namespace correct_option_l565_565559

-- Definition of the conditions
def conditionA : Prop := (Real.sqrt ((-1 : ℝ)^2) = 1)
def conditionB : Prop := (Real.sqrt ((-1 : ℝ)^2) = -1)
def conditionC : Prop := (Real.sqrt (-(1^2) : ℝ) = 1)
def conditionD : Prop := (Real.sqrt (-(1^2) : ℝ) = -1)

-- Proving the correct condition
theorem correct_option : conditionA := by
  sorry

end correct_option_l565_565559


namespace garden_fencing_l565_565500

theorem garden_fencing :
  ∃ (P : ℝ), let L := 100 in let W := L / 2 in P = 2 * (L + W) ∧ P = 300 := by
sorry

end garden_fencing_l565_565500


namespace solution_l565_565364

-- Define the arithmetic sequences and their sum functions
variables {a b : ℕ → ℝ}
variables {S T : ℕ → ℝ}

-- Establish the given conditions as hypotheses
def arithmetic_sequences (a b : ℕ → ℝ) : Prop := 
  ∃ d1 d2 (a0 b0 : ℝ), (∀ n : ℕ, a n = a0 + n * d1) ∧ (∀ n : ℕ, b n = b0 + n * d2)

def sum_of_first_n_terms (S T : ℕ → ℝ) (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, S n = (n * (a 0 + a n)) / 2) ∧ (∀ n : ℕ, T n = (n * (b 0 + b n)) / 2)

-- State the given condition about the ratio of terms
def ratio_of_terms (a b : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, b n ≠ 0 → a n / b n = (2 * n - 1) / (n + 1)

-- Main proof statement
theorem solution : arithmetic_sequences a b →
                   sum_of_first_n_terms S T a b →
                   ratio_of_terms a b →
                   S 11 / T 11 = 11 / 7 :=
by obviously

end solution_l565_565364


namespace derivative_at_1_l565_565681

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end derivative_at_1_l565_565681


namespace cost_price_A_B_jersyes_profit_function_max_profit_max_profit_case2_max_profit_case3_l565_565516

-- Defining variables and conditions
variables (x y a m : ℕ)
variables (m : ℕ) (a : ℕ)

-- Given conditions
def price_B_jersey := 180
def price_A_jersey := 200

-- Cost prices of jerseys problem
theorem cost_price_A_B_jersyes : 
  let x := 180 in
  let y := 200 in
  (30000 / y = 3 * 9000 / x) → 
  (y = x + 20) → 
  x = 180 ∧ y = 200 := sorry

-- Profit function problem
theorem profit_function (m : ℕ) : 
  100 ≤ m ∧ m ≤ 140 → 
  ∃ W, (W = 20 * m + 21000) := sorry

-- Maximum profit after support
theorem max_profit (a m : ℕ) :
  (0 < a ∧ a < 20) → 
  (100 ≤ m ∧ m ≤ 140) → 
  ∃ Q, (Q = (20 - a) * 140 + 21000) → 
  ∃ Q_max, Q_max = 23800 - 140 * a  :=
sorry

theorem max_profit_case2 (a m : ℕ) :
  (a = 20) → 
  (100 ≤ m ∧ m ≤ 140) → 
  ∃ Q, Q = 21000 :=
sorry

theorem max_profit_case3 (a m : ℕ) :
  (a > 20) → 
  (100 ≤ m ∧ m ≤ 140) → 
  ∃ Q, (Q = (20 - a) * 100 + 21000) → 
  ∃ Q_max, Q_max = 23000 - 100 * a :=
sorry

end cost_price_A_B_jersyes_profit_function_max_profit_max_profit_case2_max_profit_case3_l565_565516


namespace division_of_fractions_l565_565666

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l565_565666


namespace S_contains_infinitely_many_primes_l565_565363

def is_in_S (x : ℚ) : Prop :=
  ∃ (n : ℕ) (ai bi : Fin n → ℕ)
  (h_positive : ∀ i, ai i > 0 ∧ bi i > 0),
  x = (∏ i, (ai i)^2 + ai i - 1) / (∏ i, (bi i)^2 + bi i - 1)

theorem S_contains_infinitely_many_primes :
  ∀ N : ℕ, ∃ p : ℕ, p > N ∧ Nat.Prime p ∧ is_in_S p :=
sorry

end S_contains_infinitely_many_primes_l565_565363


namespace problem_l565_565991

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem problem (surj_f : ∀ y, ∃ x, f x = y) 
                (inj_g : ∀ x1 x2, g x1 = g x2 → x1 = x2)
                (f_ge_g : ∀ n, f n ≥ g n) :
  ∀ n, f n = g n := 
by 
  sorry

end problem_l565_565991


namespace total_number_of_games_l565_565430

theorem total_number_of_games (n : ℕ) (k : ℕ) (teams : Finset ℕ)
  (h_n : n = 8) (h_k : k = 2) (h_teams : teams.card = n) :
  (teams.card.choose k) = 28 :=
by
  sorry

end total_number_of_games_l565_565430


namespace product_of_invertible_function_labels_l565_565593

noncomputable def f_6 (x : ℝ) : ℝ := x^3 - 3 * x
def f_6_domain := set.Icc (-2:ℝ) 3

def f_7 : set (ℝ × ℝ) := {(-6, 2), (-5, 5), (-4, 0), (-3, -3), (-2, -5), (-1, -4), (0, -2), (1, 1), (2, 4), (3, 6)}

noncomputable def f_8 (x : ℝ) : ℝ := real.tan x
def f_8_domain := {(-6 : ℝ), -5, -4, -3, -2, -1, 0, 1, 2, 3}

noncomputable def f_9 (x : ℝ) : ℝ := real.sqrt x
def f_9_domain := set.Icc (0:ℝ) 4

theorem product_of_invertible_function_labels : 
  (∃ (is_invertible_6 : bijective (λ x ∈ f_6_domain, f_6 x)), false) ∧
  (∃ (is_invertible_7 : bijective (λ p ∈ f_7, p.2)), true) ∧
  (∃ (is_invertible_8 : bijective (λ x ∈ f_8_domain, f_8 x)), false) ∧
  (∃ (is_invertible_9 : bijective (λ x ∈ f_9_domain, f_9 x)), true) →
  7 * 9 = 63 :=
by sorry

end product_of_invertible_function_labels_l565_565593


namespace sum_of_divisors_of_24_is_60_l565_565060

theorem sum_of_divisors_of_24_is_60 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_of_divisors_of_24_is_60_l565_565060


namespace sum_of_divisors_of_24_l565_565095

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565095


namespace area_of_midpoints_l565_565504

theorem area_of_midpoints (A : ℝ) (hA : A = 9 - (2.25 * Real.pi)) :
  100 * Float.round (Real.to_float A * 100) = 186 := by
sorry

end area_of_midpoints_l565_565504


namespace number_of_sequences_mod_1000_l565_565539

theorem number_of_sequences_mod_1000 : 
  (nat.choose 1016 12) % 1000 = 16 :=
  sorry

end number_of_sequences_mod_1000_l565_565539


namespace longest_side_cosine_value_l565_565690

theorem longest_side_cosine_value (A B C : ℝ) (a b c k : ℝ) 
  (hABC : A + B + C = π) 
  (h_ratio_sines : sin A / sin B = 2 / 3 ∧ sin B / sin C = 3 / 4) 
  (hSides : a = 2 * k ∧ b = 3 * k ∧ c = 4 * k) 
  (hLargestAngle : C = π - (A + B)) : 
  cos C = -1 / 4 := 
by 
  sorry

end longest_side_cosine_value_l565_565690


namespace rotate_90_clockwise_l565_565324

theorem rotate_90_clockwise (x y : ℝ) (h : (x, y) = (-2, 0)) : (y, -x) = (0, 2) :=
by
  rw [h]  -- Replace (x, y) with (-2, 0)
  -- Final goal: (0, 2) = (0, 2), which is true by refl
  exact rfl

end rotate_90_clockwise_l565_565324


namespace find_circle_area_l565_565328

-- Definitions for the conditions
variables {r : ℝ} -- the radius of the circle
variables {O F : Type*} -- center O and point F
variables (DF FG : ℝ) (x : ℝ) -- lengths on the circle diameter

-- Given conditions
def diameter_condition (DF FG : ℝ) : Prop :=
  DF = 8 ∧ FG = 4

def distances_condition (DF FG : ℝ) : ℝ :=
  let DG := DF + FG in DG

-- Hypothesis Statements
lemma geometry_hypothesis (DF FG x : ℝ) (hD : diameter_condition DF FG) : r^2 = x^2 + 64 :=
begin
  have h1 : DF = 8 := hD.left,
  have h2 : FG = 4 := hD.right,
  sorry
end

lemma power_point_condition (DF FG x : ℝ) (hD : diameter_condition DF FG) : r^2 - x^2 = 32 :=
begin
  have h1 : DF = 8 := hD.left,
  have h2 : FG = 4 := hD.right,
  sorry
end

-- The proper proof statement to show r^2 = 32 given the conditions
theorem find_circle_area (DF FG x : ℝ) 
  (hD : diameter_condition DF FG)
  (h_geom : r^2 = x^2 + 64)
  (h_pow : r^2 - x^2 = 32) : ∃ r, r^2 = 32 :=
begin
  sorry
end

-- Concluding the problem by defining the area
lemma area_of_circle : Real := π * 32

end find_circle_area_l565_565328


namespace find_percentage_l565_565478

noncomputable def percentage (X : ℝ) : ℝ := (377.8020134228188 * 100 * 5.96) / 1265

theorem find_percentage : percentage 178 = 178 := by
  -- Conditions
  let P : ℝ := 178
  let A : ℝ := 1265
  let divisor : ℝ := 5.96
  let result : ℝ := 377.8020134228188

  -- Define the percentage calculation
  let X := (result * 100 * divisor) / A

  -- Verify the calculation matches
  have h : X = P := by sorry

  trivial

end find_percentage_l565_565478


namespace sum_of_divisors_of_24_l565_565101

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565101


namespace number_of_correct_propositions_l565_565202

theorem number_of_correct_propositions :
  let prop1 := ∀ (a b c : ℝ), b^2 = a*c → ¬(a, b, c) is_geometric_sequence,
      prop2 := ∫ x in 1..2, (Real.exp x + 1 / x) = Real.exp 2 - Real.exp 1 + Real.log 2,
      prop3 := ∀ (α β : Plane) (l : Line), α ≠ β → α ⊥ β → l ⊥ β → ¬(l ∥ α),
      prop4 := ∀ (A B C P : Point), P ∈ triangle A B C → P = A + 1/2 * (B - A) + 1/3 * (C - A) → area_ratio (triangle A B P) (triangle A B C) = 1/3
  in (¬prop1) + prop2 + (¬prop3) + prop4 = 2 := sorry

end number_of_correct_propositions_l565_565202


namespace pair_factorial_power_of_5_l565_565577

theorem pair_factorial_power_of_5 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h1 : ∃ k : ℕ, a! + b = 5^k) (h2 : ∃ m : ℕ, b! + a = 5^m) : a = 5 ∧ b = 5 :=
begin
  sorry
end

end pair_factorial_power_of_5_l565_565577


namespace find_n_l565_565840

theorem find_n (n : ℕ) (h : ∀ x : ℝ, (n : ℝ) < x ∧ x < (n + 1 : ℝ) → 3 * x - 5 = 0) :
  n = 1 :=
sorry

end find_n_l565_565840


namespace sunflower_is_taller_l565_565751

def sister_height_ft : Nat := 4
def sister_height_in : Nat := 3
def sunflower_height_ft : Nat := 6

def feet_to_inches (ft : Nat) : Nat := ft * 12

def sister_height := feet_to_inches sister_height_ft + sister_height_in
def sunflower_height := feet_to_inches sunflower_height_ft

def height_difference : Nat := sunflower_height - sister_height

theorem sunflower_is_taller : height_difference = 21 :=
by
  -- proof has to be provided:
  sorry

end sunflower_is_taller_l565_565751


namespace division_of_fractions_l565_565667

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l565_565667


namespace min_value_fraction_l565_565966

noncomputable section

open Real

theorem min_value_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 4) : 
  ∃ t : ℝ, (∀ x' y' : ℝ, (x' > 0 ∧ y' > 0 ∧ x' + 2 * y' = 4) → (2 / x' + 1 / y') ≥ t) ∧ t = 2 :=
by
  sorry

end min_value_fraction_l565_565966


namespace construct_triangle_ABC_l565_565225

variables {A B C C1 A1 B1 : Type} [AddCommGroup A] [Module ℝ A]

-- Conditions
variables (p q : ℝ) (hpq : p > 0 ∧ q > 0) (h_ratio : (p + q) > 0)

-- Points and vectors
variables (a b c a1 b1 c1 : A)

-- Ratios and their properties
def lambda (p q : ℝ) : ℝ := p / (p + q)
def mu (p q : ℝ) : ℝ := q / (p + q)

theorem construct_triangle_ABC :
  (λ AC1 C1B BA1 A1C CB1 B1A,
    λ h1 : AC1 = lambda p q * (AC1 + C1B),
        h2 : C1B = mu p q * (AC1 + C1B),
        h3 : BA1 = lambda p q * (BA1 + A1C),
        h4 : A1C = mu p q * (BA1 + A1C),
        h5 : CB1 = lambda p q * (CB1 + B1A),
        h6 : B1A = mu p q * (CB1 + B1A),
    ∃ A B C : A, (A = a ∧ B = b ∧ C = c) ∧ (a1 = mu p q * b + lambda p q * c) ∧
                                      (b1 = mu p q * c + lambda p q * a) ∧
                                      (c1 = mu p q * a + lambda p q * b))
  p q AC1 C1B BA1 A1C CB1 B1A sorry sorry sorry sorry sorry sorry := sorry

end construct_triangle_ABC_l565_565225


namespace range_of_m_l565_565947

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h₁ : x1 < x2) (h₂ : y1 < y2)
  (A_on_line : y1 = (2 * m - 1) * x1 + 1)
  (B_on_line : y2 = (2 * m - 1) * x2 + 1) :
  m > 0.5 :=
sorry

end range_of_m_l565_565947


namespace range_of_a_l565_565271

open Real

noncomputable def p (a : ℝ) := ∀ (x : ℝ), x ≥ 1 → (2 * x - 3 * a) ≥ 0
noncomputable def q (a : ℝ) := (0 < 2 * a - 1) ∧ (2 * a - 1 < 1)

theorem range_of_a (a : ℝ) : p a ∧ q a ↔ (1/2 < a ∧ a ≤ 2/3) := by
  sorry

end range_of_a_l565_565271


namespace probability_of_selecting_leaders_l565_565322

theorem probability_of_selecting_leaders :
  let clubs := [6, 8, 9, 10]
  let club_probability := 1 / 4
  let probability (n : ℕ) : ℚ := (Nat.choose (n - 3) 1 : ℚ) / (Nat.choose n 4)
  let total_probability := club_probability * 
    ((probability 6 + probability 8 + probability 9 + probability 10) : ℚ)
  total_probability = 37 / 420 :=
by
  sorry

end probability_of_selecting_leaders_l565_565322


namespace gcd_8251_6105_l565_565440

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l565_565440


namespace problem_statement_l565_565748

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom domain_real : ∀ x : ℝ, x ∈ ℝ
axiom increasing_on_nonnegative : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem problem_statement : f (-2) < f (-3) ∧ f (-3) < f π :=
by {
  have h1 : f (-3) = f 3 := even_function 3,
  have h2 : f (-2) = f 2 := even_function 2,
  have h3 : 2 < 3 := by linarith,
  have h4 : 3 < π := by norm_num,
  have h5 : 0 ≤ 2 := by linarith,
  have h6 : 0 ≤ 3 := by linarith,
  have h7 : 2 ≤ 3 := by linarith,
  have h8 : 3 ≤ π := by norm_num,
  have h9 : f 2 < f 3 := increasing_on_nonnegative 2 3 h5 h7,
  have h10 : f 3 < f π := increasing_on_nonnegative 3 π h6 h8,
  split,
  { rw h2, rw h1, exact h9, },
  { exact h10, }
}

end problem_statement_l565_565748


namespace sum_of_divisors_of_24_l565_565018

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565018


namespace div_equiv_l565_565405

theorem div_equiv : (0.75 / 25) = (7.5 / 250) :=
by
  sorry

end div_equiv_l565_565405


namespace PQ_fraction_of_ST_l565_565759

theorem PQ_fraction_of_ST (P Q S T : Point) (SP PT SQ QT PQ ST : ℝ)
  (h1 : P ∈ lineSegment S T)
  (h2 : Q ∈ lineSegment S T)
  (h3 : SP = 5 * PT)
  (h4 : SQ = 7 * QT)
  (h5 : ST = SP + PT)
  (h6 : ST = SQ + QT)
  (h7 : PQ = SQ - SP):
  PQ = (1 / 24) * ST := 
sorry

end PQ_fraction_of_ST_l565_565759


namespace cost_price_A_B_jersyes_profit_function_max_profit_max_profit_case2_max_profit_case3_l565_565515

-- Defining variables and conditions
variables (x y a m : ℕ)
variables (m : ℕ) (a : ℕ)

-- Given conditions
def price_B_jersey := 180
def price_A_jersey := 200

-- Cost prices of jerseys problem
theorem cost_price_A_B_jersyes : 
  let x := 180 in
  let y := 200 in
  (30000 / y = 3 * 9000 / x) → 
  (y = x + 20) → 
  x = 180 ∧ y = 200 := sorry

-- Profit function problem
theorem profit_function (m : ℕ) : 
  100 ≤ m ∧ m ≤ 140 → 
  ∃ W, (W = 20 * m + 21000) := sorry

-- Maximum profit after support
theorem max_profit (a m : ℕ) :
  (0 < a ∧ a < 20) → 
  (100 ≤ m ∧ m ≤ 140) → 
  ∃ Q, (Q = (20 - a) * 140 + 21000) → 
  ∃ Q_max, Q_max = 23800 - 140 * a  :=
sorry

theorem max_profit_case2 (a m : ℕ) :
  (a = 20) → 
  (100 ≤ m ∧ m ≤ 140) → 
  ∃ Q, Q = 21000 :=
sorry

theorem max_profit_case3 (a m : ℕ) :
  (a > 20) → 
  (100 ≤ m ∧ m ≤ 140) → 
  ∃ Q, (Q = (20 - a) * 100 + 21000) → 
  ∃ Q_max, Q_max = 23000 - 100 * a :=
sorry

end cost_price_A_B_jersyes_profit_function_max_profit_max_profit_case2_max_profit_case3_l565_565515


namespace tuition_difference_l565_565525

theorem tuition_difference :
  ∃ room_and_board_cost diff, 
  let total_cost := 2584 in
  let tuition_cost := 1644 in
  room_and_board_cost = total_cost - tuition_cost ∧
  diff = tuition_cost - room_and_board_cost ∧
  diff = 704 :=
by
  use 2584 - 1644
  use 1644 - (2584 - 1644)
  split
  { refl }
  split
  { refl }
  refl

end tuition_difference_l565_565525


namespace constant_function_graph_is_straight_line_l565_565315

-- Define the problem using the given conditions and required proof.
theorem constant_function_graph_is_straight_line
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, deriv f x = 0) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
begin
  sorry
end

end constant_function_graph_is_straight_line_l565_565315


namespace inequality_proof_l565_565362

theorem inequality_proof (a b c : ℝ) (k : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : k ≥ 1) : 
  (a^(k + 1) / b^k + b^(k + 1) / c^k + c^(k + 1) / a^k) ≥ (a^k / b^(k - 1) + b^k / c^(k - 1) + c^k / a^(k - 1)) :=
by
  sorry

end inequality_proof_l565_565362


namespace gcd_8251_6105_l565_565439

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l565_565439


namespace mel_weight_l565_565531

variable (m : ℕ)

/-- Brenda's weight is 10 pounds more than three times Mel's weight. 
    Given Brenda's weight is 220 pounds, we prove Mel's weight is 70 pounds. -/
theorem mel_weight : (3 * m + 10 = 220) → (m = 70) :=
by
  intros h,
  sorry

end mel_weight_l565_565531


namespace lattice_points_count_l565_565996

theorem lattice_points_count : ∃ (S : Finset (ℤ × ℤ)), 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 45}.toFinset = S ∧ S.card = 6 := 
sorry

end lattice_points_count_l565_565996


namespace symmetric_poly_roots_identity_l565_565652

variable (a b c : ℝ)

theorem symmetric_poly_roots_identity (h1 : a + b + c = 6) (h2 : ab + bc + ca = 5) (h3 : abc = 1) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) = 38 :=
by
  sorry

end symmetric_poly_roots_identity_l565_565652


namespace sequence_ineq_l565_565549

noncomputable def a : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := a (n+1) + a n

theorem sequence_ineq (n : ℕ) (h : 0 < n) : real.sqrt (a 0) ≤ 1 :=
begin
  -- Prove given inequality
  sorry
end

end sequence_ineq_l565_565549


namespace max_bottles_drank_l565_565164

-- Definitions based on conditions from part a)
def cost_per_bottle := 2
def exchange_rate := 2
def initial_yuan := 30

-- The maximum number of bottles that can be drunk
theorem max_bottles_drank (cost_per_bottle exchange_rate initial_yuan : ℕ) (h₁ : cost_per_bottle = 2) (h₂ : exchange_rate = 2) (h₃ : initial_yuan = 30) : 
  let initial_bottles := initial_yuan / cost_per_bottle
      additional_bottles := (initial_bottles - 2 *_unit) 
  initial_bottles + additional_bottles = 29 := 
by
  sorry

end max_bottles_drank_l565_565164


namespace occupied_rooms_expression_price_for_target_profit_l565_565856

-- We will start with a noncomputable theory to ease the definition of mathematical proofs
noncomputable theory

-- Conditions
def total_rooms : ℕ := 50
def original_price : ℝ := 190
def reduced_price : ℝ := 180
def cost_per_room : ℝ := 20
def max_price : ℝ := 1.5 * original_price

-- Problem to prove (Part 1)
def occupied_rooms (x : ℝ) : ℝ := total_rooms - x / 10

-- Statement for Proof 1
theorem occupied_rooms_expression (x : ℝ) : occupied_rooms x = 50 - x / 10 :=
by
  sorry

-- Problem to prove (Part 2)
theorem price_for_target_profit (x : ℝ) : 
  180 + x = 230 ∧ (180 + x - cost_per_room) * occupied_rooms x = 9450 :=
by
  sorry


end occupied_rooms_expression_price_for_target_profit_l565_565856


namespace problem_solution_l565_565358

noncomputable def problem_statement (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : Prop :=
  ∑ (x, y) in ({(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)} : Finset (ℝ × ℝ)), x / y + 6 ≥
  2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c))

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  problem_statement a b c h1 h2 h3 h4 :=
by
  sorry

end problem_solution_l565_565358


namespace area_of_larger_square_only_l565_565708

def square_area (side_length : ℝ) : ℝ := side_length * side_length

theorem area_of_larger_square_only :
  let larger_square_side := 8
  let smaller_square_side := 4
  let larger_square_area := square_area larger_square_side
  let smaller_square_area := square_area smaller_square_side
  let overlap_area := smaller_square_area
  larger_square_area - overlap_area = 48 := by
  let larger_square_side := 8
  let smaller_square_side := 4
  let larger_square_area := square_area larger_square_side
  let smaller_square_area := square_area smaller_square_side
  let overlap_area := smaller_square_area
  show larger_square_area - overlap_area = 48 from sorry

end area_of_larger_square_only_l565_565708


namespace maximum_value_of_f_l565_565243

noncomputable def f (x : ℝ) : ℝ := sin (π / 2 + 2 * x) - 5 * sin x

theorem maximum_value_of_f : ∃ x, f x = 4 := by
sorry

end maximum_value_of_f_l565_565243


namespace complete_the_square_l565_565770

theorem complete_the_square (d e f : ℤ) (h1 : 0 < d)
    (h2 : ∀ x : ℝ, 100 * x^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f) :
  d + e + f = 112 := by
  sorry

end complete_the_square_l565_565770


namespace intersection_eq_l565_565988

def A : set ℝ := { x | 1 < x }
def B : set ℝ := { y | 3 < y }

theorem intersection_eq : (A ∩ B) = { z | 3 < z } :=
by sorry

end intersection_eq_l565_565988


namespace total_songs_is_correct_l565_565837

-- Define the conditions in Lean
def country_albums : ℕ := 4
def pop_albums : ℕ := 5
def songs_per_album : ℕ := 8

-- Define the total number of albums
def total_albums : ℕ := country_albums + pop_albums

-- Prove the total number of songs Isabel bought
theorem total_songs_is_correct : total_albums * songs_per_album = 72 :=
by {
  have h1 : total_albums = 9 := by rfl,
  have h2 : 9 * songs_per_album = 72 := by norm_num,
  rw [h1],
  exact h2,
}

end total_songs_is_correct_l565_565837


namespace sum_sequence_l565_565214

-- Define the sequence and the telescoping property
def sequence_term (n : ℕ) : ℚ := 1 / (n * (n + 1))

-- Property of the sequence term
lemma sequence_property (n : ℕ) (h : 2 ≤ n ∧ n ≤ 10) :
  sequence_term n = 1 / n - 1 / (n + 1) :=
begin
  have h1 : (n : ℚ) ≠ 0 := by linarith,
  have h2 : (n + 1 : ℚ) ≠ 0 := by linarith,
  field_simp [sequence_term, h1, h2],
  rw [mul_add, mul_one],
  ring,
end

-- The main theorem to prove the sum of the sequence
theorem sum_sequence : 
  (∑ n in finset.range 9, sequence_term (n + 2)) = 9 / 22 :=
begin
  -- Skipping the actual proof for now
  sorry
end

end sum_sequence_l565_565214


namespace sum_of_divisors_of_24_l565_565071

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565071


namespace usual_time_of_train_l565_565826

theorem usual_time_of_train (S T : ℝ) (h_speed : S ≠ 0) 
(h_speed_ratio : ∀ (T' : ℝ), T' = T + 3/4 → S * T = (4/5) * S * T' → T = 3) : Prop :=
  T = 3

end usual_time_of_train_l565_565826


namespace ellipse_proof_l565_565974

-- Ellipse definition and properties
def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Condition definitions
variables (a b c : ℝ)
def eccentricity (e : ℝ) : Prop := e = c / a
def vertices (b : ℝ) : Prop := b = 2
def ellipse_property (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- The main proof statement
theorem ellipse_proof
  (a b c : ℝ)
  (x y : ℝ)
  (e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : b < a)
  (h4 : eccentricity a c e)
  (h5 : e = (√5) / 5)
  (h6 : vertices b)
  (h7 : ellipse_property a b c) :
  ellipse_equation (√5) 2 x y :=
by
  -- "sorry" is a placeholder. The actual proof would go here.
  sorry

end ellipse_proof_l565_565974


namespace octagon_diagonal_length_l565_565583

theorem octagon_diagonal_length (s : ℝ) (h : s = 12) : 
  let AC := s * Real.sqrt 2
  AC = 12 * Real.sqrt 2 :=
by
  -- Definitions based on given conditions
  let side_length := 12
  have h_side_length : s = side_length := h
  -- Use the diagonal length of a regular octagon property
  let diagonal_length := side_length * Real.sqrt 2
  -- Theorem statement
  show AC = diagonal_length
  -- Proof not required, since we are only providing the statement
  sorry

end octagon_diagonal_length_l565_565583


namespace remainder_when_divided_by_7_l565_565454

theorem remainder_when_divided_by_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l565_565454


namespace no_subseq_010101_l565_565335

def seq : ℕ → ℕ
| 0 := 1
| 1 := 0
| 2 := 1
| 3 := 0
| 4 := 1
| 5 := 0
| (n + 6) := (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3) + seq (n + 4) + seq (n + 5)) % 10

theorem no_subseq_010101 :
  ¬ (∃ n, seq n = 0 ∧ seq (n + 1) = 1 ∧ seq (n + 2) = 0 ∧ seq (n + 3) = 1 ∧ seq (n + 4) = 0 ∧ seq (n + 5) = 1) :=
sorry

end no_subseq_010101_l565_565335


namespace calculate_area_ADC_l565_565709

def area_AD (BD DC : ℕ) (area_ABD : ℕ) := 
  area_ABD * DC / BD

theorem calculate_area_ADC
  (BD DC : ℕ) 
  (h_ratio : BD = 5 * DC / 2)
  (area_ABD : ℕ)
  (h_area_ABD : area_ABD = 35) :
  area_AD BD DC area_ABD = 14 := 
by 
  sorry

end calculate_area_ADC_l565_565709


namespace sum_of_divisors_of_24_l565_565111

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565111


namespace two_talents_students_l565_565859

-- Definitions and conditions
def total_students : ℕ := 120
def cannot_sing : ℕ := 50
def cannot_dance : ℕ := 75
def cannot_act : ℕ := 35

-- Definitions based on conditions
def can_sing : ℕ := total_students - cannot_sing
def can_dance : ℕ := total_students - cannot_dance
def can_act : ℕ := total_students - cannot_act

-- The main theorem statement
theorem two_talents_students : can_sing + can_dance + can_act - total_students = 80 :=
by
  -- substituting actual numbers to prove directly
  have h_can_sing : can_sing = 70 := rfl
  have h_can_dance : can_dance = 45 := rfl
  have h_can_act : can_act = 85 := rfl
  sorry

end two_talents_students_l565_565859


namespace sum_of_divisors_of_24_l565_565079

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565079


namespace sequence_uv_sum_l565_565229

theorem sequence_uv_sum :
  ∀ (s : ℕ → ℚ) (r u v : ℚ), s 0 = 4096 ∧ 
  (∀ n, s (n+1) = s n * r) ∧ 
  s 5 = 4 ∧ 
  s 6 = 1 ∧ 
  4 * r = 1 ∧ 
  u = s 3 ∧ 
  v = s 4 →
  (u + v = 80) := 
begin
  intros s r u v hs, -- Introduce hypotheses
  sorry, -- Placeholder for proof
end

end sequence_uv_sum_l565_565229


namespace similar_triangle_area_ratio_l565_565673

theorem similar_triangle_area_ratio
  (ABC DEF : Type) [semimodule ℝ ABC] [semimodule ℝ DEF]
  (h_sim : ABC ∼ DEF)
  (h_ratio : ∀ (AB DE : ℝ), AB / DE = 2)
  (area_ABC : ℝ) (area_eq : area_ABC = 8) :
  ∃ (area_DEF : ℝ), area_DEF = 2 :=
by
  sorry

end similar_triangle_area_ratio_l565_565673


namespace parabola_vertex_x_coord_l565_565414

theorem parabola_vertex_x_coord (a b c : ℝ) :
  (∀ x, y = ax^2 + bx + c → ( (-2, 9) ∈ set.pi_univ (λ _, ℝ) ) → 
                     ( (4, 9) ∈ set.pi_univ (λ _, ℝ) ) → 
                     ( (7, 14) ∈ set.pi_univ (λ _, ℝ) ) ) →
  vertex_x_coord(ax^2 + bx + c) = 1 :=
sorry

end parabola_vertex_x_coord_l565_565414


namespace sum_of_divisors_of_24_l565_565055

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n : ℕ, n > 0 ∧ 24 % n = 0) (Finset.range 25)), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l565_565055


namespace smallest_positive_period_of_f_max_min_of_f_increasing_interval_of_f_l565_565295

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos (x + π / 3) + sqrt 3 * (cos x)^2 + (1 / 2) * sin (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := sorry

theorem max_min_of_f :
  ∃ a b : ℝ, (∀ x : ℝ, f x ≤ a ∧ b ≤ f x) ∧ a = 2 ∧ b = -2 := sorry

theorem increasing_interval_of_f : 
  ∀ k : ℤ, ∀ x : ℝ,
  (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12 ∧ 
  ∀ y : ℝ, (k * π - 5 * π / 12 ≤ y ∧ y ≤ k * π + π / 12) → f y = f x) := sorry

end smallest_positive_period_of_f_max_min_of_f_increasing_interval_of_f_l565_565295


namespace ratio_of_democrats_l565_565802

theorem ratio_of_democrats (F M : ℕ) 
  (h1 : F + M = 990) 
  (h2 : (1 / 2 : ℚ) * F = 165) 
  (h3 : (1 / 4 : ℚ) * M = 165) : 
  (165 + 165) / 990 = 1 / 3 := 
by
  sorry

end ratio_of_democrats_l565_565802


namespace points_and_area_l565_565293

noncomputable def curve (ρ θ : ℝ) : Prop := ρ * (sin θ) ^ 2 = 4 * cos θ

noncomputable def line1 (θ : ℝ) : Prop := θ = π / 3

noncomputable def line2 (ρ θ : ℝ) : Prop := ρ * sin θ = 4 * sqrt 3

noncomputable def pointA : ℝ × ℝ := (8 / 3, π / 3)

noncomputable def pointB : ℝ × ℝ := (8 * sqrt 3, π / 6)

theorem points_and_area (A B : ℝ × ℝ) (OA OB : ℝ) (angleAOB : ℝ):
  (curve (fst pointA) (snd pointA)) →
  (line1 (snd pointA)) →
  (A = pointA) →
  (curve (fst pointB) (snd pointB)) →
  (line2 (fst pointB) (snd pointB)) →
  (B = pointB) →
  OA = fst pointA →
  OB = fst pointB →
  angleAOB = π / 6 →
  (1 / 2 * OA * OB * sin angleAOB = 16 / 3 * sqrt 3) :=
by
  -- Proof omitted
  sorry

end points_and_area_l565_565293


namespace sum_of_divisors_of_24_l565_565016

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565016


namespace find_principal_amount_l565_565505

theorem find_principal_amount
  (P : ℝ) (R : ℝ) (T : ℝ) (additional_interest : ℝ) (new_R : ℝ)
  (hT : T = 5)
  (hAdditionalInterest : additional_interest = 1200)
  (hNewR : new_R = R + 3)
  (hInterestDifference : ((P * new_R * T) / 100) - ((P * R * T) / 100) = additional_interest) :
  P = 8000 :=
begin
  sorry
end

end find_principal_amount_l565_565505


namespace greatest_t_solution_l565_565241

theorem greatest_t_solution :
  ∀ t : ℝ, t ≠ 8 ∧ t ≠ -5 →
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) →
  t ≤ -2 :=
by
  sorry

end greatest_t_solution_l565_565241


namespace sum_of_divisors_of_24_l565_565098

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565098


namespace math_problem_l565_565645

-- Define the function f and the conditions for ω
def f (x ω : ℝ) := 2 * (cos (ω * x))^2 - 1 + 2 * sqrt 3 * cos (ω * x) * sin (ω * x)
def axis_of_symmetry (ω : ℝ) := ∀ x, f x ω = f (π/3 - x) ω

-- Given ω, function f reduces to a simpler form
def f_simplified (x : ℝ) := 2 * sin(2 * (1/2) * x + π/6)

-- Define the transformed function g and its condition
def g (x : ℝ) := 2 * sin ((1/2) * (x + 2 * π / 3) + π / 6)
def g_condition (α : ℝ) := g (2 * α + π / 3) = 6 / 5

-- The problem statement in Lean 4
theorem math_problem (ω α : ℝ) (h_ω1 : 0 < ω) (h_ω2 : ω < 1)
  (h_axis_of_symmetry : axis_of_symmetry ω)
  (h_g_condition : α ∈ Ioo 0 (π/2) ∧ g_condition α) :
  ω = 1/2 ∧ sin α = (4 * sqrt 3 - 3) / 10 := 
sorry

end math_problem_l565_565645


namespace john_splits_profit_correctly_l565_565340

-- Conditions
def total_cookies : ℕ := 6 * 12
def revenue_per_cookie : ℝ := 1.5
def cost_per_cookie : ℝ := 0.25
def amount_per_charity : ℝ := 45

-- Computations based on conditions
def total_revenue : ℝ := total_cookies * revenue_per_cookie
def total_cost : ℝ := total_cookies * cost_per_cookie
def total_profit : ℝ := total_revenue - total_cost

-- Proof statement
theorem john_splits_profit_correctly : total_profit / amount_per_charity = 2 := by
  sorry

end john_splits_profit_correctly_l565_565340


namespace crayons_per_child_l565_565228

theorem crayons_per_child (total_crayons children : ℕ) (h_total : total_crayons = 56) (h_children : children = 7) : (total_crayons / children) = 8 := by
  -- proof will go here
  sorry

end crayons_per_child_l565_565228


namespace trigonometric_identity_l565_565562

theorem trigonometric_identity :
  cos (75 * real.pi / 180) * cos (15 * real.pi / 180) - sin (435 * real.pi / 180) * sin (15 * real.pi / 180) = 0 :=
by
  sorry

end trigonometric_identity_l565_565562


namespace daily_pre_promotion_hours_l565_565536

-- Defining conditions
def weekly_additional_hours := 6
def hours_driven_in_two_weeks_after_promotion := 40
def days_in_two_weeks := 14
def hours_added_in_two_weeks := 2 * weekly_additional_hours

-- Math proof problem statement
theorem daily_pre_promotion_hours :
  (hours_driven_in_two_weeks_after_promotion - hours_added_in_two_weeks) / days_in_two_weeks = 2 :=
by
  sorry

end daily_pre_promotion_hours_l565_565536


namespace attention_index_proof_l565_565186

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 10 then 100 * a ^ (x / 10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15 * x
  else 0

theorem attention_index_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 5 a = 140) :
  a = 4 ∧ f 5 4 > f 35 4 ∧ (5 ≤ (x : ℝ) ∧ x ≤ 100 / 3 → f x 4 ≥ 140) :=
by
  sorry

end attention_index_proof_l565_565186


namespace marked_price_correct_l565_565183

-- Condition Definitions
def initial_price : ℝ := 30
def discount_15_perc : ℝ := 0.15
def profit_40_perc : ℝ := 0.40
def discount_25_perc : ℝ := 0.25

-- The theorem we want to prove
theorem marked_price_correct : 
  let cost_price := initial_price * (1 - discount_15_perc),
      selling_price := cost_price * (1 + profit_40_perc),
      marked_price := selling_price / (1 - discount_25_perc) in
  marked_price = 47.60 :=
by
  sorry

end marked_price_correct_l565_565183


namespace polynomial_solution_l565_565578

theorem polynomial_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, x * P.eval (x - 1) = (x - 2) * P.eval x) →
  ∃ a : ℝ, P = a * Polynomial.C (Polynomial.X ^ 2 - Polynomial.X) :=
by sorry

end polynomial_solution_l565_565578


namespace min_value_of_F_l565_565262

noncomputable def f (a x : ℝ) : ℝ := ∫ t in -a..x, 12 * t + 4 * a

noncomputable def F (a : ℝ) : ℝ := ∫ x in 0..1, f a x + 3 * a^2

theorem min_value_of_F : ∃ a : ℝ, ∀ x : ℝ, F x ≥ 1 ∧ F a = 1 :=
by
  sorry

end min_value_of_F_l565_565262


namespace pentagon_area_l565_565266

open Real

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 950 square units -/
theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25) : 
  ∃ (area : ℝ), area = 950 :=
by {
  sorry
}

end pentagon_area_l565_565266


namespace max_equal_product_l565_565904

theorem max_equal_product (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 20) (h4 : d = 30) (h5 : e = 40) (h6 : f = 60) :
  ∃ S, (a * b * c * d * e * f) * 450 = S^3 ∧ S = 18000 := 
by
  sorry

end max_equal_product_l565_565904


namespace inequality_sum_l565_565760

theorem inequality_sum {n : ℕ} (h1 : 2 ≤ n) (a : ℕ → ℝ) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) :
  let s := ∑ i in finset.range n, a i in
  (∑ i in finset.range n, a i / (s - a i)) ≥ n / (n - 1) := by
  sorry

end inequality_sum_l565_565760


namespace parallelograms_count_l565_565383

def A : (ℝ × ℝ) := (0, 0)

def B (b : ℝ) : (ℝ × ℝ) := (b, b)  -- B is on the line y = x

def D (d : ℝ) : (ℝ × ℝ) := (d, 3/2 * d)  -- D is on the line y = 3/2 * x

def area (b d : ℝ) : ℝ :=
  3/4 * (b * d)

def valid_parallelogram (b d : ℝ) : Prop :=
  b > 0 ∧ d > 0 ∧ b * d = 1728000  -- area condition

def num_valid_parallelograms : ℕ :=
  56  -- correct answer

theorem parallelograms_count : ∃ n, n = 56 ∧ (∀ b d : ℝ, valid_parallelogram b d → True) :
  sorry

end parallelograms_count_l565_565383


namespace basket_white_ball_probability_l565_565208

noncomputable def basket_problem_proof : Prop :=
  let P_A := 1 / 2
  let P_B := 1 / 2
  let P_W_given_A := 2 / 5
  let P_W_given_B := 1 / 4
  let P_W := P_A * P_W_given_A + P_B * P_W_given_B
  let P_A_given_W := (P_A * P_W_given_A) / P_W
  P_A_given_W = 8 / 13

theorem basket_white_ball_probability :
  basket_problem_proof :=
  sorry

end basket_white_ball_probability_l565_565208


namespace second_number_value_l565_565798

theorem second_number_value 
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a / b = 3 / 4)
  (h3 : b / c = 2 / 5) :
  b = 480 / 17 :=
by
  sorry

end second_number_value_l565_565798


namespace proof_l565_565270

-- Define proposition p as negated form: ∀ x < 1, log_3 x ≤ 0
def p : Prop := ∀ x : ℝ, x < 1 → Real.log x / Real.log 3 ≤ 0

-- Define proposition q: ∃ x_0 ∈ ℝ, x_0^2 ≥ 2^x_0
def q : Prop := ∃ x_0 : ℝ, x_0^2 ≥ Real.exp (x_0 * Real.log 2)

-- State we need to prove: p ∨ q
theorem proof : p ∨ q := sorry

end proof_l565_565270


namespace find_b_value_l565_565251

theorem find_b_value (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : b = 8 :=
by
  sorry

end find_b_value_l565_565251


namespace problem1_problem2_l565_565942

-- Definitions of conditions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, 0 < x ∧ x^2 - 2 * Real.exp(1) * Real.log x ≤ m
def q (m : ℝ) : Prop := (2 * m)^2 - 4 > 0

-- Defining the proof problems as Lean theorems
theorem problem1 (m : ℝ) : ¬ (p m ∨ q m) → m ∈ set.Ico (-1 : ℝ) 0 :=
by sorry

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → m ∈ set.Iic 1 ∪ set.Iio (-1) :=
by sorry

end problem1_problem2_l565_565942


namespace q_can_be_true_or_false_l565_565688

theorem q_can_be_true_or_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬p) : q ∨ ¬q :=
by
  sorry

end q_can_be_true_or_false_l565_565688


namespace sum_of_divisors_of_24_is_60_l565_565039

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565039


namespace sum_of_divisors_of_24_l565_565109

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565109


namespace cos_theta_value_l565_565305

noncomputable def a : ℝ × ℝ := (3, -1)
noncomputable def b := ( -1 + 3, 1 - 1) -- Simplifying in same step to directly get (2,0)
def theta : ℝ

theorem cos_theta_value : cos theta = 3 * sqrt 10 / 10 :=
by
  -- Computing magnitudes and the rest to ensure the theorem structure
  let mag_a := real.sqrt (3^2 + (-1)^2)
  let mag_b := real.sqrt (2^2 + 0^2)
  let dot_prod := 3 * 2 + (-1) * 0
  have h1 : mag_a = real.sqrt 10 := sorry
  have h2 : mag_b = 2 := sorry
  have h3 : dot_prod = 6 := sorry
  exact sorry -- Placeholder for proof

end cos_theta_value_l565_565305


namespace proof_problem_l565_565538

theorem proof_problem:
  9^(1 / 2) + log 2 4 = 5 := sorry

end proof_problem_l565_565538


namespace sum_of_divisors_of_24_l565_565100

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565100


namespace rectangle_other_side_l565_565499

theorem rectangle_other_side (A x y : ℝ) (hA : A = 1 / 8) (hx : x = 1 / 2) (hArea : A = x * y) :
    y = 1 / 4 := 
  sorry

end rectangle_other_side_l565_565499


namespace derangements_1_to_4_l565_565905

theorem derangements_1_to_4 : derange 4 = 9 := by
  sorry

end derangements_1_to_4_l565_565905


namespace reduction_in_carbon_emission_l565_565692

theorem reduction_in_carbon_emission 
  (population : ℕ)
  (carbon_per_car : ℕ)
  (carbon_per_bus : ℕ)
  (bus_capacity : ℕ)
  (percentage_take_bus : ℝ)
  (initial_cars : ℕ := population) :
  (initial_cars * carbon_per_car - 
  (population * percentage_take_bus.floor.to_nat * carbon_per_bus +
   (population - population * percentage_take_bus.floor.to_nat) * carbon_per_car)) = 100 :=
by
  -- Given conditions
  let initial_emission := initial_cars * 10 -- Total carbon emission from all cars
  let people_take_bus := (population * 0.25).to_nat -- 25% of people switch to the bus
  let remaining_cars := population - people_take_bus -- People still driving
  let new_emission := 100 + remaining_cars * 10 -- New total emission
  have reduction := initial_emission - new_emission -- Reduction in carbon emission
  -- Correct answer
  exact reduction = 100

end reduction_in_carbon_emission_l565_565692


namespace range_of_quadratic_within_domain_l565_565794

noncomputable def quadratic_func (x : ℝ) : ℝ := x^2 - 2*x 

theorem range_of_quadratic_within_domain : 
  ∀ y, y ∈ set.range (λ x, quadratic_func x) ↔ ∃ x, -1 ≤ x ∧ x ≤ 3 ∧ y = quadratic_func x := by
  sorry

end range_of_quadratic_within_domain_l565_565794


namespace harmonic_mean_heptagon_diagonals_l565_565764

def harmonic_mean (x y : ℝ) := 2 * (x * y) / (x + y)

theorem harmonic_mean_heptagon_diagonals {x y : ℝ} (h1 : 7 * x + 7 * y = 14)
    (h2 : 2 * x * y = x + y) :
    harmonic_mean x y = 2 := by
  sorry

end harmonic_mean_heptagon_diagonals_l565_565764


namespace abs_lt_2_sufficient_not_necessary_l565_565156

theorem abs_lt_2_sufficient_not_necessary (x : ℝ) :
  (|x| < 2 → x^2 - x - 6 < 0) ∧ ¬ (x^2 - x - 6 < 0 → |x| < 2) :=
by {
  sorry
}

end abs_lt_2_sufficient_not_necessary_l565_565156


namespace gcd_8251_6105_l565_565436

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l565_565436


namespace min_ratio_circumradius_inradius_l565_565612

theorem min_ratio_circumradius_inradius (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (triangle_right : right_angled_triangle α) : 
  ∃ α_min, α_min = π / 4 ∧ 
  let R := circumradius (right_angled_triangle α_min)
      r := inradius (right_angled_triangle α_min)
  in (R / r) = sqrt 2 + 1 :=
by 
  sorry

end min_ratio_circumradius_inradius_l565_565612


namespace student_total_marks_l565_565323

theorem student_total_marks (total_questions correct_answers incorrect_mark correct_mark : ℕ) 
                             (H1 : total_questions = 60) 
                             (H2 : correct_answers = 34)
                             (H3 : incorrect_mark = 1)
                             (H4 : correct_mark = 4) :
  ((correct_answers * correct_mark) - ((total_questions - correct_answers) * incorrect_mark)) = 110 := 
by {
  -- The proof goes here.
  sorry
}

end student_total_marks_l565_565323


namespace limit_exists_implies_d_eq_zero_l565_565349

variable (a₁ d : ℝ) (S : ℕ → ℝ)

noncomputable def limExists := ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (S n - L) < ε

def is_sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) :=
  ∀ n : ℕ, S n = (a₁ * n + d * (n * (n - 1) / 2))

theorem limit_exists_implies_d_eq_zero (h₁ : ∀ n : ℕ, n > 0 → S n = (a₁ * n + d * (n * (n - 1) / 2))) :
  limExists S → d = 0 :=
by sorry

end limit_exists_implies_d_eq_zero_l565_565349


namespace system_solution_exists_l565_565238

theorem system_solution_exists (a : ℝ) : 
  ∃ (x y : ℝ), (2 * y - 2 = a * (x - 2)) ∧ (4 * y / (|x| + x) = Real.sqrt y) := sorry

end system_solution_exists_l565_565238


namespace triangles_congruent_l565_565804

/-- 
Given three equal circles that intersect at a common point \( M \), 
with centers \( O_1 \), \( O_2 \), and \( O_3 \), 
prove that the triangle \( ABC \) (where \( A \), \( B \), and \( C \) 
are the remaining points of pairwise intersections) is 
congruent to the triangle \( O_1O_2O_3 \). 
-/

theorem triangles_congruent 
  (O1 O2 O3 A B C M : Point) 
  (r : ℝ) 
  (h1 : distance O1 M = r)
  (h2 : distance O2 M = r)
  (h3 : distance O3 M = r)
  (h4 : distance O1 A = r)
  (h5 : distance O2 A = r)
  (h6 : distance O2 B = r)
  (h7 : distance O3 B = r)
  (h8 : distance O1 C = r)
  (h9 : distance O3 C = r) : 
  triangle.congruent ⟨A, B, C⟩ ⟨O1, O2, O3⟩ := 
sorry

end triangles_congruent_l565_565804


namespace sum_of_divisors_of_twenty_four_l565_565123

theorem sum_of_divisors_of_twenty_four : 
  (∑ n in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_of_divisors_of_twenty_four_l565_565123


namespace sum_divisors_24_l565_565036

theorem sum_divisors_24 :
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range 25)), n) = 60 :=
by
  sorry

end sum_divisors_24_l565_565036


namespace sum_of_divisors_of_24_l565_565112

theorem sum_of_divisors_of_24 : (∑ n in Finset.filter (λ n => 24 ∣ n) (Finset.range (24+1)), n) = 60 :=
by {
  sorry
}

end sum_of_divisors_of_24_l565_565112


namespace geometric_sequence_first_term_l565_565795

theorem geometric_sequence_first_term 
  (a r : ℝ) 
  (h1 : a * r^6 = real.factorial 8) 
  (h2 : a * r^9 = real.factorial 11) : 
  a = 8 / 245 :=
by
  sorry

end geometric_sequence_first_term_l565_565795


namespace selling_price_correct_l565_565824

-- Conditions from the problem
def cost_price : ℝ := 1800
def loss_percentage : ℝ := 10

-- Definitions derived from conditions
def loss_amount : ℝ := (loss_percentage / 100) * cost_price
def selling_price : ℝ := cost_price - loss_amount

-- The theorem to prove that the selling price is 1620
theorem selling_price_correct : selling_price = 1620 :=
by
  sorry

end selling_price_correct_l565_565824


namespace sum_of_divisors_of_24_l565_565096

def is_positive_integer (n : ℕ) : Prop := n > 0

def divides (d n : ℕ) : Prop := n % d = 0

theorem sum_of_divisors_of_24 : 
  ∑ n in (Finset.filter is_positive_integer (Finset.range 25)) (λ n, if divides n 24 then n else 0) = 60 :=
by
  sorry

end sum_of_divisors_of_24_l565_565096


namespace sum_of_divisors_of_24_l565_565070

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 := 
by sorry

end sum_of_divisors_of_24_l565_565070


namespace remaining_painting_time_l565_565493

-- Define the conditions
def total_rooms : ℕ := 10
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 8

-- Define what we want to prove
theorem remaining_painting_time : (total_rooms - rooms_painted) * hours_per_room = 16 :=
by
  -- Here is where you would provide the proof
  sorry

end remaining_painting_time_l565_565493


namespace smallest_omega_l565_565408

theorem smallest_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∃ k : ℤ, (2 / 3) * ω = 2 * k) -> ω = 3 :=
by
  sorry

end smallest_omega_l565_565408


namespace first_term_of_geometric_sequence_l565_565422

noncomputable def factorial : ℕ → ℝ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def find_first_term (a r : ℝ) : Prop :=
  a * r^5 = factorial 6 ∧ a * r^8 = factorial 9 ∧ a ≈ 0.66

theorem first_term_of_geometric_sequence (a r : ℝ) 
  (h₁ : a * r^5 = factorial 6) (h₂ : a * r^8 = factorial 9) : 
  a ≈ 0.66 := 
sorry

end first_term_of_geometric_sequence_l565_565422


namespace polynomial_solution_count_l565_565223

def Q (a b c d : ℕ) : ℤ := (-a : ℤ) + b - c + d

def in_range (n : ℕ) : Prop := n ≤ 9

theorem polynomial_solution_count :
  (∃ (a b c d : ℕ), in_range a ∧ in_range b ∧ in_range c ∧ in_range d ∧ Q a b c d = 1) → 4 :=
by
  sorry

end polynomial_solution_count_l565_565223


namespace subtract_fractions_l565_565464

theorem subtract_fractions (p q : ℚ) (h₁ : 4 / p = 8) (h₂ : 4 / q = 18) : p - q = 5 / 18 := 
by 
  sorry

end subtract_fractions_l565_565464


namespace sum_of_squares_l565_565766

/-- The sum of the squares of the first n natural numbers is equal to (n * (n + 1) * (2 * n + 1)) / 6. -/
theorem sum_of_squares (n : ℕ) : (∑ i in Finset.range (n + 1), i^2) = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end sum_of_squares_l565_565766


namespace modulus_of_z_l565_565292

def i : ℂ := complex.I
def z : ℂ := (10 / (3 + i)) - 2 * i
def modulus_z := complex.abs z

theorem modulus_of_z : modulus_z = real.sqrt 10 :=
by
  sorry

end modulus_of_z_l565_565292


namespace Freddy_travel_time_l565_565570

theorem Freddy_travel_time
  (Eddy_Time : ℝ)
  (Distance_AB : ℝ)
  (Distance_AC : ℝ)
  (Speed_Ratio : ℝ)
  (Freddy_Time : ℝ) :
  Eddy_Time = 3 ∧ Distance_AB = 450 ∧ Distance_AC = 300 ∧ Speed_Ratio = 2 →
  Freddy_Time = 4 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  have Eddy_Speed : ℝ := 150
  have Freddy_Speed : ℝ := Eddy_Speed / h4
  have Freddy_Time_ := Distance_AC / Freddy_Speed
  exact eq.trans Freddy_Time_ 4 sorry

end Freddy_travel_time_l565_565570


namespace find_balls_l565_565523

theorem find_balls (x y : ℕ) (h1 : (x + y : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) - 1 / 15)
                   (h2 : (y + 18 : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) * 11 / 10) :
  x = 12 ∧ y = 15 :=
sorry

end find_balls_l565_565523


namespace eq_system_solution_l565_565391

theorem eq_system_solution (a b c d : ℝ) (h1 : a - sqrt (1 - b^2) + sqrt (1 - c^2) = d)
  (h2 : b - sqrt (1 - c^2) + sqrt (1 - d^2) = a)
  (h3 : c - sqrt (1 - d^2) + sqrt (1 - a^2) = b)
  (h4 : d - sqrt (1 - a^2) + sqrt (1 - b^2) = c)
  (ha : -1 ≤ a) (hb : a ≤ 1)
  (hb1 : -1 ≤ b) (hb2 : b ≤ 1)
  (hc1 : -1 ≤ c) (hc2 : c ≤ 1)
  (hd1 : -1 ≤ d) (hd2 : d ≤ 1) 
  : a = b ∧ a = c ∧ a = d :=
sorry

end eq_system_solution_l565_565391


namespace minimum_red_beads_l565_565490

theorem minimum_red_beads (n : ℕ) (r : ℕ) (necklace : ℕ → Prop) :
  (necklace = λ k, n * k + r) 
  → (∀ i, (segment_contains_blue i 8 → segment_contains_red i 4))
  → (cyclic_beads necklace)
  → r ≥ 29 :=
by
  sorry

-- Definitions to support the theorem
def segment_contains_blue (i : ℕ) (b : ℕ) : Prop := 
sorry -- Placeholder for the predicate that checks if a segment contains exactly 'b' blue beads.

def segment_contains_red (i : ℕ) (r : ℕ) : Prop := 
sorry -- Placeholder for the predicate that checks if a segment contains at least 'r' red beads.

def cyclic_beads (necklace : ℕ → Prop) : Prop := 
sorry -- Placeholder for the property that defines the necklace as cyclic.

end minimum_red_beads_l565_565490


namespace rhombus_circle_area_ratio_l565_565789

theorem rhombus_circle_area_ratio (x : ℝ) (h : 0 < x) :
  let BD := 6 * x,
      AC := 8 * x,
      area_rhombus := (1/2) * AC * BD,
      r := (12 * x) / 5,
      area_circle := π * r^2
  in area_rhombus / area_circle = 25 / (6 * π) :=
by
  sorry

end rhombus_circle_area_ratio_l565_565789


namespace area_of_region_inside_hexagon_outside_sectors_is_correct_l565_565182

noncomputable def area_hexagon_outside_sectors : ℝ :=
  let side_length := 10
  let radius := 5
  let sector_angle := 120
  let hexagon_area := 6 * (sqrt 3 / 4 * side_length ^ 2)
  let sector_area := 1 / 3 * π * radius ^ 2
  let total_sector_area := 6 * sector_area
  hexagon_area - total_sector_area

theorem area_of_region_inside_hexagon_outside_sectors_is_correct :
  area_hexagon_outside_sectors = 150 * sqrt 3 - 50 * π := by
  sorry

end area_of_region_inside_hexagon_outside_sectors_is_correct_l565_565182


namespace exist_congruent_polygons_in_rectangle_exist_congruent_polygons_in_square_l565_565566

/-- There exists 15 congruent polygons that are not rectangles such that they can dissect a given rectangle. -/
theorem exist_congruent_polygons_in_rectangle : 
  ∃ (P : Type) [polygon P], 
    congruent_polygons P 15 ∧ non_rectangular_polygons P ∧ dissects_rectangle P :=
sorry

/-- There exists 15 congruent polygons that are not rectangles such that they can dissect a given square. -/
theorem exist_congruent_polygons_in_square : 
  ∃ (P : Type) [polygon P], 
    congruent_polygons P 15 ∧ non_rectangular_polygons P ∧ dissects_square P :=
sorry

end exist_congruent_polygons_in_rectangle_exist_congruent_polygons_in_square_l565_565566


namespace calculate_operation_result_l565_565928

def operation (a b : ℝ) : ℝ :=
if a ≥ b then (real.sqrt a) - (real.sqrt b) 
else (real.sqrt b) - (real.sqrt a)

theorem calculate_operation_result : operation 9 8 + operation 16 18 = real.sqrt 2 - 1 := by
  sorry

end calculate_operation_result_l565_565928


namespace sum_of_divisors_of_24_is_60_l565_565043

theorem sum_of_divisors_of_24_is_60 :
  ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 :=
by
  sorry

end sum_of_divisors_of_24_is_60_l565_565043


namespace minimum_intersection_percentage_l565_565897

variables {α : Type*} {A B : set α}
variable [fintype α]
variable (n : ℕ)
variable (pop : finset α)

noncomputable def percent_set (s : set α) : ℝ := (finset.card (s.to_finset) : ℝ) / (finset.card pop : ℝ)

theorem minimum_intersection_percentage
    (hA : percent_set A = 0.85)
    (hB : percent_set B = 0.75) :
    percent_set (A ∩ B) ≥ 0.60 :=
sorry

end minimum_intersection_percentage_l565_565897


namespace find_c_k_l565_565419

noncomputable def common_difference (a : ℕ → ℕ) : ℕ := sorry
noncomputable def common_ratio (b : ℕ → ℕ) : ℕ := sorry
noncomputable def arith_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def geom_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
noncomputable def combined_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ := a n + b n

variable (k : ℕ) (d : ℕ) (r : ℕ)

-- Conditions
axiom arith_condition : common_difference (arith_seq d) = d
axiom geom_condition : common_ratio (geom_seq r) = r
axiom combined_k_minus_1 : combined_seq (arith_seq d) (geom_seq r) (k - 1) = 50
axiom combined_k_plus_1 : combined_seq (arith_seq d) (geom_seq r) (k + 1) = 1500

-- Prove that c_k = 2406
theorem find_c_k : combined_seq (arith_seq d) (geom_seq r) k = 2406 := by
  sorry

end find_c_k_l565_565419


namespace relationship_y1_y2_y3_l565_565618

-- Definition of points and function in conditions
variables (k : ℝ) (y1 y2 y3 : ℝ)
hypothesis h_neg_k : k < 0 -- k is less than 0
hypothesis h_A : y1 = k / -4 -- Point A(-4, y1) on the graph
hypothesis h_B : y2 = k / -2 -- Point B(-2, y2) on the graph
hypothesis h_C : y3 = k / 3 -- Point C(3, y3) on the graph

-- Statement showing the relationship between y1, y2, y3
theorem relationship_y1_y2_y3 : y3 < y1 ∧ y1 < y2 :=
by {
  -- Proof will be inserted here
  sorry
}

end relationship_y1_y2_y3_l565_565618


namespace relationship_y1_y2_y3_l565_565619

-- Definition of points and function in conditions
variables (k : ℝ) (y1 y2 y3 : ℝ)
hypothesis h_neg_k : k < 0 -- k is less than 0
hypothesis h_A : y1 = k / -4 -- Point A(-4, y1) on the graph
hypothesis h_B : y2 = k / -2 -- Point B(-2, y2) on the graph
hypothesis h_C : y3 = k / 3 -- Point C(3, y3) on the graph

-- Statement showing the relationship between y1, y2, y3
theorem relationship_y1_y2_y3 : y3 < y1 ∧ y1 < y2 :=
by {
  -- Proof will be inserted here
  sorry
}

end relationship_y1_y2_y3_l565_565619


namespace problem_solution_l565_565869

def equal_group_B : Prop :=
  (-2)^3 = -(2^3)

theorem problem_solution : equal_group_B := by
  sorry

end problem_solution_l565_565869


namespace barbell_cost_l565_565339

-- Define the conditions in Lean.
def num_barbells : ℕ := 3
def given_amount : ℝ := 850
def change_received : ℝ := 40

-- Define the statement to be proved.
theorem barbell_cost :
  ∃ cost_per_barbell : ℝ,
    (given_amount - change_received) / num_barbells = cost_per_barbell ∧
    cost_per_barbell = 270 :=
by
  use (850 - 40) / 3
  simp
  split
  . rfl
  . norm_num

end barbell_cost_l565_565339


namespace sum_of_remainders_l565_565453

theorem sum_of_remainders (n : ℤ) (h : n % 15 = 7) : 
  (n % 3) + (n % 5) = 3 := 
by
  -- the proof will go here
  sorry

end sum_of_remainders_l565_565453


namespace derivative_at_2_l565_565298

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_at_2 : deriv f 2 = Real.sqrt 2 / 4 := by
  sorry

end derivative_at_2_l565_565298


namespace find_x_l565_565922

theorem find_x (x : ℝ) (h1: x > 0) (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * Real.sqrt 3 :=
sorry

end find_x_l565_565922


namespace boys_or_girls_rink_l565_565337

variables (Class : Type) (is_boy : Class → Prop) (is_girl : Class → Prop) (visited_rink : Class → Prop) (met_at_rink : Class → Class → Prop)

-- Every student in the class visited the rink at least once.
axiom all_students_visited : ∀ (s : Class), visited_rink s

-- Every boy met every girl at the rink.
axiom boys_meet_girls : ∀ (b g : Class), is_boy b → is_girl g → met_at_rink b g

-- Prove that there exists a time when all the boys, or all the girls were simultaneously on the rink.
theorem boys_or_girls_rink : ∃ (t : Prop), (∀ b, is_boy b → visited_rink b) ∨ (∀ g, is_girl g → visited_rink g) :=
sorry

end boys_or_girls_rink_l565_565337


namespace derivative_of_y_l565_565580

noncomputable def y (x : ℝ) : ℝ :=
  -1/4 * Real.arcsin ((5 + 3 * Real.cosh x) / (3 + 5 * Real.cosh x))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (3 + 5 * Real.cosh x) :=
sorry

end derivative_of_y_l565_565580


namespace chessboard_piece_arrangements_l565_565435

-- Define the problem in Lean
theorem chessboard_piece_arrangements (black_pos white_pos : ℕ)
  (black_pos_neq_white_pos : black_pos ≠ white_pos)
  (valid_position : black_pos < 64 ∧ white_pos < 64) :
  ¬(∀ (move : ℕ → ℕ → Prop), (move black_pos white_pos) → ∃! (p : ℕ × ℕ), move (p.fst) (p.snd)) :=
by sorry

end chessboard_piece_arrangements_l565_565435


namespace no_n_makes_g_multiple_of_5_and_7_l565_565590

def g (n : ℕ) : ℕ := 4 + 2 * n + 3 * n^2 + n^3 + 4 * n^4 + 3 * n^5

theorem no_n_makes_g_multiple_of_5_and_7 :
  ¬ ∃ n, (2 ≤ n ∧ n ≤ 100) ∧ (g n % 5 = 0 ∧ g n % 7 = 0) :=
by
  -- Proof goes here
  sorry

end no_n_makes_g_multiple_of_5_and_7_l565_565590


namespace initial_markup_percentage_l565_565495

variable (C : ℝ) -- Cost price of the turtleneck sweaters
variable (M : ℝ) -- Initial markup percentage (in decimal form)

-- Selling price after initial markup
def S1 := C * (1 + M)

-- Selling price after New Year's markup
def S2 := S1 * 1.25

-- Selling price after February discount
def S3 := S2 * 0.93

-- Final selling price representing 39.5% profit
def final_selling_price := C * 1.395

theorem initial_markup_percentage :
  ∀ (C : ℝ) (M : ℝ),
    C > 0 →
    M > 0 →
    S3 C M = final_selling_price C →
    M = 0.20 :=
by
  intro C M hC hM hS3eqfinal
  sorry

end initial_markup_percentage_l565_565495


namespace cupboard_cost_price_l565_565465

theorem cupboard_cost_price (C SP NSP : ℝ) (h1 : SP = 0.84 * C) (h2 : NSP = 1.16 * C) (h3 : NSP = SP + 1200) : C = 3750 :=
by
  sorry

end cupboard_cost_price_l565_565465


namespace resistance_between_C_and_G_l565_565833

noncomputable def resistance_structure : ℝ :=
  let R₀ := 1 -- Resistance of segment AB
  let R_total := 2 * R₀ -- Total resistance due to symmetric property
  R_total -- The resulting resistance

theorem resistance_between_C_and_G (R₀ : ℝ) (hR₀ : R₀ = 1) : resistance_structure = 2 :=
by
  rw [resistance_structure]
  exact hR₀
  sorry

end resistance_between_C_and_G_l565_565833


namespace certain_number_exists_l565_565310

theorem certain_number_exists :
  ∃ x : ℕ, ∃ k : ℕ, (15 ^ k ∣ 823435) ∧ (x ^ k - k ^ 5 = 1) :=
by {
  use [2, 1],
  split,
  { exact dvd_of_factors_sublist (factorization_sublist_factorization_of_is_prime 15 823435 (by norm_num)) },
  { norm_num }
}

end certain_number_exists_l565_565310


namespace chord_length_y_intercept_line_tangent_l565_565263

def CircleRadius : ℝ := sqrt 2

def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 = 4

def LineL1 (x y : ℝ) : Prop := x - y - 2 * sqrt 2 = 0

def LineL2 (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

def PointG : ℝ × ℝ := (1, 3)

theorem chord_length : ∀ (A B : ℝ × ℝ), 
  CircleEquation A.1 A.2 → CircleEquation B.1 B.2 → 
  LineL2 A.1 A.2 → LineL2 B.1 B.2 → 
  dist (0, 0) (0, 0) = CircleRadius → 
  2 * sqrt (2^2 - 1^2) = 2 * sqrt 3 := 
sorry

theorem y_intercept : ∀ (b : ℝ), 
  LineL1 (0, b) 0 → 
  (-2 * b)^2 - 8 * (b^2 - 4) > 0 → 
  b = 2 ∨ b = -2 := 
sorry 

theorem line_tangent : ∀ (M N : ℝ × ℝ),
  CircleEquation M.1 M.2 → CircleEquation N.1 N.2 → 
  dist (0, 0) PointG = 10 → 
  x + 3 * y - 4 = 0 := 
sorry

end chord_length_y_intercept_line_tangent_l565_565263


namespace ellipse_properties_l565_565283

noncomputable def ellipse_eq := ∀ (x y : ℝ), (x^2) / 2 + y^2 = 1

noncomputable def area_max (m : ℝ) : Prop :=
  ∀ (M N : ℝ × ℝ),
  M.y = 2 * M.x + m ∧ N.y = 2 * N.x + m ∧
  ((M.x^2 / 2 + M.y^2 = 1) ∧ (N.x^2 / 2 + N.y^2 = 1)) →
  let 
    MN_dist := (M.x - N.x)^2 + (M.y - N.y)^2
    d := abs m / sqrt 5
  in
    (1 / 2) * sqrt MN_dist * d ≤ sqrt 2 / 2

theorem ellipse_properties (m : ℝ) :
  let 
    a := sqrt 2
    c := 1
  in
    ellipse_eq ∧
    ∀ (m < 3), area_max m :=
sorry

end ellipse_properties_l565_565283


namespace sum_of_divisors_of_24_l565_565024

theorem sum_of_divisors_of_24 :
  (∑ n in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), n) = 60 :=
sorry

end sum_of_divisors_of_24_l565_565024


namespace triangle_with_small_area_l565_565836

open Set

noncomputable def exists_triangle_with_area_leq_eighth (points : Fin 9 → Point) : Prop :=
  ∃ (T : Triangle), (area T ≤ 1 / 8) ∧ (T.v₁ ∈ points) ∧ (T.v₂ ∈ points) ∧ (T.v₃ ∈ points)

theorem triangle_with_small_area
  (points : Fin 9 → Point)
  (h : ∀ i, points i ∈ unit_square) :
  exists_triangle_with_area_leq_eighth points :=
sorry

end triangle_with_small_area_l565_565836


namespace find_x_l565_565138

theorem find_x (x y : ℝ) (h₁ : 2 * x - y = 14) (h₂ : y = 2) : x = 8 :=
by
  sorry

end find_x_l565_565138


namespace symmetric_line_eq_l565_565784

theorem symmetric_line_eq :
  ∀ (x y : ℝ), (2 * x - 3 * y + 1 = 0) → (3 * y - 2 * x + 1 = 0) :=
begin
  sorry
end

end symmetric_line_eq_l565_565784


namespace max_diff_sum_balls_l565_565319

-- Definitions for the problem setup
def balls := {x : ℕ | 101 ≤ x ∧ x ≤ 300}
def person_A := {x : ℕ | 201 ≤ x ∧ x ≤ 300} - {280} ∪ {102}
def person_B := {x : ℕ | 101 ≤ x ∧ x ≤ 200} - {102} ∪ {280}

-- Lean statement
theorem max_diff_sum_balls :
  (finset.sum (finset.filter (λ x, x ∈ person_A) (finset.range 301)))
  - (finset.sum (finset.filter (λ x, x ∈ person_B) (finset.range 301))) = 9644 :=
by
  sorry

end max_diff_sum_balls_l565_565319


namespace non_coincident_planes_l565_565304

def condition_1 (α β γ : Plane) : Prop :=
  α ∥ γ ∧ β ∥ γ

def condition_2 (α β γ : Plane) : Prop :=
  α ⊥ γ ∧ β ⊥ γ

def condition_3 (α β : Plane) : Prop :=
  ∃ A B C : Point, 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    A ∈ α ∧ B ∈ α ∧ C ∈ α ∧
    ∃ d, d > 0 ∧ dist_point_plane A β = d ∧ dist_point_plane B β = d ∧ dist_point_plane C β = d

def condition_4 (α β : Plane) (l m : Line) : Prop :=
  l ∥ α ∧ l ∥ β ∧ m ∥ α ∧ m ∥ β ∧ skew l m

theorem non_coincident_planes (α β : Plane) : 
  (count_valid_conditions α β) = 3 := 
sorry

end non_coincident_planes_l565_565304
