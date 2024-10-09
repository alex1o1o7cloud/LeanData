import Mathlib

namespace algebraic_notation_3m_minus_n_squared_l1570_157085

theorem algebraic_notation_3m_minus_n_squared (m n : ℝ) : 
  (3 * m - n)^2 = (3 * m - n) ^ 2 :=
by sorry

end algebraic_notation_3m_minus_n_squared_l1570_157085


namespace money_left_l1570_157069

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end money_left_l1570_157069


namespace complex_division_l1570_157091

def imaginary_unit := Complex.I

theorem complex_division :
  (1 - 3 * imaginary_unit) / (2 + imaginary_unit) = -1 / 5 - 7 / 5 * imaginary_unit := by
  sorry

end complex_division_l1570_157091


namespace problem1_problem2_l1570_157079

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end problem1_problem2_l1570_157079


namespace sum_of_faces_edges_vertices_l1570_157046

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l1570_157046


namespace algebraic_expression_eq_five_l1570_157032

theorem algebraic_expression_eq_five (a b : ℝ)
  (h₁ : a^2 - a = 1)
  (h₂ : b^2 - b = 1) :
  3 * a^2 + 2 * b^2 - 3 * a - 2 * b = 5 :=
by
  sorry

end algebraic_expression_eq_five_l1570_157032


namespace find_k_l1570_157021

theorem find_k (k : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -1 / 2 :=
by 
  sorry

end find_k_l1570_157021


namespace total_volume_of_four_cubes_is_500_l1570_157002

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l1570_157002


namespace gcf_and_multiples_l1570_157018

theorem gcf_and_multiples (a b gcf : ℕ) : 
  (a = 90) → (b = 135) → gcd a b = gcf → 
  (gcf = 45) ∧ (45 % gcf = 0) ∧ (90 % gcf = 0) ∧ (135 % gcf = 0) := 
by
  intros ha hb hgcf
  rw [ha, hb] at hgcf
  sorry

end gcf_and_multiples_l1570_157018


namespace combined_weight_loss_l1570_157024

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end combined_weight_loss_l1570_157024


namespace final_center_coordinates_l1570_157099

-- Definition of the initial condition: the center of Circle U
def center_initial : ℝ × ℝ := (3, -4)

-- Definition of the reflection function across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Definition of the translation function to translate a point 5 units up
def translate_up_5 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 5)

-- Defining the final coordinates after reflection and translation
def center_final : ℝ × ℝ :=
  translate_up_5 (reflect_y_axis center_initial)

-- Problem statement: Prove that the final center coordinates are (-3, 1)
theorem final_center_coordinates :
  center_final = (-3, 1) :=
by {
  -- Skipping the proof itself, but the theorem statement should be equivalent
  sorry
}

end final_center_coordinates_l1570_157099


namespace triangle_inequality_l1570_157059

theorem triangle_inequality
  (A B C : ℝ)
  (hA : 0 < A)
  (hB : 0 < B)
  (hC : 0 < C)
  (hABC : A + B + C = Real.pi) :
  Real.sin (3 * A / 2) + Real.sin (3 * B / 2) + Real.sin (3 * C / 2) ≤
  Real.cos ((A - B) / 2) + Real.cos ((B - C) / 2) + Real.cos ((C - A) / 2) :=
by
  sorry

end triangle_inequality_l1570_157059


namespace reciprocal_of_2016_is_1_div_2016_l1570_157096

theorem reciprocal_of_2016_is_1_div_2016 : (2016 * (1 / 2016) = 1) :=
by
  sorry

end reciprocal_of_2016_is_1_div_2016_l1570_157096


namespace family_reunion_kids_l1570_157065

theorem family_reunion_kids (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h_adults : adults = 123) (h_tables : tables = 14) 
  (h_people_per_table : people_per_table = 12) :
  (tables * people_per_table - adults) = 45 :=
by
  sorry

end family_reunion_kids_l1570_157065


namespace units_digit_quotient_l1570_157077

theorem units_digit_quotient (n : ℕ) :
  (2^1993 + 3^1993) % 5 = 0 →
  ((2^1993 + 3^1993) / 5) % 10 = 3 := by
  sorry

end units_digit_quotient_l1570_157077


namespace voter_ratio_l1570_157043

theorem voter_ratio (Vx Vy : ℝ) (hx : 0.72 * Vx + 0.36 * Vy = 0.60 * (Vx + Vy)) : Vx = 2 * Vy :=
by
sorry

end voter_ratio_l1570_157043


namespace difference_in_average_speed_l1570_157009

theorem difference_in_average_speed 
  (distance : ℕ) 
  (time_diff : ℕ) 
  (speed_B : ℕ) 
  (time_B : ℕ) 
  (time_A : ℕ) 
  (speed_A : ℕ)
  (h1 : distance = 300)
  (h2 : time_diff = 3)
  (h3 : speed_B = 20)
  (h4 : time_B = distance / speed_B)
  (h5 : time_A = time_B - time_diff)
  (h6 : speed_A = distance / time_A) 
  : speed_A - speed_B = 5 := 
sorry

end difference_in_average_speed_l1570_157009


namespace mismatching_socks_l1570_157076

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end mismatching_socks_l1570_157076


namespace sufficient_but_not_necessary_condition_l1570_157033

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, q x a → p x) ∧ (∃ x, ¬q x a ∧ p x) → a ∈ Set.Ici 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l1570_157033


namespace race_length_l1570_157072

noncomputable def solve_race_length (a b c d : ℝ) : Prop :=
  (d > 0) →
  (d / a = (d - 40) / b) →
  (d / b = (d - 30) / c) →
  (d / a = (d - 65) / c) →
  d = 240

theorem race_length : ∃ (d : ℝ), solve_race_length a b c d :=
by
  use 240
  sorry

end race_length_l1570_157072


namespace log_expression_value_l1570_157082

theorem log_expression_value
  (h₁ : x + (Real.log 32 / Real.log 8) = 1.6666666666666667)
  (h₂ : Real.log 32 / Real.log 8 = 1.6666666666666667) :
  x = 0 :=
by
  sorry

end log_expression_value_l1570_157082


namespace intersection_result_l1570_157061

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x : ℝ | abs (x - 1) > 2 }

-- Define set B
def B : Set ℝ := { x : ℝ | -x^2 + 6 * x - 8 > 0 }

-- Define the complement of A in U
def compl_A : Set ℝ := U \ A

-- Define the intersection of compl_A and B
def inter_complA_B : Set ℝ := compl_A ∩ B

-- Prove that the intersection is equal to the given set
theorem intersection_result : inter_complA_B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_result_l1570_157061


namespace jerry_pool_time_l1570_157035

variables (J : ℕ) -- Denote the time Jerry was in the pool

-- Conditions
def Elaine_time := 2 * J -- Elaine stayed in the pool for twice as long as Jerry
def George_time := (2 / 3) * J -- George could only stay in the pool for one-third as long as Elaine
def Kramer_time := 0 -- Kramer did not find the pool

-- Combined total time
def total_time : ℕ := J + Elaine_time J + George_time J + Kramer_time

-- Theorem stating that J = 3 given the combined total time of 11 minutes
theorem jerry_pool_time (h : total_time J = 11) : J = 3 :=
by
  sorry

end jerry_pool_time_l1570_157035


namespace smallest_k_l1570_157051

theorem smallest_k (k: ℕ) : k > 1 ∧ (k % 23 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) → k = 484 :=
sorry

end smallest_k_l1570_157051


namespace team_total_points_l1570_157030

theorem team_total_points (T : ℕ) (h1 : ∃ x : ℕ, x = T / 6)
    (h2 : (T + (92 - 85)) / 6 = 84) : T = 497 := 
by sorry

end team_total_points_l1570_157030


namespace exam_cutoff_mark_l1570_157000

theorem exam_cutoff_mark
  (num_students : ℕ)
  (absent_percentage : ℝ)
  (fail_percentage : ℝ)
  (fail_mark_diff : ℝ)
  (just_pass_percentage : ℝ)
  (remaining_avg_mark : ℝ)
  (class_avg_mark : ℝ)
  (absent_students : ℕ)
  (fail_students : ℕ)
  (just_pass_students : ℕ)
  (remaining_students : ℕ)
  (total_marks : ℝ)
  (P : ℝ) :
  absent_percentage = 0.2 →
  fail_percentage = 0.3 →
  fail_mark_diff = 20 →
  just_pass_percentage = 0.1 →
  remaining_avg_mark = 65 →
  class_avg_mark = 36 →
  absent_students = (num_students * absent_percentage) →
  fail_students = (num_students * fail_percentage) →
  just_pass_students = (num_students * just_pass_percentage) →
  remaining_students = num_students - absent_students - fail_students - just_pass_students →
  total_marks = (absent_students * 0) + (fail_students * (P - fail_mark_diff)) + (just_pass_students * P) + (remaining_students * remaining_avg_mark) →
  class_avg_mark = total_marks / num_students →
  P = 40 :=
by
  intros
  sorry

end exam_cutoff_mark_l1570_157000


namespace smallest_bob_number_l1570_157041

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ n }

def alice_number := 36
def bob_number (m : ℕ) : Prop := prime_factors alice_number ⊆ prime_factors m

-- Proof problem statement
theorem smallest_bob_number :
  ∃ m, bob_number m ∧ m = 6 :=
sorry

end smallest_bob_number_l1570_157041


namespace f_neg_l1570_157037

-- Define the function f and its properties
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else sorry

-- Define the property of f being an odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of f for non-negative x
axiom f_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

-- The theorem to be proven
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := by
  sorry

end f_neg_l1570_157037


namespace ratio_songs_kept_to_deleted_l1570_157064

theorem ratio_songs_kept_to_deleted (initial_songs deleted_songs kept_songs : ℕ) 
  (h_initial : initial_songs = 54) (h_deleted : deleted_songs = 9) (h_kept : kept_songs = initial_songs - deleted_songs) :
  (kept_songs : ℚ) / (deleted_songs : ℚ) = 5 / 1 :=
by
  sorry

end ratio_songs_kept_to_deleted_l1570_157064


namespace speed_of_student_B_l1570_157013

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l1570_157013


namespace melissa_games_l1570_157086

noncomputable def total_points_scored := 91
noncomputable def points_per_game := 7
noncomputable def number_of_games_played := total_points_scored / points_per_game

theorem melissa_games : number_of_games_played = 13 :=
by 
  sorry

end melissa_games_l1570_157086


namespace cost_of_7_enchiladas_and_6_tacos_l1570_157071

theorem cost_of_7_enchiladas_and_6_tacos (e t : ℝ) 
  (h₁ : 4 * e + 5 * t = 5.00) 
  (h₂ : 6 * e + 3 * t = 5.40) : 
  7 * e + 6 * t = 7.47 := 
sorry

end cost_of_7_enchiladas_and_6_tacos_l1570_157071


namespace power_function_is_odd_l1570_157090

theorem power_function_is_odd (m : ℝ) (x : ℝ) (h : (m^2 - m - 1) * (-x)^m = -(m^2 - m - 1) * x^m) : m = -1 :=
sorry

end power_function_is_odd_l1570_157090


namespace smallest_y_of_arithmetic_sequence_l1570_157048

theorem smallest_y_of_arithmetic_sequence
  (x y z d : ℝ)
  (h_arith_series_x : x = y - d)
  (h_arith_series_z : z = y + d)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_product : x * y * z = 216) : y = 6 :=
sorry

end smallest_y_of_arithmetic_sequence_l1570_157048


namespace total_books_l1570_157068

def sam_books := 110
def joan_books := 102
def tom_books := 125
def alice_books := 97

theorem total_books : sam_books + joan_books + tom_books + alice_books = 434 :=
by
  sorry

end total_books_l1570_157068


namespace min_points_dodecahedron_min_points_icosahedron_l1570_157040

-- Definitions for the dodecahedron
def dodecahedron_faces : ℕ := 12
def vertices_per_face_dodecahedron : ℕ := 3

-- Prove the minimum number of points to mark each face of a dodecahedron
theorem min_points_dodecahedron (n : ℕ) (h : 3 * n >= dodecahedron_faces) : n >= 4 :=
sorry

-- Definitions for the icosahedron
def icosahedron_faces : ℕ := 20
def icosahedron_vertices : ℕ := 12

-- Prove the minimum number of points to mark each face of an icosahedron
theorem min_points_icosahedron (n : ℕ) (h : n >= 6) : n = 6 :=
sorry

end min_points_dodecahedron_min_points_icosahedron_l1570_157040


namespace quadratic_function_min_value_l1570_157083

theorem quadratic_function_min_value :
  ∃ x, ∀ y, 5 * x^2 - 15 * x + 2 ≤ 5 * y^2 - 15 * y + 2 ∧ (5 * x^2 - 15 * x + 2 = -9.25) :=
by
  sorry

end quadratic_function_min_value_l1570_157083


namespace find_multiplication_value_l1570_157050

-- Define the given conditions
def student_chosen_number : ℤ := 63
def subtracted_value : ℤ := 142
def result_after_subtraction : ℤ := 110

-- Define the value he multiplied the number by
def multiplication_value (x : ℤ) : Prop := 
  (student_chosen_number * x) - subtracted_value = result_after_subtraction

-- Statement to prove that the value he multiplied the number by is 4
theorem find_multiplication_value : 
  ∃ x : ℤ, multiplication_value x ∧ x = 4 :=
by 
  -- Placeholder for the actual proof
  sorry

end find_multiplication_value_l1570_157050


namespace calculate_heartsuit_ratio_l1570_157095

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem calculate_heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by sorry

end calculate_heartsuit_ratio_l1570_157095


namespace smallest_number_when_diminished_by_7_is_divisible_l1570_157047

-- Variables for divisors
def divisor1 : Nat := 12
def divisor2 : Nat := 16
def divisor3 : Nat := 18
def divisor4 : Nat := 21
def divisor5 : Nat := 28

-- The smallest number x which, when diminished by 7, is divisible by the divisors.
theorem smallest_number_when_diminished_by_7_is_divisible (x : Nat) : 
  (x - 7) % divisor1 = 0 ∧ 
  (x - 7) % divisor2 = 0 ∧ 
  (x - 7) % divisor3 = 0 ∧ 
  (x - 7) % divisor4 = 0 ∧ 
  (x - 7) % divisor5 = 0 → 
  x = 1015 := 
sorry

end smallest_number_when_diminished_by_7_is_divisible_l1570_157047


namespace probability_defective_is_three_tenths_l1570_157056

open Classical

noncomputable def probability_of_defective_product (total_products defective_products: ℕ) : ℝ :=
  (defective_products * 1.0) / (total_products * 1.0)

theorem probability_defective_is_three_tenths :
  probability_of_defective_product 10 3 = 3 / 10 := by
  sorry

end probability_defective_is_three_tenths_l1570_157056


namespace division_yields_square_l1570_157020

theorem division_yields_square (a b : ℕ) (hab : ab + 1 ∣ a^2 + b^2) :
  ∃ m : ℕ, m^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end division_yields_square_l1570_157020


namespace toll_for_18_wheel_truck_l1570_157045

-- Definitions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def rear_axle_wheels_per_axle : ℕ := 4
def toll_formula (x : ℕ) : ℝ := 0.50 + 0.50 * (x - 2)

-- Theorem statement
theorem toll_for_18_wheel_truck : 
  ∃ t : ℝ, t = 2.00 ∧
  ∃ x : ℕ, x = (1 + ((total_wheels - front_axle_wheels) / rear_axle_wheels_per_axle)) ∧
  t = toll_formula x := 
by
  -- Proof to be provided
  sorry

end toll_for_18_wheel_truck_l1570_157045


namespace lowest_possible_students_l1570_157015

-- Definitions based on conditions
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

def canBeDividedIntoTeams (num_students num_teams : ℕ) : Prop := isDivisibleBy num_students num_teams

-- Theorem statement for the lowest possible number of students
theorem lowest_possible_students (n : ℕ) : 
  (canBeDividedIntoTeams n 8) ∧ (canBeDividedIntoTeams n 12) → n = 24 := by
  sorry

end lowest_possible_students_l1570_157015


namespace expression_equals_one_l1570_157081

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l1570_157081


namespace cos_third_quadrant_l1570_157005

theorem cos_third_quadrant (B : ℝ) (hB : -π < B ∧ B < -π / 2) (sin_B : Real.sin B = 5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l1570_157005


namespace no_preimage_for_p_gt_1_l1570_157074

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem no_preimage_for_p_gt_1 (P : ℝ) (hP : P > 1) : ¬ ∃ x : ℝ, f x = P :=
sorry

end no_preimage_for_p_gt_1_l1570_157074


namespace geometric_sequence_common_ratio_l1570_157026

theorem geometric_sequence_common_ratio (a_1 q : ℝ) 
  (h1 : a_1 * q^2 = 9) 
  (h2 : a_1 * (1 + q) + 9 = 27) : 
  q = 1 ∨ q = -1/2 := 
by
  sorry

end geometric_sequence_common_ratio_l1570_157026


namespace minimum_value_expression_l1570_157010

theorem minimum_value_expression 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5) 
  (h_cond : x1^3 + x2^3 + x3^3 + x4^3 + x5^3 = 1) : 
  ∃ y, y = (3 * Real.sqrt 3) / 2 ∧ 
  (y = (x1 / (1 - x1^2) + x2 / (1 - x2^2) + x3 / (1 - x3^2) + x4 / (1 - x4^2) + x5 / (1 - x5^2))) :=
sorry

end minimum_value_expression_l1570_157010


namespace spherical_to_rectangular_coordinates_l1570_157023

-- Define the given conditions
variable (ρ : ℝ) (θ : ℝ) (φ : ℝ)
variable (hρ : ρ = 6) (hθ : θ = 7 * Real.pi / 4) (hφ : φ = Real.pi / 2)

-- Convert spherical coordinates (ρ, θ, φ) to rectangular coordinates (x, y, z) and prove the values
theorem spherical_to_rectangular_coordinates :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2 ∧ z = 0 :=
by
  sorry

end spherical_to_rectangular_coordinates_l1570_157023


namespace restaurant_customer_problem_l1570_157054

theorem restaurant_customer_problem (x y z : ℕ) 
  (h1 : x = 2 * z)
  (h2 : y = x - 3)
  (h3 : 3 + x + y - z = 8) :
  x = 6 ∧ y = 3 ∧ z = 3 ∧ (x + y = 9) :=
by
  sorry

end restaurant_customer_problem_l1570_157054


namespace division_remainder_l1570_157094

-- Define the conditions
def dividend : ℝ := 9087.42
def divisor : ℝ := 417.35
def quotient : ℝ := 21

-- Define the expected remainder
def expected_remainder : ℝ := 323.07

-- Statement of the problem
theorem division_remainder : dividend - divisor * quotient = expected_remainder :=
by
  sorry

end division_remainder_l1570_157094


namespace compute_H_five_times_l1570_157038

def H (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem compute_H_five_times : H (H (H (H (H 2)))) = -1 := by
  sorry

end compute_H_five_times_l1570_157038


namespace y_coordinate_of_point_on_line_l1570_157001

theorem y_coordinate_of_point_on_line (x y : ℝ) (h1 : -4 = x) (h2 : ∃ m b : ℝ, y = m * x + b ∧ y = 3 ∧ x = 10 ∧ m * 4 + b = 0) : y = -4 :=
sorry

end y_coordinate_of_point_on_line_l1570_157001


namespace correct_value_l1570_157057

theorem correct_value (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 5/4 :=
sorry

end correct_value_l1570_157057


namespace divides_343_l1570_157029

theorem divides_343 
  (x y z : ℕ) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : 7 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y)) :
  343 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y) :=
by sorry

end divides_343_l1570_157029


namespace least_possible_area_l1570_157028

variable (x y : ℝ) (n : ℤ)

-- Conditions
def is_integer (x : ℝ) := ∃ k : ℤ, x = k
def is_half_integer (y : ℝ) := ∃ n : ℤ, y = n + 0.5

-- Problem statement in Lean 4
theorem least_possible_area (h1 : is_integer x) (h2 : is_half_integer y)
(h3 : 2 * (x + y) = 150) : ∃ A, A = 0 :=
sorry

end least_possible_area_l1570_157028


namespace amount_of_CaCO3_required_l1570_157027

-- Define the balanced chemical reaction
def balanced_reaction (CaCO3 HCl CaCl2 CO2 H2O : ℕ) : Prop :=
  CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O

-- Define the required conditions
def conditions (HCl_req CaCl2_req CO2_req H2O_req : ℕ) : Prop :=
  HCl_req = 4 ∧ CaCl2_req = 2 ∧ CO2_req = 2 ∧ H2O_req = 2

-- The main theorem to be proved
theorem amount_of_CaCO3_required :
  ∃ (CaCO3_req : ℕ), conditions 4 2 2 2 ∧ balanced_reaction CaCO3_req 4 2 2 2 ∧ CaCO3_req = 2 :=
by 
  sorry

end amount_of_CaCO3_required_l1570_157027


namespace minuend_calculation_l1570_157042

theorem minuend_calculation (subtrahend difference : ℕ) (h : subtrahend + difference + 300 = 600) :
  300 = 300 :=
sorry

end minuend_calculation_l1570_157042


namespace correct_option_D_l1570_157060

theorem correct_option_D (y : ℝ): 
  3 * y^2 - 2 * y^2 = y^2 :=
by
  sorry

end correct_option_D_l1570_157060


namespace original_price_after_discount_l1570_157044

theorem original_price_after_discount (a x : ℝ) (h : 0.7 * x = a) : x = (10 / 7) * a := 
sorry

end original_price_after_discount_l1570_157044


namespace cost_price_of_book_l1570_157006

theorem cost_price_of_book
  (C : ℝ)
  (h : 1.09 * C - 0.91 * C = 9) :
  C = 50 :=
sorry

end cost_price_of_book_l1570_157006


namespace find_m_values_l1570_157075

noncomputable def lines_cannot_form_triangle (m : ℝ) : Prop :=
  (4 * m - 1 = 0) ∨ (6 * m + 1 = 0) ∨ (m^2 + m / 3 - 2 / 3 = 0)

theorem find_m_values :
  { m : ℝ | lines_cannot_form_triangle m } = {4, -1 / 6, -1, 2 / 3} :=
by
  sorry

end find_m_values_l1570_157075


namespace jake_third_test_marks_l1570_157088

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end jake_third_test_marks_l1570_157088


namespace find_number_of_partners_l1570_157052

noncomputable def law_firm_partners (P A : ℕ) : Prop :=
  (P / A = 3 / 97) ∧ (P / (A + 130) = 1 / 58)

theorem find_number_of_partners (P A : ℕ) (h : law_firm_partners P A) : P = 5 :=
  sorry

end find_number_of_partners_l1570_157052


namespace range_of_a_bisection_method_solution_l1570_157093

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a) ∧ (a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

theorem bisection_method_solution (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f (32 / 17) x = 0) :
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ (|f (32 / 17) x| < 0.1) :=
sorry

end range_of_a_bisection_method_solution_l1570_157093


namespace percentage_reduction_in_price_l1570_157011

noncomputable def original_price_per_mango : ℝ := 416.67 / 125

noncomputable def original_num_mangoes : ℝ := 360 / original_price_per_mango

def additional_mangoes : ℝ := 12

noncomputable def new_num_mangoes : ℝ := original_num_mangoes + additional_mangoes

noncomputable def new_price_per_mango : ℝ := 360 / new_num_mangoes

noncomputable def percentage_reduction : ℝ := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100

theorem percentage_reduction_in_price : percentage_reduction = 10 := by
  sorry

end percentage_reduction_in_price_l1570_157011


namespace arithmetic_seq_75th_term_difference_l1570_157036

theorem arithmetic_seq_75th_term_difference :
  ∃ (d : ℝ), 300 * (50 + d) = 15000 ∧ -30 / 299 ≤ d ∧ d ≤ 30 / 299 ∧
  let L := 50 - 225 * (30 / 299)
  let G := 50 + 225 * (30 / 299)
  G - L = 13500 / 299 := by
sorry

end arithmetic_seq_75th_term_difference_l1570_157036


namespace determine_cubic_coeffs_l1570_157053

-- Define the cubic function f(x)
def cubic_function (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define the expression f(f(x) + x)
def composition_expression (a b c x : ℝ) : ℝ :=
  cubic_function a b c (cubic_function a b c x + x)

-- Given that the fraction of the compositions equals the given polynomial
def given_fraction_equals_polynomial (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (composition_expression a b c x) / (cubic_function a b c x) = x^3 + 2023 * x^2 + 1776 * x + 2010

-- Prove that this implies specific values of a, b, and c
theorem determine_cubic_coeffs (a b c : ℝ) :
  given_fraction_equals_polynomial a b c →
  (a = 2022 ∧ b = 1776 ∧ c = 2010) :=
by
  sorry

end determine_cubic_coeffs_l1570_157053


namespace most_efficient_packing_l1570_157016

theorem most_efficient_packing :
  ∃ box_size, 
  (box_size = 3 ∨ box_size = 6 ∨ box_size = 9) ∧ 
  (∀ q ∈ [21, 18, 15, 12, 9], q % box_size = 0) ∧
  box_size = 3 :=
by
  sorry

end most_efficient_packing_l1570_157016


namespace cost_of_candy_l1570_157004

theorem cost_of_candy (initial_amount pencil_cost remaining_after_candy : ℕ) 
  (h1 : initial_amount = 43) 
  (h2 : pencil_cost = 20) 
  (h3 : remaining_after_candy = 18) :
  ∃ candy_cost : ℕ, candy_cost = initial_amount - pencil_cost - remaining_after_candy :=
by
  sorry

end cost_of_candy_l1570_157004


namespace inequality_proof_l1570_157092

variable (x y z : ℝ)

theorem inequality_proof (h : x + y + z = x * y + y * z + z * x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 :=
sorry

end inequality_proof_l1570_157092


namespace reinforcement_left_after_days_l1570_157017

theorem reinforcement_left_after_days
  (initial_men : ℕ) (initial_days : ℕ) (remaining_days : ℕ) (men_left : ℕ)
  (remaining_men : ℕ) (x : ℕ) :
  initial_men = 400 ∧
  initial_days = 31 ∧
  remaining_days = 8 ∧
  men_left = initial_men - remaining_men ∧
  remaining_men = 200 ∧
  400 * 31 - 400 * x = 200 * 8 →
  x = 27 :=
by
  intros h
  sorry

end reinforcement_left_after_days_l1570_157017


namespace dogs_on_mon_wed_fri_l1570_157022

def dogs_on_tuesday : ℕ := 12
def dogs_on_thursday : ℕ := 9
def pay_per_dog : ℕ := 5
def total_earnings : ℕ := 210

theorem dogs_on_mon_wed_fri :
  ∃ (d : ℕ), d = 21 ∧ d * pay_per_dog = total_earnings - (dogs_on_tuesday + dogs_on_thursday) * pay_per_dog :=
by 
  sorry

end dogs_on_mon_wed_fri_l1570_157022


namespace infinite_squares_of_form_l1570_157070

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end infinite_squares_of_form_l1570_157070


namespace width_of_rectangular_prism_l1570_157007

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end width_of_rectangular_prism_l1570_157007


namespace students_count_inconsistent_l1570_157039

-- Define the conditions
variables (total_students boys_more_than_girls : ℤ)

-- Define the main theorem: The computed number of girls is not an integer
theorem students_count_inconsistent 
  (h1 : total_students = 3688) 
  (h2 : boys_more_than_girls = 373) 
  : ¬ ∃ x : ℤ, 2 * x + boys_more_than_girls = total_students := 
by
  sorry

end students_count_inconsistent_l1570_157039


namespace sticks_at_20_l1570_157073

-- Define the sequence of sticks used at each stage
def sticks (n : ℕ) : ℕ :=
  if n = 1 then 5
  else if n ≤ 10 then 5 + 3 * (n - 1)
  else 32 + 4 * (n - 11)

-- Prove that the number of sticks at the 20th stage is 68
theorem sticks_at_20 : sticks 20 = 68 := by
  sorry

end sticks_at_20_l1570_157073


namespace geometric_sequence_condition_l1570_157003

-- Definitions based on conditions
def S (n : ℕ) (m : ℤ) : ℤ := 3^(n + 1) + m
def a1 (m : ℤ) : ℤ := S 1 m
def a_n (n : ℕ) : ℤ := if n = 1 then a1 (-3) else 2 * 3^n

-- The proof statement
theorem geometric_sequence_condition (m : ℤ) (h1 : a1 m = 3^2 + m) (h2 : ∀ n, n ≥ 2 → a_n n = 2 * 3^n) :
  m = -3 :=
sorry

end geometric_sequence_condition_l1570_157003


namespace fraction_addition_l1570_157063

theorem fraction_addition :
  (5 / (8 / 13) + 4 / 7) = (487 / 56) := by
  sorry

end fraction_addition_l1570_157063


namespace greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l1570_157087

theorem greatest_divisor_of_product_of_5_consecutive_multiples_of_4 :
  let n1 := 4
  let n2 := 8
  let n3 := 12
  let n4 := 16
  let n5 := 20
  let spf1 := 2 -- smallest prime factor of 4
  let spf2 := 2 -- smallest prime factor of 8
  let spf3 := 2 -- smallest prime factor of 12
  let spf4 := 2 -- smallest prime factor of 16
  let spf5 := 2 -- smallest prime factor of 20
  let p1 := n1^spf1
  let p2 := n2^spf2
  let p3 := n3^spf3
  let p4 := n4^spf4
  let p5 := n5^spf5
  let product := p1 * p2 * p3 * p4 * p5
  product % (2^24) = 0 :=
by 
  sorry

end greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l1570_157087


namespace smallest_coprime_to_210_l1570_157014

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end smallest_coprime_to_210_l1570_157014


namespace not_a_fraction_l1570_157034

axiom x : ℝ
axiom a : ℝ
axiom b : ℝ

noncomputable def A := 1 / (x^2)
noncomputable def B := (b + 3) / a
noncomputable def C := (x^2 - 1) / (x + 1)
noncomputable def D := (2 / 7) * a

theorem not_a_fraction : ¬ (D = A) ∧ ¬ (D = B) ∧ ¬ (D = C) :=
by 
  sorry

end not_a_fraction_l1570_157034


namespace price_of_each_bottle_is_3_l1570_157062

/-- Each bottle of iced coffee has 6 servings. -/
def servings_per_bottle : ℕ := 6

/-- Tricia drinks half a container (bottle) a day. -/
def daily_consumption_rate : ℕ := servings_per_bottle / 2

/-- Number of days in 2 weeks. -/
def duration_days : ℕ := 14

/-- Number of servings Tricia consumes in 2 weeks. -/
def total_servings : ℕ := daily_consumption_rate * duration_days

/-- Number of bottles needed to get the total servings. -/
def bottles_needed : ℕ := total_servings / servings_per_bottle

/-- The total cost of the bottles is $21. -/
def total_cost : ℕ := 21

/-- The price per bottle is the total cost divided by the number of bottles. -/
def price_per_bottle : ℕ := total_cost / bottles_needed

/-- The price of each bottle is $3. -/
theorem price_of_each_bottle_is_3 : price_per_bottle = 3 :=
by
  -- We assume the necessary steps and mathematical verifications have been done.
  sorry

end price_of_each_bottle_is_3_l1570_157062


namespace unique_not_in_range_l1570_157008

open Real

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 10 = 10) (h₆ : f a b c d 50 = 50) 
  (h₇ : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) :
  ∃! x, ¬ ∃ y, f a b c d y = x :=
  sorry

end unique_not_in_range_l1570_157008


namespace recycling_drive_l1570_157078

theorem recycling_drive (S : ℕ) 
  (h1 : ∀ (n : ℕ), n = 280 * S) -- Each section collected 280 kilos in two weeks
  (h2 : ∀ (t : ℕ), t = 2000 - 320) -- After the third week, they needed 320 kilos more to reach their target of 2000 kilos
  : S = 3 :=
by
  sorry

end recycling_drive_l1570_157078


namespace equal_sharing_of_chicken_wings_l1570_157089

theorem equal_sharing_of_chicken_wings 
  (initial_wings : ℕ) (additional_wings : ℕ) (number_of_friends : ℕ)
  (total_wings : ℕ) (wings_per_person : ℕ)
  (h_initial : initial_wings = 8)
  (h_additional : additional_wings = 10)
  (h_number : number_of_friends = 3)
  (h_total : total_wings = initial_wings + additional_wings)
  (h_division : wings_per_person = total_wings / number_of_friends) :
  wings_per_person = 6 := 
  by
  sorry

end equal_sharing_of_chicken_wings_l1570_157089


namespace six_digit_number_consecutive_evens_l1570_157058

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l1570_157058


namespace asymptotes_of_hyperbola_l1570_157098

-- Definitions for the hyperbola and the asymptotes
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1
def asymptote_equation (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = - (Real.sqrt 2 / 2) * x

-- The theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) (h : hyperbola_equation x y) :
  asymptote_equation x y :=
sorry

end asymptotes_of_hyperbola_l1570_157098


namespace ratio_of_sums_l1570_157025

theorem ratio_of_sums (p q r u v w : ℝ) 
  (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : u > 0) (h5 : v > 0) (h6 : w > 0)
  (h7 : p^2 + q^2 + r^2 = 49) (h8 : u^2 + v^2 + w^2 = 64)
  (h9 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_l1570_157025


namespace sum_first_15_terms_l1570_157049

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- Conditions
def a_7 := 1
def a_9 := 5

-- Prove that S_15 = 45
theorem sum_first_15_terms : 
  ∃ (a d : ℤ), 
    (arithmetic_sequence a d 7 = a_7) ∧ 
    (arithmetic_sequence a d 9 = a_9) ∧ 
    (sum_first_n_terms a d 15 = 45) :=
sorry

end sum_first_15_terms_l1570_157049


namespace probability_N_14_mod_5_is_1_l1570_157097

theorem probability_N_14_mod_5_is_1 :
  let total := 1950
  let favorable := 2
  let outcomes := 5
  (favorable / outcomes) = (2 / 5) := by
  sorry

end probability_N_14_mod_5_is_1_l1570_157097


namespace correct_calculation_l1570_157080

-- Define the base type for exponents
variables (a : ℝ)

theorem correct_calculation :
  (a^3 * a^5 = a^8) ∧ 
  ¬((a^3)^2 = a^5) ∧ 
  ¬(a^5 + a^2 = a^7) ∧ 
  ¬(a^6 / a^2 = a^3) :=
by
  sorry

end correct_calculation_l1570_157080


namespace rectangle_area_k_l1570_157055

theorem rectangle_area_k (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2)
  (h_diag : (length ^ 2 + width ^ 2) = d ^ 2) :
  ∃ (k : ℝ), k = 10 / 29 ∧ length * width = k * d ^ 2 := by
  sorry

end rectangle_area_k_l1570_157055


namespace domain_of_f_l1570_157084

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f : 
  {x : ℝ | x > -1 ∧ x ≤ 2 ∧ x ≠ 0 ∧ 4 - x^2 ≥ 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l1570_157084


namespace upsilon_value_l1570_157031

theorem upsilon_value (Upsilon : ℤ) (h : 5 * (-3) = Upsilon - 3) : Upsilon = -12 :=
by
  sorry

end upsilon_value_l1570_157031


namespace intersection_ST_l1570_157066

def S : Set ℝ := { x : ℝ | x < -5 } ∪ { x : ℝ | x > 5 }
def T : Set ℝ := { x : ℝ | -7 < x ∧ x < 3 }

theorem intersection_ST : S ∩ T = { x : ℝ | -7 < x ∧ x < -5 } := 
by 
  sorry

end intersection_ST_l1570_157066


namespace train_crossing_time_l1570_157019

-- Definitions for conditions
def train_length : ℝ := 100 -- train length in meters
def train_speed_kmh : ℝ := 90 -- train speed in km/hr
def train_speed_mps : ℝ := 25 -- train speed in m/s after conversion

-- Lean 4 statement to prove the time taken for the train to cross the electric pole is 4 seconds
theorem train_crossing_time : (train_length / train_speed_mps) = 4 := by
  sorry

end train_crossing_time_l1570_157019


namespace min_f_when_a_neg3_range_of_a_l1570_157012

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- First statement: Minimum value of f(x) when a = -3
theorem min_f_when_a_neg3 : (∀ x : ℝ, f x (-3) ≥ 4) ∧ (∃ x : ℝ,  f x (-3) = 4) := by
  sorry

-- Second statement: Range of a given the condition
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≤ 2 * a + 2 * abs (x - 1)) ↔ a ≥ 1/3 := by
  sorry

end min_f_when_a_neg3_range_of_a_l1570_157012


namespace equal_potatoes_l1570_157067

theorem equal_potatoes (total_potatoes : ℕ) (total_people : ℕ) (h_potatoes : total_potatoes = 24) (h_people : total_people = 3) :
  (total_potatoes / total_people) = 8 :=
by {
  sorry
}

end equal_potatoes_l1570_157067
