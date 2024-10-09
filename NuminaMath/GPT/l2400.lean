import Mathlib

namespace finite_decimals_are_rational_l2400_240086

-- Conditions as definitions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_infinite_decimal (x : ℝ) : Prop := ¬∃ (n : ℤ), x = ↑n
def is_finite_decimal (x : ℝ) : Prop := ∃ (a b : ℕ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Equivalence to statement C: Finite decimals are rational numbers
theorem finite_decimals_are_rational : ∀ (x : ℝ), is_finite_decimal x → is_rational x := by
  sorry

end finite_decimals_are_rational_l2400_240086


namespace max_f_value_range_of_a_l2400_240029

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem max_f_value (a : ℝ) : ∃ x, f x a = 5 - a :=
sorry

theorem range_of_a (a : ℝ) : (∃ x, f x a ≥ (4 / a) + 1) ↔ (a = 2 ∨ a < 0) :=
sorry

end max_f_value_range_of_a_l2400_240029


namespace g_1_5_l2400_240008

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_defined (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g x ≠ 0

axiom g_zero : g 0 = 0

axiom g_mono (x y : ℝ) (hx : 0 ≤ x ∧ x < y ∧ y ≤ 1) : g x ≤ g y

axiom g_symmetry (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (1 - x) = 1 - g x

axiom g_scaling (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (x/4) = g x / 2

theorem g_1_5 : g (1 / 5) = 1 / 4 := 
sorry

end g_1_5_l2400_240008


namespace largest_number_of_hcf_lcm_l2400_240067

theorem largest_number_of_hcf_lcm (HCF : ℕ) (factor1 factor2 : ℕ) (n1 n2 : ℕ) (largest : ℕ) 
  (h1 : HCF = 52) 
  (h2 : factor1 = 11) 
  (h3 : factor2 = 12) 
  (h4 : n1 = HCF * factor1) 
  (h5 : n2 = HCF * factor2) 
  (h6 : largest = max n1 n2) : 
  largest = 624 := 
by 
  sorry

end largest_number_of_hcf_lcm_l2400_240067


namespace angle_y_equals_90_l2400_240074

/-- In a geometric configuration, if ∠CBD = 120° and ∠ABE = 30°, 
    then the measure of angle y is 90°. -/
theorem angle_y_equals_90 (angle_CBD angle_ABE : ℝ) 
  (h1 : angle_CBD = 120) 
  (h2 : angle_ABE = 30) : 
  ∃ y : ℝ, y = 90 := 
by
  sorry

end angle_y_equals_90_l2400_240074


namespace arithmetic_geometric_common_ratio_l2400_240028

theorem arithmetic_geometric_common_ratio (a₁ r : ℝ) 
  (h₁ : a₁ + a₁ * r^2 = 10) 
  (h₂ : a₁ * (1 + r + r^2 + r^3) = 15) : 
  r = 1/2 ∨ r = -1/2 :=
by {
  sorry
}

end arithmetic_geometric_common_ratio_l2400_240028


namespace total_cable_cost_neighborhood_l2400_240061

-- Define the number of east-west streets and their length
def ew_streets : ℕ := 18
def ew_length_per_street : ℕ := 2

-- Define the number of north-south streets and their length
def ns_streets : ℕ := 10
def ns_length_per_street : ℕ := 4

-- Define the cable requirements and cost
def cable_per_mile_of_street : ℕ := 5
def cable_cost_per_mile : ℕ := 2000

-- Calculate total length of east-west streets
def ew_total_length : ℕ := ew_streets * ew_length_per_street

-- Calculate total length of north-south streets
def ns_total_length : ℕ := ns_streets * ns_length_per_street

-- Calculate total length of all streets
def total_street_length : ℕ := ew_total_length + ns_total_length

-- Calculate total length of cable required
def total_cable_length : ℕ := total_street_length * cable_per_mile_of_street

-- Calculate total cost of the cable
def total_cost : ℕ := total_cable_length * cable_cost_per_mile

-- The statement to prove
theorem total_cable_cost_neighborhood : total_cost = 760000 :=
by
  sorry

end total_cable_cost_neighborhood_l2400_240061


namespace parabola_relative_positions_l2400_240073

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 3
def parabola3 (x : ℝ) : ℝ := x^2 + 2*x + 3

noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

theorem parabola_relative_positions :
  vertex_x 1 (-1) 3 < vertex_x 1 1 3 ∧ vertex_x 1 1 3 < vertex_x 1 2 3 :=
by {
  sorry
}

end parabola_relative_positions_l2400_240073


namespace cone_volume_l2400_240055

theorem cone_volume (r h : ℝ) (π : ℝ) (V : ℝ) :
    r = 3 → h = 4 → π = Real.pi → V = (1/3) * π * r^2 * h → V = 37.68 :=
by
  sorry

end cone_volume_l2400_240055


namespace sniper_B_has_greater_chance_of_winning_l2400_240069

-- Define the probabilities for sniper A
def p_A_1 := 0.4
def p_A_2 := 0.1
def p_A_3 := 0.5

-- Define the probabilities for sniper B
def p_B_1 := 0.1
def p_B_2 := 0.6
def p_B_3 := 0.3

-- Define the expected scores for sniper A and B
def E_A := 1 * p_A_1 + 2 * p_A_2 + 3 * p_A_3
def E_B := 1 * p_B_1 + 2 * p_B_2 + 3 * p_B_3

-- The statement we want to prove
theorem sniper_B_has_greater_chance_of_winning : E_B > E_A := by
  simp [E_A, E_B, p_A_1, p_A_2, p_A_3, p_B_1, p_B_2, p_B_3]
  sorry

end sniper_B_has_greater_chance_of_winning_l2400_240069


namespace value_of_8x_minus_5_squared_l2400_240034

theorem value_of_8x_minus_5_squared (x : ℝ) (h : 8 * x ^ 2 + 7 = 12 * x + 17) : (8 * x - 5) ^ 2 = 465 := 
sorry

end value_of_8x_minus_5_squared_l2400_240034


namespace sum_of_all_possible_x_l2400_240012

theorem sum_of_all_possible_x : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 8 ∨ x = 2)) → ( ∃ (x1 x2 : ℝ), (x1 = 8) ∧ (x2 = 2) ∧ (x1 + x2 = 10) ) :=
by
  admit

end sum_of_all_possible_x_l2400_240012


namespace Vasya_numbers_l2400_240083

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l2400_240083


namespace sum_of_cubes_div_xyz_l2400_240087

-- Given: x, y, z are non-zero real numbers, and x + y + z = 0.
-- Prove: (x^3 + y^3 + z^3) / (xyz) = 3.
theorem sum_of_cubes_div_xyz (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by
  sorry

end sum_of_cubes_div_xyz_l2400_240087


namespace tan_45_eq_one_l2400_240058

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l2400_240058


namespace sum_smallest_largest_even_integers_l2400_240024

theorem sum_smallest_largest_even_integers (n : ℕ) (h_odd : n % 2 = 1) (b z : ℤ)
  (h_mean : z = b + n - 1) : (b + (b + 2 * (n - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l2400_240024


namespace two_f_eq_eight_over_four_plus_x_l2400_240030

noncomputable def f : ℝ → ℝ := sorry

theorem two_f_eq_eight_over_four_plus_x (f_def : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) :=
by
  sorry

end two_f_eq_eight_over_four_plus_x_l2400_240030


namespace apples_per_pie_l2400_240016

theorem apples_per_pie (total_apples : ℕ) (unripe_apples : ℕ) (pies : ℕ) (ripe_apples : ℕ)
  (H1 : total_apples = 34)
  (H2 : unripe_apples = 6)
  (H3 : pies = 7)
  (H4 : ripe_apples = total_apples - unripe_apples) :
  ripe_apples / pies = 4 := by
  sorry

end apples_per_pie_l2400_240016


namespace andrea_fewer_apples_l2400_240017

theorem andrea_fewer_apples {total_apples given_to_zenny kept_by_yanna given_to_andrea : ℕ} 
  (h1 : total_apples = 60) 
  (h2 : given_to_zenny = 18) 
  (h3 : kept_by_yanna = 36) 
  (h4 : given_to_andrea = total_apples - kept_by_yanna - given_to_zenny) : 
  (given_to_andrea + 12 = given_to_zenny) := 
sorry

end andrea_fewer_apples_l2400_240017


namespace angle_sum_l2400_240097

theorem angle_sum (x : ℝ) (h1 : 2 * x + x = 90) : x = 30 := 
sorry

end angle_sum_l2400_240097


namespace hyperbola_asymptote_a_value_l2400_240072

theorem hyperbola_asymptote_a_value (a : ℝ) (h : 0 < a) 
  (asymptote_eq : y = (3 / 5) * x) :
  (x^2 / a^2 - y^2 / 9 = 1) → a = 5 :=
by
  sorry

end hyperbola_asymptote_a_value_l2400_240072


namespace sphere_radius_l2400_240094

-- Define the conditions
variable (r : ℝ) -- Radius of the sphere
variable (sphere_shadow : ℝ) (stick_height : ℝ) (stick_shadow : ℝ)

-- Given conditions
axiom sphere_shadow_equals_10 : sphere_shadow = 10
axiom stick_height_equals_1 : stick_height = 1
axiom stick_shadow_equals_2 : stick_shadow = 2

-- Using similar triangles and tangent relations, we want to prove the radius of sphere.
theorem sphere_radius (h1 : sphere_shadow = 10)
    (h2 : stick_height = 1)
    (h3 : stick_shadow = 2) : r = 5 :=
by
  -- Placeholder for the proof
  sorry

end sphere_radius_l2400_240094


namespace sum_first_n_arithmetic_sequence_l2400_240088

theorem sum_first_n_arithmetic_sequence (a1 d : ℝ) (S : ℕ → ℝ) :
  (S 3 + S 6 = 18) → 
  S 3 = 3 * a1 + 3 * d → 
  S 6 = 6 * a1 + 15 * d → 
  S 5 = 10 :=
by
  sorry

end sum_first_n_arithmetic_sequence_l2400_240088


namespace square_of_99_is_9801_l2400_240002

theorem square_of_99_is_9801 : 99 ^ 2 = 9801 := 
by
  sorry

end square_of_99_is_9801_l2400_240002


namespace alice_score_record_l2400_240011

def total_points : ℝ := 72
def average_points_others : ℝ := 4.7
def others_count : ℕ := 7

def total_points_others : ℝ := others_count * average_points_others
def alice_points : ℝ := total_points - total_points_others

theorem alice_score_record : alice_points = 39.1 :=
by {
  -- Proof should be inserted here
  sorry
}

end alice_score_record_l2400_240011


namespace g_4_minus_g_7_l2400_240023

theorem g_4_minus_g_7 (g : ℝ → ℝ) (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ k : ℝ, g (k + 1) - g k = 5) : g 4 - g 7 = -15 :=
by
  sorry

end g_4_minus_g_7_l2400_240023


namespace wxyz_sum_l2400_240036

noncomputable def wxyz (w x y z : ℕ) := 2^w * 3^x * 5^y * 7^z

theorem wxyz_sum (w x y z : ℕ) (h : wxyz w x y z = 1260) : w + 2 * x + 3 * y + 4 * z = 13 :=
sorry

end wxyz_sum_l2400_240036


namespace train_speed_l2400_240084

/-- Given: 
1. A train travels a distance of 80 km in 40 minutes. 
2. We need to prove that the speed of the train is 120 km/h.
-/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h_distance : distance = 80) 
  (h_time_minutes : time_minutes = 40) 
  (h_time_hours : time_hours = 40 / 60) 
  (h_speed : speed = distance / time_hours) : 
  speed = 120 :=
sorry

end train_speed_l2400_240084


namespace eight_machines_produce_ninety_six_bottles_in_three_minutes_l2400_240033

-- Define the initial conditions
def rate_per_machine: ℕ := 16 / 4 -- bottles per minute per machine

def total_bottles_8_machines_3_minutes: ℕ := 8 * rate_per_machine * 3

-- Prove the question
theorem eight_machines_produce_ninety_six_bottles_in_three_minutes:
  total_bottles_8_machines_3_minutes = 96 :=
by
  sorry

end eight_machines_produce_ninety_six_bottles_in_three_minutes_l2400_240033


namespace mowers_mow_l2400_240004

theorem mowers_mow (mowers hectares days mowers_new days_new : ℕ)
  (h1 : 3 * 3 * days = 3 * hectares)
  (h2 : 5 * days_new = 5 * (days_new * hectares / days)) :
  5 * days_new * (hectares / (3 * days)) = 25 / 3 :=
sorry

end mowers_mow_l2400_240004


namespace solve_quadratic_eq_l2400_240075

theorem solve_quadratic_eq (x : ℝ) : (x^2 + 4 * x = 5) ↔ (x = 1 ∨ x = -5) :=
by
  sorry

end solve_quadratic_eq_l2400_240075


namespace average_salary_rest_workers_l2400_240013

-- Define the conditions
def total_workers : Nat := 21
def average_salary_all_workers : ℝ := 8000
def number_of_technicians : Nat := 7
def average_salary_technicians : ℝ := 12000

-- Define the task
theorem average_salary_rest_workers :
  let number_of_rest := total_workers - number_of_technicians
  let total_salary_all := average_salary_all_workers * total_workers
  let total_salary_technicians := average_salary_technicians * number_of_technicians
  let total_salary_rest := total_salary_all - total_salary_technicians
  let average_salary_rest := total_salary_rest / number_of_rest
  average_salary_rest = 6000 :=
by
  sorry

end average_salary_rest_workers_l2400_240013


namespace sum_of_coefficients_of_expansion_l2400_240047

theorem sum_of_coefficients_of_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + a_2 + a_3 + a_4 + a_5 = 2 :=
by
  intro h
  have h0 := h 0
  have h1 := h 1
  sorry

end sum_of_coefficients_of_expansion_l2400_240047


namespace tangent_with_min_slope_has_given_equation_l2400_240080

-- Define the given function f(x)
def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the function f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the coordinates of the tangent point
def tangent_point : ℝ × ℝ := (-1, f (-1))

-- Define the equation of the tangent line at the point with the minimum slope
def tangent_line_equation (x y : ℝ) : Prop := 3 * x - y - 11 = 0

-- Main theorem statement that needs to be proved
theorem tangent_with_min_slope_has_given_equation :
  tangent_line_equation (-1) (f (-1)) :=
sorry

end tangent_with_min_slope_has_given_equation_l2400_240080


namespace Jeremy_age_l2400_240076

noncomputable def A : ℝ := sorry
noncomputable def J : ℝ := sorry
noncomputable def C : ℝ := sorry

-- Conditions
axiom h1 : A + J + C = 132
axiom h2 : A = (1/3) * J
axiom h3 : C = 2 * A

-- The goal is to prove J = 66
theorem Jeremy_age : J = 66 :=
sorry

end Jeremy_age_l2400_240076


namespace square_placement_conditions_l2400_240020

-- Definitions for natural numbers at vertices and center
def top_left := 14
def top_right := 6
def bottom_right := 15
def bottom_left := 35
def center := 210

theorem square_placement_conditions :
  (∃ gcd1 > 1, gcd1 = Nat.gcd top_left top_right) ∧
  (∃ gcd2 > 1, gcd2 = Nat.gcd top_right bottom_right) ∧
  (∃ gcd3 > 1, gcd3 = Nat.gcd bottom_right bottom_left) ∧
  (∃ gcd4 > 1, gcd4 = Nat.gcd bottom_left top_left) ∧
  (Nat.gcd top_left bottom_right = 1) ∧
  (Nat.gcd top_right bottom_left = 1) ∧
  (Nat.gcd top_left center > 1) ∧
  (Nat.gcd top_right center > 1) ∧
  (Nat.gcd bottom_right center > 1) ∧
  (Nat.gcd bottom_left center > 1) 
 := by
sorry

end square_placement_conditions_l2400_240020


namespace vanya_exam_scores_l2400_240082

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end vanya_exam_scores_l2400_240082


namespace no_integer_solutions_for_square_polynomial_l2400_240018

theorem no_integer_solutions_for_square_polynomial :
  (∀ x : ℤ, ∃ k : ℤ, k^2 = x^4 + 5*x^3 + 10*x^2 + 5*x + 25 → false) :=
by
  sorry

end no_integer_solutions_for_square_polynomial_l2400_240018


namespace min_a2_b2_c2_l2400_240005

theorem min_a2_b2_c2 (a b c : ℕ) (h : a + 2 * b + 3 * c = 73) : a^2 + b^2 + c^2 ≥ 381 :=
by sorry

end min_a2_b2_c2_l2400_240005


namespace hancho_milk_consumption_l2400_240042

theorem hancho_milk_consumption :
  ∀ (initial_yeseul_consumption gayoung_bonus liters_left initial_milk consumption_yeseul consumption_gayoung consumption_total), 
  initial_yeseul_consumption = 0.1 →
  gayoung_bonus = 0.2 →
  liters_left = 0.3 →
  initial_milk = 1 →
  consumption_yeseul = initial_yeseul_consumption →
  consumption_gayoung = initial_yeseul_consumption + gayoung_bonus →
  consumption_total = consumption_yeseul + consumption_gayoung →
  (initial_milk - (consumption_total + liters_left)) = 0.3 :=
by sorry

end hancho_milk_consumption_l2400_240042


namespace equation1_solution_equation2_solution_l2400_240038

-- Equation 1: x^2 + 2x - 8 = 0 has solutions x = -4 and x = 2.
theorem equation1_solution (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := by
  sorry

-- Equation 2: 2(x+3)^2 = x(x+3) has solutions x = -3 and x = -6.
theorem equation2_solution (x : ℝ) : 2 * (x + 3)^2 = x * (x + 3) ↔ x = -3 ∨ x = -6 := by
  sorry

end equation1_solution_equation2_solution_l2400_240038


namespace random_events_l2400_240077

def is_random_event_1 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a + d < 0 ∨ b + c > 0

def is_random_event_2 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a - d > 0 ∨ b - c < 0

def is_impossible_event_3 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a * b > 0

def is_certain_event_4 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a / b < 0

theorem random_events (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  is_random_event_1 a b ha hb ∧ is_random_event_2 a b ha hb :=
by
  sorry

end random_events_l2400_240077


namespace length_of_string_for_circle_l2400_240056

theorem length_of_string_for_circle (A : ℝ) (pi_approx : ℝ) (extra_length : ℝ) (hA : A = 616) (hpi : pi_approx = 22 / 7) (hextra : extra_length = 5) :
  ∃ (length : ℝ), length = 93 :=
by {
  sorry
}

end length_of_string_for_circle_l2400_240056


namespace correct_set_of_equations_l2400_240053

-- Define the digits x and y as integers
def digits (x y : ℕ) := x + y = 8

-- Conditions
def condition_1 (x y : ℕ) := 10*y + x + 18 = 10*x + y

theorem correct_set_of_equations : 
  ∃ (x y : ℕ), digits x y ∧ condition_1 x y :=
sorry

end correct_set_of_equations_l2400_240053


namespace find_x_solution_l2400_240001

theorem find_x_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (h_eq : (4 * x)^(Real.log 4 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 :=
by
  sorry

end find_x_solution_l2400_240001


namespace frog_jump_correct_l2400_240092

def grasshopper_jump : ℤ := 25
def additional_distance : ℤ := 15
def frog_jump : ℤ := grasshopper_jump + additional_distance

theorem frog_jump_correct : frog_jump = 40 := by
  sorry

end frog_jump_correct_l2400_240092


namespace simplify_polynomial_l2400_240026

variable (y : ℤ)

theorem simplify_polynomial :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 2 * y^9 + 4) = 
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 :=
by
  sorry

end simplify_polynomial_l2400_240026


namespace find_distance_between_PQ_l2400_240027

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end find_distance_between_PQ_l2400_240027


namespace reciprocal_of_neg_2023_l2400_240062

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l2400_240062


namespace harry_weekly_earnings_l2400_240035

def dogs_walked_per_day : Nat → Nat
| 1 => 7  -- Monday
| 2 => 12 -- Tuesday
| 3 => 7  -- Wednesday
| 4 => 9  -- Thursday
| 5 => 7  -- Friday
| _ => 0  -- Other days (not relevant for this problem)

def payment_per_dog : Nat := 5

def daily_earnings (day : Nat) : Nat :=
  dogs_walked_per_day day * payment_per_dog

def total_weekly_earnings : Nat :=
  (daily_earnings 1) + (daily_earnings 2) + (daily_earnings 3) +
  (daily_earnings 4) + (daily_earnings 5)

theorem harry_weekly_earnings : total_weekly_earnings = 210 :=
by
  sorry

end harry_weekly_earnings_l2400_240035


namespace email_scam_check_l2400_240057

-- Define the condition for receiving an email about winning a car
def received_email (info: String) : Prop :=
  info = "You received an email informing you that you have won a car. You are asked to provide your mobile phone number for contact and to transfer 150 rubles to a bank card to cover the postage fee for sending the invitation letter."

-- Define what indicates a scam
def is_scam (info: String) : Prop :=
  info = "Request for mobile number already known to the sender and an upfront payment."

-- Proving that the information in the email implies it is a scam
theorem email_scam_check (info: String) (h1: received_email info) : is_scam info :=
by
  sorry

end email_scam_check_l2400_240057


namespace sum_of_digits_9ab_l2400_240022

def a : ℕ := 999
def b : ℕ := 666

theorem sum_of_digits_9ab : 
  let n := 9 * a * b
  (n.digits 10).sum = 36 := 
by
  sorry

end sum_of_digits_9ab_l2400_240022


namespace perpendicular_vectors_k_zero_l2400_240085

theorem perpendicular_vectors_k_zero
  (k : ℝ)
  (a : ℝ × ℝ := (3, 1))
  (b : ℝ × ℝ := (1, 3))
  (c : ℝ × ℝ := (k, 2)) 
  (h : (a.1 - c.1, a.2 - c.2).1 * b.1 + (a.1 - c.1, a.2 - c.2).2 * b.2 = 0) :
  k = 0 :=
by
  sorry

end perpendicular_vectors_k_zero_l2400_240085


namespace initial_bananas_on_tree_l2400_240051

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end initial_bananas_on_tree_l2400_240051


namespace sum_of_remaining_six_numbers_l2400_240066

theorem sum_of_remaining_six_numbers :
  ∀ (S T U : ℕ), 
    S = 20 * 500 → T = 14 * 390 → U = S - T → U = 4540 :=
by
  intros S T U hS hT hU
  sorry

end sum_of_remaining_six_numbers_l2400_240066


namespace ellipse_minor_axis_length_l2400_240031

noncomputable def minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ) :=
  if (a > b ∧ b > 0 ∧ eccentricity = (Real.sqrt 5) / 3 ∧ sum_distances = 12) then
    2 * b
  else
    0

theorem ellipse_minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : eccentricity = (Real.sqrt 5) / 3) (h4 : sum_distances = 12) :
  minor_axis_length a b eccentricity sum_distances = 8 :=
sorry

end ellipse_minor_axis_length_l2400_240031


namespace households_in_city_l2400_240099

theorem households_in_city (x : ℕ) (h1 : x < 100) (h2 : x + x / 3 = 100) : x = 75 :=
sorry

end households_in_city_l2400_240099


namespace max_k_solution_l2400_240054

theorem max_k_solution
  (k x y : ℝ)
  (h_pos: 0 < k ∧ 0 < x ∧ 0 < y)
  (h_eq: 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  ∃ k, 8*k^3 - 8*k^2 - 7*k = 0 := 
sorry

end max_k_solution_l2400_240054


namespace explain_education_policy_l2400_240049

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end explain_education_policy_l2400_240049


namespace clock_angle_7_35_l2400_240009

theorem clock_angle_7_35 : 
  let minute_hand_angle := (35 / 60) * 360
  let hour_hand_angle := 7 * 30 + (35 / 60) * 30
  let angle_between := hour_hand_angle - minute_hand_angle
  angle_between = 17.5 := by
sorry

end clock_angle_7_35_l2400_240009


namespace final_apples_count_l2400_240093

def initial_apples : ℝ := 5708
def apples_given_away : ℝ := 2347.5
def additional_apples_harvested : ℝ := 1526.75

theorem final_apples_count :
  initial_apples - apples_given_away + additional_apples_harvested = 4887.25 :=
by
  sorry

end final_apples_count_l2400_240093


namespace expression_for_A_plus_2B_A_plus_2B_independent_of_b_l2400_240090

theorem expression_for_A_plus_2B (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * b - 1
  let B := -a^2 - a * b + 1
  A + 2 * B = a * b - 2 * b + 1 :=
by
  sorry

theorem A_plus_2B_independent_of_b (a : ℝ) :
  (∀ b : ℝ, let A := 2 * a^2 + 3 * a * b - 2 * b - 1
            let B := -a^2 - a * b + 1
            A + 2 * B = a * b - 2 * b + 1) →
  a = 2 :=
by
  sorry

end expression_for_A_plus_2B_A_plus_2B_independent_of_b_l2400_240090


namespace timeSpentReading_l2400_240064

def totalTime : ℕ := 120
def timeOnPiano : ℕ := 30
def timeWritingMusic : ℕ := 25
def timeUsingExerciser : ℕ := 27

theorem timeSpentReading :
  totalTime - timeOnPiano - timeWritingMusic - timeUsingExerciser = 38 := by
  sorry

end timeSpentReading_l2400_240064


namespace height_of_triangle_l2400_240096

variables (a b h' : ℝ)

theorem height_of_triangle (h : (1/2) * a * h' = a * b) : h' = 2 * b :=
sorry

end height_of_triangle_l2400_240096


namespace min_colors_to_distinguish_keys_l2400_240059

def min_colors_needed (n : Nat) : Nat :=
  if n <= 2 then n
  else if n >= 6 then 2
  else 3

theorem min_colors_to_distinguish_keys (n : Nat) :
  (n ≤ 2 → min_colors_needed n = n) ∧
  (3 ≤ n ∧ n ≤ 5 → min_colors_needed n = 3) ∧
  (n ≥ 6 → min_colors_needed n = 2) :=
by
  sorry

end min_colors_to_distinguish_keys_l2400_240059


namespace train_length_is_300_l2400_240070

noncomputable def length_of_train (V L : ℝ) : Prop :=
  (L = V * 18) ∧ (L + 500 = V * 48)

theorem train_length_is_300
  (V : ℝ) (L : ℝ) (h : length_of_train V L) : L = 300 :=
by
  sorry

end train_length_is_300_l2400_240070


namespace slope_condition_l2400_240006

theorem slope_condition {m : ℝ} : 
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end slope_condition_l2400_240006


namespace calculate_angle_C_l2400_240046

variable (A B C : ℝ)

theorem calculate_angle_C (h1 : A = C - 40) (h2 : B = 2 * A) (h3 : A + B + C = 180) :
  C = 75 :=
by
  sorry

end calculate_angle_C_l2400_240046


namespace parallel_condition_sufficient_not_necessary_l2400_240003

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x + 1, 3)

theorem parallel_condition_sufficient_not_necessary (x : ℝ) :
  (x = 2) → (a x = b x) ∨ (a (-2) = b (-2)) :=
by sorry

end parallel_condition_sufficient_not_necessary_l2400_240003


namespace cosine_of_eight_times_alpha_l2400_240081

theorem cosine_of_eight_times_alpha (α : ℝ) (hypotenuse : ℝ) 
  (cos_α : ℝ) (cos_2α : ℝ) (cos_4α : ℝ) 
  (h₀ : hypotenuse = Real.sqrt (1^2 + (Real.sqrt 2)^2))
  (h₁ : cos_α = (Real.sqrt 2) / hypotenuse)
  (h₂ : cos_2α = 2 * cos_α^2 - 1)
  (h₃ : cos_4α = 2 * cos_2α^2 - 1)
  (h₄ : cos_8α = 2 * cos_4α^2 - 1) :
  cos_8α = 17 / 81 := 
  by
  sorry

end cosine_of_eight_times_alpha_l2400_240081


namespace tips_fraction_of_income_l2400_240045

theorem tips_fraction_of_income
  (S T : ℝ)
  (h1 : T = (2 / 4) * S) :
  T / (S + T) = 1 / 3 :=
by
  -- Proof goes here
  sorry

end tips_fraction_of_income_l2400_240045


namespace homework_total_l2400_240021

theorem homework_total :
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  math_pages + reading_pages + science_pages = 62 :=
by
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  show math_pages + reading_pages + science_pages = 62
  sorry

end homework_total_l2400_240021


namespace distance_between_towns_proof_l2400_240050

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end distance_between_towns_proof_l2400_240050


namespace min_value_of_f_l2400_240098

noncomputable def f (x : ℝ) : ℝ := max (2 * x + 1) (5 - x)

theorem min_value_of_f : ∃ y, (∀ x : ℝ, f x ≥ y) ∧ y = 11 / 3 :=
by 
  sorry

end min_value_of_f_l2400_240098


namespace find_x_l2400_240048

def F (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ := a^b + c * d

theorem find_x (x : ℕ) : F 3 x 5 9 = 500 → x = 6 := 
by 
  sorry

end find_x_l2400_240048


namespace simplest_form_fraction_C_l2400_240068

def fraction_A (x : ℤ) (y : ℤ) : ℚ := (2 * x + 4) / (6 * x + 8)
def fraction_B (x : ℤ) (y : ℤ) : ℚ := (x + y) / (x^2 - y^2)
def fraction_C (x : ℤ) (y : ℤ) : ℚ := (x^2 + y^2) / (x + y)
def fraction_D (x : ℤ) (y : ℤ) : ℚ := (x^2 - y^2) / (x^2 - 2 * x * y + y^2)

theorem simplest_form_fraction_C (x y : ℤ) :
  ¬ (∃ (A : ℚ), A ≠ fraction_C x y ∧ (A = fraction_C x y)) :=
by
  intros
  sorry

end simplest_form_fraction_C_l2400_240068


namespace fermat_little_theorem_l2400_240063

theorem fermat_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a ^ p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l2400_240063


namespace number_of_ways_to_select_books_l2400_240037

theorem number_of_ways_to_select_books :
  let bag1 := 4
  let bag2 := 5
  bag1 * bag2 = 20 :=
by
  sorry

end number_of_ways_to_select_books_l2400_240037


namespace order_of_A_B_C_D_l2400_240052

def A := Nat.factorial 8 ^ Nat.factorial 8
def B := 8 ^ (8 ^ 8)
def C := 8 ^ 88
def D := 8 ^ 64

theorem order_of_A_B_C_D : D < C ∧ C < B ∧ B < A := by
  sorry

end order_of_A_B_C_D_l2400_240052


namespace ratio_of_speeds_correct_l2400_240095

noncomputable def ratio_speeds_proof_problem : Prop :=
  ∃ (v_A v_B : ℝ),
    (∀ t : ℝ, 0 ≤ t ∧ t = 3 → 3 * v_A = abs (-800 + 3 * v_B)) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t = 15 → 15 * v_A = abs (-800 + 15 * v_B)) ∧
    (3 * 15 * v_A / (15 * v_B) = 3 / 4)

theorem ratio_of_speeds_correct : ratio_speeds_proof_problem :=
sorry

end ratio_of_speeds_correct_l2400_240095


namespace solution_criteria_l2400_240089

def is_solution (M : ℕ) : Prop :=
  5 ∣ (1989^M + M^1989)

theorem solution_criteria (M : ℕ) (h : M < 10) : is_solution M ↔ (M = 1 ∨ M = 4) :=
sorry

end solution_criteria_l2400_240089


namespace requiredSheetsOfPaper_l2400_240007

-- Define the conditions
def englishAlphabetLetters : ℕ := 26
def timesWrittenPerLetter : ℕ := 3
def sheetsOfPaperPerLetter (letters : ℕ) (times : ℕ) : ℕ := letters * times

-- State the theorem equivalent to the original math problem
theorem requiredSheetsOfPaper : sheetsOfPaperPerLetter englishAlphabetLetters timesWrittenPerLetter = 78 := by
  sorry

end requiredSheetsOfPaper_l2400_240007


namespace pete_ate_percentage_l2400_240040

-- Definitions of the conditions
def total_slices : ℕ := 2 * 12
def stephen_ate_slices : ℕ := (25 * total_slices) / 100
def remaining_slices_after_stephen : ℕ := total_slices - stephen_ate_slices
def slices_left_after_pete : ℕ := 9

-- The statement to be proved
theorem pete_ate_percentage (h1 : total_slices = 24)
                            (h2 : stephen_ate_slices = 6)
                            (h3 : remaining_slices_after_stephen = 18)
                            (h4 : slices_left_after_pete = 9) :
  ((remaining_slices_after_stephen - slices_left_after_pete) * 100 / remaining_slices_after_stephen) = 50 :=
sorry

end pete_ate_percentage_l2400_240040


namespace john_saves_water_l2400_240039

-- Define the conditions
def old_water_per_flush : ℕ := 5
def num_flushes_per_day : ℕ := 15
def reduction_percentage : ℕ := 80
def days_in_june : ℕ := 30

-- Define the savings calculation
def water_saved_in_june : ℕ :=
  let old_daily_usage := old_water_per_flush * num_flushes_per_day
  let old_june_usage := old_daily_usage * days_in_june
  let new_water_per_flush := old_water_per_flush * (100 - reduction_percentage) / 100
  let new_daily_usage := new_water_per_flush * num_flushes_per_day
  let new_june_usage := new_daily_usage * days_in_june
  old_june_usage - new_june_usage

-- The proof problem statement
theorem john_saves_water : water_saved_in_june = 1800 := 
by
  -- Proof would go here
  sorry

end john_saves_water_l2400_240039


namespace range_of_a_l2400_240032

noncomputable def A := {x : ℝ | x^2 - 2*x - 8 < 0}
noncomputable def B := {x : ℝ | x^2 + 2*x - 3 > 0}
noncomputable def C (a : ℝ) := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem range_of_a (a : ℝ) :
  (C a ⊆ A ∩ B) ↔ (1 ≤ a ∧ a ≤ 2 ∨ a = 0) :=
sorry

end range_of_a_l2400_240032


namespace mean_height_is_approx_correct_l2400_240014

def heights : List ℕ := [120, 123, 127, 132, 133, 135, 140, 142, 145, 148, 152, 155, 158, 160]

def mean_height : ℚ := heights.sum / heights.length

theorem mean_height_is_approx_correct : 
  abs (mean_height - 140.71) < 0.01 := 
by
  sorry

end mean_height_is_approx_correct_l2400_240014


namespace previous_day_visitors_l2400_240078

-- Define the number of visitors on the day Rachel visited
def visitors_on_day_rachel_visited : ℕ := 317

-- Define the difference in the number of visitors between the day Rachel visited and the previous day
def extra_visitors : ℕ := 22

-- Prove that the number of visitors on the previous day is 295
theorem previous_day_visitors : visitors_on_day_rachel_visited - extra_visitors = 295 :=
by
  sorry

end previous_day_visitors_l2400_240078


namespace even_integer_squares_l2400_240015

noncomputable def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 25

theorem even_integer_squares (x : ℤ) (hx : x % 2 = 0) :
  (∃ (a : ℤ), Q x = a ^ 2) → x = 8 :=
by
  sorry

end even_integer_squares_l2400_240015


namespace sqrt_expression_evaluation_l2400_240044

theorem sqrt_expression_evaluation :
  (Real.sqrt 48 - 6 * Real.sqrt (1 / 3) - Real.sqrt 18 / Real.sqrt 6) = Real.sqrt 3 :=
by
  sorry

end sqrt_expression_evaluation_l2400_240044


namespace length_YW_l2400_240000

-- Definitions of the sides of the triangle
def XY := 6
def YZ := 8
def XZ := 10

-- The total perimeter of triangle XYZ
def perimeter : ℕ := XY + YZ + XZ

-- Each ant travels half the perimeter
def halfPerimeter : ℕ := perimeter / 2

-- Distance one ant travels from X to W through Y
def distanceXtoW : ℕ := XY + 6

-- Prove that the distance segment YW is 6
theorem length_YW : distanceXtoW = halfPerimeter := by sorry

end length_YW_l2400_240000


namespace geralds_average_speed_l2400_240091

theorem geralds_average_speed :
  ∀ (track_length : ℝ) (pollys_laps : ℕ) (pollys_time : ℝ) (geralds_factor : ℝ),
  track_length = 0.25 →
  pollys_laps = 12 →
  pollys_time = 0.5 →
  geralds_factor = 0.5 →
  (geralds_factor * (pollys_laps * track_length / pollys_time)) = 3 :=
by
  intro track_length pollys_laps pollys_time geralds_factor
  intro h_track_len h_pol_lys_laps h_pollys_time h_ger_factor
  sorry

end geralds_average_speed_l2400_240091


namespace work_ratio_l2400_240043

theorem work_ratio 
  (m b : ℝ) 
  (h : 7 * m + 2 * b = 6 * (m + b)) : 
  m / b = 4 := 
sorry

end work_ratio_l2400_240043


namespace baseball_card_total_percent_decrease_l2400_240019

theorem baseball_card_total_percent_decrease :
  ∀ (original_value first_year_decrease second_year_decrease : ℝ),
  first_year_decrease = 0.60 →
  second_year_decrease = 0.10 →
  original_value > 0 →
  (original_value - original_value * first_year_decrease - (original_value * (1 - first_year_decrease)) * second_year_decrease) =
  original_value * (1 - 0.64) :=
by
  intros original_value first_year_decrease second_year_decrease h_first_year h_second_year h_original_pos
  sorry

end baseball_card_total_percent_decrease_l2400_240019


namespace find_f_neg2_l2400_240060

-- Condition (1): f is an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Condition (2): f(x) = x^2 + 1 for x > 0
def function_defined_for_positive_x {f : ℝ → ℝ} (h_even : even_function f): Prop :=
  ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Proof problem: prove that given the conditions, f(-2) = 5
theorem find_f_neg2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_pos : function_defined_for_positive_x h_even) : 
  f (-2) = 5 := 
sorry

end find_f_neg2_l2400_240060


namespace ratio_of_bottles_l2400_240071

theorem ratio_of_bottles
  (initial_money : ℤ)
  (initial_bottles : ℕ)
  (cost_per_bottle : ℤ)
  (cost_per_pound_cheese : ℤ)
  (cheese_pounds : ℚ)
  (remaining_money : ℤ) :
  initial_money = 100 →
  initial_bottles = 4 →
  cost_per_bottle = 2 →
  cost_per_pound_cheese = 10 →
  cheese_pounds = 0.5 →
  remaining_money = 71 →
  (2 * initial_bottles) / initial_bottles = 2 :=
by 
  sorry

end ratio_of_bottles_l2400_240071


namespace arithmetic_sequence_value_l2400_240010

theorem arithmetic_sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)) -- definition of arithmetic sequence
  (h2 : a 2 + a 10 = -12) -- given that a_2 + a_{10} = -12
  (h3 : a_2 = -6) -- given that a_6 is the average of a_2 and a_{10}
  : a 6 = -6 :=
sorry

end arithmetic_sequence_value_l2400_240010


namespace total_earrings_after_one_year_l2400_240041

theorem total_earrings_after_one_year :
  let bella_earrings := 10
  let monica_earrings := 10 / 0.25
  let rachel_earrings := monica_earrings / 2
  let initial_total := bella_earrings + monica_earrings + rachel_earrings
  let olivia_earrings_initial := initial_total + 5
  let olivia_earrings_after := olivia_earrings_initial * 1.2
  let total_earrings := bella_earrings + monica_earrings + rachel_earrings + olivia_earrings_after
  total_earrings = 160 :=
by
  sorry

end total_earrings_after_one_year_l2400_240041


namespace hardcover_volumes_l2400_240025

theorem hardcover_volumes (h p : ℕ) (h1 : h + p = 10) (h2 : 25 * h + 15 * p = 220) : h = 7 :=
by sorry

end hardcover_volumes_l2400_240025


namespace senior_tickets_count_l2400_240065

theorem senior_tickets_count (A S : ℕ) 
  (h1 : A + S = 510)
  (h2 : 21 * A + 15 * S = 8748) :
  S = 327 :=
sorry

end senior_tickets_count_l2400_240065


namespace sequence_missing_number_l2400_240079

theorem sequence_missing_number : 
  ∃ x, (x - 21 = 7 ∧ 37 - x = 9) ∧ x = 28 := by
  sorry

end sequence_missing_number_l2400_240079
