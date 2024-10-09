import Mathlib

namespace min_sum_abc_l180_18035

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end min_sum_abc_l180_18035


namespace largest_common_term_l180_18028

theorem largest_common_term (a : ℕ) (k l : ℕ) (hk : a = 4 + 5 * k) (hl : a = 5 + 10 * l) (h : a < 300) : a = 299 :=
by {
  sorry
}

end largest_common_term_l180_18028


namespace rate_per_sq_meter_l180_18046

theorem rate_per_sq_meter
  (length : Float := 9)
  (width : Float := 4.75)
  (total_cost : Float := 38475)
  : (total_cost / (length * width)) = 900 := 
by
  sorry

end rate_per_sq_meter_l180_18046


namespace stuffed_animals_total_l180_18091

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l180_18091


namespace cos_alpha_value_l180_18036

theorem cos_alpha_value (α β : Real) (hα1 : 0 < α) (hα2 : α < π / 2) 
    (hβ1 : π / 2 < β) (hβ2 : β < π) (hcosβ : Real.cos β = -1/3)
    (hsin_alpha_beta : Real.sin (α + β) = 1/3) : 
    Real.cos α = 4 * Real.sqrt 2 / 9 := by
  sorry

end cos_alpha_value_l180_18036


namespace billy_age_l180_18070

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 64) : B = 48 :=
by
  sorry

end billy_age_l180_18070


namespace sum_of_n_and_k_l180_18038

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end sum_of_n_and_k_l180_18038


namespace ribbon_total_length_l180_18062

theorem ribbon_total_length (R : ℝ)
  (h_first : R - (1/2)*R = (1/2)*R)
  (h_second : (1/2)*R - (1/3)*((1/2)*R) = (1/3)*R)
  (h_third : (1/3)*R - (1/2)*((1/3)*R) = (1/6)*R)
  (h_remaining : (1/6)*R = 250) :
  R = 1500 :=
sorry

end ribbon_total_length_l180_18062


namespace sequence_is_constant_l180_18041

theorem sequence_is_constant
  (a : ℕ+ → ℝ)
  (S : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, S n + S (n + 1) = a (n + 1))
  : ∀ n : ℕ+, a n = 0 :=
by
  sorry

end sequence_is_constant_l180_18041


namespace fuel_tank_initial_capacity_l180_18039

variables (fuel_consumption : ℕ) (journey_distance remaining_fuel initial_fuel : ℕ)

-- Define conditions
def fuel_consumption_rate := 12      -- liters per 100 km
def journey := 275                  -- km
def remaining := 14                 -- liters
def fuel_converted := (fuel_consumption_rate * journey) / 100

-- Define the proposition to be proved
theorem fuel_tank_initial_capacity :
  initial_fuel = fuel_converted + remaining :=
sorry

end fuel_tank_initial_capacity_l180_18039


namespace tan_sum_half_l180_18099

theorem tan_sum_half (a b : ℝ) (h1 : Real.cos a + Real.cos b = 3/5) (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1 / 3 := 
by
  sorry

end tan_sum_half_l180_18099


namespace division_of_fraction_simplified_l180_18076

theorem division_of_fraction_simplified :
  12 / (2 / (5 - 3)) = 12 := 
by
  sorry

end division_of_fraction_simplified_l180_18076


namespace parallel_lines_condition_l180_18011

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 6 = 0 → (a - 2) * x + 3 * y + 2 * a = 0 → False) ↔ a = -1 :=
sorry

end parallel_lines_condition_l180_18011


namespace smallest_whole_number_l180_18090

theorem smallest_whole_number (a b c d : ℤ)
  (h₁ : a = 3 + 1 / 3)
  (h₂ : b = 4 + 1 / 4)
  (h₃ : c = 5 + 1 / 6)
  (h₄ : d = 6 + 1 / 8)
  (h₅ : a + b + c + d - 2 > 16)
  (h₆ : a + b + c + d - 2 < 17) :
  17 > 16 + (a + b + c + d - 18) - 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 8 :=
  sorry

end smallest_whole_number_l180_18090


namespace smallest_four_digit_divisible_by_4_and_5_l180_18095

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l180_18095


namespace vector_CB_correct_l180_18051

-- Define the vectors AB and AC
def AB : ℝ × ℝ := (2, 3)
def AC : ℝ × ℝ := (-1, 2)

-- Define the vector CB as the difference of AB and AC
def CB (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Prove that CB = (3, 1) given AB and AC
theorem vector_CB_correct : CB AB AC = (3, 1) :=
by
  sorry

end vector_CB_correct_l180_18051


namespace solutionSet_l180_18065

def passesThroughQuadrants (a b : ℝ) : Prop :=
  a > 0

def intersectsXAxisAt (a b : ℝ) : Prop :=
  b = 2 * a

theorem solutionSet (a b x : ℝ) (hq : passesThroughQuadrants a b) (hi : intersectsXAxisAt a b) :
  (a * x > b) ↔ (x > 2) :=
by
  sorry

end solutionSet_l180_18065


namespace roots_of_equation_l180_18026

theorem roots_of_equation {x : ℝ} :
  (12 * x^2 - 31 * x - 6 = 0) →
  (x = (31 + Real.sqrt 1249) / 24 ∨ x = (31 - Real.sqrt 1249) / 24) :=
by
  sorry

end roots_of_equation_l180_18026


namespace total_weight_of_arrangement_l180_18084

def original_side_length : ℤ := 4
def original_weight : ℤ := 16
def larger_side_length : ℤ := 10

theorem total_weight_of_arrangement :
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  total_weight = 96 :=
by
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  sorry

end total_weight_of_arrangement_l180_18084


namespace half_radius_y_l180_18001

theorem half_radius_y (r_x r_y : ℝ) (hx : 2 * Real.pi * r_x = 12 * Real.pi) (harea : Real.pi * r_x ^ 2 = Real.pi * r_y ^ 2) : r_y / 2 = 3 := by
  sorry

end half_radius_y_l180_18001


namespace parallel_lines_condition_l180_18014

theorem parallel_lines_condition (a : ℝ) (l : ℝ) :
  (∀ (x y : ℝ), ax + 3*y + 3 = 0 → x + (a - 2)*y + l = 0 → a = -1) ∧ (a = -1 → ∀ (x y : ℝ), (ax + 3*y + 3 = 0 ↔ x + (a - 2)*y + l = 0)) :=
sorry

end parallel_lines_condition_l180_18014


namespace volume_of_sphere_in_cone_l180_18022

theorem volume_of_sphere_in_cone :
  let r_base := 9
  let h_cone := 9
  let diameter_sphere := 9 * Real.sqrt 2
  let radius_sphere := diameter_sphere / 2
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere ^ 3
  volume_sphere = (1458 * Real.sqrt 2 / 4) * Real.pi :=
by
  sorry

end volume_of_sphere_in_cone_l180_18022


namespace min_sum_xy_l180_18052

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l180_18052


namespace polynomial_solution_l180_18058

theorem polynomial_solution (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) : x^4 + 2 * x^3 + 2 * x^2 + 2 * x + 1 = 0 := 
by {
  sorry
}

end polynomial_solution_l180_18058


namespace sqrt_144000_simplified_l180_18073

theorem sqrt_144000_simplified : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_simplified_l180_18073


namespace min_rainfall_on_fourth_day_l180_18010

theorem min_rainfall_on_fourth_day : 
  let capacity_ft := 6
  let drain_per_day_in := 3
  let rain_first_day_in := 10
  let rain_second_day_in := 2 * rain_first_day_in
  let rain_third_day_in := 1.5 * rain_second_day_in
  let total_rain_first_three_days_in := rain_first_day_in + rain_second_day_in + rain_third_day_in
  let total_drain_in := 3 * drain_per_day_in
  let water_level_start_fourth_day_in := total_rain_first_three_days_in - total_drain_in
  let capacity_in := capacity_ft * 12
  capacity_in = water_level_start_fourth_day_in + 21 :=
by
  sorry

end min_rainfall_on_fourth_day_l180_18010


namespace circles_are_intersecting_l180_18093

-- Define the circles and the distances given
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 5
def distance_O1O2 : ℝ := 2

-- Define the positional relationships
inductive PositionalRelationship
| externally_tangent
| intersecting
| internally_tangent
| contained_within_each_other

open PositionalRelationship

-- State the theorem to be proved
theorem circles_are_intersecting :
  distance_O1O2 > 0 ∧ distance_O1O2 < (radius_O1 + radius_O2) ∧ distance_O1O2 > abs (radius_O1 - radius_O2) →
  PositionalRelationship := 
by
  intro h
  exact PositionalRelationship.intersecting

end circles_are_intersecting_l180_18093


namespace four_x_plus_t_odd_l180_18080

theorem four_x_plus_t_odd (x t : ℤ) (hx : 2 * x - t = 11) : ¬(∃ n : ℤ, 4 * x + t = 2 * n) :=
by
  -- Since we need to prove the statement, we start a proof block
  sorry -- skipping the actual proof part for this statement

end four_x_plus_t_odd_l180_18080


namespace scalene_polygon_exists_l180_18059

theorem scalene_polygon_exists (n: ℕ) (a: Fin n → ℝ) (h: ∀ i, 1 ≤ a i ∧ a i ≤ 2013) (h_geq: n ≥ 13):
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ a A + a B > a C ∧ a A + a C > a B ∧ a B + a C > a A :=
sorry

end scalene_polygon_exists_l180_18059


namespace product_of_differences_l180_18088

theorem product_of_differences (p q p' q' α β α' β' : ℝ)
  (h1 : α + β = -p) (h2 : α * β = q)
  (h3 : α' + β' = -p') (h4 : α' * β' = q') :
  ((α - α') * (α - β') * (β - α') * (β - β') = (q - q')^2 + (p - p') * (q' * p - p' * q)) :=
sorry

end product_of_differences_l180_18088


namespace cranes_in_each_flock_l180_18086

theorem cranes_in_each_flock (c : ℕ) (h1 : ∃ n : ℕ, 13 * n = 221)
  (h2 : ∃ n : ℕ, c * n = 221) :
  c = 221 :=
by sorry

end cranes_in_each_flock_l180_18086


namespace find_a_l180_18087

def E (a b c : ℤ) : ℤ := a * b * b + c

theorem find_a (a : ℤ) : E a 3 1 = E a 5 11 → a = -5 / 8 := 
by sorry

end find_a_l180_18087


namespace boat_distance_along_stream_l180_18007

-- Define the conditions
def speed_of_boat_still_water : ℝ := 9
def distance_against_stream_per_hour : ℝ := 7

-- Define the speed of the stream
def speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour

-- Define the speed of the boat along the stream
def speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream

-- Theorem statement
theorem boat_distance_along_stream (speed_of_boat_still_water : ℝ)
                                    (distance_against_stream_per_hour : ℝ)
                                    (effective_speed_against_stream : ℝ := speed_of_boat_still_water - speed_of_stream)
                                    (speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour)
                                    (speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream)
                                    (one_hour : ℝ := 1) :
  speed_of_boat_along_stream = 11 := 
  by
    sorry

end boat_distance_along_stream_l180_18007


namespace fraction_product_eq_one_l180_18072

theorem fraction_product_eq_one :
  (7 / 4 : ℚ) * (8 / 14) * (21 / 12) * (16 / 28) * (49 / 28) * (24 / 42) * (63 / 36) * (32 / 56) = 1 := by
  sorry

end fraction_product_eq_one_l180_18072


namespace point_quadrant_l180_18077

theorem point_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : b < 0 ∧ a > 0 := 
by {
  sorry
}

end point_quadrant_l180_18077


namespace ratio_of_work_speeds_l180_18017

theorem ratio_of_work_speeds (B_speed : ℚ) (combined_speed : ℚ) (A_speed : ℚ) 
  (h1 : B_speed = 1/12) 
  (h2 : combined_speed = 1/4) 
  (h3 : A_speed + B_speed = combined_speed) : 
  A_speed / B_speed = 2 := 
sorry

end ratio_of_work_speeds_l180_18017


namespace sum_of_interior_angles_of_remaining_polygon_l180_18030

theorem sum_of_interior_angles_of_remaining_polygon (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 5) :
  (n - 2) * 180 ≠ 270 :=
by 
  sorry

end sum_of_interior_angles_of_remaining_polygon_l180_18030


namespace andrea_reaches_lauren_in_25_minutes_l180_18081

noncomputable def initial_distance : ℝ := 30
noncomputable def decrease_rate : ℝ := 90
noncomputable def Lauren_stop_time : ℝ := 10 / 60

theorem andrea_reaches_lauren_in_25_minutes :
  ∃ v_L v_A : ℝ, v_A = 2 * v_L ∧ v_A + v_L = decrease_rate ∧ ∃ remaining_distance remaining_time final_time : ℝ, 
  remaining_distance = initial_distance - decrease_rate * Lauren_stop_time ∧ 
  remaining_time = remaining_distance / v_A ∧ 
  final_time = Lauren_stop_time + remaining_time ∧ 
  final_time * 60 = 25 :=
sorry

end andrea_reaches_lauren_in_25_minutes_l180_18081


namespace expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l180_18049

-- Definitions to add parentheses in the given expressions to achieve the desired results.
def expr1 := 7 * (9 + 12 / 3)
def expr2 := (7 * 9 + 12) / 3
def expr3 := 7 * (9 + 12) / 3
def expr4 := (48 * 6) / (48 * 6)

-- Proof statements
theorem expr1_is_91 : expr1 = 91 := 
by sorry

theorem expr2_is_25 : expr2 = 25 :=
by sorry

theorem expr3_is_49 : expr3 = 49 :=
by sorry

theorem expr4_is_1 : expr4 = 1 :=
by sorry

end expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l180_18049


namespace dice_probability_l180_18079

theorem dice_probability :
  let outcomes : List ℕ := [2, 3, 4, 5]
  let total_possible_outcomes := 6 * 6 * 6
  let successful_outcomes := 4 * 4 * 4
  (successful_outcomes / total_possible_outcomes : ℚ) = 8 / 27 :=
by
  sorry

end dice_probability_l180_18079


namespace snickers_cost_l180_18012

variable (S : ℝ)

def cost_of_snickers (n : ℝ) : Prop :=
  2 * n + 3 * (2 * n) = 12

theorem snickers_cost (h : cost_of_snickers S) : S = 1.50 :=
by
  sorry

end snickers_cost_l180_18012


namespace percentage_fractions_l180_18085

theorem percentage_fractions : (3 / 8 / 100) * (160 : ℚ) = 3 / 5 :=
by
  sorry

end percentage_fractions_l180_18085


namespace lottery_numbers_bound_l180_18056

theorem lottery_numbers_bound (s : ℕ) (k : ℕ) (num_tickets : ℕ) (num_numbers : ℕ) (nums_per_ticket : ℕ)
  (h_tickets : num_tickets = 100) (h_numbers : num_numbers = 90) (h_nums_per_ticket : nums_per_ticket = 5)
  (h_s : s = num_tickets) (h_k : k = 49) :
  ∃ n : ℕ, n ≤ 10 :=
by
  sorry

end lottery_numbers_bound_l180_18056


namespace minimize_abs_difference_and_product_l180_18089

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end minimize_abs_difference_and_product_l180_18089


namespace negation_equivalence_l180_18071

-- Define the proposition P stating 'there exists an x in ℝ such that x^2 - 2x + 4 > 0'
def P : Prop := ∃ x : ℝ, x^2 - 2*x + 4 > 0

-- Define the proposition Q which is the negation of proposition P
def Q : Prop := ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0

-- State the proof problem: Prove that the negation of proposition P is equivalent to proposition Q
theorem negation_equivalence : ¬ P ↔ Q := by
  -- Proof to be provided.
  sorry

end negation_equivalence_l180_18071


namespace halfway_fraction_l180_18097

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l180_18097


namespace jeff_corrected_mean_l180_18047

def initial_scores : List ℕ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℕ := [85, 90, 92, 93, 89, 89, 88]

noncomputable def arithmetic_mean (scores : List ℕ) : ℝ :=
  (scores.sum : ℝ) / (scores.length : ℝ)

theorem jeff_corrected_mean :
  arithmetic_mean corrected_scores = 89.42857142857143 := 
by
  sorry

end jeff_corrected_mean_l180_18047


namespace gcd_3375_9180_l180_18061

-- Definition of gcd and the problem condition
theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry -- Proof can be filled in with the steps using the Euclidean algorithm

end gcd_3375_9180_l180_18061


namespace probability_one_side_is_side_of_decagon_l180_18054

theorem probability_one_side_is_side_of_decagon :
  let decagon_vertices := 10
  let total_triangles := Nat.choose decagon_vertices 3
  let favorable_one_side :=
    decagon_vertices * (decagon_vertices - 3) / 2
  let favorable_two_sides := decagon_vertices
  let favorable_outcomes := favorable_one_side + favorable_two_sides
  let probability := favorable_outcomes / total_triangles
  total_triangles = 120 ∧ favorable_outcomes = 60 ∧ probability = 1 / 2 := 
by
  sorry

end probability_one_side_is_side_of_decagon_l180_18054


namespace colton_stickers_left_l180_18018

theorem colton_stickers_left :
  let C := 72
  let F := 4 * 3 -- stickers given to three friends
  let M := F + 2 -- stickers given to Mandy
  let J := M - 10 -- stickers given to Justin
  let T := F + M + J -- total stickers given away
  C - T = 42 := by
  sorry

end colton_stickers_left_l180_18018


namespace area_of_inscribed_hexagon_in_square_is_27sqrt3_l180_18013

noncomputable def side_length_of_triangle : ℝ := 6
noncomputable def radius_of_circle (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2
noncomputable def side_length_of_square (r : ℝ) : ℝ := 2 * r
noncomputable def side_length_of_hexagon_in_square (s : ℝ) : ℝ := s / (Real.sqrt 2)
noncomputable def area_of_hexagon (side_hexagon : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side_hexagon^2

theorem area_of_inscribed_hexagon_in_square_is_27sqrt3 :
  ∀ (a r s side_hex : ℝ), 
    a = side_length_of_triangle →
    r = radius_of_circle a →
    s = side_length_of_square r →
    side_hex = side_length_of_hexagon_in_square s →
    area_of_hexagon side_hex = 27 * Real.sqrt 3 :=
by
  intros a r s side_hex h_a h_r h_s h_side_hex
  sorry

end area_of_inscribed_hexagon_in_square_is_27sqrt3_l180_18013


namespace cement_bought_l180_18074

-- Define the three conditions given in the problem
def original_cement : ℕ := 98
def son_contribution : ℕ := 137
def total_cement : ℕ := 450

-- Using those conditions, state that the amount of cement he bought is 215 lbs
theorem cement_bought :
  original_cement + son_contribution = 235 ∧ total_cement - (original_cement + son_contribution) = 215 := 
by {
  sorry
}

end cement_bought_l180_18074


namespace problem_x2_minus_y2_l180_18002

-- Problem statement: Given the conditions, prove x^2 - y^2 = 5 / 1111
theorem problem_x2_minus_y2 (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 101) :
  x^2 - y^2 = 5 / 1111 :=
by
  sorry

end problem_x2_minus_y2_l180_18002


namespace probability_A2_l180_18025

-- Define events and their probabilities
variable (A1 : Prop) (A2 : Prop) (B1 : Prop)
variable (P : Prop → ℝ)
variable [MeasureTheory.MeasureSpace ℝ]

-- Conditions given in the problem
axiom P_A1 : P A1 = 0.5
axiom P_B1 : P B1 = 0.5
axiom P_A2_given_A1 : P (A2 ∧ A1) / P A1 = 0.7
axiom P_A2_given_B1 : P (A2 ∧ B1) / P B1 = 0.8

-- Theorem statement to prove
theorem probability_A2 : P A2 = 0.75 :=
by
  -- Skipping the proof as per instructions
  sorry

end probability_A2_l180_18025


namespace simplest_fraction_is_D_l180_18024

def fractionA (x : ℕ) : ℚ := 10 / (15 * x)
def fractionB (a b : ℕ) : ℚ := (2 * a * b) / (3 * a * a)
def fractionC (x : ℕ) : ℚ := (x + 1) / (3 * x + 3)
def fractionD (x : ℕ) : ℚ := (x + 1) / (x * x + 1)

theorem simplest_fraction_is_D (x a b : ℕ) :
  ¬ ∃ c, c ≠ 1 ∧
    (fractionA x = (fractionA x / c) ∨
     fractionB a b = (fractionB a b / c) ∨
     fractionC x = (fractionC x / c)) ∧
    ∀ d, d ≠ 1 → fractionD x ≠ (fractionD x / d) := 
  sorry

end simplest_fraction_is_D_l180_18024


namespace six_divides_p_plus_one_l180_18043

theorem six_divides_p_plus_one 
  (p : ℕ) 
  (prime_p : Nat.Prime p) 
  (gt_three_p : p > 3) 
  (prime_p_plus_two : Nat.Prime (p + 2)) 
  (gt_three_p_plus_two : p + 2 > 3) : 
  6 ∣ (p + 1) := 
sorry

end six_divides_p_plus_one_l180_18043


namespace probability_shattering_l180_18009

theorem probability_shattering (total_cars : ℕ) (shattered_windshields : ℕ) (p : ℚ) 
  (h_total : total_cars = 20000) 
  (h_shattered: shattered_windshields = 600) 
  (h_p : p = shattered_windshields / total_cars) : 
  p = 0.03 := 
by 
  -- skipped proof
  sorry

end probability_shattering_l180_18009


namespace integer_solutions_of_quadratic_eq_l180_18003

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end integer_solutions_of_quadratic_eq_l180_18003


namespace line_tangent_circle_iff_m_l180_18005

/-- Definition of the circle and the line -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- Prove that the line is tangent to the circle if and only if m = -3 or m = -13 -/
theorem line_tangent_circle_iff_m (m : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y m) ↔ m = -3 ∨ m = -13 :=
by
  sorry

end line_tangent_circle_iff_m_l180_18005


namespace max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l180_18045

/-- Define the given function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

/-- The maximum value of the function f(x) is sqrt(2) -/
theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 := 
sorry

/-- The smallest positive period of the function f(x) -/
theorem smallest_positive_period_of_f :
  ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = Real.pi :=
sorry

/-- The set of values x that satisfy f(x) ≥ 1 -/
theorem values_of_x_satisfying_f_ge_1 :
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4 :=
sorry

end max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l180_18045


namespace find_four_real_numbers_l180_18037

theorem find_four_real_numbers (x1 x2 x3 x4 : ℝ) :
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
sorry

end find_four_real_numbers_l180_18037


namespace different_values_of_t_l180_18050

-- Define the conditions on the numbers
variables (p q r s t : ℕ)

-- Define the constraints: p, q, r, s, and t are distinct single-digit numbers
def valid_single_digit (x : ℕ) := x > 0 ∧ x < 10
def distinct_single_digits (p q r s t : ℕ) := 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

-- Define the relationships given in the problem
def conditions (p q r s t : ℕ) :=
  valid_single_digit p ∧
  valid_single_digit q ∧
  valid_single_digit r ∧
  valid_single_digit s ∧
  valid_single_digit t ∧
  distinct_single_digits p q r s t ∧
  p - q = r ∧
  r - s = t

-- Theorem to be proven
theorem different_values_of_t : 
  ∃! (count : ℕ), count = 6 ∧ (∃ p q r s t, conditions p q r s t) := 
sorry

end different_values_of_t_l180_18050


namespace complex_quadratic_solution_l180_18032

theorem complex_quadratic_solution (a b : ℝ) (h₁ : ∀ (x : ℂ), 5 * x ^ 2 - 4 * x + 20 = 0 → x = a + b * Complex.I ∨ x = a - b * Complex.I) :
 a + b ^ 2 = 394 / 25 := 
sorry

end complex_quadratic_solution_l180_18032


namespace profit_percentage_l180_18015

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 500) (h_selling : selling_price = 750) :
  ((selling_price - cost_price) / cost_price) * 100 = 50 :=
by
  sorry

end profit_percentage_l180_18015


namespace three_digit_integers_sat_f_n_eq_f_2005_l180_18040

theorem three_digit_integers_sat_f_n_eq_f_2005 
  (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m + n) = f (f m + n))
  (h2 : f 6 = 2)
  (h3 : f 6 ≠ f 9)
  (h4 : f 6 ≠ f 12)
  (h5 : f 6 ≠ f 15)
  (h6 : f 9 ≠ f 12)
  (h7 : f 9 ≠ f 15)
  (h8 : f 12 ≠ f 15) :
  ∃! n, 100 ≤ n ∧ n ≤ 999 ∧ f n = f 2005 → n = 225 := 
  sorry

end three_digit_integers_sat_f_n_eq_f_2005_l180_18040


namespace number_of_equilateral_triangles_in_lattice_l180_18075

-- Definitions representing the conditions of the problem
def is_unit_distance (a b : ℕ) : Prop :=
  true -- Assume true as we are not focusing on the definition

def expanded_hexagonal_lattice (p : ℕ) : Prop :=
  true -- Assume true as the specific construction details are abstracted

-- The target theorem statement
theorem number_of_equilateral_triangles_in_lattice 
  (lattice : ℕ → Prop) (dist : ℕ → ℕ → Prop) 
  (h₁ : ∀ p, lattice p → dist p p) 
  (h₂ : ∀ p, (expanded_hexagonal_lattice p) ↔ lattice p ∧ dist p p) : 
  ∃ n, n = 32 :=
by 
  existsi 32
  sorry

end number_of_equilateral_triangles_in_lattice_l180_18075


namespace triplet_sum_not_zero_l180_18016

def sum_triplet (a b c : ℝ) : ℝ := a + b + c

theorem triplet_sum_not_zero :
  ¬ (sum_triplet 3 (-5) 2 = 0) ∧
  (sum_triplet (1/4) (1/4) (-1/2) = 0) ∧
  (sum_triplet 0.3 (-0.1) (-0.2) = 0) ∧
  (sum_triplet 0.5 (-0.3) (-0.2) = 0) ∧
  (sum_triplet (1/3) (-1/6) (-1/6) = 0) :=
by 
  sorry

end triplet_sum_not_zero_l180_18016


namespace room_length_l180_18083

theorem room_length (L : ℝ) (width height door_area window_area cost_per_sq_ft total_cost : ℝ) 
    (num_windows : ℕ) (door_w window_w door_h window_h : ℝ)
    (h_width : width = 15) (h_height : height = 12) 
    (h_cost_per_sq_ft : cost_per_sq_ft = 9)
    (h_door_area : door_area = door_w * door_h)
    (h_window_area : window_area = window_w * window_h)
    (h_num_windows : num_windows = 3)
    (h_door_dim : door_w = 6 ∧ door_h = 3)
    (h_window_dim : window_w = 4 ∧ window_h = 3)
    (h_total_cost : total_cost = 8154) :
    (2 * height * (L + width) - (door_area + num_windows * window_area)) * cost_per_sq_ft = total_cost →
    L = 25 := 
by
  intros h_cost_eq
  sorry

end room_length_l180_18083


namespace find_sum_l180_18055

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem find_sum (h₁ : a * b = 2 * (a + b))
                (h₂ : b * c = 3 * (b + c))
                (h₃ : c * a = 4 * (a + c))
                (ha : a ≠ 0)
                (hb : b ≠ 0)
                (hc : c ≠ 0) 
                : a + b + c = 1128 / 35 :=
by
  sorry

end find_sum_l180_18055


namespace present_age_of_A_l180_18020

theorem present_age_of_A (A B C : ℕ) 
  (h1 : A + B + C = 57)
  (h2 : B - 3 = 2 * (A - 3))
  (h3 : C - 3 = 3 * (A - 3)) :
  A = 11 :=
sorry

end present_age_of_A_l180_18020


namespace new_acute_angle_l180_18068

/- Definitions -/
def initial_angle_A (ACB : ℝ) (angle_CAB : ℝ) := angle_CAB = 40
def rotation_degrees (rotation : ℝ) := rotation = 480

/- Theorem Statement -/
theorem new_acute_angle (ACB : ℝ) (angle_CAB : ℝ) (rotation : ℝ) :
  initial_angle_A angle_CAB ACB ∧ rotation_degrees rotation → angle_CAB = 80 := 
by
  intros h
  -- This is where you'd provide the proof steps, but we use 'sorry' to indicate the proof is skipped.
  sorry

end new_acute_angle_l180_18068


namespace remainder_div_84_l180_18048

def a := (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)

theorem remainder_div_84 (a : ℕ) (h : a = (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)) : a % 84 = 63 := 
by 
  -- Placeholder for the actual steps to prove
  sorry

end remainder_div_84_l180_18048


namespace total_value_correct_l180_18057

-- Define conditions
def import_tax_rate : ℝ := 0.07
def tax_paid : ℝ := 109.90
def tax_exempt_value : ℝ := 1000

-- Define total value
def total_value (V : ℝ) : Prop :=
  V - tax_exempt_value = tax_paid / import_tax_rate

-- Theorem stating that the total value is $2570
theorem total_value_correct : total_value 2570 := by
  sorry

end total_value_correct_l180_18057


namespace john_bought_3_tshirts_l180_18078

theorem john_bought_3_tshirts (T : ℕ) (h : 20 * T + 50 = 110) : T = 3 := 
by 
  sorry

end john_bought_3_tshirts_l180_18078


namespace find_x_l180_18004

theorem find_x (x : ℝ) (h₀ : ⌊x⌋ * x = 162) : x = 13.5 :=
sorry

end find_x_l180_18004


namespace value_of_f_g_3_l180_18021

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x + 2

theorem value_of_f_g_3 : f (g 3) = 83 := by
  sorry

end value_of_f_g_3_l180_18021


namespace probability_of_purple_marble_l180_18000

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) 
  (h_blue : p_blue = 0.3) 
  (h_green : p_green = 0.4) 
  (h_sum : p_blue + p_green + p_purple = 1) : 
  p_purple = 0.3 := 
by 
  -- proof goes here
  sorry

end probability_of_purple_marble_l180_18000


namespace cats_remaining_l180_18008

theorem cats_remaining 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 := 
by
  sorry

end cats_remaining_l180_18008


namespace angle_alpha_range_l180_18060

/-- Given point P (tan α, sin α - cos α) is in the first quadrant, 
and 0 ≤ α ≤ 2π, then the range of values for angle α is (π/4, π/2) ∪ (π, 5π/4). -/
theorem angle_alpha_range (α : ℝ) 
  (h0 : 0 ≤ α) (h1 : α ≤ 2 * Real.pi) 
  (h2 : Real.tan α > 0) (h3 : Real.sin α - Real.cos α > 0) : 
  (Real.pi / 4 < α ∧ α < Real.pi / 2) ∨ 
  (Real.pi < α ∧ α < 5 * Real.pi / 4) :=
sorry

end angle_alpha_range_l180_18060


namespace sum_of_positive_integers_l180_18096

theorem sum_of_positive_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 272) : x + y = 32 := 
by 
  sorry

end sum_of_positive_integers_l180_18096


namespace max_radius_of_sector_l180_18006

def sector_perimeter_area (r : ℝ) : ℝ := -r^2 + 10 * r

theorem max_radius_of_sector (R A : ℝ) (h : 2 * R + A = 20) : R = 5 :=
by
  sorry

end max_radius_of_sector_l180_18006


namespace perimeter_triangle_ABC_is_correct_l180_18029

noncomputable def semicircle_perimeter_trianlge_ABC : ℝ :=
  let BE := (1 : ℝ)
  let EF := (24 : ℝ)
  let FC := (3 : ℝ)
  let BC := BE + EF + FC
  let r := EF / 2
  let x := 71.5
  let AB := x + BE
  let AC := x + FC
  AB + BC + AC

theorem perimeter_triangle_ABC_is_correct : semicircle_perimeter_trianlge_ABC = 175 := by
  sorry

end perimeter_triangle_ABC_is_correct_l180_18029


namespace find_difference_l180_18094

theorem find_difference (x y : ℕ) (hx : ∃ k : ℕ, x = k^2) (h_sum_prod : x + y = x * y - 2006) : y - x = 666 :=
sorry

end find_difference_l180_18094


namespace packs_sold_in_other_villages_l180_18067

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end packs_sold_in_other_villages_l180_18067


namespace parabola_equation_l180_18053

variables (x y : ℝ)

def parabola_passes_through_point (x y : ℝ) : Prop :=
(x = 2 ∧ y = 7)

def focus_x_coord_five (x : ℝ) : Prop :=
(x = 5)

def axis_of_symmetry_parallel_to_y : Prop := True

def vertex_lies_on_x_axis (x y : ℝ) : Prop :=
(x = 5 ∧ y = 0)

theorem parabola_equation
  (h1 : parabola_passes_through_point x y)
  (h2 : focus_x_coord_five x)
  (h3 : axis_of_symmetry_parallel_to_y)
  (h4 : vertex_lies_on_x_axis x y) :
  49 * x + 3 * y^2 - 245 = 0
:= sorry

end parabola_equation_l180_18053


namespace original_cube_volume_eq_216_l180_18042

theorem original_cube_volume_eq_216 (a : ℕ)
  (h1 : ∀ (a : ℕ), ∃ V_orig V_new : ℕ, 
    V_orig = a^3 ∧ 
    V_new = (a + 1) * (a + 1) * (a - 2) ∧ 
    V_orig = V_new + 10) : 
  a = 6 → a^3 = 216 := 
by
  sorry

end original_cube_volume_eq_216_l180_18042


namespace even_func_decreasing_on_neg_interval_l180_18098

variable {f : ℝ → ℝ}

theorem even_func_decreasing_on_neg_interval
  (h_even : ∀ x, f x = f (-x))
  (h_increasing : ∀ (a b : ℝ), 3 ≤ a → a < b → b ≤ 7 → f a < f b)
  (h_min_val : ∀ x, 3 ≤ x → x ≤ 7 → f x ≥ 2) :
  (∀ (a b : ℝ), -7 ≤ a → a < b → b ≤ -3 → f a > f b) ∧ (∀ x, -7 ≤ x → x ≤ -3 → f x ≤ 2) :=
by
  sorry

end even_func_decreasing_on_neg_interval_l180_18098


namespace cost_of_painting_new_room_l180_18082

theorem cost_of_painting_new_room
  (L B H : ℝ)    -- Dimensions of the original room
  (c : ℝ)        -- Cost to paint the original room
  (h₁ : c = 350) -- Given that the cost of painting the original room is Rs. 350
  (A : ℝ)        -- Area of the walls of the original room
  (h₂ : A = 2 * (L + B) * H) -- Given the area calculation for the original room
  (newA : ℝ)     -- Area of the walls of the new room
  (h₃ : newA = 18 * (L + B) * H) -- Given the area calculation for the new room
  : (350 / (2 * (L + B) * H)) * (18 * (L + B) * H) = 3150 :=
by
  sorry

end cost_of_painting_new_room_l180_18082


namespace fixed_cost_calculation_l180_18069

theorem fixed_cost_calculation (TC MC n FC : ℕ) (h1 : TC = 16000) (h2 : MC = 200) (h3 : n = 20) (h4 : TC = FC + MC * n) : FC = 12000 :=
by
  sorry

end fixed_cost_calculation_l180_18069


namespace intersection_of_A_and_B_l180_18034

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {4} :=
by
  sorry

end intersection_of_A_and_B_l180_18034


namespace each_car_has_4_wheels_l180_18023
-- Import necessary libraries

-- Define the conditions
def number_of_guests := 40
def number_of_parent_cars := 2
def wheels_per_parent_car := 4
def number_of_guest_cars := 10
def total_wheels := 48
def parent_car_wheels := number_of_parent_cars * wheels_per_parent_car
def guest_car_wheels := total_wheels - parent_car_wheels

-- Define the proposition to prove
theorem each_car_has_4_wheels : (guest_car_wheels / number_of_guest_cars) = 4 :=
by
  sorry

end each_car_has_4_wheels_l180_18023


namespace total_items_deleted_l180_18064

-- Define the initial conditions
def initial_apps : Nat := 17
def initial_files : Nat := 21
def remaining_apps : Nat := 3
def remaining_files : Nat := 7
def transferred_files : Nat := 4

-- Prove the total number of deleted items
theorem total_items_deleted : (initial_apps - remaining_apps) + (initial_files - (remaining_files + transferred_files)) = 24 :=
by
  sorry

end total_items_deleted_l180_18064


namespace compute_x_over_w_l180_18019

theorem compute_x_over_w (w x y z : ℚ) (hw : w ≠ 0)
  (h1 : (x + 6 * y - 3 * z) / (-3 * x + 4 * w) = 2 / 3)
  (h2 : (-2 * y + z) / (x - w) = 2 / 3) :
  x / w = 2 / 3 :=
sorry

end compute_x_over_w_l180_18019


namespace sum_of_number_and_square_is_306_l180_18066

theorem sum_of_number_and_square_is_306 (n : ℕ) (h : n = 17) : n + n^2 = 306 :=
by
  sorry

end sum_of_number_and_square_is_306_l180_18066


namespace min_x_prime_sum_l180_18031

theorem min_x_prime_sum (x y : ℕ) (h : 3 * x^2 = 5 * y^4) :
  ∃ a b c d : ℕ, x = a^b * c^d ∧ (a + b + c + d = 11) := 
by sorry

end min_x_prime_sum_l180_18031


namespace inscribed_square_area_ratio_l180_18044

theorem inscribed_square_area_ratio (side_length : ℝ) (h_pos : side_length > 0) :
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  (inscribed_square_area / large_square_area) = (1 / 4) :=
by
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  sorry

end inscribed_square_area_ratio_l180_18044


namespace correct_relation_l180_18092

-- Define the set A
def A : Set ℤ := { x | x^2 - 4 = 0 }

-- The statement that 2 is an element of A
theorem correct_relation : 2 ∈ A :=
by 
    -- We skip the proof here
    sorry

end correct_relation_l180_18092


namespace find_a_b_find_extreme_point_g_num_zeros_h_l180_18033

-- (1) Proving the values of a and b
theorem find_a_b (a b : ℝ)
  (h1 : (3 + 2 * a + b = 0))
  (h2 : (3 - 2 * a + b = 0)) : 
  a = 0 ∧ b = -3 :=
sorry

-- (2) Proving the extreme points of g(x)
theorem find_extreme_point_g (x : ℝ) : 
  x = -2 :=
sorry

-- (3) Proving the number of zeros of h(x)
theorem num_zeros_h (c : ℝ) (h : -2 ≤ c ∧ c ≤ 2) :
  (|c| = 2 → ∃ y, y = 5) ∧ (|c| < 2 → ∃ y, y = 9) :=
sorry

end find_a_b_find_extreme_point_g_num_zeros_h_l180_18033


namespace total_basketballs_l180_18027

theorem total_basketballs (soccer_balls : ℕ) (soccer_balls_with_holes : ℕ) (basketballs_with_holes : ℕ) (balls_without_holes : ℕ) 
  (h1 : soccer_balls = 40) 
  (h2 : soccer_balls_with_holes = 30) 
  (h3 : basketballs_with_holes = 7) 
  (h4 : balls_without_holes = 18)
  (soccer_balls_without_holes : ℕ) 
  (basketballs_without_holes : ℕ) 
  (total_basketballs : ℕ)
  (h5 : soccer_balls_without_holes = soccer_balls - soccer_balls_with_holes)
  (h6 : basketballs_without_holes = balls_without_holes - soccer_balls_without_holes)
  (h7 : total_basketballs = basketballs_without_holes + basketballs_with_holes) : 
  total_basketballs = 15 := 
sorry

end total_basketballs_l180_18027


namespace milk_cost_correct_l180_18063

-- Definitions of the given conditions
def bagelCost : ℝ := 0.95
def orangeJuiceCost : ℝ := 0.85
def sandwichCost : ℝ := 4.65
def lunchExtraCost : ℝ := 4.0

-- Total cost of breakfast
def breakfastCost : ℝ := bagelCost + orangeJuiceCost

-- Total cost of lunch
def lunchCost : ℝ := breakfastCost + lunchExtraCost

-- Cost of milk
def milkCost : ℝ := lunchCost - sandwichCost

-- Theorem to prove the cost of milk
theorem milk_cost_correct : milkCost = 1.15 :=
by
  sorry

end milk_cost_correct_l180_18063
