import Mathlib

namespace NUMINAMATH_GPT_fewest_number_of_gymnasts_l90_9066

theorem fewest_number_of_gymnasts (n : ℕ) (h : n % 2 = 0)
  (handshakes : ∀ (n : ℕ), (n * (n - 1) / 2) + n = 465) : 
  n = 30 :=
by
  sorry

end NUMINAMATH_GPT_fewest_number_of_gymnasts_l90_9066


namespace NUMINAMATH_GPT_linear_function_difference_l90_9001

noncomputable def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

theorem linear_function_difference (f : ℝ → ℝ) 
  (h_linear : linear_function f)
  (h_cond1 : f 10 - f 5 = 20)
  (h_cond2 : f 0 = 3) :
  f 15 - f 5 = 40 :=
sorry

end NUMINAMATH_GPT_linear_function_difference_l90_9001


namespace NUMINAMATH_GPT_sugar_percentage_is_7_5_l90_9028

theorem sugar_percentage_is_7_5 
  (V1 : ℕ := 340)
  (p_water : ℝ := 88/100)
  (p_kola : ℝ := 5/100)
  (p_sugar : ℝ := 7/100)
  (V_sugar_add : ℝ := 3.2)
  (V_water_add : ℝ := 10)
  (V_kola_add : ℝ := 6.8) : 
  (
    (23.8 + 3.2) / (340 + 3.2 + 10 + 6.8) * 100 = 7.5
  ) :=
  by
  sorry

end NUMINAMATH_GPT_sugar_percentage_is_7_5_l90_9028


namespace NUMINAMATH_GPT_width_of_margin_l90_9006

-- Given conditions as definitions
def total_area : ℝ := 20 * 30
def percentage_used : ℝ := 0.64
def used_area : ℝ := percentage_used * total_area

-- Definition of the width of the typing area
def width_after_margin (x : ℝ) : ℝ := 20 - 2 * x

-- Definition of the length after top and bottom margins
def length_after_margin : ℝ := 30 - 6

-- Calculate the area used considering the margins
def typing_area (x : ℝ) : ℝ := (width_after_margin x) * length_after_margin

-- Statement to prove
theorem width_of_margin : ∃ x : ℝ, typing_area x = used_area ∧ x = 2 := by
  -- We give the prompt to eventually prove the theorem with the correct value
  sorry

end NUMINAMATH_GPT_width_of_margin_l90_9006


namespace NUMINAMATH_GPT_rectangle_perimeter_at_least_l90_9074

theorem rectangle_perimeter_at_least (m : ℕ) (m_pos : 0 < m) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a * b ≥ 1 / (m * m) ∧ 2 * (a + b) ≥ 4 / m) := sorry

end NUMINAMATH_GPT_rectangle_perimeter_at_least_l90_9074


namespace NUMINAMATH_GPT_sticks_form_triangle_l90_9013

theorem sticks_form_triangle:
  (2 + 3 > 4) ∧ (2 + 4 > 3) ∧ (3 + 4 > 2) := by
  sorry

end NUMINAMATH_GPT_sticks_form_triangle_l90_9013


namespace NUMINAMATH_GPT_marc_trip_equation_l90_9018

theorem marc_trip_equation (t : ℝ) 
  (before_stop_speed : ℝ := 90)
  (stop_time : ℝ := 0.5)
  (after_stop_speed : ℝ := 110)
  (total_distance : ℝ := 300)
  (total_trip_time : ℝ := 3.5) :
  before_stop_speed * t + after_stop_speed * (total_trip_time - stop_time - t) = total_distance :=
by 
  sorry

end NUMINAMATH_GPT_marc_trip_equation_l90_9018


namespace NUMINAMATH_GPT_max_true_statements_l90_9097

theorem max_true_statements {p q : ℝ} (hp : p > 0) (hq : q < 0) :
  ∀ (s1 s2 s3 s4 s5 : Prop), 
  s1 = (1 / p > 1 / q) →
  s2 = (p^3 > q^3) →
  s3 = (p^2 < q^2) →
  s4 = (p > 0) →
  s5 = (q < 0) →
  s1 ∧ s2 ∧ s4 ∧ s5 ∧ ¬s3 → 
  ∃ m : ℕ, m = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_true_statements_l90_9097


namespace NUMINAMATH_GPT_minimum_f_value_g_ge_f_implies_a_ge_4_l90_9098

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem minimum_f_value : (∃ x : ℝ, f x = 2 / Real.exp 1) :=
  sorry

theorem g_ge_f_implies_a_ge_4 (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ g x a) → a ≥ 4 :=
  sorry

end NUMINAMATH_GPT_minimum_f_value_g_ge_f_implies_a_ge_4_l90_9098


namespace NUMINAMATH_GPT_pyramid_base_length_l90_9055

theorem pyramid_base_length (A s h : ℝ): A = 120 ∧ h = 40 ∧ (A = 1/2 * s * h) → s = 6 := 
by
  sorry

end NUMINAMATH_GPT_pyramid_base_length_l90_9055


namespace NUMINAMATH_GPT_intersection_complement_l90_9076

universe u

def U := Real

def M : Set Real := { x | -2 ≤ x ∧ x ≤ 2 }

def N : Set Real := { x | x * (x - 3) ≤ 0 }

def complement_U (S : Set Real) : Set Real := { x | x ∉ S }

theorem intersection_complement :
  M ∩ (complement_U N) = { x | -2 ≤ x ∧ x < 0 } := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l90_9076


namespace NUMINAMATH_GPT_find_value_of_c_l90_9088

noncomputable def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem find_value_of_c (b c : ℝ) 
    (h1 : parabola b c 1 = 2)
    (h2 : parabola b c 5 = 2) :
    c = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_c_l90_9088


namespace NUMINAMATH_GPT_bet_final_result_l90_9003

theorem bet_final_result :
  let M₀ := 64
  let final_money := (3 / 2) ^ 3 * (1 / 2) ^ 3 * M₀
  final_money = 27 ∧ M₀ - final_money = 37 :=
by
  sorry

end NUMINAMATH_GPT_bet_final_result_l90_9003


namespace NUMINAMATH_GPT_abs_eq_inequality_l90_9049

theorem abs_eq_inequality (m : ℝ) (h : |m - 9| = 9 - m) : m ≤ 9 :=
sorry

end NUMINAMATH_GPT_abs_eq_inequality_l90_9049


namespace NUMINAMATH_GPT_dart_within_triangle_probability_l90_9035

theorem dart_within_triangle_probability (s : ℝ) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  (triangle_area / hexagon_area) = 1 / 24 :=
by sorry

end NUMINAMATH_GPT_dart_within_triangle_probability_l90_9035


namespace NUMINAMATH_GPT_transform_polynomial_l90_9044

theorem transform_polynomial (x y : ℝ) 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 - x^3 - 2 * x^2 - x + 1 = 0) : x^2 * (y^2 - y - 4) = 0 :=
sorry

end NUMINAMATH_GPT_transform_polynomial_l90_9044


namespace NUMINAMATH_GPT_diophantine_infinite_solutions_l90_9075

theorem diophantine_infinite_solutions
  (l m n : ℕ) (h_l_positive : l > 0) (h_m_positive : m > 0) (h_n_positive : n > 0)
  (h_gcd_lm_n : gcd (l * m) n = 1) (h_gcd_ln_m : gcd (l * n) m = 1) (h_gcd_mn_l : gcd (m * n) l = 1)
  : ∃ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0 ∧ (x ^ l + y ^ m = z ^ n)) ∧ (∀ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ (a ^ l + b ^ m = c ^ n)) → ∀ d : ℕ, d > 0 → ∃ e f g : ℕ, (e > 0 ∧ f > 0 ∧ g > 0 ∧ (e ^ l + f ^ m = g ^ n))) :=
sorry

end NUMINAMATH_GPT_diophantine_infinite_solutions_l90_9075


namespace NUMINAMATH_GPT_remainder_poly_l90_9024

theorem remainder_poly (x : ℂ) (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) :
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 :=
by sorry

end NUMINAMATH_GPT_remainder_poly_l90_9024


namespace NUMINAMATH_GPT_desks_increase_l90_9046

theorem desks_increase 
  (rows : ℕ) (first_row_desks : ℕ) (total_desks : ℕ) 
  (d : ℕ) 
  (h_rows : rows = 8) 
  (h_first_row : first_row_desks = 10) 
  (h_total_desks : total_desks = 136)
  (h_desks_sum : 10 + (10 + d) + (10 + 2 * d) + (10 + 3 * d) + (10 + 4 * d) + (10 + 5 * d) + (10 + 6 * d) + (10 + 7 * d) = total_desks) : 
  d = 2 := 
by 
  sorry

end NUMINAMATH_GPT_desks_increase_l90_9046


namespace NUMINAMATH_GPT_point_in_third_quadrant_l90_9000

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := 
sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l90_9000


namespace NUMINAMATH_GPT_specified_time_eq_l90_9083

noncomputable def slow_horse_days (x : ℝ) := x + 1
noncomputable def fast_horse_days (x : ℝ) := x - 3

theorem specified_time_eq (x : ℝ) (h1 : slow_horse_days x > 0) (h2 : fast_horse_days x > 0) :
  (900 / slow_horse_days x) * 2 = 900 / fast_horse_days x :=
by
  sorry

end NUMINAMATH_GPT_specified_time_eq_l90_9083


namespace NUMINAMATH_GPT_find_unique_function_l90_9021

theorem find_unique_function (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_unique_function_l90_9021


namespace NUMINAMATH_GPT_derivative_of_log_base_3_derivative_of_exp_base_2_l90_9042

noncomputable def log_base_3_deriv (x : ℝ) : ℝ := (Real.log x / Real.log 3)
noncomputable def exp_base_2_deriv (x : ℝ) : ℝ := Real.exp (x * Real.log 2)

theorem derivative_of_log_base_3 (x : ℝ) (h : x > 0) :
  (log_base_3_deriv x) = (1 / (x * Real.log 3)) :=
by
  sorry

theorem derivative_of_exp_base_2 (x : ℝ) :
  (exp_base_2_deriv x) = (Real.exp (x * Real.log 2) * Real.log 2) :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_log_base_3_derivative_of_exp_base_2_l90_9042


namespace NUMINAMATH_GPT_germs_left_percentage_l90_9085

-- Defining the conditions
def first_spray_kill_percentage : ℝ := 0.50
def second_spray_kill_percentage : ℝ := 0.25
def overlap_percentage : ℝ := 0.05
def total_kill_percentage : ℝ := first_spray_kill_percentage + second_spray_kill_percentage - overlap_percentage

-- The statement to be proved
theorem germs_left_percentage :
  1 - total_kill_percentage = 0.30 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_germs_left_percentage_l90_9085


namespace NUMINAMATH_GPT_expression_value_l90_9094

theorem expression_value (a b : ℚ) (h : a + 2 * b = 0) : 
  abs (a / |b| - 1) + abs (|a| / b - 2) + abs (|a / b| - 3) = 4 :=
sorry

end NUMINAMATH_GPT_expression_value_l90_9094


namespace NUMINAMATH_GPT_circle_equation_through_origin_l90_9012

theorem circle_equation_through_origin (focus : ℝ × ℝ) (radius : ℝ) (x y : ℝ) 
  (h1 : focus = (1, 0)) 
  (h2 : (x - 1)^2 + y^2 = radius^2) : 
  x^2 + y^2 - 2*x = 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_through_origin_l90_9012


namespace NUMINAMATH_GPT_solve_for_x_l90_9051

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  (4 * y^2 + y + 6 = 3 * (9 * x^2 + y + 3)) ↔ (x = 1 ∨ x = -1/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l90_9051


namespace NUMINAMATH_GPT_find_x_l90_9040

theorem find_x (x y z : ℝ) (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 :=
sorry

end NUMINAMATH_GPT_find_x_l90_9040


namespace NUMINAMATH_GPT_product_neg_six_l90_9086

theorem product_neg_six (m b : ℝ)
  (h1 : m = 2)
  (h2 : b = -3) : m * b < -3 := by
-- Proof skipped
sorry

end NUMINAMATH_GPT_product_neg_six_l90_9086


namespace NUMINAMATH_GPT_joan_total_cost_is_correct_l90_9015

def year1_home_games := 6
def year1_away_games := 3
def year1_home_playoff_games := 1
def year1_away_playoff_games := 1

def year2_home_games := 2
def year2_away_games := 2
def year2_home_playoff_games := 1
def year2_away_playoff_games := 0

def home_game_ticket := 60
def away_game_ticket := 75
def home_playoff_ticket := 120
def away_playoff_ticket := 100

def friend_home_game_ticket := 45
def friend_away_game_ticket := 75

def home_game_transportation := 25
def away_game_transportation := 50

noncomputable def year1_total_cost : ℕ :=
  (year1_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year1_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def year2_total_cost : ℕ :=
  (year2_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year2_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def total_cost : ℕ := year1_total_cost + year2_total_cost

theorem joan_total_cost_is_correct : total_cost = 2645 := by
  sorry

end NUMINAMATH_GPT_joan_total_cost_is_correct_l90_9015


namespace NUMINAMATH_GPT_bug_visits_exactly_16_pavers_l90_9077

-- Defining the dimensions of the garden and the pavers
def garden_width : ℕ := 14
def garden_length : ℕ := 19
def paver_size : ℕ := 2

-- Calculating the number of pavers in width and length
def pavers_width : ℕ := garden_width / paver_size
def pavers_length : ℕ := (garden_length + paver_size - 1) / paver_size  -- Taking ceiling of 19/2

-- Calculating the GCD of the pavers count in width and length
def gcd_pavers : ℕ := Nat.gcd pavers_width pavers_length

-- Calculating the number of pavers the bug crosses
def pavers_crossed : ℕ := pavers_width + pavers_length - gcd_pavers

-- Theorem that states the number of pavers visited
theorem bug_visits_exactly_16_pavers :
  pavers_crossed = 16 := by
  -- Sorry is used to skip the proof steps
  sorry

end NUMINAMATH_GPT_bug_visits_exactly_16_pavers_l90_9077


namespace NUMINAMATH_GPT_x_value_l90_9064

def x_is_75_percent_greater (x : ℝ) (y : ℝ) : Prop := x = y + 0.75 * y

theorem x_value (x : ℝ) : x_is_75_percent_greater x 150 → x = 262.5 :=
by
  intro h
  rw [x_is_75_percent_greater] at h
  sorry

end NUMINAMATH_GPT_x_value_l90_9064


namespace NUMINAMATH_GPT_zero_point_in_interval_l90_9023

noncomputable def f (x a : ℝ) := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : 
  (∃ x, 1 < x ∧ x < 2 ∧ f x a = 0) → 0 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_zero_point_in_interval_l90_9023


namespace NUMINAMATH_GPT_line_in_slope_intercept_form_l90_9020

-- Given the condition
def line_eq (x y : ℝ) : Prop :=
  (2 * (x - 3)) - (y + 4) = 0

-- Prove that the line equation can be expressed as y = 2x - 10.
theorem line_in_slope_intercept_form (x y : ℝ) :
  line_eq x y ↔ y = 2 * x - 10 := 
sorry

end NUMINAMATH_GPT_line_in_slope_intercept_form_l90_9020


namespace NUMINAMATH_GPT_complement_union_l90_9005

theorem complement_union (U A B complement_U_A : Set Int) (hU : U = {-1, 0, 1, 2}) 
  (hA : A = {-1, 2}) (hB : B = {0, 2}) (hC : complement_U_A = {0, 1}) :
  complement_U_A ∪ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_complement_union_l90_9005


namespace NUMINAMATH_GPT_least_number_subtracted_divisible_by_17_and_23_l90_9091

-- Conditions
def is_divisible_by_17_and_23 (n : ℕ) : Prop := 
  n % 17 = 0 ∧ n % 23 = 0

def target_number : ℕ := 7538

-- The least number to be subtracted
noncomputable def least_number_to_subtract : ℕ := 109

-- Theorem statement
theorem least_number_subtracted_divisible_by_17_and_23 : 
  is_divisible_by_17_and_23 (target_number - least_number_to_subtract) :=
by 
  -- Proof details would normally follow here.
  sorry

end NUMINAMATH_GPT_least_number_subtracted_divisible_by_17_and_23_l90_9091


namespace NUMINAMATH_GPT_find_f_at_6_l90_9073

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2

theorem find_f_at_6 (f : ℝ → ℝ) (h : example_function f) : f 6 = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_f_at_6_l90_9073


namespace NUMINAMATH_GPT_gcd_108_45_l90_9010

theorem gcd_108_45 :
  ∃ g, g = Nat.gcd 108 45 ∧ g = 9 :=
by
  sorry

end NUMINAMATH_GPT_gcd_108_45_l90_9010


namespace NUMINAMATH_GPT_car_trip_cost_proof_l90_9080

def car_trip_cost 
  (d1 d2 d3 d4 : ℕ) 
  (efficiency : ℕ) 
  (cost_per_gallon : ℕ) 
  (total_distance : ℕ) 
  (gallons_used : ℕ) 
  (cost : ℕ) : Prop :=
  d1 = 8 ∧
  d2 = 6 ∧
  d3 = 12 ∧
  d4 = 2 * d3 ∧
  efficiency = 25 ∧
  cost_per_gallon = 250 ∧
  total_distance = d1 + d2 + d3 + d4 ∧
  gallons_used = total_distance / efficiency ∧
  cost = gallons_used * cost_per_gallon ∧
  cost = 500

theorem car_trip_cost_proof : car_trip_cost 8 6 12 (2 * 12) 25 250 (8 + 6 + 12 + (2 * 12)) ((8 + 6 + 12 + (2 * 12)) / 25) (((8 + 6 + 12 + (2 * 12)) / 25) * 250) :=
by 
  sorry

end NUMINAMATH_GPT_car_trip_cost_proof_l90_9080


namespace NUMINAMATH_GPT_range_of_x_satisfying_inequality_l90_9002

def f (x : ℝ) : ℝ := -- Define the function f (we will leave this definition open for now)
sorry
@[continuity] axiom f_increasing (x y : ℝ) (h : x < y) : f x < f y
axiom f_2_eq_1 : f 2 = 1
axiom f_xy_eq_f_x_add_f_y (x y : ℝ) : f (x * y) = f x + f y

noncomputable def f_4_eq_2 : f 4 = 2 := sorry

theorem range_of_x_satisfying_inequality (x : ℝ) :
  3 < x ∧ x ≤ 4 ↔ f x + f (x - 3) ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_satisfying_inequality_l90_9002


namespace NUMINAMATH_GPT_betty_red_beads_l90_9037

theorem betty_red_beads (r b : ℕ) (h_ratio : r / b = 3 / 2) (h_blue_beads : b = 20) : r = 30 :=
by
  sorry

end NUMINAMATH_GPT_betty_red_beads_l90_9037


namespace NUMINAMATH_GPT_yellow_ball_kids_l90_9036

theorem yellow_ball_kids (total_kids white_ball_kids both_ball_kids : ℕ) :
  total_kids = 35 → white_ball_kids = 26 → both_ball_kids = 19 → 
  (total_kids = white_ball_kids + (total_kids - both_ball_kids)) → 
  (total_kids - (white_ball_kids - both_ball_kids)) = 28 :=
by
  sorry

end NUMINAMATH_GPT_yellow_ball_kids_l90_9036


namespace NUMINAMATH_GPT_candy_last_days_l90_9019

def pieces_from_neighbors : ℝ := 11.0
def pieces_from_sister : ℝ := 5.0
def pieces_per_day : ℝ := 8.0
def total_pieces : ℝ := pieces_from_neighbors + pieces_from_sister

theorem candy_last_days : total_pieces / pieces_per_day = 2 := by
    sorry

end NUMINAMATH_GPT_candy_last_days_l90_9019


namespace NUMINAMATH_GPT_loaves_of_bread_l90_9084

-- Definitions for the given conditions
def total_flour : ℝ := 5
def flour_per_loaf : ℝ := 2.5

-- The statement of the problem
theorem loaves_of_bread (total_flour : ℝ) (flour_per_loaf : ℝ) : 
  total_flour / flour_per_loaf = 2 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_loaves_of_bread_l90_9084


namespace NUMINAMATH_GPT_more_likely_millionaire_city_resident_l90_9043

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end NUMINAMATH_GPT_more_likely_millionaire_city_resident_l90_9043


namespace NUMINAMATH_GPT_a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l90_9017

variable (a0 a1 a2 a3 a4 a5 : ℝ)

noncomputable def polynomial (x : ℝ) : ℝ :=
  a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5

theorem a3_is_neg_10 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a3 = -10 :=
sorry

theorem a1_a3_a5_sum_is_neg_16 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a1 + a3 + a5 = -16 :=
sorry

end NUMINAMATH_GPT_a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l90_9017


namespace NUMINAMATH_GPT_min_value_S_max_value_S_l90_9004

theorem min_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≥ -512 := 
sorry

theorem max_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288 := 
sorry

end NUMINAMATH_GPT_min_value_S_max_value_S_l90_9004


namespace NUMINAMATH_GPT_part_I_part_II_l90_9087

def f (x a : ℝ) := |x - a| + |x - 1|

theorem part_I {x : ℝ} : Set.Icc 0 4 = {y | f y 3 ≤ 4} := 
sorry

theorem part_II {a : ℝ} : (∀ x, ¬ (f x a < 2)) ↔ a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l90_9087


namespace NUMINAMATH_GPT_find_speed_of_boat_l90_9027

theorem find_speed_of_boat (r d t : ℝ) (x : ℝ) (h_rate : r = 4) (h_dist : d = 33.733333333333334) (h_time : t = 44 / 60) 
  (h_eq : d = (x + r) * t) : x = 42.09090909090909 :=
  sorry

end NUMINAMATH_GPT_find_speed_of_boat_l90_9027


namespace NUMINAMATH_GPT_estimated_height_is_644_l90_9053

noncomputable def height_of_second_building : ℝ := 100
noncomputable def height_of_first_building : ℝ := 0.8 * height_of_second_building
noncomputable def height_of_third_building : ℝ := (height_of_first_building + height_of_second_building) - 20
noncomputable def height_of_fourth_building : ℝ := 1.15 * height_of_third_building
noncomputable def height_of_fifth_building : ℝ := 2 * |height_of_second_building - height_of_third_building|
noncomputable def total_estimated_height : ℝ := height_of_first_building + height_of_second_building + height_of_third_building + height_of_fourth_building + height_of_fifth_building

theorem estimated_height_is_644 : total_estimated_height = 644 := by
  sorry

end NUMINAMATH_GPT_estimated_height_is_644_l90_9053


namespace NUMINAMATH_GPT_one_set_working_communication_possible_l90_9082

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

end NUMINAMATH_GPT_one_set_working_communication_possible_l90_9082


namespace NUMINAMATH_GPT_find_a1_l90_9011

noncomputable def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a (n+1) + a n = 4*n

theorem find_a1 (a : ℕ → ℕ) (h : is_arithmetic_sequence a) : a 1 = 1 := by
  sorry

end NUMINAMATH_GPT_find_a1_l90_9011


namespace NUMINAMATH_GPT_negation_of_exists_x_quad_eq_zero_l90_9014

theorem negation_of_exists_x_quad_eq_zero :
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0 ↔ ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_x_quad_eq_zero_l90_9014


namespace NUMINAMATH_GPT_dave_paid_more_l90_9078

-- Definitions based on conditions in the problem statement
def total_pizza_cost : ℕ := 11  -- Total cost of the pizza in dollars
def num_slices : ℕ := 8  -- Total number of slices in the pizza
def plain_pizza_cost : ℕ := 8  -- Cost of the plain pizza in dollars
def anchovies_cost : ℕ := 2  -- Extra cost of adding anchovies in dollars
def mushrooms_cost : ℕ := 1  -- Extra cost of adding mushrooms in dollars
def dave_slices : ℕ := 7  -- Number of slices Dave ate
def doug_slices : ℕ := 1  -- Number of slices Doug ate
def doug_payment : ℕ := 1  -- Amount Doug paid in dollars
def dave_payment : ℕ := total_pizza_cost - doug_payment  -- Amount Dave paid in dollars

-- Prove that Dave paid 9 dollars more than Doug
theorem dave_paid_more : dave_payment - doug_payment = 9 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_dave_paid_more_l90_9078


namespace NUMINAMATH_GPT_area_of_L_shape_l90_9056

theorem area_of_L_shape (a : ℝ) (h_pos : a > 0) (h_eq : 4 * ((a + 3)^2 - a^2) = 5 * a^2) : 
  (a + 3)^2 - a^2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_area_of_L_shape_l90_9056


namespace NUMINAMATH_GPT_three_points_in_circle_of_radius_one_seventh_l90_9052

-- Define the problem
theorem three_points_in_circle_of_radius_one_seventh (P : Fin 51 → ℝ × ℝ) :
  (∀ i, 0 ≤ (P i).1 ∧ (P i).1 ≤ 1 ∧ 0 ≤ (P i).2 ∧ (P i).2 ≤ 1) →
  ∃ (i j k : Fin 51), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    dist (P i) (P j) ≤ 2/7 ∧ dist (P j) (P k) ≤ 2/7 ∧ dist (P k) (P i) ≤ 2/7 :=
by
  sorry

end NUMINAMATH_GPT_three_points_in_circle_of_radius_one_seventh_l90_9052


namespace NUMINAMATH_GPT_perimeter_of_square_l90_9045

theorem perimeter_of_square
  (length_rect : ℕ) (width_rect : ℕ) (area_rect : ℕ)
  (area_square : ℕ) (side_square : ℕ) (perimeter_square : ℕ) :
  (length_rect = 32) → (width_rect = 10) → 
  (area_rect = length_rect * width_rect) →
  (area_square = 5 * area_rect) →
  (side_square * side_square = area_square) →
  (perimeter_square = 4 * side_square) →
  perimeter_square = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l90_9045


namespace NUMINAMATH_GPT_mary_initial_sugar_eq_4_l90_9057

/-- Mary is baking a cake. The recipe calls for 7 cups of sugar and she needs to add 3 more cups of sugar. -/
def total_sugar : ℕ := 7
def additional_sugar : ℕ := 3

theorem mary_initial_sugar_eq_4 :
  ∃ initial_sugar : ℕ, initial_sugar + additional_sugar = total_sugar ∧ initial_sugar = 4 :=
sorry

end NUMINAMATH_GPT_mary_initial_sugar_eq_4_l90_9057


namespace NUMINAMATH_GPT_gcd_euclidean_120_168_gcd_subtraction_459_357_l90_9034

theorem gcd_euclidean_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

theorem gcd_subtraction_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_GPT_gcd_euclidean_120_168_gcd_subtraction_459_357_l90_9034


namespace NUMINAMATH_GPT_circle_tangent_sum_radii_l90_9067

theorem circle_tangent_sum_radii :
  let r1 := 6 + 2 * Real.sqrt 6
  let r2 := 6 - 2 * Real.sqrt 6
  r1 + r2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_sum_radii_l90_9067


namespace NUMINAMATH_GPT_age_of_youngest_child_l90_9060

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60) : 
  x = 6 :=
sorry

end NUMINAMATH_GPT_age_of_youngest_child_l90_9060


namespace NUMINAMATH_GPT_batches_engine_count_l90_9025

theorem batches_engine_count (x : ℕ) 
  (h1 : ∀ e, 1/4 * e = 0) -- every batch has engines, no proof needed for this question
  (h2 : 5 * (3/4 : ℚ) * x = 300) : 
  x = 80 := 
sorry

end NUMINAMATH_GPT_batches_engine_count_l90_9025


namespace NUMINAMATH_GPT_back_seat_can_hold_8_people_l90_9092

def totalPeopleOnSides : ℕ :=
  let left_seats := 15
  let right_seats := left_seats - 3
  let people_per_seat := 3
  (left_seats + right_seats) * people_per_seat

def bus_total_capacity : ℕ := 89

def back_seat_capacity : ℕ :=
  bus_total_capacity - totalPeopleOnSides

theorem back_seat_can_hold_8_people : back_seat_capacity = 8 := by
  sorry

end NUMINAMATH_GPT_back_seat_can_hold_8_people_l90_9092


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l90_9016

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_l90_9016


namespace NUMINAMATH_GPT_sum_coordinates_l90_9061

variables (x y : ℝ)
def A_coord := (9, 3)
def M_coord := (3, 7)

def midpoint_condition_x : Prop := (x + 9) / 2 = 3
def midpoint_condition_y : Prop := (y + 3) / 2 = 7

theorem sum_coordinates (h1 : midpoint_condition_x x) (h2 : midpoint_condition_y y) : 
  x + y = 8 :=
by 
  sorry

end NUMINAMATH_GPT_sum_coordinates_l90_9061


namespace NUMINAMATH_GPT_exists_subset_sum_divisible_by_2n_l90_9089

open BigOperators

theorem exists_subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) (a : Fin n → ℤ)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_interval : ∀ i : Fin n, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % (2 * n) = 0 :=
sorry

end NUMINAMATH_GPT_exists_subset_sum_divisible_by_2n_l90_9089


namespace NUMINAMATH_GPT_tan_alpha_value_l90_9062

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (Real.pi + α) = 3 / 5) 
  (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan α = 3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_alpha_value_l90_9062


namespace NUMINAMATH_GPT_combinedHeightOfBuildingsIsCorrect_l90_9026

-- Define the heights to the top floor of the buildings (in feet)
def empireStateBuildingHeightFeet : Float := 1250
def willisTowerHeightFeet : Float := 1450
def oneWorldTradeCenterHeightFeet : Float := 1368

-- Define the antenna heights of the buildings (in feet)
def empireStateBuildingAntennaFeet : Float := 204
def willisTowerAntennaFeet : Float := 280
def oneWorldTradeCenterAntennaFeet : Float := 408

-- Define the conversion factor from feet to meters
def feetToMeters : Float := 0.3048

-- Calculate the total heights of the buildings in meters
def empireStateBuildingTotalHeightMeters : Float := (empireStateBuildingHeightFeet + empireStateBuildingAntennaFeet) * feetToMeters
def willisTowerTotalHeightMeters : Float := (willisTowerHeightFeet + willisTowerAntennaFeet) * feetToMeters
def oneWorldTradeCenterTotalHeightMeters : Float := (oneWorldTradeCenterHeightFeet + oneWorldTradeCenterAntennaFeet) * feetToMeters

-- Calculate the combined total height of the three buildings in meters
def combinedTotalHeightMeters : Float :=
  empireStateBuildingTotalHeightMeters + willisTowerTotalHeightMeters + oneWorldTradeCenterTotalHeightMeters

-- The statement to prove
theorem combinedHeightOfBuildingsIsCorrect : combinedTotalHeightMeters = 1511.8164 := by
  sorry

end NUMINAMATH_GPT_combinedHeightOfBuildingsIsCorrect_l90_9026


namespace NUMINAMATH_GPT_boys_on_trip_l90_9068

theorem boys_on_trip (B G : ℕ) 
    (h1 : G = B + (2 / 5 : ℚ) * B) 
    (h2 : 1 + 1 + 1 + B + G = 123) : 
    B = 50 := 
by 
  -- Proof skipped 
  sorry

end NUMINAMATH_GPT_boys_on_trip_l90_9068


namespace NUMINAMATH_GPT_right_triangles_count_l90_9038

theorem right_triangles_count (b a : ℕ) (h₁: b < 150) (h₂: (a^2 + b^2 = (b + 2)^2)) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 12 ∧ b = n^2 - 1 :=
by
  -- This intended to state the desired number and form of the right triangles.
  sorry

def count_right_triangles : ℕ :=
  12 -- Result as a constant based on proof steps

#eval count_right_triangles -- Should output 12

end NUMINAMATH_GPT_right_triangles_count_l90_9038


namespace NUMINAMATH_GPT_positive_roots_implies_nonnegative_m_l90_9022

variables {x1 x2 m : ℝ}

theorem positive_roots_implies_nonnegative_m (h1 : x1 > 0) (h2 : x2 > 0)
  (h3 : x1 * x2 = 1) (h4 : x1 + x2 = m + 2) : m ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_positive_roots_implies_nonnegative_m_l90_9022


namespace NUMINAMATH_GPT_unique_positive_integers_pqr_l90_9081

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 61) / 2 + 5 / 2)

lemma problem_condition (p q r : ℕ) (py : ℝ) :
  py = y^100
  ∧ py = 2 * (y^98)
  ∧ py = 16 * (y^96)
  ∧ py = 13 * (y^94)
  ∧ py = - y^50
  ∧ py = ↑p * y^46
  ∧ py = ↑q * y^44
  ∧ py = ↑r * y^40 :=
sorry

theorem unique_positive_integers_pqr : 
  ∃! (p q r : ℕ), 
    p = 37 ∧ q = 47 ∧ r = 298 ∧ 
    y^100 = 2 * y^98 + 16 * y^96 + 13 * y^94 - y^50 + ↑p * y^46 + ↑q * y^44 + ↑r * y^40 :=
sorry

end NUMINAMATH_GPT_unique_positive_integers_pqr_l90_9081


namespace NUMINAMATH_GPT_brogan_total_red_apples_l90_9047

def red_apples (total_apples percentage_red : ℕ) : ℕ :=
  (total_apples * percentage_red) / 100

theorem brogan_total_red_apples :
  red_apples 20 40 + red_apples 20 50 = 18 :=
by
  sorry

end NUMINAMATH_GPT_brogan_total_red_apples_l90_9047


namespace NUMINAMATH_GPT_bicycle_cost_correct_l90_9030

def pay_rate : ℕ := 5
def hours_p_week : ℕ := 2 + 1 + 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

theorem bicycle_cost_correct :
  pay_rate * hours_p_week * weeks = bicycle_cost :=
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_correct_l90_9030


namespace NUMINAMATH_GPT_initial_pile_counts_l90_9050

def pile_transfers (A B C : ℕ) : Prop :=
  (A + B + C = 48) ∧
  ∃ (A' B' C' : ℕ), 
    (A' = A + B) ∧ (B' = B + C) ∧ (C' = C + A) ∧
    (A' = 2 * 16) ∧ (B' = 2 * 12) ∧ (C' = 2 * 14)

theorem initial_pile_counts :
  ∃ A B C : ℕ, pile_transfers A B C ∧ A = 22 ∧ B = 14 ∧ C = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_pile_counts_l90_9050


namespace NUMINAMATH_GPT_find_a_l90_9008

-- Define the binomial coefficient function in Lean
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions and the proof problem statement
theorem find_a (a : ℝ) (h: (-a)^7 * binomial 10 7 = -120) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l90_9008


namespace NUMINAMATH_GPT_every_nat_as_diff_of_same_prime_divisors_l90_9007

-- Conditions
def prime_divisors (n : ℕ) : ℕ :=
  -- function to count the number of distinct prime divisors of n
  sorry

-- Tuple translation
theorem every_nat_as_diff_of_same_prime_divisors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ prime_divisors a = prime_divisors b := 
by
  sorry

end NUMINAMATH_GPT_every_nat_as_diff_of_same_prime_divisors_l90_9007


namespace NUMINAMATH_GPT_question_1_question_2_l90_9071

def curve_is_ellipse (m : ℝ) : Prop :=
  (3 - m > 0) ∧ (m - 1 > 0) ∧ (3 - m > m - 1)

def domain_is_R (m : ℝ) : Prop :=
  m^2 < (9 / 4)

theorem question_1 (m : ℝ) :
  curve_is_ellipse m → 1 < m ∧ m < 2 :=
sorry

theorem question_2 (m : ℝ) :
  (curve_is_ellipse m ∧ domain_is_R m) → 1 < m ∧ m < (3 / 2) :=
sorry

end NUMINAMATH_GPT_question_1_question_2_l90_9071


namespace NUMINAMATH_GPT_total_fault_line_movement_l90_9054

-- Define the movements in specific years.
def movement_past_year : ℝ := 1.25
def movement_year_before : ℝ := 5.25

-- Theorem stating the total movement of the fault line over the two years.
theorem total_fault_line_movement : movement_past_year + movement_year_before = 6.50 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_total_fault_line_movement_l90_9054


namespace NUMINAMATH_GPT_James_total_passengers_l90_9059

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end NUMINAMATH_GPT_James_total_passengers_l90_9059


namespace NUMINAMATH_GPT_correct_judgment_l90_9063

open Real

def period_sin2x (T : ℝ) : Prop := ∀ x, sin (2 * x) = sin (2 * (x + T))
def smallest_positive_period_sin2x : Prop := ∃ T > 0, period_sin2x T ∧ ∀ T' > 0, period_sin2x T' → T ≤ T'
def smallest_positive_period_sin2x_is_pi : Prop := ∃ T, smallest_positive_period_sin2x ∧ T = π

def symmetry_cosx (L : ℝ) : Prop := ∀ x, cos (L - x) = cos (L + x)
def symmetry_about_line_cosx (L : ℝ) : Prop := L = π / 2

def p : Prop := smallest_positive_period_sin2x_is_pi
def q : Prop := symmetry_about_line_cosx (π / 2)

theorem correct_judgment : ¬ (p ∧ q) :=
by 
  sorry

end NUMINAMATH_GPT_correct_judgment_l90_9063


namespace NUMINAMATH_GPT_initial_men_count_l90_9095

theorem initial_men_count (M : ℕ) (F : ℕ) (h1 : F = M * 22) (h2 : (M + 2280) * 5 = M * 20) : M = 760 := by
  sorry

end NUMINAMATH_GPT_initial_men_count_l90_9095


namespace NUMINAMATH_GPT_sum_lent_250_l90_9096

theorem sum_lent_250 (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (hR : R = 4) (hT : T = 8) (hSI1 : SI = P - 170) 
  (hSI2 : SI = (P * R * T) / 100) : 
  P = 250 := 
by 
  sorry

end NUMINAMATH_GPT_sum_lent_250_l90_9096


namespace NUMINAMATH_GPT_t_shirt_cost_l90_9079

theorem t_shirt_cost (total_amount_spent : ℝ) (number_of_t_shirts : ℕ) (cost_per_t_shirt : ℝ)
  (h0 : total_amount_spent = 201) 
  (h1 : number_of_t_shirts = 22)
  (h2 : cost_per_t_shirt = total_amount_spent / number_of_t_shirts) :
  cost_per_t_shirt = 9.14 := 
sorry

end NUMINAMATH_GPT_t_shirt_cost_l90_9079


namespace NUMINAMATH_GPT_circle_intersects_y_axis_at_one_l90_9048

theorem circle_intersects_y_axis_at_one :
  let A := (-2011, 0)
  let B := (2010, 0)
  let C := (0, (-2010) * 2011)
  ∃ (D : ℝ × ℝ), D = (0, 1) ∧
    (∃ O : ℝ × ℝ, O = (0, 0) ∧
    (dist O A) * (dist O B) = (dist O C) * (dist O D)) :=
by
  sorry -- Proof of the theorem

end NUMINAMATH_GPT_circle_intersects_y_axis_at_one_l90_9048


namespace NUMINAMATH_GPT_wendy_created_albums_l90_9039

theorem wendy_created_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums : ℕ) :
  phone_pics = 22 → camera_pics = 2 → pics_per_album = 6 → total_pics = phone_pics + camera_pics → albums = total_pics / pics_per_album → albums = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_wendy_created_albums_l90_9039


namespace NUMINAMATH_GPT_binom_n_plus_1_n_minus_1_eq_l90_9041

theorem binom_n_plus_1_n_minus_1_eq (n : ℕ) (h : 0 < n) : (Nat.choose (n + 1) (n - 1)) = n * (n + 1) / 2 := 
by sorry

end NUMINAMATH_GPT_binom_n_plus_1_n_minus_1_eq_l90_9041


namespace NUMINAMATH_GPT_part1_part2_l90_9069

-- Define the universal set U as real numbers ℝ
def U : Set ℝ := Set.univ

-- Define Set A
def A (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1 }

-- Define Set B
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0 }

-- Part 1: Prove A ∪ B when a = 4
theorem part1 : A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} :=
sorry

-- Part 2: Prove the range of values for a given A ∩ B = A
theorem part2 (a : ℝ) (h : A a ∩ B = A a) : a ≥ 5 ∨ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_part1_part2_l90_9069


namespace NUMINAMATH_GPT_distinct_four_digit_odd_numbers_l90_9029

-- Define the conditions as Lean definitions
def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def valid_first_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

-- The proposition we want to prove
theorem distinct_four_digit_odd_numbers (n : ℕ) :
  (∀ d, d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → is_odd_digit d) →
  valid_first_digit (n / 1000 % 10) →
  1000 ≤ n ∧ n < 10000 →
  n = 500 :=
sorry

end NUMINAMATH_GPT_distinct_four_digit_odd_numbers_l90_9029


namespace NUMINAMATH_GPT_chess_champion_probability_l90_9032

theorem chess_champion_probability :
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  1000 * P = 343 :=
by 
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  show 1000 * P = 343
  sorry

end NUMINAMATH_GPT_chess_champion_probability_l90_9032


namespace NUMINAMATH_GPT_simplify_fraction_subtraction_l90_9090

theorem simplify_fraction_subtraction : (7 / 3) - (5 / 6) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_subtraction_l90_9090


namespace NUMINAMATH_GPT_heaviest_person_is_Vanya_l90_9072

variables (A D T V M : ℕ)

-- conditions
def condition1 : Prop := A + D = 82
def condition2 : Prop := D + T = 74
def condition3 : Prop := T + V = 75
def condition4 : Prop := V + M = 65
def condition5 : Prop := M + A = 62

theorem heaviest_person_is_Vanya (h1 : condition1 A D) (h2 : condition2 D T) (h3 : condition3 T V) (h4 : condition4 V M) (h5 : condition5 M A) :
  V = 43 :=
sorry

end NUMINAMATH_GPT_heaviest_person_is_Vanya_l90_9072


namespace NUMINAMATH_GPT_farmer_brown_leg_wing_count_l90_9033

theorem farmer_brown_leg_wing_count :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let pigeons := 4
  let kangaroos := 2
  
  let chicken_legs := 2
  let chicken_wings := 2
  let sheep_legs := 4
  let grasshopper_legs := 6
  let grasshopper_wings := 2
  let spider_legs := 8
  let pigeon_legs := 2
  let pigeon_wings := 2
  let kangaroo_legs := 2

  (chickens * (chicken_legs + chicken_wings) +
  sheep * sheep_legs +
  grasshoppers * (grasshopper_legs + grasshopper_wings) +
  spiders * spider_legs +
  pigeons * (pigeon_legs + pigeon_wings) +
  kangaroos * kangaroo_legs) = 172 := 
by
  sorry

end NUMINAMATH_GPT_farmer_brown_leg_wing_count_l90_9033


namespace NUMINAMATH_GPT_value_of_a_l90_9093

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l90_9093


namespace NUMINAMATH_GPT_ratio_of_millipedes_l90_9058

-- Define the given conditions
def total_segments_needed : ℕ := 800
def first_millipede_segments : ℕ := 60
def millipedes_segments (x : ℕ) : ℕ := x
def ten_millipedes_segments : ℕ := 10 * 50

-- State the main theorem
theorem ratio_of_millipedes (x : ℕ) : 
  total_segments_needed = 60 + 2 * x + 10 * 50 →
  2 * x / 60 = 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_millipedes_l90_9058


namespace NUMINAMATH_GPT_distance_B_amusement_park_l90_9009

variable (d_A d_B v_A v_B t_A t_B : ℝ)

axiom h1 : v_A = 3
axiom h2 : v_B = 4
axiom h3 : d_B = d_A + 2
axiom h4 : t_A + t_B = 4
axiom h5 : t_A = d_A / v_A
axiom h6 : t_B = d_B / v_B

theorem distance_B_amusement_park:
  d_A / 3 + (d_A + 2) / 4 = 4 → d_B = 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_B_amusement_park_l90_9009


namespace NUMINAMATH_GPT_max_ab_l90_9065

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 6) : ab ≤ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_l90_9065


namespace NUMINAMATH_GPT_train_crosses_platform_in_26_seconds_l90_9099

def km_per_hr_to_m_per_s (km_per_hr : ℕ) : ℕ :=
  km_per_hr * 5 / 18

def train_crossing_time
  (train_speed_km_per_hr : ℕ)
  (train_length_m : ℕ)
  (platform_length_m : ℕ) : ℕ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr
  total_distance_m / train_speed_m_per_s

theorem train_crosses_platform_in_26_seconds :
  train_crossing_time 72 300 220 = 26 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_in_26_seconds_l90_9099


namespace NUMINAMATH_GPT_remainder_b94_mod_55_eq_29_l90_9070

theorem remainder_b94_mod_55_eq_29 :
  (5^94 + 7^94) % 55 = 29 := 
by
  -- conditions: local definitions for bn, modulo, etc.
  sorry

end NUMINAMATH_GPT_remainder_b94_mod_55_eq_29_l90_9070


namespace NUMINAMATH_GPT_negation_of_universal_prop_correct_l90_9031

def negation_of_universal_prop : Prop :=
  ¬ (∀ x : ℝ, x = |x|) ↔ ∃ x : ℝ, x ≠ |x|

theorem negation_of_universal_prop_correct : negation_of_universal_prop := 
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_prop_correct_l90_9031
