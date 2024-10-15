import Mathlib

namespace NUMINAMATH_GPT_total_worth_of_stock_l2109_210993

theorem total_worth_of_stock (X : ℝ) (h1 : 0.1 * X * 1.2 - 0.9 * X * 0.95 = -400) : X = 16000 :=
by
  -- actual proof
  sorry

end NUMINAMATH_GPT_total_worth_of_stock_l2109_210993


namespace NUMINAMATH_GPT_total_assignments_for_28_points_l2109_210987

-- Definitions based on conditions
def assignments_needed (points : ℕ) : ℕ :=
  (points / 7 + 1) * (points % 7) + (points / 7) * (7 - points % 7)

-- The theorem statement, which asserts the answer to the given problem
theorem total_assignments_for_28_points : assignments_needed 28 = 70 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_total_assignments_for_28_points_l2109_210987


namespace NUMINAMATH_GPT_find_value_of_expression_l2109_210956

variable (α β : ℝ)

-- Defining the conditions
def is_root (α : ℝ) : Prop := α^2 - 3 * α + 1 = 0
def add_roots_eq (α β : ℝ) : Prop := α + β = 3
def mult_roots_eq (α β : ℝ) : Prop := α * β = 1

-- The main statement we want to prove
theorem find_value_of_expression {α β : ℝ} 
  (hα : is_root α) 
  (hβ : is_root β)
  (h_add : add_roots_eq α β)
  (h_mul : mult_roots_eq α β) :
  3 * α^5 + 7 * β^4 = 817 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l2109_210956


namespace NUMINAMATH_GPT_bus_ride_difference_l2109_210913

def oscars_bus_ride : ℝ := 0.75
def charlies_bus_ride : ℝ := 0.25

theorem bus_ride_difference :
  oscars_bus_ride - charlies_bus_ride = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_bus_ride_difference_l2109_210913


namespace NUMINAMATH_GPT_smallest_x_undefined_l2109_210997

theorem smallest_x_undefined :
  (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1 ∨ x = 8) → (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_undefined_l2109_210997


namespace NUMINAMATH_GPT_simple_interest_rate_l2109_210976

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (H : T = 4 ∧ (7 / 6) * P = P + (P * R * T / 100)) :
  R = 4.17 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2109_210976


namespace NUMINAMATH_GPT_minimum_dot_product_l2109_210959

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (1, 2, 0)
def B : ℝ × ℝ × ℝ := (0, 1, -1)

-- Define the vector AP
def vector_AP (x : ℝ) := (x - 1, -2, 0)

-- Define the vector BP
def vector_BP (x : ℝ) := (x, -1, 1)

-- Define the dot product of vector AP and vector BP
def dot_product (x : ℝ) : ℝ := (x - 1) * x + (-2) * (-1) + 0 * 1

-- State the theorem
theorem minimum_dot_product : ∃ x : ℝ, dot_product x = (x - 1) * x + 2 ∧ 
  (∀ y : ℝ, dot_product y ≥ dot_product (1/2)) := 
sorry

end NUMINAMATH_GPT_minimum_dot_product_l2109_210959


namespace NUMINAMATH_GPT_quadratic_function_passing_through_origin_l2109_210925

-- Define the quadratic function y
def quadratic_function (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

-- State the problem as a theorem
theorem quadratic_function_passing_through_origin (m : ℝ) (h: quadratic_function m 0 = 0) : m = -4 :=
by
  -- Since we only need the statement, we put sorry here
  sorry

end NUMINAMATH_GPT_quadratic_function_passing_through_origin_l2109_210925


namespace NUMINAMATH_GPT_function_order_l2109_210969

theorem function_order (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (2 - x) = f x)
  (h2 : ∀ x : ℝ, f (x + 2) = f (x - 2))
  (h3 : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 ∧ 1 ≤ x2 ∧ x2 ≤ 3 → (f x1 - f x2) / (x1 - x2) < 0) :
  f 2016 = f 2014 ∧ f 2014 > f 2015 :=
by
  sorry

end NUMINAMATH_GPT_function_order_l2109_210969


namespace NUMINAMATH_GPT_synodic_month_is_approx_29_5306_l2109_210981

noncomputable def sidereal_month_moon : ℝ := 
27 + 7/24 + 43/1440  -- conversion of 7 hours and 43 minutes to days

noncomputable def sidereal_year_earth : ℝ := 
365 + 6/24 + 9/1440  -- conversion of 6 hours and 9 minutes to days

noncomputable def synodic_month (T_H T_F: ℝ) : ℝ := 
(T_H * T_F) / (T_F - T_H)

theorem synodic_month_is_approx_29_5306 : 
  abs (synodic_month sidereal_month_moon sidereal_year_earth - (29 + 12/24 + 44/1440)) < 0.0001 :=
by 
  sorry

end NUMINAMATH_GPT_synodic_month_is_approx_29_5306_l2109_210981


namespace NUMINAMATH_GPT_range_of_a_l2109_210972

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2109_210972


namespace NUMINAMATH_GPT_original_number_is_8_l2109_210957

open Real

theorem original_number_is_8 
  (x : ℝ)
  (h1 : |(x + 5) - (x - 5)| = 10)
  (h2 : (10 / (x + 5)) * 100 = 76.92) : 
  x = 8 := 
by
  sorry

end NUMINAMATH_GPT_original_number_is_8_l2109_210957


namespace NUMINAMATH_GPT_distance_from_p_to_center_is_2_sqrt_10_l2109_210954

-- Define the conditions
def r : ℝ := 4
def PA : ℝ := 4
def PB : ℝ := 6

-- The conjecture to prove
theorem distance_from_p_to_center_is_2_sqrt_10
  (r : ℝ) (PA : ℝ) (PB : ℝ) 
  (PA_mul_PB : PA * PB = 24) 
  (r_squared : r = 4)  : 
  ∃ d : ℝ, d = 2 * Real.sqrt 10 := 
by sorry

end NUMINAMATH_GPT_distance_from_p_to_center_is_2_sqrt_10_l2109_210954


namespace NUMINAMATH_GPT_find_b_l2109_210983

-- Definitions for conditions
def eq1 (a : ℤ) : Prop := 2 * a + 1 = 1
def eq2 (a b : ℤ) : Prop := 2 * b - 3 * a = 2

-- The theorem statement
theorem find_b (a b : ℤ) (h1 : eq1 a) (h2 : eq2 a b) : b = 1 :=
  sorry  -- Proof to be filled in.

end NUMINAMATH_GPT_find_b_l2109_210983


namespace NUMINAMATH_GPT_expression_one_expression_two_l2109_210968

-- Define the expressions to be proved.
theorem expression_one : (3.6 - 0.8) * (1.8 + 2.05) = 10.78 :=
by sorry

theorem expression_two : (34.28 / 2) - (16.2 / 4) = 13.09 :=
by sorry

end NUMINAMATH_GPT_expression_one_expression_two_l2109_210968


namespace NUMINAMATH_GPT_range_of_m_l2109_210927

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m
  (m : ℝ)
  (hθ : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2)
  (h : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) :
  m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2109_210927


namespace NUMINAMATH_GPT_max_a4_l2109_210923

theorem max_a4 (a1 d a4 : ℝ) 
  (h1 : 2 * a1 + 3 * d ≥ 5) 
  (h2 : a1 + 2 * d ≤ 3) 
  (ha4 : a4 = a1 + 3 * d) : 
  a4 ≤ 4 := 
by 
  sorry

end NUMINAMATH_GPT_max_a4_l2109_210923


namespace NUMINAMATH_GPT_solve_for_k_l2109_210941

theorem solve_for_k (x y k : ℤ) (h1 : x = -3) (h2 : y = 2) (h3 : 2 * x + k * y = 0) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2109_210941


namespace NUMINAMATH_GPT_trig_expression_value_l2109_210904

theorem trig_expression_value
  (x : ℝ)
  (h : Real.tan (x + Real.pi / 4) = -3) :
  (Real.sin x + 2 * Real.cos x) / (3 * Real.sin x + 4 * Real.cos x) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l2109_210904


namespace NUMINAMATH_GPT_train_speed_clicks_l2109_210932

theorem train_speed_clicks (x : ℝ) (v : ℝ) (t : ℝ) 
  (h1 : v = x * 5280 / 60) 
  (h2 : t = 25) 
  (h3 : 70 * t = v * 25) : v = 70 := sorry

end NUMINAMATH_GPT_train_speed_clicks_l2109_210932


namespace NUMINAMATH_GPT_max_distance_right_triangle_l2109_210902

theorem max_distance_right_triangle (a b : ℝ) 
  (h1: ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
    (a * A.1 + 2 * b * A.2 = 1) ∧ (a * B.1 + 2 * b * B.2 = 1) ∧ 
    ∃ (C : ℝ × ℝ), C = (0,0) ∧ (A.1 * B.1 + A.2 * B.2 = 0)): 
  ∃ (d : ℝ), d = (Real.sqrt (a^2 + b^2)) ∧ d ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_distance_right_triangle_l2109_210902


namespace NUMINAMATH_GPT_rectangular_plot_area_l2109_210907

theorem rectangular_plot_area (breadth length : ℕ) (h1 : breadth = 14) (h2 : length = 3 * breadth) : (length * breadth) = 588 := 
by 
  -- imports, noncomputable keyword, and placeholder proof for compilation
  sorry

end NUMINAMATH_GPT_rectangular_plot_area_l2109_210907


namespace NUMINAMATH_GPT_product_of_solutions_l2109_210914

theorem product_of_solutions (x : ℝ) :
  ∃ (α β : ℝ), (x^2 - 4*x - 21 = 0) ∧ α * β = -21 := sorry

end NUMINAMATH_GPT_product_of_solutions_l2109_210914


namespace NUMINAMATH_GPT_annual_rent_per_square_foot_is_172_l2109_210964

def monthly_rent : ℕ := 3600
def local_taxes : ℕ := 500
def maintenance_fees : ℕ := 200
def length_of_shop : ℕ := 20
def width_of_shop : ℕ := 15

def total_monthly_cost : ℕ := monthly_rent + local_taxes + maintenance_fees
def annual_cost : ℕ := total_monthly_cost * 12
def area_of_shop : ℕ := length_of_shop * width_of_shop
def annual_rent_per_square_foot : ℕ := annual_cost / area_of_shop

theorem annual_rent_per_square_foot_is_172 :
  annual_rent_per_square_foot = 172 := by
    sorry

end NUMINAMATH_GPT_annual_rent_per_square_foot_is_172_l2109_210964


namespace NUMINAMATH_GPT_evaluate_expression_l2109_210961

theorem evaluate_expression (c : ℕ) (hc : c = 4) : 
  ((c^c - 2 * c * (c-2)^c + c^2)^c) = 431441456 :=
by
  rw [hc]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2109_210961


namespace NUMINAMATH_GPT_ball_distribution_ways_l2109_210903

theorem ball_distribution_ways :
  ∃ (ways : ℕ), ways = 10 ∧
    ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 4 ∧ 
    (∀ (b : ℕ), b < boxes → b > 0) →
    ways = 10 :=
sorry

end NUMINAMATH_GPT_ball_distribution_ways_l2109_210903


namespace NUMINAMATH_GPT_inequality_1_inequality_2_l2109_210988

theorem inequality_1 (x : ℝ) : (2 * x^2 - 3 * x + 1 < 0) ↔ (1 / 2 < x ∧ x < 1) := 
by sorry

theorem inequality_2 (x : ℝ) (h : x ≠ -1) : (2 * x / (x + 1) ≥ 1) ↔ (x < -1 ∨ x ≥ 1) := 
by sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_l2109_210988


namespace NUMINAMATH_GPT_miles_monday_calculation_l2109_210924

-- Define the constants
def flat_fee : ℕ := 150
def cost_per_mile : ℝ := 0.50
def miles_thursday : ℕ := 744
def total_cost : ℕ := 832

-- Define the equation to be proved
theorem miles_monday_calculation :
  ∃ M : ℕ, (flat_fee + (M : ℝ) * cost_per_mile + (miles_thursday : ℝ) * cost_per_mile = total_cost) ∧ M = 620 :=
by
  sorry

end NUMINAMATH_GPT_miles_monday_calculation_l2109_210924


namespace NUMINAMATH_GPT_line_through_two_points_l2109_210965

theorem line_through_two_points (P Q : ℝ × ℝ) (hP : P = (2, 5)) (hQ : Q = (2, -5)) :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q → x = 2) :=
by
  sorry

end NUMINAMATH_GPT_line_through_two_points_l2109_210965


namespace NUMINAMATH_GPT_aquarium_counts_l2109_210958

theorem aquarium_counts :
  ∃ (O S L : ℕ), O + S = 7 ∧ L + S = 6 ∧ O + L = 5 ∧ (O ≤ S ∧ O ≤ L) ∧ O = 5 ∧ S = 7 ∧ L = 6 :=
by
  sorry

end NUMINAMATH_GPT_aquarium_counts_l2109_210958


namespace NUMINAMATH_GPT_reading_time_per_week_l2109_210971

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end NUMINAMATH_GPT_reading_time_per_week_l2109_210971


namespace NUMINAMATH_GPT_width_of_wall_is_6_l2109_210937

-- Definitions of the conditions given in the problem
def height_of_wall (w : ℝ) := 4 * w
def length_of_wall (h : ℝ) := 3 * h
def volume_of_wall (w h l : ℝ) := w * h * l

-- Proof statement that the width of the wall is 6 meters given the conditions
theorem width_of_wall_is_6 :
  ∃ w : ℝ, 
  (height_of_wall w = 4 * w) ∧ 
  (length_of_wall (height_of_wall w) = 3 * (height_of_wall w)) ∧ 
  (volume_of_wall w (height_of_wall w) (length_of_wall (height_of_wall w)) = 10368) ∧ 
  (w = 6) :=
sorry

end NUMINAMATH_GPT_width_of_wall_is_6_l2109_210937


namespace NUMINAMATH_GPT_trigonometric_inequality_solution_l2109_210960

theorem trigonometric_inequality_solution (k : ℤ) :
  ∃ x : ℝ, x = - (3 * Real.pi) / 2 + 4 * Real.pi * k ∧
           (Real.cos (x / 2) + Real.sin (x / 2) ≤ (Real.sin x - 3) / Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_solution_l2109_210960


namespace NUMINAMATH_GPT_peter_runs_more_than_andrew_each_day_l2109_210917

-- Define the constants based on the conditions
def miles_andrew : ℕ := 2
def total_days : ℕ := 5
def total_miles : ℕ := 35

-- Define a theorem to prove the number of miles Peter runs more than Andrew each day
theorem peter_runs_more_than_andrew_each_day : 
  ∃ x : ℕ, total_days * (miles_andrew + x) + total_days * miles_andrew = total_miles ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_peter_runs_more_than_andrew_each_day_l2109_210917


namespace NUMINAMATH_GPT_arc_length_parametric_l2109_210935

open Real Interval

noncomputable def arc_length (f_x f_y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in Set.Icc t1 t2, sqrt ((deriv f_x t)^2 + (deriv f_y t)^2)

theorem arc_length_parametric :
  arc_length
    (λ t => 2.5 * (t - sin t))
    (λ t => 2.5 * (1 - cos t))
    (π / 2) π = 5 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_parametric_l2109_210935


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2109_210998

-- Define sets A and B
def setA : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x < 2}

-- Prove that A ∩ B = (-3, 2)
theorem intersection_of_A_and_B : {x : ℝ | x ∈ setA ∧ x ∈ setB} = {x : ℝ | -3 < x ∧ x < 2} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2109_210998


namespace NUMINAMATH_GPT_negation_proof_l2109_210982

theorem negation_proof :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) → (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l2109_210982


namespace NUMINAMATH_GPT_other_leg_length_l2109_210911

theorem other_leg_length (a b c : ℕ) (ha : a = 24) (hc : c = 25) 
  (h : a * a + b * b = c * c) : b = 7 := 
by 
  sorry

end NUMINAMATH_GPT_other_leg_length_l2109_210911


namespace NUMINAMATH_GPT_lifespan_of_bat_l2109_210933

theorem lifespan_of_bat (B : ℕ) (h₁ : ∀ B, B - 6 < B)
    (h₂ : ∀ B, 4 * (B - 6) < 4 * B)
    (h₃ : B + (B - 6) + 4 * (B - 6) = 30) :
    B = 10 := by
  sorry

end NUMINAMATH_GPT_lifespan_of_bat_l2109_210933


namespace NUMINAMATH_GPT_boat_speed_5_kmh_l2109_210977

noncomputable def boat_speed_in_still_water (V_s : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  (d / t) - V_s

theorem boat_speed_5_kmh :
  boat_speed_in_still_water 5 10 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_5_kmh_l2109_210977


namespace NUMINAMATH_GPT_conference_duration_l2109_210926

theorem conference_duration (hours minutes lunch_break total_minutes active_session : ℕ) 
  (h1 : hours = 8) 
  (h2 : minutes = 40) 
  (h3 : lunch_break = 15) 
  (h4 : total_minutes = hours * 60 + minutes)
  (h5 : active_session = total_minutes - lunch_break) :
  active_session = 505 := 
by {
  sorry
}

end NUMINAMATH_GPT_conference_duration_l2109_210926


namespace NUMINAMATH_GPT_smallest_n_for_sum_exceed_10_pow_5_l2109_210967

def a₁ : ℕ := 9
def r : ℕ := 10
def S (n : ℕ) : ℕ := 5 * n^2 + 4 * n
def target_sum : ℕ := 10^5

theorem smallest_n_for_sum_exceed_10_pow_5 : 
  ∃ n : ℕ, S n > target_sum ∧ ∀ m < n, ¬(S m > target_sum) := 
sorry

end NUMINAMATH_GPT_smallest_n_for_sum_exceed_10_pow_5_l2109_210967


namespace NUMINAMATH_GPT_cattle_transport_problem_l2109_210905

noncomputable def truck_capacity 
    (total_cattle : ℕ)
    (distance_one_way : ℕ)
    (speed : ℕ)
    (total_time : ℕ) : ℕ :=
  total_cattle / (total_time / ((distance_one_way * 2) / speed))

theorem cattle_transport_problem :
  truck_capacity 400 60 60 40 = 20 := by
  -- The theorem statement follows the structure from the conditions and question
  sorry

end NUMINAMATH_GPT_cattle_transport_problem_l2109_210905


namespace NUMINAMATH_GPT_pref_card_game_arrangements_l2109_210975

noncomputable def number_of_arrangements :=
  (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3)

theorem pref_card_game_arrangements :
  number_of_arrangements = (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3) :=
by
  sorry

end NUMINAMATH_GPT_pref_card_game_arrangements_l2109_210975


namespace NUMINAMATH_GPT_shaded_area_of_hexagon_with_quarter_circles_l2109_210966

noncomputable def area_inside_hexagon_outside_circles
  (s : ℝ) (h : s = 4) : ℝ :=
  let hex_area := (3 * Real.sqrt 3) / 2 * s^2
  let quarter_circle_area := (1 / 4) * Real.pi * s^2
  let total_quarter_circles_area := 6 * quarter_circle_area
  hex_area - total_quarter_circles_area

theorem shaded_area_of_hexagon_with_quarter_circles :
  area_inside_hexagon_outside_circles 4 rfl = 48 * Real.sqrt 3 - 24 * Real.pi := by
  sorry

end NUMINAMATH_GPT_shaded_area_of_hexagon_with_quarter_circles_l2109_210966


namespace NUMINAMATH_GPT_value_of_fraction_l2109_210962

theorem value_of_fraction (x y : ℤ) (h1 : x = 3) (h2 : y = 4) : (x^5 + 3 * y^3) / 9 = 48 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l2109_210962


namespace NUMINAMATH_GPT_exists_solution_interval_inequality_l2109_210952

theorem exists_solution_interval_inequality :
  ∀ x : ℝ, (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) := 
by
  sorry

end NUMINAMATH_GPT_exists_solution_interval_inequality_l2109_210952


namespace NUMINAMATH_GPT_num_cells_after_10_moves_l2109_210909

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end NUMINAMATH_GPT_num_cells_after_10_moves_l2109_210909


namespace NUMINAMATH_GPT_expression_value_l2109_210934

theorem expression_value : (19 + 12) ^ 2 - (12 ^ 2 + 19 ^ 2) = 456 := 
by sorry

end NUMINAMATH_GPT_expression_value_l2109_210934


namespace NUMINAMATH_GPT_shauna_fifth_test_score_l2109_210978

theorem shauna_fifth_test_score :
  ∀ (a1 a2 a3 a4: ℕ), a1 = 76 → a2 = 94 → a3 = 87 → a4 = 92 →
  (∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / 5 = 85 ∧ a5 = 76) :=
by
  sorry

end NUMINAMATH_GPT_shauna_fifth_test_score_l2109_210978


namespace NUMINAMATH_GPT_M_inter_N_l2109_210908

namespace ProofProblem

def M : Set ℝ := { x | 3 * x - x^2 > 0 }
def N : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem M_inter_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
sorry

end ProofProblem

end NUMINAMATH_GPT_M_inter_N_l2109_210908


namespace NUMINAMATH_GPT_farmer_initial_apples_l2109_210979

variable (initial_apples given_away_apples remaining_apples : ℕ)

def initial_apple_count (given_away_apples remaining_apples : ℕ) : ℕ :=
  given_away_apples + remaining_apples

theorem farmer_initial_apples : initial_apple_count 88 39 = 127 := by
  -- Given conditions
  let given_away_apples := 88
  let remaining_apples := 39

  -- Calculate the initial apples
  let initial_apples := initial_apple_count given_away_apples remaining_apples

  -- We are supposed to prove initial apples count is 127
  show initial_apples = 127
  sorry

end NUMINAMATH_GPT_farmer_initial_apples_l2109_210979


namespace NUMINAMATH_GPT_gcd_153_68_eq_17_l2109_210974

theorem gcd_153_68_eq_17 : Int.gcd 153 68 = 17 :=
by
  sorry

end NUMINAMATH_GPT_gcd_153_68_eq_17_l2109_210974


namespace NUMINAMATH_GPT_ratio_of_probabilities_l2109_210980

-- Define the total number of balls and bins
def balls : ℕ := 20
def bins : ℕ := 6

-- Define the sets A and B based on the given conditions
def A : ℕ := Nat.choose bins 1 * Nat.choose (bins - 1) 1 * (Nat.factorial balls / (Nat.factorial 2 * Nat.factorial 5 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def B : ℕ := Nat.choose bins 2 * (Nat.factorial balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))

-- Define the probabilities p and q
def p : ℚ := A / (Nat.factorial balls * Nat.factorial bins)
def q : ℚ := B / (Nat.factorial balls * Nat.factorial bins)

-- Prove the ratio of probabilities p and q equals 2
theorem ratio_of_probabilities : p / q = 2 := by sorry

end NUMINAMATH_GPT_ratio_of_probabilities_l2109_210980


namespace NUMINAMATH_GPT_minimum_one_by_one_squares_l2109_210963

theorem minimum_one_by_one_squares :
  ∀ (x y z : ℕ), 9 * x + 4 * y + z = 49 → (z = 3) :=
  sorry

end NUMINAMATH_GPT_minimum_one_by_one_squares_l2109_210963


namespace NUMINAMATH_GPT_Force_Inversely_Proportional_l2109_210912

theorem Force_Inversely_Proportional
  (L₁ F₁ L₂ F₂ : ℝ)
  (h₁ : L₁ = 12)
  (h₂ : F₁ = 480)
  (h₃ : L₂ = 18)
  (h_inv : F₁ * L₁ = F₂ * L₂) :
  F₂ = 320 :=
by
  sorry

end NUMINAMATH_GPT_Force_Inversely_Proportional_l2109_210912


namespace NUMINAMATH_GPT_product_of_roots_l2109_210922

noncomputable def quadratic_has_product_of_roots (A B C : ℤ) : ℚ :=
  C / A

theorem product_of_roots (α β : ℚ) (h : 12 * α^2 + 28 * α - 320 = 0) (h2 : 12 * β^2 + 28 * β - 320 = 0) :
  quadratic_has_product_of_roots 12 28 (-320) = -80 / 3 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_product_of_roots_l2109_210922


namespace NUMINAMATH_GPT_polynomial_satisfies_condition_l2109_210919

open Polynomial

noncomputable def polynomial_f : Polynomial ℝ := 6 * X ^ 2 + 5 * X + 1
noncomputable def polynomial_g : Polynomial ℝ := 3 * X ^ 2 + 7 * X + 2

def sum_of_squares (p : Polynomial ℝ) : ℝ :=
  p.coeff 0 ^ 2 + p.coeff 1 ^ 2 + p.coeff 2 ^ 2 + p.coeff 3 ^ 2 + -- ...
  sorry -- Extend as necessary for the degree of the polynomial

theorem polynomial_satisfies_condition :
  (∀ n : ℕ, sum_of_squares (polynomial_f ^ n) = sum_of_squares (polynomial_g ^ n)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_satisfies_condition_l2109_210919


namespace NUMINAMATH_GPT_ratio_c_d_l2109_210999

theorem ratio_c_d (x y c d : ℝ) 
  (h1 : 8 * x - 5 * y = c)
  (h2 : 10 * y - 12 * x = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0) :
  c / d = -2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_c_d_l2109_210999


namespace NUMINAMATH_GPT_person_income_l2109_210942

/-- If the income and expenditure of a person are in the ratio 15:8 and the savings are Rs. 7000, then the income of the person is Rs. 15000. -/
theorem person_income (x : ℝ) (income expenditure : ℝ) (savings : ℝ) 
  (h1 : income = 15 * x) 
  (h2 : expenditure = 8 * x) 
  (h3 : savings = income - expenditure) 
  (h4 : savings = 7000) : 
  income = 15000 := 
by 
  sorry

end NUMINAMATH_GPT_person_income_l2109_210942


namespace NUMINAMATH_GPT_cistern_fill_time_l2109_210915

theorem cistern_fill_time (F : ℝ) (E : ℝ) (net_rate : ℝ) (time : ℝ)
  (h_F : F = 1 / 4)
  (h_E : E = 1 / 8)
  (h_net : net_rate = F - E)
  (h_time : time = 1 / net_rate) :
  time = 8 := 
sorry

end NUMINAMATH_GPT_cistern_fill_time_l2109_210915


namespace NUMINAMATH_GPT_range_of_a_l2109_210918

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x + x^2 else x - x^2

theorem range_of_a (a : ℝ) : (∀ x, -1/2 ≤ x ∧ x ≤ 1/2 → f (x^2 + 1) > f (a * x)) ↔ -5/2 < a ∧ a < 5/2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2109_210918


namespace NUMINAMATH_GPT_not_an_algorithm_option_B_l2109_210944

def is_algorithm (description : String) : Prop :=
  description = "clear and finite steps to solve a problem producing correct results when executed by a computer"

def operation_to_string (option : Char) : String :=
  match option with
  | 'A' => "Calculating the area of a circle given its radius"
  | 'B' => "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
  | 'C' => "Finding the equation of a line given two points in the coordinate plane"
  | 'D' => "The rules of addition, subtraction, multiplication, and division"
  | _ => ""

noncomputable def categorize_operation (option : Char) : Prop :=
  option = 'B' ↔ ¬ is_algorithm (operation_to_string option)

theorem not_an_algorithm_option_B :
  categorize_operation 'B' :=
by
  sorry

end NUMINAMATH_GPT_not_an_algorithm_option_B_l2109_210944


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2109_210938

noncomputable def a := 2 * Real.sqrt 3 + 3
noncomputable def expr := (1 - 1 / (a - 2)) / ((a ^ 2 - 6 * a + 9) / (2 * a - 4))

theorem simplify_and_evaluate : expr = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2109_210938


namespace NUMINAMATH_GPT_plankton_consumption_difference_l2109_210947

theorem plankton_consumption_difference 
  (x : ℕ) 
  (d : ℕ) 
  (total_hours : ℕ := 9) 
  (total_consumption : ℕ := 360)
  (sixth_hour_consumption : ℕ := 43)
  (total_series_sum : x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) + (x + 5 * d) + (x + 6 * d) + (x + 7 * d) + (x + 8 * d) = total_consumption)
  (sixth_hour_eq : x + 5 * d = sixth_hour_consumption)
  : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_plankton_consumption_difference_l2109_210947


namespace NUMINAMATH_GPT_domain_of_f_exp_l2109_210901

theorem domain_of_f_exp (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x + 1 ∧ x + 1 < 4 → ∃ y, f y = f (x + 1)) →
  (∀ x, 1 ≤ 2^x ∧ 2^x < 4 → ∃ y, f y = f (2^x)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_exp_l2109_210901


namespace NUMINAMATH_GPT_find_initial_salt_concentration_l2109_210945

noncomputable def initial_salt_concentration 
  (x : ℝ) (final_concentration : ℝ) (extra_water : ℝ) (extra_salt : ℝ) (evaporation_fraction : ℝ) : ℝ :=
  let initial_volume : ℝ := x
  let remaining_volume : ℝ := evaporation_fraction * initial_volume
  let mixed_volume : ℝ := remaining_volume + extra_water + extra_salt
  let target_salt_volume_fraction : ℝ := final_concentration / 100
  let initial_salt_volume_fraction : ℝ := (target_salt_volume_fraction * mixed_volume - extra_salt) / initial_volume * 100
  initial_salt_volume_fraction

theorem find_initial_salt_concentration :
  initial_salt_concentration 120 33.333333333333336 8 16 (3 / 4) = 18.333333333333332 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_salt_concentration_l2109_210945


namespace NUMINAMATH_GPT_sum_of_interior_angles_l2109_210955

theorem sum_of_interior_angles (n : ℕ) : 
  (∀ θ, θ = 40 ∧ (n = 360 / θ)) → (n - 2) * 180 = 1260 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l2109_210955


namespace NUMINAMATH_GPT_cos_half_angle_neg_sqrt_l2109_210992

theorem cos_half_angle_neg_sqrt (theta m : ℝ) 
  (h1 : (5 / 2) * Real.pi < theta ∧ theta < 3 * Real.pi)
  (h2 : |Real.cos theta| = m) : 
  Real.cos (theta / 2) = -Real.sqrt ((1 - m) / 2) :=
sorry

end NUMINAMATH_GPT_cos_half_angle_neg_sqrt_l2109_210992


namespace NUMINAMATH_GPT_pages_in_first_chapter_l2109_210996

/--
Rita is reading a five-chapter book with 95 pages. Each chapter has three pages more than the previous one. 
Prove the number of pages in the first chapter.
-/
theorem pages_in_first_chapter (h : ∃ p1 p2 p3 p4 p5 : ℕ, p1 + p2 + p3 + p4 + p5 = 95 ∧ p2 = p1 + 3 ∧ p3 = p1 + 6 ∧ p4 = p1 + 9 ∧ p5 = p1 + 12) : 
  ∃ x : ℕ, x = 13 := 
by
  sorry

end NUMINAMATH_GPT_pages_in_first_chapter_l2109_210996


namespace NUMINAMATH_GPT_largest_intersection_value_l2109_210949

theorem largest_intersection_value (b c d : ℝ) :
  ∀ x : ℝ, (x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3 = c*x - d) → x ≤ 6 := sorry

end NUMINAMATH_GPT_largest_intersection_value_l2109_210949


namespace NUMINAMATH_GPT_program_result_l2109_210943

def program_loop (i : ℕ) (s : ℕ) : ℕ :=
if i < 9 then s else program_loop (i - 1) (s * i)

theorem program_result : 
  program_loop 11 1 = 990 :=
by 
  sorry

end NUMINAMATH_GPT_program_result_l2109_210943


namespace NUMINAMATH_GPT_value_of_a_plus_b_l2109_210994

theorem value_of_a_plus_b (a b : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b) 
  (hg : ∀ x, g x = -3 * x + 2)
  (hgf : ∀ x, g (f x) = -2 * x - 3) :
  a + b = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l2109_210994


namespace NUMINAMATH_GPT_trig_identity_cos_sin_l2109_210940

theorem trig_identity_cos_sin : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
sorry

end NUMINAMATH_GPT_trig_identity_cos_sin_l2109_210940


namespace NUMINAMATH_GPT_balloon_height_l2109_210950

theorem balloon_height :
  let initial_money : ℝ := 200
  let cost_sheet : ℝ := 42
  let cost_rope : ℝ := 18
  let cost_tank_and_burner : ℝ := 14
  let helium_price_per_ounce : ℝ := 1.5
  let lift_per_ounce : ℝ := 113
  let remaining_money := initial_money - cost_sheet - cost_rope - cost_tank_and_burner
  let ounces_of_helium := remaining_money / helium_price_per_ounce
  let height := ounces_of_helium * lift_per_ounce
  height = 9492 :=
by
  sorry

end NUMINAMATH_GPT_balloon_height_l2109_210950


namespace NUMINAMATH_GPT_ajays_monthly_income_l2109_210930

theorem ajays_monthly_income :
  ∀ (I : ℝ), 
  (0.50 * I) + (0.25 * I) + (0.15 * I) + 9000 = I → I = 90000 :=
by
  sorry

end NUMINAMATH_GPT_ajays_monthly_income_l2109_210930


namespace NUMINAMATH_GPT_math_proof_problem_l2109_210906

variable (a d e : ℝ)

theorem math_proof_problem (h1 : a < 0) (h2 : a < d) (h3 : d < e) :
  (a * d < a * e) ∧ (a + d < d + e) ∧ (e / a < 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_math_proof_problem_l2109_210906


namespace NUMINAMATH_GPT_total_guppies_l2109_210929

-- Define conditions
def Haylee_guppies : ℕ := 3 * 12
def Jose_guppies : ℕ := Haylee_guppies / 2
def Charliz_guppies : ℕ := Jose_guppies / 3
def Nicolai_guppies : ℕ := Charliz_guppies * 4

-- Theorem statement: total number of guppies is 84
theorem total_guppies : Haylee_guppies + Jose_guppies + Charliz_guppies + Nicolai_guppies = 84 := 
by 
  sorry

end NUMINAMATH_GPT_total_guppies_l2109_210929


namespace NUMINAMATH_GPT_solid_is_cone_l2109_210970

-- Definitions of the conditions.
def front_and_side_views_are_equilateral_triangles (S : Type) : Prop :=
∀ (F : S → Prop) (E : S → Prop), (∃ T : S, F T ∧ E T ∧ T = T) 

def top_view_is_circle_with_center (S : Type) : Prop :=
∀ (C : S → Prop), (∃ O : S, C O ∧ O = O)

-- The proof statement that given the above conditions, the solid is a cone
theorem solid_is_cone (S : Type)
  (H1 : front_and_side_views_are_equilateral_triangles S)
  (H2 : top_view_is_circle_with_center S) : 
  ∃ C : S, C = C :=
by 
  sorry

end NUMINAMATH_GPT_solid_is_cone_l2109_210970


namespace NUMINAMATH_GPT_baker_sold_pastries_l2109_210991

theorem baker_sold_pastries : 
  ∃ P : ℕ, (97 = P + 89) ∧ P = 8 :=
by 
  sorry

end NUMINAMATH_GPT_baker_sold_pastries_l2109_210991


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l2109_210953

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) 
    (h1 : a 2 = 2) 
    (h2 : a 3 = 4) : 
    a 10 = 18 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l2109_210953


namespace NUMINAMATH_GPT_problem_l2109_210931

theorem problem (p q : ℕ) (hp: p > 1) (hq: q > 1) (h1 : (2 * p - 1) % q = 0) (h2 : (2 * q - 1) % p = 0) : p + q = 8 := 
sorry

end NUMINAMATH_GPT_problem_l2109_210931


namespace NUMINAMATH_GPT_find_constant_l2109_210910

theorem find_constant
  (k : ℝ)
  (r : ℝ := 36)
  (C : ℝ := 72 * k)
  (h1 : C = 2 * Real.pi * r)
  : k = Real.pi := by
  sorry

end NUMINAMATH_GPT_find_constant_l2109_210910


namespace NUMINAMATH_GPT_janet_total_lives_l2109_210951

/-
  Janet's initial lives: 38
  Lives lost: 16
  Lives gained: 32
  Prove that total lives == 54 after the changes
-/

theorem janet_total_lives (initial_lives lost_lives gained_lives : ℕ) 
(h1 : initial_lives = 38)
(h2 : lost_lives = 16)
(h3 : gained_lives = 32):
  initial_lives - lost_lives + gained_lives = 54 := by
  sorry

end NUMINAMATH_GPT_janet_total_lives_l2109_210951


namespace NUMINAMATH_GPT_poly_perfect_fourth_l2109_210939

theorem poly_perfect_fourth (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, (a * x^2 + b * x + c) = k^4) : 
  a = 0 ∧ b = 0 :=
sorry

end NUMINAMATH_GPT_poly_perfect_fourth_l2109_210939


namespace NUMINAMATH_GPT_tan_ratio_proof_l2109_210985

noncomputable def tan_ratio (a b : ℝ) : ℝ := Real.tan a / Real.tan b

theorem tan_ratio_proof (a b : ℝ) (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 3) : 
tan_ratio a b = 23 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_ratio_proof_l2109_210985


namespace NUMINAMATH_GPT_solutions_to_x_squared_eq_x_l2109_210921

theorem solutions_to_x_squared_eq_x (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := 
sorry

end NUMINAMATH_GPT_solutions_to_x_squared_eq_x_l2109_210921


namespace NUMINAMATH_GPT_product_of_integers_eq_expected_result_l2109_210948

theorem product_of_integers_eq_expected_result
  (E F G H I : ℚ) 
  (h1 : E + F + G + H + I = 80) 
  (h2 : E + 2 = F - 2) 
  (h3 : F - 2 = G * 2) 
  (h4 : G * 2 = H * 3) 
  (h5 : H * 3 = I / 2) :
  E * F * G * H * I = (5120000 / 81) := 
by 
  sorry

end NUMINAMATH_GPT_product_of_integers_eq_expected_result_l2109_210948


namespace NUMINAMATH_GPT_find_f_2015_l2109_210973

def f (x : ℝ) := 2 * x - 1 

theorem find_f_2015 (f : ℝ → ℝ)
  (H1 : ∀ a b : ℝ, f ((2 * a + b) / 3) = (2 * f a + f b) / 3)
  (H2 : f 1 = 1)
  (H3 : f 4 = 7) :
  f 2015 = 4029 := by 
  sorry

end NUMINAMATH_GPT_find_f_2015_l2109_210973


namespace NUMINAMATH_GPT_max_product_xy_l2109_210936

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end NUMINAMATH_GPT_max_product_xy_l2109_210936


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2109_210989

theorem solution_set_of_inequality :
  { x : ℝ | - (1 : ℝ) / 2 < x ∧ x <= 1 } =
  { x : ℝ | (x - 1) / (2 * x + 1) <= 0 ∧ x ≠ - (1 : ℝ) / 2 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2109_210989


namespace NUMINAMATH_GPT_sum_is_zero_l2109_210928

-- Define the conditions: the function f is invertible, and f(a) = 3, f(b) = 7
variables {α β : Type} [Inhabited α] [Inhabited β]

def invertible {α β : Type} (f : α → β) :=
  ∃ g : β → α, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

variables (f : ℝ → ℝ) (a b : ℝ)

-- Assume f is invertible and the given conditions f(a) = 3 and f(b) = 7
axiom f_invertible : invertible f
axiom f_a : f a = 3
axiom f_b : f b = 7

-- Prove that a + b = 0
theorem sum_is_zero : a + b = 0 :=
sorry

end NUMINAMATH_GPT_sum_is_zero_l2109_210928


namespace NUMINAMATH_GPT_unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l2109_210990

theorem unit_digit_of_product_of_nine_consecutive_numbers_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l2109_210990


namespace NUMINAMATH_GPT_three_digit_avg_permutations_l2109_210984

theorem three_digit_avg_permutations (a b c: ℕ) (A: ℕ) (h₀: 1 ≤ a ∧ a ≤ 9) (h₁: 0 ≤ b ∧ b ≤ 9) (h₂: 0 ≤ c ∧ c ≤ 9) (h₃: A = 100 * a + 10 * b + c):
  ((100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)) / 6 = A ↔ 7 * a = 3 * b + 4 * c := by
  sorry

end NUMINAMATH_GPT_three_digit_avg_permutations_l2109_210984


namespace NUMINAMATH_GPT_B_completes_work_in_12_hours_l2109_210995

theorem B_completes_work_in_12_hours:
  let A := 1 / 4
  let C := (1 / 2) - A
  let B := (1 / 3) - C
  (1 / B) = 12 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_B_completes_work_in_12_hours_l2109_210995


namespace NUMINAMATH_GPT_eq_relation_q_r_l2109_210946

-- Define the angles in the context of the problem
variables {A B C D E F : Type}
variables {angle_BAC angle_BFD angle_ADE angle_FEC : ℝ}
variables (right_triangle_ABC : A → B → C → angle_BAC = 90)

-- Equilateral triangle DEF inscribed in ABC
variables (inscribed_equilateral_DEF : D → E → F)
variables (angle_BFD_eq_p : ∀ p : ℝ, angle_BFD = p)
variables (angle_ADE_eq_q : ∀ q : ℝ, angle_ADE = q)
variables (angle_FEC_eq_r : ∀ r : ℝ, angle_FEC = r)

-- Main statement to be proved
theorem eq_relation_q_r {p q r : ℝ} 
  (right_triangle_ABC : angle_BAC = 90)
  (angle_BFD : angle_BFD = 30 + q)
  (angle_FEC : angle_FEC = 120 - r) :
  q + r = 60 :=
sorry

end NUMINAMATH_GPT_eq_relation_q_r_l2109_210946


namespace NUMINAMATH_GPT_polynomial_positive_values_l2109_210900

noncomputable def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_values :
  ∀ (z : ℝ), (∃ (x y : ℝ), P x y = z) ↔ z > 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_positive_values_l2109_210900


namespace NUMINAMATH_GPT_find_total_bricks_l2109_210920

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

end NUMINAMATH_GPT_find_total_bricks_l2109_210920


namespace NUMINAMATH_GPT_joyful_not_blue_l2109_210986

variables {Snakes : Type} 
variables (isJoyful : Snakes → Prop) (isBlue : Snakes → Prop)
variables (canMultiply : Snakes → Prop) (canDivide : Snakes → Prop)

-- Conditions
axiom H1 : ∀ s : Snakes, isJoyful s → canMultiply s
axiom H2 : ∀ s : Snakes, isBlue s → ¬ canDivide s
axiom H3 : ∀ s : Snakes, ¬ canDivide s → ¬ canMultiply s

theorem joyful_not_blue (s : Snakes) : isJoyful s → ¬ isBlue s :=
by sorry

end NUMINAMATH_GPT_joyful_not_blue_l2109_210986


namespace NUMINAMATH_GPT_cookies_per_day_l2109_210916

theorem cookies_per_day (cost_per_cookie : ℕ) (total_spent : ℕ) (days_in_march : ℕ) (h1 : cost_per_cookie = 16) (h2 : total_spent = 992) (h3 : days_in_march = 31) :
  (total_spent / cost_per_cookie) / days_in_march = 2 :=
by sorry

end NUMINAMATH_GPT_cookies_per_day_l2109_210916
