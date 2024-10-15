import Mathlib

namespace NUMINAMATH_GPT_num_turtles_on_sand_l741_74151

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end NUMINAMATH_GPT_num_turtles_on_sand_l741_74151


namespace NUMINAMATH_GPT_sad_girls_count_l741_74143

variables (total_children happy_children sad_children neither_children : ℕ)
variables (total_boys total_girls happy_boys sad_children total_sad_boys : ℕ)

theorem sad_girls_count :
  total_children = 60 ∧ 
  happy_children = 30 ∧ 
  sad_children = 10 ∧ 
  neither_children = 20 ∧ 
  total_boys = 17 ∧ 
  total_girls = 43 ∧ 
  happy_boys = 6 ∧ 
  neither_boys = 5 ∧ 
  sad_children = total_sad_boys + (sad_children - total_sad_boys) ∧ 
  total_sad_boys = total_boys - happy_boys - neither_boys → 
  (sad_children - total_sad_boys = 4) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_sad_girls_count_l741_74143


namespace NUMINAMATH_GPT_probability_yellow_ball_l741_74157

-- Definitions of the conditions
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := white_balls + yellow_balls

-- Theorem statement
theorem probability_yellow_ball : (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by
  -- Using tactics to facilitate the proof
  simp [yellow_balls, total_balls]
  sorry

end NUMINAMATH_GPT_probability_yellow_ball_l741_74157


namespace NUMINAMATH_GPT_smallest_y_value_l741_74121

theorem smallest_y_value : 
  ∀ y : ℝ, (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_value_l741_74121


namespace NUMINAMATH_GPT_min_value_a_plus_b_plus_c_l741_74153

theorem min_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 9 * a + 4 * b = a * b * c) : a + b + c ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_b_plus_c_l741_74153


namespace NUMINAMATH_GPT_boat_distance_against_stream_l741_74174

-- Definitions from Step a)
def speed_boat_still_water : ℝ := 15  -- speed of the boat in still water in km/hr
def distance_downstream : ℝ := 21  -- distance traveled downstream in one hour in km
def time_hours : ℝ := 1  -- time in hours

-- Translation of the described problem proof
theorem boat_distance_against_stream :
  ∃ (v_s : ℝ), (speed_boat_still_water + v_s = distance_downstream / time_hours) → 
               (15 - v_s = 9) :=
by
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_l741_74174


namespace NUMINAMATH_GPT_angle_between_vectors_is_90_degrees_l741_74181

noncomputable def vec_angle (v₁ v₂ : ℝ × ℝ) : ℝ :=
sorry -- This would be the implementation that calculates the angle between two vectors

theorem angle_between_vectors_is_90_degrees
  (A B C O : ℝ × ℝ)
  (h1 : dist O A = dist O B)
  (h2 : dist O A = dist O C)
  (h3 : dist O B = dist O C)
  (h4 : 2 • (A - O) = (B - O) + (C - O)) :
  vec_angle (B - A) (C - A) = 90 :=
sorry

end NUMINAMATH_GPT_angle_between_vectors_is_90_degrees_l741_74181


namespace NUMINAMATH_GPT_best_of_five_advantageous_l741_74114

theorem best_of_five_advantageous (p : ℝ) (h : p > 0.5) :
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    p2 > p1 :=
by 
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    sorry -- an actual proof would go here

end NUMINAMATH_GPT_best_of_five_advantageous_l741_74114


namespace NUMINAMATH_GPT_logarithmic_inequality_l741_74192

theorem logarithmic_inequality : 
  (a = Real.log 9 / Real.log 2) →
  (b = Real.log 27 / Real.log 3) →
  (c = Real.log 15 / Real.log 5) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_logarithmic_inequality_l741_74192


namespace NUMINAMATH_GPT_last_passenger_probability_l741_74119

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_last_passenger_probability_l741_74119


namespace NUMINAMATH_GPT_problem_statement_l741_74117

noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

noncomputable def a : ℝ :=
1 / Real.logb (1 / 4) (1 / 2015) + 1 / Real.logb (1 / 504) (1 / 2015)

def b : ℝ := 2017

theorem problem_statement :
  (a + b + (a - b) * sgn (a - b)) / 2 = 2017 :=
sorry

end NUMINAMATH_GPT_problem_statement_l741_74117


namespace NUMINAMATH_GPT_cows_number_l741_74112

theorem cows_number (D C : ℕ) (L H : ℕ) 
  (h1 : L = 2 * D + 4 * C)
  (h2 : H = D + C)
  (h3 : L = 2 * H + 12) 
  : C = 6 := 
by
  sorry

end NUMINAMATH_GPT_cows_number_l741_74112


namespace NUMINAMATH_GPT_perimeter_correct_l741_74199

-- Definitions based on the conditions
def large_rectangle_area : ℕ := 12 * 12
def shaded_rectangle_area : ℕ := 6 * 4
def non_shaded_area : ℕ := large_rectangle_area - shaded_rectangle_area
def perimeter_of_non_shaded_region : ℕ := 2 * ((12 - 6) + (12 - 4))

-- The theorem to prove
theorem perimeter_correct (large_rectangle_area_eq : large_rectangle_area = 144) :
  perimeter_of_non_shaded_region = 28 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_correct_l741_74199


namespace NUMINAMATH_GPT_solve_system_of_equations_l741_74125

theorem solve_system_of_equations (u v w : ℝ) (h₀ : u ≠ 0) (h₁ : v ≠ 0) (h₂ : w ≠ 0) :
  (3 / (u * v) + 15 / (v * w) = 2) ∧
  (15 / (v * w) + 5 / (w * u) = 2) ∧
  (5 / (w * u) + 3 / (u * v) = 2) →
  (u = 1 ∧ v = 3 ∧ w = 5) ∨
  (u = -1 ∧ v = -3 ∧ w = -5) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l741_74125


namespace NUMINAMATH_GPT_paths_A_to_C_l741_74168

theorem paths_A_to_C :
  let paths_AB := 2
  let paths_BD := 3
  let paths_DC := 3
  let paths_AC_direct := 1
  paths_AB * paths_BD * paths_DC + paths_AC_direct = 19 :=
by
  sorry

end NUMINAMATH_GPT_paths_A_to_C_l741_74168


namespace NUMINAMATH_GPT_total_dots_not_visible_proof_l741_74172

def total_dots_on_one_die : ℕ := 21

def total_dots_on_five_dice : ℕ := 5 * total_dots_on_one_die

def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

def sum_visible_numbers : ℕ := visible_numbers.sum

def total_dots_not_visible (total : ℕ) (visible_sum : ℕ) : ℕ :=
  total - visible_sum

theorem total_dots_not_visible_proof :
  total_dots_not_visible total_dots_on_five_dice sum_visible_numbers = 81 :=
by
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_proof_l741_74172


namespace NUMINAMATH_GPT_chocolate_bar_count_l741_74180

theorem chocolate_bar_count (bar_weight : ℕ) (box_weight : ℕ) (H1 : bar_weight = 125) (H2 : box_weight = 2000) : box_weight / bar_weight = 16 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bar_count_l741_74180


namespace NUMINAMATH_GPT_find_x_in_interval_l741_74110

theorem find_x_in_interval (x : ℝ) 
  (h₁ : 4 ≤ (x + 1) / (3 * x - 7)) 
  (h₂ : (x + 1) / (3 * x - 7) < 9) : 
  x ∈ Set.Ioc (32 / 13) (29 / 11) := 
sorry

end NUMINAMATH_GPT_find_x_in_interval_l741_74110


namespace NUMINAMATH_GPT_distance_from_circle_to_line_l741_74105

def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def polar_line (θ : ℝ) : Prop := θ = Real.pi / 6

theorem distance_from_circle_to_line : 
  ∃ d : ℝ, polar_circle ρ θ ∧ polar_line θ → d = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_distance_from_circle_to_line_l741_74105


namespace NUMINAMATH_GPT_find_percentage_of_alcohol_in_second_solution_l741_74103

def alcohol_content_second_solution (V2: ℕ) (p1 p2 p_final: ℕ) (V1 V_final: ℕ) : ℕ :=
  ((V_final * p_final) - (V1 * p1)) * 100 / V2

def percentage_correct : Prop :=
  alcohol_content_second_solution 125 20 12 15 75 200 = 12

theorem find_percentage_of_alcohol_in_second_solution : percentage_correct :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_of_alcohol_in_second_solution_l741_74103


namespace NUMINAMATH_GPT_fourth_number_of_expression_l741_74100

theorem fourth_number_of_expression (x : ℝ) (h : 0.3 * 0.8 + 0.1 * x = 0.29) : x = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_fourth_number_of_expression_l741_74100


namespace NUMINAMATH_GPT_mom_tshirts_count_l741_74145

def packages : ℕ := 71
def tshirts_per_package : ℕ := 6

theorem mom_tshirts_count : packages * tshirts_per_package = 426 := by
  sorry

end NUMINAMATH_GPT_mom_tshirts_count_l741_74145


namespace NUMINAMATH_GPT_intersection_A_B_l741_74124

-- Conditions
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = 3 * x - 2 }

-- Question and proof statement
theorem intersection_A_B :
  A ∩ B = {1, 4} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l741_74124


namespace NUMINAMATH_GPT_units_of_Product_C_sold_l741_74135

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_units_of_Product_C_sold_l741_74135


namespace NUMINAMATH_GPT_bumper_car_line_total_in_both_lines_l741_74183

theorem bumper_car_line (x y Z : ℕ) (hZ : Z = 25 - x + y) : Z = 25 - x + y :=
by
  sorry

theorem total_in_both_lines (x y Z : ℕ) (hZ : Z = 25 - x + y) : 40 - x + y = Z + 15 :=
by
  sorry

end NUMINAMATH_GPT_bumper_car_line_total_in_both_lines_l741_74183


namespace NUMINAMATH_GPT_joan_payment_l741_74108

theorem joan_payment (cat_toy_cost cage_cost change_received : ℝ) 
  (h1 : cat_toy_cost = 8.77) 
  (h2 : cage_cost = 10.97) 
  (h3 : change_received = 0.26) : 
  cat_toy_cost + cage_cost - change_received = 19.48 := 
by 
  sorry

end NUMINAMATH_GPT_joan_payment_l741_74108


namespace NUMINAMATH_GPT_sum_squares_divisible_by_7_implies_both_divisible_l741_74155

theorem sum_squares_divisible_by_7_implies_both_divisible (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 7 ∣ a ∧ 7 ∣ b :=
sorry

end NUMINAMATH_GPT_sum_squares_divisible_by_7_implies_both_divisible_l741_74155


namespace NUMINAMATH_GPT_money_distribution_l741_74154

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 360) (h3 : C = 60) : A + B + C = 500 := by
  sorry

end NUMINAMATH_GPT_money_distribution_l741_74154


namespace NUMINAMATH_GPT_fraction_of_full_fare_half_ticket_l741_74148

theorem fraction_of_full_fare_half_ticket (F R : ℝ) 
  (h1 : F + R = 216) 
  (h2 : F + (1/2)*F + 2*R = 327) : 
  (1/2) = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_full_fare_half_ticket_l741_74148


namespace NUMINAMATH_GPT_parabola_hyperbola_tangent_l741_74160

open Real

theorem parabola_hyperbola_tangent (n : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 6 → y^2 - n * x^2 = 4 → y ≥ 6) ↔ (n = 12 + 4 * sqrt 7 ∨ n = 12 - 4 * sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_tangent_l741_74160


namespace NUMINAMATH_GPT_tangent_line_equation_l741_74187

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := 2 * x^2 - x
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Statement of the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (b = 1 - 3 * 1) ∧ 
  (m = 3) ∧ 
  ∀ (x y : ℝ), y = m * x + b → 3 * x - y - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l741_74187


namespace NUMINAMATH_GPT_calculate_savings_l741_74152

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_calculate_savings_l741_74152


namespace NUMINAMATH_GPT_constant_term_zero_quadratic_l741_74159

theorem constant_term_zero_quadratic (m : ℝ) :
  (-m^2 + 1 = 0) → m = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_constant_term_zero_quadratic_l741_74159


namespace NUMINAMATH_GPT_cakes_left_l741_74198

def cakes_yesterday : ℕ := 3
def baked_today : ℕ := 5
def sold_today : ℕ := 6

theorem cakes_left (cakes_yesterday baked_today sold_today : ℕ) : cakes_yesterday + baked_today - sold_today = 2 := by
  sorry

end NUMINAMATH_GPT_cakes_left_l741_74198


namespace NUMINAMATH_GPT_average_candies_l741_74126

theorem average_candies {a b c d e f : ℕ} (h₁ : a = 16) (h₂ : b = 22) (h₃ : c = 30) (h₄ : d = 26) (h₅ : e = 18) (h₆ : f = 20) :
  (a + b + c + d + e + f) / 6 = 22 := by
  sorry

end NUMINAMATH_GPT_average_candies_l741_74126


namespace NUMINAMATH_GPT_numbers_whose_triples_plus_1_are_primes_l741_74150

def is_prime (n : ℕ) : Prop := Nat.Prime n

def in_prime_range (n : ℕ) : Prop := 
  is_prime n ∧ 70 ≤ n ∧ n ≤ 110

def transformed_by_3_and_1 (x : ℕ) : ℕ := 3 * x + 1

theorem numbers_whose_triples_plus_1_are_primes :
  { x : ℕ | in_prime_range (transformed_by_3_and_1 x) } = {24, 26, 32, 34, 36} :=
by
  sorry

end NUMINAMATH_GPT_numbers_whose_triples_plus_1_are_primes_l741_74150


namespace NUMINAMATH_GPT_find_initial_period_l741_74107

theorem find_initial_period (P : ℝ) (T : ℝ) 
  (h1 : 1680 = (P * 4 * T) / 100)
  (h2 : 1680 = (P * 5 * 4) / 100) 
  : T = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_initial_period_l741_74107


namespace NUMINAMATH_GPT_andy_remaining_demerits_l741_74102

-- Definitions based on conditions
def max_demerits : ℕ := 50
def demerits_per_late_instance : ℕ := 2
def late_instances : ℕ := 6
def joke_demerits : ℕ := 15

-- Calculation of total demerits for the month
def total_demerits : ℕ := (demerits_per_late_instance * late_instances) + joke_demerits

-- Proof statement: Andy can receive 23 more demerits without being fired
theorem andy_remaining_demerits : max_demerits - total_demerits = 23 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_andy_remaining_demerits_l741_74102


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l741_74106

theorem common_ratio_of_geometric_sequence (a_1 q : ℝ) (hq : q ≠ 1) 
  (S : ℕ → ℝ) (hS: ∀ n, S n = a_1 * (1 - q^n) / (1 - q))
  (arithmetic_seq : 2 * S 7 = S 8 + S 9) :
  q = -2 :=
by sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l741_74106


namespace NUMINAMATH_GPT_jacks_paycheck_l741_74147

theorem jacks_paycheck (P : ℝ) (h1 : 0.2 * 0.8 * P = 20) : P = 125 :=
sorry

end NUMINAMATH_GPT_jacks_paycheck_l741_74147


namespace NUMINAMATH_GPT_weekly_caloric_allowance_l741_74188

-- Define the given conditions
def average_daily_allowance : ℕ := 2000
def daily_reduction_goal : ℕ := 500
def intense_workout_extra_calories : ℕ := 300
def moderate_exercise_extra_calories : ℕ := 200
def days_intense_workout : ℕ := 2
def days_moderate_exercise : ℕ := 3
def days_rest : ℕ := 2

-- Lean statement to prove the total weekly caloric intake
theorem weekly_caloric_allowance :
  (days_intense_workout * (average_daily_allowance - daily_reduction_goal + intense_workout_extra_calories)) +
  (days_moderate_exercise * (average_daily_allowance - daily_reduction_goal + moderate_exercise_extra_calories)) +
  (days_rest * (average_daily_allowance - daily_reduction_goal)) = 11700 := by
  sorry

end NUMINAMATH_GPT_weekly_caloric_allowance_l741_74188


namespace NUMINAMATH_GPT_number_of_girls_l741_74165

theorem number_of_girls
  (total_students : ℕ)
  (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_non_binary : ℕ)
  (h_ratio : ratio_girls = 3 ∧ ratio_boys = 2 ∧ ratio_non_binary = 1)
  (h_total : total_students = 72) :
  ∃ (k : ℕ), 3 * k = (total_students * 3) / 6 ∧ 3 * k = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l741_74165


namespace NUMINAMATH_GPT_line_connecting_centers_l741_74194

-- Define the first circle equation
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x + 6*y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_eq (x y : ℝ) := 3*x - y - 9 = 0

-- Prove that the line connecting the centers of the circles has the given equation
theorem line_connecting_centers :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → line_eq x y := 
sorry

end NUMINAMATH_GPT_line_connecting_centers_l741_74194


namespace NUMINAMATH_GPT_final_digit_is_two_l741_74123

-- Define initial conditions
def initial_ones : ℕ := 10
def initial_twos : ℕ := 10

-- Define the possible moves and the parity properties
def erase_identical (ones twos : ℕ) : ℕ × ℕ :=
  if ones ≥ 2 then (ones - 2, twos + 1)
  else (ones, twos - 1) -- for the case where two twos are removed

def erase_different (ones twos : ℕ) : ℕ × ℕ :=
  (ones, twos - 1)

-- Theorem stating that the final digit must be a two
theorem final_digit_is_two : 
∀ (ones twos : ℕ), ones = initial_ones → twos = initial_twos → 
(∃ n, ones + twos = n ∧ n = 1 ∧ (ones % 2 = 0)) → 
(∃ n, ones + twos = n ∧ n = 0 ∧ twos = 1) := 
by
  intros ones twos h_ones h_twos condition
  -- Constructing the proof should be done here
  sorry

end NUMINAMATH_GPT_final_digit_is_two_l741_74123


namespace NUMINAMATH_GPT_local_minimum_interval_l741_74138

-- Definitions of the function and its derivative
def y (x a : ℝ) : ℝ := x^3 - 2 * a * x + a
def y_prime (x a : ℝ) : ℝ := 3 * x^2 - 2 * a

-- The proof problem statement
theorem local_minimum_interval (a : ℝ) : 
  (0 < a ∧ a < 3 / 2) ↔ ∃ (x : ℝ), (0 < x ∧ x < 1) ∧ y_prime x a = 0 :=
sorry

end NUMINAMATH_GPT_local_minimum_interval_l741_74138


namespace NUMINAMATH_GPT_beth_sells_half_of_coins_l741_74120

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_beth_sells_half_of_coins_l741_74120


namespace NUMINAMATH_GPT_find_y_l741_74139

theorem find_y (y : ℝ) (h : 3 * y / 7 = 21) : y = 49 := 
sorry

end NUMINAMATH_GPT_find_y_l741_74139


namespace NUMINAMATH_GPT_total_amount_paid_correct_l741_74134

/--
Given:
1. The marked price of each article is $17.5.
2. A discount of 30% was applied to the total marked price of the pair of articles.

Prove:
The total amount paid for the pair of articles is $24.5.
-/
def total_amount_paid (marked_price_each : ℝ) (discount_rate : ℝ) : ℝ :=
  let marked_price_pair := marked_price_each * 2
  let discount := discount_rate * marked_price_pair
  marked_price_pair - discount

theorem total_amount_paid_correct :
  total_amount_paid 17.5 0.30 = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_correct_l741_74134


namespace NUMINAMATH_GPT_birthday_candles_l741_74142

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * candles_Ambika →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intro candles_Ambika candles_Aniyah h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_birthday_candles_l741_74142


namespace NUMINAMATH_GPT_intersection_eq_l741_74111

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l741_74111


namespace NUMINAMATH_GPT_find_natural_number_n_l741_74136

def is_terminating_decimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ), x = (a / b) ∧ (∃ (m n : ℕ), b = 2 ^ m * 5 ^ n)

theorem find_natural_number_n (n : ℕ) (h₁ : is_terminating_decimal (1 / n)) (h₂ : is_terminating_decimal (1 / (n + 1))) : n = 4 :=
by sorry

end NUMINAMATH_GPT_find_natural_number_n_l741_74136


namespace NUMINAMATH_GPT_find_a_l741_74171

theorem find_a (x y a : ℝ) (hx_pos_even : x > 0 ∧ ∃ n : ℕ, x = 2 * n) (hx_le_y : x ≤ y) 
  (h_eq_zero : |3 * y - 18| + |a * x - y| = 0) : 
  a = 3 ∨ a = 3 / 2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l741_74171


namespace NUMINAMATH_GPT_slower_train_speed_is_36_l741_74115

def speed_of_slower_train (v : ℕ) : Prop :=
  let length_of_each_train := 100
  let distance_covered := length_of_each_train * 2
  let time_taken := 72
  let faster_train_speed := 46
  let relative_speed := (faster_train_speed - v) * (1000 / 3600)
  distance_covered = relative_speed * time_taken

theorem slower_train_speed_is_36 : ∃ v, speed_of_slower_train v ∧ v = 36 :=
by
  use 36
  unfold speed_of_slower_train
  -- Prove that the equation holds when v = 36
  sorry

end NUMINAMATH_GPT_slower_train_speed_is_36_l741_74115


namespace NUMINAMATH_GPT_find_5a_plus_5b_l741_74132

noncomputable def g (x : ℝ) : ℝ := 5 * x - 4
noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def f_inv (a b x : ℝ) : ℝ := g x + 3

theorem find_5a_plus_5b (a b : ℝ) (h_inverse : ∀ x, f_inv a b (f a b x) = x) : 5 * a + 5 * b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_5a_plus_5b_l741_74132


namespace NUMINAMATH_GPT_eval_expression_l741_74195

open Real

noncomputable def e : ℝ := 2.71828

theorem eval_expression : abs (5 * e - 15) = 1.4086 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l741_74195


namespace NUMINAMATH_GPT_total_travel_options_l741_74149

theorem total_travel_options (trains_A_to_B : ℕ) (ferries_B_to_C : ℕ) (flights_A_to_C : ℕ) 
  (h1 : trains_A_to_B = 3) (h2 : ferries_B_to_C = 2) (h3 : flights_A_to_C = 2) :
  (trains_A_to_B * ferries_B_to_C + flights_A_to_C = 8) :=
by
  sorry

end NUMINAMATH_GPT_total_travel_options_l741_74149


namespace NUMINAMATH_GPT_determine_true_proposition_l741_74177

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x > 1

def proposition_q : Prop :=
  let focus_distance := 3/4 -- Distance from the focus to the directrix in y = (1/3)x^2
  focus_distance = 1/6

def true_proposition : Prop :=
  proposition_p ∧ ¬proposition_q

theorem determine_true_proposition :
  (proposition_p ∧ ¬proposition_q) = true_proposition :=
by
  sorry -- Proof will go here

end NUMINAMATH_GPT_determine_true_proposition_l741_74177


namespace NUMINAMATH_GPT_correct_proposition_l741_74129

-- Definitions
def p (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬ (x > 1 → x > 2)

def q (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Propositions
def p_and_q (x a b : ℝ) := p x ∧ q a b
def not_p_or_q (x a b : ℝ) := ¬ (p x) ∨ q a b
def p_and_not_q (x a b : ℝ) := p x ∧ ¬ (q a b)
def not_p_and_not_q (x a b : ℝ) := ¬ (p x) ∧ ¬ (q a b)

-- Main theorem
theorem correct_proposition (x a b : ℝ) (h_p : p x) (h_q : ¬ (q a b)) :
  (p_and_q x a b = false) ∧
  (not_p_or_q x a b = false) ∧
  (p_and_not_q x a b = true) ∧
  (not_p_and_not_q x a b = false) :=
by
  sorry

end NUMINAMATH_GPT_correct_proposition_l741_74129


namespace NUMINAMATH_GPT_findTwoHeaviestStonesWith35Weighings_l741_74170

-- Define the problem with conditions
def canFindTwoHeaviestStones (stones : Fin 32 → ℝ) (weighings : ℕ) : Prop :=
  ∀ (balanceScale : (Fin 32 × Fin 32) → Bool), weighings ≤ 35 → 
  ∃ (heaviest : Fin 32) (secondHeaviest : Fin 32), 
  (heaviest ≠ secondHeaviest) ∧ 
  (∀ i : Fin 32, stones heaviest ≥ stones i) ∧ 
  (∀ j : Fin 32, j ≠ heaviest → stones secondHeaviest ≥ stones j)

-- Formally state the theorem
theorem findTwoHeaviestStonesWith35Weighings (stones : Fin 32 → ℝ) :
  canFindTwoHeaviestStones stones 35 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_findTwoHeaviestStonesWith35Weighings_l741_74170


namespace NUMINAMATH_GPT_Joel_laps_count_l741_74113

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end NUMINAMATH_GPT_Joel_laps_count_l741_74113


namespace NUMINAMATH_GPT_find_x_l741_74185

-- Define the condition as a theorem
theorem find_x (x : ℝ) (h : (1 + 3 + x) / 3 = 3) : x = 5 :=
by
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_find_x_l741_74185


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l741_74104

theorem problem1 : 0 - (-22) = 22 := 
by 
  sorry

theorem problem2 : 8.5 - (-1.5) = 10 := 
by 
  sorry

theorem problem3 : (-13 : ℚ) - (4/7) - (-13 : ℚ) - (5/7) = 1/7 := 
by 
  sorry

theorem problem4 : (-1/2 : ℚ) - (1/4 : ℚ) = -3/4 := 
by 
  sorry

theorem problem5 : -51 + 12 + (-7) + (-11) + 36 = -21 := 
by 
  sorry

theorem problem6 : (5/6 : ℚ) + (-2/3) + 1 + (1/6) + (-1/3) = 1 := 
by 
  sorry

theorem problem7 : -13 + (-7) - 20 - (-40) + 16 = 16 := 
by 
  sorry

theorem problem8 : 4.7 - (-8.9) - 7.5 + (-6) = 0.1 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l741_74104


namespace NUMINAMATH_GPT_avg_weight_A_l741_74176

-- Define the conditions
def num_students_A : ℕ := 40
def num_students_B : ℕ := 20
def avg_weight_B : ℝ := 40
def avg_weight_whole_class : ℝ := 46.67

-- State the theorem using these definitions
theorem avg_weight_A :
  ∃ W_A : ℝ,
    (num_students_A * W_A + num_students_B * avg_weight_B = (num_students_A + num_students_B) * avg_weight_whole_class) ∧
    W_A = 50.005 :=
by
  sorry

end NUMINAMATH_GPT_avg_weight_A_l741_74176


namespace NUMINAMATH_GPT_exponent_of_9_in_9_pow_7_l741_74167

theorem exponent_of_9_in_9_pow_7 : ∀ x : ℕ, (3 ^ x ∣ 9 ^ 7) ↔ x ≤ 14 := by
  sorry

end NUMINAMATH_GPT_exponent_of_9_in_9_pow_7_l741_74167


namespace NUMINAMATH_GPT_tens_digit_of_M_l741_74189

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

theorem tens_digit_of_M {M : ℕ} (h : 10 ≤ M ∧ M < 100) (h_eq : M = P M + S M + 6) :
  M / 10 = 1 ∨ M / 10 = 2 :=
sorry

end NUMINAMATH_GPT_tens_digit_of_M_l741_74189


namespace NUMINAMATH_GPT_problem_ab_cd_eq_l741_74158

theorem problem_ab_cd_eq (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 14) :
  ab + cd = 45 := 
by
  sorry

end NUMINAMATH_GPT_problem_ab_cd_eq_l741_74158


namespace NUMINAMATH_GPT_conor_total_vegetables_l741_74128

-- Definitions for each day of the week
def vegetables_per_day_mon_wed : Nat := 12 + 9 + 8 + 15 + 7
def vegetables_per_day_thu_sat : Nat := 7 + 5 + 4 + 10 + 4
def total_vegetables : Nat := 3 * vegetables_per_day_mon_wed + 3 * vegetables_per_day_thu_sat

-- Lean statement for the proof problem
theorem conor_total_vegetables : total_vegetables = 243 := by
  sorry

end NUMINAMATH_GPT_conor_total_vegetables_l741_74128


namespace NUMINAMATH_GPT_exists_natural_2001_digits_l741_74196

theorem exists_natural_2001_digits (N : ℕ) (hN: N = 5 * 10^2000 + 1) : 
  ∃ K : ℕ, (K = N) ∧ (N^(2001) % 10^2001 = N % 10^2001) :=
by
  sorry

end NUMINAMATH_GPT_exists_natural_2001_digits_l741_74196


namespace NUMINAMATH_GPT_legendre_symbol_two_l741_74122

theorem legendre_symbol_two (m : ℕ) [Fact (Nat.Prime m)] (hm : Odd m) :
  (legendreSym 2 m) = (-1 : ℤ) ^ ((m^2 - 1) / 8) :=
sorry

end NUMINAMATH_GPT_legendre_symbol_two_l741_74122


namespace NUMINAMATH_GPT_intersection_point_l741_74164

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 3) / (2 * x - 6)
noncomputable def g (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) (h_asymp : ¬(2 = 0) ∧ (a ≠ 0 ∨ b ≠ 0)) (h_perpendicular : True) (h_y_intersect : g a b c 0 = 1) (h_intersects : f (-1) = g a b c (-1)):
  f 1 = 0 :=
by
  dsimp [f, g] at *
  sorry

end NUMINAMATH_GPT_intersection_point_l741_74164


namespace NUMINAMATH_GPT_perpendicular_vectors_l741_74133

/-- Given vectors a and b which are perpendicular, find the value of m -/
theorem perpendicular_vectors (m : ℝ) (a b : ℝ × ℝ)
  (h1 : a = (2 * m, 1))
  (h2 : b = (1, m - 3))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l741_74133


namespace NUMINAMATH_GPT_staircase_ways_four_steps_l741_74193

theorem staircase_ways_four_steps : 
  let one_step := 1
  let two_steps := 2
  let three_steps := 3
  let four_steps := 4
  1           -- one step at a time
  + 3         -- combination of one and two steps
  + 2         -- combination of one and three steps
  + 1         -- two steps at a time
  + 1 = 8     -- all four steps in one stride
:= by
  sorry

end NUMINAMATH_GPT_staircase_ways_four_steps_l741_74193


namespace NUMINAMATH_GPT_quotient_of_division_l741_74127

theorem quotient_of_division
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1370)
  (h2 : larger = 1626)
  (h3 : ∃ q r, larger = smaller * q + r ∧ r = 15) :
  ∃ q, larger = smaller * q + 15 ∧ q = 6 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_division_l741_74127


namespace NUMINAMATH_GPT_sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l741_74197

-- Definition for the sum of the first n natural numbers
def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition for the sum from 1 to 60
def sum_1_to_60 : ℕ := sum_upto 60

-- Definition for the sum from 1 to 50
def sum_1_to_50 : ℕ := sum_upto 50

-- Proof problem 1
theorem sum_from_1_to_60_is_1830 : sum_1_to_60 = 1830 := 
by
  sorry

-- Definition for the sum from 51 to 60
def sum_51_to_60 : ℕ := sum_1_to_60 - sum_1_to_50

-- Proof problem 2
theorem sum_from_51_to_60_is_555 : sum_51_to_60 = 555 := 
by
  sorry

end NUMINAMATH_GPT_sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l741_74197


namespace NUMINAMATH_GPT_gcd_180_270_eq_90_l741_74191

theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := sorry

end NUMINAMATH_GPT_gcd_180_270_eq_90_l741_74191


namespace NUMINAMATH_GPT_combine_like_terms_l741_74109

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2) * x * y = -5 * x * y := by
  sorry

end NUMINAMATH_GPT_combine_like_terms_l741_74109


namespace NUMINAMATH_GPT_student_loses_one_mark_per_wrong_answer_l741_74144

noncomputable def marks_lost_per_wrong_answer (x : ℝ) : Prop :=
  let total_questions := 60
  let correct_answers := 42
  let wrong_answers := total_questions - correct_answers
  let marks_per_correct := 4
  let total_marks := 150
  correct_answers * marks_per_correct - wrong_answers * x = total_marks

theorem student_loses_one_mark_per_wrong_answer : marks_lost_per_wrong_answer 1 :=
by
  sorry

end NUMINAMATH_GPT_student_loses_one_mark_per_wrong_answer_l741_74144


namespace NUMINAMATH_GPT_house_number_is_fourteen_l741_74166

theorem house_number_is_fourteen (a b c n : ℕ) (h1 : a * b * c = 40) (h2 : a + b + c = n) (h3 : 
  ∃ (a b c : ℕ), a * b * c = 40 ∧ (a = 1 ∧ b = 5 ∧ c = 8) ∨ (a = 2 ∧ b = 2 ∧ c = 10) ∧ n = 14) :
  n = 14 :=
sorry

end NUMINAMATH_GPT_house_number_is_fourteen_l741_74166


namespace NUMINAMATH_GPT_nontrivial_power_of_nat_l741_74118

theorem nontrivial_power_of_nat (n : ℕ) :
  (∃ A p : ℕ, 2^n + 1 = A^p ∧ p > 1) → n = 3 :=
by
  sorry

end NUMINAMATH_GPT_nontrivial_power_of_nat_l741_74118


namespace NUMINAMATH_GPT_sum_of_roots_l741_74162

theorem sum_of_roots (p : ℝ) (h : (4 - p) / 2 = 9) : (p / 2 = 7) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_l741_74162


namespace NUMINAMATH_GPT_length_OC_l741_74186

theorem length_OC (a b : ℝ) (h_perpendicular : ∀ x, x^2 + a * x + b = 0 → x = 1 ∨ x = b) : 
  1 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_length_OC_l741_74186


namespace NUMINAMATH_GPT_total_population_l741_74182

variables (b g t : ℕ)

-- Conditions
def cond1 := b = 4 * g
def cond2 := g = 2 * t

-- Theorem statement
theorem total_population (h1 : cond1 b g) (h2 : cond2 g t) : b + g + t = 11 * b / 8 :=
by sorry

end NUMINAMATH_GPT_total_population_l741_74182


namespace NUMINAMATH_GPT_reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l741_74163

-- Definition: reversing a deck of n cards in k operations
def can_reverse_deck (n k : ℕ) : Prop := sorry -- Placeholder definition

-- Proof Part (a)
theorem reverse_9_in_5_operations :
  can_reverse_deck 9 5 :=
sorry

-- Proof Part (b)
theorem reverse_52_in_27_operations :
  can_reverse_deck 52 27 :=
sorry

-- Proof Part (c)
theorem not_reverse_52_in_17_operations :
  ¬can_reverse_deck 52 17 :=
sorry

-- Proof Part (d)
theorem not_reverse_52_in_26_operations :
  ¬can_reverse_deck 52 26 :=
sorry

end NUMINAMATH_GPT_reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l741_74163


namespace NUMINAMATH_GPT_cos_sq_plus_two_sin_double_l741_74178

theorem cos_sq_plus_two_sin_double (α : ℝ) (h : Real.tan α = 3 / 4) : Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_sq_plus_two_sin_double_l741_74178


namespace NUMINAMATH_GPT_series_divergence_l741_74173

theorem series_divergence (a : ℕ → ℝ) (hdiv : ¬ ∃ l, ∑' n, a n = l) (hpos : ∀ n, a n > 0) (hnoninc : ∀ n m, n ≤ m → a m ≤ a n) : 
  ¬ ∃ l, ∑' n, (a n / (1 + n * a n)) = l :=
by
  sorry

end NUMINAMATH_GPT_series_divergence_l741_74173


namespace NUMINAMATH_GPT_inequality_transformations_l741_74169

theorem inequality_transformations (a b : ℝ) (h : a > b) :
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_transformations_l741_74169


namespace NUMINAMATH_GPT_parabola_line_intersect_at_one_point_l741_74141

theorem parabola_line_intersect_at_one_point :
  ∃ a : ℝ, (∀ x : ℝ, (ax^2 + 5 * x + 2 = -2 * x + 1)) ↔ a = 49 / 4 :=
by sorry

end NUMINAMATH_GPT_parabola_line_intersect_at_one_point_l741_74141


namespace NUMINAMATH_GPT_value_of_expression_at_x_4_l741_74190

theorem value_of_expression_at_x_4 :
  ∀ (x : ℝ), x = 4 → (x^2 - 2 * x - 8) / (x - 4) = 6 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_value_of_expression_at_x_4_l741_74190


namespace NUMINAMATH_GPT_abs_ineq_solution_set_l741_74156

theorem abs_ineq_solution_set (x : ℝ) :
  |x - 5| + |x + 3| ≥ 10 ↔ x ≤ -4 ∨ x ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_solution_set_l741_74156


namespace NUMINAMATH_GPT_mallory_travel_expenses_l741_74179

theorem mallory_travel_expenses (fuel_tank_cost : ℕ) (fuel_tank_miles : ℕ) (total_miles : ℕ) (food_ratio : ℚ)
  (h_fuel_tank_cost : fuel_tank_cost = 45)
  (h_fuel_tank_miles : fuel_tank_miles = 500)
  (h_total_miles : total_miles = 2000)
  (h_food_ratio : food_ratio = 3/5) :
  ∃ total_cost : ℕ, total_cost = 288 :=
by
  sorry

end NUMINAMATH_GPT_mallory_travel_expenses_l741_74179


namespace NUMINAMATH_GPT_part1_part2_l741_74175

-- Definition of p: x² + 2x - 8 < 0
def p (x : ℝ) : Prop := x^2 + 2 * x - 8 < 0

-- Definition of q: (x - 1 + m)(x - 1 - m) ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Define A as the set of real numbers that satisfy p
def A : Set ℝ := { x | p x }

-- Define B as the set of real numbers that satisfy q when m = 2
def B (m : ℝ) : Set ℝ := { x | q x m }

theorem part1 : A ∩ B 2 = { x | -1 ≤ x ∧ x < 2 } :=
sorry

-- Prove that m ≥ 5 is the range for which p is a sufficient but not necessary condition for q
theorem part2 : ∀ m : ℝ, (∀ x: ℝ, p x → q x m) ∧ (∃ x: ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l741_74175


namespace NUMINAMATH_GPT_fewer_free_throws_l741_74131

noncomputable def Deshawn_free_throws : ℕ := 12
noncomputable def Kayla_free_throws : ℕ := Deshawn_free_throws + (Deshawn_free_throws / 2)
noncomputable def Annieka_free_throws : ℕ := 14

theorem fewer_free_throws :
  Annieka_free_throws = Kayla_free_throws - 4 :=
by
  sorry

end NUMINAMATH_GPT_fewer_free_throws_l741_74131


namespace NUMINAMATH_GPT_possible_sums_of_digits_l741_74161

-- Defining the main theorem
theorem possible_sums_of_digits 
  (A B C : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (hdiv : (A + 6 + 2 + 8 + B + 7 + C + 3) % 9 = 0) :
  A + B + C = 1 ∨ A + B + C = 10 ∨ A + B + C = 19 :=
by
  sorry

end NUMINAMATH_GPT_possible_sums_of_digits_l741_74161


namespace NUMINAMATH_GPT_parallel_line_through_point_l741_74130

theorem parallel_line_through_point :
  ∀ {x y : ℝ}, (3 * x + 4 * y + 1 = 0) ∧ (∃ (a b : ℝ), a = 1 ∧ b = 2 ∧ (3 * a + 4 * b + x0 = 0) → (x = -11)) :=
sorry

end NUMINAMATH_GPT_parallel_line_through_point_l741_74130


namespace NUMINAMATH_GPT_scheduling_competitions_l741_74140

-- Define the problem conditions
def scheduling_conditions (gyms : ℕ) (sports : ℕ) (max_sports_per_gym : ℕ) : Prop :=
  gyms = 4 ∧ sports = 3 ∧ max_sports_per_gym = 2

-- Define the main statement
theorem scheduling_competitions :
  scheduling_conditions 4 3 2 →
  (number_of_arrangements = 60) :=
by
  sorry

end NUMINAMATH_GPT_scheduling_competitions_l741_74140


namespace NUMINAMATH_GPT_smallest_w_l741_74137

theorem smallest_w (w : ℕ) (hw : w > 0) (h1 : ∃ k1, 936 * w = 2^5 * k1) (h2 : ∃ k2, 936 * w = 3^3 * k2) (h3 : ∃ k3, 936 * w = 10^2 * k3) : 
  w = 300 :=
by
  sorry

end NUMINAMATH_GPT_smallest_w_l741_74137


namespace NUMINAMATH_GPT_product_value_l741_74101

theorem product_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 = 81 :=
  sorry

end NUMINAMATH_GPT_product_value_l741_74101


namespace NUMINAMATH_GPT_find_a1_l741_74184

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

noncomputable def sumOfArithmeticSequence (a d : α) (n : ℕ) : α :=
  n * a + d * (n * (n - 1) / 2)

theorem find_a1 (a1 d : α) :
  arithmeticSequence a1 d 2 + arithmeticSequence a1 d 8 = 34 →
  sumOfArithmeticSequence a1 d 4 = 38 →
  a1 = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_a1_l741_74184


namespace NUMINAMATH_GPT_cube_side_length_l741_74146

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end NUMINAMATH_GPT_cube_side_length_l741_74146


namespace NUMINAMATH_GPT_number_of_houses_around_square_l741_74116

namespace HouseCounting

-- Definitions for the conditions
def M (k : ℕ) : ℕ := k
def J (k : ℕ) : ℕ := k

-- The main theorem stating the solution
theorem number_of_houses_around_square (n : ℕ)
  (h1 : M 5 % n = J 12 % n)
  (h2 : J 5 % n = M 30 % n) : n = 32 :=
sorry

end HouseCounting

end NUMINAMATH_GPT_number_of_houses_around_square_l741_74116
