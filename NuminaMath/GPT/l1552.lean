import Mathlib

namespace triangle_side_a_l1552_155235

theorem triangle_side_a (a : ℝ) (h1 : 4 < a) (h2 : a < 10) : a = 8 :=
  by
  sorry

end triangle_side_a_l1552_155235


namespace batsman_new_average_l1552_155295

def batsman_average_after_16_innings (A : ℕ) (new_avg : ℕ) (runs_16th : ℕ) : Prop :=
  15 * A + runs_16th = 16 * new_avg

theorem batsman_new_average (A : ℕ) (runs_16th : ℕ) (h1 : batsman_average_after_16_innings A (A + 3) runs_16th) : A + 3 = 19 :=
by
  sorry

end batsman_new_average_l1552_155295


namespace heaviest_and_lightest_in_13_weighings_l1552_155289

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l1552_155289


namespace factorize_expression_l1552_155277

variable (x y : ℝ)

theorem factorize_expression : (x - y) ^ 2 + 2 * y * (x - y) = (x - y) * (x + y) := by
  sorry

end factorize_expression_l1552_155277


namespace perpendicular_vectors_l1552_155217

/-- If vectors a = (1, 2) and b = (x, 4) are perpendicular, then x = -8. -/
theorem perpendicular_vectors (x : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (x, 4)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : x = -8 :=
by {
  sorry
}

end perpendicular_vectors_l1552_155217


namespace hourly_wage_calculation_l1552_155293

variable (H : ℝ)
variable (hours_per_week : ℝ := 40)
variable (wage_per_widget : ℝ := 0.16)
variable (widgets_per_week : ℝ := 500)
variable (total_earnings : ℝ := 580)

theorem hourly_wage_calculation :
  (hours_per_week * H + widgets_per_week * wage_per_widget = total_earnings) →
  H = 12.5 :=
by
  intro h_equation
  -- Proof steps would go here
  sorry

end hourly_wage_calculation_l1552_155293


namespace find_x_given_y_l1552_155297

noncomputable def constantRatio : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (5 * x - 6) / (2 * y + 10) = k

theorem find_x_given_y :
  (constantRatio ∧ (3, 2) ∈ {(x, y) | (5 * x - 6) / (2 * y + 10) = 9 / 14}) →
  ∃ x : ℚ, ((5 * x - 6) / 20 = 9 / 14 ∧ x = 53 / 14) :=
by
  sorry

end find_x_given_y_l1552_155297


namespace price_of_orange_is_60_l1552_155247

-- Given: 
-- 1. The price of each apple is 40 cents.
-- 2. Mary selects 10 pieces of fruit in total.
-- 3. The average price of these 10 pieces is 56 cents.
-- 4. Mary must put back 6 oranges so that the remaining average price is 50 cents.
-- Prove: The price of each orange is 60 cents.

theorem price_of_orange_is_60 (a o : ℕ) (x : ℕ) 
  (h1 : a + o = 10)
  (h2 : 40 * a + x * o = 560)
  (h3 : 40 * a + x * (o - 6) = 200) : 
  x = 60 :=
by
  have eq1 : 40 * a + x * o = 560 := h2
  have eq2 : 40 * a + x * (o - 6) = 200 := h3
  sorry

end price_of_orange_is_60_l1552_155247


namespace final_price_of_book_l1552_155241

theorem final_price_of_book (original_price : ℝ) (d1_percentage : ℝ) (d2_percentage : ℝ) 
  (first_discount : ℝ) (second_discount : ℝ) (new_price1 : ℝ) (final_price : ℝ) :
  original_price = 15 ∧ d1_percentage = 0.20 ∧ d2_percentage = 0.25 ∧
  first_discount = d1_percentage * original_price ∧ new_price1 = original_price - first_discount ∧
  second_discount = d2_percentage * new_price1 ∧ 
  final_price = new_price1 - second_discount → final_price = 9 := 
by 
  sorry

end final_price_of_book_l1552_155241


namespace anna_money_ratio_l1552_155226

theorem anna_money_ratio (total_money spent_furniture left_money given_to_Anna : ℕ)
  (h_total : total_money = 2000)
  (h_spent : spent_furniture = 400)
  (h_left : left_money = 400)
  (h_after_furniture : total_money - spent_furniture = given_to_Anna + left_money) :
  (given_to_Anna / left_money) = 3 :=
by
  have h1 : total_money - spent_furniture = 1600 := by sorry
  have h2 : given_to_Anna = 1200 := by sorry
  have h3 : given_to_Anna / left_money = 3 := by sorry
  exact h3

end anna_money_ratio_l1552_155226


namespace percentage_of_3rd_graders_l1552_155259

theorem percentage_of_3rd_graders (students_jackson students_madison : ℕ)
  (percent_3rd_grade_jackson percent_3rd_grade_madison : ℝ) :
  students_jackson = 200 → percent_3rd_grade_jackson = 25 →
  students_madison = 300 → percent_3rd_grade_madison = 35 →
  ((percent_3rd_grade_jackson / 100 * students_jackson +
    percent_3rd_grade_madison / 100 * students_madison) /
   (students_jackson + students_madison) * 100) = 31 :=
by 
  intros hjackson_percent hmpercent 
    hpercent_jack_percent hpercent_mad_percent
  -- Proof Placeholder
  sorry

end percentage_of_3rd_graders_l1552_155259


namespace maximum_xy_l1552_155255

theorem maximum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : xy ≤ 2 :=
sorry

end maximum_xy_l1552_155255


namespace max_value_expression_l1552_155287

theorem max_value_expression (a b : ℝ) (ha: 0 < a) (hb: 0 < b) :
  ∃ M, M = 2 * Real.sqrt 87 ∧
       (∀ a b: ℝ, 0 < a → 0 < b →
       (|4 * a - 10 * b| + |2 * (a - b * Real.sqrt 3) - 5 * (a * Real.sqrt 3 + b)|) / Real.sqrt (a ^ 2 + b ^ 2) ≤ M) :=
sorry

end max_value_expression_l1552_155287


namespace negative_number_is_d_l1552_155206

def a : Int := -(-2)
def b : Int := abs (-2)
def c : Int := (-2) ^ 2
def d : Int := (-2) ^ 3

theorem negative_number_is_d : d < 0 :=
  by
  sorry

end negative_number_is_d_l1552_155206


namespace diana_hourly_wage_l1552_155280

theorem diana_hourly_wage :
  (∃ (hours_monday : ℕ) (hours_tuesday : ℕ) (hours_wednesday : ℕ) (hours_thursday : ℕ) (hours_friday : ℕ) (weekly_earnings : ℝ),
    hours_monday = 10 ∧
    hours_tuesday = 15 ∧
    hours_wednesday = 10 ∧
    hours_thursday = 15 ∧
    hours_friday = 10 ∧
    weekly_earnings = 1800 ∧
    (weekly_earnings / (hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday) = 30)) :=
sorry

end diana_hourly_wage_l1552_155280


namespace white_line_longer_l1552_155253

theorem white_line_longer :
  let white_line := 7.67
  let blue_line := 3.33
  white_line - blue_line = 4.34 := by
  sorry

end white_line_longer_l1552_155253


namespace fish_left_in_tank_l1552_155237

theorem fish_left_in_tank (initial_fish : ℕ) (fish_taken_out : ℕ) (fish_left : ℕ) 
  (h1 : initial_fish = 19) (h2 : fish_taken_out = 16) : fish_left = initial_fish - fish_taken_out :=
by
  simp [h1, h2]
  sorry

end fish_left_in_tank_l1552_155237


namespace factorize_expression_l1552_155218

theorem factorize_expression (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := 
by
  sorry

end factorize_expression_l1552_155218


namespace odd_divisors_l1552_155229

-- Define p_1, p_2, p_3 as distinct prime numbers greater than 3
variables {p_1 p_2 p_3 : ℕ}
-- Define k, a, b, c as positive integers
variables {n k a b c : ℕ}

-- The conditions
def distinct_primes (p_1 p_2 p_3 : ℕ) : Prop :=
  p_1 > 3 ∧ p_2 > 3 ∧ p_3 > 3 ∧ p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_2 ≠ p_3

def is_n (n k p_1 p_2 p_3 a b c : ℕ) : Prop :=
  n = 2^k * p_1^a * p_2^b * p_3^c

def conditions (a b c : ℕ) : Prop :=
  a + b > c ∧ 1 ≤ b ∧ b ≤ c

-- The main statement
theorem odd_divisors
  (h_prime : distinct_primes p_1 p_2 p_3)
  (h_n : is_n n k p_1 p_2 p_3 a b c)
  (h_cond : conditions a b c) : 
  ∃ d : ℕ, d = (a + 1) * (b + 1) * (c + 1) :=
by sorry

end odd_divisors_l1552_155229


namespace log_order_l1552_155240

theorem log_order (a b c : ℝ) (h_a : a = Real.log 6 / Real.log 2) 
  (h_b : b = Real.log 15 / Real.log 5) (h_c : c = Real.log 21 / Real.log 7) : 
  a > b ∧ b > c := by sorry

end log_order_l1552_155240


namespace relatively_prime_27x_plus_4_18x_plus_3_l1552_155273

theorem relatively_prime_27x_plus_4_18x_plus_3 (x : ℕ) :
  Nat.gcd (27 * x + 4) (18 * x + 3) = 1 :=
sorry

end relatively_prime_27x_plus_4_18x_plus_3_l1552_155273


namespace interval_between_segments_l1552_155216

def population_size : ℕ := 800
def sample_size : ℕ := 40

theorem interval_between_segments : population_size / sample_size = 20 :=
by
  -- Insert proof here
  sorry

end interval_between_segments_l1552_155216


namespace minimum_tickets_needed_l1552_155267

noncomputable def min_tickets {α : Type*} (winning_permutation : Fin 50 → α) (tickets : List (Fin 50 → α)) : ℕ :=
  List.length tickets

theorem minimum_tickets_needed
  (winning_permutation : Fin 50 → ℕ)
  (tickets : List (Fin 50 → ℕ))
  (h_tickets_valid : ∀ t ∈ tickets, Function.Surjective t)
  (h_at_least_one_match : ∀ winning_permutation : Fin 50 → ℕ,
      ∃ t ∈ tickets, ∃ i : Fin 50, t i = winning_permutation i) : 
  min_tickets winning_permutation tickets ≥ 26 :=
sorry

end minimum_tickets_needed_l1552_155267


namespace girls_boys_ratio_l1552_155236

theorem girls_boys_ratio (G B : ℕ) (h1 : G + B = 100) (h2 : 0.20 * (G : ℝ) + 0.10 * (B : ℝ) = 15) : G / B = 1 :=
by
  -- Proof steps are omitted
  sorry

end girls_boys_ratio_l1552_155236


namespace paula_go_kart_rides_l1552_155238

theorem paula_go_kart_rides
  (g : ℕ)
  (ticket_cost_go_karts : ℕ := 4 * g)
  (ticket_cost_bumper_cars : ℕ := 20)
  (total_tickets : ℕ := 24) :
  ticket_cost_go_karts + ticket_cost_bumper_cars = total_tickets → g = 1 :=
by {
  sorry
}

end paula_go_kart_rides_l1552_155238


namespace square_side_length_increase_l1552_155231

variables {a x : ℝ}

theorem square_side_length_increase 
  (h : (a * (1 + x / 100) * 1.8)^2 = (1 + 159.20000000000002 / 100) * (a^2 + (a * (1 + x / 100))^2)) : 
  x = 100 :=
by sorry

end square_side_length_increase_l1552_155231


namespace exercise_b_c_values_l1552_155233

open Set

universe u

theorem exercise_b_c_values : 
  ∀ (b c : ℝ), let U : Set ℝ := {2, 3, 5}
               let A : Set ℝ := {x | x^2 + b * x + c = 0}
               (U \ A = {2}) → (b = -8 ∧ c = 15) :=
by
  intros b c U A H
  let U : Set ℝ := {2, 3, 5}
  let A : Set ℝ := {x | x^2 + b * x + c = 0}
  have H1 : U \ A = {2} := H
  sorry

end exercise_b_c_values_l1552_155233


namespace polygon_sides_l1552_155246

theorem polygon_sides (side_length perimeter : ℕ) (h1 : side_length = 4) (h2 : perimeter = 24) : 
  perimeter / side_length = 6 :=
by 
  sorry

end polygon_sides_l1552_155246


namespace wheel_center_travel_distance_l1552_155215

theorem wheel_center_travel_distance (radius : ℝ) (revolutions : ℝ) (flat_surface : Prop) 
  (h_radius : radius = 2) (h_revolutions : revolutions = 2) : 
  radius * 2 * π * revolutions = 8 * π :=
by
  rw [h_radius, h_revolutions]
  simp [mul_assoc, mul_comm]
  sorry

end wheel_center_travel_distance_l1552_155215


namespace Jerry_wants_to_raise_average_l1552_155254

theorem Jerry_wants_to_raise_average 
  (first_three_tests_avg : ℕ) (fourth_test_score : ℕ) (desired_increase : ℕ) 
  (h1 : first_three_tests_avg = 90) (h2 : fourth_test_score = 98) 
  : desired_increase = 2 := 
by
  sorry

end Jerry_wants_to_raise_average_l1552_155254


namespace journey_ratio_l1552_155227

/-- Given a full-circle journey broken into parts,
  including paths through the Zoo Park (Z), the Circus (C), and the Park (P), 
  prove that the journey avoiding the Zoo Park is 11 times shorter. -/
theorem journey_ratio (Z C P : ℝ) (h1 : C = (3 / 4) * Z) 
                      (h2 : P = (1 / 4) * Z) : 
  Z = 11 * P := 
sorry

end journey_ratio_l1552_155227


namespace algebra_problem_l1552_155219

theorem algebra_problem 
  (a : ℝ) 
  (h : a^3 + 3 * a^2 + 3 * a + 2 = 0) :
  (a + 1) ^ 2008 + (a + 1) ^ 2009 + (a + 1) ^ 2010 = 1 :=
by 
  sorry

end algebra_problem_l1552_155219


namespace salty_cookies_initial_at_least_34_l1552_155266

variable {S : ℕ}  -- S will represent the initial number of salty cookies

-- Conditions from the problem
def sweet_cookies_initial := 8
def sweet_cookies_ate := 20
def salty_cookies_ate := 34
def more_salty_than_sweet := 14

theorem salty_cookies_initial_at_least_34 :
  8 = sweet_cookies_initial ∧
  20 = sweet_cookies_ate ∧
  34 = salty_cookies_ate ∧
  salty_cookies_ate = sweet_cookies_ate + more_salty_than_sweet
  → S ≥ 34 :=
by sorry

end salty_cookies_initial_at_least_34_l1552_155266


namespace volume_of_ABDH_is_4_3_l1552_155207

-- Define the vertices of the cube
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def D : (ℝ × ℝ × ℝ) := (0, 2, 0)
def H : (ℝ × ℝ × ℝ) := (0, 0, 2)

-- Function to calculate the volume of the pyramid
noncomputable def volume_of_pyramid (A B D H : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * 2 * 2 * 2

-- Theorem stating the volume of the pyramid ABDH is 4/3 cubic units
theorem volume_of_ABDH_is_4_3 : volume_of_pyramid A B D H = 4 / 3 := by
  sorry

end volume_of_ABDH_is_4_3_l1552_155207


namespace spring_mass_relationship_l1552_155275

theorem spring_mass_relationship (x y : ℕ) (h1 : y = 18 + 2 * x) : 
  y = 32 → x = 7 :=
by
  sorry

end spring_mass_relationship_l1552_155275


namespace distinct_real_roots_of_quadratic_l1552_155283

theorem distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, x^2 - 4*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂)) ↔ m < 2 := by
sorry

end distinct_real_roots_of_quadratic_l1552_155283


namespace locus_of_P_coordinates_of_P_l1552_155286

-- Define the points A and B
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (2, -1)

-- Define the line l : 4x + 3y - 2 = 0
def l (x y: ℝ) := 4 * x + 3 * y - 2 = 0

-- Problem (1): Equation of the locus of point P such that |PA| = |PB|
theorem locus_of_P (P : ℝ × ℝ) :
  (∃ P, dist P A = dist P B) ↔ (∀ x y : ℝ, P = (x, y) → x - y - 5 = 0) :=
sorry

-- Problem (2): Coordinates of P such that |PA| = |PB| and the distance from P to line l is 2
theorem coordinates_of_P (a b : ℝ):
  (dist (a, b) A = dist (a, b) B ∧ abs (4 * a + 3 * b - 2) / 5 = 2) ↔
  ((a = 1 ∧ b = -4) ∨ (a = 27 / 7 ∧ b = -8 / 7)) :=
sorry

end locus_of_P_coordinates_of_P_l1552_155286


namespace count_neither_3_nor_4_l1552_155230

def is_multiple_of_3_or_4 (n : Nat) : Bool := (n % 3 = 0) ∨ (n % 4 = 0)

def three_digit_numbers := List.range' 100 900 -- Generates a list from 100 to 999 (inclusive)

def count_multiples_of_3_or_4 : Nat := three_digit_numbers.filter is_multiple_of_3_or_4 |>.length

def count_total := 900 -- Since three-digit numbers range from 100 to 999

theorem count_neither_3_nor_4 : count_total - count_multiples_of_3_or_4 = 450 := by
  sorry

end count_neither_3_nor_4_l1552_155230


namespace maximum_value_parabola_l1552_155281

theorem maximum_value_parabola (x : ℝ) : 
  ∃ y : ℝ, y = -3 * x^2 + 6 ∧ ∀ z : ℝ, (∃ a : ℝ, z = -3 * a^2 + 6) → z ≤ 6 :=
by
  sorry

end maximum_value_parabola_l1552_155281


namespace minimum_filtration_process_l1552_155274

noncomputable def filtration_process (n : ℕ) : Prop :=
  (0.8 : ℝ) ^ n < 0.05

theorem minimum_filtration_process : ∃ n : ℕ, filtration_process n ∧ n ≥ 14 := 
  sorry

end minimum_filtration_process_l1552_155274


namespace derivative_of_log_base2_inv_x_l1552_155200

noncomputable def my_function (x : ℝ) : ℝ := (Real.log x⁻¹) / (Real.log 2)

theorem derivative_of_log_base2_inv_x : 
  ∀ x : ℝ, x > 0 → deriv my_function x = -1 / (x * Real.log 2) :=
by
  intros x hx
  sorry

end derivative_of_log_base2_inv_x_l1552_155200


namespace how_many_trucks_l1552_155272

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end how_many_trucks_l1552_155272


namespace candy_necklaces_l1552_155291

theorem candy_necklaces (friends : ℕ) (candies_per_necklace : ℕ) (candies_per_block : ℕ)(blocks_needed : ℕ):
  friends = 8 →
  candies_per_necklace = 10 →
  candies_per_block = 30 →
  80 / 30 > 2.67 →
  blocks_needed = 3 :=
by
  intros
  sorry

end candy_necklaces_l1552_155291


namespace dividend_is_10_l1552_155232

theorem dividend_is_10
  (q d r : ℕ)
  (hq : q = 3)
  (hd : d = 3)
  (hr : d = 3 * r) :
  (q * d + r = 10) :=
by
  sorry

end dividend_is_10_l1552_155232


namespace part1_part2_l1552_155262

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

def calculate_P (x y : ℤ) : ℤ := 
  (x - y) / 9

def y_from_x (x : ℤ) : ℤ :=
  let first_three := x / 10
  let last_digit := x % 10
  last_digit * 1000 + first_three

def calculate_s (a b : ℕ) : ℤ :=
  1100 + 20 * a + b

def calculate_t (a b : ℕ) : ℤ :=
  b * 1000 + a * 100 + 23

theorem part1 : calculate_P 5324 (y_from_x 5324) = 88 := by
  sorry

theorem part2 :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 9 ∧
  let s := calculate_s a b
  let t := calculate_t a b
  let P_s := calculate_P s (y_from_x s)
  let P_t := calculate_P t (y_from_x t)
  let difference := P_t - P_s - a - b
  is_perfect_square difference ∧ P_t = -161 := by
  sorry

end part1_part2_l1552_155262


namespace train_average_speed_with_stoppages_l1552_155202

theorem train_average_speed_with_stoppages (D : ℝ) :
  let speed_without_stoppages := 200
  let stoppage_time_per_hour_in_hours := 12 / 60.0
  let effective_running_time := 1 - stoppage_time_per_hour_in_hours
  let speed_with_stoppages := effective_running_time * speed_without_stoppages
  speed_with_stoppages = 160 := by
  sorry

end train_average_speed_with_stoppages_l1552_155202


namespace total_students_in_classrooms_l1552_155209

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end total_students_in_classrooms_l1552_155209


namespace small_seat_capacity_l1552_155276

-- Definitions for the conditions
def smallSeats : Nat := 2
def largeSeats : Nat := 23
def capacityLargeSeat : Nat := 54
def totalPeopleSmallSeats : Nat := 28

-- Theorem statement
theorem small_seat_capacity : totalPeopleSmallSeats / smallSeats = 14 := by
  sorry

end small_seat_capacity_l1552_155276


namespace initial_pencils_l1552_155205

theorem initial_pencils (pencils_added initial_pencils total_pencils : ℕ) 
  (h1 : pencils_added = 3) 
  (h2 : total_pencils = 5) :
  initial_pencils = total_pencils - pencils_added := 
by 
  sorry

end initial_pencils_l1552_155205


namespace rachel_total_apples_l1552_155201

noncomputable def totalRemainingApples (X : ℕ) : ℕ :=
  let remainingFirstFour := 10 + 40 + 15 + 22
  let remainingOtherTrees := 48 * X
  remainingFirstFour + remainingOtherTrees

theorem rachel_total_apples (X : ℕ) :
  totalRemainingApples X = 87 + 48 * X :=
by
  sorry

end rachel_total_apples_l1552_155201


namespace least_five_digit_perfect_square_and_cube_l1552_155244

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l1552_155244


namespace total_sets_needed_l1552_155220

-- Conditions
variable (n : ℕ)

-- Theorem statement
theorem total_sets_needed : 3 * n = 3 * n :=
by sorry

end total_sets_needed_l1552_155220


namespace intersection_M_N_l1552_155203

def M : Set ℤ := { x | x^2 > 1 }
def N : Set ℤ := { -2, -1, 0, 1, 2 }

theorem intersection_M_N : (M ∩ N) = { -2, 2 } :=
sorry

end intersection_M_N_l1552_155203


namespace radius_is_independent_variable_l1552_155210

theorem radius_is_independent_variable 
  (r C : ℝ)
  (h : C = 2 * Real.pi * r) : 
  ∃ r_independent, r_independent = r := 
by
  sorry

end radius_is_independent_variable_l1552_155210


namespace avg_last_four_is_63_75_l1552_155224

noncomputable def average_of_list (l : List ℝ) : ℝ :=
  l.sum / l.length

variable (l : List ℝ)
variable (h_lenl : l.length = 7)
variable (h_avg7 : average_of_list l = 60)
variable (h_l3 : List ℝ := l.take 3)
variable (h_l4 : List ℝ := l.drop 3)
variable (h_avg3 : average_of_list h_l3 = 55)

theorem avg_last_four_is_63_75 : average_of_list h_l4 = 63.75 :=
by
  sorry

end avg_last_four_is_63_75_l1552_155224


namespace proof_problem_l1552_155264

theorem proof_problem (x : ℝ) (h : x < 1) : -2 * x + 2 > 0 :=
by
  sorry

end proof_problem_l1552_155264


namespace tom_gave_8_boxes_l1552_155228

-- Define the given conditions and the question in terms of variables
variables (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (boxes_given : ℕ)

-- Specify the actual values for the given problem
def tom_initial_pieces := total_boxes * pieces_per_box
def pieces_given := tom_initial_pieces - pieces_left
def calculated_boxes_given := pieces_given / pieces_per_box

-- Prove the number of boxes Tom gave to his little brother
theorem tom_gave_8_boxes
  (h1 : total_boxes = 14)
  (h2 : pieces_per_box = 3)
  (h3 : pieces_left = 18)
  (h4 : calculated_boxes_given = boxes_given) :
  boxes_given = 8 :=
by
  sorry

end tom_gave_8_boxes_l1552_155228


namespace connor_sleep_duration_l1552_155252

variables {Connor_sleep Luke_sleep Puppy_sleep : ℕ}

def sleeps_two_hours_longer (Luke_sleep Connor_sleep : ℕ) : Prop :=
  Luke_sleep = Connor_sleep + 2

def sleeps_twice_as_long (Puppy_sleep Luke_sleep : ℕ) : Prop :=
  Puppy_sleep = 2 * Luke_sleep

def sleeps_sixteen_hours (Puppy_sleep : ℕ) : Prop :=
  Puppy_sleep = 16

theorem connor_sleep_duration 
  (h1 : sleeps_two_hours_longer Luke_sleep Connor_sleep)
  (h2 : sleeps_twice_as_long Puppy_sleep Luke_sleep)
  (h3 : sleeps_sixteen_hours Puppy_sleep) :
  Connor_sleep = 6 :=
by {
  sorry
}

end connor_sleep_duration_l1552_155252


namespace geometric_sequence_solution_l1552_155263

-- Define the geometric sequence a_n with a common ratio q and first term a_1
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q^n

-- Given conditions in the problem
variables {a : ℕ → ℝ} {q a1 : ℝ}

-- Common ratio is greater than 1
axiom ratio_gt_one : q > 1

-- Given conditions a_3a_7 = 72 and a_2 + a_8 = 27
axiom condition1 : a 3 * a 7 = 72
axiom condition2 : a 2 + a 8 = 27

-- Defining the property that we are looking to prove a_12 = 96
theorem geometric_sequence_solution :
  geometric_sequence a a1 q →
  a 12 = 96 :=
by
  -- This part of the proof would be filled in
  -- Show the conditions and relations leading to the solution a_12 = 96
  sorry

end geometric_sequence_solution_l1552_155263


namespace number_of_schools_in_pythagoras_city_l1552_155271

theorem number_of_schools_in_pythagoras_city (n : ℕ) (h1 : true) 
    (h2 : true) (h3 : ∃ m, m = (3 * n + 1) / 2)
    (h4 : true) (h5 : true) : n = 24 :=
by 
  have h6 : 69 < 3 * n := sorry
  have h7 : 3 * n < 79 := sorry
  sorry

end number_of_schools_in_pythagoras_city_l1552_155271


namespace average_speed_of_bike_l1552_155294

theorem average_speed_of_bike (distance : ℕ) (time : ℕ) (h1 : distance = 21) (h2 : time = 7) : distance / time = 3 := by
  sorry

end average_speed_of_bike_l1552_155294


namespace value_of_M_l1552_155257

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 1200) : M = 1680 := 
sorry

end value_of_M_l1552_155257


namespace altitudes_order_l1552_155211

variable {A a b c h_a h_b h_c : ℝ}

-- Conditions
axiom area_eq : A = (1/2) * a * h_a
axiom area_eq_b : A = (1/2) * b * h_b
axiom area_eq_c : A = (1/2) * c * h_c
axiom sides_order : a > b ∧ b > c

-- Conclusion
theorem altitudes_order : h_a < h_b ∧ h_b < h_c :=
by
  sorry

end altitudes_order_l1552_155211


namespace pages_called_this_week_l1552_155278

-- Definitions as per conditions
def pages_called_last_week := 10.2
def total_pages_called := 18.8

-- Theorem to prove the solution
theorem pages_called_this_week :
  total_pages_called - pages_called_last_week = 8.6 :=
by
  sorry

end pages_called_this_week_l1552_155278


namespace flowers_given_l1552_155245

theorem flowers_given (initial_flowers total_flowers flowers_given : ℕ) 
  (h1 : initial_flowers = 67) 
  (h2 : total_flowers = 90) 
  (h3 : total_flowers = initial_flowers + flowers_given) : 
  flowers_given = 23 :=
by {
  sorry
}

end flowers_given_l1552_155245


namespace range_of_x_l1552_155249

open Set

noncomputable def M (x : ℝ) : Set ℝ := {x^2, 1}

theorem range_of_x (x : ℝ) (hx : M x) : x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end range_of_x_l1552_155249


namespace original_people_in_room_l1552_155288

theorem original_people_in_room (x : ℕ) (h1 : 18 = (2 * x / 3) - (x / 6)) : x = 36 :=
by sorry

end original_people_in_room_l1552_155288


namespace equivalent_spherical_coords_l1552_155285

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end equivalent_spherical_coords_l1552_155285


namespace third_part_of_156_division_proof_l1552_155290

theorem third_part_of_156_division_proof :
  ∃ (x : ℚ), (2 * x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 156) ∧ ((1 / 4) * x = 13 + 15 / 23) :=
by
  sorry

end third_part_of_156_division_proof_l1552_155290


namespace tangent_product_equals_2_pow_23_l1552_155248

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180)) *
  (1 + Real.tan (2 * Real.pi / 180)) *
  (1 + Real.tan (3 * Real.pi / 180)) *
  (1 + Real.tan (4 * Real.pi / 180)) *
  (1 + Real.tan (5 * Real.pi / 180)) *
  (1 + Real.tan (6 * Real.pi / 180)) *
  (1 + Real.tan (7 * Real.pi / 180)) *
  (1 + Real.tan (8 * Real.pi / 180)) *
  (1 + Real.tan (9 * Real.pi / 180)) *
  (1 + Real.tan (10 * Real.pi / 180)) *
  (1 + Real.tan (11 * Real.pi / 180)) *
  (1 + Real.tan (12 * Real.pi / 180)) *
  (1 + Real.tan (13 * Real.pi / 180)) *
  (1 + Real.tan (14 * Real.pi / 180)) *
  (1 + Real.tan (15 * Real.pi / 180)) *
  (1 + Real.tan (16 * Real.pi / 180)) *
  (1 + Real.tan (17 * Real.pi / 180)) *
  (1 + Real.tan (18 * Real.pi / 180)) *
  (1 + Real.tan (19 * Real.pi / 180)) *
  (1 + Real.tan (20 * Real.pi / 180)) *
  (1 + Real.tan (21 * Real.pi / 180)) *
  (1 + Real.tan (22 * Real.pi / 180)) *
  (1 + Real.tan (23 * Real.pi / 180)) *
  (1 + Real.tan (24 * Real.pi / 180)) *
  (1 + Real.tan (25 * Real.pi / 180)) *
  (1 + Real.tan (26 * Real.pi / 180)) *
  (1 + Real.tan (27 * Real.pi / 180)) *
  (1 + Real.tan (28 * Real.pi / 180)) *
  (1 + Real.tan (29 * Real.pi / 180)) *
  (1 + Real.tan (30 * Real.pi / 180)) *
  (1 + Real.tan (31 * Real.pi / 180)) *
  (1 + Real.tan (32 * Real.pi / 180)) *
  (1 + Real.tan (33 * Real.pi / 180)) *
  (1 + Real.tan (34 * Real.pi / 180)) *
  (1 + Real.tan (35 * Real.pi / 180)) *
  (1 + Real.tan (36 * Real.pi / 180)) *
  (1 + Real.tan (37 * Real.pi / 180)) *
  (1 + Real.tan (38 * Real.pi / 180)) *
  (1 + Real.tan (39 * Real.pi / 180)) *
  (1 + Real.tan (40 * Real.pi / 180)) *
  (1 + Real.tan (41 * Real.pi / 180)) *
  (1 + Real.tan (42 * Real.pi / 180)) *
  (1 + Real.tan (43 * Real.pi / 180)) *
  (1 + Real.tan (44 * Real.pi / 180)) *
  (1 + Real.tan (45 * Real.pi / 180))

theorem tangent_product_equals_2_pow_23 : tangent_product = 2 ^ 23 :=
  sorry

end tangent_product_equals_2_pow_23_l1552_155248


namespace product_of_three_numbers_l1552_155268

theorem product_of_three_numbers :
  ∃ (x y z : ℚ), 
    (x + y + z = 30) ∧ 
    (x = 3 * (y + z)) ∧ 
    (y = 8 * z) ∧ 
    (x * y * z = 125) := 
by
  sorry

end product_of_three_numbers_l1552_155268


namespace fraction_of_single_men_l1552_155204

theorem fraction_of_single_men :
  ∀ (total_faculty : ℕ) (women_percentage : ℝ) (married_percentage : ℝ) (married_men_ratio : ℝ),
    women_percentage = 0.7 → married_percentage = 0.4 → married_men_ratio = 2/3 →
    (total_faculty * (1 - women_percentage)) * (1 - married_men_ratio) / 
    (total_faculty * (1 - women_percentage)) = 1/3 :=
by 
  intros total_faculty women_percentage married_percentage married_men_ratio h_women h_married h_men_marry
  sorry

end fraction_of_single_men_l1552_155204


namespace find_b_l1552_155239

theorem find_b (a b c : ℝ) (h1 : a + b + c = 150) (h2 : a + 10 = c^2) (h3 : b - 5 = c^2) : 
  b = (1322 - 2 * Real.sqrt 1241) / 16 := 
by 
  sorry

end find_b_l1552_155239


namespace red_balls_count_l1552_155234

theorem red_balls_count (R W N_1 N_2 : ℕ) 
  (h1 : R - 2 * N_1 = 18) 
  (h2 : W = 3 * N_1) 
  (h3 : R - 5 * N_2 = 0) 
  (h4 : W - 3 * N_2 = 18)
  : R = 50 :=
sorry

end red_balls_count_l1552_155234


namespace gcd_a_b_eq_one_l1552_155296

def a : ℕ := 47^5 + 1
def b : ℕ := 47^5 + 47^3 + 1

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l1552_155296


namespace general_term_formula_l1552_155299

def seq (n : ℕ) : ℤ :=
match n with
| 0       => 1
| 1       => -3
| 2       => 5
| 3       => -7
| 4       => 9
| (n + 1) => (-1)^(n+1) * (2*n + 1) -- extends indefinitely for general natural number

theorem general_term_formula (n : ℕ) : 
  seq n = (-1)^(n+1) * (2*n-1) :=
sorry

end general_term_formula_l1552_155299


namespace subcommittees_with_at_least_one_teacher_l1552_155270

theorem subcommittees_with_at_least_one_teacher :
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  total_subcommittees - non_teacher_subcommittees = 460 :=
by
  -- Definitions and conditions based on the problem statement
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  sorry -- Proof goes here

end subcommittees_with_at_least_one_teacher_l1552_155270


namespace part1_part2_l1552_155214

noncomputable def f (x a : ℝ) := (x + 1) * Real.log x - a * (x - 1)

theorem part1 : (∀ x a : ℝ, (x + 1) * Real.log x - a * (x - 1) = x - 1 → a = 1) := 
by sorry

theorem part2 (x : ℝ) (h : 1 < x ∧ x < 2) : 
  ( 1 / Real.log x - 1 / Real.log (x - 1) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end part1_part2_l1552_155214


namespace jill_total_trip_duration_is_101_l1552_155221

def first_bus_wait_time : Nat := 12
def first_bus_ride_time : Nat := 30
def first_bus_delay_time : Nat := 5

def walk_time_to_train : Nat := 10
def train_wait_time : Nat := 8
def train_ride_time : Nat := 20
def train_delay_time : Nat := 3

def second_bus_wait_time : Nat := 20
def second_bus_ride_time : Nat := 6

def route_b_combined_time := (second_bus_wait_time + second_bus_ride_time) / 2

def total_trip_duration : Nat := 
  first_bus_wait_time + first_bus_ride_time + first_bus_delay_time +
  walk_time_to_train + train_wait_time + train_ride_time + train_delay_time +
  route_b_combined_time

theorem jill_total_trip_duration_is_101 : total_trip_duration = 101 := by
  sorry

end jill_total_trip_duration_is_101_l1552_155221


namespace roots_quadratic_eq_sum_prod_l1552_155250

theorem roots_quadratic_eq_sum_prod (r s p q : ℝ) (hr : r + s = p) (hq : r * s = q) : r^2 + s^2 = p^2 - 2 * q :=
by
  sorry

end roots_quadratic_eq_sum_prod_l1552_155250


namespace uncovered_area_is_8_l1552_155243

-- Conditions
def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side_length : ℕ := 4

-- Theorem to prove
theorem uncovered_area_is_8
  (sh_height : ℕ := shoebox_height)
  (sh_width : ℕ := shoebox_width)
  (bl_length : ℕ := block_side_length)
  : sh_height * sh_width - bl_length * bl_length = 8 :=
by {
  -- Placeholder for proof; we are not proving it as per instructions.
  sorry
}

end uncovered_area_is_8_l1552_155243


namespace total_pages_is_1200_l1552_155222

theorem total_pages_is_1200 (A B : ℕ) (h1 : 24 * (A + B) = 60 * A) (h2 : B = A + 10) : (60 * A) = 1200 := by
  sorry

end total_pages_is_1200_l1552_155222


namespace value_of_k_l1552_155279

theorem value_of_k (k : ℝ) : 
  (∃ x y : ℝ, x = 1/3 ∧ y = -8 ∧ -3/4 - 3 * k * x = 7 * y) → k = 55.25 :=
by
  intro h
  sorry

end value_of_k_l1552_155279


namespace inequality_solution_l1552_155208

theorem inequality_solution (x : ℝ) : |x - 3| + |x - 5| ≥ 4 → x ≥ 6 ∨ x ≤ 2 :=
by
  sorry

end inequality_solution_l1552_155208


namespace age_of_b_l1552_155284

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 72) : b = 28 :=
by
  sorry

end age_of_b_l1552_155284


namespace trig_expression_value_l1552_155282

theorem trig_expression_value (α : Real) (h : Real.tan (3 * Real.pi + α) = 3) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) /
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 :=
by
  sorry

end trig_expression_value_l1552_155282


namespace M_infinite_l1552_155251

open Nat

-- Define the set M
def M : Set ℕ := {k | ∃ n : ℕ, 3 ^ n % n = k % n}

-- Statement of the problem
theorem M_infinite : Set.Infinite M :=
sorry

end M_infinite_l1552_155251


namespace number_of_articles_sold_at_cost_price_l1552_155260

-- Let C be the cost price of one article.
-- Let S be the selling price of one article.
-- Let X be the number of articles sold at cost price.

variables (C S : ℝ) (X : ℕ)

-- Condition 1: The cost price of X articles is equal to the selling price of 32 articles.
axiom condition1 : (X : ℝ) * C = 32 * S

-- Condition 2: The profit is 25%, so the selling price S is 1.25 times the cost price C.
axiom condition2 : S = 1.25 * C

-- The theorem we need to prove
theorem number_of_articles_sold_at_cost_price : X = 40 :=
by
  -- Proof here
  sorry

end number_of_articles_sold_at_cost_price_l1552_155260


namespace total_value_of_coins_is_correct_l1552_155242

-- Definitions for the problem conditions
def number_of_dimes : ℕ := 22
def number_of_quarters : ℕ := 10
def value_of_dime : ℝ := 0.10
def value_of_quarter : ℝ := 0.25
def total_value_of_dimes : ℝ := number_of_dimes * value_of_dime
def total_value_of_quarters : ℝ := number_of_quarters * value_of_quarter
def total_value : ℝ := total_value_of_dimes + total_value_of_quarters

-- Theorem statement
theorem total_value_of_coins_is_correct : total_value = 4.70 := sorry

end total_value_of_coins_is_correct_l1552_155242


namespace jelly_bean_probability_l1552_155261

variable (P_red P_orange P_yellow P_green : ℝ)

theorem jelly_bean_probability :
  P_red = 0.15 ∧ P_orange = 0.35 ∧ (P_red + P_orange + P_yellow + P_green = 1) →
  (P_yellow + P_green = 0.5) :=
by
  intro h
  obtain ⟨h_red, h_orange, h_total⟩ := h
  sorry

end jelly_bean_probability_l1552_155261


namespace tenth_term_is_98415_over_262144_l1552_155223

def first_term : ℚ := 5
def common_ratio : ℚ := 3 / 4

def tenth_term_geom_seq (a r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem tenth_term_is_98415_over_262144 :
  tenth_term_geom_seq first_term common_ratio 10 = 98415 / 262144 :=
sorry

end tenth_term_is_98415_over_262144_l1552_155223


namespace smallest_three_digit_multiple_of_17_l1552_155213

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l1552_155213


namespace probability_neither_A_nor_B_l1552_155212

noncomputable def pA : ℝ := 0.25
noncomputable def pB : ℝ := 0.35
noncomputable def pA_and_B : ℝ := 0.15

theorem probability_neither_A_nor_B :
  1 - (pA + pB - pA_and_B) = 0.55 :=
by
  simp [pA, pB, pA_and_B]
  norm_num
  sorry

end probability_neither_A_nor_B_l1552_155212


namespace find_smaller_number_l1552_155256

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : x = 21 := 
by
  sorry

end find_smaller_number_l1552_155256


namespace correct_option_division_l1552_155225

theorem correct_option_division (x : ℝ) : 
  (-6 * x^3) / (-2 * x^2) = 3 * x :=
by 
  sorry

end correct_option_division_l1552_155225


namespace greatest_possible_length_l1552_155269

-- Define the lengths of the ropes
def rope_lengths : List ℕ := [72, 48, 120, 96]

-- Define the gcd function to find the greatest common divisor of a list of numbers
def list_gcd (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- Define the target problem statement
theorem greatest_possible_length 
  (h : list_gcd rope_lengths = 24) : 
  ∀ length ∈ rope_lengths, length % 24 = 0 :=
by
  intros length h_length
  sorry

end greatest_possible_length_l1552_155269


namespace ab_plus_cd_eq_neg_346_over_9_l1552_155298

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end ab_plus_cd_eq_neg_346_over_9_l1552_155298


namespace find_rate_of_interest_l1552_155292

-- Definitions based on conditions
def Principal : ℝ := 7200
def SimpleInterest : ℝ := 3150
def Time : ℝ := 2.5
def RatePerAnnum (R : ℝ) : Prop := SimpleInterest = (Principal * R * Time) / 100

-- Theorem statement
theorem find_rate_of_interest (R : ℝ) (h : RatePerAnnum R) : R = 17.5 :=
by { sorry }

end find_rate_of_interest_l1552_155292


namespace smaller_angle_at_9_am_l1552_155258

-- Define the angular positions of the minute and hour hands
def minute_hand_angle (minute : Nat) : ℕ := 0  -- At the 12 position
def hour_hand_angle (hour : Nat) : ℕ := hour * 30  -- 30 degrees per hour

-- Define the function to get the smaller angle between two angles on the clock from 0 to 360 degrees
def smaller_angle (angle1 angle2 : ℕ) : ℕ :=
  let angle_diff := Int.natAbs (angle1 - angle2)
  min angle_diff (360 - angle_diff)

-- The theorem to prove
theorem smaller_angle_at_9_am : smaller_angle (minute_hand_angle 0) (hour_hand_angle 9) = 90 := sorry

end smaller_angle_at_9_am_l1552_155258


namespace gift_certificate_value_is_correct_l1552_155265

-- Define the conditions
def total_race_time_minutes : ℕ := 12
def one_lap_meters : ℕ := 100
def total_laps : ℕ := 24
def earning_rate_per_minute : ℕ := 7

-- The total distance run in meters
def total_distance_meters : ℕ := total_laps * one_lap_meters

-- The total earnings in dollars
def total_earnings_dollars : ℕ := earning_rate_per_minute * total_race_time_minutes

-- The worth of the gift certificate per 100 meters (to be proven as 3.50 dollars)
def gift_certificate_value : ℚ := total_earnings_dollars / (total_distance_meters / one_lap_meters)

-- Prove that the gift certificate value is $3.50
theorem gift_certificate_value_is_correct : 
    gift_certificate_value = 3.5 := by
  sorry

end gift_certificate_value_is_correct_l1552_155265
