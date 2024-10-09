import Mathlib

namespace sandy_walks_before_meet_l847_84721

/-
Sandy leaves her home and walks toward Ed's house.
Two hours later, Ed leaves his home and walks toward Sandy's house.
The distance between their homes is 52 kilometers.
Sandy's walking speed is 6 km/h.
Ed's walking speed is 4 km/h.
Prove that Sandy will walk 36 kilometers before she meets Ed.
-/

theorem sandy_walks_before_meet
    (distance_between_homes : ℕ)
    (sandy_speed ed_speed : ℕ)
    (sandy_start_time ed_start_time : ℕ)
    (time_to_meet : ℕ) :
  distance_between_homes = 52 →
  sandy_speed = 6 →
  ed_speed = 4 →
  sandy_start_time = 2 →
  ed_start_time = 0 →
  time_to_meet = 4 →
  (sandy_start_time * sandy_speed + time_to_meet * sandy_speed) = 36 := 
by
  sorry

end sandy_walks_before_meet_l847_84721


namespace range_of_y_l847_84722

noncomputable def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_y :
  (∀ x : ℝ, operation (x - y) (x + y) < 1) ↔ - (1 : ℝ) / 2 < y ∧ y < (3 : ℝ) / 2 :=
by
  sorry

end range_of_y_l847_84722


namespace min_value_of_quadratic_fun_min_value_is_reached_l847_84713

theorem min_value_of_quadratic_fun (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1) :
  (3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 ≥ (15 / 782)) :=
sorry

theorem min_value_is_reached (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1)
  (h2 : 3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 = (15 / 782)) :
  true :=
sorry

end min_value_of_quadratic_fun_min_value_is_reached_l847_84713


namespace christina_payment_l847_84701

theorem christina_payment :
  let pay_flowers_per_flower := (8 : ℚ) / 3
  let pay_lawn_per_meter := (5 : ℚ) / 2
  let num_flowers := (9 : ℚ) / 4
  let area_lawn := (7 : ℚ) / 3
  let total_payment := pay_flowers_per_flower * num_flowers + pay_lawn_per_meter * area_lawn
  total_payment = 71 / 6 :=
by
  sorry

end christina_payment_l847_84701


namespace laborer_monthly_income_l847_84799

theorem laborer_monthly_income :
  (∃ (I D : ℤ),
    6 * I + D = 540 ∧
    4 * I - D = 270) →
  (∃ I : ℤ,
    I = 81) :=
by
  sorry

end laborer_monthly_income_l847_84799


namespace total_fruits_picked_l847_84764

variable (L M P B : Nat)

theorem total_fruits_picked (hL : L = 25) (hM : M = 32) (hP : P = 12) (hB : B = 18) : L + M + P = 69 :=
by
  sorry

end total_fruits_picked_l847_84764


namespace power_identity_l847_84739

theorem power_identity (x : ℕ) (h : 2^x = 16) : 2^(x + 3) = 128 := 
sorry

end power_identity_l847_84739


namespace monica_tiles_l847_84718

-- Define the dimensions of the living room
def living_room_length : ℕ := 20
def living_room_width : ℕ := 15

-- Define the size of the border tiles and inner tiles
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Prove the number of tiles used is 44
theorem monica_tiles (border_tile_count inner_tile_count total_tiles : ℕ)
  (h_border : border_tile_count = ((2 * ((living_room_length - 4) / border_tile_size) + 2 * ((living_room_width - 4) / border_tile_size) - 4)))
  (h_inner : inner_tile_count = (176 / (inner_tile_size * inner_tile_size)))
  (h_total : total_tiles = border_tile_count + inner_tile_count) :
  total_tiles = 44 :=
by
  sorry

end monica_tiles_l847_84718


namespace find_k_l847_84719

-- Define the problem conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- Define the dot product for 2D vectors
def dot_prod (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- State the theorem
theorem find_k (k : ℝ) (h : dot_prod b (c k) = 0) : k = -3/2 :=
by
  sorry

end find_k_l847_84719


namespace sqrt_sixteen_equals_four_l847_84742

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l847_84742


namespace circle_count_2012_l847_84707

/-
The pattern is defined as follows: 
○●, ○○●, ○○○●, ○○○○●, …
We need to prove that the number of ● in the first 2012 circles is 61.
-/

-- Define the pattern sequence
def circlePattern (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Total number of circles in the first k segments:
def totalCircles (k : ℕ) : ℕ :=
  k * (k + 1) / 2 + k

theorem circle_count_2012 : 
  ∃ (n : ℕ), totalCircles n ≤ 2012 ∧ 2012 < totalCircles (n + 1) ∧ n = 61 :=
by
  sorry

end circle_count_2012_l847_84707


namespace crease_points_ellipse_l847_84756

theorem crease_points_ellipse (R a : ℝ) (x y : ℝ) (h1 : 0 < R) (h2 : 0 < a) (h3 : a < R) : 
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) ≥ 1 :=
by
  -- Omitted detailed proof steps
  sorry

end crease_points_ellipse_l847_84756


namespace total_carriages_in_towns_l847_84749

noncomputable def total_carriages (euston norfolk norwich flyingScotsman victoria waterloo : ℕ) : ℕ :=
  euston + norfolk + norwich + flyingScotsman + victoria + waterloo

theorem total_carriages_in_towns :
  let euston := 130
  let norfolk := euston - (20 * euston / 100)
  let norwich := 100
  let flyingScotsman := 3 * norwich / 2
  let victoria := euston - (15 * euston / 100)
  let waterloo := 2 * norwich
  total_carriages euston norfolk norwich flyingScotsman victoria waterloo = 794 :=
by
  sorry

end total_carriages_in_towns_l847_84749


namespace paths_inequality_l847_84783
open Nat

-- Definitions
def m : ℕ := sorry -- m represents the number of rows.
def n : ℕ := sorry -- n represents the number of columns.
def N : ℕ := sorry -- N is the number of ways to color the grid such that there is a path composed of black cells from the left edge to the right edge.
def M : ℕ := sorry -- M is the number of ways to color the grid such that there are two non-intersecting paths composed of black cells from the left edge to the right edge.

-- Theorem statement
theorem paths_inequality : (N ^ 2) ≥ 2 ^ (m * n) * M := 
by
  sorry

end paths_inequality_l847_84783


namespace star_proof_l847_84793

def star (a b : ℕ) : ℕ := 3 + b ^ a

theorem star_proof : star (star 2 1) 4 = 259 :=
by
  sorry

end star_proof_l847_84793


namespace fifth_observation_l847_84781

theorem fifth_observation (O1 O2 O3 O4 O5 O6 O7 O8 O9 : ℝ)
  (h1 : O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 = 72)
  (h2 : O1 + O2 + O3 + O4 + O5 = 50)
  (h3 : O5 + O6 + O7 + O8 + O9 = 40) :
  O5 = 18 := 
  sorry

end fifth_observation_l847_84781


namespace visitors_on_previous_day_is_246_l847_84775

def visitors_on_previous_day : Nat := 246
def total_visitors_in_25_days : Nat := 949

theorem visitors_on_previous_day_is_246 :
  visitors_on_previous_day = 246 := 
by
  rfl

end visitors_on_previous_day_is_246_l847_84775


namespace mary_biking_time_l847_84776

-- Define the conditions and the task
def total_time_away := 570 -- in minutes
def time_in_classes := 7 * 45 -- in minutes
def lunch_time := 40 -- in minutes
def additional_activities := 105 -- in minutes
def time_in_school_activities := time_in_classes + lunch_time + additional_activities

-- Define the total biking time based on given conditions
theorem mary_biking_time : 
  total_time_away - time_in_school_activities = 110 :=
by 
-- sorry is used to skip the proof step.
  sorry

end mary_biking_time_l847_84776


namespace range_of_m_l847_84720

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x - m)/2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) →
  ∃ x : ℝ, x = 2 ∧ -3 < m ∧ m ≤ -2 :=
by
  sorry

end range_of_m_l847_84720


namespace boxes_per_week_l847_84789

-- Define the given conditions
def cost_per_box : ℝ := 3.00
def weeks_in_year : ℝ := 52
def total_spent_per_year : ℝ := 312

-- The question we want to prove:
theorem boxes_per_week:
  (total_spent_per_year = cost_per_box * weeks_in_year * (total_spent_per_year / (weeks_in_year * cost_per_box))) → 
  (total_spent_per_year / (weeks_in_year * cost_per_box)) = 2 := sorry

end boxes_per_week_l847_84789


namespace solution_set_f_cos_x_l847_84727

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 3 then -(x-2)^2 + 1
else if x = 0 then 0
else if -3 < x ∧ x < 0 then (x+2)^2 - 1
else 0 -- Defined as 0 outside the given interval for simplicity

theorem solution_set_f_cos_x :
  {x : ℝ | f x * Real.cos x < 0} = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)} :=
sorry

end solution_set_f_cos_x_l847_84727


namespace pizza_problem_l847_84744

theorem pizza_problem :
  ∃ (x : ℕ), x = 20 ∧ (3 * x ^ 2 = 3 * 14 ^ 2 * 2 + 49) :=
by
  let small_pizza_side := 14
  let large_pizza_cost := 20
  let pool_cost := 60
  let individually_cost := 30
  have total_individual_area := 2 * 3 * (small_pizza_side ^ 2)
  have extra_area := 49
  sorry

end pizza_problem_l847_84744


namespace tap_fills_tank_without_leakage_in_12_hours_l847_84795

theorem tap_fills_tank_without_leakage_in_12_hours 
  (R_t R_l : ℝ)
  (h1 : (R_t - R_l) * 18 = 1)
  (h2 : R_l * 36 = 1) :
  1 / R_t = 12 := 
by
  sorry

end tap_fills_tank_without_leakage_in_12_hours_l847_84795


namespace exists_infinite_triples_a_no_triples_b_l847_84752

-- Question (a)
theorem exists_infinite_triples_a : ∀ k : ℕ, ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2 - 1) :=
by {
  sorry
}

-- Question (b)
theorem no_triples_b : ¬ ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2) :=
by {
  sorry
}

end exists_infinite_triples_a_no_triples_b_l847_84752


namespace houses_with_pools_l847_84708

theorem houses_with_pools (total G overlap N P : ℕ) 
  (h1 : total = 70) 
  (h2 : G = 50) 
  (h3 : overlap = 35) 
  (h4 : N = 15) 
  (h_eq : total = G + P - overlap + N) : 
  P = 40 := by
  sorry

end houses_with_pools_l847_84708


namespace fraction_juniors_study_Japanese_l847_84746

-- Define the size of the junior and senior classes
variable (J S : ℕ)

-- Condition 1: The senior class is twice the size of the junior class
axiom senior_twice_junior : S = 2 * J

-- The fraction of the seniors studying Japanese
noncomputable def fraction_seniors_study_Japanese : ℚ := 3 / 8

-- The total fraction of students in both classes that study Japanese
noncomputable def fraction_total_study_Japanese : ℚ := 1 / 3

-- Define the unknown fraction of juniors studying Japanese
variable (x : ℚ)

-- The proof problem transformed from the questions and the correct answer
theorem fraction_juniors_study_Japanese :
  (fraction_seniors_study_Japanese * ↑S + x * ↑J = fraction_total_study_Japanese * (↑J + ↑S)) → (x = 1 / 4) :=
by
  -- We use the given conditions and solve for x
  sorry

end fraction_juniors_study_Japanese_l847_84746


namespace find_principal_l847_84762

theorem find_principal
  (R : ℝ) (T : ℕ) (interest_less_than_principal : ℝ) : 
  R = 0.05 → 
  T = 10 → 
  interest_less_than_principal = 3100 → 
  ∃ P : ℝ, P - ((P * R * T): ℝ) = P - interest_less_than_principal ∧ P = 6200 :=
by
  sorry

end find_principal_l847_84762


namespace gcd_lcm_sum_l847_84760

theorem gcd_lcm_sum :
  gcd 42 70 + lcm 15 45 = 59 :=
by sorry

end gcd_lcm_sum_l847_84760


namespace nancy_hourly_wage_l847_84788

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l847_84788


namespace six_div_one_minus_three_div_ten_equals_twenty_four_l847_84791

theorem six_div_one_minus_three_div_ten_equals_twenty_four :
  (6 : ℤ) / (1 - (3 : ℤ) / (10 : ℤ)) = 24 := 
by
  sorry

end six_div_one_minus_three_div_ten_equals_twenty_four_l847_84791


namespace MiaShots_l847_84780

theorem MiaShots (shots_game1_to_5 : ℕ) (total_shots_game1_to_5 : ℕ) (initial_avg : ℕ → ℕ → Prop)
  (shots_game6 : ℕ) (new_avg_shots : ℕ → ℕ → Prop) (total_shots : ℕ) (new_avg : ℕ): 
  shots_game1_to_5 = 20 →
  total_shots_game1_to_5 = 50 →
  initial_avg shots_game1_to_5 total_shots_game1_to_5 →
  shots_game6 = 15 →
  new_avg_shots 29 65 →
  total_shots = total_shots_game1_to_5 + shots_game6 →
  new_avg = 45 →
  (∃ shots_made_game6 : ℕ, shots_made_game6 = 29 - shots_game1_to_5 ∧ shots_made_game6 = 9) :=
by
  sorry

end MiaShots_l847_84780


namespace least_m_plus_n_l847_84714

theorem least_m_plus_n (m n : ℕ) (hmn : Nat.gcd (m + n) 330 = 1) (hm_multiple : m^m % n^n = 0) (hm_not_multiple : ¬ (m % n = 0)) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  m + n = 119 :=
sorry

end least_m_plus_n_l847_84714


namespace ben_weekly_eggs_l847_84751

-- Definitions for the conditions
def weekly_saly_eggs : ℕ := 10
def weekly_ben_eggs (B : ℕ) : ℕ := B
def weekly_ked_eggs (B : ℕ) : ℕ := B / 2

def weekly_production (B : ℕ) : ℕ :=
  weekly_saly_eggs + weekly_ben_eggs B + weekly_ked_eggs B

def monthly_production (B : ℕ) : ℕ := 4 * weekly_production B

-- Theorem for the proof
theorem ben_weekly_eggs (B : ℕ) (h : monthly_production B = 124) : B = 14 :=
sorry

end ben_weekly_eggs_l847_84751


namespace remainder_when_dividing_n_by_d_l847_84765

def n : ℕ := 25197638
def d : ℕ := 4
def r : ℕ := 2

theorem remainder_when_dividing_n_by_d :
  n % d = r :=
by
  sorry

end remainder_when_dividing_n_by_d_l847_84765


namespace lines_parallel_if_perpendicular_to_plane_l847_84777

variables (m n l : Line) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m n : Line) : Prop := sorry

theorem lines_parallel_if_perpendicular_to_plane
  (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l847_84777


namespace sum_of_transformed_numbers_l847_84743

theorem sum_of_transformed_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := 
by
  sorry

end sum_of_transformed_numbers_l847_84743


namespace largest_base4_to_base10_l847_84703

theorem largest_base4_to_base10 : 
  (3 * 4^2 + 3 * 4^1 + 3 * 4^0) = 63 := 
by
  -- sorry to skip the proof steps
  sorry

end largest_base4_to_base10_l847_84703


namespace simplify_expression_l847_84754

theorem simplify_expression (w x : ℤ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 20 * x + 24 = 45 * w + 20 * x + 24 :=
by sorry

end simplify_expression_l847_84754


namespace second_ball_red_probability_l847_84768

-- Definitions based on given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4
def first_ball_is_red : Prop := true

-- The probability that the second ball drawn is red given the first ball drawn is red
def prob_second_red_given_first_red : ℚ :=
  (red_balls - 1) / (total_balls - 1)

theorem second_ball_red_probability :
  first_ball_is_red → prob_second_red_given_first_red = 5 / 9 :=
by
  intro _
  -- proof goes here
  sorry

end second_ball_red_probability_l847_84768


namespace proof_problem_l847_84766

noncomputable def p : Prop := ∃ x : ℝ, Real.sin x > 1
noncomputable def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

theorem proof_problem : ¬ (p ∨ q) :=
by sorry

end proof_problem_l847_84766


namespace value_of_x_plus_y_l847_84785

-- Define the sum of integers from 50 to 60
def sum_integers_50_to_60 : ℤ := List.sum (List.range' 50 (60 - 50 + 1))

-- Calculate the number of even integers from 50 to 60
def count_even_integers_50_to_60 : ℤ := List.length (List.filter (λ n => n % 2 = 0) (List.range' 50 (60 - 50 + 1)))

-- Define x and y based on the given conditions
def x : ℤ := sum_integers_50_to_60
def y : ℤ := count_even_integers_50_to_60

-- The main theorem to prove
theorem value_of_x_plus_y : x + y = 611 := by
  -- Placeholder for the proof
  sorry

end value_of_x_plus_y_l847_84785


namespace cubes_sum_eq_zero_l847_84747

theorem cubes_sum_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 :=
by
  sorry

end cubes_sum_eq_zero_l847_84747


namespace base_8_not_divisible_by_five_l847_84773

def base_b_subtraction_not_divisible_by_five (b : ℕ) : Prop :=
  let num1 := 3 * b^3 + 1 * b^2 + 0 * b + 2
  let num2 := 3 * b^2 + 0 * b + 2
  let diff := num1 - num2
  ¬ (diff % 5 = 0)

theorem base_8_not_divisible_by_five : base_b_subtraction_not_divisible_by_five 8 := 
by
  sorry

end base_8_not_divisible_by_five_l847_84773


namespace van_distance_l847_84778

theorem van_distance (D : ℝ) (t_initial t_new : ℝ) (speed_new : ℝ) 
  (h1 : t_initial = 6) 
  (h2 : t_new = (3 / 2) * t_initial) 
  (h3 : speed_new = 30) 
  (h4 : D = speed_new * t_new) : 
  D = 270 :=
by
  sorry

end van_distance_l847_84778


namespace sum_of_coefficients_is_7_l847_84755

noncomputable def v (n : ℕ) : ℕ := sorry

theorem sum_of_coefficients_is_7 : 
  (∀ n : ℕ, v (n + 1) - v n = 3 * n + 2) → (v 1 = 7) → (∃ a b c : ℝ, (a * n^2 + b * n + c = v n) ∧ (a + b + c = 7)) := 
by
  intros H1 H2
  sorry

end sum_of_coefficients_is_7_l847_84755


namespace player2_winning_strategy_l847_84702

-- Definitions of the game setup
def initial_position_player1 := (1, 1)
def initial_position_player2 := (998, 1998)

def adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 - 1 ∨ p1.2 = p2.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 - 1 ∨ p1.1 = p2.1 + 1))

-- A function defining the winning condition for Player 2
def player2_wins (p1 p2 : ℕ × ℕ) : Prop :=
  p1 = p2 ∨ p1.1 = (initial_position_player2.1)

-- Theorem stating the pair (998, 1998) guarantees a win for Player 2
theorem player2_winning_strategy : player2_wins (998, 0) (998, 1998) :=
sorry

end player2_winning_strategy_l847_84702


namespace correct_average_marks_l847_84798

-- Define all the given conditions
def average_marks : ℕ := 92
def number_of_students : ℕ := 25
def wrong_mark : ℕ := 75
def correct_mark : ℕ := 30

-- Define variables for total marks calculations
def total_marks_with_wrong : ℕ := average_marks * number_of_students
def total_marks_with_correct : ℕ := total_marks_with_wrong - wrong_mark + correct_mark

-- Goal: Prove that the correct average marks is 90.2
theorem correct_average_marks :
  (total_marks_with_correct : ℝ) / (number_of_students : ℝ) = 90.2 :=
by
  sorry

end correct_average_marks_l847_84798


namespace triangle_area_ratio_l847_84706

theorem triangle_area_ratio (a n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  let area_A := (a^2 : ℝ) / (4 * n^2)
  let area_B := (a^2 : ℝ) / (4 * m^2)
  (area_A / area_B) = (m^2 : ℝ) / (n^2 : ℝ) :=
by
  sorry

end triangle_area_ratio_l847_84706


namespace division_result_l847_84772

def m : ℕ := 16 ^ 2024

theorem division_result : m / 8 = 8 * 16 ^ 2020 :=
by
  -- sorry for the actual proof
  sorry

end division_result_l847_84772


namespace trey_total_time_is_two_hours_l847_84732

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end trey_total_time_is_two_hours_l847_84732


namespace range_f_l847_84730

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4 

theorem range_f : Set.Icc (0 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_f_l847_84730


namespace inequality_abs_l847_84712

noncomputable def f (x : ℝ) : ℝ := abs (x - 1/2) + abs (x + 1/2)

def M : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem inequality_abs (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + a * b| := 
by
  sorry

end inequality_abs_l847_84712


namespace probability_of_sequence_l847_84759

theorem probability_of_sequence :
  let total_cards := 52
  let face_cards := 12
  let hearts := 13
  let first_card_face_prob := (face_cards : ℝ) / total_cards
  let second_card_heart_prob := (10 : ℝ) / (total_cards - 1)
  let third_card_face_prob := (11 : ℝ) / (total_cards - 2)
  let total_prob := first_card_face_prob * second_card_heart_prob * third_card_face_prob
  total_prob = 1 / 100.455 :=
by
  sorry

end probability_of_sequence_l847_84759


namespace find_principal_amount_l847_84735

-- Given conditions
def SI : ℝ := 4016.25
def R : ℝ := 0.14
def T : ℕ := 5

-- Question: What is the principal amount P?
theorem find_principal_amount : (SI / (R * T) = 5737.5) :=
sorry

end find_principal_amount_l847_84735


namespace original_days_to_finish_work_l847_84734

theorem original_days_to_finish_work : 
  ∀ (D : ℕ), 
  (∃ (W : ℕ), 15 * D * W = 25 * (D - 3) * W) → 
  D = 8 :=
by
  intros D h
  sorry

end original_days_to_finish_work_l847_84734


namespace mean_first_second_fifth_sixth_diff_l847_84757

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end mean_first_second_fifth_sixth_diff_l847_84757


namespace exists_colored_subset_l847_84711

theorem exists_colored_subset (n : ℕ) (h_positive : n > 0) (colors : ℕ → ℕ) (h_colors : ∀ a b : ℕ, a < b → a + b ≤ n → 
  (colors a = colors b ∨ colors b = colors (a + b) ∨ colors a = colors (a + b))) :
  ∃ c, ∃ s : Finset ℕ, s.card ≥ (2 * n / 5) ∧ ∀ x ∈ s, colors x = c :=
sorry

end exists_colored_subset_l847_84711


namespace fraction_computation_l847_84741

noncomputable def compute_fraction : ℚ :=
  (64^4 + 324) * (52^4 + 324) * (40^4 + 324) * (28^4 + 324) * (16^4 + 324) /
  (58^4 + 324) * (46^4 + 324) * (34^4 + 324) * (22^4 + 324) * (10^4 + 324)

theorem fraction_computation :
  compute_fraction = 137 / 1513 :=
by sorry

end fraction_computation_l847_84741


namespace michael_twice_jacob_in_11_years_l847_84774

-- Definitions
def jacob_age_4_years := 5
def jacob_current_age := jacob_age_4_years - 4
def michael_current_age := jacob_current_age + 12

-- Theorem to prove
theorem michael_twice_jacob_in_11_years :
  ∀ (x : ℕ), jacob_current_age + x = 1 →
    michael_current_age + x = 13 →
    michael_current_age + (11 : ℕ) = 2 * (jacob_current_age + (11 : ℕ)) :=
by
  intros x h1 h2
  sorry

end michael_twice_jacob_in_11_years_l847_84774


namespace total_payment_l847_84796

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l847_84796


namespace cuboid_length_l847_84748

theorem cuboid_length (SA w h : ℕ) (h_SA : SA = 700) (h_w : w = 14) (h_h : h = 7) 
  (h_surface_area : SA = 2 * l * w + 2 * l * h + 2 * w * h) : l = 12 :=
by
  intros
  sorry

end cuboid_length_l847_84748


namespace hours_sunday_correct_l847_84717

-- Definitions of given conditions
def hours_saturday : ℕ := 6
def total_hours : ℕ := 9

-- The question translated to a proof problem
theorem hours_sunday_correct : total_hours - hours_saturday = 3 := 
by
  -- The proof is skipped and replaced by sorry
  sorry

end hours_sunday_correct_l847_84717


namespace max_a_monotonic_f_l847_84726

theorem max_a_monotonic_f {a : ℝ} (h1 : 0 < a)
  (h2 : ∀ x ≥ 1, 0 ≤ (3 * x^2 - a)) : a ≤ 3 := by
  -- Proof to be provided
  sorry

end max_a_monotonic_f_l847_84726


namespace electronics_sale_negation_l847_84769

variables (E : Type) (storeElectronics : E → Prop) (onSale : E → Prop)

theorem electronics_sale_negation
  (H : ¬ ∀ e, storeElectronics e → onSale e) :
  (∃ e, storeElectronics e ∧ ¬ onSale e) ∧ ¬ ∀ e, storeElectronics e → onSale e :=
by
  -- Proving that at least one electronic is not on sale follows directly from the negation of the universal statement
  sorry

end electronics_sale_negation_l847_84769


namespace triangle_side_a_value_l847_84792

noncomputable def a_value (A B c : ℝ) : ℝ :=
  30 * Real.sqrt 2 - 10 * Real.sqrt 6

theorem triangle_side_a_value
  (A B : ℝ) (c : ℝ)
  (hA : A = 60)
  (hB : B = 45)
  (hc : c = 20) :
  a_value A B c = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by
  sorry

end triangle_side_a_value_l847_84792


namespace school_selection_theorem_l847_84779

-- Define the basic setup and conditions
def school_selection_problem : Prop :=
  let schools := ["A", "B", "C", "D"]
  let total_schools := 4
  let selected_schools := 2
  let combinations := Nat.choose total_schools selected_schools
  let favorable_outcomes := Nat.choose (total_schools - 1) (selected_schools - 1)
  let probability := (favorable_outcomes : ℚ) / (combinations : ℚ)
  probability = 1 / 2

-- Proof is yet to be provided
theorem school_selection_theorem : school_selection_problem := sorry

end school_selection_theorem_l847_84779


namespace minimum_w_value_l847_84715

theorem minimum_w_value : 
  (∀ x y : ℝ, w = 2*x^2 + 3*y^2 - 12*x + 9*y + 35) → 
  ∃ w_min : ℝ, w_min = 41 / 4 ∧ 
  (∀ x y : ℝ, 2*x^2 + 3*y^2 - 12*x + 9*y + 35 ≥ w_min) :=
by
  sorry

end minimum_w_value_l847_84715


namespace common_sum_l847_84736

theorem common_sum (a l : ℤ) (n r c : ℕ) (S x : ℤ) 
  (h_a : a = -18) 
  (h_l : l = 30) 
  (h_n : n = 49) 
  (h_S : S = (n * (a + l)) / 2) 
  (h_r : r = 7) 
  (h_c : c = 7) 
  (h_sum_eq : r * x = S) :
  x = 42 := 
sorry

end common_sum_l847_84736


namespace ant_climbing_floors_l847_84705

theorem ant_climbing_floors (time_per_floor : ℕ) (total_time : ℕ) (floors_climbed : ℕ) :
  time_per_floor = 15 →
  total_time = 105 →
  floors_climbed = total_time / time_per_floor + 1 →
  floors_climbed = 8 :=
by
  intros
  sorry

end ant_climbing_floors_l847_84705


namespace product_floor_ceil_sequence_l847_84723

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end product_floor_ceil_sequence_l847_84723


namespace graph_forms_l847_84740

theorem graph_forms (x y : ℝ) :
  x^3 * (2 * x + 2 * y + 3) = y^3 * (2 * x + 2 * y + 3) →
  (∀ x y : ℝ, y ≠ x → y = -x - 3 / 2) ∨ (y = x) :=
sorry

end graph_forms_l847_84740


namespace workshop_participants_problem_l847_84737

variable (WorkshopSize : ℕ) 
variable (LeftHanded : ℕ) 
variable (RockMusicLovers : ℕ) 
variable (RightHandedDislikeRock : ℕ) 
variable (Under25 : ℕ)
variable (RightHandedUnder25RockMusicLovers : ℕ)
variable (y : ℕ)

theorem workshop_participants_problem
  (h1 : WorkshopSize = 30)
  (h2 : LeftHanded = 12)
  (h3 : RockMusicLovers = 18)
  (h4 : RightHandedDislikeRock = 5)
  (h5 : Under25 = 9)
  (h6 : RightHandedUnder25RockMusicLovers = 3)
  (h7 : WorkshopSize = LeftHanded + (WorkshopSize - LeftHanded))
  (h8 : WorkshopSize - LeftHanded = RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + (WorkshopSize - LeftHanded - RightHandedDislikeRock - RightHandedUnder25RockMusicLovers - y))
  (h9 : WorkshopSize - (RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + Under25 - y - (RockMusicLovers - y)) - (LeftHanded - y) = WorkshopSize) :
  y = 5 := by
  sorry

end workshop_participants_problem_l847_84737


namespace single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l847_84750

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l847_84750


namespace largest_integer_divisor_l847_84729

theorem largest_integer_divisor (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end largest_integer_divisor_l847_84729


namespace original_cost_l847_84784

theorem original_cost (A : ℝ) (discount : ℝ) (sale_price : ℝ) (original_price : ℝ) (h1 : discount = 0.30) (h2 : sale_price = 35) (h3 : sale_price = (1 - discount) * original_price) : 
  original_price = 50 := by
  sorry

end original_cost_l847_84784


namespace no_member_of_T_is_divisible_by_4_l847_84709

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4 : ∀ n : ℤ, ¬ (sum_of_squares_of_four_consecutive_integers n % 4 = 0) := by
  intro n
  sorry

end no_member_of_T_is_divisible_by_4_l847_84709


namespace commercials_per_hour_l847_84725

theorem commercials_per_hour (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : ∃ x : ℝ, x = (1 - p) * 60 := 
sorry

end commercials_per_hour_l847_84725


namespace percentage_of_first_to_second_l847_84761

theorem percentage_of_first_to_second (X : ℝ) (h1 : first = (7/100) * X) (h2 : second = (14/100) * X) : (first / second) * 100 = 50 := 
by
  sorry

end percentage_of_first_to_second_l847_84761


namespace num_five_ruble_coins_l847_84753

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l847_84753


namespace liam_total_money_l847_84786

-- Define the conditions as noncomputable since they involve monetary calculations
noncomputable def liam_money (initial_bottles : ℕ) (price_per_bottle : ℕ) (bottles_sold : ℕ) (extra_money : ℕ) : ℚ :=
  let cost := initial_bottles * price_per_bottle
  let money_after_selling_part := cost + extra_money
  let selling_price_per_bottle := money_after_selling_part / bottles_sold
  let total_revenue := initial_bottles * selling_price_per_bottle
  total_revenue

-- State the theorem with the given problem
theorem liam_total_money :
  let initial_bottles := 50
  let price_per_bottle := 1
  let bottles_sold := 40
  let extra_money := 10
  liam_money initial_bottles price_per_bottle bottles_sold extra_money = 75 := 
sorry

end liam_total_money_l847_84786


namespace plan_A_fee_eq_nine_l847_84770

theorem plan_A_fee_eq_nine :
  ∃ F : ℝ, (0.25 * 60 + F = 0.40 * 60) ∧ (F = 9) :=
by
  sorry

end plan_A_fee_eq_nine_l847_84770


namespace min_value_ineq_l847_84767

noncomputable def function_y (a : ℝ) (x : ℝ) : ℝ := a^(1-x)

theorem min_value_ineq (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_ineq_l847_84767


namespace largest_divisor_expression_l847_84794

theorem largest_divisor_expression (y : ℤ) (h : y % 2 = 1) : 
  4320 ∣ (15 * y + 3) * (15 * y + 9) * (10 * y + 10) :=
sorry  

end largest_divisor_expression_l847_84794


namespace system_of_equations_solution_l847_84745

theorem system_of_equations_solution (x y : ℚ) :
  (x / 3 + y / 4 = 4 ∧ 2 * x - 3 * y = 12) → (x = 10 ∧ y = 8 / 3) :=
by
  sorry

end system_of_equations_solution_l847_84745


namespace impossible_sum_of_two_smaller_angles_l847_84716

theorem impossible_sum_of_two_smaller_angles
  {α β γ : ℝ}
  (h1 : α + β + γ = 180)
  (h2 : 0 < α + β ∧ α + β < 180) :
  α + β ≠ 130 :=
sorry

end impossible_sum_of_two_smaller_angles_l847_84716


namespace range_of_a_l847_84771

noncomputable def inequality_always_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (a : ℝ) : inequality_always_holds a ↔ 0 ≤ a ∧ a < 1 := 
by
  sorry

end range_of_a_l847_84771


namespace jane_exercises_40_hours_l847_84731

-- Define the conditions
def hours_per_day : ℝ := 1
def days_per_week : ℝ := 5
def weeks : ℝ := 8

-- Define total_hours using the conditions
def total_hours : ℝ := (hours_per_day * days_per_week) * weeks

-- The theorem stating the result
theorem jane_exercises_40_hours :
  total_hours = 40 := by
  sorry

end jane_exercises_40_hours_l847_84731


namespace convert_base7_to_base2_l847_84790

-- Definitions and conditions
def base7_to_decimal (n : ℕ) : ℕ :=
  2 * 7^1 + 5 * 7^0

def decimal_to_binary (n : ℕ) : ℕ :=
  -- Reversing the binary conversion steps
  -- 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 19
  1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Proof problem
theorem convert_base7_to_base2 : decimal_to_binary (base7_to_decimal 25) = 10011 :=
by {
  sorry
}

end convert_base7_to_base2_l847_84790


namespace neg_P_4_of_P_implication_and_neg_P_5_l847_84710

variable (P : ℕ → Prop)

theorem neg_P_4_of_P_implication_and_neg_P_5
  (h1 : ∀ k : ℕ, 0 < k → (P k → P (k+1)))
  (h2 : ¬ P 5) :
  ¬ P 4 :=
by
  sorry

end neg_P_4_of_P_implication_and_neg_P_5_l847_84710


namespace merchant_articles_l847_84782

theorem merchant_articles 
   (CP SP : ℝ)
   (N : ℝ)
   (h1 : SP = 1.25 * CP)
   (h2 : N * CP = 16 * SP) : 
   N = 20 := by
   sorry

end merchant_articles_l847_84782


namespace problem_l847_84797

theorem problem (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a + b + c)) : (a + b) * (b + c) * (a + c) = 0 := 
by
  sorry

end problem_l847_84797


namespace sodas_purchasable_l847_84704

namespace SodaPurchase

variable {D C : ℕ}

theorem sodas_purchasable (D C : ℕ) : (3 * (4 * D) / 5 + 5 * C / 15) = (36 * D + 5 * C) / 15 := 
  sorry

end SodaPurchase

end sodas_purchasable_l847_84704


namespace slices_with_both_toppings_l847_84728

theorem slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices : ℕ)
    (all_have_topping : total_slices = 24)
    (pepperoni_cond: pepperoni_slices = 14)
    (mushroom_cond: mushroom_slices = 16)
    (at_least_one_topping : total_slices = pepperoni_slices + mushroom_slices - slices_with_both):
    slices_with_both = 6 := by
  sorry

end slices_with_both_toppings_l847_84728


namespace geo_seq_decreasing_l847_84787

variables (a_1 q : ℝ) (a : ℕ → ℝ)
-- Define the geometric sequence
def geo_seq (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q ^ n

-- The problem statement as a Lean theorem
theorem geo_seq_decreasing (h1 : a_1 * (q - 1) < 0) (h2 : q > 0) :
  ∀ n : ℕ, geo_seq a_1 q (n + 1) < geo_seq a_1 q n :=
by
  sorry

end geo_seq_decreasing_l847_84787


namespace tank_capacity_l847_84758

theorem tank_capacity (x : ℝ) (h₁ : 0.25 * x = 60) (h₂ : 0.05 * x = 12) : x = 240 :=
sorry

end tank_capacity_l847_84758


namespace min_value_condition_l847_84700

theorem min_value_condition {a b c d e f g h : ℝ} (h1 : a * b * c * d = 16) (h2 : e * f * g * h = 25) :
  (a^2 * e^2 + b^2 * f^2 + c^2 * g^2 + d^2 * h^2) ≥ 160 :=
  sorry

end min_value_condition_l847_84700


namespace chad_bbq_people_l847_84724

theorem chad_bbq_people (ice_cost_per_pack : ℝ) (packs_included : ℕ) (total_money_spent : ℝ) (pounds_needed_per_person : ℝ) :
  total_money_spent = 9 → 
  ice_cost_per_pack = 3 → 
  packs_included = 10 → 
  pounds_needed_per_person = 2 → 
  ∃ (people : ℕ), people = 15 :=
by intros; sorry

end chad_bbq_people_l847_84724


namespace license_plates_count_l847_84733

def num_consonants : Nat := 20
def num_vowels : Nat := 6
def num_digits : Nat := 10
def num_symbols : Nat := 3

theorem license_plates_count : 
  num_consonants * num_vowels * num_consonants * num_digits * num_symbols = 72000 :=
by 
  sorry

end license_plates_count_l847_84733


namespace number_of_distinct_intersections_of_curves_l847_84738

theorem number_of_distinct_intersections_of_curves (x y : ℝ) :
  (∀ x y, x^2 - 4*y^2 = 4) ∧ (∀ x y, 4*x^2 + y^2 = 16) → 
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), 
    ((x1, y1) ≠ (x2, y2)) ∧
    ((x1^2 - 4*y1^2 = 4) ∧ (4*x1^2 + y1^2 = 16)) ∧
    ((x2^2 - 4*y2^2 = 4) ∧ (4*x2^2 + y2^2 = 16)) ∧
    ∀ (x' y' : ℝ), 
      ((x'^2 - 4*y'^2 = 4) ∧ (4*x'^2 + y'^2 = 16)) → 
      ((x', y') = (x1, y1) ∨ (x', y') = (x2, y2)) := 
sorry

end number_of_distinct_intersections_of_curves_l847_84738


namespace sequence_u5_eq_27_l847_84763

theorem sequence_u5_eq_27 (u : ℕ → ℝ) 
  (h_recurrence : ∀ n, u (n + 2) = 3 * u (n + 1) - 2 * u n)
  (h_u3 : u 3 = 15)
  (h_u6 : u 6 = 43) :
  u 5 = 27 :=
  sorry

end sequence_u5_eq_27_l847_84763
