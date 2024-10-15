import Mathlib

namespace NUMINAMATH_GPT_least_addition_l344_34432

theorem least_addition (a b n : ℕ) (h_a : Nat.Prime a) (h_b : Nat.Prime b) (h_a_val : a = 23) (h_b_val : b = 29) (h_n : n = 1056) :
  ∃ m : ℕ, (m + n) % (a * b) = 0 ∧ m = 278 :=
by
  sorry

end NUMINAMATH_GPT_least_addition_l344_34432


namespace NUMINAMATH_GPT_final_result_is_102_l344_34487

-- Definitions and conditions from the problem
def chosen_number : ℕ := 120
def multiplied_result : ℕ := 2 * chosen_number
def final_result : ℕ := multiplied_result - 138

-- The proof statement
theorem final_result_is_102 : final_result = 102 := 
by 
sorry

end NUMINAMATH_GPT_final_result_is_102_l344_34487


namespace NUMINAMATH_GPT_men_left_hostel_l344_34482

-- Definitions based on the conditions given
def initialMen : ℕ := 250
def initialDays : ℕ := 28
def remainingDays : ℕ := 35

-- The theorem we need to prove
theorem men_left_hostel (x : ℕ) (h : initialMen * initialDays = (initialMen - x) * remainingDays) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_men_left_hostel_l344_34482


namespace NUMINAMATH_GPT_arithmetic_sequence_k_value_l344_34484

theorem arithmetic_sequence_k_value (a1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a1 = 1)
  (h2 : d = 2)
  (h3 : ∀ k : ℕ, S (k+2) - S k = 24) : k = 5 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_value_l344_34484


namespace NUMINAMATH_GPT_S_gt_inverse_1988_cubed_l344_34489

theorem S_gt_inverse_1988_cubed (a b c d : ℕ) (hb: 0 < b) (hd: 0 < d) 
  (h1: a + c < 1988) (h2: 1 - (a / b) - (c / d) > 0) : 
  1 - (a / b) - (c / d) > 1 / (1988^3) := 
sorry

end NUMINAMATH_GPT_S_gt_inverse_1988_cubed_l344_34489


namespace NUMINAMATH_GPT_net_cut_square_l344_34449

-- Define the dimensions of the parallelepiped
structure Parallelepiped :=
  (length width height : ℕ)
  (length_eq : length = 2)
  (width_eq : width = 1)
  (height_eq : height = 1)

-- Define the net of the parallelepiped
structure NetConfig :=
  (total_squares : ℕ)
  (cut_squares : ℕ)
  (remaining_squares : ℕ)
  (cut_positions : Fin 5) -- Five possible cut positions

-- The remaining net has 9 squares after cutting one square
theorem net_cut_square (p : Parallelepiped) : 
  ∃ net : NetConfig, net.total_squares = 10 ∧ net.cut_squares = 1 ∧ net.remaining_squares = 9 ∧ net.cut_positions = 5 := 
sorry

end NUMINAMATH_GPT_net_cut_square_l344_34449


namespace NUMINAMATH_GPT_find_m_value_l344_34425

theorem find_m_value 
  (h : ∀ x y m : ℝ, 2*x + y + m = 0 → (1 : ℝ)*x + (-2 : ℝ)*y + 0 = 0)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ m : ℝ, m = 0 :=
sorry

end NUMINAMATH_GPT_find_m_value_l344_34425


namespace NUMINAMATH_GPT_parallel_lines_determine_plane_l344_34474

def determine_plane_by_parallel_lines := 
  let condition_4 := true -- Two parallel lines
  condition_4 = true

theorem parallel_lines_determine_plane : determine_plane_by_parallel_lines = true :=
by 
  sorry

end NUMINAMATH_GPT_parallel_lines_determine_plane_l344_34474


namespace NUMINAMATH_GPT_ava_first_coupon_day_l344_34478

theorem ava_first_coupon_day (first_coupon_day : ℕ) (coupon_interval : ℕ) 
    (closed_day : ℕ) (days_in_week : ℕ):
  first_coupon_day = 2 →  -- starting on Tuesday (considering Monday as 1)
  coupon_interval = 13 →
  closed_day = 7 →        -- Saturday is represented by 7
  days_in_week = 7 →
  ∀ n : ℕ, ((first_coupon_day + n * coupon_interval) % days_in_week) ≠ closed_day :=
by 
  -- Proof can be filled here.
  sorry

end NUMINAMATH_GPT_ava_first_coupon_day_l344_34478


namespace NUMINAMATH_GPT_number_of_knights_l344_34417

/--
On the island of Liars and Knights, a circular arrangement is called correct if everyone standing in the circle
can say that among his two neighbors there is a representative of his tribe. One day, 2019 natives formed a correct
arrangement in a circle. A liar approached them and said: "Now together we can also form a correct arrangement in a circle."
Prove that the number of knights in the initial arrangement is 1346.
-/
theorem number_of_knights : 
  ∀ (K L : ℕ), 
    (K + L = 2019) → 
    (K ≥ 2 * L) → 
    (K ≤ 2 * L + 1) → 
  K = 1346 :=
by
  intros K L h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_knights_l344_34417


namespace NUMINAMATH_GPT_min_colors_shapes_l344_34469

def representable_centers (C S : Nat) : Nat :=
  C + (C * (C - 1)) / 2 + S + S * (S - 1)

theorem min_colors_shapes (C S : Nat) :
  ∀ (C S : Nat), (C + (C * (C - 1)) / 2 + S + S * (S - 1)) ≥ 12 → (C, S) = (3, 3) :=
sorry

end NUMINAMATH_GPT_min_colors_shapes_l344_34469


namespace NUMINAMATH_GPT_bird_count_l344_34438

theorem bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : parakeets_per_cage = 7) 
  (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) : 
  total_birds = 72 := 
  by
  sorry

end NUMINAMATH_GPT_bird_count_l344_34438


namespace NUMINAMATH_GPT_distance_travelled_l344_34445

def speed : ℕ := 3 -- speed in feet per second
def time : ℕ := 3600 -- time in seconds (1 hour)

theorem distance_travelled : speed * time = 10800 := by
  sorry

end NUMINAMATH_GPT_distance_travelled_l344_34445


namespace NUMINAMATH_GPT_average_of_six_numbers_l344_34405

theorem average_of_six_numbers (a b c d e f : ℝ)
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 :=
by sorry

end NUMINAMATH_GPT_average_of_six_numbers_l344_34405


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l344_34494

theorem geometric_sequence_first_term (a1 q : ℝ) 
  (h1 : (a1 * (1 - q^4)) / (1 - q) = 240)
  (h2 : a1 * q + a1 * q^3 = 180) : 
  a1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l344_34494


namespace NUMINAMATH_GPT_total_weight_of_rings_l344_34477

-- Define the weights of the rings
def weight_orange : Real := 0.08
def weight_purple : Real := 0.33
def weight_white : Real := 0.42
def weight_blue : Real := 0.59
def weight_red : Real := 0.24
def weight_green : Real := 0.16

-- Define the total weight of the rings
def total_weight : Real :=
  weight_orange + weight_purple + weight_white + weight_blue + weight_red + weight_green

-- The task is to prove that the total weight equals 1.82
theorem total_weight_of_rings : total_weight = 1.82 := 
  by
    sorry

end NUMINAMATH_GPT_total_weight_of_rings_l344_34477


namespace NUMINAMATH_GPT_tractor_planting_rate_l344_34490

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tractor_planting_rate_l344_34490


namespace NUMINAMATH_GPT_triangle_problems_l344_34414

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem triangle_problems
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13)
  (h3 : b + c = 5) :
  (A = π / 3) ∧ (S = Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_triangle_problems_l344_34414


namespace NUMINAMATH_GPT_find_y_when_x_is_4_l344_34452

def inverse_proportional (x y : ℝ) : Prop :=
  ∃ C : ℝ, x * y = C

theorem find_y_when_x_is_4 :
  ∀ x y : ℝ,
  inverse_proportional x y →
  (x + y = 20) →
  (x - y = 4) →
  (∃ y, y = 24 ∧ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_4_l344_34452


namespace NUMINAMATH_GPT_unique_solution_exists_l344_34470

def f (x y z : ℕ) : ℕ := (x + y - 2) * (x + y - 1) / 2 - z

theorem unique_solution_exists :
  ∀ (a b c d : ℕ), f a b c = 1993 ∧ f c d a = 1993 → (a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42) :=
by
  intros a b c d h
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l344_34470


namespace NUMINAMATH_GPT_ab_diff_2023_l344_34458

theorem ab_diff_2023 (a b : ℝ) 
  (h : a^2 + b^2 - 4 * a - 6 * b + 13 = 0) : (a - b) ^ 2023 = -1 :=
sorry

end NUMINAMATH_GPT_ab_diff_2023_l344_34458


namespace NUMINAMATH_GPT_parallelogram_angle_l344_34462

theorem parallelogram_angle (a b : ℕ) (h : a + b = 180) (exceed_by_10 : b = a + 10) : a = 85 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_parallelogram_angle_l344_34462


namespace NUMINAMATH_GPT_brochures_multiple_of_6_l344_34488

theorem brochures_multiple_of_6 (n : ℕ) (P : ℕ) (B : ℕ) 
  (hP : P = 12) (hn : n = 6) : ∃ k : ℕ, B = 6 * k := 
sorry

end NUMINAMATH_GPT_brochures_multiple_of_6_l344_34488


namespace NUMINAMATH_GPT_length_of_AB_l344_34429

theorem length_of_AB
  (AP PB AQ QB : ℝ) 
  (h_ratioP : 5 * AP = 3 * PB)
  (h_ratioQ : 3 * AQ = 2 * QB)
  (h_PQ : AQ = AP + 3 ∧ QB = PB - 3)
  (h_PQ_length : AQ - AP = 3)
  : AP + PB = 120 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_AB_l344_34429


namespace NUMINAMATH_GPT_range_of_k_l344_34410

theorem range_of_k (k : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (k + 2) * x1 - 1 > (k + 2) * x2 - 1) → k < -2 := by
  sorry

end NUMINAMATH_GPT_range_of_k_l344_34410


namespace NUMINAMATH_GPT_simplify_expression_l344_34473

theorem simplify_expression : 
  (Real.sqrt 2 * 2^(1/2) * 2) + (18 / 3 * 2) - (8^(1/2) * 4) = 16 - 8 * Real.sqrt 2 :=
by 
  sorry  -- proof omitted

end NUMINAMATH_GPT_simplify_expression_l344_34473


namespace NUMINAMATH_GPT_number_of_adults_l344_34415

theorem number_of_adults
  (A C : ℕ)
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) :
  A = 350 :=
by
  sorry

end NUMINAMATH_GPT_number_of_adults_l344_34415


namespace NUMINAMATH_GPT_slower_train_speed_l344_34431

theorem slower_train_speed
  (v : ℝ)  -- The speed of the slower train
  (faster_train_speed : ℝ := 46)  -- The speed of the faster train
  (train_length : ℝ := 37.5)  -- The length of each train in meters
  (time_to_pass : ℝ := 27)  -- Time taken to pass in seconds
  (kms_to_ms : ℝ := 1000 / 3600)  -- Conversion factor from km/hr to m/s
  (relative_distance : ℝ := 2 * train_length)  -- Distance covered when passing

  (h : relative_distance = (faster_train_speed - v) * kms_to_ms * time_to_pass) :
  v = 36 :=
by
  -- The proof should be placed here
  sorry

end NUMINAMATH_GPT_slower_train_speed_l344_34431


namespace NUMINAMATH_GPT_cube_sum_eq_2702_l344_34441

noncomputable def x : ℝ := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
noncomputable def y : ℝ := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)

theorem cube_sum_eq_2702 : x^3 + y^3 = 2702 :=
by
  sorry

end NUMINAMATH_GPT_cube_sum_eq_2702_l344_34441


namespace NUMINAMATH_GPT_third_red_yellow_flash_is_60_l344_34430

-- Define the flashing intervals for red, yellow, and green lights
def red_interval : Nat := 3
def yellow_interval : Nat := 4
def green_interval : Nat := 8

-- Define the function for finding the time of the third occurrence of only red and yellow lights flashing together
def third_red_yellow_flash : Nat :=
  let lcm_red_yellow := Nat.lcm red_interval yellow_interval
  let times := (List.range (100)).filter (fun t => t % lcm_red_yellow = 0 ∧ t % green_interval ≠ 0)
  times[2] -- Getting the third occurrence

-- Prove that the third occurrence time is 60 seconds
theorem third_red_yellow_flash_is_60 :
  third_red_yellow_flash = 60 :=
  by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_third_red_yellow_flash_is_60_l344_34430


namespace NUMINAMATH_GPT_complement_supplement_measure_l344_34467

theorem complement_supplement_measure (x : ℝ) (h : 180 - x = 3 * (90 - x)) : 
  (180 - x = 135) ∧ (90 - x = 45) :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_supplement_measure_l344_34467


namespace NUMINAMATH_GPT_probability_of_three_faces_painted_l344_34451

def total_cubes : Nat := 27
def corner_cubes_painted (total : Nat) : Nat := 8
def probability_of_corner_cube (corner : Nat) (total : Nat) : Rat := corner / total

theorem probability_of_three_faces_painted :
    probability_of_corner_cube (corner_cubes_painted total_cubes) total_cubes = 8 / 27 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_three_faces_painted_l344_34451


namespace NUMINAMATH_GPT_area_of_triangle_l344_34471

-- Definitions
variables {A B C : Type}
variables {i j k : ℕ}
variables (AB AC : ℝ)
variables (s t : ℝ)
variables (sinA : ℝ) (cosA : ℝ)

-- Conditions 
axiom sin_A : sinA = 4 / 5
axiom dot_product : s * t * cosA = 6

-- The problem theorem
theorem area_of_triangle : (1 / 2) * s * t * sinA = 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l344_34471


namespace NUMINAMATH_GPT_distance_to_canada_l344_34437

theorem distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) (driving_time : ℝ) (distance : ℝ) :
  speed = 60 ∧ total_time = 7 ∧ stop_time = 1 ∧ driving_time = total_time - stop_time ∧
  distance = speed * driving_time → distance = 360 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_canada_l344_34437


namespace NUMINAMATH_GPT_matrix_pow_2018_l344_34475

open Matrix

-- Define the specific matrix
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![1, 1]]

-- Formalize the statement
theorem matrix_pow_2018 : A ^ 2018 = ![![1, 0], ![2018, 1]] :=
  sorry

end NUMINAMATH_GPT_matrix_pow_2018_l344_34475


namespace NUMINAMATH_GPT_set_M_listed_correctly_l344_34456

theorem set_M_listed_correctly :
  {a : ℕ+ | ∃ (n : ℤ), 4 = n * (1 - a)} = {2, 3, 4} := by
sorry

end NUMINAMATH_GPT_set_M_listed_correctly_l344_34456


namespace NUMINAMATH_GPT_fraction_identity_l344_34447

theorem fraction_identity :
  ( (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) = (432 / 1105) ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l344_34447


namespace NUMINAMATH_GPT_total_games_is_seven_l344_34465

def total_football_games (games_missed : ℕ) (games_attended : ℕ) : ℕ :=
  games_missed + games_attended

theorem total_games_is_seven : total_football_games 4 3 = 7 := 
by
  sorry

end NUMINAMATH_GPT_total_games_is_seven_l344_34465


namespace NUMINAMATH_GPT_find_six_digit_number_l344_34485

theorem find_six_digit_number (a b c d e f : ℕ) (N : ℕ) :
  a = 1 ∧ f = 7 ∧
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
  (f - 1) * 10^5 + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e = 5 * N →
  N = 142857 :=
by
  sorry

end NUMINAMATH_GPT_find_six_digit_number_l344_34485


namespace NUMINAMATH_GPT_exponential_fixed_point_l344_34436

theorem exponential_fixed_point (a : ℝ) (hx₁ : a > 0) (hx₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a ^ x) } := by
  sorry 

end NUMINAMATH_GPT_exponential_fixed_point_l344_34436


namespace NUMINAMATH_GPT_oranges_now_is_50_l344_34412

def initial_fruits : ℕ := 150
def remaining_fruits : ℕ := initial_fruits / 2
def num_limes (L : ℕ) (O : ℕ) : Prop := O = 2 * L
def total_remaining_fruits (L : ℕ) (O : ℕ) : Prop := O + L = remaining_fruits

theorem oranges_now_is_50 : ∃ O L : ℕ, num_limes L O ∧ total_remaining_fruits L O ∧ O = 50 := by
  sorry

end NUMINAMATH_GPT_oranges_now_is_50_l344_34412


namespace NUMINAMATH_GPT_fraction_sum_l344_34440

theorem fraction_sum :
  (1 / 4 : ℚ) + (2 / 9) + (3 / 6) = 35 / 36 := 
sorry

end NUMINAMATH_GPT_fraction_sum_l344_34440


namespace NUMINAMATH_GPT_mechanic_earns_on_fourth_day_l344_34486

theorem mechanic_earns_on_fourth_day 
  (E1 E2 E3 E4 E5 E6 E7 : ℝ)
  (h1 : (E1 + E2 + E3 + E4) / 4 = 18)
  (h2 : (E4 + E5 + E6 + E7) / 4 = 22)
  (h3 : (E1 + E2 + E3 + E4 + E5 + E6 + E7) / 7 = 21) 
  : E4 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_mechanic_earns_on_fourth_day_l344_34486


namespace NUMINAMATH_GPT_polynomial_root_reciprocal_square_sum_l344_34427

theorem polynomial_root_reciprocal_square_sum :
  ∀ (a b c : ℝ), (a + b + c = 6) → (a * b + b * c + c * a = 11) → (a * b * c = 6) →
  (1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 = 49 / 36) :=
by
  intros a b c h_sum h_prod_sum h_prod
  sorry

end NUMINAMATH_GPT_polynomial_root_reciprocal_square_sum_l344_34427


namespace NUMINAMATH_GPT_kaleb_money_earned_l344_34403

-- Definitions based on the conditions
def total_games : ℕ := 10
def non_working_games : ℕ := 8
def price_per_game : ℕ := 6

-- Calculate the number of working games
def working_games : ℕ := total_games - non_working_games

-- Calculate the total money earned by Kaleb
def money_earned : ℕ := working_games * price_per_game

-- The theorem to prove
theorem kaleb_money_earned : money_earned = 12 := by sorry

end NUMINAMATH_GPT_kaleb_money_earned_l344_34403


namespace NUMINAMATH_GPT_value_of_S_l344_34400

-- Defining the condition as an assumption
def one_third_one_eighth_S (S : ℝ) : Prop :=
  (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120

-- The statement we need to prove
theorem value_of_S (S : ℝ) (h : one_third_one_eighth_S S) : S = 120 :=
by
  sorry

end NUMINAMATH_GPT_value_of_S_l344_34400


namespace NUMINAMATH_GPT_regular_polygon_sides_l344_34426

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l344_34426


namespace NUMINAMATH_GPT_tan_150_degree_is_correct_l344_34496

noncomputable def tan_150_degree_is_negative_sqrt_3_div_3 : Prop :=
  let theta := Real.pi * 150 / 180
  let ref_angle := Real.pi * 30 / 180
  let cos_150 := -Real.cos ref_angle
  let sin_150 := Real.sin ref_angle
  Real.tan theta = -Real.sqrt 3 / 3

theorem tan_150_degree_is_correct :
  tan_150_degree_is_negative_sqrt_3_div_3 :=
by
  sorry

end NUMINAMATH_GPT_tan_150_degree_is_correct_l344_34496


namespace NUMINAMATH_GPT_solve_for_x_l344_34408

theorem solve_for_x : (∃ x : ℝ, ((10 - 2 * x) ^ 2 = 4 * x ^ 2 + 16) ∧ x = 2.1) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l344_34408


namespace NUMINAMATH_GPT_sam_distance_traveled_l344_34407

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end NUMINAMATH_GPT_sam_distance_traveled_l344_34407


namespace NUMINAMATH_GPT_isabella_stops_l344_34481

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_stops (P : ℕ → ℚ) (h : ∀ n, P n = 1 / (n * (n + 1))) : 
  ∃ n : ℕ, n = 55 ∧ P n < 1 / 3000 :=
by {
  sorry
}

end NUMINAMATH_GPT_isabella_stops_l344_34481


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_sum_of_first_n_terms_l344_34422

theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ) (h_d_nonzero : d ≠ 0)
  (h_arith : ∀ n, a_n = a_n 0 + n * d)
  (h_S9 : S 9 = 90)
  (h_geom : ∃ (a1 a2 a4 : ℕ), a2^2 = a1 * a4)
  (h_common_diff : d = a_n 1 - a_n 0)
  : ∀ n, a_n = 2 * n  := 
sorry

theorem sum_of_first_n_terms
  (b_n : ℕ → ℕ)
  (T : ℕ → ℕ)
  (a_n : ℕ → ℕ) 
  (h_b_def : ∀ n, b_n = 1 / (a_n n * a_n (n+1)))
  (h_a_form : ∀ n, a_n = 2 * n)
  : ∀ n, T n = n / (4 * n + 4) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_sum_of_first_n_terms_l344_34422


namespace NUMINAMATH_GPT_find_T_value_l344_34443

theorem find_T_value (x y : ℤ) (R : ℤ) (h : R = 30) (h2 : (R / 2) * x * y = 21 * x + 20 * y - 13) :
    x = 3 ∧ y = 2 → x * y = 6 := by
  sorry

end NUMINAMATH_GPT_find_T_value_l344_34443


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l344_34433

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6 ∨ a = 7) (h₂ : b = 6 ∨ b = 7) (h₃ : a ≠ b) :
  (2 * a + b = 19) ∨ (2 * b + a = 20) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l344_34433


namespace NUMINAMATH_GPT_a_b_finish_job_in_15_days_l344_34402

theorem a_b_finish_job_in_15_days (A B C : ℝ) 
  (h1 : A + B + C = 1 / 5)
  (h2 : C = 1 / 7.5) : 
  (1 / (A + B)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_a_b_finish_job_in_15_days_l344_34402


namespace NUMINAMATH_GPT_point_on_parabola_distance_to_directrix_is_4_l344_34493

noncomputable def distance_from_point_to_directrix (x y : ℝ) (directrix : ℝ) : ℝ :=
  abs (x - directrix)

def parabola (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

theorem point_on_parabola_distance_to_directrix_is_4 (m : ℝ) (t : ℝ) :
  parabola t = (3, m) → distance_from_point_to_directrix 3 m (-1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_point_on_parabola_distance_to_directrix_is_4_l344_34493


namespace NUMINAMATH_GPT_area_of_circular_flower_bed_l344_34468

theorem area_of_circular_flower_bed (C : ℝ) (hC : C = 62.8) : ∃ (A : ℝ), A = 314 :=
by
  sorry

end NUMINAMATH_GPT_area_of_circular_flower_bed_l344_34468


namespace NUMINAMATH_GPT_number_of_rectangles_l344_34454

open Real Set

-- Given points A, B, C, D on a line L and a length k
variables {A B C D : ℝ} (L : Set ℝ) (k : ℝ)

-- The points are distinct and ordered on the line
axiom h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D
axiom h2 : A < B ∧ B < C ∧ C < D

-- We need to show there are two rectangles with certain properties
theorem number_of_rectangles : 
  (∃ (rect1 rect2 : Set ℝ), 
    rect1 ≠ rect2 ∧ 
    (∃ (a1 b1 c1 d1 : ℝ), rect1 = {a1, b1, c1, d1} ∧ 
      a1 < b1 ∧ b1 < c1 ∧ c1 < d1 ∧ 
      (d1 - c1 = k ∨ c1 - b1 = k)) ∧ 
    (∃ (a2 b2 c2 d2 : ℝ), rect2 = {a2, b2, c2, d2} ∧ 
      a2 < b2 ∧ b2 < c2 ∧ c2 < d2 ∧ 
      (d2 - c2 = k ∨ c2 - b2 = k))
  ) :=
sorry

end NUMINAMATH_GPT_number_of_rectangles_l344_34454


namespace NUMINAMATH_GPT_problem_statement_l344_34483

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f (x)
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom initial_condition : f 2 = 3

theorem problem_statement : f 2006 + f 2007 = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l344_34483


namespace NUMINAMATH_GPT_evaluate_power_l344_34406

theorem evaluate_power (x : ℝ) (hx : (8:ℝ)^(2 * x) = 11) : 
  2^(x + 1.5) = 11^(1 / 6) * 2 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_power_l344_34406


namespace NUMINAMATH_GPT_largest_number_value_l344_34439

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end NUMINAMATH_GPT_largest_number_value_l344_34439


namespace NUMINAMATH_GPT_gangster_avoid_police_l344_34424

variable (a v : ℝ)
variable (house_side_length streets_distance neighbouring_distance police_interval : ℝ)
variable (police_speed gangster_speed_to_avoid_police : ℝ)

-- Given conditions
axiom house_properties : house_side_length = a ∧ neighbouring_distance = 2 * a
axiom streets_properties : streets_distance = 3 * a
axiom police_properties : police_interval = 9 * a ∧ police_speed = v

-- Correct answer in terms of Lean
theorem gangster_avoid_police :
  gangster_speed_to_avoid_police = 2 * v ∨ gangster_speed_to_avoid_police = v / 2 :=
by
  sorry

end NUMINAMATH_GPT_gangster_avoid_police_l344_34424


namespace NUMINAMATH_GPT_total_number_of_students_is_40_l344_34459

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end NUMINAMATH_GPT_total_number_of_students_is_40_l344_34459


namespace NUMINAMATH_GPT_triangle_ratio_condition_l344_34409

theorem triangle_ratio_condition (a b c : ℝ) (A B C : ℝ) (h1 : b * Real.cos C + c * Real.cos B = 2 * b)
  (h2 : a = b * Real.sin A / Real.sin B)
  (h3 : b = a * Real.sin B / Real.sin A)
  (h4 : c = a * Real.sin C / Real.sin A)
  (h5 : ∀ x, Real.sin (B + C) = Real.sin x): 
  b / a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ratio_condition_l344_34409


namespace NUMINAMATH_GPT_number_of_bowls_l344_34499

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end NUMINAMATH_GPT_number_of_bowls_l344_34499


namespace NUMINAMATH_GPT_sum_youngest_oldest_l344_34448

variables {a1 a2 a3 a4 a5 : ℕ}

def mean_age (x y z u v : ℕ) : ℕ := (x + y + z + u + v) / 5
def median_age (x y z u v : ℕ) : ℕ := z

theorem sum_youngest_oldest
  (h_mean: mean_age a1 a2 a3 a4 a5 = 10) 
  (h_median: median_age a1 a2 a3 a4 a5 = 7)
  (h_sorted: a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) :
  a1 + a5 = 23 :=
sorry

end NUMINAMATH_GPT_sum_youngest_oldest_l344_34448


namespace NUMINAMATH_GPT_solve_modified_system_l344_34466

theorem solve_modified_system (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : 4 * a1 + 6 * b1 = c1) 
  (h2 : 4 * a2 + 6 * b2 = c2) :
  (4 * a1 * 5 + 3 * b1 * 10 = 5 * c1) ∧ (4 * a2 * 5 + 3 * b2 * 10 = 5 * c2) :=
by
  sorry

end NUMINAMATH_GPT_solve_modified_system_l344_34466


namespace NUMINAMATH_GPT_cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l344_34413

theorem cannot_represent_1986_as_sum_of_squares_of_6_odd_integers
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 % 2 = 1) 
  (h2 : a2 % 2 = 1) 
  (h3 : a3 % 2 = 1) 
  (h4 : a4 % 2 = 1) 
  (h5 : a5 % 2 = 1) 
  (h6 : a6 % 2 = 1) : 
  ¬ (1986 = a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2) := 
by 
  sorry

end NUMINAMATH_GPT_cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l344_34413


namespace NUMINAMATH_GPT_cone_from_sector_l344_34480

theorem cone_from_sector 
  (sector_angle : ℝ) (sector_radius : ℝ)
  (circumference : ℝ := (sector_angle / 360) * (2 * Real.pi * sector_radius))
  (base_radius : ℝ := circumference / (2 * Real.pi))
  (slant_height : ℝ := sector_radius) :
  sector_angle = 270 ∧ sector_radius = 12 → base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end NUMINAMATH_GPT_cone_from_sector_l344_34480


namespace NUMINAMATH_GPT_basketball_weight_l344_34453

-- Definitions based on the given conditions
variables (b c : ℕ) -- weights of basketball and bicycle in pounds

-- Condition 1: Nine basketballs weigh the same as six bicycles
axiom condition1 : 9 * b = 6 * c

-- Condition 2: Four bicycles weigh a total of 120 pounds
axiom condition2 : 4 * c = 120

-- The proof statement we need to prove
theorem basketball_weight : b = 20 :=
by
  sorry

end NUMINAMATH_GPT_basketball_weight_l344_34453


namespace NUMINAMATH_GPT_intersection_of_sets_l344_34434

def set_M : Set ℝ := { x | x >= 2 }
def set_N : Set ℝ := { x | -1 <= x ∧ x <= 3 }
def set_intersection : Set ℝ := { x | 2 <= x ∧ x <= 3 }

theorem intersection_of_sets : (set_M ∩ set_N) = set_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l344_34434


namespace NUMINAMATH_GPT_assignment_increment_l344_34491

theorem assignment_increment (M : ℤ) : (M = M + 3) → false :=
by
  sorry

end NUMINAMATH_GPT_assignment_increment_l344_34491


namespace NUMINAMATH_GPT_total_points_l344_34492

theorem total_points (paul_points cousin_points : ℕ) 
  (h_paul : paul_points = 3103) 
  (h_cousin : cousin_points = 2713) : 
  paul_points + cousin_points = 5816 := by
sorry

end NUMINAMATH_GPT_total_points_l344_34492


namespace NUMINAMATH_GPT_problem_abc_l344_34411

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end NUMINAMATH_GPT_problem_abc_l344_34411


namespace NUMINAMATH_GPT_Mona_bikes_30_miles_each_week_l344_34428

theorem Mona_bikes_30_miles_each_week :
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  total_distance = 30 := by
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  show total_distance = 30
  sorry

end NUMINAMATH_GPT_Mona_bikes_30_miles_each_week_l344_34428


namespace NUMINAMATH_GPT_value_of_x_l344_34463

theorem value_of_x (x : ℝ) (h : 4 * x + 5 * x + x + 2 * x = 360) : x = 30 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l344_34463


namespace NUMINAMATH_GPT_ratio_a_div_8_to_b_div_7_l344_34420

theorem ratio_a_div_8_to_b_div_7 (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 8) / (b / 7) = 1 :=
sorry

end NUMINAMATH_GPT_ratio_a_div_8_to_b_div_7_l344_34420


namespace NUMINAMATH_GPT_recurring_decimal_exceeds_by_fraction_l344_34418

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_exceeds_by_fraction_l344_34418


namespace NUMINAMATH_GPT_necessary_condition_inequality_l344_34472

theorem necessary_condition_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 := 
sorry

end NUMINAMATH_GPT_necessary_condition_inequality_l344_34472


namespace NUMINAMATH_GPT_three_circles_area_less_than_total_radius_squared_l344_34419

theorem three_circles_area_less_than_total_radius_squared
    (x y z R : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : R > 0)
    (descartes_theorem : ( (1/x + 1/y + 1/z - 1/R)^2 = 2 * ( (1/x)^2 + (1/y)^2 + (1/z)^2 + (1/R)^2 ) )) :
    x^2 + y^2 + z^2 < 4 * R^2 := 
sorry

end NUMINAMATH_GPT_three_circles_area_less_than_total_radius_squared_l344_34419


namespace NUMINAMATH_GPT_baby_turtles_on_sand_l344_34404

theorem baby_turtles_on_sand (total_swept : ℕ) (total_hatched : ℕ) (h1 : total_hatched = 42) (h2 : total_swept = total_hatched / 3) :
  total_hatched - total_swept = 28 := by
  sorry

end NUMINAMATH_GPT_baby_turtles_on_sand_l344_34404


namespace NUMINAMATH_GPT_values_of_a_and_b_solution_set_inequality_l344_34457

-- Part (I)
theorem values_of_a_and_b (a b : ℝ) (h : ∀ x, -1 < x ∧ x < 1 → x^2 - a * x - x + b < 0) :
  a = -1 ∧ b = -1 := sorry

-- Part (II)
theorem solution_set_inequality (a : ℝ) (h : a = b) :
  (∀ x, x^2 - a * x - x + a < 0 → (x = 1 → false) 
      ∧ (0 < 1 - a → (x = 1 → false))
      ∧ (1 < - a → (x = 1 → false))) := sorry

end NUMINAMATH_GPT_values_of_a_and_b_solution_set_inequality_l344_34457


namespace NUMINAMATH_GPT_find_omega_l344_34446

theorem find_omega 
  (w : ℝ) 
  (h₁ : 0 < w)
  (h₂ : (π / w) = (π / 2)) : w = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_omega_l344_34446


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l344_34444

noncomputable def arithmeticSeq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ d : ℕ) (n : ℕ) (h1 : d = 2)
  (h2 : (a₁ + d)^2 = a₁ * (a₁ + 3 * d)) :
  (a₁ = 2) ∧ (∃ S, S = (n * (2 * a₁ + (n - 1) * d)) / 2 ∧ S = n^2 + n) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l344_34444


namespace NUMINAMATH_GPT_bond_value_after_8_years_l344_34476

theorem bond_value_after_8_years :
  ∀ (P A r t : ℝ), P = 240 → r = 0.0833333333333332 → t = 8 →
  (A = P * (1 + r * t)) → A = 400 :=
by
  sorry

end NUMINAMATH_GPT_bond_value_after_8_years_l344_34476


namespace NUMINAMATH_GPT_minimum_value_of_expression_l344_34435

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (sum_eq : x + y + z = 5) :
  (9 / x + 4 / y + 25 / z) ≥ 20 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l344_34435


namespace NUMINAMATH_GPT_total_tickets_sold_l344_34495

theorem total_tickets_sold (A C : ℕ) (total_revenue : ℝ) (cost_adult cost_child : ℝ) :
  (cost_adult = 6.00) →
  (cost_child = 4.50) →
  (total_revenue = 2100.00) →
  (C = 200) →
  (cost_adult * ↑A + cost_child * ↑C = total_revenue) →
  A + C = 400 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l344_34495


namespace NUMINAMATH_GPT_bottles_not_in_crates_l344_34497

def total_bottles : ℕ := 250
def num_small_crates : ℕ := 5
def num_medium_crates : ℕ := 5
def num_large_crates : ℕ := 5
def bottles_per_small_crate : ℕ := 8
def bottles_per_medium_crate : ℕ := 12
def bottles_per_large_crate : ℕ := 20

theorem bottles_not_in_crates : 
  num_small_crates * bottles_per_small_crate + 
  num_medium_crates * bottles_per_medium_crate + 
  num_large_crates * bottles_per_large_crate = 200 → 
  total_bottles - 200 = 50 := 
by
  sorry

end NUMINAMATH_GPT_bottles_not_in_crates_l344_34497


namespace NUMINAMATH_GPT_gcd_735_1287_l344_34498

theorem gcd_735_1287 : Int.gcd 735 1287 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_735_1287_l344_34498


namespace NUMINAMATH_GPT_corn_cobs_each_row_l344_34442

theorem corn_cobs_each_row (x : ℕ) 
  (h1 : 13 * x + 16 * x = 116) : 
  x = 4 :=
by sorry

end NUMINAMATH_GPT_corn_cobs_each_row_l344_34442


namespace NUMINAMATH_GPT_cost_of_bananas_l344_34401

-- Definitions of the conditions from the problem
namespace BananasCost

variables (A B : ℝ)

-- Condition equations
def condition1 : Prop := 2 * A + B = 7
def condition2 : Prop := A + B = 5

-- The theorem to prove the cost of a bunch of bananas
theorem cost_of_bananas (h1 : condition1 A B) (h2 : condition2 A B) : B = 3 := 
  sorry

end BananasCost

end NUMINAMATH_GPT_cost_of_bananas_l344_34401


namespace NUMINAMATH_GPT_water_polo_team_selection_l344_34421

theorem water_polo_team_selection :
  let total_players := 20
  let team_size := 9
  let goalies := 2
  let remaining_players := total_players - goalies
  let combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  combination total_players goalies * combination remaining_players (team_size - goalies) = 6046560 :=
by
  -- Definitions and calculations to be filled here.
  sorry

end NUMINAMATH_GPT_water_polo_team_selection_l344_34421


namespace NUMINAMATH_GPT_range_of_a_l344_34479

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l344_34479


namespace NUMINAMATH_GPT_number_of_monomials_is_3_l344_34416

def isMonomial (term : String) : Bool :=
  match term with
  | "0" => true
  | "-a" => true
  | "-3x^2y" => true
  | _ => false

def monomialCount (terms : List String) : Nat :=
  terms.filter isMonomial |>.length

theorem number_of_monomials_is_3 :
  monomialCount ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"] = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_monomials_is_3_l344_34416


namespace NUMINAMATH_GPT_sum_of_areas_is_correct_l344_34464

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end NUMINAMATH_GPT_sum_of_areas_is_correct_l344_34464


namespace NUMINAMATH_GPT_real_y_values_l344_34460

theorem real_y_values (x : ℝ) :
  (∃ y : ℝ, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23 / 9 ∨ x ≥ 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_real_y_values_l344_34460


namespace NUMINAMATH_GPT_lockers_count_l344_34455

theorem lockers_count 
(TotalCost : ℝ) 
(first_cents : ℝ) 
(additional_cents : ℝ) 
(locker_start : ℕ) 
(locker_end : ℕ) : 
  TotalCost = 155.94 
  → first_cents = 0 
  → additional_cents = 0.03 
  → locker_start = 2 
  → locker_end = 1825 := 
by
  -- Declare the number of lockers as a variable and use it to construct the proof
  let num_lockers := locker_end - locker_start + 1
  -- The cost for labeling can be calculated and matched with TotalCost
  sorry

end NUMINAMATH_GPT_lockers_count_l344_34455


namespace NUMINAMATH_GPT_pathway_bricks_total_is_280_l344_34450

def total_bricks (n : ℕ) : ℕ :=
  let odd_bricks := 2 * (1 + 1 + ((n / 2) - 1) * 2)
  let even_bricks := 4 * (1 + 2 + (n / 2 - 1) * 2)
  odd_bricks + even_bricks
   
theorem pathway_bricks_total_is_280 (n : ℕ) (h : total_bricks n = 280) : n = 10 :=
sorry

end NUMINAMATH_GPT_pathway_bricks_total_is_280_l344_34450


namespace NUMINAMATH_GPT_pizza_cost_per_slice_correct_l344_34423

noncomputable def pizza_cost_per_slice : ℝ :=
  let base_pizza_cost := 10.00
  let first_topping_cost := 2.00
  let next_two_toppings_cost := 2.00
  let remaining_toppings_cost := 2.00
  let total_cost := base_pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  total_cost / 8

theorem pizza_cost_per_slice_correct :
  pizza_cost_per_slice = 2.00 :=
by
  unfold pizza_cost_per_slice
  sorry

end NUMINAMATH_GPT_pizza_cost_per_slice_correct_l344_34423


namespace NUMINAMATH_GPT_problem_l344_34461

variables {a b c d : ℝ}

theorem problem (h1 : c + d = 14 * a) (h2 : c * d = 15 * b) (h3 : a + b = 14 * c) (h4 : a * b = 15 * d) (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) :
  a + b + c + d = 3150 := sorry

end NUMINAMATH_GPT_problem_l344_34461
