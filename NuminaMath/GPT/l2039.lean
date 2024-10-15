import Mathlib

namespace NUMINAMATH_GPT_matrix_mult_correct_l2039_203954

-- Definition of matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 1],
  ![4, -2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![7, -3],
  ![2, 4]
]

-- The goal is to prove that A * B yields the matrix C
def matrix_product : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![23, -5],
  ![24, -20]
]

theorem matrix_mult_correct : A * B = matrix_product := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_matrix_mult_correct_l2039_203954


namespace NUMINAMATH_GPT_expression_value_l2039_203950

theorem expression_value (x : ℕ) (h : x = 12) : (3 / 2 * x - 3 : ℚ) = 15 := by
  rw [h]
  norm_num
-- sorry to skip the proof if necessary
-- sorry 

end NUMINAMATH_GPT_expression_value_l2039_203950


namespace NUMINAMATH_GPT_twin_ages_l2039_203949

theorem twin_ages (x : ℕ) (h : (x + 1) ^ 2 = x ^ 2 + 15) : x = 7 :=
sorry

end NUMINAMATH_GPT_twin_ages_l2039_203949


namespace NUMINAMATH_GPT_tan_theta_half_l2039_203916

open Real

theorem tan_theta_half (θ : ℝ) 
  (h0 : 0 < θ) 
  (h1 : θ < π / 2) 
  (h2 : ∃ k : ℝ, (sin (2 * θ), cos θ) = k • (cos θ, 1)) : 
  tan θ = 1 / 2 := by 
sorry

end NUMINAMATH_GPT_tan_theta_half_l2039_203916


namespace NUMINAMATH_GPT_provisions_last_days_after_reinforcement_l2039_203936

-- Definitions based on the conditions
def initial_men := 2000
def initial_days := 40
def reinforcement_men := 2000
def days_passed := 20

-- Calculate the total provisions initially
def total_provisions := initial_men * initial_days

-- Calculate the remaining provisions after some days passed
def remaining_provisions := total_provisions - (initial_men * days_passed)

-- Total number of men after reinforcement
def total_men := initial_men + reinforcement_men

-- The Lean statement proving the duration the remaining provisions will last
theorem provisions_last_days_after_reinforcement :
  remaining_provisions / total_men = 10 := by
  sorry

end NUMINAMATH_GPT_provisions_last_days_after_reinforcement_l2039_203936


namespace NUMINAMATH_GPT_infinite_solutions_c_l2039_203920

theorem infinite_solutions_c (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 :=
sorry

end NUMINAMATH_GPT_infinite_solutions_c_l2039_203920


namespace NUMINAMATH_GPT_cost_price_of_book_l2039_203938

theorem cost_price_of_book
  (SP : Real)
  (profit_percentage : Real)
  (h1 : SP = 300)
  (h2 : profit_percentage = 0.20) :
  ∃ CP : Real, CP = 250 :=
by
  -- Proof of the statement
  sorry

end NUMINAMATH_GPT_cost_price_of_book_l2039_203938


namespace NUMINAMATH_GPT_giraffe_statue_price_l2039_203939

variable (G : ℕ) -- Price of a giraffe statue in dollars

-- Conditions as definitions in Lean 4
def giraffe_jade_usage := 120 -- grams
def elephant_jade_usage := 2 * giraffe_jade_usage -- 240 grams
def elephant_price := 350 -- dollars
def total_jade := 1920 -- grams
def additional_profit_with_elephants := 400 -- dollars

-- Prove that the price of a giraffe statue is $150
theorem giraffe_statue_price : 
  16 * G + additional_profit_with_elephants = 8 * elephant_price → G = 150 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_giraffe_statue_price_l2039_203939


namespace NUMINAMATH_GPT_problem_solution_l2039_203914

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2039_203914


namespace NUMINAMATH_GPT_expected_total_rain_correct_l2039_203990

-- Define the probabilities and rain amounts for one day.
def prob_sun : ℝ := 0.30
def prob_rain3 : ℝ := 0.40
def prob_rain8 : ℝ := 0.30
def rain_sun : ℝ := 0
def rain_three : ℝ := 3
def rain_eight : ℝ := 8
def days : ℕ := 7

-- Define the expected value of daily rain.
def E_daily_rain : ℝ :=
  prob_sun * rain_sun + prob_rain3 * rain_three + prob_rain8 * rain_eight

-- Define the expected total rain over seven days.
def E_total_rain : ℝ :=
  days * E_daily_rain

-- Statement of the proof problem.
theorem expected_total_rain_correct : E_total_rain = 25.2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_expected_total_rain_correct_l2039_203990


namespace NUMINAMATH_GPT_initial_games_l2039_203972

def games_given_away : ℕ := 91
def games_left : ℕ := 92

theorem initial_games :
  games_given_away + games_left = 183 :=
by
  sorry

end NUMINAMATH_GPT_initial_games_l2039_203972


namespace NUMINAMATH_GPT_part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l2039_203930

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (2 * (Real.cos (x/2))^2 + Real.sin x) + b

theorem part1_monotonically_increasing_interval (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4 + 2 * k * Real.pi) (Real.pi / 4 + 2 * k * Real.pi) ->
    f x <= f (x + Real.pi) :=
sorry

theorem part1_symmetry_axis (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, f x = f (2 * (Real.pi / 4 + k * Real.pi) - x) :=
sorry

theorem part2_find_a_b (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi)
  (h2 : ∃ (a b : ℝ), ∀ x, x ∈ Set.Icc 0 Real.pi → (a > 0 ∧ 3 ≤ f a b x ∧ f a b x ≤ 4)) :
  (1 - Real.sqrt 2 < a ∧ a < 1 + Real.sqrt 2) ∧ b = 3 :=
sorry

end NUMINAMATH_GPT_part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l2039_203930


namespace NUMINAMATH_GPT_parking_spots_first_level_l2039_203993

theorem parking_spots_first_level (x : ℕ) 
    (h1 : ∃ x, x + (x + 7) + (x + 13) + 14 = 46) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_parking_spots_first_level_l2039_203993


namespace NUMINAMATH_GPT_aitana_jayda_total_spending_l2039_203988

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end NUMINAMATH_GPT_aitana_jayda_total_spending_l2039_203988


namespace NUMINAMATH_GPT_smallest_value_in_interval_l2039_203919

open Real

noncomputable def smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : Prop :=
  1 / x^2 < x ∧
  1 / x^2 < x^2 ∧
  1 / x^2 < 2 * x^2 ∧
  1 / x^2 < 3 * x ∧
  1 / x^2 < sqrt x ∧
  1 / x^2 < 1 / x

theorem smallest_value_in_interval (x : ℝ) (h : 1 < x ∧ x < 2) : smallest_value x h :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_in_interval_l2039_203919


namespace NUMINAMATH_GPT_function_passes_through_one_one_l2039_203910

noncomputable def f (a x : ℝ) : ℝ := a^(x - 1)

theorem function_passes_through_one_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_function_passes_through_one_one_l2039_203910


namespace NUMINAMATH_GPT_whole_process_time_is_6_hours_l2039_203999

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end NUMINAMATH_GPT_whole_process_time_is_6_hours_l2039_203999


namespace NUMINAMATH_GPT_kids_on_excursions_l2039_203948

theorem kids_on_excursions (total_kids : ℕ) (one_fourth_kids_tubing two := total_kids / 4) (half_tubers_rafting : ℕ := one_fourth_kids_tubing / 2) :
  total_kids = 40 → one_fourth_kids_tubing = 10 → half_tubers_rafting = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_kids_on_excursions_l2039_203948


namespace NUMINAMATH_GPT_steps_per_level_l2039_203931

def number_of_steps_per_level (blocks_per_step total_blocks total_levels : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / total_levels

theorem steps_per_level (blocks_per_step : ℕ) (total_blocks : ℕ) (total_levels : ℕ) (h1 : blocks_per_step = 3) (h2 : total_blocks = 96) (h3 : total_levels = 4) :
  number_of_steps_per_level blocks_per_step total_blocks total_levels = 8 := 
by
  sorry

end NUMINAMATH_GPT_steps_per_level_l2039_203931


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l2039_203908

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}
variable {a1 : ℝ}

theorem arithmetic_sequence_properties 
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a1 > 0) 
  (h3 : a 9 + a 10 = a 11) :
  (∀ m n, m < n → a m > a n) ∧ (∀ n, S n = n * (a1 + (d * (n - 1) / 2))) ∧ S 14 > 0 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l2039_203908


namespace NUMINAMATH_GPT_stephen_total_distance_l2039_203911

theorem stephen_total_distance :
  let mountain_height := 40000
  let ascent_fraction := 3 / 4
  let descent_fraction := 2 / 3
  let extra_distance_fraction := 0.10
  let normal_trips := 8
  let harsh_trips := 2
  let ascent_distance := ascent_fraction * mountain_height
  let descent_distance := descent_fraction * ascent_distance
  let normal_trip_distance := ascent_distance + descent_distance
  let harsh_trip_extra_distance := extra_distance_fraction * ascent_distance
  let harsh_trip_distance := ascent_distance + harsh_trip_extra_distance + descent_distance
  let total_normal_distance := normal_trip_distance * normal_trips
  let total_harsh_distance := harsh_trip_distance * harsh_trips
  let total_distance := total_normal_distance + total_harsh_distance
  total_distance = 506000 :=
by
  sorry

end NUMINAMATH_GPT_stephen_total_distance_l2039_203911


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l2039_203958

-- Define the given conditions
def speed_of_stream : ℝ := 4  -- Speed of the stream in km/hr
def distance_downstream : ℝ := 60  -- Distance traveled downstream in km
def time_downstream : ℝ := 3  -- Time taken to travel downstream in hours

-- The statement we need to prove
theorem speed_of_boat_in_still_water (V_b : ℝ) (V_d : ℝ) :
  V_d = distance_downstream / time_downstream →
  V_d = V_b + speed_of_stream →
  V_b = 16 :=
by
  intros Vd_eq D_eq
  sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l2039_203958


namespace NUMINAMATH_GPT_fermats_little_theorem_analogue_l2039_203904

theorem fermats_little_theorem_analogue 
  (a : ℤ) (h1 : Int.gcd a 561 = 1) : a ^ 560 ≡ 1 [ZMOD 561] := 
sorry

end NUMINAMATH_GPT_fermats_little_theorem_analogue_l2039_203904


namespace NUMINAMATH_GPT_train_passes_jogger_in_approximately_25_8_seconds_l2039_203925

noncomputable def jogger_speed_kmh := 7
noncomputable def train_speed_kmh := 60
noncomputable def jogger_head_start_m := 180
noncomputable def train_length_m := 200

noncomputable def kmh_to_ms (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_ms := kmh_to_ms jogger_speed_kmh
noncomputable def train_speed_ms := kmh_to_ms train_speed_kmh

noncomputable def relative_speed_ms := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m
noncomputable def time_to_pass_sec := total_distance_to_cover_m / (relative_speed_ms : ℝ) 

theorem train_passes_jogger_in_approximately_25_8_seconds :
  abs (time_to_pass_sec - 25.8) < 0.1 := sorry

end NUMINAMATH_GPT_train_passes_jogger_in_approximately_25_8_seconds_l2039_203925


namespace NUMINAMATH_GPT_simplify_fraction_complex_l2039_203971

open Complex

theorem simplify_fraction_complex :
  (3 - I) / (2 + 5 * I) = (1 / 29) - (17 / 29) * I := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_complex_l2039_203971


namespace NUMINAMATH_GPT_smallest_integer_solution_l2039_203981

theorem smallest_integer_solution (x : ℤ) :
  (7 - 5 * x < 12) → ∃ (n : ℤ), x = n ∧ n = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_l2039_203981


namespace NUMINAMATH_GPT_intersection_of_M_and_N_is_N_l2039_203917

def M := {x : ℝ | x ≥ -1}
def N := {y : ℝ | y ≥ 0}

theorem intersection_of_M_and_N_is_N : M ∩ N = N := sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_is_N_l2039_203917


namespace NUMINAMATH_GPT_simon_age_in_2010_l2039_203965

theorem simon_age_in_2010 :
  ∀ (s j : ℕ), (j = 16 → (j + 24 = s) → j + (2010 - 2005) + 24 = 45) :=
by 
  intros s j h1 h2 
  sorry

end NUMINAMATH_GPT_simon_age_in_2010_l2039_203965


namespace NUMINAMATH_GPT_even_blue_faces_cubes_correct_l2039_203926

/-- A rectangular wooden block is 6 inches long, 3 inches wide, and 2 inches high.
    The block is painted blue on all six sides and then cut into 1 inch cubes.
    This function determines the number of 1-inch cubes that have a total number
    of blue faces that is an even number (in this case, 2 blue faces). -/
def count_even_blue_faces_cubes : Nat :=
  let length := 6
  let width := 3
  let height := 2
  let total_cubes := length * width * height
  
  -- Calculate corner cubes
  let corners := 8

  -- Calculate edges but not corners cubes
  let edge_not_corners := 
    (4 * (length - 2)) + 
    (4 * (width - 2)) + 
    (4 * (height - 2))

  -- Calculate even number of blue faces cubes 
  let even_number_blue_faces := edge_not_corners

  even_number_blue_faces

theorem even_blue_faces_cubes_correct : count_even_blue_faces_cubes = 20 := by
  -- Place your proof here.
  sorry

end NUMINAMATH_GPT_even_blue_faces_cubes_correct_l2039_203926


namespace NUMINAMATH_GPT_problem_statements_correctness_l2039_203953

theorem problem_statements_correctness :
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (12 ∣ 72 ∧ 12 ∣ 120) ∧ (7 ∣ 49 ∧ 7 ∣ 84) ∧ (7 ∣ 63) → 
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (7 ∣ 63) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_statements_correctness_l2039_203953


namespace NUMINAMATH_GPT_round_robin_teams_l2039_203918

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end NUMINAMATH_GPT_round_robin_teams_l2039_203918


namespace NUMINAMATH_GPT_number_of_tiles_per_row_l2039_203960

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tiles_per_row_l2039_203960


namespace NUMINAMATH_GPT_tan_theta_perpendicular_vectors_l2039_203909

theorem tan_theta_perpendicular_vectors (θ : ℝ) (h : Real.sqrt 3 * Real.cos θ + Real.sin θ = 0) : Real.tan θ = - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_perpendicular_vectors_l2039_203909


namespace NUMINAMATH_GPT_determine_m_l2039_203906

-- Define the conditions: the quadratic equation and the sum of roots
def quadratic_eq (x m : ℝ) : Prop :=
  x^2 + m * x + 2 = 0

def sum_of_roots (x1 x2 : ℝ) : ℝ := x1 + x2

-- Problem Statement: Prove that m = 4
theorem determine_m (x1 x2 m : ℝ) 
  (h1 : quadratic_eq x1 m) 
  (h2 : quadratic_eq x2 m)
  (h3 : sum_of_roots x1 x2 = -4) : 
  m = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l2039_203906


namespace NUMINAMATH_GPT_correct_statements_l2039_203987

-- Definitions
def p_A : ℚ := 1 / 2
def p_B : ℚ := 1 / 3

-- Statements to be verified
def statement1 := (p_A * (1 - p_B) + (1 - p_A) * p_B) = (1 / 2 + 1 / 3)
def statement2 := (p_A * p_B) = (1 / 2 * 1 / 3)
def statement3 := (p_A * (1 - p_B) + p_A * p_B) = (1 / 2 * 2 / 3 + 1 / 2 * 1 / 3)
def statement4 := (1 - (1 - p_A) * (1 - p_B)) = (1 - 1 / 2 * 2 / 3)

-- Theorem stating the correct sequence of statements
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬(statement1 ∨ statement3) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l2039_203987


namespace NUMINAMATH_GPT_sqrt_expression_eq_36_l2039_203900

theorem sqrt_expression_eq_36 : (Real.sqrt ((3^2 + 3^3)^2)) = 36 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_36_l2039_203900


namespace NUMINAMATH_GPT_emily_initial_toys_l2039_203956

theorem emily_initial_toys : ∃ (initial_toys : ℕ), initial_toys = 3 + 4 :=
by
  existsi 7
  sorry

end NUMINAMATH_GPT_emily_initial_toys_l2039_203956


namespace NUMINAMATH_GPT_alpha_beta_range_l2039_203997

theorem alpha_beta_range (α β : ℝ) (P : ℝ × ℝ)
  (h1 : α > 0) 
  (h2 : β > 0) 
  (h3 : P = (α, 3 * β))
  (circle_eq : (α - 1)^2 + 9 * (β^2) = 1) :
  1 < α + β ∧ α + β < 5 / 3 :=
sorry

end NUMINAMATH_GPT_alpha_beta_range_l2039_203997


namespace NUMINAMATH_GPT_stephanie_oranges_l2039_203962

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end NUMINAMATH_GPT_stephanie_oranges_l2039_203962


namespace NUMINAMATH_GPT_chocolate_bar_min_breaks_l2039_203901

theorem chocolate_bar_min_breaks (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∃ k, k = m * n - 1 := by
  sorry

end NUMINAMATH_GPT_chocolate_bar_min_breaks_l2039_203901


namespace NUMINAMATH_GPT_oldest_child_age_correct_l2039_203928

-- Defining the conditions
def jane_start_age := 16
def jane_current_age := 32
def jane_stopped_babysitting_years_ago := 10
def half (x : ℕ) := x / 2

-- Expressing the conditions
def jane_last_babysitting_age := jane_current_age - jane_stopped_babysitting_years_ago
def max_child_age_when_jane_stopped := half jane_last_babysitting_age
def years_since_jane_stopped := jane_stopped_babysitting_years_ago

def calculate_oldest_child_current_age (age : ℕ) : ℕ :=
  age + years_since_jane_stopped

def child_age_when_stopped := max_child_age_when_jane_stopped
def expected_oldest_child_current_age := 21

-- The theorem stating the equivalence
theorem oldest_child_age_correct : 
  calculate_oldest_child_current_age child_age_when_stopped = expected_oldest_child_current_age :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_oldest_child_age_correct_l2039_203928


namespace NUMINAMATH_GPT_quadratic_zeros_interval_l2039_203907

theorem quadratic_zeros_interval (a : ℝ) :
  (5 - 2 * a > 0) ∧ (4 * a^2 - 16 > 0) ∧ (a > 1) ↔ (2 < a ∧ a < 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_zeros_interval_l2039_203907


namespace NUMINAMATH_GPT_num_ordered_pairs_l2039_203989

theorem num_ordered_pairs (N : ℕ) :
  (N = 20) ↔ ∃ (a b : ℕ), 
  (a < b) ∧ (100 ≤ a ∧ a ≤ 1000)
  ∧ (100 ≤ b ∧ b ≤ 1000)
  ∧ (gcd a b * lcm a b = 495 * gcd a b)
  := 
sorry

end NUMINAMATH_GPT_num_ordered_pairs_l2039_203989


namespace NUMINAMATH_GPT_keith_spent_on_tires_l2039_203903

noncomputable def money_spent_on_speakers : ℝ := 136.01
noncomputable def money_spent_on_cd_player : ℝ := 139.38
noncomputable def total_expenditure : ℝ := 387.85
noncomputable def total_spent_on_speakers_and_cd_player : ℝ := money_spent_on_speakers + money_spent_on_cd_player
noncomputable def money_spent_on_new_tires : ℝ := total_expenditure - total_spent_on_speakers_and_cd_player

theorem keith_spent_on_tires :
  money_spent_on_new_tires = 112.46 :=
by
  sorry

end NUMINAMATH_GPT_keith_spent_on_tires_l2039_203903


namespace NUMINAMATH_GPT_intersection_M_N_l2039_203905

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l2039_203905


namespace NUMINAMATH_GPT_like_terms_powers_eq_l2039_203983

theorem like_terms_powers_eq (m n : ℕ) :
  (-2 : ℝ) * (x : ℝ) * (y : ℝ) ^ m = (1 / 3 : ℝ) * (x : ℝ) ^ n * (y : ℝ) ^ 3 → m = 3 ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_powers_eq_l2039_203983


namespace NUMINAMATH_GPT_negate_proposition_l2039_203959

theorem negate_proposition : (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negate_proposition_l2039_203959


namespace NUMINAMATH_GPT_option_b_correct_l2039_203940

theorem option_b_correct : (-(-2)) = abs (-2) := by
  sorry

end NUMINAMATH_GPT_option_b_correct_l2039_203940


namespace NUMINAMATH_GPT_factor_theorem_l2039_203927

noncomputable def Q (b x : ℝ) : ℝ := x^4 - 3 * x^3 + b * x^2 - 12 * x + 24

theorem factor_theorem (b : ℝ) : (∃ x : ℝ, x = -2) ∧ (Q b x = 0) → b = -22 :=
by
  sorry

end NUMINAMATH_GPT_factor_theorem_l2039_203927


namespace NUMINAMATH_GPT_find_x_l2039_203902

theorem find_x {x : ℝ} :
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 :=
by
  intro h
  -- Solution steps would go here, but they are omitted.
  sorry

end NUMINAMATH_GPT_find_x_l2039_203902


namespace NUMINAMATH_GPT_smallest_t_for_temperature_104_l2039_203912

theorem smallest_t_for_temperature_104 : 
  ∃ t : ℝ, (-t^2 + 16*t + 40 = 104) ∧ (t > 0) ∧ (∀ s : ℝ, (-s^2 + 16*s + 40 = 104) ∧ (s > 0) → t ≤ s) :=
sorry

end NUMINAMATH_GPT_smallest_t_for_temperature_104_l2039_203912


namespace NUMINAMATH_GPT_urban_general_hospital_problem_l2039_203934

theorem urban_general_hospital_problem
  (a b c d : ℕ)
  (h1 : b = 3 * c)
  (h2 : a = 2 * b)
  (h3 : d = c / 2)
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1500) :
  5 * d = 1500 / 11 := by
  sorry

end NUMINAMATH_GPT_urban_general_hospital_problem_l2039_203934


namespace NUMINAMATH_GPT_train_speed_l2039_203922

def train_length : ℝ := 360 -- length of the train in meters
def crossing_time : ℝ := 6 -- time taken to cross the man in seconds

theorem train_speed (train_length crossing_time : ℝ) : 
  (train_length = 360) → (crossing_time = 6) → (train_length / crossing_time = 60) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_train_speed_l2039_203922


namespace NUMINAMATH_GPT_remainder_abc_l2039_203937

theorem remainder_abc (a b c : ℕ) 
  (h₀ : a < 9) (h₁ : b < 9) (h₂ : c < 9)
  (h₃ : (a + 3 * b + 2 * c) % 9 = 0)
  (h₄ : (2 * a + 2 * b + 3 * c) % 9 = 3)
  (h₅ : (3 * a + b + 2 * c) % 9 = 6) : 
  (a * b * c) % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_abc_l2039_203937


namespace NUMINAMATH_GPT_value_of_fraction_l2039_203961

theorem value_of_fraction (a b : ℚ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_fraction_l2039_203961


namespace NUMINAMATH_GPT_angle_A_is_60_degrees_l2039_203984

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end NUMINAMATH_GPT_angle_A_is_60_degrees_l2039_203984


namespace NUMINAMATH_GPT_no_root_l2039_203966

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_root_l2039_203966


namespace NUMINAMATH_GPT_factorize1_factorize2_factorize3_factorize4_l2039_203929

-- 1. Factorize 3x - 12x^3
theorem factorize1 (x : ℝ) : 3 * x - 12 * x^3 = 3 * x * (1 - 2 * x) * (1 + 2 * x) := 
sorry

-- 2. Factorize 9m^2 - 4n^2
theorem factorize2 (m n : ℝ) : 9 * m^2 - 4 * n^2 = (3 * m + 2 * n) * (3 * m - 2 * n) := 
sorry

-- 3. Factorize a^2(x - y) + b^2(y - x)
theorem factorize3 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := 
sorry

-- 4. Factorize x^2 - 4xy + 4y^2 - 1
theorem factorize4 (x y : ℝ) : x^2 - 4 * x * y + 4 * y^2 - 1 = (x - y + 1) * (x - y - 1) := 
sorry

end NUMINAMATH_GPT_factorize1_factorize2_factorize3_factorize4_l2039_203929


namespace NUMINAMATH_GPT_farm_entrance_fee_for_students_is_five_l2039_203992

theorem farm_entrance_fee_for_students_is_five
  (students : ℕ) (adults : ℕ) (adult_fee : ℕ) (total_cost : ℕ) (student_fee : ℕ)
  (h_students : students = 35)
  (h_adults : adults = 4)
  (h_adult_fee : adult_fee = 6)
  (h_total_cost : total_cost = 199)
  (h_equation : students * student_fee + adults * adult_fee = total_cost) :
  student_fee = 5 :=
by
  sorry

end NUMINAMATH_GPT_farm_entrance_fee_for_students_is_five_l2039_203992


namespace NUMINAMATH_GPT_sum_series_eq_4_div_9_l2039_203943

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end NUMINAMATH_GPT_sum_series_eq_4_div_9_l2039_203943


namespace NUMINAMATH_GPT_Roy_height_l2039_203977

theorem Roy_height (Sara_height Joe_height Roy_height : ℕ) 
  (h1 : Sara_height = 45)
  (h2 : Sara_height = Joe_height + 6)
  (h3 : Joe_height = Roy_height + 3) :
  Roy_height = 36 :=
by
  sorry

end NUMINAMATH_GPT_Roy_height_l2039_203977


namespace NUMINAMATH_GPT_cookies_per_person_l2039_203955

theorem cookies_per_person (cookies_per_bag : ℕ) (bags : ℕ) (damaged_cookies_per_bag : ℕ) (people : ℕ) (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_each : ℕ) :
  (cookies_per_bag = 738) →
  (bags = 295) →
  (damaged_cookies_per_bag = 13) →
  (people = 125) →
  (total_cookies = cookies_per_bag * bags) →
  (remaining_cookies = total_cookies - (damaged_cookies_per_bag * bags)) →
  (cookies_each = remaining_cookies / people) →
  cookies_each = 1711 :=
by
  sorry 

end NUMINAMATH_GPT_cookies_per_person_l2039_203955


namespace NUMINAMATH_GPT_vacation_cost_l2039_203975

theorem vacation_cost (C : ℝ) (h : C / 3 - C / 4 = 60) : C = 720 := 
by sorry

end NUMINAMATH_GPT_vacation_cost_l2039_203975


namespace NUMINAMATH_GPT_total_cost_production_l2039_203969

-- Define the fixed cost and marginal cost per product as constants
def fixedCost : ℤ := 12000
def marginalCostPerProduct : ℤ := 200
def numberOfProducts : ℤ := 20

-- Define the total cost as the sum of fixed cost and total variable cost
def totalCost : ℤ := fixedCost + (marginalCostPerProduct * numberOfProducts)

-- Prove that the total cost is equal to 16000
theorem total_cost_production : totalCost = 16000 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_production_l2039_203969


namespace NUMINAMATH_GPT_jerry_mowing_income_l2039_203913

theorem jerry_mowing_income (M : ℕ) (week_spending : ℕ) (money_weed_eating : ℕ) (weeks : ℕ)
  (H1 : week_spending = 5)
  (H2 : money_weed_eating = 31)
  (H3 : weeks = 9)
  (H4 : (M + money_weed_eating) = week_spending * weeks)
  : M = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_jerry_mowing_income_l2039_203913


namespace NUMINAMATH_GPT_proof_x_plus_y_sum_l2039_203964

noncomputable def x_and_y_sum (x y : ℝ) : Prop := 31.25 / x = 100 / 9.6 ∧ 13.75 / x = y / 9.6

theorem proof_x_plus_y_sum (x y : ℝ) (h : x_and_y_sum x y) : x + y = 47 :=
sorry

end NUMINAMATH_GPT_proof_x_plus_y_sum_l2039_203964


namespace NUMINAMATH_GPT_sum_of_distances_l2039_203973

theorem sum_of_distances (A B C : ℝ × ℝ) (hA : A.2^2 = 8 * A.1) (hB : B.2^2 = 8 * B.1) 
(hC : C.2^2 = 8 * C.1) (h_centroid : (A.1 + B.1 + C.1) / 3 = 2) : 
  dist (2, 0) A + dist (2, 0) B + dist (2, 0) C = 12 := 
sorry

end NUMINAMATH_GPT_sum_of_distances_l2039_203973


namespace NUMINAMATH_GPT_y_when_x_is_4_l2039_203933

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end NUMINAMATH_GPT_y_when_x_is_4_l2039_203933


namespace NUMINAMATH_GPT_speed_of_stream_l2039_203986

-- Define the conditions as premises
def boat_speed_in_still_water : ℝ := 24
def travel_time_downstream : ℝ := 3
def distance_downstream : ℝ := 84

-- The effective speed downstream is the sum of the boat's speed and the speed of the stream
def effective_speed_downstream (stream_speed : ℝ) : ℝ :=
  boat_speed_in_still_water + stream_speed

-- The speed of the stream
theorem speed_of_stream (stream_speed : ℝ) :
  84 = effective_speed_downstream stream_speed * travel_time_downstream →
  stream_speed = 4 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l2039_203986


namespace NUMINAMATH_GPT_negation_proposition_l2039_203946

variables {a b c : ℝ}

theorem negation_proposition (h : a ≤ b) : a + c ≤ b + c :=
sorry

end NUMINAMATH_GPT_negation_proposition_l2039_203946


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2039_203952

-- Define the sets A and B
def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

-- Prove that the intersection of A and B is {8, 10}
theorem intersection_of_A_and_B : A ∩ B = {8, 10} :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2039_203952


namespace NUMINAMATH_GPT_toy_cars_ratio_proof_l2039_203944

theorem toy_cars_ratio_proof (toys_original : ℕ) (toys_bought_last_month : ℕ) (toys_total : ℕ) :
  toys_original = 25 ∧ toys_bought_last_month = 5 ∧ toys_total = 40 →
  (toys_total - toys_original - toys_bought_last_month) / toys_bought_last_month = 2 :=
by
  sorry

end NUMINAMATH_GPT_toy_cars_ratio_proof_l2039_203944


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l2039_203968

-- Definitions derived from the conditions in the problem:

def molecular_weight_nine_moles (w : ℕ) : ℕ :=
  2664

def molecular_weight_one_mole (w : ℕ) : ℕ :=
  w / 9

-- The theorem to prove, based on the above definitions and conditions:
theorem molecular_weight_of_one_mole (w : ℕ) (hw : molecular_weight_nine_moles w = 2664) :
  molecular_weight_one_mole w = 296 :=
sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l2039_203968


namespace NUMINAMATH_GPT_solve_for_x_l2039_203994

theorem solve_for_x (x : ℝ) (h : 3*x - 4*x + 5*x = 140) : x = 35 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2039_203994


namespace NUMINAMATH_GPT_solve_for_x_l2039_203932

theorem solve_for_x (x : ℝ) (h₁ : 3 * x^2 - 9 * x = 0) (h₂ : x ≠ 0) : x = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l2039_203932


namespace NUMINAMATH_GPT_expected_up_right_paths_l2039_203915

def lattice_points := {p : ℕ × ℕ // p.1 ≤ 5 ∧ p.2 ≤ 5}

def total_paths : ℕ := Nat.choose 10 5

def calculate_paths (x y : ℕ) : ℕ :=
  if h : x ≤ 5 ∧ y ≤ 5 then
    let F := total_paths * 25
    F / 36
  else
    0

theorem expected_up_right_paths : ∃ S, S = 175 :=
  sorry

end NUMINAMATH_GPT_expected_up_right_paths_l2039_203915


namespace NUMINAMATH_GPT_max_reached_at_2001_l2039_203979

noncomputable def a (n : ℕ) : ℝ := n^2 / 1.001^n

theorem max_reached_at_2001 : ∀ n : ℕ, a 2001 ≥ a n := 
sorry

end NUMINAMATH_GPT_max_reached_at_2001_l2039_203979


namespace NUMINAMATH_GPT_cat_chase_rat_l2039_203991

/--
Given:
- The cat chases a rat 6 hours after the rat runs.
- The cat takes 4 hours to reach the rat.
- The average speed of the rat is 36 km/h.
Prove that the average speed of the cat is 90 km/h.
-/
theorem cat_chase_rat
  (t_rat_start : ℕ)
  (t_cat_chase : ℕ)
  (v_rat : ℕ)
  (h1 : t_rat_start = 6)
  (h2 : t_cat_chase = 4)
  (h3 : v_rat = 36)
  (v_cat : ℕ)
  (h4 : 4 * v_cat = t_rat_start * v_rat + t_cat_chase * v_rat) :
  v_cat = 90 :=
by
  sorry

end NUMINAMATH_GPT_cat_chase_rat_l2039_203991


namespace NUMINAMATH_GPT_largest_square_with_five_interior_lattice_points_l2039_203957

theorem largest_square_with_five_interior_lattice_points :
  ∃ (s : ℝ), (∀ (x y : ℤ), 1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → ((⌊s⌋ - 1)^2 = 5) ∧ s^2 = 18 := sorry

end NUMINAMATH_GPT_largest_square_with_five_interior_lattice_points_l2039_203957


namespace NUMINAMATH_GPT_terminating_decimal_of_fraction_l2039_203951

theorem terminating_decimal_of_fraction (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 624) : 
  (∃ m : ℕ, 10^m * (n / 625) = k) → ∃ m, m = 624 :=
sorry

end NUMINAMATH_GPT_terminating_decimal_of_fraction_l2039_203951


namespace NUMINAMATH_GPT_inequality_division_l2039_203980

variable {a b c : ℝ}

theorem inequality_division (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : 
  (a / (a - c)) > (b / (b - c)) := 
sorry

end NUMINAMATH_GPT_inequality_division_l2039_203980


namespace NUMINAMATH_GPT_cube_greater_than_quadratic_minus_linear_plus_one_l2039_203998

variable (x : ℝ)

theorem cube_greater_than_quadratic_minus_linear_plus_one (h : x > 1) :
  x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_GPT_cube_greater_than_quadratic_minus_linear_plus_one_l2039_203998


namespace NUMINAMATH_GPT_intersection_M_N_l2039_203970

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2039_203970


namespace NUMINAMATH_GPT_ratio_of_sums_l2039_203982

theorem ratio_of_sums (a b c d : ℚ) (h1 : b / a = 3) (h2 : d / b = 4) (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l2039_203982


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_no_solution_l2039_203942

-- Problem 1
theorem equation_one_solution (x : ℝ) (h : x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) : x = 0 := 
by 
  sorry

-- Problem 2
theorem equation_two_no_solution (x : ℝ) (h : 2 * x + 9 / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2) : False := 
by 
  sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_no_solution_l2039_203942


namespace NUMINAMATH_GPT_people_in_group_l2039_203924

theorem people_in_group
  (N : ℕ)
  (h1 : ∃ w1 w2 : ℝ, w1 = 65 ∧ w2 = 71 ∧ w2 - w1 = 6)
  (h2 : ∃ avg_increase : ℝ, avg_increase = 1.5 ∧ 6 = avg_increase * N) :
  N = 4 :=
sorry

end NUMINAMATH_GPT_people_in_group_l2039_203924


namespace NUMINAMATH_GPT_coterminal_angle_l2039_203923

theorem coterminal_angle (theta : ℝ) (lower : ℝ) (upper : ℝ) (k : ℤ) : 
  -950 = k * 360 + theta ∧ (lower ≤ theta ∧ theta ≤ upper) → theta = 130 :=
by
  -- Given conditions
  sorry

end NUMINAMATH_GPT_coterminal_angle_l2039_203923


namespace NUMINAMATH_GPT_total_instruments_correct_l2039_203995

def fingers : Nat := 10
def hands : Nat := 2
def heads : Nat := 1

def trumpets := fingers - 3
def guitars := hands + 2
def trombones := heads + 2
def french_horns := guitars - 1
def violins := trumpets / 2
def saxophones := trombones / 3

theorem total_instruments_correct : 
  (trumpets + guitars = trombones + violins + saxophones) →
  trumpets + guitars + trombones + french_horns + violins + saxophones = 21 := by
  sorry

end NUMINAMATH_GPT_total_instruments_correct_l2039_203995


namespace NUMINAMATH_GPT_false_propositions_l2039_203945

theorem false_propositions (p q : Prop) (hnp : ¬ p) (hq : q) :
  (¬ p) ∧ (¬ (p ∧ q)) ∧ (¬ ¬ q) :=
by {
  exact ⟨hnp, not_and_of_not_left q hnp, not_not_intro hq⟩
}

end NUMINAMATH_GPT_false_propositions_l2039_203945


namespace NUMINAMATH_GPT_f_D_not_mapping_to_B_l2039_203935

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B := {y : ℝ | 1 ≤ y ∧ y <= 4}
def f_D (x : ℝ) := 4 - x^2

theorem f_D_not_mapping_to_B : ¬ (∀ x ∈ A, f_D x ∈ B) := sorry

end NUMINAMATH_GPT_f_D_not_mapping_to_B_l2039_203935


namespace NUMINAMATH_GPT_geometric_progression_problem_l2039_203967

open Real

theorem geometric_progression_problem
  (a b c r : ℝ)
  (h1 : a = 20)
  (h2 : b = 40)
  (h3 : c = 10)
  (h4 : b = r * a)
  (h5 : c = r * b) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end NUMINAMATH_GPT_geometric_progression_problem_l2039_203967


namespace NUMINAMATH_GPT_vertex_coordinates_l2039_203974

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 8

-- State the theorem for the coordinates of the vertex
theorem vertex_coordinates : 
  (∃ h k : ℝ, ∀ x : ℝ, parabola x = 2 * (x - h)^2 + k) ∧ h = 1 ∧ k = 8 :=
sorry

end NUMINAMATH_GPT_vertex_coordinates_l2039_203974


namespace NUMINAMATH_GPT_exists_small_area_triangle_l2039_203921

def lattice_point (x y : ℤ) : Prop := |x| ≤ 2 ∧ |y| ≤ 2

def no_three_collinear (points : List (ℤ × ℤ)) : Prop :=
∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
(p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) →
¬ (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) = 0)

noncomputable def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ :=
(1 / 2 : ℚ) * |(p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))|

theorem exists_small_area_triangle {points : List (ℤ × ℤ)}
  (h1 : points.length = 6)
  (h2 : ∀ (p : ℤ × ℤ), p ∈ points → lattice_point p.1 p.2)
  (h3 : no_three_collinear points) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
  triangle_area p1 p2 p3 ≤ 2 := 
sorry

end NUMINAMATH_GPT_exists_small_area_triangle_l2039_203921


namespace NUMINAMATH_GPT_tangent_line_circle_m_values_l2039_203985

theorem tangent_line_circle_m_values {m : ℝ} :
  (∀ (x y: ℝ), 3 * x + 4 * y + m = 0 → (x - 1)^2 + (y + 2)^2 = 4) →
  (m = 15 ∨ m = -5) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_m_values_l2039_203985


namespace NUMINAMATH_GPT_points_on_line_l2039_203941

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end NUMINAMATH_GPT_points_on_line_l2039_203941


namespace NUMINAMATH_GPT_average_other_students_l2039_203976

theorem average_other_students (total_students other_students : ℕ) (mean_score_first : ℕ) 
 (mean_score_class : ℕ) (mean_score_other : ℕ) (h1 : total_students = 20) (h2 : other_students = 10)
 (h3 : mean_score_first = 80) (h4 : mean_score_class = 70) :
 mean_score_other = 60 :=
by
  sorry

end NUMINAMATH_GPT_average_other_students_l2039_203976


namespace NUMINAMATH_GPT_emily_trip_duration_same_l2039_203978

theorem emily_trip_duration_same (s : ℝ) (h_s_pos : 0 < s) : 
  let t1 := (90 : ℝ) / s
  let t2 := (360 : ℝ) / (4 * s)
  t2 = t1 := sorry

end NUMINAMATH_GPT_emily_trip_duration_same_l2039_203978


namespace NUMINAMATH_GPT_keychain_arrangement_l2039_203963

theorem keychain_arrangement (house car locker office key5 key6 : ℕ) :
  (∃ (A B : ℕ), house = A ∧ car = A ∧ locker = B ∧ office = B) →
  (∃ (arrangements : ℕ), arrangements = 24) :=
by
  sorry

end NUMINAMATH_GPT_keychain_arrangement_l2039_203963


namespace NUMINAMATH_GPT_student_correct_answers_l2039_203996

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 64) : C = 88 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l2039_203996


namespace NUMINAMATH_GPT_other_solution_of_quadratic_l2039_203947

theorem other_solution_of_quadratic (x : ℚ) 
  (hx1 : 77 * x^2 - 125 * x + 49 = 0) (hx2 : x = 8/11) : 
  77 * (1 : ℚ)^2 - 125 * (1 : ℚ) + 49 = 0 :=
by sorry

end NUMINAMATH_GPT_other_solution_of_quadratic_l2039_203947
