import Mathlib

namespace NUMINAMATH_GPT_hyperbola_n_range_l1475_147559

noncomputable def hyperbola_range_n (m n : ℝ) : Set ℝ :=
  {n | ∃ (m : ℝ), (m^2 + n) + (3 * m^2 - n) = 4 ∧ ((m^2 + n) * (3 * m^2 - n) > 0) }

theorem hyperbola_n_range : ∀ n : ℝ, n ∈ hyperbola_range_n m n ↔ -1 < n ∧ n < 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_n_range_l1475_147559


namespace NUMINAMATH_GPT_John_profit_is_1500_l1475_147525

-- Defining the conditions
def P_initial : ℕ := 8
def Puppies_given_away : ℕ := P_initial / 2
def Puppies_kept : ℕ := 1
def Price_per_puppy : ℕ := 600
def Payment_stud_owner : ℕ := 300

-- Define the number of puppies John's selling
def Puppies_selling := Puppies_given_away - Puppies_kept

-- Define the total revenue from selling the puppies
def Total_revenue := Puppies_selling * Price_per_puppy

-- Define John’s profit 
def John_profit := Total_revenue - Payment_stud_owner

-- The statement to prove
theorem John_profit_is_1500 : John_profit = 1500 := by
  sorry

end NUMINAMATH_GPT_John_profit_is_1500_l1475_147525


namespace NUMINAMATH_GPT_stratified_sampling_expected_elderly_chosen_l1475_147583

theorem stratified_sampling_expected_elderly_chosen :
  let total := 165
  let to_choose := 15
  let elderly := 22
  (22 : ℚ) / 165 * 15 = 2 := sorry

end NUMINAMATH_GPT_stratified_sampling_expected_elderly_chosen_l1475_147583


namespace NUMINAMATH_GPT_probability_x_lt_y_in_rectangle_l1475_147529

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_x_lt_y_in_rectangle_l1475_147529


namespace NUMINAMATH_GPT_discount_difference_l1475_147544

theorem discount_difference (P : ℝ) (h₁ : 0 < P) : 
  let actual_combined_discount := 1 - (0.75 * 0.85)
  let claimed_discount := 0.40
  actual_combined_discount - claimed_discount = 0.0375 :=
by 
  sorry

end NUMINAMATH_GPT_discount_difference_l1475_147544


namespace NUMINAMATH_GPT_matrix_determinant_l1475_147502

theorem matrix_determinant (x : ℝ) :
  Matrix.det ![![x, x + 2], ![3, 2 * x]] = 2 * x^2 - 3 * x - 6 :=
by
  sorry

end NUMINAMATH_GPT_matrix_determinant_l1475_147502


namespace NUMINAMATH_GPT_xy_zero_l1475_147589

theorem xy_zero (x y : ℝ) (h1 : x + y = 4) (h2 : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end NUMINAMATH_GPT_xy_zero_l1475_147589


namespace NUMINAMATH_GPT_corrected_mean_l1475_147507

theorem corrected_mean (n : ℕ) (incorrect_mean old_obs new_obs : ℚ) 
  (hn : n = 50) (h_mean : incorrect_mean = 40) (hold : old_obs = 15) (hnew : new_obs = 45) :
  ((n * incorrect_mean + (new_obs - old_obs)) / n) = 40.6 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l1475_147507


namespace NUMINAMATH_GPT_least_value_of_expression_l1475_147567

theorem least_value_of_expression : ∃ (x y : ℝ), (2 * x - y + 3)^2 + (x + 2 * y - 1)^2 = 295 / 72 := sorry

end NUMINAMATH_GPT_least_value_of_expression_l1475_147567


namespace NUMINAMATH_GPT_subset_P_Q_l1475_147555

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x^2 - 3 * x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Statement to prove P ⊆ Q
theorem subset_P_Q : P ⊆ Q :=
sorry

end NUMINAMATH_GPT_subset_P_Q_l1475_147555


namespace NUMINAMATH_GPT_train_speed_l1475_147587

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 350) (h_time : time = 7) : 
  length / time = 50 :=
by
  rw [h_length, h_time]
  norm_num

end NUMINAMATH_GPT_train_speed_l1475_147587


namespace NUMINAMATH_GPT_sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l1475_147527

theorem sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6 : 
  (Nat.sqrt 8 - Nat.sqrt 2 - (Nat.sqrt (1 / 3) * Nat.sqrt 6) = 0) :=
by
  sorry

theorem sqrt15_div_sqrt3_add_sqrt5_sub1_sq : 
  (Nat.sqrt 15 / Nat.sqrt 3 + (Nat.sqrt 5 - 1) ^ 2 = 6 - Nat.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l1475_147527


namespace NUMINAMATH_GPT_find_values_of_a_and_b_l1475_147582

-- Definition of the problem and required conditions:
def symmetric_point (a b : ℝ) : Prop :=
  (a = -2) ∧ (b = -3)

theorem find_values_of_a_and_b (a b : ℝ) 
  (h : (a, -3) = (-2, -3) ∨ (2, b) = (2, -3) ∧ (a = -2)) :
  symmetric_point a b :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_a_and_b_l1475_147582


namespace NUMINAMATH_GPT_problem_1_problem_2_l1475_147518

-- Definitions for the sets A and B
def A (x : ℝ) : Prop := -1 < x ∧ x < 2
def B (a : ℝ) (x : ℝ) : Prop := 2 * a - 1 < x ∧ x < 2 * a + 3

-- Problem 1: Range of values for a such that A ⊂ B
theorem problem_1 (a : ℝ) : (∀ x, A x → B a x) ↔ (-1/2 ≤ a ∧ a ≤ 0) := sorry

-- Problem 2: Range of values for a such that A ∩ B = ∅
theorem problem_2 (a : ℝ) : (∀ x, A x → ¬ B a x) ↔ (a ≤ -2 ∨ 3/2 ≤ a) := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1475_147518


namespace NUMINAMATH_GPT_reflections_in_mirrors_l1475_147576

theorem reflections_in_mirrors (x : ℕ)
  (h1 : 30 = 10 * 3)
  (h2 : 18 = 6 * 3)
  (h3 : 88 = 30 + 5 * x + 18 + 3 * x) :
  x = 5 := by
  sorry

end NUMINAMATH_GPT_reflections_in_mirrors_l1475_147576


namespace NUMINAMATH_GPT_jogging_distance_apart_l1475_147593

theorem jogging_distance_apart 
  (anna_rate : ℕ) (mark_rate : ℕ) (time_hours : ℕ) :
  anna_rate = (1 / 20) ∧ mark_rate = (3 / 40) ∧ time_hours = 2 → 
  6 + 3 = 9 :=
by
  -- setting up constants and translating conditions into variables
  have anna_distance : ℕ := 6
  have mark_distance : ℕ := 3
  sorry

end NUMINAMATH_GPT_jogging_distance_apart_l1475_147593


namespace NUMINAMATH_GPT_unit_digit_of_expression_l1475_147565

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end NUMINAMATH_GPT_unit_digit_of_expression_l1475_147565


namespace NUMINAMATH_GPT_teacher_student_arrangements_boy_girl_selection_program_arrangements_l1475_147556

-- Question 1
theorem teacher_student_arrangements : 
  let positions := 5
  let student_arrangements := 720
  positions * student_arrangements = 3600 :=
by
  sorry

-- Question 2
theorem boy_girl_selection :
  let total_selections := 330
  let opposite_selections := 20
  total_selections - opposite_selections = 310 :=
by
  sorry

-- Question 3
theorem program_arrangements :
  let total_permutations := 120
  let relative_order_permutations := 6
  total_permutations / relative_order_permutations = 20 :=
by
  sorry

end NUMINAMATH_GPT_teacher_student_arrangements_boy_girl_selection_program_arrangements_l1475_147556


namespace NUMINAMATH_GPT_closest_to_zero_is_neg_1001_l1475_147554

-- Definitions used in the conditions
def list_of_integers : List Int := [-1101, 1011, -1010, -1001, 1110]

-- Problem statement
theorem closest_to_zero_is_neg_1001 (x : Int) (H : x ∈ list_of_integers) :
  x = -1001 ↔ ∀ y ∈ list_of_integers, abs x ≤ abs y :=
sorry

end NUMINAMATH_GPT_closest_to_zero_is_neg_1001_l1475_147554


namespace NUMINAMATH_GPT_final_answer_is_15_l1475_147504

-- We will translate the conditions from the problem into definitions and then formulate the theorem

-- Define the product of 10 and 12
def product : ℕ := 10 * 12

-- Define the result of dividing this product by 2
def divided_result : ℕ := product / 2

-- Define one-fourth of the divided result
def one_fourth : ℚ := (1/4 : ℚ) * divided_result

-- The theorem statement that verifies the final answer
theorem final_answer_is_15 : one_fourth = 15 := by
  sorry

end NUMINAMATH_GPT_final_answer_is_15_l1475_147504


namespace NUMINAMATH_GPT_value_of_percent_l1475_147571

theorem value_of_percent (x : ℝ) (h : 0.50 * x = 200) : 0.40 * x = 160 :=
sorry

end NUMINAMATH_GPT_value_of_percent_l1475_147571


namespace NUMINAMATH_GPT_smallest_value_expression_l1475_147520

theorem smallest_value_expression (n : ℕ) (hn : n > 0) : (n = 8) ↔ ((n / 2) + (32 / n) = 8) := by
  sorry

end NUMINAMATH_GPT_smallest_value_expression_l1475_147520


namespace NUMINAMATH_GPT_kat_average_training_hours_l1475_147572

def strength_training_sessions_per_week : ℕ := 3
def strength_training_hour_per_session : ℕ := 1
def strength_training_missed_sessions_per_2_weeks : ℕ := 1

def boxing_training_sessions_per_week : ℕ := 4
def boxing_training_hour_per_session : ℝ := 1.5
def boxing_training_skipped_sessions_per_2_weeks : ℕ := 1

def cardio_workout_sessions_per_week : ℕ := 2
def cardio_workout_minutes_per_session : ℕ := 30

def flexibility_training_sessions_per_week : ℕ := 1
def flexibility_training_minutes_per_session : ℕ := 45

def interval_training_sessions_per_week : ℕ := 1
def interval_training_hour_per_session : ℝ := 1.25 -- 1 hour and 15 minutes 

noncomputable def average_hours_per_week : ℝ :=
  let strength_training_per_week : ℝ := ((5 / 2) * strength_training_hour_per_session)
  let boxing_training_per_week : ℝ := ((7 / 2) * boxing_training_hour_per_session)
  let cardio_workout_per_week : ℝ := (cardio_workout_sessions_per_week * cardio_workout_minutes_per_session / 60)
  let flexibility_training_per_week : ℝ := (flexibility_training_sessions_per_week * flexibility_training_minutes_per_session / 60)
  let interval_training_per_week : ℝ := interval_training_hour_per_session
  strength_training_per_week + boxing_training_per_week + cardio_workout_per_week + flexibility_training_per_week + interval_training_per_week

theorem kat_average_training_hours : average_hours_per_week = 10.75 := by
  unfold average_hours_per_week
  norm_num
  sorry

end NUMINAMATH_GPT_kat_average_training_hours_l1475_147572


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1475_147574

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, |x - 3/4| ≤ 1/4 → (x - a) * (x - (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (x - a) * (x - (a + 1)) ≤ 0 → |x - 3/4| ≤ 1/4) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1475_147574


namespace NUMINAMATH_GPT_trader_profit_percent_l1475_147566

-- Definitions based on the conditions
variables (P : ℝ) -- Original price of the car
def discount_price := 0.95 * P
def taxes := 0.03 * P
def maintenance := 0.02 * P
def total_cost := discount_price + taxes + maintenance 
def selling_price := 0.95 * P * 1.60
def profit := selling_price - total_cost

-- Theorem
theorem trader_profit_percent : (profit P / P) * 100 = 52 :=
by
  sorry

end NUMINAMATH_GPT_trader_profit_percent_l1475_147566


namespace NUMINAMATH_GPT_functional_equation_solution_l1475_147506

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y) →
  (f = id ∨ f = abs) :=
by sorry

end NUMINAMATH_GPT_functional_equation_solution_l1475_147506


namespace NUMINAMATH_GPT_most_reasonable_sampling_method_l1475_147551

-- Define the conditions
def significant_difference_by_educational_stage := true
def no_significant_difference_by_gender := true

-- Define the statement
theorem most_reasonable_sampling_method :
  (significant_difference_by_educational_stage ∧ no_significant_difference_by_gender) →
  "Stratified sampling by educational stage" = "most reasonable sampling method" :=
by
  sorry

end NUMINAMATH_GPT_most_reasonable_sampling_method_l1475_147551


namespace NUMINAMATH_GPT_average_of_first_two_numbers_l1475_147516

theorem average_of_first_two_numbers (s1 s2 s3 s4 s5 s6 a b c : ℝ) 
  (h_average_six : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 4.6)
  (h_average_set2 : (s3 + s4) / 2 = 3.8)
  (h_average_set3 : (s5 + s6) / 2 = 6.6)
  (h_total_sum : s1 + s2 + s3 + s4 + s5 + s6 = 27.6) : 
  (s1 + s2) / 2 = 3.4 :=
sorry

end NUMINAMATH_GPT_average_of_first_two_numbers_l1475_147516


namespace NUMINAMATH_GPT_probability_at_least_one_blue_l1475_147532

-- Definitions of the setup
def red_balls := 2
def blue_balls := 2
def total_balls := red_balls + blue_balls
def total_outcomes := (total_balls * (total_balls - 1)) / 2  -- choose 2 out of total
def favorable_outcomes := 10  -- by counting outcomes with at least one blue ball

-- Definition of the proof problem
theorem probability_at_least_one_blue (a b : ℕ) (h1: a = red_balls) (h2: b = blue_balls) :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry  

end NUMINAMATH_GPT_probability_at_least_one_blue_l1475_147532


namespace NUMINAMATH_GPT_cyclist_speed_25_l1475_147564

def speeds_system_eqns (x : ℝ) (y : ℝ) : Prop :=
  (20 / x - 20 / 50 = y) ∧ (70 - (8 / 3) * x = 50 * (7 / 15 - y))

theorem cyclist_speed_25 :
  ∃ y : ℝ, speeds_system_eqns 25 y :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_25_l1475_147564


namespace NUMINAMATH_GPT_batsman_average_20th_l1475_147540

noncomputable def average_after_20th (A : ℕ) : ℕ :=
  let total_runs_19 := 19 * A
  let total_runs_20 := total_runs_19 + 85
  let new_average := (total_runs_20) / 20
  new_average
  
theorem batsman_average_20th (A : ℕ) (h1 : 19 * A + 85 = 20 * (A + 4)) : average_after_20th A = 9 := by
  sorry

end NUMINAMATH_GPT_batsman_average_20th_l1475_147540


namespace NUMINAMATH_GPT_cone_rotation_ratio_l1475_147528

theorem cone_rotation_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rotation_eq : (20 : ℝ) * (2 * Real.pi * r) = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  let p := 1
  let q := 399
  1 + 399 = 400 := by
{
  sorry
}

end NUMINAMATH_GPT_cone_rotation_ratio_l1475_147528


namespace NUMINAMATH_GPT_actual_distance_between_towns_l1475_147531

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_between_towns_l1475_147531


namespace NUMINAMATH_GPT_min_perimeter_l1475_147573

theorem min_perimeter :
  ∃ (a b c : ℕ), 
  (2 * a + 18 * c = 2 * b + 20 * c) ∧ 
  (9 * Real.sqrt (a^2 - (9 * c)^2) = 10 * Real.sqrt (b^2 - (10 * c)^2)) ∧ 
  (10 * (a - b) = 9 * c) ∧ 
  2 * a + 18 * c = 362 := 
sorry

end NUMINAMATH_GPT_min_perimeter_l1475_147573


namespace NUMINAMATH_GPT_kombucha_cost_l1475_147541

variable (C : ℝ)

-- Henry drinks 15 bottles of kombucha every month
def bottles_per_month : ℝ := 15

-- A year has 12 months
def months_per_year : ℝ := 12

-- Total bottles consumed in a year
def total_bottles := bottles_per_month * months_per_year

-- Cash refund per bottle
def refund_per_bottle : ℝ := 0.10

-- Total cash refund for all bottles in a year
def total_refund := total_bottles * refund_per_bottle

-- Number of bottles he can buy with the total refund
def bottles_purchasable_with_refund : ℝ := 6

-- Given that the total refund allows purchasing 6 bottles
def cost_per_bottle_eq : Prop := bottles_purchasable_with_refund * C = total_refund

-- Statement to prove
theorem kombucha_cost : cost_per_bottle_eq C → C = 3 := by
  intros
  sorry

end NUMINAMATH_GPT_kombucha_cost_l1475_147541


namespace NUMINAMATH_GPT_part1_part2_l1475_147575

open Real

noncomputable def f (x a : ℝ) : ℝ := exp x - x^a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) → a ≤ exp 1 :=
sorry

theorem part2 (a x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx : x1 > x2) :
  f x1 a = 0 → f x2 a = 0 → x1 + x2 > 2 * a :=
sorry

end NUMINAMATH_GPT_part1_part2_l1475_147575


namespace NUMINAMATH_GPT_range_of_x_for_sqrt_l1475_147537

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end NUMINAMATH_GPT_range_of_x_for_sqrt_l1475_147537


namespace NUMINAMATH_GPT_perpendicular_vectors_x_l1475_147515

theorem perpendicular_vectors_x 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, -2))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : 
  x = 4 := 
  by 
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_l1475_147515


namespace NUMINAMATH_GPT_cookies_difference_l1475_147530

theorem cookies_difference 
    (initial_sweet : ℕ) (initial_salty : ℕ) (initial_chocolate : ℕ)
    (ate_sweet : ℕ) (ate_salty : ℕ) (ate_chocolate : ℕ)
    (ratio_sweet : ℕ) (ratio_salty : ℕ) (ratio_chocolate : ℕ) :
    initial_sweet = 39 →
    initial_salty = 18 →
    initial_chocolate = 12 →
    ate_sweet = 27 →
    ate_salty = 6 →
    ate_chocolate = 8 →
    ratio_sweet = 3 →
    ratio_salty = 1 →
    ratio_chocolate = 2 →
    ate_sweet - ate_salty = 21 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_cookies_difference_l1475_147530


namespace NUMINAMATH_GPT_perimeter_difference_l1475_147590

theorem perimeter_difference (x : ℝ) :
  let small_square_perimeter := 4 * x
  let large_square_perimeter := 4 * (x + 8)
  large_square_perimeter - small_square_perimeter = 32 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_difference_l1475_147590


namespace NUMINAMATH_GPT_cost_of_25kg_l1475_147563

-- Definitions and conditions
def price_33kg (l q : ℕ) : Prop := 30 * l + 3 * q = 360
def price_36kg (l q : ℕ) : Prop := 30 * l + 6 * q = 420

-- Theorem statement
theorem cost_of_25kg (l q : ℕ) (h1 : 30 * l + 3 * q = 360) (h2 : 30 * l + 6 * q = 420) : 25 * l = 250 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_25kg_l1475_147563


namespace NUMINAMATH_GPT_increase_in_lighting_power_l1475_147535

-- Conditions
def N_before : ℕ := 240
def N_after : ℕ := 300

-- Theorem
theorem increase_in_lighting_power : N_after - N_before = 60 := by
  sorry

end NUMINAMATH_GPT_increase_in_lighting_power_l1475_147535


namespace NUMINAMATH_GPT_find_percentage_l1475_147586

theorem find_percentage (P : ℝ) : 100 * (P / 100) + 20 = 100 → P = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1475_147586


namespace NUMINAMATH_GPT_geometric_sequence_S4_l1475_147581

noncomputable section

def geometric_series_sum (a1 q : ℚ) (n : ℕ) : ℚ := 
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_S4 (a1 : ℚ) (q : ℚ)
  (h1 : a1 * q^3 = 2 * a1)
  (h2 : 5 / 2 = a1 * (q^3 + 2 * q^6)) :
  geometric_series_sum a1 q 4 = 30 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S4_l1475_147581


namespace NUMINAMATH_GPT_draw_points_worth_two_l1475_147588

/-
In a certain football competition, a victory is worth 3 points, a draw is worth some points, and a defeat is worth 0 points. Each team plays 20 matches. A team scored 14 points after 5 games. The team needs to win at least 6 of the remaining matches to reach the 40-point mark by the end of the tournament. Prove that the number of points a draw is worth is 2.
-/

theorem draw_points_worth_two :
  ∃ D, (∀ (victory_points draw_points defeat_points total_matches matches_played points_scored remaining_matches wins_needed target_points),
    victory_points = 3 ∧
    defeat_points = 0 ∧
    total_matches = 20 ∧
    matches_played = 5 ∧
    points_scored = 14 ∧
    remaining_matches = total_matches - matches_played ∧
    wins_needed = 6 ∧
    target_points = 40 ∧
    points_scored + 6 * victory_points + (remaining_matches - wins_needed) * D = target_points ∧
    draw_points = D) →
    D = 2 :=
by
  sorry

end NUMINAMATH_GPT_draw_points_worth_two_l1475_147588


namespace NUMINAMATH_GPT_range_of_a_l1475_147534

def sets_nonempty_intersect (a : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x < 2 ∧ x < a

theorem range_of_a (a : ℝ) (h : sets_nonempty_intersect a) : a > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1475_147534


namespace NUMINAMATH_GPT_find_sum_A_B_l1475_147501

-- Define ω as a root of the polynomial x^2 + x + 1
noncomputable def ω : ℂ := sorry

-- Define the polynomial P
noncomputable def P (x : ℂ) (A B : ℂ) : ℂ := x^101 + A * x + B

-- State the main theorem
theorem find_sum_A_B (A B : ℂ) : 
  (∀ x : ℂ, (x^2 + x + 1 = 0) → P x A B = 0) → A + B = 2 :=
by
  intros Divisibility
  -- Here, you would provide the steps to prove the theorem if necessary
  sorry

end NUMINAMATH_GPT_find_sum_A_B_l1475_147501


namespace NUMINAMATH_GPT_original_number_is_3199_l1475_147596

theorem original_number_is_3199 (n : ℕ) (k : ℕ) (h1 : k = 3200) (h2 : (n + k) % 8 = 0) : n = 3199 :=
sorry

end NUMINAMATH_GPT_original_number_is_3199_l1475_147596


namespace NUMINAMATH_GPT_find_u_value_l1475_147598

theorem find_u_value (h : ∃ n : ℕ, n = 2012) : ∃ u : ℕ, u = 2015 := 
by
  sorry

end NUMINAMATH_GPT_find_u_value_l1475_147598


namespace NUMINAMATH_GPT_concert_ticket_revenue_l1475_147549

theorem concert_ticket_revenue :
  let original_price := 20
  let first_group_discount := 0.40
  let second_group_discount := 0.15
  let third_group_premium := 0.10
  let first_group_size := 10
  let second_group_size := 20
  let third_group_size := 15
  (first_group_size * (original_price - first_group_discount * original_price)) +
  (second_group_size * (original_price - second_group_discount * original_price)) +
  (third_group_size * (original_price + third_group_premium * original_price)) = 790 :=
by
  simp
  sorry

end NUMINAMATH_GPT_concert_ticket_revenue_l1475_147549


namespace NUMINAMATH_GPT_right_triangle_third_side_l1475_147547

theorem right_triangle_third_side (a b : ℝ) (h : a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2)
  (h1 : a = 3 ∧ b = 5 ∨ a = 5 ∧ b = 3) : c = 4 ∨ c = Real.sqrt 34 :=
sorry

end NUMINAMATH_GPT_right_triangle_third_side_l1475_147547


namespace NUMINAMATH_GPT_ab_is_zero_l1475_147521

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem ab_is_zero (a b : ℝ) (h : a - 1 = 0) : a * b = 0 := by
  sorry

end NUMINAMATH_GPT_ab_is_zero_l1475_147521


namespace NUMINAMATH_GPT_probability_of_reaching_last_floor_l1475_147548

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_reaching_last_floor_l1475_147548


namespace NUMINAMATH_GPT_disjoint_union_A_B_l1475_147539

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℕ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

def symmetric_difference (M P : Set ℕ) : Set ℕ :=
  {x | (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P}

theorem disjoint_union_A_B :
  symmetric_difference A B = {1, 3} := by
  sorry

end NUMINAMATH_GPT_disjoint_union_A_B_l1475_147539


namespace NUMINAMATH_GPT_calculate_1307_squared_l1475_147514

theorem calculate_1307_squared : 1307 * 1307 = 1709849 := sorry

end NUMINAMATH_GPT_calculate_1307_squared_l1475_147514


namespace NUMINAMATH_GPT_probability_sequence_l1475_147545

def total_cards := 52
def first_card_is_six_of_diamonds := 1 / total_cards
def remaining_cards := total_cards - 1
def second_card_is_queen_of_hearts (first_card_was_six_of_diamonds : Prop) := 1 / remaining_cards
def probability_six_of_diamonds_and_queen_of_hearts : ℝ :=
  first_card_is_six_of_diamonds * second_card_is_queen_of_hearts sorry

theorem probability_sequence : 
  probability_six_of_diamonds_and_queen_of_hearts = 1 / 2652 := sorry

end NUMINAMATH_GPT_probability_sequence_l1475_147545


namespace NUMINAMATH_GPT_pounds_in_a_ton_l1475_147546

-- Definition of variables based on the given conditions
variables (T E D : ℝ)

-- Condition 1: The elephant weighs 3 tons.
def elephant_weight := E = 3 * T

-- Condition 2: The donkey weighs 90% less than the elephant.
def donkey_weight := D = 0.1 * E

-- Condition 3: Their combined weight is 6600 pounds.
def combined_weight := E + D = 6600

-- Main theorem to prove
theorem pounds_in_a_ton (h1 : elephant_weight T E) (h2 : donkey_weight E D) (h3 : combined_weight E D) : T = 2000 :=
by
  sorry

end NUMINAMATH_GPT_pounds_in_a_ton_l1475_147546


namespace NUMINAMATH_GPT_area_larger_sphere_red_is_83_point_25_l1475_147558

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_area_larger_sphere_red_is_83_point_25_l1475_147558


namespace NUMINAMATH_GPT_eq_implies_neq_neq_not_implies_eq_l1475_147503

variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := a^2 = b^2
def condition2 : Prop := a^2 + b^2 = 2 * a * b

-- Theorem statement representing the problem and conclusion
theorem eq_implies_neq (h : condition2 a b) : condition1 a b :=
by
  sorry

theorem neq_not_implies_eq (h : condition1 a b) : ¬ condition2 a b :=
by
  sorry

end NUMINAMATH_GPT_eq_implies_neq_neq_not_implies_eq_l1475_147503


namespace NUMINAMATH_GPT_distance_between_foci_l1475_147597

-- Defining the given ellipse equation 
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 150 * x + 4 * y^2 + 8 * y + 9 = 0

-- Statement to prove the distance between the foci
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 46.2 := 
sorry

end NUMINAMATH_GPT_distance_between_foci_l1475_147597


namespace NUMINAMATH_GPT_sophia_book_problem_l1475_147524

/-
Prove that the total length of the book P is 270 pages, and verify the number of pages read by Sophia
on the 4th and 5th days (50 and 40 pages respectively), given the following conditions:
1. Sophia finished 2/3 of the book in the first three days.
2. She calculated that she finished 90 more pages than she has yet to read.
3. She plans to finish the entire book within 5 days.
4. She will read 10 fewer pages each day from the 4th day until she finishes.
-/

theorem sophia_book_problem
  (P : ℕ)
  (h1 : (2/3 : ℝ) * P = P - (90 + (1/3 : ℝ) * P))
  (h2 : P = 3 * 90)
  (remaining_pages : ℕ := P / 3)
  (h3 : remaining_pages = 90)
  (pages_day4 : ℕ)
  (pages_day5 : ℕ := pages_day4 - 10)
  (h4 : pages_day4 + pages_day4 - 10 = 90)
  (h5 : 2 * pages_day4 - 10 = 90)
  (h6 : 2 * pages_day4 = 100)
  (h7 : pages_day4 = 50) :
  P = 270 ∧ pages_day4 = 50 ∧ pages_day5 = 40 := 
by {
  sorry -- Proof is skipped
}

end NUMINAMATH_GPT_sophia_book_problem_l1475_147524


namespace NUMINAMATH_GPT_speed_first_32_miles_l1475_147569

theorem speed_first_32_miles (x : ℝ) (y : ℝ) : 
  (100 / x + 0.52 * 100 / x = 32 / y + 68 / (x / 2)) → 
  y = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_speed_first_32_miles_l1475_147569


namespace NUMINAMATH_GPT_area_of_quadrilateral_centroids_l1475_147599

noncomputable def square_side_length : ℝ := 40
noncomputable def point_Q_XQ : ℝ := 15
noncomputable def point_Q_YQ : ℝ := 35

theorem area_of_quadrilateral_centroids (h1 : square_side_length = 40)
    (h2 : point_Q_XQ = 15)
    (h3 : point_Q_YQ = 35) :
    ∃ (area : ℝ), area = 800 / 9 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_centroids_l1475_147599


namespace NUMINAMATH_GPT_minimum_value_of_z_l1475_147560

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_z_l1475_147560


namespace NUMINAMATH_GPT_choir_robe_costs_l1475_147594

theorem choir_robe_costs:
  ∀ (total_robes needed_robes total_cost robe_cost : ℕ),
  total_robes = 30 →
  needed_robes = 30 - 12 →
  total_cost = 36 →
  total_cost = needed_robes * robe_cost →
  robe_cost = 2 :=
by
  intros total_robes needed_robes total_cost robe_cost
  intro h_total_robes h_needed_robes h_total_cost h_cost_eq
  sorry

end NUMINAMATH_GPT_choir_robe_costs_l1475_147594


namespace NUMINAMATH_GPT_suitcase_problem_l1475_147585

noncomputable def weight_of_electronics (k : ℝ) : ℝ :=
  2 * k

theorem suitcase_problem (k : ℝ) (B C E T : ℝ) (hc1 : B = 5 * k) (hc2 : C = 4 * k) (hc3 : E = 2 * k) (hc4 : T = 3 * k) (new_ratio : 5 * k / (4 * k - 7) = 3) :
  E = 6 :=
by
  sorry

end NUMINAMATH_GPT_suitcase_problem_l1475_147585


namespace NUMINAMATH_GPT_distance_between_stripes_correct_l1475_147570

noncomputable def distance_between_stripes : ℝ :=
  let base1 := 20
  let height1 := 50
  let base2 := 65
  let area := base1 * height1
  let d := area / base2
  d

theorem distance_between_stripes_correct : distance_between_stripes = 200 / 13 := by
  sorry

end NUMINAMATH_GPT_distance_between_stripes_correct_l1475_147570


namespace NUMINAMATH_GPT_greatest_common_divisor_XYXY_pattern_l1475_147580

theorem greatest_common_divisor_XYXY_pattern (X Y : ℕ) (hX : X ≥ 0 ∧ X ≤ 9) (hY : Y ≥ 0 ∧ Y ≤ 9) :
  ∃ k, 11 * k = 1001 * X + 10 * Y :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_XYXY_pattern_l1475_147580


namespace NUMINAMATH_GPT_entrance_ticket_cost_l1475_147500

theorem entrance_ticket_cost
  (students teachers : ℕ)
  (total_cost : ℕ)
  (students_count : students = 20)
  (teachers_count : teachers = 3)
  (cost : total_cost = 115) :
  total_cost / (students + teachers) = 5 := by
  sorry

end NUMINAMATH_GPT_entrance_ticket_cost_l1475_147500


namespace NUMINAMATH_GPT_obtuse_angles_at_intersection_l1475_147595

theorem obtuse_angles_at_intersection (lines_intersect_x_at_diff_points : Prop) (lines_not_perpendicular : Prop) 
(lines_form_obtuse_angle_at_intersection : Prop) : 
(lines_intersect_x_at_diff_points ∧ lines_not_perpendicular ∧ lines_form_obtuse_angle_at_intersection) → 
  ∃ (n : ℕ), n = 2 :=
by 
  sorry

end NUMINAMATH_GPT_obtuse_angles_at_intersection_l1475_147595


namespace NUMINAMATH_GPT_evaluate_g_at_8_l1475_147523

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 37 * x ^ 2 - 28 * x - 84

theorem evaluate_g_at_8 : g 8 = 1036 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_8_l1475_147523


namespace NUMINAMATH_GPT_determine_k_l1475_147561

variable (x y z k : ℝ)

theorem determine_k (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end NUMINAMATH_GPT_determine_k_l1475_147561


namespace NUMINAMATH_GPT_number_of_movies_l1475_147553

theorem number_of_movies (B M : ℕ)
  (h1 : B = 15)
  (h2 : B = M + 1) : M = 14 :=
by sorry

end NUMINAMATH_GPT_number_of_movies_l1475_147553


namespace NUMINAMATH_GPT_max_value_2ab_plus_2ac_sqrt3_l1475_147508

variable (a b c : ℝ)
variable (h1 : a^2 + b^2 + c^2 = 1)
variable (h2 : 0 ≤ a)
variable (h3 : 0 ≤ b)
variable (h4 : 0 ≤ c)

theorem max_value_2ab_plus_2ac_sqrt3 : 2 * a * b + 2 * a * c * Real.sqrt 3 ≤ 1 := by
  sorry

end NUMINAMATH_GPT_max_value_2ab_plus_2ac_sqrt3_l1475_147508


namespace NUMINAMATH_GPT_find_a10_l1475_147512

noncomputable def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d a₁, a 1 = a₁ ∧ ∀ n, a (n + 1) = a n + d

theorem find_a10 (a : ℕ → ℤ) (h_seq : arithmeticSequence a) 
  (h1 : a 1 + a 3 + a 5 = 9) 
  (h2 : a 3 * (a 4) ^ 2 = 27) :
  a 10 = -39 ∨ a 10 = 30 :=
sorry

end NUMINAMATH_GPT_find_a10_l1475_147512


namespace NUMINAMATH_GPT_solve_problem_l1475_147577

def problem_statement : Prop :=
  ∀ (n1 n2 c1 : ℕ) (C : ℕ),
  n1 = 18 → 
  c1 = 60 → 
  n2 = 216 →
  n1 * c1 = n2 * C →
  C = 5

theorem solve_problem : problem_statement := by
  intros n1 n2 c1 C h1 h2 h3 h4
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_solve_problem_l1475_147577


namespace NUMINAMATH_GPT_bin_to_oct_l1475_147519

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end NUMINAMATH_GPT_bin_to_oct_l1475_147519


namespace NUMINAMATH_GPT_tangent_315_deg_l1475_147592

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_315_deg_l1475_147592


namespace NUMINAMATH_GPT_man_speed_with_stream_is_4_l1475_147538

noncomputable def man's_speed_with_stream (Vm Vs : ℝ) : ℝ := Vm + Vs

theorem man_speed_with_stream_is_4 (Vm : ℝ) (Vs : ℝ) 
  (h1 : Vm - Vs = 4) 
  (h2 : Vm = 4) : man's_speed_with_stream Vm Vs = 4 :=
by 
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_man_speed_with_stream_is_4_l1475_147538


namespace NUMINAMATH_GPT_total_seeds_eaten_proof_l1475_147505

-- Define the information about the number of seeds eaten by each player
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds

-- Sum the seeds eaten by all the players
def total_seeds_eaten : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds

-- Prove that the total number of seeds eaten is 380
theorem total_seeds_eaten_proof : total_seeds_eaten = 380 :=
by
  -- To be filled in by actual proof steps
  sorry

end NUMINAMATH_GPT_total_seeds_eaten_proof_l1475_147505


namespace NUMINAMATH_GPT_system_of_equations_solutions_l1475_147522

theorem system_of_equations_solutions (x y : ℝ) (h1 : x ^ 5 + y ^ 5 = 1) (h2 : x ^ 6 + y ^ 6 = 1) :
    (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_system_of_equations_solutions_l1475_147522


namespace NUMINAMATH_GPT_find_possible_K_l1475_147591

theorem find_possible_K (K : ℕ) (N : ℕ) (h1 : K * (K + 1) / 2 = N^2) (h2 : N < 150)
  (h3 : ∃ m : ℕ, N^2 = m * (m + 1) / 2) : K = 1 ∨ K = 8 ∨ K = 39 ∨ K = 92 ∨ K = 168 := by
  sorry

end NUMINAMATH_GPT_find_possible_K_l1475_147591


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1475_147568

theorem geometric_sequence_seventh_term (a1 : ℕ) (a6 : ℕ) (r : ℚ)
  (ha1 : a1 = 3) (ha6 : a1 * r^5 = 972) : 
  a1 * r^6 = 2187 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1475_147568


namespace NUMINAMATH_GPT_find_divisor_l1475_147517

theorem find_divisor (d : ℕ) (q r : ℕ) (h₁ : 190 = q * d + r) (h₂ : q = 9) (h₃ : r = 1) : d = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1475_147517


namespace NUMINAMATH_GPT_find_f_l1475_147557

-- Definitions of odd and even functions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x, g (-x) = g x

-- Main theorem
theorem find_f (f g : ℝ → ℝ) (h_odd_f : odd_function f) (h_even_g : even_function g) 
    (h_eq : ∀ x, f x + g x = 1 / (x - 1)) :
  ∀ x, f x = x / (x ^ 2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_f_l1475_147557


namespace NUMINAMATH_GPT_perfect_square_trinomial_m6_l1475_147562

theorem perfect_square_trinomial_m6 (m : ℚ) (h₁ : 0 < m) (h₂ : ∃ a : ℚ, x^2 - 2 * m * x + 36 = (x - a)^2) : m = 6 :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m6_l1475_147562


namespace NUMINAMATH_GPT_min_value_at_constraints_l1475_147543

open Classical

noncomputable def min_value (x y : ℝ) : ℝ := (x^2 + y^2 + x) / (x * y)

def constraints (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x + 2 * y = 1

theorem min_value_at_constraints : 
∃ (x y : ℝ), constraints x y ∧ min_value x y = 2 * Real.sqrt 2 + 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_at_constraints_l1475_147543


namespace NUMINAMATH_GPT_fenced_area_correct_l1475_147584

-- Define the dimensions of the rectangle
def length := 20
def width := 18

-- Define the dimensions of the cutouts
def square_cutout1 := 4
def square_cutout2 := 2

-- Define the areas of the rectangle and the cutouts
def area_rectangle := length * width
def area_cutout1 := square_cutout1 * square_cutout1
def area_cutout2 := square_cutout2 * square_cutout2

-- Define the total area within the fence
def total_area_within_fence := area_rectangle - area_cutout1 - area_cutout2

-- The theorem that needs to be proven
theorem fenced_area_correct : total_area_within_fence = 340 := by
  sorry

end NUMINAMATH_GPT_fenced_area_correct_l1475_147584


namespace NUMINAMATH_GPT_number_divisible_by_37_l1475_147552

def consecutive_ones_1998 : ℕ := (10 ^ 1998 - 1) / 9

theorem number_divisible_by_37 : 37 ∣ consecutive_ones_1998 :=
sorry

end NUMINAMATH_GPT_number_divisible_by_37_l1475_147552


namespace NUMINAMATH_GPT_min_value_objective_function_l1475_147536

theorem min_value_objective_function :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ x - 2 * y - 3 ≤ 0 ∧ (∀ x' y', (x' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - 2 * y' - 3 ≤ 0) → 2 * x' + y' ≥ 2 * x + y)) →
  2 * x + y = 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_objective_function_l1475_147536


namespace NUMINAMATH_GPT_tan_sin_cos_eq_l1475_147513

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end NUMINAMATH_GPT_tan_sin_cos_eq_l1475_147513


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_system1_solve_system2_l1475_147510

-- Problem 1
theorem solve_equation1 (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 :=
by sorry

-- Problem 2
theorem solve_equation2 (x : ℚ) : (3 * x - 2) / 2 = (4 * x + 2) / 3 - 1 → x = 4 :=
by sorry

-- Problem 3
theorem solve_system1 (x y : ℚ) : (3 * x - 7 * y = 8) ∧ (2 * x + y = 11) → x = 5 ∧ y = 1 :=
by sorry

-- Problem 4
theorem solve_system2 (a b c : ℚ) : (a - b + c = 0) ∧ (4 * a + 2 * b + c = 3) ∧ (25 * a + 5 * b + c = 60) → (a = 3) ∧ (b = -2) ∧ (c = -5) :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_system1_solve_system2_l1475_147510


namespace NUMINAMATH_GPT_platform_length_l1475_147542

theorem platform_length (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
  (h_train_length : train_length = 300) (h_time_pole : time_pole = 12) (h_time_platform : time_platform = 39) : 
  ∃ L : ℕ, L = 675 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l1475_147542


namespace NUMINAMATH_GPT_solve_system_of_equations_l1475_147578

theorem solve_system_of_equations :
    ∀ (x y : ℝ), 
    (x^3 * y + x * y^3 = 10) ∧ (x^4 + y^4 = 17) ↔
    (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1) :=
by
    sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1475_147578


namespace NUMINAMATH_GPT_sufficient_not_necessary_for_one_zero_l1475_147511

variable {a x : ℝ}

def f (a x : ℝ) : ℝ := a * x ^ 2 - 2 * x + 1

theorem sufficient_not_necessary_for_one_zero :
  (∃ x : ℝ, f 1 x = 0) ∧ (∀ x : ℝ, f 0 x = -2 * x + 1 → x ≠ 0) → 
  (∃ x : ℝ, f a x = 0) → (a = 1 ∨ f 0 x = 0)  :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_for_one_zero_l1475_147511


namespace NUMINAMATH_GPT_no_integers_for_sum_of_squares_l1475_147579

theorem no_integers_for_sum_of_squares :
  ¬ ∃ a b : ℤ, a^2 + b^2 = 10^100 + 3 :=
by
  sorry

end NUMINAMATH_GPT_no_integers_for_sum_of_squares_l1475_147579


namespace NUMINAMATH_GPT_domain_of_function_l1475_147526

theorem domain_of_function :
  {x : ℝ | -3 < x ∧ x < 2 ∧ x ≠ 1} = {x : ℝ | (2 - x > 0) ∧ (12 + x - x^2 ≥ 0) ∧ (x ≠ 1)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1475_147526


namespace NUMINAMATH_GPT_determine_a_l1475_147550

theorem determine_a (a : ℕ) (h : a / (a + 36) = 9 / 10) : a = 324 :=
sorry

end NUMINAMATH_GPT_determine_a_l1475_147550


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1475_147533

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x ≥ 2}

theorem intersection_of_A_and_B :
  (A ∩ B) = {2} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_A_and_B_l1475_147533


namespace NUMINAMATH_GPT_johns_calorie_intake_l1475_147509

theorem johns_calorie_intake
  (servings : ℕ)
  (calories_per_serving : ℕ)
  (total_calories : ℕ)
  (half_package_calories : ℕ)
  (h1 : servings = 3)
  (h2 : calories_per_serving = 120)
  (h3 : total_calories = servings * calories_per_serving)
  (h4 : half_package_calories = total_calories / 2)
  : half_package_calories = 180 :=
by sorry

end NUMINAMATH_GPT_johns_calorie_intake_l1475_147509
