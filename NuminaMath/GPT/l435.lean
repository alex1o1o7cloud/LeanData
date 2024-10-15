import Mathlib

namespace NUMINAMATH_GPT_population_growth_proof_l435_43591

noncomputable def population_growth (P0 : ℕ) (P200 : ℕ) (t : ℕ) (x : ℝ) : Prop :=
  P200 = P0 * (1 + 1 / x)^t

theorem population_growth_proof :
  population_growth 6 1000000 200 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_population_growth_proof_l435_43591


namespace NUMINAMATH_GPT_find_minimum_value_of_quadratic_l435_43561

theorem find_minimum_value_of_quadratic :
  ∀ (x : ℝ), (x = 5/2) -> (∀ y, y = 3 * x ^ 2 - 15 * x + 7 -> ∀ z, z ≥ y) := 
sorry

end NUMINAMATH_GPT_find_minimum_value_of_quadratic_l435_43561


namespace NUMINAMATH_GPT_smallest_b_for_perfect_square_l435_43525

theorem smallest_b_for_perfect_square (b : ℤ) (h1 : b > 4) (h2 : ∃ n : ℤ, 3 * b + 4 = n * n) : b = 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_perfect_square_l435_43525


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l435_43552

theorem relationship_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (1 / 2) ^ (3 / 2))
  (hb : b = Real.log pi)
  (hc : c = Real.logb 0.5 (3 / 2)) :
  c < a ∧ a < b :=
by 
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l435_43552


namespace NUMINAMATH_GPT_ring_roads_count_l435_43538

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem ring_roads_count : 
  binomial 8 4 * binomial 8 4 - (binomial 10 4 * binomial 6 4) = 1750 := by 
sorry

end NUMINAMATH_GPT_ring_roads_count_l435_43538


namespace NUMINAMATH_GPT_proof_problem_l435_43587

-- Declare x, y as real numbers
variables (x y : ℝ)

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k

-- The main conclusion we need to prove given the condition
theorem proof_problem (k : ℝ) (h : condition x y k) :
  (x^8 + y^8) / (x^8 - y^8) + (x^8 - y^8) / (x^8 + y^8) = (k^4 + 24 * k^2 + 16) / (4 * k^3 + 16 * k) :=
sorry

end NUMINAMATH_GPT_proof_problem_l435_43587


namespace NUMINAMATH_GPT_area_outside_squares_inside_triangle_l435_43516

noncomputable def side_length_large_square : ℝ := 6
noncomputable def side_length_small_square1 : ℝ := 2
noncomputable def side_length_small_square2 : ℝ := 3
noncomputable def area_large_square := side_length_large_square ^ 2
noncomputable def area_small_square1 := side_length_small_square1 ^ 2
noncomputable def area_small_square2 := side_length_small_square2 ^ 2
noncomputable def area_triangle_EFG := area_large_square / 2
noncomputable def total_area_small_squares := area_small_square1 + area_small_square2

theorem area_outside_squares_inside_triangle :
  (area_triangle_EFG - total_area_small_squares) = 5 :=
by
  sorry

end NUMINAMATH_GPT_area_outside_squares_inside_triangle_l435_43516


namespace NUMINAMATH_GPT_simplify_absolute_values_l435_43514

theorem simplify_absolute_values (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a + 2| = 2 :=
sorry

end NUMINAMATH_GPT_simplify_absolute_values_l435_43514


namespace NUMINAMATH_GPT_larger_number_l435_43584

theorem larger_number (HCF A B : ℕ) (factor1 factor2 : ℕ) (h_HCF : HCF = 23) (h_factor1 : factor1 = 14) (h_factor2 : factor2 = 15) (h_LCM : HCF * factor1 * factor2 = A * B) (h_A : A = HCF * factor2) (h_B : B = HCF * factor1) : A = 345 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_l435_43584


namespace NUMINAMATH_GPT_set_intersection_l435_43507

theorem set_intersection (M N : Set ℝ) (hM : M = {x | x < 3}) (hN : N = {x | x > 2}) :
  M ∩ N = {x | 2 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_set_intersection_l435_43507


namespace NUMINAMATH_GPT_repeat_block_of_7_div_13_l435_43532

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end NUMINAMATH_GPT_repeat_block_of_7_div_13_l435_43532


namespace NUMINAMATH_GPT_simple_interest_years_l435_43563

theorem simple_interest_years (P R : ℝ) (T : ℝ) :
  P = 2500 → (2500 * (R + 2) / 100 * T = 2500 * R / 100 * T + 250) → T = 5 :=
by
  intro hP h
  -- Note: Actual proof details would go here
  sorry

end NUMINAMATH_GPT_simple_interest_years_l435_43563


namespace NUMINAMATH_GPT_ticket_prices_count_l435_43547

theorem ticket_prices_count :
  let y := 30
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  ∀ (k : ℕ), (k ∈ divisors) ↔ (60 % k = 0 ∧ 90 % k = 0) → 
  (∃ n : ℕ, n = 8) :=
by
  sorry

end NUMINAMATH_GPT_ticket_prices_count_l435_43547


namespace NUMINAMATH_GPT_find_line_eq_of_given_conditions_l435_43545

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * y + 5 = 0
def line_perpendicular (a b : ℝ) : Prop := a + b + 1 = 0
def is_center (x y : ℝ) : Prop := (x, y) = (0, 3)
def is_eq_of_line (x y : ℝ) : Prop := x - y + 3 = 0

theorem find_line_eq_of_given_conditions (x y : ℝ) (h1 : circle_eq x y) (h2 : line_perpendicular x y) (h3 : is_center x y) : is_eq_of_line x y :=
by
  sorry

end NUMINAMATH_GPT_find_line_eq_of_given_conditions_l435_43545


namespace NUMINAMATH_GPT_determine_set_A_l435_43506

variable (U : Set ℕ) (A : Set ℕ)

theorem determine_set_A (hU : U = {0, 1, 2, 3}) (hcompl : U \ A = {2}) :
  A = {0, 1, 3} :=
by
  sorry

end NUMINAMATH_GPT_determine_set_A_l435_43506


namespace NUMINAMATH_GPT_derivative_at_five_l435_43537

noncomputable def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_five : deriv g 5 = 26 :=
sorry

end NUMINAMATH_GPT_derivative_at_five_l435_43537


namespace NUMINAMATH_GPT_largest_is_C_l435_43513

def A : ℝ := 0.978
def B : ℝ := 0.9719
def C : ℝ := 0.9781
def D : ℝ := 0.917
def E : ℝ := 0.9189

theorem largest_is_C : 
  (C > A) ∧ 
  (C > B) ∧ 
  (C > D) ∧ 
  (C > E) := by
  sorry

end NUMINAMATH_GPT_largest_is_C_l435_43513


namespace NUMINAMATH_GPT_diamond_expression_calculation_l435_43579

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_expression_calculation :
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 :=
by
  sorry

end NUMINAMATH_GPT_diamond_expression_calculation_l435_43579


namespace NUMINAMATH_GPT_second_movie_duration_proof_l435_43559

-- initial duration for the first movie (in minutes)
def first_movie_duration_minutes : ℕ := 1 * 60 + 48

-- additional duration for the second movie (in minutes)
def additional_duration_minutes : ℕ := 25

-- total duration for the second movie (in minutes)
def second_movie_duration_minutes : ℕ := first_movie_duration_minutes + additional_duration_minutes

-- convert total minutes to hours and minutes
def duration_in_hours_and_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem second_movie_duration_proof :
  duration_in_hours_and_minutes second_movie_duration_minutes = (2, 13) :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_second_movie_duration_proof_l435_43559


namespace NUMINAMATH_GPT_solve_for_x_l435_43541

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l435_43541


namespace NUMINAMATH_GPT_area_of_rectangle_l435_43546

-- Define the problem statement and conditions
theorem area_of_rectangle (p d : ℝ) :
  ∃ A : ℝ, (∀ (x y : ℝ), 2 * x + 2 * y = p ∧ x^2 + y^2 = d^2 → A = x * y) →
  A = (p^2 - 4 * d^2) / 8 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l435_43546


namespace NUMINAMATH_GPT_marcia_average_cost_l435_43557

theorem marcia_average_cost :
  let price_apples := 2
  let price_bananas := 1
  let price_oranges := 3
  let count_apples := 12
  let count_bananas := 4
  let count_oranges := 4
  let offer_apples_free := count_apples / 10 * 2
  let offer_oranges_free := count_oranges / 3
  let total_apples := count_apples + offer_apples_free
  let total_oranges := count_oranges + offer_oranges_free
  let total_fruits := total_apples + count_bananas + count_oranges
  let cost_apples := price_apples * (count_apples - offer_apples_free)
  let cost_bananas := price_bananas * count_bananas
  let cost_oranges := price_oranges * (count_oranges - offer_oranges_free)
  let total_cost := cost_apples + cost_bananas + cost_oranges
  let average_cost := total_cost / total_fruits
  average_cost = 1.85 :=
  sorry

end NUMINAMATH_GPT_marcia_average_cost_l435_43557


namespace NUMINAMATH_GPT_alice_winning_strategy_l435_43519

theorem alice_winning_strategy (n : ℕ) (hn : n ≥ 2) : 
  (Alice_has_winning_strategy ↔ n % 4 = 3) :=
sorry

end NUMINAMATH_GPT_alice_winning_strategy_l435_43519


namespace NUMINAMATH_GPT_range_of_a_l435_43548

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, e^x + 1/e^x > a) ∧ (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ↔ (-4 ≤ a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l435_43548


namespace NUMINAMATH_GPT_esther_evening_speed_l435_43596

/-- Esther's average speed in the evening was 30 miles per hour -/
theorem esther_evening_speed : 
  let morning_speed := 45   -- miles per hour
  let total_commuting_time := 1 -- hour
  let morning_distance := 18  -- miles
  let evening_distance := 18  -- miles (same route)
  let time_morning := morning_distance / morning_speed
  let time_evening := total_commuting_time - time_morning
  let evening_speed := evening_distance / time_evening
  evening_speed = 30 := 
by sorry

end NUMINAMATH_GPT_esther_evening_speed_l435_43596


namespace NUMINAMATH_GPT_intersection_of_M_and_N_is_correct_l435_43578

-- Definitions according to conditions
def M : Set ℤ := {-4, -2, 0, 2, 4, 6}
def N : Set ℤ := {x | -3 ≤ x ∧ x ≤ 4}

-- Proof statement
theorem intersection_of_M_and_N_is_correct : (M ∩ N) = {-2, 0, 2, 4} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_is_correct_l435_43578


namespace NUMINAMATH_GPT_terminal_side_angle_l435_43534

open Real

theorem terminal_side_angle (α : ℝ) (m n : ℝ) (h_line : n = 3 * m) (h_radius : m^2 + n^2 = 10) (h_sin : sin α < 0) (h_coincide : tan α = 3) : m - n = 2 :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_angle_l435_43534


namespace NUMINAMATH_GPT_circumscribed_triangle_area_relationship_l435_43550

theorem circumscribed_triangle_area_relationship (X Y Z : ℝ) :
  let a := 15
  let b := 20
  let c := 25
  let triangle_area := (1/2) * a * b
  let diameter := c
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let Z := circle_area / 2
  (X + Y + triangle_area = Z) :=
sorry

end NUMINAMATH_GPT_circumscribed_triangle_area_relationship_l435_43550


namespace NUMINAMATH_GPT_total_exercise_time_l435_43542

-- Definition of constants and speeds for each day
def monday_speed := 2 -- miles per hour
def wednesday_speed := 3 -- miles per hour
def friday_speed := 6 -- miles per hour
def distance := 6 -- miles

-- Function to calculate time given distance and speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Prove the total time spent in a week
theorem total_exercise_time :
  time distance monday_speed + time distance wednesday_speed + time distance friday_speed = 6 :=
by
  -- Insert detailed proof steps here
  sorry

end NUMINAMATH_GPT_total_exercise_time_l435_43542


namespace NUMINAMATH_GPT_certain_number_l435_43530

theorem certain_number (x y : ℝ) (h1 : 0.20 * x = 0.15 * y - 15) (h2 : x = 1050) : y = 1500 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l435_43530


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l435_43574

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a-2)*x + a*y = 1 ↔ 2*x + 3*y = 5) → a = 4/5 := by
sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l435_43574


namespace NUMINAMATH_GPT_carpet_cost_calculation_l435_43512

theorem carpet_cost_calculation
  (length_feet : ℕ)
  (width_feet : ℕ)
  (feet_to_yards : ℕ)
  (cost_per_square_yard : ℕ)
  (h_length : length_feet = 15)
  (h_width : width_feet = 12)
  (h_convert : feet_to_yards = 3)
  (h_cost : cost_per_square_yard = 10) :
  (length_feet / feet_to_yards) *
  (width_feet / feet_to_yards) *
  cost_per_square_yard = 200 := by
  sorry

end NUMINAMATH_GPT_carpet_cost_calculation_l435_43512


namespace NUMINAMATH_GPT_three_op_six_l435_43521

-- Define the new operation @.
def op (a b : ℕ) : ℕ := (a * a * b) / (a + b)

-- The theorem to prove that the value of 3 @ 6 is 6.
theorem three_op_six : op 3 6 = 6 := by 
  sorry

end NUMINAMATH_GPT_three_op_six_l435_43521


namespace NUMINAMATH_GPT_texts_sent_on_Tuesday_l435_43553

theorem texts_sent_on_Tuesday (total_texts monday_texts : Nat) (texts_each_monday : Nat)
  (h_monday : texts_each_monday = 5)
  (h_total : total_texts = 40)
  (h_monday_total : monday_texts = 2 * texts_each_monday) :
  total_texts - monday_texts = 30 := by
  sorry

end NUMINAMATH_GPT_texts_sent_on_Tuesday_l435_43553


namespace NUMINAMATH_GPT_min_fraction_value_l435_43500

noncomputable def min_value_fraction (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z+1)^2 / (2 * x * y * z)

theorem min_fraction_value (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) :
  min_value_fraction x y z h h₁ = 3 + 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_min_fraction_value_l435_43500


namespace NUMINAMATH_GPT_machines_initially_working_l435_43508

theorem machines_initially_working (N x : ℕ) (h1 : N * 4 * R = x)
  (h2 : 20 * 6 * R = 3 * x) : N = 10 :=
by
  sorry

end NUMINAMATH_GPT_machines_initially_working_l435_43508


namespace NUMINAMATH_GPT_sine_cosine_obtuse_angle_l435_43539

theorem sine_cosine_obtuse_angle :
  ∀ P : (ℝ × ℝ), P = (Real.sin 2, Real.cos 2) → (Real.sin 2 > 0) ∧ (Real.cos 2 < 0) → 
  (P.1 > 0) ∧ (P.2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_sine_cosine_obtuse_angle_l435_43539


namespace NUMINAMATH_GPT_value_of_k_l435_43524

theorem value_of_k (k : ℤ) : (1/2)^(22) * (1/(81 : ℝ))^k = 1/(18 : ℝ)^(22) → k = 11 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l435_43524


namespace NUMINAMATH_GPT_initial_distance_l435_43517

-- Define conditions
def fred_speed : ℝ := 4
def sam_speed : ℝ := 4
def sam_distance_when_meet : ℝ := 20

-- States that the initial distance between Fred and Sam is 40 miles considering the given conditions.
theorem initial_distance (d : ℝ) (fred_speed_eq : fred_speed = 4) (sam_speed_eq : sam_speed = 4) (sam_distance_eq : sam_distance_when_meet = 20) :
  d = 40 :=
  sorry

end NUMINAMATH_GPT_initial_distance_l435_43517


namespace NUMINAMATH_GPT_total_area_l435_43543

-- Defining basic dimensions as conditions
def left_vertical_length : ℕ := 7
def top_horizontal_length_left : ℕ := 5
def left_vertical_length_near_top : ℕ := 3
def top_horizontal_length_right_of_center : ℕ := 2
def right_vertical_length_near_center : ℕ := 3
def top_horizontal_length_far_right : ℕ := 2

-- Defining areas of partitioned rectangles
def area_bottom_left_rectangle : ℕ := 7 * 8
def area_middle_rectangle : ℕ := 5 * 3
def area_top_left_rectangle : ℕ := 2 * 8
def area_top_right_rectangle : ℕ := 2 * 7
def area_bottom_right_rectangle : ℕ := 4 * 4

-- Calculate the total area of the figure
theorem total_area : 
  area_bottom_left_rectangle + area_middle_rectangle + area_top_left_rectangle + area_top_right_rectangle + area_bottom_right_rectangle = 117 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_total_area_l435_43543


namespace NUMINAMATH_GPT_factorize_x_squared_minus_one_l435_43504

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_one_l435_43504


namespace NUMINAMATH_GPT_compute_fg_l435_43540

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem compute_fg : f (g (-3)) = 3 := by
  sorry

end NUMINAMATH_GPT_compute_fg_l435_43540


namespace NUMINAMATH_GPT_custom_op_evaluation_l435_43533

def custom_op (x y : ℕ) : ℕ := x * y + x - y

theorem custom_op_evaluation : (custom_op 7 4) - (custom_op 4 7) = 6 := by
  sorry

end NUMINAMATH_GPT_custom_op_evaluation_l435_43533


namespace NUMINAMATH_GPT_farmer_land_l435_43528

-- Define A to be the total land owned by the farmer
variables (A : ℝ)

-- Define the conditions of the problem
def condition_1 (A : ℝ) : ℝ := 0.90 * A
def condition_2 (cleared_land : ℝ) : ℝ := 0.20 * cleared_land
def condition_3 (cleared_land : ℝ) : ℝ := 0.70 * cleared_land
def condition_4 (cleared_land : ℝ) : ℝ := cleared_land - condition_2 cleared_land - condition_3 cleared_land

-- Define the assertion we need to prove
theorem farmer_land (h : condition_4 (condition_1 A) = 630) : A = 7000 :=
by
  sorry

end NUMINAMATH_GPT_farmer_land_l435_43528


namespace NUMINAMATH_GPT_ellen_smoothie_total_l435_43531

theorem ellen_smoothie_total :
  0.2 + 0.1 + 0.2 + 0.15 + 0.05 = 0.7 :=
by sorry

end NUMINAMATH_GPT_ellen_smoothie_total_l435_43531


namespace NUMINAMATH_GPT_candy_distribution_l435_43581

theorem candy_distribution (n : ℕ) (h : n ≥ 2) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, ((k * (k + 1)) / 2) % n = i) ↔ ∃ k : ℕ, n = 2 ^ k :=
by
  sorry

end NUMINAMATH_GPT_candy_distribution_l435_43581


namespace NUMINAMATH_GPT_sum_of_squares_l435_43564

theorem sum_of_squares (r b s : ℕ) 
  (h1 : 2 * r + 3 * b + s = 80) 
  (h2 : 4 * r + 2 * b + 3 * s = 98) : 
  r^2 + b^2 + s^2 = 485 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_squares_l435_43564


namespace NUMINAMATH_GPT_area_after_trimming_l435_43535

-- Define the conditions
def original_side_length : ℝ := 22
def trim_x : ℝ := 6
def trim_y : ℝ := 5

-- Calculate dimensions after trimming
def new_length : ℝ := original_side_length - trim_x
def new_width : ℝ := original_side_length - trim_y

-- Define the goal
theorem area_after_trimming : new_length * new_width = 272 := by
  sorry

end NUMINAMATH_GPT_area_after_trimming_l435_43535


namespace NUMINAMATH_GPT_find_a7_l435_43586

def arithmetic_seq (a₁ d : ℤ) (n : ℤ) : ℤ := a₁ + (n-1) * d

theorem find_a7 (a₁ d : ℤ)
  (h₁ : arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 7 - arithmetic_seq a₁ d 10 = -1)
  (h₂ : arithmetic_seq a₁ d 11 - arithmetic_seq a₁ d 4 = 21) :
  arithmetic_seq a₁ d 7 = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_a7_l435_43586


namespace NUMINAMATH_GPT_clock_angle_at_3_40_l435_43556

noncomputable def hour_hand_angle (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
noncomputable def minute_hand_angle (m : ℕ) : ℝ := m * 6
noncomputable def angle_between_hands (h m : ℕ) : ℝ := 
  let angle := |minute_hand_angle m - hour_hand_angle h m|
  if angle > 180 then 360 - angle else angle

theorem clock_angle_at_3_40 : angle_between_hands 3 40 = 130.0 := 
by
  sorry

end NUMINAMATH_GPT_clock_angle_at_3_40_l435_43556


namespace NUMINAMATH_GPT_quiz_score_of_dropped_student_l435_43572

theorem quiz_score_of_dropped_student (avg16 : ℝ) (avg15 : ℝ) (num_students : ℝ) (dropped_students : ℝ) (x : ℝ)
  (h1 : avg16 = 60.5) (h2 : avg15 = 64) (h3 : num_students = 16) (h4 : dropped_students = 1) :
  x = 60.5 * 16 - 64 * 15 :=
by
  sorry

end NUMINAMATH_GPT_quiz_score_of_dropped_student_l435_43572


namespace NUMINAMATH_GPT_no_super_plus_good_exists_at_most_one_super_plus_good_l435_43505

def is_super_plus_good (board : ℕ → ℕ → ℕ) (n : ℕ) (i j : ℕ) : Prop :=
  (∀ k, k < n → board i k ≤ board i j) ∧ 
  (∀ k, k < n → board k j ≥ board i j)

def arrangement (n : ℕ) := { board : ℕ → ℕ → ℕ // ∀ i j, i < n → j < n → 1 ≤ board i j ∧ board i j ≤ n * n }

-- Prove that in some arrangements, there is no super-plus-good number.
theorem no_super_plus_good_exists (n : ℕ) (h₁ : n = 8) :
  ∃ (b : arrangement n), ∀ i j, ¬ is_super_plus_good b.val n i j := sorry

-- Prove that in every arrangement, there is at most one super-plus-good number.
theorem at_most_one_super_plus_good (n : ℕ) (h : n = 8) :
  ∀ (b : arrangement n), ∃! i j, is_super_plus_good b.val n i j := sorry

end NUMINAMATH_GPT_no_super_plus_good_exists_at_most_one_super_plus_good_l435_43505


namespace NUMINAMATH_GPT_number_of_players_in_hockey_club_l435_43560

-- Defining the problem parameters
def cost_of_gloves : ℕ := 6
def cost_of_helmet := cost_of_gloves + 7
def total_cost_per_set := cost_of_gloves + cost_of_helmet
def total_cost_per_player := 2 * total_cost_per_set
def total_expenditure : ℕ := 3120

-- Defining the target number of players
def num_players : ℕ := total_expenditure / total_cost_per_player

theorem number_of_players_in_hockey_club : num_players = 82 := by
  sorry

end NUMINAMATH_GPT_number_of_players_in_hockey_club_l435_43560


namespace NUMINAMATH_GPT_exists_m_divisible_by_1988_l435_43522

def f (x : ℕ) : ℕ := 3 * x + 2
def iter_function (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iter_function n x)

theorem exists_m_divisible_by_1988 : ∃ m : ℕ, 1988 ∣ iter_function 100 m :=
by sorry

end NUMINAMATH_GPT_exists_m_divisible_by_1988_l435_43522


namespace NUMINAMATH_GPT_find_special_5_digit_number_l435_43502

theorem find_special_5_digit_number :
  ∃! (A : ℤ), (10000 ≤ A ∧ A < 100000) ∧ (A^2 % 100000 = A) ∧ A = 90625 :=
sorry

end NUMINAMATH_GPT_find_special_5_digit_number_l435_43502


namespace NUMINAMATH_GPT_smallest_n_for_two_distinct_tuples_l435_43592

theorem smallest_n_for_two_distinct_tuples : ∃ (n : ℕ), n = 1729 ∧ 
  (∃ (x1 y1 x2 y2 : ℕ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ n = x1^3 + y1^3 ∧ n = x2^3 + y2^3 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2) := sorry

end NUMINAMATH_GPT_smallest_n_for_two_distinct_tuples_l435_43592


namespace NUMINAMATH_GPT_bridget_bakery_profit_l435_43598

theorem bridget_bakery_profit :
  let loaves := 36
  let cost_per_loaf := 1
  let morning_sale_price := 3
  let afternoon_sale_price := 1.5
  let late_afternoon_sale_price := 1
  
  let morning_loaves := (2/3 : ℝ) * loaves
  let morning_revenue := morning_loaves * morning_sale_price
  
  let remaining_after_morning := loaves - morning_loaves
  let afternoon_loaves := (1/2 : ℝ) * remaining_after_morning
  let afternoon_revenue := afternoon_loaves * afternoon_sale_price
  
  let late_afternoon_loaves := remaining_after_morning - afternoon_loaves
  let late_afternoon_revenue := late_afternoon_loaves * late_afternoon_sale_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := loaves * cost_per_loaf
  
  total_revenue - total_cost = 51 := by sorry

end NUMINAMATH_GPT_bridget_bakery_profit_l435_43598


namespace NUMINAMATH_GPT_simplify_expr_l435_43518

theorem simplify_expr (a b : ℝ) (h₁ : a + b = 0) (h₂ : a ≠ b) : (1 - a) + (1 - b) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l435_43518


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l435_43515

theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  (∀ x : ℝ, - (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l435_43515


namespace NUMINAMATH_GPT_range_of_a_for_local_min_l435_43583

noncomputable def f (a x : ℝ) : ℝ := (x - 2 * a) * (x^2 + a^2 * x + 2 * a^3)

theorem range_of_a_for_local_min :
  (∀ a : ℝ, (∃ δ > 0, ∀ ε ∈ Set.Ioo (-δ) δ, f a ε > f a 0) → a < 0 ∨ a > 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_local_min_l435_43583


namespace NUMINAMATH_GPT_solve_for_x_l435_43509

-- Problem definition
def problem_statement (x : ℕ) : Prop :=
  (3 * x / 7 = 15) → x = 35

-- Theorem statement in Lean 4
theorem solve_for_x (x : ℕ) : problem_statement x :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_for_x_l435_43509


namespace NUMINAMATH_GPT_michael_birth_year_l435_43595

theorem michael_birth_year (first_imo_year : ℕ) (annual_event : ∀ n : ℕ, n > 0 → (first_imo_year + n) ≥ first_imo_year) 
  (michael_age_at_10th_imo : ℕ) (imo_count : ℕ) 
  (H1 : first_imo_year = 1959) (H2 : imo_count = 10) (H3 : michael_age_at_10th_imo = 15) : 
  (first_imo_year + imo_count - 1 - michael_age_at_10th_imo = 1953) := 
by 
  sorry

end NUMINAMATH_GPT_michael_birth_year_l435_43595


namespace NUMINAMATH_GPT_lego_count_l435_43503

theorem lego_count 
  (total_legos : ℕ := 500)
  (used_legos : ℕ := total_legos / 2)
  (missing_legos : ℕ := 5) :
  total_legos - used_legos - missing_legos = 245 := 
sorry

end NUMINAMATH_GPT_lego_count_l435_43503


namespace NUMINAMATH_GPT_sum_of_possible_values_of_N_l435_43573

variable (N S : ℝ) (hN : N ≠ 0)

theorem sum_of_possible_values_of_N : 
  (3 * N + 5 / N = S) → 
  ∀ N1 N2 : ℝ, (3 * N1^2 - S * N1 + 5 = 0) ∧ (3 * N2^2 - S * N2 + 5 = 0) → 
  N1 + N2 = S / 3 :=
by 
  intro hS hRoots
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_N_l435_43573


namespace NUMINAMATH_GPT_difference_of_numbers_l435_43570

theorem difference_of_numbers (a b : ℕ) (h1 : a = 2 * b) (h2 : (a + 4) / (b + 4) = 5 / 7) : a - b = 8 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l435_43570


namespace NUMINAMATH_GPT_sandwich_cost_is_five_l435_43599

-- Define the cost of each sandwich
variables (x : ℝ)

-- Conditions
def jack_orders_sandwiches (cost_per_sandwich : ℝ) : Prop :=
  3 * cost_per_sandwich = 15

-- Proof problem statement (no proof provided)
theorem sandwich_cost_is_five (h : jack_orders_sandwiches x) : x = 5 :=
sorry

end NUMINAMATH_GPT_sandwich_cost_is_five_l435_43599


namespace NUMINAMATH_GPT_problem_statement_l435_43566

theorem problem_statement (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) = a (i + 1) ^ a (i + 2)): 
  a 0 = a 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l435_43566


namespace NUMINAMATH_GPT_largest_is_21_l435_43549

theorem largest_is_21(a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29):
  d = 21 := 
sorry

end NUMINAMATH_GPT_largest_is_21_l435_43549


namespace NUMINAMATH_GPT_distance_from_A_to_D_l435_43536

theorem distance_from_A_to_D 
  (A B C D : Type)
  (east_of : B → A)
  (north_of : C → B)
  (distance_AC : Real)
  (angle_BAC : ℝ)
  (north_of_D : D → C)
  (distance_CD : Real) : 
  distance_AC = 5 * Real.sqrt 5 → 
  angle_BAC = 60 → 
  distance_CD = 15 → 
  ∃ (AD : Real), AD =
    Real.sqrt (
      (5 * Real.sqrt 15 / 2) ^ 2 + 
      (5 * Real.sqrt 5 / 2 + 15) ^ 2
    ) :=
by
  intros
  sorry


end NUMINAMATH_GPT_distance_from_A_to_D_l435_43536


namespace NUMINAMATH_GPT_rental_cost_l435_43569

theorem rental_cost (total_cost gallons gas_price mile_cost miles : ℝ)
    (H1 : gallons = 8)
    (H2 : gas_price = 3.50)
    (H3 : mile_cost = 0.50)
    (H4 : miles = 320)
    (H5 : total_cost = 338) :
    total_cost - (gallons * gas_price + miles * mile_cost) = 150 := by
  sorry

end NUMINAMATH_GPT_rental_cost_l435_43569


namespace NUMINAMATH_GPT_James_final_assets_correct_l435_43520

/-- Given the following initial conditions:
- James starts with 60 gold bars.
- He pays 10% in tax.
- He loses half of what is left in a divorce.
- He invests 25% of the remaining gold bars in a stock market and earns an additional gold bar.
- On Monday, he exchanges half of his remaining gold bars at a rate of 5 silver bars for 1 gold bar.
- On Tuesday, he exchanges half of his remaining gold bars at a rate of 7 silver bars for 1 gold bar.
- On Wednesday, he exchanges half of his remaining gold bars at a rate of 3 silver bars for 1 gold bar.

We need to determine:
- The number of silver bars James has,
- The number of remaining gold bars James has, and
- The number of gold bars worth from the stock investment James has after these transactions.
-/
noncomputable def James_final_assets (init_gold : ℕ) : ℕ × ℕ × ℕ :=
  let tax := init_gold / 10
  let gold_after_tax := init_gold - tax
  let gold_after_divorce := gold_after_tax / 2
  let invest_gold := gold_after_divorce * 25 / 100
  let remaining_gold_after_invest := gold_after_divorce - invest_gold
  let gold_after_stock := remaining_gold_after_invest + 1
  let monday_gold_exchanged := gold_after_stock / 2
  let monday_silver := monday_gold_exchanged * 5
  let remaining_gold_after_monday := gold_after_stock - monday_gold_exchanged
  let tuesday_gold_exchanged := remaining_gold_after_monday / 2
  let tuesday_silver := tuesday_gold_exchanged * 7
  let remaining_gold_after_tuesday := remaining_gold_after_monday - tuesday_gold_exchanged
  let wednesday_gold_exchanged := remaining_gold_after_tuesday / 2
  let wednesday_silver := wednesday_gold_exchanged * 3
  let remaining_gold_after_wednesday := remaining_gold_after_tuesday - wednesday_gold_exchanged
  let total_silver := monday_silver + tuesday_silver + wednesday_silver
  (total_silver, remaining_gold_after_wednesday, invest_gold)

theorem James_final_assets_correct : James_final_assets 60 = (99, 3, 6) := 
sorry

end NUMINAMATH_GPT_James_final_assets_correct_l435_43520


namespace NUMINAMATH_GPT_spherical_caps_ratio_l435_43582

theorem spherical_caps_ratio (r : ℝ) (m₁ m₂ : ℝ) (σ₁ σ₂ : ℝ)
  (h₁ : r = 1)
  (h₂ : σ₁ = 2 * π * m₁ + π * (1 - (1 - m₁)^2))
  (h₃ : σ₂ = 2 * π * m₂ + π * (1 - (1 - m₂)^2))
  (h₄ : σ₁ + σ₂ = 5 * π)
  (h₅ : m₁ + m₂ = 2) :
  (2 * m₁ + (1 - (1 - m₁)^2)) / (2 * m₂ + (1 - (1 - m₂)^2)) = 3.6 :=
sorry

end NUMINAMATH_GPT_spherical_caps_ratio_l435_43582


namespace NUMINAMATH_GPT_mod_mult_congruence_l435_43594

theorem mod_mult_congruence (n : ℤ) (h1 : 215 ≡ 65 [ZMOD 75])
  (h2 : 789 ≡ 39 [ZMOD 75]) (h3 : 215 * 789 ≡ n [ZMOD 75]) (hn : 0 ≤ n ∧ n < 75) :
  n = 60 :=
by
  sorry

end NUMINAMATH_GPT_mod_mult_congruence_l435_43594


namespace NUMINAMATH_GPT_octal_rep_square_l435_43575

theorem octal_rep_square (a b c : ℕ) (n : ℕ) (h : n^2 = 8^3 * a + 8^2 * b + 8 * 3 + c) (h₀ : a ≠ 0) : c = 1 :=
sorry

end NUMINAMATH_GPT_octal_rep_square_l435_43575


namespace NUMINAMATH_GPT_correct_statement_l435_43589

theorem correct_statement : 
  (∀ x : ℝ, (x < 0 → x^2 > x)) ∧
  (¬ ∀ x : ℝ, (x^2 > 0 → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x < 0)) ∧
  (¬ ∀ x : ℝ, (x < 1 → x^2 < x)) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l435_43589


namespace NUMINAMATH_GPT_max_k_no_real_roots_max_integer_value_k_no_real_roots_l435_43501

-- Define the quadratic equation with the condition on the discriminant.
theorem max_k_no_real_roots : ∀ k : ℤ, (4 + 4 * (k : ℝ) < 0) ↔ k < -1 := sorry

-- Prove that the maximum integer value of k satisfying this condition is -2.
theorem max_integer_value_k_no_real_roots : ∃ k_max : ℤ, k_max ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } ∧ ∀ k' : ℤ, k' ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } → k' ≤ k_max :=
sorry

end NUMINAMATH_GPT_max_k_no_real_roots_max_integer_value_k_no_real_roots_l435_43501


namespace NUMINAMATH_GPT_faster_train_speed_l435_43593

theorem faster_train_speed (dist_between_stations : ℕ) (extra_distance : ℕ) (slower_speed : ℕ) 
  (dist_between_stations_eq : dist_between_stations = 444)
  (extra_distance_eq : extra_distance = 60) 
  (slower_speed_eq : slower_speed = 16) :
  ∃ (faster_speed : ℕ), faster_speed = 21 := by
  sorry

end NUMINAMATH_GPT_faster_train_speed_l435_43593


namespace NUMINAMATH_GPT_zeros_of_quadratic_l435_43523

def f (x : ℝ) := x^2 - 2 * x - 3

theorem zeros_of_quadratic : ∀ x, f x = 0 ↔ (x = 3 ∨ x = -1) := 
by 
  sorry

end NUMINAMATH_GPT_zeros_of_quadratic_l435_43523


namespace NUMINAMATH_GPT_complex_imaginary_part_l435_43562

theorem complex_imaginary_part (z : ℂ) (h : z + (3 - 4 * I) = 1) : z.im = 4 :=
  sorry

end NUMINAMATH_GPT_complex_imaginary_part_l435_43562


namespace NUMINAMATH_GPT_employee_pays_correct_amount_l435_43597

def wholesale_cost : ℝ := 200
def markup_percentage : ℝ := 0.20
def discount_percentage : ℝ := 0.10

def retail_price (wholesale: ℝ) (markup_percentage: ℝ) : ℝ :=
  wholesale * (1 + markup_percentage)

def discount_amount (price: ℝ) (discount_percentage: ℝ) : ℝ :=
  price * discount_percentage

def final_price (retail: ℝ) (discount: ℝ) : ℝ :=
  retail - discount

theorem employee_pays_correct_amount : final_price (retail_price wholesale_cost markup_percentage) 
                                                     (discount_amount (retail_price wholesale_cost markup_percentage) discount_percentage) = 216 := 
by
  sorry

end NUMINAMATH_GPT_employee_pays_correct_amount_l435_43597


namespace NUMINAMATH_GPT_sum_of_divisors_117_l435_43590

-- Defining the conditions in Lean
def n : ℕ := 117
def is_factorization : n = 3^2 * 13 := by rfl

-- The sum-of-divisors function can be defined based on the problem
def sum_of_divisors (n : ℕ) : ℕ :=
  (1 + 3 + 3^2) * (1 + 13)

-- Assertion of the correct answer
theorem sum_of_divisors_117 : sum_of_divisors n = 182 := by
  sorry

end NUMINAMATH_GPT_sum_of_divisors_117_l435_43590


namespace NUMINAMATH_GPT_tom_spent_correct_amount_l435_43576

-- Define the prices of the games
def batman_game_price : ℝ := 13.6
def superman_game_price : ℝ := 5.06

-- Define the total amount spent calculation
def total_spent := batman_game_price + superman_game_price

-- The main statement to prove
theorem tom_spent_correct_amount : total_spent = 18.66 := by
  -- Proof (intended)
  sorry

end NUMINAMATH_GPT_tom_spent_correct_amount_l435_43576


namespace NUMINAMATH_GPT_square_window_side_length_l435_43577

-- Definitions based on the conditions
def total_panes := 8
def rows := 2
def cols := 4
def height_ratio := 3
def width_ratio := 1
def border_width := 3

-- The statement to prove
theorem square_window_side_length :
  let height := 3 * (1 : ℝ)
  let width := 1 * (1 : ℝ)
  let total_width := cols * width + (cols + 1) * border_width
  let total_height := rows * height + (rows + 1) * border_width
  total_width = total_height → total_width = 27 :=
by
  sorry

end NUMINAMATH_GPT_square_window_side_length_l435_43577


namespace NUMINAMATH_GPT_total_roasted_marshmallows_l435_43527

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end NUMINAMATH_GPT_total_roasted_marshmallows_l435_43527


namespace NUMINAMATH_GPT_problem_correct_l435_43551

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def is_nat_lt_10 (n : ℕ) : Prop := n < 10
def not_zero (n : ℕ) : Prop := n ≠ 0

structure Matrix4x4 :=
  (a₀₀ a₀₁ a₀₂ a₀₃ : ℕ)
  (a₁₀ a₁₁ a₁₂ a₁₃ : ℕ)
  (a₂₀ a₂₁ a₂₂ a₂₃ : ℕ)
  (a₃₀ a₃₁ a₃₂ a₃₃ : ℕ)

def valid_matrix (M : Matrix4x4) : Prop :=
  -- Each cell must be a natural number less than 10
  is_nat_lt_10 M.a₀₀ ∧ is_nat_lt_10 M.a₀₁ ∧ is_nat_lt_10 M.a₀₂ ∧ is_nat_lt_10 M.a₀₃ ∧
  is_nat_lt_10 M.a₁₀ ∧ is_nat_lt_10 M.a₁₁ ∧ is_nat_lt_10 M.a₁₂ ∧ is_nat_lt_10 M.a₁₃ ∧
  is_nat_lt_10 M.a₂₀ ∧ is_nat_lt_10 M.a₂₁ ∧ is_nat_lt_10 M.a₂₂ ∧ is_nat_lt_10 M.a₂₃ ∧
  is_nat_lt_10 M.a₃₀ ∧ is_nat_lt_10 M.a₃₁ ∧ is_nat_lt_10 M.a₃₂ ∧ is_nat_lt_10 M.a₃₃ ∧

  -- Cells in the same region must contain the same number
  M.a₀₀ = M.a₁₀ ∧ M.a₀₀ = M.a₂₀ ∧ M.a₀₀ = M.a₃₀ ∧
  M.a₂₀ = M.a₂₁ ∧
  M.a₂₂ = M.a₂₃ ∧ M.a₂₂ = M.a₃₂ ∧ M.a₂₂ = M.a₃₃ ∧
  M.a₀₃ = M.a₁₃ ∧
  
  -- Cells in the leftmost column cannot contain the number 0
  not_zero M.a₀₀ ∧ not_zero M.a₁₀ ∧ not_zero M.a₂₀ ∧ not_zero M.a₃₀ ∧

  -- The four-digit number formed by the first row is 2187
  is_four_digit (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃) ∧ 
  (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃ = 2187) ∧
  
  -- The four-digit number formed by the second row is 7387
  is_four_digit (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃) ∧ 
  (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃ = 7387) ∧
  
  -- The four-digit number formed by the third row is 7744
  is_four_digit (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃) ∧ 
  (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃ = 7744) ∧
  
  -- The four-digit number formed by the fourth row is 7844
  is_four_digit (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃) ∧ 
  (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃ = 7844)

noncomputable def problem_solution : Matrix4x4 :=
{ a₀₀ := 2, a₀₁ := 1, a₀₂ := 8, a₀₃ := 7,
  a₁₀ := 7, a₁₁ := 3, a₁₂ := 8, a₁₃ := 7,
  a₂₀ := 7, a₂₁ := 7, a₂₂ := 4, a₂₃ := 4,
  a₃₀ := 7, a₃₁ := 8, a₃₂ := 4, a₃₃ := 4 }

theorem problem_correct : valid_matrix problem_solution :=
by
  -- The proof would go here to show that problem_solution meets valid_matrix
  sorry

end NUMINAMATH_GPT_problem_correct_l435_43551


namespace NUMINAMATH_GPT_find_a_l435_43555

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

theorem find_a (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = 1 / 2 ∨ a = 1 / 3) := by
  sorry

end NUMINAMATH_GPT_find_a_l435_43555


namespace NUMINAMATH_GPT_bernoulli_inequality_l435_43554

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > -1) (hn : n > 0) : 
  (1 + x) ^ n ≥ 1 + n * x := 
sorry

end NUMINAMATH_GPT_bernoulli_inequality_l435_43554


namespace NUMINAMATH_GPT_sum_of_7_terms_arithmetic_seq_l435_43565

variable {α : Type*} [LinearOrderedField α]

def arithmetic_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_7_terms_arithmetic_seq (a : ℕ → α) (h_arith : arithmetic_seq a)
  (h_a4 : a 4 = 2) :
  (7 * (a 1 + a 7)) / 2 = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_7_terms_arithmetic_seq_l435_43565


namespace NUMINAMATH_GPT_weight_of_apples_l435_43511

-- Definitions based on conditions
def total_weight : ℕ := 10
def weight_orange : ℕ := 1
def weight_grape : ℕ := 3
def weight_strawberry : ℕ := 3

-- Prove that the weight of apples is 3 kilograms
theorem weight_of_apples : (total_weight - (weight_orange + weight_grape + weight_strawberry)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_apples_l435_43511


namespace NUMINAMATH_GPT_m_range_positive_real_number_l435_43544

theorem m_range_positive_real_number (m : ℝ) (x : ℝ) 
  (h : m * x - 1 = 2 * x) (h_pos : x > 0) : m > 2 :=
sorry

end NUMINAMATH_GPT_m_range_positive_real_number_l435_43544


namespace NUMINAMATH_GPT_initial_books_l435_43571

theorem initial_books (B : ℕ) (h : B + 5 = 7) : B = 2 :=
by sorry

end NUMINAMATH_GPT_initial_books_l435_43571


namespace NUMINAMATH_GPT_find_r_divisibility_l435_43529

theorem find_r_divisibility :
  ∃ r : ℝ, (10 * r ^ 2 - 4 * r - 26 = 0 ∧ (r = (19 / 10) ∨ r = (-3 / 2))) ∧ (r = -3 / 2) ∧ (10 * r ^ 3 - 5 * r ^ 2 - 52 * r + 60 = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_r_divisibility_l435_43529


namespace NUMINAMATH_GPT_arcade_spending_fraction_l435_43588

theorem arcade_spending_fraction (allowance remaining_after_arcade remaining_after_toystore: ℝ) (f: ℝ) : 
  allowance = 3.75 ∧
  remaining_after_arcade = (1 - f) * allowance ∧
  remaining_after_toystore = remaining_after_arcade - (1 / 3) * remaining_after_arcade ∧
  remaining_after_toystore = 1 →
  f = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_arcade_spending_fraction_l435_43588


namespace NUMINAMATH_GPT_find_a_l435_43510

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x * x - 4 <= 0) → (2 * x + a <= 0)) ↔ (a = -4) := by
  sorry

end NUMINAMATH_GPT_find_a_l435_43510


namespace NUMINAMATH_GPT_eccentricity_ellipse_l435_43526

variable (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variable (c : ℝ) (h3 : c = Real.sqrt (a ^ 2 - b ^ 2))
variable (h4 : b = c)
variable (ellipse_eq : ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1)

theorem eccentricity_ellipse :
  c / a = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_ellipse_l435_43526


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_condition_l435_43568

noncomputable def geometric_sequence_ratio (q : ℝ) : Prop :=
  q > 0

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  2 * a₃ = a₁ + 2 * a₂

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * q ^ n

theorem geometric_sequence_arithmetic_condition
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (q : ℝ)
  (hq : geometric_sequence_ratio q)
  (h_arith : arithmetic_sequence (a 0) (geometric_sequence a q 1) (geometric_sequence a q 2)) :
  (geometric_sequence a q 9 + geometric_sequence a q 10) / 
  (geometric_sequence a q 7 + geometric_sequence a q 8) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_condition_l435_43568


namespace NUMINAMATH_GPT_bowling_ball_weight_l435_43580

theorem bowling_ball_weight (b c : ℝ) (h1 : c = 36) (h2 : 5 * b = 4 * c) : b = 28.8 := by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l435_43580


namespace NUMINAMATH_GPT_squared_product_l435_43567

theorem squared_product (a b : ℝ) : (- (1 / 2) * a^2 * b)^2 = (1 / 4) * a^4 * b^2 := by 
  sorry

end NUMINAMATH_GPT_squared_product_l435_43567


namespace NUMINAMATH_GPT_train_speed_second_part_l435_43585

variables (x v : ℝ)

theorem train_speed_second_part
  (h1 : ∀ t1 : ℝ, t1 = x / 30)
  (h2 : ∀ t2 : ℝ, t2 = 2 * x / v)
  (h3 : ∀ t : ℝ, t = 3 * x / 22.5) :
  (x / 30) + (2 * x / v) = (3 * x / 22.5) → v = 20 :=
by
  intros h4
  sorry

end NUMINAMATH_GPT_train_speed_second_part_l435_43585


namespace NUMINAMATH_GPT_arithmetic_seq_slope_l435_43558

theorem arithmetic_seq_slope {a : ℕ → ℤ} (h : a 2 - a 4 = 2) : ∃ a1 : ℤ, ∀ n : ℕ, a n = -n + (a 1) + 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_seq_slope_l435_43558
