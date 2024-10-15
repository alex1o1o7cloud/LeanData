import Mathlib

namespace NUMINAMATH_GPT_intersection_P_Q_l2077_207786

def P : Set ℝ := {x | |x| > 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x | -2 ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l2077_207786


namespace NUMINAMATH_GPT_total_cost_sandwiches_sodas_l2077_207764

theorem total_cost_sandwiches_sodas (cost_per_sandwich cost_per_soda : ℝ) 
  (num_sandwiches num_sodas : ℕ) (discount_rate : ℝ) (total_items : ℕ) :
  cost_per_sandwich = 4 → 
  cost_per_soda = 3 → 
  num_sandwiches = 6 → 
  num_sodas = 7 → 
  discount_rate = 0.10 → 
  total_items = num_sandwiches + num_sodas → 
  total_items > 10 → 
  (num_sandwiches * cost_per_sandwich + num_sodas * cost_per_soda) * (1 - discount_rate) = 40.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_cost_sandwiches_sodas_l2077_207764


namespace NUMINAMATH_GPT_smallest_three_digit_divisible_by_3_and_6_l2077_207758

theorem smallest_three_digit_divisible_by_3_and_6 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999 ∧ n % 3 = 0 ∧ n % 6 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 3 = 0 ∧ m % 6 = 0 → n ≤ m) ∧ n = 102 := 
by {sorry}

end NUMINAMATH_GPT_smallest_three_digit_divisible_by_3_and_6_l2077_207758


namespace NUMINAMATH_GPT_largest_x_value_l2077_207730

theorem largest_x_value (x : ℝ) :
  (x ≠ 9) ∧ (x ≠ -4) ∧ ((x ^ 2 - x - 72) / (x - 9) = 5 / (x + 4)) → x = -3 :=
sorry

end NUMINAMATH_GPT_largest_x_value_l2077_207730


namespace NUMINAMATH_GPT_jake_has_one_more_balloon_than_allan_l2077_207727

def balloons_allan : ℕ := 6
def balloons_jake_initial : ℕ := 3
def balloons_jake_additional : ℕ := 4

theorem jake_has_one_more_balloon_than_allan :
  (balloons_jake_initial + balloons_jake_additional - balloons_allan) = 1 :=
by
  sorry

end NUMINAMATH_GPT_jake_has_one_more_balloon_than_allan_l2077_207727


namespace NUMINAMATH_GPT_smallest_non_lucky_multiple_of_8_correct_l2077_207719

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def smallest_non_lucky_multiple_of_8 := 16

theorem smallest_non_lucky_multiple_of_8_correct :
  smallest_non_lucky_multiple_of_8 = 16 ∧
  is_lucky smallest_non_lucky_multiple_of_8 = false :=
by
  sorry

end NUMINAMATH_GPT_smallest_non_lucky_multiple_of_8_correct_l2077_207719


namespace NUMINAMATH_GPT_balls_in_boxes_l2077_207755

theorem balls_in_boxes :
  ∃ (f : Fin 5 → Fin 3), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ b : Fin 3, ∃ i, f i = b) ∧
    f 0 ≠ f 1 :=
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l2077_207755


namespace NUMINAMATH_GPT_boat_distance_against_stream_in_one_hour_l2077_207743

-- Define the conditions
def speed_in_still_water : ℝ := 4 -- speed of the boat in still water (km/hr)
def downstream_distance_in_one_hour : ℝ := 6 -- distance traveled along the stream in one hour (km)

-- Define the function to compute the speed of the stream
def speed_of_stream (downstream_distance : ℝ) (boat_speed_still_water : ℝ) : ℝ :=
  downstream_distance - boat_speed_still_water

-- Define the effective speed against the stream
def effective_speed_against_stream (boat_speed_still_water : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed_still_water - stream_speed

-- Prove that the boat travels 2 km against the stream in one hour given the conditions
theorem boat_distance_against_stream_in_one_hour :
  effective_speed_against_stream speed_in_still_water (speed_of_stream downstream_distance_in_one_hour speed_in_still_water) * 1 = 2 := 
by
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_in_one_hour_l2077_207743


namespace NUMINAMATH_GPT_job_completion_time_l2077_207704

def time (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

noncomputable def start_time : ℕ := time 9 45
noncomputable def half_completion_time : ℕ := time 13 0  -- 1:00 PM in 24-hour time format

theorem job_completion_time :
  ∃ finish_time, finish_time = time 16 15 ∧
  (half_completion_time - start_time) * 2 = finish_time - start_time :=
by
  sorry

end NUMINAMATH_GPT_job_completion_time_l2077_207704


namespace NUMINAMATH_GPT_base_conversion_zero_l2077_207723

theorem base_conversion_zero (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 8 * A + B = 6 * B + A) : 8 * A + B = 0 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_zero_l2077_207723


namespace NUMINAMATH_GPT_find_x_l2077_207736

variable (A B x : ℝ)
variable (h1 : A > 0) (h2 : B > 0)
variable (h3 : A = (x / 100) * B)

theorem find_x : x = 100 * (A / B) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2077_207736


namespace NUMINAMATH_GPT_no_solution_exists_l2077_207773

theorem no_solution_exists (a b : ℤ) : ∃ c : ℤ, ∀ m n : ℤ, m^2 + a * m + b ≠ 2 * n^2 + 2 * n + c :=
by {
  -- Insert correct proof here
  sorry
}

end NUMINAMATH_GPT_no_solution_exists_l2077_207773


namespace NUMINAMATH_GPT_correct_equation_l2077_207732

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l2077_207732


namespace NUMINAMATH_GPT_symmetric_points_origin_l2077_207737

theorem symmetric_points_origin (a b : ℝ)
  (h1 : (-2 : ℝ) = -a)
  (h2 : (b : ℝ) = -3) : a - b = 5 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_l2077_207737


namespace NUMINAMATH_GPT_power_of_3_l2077_207733

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end NUMINAMATH_GPT_power_of_3_l2077_207733


namespace NUMINAMATH_GPT_part1_inequality_part2_inequality_l2077_207757

-- Problem Part 1
def f (x : ℝ) : ℝ := abs (x - 2) - abs (x + 1)

theorem part1_inequality (x : ℝ) : f x ≤ 1 ↔ 0 ≤ x :=
by sorry

-- Problem Part 2
def max_f_value : ℝ := 3
def a : ℝ := sorry  -- Define in context
def b : ℝ := sorry  -- Define in context
def c : ℝ := sorry  -- Define in context

-- Prove √a + √b + √c ≤ 3 given a + b + c = 3
theorem part2_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = max_f_value) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 :=
by sorry

end NUMINAMATH_GPT_part1_inequality_part2_inequality_l2077_207757


namespace NUMINAMATH_GPT_response_rate_percentage_50_l2077_207794

def questionnaire_response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℕ :=
  (responses_needed * 100) / questionnaires_mailed

theorem response_rate_percentage_50 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 600) : 
  questionnaire_response_rate_percentage responses_needed questionnaires_mailed = 50 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_response_rate_percentage_50_l2077_207794


namespace NUMINAMATH_GPT_parallelogram_area_15_l2077_207715

def point := (ℝ × ℝ)

def base_length (p1 p2 : point) : ℝ :=
  abs (p2.1 - p1.1)

def height_length (p3 p4 : point) : ℝ :=
  abs (p3.2 - p4.2)

def parallelogram_area (p1 p2 p3 p4 : point) : ℝ :=
  base_length p1 p2 * height_length p1 p3

theorem parallelogram_area_15 :
  parallelogram_area (0, 0) (3, 0) (1, 5) (4, 5) = 15 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_15_l2077_207715


namespace NUMINAMATH_GPT_find_prime_between_20_and_35_with_remainder_7_l2077_207754

theorem find_prime_between_20_and_35_with_remainder_7 : 
  ∃ p : ℕ, Nat.Prime p ∧ 20 ≤ p ∧ p ≤ 35 ∧ p % 11 = 7 ∧ p = 29 := 
by 
  sorry

end NUMINAMATH_GPT_find_prime_between_20_and_35_with_remainder_7_l2077_207754


namespace NUMINAMATH_GPT_tennis_balls_in_each_container_l2077_207735

theorem tennis_balls_in_each_container :
  let total_balls := 100
  let given_away := total_balls / 2
  let remaining := total_balls - given_away
  let containers := 5
  remaining / containers = 10 := 
by
  sorry

end NUMINAMATH_GPT_tennis_balls_in_each_container_l2077_207735


namespace NUMINAMATH_GPT_right_triangle_area_l2077_207717

theorem right_triangle_area (a b c p S : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a^2 + b^2 = c^2)
  (h4 : p = (a + b + c) / 2) (h5 : S = a * b / 2) :
  p * (p - c) = S ∧ (p - a) * (p - b) = S :=
sorry

end NUMINAMATH_GPT_right_triangle_area_l2077_207717


namespace NUMINAMATH_GPT_B_investment_amount_l2077_207746

-- Definitions based on given conditions
variable (A_investment : ℕ := 300) -- A's investment in dollars
variable (B_investment : ℕ)        -- B's investment in dollars
variable (A_time : ℕ := 12)        -- Time A's investment was in the business in months
variable (B_time : ℕ := 6)         -- Time B's investment was in the business in months
variable (profit : ℕ := 100)       -- Total profit in dollars
variable (A_share : ℕ := 75)       -- A's share of the profit in dollars

-- The mathematically equivalent proof problem to prove that B invested $200
theorem B_investment_amount (h : A_share * (A_investment * A_time + B_investment * B_time) / profit = A_investment * A_time) : 
  B_investment = 200 := by
  sorry

end NUMINAMATH_GPT_B_investment_amount_l2077_207746


namespace NUMINAMATH_GPT_number_of_pieces_l2077_207738

def pan_length : ℕ := 24
def pan_width : ℕ := 30
def brownie_length : ℕ := 3
def brownie_width : ℕ := 4

def area (length : ℕ) (width : ℕ) : ℕ := length * width

theorem number_of_pieces :
  (area pan_length pan_width) / (area brownie_length brownie_width) = 60 := by
  sorry

end NUMINAMATH_GPT_number_of_pieces_l2077_207738


namespace NUMINAMATH_GPT_rate_of_painting_per_sq_m_l2077_207742

def length_of_floor : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def ratio_of_length_to_breadth : ℝ := 3

theorem rate_of_painting_per_sq_m :
  ∃ (rate : ℝ), rate = 3 :=
by
  let B := length_of_floor / ratio_of_length_to_breadth
  let A := length_of_floor * B
  let rate := total_cost / A
  use rate
  sorry  -- Skipping proof as instructed

end NUMINAMATH_GPT_rate_of_painting_per_sq_m_l2077_207742


namespace NUMINAMATH_GPT_inequality_holds_l2077_207745

theorem inequality_holds (a b : ℝ) (h : a ≠ b) : a^4 + 6 * a^2 * b^2 + b^4 > 4 * a * b * (a^2 + b^2) := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2077_207745


namespace NUMINAMATH_GPT_age_of_b_l2077_207767

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 32) : b = 12 :=
by sorry

end NUMINAMATH_GPT_age_of_b_l2077_207767


namespace NUMINAMATH_GPT_new_mean_after_adding_constant_l2077_207728

theorem new_mean_after_adding_constant (S : ℝ) (average : ℝ) (n : ℕ) (a : ℝ) :
  n = 15 → average = 40 → a = 15 → S = n * average → (S + n * a) / n = 55 :=
by
  intros hn haverage ha hS
  sorry

end NUMINAMATH_GPT_new_mean_after_adding_constant_l2077_207728


namespace NUMINAMATH_GPT_total_people_going_to_museum_l2077_207788

def number_of_people_on_first_bus := 12
def number_of_people_on_second_bus := 2 * number_of_people_on_first_bus
def number_of_people_on_third_bus := number_of_people_on_second_bus - 6
def number_of_people_on_fourth_bus := number_of_people_on_first_bus + 9

theorem total_people_going_to_museum :
  number_of_people_on_first_bus + number_of_people_on_second_bus + number_of_people_on_third_bus + number_of_people_on_fourth_bus = 75 :=
by
  sorry

end NUMINAMATH_GPT_total_people_going_to_museum_l2077_207788


namespace NUMINAMATH_GPT_range_of_a_l2077_207789

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  a > 1

-- Translate the problem to a Lean 4 statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → a ∈ Set.Icc (-2 : ℝ) 1 ∪ Set.Ici 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2077_207789


namespace NUMINAMATH_GPT_batches_of_muffins_l2077_207769

-- Definitions of the costs and savings
def cost_blueberries_6oz : ℝ := 5
def cost_raspberries_12oz : ℝ := 3
def ounces_per_batch : ℝ := 12
def total_savings : ℝ := 22

-- The proof problem is to show the number of batches Bill plans to make
theorem batches_of_muffins : (total_savings / (2 * cost_blueberries_6oz - cost_raspberries_12oz)) = 3 := 
by 
  sorry  -- Proof goes here

end NUMINAMATH_GPT_batches_of_muffins_l2077_207769


namespace NUMINAMATH_GPT_combination_10_3_l2077_207710

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end NUMINAMATH_GPT_combination_10_3_l2077_207710


namespace NUMINAMATH_GPT_symmetric_angles_y_axis_l2077_207765

theorem symmetric_angles_y_axis (α β : ℝ) (k : ℤ)
  (h : ∃ k : ℤ, β = 2 * k * π + (π - α)) :
  α + β = (2 * k + 1) * π ∨ α = -β + (2 * k + 1) * π :=
by sorry

end NUMINAMATH_GPT_symmetric_angles_y_axis_l2077_207765


namespace NUMINAMATH_GPT_smoothie_combinations_l2077_207739

theorem smoothie_combinations :
  let flavors := 5
  let supplements := 8
  (flavors * Nat.choose supplements 3) = 280 :=
by
  sorry

end NUMINAMATH_GPT_smoothie_combinations_l2077_207739


namespace NUMINAMATH_GPT_max_daily_sales_l2077_207759

def f (t : ℕ) : ℝ := -2 * t + 200
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30
  else 45

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales : ∃ t, 1 ≤ t ∧ t ≤ 50 ∧ S t = 54600 := 
  sorry

end NUMINAMATH_GPT_max_daily_sales_l2077_207759


namespace NUMINAMATH_GPT_yura_finishes_problems_by_sept_12_l2077_207783

def total_problems := 91
def initial_date := 6 -- September 6
def problems_left_date := 8 -- September 8
def remaining_problems := 46
def decreasing_rate := 1

def problems_solved (z : ℕ) (day : ℕ) : ℕ :=
if day = 6 then z + 1 else if day = 7 then z else if day = 8 then z - 1 else z - (day - 7)

theorem yura_finishes_problems_by_sept_12 :
  ∃ z : ℕ, (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 = total_problems - remaining_problems) ∧
           (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 + problems_solved z 9 + problems_solved z 10 + problems_solved z 11 + problems_solved z 12 = total_problems) :=
sorry

end NUMINAMATH_GPT_yura_finishes_problems_by_sept_12_l2077_207783


namespace NUMINAMATH_GPT_probability_x_lt_y_l2077_207750

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end NUMINAMATH_GPT_probability_x_lt_y_l2077_207750


namespace NUMINAMATH_GPT_mirasol_balance_l2077_207762

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end NUMINAMATH_GPT_mirasol_balance_l2077_207762


namespace NUMINAMATH_GPT_hotel_cost_l2077_207702

/--
Let the total cost of the hotel be denoted as x dollars.
Initially, the cost for each of the original four colleagues is x / 4.
After three more colleagues joined, the cost per person becomes x / 7.
Given that the amount paid by each of the original four decreased by 15,
prove that the total cost of the hotel is 140 dollars.
-/
theorem hotel_cost (x : ℕ) (h : x / 4 - 15 = x / 7) : x = 140 := 
by
  sorry

end NUMINAMATH_GPT_hotel_cost_l2077_207702


namespace NUMINAMATH_GPT_randy_piggy_bank_balance_l2077_207720

def initial_amount : ℕ := 200
def store_trip_cost : ℕ := 2
def trips_per_month : ℕ := 4
def extra_cost_trip : ℕ := 1
def extra_trip_interval : ℕ := 3
def months_in_year : ℕ := 12
def weekly_income : ℕ := 15
def internet_bill_per_month : ℕ := 20
def birthday_gift : ℕ := 100
def weeks_in_year : ℕ := 52

-- To be proved
theorem randy_piggy_bank_balance : 
  initial_amount 
  + (weekly_income * weeks_in_year) 
  + birthday_gift 
  - ((store_trip_cost * trips_per_month * months_in_year)
  + (months_in_year / extra_trip_interval) * extra_cost_trip
  + (internet_bill_per_month * months_in_year))
  = 740 :=
by
  sorry

end NUMINAMATH_GPT_randy_piggy_bank_balance_l2077_207720


namespace NUMINAMATH_GPT_ball_reaches_top_left_pocket_l2077_207766

-- Definitions based on the given problem
def table_width : ℕ := 26
def table_height : ℕ := 1965
def pocket_start : (ℕ × ℕ) := (0, 0)
def pocket_end : (ℕ × ℕ) := (0, table_height)
def angle_of_release : ℝ := 45

-- The goal is to prove that the ball will reach the top left pocket after reflections
theorem ball_reaches_top_left_pocket :
  ∃ reflections : ℕ, (reflections * table_width, reflections * table_height) = pocket_end :=
sorry

end NUMINAMATH_GPT_ball_reaches_top_left_pocket_l2077_207766


namespace NUMINAMATH_GPT_intersection_complement_eq_l2077_207749

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def P : Finset ℕ := {1, 2, 3, 4}
def Q : Finset ℕ := {3, 4, 5}
def U_complement_Q : Finset ℕ := U \ Q

theorem intersection_complement_eq : P ∩ U_complement_Q = {1, 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_complement_eq_l2077_207749


namespace NUMINAMATH_GPT_min_value_ge_9_l2077_207791

noncomputable def minValue (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : ℝ :=
  1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2

theorem min_value_ge_9 (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : minValue θ h ≥ 9 := 
  sorry

end NUMINAMATH_GPT_min_value_ge_9_l2077_207791


namespace NUMINAMATH_GPT_enrollment_inversely_proportional_l2077_207775

theorem enrollment_inversely_proportional :
  ∃ k : ℝ, (40 * 2000 = k) → (s * 2500 = k) → s = 32 :=
by
  sorry

end NUMINAMATH_GPT_enrollment_inversely_proportional_l2077_207775


namespace NUMINAMATH_GPT_exists_c_gt_zero_l2077_207796

theorem exists_c_gt_zero (a b : ℕ) (h_a_square_free : ¬ ∃ (k : ℕ), k^2 ∣ a)
    (h_b_square_free : ¬ ∃ (k : ℕ), k^2 ∣ b) (h_a_b_distinct : a ≠ b) :
    ∃ c > 0, ∀ n : ℕ, n > 0 →
    |(n * Real.sqrt a % 1) - (n * Real.sqrt b % 1)| > c / n^3 := sorry

end NUMINAMATH_GPT_exists_c_gt_zero_l2077_207796


namespace NUMINAMATH_GPT_marked_price_percentage_l2077_207706

theorem marked_price_percentage
  (CP MP SP : ℝ)
  (h_profit : SP = 1.08 * CP)
  (h_discount : SP = 0.8307692307692308 * MP) :
  MP = CP * 1.3 :=
by sorry

end NUMINAMATH_GPT_marked_price_percentage_l2077_207706


namespace NUMINAMATH_GPT_initial_animal_types_l2077_207724

theorem initial_animal_types (x : ℕ) (h1 : 6 * (x + 4) = 54) : x = 5 := 
sorry

end NUMINAMATH_GPT_initial_animal_types_l2077_207724


namespace NUMINAMATH_GPT_simplify_expression_l2077_207777

theorem simplify_expression (x : ℝ) 
  (h1 : x^2 - 4*x + 3 = (x-3)*(x-1))
  (h2 : x^2 - 6*x + 9 = (x-3)^2)
  (h3 : x^2 - 6*x + 8 = (x-2)*(x-4))
  (h4 : x^2 - 8*x + 15 = (x-3)*(x-5)) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = (x-1)*(x-5) / ((x-2)*(x-4)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2077_207777


namespace NUMINAMATH_GPT_f_is_neither_odd_nor_even_l2077_207751

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^2 + 6 * x

-- Defining the concept of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Defining the concept of an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

-- The goal is to prove that f is neither odd nor even
theorem f_is_neither_odd_nor_even : ¬ is_odd f ∧ ¬ is_even f :=
by
  sorry

end NUMINAMATH_GPT_f_is_neither_odd_nor_even_l2077_207751


namespace NUMINAMATH_GPT_right_triangle_perimeter_5_shortest_altitude_1_l2077_207721

-- Definition of a right-angled triangle's sides with given perimeter and altitude
def right_angled_triangle (a b c : ℚ) : Prop :=
a^2 + b^2 = c^2 ∧ a + b + c = 5 ∧ a * b = c

-- Statement of the theorem to prove the side lengths of the triangle
theorem right_triangle_perimeter_5_shortest_altitude_1 :
  ∃ (a b c : ℚ), right_angled_triangle a b c ∧ (a = 5 / 3 ∧ b = 5 / 4 ∧ c = 25 / 12) ∨ (a = 5 / 4 ∧ b = 5 / 3 ∧ c = 25 / 12) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_5_shortest_altitude_1_l2077_207721


namespace NUMINAMATH_GPT_mary_principal_amount_l2077_207734

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mary_principal_amount_l2077_207734


namespace NUMINAMATH_GPT_triangle_inequality_sum_2_l2077_207793

theorem triangle_inequality_sum_2 (a b c : ℝ) (h_triangle : a + b + c = 2) (h_side_ineq : a + c > b ∧ a + b > c ∧ b + c > a):
  1 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_sum_2_l2077_207793


namespace NUMINAMATH_GPT_leaves_blew_away_l2077_207747

theorem leaves_blew_away (initial_leaves : ℕ) (leaves_left : ℕ) (blew_away : ℕ) 
  (h1 : initial_leaves = 356) (h2 : leaves_left = 112) (h3 : blew_away = initial_leaves - leaves_left) :
  blew_away = 244 :=
by
  sorry

end NUMINAMATH_GPT_leaves_blew_away_l2077_207747


namespace NUMINAMATH_GPT_false_statements_count_is_3_l2077_207776

-- Define the statements
def statement1_false : Prop := ¬ (1 ≠ 1)     -- Not exactly one statement is false
def statement2_false : Prop := ¬ (2 ≠ 2)     -- Not exactly two statements are false
def statement3_false : Prop := ¬ (3 ≠ 3)     -- Not exactly three statements are false
def statement4_false : Prop := ¬ (4 ≠ 4)     -- Not exactly four statements are false
def statement5_false : Prop := ¬ (5 ≠ 5)     -- Not all statements are false

-- Prove that the number of false statements is 3
theorem false_statements_count_is_3 :
  (statement1_false → statement2_false →
  statement3_false → statement4_false →
  statement5_false → (3 = 3)) := by
  sorry

end NUMINAMATH_GPT_false_statements_count_is_3_l2077_207776


namespace NUMINAMATH_GPT_find_a_b_and_m_range_l2077_207707

-- Definitions and initial conditions
def f (x : ℝ) (a b m : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + m
def f_prime (x : ℝ) (a b : ℝ) : ℝ := 6*x^2 + 2*a*x + b

-- Problem statement
theorem find_a_b_and_m_range (a b m : ℝ) :
  (∀ x, f_prime x a b = 6 * (x + 0.5)^2 - k) →
  f_prime 1 a b = 0 →
  a = 3 ∧ b = -12 ∧ -20 < m ∧ m < 7 :=
sorry

end NUMINAMATH_GPT_find_a_b_and_m_range_l2077_207707


namespace NUMINAMATH_GPT_average_snack_sales_per_ticket_l2077_207712

theorem average_snack_sales_per_ticket :
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  (total_sales / movie_tickets = 2.79) :=
by
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  show total_sales / movie_tickets = 2.79
  sorry

end NUMINAMATH_GPT_average_snack_sales_per_ticket_l2077_207712


namespace NUMINAMATH_GPT_sequence_fraction_l2077_207713

-- Definitions for arithmetic and geometric sequences
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def isGeometricSeq (a b c : ℝ) :=
  b^2 = a * c

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}

-- a is an arithmetic sequence with common difference d ≠ 0
axiom h1 : isArithmeticSeq a d
axiom h2 : d ≠ 0

-- a_2, a_3, a_9 form a geometric sequence
axiom h3 : isGeometricSeq (a 2) (a 3) (a 9)

-- Goal: prove the value of the given expression
theorem sequence_fraction {a : ℕ → ℝ} {d : ℝ} (h1 : isArithmeticSeq a d) (h2 : d ≠ 0) (h3 : isGeometricSeq (a 2) (a 3) (a 9)) :
  (a 2 + a 3 + a 4) / (a 4 + a 5 + a 6) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_sequence_fraction_l2077_207713


namespace NUMINAMATH_GPT_algebraic_expression_value_l2077_207744

theorem algebraic_expression_value (m n : ℤ) (h : n - m = 2):
  (m^2 - n^2) / m * (2 * m / (m + n)) = -4 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2077_207744


namespace NUMINAMATH_GPT_third_height_of_triangle_l2077_207740

theorem third_height_of_triangle 
  (a b c ha hb hc : ℝ)
  (h_abc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_heights : ∃ (h1 h2 h3 : ℕ), h1 = 3 ∧ h2 = 10 ∧ h3 ≠ h1 ∧ h3 ≠ h2) :
  ∃ (h3 : ℕ), h3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_third_height_of_triangle_l2077_207740


namespace NUMINAMATH_GPT_find_nm_2023_l2077_207708

theorem find_nm_2023 (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : (n + m) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_GPT_find_nm_2023_l2077_207708


namespace NUMINAMATH_GPT_find_const_functions_l2077_207787

theorem find_const_functions
  (f g : ℝ → ℝ)
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → f (x^2 + y^2) = g (x * y)) :
  ∃ c : ℝ, (∀ x, 0 < x → f x = c) ∧ (∀ x, 0 < x → g x = c) :=
sorry

end NUMINAMATH_GPT_find_const_functions_l2077_207787


namespace NUMINAMATH_GPT_mapping_f_correct_l2077_207729

theorem mapping_f_correct (a1 a2 a3 a4 b1 b2 b3 b4 : ℤ) :
  (∀ (x : ℤ), x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4 = (x + 1)^4 + b1 * (x + 1)^3 + b2 * (x + 1)^2 + b3 * (x + 1) + b4) →
  a1 = 4 → a2 = 3 → a3 = 2 → a4 = 1 →
  b1 = 0 → b1 + b2 + b3 + b4 = 0 →
  (b1, b2, b3, b4) = (0, -3, 4, -1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_mapping_f_correct_l2077_207729


namespace NUMINAMATH_GPT_solve_equation_l2077_207703

noncomputable def equation_solution (x : ℝ) : Prop :=
  (3 / x = 2 / (x - 2)) ∧ x ≠ 0 ∧ x - 2 ≠ 0

theorem solve_equation : (equation_solution 6) :=
  by
    sorry

end NUMINAMATH_GPT_solve_equation_l2077_207703


namespace NUMINAMATH_GPT_quadrilateral_smallest_angle_l2077_207772

theorem quadrilateral_smallest_angle
  (a d : ℝ)
  (h1 : a + (a + 2 * d) = 160)
  (h2 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) :
  a = 60 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_smallest_angle_l2077_207772


namespace NUMINAMATH_GPT_largest_whole_number_less_than_100_l2077_207748

theorem largest_whole_number_less_than_100 (x : ℕ) (h1 : 7 * x < 100) (h_max : ∀ y : ℕ, 7 * y < 100 → y ≤ x) :
  x = 14 := 
sorry

end NUMINAMATH_GPT_largest_whole_number_less_than_100_l2077_207748


namespace NUMINAMATH_GPT_candy_bar_cost_l2077_207761

def cost_soft_drink : ℕ := 2
def num_candy_bars : ℕ := 5
def total_spent : ℕ := 27
def cost_per_candy_bar (C : ℕ) : Prop := cost_soft_drink + num_candy_bars * C = total_spent

-- The theorem we want to prove
theorem candy_bar_cost (C : ℕ) (h : cost_per_candy_bar C) : C = 5 :=
by sorry

end NUMINAMATH_GPT_candy_bar_cost_l2077_207761


namespace NUMINAMATH_GPT_gray_region_area_l2077_207722

theorem gray_region_area (r R : ℝ) (hR : R = 3 * r) (h_diff : R - r = 3) :
  π * (R^2 - r^2) = 18 * π :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_gray_region_area_l2077_207722


namespace NUMINAMATH_GPT_point_in_second_quadrant_coordinates_l2077_207798

variable (x y : ℝ)
variable (P : ℝ × ℝ)
variable (h1 : P.1 = x)
variable (h2 : P.2 = y)

def isInSecondQuadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distanceToXAxis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distanceToYAxis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem point_in_second_quadrant_coordinates (h1 : isInSecondQuadrant P)
    (h2 : distanceToXAxis P = 2)
    (h3 : distanceToYAxis P = 1) :
    P = (-1, 2) :=
by 
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_coordinates_l2077_207798


namespace NUMINAMATH_GPT_find_alpha_l2077_207768

noncomputable def angle_in_interval (α : ℝ) : Prop :=
  370 < α ∧ α < 520 

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = 1 / 2) (h_interval: angle_in_interval α) : α = 420 :=
sorry

end NUMINAMATH_GPT_find_alpha_l2077_207768


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2077_207782

theorem hyperbola_eccentricity
    (a b e : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (h_hyperbola : ∀ x y, x ^ 2 / a^2 - y^2 / b^2 = 1)
    (h_circle : ∀ x y, (x - 2) ^ 2 + y ^ 2 = 4)
    (h_chord_length : ∀ x y, (x ^ 2 + y ^ 2)^(1/2) = 2) :
    e = 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2077_207782


namespace NUMINAMATH_GPT_find_lines_through_p_and_intersecting_circle_l2077_207792

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def passes_through (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = l P.1

noncomputable def chord_length (c p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

theorem find_lines_through_p_and_intersecting_circle :
  ∃ l : ℝ → ℝ, (passes_through l (-2, 3)) ∧
  (∃ p1 p2 : ℝ × ℝ, trajectory_equation p1.1 p1.2 ∧ trajectory_equation p2.1 p2.2 ∧
  chord_length (1, 2) p1 p2 = 8^2) :=
by
  sorry

end NUMINAMATH_GPT_find_lines_through_p_and_intersecting_circle_l2077_207792


namespace NUMINAMATH_GPT_only_A_forms_triangle_l2077_207799

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_A_forms_triangle :
  (triangle_inequality 5 6 10) ∧ ¬(triangle_inequality 5 2 9) ∧ ¬(triangle_inequality 5 7 12) ∧ ¬(triangle_inequality 3 4 8) :=
by
  sorry

end NUMINAMATH_GPT_only_A_forms_triangle_l2077_207799


namespace NUMINAMATH_GPT_last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l2077_207778

-- Definition of function to calculate the last digit of a number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Proof statements
theorem last_digit_11_power_11 : last_digit (11 ^ 11) = 1 := sorry

theorem last_digit_9_power_9 : last_digit (9 ^ 9) = 9 := sorry

theorem last_digit_9219_power_9219 : last_digit (9219 ^ 9219) = 9 := sorry

theorem last_digit_2014_power_2014 : last_digit (2014 ^ 2014) = 6 := sorry

end NUMINAMATH_GPT_last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l2077_207778


namespace NUMINAMATH_GPT_solve_equation_l2077_207763

theorem solve_equation (x y : ℝ) (k : ℤ) :
  x^2 - 2 * x * Real.sin (x * y) + 1 = 0 ↔ (x = 1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) ∨ (x = -1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) :=
by
  -- Logical content will be filled here, sorry is used because proof steps are not required.
  sorry

end NUMINAMATH_GPT_solve_equation_l2077_207763


namespace NUMINAMATH_GPT_num_common_tangents_l2077_207700

-- Define the first circle
def circle1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 4
-- Define the second circle
def circle2 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 9

-- Prove that the number of common tangent lines between the given circles is 2
theorem num_common_tangents : ∃ (n : ℕ), n = 2 ∧
  -- The circles do not intersect nor are they internally tangent
  (∀ (x y : ℝ), ¬(circle1 x y ∧ circle2 x y) ∧ 
  -- There exist exactly n common tangent lines
  ∃ (C : ℕ), C = n) :=
sorry

end NUMINAMATH_GPT_num_common_tangents_l2077_207700


namespace NUMINAMATH_GPT_Events_B_and_C_mutex_l2077_207741

-- Definitions of events based on scores
def EventA (score : ℕ) := score ≥ 1 ∧ score ≤ 10
def EventB (score : ℕ) := score > 5 ∧ score ≤ 10
def EventC (score : ℕ) := score > 1 ∧ score < 6
def EventD (score : ℕ) := score > 0 ∧ score < 6

-- Mutually exclusive definition:
def mutually_exclusive (P Q : ℕ → Prop) := ∀ (x : ℕ), ¬ (P x ∧ Q x)

-- The proof statement:
theorem Events_B_and_C_mutex : mutually_exclusive EventB EventC :=
by
  sorry

end NUMINAMATH_GPT_Events_B_and_C_mutex_l2077_207741


namespace NUMINAMATH_GPT_find_common_difference_l2077_207771

def arithmetic_sequence (S_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)

theorem find_common_difference (S_n : ℕ → ℝ) (d : ℝ) (h : ∀n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)) 
    (h_condition : S_n 3 / 3 - S_n 2 / 2 = 1) :
  d = 2 :=
sorry

end NUMINAMATH_GPT_find_common_difference_l2077_207771


namespace NUMINAMATH_GPT_paul_money_duration_l2077_207753

theorem paul_money_duration (earn1 earn2 spend : ℕ) (h1 : earn1 = 3) (h2 : earn2 = 3) (h_spend : spend = 3) : 
  (earn1 + earn2) / spend = 2 :=
by
  sorry

end NUMINAMATH_GPT_paul_money_duration_l2077_207753


namespace NUMINAMATH_GPT_simultaneous_equations_solution_l2077_207718

theorem simultaneous_equations_solution (x y : ℚ) :
  3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1 ↔ 
  (x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5) :=
by
  sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_l2077_207718


namespace NUMINAMATH_GPT_proof_problem_l2077_207756

variables {a b c d : ℝ} (h1 : a ≠ -2) (h2 : b ≠ -2) (h3 : c ≠ -2) (h4 : d ≠ -2)
variable (ω : ℂ) (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
variable (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω)

theorem proof_problem : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 :=
sorry

end NUMINAMATH_GPT_proof_problem_l2077_207756


namespace NUMINAMATH_GPT_orig_polygon_sides_l2077_207760

theorem orig_polygon_sides (n : ℕ) (S : ℕ) :
  (n - 1 > 2) ∧ S = 1620 → (n = 10 ∨ n = 11 ∨ n = 12) :=
by
  sorry

end NUMINAMATH_GPT_orig_polygon_sides_l2077_207760


namespace NUMINAMATH_GPT_find_quadratic_eq_with_given_roots_l2077_207701

theorem find_quadratic_eq_with_given_roots (A z x1 x2 : ℝ) 
  (h1 : A * z * x1^2 + x1 * x1 + x2 = 0) 
  (h2 : A * z * x2^2 + x1 * x2 + x2 = 0) : 
  (A * z * x^2 + x1 * x - x2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_quadratic_eq_with_given_roots_l2077_207701


namespace NUMINAMATH_GPT_sum_S6_l2077_207780

variable (a_n : ℕ → ℚ)
variable (d : ℚ)
variable (S : ℕ → ℚ)
variable (a1 : ℚ)

/-- Define arithmetic sequence with common difference -/
def arithmetic_seq (n : ℕ) := a1 + n * d

/-- Define the sum of the first n terms of the sequence -/
def sum_of_arith_seq (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

/-- The given conditions -/
axiom h1 : d = 5
axiom h2 : (a_n 1 = a1) ∧ (a_n 2 = a1 + d) ∧ (a_n 5 = a1 + 4 * d)
axiom geom_seq : (a1 + d)^2 = a1 * (a1 + 4 * d)

theorem sum_S6 : S 6 = 90 := by
  sorry

end NUMINAMATH_GPT_sum_S6_l2077_207780


namespace NUMINAMATH_GPT_find_constants_l2077_207774

theorem find_constants (c d : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
     (r^3 + c*r^2 + 17*r + 10 = 0) ∧ (s^3 + c*s^2 + 17*s + 10 = 0) ∧
     (r^3 + d*r^2 + 22*r + 14 = 0) ∧ (s^3 + d*s^2 + 22*s + 14 = 0)) →
  (c = 8 ∧ d = 9) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l2077_207774


namespace NUMINAMATH_GPT_translate_statement_to_inequality_l2077_207752

theorem translate_statement_to_inequality (y : ℝ) : (1/2) * y + 5 > 0 ↔ True := 
sorry

end NUMINAMATH_GPT_translate_statement_to_inequality_l2077_207752


namespace NUMINAMATH_GPT_sales_tax_difference_l2077_207779

/-- The difference in sales tax calculation given the changes in rate. -/
theorem sales_tax_difference 
  (market_price : ℝ := 9000) 
  (original_rate : ℝ := 0.035) 
  (new_rate : ℝ := 0.0333) 
  (difference : ℝ := 15.3) :
  market_price * original_rate - market_price * new_rate = difference :=
by
  /- The proof is omitted as per the instructions. -/
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l2077_207779


namespace NUMINAMATH_GPT_profit_percentage_is_4_l2077_207790

-- Define the cost price and selling price
def cost_price : Nat := 600
def selling_price : Nat := 624

-- Calculate profit in dollars
def profit_dollars : Nat := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage : Nat := (profit_dollars * 100) / cost_price

-- Prove that the profit percentage is 4%
theorem profit_percentage_is_4 : profit_percentage = 4 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_4_l2077_207790


namespace NUMINAMATH_GPT_inequality_holds_l2077_207785

theorem inequality_holds (c : ℝ) (X Y : ℝ) (h1 : X^2 - c * X - c = 0) (h2 : Y^2 - c * Y - c = 0) :
    X^3 + Y^3 + (X * Y)^3 ≥ 0 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2077_207785


namespace NUMINAMATH_GPT_remainder_problem_l2077_207714

theorem remainder_problem {x y z : ℤ} (h1 : x % 102 = 56) (h2 : y % 154 = 79) (h3 : z % 297 = 183) :
  x % 19 = 18 ∧ y % 22 = 13 ∧ z % 33 = 18 :=
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l2077_207714


namespace NUMINAMATH_GPT_triangle_area_difference_l2077_207795

-- Definitions per conditions
def right_angle (A B C : Type) (angle_EAB : Prop) : Prop := angle_EAB
def angle_ABC_eq_30 (A B C : Type) (angle_ABC : ℝ) : Prop := angle_ABC = 30
def length_AB_eq_5 (A B : Type) (AB : ℝ) : Prop := AB = 5
def length_BC_eq_7 (B C : Type) (BC : ℝ) : Prop := BC = 7
def length_AE_eq_10 (A E : Type) (AE : ℝ) : Prop := AE = 10
def lines_intersect_at_D (A B C E D : Type) (intersects : Prop) : Prop := intersects

-- Main theorem statement
theorem triangle_area_difference
  (A B C E D : Type)
  (angle_EAB : Prop)
  (right_EAB : right_angle A E B angle_EAB)
  (angle_ABC : ℝ)
  (angle_ABC_is_30 : angle_ABC_eq_30 A B C angle_ABC)
  (AB : ℝ)
  (AB_is_5 : length_AB_eq_5 A B AB)
  (BC : ℝ)
  (BC_is_7 : length_BC_eq_7 B C BC)
  (AE : ℝ)
  (AE_is_10 : length_AE_eq_10 A E AE)
  (intersects : Prop)
  (intersects_at_D : lines_intersect_at_D A B C E D intersects) :
  (area_ADE - area_BDC) = 16.25 := sorry

end NUMINAMATH_GPT_triangle_area_difference_l2077_207795


namespace NUMINAMATH_GPT_find_x_of_equation_l2077_207731

theorem find_x_of_equation (x : ℝ) (hx : x ≠ 0) : (7 * x)^4 = (14 * x)^3 → x = 8 / 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_of_equation_l2077_207731


namespace NUMINAMATH_GPT_squares_ratio_l2077_207784

noncomputable def inscribed_squares_ratio :=
  let x := 60 / 17
  let y := 780 / 169
  (x / y : ℚ)

theorem squares_ratio (x y : ℚ) (h₁ : x = 60 / 17) (h₂ : y = 780 / 169) :
  x / y = 169 / 220 := by
  rw [h₁, h₂]
  -- Here we would perform calculations to show equality, omitted for brevity.
  sorry

end NUMINAMATH_GPT_squares_ratio_l2077_207784


namespace NUMINAMATH_GPT_frequency_of_middle_group_l2077_207711

theorem frequency_of_middle_group
    (num_rectangles : ℕ)
    (middle_area : ℝ)
    (other_areas_sum : ℝ)
    (sample_size : ℕ)
    (total_area_norm : ℝ)
    (h1 : num_rectangles = 11)
    (h2 : middle_area = other_areas_sum)
    (h3 : sample_size = 160)
    (h4 : middle_area + other_areas_sum = total_area_norm)
    (h5 : total_area_norm = 1):
    160 * (middle_area / total_area_norm) = 80 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_middle_group_l2077_207711


namespace NUMINAMATH_GPT_find_number_l2077_207770

noncomputable def solve_N (x : ℝ) (N : ℝ) : Prop :=
  ((N / x) / (3.6 * 0.2) = 2)

theorem find_number (x : ℝ) (N : ℝ) (h1 : x = 12) (h2 : solve_N x N) : N = 17.28 :=
  by
  sorry

end NUMINAMATH_GPT_find_number_l2077_207770


namespace NUMINAMATH_GPT_time_to_cross_bridge_l2077_207781

-- Defining the given conditions
def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 140

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

-- Calculating the speed in m/s
def speed_of_train_ms : ℚ := kmh_to_ms speed_of_train_kmh

-- Calculating total distance to be covered
def total_distance : ℕ := length_of_train + length_of_bridge

-- Expected time to cross the bridge
def expected_time : ℚ := total_distance / speed_of_train_ms

-- The proof statement
theorem time_to_cross_bridge :
  expected_time = 12.5 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l2077_207781


namespace NUMINAMATH_GPT_total_amount_is_33_l2077_207726

variable (n : ℕ) (c t : ℝ)

def total_amount_paid (n : ℕ) (c t : ℝ) : ℝ :=
  let cost_before_tax := n * c
  let tax := t * cost_before_tax
  cost_before_tax + tax

theorem total_amount_is_33
  (h1 : n = 5)
  (h2 : c = 6)
  (h3 : t = 0.10) :
  total_amount_paid n c t = 33 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_total_amount_is_33_l2077_207726


namespace NUMINAMATH_GPT_sets_produced_and_sold_is_500_l2077_207725

-- Define the initial conditions as constants
def initial_outlay : ℕ := 10000
def manufacturing_cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def total_profit : ℕ := 5000

-- The proof goal
theorem sets_produced_and_sold_is_500 (x : ℕ) : 
  (total_profit = selling_price_per_set * x - (initial_outlay + manufacturing_cost_per_set * x)) → 
  x = 500 :=
by 
  sorry

end NUMINAMATH_GPT_sets_produced_and_sold_is_500_l2077_207725


namespace NUMINAMATH_GPT_largest_common_divisor_l2077_207716

theorem largest_common_divisor (d h m s : ℕ) : 
  40 ∣ (1000000 * d + 10000 * h + 100 * m + s - (86400 * d + 3600 * h + 60 * m + s)) :=
by
  sorry

end NUMINAMATH_GPT_largest_common_divisor_l2077_207716


namespace NUMINAMATH_GPT_remainder_modulus_9_l2077_207797

theorem remainder_modulus_9 : (9 * 7^18 + 2^18) % 9 = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_modulus_9_l2077_207797


namespace NUMINAMATH_GPT_find_two_digit_numbers_l2077_207705

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_two_digit_numbers_l2077_207705


namespace NUMINAMATH_GPT_number_of_girls_l2077_207709

theorem number_of_girls
  (B G : ℕ)
  (h1 : B = (8 * G) / 5)
  (h2 : B + G = 351) :
  G = 135 :=
sorry

end NUMINAMATH_GPT_number_of_girls_l2077_207709
