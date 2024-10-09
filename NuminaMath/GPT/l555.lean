import Mathlib

namespace complement_of_A_in_I_l555_55595

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6, 7}
def C_I_A : Set ℕ := {1, 3, 5}

theorem complement_of_A_in_I :
  (I \ A) = C_I_A := by
  sorry

end complement_of_A_in_I_l555_55595


namespace base_2_representation_of_123_l555_55569

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l555_55569


namespace wall_clock_time_at_car_5PM_l555_55590

-- Define the initial known conditions
def initial_time : ℕ := 7 -- 7:00 AM
def wall_time_at_10AM : ℕ := 10 -- 10:00 AM
def car_time_at_10AM : ℕ := 11 -- 11:00 AM
def car_time_at_5PM : ℕ := 17 -- 5:00 PM = 17:00 in 24-hour format

-- Define the calculations for the rate of the car clock
def rate_of_car_clock : ℚ := (car_time_at_10AM - initial_time : ℚ) / (wall_time_at_10AM - initial_time : ℚ) -- rate = 4/3

-- Prove the actual time according to the wall clock when the car clock shows 5:00 PM
theorem wall_clock_time_at_car_5PM :
  let elapsed_real_time := (car_time_at_5PM - car_time_at_10AM) * (3 : ℚ) / (4 : ℚ)
  let actual_time := wall_time_at_10AM + elapsed_real_time
  (actual_time : ℚ) = 15 + (15 / 60 : ℚ) := -- 3:15 PM as 15.25 in 24-hour time
by
  sorry

end wall_clock_time_at_car_5PM_l555_55590


namespace move_point_right_3_units_l555_55537

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end move_point_right_3_units_l555_55537


namespace fill_tank_time_l555_55579

variable (A_rate := 1/3)
variable (B_rate := 1/4)
variable (C_rate := -1/4)

def combined_rate := A_rate + B_rate + C_rate

theorem fill_tank_time (hA : A_rate = 1/3) (hB : B_rate = 1/4) (hC : C_rate = -1/4) : (1 / combined_rate) = 3 := by
  sorry

end fill_tank_time_l555_55579


namespace arithmetic_sequence_sum_l555_55517

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ)
variable (h_arith : is_arithmetic_sequence a)
variable (h_sum : a 2 + a 3 + a 10 + a 11 = 48)

-- Goal
theorem arithmetic_sequence_sum : a 6 + a 7 = 24 :=
sorry

end arithmetic_sequence_sum_l555_55517


namespace Oliver_monster_club_cards_l555_55541

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end Oliver_monster_club_cards_l555_55541


namespace sum_of_solutions_l555_55536

theorem sum_of_solutions (x : ℝ) : (∃ x₁ x₂ : ℝ, (x - 4)^2 = 16 ∧ x = x₁ ∨ x = x₂ ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_l555_55536


namespace y_intercept_of_parallel_line_l555_55544

-- Define the conditions for the problem
def line_parallel (m1 m2 : ℝ) : Prop := 
  m1 = m2

def point_on_line (m : ℝ) (b x1 y1 : ℝ) : Prop := 
  y1 = m * x1 + b

-- Define the main problem statement
theorem y_intercept_of_parallel_line (m b1 b2 x1 y1 : ℝ) 
  (h1 : line_parallel m 3) 
  (h2 : point_on_line m b1 x1 y1) 
  (h3 : x1 = 1) 
  (h4 : y1 = 2) 
  : b1 = -1 :=
sorry

end y_intercept_of_parallel_line_l555_55544


namespace car_speed_first_hour_l555_55577

theorem car_speed_first_hour (x : ℕ) :
  (x + 60) / 2 = 75 → x = 90 :=
by
  -- To complete the proof in Lean, we would need to solve the equation,
  -- reversing the steps provided in the solution. 
  -- But as per instructions, we don't need the proof, hence we put sorry.
  sorry

end car_speed_first_hour_l555_55577


namespace cindy_correct_answer_l555_55559

theorem cindy_correct_answer (x : ℝ) (h₀ : (x - 12) / 4 = 32) : (x - 7) / 5 = 27 :=
by
  sorry

end cindy_correct_answer_l555_55559


namespace find_insect_stickers_l555_55587

noncomputable def flower_stickers : ℝ := 15
noncomputable def animal_stickers : ℝ := 2 * flower_stickers - 3.5
noncomputable def space_stickers : ℝ := 1.5 * flower_stickers + 5.5
noncomputable def total_stickers : ℝ := 70
noncomputable def insect_stickers : ℝ := total_stickers - (animal_stickers + space_stickers)

theorem find_insect_stickers : insect_stickers = 15.5 := by
  sorry

end find_insect_stickers_l555_55587


namespace number_20_l555_55557

def Jo (n : ℕ) : ℕ :=
  1 + 5 * (n - 1)

def Blair (n : ℕ) : ℕ :=
  3 + 5 * (n - 1)

def number_at_turn (k : ℕ) : ℕ :=
  if k % 2 = 1 then Jo ((k + 1) / 2) else Blair (k / 2)

theorem number_20 : number_at_turn 20 = 48 :=
by
  sorry

end number_20_l555_55557


namespace find_price_of_pants_l555_55504

theorem find_price_of_pants
  (price_jacket : ℕ)
  (num_jackets : ℕ)
  (price_shorts : ℕ)
  (num_shorts : ℕ)
  (num_pants : ℕ)
  (total_cost : ℕ)
  (h1 : price_jacket = 10)
  (h2 : num_jackets = 3)
  (h3 : price_shorts = 6)
  (h4 : num_shorts = 2)
  (h5 : num_pants = 4)
  (h6 : total_cost = 90)
  : (total_cost - (num_jackets * price_jacket + num_shorts * price_shorts)) / num_pants = 12 :=
by sorry

end find_price_of_pants_l555_55504


namespace pet_store_initial_house_cats_l555_55532

theorem pet_store_initial_house_cats
    (H : ℕ)
    (h1 : 13 + H - 10 = 8) :
    H = 5 :=
by
  sorry

end pet_store_initial_house_cats_l555_55532


namespace a_plus_b_eq_neg7_l555_55585

theorem a_plus_b_eq_neg7 (a b : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * x - 3 > 0) ∨ (x^2 + a * x + b ≤ 0)) ∧
  (∀ x : ℝ, (3 < x ∧ x ≤ 4) → ((x^2 - 2 * x - 3 > 0) ∧ (x^2 + a * x + b ≤ 0))) →
  a + b = -7 :=
by
  sorry

end a_plus_b_eq_neg7_l555_55585


namespace circle_center_sum_is_one_l555_55597

def circle_center_sum (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 6 * y = 3) → ((h = -2) ∧ (k = 3))

theorem circle_center_sum_is_one :
  ∀ h k : ℝ, circle_center_sum h k → h + k = 1 :=
by
  intros h k hc
  sorry

end circle_center_sum_is_one_l555_55597


namespace jeff_pencils_initial_l555_55549

def jeff_initial_pencils (J : ℝ) := J
def jeff_remaining_pencils (J : ℝ) := 0.70 * J
def vicki_initial_pencils (J : ℝ) := 2 * J
def vicki_remaining_pencils (J : ℝ) := 0.25 * vicki_initial_pencils J
def remaining_pencils (J : ℝ) := jeff_remaining_pencils J + vicki_remaining_pencils J

theorem jeff_pencils_initial (J : ℝ) (h : remaining_pencils J = 360) : J = 300 :=
by
  sorry

end jeff_pencils_initial_l555_55549


namespace range_of_f_l555_55511

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.cos x - 4 * Real.sin x

theorem range_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → f x ∈ Set.Icc (-5) 3) ∧
  (∀ y : ℝ, y ∈ Set.Icc (-5) 3 → ∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ f x = y) :=
by
  sorry

end range_of_f_l555_55511


namespace joan_total_socks_l555_55596

theorem joan_total_socks (n : ℕ) (h1 : n / 3 = 60) : n = 180 :=
by
  -- Proof goes here
  sorry

end joan_total_socks_l555_55596


namespace remainder_of_sum_division_l555_55552

theorem remainder_of_sum_division (f y : ℤ) (a b : ℤ) (h_f : f = 5 * a + 3) (h_y : y = 5 * b + 4) :  
  (f + y) % 5 = 2 :=
by
  sorry

end remainder_of_sum_division_l555_55552


namespace probability_queen_in_center_after_2004_moves_l555_55505

def initial_probability (n : ℕ) : ℚ :=
if n = 0 then 1
else if n = 1 then 0
else if n % 2 = 0 then (1 : ℚ) / 2^(n / 2)
else (1 - (1 : ℚ) / 2^((n - 1) / 2)) / 2

theorem probability_queen_in_center_after_2004_moves :
  initial_probability 2004 = 1 / 3 + 1 / (3 * 2^2003) :=
sorry

end probability_queen_in_center_after_2004_moves_l555_55505


namespace triple_solution_exists_and_unique_l555_55553

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end triple_solution_exists_and_unique_l555_55553


namespace IvanPetrovich_daily_lessons_and_charity_l555_55566

def IvanPetrovichConditions (L k : ℕ) : Prop :=
  24 = 8 + 3*L + k ∧
  3000 * L * 21 + 14000 = 70000 + (7000 * k / 3)

theorem IvanPetrovich_daily_lessons_and_charity
  (L k : ℕ) (h : IvanPetrovichConditions L k) :
  L = 2 ∧ 7000 * k / 3 = 70000 := 
by
  sorry

end IvanPetrovich_daily_lessons_and_charity_l555_55566


namespace fixed_rate_calculation_l555_55593

theorem fixed_rate_calculation (f n : ℕ) (h1 : f + 4 * n = 220) (h2 : f + 7 * n = 370) : f = 20 :=
by
  sorry

end fixed_rate_calculation_l555_55593


namespace parabola_through_point_l555_55535

-- Define the parabola equation property
def parabola (a x : ℝ) : ℝ := x^2 + (a+1) * x + a

-- Introduce the main problem statement
theorem parabola_through_point (a m : ℝ) (h : parabola a (-1) = m) : m = 0 :=
by
  sorry

end parabola_through_point_l555_55535


namespace probability_defective_units_l555_55558

theorem probability_defective_units (X : ℝ) (hX : X > 0) :
  let defectA := (14 / 2000) * (0.40 * X)
  let defectB := (9 / 1500) * (0.35 * X)
  let defectC := (7 / 1000) * (0.25 * X)
  let total_defects := defectA + defectB + defectC
  let total_units := X
  let probability := total_defects / total_units
  probability = 0.00665 :=
by
  sorry

end probability_defective_units_l555_55558


namespace training_cost_per_month_correct_l555_55576

-- Define the conditions
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_duration : ℕ := 3
def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2 : ℕ := (45000 / 100) -- 1% of salary2 which is 450
def net_gain_diff : ℕ := 850

-- Define the monthly training cost for the first applicant
def monthly_training_cost : ℕ := 1786667 / 100

-- Prove that the monthly training cost for the first applicant is correct
theorem training_cost_per_month_correct :
  (revenue1 - (salary1 + 3 * monthly_training_cost) = revenue2 - (salary2 + bonus2) + net_gain_diff) :=
by
  sorry

end training_cost_per_month_correct_l555_55576


namespace iron_heating_time_l555_55500

-- Define the conditions as constants
def ironHeatingRate : ℝ := 9 -- degrees Celsius per 20 seconds
def ironCoolingRate : ℝ := 15 -- degrees Celsius per 30 seconds
def coolingTime : ℝ := 180 -- seconds

-- Define the theorem to prove the heating back time
theorem iron_heating_time :
  (coolingTime / 30) * ironCoolingRate = 90 →
  (90 / ironHeatingRate) * 20 = 200 :=
by
  sorry

end iron_heating_time_l555_55500


namespace factor_polynomial_l555_55573

theorem factor_polynomial (x : ℝ) : 54*x^3 - 135*x^5 = 27*x^3*(2 - 5*x^2) := 
by
  sorry

end factor_polynomial_l555_55573


namespace mean_squared_sum_l555_55502

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem mean_squared_sum :
  (x + y + z = 30) ∧ 
  (xyz = 125) ∧ 
  ((1 / x + 1 / y + 1 / z) = 3 / 4) 
  → x^2 + y^2 + z^2 = 712.5 :=
by
  intros h
  have h₁ : x + y + z = 30 := h.1
  have h₂ : xyz = 125 := h.2.1
  have h₃ : (1 / x + 1 / y + 1 / z) = 3 / 4 := h.2.2
  sorry

end mean_squared_sum_l555_55502


namespace total_people_correct_current_people_correct_l555_55548

-- Define the conditions as constants
def morning_people : ℕ := 473
def noon_left : ℕ := 179
def afternoon_people : ℕ := 268

-- Define the total number of people
def total_people : ℕ := morning_people + afternoon_people

-- Define the current number of people in the amusement park
def current_people : ℕ := morning_people - noon_left + afternoon_people

-- Theorem proofs
theorem total_people_correct : total_people = 741 := by sorry
theorem current_people_correct : current_people = 562 := by sorry

end total_people_correct_current_people_correct_l555_55548


namespace lunch_to_novel_ratio_l555_55515

theorem lunch_to_novel_ratio 
  (initial_amount : ℕ) 
  (novel_cost : ℕ) 
  (remaining_after_mall : ℕ) 
  (spent_on_lunch : ℕ)
  (h1 : initial_amount = 50) 
  (h2 : novel_cost = 7) 
  (h3 : remaining_after_mall = 29) 
  (h4 : spent_on_lunch = initial_amount - novel_cost - remaining_after_mall) :
  spent_on_lunch / novel_cost = 2 := 
  sorry

end lunch_to_novel_ratio_l555_55515


namespace find_speed_of_man_l555_55514

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
(v_m + v_s = 6) ∧ (v_m - v_s = 8)

theorem find_speed_of_man :
  ∃ v_m v_s : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 7 :=
by
  sorry

end find_speed_of_man_l555_55514


namespace min_value_2a_plus_b_l555_55550

theorem min_value_2a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + b = a^2 + a * b) :
  2 * a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_2a_plus_b_l555_55550


namespace shopkeeper_loss_percentage_l555_55522

theorem shopkeeper_loss_percentage {cp sp : ℝ} (h1 : cp = 100) (h2 : sp = cp * 1.1) (h_loss : sp * 0.33 = cp * (1 - x / 100)) :
  x = 70 :=
by
  sorry

end shopkeeper_loss_percentage_l555_55522


namespace cost_when_q_is_2_l555_55578

-- Defining the cost function
def cost (q : ℕ) : ℕ := q^3 + q - 1

-- Theorem to prove the cost when q = 2
theorem cost_when_q_is_2 : cost 2 = 9 :=
by
  -- placeholder for the proof
  sorry

end cost_when_q_is_2_l555_55578


namespace meaningful_fraction_l555_55507

theorem meaningful_fraction (x : ℝ) : (∃ (f : ℝ), f = 2 / x) ↔ x ≠ 0 :=
by
  sorry

end meaningful_fraction_l555_55507


namespace quiz_answer_key_count_l555_55543

theorem quiz_answer_key_count :
  let true_false_possibilities := 6  -- Combinations for 3 T/F questions where not all are same
  let multiple_choice_possibilities := 4^3  -- 4 choices for each of 3 multiple-choice questions
  true_false_possibilities * multiple_choice_possibilities = 384 := by
  sorry

end quiz_answer_key_count_l555_55543


namespace f_has_one_zero_l555_55521

noncomputable def f (x : ℝ) : ℝ := 2 * x - 5 - Real.log x

theorem f_has_one_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end f_has_one_zero_l555_55521


namespace johnny_closed_days_l555_55546

theorem johnny_closed_days :
  let dishes_per_day := 40
  let pounds_per_dish := 1.5
  let price_per_pound := 8
  let weekly_expenditure := 1920
  let daily_pounds := dishes_per_day * pounds_per_dish
  let daily_cost := daily_pounds * price_per_pound
  let days_open := weekly_expenditure / daily_cost
  let days_in_week := 7
  let days_closed := days_in_week - days_open
  days_closed = 3 :=
by
  sorry

end johnny_closed_days_l555_55546


namespace ticket_cost_per_ride_l555_55518

theorem ticket_cost_per_ride
  (total_tickets: ℕ) 
  (spent_tickets: ℕ)
  (rides: ℕ)
  (remaining_tickets: ℕ)
  (ride_cost: ℕ)
  (h1: total_tickets = 79)
  (h2: spent_tickets = 23)
  (h3: rides = 8)
  (h4: remaining_tickets = total_tickets - spent_tickets)
  (h5: remaining_tickets = ride_cost * rides):
  ride_cost = 7 :=
by
  sorry

end ticket_cost_per_ride_l555_55518


namespace smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l555_55503

theorem smallest_four_digit_palindrome_div_by_3_with_odd_first_digit :
  ∃ (n : ℕ), (∃ A B : ℕ, n = 1001 * A + 110 * B ∧ 1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ A % 2 = 1) ∧ 3 ∣ n ∧ n = 1221 :=
by
  sorry

end smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l555_55503


namespace proportion_solution_l555_55524

theorem proportion_solution (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) : 
  x = 17.5 * c / (4 * b) := 
sorry

end proportion_solution_l555_55524


namespace solve_system_l555_55520

theorem solve_system :
  {p : ℝ × ℝ // 
    (p.1 + |p.2| = 3 ∧ 2 * |p.1| - p.2 = 3) ∧
    (p = (2, 1) ∨ p = (0, -3) ∨ p = (-6, 9))} :=
by { sorry }

end solve_system_l555_55520


namespace sum_of_digits_power_of_9_gt_9_l555_55556

def sum_of_digits (n : ℕ) : ℕ :=
  -- function to calculate the sum of digits of n 
  sorry

theorem sum_of_digits_power_of_9_gt_9 (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 :=
  sorry

end sum_of_digits_power_of_9_gt_9_l555_55556


namespace find_f_2_l555_55510

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- The statement to prove: if f is monotonically increasing and satisfies the functional equation
-- for all x, then f(2) = e^2 + 1.
theorem find_f_2
  (h_mono : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)
  (h_eq : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1) :
  f 2 = exp 2 + 1 := sorry

end find_f_2_l555_55510


namespace multiplication_expression_l555_55584

theorem multiplication_expression : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end multiplication_expression_l555_55584


namespace circumcircle_area_l555_55565

theorem circumcircle_area (a b c A B C : ℝ) (h : a * Real.cos B + b * Real.cos A = 4 * Real.sin C) :
    π * (2 : ℝ) ^ 2 = 4 * π :=
by
  sorry

end circumcircle_area_l555_55565


namespace line_intersects_x_axis_l555_55571

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end line_intersects_x_axis_l555_55571


namespace odd_function_value_l555_55574

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else -(x^2 + x)

theorem odd_function_value : f (-3) = -12 :=
by
  -- proof goes here
  sorry

end odd_function_value_l555_55574


namespace inequality_holds_for_all_x_l555_55598

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x)) ↔ -2 < m ∧ m ≤ 2 := 
by
  sorry

end inequality_holds_for_all_x_l555_55598


namespace intersect_sets_example_l555_55539

open Set

theorem intersect_sets_example : 
  let A := {x : ℝ | -1 < x ∧ x ≤ 3}
  let B := {x : ℝ | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4}
  A ∩ B = {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3} :=
by
  sorry

end intersect_sets_example_l555_55539


namespace parabola_vertex_sum_l555_55589

theorem parabola_vertex_sum (p q r : ℝ) 
  (h1 : ∃ (a b c : ℝ), ∀ (x : ℝ), a * x ^ 2 + b * x + c = y)
  (h2 : ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = -1)
  (h3 : ∀ (x : ℝ), y = p * x ^ 2 + q * x + r)
  (h4 : y = p * (0 - 3) ^ 2 + r - 1)
  (h5 : y = 8)
  : p + q + r = 3 := 
by
  sorry

end parabola_vertex_sum_l555_55589


namespace circles_intersect_range_l555_55547

def circle1_radius := 3
def circle2_radius := 5

theorem circles_intersect_range : 2 < d ∧ d < 8 :=
by
  let r1 := circle1_radius
  let r2 := circle2_radius
  have h1 : d > r2 - r1 := sorry
  have h2 : d < r2 + r1 := sorry
  exact ⟨h1, h2⟩

end circles_intersect_range_l555_55547


namespace vampire_daily_needs_l555_55506

theorem vampire_daily_needs :
  (7 * 8) / 2 / 7 = 4 :=
by
  sorry

end vampire_daily_needs_l555_55506


namespace avg_of_nine_numbers_l555_55586

theorem avg_of_nine_numbers (average : ℝ) (sum : ℝ) (h : average = (sum / 9)) (h_avg : average = 5.3) : sum = 47.7 := by
  sorry

end avg_of_nine_numbers_l555_55586


namespace area_of_right_square_l555_55568

theorem area_of_right_square (side_length_left : ℕ) (side_length_left_eq : side_length_left = 10) : ∃ area_right, area_right = 68 := 
by
  sorry

end area_of_right_square_l555_55568


namespace amount_needed_is_72_l555_55592

-- Define the given conditions
def original_price : ℝ := 90
def discount_rate : ℝ := 20

-- The goal is to prove that the amount of money needed after the discount is $72
theorem amount_needed_is_72 (P : ℝ) (D : ℝ) (hP : P = original_price) (hD : D = discount_rate) : P - (D / 100 * P) = 72 := 
by sorry

end amount_needed_is_72_l555_55592


namespace distance_P_to_AB_l555_55588

def point_P_condition (P : ℝ) : Prop :=
  P > 0 ∧ P < 1

def parallel_line_property (P : ℝ) (h : ℝ) : Prop :=
  h = 1 - P / 1

theorem distance_P_to_AB (P h : ℝ) (area_total : ℝ) (area_smaller : ℝ) :
  point_P_condition P →
  parallel_line_property P h →
  (area_smaller / area_total) = 1 / 3 →
  h = 2 / 3 :=
by
  intro hP hp hratio
  sorry

end distance_P_to_AB_l555_55588


namespace equal_cost_at_150_miles_l555_55580

def cost_Safety (m : ℝ) := 41.95 + 0.29 * m
def cost_City (m : ℝ) := 38.95 + 0.31 * m
def cost_Metro (m : ℝ) := 44.95 + 0.27 * m

theorem equal_cost_at_150_miles (m : ℝ) :
  cost_Safety m = cost_City m ∧ cost_Safety m = cost_Metro m → m = 150 :=
by
  sorry

end equal_cost_at_150_miles_l555_55580


namespace sum_of_product_of_consecutive_numbers_divisible_by_12_l555_55519

theorem sum_of_product_of_consecutive_numbers_divisible_by_12 (a : ℤ) : 
  (a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1) % 12 = 0 :=
by sorry

end sum_of_product_of_consecutive_numbers_divisible_by_12_l555_55519


namespace feet_of_pipe_per_bolt_l555_55572

-- Definition of the initial conditions
def total_pipe_length := 40 -- total feet of pipe
def washers_per_bolt := 2
def initial_washers := 20
def remaining_washers := 4

-- The proof statement
theorem feet_of_pipe_per_bolt :
  ∀ (total_pipe_length washers_per_bolt initial_washers remaining_washers : ℕ),
  initial_washers - remaining_washers = 16 → -- 16 washers used
  16 / washers_per_bolt = 8 → -- 8 bolts used
  total_pipe_length / 8 = 5 :=
by
  intros
  sorry

end feet_of_pipe_per_bolt_l555_55572


namespace evaluate_expression_l555_55561

theorem evaluate_expression : 5^2 + 15 / 3 - (3 * 2)^2 = -6 := 
by
  sorry

end evaluate_expression_l555_55561


namespace not_unique_y_20_paise_l555_55563

theorem not_unique_y_20_paise (x y z w : ℕ) : 
  x + y + z + w = 750 → 10 * x + 20 * y + 50 * z + 100 * w = 27500 → ∃ (y₁ y₂ : ℕ), y₁ ≠ y₂ :=
by 
  intro h1 h2
  -- Without additional constraints on x, y, z, w,
  -- suppose that there are at least two different solutions satisfying both equations,
  -- demonstrating the non-uniqueness of y.
  sorry

end not_unique_y_20_paise_l555_55563


namespace a_share_in_gain_l555_55540

noncomputable def investment_share (x: ℝ) (total_gain: ℝ): ℝ := 
  let a_interest := x * 0.1
  let b_interest := (2 * x) * (7 / 100) * (1.5)
  let c_interest := (3 * x) * (10 / 100) * (1.33)
  let total_interest := a_interest + b_interest + c_interest
  a_interest

theorem a_share_in_gain (total_gain: ℝ) (a_share: ℝ) (x: ℝ)
  (hx: 0.709 * x = total_gain):
  investment_share x total_gain = a_share :=
sorry

end a_share_in_gain_l555_55540


namespace semicircle_radius_l555_55512

theorem semicircle_radius (P L W : ℝ) (π : Real) (r : ℝ) 
  (hP : P = 144) (hL : L = 48) (hW : W = 24) (hD : ∃ d, d = 2 * r ∧ d = L) :
  r = 48 / (π + 2) := 
by
  sorry

end semicircle_radius_l555_55512


namespace flu_infection_equation_l555_55555

theorem flu_infection_equation (x : ℝ) :
  (1 + x)^2 = 144 :=
sorry

end flu_infection_equation_l555_55555


namespace largest_five_digit_integer_l555_55570

/-- The product of the digits of the integer 98752 is (7 * 6 * 5 * 4 * 3 * 2 * 1), and
    98752 is the largest five-digit integer with this property. -/
theorem largest_five_digit_integer :
  (∃ (n : ℕ), n = 98752 ∧ (∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10^4 + d2 * 10^3 + d3 * 10^2 + d4 * 10 + d5 ∧
    (d1 * d2 * d3 * d4 * d5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) ∧
    (∀ (m : ℕ), m ≠ 98752 → m < 100000 ∧ (∃ (e1 e2 e3 e4 e5 : ℕ),
    m = e1 * 10^4 + e2 * 10^3 + e3 * 10^2 + e4 * 10 + e5 →
    (e1 * e2 * e3 * e4 * e5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) → m < 98752)))) :=
  sorry

end largest_five_digit_integer_l555_55570


namespace proof_abc_div_def_l555_55581

def abc_div_def (a b c d e f : ℚ) : Prop := 
  a / b = 1 / 3 ∧ b / c = 2 ∧ c / d = 1 / 2 ∧ d / e = 3 ∧ e / f = 1 / 8 → (a * b * c) / (d * e * f) = 1 / 16

theorem proof_abc_div_def (a b c d e f : ℚ) :
  abc_div_def a b c d e f :=
by 
  sorry

end proof_abc_div_def_l555_55581


namespace rectangle_ratio_l555_55564

theorem rectangle_ratio (s x y : ℝ) 
  (h_outer_area : (2 * s) ^ 2 = 4 * s ^ 2)
  (h_inner_sides : s + 2 * y = 2 * s)
  (h_outer_sides : x + y = 2 * s) :
  x / y = 3 :=
by
  sorry

end rectangle_ratio_l555_55564


namespace smallest_odd_digit_number_gt_1000_mult_5_l555_55538

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end smallest_odd_digit_number_gt_1000_mult_5_l555_55538


namespace inequality_holds_l555_55599

variable {a b c : ℝ}

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
sorry

end inequality_holds_l555_55599


namespace h_at_3_l555_55529

noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := Real.sqrt (f x) - 3
noncomputable def h (x : ℝ) := g (f x)

theorem h_at_3 : h 3 = Real.sqrt 43 - 3 := by
  sorry

end h_at_3_l555_55529


namespace number_of_japanese_selectors_l555_55508

theorem number_of_japanese_selectors (F C J : ℕ) (h1 : J = 3 * C) (h2 : C = F + 15) (h3 : J + C + F = 165) : J = 108 :=
by
sorry

end number_of_japanese_selectors_l555_55508


namespace inequality_solution_absolute_inequality_l555_55501

-- Statement for Inequality Solution Problem
theorem inequality_solution (x : ℝ) : |x - 1| + |2 * x + 1| > 3 ↔ (x < -1 ∨ x > 1) := sorry

-- Statement for Absolute Inequality Problem with Bounds
theorem absolute_inequality (a b : ℝ) (ha : -1 ≤ a) (hb : a ≤ 1) (hc : -1 ≤ b) (hd : b ≤ 1) : 
  |1 + (a * b) / 4| > |(a + b) / 2| := sorry

end inequality_solution_absolute_inequality_l555_55501


namespace hyperbola_range_k_l555_55582

noncomputable def hyperbola_equation (x y k : ℝ) : Prop :=
    (x^2) / (|k|-2) + (y^2) / (5-k) = 1

theorem hyperbola_range_k (k : ℝ) :
    (∃ x y, hyperbola_equation x y k) → (k > 5 ∨ (-2 < k ∧ k < 2)) :=
by 
    sorry

end hyperbola_range_k_l555_55582


namespace smallest_positive_period_f_max_min_f_on_interval_l555_55526

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem smallest_positive_period_f : 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ Real.pi) :=
sorry

theorem max_min_f_on_interval :
  let a := Real.pi / 4
  let b := 2 * Real.pi / 3
  ∃ M m, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M ∧ f x ≥ m) ∧ (M = 2) ∧ (m = -1) :=
sorry

end smallest_positive_period_f_max_min_f_on_interval_l555_55526


namespace percentage_cut_second_week_l555_55542

noncomputable def calculate_final_weight (initial_weight : ℝ) (percentage1 : ℝ) (percentage2 : ℝ) (percentage3 : ℝ) : ℝ :=
  let weight_after_first_week := (1 - percentage1 / 100) * initial_weight
  let weight_after_second_week := (1 - percentage2 / 100) * weight_after_first_week
  let final_weight := (1 - percentage3 / 100) * weight_after_second_week
  final_weight

theorem percentage_cut_second_week : 
  ∀ (initial_weight : ℝ) (final_weight : ℝ), (initial_weight = 250) → (final_weight = 105) →
    (calculate_final_weight initial_weight 30 x 25 = final_weight) → 
    x = 20 := 
by 
  intros initial_weight final_weight h1 h2 h3
  sorry

end percentage_cut_second_week_l555_55542


namespace hamburgers_total_l555_55528

theorem hamburgers_total (initial_hamburgers : ℝ) (additional_hamburgers : ℝ) (h₁ : initial_hamburgers = 9.0) (h₂ : additional_hamburgers = 3.0) : initial_hamburgers + additional_hamburgers = 12.0 :=
by
  rw [h₁, h₂]
  norm_num

end hamburgers_total_l555_55528


namespace find_tip_percentage_l555_55554

def original_bill : ℝ := 139.00
def per_person_share : ℝ := 30.58
def number_of_people : ℕ := 5

theorem find_tip_percentage (original_bill : ℝ) (per_person_share : ℝ) (number_of_people : ℕ) 
  (total_paid : ℝ := per_person_share * number_of_people) 
  (tip_amount : ℝ := total_paid - original_bill) : 
  (tip_amount / original_bill) * 100 = 10 :=
by
  sorry

end find_tip_percentage_l555_55554


namespace find_x_range_l555_55525

theorem find_x_range {x : ℝ} : 
  (∀ (m : ℝ), abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 ) →
  ( ( -1 + Real.sqrt 7 ) / 2 < x ∧ x < ( 1 + Real.sqrt 3 ) / 2 ) :=
by
  intros h
  sorry

end find_x_range_l555_55525


namespace parametric_line_eq_l555_55545

-- Define the parameterized functions for x and y 
def parametric_x (t : ℝ) : ℝ := 3 * t + 7
def parametric_y (t : ℝ) : ℝ := 5 * t - 8

-- Define the equation of the line (here it's a relation that relates x and y)
def line_equation (x y : ℝ) : Prop := 
  y = (5 / 3) * x - (59 / 3)

theorem parametric_line_eq : 
  ∃ t : ℝ, line_equation (parametric_x t) (parametric_y t) := 
by
  -- Proof goes here
  sorry

end parametric_line_eq_l555_55545


namespace max_download_speed_l555_55534

def download_speed (size_GB : ℕ) (time_hours : ℕ) : ℚ :=
  let size_MB := size_GB * 1024
  let time_seconds := time_hours * 60 * 60
  size_MB / time_seconds

theorem max_download_speed (h₁ : size_GB = 360) (h₂ : time_hours = 2) :
  download_speed size_GB time_hours = 51.2 :=
by
  sorry

end max_download_speed_l555_55534


namespace number_of_men_in_second_group_l555_55562

theorem number_of_men_in_second_group 
  (work : ℕ)
  (days_first_group days_second_group : ℕ)
  (men_first_group men_second_group : ℕ)
  (h1 : work = men_first_group * days_first_group)
  (h2 : work = men_second_group * days_second_group)
  (h3 : men_first_group = 20)
  (h4 : days_first_group = 30)
  (h5 : days_second_group = 24) :
  men_second_group = 25 :=
by
  sorry

end number_of_men_in_second_group_l555_55562


namespace evaluate_fraction_sum_l555_55591

theorem evaluate_fraction_sum (a b c : ℝ) (h : a ≠ 40) (h_a : b ≠ 75) (h_b : c ≠ 85)
  (h_cond : (a / (40 - a)) + (b / (75 - b)) + (c / (85 - c)) = 8) :
  (8 / (40 - a)) + (15 / (75 - b)) + (17 / (85 - c)) = 40 := 
sorry

end evaluate_fraction_sum_l555_55591


namespace rectangle_symmetry_l555_55551

-- Define basic geometric terms and the notion of symmetry
structure Rectangle where
  length : ℝ
  width : ℝ
  (length_pos : 0 < length)
  (width_pos : 0 < width)

def is_axes_of_symmetry (r : Rectangle) (n : ℕ) : Prop :=
  -- A hypothetical function that determines whether a rectangle r has n axes of symmetry
  sorry

theorem rectangle_symmetry (r : Rectangle) : is_axes_of_symmetry r 2 := 
  -- This theorem states that a rectangle has exactly 2 axes of symmetry
  sorry

end rectangle_symmetry_l555_55551


namespace paul_can_buy_toys_l555_55560

-- Definitions of the given conditions
def initial_dollars : ℕ := 3
def allowance : ℕ := 7
def toy_cost : ℕ := 5

-- Required proof statement
theorem paul_can_buy_toys : (initial_dollars + allowance) / toy_cost = 2 := by
  sorry

end paul_can_buy_toys_l555_55560


namespace hours_per_day_l555_55509

-- Define the parameters
def A1 := 57
def D1 := 12
def H2 := 6
def A2 := 30
def D2 := 19

-- Define the target Equation
theorem hours_per_day :
  A1 * D1 * H = A2 * D2 * H2 → H = 5 :=
by
  sorry

end hours_per_day_l555_55509


namespace solution_set_ineq_l555_55533

noncomputable
def f (x : ℝ) : ℝ := sorry
noncomputable
def g (x : ℝ) : ℝ := sorry

axiom h_f_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_g_even : ∀ x : ℝ, g (-x) = g x
axiom h_deriv_pos : ∀ x : ℝ, x < 0 → deriv f x * g x + f x * deriv g x > 0
axiom h_g_neg_three_zero : g (-3) = 0

theorem solution_set_ineq : { x : ℝ | f x * g x < 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | 0 < x ∧ x < 3 } := 
by sorry

end solution_set_ineq_l555_55533


namespace textbook_weight_difference_l555_55594

variable (chemWeight : ℝ) (geomWeight : ℝ)

def chem_weight := chemWeight = 7.12
def geom_weight := geomWeight = 0.62

theorem textbook_weight_difference : chemWeight - geomWeight = 6.50 :=
by
  sorry

end textbook_weight_difference_l555_55594


namespace sum_three_times_integers_15_to_25_l555_55583

noncomputable def sumArithmeticSequence (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem sum_three_times_integers_15_to_25 :
  let a := 15
  let d := 1
  let n := 25 - 15 + 1
  3 * sumArithmeticSequence a d n = 660 := by
  -- This part can be filled in with the actual proof
  sorry

end sum_three_times_integers_15_to_25_l555_55583


namespace triangle_angle_A_l555_55531

theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) (hC : C = Real.pi / 6) (hCos : c = 2 * a * Real.cos B) : A = (5 * Real.pi) / 12 :=
  sorry

end triangle_angle_A_l555_55531


namespace find_a5_l555_55513

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2 + 1) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h3 : S 1 = 2) :
  a 5 = 9 :=
sorry

end find_a5_l555_55513


namespace expression_simplification_l555_55575

variable (x : ℝ)

-- Define the expression as given in the problem
def Expr : ℝ := (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 3)

-- Lean statement to verify that the expression simplifies to the given polynomial
theorem expression_simplification : Expr x = 6 * x^3 - 16 * x^2 + 43 * x - 70 := by
  sorry

end expression_simplification_l555_55575


namespace janet_total_owed_l555_55516

def warehouseHourlyWage : ℝ := 15
def managerHourlyWage : ℝ := 20
def numWarehouseWorkers : ℕ := 4
def numManagers : ℕ := 2
def workDaysPerMonth : ℕ := 25
def workHoursPerDay : ℕ := 8
def ficaTaxRate : ℝ := 0.10

theorem janet_total_owed : 
  let warehouseWorkerMonthlyWage := warehouseHourlyWage * workDaysPerMonth * workHoursPerDay
  let managerMonthlyWage := managerHourlyWage * workDaysPerMonth * workHoursPerDay
  let totalMonthlyWages := (warehouseWorkerMonthlyWage * numWarehouseWorkers) + (managerMonthlyWage * numManagers)
  let ficaTaxes := totalMonthlyWages * ficaTaxRate
  let totalAmountOwed := totalMonthlyWages + ficaTaxes
  totalAmountOwed = 22000 := by
  sorry

end janet_total_owed_l555_55516


namespace exists_five_distinct_natural_numbers_product_eq_1000_l555_55523

theorem exists_five_distinct_natural_numbers_product_eq_1000 :
  ∃ (a b c d e : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 1000 := sorry

end exists_five_distinct_natural_numbers_product_eq_1000_l555_55523


namespace car_average_speed_l555_55530

-- Definitions based on conditions
def distance_first_hour : ℤ := 100
def distance_second_hour : ℤ := 60
def time_first_hour : ℤ := 1
def time_second_hour : ℤ := 1

-- Total distance and time calculations
def total_distance : ℤ := distance_first_hour + distance_second_hour
def total_time : ℤ := time_first_hour + time_second_hour

-- The average speed of the car
def average_speed : ℤ := total_distance / total_time

-- Proof statement
theorem car_average_speed : average_speed = 80 := by
  sorry

end car_average_speed_l555_55530


namespace ralph_has_18_fewer_pictures_l555_55567

/-- Ralph has 58 pictures of wild animals. Derrick has 76 pictures of wild animals.
    Prove that Ralph has 18 fewer pictures of wild animals compared to Derrick. -/
theorem ralph_has_18_fewer_pictures :
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  76 - 58 = 18 :=
by
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  show 76 - 58 = 18
  sorry

end ralph_has_18_fewer_pictures_l555_55567


namespace condition_sufficiency_not_necessity_l555_55527

variable {x y : ℝ}

theorem condition_sufficiency_not_necessity (hx : x ≥ 0) (hy : y ≥ 0) :
  (xy > 0 → |x + y| = |x| + |y|) ∧ (|x + y| = |x| + |y| → xy ≥ 0) :=
sorry

end condition_sufficiency_not_necessity_l555_55527
