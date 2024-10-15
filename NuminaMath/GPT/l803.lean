import Mathlib

namespace NUMINAMATH_GPT_y_intercept_of_line_l803_80306

theorem y_intercept_of_line (m : ℝ) (x1 y1 : ℝ) (h_slope : m = -3) (h_x_intercept : x1 = 7) (h_y_intercept : y1 = 0) : 
  ∃ y_intercept : ℝ, y_intercept = 21 ∧ y_intercept = -m * 0 + 21 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l803_80306


namespace NUMINAMATH_GPT_annalise_spending_l803_80389

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end NUMINAMATH_GPT_annalise_spending_l803_80389


namespace NUMINAMATH_GPT_problem1_problem2_l803_80349

section ProofProblems

variables {a b : ℝ}

-- Given that a and b are distinct positive numbers
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_neq_b : a ≠ b

-- Problem (i): Prove that a^4 + b^4 > a^3 * b + a * b^3
theorem problem1 : a^4 + b^4 > a^3 * b + a * b^3 :=
by {
  sorry
}

-- Problem (ii): Prove that a^5 + b^5 > a^3 * b^2 + a^2 * b^3
theorem problem2 : a^5 + b^5 > a^3 * b^2 + a^2 * b^3 :=
by {
  sorry
}

end ProofProblems

end NUMINAMATH_GPT_problem1_problem2_l803_80349


namespace NUMINAMATH_GPT_lap_distance_l803_80322

theorem lap_distance (boys_laps : ℕ) (girls_extra_laps : ℕ) (total_girls_miles : ℚ) : 
  boys_laps = 27 → girls_extra_laps = 9 → total_girls_miles = 27 →
  (total_girls_miles / (boys_laps + girls_extra_laps) = 3 / 4) :=
by
  intros hb hg hm
  sorry

end NUMINAMATH_GPT_lap_distance_l803_80322


namespace NUMINAMATH_GPT_f_of_g_of_2_l803_80383

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end NUMINAMATH_GPT_f_of_g_of_2_l803_80383


namespace NUMINAMATH_GPT_eunji_received_900_won_l803_80345

-- Define the conditions
def eunji_pocket_money (X : ℝ) : Prop :=
  (X / 2 + 550 = 1000)

-- Define the theorem to prove the question equals the correct answer
theorem eunji_received_900_won {X : ℝ} (h : eunji_pocket_money X) : X = 900 :=
  by
    sorry

end NUMINAMATH_GPT_eunji_received_900_won_l803_80345


namespace NUMINAMATH_GPT_solve_for_y_l803_80357

theorem solve_for_y (x y : ℝ) (h : 5 * x - y = 6) : y = 5 * x - 6 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l803_80357


namespace NUMINAMATH_GPT_train_length_correct_l803_80316

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms - speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_correct :
  length_of_first_train 72 36 69.99440044796417 300 = 399.9440044796417 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l803_80316


namespace NUMINAMATH_GPT_solve_students_and_apples_l803_80343

noncomputable def students_and_apples : Prop :=
  ∃ (x y : ℕ), y = 4 * x + 3 ∧ 6 * (x - 1) ≤ y ∧ y ≤ 6 * (x - 1) + 2 ∧ x = 4 ∧ y = 19

theorem solve_students_and_apples : students_and_apples :=
  sorry

end NUMINAMATH_GPT_solve_students_and_apples_l803_80343


namespace NUMINAMATH_GPT_container_unoccupied_volume_is_628_l803_80328

def rectangular_prism_volume (length width height : ℕ) : ℕ :=
  length * width * height

def water_volume (total_volume : ℕ) : ℕ :=
  total_volume / 3

def ice_cubes_volume (number_of_cubes volume_per_cube : ℕ) : ℕ :=
  number_of_cubes * volume_per_cube

def unoccupied_volume (total_volume occupied_volume : ℕ) : ℕ :=
  total_volume - occupied_volume

theorem container_unoccupied_volume_is_628 :
  let length := 12
  let width := 10
  let height := 8
  let number_of_ice_cubes := 12
  let volume_per_ice_cube := 1
  let V := rectangular_prism_volume length width height
  let V_water := water_volume V
  let V_ice := ice_cubes_volume number_of_ice_cubes volume_per_ice_cube
  let V_occupied := V_water + V_ice
  unoccupied_volume V V_occupied = 628 :=
by
  sorry

end NUMINAMATH_GPT_container_unoccupied_volume_is_628_l803_80328


namespace NUMINAMATH_GPT_new_price_after_increase_l803_80359

def original_price (y : ℝ) : Prop := 2 * y = 540

theorem new_price_after_increase (y : ℝ) (h : original_price y) : 1.3 * y = 351 :=
by sorry

end NUMINAMATH_GPT_new_price_after_increase_l803_80359


namespace NUMINAMATH_GPT_hearing_news_probability_l803_80365

noncomputable def probability_of_hearing_news : ℚ :=
  let broadcast_cycle := 30 -- total time in minutes for each broadcast cycle
  let news_duration := 5  -- duration of each news broadcast in minutes
  news_duration / broadcast_cycle

theorem hearing_news_probability : probability_of_hearing_news = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_hearing_news_probability_l803_80365


namespace NUMINAMATH_GPT_big_bottles_sold_percentage_l803_80358

-- Definitions based on conditions
def small_bottles_initial : ℕ := 5000
def big_bottles_initial : ℕ := 12000
def small_bottles_sold_percentage : ℝ := 0.15
def total_bottles_remaining : ℕ := 14090

-- Question in Lean 4
theorem big_bottles_sold_percentage : 
  (12000 - (12000 * x / 100) + 5000 - (5000 * 15 / 100)) = 14090 → x = 18 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_big_bottles_sold_percentage_l803_80358


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_y_l803_80391

theorem minimum_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (x - 1) * (y - 1) = 1) : x + y = 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_y_l803_80391


namespace NUMINAMATH_GPT_combined_area_difference_l803_80362

theorem combined_area_difference :
  let rect1_len := 11
  let rect1_wid := 11
  let rect2_len := 5.5
  let rect2_wid := 11
  2 * (rect1_len * rect1_wid) - 2 * (rect2_len * rect2_wid) = 121 := by
  sorry

end NUMINAMATH_GPT_combined_area_difference_l803_80362


namespace NUMINAMATH_GPT_inequality_solution_set_l803_80308

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x → x > -3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l803_80308


namespace NUMINAMATH_GPT_compare_charges_l803_80396

/-
Travel agencies A and B have group discount methods with the original price being $200 per person.
- Agency A: Buy 4 full-price tickets, the rest are half price.
- Agency B: All customers get a 30% discount.
Prove the given relationships based on the number of travelers.
-/

def agency_a_cost (x : ℕ) : ℕ :=
  if 0 < x ∧ x < 4 then 200 * x
  else if x ≥ 4 then 100 * x + 400
  else 0

def agency_b_cost (x : ℕ) : ℕ :=
  140 * x

theorem compare_charges (x : ℕ) :
  (agency_a_cost x < agency_b_cost x -> x > 10) ∧
  (agency_a_cost x = agency_b_cost x -> x = 10) ∧
  (agency_a_cost x > agency_b_cost x -> x < 10) :=
by
  sorry

end NUMINAMATH_GPT_compare_charges_l803_80396


namespace NUMINAMATH_GPT_no_x2_term_imp_a_eq_half_l803_80355

theorem no_x2_term_imp_a_eq_half (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x^2 - 2 * a * x + a^2) = x^3 + (1 - 2 * a) * x^2 + ((a^2 - 2 * a) * x + a^2)) →
  (∀ c : ℝ, (1 - 2 * a) = 0) →
  a = 1 / 2 :=
by
  intros h_prod h_eq
  have h_eq' : 1 - 2 * a = 0 := h_eq 0
  linarith

end NUMINAMATH_GPT_no_x2_term_imp_a_eq_half_l803_80355


namespace NUMINAMATH_GPT_probability_same_number_l803_80304

def is_multiple (n factor : ℕ) : Prop :=
  ∃ k : ℕ, n = k * factor

def multiples_below (factor upper_limit : ℕ) : ℕ :=
  (upper_limit - 1) / factor

theorem probability_same_number :
  let upper_limit := 250
  let billy_factor := 20
  let bobbi_factor := 30
  let common_factor := 60
  let billy_multiples := multiples_below billy_factor upper_limit
  let bobbi_multiples := multiples_below bobbi_factor upper_limit
  let common_multiples := multiples_below common_factor upper_limit
  (common_multiples : ℚ) / (billy_multiples * bobbi_multiples) = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_number_l803_80304


namespace NUMINAMATH_GPT_find_coordinates_of_P0_find_equation_of_l_l803_80325

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

def is_in_third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

/-- Problem statement 1: Find the coordinates of P₀ --/
theorem find_coordinates_of_P0 (p0 : ℝ × ℝ)
    (h_tangent_parallel : tangent_slope p0.1 = 4)
    (h_third_quadrant : is_in_third_quadrant p0) :
    p0 = (-1, -4) :=
sorry

/-- Problem statement 2: Find the equation of line l --/
theorem find_equation_of_l (P0 : ℝ × ℝ)
    (h_P0_coordinates: P0 = (-1, -4))
    (h_perpendicular : ∀ (l1_slope : ℝ), l1_slope = 4 → ∃ l_slope : ℝ, l_slope = (-1) / 4)
    (x y : ℝ) : 
    line_eq 1 4 17 x y :=
sorry

end NUMINAMATH_GPT_find_coordinates_of_P0_find_equation_of_l_l803_80325


namespace NUMINAMATH_GPT_class_weighted_average_l803_80388

theorem class_weighted_average
    (num_students : ℕ)
    (sect1_avg sect2_avg sect3_avg remainder_avg : ℝ)
    (sect1_pct sect2_pct sect3_pct remainder_pct : ℝ)
    (weight1 weight2 weight3 weight4 : ℝ)
    (h_total_students : num_students = 120)
    (h_sect1_avg : sect1_avg = 96.5)
    (h_sect2_avg : sect2_avg = 78.4)
    (h_sect3_avg : sect3_avg = 88.2)
    (h_remainder_avg : remainder_avg = 64.7)
    (h_sect1_pct : sect1_pct = 0.187)
    (h_sect2_pct : sect2_pct = 0.355)
    (h_sect3_pct : sect3_pct = 0.258)
    (h_remainder_pct : remainder_pct = 1 - (sect1_pct + sect2_pct + sect3_pct))
    (h_weight1 : weight1 = 0.35)
    (h_weight2 : weight2 = 0.25)
    (h_weight3 : weight3 = 0.30)
    (h_weight4 : weight4 = 0.10) :
    (sect1_avg * weight1 + sect2_avg * weight2 + sect3_avg * weight3 + remainder_avg * weight4) * 100 = 86 := 
sorry

end NUMINAMATH_GPT_class_weighted_average_l803_80388


namespace NUMINAMATH_GPT_gold_initial_amount_l803_80314

theorem gold_initial_amount :
  ∃ x : ℝ, x - (x / 2 * (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6)) = 1 ∧ x = 1.2 :=
by
  existsi 1.2
  sorry

end NUMINAMATH_GPT_gold_initial_amount_l803_80314


namespace NUMINAMATH_GPT_two_rides_combinations_l803_80326

-- Define the number of friends
def num_friends : ℕ := 7

-- Define the size of the group for one ride
def ride_group_size : ℕ := 4

-- Define the number of combinations of choosing 'ride_group_size' out of 'num_friends'
def combinations_first_ride : ℕ := Nat.choose num_friends ride_group_size

-- Define the number of friends left for the second ride
def remaining_friends : ℕ := num_friends - ride_group_size

-- Define the number of combinations of choosing 'ride_group_size' out of 'remaining_friends' friends
def combinations_second_ride : ℕ := Nat.choose remaining_friends ride_group_size

-- Define the total number of possible combinations for two rides
def total_combinations : ℕ := combinations_first_ride * combinations_second_ride

-- The final theorem stating the total number of combinations is equal to 525
theorem two_rides_combinations : total_combinations = 525 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_two_rides_combinations_l803_80326


namespace NUMINAMATH_GPT_annual_income_calculation_l803_80397

noncomputable def annual_income (investment : ℝ) (price_per_share : ℝ) (dividend_rate : ℝ) (face_value : ℝ) : ℝ :=
  let number_of_shares := investment / price_per_share
  number_of_shares * face_value * dividend_rate

theorem annual_income_calculation :
  annual_income 4455 8.25 0.12 10 = 648 :=
by
  sorry

end NUMINAMATH_GPT_annual_income_calculation_l803_80397


namespace NUMINAMATH_GPT_determine_m_for_unique_solution_l803_80338

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end NUMINAMATH_GPT_determine_m_for_unique_solution_l803_80338


namespace NUMINAMATH_GPT_new_year_markup_l803_80380

variable (C : ℝ) -- original cost of the turtleneck sweater
variable (N : ℝ) -- New Year season markup in decimal form
variable (final_price : ℝ) -- final price in February

-- Conditions
def initial_markup (C : ℝ) := 1.20 * C
def after_new_year_markup (C : ℝ) (N : ℝ) := (1 + N) * initial_markup C
def discount_in_february (C : ℝ) (N : ℝ) := 0.94 * after_new_year_markup C N
def profit_in_february (C : ℝ) := 1.41 * C

-- Mathematically equivalent proof problem (statement only)
theorem new_year_markup :
  ∀ C : ℝ, ∀ N : ℝ,
    discount_in_february C N = profit_in_february C →
    N = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_new_year_markup_l803_80380


namespace NUMINAMATH_GPT_width_of_room_l803_80312

theorem width_of_room
  (carpet_has : ℕ)
  (room_length : ℕ)
  (carpet_needs : ℕ)
  (h1 : carpet_has = 18)
  (h2 : room_length = 4)
  (h3 : carpet_needs = 62) :
  (carpet_has + carpet_needs) = room_length * 20 :=
by
  sorry

end NUMINAMATH_GPT_width_of_room_l803_80312


namespace NUMINAMATH_GPT_kelcie_books_multiple_l803_80302

theorem kelcie_books_multiple (x : ℕ) :
  let megan_books := 32
  let kelcie_books := megan_books / 4
  let greg_books := x * kelcie_books + 9
  let total_books := megan_books + kelcie_books + greg_books
  total_books = 65 → x = 2 :=
by
  intros megan_books kelcie_books greg_books total_books h
  sorry

end NUMINAMATH_GPT_kelcie_books_multiple_l803_80302


namespace NUMINAMATH_GPT_infinite_68_in_cells_no_repeats_in_cells_l803_80317

-- Define the spiral placement function
def spiral (n : ℕ) : ℕ := sorry  -- This function should describe the placement of numbers in the spiral

-- Define a function to get the sum of the numbers in the nodes of a cell.
def cell_sum (cell : ℕ) : ℕ := sorry  -- This function should calculate the sum based on the spiral placement.

-- Proving that numbers divisible by 68 appear infinitely many times in cell centers
theorem infinite_68_in_cells : ∀ N : ℕ, ∃ n > N, 68 ∣ cell_sum n :=
by sorry

-- Proving that numbers in cell centers do not repeat
theorem no_repeats_in_cells : ∀ m n : ℕ, m ≠ n → cell_sum m ≠ cell_sum n :=
by sorry

end NUMINAMATH_GPT_infinite_68_in_cells_no_repeats_in_cells_l803_80317


namespace NUMINAMATH_GPT_bella_steps_l803_80300

/-- Bella begins to walk from her house toward her friend Ella's house. At the same time, Ella starts to skate toward Bella's house. They each maintain a constant speed, and Ella skates three times as fast as Bella walks. The distance between their houses is 10560 feet, and Bella covers 3 feet with each step. Prove that Bella will take 880 steps by the time she meets Ella. -/
theorem bella_steps 
  (d : ℝ)    -- distance between their houses in feet
  (s_bella : ℝ)    -- speed of Bella in feet per minute
  (s_ella : ℝ)    -- speed of Ella in feet per minute
  (steps_per_ft : ℝ)    -- feet per step of Bella
  (h1 : d = 10560)    -- distance between their houses is 10560 feet
  (h2 : s_ella = 3 * s_bella)    -- Ella skates three times as fast as Bella
  (h3 : steps_per_ft = 3)    -- Bella covers 3 feet with each step
  : (10560 / (4 * s_bella)) * s_bella / 3 = 880 :=
by
  -- proof here 
  sorry

end NUMINAMATH_GPT_bella_steps_l803_80300


namespace NUMINAMATH_GPT_arithmetic_mean_of_14_22_36_l803_80385

theorem arithmetic_mean_of_14_22_36 : (14 + 22 + 36) / 3 = 24 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_14_22_36_l803_80385


namespace NUMINAMATH_GPT_how_many_pints_did_Annie_pick_l803_80371

theorem how_many_pints_did_Annie_pick (x : ℕ) (h1 : Kathryn = x + 2)
                                      (h2 : Ben = Kathryn - 3)
                                      (h3 : x + Kathryn + Ben = 25) : x = 8 :=
  sorry

end NUMINAMATH_GPT_how_many_pints_did_Annie_pick_l803_80371


namespace NUMINAMATH_GPT_robert_arrival_time_l803_80361

def arrival_time (T : ℕ) : Prop :=
  ∃ D : ℕ, D = 10 * (12 - T) ∧ D = 15 * (13 - T)

theorem robert_arrival_time : arrival_time 15 :=
by
  sorry

end NUMINAMATH_GPT_robert_arrival_time_l803_80361


namespace NUMINAMATH_GPT_cost_B_solution_l803_80378

variable (cost_B : ℝ)

/-- The number of items of type A that can be purchased with 1000 yuan 
is equal to the number of items of type B that can be purchased with 800 yuan. -/
def items_purchased_equality (cost_B : ℝ) : Prop :=
  1000 / (cost_B + 10) = 800 / cost_B

/-- The cost of each item of type A is 10 yuan more than the cost of each item of type B. -/
def cost_difference (cost_B : ℝ) : Prop :=
  cost_B + 10 - cost_B = 10

/-- The cost of each item of type B is 40 yuan. -/
theorem cost_B_solution (h1: items_purchased_equality cost_B) (h2: cost_difference cost_B) :
  cost_B = 40 := by
sorry

end NUMINAMATH_GPT_cost_B_solution_l803_80378


namespace NUMINAMATH_GPT_charlotte_total_dog_walking_time_l803_80373

def poodles_monday : ℕ := 4
def chihuahuas_monday : ℕ := 2
def poodles_tuesday : ℕ := 4
def chihuahuas_tuesday : ℕ := 2
def labradors_wednesday : ℕ := 4

def time_poodle : ℕ := 2
def time_chihuahua : ℕ := 1
def time_labrador : ℕ := 3

def total_time_monday : ℕ := poodles_monday * time_poodle + chihuahuas_monday * time_chihuahua
def total_time_tuesday : ℕ := poodles_tuesday * time_poodle + chihuahuas_tuesday * time_chihuahua
def total_time_wednesday : ℕ := labradors_wednesday * time_labrador

def total_time_week : ℕ := total_time_monday + total_time_tuesday + total_time_wednesday

theorem charlotte_total_dog_walking_time : total_time_week = 32 := by
  -- Lean allows us to state the theorem without proving it.
  sorry

end NUMINAMATH_GPT_charlotte_total_dog_walking_time_l803_80373


namespace NUMINAMATH_GPT_three_digit_numbers_l803_80318

theorem three_digit_numbers (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : n^2 % 1000 = n % 1000) : 
  n = 376 ∨ n = 625 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_l803_80318


namespace NUMINAMATH_GPT_silas_payment_ratio_l803_80364

theorem silas_payment_ratio (total_bill : ℕ) (tip_rate : ℝ) (friend_payment : ℕ) (S : ℕ) :
  total_bill = 150 →
  tip_rate = 0.10 →
  friend_payment = 18 →
  (S + 5 * friend_payment = total_bill + total_bill * tip_rate) →
  (S : ℝ) / total_bill = 1 / 2 :=
by
  intros h_total_bill h_tip_rate h_friend_payment h_budget_eq
  sorry

end NUMINAMATH_GPT_silas_payment_ratio_l803_80364


namespace NUMINAMATH_GPT_min_solution_l803_80305

theorem min_solution :
  ∀ (x : ℝ), (min (1 / (1 - x)) (2 / (1 - x)) = 2 / (x - 1) - 3) → x = 7 / 3 := 
by
  sorry

end NUMINAMATH_GPT_min_solution_l803_80305


namespace NUMINAMATH_GPT_total_juice_drank_l803_80310

open BigOperators

theorem total_juice_drank (joe_juice sam_fraction alex_fraction : ℚ) :
  joe_juice = 3 / 4 ∧ sam_fraction = 1 / 2 ∧ alex_fraction = 1 / 4 → 
  sam_fraction * joe_juice + alex_fraction * joe_juice = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_total_juice_drank_l803_80310


namespace NUMINAMATH_GPT_robin_total_cost_l803_80351

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end NUMINAMATH_GPT_robin_total_cost_l803_80351


namespace NUMINAMATH_GPT_sequence_inequality_l803_80394

theorem sequence_inequality (a : ℕ → ℕ) 
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_additive : ∀ m n, a (n + m) ≤ a n + a m) 
  (N n : ℕ) 
  (h_N_ge_n : N ≥ n) : 
  a n + a N ≤ n * a 1 + N / n * a n :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l803_80394


namespace NUMINAMATH_GPT_least_value_m_n_l803_80363

theorem least_value_m_n :
  ∃ m n : ℕ, (m > 0 ∧ n > 0) ∧
            (Nat.gcd (m + n) 231 = 1) ∧
            (n^n ∣ m^m) ∧
            ¬ (m % n = 0) ∧
            m + n = 377 :=
by 
  sorry

end NUMINAMATH_GPT_least_value_m_n_l803_80363


namespace NUMINAMATH_GPT_opposite_of_3_is_neg3_l803_80340

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_3_is_neg3_l803_80340


namespace NUMINAMATH_GPT_product_of_mixed_numbers_l803_80370

theorem product_of_mixed_numbers :
  let fraction1 := (13 : ℚ) / 6
  let fraction2 := (29 : ℚ) / 9
  (fraction1 * fraction2) = 377 / 54 := 
by
  sorry

end NUMINAMATH_GPT_product_of_mixed_numbers_l803_80370


namespace NUMINAMATH_GPT_arrangement_of_numbers_l803_80353

theorem arrangement_of_numbers (numbers : Finset ℕ) 
  (h1 : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) 
  (h_sum : ∀ a b c d e f, a + b + c + d + e + f = 33)
  (h_group_sum : ∀ k1 k2 k3 k4, k1 + k2 + k3 + k4 = 26)
  : ∃ (n : ℕ), n = 2304 := by
  sorry

end NUMINAMATH_GPT_arrangement_of_numbers_l803_80353


namespace NUMINAMATH_GPT_arithmetic_sequence_s9_l803_80334

noncomputable def arithmetic_sum (a1 d n : ℝ) : ℝ :=
  n * (2*a1 + (n - 1)*d) / 2

noncomputable def general_term (a1 d n : ℝ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_s9 (a1 d : ℝ)
  (h1 : general_term a1 d 3 + general_term a1 d 4 + general_term a1 d 8 = 25) :
  arithmetic_sum a1 d 9 = 75 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_s9_l803_80334


namespace NUMINAMATH_GPT_unique_solutions_l803_80354

noncomputable def func_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)

theorem unique_solutions (f : ℝ → ℝ) :
  func_solution f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_unique_solutions_l803_80354


namespace NUMINAMATH_GPT_find_y_find_x_l803_80321

section
variables (a b : ℝ × ℝ) (x y : ℝ)

-- Definition of vectors a and b
def vec_a : ℝ × ℝ := (3, -2)
def vec_b (y : ℝ) : ℝ × ℝ := (-1, y)

-- Definition of perpendicular condition
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
-- Proof that y = -3/2 if a is perpendicular to b
theorem find_y (h : perpendicular vec_a (vec_b y)) : y = -3 / 2 :=
sorry

-- Definition of vectors a and c
def vec_c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Definition of parallel condition
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2
-- Proof that x = -15/2 if a is parallel to c
theorem find_x (h : parallel vec_a (vec_c x)) : x = -15 / 2 :=
sorry
end

end NUMINAMATH_GPT_find_y_find_x_l803_80321


namespace NUMINAMATH_GPT_decreasing_function_range_l803_80347

noncomputable def f (a x : ℝ) := a * (x^3) - x + 1

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ 0 := by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l803_80347


namespace NUMINAMATH_GPT_sum_of_ratios_l803_80315

theorem sum_of_ratios (a b c : ℤ) (h : (a * a : ℚ) / (b * b) = 32 / 63) : a + b + c = 39 :=
sorry

end NUMINAMATH_GPT_sum_of_ratios_l803_80315


namespace NUMINAMATH_GPT_number_of_real_roots_l803_80395

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then Real.exp x else -x^2 + 2.5 * x

theorem number_of_real_roots : ∃! x, f x = 0.5 * x + 1 :=
sorry

end NUMINAMATH_GPT_number_of_real_roots_l803_80395


namespace NUMINAMATH_GPT_minimum_greeting_pairs_l803_80399

def minimum_mutual_greetings (n: ℕ) (g: ℕ) : ℕ :=
  (n * g - (n * (n - 1)) / 2)

theorem minimum_greeting_pairs :
  minimum_mutual_greetings 400 200 = 200 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_greeting_pairs_l803_80399


namespace NUMINAMATH_GPT_find_star_value_l803_80331

theorem find_star_value (x : ℤ) :
  45 - (28 - (37 - (15 - x))) = 58 ↔ x = 19 :=
  by
    sorry

end NUMINAMATH_GPT_find_star_value_l803_80331


namespace NUMINAMATH_GPT_initial_strawberry_plants_l803_80320

theorem initial_strawberry_plants (P : ℕ) (h1 : 24 * P - 4 = 500) : P = 21 := 
by
  sorry

end NUMINAMATH_GPT_initial_strawberry_plants_l803_80320


namespace NUMINAMATH_GPT_last_two_digits_l803_80307

theorem last_two_digits (a b : ℕ) (n : ℕ) (h : b ≡ 25 [MOD 100]) (h_pow : (25 : ℕ) ^ n ≡ 25 [MOD 100]) :
  (33 * b ^ n) % 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_l803_80307


namespace NUMINAMATH_GPT_roots_eq_s_l803_80342

theorem roots_eq_s (n c d : ℝ) (h₁ : c * d = 6) (h₂ : c + d = n)
  (h₃ : c^2 + 1 / d = c^2 + d^2 + 1 / c): 
  (n + 217 / 6) = d^2 + 1/ c * (n + c + d)
  :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_roots_eq_s_l803_80342


namespace NUMINAMATH_GPT_composite_fraction_l803_80303

theorem composite_fraction (x : ℤ) (hx : x = 5^25) : 
  ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ a * b = x^4 + x^3 + x^2 + x + 1 :=
by sorry

end NUMINAMATH_GPT_composite_fraction_l803_80303


namespace NUMINAMATH_GPT_correct_operation_l803_80387

theorem correct_operation (a : ℝ) : a^4 / a^2 = a^2 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l803_80387


namespace NUMINAMATH_GPT_augmented_matrix_correct_l803_80374

-- Define the system of linear equations as a pair of equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x + y = 1) ∧ (3 * x - 2 * y = 0)

-- Define what it means to be the correct augmented matrix for the system
def is_augmented_matrix (A : Matrix (Fin 2) (Fin 3) ℝ) : Prop :=
  A = ![
    ![2, 1, 1],
    ![3, -2, 0]
  ]

-- The theorem states that the augmented matrix of the given system of equations is the specified matrix
theorem augmented_matrix_correct :
  ∃ x y : ℝ, system_of_equations x y ∧ is_augmented_matrix ![
    ![2, 1, 1],
    ![3, -2, 0]
  ] :=
sorry

end NUMINAMATH_GPT_augmented_matrix_correct_l803_80374


namespace NUMINAMATH_GPT_division_correct_result_l803_80360

theorem division_correct_result (x : ℝ) (h : 8 * x = 56) : 42 / x = 6 := by
  sorry

end NUMINAMATH_GPT_division_correct_result_l803_80360


namespace NUMINAMATH_GPT_population_net_increase_in_one_day_l803_80393

-- Definitions based on the conditions
def birth_rate_per_two_seconds : ℝ := 4
def death_rate_per_two_seconds : ℝ := 3
def seconds_in_a_day : ℝ := 86400

-- The main theorem to prove
theorem population_net_increase_in_one_day : 
  (birth_rate_per_two_seconds / 2 - death_rate_per_two_seconds / 2) * seconds_in_a_day = 43200 :=
by
  sorry

end NUMINAMATH_GPT_population_net_increase_in_one_day_l803_80393


namespace NUMINAMATH_GPT_sum_of_four_consecutive_integers_with_product_5040_eq_34_l803_80301

theorem sum_of_four_consecutive_integers_with_product_5040_eq_34 :
  ∃ a b c d : ℕ, a * b * c * d = 5040 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a + b + c + d) = 34 :=
sorry

end NUMINAMATH_GPT_sum_of_four_consecutive_integers_with_product_5040_eq_34_l803_80301


namespace NUMINAMATH_GPT_factorize_expression_l803_80379

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 := 
  sorry

end NUMINAMATH_GPT_factorize_expression_l803_80379


namespace NUMINAMATH_GPT_min_segments_for_7_points_l803_80392

theorem min_segments_for_7_points (points : Fin 7 → ℝ × ℝ) : 
  ∃ (segments : Finset (Fin 7 × Fin 7)), 
    (∀ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a → (a, b) ∈ segments ∨ (b, c) ∈ segments ∨ (c, a) ∈ segments) ∧
    segments.card = 9 :=
sorry

end NUMINAMATH_GPT_min_segments_for_7_points_l803_80392


namespace NUMINAMATH_GPT_average_speed_is_6_point_5_l803_80368

-- Define the given values
def total_distance : ℝ := 42
def riding_time : ℝ := 6
def break_time : ℝ := 0.5

-- Prove the average speed given the conditions
theorem average_speed_is_6_point_5 :
  (total_distance / (riding_time + break_time)) = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_is_6_point_5_l803_80368


namespace NUMINAMATH_GPT_stratified_sampling_category_A_l803_80398

def total_students_A : ℕ := 2000
def total_students_B : ℕ := 3000
def total_students_C : ℕ := 4000
def total_students : ℕ := total_students_A + total_students_B + total_students_C
def total_selected : ℕ := 900

theorem stratified_sampling_category_A :
  (total_students_A * total_selected) / total_students = 200 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_category_A_l803_80398


namespace NUMINAMATH_GPT_find_a_b_l803_80372

-- Define that the roots of the corresponding equality yield the specific conditions.
theorem find_a_b (a b : ℝ) :
    (∀ x : ℝ, x^2 + (a + 1) * x + ab > 0 ↔ (x < -1 ∨ x > 4)) →
    a = -4 ∧ b = 1 := 
by
    sorry

end NUMINAMATH_GPT_find_a_b_l803_80372


namespace NUMINAMATH_GPT_Andrena_more_than_Debelyn_l803_80369

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end NUMINAMATH_GPT_Andrena_more_than_Debelyn_l803_80369


namespace NUMINAMATH_GPT_problem1_problem2_l803_80356

-- Definition and proof statement for Problem 1
theorem problem1 (y : ℝ) : 
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 := 
by sorry

-- Definition and proof statement for Problem 2
theorem problem2 (x : ℝ) (h : x ≠ -1) :
  (1 + 2 / (x + 1)) / ((x^2 + 6 * x + 9) / (x + 1)) = 1 / (x + 3) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l803_80356


namespace NUMINAMATH_GPT_candle_height_relation_l803_80330

variables (t : ℝ)

def height_candle_A (t : ℝ) := 12 - 2 * t
def height_candle_B (t : ℝ) := 9 - 2 * t

theorem candle_height_relation : 
  12 - 2 * (15 / 4) = 3 * (9 - 2 * (15 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_candle_height_relation_l803_80330


namespace NUMINAMATH_GPT_books_left_over_after_repacking_l803_80367

def initial_boxes : ℕ := 1430
def books_per_initial_box : ℕ := 42
def weight_per_book : ℕ := 200 -- in grams
def books_per_new_box : ℕ := 45
def max_weight_per_new_box : ℕ := 9000 -- in grams (9 kg)

def total_books : ℕ := initial_boxes * books_per_initial_box

theorem books_left_over_after_repacking :
  total_books % books_per_new_box = 30 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_books_left_over_after_repacking_l803_80367


namespace NUMINAMATH_GPT_train_crosses_signal_pole_l803_80384

theorem train_crosses_signal_pole 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (time_cross_platform : ℝ) 
  (speed : ℝ) 
  (time_cross_signal_pole : ℝ) : 
  length_train = 400 → 
  length_platform = 200 → 
  time_cross_platform = 45 → 
  speed = (length_train + length_platform) / time_cross_platform → 
  time_cross_signal_pole = length_train / speed -> 
  time_cross_signal_pole = 30 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1] at h5
  -- Add the necessary calculations here
  sorry

end NUMINAMATH_GPT_train_crosses_signal_pole_l803_80384


namespace NUMINAMATH_GPT_max_value_relationship_l803_80327

theorem max_value_relationship (x y : ℝ) :
  (2005 - (x + y)^2 = 2005) → (x = -y) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_max_value_relationship_l803_80327


namespace NUMINAMATH_GPT_product_positive_l803_80336

variables {x y : ℝ}

noncomputable def non_zero (z : ℝ) := z ≠ 0

theorem product_positive (hx : non_zero x) (hy : non_zero y) 
(h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 :=
by
  sorry

end NUMINAMATH_GPT_product_positive_l803_80336


namespace NUMINAMATH_GPT_percentage_increase_ticket_price_l803_80352

-- Definitions for the conditions
def last_year_income := 100.0
def clubs_share_last_year := 0.10 * last_year_income
def rental_cost := 0.90 * last_year_income
def new_clubs_share := 0.20
def new_income := rental_cost / (1 - new_clubs_share)

-- Lean 4 theorem statement
theorem percentage_increase_ticket_price : 
  new_income = 112.5 → ((new_income - last_year_income) / last_year_income * 100) = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_ticket_price_l803_80352


namespace NUMINAMATH_GPT_closest_integer_to_99_times_9_l803_80381

theorem closest_integer_to_99_times_9 :
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  1000 ∈ choices ∧ ∀ (n : ℤ), n ∈ choices → dist 1000 ≤ dist n :=
by
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  sorry

end NUMINAMATH_GPT_closest_integer_to_99_times_9_l803_80381


namespace NUMINAMATH_GPT_calculate_g_g_2_l803_80323

def g (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

theorem calculate_g_g_2 : g (g 2) = 263 :=
by
  sorry

end NUMINAMATH_GPT_calculate_g_g_2_l803_80323


namespace NUMINAMATH_GPT_proportion_of_face_cards_l803_80350

theorem proportion_of_face_cards (p : ℝ) (h : 1 - (1 - p)^3 = 19 / 27) : p = 1 / 3 :=
sorry

end NUMINAMATH_GPT_proportion_of_face_cards_l803_80350


namespace NUMINAMATH_GPT_females_in_band_not_orchestra_l803_80332

/-- The band at Pythagoras High School has 120 female members. -/
def females_in_band : ℕ := 120

/-- The orchestra at Pythagoras High School has 70 female members. -/
def females_in_orchestra : ℕ := 70

/-- There are 45 females who are members of both the band and the orchestra. -/
def females_in_both : ℕ := 45

/-- The combined total number of students involved in either the band or orchestra or both is 250. -/
def total_students : ℕ := 250

/-- The number of females in the band who are NOT in the orchestra. -/
def females_in_band_only : ℕ := females_in_band - females_in_both

theorem females_in_band_not_orchestra : females_in_band_only = 75 := by
  sorry

end NUMINAMATH_GPT_females_in_band_not_orchestra_l803_80332


namespace NUMINAMATH_GPT_locus_equation_rectangle_perimeter_greater_l803_80382

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_locus_equation_rectangle_perimeter_greater_l803_80382


namespace NUMINAMATH_GPT_selling_price_to_equal_percentage_profit_and_loss_l803_80319

-- Definition of the variables and conditions
def cost_price : ℝ := 1500
def sp_profit_25 : ℝ := 1875
def sp_loss : ℝ := 1280

theorem selling_price_to_equal_percentage_profit_and_loss :
  ∃ SP : ℝ, SP = 1720.05 ∧
  (sp_profit_25 = cost_price * 1.25) ∧
  (sp_loss < cost_price) ∧
  (14.67 = ((SP - cost_price) / cost_price) * 100) ∧
  (14.67 = ((cost_price - sp_loss) / cost_price) * 100) :=
by
  sorry

end NUMINAMATH_GPT_selling_price_to_equal_percentage_profit_and_loss_l803_80319


namespace NUMINAMATH_GPT_students_in_college_l803_80390

variable (P S : ℕ)

def condition1 : Prop := S = 15 * P
def condition2 : Prop := S + P = 40000

theorem students_in_college (h1 : condition1 S P) (h2 : condition2 S P) : S = 37500 := by
  sorry

end NUMINAMATH_GPT_students_in_college_l803_80390


namespace NUMINAMATH_GPT_paths_A_to_D_l803_80337

noncomputable def num_paths_from_A_to_D : ℕ := 
  2 * 2 * 2 + 1

theorem paths_A_to_D : num_paths_from_A_to_D = 9 := 
by
  sorry

end NUMINAMATH_GPT_paths_A_to_D_l803_80337


namespace NUMINAMATH_GPT_license_plate_combinations_l803_80386

theorem license_plate_combinations :
  let letters := 26
  let two_other_letters := Nat.choose 25 2
  let repeated_positions := Nat.choose 4 2
  let arrange_two_letters := 2
  let first_digit_choices := 10
  let second_digit_choices := 9
  letters * two_other_letters * repeated_positions * arrange_two_letters * first_digit_choices * second_digit_choices = 8424000 :=
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l803_80386


namespace NUMINAMATH_GPT_total_games_l803_80344

-- The conditions
def working_games : ℕ := 6
def bad_games : ℕ := 5

-- The theorem to prove
theorem total_games : working_games + bad_games = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_games_l803_80344


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_equation3_solve_equation4_l803_80339

theorem solve_equation1 (x : ℝ) : (x - 1) ^ 2 = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

theorem solve_equation2 (x : ℝ) : x ^ 2 + 3 * x - 4 = 0 ↔ x = 1 ∨ x = -4 :=
by sorry

theorem solve_equation3 (x : ℝ) : 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1 / 2 ∨ x = 3 / 4 :=
by sorry

theorem solve_equation4 (x : ℝ) : 2 * x ^ 2 + 5 * x - 3 = 0 ↔ x = 1 / 2 ∨ x = -3 :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_equation3_solve_equation4_l803_80339


namespace NUMINAMATH_GPT_circle_tangent_to_y_axis_l803_80313

theorem circle_tangent_to_y_axis (m : ℝ) :
  (0 < m) → (∀ p : ℝ × ℝ, (p.1 - m)^2 + p.2^2 = 4 ↔ p.1 ^ 2 = p.2^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_y_axis_l803_80313


namespace NUMINAMATH_GPT_coffee_blend_price_l803_80377

theorem coffee_blend_price (x : ℝ) : 
  (9 * 8 + x * 12) / 20 = 8.4 → x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_coffee_blend_price_l803_80377


namespace NUMINAMATH_GPT_paint_time_for_two_people_l803_80341

/-- 
Proof Problem Statement: Prove that it would take 12 hours for two people to paint the house
given that six people can paint it in 4 hours, assuming everyone works at the same rate.
--/
theorem paint_time_for_two_people 
  (h1 : 6 * 4 = 24) 
  (h2 : ∀ (n : ℕ) (t : ℕ), n * t = 24 → t = 24 / n) : 
  2 * 12 = 24 :=
sorry

end NUMINAMATH_GPT_paint_time_for_two_people_l803_80341


namespace NUMINAMATH_GPT_total_commencement_addresses_l803_80311

-- Define the given conditions
def sandoval_addresses := 12
def sandoval_rainy_addresses := 5
def sandoval_public_holidays := 2
def sandoval_non_rainy_addresses := sandoval_addresses - sandoval_rainy_addresses

def hawkins_addresses := sandoval_addresses / 2
def sloan_addresses := sandoval_addresses + 10
def sloan_non_rainy_addresses := sloan_addresses -- assuming no rainy day details are provided

def davenport_addresses := (sandoval_non_rainy_addresses + sloan_non_rainy_addresses) / 2 - 3
def davenport_addresses_rounded := 11 -- rounding down to nearest integer as per given solution

def adkins_addresses := hawkins_addresses + davenport_addresses_rounded + 2

-- Calculate the total number of addresses
def total_addresses := sandoval_addresses + hawkins_addresses + sloan_addresses + davenport_addresses_rounded + adkins_addresses

-- The proof goal statement
theorem total_commencement_addresses : total_addresses = 70 := by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_total_commencement_addresses_l803_80311


namespace NUMINAMATH_GPT_price_each_puppy_l803_80346

def puppies_initial : ℕ := 8
def puppies_given_away : ℕ := puppies_initial / 2
def puppies_remaining_after_giveaway : ℕ := puppies_initial - puppies_given_away
def puppies_kept : ℕ := 1
def puppies_to_sell : ℕ := puppies_remaining_after_giveaway - puppies_kept
def stud_fee : ℕ := 300
def profit : ℕ := 1500
def total_amount_made : ℕ := profit + stud_fee
def price_per_puppy : ℕ := total_amount_made / puppies_to_sell

theorem price_each_puppy :
  price_per_puppy = 600 :=
sorry

end NUMINAMATH_GPT_price_each_puppy_l803_80346


namespace NUMINAMATH_GPT_sum_of_a_b_l803_80329

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end NUMINAMATH_GPT_sum_of_a_b_l803_80329


namespace NUMINAMATH_GPT_g_neg_one_l803_80366

def g (d e f x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

theorem g_neg_one {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end NUMINAMATH_GPT_g_neg_one_l803_80366


namespace NUMINAMATH_GPT_circle_center_radius_l803_80376

theorem circle_center_radius (x y : ℝ) :
  (x^2 + y^2 + 4 * x - 6 * y = 11) →
  ∃ (h k r : ℝ), h = -2 ∧ k = 3 ∧ r = 2 * Real.sqrt 6 ∧
  (x+h)^2 + (y+k)^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l803_80376


namespace NUMINAMATH_GPT_find_solutions_l803_80375

-- Definitions
def is_solution (x y z n : ℕ) : Prop :=
  x^3 + y^3 + z^3 = n * (x^2) * (y^2) * (z^2)

-- Theorem statement
theorem find_solutions :
  {sol : ℕ × ℕ × ℕ × ℕ | is_solution sol.1 sol.2.1 sol.2.2.1 sol.2.2.2} =
  {(1, 1, 1, 3), (1, 2, 3, 1), (2, 1, 3, 1)} :=
by sorry

end NUMINAMATH_GPT_find_solutions_l803_80375


namespace NUMINAMATH_GPT_ratio_of_overtime_to_regular_rate_l803_80348

def regular_rate : ℝ := 3
def regular_hours : ℕ := 40
def total_pay : ℝ := 186
def overtime_hours : ℕ := 11

theorem ratio_of_overtime_to_regular_rate 
  (r : ℝ) (h : ℕ) (T : ℝ) (h_ot : ℕ) 
  (h_r : r = regular_rate) 
  (h_h : h = regular_hours) 
  (h_T : T = total_pay)
  (h_hot : h_ot = overtime_hours) :
  (T - (h * r)) / h_ot / r = 2 := 
by {
  sorry 
}

end NUMINAMATH_GPT_ratio_of_overtime_to_regular_rate_l803_80348


namespace NUMINAMATH_GPT_distribute_cousins_l803_80324

-- Define the variables and the conditions
noncomputable def ways_to_distribute_cousins (cousins : ℕ) (rooms : ℕ) : ℕ :=
  if cousins = 5 ∧ rooms = 3 then 66 else sorry

-- State the problem
theorem distribute_cousins: ways_to_distribute_cousins 5 3 = 66 :=
by
  sorry

end NUMINAMATH_GPT_distribute_cousins_l803_80324


namespace NUMINAMATH_GPT_vasya_read_entire_book_l803_80333

theorem vasya_read_entire_book :
  let day1 := 1 / 2
  let day2 := 1 / 3 * (1 - day1)
  let days12 := day1 + day2
  let day3 := 1 / 2 * days12
  (days12 + day3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_vasya_read_entire_book_l803_80333


namespace NUMINAMATH_GPT_double_point_quadratic_l803_80309

theorem double_point_quadratic (m x1 x2 : ℝ) 
  (H1 : x1 < 1) (H2 : 1 < x2)
  (H3 : ∃ (y1 y2 : ℝ), y1 = 2 * x1 ∧ y2 = 2 * x2 ∧ y1 = x1^2 + 2 * m * x1 - m ∧ y2 = x2^2 + 2 * m * x2 - m)
  : m < 1 :=
sorry

end NUMINAMATH_GPT_double_point_quadratic_l803_80309


namespace NUMINAMATH_GPT_regular_polygon_sides_l803_80335

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end NUMINAMATH_GPT_regular_polygon_sides_l803_80335
