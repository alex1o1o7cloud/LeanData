import Mathlib

namespace NUMINAMATH_GPT_chord_bisected_by_point_of_ellipse_l365_36509

theorem chord_bisected_by_point_of_ellipse 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1)
  (bisecting_point : ∃ x y : ℝ, x = 4 ∧ y = 2) :
  ∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -8 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 :=
by
   sorry

end NUMINAMATH_GPT_chord_bisected_by_point_of_ellipse_l365_36509


namespace NUMINAMATH_GPT_kennedy_distance_to_school_l365_36584

def miles_per_gallon : ℕ := 19
def initial_gallons : ℕ := 2
def distance_softball_park : ℕ := 6
def distance_burger_restaurant : ℕ := 2
def distance_friends_house : ℕ := 4
def distance_home : ℕ := 11

def total_distance_possible : ℕ := miles_per_gallon * initial_gallons
def distance_after_school : ℕ := distance_softball_park + distance_burger_restaurant + distance_friends_house + distance_home
def distance_to_school : ℕ := total_distance_possible - distance_after_school

theorem kennedy_distance_to_school :
  distance_to_school = 15 :=
by
  sorry

end NUMINAMATH_GPT_kennedy_distance_to_school_l365_36584


namespace NUMINAMATH_GPT_sum_of_squares_five_consecutive_not_perfect_square_l365_36539

theorem sum_of_squares_five_consecutive_not_perfect_square 
  (x : ℤ) : ¬ ∃ k : ℤ, (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 = k^2 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_five_consecutive_not_perfect_square_l365_36539


namespace NUMINAMATH_GPT_average_of_b_and_c_l365_36581

theorem average_of_b_and_c (a b c : ℝ) 
  (h₁ : (a + b) / 2 = 50) 
  (h₂ : c - a = 40) : 
  (b + c) / 2 = 70 := 
by
  sorry

end NUMINAMATH_GPT_average_of_b_and_c_l365_36581


namespace NUMINAMATH_GPT_total_journey_time_l365_36541

def distance_to_post_office : ℝ := 19.999999999999996
def speed_to_post_office : ℝ := 25
def speed_back : ℝ := 4

theorem total_journey_time : 
  (distance_to_post_office / speed_to_post_office) + (distance_to_post_office / speed_back) = 5.8 :=
by
  sorry

end NUMINAMATH_GPT_total_journey_time_l365_36541


namespace NUMINAMATH_GPT_units_digit_of_6_pow_5_l365_36547

theorem units_digit_of_6_pow_5 : (6^5 % 10) = 6 := 
by sorry

end NUMINAMATH_GPT_units_digit_of_6_pow_5_l365_36547


namespace NUMINAMATH_GPT_temperature_notation_l365_36534

-- Define what it means to denote temperatures in degrees Celsius
def denote_temperature (t : ℤ) : String :=
  if t < 0 then "-" ++ toString t ++ "°C"
  else if t > 0 then "+" ++ toString t ++ "°C"
  else toString t ++ "°C"

-- Theorem statement
theorem temperature_notation (t : ℤ) (ht : t = 2) : denote_temperature t = "+2°C" :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_temperature_notation_l365_36534


namespace NUMINAMATH_GPT_sum_of_coordinates_of_intersection_l365_36594

theorem sum_of_coordinates_of_intersection :
  let A := (0, 4)
  let B := (6, 0)
  let C := (9, 3)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let line_AE := (fun x : ℚ => (-1/3) * x + 4)
  let line_CD := (fun x : ℚ => (1/6) * x + 1/2)
  let F_x := (21 : ℚ) / 3
  let F_y := line_AE F_x
  F_x + F_y = 26 / 3 := sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_intersection_l365_36594


namespace NUMINAMATH_GPT_highest_car_color_is_blue_l365_36591

def total_cars : ℕ := 24
def red_cars : ℕ := total_cars / 4
def blue_cars : ℕ := red_cars + 6
def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem highest_car_color_is_blue :
  blue_cars > red_cars ∧ blue_cars > yellow_cars :=
by sorry

end NUMINAMATH_GPT_highest_car_color_is_blue_l365_36591


namespace NUMINAMATH_GPT_candy_bar_cost_l365_36589

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def cost_of_candy_bar : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost : cost_of_candy_bar = 1 := by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l365_36589


namespace NUMINAMATH_GPT_cost_of_tax_free_items_l365_36585

/-- 
Daniel went to a shop and bought items worth Rs 25, including a 30 paise sales tax on taxable items
with a tax rate of 10%. Prove that the cost of tax-free items is Rs 22.
-/
theorem cost_of_tax_free_items (total_spent taxable_amount sales_tax rate : ℝ)
  (h1 : total_spent = 25)
  (h2 : sales_tax = 0.3)
  (h3 : rate = 0.1)
  (h4 : taxable_amount = sales_tax / rate) :
  (total_spent - taxable_amount = 22) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_tax_free_items_l365_36585


namespace NUMINAMATH_GPT_jana_winning_strategy_l365_36552

theorem jana_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (m + n) % 2 = 1 ∨ m = 1 ∨ n = 1 := sorry

end NUMINAMATH_GPT_jana_winning_strategy_l365_36552


namespace NUMINAMATH_GPT_largest_perfect_square_factor_4410_l365_36520

theorem largest_perfect_square_factor_4410 : ∀ (n : ℕ), n = 441 → (∃ k : ℕ, k^2 ∣ 4410 ∧ ∀ m : ℕ, m^2 ∣ 4410 → m^2 ≤ k^2) := 
by
  sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_4410_l365_36520


namespace NUMINAMATH_GPT_heejin_most_balls_is_volleyballs_l365_36558

def heejin_basketballs : ℕ := 3
def heejin_volleyballs : ℕ := 5
def heejin_baseballs : ℕ := 1

theorem heejin_most_balls_is_volleyballs :
  heejin_volleyballs > heejin_basketballs ∧ heejin_volleyballs > heejin_baseballs :=
by
  sorry

end NUMINAMATH_GPT_heejin_most_balls_is_volleyballs_l365_36558


namespace NUMINAMATH_GPT_value_of_A_is_18_l365_36521

theorem value_of_A_is_18
  (A B C D : ℕ)
  (h1 : A ≠ B)
  (h2 : A ≠ C)
  (h3 : A ≠ D)
  (h4 : B ≠ C)
  (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : A * B = 72)
  (h8 : C * D = 72)
  (h9 : A - B = C + D) : A = 18 :=
sorry

end NUMINAMATH_GPT_value_of_A_is_18_l365_36521


namespace NUMINAMATH_GPT_choose_one_from_ten_l365_36531

theorem choose_one_from_ten :
  Nat.choose 10 1 = 10 :=
by
  sorry

end NUMINAMATH_GPT_choose_one_from_ten_l365_36531


namespace NUMINAMATH_GPT_distance_points_l365_36538

theorem distance_points : 
  let P1 := (2, -1)
  let P2 := (7, 6)
  dist P1 P2 = Real.sqrt 74 :=
by
  sorry

end NUMINAMATH_GPT_distance_points_l365_36538


namespace NUMINAMATH_GPT_alyssa_puppies_left_l365_36593

def initial_puppies : Nat := 7
def puppies_per_puppy : Nat := 4
def given_away : Nat := 15

theorem alyssa_puppies_left :
  (initial_puppies + initial_puppies * puppies_per_puppy) - given_away = 20 := 
  by
    sorry

end NUMINAMATH_GPT_alyssa_puppies_left_l365_36593


namespace NUMINAMATH_GPT_correct_dispersion_statements_l365_36524

def statement1 (make_use_of_data : Prop) : Prop :=
make_use_of_data = true

def statement2 (multi_numerical_values : Prop) : Prop :=
multi_numerical_values = true

def statement3 (dispersion_large_value_small : Prop) : Prop :=
dispersion_large_value_small = false

theorem correct_dispersion_statements
  (make_use_of_data : Prop)
  (multi_numerical_values : Prop)
  (dispersion_large_value_small : Prop)
  (h1 : statement1 make_use_of_data)
  (h2 : statement2 multi_numerical_values)
  (h3 : statement3 dispersion_large_value_small) :
  (make_use_of_data ∧ multi_numerical_values ∧ ¬ dispersion_large_value_small) = true :=
by
  sorry

end NUMINAMATH_GPT_correct_dispersion_statements_l365_36524


namespace NUMINAMATH_GPT_quadratic_form_completion_l365_36587

theorem quadratic_form_completion (b c : ℤ)
  (h : ∀ x:ℂ, x^2 + 520*x + 600 = (x+b)^2 + c) :
  c / b = -258 :=
by sorry

end NUMINAMATH_GPT_quadratic_form_completion_l365_36587


namespace NUMINAMATH_GPT_even_odd_product_l365_36506

theorem even_odd_product (n : ℕ) (i : Fin n → Fin n) (h_perm : ∀ j : Fin n, ∃ k : Fin n, i k = j) :
  (∃ l, l % 2 = 0) → 
  ∀ (k : Fin n), ¬(i k = k) → 
  (n % 2 = 0 → (∃ m : ℤ, m + 1 % 2 = 1) ∨ (∃ m : ℤ, m + 1 % 2 = 0)) ∧ 
  (n % 2 = 1 → (∃ m : ℤ, m + 1 % 2 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_even_odd_product_l365_36506


namespace NUMINAMATH_GPT_union_comm_union_assoc_inter_distrib_union_l365_36517

variables {α : Type*} (A B C : Set α)

theorem union_comm : A ∪ B = B ∪ A := sorry

theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry

theorem inter_distrib_union : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry

end NUMINAMATH_GPT_union_comm_union_assoc_inter_distrib_union_l365_36517


namespace NUMINAMATH_GPT_wall_length_l365_36596

theorem wall_length
    (brick_length brick_width brick_height : ℝ)
    (wall_height wall_width : ℝ)
    (num_bricks : ℕ)
    (wall_length_cm : ℝ)
    (h_brick_volume : brick_length * brick_width * brick_height = 1687.5)
    (h_wall_volume :
        wall_length_cm * wall_height * wall_width
        = (brick_length * brick_width * brick_height) * num_bricks)
    (h_wall_height : wall_height = 600)
    (h_wall_width : wall_width = 22.5)
    (h_num_bricks : num_bricks = 7200) :
    wall_length_cm / 100 = 9 := 
by
  sorry

end NUMINAMATH_GPT_wall_length_l365_36596


namespace NUMINAMATH_GPT_measure_angle_4_l365_36537

theorem measure_angle_4 (m1 m2 m3 m5 m6 m4 : ℝ) 
  (h1 : m1 = 82) 
  (h2 : m2 = 34) 
  (h3 : m3 = 19) 
  (h4 : m5 = m6 + 10) 
  (h5 : m1 + m2 + m3 + m5 + m6 = 180)
  (h6 : m4 + m5 + m6 = 180) : 
  m4 = 135 :=
by
  -- Placeholder for the full proof, omitted due to instructions
  sorry

end NUMINAMATH_GPT_measure_angle_4_l365_36537


namespace NUMINAMATH_GPT_product_value_l365_36592

-- Definitions of each term
def term (n : Nat) : Rat :=
  1 + 1 / (n^2 : ℚ)

-- Define the product of these terms
def product : Rat :=
  term 1 * term 2 * term 3 * term 4 * term 5 * term 6

-- The proof problem statement that needs to be verified
theorem product_value :
  product = 16661 / 3240 :=
sorry

end NUMINAMATH_GPT_product_value_l365_36592


namespace NUMINAMATH_GPT_length_FD_of_folded_square_l365_36543

theorem length_FD_of_folded_square :
  let A := (0, 0)
  let B := (8, 0)
  let D := (0, 8)
  let C := (8, 8)
  let E := (6, 0)
  let F := (8, 8 - (FD : ℝ))
  (ABCD_square : ∀ {x y : ℝ}, (x = 0 ∨ x = 8) ∧ (y = 0 ∨ y = 8)) →  
  let DE := (6 - 0 : ℝ)
  let Pythagorean_statement := (8 - FD) ^ 2 = FD ^ 2 + 6 ^ 2
  ∃ FD : ℝ, FD = 7 / 4 :=
sorry

end NUMINAMATH_GPT_length_FD_of_folded_square_l365_36543


namespace NUMINAMATH_GPT_volume_at_10_l365_36562

noncomputable def gas_volume (T : ℝ) : ℝ :=
  if T = 30 then 40 else 40 - (30 - T) / 5 * 5

theorem volume_at_10 :
  gas_volume 10 = 20 :=
by
  simp [gas_volume]
  sorry

end NUMINAMATH_GPT_volume_at_10_l365_36562


namespace NUMINAMATH_GPT_price_of_paint_models_max_boxes_of_paint_A_l365_36508

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end NUMINAMATH_GPT_price_of_paint_models_max_boxes_of_paint_A_l365_36508


namespace NUMINAMATH_GPT_certain_event_l365_36595

theorem certain_event (a : ℝ) : a^2 ≥ 0 := 
sorry

end NUMINAMATH_GPT_certain_event_l365_36595


namespace NUMINAMATH_GPT_polynomial_expansion_l365_36545

theorem polynomial_expansion :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a_1 * x^4 + a_2 * x^3 + a_3 * x^2 + 16 * x + 4) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l365_36545


namespace NUMINAMATH_GPT_min_PM_PN_min_PM_squared_PN_squared_l365_36526

noncomputable def min_value_PM_PN := 3 * Real.sqrt 5

noncomputable def min_value_PM_squared_PN_squared := 229 / 10

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 5⟩
def N : Point := ⟨-2, 4⟩

def on_line (P : Point) : Prop :=
  P.x - 2 * P.y + 3 = 0

theorem min_PM_PN {P : Point} (h : on_line P) :
  dist (P.x, P.y) (M.x, M.y) + dist (P.x, P.y) (N.x, N.y) = min_value_PM_PN := sorry

theorem min_PM_squared_PN_squared {P : Point} (h : on_line P) :
  (dist (P.x, P.y) (M.x, M.y))^2 + (dist (P.x, P.y) (N.x, N.y))^2 = min_value_PM_squared_PN_squared := sorry

end NUMINAMATH_GPT_min_PM_PN_min_PM_squared_PN_squared_l365_36526


namespace NUMINAMATH_GPT_probability_of_three_different_colors_draw_l365_36527

open ProbabilityTheory

def number_of_blue_chips : ℕ := 4
def number_of_green_chips : ℕ := 5
def number_of_red_chips : ℕ := 6
def number_of_yellow_chips : ℕ := 3
def total_number_of_chips : ℕ := 18

def P_B : ℚ := number_of_blue_chips / total_number_of_chips
def P_G : ℚ := number_of_green_chips / total_number_of_chips
def P_R : ℚ := number_of_red_chips / total_number_of_chips
def P_Y : ℚ := number_of_yellow_chips / total_number_of_chips

def P_different_colors : ℚ := 2 * ((P_B * P_G + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G + P_R * P_Y) +
                                    (P_B * P_R + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G))

theorem probability_of_three_different_colors_draw :
  P_different_colors = 141 / 162 :=
by
  -- Placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_probability_of_three_different_colors_draw_l365_36527


namespace NUMINAMATH_GPT_jebb_expense_l365_36556

-- Define the costs
def seafood_platter := 45.0
def rib_eye_steak := 38.0
def vintage_wine_glass := 18.0
def chocolate_dessert := 12.0

-- Define the rules and discounts
def discount_percentage := 0.10
def service_fee_12 := 0.12
def service_fee_15 := 0.15
def tip_percentage := 0.20

-- Total food and wine cost
def total_food_and_wine_cost := 
  seafood_platter + rib_eye_steak + (2 * vintage_wine_glass) + chocolate_dessert

-- Total food cost excluding wine
def food_cost_excluding_wine := 
  seafood_platter + rib_eye_steak + chocolate_dessert

-- 10% discount on food cost excluding wine
def discount_amount := discount_percentage * food_cost_excluding_wine
def reduced_food_cost := food_cost_excluding_wine - discount_amount

-- New total cost before applying the service fee
def total_cost_before_service_fee := reduced_food_cost + (2 * vintage_wine_glass)

-- Determine the service fee based on cost
def service_fee := 
  if total_cost_before_service_fee > 80.0 then 
    service_fee_15 * total_cost_before_service_fee 
  else if total_cost_before_service_fee >= 50.0 then 
    service_fee_12 * total_cost_before_service_fee 
  else 
    0.0

-- Total cost after discount and service fee
def total_cost_after_service_fee := total_cost_before_service_fee + service_fee

-- Tip amount (20% of total cost after discount and service fee)
def tip_amount := tip_percentage * total_cost_after_service_fee

-- Total amount Jebb spent
def total_amount_spent := total_cost_after_service_fee + tip_amount

-- Lean theorem statement
theorem jebb_expense :
  total_amount_spent = 167.67 :=
by
  -- prove the theorem here
  sorry

end NUMINAMATH_GPT_jebb_expense_l365_36556


namespace NUMINAMATH_GPT_decimal_expansion_2023rd_digit_l365_36522

theorem decimal_expansion_2023rd_digit 
  (x : ℚ) 
  (hx : x = 7 / 26) 
  (decimal_expansion : ℕ → ℕ)
  (hdecimal : ∀ n : ℕ, decimal_expansion n = if n % 12 = 0 
                        then 2 
                        else if n % 12 = 1 
                          then 7 
                          else if n % 12 = 2 
                            then 9 
                            else if n % 12 = 3 
                              then 2 
                              else if n % 12 = 4 
                                then 3 
                                else if n % 12 = 5 
                                  then 0 
                                  else if n % 12 = 6 
                                    then 7 
                                    else if n % 12 = 7 
                                      then 6 
                                      else if n % 12 = 8 
                                        then 9 
                                        else if n % 12 = 9 
                                          then 2 
                                          else if n % 12 = 10 
                                            then 3 
                                            else 0) :
  decimal_expansion 2023 = 0 :=
sorry

end NUMINAMATH_GPT_decimal_expansion_2023rd_digit_l365_36522


namespace NUMINAMATH_GPT_total_buttons_l365_36532

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end NUMINAMATH_GPT_total_buttons_l365_36532


namespace NUMINAMATH_GPT_age_of_James_when_Thomas_reaches_current_age_l365_36554
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end NUMINAMATH_GPT_age_of_James_when_Thomas_reaches_current_age_l365_36554


namespace NUMINAMATH_GPT_number_of_routes_jack_to_jill_l365_36597

def num_routes_avoiding (start goal avoid : ℕ × ℕ) : ℕ := sorry

theorem number_of_routes_jack_to_jill : 
  num_routes_avoiding (0,0) (3,2) (1,1) = 4 :=
sorry

end NUMINAMATH_GPT_number_of_routes_jack_to_jill_l365_36597


namespace NUMINAMATH_GPT_apple_juice_less_than_cherry_punch_l365_36563

def orange_punch : ℝ := 4.5
def total_punch : ℝ := 21
def cherry_punch : ℝ := 2 * orange_punch
def combined_punch : ℝ := orange_punch + cherry_punch
def apple_juice : ℝ := total_punch - combined_punch

theorem apple_juice_less_than_cherry_punch : cherry_punch - apple_juice = 1.5 := by
  sorry

end NUMINAMATH_GPT_apple_juice_less_than_cherry_punch_l365_36563


namespace NUMINAMATH_GPT_negation_of_statement_l365_36573

theorem negation_of_statement :
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n > x^2) ↔ (∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2) := by
sorry

end NUMINAMATH_GPT_negation_of_statement_l365_36573


namespace NUMINAMATH_GPT_max_xy_l365_36513

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 16) : 
  xy ≤ 32 :=
sorry

end NUMINAMATH_GPT_max_xy_l365_36513


namespace NUMINAMATH_GPT_height_difference_is_9_l365_36560

-- Definitions of the height of Petronas Towers and Empire State Building.
def height_Petronas : ℕ := 452
def height_EmpireState : ℕ := 443

-- Definition stating the height difference.
def height_difference := height_Petronas - height_EmpireState

-- Proving the height difference is 9 meters.
theorem height_difference_is_9 : height_difference = 9 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_height_difference_is_9_l365_36560


namespace NUMINAMATH_GPT_find_k_l365_36512

theorem find_k (x y k : ℤ) (h₁ : x = -3) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) : k = 6 :=
by
  rw [h₁, h₂] at h₃
  -- Substitute x and y in the equation
  -- 2 * (-3) + k * 2 = 6
  sorry

end NUMINAMATH_GPT_find_k_l365_36512


namespace NUMINAMATH_GPT_reciprocal_of_neg_two_l365_36565

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end NUMINAMATH_GPT_reciprocal_of_neg_two_l365_36565


namespace NUMINAMATH_GPT_perpendicular_lines_l365_36567

def line1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y) ∧ (∀ x y : ℝ, line2 a x y) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, 
    (line1 a x1 y1) ∧ (line2 a x2 y2) → 
    (-a / 2) * (-1 / (a - 1)) = -1) → a = 2 / 3 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_l365_36567


namespace NUMINAMATH_GPT_gcd_1037_425_l365_36529

theorem gcd_1037_425 : Int.gcd 1037 425 = 17 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1037_425_l365_36529


namespace NUMINAMATH_GPT_inscribed_circle_radius_l365_36579

theorem inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) : 
  ∃ r : ℝ, r = (105 * Real.sqrt 274) / 274 := 
by 
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l365_36579


namespace NUMINAMATH_GPT_volume_parallelepiped_eq_20_l365_36502

theorem volume_parallelepiped_eq_20 (k : ℝ) (h : k > 0) (hvol : abs (3 * k^2 - 7 * k - 6) = 20) :
  k = 13 / 3 :=
sorry

end NUMINAMATH_GPT_volume_parallelepiped_eq_20_l365_36502


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l365_36564

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := 14
  let c := 5
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l365_36564


namespace NUMINAMATH_GPT_math_problem_l365_36599

noncomputable def proof_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  a^2 + 4 * b^2 + 1 / (a * b) ≥ 4

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : proof_problem a b ha hb :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l365_36599


namespace NUMINAMATH_GPT_students_wearing_other_colors_l365_36516

variable (total_students blue_percentage red_percentage green_percentage : ℕ)
variable (h_total : total_students = 600)
variable (h_blue : blue_percentage = 45)
variable (h_red : red_percentage = 23)
variable (h_green : green_percentage = 15)

theorem students_wearing_other_colors :
  (total_students * (100 - (blue_percentage + red_percentage + green_percentage)) / 100 = 102) :=
by
  sorry

end NUMINAMATH_GPT_students_wearing_other_colors_l365_36516


namespace NUMINAMATH_GPT_min_sticks_cover_200cm_l365_36546

def length_covered (n6 n7 : ℕ) : ℕ :=
  6 * n6 + 7 * n7

theorem min_sticks_cover_200cm :
  ∃ (n6 n7 : ℕ), length_covered n6 n7 = 200 ∧ (∀ (m6 m7 : ℕ), (length_covered m6 m7 = 200 → m6 + m7 ≥ n6 + n7)) ∧ (n6 + n7 = 29) :=
sorry

end NUMINAMATH_GPT_min_sticks_cover_200cm_l365_36546


namespace NUMINAMATH_GPT_speed_of_the_stream_l365_36523

theorem speed_of_the_stream (d v_s : ℝ) :
  (∀ (t_up t_down : ℝ), t_up = d / (57 - v_s) ∧ t_down = d / (57 + v_s) ∧ t_up = 2 * t_down) →
  v_s = 19 := by
  sorry

end NUMINAMATH_GPT_speed_of_the_stream_l365_36523


namespace NUMINAMATH_GPT_find_x_intercept_of_perpendicular_line_l365_36572

noncomputable def line_y_intercept : ℝ × ℝ := (0, 3)
noncomputable def given_line (x y : ℝ) : Prop := 2 * x + y = 3
noncomputable def x_intercept_of_perpendicular_line : ℝ × ℝ := (-6, 0)

theorem find_x_intercept_of_perpendicular_line :
  (∀ (x y : ℝ), given_line x y → (slope_of_perpendicular_line : ℝ) = 1/2 ∧ 
  ∀ (b : ℝ), line_y_intercept = (0, b) → ∀ (y : ℝ), y = 1/2 * x + b → (x, 0) = x_intercept_of_perpendicular_line) :=
sorry

end NUMINAMATH_GPT_find_x_intercept_of_perpendicular_line_l365_36572


namespace NUMINAMATH_GPT_pyramid_surface_area_l365_36553

theorem pyramid_surface_area
  (base_side_length : ℝ)
  (peak_height : ℝ)
  (base_area : ℝ)
  (slant_height : ℝ)
  (triangular_face_area : ℝ)
  (total_surface_area : ℝ)
  (h1 : base_side_length = 10)
  (h2 : peak_height = 12)
  (h3 : base_area = base_side_length ^ 2)
  (h4 : slant_height = Real.sqrt (peak_height ^ 2 + (base_side_length / 2) ^ 2))
  (h5 : triangular_face_area = 0.5 * base_side_length * slant_height)
  (h6 : total_surface_area = base_area + 4 * triangular_face_area)
  : total_surface_area = 360 := 
sorry

end NUMINAMATH_GPT_pyramid_surface_area_l365_36553


namespace NUMINAMATH_GPT_age_of_new_teacher_l365_36568

-- Definitions of conditions
def avg_age_20_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 49 * 20

def avg_age_21_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 48 * 21

-- The proof goal
theorem age_of_new_teacher (sum_age_20 : ℕ) (sum_age_21 : ℕ) (h1 : avg_age_20_teachers sum_age_20) (h2 : avg_age_21_teachers sum_age_21) : 
  sum_age_21 - sum_age_20 = 28 :=
sorry

end NUMINAMATH_GPT_age_of_new_teacher_l365_36568


namespace NUMINAMATH_GPT_part1_solution_set_eq_part2_a_range_l365_36586

theorem part1_solution_set_eq : {x : ℝ | |2 * x + 1| + |2 * x - 3| ≤ 6} = Set.Icc (-1) 2 :=
by sorry

theorem part2_a_range (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, |2 * x + 1| + |2 * x - 3| < |a - 2|) → 6 < a :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_eq_part2_a_range_l365_36586


namespace NUMINAMATH_GPT_gcd_324_243_135_l365_36574

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 := by
  sorry

end NUMINAMATH_GPT_gcd_324_243_135_l365_36574


namespace NUMINAMATH_GPT_next_hexagon_dots_l365_36590

theorem next_hexagon_dots (base_dots : ℕ) (increment : ℕ) : base_dots = 2 → increment = 2 → 
  (2 + 6*2) + 6*(2*2) + 6*(3*2) + 6*(4*2) = 122 := 
by
  intros hbd hi
  sorry

end NUMINAMATH_GPT_next_hexagon_dots_l365_36590


namespace NUMINAMATH_GPT_reporters_not_covering_politics_l365_36542

theorem reporters_not_covering_politics (P_X P_Y P_Z intlPol otherPol econOthers : ℝ)
  (h1 : P_X = 0.15) (h2 : P_Y = 0.10) (h3 : P_Z = 0.08)
  (h4 : otherPol = 0.50) (h5 : intlPol = 0.05) (h6 : econOthers = 0.02) :
  (1 - (P_X + P_Y + P_Z + intlPol + otherPol + econOthers)) = 0.10 := by
  sorry

end NUMINAMATH_GPT_reporters_not_covering_politics_l365_36542


namespace NUMINAMATH_GPT_problem_l365_36505

open Real

theorem problem (x y : ℝ) (h1 : 3 * x + 2 * y = 8) (h2 : 2 * x + 3 * y = 11) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 2041 / 25 :=
sorry

end NUMINAMATH_GPT_problem_l365_36505


namespace NUMINAMATH_GPT_jessica_balloon_count_l365_36500

theorem jessica_balloon_count :
  (∀ (joan_initial_balloon_count sally_popped_balloon_count total_balloon_count: ℕ),
  joan_initial_balloon_count = 9 →
  sally_popped_balloon_count = 5 →
  total_balloon_count = 6 →
  ∃ (jessica_balloon_count: ℕ),
    jessica_balloon_count = total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count) →
    jessica_balloon_count = 2) :=
by
  intros joan_initial_balloon_count sally_popped_balloon_count total_balloon_count j1 j2 t1
  use total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count)
  sorry

end NUMINAMATH_GPT_jessica_balloon_count_l365_36500


namespace NUMINAMATH_GPT_usual_time_72_l365_36598

namespace TypicalTimeProof

variables (S T : ℝ) 

theorem usual_time_72 (h : T ≠ 0) (h2 : 0.75 * S ≠ 0) (h3 : 4 * T = 3 * (T + 24)) : T = 72 := by
  sorry

end TypicalTimeProof

end NUMINAMATH_GPT_usual_time_72_l365_36598


namespace NUMINAMATH_GPT_math_problem_l365_36569

theorem math_problem 
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : ∀ x, (x < -2 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 )) :
  a + 2 * b + 3 * c = 86 :=
sorry

end NUMINAMATH_GPT_math_problem_l365_36569


namespace NUMINAMATH_GPT_bus_speed_kmph_l365_36504

theorem bus_speed_kmph : 
  let distance := 600.048 
  let time := 30
  (distance / time) * 3.6 = 72.006 :=
by
  sorry

end NUMINAMATH_GPT_bus_speed_kmph_l365_36504


namespace NUMINAMATH_GPT_sum_of_coordinates_after_reflections_l365_36501

theorem sum_of_coordinates_after_reflections :
  let A := (3, 2)
  let B := (9, 18)
  let N := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let reflect_y (P : ℤ × ℤ) := (-P.1, P.2)
  let reflect_x (P : ℤ × ℤ) := (P.1, -P.2)
  let N' := reflect_y N
  let N'' := reflect_x N'
  N''.1 + N''.2 = -16 := by sorry

end NUMINAMATH_GPT_sum_of_coordinates_after_reflections_l365_36501


namespace NUMINAMATH_GPT_abc_less_than_one_l365_36525

variables {a b c : ℝ}

theorem abc_less_than_one (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1: a^2 < b) (h2: b^2 < c) (h3: c^2 < a) : a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end NUMINAMATH_GPT_abc_less_than_one_l365_36525


namespace NUMINAMATH_GPT_value_of_y_l365_36551

theorem value_of_y (x y : ℝ) (h1 : x ^ (2 * y) = 81) (h2 : x = 9) : y = 1 :=
sorry

end NUMINAMATH_GPT_value_of_y_l365_36551


namespace NUMINAMATH_GPT_mass_percentage_Al_in_mixture_l365_36570

/-- Define molar masses for the respective compounds -/
def molar_mass_AlCl3 : ℝ := 133.33
def molar_mass_Al2SO4_3 : ℝ := 342.17
def molar_mass_AlOH3 : ℝ := 78.01

/-- Define masses of respective compounds given in grams -/
def mass_AlCl3 : ℝ := 50
def mass_Al2SO4_3 : ℝ := 70
def mass_AlOH3 : ℝ := 40

/-- Define molar mass of Al -/
def molar_mass_Al : ℝ := 26.98

theorem mass_percentage_Al_in_mixture :
  (mass_AlCl3 / molar_mass_AlCl3 * molar_mass_Al +
   mass_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al) +
   mass_AlOH3 / molar_mass_AlOH3 * molar_mass_Al) / 
  (mass_AlCl3 + mass_Al2SO4_3 + mass_AlOH3) * 100 
  = 21.87 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_Al_in_mixture_l365_36570


namespace NUMINAMATH_GPT_scientific_notation_of_114_trillion_l365_36557

theorem scientific_notation_of_114_trillion :
  (114 : ℝ) * 10^12 = (1.14 : ℝ) * 10^14 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_114_trillion_l365_36557


namespace NUMINAMATH_GPT_find_n_l365_36576

theorem find_n (n : ℕ) (hnpos : 0 < n)
  (hsquare : ∃ k : ℕ, k^2 = n^4 + 2*n^3 + 5*n^2 + 12*n + 5) :
  n = 1 ∨ n = 2 := 
sorry

end NUMINAMATH_GPT_find_n_l365_36576


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_numbers_l365_36566

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_numbers_l365_36566


namespace NUMINAMATH_GPT_dress_design_count_l365_36578

-- Definitions of the given conditions
def number_of_colors : Nat := 4
def number_of_patterns : Nat := 5

-- Statement to prove the total number of unique dress designs
theorem dress_design_count :
  number_of_colors * number_of_patterns = 20 := by
  sorry

end NUMINAMATH_GPT_dress_design_count_l365_36578


namespace NUMINAMATH_GPT_find_n_l365_36577

theorem find_n
  (c d : ℝ)
  (H1 : 450 * c + 300 * d = 300 * c + 375 * d)
  (H2 : ∃ t1 t2 t3 : ℝ, t1 = 4 ∧ t2 = 1 ∧ t3 = n ∧ 75 * 4 * (c + d) = 900 * c + t3 * d)
  : n = 600 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l365_36577


namespace NUMINAMATH_GPT_middle_number_of_ratio_l365_36503

theorem middle_number_of_ratio (x : ℝ) (h : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862) : 2 * x = 14 :=
sorry

end NUMINAMATH_GPT_middle_number_of_ratio_l365_36503


namespace NUMINAMATH_GPT_amusement_park_weekly_revenue_l365_36583

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end NUMINAMATH_GPT_amusement_park_weekly_revenue_l365_36583


namespace NUMINAMATH_GPT_angle_measure_is_fifty_l365_36582

theorem angle_measure_is_fifty (x : ℝ) :
  (90 - x = (1 / 2) * (180 - x) - 25) → x = 50 := by
  intro h
  sorry

end NUMINAMATH_GPT_angle_measure_is_fifty_l365_36582


namespace NUMINAMATH_GPT_infinite_rel_prime_set_of_form_2n_minus_3_l365_36528

theorem infinite_rel_prime_set_of_form_2n_minus_3 : ∃ S : Set ℕ, (∀ x ∈ S, ∃ n : ℕ, x = 2^n - 3) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → Nat.gcd x y = 1) ∧ S.Infinite := 
by
  sorry

end NUMINAMATH_GPT_infinite_rel_prime_set_of_form_2n_minus_3_l365_36528


namespace NUMINAMATH_GPT_geom_seq_common_ratio_l365_36510

theorem geom_seq_common_ratio (S_3 S_6 : ℕ) (h1 : S_3 = 7) (h2 : S_6 = 63) : 
  ∃ q : ℕ, q = 2 := 
by
  sorry

end NUMINAMATH_GPT_geom_seq_common_ratio_l365_36510


namespace NUMINAMATH_GPT_range_of_x_for_f_ln_x_gt_f_1_l365_36540

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

noncomputable def is_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_x_for_f_ln_x_gt_f_1
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_dec : is_decreasing_on_nonneg f)
  (hf_condition : ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e) :
  ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e := sorry

end NUMINAMATH_GPT_range_of_x_for_f_ln_x_gt_f_1_l365_36540


namespace NUMINAMATH_GPT_sum_first_n_terms_l365_36530

-- Define the sequence a_n
def geom_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = r * a n

-- Define the main conditions from the problem
axiom a7_cond (a : ℕ → ℕ) : a 7 = 8 * a 4
axiom arithmetic_seq_cond (a : ℕ → ℕ) : (1 / 2 : ℝ) * a 2 < (a 3 - 4) ∧ (a 3 - 4) < (a 4 - 12)

-- Define the sequences a_n and b_n using the conditions
def a_n (n : ℕ) : ℕ := 2^(n + 1)
def b_n (n : ℕ) : ℤ := (-1)^n * (Int.ofNat (n + 1))

-- Define the sum of the first n terms of b_n
noncomputable def T_n (n : ℕ) : ℤ :=
  (Finset.range n).sum b_n

-- Main theorem statement
theorem sum_first_n_terms (k : ℕ) : |T_n k| = 20 → k = 40 ∨ k = 37 :=
sorry

end NUMINAMATH_GPT_sum_first_n_terms_l365_36530


namespace NUMINAMATH_GPT_find_n_l365_36561

noncomputable def n (n : ℕ) : Prop :=
  lcm n 12 = 42 ∧ gcd n 12 = 6

theorem find_n (n : ℕ) (h : lcm n 12 = 42) (h1 : gcd n 12 = 6) : n = 21 :=
by sorry

end NUMINAMATH_GPT_find_n_l365_36561


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l365_36533

theorem solve_eq1 (y : ℝ) : 6 - 3 * y = 15 + 6 * y ↔ y = -1 := by
  sorry

theorem solve_eq2 (x : ℝ) : (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 2 ↔ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l365_36533


namespace NUMINAMATH_GPT_simplify_and_evaluate_l365_36515

theorem simplify_and_evaluate (a : ℕ) (h : a = 2023) : (a + 1) / a / (a - 1 / a) = 1 / 2022 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l365_36515


namespace NUMINAMATH_GPT_AY_is_2_sqrt_55_l365_36550

noncomputable def AY_length : ℝ :=
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  2 * Real.sqrt (rA^2 + BD^2)

theorem AY_is_2_sqrt_55 :
  AY_length = 2 * Real.sqrt 55 :=
by
  -- Assuming the given problem's conditions.
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  show AY_length = 2 * Real.sqrt 55
  sorry

end NUMINAMATH_GPT_AY_is_2_sqrt_55_l365_36550


namespace NUMINAMATH_GPT_find_m_l365_36511

theorem find_m (x m : ℤ) (h : x = -1 ∧ x - 2 * m = 9) : m = -5 :=
sorry

end NUMINAMATH_GPT_find_m_l365_36511


namespace NUMINAMATH_GPT_range_of_x_plus_y_l365_36559

open Real

theorem range_of_x_plus_y (x y : ℝ) (h : x - sqrt (x + 1) = sqrt (y + 1) - y) :
  -sqrt 5 + 1 ≤ x + y ∧ x + y ≤ sqrt 5 + 1 :=
by sorry

end NUMINAMATH_GPT_range_of_x_plus_y_l365_36559


namespace NUMINAMATH_GPT_ratio_of_powers_l365_36536

theorem ratio_of_powers (a x : ℝ) (h : a^(2 * x) = Real.sqrt 2 - 1) : (a^(3 * x) + a^(-3 * x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_powers_l365_36536


namespace NUMINAMATH_GPT_capsule_depth_equation_l365_36588

theorem capsule_depth_equation (x y z : ℝ) (h : y = 4 * x + z) : y = 4 * x + z := 
by 
  exact h

end NUMINAMATH_GPT_capsule_depth_equation_l365_36588


namespace NUMINAMATH_GPT_num_students_59_l365_36535

theorem num_students_59 (apples : ℕ) (taken_each : ℕ) (students : ℕ) 
  (h_apples : apples = 120) 
  (h_taken_each : taken_each = 2) 
  (h_students_divisors : ∀ d, d = 59 → d ∣ (apples / taken_each)) : students = 59 :=
sorry

end NUMINAMATH_GPT_num_students_59_l365_36535


namespace NUMINAMATH_GPT_brass_selling_price_l365_36571

noncomputable def copper_price : ℝ := 0.65
noncomputable def zinc_price : ℝ := 0.30
noncomputable def total_weight_brass : ℝ := 70
noncomputable def weight_copper : ℝ := 30
noncomputable def weight_zinc := total_weight_brass - weight_copper
noncomputable def cost_copper := weight_copper * copper_price
noncomputable def cost_zinc := weight_zinc * zinc_price
noncomputable def total_cost := cost_copper + cost_zinc
noncomputable def selling_price_per_pound := total_cost / total_weight_brass

theorem brass_selling_price :
  selling_price_per_pound = 0.45 :=
by
  sorry

end NUMINAMATH_GPT_brass_selling_price_l365_36571


namespace NUMINAMATH_GPT_common_tangent_slope_l365_36518

theorem common_tangent_slope (a m : ℝ) : 
  ((∃ a, ∃ m, l = (2 * a) ∧ l = (3 * m^2) ∧ a^2 = 2 * m^3) → (l = 0 ∨ l = 64 / 27)) := 
sorry

end NUMINAMATH_GPT_common_tangent_slope_l365_36518


namespace NUMINAMATH_GPT_average_num_divisors_2019_l365_36580

def num_divisors (n : ℕ) : ℕ :=
  (n.divisors).card

theorem average_num_divisors_2019 :
  1 / 2019 * (Finset.sum (Finset.range 2020) num_divisors) = 15682 / 2019 :=
by
  sorry

end NUMINAMATH_GPT_average_num_divisors_2019_l365_36580


namespace NUMINAMATH_GPT_train_cross_time_l365_36544

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 255.03
noncomputable def train_speed_ms : ℝ := 12.5
noncomputable def distance_to_travel : ℝ := train_length + bridge_length
noncomputable def expected_time : ℝ := 30.0024

theorem train_cross_time :
  (distance_to_travel / train_speed_ms) = expected_time :=
by sorry

end NUMINAMATH_GPT_train_cross_time_l365_36544


namespace NUMINAMATH_GPT_james_problem_l365_36575

def probability_at_least_two_green_apples (total: ℕ) (red: ℕ) (green: ℕ) (yellow: ℕ) (choices: ℕ) : ℚ :=
  let favorable_outcomes := (Nat.choose green 2) * (Nat.choose (total - green) 1) + (Nat.choose green 3)
  let total_outcomes := Nat.choose total choices
  favorable_outcomes / total_outcomes

theorem james_problem : probability_at_least_two_green_apples 10 5 3 2 3 = 11 / 60 :=
by sorry

end NUMINAMATH_GPT_james_problem_l365_36575


namespace NUMINAMATH_GPT_probability_two_red_balls_l365_36549

open Nat

theorem probability_two_red_balls (total_balls red_balls blue_balls green_balls balls_picked : Nat) 
  (total_eq : total_balls = red_balls + blue_balls + green_balls) 
  (red_eq : red_balls = 7) 
  (blue_eq : blue_balls = 5) 
  (green_eq : green_balls = 4) 
  (picked_eq : balls_picked = 2) :
  (choose red_balls balls_picked) / (choose total_balls balls_picked) = 7 / 40 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_red_balls_l365_36549


namespace NUMINAMATH_GPT_log_inequalities_l365_36548

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_inequalities : c < b ∧ b < a :=
  sorry

end NUMINAMATH_GPT_log_inequalities_l365_36548


namespace NUMINAMATH_GPT_solve_abs_inequality_l365_36519

theorem solve_abs_inequality (x : ℝ) (h : 1 < |x - 1| ∧ |x - 1| < 4) : (-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l365_36519


namespace NUMINAMATH_GPT_quadratic_root_range_l365_36514

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end NUMINAMATH_GPT_quadratic_root_range_l365_36514


namespace NUMINAMATH_GPT_prove_frac_addition_l365_36555

def frac_addition_correct : Prop :=
  (3 / 8 + 9 / 12 = 9 / 8)

theorem prove_frac_addition : frac_addition_correct :=
  by
  -- We assume the necessary fractions and their properties.
  sorry

end NUMINAMATH_GPT_prove_frac_addition_l365_36555


namespace NUMINAMATH_GPT_transformation_correct_l365_36507

theorem transformation_correct (a b c : ℝ) : a = b → ac = bc :=
by sorry

end NUMINAMATH_GPT_transformation_correct_l365_36507
