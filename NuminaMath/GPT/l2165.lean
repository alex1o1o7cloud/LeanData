import Mathlib

namespace distance_between_5th_and_29th_red_light_in_feet_l2165_216510

-- Define the repeating pattern length and individual light distance
def pattern_length := 7
def red_light_positions := {k | k % pattern_length < 3}
def distance_between_lights := 8 / 12  -- converting inches to feet

-- Positions of the 5th and 29th red lights in terms of pattern repetition
def position_of_nth_red_light (n : ℕ) : ℕ :=
  ((n-1) / 3) * pattern_length + (n-1) % 3 + 1

def position_5th_red_light := position_of_nth_red_light 5
def position_29th_red_light := position_of_nth_red_light 29

theorem distance_between_5th_and_29th_red_light_in_feet :
  (position_29th_red_light - position_5th_red_light - 1) * distance_between_lights = 37 := by
  sorry

end distance_between_5th_and_29th_red_light_in_feet_l2165_216510


namespace stock_price_is_108_l2165_216509

noncomputable def dividend_income (FV : ℕ) (D : ℕ) : ℕ :=
  FV * D / 100

noncomputable def face_value_of_stock (I : ℕ) (D : ℕ) : ℕ :=
  I * 100 / D

noncomputable def price_of_stock (Inv : ℕ) (FV : ℕ) : ℕ :=
  Inv * 100 / FV

theorem stock_price_is_108 (I D Inv : ℕ) (hI : I = 450) (hD : D = 10) (hInv : Inv = 4860) :
  price_of_stock Inv (face_value_of_stock I D) = 108 :=
by
  -- Placeholder for proof
  sorry

end stock_price_is_108_l2165_216509


namespace Sam_drinks_l2165_216599

theorem Sam_drinks (juice_don : ℚ) (fraction_sam : ℚ) 
  (h1 : juice_don = 3 / 7) (h2 : fraction_sam = 4 / 5) : 
  (fraction_sam * juice_don = 12 / 35) :=
by
  sorry

end Sam_drinks_l2165_216599


namespace find_number_l2165_216587

theorem find_number (x : ℝ) 
(h : x * 13.26 + x * 9.43 + x * 77.31 = 470) : 
x = 4.7 := 
sorry

end find_number_l2165_216587


namespace find_number_l2165_216581

theorem find_number (N : ℝ) (h : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N) : N = 180 :=
by 
  sorry

end find_number_l2165_216581


namespace point_inside_circle_l2165_216535

theorem point_inside_circle (r OP : ℝ) (h₁ : r = 3) (h₂ : OP = 2) : OP < r :=
by
  sorry

end point_inside_circle_l2165_216535


namespace find_n_tan_eq_l2165_216543

theorem find_n_tan_eq (n : ℤ) (h₁ : -180 < n) (h₂ : n < 180) 
  (h₃ : Real.tan (n * (Real.pi / 180)) = Real.tan (276 * (Real.pi / 180))) : 
  n = 96 :=
sorry

end find_n_tan_eq_l2165_216543


namespace miles_mike_ride_l2165_216527

theorem miles_mike_ride
  (cost_per_mile : ℝ) (start_fee : ℝ) (bridge_toll : ℝ)
  (annie_miles : ℝ) (annie_total_cost : ℝ)
  (mike_total_cost : ℝ) (M : ℝ)
  (h1 : cost_per_mile = 0.25)
  (h2 : start_fee = 2.50)
  (h3 : bridge_toll = 5.00)
  (h4 : annie_miles = 26)
  (h5 : annie_total_cost = start_fee + bridge_toll + cost_per_mile * annie_miles)
  (h6 : mike_total_cost = start_fee + cost_per_mile * M)
  (h7 : mike_total_cost = annie_total_cost) :
  M = 36 := 
sorry

end miles_mike_ride_l2165_216527


namespace cannot_pay_exactly_500_can_pay_exactly_600_l2165_216595

-- Defining the costs and relevant equations
def price_of_bun : ℕ := 15
def price_of_croissant : ℕ := 12

-- Proving the non-existence for the 500 Ft case
theorem cannot_pay_exactly_500 : ¬ ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 500 :=
sorry

-- Proving the existence for the 600 Ft case
theorem can_pay_exactly_600 : ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 600 :=
sorry

end cannot_pay_exactly_500_can_pay_exactly_600_l2165_216595


namespace find_number_l2165_216534

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end find_number_l2165_216534


namespace period_of_f_g_is_2_sin_x_g_is_odd_l2165_216553

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

-- Theorem 1: Prove that f has period 2π.
theorem period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

-- Define g and prove the related properties.
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

-- Theorem 2: Prove that g(x) = 2 * sin x.
theorem g_is_2_sin_x : ∀ x : ℝ, g x = 2 * Real.sin x := by
  sorry

-- Theorem 3: Prove that g is an odd function.
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end period_of_f_g_is_2_sin_x_g_is_odd_l2165_216553


namespace tesses_ride_is_longer_l2165_216557

noncomputable def tesses_total_distance : ℝ := 0.75 + 0.85 + 1.15
noncomputable def oscars_total_distance : ℝ := 0.25 + 1.35

theorem tesses_ride_is_longer :
  (tesses_total_distance - oscars_total_distance) = 1.15 := by
  sorry

end tesses_ride_is_longer_l2165_216557


namespace monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l2165_216571

-- Definition of given conditions regarding tourists count in February and April
def tourists_in_february : ℕ := 16000
def tourists_in_april : ℕ := 25000

-- Theorem 1: Monthly average growth rate of tourists from February to April is 25%.
theorem monthly_avg_growth_rate_25 :
  (tourists_in_april : ℝ) = tourists_in_february * (1 + 0.25)^2 :=
sorry

-- Definition of given conditions for tourists count from May 1st to May 21st
def tourists_may_1_to_21 : ℕ := 21250
def max_total_tourists_may : ℕ := 31250 -- Expressed in thousands as 31.25 in millions

-- Theorem 2: Maximum average number of tourists per day in the next 10 days of May.
theorem max_avg_tourists_next_10_days :
  ∀ (a : ℝ), tourists_may_1_to_21 + 10 * a ≤ max_total_tourists_may →
  a ≤ 10000 :=
sorry

end monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l2165_216571


namespace french_fries_cost_is_10_l2165_216597

-- Define the costs as given in the problem conditions
def taco_salad_cost : ℕ := 10
def daves_single_cost : ℕ := 5
def peach_lemonade_cost : ℕ := 2
def num_friends : ℕ := 5
def friend_payment : ℕ := 11

-- Define the total amount collected from friends
def total_collected : ℕ := num_friends * friend_payment

-- Define the subtotal for the known items
def subtotal : ℕ := taco_salad_cost + (num_friends * daves_single_cost) + (num_friends * peach_lemonade_cost)

-- The total cost of french fries
def total_french_fries_cost := total_collected - subtotal

-- The proof statement:
theorem french_fries_cost_is_10 : total_french_fries_cost = 10 := by
  sorry

end french_fries_cost_is_10_l2165_216597


namespace geometric_progression_a5_value_l2165_216513

theorem geometric_progression_a5_value
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n, a (n + 1) = a n * r)
  (h_roots : ∃ x y, x^2 - 5*x + 4 = 0 ∧ y^2 - 5*y + 4 = 0 ∧ x = a 3 ∧ y = a 7) :
  a 5 = 2 :=
by
  sorry

end geometric_progression_a5_value_l2165_216513


namespace power_sum_l2165_216555

theorem power_sum :
  (-1:ℤ)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 :=
by
  sorry

end power_sum_l2165_216555


namespace walk_two_dogs_for_7_minutes_l2165_216569

variable (x : ℕ)

def charge_per_dog : ℕ := 20
def charge_per_minute_per_dog : ℕ := 1
def total_earnings : ℕ := 171

def charge_one_dog := charge_per_dog + charge_per_minute_per_dog * 10
def charge_three_dogs := charge_per_dog * 3 + charge_per_minute_per_dog * 9 * 3
def charge_two_dogs (x : ℕ) := charge_per_dog * 2 + charge_per_minute_per_dog * x * 2

theorem walk_two_dogs_for_7_minutes 
  (h1 : charge_one_dog = 30)
  (h2 : charge_three_dogs = 87)
  (h3 : charge_one_dog + charge_three_dogs + charge_two_dogs x = total_earnings) : 
  x = 7 :=
by
  unfold charge_one_dog charge_three_dogs charge_per_dog charge_per_minute_per_dog total_earnings at *
  sorry

end walk_two_dogs_for_7_minutes_l2165_216569


namespace alice_speed_exceeds_l2165_216537

theorem alice_speed_exceeds (distance : ℕ) (v_bob : ℕ) (time_diff : ℕ) (v_alice : ℕ)
  (h_distance : distance = 220)
  (h_v_bob : v_bob = 40)
  (h_time_diff : time_diff = 1/2) : 
  v_alice > 44 := 
sorry

end alice_speed_exceeds_l2165_216537


namespace num_int_values_not_satisfying_l2165_216570

theorem num_int_values_not_satisfying:
  (∃ n : ℕ, n = 7 ∧ (∃ x : ℤ, 7 * x^2 + 25 * x + 24 ≤ 30)) :=
sorry

end num_int_values_not_satisfying_l2165_216570


namespace impossible_to_form_16_unique_remainders_with_3_digits_l2165_216536

theorem impossible_to_form_16_unique_remainders_with_3_digits :
  ¬∃ (digits : Finset ℕ) (num_fun : Fin 16 → ℕ), digits.card = 3 ∧ 
  ∀ i j : Fin 16, i ≠ j → num_fun i % 16 ≠ num_fun j % 16 ∧ 
  ∀ n : ℕ, n ∈ (digits : Set ℕ) → 100 ≤ num_fun i ∧ num_fun i < 1000 :=
sorry

end impossible_to_form_16_unique_remainders_with_3_digits_l2165_216536


namespace sin_cos_15_sin_cos_18_l2165_216578

theorem sin_cos_15 (h45sin : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h45cos : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h30sin : Real.sin (30 * Real.pi / 180) = 1 / 2)
                  (h30cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2) :
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 ∧
  Real.cos (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

theorem sin_cos_18 (h18sin : Real.sin (18 * Real.pi / 180) = (-1 + Real.sqrt 5) / 4)
                   (h36cos : Real.cos (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 4) :
  Real.cos (18 * Real.pi / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4 := by
  sorry

end sin_cos_15_sin_cos_18_l2165_216578


namespace horner_method_multiplications_and_additions_l2165_216556

noncomputable def f (x : ℕ) : ℕ :=
  12 * x ^ 6 + 5 * x ^ 5 + 11 * x ^ 2 + 2 * x + 5

theorem horner_method_multiplications_and_additions (x : ℕ) :
  let multiplications := 6
  let additions := 4
  multiplications = 6 ∧ additions = 4 :=
sorry

end horner_method_multiplications_and_additions_l2165_216556


namespace certain_number_is_3_l2165_216530

-- Given conditions
variables (z x : ℤ)
variable (k : ℤ)
variable (n : ℤ)

-- Conditions
-- Remainder when z is divided by 9 is 6
def is_remainder_6 (z : ℤ) := ∃ k : ℤ, z = 9 * k + 6
-- (z + x) / 9 is an integer
def is_integer_division (z x : ℤ) := ∃ m : ℤ, (z + x) / 9 = m

-- Proof to show that x must be 3
theorem certain_number_is_3 (z : ℤ) (h1 : is_remainder_6 z) (h2 : is_integer_division z x) : x = 3 :=
sorry

end certain_number_is_3_l2165_216530


namespace shirt_price_is_correct_l2165_216583

noncomputable def sweater_price (T : ℝ) : ℝ := T + 7.43 

def discounted_price (S : ℝ) : ℝ := S * 0.90

theorem shirt_price_is_correct :
  ∃ (T S : ℝ), T + discounted_price S = 80.34 ∧ T = S - 7.43 ∧ T = 38.76 :=
by
  sorry

end shirt_price_is_correct_l2165_216583


namespace yolanda_three_point_avg_l2165_216541

-- Definitions based on conditions
def total_points_season := 345
def total_games := 15
def free_throws_per_game := 4
def two_point_baskets_per_game := 5

-- Definitions based on the derived quantities
def average_points_per_game := total_points_season / total_games
def points_from_two_point_baskets := two_point_baskets_per_game * 2
def points_from_free_throws := free_throws_per_game * 1
def points_from_non_three_point_baskets := points_from_two_point_baskets + points_from_free_throws
def points_from_three_point_baskets := average_points_per_game - points_from_non_three_point_baskets
def three_point_baskets_per_game := points_from_three_point_baskets / 3

-- The theorem to prove that Yolanda averaged 3 three-point baskets per game
theorem yolanda_three_point_avg:
  three_point_baskets_per_game = 3 := sorry

end yolanda_three_point_avg_l2165_216541


namespace impossible_to_place_integers_35x35_l2165_216598

theorem impossible_to_place_integers_35x35 (f : Fin 35 → Fin 35 → ℤ) :
  (∀ i j, abs (f i j - f (i + 1) j) ≤ 18 ∧ abs (f i j - f i (j + 1)) ≤ 18) →
  ∃ i j, i ≠ j ∧ f i j = f i j → False :=
by sorry

end impossible_to_place_integers_35x35_l2165_216598


namespace ratio_books_to_pens_l2165_216503

-- Define the given ratios and known constants.
def ratio_pencils : ℕ := 14
def ratio_pens : ℕ := 4
def ratio_books : ℕ := 3
def actual_pencils : ℕ := 140

-- Assume the actual number of pens can be calculated from ratio.
def actual_pens : ℕ := (actual_pencils / ratio_pencils) * ratio_pens

-- Prove that the ratio of exercise books to pens is as expected.
theorem ratio_books_to_pens (h1 : actual_pencils = 140) 
                            (h2 : actual_pens = 40) : 
  ((actual_pencils / ratio_pencils) * ratio_books) / actual_pens = 3 / 4 :=
by
  -- The following proof steps are omitted as per instruction
  sorry

end ratio_books_to_pens_l2165_216503


namespace jean_vs_pauline_cost_l2165_216528

-- Definitions based on the conditions given
def patty_cost (ida_cost : ℕ) : ℕ := ida_cost + 10
def ida_cost (jean_cost : ℕ) : ℕ := jean_cost + 30
def pauline_cost : ℕ := 30

noncomputable def total_cost (jean_cost : ℕ) : ℕ :=
jean_cost + ida_cost jean_cost + patty_cost (ida_cost jean_cost) + pauline_cost

-- Lean 4 statement to prove the required condition
theorem jean_vs_pauline_cost :
  ∃ (jean_cost : ℕ), total_cost jean_cost = 160 ∧ pauline_cost - jean_cost = 10 :=
by
  sorry

end jean_vs_pauline_cost_l2165_216528


namespace incorrect_number_read_l2165_216504

theorem incorrect_number_read (incorrect_avg correct_avg : ℕ) (n correct_number incorrect_sum correct_sum : ℕ)
  (h1 : incorrect_avg = 17)
  (h2 : correct_avg = 20)
  (h3 : n = 10)
  (h4 : correct_number = 56)
  (h5 : incorrect_sum = n * incorrect_avg)
  (h6 : correct_sum = n * correct_avg)
  (h7 : correct_sum - incorrect_sum = correct_number - X) :
  X = 26 :=
by
  sorry

end incorrect_number_read_l2165_216504


namespace quadratic_inequality_solution_l2165_216574

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 - 5 * x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end quadratic_inequality_solution_l2165_216574


namespace people_in_house_l2165_216591

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l2165_216591


namespace mark_additional_inches_l2165_216545

theorem mark_additional_inches
  (mark_feet : ℕ)
  (mark_inches : ℕ)
  (mike_feet : ℕ)
  (mike_inches : ℕ)
  (foot_to_inches : ℕ)
  (mike_taller_than_mark : ℕ) :
  mark_feet = 5 →
  mike_feet = 6 →
  mike_inches = 1 →
  mike_taller_than_mark = 10 →
  foot_to_inches = 12 →
  5 * 12 + mark_inches + 10 = 6 * 12 + 1 →
  mark_inches = 3 :=
by
  intros
  sorry

end mark_additional_inches_l2165_216545


namespace regular_polygon_exterior_angle_l2165_216552

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end regular_polygon_exterior_angle_l2165_216552


namespace base_256_6_digits_l2165_216520

theorem base_256_6_digits (b : ℕ) (h1 : b ^ 5 ≤ 256) (h2 : 256 < b ^ 6) : b = 3 := 
sorry

end base_256_6_digits_l2165_216520


namespace find_primes_satisfying_equation_l2165_216538

theorem find_primes_satisfying_equation :
  {p : ℕ | p.Prime ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p} = {2, 3, 7} :=
by
  sorry

end find_primes_satisfying_equation_l2165_216538


namespace find_x_l2165_216500

theorem find_x
  (PQR_straight : ∀ x y : ℝ, x + y = 76 → 3 * x + 2 * y = 180)
  (h : x + y = 76) :
  x = 28 :=
by
  sorry

end find_x_l2165_216500


namespace personal_planner_cost_l2165_216561

variable (P : ℝ)
variable (C_spiral_notebook : ℝ := 15)
variable (total_cost_with_discount : ℝ := 112)
variable (discount_rate : ℝ := 0.20)
variable (num_spiral_notebooks : ℝ := 4)
variable (num_personal_planners : ℝ := 8)

theorem personal_planner_cost : (4 * C_spiral_notebook + 8 * P) * (1 - 0.20) = 112 → 
  P = 10 :=
by
  sorry

end personal_planner_cost_l2165_216561


namespace cos_value_l2165_216522

variable (α : ℝ)

theorem cos_value (h : Real.sin (π / 4 + α) = 2 / 3) : Real.cos (π / 4 - α) = 2 / 3 := 
by 
  sorry 

end cos_value_l2165_216522


namespace total_dog_food_per_day_l2165_216558

-- Definitions based on conditions
def dog1_eats_per_day : ℝ := 0.125
def dog2_eats_per_day : ℝ := 0.125
def number_of_dogs : ℕ := 2

-- Mathematically equivalent proof problem statement
theorem total_dog_food_per_day : dog1_eats_per_day + dog2_eats_per_day = 0.25 := 
by
  sorry

end total_dog_food_per_day_l2165_216558


namespace geometric_arithmetic_sequence_l2165_216539

theorem geometric_arithmetic_sequence (a_n : ℕ → ℕ) (q : ℕ) (a1_eq : a_n 1 = 3)
  (an_geometric : ∀ n, a_n (n + 1) = a_n n * q)
  (arithmetic_condition : 4 * a_n 1 + a_n 3 = 8 * a_n 2) :
  a_n 3 + a_n 4 + a_n 5 = 84 := by
  sorry

end geometric_arithmetic_sequence_l2165_216539


namespace player_a_winning_strategy_l2165_216579

theorem player_a_winning_strategy (P : ℝ) : 
  (∃ m n : ℕ, P = m / (2 ^ n) ∧ m < 2 ^ n)
  ∨ P = 0
  ∨ P = 1 ↔
  (∀ d : ℝ, ∃ d_direction : ℤ, 
    (P + (d * d_direction) = 0) ∨ (P + (d * d_direction) = 1)) :=
sorry

end player_a_winning_strategy_l2165_216579


namespace average_of_values_l2165_216506

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end average_of_values_l2165_216506


namespace length_AB_is_correct_l2165_216586

noncomputable def length_of_AB (x y : ℚ) : ℚ :=
  let a := 3 * x
  let b := 2 * x
  let c := 4 * y
  let d := 5 * y
  let pq_distance := abs (c - a)
  if 5 * x = 9 * y ∧ pq_distance = 3 then 5 * x else 0

theorem length_AB_is_correct : 
  ∃ x y : ℚ, 5 * x = 9 * y ∧ (abs (4 * y - 3 * x)) = 3 ∧ length_of_AB x y = 135 / 7 := 
by
  sorry

end length_AB_is_correct_l2165_216586


namespace cash_realized_before_brokerage_l2165_216590

theorem cash_realized_before_brokerage (C : ℝ) (h1 : 0.25 / 100 * C = C / 400)
(h2 : C - C / 400 = 108) : C = 108.27 :=
by
  sorry

end cash_realized_before_brokerage_l2165_216590


namespace ratio_both_to_onlyB_is_2_l2165_216523

variables (num_A num_B both: ℕ)

-- Given conditions
axiom A_eq_2B : num_A = 2 * num_B
axiom both_eq_500 : both = 500
axiom both_multiple_of_only_B : ∃ k : ℕ, both = k * (num_B - both)
axiom only_A_eq_1000 : (num_A - both) = 1000

-- Define the Lean theorem statement
theorem ratio_both_to_onlyB_is_2 : (both : ℝ) / (num_B - both : ℝ) = 2 := 
sorry

end ratio_both_to_onlyB_is_2_l2165_216523


namespace power_expression_evaluation_l2165_216529

theorem power_expression_evaluation :
  (1 / 2) ^ 2016 * (-2) ^ 2017 * (-1) ^ 2017 = 2 := 
by
  sorry

end power_expression_evaluation_l2165_216529


namespace johns_profit_l2165_216532

noncomputable def profit_made 
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ) : ℕ :=
(trees_chopped * planks_per_tree / planks_per_table) * price_per_table - labor_cost

theorem johns_profit : profit_made 30 25 15 300 3000 = 12000 :=
by sorry

end johns_profit_l2165_216532


namespace ratio_a_b_c_l2165_216575

theorem ratio_a_b_c (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b + c) / 3 = 42) (h5 : a = 28) : 
  ∃ y z : ℕ, a / 28 = 1 ∧ b / (ky) = 1 / k ∧ c / (kz) = 1 / k ∧ (b + c) = 98 :=
by sorry

end ratio_a_b_c_l2165_216575


namespace find_x_l2165_216546

theorem find_x (x : ℝ) (h : (2015 + x)^2 = x^2) : x = -2015 / 2 := by
  sorry

end find_x_l2165_216546


namespace parabola_distance_focus_l2165_216505

theorem parabola_distance_focus (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 16) : x = 3 := by
  sorry

end parabola_distance_focus_l2165_216505


namespace arithmetic_seq_problem_l2165_216564

theorem arithmetic_seq_problem (S : ℕ → ℤ) (n : ℕ) (h1 : S 6 = 36) 
                               (h2 : S n = 324) (h3 : S (n - 6) = 144) (hn : n > 6) : 
  n = 18 := 
sorry

end arithmetic_seq_problem_l2165_216564


namespace bill_difference_zero_l2165_216507

theorem bill_difference_zero (l m : ℝ) 
  (hL : (25 / 100) * l = 5) 
  (hM : (15 / 100) * m = 3) : 
  l - m = 0 := 
sorry

end bill_difference_zero_l2165_216507


namespace minimum_abs_ab_l2165_216584

theorem minimum_abs_ab (a b : ℝ) (h : (a^2) * (b / (a^2 + 1)) = 1) : abs (a * b) = 2 := 
  sorry

end minimum_abs_ab_l2165_216584


namespace cooking_time_l2165_216516

theorem cooking_time (total_potatoes cooked_potatoes potato_time : ℕ) 
    (h1 : total_potatoes = 15) 
    (h2 : cooked_potatoes = 6) 
    (h3 : potato_time = 8) : 
    total_potatoes - cooked_potatoes * potato_time = 72 :=
by
    sorry

end cooking_time_l2165_216516


namespace roots_of_polynomial_l2165_216519

theorem roots_of_polynomial :
  ∀ x : ℝ, x * (x + 2)^2 * (3 - x) * (5 + x) = 0 ↔ (x = 0 ∨ x = -2 ∨ x = 3 ∨ x = -5) :=
by
  sorry

end roots_of_polynomial_l2165_216519


namespace evaluate_expression_l2165_216565

-- Given variables x and y are non-zero
variables (x y : ℝ)

-- Condition
axiom xy_nonzero : x * y ≠ 0

-- Statement of the proof
theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y + (x^3 - 2) / y * (y^3 - 2) / x) = 2 * x * y * (x^2 * y^2) + 8 / (x * y) := 
by {
  sorry
}

end evaluate_expression_l2165_216565


namespace store_money_left_l2165_216502

variable (total_items : Nat) (original_price : ℝ) (discount_percent : ℝ)
variable (percent_sold : ℝ) (amount_owed : ℝ)

theorem store_money_left
  (h_total_items : total_items = 2000)
  (h_original_price : original_price = 50)
  (h_discount_percent : discount_percent = 0.80)
  (h_percent_sold : percent_sold = 0.90)
  (h_amount_owed : amount_owed = 15000)
  : (total_items * original_price * (1 - discount_percent) * percent_sold - amount_owed) = 3000 := 
by 
  sorry

end store_money_left_l2165_216502


namespace ratio_pat_to_mark_l2165_216588

theorem ratio_pat_to_mark (K P M : ℕ) 
  (h1 : P + K + M = 117) 
  (h2 : P = 2 * K) 
  (h3 : M = K + 65) : 
  P / Nat.gcd P M = 1 ∧ M / Nat.gcd P M = 3 := 
by
  sorry

end ratio_pat_to_mark_l2165_216588


namespace axis_of_parabola_l2165_216515

-- Define the given equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8 * y

-- Define the standard form of a vertical parabola and the value we need to prove (axis of the parabola)
def standard_form (p y : ℝ) : Prop := y = 2

-- The proof problem: Given the equation of the parabola, prove the equation of its axis.
theorem axis_of_parabola : 
  ∀ x y : ℝ, (parabola x y) → (standard_form y 2) :=
by
  intros x y h
  sorry

end axis_of_parabola_l2165_216515


namespace probability_x_gt_3y_in_rectangle_l2165_216525

noncomputable def probability_of_x_gt_3y :ℝ :=
  let base := 2010
  let height := 2011
  let triangle_height := 670
  (1/2 * base * triangle_height) / (base * height)

theorem probability_x_gt_3y_in_rectangle:
  probability_of_x_gt_3y = 335 / 2011 := 
by
  sorry

end probability_x_gt_3y_in_rectangle_l2165_216525


namespace expression_simplification_l2165_216554

theorem expression_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := 
by
  sorry

end expression_simplification_l2165_216554


namespace coprime_exponents_iff_l2165_216501

theorem coprime_exponents_iff (p q : ℕ) : 
  Nat.gcd (2^p - 1) (2^q - 1) = 1 ↔ Nat.gcd p q = 1 :=
by 
  sorry

end coprime_exponents_iff_l2165_216501


namespace find_p_root_relation_l2165_216508

theorem find_p_root_relation (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 = 3 * x1 ∧ x1^2 + p * x1 + 2 * p = 0 ∧ x2^2 + p * x2 + 2 * p = 0) ↔ (p = 0 ∨ p = 32 / 3) :=
by sorry

end find_p_root_relation_l2165_216508


namespace inequality_of_pos_real_product_l2165_216544

theorem inequality_of_pos_real_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) :=
sorry

end inequality_of_pos_real_product_l2165_216544


namespace find_bounds_l2165_216549

open Set

variable {U : Type} [TopologicalSpace U]

def A := {x : ℝ | 3 ≤ x ∧ x ≤ 4}
def C_UA := {x : ℝ | x > 4 ∨ x < 3}

theorem find_bounds (T : Type) [TopologicalSpace T] : 3 = 3 ∧ 4 = 4 := 
 by sorry

end find_bounds_l2165_216549


namespace diff_of_squares_example_l2165_216576

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l2165_216576


namespace positive_root_exists_iff_m_eq_neg_one_l2165_216560

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end positive_root_exists_iff_m_eq_neg_one_l2165_216560


namespace solve_inequality_l2165_216533

theorem solve_inequality (x : ℝ) : 
  (x ≠ 1) → ( (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ) ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) := 
sorry

end solve_inequality_l2165_216533


namespace eliminate_denominator_l2165_216589

theorem eliminate_denominator (x : ℝ) : 6 - (x - 2) / 2 = x → 12 - x + 2 = 2 * x :=
by
  intro h
  sorry

end eliminate_denominator_l2165_216589


namespace average_hamburgers_per_day_l2165_216526

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end average_hamburgers_per_day_l2165_216526


namespace circular_pond_area_l2165_216548

theorem circular_pond_area (AB CD : ℝ) (D_is_midpoint : Prop) (hAB : AB = 20) (hCD : CD = 12)
  (hD_midpoint : D_is_midpoint ∧ D_is_midpoint = (AB / 2 = 10)) :
  ∃ (A : ℝ), A = 244 * Real.pi :=
by
  sorry

end circular_pond_area_l2165_216548


namespace activities_equally_popular_l2165_216577

def Dodgeball_prefers : ℚ := 10 / 25
def ArtWorkshop_prefers : ℚ := 12 / 30
def MovieScreening_prefers : ℚ := 18 / 45
def QuizBowl_prefers : ℚ := 16 / 40

theorem activities_equally_popular :
  Dodgeball_prefers = ArtWorkshop_prefers ∧
  ArtWorkshop_prefers = MovieScreening_prefers ∧
  MovieScreening_prefers = QuizBowl_prefers :=
by
  sorry

end activities_equally_popular_l2165_216577


namespace projection_of_AB_onto_CD_l2165_216559

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (1, 2)
noncomputable def C : ℝ × ℝ := (-2, -1)
noncomputable def D : ℝ × ℝ := (3, 4)

noncomputable def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_of_AB_onto_CD :
  let AB := vector_sub A B
  let CD := vector_sub C D
  (magnitude AB) * (dot_product AB CD) / (magnitude CD) ^ 2 = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end projection_of_AB_onto_CD_l2165_216559


namespace line_intersects_circle_l2165_216582

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 9 = 0
def line_eq (m x y : ℝ) : Prop := m*x + y + m - 2 = 0

-- Theorem statement based on question and correct answer
theorem line_intersects_circle (m : ℝ) :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

end line_intersects_circle_l2165_216582


namespace years_required_l2165_216585

def num_stadiums := 30
def avg_cost_per_stadium := 900
def annual_savings := 1500
def total_cost := num_stadiums * avg_cost_per_stadium

theorem years_required : total_cost / annual_savings = 18 :=
by
  sorry

end years_required_l2165_216585


namespace Lisa_weight_l2165_216580

theorem Lisa_weight : ∃ l a : ℝ, a + l = 240 ∧ l - a = l / 3 ∧ l = 144 :=
by
  sorry

end Lisa_weight_l2165_216580


namespace gcd_8251_6105_l2165_216521

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l2165_216521


namespace reciprocal_check_C_l2165_216562

theorem reciprocal_check_C : 0.1 * 10 = 1 := 
by 
  sorry

end reciprocal_check_C_l2165_216562


namespace elizabeth_net_profit_l2165_216517

-- Define the conditions
def cost_per_bag : ℝ := 3.00
def bags_produced : ℕ := 20
def selling_price_per_bag : ℝ := 6.00
def bags_sold_full_price : ℕ := 15
def discount_percentage : ℝ := 0.25

-- Define the net profit computation
def net_profit : ℝ :=
  let revenue_full_price := bags_sold_full_price * selling_price_per_bag
  let remaining_bags := bags_produced - bags_sold_full_price
  let discounted_price_per_bag := selling_price_per_bag * (1 - discount_percentage)
  let revenue_discounted := remaining_bags * discounted_price_per_bag
  let total_revenue := revenue_full_price + revenue_discounted
  let total_cost := bags_produced * cost_per_bag
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.50 := by
  sorry

end elizabeth_net_profit_l2165_216517


namespace vec_sub_eq_l2165_216573

variables (a b : ℝ × ℝ)
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (-3, 4)

theorem vec_sub_eq : vec_a - vec_b = (5, -3) :=
by 
  -- You can fill in the proof steps here
  sorry

end vec_sub_eq_l2165_216573


namespace cookies_per_batch_l2165_216550

theorem cookies_per_batch (students : ℕ) (cookies_per_student : ℕ) (chocolate_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) (cookies_needed : ℕ) (dozens_per_batch : ℕ) :
  (students = 24) →
  (cookies_per_student = 10) →
  (chocolate_batches = 2) →
  (oatmeal_batches = 1) →
  (additional_batches = 2) →
  (cookies_needed = students * cookies_per_student) →
  dozens_per_batch * (12 * (chocolate_batches + oatmeal_batches + additional_batches)) = cookies_needed →
  dozens_per_batch = 4 :=
by
  intros
  sorry

end cookies_per_batch_l2165_216550


namespace range_of_M_l2165_216514

theorem range_of_M (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by
  -- We would start the proof here by using the given constraints
  sorry

end range_of_M_l2165_216514


namespace max_students_seated_l2165_216593

/-- Problem statement:
There are a total of 8 rows of desks.
The first row has 10 desks.
Each subsequent row has 2 more desks than the previous row.
We need to prove that the maximum number of students that can be seated in the class is 136.
-/
theorem max_students_seated : 
  let n := 8      -- number of rows
  let a1 := 10    -- desks in the first row
  let d := 2      -- common difference
  let an := a1 + (n - 1) * d  -- desks in the n-th row
  let S := n / 2 * (a1 + an)  -- sum of the arithmetic series
  S = 136 :=
by
  sorry

end max_students_seated_l2165_216593


namespace arithmetic_sequence_product_l2165_216563

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_product (a_1 a_6 a_7 a_4 a_9 : ℝ) (d : ℝ) :
  a_1 = 2 →
  a_6 = a_1 + 5 * d →
  a_7 = a_1 + 6 * d →
  a_6 * a_7 = 15 →
  a_4 = a_1 + 3 * d →
  a_9 = a_1 + 8 * d →
  a_4 * a_9 = 234 / 25 :=
sorry

end arithmetic_sequence_product_l2165_216563


namespace envelope_weight_l2165_216567

theorem envelope_weight (E : ℝ) :
  (8 * (1 / 5) + E ≤ 2) ∧ (1 < 8 * (1 / 5) + E) ∧ (E ≥ 0) ↔ E = 2 / 5 :=
by
  sorry

end envelope_weight_l2165_216567


namespace parabola_translation_l2165_216592

theorem parabola_translation :
  ∀ x y, (y = -2 * x^2) →
    ∃ x' y', y' = -2 * (x' - 2)^2 + 1 ∧ x' = x ∧ y' = y + 1 :=
sorry

end parabola_translation_l2165_216592


namespace evaluate_expression_l2165_216518

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 :=
by
  sorry

end evaluate_expression_l2165_216518


namespace smallest_integer_remainder_l2165_216566

theorem smallest_integer_remainder (n : ℕ) 
  (h5 : n ≡ 1 [MOD 5]) (h7 : n ≡ 1 [MOD 7]) (h8 : n ≡ 1 [MOD 8]) :
  80 < n ∧ n < 299 := 
sorry

end smallest_integer_remainder_l2165_216566


namespace maple_is_taller_l2165_216542

def pine_tree_height : ℚ := 13 + 1/4
def maple_tree_height : ℚ := 20 + 1/2
def height_difference : ℚ := maple_tree_height - pine_tree_height

theorem maple_is_taller : height_difference = 7 + 1/4 := by
  sorry

end maple_is_taller_l2165_216542


namespace range_of_a_for_quad_ineq_false_l2165_216547

variable (a : ℝ)

def quad_ineq_holds : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0

theorem range_of_a_for_quad_ineq_false :
  ¬ quad_ineq_holds a → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_quad_ineq_false_l2165_216547


namespace cosine_of_angle_between_vectors_l2165_216540

theorem cosine_of_angle_between_vectors (a1 b1 c1 a2 b2 c2 : ℝ) :
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  dot_product / (magnitude_u * magnitude_v) = 
      (a1 * a2 + b1 * b2 + c1 * c2) / (Real.sqrt (a1^2 + b1^2 + c1^2) * Real.sqrt (a2^2 + b2^2 + c2^2)) :=
by
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  sorry

end cosine_of_angle_between_vectors_l2165_216540


namespace combined_original_price_l2165_216594

def original_price_shoes (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

def original_price_dress (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

theorem combined_original_price (shoes_price : ℚ) (shoes_discount : ℚ) (dress_price : ℚ) (dress_discount : ℚ) 
  (h_shoes : shoes_discount = 0.20 ∧ shoes_price = 480) 
  (h_dress : dress_discount = 0.30 ∧ dress_price = 350) : 
  original_price_shoes shoes_price shoes_discount + original_price_dress dress_price dress_discount = 1100 := by
  sorry

end combined_original_price_l2165_216594


namespace fraction_of_area_in_triangle_l2165_216512

theorem fraction_of_area_in_triangle :
  let vertex1 := (3, 3)
  let vertex2 := (5, 5)
  let vertex3 := (3, 5)
  let base := (5 - 3)
  let height := (5 - 3)
  let area_triangle := (1 / 2) * base * height
  let area_square := 6 * 6
  let fraction := area_triangle / area_square
  fraction = (1 / 18) :=
by 
  sorry

end fraction_of_area_in_triangle_l2165_216512


namespace distance_from_P_to_AB_l2165_216568

-- Definitions of conditions
def is_point_in_triangle (P A B C : ℝ×ℝ) : Prop := sorry
def parallel_to_base (P A B C : ℝ×ℝ) : Prop := sorry
def divides_area_in_ratio (P A B C : ℝ×ℝ) (r1 r2 : ℕ) : Prop := sorry

theorem distance_from_P_to_AB (P A B C : ℝ×ℝ) 
  (H_in_triangle : is_point_in_triangle P A B C)
  (H_parallel : parallel_to_base P A B C)
  (H_area_ratio : divides_area_in_ratio P A B C 1 3)
  (H_altitude : ∃ h : ℝ, h = 1) :
  ∃ d : ℝ, d = 3/4 :=
by
  sorry

end distance_from_P_to_AB_l2165_216568


namespace find_25_percent_l2165_216531

theorem find_25_percent (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 :=
by
  sorry

end find_25_percent_l2165_216531


namespace rectangle_measurement_error_l2165_216596

theorem rectangle_measurement_error
    (L W : ℝ) -- actual lengths of the sides
    (x : ℝ) -- percentage in excess for the first side
    (h1 : 0 ≤ x) -- ensuring percentage cannot be negative
    (h2 : (L * (1 + x / 100)) * (W * 0.95) = L * W * 1.045) -- given condition on areas
    : x = 10 :=
by
  sorry

end rectangle_measurement_error_l2165_216596


namespace necessary_but_not_sufficient_l2165_216511

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 1 ∨ x > 4) → (x^2 - 3 * x + 2 > 0) ∧ ¬((x^2 - 3 * x + 2 > 0) → (x < 1 ∨ x > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l2165_216511


namespace Maddie_spent_on_tshirts_l2165_216524

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end Maddie_spent_on_tshirts_l2165_216524


namespace find_num_cows_l2165_216572

variable (num_cows num_pigs : ℕ)

theorem find_num_cows (h1 : 4 * num_cows + 24 + 4 * num_pigs = 20 + 2 * (num_cows + 6 + num_pigs)) 
                      (h2 : 6 = 6) 
                      (h3 : ∀x, 2 * x = x + x) 
                      (h4 : ∀x, 4 * x = 2 * 2 * x) 
                      (h5 : ∀x, 4 * x = 4 * x) : 
                      num_cows = 6 := 
by {
  sorry
}

end find_num_cows_l2165_216572


namespace chord_lengths_equal_l2165_216551

theorem chord_lengths_equal (D E F : ℝ) (hcond_1 : D^2 ≠ E^2) (hcond_2 : E^2 > 4 * F) :
  ∀ x y, (x^2 + y^2 + D * x + E * y + F = 0) → 
  (abs x = abs y) :=
by
  sorry

end chord_lengths_equal_l2165_216551
