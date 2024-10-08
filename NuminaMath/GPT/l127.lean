import Mathlib

namespace candy_distribution_l127_127383

-- Definition of the problem
def emily_candies : ℕ := 30
def friends : ℕ := 4

-- Lean statement to prove
theorem candy_distribution : emily_candies % friends = 2 :=
by sorry

end candy_distribution_l127_127383


namespace min_rectangles_to_cover_square_exactly_l127_127427

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l127_127427


namespace sine_addition_l127_127093

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l127_127093


namespace complementary_angles_difference_l127_127979

-- Given that the measures of two complementary angles are in the ratio 4:1,
-- we want to prove that the positive difference between the measures of the two angles is 54 degrees.

theorem complementary_angles_difference (x : ℝ) (h_complementary : 4 * x + x = 90) : 
  abs (4 * x - x) = 54 :=
by
  sorry

end complementary_angles_difference_l127_127979


namespace total_weight_of_lifts_l127_127840

theorem total_weight_of_lifts 
  (F S : ℕ)
  (h1 : F = 400)
  (h2 : 2 * F = S + 300) :
  F + S = 900 :=
by
  sorry

end total_weight_of_lifts_l127_127840


namespace pima_investment_value_l127_127016

noncomputable def pima_investment_worth (initial_investment : ℕ) (first_week_gain_percentage : ℕ) (second_week_gain_percentage : ℕ) : ℕ :=
  let first_week_value := initial_investment + (initial_investment * first_week_gain_percentage / 100)
  let second_week_value := first_week_value + (first_week_value * second_week_gain_percentage / 100)
  second_week_value

-- Conditions
def initial_investment := 400
def first_week_gain_percentage := 25
def second_week_gain_percentage := 50

theorem pima_investment_value :
  pima_investment_worth initial_investment first_week_gain_percentage second_week_gain_percentage = 750 := by
  sorry

end pima_investment_value_l127_127016


namespace books_sold_over_summer_l127_127199

theorem books_sold_over_summer (n l t : ℕ) (h1 : n = 37835) (h2 : l = 143) (h3 : t = 271) : 
  t - l = 128 :=
by
  sorry

end books_sold_over_summer_l127_127199


namespace gcd_160_200_360_l127_127449

theorem gcd_160_200_360 : Nat.gcd (Nat.gcd 160 200) 360 = 40 := by
  sorry

end gcd_160_200_360_l127_127449


namespace raft_travel_time_l127_127738

noncomputable def downstream_speed (x y : ℝ) : ℝ := x + y
noncomputable def upstream_speed (x y : ℝ) : ℝ := x - y

theorem raft_travel_time {x y : ℝ} 
  (h1 : 7 * upstream_speed x y = 5 * downstream_speed x y) : (35 : ℝ) = (downstream_speed x y) * 7 / 4 := by sorry

end raft_travel_time_l127_127738


namespace total_canoes_built_l127_127263

-- Definition of the conditions as suggested by the problem
def num_canoes_in_february : Nat := 5
def growth_rate : Nat := 3
def number_of_months : Nat := 5

-- Final statement to prove
theorem total_canoes_built : (num_canoes_in_february * (growth_rate^number_of_months - 1)) / (growth_rate - 1) = 605 := 
by sorry

end total_canoes_built_l127_127263


namespace find_f_neg2_l127_127510

-- Define the function f and the given conditions
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 4

theorem find_f_neg2 (a b : ℝ) (h₁ : f 2 a b = 6) : f (-2) a b = -14 :=
by
  sorry

end find_f_neg2_l127_127510


namespace work_rate_problem_l127_127224

theorem work_rate_problem 
  (W : ℝ)
  (rate_ab : ℝ)
  (rate_c : ℝ)
  (rate_abc : ℝ)
  (cond1 : rate_c = W / 2)
  (cond2 : rate_abc = W / 1)
  (cond3 : rate_ab = (W / 1) - rate_c) :
  rate_ab = W / 2 :=
by 
  -- We can add the solution steps here, but we skip that part following the guidelines
  sorry

end work_rate_problem_l127_127224


namespace usual_time_catch_bus_l127_127001

-- Define the problem context
variable (S T : ℝ)

-- Hypotheses for the conditions given
def condition1 : Prop := S * T = (4 / 5) * S * (T + 4)
def condition2 : Prop := S ≠ 0

-- Theorem that states the fact we need to prove
theorem usual_time_catch_bus (h1 : condition1 S T) (h2 : condition2 S) : T = 16 :=
by
  -- proof omitted
  sorry

end usual_time_catch_bus_l127_127001


namespace NinaCalculationCorrectAnswer_l127_127453

variable (y : ℝ)

noncomputable def NinaMistakenCalculation (y : ℝ) : ℝ :=
(y + 25) * 5

noncomputable def NinaCorrectCalculation (y : ℝ) : ℝ :=
(y - 25) / 5

theorem NinaCalculationCorrectAnswer (hy : (NinaMistakenCalculation y) = 200) :
  (NinaCorrectCalculation y) = -2 := by
  sorry

end NinaCalculationCorrectAnswer_l127_127453


namespace mean_equivalence_l127_127698

theorem mean_equivalence {x : ℚ} :
  (8 + 15 + 21) / 3 = (18 + x) / 2 → x = 34 / 3 :=
by
  sorry

end mean_equivalence_l127_127698


namespace units_digit_k_squared_plus_2_k_l127_127784

noncomputable def k : ℕ := 2009^2 + 2^2009 - 3

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_k_squared_plus_2_k : units_digit (k^2 + 2^k) = 1 := by
  sorry

end units_digit_k_squared_plus_2_k_l127_127784


namespace regular_polygon_num_sides_l127_127506

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l127_127506


namespace cost_price_per_meter_l127_127540

-- We define the given conditions
def meters_sold : ℕ := 60
def selling_price : ℕ := 8400
def profit_per_meter : ℕ := 12

-- We need to prove that the cost price per meter is Rs. 128
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 128 :=
by
  sorry

end cost_price_per_meter_l127_127540


namespace pet_food_weight_in_ounces_l127_127045

-- Define the given conditions
def cat_food_bags := 2
def cat_food_weight_per_bag := 3 -- in pounds
def dog_food_bags := 2
def additional_dog_food_weight := 2 -- additional weight per bag compared to cat food
def pounds_to_ounces := 16

-- Calculate the total weight of cat food in pounds
def total_cat_food_weight := cat_food_bags * cat_food_weight_per_bag

-- Calculate the weight of each bag of dog food in pounds
def dog_food_weight_per_bag := cat_food_weight_per_bag + additional_dog_food_weight

-- Calculate the total weight of dog food in pounds
def total_dog_food_weight := dog_food_bags * dog_food_weight_per_bag

-- Calculate the total weight of pet food in pounds
def total_pet_food_weight_pounds := total_cat_food_weight + total_dog_food_weight

-- Convert the total weight to ounces
def total_pet_food_weight_ounces := total_pet_food_weight_pounds * pounds_to_ounces

-- Statement of the problem in Lean 4
theorem pet_food_weight_in_ounces : total_pet_food_weight_ounces = 256 := by
  sorry

end pet_food_weight_in_ounces_l127_127045


namespace shoveling_hours_l127_127637

def initial_rate := 25

def rate_decrease := 2

def snow_volume := 6 * 12 * 3

def shoveling_rate (hour : ℕ) : ℕ :=
  if hour = 0 then initial_rate
  else initial_rate - rate_decrease * hour

def cumulative_snow (hour : ℕ) : ℕ :=
  if hour = 0 then snow_volume - shoveling_rate 0
  else cumulative_snow (hour - 1) - shoveling_rate hour

theorem shoveling_hours : cumulative_snow 12 ≠ 0 ∧ cumulative_snow 13 = 47 := by
  sorry

end shoveling_hours_l127_127637


namespace problem_statement_l127_127294

variable {x y z : ℝ}

theorem problem_statement (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
    (h₁ : x^2 - y^2 = y * z) (h₂ : y^2 - z^2 = x * z) : 
    x^2 - z^2 = x * y := 
by
  sorry

end problem_statement_l127_127294


namespace x_intercept_perpendicular_line_l127_127330

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end x_intercept_perpendicular_line_l127_127330


namespace regular_hexagon_area_decrease_l127_127184

noncomputable def area_decrease (original_area : ℝ) (side_decrease : ℝ) : ℝ :=
  let s := (2 * original_area) / (3 * Real.sqrt 3)
  let new_side := s - side_decrease
  let new_area := (3 * Real.sqrt 3 / 2) * new_side ^ 2
  original_area - new_area

theorem regular_hexagon_area_decrease :
  area_decrease (150 * Real.sqrt 3) 3 = 76.5 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_decrease_l127_127184


namespace total_juice_drunk_l127_127604

noncomputable def juiceConsumption (samDrink benDrink : ℕ) (samConsRatio benConsRatio : ℚ) : ℚ :=
  let samConsumed := samConsRatio * samDrink
  let samRemaining := samDrink - samConsumed
  let benConsumed := benConsRatio * benDrink
  let benRemaining := benDrink - benConsumed
  let benToSam := (1 / 2) * benRemaining + 1
  let samTotal := samConsumed + benToSam
  let benTotal := benConsumed - benToSam
  samTotal + benTotal

theorem total_juice_drunk : juiceConsumption 12 20 (2 / 3 : ℚ) (2 / 3 : ℚ) = 32 :=
sorry

end total_juice_drunk_l127_127604


namespace boat_trip_duration_l127_127844

noncomputable def boat_trip_time (B P : ℝ) : Prop :=
  (P = 4 * B) ∧ (B + P = 10)

theorem boat_trip_duration (B P : ℝ) (h : boat_trip_time B P) : B = 2 :=
by
  cases h with
  | intro hP hTotal =>
    sorry

end boat_trip_duration_l127_127844


namespace find_common_ratio_l127_127939

variable (a_n : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

theorem find_common_ratio (h1 : a_n 1 = 2) (h2 : a_n 4 = 16) (h_geom : ∀ n, a_n n = a_n (n - 1) * q)
  : q = 2 := by
  sorry

end find_common_ratio_l127_127939


namespace box_neg2_0_3_eq_10_div_9_l127_127785

def box (a b c : ℤ) : ℚ :=
  a^b - b^c + c^a

theorem box_neg2_0_3_eq_10_div_9 : box (-2) 0 3 = 10 / 9 :=
by
  sorry

end box_neg2_0_3_eq_10_div_9_l127_127785


namespace joan_apples_l127_127320

theorem joan_apples (initial_apples : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : 
  initial_apples = 43 ∧ given_to_melanie = 27 ∧ given_to_sarah = 11 → (initial_apples - given_to_melanie - given_to_sarah) = 5 := 
by
  sorry

end joan_apples_l127_127320


namespace expected_profit_calculation_l127_127846

theorem expected_profit_calculation:
  let odd1 := 1.28
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let initial_bet := 5.00
  let total_payout := initial_bet * (odd1 * odd2 * odd3 * odd4)
  let expected_profit := total_payout - initial_bet
  expected_profit = 212.822 := by
  sorry

end expected_profit_calculation_l127_127846


namespace units_digit_product_l127_127713

theorem units_digit_product : (3^5 * 2^3) % 10 = 4 := 
sorry

end units_digit_product_l127_127713


namespace connie_needs_more_money_l127_127084

variable (cost_connie : ℕ) (cost_watch : ℕ)

theorem connie_needs_more_money 
  (h_connie : cost_connie = 39)
  (h_watch : cost_watch = 55) :
  cost_watch - cost_connie = 16 :=
by sorry

end connie_needs_more_money_l127_127084


namespace projection_sum_of_squares_l127_127543

theorem projection_sum_of_squares (a : ℝ) (α β γ : ℝ) 
    (h1 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) 
    (h2 : (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = 2) :
    4 * a^2 * ((Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2) = 8 * a^2 := 
by
  sorry

end projection_sum_of_squares_l127_127543


namespace express_y_l127_127422

theorem express_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 :=
by {
  sorry
}

end express_y_l127_127422


namespace unique_solution_tan_eq_sin_cos_l127_127725

theorem unique_solution_tan_eq_sin_cos :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arccos 0.1 ∧ Real.tan x = Real.sin (Real.cos x) :=
sorry

end unique_solution_tan_eq_sin_cos_l127_127725


namespace pie_price_l127_127148

theorem pie_price (cakes_sold : ℕ) (cake_price : ℕ) (cakes_total_earnings : ℕ)
                  (pies_sold : ℕ) (total_earnings : ℕ) (price_per_pie : ℕ)
                  (H1 : cakes_sold = 453)
                  (H2 : cake_price = 12)
                  (H3 : pies_sold = 126)
                  (H4 : total_earnings = 6318)
                  (H5 : cakes_total_earnings = cakes_sold * cake_price)
                  (H6 : price_per_pie * pies_sold = total_earnings - cakes_total_earnings) :
    price_per_pie = 7 := by
    sorry

end pie_price_l127_127148


namespace solve_quadratic_simplify_expression_l127_127574

-- 1. Solve the equation 2x^2 - 3x + 1 = 0
theorem solve_quadratic (x : ℝ) :
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
sorry

-- 2. Simplify the given expression
theorem simplify_expression (a b : ℝ) :
  ( (a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a) ) / (b^2 / (a^2 - a*b)) = a / b :=
sorry

end solve_quadratic_simplify_expression_l127_127574


namespace remainder_2021_2025_mod_17_l127_127712

theorem remainder_2021_2025_mod_17 : 
  (2021 * 2022 * 2023 * 2024 * 2025) % 17 = 0 :=
by 
  -- Proof omitted for brevity
  sorry

end remainder_2021_2025_mod_17_l127_127712


namespace min_value_of_f_l127_127579

noncomputable def f (x : ℝ) : ℝ :=
  x^2 / (x - 3)

theorem min_value_of_f : ∀ x > 3, f x ≥ 12 :=
by
  sorry

end min_value_of_f_l127_127579


namespace floor_sqrt_23_squared_l127_127634

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l127_127634


namespace problem1_solution_set_problem2_range_of_m_l127_127886

open Real

noncomputable def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem problem1_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

theorem problem2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5 / 4 :=
sorry

end problem1_solution_set_problem2_range_of_m_l127_127886


namespace horner_eval_hex_to_decimal_l127_127888

-- Problem 1: Evaluate the polynomial using Horner's method
theorem horner_eval (x : ℤ) (f : ℤ → ℤ) (v3 : ℤ) :
  (f x = 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12) →
  x = -4 →
  v3 = (((((3 * x + 5) * x + 6) * x + 79) * x - 8) * x + 35) * x + 12 →
  v3 = -57 :=
by
  intros hf hx hv
  sorry

-- Problem 2: Convert hexadecimal base-6 to decimal
theorem hex_to_decimal (hex : ℕ) (dec : ℕ) :
  hex = 210 →
  dec = 0 * 6^0 + 1 * 6^1 + 2 * 6^2 →
  dec = 78 :=
by
  intros hhex hdec
  sorry

end horner_eval_hex_to_decimal_l127_127888


namespace quadratic_sum_l127_127562

theorem quadratic_sum (b c : ℤ) : 
  (∃ b c : ℤ, (x^2 - 10*x + 15 = 0) ↔ ((x + b)^2 = c)) → b + c = 5 :=
by
  intros h
  sorry

end quadratic_sum_l127_127562


namespace player1_wins_game_533_player1_wins_game_1000_l127_127524

-- Defining a structure for the game conditions
structure Game :=
  (target_sum : ℕ)
  (player1_wins_optimal : Bool)

-- Definition of the game scenarios
def game_533 := Game.mk 533 true
def game_1000 := Game.mk 1000 true

-- Theorem statements for the respective games
theorem player1_wins_game_533 : game_533.player1_wins_optimal :=
by sorry

theorem player1_wins_game_1000 : game_1000.player1_wins_optimal :=
by sorry

end player1_wins_game_533_player1_wins_game_1000_l127_127524


namespace quadratic_eq_of_sum_and_product_l127_127050

theorem quadratic_eq_of_sum_and_product (a b c : ℝ) (h_sum : -b / a = 4) (h_product : c / a = 3) :
    ∀ (x : ℝ), a * x^2 + b * x + c = a * x^2 - 4 * a * x + 3 * a :=
by
  sorry

end quadratic_eq_of_sum_and_product_l127_127050


namespace quadratic_inequality_solution_l127_127673

-- Given a quadratic inequality, prove the solution set in interval notation.
theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x ^ 2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 :=
sorry

end quadratic_inequality_solution_l127_127673


namespace measure_of_angle_A_possibilities_l127_127878

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l127_127878


namespace complex_number_solution_l127_127884

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end complex_number_solution_l127_127884


namespace relationship_among_numbers_l127_127967

theorem relationship_among_numbers :
  let a := 0.7 ^ 2.1
  let b := 0.7 ^ 2.5
  let c := 2.1 ^ 0.7
  b < a ∧ a < c := by
  sorry

end relationship_among_numbers_l127_127967


namespace Bryan_did_258_pushups_l127_127461

-- Define the conditions
def sets : ℕ := 15
def pushups_per_set : ℕ := 18
def pushups_fewer_last_set : ℕ := 12

-- Define the planned total push-ups
def planned_total_pushups : ℕ := sets * pushups_per_set

-- Define the actual push-ups in the last set
def last_set_pushups : ℕ := pushups_per_set - pushups_fewer_last_set

-- Define the total push-ups Bryan did
def total_pushups : ℕ := (sets - 1) * pushups_per_set + last_set_pushups

-- The theorem to prove
theorem Bryan_did_258_pushups :
  total_pushups = 258 := by
  sorry

end Bryan_did_258_pushups_l127_127461


namespace no_possible_values_for_n_l127_127323

theorem no_possible_values_for_n (n a : ℤ) (h : n > 1) (d : ℤ := 3) (Sn : ℤ := 180) :
  ∃ n > 1, ∃ k : ℤ, a = k^2 ∧ Sn = n / 2 * (2 * a + (n - 1) * d) :=
sorry

end no_possible_values_for_n_l127_127323


namespace target1_target2_l127_127095

variable (α : ℝ)

-- Define the condition
def tan_alpha := Real.tan α = 2

-- State the first target with the condition considered
theorem target1 (h : tan_alpha α) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := by
  sorry

-- State the second target with the condition considered
theorem target2 (h : tan_alpha α) : 
  4 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = 1 := by
  sorry

end target1_target2_l127_127095


namespace conical_pile_volume_l127_127337

noncomputable def volume_of_cone (d : ℝ) (h : ℝ) : ℝ :=
  (Real.pi * (d / 2) ^ 2 * h) / 3

theorem conical_pile_volume :
  let diameter := 10
  let height := 0.60 * diameter
  volume_of_cone diameter height = 50 * Real.pi :=
by
  sorry

end conical_pile_volume_l127_127337


namespace find_x_y_l127_127576

theorem find_x_y (x y : ℝ) : 
    (3 * x + 2 * y + 5 * x + 7 * x = 360) →
    (x = y) →
    (x = 360 / 17) ∧ (y = 360 / 17) := by
  intros h₁ h₂
  sorry

end find_x_y_l127_127576


namespace star_eq_122_l127_127032

noncomputable def solveForStar (star : ℕ) : Prop :=
  45 - (28 - (37 - (15 - star))) = 56

theorem star_eq_122 : solveForStar 122 :=
by
  -- proof
  sorry

end star_eq_122_l127_127032


namespace xiao_peach_days_l127_127770

theorem xiao_peach_days :
  ∀ (xiao_ming_apples xiao_ming_pears xiao_ming_peaches : ℕ)
    (xiao_hong_apples xiao_hong_pears xiao_hong_peaches : ℕ)
    (both_eat_apples both_eat_pears : ℕ)
    (one_eats_apple_other_eats_pear : ℕ),
    xiao_ming_apples = 4 →
    xiao_ming_pears = 6 →
    xiao_ming_peaches = 8 →
    xiao_hong_apples = 5 →
    xiao_hong_pears = 7 →
    xiao_hong_peaches = 6 →
    both_eat_apples = 3 →
    both_eat_pears = 2 →
    one_eats_apple_other_eats_pear = 3 →
    ∃ (both_eat_peaches_days : ℕ),
      both_eat_peaches_days = 4 := 
sorry

end xiao_peach_days_l127_127770


namespace tan_150_eq_neg_one_over_sqrt_three_l127_127310

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l127_127310


namespace evaluate_expression_l127_127866

-- Define the expression as given in the problem
def expr1 : ℤ := |9 - 8 * (3 - 12)|
def expr2 : ℤ := |5 - 11|

-- Define the mathematical equivalence
theorem evaluate_expression : (expr1 - expr2) = 75 := by
  sorry

end evaluate_expression_l127_127866


namespace pinedale_mall_distance_l127_127248

theorem pinedale_mall_distance 
  (speed : ℝ) (time_between_stops : ℝ) (num_stops : ℕ) (distance : ℝ) 
  (h_speed : speed = 60) 
  (h_time_between_stops : time_between_stops = 5 / 60) 
  (h_num_stops : ↑num_stops = 5) :
  distance = 25 :=
by
  sorry

end pinedale_mall_distance_l127_127248


namespace family_reunion_handshakes_l127_127707

theorem family_reunion_handshakes (married_couples : ℕ) (participants : ℕ) (allowed_handshakes : ℕ) (total_handshakes : ℕ) :
  married_couples = 8 →
  participants = married_couples * 2 →
  allowed_handshakes = participants - 1 - 1 - 6 →
  total_handshakes = (participants * allowed_handshakes) / 2 →
  total_handshakes = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end family_reunion_handshakes_l127_127707


namespace no_infinite_monochromatic_arithmetic_progression_l127_127408

theorem no_infinite_monochromatic_arithmetic_progression : 
  ∃ (coloring : ℕ → ℕ), (∀ (q r : ℕ), ∃ (n1 n2 : ℕ), coloring (q * n1 + r) ≠ coloring (q * n2 + r)) := sorry

end no_infinite_monochromatic_arithmetic_progression_l127_127408


namespace problem1_problem2_l127_127629

-- Define A and B as given
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y - 5 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

-- Problem statement 1: Prove 3A + 6B simplifies as expected
theorem problem1 (x y : ℝ) : 3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9 :=
  by
    sorry

-- Problem statement 2: Prove that if 3A + 6B is independent of x, then y = -5
theorem problem2 (y : ℝ) (h : ∀ x : ℝ, 3 * A x y + 6 * B x y = -9) : y = -5 :=
  by
    sorry

end problem1_problem2_l127_127629


namespace rectangle_perimeter_l127_127657

-- Definitions based on the conditions
def length : ℕ := 15
def width : ℕ := 8

-- Definition of the perimeter function
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Statement of the theorem we need to prove
theorem rectangle_perimeter : perimeter length width = 46 := by
  sorry

end rectangle_perimeter_l127_127657


namespace negation_of_proposition_l127_127892

theorem negation_of_proposition (p : Real → Prop) : 
  (∀ x : Real, p x) → ¬(∀ x : Real, x ≥ 1) ↔ (∃ x : Real, x < 1) := 
by sorry

end negation_of_proposition_l127_127892


namespace most_accurate_method_is_independence_test_l127_127333

-- Definitions and assumptions
inductive Methods
| contingency_table
| independence_test
| stacked_bar_chart
| others

def related_or_independent_method : Methods := Methods.independence_test

-- Proof statement
theorem most_accurate_method_is_independence_test :
  related_or_independent_method = Methods.independence_test :=
sorry

end most_accurate_method_is_independence_test_l127_127333


namespace range_of_m_l127_127386

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = m - 1)
  (h2 : x - 3 * y = 2 * m)
  (h3 : x + 2 * y ≥ 0) : 
  m ≤ -1 := 
sorry

end range_of_m_l127_127386


namespace square_of_product_of_third_sides_l127_127719

-- Given data for triangles P1 and P2
variables {a b c d : ℝ}

-- Areas of triangles P1 and P2
def area_P1_pos (a b : ℝ) : Prop := a * b / 2 = 3
def area_P2_pos (a d : ℝ) : Prop := a * d / 2 = 6

-- Condition that b = d / 2
def side_ratio (b d : ℝ) : Prop := b = d / 2

-- Pythagorean theorem applied to both triangles
def pythagorean_P1 (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def pythagorean_P2 (a d c : ℝ) : Prop := a^2 + d^2 = c^2

-- The goal is to prove (cd)^2 = 120
theorem square_of_product_of_third_sides (a b c d : ℝ)
  (h_area_P1: area_P1_pos a b) 
  (h_area_P2: area_P2_pos a d) 
  (h_side_ratio: side_ratio b d) 
  (h_pythagorean_P1: pythagorean_P1 a b c) 
  (h_pythagorean_P2: pythagorean_P2 a d c) :
  (c * d)^2 = 120 := 
sorry

end square_of_product_of_third_sides_l127_127719


namespace algebraic_identity_l127_127098

theorem algebraic_identity 
  (p q r a b c : ℝ)
  (h₁ : p + q + r = 1)
  (h₂ : 1 / p + 1 / q + 1 / r = 0) :
  a^2 + b^2 + c^2 = (p * a + q * b + r * c)^2 + (q * a + r * b + p * c)^2 + (r * a + p * b + q * c)^2 := by
  sorry

end algebraic_identity_l127_127098


namespace arithmetic_and_geometric_sequence_statement_l127_127115

-- Arithmetic sequence definitions
def arithmetic_seq (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Conditions
def a_2 : ℕ := 9
def a_5 : ℕ := 21

-- General formula and solution for part (Ⅰ)
def general_formula_arithmetic_sequence : Prop :=
  ∃ (a d : ℕ), (a + d = a_2 ∧ a + 4 * d = a_5) ∧ ∀ n : ℕ, arithmetic_seq a d n = 4 * n + 1

-- Definitions and conditions for geometric sequence derived from arithmetic sequence
def b_n (n : ℕ) : ℕ := 2 ^ (4 * n + 1)

-- Sum of the first n terms of the sequence {b_n}
def S_n (n : ℕ) : ℕ := (32 * (2 ^ (4 * n) - 1)) / 15

-- Statement that needs to be proven
theorem arithmetic_and_geometric_sequence_statement :
  general_formula_arithmetic_sequence ∧ (∀ n, S_n n = (32 * (2 ^ (4 * n) - 1)) / 15) := by
  sorry

end arithmetic_and_geometric_sequence_statement_l127_127115


namespace multiplication_approximation_correct_l127_127466

noncomputable def closest_approximation (x : ℝ) : ℝ := 
  if 15700 <= x ∧ x < 15750 then 15700
  else if 15750 <= x ∧ x < 15800 then 15750
  else if 15800 <= x ∧ x < 15900 then 15800
  else if 15900 <= x ∧ x < 16000 then 15900
  else 16000

theorem multiplication_approximation_correct :
  closest_approximation (0.00525 * 3153420) = 15750 := 
by
  sorry

end multiplication_approximation_correct_l127_127466


namespace value_of_y_minus_x_l127_127364

theorem value_of_y_minus_x (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : x + y = 8) 
  (h3 : y - 3 * x + z = 9) : 
  y - x = 6.5 :=
by
  -- Proof steps would go here
  sorry

end value_of_y_minus_x_l127_127364


namespace find_target_number_l127_127296

theorem find_target_number : ∃ n ≥ 0, (∀ k < 5, ∃ m, 0 ≤ m ∧ m ≤ n ∧ m % 11 = 3 ∧ m = 3 + k * 11) ∧ n = 47 :=
by
  sorry

end find_target_number_l127_127296


namespace interval_for_x_l127_127411

theorem interval_for_x (x : ℝ) 
  (hx1 : 1/x < 2) 
  (hx2 : 1/x > -3) : 
  x > 1/2 ∨ x < -1/3 :=
  sorry

end interval_for_x_l127_127411


namespace greatest_root_f_l127_127969

noncomputable def f (x : ℝ) : ℝ := 21 * x ^ 4 - 20 * x ^ 2 + 3

theorem greatest_root_f :
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
sorry

end greatest_root_f_l127_127969


namespace dvd_cd_ratio_l127_127591

theorem dvd_cd_ratio (total_sales : ℕ) (dvd_sales : ℕ) (cd_sales : ℕ) (h1 : total_sales = 273) (h2 : dvd_sales = 168) (h3 : cd_sales = total_sales - dvd_sales) : (dvd_sales / Nat.gcd dvd_sales cd_sales) = 8 ∧ (cd_sales / Nat.gcd dvd_sales cd_sales) = 5 :=
by
  sorry

end dvd_cd_ratio_l127_127591


namespace variance_of_scores_l127_127361

def scores : List ℝ := [8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (List.sum (List.map (λ x => (x - m) ^ 2) xs)) / (xs.length)

theorem variance_of_scores : variance scores = 40 / 9 :=
by
  sorry

end variance_of_scores_l127_127361


namespace no_lattice_points_on_hyperbola_l127_127669

theorem no_lattice_points_on_hyperbola : ∀ x y : ℤ, x^2 - y^2 ≠ 2022 :=
by
  intro x y
  -- proof omitted
  sorry

end no_lattice_points_on_hyperbola_l127_127669


namespace point_coordinates_with_respect_to_origin_l127_127297

theorem point_coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  sorry

end point_coordinates_with_respect_to_origin_l127_127297


namespace problem1_l127_127753

theorem problem1 (x y : ℝ) (h1 : x + y = 4) (h2 : 2 * x - y = 5) : 
  x = 3 ∧ y = 1 := sorry

end problem1_l127_127753


namespace area_of_CDE_in_isosceles_triangle_l127_127211

noncomputable def isosceles_triangle_area (b : ℝ) (s : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * b * s

noncomputable def cot (α : ℝ) : ℝ := 1 / Real.tan α

noncomputable def isosceles_triangle_vertex_angle (b : ℝ) (area : ℝ) (θ : ℝ) : Prop :=
  area = (b^2 / 4) * cot (θ / 2)

theorem area_of_CDE_in_isosceles_triangle (b θ area : ℝ) (hb : b = 3 * (2 * b / 3)) (hθ : θ = 100) (ha : area = 30) :
  ∃ CDE_area, CDE_area = area / 9 ∧ CDE_area = 10 / 3 :=
by
  sorry

end area_of_CDE_in_isosceles_triangle_l127_127211


namespace distinct_pairs_l127_127229

theorem distinct_pairs (x y : ℝ) (h : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧ x^200 - y^200 = 2^199 * (x - y) ↔ (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
by
  sorry

end distinct_pairs_l127_127229


namespace sqrt_a_squared_b_l127_127953

variable {a b : ℝ}

theorem sqrt_a_squared_b (h: a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end sqrt_a_squared_b_l127_127953


namespace perfect_square_solution_l127_127362

theorem perfect_square_solution (x : ℤ) : 
  ∃ k : ℤ, x^2 - 14 * x - 256 = k^2 ↔ x = 15 ∨ x = -1 :=
by
  sorry

end perfect_square_solution_l127_127362


namespace integral_identity_proof_l127_127983

noncomputable def integral_identity : Prop :=
  ∫ x in (0 : Real)..(Real.pi / 2), (Real.cos (Real.cos x))^2 + (Real.sin (Real.sin x))^2 = Real.pi / 2

theorem integral_identity_proof : integral_identity :=
sorry

end integral_identity_proof_l127_127983


namespace min_value_proof_l127_127200

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  4 / (x + 3 * y) + 1 / (x - y)

theorem min_value_proof (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) : 
  min_value_expr x y = 9 / 4 := 
sorry

end min_value_proof_l127_127200


namespace find_m_l127_127168

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^3 + 6*x^2 - m

theorem find_m (m : ℝ) (h : ∃ x : ℝ, f x m = 12) : m = 20 :=
by
  sorry

end find_m_l127_127168


namespace least_number_four_digits_divisible_by_15_25_40_75_l127_127913

noncomputable def least_four_digit_multiple : ℕ :=
  1200

theorem least_number_four_digits_divisible_by_15_25_40_75 :
  (∀ n, (n ∣ 15) ∧ (n ∣ 25) ∧ (n ∣ 40) ∧ (n ∣ 75)) → least_four_digit_multiple = 1200 :=
sorry

end least_number_four_digits_divisible_by_15_25_40_75_l127_127913


namespace eugene_total_pencils_l127_127470

def initial_pencils : ℕ := 51
def additional_pencils : ℕ := 6
def total_pencils : ℕ := initial_pencils + additional_pencils

theorem eugene_total_pencils : total_pencils = 57 := by
  sorry

end eugene_total_pencils_l127_127470


namespace problem1_problem2_l127_127862

theorem problem1 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := sorry

theorem problem2 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (A + C) / (2 * B + A) = 9 / 5 := sorry

end problem1_problem2_l127_127862


namespace wickets_before_last_match_l127_127348

theorem wickets_before_last_match
  (W : ℝ)  -- Number of wickets before last match
  (R : ℝ)  -- Total runs before last match
  (h1 : R = 12.4 * W)
  (h2 : (R + 26) / (W + 8) = 12.0)
  : W = 175 :=
sorry

end wickets_before_last_match_l127_127348


namespace remainder_when_divided_l127_127342

theorem remainder_when_divided (P D Q R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D * D') = R + R' * D :=
by
  sorry

end remainder_when_divided_l127_127342


namespace altitudes_not_form_triangle_l127_127187

theorem altitudes_not_form_triangle (h₁ h₂ h₃ : ℝ) :
  ¬(h₁ = 5 ∧ h₂ = 12 ∧ h₃ = 13 ∧ ∃ a b c : ℝ, a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ ∧
    a < b + c ∧ b < a + c ∧ c < a + b) :=
by sorry

end altitudes_not_form_triangle_l127_127187


namespace limit_example_l127_127592

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - 11| ∧ |x - 11| < δ) → |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε :=
by
  sorry

end limit_example_l127_127592


namespace base10_to_base7_l127_127446

theorem base10_to_base7 : 
  ∃ a b c d : ℕ, a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 = 729 ∧ a = 2 ∧ b = 0 ∧ c = 6 ∧ d = 1 :=
sorry

end base10_to_base7_l127_127446


namespace compound_interest_rate_l127_127747

theorem compound_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Annual interest rate in decimal
  (A2 A3 : ℝ)  -- Amounts after 2 and 3 years
  (h2 : A2 = P * (1 + r)^2)
  (h3 : A3 = P * (1 + r)^3) :
  A2 = 17640 → A3 = 22932 → r = 0.3 := by
  sorry

end compound_interest_rate_l127_127747


namespace johns_weekly_earnings_increase_l127_127299

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end johns_weekly_earnings_increase_l127_127299


namespace largest_k_inequality_l127_127864

theorem largest_k_inequality :
  ∃ k : ℝ, (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a + b + c = 3 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a * b - b * c - c * a)) ∧ k = 5 :=
sorry

end largest_k_inequality_l127_127864


namespace num_positive_integers_m_l127_127919

theorem num_positive_integers_m (h : ∀ m : ℕ, ∃ d : ℕ, 3087 = d ∧ m^2 = d + 3) :
  ∃! m : ℕ, 0 < m ∧ (3087 % (m^2 - 3) = 0) := by
  sorry

end num_positive_integers_m_l127_127919


namespace determine_value_of_m_l127_127041

theorem determine_value_of_m (m : ℤ) :
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 ↔ m = 11 := 
sorry

end determine_value_of_m_l127_127041


namespace first_term_geometric_l127_127814

-- Definition: geometric sequence properties
variables (a r : ℚ) -- sequence terms are rational numbers
variables (n : ℕ)

-- Conditions: fifth and sixth terms of a geometric sequence
def fifth_term_geometric (a r : ℚ) : ℚ := a * r^4
def sixth_term_geometric (a r : ℚ) : ℚ := a * r^5

-- Proof: given conditions
theorem first_term_geometric (a r : ℚ) (h1 : fifth_term_geometric a r = 48) 
  (h2 : sixth_term_geometric a r = 72) : a = 768 / 81 :=
by {
  sorry
}

end first_term_geometric_l127_127814


namespace jemma_grasshoppers_l127_127773

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end jemma_grasshoppers_l127_127773


namespace gcd_calculation_l127_127988

theorem gcd_calculation : 
  Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := 
by
  sorry

end gcd_calculation_l127_127988


namespace canonical_form_lines_l127_127379

theorem canonical_form_lines (x y z : ℝ) :
  (2 * x - y + 3 * z - 1 = 0) →
  (5 * x + 4 * y - z - 7 = 0) →
  (∃ (k : ℝ), x = -11 * k ∧ y = 17 * k + 2 ∧ z = 13 * k + 1) :=
by
  intros h1 h2
  sorry

end canonical_form_lines_l127_127379


namespace find_number_l127_127111

theorem find_number (x : ℝ) (h : x / 3 = 1.005 * 400) : x = 1206 := 
by 
sorry

end find_number_l127_127111


namespace roots_product_of_quadratic_equation_l127_127943

variables (a b : ℝ)

-- Given that a and b are roots of the quadratic equation x^2 - ax + b = 0
-- and given conditions that a + b = 5 and ab = 6,
-- prove that a * b = 6.
theorem roots_product_of_quadratic_equation 
  (h₁ : a + b = 5) 
  (h₂ : a * b = 6) : 
  a * b = 6 := 
by 
 sorry

end roots_product_of_quadratic_equation_l127_127943


namespace profit_calculation_more_profitable_method_l127_127762

def profit_end_of_month (x : ℝ) : ℝ :=
  0.3 * x - 900

def profit_beginning_of_month (x : ℝ) : ℝ :=
  0.26 * x

theorem profit_calculation (x : ℝ) (h₁ : profit_end_of_month x = 0.3 * x - 900)
  (h₂ : profit_beginning_of_month x = 0.26 * x) :
  profit_end_of_month x = 0.3 * x - 900 ∧ profit_beginning_of_month x = 0.26 * x :=
by 
  sorry

theorem more_profitable_method (x : ℝ) (hx : x = 20000)
  (h_beg : profit_beginning_of_month x = 0.26 * x)
  (h_end : profit_end_of_month x = 0.3 * x - 900) :
  profit_beginning_of_month x > profit_end_of_month x ∧ profit_beginning_of_month x = 5200 :=
by 
  sorry

end profit_calculation_more_profitable_method_l127_127762


namespace mask_digit_identification_l127_127489

theorem mask_digit_identification :
  ∃ (elephant_mask mouse_mask pig_mask panda_mask : ℕ),
    (4 * 4 = 16) ∧
    (7 * 7 = 49) ∧
    (8 * 8 = 64) ∧
    (9 * 9 = 81) ∧
    elephant_mask = 6 ∧
    mouse_mask = 4 ∧
    pig_mask = 8 ∧
    panda_mask = 1 :=
by
  sorry

end mask_digit_identification_l127_127489


namespace gina_can_paint_6_rose_cups_an_hour_l127_127034

def number_of_rose_cups_painted_in_an_hour 
  (R : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (total_payment : ℕ) (hourly_rate : ℕ)
  (lily_hours : ℕ) (total_hours : ℕ) (rose_hours : ℕ) : Prop :=
  (lily_rate = 7) ∧
  (rose_order = 6) ∧
  (lily_order = 14) ∧
  (total_payment = 90) ∧
  (hourly_rate = 30) ∧
  (lily_hours = lily_order / lily_rate) ∧
  (total_hours = total_payment / hourly_rate) ∧
  (rose_hours = total_hours - lily_hours) ∧
  (rose_order = R * rose_hours)

theorem gina_can_paint_6_rose_cups_an_hour :
  ∃ R, number_of_rose_cups_painted_in_an_hour 
    R 7 6 14 90 30 (14 / 7) (90 / 30)  (90 / 30 - 14 / 7) ∧ R = 6 :=
by
  -- proof is left out intentionally
  sorry

end gina_can_paint_6_rose_cups_an_hour_l127_127034


namespace number_of_boys_at_reunion_l127_127077

theorem number_of_boys_at_reunion (n : ℕ) (h : n * (n - 1) / 2 = 66) : n = 12 :=
sorry

end number_of_boys_at_reunion_l127_127077


namespace solution_set_of_inequality_eq_l127_127319

noncomputable def inequality_solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem solution_set_of_inequality_eq :
  {x : ℝ | (2 * x) / (x - 1) < 1} = inequality_solution_set := by
  sorry

end solution_set_of_inequality_eq_l127_127319


namespace area_of_shaded_region_l127_127328

theorem area_of_shaded_region:
  let b := 10
  let h := 6
  let n := 14
  let rect_length := 2
  let rect_height := 1.5
  (n * rect_length * rect_height - (1/2 * b * h)) = 12 := 
by
  sorry

end area_of_shaded_region_l127_127328


namespace exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l127_127117

-- Definition for the condition that ab + 10 is a perfect square
def is_perfect_square_sum (a b : ℕ) : Prop := ∃ k : ℕ, a * b + 10 = k * k

-- Problem: Existence of three different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem exists_three_naturals_sum_perfect_square :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_perfect_square_sum a b ∧ is_perfect_square_sum b c ∧ is_perfect_square_sum c a := sorry

-- Problem: Non-existence of four different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem no_four_naturals_sum_perfect_square :
  ¬ ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧
    is_perfect_square_sum a b ∧ is_perfect_square_sum a c ∧ is_perfect_square_sum a d ∧
    is_perfect_square_sum b c ∧ is_perfect_square_sum b d ∧ is_perfect_square_sum c d := sorry

end exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l127_127117


namespace complement_event_l127_127292

-- Definitions based on conditions
variables (shoot1 shoot2 : Prop) -- shoots the target on the first and second attempt

-- Definition based on the question and answer
def hits_at_least_once : Prop := shoot1 ∨ shoot2
def misses_both_times : Prop := ¬shoot1 ∧ ¬shoot2

-- Theorem statement based on the mathematical translation
theorem complement_event :
  misses_both_times shoot1 shoot2 = ¬hits_at_least_once shoot1 shoot2 :=
by sorry

end complement_event_l127_127292


namespace initial_percentage_female_workers_l127_127661

theorem initial_percentage_female_workers
(E : ℕ) (F : ℝ) 
(h1 : E + 30 = 360) 
(h2 : (F / 100) * E = (55 / 100) * (E + 30)) :
F = 60 :=
by
  -- proof omitted
  sorry

end initial_percentage_female_workers_l127_127661


namespace fourth_term_of_gp_is_negative_10_point_42_l127_127375

theorem fourth_term_of_gp_is_negative_10_point_42 (x : ℝ) 
  (h : ∃ r : ℝ, r * (5 * x + 5) = (3 * x + 3) * ((3 * x + 3) / x)) :
  r * (5 * x + 5) * ((3 * x + 3) / x) * ((3 * x + 3) / x) = -10.42 :=
by
  sorry

end fourth_term_of_gp_is_negative_10_point_42_l127_127375


namespace cd_leq_one_l127_127074

variables {a b c d : ℝ}

theorem cd_leq_one (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := 
sorry

end cd_leq_one_l127_127074


namespace least_possible_value_of_one_integer_l127_127177

theorem least_possible_value_of_one_integer (
  A B C D E F : ℤ
) (h1 : (A + B + C + D + E + F) / 6 = 63)
  (h2 : A ≤ 100 ∧ B ≤ 100 ∧ C ≤ 100 ∧ D ≤ 100 ∧ E ≤ 100 ∧ F ≤ 100)
  (h3 : (A + B + C) / 3 = 65) : 
  ∃ D E F, (D + E + F) = 183 ∧ min D (min E F) = 83 := sorry

end least_possible_value_of_one_integer_l127_127177


namespace isosceles_triangle_perimeter_l127_127642

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : is_isosceles_triangle a b b) (h₂ : is_valid_triangle a b b) : a + b + b = 15 :=
  sorry

end isosceles_triangle_perimeter_l127_127642


namespace partial_fraction_decomposition_l127_127924

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), (0 ≤ a ∧ a < 5) ∧ (0 ≤ b ∧ b < 13) ∧ (1 / 2015 = a / 5 + b / 13 + c / 31) ∧ (a + b = 14) :=
sorry

end partial_fraction_decomposition_l127_127924


namespace find_x_l127_127571

theorem find_x (x : ℕ) (hx : x > 0 ∧ x <= 100) 
    (mean_twice_mode : (40 + 57 + 76 + 90 + x + x) / 6 = 2 * x) : 
    x = 26 :=
sorry

end find_x_l127_127571


namespace stratified_sampling_major_C_l127_127941

theorem stratified_sampling_major_C
  (students_A : ℕ) (students_B : ℕ) (students_C : ℕ) (students_D : ℕ)
  (total_students : ℕ) (sample_size : ℕ)
  (hA : students_A = 150) (hB : students_B = 150) (hC : students_C = 400) (hD : students_D = 300)
  (hTotal : total_students = students_A + students_B + students_C + students_D)
  (hSample : sample_size = 40)
  : students_C * (sample_size / total_students) = 16 :=
by
  sorry

end stratified_sampling_major_C_l127_127941


namespace find_selling_price_l127_127895

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end find_selling_price_l127_127895


namespace sale_in_third_month_l127_127652

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (average_sales : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 6470)
  (h_avg : average_sales = 6100) : 
  ∃ sale3, sale1 + sale2 + sale3 + sale4 + sale5 + sale6 = average_sales * 6 ∧ sale3 = 6200 :=
by
  sorry

end sale_in_third_month_l127_127652


namespace average_speed_of_entire_trip_l127_127708

/-- Conditions -/
def distance_local : ℝ := 40  -- miles
def speed_local : ℝ := 20  -- mph
def distance_highway : ℝ := 180  -- miles
def speed_highway : ℝ := 60  -- mph

/-- Average speed proof statement -/
theorem average_speed_of_entire_trip :
  let total_distance := distance_local + distance_highway
  let total_time := distance_local / speed_local + distance_highway / speed_highway
  total_distance / total_time = 44 :=
by
  sorry

end average_speed_of_entire_trip_l127_127708


namespace largest_A_divisible_by_8_equal_quotient_remainder_l127_127189

theorem largest_A_divisible_by_8_equal_quotient_remainder :
  ∃ (A B C : ℕ), A = 8 * B + C ∧ B = C ∧ C < 8 ∧ A = 63 := by
  sorry

end largest_A_divisible_by_8_equal_quotient_remainder_l127_127189


namespace count_of_squares_difference_l127_127006

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l127_127006


namespace fraction_sum_neg_one_l127_127137

variable (a : ℚ)

theorem fraction_sum_neg_one (h : a ≠ 1/2) : (a / (1 - 2 * a)) + ((a - 1) / (1 - 2 * a)) = -1 := 
sorry

end fraction_sum_neg_one_l127_127137


namespace frustum_shortest_distance_l127_127729

open Real

noncomputable def shortest_distance (R1 R2 : ℝ) (AB : ℝ) (string_from_midpoint : Bool) : ℝ :=
  if R1 = 5 ∧ R2 = 10 ∧ AB = 20 ∧ string_from_midpoint = true then 4 else 0

theorem frustum_shortest_distance : 
  shortest_distance 5 10 20 true = 4 :=
by sorry

end frustum_shortest_distance_l127_127729


namespace average_minutes_per_player_l127_127554

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l127_127554


namespace fraction_ratio_x_div_y_l127_127937

theorem fraction_ratio_x_div_y (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : y / (x + z) = (x - y) / z) 
(h5 : y / (x + z) = x / (y + 2 * z)) :
  x / y = 2 / 3 := 
  sorry

end fraction_ratio_x_div_y_l127_127937


namespace initial_books_from_library_l127_127285

-- Definitions of the problem conditions
def booksGivenAway : ℝ := 23.0
def booksLeft : ℝ := 31.0

-- Statement of the problem, proving that the initial number of books
def initialBooks (x : ℝ) : Prop :=
  x = booksGivenAway + booksLeft

-- Main theorem
theorem initial_books_from_library : initialBooks 54.0 :=
by
  -- Proof pending
  sorry

end initial_books_from_library_l127_127285


namespace insurance_covers_80_percent_l127_127841

-- Definitions from the problem conditions
def cost_per_aid : ℕ := 2500
def num_aids : ℕ := 2
def johns_payment : ℕ := 1000

-- Total cost of hearing aids
def total_cost : ℕ := cost_per_aid * num_aids

-- Insurance payment
def insurance_payment : ℕ := total_cost - johns_payment

-- The theorem to prove
theorem insurance_covers_80_percent :
  (insurance_payment * 100 / total_cost) = 80 :=
by
  sorry

end insurance_covers_80_percent_l127_127841


namespace last_recess_break_duration_l127_127021

-- Definitions based on the conditions
def first_recess_break : ℕ := 15
def second_recess_break : ℕ := 15
def lunch_break : ℕ := 30
def total_outside_class_time : ℕ := 80

-- The theorem we need to prove
theorem last_recess_break_duration :
  total_outside_class_time = first_recess_break + second_recess_break + lunch_break + 20 :=
sorry

end last_recess_break_duration_l127_127021


namespace sangeun_initial_money_l127_127976

theorem sangeun_initial_money :
  ∃ (X : ℝ), 
  ((X / 2 - 2000) / 2 - 2000 = 0) ∧ 
  X = 12000 :=
by sorry

end sangeun_initial_money_l127_127976


namespace division_result_l127_127957

theorem division_result (x : ℕ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end division_result_l127_127957


namespace find_certain_number_l127_127899

theorem find_certain_number (x certain_number : ℤ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (certain_number + 62 + 98 + 124 + x) / 5 = 78) : 
  certain_number = 106 := 
by 
  sorry

end find_certain_number_l127_127899


namespace abs_sq_lt_self_iff_l127_127188

theorem abs_sq_lt_self_iff {x : ℝ} : abs x * abs x < x ↔ (0 < x ∧ x < 1) ∨ (x < -1) :=
by
  sorry

end abs_sq_lt_self_iff_l127_127188


namespace new_commission_percentage_l127_127486

theorem new_commission_percentage
  (fixed_salary : ℝ)
  (total_sales : ℝ)
  (sales_threshold : ℝ)
  (previous_commission_rate : ℝ)
  (additional_earnings : ℝ)
  (prev_commission : ℝ)
  (extra_sales : ℝ)
  (new_commission : ℝ)
  (new_remuneration : ℝ) :
  fixed_salary = 1000 →
  total_sales = 12000 →
  sales_threshold = 4000 →
  previous_commission_rate = 0.05 →
  additional_earnings = 600 →
  prev_commission = previous_commission_rate * total_sales →
  extra_sales = total_sales - sales_threshold →
  new_remuneration = fixed_salary + new_commission * extra_sales →
  new_remuneration = prev_commission + additional_earnings →
  new_commission = 2.5 / 100 :=
by
  intros
  sorry

end new_commission_percentage_l127_127486


namespace smallest_positive_n_l127_127626

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l127_127626


namespace least_number_to_subtract_l127_127066

theorem least_number_to_subtract (n : ℕ) (p : ℕ) (hdiv : p = 47) (hn : n = 929) 
: ∃ k, n - 44 = k * p := by
  sorry

end least_number_to_subtract_l127_127066


namespace f_1984_can_be_any_real_l127_127303

noncomputable def f : ℤ → ℝ := sorry

axiom f_condition : ∀ (x y : ℤ), f (x - y^2) = f x + (y^2 - 2 * x) * f y

theorem f_1984_can_be_any_real
    (a : ℝ)
    (h : f 1 = a) : f 1984 = 1984^2 * a := sorry

end f_1984_can_be_any_real_l127_127303


namespace divides_power_of_odd_l127_127072

theorem divides_power_of_odd (k : ℕ) (hk : k % 2 = 1) (n : ℕ) (hn : n ≥ 1) : 2^(n + 2) ∣ (k^(2^n) - 1) :=
by
  sorry

end divides_power_of_odd_l127_127072


namespace sum_mobile_phone_keypad_l127_127174

/-- The numbers on a standard mobile phone keypad are 0 through 9. -/
def mobile_phone_keypad : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The sum of all the numbers on a standard mobile phone keypad is 45. -/
theorem sum_mobile_phone_keypad : mobile_phone_keypad.sum = 45 := by
  sorry

end sum_mobile_phone_keypad_l127_127174


namespace exponent_of_5_in_30_fact_l127_127993

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l127_127993


namespace geometric_sequence_problem_l127_127646

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * (Real.log x)
  else (Real.log x) / x

theorem geometric_sequence_problem
  (a : ℕ → ℝ) 
  (r : ℝ)
  (h1 : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h2 : a 3 * a 4 * a 5 = 1)
  (h3 : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1) :
  a 1 = Real.exp 2 :=
sorry

end geometric_sequence_problem_l127_127646


namespace problem_solution_l127_127278

theorem problem_solution (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 :=
by
  sorry

end problem_solution_l127_127278


namespace tom_buys_oranges_l127_127237

theorem tom_buys_oranges (o a : ℕ) (h₁ : o + a = 7) (h₂ : (90 * o + 60 * a) % 100 = 0) : o = 6 := 
by 
  sorry

end tom_buys_oranges_l127_127237


namespace factorize_problem1_factorize_problem2_l127_127763

-- Problem 1: Factorization of 4x^2 - 16
theorem factorize_problem1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Factorization of a^2b - 4ab + 4b
theorem factorize_problem2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2) ^ 2 :=
by
  sorry

end factorize_problem1_factorize_problem2_l127_127763


namespace original_time_between_maintenance_checks_l127_127526

theorem original_time_between_maintenance_checks (x : ℝ) 
  (h1 : 2 * x = 60) : x = 30 := sorry

end original_time_between_maintenance_checks_l127_127526


namespace sum_of_squares_gt_five_l127_127421

theorem sum_of_squares_gt_five (a b c : ℝ) (h : a + b + c = 4) : a^2 + b^2 + c^2 > 5 :=
sorry

end sum_of_squares_gt_five_l127_127421


namespace find_line_eq_l127_127930

-- Definitions for the conditions
def passes_through_M (l : ℝ × ℝ) : Prop :=
  l = (1, 2)

def segment_intercepted_length (l : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ,
    ∀ p : ℝ × ℝ, l p → ((4 * p.1 + 3 * p.2 + 1 = 0 ∨ 4 * p.1 + 3 * p.2 + 6 = 0) ∧ (A = p ∨ B = p)) ∧
    dist A B = Real.sqrt 2

-- Predicates for the lines to be proven
def line_eq1 (p : ℝ × ℝ) : Prop :=
  p.1 + 7 * p.2 = 15

def line_eq2 (p : ℝ × ℝ) : Prop :=
  7 * p.1 - p.2 = 5

-- The proof problem statement
theorem find_line_eq (l : ℝ × ℝ → Prop) :
  passes_through_M (1, 2) →
  segment_intercepted_length l →
  (∀ p, l p → line_eq1 p) ∨ (∀ p, l p → line_eq2 p) :=
by
  sorry

end find_line_eq_l127_127930


namespace three_digit_log3_eq_whole_and_log3_log9_eq_whole_l127_127276

noncomputable def logBase (b : ℝ) (x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem three_digit_log3_eq_whole_and_log3_log9_eq_whole (n : ℕ) (hn : 100 ≤ n ∧ n ≤ 999) (hlog3 : ∃ x : ℤ, logBase 3 n = x) (hlog3log9 : ∃ k : ℤ, logBase 3 n + logBase 9 n = k) :
  n = 729 := sorry

end three_digit_log3_eq_whole_and_log3_log9_eq_whole_l127_127276


namespace tree_sidewalk_space_l127_127255

theorem tree_sidewalk_space (num_trees : ℕ) (tree_distance: ℝ) (total_road_length: ℝ): 
  num_trees = 13 → 
  tree_distance = 12 → 
  total_road_length = 157 → 
  (total_road_length - tree_distance * (num_trees - 1)) / num_trees = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end tree_sidewalk_space_l127_127255


namespace greatest_4_digit_base7_divisible_by_7_l127_127487

-- Definitions and conditions
def is_base7_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 7, d < 7

def is_4_digit_base7 (n : ℕ) : Prop :=
  is_base7_number n ∧ 343 ≤ n ∧ n < 2401 -- 343 = 7^3 (smallest 4-digit base 7) and 2401 = 7^4

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Proof problem statement
theorem greatest_4_digit_base7_divisible_by_7 :
  ∃ (n : ℕ), is_4_digit_base7 n ∧ is_divisible_by_7 n ∧ n = 2346 :=
sorry

end greatest_4_digit_base7_divisible_by_7_l127_127487


namespace marie_keeps_lollipops_l127_127889

def total_lollipops (raspberry mint blueberry coconut : ℕ) : ℕ :=
  raspberry + mint + blueberry + coconut

def lollipops_per_friend (total friends : ℕ) : ℕ :=
  total / friends

def lollipops_kept (total friends : ℕ) : ℕ :=
  total % friends

theorem marie_keeps_lollipops :
  lollipops_kept (total_lollipops 75 132 9 315) 13 = 11 :=
by
  sorry

end marie_keeps_lollipops_l127_127889


namespace geometric_sequence_expression_l127_127995

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 2 = 1)
(h2 : a 3 * a 5 = 2 * a 7) : a n = 1 / 2 ^ (n - 2) :=
sorry

end geometric_sequence_expression_l127_127995


namespace line_passes_through_point_l127_127823

theorem line_passes_through_point :
  ∀ (m : ℝ), (∃ y : ℝ, y - 2 = m * (-1) + m) :=
by
  intros m
  use 2
  sorry

end line_passes_through_point_l127_127823


namespace fraction_identity_l127_127339

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end fraction_identity_l127_127339


namespace time_to_office_l127_127096

theorem time_to_office (S T : ℝ) (h1 : T > 0) (h2 : S > 0) 
    (h : S * (T + 15) = (4/5) * S * T) :
    T = 75 := by
  sorry

end time_to_office_l127_127096


namespace number_of_marks_for_passing_l127_127855

theorem number_of_marks_for_passing (T P : ℝ) 
  (h1 : 0.40 * T = P - 40) 
  (h2 : 0.60 * T = P + 20) 
  (h3 : 0.45 * T = P - 10) :
  P = 160 :=
by
  sorry

end number_of_marks_for_passing_l127_127855


namespace find_value_of_a_l127_127958

theorem find_value_of_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53 ^ 2017 + a) % 13 = 0) : a = 12 := 
by 
  sorry

end find_value_of_a_l127_127958


namespace solution_set_of_inequality_l127_127997

theorem solution_set_of_inequality (x : ℝ) :
  (|x| - 2) * (x - 1) ≥ 0 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l127_127997


namespace larger_value_algebraic_expression_is_2_l127_127508

noncomputable def algebraic_expression (a b c d x : ℝ) : ℝ :=
  x^2 + a + b + c * d * x

theorem larger_value_algebraic_expression_is_2
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : x = 1 ∨ x = -1) :
  max (algebraic_expression a b c d 1) (algebraic_expression a b c d (-1)) = 2 :=
by
  -- Proof is omitted.
  sorry

end larger_value_algebraic_expression_is_2_l127_127508


namespace train_cross_time_l127_127714

noncomputable def speed_kmh := 72
noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def length_train := 180
noncomputable def length_bridge := 270
noncomputable def total_distance := length_train + length_bridge
noncomputable def time_to_cross := total_distance / speed_mps

theorem train_cross_time :
  time_to_cross = 22.5 := 
sorry

end train_cross_time_l127_127714


namespace how_many_bananas_l127_127228

theorem how_many_bananas (total_fruit apples oranges : ℕ) 
  (h_total : total_fruit = 12) (h_apples : apples = 3) (h_oranges : oranges = 5) :
  total_fruit - apples - oranges = 4 :=
by
  sorry

end how_many_bananas_l127_127228


namespace Andrew_is_19_l127_127744

-- Define individuals and their relationships
def Andrew_age (Bella_age : ℕ) : ℕ := Bella_age - 5
def Bella_age (Carlos_age : ℕ) : ℕ := Carlos_age + 4
def Carlos_age : ℕ := 20

-- Formulate the problem statement
theorem Andrew_is_19 : Andrew_age (Bella_age Carlos_age) = 19 :=
by
  sorry

end Andrew_is_19_l127_127744


namespace chessboard_number_determination_l127_127589

theorem chessboard_number_determination (d_n : ℤ) (a_n b_n a_1 b_1 c_0 d_0 : ℤ) :
  (∀ i j : ℤ, d_n + a_n = b_n + a_1 + b_1 - (c_0 + d_0) → 
   a_n + b_n = c_0 + d_0 + d_n) →
  ∃ x : ℤ, x = a_1 + b_1 - d_n ∧ 
  x = d_n + (a_1 - c_0) + (b_1 - d_0) :=
by
  sorry

end chessboard_number_determination_l127_127589


namespace value_of_expression_l127_127347

-- Define the variables and conditions
variables (x y : ℝ)
axiom h1 : x + 2 * y = 4
axiom h2 : x * y = -8

-- Define the statement to be proven
theorem value_of_expression : x^2 + 4 * y^2 = 48 := 
by
  sorry

end value_of_expression_l127_127347


namespace avg_age_of_community_l127_127214

def ratio_of_populations (w m : ℕ) : Prop := w * 2 = m * 3
def avg_age (total_age population : ℚ) : ℚ := total_age / population

theorem avg_age_of_community 
    (k : ℕ)
    (total_women : ℕ := 3 * k) 
    (total_men : ℕ := 2 * k)
    (total_children : ℚ := (2 * k : ℚ) / 3)
    (avg_women_age : ℚ := 40)
    (avg_men_age : ℚ := 36)
    (avg_children_age : ℚ := 10)
    (total_women_age : ℚ := 40 * (3 * k))
    (total_men_age : ℚ := 36 * (2 * k))
    (total_children_age : ℚ := 10 * (total_children)) : 
    avg_age (total_women_age + total_men_age + total_children_age) (total_women + total_men + total_children) = 35 := 
    sorry

end avg_age_of_community_l127_127214


namespace find_larger_number_l127_127950

variable {x y : ℕ} 

theorem find_larger_number (h_ratio : 4 * x = 3 * y) (h_sum : x + y + 100 = 500) : y = 1600 / 7 := by 
  sorry

end find_larger_number_l127_127950


namespace lily_received_books_l127_127198

def mike_books : ℕ := 45
def corey_books : ℕ := 2 * mike_books
def mike_gave_lily : ℕ := 10
def corey_gave_lily : ℕ := mike_gave_lily + 15
def lily_books_received : ℕ := mike_gave_lily + corey_gave_lily

theorem lily_received_books : lily_books_received = 35 := by
  sorry

end lily_received_books_l127_127198


namespace simplify_expression_l127_127424

theorem simplify_expression (a b : ℝ) (h : a + b < 0) : 
  |a + b - 1| - |3 - (a + b)| = -2 :=
by 
  sorry

end simplify_expression_l127_127424


namespace maximum_number_of_intersections_of_150_lines_is_7171_l127_127378

def lines_are_distinct (L : ℕ → Type) : Prop := 
  ∀ n m : ℕ, n ≠ m → L n ≠ L m

def lines_parallel_to_each_other (L : ℕ → Type) (k : ℕ) : Prop :=
  ∀ n m : ℕ, n ≠ m → L (k * n) = L (k * m)

def lines_pass_through_point_B (L : ℕ → Type) (B : Type) (k : ℕ) : Prop :=
  ∀ n : ℕ, L (k * n - 4) = B

def lines_not_parallel (L : ℕ → Type) (k1 k2 : ℕ) : Prop :=
  ∀ n m : ℕ, L (k1 * n) ≠ L (k2 * m)

noncomputable def max_points_of_intersection
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : ℕ :=
  7171

theorem maximum_number_of_intersections_of_150_lines_is_7171
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : max_points_of_intersection L B k1 k2 h_distinct h_parallel1 h_parallel2 h_pass_through_B h_not_parallel = 7171 := 
  by 
  sorry

end maximum_number_of_intersections_of_150_lines_is_7171_l127_127378


namespace solve_system_of_equations_l127_127469

theorem solve_system_of_equations :
  ∃ (x y z w : ℤ), 
    x - y + z - w = 2 ∧
    x^2 - y^2 + z^2 - w^2 = 6 ∧
    x^3 - y^3 + z^3 - w^3 = 20 ∧
    x^4 - y^4 + z^4 - w^4 = 66 ∧
    (x, y, z, w) = (1, 3, 0, 2) := 
  by
    sorry

end solve_system_of_equations_l127_127469


namespace number_of_girls_l127_127849

theorem number_of_girls
  (total_boys : ℕ)
  (total_boys_eq : total_boys = 10)
  (fraction_girls_reading : ℚ)
  (fraction_girls_reading_eq : fraction_girls_reading = 5/6)
  (fraction_boys_reading : ℚ)
  (fraction_boys_reading_eq : fraction_boys_reading = 4/5)
  (total_not_reading : ℕ)
  (total_not_reading_eq : total_not_reading = 4)
  (G : ℝ)
  (remaining_girls_reading : (1 - fraction_girls_reading) * G = 2)
  (remaining_boys_not_reading : (1 - fraction_boys_reading) * total_boys = 2)
  (remaining_total_not_reading : 2 + 2 = total_not_reading)
  : G = 12 :=
by
  sorry

end number_of_girls_l127_127849


namespace joyce_gave_apples_l127_127286

theorem joyce_gave_apples : 
  ∀ (initial_apples final_apples given_apples : ℕ), (initial_apples = 75) ∧ (final_apples = 23) → (given_apples = initial_apples - final_apples) → (given_apples = 52) :=
by
  intros
  sorry

end joyce_gave_apples_l127_127286


namespace trig_values_same_terminal_side_l127_127655

-- Statement: The trigonometric function values of angles with the same terminal side are equal.
theorem trig_values_same_terminal_side (θ₁ θ₂ : ℝ) (h : ∃ k : ℤ, θ₂ = θ₁ + 2 * k * π) :
  (∀ f : ℝ -> ℝ, f θ₁ = f θ₂) :=
by
  sorry

end trig_values_same_terminal_side_l127_127655


namespace find_remainder_l127_127311

def mod_condition : Prop :=
  (764251 % 31 = 5) ∧
  (1095223 % 31 = 6) ∧
  (1487719 % 31 = 1) ∧
  (263311 % 31 = 0) ∧
  (12097 % 31 = 25) ∧
  (16817 % 31 = 26) ∧
  (23431 % 31 = 0) ∧
  (305643 % 31 = 20)

theorem find_remainder (h : mod_condition) : 
  ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := 
by
  sorry

end find_remainder_l127_127311


namespace part_I_part_II_l127_127047

variable (α : ℝ)

-- The given conditions.
variable (h1 : π < α)
variable (h2 : α < (3 * π) / 2)
variable (h3 : Real.sin α = -4/5)

-- Part (I): Prove cos α = -3/5
theorem part_I : Real.cos α = -3/5 :=
sorry

-- Part (II): Prove sin 2α + 3 tan α = 24/25 + 4
theorem part_II : Real.sin (2 * α) + 3 * Real.tan α = 24/25 + 4 :=
sorry

end part_I_part_II_l127_127047


namespace lucas_age_correct_l127_127345

variable (Noah_age : ℕ) (Mia_age : ℕ) (Lucas_age : ℕ)

-- Conditions
axiom h1 : Noah_age = 12
axiom h2 : Mia_age = Noah_age + 5
axiom h3 : Lucas_age = Mia_age - 6

-- Goal
theorem lucas_age_correct : Lucas_age = 11 := by
  sorry

end lucas_age_correct_l127_127345


namespace total_distance_from_A_through_B_to_C_l127_127745

noncomputable def distance_A_B_map : ℝ := 120
noncomputable def distance_B_C_map : ℝ := 70
noncomputable def map_scale : ℝ := 10 -- km per cm

noncomputable def distance_A_B := distance_A_B_map * map_scale -- Distance from City A to City B in km
noncomputable def distance_B_C := distance_B_C_map * map_scale -- Distance from City B to City C in km
noncomputable def total_distance := distance_A_B + distance_B_C -- Total distance in km

theorem total_distance_from_A_through_B_to_C :
  total_distance = 1900 := by
  sorry

end total_distance_from_A_through_B_to_C_l127_127745


namespace inconsistency_proof_l127_127004

-- Let TotalBoys be the number of boys, which is 120
def TotalBoys := 120

-- Let AverageMarks be the average marks obtained by 120 boys, which is 40
def AverageMarks := 40

-- Let PassedBoys be the number of boys who passed, which is 125
def PassedBoys := 125

-- Let AverageMarksFailed be the average marks of failed boys, which is 15
def AverageMarksFailed := 15

-- We need to prove the inconsistency
theorem inconsistency_proof :
  ∀ (P : ℝ), 
    (TotalBoys * AverageMarks = PassedBoys * P + (TotalBoys - PassedBoys) * AverageMarksFailed) →
    False :=
by
  intro P h
  sorry

end inconsistency_proof_l127_127004


namespace find_ratio_l127_127702

noncomputable def decagon_area : ℝ := 12
noncomputable def area_below_PQ : ℝ := 6
noncomputable def unit_square_area : ℝ := 1
noncomputable def triangle_base : ℝ := 6
noncomputable def area_above_PQ : ℝ := 6
noncomputable def XQ : ℝ := 4
noncomputable def QY : ℝ := 2

theorem find_ratio {XQ QY : ℝ} (h1 : decagon_area = 12) (h2 : area_below_PQ = 6)
                   (h3 : unit_square_area = 1) (h4 : triangle_base = 6)
                   (h5 : area_above_PQ = 6) (h6 : XQ + QY = 6) :
  XQ / QY = 2 := by { sorry }

end find_ratio_l127_127702


namespace difference_one_third_0_333_l127_127260

theorem difference_one_third_0_333 :
  let one_third : ℚ := 1 / 3
  let three_hundred_thirty_three_thousandth : ℚ := 333 / 1000
  one_third - three_hundred_thirty_three_thousandth = 1 / 3000 :=
by
  sorry

end difference_one_third_0_333_l127_127260


namespace possible_values_expression_l127_127790

theorem possible_values_expression 
  (a b : ℝ) 
  (h₁ : a^2 = 16) 
  (h₂ : |b| = 3) 
  (h₃ : ab < 0) : 
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := 
by 
  sorry

end possible_values_expression_l127_127790


namespace garrison_provisions_last_initially_l127_127633

noncomputable def garrison_initial_provisions (x : ℕ) : Prop :=
  ∃ x : ℕ, 2000 * (x - 21) = 3300 * 20 ∧ x = 54

theorem garrison_provisions_last_initially :
  garrison_initial_provisions 54 :=
by
  sorry

end garrison_provisions_last_initially_l127_127633


namespace intersection_M_N_l127_127448

def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {0} :=
  sorry

end intersection_M_N_l127_127448


namespace fraction_calculation_l127_127965

theorem fraction_calculation :
  ((1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25) := 
by 
  sorry

end fraction_calculation_l127_127965


namespace train_travel_distance_l127_127125

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l127_127125


namespace probability_is_correct_l127_127114

-- Given definitions
def total_marbles : ℕ := 100
def red_marbles : ℕ := 35
def white_marbles : ℕ := 30
def green_marbles : ℕ := 10

-- Probe the probability
noncomputable def probability_red_white_green : ℚ :=
  (red_marbles + white_marbles + green_marbles : ℚ) / total_marbles

-- The theorem we need to prove
theorem probability_is_correct :
  probability_red_white_green = 0.75 := by
  sorry

end probability_is_correct_l127_127114


namespace milk_left_l127_127645

theorem milk_left (initial_milk : ℝ) (milk_james : ℝ) (milk_maria : ℝ) :
  initial_milk = 5 → milk_james = 15 / 4 → milk_maria = 3 / 4 → 
  initial_milk - (milk_james + milk_maria) = 1 / 2 :=
by
  intros h_initial h_james h_maria
  rw [h_initial, h_james, h_maria]
  -- The calculation would be performed here.
  sorry

end milk_left_l127_127645


namespace value_of_x_y_l127_127658

noncomputable def real_ln : ℝ → ℝ := sorry

theorem value_of_x_y (x y : ℝ) (h : 3 * x - y ≤ real_ln (x + 2 * y - 3) + real_ln (2 * x - 3 * y + 5)) :
  x + y = 16 / 7 :=
sorry

end value_of_x_y_l127_127658


namespace brooke_butter_price_l127_127207

variables (price_per_gallon_of_milk : ℝ)
variables (gallons_to_butter_conversion : ℝ)
variables (number_of_cows : ℕ)
variables (milk_per_cow : ℝ)
variables (number_of_customers : ℕ)
variables (milk_demand_per_customer : ℝ)
variables (total_earnings : ℝ)

theorem brooke_butter_price :
    price_per_gallon_of_milk = 3 →
    gallons_to_butter_conversion = 2 →
    number_of_cows = 12 →
    milk_per_cow = 4 →
    number_of_customers = 6 →
    milk_demand_per_customer = 6 →
    total_earnings = 144 →
    (total_earnings - number_of_customers * milk_demand_per_customer * price_per_gallon_of_milk) /
    (number_of_cows * milk_per_cow - number_of_customers * milk_demand_per_customer) *
    gallons_to_butter_conversion = 1.50 :=
by { sorry }

end brooke_butter_price_l127_127207


namespace height_percentage_difference_l127_127709

theorem height_percentage_difference (A B : ℝ) (h : B = A * (4/3)) : 
  (A * (1/3) / B) * 100 = 25 := by
  sorry

end height_percentage_difference_l127_127709


namespace number_of_pieces_of_tape_l127_127144

variable (length_of_tape : ℝ := 8.8)
variable (overlap : ℝ := 0.5)
variable (total_length : ℝ := 282.7)

theorem number_of_pieces_of_tape : 
  ∃ (N : ℕ), total_length = length_of_tape + (N - 1) * (length_of_tape - overlap) ∧ N = 34 :=
sorry

end number_of_pieces_of_tape_l127_127144


namespace correct_substitution_l127_127476

theorem correct_substitution (x y : ℝ) (h1 : y = 1 - x) (h2 : x - 2 * y = 4) : x - 2 * (1 - x) = 4 → x - 2 + 2 * x = 4 := by
  sorry

end correct_substitution_l127_127476


namespace distance_between_points_l127_127868

open Real

theorem distance_between_points :
  let P := (1, 3)
  let Q := (-5, 7)
  dist P Q = 2 * sqrt 13 :=
by
  let P := (1, 3)
  let Q := (-5, 7)
  sorry

end distance_between_points_l127_127868


namespace largest_number_4597_l127_127274

def swap_adjacent_digits (n : ℕ) : ℕ :=
  sorry

def max_number_after_two_swaps_subtract_100 (n : ℕ) : ℕ :=
  -- logic to perform up to two adjacent digit swaps and subtract 100
  sorry

theorem largest_number_4597 : max_number_after_two_swaps_subtract_100 4597 = 4659 :=
  sorry

end largest_number_4597_l127_127274


namespace exists_six_digit_no_identical_six_endings_l127_127989

theorem exists_six_digit_no_identical_six_endings :
  ∃ (A : ℕ), (100000 ≤ A ∧ A < 1000000) ∧ ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 500000) → 
  (∀ d, d ≠ 0 → d < 10 → (k * A) % 1000000 ≠ d * 111111) :=
by
  sorry

end exists_six_digit_no_identical_six_endings_l127_127989


namespace range_of_a_intersection_nonempty_range_of_a_intersection_A_l127_127258

noncomputable def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a_intersection_nonempty (a : ℝ) : (A a ∩ B ≠ ∅) ↔ (a < -1 ∨ a > 2) :=
sorry

theorem range_of_a_intersection_A (a : ℝ) : (A a ∩ B = A a) ↔ (a < -4 ∨ a > 5) :=
sorry

end range_of_a_intersection_nonempty_range_of_a_intersection_A_l127_127258


namespace inequality_proof_l127_127346

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end inequality_proof_l127_127346


namespace scott_awards_l127_127665

theorem scott_awards (S : ℕ) 
  (h1 : ∃ J, J = 3 * S)
  (h2 : ∃ B, B = 2 * (3 * S) ∧ B = 24) : S = 4 := 
by 
  sorry

end scott_awards_l127_127665


namespace polynomial_identity_l127_127404

theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = 
  (a - b) * (b - c) * (c - a) * (a + b + c) :=
sorry

end polynomial_identity_l127_127404


namespace one_non_congruent_triangle_with_perimeter_10_l127_127595

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end one_non_congruent_triangle_with_perimeter_10_l127_127595


namespace average_of_second_pair_l127_127322

theorem average_of_second_pair (S : ℝ) (S1 : ℝ) (S3 : ℝ) (S2 : ℝ) (avg : ℝ) :
  (S / 6 = 3.95) →
  (S1 / 2 = 3.8) →
  (S3 / 2 = 4.200000000000001) →
  (S = S1 + S2 + S3) →
  (avg = S2 / 2) →
  avg = 3.85 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end average_of_second_pair_l127_127322


namespace polynomial_horner_v4_value_l127_127964

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define Horner's Rule step by step for x = 2
def horner_eval (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  let v4 := v3 * x + 240
  v4

-- Prove that the value of v4 when x = 2 is 80
theorem polynomial_horner_v4_value : horner_eval 2 = 80 := by
  sorry

end polynomial_horner_v4_value_l127_127964


namespace odd_n_divisibility_l127_127854

theorem odd_n_divisibility (n : ℤ) : (∃ a : ℤ, n ∣ 4 * a^2 - 1) ↔ (n % 2 ≠ 0) :=
by
  sorry

end odd_n_divisibility_l127_127854


namespace distance_focus_to_asymptote_l127_127624

theorem distance_focus_to_asymptote (m : ℝ) (x y : ℝ) (h1 : (x^2) / 9 - (y^2) / m = 1) 
  (h2 : (Real.sqrt 14) / 3 = (Real.sqrt (9 + m)) / 3) : 
  ∃ d : ℝ, d = Real.sqrt 5 := 
by 
  sorry

end distance_focus_to_asymptote_l127_127624


namespace oprod_eval_l127_127985

def oprod (a b : ℕ) : ℕ :=
  (a * 2 + b) / 2

theorem oprod_eval : oprod (oprod 4 6) 8 = 11 :=
by
  -- Definitions given in conditions
  let r := (4 * 2 + 6) / 2
  have h1 : oprod 4 6 = r := by rfl
  let s := (r * 2 + 8) / 2
  have h2 : oprod r 8 = s := by rfl
  exact (show s = 11 from sorry)

end oprod_eval_l127_127985


namespace inscribed_sphere_volume_l127_127443

theorem inscribed_sphere_volume
  (a : ℝ)
  (h_cube_surface_area : 6 * a^2 = 24) :
  (4 / 3) * Real.pi * (a / 2)^3 = (4 / 3) * Real.pi :=
by
  -- sorry to skip the actual proof
  sorry

end inscribed_sphere_volume_l127_127443


namespace solve_for_y_l127_127437

theorem solve_for_y (y : ℚ) : 
  y + 5 / 8 = 2 / 9 + 1 / 2 → 
  y = 7 / 72 := 
by 
  intro h1
  sorry

end solve_for_y_l127_127437


namespace negation_of_existence_is_universal_l127_127244

theorem negation_of_existence_is_universal (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
sorry

end negation_of_existence_is_universal_l127_127244


namespace area_enclosed_by_trajectory_of_P_l127_127033

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Definition of fixed points A and B
def A : Point := { x := -3, y := 0 }
def B : Point := { x := 3, y := 0 }

-- Condition for the ratio of distances
def ratio_condition (P : Point) : Prop :=
  ((P.x + 3)^2 + P.y^2) / ((P.x - 3)^2 + P.y^2) = 1 / 4

-- Definition of a circle based on the derived condition in the solution
def circle_eq (P : Point) : Prop :=
  (P.x + 5)^2 + P.y^2 = 16

-- Theorem stating the area enclosed by the trajectory of point P is 16π
theorem area_enclosed_by_trajectory_of_P : 
  (∀ P : Point, ratio_condition P → circle_eq P) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  sorry

end area_enclosed_by_trajectory_of_P_l127_127033


namespace annual_concert_tickets_l127_127751

theorem annual_concert_tickets (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : NS = 60 :=
by
  sorry

end annual_concert_tickets_l127_127751


namespace years_before_marriage_l127_127882

theorem years_before_marriage {wedding_anniversary : ℕ} 
  (current_year : ℕ) (met_year : ℕ) (years_before_dating : ℕ) :
  wedding_anniversary = 20 →
  current_year = 2025 →
  met_year = 2000 →
  years_before_dating = 2 →
  met_year + years_before_dating + (current_year - met_year - wedding_anniversary) = current_year - wedding_anniversary - years_before_dating + wedding_anniversary - current_year :=
by
  sorry

end years_before_marriage_l127_127882


namespace total_collisions_100_balls_l127_127740

def num_of_collisions (n: ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_collisions_100_balls :
  num_of_collisions 100 = 4950 :=
by
  sorry

end total_collisions_100_balls_l127_127740


namespace max_n_value_l127_127145

noncomputable def max_n_avoid_repetition : ℕ :=
sorry

theorem max_n_value : max_n_avoid_repetition = 155 :=
by
  -- Assume factorial reciprocals range from 80 to 99
  -- We show no n-digit segments are repeated in such range while n <= 155
  sorry

end max_n_value_l127_127145


namespace average_eq_35_implies_y_eq_50_l127_127532

theorem average_eq_35_implies_y_eq_50 (y : ℤ) (h : (15 + 30 + 45 + y) / 4 = 35) : y = 50 :=
by
  sorry

end average_eq_35_implies_y_eq_50_l127_127532


namespace factor_expression_l127_127256

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l127_127256


namespace inscribed_sphere_radius_l127_127169

-- Define the distances from points X and Y to the faces of the tetrahedron
variable (X_AB X_AD X_AC X_BC : ℝ)
variable (Y_AB Y_AD Y_AC Y_BC : ℝ)

-- Setting the given distances in the problem
axiom dist_X_AB : X_AB = 14
axiom dist_X_AD : X_AD = 11
axiom dist_X_AC : X_AC = 29
axiom dist_X_BC : X_BC = 8

axiom dist_Y_AB : Y_AB = 15
axiom dist_Y_AD : Y_AD = 13
axiom dist_Y_AC : Y_AC = 25
axiom dist_Y_BC : Y_BC = 11

-- The theorem to prove that the radius of the inscribed sphere of the tetrahedron is 17
theorem inscribed_sphere_radius : 
  ∃ r : ℝ, r = 17 ∧ 
  (∀ (d_X_AB d_X_AD d_X_AC d_X_BC d_Y_AB d_Y_AD d_Y_AC d_Y_BC: ℝ),
    d_X_AB = 14 ∧ d_X_AD = 11 ∧ d_X_AC = 29 ∧ d_X_BC = 8 ∧
    d_Y_AB = 15 ∧ d_Y_AD = 13 ∧ d_Y_AC = 25 ∧ d_Y_BC = 11 → 
    r = 17) :=
sorry

end inscribed_sphere_radius_l127_127169


namespace count_integers_in_solution_set_l127_127893

-- Define the predicate for the condition given in the problem
def condition (x : ℝ) : Prop := abs (x - 3) ≤ 4.5

-- Define the list of integers within the range of the condition
def solution_set : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7]

-- Prove that the number of integers satisfying the condition is 8
theorem count_integers_in_solution_set : solution_set.length = 8 :=
by
  sorry

end count_integers_in_solution_set_l127_127893


namespace homer_second_try_points_l127_127369

theorem homer_second_try_points (x : ℕ) :
  400 + x + 2 * x = 1390 → x = 330 :=
by
  sorry

end homer_second_try_points_l127_127369


namespace students_helped_on_third_day_l127_127905

theorem students_helped_on_third_day (books_total : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day4 : ℕ) (books_day3 : ℕ) :
  books_total = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day4 = 9 →
  books_day3 = books_total - ((students_day1 + students_day2 + students_day4) * books_per_student) →
  books_day3 / books_per_student = 6 :=
by
  sorry

end students_helped_on_third_day_l127_127905


namespace compound_difference_l127_127481

noncomputable def monthly_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let monthly_rate := annual_rate / 12
  let periods := 12 * years
  principal * (1 + monthly_rate) ^ periods

noncomputable def semi_annual_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let semi_annual_rate := annual_rate / 2
  let periods := 2 * years
  principal * (1 + semi_annual_rate) ^ periods

theorem compound_difference (principal : ℝ) (annual_rate : ℝ) (years : ℝ) :
  monthly_compound_amount principal annual_rate years - semi_annual_compound_amount principal annual_rate years = 23.36 :=
by
  let principal := 8000
  let annual_rate := 0.08
  let years := 3
  sorry

end compound_difference_l127_127481


namespace find_a_l127_127444

theorem find_a
  (x y a : ℝ)
  (h1 : x + y = 1)
  (h2 : 2 * x + y = 0)
  (h3 : a * x - 3 * y = 0) :
  a = -6 :=
sorry

end find_a_l127_127444


namespace worker_idle_days_l127_127195

variable (x y : ℤ)

theorem worker_idle_days :
  (30 * x - 5 * y = 500) ∧ (x + y = 60) → y = 38 :=
by
  intros h
  have h1 : 30 * x - 5 * y = 500 := h.left
  have h2 : x + y = 60 := h.right
  sorry

end worker_idle_days_l127_127195


namespace prob_l127_127662

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (2 + 1 / x))

theorem prob (x1 x2 x3 : ℝ) (h1 : x1 = 0) 
  (h2 : 2 + 1 / x2 = 0) 
  (h3 : 2 + 1 / (2 + 1 / x3) = 0) : 
  x1 + x2 + x3 = -9 / 10 := 
sorry

end prob_l127_127662


namespace remainder_when_55_times_57_divided_by_8_l127_127329

theorem remainder_when_55_times_57_divided_by_8 :
  (55 * 57) % 8 = 7 :=
by
  -- Insert the proof here
  sorry

end remainder_when_55_times_57_divided_by_8_l127_127329


namespace red_bowl_values_possible_l127_127632

theorem red_bowl_values_possible (r b y : ℕ) 
(h1 : r + b + y = 27)
(h2 : 15 * r + 3 * b + 18 * y = 378) : 
  r = 11 ∨ r = 16 ∨ r = 21 := 
  sorry

end red_bowl_values_possible_l127_127632


namespace son_l127_127062

noncomputable def my_age_in_years : ℕ := 84
noncomputable def total_age_in_years : ℕ := 140
noncomputable def months_in_a_year : ℕ := 12
noncomputable def weeks_in_a_year : ℕ := 52

theorem son's_age_in_weeks (G_d S_m G_m S_y : ℕ) (G_y : ℚ) :
  G_d = S_m →
  G_m = my_age_in_years * months_in_a_year →
  G_y = (G_m : ℚ) / months_in_a_year →
  G_y + S_y + my_age_in_years = total_age_in_years →
  S_y * weeks_in_a_year = 2548 :=
by
  intros h1 h2 h3 h4
  sorry

end son_l127_127062


namespace number_of_blue_stamps_l127_127433

theorem number_of_blue_stamps (
    red_stamps : ℕ := 20
) (
    yellow_stamps : ℕ := 7
) (
    price_per_red_stamp : ℝ := 1.1
) (
    price_per_blue_stamp : ℝ := 0.8
) (
    total_earnings : ℝ := 100
) (
    price_per_yellow_stamp : ℝ := 2
) : red_stamps = 20 ∧ yellow_stamps = 7 ∧ price_per_red_stamp = 1.1 ∧ price_per_blue_stamp = 0.8 ∧ total_earnings = 100 ∧ price_per_yellow_stamp = 2 → ∃ (blue_stamps : ℕ), blue_stamps = 80 :=
by
  sorry

end number_of_blue_stamps_l127_127433


namespace find_m_l127_127490

theorem find_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2 * y - 4 = 0) →
  (∀ x y : ℝ, x - 2 * y + m = 0) →
  (m = 7 ∨ m = -3) :=
by
  sorry

end find_m_l127_127490


namespace length_PQ_calc_l127_127242

noncomputable def length_PQ 
  (F : ℝ × ℝ) 
  (P Q : ℝ × ℝ) 
  (hF : F = (1, 0)) 
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1) 
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1) 
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1) 
  (hx1x2 : P.1 + Q.1 = 9) : ℝ :=
|P.1 - Q.1|

theorem length_PQ_calc : ∀ F P Q
  (hF : F = (1, 0))
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1)
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1)
  (hx1x2 : P.1 + Q.1 = 9),
  length_PQ F P Q hF hP_on_parabola hQ_on_parabola hLine_through_focus hx1x2 = 11 := 
by
  sorry

end length_PQ_calc_l127_127242


namespace domain_of_h_l127_127356

theorem domain_of_h (x : ℝ) : |x - 5| + |x + 3| ≠ 0 := by
  sorry

end domain_of_h_l127_127356


namespace cosine_inequality_l127_127890

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y :=
sorry

end cosine_inequality_l127_127890


namespace price_reduction_l127_127647

variable (T : ℝ) -- The original price of the television
variable (first_discount : ℝ) -- First discount in percentage
variable (second_discount : ℝ) -- Second discount in percentage

theorem price_reduction (h1 : first_discount = 0.4) (h2 : second_discount = 0.4) : 
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.64 :=
by
  sorry

end price_reduction_l127_127647


namespace simplify_product_of_fractions_l127_127157

theorem simplify_product_of_fractions :
  (25 / 24) * (18 / 35) * (56 / 45) = (50 / 3) :=
by sorry

end simplify_product_of_fractions_l127_127157


namespace division_by_n_minus_1_squared_l127_127837

theorem division_by_n_minus_1_squared (n : ℕ) (h : n > 2) : (n ^ (n - 1) - 1) % ((n - 1) ^ 2) = 0 :=
sorry

end division_by_n_minus_1_squared_l127_127837


namespace smallest_sum_of_inverses_l127_127845

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l127_127845


namespace combined_area_correct_l127_127561

def popsicle_stick_length_gino : ℚ := 9 / 2
def popsicle_stick_width_gino : ℚ := 2 / 5
def popsicle_stick_length_me : ℚ := 6
def popsicle_stick_width_me : ℚ := 3 / 5

def number_of_sticks_gino : ℕ := 63
def number_of_sticks_me : ℕ := 50

def side_length_square : ℚ := number_of_sticks_gino / 4 * popsicle_stick_length_gino
def area_square : ℚ := side_length_square ^ 2

def length_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_length_me
def width_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_width_me
def area_rectangle : ℚ := length_rectangle * width_rectangle

def combined_area : ℚ := area_square + area_rectangle

theorem combined_area_correct : combined_area = 6806.25 := by
  sorry

end combined_area_correct_l127_127561


namespace ratio_minutes_l127_127359

theorem ratio_minutes (x : ℝ) : 
  (12 / 8) = (6 / (x * 60)) → x = 1 / 15 :=
by
  sorry

end ratio_minutes_l127_127359


namespace daniel_biked_more_l127_127220

def miles_biked_after_4_hours_more (speed_plain_daniel : ℕ) (speed_plain_elsa : ℕ) (time_plain : ℕ) 
(speed_hilly_daniel : ℕ) (speed_hilly_elsa : ℕ) (time_hilly : ℕ) : ℕ :=
(speed_plain_daniel * time_plain + speed_hilly_daniel * time_hilly) - 
(speed_plain_elsa * time_plain + speed_hilly_elsa * time_hilly)

theorem daniel_biked_more : miles_biked_after_4_hours_more 20 18 3 16 15 1 = 7 :=
by
  sorry

end daniel_biked_more_l127_127220


namespace point_B_in_first_quadrant_l127_127748

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_B_in_first_quadrant : is_first_quadrant (1, 2) :=
by
  sorry

end point_B_in_first_quadrant_l127_127748


namespace find_number_l127_127067

theorem find_number :
  ∃ n : ℕ, n * (1 / 7)^2 = 7^3 :=
by
  sorry

end find_number_l127_127067


namespace not_possible_to_tile_l127_127037

theorem not_possible_to_tile 
    (m n : ℕ) (a b : ℕ)
    (h_m : m = 2018)
    (h_n : n = 2020)
    (h_a : a = 5)
    (h_b : b = 8) :
    ¬ ∃ k : ℕ, k * (a * b) = m * n := by
sorry

end not_possible_to_tile_l127_127037


namespace find_total_income_l127_127430

theorem find_total_income (I : ℝ)
  (h1 : 0.6 * I + 0.3 * I + 0.005 * (I - (0.6 * I + 0.3 * I)) + 50000 = I) : 
  I = 526315.79 :=
by
  sorry

end find_total_income_l127_127430


namespace money_given_to_each_friend_l127_127908

-- Define the conditions
def initial_amount : ℝ := 20.10
def money_spent_on_sweets : ℝ := 1.05
def amount_left : ℝ := 17.05
def number_of_friends : ℝ := 2.0

-- Theorem statement
theorem money_given_to_each_friend :
  (initial_amount - amount_left - money_spent_on_sweets) / number_of_friends = 1.00 :=
by
  sorry

end money_given_to_each_friend_l127_127908


namespace sufficient_but_not_necessary_condition_l127_127384

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 3) → (x ≥ 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l127_127384


namespace b_minus_d_sq_value_l127_127419

theorem b_minus_d_sq_value 
  (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2 * a - 3 * b + c + 4 * d = 17) :
  (b - d) ^ 2 = 25 :=
by
  sorry

end b_minus_d_sq_value_l127_127419


namespace ellipse_eccentricity_l127_127467

theorem ellipse_eccentricity (a b c : ℝ) (h_eq : a * a = 16) (h_b : b * b = 12) (h_c : c * c = a * a - b * b) :
  c / a = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l127_127467


namespace emily_strawberry_harvest_l127_127372

-- Define the dimensions of the garden
def garden_length : ℕ := 10
def garden_width : ℕ := 7

-- Define the planting density
def plants_per_sqft : ℕ := 3

-- Define the yield per plant
def strawberries_per_plant : ℕ := 12

-- Define the expected number of strawberries
def expected_strawberries : ℕ := 2520

-- Theorem statement to prove the total number of strawberries
theorem emily_strawberry_harvest :
  garden_length * garden_width * plants_per_sqft * strawberries_per_plant = expected_strawberries :=
by
  -- Proof goes here (for now, we use sorry to indicate the proof is omitted)
  sorry

end emily_strawberry_harvest_l127_127372


namespace abc_geq_inequality_l127_127786

open Real

theorem abc_geq_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end abc_geq_inequality_l127_127786


namespace find_digit_l127_127861

theorem find_digit (p q r : ℕ) (hq : p ≠ q) (hr : p ≠ r) (hq' : q ≠ r) 
    (hp_pos : 0 < p ∧ p < 10)
    (hq_pos : 0 < q ∧ q < 10)
    (hr_pos : 0 < r ∧ r < 10)
    (h1 : 10 * p + q = 17)
    (h2 : 10 * p + r = 13)
    (h3 : p + q + r = 11) : 
    q = 7 :=
sorry

end find_digit_l127_127861


namespace haley_trees_grown_after_typhoon_l127_127631

def original_trees := 9
def trees_died := 4
def current_trees := 10

theorem haley_trees_grown_after_typhoon (newly_grown_trees : ℕ) :
  (original_trees - trees_died) + newly_grown_trees = current_trees → newly_grown_trees = 5 :=
by
  sorry

end haley_trees_grown_after_typhoon_l127_127631


namespace y_decreases_as_x_less_than_4_l127_127813

theorem y_decreases_as_x_less_than_4 (x : ℝ) : (x < 4) → ((x - 4)^2 + 3 < (4 - 4)^2 + 3) :=
by
  sorry

end y_decreases_as_x_less_than_4_l127_127813


namespace compute_a_d_sum_l127_127271

variables {a1 a2 a3 d1 d2 d3 : ℝ}

theorem compute_a_d_sum
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a_d_sum_l127_127271


namespace kerosene_cost_l127_127179

theorem kerosene_cost (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = A / 2)
  (h3 : C * 2 = 24 / 100) :
  24 = 24 := 
sorry

end kerosene_cost_l127_127179


namespace pirate_treasure_l127_127142

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l127_127142


namespace probability_red_or_white_l127_127190

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 := 
  sorry

end probability_red_or_white_l127_127190


namespace range_of_a_for_monotonic_f_l127_127577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a^2 * x^2 + a * x

theorem range_of_a_for_monotonic_f (a : ℝ) : 
  (∀ x, 1 < x → f a x ≤ f a (1 : ℝ)) ↔ (a ≤ -1 / 2 ∨ 1 ≤ a) := 
by
  sorry

end range_of_a_for_monotonic_f_l127_127577


namespace find_difference_of_segments_l127_127833

theorem find_difference_of_segments 
  (a b c d x y : ℝ)
  (h1 : a + b = 70)
  (h2 : b + c = 90)
  (h3 : c + d = 130)
  (h4 : a + d = 110)
  (hx_y_sum : x + y = 130)
  (hx_c : x = c)
  (hy_d : y = d) : 
  |x - y| = 13 :=
sorry

end find_difference_of_segments_l127_127833


namespace g_of_3_eq_seven_over_two_l127_127723

theorem g_of_3_eq_seven_over_two :
  ∀ f g : ℝ → ℝ,
  (∀ x, f x = (2 * x + 3) / (x - 1)) →
  (∀ x, g x = (x + 4) / (x - 1)) →
  g 3 = 7 / 2 :=
by
  sorry

end g_of_3_eq_seven_over_two_l127_127723


namespace same_solution_m_iff_m_eq_2_l127_127046

theorem same_solution_m_iff_m_eq_2 (m y : ℝ) (h1 : my - 2 = 4) (h2 : y - 2 = 1) : m = 2 :=
by {
  sorry
}

end same_solution_m_iff_m_eq_2_l127_127046


namespace light_intensity_after_glass_pieces_minimum_glass_pieces_l127_127025

theorem light_intensity_after_glass_pieces (a : ℝ) (x : ℕ) : 
  (y : ℝ) = a * (0.9 ^ x) :=
sorry

theorem minimum_glass_pieces (a : ℝ) (x : ℕ) : 
  a * (0.9 ^ x) < a / 3 ↔ x ≥ 11 :=
sorry

end light_intensity_after_glass_pieces_minimum_glass_pieces_l127_127025


namespace next_sales_amount_l127_127663

theorem next_sales_amount
  (royalties1: ℝ)
  (sales1: ℝ)
  (royalties2: ℝ)
  (percentage_decrease: ℝ)
  (X: ℝ)
  (h1: royalties1 = 4)
  (h2: sales1 = 20)
  (h3: royalties2 = 9)
  (h4: percentage_decrease = 58.333333333333336 / 100)
  (h5: royalties2 / X = royalties1 / sales1 - ((royalties1 / sales1) * percentage_decrease)): 
  X = 108 := 
  by 
    -- Proof omitted
    sorry

end next_sales_amount_l127_127663


namespace cost_comparison_l127_127264

-- Definitions based on the given conditions
def suit_price : ℕ := 200
def tie_price : ℕ := 40
def num_suits : ℕ := 20
def discount_rate : ℚ := 0.9

-- Define cost expressions for the two options
def option1_cost (x : ℕ) : ℕ :=
  (suit_price * num_suits) + (tie_price * (x - num_suits))

def option2_cost (x : ℕ) : ℚ :=
  ((suit_price * num_suits + tie_price * x) * discount_rate : ℚ)

-- Main theorem to prove the given answers
theorem cost_comparison (x : ℕ) (hx : x > 20) :
  option1_cost x = 40 * x + 3200 ∧
  option2_cost x = 3600 + 36 * x ∧
  (x = 30 → option1_cost 30 < option2_cost 30) :=
by
  sorry

end cost_comparison_l127_127264


namespace largest_prime_factor_among_numbers_l127_127743

-- Definitions of the numbers with their prime factors
def num1 := 39
def num2 := 51
def num3 := 77
def num4 := 91
def num5 := 121

def prime_factors (n : ℕ) : List ℕ := sorry  -- Placeholder for the prime factors function

-- Prime factors for the given numbers
def factors_num1 := prime_factors num1
def factors_num2 := prime_factors num2
def factors_num3 := prime_factors num3
def factors_num4 := prime_factors num4
def factors_num5 := prime_factors num5

-- Extract the largest prime factor from a list of factors
def largest_prime_factor (factors : List ℕ) : ℕ := sorry  -- Placeholder for the largest_prime_factor function

-- Largest prime factors for each number
def largest_prime_factor_num1 := largest_prime_factor factors_num1
def largest_prime_factor_num2 := largest_prime_factor factors_num2
def largest_prime_factor_num3 := largest_prime_factor factors_num3
def largest_prime_factor_num4 := largest_prime_factor factors_num4
def largest_prime_factor_num5 := largest_prime_factor factors_num5

theorem largest_prime_factor_among_numbers :
  largest_prime_factor_num2 = 17 ∧
  largest_prime_factor_num1 = 13 ∧
  largest_prime_factor_num3 = 11 ∧
  largest_prime_factor_num4 = 13 ∧
  largest_prime_factor_num5 = 11 ∧
  (largest_prime_factor_num2 > largest_prime_factor_num1) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num3) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num4) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num5)
:= by
  -- skeleton proof, details to be filled in
  sorry

end largest_prime_factor_among_numbers_l127_127743


namespace proof_f_3_eq_9_ln_3_l127_127397

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

theorem proof_f_3_eq_9_ln_3 (a : ℝ) (h : deriv (deriv (f a)) 1 = 3) : f a 3 = 9 * Real.log 3 :=
by
  sorry

end proof_f_3_eq_9_ln_3_l127_127397


namespace equal_share_is_168_l127_127705

namespace StrawberryProblem

def brother_baskets : ℕ := 3
def strawberries_per_basket : ℕ := 15
def brother_strawberries : ℕ := brother_baskets * strawberries_per_basket

def kimberly_multiplier : ℕ := 8
def kimberly_strawberries : ℕ := kimberly_multiplier * brother_strawberries

def parents_difference : ℕ := 93
def parents_strawberries : ℕ := kimberly_strawberries - parents_difference

def total_strawberries : ℕ := kimberly_strawberries + brother_strawberries + parents_strawberries
def total_people : ℕ := 4

def equal_share : ℕ := total_strawberries / total_people

theorem equal_share_is_168 :
  equal_share = 168 := by
  -- We state that for the given problem conditions,
  -- the total number of strawberries divided equally among the family members results in 168 strawberries per person.
  sorry

end StrawberryProblem

end equal_share_is_168_l127_127705


namespace remainder_when_ab_div_by_40_l127_127516

theorem remainder_when_ab_div_by_40 (a b : ℤ) (k j : ℤ)
  (ha : a = 80 * k + 75)
  (hb : b = 90 * j + 85):
  (a + b) % 40 = 0 :=
by sorry

end remainder_when_ab_div_by_40_l127_127516


namespace number_of_girls_in_school_l127_127204

theorem number_of_girls_in_school :
  ∃ G B : ℕ, 
    G + B = 1600 ∧
    (G * 200 / 1600) - 20 = (B * 200 / 1600) ∧
    G = 860 :=
by
  sorry

end number_of_girls_in_school_l127_127204


namespace length_of_first_platform_l127_127783

noncomputable def speed (distance time : ℕ) :=
  distance / time

theorem length_of_first_platform 
  (L : ℕ) (train_length : ℕ) (time1 time2 : ℕ) (platform2_length : ℕ) (speed : ℕ) 
  (H1 : L + train_length = speed * time1) 
  (H2 : platform2_length + train_length = speed * time2) 
  (train_length_eq : train_length = 30) 
  (time1_eq : time1 = 12) 
  (time2_eq : time2 = 15) 
  (platform2_length_eq : platform2_length = 120) 
  (speed_eq : speed = 10) : L = 90 :=
by
  sorry

end length_of_first_platform_l127_127783


namespace total_stock_worth_is_15000_l127_127511

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the conditions
def stock_condition_1 := 0.20 * X -- Worth of 20% of the stock
def stock_condition_2 := 0.10 * (0.20 * X) -- Profit from 20% of the stock
def stock_condition_3 := 0.80 * X -- Worth of 80% of the stock
def stock_condition_4 := 0.05 * (0.80 * X) -- Loss from 80% of the stock
def overall_loss := 0.04 * X - 0.02 * X

-- The question rewritten as a theorem statement
theorem total_stock_worth_is_15000 (h1 : overall_loss X = 300) : X = 15000 :=
by sorry

end total_stock_worth_is_15000_l127_127511


namespace cost_price_of_article_l127_127182

theorem cost_price_of_article :
  ∃ (C : ℝ), 
  (∃ (G : ℝ), C + G = 500 ∧ C + 1.15 * G = 570) ∧ 
  C = (100 / 3) :=
by sorry

end cost_price_of_article_l127_127182


namespace incorrect_conclusion_l127_127147

variable {a b c : ℝ}

theorem incorrect_conclusion
  (h1 : a^2 + a * b = c)
  (h2 : a * b + b^2 = c + 5) :
  ¬(2 * c + 5 < 0) ∧ ¬(∃ k, a^2 - b^2 ≠ k) ∧ ¬(a = b ∨ a = -b) ∧ ¬(b / a > 1) :=
by sorry

end incorrect_conclusion_l127_127147


namespace swimming_speed_l127_127739

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (swim_time : ℝ) (distance : ℝ) :
  water_speed = 8 →
  swim_time = 8 →
  distance = 16 →
  distance = (v - water_speed) * swim_time →
  v = 10 := 
by
  intros h1 h2 h3 h4
  sorry

end swimming_speed_l127_127739


namespace number_of_tests_initially_l127_127980

theorem number_of_tests_initially (n : ℕ) (h1 : (90 * n) / n = 90)
  (h2 : ((90 * n) - 75) / (n - 1) = 95) : n = 4 :=
sorry

end number_of_tests_initially_l127_127980


namespace arithmetic_sqrt_of_sqrt_16_l127_127039

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l127_127039


namespace sally_earnings_in_dozens_l127_127284

theorem sally_earnings_in_dozens (earnings_per_house : ℕ) (houses_cleaned : ℕ) (dozens_of_dollars : ℕ) : 
  earnings_per_house = 25 ∧ houses_cleaned = 96 → dozens_of_dollars = 200 := 
by
  intros h
  sorry

end sally_earnings_in_dozens_l127_127284


namespace simplify_polynomial_l127_127022

theorem simplify_polynomial : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomial_l127_127022


namespace inequality_holds_for_unit_interval_l127_127600

theorem inequality_holds_for_unit_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
    5 * (x ^ 2 + y ^ 2) ^ 2 ≤ 4 + (x + y) ^ 4 :=
by
    sorry

end inequality_holds_for_unit_interval_l127_127600


namespace distance_relationship_l127_127215

noncomputable def plane_parallel (α β : Type) : Prop := sorry
noncomputable def line_in_plane (m : Type) (α : Type) : Prop := sorry
noncomputable def point_on_line (A : Type) (m : Type) : Prop := sorry
noncomputable def distance (A B : Type) : ℝ := sorry
noncomputable def distance_point_to_line (A : Type) (n : Type) : ℝ := sorry
noncomputable def distance_between_lines (m n : Type) : ℝ := sorry

variables (α β m n A B : Type)
variables (a b c : ℝ)

axiom plane_parallel_condition : plane_parallel α β
axiom line_m_in_alpha : line_in_plane m α
axiom line_n_in_beta : line_in_plane n β
axiom point_A_on_m : point_on_line A m
axiom point_B_on_n : point_on_line B n
axiom distance_a : a = distance A B
axiom distance_b : b = distance_point_to_line A n
axiom distance_c : c = distance_between_lines m n

theorem distance_relationship : c ≤ b ∧ b ≤ a := by
  sorry

end distance_relationship_l127_127215


namespace combine_sum_l127_127213

def A (n m : Nat) : Nat := n.factorial / (n - m).factorial
def C (n m : Nat) : Nat := n.factorial / (m.factorial * (n - m).factorial)

theorem combine_sum (n m : Nat) (hA : A n m = 272) (hC : C n m = 136) : m + n = 19 := by
  sorry

end combine_sum_l127_127213


namespace eq_of_div_eq_div_l127_127881

theorem eq_of_div_eq_div {a b c : ℝ} (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end eq_of_div_eq_div_l127_127881


namespace elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l127_127132

-- Define the initial conditions
def sea_level_drop : ℝ := 397
def submerged_depth_initial : ℝ := 5000
def height_diff_mauna_kea_everest : ℝ := 358

-- Define intermediate calculations based on conditions
def submerged_depth_adjusted : ℝ := submerged_depth_initial - sea_level_drop
def total_height_mauna_kea : ℝ := 2 * submerged_depth_adjusted
def elevation_above_sea_level_mauna_kea : ℝ := total_height_mauna_kea - submerged_depth_initial
def elevation_mount_everest : ℝ := total_height_mauna_kea - height_diff_mauna_kea_everest

-- Define the proof statements
theorem elevation_above_sea_level_mauna_kea_correct :
  elevation_above_sea_level_mauna_kea = 4206 := by
  sorry

theorem total_height_mauna_kea_correct :
  total_height_mauna_kea = 9206 := by
  sorry

theorem elevation_mount_everest_correct :
  elevation_mount_everest = 8848 := by
  sorry

end elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l127_127132


namespace sale_in_first_month_l127_127261

theorem sale_in_first_month 
  (sale_2 : ℝ) (sale_3 : ℝ) (sale_4 : ℝ) (sale_5 : ℝ) (sale_6 : ℝ) (avg_sale : ℝ)
  (h_sale_2 : sale_2 = 5366) (h_sale_3 : sale_3 = 5808) 
  (h_sale_4 : sale_4 = 5399) (h_sale_5 : sale_5 = 6124) 
  (h_sale_6 : sale_6 = 4579) (h_avg_sale : avg_sale = 5400) :
  ∃ (sale_1 : ℝ), sale_1 = 5124 :=
by
  let total_sales := avg_sale * 6
  let known_sales := sale_2 + sale_3 + sale_4 + sale_5 + sale_6
  have h_total_sales : total_sales = 32400 := by sorry
  have h_known_sales : known_sales = 27276 := by sorry
  let sale_1 := total_sales - known_sales
  use sale_1
  have h_sale_1 : sale_1 = 5124 := by sorry
  exact h_sale_1

end sale_in_first_month_l127_127261


namespace incorrect_transformation_l127_127265

theorem incorrect_transformation (a b : ℤ) : ¬ (a / b = (a + 1) / (b + 1)) :=
sorry

end incorrect_transformation_l127_127265


namespace total_cookies_eaten_l127_127672

-- Definitions of the cookies eaten
def charlie_cookies := 15
def father_cookies := 10
def mother_cookies := 5

-- The theorem to prove the total number of cookies eaten
theorem total_cookies_eaten : charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end total_cookies_eaten_l127_127672


namespace geometric_series_second_term_l127_127165

theorem geometric_series_second_term 
  (r : ℚ) (S : ℚ) (a : ℚ) (second_term : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 16)
  (h3 : S = a / (1 - r))
  : second_term = a * r := 
sorry

end geometric_series_second_term_l127_127165


namespace square_flag_side_length_side_length_of_square_flags_is_4_l127_127097

theorem square_flag_side_length 
  (total_fabric : ℕ)
  (fabric_left : ℕ)
  (num_square_flags : ℕ)
  (num_wide_flags : ℕ)
  (num_tall_flags : ℕ)
  (wide_flag_length : ℕ)
  (wide_flag_width : ℕ)
  (tall_flag_length : ℕ)
  (tall_flag_width : ℕ)
  (fabric_used_on_wide_and_tall_flags : ℕ)
  (fabric_used_on_all_flags : ℕ)
  (fabric_used_on_square_flags : ℕ)
  (square_flag_area : ℕ)
  (side_length : ℕ) : Prop :=
  total_fabric = 1000 ∧
  fabric_left = 294 ∧
  num_square_flags = 16 ∧
  num_wide_flags = 20 ∧
  num_tall_flags = 10 ∧
  wide_flag_length = 5 ∧
  wide_flag_width = 3 ∧
  tall_flag_length = 5 ∧
  tall_flag_width = 3 ∧
  fabric_used_on_wide_and_tall_flags = (num_wide_flags + num_tall_flags) * (wide_flag_length * wide_flag_width) ∧
  fabric_used_on_all_flags = total_fabric - fabric_left ∧
  fabric_used_on_square_flags = fabric_used_on_all_flags - fabric_used_on_wide_and_tall_flags ∧
  square_flag_area = fabric_used_on_square_flags / num_square_flags ∧
  side_length = Int.sqrt square_flag_area ∧
  side_length = 4

theorem side_length_of_square_flags_is_4 : 
  square_flag_side_length 1000 294 16 20 10 5 3 5 3 450 706 256 16 4 :=
  by
    sorry

end square_flag_side_length_side_length_of_square_flags_is_4_l127_127097


namespace triangle_area_ratio_l127_127585

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l127_127585


namespace g_triple_3_eq_31_l127_127210

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 1 else 2 * n - 3

theorem g_triple_3_eq_31 : g (g (g 3)) = 31 := by
  sorry

end g_triple_3_eq_31_l127_127210


namespace complex_multiplication_value_l127_127504

theorem complex_multiplication_value (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_value_l127_127504


namespace geometric_progression_product_l127_127942

theorem geometric_progression_product (n : ℕ) (S R : ℝ) (hS : S > 0) (hR : R > 0)
  (h_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ S = a * (q^n - 1) / (q - 1))
  (h_reciprocal_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ R = (1 - q^n) / (a * q^(n-1) * (q - 1))) :
  ∃ P : ℝ, P = (S / R)^(n / 2) := sorry

end geometric_progression_product_l127_127942


namespace fans_received_all_offers_l127_127722

theorem fans_received_all_offers :
  let hotdog_freq := 90
  let soda_freq := 45
  let popcorn_freq := 60
  let stadium_capacity := 4500
  let lcm_freq := Nat.lcm (Nat.lcm hotdog_freq soda_freq) popcorn_freq
  (stadium_capacity / lcm_freq) = 25 :=
by
  sorry

end fans_received_all_offers_l127_127722


namespace abs_inequality_solution_l127_127109

theorem abs_inequality_solution (x : ℝ) :
  |2 * x - 2| + |2 * x + 4| < 10 ↔ x ∈ Set.Ioo (-4 : ℝ) (2 : ℝ) := 
by sorry

end abs_inequality_solution_l127_127109


namespace a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l127_127758

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l127_127758


namespace charlotte_avg_speed_l127_127666

def distance : ℕ := 60  -- distance in miles
def time : ℕ := 6       -- time in hours

theorem charlotte_avg_speed : (distance / time) = 10 := by
  sorry

end charlotte_avg_speed_l127_127666


namespace green_beans_weight_l127_127555

/-- 
    Mary uses plastic grocery bags that can hold a maximum of twenty pounds. 
    She buys some green beans, 6 pounds milk, and twice the amount of carrots as green beans. 
    She can fit 2 more pounds of groceries in that bag. 
    Prove that the weight of green beans she bought is equal to 4 pounds.
-/
theorem green_beans_weight (G : ℕ) (H1 : ∀ g : ℕ, g + 6 + 2 * g ≤ 20 - 2) : G = 4 :=
by 
  have H := H1 4
  sorry

end green_beans_weight_l127_127555


namespace determine_a_if_fx_odd_l127_127343

theorem determine_a_if_fx_odd (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = 2^x + a * 2^(-x)) (h2 : ∀ x, f (-x) = -f x) : a = -1 :=
by
  sorry

end determine_a_if_fx_odd_l127_127343


namespace quadratic_has_distinct_real_roots_l127_127155

theorem quadratic_has_distinct_real_roots {m : ℝ} (hm : m > 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 - 2 = m) ∧ (x2^2 + x2 - 2 = m) :=
by
  sorry

end quadratic_has_distinct_real_roots_l127_127155


namespace smallest_value_of_3a_plus_1_l127_127442

theorem smallest_value_of_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 2 = 2) : 3 * a + 1 = -5/4 :=
by
  sorry

end smallest_value_of_3a_plus_1_l127_127442


namespace remainder_sum_of_six_primes_div_seventh_prime_l127_127354

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l127_127354


namespace Polyas_probability_relation_l127_127267

variable (Z : ℕ → ℤ → ℝ)

theorem Polyas_probability_relation (n : ℕ) (k : ℤ) :
  Z n k = (1/2) * (Z (n-1) (k-1) + Z (n-1) (k+1)) :=
by
  sorry

end Polyas_probability_relation_l127_127267


namespace angle_measure_l127_127099

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l127_127099


namespace correct_conclusions_l127_127742

def pos_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

def sum_of_n_terms (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) * S (n+1) = 9

def second_term_less_than_3 (a S : ℕ → ℝ) : Prop :=
  a 1 < 3

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

def exists_term_less_than_1_over_100 (a : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, a n < 1/100

theorem correct_conclusions (a S : ℕ → ℝ) :
  pos_sequence a → sum_of_n_terms S a →
  second_term_less_than_3 a S ∧ (¬(∀ q : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n = r * q ^ n)) ∧ is_decreasing_sequence a ∧ exists_term_less_than_1_over_100 a :=
sorry

end correct_conclusions_l127_127742


namespace primes_p_plus_10_plus_14_l127_127537

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_p_plus_10_plus_14 (p : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (p + 10)) 
  (h3 : is_prime (p + 14)) 
  : p = 3 := sorry

end primes_p_plus_10_plus_14_l127_127537


namespace original_number_l127_127801

theorem original_number (x : ℝ) (h : 1.2 * x = 1080) : x = 900 := by
  sorry

end original_number_l127_127801


namespace complex_product_l127_127641

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end complex_product_l127_127641


namespace necessary_but_not_sufficient_l127_127963

open Set

namespace Mathlib

noncomputable def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) :=
by
  sorry

end Mathlib

end necessary_but_not_sufficient_l127_127963


namespace binary_to_base4_conversion_l127_127438

theorem binary_to_base4_conversion : 
  let binary := (1*2^7 + 1*2^6 + 0*2^5 + 1*2^4 + 1*2^3 + 0*2^2 + 0*2^1 + 1*2^0) 
  let base4 := (3*4^3 + 1*4^2 + 2*4^1 + 1*4^0)
  binary = base4 := by
  sorry

end binary_to_base4_conversion_l127_127438


namespace remainder_of_power_mod_five_l127_127118

theorem remainder_of_power_mod_five : (4 ^ 11) % 5 = 4 :=
by
  sorry

end remainder_of_power_mod_five_l127_127118


namespace each_wolf_needs_to_kill_one_deer_l127_127163

-- Conditions
def wolves_out_hunting : ℕ := 4
def additional_wolves : ℕ := 16
def wolves_total : ℕ := wolves_out_hunting + additional_wolves
def meat_per_wolf_per_day : ℕ := 8
def days_no_hunt : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat needed for all wolves over five days.
def total_meat_needed : ℕ := wolves_total * meat_per_wolf_per_day * days_no_hunt
-- Calculate total number of deer needed to meet the meat requirement.
def deer_needed : ℕ := total_meat_needed / meat_per_deer
-- Calculate number of deer each hunting wolf needs to kill.
def deer_per_wolf : ℕ := deer_needed / wolves_out_hunting

-- The proof statement
theorem each_wolf_needs_to_kill_one_deer : deer_per_wolf = 1 := 
by { sorry }

end each_wolf_needs_to_kill_one_deer_l127_127163


namespace simplify_expression_l127_127176

theorem simplify_expression (x : ℝ) : 
  x^2 * (4 * x^3 - 3 * x + 1) - 6 * (x^3 - 3 * x^2 + 4 * x - 5) = 
  4 * x^5 - 9 * x^3 + 19 * x^2 - 24 * x + 30 := by
  sorry

end simplify_expression_l127_127176


namespace cost_of_five_trip_ticket_l127_127321

-- Variables for the costs of the tickets
variables (x y z : ℕ)

-- Conditions from the problem
def condition1 : Prop := 5 * x > y
def condition2 : Prop := 4 * y > z
def condition3 : Prop := z + 3 * y = 33
def condition4 : Prop := 20 + 3 * 5 = 35

-- The theorem to prove
theorem cost_of_five_trip_ticket (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z y) (h4 : condition4) : y = 5 := 
by
  sorry

end cost_of_five_trip_ticket_l127_127321


namespace power_computation_l127_127040

theorem power_computation :
  16^10 * 8^6 / 4^22 = 16384 :=
by
  sorry

end power_computation_l127_127040


namespace total_pieces_10_rows_l127_127717

-- Define the conditions for the rods
def rod_seq (n : ℕ) : ℕ := 3 * n

-- Define the sum of the arithmetic sequence for rods
def sum_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

-- Define the conditions for the connectors
def connector_seq (n : ℕ) : ℕ := n + 1

-- Define the sum of the arithmetic sequence for connectors
def sum_connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Define the total pieces calculation
def total_pieces (n : ℕ) : ℕ := sum_rods n + sum_connectors (n + 1)

-- The target statement
theorem total_pieces_10_rows : total_pieces 10 = 231 :=
by
  sorry

end total_pieces_10_rows_l127_127717


namespace same_points_among_teams_l127_127955

theorem same_points_among_teams :
  ∀ (n : Nat), n = 28 → 
  ∀ (G D N : Nat), G = 378 → D >= 284 → N <= 94 →
  (∃ (team_scores : Fin n → Int), ∀ (i j : Fin n), i ≠ j → team_scores i = team_scores j) := by
sorry

end same_points_among_teams_l127_127955


namespace benjie_is_6_years_old_l127_127459

-- Definitions based on conditions
def margo_age_in_3_years := 4
def years_until_then := 3
def age_difference := 5

-- Current age of Margo
def margo_current_age := margo_age_in_3_years - years_until_then

-- Current age of Benjie
def benjie_current_age := margo_current_age + age_difference

-- The theorem we need to prove
theorem benjie_is_6_years_old : benjie_current_age = 6 :=
by
  -- Proof
  sorry

end benjie_is_6_years_old_l127_127459


namespace casey_correct_result_l127_127640

variable (x : ℕ)

def incorrect_divide (x : ℕ) := x / 7
def incorrect_subtract (x : ℕ) := x - 20
def incorrect_result := 19

def reverse_subtract (x : ℕ) := x + 20
def reverse_divide (x : ℕ) := x * 7

def correct_multiply (x : ℕ) := x * 7
def correct_add (x : ℕ) := x + 20

theorem casey_correct_result (x : ℕ) (h : reverse_divide (reverse_subtract incorrect_result) = x) : correct_add (correct_multiply x) = 1931 :=
by
  sorry

end casey_correct_result_l127_127640


namespace valid_parameterizations_l127_127733

noncomputable def line_equation (x y : ℝ) : Prop := y = (5/3) * x + 1

def parametrize_A (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (3 + t * 3, 6 + t * 5) ∧ line_equation x y

def parametrize_D (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (-1 + t * 3, -2/3 + t * 5) ∧ line_equation x y

theorem valid_parameterizations : parametrize_A t ∧ parametrize_D t :=
by
  -- Proof steps are skipped
  sorry

end valid_parameterizations_l127_127733


namespace find_numer_denom_n_l127_127252

theorem find_numer_denom_n (n : ℕ) 
    (h : (2 + n) / (7 + n) = (3 : ℤ) / 4) : n = 13 := sorry

end find_numer_denom_n_l127_127252


namespace lcm_Anthony_Bethany_Casey_Dana_l127_127318

theorem lcm_Anthony_Bethany_Casey_Dana : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 10) = 120 := 
by
  sorry

end lcm_Anthony_Bethany_Casey_Dana_l127_127318


namespace time_jack_first_half_l127_127058

-- Define the conditions
def t_Jill : ℕ := 32
def t_2 : ℕ := 6
def t_Jack : ℕ := t_Jill - 7

-- Define the time Jack took for the first half
def t_1 : ℕ := t_Jack - t_2

-- State the theorem to prove
theorem time_jack_first_half : t_1 = 19 := by
  sorry

end time_jack_first_half_l127_127058


namespace eval_expr_l127_127146

theorem eval_expr : 4 * (8 - 3 + 2) / 2 = 14 := 
by
  sorry

end eval_expr_l127_127146


namespace not_possible_acquaintance_arrangement_l127_127390

-- Definitions and conditions for the problem
def num_people : ℕ := 40
def even_people_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 0 → A ≠ B → true -- A and B have a mutual acquaintance if an even number of people sit between them

def odd_people_not_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 1 → A ≠ B → true -- A and B do not have a mutual acquaintance if an odd number of people sit between them

theorem not_possible_acquaintance_arrangement : ¬ (∀ A B : ℕ, A ≠ B →
  (∀ num_between : ℕ, (num_between % 2 = 0 → even_people_acquainted A B num_between) ∧
  (num_between % 2 = 1 → odd_people_not_acquainted A B num_between))) :=
sorry

end not_possible_acquaintance_arrangement_l127_127390


namespace solution_set_of_inequality_l127_127699

theorem solution_set_of_inequality (x : ℝ) (n : ℕ) (h1 : n ≤ x ∧ x < n + 1 ∧ 0 < n) :
  4 * (⌊x⌋ : ℝ)^2 - 36 * (⌊x⌋ : ℝ) + 45 < 0 ↔ ∃ k : ℕ, (2 ≤ k ∧ k < 8 ∧ ⌊x⌋ = k) :=
by sorry

end solution_set_of_inequality_l127_127699


namespace midpoint_line_l127_127794

theorem midpoint_line (a : ℝ) (P Q M : ℝ × ℝ) (hP : P = (a, 5 * a + 3)) (hQ : Q = (3, -2))
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) : M.2 = 5 * M.1 - 7 := 
sorry

end midpoint_line_l127_127794


namespace distance_travelled_l127_127821

theorem distance_travelled
  (d : ℝ)                   -- distance in kilometers
  (train_speed : ℝ)         -- train speed in km/h
  (ship_speed : ℝ)          -- ship speed in km/h
  (time_difference : ℝ)     -- time difference in hours
  (h1 : train_speed = 48)
  (h2 : ship_speed = 60)
  (h3 : time_difference = 2) :
  d = 480 := 
by
  sorry

end distance_travelled_l127_127821


namespace width_of_domain_of_g_l127_127514

variable (h : ℝ → ℝ) (dom_h : ∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x)

noncomputable def g (x : ℝ) : ℝ := h (x / 3)

theorem width_of_domain_of_g :
  (∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x) →
  (∀ y : ℝ, -30 ≤ y ∧ y ≤ 30 → h (y / 3) = h (y / 3)) →
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧  (∃ w : ℝ, w = b - a ∧ w = 60)) :=
by
  sorry

end width_of_domain_of_g_l127_127514


namespace largest_constant_inequality_l127_127685

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = Real.sqrt (4 / 3) :=
by {
  sorry
}

end largest_constant_inequality_l127_127685


namespace real_root_exists_l127_127724

theorem real_root_exists (a b c : ℝ) :
  (∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b - c) * x + (c - a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c - a) * x + (a - b) = 0) :=
by {
  sorry
}

end real_root_exists_l127_127724


namespace avg_score_all_matches_l127_127917

-- Definitions from the conditions
variable (score1 score2 : ℕ → ℕ) 
variable (avg1 avg2 : ℕ)
variable (count1 count2 : ℕ)

-- Assumptions from the conditions
axiom avg_score1 : avg1 = 30
axiom avg_score2 : avg2 = 40
axiom count1_matches : count1 = 2
axiom count2_matches : count2 = 3

-- The proof statement
theorem avg_score_all_matches : 
  ((score1 0 + score1 1) + (score2 0 + score2 1 + score2 2)) / (count1 + count2) = 36 := 
  sorry

end avg_score_all_matches_l127_127917


namespace engineers_meeting_probability_l127_127279

theorem engineers_meeting_probability :
  ∀ (x y z : ℝ), 
    (0 ≤ x ∧ x ≤ 2) → 
    (0 ≤ y ∧ y ≤ 2) → 
    (0 ≤ z ∧ z ≤ 2) → 
    (abs (x - y) ≤ 0.5) → 
    (abs (y - z) ≤ 0.5) → 
    (abs (z - x) ≤ 0.5) → 
    Π (volume_region : ℝ) (total_volume : ℝ),
    (volume_region = 1.5 * 1.5 * 1.5) → 
    (total_volume = 2 * 2 * 2) → 
    (volume_region / total_volume = 0.421875) :=
by
  intros x y z hx hy hz hxy hyz hzx volume_region total_volume hr ht
  sorry

end engineers_meeting_probability_l127_127279


namespace correct_calculation_l127_127324

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 :=
by
  sorry

end correct_calculation_l127_127324


namespace circumference_of_circle_inscribing_rectangle_l127_127477

theorem circumference_of_circle_inscribing_rectangle (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi := by
  sorry

end circumference_of_circle_inscribing_rectangle_l127_127477


namespace div_by_10_3pow_l127_127507

theorem div_by_10_3pow
    (m : ℤ)
    (n : ℕ)
    (h : (3^n + m) % 10 = 0) :
    (3^(n + 4) + m) % 10 = 0 := by
  sorry

end div_by_10_3pow_l127_127507


namespace range_of_m_common_tangents_with_opposite_abscissas_l127_127569

section part1
variable {x : ℝ}

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def h (m : ℝ) (x : ℝ) := m * f x / Real.sin x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 0 Real.pi, h m x ≥ Real.sqrt 2) ↔ m ∈ Set.Ici (Real.sqrt 2 / Real.exp (Real.pi / 4)) := 
by
  sorry
end part1

section part2
variable {x : ℝ}

noncomputable def g (x : ℝ) := Real.log x
noncomputable def f_tangent_line_at (x₁ : ℝ) (x : ℝ) := Real.exp x₁ * x + (1 - x₁) * Real.exp x₁
noncomputable def g_tangent_line_at (x₂ : ℝ) (x : ℝ) := x / x₂ + Real.log x₂ - 1

theorem common_tangents_with_opposite_abscissas :
  ∃ x₁ x₂ : ℝ, (f_tangent_line_at x₁ = g_tangent_line_at (Real.exp (-x₁))) ∧ (x₁ = -x₂) :=
by
  sorry
end part2

end range_of_m_common_tangents_with_opposite_abscissas_l127_127569


namespace cube_edge_length_l127_127316

-- Define the edge length 'a'
variable (a : ℝ)

-- Given conditions: 6a^2 = 24
theorem cube_edge_length (h : 6 * a^2 = 24) : a = 2 :=
by {
  -- The actual proof would go here, but we use sorry to skip it as per instructions.
  sorry
}

end cube_edge_length_l127_127316


namespace youngest_child_age_l127_127301

variables (child_ages : Fin 5 → ℕ)

def child_ages_eq_intervals (x : ℕ) : Prop :=
  child_ages 0 = x ∧ child_ages 1 = x + 8 ∧ child_ages 2 = x + 16 ∧ child_ages 3 = x + 24 ∧ child_ages 4 = x + 32

def sum_of_ages_eq (child_ages : Fin 5 → ℕ) (sum : ℕ) : Prop :=
  (Finset.univ : Finset (Fin 5)).sum child_ages = sum

theorem youngest_child_age (child_ages : Fin 5 → ℕ) (h1 : ∃ x, child_ages_eq_intervals child_ages x) (h2 : sum_of_ages_eq child_ages 90) :
  ∃ x, x = 2 ∧ child_ages 0 = x :=
sorry

end youngest_child_age_l127_127301


namespace solve_log_sin_eq_l127_127059

noncomputable def log_base (b : ℝ) (a : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem solve_log_sin_eq :
  ∀ x : ℝ, 
  (0 < Real.sin x ∧ Real.sin x < 1) →
  log_base (Real.sin x) 4 * log_base (Real.sin x ^ 2) 2 = 4 →
  ∃ k : ℤ, x = (-1)^k * (Real.pi / 4) + Real.pi * k := 
by
  sorry

end solve_log_sin_eq_l127_127059


namespace find_m_from_expansion_l127_127829

theorem find_m_from_expansion (m n : ℤ) (h : (x : ℝ) → (x + 3) * (x + n) = x^2 + m * x - 21) : m = -4 :=
by
  sorry

end find_m_from_expansion_l127_127829


namespace gaussian_guardians_points_l127_127484

theorem gaussian_guardians_points :
  let Daniel := 7
  let Curtis := 8
  let Sid := 2
  let Emily := 11
  let Kalyn := 6
  let Hyojeong := 12
  let Ty := 1
  let Winston := 7
  Daniel + Curtis + Sid + Emily + Kalyn + Hyojeong + Ty + Winston = 54 :=
by
  sorry

end gaussian_guardians_points_l127_127484


namespace complete_square_add_term_l127_127181

theorem complete_square_add_term (x : ℝ) :
  ∃ (c : ℝ), (c = 4 * x ^ 4 ∨ c = 4 * x ∨ c = -4 * x ∨ c = -1 ∨ c = -4 * x ^2) ∧
  (4 * x ^ 2 + 1 + c) * (4 * x ^ 2 + 1 + c) = (2 * x + 1) * (2 * x + 1) :=
sorry

end complete_square_add_term_l127_127181


namespace total_heads_l127_127401

variables (H C : ℕ)

theorem total_heads (h_hens: H = 22) (h_feet: 2 * H + 4 * C = 140) : H + C = 46 :=
by
  sorry

end total_heads_l127_127401


namespace sum_of_roots_of_quadratic_l127_127079

open Polynomial

theorem sum_of_roots_of_quadratic :
  ∀ (m n : ℝ), (m ≠ n ∧ (∀ x, x^2 + 2*x - 1 = 0 → x = m ∨ x = n)) → m + n = -2 :=
by
  sorry

end sum_of_roots_of_quadratic_l127_127079


namespace unique_zero_in_interval_l127_127689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x ^ 2

theorem unique_zero_in_interval
  (a : ℝ) (ha : a > 0)
  (x₀ : ℝ) (hx₀ : f a x₀ = 0)
  (h_interval : -1 < x₀ ∧ x₀ < 0) :
  Real.exp (-2) < x₀ + 1 ∧ x₀ + 1 < Real.exp (-1) :=
sorry

end unique_zero_in_interval_l127_127689


namespace find_slope_l127_127162

noncomputable def slope_of_first_line
    (m : ℝ)
    (intersect_point : ℝ × ℝ)
    (slope_second_line : ℝ)
    (x_intercept_distance : ℝ) 
    : Prop :=
  let (x₀, y₀) := intersect_point
  let x_intercept_first := (40 * m - 30) / m
  let x_intercept_second := 35
  abs (x_intercept_first - x_intercept_second) = x_intercept_distance

theorem find_slope : ∃ m : ℝ, slope_of_first_line m (40, 30) 6 10 :=
by
  use 2
  sorry

end find_slope_l127_127162


namespace skiing_ratio_l127_127594

theorem skiing_ratio (S : ℕ) (H1 : 4000 ≤ 12000) (H2 : S + 4000 = 12000) : S / 4000 = 2 :=
by {
  sorry
}

end skiing_ratio_l127_127594


namespace number_of_classes_l127_127720

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 := by
  sorry

end number_of_classes_l127_127720


namespace cookies_sum_l127_127275

theorem cookies_sum (C : ℕ) (h1 : C % 6 = 5) (h2 : C % 9 = 7) (h3 : C < 80) :
  C = 29 :=
by sorry

end cookies_sum_l127_127275


namespace game_ends_in_draw_for_all_n_l127_127970

noncomputable def andrey_representation_count (n : ℕ) : ℕ := 
  -- The function to count Andrey's representation should be defined here
  sorry

noncomputable def petya_representation_count (n : ℕ) : ℕ := 
  -- The function to count Petya's representation should be defined here
  sorry

theorem game_ends_in_draw_for_all_n (n : ℕ) (h : 0 < n) : 
  andrey_representation_count n = petya_representation_count n :=
  sorry

end game_ends_in_draw_for_all_n_l127_127970


namespace least_positive_integer_for_multiple_of_five_l127_127357

theorem least_positive_integer_for_multiple_of_five (x : ℕ) (h_pos : 0 < x) (h_multiple : (625 + x) % 5 = 0) : x = 5 :=
sorry

end least_positive_integer_for_multiple_of_five_l127_127357


namespace quadratic_solution_m_l127_127414

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end quadratic_solution_m_l127_127414


namespace tenth_term_of_sequence_l127_127082

theorem tenth_term_of_sequence : 
  let a_1 := 3
  let d := 6 
  let n := 10 
  (a_1 + (n-1) * d) = 57 := by
  sorry

end tenth_term_of_sequence_l127_127082


namespace power_of_power_evaluation_l127_127455

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l127_127455


namespace min_vertical_distance_between_graphs_l127_127464

noncomputable def min_distance (x : ℝ) : ℝ :=
  |x| - (-x^2 - 4 * x - 2)

theorem min_vertical_distance_between_graphs :
  ∃ x : ℝ, ∀ y : ℝ, min_distance x ≤ min_distance y := 
    sorry

end min_vertical_distance_between_graphs_l127_127464


namespace geometric_difference_l127_127334

def is_geometric_sequence (n : ℕ) : Prop :=
∃ (a b c : ℤ), n = a * 100 + b * 10 + c ∧
a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
(b^2 = a * c) ∧
(b % 2 = 1)

theorem geometric_difference :
  ∃ (n1 n2 : ℕ), is_geometric_sequence n1 ∧ is_geometric_sequence n2 ∧
  n2 > n1 ∧
  n2 - n1 = 220 :=
sorry

end geometric_difference_l127_127334


namespace required_more_visits_l127_127830

-- Define the conditions
def n := 395
def m := 2
def v1 := 135
def v2 := 112
def v3 := 97

-- Define the target statement
theorem required_more_visits : (n * m) - (v1 + v2 + v3) = 446 := by
  sorry

end required_more_visits_l127_127830


namespace factory_produces_correct_number_of_candies_l127_127381

-- Definitions of the given conditions
def candies_per_hour : ℕ := 50
def hours_per_day : ℕ := 10
def days_to_complete_order : ℕ := 8

-- The theorem we want to prove
theorem factory_produces_correct_number_of_candies :
  days_to_complete_order * hours_per_day * candies_per_hour = 4000 :=
by 
  sorry

end factory_produces_correct_number_of_candies_l127_127381


namespace number_of_distinct_digit_odd_numbers_l127_127056

theorem number_of_distinct_digit_odd_numbers (a b c d : ℕ) :
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧
  a * 1000 + b * 100 + c * 10 + d ≤ 9999 ∧
  (a * 1000 + b * 100 + c * 10 + d) % 2 = 1 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0
  → ∃ (n : ℕ), n = 2240 :=
by 
  sorry

end number_of_distinct_digit_odd_numbers_l127_127056


namespace min_neighbor_pairs_l127_127232

theorem min_neighbor_pairs (n : ℕ) (h : n = 2005) :
  ∃ (pairs : ℕ), pairs = 56430 :=
by
  sorry

end min_neighbor_pairs_l127_127232


namespace find_ab_l127_127839

theorem find_ab (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by
  sorry

end find_ab_l127_127839


namespace inequality_proof_l127_127107

section
variable {a b x y : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hab : a + b = 1) :
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
  sorry
end

end inequality_proof_l127_127107


namespace one_eighth_of_2_pow_33_eq_2_pow_x_l127_127434

theorem one_eighth_of_2_pow_33_eq_2_pow_x (x : ℕ) : (1 / 8) * (2 : ℝ) ^ 33 = (2 : ℝ) ^ x → x = 30 := by
  intro h
  sorry

end one_eighth_of_2_pow_33_eq_2_pow_x_l127_127434


namespace smallest_x_satisfies_equation_l127_127831

theorem smallest_x_satisfies_equation : 
  ∀ x : ℚ, 7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45) → x = -7 / 5 :=
by {
  sorry
}

end smallest_x_satisfies_equation_l127_127831


namespace reservoir_capacity_l127_127835

-- Definitions based on the conditions
def storm_deposit : ℚ := 120 * 10^9
def final_full_percentage : ℚ := 0.85
def initial_full_percentage : ℚ := 0.55
variable (C : ℚ) -- total capacity of the reservoir in gallons

-- The statement we want to prove
theorem reservoir_capacity :
  final_full_percentage * C - initial_full_percentage * C = storm_deposit →
  C = 400 * 10^9
:= by
  sorry

end reservoir_capacity_l127_127835


namespace triangle_angle_contradiction_l127_127807

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l127_127807


namespace rightmost_three_digits_of_7_pow_1993_l127_127586

theorem rightmost_three_digits_of_7_pow_1993 :
  7^1993 % 1000 = 407 := 
sorry

end rightmost_three_digits_of_7_pow_1993_l127_127586


namespace find_m_l127_127710

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n / 2

theorem find_m (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 16) : m = 59 ∨ m = 91 :=
by sorry

end find_m_l127_127710


namespace solve_inequality_l127_127877

-- Define the inequality as a function
def inequality_holds (x : ℝ) : Prop :=
  (2 * x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10)

-- Define the solution set as intervals excluding the points
def solution_set (x : ℝ) : Prop :=
  x < -5 / 2 ∨ x > -2

theorem solve_inequality (x : ℝ) : inequality_holds x ↔ solution_set x :=
by sorry

end solve_inequality_l127_127877


namespace perimeter_of_one_rectangle_l127_127616

-- Define the conditions
def is_divided_into_congruent_rectangles (s : ℕ) : Prop :=
  ∃ (height width : ℕ), height = s ∧ width = s / 4

-- Main proof statement
theorem perimeter_of_one_rectangle {s : ℕ} (h₁ : 4 * s = 144)
  (h₂ : is_divided_into_congruent_rectangles s) : 
  ∃ (perimeter : ℕ), perimeter = 90 :=
by 
  sorry

end perimeter_of_one_rectangle_l127_127616


namespace problem1_problem2_l127_127429

-- Problem (1)
theorem problem1 (a : ℝ) (h : a = 1) (p q : ℝ → Prop) 
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0) 
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1) :
  (∀ x, (p x ∧ q x) ↔ (2 < x ∧ x < 3)) :=
by sorry

-- Problem (2)
theorem problem2 (a : ℝ) (p q : ℝ → Prop)
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1)
  (hnpc : ∀ x, ¬p x → ¬q x) 
  (hnpc_not_necessary : ∃ x, ¬p x ∧ q x) :
  (4 / 3 ≤ a ∧ a ≤ 2) :=
by sorry

end problem1_problem2_l127_127429


namespace coffee_shop_ratio_l127_127300

theorem coffee_shop_ratio (morning_usage afternoon_multiplier weekly_usage days_per_week : ℕ) (r : ℕ) 
  (h_morning : morning_usage = 3)
  (h_afternoon : afternoon_multiplier = 3)
  (h_weekly : weekly_usage = 126)
  (h_days : days_per_week = 7):
  weekly_usage = days_per_week * (morning_usage + afternoon_multiplier * morning_usage + r * morning_usage) →
  r = 2 :=
by
  intros h_eq
  sorry

end coffee_shop_ratio_l127_127300


namespace steel_mill_production_2010_l127_127922

noncomputable def steel_mill_production (P : ℕ → ℕ) : Prop :=
  (P 1990 = 400000) ∧ (P 2000 = 500000) ∧ ∀ n, (P n) = (P (n-1)) + (500000 - 400000) / 10

theorem steel_mill_production_2010 (P : ℕ → ℕ) (h : steel_mill_production P) : P 2010 = 630000 :=
by
  sorry -- proof omitted

end steel_mill_production_2010_l127_127922


namespace largest_root_range_l127_127760

theorem largest_root_range (b_0 b_1 b_2 b_3 : ℝ)
  (hb_0 : |b_0| ≤ 3) (hb_1 : |b_1| ≤ 3) (hb_2 : |b_2| ≤ 3) (hb_3 : |b_3| ≤ 3) :
  ∃ s : ℝ, (∃ x : ℝ, x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 = 0 ∧ x > 0 ∧ s = x) ∧ 3 < s ∧ s < 4 := 
sorry

end largest_root_range_l127_127760


namespace greatest_two_digit_number_with_digit_product_16_l127_127128

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digit_product (n m : ℕ) : Prop :=
  n * m = 16

def from_digits (n m : ℕ) : ℕ :=
  10 * n + m

theorem greatest_two_digit_number_with_digit_product_16 :
  ∀ n m, is_two_digit_number (from_digits n m) → digit_product n m → (82 ≥ from_digits n m) :=
by
  intros n m h1 h2
  sorry

end greatest_two_digit_number_with_digit_product_16_l127_127128


namespace hilton_final_marbles_l127_127798

def initial_marbles : ℕ := 26
def marbles_found : ℕ := 6
def marbles_lost : ℕ := 10
def marbles_from_lori := 2 * marbles_lost

def final_marbles := initial_marbles + marbles_found - marbles_lost + marbles_from_lori

theorem hilton_final_marbles : final_marbles = 42 := sorry

end hilton_final_marbles_l127_127798


namespace solve_for_g2_l127_127415

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end solve_for_g2_l127_127415


namespace shaded_fraction_l127_127398

theorem shaded_fraction {S : ℝ} (h : 0 < S) :
  let frac_area := ∑' n : ℕ, (1/(4:ℝ)^1) * (1/(4:ℝ)^n)
  1/3 = frac_area :=
by
  sorry

end shaded_fraction_l127_127398


namespace range_of_expr_l127_127257

noncomputable def expr (x y : ℝ) : ℝ := (x + 2 * y + 3) / (x + 1)

theorem range_of_expr : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ x → 4 * x + 3 * y ≤ 12 → 3 ≤ expr x y ∧ expr x y ≤ 11) :=
by
  sorry

end range_of_expr_l127_127257


namespace parabola_vertex_location_l127_127119

theorem parabola_vertex_location (a b c : ℝ) (h1 : ∀ x < 0, a * x^2 + b * x + c ≤ 0) (h2 : a < 0) : 
  -b / (2 * a) ≥ 0 :=
by
  sorry

end parabola_vertex_location_l127_127119


namespace restore_axes_with_parabola_l127_127757

-- Define the given parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Problem: Prove that you can restore the coordinate axes using the given parabola and tools.
theorem restore_axes_with_parabola : 
  ∃ O X Y : ℝ × ℝ, 
  (∀ x, parabola x = (x, x^2).snd) ∧ 
  (X.fst = 0 ∧ Y.snd = 0) ∧
  (O = (0,0)) :=
sorry

end restore_axes_with_parabola_l127_127757


namespace original_proposition_true_converse_false_l127_127973

-- Lean 4 statement for the equivalent proof problem
theorem original_proposition_true_converse_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬((a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_false_l127_127973


namespace find_M_for_same_asymptotes_l127_127288

theorem find_M_for_same_asymptotes :
  ∃ M : ℝ, ∀ x y : ℝ,
    (x^2 / 16 - y^2 / 25 = 1) →
    (y^2 / 50 - x^2 / M = 1) →
    (∀ x : ℝ, ∃ k : ℝ, y = k * x ↔ k = 5 / 4) →
    M = 32 :=
by
  sorry

end find_M_for_same_asymptotes_l127_127288


namespace round_trip_time_l127_127749

theorem round_trip_time 
  (d1 d2 d3 : ℝ) 
  (s1 s2 s3 t : ℝ) 
  (h1 : d1 = 18) 
  (h2 : d2 = 18) 
  (h3 : d3 = 36) 
  (h4 : s1 = 12) 
  (h5 : s2 = 10) 
  (h6 : s3 = 9) 
  (h7 : t = (d1 / s1) + (d2 / s2) + (d3 / s3)) :
  t = 7.3 :=
by
  sorry

end round_trip_time_l127_127749


namespace quadratic_roots_sum_product_l127_127593

noncomputable def quadratic_sum (a b c : ℝ) : ℝ := -b / a
noncomputable def quadratic_product (a b c : ℝ) : ℝ := c / a

theorem quadratic_roots_sum_product :
  let a := 9
  let b := -45
  let c := 50
  quadratic_sum a b c = 5 ∧ quadratic_product a b c = 50 / 9 :=
by
  sorry

end quadratic_roots_sum_product_l127_127593


namespace find_e_l127_127956

variable (p j t e : ℝ)

def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - e / 100)

theorem find_e (h1 : condition1 p j)
               (h2 : condition2 j t)
               (h3 : condition3 t e p) : e = 6.25 :=
by sorry

end find_e_l127_127956


namespace greatest_divisor_under_100_l127_127409

theorem greatest_divisor_under_100 (d : ℕ) :
  d ∣ 780 ∧ d < 100 ∧ d ∣ 180 ∧ d ∣ 240 ↔ d ≤ 60 := by
  sorry

end greatest_divisor_under_100_l127_127409


namespace length_of_AB_l127_127141

theorem length_of_AB {L : ℝ} (h : 9 * Real.pi * L + 36 * Real.pi = 216 * Real.pi) : L = 20 :=
sorry

end length_of_AB_l127_127141


namespace prob_of_different_colors_l127_127057

def total_balls_A : ℕ := 4 + 5 + 6
def total_balls_B : ℕ := 7 + 6 + 2

noncomputable def prob_same_color : ℚ :=
  (4 / ↑total_balls_A * 7 / ↑total_balls_B) +
  (5 / ↑total_balls_A * 6 / ↑total_balls_B) +
  (6 / ↑total_balls_A * 2 / ↑total_balls_B)

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_of_different_colors :
  prob_different_color = 31 / 45 :=
by
  sorry

end prob_of_different_colors_l127_127057


namespace sum_distances_l127_127872

noncomputable def lengthAB : ℝ := 2
noncomputable def lengthA'B' : ℝ := 5
noncomputable def midpointAB : ℝ := lengthAB / 2
noncomputable def midpointA'B' : ℝ := lengthA'B' / 2
noncomputable def distancePtoD : ℝ := 0.5
noncomputable def proportionality_constant : ℝ := lengthA'B' / lengthAB

theorem sum_distances : distancePtoD + (proportionality_constant * distancePtoD) = 1.75 := by
  sorry

end sum_distances_l127_127872


namespace spinner_win_sector_area_l127_127871

open Real

theorem spinner_win_sector_area (r : ℝ) (P : ℝ)
  (h_r : r = 8) (h_P : P = 3 / 7) : 
  ∃ A : ℝ, A = 192 * π / 7 :=
by
  sorry

end spinner_win_sector_area_l127_127871


namespace arrangement_condition_l127_127061

theorem arrangement_condition (x y z : ℕ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (H1 : x ≤ y + z) 
  (H2 : y ≤ x + z) 
  (H3 : z ≤ x + y) : 
  ∃ (A : ℕ) (B : ℕ) (C : ℕ), 
    A = x ∧ B = y ∧ C = z ∧
    A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1 ∧
    (A ≤ B + C) ∧ (B ≤ A + C) ∧ (C ≤ A + B) :=
by
  sorry

end arrangement_condition_l127_127061


namespace solution_set_of_inequality_cauchy_schwarz_application_l127_127191

theorem solution_set_of_inequality (c : ℝ) (h1 : c > 0) (h2 : ∀ x : ℝ, x + |x - 2 * c| ≥ 2) : 
  c ≥ 1 :=
by
  sorry

theorem cauchy_schwarz_application (m p q r : ℝ) (h1 : m ≥ 1) (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : p + q + r = 3 * m) : 
  p^2 + q^2 + r^2 ≥ 3 :=
by
  sorry

end solution_set_of_inequality_cauchy_schwarz_application_l127_127191


namespace complement_union_l127_127012

def M := { x : ℝ | (x + 3) * (x - 1) < 0 }
def N := { x : ℝ | x ≤ -3 }
def union_set := M ∪ N

theorem complement_union :
  ∀ x : ℝ, x ∈ (⊤ \ union_set) ↔ x ≥ 1 :=
by
  sorry

end complement_union_l127_127012


namespace find_x_value_l127_127326

theorem find_x_value (x : ℚ) (h1 : 9 * x ^ 2 + 8 * x - 1 = 0) (h2 : 27 * x ^ 2 + 65 * x - 8 = 0) : x = 1 / 9 :=
sorry

end find_x_value_l127_127326


namespace find_power_l127_127920

theorem find_power (a b c d e : ℕ) (h1 : a = 105) (h2 : b = 21) (h3 : c = 25) (h4 : d = 45) (h5 : e = 49) 
(h6 : a ^ (3 : ℕ) = b * c * d * e) : 3 = 3 := by
  sorry

end find_power_l127_127920


namespace derivative_of_curve_tangent_line_at_one_l127_127371

-- Definition of the curve
def curve (x : ℝ) : ℝ := x^3 + 5 * x^2 + 3 * x

-- Part 1: Prove the derivative of the curve
theorem derivative_of_curve (x : ℝ) :
  deriv curve x = 3 * x^2 + 10 * x + 3 :=
sorry

-- Part 2: Prove the equation of the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a = 16 ∧ b = -1 ∧ c = -7 ∧
  ∀ (x y : ℝ), curve 1 = 9 → y - 9 = 16 * (x - 1) → a * x + b * y + c = 0 :=
sorry

end derivative_of_curve_tangent_line_at_one_l127_127371


namespace triangle_is_obtuse_l127_127703

theorem triangle_is_obtuse
  (α : ℝ)
  (h1 : α > 0 ∧ α < π)
  (h2 : Real.sin α + Real.cos α = 2 / 3) :
  ∃ β γ, β > 0 ∧ β < π ∧ γ > 0 ∧ γ < π ∧ β + γ + α = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2) :=
sorry

end triangle_is_obtuse_l127_127703


namespace homework_problems_l127_127253

theorem homework_problems (p t : ℕ) (h1 : p >= 10) (h2 : pt = (2 * p + 2) * (t + 1)) : p * t = 60 :=
by
  sorry

end homework_problems_l127_127253


namespace perimeter_of_rectangle_l127_127701

theorem perimeter_of_rectangle (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  ∃ perimeter length, length = area / width ∧ perimeter = 2 * (length + width) ∧ perimeter = 110 := by
  sorry

end perimeter_of_rectangle_l127_127701


namespace problem_solution_l127_127796

theorem problem_solution (A B : ℝ) (h : ∀ x, x ≠ 3 → (A / (x - 3)) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) : 
  A + B = 46 :=
sorry

end problem_solution_l127_127796


namespace pardee_road_length_l127_127156

theorem pardee_road_length (t p : ℕ) (h1 : t = 162 * 1000) (h2 : t = p + 150 * 1000) : p = 12 * 1000 :=
by
  -- Proof goes here
  sorry

end pardee_road_length_l127_127156


namespace length_of_generatrix_l127_127094

/-- Given that the base radius of a cone is sqrt(2), and its lateral surface is unfolded into a semicircle,
prove that the length of the generatrix of the cone is 2 sqrt(2). -/
theorem length_of_generatrix (r l : ℝ) (h1 : r = Real.sqrt 2)
    (h2 : 2 * Real.pi * r = Real.pi * l) : l = 2 * Real.sqrt 2 :=
by
  sorry

end length_of_generatrix_l127_127094


namespace regular_tetrahedron_fourth_vertex_l127_127824

theorem regular_tetrahedron_fourth_vertex :
  ∃ (x y z : ℤ), 
    ((x, y, z) = (0, 0, 6) ∨ (x, y, z) = (0, 0, -6)) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 6) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 5) ^ 2 + (y - 0) ^ 2 + (z - 6) ^ 2 = 36) := 
by
  sorry

end regular_tetrahedron_fourth_vertex_l127_127824


namespace area_difference_triangles_l127_127133

theorem area_difference_triangles
  (A B C F D : Type)
  (angle_FAB_right : true) 
  (angle_ABC_right : true) 
  (AB : Real) (hAB : AB = 5)
  (BC : Real) (hBC : BC = 3)
  (AF : Real) (hAF : AF = 7)
  (area_triangle : A -> B -> C -> Real)
  (angle_bet : A -> D -> F) 
  (angle_bet : B -> D -> C)
  (area_ADF : Real)
  (area_BDC : Real) : (area_ADF - area_BDC = 10) :=
sorry

end area_difference_triangles_l127_127133


namespace probability_edge_within_five_hops_l127_127630

def is_edge_square (n : ℕ) (coord : ℕ × ℕ) : Prop := 
  coord.1 = 1 ∨ coord.1 = n ∨ coord.2 = 1 ∨ coord.2 = n

def is_central_square (coord : ℕ × ℕ) : Prop :=
  (coord = (2, 2)) ∨ (coord = (2, 3)) ∨ (coord = (3, 2)) ∨ (coord = (3, 3))

noncomputable def probability_of_edge_in_n_hops (n : ℕ) : ℚ := sorry

theorem probability_edge_within_five_hops : probability_of_edge_in_n_hops 4 = 7 / 8 :=
sorry

end probability_edge_within_five_hops_l127_127630


namespace slices_with_both_l127_127351

theorem slices_with_both (n total_slices pepperoni_slices mushroom_slices other_slices : ℕ)
  (h1 : total_slices = 24) 
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 14)
  (h4 : (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices) :
  n = 5 :=
sorry

end slices_with_both_l127_127351


namespace alice_gadgets_sales_l127_127573

variable (S : ℝ) -- Variable to denote the worth of gadgets Alice sold
variable (E : ℝ) -- Variable to denote Alice's total earnings

theorem alice_gadgets_sales :
  let basic_salary := 240
  let commission_percentage := 0.02
  let save_amount := 29
  let save_percentage := 0.10
  
  -- Total earnings equation
  let earnings_eq := E = basic_salary + commission_percentage * S
  
  -- Savings equation
  let savings_eq := save_percentage * E = save_amount
  
  -- Solve the system of equations to show S = 2500
  S = 2500 :=
by
  sorry

end alice_gadgets_sales_l127_127573


namespace tournament_min_cost_l127_127196

variables (k : ℕ) (m : ℕ) (S E : ℕ → ℕ)

noncomputable def min_cost (k : ℕ) : ℕ :=
  k * (4 * k^2 + k - 1) / 2

theorem tournament_min_cost (k_pos : 0 < k) (players : m = 2 * k)
  (each_plays_once 
      : ∀ i j, i ≠ j → ∃ d, S d = i ∧ E d = j) -- every two players play once, matches have days
  (one_match_per_day : ∀ d, ∃! i j, i ≠ j ∧ S d = i ∧ E d = j) -- exactly one match per day
  : min_cost k = k * (4 * k^2 + k - 1) / 2 := 
sorry

end tournament_min_cost_l127_127196


namespace cows_dogs_ratio_l127_127936

theorem cows_dogs_ratio (C D : ℕ) (hC : C = 184) (hC_remain : 3 / 4 * C = 138)
  (hD_remain : 1 / 4 * D + 138 = 161) : C / D = 2 :=
sorry

end cows_dogs_ratio_l127_127936


namespace arithmetic_sequence_value_l127_127809

theorem arithmetic_sequence_value (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_cond : a 3 + a 9 = 15 - a 6) : a 6 = 5 :=
sorry

end arithmetic_sequence_value_l127_127809


namespace arithmetic_sequence_sum_l127_127517

-- Define the variables and conditions
def a : ℕ := 71
def d : ℕ := 2
def l : ℕ := 99

-- Calculate the number of terms in the sequence
def n : ℕ := ((l - a) / d) + 1

-- Define the sum of the arithmetic sequence
def S : ℕ := (n * (a + l)) / 2

-- Statement to be proven
theorem arithmetic_sequence_sum :
  3 * S = 3825 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_sum_l127_127517


namespace candle_height_half_after_9_hours_l127_127812

-- Define the initial heights and burn rates
def initial_height_first : ℝ := 12
def burn_rate_first : ℝ := 2
def initial_height_second : ℝ := 15
def burn_rate_second : ℝ := 3

-- Define the height functions after t hours
def height_first (t : ℝ) : ℝ := initial_height_first - burn_rate_first * t
def height_second (t : ℝ) : ℝ := initial_height_second - burn_rate_second * t

-- Prove that at t = 9, the height of the first candle is half the height of the second candle
theorem candle_height_half_after_9_hours : height_first 9 = 0.5 * height_second 9 := by
  sorry

end candle_height_half_after_9_hours_l127_127812


namespace min_distance_l127_127639

theorem min_distance (W : ℝ) (b : ℝ) (n : ℕ) (H_W : W = 42) (H_b : b = 3) (H_n : n = 8) : 
  ∃ d : ℝ, d = 2 ∧ (W - n * b = 9 * d) := 
by 
  -- Here should go the proof
  sorry

end min_distance_l127_127639


namespace train_cross_pole_time_l127_127399

def speed_kmh := 90 -- speed of the train in km/hr
def length_m := 375 -- length of the train in meters

/-- Convert speed from km/hr to m/s -/
def convert_speed (v_kmh : ℕ) : ℕ := v_kmh * 1000 / 3600

/-- Calculate the time it takes for the train to cross the pole -/
def time_to_cross_pole (length_m : ℕ) (speed_m_s : ℕ) : ℕ := length_m / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole length_m (convert_speed speed_kmh) = 15 :=
by
  sorry

end train_cross_pole_time_l127_127399


namespace man_savings_l127_127394

theorem man_savings (I : ℝ) (S : ℝ) (h1 : S = 0.35) (h2 : 2 * (0.65 * I) = 0.65 * I + 0.70 * I) :
  S = 0.35 :=
by
  -- Introduce necessary assumptions
  let savings_first_year := S * I
  let expenditure_first_year := I - savings_first_year
  let savings_second_year := 2 * savings_first_year

  have h3 : expenditure_first_year = 0.65 * I := by sorry
  have h4 : savings_first_year = 0.35 * I := by sorry

  -- Using given condition to resolve S
  exact h1

end man_savings_l127_127394


namespace conference_problem_l127_127008

noncomputable def exists_round_table (n : ℕ) (scientists : Finset ℕ) (acquaintance : ℕ → Finset ℕ) : Prop :=
  ∃ (A B C D : ℕ), A ∈ scientists ∧ B ∈ scientists ∧ C ∈ scientists ∧ D ∈ scientists ∧
  ((A ≠ B ∧ A ≠ C ∧ A ≠ D) ∧ (B ≠ C ∧ B ≠ D) ∧ (C ≠ D)) ∧
  (B ∈ acquaintance A ∧ C ∈ acquaintance B ∧ D ∈ acquaintance C ∧ A ∈ acquaintance D)

theorem conference_problem :
  ∀ (scientists : Finset ℕ),
  ∀ (acquaintance : ℕ → Finset ℕ),
    (scientists.card = 50) →
    (∀ s ∈ scientists, (acquaintance s).card ≥ 25) →
    exists_round_table 50 scientists acquaintance :=
sorry

end conference_problem_l127_127008


namespace function_three_distinct_zeros_l127_127552

theorem function_three_distinct_zeros (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - 3 * a * x + a) ∧ (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  a > 1/4 :=
by
  sorry

end function_three_distinct_zeros_l127_127552


namespace area_of_rhombus_l127_127439

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 10) : 
  1 / 2 * d1 * d2 = 30 :=
by 
  rw [h1, h2]
  norm_num

end area_of_rhombus_l127_127439


namespace total_capsules_sold_in_2_weeks_l127_127612

-- Define the conditions as constants
def Earnings100mgPerWeek := 80
def CostPer100mgCapsule := 5
def Earnings500mgPerWeek := 60
def CostPer500mgCapsule := 2

-- Theorem to prove the total number of capsules sold in 2 weeks
theorem total_capsules_sold_in_2_weeks : 
  (Earnings100mgPerWeek / CostPer100mgCapsule) * 2 + (Earnings500mgPerWeek / CostPer500mgCapsule) * 2 = 92 :=
by
  sorry

end total_capsules_sold_in_2_weeks_l127_127612


namespace lightbulb_stops_on_friday_l127_127038

theorem lightbulb_stops_on_friday
  (total_hours : ℕ) (daily_usage : ℕ) (start_day : ℕ) (stops_day : ℕ)
  (h_total_hours : total_hours = 24999)
  (h_daily_usage : daily_usage = 2)
  (h_start_day : start_day = 1) : 
  stops_day = 5 := by
  sorry

end lightbulb_stops_on_friday_l127_127038


namespace Travis_spends_on_cereal_l127_127186

theorem Travis_spends_on_cereal (boxes_per_week : ℕ) (cost_per_box : ℝ) (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3.00) 
  (h3 : weeks_per_year = 52) 
: boxes_per_week * weeks_per_year * cost_per_box = 312.00 := 
by
  sorry

end Travis_spends_on_cereal_l127_127186


namespace value_of_other_bills_l127_127091

theorem value_of_other_bills (x : ℕ) : 
  (∃ (num_twenty num_x : ℕ), num_twenty = 3 ∧
                           num_x = 2 * num_twenty ∧
                           20 * num_twenty + x * num_x = 120) → 
  x * 6 = 60 :=
by
  intro h
  obtain ⟨num_twenty, num_x, h1, h2, h3⟩ := h
  have : num_twenty = 3 := h1
  have : num_x = 2 * num_twenty := h2
  have : x * 6 = 60 := sorry
  exact this

end value_of_other_bills_l127_127091


namespace sum_of_squares_of_four_consecutive_even_numbers_l127_127024

open Int

theorem sum_of_squares_of_four_consecutive_even_numbers (x y z w : ℤ) 
    (hx : x % 2 = 0) (hy : y = x + 2) (hz : z = x + 4) (hw : w = x + 6)
    : x + y + z + w = 36 → x^2 + y^2 + z^2 + w^2 = 344 := by
  sorry

end sum_of_squares_of_four_consecutive_even_numbers_l127_127024


namespace trigonometric_identity_proof_l127_127802

noncomputable def a : ℝ := -35 / 6 * Real.pi

theorem trigonometric_identity_proof :
  (2 * Real.sin (Real.pi + a) * Real.cos (Real.pi - a) - Real.cos (Real.pi + a)) / 
  (1 + Real.sin a ^ 2 + Real.sin (Real.pi - a) - Real.cos (Real.pi + a) ^ 2) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_proof_l127_127802


namespace two_integers_difference_l127_127556

theorem two_integers_difference
  (x y : ℕ)
  (h_sum : x + y = 5)
  (h_cube_diff : x^3 - y^3 = 63)
  (h_gt : x > y) :
  x - y = 3 := 
sorry

end two_integers_difference_l127_127556


namespace dave_more_than_jerry_games_l127_127901

variable (K D J : ℕ)  -- Declaring the variables for Ken, Dave, and Jerry respectively

-- Defining the conditions
def ken_more_games := K = D + 5
def dave_more_than_jerry := D > 7
def jerry_games := J = 7
def total_games := K + D + 7 = 32

-- Defining the proof problem
theorem dave_more_than_jerry_games (hK : ken_more_games K D) (hD : dave_more_than_jerry D) (hJ : jerry_games J) (hT : total_games K D) : D - 7 = 3 :=
by
  sorry

end dave_more_than_jerry_games_l127_127901


namespace differentiable_inequality_l127_127202

theorem differentiable_inequality 
  {a b : ℝ} 
  {f g : ℝ → ℝ} 
  (hdiff_f : DifferentiableOn ℝ f (Set.Icc a b))
  (hdiff_g : DifferentiableOn ℝ g (Set.Icc a b))
  (hderiv_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x > deriv g x)) :
  ∀ x ∈ Set.Ioo a b, f x + g a > g x + f a :=
by 
  sorry

end differentiable_inequality_l127_127202


namespace sum_of_three_pairwise_relatively_prime_integers_l127_127653

theorem sum_of_three_pairwise_relatively_prime_integers
  (a b c : ℕ)
  (h1 : a > 1)
  (h2 : b > 1)
  (h3 : c > 1)
  (h4 : a * b * c = 13824)
  (h5 : Nat.gcd a b = 1)
  (h6 : Nat.gcd b c = 1)
  (h7 : Nat.gcd a c = 1) :
  a + b + c = 144 :=
by
  sorry

end sum_of_three_pairwise_relatively_prime_integers_l127_127653


namespace smallest_palindromic_odd_integer_in_base2_and_4_l127_127116

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := n.digits base
  digits = digits.reverse

theorem smallest_palindromic_odd_integer_in_base2_and_4 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ Odd n ∧ ∀ m : ℕ, (m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 ∧ Odd m) → n <= m :=
  sorry

end smallest_palindromic_odd_integer_in_base2_and_4_l127_127116


namespace mass_percentage_of_Ba_l127_127843

theorem mass_percentage_of_Ba {BaX : Type} {molar_mass_Ba : ℝ} {compound_mass : ℝ} {mass_Ba : ℝ}:
  molar_mass_Ba = 137.33 ∧ 
  compound_mass = 100 ∧
  mass_Ba = 66.18 →
  (mass_Ba / compound_mass * 100) = 66.18 :=
by
  sorry

end mass_percentage_of_Ba_l127_127843


namespace quadratic_one_real_root_positive_n_l127_127961

theorem quadratic_one_real_root_positive_n (n : ℝ) (h : (n ≠ 0)) :
  (∃ x : ℝ, (x^2 - 6*n*x - 9*n) = 0) ∧
  (∀ x y : ℝ, (x^2 - 6*n*x - 9*n) = 0 → (y^2 - 6*n*y - 9*n) = 0 → x = y) ↔
  n = 0 := by
  sorry

end quadratic_one_real_root_positive_n_l127_127961


namespace sum_of_consecutive_even_negative_integers_l127_127389

theorem sum_of_consecutive_even_negative_integers (n m : ℤ) 
  (h1 : n % 2 = 0)
  (h2 : m % 2 = 0)
  (h3 : n < 0)
  (h4 : m < 0)
  (h5 : m = n + 2)
  (h6 : n * m = 2496) : n + m = -102 := 
sorry

end sum_of_consecutive_even_negative_integers_l127_127389


namespace pizza_slices_correct_l127_127536

-- Definitions based on conditions
def john_slices : Nat := 3
def sam_slices : Nat := 2 * john_slices
def eaten_slices : Nat := john_slices + sam_slices
def remaining_slices : Nat := 3
def total_slices : Nat := eaten_slices + remaining_slices

-- The statement to be proven.
theorem pizza_slices_correct : total_slices = 12 := by
  sorry

end pizza_slices_correct_l127_127536


namespace max_area_of_rectangular_pen_l127_127002

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l127_127002


namespace Q_gets_less_than_P_l127_127070

theorem Q_gets_less_than_P (x : Real) (hx : x > 0) (hP : P = 1.25 * x): 
  Q = P * 0.8 := 
sorry

end Q_gets_less_than_P_l127_127070


namespace simplify_expression_l127_127048

variable (c d : ℝ)
variable (hc : 0 < c)
variable (hd : 0 < d)
variable (h : c^3 + d^3 = 3 * (c + d))

theorem simplify_expression : (c / d) + (d / c) - (3 / (c * d)) = 1 := by
  sorry

end simplify_expression_l127_127048


namespace min_cells_marked_l127_127885

/-- The minimum number of cells that need to be marked in a 50x50 grid so
each 1x6 vertical or horizontal strip has at least one marked cell is 416. -/
theorem min_cells_marked {n : ℕ} : n = 416 → 
  (∀ grid : Fin 50 × Fin 50, ∃ cells : Finset (Fin 50 × Fin 50), 
    (∀ (r c : Fin 50), (r = 6 * i + k ∨ c = 6 * i + k) →
      (∃ (cell : Fin 50 × Fin 50), cell ∈ cells)) →
    cells.card = n) := 
sorry

end min_cells_marked_l127_127885


namespace boat_travels_125_km_downstream_l127_127068

/-- The speed of the boat in still water is 20 km/hr -/
def boat_speed_still_water : ℝ := 20

/-- The speed of the stream is 5 km/hr -/
def stream_speed : ℝ := 5

/-- The total time taken downstream is 5 hours -/
def total_time_downstream : ℝ := 5

/-- The effective speed of the boat downstream -/
def effective_speed_downstream : ℝ := boat_speed_still_water + stream_speed

/-- The distance the boat travels downstream -/
def distance_downstream : ℝ := effective_speed_downstream * total_time_downstream

/-- The boat travels 125 km downstream -/
theorem boat_travels_125_km_downstream :
  distance_downstream = 125 := 
sorry

end boat_travels_125_km_downstream_l127_127068


namespace inequality_solution_set_l127_127277

open Set

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5 * a) * (x + a) > 0} = {x | x < 5 * a ∨ x > -a} :=
sorry

end inequality_solution_set_l127_127277


namespace inequality_solution_l127_127482

theorem inequality_solution (x : ℝ) : 
  x^2 - 9 * x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := 
by
  sorry

end inequality_solution_l127_127482


namespace exists_member_T_divisible_by_3_l127_127023

-- Define the set T of all numbers which are the sum of the squares of four consecutive integers
def T := { x : ℤ | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 }

-- Theorem to prove that there exists a member in T which is divisible by 3
theorem exists_member_T_divisible_by_3 : ∃ x ∈ T, x % 3 = 0 :=
by
  sorry

end exists_member_T_divisible_by_3_l127_127023


namespace solve_diophantine_l127_127870

theorem solve_diophantine : ∀ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ (x^3 - y^3 = x * y + 61) → (x, y) = (6, 5) :=
by
  intros x y h
  sorry

end solve_diophantine_l127_127870


namespace det_A_is_half_l127_127974

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
![![Real.cos (20 * Real.pi / 180), Real.sin (40 * Real.pi / 180)], ![Real.sin (20 * Real.pi / 180), Real.cos (40 * Real.pi / 180)]]

theorem det_A_is_half : A.det = 1 / 2 := by
  sorry

end det_A_is_half_l127_127974


namespace tiling_impossible_2003x2003_l127_127272

theorem tiling_impossible_2003x2003 :
  ¬ (∃ (f : Fin 2003 × Fin 2003 → ℕ),
  (∀ p : Fin 2003 × Fin 2003, f p = 1 ∨ f p = 2) ∧
  (∀ p : Fin 2003, (f (p, 0) + f (p, 1)) % 3 = 0) ∧
  (∀ p : Fin 2003, (f (0, p) + f (1, p) + f (2, p)) % 3 = 0)) := 
sorry

end tiling_impossible_2003x2003_l127_127272


namespace solution_proof_l127_127520

variable (x y z : ℝ)

-- Given system of equations
def equation1 := 6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12
def equation2 := 9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3
def equation3 := 2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

theorem solution_proof : 
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end solution_proof_l127_127520


namespace least_number_to_subtract_l127_127766

theorem least_number_to_subtract (x : ℕ) (h : 509 - x = 45 * n) : ∃ x, (509 - x) % 9 = 0 ∧ (509 - x) % 15 = 0 ∧ x = 14 := by
  sorry

end least_number_to_subtract_l127_127766


namespace rectangular_reconfiguration_l127_127706

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end rectangular_reconfiguration_l127_127706


namespace specialCollectionAtEndOfMonth_l127_127436

noncomputable def specialCollectionBooksEndOfMonth (initialBooks loanedBooks returnedPercentage : ℕ) :=
  initialBooks - (loanedBooks - loanedBooks * returnedPercentage / 100)

theorem specialCollectionAtEndOfMonth :
  specialCollectionBooksEndOfMonth 150 80 65 = 122 :=
by
  sorry

end specialCollectionAtEndOfMonth_l127_127436


namespace can_lids_per_box_l127_127800

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l127_127800


namespace max_marks_for_test_l127_127679

theorem max_marks_for_test (M : ℝ) (h1: (0.30 * M) = 180) : M = 600 :=
by
  sorry

end max_marks_for_test_l127_127679


namespace granddaughter_age_is_12_l127_127588

/-
Conditions:
- Betty is 60 years old.
- Her daughter is 40 percent younger than Betty.
- Her granddaughter is one-third her mother's age.

Question:
- Prove that the granddaughter is 12 years old.
-/

def age_of_Betty := 60

def age_of_daughter (age_of_Betty : ℕ) : ℕ :=
  age_of_Betty - age_of_Betty * 40 / 100

def age_of_granddaughter (age_of_daughter : ℕ) : ℕ :=
  age_of_daughter / 3

theorem granddaughter_age_is_12 (h1 : age_of_Betty = 60) : age_of_granddaughter (age_of_daughter age_of_Betty) = 12 := by
  sorry

end granddaughter_age_is_12_l127_127588


namespace cos_sum_to_product_l127_127718

theorem cos_sum_to_product (x : ℝ) : 
  (∃ a b c d : ℕ, a * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x) =
  Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (10 * x) + Real.cos (14 * x) 
  ∧ a + b + c + d = 18) :=
sorry

end cos_sum_to_product_l127_127718


namespace river_bend_students_more_than_pets_l127_127529

theorem river_bend_students_more_than_pets 
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (hamsters_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ := students_per_classroom * number_of_classrooms)
  (total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms)
  (total_hamsters : ℕ := hamsters_per_classroom * number_of_classrooms)
  (total_pets : ℕ := total_rabbits + total_hamsters) :
  students_per_classroom = 24 ∧ rabbits_per_classroom = 2 ∧ hamsters_per_classroom = 3 ∧ number_of_classrooms = 5 →
  total_students - total_pets = 95 :=
by
  sorry

end river_bend_students_more_than_pets_l127_127529


namespace two_numbers_equal_l127_127465

variables {a b c : ℝ}
variable (h1 : a + b^2 + c^2 = a^2 + b + c^2)
variable (h2 : a^2 + b + c^2 = a^2 + b^2 + c)

theorem two_numbers_equal (h1 : a + b^2 + c^2 = a^2 + b + c^2) (h2 : a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ a = c ∨ b = c :=
by
  sorry

end two_numbers_equal_l127_127465


namespace evaluate_complex_modulus_l127_127306

namespace ComplexProblem

open Complex

theorem evaluate_complex_modulus : 
  abs ((1 / 2 : ℂ) - (3 / 8) * Complex.I) = 5 / 8 :=
by
  sorry

end ComplexProblem

end evaluate_complex_modulus_l127_127306


namespace necessary_and_sufficient_condition_l127_127952

theorem necessary_and_sufficient_condition (x : ℝ) :
  x > 0 ↔ x + 1/x ≥ 2 :=
by sorry

end necessary_and_sufficient_condition_l127_127952


namespace ratio_of_tagged_fish_is_1_over_25_l127_127787

-- Define the conditions
def T70 : ℕ := 70  -- Number of tagged fish first caught and tagged
def T50 : ℕ := 50  -- Total number of fish caught in the second sample
def t2 : ℕ := 2    -- Number of tagged fish in the second sample

-- State the theorem/question
theorem ratio_of_tagged_fish_is_1_over_25 : (t2 / T50) = 1 / 25 :=
by
  sorry

end ratio_of_tagged_fish_is_1_over_25_l127_127787


namespace toll_constant_l127_127972

theorem toll_constant (t : ℝ) (x : ℝ) (constant : ℝ) : 
  (t = 1.50 + 0.50 * (x - constant)) → 
  (x = 18 / 2) → 
  (t = 5) → 
  constant = 2 :=
by
  intros h1 h2 h3
  sorry

end toll_constant_l127_127972


namespace solveEquation1_proof_solveEquation2_proof_l127_127911

noncomputable def solveEquation1 : Set ℝ :=
  { x | 2 * x^2 - 5 * x = 0 }

theorem solveEquation1_proof :
  solveEquation1 = { 0, (5 / 2 : ℝ) } :=
by
  sorry

noncomputable def solveEquation2 : Set ℝ :=
  { x | x^2 + 3 * x - 3 = 0 }

theorem solveEquation2_proof :
  solveEquation2 = { ( (-3 + Real.sqrt 21) / 2 : ℝ ), ( (-3 - Real.sqrt 21) / 2 : ℝ ) } :=
by
  sorry

end solveEquation1_proof_solveEquation2_proof_l127_127911


namespace wages_problem_l127_127028

variable {S W_y W_x : ℝ}
variable {D_x : ℝ}

theorem wages_problem
  (h1 : S = 45 * W_y)
  (h2 : S = 20 * (W_x + W_y))
  (h3 : S = D_x * W_x) :
  D_x = 36 :=
sorry

end wages_problem_l127_127028


namespace find_y_arithmetic_mean_l127_127674

theorem find_y_arithmetic_mean (y : ℝ) 
  (h : (8 + 15 + 20 + 7 + y + 9) / 6 = 12) : 
  y = 13 :=
sorry

end find_y_arithmetic_mean_l127_127674


namespace S13_equals_26_l127_127566

open Nat

variable (a : Nat → ℕ)

-- Define the arithmetic sequence property
def arithmetic_sequence (d a₁ : Nat → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a₁ + n * d

-- Define the summation property
def sum_of_first_n_terms (S : Nat → ℕ) (a₁ : ℕ) (d : ℕ) : Prop :=
   ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2

-- The given condition
def condition (a₁ d : ℕ) : Prop :=
  2 * (a₁ + 4 * d) + 3 * (a₁ + 6 * d) + 2 * (a₁ + 8 * d) = 14

-- The Lean statement for the proof problem
theorem S13_equals_26 (a₁ d : ℕ) (S : Nat → ℕ) 
  (h_seq : arithmetic_sequence a d a₁) 
  (h_sum : sum_of_first_n_terms S a₁ d)
  (h_cond : condition a₁ d) : 
  S 13 = 26 := 
sorry

end S13_equals_26_l127_127566


namespace inequality_proof_l127_127606

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l127_127606


namespace intersection_of_sets_l127_127918

def setP : Set ℝ := { x | x ≤ 3 }
def setQ : Set ℝ := { x | x > 1 }

theorem intersection_of_sets : setP ∩ setQ = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l127_127918


namespace solve_system_l127_127363

theorem solve_system (x y : ℝ) :
  (x + 3*y + 3*x*y = -1) ∧ (x^2*y + 3*x*y^2 = -4) →
  (x = -3 ∧ y = -1/3) ∨ (x = -1 ∧ y = -1) ∨ (x = -1 ∧ y = 4/3) ∨ (x = 4 ∧ y = -1/3) :=
by
  sorry

end solve_system_l127_127363


namespace no_common_points_implies_parallel_l127_127458

variable (a : Type) (P : Type) [LinearOrder P] [AddGroupWithOne P]
variable (has_no_common_point : a → P → Prop)
variable (is_parallel : a → P → Prop)

theorem no_common_points_implies_parallel (a_line : a) (a_plane : P) :
  has_no_common_point a_line a_plane ↔ is_parallel a_line a_plane :=
sorry

end no_common_points_implies_parallel_l127_127458


namespace series_sum_l127_127726

noncomputable def sum_series : Real :=
  ∑' n: ℕ, (4 * (n + 1) + 2) / (3 : ℝ)^(n + 1)

theorem series_sum : sum_series = 3 := by
  sorry

end series_sum_l127_127726


namespace increasing_function_range_l127_127071

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 - 2 * x + Real.log x

theorem increasing_function_range (m : ℝ) : (∀ x > 0, m * x + (1 / x) - 2 ≥ 0) ↔ m ≥ 1 := 
by 
  sorry

end increasing_function_range_l127_127071


namespace find_exercise_books_l127_127515

theorem find_exercise_books
  (pencil_ratio pen_ratio exercise_book_ratio eraser_ratio : ℕ)
  (total_pencils total_ratio_units : ℕ)
  (h1 : pencil_ratio = 10)
  (h2 : pen_ratio = 2)
  (h3 : exercise_book_ratio = 3)
  (h4 : eraser_ratio = 4)
  (h5 : total_pencils = 150)
  (h6 : total_ratio_units = pencil_ratio + pen_ratio + exercise_book_ratio + eraser_ratio) :
  (total_pencils / pencil_ratio) * exercise_book_ratio = 45 :=
by
  sorry

end find_exercise_books_l127_127515


namespace div_gt_sum_div_sq_l127_127015

theorem div_gt_sum_div_sq (n d d' : ℕ) (h₁ : d' > d) (h₂ : d ∣ n) (h₃ : d' ∣ n) : 
  d' > d + d * d / n :=
by 
  sorry

end div_gt_sum_div_sq_l127_127015


namespace rectangle_perimeter_l127_127683

theorem rectangle_perimeter
  (L W : ℕ)
  (h1 : L * W = 360)
  (h2 : (L + 10) * (W - 6) = 360) :
  2 * L + 2 * W = 76 := 
sorry

end rectangle_perimeter_l127_127683


namespace distinct_negative_real_roots_l127_127491

def poly (p : ℝ) (x : ℝ) : ℝ := x^4 + 2*p*x^3 + x^2 + 2*p*x + 1

theorem distinct_negative_real_roots (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly p x1 = 0 ∧ poly p x2 = 0) ↔ p > 3/4 :=
sorry

end distinct_negative_real_roots_l127_127491


namespace simplify_fraction_l127_127894

theorem simplify_fraction (x : ℝ) (hx : x ≠ 1) : (x^2 / (x-1)) - (1 / (x-1)) = x + 1 :=
by 
  sorry

end simplify_fraction_l127_127894


namespace Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l127_127158

-- Definition of the problem conditions
def initially_painted (m n : ℕ) : Prop :=
  ∃ i j : ℕ, i < m ∧ j < n
  
def odd_painted_neighbors (m n : ℕ) : Prop :=
  ∀ i j : ℕ, i < m ∧ j < n →
  (∃ k l : ℕ, (k = i+1 ∨ k = i-1 ∨ l = j+1 ∨ l = j-1) ∧ k < m ∧ l < n → true)

-- Part (a): 8x9 rectangle
theorem Sasha_can_paint_8x9 : (initially_painted 8 9 ∧ odd_painted_neighbors 8 9) → ∀ (i j : ℕ), i < 8 ∧ j < 9 :=
by
  -- Proof here
  sorry

-- Part (b): 8x10 rectangle
theorem Sasha_cannot_paint_8x10 : (initially_painted 8 10 ∧ odd_painted_neighbors 8 10) → ¬ (∀ (i j : ℕ), i < 8 ∧ j < 10) :=
by
  -- Proof here
  sorry

end Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l127_127158


namespace other_root_of_quadratic_l127_127775

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, (x^2 + 2*x - a) = 0 → x = -3) → (∃ z, z = 1 ∧ (z^2 + 2*z - a) = 0) :=
by
  sorry

end other_root_of_quadratic_l127_127775


namespace thirty_divides_p_squared_minus_one_iff_p_eq_five_l127_127216

theorem thirty_divides_p_squared_minus_one_iff_p_eq_five (p : ℕ) (hp : Nat.Prime p) (h_ge : p ≥ 5) : 30 ∣ (p^2 - 1) ↔ p = 5 :=
by
  sorry

end thirty_divides_p_squared_minus_one_iff_p_eq_five_l127_127216


namespace determine_a_l127_127420

theorem determine_a (a : ℝ) (h : ∃ r : ℝ, (a / (1+1*I : ℂ) + (1+1*I : ℂ) / 2).im = 0) : a = 1 :=
sorry

end determine_a_l127_127420


namespace smallest_positive_integer_l127_127425

theorem smallest_positive_integer (n : ℕ) (h1 : 0 < n) (h2 : ∃ k1 : ℕ, 3 * n = k1^2) (h3 : ∃ k2 : ℕ, 4 * n = k2^3) : 
  n = 54 := 
sorry

end smallest_positive_integer_l127_127425


namespace find_fraction_l127_127122

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end find_fraction_l127_127122


namespace area_triangle_AMC_l127_127269

open Real

-- Definitions: Define the points A, B, C, D such that they form a rectangle
-- Define midpoint M of \overline{AD}

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def A : Point := {x := 0, y := 0}
noncomputable def B : Point := {x := 6, y := 0}
noncomputable def D : Point := {x := 0, y := 8}
noncomputable def C : Point := {x := 6, y := 8}
noncomputable def M : Point := {x := 0, y := 4} -- midpoint of AD

-- Function to compute the area of triangle AMC
noncomputable def triangle_area (A M C : Point) : ℝ :=
  (1 / 2 : ℝ) * abs ((A.x - C.x) * (M.y - A.y) - (A.x - M.x) * (C.y - A.y))

-- The theorem to prove
theorem area_triangle_AMC : triangle_area A M C = 12 :=
by
  sorry

end area_triangle_AMC_l127_127269


namespace minimize_payment_l127_127768

theorem minimize_payment :
  ∀ (bd_A td_A bd_B td_B bd_C td_C : ℕ),
    bd_A = 42 → td_A = 36 →
    bd_B = 48 → td_B = 41 →
    bd_C = 54 → td_C = 47 →
    ∃ (S : ℕ), S = 36 ∧ 
      (S = bd_A - (bd_A - td_A)) ∧
      (S < bd_B - (bd_B - td_B)) ∧
      (S < bd_C - (bd_C - td_C)) := 
by {
  sorry
}

end minimize_payment_l127_127768


namespace bell_rings_count_l127_127528

def classes : List String := ["Maths", "English", "History", "Geography", "Chemistry", "Physics", "Literature", "Music"]

def total_classes : Nat := classes.length

def rings_per_class : Nat := 2

def classes_before_music : Nat := total_classes - 1

def rings_before_music : Nat := classes_before_music * rings_per_class

def current_class_rings : Nat := 1

def total_rings_by_now : Nat := rings_before_music + current_class_rings

theorem bell_rings_count :
  total_rings_by_now = 15 := by
  sorry

end bell_rings_count_l127_127528


namespace point_N_coordinates_l127_127879

/--
Given:
- point M with coordinates (5, -6)
- vector a = (1, -2)
- the vector NM equals 3 times vector a
Prove:
- the coordinates of point N are (2, 0)
-/

theorem point_N_coordinates (x y : ℝ) :
  let M := (5, -6)
  let a := (1, -2)
  let NM := (5 - x, -6 - y)
  3 * a = NM → 
  (x = 2 ∧ y = 0) :=
by 
  intros
  sorry

end point_N_coordinates_l127_127879


namespace hexahedron_octahedron_ratio_l127_127611

open Real

theorem hexahedron_octahedron_ratio (a : ℝ) (h_a_pos : 0 < a) :
  let r1 := (sqrt 6 * a / 9)
  let r2 := (sqrt 6 * a / 6)
  let ratio := r1 / r2
  ∃ m n : ℕ, gcd m n = 1 ∧ (ratio = (m : ℝ) / (n : ℝ)) ∧ (m * n = 6) :=
by {
  sorry
}

end hexahedron_octahedron_ratio_l127_127611


namespace ab_value_l127_127756

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 :=
by
  sorry

end ab_value_l127_127756


namespace total_votes_l127_127806

noncomputable def total_votes_proof : Prop :=
  ∃ T A : ℝ, 
    A = 0.40 * T ∧ 
    T = A + (A + 70) ∧ 
    T = 350

theorem total_votes : total_votes_proof :=
sorry

end total_votes_l127_127806


namespace monkey_total_distance_l127_127143

theorem monkey_total_distance :
  let speedRunning := 15
  let timeRunning := 5
  let speedSwinging := 10
  let timeSwinging := 10
  let distanceRunning := speedRunning * timeRunning
  let distanceSwinging := speedSwinging * timeSwinging
  let totalDistance := distanceRunning + distanceSwinging
  totalDistance = 175 :=
by
  sorry

end monkey_total_distance_l127_127143


namespace squirrel_acorns_initial_stash_l127_127857

theorem squirrel_acorns_initial_stash (A : ℕ) 
  (h1 : 3 * (A / 3 - 60) = 30) : A = 210 := 
sorry

end squirrel_acorns_initial_stash_l127_127857


namespace eyes_given_to_dog_l127_127581

-- Definitions of the conditions
def fish_per_person : ℕ := 4
def number_of_people : ℕ := 3
def eyes_per_fish : ℕ := 2
def eyes_eaten_by_Oomyapeck : ℕ := 22

-- The proof statement
theorem eyes_given_to_dog : ∃ (eyes_given_to_dog : ℕ), eyes_given_to_dog = 4 * 3 * 2 - 22 := by
  sorry

end eyes_given_to_dog_l127_127581


namespace part1_part2_l127_127774

open BigOperators

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≠ 0 → a n > 0) ∧
  (a 1 = 2) ∧
  (∀ n : ℕ, n ≠ 0 → (n + 1) * (a (n + 1)) ^ 2 = n * (a n) ^ 2 + a n)

theorem part1 (a : ℕ → ℝ) (h : seq a)
  (n : ℕ) (hn : n ≠ 0) 
  : 1 < a (n+1) ∧ a (n+1) < a n :=
sorry

theorem part2 (a : ℕ → ℝ) (h : seq a)
  : ∑ k in Finset.range 2022 \ {0}, (a (k+1))^2 / (k+1)^2 < 2 :=
sorry

end part1_part2_l127_127774


namespace complement_is_correct_l127_127395

-- Define the universal set U and set M
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

-- Define the complement of M with respect to U
def complement_U (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

-- State the theorem to be proved
theorem complement_is_correct : complement_U U M = {3, 5, 6} :=
by
  sorry

end complement_is_correct_l127_127395


namespace math_problem_l127_127090

def calc_expr : Int := 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001

theorem math_problem :
  calc_expr = 76802 := 
by
  sorry

end math_problem_l127_127090


namespace magic_square_sum_l127_127583

theorem magic_square_sum (x y z w v: ℕ) (h1: 27 + w + 22 = 49 + w)
  (h2: 27 + 18 + x = 45 + x) (h3: 22 + 24 + y = 46 + y)
  (h4: 49 + w = 46 + y) (hw: w = y - 3) (hx: x = y + 1)
  (hz: z = x + 3) : x + z = 45 :=
by {
  sorry
}

end magic_square_sum_l127_127583


namespace jills_present_age_l127_127078

-- Define the problem parameters and conditions
variables (H J : ℕ)
axiom cond1 : H + J = 43
axiom cond2 : H - 5 = 2 * (J - 5)

-- State the goal
theorem jills_present_age : J = 16 :=
sorry

end jills_present_age_l127_127078


namespace worm_length_difference_is_correct_l127_127998

-- Define the lengths of the worms
def worm1_length : ℝ := 0.8
def worm2_length : ℝ := 0.1

-- Define the difference in length between the longer worm and the shorter worm
def length_difference (a b : ℝ) : ℝ := a - b

-- State the theorem that the length difference is 0.7 inches
theorem worm_length_difference_is_correct (h1 : worm1_length = 0.8) (h2 : worm2_length = 0.1) :
  length_difference worm1_length worm2_length = 0.7 :=
by
  sorry

end worm_length_difference_is_correct_l127_127998


namespace binom_divisibility_l127_127474

theorem binom_divisibility (p : ℕ) (h₀ : Nat.Prime p) (h₁ : p % 2 = 1) : 
  (Nat.choose (2 * p - 1) (p - 1) - 1) % (p^2) = 0 := 
by 
  sorry

end binom_divisibility_l127_127474


namespace find_the_number_l127_127234

variable (x : ℕ)

theorem find_the_number (h : 43 + 3 * x = 58) : x = 5 :=
by 
  sorry

end find_the_number_l127_127234


namespace reflect_origin_l127_127691

theorem reflect_origin (x y : ℝ) (h₁ : x = 4) (h₂ : y = -3) : 
  (-x, -y) = (-4, 3) :=
by {
  sorry
}

end reflect_origin_l127_127691


namespace grounded_days_for_lying_l127_127623

def extra_days_per_grade_below_b : ℕ := 3
def grades_below_b : ℕ := 4
def total_days_grounded : ℕ := 26

theorem grounded_days_for_lying : 
  (total_days_grounded - (grades_below_b * extra_days_per_grade_below_b) = 14) := 
by 
  sorry

end grounded_days_for_lying_l127_127623


namespace area_of_rectangle_l127_127471

theorem area_of_rectangle (A G Y : ℝ) 
  (hG : G = 0.15 * A) 
  (hY : Y = 21) 
  (hG_plus_Y : G + Y = 0.5 * A) : 
  A = 60 := 
by 
  -- proof goes here
  sorry

end area_of_rectangle_l127_127471


namespace remaining_cube_height_l127_127020

/-- Given a cube with side length 2 units, where a corner is chopped off such that the cut runs
    through points on the three edges adjacent to a selected vertex, each at 1 unit distance
    from that vertex, the height of the remaining portion of the cube when the freshly cut face 
    is placed on a table is equal to (5 * sqrt 3) / 3. -/
theorem remaining_cube_height (s : ℝ) (h : ℝ) : 
    s = 2 → h = 1 → 
    ∃ height : ℝ, height = (5 * Real.sqrt 3) / 3 := 
by
    sorry

end remaining_cube_height_l127_127020


namespace center_of_circle_sum_eq_seven_l127_127221

theorem center_of_circle_sum_eq_seven 
  (h k : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 6 * x + 8 * y - 15 → (x - h)^2 + (y - k)^2 = 10) :
  h + k = 7 := 
sorry

end center_of_circle_sum_eq_seven_l127_127221


namespace total_animals_correct_l127_127208

-- Define the number of aquariums and the number of animals per aquarium.
def num_aquariums : ℕ := 26
def animals_per_aquarium : ℕ := 2

-- Define the total number of saltwater animals.
def total_animals : ℕ := num_aquariums * animals_per_aquarium

-- The statement we want to prove.
theorem total_animals_correct : total_animals = 52 := by
  -- Proof is omitted.
  sorry

end total_animals_correct_l127_127208


namespace days_taken_to_complete_work_l127_127792

-- Conditions
def work_rate_B : ℚ := 1 / 33
def work_rate_A : ℚ := 2 * work_rate_B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Proof statement
theorem days_taken_to_complete_work : combined_work_rate ≠ 0 → 1 / combined_work_rate = 11 :=
by
  sorry

end days_taken_to_complete_work_l127_127792


namespace intersection_of_A_and_B_l127_127999

open Set Int

def A : Set ℝ := { x | x ^ 2 - 6 * x + 8 ≤ 0 }
def B : Set ℤ := { x | abs (x - 3) < 2 }

theorem intersection_of_A_and_B :
  (A ∩ (coe '' B) = { x : ℝ | x = 2 ∨ x = 3 ∨ x = 4 }) :=
by
  sorry

end intersection_of_A_and_B_l127_127999


namespace sum_first_2017_terms_l127_127086

theorem sum_first_2017_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → S (n + 1) - S n = 3^n / a n) :
  S 2017 = 3^1009 - 2 := sorry

end sum_first_2017_terms_l127_127086


namespace intersection_M_N_l127_127628

open Real

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 2 - abs x > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
sorry

end intersection_M_N_l127_127628


namespace simplify_expr_l127_127765

-- Define the expression
def expr := |-4^2 + 7|

-- State the theorem
theorem simplify_expr : expr = 9 :=
by sorry

end simplify_expr_l127_127765


namespace find_eighth_term_l127_127614

noncomputable def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + n * d

theorem find_eighth_term (a d : ℕ) :
  (arithmetic_sequence a d 0) + 
  (arithmetic_sequence a d 1) + 
  (arithmetic_sequence a d 2) + 
  (arithmetic_sequence a d 3) + 
  (arithmetic_sequence a d 4) + 
  (arithmetic_sequence a d 5) = 21 ∧
  arithmetic_sequence a d 6 = 7 →
  arithmetic_sequence a d 7 = 8 :=
by
  sorry

end find_eighth_term_l127_127614


namespace trigonometric_identity_x1_trigonometric_identity_x2_l127_127350

noncomputable def x1 (n : ℤ) : ℝ := (2 * n + 1) * (Real.pi / 4)
noncomputable def x2 (k : ℤ) : ℝ := ((-1)^(k + 1)) * (Real.pi / 8) + k * (Real.pi / 2)

theorem trigonometric_identity_x1 (n : ℤ) : 
  (Real.cos (4 * x1 n) * Real.cos (Real.pi + 2 * x1 n) - 
   Real.sin (2 * x1 n) * Real.cos (Real.pi / 2 - 4 * x1 n)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x1 n) := 
by
  sorry

theorem trigonometric_identity_x2 (k : ℤ) : 
  (Real.cos (4 * x2 k) * Real.cos (Real.pi + 2 * x2 k) - 
   Real.sin (2 * x2 k) * Real.cos (Real.pi / 2 - 4 * x2 k)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x2 k) := 
by
  sorry

end trigonometric_identity_x1_trigonometric_identity_x2_l127_127350


namespace convert_234_base5_to_binary_l127_127259

def base5_to_decimal (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 4 * 5^0

def decimal_to_binary (n : Nat) : List Nat :=
  let rec to_binary_aux (n : Nat) (accum : List Nat) : List Nat :=
    if n = 0 then accum
    else to_binary_aux (n / 2) ((n % 2) :: accum)
  to_binary_aux n []

theorem convert_234_base5_to_binary :
  (base5_to_decimal 234 = 69) ∧ (decimal_to_binary 69 = [1,0,0,0,1,0,1]) :=
by
  sorry

end convert_234_base5_to_binary_l127_127259


namespace investor_difference_l127_127130

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end investor_difference_l127_127130


namespace find_f_neg2014_l127_127280

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem find_f_neg2014 (a b : ℝ) (h : f 2014 a b = 3) : f (-2014) a b = -7 :=
by sorry

end find_f_neg2014_l127_127280


namespace volume_of_water_flowing_per_minute_l127_127019

variable (d w r : ℝ) (V : ℝ)

theorem volume_of_water_flowing_per_minute (h1 : d = 3) 
                                           (h2 : w = 32) 
                                           (h3 : r = 33.33) : 
  V = 3199.68 :=
by
  sorry

end volume_of_water_flowing_per_minute_l127_127019


namespace combined_salaries_l127_127336
-- Import the required libraries

-- Define the salaries and conditions
def salary_c := 14000
def avg_salary_five := 8600
def num_individuals := 5
def total_salary := avg_salary_five * num_individuals

-- Define what we need to prove
theorem combined_salaries : total_salary - salary_c = 29000 :=
by
  -- The theorem statement
  sorry

end combined_salaries_l127_127336


namespace initial_blocks_l127_127201

variable (x : ℕ)

theorem initial_blocks (h : x + 30 = 65) : x = 35 := by
  sorry

end initial_blocks_l127_127201


namespace green_shirt_pairs_l127_127380

theorem green_shirt_pairs (blue_shirts green_shirts total_pairs blue_blue_pairs : ℕ) 
(h1 : blue_shirts = 68) 
(h2 : green_shirts = 82) 
(h3 : total_pairs = 75) 
(h4 : blue_blue_pairs = 30) 
: (green_shirts - (blue_shirts - 2 * blue_blue_pairs)) / 2 = 37 := 
by 
  -- This is where the proof would be written, but we use sorry to skip it.
  sorry

end green_shirt_pairs_l127_127380


namespace number_of_6mb_pictures_l127_127741

theorem number_of_6mb_pictures
    (n : ℕ)             -- initial number of pictures
    (size_old : ℕ)      -- size of old pictures in megabytes
    (size_new : ℕ)      -- size of new pictures in megabytes
    (total_capacity : ℕ)  -- total capacity of the memory card in megabytes
    (h1 : n = 3000)      -- given memory card can hold 3000 pictures
    (h2 : size_old = 8)  -- each old picture is 8 megabytes
    (h3 : size_new = 6)  -- each new picture is 6 megabytes
    (h4 : total_capacity = n * size_old)  -- total capacity calculated from old pictures
    : total_capacity / size_new = 4000 :=  -- the number of new pictures that can be held
by
  sorry

end number_of_6mb_pictures_l127_127741


namespace mouse_away_from_cheese_l127_127194

theorem mouse_away_from_cheese:
  ∃ a b : ℝ, a = 3 ∧ b = 3 ∧ (a + b = 6) ∧
  ∀ x y : ℝ, (y = -3 * x + 12) → 
  ∀ (a y₀ : ℝ), y₀ = (1/3) * a + 11 →
  (a, b) = (3, 3) :=
by
  sorry

end mouse_away_from_cheese_l127_127194


namespace Nikolai_faster_than_Gennady_l127_127853

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l127_127853


namespace evaluate_expression_l127_127042

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end evaluate_expression_l127_127042


namespace pet_store_problem_l127_127711

noncomputable def num_ways_to_buy_pets (puppies kittens hamsters birds : ℕ) (people : ℕ) : ℕ :=
  (puppies * kittens * hamsters * birds) * (people.factorial)

theorem pet_store_problem :
  num_ways_to_buy_pets 12 10 5 3 4 = 43200 :=
by
  sorry

end pet_store_problem_l127_127711


namespace tennis_handshakes_l127_127206

theorem tennis_handshakes :
  let num_teams := 4
  let women_per_team := 2
  let total_women := num_teams * women_per_team
  let handshakes_per_woman := total_women - 2
  let total_handshakes_before_division := total_women * handshakes_per_woman
  let actual_handshakes := total_handshakes_before_division / 2
  actual_handshakes = 24 :=
by sorry

end tennis_handshakes_l127_127206


namespace find_y_l127_127360

theorem find_y (y : ℚ) (h : Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9) : y = 68 / 3 := 
by
  sorry

end find_y_l127_127360


namespace modulo_problem_l127_127472

theorem modulo_problem :
  (47 ^ 2051 - 25 ^ 2051) % 5 = 3 := by
  sorry

end modulo_problem_l127_127472


namespace tank_full_capacity_is_72_l127_127483

theorem tank_full_capacity_is_72 (x : ℝ) 
  (h1 : 0.9 * x - 0.4 * x = 36) : 
  x = 72 := 
sorry

end tank_full_capacity_is_72_l127_127483


namespace sqrt_200_eq_10_sqrt_2_l127_127991

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l127_127991


namespace equation_has_one_integral_root_l127_127949

theorem equation_has_one_integral_root:
  ∃ x : ℤ, (x - 9 / (x + 4 : ℝ) = 2 - 9 / (x + 4 : ℝ)) ∧ ∀ y : ℤ, 
  (y - 9 / (y + 4 : ℝ) = 2 - 9 / (y + 4 : ℝ)) → y = x := 
by
  sorry

end equation_has_one_integral_root_l127_127949


namespace minuend_is_not_integer_l127_127730

theorem minuend_is_not_integer (M S D : ℚ) (h1 : M + S + D = 555) (h2 : M - S = D) : ¬ ∃ n : ℤ, M = n := 
by
  sorry

end minuend_is_not_integer_l127_127730


namespace unpainted_cubes_eq_210_l127_127896

-- Defining the structure of the 6x6x6 cube
def cube := Fin 6 × Fin 6 × Fin 6

-- Number of unit cubes in a 6x6x6 cube
def total_cubes : ℕ := 6 * 6 * 6

-- Number of unit squares painted by the plus pattern on each face
def squares_per_face := 13

-- Number of faces on the cube
def faces := 6

-- Initial total number of painted squares
def initial_painted_squares := squares_per_face * faces

-- Number of over-counted squares along edges
def edge_overcount := 12 * 2

-- Number of over-counted squares at corners
def corner_overcount := 8 * 1

-- Adjusted number of painted unit squares accounting for overcounts
noncomputable def adjusted_painted_squares := initial_painted_squares - edge_overcount - corner_overcount

-- Overlap adjustment: edge units and corner units
def edges_overlap := 24
def corners_overlap := 16

-- Final number of unique painted unit cubes
noncomputable def unique_painted_cubes := adjusted_painted_squares - edges_overlap - corners_overlap

-- Final unpainted unit cubes calculation
noncomputable def unpainted_cubes := total_cubes - unique_painted_cubes

-- Theorem to prove the number of unpainted unit cubes is 210
theorem unpainted_cubes_eq_210 : unpainted_cubes = 210 := by
  sorry

end unpainted_cubes_eq_210_l127_127896


namespace book_pairs_count_l127_127374

theorem book_pairs_count :
  let mystery_books := 4
  let science_fiction_books := 4
  let historical_books := 4
  (mystery_books + science_fiction_books + historical_books) = 12 ∧ 
  (mystery_books = 4 ∧ science_fiction_books = 4 ∧ historical_books = 4) →
  let genres := 3
  ∃ pairs, pairs = 48 :=
by
  sorry

end book_pairs_count_l127_127374


namespace bobby_pancakes_left_l127_127643

def total_pancakes : ℕ := 21
def pancakes_eaten_by_bobby : ℕ := 5
def pancakes_eaten_by_dog : ℕ := 7

theorem bobby_pancakes_left : total_pancakes - (pancakes_eaten_by_bobby + pancakes_eaten_by_dog) = 9 :=
  by
  sorry

end bobby_pancakes_left_l127_127643


namespace part_I_part_II_l127_127151

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∀ x : ℝ, g x 3 > -1 ↔ x = -3) :=
by
  sorry

theorem part_II (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x a ≥ g x m) ↔ (a < 4) :=
by
  sorry

end part_I_part_II_l127_127151


namespace simplify_fraction_l127_127298

-- Define the fractions and the product
def fraction1 : ℚ := 18 / 11
def fraction2 : ℚ := -42 / 45
def product : ℚ := 15 * fraction1 * fraction2

-- State the theorem to prove the correctness of the simplification
theorem simplify_fraction : product = -23 + 1 / 11 :=
by
  -- Adding this as a placeholder. The proof would go here.
  sorry

end simplify_fraction_l127_127298


namespace intersection_unique_point_l127_127615

theorem intersection_unique_point
    (h1 : ∀ (x y : ℝ), 2 * x + 3 * y = 6)
    (h2 : ∀ (x y : ℝ), 4 * x - 3 * y = 6)
    (h3 : ∀ y : ℝ, 2 = 2)
    (h4 : ∀ x : ℝ, y = 2 / 3)
    : ∃! (x y : ℝ), (2 * x + 3 * y = 6) ∧ (4 * x - 3 * y = 6) ∧ (x = 2) ∧ (y = 2 / 3) := 
by
    sorry

end intersection_unique_point_l127_127615


namespace halfway_fraction_l127_127003

theorem halfway_fraction (a b : ℚ) (h1 : a = 2 / 9) (h2 : b = 1 / 3) :
  (a + b) / 2 = 5 / 18 :=
by
  sorry

end halfway_fraction_l127_127003


namespace max_total_profit_max_avg_annual_profit_l127_127060

noncomputable def total_profit (x : ℕ) : ℝ := - (x : ℝ)^2 + 18 * x - 36
noncomputable def avg_annual_profit (x : ℕ) : ℝ := (total_profit x) / x

theorem max_total_profit : ∃ x : ℕ, total_profit x = 45 ∧ x = 9 :=
  by sorry

theorem max_avg_annual_profit : ∃ x : ℕ, avg_annual_profit x = 6 ∧ x = 6 :=
  by sorry

end max_total_profit_max_avg_annual_profit_l127_127060


namespace translate_parabola_l127_127700

theorem translate_parabola (x : ℝ) :
  (x^2 + 3) = (x - 5)^2 + 3 :=
sorry

end translate_parabola_l127_127700


namespace jindra_gray_fields_counts_l127_127410

-- Definitions for the problem setup
noncomputable def initial_gray_fields: ℕ := 7
noncomputable def rotation_90_gray_fields: ℕ := 8
noncomputable def rotation_180_gray_fields: ℕ := 4

-- Statement of the theorem to be proved
theorem jindra_gray_fields_counts:
  initial_gray_fields = 7 ∧
  rotation_90_gray_fields = 8 ∧
  rotation_180_gray_fields = 4 := by
  sorry

end jindra_gray_fields_counts_l127_127410


namespace ceiling_example_l127_127140

/-- Lean 4 statement of the proof problem:
    Prove that ⌈4 (8 - 1/3)⌉ = 31.
-/
theorem ceiling_example : Int.ceil (4 * (8 - (1 / 3 : ℝ))) = 31 := 
by
  sorry

end ceiling_example_l127_127140


namespace total_distance_dog_runs_l127_127494

-- Define the distance between Xiaoqiang's home and his grandmother's house in meters
def distance_home_to_grandma : ℕ := 1000

-- Define Xiaoqiang's walking speed in meters per minute
def xiaoqiang_speed : ℕ := 50

-- Define the dog's running speed in meters per minute
def dog_speed : ℕ := 200

-- Define the time Xiaoqiang takes to reach his grandmother's house
def xiaoqiang_time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the total distance the dog runs given the speeds and distances
theorem total_distance_dog_runs (d x_speed dog_speed : ℕ) 
  (hx : x_speed > 0) (hd : dog_speed > 0) : (d / x_speed) * dog_speed = 4000 :=
  sorry

end total_distance_dog_runs_l127_127494


namespace unique_prime_p_l127_127100

def f (x : ℤ) : ℤ := x^3 + 7 * x^2 + 9 * x + 10

theorem unique_prime_p (p : ℕ) (hp : p = 5 ∨ p = 7 ∨ p = 11 ∨ p = 13 ∨ p = 17) :
  (∀ a b : ℤ, f a ≡ f b [ZMOD p] → a ≡ b [ZMOD p]) ↔ p = 11 :=
by
  sorry

end unique_prime_p_l127_127100


namespace length_of_each_part_l127_127938

-- Conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def parts_count : ℕ := 4

-- Question
theorem length_of_each_part : total_length_in_inches / parts_count = 20 :=
by
  -- leave the proof as a sorry
  sorry

end length_of_each_part_l127_127938


namespace sum_of_possible_values_N_l127_127715

variable (a b c N : ℕ)

theorem sum_of_possible_values_N :
  (N = a * b * c) ∧ (N = 8 * (a + b + c)) ∧ (c = 2 * a + b) → N = 136 := 
by
  sorry

end sum_of_possible_values_N_l127_127715


namespace simplify_expression_l127_127154

theorem simplify_expression (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) 
  (h : x^2 + y^2 + z^2 = xy + yz + zx) : 
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) = 3 / x^2 := 
by
  sorry

end simplify_expression_l127_127154


namespace first_day_more_than_300_l127_127601

def paperclips (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_more_than_300 : ∃ n, paperclips n > 300 ∧ n = 4 := by
  sorry

end first_day_more_than_300_l127_127601


namespace cost_of_50_tulips_l127_127635

theorem cost_of_50_tulips (c : ℕ → ℝ) :
  (∀ n : ℕ, n ≤ 40 → c n = n * (36 / 18)) ∧
  (∀ n : ℕ, n > 40 → c n = (40 * (36 / 18) + (n - 40) * (36 / 18)) * 0.9) ∧
  (c 18 = 36) →
  c 50 = 90 := sorry

end cost_of_50_tulips_l127_127635


namespace vanessa_savings_weeks_l127_127402

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l127_127402


namespace walter_percent_of_dollar_l127_127850

theorem walter_percent_of_dollar
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (penny_value : Nat := 1)
  (nickel_value : Nat := 5)
  (dime_value : Nat := 10)
  (dollar_value : Nat := 100)
  (total_value := pennies * penny_value + nickels * nickel_value + dimes * dime_value) :
  pennies = 2 ∧ nickels = 3 ∧ dimes = 2 →
  (total_value * 100) / dollar_value = 37 :=
by
  sorry

end walter_percent_of_dollar_l127_127850


namespace intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l127_127376

noncomputable def setA : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | 1 < x }

theorem intersection_AB : setA ∩ setB = { x : ℝ | 1 < x ∧ x < 2 } := by
  sorry

theorem union_AB : setA ∪ setB = { x : ℝ | -1 < x } := by
  sorry

theorem difference_A_minus_B : setA \ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } := by
  sorry

theorem difference_B_minus_A : setB \ setA = { x : ℝ | 2 ≤ x } := by
  sorry

end intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l127_127376


namespace floor_eq_48_iff_l127_127102

-- Define the real number set I to be [8, 49/6)
def I : Set ℝ := { x | 8 ≤ x ∧ x < 49/6 }

-- The main statement to be proven
theorem floor_eq_48_iff (x : ℝ) : (Int.floor (x * Int.floor x) = 48) ↔ x ∈ I := 
by
  sorry

end floor_eq_48_iff_l127_127102


namespace value_of_a_l127_127525

theorem value_of_a (a : ℝ) (h : (a ^ 3) * ((5).choose (2)) = 80) : a = 2 :=
  sorry

end value_of_a_l127_127525


namespace amy_tickets_initial_l127_127312

theorem amy_tickets_initial (x : ℕ) (h1 : x + 21 = 54) : x = 33 :=
by sorry

end amy_tickets_initial_l127_127312


namespace union_of_A_and_B_l127_127906

open Set

def A : Set ℕ := {1, 3, 7, 8}
def B : Set ℕ := {1, 5, 8}

theorem union_of_A_and_B : A ∪ B = {1, 3, 5, 7, 8} := by
  sorry

end union_of_A_and_B_l127_127906


namespace fox_jeans_price_l127_127542

theorem fox_jeans_price (F : ℝ) (P : ℝ) 
  (pony_price : P = 18) 
  (total_savings : 3 * F * 0.08 + 2 * P * 0.14 = 8.64)
  (total_discount_rate : 0.08 + 0.14 = 0.22)
  (pony_discount_rate : 0.14 = 13.999999999999993 / 100) 
  : F = 15 :=
by
  sorry

end fox_jeans_price_l127_127542


namespace smallest_number_of_marbles_l127_127771

theorem smallest_number_of_marbles :
  ∃ (r w b g y : ℕ), 
  (r + w + b + g + y = 13) ∧ 
  (r ≥ 5) ∧
  (r - 4 = 5 * w) ∧
  ((r - 3) * (r - 4) = 20 * w * b) ∧
  sorry := sorry

end smallest_number_of_marbles_l127_127771


namespace janet_final_lives_l127_127560

-- Given conditions
def initial_lives : ℕ := 47
def lives_lost_in_game : ℕ := 23
def points_collected : ℕ := 1840
def lives_per_100_points : ℕ := 2
def penalty_per_200_points : ℕ := 1

-- Definitions based on conditions
def remaining_lives_after_game : ℕ := initial_lives - lives_lost_in_game
def lives_earned_from_points : ℕ := (points_collected / 100) * lives_per_100_points
def lives_lost_due_to_penalties : ℕ := points_collected / 200

-- Theorem statement
theorem janet_final_lives : remaining_lives_after_game + lives_earned_from_points - lives_lost_due_to_penalties = 51 :=
by
  sorry

end janet_final_lives_l127_127560


namespace largest_common_element_l127_127131

theorem largest_common_element (S1 S2 : ℕ → ℕ) (a_max : ℕ) :
  (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) →
  (147 < a_max) →
  ∀ m, (m < a_max → (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) → 147 = 27 + 40 * 3) :=
sorry

end largest_common_element_l127_127131


namespace gcd_360_1260_l127_127238

theorem gcd_360_1260 : gcd 360 1260 = 180 := by
  /- 
  Prime factorization of 360 and 1260 is given:
  360 = 2^3 * 3^2 * 5
  1260 = 2^2 * 3^2 * 5 * 7
  These conditions are implicitly used to deduce the answer.
  -/
  sorry

end gcd_360_1260_l127_127238


namespace trim_length_l127_127931

theorem trim_length {π : ℝ} (r : ℝ)
  (π_approx : π = 22 / 7)
  (area : π * r^2 = 616) :
  2 * π * r + 5 = 93 :=
by
  sorry

end trim_length_l127_127931


namespace tom_read_in_five_months_l127_127223

def books_in_may : ℕ := 2
def books_in_june : ℕ := 6
def books_in_july : ℕ := 12
def books_in_august : ℕ := 20
def books_in_september : ℕ := 30

theorem tom_read_in_five_months : 
  books_in_may + books_in_june + books_in_july + books_in_august + books_in_september = 70 := by
  sorry

end tom_read_in_five_months_l127_127223


namespace pages_written_on_wednesday_l127_127912

variable (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ)
variable (totalPages : ℕ)

def pagesOnMonday (minutesMonday rateMonday : ℕ) : ℕ :=
  minutesMonday / rateMonday

def pagesOnTuesday (minutesTuesday rateTuesday : ℕ) : ℕ :=
  minutesTuesday / rateTuesday

def totalPagesMondayAndTuesday (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ) : ℕ :=
  pagesOnMonday minutesMonday rateMonday + pagesOnTuesday minutesTuesday rateTuesday

def pagesOnWednesday (minutesMonday minutesTuesday rateMonday rateTuesday totalPages : ℕ) : ℕ :=
  totalPages - totalPagesMondayAndTuesday minutesMonday minutesTuesday rateMonday rateTuesday

theorem pages_written_on_wednesday :
  pagesOnWednesday 60 45 30 15 10 = 5 := by
  sorry

end pages_written_on_wednesday_l127_127912


namespace mixture_replacement_l127_127226

theorem mixture_replacement
  (A B : ℕ)
  (hA : A = 48)
  (h_ratio1 : A / B = 4)
  (x : ℕ)
  (h_ratio2 : A / (B + x) = 2 / 3) :
  x = 60 :=
by
  sorry

end mixture_replacement_l127_127226


namespace remainder_x2023_l127_127393

theorem remainder_x2023 (x : ℤ) : 
  let dividend := x^2023 + 1
  let divisor := x^6 - x^4 + x^2 - 1
  let remainder := -x^7 + 1
  dividend % divisor = remainder :=
by
  sorry

end remainder_x2023_l127_127393


namespace equal_sum_sequence_S_9_l127_127011

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions taken from the problem statement
def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) :=
  ∀ n : ℕ, a n + a (n + 1) = c

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Lean statement of the problem
theorem equal_sum_sequence_S_9
  (h1 : equal_sum_sequence a 5)
  (h2 : a 1 = 2)
  : sum_first_n_terms a 9 = 22 :=
sorry

end equal_sum_sequence_S_9_l127_127011


namespace bike_price_l127_127055

theorem bike_price (x : ℝ) (h1 : 0.1 * x = 150) : x = 1500 := 
by sorry

end bike_price_l127_127055


namespace sin_cos_value_l127_127029

-- Given function definition
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^2 + (Real.sin α - 2 * Real.cos α) * x + 1

-- Definitions and proof problem statement
theorem sin_cos_value (α : ℝ) : 
  (∀ x : ℝ, f α x = f α (-x)) → (Real.sin α * Real.cos α = 2 / 5) :=
by
  intro h_even
  sorry

end sin_cos_value_l127_127029


namespace perfect_squares_difference_l127_127087

theorem perfect_squares_difference : 
  let N : ℕ := 20000;
  let diff_squared (b : ℤ) : ℤ := (b+2)^2 - b^2;
  ∃ k : ℕ, (1 ≤ k ∧ k ≤ 70) ∧ (∀ m : ℕ, (m < N) → (∃ b : ℤ, m = diff_squared b) → m = (2 * k)^2)
:= sorry

end perfect_squares_difference_l127_127087


namespace gabby_needs_more_money_l127_127413

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end gabby_needs_more_money_l127_127413


namespace burmese_python_eats_alligators_l127_127779

theorem burmese_python_eats_alligators (snake_length : ℝ) (alligator_length : ℝ) (alligator_per_week : ℝ) (total_alligators : ℝ) :
  snake_length = 1.4 → alligator_length = 0.5 → alligator_per_week = 1 → total_alligators = 88 →
  (total_alligators / alligator_per_week) * 7 = 616 := by
  intros
  sorry

end burmese_python_eats_alligators_l127_127779


namespace angle_half_second_quadrant_l127_127842

theorem angle_half_second_quadrant (α : ℝ) (k : ℤ) :
  (π / 2 + 2 * k * π < α ∧ α < π + 2 * k * π) → 
  (∃ m : ℤ, (π / 4 + m * π < α / 2 ∧ α / 2 < π / 2 + m * π)) ∨ 
  (∃ n : ℤ, (5 * π / 4 + n * π < α / 2 ∧ α / 2 < 3 * π / 2 + n * π)) :=
by
  sorry

end angle_half_second_quadrant_l127_127842


namespace number_of_rings_l127_127538

def is_number_ring (A : Set ℝ) : Prop :=
  ∀ (a b : ℝ), a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A ∧ (a * b) ∈ A

def Z := { n : ℝ | ∃ k : ℤ, n = k }
def N := { n : ℝ | ∃ k : ℕ, n = k }
def Q := { n : ℝ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b }
def R := { n : ℝ | True }
def M := { x : ℝ | ∃ (n m : ℤ), x = n + m * Real.sqrt 2 }
def P := { x : ℝ | ∃ (m n : ℕ), n ≠ 0 ∧ x = m / (2 * n) }

theorem number_of_rings :
  (is_number_ring Z) ∧ ¬(is_number_ring N) ∧ (is_number_ring Q) ∧ 
  (is_number_ring R) ∧ (is_number_ring M) ∧ ¬(is_number_ring P) :=
by sorry

end number_of_rings_l127_127538


namespace circleEquation_and_pointOnCircle_l127_127149

-- Definition of the Cartesian coordinate system and the circle conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def inSecondQuadrant (p : ℝ × ℝ) := p.1 < 0 ∧ p.2 > 0

def tangentToLine (C : Circle) (line : ℝ → ℝ) (tangentPoint : ℝ × ℝ) :=
  let centerToLineDistance := (abs (C.center.1 - C.center.2)) / Real.sqrt 2
  C.radius = centerToLineDistance ∧ tangentPoint = (0, 0)

-- Main statements to prove
theorem circleEquation_and_pointOnCircle :
  ∃ C : Circle, ∃ Q : ℝ × ℝ,
    inSecondQuadrant C.center ∧
    C.radius = 2 * Real.sqrt 2 ∧
    tangentToLine C (fun x => x) (0, 0) ∧
    ((∃ p : ℝ × ℝ, p = (-2, 2) ∧ C = Circle.mk p (2 * Real.sqrt 2) ∧
      (∀ x y : ℝ, ((x + 2)^2 + (y - 2)^2 = 8))) ∧
    (Q = (4/5, 12/5) ∧
      ((Q.1 + 2)^2 + (Q.2 - 2)^2 = 8) ∧
      Real.sqrt ((Q.1 - 4)^2 + Q.2^2) = 4))
    := sorry

end circleEquation_and_pointOnCircle_l127_127149


namespace b_investment_months_after_a_l127_127696

-- Definitions based on the conditions
def a_investment : ℕ := 100
def b_investment : ℕ := 200
def total_yearly_investment_period : ℕ := 12
def total_profit : ℕ := 100
def a_share_of_profit : ℕ := 50
def x (x_val : ℕ) : Prop := x_val = 6

-- Main theorem to prove
theorem b_investment_months_after_a (x_val : ℕ) 
  (h1 : a_investment = 100)
  (h2 : b_investment = 200)
  (h3 : total_yearly_investment_period = 12)
  (h4 : total_profit = 100)
  (h5 : a_share_of_profit = 50) :
  (100 * total_yearly_investment_period) = 200 * (total_yearly_investment_period - x_val) → 
  x x_val := 
by
  sorry

end b_investment_months_after_a_l127_127696


namespace shuttlecock_weight_probability_l127_127825

variable (p_lt_4_8 : ℝ) -- Probability that its weight is less than 4.8 g
variable (p_le_4_85 : ℝ) -- Probability that its weight is not greater than 4.85 g

theorem shuttlecock_weight_probability (h1 : p_lt_4_8 = 0.3) (h2 : p_le_4_85 = 0.32) :
  p_le_4_85 - p_lt_4_8 = 0.02 :=
by
  sorry

end shuttlecock_weight_probability_l127_127825


namespace initial_money_l127_127791

theorem initial_money (x : ℝ) (cupcake_cost total_cookie_cost total_cost money_left : ℝ) 
  (h1 : cupcake_cost = 10 * 1.5) 
  (h2 : total_cookie_cost = 5 * 3)
  (h3 : total_cost = cupcake_cost + total_cookie_cost)
  (h4 : money_left = 30)
  (h5 : 3 * x = total_cost + money_left) 
  : x = 20 := 
sorry

end initial_money_l127_127791


namespace length_of_wooden_block_l127_127544

theorem length_of_wooden_block (cm_to_m : ℝ := 30 / 100) (base_length : ℝ := 31) :
  base_length + cm_to_m = 31.3 :=
by
  sorry

end length_of_wooden_block_l127_127544


namespace molar_weight_of_BaF2_l127_127777

theorem molar_weight_of_BaF2 (Ba_weight : Real) (F_weight : Real) (num_moles : ℕ) 
    (Ba_weight_val : Ba_weight = 137.33) (F_weight_val : F_weight = 18.998) 
    (num_moles_val : num_moles = 6) 
    : (137.33 + 2 * 18.998) * 6 = 1051.956 := 
by
  sorry

end molar_weight_of_BaF2_l127_127777


namespace niko_total_profit_l127_127875

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end niko_total_profit_l127_127875


namespace joe_money_left_l127_127368

theorem joe_money_left
  (joe_savings : ℕ := 6000)
  (flight_cost : ℕ := 1200)
  (hotel_cost : ℕ := 800)
  (food_cost : ℕ := 3000) :
  joe_savings - (flight_cost + hotel_cost + food_cost) = 1000 :=
by
  sorry

end joe_money_left_l127_127368


namespace largest_divisor_of_expression_l127_127523

theorem largest_divisor_of_expression (x : ℤ) (h : x % 2 = 1) : 
  324 ∣ (12 * x + 3) * (12 * x + 9) * (6 * x + 6) :=
sorry

end largest_divisor_of_expression_l127_127523


namespace smallest_y_value_l127_127377

theorem smallest_y_value (y : ℝ) : (12 * y^2 - 56 * y + 48 = 0) → y = 2 :=
by
  sorry

end smallest_y_value_l127_127377


namespace pastries_left_to_take_home_l127_127851

def initial_cupcakes : ℕ := 7
def initial_cookies : ℕ := 5
def pastries_sold : ℕ := 4

theorem pastries_left_to_take_home :
  initial_cupcakes + initial_cookies - pastries_sold = 8 := by
  sorry

end pastries_left_to_take_home_l127_127851


namespace least_x_l127_127452

theorem least_x (x p : ℕ) (h1 : 0 < x) (h2: Nat.Prime p) (h3: ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : x ≥ 66 := 
sorry

end least_x_l127_127452


namespace temperature_difference_l127_127488

theorem temperature_difference (high low : ℝ) (h_high : high = 5) (h_low : low = -3) :
  high - low = 8 :=
by {
  -- Proof goes here
  sorry
}

end temperature_difference_l127_127488


namespace cauliflower_sales_l127_127767

namespace WeeklyMarket

def broccoliPrice := 3
def totalEarnings := 520
def broccolisSold := 19

def carrotPrice := 2
def spinachPrice := 4
def spinachWeight := 8 -- This is derived from solving $4S = 2S + $16 

def broccoliEarnings := broccolisSold * broccoliPrice
def carrotEarnings := spinachWeight * carrotPrice -- This is twice copied

def spinachEarnings : ℕ := spinachWeight * spinachPrice
def tomatoEarnings := broccoliEarnings + spinachEarnings

def otherEarnings : ℕ := broccoliEarnings + carrotEarnings + spinachEarnings + tomatoEarnings

def cauliflowerEarnings : ℕ := totalEarnings - otherEarnings -- This directly from subtraction of earnings

theorem cauliflower_sales : cauliflowerEarnings = 310 :=
  by
    -- only the statement part, no actual proof needed
    sorry

end WeeklyMarket

end cauliflower_sales_l127_127767


namespace scatter_plot_convention_l127_127874

def explanatory_variable := "x-axis"
def predictor_variable := "y-axis"

theorem scatter_plot_convention :
  explanatory_variable = "x-axis" ∧ predictor_variable = "y-axis" :=
by sorry

end scatter_plot_convention_l127_127874


namespace distinct_prime_factors_of_90_l127_127914

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l127_127914


namespace solution_set_f_derivative_l127_127539

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 1

theorem solution_set_f_derivative :
  { x : ℝ | (deriv f x) < 0 } = { x : ℝ | -1 < x ∧ x < 3 } :=
by
  sorry

end solution_set_f_derivative_l127_127539


namespace Maria_score_l127_127502

theorem Maria_score (x : ℝ) (y : ℝ) (h1 : x = y + 50) (h2 : (x + y) / 2 = 105) : x = 130 :=
by
  sorry

end Maria_score_l127_127502


namespace arithmetic_seq_a12_l127_127859

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (a1 d : ℝ) 
  (h_arith : arithmetic_seq a a1 d)
  (h7_and_9 : a 7 + a 9 = 16)
  (h4 : a 4 = 1) :
  a 12 = 15 :=
by
  sorry

end arithmetic_seq_a12_l127_127859


namespace remove_terms_sum_l127_127205

theorem remove_terms_sum :
  let s := (1/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13 + 1/15 : ℚ)
  s = 16339/15015 →
  (1/13 + 1/15 = 2061/5005) →
  s - (1/13 + 1/15) = 3/2 :=
by
  intros s hs hremove
  have hrem : (s - (1/13 + 1/15 : ℚ) = 3/2) ↔ (16339/15015 - 2061/5005 = 3/2) := sorry
  exact hrem.mpr sorry

end remove_terms_sum_l127_127205


namespace lowest_possible_price_l127_127317

theorem lowest_possible_price
  (MSRP : ℝ)
  (D1 : ℝ)
  (D2 : ℝ)
  (P_final : ℝ)
  (h1 : MSRP = 45.00)
  (h2 : 0.10 ≤ D1 ∧ D1 ≤ 0.30)
  (h3 : D2 = 0.20) :
  P_final = 25.20 :=
by
  sorry

end lowest_possible_price_l127_127317


namespace loan_amount_principal_l127_127898

-- Definitions based on conditions
def rate_of_interest := 3
def time_period := 3
def simple_interest := 108

-- Question translated to Lean 4 statement
theorem loan_amount_principal : ∃ P, (simple_interest = (P * rate_of_interest * time_period) / 100) ∧ P = 1200 :=
sorry

end loan_amount_principal_l127_127898


namespace inequality_proof_l127_127355

theorem inequality_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
    (a * b + b * c + c * a) * (1 / (a + b)^2 + 1 / (b + c)^2 + 1 / (c + a)^2) ≥ 9 / 4 := 
by
  sorry

end inequality_proof_l127_127355


namespace julia_cakes_remaining_l127_127305

namespace CakeProblem

def cakes_per_day : ℕ := 5 - 1
def days_baked : ℕ := 6
def total_cakes_baked : ℕ := cakes_per_day * days_baked
def days_clifford_eats : ℕ := days_baked / 2
def cakes_eaten_by_clifford : ℕ := days_clifford_eats

theorem julia_cakes_remaining : total_cakes_baked - cakes_eaten_by_clifford = 21 :=
by
  -- proof goes here
  sorry

end CakeProblem

end julia_cakes_remaining_l127_127305


namespace unique_prime_sum_diff_l127_127150

theorem unique_prime_sum_diff (p : ℕ) (primeP : Prime p)
  (hx : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p = x + y)
  (hz : ∃ (z w : ℕ), Prime z ∧ Prime w ∧ p = z - w) : p = 5 :=
sorry

end unique_prime_sum_diff_l127_127150


namespace greatest_odd_factors_l127_127496

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l127_127496


namespace nonnegative_values_ineq_l127_127108

theorem nonnegative_values_ineq {x : ℝ} : 
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Iic 3 := 
sorry

end nonnegative_values_ineq_l127_127108


namespace g_26_equals_125_l127_127548

noncomputable def g : ℕ → ℕ := sorry

axiom g_property : ∀ x, g (x + g x) = 5 * g x
axiom g_initial : g 1 = 5

theorem g_26_equals_125 : g 26 = 125 :=
by
  sorry

end g_26_equals_125_l127_127548


namespace trig_identity_30deg_l127_127627

theorem trig_identity_30deg :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  let c30 := Real.cos (Real.pi / 6)
  t30 = (Real.sqrt 3) / 3 ∧ s30 = 1 / 2 ∧ c30 = (Real.sqrt 3) / 2 →
  t30 + 4 * s30 + 2 * c30 = (2 * (Real.sqrt 3) + 3) / 3 := 
by
  intros
  sorry

end trig_identity_30deg_l127_127627


namespace sum_of_integers_ending_in_2_between_100_and_600_l127_127435

theorem sum_of_integers_ending_in_2_between_100_and_600 :
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  ∃ S : ℤ, S = n * (a + l) / 2 ∧ S = 17350 := 
by
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  use n * (a + l) / 2
  sorry

end sum_of_integers_ending_in_2_between_100_and_600_l127_127435


namespace trigonometric_comparison_l127_127716

noncomputable def a : ℝ := 2 * Real.sin (13 * Real.pi / 180) * Real.cos (13 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.tan (76 * Real.pi / 180) / (1 + Real.tan (76 * Real.pi / 180)^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem trigonometric_comparison : b > a ∧ a > c := by
  sorry

end trigonometric_comparison_l127_127716


namespace value_of_f_of_g_l127_127112

def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := x^2 - 9

theorem value_of_f_of_g : f (g 3) = 4 :=
by
  -- The proof would go here. Since we are only defining the statement, we can leave this as 'sorry'.
  sorry

end value_of_f_of_g_l127_127112


namespace sin_three_pi_four_minus_alpha_l127_127134

theorem sin_three_pi_four_minus_alpha 
  (α : ℝ) 
  (h₁ : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 :=
by
  sorry

end sin_three_pi_four_minus_alpha_l127_127134


namespace find_constants_l127_127403

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ -2 → 
  (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) = 
  A / (x - 1) + B / (x - 4) + C / (x + 2)) →
  A = 4 / 9 ∧ B = 28 / 9 ∧ C = -1 / 3 :=
by
  sorry

end find_constants_l127_127403


namespace solve_for_m_l127_127308

theorem solve_for_m (m x : ℝ) (h1 : 3 * m - 2 * x = 6) (h2 : x = 3) : m = 4 := by
  sorry

end solve_for_m_l127_127308


namespace solution_set_l127_127553

def within_bounds (x : ℝ) : Prop := |2 * x + 1| < 1

theorem solution_set : {x : ℝ | within_bounds x} = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end solution_set_l127_127553


namespace total_fencing_cost_l127_127475

theorem total_fencing_cost
  (length : ℝ) 
  (breadth : ℝ)
  (cost_per_meter : ℝ)
  (h1 : length = 61)
  (h2 : length = breadth + 22)
  (h3 : cost_per_meter = 26.50) : 
  2 * (length + breadth) * cost_per_meter = 5300 := 
by 
  sorry

end total_fencing_cost_l127_127475


namespace all_plants_diseased_l127_127081

theorem all_plants_diseased (n : ℕ) (h : n = 1007) : 
  n * 2 = 2014 := by
  sorry

end all_plants_diseased_l127_127081


namespace max_distinct_tangent_counts_l127_127031

-- Define the types and conditions for our circles and tangents
structure Circle where
  radius : ℝ

def circle1 : Circle := { radius := 3 }
def circle2 : Circle := { radius := 4 }

-- Define the statement to be proved
theorem max_distinct_tangent_counts :
  ∃ (k : ℕ), k = 5 :=
sorry

end max_distinct_tangent_counts_l127_127031


namespace transformed_polynomial_l127_127161

noncomputable def P : Polynomial ℝ := Polynomial.C 9 + Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 

noncomputable def Q : Polynomial ℝ := Polynomial.C 243 + Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 

theorem transformed_polynomial :
  ∀ (r : ℝ), Polynomial.aeval r P = 0 → Polynomial.aeval (3 * r) Q = 0 := 
by
  sorry

end transformed_polynomial_l127_127161


namespace speed_of_stream_l127_127873

theorem speed_of_stream (downstream_speed upstream_speed : ℕ) (h1 : downstream_speed = 12) (h2 : upstream_speed = 8) : 
  (downstream_speed - upstream_speed) / 2 = 2 :=
by
  sorry

end speed_of_stream_l127_127873


namespace brazil_medal_fraction_closest_l127_127986

theorem brazil_medal_fraction_closest :
  let frac_win : ℚ := 23 / 150
  let frac_1_6 : ℚ := 1 / 6
  let frac_1_7 : ℚ := 1 / 7
  let frac_1_8 : ℚ := 1 / 8
  let frac_1_9 : ℚ := 1 / 9
  let frac_1_10 : ℚ := 1 / 10
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_6) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_8) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_9) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_10) :=
by
  sorry

end brazil_medal_fraction_closest_l127_127986


namespace basketball_weight_l127_127677

variable {b c : ℝ}

theorem basketball_weight (h1 : 8 * b = 4 * c) (h2 : 3 * c = 120) : b = 20 :=
by
  -- Proof omitted
  sorry

end basketball_weight_l127_127677


namespace composite_evaluation_at_two_l127_127944

-- Define that P(x) is a polynomial with coefficients in {0, 1}
def is_binary_coefficient_polynomial (P : Polynomial ℤ) : Prop :=
  ∀ (n : ℕ), P.coeff n = 0 ∨ P.coeff n = 1

-- Define that P(x) can be factored into two nonconstant polynomials with integer coefficients
def is_reducible_to_nonconstant_polynomials (P : Polynomial ℤ) : Prop :=
  ∃ (f g : Polynomial ℤ), f.degree > 0 ∧ g.degree > 0 ∧ P = f * g

theorem composite_evaluation_at_two {P : Polynomial ℤ}
  (h1 : is_binary_coefficient_polynomial P)
  (h2 : is_reducible_to_nonconstant_polynomials P) :
  ∃ (m n : ℤ), m > 1 ∧ n > 1 ∧ P.eval 2 = m * n := sorry

end composite_evaluation_at_two_l127_127944


namespace tim_surprises_combinations_l127_127231

theorem tim_surprises_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 120 :=
by
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  sorry

end tim_surprises_combinations_l127_127231


namespace avg_last_three_numbers_l127_127172

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end avg_last_three_numbers_l127_127172


namespace square_ratio_l127_127266

def area (side_length : ℝ) : ℝ := side_length^2

theorem square_ratio (x : ℝ) (x_pos : 0 < x) :
  let A := area x
  let B := area (3*x)
  let C := area (2*x)
  A / (B + C) = 1 / 13 :=
by
  sorry

end square_ratio_l127_127266


namespace complex_number_purely_imaginary_l127_127836

variable {m : ℝ}

theorem complex_number_purely_imaginary (h1 : 2 * m^2 + m - 1 = 0) (h2 : -m^2 - 3 * m - 2 ≠ 0) : m = 1/2 := by
  sorry

end complex_number_purely_imaginary_l127_127836


namespace adults_wearing_hats_l127_127503

theorem adults_wearing_hats (total_adults : ℕ) (percent_men : ℝ) (percent_men_hats : ℝ) 
  (percent_women_hats : ℝ) (num_hats : ℕ) 
  (h1 : total_adults = 3600) 
  (h2 : percent_men = 0.40) 
  (h3 : percent_men_hats = 0.15) 
  (h4 : percent_women_hats = 0.25) 
  (h5 : num_hats = 756) : 
  (percent_men * total_adults) * percent_men_hats + (total_adults - (percent_men * total_adults)) * percent_women_hats = num_hats := 
sorry

end adults_wearing_hats_l127_127503


namespace truncated_cone_volume_l127_127457

noncomputable def volume_of_truncated_cone (R r h : ℝ) : ℝ :=
  let V_large := (1 / 3) * Real.pi * R^2 * (h + h)  -- Height of larger cone is h + x = h + h
  let V_small := (1 / 3) * Real.pi * r^2 * h       -- Height of smaller cone is h
  V_large - V_small

theorem truncated_cone_volume (R r h : ℝ) (hR : R = 8) (hr : r = 4) (hh : h = 6) :
  volume_of_truncated_cone R r h = 224 * Real.pi :=
by
  sorry

end truncated_cone_volume_l127_127457


namespace man_speed_against_current_proof_l127_127834

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l127_127834


namespace smallest_sphere_radius_l127_127287

noncomputable def radius_smallest_sphere : ℝ := 2 * Real.sqrt 3 + 2

theorem smallest_sphere_radius (r : ℝ) (h : r = 2) : radius_smallest_sphere = 2 * Real.sqrt 3 + 2 := by
  sorry

end smallest_sphere_radius_l127_127287


namespace problem1_problem2_problem3_l127_127175

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x + 2 * x

theorem problem1 (a : ℝ) : a = 1 → ∀ x : ℝ, f 1 x = x^2 - 3 * x + Real.log x → 
  (∀ x : ℝ, f 1 1 = -2) :=
by sorry

theorem problem2 (a : ℝ) (h : 0 < a) : (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → a ≥ 1 :=
by sorry

theorem problem3 (a : ℝ) : (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) → 0 ≤ a ∧ a ≤ 8 :=
by sorry

end problem1_problem2_problem3_l127_127175


namespace count_quadruples_l127_127923

open Real

theorem count_quadruples:
  ∃ qs : Finset (ℝ × ℝ × ℝ × ℝ),
  (∀ (a b c k : ℝ), (a, b, c, k) ∈ qs ↔ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a^k = b * c ∧
    b^k = c * a ∧
    c^k = a * b
  ) ∧
  qs.card = 8 :=
sorry

end count_quadruples_l127_127923


namespace fraction_to_decimal_l127_127975

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l127_127975


namespace quadratic_roots_ratio_l127_127531

theorem quadratic_roots_ratio {m n p : ℤ} (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : p ≠ 0)
  (h₃ : ∃ r1 r2 : ℤ, r1 * r2 = m ∧ n = 9 * r1 * r2 ∧ p = -(r1 + r2) ∧ m = -3 * (r1 + r2)) :
  n / p = -27 := by
  sorry

end quadratic_roots_ratio_l127_127531


namespace measure_of_angle_F_l127_127302

theorem measure_of_angle_F (D E F : ℝ) (h₁ : D = 85) (h₂ : E = 4 * F + 15) (h₃ : D + E + F = 180) : 
  F = 16 :=
by
  sorry

end measure_of_angle_F_l127_127302


namespace linda_savings_l127_127681

theorem linda_savings (S : ℝ) (h1 : 1 / 4 * S = 150) : S = 600 :=
sorry

end linda_savings_l127_127681


namespace largest_fraction_sum_l127_127480

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end largest_fraction_sum_l127_127480


namespace exercise_l127_127558

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end exercise_l127_127558


namespace smallest_set_of_circular_handshakes_l127_127876

def circular_handshake_smallest_set (n : ℕ) : ℕ :=
  if h : n % 2 = 0 then n / 2 else (n / 2) + 1

theorem smallest_set_of_circular_handshakes :
  circular_handshake_smallest_set 36 = 18 :=
by
  sorry

end smallest_set_of_circular_handshakes_l127_127876


namespace find_k_l127_127789

/- Definitions for vectors -/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/- Prove that if ka + b is perpendicular to a, then k = -1/5 -/
theorem find_k (k : ℝ) : 
  dot_product (k • (1, 2) + (-3, 2)) (1, 2) = 0 → 
  k = -1 / 5 := 
  sorry

end find_k_l127_127789


namespace correct_statements_count_l127_127120

/-
  Question: How many students have given correct interpretations of the algebraic expression \( 7x \)?
  Conditions:
    - Xiaoming's Statement: \( 7x \) can represent the sum of \( 7 \) and \( x \).
    - Xiaogang's Statement: \( 7x \) can represent the product of \( 7 \) and \( x \).
    - Xiaoliang's Statement: \( 7x \) can represent the total price of buying \( x \) pens at a unit price of \( 7 \) yuan.
  Given these conditions, prove that the number of correct statements is \( 2 \).
-/

theorem correct_statements_count (x : ℕ) :
  (if 7 * x = 7 + x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) = 2 := sorry

end correct_statements_count_l127_127120


namespace janet_dresses_total_pockets_l127_127567

theorem janet_dresses_total_pockets :
  ∃ dresses pockets pocket_2 pocket_3,
  dresses = 24 ∧ 
  pockets = dresses / 2 ∧ 
  pocket_2 = pockets / 3 ∧ 
  pocket_3 = pockets - pocket_2 ∧ 
  (pocket_2 * 2 + pocket_3 * 3) = 32 := by
    sorry

end janet_dresses_total_pockets_l127_127567


namespace good_students_l127_127192

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l127_127192


namespace total_amount_division_l127_127883

variables (w x y z : ℝ)

theorem total_amount_division (h_w : w = 2)
                              (h_x : x = 0.75)
                              (h_y : y = 1.25)
                              (h_z : z = 0.85)
                              (h_share_y : y * Rs48_50 = Rs48_50) :
                              total_amount = 4.85 * 38.80 := sorry

end total_amount_division_l127_127883


namespace beggars_society_votes_l127_127530

def total_voting_members (votes_for votes_against additional_against : ℕ) :=
  let majority := additional_against / 4
  let initial_difference := votes_for - votes_against
  let updated_against := votes_against + additional_against
  let updated_for := votes_for - additional_against
  updated_for + updated_against

theorem beggars_society_votes :
  total_voting_members 115 92 12 = 207 :=
by
  -- Proof goes here
  sorry

end beggars_society_votes_l127_127530


namespace maximize_probability_remove_6_l127_127761

-- Definitions
def integers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12] -- After removing 6
def initial_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Theorem Statement
theorem maximize_probability_remove_6 :
  ∀ (n : Int),
  n ∈ initial_list →
  n ≠ 6 →
  ∃ (a b : Int), a ∈ integers_list ∧ b ∈ integers_list ∧ a ≠ b ∧ a + b = 12 → False :=
by
  intros n hn hn6
  -- Placeholder for proof
  sorry

end maximize_probability_remove_6_l127_127761


namespace solve_for_x_l127_127915

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = (9 / (x / 3))^2) : x = 23.43 :=
by {
  sorry
}

end solve_for_x_l127_127915


namespace population_increase_l127_127468

theorem population_increase (P : ℝ) (h₁ : 11000 * (1 + P / 100) * (1 + P / 100) = 13310) : 
  P = 10 :=
sorry

end population_increase_l127_127468


namespace c_plus_d_l127_127335

theorem c_plus_d (a b c d : ℝ) (h1 : a + b = 11) (h2 : b + c = 9) (h3 : a + d = 5) :
  c + d = 3 + b :=
by
  sorry

end c_plus_d_l127_127335


namespace solve_for_x_l127_127694

theorem solve_for_x (x y : ℕ) (h₁ : 9 ^ y = 3 ^ x) (h₂ : y = 6) : x = 12 :=
by
  sorry

end solve_for_x_l127_127694


namespace negation_exists_x_squared_leq_abs_x_l127_127736

theorem negation_exists_x_squared_leq_abs_x :
  (¬ ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) ∧ x^2 ≤ |x|) ↔ (∀ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) → x^2 > |x|) :=
by
  sorry

end negation_exists_x_squared_leq_abs_x_l127_127736


namespace paving_stone_width_l127_127010

theorem paving_stone_width :
  ∀ (L₁ L₂ : ℝ) (n : ℕ) (length width : ℝ), 
    L₁ = 30 → L₂ = 16 → length = 2 → n = 240 →
    (L₁ * L₂ = n * (length * width)) → width = 1 :=
by
  sorry

end paving_stone_width_l127_127010


namespace range_of_cos_neg_alpha_l127_127405

theorem range_of_cos_neg_alpha (α : ℝ) (h : 12 * (Real.sin α)^2 + Real.cos α > 11) :
  -1 / 4 < Real.cos (-α) ∧ Real.cos (-α) < 1 / 3 := 
sorry

end range_of_cos_neg_alpha_l127_127405


namespace power_equivalence_l127_127778

theorem power_equivalence (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 :=
by sorry

end power_equivalence_l127_127778


namespace log_a1_plus_log_a9_l127_127981

variable {a : ℕ → ℝ}
variable {log : ℝ → ℝ}

-- Assume the provided conditions
axiom is_geometric_sequence : ∀ n, a (n + 1) / a n = a 1 / a 0
axiom a3a5a7_eq_one : a 3 * a 5 * a 7 = 1
axiom log_mul : ∀ x y, log (x * y) = log x + log y
axiom log_one_eq_zero : log 1 = 0

theorem log_a1_plus_log_a9 : log (a 1) + log (a 9) = 0 := 
by {
    sorry
}

end log_a1_plus_log_a9_l127_127981


namespace trig_identity_1_trig_identity_2_l127_127804

noncomputable def point := ℚ × ℚ

namespace TrigProblem

open Real

def point_on_terminal_side (α : ℝ) (p : point) : Prop :=
  let (x, y) := p
  ∃ r : ℝ, r = sqrt (x^2 + y^2) ∧ x/r = cos α ∧ y/r = sin α

theorem trig_identity_1 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  (sin (π / 2 + α) - cos (π + α)) / (sin (π / 2 - α) - sin (π - α)) = 8 / 7 :=
sorry

theorem trig_identity_2 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  sin α * cos α = -12 / 25 :=
sorry

end TrigProblem

end trig_identity_1_trig_identity_2_l127_127804


namespace sufficient_but_not_necessary_l127_127052

theorem sufficient_but_not_necessary (a : ℝ) : (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∀ x : ℝ, (x - 1) * (x - 2) = 0 → x ≠ 2 → x = 1) ∧
  (a = 2 → (1 ≠ 2)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l127_127052


namespace B_contribution_l127_127620

-- Define the conditions
def capitalA : ℝ := 3500
def monthsA : ℕ := 12
def monthsB : ℕ := 7
def profit_ratio_A : ℕ := 2
def profit_ratio_B : ℕ := 3

-- Statement: B's contribution to the capital
theorem B_contribution :
  (capitalA * monthsA * profit_ratio_B) / (monthsB * profit_ratio_A) = 4500 := by
  sorry

end B_contribution_l127_127620


namespace inequality_proof_l127_127978

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l127_127978


namespace tommy_total_balloons_l127_127811

-- Define the conditions from part (a)
def original_balloons : Nat := 26
def additional_balloons : Nat := 34

-- Define the proof problem from part (c)
theorem tommy_total_balloons : original_balloons + additional_balloons = 60 := by
  -- Skip the actual proof
  sorry

end tommy_total_balloons_l127_127811


namespace find_f8_l127_127951

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x * f y
axiom initial_condition : f 2 = 4

theorem find_f8 : f 8 = 256 := by
  sorry

end find_f8_l127_127951


namespace darry_steps_l127_127852

theorem darry_steps (f_steps : ℕ) (f_times : ℕ) (s_steps : ℕ) (s_times : ℕ) (no_other_steps : ℕ)
  (hf : f_steps = 11)
  (hf_times : f_times = 10)
  (hs : s_steps = 6)
  (hs_times : s_times = 7)
  (h_no_other : no_other_steps = 0) :
  (f_steps * f_times + s_steps * s_times + no_other_steps = 152) :=
by
  sorry

end darry_steps_l127_127852


namespace sum_non_prime_between_50_and_60_eq_383_l127_127684

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def non_primes_between_50_and_60 : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58]

def sum_non_primes_between_50_and_60 : ℕ :=
  non_primes_between_50_and_60.sum

theorem sum_non_prime_between_50_and_60_eq_383 :
  sum_non_primes_between_50_and_60 = 383 :=
by
  sorry

end sum_non_prime_between_50_and_60_eq_383_l127_127684


namespace function_decreasing_range_l127_127867

theorem function_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1) ≤ (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1)) ↔ (0 ≤ a ∧ a ≤ 1 / 3) :=
sorry

end function_decreasing_range_l127_127867


namespace triangle_even_number_in_each_row_from_third_l127_127035

/-- Each number in the (n+1)-th row of the triangle is the sum of three numbers 
  from the n-th row directly above this number and its immediate left and right neighbors.
  If such neighbors do not exist, they are considered as zeros.
  Prove that in each row of the triangle, starting from the third row,
  there is at least one even number. -/

theorem triangle_even_number_in_each_row_from_third (triangle : ℕ → ℕ → ℕ) :
  (∀ n i : ℕ, i > n → triangle n i = 0) →
  (∀ n i : ℕ, triangle (n+1) i = triangle n (i-1) + triangle n i + triangle n (i+1)) →
  ∀ n : ℕ, n ≥ 2 → ∃ i : ℕ, i ≤ n ∧ 2 ∣ triangle n i :=
by
  intros
  sorry

end triangle_even_number_in_each_row_from_third_l127_127035


namespace distinct_pairs_count_l127_127541

theorem distinct_pairs_count : 
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ x = x^2 + y^2 ∧ y = 3 * x * y) ∧ 
    S.card = 4 :=
by
  sorry

end distinct_pairs_count_l127_127541


namespace intersection_of_sets_l127_127780

open Set

theorem intersection_of_sets :
  let A := {x : ℤ | |x| < 3}
  let B := {x : ℤ | |x| > 1}
  A ∩ B = ({-2, 2} : Set ℤ) := by
  sorry

end intersection_of_sets_l127_127780


namespace quadratic_inequality_l127_127217

theorem quadratic_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end quadratic_inequality_l127_127217


namespace min_value_of_reciprocal_sum_l127_127527

theorem min_value_of_reciprocal_sum {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (hgeom : 3 = Real.sqrt (3^a * 3^b)) : (1 / a + 1 / b) = 2 :=
sorry  -- Proof not required, only the statement is needed.

end min_value_of_reciprocal_sum_l127_127527


namespace digits_sum_unique_l127_127656

variable (A B C D E F G H : ℕ)

theorem digits_sum_unique :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
  F ≠ G ∧ F ≠ H ∧
  G ≠ H ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  0 ≤ E ∧ E ≤ 9 ∧ 0 ≤ F ∧ F ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ H ∧ H ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D) + (E * 1000 + F * 100 + G * 10 + H) = 10652 ∧
  A = 9 ∧ B = 5 ∧ C = 6 ∧ D = 7 ∧
  E = 1 ∧ F = 0 ∧ G = 8 ∧ H = 5 :=
sorry

end digits_sum_unique_l127_127656


namespace num_girls_in_school_l127_127863

noncomputable def total_students : ℕ := 1600
noncomputable def sample_students : ℕ := 200
noncomputable def girls_less_than_boys_in_sample : ℕ := 10

-- Equations from conditions
def boys_in_sample (B G : ℕ) : Prop := G = B - girls_less_than_boys_in_sample
def sample_size (B G : ℕ) : Prop := B + G = sample_students

-- Proportion condition
def proportional_condition (G G_total : ℕ) : Prop := G * total_students = G_total * sample_students

-- Total number of girls in the school
def total_girls_in_school (G_total : ℕ) : Prop := G_total = 760

theorem num_girls_in_school :
  ∃ B G G_total : ℕ, boys_in_sample B G ∧ sample_size B G ∧ proportional_condition G G_total ∧ total_girls_in_school G_total :=
sorry

end num_girls_in_school_l127_127863


namespace n_c_equation_l127_127495

theorem n_c_equation (n c : ℕ) (hn : 0 < n) (hc : 0 < c) :
  (∀ x : ℕ, (↑x + n * ↑x / 100) * (1 - c / 100) = x) →
  (n^2 / c^2 = (100 + n) / (100 - c)) :=
by sorry

end n_c_equation_l127_127495


namespace inheritance_amount_l127_127497

theorem inheritance_amount (x : ℝ) (h1 : x * 0.25 + (x * 0.75) * 0.15 + 2500 = 16500) : x = 38621 := 
by
  sorry

end inheritance_amount_l127_127497


namespace combined_rent_C_D_l127_127450

theorem combined_rent_C_D :
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  combined_rent = 1020 :=
by
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  show combined_rent = 1020
  sorry

end combined_rent_C_D_l127_127450


namespace sandwich_total_calories_l127_127638

-- Given conditions
def bacon_calories := 2 * 125
def bacon_percentage := 20 / 100

-- Statement to prove
theorem sandwich_total_calories :
  bacon_calories / bacon_percentage = 1250 := 
sorry

end sandwich_total_calories_l127_127638


namespace rectangle_area_l127_127080

theorem rectangle_area 
  (P : ℝ) (r : ℝ) (hP : P = 40) (hr : r = 3 / 2) : 
  ∃ (length width : ℝ), 2 * (length + width) = P ∧ length = 3 * (width / 2) ∧ (length * width) = 96 :=
by
  sorry

end rectangle_area_l127_127080


namespace radius_base_circle_of_cone_l127_127815

theorem radius_base_circle_of_cone 
  (θ : ℝ) (R : ℝ) (arc_length : ℝ) (r : ℝ)
  (h1 : θ = 120) 
  (h2 : R = 9)
  (h3 : arc_length = (θ / 360) * 2 * Real.pi * R)
  (h4 : 2 * Real.pi * r = arc_length)
  : r = 3 := 
sorry

end radius_base_circle_of_cone_l127_127815


namespace bus_speed_l127_127764

theorem bus_speed (distance time : ℝ) (h_distance : distance = 201) (h_time : time = 3) : 
  distance / time = 67 :=
by
  sorry

end bus_speed_l127_127764


namespace simplify_and_evaluate_l127_127385

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2) :
  ( ( (2 * x - 1) / (x + 1) - x + 1 ) / (x - 2) / (x^2 + 2 * x + 1) ) = -2 - Real.sqrt 2 :=
by sorry

end simplify_and_evaluate_l127_127385


namespace probability_x_lt_2y_l127_127314

noncomputable def rectangle := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }

noncomputable def region_of_interest := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 2 * p.2 }

noncomputable def area_rectangle := 6 * 2

noncomputable def area_trapezoid := (1 / 2) * (4 + 6) * 2

theorem probability_x_lt_2y : (area_trapezoid / area_rectangle) = 5 / 6 :=
by
  -- skip the proof
  sorry

end probability_x_lt_2y_l127_127314


namespace cat_catches_mouse_l127_127688

-- Define the distances
def AB := 200
def BC := 140
def CD := 20

-- Define the speeds (in meters per minute)
def mouse_speed := 60
def cat_speed := 80

-- Define the total distances the mouse and cat travel
def mouse_total_distance := 320 -- The mouse path is along a zigzag route initially specified in the problem
def cat_total_distance := AB + BC + CD -- 360 meters as calculated

-- Define the times they take to reach point D
def mouse_time := mouse_total_distance / mouse_speed -- 5.33 minutes
def cat_time := cat_total_distance / cat_speed -- 4.5 minutes

-- Proof problem statement
theorem cat_catches_mouse : cat_time < mouse_time := 
by
  sorry

end cat_catches_mouse_l127_127688


namespace cos_seven_pi_over_four_l127_127928

theorem cos_seven_pi_over_four : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_seven_pi_over_four_l127_127928


namespace total_amount_received_l127_127203

theorem total_amount_received (B : ℝ) (h1 : (1/3) * B = 36) : (2/3 * B) * 4 = 288 :=
by
  sorry

end total_amount_received_l127_127203


namespace distinct_integers_sum_l127_127391

theorem distinct_integers_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 357) : a + b + c + d = 28 :=
by
  sorry

end distinct_integers_sum_l127_127391


namespace cyclic_quadrilateral_ptolemy_l127_127644

theorem cyclic_quadrilateral_ptolemy 
  (a b c d : ℝ) 
  (h : a + b + c + d = Real.pi) :
  Real.sin (a + b) * Real.sin (b + c) = Real.sin a * Real.sin c + Real.sin b * Real.sin d :=
by
  sorry

end cyclic_quadrilateral_ptolemy_l127_127644


namespace find_k_l127_127940

noncomputable def is_perfect_square (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ a : ℝ, x^2 + 2*(k-1)*x + 64 = (x + a)^2

theorem find_k (k : ℝ) : is_perfect_square k ↔ (k = 9 ∨ k = -7) :=
sorry

end find_k_l127_127940


namespace inequality_sum_l127_127026

open Real
open BigOperators

theorem inequality_sum 
  (n : ℕ) 
  (h : n > 1) 
  (x : Fin n → ℝ)
  (hx1 : ∀ i, 0 < x i) 
  (hx2 : ∑ i, x i = 1) :
  ∑ i, x i / sqrt (1 - x i) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) :=
sorry

end inequality_sum_l127_127026


namespace correct_conclusions_l127_127904

variable (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0)

def f (x : ℝ) : ℝ := x^2

theorem correct_conclusions (h_distinct : x1 ≠ x2) :
  (f x1 * x2 = f x1 * f x2) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) ∧
  (f ((x1 + x2) / 2) < (f x1 + f x2) / 2) :=
by
  sorry

end correct_conclusions_l127_127904


namespace blue_pens_count_l127_127073

variable (x y : ℕ) -- Define x as the number of red pens and y as the number of blue pens.
variable (h1 : 5 * x + 7 * y = 102) -- Condition 1: Total cost equation.
variable (h2 : x + y = 16) -- Condition 2: Total number of pens equation.

theorem blue_pens_count : y = 11 :=
by
  sorry

end blue_pens_count_l127_127073


namespace dartboard_central_angle_l127_127545

-- Define the conditions
variables {A : ℝ} {x : ℝ}

-- State the theorem
theorem dartboard_central_angle (h₁ : A > 0) (h₂ : (1/4 : ℝ) = ((x / 360) * A) / A) : x = 90 := 
by sorry

end dartboard_central_angle_l127_127545


namespace max_profit_at_80_l127_127869

-- Definitions based on conditions
def cost_price : ℝ := 40
def functional_relationship (x : ℝ) : ℝ := -x + 140
def profit (x : ℝ) : ℝ := (x - cost_price) * functional_relationship x

-- Statement to prove that maximum profit is achieved at x = 80
theorem max_profit_at_80 : (40 ≤ 80) → (80 ≤ 80) → profit 80 = 2400 := by
  sorry

end max_profit_at_80_l127_127869


namespace percentage_increase_decrease_l127_127152

theorem percentage_increase_decrease (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq100 : q < 100) :
  (M * (1 + p / 100) * (1 - q / 100) = 1.1 * M) ↔ (p = (10 + 100 * q) / (100 - q)) :=
by 
  sorry

end percentage_increase_decrease_l127_127152


namespace y1_lt_y2_l127_127795

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l127_127795


namespace president_and_committee_l127_127121

def combinatorial (n k : ℕ) : ℕ := Nat.choose n k

theorem president_and_committee :
  let num_people := 10
  let num_president := 1
  let num_committee := 3
  let num_ways_president := 10
  let num_remaining_people := num_people - num_president
  let num_ways_committee := combinatorial num_remaining_people num_committee
  num_ways_president * num_ways_committee = 840 := 
by
  sorry

end president_and_committee_l127_127121


namespace range_of_x_l127_127499

theorem range_of_x (x : ℝ) (h : 2 * x + 1 ≤ 0) : x ≤ -1 / 2 := 
  sorry

end range_of_x_l127_127499


namespace credit_card_more_beneficial_l127_127959

def gift_cost : ℝ := 8000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.0075
def debit_card_interest_rate : ℝ := 0.005

def credit_card_total_income : ℝ := gift_cost * (credit_card_cashback_rate + debit_card_interest_rate)
def debit_card_total_income : ℝ := gift_cost * debit_card_cashback_rate

theorem credit_card_more_beneficial :
  credit_card_total_income > debit_card_total_income :=
by
  sorry

end credit_card_more_beneficial_l127_127959


namespace simplify_expression_l127_127485

theorem simplify_expression (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end simplify_expression_l127_127485


namespace total_crayons_l127_127584

-- Definitions for the conditions
def crayons_per_child : Nat := 12
def number_of_children : Nat := 18

-- The statement to be proved
theorem total_crayons :
  (crayons_per_child * number_of_children = 216) := 
by
  sorry

end total_crayons_l127_127584


namespace contains_all_integers_l127_127992

def is_closed_under_divisors (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, b ∣ a → a ∈ A → b ∈ A

def contains_product_plus_one (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, 1 < a → a < b → a ∈ A → b ∈ A → (1 + a * b) ∈ A

theorem contains_all_integers
  (A : Set ℕ)
  (h1 : is_closed_under_divisors A)
  (h2 : contains_product_plus_one A)
  (h3 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ 1 < a ∧ 1 < b ∧ 1 < c) :
  ∀ n : ℕ, n > 0 → n ∈ A := 
  by 
    sorry

end contains_all_integers_l127_127992


namespace geometric_sequence_sum_l127_127250

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end geometric_sequence_sum_l127_127250


namespace range_of_m_l127_127900

-- Define the set A and condition
def A (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2 * x + m = 0 }

-- The theorem stating the range of m
theorem range_of_m (m : ℝ) : (A m = ∅) ↔ m > 1 :=
by
  sorry

end range_of_m_l127_127900


namespace problem_a_problem_c_l127_127817

variable {a b : ℝ}

theorem problem_a (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : ab ≤ 1 / 8 :=
by
  sorry

theorem problem_c (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : 1 / a + 2 / b ≥ 9 :=
by
  sorry

end problem_a_problem_c_l127_127817


namespace mixed_sum_proof_l127_127587

def mixed_sum : ℚ :=
  3 + 1/3 + 4 + 1/2 + 5 + 1/5 + 6 + 1/6

def smallest_whole_number_greater_than_mixed_sum : ℤ :=
  Int.ceil (mixed_sum)

theorem mixed_sum_proof :
  smallest_whole_number_greater_than_mixed_sum = 20 := by
  sorry

end mixed_sum_proof_l127_127587


namespace find_k_series_sum_l127_127570

theorem find_k_series_sum (k : ℝ) :
  (2 + ∑' n : ℕ, (2 + (n + 1) * k) / 2 ^ (n + 1)) = 6 -> k = 1 :=
by 
  sorry

end find_k_series_sum_l127_127570


namespace min_socks_no_conditions_l127_127667

theorem min_socks_no_conditions (m n : Nat) (h : (m * (m - 1) = 2 * (m + n) * (m + n - 1))) : 
  m + n ≥ 4 := sorry

end min_socks_no_conditions_l127_127667


namespace jake_work_hours_l127_127243

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end jake_work_hours_l127_127243


namespace candle_height_problem_l127_127441

-- Define the conditions given in the problem
def same_initial_height (height : ℝ := 1) := height = 1

def burn_rate_first_candle := 1 / 5

def burn_rate_second_candle := 1 / 4

def height_first_candle (t : ℝ) := 1 - (burn_rate_first_candle * t)

def height_second_candle (t : ℝ) := 1 - (burn_rate_second_candle * t)

-- Define the proof problem
theorem candle_height_problem : ∃ t : ℝ, height_first_candle t = 3 * height_second_candle t ∧ t = 40 / 11 :=
by
  sorry

end candle_height_problem_l127_127441


namespace perimeter_of_intersection_triangle_l127_127249

theorem perimeter_of_intersection_triangle :
  ∀ (P Q R : Type) (dist : P → Q → ℝ) (length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR : ℝ),
  (length_PQ = 150) →
  (length_QR = 250) →
  (length_PR = 200) →
  (seg_ellP = 75) →
  (seg_ellQ = 50) →
  (seg_ellR = 25) →
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  TU + US + ST = 266.67 :=
by
  intros P Q R dist length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR hPQ hQR hPR hP hQ hR
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  have : TU + US + ST = 266.67 := sorry
  exact this

end perimeter_of_intersection_triangle_l127_127249


namespace problem_trip_l127_127218

noncomputable def validate_trip (a b c : ℕ) (t : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10 ∧ 60 * t = 9 * c - 10 * b

theorem problem_trip (a b c t : ℕ) (h : validate_trip a b c t) : a^2 + b^2 + c^2 = 26 :=
sorry

end problem_trip_l127_127218


namespace problem_solution_l127_127406

theorem problem_solution :
  (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 :=
by sorry

end problem_solution_l127_127406


namespace domain_of_f_l127_127138

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 5)) + (1 / (x^2 - 4)) + (1 / (x^3 - 27))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 ↔
          ∃ y : ℝ, f y = f x :=
by
  sorry

end domain_of_f_l127_127138


namespace players_either_left_handed_or_throwers_l127_127180

theorem players_either_left_handed_or_throwers (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 34) (h3 : ∀ n, n = total_players - throwers → 1 / 3 * n = n / 3) :
  ∃ n, n = 46 := 
sorry

end players_either_left_handed_or_throwers_l127_127180


namespace pure_imaginary_number_solution_l127_127092

-- Definition of the problem
theorem pure_imaginary_number_solution (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a^2 - 3 * a + 2 ≠ 0) : a = -2 :=
sorry

end pure_imaginary_number_solution_l127_127092


namespace find_constants_l127_127101

variable (x : ℝ)

/-- Restate the equation problem and the constants A, B, C, D to be found. -/
theorem find_constants 
  (A B C D : ℝ)
  (h : ∀ x, x^3 - 7 = A * (x - 3) * (x - 5) * (x - 7) + B * (x - 2) * (x - 5) * (x - 7) + C * (x - 2) * (x - 3) * (x - 7) + D * (x - 2) * (x - 3) * (x - 5)) :
  A = 1/15 ∧ B = 5/2 ∧ C = -59/6 ∧ D = 42/5 :=
  sorry

end find_constants_l127_127101


namespace geometric_series_problem_l127_127473

theorem geometric_series_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ)
  (h_seq : ∀ n, a n + a (n + 1) = 3 * 2^n) :
  S (k + 2) - 2 * S (k + 1) + S k = 2^(k + 1) :=
sorry

end geometric_series_problem_l127_127473


namespace cube_edge_percentage_growth_l127_127432

theorem cube_edge_percentage_growth (p : ℝ) 
  (h : (1 + p / 100) ^ 2 - 1 = 0.96) : p = 40 :=
by
  sorry

end cube_edge_percentage_growth_l127_127432


namespace number_of_members_l127_127027

theorem number_of_members
  (headband_cost : ℕ := 3)
  (jersey_cost : ℕ := 10)
  (total_cost : ℕ := 2700)
  (cost_per_member : ℕ := 26) :
  total_cost / cost_per_member = 103 := by
  sorry

end number_of_members_l127_127027


namespace multiplier_condition_l127_127769

theorem multiplier_condition (a b : ℚ) (h : a * b ≤ b) : (b ≥ 0 ∧ a ≤ 1) ∨ (b ≤ 0 ∧ a ≥ 1) :=
by 
  sorry

end multiplier_condition_l127_127769


namespace problem1_problem2_problem3_problem4_l127_127654

-- Problem 1
theorem problem1 : (- (3 : ℝ) / 7) + (1 / 5) + (2 / 7) + (- (6 / 5)) = - (8 / 7) :=
by
  sorry

-- Problem 2
theorem problem2 : -(-1) + 3^2 / (1 - 4) * 2 = -5 :=
by
  sorry

-- Problem 3
theorem problem3 :  (-(1 / 6))^2 / ((1 / 2 - 1 / 3)^2) / (abs (-6))^2 = 1 / 36 :=
by
  sorry

-- Problem 4
theorem problem4 : (-1) ^ 1000 - 2.45 * 8 + 2.55 * (-8) = -39 :=
by
  sorry

end problem1_problem2_problem3_problem4_l127_127654


namespace find_theta_l127_127660

theorem find_theta (θ : ℝ) :
  (0 : ℝ) ≤ θ ∧ θ ≤ 2 * Real.pi →
  (∀ x, (0 : ℝ) ≤ x ∧ x ≤ 2 →
    x^2 * Real.cos θ - 2 * x * (1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intros hθ hx
  sorry

end find_theta_l127_127660


namespace remaining_days_temperature_l127_127053

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end remaining_days_temperature_l127_127053


namespace remainder_9876543210_mod_101_l127_127235

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l127_127235


namespace percentage_decrease_to_gain_30_percent_profit_l127_127676

theorem percentage_decrease_to_gain_30_percent_profit
  (C : ℝ) (P : ℝ) (S : ℝ) (S_new : ℝ) 
  (C_eq : C = 60)
  (S_eq : S = 1.25 * C)
  (S_new_eq1 : S_new = S - 12.60)
  (S_new_eq2 : S_new = 1.30 * (C - P * C)) : 
  P = 0.20 := by
  sorry

end percentage_decrease_to_gain_30_percent_profit_l127_127676


namespace average_daily_net_income_correct_l127_127344

-- Define the income, tips, and expenses for each day.
def day1_income := 300
def day1_tips := 50
def day1_expenses := 80

def day2_income := 150
def day2_tips := 20
def day2_expenses := 40

def day3_income := 750
def day3_tips := 100
def day3_expenses := 150

def day4_income := 200
def day4_tips := 30
def day4_expenses := 50

def day5_income := 600
def day5_tips := 70
def day5_expenses := 120

-- Define the net income for each day as income + tips - expenses.
def day1_net_income := day1_income + day1_tips - day1_expenses
def day2_net_income := day2_income + day2_tips - day2_expenses
def day3_net_income := day3_income + day3_tips - day3_expenses
def day4_net_income := day4_income + day4_tips - day4_expenses
def day5_net_income := day5_income + day5_tips - day5_expenses

-- Calculate the total net income over the 5 days.
def total_net_income := 
  day1_net_income + day2_net_income + day3_net_income + day4_net_income + day5_net_income

-- Define the number of days.
def number_of_days := 5

-- Calculate the average daily net income.
def average_daily_net_income := total_net_income / number_of_days

-- Statement to prove that the average daily net income is $366.
theorem average_daily_net_income_correct :
  average_daily_net_income = 366 := by
  sorry

end average_daily_net_income_correct_l127_127344


namespace find_intersection_l127_127164

noncomputable def f (n : ℕ) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def f_set (s : Set ℕ) : Set ℕ := {n | f n ∈ s}

theorem find_intersection : f_set A ∩ f_set B = {1, 2} := 
by {
  sorry
}

end find_intersection_l127_127164


namespace matt_needs_38_plates_l127_127996

def plates_needed (days_with_only_matt_and_son days_with_parents plates_per_day plates_per_person_with_parents : ℕ) : ℕ :=
  (days_with_only_matt_and_son * plates_per_day) + (days_with_parents * 4 * plates_per_person_with_parents)

theorem matt_needs_38_plates :
  plates_needed 3 4 2 2 = 38 :=
by
  sorry

end matt_needs_38_plates_l127_127996


namespace average_annual_percent_change_l127_127327

-- Define the initial and final population, and the time period
def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def decade_years : ℕ := 10

-- Define the theorem to find the resulting average percent change per year
theorem average_annual_percent_change
    (P₀ : ℕ := initial_population)
    (P₁₀ : ℕ := final_population)
    (years : ℕ := decade_years) :
    ((P₁₀ - P₀ : ℝ) / P₀ * 100) / years = 7 := by
        sorry

end average_annual_percent_change_l127_127327


namespace average_score_of_class_l127_127063

theorem average_score_of_class : 
  ∀ (total_students assigned_students make_up_students : ℕ)
    (assigned_avg_score make_up_avg_score : ℚ),
    total_students = 100 →
    assigned_students = 70 →
    make_up_students = total_students - assigned_students →
    assigned_avg_score = 60 →
    make_up_avg_score = 80 →
    (assigned_students * assigned_avg_score + make_up_students * make_up_avg_score) / total_students = 66 :=
by
  intro total_students assigned_students make_up_students assigned_avg_score make_up_avg_score
  intros h_total_students h_assigned_students h_make_up_students h_assigned_avg_score h_make_up_avg_score
  sorry

end average_score_of_class_l127_127063


namespace percentage_temporary_employees_is_correct_l127_127241

noncomputable def percentage_temporary_employees
    (technicians_percentage : ℝ) (skilled_laborers_percentage : ℝ) (unskilled_laborers_percentage : ℝ)
    (permanent_technicians_percentage : ℝ) (permanent_skilled_laborers_percentage : ℝ)
    (permanent_unskilled_laborers_percentage : ℝ) : ℝ :=
  let total_workers : ℝ := 100
  let total_temporary_technicians := technicians_percentage * (1 - permanent_technicians_percentage / 100)
  let total_temporary_skilled_laborers := skilled_laborers_percentage * (1 - permanent_skilled_laborers_percentage / 100)
  let total_temporary_unskilled_laborers := unskilled_laborers_percentage * (1 - permanent_unskilled_laborers_percentage / 100)
  let total_temporary_workers := total_temporary_technicians + total_temporary_skilled_laborers + total_temporary_unskilled_laborers
  (total_temporary_workers / total_workers) * 100

theorem percentage_temporary_employees_is_correct :
  percentage_temporary_employees 40 35 25 60 45 35 = 51.5 :=
by
  sorry

end percentage_temporary_employees_is_correct_l127_127241


namespace clara_gave_10_stickers_l127_127103

-- Defining the conditions
def initial_stickers : ℕ := 100
def remaining_after_boy (B : ℕ) : ℕ := initial_stickers - B
def remaining_after_friends (B : ℕ) : ℕ := (remaining_after_boy B) / 2

-- Theorem stating that Clara gave 10 stickers to the boy
theorem clara_gave_10_stickers (B : ℕ) (h : remaining_after_friends B = 45) : B = 10 :=
by
  sorry

end clara_gave_10_stickers_l127_127103


namespace vector_BC_l127_127746

/-- Given points A (0,1), B (3,2) and vector AC (-4,-3), prove that BC = (-7, -4) -/
theorem vector_BC
  (A B : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (hA : A = (0, 1))
  (hB : B = (3, 2))
  (hAC : AC = (-4, -3)) :
  (AC - (B - A)) = (-7, -4) :=
by
  sorry

end vector_BC_l127_127746


namespace range_of_a_l127_127816

noncomputable def f (a x : ℝ) : ℝ :=
if h : a ≤ x ∧ x < 0 then -((1/2)^x)
else if h' : 0 ≤ x ∧ x ≤ 4 then -(x^2) + 2*x
else 0

theorem range_of_a (a : ℝ) (h : ∀ x, f a x ∈ Set.Icc (-8 : ℝ) (1 : ℝ)) : 
  a ∈ Set.Ico (-3 : ℝ) 0 :=
sorry

end range_of_a_l127_127816


namespace parallel_to_a_perpendicular_to_a_l127_127512

-- Definition of vectors a and b and conditions
def a : ℝ × ℝ := (3, 4)
def b (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Mathematical statement for Problem (1)
theorem parallel_to_a (x y : ℝ) (h : b x y) (h_parallel : 3 * y - 4 * x = 0) :
  (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) := 
sorry

-- Mathematical statement for Problem (2)
theorem perpendicular_to_a (x y : ℝ) (h : b x y) (h_perpendicular : 3 * x + 4 * y = 0) :
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) := 
sorry

end parallel_to_a_perpendicular_to_a_l127_127512


namespace triangle_has_at_most_one_obtuse_angle_l127_127340

-- Definitions
def Triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def Obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

def Two_obtuse_angles (α β γ : ℝ) : Prop :=
  Obtuse_angle α ∧ Obtuse_angle β

-- Theorem Statement
theorem triangle_has_at_most_one_obtuse_angle (α β γ : ℝ) (h_triangle : Triangle α β γ) :
  ¬ Two_obtuse_angles α β γ := 
sorry

end triangle_has_at_most_one_obtuse_angle_l127_127340


namespace interval_where_f_increasing_l127_127246

noncomputable def f (x : ℝ) : ℝ := Real.log (4 * x - x^2) / Real.log (1 / 2)

theorem interval_where_f_increasing : ∀ x : ℝ, 2 ≤ x ∧ x < 4 → f x < f (x + 1) :=
by 
  sorry

end interval_where_f_increasing_l127_127246


namespace units_digit_of_fraction_l127_127987

theorem units_digit_of_fraction :
  ((30 * 31 * 32 * 33 * 34) / 400) % 10 = 4 :=
by
  sorry

end units_digit_of_fraction_l127_127987


namespace angles_same_terminal_side_l127_127945

def angle_equiv (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angles_same_terminal_side : angle_equiv (-390 : ℝ) (330 : ℝ) :=
sorry

end angles_same_terminal_side_l127_127945


namespace product_sum_divisibility_l127_127617

theorem product_sum_divisibility (m n : ℕ) (h : (m + n) ∣ (m * n)) (hm : 0 < m) (hn : 0 < n) : m + n ≤ n^2 :=
sorry

end product_sum_divisibility_l127_127617


namespace brooke_earns_144_dollars_l127_127932

-- Definitions based on the identified conditions
def price_of_milk_per_gallon : ℝ := 3
def production_cost_per_gallon_of_butter : ℝ := 0.5
def sticks_of_butter_per_gallon : ℝ := 2
def price_of_butter_per_stick : ℝ := 1.5
def number_of_cows : ℕ := 12
def milk_per_cow : ℝ := 4
def number_of_customers : ℕ := 6
def min_milk_per_customer : ℝ := 4
def max_milk_per_customer : ℝ := 8

-- Auxiliary calculations
def total_milk_produced : ℝ := number_of_cows * milk_per_cow
def min_total_customer_demand : ℝ := number_of_customers * min_milk_per_customer
def max_total_customer_demand : ℝ := number_of_customers * max_milk_per_customer

-- Problem statement
theorem brooke_earns_144_dollars :
  (0 <= total_milk_produced) ∧
  (min_total_customer_demand <= max_total_customer_demand) ∧
  (total_milk_produced = max_total_customer_demand) →
  (total_milk_produced * price_of_milk_per_gallon = 144) :=
by
  -- Sorry is added here since the proof is not required
  sorry

end brooke_earns_144_dollars_l127_127932


namespace jellybean_probability_l127_127575

theorem jellybean_probability :
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 2
  let yellow_jellybeans := 5
  let total_picks := 4
  let successful_outcomes := 10 * 7 
  let total_outcomes := Nat.choose 12 4 
  let required_probability := 14 / 99 
  successful_outcomes = 70 ∧ total_outcomes = 495 → 
  successful_outcomes / total_outcomes = required_probability := 
by 
  intros
  sorry

end jellybean_probability_l127_127575


namespace exists_unique_line_prime_x_intercept_positive_y_intercept_l127_127304

/-- There is exactly one line with x-intercept that is a prime number less than 10 and y-intercept that is a positive integer not equal to 5, which passes through the point (5, 4) -/
theorem exists_unique_line_prime_x_intercept_positive_y_intercept (x_intercept : ℕ) (hx : Nat.Prime x_intercept) (hx_lt_10 : x_intercept < 10) (y_intercept : ℕ) (hy_pos : y_intercept > 0) (hy_ne_5 : y_intercept ≠ 5) :
  (∃ (a b : ℕ), a = x_intercept ∧ b = y_intercept ∧ (∀ p q : ℕ, p = 5 ∧ q = 4 → (p / a) + (q / b) = 1)) :=
sorry

end exists_unique_line_prime_x_intercept_positive_y_intercept_l127_127304


namespace domain_ln_l127_127597

theorem domain_ln (x : ℝ) : (1 - 2 * x > 0) ↔ x < (1 / 2) :=
by
  sorry

end domain_ln_l127_127597


namespace probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l127_127580

-- Conditions
def red_ball_probability := 1 / 2
def yellow_ball_probability := 1 / 2
def num_draws := 3

-- Define the events and their probabilities
def prob_three_red : ℚ := red_ball_probability ^ num_draws
def prob_three_same : ℚ := 2 * (red_ball_probability ^ num_draws)
def prob_not_all_same : ℚ := 1 - prob_three_same / 2

-- Lean statements
theorem probability_three_red_balls : prob_three_red = 1 / 8 :=
by
  sorry

theorem probability_three_same_color_balls : prob_three_same = 1 / 4 :=
by
  sorry

theorem probability_not_all_same_color_balls : prob_not_all_same = 3 / 4 :=
by
  sorry

end probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l127_127580


namespace solveAdultsMonday_l127_127902

def numAdultsMonday (A : ℕ) : Prop :=
  let childrenMondayCost := 7 * 3
  let childrenTuesdayCost := 4 * 3
  let adultsTuesdayCost := 2 * 4
  let totalChildrenCost := childrenMondayCost + childrenTuesdayCost
  let totalAdultsCost := A * 4 + adultsTuesdayCost
  let totalRevenue := totalChildrenCost + totalAdultsCost
  totalRevenue = 61

theorem solveAdultsMonday : numAdultsMonday 5 := 
  by 
    -- Proof goes here
    sorry

end solveAdultsMonday_l127_127902


namespace vector_dot_product_l127_127417

open Complex

def a : Complex := (1 : ℝ) + (-(2 : ℝ)) * Complex.I
def b : Complex := (-3 : ℝ) + (4 : ℝ) * Complex.I
def c : Complex := (3 : ℝ) + (2 : ℝ) * Complex.I

-- Note: Using real coordinates to simulate vector operations.
theorem vector_dot_product :
  let a_vec := (1, -2)
  let b_vec := (-3, 4)
  let c_vec := (3, 2)
  let linear_combination := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (linear_combination.1 * c_vec.1 + linear_combination.2 * c_vec.2) = -3 := 
by
  sorry

end vector_dot_product_l127_127417


namespace solution_l127_127521

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions:
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (-x + 1) = f (x + 1)) ∧ f (-1) = 1 :=
sorry

theorem solution : f 2017 = -1 :=
sorry

end solution_l127_127521


namespace inequality_proof_l127_127551

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a * b * c := 
by {
  sorry
}

end inequality_proof_l127_127551


namespace squirrel_can_catch_nut_l127_127610

-- Define the initial distance between Gabriel and the squirrel.
def initial_distance : ℝ := 3.75

-- Define the speed of the nut.
def nut_speed : ℝ := 5.0

-- Define the jumping distance of the squirrel.
def squirrel_jump_distance : ℝ := 1.8

-- Define the acceleration due to gravity.
def gravity : ℝ := 10.0

-- Define the positions of the nut and the squirrel as functions of time.
def nut_position_x (t : ℝ) : ℝ := nut_speed * t
def squirrel_position_x : ℝ := initial_distance
def nut_position_y (t : ℝ) : ℝ := 0.5 * gravity * t^2

-- Define the squared distance between the nut and the squirrel.
def distance_squared (t : ℝ) : ℝ :=
  (nut_position_x t - squirrel_position_x)^2 + (nut_position_y t)^2

-- Prove that the minimum distance squared is less than or equal to the squirrel's jumping distance squared.
theorem squirrel_can_catch_nut : ∃ t : ℝ, distance_squared t ≤ squirrel_jump_distance^2 := by
  -- Sorry placeholder, as the proof is not required.
  sorry

end squirrel_can_catch_nut_l127_127610


namespace coord_relationship_M_l127_127268

theorem coord_relationship_M (x y z : ℝ) (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, -1)) (hB : B = (2, 0, 2))
  (hM : ∃ M : ℝ × ℝ × ℝ, M = (x, y, z) ∧ y = 0 ∧ |(1 - x)^2 + 2^2 + (-1 - z)^2| = |(2 - x)^2 + (0 - z)^2|) :
  x + 3 * z - 1 = 0 ∧ y = 0 := 
sorry

end coord_relationship_M_l127_127268


namespace parallelogram_base_length_l127_127750

theorem parallelogram_base_length (A h : ℕ) (hA : A = 32) (hh : h = 8) : (A / h) = 4 := by
  sorry

end parallelogram_base_length_l127_127750


namespace car_speed_second_hour_l127_127136

variable (x : ℝ)
variable (s1 : ℝ := 100)
variable (avg_speed : ℝ := 90)
variable (total_time : ℝ := 2)

-- The Lean statement equivalent to the problem
theorem car_speed_second_hour : (100 + x) / 2 = 90 → x = 80 := by 
  intro h
  have h₁ : 2 * 90 = 100 + x := by 
    linarith [h]
  linarith [h₁]

end car_speed_second_hour_l127_127136


namespace sum_of_plane_angles_l127_127440

theorem sum_of_plane_angles (v f p : ℕ) (h : v = p) :
    (2 * π * (v - f) = 2 * π * (p - 2)) :=
by sorry

end sum_of_plane_angles_l127_127440


namespace number_of_people_in_tour_l127_127167

theorem number_of_people_in_tour (x : ℕ) : 
  (x ≤ 25 ∧ 100 * x = 2700 ∨ 
  (x > 25 ∧ 
   (100 - 2 * (x - 25)) * x = 2700 ∧ 
   70 ≤ 100 - 2 * (x - 25))) → 
  x = 30 := 
by
  sorry

end number_of_people_in_tour_l127_127167


namespace trajectory_equation_l127_127009

theorem trajectory_equation : ∀ (x y : ℝ),
  (x + 3)^2 + y^2 + (x - 3)^2 + y^2 = 38 → x^2 + y^2 = 10 :=
by
  intros x y h
  sorry

end trajectory_equation_l127_127009


namespace triangle_right_triangle_l127_127365

theorem triangle_right_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 3) (h2 : b = 4) (h3 : c^2 - 10 * c + 25 = 0) : 
  a^2 + b^2 = c^2 :=
by
  -- We know the values of a, b, and c by the conditions
  sorry

end triangle_right_triangle_l127_127365


namespace binom_30_3_l127_127017

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l127_127017


namespace factor_check_l127_127454

theorem factor_check :
  ∃ (f : ℕ → ℕ) (x : ℝ), f 1 = (x^2 - 2 * x + 3) ∧ f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 :=
by
  let f : ℕ → ℕ := sorry -- Define a sequence or function for the proof context
  let x : ℝ := sorry -- Define the variable x in our context
  have h₁ : f 1 = (x^2 - 2 * x + 3) := sorry -- Establish the first factor
  have h₂ : f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 := sorry -- Establish the polynomial expression
  exact ⟨f, x, h₁, h₂⟩ -- Use existential quantifier to capture the required form

end factor_check_l127_127454


namespace max_value_expr_l127_127897

theorem max_value_expr (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxyz : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (x - y + z) ≤ 2187 / 216 :=
sorry

end max_value_expr_l127_127897


namespace complement_union_correct_l127_127135

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {2, 3, 4})
variable (hB : B = {1, 4})

theorem complement_union_correct :
  (compl A ∪ B) = {1, 4, 5} :=
by
  sorry

end complement_union_correct_l127_127135


namespace age_ratio_holds_l127_127233

variables (e s : ℕ)

-- Conditions based on the problem statement
def condition_1 : Prop := e - 3 = 2 * (s - 3)
def condition_2 : Prop := e - 5 = 3 * (s - 5)

-- Proposition to prove that in 1 year, the age ratio will be 3:2
def age_ratio_in_one_year : Prop := (e + 1) * 2 = (s + 1) * 3

theorem age_ratio_holds (h1 : condition_1 e s) (h2 : condition_2 e s) : age_ratio_in_one_year e s :=
by {
  sorry
}

end age_ratio_holds_l127_127233


namespace fractions_ordered_l127_127670

theorem fractions_ordered :
  (2 / 5 : ℚ) < (3 / 5) ∧ (3 / 5) < (4 / 6) ∧ (4 / 6) < (4 / 5) ∧ (4 / 5) < (6 / 5) ∧ (6 / 5) < (4 / 3) :=
by
  sorry

end fractions_ordered_l127_127670


namespace largest_w_l127_127283

variable {x y z w : ℝ}

def x_value (x y z w : ℝ) := 
  x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4

theorem largest_w (h : x_value x y z w) : 
  max x (max y (max z w)) = w := 
sorry

end largest_w_l127_127283


namespace frosting_cans_needed_l127_127160

theorem frosting_cans_needed :
  let daily_cakes := 10
  let days := 5
  let total_cakes := daily_cakes * days
  let eaten_cakes := 12
  let remaining_cakes := total_cakes - eaten_cakes
  let cans_per_cake := 2
  let total_cans := remaining_cakes * cans_per_cake
  total_cans = 76 := 
by
  sorry

end frosting_cans_needed_l127_127160


namespace minimum_boxes_required_l127_127598

theorem minimum_boxes_required 
  (total_brochures : ℕ)
  (small_box_capacity : ℕ) (small_boxes_available : ℕ)
  (medium_box_capacity : ℕ) (medium_boxes_available : ℕ)
  (large_box_capacity : ℕ) (large_boxes_available : ℕ)
  (complete_fill : ∀ (box_capacity brochures : ℕ), box_capacity ∣ brochures)
  (min_boxes_required : ℕ) :
  total_brochures = 10000 →
  small_box_capacity = 50 →
  small_boxes_available = 40 →
  medium_box_capacity = 200 →
  medium_boxes_available = 25 →
  large_box_capacity = 500 →
  large_boxes_available = 10 →
  min_boxes_required = 35 :=
by
  intros
  sorry

end minimum_boxes_required_l127_127598


namespace max_brownies_l127_127522

theorem max_brownies (m n : ℕ) (h : (m - 2) * (n - 2) = 2 * m + 2 * n - 4) : m * n ≤ 60 :=
sorry

end max_brownies_l127_127522


namespace possible_values_of_a_l127_127315

theorem possible_values_of_a (a b c : ℝ) (h1 : a + b + c = 2005) (h2 : (a - 1 = a ∨ a - 1 = b ∨ a - 1 = c) ∧ (b + 1 = a ∨ b + 1 = b ∨ b + 1 = c) ∧ (c ^ 2 = a ∨ c ^ 2 = b ∨ c ^ 2 = c)) :
  a = 1003 ∨ a = 1002.5 :=
sorry

end possible_values_of_a_l127_127315


namespace percentage_increase_correct_l127_127828

def bookstore_earnings : ℕ := 60
def tutoring_earnings : ℕ := 40
def new_bookstore_earnings : ℕ := 100
def additional_tutoring_fee : ℕ := 15
def old_total_earnings : ℕ := bookstore_earnings + tutoring_earnings
def new_total_earnings : ℕ := new_bookstore_earnings + (tutoring_earnings + additional_tutoring_fee)
def overall_percentage_increase : ℚ := (((new_total_earnings - old_total_earnings : ℚ) / old_total_earnings) * 100)

theorem percentage_increase_correct :
  overall_percentage_increase = 55 := sorry

end percentage_increase_correct_l127_127828


namespace least_red_chips_l127_127075

/--
  There are 70 chips in a box. Each chip is either red or blue.
  If the sum of the number of red chips and twice the number of blue chips equals a prime number,
  proving that the least possible number of red chips is 69.
-/
theorem least_red_chips (r b : ℕ) (p : ℕ) (h1 : r + b = 70) (h2 : r + 2 * b = p) (hp : Nat.Prime p) :
  r = 69 :=
by
  -- Proof goes here
  sorry

end least_red_chips_l127_127075


namespace distinct_arrangements_ballon_l127_127127

theorem distinct_arrangements_ballon : 
  let n := 6
  let repetitions := 2
  n! / repetitions! = 360 :=
by
  sorry

end distinct_arrangements_ballon_l127_127127


namespace fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l127_127349

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Condition 1: a = 1, b = 5; the fixed points are x = -1 or x = -4
theorem fixed_points_a_one_b_five : 
  ∀ x : ℝ, is_fixed_point (f 1 5) x ↔ x = -1 ∨ x = -4 := by
  -- Proof goes here
  sorry

-- Condition 2: For any real b, f(x) always having two distinct fixed points implies 0 < a < 1
theorem range_of_a_two_distinct_fixed_points : 
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) ↔ 0 < a ∧ a < 1 := by
  -- Proof goes here
  sorry

end fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l127_127349


namespace sum_of_squares_and_products_l127_127602

theorem sum_of_squares_and_products
  (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 := 
by
  sorry

end sum_of_squares_and_products_l127_127602


namespace range_of_a_l127_127387

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l127_127387


namespace definite_integral_ln_l127_127625

open Real

theorem definite_integral_ln (a b : ℝ) (h₁ : a = 1) (h₂ : b = exp 1) :
  ∫ x in a..b, (1 + log x) = exp 1 := by
  sorry

end definite_integral_ln_l127_127625


namespace expansion_contains_x4_l127_127603

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def expansion_term (x : ℂ) (i : ℂ) : ℂ :=
  binomial_coeff 6 2 * x^4 * i^2

theorem expansion_contains_x4 (x i : ℂ) (hi : i = Complex.I) : 
  expansion_term x i = -15 * x^4 := by
  sorry

end expansion_contains_x4_l127_127603


namespace marcus_dropped_8_pies_l127_127926

-- Step d): Rewrite as a Lean 4 statement
-- Define all conditions from the problem
def total_pies (pies_per_batch : ℕ) (batches : ℕ) : ℕ :=
  pies_per_batch * batches

def pies_dropped (total_pies : ℕ) (remaining_pies : ℕ) : ℕ :=
  total_pies - remaining_pies

-- Prove that Marcus dropped 8 pies
theorem marcus_dropped_8_pies : 
  total_pies 5 7 - 27 = 8 := by
  sorry

end marcus_dropped_8_pies_l127_127926


namespace area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l127_127847

-- Defining the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intersection point P of line1 and line2
def P : ℝ × ℝ := (-2, 2)

-- Perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Line l, passing through P and perpendicular to perpendicular_line
def line_l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intercepts of line_l with axes
def x_intercept : ℝ := -1
def y_intercept : ℝ := -2

-- Verifying area of the triangle formed by the intercepts
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

#check line1
#check line2
#check P
#check perpendicular_line
#check line_l
#check x_intercept
#check y_intercept
#check area_of_triangle

theorem area_of_triangle_formed_by_line_l_and_axes :
  ∀ (x : ℝ) (y : ℝ),
    line_l x 0 → line_l 0 y →
    area_of_triangle (abs x) (abs y) = 1 :=
by
  intros x y hx hy
  sorry

theorem equation_of_line_l :
  ∀ (x y : ℝ),
    (line1 x y ∧ line2 x y) →
    (perpendicular_line x y) →
    line_l x y :=
by
  intros x y h1 h2
  sorry

end area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l127_127847


namespace crackers_per_person_l127_127618

variable (darrenA : Nat)
variable (darrenB : Nat)
variable (aCrackersPerBox : Nat)
variable (bCrackersPerBox : Nat)
variable (calvinA : Nat)
variable (calvinB : Nat)
variable (totalPeople : Nat)

-- Definitions based on the conditions
def totalDarrenCrackers := darrenA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCalvinA := 2 * darrenA - 1
def totalCalvinCrackers := totalCalvinA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCrackers := totalDarrenCrackers + totalCalvinCrackers
def crackersPerPerson := totalCrackers / totalPeople

-- The theorem to prove the question equals the answer given the conditions
theorem crackers_per_person :
  darrenA = 4 →
  darrenB = 2 →
  aCrackersPerBox = 24 →
  bCrackersPerBox = 30 →
  calvinA = 7 →
  calvinB = darrenB →
  totalPeople = 5 →
  crackersPerPerson = 76 :=
by
  intros
  sorry

end crackers_per_person_l127_127618


namespace fraction_product_l127_127962

theorem fraction_product :
  (7 / 4) * (8 / 14) * (16 / 24) * (32 / 48) * (28 / 7) * (15 / 9) *
  (50 / 25) * (21 / 35) = 32 / 3 :=
by
  sorry

end fraction_product_l127_127962


namespace function_inequality_l127_127462

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x : ℝ, x ≥ 1 → f x ≤ x)
  (h2 : ∀ x : ℝ, x ≥ 1 → f (2 * x) / Real.sqrt 2 ≤ f x) :
  ∀ x ≥ 1, f x < Real.sqrt (2 * x) :=
sorry

end function_inequality_l127_127462


namespace solve_for_x_l127_127622

theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)⁻¹) : x = 8 := 
by 
  sorry

end solve_for_x_l127_127622


namespace trajectory_of_M_l127_127858

theorem trajectory_of_M (M : ℝ × ℝ) (h : (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2)) :
  (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2) :=
by
  sorry

end trajectory_of_M_l127_127858


namespace inequality_proof_l127_127909

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l127_127909


namespace scientist_prob_rain_l127_127036

theorem scientist_prob_rain (x : ℝ) (p0 p1 : ℝ)
  (h0 : p0 + p1 = 1)
  (h1 : ∀ x : ℝ, x = (p0 * x^2 + p0 * (1 - x) * x + p1 * (1 - x) * x) / x + (1 - x) - x^2 / (x + 1))
  (h2 : (x + p0 / (x + 1) - x^2 / (x + 1)) = 0.2) :
  x = 1/9 := 
sorry

end scientist_prob_rain_l127_127036


namespace trains_crossing_time_l127_127664

-- Definitions based on conditions
def train_length : ℕ := 120
def time_train1_cross_pole : ℕ := 10
def time_train2_cross_pole : ℕ := 15

-- Question reformulated as a proof goal
theorem trains_crossing_time :
  let v1 := train_length / time_train1_cross_pole  -- Speed of train 1
  let v2 := train_length / time_train2_cross_pole  -- Speed of train 2
  let relative_speed := v1 + v2                    -- Relative speed in opposite directions
  let total_distance := train_length + train_length -- Sum of both trains' lengths
  let time_to_cross := total_distance / relative_speed -- Time to cross each other
  time_to_cross = 12 := 
by
  -- The proof here is stated, but not needed in this task
  -- All necessary computation steps
  sorry

end trains_crossing_time_l127_127664


namespace aziz_age_l127_127325

-- Definitions of the conditions
def year_moved : ℕ := 1982
def years_before_birth : ℕ := 3
def current_year : ℕ := 2021

-- Prove the main statement
theorem aziz_age : current_year - (year_moved + years_before_birth) = 36 :=
by
  sorry

end aziz_age_l127_127325


namespace profit_function_maximize_profit_l127_127007

def cost_per_item : ℝ := 80
def purchase_quantity : ℝ := 1000
def selling_price_initial : ℝ := 100
def price_increase_per_item : ℝ := 1
def sales_decrease_per_yuan : ℝ := 10
def selling_price (x : ℕ) : ℝ := selling_price_initial + x
def profit (x : ℕ) : ℝ := (selling_price x - cost_per_item) * (purchase_quantity - sales_decrease_per_yuan * x)

theorem profit_function (x : ℕ) (h : 0 ≤ x ∧ x ≤ 100) : 
  profit x = -10 * (x : ℝ)^2 + 800 * (x : ℝ) + 20000 :=
by sorry

theorem maximize_profit :
  ∃ max_x, (0 ≤ max_x ∧ max_x ≤ 100) ∧ 
  (∀ x : ℕ, (0 ≤ x ∧ x ≤ 100) → profit x ≤ profit max_x) ∧ 
  max_x = 40 ∧ 
  profit max_x = 36000 :=
by sorry

end profit_function_maximize_profit_l127_127007


namespace divisibility_problem_l127_127105

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end divisibility_problem_l127_127105


namespace largest_consecutive_even_integer_l127_127245

theorem largest_consecutive_even_integer (n : ℕ) (h : 5 * n - 20 = 2 * 15 * 16 / 2) : n = 52 :=
sorry

end largest_consecutive_even_integer_l127_127245


namespace _l127_127236

lemma triangle_inequality_theorem (a b c : ℝ) : 
  a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a > 0 ∧ b > 0 ∧ c > 0) := sorry

lemma no_triangle_1_2_3 : ¬ (1 + 2 > 3 ∧ 1 + 3 > 2 ∧ 2 + 3 > 1) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_3_8_5 : ¬ (3 + 8 > 5 ∧ 3 + 5 > 8 ∧ 8 + 5 > 3) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_4_5_10 : ¬ (4 + 5 > 10 ∧ 4 + 10 > 5 ∧ 5 + 10 > 4) := 
by simp [triangle_inequality_theorem]

lemma triangle_4_5_6 : 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4 := 
by simp [triangle_inequality_theorem]

end _l127_127236


namespace red_marble_count_l127_127479

theorem red_marble_count (x y : ℕ) (total_yellow : ℕ) (total_diff : ℕ) 
  (jar1_ratio_red jar1_ratio_yellow : ℕ) (jar2_ratio_red jar2_ratio_yellow : ℕ) 
  (h1 : jar1_ratio_red = 7) (h2 : jar1_ratio_yellow = 2) 
  (h3 : jar2_ratio_red = 5) (h4 : jar2_ratio_yellow = 3) 
  (h5 : 2 * x + 3 * y = 50) (h6 : 8 * y = 9 * x + 20) :
  7 * x + 2 = 5 * y :=
sorry

end red_marble_count_l127_127479


namespace pieces_after_10_cuts_l127_127891

-- Define the number of cuts
def cuts : ℕ := 10

-- Define the function that calculates the number of pieces
def pieces (k : ℕ) : ℕ := k + 1

-- State the theorem to prove the number of pieces given 10 cuts
theorem pieces_after_10_cuts : pieces cuts = 11 :=
by
  -- Proof goes here
  sorry

end pieces_after_10_cuts_l127_127891


namespace workers_problem_l127_127043

theorem workers_problem (W : ℕ) (A : ℕ) :
  (W * 45 = A) ∧ ((W + 10) * 35 = A) → W = 35 :=
by
  sorry

end workers_problem_l127_127043


namespace bookmarks_per_day_l127_127977

theorem bookmarks_per_day (pages_now : ℕ) (pages_end_march : ℕ) (days_in_march : ℕ) (pages_added : ℕ) (pages_per_day : ℕ)
  (h1 : pages_now = 400)
  (h2 : pages_end_march = 1330)
  (h3 : days_in_march = 31)
  (h4 : pages_added = pages_end_march - pages_now)
  (h5 : pages_per_day = pages_added / days_in_march) :
  pages_per_day = 30 := sorry

end bookmarks_per_day_l127_127977


namespace simplify_and_evaluate_expr_l127_127400

theorem simplify_and_evaluate_expr (x : ℤ) (h : x = -2) : 
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -8 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l127_127400


namespace total_votes_election_l127_127227

theorem total_votes_election 
  (votes_A : ℝ) 
  (valid_votes_percentage : ℝ) 
  (invalid_votes_percentage : ℝ)
  (votes_candidate_A : ℝ) 
  (total_votes : ℝ) 
  (h1 : votes_A = 0.60) 
  (h2 : invalid_votes_percentage = 0.15) 
  (h3 : votes_candidate_A = 285600) 
  (h4 : valid_votes_percentage = 0.85) 
  (h5 : total_votes = 560000) 
  : 
  ((votes_A * valid_votes_percentage * total_votes) = votes_candidate_A) 
  := 
  by sorry

end total_votes_election_l127_127227


namespace tan_double_angle_third_quadrant_l127_127123

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : sin (π - α) = - (3 / 5)) : 
  tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_third_quadrant_l127_127123


namespace negation_of_exists_real_solution_equiv_l127_127535

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l127_127535


namespace exists_positive_integers_seq_l127_127754

def sum_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.sum

def prod_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.prod

theorem exists_positive_integers_seq (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n.succ → ℕ),
    (∀ i : Fin n, sum_of_digits (a i) < sum_of_digits (a i.succ)) ∧
    (∀ i : Fin n, sum_of_digits (a i) = prod_of_digits (a i.succ)) ∧
    (∀ i : Fin n, 0 < (a i)) :=
by
  sorry

end exists_positive_integers_seq_l127_127754


namespace intersection_points_form_rectangle_l127_127966

theorem intersection_points_form_rectangle
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 + y^2 = 34) :
  ∃ (a b u v : ℝ), (a * b = 8) ∧ (a^2 + b^2 = 34) ∧ 
  (u * v = 8) ∧ (u^2 + v^2 = 34) ∧
  ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧ 
  ((u = -x ∧ v = -y) ∨ (u = -y ∧ v = -x)) ∧
  ((a = u ∧ b = v) ∨ (a = v ∧ b = u)) ∧ 
  ((x = -u ∧ y = -v) ∨ (x = -v ∧ y = -u)) ∧
  (
    (a, b) ≠ (u, v) ∧ (a, b) ≠ (-u, -v) ∧ 
    (a, b) ≠ (v, u) ∧ (a, b) ≠ (-v, -u) ∧
    (u, v) ≠ (-a, -b) ∧ (u, v) ≠ (b, a) ∧ 
    (u, v) ≠ (-b, -a)
  ) :=
by sorry

end intersection_points_form_rectangle_l127_127966


namespace jared_current_age_condition_l127_127609

variable (t j: ℕ)

-- Conditions
def tom_current_age := 25
def tom_future_age_condition := t + 5 = 30
def jared_past_age_condition := j - 2 = 2 * (t - 2)

-- Question
theorem jared_current_age_condition : 
  (t + 5 = 30) ∧ (j - 2 = 2 * (t - 2)) → j = 48 :=
by
  sorry

end jared_current_age_condition_l127_127609


namespace line_tangent_to_circle_l127_127478

theorem line_tangent_to_circle {m : ℝ} : 
  (∀ x y : ℝ, y = m * x) → (∀ x y : ℝ, x^2 + y^2 - 4 * x + 2 = 0) → 
  (m = 1 ∨ m = -1) := 
by 
  sorry

end line_tangent_to_circle_l127_127478


namespace max_value_sqrt_abc_expression_l127_127222

theorem max_value_sqrt_abc_expression (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                                       (hb : 0 ≤ b) (hb1 : b ≤ 1)
                                       (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1) :=
sorry

end max_value_sqrt_abc_expression_l127_127222


namespace value_of_expression_l127_127309

variable {a : ℕ → ℤ}
variable {a₁ a₄ a₁₀ a₁₆ a₁₉ : ℤ}
variable {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + d * n

-- Given conditions
axiom h₀ : arithmetic_sequence a a₁ d
axiom h₁ : a₁ + a₄ + a₁₀ + a₁₆ + a₁₉ = 150

-- Prove the required statement
theorem value_of_expression :
  a 20 - a 26 + a 16 = 30 :=
sorry

end value_of_expression_l127_127309


namespace four_distinct_sum_equal_l127_127247

theorem four_distinct_sum_equal (S : Finset ℕ) (hS : S.card = 10) (hS_subset : S ⊆ Finset.range 38) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by
  sorry

end four_distinct_sum_equal_l127_127247


namespace total_balloons_l127_127178

theorem total_balloons (allan_balloons : ℕ) (jake_balloons : ℕ)
  (h_allan : allan_balloons = 2)
  (h_jake : jake_balloons = 1) :
  allan_balloons + jake_balloons = 3 :=
by 
  -- Provide proof here
  sorry

end total_balloons_l127_127178


namespace find_a5_l127_127519

variable {a_n : ℕ → ℤ}
variable (d : ℤ)

def arithmetic_sequence (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  a_n 1 = a1 ∧ ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 (h_seq : arithmetic_sequence a_n 6 d) (h_a3 : a_n 3 = 2) : a_n 5 = -2 :=
by
  obtain ⟨h_a1, h_arith⟩ := h_seq
  sorry

end find_a5_l127_127519


namespace smallest_four_digit_divisible_by_primes_l127_127005

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l127_127005


namespace part_a_part_b_l127_127781

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≥ 3) :
  ¬ (1/x + 1/y + 1/z ≤ 3) :=
sorry

theorem part_b (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end part_a_part_b_l127_127781


namespace no_all_perfect_squares_l127_127822

theorem no_all_perfect_squares (x : ℤ) 
  (h1 : ∃ a : ℤ, 2 * x - 1 = a^2) 
  (h2 : ∃ b : ℤ, 5 * x - 1 = b^2) 
  (h3 : ∃ c : ℤ, 13 * x - 1 = c^2) : 
  False :=
sorry

end no_all_perfect_squares_l127_127822


namespace closest_ratio_l127_127731

theorem closest_ratio
  (a_0 : ℝ)
  (h_pos : a_0 > 0)
  (a_10 : ℝ)
  (h_eq : a_10 = a_0 * (1 + 0.05) ^ 10) :
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.5) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.7) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.8) := 
sorry

end closest_ratio_l127_127731


namespace students_attend_Purum_Elementary_School_l127_127695
open Nat

theorem students_attend_Purum_Elementary_School (P N : ℕ) 
  (h1 : P + N = 41) (h2 : P = N + 3) : P = 22 :=
sorry

end students_attend_Purum_Elementary_School_l127_127695


namespace urea_moles_produced_l127_127994

-- Define the reaction
def chemical_reaction (CO2 NH3 Urea Water : ℕ) :=
  CO2 = 1 ∧ NH3 = 2 ∧ Urea = 1 ∧ Water = 1

-- Given initial moles of reactants
def initial_moles (CO2 NH3 : ℕ) :=
  CO2 = 1 ∧ NH3 = 2

-- The main theorem to prove
theorem urea_moles_produced (CO2 NH3 Urea Water : ℕ) :
  initial_moles CO2 NH3 → chemical_reaction CO2 NH3 Urea Water → Urea = 1 :=
by
  intro H1 H2
  rcases H1 with ⟨HCO2, HNH3⟩
  rcases H2 with ⟨HCO2', HNH3', HUrea, _⟩
  sorry

end urea_moles_produced_l127_127994


namespace max_height_reached_l127_127984

def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

theorem max_height_reached : ∃ (t : ℝ), h t = 41.25 :=
by
  use 1.25
  sorry

end max_height_reached_l127_127984


namespace find_abs_xyz_l127_127568

variables {x y z : ℝ}

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem find_abs_xyz
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : distinct x y z)
  (h3 : x + 1 / y = 2)
  (h4 : y + 1 / z = 2)
  (h5 : z + 1 / x = 2) :
  |x * y * z| = 1 :=
by sorry

end find_abs_xyz_l127_127568


namespace eval_expression_l127_127110

theorem eval_expression : ⌈- (7 / 3 : ℚ)⌉ + ⌊(7 / 3 : ℚ)⌋ = 0 := 
by 
  sorry

end eval_expression_l127_127110


namespace danny_wrappers_more_than_soda_cans_l127_127721

theorem danny_wrappers_more_than_soda_cans :
  (67 - 22 = 45) := sorry

end danny_wrappers_more_than_soda_cans_l127_127721


namespace gcd_polynomials_l127_127197

-- State the problem in Lean 4.
theorem gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * 2 * k) : 
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 :=
by
  sorry

end gcd_polynomials_l127_127197


namespace circles_intersect_l127_127338

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end circles_intersect_l127_127338


namespace expression_meaningful_l127_127173

theorem expression_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by
  sorry

end expression_meaningful_l127_127173


namespace correct_option_l127_127916

theorem correct_option :
  (∀ a : ℝ, a ≠ 0 → (a ^ 0 = 1)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → (a^6 / a^3 = a^2)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → ((a^2)^3 = a^5)) ∧
  ¬(∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / (a + b)^2 + b / (a + b)^2 = a + b)) :=
by {
  sorry
}

end correct_option_l127_127916


namespace max_principals_in_10_years_l127_127295

theorem max_principals_in_10_years : ∀ term_length num_years,
  (term_length = 4) ∧ (num_years = 10) →
  ∃ max_principals, max_principals = 3
:=
  by intros term_length num_years h
     sorry

end max_principals_in_10_years_l127_127295


namespace left_square_side_length_l127_127819

theorem left_square_side_length (x : ℕ) (h1 : x + (x + 17) + (x + 11) = 52) : x = 8 :=
sorry

end left_square_side_length_l127_127819


namespace inequality_proof_l127_127636

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) :=
by
  sorry

end inequality_proof_l127_127636


namespace proof_statements_l127_127782

theorem proof_statements (m : ℝ) (x y : ℝ)
  (h1 : 2 * x + y = 4 - m)
  (h2 : x - 2 * y = 3 * m) :
  (m = 1 → (x = 9 / 5 ∧ y = -3 / 5)) ∧
  (3 * x - y = 4 + 2 * m) ∧
  ¬(∃ (m' : ℝ), (8 + m') / 5 < 0 ∧ (4 - 7 * m') / 5 < 0) :=
sorry

end proof_statements_l127_127782


namespace sum_of_first_five_terms_l127_127960

noncomputable def S₅ (a : ℕ → ℝ) := (a 1 + a 5) / 2 * 5

theorem sum_of_first_five_terms (a : ℕ → ℝ) (a_2 a_4 : ℝ)
  (h1 : a 2 = 4)
  (h2 : a 4 = 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S₅ a = 15 :=
sorry

end sum_of_first_five_terms_l127_127960


namespace find_m_of_slope_is_12_l127_127934

theorem find_m_of_slope_is_12 (m : ℝ) :
  let A := (-m, 6)
  let B := (1, 3 * m)
  let slope := (3 * m - 6) / (1 + m)
  slope = 12 → m = -2 :=
by
  sorry

end find_m_of_slope_is_12_l127_127934


namespace distance_between_stations_l127_127270

theorem distance_between_stations 
  (distance_P_to_meeting : ℝ)
  (distance_Q_to_meeting : ℝ)
  (h1 : distance_P_to_meeting = 20 * 3)
  (h2 : distance_Q_to_meeting = 25 * 2)
  (h3 : distance_P_to_meeting + distance_Q_to_meeting = D) :
  D = 110 :=
by
  sorry

end distance_between_stations_l127_127270


namespace total_cost_of_repair_l127_127407

noncomputable def cost_of_repair (tire_cost: ℝ) (num_tires: ℕ) (tax: ℝ) (city_fee: ℝ) (discount: ℝ) : ℝ :=
  let total_cost := (tire_cost * num_tires : ℝ)
  let total_tax := (tax * num_tires : ℝ)
  let total_city_fee := (city_fee * num_tires : ℝ)
  (total_cost + total_tax + total_city_fee - discount)

def car_A_tire_cost : ℝ := 7
def car_A_num_tires : ℕ := 3
def car_A_tax : ℝ := 0.5
def car_A_city_fee : ℝ := 2.5
def car_A_discount : ℝ := (car_A_tire_cost * car_A_num_tires) * 0.05

def car_B_tire_cost : ℝ := 8.5
def car_B_num_tires : ℕ := 2
def car_B_tax : ℝ := 0 -- no sales tax
def car_B_city_fee : ℝ := 2.5
def car_B_discount : ℝ := 0 -- expired coupon

theorem total_cost_of_repair : 
  cost_of_repair car_A_tire_cost car_A_num_tires car_A_tax car_A_city_fee car_A_discount + 
  cost_of_repair car_B_tire_cost car_B_num_tires car_B_tax car_B_city_fee car_B_discount = 50.95 :=
by
  sorry

end total_cost_of_repair_l127_127407


namespace probability_calculation_l127_127509

noncomputable def probability_at_least_seven_at_least_three_times : ℚ :=
  let p := 1 / 4
  let q := 3 / 4
  (4 * p^3 * q) + (p^4)

theorem probability_calculation :
  probability_at_least_seven_at_least_three_times = 13 / 256 :=
by sorry

end probability_calculation_l127_127509


namespace min_red_chips_l127_127596

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ w / 3) 
  (h2 : b ≤ r / 4) 
  (h3 : w + b ≥ 72) :
  72 ≤ r :=
by
  sorry

end min_red_chips_l127_127596


namespace gino_initial_sticks_l127_127564

-- Definitions based on the conditions
def given_sticks : ℕ := 50
def remaining_sticks : ℕ := 13
def initial_sticks (x y : ℕ) : ℕ := x + y

-- Theorem statement based on the mathematically equivalent proof problem
theorem gino_initial_sticks :
  initial_sticks given_sticks remaining_sticks = 63 :=
by
  sorry

end gino_initial_sticks_l127_127564


namespace number_of_students_in_chemistry_class_l127_127171

variables (students : Finset ℕ) (n : ℕ)
  (x y z cb cp bp c b : ℕ)
  (students_in_total : students.card = 120)
  (chem_bio : cb = 35)
  (bio_phys : bp = 15)
  (chem_phys : cp = 10)
  (total_equation : 120 = x + y + z + cb + bp + cp)
  (chem_equation : c = y + cb + cp)
  (bio_equation : b = x + cb + bp)
  (chem_bio_relation : 4 * b = c)
  (no_all_three_classes : true)

theorem number_of_students_in_chemistry_class : c = 153 :=
  sorry

end number_of_students_in_chemistry_class_l127_127171


namespace symmetrical_polynomial_l127_127418

noncomputable def Q (x : ℝ) (f g h i j k : ℝ) : ℝ :=
  x^6 + f * x^5 + g * x^4 + h * x^3 + i * x^2 + j * x + k

theorem symmetrical_polynomial (f g h i j k : ℝ) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ Q 0 f g h i j k = 0 ∧
    Q x f g h i j k = x * (x - a) * (x + a) * (x - b) * (x + b) * (x - c) ∧
    Q x f g h i j k = Q (-x) f g h i j k) →
  f = 0 :=
by sorry

end symmetrical_polynomial_l127_127418


namespace ratio_of_areas_l127_127262

theorem ratio_of_areas (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : r₁ = (Real.sqrt 2) / 4)
  (h₂ : A₁ = π * r₁^2) (h₃ : r₂ = (Real.sqrt 2) * r₁) (h₄ : A₂ = π * r₂^2) :
  A₂ / A₁ = 2 :=
by
  sorry

end ratio_of_areas_l127_127262


namespace bus_driver_total_earnings_l127_127776

noncomputable def regular_rate : ℝ := 20
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours : ℝ := 45.714285714285715
noncomputable def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
noncomputable def overtime_hours : ℝ := total_hours - regular_hours
noncomputable def regular_pay : ℝ := regular_rate * regular_hours
noncomputable def overtime_pay : ℝ := overtime_rate * overtime_hours
noncomputable def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_total_earnings :
  total_compensation = 1000 :=
by
  sorry

end bus_driver_total_earnings_l127_127776


namespace probability_hits_10_ring_l127_127291

-- Definitions based on conditions
def total_shots : ℕ := 10
def hits_10_ring : ℕ := 2

-- Theorem stating the question and answer equivalence.
theorem probability_hits_10_ring : (hits_10_ring : ℚ) / total_shots = 0.2 := by
  -- We are skipping the proof with 'sorry'
  sorry

end probability_hits_10_ring_l127_127291


namespace common_ratio_of_geometric_sequence_l127_127426

variables (a : ℕ → ℝ) (q : ℝ)
axiom h1 : a 1 = 2
axiom h2 : ∀ n : ℕ, a (n + 1) - a n ≠ 0 -- Common difference is non-zero
axiom h3 : a 3 = (a 1) * q
axiom h4 : a 11 = (a 1) * q^2
axiom h5 : a 11 = a 1 + 5 * (a 3 - a 1)

theorem common_ratio_of_geometric_sequence : q = 4 := 
by sorry

end common_ratio_of_geometric_sequence_l127_127426


namespace find_m_l127_127212

-- Definitions based on conditions
def is_eccentricity (a b c : ℝ) (e : ℝ) : Prop :=
  e = c / a

def ellipse_relation (a b m : ℝ) : Prop :=
  a ^ 2 = 3 ∧ b ^ 2 = m

def eccentricity_square_relation (c a : ℝ) : Prop :=
  (c / a) ^ 2 = 1 / 4

-- Main theorem statement
theorem find_m (m : ℝ) :
  (∀ (a b c : ℝ), ellipse_relation a b m → is_eccentricity a b c (1 / 2) → eccentricity_square_relation c a)
  → (m = 9 / 4 ∨ m = 4) := sorry

end find_m_l127_127212


namespace trap_speed_independent_of_location_l127_127682

theorem trap_speed_independent_of_location 
  (h b a : ℝ) (v_mouse : ℝ) 
  (path_length : ℝ := Real.sqrt (a^2 + (3*h)^2)) 
  (T : ℝ := path_length / v_mouse) 
  (step_height : ℝ := h) 
  (v_trap : ℝ := step_height / T) 
  (h_val : h = 3) 
  (b_val : b = 1) 
  (a_val : a = 8) 
  (v_mouse_val : v_mouse = 17) : 
  v_trap = 8 := by
  sorry

end trap_speed_independent_of_location_l127_127682


namespace WangLei_is_13_l127_127797

-- We need to define the conditions and question in Lean 4
def WangLei_age (x : ℕ) : Prop :=
  3 * x - 8 = 31

theorem WangLei_is_13 : ∃ x : ℕ, WangLei_age x ∧ x = 13 :=
by
  use 13
  unfold WangLei_age
  sorry

end WangLei_is_13_l127_127797


namespace fourth_intersection_point_l127_127755

def intersect_curve_circle : Prop :=
  let curve_eq (x y : ℝ) : Prop := x * y = 1
  let circle_intersects_points (h k s : ℝ) : Prop :=
    ∃ (x1 y1 x2 y2 x3 y3 : ℝ), 
    (x1, y1) = (3, (1 : ℝ) / 3) ∧ 
    (x2, y2) = (-4, -(1 : ℝ) / 4) ∧ 
    (x3, y3) = ((1 : ℝ) / 6, 6) ∧ 
    (x1 - h)^2 + (y1 - k)^2 = s^2 ∧
    (x2 - h)^2 + (y2 - k)^2 = s^2 ∧
    (x3 - h)^2 + (y3 - k)^2 = s^2 
  let fourth_point_of_intersection (x y : ℝ) : Prop := 
    x = -(1 : ℝ) / 2 ∧ 
    y = -2
  curve_eq 3 ((1 : ℝ) / 3) ∧
  curve_eq (-4) (-(1 : ℝ) / 4) ∧
  curve_eq ((1 : ℝ) / 6) 6 ∧
  ∃ h k s, circle_intersects_points h k s →
  ∃ (x4 y4 : ℝ), curve_eq x4 y4 ∧
  fourth_point_of_intersection x4 y4

theorem fourth_intersection_point :
  intersect_curve_circle := by
  sorry

end fourth_intersection_point_l127_127755


namespace warriors_games_won_l127_127290

open Set

-- Define the variables for the number of games each team won
variables (games_L games_H games_W games_F games_R : ℕ)

-- Define the set of possible game scores
def game_scores : Set ℕ := {19, 23, 28, 32, 36}

-- Define the conditions as assumptions
axiom h1 : games_L > games_H
axiom h2 : games_W > games_F
axiom h3 : games_W < games_R
axiom h4 : games_F > 18
axiom h5 : ∃ min_games ∈ game_scores, min_games > games_H ∧ min_games < 20

-- Prove the main statement
theorem warriors_games_won : games_W = 32 :=
sorry

end warriors_games_won_l127_127290


namespace hyperbola_equation_l127_127518

-- Definition of the ellipse given in the problem
def ellipse (x y : ℝ) := y^2 / 5 + x^2 = 1

-- Definition of the conditions for the hyperbola:
-- 1. The hyperbola shares a common focus with the ellipse.
-- 2. Distance from the focus to the asymptote of the hyperbola is 1.
def hyperbola (x y : ℝ) (c : ℝ) :=
  ∃ a b : ℝ, c = 2 ∧ a^2 + b^2 = c^2 ∧
             (b = 1 ∧ y = if x = 0 then 0 else x * (a / b))

-- The statement we need to prove
theorem hyperbola_equation : 
  (∃ a b : ℝ, ellipse x y ∧ hyperbola x y 2 ∧ b = 1 ∧ a^2 = 3) → 
  (y^2 / 3 - x^2 = 1) :=
sorry

end hyperbola_equation_l127_127518


namespace base8_addition_l127_127704

theorem base8_addition : (234 : ℕ) + (157 : ℕ) = (4 * 8^2 + 1 * 8^1 + 3 * 8^0 : ℕ) :=
by sorry

end base8_addition_l127_127704


namespace line_through_P_perpendicular_l127_127281

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end line_through_P_perpendicular_l127_127281


namespace calculation_correct_l127_127013

def grid_coloring_probability : ℚ := 591 / 1024

theorem calculation_correct : (m + n = 1615) ↔ (∃ m n : ℕ, m + n = 1615 ∧ gcd m n = 1 ∧ grid_coloring_probability = m / n) := sorry

end calculation_correct_l127_127013


namespace number_of_cherry_pie_days_l127_127810

theorem number_of_cherry_pie_days (A C : ℕ) (h1 : A + C = 7) (h2 : 12 * A = 12 * C + 12) : C = 3 :=
sorry

end number_of_cherry_pie_days_l127_127810


namespace fraction_of_satisfactory_is_15_over_23_l127_127690

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 2
def num_students_with_grade_F : ℕ := 6

def num_satisfactory_students : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + num_students_with_grade_C

def total_students : ℕ := 
  num_satisfactory_students + num_students_with_grade_D + num_students_with_grade_F

def fraction_satisfactory : ℚ := 
  (num_satisfactory_students : ℚ) / (total_students : ℚ)

theorem fraction_of_satisfactory_is_15_over_23 : 
  fraction_satisfactory = 15/23 :=
by
  -- proof omitted
  sorry

end fraction_of_satisfactory_is_15_over_23_l127_127690


namespace probability_sum_18_two_12_sided_dice_l127_127727

theorem probability_sum_18_two_12_sided_dice :
  let total_outcomes := 12 * 12
  let successful_outcomes := 7
  successful_outcomes / total_outcomes = 7 / 144 := by
sorry

end probability_sum_18_two_12_sided_dice_l127_127727


namespace find_g1_l127_127982

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x : ℝ) (hx : x ≠ 1 / 2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x + 1

theorem find_g1 : g 1 = 39 / 11 :=
by
  sorry

end find_g1_l127_127982


namespace percentage_increase_book_price_l127_127621

theorem percentage_increase_book_price (OldP NewP : ℕ) (hOldP : OldP = 300) (hNewP : NewP = 330) :
  ((NewP - OldP : ℕ) / OldP : ℚ) * 100 = 10 := by
  sorry

end percentage_increase_book_price_l127_127621


namespace distance_between_parallel_lines_l127_127737

theorem distance_between_parallel_lines (r d : ℝ) :
  let c₁ := 36
  let c₂ := 36
  let c₃ := 40
  let expr1 := (324 : ℝ) + (1 / 4) * d^2
  let expr2 := (400 : ℝ) + d^2
  let radius_eq1 := r^2 = expr1
  let radius_eq2 := r^2 = expr2
  radius_eq1 ∧ radius_eq2 → d = Real.sqrt (304 / 3) :=
by
  sorry

end distance_between_parallel_lines_l127_127737


namespace evaluate_rr2_l127_127366

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := (x - 3) * (x - 2)

theorem evaluate_rr2 : r (r 2) = 6 :=
by
  -- proof goes here
  sorry

end evaluate_rr2_l127_127366


namespace inequality_solutions_l127_127865

theorem inequality_solutions (a : ℚ) :
  (∀ x : ℕ, 0 < x ∧ x ≤ 3 → 3 * (x - 1) < 2 * (x + a) - 5) →
  (∃ x : ℕ, 0 < x ∧ x = 4 → ¬ (3 * (x - 1) < 2 * (x + a) - 5)) →
  (5 / 2 < a ∧ a ≤ 3) :=
sorry

end inequality_solutions_l127_127865


namespace harmonic_mean_lcm_gcd_sum_l127_127921

theorem harmonic_mean_lcm_gcd_sum {m n : ℕ} (h_lcm : Nat.lcm m n = 210) (h_gcd : Nat.gcd m n = 6) (h_sum : m + n = 72) :
  (1 / (m : ℚ) + 1 / (n : ℚ)) = 2 / 35 := 
sorry

end harmonic_mean_lcm_gcd_sum_l127_127921


namespace mass_percentage_Na_in_NaClO_l127_127358

theorem mass_percentage_Na_in_NaClO :
  let mass_Na : ℝ := 22.99
  let mass_Cl : ℝ := 35.45
  let mass_O : ℝ := 16.00
  let mass_NaClO : ℝ := mass_Na + mass_Cl + mass_O
  (mass_Na / mass_NaClO) * 100 = 30.89 := by
sorry

end mass_percentage_Na_in_NaClO_l127_127358


namespace min_distinct_values_l127_127929

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) 
  (h_mode : mode_count = 10) (h_total : total_count = 2018) 
  (h_distinct : ∀ k, k ≠ mode_count → k < 10) : 
  n ≥ 225 :=
by
  sorry

end min_distinct_values_l127_127929


namespace find_f_neg_a_l127_127546

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end find_f_neg_a_l127_127546


namespace total_expenditure_correct_l127_127680

def length : ℝ := 50
def width : ℝ := 30
def cost_per_square_meter : ℝ := 100

def area (L W : ℝ) : ℝ := L * W
def total_expenditure (A C : ℝ) : ℝ := A * C

theorem total_expenditure_correct :
  total_expenditure (area length width) cost_per_square_meter = 150000 := by
  sorry

end total_expenditure_correct_l127_127680


namespace circle_eq_tangent_x_axis_l127_127505

theorem circle_eq_tangent_x_axis (h k r : ℝ) (x y : ℝ)
  (h_center : h = -5)
  (k_center : k = 4)
  (tangent_x_axis : r = 4) :
  (x + 5)^2 + (y - 4)^2 = 16 :=
sorry

end circle_eq_tangent_x_axis_l127_127505


namespace paint_needed_l127_127069

theorem paint_needed (wall_area : ℕ) (coverage_per_gallon : ℕ) (number_of_coats : ℕ) (h_wall_area : wall_area = 600) (h_coverage_per_gallon : coverage_per_gallon = 400) (h_number_of_coats : number_of_coats = 2) : 
    ((number_of_coats * wall_area) / coverage_per_gallon) = 3 :=
by
  sorry

end paint_needed_l127_127069


namespace polynomial_decomposition_l127_127649

-- Define the given polynomial
def P (x y z : ℝ) : ℝ := x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2

-- Define the target decomposition
def Q (x y z : ℝ) : ℝ := (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2

theorem polynomial_decomposition (x y z : ℝ) : P x y z = Q x y z :=
  sorry

end polynomial_decomposition_l127_127649


namespace missing_fraction_l127_127903

theorem missing_fraction (x : ℕ) (h1 : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 9 :=
by
  sorry

end missing_fraction_l127_127903


namespace upper_bound_y_l127_127392

theorem upper_bound_y 
  (U : ℤ) 
  (x y : ℤ)
  (h1 : 3 < x ∧ x < 6) 
  (h2 : 6 < y ∧ y < U) 
  (h3 : y - x = 4) : 
  U = 10 := 
sorry

end upper_bound_y_l127_127392


namespace smallest_range_of_sample_l127_127412

open Real

theorem smallest_range_of_sample {a b c d e f g : ℝ}
  (h1 : (a + b + c + d + e + f + g) / 7 = 8)
  (h2 : d = 10)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ f ∧ f ≤ g) :
  ∃ r, r = g - a ∧ r = 8 :=
by
  sorry

end smallest_range_of_sample_l127_127412


namespace sum_of_squares_and_cube_unique_l127_127793

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem sum_of_squares_and_cube_unique : 
  ∃! (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_cube c ∧ a + b + c = 100 :=
sorry

end sum_of_squares_and_cube_unique_l127_127793


namespace simplify_expression_l127_127240

theorem simplify_expression (x y : ℝ) : (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3 * y + x * y^3)⁻¹ * (x + y) :=
by sorry

end simplify_expression_l127_127240


namespace logical_inconsistency_in_dihedral_angle_def_l127_127557

-- Define the given incorrect definition
def incorrect_dihedral_angle_def : String :=
  "A dihedral angle is an angle formed by two half-planes originating from one straight line."

-- Define the correct definition
def correct_dihedral_angle_def : String :=
  "A dihedral angle is a spatial figure consisting of two half-planes that share a common edge."

-- Define the logical inconsistency
theorem logical_inconsistency_in_dihedral_angle_def :
  incorrect_dihedral_angle_def ≠ correct_dihedral_angle_def := by
  sorry

end logical_inconsistency_in_dihedral_angle_def_l127_127557


namespace combined_selling_price_correct_l127_127687

def cost_A : ℕ := 500
def cost_B : ℕ := 800
def cost_C : ℕ := 1200
def profit_A : ℕ := 25
def profit_B : ℕ := 30
def profit_C : ℕ := 20

def selling_price (cost profit_percentage : ℕ) : ℕ :=
  cost + (profit_percentage * cost / 100)

def combined_selling_price : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

theorem combined_selling_price_correct : combined_selling_price = 3105 := by
  sorry

end combined_selling_price_correct_l127_127687


namespace find_a2_l127_127373

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end find_a2_l127_127373


namespace betty_blue_beads_l127_127935

theorem betty_blue_beads (r b : ℕ) (h1 : r = 30) (h2 : 3 * b = 2 * r) : b = 20 :=
by
  sorry

end betty_blue_beads_l127_127935


namespace standard_deviation_is_2point5_l127_127332

noncomputable def mean : ℝ := 17.5
noncomputable def given_value : ℝ := 12.5

theorem standard_deviation_is_2point5 :
  ∀ (σ : ℝ), mean - 2 * σ = given_value → σ = 2.5 := by
  sorry

end standard_deviation_is_2point5_l127_127332


namespace system_solution_b_l127_127185

theorem system_solution_b (x y b : ℚ) 
  (h1 : 4 * x + 2 * y = b) 
  (h2 : 3 * x + 7 * y = 3 * b) 
  (hy : y = 3) : 
  b = 22 / 3 := 
by
  sorry

end system_solution_b_l127_127185


namespace infinite_solutions_implies_a_eq_2_l127_127064

theorem infinite_solutions_implies_a_eq_2 (a b : ℝ) (h : b = 1) :
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → a = 2 :=
by
  intro H
  sorry

end infinite_solutions_implies_a_eq_2_l127_127064


namespace bonifac_distance_l127_127153

/-- Given the conditions provided regarding the paths of Pankrác, Servác, and Bonifác,
prove that the total distance Bonifác walked is 625 meters. -/
theorem bonifac_distance
  (path_Pankrac : ℕ)  -- distance of Pankráč's path in segments
  (meters_Pankrac : ℕ)  -- distance Pankráč walked in meters
  (path_Bonifac : ℕ)  -- distance of Bonifác's path in segments
  (meters_per_segment : ℚ)  -- meters per segment walked
  (Hp : path_Pankrac = 40)  -- Pankráč's path in segments
  (Hm : meters_Pankrac = 500)  -- Pankráč walked 500 meters
  (Hms : meters_per_segment = 500 / 40)  -- meters per segment
  (Hb : path_Bonifac = 50)  -- Bonifác's path in segments
  : path_Bonifac * meters_per_segment = 625 := sorry

end bonifac_distance_l127_127153


namespace quadratic_radical_condition_l127_127498

variable (x : ℝ)

theorem quadratic_radical_condition : 
  (∃ (r : ℝ), r = x^2 + 1 ∧ r ≥ 0) ↔ (True) := by
  sorry

end quadratic_radical_condition_l127_127498


namespace new_interest_rate_l127_127051

theorem new_interest_rate 
  (initial_interest : ℝ) 
  (additional_interest : ℝ) 
  (initial_rate : ℝ) 
  (time : ℝ) 
  (new_total_interest : ℝ)
  (principal : ℝ)
  (new_rate : ℝ) 
  (h1 : initial_interest = principal * initial_rate * time)
  (h2 : new_total_interest = initial_interest + additional_interest)
  (h3 : new_total_interest = principal * new_rate * time)
  (principal_val : principal = initial_interest / initial_rate) :
  new_rate = 0.05 :=
by
  sorry

end new_interest_rate_l127_127051


namespace not_sufficient_nor_necessary_l127_127686

theorem not_sufficient_nor_necessary (a b : ℝ) :
  ¬((a^2 > b^2) → (a > b)) ∧ ¬((a > b) → (a^2 > b^2)) :=
by
  sorry

end not_sufficient_nor_necessary_l127_127686


namespace min_a_plus_b_l127_127370

-- Given conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Equation of line L passing through point (4,1) with intercepts a and b
def line_eq (a b : ℝ) : Prop := (4 / a) + (1 / b) = 1

-- Proof statement
theorem min_a_plus_b (h : line_eq a b) : a + b ≥ 9 :=
sorry

end min_a_plus_b_l127_127370


namespace value_of_a_squared_plus_b_squared_plus_2ab_l127_127788

theorem value_of_a_squared_plus_b_squared_plus_2ab (a b : ℝ) (h : a + b = -1) :
  a^2 + b^2 + 2 * a * b = 1 :=
by sorry

end value_of_a_squared_plus_b_squared_plus_2ab_l127_127788


namespace shorter_leg_in_right_triangle_l127_127225

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l127_127225


namespace lasagna_ground_mince_l127_127534

theorem lasagna_ground_mince (total_ground_mince : ℕ) (num_cottage_pies : ℕ) (ground_mince_per_cottage_pie : ℕ) 
  (num_lasagnas : ℕ) (L : ℕ) : 
  total_ground_mince = 500 ∧ num_cottage_pies = 100 ∧ ground_mince_per_cottage_pie = 3 
  ∧ num_lasagnas = 100 ∧ total_ground_mince - num_cottage_pies * ground_mince_per_cottage_pie = num_lasagnas * L 
  → L = 2 := 
by sorry

end lasagna_ground_mince_l127_127534


namespace abs_value_condition_l127_127447

theorem abs_value_condition (m : ℝ) (h : |m - 1| = m - 1) : m ≥ 1 :=
by {
  sorry
}

end abs_value_condition_l127_127447


namespace cow_spots_total_l127_127313

theorem cow_spots_total
  (left_spots : ℕ) (right_spots : ℕ)
  (left_spots_eq : left_spots = 16)
  (right_spots_eq : right_spots = 3 * left_spots + 7) :
  left_spots + right_spots = 71 :=
by
  sorry

end cow_spots_total_l127_127313


namespace valid_third_side_length_l127_127089

theorem valid_third_side_length : 4 < 6 ∧ 6 < 10 :=
by
  exact ⟨by norm_num, by norm_num⟩

end valid_third_side_length_l127_127089


namespace part1_l127_127423

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

theorem part1 (U_eq : U = univ) 
  (A_eq : A = {x | (x - 5) / (x - 2) ≤ 0}) 
  (B_eq : B = {x | 1 < x ∧ x < 3}) :
  compl A ∩ compl B = {x | x ≤ 1 ∨ x > 5} := 
  sorry

end part1_l127_127423


namespace Ashok_took_six_subjects_l127_127880

theorem Ashok_took_six_subjects
  (n : ℕ) -- number of subjects Ashok took
  (T : ℕ) -- total marks secured in those subjects
  (h_avg_n : T = n * 72) -- condition: average of marks in n subjects is 72
  (h_avg_5 : 5 * 74 = 370) -- condition: average of marks in 5 subjects is 74
  (h_6th_mark : 62 > 0) -- condition: the 6th subject's mark is 62
  (h_T : T = 370 + 62) -- condition: total marks including the 6th subject
  : n = 6 := 
sorry


end Ashok_took_six_subjects_l127_127880


namespace compare_triangle_operations_l127_127030

def tri_op (a b : ℤ) : ℤ := a * b - a - b + 1

theorem compare_triangle_operations : tri_op (-3) 4 = tri_op 4 (-3) :=
by
  unfold tri_op
  sorry

end compare_triangle_operations_l127_127030


namespace matchstick_ratio_is_one_half_l127_127049

def matchsticks_used (houses : ℕ) (matchsticks_per_house : ℕ) : ℕ :=
  houses * matchsticks_per_house

def ratio (a b : ℕ) : ℚ := a / b

def michael_original_matchsticks : ℕ := 600
def michael_houses : ℕ := 30
def matchsticks_per_house : ℕ := 10
def michael_used_matchsticks : ℕ := matchsticks_used michael_houses matchsticks_per_house

theorem matchstick_ratio_is_one_half :
  ratio michael_used_matchsticks michael_original_matchsticks = 1 / 2 :=
by
  sorry

end matchstick_ratio_is_one_half_l127_127049


namespace geometric_sum_2015_2016_l127_127925

theorem geometric_sum_2015_2016 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 2)
  (h_a2_a5 : a 2 + a 5 = 0)
  (h_Sn : ∀ n, S n = (1 - (-1)^n)) :
  S 2015 + S 2016 = 2 :=
by sorry

end geometric_sum_2015_2016_l127_127925


namespace correct_statements_l127_127990

def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonic_increasing_on_neg1_0 : ∀ ⦃x y : ℝ⦄, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y
axiom functional_eqn (x : ℝ) : f (1 - x) + f (1 + x) = 0

theorem correct_statements :
  (∀ x, f (1 - x) = -f (1 + x)) ∧ f 2 ≤ f x :=
by
  sorry

end correct_statements_l127_127990


namespace intersection_M_N_l127_127608

-- Definitions of sets M and N
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

-- Lean statement to prove that the intersection of M and N is {1, 2}
theorem intersection_M_N : M ∩ N = {1, 2} :=
by
  sorry

end intersection_M_N_l127_127608


namespace sum_of_sides_le_twice_third_side_l127_127565

theorem sum_of_sides_le_twice_third_side 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180)
  (h3 : a / (Real.sin A) = b / (Real.sin B))
  (h4 : a / (Real.sin A) = c / (Real.sin C))
  (h5 : b / (Real.sin B) = c / (Real.sin C)) : 
  a + c ≤ 2 * b := 
by 
  sorry

end sum_of_sides_le_twice_third_side_l127_127565


namespace functional_equation_solution_l127_127170

-- Define the function
def f : ℝ → ℝ := sorry

-- The main theorem to prove
theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) → (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end functional_equation_solution_l127_127170


namespace number_of_deluxe_volumes_l127_127113

theorem number_of_deluxe_volumes (d s : ℕ) 
  (h1 : d + s = 15)
  (h2 : 30 * d + 20 * s = 390) : 
  d = 9 :=
by
  sorry

end number_of_deluxe_volumes_l127_127113


namespace percentage_of_green_ducks_l127_127367

def total_ducks := 100
def green_ducks_smaller_pond := 9
def green_ducks_larger_pond := 22
def total_green_ducks := green_ducks_smaller_pond + green_ducks_larger_pond

theorem percentage_of_green_ducks :
  (total_green_ducks / total_ducks) * 100 = 31 :=
by
  sorry

end percentage_of_green_ducks_l127_127367


namespace inequality_solution_l127_127808

theorem inequality_solution (x : ℝ) : 
  (7 - 2 * (x + 1) ≥ 1 - 6 * x) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (-1 ≤ x ∧ x < 4) := 
by
  sorry

end inequality_solution_l127_127808


namespace angle4_is_35_l127_127803

theorem angle4_is_35
  (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (ha : angle1 = 50)
  (h_opposite : angle5 = 60)
  (triangle_sum : angle1 + angle5 + angle6 = 180)
  (supplementary_angle : angle2 + angle6 = 180) :
  angle4 = 35 :=
by
  sorry

end angle4_is_35_l127_127803


namespace min_value_expression_l127_127428

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 8) : 
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 :=
sorry

end min_value_expression_l127_127428


namespace mona_cookie_count_l127_127513

theorem mona_cookie_count {M : ℕ} (h1 : (M - 5) + (M - 5 + 10) + M = 60) : M = 20 :=
by
  sorry

end mona_cookie_count_l127_127513


namespace arithmetic_sequence_a5_l127_127563

variable (a : ℕ → ℝ)

-- Conditions translated to Lean definitions
def cond1 : Prop := a 3 = 7
def cond2 : Prop := a 9 = 19

-- Theorem statement that needs to be proved
theorem arithmetic_sequence_a5 (h1 : cond1 a) (h2 : cond2 a) : a 5 = 11 :=
sorry

end arithmetic_sequence_a5_l127_127563


namespace factory_output_l127_127106

theorem factory_output :
  ∀ (J M : ℝ), M = J * 0.8 → J = M * 1.25 :=
by
  intros J M h
  sorry

end factory_output_l127_127106


namespace max_value_of_a_plus_b_l127_127678

theorem max_value_of_a_plus_b (a b : ℕ) (h1 : 7 * a + 19 * b = 213) (h2 : a > 0) (h3 : b > 0) : a + b = 27 :=
sorry

end max_value_of_a_plus_b_l127_127678


namespace initial_price_of_phone_l127_127352

theorem initial_price_of_phone
  (initial_price_TV : ℕ)
  (increase_TV_fraction : ℚ)
  (initial_price_phone : ℚ)
  (increase_phone_percentage : ℚ)
  (total_amount : ℚ)
  (h1 : initial_price_TV = 500)
  (h2 : increase_TV_fraction = 2/5)
  (h3 : increase_phone_percentage = 0.40)
  (h4 : total_amount = 1260) :
  initial_price_phone = 400 := by
  sorry

end initial_price_of_phone_l127_127352


namespace brad_zip_code_l127_127382

theorem brad_zip_code (a b c d e : ℕ) 
  (h1 : a = b) 
  (h2 : c = 0) 
  (h3 : d = 2 * a) 
  (h4 : d + e = 8) 
  (h5 : a + b + c + d + e = 10) : 
  (a, b, c, d, e) = (1, 1, 0, 2, 6) :=
by 
  -- Proof omitted on purpose
  sorry

end brad_zip_code_l127_127382


namespace kittens_total_number_l127_127139

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end kittens_total_number_l127_127139


namespace common_speed_is_10_l127_127971

noncomputable def speed_jack (x : ℝ) : ℝ := x^2 - 11 * x - 22
noncomputable def speed_jill (x : ℝ) : ℝ := 
  if x = -6 then 0 else (x^2 - 4 * x - 12) / (x + 6)

theorem common_speed_is_10 (x : ℝ) (h : speed_jack x = speed_jill x) (hx : x = 16) : 
  speed_jack x = 10 :=
by
  sorry

end common_speed_is_10_l127_127971


namespace sum_of_three_pairwise_rel_prime_integers_l127_127307

theorem sum_of_three_pairwise_rel_prime_integers (a b c : ℕ)
  (h1: 1 < a) (h2: 1 < b) (h3: 1 < c)
  (prod: a * b * c = 216000)
  (rel_prime_ab : Nat.gcd a b = 1)
  (rel_prime_ac : Nat.gcd a c = 1)
  (rel_prime_bc : Nat.gcd b c = 1) : 
  a + b + c = 184 := 
sorry

end sum_of_three_pairwise_rel_prime_integers_l127_127307


namespace extreme_value_and_inequality_l127_127818

theorem extreme_value_and_inequality
  (f : ℝ → ℝ)
  (a c : ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_extreme : f 1 = -2)
  (h_f_def : ∀ x : ℝ, f x = a * x^3 + c * x)
  (h_a_c : a = 1 ∧ c = -3) :
  (∀ x : ℝ, x < -1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0) ∧
  (∀ x : ℝ, 1 < x → deriv f x > 0) ∧
  f (-1) = 2 ∧
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 → |f x₁ - f x₂| < 4) :=
by sorry

end extreme_value_and_inequality_l127_127818


namespace sequence_a_n_a31_l127_127273

theorem sequence_a_n_a31 (a : ℕ → ℤ) 
  (h_initial : a 1 = 2)
  (h_recurrence : ∀ n : ℕ, a n + a (n + 1) + n^2 = 0) :
  a 31 = -463 :=
sorry

end sequence_a_n_a31_l127_127273


namespace man_speed_proof_l127_127578

noncomputable def man_speed_to_post_office (v : ℝ) : Prop :=
  let distance := 19.999999999999996
  let time_back := distance / 4
  let total_time := 5 + 48 / 60
  v > 0 ∧ distance / v + time_back = total_time

theorem man_speed_proof : ∃ v : ℝ, man_speed_to_post_office v ∧ v = 25 := by
  sorry

end man_speed_proof_l127_127578


namespace smallest_odd_prime_factor_l127_127848

theorem smallest_odd_prime_factor (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (2023 ^ 8 + 1) % p = 0 ↔ p = 17 := 
by
  sorry

end smallest_odd_prime_factor_l127_127848


namespace inequality_proof_l127_127018

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_eq : a + b + c = 4 * (abc)^(1/3)) :
  2 * (ab + bc + ca) + 4 * min (a^2) (min (b^2) (c^2)) ≥ a^2 + b^2 + c^2 :=
by
  sorry

end inequality_proof_l127_127018


namespace solve_linear_system_l127_127933

variable {x y : ℚ}

theorem solve_linear_system (h1 : 4 * x - 3 * y = -17) (h2 : 5 * x + 6 * y = -4) :
  (x, y) = (-(74 / 13 : ℚ), -(25 / 13 : ℚ)) :=
by
  sorry

end solve_linear_system_l127_127933


namespace fraction_to_decimal_l127_127799

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l127_127799


namespace marble_problem_l127_127954

-- Define the initial number of marbles
def initial_marbles : Prop :=
  ∃ (x y : ℕ), (y - 4 = 2 * (x + 4)) ∧ (y + 2 = 11 * (x - 2)) ∧ (y = 20) ∧ (x = 4)

-- The main theorem to prove the initial number of marbles
theorem marble_problem (x y : ℕ) (cond1 : y - 4 = 2 * (x + 4)) (cond2 : y + 2 = 11 * (x - 2)) :
  y = 20 ∧ x = 4 :=
sorry

end marble_problem_l127_127954


namespace base_subtraction_l127_127856

-- Define the base 8 number 765432_8 and its conversion to base 10
def base8Number : ℕ := 7 * (8^5) + 6 * (8^4) + 5 * (8^3) + 4 * (8^2) + 3 * (8^1) + 2 * (8^0)

-- Define the base 9 number 543210_9 and its conversion to base 10
def base9Number : ℕ := 5 * (9^5) + 4 * (9^4) + 3 * (9^3) + 2 * (9^2) + 1 * (9^1) + 0 * (9^0)

-- Lean 4 statement for the proof problem
theorem base_subtraction : (base8Number : ℤ) - (base9Number : ℤ) = -67053 := by
    sorry

end base_subtraction_l127_127856


namespace greatest_consecutive_integers_sum_36_l127_127388

-- Definition of the sum of N consecutive integers starting from a
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Problem statement in Lean 4
theorem greatest_consecutive_integers_sum_36 (N : ℤ) (h : sum_consecutive_integers (-35) 72 = 36) : N = 72 := by
  sorry

end greatest_consecutive_integers_sum_36_l127_127388


namespace probability_heads_and_multiple_of_five_l127_127590

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def coin_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

def die_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

theorem probability_heads_and_multiple_of_five :
  coin_is_fair ∧ die_is_fair →
  (1 / 2) * (1 / 6) = (1 / 12) :=
by
  intro h
  sorry

end probability_heads_and_multiple_of_five_l127_127590


namespace alice_total_spending_l127_127668

theorem alice_total_spending :
  let book_price_gbp := 15
  let souvenir_price_eur := 20
  let gbp_to_usd_rate := 1.25
  let eur_to_usd_rate := 1.10
  let book_price_usd := book_price_gbp * gbp_to_usd_rate
  let souvenir_price_usd := souvenir_price_eur * eur_to_usd_rate
  let total_usd := book_price_usd + souvenir_price_usd
  total_usd = 40.75 :=
by
  sorry

end alice_total_spending_l127_127668


namespace mass_of_added_water_with_temp_conditions_l127_127648

theorem mass_of_added_water_with_temp_conditions
  (m_l : ℝ) (t_pi t_B t : ℝ) (c_B c_l lambda : ℝ) :
  m_l = 0.05 →
  t_pi = -10 →
  t_B = 10 →
  t = 0 →
  c_B = 4200 →
  c_l = 2100 →
  lambda = 3.3 * 10^5 →
  (0.0028 ≤ (2.1 * m_l * 10 + lambda * m_l) / (42 * 10) 
  ∧ (2.1 * m_l * 10) / (42 * 10) ≤ 0.418) :=
by
  sorry

end mass_of_added_water_with_temp_conditions_l127_127648


namespace find_ratio_b_over_a_l127_127599

theorem find_ratio_b_over_a (a b : ℝ)
  (h1 : ∀ x, deriv (fun x => a * x^2 + b) x = 2 * a * x)
  (h2 : deriv (fun x => a * x^2 + b) 1 = 2)
  (h3 : a * 1^2 + b = 3) : b / a = 2 := 
sorry

end find_ratio_b_over_a_l127_127599


namespace molecular_weight_neutralization_l127_127209

def molecular_weight_acetic_acid : ℝ := 
  (12.01 * 2) + (1.008 * 4) + (16.00 * 2)

def molecular_weight_sodium_hydroxide : ℝ := 
  22.99 + 16.00 + 1.008

def total_weight_acetic_acid (moles : ℝ) : ℝ := 
  molecular_weight_acetic_acid * moles

def total_weight_sodium_hydroxide (moles : ℝ) : ℝ := 
  molecular_weight_sodium_hydroxide * moles

def total_molecular_weight (moles_ac: ℝ) (moles_naoh : ℝ) : ℝ :=
  total_weight_acetic_acid moles_ac + 
  total_weight_sodium_hydroxide moles_naoh

theorem molecular_weight_neutralization :
  total_molecular_weight 7 10 = 820.344 :=
by
  sorry

end molecular_weight_neutralization_l127_127209


namespace cost_of_15_brown_socks_is_3_dollars_l127_127697

def price_of_brown_sock (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) : ℚ :=
  (price_white_socks - price_white_more_than_brown) / 2

def cost_of_15_brown_socks (price_brown_sock : ℚ) : ℚ :=
  15 * price_brown_sock

theorem cost_of_15_brown_socks_is_3_dollars
  (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) 
  (h1 : price_white_socks = 0.45) (h2 : price_white_more_than_brown = 0.25) :
  cost_of_15_brown_socks (price_of_brown_sock price_white_socks price_white_more_than_brown) = 3 := 
by
  sorry

end cost_of_15_brown_socks_is_3_dollars_l127_127697


namespace minimum_time_for_xiang_qing_fried_eggs_l127_127671

-- Define the time taken for each individual step
def wash_scallions_time : ℕ := 1
def beat_eggs_time : ℕ := 1 / 2
def mix_egg_scallions_time : ℕ := 1
def wash_pan_time : ℕ := 1 / 2
def heat_pan_time : ℕ := 1 / 2
def heat_oil_time : ℕ := 1 / 2
def cook_dish_time : ℕ := 2

-- Define the total minimum time required
def minimum_time : ℕ := 5

-- The main theorem stating that the minimum time required is 5 minutes
theorem minimum_time_for_xiang_qing_fried_eggs :
  wash_scallions_time + beat_eggs_time + mix_egg_scallions_time + wash_pan_time + heat_pan_time + heat_oil_time + cook_dish_time = minimum_time := 
by sorry

end minimum_time_for_xiang_qing_fried_eggs_l127_127671


namespace integer_solutions_count_l127_127230

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l127_127230


namespace unique_solution_l127_127085

theorem unique_solution (x : ℝ) (h : (1 / (x - 1)) = (3 / (2 * x - 3))) : x = 0 := 
sorry

end unique_solution_l127_127085


namespace charlie_and_dana_proof_l127_127732

noncomputable def charlie_and_dana_ways 
    (cookies : ℕ) (smoothies : ℕ) (total_items : ℕ) 
    (distinct_charlie : ℕ) 
    (repeatable_dana : ℕ) : ℕ :=
    if cookies = 8 ∧ smoothies = 5 ∧ total_items = 5 ∧ distinct_charlie = 0 
       ∧ repeatable_dana = 0 then 27330 else 0

theorem charlie_and_dana_proof :
  charlie_and_dana_ways 8 5 5 0 0 = 27330 := 
  sorry

end charlie_and_dana_proof_l127_127732


namespace x2004_y2004_l127_127501

theorem x2004_y2004 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
  x^2004 + y^2004 = 2^2004 := 
by
  sorry

end x2004_y2004_l127_127501


namespace find_whole_number_l127_127572

theorem find_whole_number (N : ℕ) : 9.25 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 9.75 → N = 38 := by
  intros h
  have hN : 37 < (N : ℝ) ∧ (N : ℝ) < 39 := by
    -- This part follows directly from multiplying the inequality by 4.
    sorry

  -- Convert to integer comparison
  have h1 : 38 ≤ N := by
    -- Since 37 < N, N must be at least 38 as N is an integer.
    sorry
    
  have h2 : N < 39 := by
    sorry

  -- Conclude that N = 38 as it is the single whole number within the range.
  sorry

end find_whole_number_l127_127572


namespace find_y_given_conditions_l127_127734

theorem find_y_given_conditions (x y : ℝ) (h₁ : 3 * x^2 = y - 6) (h₂ : x = 4) : y = 54 :=
  sorry

end find_y_given_conditions_l127_127734


namespace next_working_day_together_l127_127827

theorem next_working_day_together : 
  let greta_days := 5
  let henry_days := 3
  let linda_days := 9
  let sam_days := 8
  ∃ n : ℕ, n = Nat.lcm (Nat.lcm (Nat.lcm greta_days henry_days) linda_days) sam_days ∧ n = 360 :=
by
  sorry

end next_working_day_together_l127_127827


namespace larger_number_is_37_l127_127159

-- Defining the conditions
def sum_of_two_numbers (a b : ℕ) : Prop := a + b = 62
def one_is_12_more (a b : ℕ) : Prop := a = b + 12

-- Proof statement
theorem larger_number_is_37 (a b : ℕ) (h₁ : sum_of_two_numbers a b) (h₂ : one_is_12_more a b) : a = 37 :=
by
  sorry

end larger_number_is_37_l127_127159


namespace problem1_problem2_l127_127547

open Real

-- Proof problem 1: Given condition and the required result.
theorem problem1 (x y : ℝ) (h : (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7) :
  x^2 + y^2 = 5 :=
sorry

-- Proof problem 2: Solve the polynomial equation.
theorem problem2 (x : ℝ) :
  (x = sqrt 2 ∨ x = -sqrt 2 ∨ x = 2 ∨ x = -2) ↔ (x^4 - 6 * x^2 + 8 = 0) :=
sorry

end problem1_problem2_l127_127547


namespace find_divisor_l127_127860

-- Definitions from the condition
def original_number : ℕ := 724946
def least_number_subtracted : ℕ := 6
def remaining_number : ℕ := original_number - least_number_subtracted

theorem find_divisor (h1 : remaining_number % least_number_subtracted = 0) :
  Nat.gcd original_number least_number_subtracted = 2 :=
sorry

end find_divisor_l127_127860


namespace value_of_q_l127_127000

open Real

theorem value_of_q (p q : ℝ) (hpq_cond1 : 1 < p ∧ p < q) 
  (hpq_cond2 : 1 / p + 1 / q = 1) (hpq_cond3 : p * q = 8) : q = 4 + 2 * sqrt 2 :=
by
  sorry

end value_of_q_l127_127000


namespace negation_universal_statement_l127_127282

theorem negation_universal_statement :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by sorry

end negation_universal_statement_l127_127282


namespace symmetric_line_equation_l127_127752

theorem symmetric_line_equation (x y : ℝ) : 
  3 * x - 4 * y + 5 = 0 → (3 * x + 4 * y - 5 = 0) :=
by
sorry

end symmetric_line_equation_l127_127752


namespace certain_number_existence_l127_127607

theorem certain_number_existence : ∃ x : ℝ, (102 * 102) + (x * x) = 19808 ∧ x = 97 := by
  sorry

end certain_number_existence_l127_127607


namespace flower_stones_per_bracelet_l127_127735

theorem flower_stones_per_bracelet (total_stones : ℝ) (bracelets : ℝ)  (H_total: total_stones = 88.0) (H_bracelets: bracelets = 8.0) :
  (total_stones / bracelets = 11.0) :=
by
  rw [H_total, H_bracelets]
  norm_num

end flower_stones_per_bracelet_l127_127735


namespace number_of_kids_stay_home_l127_127619

def total_kids : ℕ := 313473
def kids_at_camp : ℕ := 38608
def kids_stay_home : ℕ := 274865

theorem number_of_kids_stay_home :
  total_kids - kids_at_camp = kids_stay_home := 
by
  -- Subtracting the number of kids who go to camp from the total number of kids
  sorry

end number_of_kids_stay_home_l127_127619


namespace temperature_of_Huangshan_at_night_l127_127183

theorem temperature_of_Huangshan_at_night 
  (T_morning : ℤ) (Rise_noon : ℤ) (Drop_night : ℤ)
  (h1 : T_morning = -12) (h2 : Rise_noon = 8) (h3 : Drop_night = 10) :
  T_morning + Rise_noon - Drop_night = -14 :=
by
  sorry

end temperature_of_Huangshan_at_night_l127_127183


namespace cube_div_identity_l127_127650

theorem cube_div_identity (a b : ℕ) (h1 : a = 6) (h2 : b = 3) : 
  (a^3 - b^3) / (a^2 + a * b + b^2) = 3 :=
by {
  sorry
}

end cube_div_identity_l127_127650


namespace polynomial_square_solution_l127_127927

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end polynomial_square_solution_l127_127927


namespace total_length_of_segments_l127_127129

theorem total_length_of_segments
  (l1 l2 l3 l4 l5 l6 : ℕ) 
  (hl1 : l1 = 5) 
  (hl2 : l2 = 1) 
  (hl3 : l3 = 4) 
  (hl4 : l4 = 2) 
  (hl5 : l5 = 3) 
  (hl6 : l6 = 3) : 
  l1 + l2 + l3 + l4 + l5 + l6 = 18 := 
by 
  sorry

end total_length_of_segments_l127_127129


namespace n_minus_k_minus_l_square_number_l127_127065

variable (n k l x : ℕ)

theorem n_minus_k_minus_l_square_number (h1 : x^2 < n)
                                        (h2 : n < (x + 1)^2)
                                        (h3 : n - k = x^2)
                                        (h4 : n + l = (x + 1)^2) :
  ∃ m : ℕ, n - k - l = m ^ 2 :=
by
  sorry

end n_minus_k_minus_l_square_number_l127_127065


namespace find_y_value_l127_127907

-- Define the linear relationship
def linear_eq (k b x : ℝ) : ℝ := k * x + b

-- Given conditions
variables (k b : ℝ)
axiom h1 : linear_eq k b 0 = -1
axiom h2 : linear_eq k b (1/2) = 2

-- Prove that the value of y when x = -1/2 is -4
theorem find_y_value : linear_eq k b (-1/2) = -4 :=
by sorry

end find_y_value_l127_127907


namespace fg_difference_l127_127254

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference : f (g 3) - g (f 3) = 59 :=
by
  sorry

end fg_difference_l127_127254


namespace annual_interest_income_l127_127396

variables (totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate : ℝ)
           (firstInterest secondInterest totalInterest : ℝ)

def investment_conditions : Prop :=
  totalInvestment = 32000 ∧
  firstRate = 0.0575 ∧
  secondRate = 0.0625 ∧
  firstBondPrincipal = 20000 ∧
  secondBondPrincipal = totalInvestment - firstBondPrincipal

def calculate_interest (principal rate : ℝ) : ℝ := principal * rate

def total_annual_interest (firstInterest secondInterest : ℝ) : ℝ :=
  firstInterest + secondInterest

theorem annual_interest_income
  (hc : investment_conditions totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate) :
  total_annual_interest (calculate_interest firstBondPrincipal firstRate)
    (calculate_interest secondBondPrincipal secondRate) = 1900 :=
by {
  sorry
}

end annual_interest_income_l127_127396


namespace count_valid_five_digit_numbers_l127_127416

-- Define the conditions
def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def quotient_remainder_sum_divisible_by (n q r : ℕ) : Prop :=
  (n = 100 * q + r) ∧ ((q + r) % 7 = 0)

-- Define the theorem
theorem count_valid_five_digit_numbers : 
  ∃ k, k = 8160 ∧ ∀ n, is_five_digit_number n ∧ 
    is_divisible_by n 13 ∧ 
    ∃ q r, quotient_remainder_sum_divisible_by n q r → 
    k = 8160 :=
sorry

end count_valid_five_digit_numbers_l127_127416


namespace parabola_intercept_sum_l127_127124

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end parabola_intercept_sum_l127_127124


namespace money_left_l127_127083

theorem money_left (olivia_money nigel_money ticket_cost tickets_purchased : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : tickets_purchased = 6) : 
  olivia_money + nigel_money - tickets_purchased * ticket_cost = 83 := 
by 
  sorry

end money_left_l127_127083


namespace negation_all_nonzero_l127_127451

    theorem negation_all_nonzero (a b c : ℝ) : ¬ (¬ (a = 0 ∨ b = 0 ∨ c = 0)) → (a = 0 ∧ b = 0 ∧ c = 0) :=
    by
      sorry
    
end negation_all_nonzero_l127_127451


namespace find_B_divisible_by_6_l127_127728

theorem find_B_divisible_by_6 (B : ℕ) : (5170 + B) % 6 = 0 ↔ (B = 2 ∨ B = 8) :=
by
  -- Conditions extracted from the problem are directly used here:
  sorry -- Proof would be here

end find_B_divisible_by_6_l127_127728


namespace total_revenue_4706_l127_127759

noncomputable def totalTicketRevenue (seats : ℕ) (show2pm : ℕ × ℕ) (show5pm : ℕ × ℕ) (show8pm : ℕ × ℕ) : ℕ :=
  let revenue2pm := show2pm.1 * 4 + (seats - show2pm.1) * 6
  let revenue5pm := show5pm.1 * 5 + (seats - show5pm.1) * 8
  let revenue8pm := show8pm.1 * 7 + (show8pm.2 - show8pm.1) * 10
  revenue2pm + revenue5pm + revenue8pm

theorem total_revenue_4706 :
  totalTicketRevenue 250 (135, 250) (160, 250) (98, 225) = 4706 :=
by
  unfold totalTicketRevenue
  -- We provide the proof steps here in a real proof scenario.
  -- We are focusing on the statement formulation only.
  sorry

end total_revenue_4706_l127_127759


namespace unique_sequence_count_l127_127493

def is_valid_sequence (a : Fin 5 → ℕ) :=
  a 0 = 1 ∧
  a 1 > a 0 ∧
  a 2 > a 1 ∧
  a 3 > a 2 ∧
  a 4 = 15 ∧
  (a 1) ^ 2 ≤ a 0 * a 2 + 1 ∧
  (a 2) ^ 2 ≤ a 1 * a 3 + 1 ∧
  (a 3) ^ 2 ≤ a 2 * a 4 + 1

theorem unique_sequence_count : 
  ∃! (a : Fin 5 → ℕ), is_valid_sequence a :=
sorry

end unique_sequence_count_l127_127493


namespace sum_of_coeffs_l127_127492

theorem sum_of_coeffs (a_5 a_4 a_3 a_2 a_1 a : ℤ) (h_eq : (x - 2)^5 = a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) (h_a : a = -32) :
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coeffs_l127_127492


namespace committee_member_count_l127_127353

theorem committee_member_count (n : ℕ) (M : ℕ) (Q : ℚ) 
  (h₁ : M = 6) 
  (h₂ : 2 * n = M) 
  (h₃ : Q = 0.4) 
  (h₄ : Q = (n - 1) / (M - 1)) : 
  n = 3 :=
by
  sorry

end committee_member_count_l127_127353


namespace equation_of_line_l127_127044

variable {a b k T : ℝ}

theorem equation_of_line (h_b_ne_zero : b ≠ 0)
  (h_line_passing_through : ∃ (line : ℝ → ℝ), line (-a) = b)
  (h_triangle_area : ∃ (h : ℝ), T = 1 / 2 * ka * (h - b))
  (h_base_length : ∃ (base : ℝ), base = ka) :
  ∃ (x y : ℝ), 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0 :=
sorry

end equation_of_line_l127_127044


namespace math_problem_l127_127675

theorem math_problem (x : ℤ) :
  let a := 1990 * x + 1989
  let b := 1990 * x + 1990
  let c := 1990 * x + 1991
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end math_problem_l127_127675


namespace correct_calculation_l127_127456

theorem correct_calculation :
  (-7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2) ∧
  ¬ (2 * x + 3 * y = 5 * x * y) ∧
  ¬ (6 * x^2 - (-x^2) = 5 * x^2) ∧
  ¬ (4 * m * n - 3 * m * n = 1) :=
by
  sorry

end correct_calculation_l127_127456


namespace xy_value_l127_127582

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end xy_value_l127_127582


namespace tank_capacity_l127_127693

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end tank_capacity_l127_127693


namespace find_values_l127_127533

-- Define the conditions as Lean hypotheses
variables (A B : ℝ)

-- State the problem conditions
def condition1 := 30 - (4 * A + 5) = 3 * B
def condition2 := B = 2 * A

-- State the main theorem to be proved
theorem find_values (h1 : condition1 A B) (h2 : condition2 A B) : A = 2.5 ∧ B = 5 :=
by { sorry }

end find_values_l127_127533


namespace find_t_l127_127193

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2 : ℝ)^(n-1)

noncomputable def S_3n (n : ℕ) : ℝ := (1 - (2 : ℝ)^(3 * n)) / (1 - 2)

noncomputable def a_n_cubed (n : ℕ) : ℝ := (a_n n)^3

noncomputable def T_n (n : ℕ) : ℝ := (1 - (a_n_cubed 2)^n) / (1 - (a_n_cubed 2))

theorem find_t (n : ℕ) : S_3n n = 7 * T_n n :=
by
  sorry

end find_t_l127_127193


namespace smallest_x_for_M_cube_l127_127549

theorem smallest_x_for_M_cube (x M : ℤ) (h1 : 1890 * x = M^3) : x = 4900 :=
sorry

end smallest_x_for_M_cube_l127_127549


namespace a679b_multiple_of_72_l127_127500

-- Define conditions
def is_divisible_by_8 (n : Nat) : Prop :=
  n % 8 = 0

def sum_of_digits_is_divisible_by_9 (n : Nat) : Prop :=
  (n.digits 10).sum % 9 = 0

-- Define the given problem
theorem a679b_multiple_of_72 (a b : Nat) : 
  is_divisible_by_8 (7 * 100 + 9 * 10 + b) →
  sum_of_digits_is_divisible_by_9 (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) → 
  a = 3 ∧ b = 2 :=
by 
  sorry

end a679b_multiple_of_72_l127_127500


namespace dust_particles_calculation_l127_127910

theorem dust_particles_calculation (D : ℕ) (swept : ℝ) (left_by_shoes : ℕ) (total_after_walk : ℕ)  
  (h_swept : swept = 9 / 10)
  (h_left_by_shoes : left_by_shoes = 223)
  (h_total_after_walk : total_after_walk = 331)
  (h_equation : (1 - swept) * D + left_by_shoes = total_after_walk) : 
  D = 1080 := 
by
  sorry

end dust_particles_calculation_l127_127910


namespace percentage_drop_l127_127692

theorem percentage_drop (P N P' N' : ℝ) (h1 : N' = 1.60 * N) (h2 : P' * N' = 1.2800000000000003 * (P * N)) :
  P' = 0.80 * P :=
by
  sorry

end percentage_drop_l127_127692


namespace isosceles_largest_angle_eq_60_l127_127289

theorem isosceles_largest_angle_eq_60 :
  ∀ (A B C : ℝ), (
    -- Condition: A triangle is isosceles with two equal angles of 60 degrees.
    ∀ (x y : ℝ), A = x ∧ B = x ∧ C = y ∧ x = 60 →
    -- Prove that
    max A (max B C) = 60 ) :=
by
  intros A B C h
  -- Sorry denotes skipping the proof.
  sorry

end isosceles_largest_angle_eq_60_l127_127289


namespace computer_price_after_six_years_l127_127166

def price_decrease (p_0 : ℕ) (rate : ℚ) (t : ℕ) : ℚ :=
  p_0 * rate ^ (t / 2)

theorem computer_price_after_six_years :
  price_decrease 8100 (2 / 3) 6 = 2400 := by
  sorry

end computer_price_after_six_years_l127_127166


namespace total_weight_l127_127219

variable (a b c d : ℝ)

-- Conditions
axiom h1 : a + b = 250
axiom h2 : b + c = 235
axiom h3 : c + d = 260
axiom h4 : a + d = 275

-- Proving the total weight
theorem total_weight : a + b + c + d = 510 := by
  sorry

end total_weight_l127_127219


namespace shoe_length_size_15_l127_127887

theorem shoe_length_size_15 : 
  ∀ (length : ℕ → ℝ), 
    (∀ n, 8 ≤ n ∧ n ≤ 17 → length (n + 1) = length n + 1 / 4) → 
    length 17 = (1 + 0.10) * length 8 →
    length 15 = 24.25 :=
by
  intro length h_increase h_largest
  sorry

end shoe_length_size_15_l127_127887


namespace usual_time_to_catch_bus_l127_127550

variables (S T T' : ℝ)

theorem usual_time_to_catch_bus
  (h1 : T' = (5 / 4) * T)
  (h2 : T' - T = 6) : T = 24 :=
sorry

end usual_time_to_catch_bus_l127_127550


namespace articles_for_z_men_l127_127820

-- The necessary conditions and given values
def articles_produced (men hours days : ℕ) := men * hours * days

theorem articles_for_z_men (x z : ℕ) (H : articles_produced x x x = x^2) :
  articles_produced z z z = z^3 / x := by
  sorry

end articles_for_z_men_l127_127820


namespace guitar_price_proof_l127_127805

def total_guitar_price (x : ℝ) : Prop :=
  0.20 * x = 240 → x = 1200

theorem guitar_price_proof (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end guitar_price_proof_l127_127805


namespace simplify_expression_l127_127772

variable (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b)
variable (h : a^3 - b^3 = a - b)

theorem simplify_expression 
  (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b) (h : a^3 - b^3 = a - b) : 
  (a / b - b / a + 1 / (a * b)) = 2 * (1 / (a * b)) - 1 := 
sorry

end simplify_expression_l127_127772


namespace jennifer_initial_pears_l127_127445

def initialPears (P: ℕ) : Prop := (P + 20 + 2 * P - 6 = 44)

theorem jennifer_initial_pears (P: ℕ) (h : initialPears P) : P = 10 := by
  sorry

end jennifer_initial_pears_l127_127445


namespace exam_results_l127_127431

variable (E F G H : Prop)

def emma_statement : Prop := E → F
def frank_statement : Prop := F → ¬G
def george_statement : Prop := G → H
def exactly_two_asing : Prop :=
  (E ∧ F ∧ ¬G ∧ ¬H) ∨ (¬E ∧ F ∧ G ∧ ¬H) ∨
  (¬E ∧ ¬F ∧ G ∧ H) ∨ (¬E ∧ F ∧ ¬G ∧ H) ∨
  (E ∧ ¬F ∧ ¬G ∧ H)

theorem exam_results :
  (E ∧ F) ∨ (G ∧ H) :=
by {
  sorry
}

end exam_results_l127_127431


namespace fraction_of_sum_l127_127946

theorem fraction_of_sum (l : List ℝ) (n : ℝ) (h_len : l.length = 21) (h_mem : n ∈ l)
  (h_n_avg : n = 4 * (l.erase n).sum / 20) :
  n / l.sum = 1 / 6 := by
  sorry

end fraction_of_sum_l127_127946


namespace min_value_inverse_sum_l127_127054

theorem min_value_inverse_sum (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a + 2 * b = 1) : 
  ∃ (y : ℝ), y = 3 + 2 * Real.sqrt 2 ∧ (∀ x, x = (1 / a) + (1 / b) → y ≤ x) :=
sorry

end min_value_inverse_sum_l127_127054


namespace remainder_2_pow_33_mod_9_l127_127104

theorem remainder_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_2_pow_33_mod_9_l127_127104


namespace log_increasing_a_gt_one_l127_127126

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_increasing_a_gt_one (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log a 2 < log a 3) : a > 1 :=
by
  sorry

end log_increasing_a_gt_one_l127_127126


namespace eldest_sibling_age_correct_l127_127460

-- Definitions and conditions
def youngest_sibling_age (x : ℝ) := x
def second_youngest_sibling_age (x : ℝ) := x + 4
def third_youngest_sibling_age (x : ℝ) := x + 8
def fourth_youngest_sibling_age (x : ℝ) := x + 12
def fifth_youngest_sibling_age (x : ℝ) := x + 16
def sixth_youngest_sibling_age (x : ℝ) := x + 20
def seventh_youngest_sibling_age (x : ℝ) := x + 28
def eldest_sibling_age (x : ℝ) := x + 32

def combined_age_of_eight_siblings (x : ℝ) : ℝ := 
  youngest_sibling_age x +
  second_youngest_sibling_age x +
  third_youngest_sibling_age x +
  fourth_youngest_sibling_age x +
  fifth_youngest_sibling_age x +
  sixth_youngest_sibling_age x +
  seventh_youngest_sibling_age x +
  eldest_sibling_age x

-- Proving the combined age part
theorem eldest_sibling_age_correct (x : ℝ) (h : combined_age_of_eight_siblings x - youngest_sibling_age (x + 24) = 140) : 
  eldest_sibling_age x = 34.5 := by
  sorry

end eldest_sibling_age_correct_l127_127460


namespace arithmetic_seq_a7_value_l127_127838

theorem arithmetic_seq_a7_value {a : ℕ → ℝ} (h_positive : ∀ n, 0 < a n)
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_eq : 3 * a 6 - (a 7) ^ 2 + 3 * a 8 = 0) : a 7 = 6 :=
  sorry

end arithmetic_seq_a7_value_l127_127838


namespace base_8_to_10_conversion_l127_127088

theorem base_8_to_10_conversion : (2 * 8^4 + 3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 6 * 8^0) = 10030 := by 
  -- specify the summation directly 
  sorry

end base_8_to_10_conversion_l127_127088


namespace D_72_is_22_l127_127948

def D (n : ℕ) : ℕ :=
   -- function definition for D that satisfies the problem's conditions
   sorry

theorem D_72_is_22 : D 72 = 22 :=
by sorry

end D_72_is_22_l127_127948


namespace find_constants_and_intervals_l127_127651

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x^2 - 2 * x
def f' (x : ℝ) (a b : ℝ) := 3 * a * x^2 + 2 * b * x - 2

theorem find_constants_and_intervals :
  (f' (1 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (f' (-2 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) > 0 ↔ x < -2 ∨ x > 1) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) < 0 ↔ -2 < x ∧ x < 1) :=
by {
  sorry
}

end find_constants_and_intervals_l127_127651


namespace exponential_decreasing_range_l127_127613

theorem exponential_decreasing_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x y : ℝ, x < y → a^y < a^x) : 0 < a ∧ a < 1 :=
by sorry

end exponential_decreasing_range_l127_127613


namespace betty_boxes_l127_127463

theorem betty_boxes (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 24) (h2 : boxes_capacity = 8) : total_oranges / boxes_capacity = 3 :=
by sorry

end betty_boxes_l127_127463


namespace median_length_angle_bisector_length_l127_127014

variable (a b c : ℝ) (ma n : ℝ)

theorem median_length (h1 : ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4)) : 
  ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4) :=
by
  sorry

theorem angle_bisector_length (h2 : n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2)) :
  n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2) :=
by
  sorry

end median_length_angle_bisector_length_l127_127014


namespace smallest_positive_integer_ends_in_7_and_divisible_by_5_l127_127947

theorem smallest_positive_integer_ends_in_7_and_divisible_by_5 : 
  ∃ n : ℤ, n > 0 ∧ n % 10 = 7 ∧ n % 5 = 0 ∧ n = 37 := 
by 
  sorry

end smallest_positive_integer_ends_in_7_and_divisible_by_5_l127_127947


namespace dragons_total_games_l127_127968

noncomputable def numberOfGames (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) : ℕ :=
y + 12

theorem dragons_total_games (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) :
  numberOfGames y x h1 h2 = 90 := 
sorry

end dragons_total_games_l127_127968


namespace age_ratio_l127_127076

theorem age_ratio (B_current A_current B_10_years_ago A_in_10_years : ℕ) 
  (h1 : B_current = 37) 
  (h2 : A_current = B_current + 7) 
  (h3 : B_10_years_ago = B_current - 10) 
  (h4 : A_in_10_years = A_current + 10) : 
  A_in_10_years / B_10_years_ago = 2 :=
by
  sorry

end age_ratio_l127_127076


namespace negation_equiv_l127_127605

theorem negation_equiv {x : ℝ} : 
  (¬ (x^2 < 1 → -1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) :=
by
  sorry

end negation_equiv_l127_127605


namespace hyperbola_proof_l127_127832

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

def hyperbola_conditions (origin : ℝ × ℝ) (eccentricity : ℝ) (radius : ℝ) (focus : ℝ × ℝ) : Prop :=
  origin = (0, 0) ∧
  focus.1 = 0 ∧
  eccentricity = Real.sqrt 5 / 2 ∧
  radius = 2

theorem hyperbola_proof :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), hyperbola_conditions (0, 0) (Real.sqrt 5 / 2) 2 (0, c) → 
    C x y ↔ hyperbola_equation x y) :=
by
  sorry

end hyperbola_proof_l127_127832


namespace solution_set_l127_127559

-- Given conditions
variable (x : ℝ)

def inequality1 := 2 * x + 1 > 0
def inequality2 := (x + 1) / 3 > x - 1

-- The proof statement
theorem solution_set (h1 : inequality1 x) (h2 : inequality2 x) :
  -1 / 2 < x ∧ x < 2 :=
sorry

end solution_set_l127_127559


namespace triangle_inequality_l127_127251

variables {a b c : ℝ} {α : ℝ}

-- Assuming a, b, c are sides of a triangle
def triangle_sides (a b c : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Cosine rule definition
noncomputable def cos_alpha (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)

theorem triangle_inequality (h_sides: triangle_sides a b c) (h_cos : α = cos_alpha a b c) :
  (2 * b * c * (cos_alpha a b c)) / (b + c) < b + c - a
  ∧ b + c - a < 2 * b * c / a :=
by
  sorry

end triangle_inequality_l127_127251


namespace D_72_l127_127341

/-- D(n) denotes the number of ways of writing the positive integer n
    as a product n = f1 * f2 * ... * fk, where k ≥ 1, the fi are integers
    strictly greater than 1, and the order in which the factors are
    listed matters. -/
def D (n : ℕ) : ℕ := sorry

theorem D_72 : D 72 = 43 := sorry

end D_72_l127_127341


namespace find_k_min_value_quadratic_zero_l127_127659

theorem find_k_min_value_quadratic_zero (x y k : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 10 * x - 6 * y + 9 = 0) ↔ k = 1 :=
by
  sorry

end find_k_min_value_quadratic_zero_l127_127659


namespace inequality_x_y_z_l127_127239

open Real

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
    (x ^ 3) / ((1 + y) * (1 + z)) + (y ^ 3) / ((1 + z) * (1 + x)) + (z ^ 3) / ((1 + x) * (1 + y)) ≥ 3 / 4 :=
by
  sorry

end inequality_x_y_z_l127_127239


namespace find_max_min_find_angle_C_l127_127293

open Real

noncomputable def f (x : ℝ) : ℝ :=
  12 * sin (x + π / 6) * cos x - 3

theorem find_max_min (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) :
  let fx := f x 
  (∀ a, a = abs (fx - 6)) -> (∀ b, b = abs (fx - 3)) -> fx = 6 ∨ fx = 3 := sorry

theorem find_angle_C (AC BC CD : ℝ) (hAC : AC = 6) (hBC : BC = 3) (hCD : CD = 2 * sqrt 2) :
  ∃ C : ℝ, C = π / 2 := sorry

end find_max_min_find_angle_C_l127_127293


namespace add_words_to_meet_requirement_l127_127331

-- Definitions required by the problem
def yvonne_words : ℕ := 400
def janna_extra_words : ℕ := 150
def words_removed : ℕ := 20
def requirement : ℕ := 1000

-- Derived values based on the conditions
def janna_words : ℕ := yvonne_words + janna_extra_words
def initial_words : ℕ := yvonne_words + janna_words
def words_after_removal : ℕ := initial_words - words_removed
def words_added : ℕ := 2 * words_removed
def total_words_after_editing : ℕ := words_after_removal + words_added
def words_to_add : ℕ := requirement - total_words_after_editing

-- The theorem to prove
theorem add_words_to_meet_requirement : words_to_add = 30 := by
  sorry

end add_words_to_meet_requirement_l127_127331


namespace range_of_m_l127_127826

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + 4 = 0) → x > 1) ↔ (2 ≤ m ∧ m < 5/2) := sorry

end range_of_m_l127_127826
