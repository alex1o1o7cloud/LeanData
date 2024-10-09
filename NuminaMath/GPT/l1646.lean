import Mathlib

namespace cole_drive_to_work_time_l1646_164640

variables (D : ℝ) (T_work T_home : ℝ)

def speed_work : ℝ := 75
def speed_home : ℝ := 105
def total_time : ℝ := 2

theorem cole_drive_to_work_time :
  (T_work = D / speed_work) ∧
  (T_home = D / speed_home) ∧
  (T_work + T_home = total_time) →
  T_work * 60 = 70 :=
by
  sorry

end cole_drive_to_work_time_l1646_164640


namespace inequality_l1646_164650

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem inequality : c < a ∧ a < b := 
by 
  sorry

end inequality_l1646_164650


namespace binomial_square_l1646_164618

variable (c : ℝ)

theorem binomial_square (h : ∃ a : ℝ, (x^2 - 164 * x + c) = (x + a)^2) : c = 6724 := sorry

end binomial_square_l1646_164618


namespace bus_profit_problem_l1646_164671

def independent_variable := "number of passengers per month"
def dependent_variable := "monthly profit"

-- Given monthly profit equation
def monthly_profit (x : ℕ) : ℤ := 2 * x - 4000

-- 1. Independent and Dependent variables
def independent_variable_defined_correctly : Prop :=
  independent_variable = "number of passengers per month"

def dependent_variable_defined_correctly : Prop :=
  dependent_variable = "monthly profit"

-- 2. Minimum passenger volume to avoid losses
def minimum_passenger_volume_no_loss : Prop :=
  ∀ x : ℕ, (monthly_profit x >= 0) → (x >= 2000)

-- 3. Monthly profit prediction for 4230 passengers
def monthly_profit_prediction_4230 (x : ℕ) : Prop :=
  x = 4230 → monthly_profit x = 4460

theorem bus_profit_problem :
  independent_variable_defined_correctly ∧
  dependent_variable_defined_correctly ∧
  minimum_passenger_volume_no_loss ∧
  monthly_profit_prediction_4230 4230 :=
by
  sorry

end bus_profit_problem_l1646_164671


namespace exposed_sides_correct_l1646_164663

-- Define the number of sides of each polygon
def sides_triangle := 3
def sides_square := 4
def sides_pentagon := 5
def sides_hexagon := 6
def sides_heptagon := 7

-- Total sides from all polygons
def total_sides := sides_triangle + sides_square + sides_pentagon + sides_hexagon + sides_heptagon

-- Number of shared sides
def shared_sides := 4

-- Final number of exposed sides
def exposed_sides := total_sides - shared_sides

-- Statement to prove
theorem exposed_sides_correct : exposed_sides = 21 :=
by {
  -- This part will contain the proof which we do not need. Replace with 'sorry' for now.
  sorry
}

end exposed_sides_correct_l1646_164663


namespace B_A_equals_expectedBA_l1646_164649

noncomputable def MatrixA : Matrix (Fin 2) (Fin 2) ℝ := sorry
noncomputable def MatrixB : Matrix (Fin 2) (Fin 2) ℝ := sorry
def MatrixAB : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 1], ![-2, 4]]
def expectedBA : Matrix (Fin 2) (Fin 2) ℝ := ![![10, 2], ![-4, 8]]

theorem B_A_equals_expectedBA (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = MatrixAB) : 
  B * A = expectedBA := by
  sorry

end B_A_equals_expectedBA_l1646_164649


namespace equivalent_fractions_l1646_164617

variable {x y a c : ℝ}

theorem equivalent_fractions (h_nonzero_c : c ≠ 0) (h_transform : x = (a / c) * y) :
  (x + a) / (y + c) = a / c :=
by
  sorry

end equivalent_fractions_l1646_164617


namespace west_for_200_is_neg_200_l1646_164627

-- Given a definition for driving east
def driving_east (d : Int) : Int := d

-- Driving east for 80 km is +80 km
def driving_east_80 : Int := driving_east 80

-- Driving west should be the negative of driving east
def driving_west (d : Int) : Int := -d

-- Driving west for 200 km is -200 km
def driving_west_200 : Int := driving_west 200

-- Theorem to prove the given condition and expected result
theorem west_for_200_is_neg_200 : driving_west_200 = -200 :=
by
  -- Proof step is skipped
  sorry

end west_for_200_is_neg_200_l1646_164627


namespace range_of_a_l1646_164608

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then |x - 2 * a| else x + 1 / (x - 2) + a

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f a 2 ≤ f a x) : 1 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_a_l1646_164608


namespace find_horizontal_length_l1646_164624

variable (v h : ℝ)

-- Conditions
def is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 (v h : ℝ) : Prop :=
  2 * h + 2 * v = 54 ∧ h = v + 3

-- The proof we aim to show
theorem find_horizontal_length (v h : ℝ) :
  is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 v h → h = 15 :=
by
  sorry

end find_horizontal_length_l1646_164624


namespace minimize_S_n_at_7_l1646_164683

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 2 - a 1

def conditions (a : ℕ → ℤ) : Prop :=
arithmetic_sequence a ∧ a 2 = -11 ∧ (a 5 + a 9 = -2)

-- Define the sum of first n terms of the sequence
def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Define the minimum S_n and that it occurs at n = 7
theorem minimize_S_n_at_7 (a : ℕ → ℤ) (n : ℕ) (h : conditions a) :
  ∀ m, S a m ≥ S a 7 := sorry

end minimize_S_n_at_7_l1646_164683


namespace average_temperature_l1646_164629

def highTemps : List ℚ := [51, 60, 56, 55, 48, 63, 59]
def lowTemps : List ℚ := [42, 50, 44, 43, 41, 46, 45]

def dailyAverage (high low : ℚ) : ℚ :=
  (high + low) / 2

def averageOfAverages (tempsHigh tempsLow : List ℚ) : ℚ :=
  (List.sum (List.zipWith dailyAverage tempsHigh tempsLow)) / tempsHigh.length

theorem average_temperature :
  averageOfAverages highTemps lowTemps = 50.2 :=
  sorry

end average_temperature_l1646_164629


namespace sufficient_condition_ab_greater_than_1_l1646_164619

theorem sufficient_condition_ab_greater_than_1 (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : ab > 1 := 
  sorry

end sufficient_condition_ab_greater_than_1_l1646_164619


namespace inequality_solution_set_l1646_164605

open Set

noncomputable def rational_expression (x : ℝ) : ℝ := (x^2 - 16) / (x^2 + 10*x + 25)

theorem inequality_solution_set :
  {x : ℝ | rational_expression x < 0} = Ioo (-4 : ℝ) 4 :=
by
  sorry

end inequality_solution_set_l1646_164605


namespace watch_loss_percentage_l1646_164692

theorem watch_loss_percentage 
  (cost_price : ℕ) (gain_percent : ℕ) (extra_amount : ℕ) (selling_price_loss : ℕ)
  (h_cost_price : cost_price = 2500)
  (h_gain_percent : gain_percent = 10)
  (h_extra_amount : extra_amount = 500)
  (h_gain_condition : cost_price + gain_percent * cost_price / 100 = selling_price_loss + extra_amount) :
  (cost_price - selling_price_loss) * 100 / cost_price = 10 := 
by 
  sorry

end watch_loss_percentage_l1646_164692


namespace apricot_tea_calories_l1646_164697

theorem apricot_tea_calories :
  let apricot_juice_weight := 150
  let apricot_juice_calories_per_100g := 30
  let honey_weight := 50
  let honey_calories_per_100g := 304
  let water_weight := 300
  let apricot_tea_weight := apricot_juice_weight + honey_weight + water_weight
  let apricot_juice_calories := apricot_juice_weight * apricot_juice_calories_per_100g / 100
  let honey_calories := honey_weight * honey_calories_per_100g / 100
  let total_calories := apricot_juice_calories + honey_calories
  let caloric_density := total_calories / apricot_tea_weight
  let tea_weight := 250
  let calories_in_250g_tea := tea_weight * caloric_density
  calories_in_250g_tea = 98.5 := by
  sorry

end apricot_tea_calories_l1646_164697


namespace number_of_goats_l1646_164648

-- Mathematical definitions based on the conditions
def number_of_hens : ℕ := 10
def total_cost : ℤ := 2500
def price_per_hen : ℤ := 50
def price_per_goat : ℤ := 400

-- Prove the number of goats
theorem number_of_goats (G : ℕ) : 
  number_of_hens * price_per_hen + G * price_per_goat = total_cost ↔ G = 5 := 
by
  sorry

end number_of_goats_l1646_164648


namespace simplify_expression_l1646_164656

theorem simplify_expression (x : ℤ) : 120 * x - 55 * x = 65 * x := by
  sorry

end simplify_expression_l1646_164656


namespace mabel_petals_remaining_l1646_164652

/-- Mabel has 5 daisies, each with 8 petals. If she gives 2 daisies to her teacher,
how many petals does she have on the remaining daisies in her garden? -/
theorem mabel_petals_remaining :
  (5 - 2) * 8 = 24 :=
by
  sorry

end mabel_petals_remaining_l1646_164652


namespace roots_product_eq_three_l1646_164661

theorem roots_product_eq_three
  (p q r : ℝ)
  (h : (3:ℝ) * p ^ 3 - 8 * p ^ 2 + p - 9 = 0 ∧
       (3:ℝ) * q ^ 3 - 8 * q ^ 2 + q - 9 = 0 ∧
       (3:ℝ) * r ^ 3 - 8 * r ^ 2 + r - 9 = 0) :
  p * q * r = 3 :=
sorry

end roots_product_eq_three_l1646_164661


namespace cut_square_into_rectangles_l1646_164621

theorem cut_square_into_rectangles :
  ∃ x y : ℕ, 3 * x + 4 * y = 25 :=
by
  -- Given that the total area is 25 and we are using rectangles of areas 3 and 4
  -- we need to verify the existence of integers x and y such that 3x + 4y = 25
  existsi 7
  existsi 1
  sorry

end cut_square_into_rectangles_l1646_164621


namespace correct_operation_l1646_164634

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l1646_164634


namespace intersection_M_N_complement_N_U_l1646_164670

-- Definitions for the sets and the universal set
def U := Set ℝ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def N : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) } -- Simplified domain interpretation for N

-- Intersection and complement calculations
theorem intersection_M_N (x : ℝ) : x ∈ M ∧ x ∈ N ↔ x ∈ { x | -2 ≤ x ∧ x ≤ 1 } := by sorry

theorem complement_N_U (x : ℝ) : x ∉ N ↔ x ∈ { x | x > 1 } := by sorry

end intersection_M_N_complement_N_U_l1646_164670


namespace predicted_whales_l1646_164685

theorem predicted_whales (num_last_year num_this_year num_next_year : ℕ)
  (h1 : num_this_year = 2 * num_last_year)
  (h2 : num_last_year = 4000)
  (h3 : num_next_year = 8800) :
  num_next_year - num_this_year = 800 :=
by
  sorry

end predicted_whales_l1646_164685


namespace div_problem_l1646_164684

theorem div_problem (a b c : ℝ) (h1 : a / (b * c) = 4) (h2 : (a / b) / c = 12) : a / b = 4 * Real.sqrt 3 := 
by
  sorry

end div_problem_l1646_164684


namespace kids_on_Monday_l1646_164600

-- Defining the conditions
def kidsOnTuesday : ℕ := 10
def difference : ℕ := 8

-- Formulating the theorem to prove the number of kids Julia played with on Monday
theorem kids_on_Monday : kidsOnTuesday + difference = 18 := by
  sorry

end kids_on_Monday_l1646_164600


namespace teacher_works_days_in_month_l1646_164628

theorem teacher_works_days_in_month (P : ℕ) (W : ℕ) (M : ℕ) (T : ℕ) (H1 : P = 5) (H2 : W = 5) (H3 : M = 6) (H4 : T = 3600) : 
  (T / M) / (P * W) = 24 :=
by
  sorry

end teacher_works_days_in_month_l1646_164628


namespace expand_expression_l1646_164633

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end expand_expression_l1646_164633


namespace find_P_l1646_164603

variable (a b c d P : ℝ)

theorem find_P 
  (h1 : (a + b + c + d) / 4 = 8) 
  (h2 : (a + b + c + d + P) / 5 = P) : 
  P = 8 := 
by 
  sorry

end find_P_l1646_164603


namespace log_101600_l1646_164677

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_101600 (h : log_base_10 102 = 0.3010) : log_base_10 101600 = 2.3010 :=
by
  sorry

end log_101600_l1646_164677


namespace problem_I_solution_set_l1646_164615

def f1 (x : ℝ) : ℝ := |2 * x| + |x - 1| -- since a = -1

theorem problem_I_solution_set :
  {x : ℝ | f1 x ≤ 4} = Set.Icc (-1 : ℝ) ((5 : ℝ) / 3) :=
sorry

end problem_I_solution_set_l1646_164615


namespace reflect_triangle_final_position_l1646_164606

variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Definition of reflection in x-axis and y-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

theorem reflect_triangle_final_position (x1 x2 x3 y1 y2 y3 : ℝ) :
  (reflect_y (reflect_x x1 y1).1 (reflect_x x1 y1).2) = (-x1, -y1) ∧
  (reflect_y (reflect_x x2 y2).1 (reflect_x x2 y2).2) = (-x2, -y2) ∧
  (reflect_y (reflect_x x3 y3).1 (reflect_x x3 y3).2) = (-x3, -y3) :=
by
  sorry

end reflect_triangle_final_position_l1646_164606


namespace intersection_complement_l1646_164687

open Set

def U : Set ℝ := univ
def A : Set ℤ := {x : ℤ | -3 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement (U A B : Set ℝ) : A ∩ (U \ B) = {0, 1} := sorry

end intersection_complement_l1646_164687


namespace arithmetic_sequence_a3_l1646_164676

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end arithmetic_sequence_a3_l1646_164676


namespace rachel_age_when_emily_half_age_l1646_164639

-- Conditions
def Emily_current_age : ℕ := 20
def Rachel_current_age : ℕ := 24

-- Proof statement
theorem rachel_age_when_emily_half_age :
  ∃ x : ℕ, (Emily_current_age - x = (Rachel_current_age - x) / 2) ∧ (Rachel_current_age - x = 8) := 
sorry

end rachel_age_when_emily_half_age_l1646_164639


namespace temperature_difference_l1646_164655

def h : ℤ := 10
def l : ℤ := -5
def d : ℤ := 15

theorem temperature_difference : h - l = d :=
by
  rw [h, l, d]
  sorry

end temperature_difference_l1646_164655


namespace geometric_series_first_term_l1646_164699

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 24)
  (h_sum : S = a / (1 - r)) : 
  a = 18 :=
by {
  -- valid proof body goes here
  sorry
}

end geometric_series_first_term_l1646_164699


namespace smallest_n_square_partition_l1646_164642

theorem smallest_n_square_partition (n : ℕ) (h : ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧ n = 40 * a + 49 * b) : n ≥ 2000 :=
by sorry

end smallest_n_square_partition_l1646_164642


namespace solution_set_of_inequality_l1646_164612

theorem solution_set_of_inequality (x : ℝ) :
  |x^2 - 2| < 2 ↔ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l1646_164612


namespace inequality_satisfaction_l1646_164644

theorem inequality_satisfaction (x y : ℝ) : 
  y - x < Real.sqrt (x^2) ↔ (y < 0 ∨ y < 2 * x) := by 
sorry

end inequality_satisfaction_l1646_164644


namespace problem_statement_l1646_164609

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

variable (x : ℝ)

theorem problem_statement
  (h : f_iter 13 x = f_iter 31 x) :
  f_iter 16 x = (x - 1) / x :=
by
  sorry

end problem_statement_l1646_164609


namespace area_of_farm_l1646_164666

theorem area_of_farm (W L : ℝ) (hW : W = 30) 
  (hL_fence_cost : 14 * (L + W + Real.sqrt (L^2 + W^2)) = 1680) : 
  W * L = 1200 :=
by
  sorry -- Proof not required

end area_of_farm_l1646_164666


namespace canoe_upstream_speed_l1646_164668

namespace canoe_speed

def V_c : ℝ := 12.5            -- speed of the canoe in still water in km/hr
def V_downstream : ℝ := 16     -- speed of the canoe downstream in km/hr

theorem canoe_upstream_speed :
  ∃ (V_upstream : ℝ), V_upstream = V_c - (V_downstream - V_c) ∧ V_upstream = 9 := by
  sorry

end canoe_speed

end canoe_upstream_speed_l1646_164668


namespace workers_number_l1646_164623

theorem workers_number (W A : ℕ) (h1 : W * 25 = A) (h2 : (W + 10) * 15 = A) : W = 15 :=
by
  sorry

end workers_number_l1646_164623


namespace problem_statement_l1646_164610

noncomputable def percent_of_y (y : ℝ) (z : ℂ) : ℝ :=
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10).re

theorem problem_statement (y : ℝ) (z : ℂ) (hy : y > 0) : percent_of_y y z = 0.6 * y :=
by
  sorry

end problem_statement_l1646_164610


namespace perpendicular_condition_sufficient_not_necessary_l1646_164611

theorem perpendicular_condition_sufficient_not_necessary (m : ℝ) :
  (∀ x y : ℝ, m * x + (2 * m - 1) * y + 1 = 0) →
  (∀ x y : ℝ, 3 * x + m * y + 3 = 0) →
  (∀ a b : ℝ, m = -1 → (∃ c d : ℝ, 3 / a = 1 / b)) →
  (m = -1 → (m = -1 → (3 / (-m / (2 * m - 1)) * m) / 2 - (3 / m) = -1)) :=
by sorry

end perpendicular_condition_sufficient_not_necessary_l1646_164611


namespace apple_cost_is_2_l1646_164669

def total_spent (hummus_cost chicken_cost bacon_cost vegetable_cost : ℕ) : ℕ :=
  2 * hummus_cost + chicken_cost + bacon_cost + vegetable_cost

theorem apple_cost_is_2 :
  ∀ (hummus_cost chicken_cost bacon_cost vegetable_cost total_money apples_cost : ℕ),
    hummus_cost = 5 →
    chicken_cost = 20 →
    bacon_cost = 10 →
    vegetable_cost = 10 →
    total_money = 60 →
    apples_cost = 5 →
    (total_money - total_spent hummus_cost chicken_cost bacon_cost vegetable_cost) / apples_cost = 2 :=
by
  intros
  sorry

end apple_cost_is_2_l1646_164669


namespace function_has_zero_in_interval_l1646_164695

   theorem function_has_zero_in_interval (fA fB fC fD : ℝ → ℝ) (hA : ∀ x, fA x = x - 3)
       (hB : ∀ x, fB x = 2^x) (hC : ∀ x, fC x = x^2) (hD : ∀ x, fD x = Real.log x) :
       ∃ x, 0 < x ∧ x < 2 ∧ fD x = 0 :=
   by
       sorry
   
end function_has_zero_in_interval_l1646_164695


namespace pascal_triangle_ratio_l1646_164696

theorem pascal_triangle_ratio (n r : ℕ) (hn1 : 5 * r = 2 * n - 3) (hn2 : 7 * r = 3 * n - 11) : n = 34 :=
by
  -- The proof steps will fill here eventually
  sorry

end pascal_triangle_ratio_l1646_164696


namespace sum_of_remainders_l1646_164638

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9) = 3 :=
by
  sorry

end sum_of_remainders_l1646_164638


namespace spaceship_not_moving_time_l1646_164674

-- Definitions based on the conditions given
def total_journey_time : ℕ := 3 * 24  -- 3 days in hours

def first_travel_time : ℕ := 10
def first_break_time : ℕ := 3
def second_travel_time : ℕ := 10
def second_break_time : ℕ := 1

def subsequent_travel_period : ℕ := 11  -- 11 hours traveling, then 1 hour break

-- Function to compute total break time
def total_break_time (total_travel_time : ℕ) : ℕ :=
  let remaining_time := total_journey_time - (first_travel_time + first_break_time + second_travel_time + second_break_time)
  let subsequent_breaks := remaining_time / subsequent_travel_period
  first_break_time + second_break_time + subsequent_breaks

theorem spaceship_not_moving_time : total_break_time total_journey_time = 8 := by
  sorry

end spaceship_not_moving_time_l1646_164674


namespace inequality_solution_l1646_164690

noncomputable def f (a b x : ℝ) : ℝ := 1 / Real.sqrt x + 1 / Real.sqrt (a + b - x)

theorem inequality_solution 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (x : ℝ) 
  (hx : x ∈ Set.Ioo (min a b) (max a b)) : 
  f a b x < f a b a ∧ f a b x < f a b b := 
sorry

end inequality_solution_l1646_164690


namespace find_y_value_l1646_164679

theorem find_y_value (k : ℝ) (x y : ℝ) (h1 : y = k * x^(1/5)) (h2 : y = 4) (h3 : x = 32) :
  y = 6 := by
  sorry

end find_y_value_l1646_164679


namespace tan_x_eq_sqrt3_intervals_of_monotonic_increase_l1646_164636

noncomputable def m (x : ℝ) : ℝ × ℝ :=
  (Real.sin (x - Real.pi / 6), 1)

noncomputable def n (x : ℝ) : ℝ × ℝ :=
  (Real.cos x, 1)

noncomputable def f (x : ℝ) : ℝ :=
  (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Proof for part 1
theorem tan_x_eq_sqrt3 (x : ℝ) (h₀ : m x = n x) : Real.tan x = Real.sqrt 3 :=
sorry

-- Proof for part 2
theorem intervals_of_monotonic_increase (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) ↔ 
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) :=
sorry

end tan_x_eq_sqrt3_intervals_of_monotonic_increase_l1646_164636


namespace find_n_l1646_164694

noncomputable def n : ℕ := sorry -- Explicitly define n as a variable, but the value is not yet provided.

theorem find_n (h₁ : n > 0)
    (h₂ : Real.sqrt 3 > (n + 4) / (n + 1))
    (h₃ : Real.sqrt 3 < (n + 3) / n) : 
    n = 4 :=
sorry

end find_n_l1646_164694


namespace point_cannot_exist_on_line_l1646_164626

theorem point_cannot_exist_on_line (m k : ℝ) (h : m * k > 0) : ¬ (2000 * m + k = 0) :=
sorry

end point_cannot_exist_on_line_l1646_164626


namespace a_10_value_l1646_164654

-- Definitions for the initial conditions and recurrence relation.
def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧
  ∀ n, a (n + 1) = (8 / 5) * a n + (6 / 5) * (Real.sqrt (4 ^ n - a n ^ 2))

-- Statement that proves a_10 = 24576 / 25 given the conditions.
theorem a_10_value (a : ℕ → ℝ) (h : seq a) : a 10 = 24576 / 25 :=
by
  sorry

end a_10_value_l1646_164654


namespace total_amount_paid_l1646_164673

def jacket_price : ℝ := 150
def sale_discount : ℝ := 0.25
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_amount_paid : 
  (jacket_price * (1 - sale_discount) - coupon_discount) * (1 + sales_tax) = 112.75 := 
by
  sorry

end total_amount_paid_l1646_164673


namespace x_coordinate_point_P_l1646_164680

theorem x_coordinate_point_P (x y : ℝ) (h_on_parabola : y^2 = 4 * x) 
  (h_distance : dist (x, y) (1, 0) = 3) : x = 2 :=
sorry

end x_coordinate_point_P_l1646_164680


namespace max_cookies_without_ingredients_l1646_164620

-- Defining the number of cookies and their composition
def total_cookies : ℕ := 36
def peanuts : ℕ := (2 * total_cookies) / 3
def chocolate_chips : ℕ := total_cookies / 3
def raisins : ℕ := total_cookies / 4
def oats : ℕ := total_cookies / 8

-- Proving the largest number of cookies without any ingredients
theorem max_cookies_without_ingredients : (total_cookies - (max (max peanuts chocolate_chips) raisins)) = 12 := by
    sorry

end max_cookies_without_ingredients_l1646_164620


namespace sticker_price_l1646_164631

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end sticker_price_l1646_164631


namespace cyclic_sum_inequality_l1646_164602

theorem cyclic_sum_inequality (x y z : ℝ) (hp : x > 0 ∧ y > 0 ∧ z > 0) (h : x + y + z = 3) : 
  (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) < (3 + x * y + y * z + z * x) := by
  sorry

end cyclic_sum_inequality_l1646_164602


namespace original_selling_price_l1646_164651

theorem original_selling_price (P : ℝ) (h1 : ∀ P, 1.17 * P = 1.10 * P + 42) :
    1.10 * P = 660 := by
  sorry

end original_selling_price_l1646_164651


namespace smallest_y_l1646_164662

theorem smallest_y (y : ℕ) : (27^y > 3^24) ↔ (y ≥ 9) :=
sorry

end smallest_y_l1646_164662


namespace original_pencils_l1646_164657

-- Define the conditions given in the problem
variable (total_pencils_now : ℕ) [DecidableEq ℕ] (pencils_by_Mike : ℕ)

-- State the problem to prove
theorem original_pencils (h1 : total_pencils_now = 71) (h2 : pencils_by_Mike = 30) : total_pencils_now - pencils_by_Mike = 41 := by
  sorry

end original_pencils_l1646_164657


namespace conditional_probability_A_given_B_l1646_164658

noncomputable def P (A B : Prop) : ℝ := sorry -- Placeholder for the probability function

variables (A B : Prop)

axiom P_A_def : P A = 4/15
axiom P_B_def : P B = 2/15
axiom P_AB_def : P (A ∧ B) = 1/10

theorem conditional_probability_A_given_B : P (A ∧ B) / P B = 3/4 :=
by
  rw [P_AB_def, P_B_def]
  norm_num
  sorry

end conditional_probability_A_given_B_l1646_164658


namespace find_number_l1646_164693

theorem find_number (n : ℕ) (h1 : 45 = 11 * n + 1) : n = 4 :=
  sorry

end find_number_l1646_164693


namespace intersection_point_l1646_164665

theorem intersection_point : 
  ∃ (x y : ℚ), y = - (5/3 : ℚ) * x ∧ y + 3 = 15 * x - 6 ∧ x = 27 / 50 ∧ y = - 9 / 10 := 
by
  sorry

end intersection_point_l1646_164665


namespace find_n_in_permutation_combination_equation_l1646_164653

-- Lean statement for the proof problem

theorem find_n_in_permutation_combination_equation :
  ∃ (n : ℕ), (n > 0) ∧ (Nat.factorial 8 / Nat.factorial (8 - n) = 2 * (Nat.factorial 8 / (Nat.factorial 2 * Nat.factorial 6)))
  := sorry

end find_n_in_permutation_combination_equation_l1646_164653


namespace solve_for_k_l1646_164614

theorem solve_for_k (x y k : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : (1 / 2)^(25 * x) * (1 / 81)^k = 1 / (18 ^ (25 * y))) :
  k = 25 * y / 2 :=
by
  sorry

end solve_for_k_l1646_164614


namespace exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l1646_164689

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared :
  ∃ (n : ℕ), sum_of_digits n = 1000 ∧ sum_of_digits (n ^ 2) = 1000000 := sorry

end exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l1646_164689


namespace find_a_l1646_164625

theorem find_a (a x : ℝ) (h : x = -1) (heq : -2 * (x - a) = 4) : a = 1 :=
by
  sorry

end find_a_l1646_164625


namespace children_tickets_sold_l1646_164681

-- Given conditions
variables (A C : ℕ) -- A represents the number of adult tickets, C the number of children tickets.
variables (total_money total_tickets price_adult price_children : ℕ)
variables (total_money_eq : total_money = 104)
variables (total_tickets_eq : total_tickets = 21)
variables (price_adult_eq : price_adult = 6)
variables (price_children_eq : price_children = 4)
variables (money_eq : price_adult * A + price_children * C = total_money)
variables (tickets_eq : A + C = total_tickets)

-- Problem statement: prove that C = 11
theorem children_tickets_sold : C = 11 :=
by
  -- Necessary Lean code to handle proof here (omitting proof details as instructed)
  sorry

end children_tickets_sold_l1646_164681


namespace minimal_height_exists_l1646_164641

noncomputable def height_min_material (x : ℝ) : ℝ := 4 / (x^2)

theorem minimal_height_exists
  (x h : ℝ)
  (volume_cond : x^2 * h = 4)
  (surface_area_cond : h = height_min_material x) :
  h = 1 := by
  sorry

end minimal_height_exists_l1646_164641


namespace midpoint_one_sixth_one_twelfth_l1646_164698

theorem midpoint_one_sixth_one_twelfth : (1 : ℚ) / 8 = (1 / 6 + 1 / 12) / 2 := by
  sorry

end midpoint_one_sixth_one_twelfth_l1646_164698


namespace sum_coordinates_l1646_164613

theorem sum_coordinates (x : ℝ) : 
  let C := (x, 8)
  let D := (-x, 8)
  (C.1 + C.2 + D.1 + D.2) = 16 := 
by
  sorry

end sum_coordinates_l1646_164613


namespace max_cigarettes_with_staggered_packing_l1646_164682

theorem max_cigarettes_with_staggered_packing :
  ∃ n : ℕ, n > 160 ∧ n = 176 :=
by
  let diameter := 2
  let rows_initial := 8
  let cols_initial := 20
  let total_initial := rows_initial * cols_initial
  have h1 : total_initial = 160 := by norm_num
  let alternative_packing_capacity := 176
  have h2 : alternative_packing_capacity > total_initial := by norm_num
  use alternative_packing_capacity
  exact ⟨h2, rfl⟩

end max_cigarettes_with_staggered_packing_l1646_164682


namespace value_of_b_l1646_164659

theorem value_of_b (a b : ℕ) (q : ℝ)
  (h1 : q = 0.5)
  (h2 : a = 2020)
  (h3 : q = a / b) : b = 4040 := by
  sorry

end value_of_b_l1646_164659


namespace vector_subtraction_result_l1646_164604

-- definition of vectors as pairs of integers
def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

-- definition of vector subtraction for pairs of reals
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- definition of the vector AB as the subtraction of OB and OA
def AB : ℝ × ℝ := vector_sub OB OA

-- statement to assert the expected result
theorem vector_subtraction_result : AB = (-4, 3) :=
by
  -- this is where the proof would go, but we use sorry to skip it
  sorry

end vector_subtraction_result_l1646_164604


namespace proof_problem_l1646_164647

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l1646_164647


namespace find_a_l1646_164686

theorem find_a (a : ℝ) :
  {x : ℝ | (x + a) / ((x + 1) * (x + 3)) > 0} = {x : ℝ | x > -3 ∧ x ≠ -1} →
  a = 1 := 
by sorry

end find_a_l1646_164686


namespace simplify_expression_l1646_164691

noncomputable def simplify_fraction (x : ℝ) (h : x ≠ 2) : ℝ :=
  (1 + (1 / (x - 2))) / ((x - x^2) / (x - 2))

theorem simplify_expression (x : ℝ) (h : x ≠ 2) : simplify_fraction x h = -(x - 1) / x :=
  sorry

end simplify_expression_l1646_164691


namespace percentage_difference_liliane_alice_l1646_164664

theorem percentage_difference_liliane_alice :
  let J := 200
  let L := 1.30 * J
  let A := 1.15 * J
  (L - A) / A * 100 = 13.04 :=
by
  sorry

end percentage_difference_liliane_alice_l1646_164664


namespace new_average_weight_l1646_164645

def average_weight (A B C D E : ℝ) : Prop :=
  (A + B + C) / 3 = 70 ∧
  (A + B + C + D) / 4 = 70 ∧
  E = D + 3 ∧
  A = 81

theorem new_average_weight (A B C D E : ℝ) (h: average_weight A B C D E) : 
  (B + C + D + E) / 4 = 68 :=
by
  sorry

end new_average_weight_l1646_164645


namespace probability_A_wins_championship_distribution_and_expectation_B_l1646_164643

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l1646_164643


namespace intersection_eq_l1646_164630

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_eq : A ∩ B = {2, 4, 8} := by
  sorry

end intersection_eq_l1646_164630


namespace binders_can_bind_books_l1646_164632

theorem binders_can_bind_books :
  (∀ (binders books days : ℕ), binders * days * books = 18 * 10 * 900 → 
    11 * binders * 12 = 660) :=
sorry

end binders_can_bind_books_l1646_164632


namespace students_exam_percentage_l1646_164616

theorem students_exam_percentage 
  (total_students : ℕ) 
  (avg_assigned_day : ℚ) 
  (avg_makeup_day : ℚ)
  (overall_avg : ℚ) 
  (h_total : total_students = 100)
  (h_avg_assigned_day : avg_assigned_day = 0.60) 
  (h_avg_makeup_day : avg_makeup_day = 0.80) 
  (h_overall_avg : overall_avg = 0.66) : 
  ∃ x : ℚ, x = 70 / 100 :=
by
  sorry

end students_exam_percentage_l1646_164616


namespace algebraic_expression_evaluation_l1646_164635

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + 3 * x - 5 = 2) : 2 * x^2 + 6 * x - 3 = 11 :=
sorry

end algebraic_expression_evaluation_l1646_164635


namespace evaluate_expression_l1646_164678

theorem evaluate_expression (a : ℝ) (h : a = 4 / 3) : 
  (4 * a^2 - 12 * a + 9) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l1646_164678


namespace part1_part2_l1646_164675

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end part1_part2_l1646_164675


namespace rectangle_area_l1646_164667

theorem rectangle_area (x : ℝ) (w : ℝ) (h1 : (3 * w)^2 + w^2 = x^2) : (3 * w) * w = 3 * x^2 / 10 :=
by
  sorry

end rectangle_area_l1646_164667


namespace mother_daughter_ages_l1646_164607

theorem mother_daughter_ages :
  ∃ (x y : ℕ), (y = x + 22) ∧ (2 * x = (x + 22) - x) ∧ (x = 11) ∧ (y = 33) :=
by
  sorry

end mother_daughter_ages_l1646_164607


namespace eval_expression_l1646_164688

theorem eval_expression : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 :=
by
  sorry

end eval_expression_l1646_164688


namespace tangent_line_and_point_l1646_164672

theorem tangent_line_and_point (x0 y0 k: ℝ) (hx0 : x0 ≠ 0) 
  (hC : y0 = x0^3 - 3 * x0^2 + 2 * x0) (hl : y0 = k * x0) 
  (hk_tangent : k = 3 * x0^2 - 6 * x0 + 2) : 
  (k = -1/4) ∧ (x0 = 3/2) ∧ (y0 = -3/8) :=
by
  sorry

end tangent_line_and_point_l1646_164672


namespace markup_percent_based_on_discounted_price_l1646_164601

-- Defining the conditions
def original_price : ℝ := 1
def discount_percent : ℝ := 0.2
def discounted_price : ℝ := original_price * (1 - discount_percent)

-- The proof problem statement
theorem markup_percent_based_on_discounted_price :
  (original_price - discounted_price) / discounted_price = 0.25 :=
sorry

end markup_percent_based_on_discounted_price_l1646_164601


namespace knicks_eq_knocks_l1646_164660

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end knicks_eq_knocks_l1646_164660


namespace timothy_total_cost_l1646_164637

-- Define the costs of the individual items
def costOfLand (acres : Nat) (cost_per_acre : Nat) : Nat :=
  acres * cost_per_acre

def costOfHouse : Nat :=
  120000

def costOfCows (number_of_cows : Nat) (cost_per_cow : Nat) : Nat :=
  number_of_cows * cost_per_cow

def costOfChickens (number_of_chickens : Nat) (cost_per_chicken : Nat) : Nat :=
  number_of_chickens * cost_per_chicken

def installationCost (hours : Nat) (cost_per_hour : Nat) (equipment_fee : Nat) : Nat :=
  (hours * cost_per_hour) + equipment_fee

-- Define the total cost function
def totalCost : Nat :=
  costOfLand 30 20 +
  costOfHouse +
  costOfCows 20 1000 +
  costOfChickens 100 5 +
  installationCost 6 100 6000

-- Theorem to state the total cost
theorem timothy_total_cost : totalCost = 147700 :=
by
  -- Placeholder for the proof, for now leave it as sorry
  sorry

end timothy_total_cost_l1646_164637


namespace polynomial_sum_l1646_164646

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l1646_164646


namespace range_of_m_l1646_164622

def positive_numbers (a b : ℝ) : Prop := a > 0 ∧ b > 0

def equation_condition (a b : ℝ) : Prop := 9 * a + b = a * b

def inequality_for_any_x (a b m : ℝ) : Prop := ∀ x : ℝ, a + b ≥ -x^2 + 2 * x + 18 - m

theorem range_of_m :
  ∀ (a b m : ℝ),
    positive_numbers a b →
    equation_condition a b →
    inequality_for_any_x a b m →
    m ≥ 3 :=
by
  sorry

end range_of_m_l1646_164622
