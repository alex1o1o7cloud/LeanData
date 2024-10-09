import Mathlib

namespace chess_amateurs_play_with_l1878_187876

theorem chess_amateurs_play_with :
  ∃ n : ℕ, ∃ total_players : ℕ, total_players = 6 ∧
  (total_players * (total_players - 1)) / 2 = 12 ∧
  (n = total_players - 1 ∧ n = 5) :=
by
  sorry

end chess_amateurs_play_with_l1878_187876


namespace fair_coin_toss_consecutive_heads_l1878_187814

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem fair_coin_toss_consecutive_heads :
  let total_outcomes := 1024
  let favorable_outcomes := 
    1 + binom 10 1 + binom 9 2 + binom 8 3 + binom 7 4 + binom 6 5
  let prob := favorable_outcomes / total_outcomes
  let i := 9
  let j := 64
  Nat.gcd i j = 1 ∧ (prob = i / j) ∧ i + j = 73 :=
by
  sorry

end fair_coin_toss_consecutive_heads_l1878_187814


namespace volume_of_larger_cube_l1878_187850

theorem volume_of_larger_cube (s : ℝ) (V : ℝ) :
  (∀ (n : ℕ), n = 125 →
    ∀ (v_sm : ℝ), v_sm = 1 →
    V = n * v_sm →
    V = s^3 →
    s = 5 →
    ∀ (sa_large : ℝ), sa_large = 6 * s^2 →
    sa_large = 150 →
    ∀ (sa_sm_total : ℝ), sa_sm_total = n * (6 * v_sm^(2/3)) →
    sa_sm_total = 750 →
    sa_sm_total - sa_large = 600 →
    V = 125) :=
by
  intros n n125 v_sm v1 Vdef Vcube sc5 sa_large sa_large_def sa_large150 sa_sm_total sa_sm_total_def sa_sm_total750 diff600
  simp at *
  sorry

end volume_of_larger_cube_l1878_187850


namespace no_real_roots_range_l1878_187815

theorem no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 :=
by
  sorry

end no_real_roots_range_l1878_187815


namespace members_even_and_divisible_l1878_187872

structure ClubMember (α : Type) := 
  (friend : α) 
  (enemy : α)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

def can_be_divided_into_two_subclubs (members : List (ClubMember Nat)) : Prop :=
sorry -- Definition of dividing into two subclubs here

theorem members_even_and_divisible (members : List (ClubMember Nat)) :
  is_even members.length ∧ can_be_divided_into_two_subclubs members :=
sorry

end members_even_and_divisible_l1878_187872


namespace sarah_dimes_l1878_187852

theorem sarah_dimes (d n : ℕ) (h1 : d + n = 50) (h2 : 10 * d + 5 * n = 200) : d = 10 :=
sorry

end sarah_dimes_l1878_187852


namespace student_score_is_64_l1878_187869

-- Define the total number of questions and correct responses.
def total_questions : ℕ := 100
def correct_responses : ℕ := 88

-- Function to calculate the score based on the grading rule.
def calculate_score (total : ℕ) (correct : ℕ) : ℕ :=
  correct - 2 * (total - correct)

-- The theorem that states the score for the given conditions.
theorem student_score_is_64 :
  calculate_score total_questions correct_responses = 64 :=
by
  sorry

end student_score_is_64_l1878_187869


namespace total_animals_correct_l1878_187834

def L := 10

def C := 2 * L + 4

def Merry_lambs := L
def Merry_cows := C
def Merry_pigs (P : ℕ) := P
def Brother_lambs := L + 3

def Brother_chickens (R : ℕ) := R * Brother_lambs
def Brother_goats (Q : ℕ) := 2 * Brother_lambs + Q

def Merry_total (P : ℕ) := Merry_lambs + Merry_cows + Merry_pigs P
def Brother_total (R Q : ℕ) := Brother_lambs + Brother_chickens R + Brother_goats Q

def Total_animals (P R Q : ℕ) := Merry_total P + Brother_total R Q

theorem total_animals_correct (P R Q : ℕ) : 
  Total_animals P R Q = 73 + P + R * 13 + Q := by
  sorry

end total_animals_correct_l1878_187834


namespace parabola_vertex_relationship_l1878_187848

theorem parabola_vertex_relationship (m x y : ℝ) :
  (y = x^2 - 2*m*x + 2*m^2 - 3*m + 1) → (y = x^2 - 3*x + 1) :=
by
  intro h
  sorry

end parabola_vertex_relationship_l1878_187848


namespace find_a_values_l1878_187891

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l1878_187891


namespace P_has_no_negative_roots_but_at_least_one_positive_root_l1878_187812

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

-- Statement of the problem
theorem P_has_no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → P x ≠ 0 ∧ P x > 0) ∧ (∃ x : ℝ, x > 0 ∧ P x = 0) :=
by
  sorry

end P_has_no_negative_roots_but_at_least_one_positive_root_l1878_187812


namespace acute_angle_sum_equals_pi_over_two_l1878_187829

theorem acute_angle_sum_equals_pi_over_two (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
  (h1 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 1)
  (h2 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0) :
  2 * a + b = π / 2 := 
sorry

end acute_angle_sum_equals_pi_over_two_l1878_187829


namespace problem_statement_l1878_187824

theorem problem_statement :
  ∀ m n : ℕ, (m = 9) → (n = m^2 + 1) → n - m = 73 :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end problem_statement_l1878_187824


namespace number_of_green_fish_l1878_187884

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end number_of_green_fish_l1878_187884


namespace cost_of_computer_game_is_90_l1878_187810

-- Define the costs of individual items
def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

-- Define the number of items
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_quantity : ℕ := 1

-- Calculate the total cost before rebate
def total_cost_before_rebate : ℕ :=
  total_cost_after_rebate + rebate

-- Calculate the total cost of polo shirts and necklaces
def total_cost_polo_necklaces : ℕ :=
  (polo_shirt_quantity * polo_shirt_price) + (necklace_quantity * necklace_price)

-- Define the unknown cost of the computer game
def computer_game_price : ℕ :=
  total_cost_before_rebate - total_cost_polo_necklaces

-- Prove the cost of the computer game
theorem cost_of_computer_game_is_90 : computer_game_price = 90 := by
  -- The following line is a placeholder for the actual proof
  sorry

end cost_of_computer_game_is_90_l1878_187810


namespace machine_Y_produces_more_widgets_l1878_187831

-- Definitions for the rates and widgets produced
def W_x := 18 -- widgets per hour by machine X
def total_widgets := 1080

-- Calculations for time taken by each machine
def T_x := total_widgets / W_x -- time taken by machine X
def T_y := T_x - 10 -- machine Y takes 10 hours less

-- Rate at which machine Y produces widgets
def W_y := total_widgets / T_y

-- Calculation of percentage increase
def percentage_increase := (W_y - W_x) / W_x * 100

-- The final theorem to prove
theorem machine_Y_produces_more_widgets : percentage_increase = 20 := by
  sorry

end machine_Y_produces_more_widgets_l1878_187831


namespace divides_lcm_condition_l1878_187889

theorem divides_lcm_condition (x y : ℕ) (h₀ : 1 < x) (h₁ : 1 < y)
  (h₂ : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x := 
sorry

end divides_lcm_condition_l1878_187889


namespace largest_angle_isosceles_triangle_l1878_187895

theorem largest_angle_isosceles_triangle (A B C : ℕ) 
  (h_isosceles : A = B) 
  (h_base_angle : A = 50) : 
  max A (max B C) = 80 := 
by 
  -- proof is omitted  
  sorry

end largest_angle_isosceles_triangle_l1878_187895


namespace smallest_c_over_a_plus_b_l1878_187846

theorem smallest_c_over_a_plus_b (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  ∃ d : ℝ, d = (c / (a + b)) ∧ d = (Real.sqrt 2 / 2) :=
by
  sorry

end smallest_c_over_a_plus_b_l1878_187846


namespace total_gas_consumed_l1878_187863

def highway_consumption_rate : ℕ := 3
def city_consumption_rate : ℕ := 5

-- Distances driven each day
def day_1_highway_miles : ℕ := 200
def day_1_city_miles : ℕ := 300

def day_2_highway_miles : ℕ := 300
def day_2_city_miles : ℕ := 500

def day_3_highway_miles : ℕ := 150
def day_3_city_miles : ℕ := 350

-- Function to calculate the total consumption for a given day
def daily_consumption (highway_miles city_miles : ℕ) : ℕ :=
  (highway_miles * highway_consumption_rate) + (city_miles * city_consumption_rate)

-- Total consumption over three days
def total_consumption : ℕ :=
  (daily_consumption day_1_highway_miles day_1_city_miles) +
  (daily_consumption day_2_highway_miles day_2_city_miles) +
  (daily_consumption day_3_highway_miles day_3_city_miles)

-- Theorem stating the total consumption over the three days
theorem total_gas_consumed : total_consumption = 7700 := by
  sorry

end total_gas_consumed_l1878_187863


namespace potatoes_left_l1878_187873

theorem potatoes_left (initial_potatoes : ℕ) (potatoes_for_salads : ℕ) (potatoes_for_mashed : ℕ)
  (h1 : initial_potatoes = 52)
  (h2 : potatoes_for_salads = 15)
  (h3 : potatoes_for_mashed = 24) :
  initial_potatoes - (potatoes_for_salads + potatoes_for_mashed) = 13 := by
  sorry

end potatoes_left_l1878_187873


namespace car_discount_l1878_187858

variable (P D : ℝ)

theorem car_discount (h1 : 0 < P)
                     (h2 : (P - D) * 1.45 = 1.16 * P) :
                     D = 0.2 * P := by
  sorry

end car_discount_l1878_187858


namespace min_value_ineq_l1878_187807

noncomputable def a_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a 2018 = a 2017 + 2 * a 2016

theorem min_value_ineq (a : ℕ → ℝ) (m n : ℕ) 
  (h : a_sequence a) 
  (h2 : a m * a n = 16 * (a 1) ^ 2) :
  (4 / m) + (1 / n) ≥ 5 / 3 :=
sorry

end min_value_ineq_l1878_187807


namespace unique_solution_condition_l1878_187877

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l1878_187877


namespace shirts_bought_by_peter_l1878_187881

-- Define the constants and assumptions
variables (P S x : ℕ)

-- State the conditions given in the problem
def condition1 : P = 6 :=
by sorry

def condition2 : 2 * S = 20 :=
by sorry

def condition3 : 2 * P + x * S = 62 :=
by sorry

-- State the theorem to be proven
theorem shirts_bought_by_peter : x = 5 :=
by sorry

end shirts_bought_by_peter_l1878_187881


namespace rowed_upstream_distance_l1878_187893

def distance_downstream := 120
def time_downstream := 2
def distance_upstream := 2
def speed_stream := 15

def speed_boat (V_b : ℝ) := V_b

theorem rowed_upstream_distance (V_b : ℝ) (D_u : ℝ) :
  (distance_downstream = (V_b + speed_stream) * time_downstream) ∧
  (D_u = (V_b - speed_stream) * time_upstream) →
  D_u = 60 :=
by 
  sorry

end rowed_upstream_distance_l1878_187893


namespace wallpaper_removal_time_l1878_187819

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end wallpaper_removal_time_l1878_187819


namespace value_of_expression_l1878_187832

theorem value_of_expression (A B C D : ℝ) (h1 : A - B = 30) (h2 : C + D = 20) :
  (B + C) - (A - D) = -10 :=
by
  sorry

end value_of_expression_l1878_187832


namespace number_of_incorrect_statements_l1878_187866

-- Conditions
def cond1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)

def cond2 (x : ℝ) : Prop := x > 5 → x^2 - 4*x - 5 > 0

def cond3 : Prop := ∃ x0 : ℝ, x0^2 + x0 - 1 < 0

def cond3_neg : Prop := ∀ x : ℝ, x^2 + x - 1 ≥ 0

def cond4 (x : ℝ) : Prop := (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)

-- Proof problem
theorem number_of_incorrect_statements : 
  (¬ cond1 (p := true) (q := false)) ∧ (cond2 (x := 6)) ∧ (cond3 → cond3_neg) ∧ (¬ cond4 (x := 0)) → 
  2 = 2 :=
by
  sorry

end number_of_incorrect_statements_l1878_187866


namespace factor_quadratic_expression_l1878_187887

theorem factor_quadratic_expression (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) :=
by
  sorry

end factor_quadratic_expression_l1878_187887


namespace total_value_correct_l1878_187828

noncomputable def total_value (num_coins : ℕ) : ℕ :=
  let value_one_rupee := num_coins * 1
  let value_fifty_paise := (num_coins * 50) / 100
  let value_twentyfive_paise := (num_coins * 25) / 100
  value_one_rupee + value_fifty_paise + value_twentyfive_paise

theorem total_value_correct :
  let num_coins := 40
  total_value num_coins = 70 := by
  sorry

end total_value_correct_l1878_187828


namespace range_of_fx₂_l1878_187823

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + a * Real.log x

def is_extreme_point (a x : ℝ) : Prop := 
  (2 * x^2 - 2 * x + a) / x = 0

theorem range_of_fx₂ (a x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 2) 
  (h₂ : 0 < x₁ ∧ x₁ < x₂) (h₃ : is_extreme_point a x₁)
  (h₄ : is_extreme_point a x₂) : 
  (f a x₂) ∈ (Set.Ioo (-(3 + 2 * Real.log 2) / 4) (-1)) :=
sorry

end range_of_fx₂_l1878_187823


namespace remainder_of_expression_l1878_187888

theorem remainder_of_expression (n : ℤ) : (10 + n^2) % 7 = (3 + n^2) % 7 := 
by {
  sorry
}

end remainder_of_expression_l1878_187888


namespace jimmy_points_lost_for_bad_behavior_l1878_187860

theorem jimmy_points_lost_for_bad_behavior (points_per_exam : ℕ) (num_exams : ℕ) (points_needed : ℕ)
  (extra_points_allowed : ℕ) (total_points_earned : ℕ) (current_points : ℕ)
  (h1 : points_per_exam = 20) (h2 : num_exams = 3) (h3 : points_needed = 50)
  (h4 : extra_points_allowed = 5) (h5 : total_points_earned = points_per_exam * num_exams)
  (h6 : current_points = points_needed + extra_points_allowed) :
  total_points_earned - current_points = 5 :=
by
  sorry

end jimmy_points_lost_for_bad_behavior_l1878_187860


namespace find_m_l1878_187882

theorem find_m (m : ℝ) : (Real.tan (20 * Real.pi / 180) + m * Real.sin (20 * Real.pi / 180) = Real.sqrt 3) → m = 4 :=
by
  sorry

end find_m_l1878_187882


namespace solve_xy_l1878_187818

theorem solve_xy (x y : ℝ) :
  (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1 / 3 → 
  x = 34 / 3 ∧ y = 35 / 3 :=
by
  intro h
  sorry

end solve_xy_l1878_187818


namespace incorrect_statement_l1878_187849

def angles_on_x_axis := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi}
def angles_on_y_axis := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 2 + k * Real.pi}
def angles_on_axes := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi / 2}
def angles_on_y_eq_neg_x := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}

theorem incorrect_statement : ¬ (angles_on_y_eq_neg_x = {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}) :=
sorry

end incorrect_statement_l1878_187849


namespace long_furred_and_brown_dogs_l1878_187875

-- Define the total number of dogs.
def total_dogs : ℕ := 45

-- Define the number of long-furred dogs.
def long_furred_dogs : ℕ := 26

-- Define the number of brown dogs.
def brown_dogs : ℕ := 22

-- Define the number of dogs that are neither long-furred nor brown.
def neither_long_furred_nor_brown_dogs : ℕ := 8

-- Prove that the number of dogs that are both long-furred and brown is 11.
theorem long_furred_and_brown_dogs : 
  (long_furred_dogs + brown_dogs) - (total_dogs - neither_long_furred_nor_brown_dogs) = 11 :=
by
  sorry

end long_furred_and_brown_dogs_l1878_187875


namespace expected_yield_of_carrots_l1878_187835

def steps_to_feet (steps : ℕ) (step_size : ℕ) : ℕ :=
  steps * step_size

def garden_area (length width : ℕ) : ℕ :=
  length * width

def yield_of_carrots (area : ℕ) (yield_rate : ℚ) : ℚ :=
  area * yield_rate

theorem expected_yield_of_carrots :
  steps_to_feet 18 3 * steps_to_feet 25 3 = 4050 →
  yield_of_carrots 4050 (3 / 4) = 3037.5 :=
by
  sorry

end expected_yield_of_carrots_l1878_187835


namespace temperature_on_tuesday_l1878_187821

theorem temperature_on_tuesday 
  (M T W Th F Sa : ℝ)
  (h1 : (M + T + W) / 3 = 38)
  (h2 : (T + W + Th) / 3 = 42)
  (h3 : (W + Th + F) / 3 = 44)
  (h4 : (Th + F + Sa) / 3 = 46)
  (hF : F = 43)
  (pattern : M + 2 = Sa ∨ M - 1 = Sa) :
  T = 80 :=
sorry

end temperature_on_tuesday_l1878_187821


namespace arithmetic_sequence_l1878_187806

variable (p q : ℕ) -- Assuming natural numbers for simplicity, but can be generalized.

def a (n : ℕ) : ℕ := p * n + q

theorem arithmetic_sequence:
  ∀ n : ℕ, n ≥ 1 → (a n - a (n-1) = p) := by
  -- proof steps would go here
  sorry

end arithmetic_sequence_l1878_187806


namespace matt_house_wall_height_l1878_187822

noncomputable def height_of_walls_in_matt_house : ℕ :=
  let living_room_side := 40
  let bedroom_side_1 := 10
  let bedroom_side_2 := 12

  let perimeter_living_room := 4 * living_room_side
  let perimeter_living_room_3_walls := perimeter_living_room - living_room_side

  let perimeter_bedroom := 2 * (bedroom_side_1 + bedroom_side_2)

  let total_perimeter_to_paint := perimeter_living_room_3_walls + perimeter_bedroom
  let total_area_to_paint := 1640

  total_area_to_paint / total_perimeter_to_paint

theorem matt_house_wall_height :
  height_of_walls_in_matt_house = 10 := by
  sorry

end matt_house_wall_height_l1878_187822


namespace f_is_zero_l1878_187886

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_is_zero 
  (H1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a)
  (H2 : ∀ x : ℝ, |f x| ≤ 1) : ∀ x : ℝ, f x = 0 := 
sorry

end f_is_zero_l1878_187886


namespace point_in_second_or_third_quadrant_l1878_187855

theorem point_in_second_or_third_quadrant (k b : ℝ) (h₁ : k < 0) (h₂ : b ≠ 0) : 
  (k < 0 ∧ b > 0) ∨ (k < 0 ∧ b < 0) :=
by
  sorry

end point_in_second_or_third_quadrant_l1878_187855


namespace am_gm_inequality_l1878_187897

theorem am_gm_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ a + b + c :=
by
  sorry

end am_gm_inequality_l1878_187897


namespace painting_price_after_new_discount_l1878_187874

namespace PaintingPrice

-- Define the original price and the price Sarah paid
def original_price (x : ℕ) : Prop := x / 5 = 15

-- Define the new discounted price
def new_discounted_price (y x : ℕ) : Prop := y = x * 2 / 3

-- Theorem to prove the final price considering both conditions
theorem painting_price_after_new_discount (x y : ℕ) 
  (h1 : original_price x)
  (h2 : new_discounted_price y x) : y = 50 :=
by
  sorry

end PaintingPrice

end painting_price_after_new_discount_l1878_187874


namespace power_div_ex_l1878_187878

theorem power_div_ex (a b c : ℕ) (h1 : a = 2^4) (h2 : b = 2^3) (h3 : c = 2^2) :
  ((a^4) * (b^6)) / (c^12) = 1024 := 
sorry

end power_div_ex_l1878_187878


namespace geometric_sequence_common_ratio_l1878_187845

theorem geometric_sequence_common_ratio (a : ℕ → ℤ) (q : ℤ)  
  (h1 : a 1 = 3) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : 4 * a 1 + a 3 = 4 * a 2) : 
  q = 2 := 
by {
  -- Proof is omitted here
  sorry
}

end geometric_sequence_common_ratio_l1878_187845


namespace gcd_lcm_product_l1878_187820

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 := 
by
  rw [h₁, h₂]
  -- You can include specific calculation just to express the idea
  -- rw [Nat.gcd_comm, Nat.gcd_rec]
  -- rw [Nat.lcm_def]
  -- rw [Nat.mul_subst]
  sorry

end gcd_lcm_product_l1878_187820


namespace slope_of_tangent_at_A_l1878_187804

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem slope_of_tangent_at_A :
  (deriv f 0) = 1 :=
by
  sorry

end slope_of_tangent_at_A_l1878_187804


namespace pears_total_l1878_187865

-- Conditions
def keith_initial_pears : ℕ := 47
def keith_given_pears : ℕ := 46
def mike_initial_pears : ℕ := 12

-- Define the remaining pears
def keith_remaining_pears : ℕ := keith_initial_pears - keith_given_pears
def mike_remaining_pears : ℕ := mike_initial_pears

-- Theorem statement
theorem pears_total :
  keith_remaining_pears + mike_remaining_pears = 13 :=
by
  sorry

end pears_total_l1878_187865


namespace parameterize_line_l1878_187853

theorem parameterize_line (f : ℝ → ℝ) (t : ℝ) (x y : ℝ)
  (h1 : y = 2 * x - 30)
  (h2 : (x, y) = (f t, 20 * t - 10)) :
  f t = 10 * t + 10 :=
sorry

end parameterize_line_l1878_187853


namespace average_student_headcount_l1878_187856

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l1878_187856


namespace ratio_of_areas_of_similar_triangles_l1878_187840

theorem ratio_of_areas_of_similar_triangles (m1 m2 : ℝ) (med_ratio : m1 / m2 = 1 / Real.sqrt 2) :
    let area_ratio := (m1 / m2) ^ 2
    area_ratio = 1 / 2 := by
  sorry

end ratio_of_areas_of_similar_triangles_l1878_187840


namespace inequality_imply_positive_a_l1878_187842

theorem inequality_imply_positive_a 
  (a b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h_d_pos : d > 0) 
  (h : a / b > -3 / (2 * d)) : a > 0 :=
sorry

end inequality_imply_positive_a_l1878_187842


namespace problem1_problem2_l1878_187825

-- Define condition p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

-- Define the negation of p
def neg_p (x a : ℝ) : Prop := ¬ p x a
-- Define the negation of q
def neg_q (x : ℝ) : Prop := ¬ q x

-- Question 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem problem1 (x : ℝ) (h1 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬ p is a sufficient but not necessary condition for ¬ q, then 1 < a ≤ 2
theorem problem2 (a : ℝ) (h2 : ∀ x : ℝ, neg_p x a → neg_q x) : 1 < a ∧ a ≤ 2 := 
by sorry

end problem1_problem2_l1878_187825


namespace greatest_divisor_condition_gcd_of_numbers_l1878_187841

theorem greatest_divisor_condition (n : ℕ) (h100 : n ∣ 100) (h225 : n ∣ 225) (h150 : n ∣ 150) : n ≤ 25 :=
  sorry

theorem gcd_of_numbers : Nat.gcd (Nat.gcd 100 225) 150 = 25 :=
  sorry

end greatest_divisor_condition_gcd_of_numbers_l1878_187841


namespace calories_per_shake_l1878_187899

theorem calories_per_shake (total_calories_per_day : ℕ) (breakfast_calories : ℕ)
  (lunch_percentage_increase : ℕ) (dinner_multiplier : ℕ) (number_of_shakes : ℕ)
  (daily_calories : ℕ) :
  total_calories_per_day = breakfast_calories +
                            (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100)) +
                            (2 * (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100))) →
  daily_calories = total_calories_per_day + number_of_shakes * (daily_calories - total_calories_per_day) / number_of_shakes →
  daily_calories = 3275 → breakfast_calories = 500 →
  lunch_percentage_increase = 25 →
  dinner_multiplier = 2 →
  number_of_shakes = 3 →
  (daily_calories - total_calories_per_day) / number_of_shakes = 300 := by 
  sorry

end calories_per_shake_l1878_187899


namespace math_proof_problem_l1878_187871

variable {a b c : ℝ}

theorem math_proof_problem (h₁ : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h₂ : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  (a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b))) = 2017 := 
sorry

end math_proof_problem_l1878_187871


namespace compute_focus_d_l1878_187802

-- Define the given conditions as Lean definitions
structure Ellipse (d : ℝ) :=
  (first_quadrant : d > 0)
  (F1 : ℝ × ℝ := (4, 8))
  (F2 : ℝ × ℝ := (d, 8))
  (tangent_x_axis : (d + 4) / 2 > 0)
  (tangent_y_axis : (d + 4) / 2 > 0)

-- Define the proof problem to show d = 6 for the given conditions
theorem compute_focus_d (d : ℝ) (e : Ellipse d) : d = 6 := by
  sorry

end compute_focus_d_l1878_187802


namespace b_is_perfect_square_l1878_187885

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem b_is_perfect_square (a b : ℕ)
  (h_positive : 0 < a) (h_positive_b : 0 < b)
  (h_gcd_lcm_multiple : (Nat.gcd a b + Nat.lcm a b) % (a + 1) = 0)
  (h_le : b ≤ a) : is_perfect_square b :=
sorry

end b_is_perfect_square_l1878_187885


namespace dimes_total_l1878_187879

def initial_dimes : ℕ := 9
def added_dimes : ℕ := 7

theorem dimes_total : initial_dimes + added_dimes = 16 := by
  sorry

end dimes_total_l1878_187879


namespace total_journey_length_l1878_187859

theorem total_journey_length (y : ℚ)
  (h1 : y * 1 / 4 + 30 + y * 1 / 7 = y) : 
  y = 840 / 17 :=
by 
  sorry

end total_journey_length_l1878_187859


namespace moon_speed_conversion_l1878_187867

theorem moon_speed_conversion :
  ∀ (moon_speed_kps : ℝ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ),
  moon_speed_kps = 0.9 →
  seconds_in_minute = 60 →
  minutes_in_hour = 60 →
  (moon_speed_kps * (seconds_in_minute * minutes_in_hour) = 3240) := by
  sorry

end moon_speed_conversion_l1878_187867


namespace incorrect_statement_D_l1878_187827

def ordinate_of_x_axis_is_zero (p : ℝ × ℝ) : Prop :=
  p.2 = 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A_properties (a b : ℝ) : Prop :=
  let x := - a^2 - 1
  let y := abs b
  x < 0 ∧ y ≥ 0

theorem incorrect_statement_D (a b : ℝ) : 
  ∃ (x y : ℝ), point_A_properties a b ∧ (x = -a^2 - 1 ∧ y = abs b ∧ (x < 0 ∧ y = 0)) :=
by {
  sorry
}

end incorrect_statement_D_l1878_187827


namespace tan_4x_eq_cos_x_has_9_solutions_l1878_187854

theorem tan_4x_eq_cos_x_has_9_solutions :
  ∃ (s : Finset ℝ), s.card = 9 ∧ ∀ x ∈ s, (0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ (Real.tan (4 * x) = Real.cos x) :=
sorry

end tan_4x_eq_cos_x_has_9_solutions_l1878_187854


namespace find_AG_l1878_187864

theorem find_AG (AE CE BD CD AB AG : ℝ) (h1 : AE = 3)
    (h2 : CE = 1) (h3 : BD = 2) (h4 : CD = 2) (h5 : AB = 5) :
    AG = (3 * Real.sqrt 66) / 7 :=
  sorry

end find_AG_l1878_187864


namespace problem1_problem2_l1878_187862

-- Define the main assumptions and the proof problem for Lean 4
theorem problem1 (a : ℝ) (h : a ≠ 0) : (a^2)^3 / (-a)^2 = a^4 := sorry

theorem problem2 (a b : ℝ) : (a + 2 * b) * (a + b) - 3 * a * (a + b) = -2 * a^2 + 2 * b^2 := sorry

end problem1_problem2_l1878_187862


namespace min_vertical_distance_between_graphs_l1878_187808

noncomputable def absolute_value (x : ℝ) : ℝ :=
if x >= 0 then x else -x

theorem min_vertical_distance_between_graphs : 
  ∃ d : ℝ, d = 3 / 4 ∧ ∀ x : ℝ, ∃ dist : ℝ, dist = absolute_value x - (- x^2 - 4 * x - 3) ∧ dist >= d :=
by
  sorry

end min_vertical_distance_between_graphs_l1878_187808


namespace solution_correct_l1878_187813

noncomputable def solve_system (A1 A2 A3 A4 A5 : ℝ) (x1 x2 x3 x4 x5 : ℝ) :=
  (2 * x1 - 2 * x2 = A1) ∧
  (-x1 + 4 * x2 - 3 * x3 = A2) ∧
  (-2 * x2 + 6 * x3 - 4 * x4 = A3) ∧
  (-3 * x3 + 8 * x4 - 5 * x5 = A4) ∧
  (-4 * x4 + 10 * x5 = A5)

theorem solution_correct {A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 : ℝ} :
  solve_system A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 → 
  x1 = (5 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x2 = (2 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x3 = (A1 + 2 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x4 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 2 * A5) / 12 ∧
  x5 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 5 * A5) / 30 :=
sorry

end solution_correct_l1878_187813


namespace max_value_f_l1878_187861

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + x + 1)

theorem max_value_f : ∀ x : ℝ, f x ≤ 4 / 3 :=
sorry

end max_value_f_l1878_187861


namespace intersection_is_correct_l1878_187837

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem intersection_is_correct : A ∩ B = {-1, 2} := 
by 
  -- proof goes here 
  sorry

end intersection_is_correct_l1878_187837


namespace geometric_sequence_a6_l1878_187844

theorem geometric_sequence_a6
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : a 1 = 1)
  (S3 : S 3 = 7 / 4)
  (sum_S3 : S 3 = a 1 + a 1 * a 2 + a 1 * (a 2)^2) :
  a 6 = 1 / 32 := by
  sorry

end geometric_sequence_a6_l1878_187844


namespace coord_sum_D_l1878_187870

def is_midpoint (M C D : ℝ × ℝ) := M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem coord_sum_D (M C D : ℝ × ℝ) (h : is_midpoint M C D) (hM : M = (4, 6)) (hC : C = (10, 2)) :
  D.1 + D.2 = 8 :=
sorry

end coord_sum_D_l1878_187870


namespace integer_difference_divisible_by_n_l1878_187890

theorem integer_difference_divisible_by_n (n : ℕ) (h : n > 0) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end integer_difference_divisible_by_n_l1878_187890


namespace shape_is_cylinder_l1878_187803

noncomputable def shape_desc (r θ z a : ℝ) : Prop := r = a

theorem shape_is_cylinder (a : ℝ) (h_a : a > 0) :
  ∀ (r θ z : ℝ), shape_desc r θ z a → ∃ c : Set (ℝ × ℝ × ℝ), c = {p : ℝ × ℝ × ℝ | ∃ θ z, p = (a, θ, z)} :=
by
  sorry

end shape_is_cylinder_l1878_187803


namespace each_child_gets_one_slice_l1878_187805

-- Define the conditions
def couple_slices_per_person : ℕ := 3
def number_of_people : ℕ := 2
def number_of_children : ℕ := 6
def pizzas_ordered : ℕ := 3
def slices_per_pizza : ℕ := 4

-- Calculate slices required by the couple
def total_slices_for_couple : ℕ := couple_slices_per_person * number_of_people

-- Calculate total slices available
def total_slices : ℕ := pizzas_ordered * slices_per_pizza

-- Calculate slices for children
def slices_for_children : ℕ := total_slices - total_slices_for_couple

-- Calculate slices each child gets
def slices_per_child : ℕ := slices_for_children / number_of_children

-- The proof statement
theorem each_child_gets_one_slice : slices_per_child = 1 := by
  sorry

end each_child_gets_one_slice_l1878_187805


namespace suzhou_visitors_accuracy_l1878_187896

/--
In Suzhou, during the National Day holiday in 2023, the city received 17.815 million visitors.
Given that number, prove that it is accurate to the thousands place.
-/
theorem suzhou_visitors_accuracy :
  (17.815 : ℝ) * 10^6 = 17815000 ∧ true := 
by
sorry

end suzhou_visitors_accuracy_l1878_187896


namespace monotonic_range_of_a_l1878_187843

noncomputable def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1
noncomputable def f' (a x : ℝ) : ℝ := -3*x^2 + 2*a*x - 1

theorem monotonic_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f' a x ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by 
  sorry

end monotonic_range_of_a_l1878_187843


namespace Carol_width_eq_24_l1878_187839

-- Given conditions
def Carol_length : ℕ := 5
def Jordan_length : ℕ := 2
def Jordan_width : ℕ := 60

-- Required proof: Carol's width is 24 considering equal areas of both rectangles
theorem Carol_width_eq_24 (w : ℕ) (h : Carol_length * w = Jordan_length * Jordan_width) : w = 24 := 
by sorry

end Carol_width_eq_24_l1878_187839


namespace percentage_increase_l1878_187826

theorem percentage_increase (initial final : ℝ)
  (h_initial: initial = 60) (h_final: final = 90) :
  (final - initial) / initial * 100 = 50 :=
by
  sorry

end percentage_increase_l1878_187826


namespace system_solution_l1878_187868

theorem system_solution (a x0 : ℝ) (h : a ≠ 0) 
  (h1 : 3 * x0 + 2 * x0 = 15 * a) 
  (h2 : 1 / a * x0 + x0 = 9) 
  : x0 = 6 ∧ a = 2 :=
by {
  sorry
}

end system_solution_l1878_187868


namespace amount_spent_on_shirt_l1878_187816

-- Definitions and conditions
def total_spent_clothing : ℝ := 25.31
def spent_on_jacket : ℝ := 12.27

-- Goal: Prove the amount spent on the shirt is 13.04
theorem amount_spent_on_shirt : (total_spent_clothing - spent_on_jacket = 13.04) := by
  sorry

end amount_spent_on_shirt_l1878_187816


namespace perpendicular_lines_intersect_at_point_l1878_187883

theorem perpendicular_lines_intersect_at_point :
  ∀ (d k : ℝ), 
  (∀ x y, 3 * x - 4 * y = d ↔ 8 * x + k * y = d) → 
  (∃ x y, x = 2 ∧ y = -3 ∧ 3 * x - 4 * y = d ∧ 8 * x + k * y = d) → 
  d = -2 :=
by sorry

end perpendicular_lines_intersect_at_point_l1878_187883


namespace pastries_sold_correctly_l1878_187838

def cupcakes : ℕ := 4
def cookies : ℕ := 29
def total_pastries : ℕ := cupcakes + cookies
def left_over : ℕ := 24
def sold_pastries : ℕ := total_pastries - left_over

theorem pastries_sold_correctly : sold_pastries = 9 :=
by sorry

end pastries_sold_correctly_l1878_187838


namespace sam_added_later_buckets_l1878_187898

variable (initial_buckets : ℝ) (total_buckets : ℝ)

def buckets_added_later (initial_buckets total_buckets : ℝ) : ℝ :=
  total_buckets - initial_buckets

theorem sam_added_later_buckets :
  initial_buckets = 1 ∧ total_buckets = 9.8 → buckets_added_later initial_buckets total_buckets = 8.8 := by
  sorry

end sam_added_later_buckets_l1878_187898


namespace num_even_multiples_of_four_perfect_squares_lt_5000_l1878_187836

theorem num_even_multiples_of_four_perfect_squares_lt_5000 : 
  ∃ (k : ℕ), k = 17 ∧ ∀ (n : ℕ), (0 < n ∧ 16 * n^2 < 5000) ↔ (1 ≤ n ∧ n ≤ 17) :=
by
  sorry

end num_even_multiples_of_four_perfect_squares_lt_5000_l1878_187836


namespace vector_dot_product_l1878_187833

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem vector_dot_product : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by
  -- skipping the proof
  sorry

end vector_dot_product_l1878_187833


namespace total_chickens_l1878_187817

theorem total_chickens (coops chickens_per_coop : ℕ) (h1 : coops = 9) (h2 : chickens_per_coop = 60) :
  coops * chickens_per_coop = 540 := by
  sorry

end total_chickens_l1878_187817


namespace gcd_condition_l1878_187800

theorem gcd_condition (a b c : ℕ) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  Nat.gcd b c = 15 :=
sorry

end gcd_condition_l1878_187800


namespace rectangle_semicircle_problem_l1878_187857

/--
Rectangle ABCD and a semicircle with diameter AB are coplanar and have nonoverlapping interiors.
Let R denote the region enclosed by the semicircle and the rectangle.
Line ℓ meets the semicircle, segment AB, and segment CD at distinct points P, V, and S, respectively.
Line ℓ divides region R into two regions with areas in the ratio 3:1.
Suppose that AV = 120, AP = 180, and VB = 240.
Prove the length of DA = 90 * sqrt(6).
-/
theorem rectangle_semicircle_problem (DA : ℝ) (AV AP VB : ℝ) (h₁ : AV = 120) (h₂ : AP = 180) (h₃ : VB = 240) :
  DA = 90 * Real.sqrt 6 := by
  sorry

end rectangle_semicircle_problem_l1878_187857


namespace base_circumference_cone_l1878_187830

theorem base_circumference_cone (r : ℝ) (h : r = 5) (θ : ℝ) (k : θ = 180) : 
  ∃ c : ℝ, c = 5 * π :=
by
  sorry

end base_circumference_cone_l1878_187830


namespace asymptotes_of_hyperbola_min_focal_distance_l1878_187892

theorem asymptotes_of_hyperbola_min_focal_distance :
  ∀ (x y m : ℝ),
  (m = 1 → 
   (∀ x y : ℝ, (x^2 / (m^2 + 8) - y^2 / (6 - 2 * m) = 1) → 
   (y = 2/3 * x ∨ y = -2/3 * x))) := 
  sorry

end asymptotes_of_hyperbola_min_focal_distance_l1878_187892


namespace inequality_solution_set_l1878_187894

theorem inequality_solution_set (x : ℝ) :
  abs (1 + x + x^2 / 2) < 1 ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_set_l1878_187894


namespace difference_even_number_sums_l1878_187809

open Nat

def sum_of_even_numbers (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1
  n * (start + end_) / 2

theorem difference_even_number_sums :
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  sum_B - sum_A = 2100 :=
by
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  show sum_B - sum_A = 2100
  sorry

end difference_even_number_sums_l1878_187809


namespace problem_statement_l1878_187811

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 :=
sorry

end problem_statement_l1878_187811


namespace circular_film_diameter_l1878_187847

-- Definition of the problem conditions
def liquidVolume : ℝ := 576  -- volume of liquid Y in cm^3
def filmThickness : ℝ := 0.2  -- thickness of the film in cm

-- Statement of the theorem to prove the diameter of the film
theorem circular_film_diameter :
  2 * Real.sqrt (2880 / Real.pi) = 2 * Real.sqrt (liquidVolume / (filmThickness * Real.pi)) := by
  sorry

end circular_film_diameter_l1878_187847


namespace find_percentage_l1878_187801

theorem find_percentage (P : ℝ) :
  (P / 100) * 1280 = ((0.20 * 650) + 190) ↔ P = 25 :=
by
  sorry

end find_percentage_l1878_187801


namespace abcd_eq_eleven_l1878_187880

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- Conditions on a, b, c, d
axiom cond_a : a = Real.sqrt (4 + Real.sqrt (5 + a))
axiom cond_b : b = Real.sqrt (4 - Real.sqrt (5 + b))
axiom cond_c : c = Real.sqrt (4 + Real.sqrt (5 - c))
axiom cond_d : d = Real.sqrt (4 - Real.sqrt (5 - d))

-- Theorem to prove
theorem abcd_eq_eleven : a * b * c * d = 11 :=
by
  sorry

end abcd_eq_eleven_l1878_187880


namespace solve_ff_eq_x_l1878_187851

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x - 5

theorem solve_ff_eq_x (x : ℝ) :
  (f (f x) = x) ↔ 
  (x = (5 + 3 * Real.sqrt 5) / 2 ∨
   x = (5 - 3 * Real.sqrt 5) / 2 ∨
   x = (3 + Real.sqrt 41) / 2 ∨ 
   x = (3 - Real.sqrt 41) / 2) := 
by
  sorry

end solve_ff_eq_x_l1878_187851
