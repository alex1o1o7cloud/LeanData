import Mathlib

namespace area_triangle_possible_values_l670_67002

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem area_triangle_possible_values (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = Real.pi / 6) :
  ∃ S, S = 2 * Real.sqrt 3 ∨ S = Real.sqrt 3 :=
by
  -- Define the area using the given values
  sorry

end area_triangle_possible_values_l670_67002


namespace hattie_jumps_l670_67088

theorem hattie_jumps (H : ℝ) (h1 : Lorelei_jumps1 = (3/4) * H)
  (h2 : Hattie_jumps2 = (2/3) * H)
  (h3 : Lorelei_jumps2 = (2/3) * H + 50)
  (h4 : H + Lorelei_jumps1 + Hattie_jumps2 + Lorelei_jumps2 = 605) : H = 180 :=
by
  sorry

noncomputable def Lorelei_jumps1 (H : ℝ) := (3/4) * H
noncomputable def Hattie_jumps2 (H : ℝ) := (2/3) * H
noncomputable def Lorelei_jumps2 (H : ℝ) := (2/3) * H + 50

end hattie_jumps_l670_67088


namespace julie_reads_tomorrow_l670_67063

theorem julie_reads_tomorrow :
  let total_pages := 120
  let pages_read_yesterday := 12
  let pages_read_today := 2 * pages_read_yesterday
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  remaining_pages / 2 = 42 :=
by
  sorry

end julie_reads_tomorrow_l670_67063


namespace fifty_three_days_from_friday_is_tuesday_l670_67068

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l670_67068


namespace stormi_lawns_mowed_l670_67089

def num_lawns_mowed (cars_washed : ℕ) (money_per_car : ℕ) 
                    (lawns_mowed : ℕ) (money_per_lawn : ℕ) 
                    (bike_cost : ℕ) (money_needed : ℕ) : Prop :=
  (cars_washed * money_per_car + lawns_mowed * money_per_lawn) = (bike_cost - money_needed)

theorem stormi_lawns_mowed : num_lawns_mowed 3 10 2 13 80 24 :=
by
  sorry

end stormi_lawns_mowed_l670_67089


namespace value_of_f_at_9_l670_67013

def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

theorem value_of_f_at_9 : f 9 = 836 := sorry

end value_of_f_at_9_l670_67013


namespace triangle_internal_angle_60_l670_67049

theorem triangle_internal_angle_60 (A B C : ℝ) (h_sum : A + B + C = 180) : A >= 60 ∨ B >= 60 ∨ C >= 60 :=
sorry

end triangle_internal_angle_60_l670_67049


namespace Julio_spent_on_limes_l670_67041

theorem Julio_spent_on_limes
  (days : ℕ)
  (lime_cost_per_3 : ℕ)
  (mocktails_per_day : ℕ)
  (lime_juice_per_lime_tbsp : ℕ)
  (lime_juice_per_mocktail_tbsp : ℕ)
  (limes_per_set : ℕ)
  (days_eq_30 : days = 30)
  (lime_cost_per_3_eq_1 : lime_cost_per_3 = 1)
  (mocktails_per_day_eq_1 : mocktails_per_day = 1)
  (lime_juice_per_lime_tbsp_eq_2 : lime_juice_per_lime_tbsp = 2)
  (lime_juice_per_mocktail_tbsp_eq_1 : lime_juice_per_mocktail_tbsp = 1)
  (limes_per_set_eq_3 : limes_per_set = 3) :
  days * mocktails_per_day * lime_juice_per_mocktail_tbsp / lime_juice_per_lime_tbsp / limes_per_set * lime_cost_per_3 = 5 :=
sorry

end Julio_spent_on_limes_l670_67041


namespace volume_ratio_of_rotated_solids_l670_67071

theorem volume_ratio_of_rotated_solids (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := π * b^2 * a
  let V2 := π * a^2 * b
  V1 / V2 = b / a :=
by
  intros
  -- Proof omitted
  sorry

end volume_ratio_of_rotated_solids_l670_67071


namespace expected_greetings_l670_67054

theorem expected_greetings :
  let p1 := 1       -- Probability 1
  let p2 := 0.8     -- Probability 0.8
  let p3 := 0.5     -- Probability 0.5
  let p4 := 0       -- Probability 0
  let n1 := 8       -- Number of colleagues with probability 1
  let n2 := 15      -- Number of colleagues with probability 0.8
  let n3 := 14      -- Number of colleagues with probability 0.5
  let n4 := 3       -- Number of colleagues with probability 0
  p1 * n1 + p2 * n2 + p3 * n3 + p4 * n4 = 27 :=
by
  sorry

end expected_greetings_l670_67054


namespace initial_red_marbles_l670_67004

theorem initial_red_marbles (r g : ℕ) 
  (h1 : r = 5 * g / 3) 
  (h2 : (r - 20) * 5 = g + 40) : 
  r = 317 :=
by
  sorry

end initial_red_marbles_l670_67004


namespace equilateral_triangle_of_altitude_sum_l670_67023

def triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def altitude (a b c : ℝ) (S : ℝ) : ℝ := 
  2 * S / a

noncomputable def inradius (S : ℝ) (s : ℝ) : ℝ := 
  S / s

def shape_equilateral (a b c : ℝ) : Prop := 
  a = b ∧ b = c

theorem equilateral_triangle_of_altitude_sum (a b c h_a h_b h_c r S s : ℝ) 
  (habc : triangle a b c)
  (ha : h_a = altitude a b c S)
  (hb : h_b = altitude b a c S)
  (hc : h_c = altitude c a b S)
  (hr : r = inradius S s)
  (h_sum : h_a + h_b + h_c = 9 * r)
  (h_area : S = s * r)
  (h_semi : s = (a + b + c) / 2) : 
  shape_equilateral a b c := 
sorry

end equilateral_triangle_of_altitude_sum_l670_67023


namespace similar_triangles_legs_l670_67032

theorem similar_triangles_legs (y : ℝ) (h : 12 / y = 9 / 7) : y = 84 / 9 := by
  sorry

end similar_triangles_legs_l670_67032


namespace lettuce_price_1_l670_67037

theorem lettuce_price_1 (customers_per_month : ℕ) (lettuce_per_customer : ℕ) (tomatoes_per_customer : ℕ) 
(price_per_tomato : ℝ) (total_sales : ℝ)
  (h_customers : customers_per_month = 500)
  (h_lettuce_per_customer : lettuce_per_customer = 2)
  (h_tomatoes_per_customer : tomatoes_per_customer = 4)
  (h_price_per_tomato : price_per_tomato = 0.5)
  (h_total_sales : total_sales = 2000) :
  let heads_of_lettuce_sold := customers_per_month * lettuce_per_customer
  let tomato_sales := customers_per_month * tomatoes_per_customer * price_per_tomato
  let lettuce_sales := total_sales - tomato_sales
  let price_per_lettuce := lettuce_sales / heads_of_lettuce_sold
  price_per_lettuce = 1 := by
{
  sorry
}

end lettuce_price_1_l670_67037


namespace sixth_distance_l670_67038

theorem sixth_distance (A B C D : Point)
  (dist_AB dist_AC dist_BC dist_AD dist_BD dist_CD : ℝ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_lengths : (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BC = 1 ∧ dist_AD = 1) ∨
               (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AB = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_CD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1))
  (h_one_point_two : dist_AB = 1.2 ∨ dist_AC = 1.2 ∨ dist_BC = 1.2 ∨ dist_AD = 1.2 ∨ dist_BD = 1.2 ∨ dist_CD = 1.2) :
  dist_AB = 1.84 ∨ dist_AB = 0.24 ∨ dist_AB = 1.6 ∨
  dist_AC = 1.84 ∨ dist_AC = 0.24 ∨ dist_AC = 1.6 ∨
  dist_BC = 1.84 ∨ dist_BC = 0.24 ∨ dist_BC = 1.6 ∨
  dist_AD = 1.84 ∨ dist_AD = 0.24 ∨ dist_AD = 1.6 ∨
  dist_BD = 1.84 ∨ dist_BD = 0.24 ∨ dist_BD = 1.6 ∨
  dist_CD = 1.84 ∨ dist_CD = 0.24 ∨ dist_CD = 1.6 :=
sorry

end sixth_distance_l670_67038


namespace grey_eyes_black_hair_l670_67056

-- Definitions based on conditions
def num_students := 60
def num_black_hair := 36
def num_green_eyes_red_hair := 20
def num_grey_eyes := 24

-- Calculate number of students with red hair
def num_red_hair := num_students - num_black_hair

-- Calculate number of grey-eyed students with red hair
def num_grey_eyes_red_hair := num_red_hair - num_green_eyes_red_hair

-- Prove the number of grey-eyed students with black hair
theorem grey_eyes_black_hair:
  ∃ n, n = num_grey_eyes - num_grey_eyes_red_hair ∧ n = 20 :=
by
  sorry

end grey_eyes_black_hair_l670_67056


namespace find_x_l670_67042

variables (x : ℝ)
axiom h1 : (180 / x) + (5 * 12 / x) + 80 = 81

theorem find_x : x = 240 :=
by {
  sorry
}

end find_x_l670_67042


namespace sum_of_squares_of_rates_l670_67036

theorem sum_of_squares_of_rates (c j s : ℕ) (cond1 : 3 * c + 2 * j + 2 * s = 80) (cond2 : 2 * j + 2 * s + 4 * c = 104) : 
  c^2 + j^2 + s^2 = 592 :=
sorry

end sum_of_squares_of_rates_l670_67036


namespace monthly_growth_rate_selling_price_april_l670_67085

-- First problem: Proving the monthly average growth rate
theorem monthly_growth_rate (sales_jan sales_mar : ℝ) (x : ℝ) 
    (h1 : sales_jan = 256)
    (h2 : sales_mar = 400)
    (h3 : sales_mar = sales_jan * (1 + x)^2) :
  x = 0.25 := 
sorry

-- Second problem: Proving the selling price in April
theorem selling_price_april (unit_profit desired_profit current_sales sales_increase_per_yuan_change current_price new_price : ℝ)
    (h1 : unit_profit = new_price - 25)
    (h2 : desired_profit = 4200)
    (h3 : current_sales = 400)
    (h4 : sales_increase_per_yuan_change = 4)
    (h5 : current_price = 40)
    (h6 : desired_profit = unit_profit * (current_sales + sales_increase_per_yuan_change * (current_price - new_price))) :
  new_price = 35 := 
sorry

end monthly_growth_rate_selling_price_april_l670_67085


namespace find_b_minus_c_l670_67097

noncomputable def a (n : ℕ) : ℝ :=
  if h : n > 1 then 1 / Real.log 1009 * Real.log n else 0

noncomputable def b : ℝ :=
  a 2 + a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ :=
  a 15 + a 16 + a 17 + a 18 + a 19

theorem find_b_minus_c : b - c = -Real.logb 1009 1938 := by
  sorry

end find_b_minus_c_l670_67097


namespace solve_quadratic_equation_l670_67028

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end solve_quadratic_equation_l670_67028


namespace joan_number_of_games_l670_67095

open Nat

theorem joan_number_of_games (a b c d e : ℕ) (h_a : a = 10) (h_b : b = 12) (h_c : c = 6) (h_d : d = 9) (h_e : e = 4) :
  a + b + c + d + e = 41 :=
by
  sorry

end joan_number_of_games_l670_67095


namespace probability_player_A_wins_first_B_wins_second_l670_67076

theorem probability_player_A_wins_first_B_wins_second :
  (1 / 2) * (4 / 5) * (2 / 3) + (1 / 2) * (1 / 3) * (2 / 3) = 17 / 45 :=
by
  sorry

end probability_player_A_wins_first_B_wins_second_l670_67076


namespace find_f_of_2_l670_67007

variable (f : ℝ → ℝ)

def functional_equation_condition :=
  ∀ x : ℝ, f (f (f x)) + 3 * f (f x) + 9 * f x + 27 * x = 0

theorem find_f_of_2
  (h : functional_equation_condition f) :
  f (f (f (f 2))) = 162 :=
sorry

end find_f_of_2_l670_67007


namespace total_money_l670_67096

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l670_67096


namespace hyperbola_asymptotes_identical_l670_67009

theorem hyperbola_asymptotes_identical (x y M : ℝ) :
  (∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ (y = (b/a) * x ∨ y = -(b/a) * x)) ∧
  (∃ (c d : ℝ), c = 5 ∧ y = (c / d) * x ∨ y = -(c / d) * x) →
  M = (225 / 16) :=
by sorry

end hyperbola_asymptotes_identical_l670_67009


namespace coefficients_equal_l670_67070

theorem coefficients_equal (n : ℕ) (h : n ≥ 6) : 
  (n = 7) ↔ 
  (Nat.choose n 5 * 3 ^ 5 = Nat.choose n 6 * 3 ^ 6) := by
  sorry

end coefficients_equal_l670_67070


namespace perimeter_of_triangle_l670_67059

theorem perimeter_of_triangle (x y : ℝ) (h : 0 < x) (h1 : 0 < y) (h2 : x < y) :
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  (2 * leg_length + hypotenuse = (y - x) * (1 + 1 / Real.sqrt 2)) :=
by
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  sorry

end perimeter_of_triangle_l670_67059


namespace P_72_l670_67018

def P (n : ℕ) : ℕ :=
  -- The definition of P(n) should enumerate the ways of expressing n as a product
  -- of integers greater than 1, considering the order of factors.
  sorry

theorem P_72 : P 72 = 17 :=
by
  sorry

end P_72_l670_67018


namespace correct_sampling_method_is_D_l670_67029

def is_simple_random_sample (method : String) : Prop :=
  method = "drawing lots method to select 3 out of 10 products for quality inspection"

theorem correct_sampling_method_is_D : 
  is_simple_random_sample "drawing lots method to select 3 out of 10 products for quality inspection" :=
sorry

end correct_sampling_method_is_D_l670_67029


namespace order_abc_l670_67031

noncomputable def a : ℝ := (3 * (2 - Real.log 3)) / Real.exp 2
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.sqrt (Real.exp 1)) / (2 * Real.exp 1)

theorem order_abc : c < a ∧ a < b := by
  sorry

end order_abc_l670_67031


namespace probability_of_selecting_specific_letters_l670_67025

theorem probability_of_selecting_specific_letters :
  let total_cards := 15
  let amanda_cards := 6
  let chloe_or_ethan_cards := 9
  let prob_amanda_then_chloe_or_ethan := (amanda_cards / total_cards) * (chloe_or_ethan_cards / (total_cards - 1))
  let prob_chloe_or_ethan_then_amanda := (chloe_or_ethan_cards / total_cards) * (amanda_cards / (total_cards - 1))
  let total_prob := prob_amanda_then_chloe_or_ethan + prob_chloe_or_ethan_then_amanda
  total_prob = 18 / 35 :=
by
  sorry

end probability_of_selecting_specific_letters_l670_67025


namespace determine_even_condition_l670_67050

theorem determine_even_condition (x : ℤ) (m : ℤ) (h : m = x % 2) : m = 0 ↔ x % 2 = 0 :=
by sorry

end determine_even_condition_l670_67050


namespace max_value_of_sums_l670_67086

noncomputable def max_of_sums (a b c d : ℝ) : ℝ :=
  a^4 + b^4 + c^4 + d^4

theorem max_value_of_sums (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  max_of_sums a b c d ≤ 16 :=
sorry

end max_value_of_sums_l670_67086


namespace product_gcd_lcm_l670_67008

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l670_67008


namespace find_n_l670_67080

noncomputable
def equilateral_triangle_area_ratio (n : ℕ) (h : n > 4) : Prop :=
  let ratio := (2 : ℚ) / (n - 2 : ℚ)
  let area_PQR := (1 / 7 : ℚ)
  let menelaus_ap_pd := (n * (n - 2) : ℚ) / 4
  let area_triangle_ABP := (2 * (n - 2) : ℚ) / (n * (n - 2) + 4)
  let area_sum := 3 * area_triangle_ABP
  (area_sum * 7 = 6 * (n * (n - 2) + 4))

theorem find_n (n : ℕ) (h : n > 4) : 
  (equilateral_triangle_area_ratio n h) → n = 6 := sorry

end find_n_l670_67080


namespace inequality_holds_l670_67090

theorem inequality_holds (x : ℝ) : (∀ y : ℝ, y > 0 → (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y) / (x + y) > 3 * x^2 * y)) ↔ x > 0 := 
sorry

end inequality_holds_l670_67090


namespace recreation_spending_l670_67081

theorem recreation_spending : 
  ∀ (W : ℝ), 
  (last_week_spent : ℝ) -> last_week_spent = 0.20 * W →
  (this_week_wages : ℝ) -> this_week_wages = 0.80 * W →
  (this_week_spent : ℝ) -> this_week_spent = 0.40 * this_week_wages →
  this_week_spent / last_week_spent * 100 = 160 :=
by
  sorry

end recreation_spending_l670_67081


namespace johns_improvement_l670_67057

-- Declare the variables for the initial and later lap times.
def initial_minutes : ℕ := 50
def initial_laps : ℕ := 25
def later_minutes : ℕ := 54
def later_laps : ℕ := 30

-- Calculate the initial and later lap times in seconds, and the improvement.
def initial_lap_time_seconds := (initial_minutes * 60) / initial_laps 
def later_lap_time_seconds := (later_minutes * 60) / later_laps
def improvement := initial_lap_time_seconds - later_lap_time_seconds

-- State the theorem to prove the improvement is 12 seconds per lap.
theorem johns_improvement : improvement = 12 := by
  sorry

end johns_improvement_l670_67057


namespace problem_simplify_l670_67091

variable (a : ℝ)

theorem problem_simplify (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3)) :=
sorry

end problem_simplify_l670_67091


namespace prove_travel_cost_l670_67017

noncomputable def least_expensive_travel_cost
  (a_cost_per_km : ℝ) (a_booking_fee : ℝ) (b_cost_per_km : ℝ)
  (DE DF EF : ℝ) :
  ℝ := by
  let a_cost_DE := DE * a_cost_per_km + a_booking_fee
  let b_cost_DE := DE * b_cost_per_km
  let cheaper_cost_DE := min a_cost_DE b_cost_DE

  let a_cost_EF := EF * a_cost_per_km + a_booking_fee
  let b_cost_EF := EF * b_cost_per_km
  let cheaper_cost_EF := min a_cost_EF b_cost_EF

  let a_cost_DF := DF * a_cost_per_km + a_booking_fee
  let b_cost_DF := DF * b_cost_per_km
  let cheaper_cost_DF := min a_cost_DF b_cost_DF

  exact cheaper_cost_DE + cheaper_cost_EF + cheaper_cost_DF

def travel_problem : Prop :=
  let DE := 5000
  let DF := 4000
  let EF := 2500 -- derived from the Pythagorean theorem
  least_expensive_travel_cost 0.12 120 0.20 DE DF EF = 1740

theorem prove_travel_cost : travel_problem := sorry

end prove_travel_cost_l670_67017


namespace pi_minus_five_floor_value_l670_67010

noncomputable def greatest_integer_function (x : ℝ) : ℤ := Int.floor x

theorem pi_minus_five_floor_value :
  greatest_integer_function (Real.pi - 5) = -2 :=
by
  -- The proof is omitted
  sorry

end pi_minus_five_floor_value_l670_67010


namespace part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l670_67045

-- Part 1: Is x^2 - 3x + 2 = 0 a "double root equation"?
theorem part1_double_root_equation :
    ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁ * 2 = x₂) 
              ∧ (x^2 - 3 * x + 2 = 0) :=
sorry

-- Part 2: Given (x - 2)(x - m) = 0 is a "double root equation", find value of m^2 + 2m + 2.
theorem part2_value_m_squared_2m_2 (m : ℝ) :
    ∃ (v : ℝ), v = m^2 + 2 * m + 2 ∧ 
          (m = 1 ∨ m = 4) ∧
          (v = 5 ∨ v = 26) :=
sorry

-- Part 3: Determine m such that x^2 - (m-1)x + 32 = 0 is a "double root equation".
theorem part3_value_m (m : ℝ) :
    x^2 - (m - 1) * x + 32 = 0 ∧ 
    (m = 13 ∨ m = -11) :=
sorry

end part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l670_67045


namespace integer_value_of_fraction_l670_67053

theorem integer_value_of_fraction (m n p : ℕ) (hm_diff: m ≠ n) (hn_diff: n ≠ p) (hp_diff: m ≠ p) 
  (hm_range: 2 ≤ m ∧ m ≤ 9) (hn_range: 2 ≤ n ∧ n ≤ 9) (hp_range: 2 ≤ p ∧ p ≤ 9) :
  (m + n + p) / (m + n) = 2 :=
by
  sorry

end integer_value_of_fraction_l670_67053


namespace surface_area_of_sphere_with_diameter_two_l670_67066

theorem surface_area_of_sphere_with_diameter_two :
  let diameter := 2
  let radius := diameter / 2
  4 * Real.pi * radius ^ 2 = 4 * Real.pi :=
by
  sorry

end surface_area_of_sphere_with_diameter_two_l670_67066


namespace range_of_m_l670_67019

theorem range_of_m 
  (m : ℝ)
  (hM : -4 ≤ m ∧ m ≤ 4)
  (ellipse : ∀ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1 → y = 0) :
  1 ≤ m ∧ m ≤ 4 := sorry

end range_of_m_l670_67019


namespace original_number_is_twenty_l670_67047

theorem original_number_is_twenty (x : ℕ) (h : 100 * x = x + 1980) : x = 20 :=
sorry

end original_number_is_twenty_l670_67047


namespace computers_built_per_month_l670_67015

theorem computers_built_per_month (days_in_month : ℕ) (hours_per_day : ℕ) (computers_per_interval : ℚ) (intervals_per_hour : ℕ)
    (h_days : days_in_month = 28) (h_hours : hours_per_day = 24) (h_computers : computers_per_interval = 2.25) (h_intervals : intervals_per_hour = 2) :
    days_in_month * hours_per_day * intervals_per_hour * computers_per_interval = 3024 :=
by
  -- We would give the proof here, but it's omitted as per instructions.
  sorry

end computers_built_per_month_l670_67015


namespace simplify_fraction_addition_l670_67014

theorem simplify_fraction_addition (a b : ℚ) (h1 : a = 4 / 252) (h2 : b = 17 / 36) :
  a + b = 41 / 84 := 
by
  sorry

end simplify_fraction_addition_l670_67014


namespace domain_of_sqrt_expression_l670_67098

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end domain_of_sqrt_expression_l670_67098


namespace lcm_150_414_l670_67005

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 :=
by
  sorry

end lcm_150_414_l670_67005


namespace father_walk_time_l670_67033

-- Xiaoming's cycling speed is 4 times his father's walking speed.
-- Xiaoming continues for another 18 minutes to reach B after meeting his father.
-- Prove that Xiaoming's father needs 288 minutes to walk from the meeting point to A.
theorem father_walk_time {V : ℝ} (h₁ : V > 0) (h₂ : ∀ t : ℝ, t > 0 → 18 * V = (V / 4) * t) :
  288 = 4 * 72 :=
by
  sorry

end father_walk_time_l670_67033


namespace johnny_words_l670_67075

def words_johnny (J : ℕ) :=
  let words_madeline := 2 * J
  let words_timothy := 2 * J + 30
  let total_words := J + words_madeline + words_timothy
  total_words = 3 * 260 → J = 150

-- Statement of the main theorem (no proof provided, hence sorry is used)
theorem johnny_words (J : ℕ) : words_johnny J :=
by sorry

end johnny_words_l670_67075


namespace cupcakes_frosted_in_10_minutes_l670_67077

def frosting_rate (time: ℕ) (cupcakes: ℕ) : ℚ := cupcakes / time

noncomputable def combined_frosting_rate : ℚ :=
  (frosting_rate 25 1) + (frosting_rate 35 1)

def effective_working_time (total_time: ℕ) (work_period: ℕ) (break_time: ℕ) : ℕ :=
  let break_intervals := total_time / work_period
  total_time - break_intervals * break_time

def total_cupcakes (working_time: ℕ) (rate: ℚ) : ℚ :=
  working_time * rate

theorem cupcakes_frosted_in_10_minutes :
  total_cupcakes (effective_working_time 600 240 30) combined_frosting_rate = 36 := by
  sorry

end cupcakes_frosted_in_10_minutes_l670_67077


namespace no_primes_in_Q_plus_m_l670_67061

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_primes_in_Q_plus_m (m : ℕ) (hm : 2 ≤ m ∧ m ≤ 32) : ¬is_prime (Q + m) :=
by
  sorry  -- Proof would be provided here

end no_primes_in_Q_plus_m_l670_67061


namespace prob_one_boy_one_girl_l670_67024

-- Defining the probabilities of birth
def prob_boy := 2 / 3
def prob_girl := 1 / 3

-- Calculating the probability of all boys
def prob_all_boys := prob_boy ^ 4

-- Calculating the probability of all girls
def prob_all_girls := prob_girl ^ 4

-- Calculating the probability of having at least one boy and one girl
def prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)

-- Proof statement
theorem prob_one_boy_one_girl : prob_at_least_one_boy_and_one_girl = 64 / 81 :=
by sorry

end prob_one_boy_one_girl_l670_67024


namespace winning_cards_at_least_one_l670_67099

def cyclicIndex (n : ℕ) (i : ℕ) : ℕ := (i % n + n) % n

theorem winning_cards_at_least_one (a : ℕ → ℕ) (h : ∀ i, (a (cyclicIndex 8 (i - 1)) + a i + a (cyclicIndex 8 (i + 1))) % 2 = 1) :
  ∀ i, 1 ≤ a i :=
by
  sorry

end winning_cards_at_least_one_l670_67099


namespace sum_of_three_numbers_l670_67087

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_of_three_numbers_l670_67087


namespace geometric_sequence_a5_l670_67006

-- Definitions from the conditions
def a1 : ℕ := 2
def a9 : ℕ := 8

-- The statement we need to prove
theorem geometric_sequence_a5 (q : ℝ) (h1 : a1 = 2) (h2 : a9 = a1 * q ^ 8) : a1 * q ^ 4 = 4 := by
  have h_q4 : q ^ 4 = 2 := sorry
  -- Proof continues...
  sorry

end geometric_sequence_a5_l670_67006


namespace equal_distribution_l670_67065

theorem equal_distribution 
  (total_profit : ℕ) 
  (num_employees : ℕ) 
  (profit_kept_percent : ℕ) 
  (remaining_to_distribute : ℕ)
  (each_employee_gets : ℕ) :
  total_profit = 50 →
  num_employees = 9 →
  profit_kept_percent = 10 →
  remaining_to_distribute = total_profit - (total_profit * profit_kept_percent / 100) →
  each_employee_gets = remaining_to_distribute / num_employees →
  each_employee_gets = 5 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_distribution_l670_67065


namespace shaniqua_style_income_correct_l670_67092

def shaniqua_income_per_style (haircut_income : ℕ) (total_income : ℕ) (number_of_haircuts : ℕ) (number_of_styles : ℕ) : ℕ :=
  (total_income - (number_of_haircuts * haircut_income)) / number_of_styles

theorem shaniqua_style_income_correct :
  shaniqua_income_per_style 12 221 8 5 = 25 :=
by
  sorry

end shaniqua_style_income_correct_l670_67092


namespace range_of_m_l670_67058

-- Definitions given in the problem
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0
def neg_q_sufficient_for_neg_p : Prop :=
  ∀ {x m : ℝ}, (1 - m < x ∧ x < 1 + m) → (-2 ≤ x ∧ x ≤ 10)

-- The statement to prove
theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : 1 - m ≥ -2) (h3 : 1 + m ≤ 10) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l670_67058


namespace square_sum_inverse_eq_23_l670_67040

theorem square_sum_inverse_eq_23 {x : ℝ} (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end square_sum_inverse_eq_23_l670_67040


namespace gcd_cube_sum_condition_l670_67044

theorem gcd_cube_sum_condition (n : ℕ) (hn : n > 32) : Nat.gcd (n^3 + 125) (n + 5) = 1 := 
  by 
  sorry

end gcd_cube_sum_condition_l670_67044


namespace find_integer_l670_67083

theorem find_integer (n : ℕ) (hn1 : n % 20 = 0) (hn2 : 8.2 < (n : ℝ)^(1/3)) (hn3 : (n : ℝ)^(1/3) < 8.3) : n = 560 := sorry

end find_integer_l670_67083


namespace cos_alpha_minus_pi_l670_67060

theorem cos_alpha_minus_pi (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - Real.pi) = -1/6 := 
by
  sorry

end cos_alpha_minus_pi_l670_67060


namespace evaluate_expression_l670_67051

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end evaluate_expression_l670_67051


namespace angle_solution_exists_l670_67003

theorem angle_solution_exists :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ 9 * (Real.sin x) * (Real.cos x)^4 - 9 * (Real.sin x)^4 * (Real.cos x) = 1 / 2 ∧ x = 30 :=
by
  sorry

end angle_solution_exists_l670_67003


namespace otimes_identity_l670_67048

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_identity (h : ℝ) : otimes h (otimes h h) = h^2 :=
by
  sorry

end otimes_identity_l670_67048


namespace time_to_pass_platform_l670_67067

-- Conditions of the problem
def length_of_train : ℕ := 1500
def time_to_cross_tree : ℕ := 100
def length_of_platform : ℕ := 500

-- Derived values according to solution steps
def speed_of_train : ℚ := length_of_train / time_to_cross_tree
def total_distance_to_pass_platform : ℕ := length_of_train + length_of_platform

-- The theorem to be proved
theorem time_to_pass_platform :
  (total_distance_to_pass_platform / speed_of_train : ℚ) = 133.33 := sorry

end time_to_pass_platform_l670_67067


namespace initial_pieces_of_gum_l670_67069

theorem initial_pieces_of_gum (additional_pieces given_pieces leftover_pieces initial_pieces : ℕ)
  (h_additional : additional_pieces = 3)
  (h_given : given_pieces = 11)
  (h_leftover : leftover_pieces = 2)
  (h_initial : initial_pieces + additional_pieces = given_pieces + leftover_pieces) :
  initial_pieces = 10 :=
by
  sorry

end initial_pieces_of_gum_l670_67069


namespace shifted_line_does_not_pass_third_quadrant_l670_67021

def line_eq (x: ℝ) : ℝ := -2 * x - 1
def shifted_line_eq (x: ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_third_quadrant :
  ¬∃ x y : ℝ, shifted_line_eq x = y ∧ x < 0 ∧ y < 0 :=
sorry

end shifted_line_does_not_pass_third_quadrant_l670_67021


namespace minimum_room_size_for_table_l670_67016

theorem minimum_room_size_for_table (S : ℕ) :
  (∃ S, S ≥ 13) := sorry

end minimum_room_size_for_table_l670_67016


namespace smaller_angle_linear_pair_l670_67027

theorem smaller_angle_linear_pair (a b : ℝ) (h1 : a + b = 180) (h2 : a = 5 * b) : b = 30 := by
  sorry

end smaller_angle_linear_pair_l670_67027


namespace geometric_seq_general_formula_sum_c_seq_terms_l670_67062

noncomputable def a_seq (n : ℕ) : ℕ := 2 * 3 ^ (n - 1)

noncomputable def S_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (a_seq n - 2) / 2

theorem geometric_seq_general_formula (n : ℕ) (h : n > 0) : 
  a_seq n = 2 * 3 ^ (n - 1) := 
by {
  sorry
}

noncomputable def d_n (n : ℕ) : ℕ :=
  (a_seq (n + 1) - a_seq n) / (n + 1)

noncomputable def c_seq (n : ℕ) : ℕ :=
  d_n n / (n * a_seq n)

noncomputable def T_n (n : ℕ) : ℕ :=
  2 * (1 - 1 / (n + 1)) * n

theorem sum_c_seq_terms (n : ℕ) (h : n > 0) : 
  T_n n = 2 * n / (n + 1) :=
by {
  sorry
}

end geometric_seq_general_formula_sum_c_seq_terms_l670_67062


namespace cook_remaining_potatoes_l670_67093

def total_time_to_cook_remaining_potatoes (total_potatoes cooked_potatoes time_per_potato : ℕ) : ℕ :=
  (total_potatoes - cooked_potatoes) * time_per_potato

theorem cook_remaining_potatoes 
  (total_potatoes cooked_potatoes time_per_potato : ℕ) 
  (h_total_potatoes : total_potatoes = 13)
  (h_cooked_potatoes : cooked_potatoes = 5)
  (h_time_per_potato : time_per_potato = 6) : 
  total_time_to_cook_remaining_potatoes total_potatoes cooked_potatoes time_per_potato = 48 :=
by
  -- Proof not required
  sorry

end cook_remaining_potatoes_l670_67093


namespace John_more_marbles_than_Ben_l670_67052

theorem John_more_marbles_than_Ben :
  let ben_initial := 18
  let john_initial := 17
  let ben_gave := ben_initial / 2
  let ben_final := ben_initial - ben_gave
  let john_final := john_initial + ben_gave
  john_final - ben_final = 17 :=
by
  sorry

end John_more_marbles_than_Ben_l670_67052


namespace even_fn_a_eq_zero_l670_67012

def f (x a : ℝ) : ℝ := x^2 - |x + a|

theorem even_fn_a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end even_fn_a_eq_zero_l670_67012


namespace factor_quadratic_l670_67034

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end factor_quadratic_l670_67034


namespace area_of_region_l670_67074

theorem area_of_region :
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  (∫ t in (Real.pi / 3)..(Real.pi / 2), (x t) * (deriv y t)) * 2 = 2 * Real.pi - 3 * Real.sqrt 3 := by
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  have h1 : ∫ t in (Real.pi / 3)..(Real.pi / 2), x t * deriv y t = 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2*t)) / 2 := sorry
  have h2 : 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2 * t)) / 2 = 2 * Real.pi - 3 * Real.sqrt 3 := sorry
  sorry

end area_of_region_l670_67074


namespace find_digit_B_l670_67026

theorem find_digit_B (A B : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (h' : 0 ≤ B ∧ B ≤ 9) (eqn : 10 * A + 22 = 9 * B) : B = 8 := 
  sorry

end find_digit_B_l670_67026


namespace leak_empty_time_l670_67072

theorem leak_empty_time (A L : ℝ) (h1 : A = 1 / 8) (h2 : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- The proof will be provided here
  sorry

end leak_empty_time_l670_67072


namespace building_height_l670_67084

theorem building_height (H : ℝ) 
                        (bounced_height : ℕ → ℝ) 
                        (h_bounce : ∀ n, bounced_height n = H / 2 ^ (n + 1)) 
                        (h_fifth : bounced_height 5 = 3) : 
    H = 96 := 
by {
  sorry
}

end building_height_l670_67084


namespace exists_prime_seq_satisfying_condition_l670_67039

theorem exists_prime_seq_satisfying_condition :
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ m n, m < n → a m < a n) ∧ 
  (∀ i j, i ≠ j → (i * a j, j * a i) = (i, j)) :=
sorry

end exists_prime_seq_satisfying_condition_l670_67039


namespace quadratic_real_roots_a_leq_2_l670_67043

theorem quadratic_real_roots_a_leq_2
    (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + 2*a = 0) ∧ (x2^2 - 4*x2 + 2*a = 0)) →
    a ≤ 2 :=
by sorry

end quadratic_real_roots_a_leq_2_l670_67043


namespace problem_statement_l670_67055

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end problem_statement_l670_67055


namespace tetrahedron_three_edges_form_triangle_l670_67079

-- Defining a tetrahedron
structure Tetrahedron := (A B C D : ℝ)
-- length of edges - since it's a geometry problem using the absolute value
def edge_length (x y : ℝ) := abs (x - y)

theorem tetrahedron_three_edges_form_triangle (T : Tetrahedron) :
  ∃ v : ℕ, ∃ e1 e2 e3 : ℝ, 
    (edge_length T.A T.B = e1 ∨ edge_length T.A T.C = e1 ∨ edge_length T.A T.D = e1) ∧ 
    (edge_length T.B T.C = e2 ∨ edge_length T.B T.D = e2 ∨ edge_length T.C T.D = e2) ∧
    (edge_length T.A T.B < e2 + e3 ∧ edge_length T.B T.C < e1 + e3 ∧ edge_length T.C T.D < e1 + e2) := 
sorry

end tetrahedron_three_edges_form_triangle_l670_67079


namespace total_notebooks_correct_l670_67011

-- Definitions based on conditions
def total_students : ℕ := 28
def half_students : ℕ := total_students / 2
def notebooks_per_student_group1 : ℕ := 5
def notebooks_per_student_group2 : ℕ := 3

-- Total notebooks calculation
def total_notebooks : ℕ :=
  (half_students * notebooks_per_student_group1) + (half_students * notebooks_per_student_group2)

-- Theorem to be proved
theorem total_notebooks_correct : total_notebooks = 112 := by
  sorry

end total_notebooks_correct_l670_67011


namespace percentage_markup_on_cost_price_l670_67030

theorem percentage_markup_on_cost_price 
  (SP : ℝ) (CP : ℝ) (hSP : SP = 6400) (hCP : CP = 5565.217391304348) : 
  ((SP - CP) / CP) * 100 = 15 :=
by
  -- proof would go here
  sorry

end percentage_markup_on_cost_price_l670_67030


namespace michael_regular_hours_l670_67046

-- Define the constants and conditions
def regular_rate : ℝ := 7
def overtime_rate : ℝ := 14
def total_earnings : ℝ := 320
def total_hours : ℝ := 42.857142857142854

-- Declare the proof problem
theorem michael_regular_hours :
  ∃ R O : ℝ, (regular_rate * R + overtime_rate * O = total_earnings) ∧ (R + O = total_hours) ∧ (R = 40) :=
by
  sorry

end michael_regular_hours_l670_67046


namespace green_eyed_snack_min_l670_67001

variable {total_count green_eyes_count snack_bringers_count : ℕ}

def least_green_eyed_snack_bringers (total_count green_eyes_count snack_bringers_count : ℕ) : ℕ :=
  green_eyes_count - (total_count - snack_bringers_count)

theorem green_eyed_snack_min 
  (h_total : total_count = 35)
  (h_green_eyes : green_eyes_count = 18)
  (h_snack_bringers : snack_bringers_count = 24)
  : least_green_eyed_snack_bringers total_count green_eyes_count snack_bringers_count = 7 :=
by
  rw [h_total, h_green_eyes, h_snack_bringers]
  unfold least_green_eyed_snack_bringers
  norm_num

end green_eyed_snack_min_l670_67001


namespace problem_statement_l670_67073

theorem problem_statement 
  (x y z : ℝ)
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 :=
sorry

end problem_statement_l670_67073


namespace terry_daily_driving_time_l670_67082

theorem terry_daily_driving_time 
  (d1: ℝ) (s1: ℝ)
  (d2: ℝ) (s2: ℝ)
  (d3: ℝ) (s3: ℝ)
  (h1 : d1 = 15) (h2 : s1 = 30)
  (h3 : d2 = 35) (h4 : s2 = 50)
  (h5 : d3 = 10) (h6 : s3 = 40) : 
  2 * ((d1 / s1) + (d2 / s2) + (d3 / s3)) = 2.9 := 
by
  sorry

end terry_daily_driving_time_l670_67082


namespace playground_area_l670_67022

theorem playground_area (B : ℕ) (L : ℕ) (playground_area : ℕ) 
  (h1 : L = 8 * B) 
  (h2 : L = 240) 
  (h3 : playground_area = (1 / 6) * (L * B)) : 
  playground_area = 1200 :=
by
  sorry

end playground_area_l670_67022


namespace allison_greater_prob_l670_67078

noncomputable def prob_allison_greater (p_brian : ℝ) (p_noah : ℝ) : ℝ :=
  p_brian * p_noah

theorem allison_greater_prob : prob_allison_greater (2/3) (1/2) = 1/3 :=
by {
  -- Calculate the combined probability
  sorry
}

end allison_greater_prob_l670_67078


namespace find_rate_of_interest_l670_67094

theorem find_rate_of_interest (P SI : ℝ) (r : ℝ) (hP : P = 1200) (hSI : SI = 108) (ht : r = r) :
  SI = P * r * r / 100 → r = 3 := by
  intros
  sorry

end find_rate_of_interest_l670_67094


namespace height_of_pole_l670_67000

noncomputable section
open Real

theorem height_of_pole (α β γ : ℝ) (h xA xB xC : ℝ) 
  (hA : tan α = h / xA) (hB : tan β = h / xB) (hC : tan γ = h / xC) 
  (sum_angles : α + β + γ = π / 2) : h = 10 :=
by
  sorry

end height_of_pole_l670_67000


namespace hybrids_with_full_headlights_l670_67064

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l670_67064


namespace find_b_value_l670_67035

theorem find_b_value (b : ℝ) : (∃ (x y : ℝ), (x, y) = ((2 + 4) / 2, (5 + 9) / 2) ∧ x + y = b) ↔ b = 10 :=
by
  sorry

end find_b_value_l670_67035


namespace mr_kishore_savings_l670_67020

theorem mr_kishore_savings :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 3940
  let total_expenses := rent + milk + groceries + education + petrol + misc
  let savings_percentage := 0.10
  let salary := total_expenses / (1 - savings_percentage)
  let savings := savings_percentage * salary
  savings = 1937.78 := by
  sorry

end mr_kishore_savings_l670_67020
