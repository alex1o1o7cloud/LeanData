import Mathlib

namespace smaller_circle_circumference_l170_17017

-- Definitions based on the conditions given in the problem
def AB : ℝ := 24
def BC : ℝ := 45
def CD : ℝ := 28
def DA : ℝ := 53
def smaller_circle_diameter : ℝ := AB

-- Main statement to prove
theorem smaller_circle_circumference :
  let r : ℝ := smaller_circle_diameter / 2
  let circumference := 2 * Real.pi * r
  circumference = 24 * Real.pi := by
  sorry

end smaller_circle_circumference_l170_17017


namespace wheat_acres_l170_17019

def cultivate_crops (x y : ℕ) : Prop :=
  (42 * x + 30 * y = 18600) ∧ (x + y = 500) 

theorem wheat_acres : ∃ y, ∃ x, 
  cultivate_crops x y ∧ y = 200 :=
by {sorry}

end wheat_acres_l170_17019


namespace order_of_real_numbers_l170_17029

noncomputable def a : ℝ := Real.arcsin (3 / 4)
noncomputable def b : ℝ := Real.arccos (1 / 5)
noncomputable def c : ℝ := 1 + Real.arctan (2 / 3)

theorem order_of_real_numbers : a < b ∧ b < c :=
by sorry

end order_of_real_numbers_l170_17029


namespace samson_mother_age_l170_17037

variable (S M : ℕ)
variable (x : ℕ)

def problem_statement : Prop :=
  S = 6 ∧
  S - x = 2 ∧
  M - x = 4 * 2 →
  M = 16

theorem samson_mother_age (S M x : ℕ) (h : problem_statement S M x) : M = 16 :=
by
  sorry

end samson_mother_age_l170_17037


namespace boat_speed_still_water_l170_17025

theorem boat_speed_still_water (V_s : ℝ) (T_u T_d : ℝ) 
  (h1 : V_s = 24) 
  (h2 : T_u = 2 * T_d) 
  (h3 : (V_b - V_s) * T_u = (V_b + V_s) * T_d) : 
  V_b = 72 := 
sorry

end boat_speed_still_water_l170_17025


namespace spend_on_rent_and_utilities_l170_17095

variable (P : ℝ) -- The percentage of her income she used to spend on rent and utilities
variable (I : ℝ) -- Her previous monthly income
variable (increase : ℝ) -- Her salary increase
variable (new_percentage : ℝ) -- The new percentage her rent and utilities amount to

noncomputable def initial_conditions : Prop :=
I = 1000 ∧ increase = 600 ∧ new_percentage = 0.25

theorem spend_on_rent_and_utilities (h : initial_conditions I increase new_percentage) :
    (P / 100) * I = 0.25 * (I + increase) → 
    P = 40 :=
by
  sorry

end spend_on_rent_and_utilities_l170_17095


namespace other_root_is_minus_two_l170_17089

theorem other_root_is_minus_two (b : ℝ) (h : 1^2 + b * 1 - 2 = 0) : 
  ∃ (x : ℝ), x = -2 ∧ x^2 + b * x - 2 = 0 :=
by
  sorry

end other_root_is_minus_two_l170_17089


namespace area_of_triangle_ABC_l170_17057

theorem area_of_triangle_ABC :
  let A'B' := 4
  let B'C' := 3
  let angle_A'B'C' := 60
  let area_A'B'C' := (1 / 2) * A'B' * B'C' * Real.sin (angle_A'B'C' * Real.pi / 180)
  let ratio := 2 * Real.sqrt 2
  let area_ABC := ratio * area_A'B'C'
  area_ABC = 6 * Real.sqrt 6 := 
by
  sorry

end area_of_triangle_ABC_l170_17057


namespace bike_cost_l170_17090

-- Defining the problem conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def leftover : ℚ := 20  -- 20 dollars left over
def quarter_value : ℚ := 0.25

-- Define the total quarters Jenn has
def total_quarters := jars * quarters_per_jar

-- Define the total amount of money from quarters
def total_money_quarters := total_quarters * quarter_value

-- Prove that the cost of the bike is $200
theorem bike_cost : total_money_quarters + leftover - 20 = 200 :=
sorry

end bike_cost_l170_17090


namespace max_value_of_q_l170_17008

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l170_17008


namespace longest_side_in_ratio_5_6_7_l170_17096

theorem longest_side_in_ratio_5_6_7 (x : ℕ) (h : 5 * x + 6 * x + 7 * x = 720) : 7 * x = 280 := 
by
  sorry

end longest_side_in_ratio_5_6_7_l170_17096


namespace area_and_cost_of_path_l170_17055

variables (length_field width_field path_width : ℝ) (cost_per_sq_m : ℝ)

noncomputable def area_of_path (length_field width_field path_width : ℝ) : ℝ :=
  let total_length := length_field + 2 * path_width
  let total_width := width_field + 2 * path_width
  let area_with_path := total_length * total_width
  let area_grass_field := length_field * width_field
  area_with_path - area_grass_field

noncomputable def cost_of_path (area_of_path cost_per_sq_m : ℝ) : ℝ :=
  area_of_path * cost_per_sq_m

theorem area_and_cost_of_path
  (length_field width_field path_width : ℝ)
  (cost_per_sq_m : ℝ)
  (h_length_field : length_field = 75)
  (h_width_field : width_field = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_sq_m : cost_per_sq_m = 10) :
  area_of_path length_field width_field path_width = 675 ∧
  cost_of_path (area_of_path length_field width_field path_width) cost_per_sq_m = 6750 :=
by
  rw [h_length_field, h_width_field, h_path_width, h_cost_per_sq_m]
  simp [area_of_path, cost_of_path]
  sorry

end area_and_cost_of_path_l170_17055


namespace correct_operation_l170_17028

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end correct_operation_l170_17028


namespace abs_difference_of_numbers_l170_17094

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 :=
sorry

end abs_difference_of_numbers_l170_17094


namespace math_competition_l170_17098

theorem math_competition :
  let Sammy_score := 20
  let Gab_score := 2 * Sammy_score
  let Cher_score := 2 * Gab_score
  let Total_score := Sammy_score + Gab_score + Cher_score
  let Opponent_score := 85
  Total_score - Opponent_score = 55 :=
by
  sorry

end math_competition_l170_17098


namespace distance_between_points_l170_17053

theorem distance_between_points :
  let x1 := 1
  let y1 := 3
  let z1 := 2
  let x2 := 4
  let y2 := 1
  let z2 := 6
  let distance : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
  distance = Real.sqrt 29 := by
  sorry

end distance_between_points_l170_17053


namespace subtraction_of_7_305_from_neg_3_219_l170_17038

theorem subtraction_of_7_305_from_neg_3_219 :
  -3.219 - 7.305 = -10.524 :=
by
  -- The proof would go here
  sorry

end subtraction_of_7_305_from_neg_3_219_l170_17038


namespace sqrt_two_squared_l170_17045

noncomputable def sqrt_two : Real := Real.sqrt 2

theorem sqrt_two_squared : (sqrt_two) ^ 2 = 2 :=
by
  sorry

end sqrt_two_squared_l170_17045


namespace three_pumps_drain_time_l170_17049

-- Definitions of the rates of each pump
def rate1 := 1 / 9
def rate2 := 1 / 6
def rate3 := 1 / 12

-- Combined rate of all three pumps working together
def combined_rate := rate1 + rate2 + rate3

-- Time to drain the lake with all three pumps working together
def time_to_drain := 1 / combined_rate

-- Theorem: The time it takes for three pumps working together to drain the lake is 36/13 hours
theorem three_pumps_drain_time : time_to_drain = 36 / 13 := by
  sorry

end three_pumps_drain_time_l170_17049


namespace sum_units_tens_not_divisible_by_4_l170_17061

theorem sum_units_tens_not_divisible_by_4 :
  ∃ (n : ℕ), (n = 3674 ∨ n = 3684 ∨ n = 3694 ∨ n = 3704 ∨ n = 3714 ∨ n = 3722) ∧
  (¬ (∃ k, (n % 100) = 4 * k)) ∧
  ((n % 10) + (n / 10 % 10) = 11) :=
sorry

end sum_units_tens_not_divisible_by_4_l170_17061


namespace equation_of_tangent_line_l170_17071

noncomputable def f (m x : ℝ) := m * Real.exp x - x - 1

def passes_through_P (m : ℝ) : Prop :=
  f m 0 = 1

theorem equation_of_tangent_line (m : ℝ) (h : passes_through_P m) :
  (f m) 0 = 1 → (2 - 1 = 1) ∧ ((y - 1 = x) → (x - y + 1 = 0)) :=
by
  intro h
  sorry

end equation_of_tangent_line_l170_17071


namespace part_a_part_b_l170_17026

-- Define what it means for a number to be "surtido"
def is_surtido (A : ℕ) : Prop :=
  ∀ n, (1 ≤ n → n ≤ (A.digits 10).sum → ∃ B : ℕ, n = (B.digits 10).sum) 

-- Part (a): Prove that if 1, 2, 3, 4, 5, 6, 7, and 8 can be expressed as sums of digits in A, then A is "surtido".
theorem part_a (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum)
  (h8 : ∃ B8 : ℕ, 8 = (B8.digits 10).sum) : is_surtido A :=
sorry

-- Part (b): Determine if having the sums 1, 2, 3, 4, 5, 6, and 7 as sums of digits in A implies that A is "surtido".
theorem part_b (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum) : ¬is_surtido A :=
sorry

end part_a_part_b_l170_17026


namespace maximum_value_of_f_on_interval_l170_17041

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x + Real.sin x

theorem maximum_value_of_f_on_interval :
  ∃ M, M = Real.pi ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ M :=
by
  sorry

end maximum_value_of_f_on_interval_l170_17041


namespace first_set_cost_l170_17024

theorem first_set_cost (F S : ℕ) (hS : S = 50) (h_equation : 2 * F + 3 * S = 220) 
: 3 * F + S = 155 := 
sorry

end first_set_cost_l170_17024


namespace cody_final_money_l170_17033

-- Definitions for the initial conditions
def original_money : ℝ := 45
def birthday_money : ℝ := 9
def game_price : ℝ := 19
def discount_rate : ℝ := 0.10
def friend_owes : ℝ := 12

-- Calculate the final amount Cody has
def final_amount : ℝ := original_money + birthday_money - (game_price * (1 - discount_rate)) + friend_owes

-- The theorem to prove the amount of money Cody has now
theorem cody_final_money :
  final_amount = 48.90 :=
by sorry

end cody_final_money_l170_17033


namespace A_B_finish_l170_17000

theorem A_B_finish (A B C : ℕ → ℝ) (h1 : A + B + C = 1 / 6) (h2 : C = 1 / 10) :
  1 / (A + B) = 15 :=
by
  sorry

end A_B_finish_l170_17000


namespace find_digits_l170_17072

theorem find_digits (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h_sum : 100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a)) :
  a = 1 ∧ b = 9 ∧ c = 8 := by
  sorry

end find_digits_l170_17072


namespace machines_working_together_l170_17080

theorem machines_working_together (x : ℝ) :
  (∀ P Q R : ℝ, P = x + 4 ∧ Q = x + 2 ∧ R = 2 * x + 2 ∧ (1 / P + 1 / Q + 1 / R = 1 / x)) ↔ (x = 2 / 3) :=
by
  sorry

end machines_working_together_l170_17080


namespace binomial_expansion_fraction_l170_17097

theorem binomial_expansion_fraction :
  let a0 := 32
  let a1 := -80
  let a2 := 80
  let a3 := -40
  let a4 := 10
  let a5 := -1
  (2 - x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  (a0 + a2 + a4) / (a1 + a3) = -61 / 60 :=
by
  sorry

end binomial_expansion_fraction_l170_17097


namespace ice_cream_cost_proof_l170_17022

-- Assume the cost of the ice cream and toppings
def cost_of_ice_cream : ℝ := 2 -- Ice cream cost in dollars
def cost_per_topping : ℝ := 0.5 -- Cost per topping in dollars
def total_cost_of_sundae_with_10_toppings : ℝ := 7 -- Total cost in dollars

theorem ice_cream_cost_proof :
  (∀ (cost_of_ice_cream : ℝ), 
    total_cost_of_sundae_with_10_toppings = cost_of_ice_cream + 10 * cost_per_topping) →
  cost_of_ice_cream = 2 :=
by
  sorry

end ice_cream_cost_proof_l170_17022


namespace minimize_quadratic_sum_l170_17023

theorem minimize_quadratic_sum (a b : ℝ) : 
  ∃ x : ℝ, y = (x-a)^2 + (x-b)^2 ∧ (∀ x', (x'-a)^2 + (x'-b)^2 ≥ y) ∧ x = (a + b) / 2 := 
sorry

end minimize_quadratic_sum_l170_17023


namespace cab_driver_income_l170_17070

theorem cab_driver_income (x2 : ℕ) :
  (600 + x2 + 450 + 400 + 800) / 5 = 500 → x2 = 250 :=
by
  sorry

end cab_driver_income_l170_17070


namespace gcd_pow_minus_one_l170_17082

theorem gcd_pow_minus_one {m n a : ℕ} (hm : 0 < m) (hn : 0 < n) (ha : 2 ≤ a) : 
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd m n) - 1 := 
sorry

end gcd_pow_minus_one_l170_17082


namespace Jasmine_gets_off_work_at_4pm_l170_17065

-- Conditions
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_clean_time : ℕ := 10
def groomer_time : ℕ := 20
def cook_time : ℕ := 90
def dinner_time : ℕ := 19 * 60  -- 7:00 pm in minutes

-- Question to prove
theorem Jasmine_gets_off_work_at_4pm : 
  (dinner_time - cook_time - groomer_time - dry_clean_time - grocery_time - commute_time = 16 * 60) := sorry

end Jasmine_gets_off_work_at_4pm_l170_17065


namespace product_of_repeating_decimals_l170_17043

theorem product_of_repeating_decimals :
  let x := (4 / 9 : ℚ)
  let y := (7 / 9 : ℚ)
  x * y = 28 / 81 :=
by
  sorry

end product_of_repeating_decimals_l170_17043


namespace find_m_l170_17067

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {m : ℕ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def initial_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def q_condition (q : ℝ) : Prop :=
  abs q ≠ 1

def a_m_condition (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a m = a 1 * a 2 * a 3 * a 4 * a 5

-- Theorem to prove
theorem find_m (h1 : geometric_sequence a q) (h2 : initial_condition a) (h3 : q_condition q) (h4 : a_m_condition a m) : m = 11 :=
  sorry

end find_m_l170_17067


namespace decrease_percent_revenue_l170_17086

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.68 * T
  let new_consumption := 1.12 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 23.84 := by {
    sorry
  }

end decrease_percent_revenue_l170_17086


namespace original_number_of_men_l170_17062

theorem original_number_of_men (x : ℕ) 
  (h1 : 17 * x = 21 * (x - 8)) : x = 42 := 
by {
   -- proof steps can be filled in here
   sorry
}

end original_number_of_men_l170_17062


namespace find_n_l170_17077

-- Definitions of the problem conditions
def sum_coefficients (n : ℕ) : ℕ := 4^n
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- The main theorem to be proved
theorem find_n (n : ℕ) (P S : ℕ) (hP : P = sum_coefficients n) (hS : S = sum_binomial_coefficients n) (h : P + S = 272) : n = 4 :=
by
  sorry

end find_n_l170_17077


namespace binary_mul_correct_l170_17014

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l170_17014


namespace point_in_third_quadrant_coordinates_l170_17088

theorem point_in_third_quadrant_coordinates :
  ∀ (P : ℝ × ℝ), (P.1 < 0) ∧ (P.2 < 0) ∧ (|P.2| = 2) ∧ (|P.1| = 3) -> P = (-3, -2) :=
by
  intros P h
  sorry

end point_in_third_quadrant_coordinates_l170_17088


namespace evaluates_to_m_times_10_pow_1012_l170_17063

theorem evaluates_to_m_times_10_pow_1012 :
  let a := (3:ℤ) ^ 1010
  let b := (4:ℤ) ^ 1012
  (a + b) ^ 2 - (a - b) ^ 2 = 10 ^ 3642 := by
  sorry

end evaluates_to_m_times_10_pow_1012_l170_17063


namespace probability_red_given_spade_or_king_l170_17048

def num_cards := 52
def num_spades := 13
def num_kings := 4
def num_red_kings := 2

def num_non_spade_kings := num_kings - 1
def num_spades_or_kings := num_spades + num_non_spade_kings

theorem probability_red_given_spade_or_king :
  (num_red_kings : ℚ) / num_spades_or_kings = 1 / 8 :=
sorry

end probability_red_given_spade_or_king_l170_17048


namespace pepperoni_ratio_l170_17074

-- Definition of the problem's conditions
def total_pepperoni_slices : ℕ := 40
def slice_given_to_jelly_original : ℕ := 10
def slice_fallen_off : ℕ := 1

-- Our goal is to prove that the ratio is 3:10
theorem pepperoni_ratio (total_pepperoni_slices : ℕ) (slice_given_to_jelly_original : ℕ) (slice_fallen_off : ℕ) :
  (slice_given_to_jelly_original - slice_fallen_off) / (total_pepperoni_slices - slice_given_to_jelly_original) = 3 / 10 :=
by
  sorry

end pepperoni_ratio_l170_17074


namespace initial_men_count_l170_17040

theorem initial_men_count (M : ℕ) (h1 : ∃ F : ℕ, F = M * 22) (h2 : ∃ F_remaining : ℕ, F_remaining = M * 20) (h3 : ∃ F_remaining_2 : ℕ, F_remaining_2 = (M + 1140) * 8) : 
  M = 760 := 
by
  -- Code to prove the theorem goes here.
  sorry

end initial_men_count_l170_17040


namespace series_sum_eq_negative_one_third_l170_17052

noncomputable def series_sum : ℝ :=
  ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

theorem series_sum_eq_negative_one_third : series_sum = -1 / 3 := sorry

end series_sum_eq_negative_one_third_l170_17052


namespace find_original_speed_l170_17046

theorem find_original_speed (r : ℝ) (t : ℝ)
  (h_circumference : r * t = 15 / 5280)
  (h_increase : (r + 8) * (t - 1/10800) = 15 / 5280) :
  r = 7.5 :=
sorry

end find_original_speed_l170_17046


namespace probability_all_balls_same_color_probability_4_white_balls_l170_17064

-- Define initial conditions
def initial_white_balls : ℕ := 6
def initial_yellow_balls : ℕ := 4
def total_initial_balls : ℕ := initial_white_balls + initial_yellow_balls

-- Define the probability calculation for drawing balls as described
noncomputable def draw_probability_same_color_after_4_draws : ℚ :=
  (6 / 10) * (7 / 10) * (8 / 10) * (9 / 10)

noncomputable def draw_probability_4_white_balls_after_4_draws : ℚ :=
  (6 / 10) * (3 / 10) * (4 / 10) * (5 / 10) + 
  3 * ((4 / 10) * (5 / 10) * (4 / 10) * (5 / 10))

-- The theorem we want to prove about the probabilities
theorem probability_all_balls_same_color :
  draw_probability_same_color_after_4_draws = 189 / 625 := by
  sorry

theorem probability_4_white_balls :
  draw_probability_4_white_balls_after_4_draws = 19 / 125 := by
  sorry

end probability_all_balls_same_color_probability_4_white_balls_l170_17064


namespace problem1_solution_problem2_solution_l170_17018

theorem problem1_solution (x : ℝ) (h : 5 / (x - 1) = 1 / (2 * x + 1)) : x = -2 / 3 := sorry

theorem problem2_solution (x : ℝ) (h : 1 / (x - 2) + 2 = (1 - x) / (2 - x)) : false := sorry

end problem1_solution_problem2_solution_l170_17018


namespace volume_region_between_spheres_l170_17059

theorem volume_region_between_spheres 
    (r1 r2 : ℝ) 
    (h1 : r1 = 4) 
    (h2 : r2 = 7) 
    : 
    ( (4/3) * π * r2^3 - (4/3) * π * r1^3 ) = 372 * π := 
    sorry

end volume_region_between_spheres_l170_17059


namespace HCl_yield_l170_17016

noncomputable def total_moles_HCl (moles_C2H6 moles_Cl2 yield1 yield2 : ℝ) : ℝ :=
  let theoretical_yield1 := if moles_C2H6 ≤ moles_Cl2 then moles_C2H6 else moles_Cl2
  let actual_yield1 := theoretical_yield1 * yield1
  let theoretical_yield2 := actual_yield1
  let actual_yield2 := theoretical_yield2 * yield2
  actual_yield1 + actual_yield2

theorem HCl_yield (moles_C2H6 moles_Cl2 : ℝ) (yield1 yield2 : ℝ) :
  moles_C2H6 = 3 → moles_Cl2 = 3 → yield1 = 0.85 → yield2 = 0.70 →
  total_moles_HCl moles_C2H6 moles_Cl2 yield1 yield2 = 4.335 :=
by
  intros h1 h2 h3 h4
  simp [total_moles_HCl, h1, h2, h3, h4]
  sorry

end HCl_yield_l170_17016


namespace triangle_is_right_triangle_l170_17030

variable (A B C : ℝ) (a b c : ℝ)

-- Conditions definitions
def condition1 : Prop := A + B = C
def condition2 : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5 ∧ a / c = 3 / 5
def condition3 : Prop := A = 90 - B

-- Proof problem
theorem triangle_is_right_triangle (h1 : condition1 A B C) (h2 : condition2 a b c) (h3 : condition3 A B) : C = 90 := 
sorry

end triangle_is_right_triangle_l170_17030


namespace square_completing_l170_17079

theorem square_completing (b c : ℤ) (h : (x^2 - 10 * x + 15 = 0) → ((x + b)^2 = c)) : 
  b + c = 5 :=
sorry

end square_completing_l170_17079


namespace triangle_inequality_l170_17031

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l170_17031


namespace M_subset_N_l170_17012

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2 * a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

theorem M_subset_N : M ⊆ N := 
by 
  sorry

end M_subset_N_l170_17012


namespace intersecting_circles_range_of_m_l170_17060

theorem intersecting_circles_range_of_m
  (x y m : ℝ)
  (C₁_eq : x^2 + y^2 - 2 * m * x + m^2 - 4 = 0)
  (C₂_eq : x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0)
  (intersect : ∃ x y : ℝ, (x^2 + y^2 - 2 * m * x + m^2 - 4 = 0) ∧ (x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0))
  : m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := 
sorry

end intersecting_circles_range_of_m_l170_17060


namespace fifth_number_in_tenth_row_l170_17001

def nth_number_in_row (n k : ℕ) : ℕ :=
  7 * n - (7 - k)

theorem fifth_number_in_tenth_row : nth_number_in_row 10 5 = 68 :=
by
  sorry

end fifth_number_in_tenth_row_l170_17001


namespace jeans_price_difference_l170_17027

variable (x : Real)

theorem jeans_price_difference
  (hx : 0 < x) -- Assuming x > 0 for a positive cost
  (r := 1.40 * x)
  (c := 1.30 * r) :
  c = 1.82 * x :=
by
  sorry

end jeans_price_difference_l170_17027


namespace solve_system_eqns_l170_17068

theorem solve_system_eqns 
  {a b c : ℝ} (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
  {x y z : ℝ} 
  (h4 : a^3 + a^2 * x + a * y + z = 0)
  (h5 : b^3 + b^2 * x + b * y + z = 0)
  (h6 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + bc + ca ∧ z = -abc :=
by {
  sorry
}

end solve_system_eqns_l170_17068


namespace jackson_collection_goal_l170_17075

theorem jackson_collection_goal 
  (days_in_week : ℕ)
  (goal : ℕ)
  (earned_mon : ℕ)
  (earned_tue : ℕ)
  (avg_collect_per_4house : ℕ)
  (remaining_days : ℕ)
  (remaining_goal : ℕ)
  (daily_target : ℕ)
  (collect_per_house : ℚ)
  :
  days_in_week = 5 →
  goal = 1000 →
  earned_mon = 300 →
  earned_tue = 40 →
  avg_collect_per_4house = 10 →
  remaining_goal = goal - earned_mon - earned_tue →
  remaining_days = days_in_week - 2 →
  daily_target = remaining_goal / remaining_days →
  collect_per_house = avg_collect_per_4house / 4 →
  (daily_target : ℚ) / collect_per_house = 88 := 
by sorry

end jackson_collection_goal_l170_17075


namespace maria_trip_distance_l170_17003

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + ((D / 2) / 4) + 150 = D) 
  (h2 : 150 = 3 * D / 8) : 
  D = 400 :=
by
  -- Placeholder for the actual proof
  sorry

end maria_trip_distance_l170_17003


namespace union_of_sets_l170_17035

def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}
def C : Set ℝ := {x | x > -2}

theorem union_of_sets (A B : Set ℝ) : (A ∪ B) = C :=
  sorry

end union_of_sets_l170_17035


namespace trenton_earning_goal_l170_17032

-- Parameters
def fixed_weekly_earnings : ℝ := 190
def commission_rate : ℝ := 0.04
def sales_amount : ℝ := 7750
def goal : ℝ := 500

-- Proof statement
theorem trenton_earning_goal :
  fixed_weekly_earnings + (commission_rate * sales_amount) = goal :=
by
  sorry

end trenton_earning_goal_l170_17032


namespace intersection_A_B_l170_17051

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end intersection_A_B_l170_17051


namespace rainfall_sunday_l170_17042

theorem rainfall_sunday 
  (rain_sun rain_mon rain_tue : ℝ)
  (h1 : rain_mon = rain_sun + 3)
  (h2 : rain_tue = 2 * rain_mon)
  (h3 : rain_sun + rain_mon + rain_tue = 25) :
  rain_sun = 4 :=
by
  sorry

end rainfall_sunday_l170_17042


namespace area_of_triangle_l170_17058

namespace TriangleArea

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

noncomputable def area (A B C : Point3D) : ℚ :=
  let x1 := A.x
  let y1 := A.y
  let z1 := A.z
  let x2 := B.x
  let y2 := B.y
  let z2 := B.z
  let x3 := C.x
  let y3 := C.y
  let z3 := C.z
  1 / 2 * ( (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)) )

def A : Point3D := ⟨0, 3, 6⟩
def B : Point3D := ⟨-2, 2, 2⟩
def C : Point3D := ⟨-5, 5, 2⟩

theorem area_of_triangle : area A B C = 4.5 :=
by
  sorry

end TriangleArea

end area_of_triangle_l170_17058


namespace seven_k_plus_four_l170_17015

theorem seven_k_plus_four (k m n : ℕ) (h1 : 4 * k + 5 = m^2) (h2 : 9 * k + 4 = n^2) (hk : k = 5) : 
  7 * k + 4 = 39 :=
by 
  -- assume conditions
  have h1' := h1
  have h2' := h2
  have hk' := hk
  sorry

end seven_k_plus_four_l170_17015


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l170_17078

variable {a b m : ℝ}

theorem sufficient_but_not_necessary_condition (h : a * m^2 < b * m^2) : a < b := by
  sorry

-- Additional statements to express the sufficiency and not necessity nature:
theorem not_necessary_condition (h : a < b) (hm : m = 0) : ¬ (a * m^2 < b * m^2) := by
  sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l170_17078


namespace find_f_20_l170_17009

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_20 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = f (2 - x)) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x - 1 / 2) →
  f 20 = - 1 / 2 :=
sorry

end find_f_20_l170_17009


namespace brick_width_l170_17091

variable (w : ℝ)

theorem brick_width :
  ∃ (w : ℝ), 2 * (10 * w + 10 * 3 + 3 * w) = 164 → w = 4 :=
by
  sorry

end brick_width_l170_17091


namespace solve_for_2023_minus_a_minus_2b_l170_17044

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end solve_for_2023_minus_a_minus_2b_l170_17044


namespace min_m_squared_plus_n_squared_l170_17069

theorem min_m_squared_plus_n_squared {m n : ℝ} (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) :
  m^2 + n^2 = 2 :=
sorry

end min_m_squared_plus_n_squared_l170_17069


namespace victor_cannot_escape_k4_l170_17056

theorem victor_cannot_escape_k4
  (r : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ) 
  (k : ℝ)
  (hr : r = 1)
  (hk : k = 4)
  (hA_speed : speed_A = 4 * speed_B)
  (B_starts_at_center : ∃ (B : ℝ), B = 0):
  ¬(∃ (escape_strategy : ℝ → ℝ), escape_strategy 0 = 0 → escape_strategy r = 1) :=
sorry

end victor_cannot_escape_k4_l170_17056


namespace apple_bags_l170_17081

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l170_17081


namespace rationalize_denominator_l170_17036

theorem rationalize_denominator :
  (7 / (Real.sqrt 175 - Real.sqrt 75)) = (7 * (Real.sqrt 7 + Real.sqrt 3) / 20) :=
by
  have h1 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 75 = 5 * Real.sqrt 3 := sorry
  sorry

end rationalize_denominator_l170_17036


namespace problem_solution_l170_17020

theorem problem_solution (a b c d e f g : ℝ) 
  (h1 : a + b + e = 7)
  (h2 : b + c + f = 10)
  (h3 : c + d + g = 6)
  (h4 : e + f + g = 9) : 
  a + d + g = 6 := 
sorry

end problem_solution_l170_17020


namespace isosceles_triangle_height_eq_four_times_base_l170_17039

theorem isosceles_triangle_height_eq_four_times_base (b h : ℝ) 
    (same_area : (b * 2 * b) = (1/2 * b * h)) : 
    h = 4 * b :=
by 
  -- sorry allows us to skip the proof steps
  sorry

end isosceles_triangle_height_eq_four_times_base_l170_17039


namespace max_checkers_on_chessboard_l170_17099

theorem max_checkers_on_chessboard : 
  ∃ (w b : ℕ), (∀ r c : ℕ, r < 8 ∧ c < 8 → w = 2 * b) ∧ (8 * (w + b) = 48) ∧ (w + b) * 8 ≤ 64 :=
by sorry

end max_checkers_on_chessboard_l170_17099


namespace sum_of_first_eight_terms_l170_17002

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l170_17002


namespace first_floor_cost_l170_17092

-- Definitions and assumptions
variables (F : ℝ)
variables (earnings_first_floor earnings_second_floor earnings_third_floor : ℝ)
variables (total_monthly_earnings : ℝ)

-- Conditions from the problem
def costs := F
def second_floor_costs := F + 20
def third_floor_costs := 2 * F
def first_floor_rooms := 3 * costs
def second_floor_rooms := 3 * second_floor_costs
def third_floor_rooms := 3 * third_floor_costs

-- Total monthly earnings
def total_earnings := first_floor_rooms + second_floor_rooms + third_floor_rooms

-- Equality condition
axiom total_earnings_is_correct : total_earnings = 165

-- Theorem to be proved
theorem first_floor_cost :
  (F = 8.75) :=
by
  have earnings_first_floor_eq := first_floor_rooms
  have earnings_second_floor_eq := second_floor_rooms
  have earnings_third_floor_eq := third_floor_rooms
  have total_earning_eq := total_earnings_is_correct
  sorry

end first_floor_cost_l170_17092


namespace investment_amount_l170_17093

-- Conditions and given problem rewrite in Lean 4
theorem investment_amount (P y : ℝ) (h1 : P * y * 2 / 100 = 500) (h2 : P * (1 + y / 100) ^ 2 - P = 512.50) : P = 5000 :=
sorry

end investment_amount_l170_17093


namespace greatest_difference_54_l170_17054

theorem greatest_difference_54 (board : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 100) :
  ∃ i j k l, (i = k ∨ j = l) ∧ (board i j - board k l ≥ 54 ∨ board k l - board i j ≥ 54) :=
sorry

end greatest_difference_54_l170_17054


namespace proportion_of_white_pieces_l170_17084

theorem proportion_of_white_pieces (x : ℕ) (h1 : 0 < x) :
  let total_pieces := 3 * x
  let white_pieces := x + (1 - (5 / 9)) * x
  (white_pieces / total_pieces) = (13 / 27) :=
by
  sorry

end proportion_of_white_pieces_l170_17084


namespace value_of_x_plus_y_l170_17087

theorem value_of_x_plus_y (x y : ℤ) (h1 : x + 2 = 10) (h2 : y - 1 = 6) : x + y = 15 :=
by
  sorry

end value_of_x_plus_y_l170_17087


namespace correct_operation_l170_17076

theorem correct_operation (a : ℝ) :
  (a^5)^2 = a^10 :=
by sorry

end correct_operation_l170_17076


namespace find_f_neg_one_l170_17005

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f (x^2 + y) = f x + f (y^2)

theorem find_f_neg_one : f (-1) = 0 := sorry

end find_f_neg_one_l170_17005


namespace min_value_of_expression_l170_17007

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) : ∃ z : ℝ, (z = 4^x + (1/2)^y) ∧ z = 8 :=
by 
  sorry

end min_value_of_expression_l170_17007


namespace total_maggots_served_l170_17010

-- Define the conditions in Lean
def maggots_first_attempt : ℕ := 10
def maggots_second_attempt : ℕ := 10

-- Define the statement to prove
theorem total_maggots_served : maggots_first_attempt + maggots_second_attempt = 20 :=
by 
  sorry

end total_maggots_served_l170_17010


namespace closest_points_distance_l170_17047

theorem closest_points_distance :
  let center1 := (2, 2)
  let center2 := (17, 10)
  let radius1 := 2
  let radius2 := 10
  let distance_centers := Nat.sqrt ((center2.1 - center1.1) ^ 2 + (center2.2 - center1.2) ^ 2)
  distance_centers = 17 → (distance_centers - radius1 - radius2) = 5 := by
  sorry

end closest_points_distance_l170_17047


namespace benjamin_speed_l170_17006

-- Define the problem conditions
def distance : ℕ := 800 -- Distance in kilometers
def time : ℕ := 10 -- Time in hours

-- Define the main statement
theorem benjamin_speed : distance / time = 80 := by
  sorry

end benjamin_speed_l170_17006


namespace total_pencils_is_54_l170_17066

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end total_pencils_is_54_l170_17066


namespace divisor_of_425904_l170_17021

theorem divisor_of_425904 :
  ∃ (d : ℕ), d = 7 ∧ ∃ (n : ℕ), n = 425897 + 7 ∧ 425904 % d = 0 :=
by
  sorry

end divisor_of_425904_l170_17021


namespace average_speed_correct_l170_17085

noncomputable def total_distance := 120 + 70
noncomputable def total_time := 2
noncomputable def average_speed := total_distance / total_time

theorem average_speed_correct :
  average_speed = 95 := by
  sorry

end average_speed_correct_l170_17085


namespace find_2a_plus_b_l170_17083

noncomputable def f (a b x : ℝ) : ℝ := a * x - b
noncomputable def g (x : ℝ) : ℝ := -4 * x + 6
noncomputable def h (a b x : ℝ) : ℝ := f a b (g x)
noncomputable def h_inv (x : ℝ) : ℝ := x + 9

theorem find_2a_plus_b (a b : ℝ) (h_inv_eq: ∀ x : ℝ, h a b (h_inv x) = x) : 2 * a + b = 7 :=
sorry

end find_2a_plus_b_l170_17083


namespace proof_problem_l170_17034

variable {R : Type} [OrderedRing R]

-- Definitions and conditions
variable (g : R → R) (f : R → R) (k a m : R)
variable (h_odd : ∀ x : R, g (-x) = -g x)
variable (h_f_def : ∀ x : R, f x = g x + k)
variable (h_f_neg_a : f (-a) = m)

-- Theorem statement
theorem proof_problem : f a = 2 * k - m :=
by
  -- Here is where the proof would go.
  sorry

end proof_problem_l170_17034


namespace cos_alpha_value_l170_17011

open Real

theorem cos_alpha_value (α : ℝ) (h_cos : cos (α - π/6) = 15/17) (h_range : π/6 < α ∧ α < π/2) : 
  cos α = (15 * Real.sqrt 3 - 8) / 34 :=
by
  sorry

end cos_alpha_value_l170_17011


namespace can_capacity_l170_17004

-- Definitions of the conditions
variable (M W : ℕ) -- initial amounts of milk and water
variable (M' : ℕ := M + 2) -- new amount of milk after adding 2 liters
variable (ratio_initial : M / W = 1 / 5)
variable (ratio_new : M' / W = 3 / 5)

theorem can_capacity (M W : ℕ) (h_ratio_initial : M / W = 1 / 5) (h_ratio_new : (M + 2) / W = 3 / 5) : (M + W + 2) = 8 := 
by
  sorry

end can_capacity_l170_17004


namespace remi_spilled_second_time_l170_17073

-- Defining the conditions from the problem
def bottle_capacity : ℕ := 20
def daily_refills : ℕ := 3
def total_days : ℕ := 7
def total_water_consumed : ℕ := 407
def first_spill : ℕ := 5

-- Using the conditions to define the total amount of water that Remi would have drunk without spilling.
def no_spill_total : ℕ := bottle_capacity * daily_refills * total_days

-- Defining the second spill
def second_spill : ℕ := no_spill_total - first_spill - total_water_consumed

-- Stating the theorem that we need to prove
theorem remi_spilled_second_time : second_spill = 8 :=
by
  sorry

end remi_spilled_second_time_l170_17073


namespace rectangular_prism_edge_sum_l170_17050

theorem rectangular_prism_edge_sum
  (V A : ℝ)
  (hV : V = 8)
  (hA : A = 32)
  (l w h : ℝ)
  (geom_prog : l = w / h ∧ w = l * h ∧ h = l * (w / l)) :
  4 * (l + w + h) = 28 :=
by 
  sorry

end rectangular_prism_edge_sum_l170_17050


namespace quadratic_inequality_solution_is_interval_l170_17013

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x : ℝ | -3*x^2 + 9*x + 12 > 0 }

theorem quadratic_inequality_solution_is_interval :
  quadratic_inequality_solution = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end quadratic_inequality_solution_is_interval_l170_17013
