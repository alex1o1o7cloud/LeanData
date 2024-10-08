import Mathlib

namespace initial_balloons_blown_up_l184_184380
-- Import necessary libraries

-- Define the statement
theorem initial_balloons_blown_up (x : ℕ) (hx : x + 13 = 60) : x = 47 :=
by
  sorry

end initial_balloons_blown_up_l184_184380


namespace area_of_field_l184_184176

-- Define the given conditions and the problem
theorem area_of_field (L W A : ℝ) (hL : L = 20) (hFencing : 2 * W + L = 88) (hA : A = L * W) : 
  A = 680 :=
by
  sorry

end area_of_field_l184_184176


namespace henry_final_price_l184_184060

-- Definitions based on the conditions in the problem
def price_socks : ℝ := 5
def price_tshirt : ℝ := price_socks + 10
def price_jeans : ℝ := 2 * price_tshirt
def discount_jeans : ℝ := 0.15 * price_jeans
def discounted_price_jeans : ℝ := price_jeans - discount_jeans
def sales_tax_jeans : ℝ := 0.08 * discounted_price_jeans
def final_price_jeans : ℝ := discounted_price_jeans + sales_tax_jeans

-- Statement to prove
theorem henry_final_price : final_price_jeans = 27.54 := by
  sorry

end henry_final_price_l184_184060


namespace largest_divisor_of_expression_l184_184867

theorem largest_divisor_of_expression (x : ℤ) (hx : x % 2 = 1) :
  864 ∣ (12 * x + 2) * (12 * x + 6) * (12 * x + 10) * (6 * x + 3) :=
sorry

end largest_divisor_of_expression_l184_184867


namespace percentage_students_passed_l184_184773

theorem percentage_students_passed
    (total_students : ℕ)
    (students_failed : ℕ)
    (students_passed : ℕ)
    (percentage_passed : ℕ)
    (h1 : total_students = 840)
    (h2 : students_failed = 546)
    (h3 : students_passed = total_students - students_failed)
    (h4 : percentage_passed = (students_passed * 100) / total_students) :
    percentage_passed = 35 := by
  sorry

end percentage_students_passed_l184_184773


namespace taxi_fare_80_miles_l184_184550

theorem taxi_fare_80_miles (fare_60 : ℝ) (flat_rate : ℝ) (proportional_rate : ℝ) (d : ℝ) (charge_60 : ℝ) 
  (h1 : fare_60 = 150) (h2 : flat_rate = 20) (h3 : proportional_rate * 60 = charge_60) (h4 : charge_60 = (fare_60 - flat_rate)) 
  (h5 : proportional_rate * 80 = d - flat_rate) : d = 193 := 
by
  sorry

end taxi_fare_80_miles_l184_184550


namespace erica_time_is_65_l184_184069

-- Definitions for the conditions
def dave_time : ℕ := 10
def chuck_time : ℕ := 5 * dave_time
def erica_time : ℕ := chuck_time + 3 * chuck_time / 10

-- The proof statement
theorem erica_time_is_65 : erica_time = 65 := by
  sorry

end erica_time_is_65_l184_184069


namespace cyclic_quadrilateral_equality_l184_184869

variables {A B C D : ℝ} (AB BC CD DA AC BD : ℝ)

theorem cyclic_quadrilateral_equality 
  (h_cyclic: A * B * C * D = AB * BC * CD * DA)
  (h_sides: AB = A ∧ BC = B ∧ CD = C ∧ DA = D)
  (h_diagonals: AC = E ∧ BD = F) :
  E * (A * B + C * D) = F * (D * A + B * C) :=
sorry

end cyclic_quadrilateral_equality_l184_184869


namespace mod_inverse_non_existence_mod_inverse_existence_l184_184940

theorem mod_inverse_non_existence (a b c d : ℕ) (h1 : 1105 = a * b * c) (h2 : 15 = d * a) :
    ¬ ∃ x : ℕ, (15 * x) % 1105 = 1 := by sorry

theorem mod_inverse_existence (a b : ℕ) (h1 : 221 = a * b) :
    ∃ x : ℕ, (15 * x) % 221 = 59 := by sorry

end mod_inverse_non_existence_mod_inverse_existence_l184_184940


namespace probability_one_boy_one_girl_l184_184576

-- Define the total number of students (5), the number of boys (3), and the number of girls (2).
def total_students : Nat := 5
def boys : Nat := 3
def girls : Nat := 2

-- Define the probability calculation in Lean.
noncomputable def select_2_students_prob : ℚ :=
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose boys 1 * Nat.choose girls 1
  favorable_combinations / total_combinations

-- The statement we need to prove is that this probability is 3/5
theorem probability_one_boy_one_girl : select_2_students_prob = 3 / 5 := sorry

end probability_one_boy_one_girl_l184_184576


namespace range_of_m_l184_184971

theorem range_of_m (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + m - 1 = 0 ∧ x2^2 - 4 * x2 + m - 1 = 0 ∧ x1 ≠ x2) ∧ 
  (3 * (m - 1) - 4 > 2) →

  3 < m ∧ m ≤ 5 :=
sorry

end range_of_m_l184_184971


namespace certain_number_is_3_l184_184252

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

end certain_number_is_3_l184_184252


namespace find_m_l184_184373

theorem find_m (f : ℝ → ℝ) (m : ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_fx : ∀ x, 0 < x → f x = 4^(m - x)) 
  (h_f_neg2 : f (-2) = 1/8) : 
  m = 1/2 := 
by 
  sorry

end find_m_l184_184373


namespace jamie_paid_0_more_than_alex_l184_184876

/-- Conditions:
     1. Alex and Jamie shared a pizza cut into 10 equally-sized slices.
     2. Alex wanted a plain pizza.
     3. Jamie wanted a special spicy topping on one-third of the pizza.
     4. The cost of a plain pizza was $10.
     5. The spicy topping on one-third of the pizza cost an additional $3.
     6. Jamie ate all the slices with the spicy topping and two extra plain slices.
     7. Alex ate the remaining plain slices.
     8. They each paid for what they ate.
    
     Question: How many more dollars did Jamie pay than Alex?
     Answer: 0
-/
theorem jamie_paid_0_more_than_alex :
  let total_slices := 10
  let cost_plain := 10
  let cost_spicy := 3
  let total_cost := cost_plain + cost_spicy
  let cost_per_slice := total_cost / total_slices
  let jamie_slices := 5
  let alex_slices := total_slices - jamie_slices
  let jamie_cost := jamie_slices * cost_per_slice
  let alex_cost := alex_slices * cost_per_slice
  jamie_cost - alex_cost = 0 :=
by
  sorry

end jamie_paid_0_more_than_alex_l184_184876


namespace original_cost_price_l184_184606

theorem original_cost_price (C : ℝ) (h : C + 0.15 * C + 0.05 * C + 0.10 * C = 6400) : C = 4923 :=
by
  sorry

end original_cost_price_l184_184606


namespace smallest_non_representable_l184_184440

def isRepresentable (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ n : ℕ, 0 < n → ¬ isRepresentable 11 ∧ ∀ k : ℕ, 0 < k ∧ k < 11 → isRepresentable k :=
by sorry

end smallest_non_representable_l184_184440


namespace find_s_l184_184934

section
variables {a b c p q s : ℕ}

-- Conditions given in the problem
variables (h1 : a + b = p)
variables (h2 : p + c = s)
variables (h3 : s + a = q)
variables (h4 : b + c + q = 18)
variables (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
variables (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0)

-- Statement of the problem
theorem find_s (h1 : a + b = p) (h2 : p + c = s) (h3 : s + a = q) (h4 : b + c + q = 18)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
  (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0) :
  s = 9 :=
sorry
end

end find_s_l184_184934


namespace tom_total_distance_l184_184850

/-- Tom swims for 1.5 hours at 2.5 miles per hour. 
    Tom runs for 0.75 hours at 6.5 miles per hour. 
    Tom bikes for 3 hours at 12 miles per hour. 
    The total distance Tom covered is 44.625 miles.
-/
theorem tom_total_distance
  (swim_time : ℝ := 1.5) (swim_speed : ℝ := 2.5)
  (run_time : ℝ := 0.75) (run_speed : ℝ := 6.5)
  (bike_time : ℝ := 3) (bike_speed : ℝ := 12) :
  swim_time * swim_speed + run_time * run_speed + bike_time * bike_speed = 44.625 :=
by
  sorry

end tom_total_distance_l184_184850


namespace sum_mod_7_eq_5_l184_184837

theorem sum_mod_7_eq_5 : 
  (51730 + 51731 + 51732 + 51733 + 51734 + 51735) % 7 = 5 := 
by 
  sorry

end sum_mod_7_eq_5_l184_184837


namespace smallest_positive_integer_l184_184146

theorem smallest_positive_integer (
  a : ℕ
) : 
  (a ≡ 5 [MOD 6]) ∧ (a ≡ 7 [MOD 8]) → a = 23 :=
by sorry

end smallest_positive_integer_l184_184146


namespace greatest_four_digit_n_l184_184774

theorem greatest_four_digit_n :
  ∃ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) ∧ (∃ m : ℕ, n + 1 = m^2) ∧ ¬(n! % (n * (n + 1) / 2) = 0) ∧ n = 9999 :=
by sorry

end greatest_four_digit_n_l184_184774


namespace miles_mike_ride_l184_184232

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

end miles_mike_ride_l184_184232


namespace lines_through_P_and_form_area_l184_184782

-- Definition of the problem conditions
def passes_through_P (k b : ℝ) : Prop :=
  b = 2 - k

def forms_area_with_axes (k b : ℝ) : Prop :=
  b^2 = 8 * |k|

-- Theorem statement
theorem lines_through_P_and_form_area :
  ∃ (k1 k2 k3 b1 b2 b3 : ℝ),
    passes_through_P k1 b1 ∧ forms_area_with_axes k1 b1 ∧
    passes_through_P k2 b2 ∧ forms_area_with_axes k2 b2 ∧
    passes_through_P k3 b3 ∧ forms_area_with_axes k3 b3 ∧
    k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 :=
sorry

end lines_through_P_and_form_area_l184_184782


namespace simplified_expression_result_l184_184445

theorem simplified_expression_result :
  ((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by {
  sorry
}

end simplified_expression_result_l184_184445


namespace smallest_four_digit_product_is_12_l184_184024

theorem smallest_four_digit_product_is_12 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
           (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 12 ∧ a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6) ∧
           (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 →
                     (∃ a' b' c' d' : ℕ, m = 1000 * a' + 100 * b' + 10 * c' + d' ∧ a' * b' * c' * d' = 12) →
                     n ≤ m) :=
by
  sorry

end smallest_four_digit_product_is_12_l184_184024


namespace line_passes_through_parabola_vertex_l184_184575

theorem line_passes_through_parabola_vertex :
  ∃ (a : ℝ), (∃ (b : ℝ), b = a ∧ (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_parabola_vertex_l184_184575


namespace closest_multiple_of_21_to_2023_l184_184035

theorem closest_multiple_of_21_to_2023 : ∃ k : ℤ, k * 21 = 2022 ∧ ∀ m : ℤ, m * 21 = 2023 → (abs (m - 2023)) > (abs (2022 - 2023)) :=
by
  sorry

end closest_multiple_of_21_to_2023_l184_184035


namespace envelope_weight_l184_184270

theorem envelope_weight (E : ℝ) :
  (8 * (1 / 5) + E ≤ 2) ∧ (1 < 8 * (1 / 5) + E) ∧ (E ≥ 0) ↔ E = 2 / 5 :=
by
  sorry

end envelope_weight_l184_184270


namespace least_integer_sol_l184_184075

theorem least_integer_sol (x : ℤ) (h : |(2 : ℤ) * x + 7| ≤ 16) : x ≥ -11 := sorry

end least_integer_sol_l184_184075


namespace price_per_package_l184_184918

theorem price_per_package (P : ℝ) (hp1 : 10 * P + 50 * (4 / 5 * P) = 1096) :
  P = 21.92 :=
by 
  sorry

end price_per_package_l184_184918


namespace towels_after_a_week_l184_184966

theorem towels_after_a_week 
  (initial_green : ℕ) (initial_white : ℕ) (initial_blue : ℕ) 
  (daily_green : ℕ) (daily_white : ℕ) (daily_blue : ℕ) 
  (days : ℕ) 
  (H1 : initial_green = 35)
  (H2 : initial_white = 21)
  (H3 : initial_blue = 15)
  (H4 : daily_green = 3)
  (H5 : daily_white = 1)
  (H6 : daily_blue = 1)
  (H7 : days = 7) :
  (initial_green - daily_green * days) + (initial_white - daily_white * days) + (initial_blue - daily_blue * days) = 36 :=
by 
  sorry

end towels_after_a_week_l184_184966


namespace horner_method_multiplications_and_additions_l184_184278

noncomputable def f (x : ℕ) : ℕ :=
  12 * x ^ 6 + 5 * x ^ 5 + 11 * x ^ 2 + 2 * x + 5

theorem horner_method_multiplications_and_additions (x : ℕ) :
  let multiplications := 6
  let additions := 4
  multiplications = 6 ∧ additions = 4 :=
sorry

end horner_method_multiplications_and_additions_l184_184278


namespace carrie_profit_l184_184965

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end carrie_profit_l184_184965


namespace students_in_first_class_l184_184081

variable (x : ℕ)
variable (avg_marks_first_class : ℕ := 40)
variable (num_students_second_class : ℕ := 28)
variable (avg_marks_second_class : ℕ := 60)
variable (avg_marks_all : ℕ := 54)

theorem students_in_first_class : (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := 
by 
  sorry

end students_in_first_class_l184_184081


namespace pictures_per_coloring_book_l184_184094

theorem pictures_per_coloring_book
    (total_colored : ℕ)
    (remaining_pictures : ℕ)
    (two_books : ℕ)
    (h1 : total_colored = 20) 
    (h2 : remaining_pictures = 68) 
    (h3 : two_books = 2) :
  (total_colored + remaining_pictures) / two_books = 44 :=
by
  sorry

end pictures_per_coloring_book_l184_184094


namespace berry_saturday_reading_l184_184485

-- Given data
def sunday_pages := 43
def monday_pages := 65
def tuesday_pages := 28
def wednesday_pages := 0
def thursday_pages := 70
def friday_pages := 56
def average_goal := 50
def days_in_week := 7

-- Calculate total pages to meet the weekly goal
def weekly_goal := days_in_week * average_goal

-- Calculate pages read so far from Sunday to Friday
def pages_read := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

-- Calculate required pages to read on Saturday
def saturday_pages_required := weekly_goal - pages_read

-- The theorem statement: Berry needs to read 88 pages on Saturday.
theorem berry_saturday_reading : saturday_pages_required = 88 := 
by {
  -- The proof is omitted as per the instructions
  sorry
}

end berry_saturday_reading_l184_184485


namespace number_of_people_in_group_l184_184742

variable (T L : ℕ)

theorem number_of_people_in_group
  (h1 : 90 + L = T)
  (h2 : (L : ℚ) / T = 0.4) :
  T = 150 := by
  sorry

end number_of_people_in_group_l184_184742


namespace function_for_negative_x_l184_184353

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem function_for_negative_x {f : ℝ → ℝ} :
  odd_function f → given_function f → ∀ x, x < 0 → f x = x * (1 + x) :=
by
  intros h1 h2
  sorry

end function_for_negative_x_l184_184353


namespace find_number_l184_184234

theorem find_number (N : ℝ) (h : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N) : N = 180 :=
by 
  sorry

end find_number_l184_184234


namespace only_prime_such_that_2p_plus_one_is_perfect_power_l184_184789

theorem only_prime_such_that_2p_plus_one_is_perfect_power :
  ∃ (p : ℕ), p ≤ 1000 ∧ Prime p ∧ ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 :=
by
  sorry

end only_prime_such_that_2p_plus_one_is_perfect_power_l184_184789


namespace correct_operator_is_subtraction_l184_184019

theorem correct_operator_is_subtraction :
  (8 - 2) + 5 * (3 - 2) = 11 :=
by
  sorry

end correct_operator_is_subtraction_l184_184019


namespace candidate_percentage_l184_184790

theorem candidate_percentage (P : ℕ) (total_votes : ℕ) (vote_diff : ℕ)
  (h1 : total_votes = 7000)
  (h2 : vote_diff = 2100)
  (h3 : (P * total_votes / 100) + (P * total_votes / 100) + vote_diff = total_votes) :
  P = 35 :=
by
  sorry

end candidate_percentage_l184_184790


namespace ball_bounces_17_times_to_reach_below_2_feet_l184_184372

theorem ball_bounces_17_times_to_reach_below_2_feet:
  ∃ k: ℕ, (∀ n, n < k → (800 * ((2: ℝ) / 3) ^ n) ≥ 2) ∧ (800 * ((2: ℝ) / 3) ^ k < 2) ∧ k = 17 :=
by
  sorry

end ball_bounces_17_times_to_reach_below_2_feet_l184_184372


namespace molecular_weight_of_1_mole_l184_184498

theorem molecular_weight_of_1_mole (W_5 : ℝ) (W_1 : ℝ) (h : 5 * W_1 = W_5) (hW5 : W_5 = 490) : W_1 = 490 :=
by
  sorry

end molecular_weight_of_1_mole_l184_184498


namespace binary_addition_to_decimal_l184_184799

theorem binary_addition_to_decimal : (2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0)
                                     + (2^5 + 2^4 + 2^3 + 2^2) = 571 := by
  sorry

end binary_addition_to_decimal_l184_184799


namespace product_of_A_and_B_l184_184142

theorem product_of_A_and_B (A B : ℕ) (h1 : 3 / 9 = 6 / A) (h2 : B / 63 = 6 / A) : A * B = 378 :=
  sorry

end product_of_A_and_B_l184_184142


namespace Claire_plans_to_buy_five_cookies_l184_184319

theorem Claire_plans_to_buy_five_cookies :
  let initial_amount := 100
  let latte_cost := 3.75
  let croissant_cost := 3.50
  let days := 7
  let cookie_cost := 1.25
  let remaining_amount := 43
  let daily_expense := latte_cost + croissant_cost
  let weekly_expense := daily_expense * days
  let total_spent := initial_amount - remaining_amount
  let cookie_spent := total_spent - weekly_expense
  let cookies := cookie_spent / cookie_cost
  cookies = 5 :=
by {
  sorry
}

end Claire_plans_to_buy_five_cookies_l184_184319


namespace sequence_general_formula_l184_184033

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 + n * 5 

theorem sequence_general_formula (n : ℕ) : n > 0 → sequence_term n = 5 * n - 2 :=
by 
  sorry

end sequence_general_formula_l184_184033


namespace greatest_divisor_of_420_and_90_l184_184638

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- Main problem statement
theorem greatest_divisor_of_420_and_90 {d : ℕ} :
  (divides d 420) ∧ (d < 60) ∧ (divides d 90) → d ≤ 30 := 
sorry

end greatest_divisor_of_420_and_90_l184_184638


namespace train_speed_is_72_kmh_l184_184513

-- Length of the train in meters
def length_train : ℕ := 600

-- Length of the platform in meters
def length_platform : ℕ := 600

-- Time to cross the platform in minutes
def time_crossing_platform : ℕ := 1

-- Convert meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Convert minutes to hours
def minutes_to_hours (m : ℕ) : ℕ := m * 60

-- Speed of the train in km/hr given lengths in meters and time in minutes
def speed_train_kmh (distance_m : ℕ) (time_min : ℕ) : ℕ :=
  (meters_to_kilometers distance_m) / (minutes_to_hours time_min)

theorem train_speed_is_72_kmh :
  speed_train_kmh (length_train + length_platform) time_crossing_platform = 72 :=
by
  -- skipping the proof
  sorry

end train_speed_is_72_kmh_l184_184513


namespace tangent_line_at_1_f_positive_iff_a_leq_2_l184_184562

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_1 (a : ℝ) (h : a = 4) : 
  ∃ k b : ℝ, (k = -2) ∧ (b = 2) ∧ (∀ x : ℝ, f x a = k * (x - 1) + b) :=
sorry

theorem f_positive_iff_a_leq_2 : 
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_at_1_f_positive_iff_a_leq_2_l184_184562


namespace find_parabola_eq_find_range_of_b_l184_184707

-- Problem 1: Finding the equation of the parabola
theorem find_parabola_eq (p : ℝ) (h1 : p > 0) (x1 x2 y1 y2 : ℝ) 
  (A : (x1 + 4) * 2 = 2 * p * y1) (C : (x2 + 4) * 2 = 2 * p * y2)
  (h3 : x1^2 = 2 * p * y1) (h4 : x2^2 = 2 * p * y2) 
  (h5 : y2 = 4 * y1) :
  x1^2 = 4 * y1 :=
sorry

-- Problem 2: Finding the range of b
theorem find_range_of_b (k : ℝ) (h : k > 0 ∨ k < -4) : 
  ∃ b : ℝ, b = 2 * (k + 1)^2 ∧ b > 2 :=
sorry

end find_parabola_eq_find_range_of_b_l184_184707


namespace length_of_AD_l184_184705

theorem length_of_AD 
  (A B C D : Type) 
  (vertex_angle_equal: ∀ {a b c d : Type}, a = A →
    ∀ (AB AC AD : ℝ), (AB = 24) → (AC = 54) → (AD = 36)) 
  (right_triangles : ∀ {a b : Type}, a = A → ∀ {AB AC : ℝ}, (AB > 0) → (AC > 0) → (AB ^ 2 + AC ^ 2 = AD ^ 2)) :
  ∃ (AD : ℝ), AD = 36 :=
by
  sorry

end length_of_AD_l184_184705


namespace closest_ratio_of_adults_to_children_l184_184031

def total_fees (a c : ℕ) : ℕ := 20 * a + 10 * c
def adults_children_equation (a c : ℕ) : Prop := 2 * a + c = 160

theorem closest_ratio_of_adults_to_children :
  ∃ a c : ℕ, 
    total_fees a c = 1600 ∧
    a ≥ 1 ∧ c ≥ 1 ∧
    adults_children_equation a c ∧
    (∀ a' c' : ℕ, total_fees a' c' = 1600 ∧ 
        a' ≥ 1 ∧ c' ≥ 1 ∧ 
        adults_children_equation a' c' → 
        abs ((a : ℝ) / c - 1) ≤ abs ((a' : ℝ) / c' - 1)) :=
  sorry

end closest_ratio_of_adults_to_children_l184_184031


namespace right_triangle_of_pythagorean_l184_184536

theorem right_triangle_of_pythagorean
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC CA : ℝ)
  (h : AB^2 = BC^2 + CA^2) : ∃ (c : ℕ), c = 90 :=
by
  sorry

end right_triangle_of_pythagorean_l184_184536


namespace sufficient_condition_p_or_q_false_p_and_q_false_l184_184819

variables (p q : Prop)

theorem sufficient_condition_p_or_q_false_p_and_q_false :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ¬ ( (¬ (p ∧ q)) → ¬ (p ∨ q)) :=
by 
  -- Proof: If ¬ (p ∨ q), then (p ∨ q) is false, which means (p ∧ q) must also be false.
  -- The other direction would mean if at least one of p or q is false, then (p ∨ q) is false,
  -- which is not necessarily true. Therefore, it's not a necessary condition.
  sorry

end sufficient_condition_p_or_q_false_p_and_q_false_l184_184819


namespace catherine_bottle_caps_l184_184032

-- Definitions from conditions
def friends : ℕ := 6
def caps_per_friend : ℕ := 3

-- Theorem statement from question and correct answer
theorem catherine_bottle_caps : friends * caps_per_friend = 18 :=
by sorry

end catherine_bottle_caps_l184_184032


namespace num_ways_to_distribute_7_balls_in_4_boxes_l184_184706

def num_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  -- Implement the function to calculate the number of ways here, but we'll keep it as a placeholder for now.
  sorry

theorem num_ways_to_distribute_7_balls_in_4_boxes : 
  num_ways_to_distribute_balls 7 4 = 3 := 
sorry

end num_ways_to_distribute_7_balls_in_4_boxes_l184_184706


namespace men_became_absent_l184_184435

theorem men_became_absent (num_men absent : ℤ) 
  (num_men_eq : num_men = 180) 
  (days_planned : ℤ) (days_planned_eq : days_planned = 55)
  (days_taken : ℤ) (days_taken_eq : days_taken = 60)
  (work_planned : ℤ) (work_planned_eq : work_planned = num_men * days_planned)
  (work_taken : ℤ) (work_taken_eq : work_taken = (num_men - absent) * days_taken)
  (work_eq : work_planned = work_taken) :
  absent = 15 :=
  by sorry

end men_became_absent_l184_184435


namespace power_sum_l184_184277

theorem power_sum :
  (-1:ℤ)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 :=
by
  sorry

end power_sum_l184_184277


namespace total_paint_correct_l184_184528

-- Define the current gallons of paint he has
def current_paint : ℕ := 36

-- Define the gallons of paint he bought
def bought_paint : ℕ := 23

-- Define the additional gallons of paint he needs
def needed_paint : ℕ := 11

-- The total gallons of paint he needs for finishing touches
def total_paint_needed : ℕ := current_paint + bought_paint + needed_paint

-- The proof statement to show that the total paint needed is 70
theorem total_paint_correct : total_paint_needed = 70 := by
  sorry

end total_paint_correct_l184_184528


namespace age_problem_solution_l184_184314

theorem age_problem_solution 
  (x : ℕ) 
  (xiaoxiang_age : ℕ := 5) 
  (father_age : ℕ := 48) 
  (mother_age : ℕ := 42) 
  (h : (father_age + x) + (mother_age + x) = 6 * (xiaoxiang_age + x)) : 
  x = 15 :=
by {
  -- To be proved
  sorry
}

end age_problem_solution_l184_184314


namespace period_of_f_g_is_2_sin_x_g_is_odd_l184_184257

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

end period_of_f_g_is_2_sin_x_g_is_odd_l184_184257


namespace shirt_pants_outfits_l184_184056

theorem shirt_pants_outfits
  (num_shirts : ℕ) (num_pants : ℕ) (num_formal_pants : ℕ) (num_casual_pants : ℕ) (num_assignee_shirts : ℕ) :
  num_shirts = 5 →
  num_pants = 6 →
  num_formal_pants = 3 →
  num_casual_pants = 3 →
  num_assignee_shirts = 3 →
  (num_casual_pants * num_shirts) + (num_formal_pants * num_assignee_shirts) = 24 :=
by
  intros h_shirts h_pants h_formal h_casual h_assignee
  sorry

end shirt_pants_outfits_l184_184056


namespace find_x_l184_184287

theorem find_x (x : ℝ) (h : (2015 + x)^2 = x^2) : x = -2015 / 2 := by
  sorry

end find_x_l184_184287


namespace largest_perfect_square_factor_1800_l184_184666

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  if n = 1800 then 900 else sorry

theorem largest_perfect_square_factor_1800 : 
  largest_perfect_square_factor 1800 = 900 :=
by
  -- Proof is not needed, so we use sorry
  sorry

end largest_perfect_square_factor_1800_l184_184666


namespace intersection_M_N_l184_184849

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

theorem intersection_M_N : M ∩ N = {0, 3} := by
  sorry

end intersection_M_N_l184_184849


namespace complement_union_l184_184524

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 3 }
noncomputable def CR (S : Set ℝ) : Set ℝ := { x : ℝ | x ∉ S }

theorem complement_union (A B : Set ℝ) :
  (CR A ∪ B) = (Set.univ \ A ∪ Set.Ioo 1 3) := by
  sorry

end complement_union_l184_184524


namespace inequality_solution_set_l184_184714

noncomputable def solution_set := { x : ℝ | 0 < x ∧ x < 2 }

theorem inequality_solution_set : 
  { x : ℝ | (4 / x > |x|) } = solution_set :=
by sorry

end inequality_solution_set_l184_184714


namespace exists_positive_integers_for_hexagon_area_l184_184357

theorem exists_positive_integers_for_hexagon_area (S : ℕ) (a b : ℕ) (hS : S = 2016) :
  2 * (a^2 + b^2 + a * b) = S → ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 2 * (a^2 + b^2 + a * b) = S :=
by
  sorry

end exists_positive_integers_for_hexagon_area_l184_184357


namespace max_diff_six_digit_even_numbers_l184_184361

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l184_184361


namespace solve_system_unique_solution_l184_184588

theorem solve_system_unique_solution:
  ∃! (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ x = 57 / 31 ∧ y = 97 / 31 := by
  sorry

end solve_system_unique_solution_l184_184588


namespace num_units_from_batch_B_l184_184137

theorem num_units_from_batch_B
  (A B C : ℝ) -- quantities of products from batches A, B, and C
  (h_arith_seq : B - A = C - B) -- batches A, B, and C form an arithmetic sequence
  (h_total : A + B + C = 240)    -- total units from three batches
  (h_sample_size : A + B + C = 60)  -- sample size drawn equals 60
  : B = 20 := 
by {
  sorry
}

end num_units_from_batch_B_l184_184137


namespace pythagorean_triangle_exists_l184_184686

theorem pythagorean_triangle_exists (a : ℤ) (h : a ≥ 5) : 
  ∃ (b c : ℤ), c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 :=
by {
  sorry
}

end pythagorean_triangle_exists_l184_184686


namespace petya_time_l184_184405

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l184_184405


namespace shortest_distance_l184_184652

-- Define the line and the circle
def is_on_line (P : ℝ × ℝ) : Prop := P.snd = P.fst - 1

def is_on_circle (Q : ℝ × ℝ) : Prop := Q.fst^2 + Q.snd^2 + 4 * Q.fst - 2 * Q.snd + 4 = 0

-- Define the square of the Euclidean distance between two points
def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.fst - Q.fst)^2 + (P.snd - Q.snd)^2

-- State the theorem regarding the shortest distance between the points on the line and the circle
theorem shortest_distance : ∃ P Q : ℝ × ℝ, is_on_line P ∧ is_on_circle Q ∧ dist_squared P Q = 1 := sorry

end shortest_distance_l184_184652


namespace GAUSS_1998_LCM_l184_184515

/-- The periodicity of cycling the word 'GAUSS' -/
def period_GAUSS : ℕ := 5

/-- The periodicity of cycling the number '1998' -/
def period_1998 : ℕ := 4

/-- The least common multiple (LCM) of the periodicities of 'GAUSS' and '1998' is 20 -/
theorem GAUSS_1998_LCM : Nat.lcm period_GAUSS period_1998 = 20 :=
by
  sorry

end GAUSS_1998_LCM_l184_184515


namespace original_garden_side_length_l184_184061

theorem original_garden_side_length (a : ℝ) (h : (a + 3)^2 = 2 * a^2 + 9) : a = 6 :=
by
  sorry

end original_garden_side_length_l184_184061


namespace expression_value_l184_184463

-- Define the difference of squares identity
lemma diff_of_squares (x y : ℤ) : x^2 - y^2 = (x + y) * (x - y) :=
by sorry

-- Define the specific values for x and y
def x := 7
def y := 3

-- State the theorem to be proven
theorem expression_value : ((x^2 - y^2)^2) = 1600 :=
by sorry

end expression_value_l184_184463


namespace tan_alpha_second_quadrant_l184_184883

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_second_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.cos (π / 2 - α) = 4 / 5) :
  tan_alpha α = -4 / 3 :=
by
  sorry

end tan_alpha_second_quadrant_l184_184883


namespace jamie_workday_percent_l184_184500

theorem jamie_workday_percent
  (total_work_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_multiplier : ℕ)
  (break_minutes : ℕ)
  (total_minutes_per_hour : ℕ)
  (total_work_minutes : ℕ)
  (first_meeting_duration : ℕ)
  (second_meeting_duration : ℕ)
  (total_meeting_time : ℕ)
  (percentage_spent : ℚ) :
  total_work_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 2 →
  break_minutes = 30 →
  total_minutes_per_hour = 60 →
  total_work_minutes = total_work_hours * total_minutes_per_hour →
  first_meeting_duration = first_meeting_minutes →
  second_meeting_duration = second_meeting_multiplier * first_meeting_duration →
  total_meeting_time = first_meeting_duration + second_meeting_duration + break_minutes →
  percentage_spent = (total_meeting_time : ℚ) / (total_work_minutes : ℚ) * 100 →
  percentage_spent = 35 :=
sorry

end jamie_workday_percent_l184_184500


namespace part_1_a_part_1_b_part_2_l184_184436

open Set

variable (a : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def compl_U_A : Set ℝ := compl A

theorem part_1_a :
  A ∩ B 1 = {x : ℝ | x < -2} :=
by
  sorry

theorem part_1_b :
  A ∪ B 1 = {x : ℝ | x > 2 ∨ x ≤ 1} :=
by
  sorry

theorem part_2 :
  compl_U_A ⊆ B a → a ≥ 2 :=
by
  sorry

end part_1_a_part_1_b_part_2_l184_184436


namespace part1_solution_part2_solution_l184_184631

-- Definitions for propositions p and q
def p (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- The actual Lean 4 statements
theorem part1_solution (x : ℝ) (m : ℝ) (hm : m = 1) (hp : p m x) (hq : q x) : 2 ≤ x ∧ x < 3 := by
  sorry

theorem part2_solution (m : ℝ) (hm : m > 0) (hsuff : ∀ x, q x → p m x) : (4 / 3) < m ∧ m < 2 := by
  sorry

end part1_solution_part2_solution_l184_184631


namespace prove_total_weekly_allowance_l184_184573

noncomputable def total_weekly_allowance : ℕ :=
  let students := 200
  let group1 := students * 45 / 100
  let group2 := students * 30 / 100
  let group3 := students * 15 / 100
  let group4 := students - group1 - group2 - group3  -- Remaining students
  let daily_allowance := group1 * 6 + group2 * 4 + group3 * 7 + group4 * 10
  daily_allowance * 7

theorem prove_total_weekly_allowance :
  total_weekly_allowance = 8330 := by
  sorry

end prove_total_weekly_allowance_l184_184573


namespace max_sum_terms_arithmetic_seq_l184_184438

theorem max_sum_terms_arithmetic_seq (a1 d : ℝ) (h1 : a1 > 0) 
  (h2 : 3 * (2 * a1 + 2 * d) = 11 * (2 * a1 + 10 * d)) :
  ∃ (n : ℕ),  (∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d > 0) ∧  a1 + n * d ≤ 0 ∧ n = 7 :=
by
  sorry

end max_sum_terms_arithmetic_seq_l184_184438


namespace tin_can_allocation_l184_184830

-- Define the total number of sheets of tinplate available
def total_sheets := 108

-- Define the number of sheets used for can bodies
variable (x : ℕ)

-- Define the number of can bodies a single sheet makes
def can_bodies_per_sheet := 15

-- Define the number of can bottoms a single sheet makes
def can_bottoms_per_sheet := 42

-- Define the equation to be proven
theorem tin_can_allocation :
  2 * can_bodies_per_sheet * x = can_bottoms_per_sheet * (total_sheets - x) :=
  sorry

end tin_can_allocation_l184_184830


namespace largest_expression_is_A_l184_184348

def expr_A := 1 - 2 + 3 + 4
def expr_B := 1 + 2 - 3 + 4
def expr_C := 1 + 2 + 3 - 4
def expr_D := 1 + 2 - 3 - 4
def expr_E := 1 - 2 - 3 + 4

theorem largest_expression_is_A : expr_A = 6 ∧ expr_A > expr_B ∧ expr_A > expr_C ∧ expr_A > expr_D ∧ expr_A > expr_E :=
  by sorry

end largest_expression_is_A_l184_184348


namespace fraction_simplification_l184_184143

theorem fraction_simplification (x : ℝ) (h: x ≠ 1) : (5 * x / (x - 1) - 5 / (x - 1)) = 5 := 
sorry

end fraction_simplification_l184_184143


namespace sum_of_digits_of_greatest_prime_divisor_l184_184213

-- Define the number 32767
def number := 32767

-- Find the greatest prime divisor of 32767
def greatest_prime_divisor : ℕ :=
  127

-- Prove the sum of the digits of the greatest prime divisor is 10
theorem sum_of_digits_of_greatest_prime_divisor (h : greatest_prime_divisor = 127) : (1 + 2 + 7) = 10 :=
  sorry

end sum_of_digits_of_greatest_prime_divisor_l184_184213


namespace min_inequality_l184_184681

theorem min_inequality (r s u v : ℝ) : 
  min (min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2)))) ≤ 1 / 4 :=
by sorry

end min_inequality_l184_184681


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l184_184800

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l184_184800


namespace analysis_method_correct_answer_l184_184424

axiom analysis_def (conclusion: Prop): 
  ∃ sufficient_conditions: (Prop → Prop), 
    (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)

theorem analysis_method_correct_answer :
  ∀ (conclusion : Prop) , ∃ sufficient_conditions: (Prop → Prop), 
  (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)
:= by 
  intros 
  sorry

end analysis_method_correct_answer_l184_184424


namespace find_bounds_l184_184244

open Set

variable {U : Type} [TopologicalSpace U]

def A := {x : ℝ | 3 ≤ x ∧ x ≤ 4}
def C_UA := {x : ℝ | x > 4 ∨ x < 3}

theorem find_bounds (T : Type) [TopologicalSpace T] : 3 = 3 ∧ 4 = 4 := 
 by sorry

end find_bounds_l184_184244


namespace minimize_square_sum_l184_184752

theorem minimize_square_sum (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  ∃ x y z, (x + 2 * y + 3 * z = 1) ∧ (x^2 + y^2 + z^2 ≥ 0) ∧ ((x^2 + y^2 + z^2) = 1 / 14) :=
sorry

end minimize_square_sum_l184_184752


namespace root_implies_quadratic_eq_l184_184043

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end root_implies_quadratic_eq_l184_184043


namespace proof_stage_constancy_l184_184884

-- Definitions of stages
def Stage1 := "Fertilization and seed germination"
def Stage2 := "Flowering and pollination"
def Stage3 := "Meiosis and fertilization"
def Stage4 := "Formation of sperm and egg cells"

-- Question: Which stages maintain chromosome constancy and promote genetic recombination in plant life?
def Q := "Which stages maintain chromosome constancy and promote genetic recombination in plant life?"

-- Correct answer
def Answer := Stage3

-- Conditions
def s1 := Stage1
def s2 := Stage2
def s3 := Stage3
def s4 := Stage4

-- Theorem statement
theorem proof_stage_constancy : Q = Answer := by
  sorry

end proof_stage_constancy_l184_184884


namespace min_balls_to_ensure_20_l184_184226

theorem min_balls_to_ensure_20 (red green yellow blue purple white black : ℕ) (Hred : red = 30) (Hgreen : green = 25) (Hyellow : yellow = 18) (Hblue : blue = 15) (Hpurple : purple = 12) (Hwhite : white = 10) (Hblack : black = 7) :
  ∀ n, n ≥ 101 → (∃ r g y b p w bl, r + g + y + b + p + w + bl = n ∧ (r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ p ≥ 20 ∨ w ≥ 20 ∨ bl ≥ 20)) :=
by
  intro n hn
  sorry

end min_balls_to_ensure_20_l184_184226


namespace pencil_distribution_l184_184748

-- Formalize the problem in Lean
theorem pencil_distribution (x1 x2 x3 x4 : ℕ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 5) (hx2 : 1 ≤ x2 ∧ x2 ≤ 5) (hx3 : 1 ≤ x3 ∧ x3 ≤ 5) (hx4 : 1 ≤ x4 ∧ x4 ≤ 5) :
  x1 + x2 + x3 + x4 = 10 → 64 = 64 :=
by {
  sorry
}

end pencil_distribution_l184_184748


namespace greatest_two_digit_multiple_of_17_l184_184126

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l184_184126


namespace find_number_l184_184425

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
sorry

end find_number_l184_184425


namespace find_cosine_of_dihedral_angle_l184_184454

def dihedral_cosine (R r : ℝ) (α β : ℝ) : Prop :=
  R = 2 * r ∧ β = Real.pi / 4 → Real.cos α = 8 / 9

theorem find_cosine_of_dihedral_angle : ∃ α, ∀ R r : ℝ, dihedral_cosine R r α (Real.pi / 4) :=
sorry

end find_cosine_of_dihedral_angle_l184_184454


namespace pow_congr_mod_eight_l184_184004

theorem pow_congr_mod_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 := sorry

end pow_congr_mod_eight_l184_184004


namespace polynomial_third_and_fourth_equal_l184_184552

theorem polynomial_third_and_fourth_equal (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1)
  (h_eq : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = (8 : ℝ) / 11 :=
by
  sorry

end polynomial_third_and_fourth_equal_l184_184552


namespace problem_statement_l184_184393

theorem problem_statement (n : ℕ) (a b c : ℕ → ℤ)
  (h1 : n > 0)
  (h2 : ∀ i j, i ≠ j → ¬ (a i - a j) % n = 0 ∧
                           ¬ ((b i + c i) - (b j + c j)) % n = 0 ∧
                           ¬ (b i - b j) % n = 0 ∧
                           ¬ ((c i + a i) - (c j + a i)) % n = 0 ∧
                           ¬ (c i - c j) % n = 0 ∧
                           ¬ ((a i + b i) - (a j + b i)) % n = 0 ∧
                           ¬ ((a i + b i + c i) - (a j + b i + c j)) % n = 0) :
  (Odd n) ∧ (¬ ∃ k, n = 3 * k) :=
by sorry

end problem_statement_l184_184393


namespace clock_displays_unique_digits_minutes_l184_184113

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l184_184113


namespace range_of_b_l184_184325

theorem range_of_b (f g : ℝ → ℝ) (a b : ℝ)
  (hf : ∀ x, f x = Real.exp x - 1)
  (hg : ∀ x, g x = -x^2 + 4*x - 3)
  (h : f a = g b) :
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by
  sorry

end range_of_b_l184_184325


namespace solve_congruence_l184_184421

theorem solve_congruence (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 :=
by {
  sorry
}

end solve_congruence_l184_184421


namespace union_A_B_complement_intersect_B_intersection_sub_C_l184_184321

-- Define set A
def A : Set ℝ := {x | -5 < x ∧ x < 1}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 8}

-- Define set C with variable parameter a
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Problem (1): Prove A ∪ B = { x | -5 < x < 8 }
theorem union_A_B : A ∪ B = {x | -5 < x ∧ x < 8} := 
by sorry

-- Problem (1): Prove (complement R A) ∩ B = { x | 1 ≤ x < 8 }
theorem complement_intersect_B : (Aᶜ) ∩ B = {x | 1 ≤ x ∧ x < 8} :=
by sorry

-- Problem (2): If A ∩ B ⊆ C, prove a ≥ 1
theorem intersection_sub_C (a : ℝ) (h : A ∩ B ⊆ C a) : 1 ≤ a :=
by sorry

end union_A_B_complement_intersect_B_intersection_sub_C_l184_184321


namespace weigh_1_to_10_kg_l184_184942

theorem weigh_1_to_10_kg (n : ℕ) : 1 ≤ n ∧ n ≤ 10 →
  ∃ (a b c : ℤ), 
    (abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
    (n = a * 3 + b * 4 + c * 9)) :=
by sorry

end weigh_1_to_10_kg_l184_184942


namespace like_terms_sum_l184_184305

theorem like_terms_sum (m n : ℕ) (a b : ℝ) :
  (∀ c d : ℝ, -4 * a^(2 * m) * b^(3) = c * a^(6) * b^(n + 1)) →
  m + n = 5 :=
by 
  intro h
  sorry

end like_terms_sum_l184_184305


namespace number_of_lists_correct_l184_184665

noncomputable def number_of_lists : Nat :=
  15 ^ 4

theorem number_of_lists_correct :
  number_of_lists = 50625 := by
  sorry

end number_of_lists_correct_l184_184665


namespace part_one_solution_part_two_solution_l184_184767

-- (I) Prove the solution set for the given inequality with m = 2.
theorem part_one_solution (x : ℝ) : 
  (|x - 2| > 7 - |x - 1|) ↔ (x < -4 ∨ x > 5) :=
sorry

-- (II) Prove the range of m given the condition.
theorem part_two_solution (m : ℝ) : 
  (∃ x : ℝ, |x - m| > 7 + |x - 1|) ↔ (m ∈ Set.Iio (-6) ∪ Set.Ioi (8)) :=
sorry

end part_one_solution_part_two_solution_l184_184767


namespace paula_remaining_money_l184_184999

-- Define the given conditions
def given_amount : ℕ := 109
def cost_shirt : ℕ := 11
def number_shirts : ℕ := 2
def cost_pants : ℕ := 13

-- Calculate total spending
def total_spent : ℕ := (cost_shirt * number_shirts) + cost_pants

-- Define the remaining amount Paula has
def remaining_amount : ℕ := given_amount - total_spent

-- State the theorem
theorem paula_remaining_money : remaining_amount = 74 := by
  -- Proof goes here
  sorry

end paula_remaining_money_l184_184999


namespace incorrect_simplification_l184_184196

theorem incorrect_simplification :
  (-(1 + 1/2) ≠ 1 + 1/2) := 
by sorry

end incorrect_simplification_l184_184196


namespace bella_stamps_l184_184318

theorem bella_stamps :
  let snowflake_cost := 1.05
  let truck_cost := 1.20
  let rose_cost := 0.90
  let butterfly_cost := 1.15
  let snowflake_spent := 15.75
  
  let snowflake_stamps := snowflake_spent / snowflake_cost
  let truck_stamps := snowflake_stamps + 11
  let rose_stamps := truck_stamps - 17
  let butterfly_stamps := 1.5 * rose_stamps
  
  let total_stamps := snowflake_stamps + truck_stamps + rose_stamps + butterfly_stamps
  
  total_stamps = 64 := by
  sorry

end bella_stamps_l184_184318


namespace quadratic_distinct_roots_k_range_l184_184473

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_distinct_roots_k_range_l184_184473


namespace g_five_eq_one_l184_184354

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x - y) = g x * g y)
variable (h_ne_zero : ∀ x : ℝ, g x ≠ 0)

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l184_184354


namespace geometric_sum_S12_l184_184835

theorem geometric_sum_S12 
  (S : ℕ → ℝ)
  (h_S4 : S 4 = 2) 
  (h_S8 : S 8 = 6) 
  (geom_property : ∀ n, (S (2 * n + 4) - S n) ^ 2 = S n * (S (3 * n + 4) - S (2 * n + 4))) 
  : S 12 = 14 := 
by sorry

end geometric_sum_S12_l184_184835


namespace third_smallest_is_four_probability_l184_184327

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l184_184327


namespace pq_difference_l184_184923

theorem pq_difference (p q : ℝ) (h1 : 3 / p = 6) (h2 : 3 / q = 15) : p - q = 3 / 10 := by
  sorry

end pq_difference_l184_184923


namespace triangle_acute_angle_contradiction_l184_184921

theorem triangle_acute_angle_contradiction
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_tri : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_at_most_one_acute : (α < 90 ∧ β ≥ 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β < 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β ≥ 90 ∧ γ < 90)) :
  false :=
by
  sorry

end triangle_acute_angle_contradiction_l184_184921


namespace percentage_loss_is_25_l184_184897

def cost_price := 1400
def selling_price := 1050
def loss := cost_price - selling_price
def percentage_loss := (loss / cost_price) * 100

theorem percentage_loss_is_25 : percentage_loss = 25 := by
  sorry

end percentage_loss_is_25_l184_184897


namespace contrapositive_of_proposition_l184_184058

theorem contrapositive_of_proposition (a b : ℝ) : (a > b → a + 1 > b) ↔ (a + 1 ≤ b → a ≤ b) :=
sorry

end contrapositive_of_proposition_l184_184058


namespace necessary_but_not_sufficient_l184_184656

variable (a : ℝ)

theorem necessary_but_not_sufficient : (a > 2) → (a > 1) ∧ ¬((a > 1) → (a > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l184_184656


namespace divide_bill_evenly_l184_184794

variable (totalBill amountPaid : ℕ)
variable (numberOfFriends : ℕ)

theorem divide_bill_evenly (h1 : totalBill = 135) (h2 : amountPaid = 45) (h3 : numberOfFriends * amountPaid = totalBill) :
  numberOfFriends = 3 := by
  sorry

end divide_bill_evenly_l184_184794


namespace actual_distance_traveled_l184_184875

theorem actual_distance_traveled (D : ℕ) 
  (h : D / 10 = (D + 36) / 16) : D = 60 := by
  sorry

end actual_distance_traveled_l184_184875


namespace probability_at_least_one_black_ball_l184_184570

theorem probability_at_least_one_black_ball :
  let total_balls := 6
  let red_balls := 2
  let white_ball := 1
  let black_balls := 3
  let total_combinations := Nat.choose total_balls 2
  let non_black_combinations := Nat.choose (total_balls - black_balls) 2
  let probability := 1 - (non_black_combinations / total_combinations : ℚ)
  probability = 4 / 5 :=
by
  sorry

end probability_at_least_one_black_ball_l184_184570


namespace part_one_solution_set_part_two_range_of_a_l184_184303

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part_one_solution_set :
  { x : ℝ | f x ≤ 9 } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

theorem part_two_range_of_a (a : ℝ) (B := { x : ℝ | x^2 - 3 * x < 0 })
  (A := { x : ℝ | f x < 2 * x + a }) :
  B ⊆ A → 5 ≤ a :=
sorry

end part_one_solution_set_part_two_range_of_a_l184_184303


namespace pythagorean_theorem_l184_184112

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l184_184112


namespace perpendicular_lines_l184_184987

theorem perpendicular_lines :
  (∀ (x y : ℝ), (4 * y - 3 * x = 16)) ∧ 
  (∀ (x y : ℝ), (3 * y + 4 * x = 15)) → 
  (∃ (m1 m2 : ℝ), m1 * m2 = -1) :=
by
  sorry

end perpendicular_lines_l184_184987


namespace people_lost_l184_184507

-- Define the given constants
def win_ratio : ℕ := 4
def lose_ratio : ℕ := 1
def people_won : ℕ := 28

-- The statement to prove that 7 people lost
theorem people_lost (win_ratio lose_ratio people_won : ℕ) (H : win_ratio * 7 = people_won * lose_ratio) : 7 = people_won * lose_ratio / win_ratio :=
by { sorry }

end people_lost_l184_184507


namespace arithmetic_sequence_a5_value_l184_184814

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) 
  (h1 : a 2 + a 4 = 16) 
  (h2 : a 1 = 1) : 
  a 5 = 15 := 
by 
  sorry

end arithmetic_sequence_a5_value_l184_184814


namespace quadruples_solution_l184_184620

theorem quadruples_solution (a b c d : ℝ) :
  (a * b + c * d = 6) ∧
  (a * c + b * d = 3) ∧
  (a * d + b * c = 2) ∧
  (a + b + c + d = 6) ↔
  (a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0) :=
sorry

end quadruples_solution_l184_184620


namespace blue_paint_amount_l184_184495

/-- 
Prove that if Giselle uses 15 quarts of white paint, then according to the ratio 4:3:5, she should use 12 quarts of blue paint.
-/
theorem blue_paint_amount (white_paint : ℚ) (h1 : white_paint = 15) : 
  let blue_ratio := 4;
  let white_ratio := 5;
  blue_ratio / white_ratio * white_paint = 12 :=
by
  sorry

end blue_paint_amount_l184_184495


namespace range_of_x_for_sqrt_meaningful_l184_184072

theorem range_of_x_for_sqrt_meaningful (x : ℝ) (h : x + 2 ≥ 0) : x ≥ -2 :=
by {
  sorry
}

end range_of_x_for_sqrt_meaningful_l184_184072


namespace find_slope_l184_184452

theorem find_slope (k : ℝ) : (∃ x : ℝ, (y = k * x + 2) ∧ (y = 0) ∧ (abs x = 4)) ↔ (k = 1/2 ∨ k = -1/2) := by
  sorry

end find_slope_l184_184452


namespace garden_area_increase_l184_184634

theorem garden_area_increase :
  let length := 80
  let width := 20
  let additional_fence := 60
  let original_area := length * width
  let original_perimeter := 2 * (length + width)
  let total_fence := original_perimeter + additional_fence
  let side_of_square := total_fence / 4
  let square_area := side_of_square * side_of_square
  square_area - original_area = 2625 :=
by
  sorry

end garden_area_increase_l184_184634


namespace all_equal_l184_184879

variable (a : ℕ → ℝ)

axiom h1 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0
axiom h2 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0
axiom h3 : a 3 - 3 * a 4 + 2 * a 5 ≥ 0
axiom h4 : ∀ n, 4 ≤ n ∧ n ≤ 98 → a n - 3 * a (n + 1) + 2 * a (n + 2) ≥ 0
axiom h99 : a 99 - 3 * a 100 + 2 * a 1 ≥ 0
axiom h100 : a 100 - 3 * a 1 + 2 * a 2 ≥ 0

theorem all_equal : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 → a i = a j := by
  sorry

end all_equal_l184_184879


namespace range_of_a_l184_184225

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, 
    1 ≤ x ∧ x ≤ 2 ∧ 
    2 ≤ y ∧ y ≤ 3 → 
    x * y ≤ a * x^2 + 2 * y^2) ↔ 
  a ≥ -1 :=
by
  sorry

end range_of_a_l184_184225


namespace quadratic_is_perfect_square_l184_184586

theorem quadratic_is_perfect_square (a b c x : ℝ) (h : b^2 - 4 * a * c = 0) :
  a * x^2 + b * x + c = 0 ↔ (2 * a * x + b)^2 = 0 := 
by
  sorry

end quadratic_is_perfect_square_l184_184586


namespace rhombus_area_in_rectangle_l184_184089

theorem rhombus_area_in_rectangle :
  ∀ (l w : ℝ), 
  (∀ (A B C D : ℝ), 
    (2 * w = l) ∧ 
    (l * w = 72) →
    let diag1 := w 
    let diag2 := l 
    (1/2 * diag1 * diag2 = 36)) :=
by
  intros
  sorry

end rhombus_area_in_rectangle_l184_184089


namespace valid_range_for_b_l184_184330

noncomputable def f (x b : ℝ) : ℝ := -x^2 + 2 * x + b^2 - b + 1

theorem valid_range_for_b (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x b > 0) → (b < -1 ∨ b > 2) :=
by
  sorry

end valid_range_for_b_l184_184330


namespace range_of_a_part1_range_of_a_part2_l184_184764

theorem range_of_a_part1 (a : ℝ) :
  (∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) → 0 < a ∧ a < 4 :=
sorry

theorem range_of_a_part2 (a : ℝ) :
  ((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧ ¬((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a ≤ 0 ∨ (1 / 4) < a ∧ a < 4 :=
sorry

end range_of_a_part1_range_of_a_part2_l184_184764


namespace people_in_house_l184_184276

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l184_184276


namespace max_a2_b2_c2_d2_l184_184211

-- Define the conditions for a, b, c, d
variables (a b c d : ℝ) 

-- Define the hypotheses from the problem
variables (h₁ : a + b = 17)
variables (h₂ : ab + c + d = 94)
variables (h₃ : ad + bc = 195)
variables (h₄ : cd = 120)

-- Define the final statement to be proved
theorem max_a2_b2_c2_d2 : ∃ (a b c d : ℝ), a + b = 17 ∧ ab + c + d = 94 ∧ ad + bc = 195 ∧ cd = 120 ∧ (a^2 + b^2 + c^2 + d^2) = 918 :=
by sorry

end max_a2_b2_c2_d2_l184_184211


namespace speed_calculation_l184_184809

def distance := 600 -- in meters
def time := 2 -- in minutes

def distance_km := distance / 1000 -- converting meters to kilometers
def time_hr := time / 60 -- converting minutes to hours

theorem speed_calculation : (distance_km / time_hr = 18) :=
 by
  sorry

end speed_calculation_l184_184809


namespace remaining_seeds_l184_184559

def initial_seeds : Nat := 54000
def seeds_per_zone : Nat := 3123
def number_of_zones : Nat := 7

theorem remaining_seeds (initial_seeds seeds_per_zone number_of_zones : Nat) : 
  initial_seeds - (seeds_per_zone * number_of_zones) = 32139 := 
by 
  sorry

end remaining_seeds_l184_184559


namespace percentage_discount_l184_184715

theorem percentage_discount (original_price sale_price : ℝ) (h1 : original_price = 25) (h2 : sale_price = 18.75) : 
  100 * (original_price - sale_price) / original_price = 25 := 
by
  -- Begin Proof
  sorry

end percentage_discount_l184_184715


namespace common_difference_arithmetic_geometric_sequence_l184_184988

theorem common_difference_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ r, ∀ n, a (n+1) = a n * r)
  (h_a1 : a 1 = 1) :
  d = 0 :=
by
  sorry

end common_difference_arithmetic_geometric_sequence_l184_184988


namespace exactly_two_succeed_probability_l184_184413

-- Define the probabilities of events A, B, and C decrypting the code
def P_A_decrypts : ℚ := 1/5
def P_B_decrypts : ℚ := 1/4
def P_C_decrypts : ℚ := 1/3

-- Define the probabilities of events A, B, and C not decrypting the code
def P_A_not_decrypts : ℚ := 1 - P_A_decrypts
def P_B_not_decrypts : ℚ := 1 - P_B_decrypts
def P_C_not_decrypts : ℚ := 1 - P_C_decrypts

-- Define the probability that exactly two out of A, B, and C decrypt the code
def P_exactly_two_succeed : ℚ :=
  (P_A_decrypts * P_B_decrypts * P_C_not_decrypts) +
  (P_A_decrypts * P_B_not_decrypts * P_C_decrypts) +
  (P_A_not_decrypts * P_B_decrypts * P_C_decrypts)

-- Prove that this probability is equal to 3/20
theorem exactly_two_succeed_probability : P_exactly_two_succeed = 3 / 20 := by
  sorry

end exactly_two_succeed_probability_l184_184413


namespace prime_factor_of_difference_l184_184023

theorem prime_factor_of_difference {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (h_neq : A ≠ B) :
  Nat.Prime 2 ∧ (∃ B : ℕ, 20 * B = 20 * B) :=
by
  sorry

end prime_factor_of_difference_l184_184023


namespace isosceles_triangle_base_length_l184_184808

open Real

noncomputable def average_distance_sun_earth : ℝ := 1.5 * 10^8 -- in kilometers
noncomputable def base_length_given_angle_one_second (legs_length : ℝ) : ℝ := 4.848 -- in millimeters when legs are 1 kilometer

theorem isosceles_triangle_base_length 
  (vertex_angle : ℝ) (legs_length : ℝ) 
  (h1 : vertex_angle = 1 / 3600) 
  (h2 : legs_length = average_distance_sun_earth) : 
  ∃ base_length: ℝ, base_length = 727.2 := 
by 
  sorry

end isosceles_triangle_base_length_l184_184808


namespace third_altitude_is_less_than_15_l184_184173

variable (a b c : ℝ)
variable (ha hb hc : ℝ)
variable (A : ℝ)

def triangle_area (side : ℝ) (height : ℝ) : ℝ := 0.5 * side * height

axiom ha_eq : ha = 10
axiom hb_eq : hb = 6

theorem third_altitude_is_less_than_15 : hc < 15 :=
sorry

end third_altitude_is_less_than_15_l184_184173


namespace find_square_value_l184_184356

theorem find_square_value (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : (8 * y - 4)^2 = 202 := 
by
  sorry

end find_square_value_l184_184356


namespace base8_satisfies_l184_184044

noncomputable def check_base (c : ℕ) : Prop := 
  ((2 * c ^ 2 + 4 * c + 3) + (1 * c ^ 2 + 5 * c + 6)) = (4 * c ^ 2 + 2 * c + 1)

theorem base8_satisfies : check_base 8 := 
by
  -- conditions: (243_c, 156_c, 421_c) translated as provided
  -- proof is skipped here as specified
  sorry

end base8_satisfies_l184_184044


namespace total_pencils_correct_l184_184866

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end total_pencils_correct_l184_184866


namespace glucose_amount_in_45cc_l184_184827

noncomputable def glucose_in_container (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) : ℝ :=
  (concentration * poured_volume) / total_volume

theorem glucose_amount_in_45cc (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) :
  concentration = 10 → total_volume = 100 → poured_volume = 45 →
  glucose_in_container concentration total_volume poured_volume = 4.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end glucose_amount_in_45cc_l184_184827


namespace plane_split_into_8_regions_l184_184734

-- Define the conditions as separate lines in the plane.
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = (1 / 2) * x
def line3 (x y : ℝ) : Prop := x = y

-- Define a theorem stating that these lines together split the plane into 8 regions.
theorem plane_split_into_8_regions :
  (∀ (x y : ℝ), line1 x y ∨ line2 x y ∨ line3 x y) →
  -- The plane is split into exactly 8 regions by these lines
  ∃ (regions : ℕ), regions = 8 :=
sorry

end plane_split_into_8_regions_l184_184734


namespace island_challenge_probability_l184_184568
open Nat

theorem island_challenge_probability :
  let total_ways := choose 20 3
  let ways_one_tribe := choose 10 3
  let combined_ways := 2 * ways_one_tribe
  let probability := combined_ways / total_ways
  probability = (20 : ℚ) / 95 :=
by
  sorry

end island_challenge_probability_l184_184568


namespace horner_method_v2_l184_184028

def f(x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.reverse.foldl (λ acc c => acc * x + c) 0

theorem horner_method_v2 :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  sorry

end horner_method_v2_l184_184028


namespace hamburger_price_l184_184810

theorem hamburger_price (P : ℝ) 
    (h1 : 2 * 4 + 2 * 2 = 12) 
    (h2 : 12 * P + 4 * P = 50) : 
    P = 3.125 := 
by
  -- sorry added to skip the proof.
  sorry

end hamburger_price_l184_184810


namespace sum_ratio_15_l184_184497

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequences
def sum_a (n : ℕ) := S n
def sum_b (n : ℕ) := T n

-- The ratio condition
def ratio_condition := ∀ n, a n * (n + 1) = b n * (3 * n + 21)

theorem sum_ratio_15
  (ha : sum_a 15 = 15 * a 8)
  (hb : sum_b 15 = 15 * b 8)
  (h_ratio : ratio_condition a b) :
  sum_a 15 / sum_b 15 = 5 :=
sorry

end sum_ratio_15_l184_184497


namespace product_of_20_random_digits_ends_with_zero_l184_184105

noncomputable def probability_product_ends_in_zero : ℝ := 
  (1 - (9 / 10)^20) +
  (9 / 10)^20 * (1 - (5 / 9)^20) * (1 - (8 / 9)^19)

theorem product_of_20_random_digits_ends_with_zero : 
  abs (probability_product_ends_in_zero - 0.988) < 0.001 :=
by
  sorry

end product_of_20_random_digits_ends_with_zero_l184_184105


namespace ordered_triples_lcm_sum_zero_l184_184298

theorem ordered_triples_lcm_sum_zero :
  ∀ (x y z : ℕ), 
    (0 < x) → 
    (0 < y) → 
    (0 < z) → 
    Nat.lcm x y = 180 →
    Nat.lcm x z = 450 →
    Nat.lcm y z = 600 →
    x + y + z = 120 →
    false := 
by
  intros x y z hx hy hz hxy hxz hyz hs
  sorry

end ordered_triples_lcm_sum_zero_l184_184298


namespace clea_ride_escalator_time_l184_184823

theorem clea_ride_escalator_time
  (s v d : ℝ)
  (h1 : 75 * s = d)
  (h2 : 30 * (s + v) = d) :
  t = 50 :=
by
  sorry

end clea_ride_escalator_time_l184_184823


namespace brooke_sidney_ratio_l184_184955

-- Definitions for the conditions
def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50
def brooke_total : ℕ := 438

-- Total jumping jacks by Sidney
def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

-- The ratio of Brooke’s jumping jacks to Sidney's total jumping jacks
def ratio := brooke_total / sidney_total

-- The proof goal
theorem brooke_sidney_ratio : ratio = 3 :=
by
  sorry

end brooke_sidney_ratio_l184_184955


namespace find_n_for_divisibility_by_33_l184_184414

theorem find_n_for_divisibility_by_33 (n : ℕ) (hn_range : n < 10) (div11 : (12 - n) % 11 = 0) (div3 : (20 + n) % 3 = 0) : n = 1 :=
by {
  -- Proof steps go here
  sorry
}

end find_n_for_divisibility_by_33_l184_184414


namespace leo_current_weight_l184_184420

variables (L K J : ℝ)

def condition1 := L + 12 = 1.7 * K
def condition2 := L + K + J = 270
def condition3 := J = K + 30

theorem leo_current_weight (h1 : condition1 L K)
                           (h2 : condition2 L K J)
                           (h3 : condition3 K J) : L = 103.6 :=
sorry

end leo_current_weight_l184_184420


namespace max_value_fraction_l184_184155

theorem max_value_fraction : ∀ (x y : ℝ), (-5 ≤ x ∧ x ≤ -1) → (1 ≤ y ∧ y ≤ 3) → (1 + y / x ≤ -2) :=
  by
    intros x y hx hy
    sorry

end max_value_fraction_l184_184155


namespace necessarily_negative_l184_184622

theorem necessarily_negative (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : -2 < b ∧ b < 0) (h3 : 0 < c ∧ c < 1) : b + c < 0 :=
sorry

end necessarily_negative_l184_184622


namespace conditionA_is_necessary_for_conditionB_l184_184532

-- Definitions for conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (area : ℝ) -- area of the triangle

def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

def conditionA (t1 t2 : Triangle) : Prop :=
  t1.area = t2.area ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem conditionA_is_necessary_for_conditionB (t1 t2 : Triangle) :
  congruent t1 t2 → conditionA t1 t2 :=
by sorry

end conditionA_is_necessary_for_conditionB_l184_184532


namespace smallest_possible_value_l184_184592

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end smallest_possible_value_l184_184592


namespace cos_alpha_plus_20_eq_neg_alpha_l184_184502

variable (α : ℝ)

theorem cos_alpha_plus_20_eq_neg_alpha (h : Real.sin (α - 70 * Real.pi / 180) = α) :
    Real.cos (α + 20 * Real.pi / 180) = -α :=
by
  sorry

end cos_alpha_plus_20_eq_neg_alpha_l184_184502


namespace olivia_total_cost_l184_184111

-- Definitions based on conditions given in the problem.
def daily_rate : ℕ := 30 -- daily rate in dollars per day
def mileage_rate : ℕ := 25 -- mileage rate in cents per mile (converted to cents to avoid fractions)
def rental_days : ℕ := 3 -- number of days the car is rented
def miles_driven : ℕ := 500 -- number of miles driven

-- Calculate costs in cents to avoid fractions in the Lean theorem statement.
def daily_rental_cost : ℕ := daily_rate * rental_days * 100
def mileage_cost : ℕ := mileage_rate * miles_driven
def total_cost : ℕ := daily_rental_cost + mileage_cost

-- Final statement to be proved, converting total cost back to dollars.
theorem olivia_total_cost : (total_cost / 100) = 215 := by
  sorry

end olivia_total_cost_l184_184111


namespace multiply_98_102_l184_184009

theorem multiply_98_102 : 98 * 102 = 9996 :=
by sorry

end multiply_98_102_l184_184009


namespace box_width_l184_184450

theorem box_width (h : ℝ) (d : ℝ) (l : ℝ) (w : ℝ) 
  (h_eq_8 : h = 8)
  (l_eq_2h : l = 2 * h)
  (d_eq_20 : d = 20) :
  w = 4 * Real.sqrt 5 :=
by
  sorry

end box_width_l184_184450


namespace sequence_solution_l184_184351

theorem sequence_solution (a : ℕ → ℝ) :
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m + n) = a m + a n - m * n) ∧ 
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m * n) = m^2 * a n + n^2 * a m + 2 * a m * a n) →
    (∀ n, a n = -n*(n-1)/2) ∨ (∀ n, a n = -n^2/2) :=
  by
  sorry

end sequence_solution_l184_184351


namespace cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l184_184104

noncomputable def cricket_bat_selling_price : ℝ := 850
noncomputable def cricket_bat_profit : ℝ := 215
noncomputable def cricket_bat_cost_price : ℝ := cricket_bat_selling_price - cricket_bat_profit
noncomputable def cricket_bat_profit_percentage : ℝ := (cricket_bat_profit / cricket_bat_cost_price) * 100

noncomputable def football_selling_price : ℝ := 120
noncomputable def football_profit : ℝ := 45
noncomputable def football_cost_price : ℝ := football_selling_price - football_profit
noncomputable def football_profit_percentage : ℝ := (football_profit / football_cost_price) * 100

theorem cricket_bat_profit_percentage_correct :
  |cricket_bat_profit_percentage - 33.86| < 1e-2 :=
by sorry

theorem football_profit_percentage_correct :
  football_profit_percentage = 60 :=
by sorry

end cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l184_184104


namespace integer_solutions_count_l184_184589

theorem integer_solutions_count (B : ℤ) (C : ℤ) (h : B = 3) : C = 4 :=
by
  sorry

end integer_solutions_count_l184_184589


namespace root_in_interval_l184_184802

noncomputable def f (x: ℝ) : ℝ := x^2 + (Real.log x) - 4

theorem root_in_interval : 
  (∃ ξ ∈ Set.Ioo 1 2, f ξ = 0) :=
by
  sorry

end root_in_interval_l184_184802


namespace stratified_sampling_male_athletes_l184_184908

theorem stratified_sampling_male_athletes (total_males : ℕ) (total_females : ℕ) (sample_size : ℕ)
  (total_population : ℕ) (male_sample_fraction : ℚ) (n_sample_males : ℕ) :
  total_males = 56 →
  total_females = 42 →
  sample_size = 28 →
  total_population = total_males + total_females →
  male_sample_fraction = (sample_size : ℚ) / (total_population : ℚ) →
  n_sample_males = (total_males : ℚ) * male_sample_fraction →
  n_sample_males = 16 := by
  intros h_males h_females h_samples h_population h_fraction h_final
  sorry

end stratified_sampling_male_athletes_l184_184908


namespace equal_after_operations_l184_184597

theorem equal_after_operations :
  let initial_first_number := 365
  let initial_second_number := 24
  let first_number_after_n_operations := initial_first_number - 19 * 11
  let second_number_after_n_operations := initial_second_number + 12 * 11
  first_number_after_n_operations = second_number_after_n_operations := sorry

end equal_after_operations_l184_184597


namespace angle_B_is_pi_div_3_sin_C_value_l184_184519

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (cos_cos_eq : (2 * a - c) * Real.cos B = b * Real.cos C)
variable (triangle_ineq : 0 < A ∧ A < Real.pi)
variable (sin_positive : Real.sin A > 0)
variable (a_eq_2 : a = 2)
variable (c_eq_3 : c = 3)

-- Proving B = π / 3 under given conditions
theorem angle_B_is_pi_div_3 : B = Real.pi / 3 := sorry

-- Proving sin C under given additional conditions
theorem sin_C_value : Real.sin C = 3 * Real.sqrt 14 / 14 := sorry

end angle_B_is_pi_div_3_sin_C_value_l184_184519


namespace exists_b_c_with_integral_roots_l184_184610

theorem exists_b_c_with_integral_roots :
  ∃ (b c : ℝ), (∃ (p q : ℤ), (x^2 + b * x + c = 0) ∧ (x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
               ((x - p) * (x - q) = x^2 - (p + q) * x + p*q)) ∧
              (∃ (r s : ℤ), (x^2 + (b+1) * x + (c+1) = 0) ∧ 
              ((x - r) * (x - s) = x^2 - (r + s) * x + r*s)) :=
by
  sorry

end exists_b_c_with_integral_roots_l184_184610


namespace count_three_digit_integers_divisible_by_11_and_5_l184_184363

def count_three_digit_multiples (a b: ℕ) : ℕ :=
  let lcm := Nat.lcm a b
  let first_multiple := (100 + lcm - 1) / lcm
  let last_multiple := 999 / lcm
  last_multiple - first_multiple + 1

theorem count_three_digit_integers_divisible_by_11_and_5 : 
  count_three_digit_multiples 11 5 = 17 := by 
  sorry

end count_three_digit_integers_divisible_by_11_and_5_l184_184363


namespace Jordan_income_l184_184961

theorem Jordan_income (q A : ℝ) (h : A > 30000)
  (h1 : (q / 100 * 30000 + (q + 3) / 100 * (A - 30000) - 600) = (q + 0.5) / 100 * A) :
  A = 60000 :=
by
  sorry

end Jordan_income_l184_184961


namespace probability_no_adjacent_same_color_l184_184180

-- Define the problem space
def total_beads : ℕ := 9
def red_beads : ℕ := 4
def white_beads : ℕ := 3
def blue_beads : ℕ := 2

-- Define the total number of arrangements
def total_arrangements := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- State the probability computation theorem
theorem probability_no_adjacent_same_color :
  (∃ valid_arrangements : ℕ,
     valid_arrangements / total_arrangements = 1 / 63) := sorry

end probability_no_adjacent_same_color_l184_184180


namespace tan_sum_identity_l184_184135

theorem tan_sum_identity (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : Real.tan x = 1 / 3 := 
by 
  sorry

end tan_sum_identity_l184_184135


namespace remaining_amoeba_is_blue_l184_184890

-- Define the initial number of amoebas for red, blue, and yellow types.
def n1 := 47
def n2 := 40
def n3 := 53

-- Define the property that remains constant, i.e., the parity of differences
def parity_diff (a b : ℕ) : Bool := (a - b) % 2 == 1

-- Initial conditions based on the given problem
def initial_conditions : Prop :=
  parity_diff n1 n2 = true ∧  -- odd
  parity_diff n1 n3 = false ∧ -- even
  parity_diff n2 n3 = true    -- odd

-- Final statement: Prove that the remaining amoeba is blue
theorem remaining_amoeba_is_blue : Prop :=
  initial_conditions ∧ (∀ final : String, final = "Blue")

end remaining_amoeba_is_blue_l184_184890


namespace percentage_difference_l184_184015

open scoped Classical

theorem percentage_difference (original_number new_number : ℕ) (h₀ : original_number = 60) (h₁ : new_number = 30) :
  (original_number - new_number) / original_number * 100 = 50 :=
by
      sorry

end percentage_difference_l184_184015


namespace cars_cost_between_15000_and_20000_l184_184203

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end cars_cost_between_15000_and_20000_l184_184203


namespace personal_planner_cost_l184_184282

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

end personal_planner_cost_l184_184282


namespace museum_rid_paintings_l184_184529

def initial_paintings : ℕ := 1795
def leftover_paintings : ℕ := 1322

theorem museum_rid_paintings : initial_paintings - leftover_paintings = 473 := by
  sorry

end museum_rid_paintings_l184_184529


namespace true_propositions_identification_l184_184739

-- Definitions related to the propositions
def converse_prop1 (x y : ℝ) := (x + y = 0) → (x + y = 0)
-- Converse of additive inverses: If x and y are additive inverses, then x + y = 0
def converse_prop1_true (x y : ℝ) : Prop := (x + y = 0) → (x + y = 0)

def negation_prop2 : Prop := ¬(∀ (a b c d : ℝ), (a = b → c = d) → (a + b = c + d))
-- Negation of congruent triangles have equal areas: If two triangles are not congruent, areas not equal
def negation_prop2_false : Prop := ¬(∀ (a b c : ℝ), (a = b ∧ b ≠ c → a ≠ c))

def contrapositive_prop3 (q : ℝ) := (q ≤ 1) → (4 - 4 * q ≥ 0)
-- Contrapositive of real roots: If the equation x^2 + 2x + q = 0 does not have real roots then q > 1
def contrapositive_prop3_true (q : ℝ) : Prop := (4 - 4 * q < 0) → (q > 1)

def converse_prop4 (a b c : ℝ) := (a = b ∧ b = c ∧ c = a) → False
-- Converse of scalene triangle: If a triangle has three equal interior angles, it is a scalene triangle
def converse_prop4_false (a b c : ℝ) : Prop := (a = b ∧ b = c ∧ c = a) → False

theorem true_propositions_identification :
  (∀ x y : ℝ, converse_prop1_true x y) ∧
  ¬negation_prop2_false ∧
  (∀ q : ℝ, contrapositive_prop3_true q) ∧
  ¬(∀ a b c : ℝ, converse_prop4_false a b c) := by
  sorry

end true_propositions_identification_l184_184739


namespace ratio_of_white_to_yellow_balls_l184_184541

theorem ratio_of_white_to_yellow_balls (original_white original_yellow extra_yellow : ℕ) 
(h1 : original_white = 32) 
(h2 : original_yellow = 32) 
(h3 : extra_yellow = 20) : 
(original_white : ℚ) / (original_yellow + extra_yellow) = 8 / 13 := 
by
  sorry

end ratio_of_white_to_yellow_balls_l184_184541


namespace least_two_multiples_of_15_gt_450_l184_184594

-- Define a constant for the base multiple
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

-- Define a constant for being greater than 450
def is_greater_than_450 (n : ℕ) : Prop :=
  n > 450

-- Two least positive multiples of 15 greater than 450
theorem least_two_multiples_of_15_gt_450 :
  (is_multiple_of_15 465 ∧ is_greater_than_450 465 ∧
   is_multiple_of_15 480 ∧ is_greater_than_450 480) :=
by
  sorry

end least_two_multiples_of_15_gt_450_l184_184594


namespace friends_with_Ron_l184_184844

-- Ron is eating pizza with his friends 
def total_slices : Nat := 12
def slices_per_person : Nat := 4
def total_people := total_slices / slices_per_person
def ron_included := 1

theorem friends_with_Ron : total_people - ron_included = 2 := by
  sorry

end friends_with_Ron_l184_184844


namespace age_difference_l184_184429

def JobAge := 5
def StephanieAge := 4 * JobAge
def FreddyAge := 18

theorem age_difference : StephanieAge - FreddyAge = 2 := by
  sorry

end age_difference_l184_184429


namespace Carson_returned_l184_184001

theorem Carson_returned :
  ∀ (initial_oranges ate_oranges stolen_oranges final_oranges : ℕ), 
  initial_oranges = 60 →
  ate_oranges = 10 →
  stolen_oranges = (initial_oranges - ate_oranges) / 2 →
  final_oranges = 30 →
  final_oranges = (initial_oranges - ate_oranges - stolen_oranges) + 5 :=
by 
  sorry

end Carson_returned_l184_184001


namespace num_girls_l184_184039

-- Define conditions as constants
def ratio (B G : ℕ) : Prop := B = (5 * G) / 8
def total (B G : ℕ) : Prop := B + G = 260

-- State the proof problem
theorem num_girls (B G : ℕ) (h1 : ratio B G) (h2 : total B G) : G = 160 :=
by {
  -- actual proof omitted
  sorry
}

end num_girls_l184_184039


namespace deepak_present_age_l184_184412

theorem deepak_present_age (x : ℕ) (Rahul_age Deepak_age : ℕ) 
  (h1 : Rahul_age = 4 * x) (h2 : Deepak_age = 3 * x) 
  (h3 : Rahul_age + 4 = 32) : Deepak_age = 21 := by
  sorry

end deepak_present_age_l184_184412


namespace expected_value_equals_1_5_l184_184006

noncomputable def expected_value_win (roll : ℕ) : ℚ :=
  if roll = 1 then -1
  else if roll = 4 then -4
  else if roll = 2 ∨ roll = 3 ∨ roll = 5 ∨ roll = 7 then roll
  else 0

noncomputable def expected_value_total : ℚ :=
  (1/8 : ℚ) * ((expected_value_win 1) + (expected_value_win 2) + (expected_value_win 3) +
               (expected_value_win 4) + (expected_value_win 5) + (expected_value_win 6) +
               (expected_value_win 7) + (expected_value_win 8))

theorem expected_value_equals_1_5 : expected_value_total = 1.5 := by
  sorry

end expected_value_equals_1_5_l184_184006


namespace expected_value_l184_184172

theorem expected_value (p1 p2 p3 p4 p5 p6 : ℕ) (hp1 : p1 = 1) (hp2 : p2 = 5) (hp3 : p3 = 10) 
(hp4 : p4 = 25) (hp5 : p5 = 50) (hp6 : p6 = 100) :
  (p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + p5 / 2 + p6 / 2 : ℝ) = 95.5 := by
  sorry

end expected_value_l184_184172


namespace parallelepiped_analogy_l184_184556

-- Define plane figures and the concept of analogy for a parallelepiped 
-- (specifically here as a parallelogram) in space
inductive PlaneFigure where
  | triangle
  | parallelogram
  | trapezoid
  | rectangle

open PlaneFigure

/-- 
  Given the properties and definitions of a parallelepiped and plane figures,
  we want to show that the appropriate analogy for a parallelepiped in space
  is a parallelogram.
-/
theorem parallelepiped_analogy : 
  (analogy : PlaneFigure) = parallelogram :=
sorry

end parallelepiped_analogy_l184_184556


namespace dot_product_is_2_l184_184370

variable (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_is_2 (ha : a = (1, 0)) (hb : b = (2, 1)) :
  dot_product a b = 2 := by
  sorry

end dot_product_is_2_l184_184370


namespace abs_diff_m_n_l184_184156

variable (m n : ℝ)

theorem abs_diff_m_n (h1 : m * n = 6) (h2 : m + n = 7) (h3 : m^2 - n^2 = 13) : |m - n| = 13 / 7 :=
by
  sorry

end abs_diff_m_n_l184_184156


namespace units_digit_of_m_squared_plus_3_to_the_m_l184_184700

theorem units_digit_of_m_squared_plus_3_to_the_m (m : ℕ) (h : m = 2010^2 + 2^2010) : 
  (m^2 + 3^m) % 10 = 7 :=
by {
  sorry -- proof goes here
}

end units_digit_of_m_squared_plus_3_to_the_m_l184_184700


namespace function_behavior_on_intervals_l184_184712

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem function_behavior_on_intervals :
  (∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < deriv f x) ∧
  (∀ x : ℝ, Real.exp 1 < x ∧ x < 10 → deriv f x < 0) := sorry

end function_behavior_on_intervals_l184_184712


namespace largest_possible_x_l184_184817

theorem largest_possible_x :
  ∃ x : ℝ, (3*x^2 + 18*x - 84 = x*(x + 10)) ∧ ∀ y : ℝ, (3*y^2 + 18*y - 84 = y*(y + 10)) → y ≤ x :=
by
  sorry

end largest_possible_x_l184_184817


namespace good_horse_catches_up_l184_184368

noncomputable def catch_up_days : ℕ := sorry

theorem good_horse_catches_up (x : ℕ) :
  (∀ (good_horse_speed slow_horse_speed head_start_duration : ℕ),
    good_horse_speed = 200 →
    slow_horse_speed = 120 →
    head_start_duration = 10 →
    200 * x = 120 * x + 120 * 10) →
  catch_up_days = x :=
by
  intro h
  have := h 200 120 10 rfl rfl rfl
  sorry

end good_horse_catches_up_l184_184368


namespace problem_divisible_by_900_l184_184433

theorem problem_divisible_by_900 (X : ℕ) (a b c d : ℕ) 
  (h1 : 1000 <= X)
  (h2 : X < 10000)
  (h3 : X = 1000 * a + 100 * b + 10 * c + d)
  (h4 : d ≠ 0)
  (h5 : (X + (1000 * a + 100 * c + 10 * b + d)) % 900 = 0)
  : X % 90 = 45 := 
sorry

end problem_divisible_by_900_l184_184433


namespace factorable_polynomial_l184_184931

theorem factorable_polynomial (n : ℤ) :
  ∃ (a b c d e f : ℤ), 
    (a = 1) ∧ (d = 1) ∧ 
    (b + e = 2) ∧ 
    (f = b * e) ∧ 
    (c + f + b * e = 2) ∧ 
    (c * f + b * e = -n^2) ↔ 
    (n = 0 ∨ n = 2 ∨ n = -2) :=
by
  sorry

end factorable_polynomial_l184_184931


namespace simplify_expression_l184_184872

theorem simplify_expression (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = - (2 / 3) * x - 2 :=
by sorry

end simplify_expression_l184_184872


namespace arithmetic_sequence_sum_ratio_l184_184334

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℝ) (T : ℕ → ℝ) (a b : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, S n = 3 * k * n^2)
  (h2 : ∀ n, T n = k * n * (2 * n + 1))
  (h3 : ∀ n, a n = S n - S (n - 1))
  (h4 : ∀ n, b n = T n - T (n - 1))
  (h5 : ∀ n, S n / T n = (3 * n) / (2 * n + 1)) :
  (a 1 + a 2 + a 14 + a 19) / (b 1 + b 3 + b 17 + b 19) = 17 / 13 :=
sorry

end arithmetic_sequence_sum_ratio_l184_184334


namespace required_circle_equation_l184_184862

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation on which the center of the required circle lies
def center_line (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- State the final proof that the equation of the required circle is (x + 1)^2 + (y - 1)^2 = 13 under the given conditions
theorem required_circle_equation (x y : ℝ) :
  ( ∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧
    (∃ (cx cy r : ℝ), center_line cx cy ∧ (x - cx)^2 + (y - cy)^2 = r^2 ∧ (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
      (x + 1)^2 + (y - 1)^2 = 13) )
 := sorry

end required_circle_equation_l184_184862


namespace logarithm_identity_l184_184939

noncomputable section

open Real

theorem logarithm_identity : 
  log 10 = (log (sqrt 5) / log 10 + (1 / 2) * log 20) :=
sorry

end logarithm_identity_l184_184939


namespace arithmetic_sequence_S_15_l184_184174

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ}

theorem arithmetic_sequence_S_15 :
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 1 + a 15 = 2 * a 8) →
  (a 4 + a 12 = 2 * a 8) →
  S 15 a = -30 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_S_15_l184_184174


namespace max_distance_from_point_on_circle_to_line_l184_184557

noncomputable def center_of_circle : ℝ × ℝ := (5, 3)
noncomputable def radius_of_circle : ℝ := 3
noncomputable def line_eqn (x y : ℝ) : ℝ := 3 * x + 4 * y - 2
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ := (|a * px + b * py + c|) / (Real.sqrt (a * a + b * b))

theorem max_distance_from_point_on_circle_to_line :
  let Cx := (center_of_circle.1)
  let Cy := (center_of_circle.2)
  let d := distance_point_to_line Cx Cy 3 4 (-2)
  d + radius_of_circle = 8 := by
  sorry

end max_distance_from_point_on_circle_to_line_l184_184557


namespace solution_exists_l184_184121

theorem solution_exists (x : ℝ) :
  (|2 * x - 3| ≤ 3 ∧ (1 / x) < 1 ∧ x ≠ 0) ↔ (1 < x ∧ x ≤ 3) :=
by
  sorry

end solution_exists_l184_184121


namespace find_fraction_l184_184115

theorem find_fraction (x : ℝ) (h1 : 7 = (1 / 10) / 100 * 7000) (h2 : x * 7000 - 7 = 700) : x = 707 / 7000 :=
by sorry

end find_fraction_l184_184115


namespace ball_arrangement_problem_l184_184366

-- Defining the problem statement and conditions
theorem ball_arrangement_problem : 
  (∃ (A : ℕ), 
    (∀ (b : Fin 6 → ℕ), 
      (b 0 = 1 ∨ b 1 = 1) ∧ (b 0 = 2 ∨ b 1 = 2) ∧ -- 1 adjacent to 2
      b 4 ≠ 5 ∧ b 4 ≠ 6 ∧                 -- 5 not adjacent to 6 condition
      b 5 ≠ 5 ∧ b 5 ≠ 6     -- Add all other necessary conditions for arrangement
    ) →
    A = 144)
:= sorry

end ball_arrangement_problem_l184_184366


namespace area_of_BDOE_l184_184645

namespace Geometry

noncomputable def areaQuadrilateralBDOE (AE CD AB BC AC : ℝ) : ℝ :=
  if AE = 2 ∧ CD = 11 ∧ AB = 8 ∧ BC = 8 ∧ AC = 6 then
    189 * Real.sqrt 55 / 88
  else
    0

theorem area_of_BDOE :
  areaQuadrilateralBDOE 2 11 8 8 6 = 189 * Real.sqrt 55 / 88 :=
by 
  sorry

end Geometry

end area_of_BDOE_l184_184645


namespace smallest_positive_number_is_x2_l184_184087

noncomputable def x1 : ℝ := 14 - 4 * Real.sqrt 17
noncomputable def x2 : ℝ := 4 * Real.sqrt 17 - 14
noncomputable def x3 : ℝ := 23 - 7 * Real.sqrt 14
noncomputable def x4 : ℝ := 65 - 12 * Real.sqrt 34
noncomputable def x5 : ℝ := 12 * Real.sqrt 34 - 65

theorem smallest_positive_number_is_x2 :
  x2 = 4 * Real.sqrt 17 - 14 ∧
  (0 < x1 ∨ 0 < x2 ∨ 0 < x3 ∨ 0 < x4 ∨ 0 < x5) ∧
  (∀ x : ℝ, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) → 0 < x → x2 ≤ x) := sorry

end smallest_positive_number_is_x2_l184_184087


namespace solve_for_p_l184_184138

-- Conditions
def C1 (n : ℕ) : Prop := (3 : ℚ) / 4 = n / 48
def C2 (m n : ℕ) : Prop := (3 : ℚ) / 4 = (m + n) / 96
def C3 (p m : ℕ) : Prop := (3 : ℚ) / 4 = (p - m) / 160

-- Theorem to prove
theorem solve_for_p (n m p : ℕ) (h1 : C1 n) (h2 : C2 m n) (h3 : C3 p m) : p = 156 := 
by 
    sorry

end solve_for_p_l184_184138


namespace point_in_quadrant_l184_184560

theorem point_in_quadrant (m n : ℝ) (h₁ : 2 * (m - 1)^2 - 7 = -5) (h₂ : n > 3) :
  (m = 0 → 2*m - 3 < 0 ∧ (3*n - m)/2 > 0) ∧ 
  (m = 2 → 2*m - 3 > 0 ∧ (3*n - m)/2 > 0) :=
by 
  sorry

end point_in_quadrant_l184_184560


namespace dodecahedron_edge_probability_l184_184389

def numVertices := 20
def pairsChosen := Nat.choose 20 2  -- Calculates combination (20 choose 2)
def edgesPerVertex := 3
def numEdges := (numVertices * edgesPerVertex) / 2
def probability : ℚ := numEdges / pairsChosen

theorem dodecahedron_edge_probability :
  probability = 3 / 19 :=
by
  -- The proof is skipped as per the instructions
  sorry

end dodecahedron_edge_probability_l184_184389


namespace triangle_inequality_l184_184364

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := 
sorry

end triangle_inequality_l184_184364


namespace hyperbola_equation_l184_184544

theorem hyperbola_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 - 2 → (∃ k : ℝ, k ≠ 0 ∧ x * y = k) := 
by
  intros h
  sorry

end hyperbola_equation_l184_184544


namespace quadrilateral_pyramid_plane_intersection_l184_184834

-- Definitions:
-- Let MA, MB, MC, MD, MK, ML, MP, MN be lengths of respective segments
-- Let S_ABC, S_ABD, S_ACD, S_BCD be areas of respective triangles
variables {MA MB MC MD MK ML MP MN : ℝ}
variables {S_ABC S_ABD S_ACD S_BCD : ℝ}

-- Given a quadrilateral pyramid MABCD with a convex quadrilateral ABCD as base, and a plane intersecting edges MA, MB, MC, and MD at points K, L, P, and N respectively. Prove the following relation.
theorem quadrilateral_pyramid_plane_intersection :
  S_BCD * (MA / MK) + S_ADB * (MC / MP) = S_ABC * (MD / MN) + S_ACD * (MB / ML) :=
sorry

end quadrilateral_pyramid_plane_intersection_l184_184834


namespace car_length_l184_184770

variables (L E C : ℕ)

theorem car_length (h1 : 150 * E = L + 150 * C) (h2 : 30 * E = L - 30 * C) : L = 113 * E :=
by
  sorry

end car_length_l184_184770


namespace series_sum_eq_l184_184162

noncomputable def sum_series : ℝ :=
∑' n : ℕ, if h : n > 0 then (4 * n + 3) / ((4 * n)^2 * (4 * n + 4)^2) else 0

theorem series_sum_eq :
  sum_series = 1 / 256 := by
  sorry

end series_sum_eq_l184_184162


namespace parabola_translation_l184_184247

theorem parabola_translation :
  ∀ x y, (y = -2 * x^2) →
    ∃ x' y', y' = -2 * (x' - 2)^2 + 1 ∧ x' = x ∧ y' = y + 1 :=
sorry

end parabola_translation_l184_184247


namespace complex_root_sixth_power_sum_equals_38908_l184_184584

noncomputable def omega : ℂ :=
  -- By definition, omega should satisfy the below properties.
  -- The exact value of omega is not being defined, we will use algebraic properties in the proof.
  sorry

theorem complex_root_sixth_power_sum_equals_38908 : 
  ∀ (ω : ℂ), ω^3 = 1 ∧ ¬(ω.re = 1) → (2 - ω + 2 * ω^2)^6 + (2 + ω - 2 * ω^2)^6 = 38908 :=
by
  -- Proof will utilize given conditions:
  -- 1. ω^3 = 1
  -- 2. ω is not real (or ω.re is not 1)
  sorry

end complex_root_sixth_power_sum_equals_38908_l184_184584


namespace sum_of_solutions_l184_184701

theorem sum_of_solutions (x : ℝ) (hx : x + 36 / x = 12) : x = 6 ∨ x = -6 := sorry

end sum_of_solutions_l184_184701


namespace average_is_3_l184_184141

theorem average_is_3 (A B C : ℝ) (h1 : 1501 * C - 3003 * A = 6006)
                              (h2 : 1501 * B + 4504 * A = 7507)
                              (h3 : A + B = 1) :
  (A + B + C) / 3 = 3 :=
by sorry

end average_is_3_l184_184141


namespace max_distance_is_15_l184_184214

noncomputable def max_distance_between_cars (v_A v_B: ℝ) (a: ℝ) (D: ℝ) : ℝ :=
  if v_A > v_B ∧ D = a + 60 then (a * (1 - a / 60)) else 0

theorem max_distance_is_15 (v_A v_B: ℝ) (a: ℝ) (D: ℝ) :
  v_A > v_B ∧ D = a + 60 → max_distance_between_cars v_A v_B a D = 15 :=
by
  sorry

end max_distance_is_15_l184_184214


namespace calc_result_l184_184158

theorem calc_result (initial_number : ℕ) (square : ℕ → ℕ) (subtract_five : ℕ → ℕ) : 
  initial_number = 7 ∧ (square 7 = 49) ∧ (subtract_five 49 = 44) → 
  subtract_five (square initial_number) = 44 := 
by
  sorry

end calc_result_l184_184158


namespace Jessie_initial_weight_l184_184511

def lost_first_week : ℕ := 56
def after_first_week : ℕ := 36

theorem Jessie_initial_weight :
  (after_first_week + lost_first_week = 92) :=
by
  sorry

end Jessie_initial_weight_l184_184511


namespace small_circles_sixth_figure_l184_184841

-- Defining the function to calculate the number of circles in the nth figure
def small_circles (n : ℕ) : ℕ :=
  n * (n + 1) + 4

-- Statement of the theorem
theorem small_circles_sixth_figure :
  small_circles 6 = 46 :=
by sorry

end small_circles_sixth_figure_l184_184841


namespace find_number_l184_184040

theorem find_number (x : ℝ) (h : x * 9999 = 824777405) : x = 82482.5 :=
by
  sorry

end find_number_l184_184040


namespace number_of_girls_l184_184398

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end number_of_girls_l184_184398


namespace only_one_correct_guess_l184_184806

-- Define the contestants
inductive Contestant : Type
| person : ℕ → Contestant

def A_win_first (c: Contestant) : Prop :=
c = Contestant.person 4 ∨ c = Contestant.person 5

def B_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 3 

def C_win_first (c: Contestant) : Prop :=
c = Contestant.person 1 ∨ c = Contestant.person 2 ∨ c = Contestant.person 6

def D_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 4 ∧ c ≠ Contestant.person 5 ∧ c ≠ Contestant.person 6

-- The main theorem: Only one correct guess among A, B, C, and D
theorem only_one_correct_guess (win: Contestant) :
  (A_win_first win ↔ false) ∧ (B_not_win_first win ↔ false) ∧ (C_win_first win ↔ false) ∧ D_not_win_first win
:=
by
  sorry

end only_one_correct_guess_l184_184806


namespace alice_has_winning_strategy_l184_184976

def alice_has_winning_strategy_condition (nums : List ℤ) : Prop :=
  nums.length = 17 ∧ ∀ x ∈ nums, ¬ (x % 17 = 0)

theorem alice_has_winning_strategy (nums : List ℤ) (H : alice_has_winning_strategy_condition nums) : ∃ (f : List ℤ → List ℤ), ∀ k, (f^[k] nums).sum % 17 = 0 :=
sorry

end alice_has_winning_strategy_l184_184976


namespace x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l184_184144

theorem x_plus_inv_x_eq_two_implies_x_pow_six_eq_one
  (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 :=
sorry

end x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l184_184144


namespace true_universal_quantifier_l184_184832

theorem true_universal_quantifier :
  ∀ (a b : ℝ), a^2 + b^2 ≥ 2 * (a - b - 1) := by
  sorry

end true_universal_quantifier_l184_184832


namespace time_per_potato_l184_184593

-- Definitions from the conditions
def total_potatoes : ℕ := 12
def cooked_potatoes : ℕ := 6
def remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
def total_time : ℕ := 36
def remaining_time_per_potato : ℕ := total_time / remaining_potatoes

-- Theorem to be proved
theorem time_per_potato : remaining_time_per_potato = 6 := by
  sorry

end time_per_potato_l184_184593


namespace no_valid_rectangles_l184_184577

theorem no_valid_rectangles 
  (a b x y : ℝ) (h_ab_lt : a < b) (h_xa_lt : x < a) (h_ya_lt : y < a) 
  (h_perimeter : 2 * (x + y) = (2 * (a + b)) / 3) 
  (h_area : x * y = (a * b) / 3) : false := 
sorry

end no_valid_rectangles_l184_184577


namespace price_per_foot_of_fence_l184_184428

theorem price_per_foot_of_fence (area : ℝ) (total_cost : ℝ) (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) 
  (h1 : area = 289) (h2 : total_cost = 3672) (h3 : side_length = Real.sqrt area) (h4 : perimeter = 4 * side_length) (h5 : price_per_foot = total_cost / perimeter) :
  price_per_foot = 54 := by
  sorry

end price_per_foot_of_fence_l184_184428


namespace fraction_inequality_solution_l184_184331

theorem fraction_inequality_solution (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ (4 * x + 3 > 2 * (8 - 3 * x)) → (13 / 10) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l184_184331


namespace convert_speed_to_mps_l184_184128

-- Define given speeds and conversion factors
def speed_kmph : ℝ := 63
def kilometers_to_meters : ℝ := 1000
def hours_to_seconds : ℝ := 3600

-- Assert the conversion
theorem convert_speed_to_mps : speed_kmph * (kilometers_to_meters / hours_to_seconds) = 17.5 := by
  sorry

end convert_speed_to_mps_l184_184128


namespace matrix_power_four_l184_184317

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l184_184317


namespace album_count_l184_184151

def albums_total (A B K M C : ℕ) : Prop :=
  A = 30 ∧ B = A - 15 ∧ K = 6 * B ∧ M = 5 * K ∧ C = 3 * M ∧ (A + B + K + M + C) = 1935

theorem album_count (A B K M C : ℕ) : albums_total A B K M C :=
by
  sorry

end album_count_l184_184151


namespace mindy_emails_l184_184859

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end mindy_emails_l184_184859


namespace problem_condition_l184_184757

variable {f : ℝ → ℝ}
variable {a b : ℝ}

noncomputable def fx_condition (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x + x * (deriv f x) < 0

theorem problem_condition {f : ℝ → ℝ} {a b : ℝ} (h1 : fx_condition f) (h2 : a < b) :
  a * f a > b * f b :=
sorry

end problem_condition_l184_184757


namespace recipe_flour_cups_l184_184359

theorem recipe_flour_cups (F : ℕ) : 
  (exists (sugar : ℕ) (flourAdded : ℕ) (sugarExtra : ℕ), sugar = 11 ∧ flourAdded = 4 ∧ sugarExtra = 6 ∧ ((F - flourAdded) + sugarExtra = sugar)) →
  F = 9 :=
sorry

end recipe_flour_cups_l184_184359


namespace exercise_serial_matches_year_problem_serial_matches_year_l184_184047

-- Definitions for the exercise
def exercise_initial := 1169
def exercises_per_issue := 8
def issues_per_year := 9
def exercise_year := 1979
def exercises_per_year := exercises_per_issue * issues_per_year

-- Definitions for the problem
def problem_initial := 1576
def problems_per_issue := 8
def problems_per_year := problems_per_issue * issues_per_year
def problem_year := 1973

theorem exercise_serial_matches_year :
  ∃ (issue_number : ℕ) (exercise_number : ℕ),
    (issue_number = 3) ∧
    (exercise_number = 2) ∧
    (exercise_initial + 11 * exercises_per_year + 16 = exercise_year) :=
by {
  sorry
}

theorem problem_serial_matches_year :
  ∃ (issue_number : ℕ) (problem_number : ℕ),
    (issue_number = 5) ∧
    (problem_number = 5) ∧
    (problem_initial + 5 * problems_per_year + 36 = problem_year) :=
by {
  sorry
}

end exercise_serial_matches_year_problem_serial_matches_year_l184_184047


namespace roots_of_equation_in_interval_l184_184025

theorem roots_of_equation_in_interval (f : ℝ → ℝ) (interval : Set ℝ) (n_roots : ℕ) :
  (∀ x ∈ interval, f x = 8 * x * (1 - 2 * x^2) * (8 * x^4 - 8 * x^2 + 1) - 1) →
  (interval = Set.Icc 0 1) →
  (n_roots = 4) :=
by
  intros f_eq interval_eq
  sorry

end roots_of_equation_in_interval_l184_184025


namespace activities_equally_popular_l184_184250

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

end activities_equally_popular_l184_184250


namespace convert_decimal_to_fraction_l184_184929

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l184_184929


namespace find_number_l184_184280

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end find_number_l184_184280


namespace parallel_line_slope_l184_184730

theorem parallel_line_slope (x y : ℝ) : 
  (∃ b : ℝ, y = (1 / 2) * x + b) → 
  (∃ a : ℝ, 3 * x - 6 * y = a) → 
  ∃ k : ℝ, k = 1 / 2 :=
by
  intros h1 h2
  sorry

end parallel_line_slope_l184_184730


namespace adam_bought_dog_food_packages_l184_184710

-- Define the constants and conditions
def num_cat_food_packages : ℕ := 9
def cans_per_cat_food_package : ℕ := 10
def cans_per_dog_food_package : ℕ := 5
def additional_cat_food_cans : ℕ := 55

-- Define the variable for dog food packages and our equation
def num_dog_food_packages (d : ℕ) : Prop :=
  (num_cat_food_packages * cans_per_cat_food_package) = (d * cans_per_dog_food_package + additional_cat_food_cans)

-- The theorem statement representing the proof problem
theorem adam_bought_dog_food_packages : ∃ d : ℕ, num_dog_food_packages d ∧ d = 7 :=
sorry

end adam_bought_dog_food_packages_l184_184710


namespace cristine_lemons_left_l184_184188

theorem cristine_lemons_left (initial_lemons : ℕ) (given_fraction : ℚ) (exchanged_lemons : ℕ) (h1 : initial_lemons = 12) (h2 : given_fraction = 1/4) (h3 : exchanged_lemons = 2) : 
  initial_lemons - initial_lemons * given_fraction - exchanged_lemons = 7 :=
by 
  sorry

end cristine_lemons_left_l184_184188


namespace part_I_part_II_l184_184437

-- Definitions of the sets A, B, and C
def A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 3 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m }

-- Proof statements
theorem part_I : A ∩ B = { x | 3 ≤ x ∧ x ≤ 6 } :=
by sorry

theorem part_II (m : ℝ) : (B ∪ C m = B) → (m ≤ 3) :=
by sorry

end part_I_part_II_l184_184437


namespace Joan_pays_139_20_l184_184223

noncomputable def JKL : Type := ℝ × ℝ × ℝ

def conditions (J K L : ℝ) : Prop :=
  J + K + L = 600 ∧
  2 * J = K + 74 ∧
  L = K + 52

theorem Joan_pays_139_20 (J K L : ℝ) (h : conditions J K L) : J = 139.20 :=
by
  sorry

end Joan_pays_139_20_l184_184223


namespace sum_even_odd_functions_l184_184376

theorem sum_even_odd_functions (f g : ℝ → ℝ) (h₁ : ∀ x, f (-x) = f x) (h₂ : ∀ x, g (-x) = -g x) (h₃ : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := 
by 
  sorry

end sum_even_odd_functions_l184_184376


namespace problem1_problem2_l184_184797

-- Definition of f(x)
def f (x : ℝ) : ℝ := abs (x - 1)

-- Definition of g(x)
def g (x t : ℝ) : ℝ := t * abs x - 2

-- Problem 1: Proof that f(x) > 2x + 1 implies x < 0
theorem problem1 (x : ℝ) : f x > 2 * x + 1 → x < 0 := by
  sorry

-- Problem 2: Proof that if f(x) ≥ g(x) for all x, then t ≤ 1
theorem problem2 (t : ℝ) : (∀ x : ℝ, f x ≥ g x t) → t ≤ 1 := by
  sorry

end problem1_problem2_l184_184797


namespace can_transfer_increase_average_l184_184442

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l184_184442


namespace cost_per_day_is_18_l184_184415

def cost_per_day_first_week (x : ℕ) : Prop :=
  let cost_per_day_rest_week := 12
  let total_days := 23
  let total_cost := 318
  let first_week_days := 7
  let remaining_days := total_days - first_week_days
  (first_week_days * x) + (remaining_days * cost_per_day_rest_week) = total_cost

theorem cost_per_day_is_18 : cost_per_day_first_week 18 :=
  sorry

end cost_per_day_is_18_l184_184415


namespace find_value_of_a_l184_184472

theorem find_value_of_a (a : ℚ) (h : a + a / 4 - 1 / 2 = 2) : a = 2 :=
by
  sorry

end find_value_of_a_l184_184472


namespace solve_quadratic_eq_l184_184687

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 * x = 2 ↔ (x = 2 + Real.sqrt 6) ∨ (x = 2 - Real.sqrt 6) :=
by
  sorry

end solve_quadratic_eq_l184_184687


namespace boat_speed_in_still_water_l184_184049

variable (B S : ℝ)

-- conditions
def condition1 : Prop := B + S = 6
def condition2 : Prop := B - S = 2

-- question to answer
theorem boat_speed_in_still_water (h1 : condition1 B S) (h2 : condition2 B S) : B = 4 :=
by
  sorry

end boat_speed_in_still_water_l184_184049


namespace cubic_difference_l184_184725

theorem cubic_difference (x : ℝ) (h : (x + 16) ^ (1/3) - (x - 16) ^ (1/3) = 4) : 
  235 < x^2 ∧ x^2 < 240 := 
sorry

end cubic_difference_l184_184725


namespace Jill_arrives_9_minutes_later_l184_184831

theorem Jill_arrives_9_minutes_later
  (distance : ℝ)
  (Jack_speed : ℝ)
  (Jill_speed : ℝ)
  (h1 : distance = 1)
  (h2 : Jack_speed = 10)
  (h3 : Jill_speed = 4) :
  ((distance / Jill_speed) - (distance / Jack_speed)) * 60 = 9 := by
  -- Placeholder for the proof
  sorry

end Jill_arrives_9_minutes_later_l184_184831


namespace normal_price_of_article_l184_184130

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20) 
  (h3 : sale_price = 72) 
  (h4 : sale_price = (P * (1 - discount1)) * (1 - discount2)) : 
  P = 100 :=
by 
  sorry

end normal_price_of_article_l184_184130


namespace probability_at_tree_correct_expected_distance_correct_l184_184152

-- Define the initial conditions
def initial_tree (n : ℕ) : ℕ := n + 1
def total_trees (n : ℕ) : ℕ := 2 * n + 1

-- Define the probability that the drunkard is at each tree T_i (1 <= i <= 2n+1) at the end of the nth minute
def probability_at_tree (n i : ℕ) : ℚ :=
  if 1 ≤ i ∧ i ≤ total_trees n then
    (Nat.choose (2*n) (i-1)) / (2^(2*n))
  else
    0

-- Define the expected distance between the final position and the initial tree T_{n+1}
def expected_distance (n : ℕ) : ℚ :=
  n * (Nat.choose (2*n) n) / (2^(2*n))

-- Statements to prove
theorem probability_at_tree_correct (n i : ℕ) (hi : 1 ≤ i ∧ i ≤ total_trees n)  :
  probability_at_tree n i = (Nat.choose (2*n) (i-1)) / (2^(2*n)) :=
by
  sorry

theorem expected_distance_correct (n : ℕ) :
  expected_distance n = n * (Nat.choose (2*n) n) / (2^(2*n)) :=
by
  sorry

end probability_at_tree_correct_expected_distance_correct_l184_184152


namespace dart_not_land_in_circle_probability_l184_184216

theorem dart_not_land_in_circle_probability :
  let side_length := 1
  let radius := side_length / 2
  let area_square := side_length * side_length
  let area_circle := π * radius * radius
  let prob_inside_circle := area_circle / area_square
  let prob_outside_circle := 1 - prob_inside_circle
  prob_outside_circle = 1 - (π / 4) :=
by
  sorry

end dart_not_land_in_circle_probability_l184_184216


namespace final_passenger_count_l184_184324

def total_passengers (initial : ℕ) (first_stop : ℕ) (off_bus : ℕ) (on_bus : ℕ) : ℕ :=
  (initial + first_stop) - off_bus + on_bus

theorem final_passenger_count :
  total_passengers 50 16 22 5 = 49 := by
  sorry

end final_passenger_count_l184_184324


namespace percentage_discount_l184_184432

theorem percentage_discount (individual_payment_without_discount final_payment discount_per_person : ℝ)
  (h1 : 3 * individual_payment_without_discount = final_payment + 3 * discount_per_person)
  (h2 : discount_per_person = 4)
  (h3 : final_payment = 48) :
  discount_per_person / (individual_payment_without_discount * 3) * 100 = 20 :=
by
  -- Proof to be provided here
  sorry

end percentage_discount_l184_184432


namespace digits_C_not_make_1C34_divisible_by_4_l184_184564

theorem digits_C_not_make_1C34_divisible_by_4 :
  ∀ (C : ℕ), (C ≥ 0) ∧ (C ≤ 9) → ¬ (1034 + 100 * C) % 4 = 0 :=
by sorry

end digits_C_not_make_1C34_divisible_by_4_l184_184564


namespace range_of_m_l184_184345

theorem range_of_m (y : ℝ) (x : ℝ) (xy_ne_zero : x * y ≠ 0) :
  (x^2 + 4 * y^2 = (m^2 + 3 * m) * x * y) → -4 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l184_184345


namespace find_n_tan_eq_l184_184259

theorem find_n_tan_eq (n : ℤ) (h₁ : -180 < n) (h₂ : n < 180) 
  (h₃ : Real.tan (n * (Real.pi / 180)) = Real.tan (276 * (Real.pi / 180))) : 
  n = 96 :=
sorry

end find_n_tan_eq_l184_184259


namespace solve_system_of_equations_l184_184860

def proof_problem (a b c : ℚ) : Prop :=
  ((a - b = 2) ∧ (c = -5) ∧ (2 * a - 6 * b = 2)) → 
  (a = 5 / 2 ∧ b = 1 / 2 ∧ c = -5)

theorem solve_system_of_equations (a b c : ℚ) :
  proof_problem a b c :=
  by
    sorry

end solve_system_of_equations_l184_184860


namespace cat_catches_total_birds_l184_184663

theorem cat_catches_total_birds :
  let morning_birds := 15
  let morning_success_rate := 0.60
  let afternoon_birds := 25
  let afternoon_success_rate := 0.80
  let night_birds := 20
  let night_success_rate := 0.90
  
  let morning_caught := morning_birds * morning_success_rate
  let afternoon_initial_caught := 2 * morning_caught
  let afternoon_caught := min (afternoon_birds * afternoon_success_rate) afternoon_initial_caught
  let night_caught := night_birds * night_success_rate

  let total_caught := morning_caught + afternoon_caught + night_caught
  total_caught = 47 := 
by
  sorry

end cat_catches_total_birds_l184_184663


namespace integral_evaluation_l184_184119

noncomputable def integral_value : Real :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - (x - 1)^2) - x)

theorem integral_evaluation :
  integral_value = (Real.pi / 4) - 1 / 2 :=
by
  sorry

end integral_evaluation_l184_184119


namespace find_other_x_intercept_l184_184077

theorem find_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 2 → y = -3) (h_x_intercept : ∀ x, x = 5 → y = 0) : 
  ∃ x, x = -1 ∧ y = 0 := 
sorry

end find_other_x_intercept_l184_184077


namespace johns_profit_l184_184286

noncomputable def profit_made 
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ) : ℕ :=
(trees_chopped * planks_per_tree / planks_per_table) * price_per_table - labor_cost

theorem johns_profit : profit_made 30 25 15 300 3000 = 12000 :=
by sorry

end johns_profit_l184_184286


namespace sledding_small_hills_l184_184567

theorem sledding_small_hills (total_sleds tall_hills_sleds sleds_per_tall_hill sleds_per_small_hill small_hills : ℕ) 
  (h1 : total_sleds = 14)
  (h2 : tall_hills_sleds = 2)
  (h3 : sleds_per_tall_hill = 4)
  (h4 : sleds_per_small_hill = sleds_per_tall_hill / 2)
  (h5 : total_sleds = tall_hills_sleds * sleds_per_tall_hill + small_hills * sleds_per_small_hill)
  : small_hills = 3 := 
sorry

end sledding_small_hills_l184_184567


namespace exists_distinct_positive_integers_l184_184746

theorem exists_distinct_positive_integers (n : ℕ) (h : 0 < n) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end exists_distinct_positive_integers_l184_184746


namespace wage_recovery_l184_184200

theorem wage_recovery (W : ℝ) (h : W > 0) : (1 - 0.3) * W * (1 + 42.86 / 100) = W :=
by
  sorry

end wage_recovery_l184_184200


namespace journey_divided_into_portions_l184_184052

theorem journey_divided_into_portions
  (total_distance : ℕ)
  (speed : ℕ)
  (time : ℝ)
  (portion_distance : ℕ)
  (portions_covered : ℕ)
  (h1 : total_distance = 35)
  (h2 : speed = 40)
  (h3 : time = 0.7)
  (h4 : portions_covered = 4)
  (distance_covered := speed * time)
  (one_portion_distance := distance_covered / portions_covered)
  (total_portions := total_distance / one_portion_distance) :
  total_portions = 5 := 
sorry

end journey_divided_into_portions_l184_184052


namespace scientific_notation_of_2102000_l184_184217

theorem scientific_notation_of_2102000 : ∃ (x : ℝ) (n : ℤ), 2102000 = x * 10 ^ n ∧ x = 2.102 ∧ n = 6 :=
by
  sorry

end scientific_notation_of_2102000_l184_184217


namespace student_papers_count_l184_184704

theorem student_papers_count {F n k: ℝ}
  (h1 : 35 * k = 0.6 * n * F)
  (h2 : 5 * k > 0.5 * F)
  (h3 : 6 * k > 0.5 * F)
  (h4 : 7 * k > 0.5 * F)
  (h5 : 8 * k > 0.5 * F)
  (h6 : 9 * k > 0.5 * F) :
  n = 5 :=
by
  sorry

end student_papers_count_l184_184704


namespace soda_count_l184_184769

theorem soda_count
  (W : ℕ) (S : ℕ) (B : ℕ) (T : ℕ)
  (hW : W = 26) (hB : B = 17) (hT : T = 31) :
  W + S - B = T → S = 22 :=
by
  sorry

end soda_count_l184_184769


namespace min_value_of_expr_l184_184614

noncomputable def min_value (x y : ℝ) : ℝ :=
  (4 * x^2) / (y + 1) + (y^2) / (2*x + 2)

theorem min_value_of_expr : 
  ∀ (x y : ℝ), (0 < x) → (0 < y) → (2 * x + y = 2) →
  min_value x y = 4 / 5 :=
by
  intros x y hx hy hxy
  sorry

end min_value_of_expr_l184_184614


namespace n_minus_m_eq_singleton_6_l184_184804

def set_difference (A B : Set α) : Set α :=
  {x | x ∈ A ∧ x ∉ B}

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem n_minus_m_eq_singleton_6 : set_difference N M = {6} :=
by
  sorry

end n_minus_m_eq_singleton_6_l184_184804


namespace quadratic_form_h_l184_184692

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l184_184692


namespace inequality_least_one_l184_184629

theorem inequality_least_one {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4) :=
by
  sorry

end inequality_least_one_l184_184629


namespace ab_max_min_sum_l184_184632

-- Define the conditions
variables {a b : ℝ}
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 4 * b = 4

-- Problem (1)
theorem ab_max : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → a * b ≤ 1 :=
by sorry

-- Problem (2)
theorem min_sum : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → (1 / a) + (4 / b) ≥ 25 / 4 :=
by sorry

end ab_max_min_sum_l184_184632


namespace three_digit_diff_no_repeated_digits_l184_184848

theorem three_digit_diff_no_repeated_digits :
  let largest := 987
  let smallest := 102
  largest - smallest = 885 := by
  sorry

end three_digit_diff_no_repeated_digits_l184_184848


namespace average_of_middle_three_l184_184621

-- Define the conditions based on the problem statement
def isPositiveWhole (n: ℕ) := n > 0
def areDifferent (a b c d e: ℕ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def isMaximumDifference (a b c d e: ℕ) := max a (max b (max c (max d e))) - min a (min b (min c (min d e)))
def isSecondSmallest (a b c d e: ℕ) := b = 3 ∧ (a < b ∧ (c < b ∨ d < b ∨ e < b) ∧ areDifferent a b c d e)
def totalSumIs30 (a b c d e: ℕ) := a + b + c + d + e = 30

-- Average of the middle three numbers calculated
theorem average_of_middle_three {a b c d e: ℕ} (cond1: isPositiveWhole a)
  (cond2: isPositiveWhole b) (cond3: isPositiveWhole c) (cond4: isPositiveWhole d)
  (cond5: isPositiveWhole e) (cond6: areDifferent a b c d e) (cond7: b = 3)
  (cond8: max a (max c (max d e)) - min a (min c (min d e)) = 16)
  (cond9: totalSumIs30 a b c d e) : (a + c + d) / 3 = 4 :=
by sorry

end average_of_middle_three_l184_184621


namespace equivalent_expression_l184_184341

def evaluate_expression : ℚ :=
  let part1 := (2/3) * ((35/100) * 250)
  let part2 := ((75/100) * 150) / 16
  let part3 := (1/2) * ((40/100) * 500)
  part1 - part2 + part3

theorem equivalent_expression :
  evaluate_expression = 151.3020833333 :=  
by 
  sorry

end equivalent_expression_l184_184341


namespace mowing_time_approximately_correct_l184_184423

noncomputable def timeToMowLawn 
  (length width : ℝ) -- dimensions of the lawn in feet
  (swath overlap : ℝ) -- swath width and overlap in inches
  (speed : ℝ) : ℝ :=  -- walking speed in feet per hour
  (length * (width / ((swath - overlap) / 12))) / speed

theorem mowing_time_approximately_correct
  (h_length : ∀ (length : ℝ), length = 100)
  (h_width : ∀ (width : ℝ), width = 120)
  (h_swath : ∀ (swath : ℝ), swath = 30)
  (h_overlap : ∀ (overlap : ℝ), overlap = 6)
  (h_speed : ∀ (speed : ℝ), speed = 4500) :
  abs (timeToMowLawn 100 120 30 6 4500 - 1.33) < 0.01 := -- assert the answer is approximately 1.33 with a tolerance
by
  intros
  have length := h_length 100
  have width := h_width 120
  have swath := h_swath 30
  have overlap := h_overlap 6
  have speed := h_speed 4500
  rw [length, width, swath, overlap, speed]
  simp [timeToMowLawn]
  sorry

end mowing_time_approximately_correct_l184_184423


namespace compare_fractions_l184_184336

theorem compare_fractions : -(2 / 3 : ℚ) < -(3 / 5 : ℚ) :=
by sorry

end compare_fractions_l184_184336


namespace sin_cos_75_eq_quarter_l184_184411

theorem sin_cos_75_eq_quarter : (Real.sin (75 * Real.pi / 180)) * (Real.cos (75 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_75_eq_quarter_l184_184411


namespace area_union_of_reflected_triangles_l184_184956

def point : Type := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def reflect_y_eq_1 (P : point) : point := (P.1, 2 * 1 - P.2)

def area_of_union (A B C : point) (f : point → point) : ℝ :=
  let A' := f A
  let B' := f B
  let C' := f C
  triangle_area A B C + triangle_area A' B' C'

theorem area_union_of_reflected_triangles :
  area_of_union (3, 4) (5, -2) (6, 2) reflect_y_eq_1 = 11 :=
  sorry

end area_union_of_reflected_triangles_l184_184956


namespace a_n_value_l184_184892

theorem a_n_value (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 3) (h2 : ∀ n, S (n + 1) = 2 * S n) (h3 : S 1 = a 1)
  (h4 : ∀ n, S n = 3 * 2^(n - 1)) : a 4 = 12 :=
sorry

end a_n_value_l184_184892


namespace area_of_inscribed_square_l184_184320

theorem area_of_inscribed_square :
  let parabola := λ x => x^2 - 10 * x + 21
  ∃ (t : ℝ), parabola (5 + t) = -2 * t ∧ (2 * t)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end area_of_inscribed_square_l184_184320


namespace total_amount_l184_184796

theorem total_amount (N50 N: ℕ) (h1: N = 90) (h2: N50 = 77) : 
  (N50 * 50 + (N - N50) * 500) = 10350 :=
by
  sorry

end total_amount_l184_184796


namespace royal_family_children_l184_184935

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l184_184935


namespace S_12_l184_184668

variable {S : ℕ → ℕ}

-- Given conditions
axiom S_4 : S 4 = 4
axiom S_8 : S 8 = 12

-- Goal: Prove S_12
theorem S_12 : S 12 = 24 :=
by
  sorry

end S_12_l184_184668


namespace find_x_l184_184721

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (4, -6, x)
def dot_product : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ
  | (a1, a2, a3), (b1, b2, b3) => a1 * b1 + a2 * b2 + a3 * b3

theorem find_x (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -26 :=
by 
  sorry

end find_x_l184_184721


namespace lucy_first_round_cookies_l184_184369

theorem lucy_first_round_cookies (x : ℕ) : 
  (x + 27 = 61) → x = 34 :=
by
  intros h
  sorry

end lucy_first_round_cookies_l184_184369


namespace initial_soup_weight_l184_184088

theorem initial_soup_weight (W: ℕ) (h: W / 16 = 5): W = 40 :=
by
  sorry

end initial_soup_weight_l184_184088


namespace molecular_weight_correct_l184_184683

-- Define atomic weights of elements
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_D : ℝ := 2.01

-- Define the number of each type of atom in the compound
def num_Ba : ℕ := 2
def num_O : ℕ := 3
def num_H : ℕ := 4
def num_D : ℕ := 1

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  (num_Ba * atomic_weight_Ba) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_D * atomic_weight_D)

-- Theorem stating the molecular weight is 328.71 g/mol
theorem molecular_weight_correct :
  molecular_weight = 328.71 :=
by
  -- The proof will go here
  sorry

end molecular_weight_correct_l184_184683


namespace combined_sleep_hours_l184_184868

def connor_hours : ℕ := 6
def luke_hours : ℕ := connor_hours + 2
def emma_hours : ℕ := connor_hours - 1
def puppy_hours : ℕ := 2 * luke_hours

theorem combined_sleep_hours :
  connor_hours + luke_hours + emma_hours + puppy_hours = 35 := by
  sorry

end combined_sleep_hours_l184_184868


namespace diff_of_squares_example_l184_184237

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l184_184237


namespace proposition_four_l184_184027

variables (a b c : Type)

noncomputable def perpend_lines (a b : Type) : Prop := sorry
noncomputable def parallel_lines (a b : Type) : Prop := sorry

theorem proposition_four (a b c : Type) 
  (h1 : perpend_lines a b) (h2 : parallel_lines b c) :
  perpend_lines a c :=
sorry

end proposition_four_l184_184027


namespace MF1_dot_MF2_range_proof_l184_184563

noncomputable def MF1_dot_MF2_range : Set ℝ :=
  Set.Icc (24 - 16 * Real.sqrt 3) (24 + 16 * Real.sqrt 3)

theorem MF1_dot_MF2_range_proof :
  ∀ (M : ℝ × ℝ), (Prod.snd M + 4) ^ 2 + (Prod.fst M) ^ 2 = 12 →
    (Prod.fst M) ^ 2 + (Prod.snd M) ^ 2 - 4 ∈ MF1_dot_MF2_range :=
by
  sorry

end MF1_dot_MF2_range_proof_l184_184563


namespace sequence_initial_term_l184_184136

theorem sequence_initial_term (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + n)
  (h2 : a 61 = 2010) : a 1 = 180 :=
by
  sorry

end sequence_initial_term_l184_184136


namespace malfunctioning_clock_fraction_correct_l184_184333

noncomputable def malfunctioning_clock_correct_time_fraction : ℚ := 5 / 8

theorem malfunctioning_clock_fraction_correct :
  malfunctioning_clock_correct_time_fraction = 5 / 8 := 
by
  sorry

end malfunctioning_clock_fraction_correct_l184_184333


namespace scouts_earnings_over_weekend_l184_184974

def base_pay_per_hour : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def hours_worked_saturday : ℝ := 4.0
def customers_served_saturday : ℝ := 5.0
def hours_worked_sunday : ℝ := 5.0
def customers_served_sunday : ℝ := 8.0

def earnings_saturday : ℝ := (hours_worked_saturday * base_pay_per_hour) + (customers_served_saturday * tip_per_customer)
def earnings_sunday : ℝ := (hours_worked_sunday * base_pay_per_hour) + (customers_served_sunday * tip_per_customer)

def total_earnings : ℝ := earnings_saturday + earnings_sunday

theorem scouts_earnings_over_weekend : total_earnings = 155.00 := by
  sorry

end scouts_earnings_over_weekend_l184_184974


namespace correct_range_a_l184_184184

noncomputable def proposition_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def proposition_q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem correct_range_a (a : ℝ) :
  (¬ ∃ x, proposition_p a x → ¬ ∃ x, proposition_q x) →
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end correct_range_a_l184_184184


namespace intersection_points_of_parabolas_l184_184635

open Real

theorem intersection_points_of_parabolas (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ y1 y2 : ℝ, y1 = c ∧ y2 = (-2 * b^2 / (9 * a)) + c ∧ 
    ((y1 = a * (0)^2 + b * (0) + c) ∧ (y2 = a * (-b / (3 * a))^2 + b * (-b / (3 * a)) + c))) :=
by
  sorry

end intersection_points_of_parabolas_l184_184635


namespace eunji_class_total_students_l184_184374

variable (A B : Finset ℕ) (universe_students : Finset ℕ)

axiom students_play_instrument_a : A.card = 24
axiom students_play_instrument_b : B.card = 17
axiom students_play_both_instruments : (A ∩ B).card = 8
axiom no_students_without_instruments : A ∪ B = universe_students

theorem eunji_class_total_students : universe_students.card = 33 := by
  sorry

end eunji_class_total_students_l184_184374


namespace find_line_equation_l184_184595

-- Define the conditions for the x-intercept and inclination angle
def x_intercept (x : ℝ) (line : ℝ → ℝ) : Prop :=
  line x = 0

def inclination_angle (θ : ℝ) (k : ℝ) : Prop :=
  k = Real.tan θ

-- Define the properties of the line we're working with
def line (x : ℝ) : ℝ := -x + 5

theorem find_line_equation :
  x_intercept 5 line ∧ inclination_angle (3 * Real.pi / 4) (-1) → (∀ x, line x = -x + 5) :=
by
  intro h
  sorry

end find_line_equation_l184_184595


namespace necessary_but_not_sufficient_l184_184627

-- Define the propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2 * a * b
def Q (a b : ℝ) : Prop := abs (a + b) < abs a + abs b

-- Define the conditions for P and Q
def condition_for_P (a b : ℝ) : Prop := a ≠ b
def condition_for_Q (a b : ℝ) : Prop := a * b < 0

-- Define the statement
theorem necessary_but_not_sufficient (a b : ℝ) :
  (P a b → Q a b) ∧ ¬ (Q a b → P a b) :=
by
  sorry

end necessary_but_not_sufficient_l184_184627


namespace max_m_n_value_l184_184738

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l184_184738


namespace alice_savings_l184_184801

variable (B : ℝ)

def savings (B : ℝ) : ℝ :=
  let first_month := 10
  let second_month := first_month + 30 + B
  let third_month := first_month + 30 + 30
  first_month + second_month + third_month

theorem alice_savings (B : ℝ) : savings B = 120 + B :=
by
  sorry

end alice_savings_l184_184801


namespace pipe_filling_time_l184_184871

theorem pipe_filling_time 
  (rate_A : ℚ := 1/8) 
  (rate_L : ℚ := 1/24) :
  (1 / (rate_A - rate_L) = 12) :=
by
  sorry

end pipe_filling_time_l184_184871


namespace vowel_soup_sequences_count_l184_184722

theorem vowel_soup_sequences_count :
  let vowels := 5
  let sequence_length := 6
  vowels ^ sequence_length = 15625 :=
by
  sorry

end vowel_soup_sequences_count_l184_184722


namespace trajectory_is_one_branch_of_hyperbola_l184_184537

open Real

-- Condition 1: Given points F1 and F2
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Condition 2: Moving point P such that |PF1| - |PF2| = 4
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  abs (dist P F1) - abs (dist P F2) = 4

-- Prove the trajectory of point P is one branch of a hyperbola
theorem trajectory_is_one_branch_of_hyperbola (P : ℝ × ℝ) (h : satisfies_condition P) : 
  (∃ a b : ℝ, ∀ x y: ℝ, satisfies_condition (x, y) → (((x^2 / a^2) - (y^2 / b^2) = 1) ∨ ((x^2 / a^2) - (y^2 / b^2) = -1))) :=
sorry

end trajectory_is_one_branch_of_hyperbola_l184_184537


namespace unique_measures_of_A_l184_184922

theorem unique_measures_of_A : 
  ∃ n : ℕ, n = 17 ∧ 
    (∀ A B : ℕ, 
      (A > 0) ∧ (B > 0) ∧ (A + B = 180) ∧ (∃ k : ℕ, A = k * B) → 
      ∃! A : ℕ, A > 0 ∧ (A + B = 180)) :=
sorry

end unique_measures_of_A_l184_184922


namespace find_m_squared_plus_n_squared_l184_184149

theorem find_m_squared_plus_n_squared (m n : ℝ) (h1 : (m - n) ^ 2 = 8) (h2 : (m + n) ^ 2 = 2) : m ^ 2 + n ^ 2 = 5 :=
by
  sorry

end find_m_squared_plus_n_squared_l184_184149


namespace math_books_count_l184_184480

theorem math_books_count (M H : ℤ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end math_books_count_l184_184480


namespace lcm_of_3_8_9_12_l184_184914

theorem lcm_of_3_8_9_12 : Nat.lcm (Nat.lcm 3 8) (Nat.lcm 9 12) = 72 :=
by
  sorry

end lcm_of_3_8_9_12_l184_184914


namespace cube_sum_is_integer_l184_184643

theorem cube_sum_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^3 + 1/a^3 = m :=
sorry

end cube_sum_is_integer_l184_184643


namespace maxwell_meets_brad_l184_184079

theorem maxwell_meets_brad :
  ∃ t : ℝ, t = 2 ∧ 
  (∀ distance max_speed brad_speed start_time, 
   distance = 14 ∧ 
   max_speed = 4 ∧ 
   brad_speed = 6 ∧ 
   start_time = 1 → 
   max_speed * (t + start_time) + brad_speed * t = distance) :=
by
  use 1
  sorry

end maxwell_meets_brad_l184_184079


namespace sum_of_first_11_terms_of_arithmetic_seq_l184_184401

noncomputable def arithmetic_sequence_SUM (a d : ℚ) : ℚ :=  
  11 / 2 * (2 * a + 10 * d)

theorem sum_of_first_11_terms_of_arithmetic_seq
  (a d : ℚ)
  (h : a + 2 * d + a + 6 * d = 16) :
  arithmetic_sequence_SUM a d = 88 := 
  sorry

end sum_of_first_11_terms_of_arithmetic_seq_l184_184401


namespace remainder_when_x_plus_3uy_div_y_l184_184821

theorem remainder_when_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (v_lt_y : v < y) :
  ((x + 3 * u * y) % y) = v := 
sorry

end remainder_when_x_plus_3uy_div_y_l184_184821


namespace find_k_l184_184882

theorem find_k {k : ℚ} :
    (∃ x y : ℚ, y = 3 * x + 6 ∧ y = -4 * x - 20 ∧ y = 2 * x + k) →
    k = 16 / 7 := 
  sorry

end find_k_l184_184882


namespace tenth_term_of_sequence_l184_184863

-- Define the first term and the common difference
def a1 : ℤ := 10
def d : ℤ := -2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) : ℤ := a1 + d * (n - 1)

-- State the theorem about the 10th term
theorem tenth_term_of_sequence : a_n 10 = -8 := by
  -- Skip the proof
  sorry

end tenth_term_of_sequence_l184_184863


namespace expression_simplification_l184_184339

theorem expression_simplification (a : ℝ) (h : a ≠ 1) (h_beta : 1 = 1):
  (2^(Real.log (a) / Real.log (Real.sqrt 2)) - 
   3^((Real.log (a^2+1)) / (Real.log 27)) - 
   2 * a) / 
  (7^(4 * (Real.log (a) / Real.log 49)) - 
   5^((0.5 * Real.log (a)) / (Real.log (Real.sqrt 5))) - 1) = a^2 + a + 1 :=
by
  sorry

end expression_simplification_l184_184339


namespace third_term_of_arithmetic_sequence_l184_184719

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end third_term_of_arithmetic_sequence_l184_184719


namespace equal_sum_sequence_even_odd_l184_184048

-- Define the sequence a_n
variable {a : ℕ → ℤ}

-- Define the condition of the equal-sum sequence
def equal_sum_sequence (a : ℕ → ℤ) : Prop := ∀ n, a n + a (n + 1) = a (n + 1) + a (n + 2)

-- Statement to prove the odd terms are equal and the even terms are equal
theorem equal_sum_sequence_even_odd (a : ℕ → ℤ) (h : equal_sum_sequence a) : (∀ n, a (2 * n) = a 0) ∧ (∀ n, a (2 * n + 1) = a 1) :=
by
  sorry

end equal_sum_sequence_even_odd_l184_184048


namespace add_base6_numbers_l184_184522

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end add_base6_numbers_l184_184522


namespace system_soln_l184_184684

theorem system_soln (a1 b1 a2 b2 : ℚ)
  (h1 : a1 * 3 + b1 * 6 = 21)
  (h2 : a2 * 3 + b2 * 6 = 12) :
  (3 = 3 ∧ -3 = -3) ∧ (a1 * (2 * 3 + -3) + b1 * (3 - -3) = 21) ∧ (a2 * (2 * 3 + -3) + b2 * (3 - -3) = 12) :=
by
  sorry

end system_soln_l184_184684


namespace simplified_value_l184_184427

-- Define the operation ∗
def operation (m n p q : ℚ) : ℚ :=
  m * p * (n / q)

-- Prove that the simplified value of 5/4 ∗ 6/2 is 60
theorem simplified_value : operation 5 4 6 2 = 60 :=
by
  sorry

end simplified_value_l184_184427


namespace cos_90_eq_0_l184_184997

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l184_184997


namespace parabola_directrix_l184_184611

theorem parabola_directrix (x y : ℝ) (h : y = 4 * x^2) : y = -1 / 16 :=
sorry

end parabola_directrix_l184_184611


namespace smaller_number_is_three_l184_184729

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end smaller_number_is_three_l184_184729


namespace brownies_on_counter_l184_184655

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end brownies_on_counter_l184_184655


namespace ones_digit_of_prime_p_l184_184950

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l184_184950


namespace points_per_touchdown_l184_184018

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end points_per_touchdown_l184_184018


namespace intersection_M_N_l184_184763

-- Define sets M and N
def M := {x : ℝ | x^2 - 2*x ≤ 0}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- The theorem stating the intersection of M and N equals [0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_M_N_l184_184763


namespace annual_income_of_A_l184_184637

def monthly_income_ratios (A_income B_income : ℝ) : Prop := A_income / B_income = 5 / 2
def B_income_increase (B_income C_income : ℝ) : Prop := B_income = C_income + 0.12 * C_income

theorem annual_income_of_A (A_income B_income C_income : ℝ)
  (h1 : monthly_income_ratios A_income B_income)
  (h2 : B_income_increase B_income C_income)
  (h3 : C_income = 13000) :
  12 * A_income = 436800 :=
by 
  sorry

end annual_income_of_A_l184_184637


namespace find_b_l184_184553

theorem find_b (a b : ℝ) (h₁ : 2 * a + 3 = 5) (h₂ : b - a = 2) : b = 3 :=
by 
  sorry

end find_b_l184_184553


namespace eraser_difference_l184_184786

theorem eraser_difference
  (hanna_erasers rachel_erasers tanya_erasers tanya_red_erasers : ℕ)
  (h1 : hanna_erasers = 2 * rachel_erasers)
  (h2 : rachel_erasers = tanya_red_erasers)
  (h3 : tanya_erasers = 20)
  (h4 : tanya_red_erasers = tanya_erasers / 2)
  (h5 : hanna_erasers = 4) :
  rachel_erasers - (tanya_red_erasers / 2) = 5 :=
sorry

end eraser_difference_l184_184786


namespace yevgeniy_age_2014_l184_184741

theorem yevgeniy_age_2014 (birth_year : ℕ) (h1 : birth_year = 1900 + (birth_year % 100))
  (h2 : 2011 - birth_year = (birth_year / 1000) + ((birth_year % 1000) / 100) + ((birth_year % 100) / 10) + (birth_year % 10)) :
  2014 - birth_year = 23 :=
by
  sorry

end yevgeniy_age_2014_l184_184741


namespace bode_law_planet_9_l184_184989

theorem bode_law_planet_9 :
  ∃ (a b : ℝ),
    (a + b = 0.7) ∧ (a + 2 * b = 1) ∧ 
    (70 < a + b * 2^8) ∧ (a + b * 2^8 < 80) :=
by
  -- Define variables and equations based on given conditions
  let a : ℝ := 0.4
  let b : ℝ := 0.3
  
  have h1 : a + b = 0.7 := by 
    sorry  -- Proof that a + b = 0.7
  
  have h2 : a + 2 * b = 1 := by
    sorry  -- Proof that a + 2 * b = 1
  
  have hnine : 70 < a + b * 2^8 ∧ a + b * 2^8 < 80 := by
    -- Calculate a + b * 2^8 and then check the range
    sorry  -- Proof that 70 < a + b * 2^8 < 80

  exact ⟨a, b, h1, h2, hnine⟩

end bode_law_planet_9_l184_184989


namespace number_of_cards_per_page_l184_184662

variable (packs : ℕ) (cards_per_pack : ℕ) (total_pages : ℕ)

def number_of_cards (packs cards_per_pack : ℕ) : ℕ :=
  packs * cards_per_pack

def cards_per_page (total_cards total_pages : ℕ) : ℕ :=
  total_cards / total_pages

theorem number_of_cards_per_page
  (packs := 60) (cards_per_pack := 7) (total_pages := 42)
  (total_cards := number_of_cards packs cards_per_pack)
    : cards_per_page total_cards total_pages = 10 :=
by {
  sorry
}

end number_of_cards_per_page_l184_184662


namespace seller_loss_l184_184703

/--
Given:
1. The buyer took goods worth 10 rubles (v_goods : Real := 10).
2. The buyer gave 25 rubles (payment : Real := 25).
3. The seller exchanged 25 rubles of genuine currency with the neighbor (exchange : Real := 25).
4. The seller received 25 rubles in counterfeit currency from the neighbor (counterfeit : Real := 25).
5. The seller gave 15 rubles in genuine currency as change (change : Real := 15).
6. The neighbor discovered the counterfeit and the seller returned 25 rubles to the neighbor (returned : Real := 25).

Prove that the net loss incurred by the seller is 30 rubles.
-/
theorem seller_loss :
  let v_goods := 10
  let payment := 25
  let exchange := 25
  let counterfeit := 25
  let change := 15
  let returned := 25
  (exchange + change) - v_goods = 30 :=
by
  sorry

end seller_loss_l184_184703


namespace sum_of_midpoints_l184_184825

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l184_184825


namespace betty_age_l184_184960

variable (C A B : ℝ)

-- conditions
def Carol_five_times_Alice := C = 5 * A
def Alice_twelve_years_younger_than_Carol := A = C - 12
def Carol_twice_as_old_as_Betty := C = 2 * B

-- goal
theorem betty_age (hc1 : Carol_five_times_Alice C A)
                  (hc2 : Alice_twelve_years_younger_than_Carol C A)
                  (hc3 : Carol_twice_as_old_as_Betty C B) : B = 7.5 := 
  by
  sorry

end betty_age_l184_184960


namespace add_fractions_add_fractions_as_mixed_l184_184481

theorem add_fractions : (3 / 4) + (5 / 6) + (4 / 3) = (35 / 12) := sorry

theorem add_fractions_as_mixed : (3 / 4) + (5 / 6) + (4 / 3) = 2 + 11 / 12 := sorry

end add_fractions_add_fractions_as_mixed_l184_184481


namespace digit_B_for_divisibility_l184_184475

theorem digit_B_for_divisibility (B : ℕ) (h : (40000 + 1000 * B + 100 * B + 20 + 6) % 7 = 0) : B = 1 :=
sorry

end digit_B_for_divisibility_l184_184475


namespace find_k_value_l184_184582

theorem find_k_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (2 * x1^2 + k * x1 - 2 * k + 1 = 0) ∧ 
                (2 * x2^2 + k * x2 - 2 * k + 1 = 0) ∧ 
                (x1 ≠ x2)) ∧
  ((x1^2 + x2^2 = 29/4)) ↔ (k = 3) := 
sorry

end find_k_value_l184_184582


namespace sufficient_but_not_necessary_l184_184538

noncomputable def p (m : ℝ) : Prop :=
  -6 ≤ m ∧ m ≤ 6

noncomputable def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 9 ≠ 0

theorem sufficient_but_not_necessary (m : ℝ) :
  (p m → q m) ∧ (q m → ¬ p m) :=
by
  sorry

end sufficient_but_not_necessary_l184_184538


namespace probability_triangle_nonagon_l184_184555

-- Define the total number of ways to choose 3 vertices from 9 vertices
def total_ways_to_choose_triangle : ℕ := Nat.choose 9 3

-- Define the number of favorable outcomes
def favorable_outcomes_one_side : ℕ := 9 * 5
def favorable_outcomes_two_sides : ℕ := 9

def total_favorable_outcomes : ℕ := favorable_outcomes_one_side + favorable_outcomes_two_sides

-- Define the probability as a rational number
def probability_at_least_one_side_nonagon (total: ℕ) (favorable: ℕ) : ℚ :=
  favorable / total
  
-- Theorem stating the probability
theorem probability_triangle_nonagon :
  probability_at_least_one_side_nonagon total_ways_to_choose_triangle total_favorable_outcomes = 9 / 14 :=
by
  sorry

end probability_triangle_nonagon_l184_184555


namespace determine_digits_in_base_l184_184678

theorem determine_digits_in_base (x y z b : ℕ) (h1 : 1993 = x * b^2 + y * b + z) (h2 : x + y + z = 22) :
  x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 :=
sorry

end determine_digits_in_base_l184_184678


namespace fraction_equality_l184_184187

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_equality :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := 
by
  sorry

end fraction_equality_l184_184187


namespace min_value_3_div_a_add_2_div_b_l184_184386

/-- Given positive real numbers a and b, and the condition that the lines
(a + 1)x + 2y - 1 = 0 and 3x + (b - 2)y + 2 = 0 are perpendicular,
prove that the minimum value of 3/a + 2/b is 25, given the condition 3a + 2b = 1. -/
theorem min_value_3_div_a_add_2_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : 3 * a + 2 * b = 1) : 3 / a + 2 / b ≥ 25 :=
sorry

end min_value_3_div_a_add_2_div_b_l184_184386


namespace sasha_remainder_l184_184894

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l184_184894


namespace inequality_proof_l184_184787

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end inequality_proof_l184_184787


namespace arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l184_184394

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  sorry

theorem arccos_sqrt_three_over_two_eq_pi_six : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l184_184394


namespace minimum_value_l184_184377

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y = 1) :
  ∀ (z : ℝ), z = (1/x + 1/y) → z ≥ 3 + 2*Real.sqrt 2 :=
by
  sorry

end minimum_value_l184_184377


namespace focus_of_parabola_l184_184617

theorem focus_of_parabola :
  (∀ y : ℝ, x = (1 / 4) * y^2) → (focus = (-1, 0)) := by
  sorry

end focus_of_parabola_l184_184617


namespace solutions_equiv_cond_l184_184838

theorem solutions_equiv_cond (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x + 1 / (x - 1) = a + 1 / (x - 1)) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x = a) ∧ (∃ x : ℝ, x = 1 → a ≠ 4)  :=
sorry

end solutions_equiv_cond_l184_184838


namespace max_cities_l184_184680

theorem max_cities (n : ℕ) (h1 : ∀ (c : Fin n), ∃ (neighbors : Finset (Fin n)), neighbors.card ≤ 3 ∧ c ∈ neighbors) (h2 : ∀ (c1 c2 : Fin n), c1 ≠ c2 → ∃ c : Fin n, c1 ≠ c ∧ c2 ≠ c) : n ≤ 10 := 
sorry

end max_cities_l184_184680


namespace probability_allison_greater_l184_184070

theorem probability_allison_greater (A D S : ℕ) (prob_derek_less_than_4 : ℚ) (prob_sophie_less_than_4 : ℚ) : 
  (A > D) ∧ (A > S) → prob_derek_less_than_4 = 1 / 2 ∧ prob_sophie_less_than_4 = 2 / 3 → 
  (1 / 2 : ℚ) * (2 / 3 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_allison_greater_l184_184070


namespace sin_double_angle_l184_184400

-- Given Conditions
variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2) -- α is in the first quadrant
variable (h2 : Real.sin α = 3 / 5) -- sin(α) = 3/5

-- Theorem statement
theorem sin_double_angle (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := 
sorry

end sin_double_angle_l184_184400


namespace probability_three_dice_same_number_is_1_div_36_l184_184110

noncomputable def probability_same_number_three_dice : ℚ :=
  let first_die := 1
  let second_die := 1 / 6
  let third_die := 1 / 6
  first_die * second_die * third_die

theorem probability_three_dice_same_number_is_1_div_36 : probability_same_number_three_dice = 1 / 36 :=
  sorry

end probability_three_dice_same_number_is_1_div_36_l184_184110


namespace exists_points_same_color_one_meter_apart_l184_184300

-- Predicate to describe points in the 2x2 square
structure Point where
  x : ℝ
  y : ℝ
  h_x : 0 ≤ x ∧ x ≤ 2
  h_y : 0 ≤ y ∧ y ≤ 2

-- Function to describe the color assignment
def color (p : Point) : Prop := sorry -- True = Black, False = White

-- The main theorem to be proven
theorem exists_points_same_color_one_meter_apart :
  ∃ p1 p2 : Point, color p1 = color p2 ∧ dist (p1.1, p1.2) (p2.1, p2.2) = 1 :=
by
  sorry

end exists_points_same_color_one_meter_apart_l184_184300


namespace wendy_total_gas_to_add_l184_184699

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end wendy_total_gas_to_add_l184_184699


namespace geometric_arithmetic_sequence_l184_184254

theorem geometric_arithmetic_sequence (a_n : ℕ → ℕ) (q : ℕ) (a1_eq : a_n 1 = 3)
  (an_geometric : ∀ n, a_n (n + 1) = a_n n * q)
  (arithmetic_condition : 4 * a_n 1 + a_n 3 = 8 * a_n 2) :
  a_n 3 + a_n 4 + a_n 5 = 84 := by
  sorry

end geometric_arithmetic_sequence_l184_184254


namespace parabola_directrix_l184_184343

noncomputable def equation_of_directrix (a h k : ℝ) : ℝ :=
  k - 1 / (4 * a)

theorem parabola_directrix:
  ∀ (a h k : ℝ), a = -3 ∧ h = 1 ∧ k = -2 → equation_of_directrix a h k = - 23 / 12 :=
by
  intro a h k
  intro h_ahk
  sorry

end parabola_directrix_l184_184343


namespace tropical_fish_count_l184_184165

theorem tropical_fish_count (total_fish : ℕ) (koi_count : ℕ) (total_fish_eq : total_fish = 52) (koi_count_eq : koi_count = 37) : 
    (total_fish - koi_count) = 15 := by
    sorry

end tropical_fish_count_l184_184165


namespace sin_cos_ratio_l184_184982

theorem sin_cos_ratio (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (α - β) = 3) : 
  Real.sin (2 * α) / Real.cos (2 * β) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := 
by
  sorry

end sin_cos_ratio_l184_184982


namespace triangle_side_length_l184_184499

theorem triangle_side_length (a b c x : ℕ) (A C : ℝ) (h1 : b = x) (h2 : a = x - 2) (h3 : c = x + 2)
  (h4 : C = 2 * A) (h5 : x + 2 = 10) : a = 8 :=
by
  sorry

end triangle_side_length_l184_184499


namespace number_of_cows_l184_184775

def each_cow_milk_per_day : ℕ := 1000
def total_milk_per_week : ℕ := 364000
def days_in_week : ℕ := 7

theorem number_of_cows : 
  (total_milk_per_week = 364000) →
  (each_cow_milk_per_day = 1000) →
  (days_in_week = 7) →
  (total_milk_per_week / (each_cow_milk_per_day * days_in_week)) = 52 :=
by
  sorry

end number_of_cows_l184_184775


namespace base_number_is_three_l184_184693

theorem base_number_is_three (some_number : ℝ) (y : ℕ) (h1 : 9^y = some_number^14) (h2 : y = 7) : some_number = 3 :=
by { sorry }

end base_number_is_three_l184_184693


namespace translation_of_segment_l184_184727

structure Point where
  x : ℝ
  y : ℝ

variables (A B A' : Point)

def translation_vector (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y }

def translate (P Q : Point) : Point :=
  { x := P.x + Q.x,
    y := P.y + Q.y }

theorem translation_of_segment (hA : A = {x := -2, y := 0})
                                (hB : B = {x := 0, y := 3})
                                (hA' : A' = {x := 2, y := 1}) :
  translate B (translation_vector A A') = {x := 4, y := 4} := by
  sorry

end translation_of_segment_l184_184727


namespace final_inventory_is_correct_l184_184695

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct_l184_184695


namespace smallest_value_x_abs_eq_32_l184_184972

theorem smallest_value_x_abs_eq_32 : ∃ x : ℚ, (x = -29 / 5) ∧ (|5 * x - 3| = 32) ∧ 
  (∀ y : ℚ, (|5 * y - 3| = 32) → (x ≤ y)) :=
by
  sorry

end smallest_value_x_abs_eq_32_l184_184972


namespace parallel_lines_a_l184_184664

-- Definitions of the lines
def l1 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 6 * x + a * y + 2 = 0

-- The main theorem to prove
theorem parallel_lines_a (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = 3) := 
sorry

end parallel_lines_a_l184_184664


namespace total_frisbees_l184_184984

-- Let x be the number of $3 frisbees and y be the number of $4 frisbees.
variables (x y : ℕ)

-- Condition 1: Total sales amount is 200 dollars.
def condition1 : Prop := 3 * x + 4 * y = 200

-- Condition 2: At least 8 $4 frisbees were sold.
def condition2 : Prop := y >= 8

-- Prove that the total number of frisbees sold is 64.
theorem total_frisbees (h1 : condition1 x y) (h2 : condition2 y) : x + y = 64 :=
by
  sorry

end total_frisbees_l184_184984


namespace good_horse_catchup_l184_184937

theorem good_horse_catchup 
  (x : ℕ) 
  (good_horse_speed : ℕ) (slow_horse_speed : ℕ) (head_start_days : ℕ) 
  (H1 : good_horse_speed = 240)
  (H2 : slow_horse_speed = 150)
  (H3 : head_start_days = 12) :
  good_horse_speed * x - slow_horse_speed * x = slow_horse_speed * head_start_days :=
by
  sorry

end good_horse_catchup_l184_184937


namespace area_gray_region_in_terms_of_pi_l184_184197

variable (r : ℝ)

theorem area_gray_region_in_terms_of_pi 
    (h1 : ∀ (r : ℝ), ∃ (outer_r : ℝ), outer_r = r + 3)
    (h2 : width_gray_region = 3)
    : ∃ (area_gray : ℝ), area_gray = π * (6 * r + 9) := 
sorry

end area_gray_region_in_terms_of_pi_l184_184197


namespace minimum_ratio_cone_cylinder_l184_184648

theorem minimum_ratio_cone_cylinder (r : ℝ) (h : ℝ) (a : ℝ) :
  (h = 4 * r) →
  (a^2 = r^2 * h^2 / (h - 2 * r)) →
  (∀ h > 0, ∃ V_cone V_cylinder, 
    V_cone = (1/3) * π * a^2 * h ∧ 
    V_cylinder = π * r^2 * (2 * r) ∧ 
    V_cone / V_cylinder = (4 / 3)) := 
sorry

end minimum_ratio_cone_cylinder_l184_184648


namespace print_gift_wrap_price_l184_184647

theorem print_gift_wrap_price (solid_price : ℝ) (total_rolls : ℕ) (total_money : ℝ)
    (print_rolls : ℕ) (solid_rolls_money : ℝ) (print_money : ℝ) (P : ℝ) :
  solid_price = 4 ∧ total_rolls = 480 ∧ total_money = 2340 ∧ print_rolls = 210 ∧
  solid_rolls_money = 270 * 4 ∧ print_money = 1260 ∧
  total_money = solid_rolls_money + print_money ∧ P = print_money / 210 
  → P = 6 :=
by
  sorry

end print_gift_wrap_price_l184_184647


namespace find_locus_of_M_l184_184202

variables {P : Type*} [MetricSpace P] 
variables (A B C M : P)

def on_perpendicular_bisector (A B M : P) : Prop := 
  dist A M = dist B M

def on_circle (center : P) (radius : ℝ) (M : P) : Prop := 
  dist center M = radius

def M_AB (A B M : P) : Prop :=
  (on_perpendicular_bisector A B M) ∨ (on_circle A (dist A B) M) ∨ (on_circle B (dist A B) M)

def M_BC (B C M : P) : Prop :=
  (on_perpendicular_bisector B C M) ∨ (on_circle B (dist B C) M) ∨ (on_circle C (dist B C) M)

theorem find_locus_of_M :
  {M : P | M_AB A B M} ∩ {M : P | M_BC B C M} = {M : P | M_AB A B M ∧ M_BC B C M} :=
by sorry

end find_locus_of_M_l184_184202


namespace sum_powers_seventh_l184_184338

/-- Given the sequence values for sums of powers of 'a' and 'b', prove the value of the sum of the 7th powers. -/
theorem sum_powers_seventh (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := 
  sorry

end sum_powers_seventh_l184_184338


namespace num_int_values_not_satisfying_l184_184239

theorem num_int_values_not_satisfying:
  (∃ n : ℕ, n = 7 ∧ (∃ x : ℤ, 7 * x^2 + 25 * x + 24 ≤ 30)) :=
sorry

end num_int_values_not_satisfying_l184_184239


namespace find_smaller_number_l184_184510

theorem find_smaller_number (L S : ℕ) (h1 : L - S = 2468) (h2 : L = 8 * S + 27) : S = 349 :=
by
  sorry

end find_smaller_number_l184_184510


namespace polynomial_square_b_value_l184_184085

theorem polynomial_square_b_value (a b p q : ℝ) (h : (∀ x : ℝ, x^4 + x^3 - x^2 + a * x + b = (x^2 + p * x + q)^2)) : b = 25/64 := by
  sorry

end polynomial_square_b_value_l184_184085


namespace find_second_dimension_of_smaller_box_l184_184708

def volume_large_box : ℕ := 12 * 14 * 16
def volume_small_box (x : ℕ) : ℕ := 3 * x * 2
def max_small_boxes : ℕ := 64

theorem find_second_dimension_of_smaller_box (x : ℕ) : volume_large_box = max_small_boxes * volume_small_box x → x = 7 :=
by
  intros h
  unfold volume_large_box at h
  unfold volume_small_box at h
  sorry

end find_second_dimension_of_smaller_box_l184_184708


namespace find_pairs_satisfying_conditions_l184_184523

theorem find_pairs_satisfying_conditions (x y : ℝ) :
    abs (x + y) = 3 ∧ x * y = -10 →
    (x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2) :=
by
  sorry

end find_pairs_satisfying_conditions_l184_184523


namespace find_x_l184_184362

theorem find_x (a y x : ℤ) (h1 : y = 3) (h2 : a * y + x = 10) (h3 : a = 3) : x = 1 :=
by 
  sorry

end find_x_l184_184362


namespace row_col_value_2002_2003_l184_184021

theorem row_col_value_2002_2003 :
  let base_num := (2003 - 1)^2 + 1 
  let result := base_num + 2001 
  result = 2002 * 2003 :=
by
  sorry

end row_col_value_2002_2003_l184_184021


namespace chocolate_bars_gigantic_box_l184_184877

def large_boxes : ℕ := 50
def medium_boxes : ℕ := 25
def small_boxes : ℕ := 10
def chocolate_bars_per_small_box : ℕ := 45

theorem chocolate_bars_gigantic_box : 
  large_boxes * medium_boxes * small_boxes * chocolate_bars_per_small_box = 562500 :=
by
  sorry

end chocolate_bars_gigantic_box_l184_184877


namespace suitable_survey_is_D_l184_184194

-- Define the surveys
def survey_A := "Survey on the viewing of the movie 'The Long Way Home' by middle school students in our city"
def survey_B := "Survey on the germination rate of a batch of rose seeds"
def survey_C := "Survey on the water quality of the Jialing River"
def survey_D := "Survey on the health codes of students during the epidemic"

-- Define what it means for a survey to be suitable for a comprehensive census
def suitable_for_census (survey : String) : Prop :=
  survey = survey_D

-- Define the main theorem statement
theorem suitable_survey_is_D : suitable_for_census survey_D :=
by
  -- We assume sorry here to skip the proof
  sorry

end suitable_survey_is_D_l184_184194


namespace stone_145_is_5_l184_184397

theorem stone_145_is_5 :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 15) → (145 % 28) = 5 → n = 5 :=
by
  intros n h h145
  sorry

end stone_145_is_5_l184_184397


namespace ratio_of_sides_l184_184803
-- Import the complete math library

-- Define the conditions as hypotheses
variables (s x y : ℝ)
variable (h_outer_area : (3 * s)^2 = 9 * s^2)
variable (h_side_lengths : 3 * s = s + 2 * x)
variable (h_y_length : y + x = 3 * s)

-- State the theorem
theorem ratio_of_sides (h_outer_area : (3 * s)^2 = 9 * s^2)
  (h_side_lengths : 3 * s = s + 2 * x)
  (h_y_length : y + x = 3 * s) :
  y / x = 2 := by
  sorry

end ratio_of_sides_l184_184803


namespace functional_eq_solution_l184_184191

theorem functional_eq_solution (f : ℤ → ℤ) (h : ∀ x y : ℤ, x ≠ 0 →
  x * f (2 * f y - x) + y^2 * f (2 * x - f y) = (f x ^ 2) / x + f (y * f y)) :
  (∀ x: ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end functional_eq_solution_l184_184191


namespace total_cost_correct_l184_184132

-- Define the parameters
variables (a : ℕ) -- the number of books
-- Define the constants and the conditions
def unit_price : ℝ := 8
def shipping_fee_percentage : ℝ := 0.10

-- Define the total cost including the shipping fee
def total_cost (a : ℕ) : ℝ := unit_price * (1 + shipping_fee_percentage) * a

-- Prove that the total cost is equal to the expected amount
theorem total_cost_correct : total_cost a = 8 * (1 + 0.10) * a := by
  sorry

end total_cost_correct_l184_184132


namespace gcd_75_100_l184_184030

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l184_184030


namespace max_students_seated_l184_184251

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

end max_students_seated_l184_184251


namespace student_first_subject_percentage_l184_184720

variable (P : ℝ)

theorem student_first_subject_percentage 
  (H1 : 80 = 80)
  (H2 : 75 = 75)
  (H3 : (P + 80 + 75) / 3 = 75) :
  P = 70 :=
by
  sorry

end student_first_subject_percentage_l184_184720


namespace smallest_possible_value_l184_184530

theorem smallest_possible_value (a b c d : ℤ) 
  (h1 : a + b + c + d < 25) 
  (h2 : a > 8) 
  (h3 : b < 5) 
  (h4 : c % 2 = 1) 
  (h5 : d % 2 = 0) : 
  ∃ a' b' c' d' : ℤ, a' > 8 ∧ b' < 5 ∧ c' % 2 = 1 ∧ d' % 2 = 0 ∧ a' + b' + c' + d' < 25 ∧ (a' - b' + c' - d' = -4) := 
by 
  use 9, 4, 1, 10
  sorry

end smallest_possible_value_l184_184530


namespace sin_cos_15_sin_cos_18_l184_184274

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

end sin_cos_15_sin_cos_18_l184_184274


namespace count_valid_triples_l184_184753

-- Define the necessary conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_positive (n : ℕ) : Prop := n > 0
def valid_triple (p q n : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_positive n ∧ (1/p + 2013/q = n/5)

-- Lean statement for the proof problem
theorem count_valid_triples : 
  ∃ c : ℕ, c = 5 ∧ 
  (∀ p q n : ℕ, valid_triple p q n → true) :=
sorry

end count_valid_triples_l184_184753


namespace fraction_increase_by_three_l184_184636

variables (a b : ℝ)

theorem fraction_increase_by_three : 
  3 * (2 * a * b / (3 * a - 4 * b)) = 2 * (3 * a * 3 * b) / (3 * (3 * a) - 4 * (3 * b)) :=
by
  sorry

end fraction_increase_by_three_l184_184636


namespace maple_is_taller_l184_184230

def pine_tree_height : ℚ := 13 + 1/4
def maple_tree_height : ℚ := 20 + 1/2
def height_difference : ℚ := maple_tree_height - pine_tree_height

theorem maple_is_taller : height_difference = 7 + 1/4 := by
  sorry

end maple_is_taller_l184_184230


namespace rectangle_area_ratio_l184_184084

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 :=
by
  sorry

end rectangle_area_ratio_l184_184084


namespace trig_identity_l184_184484

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end trig_identity_l184_184484


namespace boys_amount_per_person_l184_184148

theorem boys_amount_per_person (total_money : ℕ) (total_children : ℕ) (per_girl : ℕ) (number_of_boys : ℕ) (amount_per_boy : ℕ) : 
  total_money = 460 ∧
  total_children = 41 ∧
  per_girl = 8 ∧
  number_of_boys = 33 → 
  amount_per_boy = 12 :=
by sorry

end boys_amount_per_person_l184_184148


namespace probability_odd_80_heads_l184_184224

noncomputable def coin_toss_probability_odd (n : ℕ) (p : ℝ) : ℝ :=
  (1 / 2) * (1 - (1 / 3^n))

theorem probability_odd_80_heads :
  coin_toss_probability_odd 80 (3 / 4) = (1 / 2) * (1 - 1 / 3^80) :=
by
  sorry

end probability_odd_80_heads_l184_184224


namespace min_distance_value_l184_184880

theorem min_distance_value (x1 x2 y1 y2 : ℝ) 
  (h1 : (e ^ x1 + 2 * x1) / (3 * y1) = 1 / 3)
  (h2 : (x2 - 1) / y2 = 1 / 3) :
  ((x1 - x2)^2 + (y1 - y2)^2) = 8 / 5 :=
by
  sorry

end min_distance_value_l184_184880


namespace sin_thirteen_pi_over_six_l184_184166

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end sin_thirteen_pi_over_six_l184_184166


namespace points_per_correct_answer_l184_184851

theorem points_per_correct_answer (x : ℕ) : 
  let total_questions := 30
  let points_deducted_per_incorrect := 5
  let total_score := 325
  let correct_answers := 19
  let incorrect_answers := total_questions - correct_answers
  let points_lost_from_incorrect := incorrect_answers * points_deducted_per_incorrect
  let score_from_correct := correct_answers * x
  (score_from_correct - points_lost_from_incorrect = total_score) → x = 20 :=
by {
  sorry
}

end points_per_correct_answer_l184_184851


namespace bicycle_distance_l184_184169

def distance : ℝ := 15

theorem bicycle_distance :
  ∀ (x y : ℝ),
  (x + 6) * (y - 5 / 60) = x * y →
  (x - 5) * (y + 6 / 60) = x * y →
  x * y = distance :=
by
  intros x y h1 h2
  sorry

end bicycle_distance_l184_184169


namespace trig_inequality_sin_cos_l184_184607

theorem trig_inequality_sin_cos :
  Real.sin 2 + Real.cos 2 + 2 * (Real.sin 1 - Real.cos 1) ≥ 1 :=
by
  sorry

end trig_inequality_sin_cos_l184_184607


namespace correct_division_result_l184_184836

theorem correct_division_result : 
  ∀ (a b : ℕ),
  (1722 / (10 * b + a) = 42) →
  (10 * a + b = 14) →
  1722 / 14 = 123 :=
by
  intros a b h1 h2
  sorry

end correct_division_result_l184_184836


namespace negation_of_universal_proposition_l184_184554
open Classical

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 ≥ 3)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := 
by
  sorry

end negation_of_universal_proposition_l184_184554


namespace power_binary_representation_zero_digit_l184_184449

theorem power_binary_representation_zero_digit
  (a n s : ℕ) (ha : a > 1) (hn : n > 1) (hs : s > 0) :
  a ^ n ≠ 2 ^ s - 1 :=
by
  sorry

end power_binary_representation_zero_digit_l184_184449


namespace like_terms_monomials_l184_184919

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end like_terms_monomials_l184_184919


namespace det_matrix_example_l184_184805

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem det_matrix_example : det_2x2 4 5 2 3 = 2 :=
by
  sorry

end det_matrix_example_l184_184805


namespace passing_percentage_is_correct_l184_184199

theorem passing_percentage_is_correct :
  ∀ (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ),
    marks_obtained = 59 →
    marks_failed_by = 40 →
    max_marks = 300 →
    (marks_obtained + marks_failed_by) / max_marks * 100 = 33 :=
by
  intros marks_obtained marks_failed_by max_marks h1 h2 h3
  sorry

end passing_percentage_is_correct_l184_184199


namespace clubsuit_problem_l184_184344

def clubsuit (x y : ℤ) : ℤ :=
  (x^2 + y^2) * (x - y)

theorem clubsuit_problem : clubsuit 2 (clubsuit 3 4) = 16983 := 
by 
  sorry

end clubsuit_problem_l184_184344


namespace max_tiles_accommodated_l184_184903

/-- 
The rectangular tiles, each of size 40 cm by 28 cm, must be laid horizontally on a rectangular floor
of size 280 cm by 240 cm, such that the tiles do not overlap, and they are placed in an alternating
checkerboard pattern with edges jutting against each other on all edges. A tile can be placed in any
orientation so long as its edges are parallel to the edges of the floor, and it follows the required
checkerboard pattern. No tile should overshoot any edge of the floor. Determine the maximum number 
of tiles that can be accommodated on the floor while adhering to the placement pattern.
-/
theorem max_tiles_accommodated (tile_len tile_wid floor_len floor_wid : ℕ)
  (h_tile_len : tile_len = 40)
  (h_tile_wid : tile_wid = 28)
  (h_floor_len : floor_len = 280)
  (h_floor_wid : floor_wid = 240) :
  tile_len * tile_wid * 12 ≤ floor_len * floor_wid :=
by 
  sorry

end max_tiles_accommodated_l184_184903


namespace value_of_b_minus_a_l184_184076

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2)

theorem value_of_b_minus_a (a b : ℝ) (h1 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1 : ℝ) 2) (h2 : ∀ x, f x = 2 * Real.sin (x / 2)) : 
  b - a ≠ 14 * Real.pi / 3 :=
sorry

end value_of_b_minus_a_l184_184076


namespace arrange_moon_l184_184118

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end arrange_moon_l184_184118


namespace abc_system_proof_l184_184461

theorem abc_system_proof (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 + a = b^2) (h5 : b^2 + b = c^2) (h6 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 :=
by
  sorry

end abc_system_proof_l184_184461


namespace minimum_draws_divisible_by_3_or_5_l184_184071

theorem minimum_draws_divisible_by_3_or_5 (n : ℕ) (h : n = 90) :
  ∃ k, k = 49 ∧ ∀ (draws : ℕ), draws < k → ¬ (∃ x, 1 ≤ x ∧ x ≤ n ∧ (x % 3 = 0 ∨ x % 5 = 0)) :=
by {
  sorry
}

end minimum_draws_divisible_by_3_or_5_l184_184071


namespace fraction_inequality_l184_184579

open Real

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) :=
  sorry

end fraction_inequality_l184_184579


namespace reciprocal_solution_l184_184765

theorem reciprocal_solution {x : ℝ} (h : x * -9 = 1) : x = -1/9 :=
sorry

end reciprocal_solution_l184_184765


namespace A_minus_one_not_prime_l184_184736

theorem A_minus_one_not_prime (n : ℕ) (h : 0 < n) (m : ℕ) (h1 : 10^(m-1) < 14^n) (h2 : 14^n < 10^m) :
  ¬ (Nat.Prime (2^n * 10^m + 14^n - 1)) :=
by
  sorry

end A_minus_one_not_prime_l184_184736


namespace parabola_ratio_l184_184978

noncomputable def ratio_AF_BF (p : ℝ) (h_pos : p > 0) : ℝ :=
  let y1 := (Real.sqrt (2 * p * (3 / 2 * p)))
  let y2 := (Real.sqrt (2 * p * (1 / 6 * p)))
  let dist1 := Real.sqrt ((3 / 2 * p - (p / 2))^2 + y1^2)
  let dist2 := Real.sqrt ((1 / 6 * p - p / 2)^2 + y2^2)
  dist1 / dist2

theorem parabola_ratio (p : ℝ) (h_pos : p > 0) : ratio_AF_BF p h_pos = 3 :=
  sorry

end parabola_ratio_l184_184978


namespace tristan_study_hours_l184_184600

theorem tristan_study_hours :
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  saturday_hours = 2 := by
{
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  sorry
}

end tristan_study_hours_l184_184600


namespace no_prime_divisible_by_77_l184_184181

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l184_184181


namespace cash_realized_before_brokerage_l184_184266

theorem cash_realized_before_brokerage (C : ℝ) (h1 : 0.25 / 100 * C = C / 400)
(h2 : C - C / 400 = 108) : C = 108.27 :=
by
  sorry

end cash_realized_before_brokerage_l184_184266


namespace work_problem_l184_184998

theorem work_problem (x : ℝ) (hx : x > 0)
    (hB : B_work_rate = 1 / 18)
    (hTogether : together_work_rate = 1 / 7.2)
    (hCombined : together_work_rate = 1 / x + B_work_rate) :
    x = 2 := by
    sorry

end work_problem_l184_184998


namespace vasya_100_using_fewer_sevens_l184_184521

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l184_184521


namespace area_R_l184_184455

-- Define the given matrix as a 2x2 real matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, -5]

-- Define the original area of region R
def area_R : ℝ := 15

-- Define the area scaling factor as the absolute value of the determinant of A
def scaling_factor : ℝ := |Matrix.det A|

-- Prove that the area of the region R' is 585
theorem area_R' : scaling_factor * area_R = 585 := by
  sorry

end area_R_l184_184455


namespace chair_cost_l184_184718

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end chair_cost_l184_184718


namespace rainfall_second_week_l184_184565

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) (first_week_rainfall : ℝ) (second_week_rainfall : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  total_rainfall = first_week_rainfall + second_week_rainfall →
  second_week_rainfall = ratio * first_week_rainfall →
  second_week_rainfall = 21 :=
by
  intros
  sorry

end rainfall_second_week_l184_184565


namespace orange_price_l184_184630

theorem orange_price (initial_apples : ℕ) (initial_oranges : ℕ) 
                     (apple_price : ℝ) (total_earnings : ℝ) 
                     (remaining_apples : ℕ) (remaining_oranges : ℕ)
                     (h1 : initial_apples = 50) (h2 : initial_oranges = 40)
                     (h3 : apple_price = 0.80) (h4 : total_earnings = 49)
                     (h5 : remaining_apples = 10) (h6 : remaining_oranges = 6) :
  ∃ orange_price : ℝ, orange_price = 0.50 :=
by
  sorry

end orange_price_l184_184630


namespace seashells_given_to_Jessica_l184_184969

-- Define the initial number of seashells Dan had
def initialSeashells : ℕ := 56

-- Define the number of seashells Dan has left
def seashellsLeft : ℕ := 22

-- Define the number of seashells Dan gave to Jessica
def seashellsGiven : ℕ := initialSeashells - seashellsLeft

-- State the theorem to prove
theorem seashells_given_to_Jessica :
  seashellsGiven = 34 :=
by
  -- Begin the proof here
  sorry

end seashells_given_to_Jessica_l184_184969


namespace area_of_isosceles_triangle_l184_184367

theorem area_of_isosceles_triangle
  (h : ℝ)
  (s : ℝ)
  (b : ℝ)
  (altitude : h = 10)
  (perimeter : s + (s - 2) + 2 * b = 40)
  (pythagoras : b^2 + h^2 = s^2) :
  (b * h) = 81.2 :=
by
  sorry

end area_of_isosceles_triangle_l184_184367


namespace solution_set_of_inequality_l184_184313

theorem solution_set_of_inequality (x : ℝ) : ((x - 1) * (2 - x) ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_of_inequality_l184_184313


namespace art_gallery_ratio_l184_184780

theorem art_gallery_ratio (A : ℕ) (D : ℕ) (S_not_displayed : ℕ) (P_not_displayed : ℕ)
  (h1 : A = 2700)
  (h2 : 1 / 6 * D = D / 6)
  (h3 : P_not_displayed = S_not_displayed / 3)
  (h4 : S_not_displayed = 1200) :
  D / A = 11 / 27 := by
  sorry

end art_gallery_ratio_l184_184780


namespace mode_of_dataSet_is_3_l184_184916

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end mode_of_dataSet_is_3_l184_184916


namespace marble_draw_probability_l184_184795

def marble_probabilities : ℚ :=
  let prob_white_a := 5 / 10
  let prob_black_a := 5 / 10
  let prob_yellow_b := 8 / 15
  let prob_yellow_c := 3 / 10
  let prob_green_d := 6 / 10
  let prob_white_then_yellow_then_green := prob_white_a * prob_yellow_b * prob_green_d
  let prob_black_then_yellow_then_green := prob_black_a * prob_yellow_c * prob_green_d
  prob_white_then_yellow_then_green + prob_black_then_yellow_then_green

theorem marble_draw_probability :
  marble_probabilities = 17 / 50 := by
  sorry

end marble_draw_probability_l184_184795


namespace find_x_l184_184064

theorem find_x :
  ∃ x : ℚ, (1 / 3) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 ∧ x = 47 / 3 :=
by
  sorry

end find_x_l184_184064


namespace subtraction_is_addition_of_negatives_l184_184628

theorem subtraction_is_addition_of_negatives : (-1) - 3 = -4 := by
  sorry

end subtraction_is_addition_of_negatives_l184_184628


namespace additional_kgs_l184_184651

variables (P R A : ℝ)
variables (h1 : R = 0.80 * P) (h2 : R = 34.2) (h3 : 684 = A * R)

theorem additional_kgs :
  A = 20 :=
by
  sorry

end additional_kgs_l184_184651


namespace vec_sub_eq_l184_184271

variables (a b : ℝ × ℝ)
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (-3, 4)

theorem vec_sub_eq : vec_a - vec_b = (5, -3) :=
by 
  -- You can fill in the proof steps here
  sorry

end vec_sub_eq_l184_184271


namespace circumference_of_tank_a_l184_184175

def is_circumference_of_tank_a (h_A h_B C_B : ℝ) (V_A_eq : ℝ → Prop) : Prop :=
  ∃ (C_A : ℝ), 
    C_B = 10 ∧ 
    h_A = 10 ∧
    h_B = 7 ∧
    V_A_eq 0.7 ∧ 
    C_A = 7

theorem circumference_of_tank_a (h_A : ℝ) (h_B : ℝ) (C_B : ℝ) (V_A_eq : ℝ → Prop) : 
  is_circumference_of_tank_a h_A h_B C_B V_A_eq := 
by
  sorry

end circumference_of_tank_a_l184_184175


namespace intersection_A_B_l184_184517

-- Conditions
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

-- Proof of the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l184_184517


namespace ratio_area_circle_to_triangle_l184_184322

theorem ratio_area_circle_to_triangle (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
    (π * r) / (h + r) = (π * r ^ 2) / (r * (h + r)) := sorry

end ratio_area_circle_to_triangle_l184_184322


namespace y_directly_varies_as_square_l184_184895

theorem y_directly_varies_as_square (k : ℚ) (y : ℚ) (x : ℚ) 
  (h1 : y = k * x ^ 2) (h2 : y = 18) (h3 : x = 3) : 
  ∃ y : ℚ, ∀ x : ℚ, x = 6 → y = 72 :=
by
  sorry

end y_directly_varies_as_square_l184_184895


namespace rachel_more_than_adam_l184_184095

variable (R J A : ℕ)

def condition1 := R = 75
def condition2 := R = J - 6
def condition3 := R > A
def condition4 := (R + J + A) / 3 = 72

theorem rachel_more_than_adam
  (h1 : condition1 R)
  (h2 : condition2 R J)
  (h3 : condition3 R A)
  (h4 : condition4 R J A) : 
  R - A = 15 := 
by
  sorry

end rachel_more_than_adam_l184_184095


namespace infinite_series_equals_3_l184_184675

noncomputable def infinite_series_sum := ∑' (k : ℕ), (12^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

theorem infinite_series_equals_3 : infinite_series_sum = 3 := by
  sorry

end infinite_series_equals_3_l184_184675


namespace problem_statement_l184_184723

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 14) : 6 * (x + 3) = 4.8 :=
sorry

end problem_statement_l184_184723


namespace unique_real_solution_l184_184378

theorem unique_real_solution (a : ℝ) : 
  (∀ x : ℝ, (x^3 - a * x^2 - (a + 1) * x + (a^2 - 2) = 0)) ↔ (a < 7 / 4) := 
sorry

end unique_real_solution_l184_184378


namespace P_subset_Q_l184_184483

def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x > 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l184_184483


namespace three_pow_zero_l184_184189

theorem three_pow_zero : 3^0 = 1 :=
by sorry

end three_pow_zero_l184_184189


namespace part1_part2_l184_184658

noncomputable def f (a : ℝ) (x : ℝ) := (a * x - 1) * (x - 1)

theorem part1 (h : ∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) : a = 1/2 :=
  sorry

theorem part2 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 1/a) ∨
  (a = 1 → ∀ x : ℝ, ¬(f a x < 0)) ∨
  (∀ x : ℝ, f a x < 0 ↔ 1/a < x ∧ x < 1) :=
  sorry

end part1_part2_l184_184658


namespace darma_peanut_consumption_l184_184688

theorem darma_peanut_consumption :
  ∀ (t : ℕ) (rate : ℕ),
  (rate = 20 / 15) →  -- Given the rate of peanut consumption
  (t = 6 * 60) →     -- Given that the total time is 6 minutes
  (rate * t = 480) :=  -- Prove that the total number of peanuts eaten in 6 minutes is 480
by
  intros t rate h_rate h_time
  sorry

end darma_peanut_consumption_l184_184688


namespace Dan_must_exceed_speed_l184_184682

theorem Dan_must_exceed_speed (distance : ℝ) (Cara_speed : ℝ) (delay : ℝ) (time_Cara : ℝ) (Dan_time : ℝ) : 
  distance = 120 ∧ Cara_speed = 30 ∧ delay = 1 ∧ time_Cara = distance / Cara_speed ∧ time_Cara = 4 ∧ Dan_time = time_Cara - delay ∧ Dan_time < 4 → 
  (distance / Dan_time) > 40 :=
by
  sorry

end Dan_must_exceed_speed_l184_184682


namespace inequality_of_pos_real_product_l184_184260

theorem inequality_of_pos_real_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) :=
sorry

end inequality_of_pos_real_product_l184_184260


namespace number_of_boxes_on_pallet_l184_184994

-- Define the total weight of the pallet.
def total_weight_of_pallet : ℤ := 267

-- Define the weight of each box.
def weight_of_each_box : ℤ := 89

-- The theorem states that given the total weight of the pallet and the weight of each box,
-- the number of boxes on the pallet is 3.
theorem number_of_boxes_on_pallet : total_weight_of_pallet / weight_of_each_box = 3 :=
by sorry

end number_of_boxes_on_pallet_l184_184994


namespace geometric_sum_over_term_l184_184100

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def geometric_term (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem geometric_sum_over_term (a₁ : ℝ) (q : ℝ) (h₁ : q = 3) :
  (geometric_sum a₁ q 4) / (geometric_term a₁ q 4) = 40 / 27 := by
  sorry

end geometric_sum_over_term_l184_184100


namespace find_parabola_equation_l184_184534

noncomputable def parabola_equation (a : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A : ℝ × ℝ), 
    F.1 = a / 4 ∧ F.2 = 0 ∧
    A.1 = 0 ∧ A.2 = a / 2 ∧
    (abs (F.1 * A.2) / 2) = 4

theorem find_parabola_equation :
  ∀ (a : ℝ), parabola_equation a → a = 8 ∨ a = -8 :=
by
  sorry

end find_parabola_equation_l184_184534


namespace x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l184_184962

variable {x : ℝ}
variable {y : ℝ}

theorem x_gt_y_necessary_not_sufficient_for_x_gt_abs_y
  (hx : x > 0) :
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|) := by
  sorry

end x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l184_184962


namespace original_number_is_15_l184_184065

theorem original_number_is_15 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (N : ℕ) (h4 : 100 * a + 10 * b + c = m)
  (h5 : 100 * a +  10 * b +   c +
        100 * a +   c + 10 * b + 
        100 * b +  10 * a +   c +
        100 * b +   c + 10 * a + 
        100 * c +  10 * a +   b +
        100 * c +   b + 10 * a = 3315) :
  m = 15 :=
sorry

end original_number_is_15_l184_184065


namespace smallest_integer_remainder_l184_184269

theorem smallest_integer_remainder (n : ℕ) 
  (h5 : n ≡ 1 [MOD 5]) (h7 : n ≡ 1 [MOD 7]) (h8 : n ≡ 1 [MOD 8]) :
  80 < n ∧ n < 299 := 
sorry

end smallest_integer_remainder_l184_184269


namespace butterfly_eq_roots_l184_184520

theorem butterfly_eq_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0)
    (h3 : (a + c)^2 - 4 * a * c = 0) : a = c :=
by
  sorry

end butterfly_eq_roots_l184_184520


namespace night_crew_fraction_of_day_l184_184674

variable (D : ℕ) -- Number of workers in the day crew
variable (N : ℕ) -- Number of workers in the night crew
variable (total_boxes : ℕ) -- Total number of boxes loaded by both crews

-- Given conditions
axiom day_fraction : D > 0 ∧ N > 0 ∧ total_boxes > 0
axiom night_workers_fraction : N = (4 * D) / 5
axiom day_crew_boxes_fraction : (5 * total_boxes) / 7 = (5 * D)
axiom night_crew_boxes_fraction : (2 * total_boxes) / 7 = (2 * N)

-- To prove
theorem night_crew_fraction_of_day : 
  let F_d := (5 : ℚ) / (7 * D)
  let F_n := (2 : ℚ) / (7 * N)
  F_n = (5 / 14) * F_d :=
by
  sorry

end night_crew_fraction_of_day_l184_184674


namespace smallest_y_for_perfect_square_l184_184854

theorem smallest_y_for_perfect_square (x y: ℕ) (h : x = 5 * 32 * 45) (hY: y = 2) : 
  ∃ v: ℕ, (x * y = v ^ 2) :=
by
  use 2
  rw [h, hY]
  -- expand and simplify
  sorry

end smallest_y_for_perfect_square_l184_184854


namespace third_square_is_G_l184_184905

-- Conditions
-- Define eight 2x2 squares, where the last placed square is E
def squares : List String := ["F", "H", "G", "D", "A", "B", "C", "E"]

-- Let the third square be G
def third_square := "G"

-- Proof statement
theorem third_square_is_G : squares.get! 2 = third_square :=
by
  sorry

end third_square_is_G_l184_184905


namespace cost_of_ground_school_l184_184029

theorem cost_of_ground_school (G : ℝ) (F : ℝ) (h1 : F = G + 625) (h2 : F = 950) :
  G = 325 :=
by
  sorry

end cost_of_ground_school_l184_184029


namespace ratio_a_b_c_l184_184295

theorem ratio_a_b_c (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b + c) / 3 = 42) (h5 : a = 28) : 
  ∃ y z : ℕ, a / 28 = 1 ∧ b / (ky) = 1 / k ∧ c / (kz) = 1 / k ∧ (b + c) = 98 :=
by sorry

end ratio_a_b_c_l184_184295


namespace insurance_covers_80_percent_l184_184776

def xray_cost : ℕ := 250
def mri_cost : ℕ := 3 * xray_cost
def total_cost : ℕ := xray_cost + mri_cost
def mike_payment : ℕ := 200
def insurance_coverage : ℕ := total_cost - mike_payment
def insurance_percentage : ℕ := (insurance_coverage * 100) / total_cost

theorem insurance_covers_80_percent : insurance_percentage = 80 := by
  -- Carry out the necessary calculations
  sorry

end insurance_covers_80_percent_l184_184776


namespace find_square_number_divisible_by_six_l184_184733

theorem find_square_number_divisible_by_six :
  ∃ x : ℕ, (∃ k : ℕ, x = k^2) ∧ x % 6 = 0 ∧ 24 < x ∧ x < 150 ∧ (x = 36 ∨ x = 144) :=
by {
  sorry
}

end find_square_number_divisible_by_six_l184_184733


namespace solution_Y_required_l184_184751

theorem solution_Y_required (V_total V_ratio_Y : ℝ) (h_total : V_total = 0.64) (h_ratio : V_ratio_Y = 3 / 8) : 
  (0.64 * (3 / 8) = 0.24) :=
by
  sorry

end solution_Y_required_l184_184751


namespace find_theta_l184_184439

theorem find_theta (θ : Real) (h : abs θ < π / 2) (h_eq : Real.sin (π + θ) = -Real.sqrt 3 * Real.cos (2 * π - θ)) :
  θ = π / 3 :=
sorry

end find_theta_l184_184439


namespace hexagon_side_relation_l184_184649

noncomputable def hexagon (a b c d e f : ℝ) :=
  ∃ (i j k l m n : ℝ), 
    i = 120 ∧ j = 120 ∧ k = 120 ∧ l = 120 ∧ m = 120 ∧ n = 120 ∧  
    a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f ∧ f = a

theorem hexagon_side_relation
  (a b c d e f : ℝ)
  (ha : hexagon a b c d e f) :
  d - a = b - e ∧ b - e = f - c :=
by
  sorry

end hexagon_side_relation_l184_184649


namespace find_larger_number_l184_184906

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1500) (h2 : y = 6 * x + 15) : y = 1797 := by
  sorry

end find_larger_number_l184_184906


namespace quadratic_inequality_solution_l184_184261

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 - 5 * x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end quadratic_inequality_solution_l184_184261


namespace solution_correctness_l184_184740

noncomputable def solution_set : Set ℝ := { x : ℝ | (x + 1) * (x - 2) > 0 }

theorem solution_correctness (x : ℝ) :
  (x ∈ solution_set) ↔ (x < -1 ∨ x > 2) :=
by sorry

end solution_correctness_l184_184740


namespace min_value_of_reciprocal_sum_l184_184673

variable (a b : ℝ)
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (condition : 2 * a + b = 1)

theorem min_value_of_reciprocal_sum : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 :=
by
  -- Proof is skipped
  sorry

end min_value_of_reciprocal_sum_l184_184673


namespace min_c_for_expression_not_min_abs_c_for_expression_l184_184036

theorem min_c_for_expression :
  ∀ c : ℝ,
  (c - 3)^2 + (c - 4)^2 + (c - 8)^2 ≥ (5 - 3)^2 + (5 - 4)^2 + (5 - 8)^2 := 
by sorry

theorem not_min_abs_c_for_expression :
  ∃ c : ℝ, |c - 3| + |c - 4| + |c - 8| < |5 - 3| + |5 - 4| + |5 - 8| := 
by sorry

end min_c_for_expression_not_min_abs_c_for_expression_l184_184036


namespace sum_of_roots_l184_184981

theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hroots : ∀ x : ℝ, x^2 - p*x + 2*q = 0) :
  p + q = p :=
by sorry

end sum_of_roots_l184_184981


namespace B_greater_than_A_l184_184034

def A := (54 : ℚ) / (5^7 * 11^4 : ℚ)
def B := (55 : ℚ) / (5^7 * 11^4 : ℚ)

theorem B_greater_than_A : B > A := by
  sorry

end B_greater_than_A_l184_184034


namespace geometric_series_sum_l184_184209

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 6
  let sum := a * ((1 - r ^ n) / (1 - r))
  sum = (4095 / 5120 : ℚ) :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l184_184209


namespace beehive_bee_count_l184_184902

theorem beehive_bee_count {a : ℕ → ℕ} (h₀ : a 0 = 1)
  (h₁ : a 1 = 6)
  (hn : ∀ n, a (n + 1) = a n + 5 * a n) :
  a 6 = 46656 :=
  sorry

end beehive_bee_count_l184_184902


namespace JiaZi_second_column_l184_184893

theorem JiaZi_second_column :
  let heavenlyStemsCycle := 10
  let earthlyBranchesCycle := 12
  let firstOccurrence := 1
  let lcmCycle := Nat.lcm heavenlyStemsCycle earthlyBranchesCycle
  let secondOccurrence := firstOccurrence + lcmCycle
  secondOccurrence = 61 :=
by
  sorry

end JiaZi_second_column_l184_184893


namespace volume_of_larger_cube_is_343_l184_184694

-- We will define the conditions first
def smaller_cube_side_length : ℤ := 1
def number_of_smaller_cubes : ℤ := 343
def volume_small_cube (l : ℤ) : ℤ := l^3
def diff_surface_area (l L : ℤ) : ℤ := (number_of_smaller_cubes * 6 * l^2) - (6 * L^2)

-- Main statement to prove the volume of the larger cube
theorem volume_of_larger_cube_is_343 :
  ∃ L, volume_small_cube smaller_cube_side_length * number_of_smaller_cubes = L^3 ∧
        diff_surface_area smaller_cube_side_length L = 1764 ∧
        volume_small_cube L = 343 :=
by
  sorry

end volume_of_larger_cube_is_343_l184_184694


namespace combined_platforms_length_is_correct_l184_184963

noncomputable def combined_length_of_platforms (lengthA lengthB speedA_kmph speedB_kmph timeA_sec timeB_sec : ℝ) : ℝ :=
  let speedA := speedA_kmph * (1000 / 3600)
  let speedB := speedB_kmph * (1000 / 3600)
  let distanceA := speedA * timeA_sec
  let distanceB := speedB * timeB_sec
  let platformA := distanceA - lengthA
  let platformB := distanceB - lengthB
  platformA + platformB

theorem combined_platforms_length_is_correct :
  combined_length_of_platforms 650 450 115 108 30 25 = 608.32 := 
by 
  sorry

end combined_platforms_length_is_correct_l184_184963


namespace range_of_m_non_perpendicular_tangent_l184_184098

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_non_perpendicular_tangent (m : ℝ) :
  (∀ x : ℝ, (deriv (f m) x ≠ -2)) → m ≤ 2 :=
by
  sorry

end range_of_m_non_perpendicular_tangent_l184_184098


namespace necessary_but_not_sufficient_l184_184646

def angle_of_inclination (α : ℝ) : Prop :=
  α > Real.pi / 4

def slope_of_line (k : ℝ) : Prop :=
  k > 1

theorem necessary_but_not_sufficient (α k : ℝ) :
  angle_of_inclination α → (slope_of_line k → (k = Real.tan α)) → (angle_of_inclination α → slope_of_line k) ∧ ¬(slope_of_line k → angle_of_inclination α) :=
by
  sorry

end necessary_but_not_sufficient_l184_184646


namespace system_linear_eq_sum_l184_184777

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end system_linear_eq_sum_l184_184777


namespace model_tower_height_l184_184743

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_cond : real_height = 80) (vol_cond : real_volume = 200000) (model_vol_cond : model_volume = 0.2) : 
  ∃ h : ℝ, h = 0.8 :=
by sorry

end model_tower_height_l184_184743


namespace remaining_length_l184_184959

variable (L₁ L₂: ℝ)
variable (H₁: L₁ = 0.41)
variable (H₂: L₂ = 0.33)

theorem remaining_length (L₁ L₂: ℝ) (H₁: L₁ = 0.41) (H₂: L₂ = 0.33) : L₁ - L₂ = 0.08 :=
by
  sorry

end remaining_length_l184_184959


namespace widgets_per_shipping_box_l184_184816

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end widgets_per_shipping_box_l184_184816


namespace percentage_shoes_polished_l184_184599

theorem percentage_shoes_polished (total_pairs : ℕ) (shoes_to_polish : ℕ)
  (total_individual_shoes : ℕ := total_pairs * 2)
  (shoes_polished : ℕ := total_individual_shoes - shoes_to_polish)
  (percentage_polished : ℚ := (shoes_polished : ℚ) / total_individual_shoes * 100) :
  total_pairs = 10 → shoes_to_polish = 11 → percentage_polished = 45 :=
by
  intros hpairs hleft
  sorry

end percentage_shoes_polished_l184_184599


namespace evaluate_sum_of_powers_of_i_l184_184190

-- Definition of the imaginary unit i with property i^2 = -1.
def i : ℂ := Complex.I

lemma i_pow_2 : i^2 = -1 := by
  sorry

lemma i_pow_4n (n : ℤ) : i^(4 * n) = 1 := by
  sorry

-- Problem statement: Evaluate i^13 + i^18 + i^23 + i^28 + i^33 + i^38.
theorem evaluate_sum_of_powers_of_i : 
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by
  sorry

end evaluate_sum_of_powers_of_i_l184_184190


namespace poly_expansion_l184_184051

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end poly_expansion_l184_184051


namespace grading_ratio_l184_184973

noncomputable def num_questions : ℕ := 100
noncomputable def correct_answers : ℕ := 91
noncomputable def score_received : ℕ := 73
noncomputable def incorrect_answers : ℕ := num_questions - correct_answers
noncomputable def total_points_subtracted : ℕ := correct_answers - score_received
noncomputable def points_per_incorrect : ℚ := total_points_subtracted / incorrect_answers

theorem grading_ratio (h: (points_per_incorrect : ℚ) = 2) :
  2 / 1 = points_per_incorrect / 1 :=
by sorry

end grading_ratio_l184_184973


namespace part1_part2_l184_184545

open Real

noncomputable def f (x a : ℝ) : ℝ := 45 * abs (x - a) + 45 * abs (x - 5)

theorem part1 (a : ℝ) :
    (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≤ 2 ∨ a ≥ 8) :=
sorry

theorem part2 (a : ℝ) (ha : a = 2) :
    ∀ (x : ℝ), (f x 2 ≥ x^2 - 8*x + 15) ↔ (2 ≤ x ∧ x ≤ 5 + Real.sqrt 3) :=
sorry

end part1_part2_l184_184545


namespace salami_pizza_fraction_l184_184697

theorem salami_pizza_fraction 
    (d_pizza : ℝ) 
    (n_salami_diameter : ℕ) 
    (n_salami_total : ℕ) 
    (h1 : d_pizza = 16)
    (h2 : n_salami_diameter = 8) 
    (h3 : n_salami_total = 32) 
    : 
    (32 * (Real.pi * (d_pizza / (2 * n_salami_diameter / 2)) ^ 2)) / (Real.pi * (d_pizza / 2) ^ 2) = 1 / 2 := 
by 
  sorry

end salami_pizza_fraction_l184_184697


namespace charles_whistles_l184_184168

theorem charles_whistles (S : ℕ) (C : ℕ) (h1 : S = 223) (h2 : S = C + 95) : C = 128 :=
by
  sorry

end charles_whistles_l184_184168


namespace papers_left_l184_184459

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end papers_left_l184_184459


namespace ram_actual_distance_from_base_l184_184103

def map_distance_between_mountains : ℝ := 312
def actual_distance_between_mountains : ℝ := 136
def ram_map_distance_from_base : ℝ := 28

theorem ram_actual_distance_from_base :
  ram_map_distance_from_base * (actual_distance_between_mountains / map_distance_between_mountains) = 12.205 :=
by sorry

end ram_actual_distance_from_base_l184_184103


namespace fraction_of_house_painted_l184_184403

theorem fraction_of_house_painted (total_time : ℝ) (paint_time : ℝ) (house : ℝ) (h1 : total_time = 60) (h2 : paint_time = 15) (h3 : house = 1) : 
  (paint_time / total_time) * house = 1 / 4 :=
by
  sorry

end fraction_of_house_painted_l184_184403


namespace inverse_shifted_point_l184_184487

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def inverse_function (f g : ℝ → ℝ) : Prop := ∀ y, f (g y) = y ∧ ∀ x, g (f x) = x

theorem inverse_shifted_point
  (f : ℝ → ℝ)
  (hf_odd : odd_function f)
  (hf_point : f (-1) = 3)
  (g : ℝ → ℝ)
  (hg_inverse : inverse_function f g) :
  g (2 - 5) = 1 :=
by
  sorry

end inverse_shifted_point_l184_184487


namespace empty_set_is_d_l184_184096

open Set

theorem empty_set_is_d : {x : ℝ | x^2 - x + 1 = 0} = ∅ :=
by
  sorry

end empty_set_is_d_l184_184096


namespace interval_intersection_l184_184127

theorem interval_intersection (x : ℝ) :
  (4 * x > 2 ∧ 4 * x < 3) ∧ (5 * x > 2 ∧ 5 * x < 3) ↔ (x > 1/2 ∧ x < 3/5) :=
by
  sorry

end interval_intersection_l184_184127


namespace parallelogram_area_l184_184045

-- Defining the vectors u and z
def u : ℝ × ℝ := (4, -1)
def z : ℝ × ℝ := (9, -3)

-- Computing the area of parallelogram formed by vectors u and z
def area_parallelogram (u z : ℝ × ℝ) : ℝ :=
  abs (u.1 * (z.2 + u.2) - u.2 * (z.1 + u.1))

-- Lean statement asserting that the area of the parallelogram is 3
theorem parallelogram_area : area_parallelogram u z = 3 := by
  sorry

end parallelogram_area_l184_184045


namespace problem_statement_l184_184316

variables (P Q : Prop)

theorem problem_statement (h1 : ¬P) (h2 : ¬(P ∧ Q)) : ¬(P ∨ Q) :=
sorry

end problem_statement_l184_184316


namespace angle_magnification_l184_184093

theorem angle_magnification (α : ℝ) (h : α = 20) : α = 20 := by
  sorry

end angle_magnification_l184_184093


namespace incorrect_propositions_l184_184080

theorem incorrect_propositions :
  ¬ (∀ P : Prop, P → P) ∨
  (¬ (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x > 0)) ∨
  (∀ (R : Type) (f : R → Prop), (∀ r, f r → ∃ r', f r') = ∃ r, f r ∧ ∃ r', f r') ∨
  (∀ (x : ℝ), x ≠ 3 → abs x = 3 → x = 3) :=
by sorry

end incorrect_propositions_l184_184080


namespace area_of_circle_l184_184951

theorem area_of_circle (r : ℝ) (h : r = 3) : 
  (∀ A : ℝ, A = π * r^2) → A = 9 * π :=
by
  intro area_formula
  sorry

end area_of_circle_l184_184951


namespace Tom_total_spend_l184_184864

theorem Tom_total_spend :
  let notebook_price := 2
  let notebook_discount := 0.75
  let notebook_count := 4
  let magazine_price := 5
  let magazine_count := 2
  let pen_price := 1.50
  let pen_discount := 0.75
  let pen_count := 3
  let book_price := 12
  let book_count := 1
  let discount_threshold := 30
  let coupon_discount := 10
  let total_cost :=
    (notebook_count * (notebook_price * notebook_discount)) +
    (magazine_count * magazine_price) +
    (pen_count * (pen_price * pen_discount)) +
    (book_count * book_price)
  let final_cost := if total_cost >= discount_threshold then total_cost - coupon_discount else total_cost
  final_cost = 21.375 :=
by
  sorry

end Tom_total_spend_l184_184864


namespace right_triangle_acute_angle_l184_184941

theorem right_triangle_acute_angle (x : ℝ) 
  (h1 : 5 * x = 90) : x = 18 :=
by sorry

end right_triangle_acute_angle_l184_184941


namespace abs_inequality_solution_bounded_a_b_inequality_l184_184516

theorem abs_inequality_solution (x : ℝ) : (-4 < x ∧ x < 0) ↔ (|x + 1| + |x + 3| < 4) := sorry

theorem bounded_a_b_inequality (a b : ℝ) (h1 : -4 < a) (h2 : a < 0) (h3 : -4 < b) (h4 : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| := sorry

end abs_inequality_solution_bounded_a_b_inequality_l184_184516


namespace vasya_can_win_l184_184667

noncomputable def initial_first : ℝ := 1 / 2009
noncomputable def initial_second : ℝ := 1 / 2008
noncomputable def increment : ℝ := 1 / (2008 * 2009)

theorem vasya_can_win :
  ∃ n : ℕ, ((2009 * n) * increment = 1) ∨ ((2008 * n) * increment = 1) :=
sorry

end vasya_can_win_l184_184667


namespace equilateral_triangle_ratio_is_correct_l184_184735

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l184_184735


namespace rhombus_fourth_vertex_l184_184657

theorem rhombus_fourth_vertex (a b : ℝ) :
  ∃ x y : ℝ, (x, y) = (a - b, a + b) ∧ dist (a, b) (x, y) = dist (-b, a) (x, y) ∧ dist (-b, a) (x, y) = dist (0, 0) (x, y) :=
by
  use (a - b)
  use (a + b)
  sorry

end rhombus_fourth_vertex_l184_184657


namespace div_ad_bc_l184_184107

theorem div_ad_bc (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) : (a - c) ∣ (a * d + b * c) :=
sorry

end div_ad_bc_l184_184107


namespace distance_between_stripes_l184_184853

theorem distance_between_stripes (d₁ d₂ L W : ℝ) (h : ℝ)
  (h₁ : d₁ = 60)  -- distance between parallel curbs
  (h₂ : L = 30)  -- length of the curb between stripes
  (h₃ : d₂ = 80)  -- length of each stripe
  (area_eq : W * L = 1800) -- area of the parallelogram with base L
: h = 22.5 :=
by
  -- This is to assume the equation derived from area calculation
  have area_eq' : d₂ * h = 1800 := by sorry
  -- Solving for h using the derived area equation
  have h_calc : h = 1800 / 80 := by sorry
  -- Simplifying the result
  have h_simplified : h = 22.5 := by sorry
  exact h_simplified

end distance_between_stripes_l184_184853


namespace gcd_in_range_l184_184779

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end gcd_in_range_l184_184779


namespace ratio_pat_to_mark_l184_184293

theorem ratio_pat_to_mark (K P M : ℕ) 
  (h1 : P + K + M = 117) 
  (h2 : P = 2 * K) 
  (h3 : M = K + 65) : 
  P / Nat.gcd P M = 1 ∧ M / Nat.gcd P M = 3 := 
by
  sorry

end ratio_pat_to_mark_l184_184293


namespace linear_function_quadrants_l184_184382

theorem linear_function_quadrants (m : ℝ) (h1 : m - 2 < 0) (h2 : m + 1 > 0) : -1 < m ∧ m < 2 := 
by 
  sorry

end linear_function_quadrants_l184_184382


namespace period2_students_is_8_l184_184920

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end period2_students_is_8_l184_184920


namespace nonneg_triple_inequality_l184_184464

theorem nonneg_triple_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) :=
by
  sorry

end nonneg_triple_inequality_l184_184464


namespace fractional_exponent_equality_l184_184930

theorem fractional_exponent_equality :
  (3 / 4 : ℚ) ^ 2017 * (- ((1:ℚ) + 1 / 3)) ^ 2018 = 4 / 3 :=
by
  sorry

end fractional_exponent_equality_l184_184930


namespace cindy_envelopes_left_l184_184346

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l184_184346


namespace log_expression_l184_184760

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression : 
  log 2 * log 50 + log 25 - log 5 * log 20 = 1 := 
by 
  sorry

end log_expression_l184_184760


namespace artist_paints_33_square_meters_l184_184608

/-
Conditions:
1. The artist has 14 cubes.
2. Each cube has an edge of 1 meter.
3. The cubes are arranged in a pyramid-like structure with three layers.
4. The top layer has 1 cube, the middle layer has 4 cubes, and the bottom layer has 9 cubes.
-/

def exposed_surface_area (num_cubes : Nat) (layer1 : Nat) (layer2 : Nat) (layer3 : Nat) : Nat :=
  let layer1_area := 5 -- Each top layer cube has 5 faces exposed
  let layer2_edge_cubes := 4 -- Count of cubes on the edge in middle layer
  let layer2_area := layer2_edge_cubes * 3 -- Each middle layer edge cube has 3 faces exposed
  let layer3_area := 9 -- Each bottom layer cube has 1 face exposed
  let top_faces := layer1 + layer2 + layer3 -- All top faces exposed
  layer1_area + layer2_area + layer3_area + top_faces

theorem artist_paints_33_square_meters :
  exposed_surface_area 14 1 4 9 = 33 := 
sorry

end artist_paints_33_square_meters_l184_184608


namespace subset_A_inter_B_eq_A_l184_184003

variable {x : ℝ}
def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem subset_A_inter_B_eq_A (k : ℝ) : (A k ∩ B = A k) ↔ (k ≤ 3 / 2) := 
sorry

end subset_A_inter_B_eq_A_l184_184003


namespace sue_initially_borrowed_six_movies_l184_184503

variable (M : ℕ)
variable (initial_books : ℕ := 15)
variable (returned_books : ℕ := 8)
variable (returned_movies_fraction : ℚ := 1/3)
variable (additional_books : ℕ := 9)
variable (total_items : ℕ := 20)

theorem sue_initially_borrowed_six_movies (hM : total_items = initial_books - returned_books + additional_books + (M - returned_movies_fraction * M)) : 
  M = 6 := by
  sorry

end sue_initially_borrowed_six_movies_l184_184503


namespace percentage_cut_third_week_l184_184759

noncomputable def initial_weight : ℝ := 300
noncomputable def first_week_percentage : ℝ := 0.30
noncomputable def second_week_percentage : ℝ := 0.30
noncomputable def final_weight : ℝ := 124.95

theorem percentage_cut_third_week :
  let remaining_after_first_week := initial_weight * (1 - first_week_percentage)
  let remaining_after_second_week := remaining_after_first_week * (1 - second_week_percentage)
  let cut_weight_third_week := remaining_after_second_week - final_weight
  let percentage_cut_third_week := (cut_weight_third_week / remaining_after_second_week) * 100
  percentage_cut_third_week = 15 :=
by
  sorry

end percentage_cut_third_week_l184_184759


namespace monthly_income_of_P_l184_184504

variable (P Q R : ℝ)

theorem monthly_income_of_P (h1 : (P + Q) / 2 = 5050) 
                           (h2 : (Q + R) / 2 = 6250) 
                           (h3 : (P + R) / 2 = 5200) : 
    P = 4000 := 
sorry

end monthly_income_of_P_l184_184504


namespace inequality_proof_l184_184815

theorem inequality_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l184_184815


namespace nuts_in_trail_mix_l184_184304

theorem nuts_in_trail_mix :
  let walnuts := 0.25
  let almonds := 0.25
  walnuts + almonds = 0.50 :=
by
  sorry

end nuts_in_trail_mix_l184_184304


namespace bus_stop_time_l184_184329

/-- 
  We are given:
  speed_ns: speed of bus without stoppages (32 km/hr)
  speed_ws: speed of bus including stoppages (16 km/hr)
  
  We need to prove the bus stops for t = 30 minutes each hour.
-/
theorem bus_stop_time
  (speed_ns speed_ws: ℕ)
  (h_ns: speed_ns = 32)
  (h_ws: speed_ws = 16):
  ∃ t: ℕ, t = 30 := 
sorry

end bus_stop_time_l184_184329


namespace unique_A3_zero_l184_184419

variable {F : Type*} [Field F]

theorem unique_A3_zero (A : Matrix (Fin 2) (Fin 2) F) 
  (h1 : A ^ 4 = 0) 
  (h2 : Matrix.trace A = 0) : 
  A ^ 3 = 0 :=
sorry

end unique_A3_zero_l184_184419


namespace least_n_for_reducible_fraction_l184_184150

theorem least_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, n - 13 = 71 * k) ∧ n = 84 := by
  sorry

end least_n_for_reducible_fraction_l184_184150


namespace triangular_weight_is_60_l184_184457

variable (w_round w_triangular w_rectangular : ℝ)

axiom rectangular_weight : w_rectangular = 90
axiom balance1 : w_round + w_triangular = 3 * w_round
axiom balance2 : 4 * w_round + w_triangular = w_triangular + w_round + w_rectangular

theorem triangular_weight_is_60 :
  w_triangular = 60 :=
by
  sorry

end triangular_weight_is_60_l184_184457


namespace car_mileage_before_modification_l184_184873

theorem car_mileage_before_modification (miles_per_gallon_before : ℝ) 
  (fuel_efficiency_modifier : ℝ := 0.75) (tank_capacity : ℝ := 12) 
  (extra_miles_after_modification : ℝ := 96) :
  (1 / fuel_efficiency_modifier) * miles_per_gallon_before * (tank_capacity - 1) = 24 :=
by
  sorry

end car_mileage_before_modification_l184_184873


namespace part1_part2_l184_184702

-- Part (1)
theorem part1 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hx : x > 0) (hy : y < 0) : x + y = -4 :=
sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hxy : x < y) : x - y = -10 ∨ x - y = -4 :=
sorry

end part1_part2_l184_184702


namespace sum_max_min_eq_four_l184_184490

noncomputable def f (x : ℝ) : ℝ :=
  (|2 * x| + x^3 + 2) / (|x| + 1)

-- Define the maximum value M and minimum value m
noncomputable def M : ℝ := sorry -- The maximum value of the function f(x)
noncomputable def m : ℝ := sorry -- The minimum value of the function f(x)

theorem sum_max_min_eq_four : M + m = 4 := by
  sorry

end sum_max_min_eq_four_l184_184490


namespace mark_additional_inches_l184_184264

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

end mark_additional_inches_l184_184264


namespace sequence_general_term_l184_184479

/-- 
  Define the sequence a_n recursively as:
  a_1 = 2
  a_n = 2 * a_(n-1) - 1

  Prove that the general term of the sequence is:
  a_n = 2^(n-1) + 1
-/
theorem sequence_general_term {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) 
  (n : ℕ) : 
  a n = 2^(n-1) + 1 := by
  sorry

end sequence_general_term_l184_184479


namespace total_chocolate_bars_l184_184471

theorem total_chocolate_bars (n_small_boxes : ℕ) (bars_per_box : ℕ) (total_bars : ℕ) :
  n_small_boxes = 16 → bars_per_box = 25 → total_bars = 16 * 25 → total_bars = 400 :=
by
  intros
  sorry

end total_chocolate_bars_l184_184471


namespace parabola_focus_l184_184157

theorem parabola_focus (y x : ℝ) (h : y^2 = 4 * x) : x = 1 → y = 0 → (1, 0) = (1, 0) :=
by 
  sorry

end parabola_focus_l184_184157


namespace tan_600_eq_sqrt3_l184_184933

theorem tan_600_eq_sqrt3 : (Real.tan (600 * Real.pi / 180)) = Real.sqrt 3 := 
by 
  -- sorry to skip the actual proof steps
  sorry

end tan_600_eq_sqrt3_l184_184933


namespace red_pill_cost_l184_184220

theorem red_pill_cost :
  ∃ (r : ℚ) (b : ℚ), (∀ (d : ℕ), d = 21 → 3 * r - 2 = 39) ∧
                      (1 ≤ d → r = b + 1) ∧
                      (21 * (r + 2 * b) = 819) → 
                      r = 41 / 3 :=
by sorry

end red_pill_cost_l184_184220


namespace tan_sum_trig_identity_l184_184904

variable {α : ℝ}

-- Part (I)
theorem tan_sum (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

-- Part (II)
theorem trig_identity (h : Real.tan α = 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = 3 / 2 :=
by
  sorry

end tan_sum_trig_identity_l184_184904


namespace years_required_l184_184292

def num_stadiums := 30
def avg_cost_per_stadium := 900
def annual_savings := 1500
def total_cost := num_stadiums * avg_cost_per_stadium

theorem years_required : total_cost / annual_savings = 18 :=
by
  sorry

end years_required_l184_184292


namespace train_length_l184_184053

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) (conversion_factor : ℝ) :
  speed_kmh = 54 →
  time_s = 33333333333333336 / 1000000000000000 →
  bridge_length_m = 140 →
  conversion_factor = 1000 / 3600 →
  ∃ (train_length_m : ℝ), 
    speed_kmh * conversion_factor * time_s + bridge_length_m = train_length_m + bridge_length_m :=
by
  intros
  use 360
  sorry

end train_length_l184_184053


namespace fraction_simplifies_correctly_l184_184022

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end fraction_simplifies_correctly_l184_184022


namespace count_perfect_squares_l184_184889

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def E1 : ℕ := 1^3 + 2^3
def E2 : ℕ := 1^3 + 2^3 + 3^3
def E3 : ℕ := 1^3 + 2^3 + 3^3 + 4^3
def E4 : ℕ := 1^3 + 2^3 + 3^3 + 4^3 + 5^3

theorem count_perfect_squares :
  (is_perfect_square E1 → true) ∧
  (is_perfect_square E2 → true) ∧
  (is_perfect_square E3 → true) ∧
  (is_perfect_square E4 → true) →
  (∀ n : ℕ, (n = 4) ↔
    ∃ E1 E2 E3 E4, is_perfect_square E1 ∧ is_perfect_square E2 ∧ is_perfect_square E3 ∧ is_perfect_square E4) :=
by
  sorry

end count_perfect_squares_l184_184889


namespace jean_vs_pauline_cost_l184_184233

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

end jean_vs_pauline_cost_l184_184233


namespace find_value_of_x_l184_184167

theorem find_value_of_x (x : ℕ) (h : (50 + x / 90) * 90 = 4520) : x = 4470 :=
sorry

end find_value_of_x_l184_184167


namespace double_decker_bus_total_capacity_l184_184008

-- Define conditions for the lower floor seating
def lower_floor_left_seats : Nat := 15
def lower_floor_right_seats : Nat := 12
def lower_floor_priority_seats : Nat := 4

-- Each seat on the left and right side of the lower floor holds 2 people
def lower_floor_left_capacity : Nat := lower_floor_left_seats * 2
def lower_floor_right_capacity : Nat := lower_floor_right_seats * 2
def lower_floor_priority_capacity : Nat := lower_floor_priority_seats * 1

-- Define conditions for the upper floor seating
def upper_floor_left_seats : Nat := 20
def upper_floor_right_seats : Nat := 20
def upper_floor_back_capacity : Nat := 15

-- Each seat on the left and right side of the upper floor holds 3 people
def upper_floor_left_capacity : Nat := upper_floor_left_seats * 3
def upper_floor_right_capacity : Nat := upper_floor_right_seats * 3

-- Total capacity of lower and upper floors
def lower_floor_total_capacity : Nat := lower_floor_left_capacity + lower_floor_right_capacity + lower_floor_priority_capacity
def upper_floor_total_capacity : Nat := upper_floor_left_capacity + upper_floor_right_capacity + upper_floor_back_capacity

-- Assert the total capacity
def bus_total_capacity : Nat := lower_floor_total_capacity + upper_floor_total_capacity

-- Prove that the total bus capacity is 193 people
theorem double_decker_bus_total_capacity : bus_total_capacity = 193 := by
  sorry

end double_decker_bus_total_capacity_l184_184008


namespace original_equation_solution_l184_184410

noncomputable def original_equation : Prop :=
  ∃ Y P A K P O C : ℕ,
  (Y = 5) ∧ (P = 2) ∧ (A = 0) ∧ (K = 2) ∧ (P = 4) ∧ (O = 0) ∧ (C = 0) ∧
  (Y.factorial * P.factorial * A.factorial = K * 10000 + P * 1000 + O * 100 + C * 10 + C)

theorem original_equation_solution : original_equation :=
  sorry

end original_equation_solution_l184_184410


namespace coffee_bean_price_l184_184506

theorem coffee_bean_price 
  (x : ℝ)
  (price_second : ℝ) (weight_first weight_second : ℝ)
  (total_weight : ℝ) (price_mixture : ℝ) 
  (value_mixture : ℝ) 
  (h1 : price_second = 12)
  (h2 : weight_first = 25)
  (h3 : weight_second = 25)
  (h4 : total_weight = 100)
  (h5 : price_mixture = 11.25)
  (h6 : value_mixture = total_weight * price_mixture)
  (h7 : weight_first + weight_second = total_weight) :
  25 * x + 25 * 12 = 100 * 11.25 → x = 33 :=
by
  intro h
  sorry

end coffee_bean_price_l184_184506


namespace positive_3_digit_numbers_divisible_by_13_count_l184_184761

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l184_184761


namespace kendra_shirts_needed_l184_184434

def school_shirts_per_week : Nat := 5
def club_shirts_per_week : Nat := 3
def spirit_day_shirt_per_week : Nat := 1
def saturday_shirts_per_week : Nat := 3
def sunday_shirts_per_week : Nat := 3
def family_reunion_shirt_per_month : Nat := 1

def total_shirts_needed_per_week : Nat :=
  school_shirts_per_week + club_shirts_per_week + spirit_day_shirt_per_week +
  saturday_shirts_per_week + sunday_shirts_per_week

def total_shirts_needed_per_four_weeks : Nat :=
  total_shirts_needed_per_week * 4 + family_reunion_shirt_per_month

theorem kendra_shirts_needed : total_shirts_needed_per_four_weeks = 61 := by
  sorry

end kendra_shirts_needed_l184_184434


namespace no_integers_solution_l184_184952

theorem no_integers_solution (k : ℕ) (x y z : ℤ) (hx1 : 0 < x) (hx2 : x < k) (hy1 : 0 < y) (hy2 : y < k) (hz : z > 0) :
  x^k + y^k ≠ z^k :=
sorry

end no_integers_solution_l184_184952


namespace ratio_of_average_speeds_l184_184496

-- Conditions
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4
def distance_ab : ℕ := 600
def distance_ac : ℕ := 360

-- Theorem to prove the ratio of their average speeds
theorem ratio_of_average_speeds : (distance_ab / time_eddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 20 ∧
                                  (distance_ac / time_freddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 9 :=
by
  -- Solution steps go here if performing an actual proof
  sorry

end ratio_of_average_speeds_l184_184496


namespace C_pow_50_l184_184672

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℝ :=
![![3, 1], ![-4, -1]]

theorem C_pow_50 :
  (C ^ 50) = ![![101, 50], ![-200, -99]] :=
by
  sorry

end C_pow_50_l184_184672


namespace minimum_F_l184_184221

noncomputable def F (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + (0.5 * x)

theorem minimum_F : ∃ x : ℝ, x ≥ 0 ∧ F x = 57.5 ∧ ∀ y ≥ 0, F y ≥ F x := by
  use 55
  sorry

end minimum_F_l184_184221


namespace lucy_fish_bought_l184_184350

def fish_bought (fish_original fish_now : ℕ) : ℕ :=
  fish_now - fish_original

theorem lucy_fish_bought : fish_bought 212 492 = 280 :=
by
  sorry

end lucy_fish_bought_l184_184350


namespace Jed_cards_after_4_weeks_l184_184161

theorem Jed_cards_after_4_weeks :
  (∀ n: ℕ, (if n % 2 = 0 then 20 + 4*n - 2*n else 20 + 4*n - 2*(n-1)) = 40) :=
by {
  sorry
}

end Jed_cards_after_4_weeks_l184_184161


namespace fractions_sum_identity_l184_184402

theorem fractions_sum_identity (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / ((b - c) ^ 2) + b / ((c - a) ^ 2) + c / ((a - b) ^ 2) = 0 :=
by
  sorry

end fractions_sum_identity_l184_184402


namespace theta_terminal_side_l184_184569

theorem theta_terminal_side (alpha : ℝ) (theta : ℝ) (h1 : alpha = 1560) (h2 : -360 < theta ∧ theta < 360) :
    (theta = 120 ∨ theta = -240) := by
  -- The proof steps would go here
  sorry

end theta_terminal_side_l184_184569


namespace xy_sum_possible_values_l184_184944

theorem xy_sum_possible_values (x y : ℕ) (h1 : x < 20) (h2 : y < 20) (h3 : 0 < x) (h4 : 0 < y) (h5 : x + y + x * y = 95) :
  x + y = 18 ∨ x + y = 20 :=
by {
  sorry
}

end xy_sum_possible_values_l184_184944


namespace volume_of_cone_l184_184422

theorem volume_of_cone (l : ℝ) (A : ℝ) (r : ℝ) (h : ℝ) : 
  l = 10 → A = 60 * Real.pi → (r = 6) → (h = Real.sqrt (10^2 - 6^2)) → 
  (1 / 3 * Real.pi * r^2 * h) = 96 * Real.pi :=
by
  intros
  -- here the proof would be written
  sorry

end volume_of_cone_l184_184422


namespace sum_numbers_eq_432_l184_184924

theorem sum_numbers_eq_432 (n : ℕ) (h : (n * (n + 1)) / 2 = 432) : n = 28 :=
sorry

end sum_numbers_eq_432_l184_184924


namespace combined_work_time_l184_184898

noncomputable def work_time_first_worker : ℤ := 5
noncomputable def work_time_second_worker : ℤ := 4

theorem combined_work_time :
  (1 / (1 / work_time_first_worker + 1 / work_time_second_worker)) = 20 / 9 :=
by
  unfold work_time_first_worker work_time_second_worker
  -- The detailed reasoning and computation would go here
  sorry

end combined_work_time_l184_184898


namespace trig_identity_l184_184430

open Real

theorem trig_identity (α : ℝ) (h_tan : tan α = 2) (h_quad : 0 < α ∧ α < π / 2) :
  sin (2 * α) + cos α = (4 + sqrt 5) / 5 :=
sorry

end trig_identity_l184_184430


namespace quadratic_inequality_solution_l184_184910

theorem quadratic_inequality_solution :
  {x : ℝ | (x^2 - 50 * x + 576) ≤ 16} = {x : ℝ | 20 ≤ x ∧ x ≤ 28} :=
sorry

end quadratic_inequality_solution_l184_184910


namespace van_capacity_l184_184583

theorem van_capacity (s a v : ℕ) (h1 : s = 2) (h2 : a = 6) (h3 : v = 2) : (s + a) / v = 4 := by
  sorry

end van_capacity_l184_184583


namespace circular_pond_area_l184_184243

theorem circular_pond_area (AB CD : ℝ) (D_is_midpoint : Prop) (hAB : AB = 20) (hCD : CD = 12)
  (hD_midpoint : D_is_midpoint ∧ D_is_midpoint = (AB / 2 = 10)) :
  ∃ (A : ℝ), A = 244 * Real.pi :=
by
  sorry

end circular_pond_area_l184_184243


namespace find_a_l184_184615

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l184_184615


namespace cost_to_fill_half_of_can_B_l184_184509

theorem cost_to_fill_half_of_can_B (r h : ℝ) (cost_fill_V : ℝ) (cost_fill_V_eq : cost_fill_V = 16)
  (V_radius_eq : 2 * r = radius_of_can_V)
  (V_height_eq: h / 2 = height_of_can_V) :
  cost_fill_half_of_can_B = 4 :=
by
  sorry

end cost_to_fill_half_of_can_B_l184_184509


namespace example_problem_l184_184964

theorem example_problem (a b : ℕ) : a = 1 → a * (a + b) + 1 ∣ (a + b) * (b + 1) - 1 :=
by
  sorry

end example_problem_l184_184964


namespace exam_full_marks_l184_184195

variables {A B C D F : ℝ}

theorem exam_full_marks
  (hA : A = 0.90 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.80 * D)
  (hA_val : A = 360)
  (hD : D = 0.80 * F) 
  : F = 500 :=
sorry

end exam_full_marks_l184_184195


namespace Leela_Hotel_all_three_reunions_l184_184055

theorem Leela_Hotel_all_three_reunions
  (A B C : Finset ℕ)
  (hA : A.card = 80)
  (hB : B.card = 90)
  (hC : C.card = 70)
  (hAB : (A ∩ B).card = 30)
  (hAC : (A ∩ C).card = 25)
  (hBC : (B ∩ C).card = 20)
  (hABC : ((A ∪ B ∪ C)).card = 150) : 
  (A ∩ B ∩ C).card = 15 :=
by
  sorry

end Leela_Hotel_all_three_reunions_l184_184055


namespace intersection_of_A_and_B_l184_184054

def setA (x : ℝ) : Prop := x^2 < 4
def setB : Set ℝ := {0, 1}

theorem intersection_of_A_and_B :
  {x : ℝ | setA x} ∩ setB = setB := by
  sorry

end intersection_of_A_and_B_l184_184054


namespace arrange_desc_l184_184477

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (35 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (35 * Real.pi / 180)
noncomputable def d : ℝ := Real.log 5

theorem arrange_desc : d > c ∧ c > b ∧ b > a := by
  sorry

end arrange_desc_l184_184477


namespace find_nm_l184_184131

theorem find_nm (h : 62^2 + 122^2 = 18728) : 
  ∃ (n m : ℕ), (n = 92 ∧ m = 30) ∨ (n = 30 ∧ m = 92) ∧ n^2 + m^2 = 9364 := 
by 
  sorry

end find_nm_l184_184131


namespace set_intersection_problem_l184_184899

def set_product (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3}
def C : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

theorem set_intersection_problem :
  (set_product A B) ∩ (set_product B C) = {2, 6} :=
by
  sorry

end set_intersection_problem_l184_184899


namespace cosine_of_angle_between_vectors_l184_184290

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

end cosine_of_angle_between_vectors_l184_184290


namespace distance_between_foci_of_hyperbola_l184_184548

theorem distance_between_foci_of_hyperbola :
  (∀ x y : ℝ, (y = 2 * x + 3) ∨ (y = -2 * x + 1)) →
  ∀ p : ℝ × ℝ, (p = (2, 1)) →
  ∃ d : ℝ, d = 2 * Real.sqrt 30 :=
by
  sorry

end distance_between_foci_of_hyperbola_l184_184548


namespace det_B_squared_minus_3IB_l184_184948

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![3, 1]]
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem det_B_squared_minus_3IB :
  det (B * B - 3 * I * B) = 100 := by
  sorry

end det_B_squared_minus_3IB_l184_184948


namespace quadrangular_pyramid_edge_length_l184_184993

theorem quadrangular_pyramid_edge_length :
  ∃ e : ℝ, 8 * e = 14.8 ∧ e = 1.85 :=
  sorry

end quadrangular_pyramid_edge_length_l184_184993


namespace sum_n_k_eq_eight_l184_184574

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem to prove that n + k = 8 given the conditions
theorem sum_n_k_eq_eight {n k : ℕ} 
  (h1 : binom n k * 3 = binom n (k + 1))
  (h2 : binom n (k + 1) * 5 = binom n (k + 2) * 3) : n + k = 8 := by
  sorry

end sum_n_k_eq_eight_l184_184574


namespace condition_sufficient_but_not_necessary_l184_184623

variables (p q : Prop)

theorem condition_sufficient_but_not_necessary (hpq : ∀ q, (¬p → ¬q)) (hpns : ¬ (¬p → ¬q ↔ p → q)) : (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end condition_sufficient_but_not_necessary_l184_184623


namespace find_p_from_parabola_and_distance_l184_184616

theorem find_p_from_parabola_and_distance 
  (p : ℝ) (hp : p > 0) 
  (M : ℝ × ℝ) (hM : M = (8 / p, 4))
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (hMF : dist M F = 4) : 
  p = 4 :=
sorry

end find_p_from_parabola_and_distance_l184_184616


namespace project_profit_starts_from_4th_year_l184_184478

def initial_investment : ℝ := 144
def maintenance_cost (n : ℕ) : ℝ := 4 * n^2 + 40 * n
def annual_income : ℝ := 100

def net_profit (n : ℕ) : ℝ := 
  annual_income * n - maintenance_cost n - initial_investment

theorem project_profit_starts_from_4th_year :
  ∀ n : ℕ, 3 < n ∧ n < 12 → net_profit n > 0 :=
by
  intros n hn
  sorry

end project_profit_starts_from_4th_year_l184_184478


namespace girls_left_to_play_kho_kho_l184_184164

theorem girls_left_to_play_kho_kho (B G x : ℕ) 
  (h_eq : B = G)
  (h_twice : B = 2 * (G - x))
  (h_total : B + G = 32) :
  x = 8 :=
by sorry

end girls_left_to_play_kho_kho_l184_184164


namespace stream_speed_l184_184208

variables (v : ℝ) (swimming_speed : ℝ) (ratio : ℝ)

theorem stream_speed (hs : swimming_speed = 4.5) (hr : ratio = 2) (h : (swimming_speed - v) / (swimming_speed + v) = 1 / ratio) :
  v = 1.5 :=
sorry

end stream_speed_l184_184208


namespace percent_increase_in_sales_l184_184469

theorem percent_increase_in_sales :
  let new := 416
  let old := 320
  (new - old) / old * 100 = 30 := by
  sorry

end percent_increase_in_sales_l184_184469


namespace length_of_second_train_l184_184494

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (crossing_time : ℝ)
  (total_distance : ℝ)
  (relative_speed_mps : ℝ)
  (length_second_train : ℝ) :
  length_first_train = 130 ∧ 
  speed_first_train = 60 ∧
  speed_second_train = 40 ∧
  crossing_time = 10.439164866810657 ∧
  relative_speed_mps = (speed_first_train + speed_second_train) * (5/18) ∧
  total_distance = relative_speed_mps * crossing_time ∧
  length_first_train + length_second_train = total_distance →
  length_second_train = 160 :=
by
  sorry

end length_of_second_train_l184_184494


namespace simplify_rationalize_l184_184384

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l184_184384


namespace total_hours_watching_tv_and_playing_games_l184_184451

-- Defining the conditions provided in the problem
def hours_watching_tv_saturday : ℕ := 6
def hours_watching_tv_sunday : ℕ := 3
def hours_watching_tv_tuesday : ℕ := 2
def hours_watching_tv_thursday : ℕ := 4

def hours_playing_games_monday : ℕ := 3
def hours_playing_games_wednesday : ℕ := 5
def hours_playing_games_friday : ℕ := 1

-- The proof statement
theorem total_hours_watching_tv_and_playing_games :
  hours_watching_tv_saturday + hours_watching_tv_sunday + hours_watching_tv_tuesday + hours_watching_tv_thursday
  + hours_playing_games_monday + hours_playing_games_wednesday + hours_playing_games_friday = 24 := 
by
  sorry

end total_hours_watching_tv_and_playing_games_l184_184451


namespace tangents_parallel_l184_184750

-- Definitions based on the conditions in part A
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_line (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

def secant_intersection (c1 c2 : Circle) (A : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  sorry

-- Main theorem statement
theorem tangents_parallel 
  (c1 c2 : Circle) (A B C : ℝ × ℝ) 
  (h1 : c1.center ≠ c2.center) 
  (h2 : dist c1.center c2.center = c1.radius + c2.radius) 
  (h3 : (B, C) = secant_intersection c1 c2 A) 
  (h4 : tangent_line c1 B ≠ tangent_line c2 C) :
  tangent_line c1 B = tangent_line c2 C :=
sorry

end tangents_parallel_l184_184750


namespace sufficient_but_not_necessary_perpendicular_l184_184731

theorem sufficient_but_not_necessary_perpendicular (a : ℝ) :
  (∃ a' : ℝ, a' = -1 ∧ (a' = -1 → (0 : ℝ) ≠ 3 * a' - 1)) ∨
  (∃ a' : ℝ, a' ≠ -1 ∧ (a' ≠ -1 → (0 : ℝ) ≠ 3 * a' - 1)) →
  (3 * a' - 1) * (a' - 3) = -1 := sorry

end sufficient_but_not_necessary_perpendicular_l184_184731


namespace problem_statement_l184_184845

noncomputable def term_with_largest_binomial_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ :=
-8064

noncomputable def term_with_largest_absolute_value_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ × ℕ :=
(-15360, 8)

theorem problem_statement (M N P : ℕ) (h_sum : M + N - P = 2016) (n : ℕ) :
  ((term_with_largest_binomial_coefficient M N P h_sum n = -8064) ∧ 
   (term_with_largest_absolute_value_coefficient M N P h_sum n = (-15360, 8))) :=
by {
  -- proof goes here
  sorry
}

end problem_statement_l184_184845


namespace no_integer_solutions_l184_184581

theorem no_integer_solutions : ¬∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := 
by
  sorry

end no_integer_solutions_l184_184581


namespace part1_part2_l184_184852

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x^2 - 3 * x + 2 = 0 }

theorem part1 (a : ℝ) : (A a = ∅) ↔ (a > 9/8) := sorry

theorem part2 (a : ℝ) : 
  (∃ x, A a = {x}) ↔ 
  (a = 0 ∧ A a = {2 / 3})
  ∨ (a = 9 / 8 ∧ A a = {4 / 3}) := sorry

end part1_part2_l184_184852


namespace tree_growth_l184_184185

theorem tree_growth (x : ℝ) : 4*x + 4*2*x + 4*2 + 4*3 = 32 → x = 1 :=
by
  intro h
  sorry

end tree_growth_l184_184185


namespace total_cost_of_fencing_l184_184778

def diameter : ℝ := 28
def cost_per_meter : ℝ := 1.50
def pi_approx : ℝ := 3.14159

noncomputable def circumference : ℝ := pi_approx * diameter
noncomputable def total_cost : ℝ := circumference * cost_per_meter

theorem total_cost_of_fencing : total_cost = 131.94 :=
by
  sorry

end total_cost_of_fencing_l184_184778


namespace angle_relation_l184_184456

theorem angle_relation (R : ℝ) (hR : R > 0) (d : ℝ) (hd : d > R) 
  (α β : ℝ) : β = 3 * α :=
sorry

end angle_relation_l184_184456


namespace slope_tangent_line_l184_184549

variable {f : ℝ → ℝ}

-- Assumption: f is differentiable
def differentiable_at (f : ℝ → ℝ) (x : ℝ) := ∃ f', ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |(f (x + h) - f x) / h - f'| < ε

-- Hypothesis: limit condition
axiom limit_condition : (∀ x, differentiable_at f (1 - x)) → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε)

-- Theorem: the slope of the tangent line to the curve y = f(x) at (1, f(1)) is -2
theorem slope_tangent_line : differentiable_at f 1 → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε) → deriv f 1 = -2 :=
by
    intro h_diff h_lim
    sorry

end slope_tangent_line_l184_184549


namespace arithmetic_seq_problem_l184_184248

theorem arithmetic_seq_problem (S : ℕ → ℤ) (n : ℕ) (h1 : S 6 = 36) 
                               (h2 : S n = 324) (h3 : S (n - 6) = 144) (hn : n > 6) : 
  n = 18 := 
sorry

end arithmetic_seq_problem_l184_184248


namespace power_mul_eq_l184_184133

variable (a : ℝ)

theorem power_mul_eq :
  (-a)^2 * a^4 = a^6 :=
by sorry

end power_mul_eq_l184_184133


namespace five_times_number_equals_hundred_l184_184696

theorem five_times_number_equals_hundred (x : ℝ) (h : 5 * x = 100) : x = 20 :=
sorry

end five_times_number_equals_hundred_l184_184696


namespace simplify_expression_l184_184745

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end simplify_expression_l184_184745


namespace sin_double_angle_l184_184489

theorem sin_double_angle (A : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : Real.cos A = 3 / 5) :
  Real.sin (2 * A) = 24 / 25 := 
by
  sorry

end sin_double_angle_l184_184489


namespace min_value_problem_l184_184792

noncomputable def minValueOfExpression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) : ℝ :=
  (x + 2 * y) * (y + 2 * z) * (x * z + 1)

theorem min_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  minValueOfExpression x y z hx hy hz hxyz = 16 :=
  sorry

end min_value_problem_l184_184792


namespace evaluate_expression_l184_184907

theorem evaluate_expression (x b : ℝ) (h : x = b + 4) : 2 * x - b + 5 = b + 13 := by
  sorry

end evaluate_expression_l184_184907


namespace fraction_of_4_is_8_l184_184754

theorem fraction_of_4_is_8 (fraction : ℝ) (h : fraction * 4 = 8) : fraction = 8 := 
sorry

end fraction_of_4_is_8_l184_184754


namespace walk_two_dogs_for_7_minutes_l184_184238

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

end walk_two_dogs_for_7_minutes_l184_184238


namespace reciprocal_check_C_l184_184283

theorem reciprocal_check_C : 0.1 * 10 = 1 := 
by 
  sorry

end reciprocal_check_C_l184_184283


namespace temp_pot_C_to_F_l184_184685

-- Definitions
def boiling_point_C : ℕ := 100
def boiling_point_F : ℕ := 212
def melting_point_C : ℕ := 0
def melting_point_F : ℕ := 32
def temp_pot_C : ℕ := 55
def celsius_to_fahrenheit (c : ℕ) : ℕ := (c * 9 / 5) + 32

-- Theorem to be proved
theorem temp_pot_C_to_F : celsius_to_fahrenheit temp_pot_C = 131 := by
  sorry

end temp_pot_C_to_F_l184_184685


namespace find_sum_of_cubes_l184_184601

-- Define the distinct real numbers p, q, and r
variables {p q r : ℝ}

-- Conditions
-- Distinctness condition
axiom h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p

-- Given condition
axiom h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r

-- Proof goal
theorem find_sum_of_cubes : p^3 + q^3 + r^3 = -21 :=
sorry

end find_sum_of_cubes_l184_184601


namespace casey_saving_l184_184016

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l184_184016


namespace fraction_to_decimal_l184_184453

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l184_184453


namespace common_root_l184_184387

def f (x : ℝ) : ℝ := x^4 - x^3 - 22 * x^2 + 16 * x + 96
def g (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 10

theorem common_root :
  f (-2) = 0 ∧ g (-2) = 0 := by
  sorry

end common_root_l184_184387


namespace all_three_pets_l184_184870

-- Definitions of the given conditions
def total_students : ℕ := 40
def dog_owners : ℕ := 20
def cat_owners : ℕ := 13
def other_pet_owners : ℕ := 8
def no_pets : ℕ := 7

-- Definitions from Venn diagram
def dogs_only : ℕ := 12
def cats_only : ℕ := 3
def other_pets_only : ℕ := 2

-- Intersection variables
variables (a b c d : ℕ)

-- Translated problem
theorem all_three_pets :
  dogs_only + cats_only + other_pets_only + a + b + c + d = total_students - no_pets ∧
  dogs_only + a + c + d = dog_owners ∧
  cats_only + a + b + d = cat_owners ∧
  other_pets_only + b + c + d = other_pet_owners ∧
  d = 2 :=
sorry

end all_three_pets_l184_184870


namespace simplify_and_evaluate_div_fraction_l184_184591

theorem simplify_and_evaluate_div_fraction (a : ℤ) (h : a = -3) : 
  (a - 2) / (1 + 2 * a + a^2) / (a - 3 * a / (a + 1)) = 1 / 6 := by
  sorry

end simplify_and_evaluate_div_fraction_l184_184591


namespace sum_of_roots_of_polynomial_l184_184747

theorem sum_of_roots_of_polynomial (a b c : ℝ) (h : 3*a^3 - 7*a^2 + 6*a = 0) : 
    (∀ x, 3*x^2 - 7*x + 6 = 0 → x = a ∨ x = b ∨ x = c) →
    (∀ (x : ℝ), (x = a ∨ x = b ∨ x = c → 3*x^3 - 7*x^2 + 6*x = 0)) → 
    a + b + c = 7 / 3 :=
sorry

end sum_of_roots_of_polynomial_l184_184747


namespace false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l184_184709

-- Proposition A
theorem false_proposition_A (a b c : ℝ) (hac : a > b) (hca : b > 0) : ac * c^2 = b * c^2 :=
  sorry

-- Proposition B
theorem false_proposition_B (a b : ℝ) (hab : a < b) : (1/a) < (1/b) :=
  sorry

-- Proposition C
theorem true_proposition_C (a b : ℝ) (hab : a > b) (hba : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
  sorry

-- Proposition D
theorem true_proposition_D (a b : ℝ) (hba : a > |b|) : a^2 > b^2 :=
  sorry

end false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l184_184709


namespace sum_of_a_and_b_l184_184391

variables {a b m : ℝ}

theorem sum_of_a_and_b (h1 : a^2 + a * b = 16 + m) (h2 : b^2 + a * b = 9 - m) : a + b = 5 ∨ a + b = -5 :=
by sorry

end sum_of_a_and_b_l184_184391


namespace distance_focus_parabola_to_line_l184_184946

theorem distance_focus_parabola_to_line :
  let focus : ℝ × ℝ := (1, 0)
  let distance (p : ℝ × ℝ) (A B C : ℝ) : ℝ := |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)
  distance focus 1 (-Real.sqrt 3) 0 = 1 / 2 :=
by
  sorry

end distance_focus_parabola_to_line_l184_184946


namespace min_distance_from_curve_to_line_l184_184913

open Real

-- Definitions and conditions
def curve_eq (x y: ℝ) : Prop := (x^2 - y - 2 * log (sqrt x) = 0)
def line_eq (x y: ℝ) : Prop := (4 * x + 4 * y + 1 = 0)

-- The main statement
theorem min_distance_from_curve_to_line :
  ∃ (x y : ℝ), curve_eq x y ∧ y = x^2 - 2 * log (sqrt x) ∧ line_eq x y ∧ y = -x - 1/4 ∧ 
               |4 * (1/2) + 4 * ((1/4) + log 2) + 1| / sqrt 32 = sqrt 2 / 2 * (1 + log 2) :=
by
  -- We skip the proof as requested, using sorry:
  sorry

end min_distance_from_curve_to_line_l184_184913


namespace debate_team_has_11_boys_l184_184613

def debate_team_boys_count (num_groups : Nat) (members_per_group : Nat) (num_girls : Nat) : Nat :=
  let total_members := num_groups * members_per_group
  total_members - num_girls

theorem debate_team_has_11_boys :
  debate_team_boys_count 8 7 45 = 11 :=
by
  sorry

end debate_team_has_11_boys_l184_184613


namespace range_of_a_no_solution_inequality_l184_184943

theorem range_of_a_no_solution_inequality (a : ℝ) :
  (∀ x : ℝ, x + 2 > 3 → x < a) ↔ a ≤ 1 :=
by {
  sorry
}

end range_of_a_no_solution_inequality_l184_184943


namespace exists_rank_with_profit_2016_l184_184526

theorem exists_rank_with_profit_2016 : ∃ n : ℕ, n * (n + 1) / 2 = 2016 :=
by 
  sorry

end exists_rank_with_profit_2016_l184_184526


namespace sum_of_solutions_of_quadratic_l184_184129

theorem sum_of_solutions_of_quadratic :
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  roots_sum = 3 / 2 :=
by
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  have h1 : roots_sum = 3 / 2 := by sorry
  exact h1

end sum_of_solutions_of_quadratic_l184_184129


namespace boxes_needed_to_sell_l184_184379

theorem boxes_needed_to_sell (total_bars : ℕ) (bars_per_box : ℕ) (target_boxes : ℕ) (h₁ : total_bars = 710) (h₂ : bars_per_box = 5) : target_boxes = 142 :=
by
  sorry

end boxes_needed_to_sell_l184_184379


namespace sum_of_base_radii_l184_184755

theorem sum_of_base_radii (R : ℝ) (hR : R = 5) (a b c : ℝ) 
  (h_ratios : a = 1 ∧ b = 2 ∧ c = 3) 
  (r1 r2 r3 : ℝ) 
  (h_r1 : r1 = (a / (a + b + c)) * R)
  (h_r2 : r2 = (b / (a + b + c)) * R)
  (h_r3 : r3 = (c / (a + b + c)) * R) : 
  r1 + r2 + r3 = 5 := 
by
  subst hR
  simp [*, ←add_assoc, add_comm]
  sorry

end sum_of_base_radii_l184_184755


namespace min_value_alpha_beta_gamma_l184_184012

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end min_value_alpha_beta_gamma_l184_184012


namespace initial_number_of_peanuts_l184_184945

theorem initial_number_of_peanuts (x : ℕ) (h : x + 2 = 6) : x = 4 :=
sorry

end initial_number_of_peanuts_l184_184945


namespace Lisa_weight_l184_184242

theorem Lisa_weight : ∃ l a : ℝ, a + l = 240 ∧ l - a = l / 3 ∧ l = 144 :=
by
  sorry

end Lisa_weight_l184_184242


namespace max_mn_l184_184932

theorem max_mn (m n : ℝ) (h : m + n = 1) : mn ≤ 1 / 4 :=
by
  sorry

end max_mn_l184_184932


namespace five_integers_sum_to_first_set_impossible_second_set_sum_l184_184205

theorem five_integers_sum_to_first_set :
  ∃ (a b c d e : ℤ), 
    (a + b = 0) ∧ (a + c = 2) ∧ (b + c = 4) ∧ (a + d = 4) ∧ (b + d = 6) ∧
    (a + e = 8) ∧ (b + e = 9) ∧ (c + d = 11) ∧ (c + e = 13) ∧ (d + e = 15) ∧ 
    (a + b + c + d + e = 18) := 
sorry

theorem impossible_second_set_sum : 
  ¬∃ (a b c d e : ℤ), 
    (a + b = 12) ∧ (a + c = 13) ∧ (a + d = 14) ∧ (a + e = 15) ∧ (b + c = 16) ∧
    (b + d = 16) ∧ (b + e = 17) ∧ (c + d = 17) ∧ (c + e = 18) ∧ (d + e = 20) ∧
    (a + b + c + d + e = 39) :=
sorry

end five_integers_sum_to_first_set_impossible_second_set_sum_l184_184205


namespace math_problem_l184_184083

theorem math_problem :
  3^(5+2) + 4^(1+3) = 39196 ∧
  2^(9+2) - 3^(4+1) = 3661 ∧
  1^(8+6) + 3^(2+3) = 250 ∧
  6^(5+4) - 4^(5+1) = 409977 → 
  5^(7+2) - 2^(5+3) = 1952869 :=
by
  sorry

end math_problem_l184_184083


namespace original_salary_l184_184470

theorem original_salary (x : ℝ)
  (h1 : x * 1.10 * 0.95 = 3135) : x = 3000 :=
by
  sorry

end original_salary_l184_184470


namespace find_sum_of_distinct_numbers_l184_184337

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l184_184337


namespace sum_base7_l184_184163

def base7_to_base10 (n : ℕ) : ℕ := 
  -- Function to convert base 7 to base 10 (implementation not shown)
  sorry

def base10_to_base7 (n : ℕ) : ℕ :=
  -- Function to convert base 10 to base 7 (implementation not shown)
  sorry

theorem sum_base7 (a b : ℕ) (ha : a = base7_to_base10 12) (hb : b = base7_to_base10 245) :
  base10_to_base7 (a + b) = 260 :=
sorry

end sum_base7_l184_184163


namespace rational_numbers_product_power_l184_184546

theorem rational_numbers_product_power (a b : ℚ) (h : |a - 2| + (2 * b + 1)^2 = 0) :
  (a * b)^2013 = -1 :=
sorry

end rational_numbers_product_power_l184_184546


namespace symmetric_points_origin_l184_184609

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end symmetric_points_origin_l184_184609


namespace polynomial_sum_l184_184057

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l184_184057


namespace problem1_problem2_l184_184050

-- Definitions based on conditions in the problem
def seq_sum (a : ℕ) (n : ℕ) : ℕ := a * 2^n - 1
def a1 (a : ℕ) : ℕ := seq_sum a 1
def a4 (a : ℕ) : ℕ := seq_sum a 4 - seq_sum a 3

-- Problem statement 1
theorem problem1 (a : ℕ) (h : a = 3) : a1 a = 5 ∧ a4 a = 24 := by 
  sorry

-- Geometric sequence conditions
def is_geometric (a_n : ℕ → ℕ) : Prop :=
  ∃ q ≠ 1, ∀ n, a_n (n + 1) = q * a_n n

-- Definitions for the geometric sequence part
def a_n (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2 * a - 1
  else if n = 2 then 2 * a
  else if n = 3 then 4 * a
  else 0 -- Simplifying for the first few terms only

-- Problem statement 2
theorem problem2 : (∃ a : ℕ, is_geometric (a_n a)) → ∃ a : ℕ, a = 1 := by
  sorry

end problem1_problem2_l184_184050


namespace solve_for_a_l184_184605

theorem solve_for_a {a x : ℝ} (H : (x - 2) * (a * x^2 - x + 1) = a * x^3 + (-1 - 2 * a) * x^2 + 3 * x - 2 ∧ (-1 - 2 * a) = 0) : a = -1/2 := sorry

end solve_for_a_l184_184605


namespace instrument_accuracy_confidence_l184_184650

noncomputable def instrument_accuracy (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ) : ℝ × ℝ :=
  let lower := s * (1 - q)
  let upper := s * (1 + q)
  (lower, upper)

theorem instrument_accuracy_confidence :
  ∀ (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ),
    n = 12 →
    s = 0.6 →
    gamma = 0.99 →
    q = 0.9 →
    0.06 < (instrument_accuracy n s gamma q).fst ∧
    (instrument_accuracy n s gamma q).snd < 1.14 :=
by
  intros n s gamma q h_n h_s h_gamma h_q
  -- proof would go here
  sorry

end instrument_accuracy_confidence_l184_184650


namespace impossible_distance_l184_184669

noncomputable def radius_O1 : ℝ := 2
noncomputable def radius_O2 : ℝ := 5

theorem impossible_distance :
  ∀ (d : ℝ), ¬ (radius_O1 ≠ radius_O2 → ¬ (d < abs (radius_O2 - radius_O1) ∨ d > radius_O2 + radius_O1) → d = 5) :=
by
  sorry

end impossible_distance_l184_184669


namespace B_and_C_complete_task_l184_184514

noncomputable def A_work_rate : ℚ := 1 / 12
noncomputable def B_work_rate : ℚ := 1.2 * A_work_rate
noncomputable def C_work_rate : ℚ := 2 * A_work_rate

theorem B_and_C_complete_task (B_work_rate C_work_rate : ℚ) 
    (A_work_rate : ℚ := 1 / 12) :
  B_work_rate = 1.2 * A_work_rate →
  C_work_rate = 2 * A_work_rate →
  (B_work_rate + C_work_rate) = 4 / 15 :=
by intros; sorry

end B_and_C_complete_task_l184_184514


namespace single_intersection_l184_184826

theorem single_intersection (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y^2 = x ∧ y + 1 = k * x) ↔ (k = 0 ∨ k = -1 / 4) :=
sorry

end single_intersection_l184_184826


namespace point_inside_circle_l184_184281

theorem point_inside_circle (r OP : ℝ) (h₁ : r = 3) (h₂ : OP = 2) : OP < r :=
by
  sorry

end point_inside_circle_l184_184281


namespace farm_field_proof_l184_184147

section FarmField

variables 
  (planned_rate daily_rate : ℕ) -- planned_rate is 260 hectares/day, daily_rate is 85 hectares/day 
  (extra_days remaining_hectares : ℕ) -- extra_days is 2, remaining_hectares is 40
  (max_hours_per_day : ℕ) -- max_hours_per_day is 12

-- Definitions for soils
variables
  (A_percent B_percent C_percent : ℚ) (A_hours B_hours C_hours : ℕ)
  -- A_percent is 0.4, B_percent is 0.3, C_percent is 0.3
  -- A_hours is 4, B_hours is 6, C_hours is 3

-- Given conditions
axiom planned_rate_eq : planned_rate = 260
axiom daily_rate_eq : daily_rate = 85
axiom extra_days_eq : extra_days = 2
axiom remaining_hectares_eq : remaining_hectares = 40
axiom max_hours_per_day_eq : max_hours_per_day = 12

axiom A_percent_eq : A_percent = 0.4
axiom B_percent_eq : B_percent = 0.3
axiom C_percent_eq : C_percent = 0.3

axiom A_hours_eq : A_hours = 4
axiom B_hours_eq : B_hours = 6
axiom C_hours_eq : C_hours = 3

-- Theorem stating the problem
theorem farm_field_proof :
  ∃ (total_area initial_days : ℕ),
    total_area = 340 ∧ initial_days = 2 :=
by
  sorry

end FarmField

end farm_field_proof_l184_184147


namespace patio_length_four_times_width_l184_184340

theorem patio_length_four_times_width (w l : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 :=
by
  sorry

end patio_length_four_times_width_l184_184340


namespace marbles_leftover_l184_184383

theorem marbles_leftover (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : (r + p) % 8 = 4 :=
by
  sorry

end marbles_leftover_l184_184383


namespace captain_age_l184_184671

theorem captain_age (C : ℕ) (h1 : ∀ W : ℕ, W = C + 3) 
                    (h2 : 21 * 11 = 231) 
                    (h3 : 21 - 1 = 20) 
                    (h4 : 20 * 9 = 180)
                    (h5 : 231 - 180 = 51) :
  C = 24 :=
by
  sorry

end captain_age_l184_184671


namespace stratified_sampling_l184_184492

/-- Given a batch of 98 water heaters with 56 from Factory A and 42 from Factory B,
    and a stratified sample of 14 units is to be drawn, prove that the number 
    of water heaters sampled from Factory A is 8 and from Factory B is 6. --/

theorem stratified_sampling (batch_size A B sample_size : ℕ) 
  (h_batch : batch_size = 98) 
  (h_fact_a : A = 56) 
  (h_fact_b : B = 42) 
  (h_sample : sample_size = 14) : 
  (A * sample_size / batch_size = 8) ∧ (B * sample_size / batch_size = 6) := 
  by
    sorry

end stratified_sampling_l184_184492


namespace line_intersects_circle_l184_184235

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 9 = 0
def line_eq (m x y : ℝ) : Prop := m*x + y + m - 2 = 0

-- Theorem statement based on question and correct answer
theorem line_intersects_circle (m : ℝ) :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

end line_intersects_circle_l184_184235


namespace solve_system_of_equations_l184_184660

theorem solve_system_of_equations (x y : ℝ) (hx: x > 0) (hy: y > 0) :
  x * y = 500 ∧ x ^ (Real.log y / Real.log 10) = 25 → (x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100) := by
  sorry

end solve_system_of_equations_l184_184660


namespace find_a8_l184_184713

noncomputable def a (n : ℕ) : ℤ := sorry

noncomputable def b (n : ℕ) : ℤ := a (n + 1) - a n

theorem find_a8 :
  (a 1 = 3) ∧
  (∀ n : ℕ, b n = b 1 + n * 2) ∧
  (b 3 = -2) ∧
  (b 10 = 12) →
  a 8 = 3 :=
by sorry

end find_a8_l184_184713


namespace pool_filling_times_l184_184991

theorem pool_filling_times:
  ∃ (x y z u : ℕ),
    (1/x + 1/y = 1/70) ∧
    (1/x + 1/z = 1/84) ∧
    (1/y + 1/z = 1/140) ∧
    (1/u = 1/x + 1/y + 1/z) ∧
    (x = 105) ∧
    (y = 210) ∧
    (z = 420) ∧
    (u = 60) := 
  sorry

end pool_filling_times_l184_184991


namespace students_in_band_l184_184396

theorem students_in_band (total_students : ℕ) (band_percentage : ℚ) (h_total_students : total_students = 840) (h_band_percentage : band_percentage = 0.2) : ∃ band_students : ℕ, band_students = 168 ∧ band_students = band_percentage * total_students := 
sorry

end students_in_band_l184_184396


namespace smallest_four_digit_multiple_of_18_l184_184409

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l184_184409


namespace hyperbola_eccentricity_range_l184_184604

theorem hyperbola_eccentricity_range 
(a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
(hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
(parabola_eq : ∀ y x, y^2 = 8 * a * x)
(right_vertex : A = (a, 0))
(focus : F = (2 * a, 0))
(P : ℝ × ℝ)
(asymptote_eq : P = (x0, b / a * x0))
(perpendicular_condition : (x0 ^ 2 - (3 * a - b^2 / a^2) * x0 + 2 * a^2 = 0))
(hyperbola_properties: c^2 = a^2 + b^2) :
1 < c / a ∧ c / a <= 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_range_l184_184604


namespace sequence_a4_l184_184716

theorem sequence_a4 :
  (∀ n : ℕ, n > 0 → ∀ (a : ℕ → ℝ),
    (a 1 = 1) →
    (∀ n > 0, a (n + 1) = (1 / 2) * a n + 1 / (2 ^ n)) →
    a 4 = 1 / 2) :=
by
  sorry

end sequence_a4_l184_184716


namespace probability_correct_l184_184728

structure SockDrawSetup where
  total_socks : ℕ
  color_pairs : ℕ
  socks_per_color : ℕ
  draw_size : ℕ

noncomputable def probability_one_pair (S : SockDrawSetup) : ℚ :=
  let total_combinations := Nat.choose S.total_socks S.draw_size
  let favorable_combinations := (Nat.choose S.color_pairs 3) * (Nat.choose 3 1) * 2 * 2
  favorable_combinations / total_combinations

theorem probability_correct (S : SockDrawSetup) (h1 : S.total_socks = 12) (h2 : S.color_pairs = 6) (h3 : S.socks_per_color = 2) (h4 : S.draw_size = 6) :
  probability_one_pair S = 20 / 77 :=
by
  apply sorry

end probability_correct_l184_184728


namespace present_age_of_son_l184_184979

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 22) (h2 : F + 2 = 2 * (S + 2)) : S = 20 :=
by
  sorry

end present_age_of_son_l184_184979


namespace find_primes_satisfying_equation_l184_184229

theorem find_primes_satisfying_equation :
  {p : ℕ | p.Prime ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p} = {2, 3, 7} :=
by
  sorry

end find_primes_satisfying_equation_l184_184229


namespace fraction_result_l184_184811

theorem fraction_result (a b c : ℝ) (h1 : a / 2 = b / 3) (h2 : b / 3 = c / 5) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  (a + b) / (c - a) = 5 / 3 :=
by
  sorry

end fraction_result_l184_184811


namespace parabola_translation_l184_184297

theorem parabola_translation :
  ∀ x : ℝ, (x^2 + 3) = ((x + 1)^2 + 3) :=
by
  skip -- proof is not needed; this is just the statement according to the instruction
  sorry

end parabola_translation_l184_184297


namespace perpendicular_bisector_eq_l184_184125

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- Prove that the perpendicular bisector of line segment AB has the equation 3x - y - 9 = 0
theorem perpendicular_bisector_eq :
  (∀ x y : ℝ, C1 x y → C2 x y → 3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_eq_l184_184125


namespace calc_expression_l184_184206

theorem calc_expression :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 :=
sorry

end calc_expression_l184_184206


namespace correct_subtraction_l184_184861

theorem correct_subtraction (x : ℕ) (h : x - 42 = 50) : x - 24 = 68 :=
  sorry

end correct_subtraction_l184_184861


namespace calculate_sum_l184_184160

open Real

theorem calculate_sum :
  (-1: ℝ) ^ 2023 + (1/2) ^ (-2: ℝ) + 3 * tan (pi / 6) - (3 - pi) ^ 0 + |sqrt 3 - 2| = 4 :=
by
  sorry

end calculate_sum_l184_184160


namespace factor_expression_l184_184082

theorem factor_expression (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by
  sorry

end factor_expression_l184_184082


namespace arithmetic_sequence_S9_l184_184820

-- Define the sum of an arithmetic sequence: S_n
def arithmetic_sequence_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := (n * (2 * a (0) + (n - 1) * d (0))) / 2

-- Conditions
variable (a d : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (h1 : S_n 3 = 9)
variable (h2 : S_n 6 = 27)

-- Question: Prove that S_9 = 54
theorem arithmetic_sequence_S9 : S_n 9 = 54 := by
    sorry

end arithmetic_sequence_S9_l184_184820


namespace planes_parallel_l184_184828

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end planes_parallel_l184_184828


namespace line_parallel_condition_l184_184408

theorem line_parallel_condition (a : ℝ) : (a = 2) ↔ (∀ x y : ℝ, (ax + 2 * y = 0 → x + y ≠ 1)) :=
by
  sorry

end line_parallel_condition_l184_184408


namespace total_amount_received_l184_184603

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end total_amount_received_l184_184603


namespace volunteer_group_selection_l184_184839

theorem volunteer_group_selection :
  let M := 4  -- Number of male teachers
  let F := 5  -- Number of female teachers
  let G := 3  -- Total number of teachers in the group
  -- Calculate the number of ways to select 2 male teachers and 1 female teacher
  let ways1 := (Nat.choose M 2) * (Nat.choose F 1)
  -- Calculate the number of ways to select 1 male teacher and 2 female teachers
  let ways2 := (Nat.choose M 1) * (Nat.choose F 2)
  -- The total number of ways to form the group
  ways1 + ways2 = 70 := by sorry

end volunteer_group_selection_l184_184839


namespace determine_F_l184_184092

theorem determine_F (A H S M F : ℕ) (ha : 0 < A) (hh : 0 < H) (hs : 0 < S) (hm : 0 < M) (hf : 0 < F):
  (A * x + H * y = z) →
  (S * x + M * y = z) →
  (F * x = z) →
  (H > A) →
  (A ≠ H) →
  (S ≠ M) →
  (F ≠ A) →
  (F ≠ H) →
  (F ≠ S) →
  (F ≠ M) →
  x = z / F →
  y = ((F - A) / H * z) / z →
  F = (A * F - S * H) / (M - H) := sorry

end determine_F_l184_184092


namespace polynomial_factorization_proof_l184_184508

noncomputable def factorizable_binary_quadratic (m : ℚ) : Prop :=
  ∃ (a b : ℚ), (3*a - 5*b = 17) ∧ (a*b = -4) ∧ (m = 2*a + 3*b)

theorem polynomial_factorization_proof :
  ∀ (m : ℚ), factorizable_binary_quadratic m ↔ (m = 5 ∨ m = -58 / 15) :=
by
  sorry

end polynomial_factorization_proof_l184_184508


namespace boxes_needed_to_pack_all_muffins_l184_184996

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l184_184996


namespace find_c_l184_184711

theorem find_c (a b c : ℝ) (h_line : 4 * a - 3 * b + c = 0) 
  (h_min : (a - 1)^2 + (b - 1)^2 = 4) : c = 9 ∨ c = -11 := 
    sorry

end find_c_l184_184711


namespace ellen_painted_17_lilies_l184_184724

theorem ellen_painted_17_lilies :
  (∃ n : ℕ, n * 5 + 10 * 7 + 6 * 3 + 20 * 2 = 213) → 
    ∃ n : ℕ, n = 17 := 
by sorry

end ellen_painted_17_lilies_l184_184724


namespace min_students_with_blue_eyes_and_backpack_l184_184659

theorem min_students_with_blue_eyes_and_backpack :
  ∀ (students : Finset ℕ), 
  (∀ s, s ∈ students → s = 1) →
  ∃ A B : Finset ℕ, 
    A.card = 18 ∧ B.card = 24 ∧ students.card = 35 ∧ 
    (A ∩ B).card ≥ 7 :=
by
  sorry

end min_students_with_blue_eyes_and_backpack_l184_184659


namespace complete_the_square_transforms_l184_184891

theorem complete_the_square_transforms (x : ℝ) :
  (x^2 + 8 * x + 7 = 0) → ((x + 4) ^ 2 = 9) :=
by
  intro h
  have step1 : x^2 + 8 * x = -7 := by sorry
  have step2 : x^2 + 8 * x + 16 = -7 + 16 := by sorry
  have step3 : (x + 4) ^ 2 = 9 := by sorry
  exact step3

end complete_the_square_transforms_l184_184891


namespace smallest_n_for_multiple_of_5_l184_184399

theorem smallest_n_for_multiple_of_5 (x y : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 5]) (h2 : y - 2 ≡ 0 [ZMOD 5]) :
  ∃ n : ℕ, n > 0 ∧ x^2 + x * y + y^2 + n ≡ 0 [ZMOD 5] ∧ n = 1 := 
sorry

end smallest_n_for_multiple_of_5_l184_184399


namespace country_x_income_l184_184670

theorem country_x_income (I : ℝ) (h1 : I > 40000) (_ : 0.15 * 40000 + 0.20 * (I - 40000) = 8000) : I = 50000 :=
sorry

end country_x_income_l184_184670


namespace factorial_division_l184_184178

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end factorial_division_l184_184178


namespace find_length_of_train_l184_184222

def speed_kmh : Real := 60
def time_to_cross_bridge : Real := 26.997840172786177
def length_of_bridge : Real := 340

noncomputable def speed_ms : Real := speed_kmh * (1000 / 3600)
noncomputable def total_distance : Real := speed_ms * time_to_cross_bridge
noncomputable def length_of_train : Real := total_distance - length_of_bridge

theorem find_length_of_train :
  length_of_train = 109.9640028797695 := 
sorry

end find_length_of_train_l184_184222


namespace shirt_price_is_correct_l184_184268

noncomputable def sweater_price (T : ℝ) : ℝ := T + 7.43 

def discounted_price (S : ℝ) : ℝ := S * 0.90

theorem shirt_price_is_correct :
  ∃ (T S : ℝ), T + discounted_price S = 80.34 ∧ T = S - 7.43 ∧ T = 38.76 :=
by
  sorry

end shirt_price_is_correct_l184_184268


namespace discount_percentage_l184_184818

theorem discount_percentage (wholesale_price retail_price selling_price profit: ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : profit = 0.20 * wholesale_price)
  (h4 : selling_price = wholesale_price + profit):
  (retail_price - selling_price) / retail_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l184_184818


namespace quadratic_inequality_solution_l184_184443

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ -8/3 < k ∧ k < 6 :=
by
  sorry

end quadratic_inequality_solution_l184_184443


namespace locus_of_projection_l184_184090

theorem locus_of_projection {a b c : ℝ} (h : (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2) :
  ∀ (x y : ℝ), (x, y) ∈ ({P : ℝ × ℝ | ∃ a b : ℝ, P = ((a * b^2) / (a^2 + b^2), (a^2 * b) / (a^2 + b^2)) ∧ (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2}) → 
    x^2 + y^2 = c^2 := 
sorry

end locus_of_projection_l184_184090


namespace number_of_players_is_correct_l184_184855

-- Defining the problem conditions
def wristband_cost : ℕ := 6
def jersey_cost : ℕ := wristband_cost + 7
def wristbands_per_player : ℕ := 4
def jerseys_per_player : ℕ := 2
def total_expenditure : ℕ := 3774

-- Calculating cost per player and stating the proof problem
def cost_per_player : ℕ := wristbands_per_player * wristband_cost +
                           jerseys_per_player * jersey_cost

def number_of_players : ℕ := total_expenditure / cost_per_player

-- The final proof statement to show that number_of_players is 75
theorem number_of_players_is_correct : number_of_players = 75 :=
by sorry

end number_of_players_is_correct_l184_184855


namespace interval_contains_root_l184_184026

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem interval_contains_root :
  f (-1) < 0 → 
  f 0 < 0 → 
  f 1 < 0 → 
  f 2 > 0 → 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intro h1 h2 h3 h4
  sorry

end interval_contains_root_l184_184026


namespace pieces_1994_impossible_pieces_1997_possible_l184_184901

def P (n : ℕ) : ℕ := 1 + 4 * n

theorem pieces_1994_impossible : ∀ n : ℕ, P n ≠ 1994 := 
by sorry

theorem pieces_1997_possible : ∃ n : ℕ, P n = 1997 := 
by sorry

end pieces_1994_impossible_pieces_1997_possible_l184_184901


namespace least_number_added_1789_l184_184491

def least_number_added_to_divisible (n d : ℕ) : ℕ := d - (n % d)

theorem least_number_added_1789 :
  least_number_added_to_divisible 1789 (Nat.lcm (Nat.lcm 5 6) (Nat.lcm 4 3)) = 11 :=
by
  -- Step definitions
  have lcm_5_6 := Nat.lcm 5 6
  have lcm_4_3 := Nat.lcm 4 3
  have lcm_total := Nat.lcm lcm_5_6 lcm_4_3
  -- Computation of the final result
  have remainder := 1789 % lcm_total
  have required_add := lcm_total - remainder
  -- Conclusion based on the computed values
  sorry

end least_number_added_1789_l184_184491


namespace total_arrangements_l184_184798

def count_arrangements : Nat :=
  let male_positions := 3
  let female_positions := 3
  let male_arrangements := Nat.factorial male_positions
  let female_arrangements := Nat.factorial (female_positions - 1)
  male_arrangements * female_arrangements / (male_positions - female_positions + 1)

theorem total_arrangements : count_arrangements = 36 := by
  sorry

end total_arrangements_l184_184798


namespace walter_exceptional_days_l184_184995

theorem walter_exceptional_days :
  ∃ (w b : ℕ), 
  b + w = 10 ∧ 
  3 * b + 5 * w = 36 ∧ 
  w = 3 :=
by
  sorry

end walter_exceptional_days_l184_184995


namespace impossible_to_form_16_unique_remainders_with_3_digits_l184_184285

theorem impossible_to_form_16_unique_remainders_with_3_digits :
  ¬∃ (digits : Finset ℕ) (num_fun : Fin 16 → ℕ), digits.card = 3 ∧ 
  ∀ i j : Fin 16, i ≠ j → num_fun i % 16 ≠ num_fun j % 16 ∧ 
  ∀ n : ℕ, n ∈ (digits : Set ℕ) → 100 ≤ num_fun i ∧ num_fun i < 1000 :=
sorry

end impossible_to_form_16_unique_remainders_with_3_digits_l184_184285


namespace eliminate_denominator_l184_184294

theorem eliminate_denominator (x : ℝ) : 6 - (x - 2) / 2 = x → 12 - x + 2 = 2 * x :=
by
  intro h
  sorry

end eliminate_denominator_l184_184294


namespace probability_of_drawing_two_red_shoes_l184_184073

/-- Given there are 7 red shoes and 3 green shoes, 
    and a total of 10 shoes, if two shoes are drawn randomly,
    prove that the probability of drawing both shoes as red is 7/15. -/
theorem probability_of_drawing_two_red_shoes :
  let total_shoes := 10
  let red_shoes := 7
  let green_shoes := 3
  let total_ways := Nat.choose total_shoes 2
  let red_ways := Nat.choose red_shoes 2
  (1 : ℚ) * red_ways / total_ways = 7 / 15  := by
  sorry

end probability_of_drawing_two_red_shoes_l184_184073


namespace sum_y_coordinates_of_intersection_with_y_axis_l184_184813

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-4, 5)
def radius : ℝ := 9

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + center.1)^2 + (y - center.2)^2 = radius^2

theorem sum_y_coordinates_of_intersection_with_y_axis : 
  ∃ y1 y2 : ℝ, circle_eq 0 y1 ∧ circle_eq 0 y2 ∧ y1 + y2 = 10 :=
by
  sorry

end sum_y_coordinates_of_intersection_with_y_axis_l184_184813


namespace intersection_A_B_l184_184911

def A := {x : ℝ | (x - 1) * (x - 4) < 0}
def B := {x : ℝ | x <= 2}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x <= 2} :=
sorry

end intersection_A_B_l184_184911


namespace problem_to_prove_l184_184395

theorem problem_to_prove
  (a b c : ℝ)
  (h1 : a + b + c = -3)
  (h2 : a * b + b * c + c * a = -10)
  (h3 : a * b * c = -5) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 70 :=
by
  sorry

end problem_to_prove_l184_184395


namespace chord_lengths_equal_l184_184256

theorem chord_lengths_equal (D E F : ℝ) (hcond_1 : D^2 ≠ E^2) (hcond_2 : E^2 > 4 * F) :
  ∀ x y, (x^2 + y^2 + D * x + E * y + F = 0) → 
  (abs x = abs y) :=
by
  sorry

end chord_lengths_equal_l184_184256


namespace coeff_sum_zero_l184_184219

theorem coeff_sum_zero (a₀ a₁ a₂ a₃ a₄ : ℝ) (h : ∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) :
  a₁ + a₂ + a₃ + a₄ = 0 :=
by
  sorry

end coeff_sum_zero_l184_184219


namespace interest_difference_20_years_l184_184856

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

theorem interest_difference_20_years :
  compound_interest 15000 0.06 20 - simple_interest 15000 0.08 20 = 9107 :=
by
  sorry

end interest_difference_20_years_l184_184856


namespace coin_tails_probability_l184_184785

theorem coin_tails_probability (p : ℝ) (h : p = 0.5) (n : ℕ) (h_n : n = 3) :
  ∃ k : ℕ, k ≤ n ∧ (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end coin_tails_probability_l184_184785


namespace sum_of_three_ints_product_5_4_l184_184842

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end sum_of_three_ints_product_5_4_l184_184842


namespace find_rate_percent_l184_184296

-- Define the conditions based on the problem statement
def principal : ℝ := 800
def simpleInterest : ℝ := 160
def time : ℝ := 5

-- Create the statement to prove the rate percent
theorem find_rate_percent : ∃ (rate : ℝ), simpleInterest = (principal * rate * time) / 100 := sorry

end find_rate_percent_l184_184296


namespace hex_A08_to_decimal_l184_184768

noncomputable def hex_A := 10
noncomputable def hex_A08_base_10 : ℕ :=
  (hex_A * 16^2) + (0 * 16^1) + (8 * 16^0)

theorem hex_A08_to_decimal :
  hex_A08_base_10 = 2568 :=
by
  sorry

end hex_A08_to_decimal_l184_184768


namespace hyperbola_real_axis_length_l184_184512

theorem hyperbola_real_axis_length : 
  (∃ (x y : ℝ), (x^2 / 2) - (y^2 / 4) = 1) → real_axis_length = 2 * Real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end hyperbola_real_axis_length_l184_184512


namespace intersection_points_zero_l184_184980

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem intersection_points_zero
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_gp : geometric_sequence a b c)
  (h_ac_pos : a * c > 0) :
  ∃ x : ℝ, quadratic_function a b c x = 0 → false :=
by
  -- Proof to be completed
  sorry

end intersection_points_zero_l184_184980


namespace part_I_part_II_l184_184140

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem part_I (x : ℝ) : (f x > 5) ↔ (x < -3 ∨ x > 2) :=
  sorry

theorem part_II (a : ℝ) : (∀ x, f x < a ↔ false) ↔ (a ≤ 3) :=
  sorry

end part_I_part_II_l184_184140


namespace temperature_problem_l184_184153

theorem temperature_problem (N : ℤ) (M L : ℤ) :
  M = L + N →
  (M - 10) - (L + 6) = 4 ∨ (M - 10) - (L + 6) = -4 →
  (N - 16 = 4 ∨ 16 - N = 4) →
  ((N = 20 ∨ N = 12) → 20 * 12 = 240) :=
by
   sorry

end temperature_problem_l184_184153


namespace consecutive_odd_numbers_l184_184360

theorem consecutive_odd_numbers (a b c d e : ℤ) (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h4 : e = a + 8) (h5 : a + c = 146) : e = 79 := 
by
  sorry

end consecutive_odd_numbers_l184_184360


namespace percentage_increase_in_spending_l184_184618

variables (P Q : ℝ)
-- Conditions
def price_increase (P : ℝ) := 1.25 * P
def quantity_decrease (Q : ℝ) := 0.88 * Q

-- Mathemtically equivalent proof problem in Lean:
theorem percentage_increase_in_spending (P Q : ℝ) : 
  (price_increase P) * (quantity_decrease Q) / (P * Q) = 1.10 :=
by
  sorry

end percentage_increase_in_spending_l184_184618


namespace average_cost_parking_l184_184925

theorem average_cost_parking :
  let cost_first_2_hours := 12.00
  let cost_per_additional_hour := 1.75
  let total_hours := 9
  let total_cost := cost_first_2_hours + cost_per_additional_hour * (total_hours - 2)
  let average_cost_per_hour := total_cost / total_hours
  average_cost_per_hour = 2.69 :=
by
  sorry

end average_cost_parking_l184_184925


namespace sum_abs_arithmetic_sequence_l184_184182

variable (n : ℕ)

def S_n (n : ℕ) : ℚ :=
  - ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n

def T_n (n : ℕ) : ℚ :=
  if n ≤ 34 then
    -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
  else
    ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502

theorem sum_abs_arithmetic_sequence :
  T_n n = (if n ≤ 34 then -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
           else ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502) :=
by sorry

end sum_abs_arithmetic_sequence_l184_184182


namespace students_not_making_cut_l184_184726

theorem students_not_making_cut :
  let girls := 39
  let boys := 4
  let called_back := 26
  let total := girls + boys
  total - called_back = 17 :=
by
  -- add the proof here
  sorry

end students_not_making_cut_l184_184726


namespace ways_to_divide_friends_l184_184957

theorem ways_to_divide_friends : (4 ^ 8 = 65536) := by
  sorry

end ways_to_divide_friends_l184_184957


namespace find_x_l184_184307

theorem find_x (x y z : ℕ) 
  (h1 : x + y = 74) 
  (h2 : (x + y) + y + z = 164) 
  (h3 : z - y = 16) : 
  x = 37 :=
sorry

end find_x_l184_184307


namespace volunteer_assignment_correct_l184_184218

def volunteerAssignment : ℕ := 5
def pavilions : ℕ := 4

def numberOfWays (volunteers pavilions : ℕ) : ℕ := 72 -- This is based on the provided correct answer.

theorem volunteer_assignment_correct : 
  numberOfWays volunteerAssignment pavilions = 72 := 
by
  sorry

end volunteer_assignment_correct_l184_184218


namespace linda_savings_l184_184302

theorem linda_savings (S : ℝ) 
  (h1 : ∃ f : ℝ, f = 0.9 * 1/2 * S) -- She spent half of her savings on furniture with a 10% discount
  (h2 : ∃ t : ℝ, t = 1/2 * S * 1.05) -- The rest of her savings, spent on TV, had a 5% sales tax applied
  (h3 : 1/2 * S * 1.05 = 300) -- The total cost of the TV after tax was $300
  : S = 571.42 := 
sorry

end linda_savings_l184_184302


namespace abc_min_value_l184_184447

open Real

theorem abc_min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_sum : a + b + c = 1) (h_bound : a ≤ b ∧ b ≤ c ∧ c ≤ 3 * a) :
  3 * a * a * (1 - 4 * a) = (9/343) := 
sorry

end abc_min_value_l184_184447


namespace maximize_annual_profit_l184_184159

theorem maximize_annual_profit : 
  ∃ n : ℕ, n ≠ 0 ∧ (∀ m : ℕ, m ≠ 0 → (110 * n - (n * n + n) - 90) / n ≥ (110 * m - (m * m + m) - 90) / m) ↔ n = 5 := 
by 
  -- Proof steps would go here
  sorry

end maximize_annual_profit_l184_184159


namespace barrels_of_pitch_needed_l184_184102

-- Define the basic properties and conditions
def total_length_road := 16
def truckloads_per_mile := 3
def bags_of_gravel_per_truckload := 2
def gravel_to_pitch_ratio := 5
def miles_paved_first_day := 4
def miles_paved_second_day := 2 * miles_paved_first_day - 1
def miles_already_paved := miles_paved_first_day + miles_paved_second_day
def remaining_miles := total_length_road - miles_already_paved
def total_truckloads := truckloads_per_mile * remaining_miles
def total_bags_of_gravel := bags_of_gravel_per_truckload * total_truckloads
def barrels_of_pitch := total_bags_of_gravel / gravel_to_pitch_ratio

-- State the theorem to prove the number of barrels of pitch needed
theorem barrels_of_pitch_needed :
    barrels_of_pitch = 6 :=
by
    sorry

end barrels_of_pitch_needed_l184_184102


namespace students_interested_in_both_l184_184653

theorem students_interested_in_both (total_students interested_in_sports interested_in_entertainment not_interested interested_in_both : ℕ)
  (h_total_students : total_students = 1400)
  (h_interested_in_sports : interested_in_sports = 1250)
  (h_interested_in_entertainment : interested_in_entertainment = 952)
  (h_not_interested : not_interested = 60)
  (h_equation : not_interested + interested_in_both + (interested_in_sports - interested_in_both) + (interested_in_entertainment - interested_in_both) = total_students) :
  interested_in_both = 862 :=
by
  sorry

end students_interested_in_both_l184_184653


namespace solve_fraction_equation_l184_184446

theorem solve_fraction_equation (x : ℚ) (h : x ≠ -1) : 
  (x / (x + 1) = 2 * x / (3 * x + 3) - 1) → x = -3 / 4 :=
by
  sorry

end solve_fraction_equation_l184_184446


namespace part1_part2_l184_184114

variables {a_n b_n : ℕ → ℤ} {k m : ℕ}

-- Part 1: Arithmetic Sequence
axiom a2_eq_3 : a_n 2 = 3
axiom S5_eq_25 : (5 * (2 * (a_n 1 + 2 * (a_n 1 + 1)) / 2)) = 25

-- Part 2: Geometric Sequence
axiom b1_eq_1 : b_n 1 = 1
axiom q_eq_3 : ∀ n, b_n n = 3^(n-1)

noncomputable def arithmetic_seq (n : ℕ) : ℤ :=
  2 * n - 1

theorem part1 : (a_n 2 + a_n 4) / 2 = 5 :=
  sorry

theorem part2 (k : ℕ) (hk : 0 < k) : ∃ m, b_n k = arithmetic_seq m ∧ m = (3^(k-1) + 1) / 2 :=
  sorry

end part1_part2_l184_184114


namespace logan_average_speed_l184_184465

theorem logan_average_speed 
  (tamika_hours : ℕ)
  (tamika_speed : ℕ)
  (logan_hours : ℕ)
  (tamika_distance : ℕ)
  (logan_distance : ℕ)
  (distance_diff : ℕ)
  (diff_condition : tamika_distance = logan_distance + distance_diff) :
  tamika_hours = 8 →
  tamika_speed = 45 →
  logan_hours = 5 →
  tamika_distance = tamika_speed * tamika_hours →
  distance_diff = 85 →
  logan_distance / logan_hours = 55 :=
by
  sorry

end logan_average_speed_l184_184465


namespace power_expression_evaluation_l184_184265

theorem power_expression_evaluation :
  (1 / 2) ^ 2016 * (-2) ^ 2017 * (-1) ^ 2017 = 2 := 
by
  sorry

end power_expression_evaluation_l184_184265


namespace linear_function_details_l184_184535

variables (x y : ℝ)

noncomputable def linear_function (k b : ℝ) := k * x + b

def passes_through (k b x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = linear_function k b x1 ∧ y2 = linear_function k b x2

def point_on_graph (k b x3 y3 : ℝ) : Prop :=
  y3 = linear_function k b x3

theorem linear_function_details :
  ∃ k b : ℝ, passes_through k b 3 5 (-4) (-9) ∧ point_on_graph k b (-1) (-3) :=
by
  -- to be proved
  sorry

end linear_function_details_l184_184535


namespace students_in_class_l184_184644

theorem students_in_class (n : ℕ) (h1 : (n : ℝ) * 100 = (n * 100 + 60 - 10)) 
  (h2 : (n : ℝ) * 98 = ((n : ℝ) * 100 - 50)) : n = 25 :=
sorry

end students_in_class_l184_184644


namespace number_of_children_l184_184091

-- Define the number of adults and their ticket price
def num_adults := 9
def adult_ticket_price := 11

-- Define the children's ticket price and the total cost difference
def child_ticket_price := 7
def cost_difference := 50

-- Define the total cost for adult tickets
def total_adult_cost := num_adults * adult_ticket_price

-- Given the conditions, prove that the number of children is 7
theorem number_of_children : ∃ c : ℕ, total_adult_cost = c * child_ticket_price + cost_difference ∧ c = 7 :=
by
  sorry

end number_of_children_l184_184091


namespace gcd_polynomial_multiple_l184_184086

theorem gcd_polynomial_multiple (b : ℤ) (h : b % 2373 = 0) : Int.gcd (b^2 + 13 * b + 40) (b + 5) = 5 :=
by
  sorry

end gcd_polynomial_multiple_l184_184086


namespace inequality_solution_l184_184619

theorem inequality_solution (x: ℝ) (h1: x ≠ -1) (h2: x ≠ 0) :
  (x-2)/(x+1) + (x-3)/(3*x) ≥ 2 ↔ x ∈ Set.Iic (-3) ∪ Set.Icc (-1) (-1/2) :=
by
  sorry

end inequality_solution_l184_184619


namespace son_age_is_18_l184_184207

theorem son_age_is_18
  (S F : ℕ)
  (h1 : F = S + 20)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 18 :=
by sorry

end son_age_is_18_l184_184207


namespace initial_volume_of_mixture_l184_184468

theorem initial_volume_of_mixture (V : ℝ) :
  let V_new := V + 8
  let initial_water := 0.20 * V
  let new_water := initial_water + 8
  let new_mixture := V_new
  new_water = 0.25 * new_mixture →
  V = 120 :=
by
  intro h
  sorry

end initial_volume_of_mixture_l184_184468


namespace new_line_length_l184_184949

/-- Eli drew a line that was 1.5 meters long and then erased 37.5 centimeters of it.
    We need to prove that the length of the line now is 112.5 centimeters. -/
theorem new_line_length (initial_length_m : ℝ) (erased_length_cm : ℝ) 
    (h1 : initial_length_m = 1.5) (h2 : erased_length_cm = 37.5) :
    initial_length_m * 100 - erased_length_cm = 112.5 :=
by
  sorry

end new_line_length_l184_184949


namespace cricket_average_l184_184108

theorem cricket_average (A : ℝ) (h : 20 * A + 120 = 21 * (A + 4)) : A = 36 :=
by sorry

end cricket_average_l184_184108


namespace alice_speed_exceeds_l184_184289

theorem alice_speed_exceeds (distance : ℕ) (v_bob : ℕ) (time_diff : ℕ) (v_alice : ℕ)
  (h_distance : distance = 220)
  (h_v_bob : v_bob = 40)
  (h_time_diff : time_diff = 1/2) : 
  v_alice > 44 := 
sorry

end alice_speed_exceeds_l184_184289


namespace smallest_positive_value_of_expression_l184_184228

theorem smallest_positive_value_of_expression :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^3 + b^3 + c^3 - 3 * a * b * c = 4) :=
by
  sorry

end smallest_positive_value_of_expression_l184_184228


namespace largest_value_fraction_l184_184000

theorem largest_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  ∃ z, z = (x^2 / (2 * y)) ∧ z ≤ 5 :=
by
  sorry

end largest_value_fraction_l184_184000


namespace bus_trip_distance_l184_184306

variable (D : ℝ) (S : ℝ := 50)

theorem bus_trip_distance :
  (D / (S + 5) = D / S - 1) → D = 550 := by
  sorry

end bus_trip_distance_l184_184306


namespace number_of_zeros_l184_184347

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 3

theorem number_of_zeros (b : ℝ) : 
  ∃ x₁ x₂ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ x₁ ≠ x₂ := by
  sorry

end number_of_zeros_l184_184347


namespace factorization_correct_l184_184732

-- Define the expression
def expression (a b : ℝ) : ℝ := 3 * a^2 - 3 * b^2

-- Define the factorized form of the expression
def factorized (a b : ℝ) : ℝ := 3 * (a + b) * (a - b)

-- The main statement we need to prove
theorem factorization_correct (a b : ℝ) : expression a b = factorized a b :=
by 
  sorry -- Proof to be filled in

end factorization_correct_l184_184732


namespace sums_remainders_equal_l184_184691

-- Definition and conditions
variables (A A' D S S' s s' : ℕ) 
variables (h1 : A > A') 
variables (h2 : A % D = S) 
variables (h3 : A' % D = S') 
variables (h4 : (A + A') % D = s) 
variables (h5 : (S + S') % D = s')

-- Proof statement
theorem sums_remainders_equal : s = s' := 
  sorry

end sums_remainders_equal_l184_184691


namespace pipes_fill_cistern_together_in_15_minutes_l184_184566

-- Define the problem's conditions in Lean
def PipeA_rate := (1 / 2) / 15
def PipeB_rate := (1 / 3) / 10

-- Define the combined rate
def combined_rate := PipeA_rate + PipeB_rate

-- Define the time to fill the cistern by both pipes working together
def time_to_fill_cistern := 1 / combined_rate

-- State the theorem to prove
theorem pipes_fill_cistern_together_in_15_minutes :
  time_to_fill_cistern = 15 := by
  sorry

end pipes_fill_cistern_together_in_15_minutes_l184_184566


namespace perpendicular_lines_condition_l184_184744

theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2) ↔ (∃ m1 m2 : ℝ, (m1 = -1/(4 : ℝ)) ∧ (m2 = (4 : ℝ)) ∧ (m1 * m2 = -1)) :=
by sorry

end perpendicular_lines_condition_l184_184744


namespace problem_a_l184_184309

theorem problem_a : (1038^2 % 1000) ≠ 4 := by
  sorry

end problem_a_l184_184309


namespace absent_children_on_teachers_day_l184_184858

theorem absent_children_on_teachers_day (A : ℕ) (h1 : ∀ n : ℕ, n = 190)
(h2 : ∀ s : ℕ, s = 38) (h3 : ∀ extra : ℕ, extra = 14) :
  (190 - A) * 38 = 190 * 24 → A = 70 :=
by
  sorry

end absent_children_on_teachers_day_l184_184858


namespace regular_polygon_exterior_angle_l184_184241

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end regular_polygon_exterior_angle_l184_184241


namespace jamie_catches_bus_probability_l184_184310

noncomputable def probability_jamie_catches_bus : ℝ :=
  let total_area := 120 * 120
  let overlap_area := 20 * 100
  overlap_area / total_area

theorem jamie_catches_bus_probability :
  probability_jamie_catches_bus = (5 / 36) :=
by
  sorry

end jamie_catches_bus_probability_l184_184310


namespace polynomial_division_result_q_neg1_r_1_sum_l184_184968

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 5 * x^3 - 4 * x^2 + 2 * x + 1
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * x - 3
noncomputable def q (x : ℝ) : ℝ := 3 * x^2 + x
noncomputable def r (x : ℝ) : ℝ := 7 * x + 4

theorem polynomial_division_result : f (-1) = q (-1) * d (-1) + r (-1)
  ∧ f 1 = q 1 * d 1 + r 1 :=
by sorry

theorem q_neg1_r_1_sum : (q (-1) + r 1) = 13 :=
by sorry

end polynomial_division_result_q_neg1_r_1_sum_l184_184968


namespace range_of_a_for_quad_ineq_false_l184_184288

variable (a : ℝ)

def quad_ineq_holds : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0

theorem range_of_a_for_quad_ineq_false :
  ¬ quad_ineq_holds a → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_quad_ineq_false_l184_184288


namespace length_AB_is_correct_l184_184262

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

end length_AB_is_correct_l184_184262


namespace range_of_a_if_in_first_quadrant_l184_184598

noncomputable def is_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem range_of_a_if_in_first_quadrant (a : ℝ) :
  is_first_quadrant ((1 + a * Complex.I) / (2 - Complex.I)) ↔ (-1/2 : ℝ) < a ∧ a < 2 := 
sorry

end range_of_a_if_in_first_quadrant_l184_184598


namespace intersection_A_B_l184_184590

-- Define the set A as natural numbers greater than 1
def A : Set ℕ := {x | x > 1}

-- Define the set B as numbers less than or equal to 3
def B : Set ℕ := {x | x ≤ 3}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x | x ∈ A ∧ x ∈ B}

-- State the theorem we want to prove
theorem intersection_A_B : A_inter_B = {2, 3} :=
  sorry

end intersection_A_B_l184_184590


namespace part1_l184_184983

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l184_184983


namespace pete_numbers_count_l184_184539

theorem pete_numbers_count :
  ∃ x_values : Finset Nat, x_values.card = 4 ∧
  ∀ x ∈ x_values, ∃ y z : Nat, 
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x + y) * z = 14 ∧ (x * y) + z = 14 :=
by
  sorry

end pete_numbers_count_l184_184539


namespace rabbit_excursion_time_l184_184406

theorem rabbit_excursion_time 
  (line_length : ℝ := 40) 
  (line_speed : ℝ := 3) 
  (rabbit_speed : ℝ := 5) : 
  -- The time calculated for the rabbit to return is 25 seconds
  (line_length / (rabbit_speed - line_speed) + line_length / (rabbit_speed + line_speed)) = 25 :=
by
  -- Placeholder for the proof, to be filled in with a detailed proof later
  sorry

end rabbit_excursion_time_l184_184406


namespace gas_volume_ranking_l184_184689

theorem gas_volume_ranking (Russia_V: ℝ) (Non_West_V: ℝ) (West_V: ℝ)
  (h_russia: Russia_V = 302790.13)
  (h_non_west: Non_West_V = 26848.55)
  (h_west: West_V = 21428): Russia_V > Non_West_V ∧ Non_West_V > West_V :=
by
  have h1: Russia_V = 302790.13 := h_russia
  have h2: Non_West_V = 26848.55 := h_non_west
  have h3: West_V = 21428 := h_west
  sorry


end gas_volume_ranking_l184_184689


namespace total_fruit_salads_is_1800_l184_184078

def Alaya_fruit_salads := 200
def Angel_fruit_salads := 2 * Alaya_fruit_salads
def Betty_fruit_salads := 3 * Angel_fruit_salads
def Total_fruit_salads := Alaya_fruit_salads + Angel_fruit_salads + Betty_fruit_salads

theorem total_fruit_salads_is_1800 : Total_fruit_salads = 1800 := by
  sorry

end total_fruit_salads_is_1800_l184_184078


namespace contest_end_time_l184_184059

-- Definitions for the conditions
def start_time_pm : Nat := 15 -- 3:00 p.m. in 24-hour format
def duration_min : Nat := 720

-- Proof that the contest ended at 3:00 a.m.
theorem contest_end_time :
  let end_time := (start_time_pm + (duration_min / 60)) % 24
  end_time = 3 :=
by
  -- This would be the place to provide the proof
  sorry

end contest_end_time_l184_184059


namespace inequality_solution_set_l184_184122

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 4)^2

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 1) / (x + 4)^2 ≥ 0} = {x : ℝ | x ≠ -4} :=
by
  sorry

end inequality_solution_set_l184_184122


namespace square_area_from_conditions_l184_184596

theorem square_area_from_conditions :
  ∀ (r s l b : ℝ), 
  l = r / 4 →
  r = s →
  l * b = 35 →
  b = 5 →
  s^2 = 784 := 
by 
  intros r s l b h1 h2 h3 h4
  sorry

end square_area_from_conditions_l184_184596


namespace custom_operator_example_l184_184766

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end custom_operator_example_l184_184766


namespace polynomial_no_ab_term_l184_184661

theorem polynomial_no_ab_term (a b m : ℝ) :
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  ∃ (m : ℝ), (p = a^2 - 12 * b^2) → (m = -2) :=
by
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  intro h
  use -2
  sorry

end polynomial_no_ab_term_l184_184661


namespace regular_21_gon_symmetries_and_angle_sum_l184_184467

theorem regular_21_gon_symmetries_and_angle_sum :
  let L' := 21
  let R' := 360 / 21
  L' + R' = 38.142857 := by
    sorry

end regular_21_gon_symmetries_and_angle_sum_l184_184467


namespace maria_first_stop_distance_is_280_l184_184193

noncomputable def maria_travel_distance : ℝ := 560
noncomputable def first_stop_distance (x : ℝ) : ℝ := x
noncomputable def distance_after_first_stop (x : ℝ) : ℝ := maria_travel_distance - first_stop_distance x
noncomputable def second_stop_distance (x : ℝ) : ℝ := (1 / 4) * distance_after_first_stop x
noncomputable def remaining_distance : ℝ := 210

theorem maria_first_stop_distance_is_280 :
  ∃ x, first_stop_distance x = 280 ∧ second_stop_distance x + remaining_distance = distance_after_first_stop x := sorry

end maria_first_stop_distance_is_280_l184_184193


namespace solve_equation_floor_l184_184120

theorem solve_equation_floor (x : ℚ) :
  (⌊(5 + 6 * x) / 8⌋ : ℚ) = (15 * x - 7) / 5 ↔ x = 7 / 15 ∨ x = 4 / 5 :=
by
  sorry

end solve_equation_floor_l184_184120


namespace profit_percentage_l184_184758

theorem profit_percentage (selling_price profit : ℝ) (h1 : selling_price = 900) (h2 : profit = 300) : 
  (profit / (selling_price - profit)) * 100 = 50 :=
by
  sorry

end profit_percentage_l184_184758


namespace Theresa_game_scores_l184_184109

theorem Theresa_game_scores 
  (h_sum_10 : 9 + 5 + 4 + 7 + 6 + 2 + 4 + 8 + 3 + 7 = 55)
  (h_p11 : ∀ p11 : ℕ, p11 < 10 → (55 + p11) % 11 = 0)
  (h_p12 : ∀ p11 p12 : ℕ, p11 < 10 → p12 < 10 → ((55 + p11 + p12) % 12 = 0)) :
  ∃ p11 p12 : ℕ, p11 < 10 ∧ p12 < 10 ∧ (55 + p11) % 11 = 0 ∧ (55 + p11 + p12) % 12 = 0 ∧ p11 * p12 = 0 :=
by
  sorry

end Theresa_game_scores_l184_184109


namespace solve_inequality_l184_184236

theorem solve_inequality (x : ℝ) : 
  (x ≠ 1) → ( (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ) ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) := 
sorry

end solve_inequality_l184_184236


namespace jungkook_biggest_l184_184198

noncomputable def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_biggest :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number :=
by
  unfold jungkook_number yoongi_number yuna_number
  sorry

end jungkook_biggest_l184_184198


namespace find_alpha_l184_184784

-- Define the given condition that alpha is inversely proportional to beta
def inv_proportional (α β : ℝ) (k : ℝ) : Prop := α * β = k

-- Main theorem statement
theorem find_alpha (α β k : ℝ) (h1 : inv_proportional 2 5 k) (h2 : inv_proportional α (-10) k) : α = -1 := by
  -- Given the conditions, the proof would follow, but it's not required here.
  sorry

end find_alpha_l184_184784


namespace range_of_f_l184_184885

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem range_of_f :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (f x) ∈ Set.Icc (-1 : ℝ) 2) :=
by
  sorry

end range_of_f_l184_184885


namespace kids_still_awake_l184_184007

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end kids_still_awake_l184_184007


namespace yolanda_three_point_avg_l184_184231

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

end yolanda_three_point_avg_l184_184231


namespace chimney_bricks_l184_184308

theorem chimney_bricks (x : ℝ) 
  (h1 : ∀ x, Brenda_rate = x / 8) 
  (h2 : ∀ x, Brandon_rate = x / 12) 
  (h3 : Combined_rate = (Brenda_rate + Brandon_rate - 15)) 
  (h4 : x = Combined_rate * 6) 
  : x = 360 := 
by 
  sorry

end chimney_bricks_l184_184308


namespace solve_equation_2021_2020_l184_184038

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end solve_equation_2021_2020_l184_184038


namespace find_AD_l184_184822

-- Define the geometrical context and constraints
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC AD BD CD : ℝ) (x : ℝ)

-- Assume the given conditions
def problem_conditions := 
  (AB = 50) ∧
  (AC = 41) ∧
  (BD = 10 * x) ∧
  (CD = 3 * x) ∧
  (AB^2 = AD^2 + BD^2) ∧
  (AC^2 = AD^2 + CD^2)

-- Formulate the problem question and the correct answer
theorem find_AD (h : problem_conditions AB AC AD BD CD x) : AD = 40 :=
sorry

end find_AD_l184_184822


namespace product_of_two_numbers_l184_184585

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_two_numbers_l184_184585


namespace obtuse_triangle_l184_184788

theorem obtuse_triangle (A B C M E : ℝ) (hM : M = (B + C) / 2) (hE : E > 0) 
(hcond : (B - E) ^ 2 + (C - E) ^ 2 >= 4 * (A - M) ^ 2): 
∃ α β γ, α > 90 ∧ β + γ < 90 ∧ α + β + γ = 180 :=
by
  sorry

end obtuse_triangle_l184_184788


namespace find_num_cows_l184_184291

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

end find_num_cows_l184_184291


namespace line_quadrants_condition_l184_184215

theorem line_quadrants_condition (m n : ℝ) (h : m * n < 0) :
  (m > 0 ∧ n < 0) :=
sorry

end line_quadrants_condition_l184_184215


namespace remainder_of_sum_l184_184970

theorem remainder_of_sum (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := 
by 
  -- proof goes here
  sorry

end remainder_of_sum_l184_184970


namespace molecular_weight_of_7_moles_AlPO4_is_correct_l184_184587

def atomic_weight_Al : Float := 26.98
def atomic_weight_P : Float := 30.97
def atomic_weight_O : Float := 16.00

def molecular_weight_AlPO4 : Float :=
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

noncomputable def weight_of_7_moles_AlPO4 : Float :=
  7 * molecular_weight_AlPO4

theorem molecular_weight_of_7_moles_AlPO4_is_correct :
  weight_of_7_moles_AlPO4 = 853.65 := by
  -- computation goes here
  sorry

end molecular_weight_of_7_moles_AlPO4_is_correct_l184_184587


namespace Jake_has_8_peaches_l184_184183

variables (Jake Steven Jill : ℕ)

-- The conditions
def condition1 : Steven = 15 := sorry
def condition2 : Steven = Jill + 14 := sorry
def condition3 : Jake = Steven - 7 := sorry

-- The proof statement
theorem Jake_has_8_peaches 
  (h1 : Steven = 15) 
  (h2 : Steven = Jill + 14) 
  (h3 : Jake = Steven - 7) : Jake = 8 :=
by
  -- The proof will go here
  sorry

end Jake_has_8_peaches_l184_184183


namespace proof_problem_l184_184358

theorem proof_problem (f g g_inv : ℝ → ℝ) (hinv : ∀ x, f (x ^ 4 - 1) = g x)
  (hginv : ∀ y, g (g_inv y) = y) (h : ∀ y, f (g_inv y) = g (g_inv y)) :
  g_inv (f 15) = 2 :=
by
  sorry

end proof_problem_l184_184358


namespace expression_simplification_l184_184272

theorem expression_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := 
by
  sorry

end expression_simplification_l184_184272


namespace inequality_sqrt_l184_184612

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l184_184612


namespace find_25_percent_l184_184253

theorem find_25_percent (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 :=
by
  sorry

end find_25_percent_l184_184253


namespace first_marvelous_monday_after_school_starts_l184_184375

def is_marvelous_monday (year : ℕ) (month : ℕ) (day : ℕ) (start_day : ℕ) : Prop :=
  let days_in_month := if month = 9 then 30 else if month = 10 then 31 else 0
  let fifth_monday := start_day + 28
  let is_monday := (fifth_monday - 1) % 7 = 0
  month = 10 ∧ day = 30 ∧ is_monday

theorem first_marvelous_monday_after_school_starts :
  ∃ (year month day : ℕ),
    year = 2023 ∧ month = 10 ∧ day = 30 ∧ is_marvelous_monday year month day 4 := sorry

end first_marvelous_monday_after_school_starts_l184_184375


namespace cos_phi_expression_l184_184426

theorem cos_phi_expression (a b c : ℝ) (φ R : ℝ)
  (habc : a > 0 ∧ b > 0 ∧ c > 0)
  (angles : 2 * φ + 3 * φ + 4 * φ = π)
  (law_of_sines : a / Real.sin (2 * φ) = 2 * R ∧ b / Real.sin (3 * φ) = 2 * R ∧ c / Real.sin (4 * φ) = 2 * R) :
  Real.cos φ = (a + c) / (2 * b) := 
by 
  sorry

end cos_phi_expression_l184_184426


namespace relations_of_sets_l184_184011

open Set

theorem relations_of_sets {A B : Set ℝ} (h : ∃ x ∈ A, x ∉ B) : 
  ¬(A ⊆ B) ∧ ((A ∩ B ≠ ∅) ∨ (B ⊆ A) ∨ (A ∩ B = ∅)) := sorry

end relations_of_sets_l184_184011


namespace ratio_of_triangle_and_hexagon_l184_184558

variable {n m : ℝ}

-- Conditions:
def is_regular_hexagon (ABCDEF : Type) : Prop := sorry
def area_of_hexagon (ABCDEF : Type) (n : ℝ) : Prop := sorry
def area_of_triangle_ACE (ABCDEF : Type) (m : ℝ) : Prop := sorry
  
theorem ratio_of_triangle_and_hexagon
  (ABCDEF : Type)
  (H1 : is_regular_hexagon ABCDEF)
  (H2 : area_of_hexagon ABCDEF n)
  (H3 : area_of_triangle_ACE ABCDEF m) :
  m / n = 2 / 3 := 
  sorry

end ratio_of_triangle_and_hexagon_l184_184558


namespace solution_is_option_C_l184_184444

-- Define the equation.
def equation (x y : ℤ) : Prop := x - 2 * y = 3

-- Define the given conditions as terms in Lean.
def option_A := (1, 1)   -- (x = 1, y = 1)
def option_B := (-1, 1)  -- (x = -1, y = 1)
def option_C := (1, -1)  -- (x = 1, y = -1)
def option_D := (-1, -1) -- (x = -1, y = -1)

-- The goal is to prove that option C is a solution to the equation.
theorem solution_is_option_C : equation 1 (-1) :=
by {
  -- Proof will go here
  sorry
}

end solution_is_option_C_l184_184444


namespace fraction_shaded_l184_184840

-- Define relevant elements
def quilt : ℕ := 9
def rows : ℕ := 3
def shaded_rows : ℕ := 1
def shaded_fraction := shaded_rows / rows

-- We are to prove the fraction of the quilt that is shaded
theorem fraction_shaded (h : quilt = 3 * 3) : shaded_fraction = 1 / 3 :=
by
  -- Proof goes here
  sorry

end fraction_shaded_l184_184840


namespace arithmetic_sequence_product_l184_184284

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

end arithmetic_sequence_product_l184_184284


namespace hall_length_l184_184865

theorem hall_length (b : ℕ) (h1 : b + 5 > 0) (h2 : (b + 5) * b = 750) : b + 5 = 30 :=
by {
  -- Proof goes here
  sorry
}

end hall_length_l184_184865


namespace expression_simplification_l184_184448

theorem expression_simplification :
  (4 * 6 / (12 * 8)) * ((5 * 12 * 8) / (4 * 5 * 5)) = 1 / 2 :=
by
  sorry

end expression_simplification_l184_184448


namespace quadratic_inequality_solution_l184_184527

variables {x : ℝ} {f : ℝ → ℝ}

def is_quadratic_and_opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_symmetric_at_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f (2 + x)

theorem quadratic_inequality_solution
  (h_quadratic : is_quadratic_and_opens_downwards f)
  (h_symmetric : is_symmetric_at_two f) :
  (1 - (Real.sqrt 14) / 4) < x ∧ x < (1 + (Real.sqrt 14) / 4) ↔
  f (Real.log ((1 / (1 / 4)) * (x^2 + x + 1 / 2))) <
  f (Real.log ((1 / (1 / 2)) * (2 * x^2 - x + 5 / 8))) :=
sorry

end quadratic_inequality_solution_l184_184527


namespace problem1_problem2_l184_184385

noncomputable def setA (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def setB : Set ℝ := {x | x^2 - 5 * x + 4 ≤ 0}

theorem problem1 (a : ℝ) (h : a = 1) : setA a ∪ setB = {x | 0 ≤ x ∧ x ≤ 4} := by
  sorry

theorem problem2 (a : ℝ) : (∀ x, x ∈ setA a → x ∈ setB) ↔ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end problem1_problem2_l184_184385


namespace minimum_abs_ab_l184_184258

theorem minimum_abs_ab (a b : ℝ) (h : (a^2) * (b / (a^2 + 1)) = 1) : abs (a * b) = 2 := 
  sorry

end minimum_abs_ab_l184_184258


namespace tetrahedron_inequality_l184_184551

variables (S A B C : Point)
variables (SA SB SC : Real)
variables (ABC : Plane)
variables (z : Real)
variable (h1 : angle B S C = π / 2)
variable (h2 : Project (point S) ABC = Orthocenter triangle ABC)
variable (h3 : RadiusInscribedCircle triangle ABC = z)

theorem tetrahedron_inequality :
  SA^2 + SB^2 + SC^2 >= 18 * z^2 :=
sorry

end tetrahedron_inequality_l184_184551


namespace intersection_of_M_and_N_l184_184926

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l184_184926


namespace hyperbola_correct_eqn_l184_184690

open Real

def hyperbola_eqn (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

theorem hyperbola_correct_eqn (e c a b x y : ℝ)
  (h_eccentricity : e = 2)
  (h_foci_distance : c = 4)
  (h_major_axis_half_length : a = 2)
  (h_minor_axis_half_length_square : b^2 = c^2 - a^2) :
  hyperbola_eqn x y :=
by
  sorry

end hyperbola_correct_eqn_l184_184690


namespace find_line_eq_l184_184505

theorem find_line_eq
  (l : ℝ → ℝ → Prop)
  (bisects_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0 → l x y)
  (perpendicular_to_line : ∀ x y : ℝ, l x y ↔ y = -1/2 * x)
  : ∀ x y : ℝ, l x y ↔ 2*x - y = 0 := by
  sorry

end find_line_eq_l184_184505


namespace final_balance_is_103_5_percent_of_initial_l184_184005

/-- Define Megan's initial balance. -/
def initial_balance : ℝ := 125

/-- Define the balance after 25% increase from babysitting. -/
def after_babysitting (balance : ℝ) : ℝ :=
  balance + (balance * 0.25)

/-- Define the balance after 20% decrease from buying shoes. -/
def after_shoes (balance : ℝ) : ℝ :=
  balance - (balance * 0.20)

/-- Define the balance after 15% increase by investing in stocks. -/
def after_stocks (balance : ℝ) : ℝ :=
  balance + (balance * 0.15)

/-- Define the balance after 10% decrease due to medical expenses. -/
def after_medical_expense (balance : ℝ) : ℝ :=
  balance - (balance * 0.10)

/-- Define the final balance. -/
def final_balance : ℝ :=
  let b1 := after_babysitting initial_balance
  let b2 := after_shoes b1
  let b3 := after_stocks b2
  after_medical_expense b3

/-- Prove that the final balance is 103.5% of the initial balance. -/
theorem final_balance_is_103_5_percent_of_initial :
  final_balance / initial_balance = 1.035 :=
by
  unfold final_balance
  unfold initial_balance
  unfold after_babysitting
  unfold after_shoes
  unfold after_stocks
  unfold after_medical_expense
  sorry

end final_balance_is_103_5_percent_of_initial_l184_184005


namespace fourth_power_nested_sqrt_l184_184192

noncomputable def nested_sqrt : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt : nested_sqrt ^ 4 = 16 := by
  sorry

end fourth_power_nested_sqrt_l184_184192


namespace sqrt_product_eq_l184_184641

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l184_184641


namespace problem_1_l184_184887

theorem problem_1 (m : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 := sorry

end problem_1_l184_184887


namespace ball_is_green_probability_l184_184037

noncomputable def probability_green_ball : ℚ :=
  let containerI_red := 8
  let containerI_green := 4
  let containerII_red := 3
  let containerII_green := 5
  let containerIII_red := 4
  let containerIII_green := 6
  let probability_container := (1 : ℚ) / 3
  let probability_green_I := (containerI_green : ℚ) / (containerI_red + containerI_green)
  let probability_green_II := (containerII_green : ℚ) / (containerII_red + containerII_green)
  let probability_green_III := (containerIII_green : ℚ) / (containerIII_red + containerIII_green)
  probability_container * probability_green_I +
  probability_container * probability_green_II +
  probability_container * probability_green_III

theorem ball_is_green_probability :
  probability_green_ball = 187 / 360 :=
by
  -- The detailed proof is omitted and left as an exercise
  sorry

end ball_is_green_probability_l184_184037


namespace inverse_value_of_f_l184_184204

theorem inverse_value_of_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2^x - 2) : f⁻¹ 2 = 3 :=
sorry

end inverse_value_of_f_l184_184204


namespace vector_subtraction_l184_184106

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end vector_subtraction_l184_184106


namespace sin_add_alpha_l184_184677

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end sin_add_alpha_l184_184677


namespace min_segments_of_polyline_l184_184462

theorem min_segments_of_polyline (n : ℕ) (h : n ≥ 2) : 
  ∃ s : ℕ, s = 2 * n - 2 := sorry

end min_segments_of_polyline_l184_184462


namespace positive_root_exists_iff_m_eq_neg_one_l184_184246

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end positive_root_exists_iff_m_eq_neg_one_l184_184246


namespace symmetric_point_proof_l184_184843

noncomputable def point_symmetric_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_proof :
  point_symmetric_to_x_axis (-2, 3) = (-2, -3) :=
by
  sorry

end symmetric_point_proof_l184_184843


namespace distance_from_P_to_AB_l184_184245

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

end distance_from_P_to_AB_l184_184245


namespace find_d_l184_184679

-- Define the six-digit number as a function of d
def six_digit_num (d : ℕ) : ℕ := 3 * 100000 + 2 * 10000 + 5 * 1000 + 4 * 100 + 7 * 10 + d

-- Define the sum of digits of the six-digit number
def sum_of_digits (d : ℕ) : ℕ := 3 + 2 + 5 + 4 + 7 + d

-- The statement we want to prove
theorem find_d (d : ℕ) : sum_of_digits d % 3 = 0 ↔ d = 3 :=
by
  sorry

end find_d_l184_184679


namespace soccer_campers_l184_184067

theorem soccer_campers (total_campers : ℕ) (basketball_campers : ℕ) (football_campers : ℕ) (h1 : total_campers = 88) (h2 : basketball_campers = 24) (h3 : football_campers = 32) : 
  total_campers - (basketball_campers + football_campers) = 32 := 
by 
  -- Proof omitted
  sorry

end soccer_campers_l184_184067


namespace arrange_abc_l184_184097

theorem arrange_abc : 
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4 / 5)
  let c := Real.sin 1
  a < c ∧ c < b := 
by
  sorry

end arrange_abc_l184_184097


namespace incorrect_expression_l184_184986

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (3 - x > 3 - y) :=
by
  sorry

end incorrect_expression_l184_184986


namespace power_log_simplification_l184_184416

theorem power_log_simplification (x : ℝ) (h : x > 0) : (16^(Real.log x / Real.log 2))^(1/4) = x :=
by sorry

end power_log_simplification_l184_184416


namespace probability_greater_than_two_on_three_dice_l184_184542

theorem probability_greater_than_two_on_three_dice :
  (4 / 6 : ℚ) ^ 3 = (8 / 27 : ℚ) :=
by
  sorry

end probability_greater_than_two_on_three_dice_l184_184542


namespace find_f_of_one_l184_184062

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_of_one : f 1 = 2 := 
by
  sorry

end find_f_of_one_l184_184062


namespace stations_visited_l184_184458

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end stations_visited_l184_184458


namespace find_number_l184_184267

theorem find_number (x : ℝ) 
(h : x * 13.26 + x * 9.43 + x * 77.31 = 470) : 
x = 4.7 := 
sorry

end find_number_l184_184267


namespace sum_gcd_lcm_l184_184938

theorem sum_gcd_lcm (A B : ℕ) (hA : A = Nat.gcd 10 (Nat.gcd 15 25)) (hB : B = Nat.lcm 10 (Nat.lcm 15 25)) :
  A + B = 155 :=
by
  sorry

end sum_gcd_lcm_l184_184938


namespace rotated_line_eq_l184_184762

theorem rotated_line_eq :
  ∀ (x y : ℝ), 
  (x - y + 4 = 0) ∨ (x - y - 4 = 0) ↔ 
  ∃ (x' y' : ℝ), (-x', -y') = (x, y) ∧ (x' - y' + 4 = 0) :=
by
  sorry

end rotated_line_eq_l184_184762


namespace distance_walked_l184_184074

theorem distance_walked (D : ℝ) (t1 t2 : ℝ): 
  (t1 = D / 4) → 
  (t2 = D / 3) → 
  (t2 - t1 = 1 / 2) → 
  D = 6 := 
by
  sorry

end distance_walked_l184_184074


namespace formula1_correct_formula2_correct_formula3_correct_l184_184967

noncomputable def formula1 (n : ℕ) := (Real.sqrt 2 / 2) * (1 - (-1 : ℝ) ^ n)
noncomputable def formula2 (n : ℕ) := Real.sqrt (1 - (-1 : ℝ) ^ n)
noncomputable def formula3 (n : ℕ) := if (n % 2 = 1) then Real.sqrt 2 else 0

theorem formula1_correct (n : ℕ) : 
  (n % 2 = 1 → formula1 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula1 n = 0) := 
by
  sorry

theorem formula2_correct (n : ℕ) : 
  (n % 2 = 1 → formula2 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula2 n = 0) := 
by
  sorry
  
theorem formula3_correct (n : ℕ) : 
  (n % 2 = 1 → formula3 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula3 n = 0) := 
by
  sorry

end formula1_correct_formula2_correct_formula3_correct_l184_184967


namespace fraction_to_decimal_l184_184139

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end fraction_to_decimal_l184_184139


namespace cookies_per_batch_l184_184255

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

end cookies_per_batch_l184_184255


namespace ana_wins_probability_l184_184977

noncomputable def probability_ana_wins : ℚ := 
  let a := (1 / 2)^5
  let r := (1 / 2)^4
  a / (1 - r)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 30 :=
by
  sorry

end ana_wins_probability_l184_184977


namespace cyclist_time_no_wind_l184_184388

theorem cyclist_time_no_wind (v w : ℝ) 
    (h1 : v + w = 1 / 3) 
    (h2 : v - w = 1 / 4) : 
    1 / v = 24 / 7 := 
by
  sorry

end cyclist_time_no_wind_l184_184388


namespace mark_first_part_playing_time_l184_184829

open Nat

theorem mark_first_part_playing_time (x : ℕ) (total_game_time second_part_playing_time sideline_time : ℕ)
  (h1 : total_game_time = 90) (h2 : second_part_playing_time = 35) (h3 : sideline_time = 35) 
  (h4 : x + second_part_playing_time + sideline_time = total_game_time) : x = 20 := 
by
  sorry

end mark_first_part_playing_time_l184_184829


namespace factory_sample_size_l184_184975

noncomputable def sample_size (A B C : ℕ) (sample_A : ℕ) : ℕ :=
  let total_ratio := A + B + C
  let ratio_A := A / total_ratio
  sample_A / ratio_A

theorem factory_sample_size
  (A B C : ℕ) (h_ratio : A = 2 ∧ B = 3 ∧ C = 5)
  (sample_A : ℕ) (h_sample_A : sample_A = 16) :
  sample_size A B C sample_A = 80 :=
by
  simp [h_ratio, h_sample_A, sample_size]
  sorry

end factory_sample_size_l184_184975


namespace min_hours_to_pass_message_ge_55_l184_184355

theorem min_hours_to_pass_message_ge_55 : 
  ∃ (n: ℕ), (∀ m: ℕ, m < n → 2^(m+1) - 2 ≤ 55) ∧ 2^(n+1) - 2 > 55 :=
by sorry

end min_hours_to_pass_message_ge_55_l184_184355


namespace train_speed_equals_36_0036_l184_184642

noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_equals_36_0036 :
  train_speed 70 6.999440044796416 = 36.0036 :=
by
  unfold train_speed
  sorry

end train_speed_equals_36_0036_l184_184642


namespace find_cost_of_books_l184_184947

theorem find_cost_of_books
  (C_L C_G1 C_G2 : ℝ)
  (h1 : C_L + C_G1 + C_G2 = 1080)
  (h2 : 0.9 * C_L = 1.15 * C_G1 + 1.25 * C_G2)
  (h3 : C_G1 + C_G2 = 1080 - C_L) :
  C_L = 784 :=
sorry

end find_cost_of_books_l184_184947


namespace option_A_option_B_option_C_option_D_l184_184332

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l184_184332


namespace player_a_winning_strategy_l184_184275

theorem player_a_winning_strategy (P : ℝ) : 
  (∃ m n : ℕ, P = m / (2 ^ n) ∧ m < 2 ^ n)
  ∨ P = 0
  ∨ P = 1 ↔
  (∀ d : ℝ, ∃ d_direction : ℤ, 
    (P + (d * d_direction) = 0) ∨ (P + (d * d_direction) = 1)) :=
sorry

end player_a_winning_strategy_l184_184275


namespace root_in_interval_l184_184404

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end root_in_interval_l184_184404


namespace alpha_beta_roots_l184_184640

variable (α β : ℝ)

theorem alpha_beta_roots (h1 : α^2 - 7 * α + 3 = 0) (h2 : β^2 - 7 * β + 3 = 0) (h3 : α > β) :
  α^2 + 7 * β = 46 :=
sorry

end alpha_beta_roots_l184_184640


namespace optimal_cookies_l184_184020

-- Define the initial state and the game's rules
def initial_blackboard : List Int := List.replicate 2020 1

def erase_two (l : List Int) (x y : Int) : List Int :=
  l.erase x |>.erase y

def write_back (l : List Int) (n : Int) : List Int :=
  n :: l

-- Define termination conditions
def game_ends_condition1 (l : List Int) : Prop :=
  ∃ x ∈ l, x > l.sum - x

def game_ends_condition2 (l : List Int) : Prop :=
  l = List.replicate (l.length) 0

def game_ends (l : List Int) : Prop :=
  game_ends_condition1 l ∨ game_ends_condition2 l

-- Define the number of cookies given to Player A
def cookies (l : List Int) : Int :=
  l.length

-- Prove that if both players play optimally, Player A receives 7 cookies
theorem optimal_cookies : cookies (initial_blackboard) = 7 :=
  sorry

end optimal_cookies_l184_184020


namespace represent_380000_in_scientific_notation_l184_184896

theorem represent_380000_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 380000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.8 ∧ n = 5 :=
by
  sorry

end represent_380000_in_scientific_notation_l184_184896


namespace blueberry_basket_count_l184_184698

noncomputable def number_of_blueberry_baskets 
    (plums_in_basket : ℕ) 
    (plum_baskets : ℕ) 
    (blueberries_in_basket : ℕ) 
    (total_fruits : ℕ) : ℕ := 
  let total_plums := plum_baskets * plums_in_basket
  let total_blueberries := total_fruits - total_plums
  total_blueberries / blueberries_in_basket

theorem blueberry_basket_count
  (plums_in_basket : ℕ) 
  (plum_baskets : ℕ) 
  (blueberries_in_basket : ℕ) 
  (total_fruits : ℕ)
  (h1 : plums_in_basket = 46)
  (h2 : plum_baskets = 19)
  (h3 : blueberries_in_basket = 170)
  (h4 : total_fruits = 1894) : 
  number_of_blueberry_baskets plums_in_basket plum_baskets blueberries_in_basket total_fruits = 6 := by
  sorry

end blueberry_basket_count_l184_184698


namespace range_of_a_l184_184783

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, (a - 1) * x > a - 1 → x < 1) : a < 1 :=
sorry

end range_of_a_l184_184783


namespace sum_of_values_of_x_l184_184953

noncomputable def g (x : ℝ) : ℝ :=
if x < 3 then 7 * x + 10 else 3 * x - 18

theorem sum_of_values_of_x (h : ∃ x : ℝ, g x = 5) :
  (∃ x1 x2 : ℝ, g x1 = 5 ∧ g x2 = 5) → (x1 + x2 = 18 / 7) :=
sorry

end sum_of_values_of_x_l184_184953


namespace range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l184_184749

variable {a b : ℝ}

theorem range_of_2a_plus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -10 < 2*a + b ∧ 2*a + b < 19 :=
by
  sorry

theorem range_of_a_minus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -9 < a - b ∧ a - b < 6 :=
by
  sorry

theorem range_of_a_div_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -2 < a / b ∧ a / b < 4 :=
by
  sorry

end range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l184_184749


namespace largest_angle_of_quadrilateral_l184_184812

open Real

theorem largest_angle_of_quadrilateral 
  (PQ QR RS : ℝ)
  (angle_RQP angle_SRQ largest_angle : ℝ)
  (h1: PQ = QR) 
  (h2: QR = RS) 
  (h3: angle_RQP = 60)
  (h4: angle_SRQ = 100)
  (h5: largest_angle = 130) : 
  largest_angle = 130 := by
  sorry

end largest_angle_of_quadrilateral_l184_184812


namespace probability_blue_given_popped_is_18_over_53_l184_184639

section PopcornProblem

/-- Representation of probabilities -/
def prob_white : ℚ := 1 / 2
def prob_yellow : ℚ := 1 / 4
def prob_blue : ℚ := 1 / 4

def pop_white_given_white : ℚ := 1 / 2
def pop_yellow_given_yellow : ℚ := 3 / 4
def pop_blue_given_blue : ℚ := 9 / 10

/-- Joint probabilities of kernel popping -/
def prob_white_popped : ℚ := prob_white * pop_white_given_white
def prob_yellow_popped : ℚ := prob_yellow * pop_yellow_given_yellow
def prob_blue_popped : ℚ := prob_blue * pop_blue_given_blue

/-- Total probability of popping -/
def prob_popped : ℚ := prob_white_popped + prob_yellow_popped + prob_blue_popped

/-- Conditional probability of being a blue kernel given that it popped -/
def prob_blue_given_popped : ℚ := prob_blue_popped / prob_popped

/-- The main theorem to prove the final probability -/
theorem probability_blue_given_popped_is_18_over_53 :
  prob_blue_given_popped = 18 / 53 :=
by sorry

end PopcornProblem

end probability_blue_given_popped_is_18_over_53_l184_184639


namespace find_speed_A_l184_184186

-- Defining the distance between the two stations as 155 km.
def distance := 155

-- Train A starts from station A at 7 a.m. and meets Train B at 11 a.m.
-- Therefore, Train A travels for 4 hours.
def time_A := 4

-- Train B starts from station B at 8 a.m. and meets Train A at 11 a.m.
-- Therefore, Train B travels for 3 hours.
def time_B := 3

-- Speed of Train B is given as 25 km/h.
def speed_B := 25

-- Condition that the total distance covered by both trains equals the distance between the two stations.
def meet_condition (v_A : ℕ) := (time_A * v_A) + (time_B * speed_B) = distance

-- The Lean theorem statement to be proved
theorem find_speed_A (v_A := 20) : meet_condition v_A :=
by
  -- Using 'sorrry' to skip the proof
  sorry

end find_speed_A_l184_184186


namespace distinct_ordered_pairs_l184_184482

theorem distinct_ordered_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h : 1/m + 1/n = 1/5) : 
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (1 / m + 1 / n = 1 / 5) :=
sorry

end distinct_ordered_pairs_l184_184482


namespace equivar_proof_l184_184909

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end equivar_proof_l184_184909


namespace james_music_listening_hours_l184_184227

theorem james_music_listening_hours (BPM : ℕ) (beats_per_week : ℕ) (hours_per_day : ℕ) 
  (h1 : BPM = 200) 
  (h2 : beats_per_week = 168000) 
  (h3 : hours_per_day * 200 * 60 * 7 = beats_per_week) : 
  hours_per_day = 2 := 
by
  sorry

end james_music_listening_hours_l184_184227


namespace toothpicks_per_card_l184_184486

-- Define the conditions of the problem
def numCardsInDeck : ℕ := 52
def numCardsNotUsed : ℕ := 16
def numCardsUsed : ℕ := numCardsInDeck - numCardsNotUsed

def numBoxesToothpicks : ℕ := 6
def toothpicksPerBox : ℕ := 450
def totalToothpicksUsed : ℕ := numBoxesToothpicks * toothpicksPerBox

-- Prove the number of toothpicks used per card
theorem toothpicks_per_card : totalToothpicksUsed / numCardsUsed = 75 := 
  by sorry

end toothpicks_per_card_l184_184486


namespace statement_A_statement_B_statement_C_l184_184915

variable {α : Type}

-- Conditions for statement A
def angle_greater (A B : ℝ) : Prop := A > B
def sin_greater (A B : ℝ) : Prop := Real.sin A > Real.sin B

-- Conditions for statement B
def acute_triangle (A B C : ℝ) : Prop := A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2
def sin_greater_than_cos (A B : ℝ) : Prop := Real.sin A > Real.cos B

-- Conditions for statement C
def obtuse_triangle (C : ℝ) : Prop := C > Real.pi / 2

-- Statement A in Lean
theorem statement_A (A B : ℝ) : angle_greater A B → sin_greater A B :=
sorry

-- Statement B in Lean
theorem statement_B {A B C : ℝ} : acute_triangle A B C → sin_greater_than_cos A B :=
sorry

-- Statement C in Lean
theorem statement_C {a b c : ℝ} (h : a^2 + b^2 < c^2) : obtuse_triangle C :=
sorry

-- Statement D in Lean (proof not needed as it's incorrect)
-- Theorem is omitted since statement D is incorrect

end statement_A_statement_B_statement_C_l184_184915


namespace tesses_ride_is_longer_l184_184279

noncomputable def tesses_total_distance : ℝ := 0.75 + 0.85 + 1.15
noncomputable def oscars_total_distance : ℝ := 0.25 + 1.35

theorem tesses_ride_is_longer :
  (tesses_total_distance - oscars_total_distance) = 1.15 := by
  sorry

end tesses_ride_is_longer_l184_184279


namespace projection_of_AB_onto_CD_l184_184263

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

end projection_of_AB_onto_CD_l184_184263


namespace car_miles_traveled_actual_miles_l184_184791

noncomputable def count_skipped_numbers (n : ℕ) : ℕ :=
  let count_digit7 (x : ℕ) : Bool := x = 7
  -- Function to count the number of occurrences of digit 7 in each place value
  let rec count (x num_skipped : ℕ) : ℕ :=
    if x = 0 then num_skipped else
    let digit := x % 10
    let new_count := if count_digit7 digit then num_skipped + 1 else num_skipped
    count (x / 10) new_count
  count n 0

theorem car_miles_traveled (odometer_reading : ℕ) : ℕ :=
  let num_skipped := count_skipped_numbers 3008
  odometer_reading - num_skipped

theorem actual_miles {odometer_reading : ℕ} (h : odometer_reading = 3008) : car_miles_traveled odometer_reading = 2194 :=
by sorry

end car_miles_traveled_actual_miles_l184_184791


namespace problem_statement_l184_184014

theorem problem_statement (a b c : ℝ) 
  (h1 : a - 2 * b + c = 0) 
  (h2 : a + 2 * b + c < 0) : b < 0 ∧ b^2 - a * c ≥ 0 :=
by
  sorry

end problem_statement_l184_184014


namespace chennai_to_hyderabad_distance_l184_184301

-- Definitions of the conditions
def david_speed := 50 -- mph
def lewis_speed := 70 -- mph
def meet_point := 250 -- miles from Chennai

-- Theorem statement
theorem chennai_to_hyderabad_distance :
  ∃ D T : ℝ, lewis_speed * T = D + (D - meet_point) ∧ david_speed * T = meet_point ∧ D = 300 :=
by
  sorry

end chennai_to_hyderabad_distance_l184_184301


namespace smallest_x_l184_184124

theorem smallest_x (x : ℕ) (h900 : 900 = 2^2 * 3^2 * 5^2) (h1152 : 1152 = 2^7 * 3^2) : 
  (900 * x) % 1152 = 0 ↔ x = 32 := 
by
  sorry

end smallest_x_l184_184124


namespace total_distance_joseph_ran_l184_184326

-- Defining the conditions
def distance_per_day : ℕ := 900
def days_run : ℕ := 3

-- The proof problem statement
theorem total_distance_joseph_ran :
  (distance_per_day * days_run) = 2700 :=
by
  sorry

end total_distance_joseph_ran_l184_184326


namespace n_cubed_plus_20n_div_48_l184_184134

theorem n_cubed_plus_20n_div_48 (n : ℕ) (h_even : n % 2 = 0) : (n^3 + 20 * n) % 48 = 0 :=
sorry

end n_cubed_plus_20n_div_48_l184_184134


namespace smallest_y_l184_184847

theorem smallest_y (y : ℕ) (h : 56 * y + 8 ≡ 6 [MOD 26]) : y = 6 := by
  sorry

end smallest_y_l184_184847


namespace value_of_a_purely_imaginary_l184_184846

-- Define the conditions under which a given complex number is purely imaginary
def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.im z * Complex.I ∧ b ≠ 0

-- Define the complex number based on the variable a
def given_complex_number (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 1⟩

-- The proof statement
theorem value_of_a_purely_imaginary :
  is_purely_imaginary (given_complex_number 2) := sorry

end value_of_a_purely_imaginary_l184_184846


namespace shortest_distance_l184_184888

theorem shortest_distance 
  (C : ℝ × ℝ) (B : ℝ × ℝ) (stream : ℝ)
  (hC : C = (0, -3))
  (hB : B = (9, -8))
  (hStream : stream = 0) :
  ∃ d : ℝ, d = 3 + Real.sqrt 202 :=
by
  sorry

end shortest_distance_l184_184888


namespace range_of_m_l184_184365

theorem range_of_m (m : ℝ) : 
  ((0 - m)^2 + (0 + m)^2 < 4) → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
by
  sorry

end range_of_m_l184_184365


namespace daisy_milk_problem_l184_184493

theorem daisy_milk_problem (total_milk : ℝ) (kids_percentage : ℝ) (remaining_milk : ℝ) (used_milk : ℝ) :
  total_milk = 16 →
  kids_percentage = 0.75 →
  remaining_milk = total_milk * (1 - kids_percentage) →
  used_milk = 2 →
  (used_milk / remaining_milk) * 100 = 50 :=
by
  intros _ _ _ _ 
  sorry

end daisy_milk_problem_l184_184493


namespace not_divisible_by_5_l184_184066

theorem not_divisible_by_5 (b : ℕ) : b = 6 ↔ ¬ (5 ∣ (2 * b ^ 3 - 2 * b ^ 2 + 2 * b - 1)) :=
sorry

end not_divisible_by_5_l184_184066


namespace number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l184_184210

def Jungkook_cards : Real := 0.8
def Yoongi_cards : Real := 0.5

theorem number_of_people_with_cards_leq_0_point_3 : 
  (Jungkook_cards <= 0.3 ∨ Yoongi_cards <= 0.3) = False := 
by 
  -- neither Jungkook nor Yoongi has number cards less than or equal to 0.3
  sorry

theorem number_of_people_with_cards_leq_0_point_3_count :
  (if (Jungkook_cards <= 0.3) then 1 else 0) + (if (Yoongi_cards <= 0.3) then 1 else 0) = 0 :=
by 
  -- calculate number of people with cards less than or equal to 0.3
  sorry

end number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l184_184210


namespace total_games_played_l184_184717

theorem total_games_played (n : ℕ) (h : n = 8) : (n.choose 2) = 28 := by
  sorry

end total_games_played_l184_184717


namespace time_to_be_d_miles_apart_l184_184342

def mary_walk_rate := 4 -- Mary's walking rate in miles per hour
def sharon_walk_rate := 6 -- Sharon's walking rate in miles per hour
def time_to_be_3_miles_apart := 0.3 -- Time in hours to be 3 miles apart
def initial_distance := 3 -- They are 3 miles apart after 0.3 hours

theorem time_to_be_d_miles_apart (d: ℝ) : ∀ t: ℝ,
  (mary_walk_rate + sharon_walk_rate) * t = d ↔ 
  t = d / (mary_walk_rate + sharon_walk_rate) :=
by
  intros
  sorry

end time_to_be_d_miles_apart_l184_184342


namespace fraction_c_over_d_l184_184474

-- Assume that we have a polynomial equation ax^3 + bx^2 + cx + d = 0 with roots 1, 2, 3
def polynomial (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The roots of the polynomial are 1, 2, 3
def roots (a b c d : ℝ) : Prop := polynomial a b c d 1 ∧ polynomial a b c d 2 ∧ polynomial a b c d 3

-- Vieta's formulas give us the relation for c and d in terms of the roots
theorem fraction_c_over_d (a b c d : ℝ) (h : roots a b c d) : c / d = -11 / 6 :=
sorry

end fraction_c_over_d_l184_184474


namespace problem1_problem2a_problem2b_problem2c_l184_184561

theorem problem1 {x : ℝ} : 3 * x ^ 2 - 5 * x - 2 < 0 → -1 / 3 < x ∧ x < 2 :=
sorry

theorem problem2a {x a : ℝ} (ha : -1 / 2 < a ∧ a < 0) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x < 2 ∨ x > -1 / a :=
sorry

theorem problem2b {x a : ℝ} (ha : a = -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x ≠ 2 :=
sorry

theorem problem2c {x a : ℝ} (ha : a < -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x > 2 ∨ x < -1 / a :=
sorry

end problem1_problem2a_problem2b_problem2c_l184_184561


namespace problem1_problem2_l184_184531

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end problem1_problem2_l184_184531


namespace rose_initial_rice_l184_184063

theorem rose_initial_rice : 
  ∀ (R : ℝ), (R - 9 / 10 * R - 1 / 4 * (R - 9 / 10 * R) = 0.75) → (R = 10) :=
by
  intro R h
  sorry

end rose_initial_rice_l184_184063


namespace three_digit_numbers_m_l184_184117

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem three_digit_numbers_m (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ sum_of_digits n = 12 ∧ 100 ≤ 2 * n ∧ 2 * n ≤ 999 ∧ sum_of_digits (2 * n) = 6 → ∃! (m : ℕ), n = m :=
sorry

end three_digit_numbers_m_l184_184117


namespace rearrangement_count_is_two_l184_184928

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 = 'a' ∧ c2 = 'b') ∨
  (c1 = 'b' ∧ c2 = 'c') ∨
  (c1 = 'c' ∧ c2 = 'd') ∨
  (c1 = 'd' ∧ c2 = 'e') ∨
  (c1 = 'b' ∧ c2 = 'a') ∨
  (c1 = 'c' ∧ c2 = 'b') ∨
  (c1 = 'd' ∧ c2 = 'c') ∨
  (c1 = 'e' ∧ c2 = 'd')

def no_adjacent_letters (s : List Char) : Bool :=
  match s with
  | [] => true
  | [_] => true
  | c1 :: c2 :: cs => 
    ¬ is_adjacent c1 c2 ∧ no_adjacent_letters (c2 :: cs)

def valid_rearrangements_count : Nat :=
  let perms := List.permutations ['a', 'b', 'c', 'd', 'e']
  perms.filter no_adjacent_letters |>.length

theorem rearrangement_count_is_two :
  valid_rearrangements_count = 2 :=
by sorry

end rearrangement_count_is_two_l184_184928


namespace waiter_initial_tables_l184_184068

theorem waiter_initial_tables
  (T : ℝ)
  (H1 : (T - 12.0) * 8.0 = 256) :
  T = 44.0 :=
sorry

end waiter_initial_tables_l184_184068


namespace bridge_length_correct_l184_184101

noncomputable def length_of_bridge 
  (train_length : ℝ) 
  (time_to_cross : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  (train_speed_kmph * (5 / 18) * time_to_cross) - train_length

theorem bridge_length_correct :
  length_of_bridge 120 31.99744020478362 36 = 199.9744020478362 :=
by
  -- Skipping the proof details
  sorry

end bridge_length_correct_l184_184101


namespace identify_counterfeit_in_three_weighings_l184_184177

def CoinType := {x // x = "gold" ∨ x = "silver"}

structure Coins where
  golds: Fin 13
  silvers: Fin 14
  is_counterfeit: CoinType
  counterfeit_weight: Int

def is_lighter (c1 c2: Coins): Prop := sorry
def is_heavier (c1 c2: Coins): Prop := sorry
def balance (c1 c2: Coins): Prop := sorry

def find_counterfeit_coin (coins: Coins): Option Coins := sorry

theorem identify_counterfeit_in_three_weighings (coins: Coins) :
  ∃ (identify: Coins → Option Coins),
  ∀ coins, ( identify coins ≠ none ) :=
sorry

end identify_counterfeit_in_three_weighings_l184_184177


namespace smallest_n_for_factors_l184_184123

theorem smallest_n_for_factors (k : ℕ) (hk : (∃ p : ℕ, k = 2^p) ) :
  ∃ (n : ℕ), ( 5^2 ∣ n * k * 36 * 343 ) ∧ ( 3^3 ∣ n * k * 36 * 343 ) ∧ n = 75 :=
by
  sorry

end smallest_n_for_factors_l184_184123


namespace num_positive_integers_l184_184793

theorem num_positive_integers (n : ℕ) :
    (0 < n ∧ n < 40 ∧ ∃ k : ℕ, k > 0 ∧ n = 40 * k / (k + 1)) ↔ 
    (n = 20 ∨ n = 30 ∨ n = 32 ∨ n = 35 ∨ n = 36 ∨ n = 38 ∨ n = 39) :=
sorry

end num_positive_integers_l184_184793


namespace boxes_needed_for_loose_crayons_l184_184171

-- Definitions based on conditions
def boxes_francine : ℕ := 5
def loose_crayons_francine : ℕ := 5
def loose_crayons_friend : ℕ := 27
def total_crayons_francine : ℕ := 85
def total_boxes_needed : ℕ := 2

-- The theorem to prove
theorem boxes_needed_for_loose_crayons 
  (hf : total_crayons_francine = boxes_francine * 16 + loose_crayons_francine)
  (htotal_loose : loose_crayons_francine + loose_crayons_friend = 32)
  (hboxes : boxes_francine = 5) : 
  total_boxes_needed = 2 :=
sorry

end boxes_needed_for_loose_crayons_l184_184171


namespace book_pages_l184_184737

theorem book_pages (n days_n : ℕ) (first_day_pages break_days : ℕ) (common_difference total_pages_read : ℕ) (portion_of_book : ℚ) :
    n = 14 → days_n = 12 → first_day_pages = 10 → break_days = 2 → common_difference = 2 →
    total_pages_read = 252 → portion_of_book = 3/4 →
    (total_pages_read : ℚ) * (4/3) = 336 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end book_pages_l184_184737


namespace solve_eq_n_fact_plus_n_eq_n_pow_k_l184_184046

theorem solve_eq_n_fact_plus_n_eq_n_pow_k :
  ∀ (n k : ℕ), 0 < n → 0 < k → (n! + n = n^k ↔ (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3)) :=
by
  sorry

end solve_eq_n_fact_plus_n_eq_n_pow_k_l184_184046


namespace customers_remaining_l184_184954

theorem customers_remaining (init : ℕ) (left : ℕ) (remaining : ℕ) :
  init = 21 → left = 9 → remaining = 12 → init - left = remaining :=
by sorry

end customers_remaining_l184_184954


namespace regular_price_of_Pony_jeans_l184_184476

theorem regular_price_of_Pony_jeans 
(Fox_price : ℝ) 
(Pony_price : ℝ) 
(savings : ℝ) 
(Fox_discount_rate : ℝ) 
(Pony_discount_rate : ℝ)
(h1 : Fox_price = 15)
(h2 : savings = 8.91)
(h3 : Fox_discount_rate + Pony_discount_rate = 0.22)
(h4 : Pony_discount_rate = 0.10999999999999996) : Pony_price = 18 := 
sorry

end regular_price_of_Pony_jeans_l184_184476


namespace percent_difference_l184_184833

theorem percent_difference : 
  let a := 0.60 * 50
  let b := 0.45 * 30
  a - b = 16.5 :=
by
  let a := 0.60 * 50
  let b := 0.45 * 30
  sorry

end percent_difference_l184_184833


namespace maximum_value_l184_184417

variable {a b c : ℝ}

-- Conditions
variable (h : a^2 + b^2 = c^2)

theorem maximum_value (h : a^2 + b^2 = c^2) : 
  (∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ 
   (∀ x y z : ℝ, x^2 + y^2 = z^2 → (x^2 + y^2 + x*y) / z^2 ≤ 1.5)) := 
sorry

end maximum_value_l184_184417


namespace book_selection_l184_184824

def num_books_in_genre (mystery fantasy biography : ℕ) : ℕ :=
  mystery + fantasy + biography

def num_combinations_two_diff_genres (mystery fantasy biography : ℕ) : ℕ :=
  if mystery = 4 ∧ fantasy = 4 ∧ biography = 4 then 48 else 0

theorem book_selection : 
  ∀ (mystery fantasy biography : ℕ),
  num_books_in_genre mystery fantasy biography = 12 →
  num_combinations_two_diff_genres mystery fantasy biography = 48 :=
by
  intros mystery fantasy biography h
  sorry

end book_selection_l184_184824


namespace toby_friends_girls_count_l184_184772

noncomputable def percentage_of_boys : ℚ := 55 / 100
noncomputable def boys_count : ℕ := 33
noncomputable def total_friends : ℚ := boys_count / percentage_of_boys
noncomputable def percentage_of_girls : ℚ := 1 - percentage_of_boys
noncomputable def girls_count : ℚ := percentage_of_girls * total_friends

theorem toby_friends_girls_count : girls_count = 27 := by
  sorry

end toby_friends_girls_count_l184_184772


namespace part1_part2_l184_184441

-- Part (1)  
theorem part1 (m : ℝ) : (∀ x : ℝ, 1 < x ∧ x < 3 → 2 * m < x ∧ x < 1 - m) ↔ (m ≤ -2) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x : ℝ, (1 < x ∧ x < 3) → ¬ (2 * m < x ∧ x < 1 - m)) ↔ (0 ≤ m) :=
sorry

end part1_part2_l184_184441


namespace ones_digit_of_9_pow_46_l184_184488

theorem ones_digit_of_9_pow_46 : (9 ^ 46) % 10 = 1 :=
by
  sorry

end ones_digit_of_9_pow_46_l184_184488


namespace assistant_increases_output_by_100_percent_l184_184878

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l184_184878


namespace photos_per_week_in_february_l184_184390

def january_photos : ℕ := 31 * 2

def total_photos (jan_feb_photos : ℕ) : ℕ := jan_feb_photos - january_photos

theorem photos_per_week_in_february (jan_feb_photos : ℕ) (weeks_in_february : ℕ)
  (h1 : jan_feb_photos = 146)
  (h2 : weeks_in_february = 4) :
  total_photos jan_feb_photos / weeks_in_february = 21 := by
  sorry

end photos_per_week_in_february_l184_184390


namespace similarity_of_triangle_l184_184547

noncomputable def side_length (AB BC AC : ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 1 → (AB, BC, AC) = (k * AB, k * BC, k * AC)

theorem similarity_of_triangle (AB BC AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : AC > 0) :
  side_length (2 * AB) (2 * BC) (2 * AC) = side_length AB BC AC :=
by sorry

end similarity_of_triangle_l184_184547


namespace import_tax_excess_amount_l184_184533

theorem import_tax_excess_amount 
    (tax_rate : ℝ) 
    (tax_paid : ℝ) 
    (total_value : ℝ)
    (X : ℝ) 
    (h1 : tax_rate = 0.07)
    (h2 : tax_paid = 109.2)
    (h3 : total_value = 2560) 
    (eq1 : tax_rate * (total_value - X) = tax_paid) :
    X = 1000 := sorry

end import_tax_excess_amount_l184_184533


namespace stacy_history_paper_length_l184_184154

theorem stacy_history_paper_length
  (days : ℕ)
  (pages_per_day : ℕ)
  (h_days : days = 6)
  (h_pages_per_day : pages_per_day = 11) :
  (days * pages_per_day) = 66 :=
by {
  sorry -- Proof goes here
}

end stacy_history_paper_length_l184_184154


namespace find_b_perpendicular_lines_l184_184917

theorem find_b_perpendicular_lines :
  ∀ (b : ℝ), (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 ∧ b * x - 3 * y + 6 = 0 →
      (2 / 3) * (b / 3) = -1) → b = -9 / 2 :=
sorry

end find_b_perpendicular_lines_l184_184917


namespace prob_Z_l184_184392

theorem prob_Z (P_X P_Y P_W P_Z : ℚ) (hX : P_X = 1/4) (hY : P_Y = 1/3) (hW : P_W = 1/6) 
(hSum : P_X + P_Y + P_Z + P_W = 1) : P_Z = 1/4 := 
by
  -- The proof will be filled in later
  sorry

end prob_Z_l184_184392


namespace number_of_games_can_buy_l184_184900

-- Definitions based on the conditions
def initial_money : ℕ := 42
def spent_money : ℕ := 10
def game_cost : ℕ := 8

-- The statement we need to prove: Mike can buy 4 games given the conditions
theorem number_of_games_can_buy : (initial_money - spent_money) / game_cost = 4 :=
by
  sorry

end number_of_games_can_buy_l184_184900


namespace first_problem_l184_184381

-- Definitions for the first problem
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (h_pos : ∀ n, a n > 0)
variable (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1))

-- Theorem statement for the first problem
theorem first_problem (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1)) :
  ∃ d, ∀ n, a (n + 1) - a n = d := sorry

end first_problem_l184_184381


namespace binomial_distrib_not_equiv_binom_expansion_l184_184874

theorem binomial_distrib_not_equiv_binom_expansion (a b : ℝ) (n : ℕ) (p : ℝ) (h1: a = p) (h2: b = 1 - p):
    ¬ (∃ k : ℕ, p ^ k * (1 - p) ^ (n - k) = (a + b) ^ n) := sorry

end binomial_distrib_not_equiv_binom_expansion_l184_184874


namespace minimum_positive_period_l184_184626

open Real

noncomputable def function := fun x : ℝ => 3 * sin (2 * x + π / 3)

theorem minimum_positive_period : ∃ T > 0, ∀ x, function (x + T) = function x ∧ (∀ T', T' > 0 → (∀ x, function (x + T') = function x) → T ≤ T') :=
  sorry

end minimum_positive_period_l184_184626


namespace remove_terms_l184_184578

-- Define the fractions
def f1 := 1 / 3
def f2 := 1 / 6
def f3 := 1 / 9
def f4 := 1 / 12
def f5 := 1 / 15
def f6 := 1 / 18

-- Define the total sum
def total_sum := f1 + f2 + f3 + f4 + f5 + f6

-- Define the target sum after removal
def target_sum := 2 / 3

-- Define the condition to be proven
theorem remove_terms {x y : Real} (h1 : (x = f4) ∧ (y = f5)) : 
  total_sum - (x + y) = target_sum := by
  sorry

end remove_terms_l184_184578


namespace leo_current_weight_l184_184299

theorem leo_current_weight (L K : ℕ) 
  (h1 : L + 10 = 3 * K / 2) 
  (h2 : L + K = 160)
  : L = 92 :=
sorry

end leo_current_weight_l184_184299


namespace inequality_sqrt_sum_l184_184002

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l184_184002


namespace cheryl_material_leftover_l184_184407

theorem cheryl_material_leftover :
  let material1 := (5 / 9 : ℚ)
  let material2 := (1 / 3 : ℚ)
  let total_bought := material1 + material2
  let used := (0.5555555555555556 : ℝ)
  let total_bought_decimal := (8 / 9 : ℝ)
  let leftover := total_bought_decimal - used
  leftover = 0.3333333333333332 := by
sorry

end cheryl_material_leftover_l184_184407


namespace miriam_flowers_total_l184_184311

theorem miriam_flowers_total :
  let monday_flowers := 45
  let tuesday_flowers := 75
  let wednesday_flowers := 35
  let thursday_flowers := 105
  let friday_flowers := 0
  let saturday_flowers := 60
  (monday_flowers + tuesday_flowers + wednesday_flowers + thursday_flowers + friday_flowers + saturday_flowers) = 320 :=
by
  -- Calculations go here but we're using sorry to skip them
  sorry

end miriam_flowers_total_l184_184311


namespace point_on_xaxis_equidistant_l184_184010

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end point_on_xaxis_equidistant_l184_184010


namespace arrangement_count_l184_184807

-- Definitions
def volunteers := 4
def elderly := 2
def total_people := volunteers + elderly
def criteria := "The 2 elderly people must be adjacent but not at the ends of the row."

-- Theorem: The number of different valid arrangements is 144
theorem arrangement_count : 
  ∃ (arrangements : Nat), arrangements = (volunteers.factorial * 3 * elderly.factorial) ∧ arrangements = 144 := 
  by 
    sorry

end arrangement_count_l184_184807


namespace abs_expression_not_positive_l184_184985

theorem abs_expression_not_positive (x : ℝ) (h : |2 * x - 7| = 0) : x = 7 / 2 :=
by
  sorry

end abs_expression_not_positive_l184_184985


namespace airline_num_airplanes_l184_184958

-- Definitions based on the conditions
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day_per_airplane : ℕ := 2
def total_passengers_per_day : ℕ := 1400

-- The theorem to prove the number of airplanes owned by the company
theorem airline_num_airplanes : 
  (total_passengers_per_day = 
   rows_per_airplane * seats_per_row * flights_per_day_per_airplane * n) → 
  n = 5 := 
by 
  sorry

end airline_num_airplanes_l184_184958


namespace football_team_progress_l184_184349

theorem football_team_progress (loss gain : ℤ) (h_loss : loss = -5) (h_gain : gain = 8) :
  (loss + gain = 3) :=
by
  sorry

end football_team_progress_l184_184349


namespace smallest_four_digit_multiple_of_15_l184_184771

theorem smallest_four_digit_multiple_of_15 :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 15 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 15 = 0) → n ≤ m) ∧ n = 1005 :=
sorry

end smallest_four_digit_multiple_of_15_l184_184771


namespace required_workers_l184_184335

variable (x : ℕ) (y : ℕ)

-- Each worker can produce x units of a craft per day.
-- A craft factory needs to produce 60 units of this craft per day.

theorem required_workers (h : x > 0) : y = 60 / x ↔ x * y = 60 :=
by sorry

end required_workers_l184_184335


namespace double_acute_angle_lt_180_l184_184540

theorem double_acute_angle_lt_180
  (α : ℝ) (h : 0 < α ∧ α < 90) : 2 * α < 180 := 
sorry

end double_acute_angle_lt_180_l184_184540


namespace find_larger_number_l184_184676

theorem find_larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
  sorry

end find_larger_number_l184_184676


namespace determine_values_of_x_l184_184431

variable (x : ℝ)

theorem determine_values_of_x (h1 : 1/x < 3) (h2 : 1/x > -4) : x > 1/3 ∨ x < -1/4 := 
  sorry


end determine_values_of_x_l184_184431


namespace total_weight_of_nuts_l184_184990

theorem total_weight_of_nuts (weight_almonds weight_pecans : ℝ) (h1 : weight_almonds = 0.14) (h2 : weight_pecans = 0.38) : weight_almonds + weight_pecans = 0.52 :=
by
  sorry

end total_weight_of_nuts_l184_184990


namespace football_team_people_count_l184_184625

theorem football_team_people_count (original_count : ℕ) (new_members : ℕ) (total_count : ℕ) 
  (h1 : original_count = 36) (h2 : new_members = 14) : total_count = 50 :=
by
  -- This is where the proof would go. We write 'sorry' because it is not required.
  sorry

end football_team_people_count_l184_184625


namespace chimney_base_radius_l184_184099

-- Given conditions
def tinplate_length := 219.8
def tinplate_width := 125.6
def pi_approx := 3.14

def radius_length (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

def radius_width (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

theorem chimney_base_radius :
  radius_length tinplate_length = 35 ∧ radius_width tinplate_width = 20 :=
by 
  sorry

end chimney_base_radius_l184_184099


namespace evaluate_expression_l184_184249

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

end evaluate_expression_l184_184249


namespace perfect_square_expression_5_l184_184633

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def expression_1 : ℕ := 3^3 * 4^4 * 7^7
def expression_2 : ℕ := 3^4 * 4^3 * 7^6
def expression_3 : ℕ := 3^5 * 4^6 * 7^5
def expression_4 : ℕ := 3^6 * 4^5 * 7^4
def expression_5 : ℕ := 3^4 * 4^6 * 7^4

theorem perfect_square_expression_5 : is_perfect_square expression_5 :=
sorry

end perfect_square_expression_5_l184_184633


namespace equivalent_statements_l184_184466

-- Definitions
variables (P Q : Prop)

-- Original statement
def original_statement := P → Q

-- Statements
def statement_I := P → Q
def statement_II := Q → P
def statement_III := ¬ Q → ¬ P
def statement_IV := ¬ P ∨ Q

-- Proof problem
theorem equivalent_statements : 
  (statement_III P Q ∧ statement_IV P Q) ↔ original_statement P Q :=
sorry

end equivalent_statements_l184_184466


namespace find_function_l184_184518

theorem find_function (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m^2 + n^2) = f m ^ 2 + f n ^ 2)
  (h2 : f 1 > 0) : ∀ n : ℕ, f n = n := 
sorry

end find_function_l184_184518


namespace monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l184_184240

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

end monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l184_184240


namespace obtuse_angle_condition_l184_184418

def dot_product (a b : (ℝ × ℝ)) : ℝ := a.1 * b.1 + a.2 * b.2

def is_obtuse_angle (a b : (ℝ × ℝ)) : Prop := dot_product a b < 0

def is_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem obtuse_angle_condition :
  (∀ (x : ℝ), x > 0 → is_obtuse_angle (-1, 0) (x, 1 - x) ∧ ¬is_parallel (-1, 0) (x, 1 - x)) ∧ 
  (∀ (x : ℝ), is_obtuse_angle (-1, 0) (x, 1 - x) → x > 0) :=
sorry

end obtuse_angle_condition_l184_184418


namespace initial_bottle_caps_l184_184145

theorem initial_bottle_caps 
    (x : ℝ) 
    (Nancy_bottle_caps : ℝ) 
    (Marilyn_current_bottle_caps : ℝ) 
    (h1 : Nancy_bottle_caps = 36.0)
    (h2 : Marilyn_current_bottle_caps = 87)
    (h3 : x + Nancy_bottle_caps = Marilyn_current_bottle_caps) : 
    x = 51 := 
by 
  sorry

end initial_bottle_caps_l184_184145


namespace complement_of_60_is_30_l184_184525

noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

theorem complement_of_60_is_30 : complement 60 = 30 :=
by 
  sorry

end complement_of_60_is_30_l184_184525


namespace square_of_binomial_l184_184116

theorem square_of_binomial (a b : ℝ) : 
  (a - 5 * b)^2 = a^2 - 10 * a * b + 25 * b^2 :=
by
  sorry

end square_of_binomial_l184_184116


namespace total_time_to_watch_movie_l184_184371

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l184_184371


namespace bacon_sold_l184_184572

variable (B : ℕ) -- Declare the variable for the number of slices of bacon sold

-- Define the given conditions as Lean definitions
def pancake_price := 4
def bacon_price := 2
def stacks_sold := 60
def total_raised := 420

-- The revenue from pancake sales alone
def pancake_revenue := stacks_sold * pancake_price
-- The revenue from bacon sales
def bacon_revenue := total_raised - pancake_revenue

-- Statement of the theorem
theorem bacon_sold :
  B = bacon_revenue / bacon_price :=
sorry

end bacon_sold_l184_184572


namespace find_eccentricity_of_ellipse_l184_184042

theorem find_eccentricity_of_ellipse
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (hx : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x, y) ∈ { p | (p.1^2 / a^2 + p.2^2 / b^2 = 1) })
  (hk : ∀ k x1 y1 x2 y2 : ℝ, y1 = k * x1 ∧ y2 = k * x2 → x1 ≠ x2 → (y1 = x1 * k ∧ y2 = x2 * k))  -- intersection points condition
  (hAB_AC : ∀ m n : ℝ, m ≠ 0 → (n - b) / m * (-n - b) / (-m) = -3/4 )
  : ∃ e : ℝ, e = 1/2 :=
sorry

end find_eccentricity_of_ellipse_l184_184042


namespace Kaarel_wins_l184_184857

theorem Kaarel_wins (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) :
  ∃ (x y a : ℕ), x ∈ Finset.range (p-1) ∧ y ∈ Finset.range (p-1) ∧ a ∈ Finset.range (p-1) ∧ 
  x ≠ y ∧ y ≠ (p - x) ∧ a ≠ x ∧ a ≠ (p - x) ∧ a ≠ y ∧ 
  (x * (p - x) + y * a) % p = 0 :=
sorry

end Kaarel_wins_l184_184857


namespace segment_ratios_l184_184912

theorem segment_ratios 
  (AB_parts BC_parts : ℝ) 
  (hAB: AB_parts = 3) 
  (hBC: BC_parts = 4) 
  : AB_parts / (AB_parts + BC_parts) = 3 / 7 ∧ BC_parts / (AB_parts + BC_parts) = 4 / 7 := 
  sorry

end segment_ratios_l184_184912


namespace minimum_value_sqrt_m2_n2_l184_184315

theorem minimum_value_sqrt_m2_n2 
  (a b m n : ℝ)
  (h1 : a^2 + b^2 = 3)
  (h2 : m*a + n*b = 3) : 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ Real.sqrt (m^2 + n^2) = k :=
by
  sorry

end minimum_value_sqrt_m2_n2_l184_184315


namespace hyperbola_asymptote_value_of_a_l184_184017

-- Define the hyperbola and the conditions given
variables {a : ℝ} (h1 : a > 0) (h2 : ∀ x y : ℝ, 3 * x + 2 * y = 0 ∧ 3 * x - 2 * y = 0)

theorem hyperbola_asymptote_value_of_a :
  a = 2 := by
  sorry

end hyperbola_asymptote_value_of_a_l184_184017


namespace solve_equation_l184_184323

theorem solve_equation :
  ∀ (x m n : ℕ), 
    0 < x → 0 < m → 0 < n → 
    x^m = 2^(2 * n + 1) + 2^n + 1 →
    (x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4) :=
by
  sorry

end solve_equation_l184_184323


namespace find_value_of_a_b_ab_l184_184571

variable (a b : ℝ)

theorem find_value_of_a_b_ab
  (h1 : 2 * a + 2 * b + a * b = 1)
  (h2 : a + b + 3 * a * b = -2) :
  a + b + a * b = 0 := 
sorry

end find_value_of_a_b_ab_l184_184571


namespace solution_l184_184013

noncomputable def problem_statement (a b : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)

theorem solution (a b : ℝ) :
  problem_statement a b ↔ (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a' b' : ℤ, (a : ℝ) = a' ∧ (b : ℝ) = b')) :=
by
  sorry

end solution_l184_184013


namespace rational_inequality_solution_l184_184886

open Set

theorem rational_inequality_solution (x : ℝ) :
  (x < -1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 5)) ↔ (x - 5) / ((x - 2) * (x^2 - 1)) < 0 := 
sorry

end rational_inequality_solution_l184_184886


namespace prop_A_prop_B_prop_C_prop_D_l184_184170

-- Proposition A: For all x ∈ ℝ, x² - x + 1 > 0
theorem prop_A (x : ℝ) : x^2 - x + 1 > 0 :=
sorry

-- Proposition B: a² + a = 0 is not a sufficient and necessary condition for a = 0
theorem prop_B : ¬(∀ a : ℝ, (a^2 + a = 0 ↔ a = 0)) :=
sorry

-- Proposition C: a > 1 and b > 1 is a sufficient and necessary condition for a + b > 2 and ab > 1
theorem prop_C (a b : ℝ) : (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b > 1) :=
sorry

-- Proposition D: a > 4 is a necessary and sufficient condition for the roots of the equation x² - ax + a = 0 to be all positive
theorem prop_D (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, x ≠ 0 → (x^2 - a*x + a = 0 → x > 0)) :=
sorry

end prop_A_prop_B_prop_C_prop_D_l184_184170


namespace smallest_four_digit_multiple_of_18_l184_184201

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l184_184201


namespace average_number_of_fish_is_75_l184_184580

-- Define the conditions
def BoastPool_fish := 75
def OnumLake_fish := BoastPool_fish + 25
def RiddlePond_fish := OnumLake_fish / 2

-- Prove the average number of fish
theorem average_number_of_fish_is_75 :
  (BoastPool_fish + OnumLake_fish + RiddlePond_fish) / 3 = 75 :=
by
  sorry

end average_number_of_fish_is_75_l184_184580


namespace g_1993_at_2_l184_184881

def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

def g_n : ℕ → ℚ → ℚ 
| 0     => id
| (n+1) => λ x => g (g_n n x)

theorem g_1993_at_2 : g_n 1993 2 = 65 / 53 := 
  sorry

end g_1993_at_2_l184_184881


namespace room_breadth_l184_184992

theorem room_breadth (length height diagonal : ℕ) (h_length : length = 12) (h_height : height = 9) (h_diagonal : diagonal = 17) : 
  ∃ breadth : ℕ, breadth = 8 :=
by
  -- Using the three-dimensional Pythagorean theorem:
  -- d² = length² + breadth² + height²
  -- 17² = 12² + b² + 9²
  -- 289 = 144 + b² + 81
  -- 289 = 225 + b²
  -- b² = 289 - 225
  -- b² = 64
  -- Taking the square root of both sides, we find:
  -- b = √64
  -- b = 8
  let b := 8
  existsi b
  -- This is a skip step, where we assert the breadth equals 8
  sorry

end room_breadth_l184_184992


namespace cat_finishes_food_on_next_wednesday_l184_184212

def cat_food_consumption_per_day : ℚ :=
  (1 / 4) + (1 / 6)

def total_food_on_day (n : ℕ) : ℚ :=
  n * cat_food_consumption_per_day

def total_cans : ℚ := 8

theorem cat_finishes_food_on_next_wednesday :
  total_food_on_day 10 = total_cans := sorry

end cat_finishes_food_on_next_wednesday_l184_184212


namespace simplify_expression_l184_184781

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

noncomputable def x : ℝ := (b / c) * (c / b)
noncomputable def y : ℝ := (a / c) * (c / a)
noncomputable def z : ℝ := (a / b) * (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 + x^2 * y^2 * z^2 = 4 := 
by {
  sorry
}

end simplify_expression_l184_184781


namespace cos_neg_pi_over_3_l184_184179

theorem cos_neg_pi_over_3 : Real.cos (-π / 3) = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l184_184179


namespace find_mass_of_aluminum_l184_184624

noncomputable def mass_of_aluminum 
  (rho_A : ℝ) (rho_M : ℝ) (delta_m : ℝ) : ℝ :=
  rho_A * delta_m / (rho_M - rho_A)

theorem find_mass_of_aluminum :
  mass_of_aluminum 2700 8900 0.06 = 26 := by
  sorry

end find_mass_of_aluminum_l184_184624


namespace total_dog_food_per_day_l184_184273

-- Definitions based on conditions
def dog1_eats_per_day : ℝ := 0.125
def dog2_eats_per_day : ℝ := 0.125
def number_of_dogs : ℕ := 2

-- Mathematically equivalent proof problem statement
theorem total_dog_food_per_day : dog1_eats_per_day + dog2_eats_per_day = 0.25 := 
by
  sorry

end total_dog_food_per_day_l184_184273


namespace blanket_thickness_after_foldings_l184_184543

theorem blanket_thickness_after_foldings (initial_thickness : ℕ) (folds : ℕ) (h1 : initial_thickness = 3) (h2 : folds = 4) :
  (initial_thickness * 2^folds) = 48 :=
by
  -- start with definitions as per the conditions
  rw [h1, h2]
  -- proof would follow
  sorry

end blanket_thickness_after_foldings_l184_184543


namespace trams_required_l184_184927

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l184_184927


namespace value_of_a5_max_sum_first_n_value_l184_184312

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem value_of_a5 (a d a5 : ℤ) :
  a5 = 4 ↔ (2 * a + 4 * d) + (a + 4 * d) + (a + 8 * d) = (a + 5 * d) + 8 :=
  sorry

theorem max_sum_first_n_value (a d : ℤ) (n : ℕ) (max_n : ℕ) :
  a = 16 →
  d = -3 →
  (∀ i, sum_first_n a d i ≤ sum_first_n a d max_n) →
  max_n = 6 :=
  sorry

end value_of_a5_max_sum_first_n_value_l184_184312


namespace man_born_year_l184_184328

theorem man_born_year (x : ℕ) : 
  (x^2 - x = 1806) ∧ (x^2 - x < 1850) ∧ (40 < x) ∧ (x < 50) → x = 43 :=
by
  sorry

end man_born_year_l184_184328


namespace simplified_factorial_fraction_l184_184654

theorem simplified_factorial_fraction :
  (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 :=
by
  sorry

end simplified_factorial_fraction_l184_184654


namespace rotational_homothety_commute_iff_centers_coincide_l184_184501

-- Define rotational homothety and its properties
structure RotationalHomothety (P : Type*) :=
(center : P)
(apply : P → P)
(is_homothety : ∀ p, apply (apply p) = apply p)

variables {P : Type*} [TopologicalSpace P] (H1 H2 : RotationalHomothety P)

-- Prove the equivalence statement
theorem rotational_homothety_commute_iff_centers_coincide :
  (H1.center = H2.center) ↔ (H1.apply ∘ H2.apply = H2.apply ∘ H1.apply) :=
sorry

end rotational_homothety_commute_iff_centers_coincide_l184_184501


namespace math_problem_l184_184756

theorem math_problem
  (numerator : ℕ := (Nat.factorial 10))
  (denominator : ℕ := (10 * 11 / 2)) :
  (numerator / denominator : ℚ) = 66069 + 1 / 11 := by
  sorry

end math_problem_l184_184756


namespace not_divisible_by_24_l184_184352

theorem not_divisible_by_24 : 
  ¬ (121416182022242628303234 % 24 = 0) := 
by
  sorry

end not_divisible_by_24_l184_184352


namespace integral_sin3_cos_l184_184602

open Real

theorem integral_sin3_cos :
  ∫ z in (π / 4)..(π / 2), sin z ^ 3 * cos z = 3 / 16 := by
  sorry

end integral_sin3_cos_l184_184602


namespace max_ab_l184_184460

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_l184_184460


namespace messages_on_monday_l184_184936

theorem messages_on_monday (M : ℕ) (h0 : 200 + 500 + 1000 = 1700) (h1 : M + 1700 = 2000) : M = 300 :=
by
  -- Maths proof step here
  sorry

end messages_on_monday_l184_184936


namespace brownie_pan_dimensions_l184_184041

def brownie_dimensions (m n : ℕ) : Prop :=
  let numSectionsLength := m - 1
  let numSectionsWidth := n - 1
  let totalPieces := (numSectionsLength + 1) * (numSectionsWidth + 1)
  let interiorPieces := (numSectionsLength - 1) * (numSectionsWidth - 1)
  let perimeterPieces := totalPieces - interiorPieces
  (numSectionsLength = 3) ∧ (numSectionsWidth = 5) ∧ (interiorPieces = 2 * perimeterPieces)

theorem brownie_pan_dimensions :
  ∃ (m n : ℕ), brownie_dimensions m n ∧ m = 6 ∧ n = 12 :=
by
  existsi 6
  existsi 12
  unfold brownie_dimensions
  simp
  exact sorry

end brownie_pan_dimensions_l184_184041
