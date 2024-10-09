import Mathlib

namespace minimum_cost_l2060_206092

noncomputable def f (x : ℝ) : ℝ := (1000 / (x + 5)) + 5 * x + (1 / 2) * (x^2 + 25)

theorem minimum_cost :
  (2 ≤ x ∧ x ≤ 8) →
  (f 5 = 150 ∧ (∀ y, 2 ≤ y ∧ y ≤ 8 → f y ≥ f 5)) :=
by
  intro h
  have f_exp : f x = (1000 / (x+5)) + 5*x + (1/2)*(x^2 + 25) := rfl
  sorry

end minimum_cost_l2060_206092


namespace fifth_rectangle_is_square_l2060_206060

-- Define the conditions
variables (s : ℝ) (a b : ℝ)
variables (R1 R2 R3 R4 : Set (ℝ × ℝ))
variables (R5 : Set (ℝ × ℝ))

-- Assume the areas of the corner rectangles are equal
def equal_area (R : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), R = {p | p.1 < a ∧ p.2 < b} ∧ a * b = k

-- State the conditions
axiom h1 : equal_area R1 a
axiom h2 : equal_area R2 a
axiom h3 : equal_area R3 a
axiom h4 : equal_area R4 a

axiom h5 : ∀ (p : ℝ × ℝ), p ∈ R5 → p.1 ≠ 0 → p.2 ≠ 0

-- Prove that the fifth rectangle is a square
theorem fifth_rectangle_is_square : ∃ c : ℝ, ∀ r1 r2, r1 ∈ R5 → r2 ∈ R5 → r1.1 - r2.1 = c ∧ r1.2 - r2.2 = c :=
by sorry

end fifth_rectangle_is_square_l2060_206060


namespace product_of_three_numbers_l2060_206044

theorem product_of_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 30) 
  (h2 : a = 5 * (b + c)) 
  (h3 : b = 9 * c) : 
  a * b * c = 56.25 := 
by 
  sorry

end product_of_three_numbers_l2060_206044


namespace tom_walking_distance_l2060_206056

noncomputable def walking_rate_miles_per_minute : ℝ := 1 / 18
def walking_time_minutes : ℝ := 15
def expected_distance_miles : ℝ := 0.8

theorem tom_walking_distance :
  walking_rate_miles_per_minute * walking_time_minutes = expected_distance_miles :=
by
  -- Calculation steps and conversion to decimal are skipped
  sorry

end tom_walking_distance_l2060_206056


namespace area_ratio_l2060_206030

theorem area_ratio (l w h : ℝ) (h1 : w * h = 288) (h2 : l * w = 432) (h3 : l * w * h = 5184) :
  (l * h) / (l * w) = 1 / 2 :=
sorry

end area_ratio_l2060_206030


namespace minimum_value_of_y_at_l2060_206050

noncomputable def y (x : ℝ) : ℝ := x * 2^x

theorem minimum_value_of_y_at :
  ∃ x : ℝ, (∀ x' : ℝ, y x ≤ y x') ∧ x = -1 / Real.log 2 :=
by
  sorry

end minimum_value_of_y_at_l2060_206050


namespace people_lost_l2060_206046

-- Define the given conditions
def ratio_won_to_lost : ℕ × ℕ := (4, 1)
def people_won : ℕ := 28

-- Define the proof problem
theorem people_lost (L : ℕ) (h_ratio : ratio_won_to_lost = (4, 1)) (h_won : people_won = 28) : L = 7 :=
by
  -- Skip the proof
  sorry

end people_lost_l2060_206046


namespace bananas_unit_measurement_l2060_206042

-- Definition of given conditions
def units_per_day : ℕ := 13
def total_bananas : ℕ := 9828
def total_weeks : ℕ := 9
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def bananas_per_day : ℕ := total_bananas / total_days
def bananas_per_unit : ℕ := bananas_per_day / units_per_day

-- Main theorem statement
theorem bananas_unit_measurement :
  bananas_per_unit = 12 := sorry

end bananas_unit_measurement_l2060_206042


namespace total_vehicles_in_lanes_l2060_206071

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end total_vehicles_in_lanes_l2060_206071


namespace find_f_of_3_l2060_206078

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l2060_206078


namespace isosceles_triangle_area_l2060_206082

open Real

noncomputable def area_of_isosceles_triangle (b : ℝ) (h : ℝ) : ℝ :=
  (1/2) * b * h

theorem isosceles_triangle_area :
  ∃ (b : ℝ) (l : ℝ), h = 8 ∧ (2 * l + b = 32) ∧ (area_of_isosceles_triangle b h = 48) :=
by
  sorry

end isosceles_triangle_area_l2060_206082


namespace image_preimage_f_l2060_206032

-- Defining the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Given conditions
def A : Set (ℝ × ℝ) := {p | True}
def B : Set (ℝ × ℝ) := {p | True}

-- Proof statement
theorem image_preimage_f :
  f (1, 3) = (4, -2) ∧ ∃ x y : ℝ, f (x, y) = (1, 3) ∧ (x, y) = (2, -1) :=
by
  sorry

end image_preimage_f_l2060_206032


namespace math_problem_real_solution_l2060_206000

theorem math_problem_real_solution (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) : 
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 :=
sorry

end math_problem_real_solution_l2060_206000


namespace bag_of_food_costs_two_dollars_l2060_206018

theorem bag_of_food_costs_two_dollars
  (cost_puppy : ℕ)
  (total_cost : ℕ)
  (daily_food : ℚ)
  (bag_food_quantity : ℚ)
  (weeks : ℕ)
  (h1 : cost_puppy = 10)
  (h2 : total_cost = 14)
  (h3 : daily_food = 1/3)
  (h4 : bag_food_quantity = 3.5)
  (h5 : weeks = 3) :
  (total_cost - cost_puppy) / (21 * daily_food / bag_food_quantity) = 2 := 
  by sorry

end bag_of_food_costs_two_dollars_l2060_206018


namespace tire_circumference_l2060_206099

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) (h_rpm : rpm = 400) (h_speed_kmh : speed_kmh = 48) :
  (C = 2) :=
by
  -- sorry statement to assume the solution for now
  sorry

end tire_circumference_l2060_206099


namespace henry_total_payment_l2060_206072

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l2060_206072


namespace each_piglet_ate_9_straws_l2060_206035

theorem each_piglet_ate_9_straws (t : ℕ) (h_t : t = 300)
                                 (p : ℕ) (h_p : p = 20)
                                 (f : ℕ) (h_f : f = (3 * t / 5)) :
  f / p = 9 :=
by
  sorry

end each_piglet_ate_9_straws_l2060_206035


namespace common_difference_l2060_206055

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end common_difference_l2060_206055


namespace find_x_l2060_206036

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 1)

theorem find_x (x : ℝ) (h : deriv f x = x) : x = 0 ∨ x = 1 :=
by
  sorry

end find_x_l2060_206036


namespace binary_division_remainder_correct_l2060_206080

-- Define the last two digits of the binary number
def b_1 : ℕ := 1
def b_0 : ℕ := 1

-- Define the function to calculate the remainder when dividing by 4
def binary_remainder (b1 b0 : ℕ) : ℕ := 2 * b1 + b0

-- Expected remainder in binary form
def remainder_in_binary : ℕ := 0b11  -- '11' in binary is 3 in decimal

-- The theorem to prove
theorem binary_division_remainder_correct :
  binary_remainder b_1 b_0 = remainder_in_binary :=
by
  -- Proof goes here
  sorry

end binary_division_remainder_correct_l2060_206080


namespace nina_expected_tomato_harvest_l2060_206089

noncomputable def expected_tomato_harvest 
  (garden_length : ℝ) (garden_width : ℝ) 
  (plants_per_sq_ft : ℝ) (tomatoes_per_plant : ℝ) : ℝ :=
  garden_length * garden_width * plants_per_sq_ft * tomatoes_per_plant

theorem nina_expected_tomato_harvest : 
  expected_tomato_harvest 10 20 5 10 = 10000 :=
by
  -- Proof would go here
  sorry

end nina_expected_tomato_harvest_l2060_206089


namespace five_cds_cost_with_discount_l2060_206015

theorem five_cds_cost_with_discount
  (price_2_cds : ℝ)
  (discount_rate : ℝ)
  (num_cds : ℕ)
  (total_cost : ℝ) 
  (h1 : price_2_cds = 40)
  (h2 : discount_rate = 0.10)
  (h3 : num_cds = 5)
  : total_cost = 90 :=
by
  sorry

end five_cds_cost_with_discount_l2060_206015


namespace ray_has_4_nickels_left_l2060_206084

theorem ray_has_4_nickels_left (initial_cents : ℕ) (given_to_peter : ℕ)
    (given_to_randi : ℕ) (value_of_nickel : ℕ) (remaining_cents : ℕ) 
    (remaining_nickels : ℕ) :
    initial_cents = 95 →
    given_to_peter = 25 →
    given_to_randi = 2 * given_to_peter →
    value_of_nickel = 5 →
    remaining_cents = initial_cents - given_to_peter - given_to_randi →
    remaining_nickels = remaining_cents / value_of_nickel →
    remaining_nickels = 4 :=
by
  intros
  sorry

end ray_has_4_nickels_left_l2060_206084


namespace usual_time_cover_journey_l2060_206051

theorem usual_time_cover_journey (S T : ℝ) (H : S / T = (5/6 * S) / (T + 8)) : T = 48 :=
by
  sorry

end usual_time_cover_journey_l2060_206051


namespace sum_first_75_terms_arith_seq_l2060_206037

theorem sum_first_75_terms_arith_seq (a_1 d : ℕ) (n : ℕ) (h_a1 : a_1 = 3) (h_d : d = 4) (h_n : n = 75) : 
  (n * (2 * a_1 + (n - 1) * d)) / 2 = 11325 := 
by
  subst h_a1
  subst h_d
  subst h_n
  sorry

end sum_first_75_terms_arith_seq_l2060_206037


namespace geometric_series_common_ratio_l2060_206026

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l2060_206026


namespace symmetric_axis_of_parabola_l2060_206062

theorem symmetric_axis_of_parabola :
  (∃ x : ℝ, x = 6 ∧ (∀ y : ℝ, y = 1/2 * x^2 - 6 * x + 21)) :=
sorry

end symmetric_axis_of_parabola_l2060_206062


namespace value_of_x_l2060_206095

theorem value_of_x (g : ℝ → ℝ) (h : ∀ x, g (5 * x + 2) = 3 * x - 4) : g (-13) = -13 :=
by {
  sorry
}

end value_of_x_l2060_206095


namespace GCD_180_252_315_l2060_206038

theorem GCD_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end GCD_180_252_315_l2060_206038


namespace piravena_flight_cost_l2060_206065

noncomputable def cost_of_flight (distance_km : ℕ) (booking_fee : ℕ) (rate_per_km : ℕ) : ℕ :=
  booking_fee + (distance_km * rate_per_km / 100)

def check_cost_of_flight : Prop :=
  let distance_bc := 1000
  let booking_fee := 100
  let rate_per_km := 10
  cost_of_flight distance_bc booking_fee rate_per_km = 200

theorem piravena_flight_cost : check_cost_of_flight := 
by {
  sorry
}

end piravena_flight_cost_l2060_206065


namespace range_of_m_l2060_206022

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → ((m^2 - m) * 4^x - 2^x < 0)) → (-1 < m ∧ m < 2) :=
by
  sorry

end range_of_m_l2060_206022


namespace tangent_points_l2060_206020

theorem tangent_points (x y : ℝ) (h : y = x^3 - 3 * x) (slope_zero : 3 * x^2 - 3 = 0) :
  (x = -1 ∧ y = 2) ∨ (x = 1 ∧ y = -2) :=
sorry

end tangent_points_l2060_206020


namespace trig_inequality_l2060_206054

theorem trig_inequality : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end trig_inequality_l2060_206054


namespace actual_distance_between_towns_l2060_206043

-- Definitions based on conditions
def scale_inch_to_miles : ℚ := 8
def map_distance_inches : ℚ := 27 / 8

-- Proof statement
theorem actual_distance_between_towns : scale_inch_to_miles * map_distance_inches / (1 / 4) = 108 := by
  sorry

end actual_distance_between_towns_l2060_206043


namespace alyssa_puppies_l2060_206086

-- Definitions from the problem conditions
def initial_puppies (P x : ℕ) : ℕ := P + x

-- Lean 4 Statement of the problem
theorem alyssa_puppies (P x : ℕ) (given_aw: 7 = 7) (remaining: 5 = 5) :
  initial_puppies P x = 12 :=
sorry

end alyssa_puppies_l2060_206086


namespace find_fifth_month_sale_l2060_206097

theorem find_fifth_month_sale (
  a1 a2 a3 a4 a6 : ℕ
) (avg_sales : ℕ)
  (h1 : a1 = 5420)
  (h2 : a2 = 5660)
  (h3 : a3 = 6200)
  (h4 : a4 = 6350)
  (h6 : a6 = 7070)
  (avg_condition : avg_sales = 6200)
  (total_condition : (a1 + a2 + a3 + a4 + a6 + (6500)) / 6 = avg_sales)
  : (∃ a5 : ℕ, a5 = 6500 ∧ (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sales) :=
by {
  sorry
}

end find_fifth_month_sale_l2060_206097


namespace max_is_twice_emily_probability_l2060_206048

noncomputable def probability_event_max_gt_twice_emily : ℝ :=
  let total_area := 1000 * 3000
  let triangle_area := 1/2 * 1000 * 1000
  let rectangle_area := 1000 * (3000 - 2000)
  let favorable_area := triangle_area + rectangle_area
  favorable_area / total_area

theorem max_is_twice_emily_probability :
  probability_event_max_gt_twice_emily = 1 / 2 :=
by
  sorry

end max_is_twice_emily_probability_l2060_206048


namespace find_vector_at_t5_l2060_206070

def vector_on_line (t : ℝ) : ℝ × ℝ := 
  let a := (0, 11) -- From solving the system of equations
  let d := (2, -4) -- From solving the system of equations
  (a.1 + t * d.1, a.2 + t * d.2)

theorem find_vector_at_t5 : vector_on_line 5 = (10, -9) := 
by 
  sorry

end find_vector_at_t5_l2060_206070


namespace frac_eval_eq_l2060_206023

theorem frac_eval_eq :
  let a := 19
  let b := 8
  let c := 35
  let d := 19 * 8 / 35
  ( (⌈a / b - ⌈c / d⌉⌉) / ⌈c / b + ⌈d⌉⌉) = (1 / 10) := by
  sorry

end frac_eval_eq_l2060_206023


namespace min_value_of_z_l2060_206096

theorem min_value_of_z (x y : ℝ) (h : y^2 = 4 * x) : 
  ∃ (z : ℝ), z = 3 ∧ ∀ (x' : ℝ) (hx' : x' ≥ 0), ∃ (y' : ℝ), y'^2 = 4 * x' → z ≤ (1/2) * y'^2 + x'^2 + 3 :=
by sorry

end min_value_of_z_l2060_206096


namespace delta_delta_delta_45_l2060_206059

def delta (P : ℚ) : ℚ := (2 / 3) * P + 2

theorem delta_delta_delta_45 :
  delta (delta (delta 45)) = 158 / 9 :=
by sorry

end delta_delta_delta_45_l2060_206059


namespace solution_2016_121_solution_2016_144_l2060_206031

-- Definitions according to the given conditions
def delta_fn (f : ℕ → ℕ → ℕ) :=
  (∀ a b : ℕ, f (a + b) b = f a b + 1) ∧ (∀ a b : ℕ, f a b * f b a = 0)

-- Proof objectives
theorem solution_2016_121 (f : ℕ → ℕ → ℕ) (h : delta_fn f) : f 2016 121 = 16 :=
sorry

theorem solution_2016_144 (f : ℕ → ℕ → ℕ) (h : delta_fn f) : f 2016 144 = 13 :=
sorry

end solution_2016_121_solution_2016_144_l2060_206031


namespace perimeter_triangle_l2060_206033

-- Definitions and conditions
def side1 : ℕ := 2
def side2 : ℕ := 5
def is_odd (n : ℕ) : Prop := n % 2 = 1
def valid_third_side (x : ℕ) : Prop := 3 < x ∧ x < 7 ∧ is_odd x

-- Theorem statement
theorem perimeter_triangle : ∃ (x : ℕ), valid_third_side x ∧ (side1 + side2 + x = 12) :=
by 
  sorry

end perimeter_triangle_l2060_206033


namespace incoming_class_student_count_l2060_206016

theorem incoming_class_student_count (n : ℕ) :
  n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 → n = 418 :=
by
  sorry

end incoming_class_student_count_l2060_206016


namespace binomial_square_coefficients_l2060_206025

noncomputable def a : ℝ := 13.5
noncomputable def b : ℝ := 18

theorem binomial_square_coefficients (c d : ℝ) :
  (∀ x : ℝ, 6 * x ^ 2 + 18 * x + a = (c * x + d) ^ 2) ∧ 
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 4 = (c * x + d) ^ 2)  → 
  a = 13.5 ∧ b = 18 := sorry

end binomial_square_coefficients_l2060_206025


namespace number_of_students_l2060_206014

theorem number_of_students (S G : ℕ) (h1 : G = 2 * S / 3) (h2 : 8 = 2 * G / 5) : S = 30 :=
by
  sorry

end number_of_students_l2060_206014


namespace calculation_2015_l2060_206094

theorem calculation_2015 :
  2015 ^ 2 - 2016 * 2014 = 1 :=
by
  sorry

end calculation_2015_l2060_206094


namespace water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l2060_206029

noncomputable def tiered_water_bill (usage : ℕ) : ℝ :=
  if usage <= 20 then
    2.3 * usage
  else if usage <= 30 then
    2.3 * 20 + 3.45 * (usage - 20)
  else
    2.3 * 20 + 3.45 * 10 + 4.6 * (usage - 30)

-- (1) Prove that if Xiao Ming's family used 32 cubic meters of water in August, 
-- their water bill is 89.7 yuan.
theorem water_bill_august_32m_cubed : tiered_water_bill 32 = 89.7 := by
  sorry

-- (2) Prove that if Xiao Ming's family paid 59.8 yuan for their water bill in October, 
-- they used 24 cubic meters of water.
theorem water_usage_october_59_8_yuan : ∃ x : ℕ, tiered_water_bill x = 59.8 ∧ x = 24 := by
  use 24
  sorry

end water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l2060_206029


namespace chocolates_bought_in_a_month_l2060_206006

theorem chocolates_bought_in_a_month :
  ∀ (chocolates_for_her: ℕ)
    (chocolates_for_sister: ℕ)
    (chocolates_for_charlie: ℕ)
    (weeks_in_a_month: ℕ), 
  weeks_in_a_month = 4 →
  chocolates_for_her = 2 →
  chocolates_for_sister = 1 →
  chocolates_for_charlie = 10 →
  (chocolates_for_her * weeks_in_a_month + chocolates_for_sister * weeks_in_a_month + chocolates_for_charlie) = 22 :=
by
  intros chocolates_for_her chocolates_for_sister chocolates_for_charlie weeks_in_a_month
  intros h_weeks h_her h_sister h_charlie
  sorry

end chocolates_bought_in_a_month_l2060_206006


namespace hawkeye_fewer_mainecoons_than_gordon_l2060_206085

-- Definitions based on conditions
def JamiePersians : ℕ := 4
def JamieMaineCoons : ℕ := 2
def GordonPersians : ℕ := JamiePersians / 2
def GordonMaineCoons : ℕ := JamieMaineCoons + 1
def TotalCats : ℕ := 13
def JamieTotalCats : ℕ := JamiePersians + JamieMaineCoons
def GordonTotalCats : ℕ := GordonPersians + GordonMaineCoons
def JamieAndGordonTotalCats : ℕ := JamieTotalCats + GordonTotalCats
def HawkeyeTotalCats : ℕ := TotalCats - JamieAndGordonTotalCats
def HawkeyePersians : ℕ := 0
def HawkeyeMaineCoons : ℕ := HawkeyeTotalCats - HawkeyePersians

-- Theorem statement to prove: Hawkeye owns 1 fewer Maine Coon than Gordon
theorem hawkeye_fewer_mainecoons_than_gordon : HawkeyeMaineCoons + 1 = GordonMaineCoons :=
by
  sorry

end hawkeye_fewer_mainecoons_than_gordon_l2060_206085


namespace student_marks_l2060_206069

theorem student_marks (T P F M : ℕ)
  (hT : T = 600)
  (hP : P = 33)
  (hF : F = 73)
  (hM : M = (P * T / 100) - F) : M = 125 := 
by 
  sorry

end student_marks_l2060_206069


namespace proper_fraction_cubed_numerator_triples_denominator_add_three_l2060_206045

theorem proper_fraction_cubed_numerator_triples_denominator_add_three
  (a b : ℕ)
  (h1 : a < b)
  (h2 : (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b) : 
  a = 2 ∧ b = 9 :=
by
  sorry

end proper_fraction_cubed_numerator_triples_denominator_add_three_l2060_206045


namespace workshop_processing_equation_l2060_206064

noncomputable def process_equation (x : ℝ) : Prop :=
  (4000 / x - 4200 / (1.5 * x) = 3)

theorem workshop_processing_equation (x : ℝ) (hx : x > 0) :
  process_equation x :=
by
  sorry

end workshop_processing_equation_l2060_206064


namespace sam_found_pennies_l2060_206041

-- Define the function that computes the number of pennies Sam found given the initial and current amounts of pennies
def find_pennies (initial_pennies current_pennies : Nat) : Nat :=
  current_pennies - initial_pennies

-- Define the main proof problem
theorem sam_found_pennies : find_pennies 98 191 = 93 := by
  -- Proof steps would go here
  sorry

end sam_found_pennies_l2060_206041


namespace octal_to_base12_conversion_l2060_206040

-- Define the computation functions required
def octalToDecimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 64 + d1 * 8 + d0

def decimalToBase12 (n : ℕ) : List ℕ :=
  let d0 := n % 12
  let n1 := n / 12
  let d1 := n1 % 12
  let n2 := n1 / 12
  let d2 := n2 % 12
  [d2, d1, d0]

-- The main theorem that combines both conversions
theorem octal_to_base12_conversion :
  decimalToBase12 (octalToDecimal 563) = [2, 6, 11] :=
sorry

end octal_to_base12_conversion_l2060_206040


namespace solve_correct_problems_l2060_206079

theorem solve_correct_problems (x : ℕ) (h1 : 3 * x + x = 120) : x = 30 :=
by
  sorry

end solve_correct_problems_l2060_206079


namespace determinant_in_terms_of_roots_l2060_206019

theorem determinant_in_terms_of_roots 
  (r s t a b c : ℝ)
  (h1 : a^3 - r*a^2 + s*a - t = 0)
  (h2 : b^3 - r*b^2 + s*b - t = 0)
  (h3 : c^3 - r*c^2 + s*c - t = 0) :
  (2 + a) * ((2 + b) * (2 + c) - 4) - 2 * (2 * (2 + c) - 4) + 2 * (2 * 2 - (2 + b) * 2) = t - 2 * s :=
by
  sorry

end determinant_in_terms_of_roots_l2060_206019


namespace average_of_consecutive_numbers_l2060_206039

-- Define the 7 consecutive numbers and their properties
variables (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (g : ℝ)

-- Conditions given in the problem
def consecutive_numbers (a b c d e f g : ℝ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6

def percent_relationship (a g : ℝ) : Prop :=
  g = 1.5 * a

-- The proof problem
theorem average_of_consecutive_numbers (a b c d e f g : ℝ)
  (h1 : consecutive_numbers a b c d e f g)
  (h2 : percent_relationship a g) :
  (a + b + c + d + e + f + g) / 7 = 15 :=
by {
  sorry -- Proof goes here
}

-- To ensure it passes the type checker but without providing the actual proof, we use sorry.

end average_of_consecutive_numbers_l2060_206039


namespace julia_more_kids_on_Monday_l2060_206024

def kids_played_on_Tuesday : Nat := 14
def kids_played_on_Monday : Nat := 22

theorem julia_more_kids_on_Monday : kids_played_on_Monday - kids_played_on_Tuesday = 8 :=
by {
  sorry
}

end julia_more_kids_on_Monday_l2060_206024


namespace price_correct_l2060_206090

noncomputable def price_per_glass_on_second_day 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * P) 
  : ℝ := 0.40

theorem price_correct 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * 0.40) 
  : price_per_glass_on_second_day O price_first_day revenue_equal = 0.40 := 
by 
  sorry

end price_correct_l2060_206090


namespace number_of_ants_l2060_206093

def spiders := 8
def spider_legs := 8
def ants := 12
def ant_legs := 6
def total_legs := 136

theorem number_of_ants :
  spiders * spider_legs + ants * ant_legs = total_legs → ants = 12 :=
by
  sorry

end number_of_ants_l2060_206093


namespace sum_series_eq_3_over_4_l2060_206088

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l2060_206088


namespace pass_in_both_subjects_l2060_206058

variable (F_H F_E F_HE : ℝ)

theorem pass_in_both_subjects (h1 : F_H = 20) (h2 : F_E = 70) (h3 : F_HE = 10) :
  100 - ((F_H + F_E) - F_HE) = 20 :=
by
  sorry

end pass_in_both_subjects_l2060_206058


namespace part_a_part_b_l2060_206077

-- Part (a)
theorem part_a (x y z : ℤ) : (x^2 + y^2 + z^2 = 2 * x * y * z) → (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

-- Part (b)
theorem part_b : ∃ (x y z v : ℤ), (x^2 + y^2 + z^2 + v^2 = 2 * x * y * z * v) → (x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) :=
by
  sorry

end part_a_part_b_l2060_206077


namespace largest_x_value_l2060_206073

theorem largest_x_value (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + x * z + y * z = 9) : x ≤ 4 := 
sorry

end largest_x_value_l2060_206073


namespace solve_fiftieth_term_l2060_206027

variable (a₇ a₂₁ : ℤ) (d : ℚ)

-- The conditions stated in the problem
def seventh_term : a₇ = 10 := by sorry
def twenty_first_term : a₂₁ = 34 := by sorry

-- The fifty term calculation assuming the common difference d
def fiftieth_term_is_fraction (d : ℚ) : ℚ := 10 + 43 * d

-- Translate the condition a₂₁ = a₇ + 14 * d
theorem solve_fiftieth_term : a₂₁ = a₇ + 14 * d → 
                              fiftieth_term_is_fraction d = 682 / 7 := by sorry


end solve_fiftieth_term_l2060_206027


namespace olivia_grocery_cost_l2060_206007

theorem olivia_grocery_cost :
  let cost_bananas := 12
  let cost_bread := 9
  let cost_milk := 7
  let cost_apples := 14
  cost_bananas + cost_bread + cost_milk + cost_apples = 42 :=
by
  rfl

end olivia_grocery_cost_l2060_206007


namespace least_pos_int_with_12_pos_factors_is_72_l2060_206011

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l2060_206011


namespace book_cost_l2060_206066

-- Definitions from conditions
def priceA : ℝ := 340
def priceB : ℝ := 350
def gain_percent_more : ℝ := 0.05

-- proof problem
theorem book_cost (C : ℝ) (G : ℝ) :
  (priceA - C = G) →
  (priceB - C = (1 + gain_percent_more) * G) →
  C = 140 :=
by
  intros
  sorry

end book_cost_l2060_206066


namespace base_length_of_parallelogram_l2060_206081

theorem base_length_of_parallelogram (A h : ℝ) (hA : A = 44) (hh : h = 11) :
  ∃ b : ℝ, b = 4 ∧ A = b * h :=
by
  sorry

end base_length_of_parallelogram_l2060_206081


namespace triangle_middle_side_at_least_sqrt_two_l2060_206034

theorem triangle_middle_side_at_least_sqrt_two
    (a b c : ℝ)
    (h1 : a ≥ b) (h2 : b ≥ c)
    (h3 : ∃ α : ℝ, 0 < α ∧ α < π ∧ 1 = 1/2 * b * c * Real.sin α) :
  b ≥ Real.sqrt 2 :=
sorry

end triangle_middle_side_at_least_sqrt_two_l2060_206034


namespace jeep_initial_distance_l2060_206010

theorem jeep_initial_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 4 → D / t = 103.33 * (3 / 8)) :
  D = 275.55 :=
sorry

end jeep_initial_distance_l2060_206010


namespace first_machine_rate_l2060_206098

theorem first_machine_rate (x : ℝ) (h : (x + 55) * 30 = 2400) : x = 25 :=
by
  sorry

end first_machine_rate_l2060_206098


namespace train_speed_l2060_206028

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) (h_length : length = 975) (h_time : time = 48) (h_speed : speed = length / time * 3.6) : 
  speed = 73.125 := 
by 
  sorry

end train_speed_l2060_206028


namespace two_roots_iff_a_greater_than_neg1_l2060_206005

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l2060_206005


namespace smallest_odd_prime_factor_2021_8_plus_1_l2060_206091

noncomputable def least_odd_prime_factor (n : ℕ) : ℕ :=
  if 2021^8 + 1 = 0 then 2021^8 + 1 else sorry 

theorem smallest_odd_prime_factor_2021_8_plus_1 :
  least_odd_prime_factor (2021^8 + 1) = 97 :=
  by
    sorry

end smallest_odd_prime_factor_2021_8_plus_1_l2060_206091


namespace fraction_equivalence_l2060_206083

-- Given fractions
def frac1 : ℚ := 3 / 7
def frac2 : ℚ := 4 / 5
def frac3 : ℚ := 5 / 12
def frac4 : ℚ := 2 / 9

-- Expectation
def result : ℚ := 1548 / 805

-- Theorem to prove the equality
theorem fraction_equivalence : ((frac1 + frac2) / (frac3 + frac4)) = result := by
  sorry

end fraction_equivalence_l2060_206083


namespace base8_subtraction_correct_l2060_206067

theorem base8_subtraction_correct :
  ∀ (a b : ℕ) (h1 : a = 7534) (h2 : b = 3267),
      (a - b) % 8 = 4243 % 8 := by
  sorry

end base8_subtraction_correct_l2060_206067


namespace octal_to_decimal_l2060_206047

theorem octal_to_decimal (d0 d1 : ℕ) (n8 : ℕ) (n10 : ℕ) 
  (h1 : d0 = 3) (h2 : d1 = 5) (h3 : n8 = 53) (h4 : n10 = 43) : 
  (d1 * 8^1 + d0 * 8^0 = n10) :=
by
  sorry

end octal_to_decimal_l2060_206047


namespace cat_weight_l2060_206002

theorem cat_weight 
  (weight1 weight2 : ℕ)
  (total_weight : ℕ)
  (h1 : weight1 = 2)
  (h2 : weight2 = 7)
  (h3 : total_weight = 13) : 
  ∃ weight3 : ℕ, weight3 = 4 := 
by
  sorry

end cat_weight_l2060_206002


namespace min_value_of_reciprocal_sum_l2060_206057

variables {m n : ℝ}
variables (h1 : m > 0)
variables (h2 : n > 0)
variables (h3 : m + n = 1)

theorem min_value_of_reciprocal_sum : 
  (1 / m + 1 / n) = 4 :=
by
  sorry

end min_value_of_reciprocal_sum_l2060_206057


namespace sector_area_l2060_206074

theorem sector_area (theta l : ℝ) (h_theta : theta = 2) (h_l : l = 2) :
    let r := l / theta
    let S := 1 / 2 * l * r
    S = 1 := by
  sorry

end sector_area_l2060_206074


namespace investment_accumulation_l2060_206053

variable (P : ℝ) -- Initial investment amount
variable (r1 r2 r3 : ℝ) -- Interest rates for the first 3 years
variable (r4 : ℝ) -- Interest rate for the fourth year
variable (r5 : ℝ) -- Interest rate for the fifth year

-- Conditions
def conditions : Prop :=
  r1 = 0.07 ∧ 
  r2 = 0.08 ∧
  r3 = 0.10 ∧
  r4 = r3 + r3 * 0.12 ∧
  r5 = r4 - r4 * 0.08

-- The accumulated amount after 5 years
def accumulated_amount : ℝ :=
  P * (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- Proof problem
theorem investment_accumulation (P : ℝ) :
  conditions r1 r2 r3 r4 r5 → 
  accumulated_amount P r1 r2 r3 r4 r5 = 1.8141 * P := by
  sorry

end investment_accumulation_l2060_206053


namespace daniel_sales_tax_l2060_206009

theorem daniel_sales_tax :
  let total_cost := 25
  let tax_rate := 0.05
  let tax_free_cost := 18.7
  let tax_paid := 0.3
  exists (taxable_cost : ℝ), 
    18.7 + taxable_cost + 0.05 * taxable_cost = total_cost ∧
    taxable_cost * tax_rate = tax_paid :=
by
  sorry

end daniel_sales_tax_l2060_206009


namespace james_total_cost_l2060_206075

def milk_cost : ℝ := 4.50
def milk_tax_rate : ℝ := 0.20
def banana_cost : ℝ := 3.00
def banana_tax_rate : ℝ := 0.15
def baguette_cost : ℝ := 2.50
def baguette_tax_rate : ℝ := 0.0
def cereal_cost : ℝ := 6.00
def cereal_discount_rate : ℝ := 0.20
def cereal_tax_rate : ℝ := 0.12
def eggs_cost : ℝ := 3.50
def eggs_coupon : ℝ := 1.00
def eggs_tax_rate : ℝ := 0.18

theorem james_total_cost :
  let milk_total := milk_cost * (1 + milk_tax_rate)
  let banana_total := banana_cost * (1 + banana_tax_rate)
  let baguette_total := baguette_cost * (1 + baguette_tax_rate)
  let cereal_discounted := cereal_cost * (1 - cereal_discount_rate)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_cost - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)
  milk_total + banana_total + baguette_total + cereal_total + eggs_total = 19.68 := 
by
  sorry

end james_total_cost_l2060_206075


namespace smallest_integer_solution_l2060_206061

theorem smallest_integer_solution :
  ∃ y : ℤ, (5 / 8 < (y - 3) / 19) ∧ ∀ z : ℤ, (5 / 8 < (z - 3) / 19) → y ≤ z :=
sorry

end smallest_integer_solution_l2060_206061


namespace peter_erasers_l2060_206003

theorem peter_erasers (initial_erasers : ℕ) (extra_erasers : ℕ) (final_erasers : ℕ)
  (h1 : initial_erasers = 8) (h2 : extra_erasers = 3) : final_erasers = 11 :=
by
  sorry

end peter_erasers_l2060_206003


namespace least_common_multiple_first_ten_integers_l2060_206063

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l2060_206063


namespace intersection_of_prime_and_even_is_two_l2060_206087

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 2 * k

theorem intersection_of_prime_and_even_is_two :
  {n : ℕ | is_prime n} ∩ {n : ℕ | is_even n} = {2} :=
by
  sorry

end intersection_of_prime_and_even_is_two_l2060_206087


namespace evaluate_expression_l2060_206013

theorem evaluate_expression :
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  sorry

end evaluate_expression_l2060_206013


namespace total_trees_planted_l2060_206049

theorem total_trees_planted :
  let fourth_graders := 30
  let fifth_graders := 2 * fourth_graders
  let sixth_graders := 3 * fifth_graders - 30
  fourth_graders + fifth_graders + sixth_graders = 240 :=
by
  sorry

end total_trees_planted_l2060_206049


namespace sum_of_first_six_terms_l2060_206001

theorem sum_of_first_six_terms 
  {S : ℕ → ℝ} 
  (h_arith_seq : ∀ n, S n = n * (-2) + (n * (n - 1) * 3 ))
  (S_2_eq_2 : S 2 = 2)
  (S_4_eq_10 : S 4 = 10) : S 6 = 18 := 
  sorry

end sum_of_first_six_terms_l2060_206001


namespace altitude_change_correct_l2060_206052

noncomputable def altitude_change (T_ground T_high : ℝ) (deltaT_per_km : ℝ) : ℝ :=
  (T_high - T_ground) / deltaT_per_km

theorem altitude_change_correct :
  altitude_change 18 (-48) (-6) = 11 :=
by 
  sorry

end altitude_change_correct_l2060_206052


namespace not_divisible_by_pow_two_l2060_206008

theorem not_divisible_by_pow_two (n : ℕ) (h : n > 1) : ¬ (2^n ∣ (3^n + 1)) :=
by
  sorry

end not_divisible_by_pow_two_l2060_206008


namespace drink_all_tea_l2060_206021

theorem drink_all_tea (cups : Fin 30 → Prop) (red blue : Fin 30 → Prop)
  (h₀ : ∀ n, cups n ↔ (red n ↔ ¬ blue n))
  (h₁ : ∃ a b, a ≠ b ∧ red a ∧ blue b)
  (h₂ : ∀ n, red n → red (n + 2))
  (h₃ : ∀ n, blue n → blue (n + 2)) :
  ∃ sequence : ℕ → Fin 30, (∀ n, cups (sequence n)) ∧ (sequence 0 ≠ sequence 1) 
  ∧ (∀ n, cups (sequence (n+1))) :=
by
  sorry

end drink_all_tea_l2060_206021


namespace max_power_sum_l2060_206068

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l2060_206068


namespace infinite_slips_have_repeated_numbers_l2060_206004

theorem infinite_slips_have_repeated_numbers
  (slips : Set ℕ) (h_inf_slips : slips.Infinite)
  (h_sub_infinite_imp_repeats : ∀ s : Set ℕ, s.Infinite → ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ x = y) :
  ∃ n : ℕ, {x ∈ slips | x = n}.Infinite :=
by sorry

end infinite_slips_have_repeated_numbers_l2060_206004


namespace penny_frogs_count_l2060_206076

theorem penny_frogs_count :
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  tree_frogs + poison_frogs + wood_frogs = 78 :=
by
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  show tree_frogs + poison_frogs + wood_frogs = 78
  sorry

end penny_frogs_count_l2060_206076


namespace profit_calculation_l2060_206012

def actors_cost : ℕ := 1200
def people_count : ℕ := 50
def cost_per_person : ℕ := 3
def food_cost : ℕ := people_count * cost_per_person
def total_cost_actors_food : ℕ := actors_cost + food_cost
def equipment_rental_cost : ℕ := 2 * total_cost_actors_food
def total_movie_cost : ℕ := total_cost_actors_food + equipment_rental_cost
def movie_sale_price : ℕ := 10000
def profit : ℕ := movie_sale_price - total_movie_cost

theorem profit_calculation : profit = 5950 := by
  sorry

end profit_calculation_l2060_206012


namespace find_number_l2060_206017

theorem find_number (x : ℝ) (h : x - x / 3 = x - 24) : x = 72 := 
by 
  sorry

end find_number_l2060_206017
