import Mathlib

namespace dana_more_pencils_than_marcus_l2074_207498

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l2074_207498


namespace red_pens_count_l2074_207442

theorem red_pens_count (R : ℕ) : 
  (∃ (black_pens blue_pens : ℕ), 
  black_pens = R + 10 ∧ 
  blue_pens = R + 7 ∧ 
  R + black_pens + blue_pens = 41) → 
  R = 8 := by
  sorry

end red_pens_count_l2074_207442


namespace initial_mean_calculated_l2074_207441

theorem initial_mean_calculated (M : ℝ) (h1 : 25 * M - 35 = 25 * 191.4 - 35) : M = 191.4 := 
  sorry

end initial_mean_calculated_l2074_207441


namespace rectangle_max_area_l2074_207482

theorem rectangle_max_area (w : ℝ) (h : ℝ) (hw : h = 2 * w) (perimeter : 2 * (w + h) = 40) :
  w * h = 800 / 9 := 
by
  -- Given: h = 2w and 2(w + h) = 40
  -- We need to prove that the area A = wh = 800/9
  sorry

end rectangle_max_area_l2074_207482


namespace inequality_proof_l2074_207423

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) : 
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := 
sorry

end inequality_proof_l2074_207423


namespace complement_of_M_in_U_l2074_207420

def universal_set : Set ℝ := {x | x > 0}
def set_M : Set ℝ := {x | x > 1}
def complement (U M : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ M}

theorem complement_of_M_in_U :
  complement universal_set set_M = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end complement_of_M_in_U_l2074_207420


namespace solution_to_problem_l2074_207413

theorem solution_to_problem (f : ℕ → ℕ) 
  (h1 : f 2 = 20)
  (h2 : ∀ n : ℕ, 0 < n → f (2 * n) + n * f 2 = f (2 * n + 2)) :
  f 10 = 220 :=
by
  sorry

end solution_to_problem_l2074_207413


namespace find_k_l2074_207467

theorem find_k (x y k : ℝ) (h1 : 2 * x + y = 4 * k) (h2 : x - y = k) (h3 : x + 2 * y = 12) : k = 4 :=
sorry

end find_k_l2074_207467


namespace cost_of_fencing_per_meter_l2074_207495

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end cost_of_fencing_per_meter_l2074_207495


namespace sector_area_is_nine_l2074_207440

-- Defining the given conditions
def arc_length (r θ : ℝ) : ℝ := r * θ
def sector_area (r θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Given conditions
variables (r : ℝ) (θ : ℝ)
variable (h1 : arc_length r θ = 6)
variable (h2 : θ = 2)

-- Goal: Prove that the area of the sector is 9
theorem sector_area_is_nine : sector_area r θ = 9 := by
  sorry

end sector_area_is_nine_l2074_207440


namespace volume_of_prism_l2074_207447

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 48) (h3 : b * c = 72) : a * b * c = 168 :=
by
  sorry

end volume_of_prism_l2074_207447


namespace find_a_l2074_207450

theorem find_a (a : ℝ) (A B : Set ℝ)
    (hA : A = {a^2, a + 1, -3})
    (hB : B = {a - 3, 2 * a - 1, a^2 + 1}) 
    (h : A ∩ B = {-3}) : a = -1 := by
  sorry

end find_a_l2074_207450


namespace solve_k_equality_l2074_207475

noncomputable def collinear_vectors (e1 e2 : ℝ) (k : ℝ) (AB CB CD : ℝ) : Prop := 
  let BD := (2 * e1 - e2) - (e1 + 3 * e2)
  BD = e1 - 4 * e2 ∧ AB = 2 * e1 + k * e2 ∧ AB = k * BD
  
theorem solve_k_equality (e1 e2 k AB CB CD : ℝ) (h_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0)) :
  collinear_vectors e1 e2 k AB CB CD → k = -8 :=
by
  intro h_collinear
  sorry

end solve_k_equality_l2074_207475


namespace smaller_angle_at_7_15_l2074_207491

theorem smaller_angle_at_7_15 (h_angle : ℝ) (m_angle : ℝ) : 
  h_angle = 210 + 0.5 * 15 →
  m_angle = 90 →
  min (abs (h_angle - m_angle)) (360 - abs (h_angle - m_angle)) = 127.5 :=
  by
    intros h_eq m_eq
    rw [h_eq, m_eq]
    sorry

end smaller_angle_at_7_15_l2074_207491


namespace binomial_sum_equal_36_l2074_207469

theorem binomial_sum_equal_36 (n : ℕ) (h : n > 0) :
  (n + n * (n - 1) / 2 = 36) → n = 8 :=
by
  sorry

end binomial_sum_equal_36_l2074_207469


namespace monkey_slips_2_feet_each_hour_l2074_207477

/-- 
  A monkey climbs a 17 ft tree, hopping 3 ft and slipping back a certain distance each hour.
  The monkey takes 15 hours to reach the top. Prove that the monkey slips back 2 feet each hour.
-/
def monkey_slips_back_distance (s : ℝ) : Prop :=
  ∃ s : ℝ, (14 * (3 - s) + 3 = 17) ∧ s = 2

theorem monkey_slips_2_feet_each_hour : monkey_slips_back_distance 2 := by
  -- Sorry, proof omitted
  sorry

end monkey_slips_2_feet_each_hour_l2074_207477


namespace second_discount_percentage_l2074_207436

theorem second_discount_percentage
    (original_price : ℝ)
    (first_discount : ℝ)
    (final_sale_price : ℝ)
    (second_discount : ℝ)
    (h1 : original_price = 390)
    (h2 : first_discount = 14)
    (h3 : final_sale_price = 285.09) :
    second_discount = 15 :=
by
  -- Since we are not providing the full proof, we assume the steps to be correct
  sorry

end second_discount_percentage_l2074_207436


namespace buses_passed_on_highway_l2074_207446

/-- Problem statement:
     Buses from Dallas to Austin leave every hour on the hour.
     Buses from Austin to Dallas leave every two hours, starting at 7:00 AM.
     The trip from one city to the other takes 6 hours.
     Assuming the buses travel on the same highway,
     how many Dallas-bound buses does an Austin-bound bus pass on the highway?
-/
theorem buses_passed_on_highway :
  ∀ (t_depart_A2D : ℕ) (trip_time : ℕ) (buses_departures_D2A : ℕ → ℕ),
  (∀ n, buses_departures_D2A n = n) →
  trip_time = 6 →
  ∃ n, t_depart_A2D = 7 ∧ 
    (∀ t, t_depart_A2D ≤ t ∧ t < t_depart_A2D + trip_time →
      ∃ m, m + 1 = t ∧ buses_departures_D2A (m - 6) ≤ t ∧ t < buses_departures_D2A (m - 6) + 6) ↔ n + 1 = 7 := 
sorry

end buses_passed_on_highway_l2074_207446


namespace successive_numbers_product_2652_l2074_207492

theorem successive_numbers_product_2652 (n : ℕ) (h : n * (n + 1) = 2652) : n = 51 :=
sorry

end successive_numbers_product_2652_l2074_207492


namespace solve_x_l2074_207460

theorem solve_x :
  (2 / 3 - 1 / 4) = 1 / (12 / 5) :=
by
  sorry

end solve_x_l2074_207460


namespace AlyssaBottleCaps_l2074_207449

def bottleCapsKatherine := 34
def bottleCapsGivenAway (bottleCaps: ℕ) := bottleCaps / 2
def bottleCapsLost (bottleCaps: ℕ) := bottleCaps - 8

theorem AlyssaBottleCaps : bottleCapsLost (bottleCapsGivenAway bottleCapsKatherine) = 9 := 
  by 
  sorry

end AlyssaBottleCaps_l2074_207449


namespace distance_inequality_solution_l2074_207484

theorem distance_inequality_solution (x : ℝ) (h : |x| > |x + 1|) : x < -1 / 2 :=
sorry

end distance_inequality_solution_l2074_207484


namespace correct_expression_l2074_207476

-- Definitions for the problem options.
def optionA (m n : ℕ) : ℕ := 2 * m + n
def optionB (m n : ℕ) : ℕ := m + 2 * n
def optionC (m n : ℕ) : ℕ := 2 * (m + n)
def optionD (m n : ℕ) : ℕ := (m + n) ^ 2

-- Statement for the proof problem.
theorem correct_expression (m n : ℕ) : optionB m n = m + 2 * n :=
by sorry

end correct_expression_l2074_207476


namespace number_of_buses_proof_l2074_207438

-- Define the conditions
def columns_per_bus : ℕ := 4
def rows_per_bus : ℕ := 10
def total_students : ℕ := 240
def seats_per_bus (c : ℕ) (r : ℕ) : ℕ := c * r
def number_of_buses (total : ℕ) (seats : ℕ) : ℕ := total / seats

-- State the theorem we want to prove
theorem number_of_buses_proof :
  number_of_buses total_students (seats_per_bus columns_per_bus rows_per_bus) = 6 := 
sorry

end number_of_buses_proof_l2074_207438


namespace sunset_time_correct_l2074_207457

theorem sunset_time_correct : 
  let sunrise := (6 * 60 + 43)       -- Sunrise time in minutes (6:43 AM)
  let daylight := (11 * 60 + 56)     -- Length of daylight in minutes (11:56)
  let sunset := (sunrise + daylight) % (24 * 60) -- Calculate sunset time considering 24-hour cycle
  let sunset_hour := sunset / 60     -- Convert sunset time back into hours
  let sunset_minute := sunset % 60   -- Calculate remaining minutes
  (sunset_hour - 12, sunset_minute) = (6, 39)    -- Convert to 12-hour format and check against 6:39 PM
:= by
  sorry

end sunset_time_correct_l2074_207457


namespace earnings_last_friday_l2074_207430

theorem earnings_last_friday 
  (price_per_kg : ℕ := 2)
  (earnings_wednesday : ℕ := 30)
  (earnings_today : ℕ := 42)
  (total_kg_sold : ℕ := 48)
  (total_earnings : ℕ := total_kg_sold * price_per_kg) 
  (F : ℕ) :
  earnings_wednesday + F + earnings_today = total_earnings → F = 24 := by
  sorry

end earnings_last_friday_l2074_207430


namespace train_length_l2074_207400

theorem train_length 
  (V : ℝ → ℝ) (L : ℝ) 
  (length_of_train : ∀ (t : ℝ), t = 8 → V t = L / 8) 
  (pass_platform : ∀ (d t : ℝ), d = L + 273 → t = 20 → V t = d / t) 
  : L = 182 := 
by
  sorry

end train_length_l2074_207400


namespace find_numbers_with_lcm_gcd_l2074_207485

theorem find_numbers_with_lcm_gcd :
  ∃ a b : ℕ, lcm a b = 90 ∧ gcd a b = 6 ∧ ((a = 18 ∧ b = 30) ∨ (a = 30 ∧ b = 18)) :=
by
  sorry

end find_numbers_with_lcm_gcd_l2074_207485


namespace min_red_chips_l2074_207459

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ (1 / 3) * w)
  (h2 : b ≤ (1 / 4) * r)
  (h3 : w + b ≥ 70) : r ≥ 72 :=
by
  sorry

end min_red_chips_l2074_207459


namespace calculate_percentage_increase_l2074_207419

variable (fish_first_round : ℕ) (fish_second_round : ℕ) (fish_total : ℕ) (fish_last_round : ℕ) (increase : ℚ) (percentage_increase : ℚ)

theorem calculate_percentage_increase
  (h1 : fish_first_round = 8)
  (h2 : fish_second_round = fish_first_round + 12)
  (h3 : fish_total = 60)
  (h4 : fish_last_round = fish_total - (fish_first_round + fish_second_round))
  (h5 : increase = fish_last_round - fish_second_round)
  (h6 : percentage_increase = (increase / fish_second_round) * 100) :
  percentage_increase = 60 := by
  sorry

end calculate_percentage_increase_l2074_207419


namespace double_pythagorean_triple_l2074_207422

theorem double_pythagorean_triple (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  (2*a)^2 + (2*b)^2 = (2*c)^2 :=
by
  sorry

end double_pythagorean_triple_l2074_207422


namespace intersection_x_value_l2074_207451

theorem intersection_x_value:
  ∃ x y : ℝ, y = 4 * x - 29 ∧ 3 * x + y = 105 ∧ x = 134 / 7 :=
by
  sorry

end intersection_x_value_l2074_207451


namespace teresa_speed_l2074_207412

def distance : ℝ := 25 -- kilometers
def time : ℝ := 5 -- hours

theorem teresa_speed :
  (distance / time) = 5 := by
  sorry

end teresa_speed_l2074_207412


namespace linear_function_does_not_pass_third_quadrant_l2074_207472

/-
Given an inverse proportion function \( y = \frac{a^2 + 1}{x} \), where \( a \) is a constant, and given two points \( (x_1, y_1) \) and \( (x_2, y_2) \) on the same branch of this function, 
with \( b = (x_1 - x_2)(y_1 - y_2) \), prove that the graph of the linear function \( y = bx - b \) does not pass through the third quadrant.
-/

theorem linear_function_does_not_pass_third_quadrant 
  (a x1 x2 : ℝ) 
  (y1 y2 : ℝ)
  (h1 : y1 = (a^2 + 1) / x1) 
  (h2 : y2 = (a^2 + 1) / x2) 
  (h3 : b = (x1 - x2) * (y1 - y2)) : 
  ∃ b, ∀ x y : ℝ, (y = b * x - b) → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) :=
by 
  sorry

end linear_function_does_not_pass_third_quadrant_l2074_207472


namespace find_x_minus_4y_l2074_207424

theorem find_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) : x - 4 * y = 5 :=
by 
  sorry

end find_x_minus_4y_l2074_207424


namespace find_base_b_l2074_207462

theorem find_base_b : ∃ b : ℕ, b > 4 ∧ (b + 2)^2 = b^2 + 4 * b + 4 ∧ b = 5 := 
sorry

end find_base_b_l2074_207462


namespace tan_alpha_value_l2074_207496

open Real

variable (α : ℝ)

/- Conditions -/
def alpha_interval : Prop := (0 < α) ∧ (α < π)
def sine_cosine_sum : Prop := sin α + cos α = -7 / 13

/- Statement -/
theorem tan_alpha_value 
  (h1 : alpha_interval α)
  (h2 : sine_cosine_sum α) : 
  tan α = -5 / 12 :=
sorry

end tan_alpha_value_l2074_207496


namespace rational_squares_solution_l2074_207403

theorem rational_squares_solution {x y u v : ℕ} (x_pos : 0 < x) (y_pos : 0 < y) (u_pos : 0 < u) (v_pos : 0 < v) 
  (h1 : ∃ q : ℚ, q = (Real.sqrt (x * y) + Real.sqrt (u * v))) 
  (h2 : |(x / 9 : ℚ) - (y / 4 : ℚ)| = |(u / 3 : ℚ) - (v / 12 : ℚ)| ∧ |(u / 3 : ℚ) - (v / 12 : ℚ)| = u * v - x * y) :
  ∃ k : ℕ, x = 9 * k ∧ y = 4 * k ∧ u = 3 * k ∧ v = 12 * k := by
  sorry

end rational_squares_solution_l2074_207403


namespace greatest_x_l2074_207437

theorem greatest_x (x : ℕ) : (x^6 / x^3 ≤ 27) → x ≤ 3 :=
by sorry

end greatest_x_l2074_207437


namespace part_I_part_II_l2074_207465

def S_n (n : ℕ) : ℕ := sorry
def a_n (n : ℕ) : ℕ := sorry

theorem part_I (n : ℕ) (h1 : 2 * S_n n = 3^n + 3) :
  a_n n = if n = 1 then 3 else 3^(n-1) :=
sorry

theorem part_II (n : ℕ) (h1 : a_n 1 = 1) (h2 : ∀ n : ℕ, a_n (n + 1) - a_n n = 2^n) :
  S_n n = 2^(n + 1) - n - 2 :=
sorry

end part_I_part_II_l2074_207465


namespace calc_g_g_neg3_l2074_207489

def g (x : ℚ) : ℚ :=
x⁻¹ + x⁻¹ / (2 + x⁻¹)

theorem calc_g_g_neg3 : g (g (-3)) = -135 / 8 := 
by
  sorry

end calc_g_g_neg3_l2074_207489


namespace balloons_left_after_distribution_l2074_207453

theorem balloons_left_after_distribution :
  (22 + 40 + 70 + 90) % 10 = 2 := by
  sorry

end balloons_left_after_distribution_l2074_207453


namespace claire_hours_cleaning_l2074_207490

-- Definitions of given conditions
def total_hours_in_day : ℕ := 24
def hours_sleeping : ℕ := 8
def hours_cooking : ℕ := 2
def hours_crafting : ℕ := 5
def total_working_hours : ℕ := total_hours_in_day - hours_sleeping

-- Definition of the question
def hours_cleaning := total_working_hours - (hours_cooking + hours_crafting + hours_crafting)

-- The proof goal
theorem claire_hours_cleaning : hours_cleaning = 4 := by
  sorry

end claire_hours_cleaning_l2074_207490


namespace value_of_x_squared_plus_y_squared_l2074_207425

theorem value_of_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
sorry

end value_of_x_squared_plus_y_squared_l2074_207425


namespace radius_of_bicycle_wheel_is_13_l2074_207421

-- Define the problem conditions
def diameter_cm : ℕ := 26

-- Define the function to calculate radius from diameter
def radius (d : ℕ) : ℕ := d / 2

-- Prove that the radius is 13 cm when diameter is 26 cm
theorem radius_of_bicycle_wheel_is_13 :
  radius diameter_cm = 13 := 
sorry

end radius_of_bicycle_wheel_is_13_l2074_207421


namespace revenue_95_percent_l2074_207427

-- Definitions based on the conditions
variables (C : ℝ) (n : ℝ)
def revenue_full : ℝ := 1.20 * C
def tickets_sold_percentage : ℝ := 0.95

-- Statement of the theorem based on the problem translation
theorem revenue_95_percent (C : ℝ) :
  (tickets_sold_percentage * revenue_full C) = 1.14 * C :=
by
  sorry -- Proof to be provided

end revenue_95_percent_l2074_207427


namespace hcf_of_12_and_15_l2074_207434

-- Definitions of LCM and HCF
def LCM (a b : ℕ) : ℕ := sorry  -- Placeholder for actual LCM definition
def HCF (a b : ℕ) : ℕ := sorry  -- Placeholder for actual HCF definition

theorem hcf_of_12_and_15 :
  LCM 12 15 = 60 → HCF 12 15 = 3 :=
by
  sorry

end hcf_of_12_and_15_l2074_207434


namespace number_of_dress_designs_is_correct_l2074_207414

-- Define the number of choices for colors, patterns, and fabric types as conditions
def num_colors : Nat := 4
def num_patterns : Nat := 5
def num_fabric_types : Nat := 2

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_patterns * num_fabric_types

-- Prove that the total number of different dress designs is 40
theorem number_of_dress_designs_is_correct : total_dress_designs = 40 := by
  sorry

end number_of_dress_designs_is_correct_l2074_207414


namespace roots_mul_shift_eq_neg_2018_l2074_207471

theorem roots_mul_shift_eq_neg_2018 {a b : ℝ}
  (h1 : a + b = -1)
  (h2 : a * b = -2020) :
  (a - 1) * (b - 1) = -2018 :=
sorry

end roots_mul_shift_eq_neg_2018_l2074_207471


namespace divides_difference_l2074_207439

theorem divides_difference (n : ℕ) (h_composite : ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k) : 
  6 ∣ ((n^2)^3 - n^2) := 
sorry

end divides_difference_l2074_207439


namespace hairstylist_monthly_earnings_l2074_207409

noncomputable def hairstylist_earnings_per_month : ℕ :=
  let monday_wednesday_friday_earnings : ℕ := (4 * 10) + (3 * 15) + (1 * 22);
  let tuesday_thursday_earnings : ℕ := (6 * 10) + (2 * 15) + (3 * 30);
  let weekend_earnings : ℕ := (10 * 22) + (5 * 30);
  let weekly_earnings : ℕ :=
    (monday_wednesday_friday_earnings * 3) +
    (tuesday_thursday_earnings * 2) +
    (weekend_earnings * 2);
  weekly_earnings * 4

theorem hairstylist_monthly_earnings : hairstylist_earnings_per_month = 5684 := by
  -- Assertion based on the provided problem conditions
  sorry

end hairstylist_monthly_earnings_l2074_207409


namespace LindasOriginalSavings_l2074_207499

theorem LindasOriginalSavings : 
  (∃ S : ℝ, (1 / 4) * S = 200) ∧ 
  (3 / 4) * S = 600 ∧ 
  (∀ F : ℝ, 0.80 * F = 600 → F = 750) → 
  S = 800 :=
by
  sorry

end LindasOriginalSavings_l2074_207499


namespace laptop_price_l2074_207401

theorem laptop_price (x : ℝ) : 
  (0.8 * x - 120) = 0.9 * x - 64 → x = 560 :=
by
  sorry

end laptop_price_l2074_207401


namespace larger_number_is_eight_l2074_207426

theorem larger_number_is_eight (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l2074_207426


namespace function_defined_for_all_reals_l2074_207464

theorem function_defined_for_all_reals (m : ℝ) :
  (∀ x : ℝ, 7 * x ^ 2 + m - 6 ≠ 0) → m > 6 :=
by
  sorry

end function_defined_for_all_reals_l2074_207464


namespace nested_expression_value_l2074_207454

theorem nested_expression_value : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))) = 87380 :=
by 
  sorry

end nested_expression_value_l2074_207454


namespace no_such_integers_x_y_l2074_207466

theorem no_such_integers_x_y (x y : ℤ) : x^2 + 1974 ≠ y^2 := by
  sorry

end no_such_integers_x_y_l2074_207466


namespace bus_dispatch_interval_l2074_207407

/--
Xiao Hua walks at a constant speed along the route of the "Chunlei Cup" bus.
He encounters a "Chunlei Cup" bus every 6 minutes head-on and is overtaken by a "Chunlei Cup" bus every 12 minutes.
Assume "Chunlei Cup" buses are dispatched at regular intervals, travel at a constant speed, and do not stop at any stations along the way.
Prove that the time interval between bus departures is 8 minutes.
-/
theorem bus_dispatch_interval
  (encounters_opposite_direction: ℕ)
  (overtakes_same_direction: ℕ)
  (constant_speed: Prop)
  (regular_intervals: Prop)
  (no_stops: Prop)
  (h1: encounters_opposite_direction = 6)
  (h2: overtakes_same_direction = 12)
  (h3: constant_speed)
  (h4: regular_intervals)
  (h5: no_stops) :
  True := 
sorry

end bus_dispatch_interval_l2074_207407


namespace solve_system_of_equations_l2074_207470

theorem solve_system_of_equations (x y : ℤ) (h1 : x + y = 8) (h2 : x - 3 * y = 4) : x = 7 ∧ y = 1 :=
by {
    -- Proof would go here
    sorry
}

end solve_system_of_equations_l2074_207470


namespace translated_parabola_correct_l2074_207497

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := x^2 + 2

-- Theorem stating that translating the original parabola up by 2 units results in the translated parabola
theorem translated_parabola_correct (x : ℝ) :
  translated_parabola x = original_parabola x + 2 :=
by
  sorry

end translated_parabola_correct_l2074_207497


namespace conference_fraction_married_men_l2074_207418

theorem conference_fraction_married_men 
  (total_women : ℕ) 
  (single_probability : ℚ) 
  (h_single_prob : single_probability = 3/7) 
  (h_total_women : total_women = 7) : 
  (4 : ℚ) / (11 : ℚ) = 4 / 11 := 
by
  sorry

end conference_fraction_married_men_l2074_207418


namespace number_of_pebbles_l2074_207461

theorem number_of_pebbles (P : ℕ) : 
  (P * (1/4 : ℝ) + 3 * (1/2 : ℝ) + 2 * 2 = 7) → P = 6 := by
  sorry

end number_of_pebbles_l2074_207461


namespace pencils_in_each_box_l2074_207456

theorem pencils_in_each_box (n : ℕ) (h : 10 * n - 10 = 40) : n = 5 := by
  sorry

end pencils_in_each_box_l2074_207456


namespace coefficient_a_for_factor_l2074_207478

noncomputable def P (a : ℚ) (x : ℚ) : ℚ := x^3 + 2 * x^2 + a * x + 20

theorem coefficient_a_for_factor (a : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ P a x) → a = -65/3 :=
by
  sorry

end coefficient_a_for_factor_l2074_207478


namespace sum_of_tens_and_ones_digits_pow_l2074_207404

theorem sum_of_tens_and_ones_digits_pow : 
  let n := 7
  let exp := 12
  (n^exp % 100) / 10 + (n^exp % 10) = 1 :=
by
  sorry

end sum_of_tens_and_ones_digits_pow_l2074_207404


namespace expression_simplification_l2074_207408

-- Definitions for P and Q based on x and y
def P (x y : ℝ) := x + y
def Q (x y : ℝ) := x - y

-- The mathematical property to prove
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (x^2 - y^2) / (x * y) := 
by
  -- Sorry is used to skip the proof here
  sorry

end expression_simplification_l2074_207408


namespace max_marks_l2074_207452

theorem max_marks (M : ℝ) (h_pass : 0.33 * M = 165) : M = 500 := 
by
  sorry

end max_marks_l2074_207452


namespace count_valid_pairs_l2074_207405

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 5 ∧ 
  ∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 40 →
  (5^j - 2^i) % 1729 = 0 →
  i = 0 ∧ j = 36 ∨ 
  i = 1 ∧ j = 37 ∨ 
  i = 2 ∧ j = 38 ∨ 
  i = 3 ∧ j = 39 ∨ 
  i = 4 ∧ j = 40 :=
by
  sorry

end count_valid_pairs_l2074_207405


namespace there_exists_l_l2074_207473

theorem there_exists_l (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≠ 0) 
  (h3 : ∀ k : ℕ, 0 < k → Nat.gcd (17 * k - 1) m = Nat.gcd (17 * k - 1) n) :
  ∃ l : ℤ, m = (17 : ℕ) ^ l.natAbs * n := 
sorry

end there_exists_l_l2074_207473


namespace impossible_coins_l2074_207486

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l2074_207486


namespace difference_shares_l2074_207493

-- Given conditions in the problem
variable (V : ℕ) (F R : ℕ)
variable (hV : V = 1500)
variable (hRatioF : F = 3 * (V / 5))
variable (hRatioR : R = 11 * (V / 5))

-- The statement we need to prove
theorem difference_shares : R - F = 2400 :=
by
  -- Using the conditions to derive the result.
  sorry

end difference_shares_l2074_207493


namespace fair_tickets_more_than_twice_baseball_tickets_l2074_207458

theorem fair_tickets_more_than_twice_baseball_tickets :
  ∃ (fair_tickets baseball_tickets : ℕ), 
    fair_tickets = 25 ∧ baseball_tickets = 56 ∧ 
    fair_tickets + 87 = 2 * baseball_tickets := 
by
  sorry

end fair_tickets_more_than_twice_baseball_tickets_l2074_207458


namespace maximum_value_of_2x_plus_y_l2074_207474

noncomputable def max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) : ℝ :=
  (2 * x + y)

theorem maximum_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  max_value_2x_plus_y x y h ≤ (2 * Real.sqrt 10) / 5 :=
sorry

end maximum_value_of_2x_plus_y_l2074_207474


namespace inequality_holds_for_all_x_iff_m_eq_1_l2074_207416

theorem inequality_holds_for_all_x_iff_m_eq_1 (m : ℝ) (h_m : m ≠ 0) :
  (∀ x > 0, x^2 - 2 * m * Real.log x ≥ 1) ↔ m = 1 :=
by
  sorry

end inequality_holds_for_all_x_iff_m_eq_1_l2074_207416


namespace find_third_root_l2074_207483

noncomputable def P (a b x : ℝ) : ℝ := a * x^3 + (a + 4 * b) * x^2 + (b - 5 * a) * x + (10 - a)

theorem find_third_root (a b : ℝ) (h1 : P a b (-1) = 0) (h2 : P a b 4 = 0) : 
 ∃ c : ℝ, c ≠ -1 ∧ c ≠ 4 ∧ P a b c = 0 ∧ c = 8 / 3 :=
 sorry

end find_third_root_l2074_207483


namespace find_k_l2074_207411

theorem find_k (k : ℕ) (hk : k > 0) (h_coeff : 15 * k^4 < 120) : k = 1 := 
by 
  sorry

end find_k_l2074_207411


namespace sequence_arithmetic_l2074_207445

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * n^2 - 2 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (∀ n, a n - a (n - 1) = 4) :=
by
  intros hS ha
  sorry

end sequence_arithmetic_l2074_207445


namespace shaded_region_area_l2074_207415

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end shaded_region_area_l2074_207415


namespace find_number_l2074_207417

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 90) : x = 4000 :=
by
  sorry

end find_number_l2074_207417


namespace vector_subtraction_l2074_207455

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vec_smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_a : ℝ × ℝ := (3, 5)
def vec_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction : vec_sub vec_a (vec_smul 2 vec_b) = (7, 3) :=
by
  sorry

end vector_subtraction_l2074_207455


namespace min_k_valid_l2074_207487

def S : Set ℕ := {1, 2, 3, 4}

def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ b : Fin 4 → ℕ,
    (∀ i : Fin 4, b i ∈ S) ∧ b 3 ≠ 1 →
    ∃ i1 i2 i3 i4 : Fin (k + 1), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
      (a i1 = b 0 ∧ a i2 = b 1 ∧ a i3 = b 2 ∧ a i4 = b 3)

def min_k := 11

theorem min_k_valid : ∀ a : ℕ → ℕ,
  valid_sequence a min_k → 
  min_k = 11 :=
sorry

end min_k_valid_l2074_207487


namespace ceil_minus_x_of_fractional_part_half_l2074_207428

theorem ceil_minus_x_of_fractional_part_half (x : ℝ) (hx : x - ⌊x⌋ = 1 / 2) : ⌈x⌉ - x = 1 / 2 :=
by
 sorry

end ceil_minus_x_of_fractional_part_half_l2074_207428


namespace minimum_value_xy_minimum_value_x_plus_2y_l2074_207443

-- (1) Prove that the minimum value of \(xy\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(36\).
theorem minimum_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x * y ≥ 36 := 
sorry

-- (2) Prove that the minimum value of \(x + 2y\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(19 + 6\sqrt{2}\).
theorem minimum_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

end minimum_value_xy_minimum_value_x_plus_2y_l2074_207443


namespace eval_expr_l2074_207402

theorem eval_expr (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a > b) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a / b)^(b - a) :=
sorry

end eval_expr_l2074_207402


namespace sum_first_3000_terms_l2074_207463

variable {α : Type*}

noncomputable def geometric_sum_1000 (a r : α) [Field α] : α := a * (r ^ 1000 - 1) / (r - 1)
noncomputable def geometric_sum_2000 (a r : α) [Field α] : α := a * (r ^ 2000 - 1) / (r - 1)
noncomputable def geometric_sum_3000 (a r : α) [Field α] : α := a * (r ^ 3000 - 1) / (r - 1)

theorem sum_first_3000_terms 
  {a r : ℝ}
  (h1 : geometric_sum_1000 a r = 1024)
  (h2 : geometric_sum_2000 a r = 2040) :
  geometric_sum_3000 a r = 3048 := 
  sorry

end sum_first_3000_terms_l2074_207463


namespace S_2011_l2074_207406

variable {α : Type*}

-- Define initial term and sum function for arithmetic sequence
def a1 : ℤ := -2011
noncomputable def S (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * 2

-- Given conditions
def condition1 : a1 = -2011 := rfl
def condition2 : (S 2010 / 2010) - (S 2008 / 2008) = 2 := by sorry

-- Proof statement
theorem S_2011 : S 2011 = -2011 := by 
  -- Use the given conditions to prove the statement
  sorry

end S_2011_l2074_207406


namespace total_pieces_of_paper_l2074_207433

/-- Definitions according to the problem's conditions -/
def pieces_after_first_cut : Nat := 10

def pieces_after_second_cut (initial_pieces : Nat) : Nat := initial_pieces + 9

def pieces_after_third_cut (after_second_cut_pieces : Nat) : Nat := after_second_cut_pieces + 9

def pieces_after_fourth_cut (after_third_cut_pieces : Nat) : Nat := after_third_cut_pieces + 9

/-- The main theorem stating the desired result -/
theorem total_pieces_of_paper : 
  pieces_after_fourth_cut (pieces_after_third_cut (pieces_after_second_cut pieces_after_first_cut)) = 37 := 
by 
  -- The proof would go here, but it's omitted as per the instructions.
  sorry

end total_pieces_of_paper_l2074_207433


namespace sqrt_200_eq_l2074_207432

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l2074_207432


namespace num_second_grade_students_is_80_l2074_207481

def ratio_fst : ℕ := 5
def ratio_snd : ℕ := 4
def ratio_trd : ℕ := 3
def total_students : ℕ := 240

def second_grade : ℕ := (ratio_snd * total_students) / (ratio_fst + ratio_snd + ratio_trd)

theorem num_second_grade_students_is_80 :
  second_grade = 80 := 
sorry

end num_second_grade_students_is_80_l2074_207481


namespace base_eight_to_base_ten_l2074_207410

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end base_eight_to_base_ten_l2074_207410


namespace arctan_tan_equiv_l2074_207468

theorem arctan_tan_equiv (h1 : Real.tan (Real.pi / 4 + Real.pi / 12) = 1 / Real.tan (Real.pi / 4 - Real.pi / 3))
  (h2 : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3):
  Real.arctan (Real.tan (5 * Real.pi / 12) - 2 * Real.tan (Real.pi / 6)) = 5 * Real.pi / 12 := 
sorry

end arctan_tan_equiv_l2074_207468


namespace initial_kittens_l2074_207435

theorem initial_kittens (kittens_given : ℕ) (kittens_left : ℕ) (initial_kittens : ℕ) :
  kittens_given = 4 → kittens_left = 4 → initial_kittens = kittens_given + kittens_left → initial_kittens = 8 :=
by
  intros hg hl hi
  rw [hg, hl] at hi
  -- Skipping proof detail
  sorry

end initial_kittens_l2074_207435


namespace factorize_expression_l2074_207448

theorem factorize_expression (x : ℝ) : (x + 3) ^ 2 - (x + 3) = (x + 3) * (x + 2) :=
by
  sorry

end factorize_expression_l2074_207448


namespace paco_ate_more_salty_than_sweet_l2074_207494

-- Define the initial conditions
def sweet_start := 8
def salty_start := 6
def sweet_ate := 20
def salty_ate := 34

-- Define the statement to prove
theorem paco_ate_more_salty_than_sweet : (salty_ate - sweet_ate) = 14 := by
    sorry

end paco_ate_more_salty_than_sweet_l2074_207494


namespace maximum_marks_l2074_207488

theorem maximum_marks (passing_percentage : ℝ) (score : ℝ) (shortfall : ℝ) (total_marks : ℝ) : 
  passing_percentage = 30 → 
  score = 212 → 
  shortfall = 16 → 
  total_marks = (score + shortfall) * 100 / passing_percentage → 
  total_marks = 760 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  assumption

end maximum_marks_l2074_207488


namespace f_expression_f_odd_l2074_207479

noncomputable def f (x : ℝ) (a b : ℝ) := (2^x + b) / (2^x + a)

theorem f_expression :
  ∃ a b, f 1 a b = 1 / 3 ∧ f 0 a b = 0 ∧ (∀ x, f x a b = (2^x - 1) / (2^x + 1)) :=
by
  sorry

theorem f_odd :
  ∀ x, f x 1 (-1) = (2^x - 1) / (2^x + 1) ∧ f (-x) 1 (-1) = -f x 1 (-1) :=
by
  sorry

end f_expression_f_odd_l2074_207479


namespace students_from_other_communities_eq_90_l2074_207431

theorem students_from_other_communities_eq_90 {total_students : ℕ} 
  (muslims_percentage : ℕ)
  (hindus_percentage : ℕ)
  (sikhs_percentage : ℕ)
  (christians_percentage : ℕ)
  (buddhists_percentage : ℕ)
  : total_students = 1000 →
    muslims_percentage = 36 →
    hindus_percentage = 24 →
    sikhs_percentage = 15 →
    christians_percentage = 10 →
    buddhists_percentage = 6 →
    (total_students * (100 - (muslims_percentage + hindus_percentage + sikhs_percentage + christians_percentage + buddhists_percentage))) / 100 = 90 :=
by
  intros h_total h_muslims h_hindus h_sikhs h_christians h_buddhists
  -- Proof can be omitted as indicated
  sorry

end students_from_other_communities_eq_90_l2074_207431


namespace orange_orchard_land_l2074_207429

theorem orange_orchard_land (F H : ℕ) 
  (h1 : F + H = 120) 
  (h2 : ∃ x : ℕ, x + (2 * x + 1) = 10) 
  (h3 : ∃ x : ℕ, 2 * x + 1 = H)
  (h4 : ∃ x : ℕ, F = x) 
  (h5 : ∃ y : ℕ, H = 2 * y + 1) :
  F = 36 ∧ H = 84 :=
by
  sorry

end orange_orchard_land_l2074_207429


namespace problem_statement_l2074_207444

noncomputable def ellipse_equation (t : ℝ) (ht : t > 0) : String :=
  if h : t = 2 then "x^2/9 + y^2/2 = 1"
  else "invalid equation"

theorem problem_statement (m : ℝ) (t : ℝ) (ht : t > 0) (ha : t = 2) 
  (A E F B : ℝ × ℝ) (hA : A = (-3, 0)) (hB : B = (1, 0))
  (hl : ∀ x y, x = m * y + 1) (area : ℝ) (har : area = 16/3) :
  ((ellipse_equation t ht) = "x^2/9 + y^2/2 = 1") ∧
  (∃ M N : ℝ × ℝ, 
    (M.1 = 3 ∧ N.1 = 3) ∧
    ((M.1 - B.1) * (N.1 - B.1) + (M.2 - B.2) * (N.2 - B.2) = 0)) := 
sorry

end problem_statement_l2074_207444


namespace line_passes_through_point_l2074_207480

theorem line_passes_through_point (k : ℝ) :
  (1 + 4 * k) * 2 - (2 - 3 * k) * 2 + 2 - 14 * k = 0 :=
by
  sorry

end line_passes_through_point_l2074_207480
