import Mathlib

namespace NUMINAMATH_GPT_students_not_picked_l2280_228019

def total_students : ℕ := 58
def number_of_groups : ℕ := 8
def students_per_group : ℕ := 6

theorem students_not_picked :
  total_students - (number_of_groups * students_per_group) = 10 := by 
  sorry

end NUMINAMATH_GPT_students_not_picked_l2280_228019


namespace NUMINAMATH_GPT_gcd_20244_46656_l2280_228036

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end NUMINAMATH_GPT_gcd_20244_46656_l2280_228036


namespace NUMINAMATH_GPT_max_value_a_l2280_228032

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a = 3 :=
sorry

end NUMINAMATH_GPT_max_value_a_l2280_228032


namespace NUMINAMATH_GPT_sequence_an_form_l2280_228080

-- Definitions based on the given conditions
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ)^2 * a n
def a_1 : ℝ := 1

-- The conjecture we need to prove
theorem sequence_an_form (a : ℕ → ℝ) (h₁ : ∀ n ≥ 2, sum_first_n_terms a n = (n : ℝ)^2 * a n)
  (h₂ : a 1 = a_1) :
  ∀ n ≥ 2, a n = 2 / (n * (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_an_form_l2280_228080


namespace NUMINAMATH_GPT_solve_system_l2280_228064

theorem solve_system :
  ∀ x y : ℚ, (3 * x + 4 * y = 12) ∧ (9 * x - 12 * y = -24) →
  (x = 2 / 3) ∧ (y = 5 / 2) :=
by
  intro x y
  intro h
  sorry

end NUMINAMATH_GPT_solve_system_l2280_228064


namespace NUMINAMATH_GPT_tim_stacked_bales_today_l2280_228027

theorem tim_stacked_bales_today (initial_bales : ℕ) (current_bales : ℕ) (initial_eq : initial_bales = 54) (current_eq : current_bales = 82) : 
  current_bales - initial_bales = 28 :=
by
  -- conditions
  have h1 : initial_bales = 54 := initial_eq
  have h2 : current_bales = 82 := current_eq
  sorry

end NUMINAMATH_GPT_tim_stacked_bales_today_l2280_228027


namespace NUMINAMATH_GPT_isosceles_triangle_angle_l2280_228030

theorem isosceles_triangle_angle (x : ℕ) (h1 : 2 * x + x + x = 180) :
  x = 45 ∧ 2 * x = 90 :=
by
  have h2 : 4 * x = 180 := by linarith
  have h3 : x = 45 := by linarith
  have h4 : 2 * x = 90 := by linarith
  exact ⟨h3, h4⟩

end NUMINAMATH_GPT_isosceles_triangle_angle_l2280_228030


namespace NUMINAMATH_GPT_handshake_problem_l2280_228048

theorem handshake_problem :
  ∃ (a b : ℕ), a + b = 20 ∧ (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 ∧ a * b = 84 :=
by
  sorry

end NUMINAMATH_GPT_handshake_problem_l2280_228048


namespace NUMINAMATH_GPT_coral_remaining_pages_l2280_228014

def pages_after_week1 (total_pages : ℕ) : ℕ :=
  total_pages / 2

def pages_after_week2 (remaining_pages_week1 : ℕ) : ℕ :=
  remaining_pages_week1 - (3 * remaining_pages_week1 / 10)

def pages_after_week3 (remaining_pages_week2 : ℕ) (reading_hours : ℕ) (reading_speed : ℕ) : ℕ :=
  remaining_pages_week2 - (reading_hours * reading_speed)

theorem coral_remaining_pages (total_pages remaining_pages_week1 remaining_pages_week2 remaining_pages_week3 : ℕ) 
  (reading_hours reading_speed unread_pages : ℕ)
  (h1 : total_pages = 600)
  (h2 : remaining_pages_week1 = pages_after_week1 total_pages)
  (h3 : remaining_pages_week2 = pages_after_week2 remaining_pages_week1)
  (h4 : reading_hours = 10)
  (h5 : reading_speed = 15)
  (h6 : remaining_pages_week3 = pages_after_week3 remaining_pages_week2 reading_hours reading_speed)
  (h7 : unread_pages = remaining_pages_week3) :
  unread_pages = 60 :=
by
  sorry

end NUMINAMATH_GPT_coral_remaining_pages_l2280_228014


namespace NUMINAMATH_GPT_determine_a_if_derivative_is_even_l2280_228070

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem determine_a_if_derivative_is_even (a : ℝ) :
  (∀ x : ℝ, f' x a = f' (-x) a) → a = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_determine_a_if_derivative_is_even_l2280_228070


namespace NUMINAMATH_GPT_milk_concentration_l2280_228097

variable {V_initial V_removed V_total : ℝ}

theorem milk_concentration (h1 : V_initial = 20) (h2 : V_removed = 2) (h3 : V_total = 20) :
    (V_initial - V_removed) / V_total * 100 = 90 := 
by 
  sorry

end NUMINAMATH_GPT_milk_concentration_l2280_228097


namespace NUMINAMATH_GPT_bug_crawl_distance_l2280_228004

-- Define the positions visited by the bug
def start_position := -3
def first_stop := 0
def second_stop := -8
def final_stop := 10

-- Define the function to calculate the total distance crawled by the bug
def total_distance : ℤ :=
  abs (first_stop - start_position) + abs (second_stop - first_stop) + abs (final_stop - second_stop)

-- Prove that the total distance is 29 units
theorem bug_crawl_distance : total_distance = 29 :=
by
  -- Definitions are used here to validate the statement
  sorry

end NUMINAMATH_GPT_bug_crawl_distance_l2280_228004


namespace NUMINAMATH_GPT_soda_original_price_l2280_228055

theorem soda_original_price (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  sorry

end NUMINAMATH_GPT_soda_original_price_l2280_228055


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l2280_228060

noncomputable def H : ℕ := 322 / 14

theorem hcf_of_two_numbers (H k : ℕ) (lcm_val : ℕ) :
  lcm_val = H * 13 * 14 ∧ 322 = H * k ∧ 322 / 14 = H → H = 23 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l2280_228060


namespace NUMINAMATH_GPT_probability_black_given_not_white_l2280_228043

theorem probability_black_given_not_white
  (total_balls : ℕ)
  (white_balls : ℕ)
  (yellow_balls : ℕ)
  (black_balls : ℕ)
  (H1 : total_balls = 25)
  (H2 : white_balls = 10)
  (H3 : yellow_balls = 5)
  (H4 : black_balls = 10)
  (H5 : total_balls = white_balls + yellow_balls + black_balls)
  (H6 : ¬white_balls = total_balls) :
  (10 / (25 - 10) : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_black_given_not_white_l2280_228043


namespace NUMINAMATH_GPT_maddie_total_cost_l2280_228020

theorem maddie_total_cost :
  let price_palette := 15
  let price_lipstick := 2.5
  let price_hair_color := 4
  let num_palettes := 3
  let num_lipsticks := 4
  let num_hair_colors := 3
  let total_cost := (num_palettes * price_palette) + (num_lipsticks * price_lipstick) + (num_hair_colors * price_hair_color)
  total_cost = 67 := by
  sorry

end NUMINAMATH_GPT_maddie_total_cost_l2280_228020


namespace NUMINAMATH_GPT_cone_slant_height_correct_l2280_228087

noncomputable def cone_slant_height (r : ℝ) : ℝ := 4 * r

theorem cone_slant_height_correct (r : ℝ) (h₁ : π * r^2 + π * r * cone_slant_height r = 5 * π)
  (h₂ : 2 * π * r = (1/4) * 2 * π * cone_slant_height r) : cone_slant_height r = 4 :=
by
  sorry

end NUMINAMATH_GPT_cone_slant_height_correct_l2280_228087


namespace NUMINAMATH_GPT_appropriate_speech_length_l2280_228093

def speech_length_min := 20
def speech_length_max := 40
def speech_rate := 120

theorem appropriate_speech_length 
  (min_words := speech_length_min * speech_rate) 
  (max_words := speech_length_max * speech_rate) : 
  ∀ n : ℕ, n >= min_words ∧ n <= max_words ↔ (n = 2500 ∨ n = 3800 ∨ n = 4600) := 
by 
  sorry

end NUMINAMATH_GPT_appropriate_speech_length_l2280_228093


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l2280_228098

theorem right_triangle_hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12)
  (h₃ : c^2 = a^2 + b^2) : c = 13 :=
by
  -- We should provide the actual proof here, but we'll use sorry for now.
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l2280_228098


namespace NUMINAMATH_GPT_total_bugs_eaten_l2280_228053

theorem total_bugs_eaten :
  let gecko_bugs := 12
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + (frog_bugs / 2)
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 :=
by
  sorry

end NUMINAMATH_GPT_total_bugs_eaten_l2280_228053


namespace NUMINAMATH_GPT_customer_difference_l2280_228069

theorem customer_difference (X Y Z : ℕ) (h1 : X - Y = 10) (h2 : 10 - Z = 4) : X - 4 = 10 :=
by sorry

end NUMINAMATH_GPT_customer_difference_l2280_228069


namespace NUMINAMATH_GPT_mod_computation_l2280_228079

theorem mod_computation (a b n : ℕ) (h_modulus : n = 7) (h_a : a = 47) (h_b : b = 28) :
  (a^2023 - b^2023) % n = 5 :=
by
  sorry

end NUMINAMATH_GPT_mod_computation_l2280_228079


namespace NUMINAMATH_GPT_fraction_of_rectangle_shaded_l2280_228025

theorem fraction_of_rectangle_shaded
  (length : ℕ) (width : ℕ)
  (one_third_part : ℕ) (half_of_third : ℕ)
  (H1 : length = 10) (H2 : width = 15)
  (H3 : one_third_part = (1/3 : ℝ) * (length * width)) 
  (H4 : half_of_third = (1/2 : ℝ) * one_third_part) :
  (half_of_third / (length * width) = 1/6) :=
sorry

end NUMINAMATH_GPT_fraction_of_rectangle_shaded_l2280_228025


namespace NUMINAMATH_GPT_range_of_b_l2280_228013

noncomputable def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}
noncomputable def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem range_of_b (b : ℝ) : 
  (set_A ∩ set_B b = ∅) ↔ (b = 0 ∨ b ≥ 1/3 ∨ b ≤ -2) :=
sorry

end NUMINAMATH_GPT_range_of_b_l2280_228013


namespace NUMINAMATH_GPT_enrique_commission_l2280_228047

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end NUMINAMATH_GPT_enrique_commission_l2280_228047


namespace NUMINAMATH_GPT_loom_weaving_rate_l2280_228017

theorem loom_weaving_rate :
  (119.04761904761905 : ℝ) > 0 ∧ (15 : ℝ) > 0 ∧ ∃ rate : ℝ, rate = 15 / 119.04761904761905 → rate = 0.126 :=
by sorry

end NUMINAMATH_GPT_loom_weaving_rate_l2280_228017


namespace NUMINAMATH_GPT_determine_k_and_a_n_and_T_n_l2280_228083

noncomputable def S_n (n : ℕ) (k : ℝ) : ℝ := -0.5 * n^2 + k * n

/-- Given the sequence S_n with sum of the first n terms S_n := -1/2 n^2 + k*n,
where k is a positive natural number. The maximum value of S_n is 8. -/
theorem determine_k_and_a_n_and_T_n (k : ℝ) (h : k = 4) :
  (∀ n : ℕ, S_n n k ≤ 8) ∧ 
  (∀ n : ℕ, ∃ a : ℝ, a = 9/2 - n) ∧
  (∀ n : ℕ, ∃ T : ℝ, T = 4 - (n + 2)/2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_determine_k_and_a_n_and_T_n_l2280_228083


namespace NUMINAMATH_GPT_inverse_proportion_function_has_m_value_l2280_228090

theorem inverse_proportion_function_has_m_value
  (k : ℝ)
  (h1 : 2 * -3 = k)
  {m : ℝ}
  (h2 : 6 = k / m) :
  m = -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_function_has_m_value_l2280_228090


namespace NUMINAMATH_GPT_simplify_expr_l2280_228038

theorem simplify_expr (x y : ℝ) (P Q : ℝ) (hP : P = x^2 + y^2) (hQ : Q = x^2 - y^2) : 
  (P * Q / (P + Q)) + ((P + Q) / (P * Q)) = ((x^4 + y^4) ^ 2) / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end NUMINAMATH_GPT_simplify_expr_l2280_228038


namespace NUMINAMATH_GPT_num_valid_m_values_for_distributing_marbles_l2280_228058

theorem num_valid_m_values_for_distributing_marbles : 
  ∃ (m_values : Finset ℕ), m_values.card = 22 ∧ 
  ∀ m ∈ m_values, ∃ n : ℕ, m * n = 360 ∧ n > 1 ∧ m > 1 :=
by
  sorry

end NUMINAMATH_GPT_num_valid_m_values_for_distributing_marbles_l2280_228058


namespace NUMINAMATH_GPT_digits_satisfy_sqrt_l2280_228067

theorem digits_satisfy_sqrt (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (b = 0 ∧ a = 0) ∨ (b = 3 ∧ a = 1) ∨ (b = 6 ∧ a = 4) ∨ (b = 9 ∧ a = 9) ↔ b^2 = 9 * a :=
by
  sorry

end NUMINAMATH_GPT_digits_satisfy_sqrt_l2280_228067


namespace NUMINAMATH_GPT_sum_of_terms_l2280_228024

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_terms_l2280_228024


namespace NUMINAMATH_GPT_min_distance_to_water_all_trees_l2280_228009

/-- Proof that the minimum distance Xiao Zhang must walk to water all 10 trees is 410 meters -/
def minimum_distance_to_water_trees (num_trees : ℕ) (distance_between_trees : ℕ) : ℕ := 
  (sorry) -- implementation to calculate the minimum distance

theorem min_distance_to_water_all_trees (num_trees distance_between_trees : ℕ) :
  num_trees = 10 → 
  distance_between_trees = 10 →
  minimum_distance_to_water_trees num_trees distance_between_trees = 410 :=
by
  intros h_num_trees h_distance_between_trees
  rw [h_num_trees, h_distance_between_trees]
  -- Add proof here that the distance is 410
  sorry

end NUMINAMATH_GPT_min_distance_to_water_all_trees_l2280_228009


namespace NUMINAMATH_GPT_multiplier_of_first_integer_l2280_228077

theorem multiplier_of_first_integer :
  ∃ m x : ℤ, x + 4 = 15 ∧ x * m = 3 + 2 * 15 ∧ m = 3 := by
  sorry

end NUMINAMATH_GPT_multiplier_of_first_integer_l2280_228077


namespace NUMINAMATH_GPT_exists_odd_integers_l2280_228012

theorem exists_odd_integers (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, x % 2 = 1 ∧ y % 2 = 1 ∧ x^2 + 7 * y^2 = 2^n :=
sorry

end NUMINAMATH_GPT_exists_odd_integers_l2280_228012


namespace NUMINAMATH_GPT_min_value_correct_l2280_228008

noncomputable def min_value (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) : ℝ :=
(1 / m) + (2 / n)

theorem min_value_correct :
  ∃ m n : ℝ, ∃ h₁ : m > 0, ∃ h₂ : n > 0, ∃ h₃ : m + n = 1,
  min_value m n h₁ h₂ h₃ = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_correct_l2280_228008


namespace NUMINAMATH_GPT_pq_ratio_at_0_l2280_228063

noncomputable def p (x : ℝ) : ℝ := -3 * (x + 4) * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 3)

theorem pq_ratio_at_0 : (p 0) / (q 0) = 0 := by
  sorry

end NUMINAMATH_GPT_pq_ratio_at_0_l2280_228063


namespace NUMINAMATH_GPT_range_of_a_l2280_228042

noncomputable def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
noncomputable def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

theorem range_of_a (a : ℝ) : 
  (S ∪ T a) = Set.univ ↔ -2 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l2280_228042


namespace NUMINAMATH_GPT_find_g_26_l2280_228062

variable {g : ℕ → ℕ}

theorem find_g_26 (hg : ∀ x, g (x + g x) = 5 * g x) (h1 : g 1 = 5) : g 26 = 120 :=
  sorry

end NUMINAMATH_GPT_find_g_26_l2280_228062


namespace NUMINAMATH_GPT_find_N_l2280_228026

theorem find_N : (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / (N : ℚ) → N = 1991 := by
sorry

end NUMINAMATH_GPT_find_N_l2280_228026


namespace NUMINAMATH_GPT_temperature_difference_in_fahrenheit_l2280_228006

-- Define the conversion formula from Celsius to Fahrenheit as a function
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperatures in Boston and New York
variables (C_B C_N : ℝ)

-- Condition: New York is 10 degrees Celsius warmer than Boston
axiom temp_difference : C_N = C_B + 10

-- Goal: The temperature difference in Fahrenheit
theorem temperature_difference_in_fahrenheit : celsius_to_fahrenheit C_N - celsius_to_fahrenheit C_B = 18 :=
by sorry

end NUMINAMATH_GPT_temperature_difference_in_fahrenheit_l2280_228006


namespace NUMINAMATH_GPT_train_speed_km_hr_calc_l2280_228034

theorem train_speed_km_hr_calc :
  let length := 175 -- length of the train in meters
  let time := 3.499720022398208 -- time to cross the pole in seconds
  let speed_mps := length / time -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed from m/s to km/hr
  speed_kmph = 180.025923226 := 
sorry

end NUMINAMATH_GPT_train_speed_km_hr_calc_l2280_228034


namespace NUMINAMATH_GPT_emma_possible_lists_l2280_228016

-- Define the number of balls
def number_of_balls : ℕ := 24

-- Define the number of draws Emma repeats independently
def number_of_draws : ℕ := 4

-- Define the calculation for the total number of different lists
def total_number_of_lists : ℕ := number_of_balls ^ number_of_draws

theorem emma_possible_lists : total_number_of_lists = 331776 := by
  sorry

end NUMINAMATH_GPT_emma_possible_lists_l2280_228016


namespace NUMINAMATH_GPT_trigonometric_identity_l2280_228086

open Real

theorem trigonometric_identity (θ : ℝ) (h : tan θ = 2) :
  (sin θ * (1 + sin (2 * θ))) / (sqrt 2 * cos (θ - π / 4)) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2280_228086


namespace NUMINAMATH_GPT_area_of_hall_l2280_228005

-- Define the conditions
def length := 25
def breadth := length - 5

-- Define the area calculation
def area := length * breadth

-- The statement to prove
theorem area_of_hall : area = 500 :=
by
  sorry

end NUMINAMATH_GPT_area_of_hall_l2280_228005


namespace NUMINAMATH_GPT_find_n_l2280_228095

theorem find_n 
  (N : ℕ) 
  (hn : ¬ (N = 0))
  (parts_inv_prop : ∀ k, 1 ≤ k → k ≤ n → N / (k * (k + 1)) = x / (n * (n + 1))) 
  (smallest_part : (N : ℝ) / 400 = N / (n * (n + 1))) : 
  n = 20 :=
sorry

end NUMINAMATH_GPT_find_n_l2280_228095


namespace NUMINAMATH_GPT_steven_weight_l2280_228078

theorem steven_weight (danny_weight : ℝ) (steven_more : ℝ) (steven_weight : ℝ) 
  (h₁ : danny_weight = 40) 
  (h₂ : steven_more = 0.2 * danny_weight) 
  (h₃ : steven_weight = danny_weight + steven_more) : 
  steven_weight = 48 := 
  by 
  sorry

end NUMINAMATH_GPT_steven_weight_l2280_228078


namespace NUMINAMATH_GPT_simplify_log_expression_l2280_228075

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem simplify_log_expression :
  let term1 := 1 / (log_base 20 3 + 1)
  let term2 := 1 / (log_base 12 5 + 1)
  let term3 := 1 / (log_base 8 7 + 1)
  term1 + term2 + term3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_log_expression_l2280_228075


namespace NUMINAMATH_GPT_smallest_n_divisible_by_one_billion_l2280_228054

-- Define the sequence parameters and the common ratio
def first_term : ℚ := 5 / 8
def second_term : ℚ := 50
def common_ratio : ℚ := second_term / first_term -- this is 80

-- Define the n-th term of the geometric sequence
noncomputable def nth_term (n : ℕ) : ℚ :=
  first_term * (common_ratio ^ (n - 1))

-- Define the target divisor (one billion)
def target_divisor : ℤ := 10 ^ 9

-- Prove that the smallest n such that nth_term n is divisible by 10^9 is 9
theorem smallest_n_divisible_by_one_billion :
  ∃ n : ℕ, nth_term n = (first_term * (common_ratio ^ (n - 1))) ∧ 
           (target_divisor : ℚ) ∣ nth_term n ∧
           n = 9 :=
by sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_one_billion_l2280_228054


namespace NUMINAMATH_GPT_count_heads_at_night_l2280_228068

variables (J T D : ℕ)

theorem count_heads_at_night (h1 : 2 * J + 4 * T + 2 * D = 56) : J + T + D = 14 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_count_heads_at_night_l2280_228068


namespace NUMINAMATH_GPT_r_sq_plus_s_sq_l2280_228028

variable {r s : ℝ}

theorem r_sq_plus_s_sq (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := 
by
  sorry

end NUMINAMATH_GPT_r_sq_plus_s_sq_l2280_228028


namespace NUMINAMATH_GPT_weight_of_bag_l2280_228056

-- Definitions
def chicken_price : ℝ := 1.50
def bag_cost : ℝ := 2
def feed_per_chicken : ℝ := 2
def profit_from_50_chickens : ℝ := 65
def total_chickens : ℕ := 50

-- Theorem
theorem weight_of_bag : 
  (bag_cost / (profit_from_50_chickens - 
               (total_chickens * chicken_price)) / 
               (feed_per_chicken * total_chickens)) = 20 := 
sorry

end NUMINAMATH_GPT_weight_of_bag_l2280_228056


namespace NUMINAMATH_GPT_distance_traveled_by_light_in_10_seconds_l2280_228049

theorem distance_traveled_by_light_in_10_seconds :
  ∃ (a : ℝ) (n : ℕ), (300000 * 10 : ℝ) = a * 10 ^ n ∧ n = 6 :=
sorry

end NUMINAMATH_GPT_distance_traveled_by_light_in_10_seconds_l2280_228049


namespace NUMINAMATH_GPT_part_a_part_b_l2280_228096

theorem part_a (k : ℕ) : ∃ (a : ℕ → ℕ), (∀ i, i ≤ k → a i > 0) ∧ (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → a i < a j) ∧ (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) :=
sorry

theorem part_b : ∃ C > 0, ∀ a : ℕ → ℕ, (∀ k : ℕ, (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) → a 1 > (k : ℕ) ^ (C * k : ℕ)) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l2280_228096


namespace NUMINAMATH_GPT_range_of_function_l2280_228091

theorem range_of_function :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 2 ≤ x^2 - 2 * x + 3 ∧ x^2 - 2 * x + 3 ≤ 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_function_l2280_228091


namespace NUMINAMATH_GPT_problem_a_lt_zero_b_lt_neg_one_l2280_228039

theorem problem_a_lt_zero_b_lt_neg_one (a b : ℝ) (ha : a < 0) (hb : b < -1) : 
  ab > a ∧ a > ab^2 := 
by
  sorry

end NUMINAMATH_GPT_problem_a_lt_zero_b_lt_neg_one_l2280_228039


namespace NUMINAMATH_GPT_speed_of_first_bus_l2280_228074

theorem speed_of_first_bus (v : ℕ) (h : (v + 60) * 4 = 460) : v = 55 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_first_bus_l2280_228074


namespace NUMINAMATH_GPT_compound_interest_is_correct_l2280_228052

noncomputable def compoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R)^T - P

theorem compound_interest_is_correct
  (P : ℝ)
  (R : ℝ)
  (T : ℝ)
  (SI : ℝ) : SI = P * R * T / 100 ∧ R = 0.10 ∧ T = 2 ∧ SI = 600 → compoundInterest P R T = 630 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_is_correct_l2280_228052


namespace NUMINAMATH_GPT_non_monotonic_piecewise_l2280_228010

theorem non_monotonic_piecewise (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ (x t : ℝ),
    (f x = if x ≤ t then (4 * a - 3) * x + (2 * a - 4) else (2 * x^3 - 6 * x)))
  : a ≤ 3 / 4 := 
sorry

end NUMINAMATH_GPT_non_monotonic_piecewise_l2280_228010


namespace NUMINAMATH_GPT_find_original_price_l2280_228022

-- Given conditions:
-- 1. 10% cashback
-- 2. $25 mail-in rebate
-- 3. Final cost is $110

def original_price (P : ℝ) (cashback : ℝ) (rebate : ℝ) (final_cost : ℝ) :=
  final_cost = P - (cashback * P + rebate)

theorem find_original_price :
  ∀ (P : ℝ), original_price P 0.10 25 110 → P = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_original_price_l2280_228022


namespace NUMINAMATH_GPT_parallelepiped_intersection_l2280_228001

/-- Given a parallelepiped A B C D A₁ B₁ C₁ D₁.
    Point X is chosen on edge A₁ D₁, and point Y is chosen on edge B C.
    It is known that A₁ X = 5, B Y = 3, and B₁ C₁ = 14.
    The plane C₁ X Y intersects ray D A at point Z.
    Prove that D Z = 20. -/
theorem parallelepiped_intersection
  (A B C D A₁ B₁ C₁ D₁ X Y Z : ℝ)
  (h₁: A₁ - X = 5)
  (h₂: B - Y = 3)
  (h₃: B₁ - C₁ = 14) :
  D - Z = 20 :=
sorry

end NUMINAMATH_GPT_parallelepiped_intersection_l2280_228001


namespace NUMINAMATH_GPT_probability_of_winning_l2280_228041

def total_products_in_box : ℕ := 6
def winning_products_in_box : ℕ := 2

theorem probability_of_winning : (winning_products_in_box : ℚ) / (total_products_in_box : ℚ) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_probability_of_winning_l2280_228041


namespace NUMINAMATH_GPT_max_min_x_plus_y_l2280_228037

theorem max_min_x_plus_y (x y : ℝ) (h : |x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) :
  -3 ≤ x + y ∧ x + y ≤ 6 := 
sorry

end NUMINAMATH_GPT_max_min_x_plus_y_l2280_228037


namespace NUMINAMATH_GPT_max_value_of_expression_l2280_228023

noncomputable def f (x y : ℝ) := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  ∃ m, m = 951625 / 256 ∧ ∀ a b : ℝ, a + b = 5 → f a b ≤ m :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2280_228023


namespace NUMINAMATH_GPT_age_difference_l2280_228018

theorem age_difference (b_age : ℕ) (bro_age : ℕ) (h1 : b_age = 5) (h2 : b_age + bro_age = 19) : 
  bro_age - b_age = 9 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l2280_228018


namespace NUMINAMATH_GPT_sqrt_sum_l2280_228044

theorem sqrt_sum (m n : ℝ) (h1 : m + n = 0) (h2 : m * n = -2023) : m + 2 * m * n + n = -4046 :=
by sorry

end NUMINAMATH_GPT_sqrt_sum_l2280_228044


namespace NUMINAMATH_GPT_cyclist_total_heartbeats_l2280_228057

theorem cyclist_total_heartbeats
  (heart_rate : ℕ := 120) -- beats per minute
  (race_distance : ℕ := 50) -- miles
  (pace : ℕ := 4) -- minutes per mile
  : (race_distance * pace) * heart_rate = 24000 := by
  sorry

end NUMINAMATH_GPT_cyclist_total_heartbeats_l2280_228057


namespace NUMINAMATH_GPT_machinery_spent_correct_l2280_228099

def raw_materials : ℝ := 3000
def total_amount : ℝ := 5714.29
def cash (total : ℝ) : ℝ := 0.30 * total
def machinery_spent (total : ℝ) (raw : ℝ) : ℝ := total - raw - cash total

theorem machinery_spent_correct :
  machinery_spent total_amount raw_materials = 1000 := 
  by
    sorry

end NUMINAMATH_GPT_machinery_spent_correct_l2280_228099


namespace NUMINAMATH_GPT_symmetric_points_origin_a_plus_b_l2280_228092

theorem symmetric_points_origin_a_plus_b (a b : ℤ) 
  (h1 : a + 3 * b = 5)
  (h2 : a + 2 * b = -3) :
  a + b = -11 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_a_plus_b_l2280_228092


namespace NUMINAMATH_GPT_faye_homework_problems_left_l2280_228015

-- Defining the problem conditions
def M : ℕ := 46
def S : ℕ := 9
def A : ℕ := 40

-- The statement to prove
theorem faye_homework_problems_left : M + S - A = 15 := by
  sorry

end NUMINAMATH_GPT_faye_homework_problems_left_l2280_228015


namespace NUMINAMATH_GPT_a_lt_one_l2280_228029

-- Define the function f(x) = |x-3| + |x+7|
def f (x : ℝ) : ℝ := |x-3| + |x+7|

-- The statement of the problem
theorem a_lt_one (a : ℝ) :
  (∀ x : ℝ, a < Real.log (f x)) → a < 1 :=
by
  intro h
  have H : f (-7) = 10 := by sorry -- piecewise definition
  have H1 : Real.log (f (-7)) = 1 := by sorry -- minimum value of log
  specialize h (-7)
  rw [H1] at h
  exact h

end NUMINAMATH_GPT_a_lt_one_l2280_228029


namespace NUMINAMATH_GPT_opposite_z_is_E_l2280_228007

noncomputable def cube_faces := ["A", "B", "C", "D", "E", "z"]

def opposite_face (net : List String) (face : String) : String :=
  if face = "z" then "E" else sorry  -- generalize this function as needed

theorem opposite_z_is_E :
  opposite_face cube_faces "z" = "E" :=
by
  sorry

end NUMINAMATH_GPT_opposite_z_is_E_l2280_228007


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_coords_l2280_228072

theorem point_in_fourth_quadrant_coords 
  (P : ℝ × ℝ)
  (h1 : P.2 < 0)
  (h2 : abs P.2 = 2)
  (h3 : P.1 > 0)
  (h4 : abs P.1 = 5) :
  P = (5, -2) :=
sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_coords_l2280_228072


namespace NUMINAMATH_GPT_min_value_problem_l2280_228089

theorem min_value_problem (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) : 
    a^2 + b^2 + c^2 + d^2 >= 24 / 5 := 
by
  sorry

end NUMINAMATH_GPT_min_value_problem_l2280_228089


namespace NUMINAMATH_GPT_find_tuesday_temperature_l2280_228000

variable (T W Th F : ℝ)

def average_temperature_1 : Prop := (T + W + Th) / 3 = 52
def average_temperature_2 : Prop := (W + Th + F) / 3 = 54
def friday_temperature : Prop := F = 53

theorem find_tuesday_temperature (h1 : average_temperature_1 T W Th) (h2 : average_temperature_2 W Th F) (h3 : friday_temperature F) :
  T = 47 :=
by
  sorry

end NUMINAMATH_GPT_find_tuesday_temperature_l2280_228000


namespace NUMINAMATH_GPT_polynomial_real_root_inequality_l2280_228071

theorem polynomial_real_root_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a * x^3 + 2 * x^2 - b * x + 1 = 0) → (a^2 + b^2 ≥ 8) :=
sorry

end NUMINAMATH_GPT_polynomial_real_root_inequality_l2280_228071


namespace NUMINAMATH_GPT_avg_growth_rate_first_brand_eq_l2280_228003

noncomputable def avg_growth_rate_first_brand : ℝ :=
  let t := 5.647
  let first_brand_households_2001 := 4.9
  let second_brand_households_2001 := 2.5
  let second_brand_growth_rate := 0.7
  let equalization_time := t
  (second_brand_households_2001 + second_brand_growth_rate * equalization_time - first_brand_households_2001) / equalization_time

theorem avg_growth_rate_first_brand_eq :
  avg_growth_rate_first_brand = 0.275 := by
  sorry

end NUMINAMATH_GPT_avg_growth_rate_first_brand_eq_l2280_228003


namespace NUMINAMATH_GPT_solve_inequality_l2280_228011

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2)
  (h3 : (x^2 + 3*x - 1) / (4 - x^2) < 1)
  (h4 : (x^2 + 3*x - 1) / (4 - x^2) ≥ -1) :
  x < -5 / 2 ∨ (-1 ≤ x ∧ x < 1) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l2280_228011


namespace NUMINAMATH_GPT_real_solutions_l2280_228073

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l2280_228073


namespace NUMINAMATH_GPT_dinesh_loop_l2280_228059

noncomputable def number_of_pentagons (n : ℕ) : ℕ :=
  if (20 * n) % 11 = 0 then 10 else 0

theorem dinesh_loop (n : ℕ) : number_of_pentagons n = 10 :=
by sorry

end NUMINAMATH_GPT_dinesh_loop_l2280_228059


namespace NUMINAMATH_GPT_weigh_80_grams_is_false_l2280_228051

def XiaoGang_weight_grams : Nat := 80000  -- 80 kilograms in grams
def weight_claim : Nat := 80  -- 80 grams claim

theorem weigh_80_grams_is_false : weight_claim ≠ XiaoGang_weight_grams :=
by
  sorry

end NUMINAMATH_GPT_weigh_80_grams_is_false_l2280_228051


namespace NUMINAMATH_GPT_well_diameter_l2280_228084

noncomputable def calculateDiameter (volume depth : ℝ) : ℝ :=
  2 * Real.sqrt (volume / (Real.pi * depth))

theorem well_diameter :
  calculateDiameter 678.5840131753953 24 = 6 :=
by
  sorry

end NUMINAMATH_GPT_well_diameter_l2280_228084


namespace NUMINAMATH_GPT_Vasya_can_win_l2280_228076

theorem Vasya_can_win 
  (a : ℕ → ℕ) -- initial sequence of natural numbers
  (x : ℕ) -- number chosen by Vasya
: ∃ (i : ℕ), ∀ (k : ℕ), ∃ (j : ℕ), (a j + k * x = 1) :=
by
  sorry

end NUMINAMATH_GPT_Vasya_can_win_l2280_228076


namespace NUMINAMATH_GPT_calculate_milk_and_oil_l2280_228094

theorem calculate_milk_and_oil (q_f div_f milk_p oil_p : ℕ) (portions q_m q_o : ℕ) :
  q_f = 1050 ∧ div_f = 350 ∧ milk_p = 70 ∧ oil_p = 30 ∧
  portions = q_f / div_f ∧
  q_m = portions * milk_p ∧
  q_o = portions * oil_p →
  q_m = 210 ∧ q_o = 90 := by
  sorry

end NUMINAMATH_GPT_calculate_milk_and_oil_l2280_228094


namespace NUMINAMATH_GPT_circles_symmetric_sin_cos_l2280_228033

noncomputable def sin_cos_product (θ : Real) : Real := Real.sin θ * Real.cos θ

theorem circles_symmetric_sin_cos (a θ : Real) 
(h1 : ∃ x1 y1, x1 = -a / 2 ∧ y1 = 0 ∧ 2*x1 - y1 - 1 = 0) 
(h2 : ∃ x2 y2, x2 = -a ∧ y2 = -Real.tan θ / 2 ∧ 2*x2 - y2 - 1 = 0) :
sin_cos_product θ = -2 / 5 := 
sorry

end NUMINAMATH_GPT_circles_symmetric_sin_cos_l2280_228033


namespace NUMINAMATH_GPT_trigonometric_eq_solution_count_l2280_228046

theorem trigonometric_eq_solution_count :
  ∃ B : Finset ℤ, B.card = 250 ∧ ∀ x ∈ B, 2000 ≤ x ∧ x ≤ 3000 ∧ 
  2 * Real.sqrt 2 * Real.sin (Real.pi * x / 4)^3 = Real.sin (Real.pi / 4 * (1 + x)) :=
sorry

end NUMINAMATH_GPT_trigonometric_eq_solution_count_l2280_228046


namespace NUMINAMATH_GPT_install_time_for_windows_l2280_228061

theorem install_time_for_windows
  (total_windows installed_windows hours_per_window : ℕ)
  (h1 : total_windows = 200)
  (h2 : installed_windows = 65)
  (h3 : hours_per_window = 12) :
  (total_windows - installed_windows) * hours_per_window = 1620 :=
by
  sorry

end NUMINAMATH_GPT_install_time_for_windows_l2280_228061


namespace NUMINAMATH_GPT_min_S_value_l2280_228081

theorem min_S_value (n : ℕ) (h₁ : n ≥ 375) :
    let R := 3000
    let S := 9 * n - R
    let dice_sum (s : ℕ) := ∃ L : List ℕ, (∀ x ∈ L, 1 ≤ x ∧ x ≤ 8) ∧ L.sum = s
    dice_sum R ∧ S = 375 := 
by
  sorry

end NUMINAMATH_GPT_min_S_value_l2280_228081


namespace NUMINAMATH_GPT_triangle_perimeter_l2280_228088

def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

def foci_distance (c : ℝ) := c = Real.sqrt 2

theorem triangle_perimeter {x y : ℝ} (A : ellipse x y) (F1 F2 : ℝ)
  (hF1 : F1 = -Real.sqrt 2) (hF2 : F2 = Real.sqrt 2) :
  |(x - F1)| + |(x - F2)| = 4 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l2280_228088


namespace NUMINAMATH_GPT_value_of_a_plus_b_l2280_228035

theorem value_of_a_plus_b 
  (a b : ℝ) 
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = a * x + b)
  (h₂ : ∀ x, g x = 3 * x - 6)
  (h₃ : ∀ x, g (f x) = 4 * x + 5) : 
  a + b = 5 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l2280_228035


namespace NUMINAMATH_GPT_pencils_on_desk_l2280_228021

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end NUMINAMATH_GPT_pencils_on_desk_l2280_228021


namespace NUMINAMATH_GPT_sin_theta_tan_theta_iff_first_third_quadrant_l2280_228031

open Real

-- Definitions from conditions
def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ ∧ θ < π / 2) ∨ (π < θ ∧ θ < 3 * π / 2)

def sin_theta_plus_tan_theta_positive (θ : ℝ) : Prop :=
  sin θ + tan θ > 0

-- Proof statement
theorem sin_theta_tan_theta_iff_first_third_quadrant (θ : ℝ) :
  sin_theta_plus_tan_theta_positive θ ↔ in_first_or_third_quadrant θ :=
sorry

end NUMINAMATH_GPT_sin_theta_tan_theta_iff_first_third_quadrant_l2280_228031


namespace NUMINAMATH_GPT_certain_number_eq_14_l2280_228065

theorem certain_number_eq_14 (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : y^2 = 4) : 2 * x - y = 14 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_eq_14_l2280_228065


namespace NUMINAMATH_GPT_train_speed_l2280_228040

/--
Given:
- The speed of the first person \(V_p\) is 4 km/h.
- The train takes 9 seconds to pass the first person completely.
- The length of the train is approximately 50 meters (49.999999999999986 meters).

Prove:
- The speed of the train \(V_t\) is 24 km/h.
-/
theorem train_speed (V_p : ℝ) (t : ℝ) (L : ℝ) (V_t : ℝ) 
  (hV_p : V_p = 4) 
  (ht : t = 9)
  (hL : L = 49.999999999999986)
  (hrel_speed : (L / t) * 3.6 = V_t - V_p) :
  V_t = 24 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l2280_228040


namespace NUMINAMATH_GPT_volume_formula_correct_l2280_228045

def volume_of_box (x : ℝ) : ℝ :=
  x * (16 - 2 * x) * (12 - 2 * x)

theorem volume_formula_correct (x : ℝ) (h : x ≤ 12 / 5) :
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x :=
by sorry

end NUMINAMATH_GPT_volume_formula_correct_l2280_228045


namespace NUMINAMATH_GPT_side_length_of_square_l2280_228066

theorem side_length_of_square (total_length : ℝ) (sides : ℕ) (h1 : total_length = 100) (h2 : sides = 4) :
  (total_length / (sides : ℝ) = 25) :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l2280_228066


namespace NUMINAMATH_GPT_find_number_l2280_228050

theorem find_number (x : ℝ) (h : (1/4) * x = (1/5) * (x + 1) + 1) : x = 24 := 
sorry

end NUMINAMATH_GPT_find_number_l2280_228050


namespace NUMINAMATH_GPT_season_duration_l2280_228082

-- Define the given conditions.
def games_per_month : ℕ := 7
def games_per_season : ℕ := 14

-- Define the property we want to prove.
theorem season_duration : games_per_season / games_per_month = 2 :=
by
  sorry

end NUMINAMATH_GPT_season_duration_l2280_228082


namespace NUMINAMATH_GPT_batsman_average_increase_l2280_228085

theorem batsman_average_increase :
  ∀ (A : ℝ), (10 * A + 110 = 11 * 60) → (60 - A = 5) :=
by
  intros A h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l2280_228085


namespace NUMINAMATH_GPT_expected_rolls_in_non_leap_year_l2280_228002

-- Define the conditions and the expected value
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def stops_rolling (n : ℕ) : Prop := is_prime n ∨ is_multiple_of_4 n

def expected_rolls_one_day : ℚ := 6 / 7

def non_leap_year_days : ℕ := 365

def expected_rolls_one_year := expected_rolls_one_day * non_leap_year_days

theorem expected_rolls_in_non_leap_year : expected_rolls_one_year = 314 :=
by
  -- Verification of the mathematical model
  sorry

end NUMINAMATH_GPT_expected_rolls_in_non_leap_year_l2280_228002
