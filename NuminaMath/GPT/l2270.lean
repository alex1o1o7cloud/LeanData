import Mathlib

namespace NUMINAMATH_GPT_math_problem_l2270_227006

theorem math_problem
  (p q r s : ℕ)
  (hpq : p^3 = q^2)
  (hrs : r^4 = s^3)
  (hrp : r - p = 25) :
  s - q = 73 := by
  sorry

end NUMINAMATH_GPT_math_problem_l2270_227006


namespace NUMINAMATH_GPT_beth_overall_score_l2270_227054

-- Definitions for conditions
def percent_score (score_pct : ℕ) (total_problems : ℕ) : ℕ :=
  (score_pct * total_problems) / 100

def total_correct_answers : ℕ :=
  percent_score 60 15 + percent_score 85 20 + percent_score 75 25

def total_problems : ℕ := 15 + 20 + 25

def combined_percentage : ℕ :=
  (total_correct_answers * 100) / total_problems

-- The statement to be proved
theorem beth_overall_score : combined_percentage = 75 := by
  sorry

end NUMINAMATH_GPT_beth_overall_score_l2270_227054


namespace NUMINAMATH_GPT_andreas_living_room_floor_area_l2270_227024

-- Definitions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_coverage_percentage : ℝ := 0.30
def carpet_area : ℝ := carpet_length * carpet_width

-- Theorem statement
theorem andreas_living_room_floor_area (A : ℝ) 
  (h1 : carpet_coverage_percentage * A = carpet_area) :
  A = 120 :=
by
  sorry

end NUMINAMATH_GPT_andreas_living_room_floor_area_l2270_227024


namespace NUMINAMATH_GPT_geometric_series_sum_150_terms_l2270_227062

theorem geometric_series_sum_150_terms (a : ℕ) (r : ℝ)
  (h₁ : a = 250)
  (h₂ : (a - a * r ^ 50) / (1 - r) = 625)
  (h₃ : (a - a * r ^ 100) / (1 - r) = 1225) :
  (a - a * r ^ 150) / (1 - r) = 1801 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_150_terms_l2270_227062


namespace NUMINAMATH_GPT_connie_correct_answer_l2270_227032

theorem connie_correct_answer (y : ℕ) (h1 : y - 8 = 32) : y + 8 = 48 := by
  sorry

end NUMINAMATH_GPT_connie_correct_answer_l2270_227032


namespace NUMINAMATH_GPT_intersection_of_lines_l2270_227009

theorem intersection_of_lines
    (x y : ℚ) 
    (h1 : y = 3 * x - 1)
    (h2 : y + 4 = -6 * x) :
    x = -1 / 3 ∧ y = -2 := 
sorry

end NUMINAMATH_GPT_intersection_of_lines_l2270_227009


namespace NUMINAMATH_GPT_sequence_negation_l2270_227096

theorem sequence_negation (x : ℕ → ℝ) (x1_pos : x 1 > 0) (x1_neq1 : x 1 ≠ 1)
  (rec_seq : ∀ n : ℕ, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∃ n : ℕ, x n ≤ x (n + 1) :=
sorry

end NUMINAMATH_GPT_sequence_negation_l2270_227096


namespace NUMINAMATH_GPT_sum_at_simple_interest_l2270_227041

theorem sum_at_simple_interest (P R : ℝ) (h1: ((3 * P * (R + 1))/ 100) = ((3 * P * R) / 100 + 72)) : P = 2400 := 
by 
  sorry

end NUMINAMATH_GPT_sum_at_simple_interest_l2270_227041


namespace NUMINAMATH_GPT_reema_simple_interest_l2270_227075

-- Definitions and conditions
def principal : ℕ := 1200
def rate_of_interest : ℕ := 6
def time_period : ℕ := rate_of_interest

-- Simple interest calculation
def calculate_simple_interest (P R T: ℕ) : ℕ :=
  (P * R * T) / 100

-- The theorem to prove that Reema paid Rs 432 as simple interest.
theorem reema_simple_interest : calculate_simple_interest principal rate_of_interest time_period = 432 := 
  sorry

end NUMINAMATH_GPT_reema_simple_interest_l2270_227075


namespace NUMINAMATH_GPT_gumball_problem_l2270_227014
-- Step d: Lean 4 statement conversion

/-- 
  Suppose Joanna initially had 40 gumballs, Jacques had 60 gumballs, 
  and Julia had 80 gumballs.
  Joanna purchased 5 times the number of gumballs she initially had,
  Jacques purchased 3 times the number of gumballs he initially had,
  and Julia purchased 2 times the number of gumballs she initially had.
  Prove that after adding their purchases:
  1. Each person will have 240 gumballs.
  2. If they combine all their gumballs and share them equally, 
     each person will still get 240 gumballs.
-/
theorem gumball_problem :
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  (joanna_final = 240) ∧ (jacques_final = 240) ∧ (julia_final = 240) ∧ 
  (total_gumballs / 3 = 240) :=
by
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  
  have h_joanna : joanna_final = 240 := sorry
  have h_jacques : jacques_final = 240 := sorry
  have h_julia : julia_final = 240 := sorry
  have h_total : total_gumballs / 3 = 240 := sorry
  
  exact ⟨h_joanna, h_jacques, h_julia, h_total⟩

end NUMINAMATH_GPT_gumball_problem_l2270_227014


namespace NUMINAMATH_GPT_Laura_bought_one_kg_of_potatoes_l2270_227084

theorem Laura_bought_one_kg_of_potatoes :
  let price_salad : ℝ := 3
  let price_beef_per_kg : ℝ := 2 * price_salad
  let price_potato_per_kg : ℝ := price_salad * (1 / 3)
  let price_juice_per_liter : ℝ := 1.5
  let total_cost : ℝ := 22
  let num_salads : ℝ := 2
  let num_beef_kg : ℝ := 2
  let num_juice_liters : ℝ := 2
  let cost_salads := num_salads * price_salad
  let cost_beef := num_beef_kg * price_beef_per_kg
  let cost_juice := num_juice_liters * price_juice_per_liter
  (total_cost - (cost_salads + cost_beef + cost_juice)) / price_potato_per_kg = 1 :=
sorry

end NUMINAMATH_GPT_Laura_bought_one_kg_of_potatoes_l2270_227084


namespace NUMINAMATH_GPT_inequality_abc_l2270_227030

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l2270_227030


namespace NUMINAMATH_GPT_find_c_minus_d_l2270_227082

variable (g : ℝ → ℝ)
variable (c d : ℝ)
variable (invertible_g : Function.Injective g)
variable (g_at_c : g c = d)
variable (g_at_d : g d = 5)

theorem find_c_minus_d : c - d = -3 := by
  sorry

end NUMINAMATH_GPT_find_c_minus_d_l2270_227082


namespace NUMINAMATH_GPT_rachel_total_time_l2270_227013

-- Define the conditions
def num_chairs : ℕ := 20
def num_tables : ℕ := 8
def time_per_piece : ℕ := 6

-- Proof statement
theorem rachel_total_time : (num_chairs + num_tables) * time_per_piece = 168 := by
  sorry

end NUMINAMATH_GPT_rachel_total_time_l2270_227013


namespace NUMINAMATH_GPT_amount_after_two_years_l2270_227050

def present_value : ℝ := 70400
def rate : ℝ := 0.125
def years : ℕ := 2
def final_amount := present_value * (1 + rate) ^ years

theorem amount_after_two_years : final_amount = 89070 :=
by sorry

end NUMINAMATH_GPT_amount_after_two_years_l2270_227050


namespace NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l2270_227064

def line (x : ℝ) : ℝ := -x + 1

-- A line passes through the point (1, 0) and has a slope of -1
def passes_through_point (P : ℝ × ℝ) : Prop :=
  ∃ m b, m = -1 ∧ P.2 = m * P.1 + b ∧ line P.1 = P.2

def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ p : ℝ × ℝ, passes_through_point p ∧ third_quadrant p :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l2270_227064


namespace NUMINAMATH_GPT_negation_of_proposition_l2270_227031

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > b → a^2 > b^2) ↔ ∃ (a b : ℝ), a ≤ b ∧ a^2 ≤ b^2 :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l2270_227031


namespace NUMINAMATH_GPT_not_all_terms_positive_l2270_227066

variable (a b c d : ℝ)
variable (e f g h : ℝ)

theorem not_all_terms_positive
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (he : e < 0) (hf : f < 0) (hg : g < 0) (hh : h < 0) :
  ¬ ((a * e + b * c > 0) ∧ (e * f + c * g > 0) ∧ (f * d + g * h > 0) ∧ (d * a + h * b > 0)) :=
sorry

end NUMINAMATH_GPT_not_all_terms_positive_l2270_227066


namespace NUMINAMATH_GPT_storage_house_blocks_needed_l2270_227074

noncomputable def volume_of_storage_house
  (L_o : ℕ) (W_o : ℕ) (H_o : ℕ) (T : ℕ) : ℕ :=
  let interior_length := L_o - 2 * T
  let interior_width := W_o - 2 * T
  let interior_height := H_o - T
  let outer_volume := L_o * W_o * H_o
  let interior_volume := interior_length * interior_width * interior_height
  outer_volume - interior_volume

theorem storage_house_blocks_needed :
  volume_of_storage_house 15 12 8 2 = 912 :=
  by
    sorry

end NUMINAMATH_GPT_storage_house_blocks_needed_l2270_227074


namespace NUMINAMATH_GPT_computer_price_in_2016_l2270_227068

def price (p₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ := p₀ * (r ^ (n / 4))

theorem computer_price_in_2016 :
  price 8100 (2/3 : ℚ) 16 = 1600 :=
by
  sorry

end NUMINAMATH_GPT_computer_price_in_2016_l2270_227068


namespace NUMINAMATH_GPT_mike_total_investment_l2270_227079

variable (T : ℝ)
variable (H1 : 0.09 * 1800 + 0.11 * (T - 1800) = 624)

theorem mike_total_investment : T = 6000 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_investment_l2270_227079


namespace NUMINAMATH_GPT_range_of_smallest_side_l2270_227091

theorem range_of_smallest_side 
  (c : ℝ) -- the perimeter of the triangle
  (a : ℝ) (b : ℝ) (A : ℝ)  -- three sides of the triangle
  (ha : 0 < a) 
  (hb : b = 2 * a) 
  (hc : a + b + A = c)
  (htriangle : a + b > A ∧ a + A > b ∧ b + A > a) 
  : 
  ∃ (l u : ℝ), l = c / 6 ∧ u = c / 4 ∧ l < a ∧ a < u 
:= sorry

end NUMINAMATH_GPT_range_of_smallest_side_l2270_227091


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2270_227049

theorem problem_part1 :
  ∀ m : ℝ, (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
by
sorry

theorem problem_part2 :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧
    (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 ↔ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
by
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2270_227049


namespace NUMINAMATH_GPT_points_four_units_away_l2270_227001

theorem points_four_units_away (x : ℤ) : (x - (-1) = 4 ∨ x - (-1) = -4) ↔ (x = 3 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_points_four_units_away_l2270_227001


namespace NUMINAMATH_GPT_scientific_notation_of_0_00003_l2270_227042

theorem scientific_notation_of_0_00003 :
  0.00003 = 3 * 10^(-5) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_0_00003_l2270_227042


namespace NUMINAMATH_GPT_lcm_hcf_product_l2270_227058

theorem lcm_hcf_product (A B : ℕ) (h_prod : A * B = 18000) (h_hcf : Nat.gcd A B = 30) : Nat.lcm A B = 600 :=
sorry

end NUMINAMATH_GPT_lcm_hcf_product_l2270_227058


namespace NUMINAMATH_GPT_compare_abc_l2270_227061

noncomputable def a : ℝ := Real.sin (145 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (52 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (47 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l2270_227061


namespace NUMINAMATH_GPT_sum_squares_of_six_consecutive_even_eq_1420_l2270_227087

theorem sum_squares_of_six_consecutive_even_eq_1420 
  (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) :
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420 :=
by
  sorry

end NUMINAMATH_GPT_sum_squares_of_six_consecutive_even_eq_1420_l2270_227087


namespace NUMINAMATH_GPT_indigo_restaurant_average_rating_l2270_227094

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end NUMINAMATH_GPT_indigo_restaurant_average_rating_l2270_227094


namespace NUMINAMATH_GPT_C_and_D_complete_work_together_in_2_86_days_l2270_227092

def work_rate (days : ℕ) : ℚ := 1 / days

def A_rate := work_rate 4
def B_rate := work_rate 10
def D_rate := work_rate 5

noncomputable def C_rate : ℚ :=
  let combined_A_B_C_rate := A_rate + B_rate + (1 / (2 : ℚ))
  let C_rate := 1 / (20 / 3 : ℚ)  -- Solved from the equations provided in the solution
  C_rate

noncomputable def combined_C_D_rate := C_rate + D_rate

noncomputable def days_for_C_and_D_to_complete_work : ℚ :=
  1 / combined_C_D_rate

theorem C_and_D_complete_work_together_in_2_86_days :
  abs (days_for_C_and_D_to_complete_work - 2.86) < 0.01 := sorry

end NUMINAMATH_GPT_C_and_D_complete_work_together_in_2_86_days_l2270_227092


namespace NUMINAMATH_GPT_coefficient_m5_n5_in_expansion_l2270_227089

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_m5_n5_in_expansion_l2270_227089


namespace NUMINAMATH_GPT_seashells_total_l2270_227003

def seashells_sam : ℕ := 18
def seashells_mary : ℕ := 47
def seashells_john : ℕ := 32
def seashells_emily : ℕ := 26

theorem seashells_total : seashells_sam + seashells_mary + seashells_john + seashells_emily = 123 := by
    sorry

end NUMINAMATH_GPT_seashells_total_l2270_227003


namespace NUMINAMATH_GPT_rectangle_sides_l2270_227081

theorem rectangle_sides (x y : ℕ) (h_diff : x ≠ y) (h_eq : x * y = 2 * x + 2 * y) : 
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) :=
sorry

end NUMINAMATH_GPT_rectangle_sides_l2270_227081


namespace NUMINAMATH_GPT_triangle_least_perimeter_l2270_227095

theorem triangle_least_perimeter (x : ℤ) (h1 : x + 27 > 34) (h2 : 34 + 27 > x) (h3 : x + 34 > 27) : 27 + 34 + x ≥ 69 :=
by
  have h1' : x > 7 := by linarith
  sorry

end NUMINAMATH_GPT_triangle_least_perimeter_l2270_227095


namespace NUMINAMATH_GPT_Cindy_walking_speed_l2270_227076

noncomputable def walking_speed (total_time : ℕ) (running_speed : ℕ) (running_distance : ℚ) (walking_distance : ℚ) : ℚ := 
  let time_to_run := running_distance / running_speed
  let walking_time := total_time - (time_to_run * 60)
  walking_distance / (walking_time / 60)

theorem Cindy_walking_speed : walking_speed 40 3 0.5 0.5 = 1 := 
  sorry

end NUMINAMATH_GPT_Cindy_walking_speed_l2270_227076


namespace NUMINAMATH_GPT_dot_product_eq_one_l2270_227047

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_GPT_dot_product_eq_one_l2270_227047


namespace NUMINAMATH_GPT_number_of_math_students_l2270_227029

-- Definitions for the problem conditions
variables (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
variable (total_students_eq : total_students = 100)
variable (both_classes_eq : both_classes = 10)
variable (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))

-- Theorem statement
theorem number_of_math_students (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
  (total_students_eq : total_students = 100)
  (both_classes_eq : both_classes = 10)
  (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))
  (total_students_eq : total_students = physics_class + math_class - both_classes) :
  math_class = 88 :=
sorry

end NUMINAMATH_GPT_number_of_math_students_l2270_227029


namespace NUMINAMATH_GPT_average_rainfall_in_normal_year_l2270_227052

def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def rainfall_difference : ℕ := 58

theorem average_rainfall_in_normal_year :
  (total_rainfall_this_year + rainfall_difference) = 140 :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_in_normal_year_l2270_227052


namespace NUMINAMATH_GPT_find_simple_interest_rate_l2270_227059

variable (P : ℝ) (n : ℕ) (r_c : ℝ) (t : ℝ) (I_c : ℝ) (I_s : ℝ) (r_s : ℝ)

noncomputable def compound_interest_amount (P r_c : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r_c / n) ^ (n * t)

noncomputable def simple_interest_amount (P r_s : ℝ) (t : ℝ) : ℝ :=
  P * r_s * t

theorem find_simple_interest_rate
  (hP : P = 5000)
  (hr_c : r_c = 0.16)
  (hn : n = 2)
  (ht : t = 1)
  (hI_c : I_c = compound_interest_amount P r_c n t - P)
  (hI_s : I_s = I_c - 16)
  (hI_s_def : I_s = simple_interest_amount P r_s t) :
  r_s = 0.1632 := sorry

end NUMINAMATH_GPT_find_simple_interest_rate_l2270_227059


namespace NUMINAMATH_GPT_other_factor_of_lcm_l2270_227023

theorem other_factor_of_lcm (A B : ℕ) 
  (hcf : Nat.gcd A B = 23) 
  (hA : A = 345) 
  (hcf_factor : 15 ∣ Nat.lcm A B) 
  : 23 ∣ Nat.lcm A B / 15 :=
sorry

end NUMINAMATH_GPT_other_factor_of_lcm_l2270_227023


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2270_227060

theorem solution_set_of_inequality (a : ℝ) (h : 0 < a) :
  {x : ℝ | x ^ 2 - 4 * a * x - 5 * a ^ 2 < 0} = {x : ℝ | -a < x ∧ x < 5 * a} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2270_227060


namespace NUMINAMATH_GPT_percentage_charge_l2270_227026

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trip_charge : ℝ := 1.5
def number_of_trips : ℕ := 40
def grocery_value : ℝ := 800
def final_savings_needed : ℝ := car_cost - initial_savings

-- The amount earned from trips
def amount_from_trips : ℝ := number_of_trips * trip_charge

-- The amount needed from percentage charge on groceries
def amount_from_percentage (P: ℝ) : ℝ := grocery_value * P

-- The required amount from percentage charge on groceries
def required_amount_from_percentage : ℝ := final_savings_needed - amount_from_trips

theorem percentage_charge (P: ℝ) (h: amount_from_percentage P = required_amount_from_percentage) : P = 0.05 :=
by 
  -- Proof follows from the given condition that amount_from_percentage P = required_amount_from_percentage
  sorry

end NUMINAMATH_GPT_percentage_charge_l2270_227026


namespace NUMINAMATH_GPT_Emily_used_10_dimes_l2270_227099

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end NUMINAMATH_GPT_Emily_used_10_dimes_l2270_227099


namespace NUMINAMATH_GPT_lloyd_earnings_l2270_227005

theorem lloyd_earnings:
  let regular_hours := 7.5
  let regular_rate := 4.50
  let overtime_multiplier := 2.0
  let hours_worked := 10.5
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 60.75 :=
by
  sorry

end NUMINAMATH_GPT_lloyd_earnings_l2270_227005


namespace NUMINAMATH_GPT_Julie_work_hours_per_week_l2270_227022

variable (hours_per_week_summer : ℕ) (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (weeks_school_year : ℕ)
variable (earnings_school_year : ℕ)

theorem Julie_work_hours_per_week :
  hours_per_week_summer = 40 →
  weeks_summer = 10 →
  earnings_summer = 4000 →
  weeks_school_year = 40 →
  earnings_school_year = 4000 →
  (∀ rate_per_hour, rate_per_hour = earnings_summer / (hours_per_week_summer * weeks_summer) →
  (earnings_school_year / (weeks_school_year * rate_per_hour) = 10)) :=
by intros h1 h2 h3 h4 h5 rate_per_hour hr; sorry

end NUMINAMATH_GPT_Julie_work_hours_per_week_l2270_227022


namespace NUMINAMATH_GPT_initial_amount_liquid_A_l2270_227080

-- Definitions and conditions
def initial_ratio (a : ℕ) (b : ℕ) := a = 4 * b
def replaced_mixture_ratio (a : ℕ) (b : ℕ) (r₀ r₁ : ℕ) := 4 * r₀ = 2 * (r₁ + 20)

-- Theorem to prove the initial amount of liquid A
theorem initial_amount_liquid_A (a b r₀ r₁ : ℕ) :
  initial_ratio a b → replaced_mixture_ratio a b r₀ r₁ → a = 16 := 
by
  sorry

end NUMINAMATH_GPT_initial_amount_liquid_A_l2270_227080


namespace NUMINAMATH_GPT_tasks_completed_correctly_l2270_227015

theorem tasks_completed_correctly (x y : ℕ) (h1 : 9 * x - 5 * y = 57) (h2 : x + y ≤ 15) : x = 8 := 
by
  sorry

end NUMINAMATH_GPT_tasks_completed_correctly_l2270_227015


namespace NUMINAMATH_GPT_percentage_of_water_in_dried_grapes_l2270_227088

theorem percentage_of_water_in_dried_grapes 
  (weight_fresh : ℝ) 
  (weight_dried : ℝ) 
  (percentage_water_fresh : ℝ) 
  (solid_weight : ℝ)
  (water_weight_dried : ℝ) 
  (percentage_water_dried : ℝ) 
  (H1 : weight_fresh = 30) 
  (H2 : weight_dried = 15) 
  (H3 : percentage_water_fresh = 0.60) 
  (H4 : solid_weight = weight_fresh * (1 - percentage_water_fresh)) 
  (H5 : water_weight_dried = weight_dried - solid_weight) 
  (H6 : percentage_water_dried = (water_weight_dried / weight_dried) * 100) 
  : percentage_water_dried = 20 := 
  by { sorry }

end NUMINAMATH_GPT_percentage_of_water_in_dried_grapes_l2270_227088


namespace NUMINAMATH_GPT_smallest_integer_x_divisibility_l2270_227085

theorem smallest_integer_x_divisibility :
  ∃ x : ℤ, (2 * x + 2) % 33 = 0 ∧ (2 * x + 2) % 44 = 0 ∧ (2 * x + 2) % 55 = 0 ∧ (2 * x + 2) % 666 = 0 ∧ x = 36629 := 
sorry

end NUMINAMATH_GPT_smallest_integer_x_divisibility_l2270_227085


namespace NUMINAMATH_GPT_emir_needs_more_money_l2270_227073

noncomputable def dictionary_cost : ℝ := 5.50
noncomputable def dinosaur_book_cost : ℝ := 11.25
noncomputable def childrens_cookbook_cost : ℝ := 5.75
noncomputable def science_experiment_kit_cost : ℝ := 8.50
noncomputable def colored_pencils_cost : ℝ := 3.60
noncomputable def world_map_poster_cost : ℝ := 2.40
noncomputable def puzzle_book_cost : ℝ := 4.65
noncomputable def sketchpad_cost : ℝ := 6.20

noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def dinosaur_discount_rate : ℝ := 0.10
noncomputable def saved_amount : ℝ := 28.30

noncomputable def total_cost_before_tax : ℝ :=
  dictionary_cost +
  (dinosaur_book_cost - dinosaur_discount_rate * dinosaur_book_cost) +
  childrens_cookbook_cost +
  science_experiment_kit_cost +
  colored_pencils_cost +
  world_map_poster_cost +
  puzzle_book_cost +
  sketchpad_cost

noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_cost_before_tax

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + total_sales_tax

noncomputable def additional_amount_needed : ℝ := total_cost_after_tax - saved_amount

theorem emir_needs_more_money : additional_amount_needed = 21.81 := by
  sorry

end NUMINAMATH_GPT_emir_needs_more_money_l2270_227073


namespace NUMINAMATH_GPT_min_value_inequality_l2270_227057

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) : 
  (2 / a + 3 / b) ≥ 14 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l2270_227057


namespace NUMINAMATH_GPT_pq_inequality_l2270_227071

theorem pq_inequality (p : ℝ) (q : ℝ) (hp : 0 ≤ p) (hp2 : p < 2) (hq : q > 0) :
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q) / (p + q) > 3 * p^2 * q :=
by {
  sorry
}

end NUMINAMATH_GPT_pq_inequality_l2270_227071


namespace NUMINAMATH_GPT_original_bet_l2270_227039

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end NUMINAMATH_GPT_original_bet_l2270_227039


namespace NUMINAMATH_GPT_words_lost_equal_137_l2270_227097

-- Definitions based on conditions
def letters_in_oz : ℕ := 68
def forbidden_letter_index : ℕ := 7

def words_lost_due_to_forbidden_letter : ℕ :=
  let one_letter_words_lost : ℕ := 1
  let two_letter_words_lost : ℕ := 2 * (letters_in_oz - 1)
  one_letter_words_lost + two_letter_words_lost

-- Theorem stating that the words lost due to prohibition is 137
theorem words_lost_equal_137 :
  words_lost_due_to_forbidden_letter = 137 :=
sorry

end NUMINAMATH_GPT_words_lost_equal_137_l2270_227097


namespace NUMINAMATH_GPT_calories_in_300g_lemonade_l2270_227070

def lemonade_calories (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) (lemon_juice_cal : Nat) (sugar_cal : Nat) : Nat :=
  (lemon_juice_in_g * lemon_juice_cal / 100) + (sugar_in_g * sugar_cal / 100)

def total_weight (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) : Nat :=
  lemon_juice_in_g + sugar_in_g + water_in_g

theorem calories_in_300g_lemonade :
  (lemonade_calories 500 200 1000 30 400) * 300 / (total_weight 500 200 1000) = 168 := 
  by
    sorry

end NUMINAMATH_GPT_calories_in_300g_lemonade_l2270_227070


namespace NUMINAMATH_GPT_f_at_3_l2270_227033

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_of_2 : f 2 = 1
axiom f_rec (x : ℝ) : f (x + 2) = f x + f 2

theorem f_at_3 : f 3 = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_f_at_3_l2270_227033


namespace NUMINAMATH_GPT_total_marbles_l2270_227002

theorem total_marbles (boxes : ℕ) (marbles_per_box : ℕ) (h1 : boxes = 10) (h2 : marbles_per_box = 100) : (boxes * marbles_per_box = 1000) :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l2270_227002


namespace NUMINAMATH_GPT_population_increase_duration_l2270_227040

noncomputable def birth_rate := 6 / 2 -- people every 2 seconds = 3 people per second
noncomputable def death_rate := 2 / 2 -- people every 2 seconds = 1 person per second
noncomputable def net_increase_per_second := (birth_rate - death_rate) -- net increase per second

def total_net_increase := 172800

theorem population_increase_duration :
  (total_net_increase / net_increase_per_second) / 3600 = 24 :=
by
  sorry

end NUMINAMATH_GPT_population_increase_duration_l2270_227040


namespace NUMINAMATH_GPT_distinct_positive_roots_log_sum_eq_5_l2270_227083

theorem distinct_positive_roots_log_sum_eq_5 (a b : ℝ)
  (h : ∀ (x : ℝ), (8 * x ^ 3 + 6 * a * x ^ 2 + 3 * b * x + a = 0) → x > 0) 
  (h_sum : ∀ u v w : ℝ, (8 * u ^ 3 + 6 * a * u ^ 2 + 3 * b * u + a = 0) ∧
                       (8 * v ^ 3 + 6 * a * v ^ 2 + 3 * b * v + a = 0) ∧
                       (8 * w ^ 3 + 6 * a * w ^ 2 + 3 * b * w + a = 0) → 
                       u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ 
                       (Real.log (u) / Real.log (3) + Real.log (v) / Real.log (3) + Real.log (w) / Real.log (3) = 5)) :
  a = -1944 :=
sorry

end NUMINAMATH_GPT_distinct_positive_roots_log_sum_eq_5_l2270_227083


namespace NUMINAMATH_GPT_circle_standard_equation_l2270_227028

theorem circle_standard_equation:
  ∃ (x y : ℝ), ((x + 2) ^ 2 + (y - 1) ^ 2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_circle_standard_equation_l2270_227028


namespace NUMINAMATH_GPT_find_x_l2270_227020

noncomputable def x : ℝ := 20

def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := x / 100 * 150 - 20 = 10

theorem find_x (x : ℝ) : condition1 x ∧ condition2 x ↔ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2270_227020


namespace NUMINAMATH_GPT_avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l2270_227035

variable (c d : ℤ)
variable (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7 :
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7) / 7 = c + 7 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l2270_227035


namespace NUMINAMATH_GPT_benny_start_cards_l2270_227098

--- Benny bought 4 new cards before the dog ate half of his collection.
def new_cards : Int := 4

--- The remaining cards after the dog ate half of the collection is 34.
def remaining_cards : Int := 34

--- The total number of cards Benny had before adding the new cards and the dog ate half.
def total_before_eating := remaining_cards * 2

theorem benny_start_cards : total_before_eating - new_cards = 64 :=
sorry

end NUMINAMATH_GPT_benny_start_cards_l2270_227098


namespace NUMINAMATH_GPT_length_of_MN_eq_5_sqrt_10_div_3_l2270_227000

theorem length_of_MN_eq_5_sqrt_10_div_3 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (D : ℝ × ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hyp_A : A = (1, 3))
  (hyp_B : B = (25 / 3, 5 / 3))
  (hyp_C : C = (22 / 3, 14 / 3))
  (hyp_eq_edges : (dist (0, 0) M = dist M N) ∧ (dist M N = dist N B))
  (hyp_D : D = (5 / 2, 15 / 2))
  (hyp_M : M = (5 / 3, 5)) :
  dist M N = 5 * Real.sqrt 10 / 3 :=
sorry

end NUMINAMATH_GPT_length_of_MN_eq_5_sqrt_10_div_3_l2270_227000


namespace NUMINAMATH_GPT_circus_dogs_ratio_l2270_227090

theorem circus_dogs_ratio :
  ∀ (x y : ℕ), 
  (x + y = 12) → (2 * x + 4 * y = 36) → (x = y) → x / y = 1 :=
by
  intros x y h1 h2 h3
  sorry

end NUMINAMATH_GPT_circus_dogs_ratio_l2270_227090


namespace NUMINAMATH_GPT_susie_initial_amount_l2270_227007

-- Definitions for conditions:
def initial_amount (X : ℝ) : Prop :=
  X + 0.20 * X = 240

-- Main theorem to prove:
theorem susie_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by 
  -- structured proof will go here
  sorry

end NUMINAMATH_GPT_susie_initial_amount_l2270_227007


namespace NUMINAMATH_GPT_nina_money_l2270_227011

theorem nina_money (C : ℝ) (h1 : C > 0) (h2 : 6 * C = 8 * (C - 2)) : 6 * C = 48 :=
by
  sorry

end NUMINAMATH_GPT_nina_money_l2270_227011


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_range_quadratic_root_product_value_l2270_227019

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_range_quadratic_root_product_value_l2270_227019


namespace NUMINAMATH_GPT_number_exceeds_25_percent_by_150_l2270_227008

theorem number_exceeds_25_percent_by_150 (x : ℝ) : (0.25 * x + 150 = x) → x = 200 :=
by
  sorry

end NUMINAMATH_GPT_number_exceeds_25_percent_by_150_l2270_227008


namespace NUMINAMATH_GPT_joan_original_seashells_l2270_227065

-- Definitions based on the conditions
def seashells_left : ℕ := 27
def seashells_given_away : ℕ := 43

-- Theorem statement
theorem joan_original_seashells : 
  seashells_left + seashells_given_away = 70 := 
by
  sorry

end NUMINAMATH_GPT_joan_original_seashells_l2270_227065


namespace NUMINAMATH_GPT_value_of_x_l2270_227046

theorem value_of_x (x : ℤ) : (x + 1) * (x + 1) = 16 ↔ (x = 3 ∨ x = -5) := 
by sorry

end NUMINAMATH_GPT_value_of_x_l2270_227046


namespace NUMINAMATH_GPT_cost_of_kid_ticket_l2270_227044

theorem cost_of_kid_ticket (total_people kids adults : ℕ) 
  (adult_ticket_cost kid_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (h_people : total_people = kids + adults)
  (h_adult_cost : adult_ticket_cost = 28)
  (h_kids : kids = 203)
  (h_total_sales : total_sales = 3864)
  (h_calculate_sales : adults * adult_ticket_cost + kids * kid_ticket_cost = total_sales)
  : kid_ticket_cost = 12 :=
by
  sorry -- Proof will be filled in

end NUMINAMATH_GPT_cost_of_kid_ticket_l2270_227044


namespace NUMINAMATH_GPT_rider_distance_traveled_l2270_227077

noncomputable def caravan_speed := 1  -- km/h
noncomputable def rider_speed := 1 + Real.sqrt 2  -- km/h

theorem rider_distance_traveled : 
  (1 / (rider_speed - 1) + 1 / (rider_speed + 1)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_rider_distance_traveled_l2270_227077


namespace NUMINAMATH_GPT_vector_addition_dot_product_l2270_227037

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

theorem vector_addition :
  let c := (1, 2) + (3, 1)
  c = (4, 3) := by
  sorry

theorem dot_product :
  let d := (1 * 3 + 2 * 1)
  d = 5 := by
  sorry

end NUMINAMATH_GPT_vector_addition_dot_product_l2270_227037


namespace NUMINAMATH_GPT_sum_of_interior_angles_l2270_227012

theorem sum_of_interior_angles (n : ℕ) (interior_angle : ℝ) :
  (interior_angle = 144) → (180 - 144) * n = 360 → n = 10 → (n - 2) * 180 = 1440 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l2270_227012


namespace NUMINAMATH_GPT_value_of_expression_l2270_227055

open Real

theorem value_of_expression {a : ℝ} (h : a^2 + 4 * a - 5 = 0) : 3 * a^2 + 12 * a = 15 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l2270_227055


namespace NUMINAMATH_GPT_expected_adjacent_black_l2270_227021

noncomputable def ExpectedBlackPairs :=
  let totalCards := 104
  let blackCards := 52
  let totalPairs := 103
  let probAdjacentBlack := (blackCards - 1) / (totalPairs)
  blackCards * probAdjacentBlack

theorem expected_adjacent_black :
  ExpectedBlackPairs = 2601 / 103 :=
by
  sorry

end NUMINAMATH_GPT_expected_adjacent_black_l2270_227021


namespace NUMINAMATH_GPT_group_c_right_angled_triangle_l2270_227086

theorem group_c_right_angled_triangle :
  (3^2 + 4^2 = 5^2) := by
  sorry

end NUMINAMATH_GPT_group_c_right_angled_triangle_l2270_227086


namespace NUMINAMATH_GPT_probability_spade_then_ace_l2270_227017

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end NUMINAMATH_GPT_probability_spade_then_ace_l2270_227017


namespace NUMINAMATH_GPT_reduced_price_correct_l2270_227069

theorem reduced_price_correct (P R Q: ℝ) (h1 : R = 0.75 * P) (h2 : 900 = Q * P) (h3 : 900 = (Q + 5) * R)  :
  R = 45 := by 
  sorry

end NUMINAMATH_GPT_reduced_price_correct_l2270_227069


namespace NUMINAMATH_GPT_tan_product_l2270_227034

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_l2270_227034


namespace NUMINAMATH_GPT_diagonal_cells_crossed_l2270_227072

theorem diagonal_cells_crossed (m n : ℕ) (h_m : m = 199) (h_n : n = 991) :
  (m + n - Nat.gcd m n) = 1189 := by
  sorry

end NUMINAMATH_GPT_diagonal_cells_crossed_l2270_227072


namespace NUMINAMATH_GPT_Justin_run_home_time_l2270_227038

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end NUMINAMATH_GPT_Justin_run_home_time_l2270_227038


namespace NUMINAMATH_GPT_ordered_pair_l2270_227018

-- Definitions
def P (x : ℝ) := x^4 - 8 * x^3 + 20 * x^2 - 34 * x + 15
def D (k : ℝ) (x : ℝ) := x^2 - 3 * x + k
def R (a : ℝ) (x : ℝ) := x + a

-- Hypothesis
def condition (k a : ℝ) : Prop := ∀ x : ℝ, P x % D k x = R a x

-- Theorem
theorem ordered_pair (k a : ℝ) (h : condition k a) : (k, a) = (5, 15) := 
  sorry

end NUMINAMATH_GPT_ordered_pair_l2270_227018


namespace NUMINAMATH_GPT_find_height_of_door_l2270_227025

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_height_of_door_l2270_227025


namespace NUMINAMATH_GPT_arithmetic_sequence_condition_l2270_227051

theorem arithmetic_sequence_condition (a : ℕ → ℝ) :
  (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2)) ↔
  (∀ n ∈ {k : ℕ | k > 0}, a (n+1) - a n = a (n+2) - a (n+1)) ∧ ¬ (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2) → a (n+1) = a n) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_condition_l2270_227051


namespace NUMINAMATH_GPT_pastries_solution_l2270_227010

def pastries_problem : Prop :=
  ∃ (F Calvin Phoebe Grace : ℕ),
  (Calvin = F + 8) ∧
  (Phoebe = F + 8) ∧
  (Grace = 30) ∧
  (F + Calvin + Phoebe + Grace = 97) ∧
  (Grace - Calvin = 5) ∧
  (Grace - Phoebe = 5)

theorem pastries_solution : pastries_problem :=
by
  sorry

end NUMINAMATH_GPT_pastries_solution_l2270_227010


namespace NUMINAMATH_GPT_wood_length_equation_l2270_227016

theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end NUMINAMATH_GPT_wood_length_equation_l2270_227016


namespace NUMINAMATH_GPT_find_n_22_or_23_l2270_227093

theorem find_n_22_or_23 (n : ℕ) : 
  (∃ (sol_count : ℕ), sol_count = 30 ∧ (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 2 * y + 4 * z = n)) → 
  (n = 22 ∨ n = 23) := 
sorry

end NUMINAMATH_GPT_find_n_22_or_23_l2270_227093


namespace NUMINAMATH_GPT_expression_value_l2270_227067

theorem expression_value (x : ℝ) (h : x = 4) :
  (x^2 - 2*x - 15) / (x - 5) = 7 :=
sorry

end NUMINAMATH_GPT_expression_value_l2270_227067


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l2270_227078

def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (2, -7)

theorem area_of_triangle_ABC : 
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := |v.1 * w.2 - v.2 * w.1|
  let triangle_area := parallelogram_area / 2
  triangle_area = 15 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l2270_227078


namespace NUMINAMATH_GPT_g_at_5_l2270_227004

def g (x : ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 30 * x^3 - 45 * x^2 + 24 * x + 50

theorem g_at_5 : g 5 = 2795 :=
by
  sorry

end NUMINAMATH_GPT_g_at_5_l2270_227004


namespace NUMINAMATH_GPT_cats_weight_difference_l2270_227027

-- Define the weights of Anne's and Meg's cats
variables (A M : ℕ)

-- Given conditions:
-- 1. Ratio of weights Meg's cat to Anne's cat is 13:21
-- 2. Meg's cat's weight is 20 kg plus half the weight of Anne's cat

theorem cats_weight_difference (h1 : M = 20 + (A / 2)) (h2 : 13 * A = 21 * M) : A - M = 64 := 
by {
    sorry
}

end NUMINAMATH_GPT_cats_weight_difference_l2270_227027


namespace NUMINAMATH_GPT_Jungkook_fewest_erasers_l2270_227048

-- Define the number of erasers each person has.
def Jungkook_erasers : ℕ := 6
def Jimin_erasers : ℕ := Jungkook_erasers + 4
def Seokjin_erasers : ℕ := Jimin_erasers - 3

-- Prove that Jungkook has the fewest erasers.
theorem Jungkook_fewest_erasers : Jungkook_erasers < Jimin_erasers ∧ Jungkook_erasers < Seokjin_erasers :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Jungkook_fewest_erasers_l2270_227048


namespace NUMINAMATH_GPT_total_milk_bottles_l2270_227043

theorem total_milk_bottles (marcus_bottles : ℕ) (john_bottles : ℕ) (h1 : marcus_bottles = 25) (h2 : john_bottles = 20) : marcus_bottles + john_bottles = 45 := by
  sorry

end NUMINAMATH_GPT_total_milk_bottles_l2270_227043


namespace NUMINAMATH_GPT_fg_at_3_l2270_227056

def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := x^2 + 5

theorem fg_at_3 : f (g 3) = 10 := by
  sorry

end NUMINAMATH_GPT_fg_at_3_l2270_227056


namespace NUMINAMATH_GPT_find_b_l2270_227063

theorem find_b (p : ℕ) (hp : Nat.Prime p) :
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p ∧ ∀ (x1 x2 : ℤ), x1 * x2 = p * b ∧ x1 + x2 = b) → 
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p) :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2270_227063


namespace NUMINAMATH_GPT_conjecture_l2270_227045

noncomputable def f (x : ℝ) : ℝ :=
  1 / (3^x + Real.sqrt 3)

theorem conjecture (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_conjecture_l2270_227045


namespace NUMINAMATH_GPT_percentage_difference_j_p_l2270_227053

theorem percentage_difference_j_p (j p t : ℝ) (h1 : j = t * 80 / 100) 
  (h2 : t = p * (100 - t) / 100) (h3 : t = 6.25) : 
  ((p - j) / p) * 100 = 25 := 
by
  sorry

end NUMINAMATH_GPT_percentage_difference_j_p_l2270_227053


namespace NUMINAMATH_GPT_mateo_orange_bottles_is_1_l2270_227036

def number_of_orange_bottles_mateo_has (mateo_orange : ℕ) : Prop :=
  let julios_orange_bottles := 4
  let julios_grape_bottles := 7
  let mateos_grape_bottles := 3
  let liters_per_bottle := 2
  let julios_total_liters := (julios_orange_bottles + julios_grape_bottles) * liters_per_bottle
  let mateos_grape_liters := mateos_grape_bottles * liters_per_bottle
  let mateos_total_liters := (mateo_orange * liters_per_bottle) + mateos_grape_liters
  let additional_liters_to_julio := 14
  julios_total_liters = mateos_total_liters + additional_liters_to_julio

/-
Prove that Mateo has exactly 1 bottle of orange soda (assuming the problem above)
-/
theorem mateo_orange_bottles_is_1 : number_of_orange_bottles_mateo_has 1 :=
sorry

end NUMINAMATH_GPT_mateo_orange_bottles_is_1_l2270_227036
