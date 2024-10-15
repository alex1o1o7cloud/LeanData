import Mathlib

namespace NUMINAMATH_GPT_solve_for_multiplier_l728_72824

namespace SashaSoup
  
-- Variables representing the amounts of salt
variables (x y : ℝ)

-- Condition provided: amount of salt added today
def initial_salt := 2 * x
def additional_salt_today := 0.5 * y

-- Given relationship
axiom salt_relationship : x = 0.5 * y

-- The multiplier k to achieve the required amount of salt
def required_multiplier : ℝ := 1.5

-- Lean theorem statement
theorem solve_for_multiplier :
  (2 * x) * required_multiplier = x + y :=
by
  -- Mathematical proof goes here but since asked to skip proof we use sorry
  sorry

end SashaSoup

end NUMINAMATH_GPT_solve_for_multiplier_l728_72824


namespace NUMINAMATH_GPT_bicycle_has_four_wheels_l728_72878

-- Define the universe and properties of cars
axiom Car : Type
axiom Bicycle : Car
axiom has_four_wheels : Car → Prop
axiom all_cars_have_four_wheels : ∀ c : Car, has_four_wheels c

-- Define the theorem
theorem bicycle_has_four_wheels : has_four_wheels Bicycle :=
by
  sorry

end NUMINAMATH_GPT_bicycle_has_four_wheels_l728_72878


namespace NUMINAMATH_GPT_math_problem_A_B_M_l728_72893

theorem math_problem_A_B_M :
  ∃ M : Set ℝ,
    M = {m | ∃ A B : Set ℝ,
      A = {x | x^2 - 5 * x + 6 = 0} ∧
      B = {x | m * x - 1 = 0} ∧
      A ∩ B = B ∧
      M = {0, (1:ℝ)/2, (1:ℝ)/3}} ∧
    ∃ subsets : Set (Set ℝ),
      subsets = {∅, {0}, {(1:ℝ)/2}, {(1:ℝ)/3}, {0, (1:ℝ)/2}, {(1:ℝ)/2, (1:ℝ)/3}, {0, (1:ℝ)/3}, {0, (1:ℝ)/2, (1:ℝ)/3}} :=
by
  sorry

end NUMINAMATH_GPT_math_problem_A_B_M_l728_72893


namespace NUMINAMATH_GPT_inverse_function_condition_l728_72897

noncomputable def f (m x : ℝ) := (3 * x + 4) / (m * x - 5)

theorem inverse_function_condition (m : ℝ) :
  (∀ x : ℝ, f m (f m x) = x) ↔ m = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_condition_l728_72897


namespace NUMINAMATH_GPT_ten_person_round_robin_l728_72823

def number_of_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem ten_person_round_robin : number_of_matches 10 = 45 :=
by
  -- Proof steps would go here, but are omitted for this task
  sorry

end NUMINAMATH_GPT_ten_person_round_robin_l728_72823


namespace NUMINAMATH_GPT_erased_number_is_six_l728_72828

theorem erased_number_is_six (n x : ℕ) (h1 : (n * (n + 1)) / 2 - x = 45 * (n - 1) / 4):
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_erased_number_is_six_l728_72828


namespace NUMINAMATH_GPT_g_h_2_eq_583_l728_72850

def g (x : ℝ) : ℝ := 3*x^2 - 5

def h (x : ℝ) : ℝ := -2*x^3 + 2

theorem g_h_2_eq_583 : g (h 2) = 583 :=
by
  sorry

end NUMINAMATH_GPT_g_h_2_eq_583_l728_72850


namespace NUMINAMATH_GPT_x_add_inv_ge_two_x_add_inv_eq_two_iff_l728_72875

theorem x_add_inv_ge_two {x : ℝ} (h : 0 < x) : x + (1 / x) ≥ 2 :=
sorry

theorem x_add_inv_eq_two_iff {x : ℝ} (h : 0 < x) : (x + (1 / x) = 2) ↔ (x = 1) :=
sorry

end NUMINAMATH_GPT_x_add_inv_ge_two_x_add_inv_eq_two_iff_l728_72875


namespace NUMINAMATH_GPT_AM_GM_inequality_example_l728_72899

theorem AM_GM_inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 6) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_example_l728_72899


namespace NUMINAMATH_GPT_pentadecagon_diagonals_l728_72853

def numberOfDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentadecagon_diagonals : numberOfDiagonals 15 = 90 :=
by
  sorry

end NUMINAMATH_GPT_pentadecagon_diagonals_l728_72853


namespace NUMINAMATH_GPT_geometric_sequence_sum_l728_72806

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h2 : a 3 + a 5 = 6) :
  a 5 + a 7 + a 9 = 28 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l728_72806


namespace NUMINAMATH_GPT_circle_center_radius_sum_l728_72864

theorem circle_center_radius_sum (u v s : ℝ) (h1 : (x + 4)^2 + (y - 1)^2 = 13)
    (h2 : (u, v) = (-4, 1)) (h3 : s = Real.sqrt 13) : 
    u + v + s = -3 + Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_sum_l728_72864


namespace NUMINAMATH_GPT_find_a_l728_72839

-- Given Conditions
def is_hyperbola (a : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a) - (y^2 / 2) = 1
def is_asymptote (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = 2 * x

-- Question
theorem find_a (a : ℝ) (f : ℝ → ℝ) (hyp : is_hyperbola a) (asym : is_asymptote f) : a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_l728_72839


namespace NUMINAMATH_GPT_extended_cross_cannot_form_cube_l728_72881

-- Define what it means to form a cube from patterns
def forms_cube (pattern : Type) : Prop := 
  sorry -- Definition for forming a cube would be detailed here

-- Define the Extended Cross pattern in a way that captures its structure
def extended_cross : Type := sorry -- Definition for Extended Cross structure

-- Define the L shape pattern in a way that captures its structure
def l_shape : Type := sorry -- Definition for L shape structure

-- The theorem statement proving that the Extended Cross pattern cannot form a cube
theorem extended_cross_cannot_form_cube : ¬(forms_cube extended_cross) := 
  sorry

end NUMINAMATH_GPT_extended_cross_cannot_form_cube_l728_72881


namespace NUMINAMATH_GPT_geometric_series_r_l728_72882

theorem geometric_series_r (a r : ℝ) 
    (h1 : a * (1 - r ^ 0) / (1 - r) = 24) 
    (h2 : a * r / (1 - r ^ 2) = 8) : 
    r = 1 / 2 := 
sorry

end NUMINAMATH_GPT_geometric_series_r_l728_72882


namespace NUMINAMATH_GPT_sum_of_arithmetic_series_l728_72868

theorem sum_of_arithmetic_series (A B C : ℕ) (n : ℕ) 
  (hA : A = n * (2 * a₁ + (n - 1) * d) / 2)
  (hB : B = 2 * n * (2 * a₁ + (2 * n - 1) * d) / 2)
  (hC : C = 3 * n * (2 * a₁ + (3 * n - 1) * d) / 2) :
  C = 3 * (B - A) := sorry

end NUMINAMATH_GPT_sum_of_arithmetic_series_l728_72868


namespace NUMINAMATH_GPT_parameter_condition_l728_72859

theorem parameter_condition (a : ℝ) :
  let D := 4 - 4 * a
  let diff_square := ((-2 / a) ^ 2 - 4 * (1 / a))
  D = 9 * diff_square -> a = -3 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_parameter_condition_l728_72859


namespace NUMINAMATH_GPT_range_of_2a_plus_3b_l728_72845

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 :=
  sorry

end NUMINAMATH_GPT_range_of_2a_plus_3b_l728_72845


namespace NUMINAMATH_GPT_polynomial_calculation_l728_72880

theorem polynomial_calculation :
  (49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1) = 254804368 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_calculation_l728_72880


namespace NUMINAMATH_GPT_problem_statement_l728_72836

theorem problem_statement
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023 * (c * d) - (a + b) / 20 = 2024 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l728_72836


namespace NUMINAMATH_GPT_lara_bag_total_chips_l728_72818

theorem lara_bag_total_chips (C : ℕ)
  (h1 : ∃ (b : ℕ), b = C / 6)
  (h2 : 34 + 16 + C / 6 = C) :
  C = 60 := by
  sorry

end NUMINAMATH_GPT_lara_bag_total_chips_l728_72818


namespace NUMINAMATH_GPT_jessica_journey_total_distance_l728_72808

theorem jessica_journey_total_distance
  (y : ℝ)
  (h1 : y = (y / 4) + 25 + (y / 4)) :
  y = 50 :=
by
  sorry

end NUMINAMATH_GPT_jessica_journey_total_distance_l728_72808


namespace NUMINAMATH_GPT_absolute_difference_l728_72854

theorem absolute_difference : |8 - 3^2| - |4^2 - 6*3| = -1 := by
  sorry

end NUMINAMATH_GPT_absolute_difference_l728_72854


namespace NUMINAMATH_GPT_yearly_savings_l728_72825

-- Define the various constants given in the problem
def weeks_in_year : ℕ := 52
def months_in_year : ℕ := 12
def non_peak_weeks : ℕ := 16
def peak_weeks : ℕ := weeks_in_year - non_peak_weeks
def non_peak_months : ℕ := 4
def peak_months : ℕ := months_in_year - non_peak_months

-- Rates
def weekly_cost_non_peak_large : ℕ := 10
def weekly_cost_peak_large : ℕ := 12
def monthly_cost_non_peak_large : ℕ := 42
def monthly_cost_peak_large : ℕ := 48

-- Additional surcharge
def holiday_weeks : ℕ := 6
def holiday_surcharge : ℕ := 2

-- Compute the yearly costs
def yearly_weekly_cost : ℕ :=
  (non_peak_weeks * weekly_cost_non_peak_large) +
  (peak_weeks * weekly_cost_peak_large) +
  (holiday_weeks * (holiday_surcharge + weekly_cost_peak_large))

def yearly_monthly_cost : ℕ :=
  (non_peak_months * monthly_cost_non_peak_large) +
  (peak_months * monthly_cost_peak_large)

theorem yearly_savings : yearly_weekly_cost - yearly_monthly_cost = 124 := by
  sorry

end NUMINAMATH_GPT_yearly_savings_l728_72825


namespace NUMINAMATH_GPT_jerry_charge_per_hour_l728_72840

-- Define the conditions from the problem
def time_painting : ℝ := 8
def time_fixing_counter : ℝ := 3 * time_painting
def time_mowing_lawn : ℝ := 6
def total_time_worked : ℝ := time_painting + time_fixing_counter + time_mowing_lawn
def total_payment : ℝ := 570

-- The proof statement
theorem jerry_charge_per_hour : 
  total_payment / total_time_worked = 15 :=
by
  sorry

end NUMINAMATH_GPT_jerry_charge_per_hour_l728_72840


namespace NUMINAMATH_GPT_second_investment_amount_l728_72842

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

theorem second_investment_amount :
  ∀ (P₁ P₂ I₁ I₂ r t : ℝ), 
    P₁ = 5000 →
    I₁ = 250 →
    I₂ = 1000 →
    I₁ = simple_interest P₁ r t →
    I₂ = simple_interest P₂ r t →
    P₂ = 20000 := 
by 
  intros P₁ P₂ I₁ I₂ r t hP₁ hI₁ hI₂ hI₁_eq hI₂_eq
  sorry

end NUMINAMATH_GPT_second_investment_amount_l728_72842


namespace NUMINAMATH_GPT_remainder_eq_159_l728_72870

def x : ℕ := 2^40
def numerator : ℕ := 2^160 + 160
def denominator : ℕ := 2^80 + 2^40 + 1

theorem remainder_eq_159 : (numerator % denominator) = 159 := 
by {
  -- Proof will be filled in here.
  sorry
}

end NUMINAMATH_GPT_remainder_eq_159_l728_72870


namespace NUMINAMATH_GPT_length_of_room_calculation_l728_72837

variable (broadness_of_room : ℝ) (width_of_carpet : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) (area_of_carpet : ℝ) (length_of_room : ℝ)

theorem length_of_room_calculation (h1 : broadness_of_room = 9) 
    (h2 : width_of_carpet = 0.75) 
    (h3 : total_cost = 1872) 
    (h4 : rate_per_sq_meter = 12) 
    (h5 : area_of_carpet = total_cost / rate_per_sq_meter)
    (h6 : area_of_carpet = length_of_room * width_of_carpet) 
    : length_of_room = 208 := 
by 
    sorry

end NUMINAMATH_GPT_length_of_room_calculation_l728_72837


namespace NUMINAMATH_GPT_raised_bed_height_l728_72801

theorem raised_bed_height : 
  ∀ (total_planks : ℕ) (num_beds : ℕ) (planks_per_bed : ℕ) (height : ℚ),
  total_planks = 50 →
  num_beds = 10 →
  planks_per_bed = 4 * height →
  (total_planks = num_beds * planks_per_bed) →
  height = 5 / 4 :=
by
  intros total_planks num_beds planks_per_bed H
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_raised_bed_height_l728_72801


namespace NUMINAMATH_GPT_find_fourth_power_sum_l728_72848

theorem find_fourth_power_sum (a b c : ℝ) 
    (h1 : a + b + c = 2) 
    (h2 : a^2 + b^2 + c^2 = 3) 
    (h3 : a^3 + b^3 + c^3 = 4) : 
    a^4 + b^4 + c^4 = 7.833 :=
sorry

end NUMINAMATH_GPT_find_fourth_power_sum_l728_72848


namespace NUMINAMATH_GPT_greatest_integer_difference_l728_72879

theorem greatest_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∀ d : ℤ, (d = y - x) → d ≤ 6 := 
sorry

end NUMINAMATH_GPT_greatest_integer_difference_l728_72879


namespace NUMINAMATH_GPT_empty_set_implies_a_range_l728_72887

theorem empty_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(a * x^2 - 2 * a * x + 1 < 0)) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end NUMINAMATH_GPT_empty_set_implies_a_range_l728_72887


namespace NUMINAMATH_GPT_remainder_of_concatenated_numbers_l728_72867

def concatenatedNumbers : ℕ :=
  let digits := List.range (50) -- [0, 1, 2, ..., 49]
  digits.foldl (fun acc d => acc * 10 ^ (Nat.digits 10 d).length + d) 0

theorem remainder_of_concatenated_numbers :
  concatenatedNumbers % 50 = 49 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_concatenated_numbers_l728_72867


namespace NUMINAMATH_GPT_probability_of_yellow_or_green_l728_72817

def bag : List (String × Nat) := [("yellow", 4), ("green", 3), ("red", 2), ("blue", 1)]

def total_marbles (bag : List (String × Nat)) : Nat := bag.foldr (fun (_, n) acc => n + acc) 0

def favorable_outcomes (bag : List (String × Nat)) : Nat :=
  (bag.filter (fun (color, _) => color = "yellow" ∨ color = "green")).foldr (fun (_, n) acc => n + acc) 0

theorem probability_of_yellow_or_green :
  (favorable_outcomes bag : ℚ) / (total_marbles bag : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_of_yellow_or_green_l728_72817


namespace NUMINAMATH_GPT_find_value_of_a_l728_72819

theorem find_value_of_a (a : ℤ) (h : ∀ x : ℚ,  x^6 - 33 * x + 20 = (x^2 - x + a) * (x^4 + b * x^3 + c * x^2 + d * x + e)) :
  a = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_of_a_l728_72819


namespace NUMINAMATH_GPT_age_of_15th_student_l728_72833

theorem age_of_15th_student (avg_age_15 avg_age_3 avg_age_11 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_3 : avg_age_3 = 14) 
  (h_avg_11 : avg_age_11 = 16) : 
  ∃ x : ℕ, x = 7 := 
by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l728_72833


namespace NUMINAMATH_GPT_socks_thrown_away_l728_72805

theorem socks_thrown_away 
  (initial_socks new_socks current_socks : ℕ) 
  (h1 : initial_socks = 11) 
  (h2 : new_socks = 26) 
  (h3 : current_socks = 33) : 
  initial_socks + new_socks - current_socks = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_socks_thrown_away_l728_72805


namespace NUMINAMATH_GPT_current_dogwood_trees_l728_72855

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end NUMINAMATH_GPT_current_dogwood_trees_l728_72855


namespace NUMINAMATH_GPT_phone_answered_before_fifth_ring_l728_72852

theorem phone_answered_before_fifth_ring:
  (0.1 + 0.2 + 0.25 + 0.25 = 0.8) :=
by
  sorry

end NUMINAMATH_GPT_phone_answered_before_fifth_ring_l728_72852


namespace NUMINAMATH_GPT_find_xyz_l728_72814

theorem find_xyz (x y z : ℝ) (h1 : x * (y + z) = 195) (h2 : y * (z + x) = 204) (h3 : z * (x + y) = 213) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * y * z = 1029 := by
  sorry

end NUMINAMATH_GPT_find_xyz_l728_72814


namespace NUMINAMATH_GPT_excircle_identity_l728_72844

variables (a b c r_a r_b r_c : ℝ)

-- Conditions: r_a, r_b, r_c are the radii of the excircles opposite vertices A, B, and C respectively.
-- In the triangle ABC, a, b, c are the sides opposite vertices A, B, and C respectively.

theorem excircle_identity:
  (a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b))) = 2 :=
by
  sorry

end NUMINAMATH_GPT_excircle_identity_l728_72844


namespace NUMINAMATH_GPT_sum_abs_a1_to_a10_l728_72843

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end NUMINAMATH_GPT_sum_abs_a1_to_a10_l728_72843


namespace NUMINAMATH_GPT_distance_traveled_on_second_day_l728_72807

theorem distance_traveled_on_second_day 
  (a₁ : ℝ) 
  (h_sum : a₁ + a₁ / 2 + a₁ / 4 + a₁ / 8 + a₁ / 16 + a₁ / 32 = 189) 
  : a₁ / 2 = 48 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_on_second_day_l728_72807


namespace NUMINAMATH_GPT_four_digit_property_l728_72898

-- Define the problem conditions and statement
theorem four_digit_property (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 0 ≤ y ∧ y < 100) :
  (100 * x + y = (x + y) ^ 2) ↔ (100 * x + y = 3025 ∨ 100 * x + y = 2025 ∨ 100 * x + y = 9801) := by
sorry

end NUMINAMATH_GPT_four_digit_property_l728_72898


namespace NUMINAMATH_GPT_mean_of_y_l728_72888

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def regression_line (x : ℝ) : ℝ :=
  2 * x + 45

theorem mean_of_y (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  mean [regression_line 1, regression_line 5, regression_line 7, regression_line 13, regression_line 19] = 63 := by
  sorry

end NUMINAMATH_GPT_mean_of_y_l728_72888


namespace NUMINAMATH_GPT_mode_of_data_set_is_60_l728_72862

theorem mode_of_data_set_is_60
  (data : List ℕ := [65, 60, 75, 60, 80])
  (mode : ℕ := 60) :
  mode = 60 ∧ (∀ x ∈ data, data.count x ≤ data.count 60) :=
by {
  sorry
}

end NUMINAMATH_GPT_mode_of_data_set_is_60_l728_72862


namespace NUMINAMATH_GPT_difference_of_squares_l728_72857

theorem difference_of_squares (x y : ℝ) (h₁ : x + y = 20) (h₂ : x - y = 10) : x^2 - y^2 = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_of_squares_l728_72857


namespace NUMINAMATH_GPT_age_ratio_l728_72835

-- Define the conditions
def ArunCurrentAgeAfter6Years (A: ℕ) : Prop := A + 6 = 36
def DeepakCurrentAge : ℕ := 42

-- Define the goal statement
theorem age_ratio (A: ℕ) (hc: ArunCurrentAgeAfter6Years A) : A / gcd A DeepakCurrentAge = 5 ∧ DeepakCurrentAge / gcd A DeepakCurrentAge = 7 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l728_72835


namespace NUMINAMATH_GPT_helga_extra_hours_last_friday_l728_72869

theorem helga_extra_hours_last_friday
  (weekly_articles : ℕ)
  (extra_hours_thursday : ℕ)
  (extra_articles_thursday : ℕ)
  (extra_articles_friday : ℕ)
  (articles_per_half_hour : ℕ)
  (half_hours_per_hour : ℕ)
  (usual_articles_per_day : ℕ)
  (days_per_week : ℕ)
  (articles_last_thursday_plus_friday : ℕ)
  (total_articles : ℕ) :
  (weekly_articles = (usual_articles_per_day * days_per_week)) →
  (extra_hours_thursday = 2) →
  (articles_per_half_hour = 5) →
  (half_hours_per_hour = 2) →
  (usual_articles_per_day = (articles_per_half_hour * 8)) →
  (extra_articles_thursday = (articles_per_half_hour * (extra_hours_thursday * half_hours_per_hour))) →
  (articles_last_thursday_plus_friday = weekly_articles + extra_articles_thursday) →
  (total_articles = 250) →
  (extra_articles_friday = total_articles - articles_last_thursday_plus_friday) →
  (extra_articles_friday = 30) →
  ((extra_articles_friday / articles_per_half_hour) = 6) →
  (3 = (6 / half_hours_per_hour)) :=
by
  intro hw1 hw2 hw3 hw4 hw5 hw6 hw7 hw8 hw9 hw10
  sorry

end NUMINAMATH_GPT_helga_extra_hours_last_friday_l728_72869


namespace NUMINAMATH_GPT_mac_total_loss_is_correct_l728_72865

def day_1_value : ℝ := 6 * 0.075 + 2 * 0.0075
def day_2_value : ℝ := 10 * 0.0045 + 5 * 0.0036
def day_3_value : ℝ := 4 * 0.10 + 1 * 0.011
def day_4_value : ℝ := 7 * 0.013 + 5 * 0.038
def day_5_value : ℝ := 3 * 0.5 + 2 * 0.0019
def day_6_value : ℝ := 12 * 0.0072 + 3 * 0.0013
def day_7_value : ℝ := 8 * 0.045 + 6 * 0.0089

def total_value : ℝ := day_1_value + day_2_value + day_3_value + day_4_value + day_5_value + day_6_value + day_7_value

def daily_loss (total_value: ℝ): ℝ := total_value - 0.25

def total_loss : ℝ := daily_loss day_1_value + daily_loss day_2_value + daily_loss day_3_value + daily_loss day_4_value + daily_loss day_5_value + daily_loss day_6_value + daily_loss day_7_value

theorem mac_total_loss_is_correct : total_loss = 2.1619 := 
by 
  simp [day_1_value, day_2_value, day_3_value, day_4_value, day_5_value, day_6_value, day_7_value, daily_loss, total_loss]
  sorry

end NUMINAMATH_GPT_mac_total_loss_is_correct_l728_72865


namespace NUMINAMATH_GPT_circle_line_distance_l728_72841

theorem circle_line_distance (c : ℝ) : 
  (∃ (P₁ P₂ P₃ : ℝ × ℝ), 
     (P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₁ ≠ P₃) ∧
     ((P₁.1 - 2)^2 + (P₁.2 - 2)^2 = 18) ∧
     ((P₂.1 - 2)^2 + (P₂.2 - 2)^2 = 18) ∧
     ((P₃.1 - 2)^2 + (P₃.2 - 2)^2 = 18) ∧
     (abs (P₁.1 - P₁.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₂.1 - P₂.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₃.1 - P₃.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2)) ↔ 
  -2 ≤ c ∧ c ≤ 2 :=
sorry

end NUMINAMATH_GPT_circle_line_distance_l728_72841


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l728_72866

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l728_72866


namespace NUMINAMATH_GPT_days_worked_per_week_l728_72889

theorem days_worked_per_week (toys_per_week toys_per_day : ℕ) (h1 : toys_per_week = 5500) (h2 : toys_per_day = 1375) : toys_per_week / toys_per_day = 4 := by
  sorry

end NUMINAMATH_GPT_days_worked_per_week_l728_72889


namespace NUMINAMATH_GPT_max_value_is_zero_l728_72821

noncomputable def max_value (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : ℝ :=
  x^2 - y^2

theorem max_value_is_zero (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : max_value x y h = 0 :=
sorry

end NUMINAMATH_GPT_max_value_is_zero_l728_72821


namespace NUMINAMATH_GPT_sqrt_sum_simplification_l728_72827

theorem sqrt_sum_simplification :
  (Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by
    sorry

end NUMINAMATH_GPT_sqrt_sum_simplification_l728_72827


namespace NUMINAMATH_GPT_cost_of_one_bag_of_onions_l728_72886

theorem cost_of_one_bag_of_onions (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) (h_price : price_per_onion = 200) (h_onions : total_onions = 180) (h_bags : num_bags = 6) :
  (total_onions / num_bags) * price_per_onion = 6000 := 
  by
  sorry

end NUMINAMATH_GPT_cost_of_one_bag_of_onions_l728_72886


namespace NUMINAMATH_GPT_sixth_graders_l728_72815

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end NUMINAMATH_GPT_sixth_graders_l728_72815


namespace NUMINAMATH_GPT_reaction_rate_reduction_l728_72894

theorem reaction_rate_reduction (k : ℝ) (NH3 Br2 NH3_new : ℝ) (v1 v2 : ℝ):
  (v1 = k * NH3^8 * Br2) →
  (v2 = k * NH3_new^8 * Br2) →
  (v2 / v1 = 60) →
  NH3_new = 60 ^ (1 / 8) :=
by
  intro hv1 hv2 hratio
  sorry

end NUMINAMATH_GPT_reaction_rate_reduction_l728_72894


namespace NUMINAMATH_GPT_sum_of_roots_l728_72810

theorem sum_of_roots (x : ℝ) (h : x + 49 / x = 14) : x + x = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l728_72810


namespace NUMINAMATH_GPT_balls_in_boxes_l728_72861

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l728_72861


namespace NUMINAMATH_GPT_students_problem_count_l728_72829

theorem students_problem_count 
  (x y z q r : ℕ) 
  (H1 : x + y + z + q + r = 30) 
  (H2 : x + 2 * y + 3 * z + 4 * q + 5 * r = 40) 
  (h_y_pos : 1 ≤ y) 
  (h_z_pos : 1 ≤ z) 
  (h_q_pos : 1 ≤ q) 
  (h_r_pos : 1 ≤ r) : 
  x = 26 := 
  sorry

end NUMINAMATH_GPT_students_problem_count_l728_72829


namespace NUMINAMATH_GPT_finalStoresAtEndOf2020_l728_72874

def initialStores : ℕ := 23
def storesOpened2019 : ℕ := 5
def storesClosed2019 : ℕ := 2
def storesOpened2020 : ℕ := 10
def storesClosed2020 : ℕ := 6

theorem finalStoresAtEndOf2020 : initialStores + (storesOpened2019 - storesClosed2019) + (storesOpened2020 - storesClosed2020) = 30 :=
by
  sorry

end NUMINAMATH_GPT_finalStoresAtEndOf2020_l728_72874


namespace NUMINAMATH_GPT_worker_time_proof_l728_72895

theorem worker_time_proof (x : ℝ) (h1 : x > 2) (h2 : (100 / (x - 2) - 100 / x) = 5 / 2) : 
  (x = 10) ∧ (x - 2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_worker_time_proof_l728_72895


namespace NUMINAMATH_GPT_average_salary_l728_72803

theorem average_salary (a b c d e : ℕ) (h1 : a = 8000) (h2 : b = 5000) (h3 : c = 16000) (h4 : d = 7000) (h5 : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_l728_72803


namespace NUMINAMATH_GPT_padic_zeros_l728_72885

variable {p : ℕ} (hp : p > 1)
variable {a : ℕ} (hnz : a % p ≠ 0)

theorem padic_zeros (k : ℕ) (hk : k ≥ 1) :
  (a^(p^(k-1)*(p-1)) - 1) % (p^k) = 0 :=
sorry

end NUMINAMATH_GPT_padic_zeros_l728_72885


namespace NUMINAMATH_GPT_total_payment_correct_l728_72871

def payment_y : ℝ := 318.1818181818182
def payment_ratio : ℝ := 1.2
def payment_x : ℝ := payment_ratio * payment_y
def total_payment : ℝ := payment_x + payment_y

theorem total_payment_correct :
  total_payment = 700.00 :=
sorry

end NUMINAMATH_GPT_total_payment_correct_l728_72871


namespace NUMINAMATH_GPT_percentage_saved_l728_72846

-- Define the actual and saved amount.
def actual_investment : ℕ := 150000
def saved_amount : ℕ := 50000

-- Define the planned investment based on the conditions.
def planned_investment : ℕ := actual_investment + saved_amount

-- Proof goal: The percentage saved is 25%.
theorem percentage_saved : (saved_amount * 100) / planned_investment = 25 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_saved_l728_72846


namespace NUMINAMATH_GPT_product_of_two_numbers_l728_72891

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 + y^2 = 200) : x * y = 28 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l728_72891


namespace NUMINAMATH_GPT_center_of_circle_l728_72816

theorem center_of_circle : ∃ c : ℝ × ℝ, (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ ((x - c.1)^2 + (y + c.2)^2 = 2))) ∧ (c = (1, -2)) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_center_of_circle_l728_72816


namespace NUMINAMATH_GPT_eval_expression_l728_72860

theorem eval_expression : (49^2 - 25^2 + 10^2) = 1876 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l728_72860


namespace NUMINAMATH_GPT_john_adds_and_subtracts_l728_72858

theorem john_adds_and_subtracts :
  (41^2 = 40^2 + 81) ∧ (39^2 = 40^2 - 79) :=
by {
  sorry
}

end NUMINAMATH_GPT_john_adds_and_subtracts_l728_72858


namespace NUMINAMATH_GPT_jill_speed_is_8_l728_72876

-- Definitions for conditions
def speed_jack1 := 12 -- speed in km/h for the first 12 km
def distance_jack1 := 12 -- distance in km for the first 12 km

def speed_jack2 := 6 -- speed in km/h for the second 12 km
def distance_jack2 := 12 -- distance in km for the second 12 km

def distance_jill := distance_jack1 + distance_jack2 -- total distance in km for Jill

-- Total time taken by Jack
def time_jack := (distance_jack1 / speed_jack1) + (distance_jack2 / speed_jack2)

-- Jill's speed calculation
def jill_speed := distance_jill / time_jack

-- Theorem stating Jill's speed is 8 km/h
theorem jill_speed_is_8 : jill_speed = 8 := by
  sorry

end NUMINAMATH_GPT_jill_speed_is_8_l728_72876


namespace NUMINAMATH_GPT_stock_price_calculation_l728_72884

def stock_price_end_of_first_year (initial_price : ℝ) (increase_percent : ℝ) : ℝ :=
  initial_price * (1 + increase_percent)

def stock_price_end_of_second_year (price_first_year : ℝ) (decrease_percent : ℝ) : ℝ :=
  price_first_year * (1 - decrease_percent)

theorem stock_price_calculation 
  (initial_price : ℝ)
  (increase_percent : ℝ)
  (decrease_percent : ℝ)
  (final_price : ℝ) :
  initial_price = 120 ∧ 
  increase_percent = 0.80 ∧
  decrease_percent = 0.30 ∧
  final_price = 151.20 → 
  stock_price_end_of_second_year (stock_price_end_of_first_year initial_price increase_percent) decrease_percent = final_price :=
by
  sorry

end NUMINAMATH_GPT_stock_price_calculation_l728_72884


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l728_72809

def A : Set ℝ := { x | x^2 - 5 * x - 6 ≤ 0 }

def B : Set ℝ := { x | x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -1 ≤ x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l728_72809


namespace NUMINAMATH_GPT_find_parabola_focus_l728_72813

theorem find_parabola_focus : 
  ∀ (x y : ℝ), (y = 2 * x ^ 2 + 4 * x - 1) → (∃ p q : ℝ, p = -1 ∧ q = -(23:ℝ) / 8 ∧ (y = 2 * x ^ 2 + 4 * x - 1) → (x, y) = (p, q)) :=
by
  sorry

end NUMINAMATH_GPT_find_parabola_focus_l728_72813


namespace NUMINAMATH_GPT_linear_regression_equation_l728_72847

theorem linear_regression_equation (x y : ℝ) (h : {(1, 2), (2, 3), (3, 4), (4, 5)} ⊆ {(x, y) | y = x + 1}) : 
  (∀ x y, (x = 1 → y = 2) ∧ (x = 2 → y = 3) ∧ (x = 3 → y = 4) ∧ (x = 4 → y = 5)) ↔ (y = x + 1) :=
by
  sorry

end NUMINAMATH_GPT_linear_regression_equation_l728_72847


namespace NUMINAMATH_GPT_average_minutes_run_per_day_l728_72802

-- Define the given averages for each grade
def sixth_grade_avg : ℕ := 10
def seventh_grade_avg : ℕ := 18
def eighth_grade_avg : ℕ := 12

-- Define the ratios of the number of students in each grade
def num_sixth_eq_three_times_num_seventh (num_seventh : ℕ) : ℕ := 3 * num_seventh
def num_eighth_eq_half_num_seventh (num_seventh : ℕ) : ℕ := num_seventh / 2

-- Average number of minutes run per day by all students
theorem average_minutes_run_per_day (num_seventh : ℕ) :
  (sixth_grade_avg * num_sixth_eq_three_times_num_seventh num_seventh +
   seventh_grade_avg * num_seventh +
   eighth_grade_avg * num_eighth_eq_half_num_seventh num_seventh) / 
  (num_sixth_eq_three_times_num_seventh num_seventh + 
   num_seventh + 
   num_eighth_eq_half_num_seventh num_seventh) = 12 := 
sorry

end NUMINAMATH_GPT_average_minutes_run_per_day_l728_72802


namespace NUMINAMATH_GPT_sum_of_a_and_b_l728_72849

noncomputable def log_function (a b x : ℝ) : ℝ := Real.log (x + b) / Real.log a

theorem sum_of_a_and_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log_function a b 2 = 1)
                      (h4 : ∃ x : ℝ, log_function a b x = 8 ∧ log_function a b x = 2) :
  a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l728_72849


namespace NUMINAMATH_GPT_number_of_observations_l728_72851

theorem number_of_observations (n : ℕ) (h1 : 200 - 6 = 194) (h2 : 200 * n - n * 6 = n * 194) :
  n > 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_observations_l728_72851


namespace NUMINAMATH_GPT_triangle_ratio_l728_72831

theorem triangle_ratio (a b c : ℝ) (P Q : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c)
  (h₄ : P > 0) (h₅ : Q > P) (h₆ : Q < c) (h₇ : P = 21) (h₈ : Q - P = 35) (h₉ : c - Q = 100)
  (h₁₀ : P + (Q - P) + (c - Q) = c)
  (angle_trisect : ∃ x y : ℝ, x ≠ y ∧ x = a / b ∧ y = 7 / 45) :
  ∃ p q r : ℕ, p + q + r = 92 ∧ p.gcd r = 1 ∧ ¬ ∃ k : ℕ, k^2 ∣ q := sorry

end NUMINAMATH_GPT_triangle_ratio_l728_72831


namespace NUMINAMATH_GPT_find_a_find_distance_l728_72811

-- Problem 1: Given conditions to find 'a'
theorem find_a (a : ℝ) :
  (∃ θ ρ, ρ = 2 * Real.cos θ ∧ 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0) →
  (a = 2 ∨ a = -8) :=
sorry

-- Problem 2: Given point and line, find the distance
theorem find_distance : 
  ∃ (d : ℝ), d = Real.sqrt 3 + 5/2 ∧
  (∃ θ ρ, θ = 11 * Real.pi / 6 ∧ ρ = 2 ∧ 
   (ρ = Real.sqrt (3 * (Real.sin θ - Real.pi / 6)^2 + (ρ * Real.cos (θ - Real.pi / 6))^2) 
   → ρ * Real.sin (θ - Real.pi / 6) = 1)) :=
sorry

end NUMINAMATH_GPT_find_a_find_distance_l728_72811


namespace NUMINAMATH_GPT_negate_proposition_l728_72838

open Classical

variable (x : ℝ)

theorem negate_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l728_72838


namespace NUMINAMATH_GPT_factorization_a_minus_b_l728_72896

theorem factorization_a_minus_b (a b : ℤ) (h1 : 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : a - b = -7 :=
by
  sorry

end NUMINAMATH_GPT_factorization_a_minus_b_l728_72896


namespace NUMINAMATH_GPT_sum_of_angles_l728_72804

theorem sum_of_angles : 
    ∀ (angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC : ℝ),
    angle1 + angle3 + angle5 = 180 ∧
    angle2 + angle4 + angle6 = 180 ∧
    angleA + angleB + angleC = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 :=
by
  intro angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_angles_l728_72804


namespace NUMINAMATH_GPT_greatest_multiple_of_5_and_6_less_than_1000_l728_72873

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_5_and_6_less_than_1000_l728_72873


namespace NUMINAMATH_GPT_hyperbola_equation_l728_72800

noncomputable def hyperbola_eqn : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (b = (1/2) * a) ∧ (a^2 + b^2 = 25) ∧ 
    (∀ x y, (x^2 / (a^2)) - (y^2 / (b^2)) = 1 ↔ (x^2 / 20) - (y^2 / 5) = 1)

theorem hyperbola_equation : hyperbola_eqn := 
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l728_72800


namespace NUMINAMATH_GPT_carly_shipping_cost_l728_72826

noncomputable def total_shipping_cost (flat_fee cost_per_pound weight : ℝ) : ℝ :=
flat_fee + cost_per_pound * weight

theorem carly_shipping_cost : 
  total_shipping_cost 5 0.80 5 = 9 :=
by 
  unfold total_shipping_cost
  norm_num

end NUMINAMATH_GPT_carly_shipping_cost_l728_72826


namespace NUMINAMATH_GPT_yellow_flower_count_l728_72872

-- Define the number of flowers of each color and total flowers based on given conditions
def total_flowers : Nat := 96
def green_flowers : Nat := 9
def red_flowers : Nat := 3 * green_flowers
def blue_flowers : Nat := total_flowers / 2

-- Define the number of yellow flowers
def yellow_flowers : Nat := total_flowers - (green_flowers + red_flowers + blue_flowers)

-- The theorem we aim to prove
theorem yellow_flower_count : yellow_flowers = 12 := by
  sorry

end NUMINAMATH_GPT_yellow_flower_count_l728_72872


namespace NUMINAMATH_GPT_identity_eq_coefficients_l728_72863

theorem identity_eq_coefficients (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) ↔ (a = c ∧ b = d) :=
by
  sorry

end NUMINAMATH_GPT_identity_eq_coefficients_l728_72863


namespace NUMINAMATH_GPT_vector_sum_correct_l728_72822

def vec1 : Fin 3 → ℤ := ![-7, 3, 5]
def vec2 : Fin 3 → ℤ := ![4, -1, -6]
def vec3 : Fin 3 → ℤ := ![1, 8, 2]
def expectedSum : Fin 3 → ℤ := ![-2, 10, 1]

theorem vector_sum_correct :
  (fun i => vec1 i + vec2 i + vec3 i) = expectedSum := 
by
  sorry

end NUMINAMATH_GPT_vector_sum_correct_l728_72822


namespace NUMINAMATH_GPT_find_common_ratio_l728_72834

-- Defining the conditions in Lean
variables (a : ℕ → ℝ) (d q : ℝ)

-- The arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) - a n = d

-- The geometric sequence condition
def is_geometric_sequence (a1 a2 a4 q : ℝ) : Prop :=
a2 ^ 2 = a1 * a4

-- Proving the main theorem
theorem find_common_ratio (a : ℕ → ℝ) (d q : ℝ) (h_arith : is_arithmetic_sequence a d) (d_ne_zero : d ≠ 0) 
(h_geom : is_geometric_sequence (a 1) (a 2) (a 4) q) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l728_72834


namespace NUMINAMATH_GPT_alcohol_solution_mixing_l728_72812

theorem alcohol_solution_mixing :
  ∀ (V_i C_i C_f C_a x : ℝ),
    V_i = 6 →
    C_i = 0.40 →
    C_f = 0.50 →
    C_a = 0.90 →
    x = 1.5 →
  0.50 * (V_i + x) = (C_i * V_i) + C_a * x →
  C_f * (V_i + x) = (C_i * V_i) + (C_a * x) := 
by
  intros V_i C_i C_f C_a x Vi_eq Ci_eq Cf_eq Ca_eq x_eq h
  sorry

end NUMINAMATH_GPT_alcohol_solution_mixing_l728_72812


namespace NUMINAMATH_GPT_find_difference_l728_72892

noncomputable def expression (x y : ℝ) : ℝ :=
  (|x + y| / (|x| + |y|))^2

theorem find_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  let m := 0
  let M := 1
  M - m = 1 :=
by
  -- Please note that the proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_GPT_find_difference_l728_72892


namespace NUMINAMATH_GPT_simplify_absolute_value_l728_72890

theorem simplify_absolute_value : abs (-(5^2) + 6 * 2) = 13 := by
  sorry

end NUMINAMATH_GPT_simplify_absolute_value_l728_72890


namespace NUMINAMATH_GPT_function_decomposition_l728_72877

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (a : ℝ) (f₁ f₂ : ℝ → ℝ), a > 0 ∧ (∀ x, f₁ x = f₁ (-x)) ∧ (∀ x, f₂ x = f₂ (2 * a - x)) ∧ (∀ x, f x = f₁ x + f₂ x) :=
sorry

end NUMINAMATH_GPT_function_decomposition_l728_72877


namespace NUMINAMATH_GPT_total_drawing_sheets_l728_72820

-- Definitions based on the conditions given
def brown_sheets := 28
def yellow_sheets := 27

-- The statement we need to prove
theorem total_drawing_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end NUMINAMATH_GPT_total_drawing_sheets_l728_72820


namespace NUMINAMATH_GPT_copy_pages_l728_72830

theorem copy_pages
  (total_cents : ℕ)
  (cost_per_page : ℚ)
  (h_total : total_cents = 2000)
  (h_cost : cost_per_page = 2.5) :
  (total_cents / cost_per_page) = 800 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_copy_pages_l728_72830


namespace NUMINAMATH_GPT_boxes_per_case_l728_72856

-- Define the conditions
def total_boxes : ℕ := 54
def total_cases : ℕ := 9

-- Define the result we want to prove
theorem boxes_per_case : total_boxes / total_cases = 6 := 
by sorry

end NUMINAMATH_GPT_boxes_per_case_l728_72856


namespace NUMINAMATH_GPT_no_consecutive_beeches_probability_l728_72883

theorem no_consecutive_beeches_probability :
  let total_trees := 12
  let oaks := 3
  let holm_oaks := 4
  let beeches := 5
  let total_arrangements := (Nat.factorial total_trees) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks) * (Nat.factorial beeches))
  let favorable_arrangements :=
    let slots := oaks + holm_oaks + 1
    Nat.choose slots beeches * ((Nat.factorial (oaks + holm_oaks)) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks)))
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by
  sorry

end NUMINAMATH_GPT_no_consecutive_beeches_probability_l728_72883


namespace NUMINAMATH_GPT_cos_neg_3pi_plus_alpha_l728_72832

/-- Given conditions: 
  1. 𝚌𝚘𝚜(3π/2 + α) = -3/5,
  2. α is an angle in the fourth quadrant,
Prove: cos(-3π + α) = -4/5 -/
theorem cos_neg_3pi_plus_alpha (α : Real) (h1 : Real.cos (3 * Real.pi / 2 + α) = -3 / 5) (h2 : 0 ≤ α ∧ α < 2 * Real.pi ∧ Real.sin α < 0) :
  Real.cos (-3 * Real.pi + α) = -4 / 5 := 
sorry

end NUMINAMATH_GPT_cos_neg_3pi_plus_alpha_l728_72832
