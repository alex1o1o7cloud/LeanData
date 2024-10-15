import Mathlib

namespace NUMINAMATH_GPT_sin_double_angle_l2152_215230

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
  sorry

end NUMINAMATH_GPT_sin_double_angle_l2152_215230


namespace NUMINAMATH_GPT_trackball_mice_count_l2152_215207

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end NUMINAMATH_GPT_trackball_mice_count_l2152_215207


namespace NUMINAMATH_GPT_min_rainfall_on_fourth_day_l2152_215239

theorem min_rainfall_on_fourth_day : 
  let capacity_ft := 6
  let drain_per_day_in := 3
  let rain_first_day_in := 10
  let rain_second_day_in := 2 * rain_first_day_in
  let rain_third_day_in := 1.5 * rain_second_day_in
  let total_rain_first_three_days_in := rain_first_day_in + rain_second_day_in + rain_third_day_in
  let total_drain_in := 3 * drain_per_day_in
  let water_level_start_fourth_day_in := total_rain_first_three_days_in - total_drain_in
  let capacity_in := capacity_ft * 12
  capacity_in = water_level_start_fourth_day_in + 21 :=
by
  sorry

end NUMINAMATH_GPT_min_rainfall_on_fourth_day_l2152_215239


namespace NUMINAMATH_GPT_the_inequality_l2152_215293

theorem the_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a / (1 + b)) + (b / (1 + c)) + (c / (1 + a)) ≥ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_the_inequality_l2152_215293


namespace NUMINAMATH_GPT_find_c_for_same_solution_l2152_215250

theorem find_c_for_same_solution (c : ℝ) (x : ℝ) :
  (3 * x + 5 = 1) ∧ (c * x + 15 = -5) → c = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_c_for_same_solution_l2152_215250


namespace NUMINAMATH_GPT_susan_spent_75_percent_l2152_215271

variables (B b s : ℝ)

-- Conditions
def condition1 : Prop := b = 0.25 * (B - 3 * s)
def condition2 : Prop := s = 0.10 * (B - 2 * b)

-- Theorem
theorem susan_spent_75_percent (h1 : condition1 B b s) (h2 : condition2 B b s) : b + s = 0.75 * B := 
sorry

end NUMINAMATH_GPT_susan_spent_75_percent_l2152_215271


namespace NUMINAMATH_GPT_inscribed_square_area_ratio_l2152_215208

theorem inscribed_square_area_ratio (side_length : ℝ) (h_pos : side_length > 0) :
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  (inscribed_square_area / large_square_area) = (1 / 4) :=
by
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  sorry

end NUMINAMATH_GPT_inscribed_square_area_ratio_l2152_215208


namespace NUMINAMATH_GPT_value_of_x_minus_y_l2152_215262

theorem value_of_x_minus_y (x y : ℝ) (h1 : x = -(-3)) (h2 : |y| = 5) (h3 : x * y < 0) : x - y = 8 := 
sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l2152_215262


namespace NUMINAMATH_GPT_flour_to_add_l2152_215253

-- Define the conditions
def total_flour_required : ℕ := 9
def flour_already_added : ℕ := 2

-- Define the proof statement
theorem flour_to_add : total_flour_required - flour_already_added = 7 := 
by {
    sorry
}

end NUMINAMATH_GPT_flour_to_add_l2152_215253


namespace NUMINAMATH_GPT_geom_seq_not_necessary_sufficient_l2152_215286

theorem geom_seq_not_necessary_sufficient (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ¬(∀ n, a n > a (n + 1) → false) ∨ ¬(∀ n, a (n + 1) > a n) :=
sorry

end NUMINAMATH_GPT_geom_seq_not_necessary_sufficient_l2152_215286


namespace NUMINAMATH_GPT_find_x_l2152_215206

theorem find_x (x : ℝ) (h₀ : ⌊x⌋ * x = 162) : x = 13.5 :=
sorry

end NUMINAMATH_GPT_find_x_l2152_215206


namespace NUMINAMATH_GPT_half_radius_y_l2152_215227

theorem half_radius_y (r_x r_y : ℝ) (hx : 2 * Real.pi * r_x = 12 * Real.pi) (harea : Real.pi * r_x ^ 2 = Real.pi * r_y ^ 2) : r_y / 2 = 3 := by
  sorry

end NUMINAMATH_GPT_half_radius_y_l2152_215227


namespace NUMINAMATH_GPT_four_digit_numbers_count_eq_l2152_215247

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_count_eq_l2152_215247


namespace NUMINAMATH_GPT_problem_x2_minus_y2_l2152_215228

-- Problem statement: Given the conditions, prove x^2 - y^2 = 5 / 1111
theorem problem_x2_minus_y2 (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 101) :
  x^2 - y^2 = 5 / 1111 :=
by
  sorry

end NUMINAMATH_GPT_problem_x2_minus_y2_l2152_215228


namespace NUMINAMATH_GPT_compute_x_over_w_l2152_215202

theorem compute_x_over_w (w x y z : ℚ) (hw : w ≠ 0)
  (h1 : (x + 6 * y - 3 * z) / (-3 * x + 4 * w) = 2 / 3)
  (h2 : (-2 * y + z) / (x - w) = 2 / 3) :
  x / w = 2 / 3 :=
sorry

end NUMINAMATH_GPT_compute_x_over_w_l2152_215202


namespace NUMINAMATH_GPT_solution_to_power_tower_l2152_215297

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solution_to_power_tower : ∃ x : ℝ, infinite_power_tower x = 4 ∧ x = Real.sqrt 2 := sorry

end NUMINAMATH_GPT_solution_to_power_tower_l2152_215297


namespace NUMINAMATH_GPT_negation_of_forall_implies_exists_l2152_215282

theorem negation_of_forall_implies_exists :
  (¬ ∀ x : ℝ, x^2 > 1) = (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_implies_exists_l2152_215282


namespace NUMINAMATH_GPT_max_radius_of_sector_l2152_215214

def sector_perimeter_area (r : ℝ) : ℝ := -r^2 + 10 * r

theorem max_radius_of_sector (R A : ℝ) (h : 2 * R + A = 20) : R = 5 :=
by
  sorry

end NUMINAMATH_GPT_max_radius_of_sector_l2152_215214


namespace NUMINAMATH_GPT_complex_power_identity_l2152_215276

theorem complex_power_identity (w : ℂ) (h : w + w⁻¹ = 2) : w^(2022 : ℕ) + (w⁻¹)^(2022 : ℕ) = 2 := by
  sorry

end NUMINAMATH_GPT_complex_power_identity_l2152_215276


namespace NUMINAMATH_GPT_car_speed_l2152_215212

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end NUMINAMATH_GPT_car_speed_l2152_215212


namespace NUMINAMATH_GPT_even_odd_function_value_l2152_215287

theorem even_odd_function_value 
  (f g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_odd : ∀ x, g (-x) = - g x)
  (h_eqn : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_even_odd_function_value_l2152_215287


namespace NUMINAMATH_GPT_cats_remaining_l2152_215201

theorem cats_remaining 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 := 
by
  sorry

end NUMINAMATH_GPT_cats_remaining_l2152_215201


namespace NUMINAMATH_GPT_fraction_of_students_on_trip_are_girls_l2152_215255

variable (b g : ℕ)
variable (H1 : g = 2 * b) -- twice as many girls as boys
variable (fraction_girls_on_trip : ℚ := 2 / 3)
variable (fraction_boys_on_trip : ℚ := 1 / 2)

def fraction_of_girls_on_trip (b g : ℕ) (H1 : g = 2 * b) (fraction_girls_on_trip : ℚ) (fraction_boys_on_trip : ℚ) :=
  let girls_on_trip := fraction_girls_on_trip * g
  let boys_on_trip := fraction_boys_on_trip * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip

theorem fraction_of_students_on_trip_are_girls (b g : ℕ) (H1 : g = 2 * b) : 
  fraction_of_girls_on_trip b g H1 (2 / 3) (1 / 2) = 8 / 11 := 
by sorry

end NUMINAMATH_GPT_fraction_of_students_on_trip_are_girls_l2152_215255


namespace NUMINAMATH_GPT_integral_eval_l2152_215246

theorem integral_eval : ∫ x in (1:ℝ)..(2:ℝ), (2*x + 1/x) = 3 + Real.log 2 := by
  sorry

end NUMINAMATH_GPT_integral_eval_l2152_215246


namespace NUMINAMATH_GPT_perimeter_triangle_ABC_is_correct_l2152_215244

noncomputable def semicircle_perimeter_trianlge_ABC : ℝ :=
  let BE := (1 : ℝ)
  let EF := (24 : ℝ)
  let FC := (3 : ℝ)
  let BC := BE + EF + FC
  let r := EF / 2
  let x := 71.5
  let AB := x + BE
  let AC := x + FC
  AB + BC + AC

theorem perimeter_triangle_ABC_is_correct : semicircle_perimeter_trianlge_ABC = 175 := by
  sorry

end NUMINAMATH_GPT_perimeter_triangle_ABC_is_correct_l2152_215244


namespace NUMINAMATH_GPT_find_number_of_hens_l2152_215296

def hens_and_cows_problem (H C : ℕ) : Prop :=
  (H + C = 50) ∧ (2 * H + 4 * C = 144)

theorem find_number_of_hens (H C : ℕ) (hc : hens_and_cows_problem H C) : H = 28 :=
by {
  -- We assume the problem conditions and skip the proof using sorry
  sorry
}

end NUMINAMATH_GPT_find_number_of_hens_l2152_215296


namespace NUMINAMATH_GPT_train_speed_l2152_215220

def distance : ℕ := 500
def time : ℕ := 10
def conversion_factor : ℝ := 3.6

theorem train_speed :
  (distance / time : ℝ) * conversion_factor = 180 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l2152_215220


namespace NUMINAMATH_GPT_isabella_hair_length_l2152_215242

theorem isabella_hair_length (h : ℕ) (g : h + 4 = 22) : h = 18 := by
  sorry

end NUMINAMATH_GPT_isabella_hair_length_l2152_215242


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2152_215229

theorem solution_set_of_inequality:
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2152_215229


namespace NUMINAMATH_GPT_cos_C_sin_B_area_l2152_215200

noncomputable def triangle_conditions (A B C a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧
  (b / c = 2 * Real.sqrt 3 / 3) ∧
  (A + 3 * C = Real.pi)

theorem cos_C (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.cos C = Real.sqrt 3 / 3 :=
sorry

theorem sin_B (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.sin B = 2 * Real.sqrt 2 / 3 :=
sorry

theorem area (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) (hb : b = 3 * Real.sqrt 3) :
  (1 / 2) * b * c * Real.sin A = 9 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_cos_C_sin_B_area_l2152_215200


namespace NUMINAMATH_GPT_remainder_of_13_pow_13_plus_13_div_14_l2152_215256

theorem remainder_of_13_pow_13_plus_13_div_14 : ((13 ^ 13 + 13) % 14) = 12 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_13_pow_13_plus_13_div_14_l2152_215256


namespace NUMINAMATH_GPT_first_percentage_increase_l2152_215224

theorem first_percentage_increase (x : ℝ) :
  (1 + x / 100) * 1.4 = 1.82 → x = 30 := 
by 
  intro h
  -- start your proof here
  sorry

end NUMINAMATH_GPT_first_percentage_increase_l2152_215224


namespace NUMINAMATH_GPT_find_k_eq_l2152_215260

theorem find_k_eq (n : ℝ) (k m : ℤ) (h : ∀ n : ℝ, n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2) : k = 3 := 
sorry

end NUMINAMATH_GPT_find_k_eq_l2152_215260


namespace NUMINAMATH_GPT_trip_savings_l2152_215272

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end NUMINAMATH_GPT_trip_savings_l2152_215272


namespace NUMINAMATH_GPT_remainder_of_12_factorial_mod_13_l2152_215257

open Nat

theorem remainder_of_12_factorial_mod_13 : (factorial 12) % 13 = 12 := by
  -- Wilson's Theorem: For a prime number \( p \), \( (p-1)! \equiv -1 \pmod{p} \)
  -- Given \( p = 13 \), we have \( 12! \equiv -1 \pmod{13} \)
  -- Thus, it follows that the remainder is 12
  sorry

end NUMINAMATH_GPT_remainder_of_12_factorial_mod_13_l2152_215257


namespace NUMINAMATH_GPT_assignment_plans_l2152_215261

theorem assignment_plans {students towns : ℕ} (h_students : students = 5) (h_towns : towns = 3) :
  ∃ plans : ℕ, plans = 150 :=
by
  -- Given conditions
  have h1 : students = 5 := h_students
  have h2 : towns = 3 := h_towns
  
  -- The required number of assignment plans
  existsi 150
  -- Proof is not supplied
  sorry

end NUMINAMATH_GPT_assignment_plans_l2152_215261


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l2152_215294

theorem problem_part_1 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  |c| ≤ 1 :=
by
  sorry

theorem problem_part_2 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → |g x| ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l2152_215294


namespace NUMINAMATH_GPT_num_subsets_of_abc_eq_eight_l2152_215245

theorem num_subsets_of_abc_eq_eight : 
  (∃ (s : Finset ℕ), s = {1, 2, 3} ∧ s.powerset.card = 8) :=
sorry

end NUMINAMATH_GPT_num_subsets_of_abc_eq_eight_l2152_215245


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2152_215225

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2152_215225


namespace NUMINAMATH_GPT_two_roses_more_than_three_carnations_l2152_215249

variable {x y : ℝ}

theorem two_roses_more_than_three_carnations
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y := 
by 
  sorry

end NUMINAMATH_GPT_two_roses_more_than_three_carnations_l2152_215249


namespace NUMINAMATH_GPT_total_lives_l2152_215274

theorem total_lives :
  ∀ (num_friends num_new_players lives_per_friend lives_per_new_player : ℕ),
  num_friends = 2 →
  lives_per_friend = 6 →
  num_new_players = 2 →
  lives_per_new_player = 6 →
  (num_friends * lives_per_friend + num_new_players * lives_per_new_player) = 24 :=
by
  intros num_friends num_new_players lives_per_friend lives_per_new_player
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_lives_l2152_215274


namespace NUMINAMATH_GPT_rectangle_width_l2152_215251

/-- Given the conditions:
    - length of a rectangle is 5.4 cm
    - area of the rectangle is 48.6 cm²
    Prove that the width of the rectangle is 9 cm.
-/
theorem rectangle_width (length width area : ℝ) 
  (h_length : length = 5.4) 
  (h_area : area = 48.6) 
  (h_area_eq : area = length * width) : 
  width = 9 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l2152_215251


namespace NUMINAMATH_GPT_correct_quotient_l2152_215265

theorem correct_quotient (D Q : ℕ) (h1 : 21 * Q = 12 * 56) : Q = 32 :=
by {
  -- Proof to be provided
  sorry
}

end NUMINAMATH_GPT_correct_quotient_l2152_215265


namespace NUMINAMATH_GPT_probability_shattering_l2152_215238

theorem probability_shattering (total_cars : ℕ) (shattered_windshields : ℕ) (p : ℚ) 
  (h_total : total_cars = 20000) 
  (h_shattered: shattered_windshields = 600) 
  (h_p : p = shattered_windshields / total_cars) : 
  p = 0.03 := 
by 
  -- skipped proof
  sorry

end NUMINAMATH_GPT_probability_shattering_l2152_215238


namespace NUMINAMATH_GPT_boat_distance_along_stream_l2152_215215

-- Define the conditions
def speed_of_boat_still_water : ℝ := 9
def distance_against_stream_per_hour : ℝ := 7

-- Define the speed of the stream
def speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour

-- Define the speed of the boat along the stream
def speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream

-- Theorem statement
theorem boat_distance_along_stream (speed_of_boat_still_water : ℝ)
                                    (distance_against_stream_per_hour : ℝ)
                                    (effective_speed_against_stream : ℝ := speed_of_boat_still_water - speed_of_stream)
                                    (speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour)
                                    (speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream)
                                    (one_hour : ℝ := 1) :
  speed_of_boat_along_stream = 11 := 
  by
    sorry

end NUMINAMATH_GPT_boat_distance_along_stream_l2152_215215


namespace NUMINAMATH_GPT_prove_m_set_l2152_215269

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}

-- Define set B as dependent on m
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}

-- The main proof statement
theorem prove_m_set : {m : ℝ | B m ∩ A = B m} = {0, 1, 2} :=
by
  -- Code here would prove the above theorem
  sorry

end NUMINAMATH_GPT_prove_m_set_l2152_215269


namespace NUMINAMATH_GPT_probability_of_purple_marble_l2152_215219

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) 
  (h_blue : p_blue = 0.3) 
  (h_green : p_green = 0.4) 
  (h_sum : p_blue + p_green + p_purple = 1) : 
  p_purple = 0.3 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_purple_marble_l2152_215219


namespace NUMINAMATH_GPT_ratio_of_work_speeds_l2152_215210

theorem ratio_of_work_speeds (B_speed : ℚ) (combined_speed : ℚ) (A_speed : ℚ) 
  (h1 : B_speed = 1/12) 
  (h2 : combined_speed = 1/4) 
  (h3 : A_speed + B_speed = combined_speed) : 
  A_speed / B_speed = 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_work_speeds_l2152_215210


namespace NUMINAMATH_GPT_line_tangent_circle_iff_m_l2152_215243

/-- Definition of the circle and the line -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- Prove that the line is tangent to the circle if and only if m = -3 or m = -13 -/
theorem line_tangent_circle_iff_m (m : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y m) ↔ m = -3 ∨ m = -13 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_circle_iff_m_l2152_215243


namespace NUMINAMATH_GPT_sum_of_exponents_l2152_215252

theorem sum_of_exponents (n : ℕ) (h : n = 2^11 + 2^10 + 2^5 + 2^4 + 2^2) : 11 + 10 + 5 + 4 + 2 = 32 :=
by {
  -- The proof could be written here
  sorry
}

end NUMINAMATH_GPT_sum_of_exponents_l2152_215252


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2152_215298

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4)
  (h3 : a = b ∨ 2 * a > b) :
  (a ≠ b ∨ b = 2 * a) → 
  ∃ p : ℝ, p = a + b + b ∧ p = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2152_215298


namespace NUMINAMATH_GPT_expression_value_l2152_215283

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l2152_215283


namespace NUMINAMATH_GPT_profit_percentage_l2152_215237

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 500) (h_selling : selling_price = 750) :
  ((selling_price - cost_price) / cost_price) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l2152_215237


namespace NUMINAMATH_GPT_range_of_a_l2152_215288

theorem range_of_a (a : ℝ) :
  (∃ M : ℝ × ℝ, (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧ (M.1)^2 + (M.2 + 3)^2 = 4 * ((M.1)^2 + (M.2)^2))
  → 0 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2152_215288


namespace NUMINAMATH_GPT_colton_stickers_left_l2152_215211

theorem colton_stickers_left :
  let C := 72
  let F := 4 * 3 -- stickers given to three friends
  let M := F + 2 -- stickers given to Mandy
  let J := M - 10 -- stickers given to Justin
  let T := F + M + J -- total stickers given away
  C - T = 42 := by
  sorry

end NUMINAMATH_GPT_colton_stickers_left_l2152_215211


namespace NUMINAMATH_GPT_men_in_club_l2152_215266

-- Definitions
variables (M W : ℕ) -- Number of men and women

-- Conditions
def club_members := M + W = 30
def event_participation := W / 3 + M = 18

-- Goal
theorem men_in_club : club_members M W → event_participation M W → M = 12 :=
sorry

end NUMINAMATH_GPT_men_in_club_l2152_215266


namespace NUMINAMATH_GPT_sum_first_2500_terms_eq_zero_l2152_215299

theorem sum_first_2500_terms_eq_zero
  (b : ℕ → ℤ)
  (h1 : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2))
  (h2 : (Finset.range 1800).sum b = 2023)
  (h3 : (Finset.range 2023).sum b = 1800) :
  (Finset.range 2500).sum b = 0 :=
sorry

end NUMINAMATH_GPT_sum_first_2500_terms_eq_zero_l2152_215299


namespace NUMINAMATH_GPT_largest_x_63_over_8_l2152_215203

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_x_63_over_8_l2152_215203


namespace NUMINAMATH_GPT_evaluate_expression_l2152_215277

theorem evaluate_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2152_215277


namespace NUMINAMATH_GPT_Liu_Wei_parts_per_day_l2152_215264

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Liu_Wei_parts_per_day_l2152_215264


namespace NUMINAMATH_GPT_vlecks_in_straight_angle_l2152_215273

theorem vlecks_in_straight_angle (V : Type) [LinearOrderedField V] (full_circle_vlecks : V) (h1 : full_circle_vlecks = 600) :
  (full_circle_vlecks / 2) = 300 :=
by
  sorry

end NUMINAMATH_GPT_vlecks_in_straight_angle_l2152_215273


namespace NUMINAMATH_GPT_washing_time_is_45_l2152_215259

-- Definitions based on conditions
variables (x : ℕ) -- time to wash one load
axiom h1 : 2 * x + 75 = 165 -- total laundry time equation

-- The statement to prove: washing one load takes 45 minutes
theorem washing_time_is_45 : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_washing_time_is_45_l2152_215259


namespace NUMINAMATH_GPT_find_four_real_numbers_l2152_215216

theorem find_four_real_numbers (x1 x2 x3 x4 : ℝ) :
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
sorry

end NUMINAMATH_GPT_find_four_real_numbers_l2152_215216


namespace NUMINAMATH_GPT_maximize_volume_l2152_215275

theorem maximize_volume
  (R H A : ℝ) (K : ℝ) (hA : 2 * π * R * H + 2 * π * R * (Real.sqrt (R ^ 2 + H ^ 2)) = A)
  (hK : K = A / (2 * π)) :
  R = (A / (π * Real.sqrt 5)) ^ (1 / 3) :=
sorry

end NUMINAMATH_GPT_maximize_volume_l2152_215275


namespace NUMINAMATH_GPT_integer_solutions_of_quadratic_eq_l2152_215205

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end NUMINAMATH_GPT_integer_solutions_of_quadratic_eq_l2152_215205


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_remaining_polygon_l2152_215240

theorem sum_of_interior_angles_of_remaining_polygon (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 5) :
  (n - 2) * 180 ≠ 270 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_remaining_polygon_l2152_215240


namespace NUMINAMATH_GPT_seashells_given_to_brothers_l2152_215204

theorem seashells_given_to_brothers :
  ∃ B : ℕ, 180 - 40 - B = 2 * 55 ∧ B = 30 := by
  sorry

end NUMINAMATH_GPT_seashells_given_to_brothers_l2152_215204


namespace NUMINAMATH_GPT_max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l2152_215209

/-- Define the given function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

/-- The maximum value of the function f(x) is sqrt(2) -/
theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 := 
sorry

/-- The smallest positive period of the function f(x) -/
theorem smallest_positive_period_of_f :
  ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = Real.pi :=
sorry

/-- The set of values x that satisfy f(x) ≥ 1 -/
theorem values_of_x_satisfying_f_ge_1 :
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l2152_215209


namespace NUMINAMATH_GPT_complex_exchange_of_apartments_in_two_days_l2152_215285

theorem complex_exchange_of_apartments_in_two_days :
  ∀ (n : ℕ) (p : Fin n → Fin n), ∃ (day1 day2 : Fin n → Fin n),
    (∀ x : Fin n, p (day1 x) = day2 x ∨ day1 (p x) = day2 x) ∧
    (∀ x : Fin n, day1 x ≠ x) ∧
    (∀ x : Fin n, day2 x ≠ x) :=
by
  sorry

end NUMINAMATH_GPT_complex_exchange_of_apartments_in_two_days_l2152_215285


namespace NUMINAMATH_GPT_present_age_of_A_l2152_215223

theorem present_age_of_A (A B C : ℕ) 
  (h1 : A + B + C = 57)
  (h2 : B - 3 = 2 * (A - 3))
  (h3 : C - 3 = 3 * (A - 3)) :
  A = 11 :=
sorry

end NUMINAMATH_GPT_present_age_of_A_l2152_215223


namespace NUMINAMATH_GPT_xyz_inequality_l2152_215284

theorem xyz_inequality (x y z : ℝ) (h : x + y + z > 0) : x^3 + y^3 + z^3 > 3 * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l2152_215284


namespace NUMINAMATH_GPT_each_car_has_4_wheels_l2152_215213
-- Import necessary libraries

-- Define the conditions
def number_of_guests := 40
def number_of_parent_cars := 2
def wheels_per_parent_car := 4
def number_of_guest_cars := 10
def total_wheels := 48
def parent_car_wheels := number_of_parent_cars * wheels_per_parent_car
def guest_car_wheels := total_wheels - parent_car_wheels

-- Define the proposition to prove
theorem each_car_has_4_wheels : (guest_car_wheels / number_of_guest_cars) = 4 :=
by
  sorry

end NUMINAMATH_GPT_each_car_has_4_wheels_l2152_215213


namespace NUMINAMATH_GPT_algebraic_expression_solution_l2152_215254

theorem algebraic_expression_solution
  (a b : ℝ)
  (h : -2 * a + 3 * b = 10) :
  9 * b - 6 * a + 2 = 32 :=
by 
  -- We would normally provide the proof here
  sorry

end NUMINAMATH_GPT_algebraic_expression_solution_l2152_215254


namespace NUMINAMATH_GPT_nine_digit_number_l2152_215279

-- Conditions as definitions
def highest_digit (n : ℕ) : Prop :=
  (n / 100000000) = 6

def million_place (n : ℕ) : Prop :=
  (n / 1000000) % 10 = 1

def hundred_place (n : ℕ) : Prop :=
  n % 1000 / 100 = 1

def rest_digits_zero (n : ℕ) : Prop :=
  (n % 1000000 / 1000) % 10 = 0 ∧ 
  (n % 1000000 / 10000) % 10 = 0 ∧ 
  (n % 1000000 / 100000) % 10 = 0 ∧ 
  (n % 100000000 / 10000000) % 10 = 0 ∧ 
  (n % 100000000 / 100000000) % 10 = 0 ∧ 
  (n % 1000000000 / 100000000) % 10 = 6

-- The nine-digit number
def given_number : ℕ := 6001000100

-- Prove number == 60,010,001,00 and approximate to 6 billion
theorem nine_digit_number :
  ∃ n : ℕ, highest_digit n ∧ million_place n ∧ hundred_place n ∧ rest_digits_zero n ∧ n = 6001000100 ∧ (n / 1000000000) = 6 :=
sorry

end NUMINAMATH_GPT_nine_digit_number_l2152_215279


namespace NUMINAMATH_GPT_find_y_given_x_inverse_square_l2152_215278

theorem find_y_given_x_inverse_square (x y : ℚ) : 
  (∀ k, (3 * y = k / x^2) ∧ (3 * 5 = k / 2^2)) → (x = 6) → y = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_y_given_x_inverse_square_l2152_215278


namespace NUMINAMATH_GPT_volume_of_sphere_in_cone_l2152_215232

theorem volume_of_sphere_in_cone :
  let r_base := 9
  let h_cone := 9
  let diameter_sphere := 9 * Real.sqrt 2
  let radius_sphere := diameter_sphere / 2
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere ^ 3
  volume_sphere = (1458 * Real.sqrt 2 / 4) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_in_cone_l2152_215232


namespace NUMINAMATH_GPT_mark_bench_press_correct_l2152_215248

def dave_weight : ℝ := 175
def dave_bench_press : ℝ := 3 * dave_weight

def craig_bench_percentage : ℝ := 0.20
def craig_bench_press : ℝ := craig_bench_percentage * dave_bench_press

def emma_bench_percentage : ℝ := 0.75
def emma_initial_bench_press : ℝ := emma_bench_percentage * dave_bench_press
def emma_actual_bench_press : ℝ := emma_initial_bench_press + 15

def combined_craig_emma : ℝ := craig_bench_press + emma_actual_bench_press

def john_bench_factor : ℝ := 2
def john_bench_press : ℝ := john_bench_factor * combined_craig_emma

def mark_reduction : ℝ := 50
def mark_bench_press : ℝ := combined_craig_emma - mark_reduction

theorem mark_bench_press_correct : mark_bench_press = 463.75 := by
  sorry

end NUMINAMATH_GPT_mark_bench_press_correct_l2152_215248


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2152_215258

theorem sum_of_three_numbers (a b c : ℝ) :
  a + b = 35 → b + c = 47 → c + a = 58 → a + b + c = 70 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2152_215258


namespace NUMINAMATH_GPT_fuel_tank_initial_capacity_l2152_215218

variables (fuel_consumption : ℕ) (journey_distance remaining_fuel initial_fuel : ℕ)

-- Define conditions
def fuel_consumption_rate := 12      -- liters per 100 km
def journey := 275                  -- km
def remaining := 14                 -- liters
def fuel_converted := (fuel_consumption_rate * journey) / 100

-- Define the proposition to be proved
theorem fuel_tank_initial_capacity :
  initial_fuel = fuel_converted + remaining :=
sorry

end NUMINAMATH_GPT_fuel_tank_initial_capacity_l2152_215218


namespace NUMINAMATH_GPT_estimate_y_value_at_x_equals_3_l2152_215270

noncomputable def estimate_y (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 3) * x + a

theorem estimate_y_value_at_x_equals_3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ) (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ),
    (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 2 * (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8)) →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 8 →
    estimate_y 3 (1 / 6) = 7 / 6 := by
  intro x1 x2 x3 x4 x5 x6 x7 x8 y1 y2 y3 y4 y5 y6 y7 y8 h_sum hx
  sorry

end NUMINAMATH_GPT_estimate_y_value_at_x_equals_3_l2152_215270


namespace NUMINAMATH_GPT_largest_common_term_l2152_215233

theorem largest_common_term (a : ℕ) (k l : ℕ) (hk : a = 4 + 5 * k) (hl : a = 5 + 10 * l) (h : a < 300) : a = 299 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_common_term_l2152_215233


namespace NUMINAMATH_GPT_area_of_trapezoid_l2152_215267

-- Define the parameters as given in the problem
def PQ : ℝ := 40
def RS : ℝ := 25
def h : ℝ := 10
def PR : ℝ := 20

-- Assert the quadrilateral is a trapezoid with bases PQ and RS parallel
def isTrapezoid (PQ RS : ℝ) (h : ℝ) (PR : ℝ) : Prop := true -- this is just a placeholder to state that it's a trapezoid

-- The main statement for the area of the trapezoid
theorem area_of_trapezoid (h : ℝ) (PQ RS : ℝ) (h : ℝ) (PR : ℝ) (is_trapezoid : isTrapezoid PQ RS h PR) : (1/2) * (PQ + RS) * h = 325 :=
by
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_l2152_215267


namespace NUMINAMATH_GPT_meaningful_fraction_range_l2152_215290

theorem meaningful_fraction_range (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) := sorry

end NUMINAMATH_GPT_meaningful_fraction_range_l2152_215290


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2152_215292

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - 1) / (x + 2) > 1 ↔ x < -2 ∨ x > 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2152_215292


namespace NUMINAMATH_GPT_min_sum_abc_l2152_215226

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end NUMINAMATH_GPT_min_sum_abc_l2152_215226


namespace NUMINAMATH_GPT_school_total_payment_l2152_215236

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end NUMINAMATH_GPT_school_total_payment_l2152_215236


namespace NUMINAMATH_GPT_complement_U_B_eq_D_l2152_215241

def B (x : ℝ) : Prop := x^2 - 3 * x + 2 < 0
def U : Set ℝ := Set.univ
def complement_U_B : Set ℝ := U \ {x | B x}

theorem complement_U_B_eq_D : complement_U_B = {x | x ≤ 1 ∨ x ≥ 2} := by
  sorry

end NUMINAMATH_GPT_complement_U_B_eq_D_l2152_215241


namespace NUMINAMATH_GPT_drop_in_water_level_l2152_215281

theorem drop_in_water_level (rise_level : ℝ) (drop_level : ℝ) 
  (h : rise_level = 1) : drop_level = -2 :=
by
  sorry

end NUMINAMATH_GPT_drop_in_water_level_l2152_215281


namespace NUMINAMATH_GPT_sequence_is_constant_l2152_215221

theorem sequence_is_constant
  (a : ℕ+ → ℝ)
  (S : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, S n + S (n + 1) = a (n + 1))
  : ∀ n : ℕ+, a n = 0 :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_constant_l2152_215221


namespace NUMINAMATH_GPT_simplest_fraction_is_D_l2152_215234

def fractionA (x : ℕ) : ℚ := 10 / (15 * x)
def fractionB (a b : ℕ) : ℚ := (2 * a * b) / (3 * a * a)
def fractionC (x : ℕ) : ℚ := (x + 1) / (3 * x + 3)
def fractionD (x : ℕ) : ℚ := (x + 1) / (x * x + 1)

theorem simplest_fraction_is_D (x a b : ℕ) :
  ¬ ∃ c, c ≠ 1 ∧
    (fractionA x = (fractionA x / c) ∨
     fractionB a b = (fractionB a b / c) ∨
     fractionC x = (fractionC x / c)) ∧
    ∀ d, d ≠ 1 → fractionD x ≠ (fractionD x / d) := 
  sorry

end NUMINAMATH_GPT_simplest_fraction_is_D_l2152_215234


namespace NUMINAMATH_GPT_equation_of_plane_passing_through_points_l2152_215263

/-
Let M1, M2, and M3 be points in three-dimensional space.
M1 = (1, 2, 0)
M2 = (1, -1, 2)
M3 = (0, 1, -1)
We need to prove that the plane passing through these points has the equation 5x - 2y - 3z - 1 = 0.
-/

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨1, -1, 2⟩
def M3 : Point3D := ⟨0, 1, -1⟩

theorem equation_of_plane_passing_through_points :
  ∃ (a b c d : ℝ), (∀ (P : Point3D), 
  P = M1 ∨ P = M2 ∨ P = M3 → a * P.x + b * P.y + c * P.z + d = 0)
  ∧ a = 5 ∧ b = -2 ∧ c = -3 ∧ d = -1 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_plane_passing_through_points_l2152_215263


namespace NUMINAMATH_GPT_field_division_l2152_215280

theorem field_division
  (total_area : ℕ)
  (part_area : ℕ)
  (diff : ℕ → ℕ)
  (X : ℕ)
  (h_total : total_area = 900)
  (h_part : part_area = 405)
  (h_diff : diff (total_area - part_area - part_area) = (1 / 5 : ℚ) * X)
  : X = 450 := 
sorry

end NUMINAMATH_GPT_field_division_l2152_215280


namespace NUMINAMATH_GPT_sqrt_double_sqrt_four_l2152_215295

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_double_sqrt_four :
  sqrt (sqrt 4) = sqrt 2 ∨ sqrt (sqrt 4) = -sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_double_sqrt_four_l2152_215295


namespace NUMINAMATH_GPT_number_of_beavers_l2152_215289

-- Definitions of the problem conditions
def total_workers : Nat := 862
def number_of_spiders : Nat := 544

-- The statement we need to prove
theorem number_of_beavers : (total_workers - number_of_spiders) = 318 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_beavers_l2152_215289


namespace NUMINAMATH_GPT_triplet_sum_not_zero_l2152_215235

def sum_triplet (a b c : ℝ) : ℝ := a + b + c

theorem triplet_sum_not_zero :
  ¬ (sum_triplet 3 (-5) 2 = 0) ∧
  (sum_triplet (1/4) (1/4) (-1/2) = 0) ∧
  (sum_triplet 0.3 (-0.1) (-0.2) = 0) ∧
  (sum_triplet 0.5 (-0.3) (-0.2) = 0) ∧
  (sum_triplet (1/3) (-1/6) (-1/6) = 0) :=
by 
  sorry

end NUMINAMATH_GPT_triplet_sum_not_zero_l2152_215235


namespace NUMINAMATH_GPT_remainder_of_2_pow_33_mod_9_l2152_215291

theorem remainder_of_2_pow_33_mod_9 : (2 ^ 33) % 9 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_2_pow_33_mod_9_l2152_215291


namespace NUMINAMATH_GPT_value_of_f_g_3_l2152_215231

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x + 2

theorem value_of_f_g_3 : f (g 3) = 83 := by
  sorry

end NUMINAMATH_GPT_value_of_f_g_3_l2152_215231


namespace NUMINAMATH_GPT_original_cube_volume_eq_216_l2152_215222

theorem original_cube_volume_eq_216 (a : ℕ)
  (h1 : ∀ (a : ℕ), ∃ V_orig V_new : ℕ, 
    V_orig = a^3 ∧ 
    V_new = (a + 1) * (a + 1) * (a - 2) ∧ 
    V_orig = V_new + 10) : 
  a = 6 → a^3 = 216 := 
by
  sorry

end NUMINAMATH_GPT_original_cube_volume_eq_216_l2152_215222


namespace NUMINAMATH_GPT_exponent_property_l2152_215268

theorem exponent_property : (-2)^2004 + 3 * (-2)^2003 = -2^2003 :=
by 
  sorry

end NUMINAMATH_GPT_exponent_property_l2152_215268


namespace NUMINAMATH_GPT_sum_of_n_and_k_l2152_215217

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end NUMINAMATH_GPT_sum_of_n_and_k_l2152_215217
