import Mathlib

namespace NUMINAMATH_GPT_remainder_of_division_l1403_140382

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 - 20 * X^2 + 45 * X + 23
noncomputable def d : Polynomial ℤ := (X - 3)^2

theorem remainder_of_division :
  ∃ q r : Polynomial ℤ, p = q * d + r ∧ degree r < degree d ∧ r = 6 * X + 41 := sorry

end NUMINAMATH_GPT_remainder_of_division_l1403_140382


namespace NUMINAMATH_GPT_max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l1403_140384

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end NUMINAMATH_GPT_max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l1403_140384


namespace NUMINAMATH_GPT_area_rectangle_relation_l1403_140393

theorem area_rectangle_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end NUMINAMATH_GPT_area_rectangle_relation_l1403_140393


namespace NUMINAMATH_GPT_find_a_l1403_140346

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1403_140346


namespace NUMINAMATH_GPT_weight_of_new_person_l1403_140335

-- Definitions for the conditions given.

-- Average weight increase
def avg_weight_increase : ℝ := 2.5

-- Number of persons
def num_persons : ℕ := 8

-- Weight of the person being replaced
def weight_replaced : ℝ := 65

-- Total weight increase
def total_weight_increase : ℝ := num_persons * avg_weight_increase

-- Statement to prove the weight of the new person
theorem weight_of_new_person : 
  ∃ (W_new : ℝ), W_new = weight_replaced + total_weight_increase :=
sorry

end NUMINAMATH_GPT_weight_of_new_person_l1403_140335


namespace NUMINAMATH_GPT_largest_inscribed_triangle_area_l1403_140364

theorem largest_inscribed_triangle_area
  (D : Type) 
  (radius : ℝ) 
  (r_eq : radius = 8) 
  (triangle_area : ℝ)
  (max_area : triangle_area = 64) :
  ∃ (base height : ℝ), (base = 2 * radius) ∧ (height = radius) ∧ (triangle_area = (1 / 2) * base * height) := 
by
  sorry

end NUMINAMATH_GPT_largest_inscribed_triangle_area_l1403_140364


namespace NUMINAMATH_GPT_find_ab_l1403_140308

theorem find_ab (a b : ℕ) (h1 : 1 <= a) (h2 : a < 10) (h3 : 0 <= b) (h4 : b < 10) (h5 : 66 * ((1 : ℝ) + ((10 * a + b : ℕ) / 100) - (↑(10 * a + b) / 99)) = 0.5) : 10 * a + b = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l1403_140308


namespace NUMINAMATH_GPT_determine_transportation_mode_l1403_140396

def distance : ℝ := 60 -- in kilometers
def time : ℝ := 3 -- in hours
def speed_of_walking : ℝ := 5 -- typical speed in km/h
def speed_of_bicycle_riding : ℝ := 15 -- lower bound of bicycle speed in km/h
def speed_of_driving_a_car : ℝ := 20 -- typical minimum speed in km/h

theorem determine_transportation_mode : (distance / time) = speed_of_driving_a_car ∧ speed_of_driving_a_car ≥ speed_of_walking + speed_of_bicycle_riding - speed_of_driving_a_car := sorry

end NUMINAMATH_GPT_determine_transportation_mode_l1403_140396


namespace NUMINAMATH_GPT_x_y_square_sum_l1403_140352

theorem x_y_square_sum (x y : ℝ) (h1 : x - y = -1) (h2 : x * y = 1 / 2) : x^2 + y^2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_x_y_square_sum_l1403_140352


namespace NUMINAMATH_GPT_opposite_numbers_expression_l1403_140389

theorem opposite_numbers_expression (a b : ℤ) (h : a + b = 0) : 3 * a + 3 * b - 2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_numbers_expression_l1403_140389


namespace NUMINAMATH_GPT_common_ratio_of_geometric_seq_l1403_140300

variable {a : ℕ → ℚ} -- The sequence
variable {d : ℚ} -- Common difference

-- Assuming the arithmetic and geometric sequence properties
def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (a1 a4 a5 : ℚ) (q : ℚ) : Prop :=
  a4 = a1 * q ∧ a5 = a4 * q

theorem common_ratio_of_geometric_seq (h_arith: is_arithmetic_seq a d) (h_nonzero_d : d ≠ 0)
  (h_geometric: is_geometric_seq (a 1) (a 4) (a 5) (1 / 3)) : (a 4 / a 1) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_seq_l1403_140300


namespace NUMINAMATH_GPT_smaller_cube_edge_length_l1403_140341

theorem smaller_cube_edge_length (x : ℝ) 
    (original_edge_length : ℝ := 7)
    (increase_percentage : ℝ := 600) 
    (original_surface_area_formula : ℝ := 6 * original_edge_length^2)
    (new_surface_area_formula : ℝ := (1 + increase_percentage / 100) * original_surface_area_formula) :
  ∃ x : ℝ, 6 * x^2 * (original_edge_length ^ 3 / x ^ 3) = new_surface_area_formula → x = 1 := by
  sorry

end NUMINAMATH_GPT_smaller_cube_edge_length_l1403_140341


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1403_140348

theorem solution_set_of_inequality (a : ℝ) (h1 : 2 * a - 3 < 0) (h2 : 1 - a < 0) : 1 < a ∧ a < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1403_140348


namespace NUMINAMATH_GPT_vacation_cost_split_l1403_140320

theorem vacation_cost_split (t d : ℕ) 
  (h_total : 105 + 125 + 175 = 405)
  (h_split : 405 / 3 = 135)
  (h_t : t = 135 - 105)
  (h_d : d = 135 - 125) : 
  t - d = 20 := by
  sorry

end NUMINAMATH_GPT_vacation_cost_split_l1403_140320


namespace NUMINAMATH_GPT_shaded_area_size_l1403_140374

noncomputable def total_shaded_area : ℝ :=
  let R := 9
  let r := R / 2
  let area_larger_circle := 81 * Real.pi
  let shaded_area_larger_circle := area_larger_circle / 2
  let area_smaller_circle := Real.pi * r^2
  let shaded_area_smaller_circle := area_smaller_circle / 2
  let total_shaded_area := shaded_area_larger_circle + shaded_area_smaller_circle
  total_shaded_area

theorem shaded_area_size:
  total_shaded_area = 50.625 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_size_l1403_140374


namespace NUMINAMATH_GPT_prime_difference_fourth_powers_is_not_prime_l1403_140358

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_difference_fourth_powers_is_not_prime (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) : 
  ¬ is_prime (p^4 - q^4) :=
sorry

end NUMINAMATH_GPT_prime_difference_fourth_powers_is_not_prime_l1403_140358


namespace NUMINAMATH_GPT_janice_total_cost_is_correct_l1403_140311

def cost_of_items (cost_juices : ℕ) (juices : ℕ) (cost_sandwiches : ℕ) (sandwiches : ℕ) (cost_pastries : ℕ) (pastries : ℕ) (cost_salad : ℕ) (discount_salad : ℕ) : ℕ :=
  let one_sandwich := cost_sandwiches / sandwiches
  let one_juice := cost_juices / juices
  let total_pastries := pastries * cost_pastries
  let discounted_salad := cost_salad - (cost_salad * discount_salad / 100)
  one_sandwich + one_juice + total_pastries + discounted_salad

-- Conditions
def cost_juices := 10
def juices := 5
def cost_sandwiches := 6
def sandwiches := 2
def cost_pastries := 4
def pastries := 2
def cost_salad := 8
def discount_salad := 20

-- Expected Total Cost
def expected_total_cost := 1940 -- in cents to avoid float numbers

theorem janice_total_cost_is_correct : 
  cost_of_items cost_juices juices cost_sandwiches sandwiches cost_pastries pastries cost_salad discount_salad = expected_total_cost :=
by
  simp [cost_of_items, cost_juices, juices, cost_sandwiches, sandwiches, cost_pastries, pastries, cost_salad, discount_salad]
  norm_num
  sorry

end NUMINAMATH_GPT_janice_total_cost_is_correct_l1403_140311


namespace NUMINAMATH_GPT_total_people_in_school_l1403_140386

def number_of_girls := 315
def number_of_boys := 309
def number_of_teachers := 772
def total_number_of_people := number_of_girls + number_of_boys + number_of_teachers

theorem total_people_in_school :
  total_number_of_people = 1396 :=
by sorry

end NUMINAMATH_GPT_total_people_in_school_l1403_140386


namespace NUMINAMATH_GPT_problem_solution_l1403_140353

theorem problem_solution (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1403_140353


namespace NUMINAMATH_GPT_min_colors_needed_l1403_140355

theorem min_colors_needed (n : ℕ) : 
  (n + (n * (n - 1)) / 2 ≥ 12) → (n = 5) :=
by
  sorry

end NUMINAMATH_GPT_min_colors_needed_l1403_140355


namespace NUMINAMATH_GPT_total_heartbeats_during_race_l1403_140394

-- Definitions for conditions
def heart_rate_per_minute : ℕ := 120
def pace_minutes_per_km : ℕ := 4
def race_distance_km : ℕ := 120

-- Lean statement of the proof problem
theorem total_heartbeats_during_race :
  120 * (4 * 120) = 57600 := by
  sorry

end NUMINAMATH_GPT_total_heartbeats_during_race_l1403_140394


namespace NUMINAMATH_GPT_max_height_of_ball_l1403_140372

noncomputable def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

theorem max_height_of_ball : ∃ t : ℝ, (h t) = 69.5 :=
sorry

end NUMINAMATH_GPT_max_height_of_ball_l1403_140372


namespace NUMINAMATH_GPT_average_hamburgers_sold_per_day_l1403_140344

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_hamburgers_sold_per_day_l1403_140344


namespace NUMINAMATH_GPT_sin_cos_of_tan_is_two_l1403_140375

theorem sin_cos_of_tan_is_two (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 :=
sorry

end NUMINAMATH_GPT_sin_cos_of_tan_is_two_l1403_140375


namespace NUMINAMATH_GPT_cats_new_total_weight_l1403_140312

noncomputable def total_weight (weights : List ℚ) : ℚ :=
  weights.sum

noncomputable def remove_min_max_weight (weights : List ℚ) : ℚ :=
  let min_weight := weights.minimum?.getD 0
  let max_weight := weights.maximum?.getD 0
  weights.sum - min_weight - max_weight

theorem cats_new_total_weight :
  let weights := [3.5, 7.2, 4.8, 6, 5.5, 9, 4]
  remove_min_max_weight weights = 27.5 := by
  sorry

end NUMINAMATH_GPT_cats_new_total_weight_l1403_140312


namespace NUMINAMATH_GPT_distance_between_closest_points_of_circles_l1403_140326

theorem distance_between_closest_points_of_circles :
  let circle1_center : ℝ × ℝ := (3, 3)
  let circle2_center : ℝ × ℝ := (20, 15)
  let circle1_radius : ℝ := 3
  let circle2_radius : ℝ := 15
  let distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (15 - 3)^2)
  distance_between_centers - (circle1_radius + circle2_radius) = 2.81 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_between_closest_points_of_circles_l1403_140326


namespace NUMINAMATH_GPT_find_integer_l1403_140356

theorem find_integer (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.cos (n * Real.pi / 180) = Real.sin (312 * Real.pi / 180)) :
  n = 42 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_l1403_140356


namespace NUMINAMATH_GPT_moles_of_NaHCO3_combined_l1403_140373

theorem moles_of_NaHCO3_combined (n_HNO3 n_NaHCO3 : ℕ) (mass_H2O : ℝ) : 
  n_HNO3 = 2 ∧ mass_H2O = 36 ∧ n_HNO3 = n_NaHCO3 → n_NaHCO3 = 2 := by
  sorry

end NUMINAMATH_GPT_moles_of_NaHCO3_combined_l1403_140373


namespace NUMINAMATH_GPT_B_pow_5_eq_rB_plus_sI_l1403_140342

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end NUMINAMATH_GPT_B_pow_5_eq_rB_plus_sI_l1403_140342


namespace NUMINAMATH_GPT_frames_per_page_l1403_140398

theorem frames_per_page (total_frames : ℕ) (pages : ℕ) (frames : ℕ) 
  (h1 : total_frames = 143) 
  (h2 : pages = 13) 
  (h3 : frames = total_frames / pages) : 
  frames = 11 := 
by 
  sorry

end NUMINAMATH_GPT_frames_per_page_l1403_140398


namespace NUMINAMATH_GPT_tangent_line_slope_l1403_140343

theorem tangent_line_slope (k : ℝ) :
  (∃ m : ℝ, (m^3 - m^2 + m = k * m) ∧ (k = 3 * m^2 - 2 * m + 1)) →
  (k = 1 ∨ k = 3 / 4) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tangent_line_slope_l1403_140343


namespace NUMINAMATH_GPT_number_in_tens_place_is_7_l1403_140334

theorem number_in_tens_place_is_7
  (digits : Finset ℕ)
  (a b c : ℕ)
  (h1 : digits = {7, 5, 2})
  (h2 : 100 * a + 10 * b + c > 530)
  (h3 : 100 * a + 10 * b + c < 710)
  (h4 : a ∈ digits)
  (h5 : b ∈ digits)
  (h6 : c ∈ digits)
  (h7 : ∀ x ∈ digits, x ≠ a → x ≠ b → x ≠ c) :
  b = 7 := sorry

end NUMINAMATH_GPT_number_in_tens_place_is_7_l1403_140334


namespace NUMINAMATH_GPT_simplify_fraction_l1403_140328

theorem simplify_fraction (a : ℕ) (h : a = 3) : (10 * a ^ 3) / (55 * a ^ 2) = 6 / 11 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1403_140328


namespace NUMINAMATH_GPT_calculate_expr_at_3_l1403_140366

-- Definition of the expression
def expr (x : ℕ) : ℕ := (x + x * x^(x^2)) * 3

-- The proof statement
theorem calculate_expr_at_3 : expr 3 = 177156 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expr_at_3_l1403_140366


namespace NUMINAMATH_GPT_magician_assistant_trick_l1403_140302

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end NUMINAMATH_GPT_magician_assistant_trick_l1403_140302


namespace NUMINAMATH_GPT_total_trees_cut_down_l1403_140361

-- Definitions based on conditions in the problem
def trees_per_day_james : ℕ := 20
def days_with_just_james : ℕ := 2
def total_trees_by_james := trees_per_day_james * days_with_just_james

def brothers : ℕ := 2
def days_with_brothers : ℕ := 3
def trees_per_day_brothers := (20 * (100 - 20)) / 100 -- 20% fewer than James
def trees_per_day_total := brothers * trees_per_day_brothers + trees_per_day_james

def total_trees_with_brothers := trees_per_day_total * days_with_brothers

-- The statement to be proved
theorem total_trees_cut_down : total_trees_by_james + total_trees_with_brothers = 136 := by
  sorry

end NUMINAMATH_GPT_total_trees_cut_down_l1403_140361


namespace NUMINAMATH_GPT_tom_ratio_is_three_fourths_l1403_140357

-- Define the years for the different programs
def bs_years : ℕ := 3
def phd_years : ℕ := 5
def tom_years : ℕ := 6
def normal_years : ℕ := bs_years + phd_years

-- Define the ratio of Tom's time to the normal time
def ratio : ℚ := tom_years / normal_years

theorem tom_ratio_is_three_fourths :
  ratio = 3 / 4 :=
by
  unfold ratio normal_years bs_years phd_years tom_years
  -- continued proof steps would go here
  sorry

end NUMINAMATH_GPT_tom_ratio_is_three_fourths_l1403_140357


namespace NUMINAMATH_GPT_chandler_bike_purchase_weeks_l1403_140385

theorem chandler_bike_purchase_weeks (bike_cost birthday_money weekly_earnings total_weeks : ℕ) 
  (h_bike_cost : bike_cost = 600)
  (h_birthday_money : birthday_money = 60 + 40 + 20 + 30)
  (h_weekly_earnings : weekly_earnings = 18)
  (h_total_weeks : total_weeks = 25) :
  birthday_money + weekly_earnings * total_weeks = bike_cost :=
by {
  sorry
}

end NUMINAMATH_GPT_chandler_bike_purchase_weeks_l1403_140385


namespace NUMINAMATH_GPT_alison_money_l1403_140336

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end NUMINAMATH_GPT_alison_money_l1403_140336


namespace NUMINAMATH_GPT_brad_has_9_green_balloons_l1403_140332

theorem brad_has_9_green_balloons (total_balloons red_balloons : ℕ) (h_total : total_balloons = 17) (h_red : red_balloons = 8) : total_balloons - red_balloons = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_brad_has_9_green_balloons_l1403_140332


namespace NUMINAMATH_GPT_pond_sustain_capacity_l1403_140309

-- Defining the initial number of frogs
def initial_frogs : ℕ := 5

-- Defining the number of tadpoles
def number_of_tadpoles (frogs: ℕ) : ℕ := 3 * frogs

-- Defining the number of matured tadpoles (those that survive to become frogs)
def matured_tadpoles (tadpoles: ℕ) : ℕ := (2 * tadpoles) / 3

-- Defining the total number of frogs after tadpoles mature
def total_frogs_after_mature (initial_frogs: ℕ) (matured_tadpoles: ℕ) : ℕ :=
  initial_frogs + matured_tadpoles

-- Defining the number of frogs that need to find a new pond
def frogs_to_leave : ℕ := 7

-- Defining the number of frogs the pond can sustain
def frogs_pond_can_sustain (total_frogs: ℕ) (frogs_to_leave: ℕ) : ℕ :=
  total_frogs - frogs_to_leave

-- The main theorem stating the number of frogs the pond can sustain given the conditions
theorem pond_sustain_capacity : frogs_pond_can_sustain
  (total_frogs_after_mature initial_frogs (matured_tadpoles (number_of_tadpoles initial_frogs)))
  frogs_to_leave = 8 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_pond_sustain_capacity_l1403_140309


namespace NUMINAMATH_GPT_smallest_positive_integer_l1403_140395

theorem smallest_positive_integer (x : ℕ) : 
  (5 * x ≡ 18 [MOD 33]) ∧ (x ≡ 4 [MOD 7]) → x = 10 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1403_140395


namespace NUMINAMATH_GPT_find_f_3_l1403_140319

def f (x : ℝ) : ℝ := x^2 + 4 * x + 8

theorem find_f_3 : f 3 = 29 := by
  sorry

end NUMINAMATH_GPT_find_f_3_l1403_140319


namespace NUMINAMATH_GPT_initial_average_age_l1403_140383

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 17) (h2 : n * A + 32 = (n + 1) * 15) : A = 14 := by
  sorry

end NUMINAMATH_GPT_initial_average_age_l1403_140383


namespace NUMINAMATH_GPT_log_a_interval_l1403_140368

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_a_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  {a | log_a a 3 - log_a a 1 = 2} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
by
  sorry

end NUMINAMATH_GPT_log_a_interval_l1403_140368


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_226_l1403_140314

theorem sum_of_squares_of_roots_eq_226 (s_1 s_2 : ℝ) (h_eq : ∀ x, x^2 - 16 * x + 15 = 0 → (x = s_1 ∨ x = s_2)) :
  s_1^2 + s_2^2 = 226 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_226_l1403_140314


namespace NUMINAMATH_GPT_correct_statements_truth_of_statements_l1403_140331

-- Define basic properties related to factor and divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Given conditions as definitions
def condition_A : Prop := is_factor 4 100
def condition_B1 : Prop := is_divisor 19 133
def condition_B2 : Prop := ¬ is_divisor 19 51
def condition_C1 : Prop := is_divisor 30 90
def condition_C2 : Prop := ¬ is_divisor 30 53
def condition_D1 : Prop := is_divisor 7 21
def condition_D2 : Prop := ¬ is_divisor 7 49
def condition_E : Prop := is_factor 10 200

-- Statement that needs to be proved
theorem correct_statements : 
  (condition_A ∧ 
  (condition_B1 ∧ condition_B2) ∧ 
  condition_E) :=
by sorry -- proof to be inserted

-- Equivalent Lean 4 statement with all conditions encapsulated
theorem truth_of_statements :
  (is_factor 4 100) ∧ 
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧ 
  is_factor 10 200 :=
by sorry -- proof to be inserted

end NUMINAMATH_GPT_correct_statements_truth_of_statements_l1403_140331


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l1403_140392

theorem sufficient_but_not_necessary (m x y a : ℝ) (h₀ : m > 0) (h₁ : |x - a| < m) (h₂ : |y - a| < m) : |x - y| < 2 * m :=
by
  sorry

theorem not_necessary (m : ℝ) (h₀ : m > 0) : ∃ x y a : ℝ, |x - y| < 2 * m ∧ ¬ (|x - a| < m ∧ |y - a| < m) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l1403_140392


namespace NUMINAMATH_GPT_product_of_two_integers_l1403_140391

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 22) (h2 : x^2 - y^2 = 44) : x * y = 120 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_integers_l1403_140391


namespace NUMINAMATH_GPT_range_of_a_l1403_140338

noncomputable def A (x : ℝ) : Prop := (3 * x) / (x + 1) ≤ 2
noncomputable def B (x a : ℝ) : Prop := a - 2 < x ∧ x < 2 * a + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, A x ↔ B x a) ↔ (1 / 2 < a ∧ a ≤ 1) := by
sorry

end NUMINAMATH_GPT_range_of_a_l1403_140338


namespace NUMINAMATH_GPT_find_x_l1403_140305

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : 
  x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1403_140305


namespace NUMINAMATH_GPT_seeds_per_bed_l1403_140323

theorem seeds_per_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 60) (h2 : flower_beds = 6) : total_seeds / flower_beds = 10 := by
  sorry

end NUMINAMATH_GPT_seeds_per_bed_l1403_140323


namespace NUMINAMATH_GPT_largest_number_using_digits_l1403_140313

theorem largest_number_using_digits (d1 d2 d3 : ℕ) (h1 : d1 = 7) (h2 : d2 = 1) (h3 : d3 = 0) : 
  ∃ n : ℕ, (n = 710) ∧ (∀ m : ℕ, (m = d1 * 100 + d2 * 10 + d3) ∨ (m = d1 * 100 + d3 * 10 + d2) ∨ (m = d2 * 100 + d1 * 10 + d3) ∨ 
  (m = d2 * 100 + d3 * 10 + d1) ∨ (m = d3 * 100 + d1 * 10 + d2) ∨ (m = d3 * 100 + d2 * 10 + d1) → n ≥ m) := 
by
  sorry

end NUMINAMATH_GPT_largest_number_using_digits_l1403_140313


namespace NUMINAMATH_GPT_abs_g_eq_abs_gx_l1403_140360

noncomputable def g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= 0 then x^2 - 2 else -x + 2

noncomputable def abs_g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= -Real.sqrt 2 then x^2 - 2
else if -Real.sqrt 2 < x ∧ x <= Real.sqrt 2 then 2 - x^2
else if Real.sqrt 2 < x ∧ x <= 2 then 2 - x
else x - 2

theorem abs_g_eq_abs_gx (x : ℝ) (hx1 : -3 <= x ∧ x <= -Real.sqrt 2) 
  (hx2 : -Real.sqrt 2 < x ∧ x <= Real.sqrt 2)
  (hx3 : Real.sqrt 2 < x ∧ x <= 2)
  (hx4 : 2 < x ∧ x <= 3) :
  abs_g x = |g x| :=
by
  sorry

end NUMINAMATH_GPT_abs_g_eq_abs_gx_l1403_140360


namespace NUMINAMATH_GPT_find_y_when_z_is_three_l1403_140306

theorem find_y_when_z_is_three
  (k : ℝ) (y z : ℝ)
  (h1 : y = 3)
  (h2 : z = 1)
  (h3 : y ^ 4 * z ^ 2 = k)
  (hc : z = 3) :
  y ^ 4 = 9 :=
sorry

end NUMINAMATH_GPT_find_y_when_z_is_three_l1403_140306


namespace NUMINAMATH_GPT_relay_team_order_count_l1403_140362

def num_ways_to_order_relay (total_members : Nat) (jordan_lap : Nat) : Nat :=
  if jordan_lap = total_members then (total_members - 1).factorial else 0

theorem relay_team_order_count : num_ways_to_order_relay 5 5 = 24 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_relay_team_order_count_l1403_140362


namespace NUMINAMATH_GPT_sets_relation_l1403_140317

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def M : Set ℚ := {x | ∃ (m : ℤ), x = m + 1/6}
def S : Set ℚ := {x | ∃ (s : ℤ), x = s/2 - 1/3}
def P : Set ℚ := {x | ∃ (p : ℤ), x = p/2 + 1/6}

theorem sets_relation : M ⊆ S ∧ S = P := by
  sorry

end NUMINAMATH_GPT_sets_relation_l1403_140317


namespace NUMINAMATH_GPT_horatio_sonnets_l1403_140399

theorem horatio_sonnets (num_lines_per_sonnet : ℕ) (heard_sonnets : ℕ) (unheard_lines : ℕ) (h1 : num_lines_per_sonnet = 16) (h2 : heard_sonnets = 9) (h3 : unheard_lines = 126) :
  ∃ total_sonnets : ℕ, total_sonnets = 16 :=
by
  -- Note: The proof is not required, hence 'sorry' is included to skip it.
  sorry

end NUMINAMATH_GPT_horatio_sonnets_l1403_140399


namespace NUMINAMATH_GPT_shaded_grid_percentage_l1403_140369

theorem shaded_grid_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 64) (h2 : shaded_squares = 48) : 
  ((shaded_squares : ℚ) / (total_squares : ℚ)) * 100 = 75 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_shaded_grid_percentage_l1403_140369


namespace NUMINAMATH_GPT_find_q_l1403_140321

-- Given polynomial Q(x) with coefficients p, q, d
variables {p q d : ℝ}

-- Define the polynomial Q(x)
def Q (x : ℝ) := x^3 + p * x^2 + q * x + d

-- Assume the conditions of the problem
theorem find_q (h1 : d = 5)                   -- y-intercept is 5
    (h2 : (-p / 3) = -d)                    -- mean of zeros = product of zeros
    (h3 : (-p / 3) = 1 + p + q + d)          -- mean of zeros = sum of coefficients
    : q = -26 := 
    sorry

end NUMINAMATH_GPT_find_q_l1403_140321


namespace NUMINAMATH_GPT_alcohol_concentration_l1403_140354

theorem alcohol_concentration 
  (x : ℝ) -- concentration of alcohol in the first vessel (as a percentage)
  (h1 : 0 ≤ x ∧ x ≤ 100) -- percentage is between 0 and 100
  (h2 : (x / 100) * 2 + (55 / 100) * 6 = (37 / 100) * 10) -- given condition for concentration balance
  : x = 20 :=
sorry

end NUMINAMATH_GPT_alcohol_concentration_l1403_140354


namespace NUMINAMATH_GPT_movie_theater_ticket_sales_l1403_140376

theorem movie_theater_ticket_sales 
  (A C : ℤ) 
  (h1 : A + C = 900) 
  (h2 : 7 * A + 4 * C = 5100) : 
  A = 500 := 
sorry

end NUMINAMATH_GPT_movie_theater_ticket_sales_l1403_140376


namespace NUMINAMATH_GPT_right_triangle_equation_l1403_140337

-- Let a, b, and c be the sides of a right triangle with a^2 + b^2 = c^2
variables (a b c : ℕ)
-- Define the semiperimeter
def semiperimeter (a b c : ℕ) : ℕ := (a + b + c) / 2
-- Define the radius of the inscribed circle
def inscribed_radius (a b c : ℕ) : ℚ := (a * b) / (2 * semiperimeter a b c)
-- State the theorem to prove
theorem right_triangle_equation : 
    ∀ a b c : ℕ, a^2 + b^2 = c^2 → semiperimeter a b c + inscribed_radius a b c = a + b := by
  sorry

end NUMINAMATH_GPT_right_triangle_equation_l1403_140337


namespace NUMINAMATH_GPT_range_of_a_l1403_140324

theorem range_of_a (a : ℝ) : 
  (∀ x, (x ≤ 1 ∨ x ≥ 3) ↔ ((a ≤ x ∧ x ≤ a + 1) → (x ≤ 1 ∨ x ≥ 3))) → 
  (a ≤ 0 ∨ a ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1403_140324


namespace NUMINAMATH_GPT_swimming_class_attendance_l1403_140303

def total_students : ℕ := 1000
def chess_ratio : ℝ := 0.25
def swimming_ratio : ℝ := 0.50

def chess_students := chess_ratio * total_students
def swimming_students := swimming_ratio * chess_students

theorem swimming_class_attendance :
  swimming_students = 125 :=
by
  sorry

end NUMINAMATH_GPT_swimming_class_attendance_l1403_140303


namespace NUMINAMATH_GPT_solve_diamond_l1403_140377

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_diamond_l1403_140377


namespace NUMINAMATH_GPT_roots_condition_l1403_140327

theorem roots_condition (r1 r2 p : ℝ) (h_eq : ∀ x : ℝ, x^2 + p * x + 12 = 0 → (x = r1 ∨ x = r2))
(h_distinct : r1 ≠ r2) (h_vieta1 : r1 + r2 = -p) (h_vieta2 : r1 * r2 = 12) : 
|r1| > 3 ∨ |r2| > 3 :=
by
  sorry

end NUMINAMATH_GPT_roots_condition_l1403_140327


namespace NUMINAMATH_GPT_question1_question2_l1403_140365

noncomputable def minimum_value (x y : ℝ) : ℝ := (1 / x) + (1 / y)

theorem question1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) : 
  minimum_value x y = 2 :=
sorry

theorem question2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
  (x + 1) * (y + 1) ≠ 5 :=
sorry

end NUMINAMATH_GPT_question1_question2_l1403_140365


namespace NUMINAMATH_GPT_find_acid_percentage_l1403_140330

theorem find_acid_percentage (P : ℕ) (x : ℕ) (h1 : 4 + x = 20) 
  (h2 : x = 20 - 4) 
  (h3 : (P : ℝ)/100 * 4 + 0.75 * 16 = 0.72 * 20) : P = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_acid_percentage_l1403_140330


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1403_140379

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - abs x - 2 < 0) ↔ (-2 < x ∧ x < 2) := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1403_140379


namespace NUMINAMATH_GPT_abs_opposite_of_three_eq_5_l1403_140397

theorem abs_opposite_of_three_eq_5 : ∀ (a : ℤ), a = -3 → |a - 2| = 5 := by
  sorry

end NUMINAMATH_GPT_abs_opposite_of_three_eq_5_l1403_140397


namespace NUMINAMATH_GPT_sam_total_yellow_marbles_l1403_140340

def sam_original_yellow_marbles : Float := 86.0
def sam_yellow_marbles_given_by_joan : Float := 25.0

theorem sam_total_yellow_marbles : sam_original_yellow_marbles + sam_yellow_marbles_given_by_joan = 111.0 := by
  sorry

end NUMINAMATH_GPT_sam_total_yellow_marbles_l1403_140340


namespace NUMINAMATH_GPT_complex_value_of_z_six_plus_z_inv_six_l1403_140388

open Complex

theorem complex_value_of_z_six_plus_z_inv_six (z : ℂ) (h : z + z⁻¹ = 1) : z^6 + (z⁻¹)^6 = 2 := by
  sorry

end NUMINAMATH_GPT_complex_value_of_z_six_plus_z_inv_six_l1403_140388


namespace NUMINAMATH_GPT_smallest_n_cond_l1403_140381

theorem smallest_n_cond (n : ℕ) (h1 : n >= 100 ∧ n < 1000) (h2 : n ≡ 3 [MOD 9]) (h3 : n ≡ 3 [MOD 4]) : n = 111 := 
sorry

end NUMINAMATH_GPT_smallest_n_cond_l1403_140381


namespace NUMINAMATH_GPT_compare_abc_l1403_140322

noncomputable def a : ℝ := 2^(1/2)
noncomputable def b : ℝ := 3^(1/3)
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > a ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l1403_140322


namespace NUMINAMATH_GPT_average_of_new_sequence_l1403_140378

variable (c : ℕ)  -- c is a positive integer
variable (d : ℕ)  -- d is the average of the sequence starting from c 

def average_of_sequence (seq : List ℕ) : ℕ :=
  if h : seq.length ≠ 0 then seq.sum / seq.length else 0

theorem average_of_new_sequence (h : d = average_of_sequence [c, c+1, c+2, c+3, c+4, c+5, c+6]) :
  average_of_sequence [d, d+1, d+2, d+3, d+4, d+5, d+6] = c + 6 := 
sorry

end NUMINAMATH_GPT_average_of_new_sequence_l1403_140378


namespace NUMINAMATH_GPT_collinear_vectors_l1403_140329

open Vector

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (a b : V) : Prop :=
¬(∃ k : ℝ, k ≠ 0 ∧ a = k • b)

theorem collinear_vectors
  {a b m n : V}
  (h1 : m = a + b)
  (h2 : n = 2 • a + 2 • b)
  (h3 : not_collinear a b) :
  ∃ k : ℝ, k ≠ 0 ∧ n = k • m :=
by
  sorry

end NUMINAMATH_GPT_collinear_vectors_l1403_140329


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l1403_140301

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.4) (t : ℝ)
  (h2 : (v + c) * t = (v - c) * (2 * t)) : 
  v = 4.2 :=
by
  sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l1403_140301


namespace NUMINAMATH_GPT_maximum_x1_x2_x3_l1403_140359

theorem maximum_x1_x2_x3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
  x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
  x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
  x1 + x2 + x3 ≤ 61 := 
by sorry

end NUMINAMATH_GPT_maximum_x1_x2_x3_l1403_140359


namespace NUMINAMATH_GPT_radius_of_hole_l1403_140380

-- Define the dimensions of the rectangular solid
def length1 : ℕ := 3
def length2 : ℕ := 8
def length3 : ℕ := 9

-- Define the radius of the hole
variable (r : ℕ)

-- Condition: The area of the 2 circles removed equals the lateral surface area of the cylinder
axiom area_condition : 2 * Real.pi * r^2 = 2 * Real.pi * r * length1

-- Prove that the radius of the cylindrical hole is 3
theorem radius_of_hole : r = 3 := by
  sorry

end NUMINAMATH_GPT_radius_of_hole_l1403_140380


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1403_140347

open Real

noncomputable def x : ℝ := 2 - sqrt 3

theorem algebraic_expression_evaluation :
  (7 + 4 * sqrt 3) * x^2 - (2 + sqrt 3) * x + sqrt 3 = 2 + sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1403_140347


namespace NUMINAMATH_GPT_work_together_l1403_140304

theorem work_together (A_rate B_rate : ℝ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) : (1 / (A_rate + B_rate) = 6) :=
by
  -- we only need to write the statement, proof is not required.
  sorry

end NUMINAMATH_GPT_work_together_l1403_140304


namespace NUMINAMATH_GPT_problem_inequality_l1403_140345

theorem problem_inequality (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ ab + 3*b + 2*c := 
by 
  sorry

end NUMINAMATH_GPT_problem_inequality_l1403_140345


namespace NUMINAMATH_GPT_cube_mono_l1403_140371

theorem cube_mono {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_mono_l1403_140371


namespace NUMINAMATH_GPT_g_f_neg5_l1403_140387

-- Define the function f
def f (x : ℝ) := 2 * x ^ 2 - 4

-- Define the function g with the known condition g(f(5)) = 12
axiom g : ℝ → ℝ
axiom g_f5 : g (f 5) = 12

-- Now state the main theorem we need to prove
theorem g_f_neg5 : g (f (-5)) = 12 := by
  sorry

end NUMINAMATH_GPT_g_f_neg5_l1403_140387


namespace NUMINAMATH_GPT_rings_sold_l1403_140351

theorem rings_sold (R : ℕ) : 
  ∀ (num_necklaces total_sales necklace_price ring_price : ℕ),
  num_necklaces = 4 →
  total_sales = 80 →
  necklace_price = 12 →
  ring_price = 4 →
  num_necklaces * necklace_price + R * ring_price = total_sales →
  R = 8 := 
by 
  intros num_necklaces total_sales necklace_price ring_price h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_rings_sold_l1403_140351


namespace NUMINAMATH_GPT_handshake_count_l1403_140307

theorem handshake_count (n : ℕ) (m : ℕ) (couples : ℕ) (people : ℕ) 
  (h1 : couples = 15) 
  (h2 : people = 2 * couples)
  (h3 : people = 30)
  (h4 : n = couples) 
  (h5 : m = people / 2)
  (h6 : ∀ i : ℕ, i < m → ∀ j : ℕ, j < m → i ≠ j → i * j + i ≠ n 
    ∧ j * i + j ≠ n) 
  : n * (n - 1) / 2 + (2 * n - 2) * n = 315 :=
by
  sorry

end NUMINAMATH_GPT_handshake_count_l1403_140307


namespace NUMINAMATH_GPT_find_b_l1403_140325

theorem find_b (a : ℝ) (A : ℝ) (B : ℝ) (b : ℝ)
  (ha : a = 5) 
  (hA : A = Real.pi / 6) 
  (htanB : Real.tan B = 3 / 4)
  (hsinB : Real.sin B = 3 / 5):
  b = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l1403_140325


namespace NUMINAMATH_GPT_range_of_a_l1403_140318

variable (a : ℝ)
def A := Set.Ico (-2 : ℝ) 4
def B := {x : ℝ | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (h : B a ⊆ A) : 0 ≤ a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1403_140318


namespace NUMINAMATH_GPT_cheese_cookie_packs_l1403_140310

def packs_per_box (P : ℕ) : Prop :=
  let cartons := 12
  let boxes_per_carton := 12
  let total_boxes := cartons * boxes_per_carton
  let total_cost := 1440
  let box_cost := total_cost / total_boxes
  let pack_cost := 1
  P = box_cost / pack_cost

theorem cheese_cookie_packs : packs_per_box 10 := by
  sorry

end NUMINAMATH_GPT_cheese_cookie_packs_l1403_140310


namespace NUMINAMATH_GPT_maximum_ab_minimum_frac_minimum_exp_l1403_140349

variable {a b : ℝ}

theorem maximum_ab (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  ab <= 1/8 :=
sorry

theorem minimum_frac (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2/a + 1/b >= 8 :=
sorry

theorem minimum_exp (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2^a + 4^b >= 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_maximum_ab_minimum_frac_minimum_exp_l1403_140349


namespace NUMINAMATH_GPT_sector_area_l1403_140367

theorem sector_area (r α : ℝ) (h_r : r = 3) (h_α : α = 2) : (1/2 * r^2 * α) = 9 := by
  sorry

end NUMINAMATH_GPT_sector_area_l1403_140367


namespace NUMINAMATH_GPT_correct_exponentiation_rule_l1403_140390

theorem correct_exponentiation_rule (x y : ℝ) : ((x^2)^3 = x^6) :=
  by sorry

end NUMINAMATH_GPT_correct_exponentiation_rule_l1403_140390


namespace NUMINAMATH_GPT_proof_op_nabla_l1403_140339

def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem proof_op_nabla :
  op_nabla (op_nabla (1/2) (1/3)) (1/4) = 9 / 11 := by
  sorry

end NUMINAMATH_GPT_proof_op_nabla_l1403_140339


namespace NUMINAMATH_GPT_total_employees_l1403_140370

variable (E : ℕ) -- E is the total number of employees

-- Conditions given in the problem
variable (male_fraction : ℚ := 0.45) -- 45% of the total employees are males
variable (males_below_50 : ℕ := 1170) -- 1170 males are below 50 years old
variable (males_total : ℕ := 2340) -- Total number of male employees

-- Condition derived from the problem (calculation of total males)
lemma male_employees_equiv (h : males_total = 2 * males_below_50) : males_total = 2340 :=
  by sorry

-- Main theorem
theorem total_employees (h : male_fraction * E = males_total) : E = 5200 :=
  by sorry

end NUMINAMATH_GPT_total_employees_l1403_140370


namespace NUMINAMATH_GPT_sean_div_julie_l1403_140363

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end NUMINAMATH_GPT_sean_div_julie_l1403_140363


namespace NUMINAMATH_GPT_correct_transformation_l1403_140350

variable (a b : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : b ≠ 0)
variable (h₂ : a / 2 = b / 3)

theorem correct_transformation : 3 / b = 2 / a :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1403_140350


namespace NUMINAMATH_GPT_speed_of_man_l1403_140315

theorem speed_of_man 
  (L : ℝ) 
  (V_t : ℝ) 
  (T : ℝ) 
  (conversion_factor : ℝ) 
  (kmph_to_mps : ℝ → ℝ)
  (final_conversion : ℝ → ℝ) 
  (relative_speed : ℝ) 
  (Vm : ℝ) : Prop := 
L = 220 ∧ V_t = 59 ∧ T = 12 ∧ 
conversion_factor = 1000 / 3600 ∧ 
kmph_to_mps V_t = V_t * conversion_factor ∧ 
relative_speed = L / T ∧ 
Vm = relative_speed - (kmph_to_mps V_t) ∧ 
final_conversion Vm = Vm * 3.6 ∧ 
final_conversion Vm = 6.984

end NUMINAMATH_GPT_speed_of_man_l1403_140315


namespace NUMINAMATH_GPT_two_thirds_greater_l1403_140316

theorem two_thirds_greater :
  let epsilon : ℚ := (2 : ℚ) / (3 * 10^8)
  let decimal_part : ℚ := 66666666 / 10^8
  (2 / 3) - decimal_part = epsilon := by
  sorry

end NUMINAMATH_GPT_two_thirds_greater_l1403_140316


namespace NUMINAMATH_GPT_probability_of_exactly_one_instrument_l1403_140333

-- Definitions
def total_people : ℕ := 800
def fraction_play_at_least_one_instrument : ℚ := 2 / 5
def num_play_two_or_more_instruments : ℕ := 96

-- Calculation
def num_play_at_least_one_instrument := fraction_play_at_least_one_instrument * total_people
def num_play_exactly_one_instrument := num_play_at_least_one_instrument - num_play_two_or_more_instruments

-- Probability calculation
def probability_play_exactly_one_instrument := num_play_exactly_one_instrument / total_people

-- Proof statement
theorem probability_of_exactly_one_instrument :
  probability_play_exactly_one_instrument = 0.28 := by
  sorry

end NUMINAMATH_GPT_probability_of_exactly_one_instrument_l1403_140333
