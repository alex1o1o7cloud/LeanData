import Mathlib

namespace NUMINAMATH_GPT_initial_hotdogs_l1041_104139

-- Definitions
variable (x : ℕ)

-- Conditions
def condition : Prop := x - 2 = 97 

-- Statement to prove
theorem initial_hotdogs (h : condition x) : x = 99 :=
  by
    sorry

end NUMINAMATH_GPT_initial_hotdogs_l1041_104139


namespace NUMINAMATH_GPT_combined_weight_of_three_new_people_l1041_104127

theorem combined_weight_of_three_new_people 
  (W : ℝ) 
  (h_avg_increase : (W + 80) / 20 = W / 20 + 4) 
  (h_replaced_weights : 60 + 75 + 85 = 220) : 
  220 + 80 = 300 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_three_new_people_l1041_104127


namespace NUMINAMATH_GPT_find_function_expression_find_range_of_m_l1041_104192

-- Statement for Part 1
theorem find_function_expression (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) : 
  y = -1/2 * x - 2 := 
sorry

-- Statement for Part 2
theorem find_range_of_m (m x : ℝ) (hx : x > -2) (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) :
  (-x + m < -1/2 * x - 2) ↔ (m ≤ -3) := 
sorry

end NUMINAMATH_GPT_find_function_expression_find_range_of_m_l1041_104192


namespace NUMINAMATH_GPT_courtyard_width_is_14_l1041_104102

-- Given conditions
def length_courtyard := 24   -- 24 meters
def num_bricks := 8960       -- Total number of bricks

@[simp]
def brick_length_m : ℝ := 0.25  -- 25 cm in meters
@[simp]
def brick_width_m : ℝ := 0.15   -- 15 cm in meters

-- Correct answer
def width_courtyard : ℝ := 14

-- Prove that the width of the courtyard is 14 meters
theorem courtyard_width_is_14 : 
  (length_courtyard * width_courtyard) = (num_bricks * (brick_length_m * brick_width_m)) :=
by
  -- Lean proof will go here
  sorry

end NUMINAMATH_GPT_courtyard_width_is_14_l1041_104102


namespace NUMINAMATH_GPT_garden_wall_additional_courses_l1041_104119

theorem garden_wall_additional_courses (initial_courses additional_courses : ℕ) (bricks_per_course total_bricks bricks_removed : ℕ) 
  (h1 : bricks_per_course = 400) 
  (h2 : initial_courses = 3) 
  (h3 : bricks_removed = bricks_per_course / 2) 
  (h4 : total_bricks = 1800) 
  (h5 : total_bricks = initial_courses * bricks_per_course + additional_courses * bricks_per_course - bricks_removed) : 
  additional_courses = 2 :=
by
  sorry

end NUMINAMATH_GPT_garden_wall_additional_courses_l1041_104119


namespace NUMINAMATH_GPT_tank_empty_time_l1041_104170

theorem tank_empty_time (R L : ℝ) (h1 : R = 1 / 7) (h2 : R - L = 1 / 8) : 
  (1 / L) = 56 :=
by
  sorry

end NUMINAMATH_GPT_tank_empty_time_l1041_104170


namespace NUMINAMATH_GPT_melony_profit_l1041_104111

theorem melony_profit (profit_3_shirts : ℝ)
  (profit_2_sandals : ℝ)
  (h1 : profit_3_shirts = 21)
  (h2 : profit_2_sandals = 4 * 21) : profit_3_shirts / 3 * 7 + profit_2_sandals / 2 * 3 = 175 := 
by 
  sorry

end NUMINAMATH_GPT_melony_profit_l1041_104111


namespace NUMINAMATH_GPT_colored_pencils_more_than_erasers_l1041_104188

def colored_pencils_initial := 67
def erasers_initial := 38

def colored_pencils_final := 50
def erasers_final := 28

theorem colored_pencils_more_than_erasers :
  colored_pencils_final - erasers_final = 22 := by
  sorry

end NUMINAMATH_GPT_colored_pencils_more_than_erasers_l1041_104188


namespace NUMINAMATH_GPT_soak_time_l1041_104124

/-- 
Bill needs to soak his clothes for 4 minutes to get rid of each grass stain.
His clothes have 3 grass stains and 1 marinara stain.
The total soaking time is 19 minutes.
Prove that the number of minutes needed to soak for each marinara stain is 7.
-/
theorem soak_time (m : ℕ) (grass_stain_time : ℕ) (num_grass_stains : ℕ) (num_marinara_stains : ℕ) (total_time : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_time = 19) :
  m = 7 :=
by sorry

end NUMINAMATH_GPT_soak_time_l1041_104124


namespace NUMINAMATH_GPT_time_between_rings_is_288_minutes_l1041_104123

def intervals_between_rings (total_rings : ℕ) (total_minutes : ℕ) : ℕ := 
  let intervals := total_rings - 1
  total_minutes / intervals

theorem time_between_rings_is_288_minutes (total_minutes_in_day total_rings : ℕ) 
  (h1 : total_minutes_in_day = 1440) (h2 : total_rings = 6) : 
  intervals_between_rings total_rings total_minutes_in_day = 288 := 
by 
  sorry

end NUMINAMATH_GPT_time_between_rings_is_288_minutes_l1041_104123


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1041_104125

theorem sum_of_squares_of_roots :
  let a := 10
  let b := 16
  let c := -18
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots ^ 2 - 2 * product_of_roots = 244 / 25 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1041_104125


namespace NUMINAMATH_GPT_relation_between_abc_l1041_104187

theorem relation_between_abc (a b c : ℕ) (h₁ : a = 3 ^ 44) (h₂ : b = 4 ^ 33) (h₃ : c = 5 ^ 22) : a > b ∧ b > c :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_relation_between_abc_l1041_104187


namespace NUMINAMATH_GPT_interest_rate_per_annum_l1041_104159

noncomputable def principal : ℝ := 933.3333333333334
noncomputable def amount : ℝ := 1120
noncomputable def time : ℝ := 4

theorem interest_rate_per_annum (P A T : ℝ) (hP : P = principal) (hA : A = amount) (hT : T = time) :
  ∃ R : ℝ, R = 1.25 :=
sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l1041_104159


namespace NUMINAMATH_GPT_circle_center_radius_1_circle_center_coordinates_radius_1_l1041_104172

theorem circle_center_radius_1 (x y : ℝ) : 
  x^2 + y^2 + 2*x - 4*y - 3 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 8 :=
sorry

theorem circle_center_coordinates_radius_1 : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 3 = 0 ∧ (x, y) = (-1, 2)) ∧ 
  (∃ r : ℝ, r = 2*Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_circle_center_radius_1_circle_center_coordinates_radius_1_l1041_104172


namespace NUMINAMATH_GPT_range_of_a_l1041_104150

variable (f : ℝ → ℝ)
variable (a : ℝ)

theorem range_of_a (h1 : ∀ a : ℝ, (f (1 - 2 * a) / 2 ≥ f a))
                  (h2 : ∀ (x1 x2 : ℝ), x1 < x2 ∧ x1 + x2 ≠ 0 → f x1 > f x2) : a > (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1041_104150


namespace NUMINAMATH_GPT_find_f_prime_at_2_l1041_104129

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end NUMINAMATH_GPT_find_f_prime_at_2_l1041_104129


namespace NUMINAMATH_GPT_factory_material_equation_correct_l1041_104174

variable (a b x : ℝ)
variable (h_a : a = 180)
variable (h_b : b = 120)
variable (h_condition : (a - 2 * x) - (b + x) = 30)

theorem factory_material_equation_correct : (180 - 2 * x) - (120 + x) = 30 := by
  rw [←h_a, ←h_b]
  exact h_condition

end NUMINAMATH_GPT_factory_material_equation_correct_l1041_104174


namespace NUMINAMATH_GPT_pre_image_of_f_5_1_l1041_104146

def f (x y : ℝ) : ℝ × ℝ := (x + y, 2 * x - y)

theorem pre_image_of_f_5_1 : ∃ (x y : ℝ), f x y = (5, 1) ∧ (x, y) = (2, 3) :=
by
  sorry

end NUMINAMATH_GPT_pre_image_of_f_5_1_l1041_104146


namespace NUMINAMATH_GPT_intersection_point_l1041_104166

theorem intersection_point (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) :
  ∃ x y, (y = a * x^2 + b * x + c) ∧ (y = a * x^2 - b * x + c + d) ∧ x = d / (2 * b) ∧ y = a * (d / (2 * b))^2 + (d / 2) + c :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1041_104166


namespace NUMINAMATH_GPT_sequence_bound_l1041_104144

-- Definitions and assumptions based on the conditions
def valid_sequence (a : ℕ → ℕ) (N : ℕ) (m : ℕ) :=
  (1 ≤ a 1) ∧ (a m ≤ N) ∧ (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) ∧ 
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N)

-- The main theorem to prove
theorem sequence_bound (a : ℕ → ℕ) (N : ℕ) (m : ℕ) 
  (h : valid_sequence a N m) : m ≤ 2 * Nat.floor (Real.sqrt N) :=
sorry

end NUMINAMATH_GPT_sequence_bound_l1041_104144


namespace NUMINAMATH_GPT_general_term_formula_sum_inequality_l1041_104154

noncomputable def a (n : ℕ) : ℝ := if n > 0 then (-1)^(n-1) * 3 / 2^n else 0

noncomputable def S (n : ℕ) : ℝ := if n > 0 then 1 - (-1/2)^n else 0

theorem general_term_formula (n : ℕ) (hn : n > 0) :
  a n = (-1)^(n-1) * (3/2^n) :=
by sorry

theorem sum_inequality (n : ℕ) (hn : n > 0) :
  S n + 1 / S n ≤ 13 / 6 :=
by sorry

end NUMINAMATH_GPT_general_term_formula_sum_inequality_l1041_104154


namespace NUMINAMATH_GPT_pentagon_arithmetic_progression_angle_l1041_104101

theorem pentagon_arithmetic_progression_angle (a n : ℝ) 
  (h1 : a + (a + n) + (a + 2 * n) + (a + 3 * n) + (a + 4 * n) = 540) :
  a + 2 * n = 108 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_arithmetic_progression_angle_l1041_104101


namespace NUMINAMATH_GPT_problem_one_problem_two_l1041_104177

-- Define p and q
def p (a x : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := |x - 3| < 1

-- Problem (1)
theorem problem_one (a : ℝ) (h_a : a = 1) (h_pq : p a x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem_two (a : ℝ) (h_a_pos : a > 0) (suff : ¬ p a x → ¬ q x) (not_necess : ¬ (¬ q x → ¬ p a x)) : 
  (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_GPT_problem_one_problem_two_l1041_104177


namespace NUMINAMATH_GPT_chicago_bulls_heat_games_total_l1041_104162

-- Statement of the problem in Lean 4
theorem chicago_bulls_heat_games_total :
  ∀ (bulls_games : ℕ) (heat_games : ℕ),
    bulls_games = 70 →
    heat_games = bulls_games + 5 →
    bulls_games + heat_games = 145 :=
by
  intros bulls_games heat_games h_bulls h_heat
  rw [h_bulls, h_heat]
  exact sorry

end NUMINAMATH_GPT_chicago_bulls_heat_games_total_l1041_104162


namespace NUMINAMATH_GPT_expr_div_24_l1041_104180

theorem expr_div_24 (a : ℤ) : 24 ∣ ((a^2 + 3*a + 1)^2 - 1) := 
by 
  sorry

end NUMINAMATH_GPT_expr_div_24_l1041_104180


namespace NUMINAMATH_GPT_probability_of_drawing_black_ball_l1041_104158

/-- The bag contains 2 black balls and 3 white balls. 
    The balls are identical except for their colors. 
    A ball is randomly drawn from the bag. -/
theorem probability_of_drawing_black_ball (b w : ℕ) (hb : b = 2) (hw : w = 3) :
    (b + w > 0) → (b / (b + w) : ℚ) = 2 / 5 :=
by
  intros h
  rw [hb, hw]
  norm_num

end NUMINAMATH_GPT_probability_of_drawing_black_ball_l1041_104158


namespace NUMINAMATH_GPT_marble_problem_l1041_104190

-- Defining the problem in Lean statement
theorem marble_problem 
  (m : ℕ) (n k : ℕ) (hx : m = 220) (hy : n = 20) : 
  (∀ x : ℕ, (k = n + x) → (m / n = 11) → (m / k = 10)) → (x = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_marble_problem_l1041_104190


namespace NUMINAMATH_GPT_percentage_change_l1041_104156

def original_income (P T : ℝ) : ℝ :=
  P * T

def new_income (P T : ℝ) : ℝ :=
  (P * 1.3333) * (T * 0.6667)

theorem percentage_change (P T : ℝ) (hP : P ≠ 0) (hT : T ≠ 0) :
  ((new_income P T - original_income P T) / original_income P T) * 100 = -11.11 :=
by
  sorry

end NUMINAMATH_GPT_percentage_change_l1041_104156


namespace NUMINAMATH_GPT_possible_strings_after_moves_l1041_104194

theorem possible_strings_after_moves : 
  let initial_string := "HHMMMMTT"
  let moves := [("HM", "MH"), ("MT", "TM"), ("TH", "HT")]
  let binom := Nat.choose 8 4
  binom = 70 := by
  sorry

end NUMINAMATH_GPT_possible_strings_after_moves_l1041_104194


namespace NUMINAMATH_GPT_clea_ride_time_l1041_104106

noncomputable def walk_down_stopped (x y : ℝ) : Prop := 90 * x = y
noncomputable def walk_down_moving (x y k : ℝ) : Prop := 30 * (x + k) = y
noncomputable def ride_time (y k t : ℝ) : Prop := t = y / k

theorem clea_ride_time (x y k t : ℝ) (h1 : walk_down_stopped x y) (h2 : walk_down_moving x y k) :
  ride_time y k t → t = 45 :=
sorry

end NUMINAMATH_GPT_clea_ride_time_l1041_104106


namespace NUMINAMATH_GPT_maximum_n_l1041_104132

def number_of_trapezoids (n : ℕ) : ℕ := n * (n - 3) * (n - 2) * (n - 1) / 24

theorem maximum_n (n : ℕ) (h : number_of_trapezoids n ≤ 2012) : n ≤ 26 :=
by
  sorry

end NUMINAMATH_GPT_maximum_n_l1041_104132


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1041_104141

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2017 - 1 ≠ (x - 1) * (y^2015 - 1) :=
by sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1041_104141


namespace NUMINAMATH_GPT_man_cannot_row_against_stream_l1041_104175

theorem man_cannot_row_against_stream (rate_in_still_water speed_with_stream : ℝ)
  (h_rate : rate_in_still_water = 1)
  (h_speed_with : speed_with_stream = 6) :
  ¬ ∃ (speed_against_stream : ℝ), speed_against_stream = rate_in_still_water - (speed_with_stream - rate_in_still_water) :=
by
  sorry

end NUMINAMATH_GPT_man_cannot_row_against_stream_l1041_104175


namespace NUMINAMATH_GPT_stratified_sampling_b_members_l1041_104133

variable (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) (sampleSize : ℕ)

-- Conditions from the problem
def condition1 : groupA = 45 := by sorry
def condition2 : groupB = 45 := by sorry
def condition3 : groupC = 60 := by sorry
def condition4 : sampleSize = 10 := by sorry

-- The proof problem statement
theorem stratified_sampling_b_members : 
  (sampleSize * groupB) / (groupA + groupB + groupC) = 3 :=
by sorry

end NUMINAMATH_GPT_stratified_sampling_b_members_l1041_104133


namespace NUMINAMATH_GPT_other_train_length_l1041_104147

noncomputable def relative_speed (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

noncomputable def speed_in_km_per_sec (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr / 3600

noncomputable def total_distance_crossed (relative_speed : ℝ) (time_sec : ℕ) : ℝ :=
  relative_speed * (time_sec : ℝ)

noncomputable def length_of_other_train (total_distance length_of_first_train : ℝ) : ℝ :=
  total_distance - length_of_first_train

theorem other_train_length :
  let speed1 := 210
  let speed2 := 90
  let length_of_first_train := 0.9
  let time_taken := 24
  let relative_speed_km_per_hr := relative_speed speed1 speed2
  let relative_speed_km_per_sec := speed_in_km_per_sec relative_speed_km_per_hr
  let total_distance := total_distance_crossed relative_speed_km_per_sec time_taken
  length_of_other_train total_distance length_of_first_train = 1.1 := 
by
  sorry

end NUMINAMATH_GPT_other_train_length_l1041_104147


namespace NUMINAMATH_GPT_unique_solution_c_min_l1041_104196

theorem unique_solution_c_min (x y : ℝ) (c : ℝ)
  (h1 : 2 * (x+7)^2 + (y-4)^2 = c)
  (h2 : (x+4)^2 + 2 * (y-7)^2 = c) :
  c = 6 :=
sorry

end NUMINAMATH_GPT_unique_solution_c_min_l1041_104196


namespace NUMINAMATH_GPT_smallest_difference_l1041_104185

theorem smallest_difference {a b : ℕ} (h1: a * b = 2010) (h2: a > b) : a - b = 37 :=
sorry

end NUMINAMATH_GPT_smallest_difference_l1041_104185


namespace NUMINAMATH_GPT_tina_assignment_time_l1041_104104

theorem tina_assignment_time (total_time clean_time_per_key remaining_keys assignment_time : ℕ) 
  (h1 : total_time = 52) 
  (h2 : clean_time_per_key = 3) 
  (h3 : remaining_keys = 14) 
  (h4 : assignment_time = total_time - remaining_keys * clean_time_per_key) :
  assignment_time = 10 :=
by
  rw [h1, h2, h3] at h4
  assumption

end NUMINAMATH_GPT_tina_assignment_time_l1041_104104


namespace NUMINAMATH_GPT_coefficient_x18_is_zero_coefficient_x17_is_3420_l1041_104115

open Polynomial

noncomputable def P : Polynomial ℚ := (1 + X^5 + X^7)^20

theorem coefficient_x18_is_zero : coeff P 18 = 0 :=
sorry

theorem coefficient_x17_is_3420 : coeff P 17 = 3420 :=
sorry

end NUMINAMATH_GPT_coefficient_x18_is_zero_coefficient_x17_is_3420_l1041_104115


namespace NUMINAMATH_GPT_mike_coins_value_l1041_104198

theorem mike_coins_value (d q : ℕ)
  (h1 : d + q = 17)
  (h2 : q + 3 = 2 * d) :
  10 * d + 25 * q = 345 :=
by
  sorry

end NUMINAMATH_GPT_mike_coins_value_l1041_104198


namespace NUMINAMATH_GPT_original_number_is_500_l1041_104121

theorem original_number_is_500 (x : ℝ) (h1 : x * 1.3 = 650) : x = 500 :=
sorry

end NUMINAMATH_GPT_original_number_is_500_l1041_104121


namespace NUMINAMATH_GPT_exists_integers_cd_iff_divides_l1041_104179

theorem exists_integers_cd_iff_divides (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (a - b) ∣ (2 * a * b) := 
by
  sorry

end NUMINAMATH_GPT_exists_integers_cd_iff_divides_l1041_104179


namespace NUMINAMATH_GPT_rate_of_interest_per_annum_l1041_104105

theorem rate_of_interest_per_annum (R : ℝ) : 
  (5000 * R * 2 / 100) + (3000 * R * 4 / 100) = 1540 → 
  R = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_rate_of_interest_per_annum_l1041_104105


namespace NUMINAMATH_GPT_correct_average_l1041_104152

theorem correct_average (avg_incorrect : ℕ) (old_num new_num : ℕ) (n : ℕ)
  (h_avg : avg_incorrect = 15)
  (h_old_num : old_num = 26)
  (h_new_num : new_num = 36)
  (h_n : n = 10) :
  (avg_incorrect * n + (new_num - old_num)) / n = 16 := by
  sorry

end NUMINAMATH_GPT_correct_average_l1041_104152


namespace NUMINAMATH_GPT_point_on_parallel_line_with_P_l1041_104183

-- Definitions
def is_on_parallel_line_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.snd = Q.snd

theorem point_on_parallel_line_with_P :
  let P := (3, -2)
  let D := (-3, -2)
  is_on_parallel_line_x_axis P D :=
by
  sorry

end NUMINAMATH_GPT_point_on_parallel_line_with_P_l1041_104183


namespace NUMINAMATH_GPT_quadrilateral_area_proof_l1041_104117

-- Assume we have a rectangle with area 24 cm^2 and two triangles with total area 7.5 cm^2.
-- We want to prove the area of the quadrilateral ABCD is 16.5 cm^2 inside this rectangle.

def rectangle_area : ℝ := 24
def triangles_area : ℝ := 7.5
def quadrilateral_area : ℝ := rectangle_area - triangles_area

theorem quadrilateral_area_proof : quadrilateral_area = 16.5 := 
by
  exact sorry

end NUMINAMATH_GPT_quadrilateral_area_proof_l1041_104117


namespace NUMINAMATH_GPT_locus_of_centers_of_tangent_circles_l1041_104197

noncomputable def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25
noncomputable def locus (a b : ℝ) : Prop := 4 * a^2 + 4 * b^2 - 6 * a - 25 = 0

theorem locus_of_centers_of_tangent_circles :
  (∃ (a b r : ℝ), a^2 + b^2 = (r + 1)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2) →
  (∃ a b : ℝ, locus a b) :=
sorry

end NUMINAMATH_GPT_locus_of_centers_of_tangent_circles_l1041_104197


namespace NUMINAMATH_GPT_geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l1041_104160

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end NUMINAMATH_GPT_geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l1041_104160


namespace NUMINAMATH_GPT_max_q_value_l1041_104145

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_q_value_l1041_104145


namespace NUMINAMATH_GPT_feathers_already_have_l1041_104116

-- Given conditions
def total_feathers : Nat := 900
def feathers_still_needed : Nat := 513

-- Prove that the number of feathers Charlie already has is 387
theorem feathers_already_have : (total_feathers - feathers_still_needed) = 387 := by
  sorry

end NUMINAMATH_GPT_feathers_already_have_l1041_104116


namespace NUMINAMATH_GPT_abs_neg_two_l1041_104178

theorem abs_neg_two : abs (-2) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_two_l1041_104178


namespace NUMINAMATH_GPT_scientific_notation_of_investment_l1041_104100

theorem scientific_notation_of_investment : 41800000000 = 4.18 * 10^10 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_investment_l1041_104100


namespace NUMINAMATH_GPT_area_of_ABCD_is_196_l1041_104112

-- Define the shorter side length of the smaller rectangles
def shorter_side : ℕ := 7

-- Define the longer side length of the smaller rectangles
def longer_side : ℕ := 2 * shorter_side

-- Define the width of rectangle ABCD
def width_ABCD : ℕ := 2 * shorter_side

-- Define the length of rectangle ABCD
def length_ABCD : ℕ := longer_side

-- Define the area of rectangle ABCD
def area_ABCD : ℕ := length_ABCD * width_ABCD

-- Statement of the problem
theorem area_of_ABCD_is_196 : area_ABCD = 196 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_area_of_ABCD_is_196_l1041_104112


namespace NUMINAMATH_GPT_polynomial_divisible_l1041_104108

theorem polynomial_divisible (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, (x-1)^3 ∣ x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_l1041_104108


namespace NUMINAMATH_GPT_average_visitors_per_day_l1041_104107

/-- The average number of visitors per day in a month of 30 days that begins with a Sunday is 188, 
given that the library has 500 visitors on Sundays and 140 visitors on other days. -/
theorem average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) 
   (starts_on_sunday : Bool) (sundays : ℕ) 
   (visitors_sunday_eq_500 : visitors_sunday = 500)
   (visitors_other_eq_140 : visitors_other = 140)
   (days_in_month_eq_30 : days_in_month = 30)
   (starts_on_sunday_eq_true : starts_on_sunday = true)
   (sundays_eq_4 : sundays = 4) :
   (visitors_sunday * sundays + visitors_other * (days_in_month - sundays)) / days_in_month = 188 := 
by {
  sorry
}

end NUMINAMATH_GPT_average_visitors_per_day_l1041_104107


namespace NUMINAMATH_GPT_area_of_shaded_region_l1041_104130

theorem area_of_shaded_region :
  let width := 10
  let height := 5
  let base_triangle := 3
  let height_triangle := 2
  let top_base_trapezoid := 3
  let bottom_base_trapezoid := 6
  let height_trapezoid := 3
  let area_rectangle := width * height
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle
  let area_trapezoid := (1 / 2 : ℝ) * (top_base_trapezoid + bottom_base_trapezoid) * height_trapezoid
  let area_shaded := area_rectangle - area_triangle - area_trapezoid
  area_shaded = 33.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1041_104130


namespace NUMINAMATH_GPT_lowest_discount_l1041_104151

theorem lowest_discount (c m : ℝ) (p : ℝ) (h_c : c = 100) (h_m : m = 150) (h_p : p = 0.05) :
  ∃ (x : ℝ), m * (x / 100) = c * (1 + p) ∧ x = 70 :=
by
  use 70
  sorry

end NUMINAMATH_GPT_lowest_discount_l1041_104151


namespace NUMINAMATH_GPT_edward_work_hours_edward_work_hours_overtime_l1041_104164

variable (H : ℕ) -- H represents the number of hours worked
variable (O : ℕ) -- O represents the number of overtime hours

theorem edward_work_hours (H_le_40 : H ≤ 40) (earning_eq_210 : 7 * H = 210) : H = 30 :=
by
  -- Proof to be filled in here
  sorry

theorem edward_work_hours_overtime (H_gt_40 : H > 40) (earning_eq_210 : 7 * 40 + 14 * (H - 40) = 210) : False :=
by
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_edward_work_hours_edward_work_hours_overtime_l1041_104164


namespace NUMINAMATH_GPT_remainder_when_divided_by_5_l1041_104195

theorem remainder_when_divided_by_5 (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3)
  (h3 : k < 41) : k % 5 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_5_l1041_104195


namespace NUMINAMATH_GPT_construct_quadratic_l1041_104110

-- Definitions from the problem's conditions
def quadratic_has_zeros (f : ℝ → ℝ) (r1 r2 : ℝ) : Prop :=
  f r1 = 0 ∧ f r2 = 0

def quadratic_value_at (f : ℝ → ℝ) (x_val value : ℝ) : Prop :=
  f x_val = value

-- Construct the Lean theorem statement
theorem construct_quadratic :
  ∃ f : ℝ → ℝ, quadratic_has_zeros f 1 5 ∧ quadratic_value_at f 3 10 ∧
  ∀ x, f x = (-5/2 : ℝ) * x^2 + 15 * x - 25 / 2 :=
sorry

end NUMINAMATH_GPT_construct_quadratic_l1041_104110


namespace NUMINAMATH_GPT_theater_workshop_l1041_104161

-- Definitions of the conditions
def total_participants : ℕ := 120
def cannot_craft_poetry : ℕ := 52
def cannot_perform_painting : ℕ := 75
def not_skilled_in_photography : ℕ := 38
def participants_with_exactly_two_skills : ℕ := 195 - total_participants

-- The theorem stating the problem
theorem theater_workshop :
  participants_with_exactly_two_skills = 75 := by
  sorry

end NUMINAMATH_GPT_theater_workshop_l1041_104161


namespace NUMINAMATH_GPT_initial_kids_l1041_104171

theorem initial_kids {N : ℕ} (h1 : 1 / 2 * N = N / 2) (h2 : 1 / 2 * (N / 2) = N / 4) (h3 : N / 4 = 5) : N = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_kids_l1041_104171


namespace NUMINAMATH_GPT_max_length_sequence_l1041_104118

def seq_term (n : ℕ) (y : ℤ) : ℤ :=
  match n with
  | 0 => 2000
  | 1 => y
  | k + 2 => seq_term (k + 1) y - seq_term k y

theorem max_length_sequence (y : ℤ) :
  1200 < y ∧ y < 1334 ∧ (∀ n, seq_term n y ≥ 0 ∨ seq_term (n + 1) y < 0) ↔ y = 1333 :=
by
  sorry

end NUMINAMATH_GPT_max_length_sequence_l1041_104118


namespace NUMINAMATH_GPT_find_xy_l1041_104184

theorem find_xy (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : (x - 10)^2 + (y - 10)^2 = 18) : 
  x * y = 91 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_xy_l1041_104184


namespace NUMINAMATH_GPT_cylinder_volume_increase_factor_l1041_104122

theorem cylinder_volume_increase_factor
    (π : Real)
    (r h : Real)
    (V_original : Real := π * r^2 * h)
    (new_height : Real := 3 * h)
    (new_radius : Real := 4 * r)
    (V_new : Real := π * (new_radius)^2 * new_height) :
    V_new / V_original = 48 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_increase_factor_l1041_104122


namespace NUMINAMATH_GPT_age_ratio_in_8_years_l1041_104169

-- Define the conditions
variables (s l : ℕ) -- Sam's and Leo's current ages

def condition1 := s - 4 = 2 * (l - 4)
def condition2 := s - 10 = 3 * (l - 10)

-- Define the final problem
theorem age_ratio_in_8_years (h1 : condition1 s l) (h2 : condition2 s l) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (l + x) = 3 / 2 :=
sorry

end NUMINAMATH_GPT_age_ratio_in_8_years_l1041_104169


namespace NUMINAMATH_GPT_inequality_holds_iff_m_eq_n_l1041_104120

theorem inequality_holds_iff_m_eq_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∀ (α β : ℝ), 
    ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ 
    ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) ↔ m = n :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_iff_m_eq_n_l1041_104120


namespace NUMINAMATH_GPT_ac_bd_bound_l1041_104109

variables {a b c d : ℝ}

theorem ac_bd_bound (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 4) : |a * c + b * d| ≤ 2 := 
sorry

end NUMINAMATH_GPT_ac_bd_bound_l1041_104109


namespace NUMINAMATH_GPT_total_ingredient_cups_l1041_104173

def butter_flour_sugar_ratio_butter := 2
def butter_flour_sugar_ratio_flour := 5
def butter_flour_sugar_ratio_sugar := 3
def flour_used := 15

theorem total_ingredient_cups :
  butter_flour_sugar_ratio_butter + 
  butter_flour_sugar_ratio_flour + 
  butter_flour_sugar_ratio_sugar = 10 →
  flour_used / butter_flour_sugar_ratio_flour = 3 →
  6 + 15 + 9 = 30 := by
  intros
  sorry

end NUMINAMATH_GPT_total_ingredient_cups_l1041_104173


namespace NUMINAMATH_GPT_pipe_B_fill_time_l1041_104140

-- Definitions based on the given conditions
def fill_time_by_ABC := 10  -- in hours
def B_is_twice_as_fast_as_C : Prop := ∀ C B, B = 2 * C
def A_is_twice_as_fast_as_B : Prop := ∀ A B, A = 2 * B

-- The main theorem to prove
theorem pipe_B_fill_time (A B C : ℝ) (h1: fill_time_by_ABC = 10) 
    (h2 : B_is_twice_as_fast_as_C) (h3 : A_is_twice_as_fast_as_B) : B = 1 / 35 :=
by
  sorry

end NUMINAMATH_GPT_pipe_B_fill_time_l1041_104140


namespace NUMINAMATH_GPT_function_increasing_no_negative_roots_l1041_104153

noncomputable def f (a x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem function_increasing (a : ℝ) (h : a > 1) : 
  ∀ (x1 x2 : ℝ), (-1 < x1) → (x1 < x2) → (f a x1 < f a x2) := 
by
  -- placeholder proof
  sorry

theorem no_negative_roots (a : ℝ) (h : a > 1) : 
  ∀ (x : ℝ), (x < 0) → (f a x ≠ 0) := 
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_function_increasing_no_negative_roots_l1041_104153


namespace NUMINAMATH_GPT_square_area_is_correct_l1041_104182

noncomputable def find_area_of_square (x : ℚ) : ℚ :=
  let side := 6 * x - 27
  side * side

theorem square_area_is_correct (x : ℚ) (h1 : 6 * x - 27 = 30 - 2 * x) :
  find_area_of_square x = 248.0625 :=
by
  sorry

end NUMINAMATH_GPT_square_area_is_correct_l1041_104182


namespace NUMINAMATH_GPT_find_xyz_l1041_104128

theorem find_xyz
  (a b c x y z : ℂ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : a = (2 * b + 3 * c) / (x - 3))
  (h2 : b = (3 * a + 2 * c) / (y - 3))
  (h3 : c = (2 * a + 2 * b) / (z - 3))
  (h4 : x * y + x * z + y * z = -1)
  (h5 : x + y + z = 1) :
  x * y * z = 1 :=
sorry

end NUMINAMATH_GPT_find_xyz_l1041_104128


namespace NUMINAMATH_GPT_Chris_age_proof_l1041_104157

theorem Chris_age_proof (m c : ℕ) (h1 : c = 3 * m - 22) (h2 : c + m = 70) : c = 47 := by
  sorry

end NUMINAMATH_GPT_Chris_age_proof_l1041_104157


namespace NUMINAMATH_GPT_squirrel_calories_l1041_104168

def rabbits_caught_per_hour := 2
def rabbits_calories := 800
def squirrels_caught_per_hour := 6
def extra_calories_squirrels := 200

theorem squirrel_calories : 
  ∀ (S : ℕ), 
  (6 * S = (2 * 800) + 200) → S = 300 := by
  intros S h
  sorry

end NUMINAMATH_GPT_squirrel_calories_l1041_104168


namespace NUMINAMATH_GPT_lemonade_problem_l1041_104189

theorem lemonade_problem (L S W : ℕ) (h1 : W = 4 * S) (h2 : S = 2 * L) (h3 : L = 3) : L + S + W = 24 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_problem_l1041_104189


namespace NUMINAMATH_GPT_total_fruit_weight_l1041_104199

def melon_weight : Real := 0.35
def berries_weight : Real := 0.48
def grapes_weight : Real := 0.29
def pineapple_weight : Real := 0.56
def oranges_weight : Real := 0.17

theorem total_fruit_weight : melon_weight + berries_weight + grapes_weight + pineapple_weight + oranges_weight = 1.85 :=
by
  unfold melon_weight berries_weight grapes_weight pineapple_weight oranges_weight
  sorry

end NUMINAMATH_GPT_total_fruit_weight_l1041_104199


namespace NUMINAMATH_GPT_student_question_choices_l1041_104163

-- Definitions based on conditions
def partA_questions := 10
def partB_questions := 10
def choose_from_partA := 8
def choose_from_partB := 5

-- The proof problem statement
theorem student_question_choices :
  (Nat.choose partA_questions choose_from_partA) * (Nat.choose partB_questions choose_from_partB) = 11340 :=
by
  sorry

end NUMINAMATH_GPT_student_question_choices_l1041_104163


namespace NUMINAMATH_GPT_frequency_of_heads_l1041_104136

theorem frequency_of_heads (n h : ℕ) (h_n : n = 100) (h_h : h = 49) : (h : ℚ) / n = 0.49 :=
by
  rw [h_n, h_h]
  norm_num

end NUMINAMATH_GPT_frequency_of_heads_l1041_104136


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l1041_104138

theorem arithmetic_sequence_sum_ratio
  (a_n : ℕ → ℝ)
  (d a1 : ℝ)
  (S_n : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n = a1 + (n-1) * d)
  (h_sum : ∀ n, S_n n = n / 2 * (2 * a1 + (n-1) * d))
  (h_ratio : S_n 4 / S_n 6 = -2 / 3) :
  S_n 5 / S_n 8 = 1 / 40.8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l1041_104138


namespace NUMINAMATH_GPT_min_value_of_sum_squares_l1041_104137

noncomputable def min_value_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : ℝ :=
  y1^2 + y2^2 + y3^2

theorem min_value_of_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : 
  min_value_sum_squares y1 y2 y3 h1 h2 h3 h4 = 14400 / 29 := 
sorry

end NUMINAMATH_GPT_min_value_of_sum_squares_l1041_104137


namespace NUMINAMATH_GPT_consecutive_integer_sum_l1041_104181

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end NUMINAMATH_GPT_consecutive_integer_sum_l1041_104181


namespace NUMINAMATH_GPT_percentage_problem_l1041_104114

theorem percentage_problem (x : ℝ) (h : (3 / 8) * x = 141) : (round (0.3208 * x) = 121) :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1041_104114


namespace NUMINAMATH_GPT_determine_right_triangle_l1041_104191

variable (A B C : ℝ)
variable (AB BC AC : ℝ)

-- Conditions as definitions
def condition1 : Prop := A + C = B
def condition2 : Prop := A = 30 ∧ B = 60 ∧ C = 90 -- Since ratio 1:2:3 means A = 30, B = 60, C = 90

-- Proof problem statement
theorem determine_right_triangle (h1 : condition1 A B C) (h2 : condition2 A B C) : (B = 90) :=
sorry

end NUMINAMATH_GPT_determine_right_triangle_l1041_104191


namespace NUMINAMATH_GPT_infinite_k_lcm_gt_ck_l1041_104165

theorem infinite_k_lcm_gt_ck 
  (a : ℕ → ℕ) 
  (distinct_pos : ∀ n m : ℕ, n ≠ m → a n ≠ a m) 
  (pos : ∀ n, 0 < a n) 
  (c : ℝ) 
  (c_pos : 0 < c) 
  (c_lt : c < 1.5) : 
  ∃ᶠ k in at_top, (Nat.lcm (a k) (a (k + 1)) : ℝ) > c * k :=
sorry

end NUMINAMATH_GPT_infinite_k_lcm_gt_ck_l1041_104165


namespace NUMINAMATH_GPT_hyperbola_foci_x_axis_range_l1041_104176

theorem hyperbola_foci_x_axis_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1) →
  (1 < m) ↔ 
  (∀ x y : ℝ, (m + 2 > 0) ∧ (m - 1 > 0)) :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_x_axis_range_l1041_104176


namespace NUMINAMATH_GPT_find_x_solutions_l1041_104126

theorem find_x_solutions :
  ∀ {x : ℝ}, (x = (1/x) + (-x)^2 + 3) → (x = -1 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_x_solutions_l1041_104126


namespace NUMINAMATH_GPT_number_of_binders_l1041_104135

-- Definitions of given conditions
def book_cost : Nat := 16
def binder_cost : Nat := 2
def notebooks_cost : Nat := 6
def total_cost : Nat := 28

-- Variable for the number of binders
variable (b : Nat)

-- Proposition that the number of binders Léa bought is 3
theorem number_of_binders (h : book_cost + binder_cost * b + notebooks_cost = total_cost) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_binders_l1041_104135


namespace NUMINAMATH_GPT_red_minus_white_more_l1041_104143

variable (flowers_total yellow_white red_yellow red_white : ℕ)
variable (h1 : flowers_total = 44)
variable (h2 : yellow_white = 13)
variable (h3 : red_yellow = 17)
variable (h4 : red_white = 14)

theorem red_minus_white_more : 
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end NUMINAMATH_GPT_red_minus_white_more_l1041_104143


namespace NUMINAMATH_GPT_regular_polygon_sides_l1041_104193

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ m : ℕ, m = 360 / n → n ≠ 0 → m = 30) : n = 12 :=
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1041_104193


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1041_104142

theorem find_x2_plus_y2 (x y : ℝ) (h : (x ^ 2 + y ^ 2 + 1) * (x ^ 2 + y ^ 2 - 3) = 5) : x ^ 2 + y ^ 2 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1041_104142


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1041_104103

theorem problem1 : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 :=
by
  sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 6) ^ 2 - (Real.sqrt 5 + Real.sqrt 6) ^ 2 = -4 * Real.sqrt 30 :=
by
  sorry

theorem problem3 : (2 * Real.sqrt (3 / 2) - Real.sqrt (1 / 2)) * (1 / 2 * Real.sqrt 8 + Real.sqrt (2 / 3)) = (5 / 3) * Real.sqrt 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1041_104103


namespace NUMINAMATH_GPT_num_sets_l1041_104149

theorem num_sets {A : Set ℕ} :
  {1} ⊆ A ∧ A ⊆ {1, 2, 3, 4, 5} → ∃ n, n = 16 := 
by
  sorry

end NUMINAMATH_GPT_num_sets_l1041_104149


namespace NUMINAMATH_GPT_combined_distance_l1041_104134

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end NUMINAMATH_GPT_combined_distance_l1041_104134


namespace NUMINAMATH_GPT_intersection_points_and_verification_l1041_104148

theorem intersection_points_and_verification :
  (∃ x y : ℝ, y = -3 * x ∧ y + 3 = 9 * x ∧ x = 1 / 4 ∧ y = -3 / 4) ∧
  ¬ (y = 2 * (1 / 4) - 1 ∧ (2 * (1 / 4) - 1 = -3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_and_verification_l1041_104148


namespace NUMINAMATH_GPT_simplify_expression_l1041_104167

theorem simplify_expression (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1041_104167


namespace NUMINAMATH_GPT_no_integer_solutions_l1041_104131

theorem no_integer_solutions
  (x y : ℤ) :
  3 * x^2 = 16 * y^2 + 8 * y + 5 → false :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1041_104131


namespace NUMINAMATH_GPT_youngest_brother_age_l1041_104113

theorem youngest_brother_age (x : ℕ) (h : x + (x + 1) + (x + 2) = 96) : x = 31 :=
sorry

end NUMINAMATH_GPT_youngest_brother_age_l1041_104113


namespace NUMINAMATH_GPT_incorrect_inequality_l1041_104155

theorem incorrect_inequality (a b : ℝ) (h : a > b ∧ b > 0) :
  ¬ (1 / a > 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_inequality_l1041_104155


namespace NUMINAMATH_GPT_error_percentage_calc_l1041_104186

theorem error_percentage_calc (y : ℝ) (hy : y > 0) : 
  let correct_result := 8 * y
  let erroneous_result := y / 8
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 := by
  sorry

end NUMINAMATH_GPT_error_percentage_calc_l1041_104186
