import Mathlib

namespace NUMINAMATH_GPT_Daniel_had_more_than_200_marbles_at_day_6_l931_93174

noncomputable def marbles (k : ℕ) : ℕ :=
  5 * 2^k

theorem Daniel_had_more_than_200_marbles_at_day_6 :
  ∃ k : ℕ, marbles k > 200 ∧ ∀ m < k, marbles m ≤ 200 :=
by
  sorry

end NUMINAMATH_GPT_Daniel_had_more_than_200_marbles_at_day_6_l931_93174


namespace NUMINAMATH_GPT_geometric_sequence_sum_n5_l931_93195

def geometric_sum (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_n5 (a₁ q : ℕ) (n : ℕ) (h₁ : a₁ = 3) (h₂ : q = 4) (h₃ : n = 5) : 
  geometric_sum a₁ q n = 1023 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_n5_l931_93195


namespace NUMINAMATH_GPT_Emily_average_speed_l931_93184

noncomputable def Emily_run_distance : ℝ := 10

noncomputable def speed_first_uphill : ℝ := 4
noncomputable def distance_first_uphill : ℝ := 2

noncomputable def speed_first_downhill : ℝ := 6
noncomputable def distance_first_downhill : ℝ := 1

noncomputable def speed_flat_ground : ℝ := 5
noncomputable def distance_flat_ground : ℝ := 3

noncomputable def speed_second_uphill : ℝ := 4.5
noncomputable def distance_second_uphill : ℝ := 2

noncomputable def speed_second_downhill : ℝ := 6
noncomputable def distance_second_downhill : ℝ := 2

noncomputable def break_first : ℝ := 5 / 60
noncomputable def break_second : ℝ := 7 / 60
noncomputable def break_third : ℝ := 3 / 60

noncomputable def time_first_uphill : ℝ := distance_first_uphill / speed_first_uphill
noncomputable def time_first_downhill : ℝ := distance_first_downhill / speed_first_downhill
noncomputable def time_flat_ground : ℝ := distance_flat_ground / speed_flat_ground
noncomputable def time_second_uphill : ℝ := distance_second_uphill / speed_second_uphill
noncomputable def time_second_downhill : ℝ := distance_second_downhill / speed_second_downhill

noncomputable def total_running_time : ℝ := time_first_uphill + time_first_downhill + time_flat_ground + time_second_uphill + time_second_downhill
noncomputable def total_break_time : ℝ := break_first + break_second + break_third
noncomputable def total_time : ℝ := total_running_time + total_break_time

noncomputable def average_speed : ℝ := Emily_run_distance / total_time

theorem Emily_average_speed : abs (average_speed - 4.36) < 0.01 := by
  sorry

end NUMINAMATH_GPT_Emily_average_speed_l931_93184


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l931_93122

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l931_93122


namespace NUMINAMATH_GPT_eliza_received_12_almonds_l931_93182

theorem eliza_received_12_almonds (y : ℕ) (h1 : y - 8 = y / 3) : y = 12 :=
sorry

end NUMINAMATH_GPT_eliza_received_12_almonds_l931_93182


namespace NUMINAMATH_GPT_triangle_properties_l931_93180

open Real

variables (A B C a b c : ℝ) (triangle_obtuse triangle_right triangle_acute : Prop)

-- Declaration of properties 
def sin_gt (A B : ℝ) := sin A > sin B
def tan_product_lt (A C : ℝ) := tan A * tan C < 1
def cos_squared_eq (A B C : ℝ) := cos A ^ 2 + cos B ^ 2 - cos C ^ 2 = 1

theorem triangle_properties :
  (sin_gt A B → A > B) ∧
  (triangle_obtuse → tan_product_lt A C) ∧
  (cos_squared_eq A B C → triangle_right) :=
  by sorry

end NUMINAMATH_GPT_triangle_properties_l931_93180


namespace NUMINAMATH_GPT_milan_long_distance_bill_l931_93116

theorem milan_long_distance_bill
  (monthly_fee : ℝ := 2)
  (per_minute_cost : ℝ := 0.12)
  (minutes_used : ℕ := 178) :
  ((minutes_used : ℝ) * per_minute_cost + monthly_fee = 23.36) :=
by
  sorry

end NUMINAMATH_GPT_milan_long_distance_bill_l931_93116


namespace NUMINAMATH_GPT_mass_of_man_l931_93168

def density_of_water : ℝ := 1000  -- kg/m³
def boat_length : ℝ := 4  -- meters
def boat_breadth : ℝ := 2  -- meters
def sinking_depth : ℝ := 0.01  -- meters (1 cm)

theorem mass_of_man
  (V : ℝ := boat_length * boat_breadth * sinking_depth)
  (m : ℝ := V * density_of_water) :
  m = 80 :=
by
  sorry

end NUMINAMATH_GPT_mass_of_man_l931_93168


namespace NUMINAMATH_GPT_probability_of_satisfaction_l931_93152

-- Definitions for the conditions given in the problem
def dissatisfied_customers_leave_negative_review_probability : ℝ := 0.8
def satisfied_customers_leave_positive_review_probability : ℝ := 0.15
def negative_reviews : ℕ := 60
def positive_reviews : ℕ := 20
def expected_satisfaction_probability : ℝ := 0.64

-- The problem to prove
theorem probability_of_satisfaction :
  ∃ p : ℝ, (dissatisfied_customers_leave_negative_review_probability * (1 - p) = negative_reviews / (negative_reviews + positive_reviews)) ∧
           (satisfied_customers_leave_positive_review_probability * p = positive_reviews / (negative_reviews + positive_reviews)) ∧
           p = expected_satisfaction_probability := 
by
  sorry

end NUMINAMATH_GPT_probability_of_satisfaction_l931_93152


namespace NUMINAMATH_GPT_prop_false_iff_a_lt_neg_13_over_2_l931_93137

theorem prop_false_iff_a_lt_neg_13_over_2 :
  (¬ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + a * x + 9 ≥ 0) ↔ a < -13 / 2 := 
sorry

end NUMINAMATH_GPT_prop_false_iff_a_lt_neg_13_over_2_l931_93137


namespace NUMINAMATH_GPT_epicenter_distance_l931_93131

noncomputable def distance_from_epicenter (v1 v2 Δt: ℝ) : ℝ :=
  Δt / ((1 / v2) - (1 / v1))

theorem epicenter_distance : 
  distance_from_epicenter 5.94 3.87 11.5 = 128 := 
by
  -- The proof will use calculations shown in the solution.
  sorry

end NUMINAMATH_GPT_epicenter_distance_l931_93131


namespace NUMINAMATH_GPT_chinese_characters_digits_l931_93191

theorem chinese_characters_digits:
  ∃ (a b g s t : ℕ), -- Chinese characters represented by digits
    -- Different characters represent different digits
    a ≠ b ∧ a ≠ g ∧ a ≠ s ∧ a ≠ t ∧
    b ≠ g ∧ b ≠ s ∧ b ≠ t ∧
    g ≠ s ∧ g ≠ t ∧
    s ≠ t ∧
    -- Equation: 业步高 * 业步高 = 高升抬步高
    (a * 100 + b * 10 + g) * (a * 100 + b * 10 + g) = (g * 10000 + s * 1000 + t * 100 + b * 10 + g) :=
by {
  -- We need to prove that the number represented by "高升抬步高" is 50625.
  sorry
}

end NUMINAMATH_GPT_chinese_characters_digits_l931_93191


namespace NUMINAMATH_GPT_leila_yards_l931_93113

variable (mile_yards : ℕ := 1760)
variable (marathon_miles : ℕ := 28)
variable (marathon_yards : ℕ := 1500)
variable (marathons_ran : ℕ := 15)

theorem leila_yards (m y : ℕ) (h1 : marathon_miles = 28) (h2 : marathon_yards = 1500) (h3 : mile_yards = 1760) (h4 : marathons_ran = 15) (hy : 0 ≤ y ∧ y < mile_yards) :
  y = 1200 :=
sorry

end NUMINAMATH_GPT_leila_yards_l931_93113


namespace NUMINAMATH_GPT_four_digit_perfect_square_l931_93150

theorem four_digit_perfect_square (N : ℕ) (a b : ℤ) :
  N = 1100 * a + 11 * b ∧
  N >= 1000 ∧ N <= 9999 ∧
  a >= 0 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧
  (∃ (x : ℤ), N = 11 * x^2) →
  N = 7744 := by
  sorry

end NUMINAMATH_GPT_four_digit_perfect_square_l931_93150


namespace NUMINAMATH_GPT_age_ratio_five_years_later_l931_93107

theorem age_ratio_five_years_later (my_age : ℕ) (son_age : ℕ) (h1 : my_age = 45) (h2 : son_age = 15) :
  (my_age + 5) / gcd (my_age + 5) (son_age + 5) = 5 ∧ (son_age + 5) / gcd (my_age + 5) (son_age + 5) = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_five_years_later_l931_93107


namespace NUMINAMATH_GPT_find_t_l931_93167

variable (a t : ℝ)

def f (x : ℝ) : ℝ := a * x + 19

theorem find_t (h1 : f a 3 = 7) (h2 : f a t = 15) : t = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l931_93167


namespace NUMINAMATH_GPT_min_value_expression_l931_93128

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (9 / a) + (16 / b) + (25 / c)

theorem min_value_expression :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 6 →
  min_expression a b c ≥ 18 :=
by
  intro a b c ha hb hc habc
  sorry

end NUMINAMATH_GPT_min_value_expression_l931_93128


namespace NUMINAMATH_GPT_student_l931_93146

noncomputable def allowance_after_video_games (A : ℝ) : ℝ := (3 / 7) * A

noncomputable def allowance_after_comic_books (remaining_after_video_games : ℝ) : ℝ := (3 / 5) * remaining_after_video_games

noncomputable def allowance_after_trading_cards (remaining_after_comic_books : ℝ) : ℝ := (5 / 8) * remaining_after_comic_books

noncomputable def last_allowance (remaining_after_trading_cards : ℝ) : ℝ := remaining_after_trading_cards

theorem student's_monthly_allowance (A : ℝ) (h1 : last_allowance (allowance_after_trading_cards (allowance_after_comic_books (allowance_after_video_games A))) = 1.20) :
  A = 7.47 := 
sorry

end NUMINAMATH_GPT_student_l931_93146


namespace NUMINAMATH_GPT_right_triangle_midpoints_distances_l931_93194

theorem right_triangle_midpoints_distances (a b : ℝ) 
  (hXON : 19^2 = a^2 + (b/2)^2)
  (hYOM : 22^2 = b^2 + (a/2)^2) :
  a^2 + b^2 = 676 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_midpoints_distances_l931_93194


namespace NUMINAMATH_GPT_intersection_result_l931_93109

noncomputable def A : Set ℝ := { x | x^2 - 5*x - 6 < 0 }
noncomputable def B : Set ℝ := { x | 2022^x > Real.sqrt 2022 }
noncomputable def intersection : Set ℝ := { x | A x ∧ B x }

theorem intersection_result : intersection = Set.Ioo (1/2 : ℝ) 6 := by
  sorry

end NUMINAMATH_GPT_intersection_result_l931_93109


namespace NUMINAMATH_GPT_area_excluding_hole_l931_93190

def area_large_rectangle (x : ℝ) : ℝ :=
  (2 * x + 9) * (x + 6)

def area_square_hole (x : ℝ) : ℝ :=
  (x - 1) * (x - 1)

theorem area_excluding_hole (x : ℝ) : 
  area_large_rectangle x - area_square_hole x = x^2 + 23 * x + 53 :=
by
  sorry

end NUMINAMATH_GPT_area_excluding_hole_l931_93190


namespace NUMINAMATH_GPT_total_flowers_correct_l931_93130

def rosa_original_flowers : ℝ := 67.5
def andre_gifted_flowers : ℝ := 90.75
def total_flowers (rosa : ℝ) (andre : ℝ) : ℝ := rosa + andre

theorem total_flowers_correct : total_flowers rosa_original_flowers andre_gifted_flowers = 158.25 :=
by 
  rw [total_flowers]
  sorry

end NUMINAMATH_GPT_total_flowers_correct_l931_93130


namespace NUMINAMATH_GPT_number_of_strikers_l931_93196

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end NUMINAMATH_GPT_number_of_strikers_l931_93196


namespace NUMINAMATH_GPT_modulus_z_eq_one_l931_93135

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z : ℂ := (1 - imaginary_unit) / (1 + imaginary_unit) 

theorem modulus_z_eq_one : Complex.abs z = 1 := 
sorry

end NUMINAMATH_GPT_modulus_z_eq_one_l931_93135


namespace NUMINAMATH_GPT_max_sum_factors_of_60_exists_max_sum_factors_of_60_l931_93185

theorem max_sum_factors_of_60 (d Δ : ℕ) (h : d * Δ = 60) : (d + Δ) ≤ 61 :=
sorry

theorem exists_max_sum_factors_of_60 : ∃ d Δ : ℕ, d * Δ = 60 ∧ d + Δ = 61 :=
sorry

end NUMINAMATH_GPT_max_sum_factors_of_60_exists_max_sum_factors_of_60_l931_93185


namespace NUMINAMATH_GPT_meals_without_restrictions_l931_93102

theorem meals_without_restrictions (total_clients vegan kosher gluten_free halal dairy_free nut_free vegan_kosher vegan_gluten_free kosher_gluten_free halal_dairy_free gluten_free_nut_free vegan_halal_gluten_free kosher_dairy_free_nut_free : ℕ) 
  (h_tc : total_clients = 80)
  (h_vegan : vegan = 15)
  (h_kosher : kosher = 18)
  (h_gluten_free : gluten_free = 12)
  (h_halal : halal = 10)
  (h_dairy_free : dairy_free = 8)
  (h_nut_free : nut_free = 4)
  (h_vegan_kosher : vegan_kosher = 5)
  (h_vegan_gluten_free : vegan_gluten_free = 6)
  (h_kosher_gluten_free : kosher_gluten_free = 3)
  (h_halal_dairy_free : halal_dairy_free = 4)
  (h_gluten_free_nut_free : gluten_free_nut_free = 2)
  (h_vegan_halal_gluten_free : vegan_halal_gluten_free = 2)
  (h_kosher_dairy_free_nut_free : kosher_dairy_free_nut_free = 1) : 
  (total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free 
  - vegan_kosher - vegan_gluten_free - kosher_gluten_free - halal_dairy_free - gluten_free_nut_free 
  + vegan_halal_gluten_free + kosher_dairy_free_nut_free) = 30) :=
by {
  -- solution steps here
  sorry
}

end NUMINAMATH_GPT_meals_without_restrictions_l931_93102


namespace NUMINAMATH_GPT_compound_interest_rate_l931_93156

theorem compound_interest_rate : 
  let P := 14800
  let interest := 4265.73
  let A := 19065.73
  let t := 2
  let n := 1
  let r := 0.13514
  (P : ℝ) * (1 + r)^t = A :=
by
-- Here we will provide the steps of the proof
sorry

end NUMINAMATH_GPT_compound_interest_rate_l931_93156


namespace NUMINAMATH_GPT_Toms_dog_age_in_6_years_l931_93121

-- Let's define the conditions
variables (B D : ℕ)
axiom h1 : B = 4 * D
axiom h2 : B + 6 = 30

-- Now we state the theorem
theorem Toms_dog_age_in_6_years :
  D + 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_Toms_dog_age_in_6_years_l931_93121


namespace NUMINAMATH_GPT_T7_value_l931_93148

-- Define the geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Define the even function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + 2 * a

-- The main theorem statement
theorem T7_value (a : ℕ → ℝ) (a2 a6 : ℝ) (a_val : ℝ) (q : ℝ) (T7 : ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : a 2 = a2)
  (h3 : a 6 = a6)
  (h4 : a2 - 2 = f a_val 0)
  (h5 : a6 - 3 = f a_val 0)
  (h6 : q > 1)
  (h7 : T7 = a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) : 
  T7 = 128 :=
sorry

end NUMINAMATH_GPT_T7_value_l931_93148


namespace NUMINAMATH_GPT_total_games_friends_l931_93197

def new_friends_games : ℕ := 88
def old_friends_games : ℕ := 53

theorem total_games_friends :
  new_friends_games + old_friends_games = 141 :=
by
  sorry

end NUMINAMATH_GPT_total_games_friends_l931_93197


namespace NUMINAMATH_GPT_terry_lunch_combo_l931_93192

theorem terry_lunch_combo :
  let lettuce_options : ℕ := 2
  let tomato_options : ℕ := 3
  let olive_options : ℕ := 4
  let soup_options : ℕ := 2
  (lettuce_options * tomato_options * olive_options * soup_options = 48) := 
by
  sorry

end NUMINAMATH_GPT_terry_lunch_combo_l931_93192


namespace NUMINAMATH_GPT_sum_of_fractions_l931_93176

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end NUMINAMATH_GPT_sum_of_fractions_l931_93176


namespace NUMINAMATH_GPT_rem_neg_one_third_quarter_l931_93142

noncomputable def rem (x y : ℝ) : ℝ :=
  x - y * ⌊x / y⌋

theorem rem_neg_one_third_quarter :
  rem (-1/3) (1/4) = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_rem_neg_one_third_quarter_l931_93142


namespace NUMINAMATH_GPT_geometric_sequence_a6_l931_93166

noncomputable def a_sequence (n : ℕ) : ℝ := 1 * 2^(n-1)

theorem geometric_sequence_a6 (S : ℕ → ℝ)
  (h1 : S 10 = 3 * S 5)
  (h2 : ∀ n, S n = (1 - 2^n) / (1 - 2))
  (h3 : a_sequence 1 = 1) :
  a_sequence 6 = 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l931_93166


namespace NUMINAMATH_GPT_num_students_is_92_l931_93154

noncomputable def total_students (S : ℕ) : Prop :=
  let remaining := S - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  walking = 27

theorem num_students_is_92 : total_students 92 :=
by
  let remaining := 92 - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  have walk_eq : walking = 27 := by sorry
  exact walk_eq

end NUMINAMATH_GPT_num_students_is_92_l931_93154


namespace NUMINAMATH_GPT_find_root_floor_l931_93160

noncomputable def g (x : ℝ) := Real.sin x - Real.cos x + 4 * Real.tan x

theorem find_root_floor :
  ∃ s : ℝ, (g s = 0) ∧ (π / 2 < s) ∧ (s < 3 * π / 2) ∧ (Int.floor s = 3) :=
  sorry

end NUMINAMATH_GPT_find_root_floor_l931_93160


namespace NUMINAMATH_GPT_regular_octagon_interior_angle_l931_93104

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end NUMINAMATH_GPT_regular_octagon_interior_angle_l931_93104


namespace NUMINAMATH_GPT_max_value_inequality_l931_93149

theorem max_value_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 3) :
  (x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) ≤ 27 := 
sorry

end NUMINAMATH_GPT_max_value_inequality_l931_93149


namespace NUMINAMATH_GPT_stratified_sampling_l931_93136

variable (H M L total_sample : ℕ)
variable (H_fams M_fams L_fams : ℕ)

-- Conditions
def community : Prop := H_fams = 150 ∧ M_fams = 360 ∧ L_fams = 90
def total_population : Prop := H_fams + M_fams + L_fams = 600
def sample_size : Prop := total_sample = 100

-- Statement
theorem stratified_sampling (H_fams M_fams L_fams : ℕ) (total_sample : ℕ)
  (h_com : community H_fams M_fams L_fams)
  (h_total_pop : total_population H_fams M_fams L_fams)
  (h_sample_size : sample_size total_sample)
  : H = 25 ∧ M = 60 ∧ L = 15 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l931_93136


namespace NUMINAMATH_GPT_evaluate_expression_l931_93132

-- Definition of the conditions
def a : ℕ := 15
def b : ℕ := 19
def c : ℕ := 13

-- Problem statement
theorem evaluate_expression :
  (225 * (1 / a - 1 / b) + 361 * (1 / b - 1 / c) + 169 * (1 / c - 1 / a))
  /
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = a + b + c :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l931_93132


namespace NUMINAMATH_GPT_Mina_additional_miles_l931_93181

theorem Mina_additional_miles:
  let distance1 := 20 -- distance in miles for the first part of the trip
  let speed1 := 40 -- speed in mph for the first part of the trip
  let speed2 := 60 -- speed in mph for the second part of the trip
  let avg_speed := 55 -- average speed needed for the entire trip in mph
  let distance2 := (distance1 / speed1 + (avg_speed * (distance1 / speed1)) / (speed1 - avg_speed * speed1 / speed2)) * speed2 -- formula to find the additional distance
  distance2 = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_Mina_additional_miles_l931_93181


namespace NUMINAMATH_GPT_sum_of_solutions_eq_320_l931_93153

theorem sum_of_solutions_eq_320 :
  ∃ (S : Finset ℝ), 
  (∀ x ∈ S, 0 < x ∧ x < 180 ∧ (1 + (Real.sin x / Real.sin (4 * x)) = (Real.sin (3 * x) / Real.sin (2 * x)))) 
  ∧ S.sum id = 320 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_solutions_eq_320_l931_93153


namespace NUMINAMATH_GPT_no_positive_integer_satisfies_l931_93177

theorem no_positive_integer_satisfies : ¬ ∃ n : ℕ, 0 < n ∧ (20 * n + 2) ∣ (2003 * n + 2002) :=
by sorry

end NUMINAMATH_GPT_no_positive_integer_satisfies_l931_93177


namespace NUMINAMATH_GPT_number_of_regular_pencils_l931_93143

def cost_eraser : ℝ := 0.8
def cost_regular : ℝ := 0.5
def cost_short : ℝ := 0.4
def num_eraser : ℕ := 200
def num_short : ℕ := 35
def total_revenue : ℝ := 194

theorem number_of_regular_pencils (num_regular : ℕ) :
  (num_eraser * cost_eraser) + (num_short * cost_short) + (num_regular * cost_regular) = total_revenue → 
  num_regular = 40 :=
by
  sorry

end NUMINAMATH_GPT_number_of_regular_pencils_l931_93143


namespace NUMINAMATH_GPT_find_x_l931_93145

def sum_sequence (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2  -- Sum of an arithmetic progression

theorem find_x (x : ℕ) (h1 : sum_sequence x 10 = 65) : x = 2 :=
by {
  -- the proof goes here
  sorry
}

end NUMINAMATH_GPT_find_x_l931_93145


namespace NUMINAMATH_GPT_triangle_area_l931_93144

theorem triangle_area (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) : 
  (1/2) * a * b = 150 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l931_93144


namespace NUMINAMATH_GPT_smallest_unpayable_amount_l931_93158

theorem smallest_unpayable_amount :
  ∀ (coins_1p coins_2p coins_3p coins_4p coins_5p : ℕ), 
    coins_1p = 1 → 
    coins_2p = 2 → 
    coins_3p = 3 → 
    coins_4p = 4 → 
    coins_5p = 5 → 
    ∃ (x : ℕ), x = 56 ∧ 
    ¬ (∃ (a b c d e : ℕ), a * 1 + b * 2 + c * 3 + d * 4 + e * 5 = x ∧ 
    a ≤ coins_1p ∧
    b ≤ coins_2p ∧
    c ≤ coins_3p ∧
    d ≤ coins_4p ∧
    e ≤ coins_5p) :=
by {
  -- Here we skip the actual proof
  sorry
}

end NUMINAMATH_GPT_smallest_unpayable_amount_l931_93158


namespace NUMINAMATH_GPT_expression_equals_eight_l931_93100

theorem expression_equals_eight
  (a b c : ℝ)
  (h1 : a + b = 2 * c)
  (h2 : b + c = 2 * a)
  (h3 : a + c = 2 * b) :
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end NUMINAMATH_GPT_expression_equals_eight_l931_93100


namespace NUMINAMATH_GPT_certain_number_example_l931_93120

theorem certain_number_example (x : ℝ) 
    (h1 : 213 * 16 = 3408)
    (h2 : 0.16 * x = 0.3408) : 
    x = 2.13 := 
by 
  sorry

end NUMINAMATH_GPT_certain_number_example_l931_93120


namespace NUMINAMATH_GPT_arithmetic_sequence_S10_l931_93106

-- Definition of an arithmetic sequence and the corresponding sums S_n.
def is_arithmetic_sequence (S : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, S (n + 1) = S n + d

theorem arithmetic_sequence_S10 
  (S : ℕ → ℕ)
  (h1 : S 1 = 10)
  (h2 : S 2 = 20)
  (h_arith : is_arithmetic_sequence S) :
  S 10 = 100 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_S10_l931_93106


namespace NUMINAMATH_GPT_inequality_solution_l931_93118

def solution_set_of_inequality (x : ℝ) : Prop :=
  x * (x - 1) < 0

theorem inequality_solution :
  { x : ℝ | solution_set_of_inequality x } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l931_93118


namespace NUMINAMATH_GPT_principal_argument_of_z_l931_93198

-- Mathematical definitions based on provided conditions
noncomputable def theta : ℝ := Real.arctan (5 / 12)

-- The complex number z defined in the problem
noncomputable def z : ℂ := (Real.cos (2 * theta) + Real.sin (2 * theta) * Complex.I) / (239 + Complex.I)

-- Lean statement to prove the argument of z
theorem principal_argument_of_z : Complex.arg z = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_principal_argument_of_z_l931_93198


namespace NUMINAMATH_GPT_average_temperature_correct_l931_93139

-- Definition of the daily temperatures
def daily_temperatures : List ℕ := [51, 64, 61, 59, 48, 63, 55]

-- Define the number of days
def number_of_days : ℕ := 7

-- Prove the average temperature calculation
theorem average_temperature_correct :
  ((List.sum daily_temperatures : ℚ) / number_of_days : ℚ) = 57.3 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_correct_l931_93139


namespace NUMINAMATH_GPT_find_x_l931_93173

/--
Given the following conditions:
1. The sum of angles around a point is 360 degrees.
2. The angles are 7x, 6x, 3x, and (2x + y).
3. y = 2x.

Prove that x = 18 degrees.
-/
theorem find_x (x y : ℝ) (h : 18 * x + y = 360) (h_y : y = 2 * x) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l931_93173


namespace NUMINAMATH_GPT_maria_trip_time_l931_93169

theorem maria_trip_time 
(s_highway : ℕ) (s_mountain : ℕ) (d_highway : ℕ) (d_mountain : ℕ) (t_mountain : ℕ) (t_break : ℕ) : 
  (s_highway = 4 * s_mountain) -> 
  (t_mountain = d_mountain / s_mountain) -> 
  t_mountain = 40 -> 
  t_break = 15 -> 
  d_highway = 100 -> 
  d_mountain = 20 ->
  s_mountain = d_mountain / t_mountain -> 
  s_highway = 4 * s_mountain -> 
  d_highway / s_highway = 50 ->
  40 + 50 + 15 = 105 := 
by 
  sorry

end NUMINAMATH_GPT_maria_trip_time_l931_93169


namespace NUMINAMATH_GPT_set_union_eq_l931_93165

open Set

noncomputable def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {-1, 0}
def C : Set ℤ := {-1, 0, 1}

theorem set_union_eq :
  A ∪ B = C :=
by {
  sorry
}

end NUMINAMATH_GPT_set_union_eq_l931_93165


namespace NUMINAMATH_GPT_train_crosses_platform_in_15_seconds_l931_93125

-- Definitions based on conditions
def length_of_train : ℝ := 330 -- in meters
def tunnel_length : ℝ := 1200 -- in meters
def time_to_cross_tunnel : ℝ := 45 -- in seconds
def platform_length : ℝ := 180 -- in meters

-- Definition based on the solution but directly asserting the correct answer.
def time_to_cross_platform : ℝ := 15 -- in seconds

-- Lean statement
theorem train_crosses_platform_in_15_seconds :
  (length_of_train + platform_length) / ((length_of_train + tunnel_length) / time_to_cross_tunnel) = time_to_cross_platform :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_in_15_seconds_l931_93125


namespace NUMINAMATH_GPT_root_relationship_specific_root_five_l931_93164

def f (x : ℝ) : ℝ := x^3 - 6 * x^2 - 39 * x - 10
def g (x : ℝ) : ℝ := x^3 + x^2 - 20 * x - 50

theorem root_relationship :
  ∃ (x_0 : ℝ), g x_0 = 0 ∧ f (2 * x_0) = 0 :=
sorry

theorem specific_root_five :
  g 5 = 0 ∧ f 10 = 0 :=
sorry

end NUMINAMATH_GPT_root_relationship_specific_root_five_l931_93164


namespace NUMINAMATH_GPT_fifth_term_is_19_l931_93101

-- Define the first term and the common difference
def a₁ : Int := 3
def d : Int := 4

-- Define the formula for the nth term in the arithmetic sequence
def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

-- Define the Lean 4 statement proving that the 5th term is 19
theorem fifth_term_is_19 : arithmetic_sequence 5 = 19 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_fifth_term_is_19_l931_93101


namespace NUMINAMATH_GPT_correct_translation_of_tradition_l931_93193

def is_adjective (s : String) : Prop :=
  s = "传统的"

def is_correct_translation (s : String) (translation : String) : Prop :=
  s = "传统的" → translation = "traditional"

theorem correct_translation_of_tradition : 
  is_adjective "传统的" ∧ is_correct_translation "传统的" "traditional" :=
by
  sorry

end NUMINAMATH_GPT_correct_translation_of_tradition_l931_93193


namespace NUMINAMATH_GPT_tomatoes_eaten_l931_93115

theorem tomatoes_eaten (initial_tomatoes : ℕ) (remaining_tomatoes : ℕ) (portion_eaten : ℚ)
  (h_init : initial_tomatoes = 21)
  (h_rem : remaining_tomatoes = 14)
  (h_portion : portion_eaten = 1/3) :
  initial_tomatoes - remaining_tomatoes = (portion_eaten * initial_tomatoes) :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_eaten_l931_93115


namespace NUMINAMATH_GPT_find_y_l931_93189

theorem find_y (x y : ℕ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : ∃ q : ℕ, x = q * y + 9) (h₃ : x / y = 96 + 3 / 20) : y = 60 :=
sorry

end NUMINAMATH_GPT_find_y_l931_93189


namespace NUMINAMATH_GPT_find_f_2010_l931_93124

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f x + f (3 - x)

theorem find_f_2010 : f 2010 = 0 := sorry

end NUMINAMATH_GPT_find_f_2010_l931_93124


namespace NUMINAMATH_GPT_steven_seeds_l931_93162

def average_seeds (fruit: String) : Nat :=
  match fruit with
  | "apple" => 6
  | "pear" => 2
  | "grape" => 3
  | "orange" => 10
  | "watermelon" => 300
  | _ => 0

def fruits := [("apple", 2), ("pear", 3), ("grape", 5), ("orange", 1), ("watermelon", 2)]

def required_seeds := 420

def total_seeds (fruit_list : List (String × Nat)) : Nat :=
  fruit_list.foldr (fun (fruit_qty : String × Nat) acc =>
    acc + (average_seeds fruit_qty.fst) * fruit_qty.snd) 0

theorem steven_seeds : total_seeds fruits - required_seeds = 223 := by
  sorry

end NUMINAMATH_GPT_steven_seeds_l931_93162


namespace NUMINAMATH_GPT_min_x2_y2_of_product_eq_zero_l931_93108

theorem min_x2_y2_of_product_eq_zero (x y : ℝ) (h : (x + 8) * (y - 8) = 0) : x^2 + y^2 = 64 :=
sorry

end NUMINAMATH_GPT_min_x2_y2_of_product_eq_zero_l931_93108


namespace NUMINAMATH_GPT_find_a9_l931_93133

variable {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a3_eq_1 (a : ℕ → ℝ) : Prop := 
  a 3 = 1

def a5_a6_a7_eq_8 (a : ℕ → ℝ) : Prop := 
  a 5 * a 6 * a 7 = 8

-- Theorem to prove
theorem find_a9 {a : ℕ → ℝ} {q : ℝ} 
  (geom : geom_seq a q)
  (ha3 : a3_eq_1 a)
  (ha5a6a7 : a5_a6_a7_eq_8 a) : a 9 = 4 := 
sorry

end NUMINAMATH_GPT_find_a9_l931_93133


namespace NUMINAMATH_GPT_expression_is_perfect_cube_l931_93157

theorem expression_is_perfect_cube {x y z : ℝ} (h : x + y + z = 0) :
  ∃ m : ℝ, 
    (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) * 
    (x^3 * y * z + x * y^3 * z + x * y * z^3) *
    (x^3 * y^2 * z + x^3 * y * z^2 + x^2 * y^3 * z + x * y^3 * z^2 + x^2 * y * z^3 + x * y^2 * z^3) =
    m ^ 3 := 
by 
  sorry

end NUMINAMATH_GPT_expression_is_perfect_cube_l931_93157


namespace NUMINAMATH_GPT_blue_dress_difference_l931_93172

theorem blue_dress_difference 
(total_space : ℕ)
(red_dresses : ℕ)
(blue_dresses : ℕ)
(h1 : total_space = 200)
(h2 : red_dresses = 83)
(h3 : blue_dresses = total_space - red_dresses) :
blue_dresses - red_dresses = 34 :=
by
  rw [h1, h2] at h3
  sorry -- Proof details go here.

end NUMINAMATH_GPT_blue_dress_difference_l931_93172


namespace NUMINAMATH_GPT_joe_dropped_score_l931_93178

theorem joe_dropped_score (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 60) (h2 : (A + B + C) / 3 = 65) :
  min A (min B (min C D)) = D → D = 45 :=
by sorry

end NUMINAMATH_GPT_joe_dropped_score_l931_93178


namespace NUMINAMATH_GPT_no_polyhedron_with_surface_2015_l931_93103

/--
It is impossible to glue together 1 × 1 × 1 cubes to form a polyhedron whose surface area is 2015.
-/
theorem no_polyhedron_with_surface_2015 (n k : ℕ) : 6 * n - 2 * k ≠ 2015 :=
by
  sorry

end NUMINAMATH_GPT_no_polyhedron_with_surface_2015_l931_93103


namespace NUMINAMATH_GPT_sum_of_four_consecutive_even_numbers_l931_93188

theorem sum_of_four_consecutive_even_numbers (n : ℤ) (h : n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) :
  n + (n + 2) + (n + 4) + (n + 6) = 36 := sorry

end NUMINAMATH_GPT_sum_of_four_consecutive_even_numbers_l931_93188


namespace NUMINAMATH_GPT_correct_choice_for_games_l931_93171
  
-- Define the problem context
def games_preferred (question : String) (answer : String) :=
  question = "Which of the two computer games did you prefer?" ∧
  answer = "Actually I didn’t like either of them."

-- Define the proof that the correct choice is 'either of them'
theorem correct_choice_for_games (question : String) (answer : String) :
  games_preferred question answer → answer = "either of them" :=
by
  -- Provided statement and proof assumptions
  intro h
  cases h
  exact sorry -- Proof steps will be here
  -- Here, the conclusion should be derived from given conditions

end NUMINAMATH_GPT_correct_choice_for_games_l931_93171


namespace NUMINAMATH_GPT_remainder_when_divided_by_9_l931_93163

noncomputable def base12_to_dec (x : ℕ) : ℕ :=
  (1 * 12^3) + (5 * 12^2) + (3 * 12) + 4
  
theorem remainder_when_divided_by_9 : base12_to_dec (1534) % 9 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_9_l931_93163


namespace NUMINAMATH_GPT_sum_of_coefficients_l931_93199

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (∀ x : ℤ, (1 + x)^6 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 63 :=
by
  intros h ha
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l931_93199


namespace NUMINAMATH_GPT_distance_A_B_l931_93186

theorem distance_A_B 
  (perimeter_small_square : ℝ)
  (area_large_square : ℝ)
  (h1 : perimeter_small_square = 8)
  (h2 : area_large_square = 64) :
  let side_small_square := perimeter_small_square / 4
  let side_large_square := Real.sqrt area_large_square
  let horizontal_distance := side_small_square + side_large_square
  let vertical_distance := side_large_square - side_small_square
  let distance_AB := Real.sqrt (horizontal_distance^2 + vertical_distance^2)
  distance_AB = 11.7 :=
  by sorry

end NUMINAMATH_GPT_distance_A_B_l931_93186


namespace NUMINAMATH_GPT_max_t_squared_value_l931_93141

noncomputable def max_t_squared (R : ℝ) : ℝ :=
  let PR_QR_sq_sum := 4 * R^2
  let max_PR_QR_prod := 2 * R^2
  PR_QR_sq_sum + 2 * max_PR_QR_prod

theorem max_t_squared_value (R : ℝ) : max_t_squared R = 8 * R^2 :=
  sorry

end NUMINAMATH_GPT_max_t_squared_value_l931_93141


namespace NUMINAMATH_GPT_ahmed_final_score_requirement_l931_93114

-- Define the given conditions
def total_assignments : ℕ := 9
def ahmed_initial_grade : ℕ := 91
def emily_initial_grade : ℕ := 92
def sarah_initial_grade : ℕ := 94
def final_assignment_weight := true -- Assuming each assignment has the same weight
def min_passing_score : ℕ := 70
def max_score : ℕ := 100
def emily_final_score : ℕ := 90

noncomputable def ahmed_min_final_score : ℕ := 98

-- The proof statement
theorem ahmed_final_score_requirement :
  let ahmed_initial_points := ahmed_initial_grade * total_assignments
  let emily_initial_points := emily_initial_grade * total_assignments
  let sarah_initial_points := sarah_initial_grade * total_assignments
  let emily_final_total := emily_initial_points + emily_final_score
  let sarah_final_total := sarah_initial_points + min_passing_score
  let ahmed_final_total_needed := sarah_final_total + 1
  let ahmed_needed_score := ahmed_final_total_needed - ahmed_initial_points
  ahmed_needed_score = ahmed_min_final_score :=
by
  sorry

end NUMINAMATH_GPT_ahmed_final_score_requirement_l931_93114


namespace NUMINAMATH_GPT_perp_bisector_chord_l931_93159

theorem perp_bisector_chord (x y : ℝ) :
  (2 * x + 3 * y + 1 = 0) ∧ (x^2 + y^2 - 2 * x + 4 * y = 0) → 
  ∃ k l m : ℝ, (3 * x - 2 * y - 7 = 0) :=
by
  sorry

end NUMINAMATH_GPT_perp_bisector_chord_l931_93159


namespace NUMINAMATH_GPT_marsha_first_package_miles_l931_93134

noncomputable def total_distance (x : ℝ) : ℝ := x + 28 + 14

noncomputable def earnings (x : ℝ) : ℝ := total_distance x * 2

theorem marsha_first_package_miles : ∃ x : ℝ, earnings x = 104 ∧ x = 10 :=
by
  use 10
  sorry

end NUMINAMATH_GPT_marsha_first_package_miles_l931_93134


namespace NUMINAMATH_GPT_find_n_l931_93129

noncomputable def f (x : ℤ) : ℤ := sorry -- f is some polynomial with integer coefficients

theorem find_n (n : ℤ) (h1 : f 1 = -1) (h4 : f 4 = 2) (h8 : f 8 = 34) (hn : f n = n^2 - 4 * n - 18) : n = 3 ∨ n = 6 :=
sorry

end NUMINAMATH_GPT_find_n_l931_93129


namespace NUMINAMATH_GPT_greatest_value_of_squares_l931_93151

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 170)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
sorry

end NUMINAMATH_GPT_greatest_value_of_squares_l931_93151


namespace NUMINAMATH_GPT_division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l931_93179

def f (n : ℕ) (a : ℚ) : ℚ := a ^ (2 - n)

theorem division_powers_5_half : f 5 (1/2) = 8 := by
  -- skip the proof
  sorry

theorem division_powers_6_3 : f 6 3 = 1/81 := by
  -- skip the proof
  sorry

theorem division_powers_formula (n : ℕ) (a : ℚ) (h : n > 0) : f n a = a^(2 - n) := by
  -- skip the proof
  sorry

theorem division_powers_combination : f 5 (1/3) * f 4 3 * f 5 (1/2) + f 5 (-1/4) / f 6 (-1/2) = 20 := by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l931_93179


namespace NUMINAMATH_GPT_weightlifter_total_weight_l931_93155

theorem weightlifter_total_weight (weight_one_hand : ℕ) (num_hands : ℕ) (condition: weight_one_hand = 8 ∧ num_hands = 2) :
  2 * weight_one_hand = 16 :=
by
  sorry

end NUMINAMATH_GPT_weightlifter_total_weight_l931_93155


namespace NUMINAMATH_GPT_alcohol_added_l931_93126

theorem alcohol_added (x : ℝ) :
  let initial_solution_volume := 40
  let initial_alcohol_percentage := 0.05
  let initial_alcohol_volume := initial_solution_volume * initial_alcohol_percentage
  let additional_water := 6.5
  let final_solution_volume := initial_solution_volume + x + additional_water
  let final_alcohol_percentage := 0.11
  let final_alcohol_volume := final_solution_volume * final_alcohol_percentage
  initial_alcohol_volume + x = final_alcohol_volume → x = 3.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alcohol_added_l931_93126


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_l931_93111

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_l931_93111


namespace NUMINAMATH_GPT_decreasing_function_range_l931_93112

theorem decreasing_function_range (k : ℝ) : (∀ x : ℝ, k + 2 < 0) ↔ k < -2 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l931_93112


namespace NUMINAMATH_GPT_find_annual_interest_rate_l931_93105

noncomputable def annual_interest_rate (P A n t : ℝ) : ℝ :=
  2 * ((A / P)^(1 / (n * t)) - 1)

theorem find_annual_interest_rate :
  Π (P A : ℝ) (n t : ℕ), P = 600 → A = 760 → n = 2 → t = 4 →
  annual_interest_rate P A n t = 0.06020727 :=
by
  intros P A n t hP hA hn ht
  rw [hP, hA, hn, ht]
  unfold annual_interest_rate
  sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l931_93105


namespace NUMINAMATH_GPT_percentage_sales_tax_on_taxable_purchases_l931_93187

-- Definitions
def total_cost : ℝ := 30
def tax_free_cost : ℝ := 24.7
def tax_rate : ℝ := 0.06

-- Statement to prove
theorem percentage_sales_tax_on_taxable_purchases :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1 := by
  sorry

end NUMINAMATH_GPT_percentage_sales_tax_on_taxable_purchases_l931_93187


namespace NUMINAMATH_GPT_class_total_students_l931_93140

theorem class_total_students (x y : ℕ)
  (initial_absent : y = (1/6) * x)
  (after_sending_chalk : y = (1/5) * (x - 1)) :
  x + y = 7 :=
by
  sorry

end NUMINAMATH_GPT_class_total_students_l931_93140


namespace NUMINAMATH_GPT_hyperbola_equation_l931_93170

theorem hyperbola_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (e : ℝ) (he : e = 2 * Real.sqrt 3 / 3)
  (dist_from_origin : ∀ A B : ℝ × ℝ, A = (0, -b) ∧ B = (a, 0) →
    abs (a * b) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2) :
  (a^2 = 3 ∧ b^2 = 1) → (∀ x y : ℝ, (x^2 / 3 - y^2 = 1)) := 
sorry

end NUMINAMATH_GPT_hyperbola_equation_l931_93170


namespace NUMINAMATH_GPT_solve_for_a_l931_93117

theorem solve_for_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_eq_exponents : a ^ b = b ^ a) (h_b_equals_3a : b = 3 * a) : a = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l931_93117


namespace NUMINAMATH_GPT_handshakes_total_l931_93119

def num_couples : ℕ := 15
def total_people : ℕ := 30
def men : ℕ := 15
def women : ℕ := 15
def youngest_man_handshakes : ℕ := 0
def men_handshakes : ℕ := (14 * 13) / 2
def men_women_handshakes : ℕ := 15 * 14

theorem handshakes_total : men_handshakes + men_women_handshakes = 301 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_handshakes_total_l931_93119


namespace NUMINAMATH_GPT_cone_volume_l931_93127

theorem cone_volume (S r : ℝ) : 
  ∃ V : ℝ, V = (1 / 3) * S * r :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l931_93127


namespace NUMINAMATH_GPT_initial_number_of_girls_l931_93138

theorem initial_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 7 * (b - 54) = g - 20) : g = 39 :=
sorry

end NUMINAMATH_GPT_initial_number_of_girls_l931_93138


namespace NUMINAMATH_GPT_find_some_number_eq_0_3_l931_93161

theorem find_some_number_eq_0_3 (X : ℝ) (h : 2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002) :
  X = 0.3 :=
by sorry

end NUMINAMATH_GPT_find_some_number_eq_0_3_l931_93161


namespace NUMINAMATH_GPT_discount_percent_l931_93123

theorem discount_percent
  (MP CP SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (gainPercent : ℝ)
  (h2 : gainPercent = 54.54545454545454 / 100)
  (h3 : (SP - CP) / CP = gainPercent)
  : ((MP - SP) / MP) * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_discount_percent_l931_93123


namespace NUMINAMATH_GPT_minimize_distance_postman_l931_93110

-- Let x be a function that maps house indices to coordinates.
def optimalPostOfficeLocation (n: ℕ) (x : ℕ → ℝ) : ℝ :=
  if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2)

theorem minimize_distance_postman (n: ℕ) (x : ℕ → ℝ)
  (h_sorted : ∀ i j, i < j → x i < x j) :
  optimalPostOfficeLocation n x = if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2) := 
  sorry

end NUMINAMATH_GPT_minimize_distance_postman_l931_93110


namespace NUMINAMATH_GPT_distance_interval_l931_93183

theorem distance_interval (d : ℝ) :
  (d < 8) ∧ (d > 7) ∧ (d > 5) ∧ (d ≠ 3) ↔ (7 < d ∧ d < 8) :=
by
  sorry

end NUMINAMATH_GPT_distance_interval_l931_93183


namespace NUMINAMATH_GPT_smaller_package_contains_correct_number_of_cupcakes_l931_93175

-- Define the conditions
def number_of_packs_large : ℕ := 4
def cupcakes_per_large_pack : ℕ := 15
def total_children : ℕ := 100
def needed_packs_small : ℕ := 4

-- Define the total cupcakes bought initially
def total_cupcakes_bought : ℕ := number_of_packs_large * cupcakes_per_large_pack

-- Define the total additional cupcakes needed
def additional_cupcakes_needed : ℕ := total_children - total_cupcakes_bought

-- Define the number of cupcakes per smaller package
def cupcakes_per_small_pack : ℕ := additional_cupcakes_needed / needed_packs_small

-- The theorem statement to prove
theorem smaller_package_contains_correct_number_of_cupcakes :
  cupcakes_per_small_pack = 10 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_smaller_package_contains_correct_number_of_cupcakes_l931_93175


namespace NUMINAMATH_GPT_small_pizza_slices_correct_l931_93147

-- Defining the total number of people involved
def people_count : ℕ := 3

-- Defining the number of slices each person can eat
def slices_per_person : ℕ := 12

-- Calculating the total number of slices needed based on the number of people and slices per person
def total_slices_needed : ℕ := people_count * slices_per_person

-- Defining the number of slices in a large pizza
def large_pizza_slices : ℕ := 14

-- Defining the number of large pizzas ordered
def large_pizzas_count : ℕ := 2

-- Calculating the total number of slices provided by the large pizzas
def total_large_pizza_slices : ℕ := large_pizza_slices * large_pizzas_count

-- Defining the number of slices in a small pizza
def small_pizza_slices : ℕ := 8

-- Total number of slices provided needs to be at least the total slices needed
theorem small_pizza_slices_correct :
  total_slices_needed ≤ total_large_pizza_slices + small_pizza_slices := by
  sorry

end NUMINAMATH_GPT_small_pizza_slices_correct_l931_93147
