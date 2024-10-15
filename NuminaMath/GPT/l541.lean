import Mathlib

namespace NUMINAMATH_GPT_total_albums_l541_54140

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end NUMINAMATH_GPT_total_albums_l541_54140


namespace NUMINAMATH_GPT_projected_increase_in_attendance_l541_54159

variable (A P : ℝ)

theorem projected_increase_in_attendance :
  (0.8 * A = 0.64 * (A + (P / 100) * A)) → P = 25 :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_projected_increase_in_attendance_l541_54159


namespace NUMINAMATH_GPT_solution_l541_54173

-- Definitions
def equation1 (x y z : ℝ) : Prop := 2 * x + y + z = 17
def equation2 (x y z : ℝ) : Prop := x + 2 * y + z = 14
def equation3 (x y z : ℝ) : Prop := x + y + 2 * z = 13

-- Theorem to prove
theorem solution (x y z : ℝ) (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solution_l541_54173


namespace NUMINAMATH_GPT_stable_set_even_subset_count_l541_54112

open Finset

-- Definitions
def is_stable (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ ⦃x y⦄, (x, y) ∈ S → ∀ x' y', x' ≤ x → y' ≤ y → (x', y') ∈ S

-- Main statement
theorem stable_set_even_subset_count (S : Finset (ℕ × ℕ)) (hS : is_stable S):
  (∃ E O : ℕ, E ≥ O ∧ E + O = 2 ^ (S.card)) :=
  sorry

end NUMINAMATH_GPT_stable_set_even_subset_count_l541_54112


namespace NUMINAMATH_GPT_calc_1_calc_2_calc_3_calc_4_l541_54130

section
variables {m n x y z : ℕ} -- assuming all variables are natural numbers for simplicity.
-- Problem 1
theorem calc_1 : (2 * m * n) / (3 * m ^ 2) * (6 * m * n) / (5 * n) = (4 * n) / 5 :=
sorry

-- Problem 2
theorem calc_2 : (5 * x - 5 * y) / (3 * x ^ 2 * y) * (9 * x * y ^ 2) / (x ^ 2 - y ^ 2) = 
  15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem calc_3 : ((x ^ 3 * y ^ 2) / z) ^ 2 * ((y * z) / x ^ 2) ^ 3 = y ^ 7 * z :=
sorry

-- Problem 4
theorem calc_4 : (4 * x ^ 2 * y ^ 2) / (2 * x + y) * (4 * x ^ 2 + 4 * x * y + y ^ 2) / (2 * x + y) / 
  ((2 * x * y) * (2 * x - y) / (4 * x ^ 2 - y ^ 2)) = 4 * x ^ 2 * y + 2 * x * y ^ 2 :=
sorry
end

end NUMINAMATH_GPT_calc_1_calc_2_calc_3_calc_4_l541_54130


namespace NUMINAMATH_GPT_greatest_perimeter_of_strips_l541_54107

theorem greatest_perimeter_of_strips :
  let base := 10
  let height := 12
  let half_base := base / 2
  let right_triangle_area := (base / 2 * height) / 2
  let number_of_pieces := 10
  let sub_area := right_triangle_area / (number_of_pieces / 2)
  let h1 := (2 * sub_area) / half_base
  let hypotenuse := Real.sqrt (h1^2 + (half_base / 2)^2)
  let perimeter := half_base + 2 * hypotenuse
  perimeter = 11.934 :=
by
  sorry

end NUMINAMATH_GPT_greatest_perimeter_of_strips_l541_54107


namespace NUMINAMATH_GPT_arith_seq_ratio_l541_54168

-- Definitions related to arithmetic sequence and sum
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arith_seq (S a : ℕ → ℝ) := ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition
def condition (a : ℕ → ℝ) := a 8 / a 7 = 13 / 5

-- Prove statement
theorem arith_seq_ratio (a S : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_sum : sum_of_arith_seq S a)
  (h_cond : condition a) :
  S 15 / S 13 = 3 := 
sorry

end NUMINAMATH_GPT_arith_seq_ratio_l541_54168


namespace NUMINAMATH_GPT_ratio_volumes_equal_ratio_areas_l541_54136

-- Defining necessary variables and functions
variables (R : ℝ) (S_sphere S_cone V_sphere V_cone : ℝ)

-- Conditions
def surface_area_sphere : Prop := S_sphere = 4 * Real.pi * R^2
def volume_sphere : Prop := V_sphere = (4 / 3) * Real.pi * R^3
def volume_polyhedron : Prop := V_cone = (S_cone * R) / 3

-- Theorem statement
theorem ratio_volumes_equal_ratio_areas
  (h1 : surface_area_sphere R S_sphere)
  (h2 : volume_sphere R V_sphere)
  (h3 : volume_polyhedron R S_cone V_cone)
  : (V_sphere / V_cone) = (S_sphere / S_cone) :=
sorry

end NUMINAMATH_GPT_ratio_volumes_equal_ratio_areas_l541_54136


namespace NUMINAMATH_GPT_one_fourth_more_equals_thirty_percent_less_l541_54192

theorem one_fourth_more_equals_thirty_percent_less :
  ∃ n : ℝ, 80 - 0.30 * 80 = (5 / 4) * n ∧ n = 44.8 :=
by
  sorry

end NUMINAMATH_GPT_one_fourth_more_equals_thirty_percent_less_l541_54192


namespace NUMINAMATH_GPT_jose_age_is_26_l541_54103

def Maria_age : ℕ := 14
def Jose_age (m : ℕ) : ℕ := m + 12

theorem jose_age_is_26 (m j : ℕ) (h1 : j = m + 12) (h2 : m + j = 40) : j = 26 :=
by
  sorry

end NUMINAMATH_GPT_jose_age_is_26_l541_54103


namespace NUMINAMATH_GPT_div64_by_expression_l541_54120

theorem div64_by_expression {n : ℕ} (h : n > 0) : ∃ k : ℤ, (3^(2 * n + 2) - 8 * ↑n - 9) = 64 * k :=
by
  sorry

end NUMINAMATH_GPT_div64_by_expression_l541_54120


namespace NUMINAMATH_GPT_find_lamp_cost_l541_54116

def lamp_and_bulb_costs (L B : ℝ) : Prop :=
  B = L - 4 ∧ 2 * L + 6 * B = 32

theorem find_lamp_cost : ∃ L : ℝ, ∃ B : ℝ, lamp_and_bulb_costs L B ∧ L = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_lamp_cost_l541_54116


namespace NUMINAMATH_GPT_symmetric_trapezoid_construction_possible_l541_54179

-- Define lengths of legs and distance from intersection point
variables (a b : ℝ)

-- Symmetric trapezoid feasibility condition
theorem symmetric_trapezoid_construction_possible : 3 * b > 2 * a := sorry

end NUMINAMATH_GPT_symmetric_trapezoid_construction_possible_l541_54179


namespace NUMINAMATH_GPT_min_expression_value_l541_54100

variable {a : ℕ → ℝ}
variable (m n : ℕ)
variable (q : ℝ)

axiom pos_seq (n : ℕ) : a n > 0
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom seq_condition : a 7 = a 6 + 2 * a 5
axiom exists_terms :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1)

theorem min_expression_value : 
  (∃m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1) ∧ 
  a 7 = a 6 + 2 * a 5 ∧ 
  (∀ n, a n > 0 ∧ a (n + 1) = q * a n)) → 
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_min_expression_value_l541_54100


namespace NUMINAMATH_GPT_math_problem_l541_54147

theorem math_problem
  (a b c d : ℕ)
  (h1 : a = 234)
  (h2 : b = 205)
  (h3 : c = 86400)
  (h4 : d = 300) :
  (a * b = 47970) ∧ (c / d = 288) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l541_54147


namespace NUMINAMATH_GPT_max_k_value_l541_54115

theorem max_k_value (x y : ℝ) (k : ℝ) (hx : 0 < x) (hy : 0 < y)
(h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) := sorry

end NUMINAMATH_GPT_max_k_value_l541_54115


namespace NUMINAMATH_GPT_wholesale_cost_l541_54174

variable (W R P : ℝ)

-- Conditions
def retail_price := R = 1.20 * W
def employee_discount := P = 0.95 * R
def employee_payment := P = 228

-- Theorem statement
theorem wholesale_cost (H1 : retail_price R W) (H2 : employee_discount P R) (H3 : employee_payment P) : W = 200 :=
by
  sorry

end NUMINAMATH_GPT_wholesale_cost_l541_54174


namespace NUMINAMATH_GPT_cos_B_eq_zero_l541_54118

variable {a b c A B C : ℝ}
variable (h1 : ∀ A B C, 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
variable (h2 : b * Real.cos A = c)

theorem cos_B_eq_zero (h1 : a = b) (h2 : b * Real.cos A = c) : Real.cos B = 0 :=
sorry

end NUMINAMATH_GPT_cos_B_eq_zero_l541_54118


namespace NUMINAMATH_GPT_min_value_expression_l541_54164

theorem min_value_expression (α β : ℝ) : 
  ∃ a b : ℝ, 
    ((2 * Real.cos α + 5 * Real.sin β - 8) ^ 2 + 
    (2 * Real.sin α + 5 * Real.cos β - 15) ^ 2  = 100) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l541_54164


namespace NUMINAMATH_GPT_triangle_inequality_l541_54150

theorem triangle_inequality 
  (a b c R : ℝ) 
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a) 
  (hR : R = (a * b * c) / (4 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))) : 
  a^2 + b^2 + c^2 ≤ 9 * R^2 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_l541_54150


namespace NUMINAMATH_GPT_cos_alpha_solution_l541_54101

theorem cos_alpha_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 1 / 2) : 
  Real.cos α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_solution_l541_54101


namespace NUMINAMATH_GPT_average_of_original_set_l541_54162

theorem average_of_original_set (A : ℝ) (h1 : (35 * A) = (7 * 75)) : A = 15 := 
by sorry

end NUMINAMATH_GPT_average_of_original_set_l541_54162


namespace NUMINAMATH_GPT_paul_sandwiches_l541_54153

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end NUMINAMATH_GPT_paul_sandwiches_l541_54153


namespace NUMINAMATH_GPT_x_plus_y_equals_two_l541_54165

variable (x y : ℝ)

def condition1 : Prop := (x - 1) ^ 2017 + 2013 * (x - 1) = -1
def condition2 : Prop := (y - 1) ^ 2017 + 2013 * (y - 1) = 1

theorem x_plus_y_equals_two (h1 : condition1 x) (h2 : condition2 y) : x + y = 2 :=
  sorry

end NUMINAMATH_GPT_x_plus_y_equals_two_l541_54165


namespace NUMINAMATH_GPT_range_of_a_l541_54184

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -3/4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l541_54184


namespace NUMINAMATH_GPT_min_sum_arth_seq_l541_54110

theorem min_sum_arth_seq (a : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1))
  (h2 : a 1 = -3)
  (h3 : 11 * a 5 = 5 * a 8) : n = 4 := by
  sorry

end NUMINAMATH_GPT_min_sum_arth_seq_l541_54110


namespace NUMINAMATH_GPT_one_over_nine_inv_half_eq_three_l541_54154

theorem one_over_nine_inv_half_eq_three : (1 / 9 : ℝ) ^ (-1 / 2 : ℝ) = 3 := 
by
  sorry

end NUMINAMATH_GPT_one_over_nine_inv_half_eq_three_l541_54154


namespace NUMINAMATH_GPT_two_pt_seven_five_as_fraction_l541_54198

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end NUMINAMATH_GPT_two_pt_seven_five_as_fraction_l541_54198


namespace NUMINAMATH_GPT_bus_stops_12_minutes_per_hour_l541_54167

noncomputable def stopping_time (speed_excluding_stoppages : ℝ) (speed_including_stoppages : ℝ) : ℝ :=
  let distance_lost_per_hour := speed_excluding_stoppages - speed_including_stoppages
  let speed_per_minute := speed_excluding_stoppages / 60
  distance_lost_per_hour / speed_per_minute

theorem bus_stops_12_minutes_per_hour :
  stopping_time 50 40 = 12 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_12_minutes_per_hour_l541_54167


namespace NUMINAMATH_GPT_probability_same_color_correct_l541_54183

def number_of_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def probability_two_balls_same_color : ℚ :=
  ((green_balls / number_of_balls)^2 + (red_balls / number_of_balls)^2 + (blue_balls / number_of_balls)^2)

theorem probability_same_color_correct :
  probability_two_balls_same_color = 49 / 128 := sorry

end NUMINAMATH_GPT_probability_same_color_correct_l541_54183


namespace NUMINAMATH_GPT_remainder_x_squared_l541_54105

theorem remainder_x_squared (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  (x^2 ≡ 4 [ZMOD 20]) :=
sorry

end NUMINAMATH_GPT_remainder_x_squared_l541_54105


namespace NUMINAMATH_GPT_find_p_for_natural_roots_l541_54133

-- The polynomial is given.
def cubic_polynomial (p x : ℝ) : ℝ := 5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1

-- Problem statement to prove that p = 76 is the only real number such that
-- the cubic polynomial cubic_polynomial equals 66 * p has at least two natural number roots.
theorem find_p_for_natural_roots (p : ℝ) :
  (∃ (u v : ℕ), u ≠ v ∧ cubic_polynomial p u = 66 * p ∧ cubic_polynomial p v = 66 * p) ↔ p = 76 :=
by
  sorry

end NUMINAMATH_GPT_find_p_for_natural_roots_l541_54133


namespace NUMINAMATH_GPT_basket_can_hold_40_fruits_l541_54180

-- Let us define the number of oranges as 10
def oranges : ℕ := 10

-- There are 3 times as many apples as oranges
def apples : ℕ := 3 * oranges

-- The total number of fruits in the basket
def total_fruits : ℕ := oranges + apples

theorem basket_can_hold_40_fruits (h₁ : oranges = 10) (h₂ : apples = 3 * oranges) : total_fruits = 40 :=
by
  -- We assume the conditions and derive the conclusion
  sorry

end NUMINAMATH_GPT_basket_can_hold_40_fruits_l541_54180


namespace NUMINAMATH_GPT_weight_of_B_l541_54189

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) : B = 39 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_B_l541_54189


namespace NUMINAMATH_GPT_linear_function_quadrant_l541_54197

theorem linear_function_quadrant (x y : ℝ) (h : y = -3 * x + 2) :
  ¬ (x > 0 ∧ y > 0) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_quadrant_l541_54197


namespace NUMINAMATH_GPT_orange_is_faster_by_l541_54170

def forest_run_time (distance speed : ℕ) : ℕ := distance / speed
def beach_run_time (distance speed : ℕ) : ℕ := distance / speed
def mountain_run_time (distance speed : ℕ) : ℕ := distance / speed

def total_time_in_minutes (forest_distance forest_speed beach_distance beach_speed mountain_distance mountain_speed : ℕ) : ℕ :=
  (forest_run_time forest_distance forest_speed + beach_run_time beach_distance beach_speed + mountain_run_time mountain_distance mountain_speed) * 60

def apple_total_time := total_time_in_minutes 18 3 6 2 3 1
def mac_total_time := total_time_in_minutes 20 4 8 3 3 1
def orange_total_time := total_time_in_minutes 22 5 10 4 3 2

def combined_time := apple_total_time + mac_total_time
def orange_time_difference := combined_time - orange_total_time

theorem orange_is_faster_by :
  orange_time_difference = 856 := sorry

end NUMINAMATH_GPT_orange_is_faster_by_l541_54170


namespace NUMINAMATH_GPT_priya_trip_time_l541_54125

noncomputable def time_to_drive_from_X_to_Z_at_50_mph : ℝ := 5

theorem priya_trip_time :
  (∀ (distance_YZ distance_XZ : ℝ), 
    distance_YZ = 60 * 2.0833333333333335 ∧
    distance_XZ = distance_YZ * 2 →
    time_to_drive_from_X_to_Z_at_50_mph = distance_XZ / 50 ) :=
sorry

end NUMINAMATH_GPT_priya_trip_time_l541_54125


namespace NUMINAMATH_GPT_increasing_function_a_range_l541_54158

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4 * a * x else (2 * a + 3) * x - 4 * a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_a_range_l541_54158


namespace NUMINAMATH_GPT_sum_is_constant_l541_54187

variable (a b c d : ℚ) -- declare variables states as rational numbers

theorem sum_is_constant :
  (a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) →
  a + b + c + d = -(14 / 3) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_is_constant_l541_54187


namespace NUMINAMATH_GPT_unique_handshakes_l541_54129

theorem unique_handshakes :
  let twins_sets := 12
  let triplets_sets := 3
  let twins := twins_sets * 2
  let triplets := triplets_sets * 3
  let twin_shakes_twins := twins * (twins - 2)
  let triplet_shakes_triplets := triplets * (triplets - 3)
  let twin_shakes_triplets := twins * (triplets / 3)
  (twin_shakes_twins + triplet_shakes_triplets + twin_shakes_triplets) / 2 = 327 := by
  sorry

end NUMINAMATH_GPT_unique_handshakes_l541_54129


namespace NUMINAMATH_GPT_divisibility_ac_bd_l541_54156

-- Conditions definitions
variable (a b c d : ℕ)
variable (hab : a ∣ b)
variable (hcd : c ∣ d)

-- Goal
theorem divisibility_ac_bd : (a * c) ∣ (b * d) :=
  sorry

end NUMINAMATH_GPT_divisibility_ac_bd_l541_54156


namespace NUMINAMATH_GPT_problem_1_problem_2_l541_54139

noncomputable def f (x m : ℝ) := |x - 4 / m| + |x + m|

theorem problem_1 (m : ℝ) (hm : 0 < m) (x : ℝ) : f x m ≥ 4 := sorry

theorem problem_2 (m : ℝ) (hm : f 2 m > 5) : 
  m ∈ Set.Ioi ((1 + Real.sqrt 17) / 2) ∪ Set.Ioo 0 1 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l541_54139


namespace NUMINAMATH_GPT_div_by_3_pow_101_l541_54145

theorem div_by_3_pow_101 : ∀ (n : ℕ), (∀ k : ℕ, (3^(k+1)) ∣ (2^(3^k) + 1)) → 3^101 ∣ 2^(3^100) + 1 :=
by
  sorry

end NUMINAMATH_GPT_div_by_3_pow_101_l541_54145


namespace NUMINAMATH_GPT_inverse_h_l541_54138

def f (x : ℝ) : ℝ := 5 * x - 7
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h : (∀ x : ℝ, h (15 * x + 3) = x) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_inverse_h_l541_54138


namespace NUMINAMATH_GPT_score_stability_l541_54134

theorem score_stability (mean_A mean_B : ℝ) (h_mean_eq : mean_A = mean_B)
  (variance_A variance_B : ℝ) (h_variance_A : variance_A = 0.06) (h_variance_B : variance_B = 0.35) :
  variance_A < variance_B :=
by
  -- Theorem statement and conditions sufficient to build successfully
  sorry

end NUMINAMATH_GPT_score_stability_l541_54134


namespace NUMINAMATH_GPT_perpendicular_dot_product_zero_l541_54102

variables (a : ℝ)
def m := (a, 2)
def n := (1, 1 - a)

theorem perpendicular_dot_product_zero : (m a).1 * (n a).1 + (m a).2 * (n a).2 = 0 → a = 2 :=
by sorry

end NUMINAMATH_GPT_perpendicular_dot_product_zero_l541_54102


namespace NUMINAMATH_GPT_construction_company_sand_weight_l541_54135

theorem construction_company_sand_weight :
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  sand_weight = 8.11 :=
by
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  -- Observing that 14.02 - 5.91 = 8.11
  have h : sand_weight = 8.11 := by sorry
  exact h

end NUMINAMATH_GPT_construction_company_sand_weight_l541_54135


namespace NUMINAMATH_GPT_no_increasing_sequence_with_unique_sum_l541_54196

theorem no_increasing_sequence_with_unique_sum :
  ¬ (∃ (a : ℕ → ℕ), (∀ n, 0 < a n) ∧ (∀ n, a n < a (n + 1)) ∧ 
  (∀ N, ∃ k ≥ N, ∀ m ≥ k, 
    (∃! (i j : ℕ), a i + a j = m))) := sorry

end NUMINAMATH_GPT_no_increasing_sequence_with_unique_sum_l541_54196


namespace NUMINAMATH_GPT_twice_perimeter_is_72_l541_54186

def twice_perimeter_of_square_field (s : ℝ) : ℝ := 2 * 4 * s

theorem twice_perimeter_is_72 (a P : ℝ) (h1 : a = s^2) (h2 : P = 36) 
    (h3 : 6 * a = 6 * (2 * P + 9)) : twice_perimeter_of_square_field s = 72 := 
by
  sorry

end NUMINAMATH_GPT_twice_perimeter_is_72_l541_54186


namespace NUMINAMATH_GPT_donuts_distribution_l541_54146

theorem donuts_distribution (kinds total min_each : ℕ) (h_kinds : kinds = 4) (h_total : total = 7) (h_min_each : min_each = 1) :
  ∃ n : ℕ, n = 20 := by
  sorry

end NUMINAMATH_GPT_donuts_distribution_l541_54146


namespace NUMINAMATH_GPT_suitable_sampling_method_l541_54142

theorem suitable_sampling_method 
  (seniorTeachers : ℕ)
  (intermediateTeachers : ℕ)
  (juniorTeachers : ℕ)
  (totalSample : ℕ)
  (totalTeachers : ℕ)
  (prob : ℚ)
  (seniorSample : ℕ)
  (intermediateSample : ℕ)
  (juniorSample : ℕ)
  (excludeOneSenior : ℕ) :
  seniorTeachers = 28 →
  intermediateTeachers = 54 →
  juniorTeachers = 81 →
  totalSample = 36 →
  excludeOneSenior = 27 →
  totalTeachers = excludeOneSenior + intermediateTeachers + juniorTeachers →
  prob = totalSample / totalTeachers →
  seniorSample = excludeOneSenior * prob →
  intermediateSample = intermediateTeachers * prob →
  juniorSample = juniorTeachers * prob →
  seniorSample + intermediateSample + juniorSample = totalSample :=
by
  intros hsenior hins hjunior htotal hexclude htotalTeachers hprob hseniorSample hintermediateSample hjuniorSample
  sorry

end NUMINAMATH_GPT_suitable_sampling_method_l541_54142


namespace NUMINAMATH_GPT_arithmetic_sequence_divisible_by_2005_l541_54190

-- Problem Statement
theorem arithmetic_sequence_divisible_by_2005
  (a : ℕ → ℕ) -- Define the arithmetic sequence
  (d : ℕ) -- Common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence condition
  (h_product_div_2005 : ∀ n, 2005 ∣ (a n) * (a (n + 31))) -- Given condition on product divisibility
  : ∀ n, 2005 ∣ a n := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_divisible_by_2005_l541_54190


namespace NUMINAMATH_GPT_inequality_am_gm_l541_54176

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / c + c / b) ≥ (4 * a / (a + b)) ∧ (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by
  -- Proof steps
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l541_54176


namespace NUMINAMATH_GPT_stephanie_speed_l541_54131

noncomputable def distance : ℝ := 15
noncomputable def time : ℝ := 3

theorem stephanie_speed :
  distance / time = 5 := 
sorry

end NUMINAMATH_GPT_stephanie_speed_l541_54131


namespace NUMINAMATH_GPT_compute_expression_l541_54169

-- Definition of the expression
def expression := 5 + 4 * (4 - 9)^2

-- Statement of the theorem, asserting the expression equals 105
theorem compute_expression : expression = 105 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l541_54169


namespace NUMINAMATH_GPT_base_k_conversion_l541_54155

theorem base_k_conversion (k : ℕ) (hk : 4 * k + 4 = 36) : 6 * 8 + 7 = 55 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_base_k_conversion_l541_54155


namespace NUMINAMATH_GPT_solution_x_l541_54143

noncomputable def find_x (x : ℝ) : Prop :=
  (Real.log (x^4))^2 = (Real.log x)^6

theorem solution_x (x : ℝ) : find_x x ↔ (x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2)) :=
sorry

end NUMINAMATH_GPT_solution_x_l541_54143


namespace NUMINAMATH_GPT_nonnegative_integer_solutions_l541_54119

theorem nonnegative_integer_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_integer_solutions_l541_54119


namespace NUMINAMATH_GPT_range_of_a_l541_54124

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x - 12| < 6 → False) → (a ≤ 6 ∨ a ≥ 18) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l541_54124


namespace NUMINAMATH_GPT_John_profit_is_correct_l541_54177

-- Definitions of conditions as necessary in Lean
variable (initial_puppies : ℕ) (given_away_puppies : ℕ) (kept_puppy : ℕ) (price_per_puppy : ℤ) (payment_to_stud_owner : ℤ)

-- Specific values from the problem
def John_initial_puppies := 8
def John_given_away_puppies := 4
def John_kept_puppy := 1
def John_price_per_puppy := 600
def John_payment_to_stud_owner := 300

-- Calculate the number of puppies left to sell
def John_remaining_puppies := John_initial_puppies - John_given_away_puppies - John_kept_puppy

-- Calculate total earnings from selling puppies
def John_earnings := John_remaining_puppies * John_price_per_puppy

-- Calculate the profit by subtracting payment to the stud owner from earnings
def John_profit := John_earnings - John_payment_to_stud_owner

-- Statement to prove
theorem John_profit_is_correct : 
  John_profit = 1500 := 
by 
  -- The proof will be here but we use sorry to skip it as requested.
  sorry

-- This ensures the definitions match the given problem conditions
#eval (John_initial_puppies, John_given_away_puppies, John_kept_puppy, John_price_per_puppy, John_payment_to_stud_owner)

end NUMINAMATH_GPT_John_profit_is_correct_l541_54177


namespace NUMINAMATH_GPT_geometric_sequence_problem_l541_54121

variable (a : ℕ → ℝ)
variable (r : ℝ) (hpos : ∀ n, 0 < a n)

theorem geometric_sequence_problem
  (hgeom : ∀ n, a (n+1) = a n * r)
  (h_eq : a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) :
  a 2 + a 6 = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l541_54121


namespace NUMINAMATH_GPT_find_other_number_l541_54178

theorem find_other_number (x y : ℕ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) (h3 : x = 7) : y = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l541_54178


namespace NUMINAMATH_GPT_total_races_needed_to_determine_champion_l541_54128

-- Defining the initial conditions
def num_sprinters : ℕ := 256
def lanes : ℕ := 8
def sprinters_per_race := lanes
def eliminated_per_race := sprinters_per_race - 1

-- The statement to be proved: The number of races required to determine the champion
theorem total_races_needed_to_determine_champion :
  ∃ (races : ℕ), races = 37 ∧
  ∀ s : ℕ, s = num_sprinters → 
  ∀ l : ℕ, l = lanes → 
  ∃ e : ℕ, e = eliminated_per_race →
  s - (races * e) = 1 :=
by sorry

end NUMINAMATH_GPT_total_races_needed_to_determine_champion_l541_54128


namespace NUMINAMATH_GPT_count_success_permutations_l541_54182

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end NUMINAMATH_GPT_count_success_permutations_l541_54182


namespace NUMINAMATH_GPT_number_of_pictures_in_first_coloring_book_l541_54191

-- Define the conditions
variable (X : ℕ)
variable (total_pictures_colored : ℕ := 44)
variable (pictures_left : ℕ := 11)
variable (pictures_in_second_coloring_book : ℕ := 32)
variable (total_pictures : ℕ := total_pictures_colored + pictures_left)

-- The theorem statement
theorem number_of_pictures_in_first_coloring_book :
  X + pictures_in_second_coloring_book = total_pictures → X = 23 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_pictures_in_first_coloring_book_l541_54191


namespace NUMINAMATH_GPT_combined_marbles_l541_54172

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end NUMINAMATH_GPT_combined_marbles_l541_54172


namespace NUMINAMATH_GPT_trigonometric_inequality_l541_54199

theorem trigonometric_inequality (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 := 
sorry

end NUMINAMATH_GPT_trigonometric_inequality_l541_54199


namespace NUMINAMATH_GPT_neg_p_equiv_l541_54113

-- The proposition p
def p : Prop := ∀ x : ℝ, x^2 - 1 < 0

-- Equivalent Lean theorem statement
theorem neg_p_equiv : ¬ p ↔ ∃ x₀ : ℝ, x₀^2 - 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_p_equiv_l541_54113


namespace NUMINAMATH_GPT_determine_a_l541_54132

theorem determine_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x^3 - 2 * x) (pt : f (-1) = 4) : a = -2 := by
  sorry

end NUMINAMATH_GPT_determine_a_l541_54132


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l541_54157

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), n = 99 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ (Int.gcd a b) * a + b^2 = 10000
  → ∃ (k : ℕ), k = 99) :=
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l541_54157


namespace NUMINAMATH_GPT_circle_regions_division_l541_54195

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_circle_regions_division_l541_54195


namespace NUMINAMATH_GPT_turtle_finishes_in_10_minutes_l541_54104

def skunk_time : ℕ := 6
def rabbit_speed_ratio : ℕ := 3
def turtle_speed_ratio : ℕ := 5
def rabbit_time := skunk_time / rabbit_speed_ratio
def turtle_time := turtle_speed_ratio * rabbit_time

theorem turtle_finishes_in_10_minutes : turtle_time = 10 := by
  sorry

end NUMINAMATH_GPT_turtle_finishes_in_10_minutes_l541_54104


namespace NUMINAMATH_GPT_short_trees_after_planting_l541_54148

-- Define the current number of short trees
def current_short_trees : ℕ := 41

-- Define the number of short trees to be planted today
def new_short_trees : ℕ := 57

-- Define the expected total number of short trees after planting
def total_short_trees_after_planting : ℕ := 98

-- The theorem to prove that the total number of short trees after planting is as expected
theorem short_trees_after_planting :
  current_short_trees + new_short_trees = total_short_trees_after_planting :=
by
  -- Proof skipped using sorry
  sorry

end NUMINAMATH_GPT_short_trees_after_planting_l541_54148


namespace NUMINAMATH_GPT_solve_for_y_l541_54185

theorem solve_for_y (y : ℕ) (h1 : 40 = 2^3 * 5) (h2 : 8 = 2^3) :
  40^3 = 8^y ↔ y = 3 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l541_54185


namespace NUMINAMATH_GPT_sum_series_l541_54106

theorem sum_series :
  3 * (List.sum (List.map (λ n => n - 1) (List.range' 2 14))) = 273 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_l541_54106


namespace NUMINAMATH_GPT_average_minutes_run_per_day_l541_54114

variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def total_minutes_run := 14 * third_graders f + 18 * fourth_graders f + 8 * fifth_graders f
def total_students := third_graders f + fourth_graders f + fifth_graders f

theorem average_minutes_run_per_day : 
  (total_minutes_run f) / (total_students f) = 128 / 9 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_per_day_l541_54114


namespace NUMINAMATH_GPT_quadratic_root_a_value_l541_54149

theorem quadratic_root_a_value (a : ℝ) (h : 2^2 - 2 * a + 6 = 0) : a = 5 :=
sorry

end NUMINAMATH_GPT_quadratic_root_a_value_l541_54149


namespace NUMINAMATH_GPT_kiki_total_money_l541_54108

theorem kiki_total_money 
  (S : ℕ) (H : ℕ) (M : ℝ)
  (h1: S = 18)
  (h2: H = 2 * S)
  (h3: 0.40 * M = 36) : 
  M = 90 :=
by
  sorry

end NUMINAMATH_GPT_kiki_total_money_l541_54108


namespace NUMINAMATH_GPT_mixture_weight_l541_54141

def almonds := 116.67
def walnuts := almonds / 5
def total_weight := almonds + walnuts

theorem mixture_weight : total_weight = 140.004 := by
  sorry

end NUMINAMATH_GPT_mixture_weight_l541_54141


namespace NUMINAMATH_GPT_ant_prob_bottom_vertex_l541_54111

theorem ant_prob_bottom_vertex :
  let top := 1
  let first_layer := 4
  let second_layer := 4
  let bottom := 1
  let prob_first_layer := 1 / first_layer
  let prob_second_layer := 1 / second_layer
  let prob_bottom := 1 / (second_layer + bottom)
  prob_first_layer * prob_second_layer * prob_bottom = 1 / 80 :=
by
  sorry

end NUMINAMATH_GPT_ant_prob_bottom_vertex_l541_54111


namespace NUMINAMATH_GPT_cos6_plus_sin6_equal_19_div_64_l541_54126

noncomputable def cos6_plus_sin6 (θ : ℝ) : ℝ :=
  (Real.cos θ) ^ 6 + (Real.sin θ) ^ 6

theorem cos6_plus_sin6_equal_19_div_64 (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) :
  cos6_plus_sin6 θ = 19 / 64 := by
  sorry

end NUMINAMATH_GPT_cos6_plus_sin6_equal_19_div_64_l541_54126


namespace NUMINAMATH_GPT_carla_final_payment_l541_54161

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end NUMINAMATH_GPT_carla_final_payment_l541_54161


namespace NUMINAMATH_GPT_xiaoming_climb_stairs_five_steps_l541_54122

def count_ways_to_climb (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else count_ways_to_climb (n - 1) + count_ways_to_climb (n - 2)

theorem xiaoming_climb_stairs_five_steps :
  count_ways_to_climb 5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_xiaoming_climb_stairs_five_steps_l541_54122


namespace NUMINAMATH_GPT_calories_burned_l541_54117

theorem calories_burned {running_minutes walking_minutes total_minutes calories_per_minute_running calories_per_minute_walking calories_total : ℕ}
    (h_run : running_minutes = 35)
    (h_total : total_minutes = 60)
    (h_calories_run : calories_per_minute_running = 10)
    (h_calories_walk : calories_per_minute_walking = 4)
    (h_walk : walking_minutes = total_minutes - running_minutes)
    (h_calories_total : calories_total = running_minutes * calories_per_minute_running + walking_minutes * calories_per_minute_walking) : 
    calories_total = 450 := by
  sorry

end NUMINAMATH_GPT_calories_burned_l541_54117


namespace NUMINAMATH_GPT_negation_of_every_function_has_parity_l541_54171

-- Assume the initial proposition
def every_function_has_parity := ∀ f : ℕ → ℕ, ∃ (p : ℕ), p = 0 ∨ p = 1

-- Negation of the original proposition
def exists_function_without_parity := ∃ f : ℕ → ℕ, ∀ p : ℕ, p ≠ 0 ∧ p ≠ 1

-- The theorem to prove
theorem negation_of_every_function_has_parity : 
  ¬ every_function_has_parity ↔ exists_function_without_parity := 
by
  unfold every_function_has_parity exists_function_without_parity
  sorry

end NUMINAMATH_GPT_negation_of_every_function_has_parity_l541_54171


namespace NUMINAMATH_GPT_find_C_plus_D_l541_54166

theorem find_C_plus_D
  (C D : ℕ)
  (h1 : D = C + 2)
  (h2 : 2 * C^2 + 5 * C + 3 - (7 * D + 5) = (C + D)^2 + 6 * (C + D) + 8)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D) :
  C + D = 26 := by
  sorry

end NUMINAMATH_GPT_find_C_plus_D_l541_54166


namespace NUMINAMATH_GPT_motorcycle_materials_cost_l541_54193

theorem motorcycle_materials_cost 
  (car_material_cost : ℕ) (cars_per_month : ℕ) (car_sale_price : ℕ)
  (motorcycles_per_month : ℕ) (motorcycle_sale_price : ℕ)
  (additional_profit : ℕ) :
  car_material_cost = 100 →
  cars_per_month = 4 →
  car_sale_price = 50 →
  motorcycles_per_month = 8 →
  motorcycle_sale_price = 50 →
  additional_profit = 50 →
  car_material_cost + additional_profit = 250 := by
  sorry

end NUMINAMATH_GPT_motorcycle_materials_cost_l541_54193


namespace NUMINAMATH_GPT_smallest_B_for_divisibility_by_4_l541_54123

theorem smallest_B_for_divisibility_by_4 : 
  ∃ (B : ℕ), B < 10 ∧ (4 * 1000000 + B * 100000 + 80000 + 3961) % 4 = 0 ∧ ∀ (B' : ℕ), (B' < B ∧ B' < 10) → ¬ ((4 * 1000000 + B' * 100000 + 80000 + 3961) % 4 = 0) := 
sorry

end NUMINAMATH_GPT_smallest_B_for_divisibility_by_4_l541_54123


namespace NUMINAMATH_GPT_value_of_A_l541_54188

theorem value_of_A (M A T E H : ℤ) (hH : H = 8) (h1 : M + A + T + H = 31) (h2 : T + E + A + M = 40) (h3 : M + E + E + T = 44) (h4 : M + A + T + E = 39) : A = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_A_l541_54188


namespace NUMINAMATH_GPT_find_k_exact_one_real_solution_l541_54144

theorem find_k_exact_one_real_solution (k : ℝ) :
  (∀ x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := 
by
  sorry

end NUMINAMATH_GPT_find_k_exact_one_real_solution_l541_54144


namespace NUMINAMATH_GPT_number_of_parents_l541_54152

theorem number_of_parents (n m : ℕ) 
  (h1 : n + m = 31) 
  (h2 : 15 + m = n) 
  : n = 23 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_parents_l541_54152


namespace NUMINAMATH_GPT_candy_ratio_l541_54127

theorem candy_ratio
  (red_candies : ℕ)
  (yellow_candies : ℕ)
  (blue_candies : ℕ)
  (total_candies : ℕ)
  (remaining_candies : ℕ)
  (h1 : red_candies = 40)
  (h2 : yellow_candies = 3 * red_candies - 20)
  (h3 : remaining_candies = 90)
  (h4 : total_candies = remaining_candies + yellow_candies)
  (h5 : blue_candies = total_candies - red_candies - yellow_candies) :
  blue_candies / yellow_candies = 1 / 2 :=
sorry

end NUMINAMATH_GPT_candy_ratio_l541_54127


namespace NUMINAMATH_GPT_final_score_eq_l541_54160

variable (initial_score : ℝ)
def deduction_lost_answer : ℝ := 1
def deduction_error : ℝ := 0.5
def deduction_checks : ℝ := 0

def total_deduction : ℝ := deduction_lost_answer + deduction_error + deduction_checks

theorem final_score_eq : final_score = initial_score - total_deduction := by
  sorry

end NUMINAMATH_GPT_final_score_eq_l541_54160


namespace NUMINAMATH_GPT_range_of_m_l541_54175

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y - x * y = 0) :
    (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y - x * y = 0 → x + 2 * y > m^2 + 2 * m) ↔ (-4 : ℝ) < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l541_54175


namespace NUMINAMATH_GPT_gcd_polynomials_l541_54137

theorem gcd_polynomials (b : ℕ) (hb : 2160 ∣ b) : 
  Nat.gcd (b ^ 2 + 9 * b + 30) (b + 6) = 12 := 
  sorry

end NUMINAMATH_GPT_gcd_polynomials_l541_54137


namespace NUMINAMATH_GPT_min_flash_drives_needed_l541_54194

theorem min_flash_drives_needed (total_files : ℕ) (capacity_per_drive : ℝ)  
  (num_files_0_9 : ℕ) (size_0_9 : ℝ) 
  (num_files_0_8 : ℕ) (size_0_8 : ℝ) 
  (size_0_6 : ℝ) 
  (remaining_files : ℕ) :
  total_files = 40 →
  capacity_per_drive = 2.88 →
  num_files_0_9 = 5 →
  size_0_9 = 0.9 →
  num_files_0_8 = 18 →
  size_0_8 = 0.8 →
  remaining_files = total_files - (num_files_0_9 + num_files_0_8) →
  size_0_6 = 0.6 →
  (num_files_0_9 * size_0_9 + num_files_0_8 * size_0_8 + remaining_files * size_0_6) / capacity_per_drive ≤ 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_flash_drives_needed_l541_54194


namespace NUMINAMATH_GPT_first_group_work_done_l541_54181

-- Define work amounts with the conditions given
variable (W : ℕ) -- amount of work 3 people can do in 3 days
variable (work_rate : ℕ → ℕ → ℕ) -- work_rate(p, d) is work done by p people in d days

-- Conditions
axiom cond1 : work_rate 3 3 = W
axiom cond2 : work_rate 6 3 = 6 * W

-- The proof statement
theorem first_group_work_done : work_rate 3 3 = 2 * W :=
by
  sorry

end NUMINAMATH_GPT_first_group_work_done_l541_54181


namespace NUMINAMATH_GPT_seed_grow_prob_l541_54163

theorem seed_grow_prob (P_G P_S_given_G : ℝ) (hP_G : P_G = 0.9) (hP_S_given_G : P_S_given_G = 0.8) :
  P_G * P_S_given_G = 0.72 :=
by
  rw [hP_G, hP_S_given_G]
  norm_num

end NUMINAMATH_GPT_seed_grow_prob_l541_54163


namespace NUMINAMATH_GPT_total_boys_school_l541_54109

variable (B : ℕ)
variables (percMuslim percHindu percSikh boysOther : ℕ)

-- Defining the conditions
def condition1 : percMuslim = 44 := by sorry
def condition2 : percHindu = 28 := by sorry
def condition3 : percSikh = 10 := by sorry
def condition4 : boysOther = 54 := by sorry

-- Main theorem statement
theorem total_boys_school (h1 : percMuslim = 44) (h2 : percHindu = 28) (h3 : percSikh = 10) (h4 : boysOther = 54) : 
  B = 300 := by sorry

end NUMINAMATH_GPT_total_boys_school_l541_54109


namespace NUMINAMATH_GPT_largest_k_l541_54151

def S : Set ℕ := {x | x > 0 ∧ x ≤ 100}

def satisfies_property (A B : Set ℕ) : Prop :=
  ∃ x ∈ A ∩ B, ∀ y ∈ A ∪ B, x ≠ y

theorem largest_k (k : ℕ) : 
  (∃ subsets : Finset (Set ℕ), 
    (subsets.card = k) ∧ 
    (∀ {A B : Set ℕ}, A ∈ subsets ∧ B ∈ subsets ∧ A ≠ B → 
      ¬(A ∩ B = ∅) ∧ satisfies_property A B)) →
  k ≤ 2^99 - 1 := sorry

end NUMINAMATH_GPT_largest_k_l541_54151
