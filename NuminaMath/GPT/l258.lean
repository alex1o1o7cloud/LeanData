import Mathlib

namespace NUMINAMATH_GPT_xy_y_sq_eq_y_sq_3y_12_l258_25867

variable (x y : ℝ)

theorem xy_y_sq_eq_y_sq_3y_12 (h : x * (x + y) = x^2 + 3 * y + 12) : 
  x * y + y^2 = y^2 + 3 * y + 12 := 
sorry

end NUMINAMATH_GPT_xy_y_sq_eq_y_sq_3y_12_l258_25867


namespace NUMINAMATH_GPT_find_f_of_3_l258_25802

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x^2 - 2 * x) : f 3 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_find_f_of_3_l258_25802


namespace NUMINAMATH_GPT_Suzanna_bike_distance_l258_25891

theorem Suzanna_bike_distance (ride_rate distance_time total_time : ℕ)
  (constant_rate : ride_rate = 3) (time_interval : distance_time = 10)
  (total_riding_time : total_time = 40) :
  (total_time / distance_time) * ride_rate = 12 :=
by
  -- Assuming the conditions:
  -- ride_rate = 3
  -- distance_time = 10
  -- total_time = 40
  sorry

end NUMINAMATH_GPT_Suzanna_bike_distance_l258_25891


namespace NUMINAMATH_GPT_point_inside_circle_range_of_a_l258_25874

/- 
  Define the circle and the point P. 
  We would show that ensuring the point lies inside the circle implies |a| < 1/13.
-/

theorem point_inside_circle_range_of_a (a : ℝ) : 
  ((5 * a + 1 - 1) ^ 2 + (12 * a) ^ 2 < 1) -> |a| < 1 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_point_inside_circle_range_of_a_l258_25874


namespace NUMINAMATH_GPT_age_problem_l258_25828

theorem age_problem 
  (P R J M : ℕ)
  (h1 : P = 1 / 2 * R)
  (h2 : R = J + 7)
  (h3 : J + 12 = 3 * P)
  (h4 : M = J + 17)
  (h5 : M = 2 * R + 4) : 
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 :=
by sorry

end NUMINAMATH_GPT_age_problem_l258_25828


namespace NUMINAMATH_GPT_baron_munchausen_not_lying_l258_25879

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end NUMINAMATH_GPT_baron_munchausen_not_lying_l258_25879


namespace NUMINAMATH_GPT_math_problem_l258_25827

theorem math_problem (a b : ℝ) :
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l258_25827


namespace NUMINAMATH_GPT_find_x_y_l258_25866

theorem find_x_y (a n x y : ℕ) (hx4 : 1000 ≤ x ∧ x < 10000) (hy4 : 1000 ≤ y ∧ y < 10000) 
  (h_yx : y > x) (h_y : y = a * 10 ^ n) 
  (h_sum : (x / 1000) + ((x % 1000) / 100) = 5 * a) 
  (ha : a = 2) (hn : n = 3) :
  x = 1990 ∧ y = 2000 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_y_l258_25866


namespace NUMINAMATH_GPT_bird_wings_l258_25801

theorem bird_wings (birds wings_per_bird : ℕ) (h1 : birds = 13) (h2 : wings_per_bird = 2) : birds * wings_per_bird = 26 := by
  sorry

end NUMINAMATH_GPT_bird_wings_l258_25801


namespace NUMINAMATH_GPT_avg_page_count_per_essay_l258_25876

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end NUMINAMATH_GPT_avg_page_count_per_essay_l258_25876


namespace NUMINAMATH_GPT_plus_signs_count_l258_25877

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end NUMINAMATH_GPT_plus_signs_count_l258_25877


namespace NUMINAMATH_GPT_total_turtles_in_lake_l258_25843

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end NUMINAMATH_GPT_total_turtles_in_lake_l258_25843


namespace NUMINAMATH_GPT_range_of_a_l258_25855

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (if x1 ≤ 1 then (-x1^2 + a*x1)
     else (a*x1 - 1)) = 
    (if x2 ≤ 1 then (-x2^2 + a*x2)
     else (a*x2 - 1))) → a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l258_25855


namespace NUMINAMATH_GPT_zero_point_exists_in_interval_l258_25822

noncomputable def f (x : ℝ) : ℝ := x + 2^x

theorem zero_point_exists_in_interval :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f x = 0 :=
by
  existsi -0.5 -- This is not a formal proof; the existi -0.5 is just for example purposes
  sorry

end NUMINAMATH_GPT_zero_point_exists_in_interval_l258_25822


namespace NUMINAMATH_GPT_no_such_prime_pair_l258_25807

open Prime

theorem no_such_prime_pair :
  ∀ (p q : ℕ), Prime p → Prime q → (p > 5) → (q > 5) →
  (p * q) ∣ ((5^p - 2^p) * (5^q - 2^q)) → false :=
by
  intros p q hp hq hp_gt5 hq_gt5 hdiv
  sorry

end NUMINAMATH_GPT_no_such_prime_pair_l258_25807


namespace NUMINAMATH_GPT_find_principal_amount_l258_25880

variables (P R : ℝ)

theorem find_principal_amount (h : (4 * P * (R + 2) / 100) - (4 * P * R / 100) = 56) : P = 700 :=
sorry

end NUMINAMATH_GPT_find_principal_amount_l258_25880


namespace NUMINAMATH_GPT_repeating_decimal_sum_l258_25878

theorem repeating_decimal_sum :
  (0.3333333333 : ℚ) + (0.0404040404 : ℚ) + (0.005005005 : ℚ) + (0.000600060006 : ℚ) = 3793 / 9999 := by
sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l258_25878


namespace NUMINAMATH_GPT_ronald_next_roll_l258_25872

/-- Ronald's rolls -/
def rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

/-- Total number of rolls after the next roll -/
def total_rolls := rolls.length + 1

/-- The desired average of the rolls -/
def desired_average : ℕ := 3

/-- The sum Ronald needs to reach after the next roll to achieve the desired average -/
def required_sum : ℕ := desired_average * total_rolls

/-- Ronald's current sum of rolls -/
def current_sum : ℕ := List.sum rolls

/-- The next roll needed to achieve the desired average -/
def next_roll_needed : ℕ := required_sum - current_sum

theorem ronald_next_roll :
  next_roll_needed = 2 := by
  sorry

end NUMINAMATH_GPT_ronald_next_roll_l258_25872


namespace NUMINAMATH_GPT_dice_probability_l258_25842

-- The context that there are three six-sided dice
def total_outcomes : ℕ := 6 * 6 * 6

-- Function to count the number of favorable outcomes where two dice sum to the third
def favorable_outcomes : ℕ :=
  let sum_cases := [1, 2, 3, 4, 5]
  sum_cases.sum
  -- sum_cases is [1, 2, 3, 4, 5] each mapping to the number of ways to form that sum with a third die

theorem dice_probability : 
  (favorable_outcomes * 3) / total_outcomes = 5 / 24 := 
by 
  -- to prove: the probability that the values on two dice sum to the value on the remaining die is 5/24
  sorry

end NUMINAMATH_GPT_dice_probability_l258_25842


namespace NUMINAMATH_GPT_palm_trees_in_forest_l258_25870

variable (F D : ℕ)

theorem palm_trees_in_forest 
  (h1 : D = 2 * F / 5)
  (h2 : D + F = 7000) :
  F = 5000 := by
  sorry

end NUMINAMATH_GPT_palm_trees_in_forest_l258_25870


namespace NUMINAMATH_GPT_solve_for_x_l258_25892

theorem solve_for_x : 
  ∃ x₁ x₂ : ℝ, abs (x₁ - 0.175) < 1e-3 ∧ abs (x₂ - 18.325) < 1e-3 ∧
    (∀ x : ℝ, (8 * x ^ 2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 → x = x₁ ∨ x = x₂) := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l258_25892


namespace NUMINAMATH_GPT_second_to_last_digit_of_n_squared_plus_2n_l258_25800
open Nat

theorem second_to_last_digit_of_n_squared_plus_2n (n : ℕ) (h : (n^2 + 2 * n) % 10 = 4) : ((n^2 + 2 * n) / 10) % 10 = 2 :=
  sorry

end NUMINAMATH_GPT_second_to_last_digit_of_n_squared_plus_2n_l258_25800


namespace NUMINAMATH_GPT_cubics_identity_l258_25819

variable (a b c x y z : ℝ)

theorem cubics_identity (X Y Z : ℝ)
  (h1 : X = a * x + b * y + c * z)
  (h2 : Y = a * y + b * z + c * x)
  (h3 : Z = a * z + b * x + c * y) :
  X^3 + Y^3 + Z^3 - 3 * X * Y * Z = 
  (x^3 + y^3 + z^3 - 3 * x * y * z) * (a^3 + b^3 + c^3 - 3 * a * b * c) :=
sorry

end NUMINAMATH_GPT_cubics_identity_l258_25819


namespace NUMINAMATH_GPT_number_of_n_values_l258_25829

-- Definition of sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

-- The main statement to prove
theorem number_of_n_values : 
  ∃ M, M = 8 ∧ ∀ n : ℕ, (n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2010) → M = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_n_values_l258_25829


namespace NUMINAMATH_GPT_sum_of_roots_eq_zero_l258_25821

theorem sum_of_roots_eq_zero :
  ∀ (x : ℝ), x^2 - 7 * |x| + 6 = 0 → (∃ a b c d : ℝ, a + b + c + d = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_zero_l258_25821


namespace NUMINAMATH_GPT_percent_other_birds_is_31_l258_25813

noncomputable def initial_hawk_percentage : ℝ := 0.30
noncomputable def initial_paddyfield_warbler_percentage : ℝ := 0.25
noncomputable def initial_kingfisher_percentage : ℝ := 0.10
noncomputable def initial_hp_k_total : ℝ := initial_hawk_percentage + initial_paddyfield_warbler_percentage + initial_kingfisher_percentage

noncomputable def migrated_hawk_percentage : ℝ := 0.8 * initial_hawk_percentage
noncomputable def migrated_kingfisher_percentage : ℝ := 2 * initial_kingfisher_percentage
noncomputable def migrated_hp_k_total : ℝ := migrated_hawk_percentage + initial_paddyfield_warbler_percentage + migrated_kingfisher_percentage

noncomputable def other_birds_percentage : ℝ := 1 - migrated_hp_k_total

theorem percent_other_birds_is_31 : other_birds_percentage = 0.31 := sorry

end NUMINAMATH_GPT_percent_other_birds_is_31_l258_25813


namespace NUMINAMATH_GPT_palindromic_condition_l258_25825

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem palindromic_condition (m n : ℕ) :
  is_palindrome (2^n + 2^m + 1) ↔ (m ≤ 9 ∨ n ≤ 9) :=
sorry

end NUMINAMATH_GPT_palindromic_condition_l258_25825


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l258_25852

theorem tangent_line_to_parabola (l : ℝ → ℝ) (y : ℝ) (x : ℝ)
  (passes_through_P : l (-2) = 0)
  (intersects_once : ∃! x, (l x)^2 = 8*x) :
  (l = fun x => 0) ∨ (l = fun x => x + 2) ∨ (l = fun x => -x - 2) :=
sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l258_25852


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l258_25836

noncomputable def repeating_decimal_solution : ℚ := 7311 / 999

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 7 + 318 / 999) : x = repeating_decimal_solution := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l258_25836


namespace NUMINAMATH_GPT_intercepted_segments_length_l258_25837

theorem intercepted_segments_length {a b c x : ℝ} 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : x = a * b * c / (a * b + b * c + c * a)) : 
  x = a * b * c / (a * b + b * c + c * a) :=
by sorry

end NUMINAMATH_GPT_intercepted_segments_length_l258_25837


namespace NUMINAMATH_GPT_aaron_guesses_correctly_l258_25864

noncomputable def P_H : ℝ := 2 / 3
noncomputable def P_T : ℝ := 1 / 3
noncomputable def P_G_H : ℝ := 2 / 3
noncomputable def P_G_T : ℝ := 1 / 3

noncomputable def p : ℝ := P_H * P_G_H + P_T * P_G_T

theorem aaron_guesses_correctly :
  9000 * p = 5000 :=
by
  sorry

end NUMINAMATH_GPT_aaron_guesses_correctly_l258_25864


namespace NUMINAMATH_GPT_expr_is_irreducible_fraction_l258_25854

def a : ℚ := 3 / 2015
def b : ℚ := 11 / 2016

noncomputable def expr : ℚ := 
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a

theorem expr_is_irreducible_fraction : expr = 11 / 112 := by
  sorry

end NUMINAMATH_GPT_expr_is_irreducible_fraction_l258_25854


namespace NUMINAMATH_GPT_product_of_areas_square_of_volume_l258_25806

-- Declare the original dimensions and volume
variables (a b c : ℝ)
def V := a * b * c

-- Declare the areas of the new box
def area_bottom := (a + 2) * (b + 2)
def area_side := (b + 2) * (c + 2)
def area_front := (c + 2) * (a + 2)

-- Final theorem to prove
theorem product_of_areas_square_of_volume :
  (area_bottom a b) * (area_side b c) * (area_front c a) = V a b c ^ 2 :=
sorry

end NUMINAMATH_GPT_product_of_areas_square_of_volume_l258_25806


namespace NUMINAMATH_GPT_pencils_cost_l258_25845

theorem pencils_cost (A B : ℕ) (C D : ℕ) (r : ℚ) : 
    A * 20 = 3200 → B * 20 = 960 → (A / B = 3200 / 960) → (A = 160) → (B = 48) → (C = 3200) → (D = 960) → 160 * 960 / 48 = 3200 :=
by
sorry

end NUMINAMATH_GPT_pencils_cost_l258_25845


namespace NUMINAMATH_GPT_part1_part2_l258_25896

theorem part1 (a : ℝ) (x : ℝ) (h : a ≠ 0) :
    (|x - a| + |x + a + (1 / a)|) ≥ 2 * Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) (h : a ≠ 0) (h₁ : |2 - a| + |2 + a + 1 / a| ≤ 3) :
    a ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ico (1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l258_25896


namespace NUMINAMATH_GPT_circumference_of_flower_bed_l258_25847

noncomputable def square_garden_circumference (a p s r C : ℝ) : Prop :=
  a = s^2 ∧
  p = 4 * s ∧
  a = 2 * p + 14.25 ∧
  r = s / 4 ∧
  C = 2 * Real.pi * r

theorem circumference_of_flower_bed (a p s r : ℝ) (h : square_garden_circumference a p s r (4.75 * Real.pi)) : 
  ∃ C, square_garden_circumference a p s r C ∧ C = 4.75 * Real.pi :=
sorry

end NUMINAMATH_GPT_circumference_of_flower_bed_l258_25847


namespace NUMINAMATH_GPT_cubic_trinomial_degree_l258_25814

theorem cubic_trinomial_degree (n : ℕ) (P : ℕ → ℕ →  ℕ → Prop) : 
  (P n 5 4) → n = 3 := 
  sorry

end NUMINAMATH_GPT_cubic_trinomial_degree_l258_25814


namespace NUMINAMATH_GPT_maria_payment_l258_25848

noncomputable def calculate_payment : ℝ :=
  let regular_price := 15
  let first_discount := 0.40 * regular_price
  let after_first_discount := regular_price - first_discount
  let holiday_discount := 0.10 * after_first_discount
  let after_holiday_discount := after_first_discount - holiday_discount
  after_holiday_discount + 2

theorem maria_payment : calculate_payment = 10.10 :=
by
  sorry

end NUMINAMATH_GPT_maria_payment_l258_25848


namespace NUMINAMATH_GPT_garden_area_l258_25893

theorem garden_area (P b l: ℕ) (hP: P = 900) (hb: b = 190) (hl: l = P / 2 - b):
  l * b = 49400 := 
by
  sorry

end NUMINAMATH_GPT_garden_area_l258_25893


namespace NUMINAMATH_GPT_solve_picnic_problem_l258_25840

def picnic_problem : Prop :=
  ∃ (M W A C : ℕ), 
    M = W + 80 ∧ 
    A = C + 80 ∧ 
    M + W = A ∧ 
    A + C = 240 ∧ 
    M = 120

theorem solve_picnic_problem : picnic_problem :=
  sorry

end NUMINAMATH_GPT_solve_picnic_problem_l258_25840


namespace NUMINAMATH_GPT_product_expression_l258_25883

theorem product_expression :
  (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) * (7^4 - 1) / (7^4 + 1) = 880 / 91 := by
sorry

end NUMINAMATH_GPT_product_expression_l258_25883


namespace NUMINAMATH_GPT_least_value_expression_l258_25856

theorem least_value_expression (x y : ℝ) : 
  (x^2 * y + x * y^2 - 1)^2 + (x + y)^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_least_value_expression_l258_25856


namespace NUMINAMATH_GPT_evaluate_polynomial_at_2_l258_25862

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + x^2 + 2 * x + 3

theorem evaluate_polynomial_at_2 : polynomial 2 = 67 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_2_l258_25862


namespace NUMINAMATH_GPT_son_work_time_l258_25858

theorem son_work_time :
  let M := (1 : ℚ) / 7
  let combined_rate := (1 : ℚ) / 3
  let S := combined_rate - M
  1 / S = 5.25 :=  
by
  sorry

end NUMINAMATH_GPT_son_work_time_l258_25858


namespace NUMINAMATH_GPT_average_speed_x_to_z_l258_25844

theorem average_speed_x_to_z 
  (d : ℝ)
  (h1 : d > 0)
  (distance_xy : ℝ := 2 * d)
  (distance_yz : ℝ := d)
  (speed_xy : ℝ := 100)
  (speed_yz : ℝ := 75)
  (total_distance : ℝ := distance_xy + distance_yz)
  (time_xy : ℝ := distance_xy / speed_xy)
  (time_yz : ℝ := distance_yz / speed_yz)
  (total_time : ℝ := time_xy + time_yz) :
  total_distance / total_time = 90 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_x_to_z_l258_25844


namespace NUMINAMATH_GPT_theta_in_second_quadrant_l258_25885

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : Real.sin θ > 0)
  (h2 : Real.tan θ < 0) :
  (π / 2 < θ) ∧ (θ < π) :=
by
  sorry

end NUMINAMATH_GPT_theta_in_second_quadrant_l258_25885


namespace NUMINAMATH_GPT_train_stops_for_10_minutes_per_hour_l258_25857

-- Define the conditions
def speed_excluding_stoppages : ℕ := 48 -- in kmph
def speed_including_stoppages : ℕ := 40 -- in kmph

-- Define the question as proving the train stops for 10 minutes per hour
theorem train_stops_for_10_minutes_per_hour :
  (speed_excluding_stoppages - speed_including_stoppages) * 60 / speed_excluding_stoppages = 10 :=
by
  sorry

end NUMINAMATH_GPT_train_stops_for_10_minutes_per_hour_l258_25857


namespace NUMINAMATH_GPT_trig_evaluation_trig_identity_value_l258_25868

-- Problem 1: Prove the trigonometric evaluation
theorem trig_evaluation :
  (Real.cos (9 * Real.pi / 4)) + (Real.tan (-Real.pi / 4)) + (Real.sin (21 * Real.pi)) = (Real.sqrt 2 / 2) - 1 :=
by
  sorry

-- Problem 2: Prove the value given the trigonometric identity
theorem trig_identity_value (θ : ℝ) (h : Real.sin θ = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_trig_evaluation_trig_identity_value_l258_25868


namespace NUMINAMATH_GPT_geq_solution_l258_25832

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a (n+1) / a n) = (a 1 / a 0)

theorem geq_solution
  (a : ℕ → ℝ)
  (h_seq : geom_seq a)
  (h_cond : a 0 * a 2 + 2 * a 1 * a 3 + a 1 * a 5 = 9) :
  a 1 + a 3 = 3 :=
sorry

end NUMINAMATH_GPT_geq_solution_l258_25832


namespace NUMINAMATH_GPT_sandwiches_prepared_l258_25890

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end NUMINAMATH_GPT_sandwiches_prepared_l258_25890


namespace NUMINAMATH_GPT_find_M_l258_25875

theorem find_M : ∀ M : ℕ, (10 + 11 + 12 : ℕ) / 3 = (2024 + 2025 + 2026 : ℕ) / M → M = 552 :=
by
  intro M
  sorry

end NUMINAMATH_GPT_find_M_l258_25875


namespace NUMINAMATH_GPT_second_character_more_lines_l258_25894

theorem second_character_more_lines
  (C1 : ℕ) (S : ℕ) (T : ℕ) (X : ℕ)
  (h1 : C1 = 20)
  (h2 : C1 = S + 8)
  (h3 : T = 2)
  (h4 : S = 3 * T + X) :
  X = 6 :=
by
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_second_character_more_lines_l258_25894


namespace NUMINAMATH_GPT_planes_parallel_if_perpendicular_to_same_line_l258_25865

variables {Point : Type} {Line : Type} {Plane : Type} 

-- Definitions and conditions
noncomputable def is_parallel (α β : Plane) : Prop := sorry
noncomputable def is_perpendicular (l : Line) (α : Plane) : Prop := sorry

variables (l1 : Line) (α β : Plane)

theorem planes_parallel_if_perpendicular_to_same_line
  (h1 : is_perpendicular l1 α)
  (h2 : is_perpendicular l1 β) : is_parallel α β := 
sorry

end NUMINAMATH_GPT_planes_parallel_if_perpendicular_to_same_line_l258_25865


namespace NUMINAMATH_GPT_find_f_l258_25860

theorem find_f 
  (h_vertex : ∃ (d e f : ℝ), ∀ x, y = d * (x - 3)^2 - 5 ∧ y = d * x^2 + e * x + f)
  (h_point : y = d * (4 - 3)^2 - 5) 
  (h_value : y = -3) :
  ∃ f, f = 13 :=
sorry

end NUMINAMATH_GPT_find_f_l258_25860


namespace NUMINAMATH_GPT_power_of_square_l258_25809

variable {R : Type*} [CommRing R] (a : R)

theorem power_of_square (a : R) : (3 * a^2)^2 = 9 * a^4 :=
by sorry

end NUMINAMATH_GPT_power_of_square_l258_25809


namespace NUMINAMATH_GPT_triangle_height_l258_25834

theorem triangle_height (base : ℝ) (height : ℝ) (area : ℝ)
  (h_base : base = 8) (h_area : area = 16) (h_area_formula : area = (base * height) / 2) :
  height = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l258_25834


namespace NUMINAMATH_GPT_fraction_of_girls_correct_l258_25826

-- Define the total number of students in each school
def total_greenwood : ℕ := 300
def total_maplewood : ℕ := 240

-- Define the ratios of boys to girls
def ratio_boys_girls_greenwood := (3, 2)
def ratio_boys_girls_maplewood := (3, 4)

-- Define the number of boys and girls at Greenwood Middle School
def boys_greenwood (x : ℕ) : ℕ := 3 * x
def girls_greenwood (x : ℕ) : ℕ := 2 * x

-- Define the number of boys and girls at Maplewood Middle School
def boys_maplewood (y : ℕ) : ℕ := 3 * y
def girls_maplewood (y : ℕ) : ℕ := 4 * y

-- Define the total fractions
def total_girls (x y : ℕ) : ℚ := (girls_greenwood x + girls_maplewood y)
def total_students : ℚ := (total_greenwood + total_maplewood)

-- Main theorem to prove the fraction of girls at the event
theorem fraction_of_girls_correct (x y : ℕ)
  (h1 : 5 * x = total_greenwood)
  (h2 : 7 * y = total_maplewood) :
  (total_girls x y) / total_students = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_girls_correct_l258_25826


namespace NUMINAMATH_GPT_soccer_league_fraction_female_proof_l258_25897

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end NUMINAMATH_GPT_soccer_league_fraction_female_proof_l258_25897


namespace NUMINAMATH_GPT_paper_clips_in_two_cases_l258_25861

-- Define the conditions
variables (c b : ℕ)

-- Define the theorem statement
theorem paper_clips_in_two_cases (c b : ℕ) : 
    2 * c * b * 400 = 2 * c * b * 400 :=
by
  sorry

end NUMINAMATH_GPT_paper_clips_in_two_cases_l258_25861


namespace NUMINAMATH_GPT_expression_undefined_l258_25846

theorem expression_undefined (a : ℝ) : (a = 2 ∨ a = -2) ↔ (a^2 - 4 = 0) :=
by sorry

end NUMINAMATH_GPT_expression_undefined_l258_25846


namespace NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l258_25803

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem condition_necessary_but_not_sufficient (a_1 d : ℝ) :
  (∀ n : ℕ, S_n a_1 d (n + 1) > S_n a_1 d n) ↔ (a_1 + d > 0) :=
sorry

end NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l258_25803


namespace NUMINAMATH_GPT_function_parallel_l258_25887

theorem function_parallel {x y : ℝ} (h : y = -2 * x + 1) : 
    ∀ {a : ℝ}, y = -2 * a + 3 -> y = -2 * x + 1 := by
    sorry

end NUMINAMATH_GPT_function_parallel_l258_25887


namespace NUMINAMATH_GPT_balls_of_yarn_per_sweater_l258_25835

-- Define the conditions as constants
def cost_per_ball := 6
def sell_price_per_sweater := 35
def total_gain := 308
def number_of_sweaters := 28

-- Define a function that models the total gain given the number of balls of yarn per sweater.
def total_gain_formula (x : ℕ) : ℕ :=
  number_of_sweaters * (sell_price_per_sweater - cost_per_ball * x)

-- State the theorem which proves the number of balls of yarn per sweater
theorem balls_of_yarn_per_sweater (x : ℕ) (h : total_gain_formula x = total_gain): x = 4 :=
sorry

end NUMINAMATH_GPT_balls_of_yarn_per_sweater_l258_25835


namespace NUMINAMATH_GPT_find_number_of_friends_l258_25881

def dante_balloons : Prop :=
  ∃ F : ℕ, (F > 0 ∧ (250 / F) - 11 = 39) ∧ F = 5

theorem find_number_of_friends : dante_balloons :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_friends_l258_25881


namespace NUMINAMATH_GPT_measure_angle_WYZ_l258_25849

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end NUMINAMATH_GPT_measure_angle_WYZ_l258_25849


namespace NUMINAMATH_GPT_rectangle_width_decrease_l258_25888

theorem rectangle_width_decrease {L W : ℝ} (A : ℝ) (hA : A = L * W) (h_new_length : A = 1.25 * L * (W * y)) : y = 0.8 :=
by sorry

end NUMINAMATH_GPT_rectangle_width_decrease_l258_25888


namespace NUMINAMATH_GPT_product_of_roots_l258_25839

theorem product_of_roots :
  let a := 18
  let b := 45
  let c := -500
  let prod_roots := c / a
  prod_roots = -250 / 9 := 
by
  -- Define coefficients
  let a := 18
  let c := -500

  -- Calculate product of roots
  let prod_roots := c / a

  -- Statement to prove
  have : prod_roots = -250 / 9 := sorry
  exact this

-- Adding sorry since the proof is not required according to the problem statement.

end NUMINAMATH_GPT_product_of_roots_l258_25839


namespace NUMINAMATH_GPT_number_of_problems_l258_25853

/-- Given the conditions of the problem, prove that the number of problems I did is exactly 140.-/
theorem number_of_problems (p t : ℕ) (h1 : p > 12) (h2 : p * t = (p + 6) * (t - 3)) : p * t = 140 :=
by
  sorry

end NUMINAMATH_GPT_number_of_problems_l258_25853


namespace NUMINAMATH_GPT_fred_current_money_l258_25820

-- Conditions
def initial_amount_fred : ℕ := 19
def earned_amount_fred : ℕ := 21

-- Question and Proof
theorem fred_current_money : initial_amount_fred + earned_amount_fred = 40 :=
by sorry

end NUMINAMATH_GPT_fred_current_money_l258_25820


namespace NUMINAMATH_GPT_problem_statement_l258_25869

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom f_pos (x : ℝ) : x > 0 → f x > 0
axiom f'_less_f (x : ℝ) : f' x < f x
axiom f_has_deriv_at : ∀ x, HasDerivAt f (f' x) x

def a : ℝ := sorry
axiom a_in_range : 0 < a ∧ a < 1

theorem problem_statement : 3 * f 0 > f a ∧ f a > a * f 1 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l258_25869


namespace NUMINAMATH_GPT_smallest_integer_to_make_1008_perfect_square_l258_25804

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_integer_to_make_1008_perfect_square : ∃ k : ℕ, k > 0 ∧ 
  (∀ m : ℕ, m > 0 → (is_perfect_square (1008 * m) → m ≥ k)) ∧ is_perfect_square (1008 * k) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_to_make_1008_perfect_square_l258_25804


namespace NUMINAMATH_GPT_distance_between_houses_l258_25823

theorem distance_between_houses (d d_JS d_QS : ℝ) (h1 : d_JS = 3) (h2 : d_QS = 1) :
  (2 ≤ d ∧ d ≤ 4) → d = 3 :=
sorry

end NUMINAMATH_GPT_distance_between_houses_l258_25823


namespace NUMINAMATH_GPT_math_contest_students_l258_25882

theorem math_contest_students (n : ℝ) (h : n / 3 + n / 4 + n / 5 + 26 = n) : n = 120 :=
by {
    sorry
}

end NUMINAMATH_GPT_math_contest_students_l258_25882


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l258_25815

-- Define the conditions
def wheels_per_axle : Nat := 2
def total_wheels : Nat := 18
def toll_formula (x : Nat) : ℝ := 1.5 + 0.5 * (x - 2)

-- Calculate number of axles from the number of wheels
def number_of_axles := total_wheels / wheels_per_axle

-- Target statement: The toll for the given truck
theorem toll_for_18_wheel_truck : toll_formula number_of_axles = 5.0 := by
  sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l258_25815


namespace NUMINAMATH_GPT_units_digit_of_square_l258_25859

theorem units_digit_of_square (a b : ℕ) (h₁ : (10 * a + b) ^ 2 % 100 / 10 = 7) : b = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_of_square_l258_25859


namespace NUMINAMATH_GPT_max_val_z_lt_2_l258_25831

-- Definitions for the variables and constraints
variable {x y m : ℝ}
variable (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1)

-- Theorem statement
theorem max_val_z_lt_2 (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1) : 
  (∀ x y, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2) ↔ 1 < m ∧ m < 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_val_z_lt_2_l258_25831


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l258_25899

theorem distance_between_foci_of_ellipse :
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  distance = 2 * Real.sqrt 61 :=
by
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l258_25899


namespace NUMINAMATH_GPT_complex_division_l258_25851

-- Define the imaginary unit 'i'
def i := Complex.I

-- Define the complex numbers as described in the problem
def num := Complex.mk 3 (-1)
def denom := Complex.mk 1 (-1)
def expected := Complex.mk 2 1

-- State the theorem to prove that the complex division is as expected
theorem complex_division : (num / denom) = expected :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l258_25851


namespace NUMINAMATH_GPT_angle_measure_l258_25850

variable (x : ℝ)

noncomputable def is_supplement (x : ℝ) : Prop := 180 - x = 3 * (90 - x) - 60

theorem angle_measure : is_supplement x → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l258_25850


namespace NUMINAMATH_GPT_Peter_can_always_ensure_three_distinct_real_roots_l258_25841

noncomputable def cubic_has_three_distinct_real_roots (b d : ℝ) : Prop :=
∃ (a : ℝ), ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
  (r1 * r2 * r3 = -a) ∧ (r1 + r2 + r3 = -b) ∧ (r1 * r2 + r2 * r3 + r3 * r1 = -d)

theorem Peter_can_always_ensure_three_distinct_real_roots (b d : ℝ) :
  cubic_has_three_distinct_real_roots b d :=
sorry

end NUMINAMATH_GPT_Peter_can_always_ensure_three_distinct_real_roots_l258_25841


namespace NUMINAMATH_GPT_remainder_geometric_series_sum_l258_25863

/-- Define the sum of the geometric series. --/
def geometric_series_sum (n : ℕ) : ℕ :=
  (13^(n+1) - 1) / 12

/-- The given geometric series. --/
def series_sum := geometric_series_sum 1004

/-- Define the modulo operation. --/
def mod_op (a b : ℕ) := a % b

/-- The main statement to prove. --/
theorem remainder_geometric_series_sum :
  mod_op series_sum 1000 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_geometric_series_sum_l258_25863


namespace NUMINAMATH_GPT_find_integers_a_b_c_l258_25833

theorem find_integers_a_b_c :
  ∃ (a b c : ℤ), (∀ (x : ℤ), (x - a) * (x - 8) + 4 = (x + b) * (x + c)) ∧ 
  (a = 20 ∨ a = 29) :=
 by {
      sorry 
}

end NUMINAMATH_GPT_find_integers_a_b_c_l258_25833


namespace NUMINAMATH_GPT_UVWXY_perimeter_l258_25895

theorem UVWXY_perimeter (U V W X Y Z : ℝ) 
  (hUV : UV = 5)
  (hVW : VW = 3)
  (hWY : WY = 5)
  (hYX : YX = 9)
  (hXU : XU = 7) :
  UV + VW + WY + YX + XU = 29 :=
by
  sorry

end NUMINAMATH_GPT_UVWXY_perimeter_l258_25895


namespace NUMINAMATH_GPT_tallest_vs_shortest_height_difference_l258_25810

-- Define the heights of the trees
def pine_tree_height := 12 + 4/5
def birch_tree_height := 18 + 1/2
def maple_tree_height := 14 + 3/5

-- Calculate improper fractions
def pine_tree_improper := 64 / 5
def birch_tree_improper := 41 / 2  -- This is 82/4 but not simplified here
def maple_tree_improper := 73 / 5

-- Calculate height difference
def height_difference := (82 / 4) - (64 / 5)

-- The statement that needs to be proven
theorem tallest_vs_shortest_height_difference : height_difference = 7 + 7 / 10 :=
by 
  sorry

end NUMINAMATH_GPT_tallest_vs_shortest_height_difference_l258_25810


namespace NUMINAMATH_GPT_correct_option_l258_25830

theorem correct_option :
  (∀ (a b : ℝ),  3 * a^2 * b - 4 * b * a^2 = -a^2 * b) ∧
  ¬(1 / 7 * (-7) + (-1 / 7) * 7 = 1) ∧
  ¬((-3 / 5)^2 = 9 / 5) ∧
  ¬(∀ (a b : ℝ), 3 * a + 5 * b = 8 * a * b) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l258_25830


namespace NUMINAMATH_GPT_miles_tankful_highway_l258_25886

variable (miles_tankful_city : ℕ)
variable (mpg_city : ℕ)
variable (mpg_highway : ℕ)

-- Relationship between miles per gallon in city and highway
axiom h_mpg_relation : mpg_highway = mpg_city + 18

-- Given the car travels 336 miles per tankful of gasoline in the city
axiom h_miles_tankful_city : miles_tankful_city = 336

-- Given the car travels 48 miles per gallon in the city
axiom h_mpg_city : mpg_city = 48

-- Prove the car travels 462 miles per tankful of gasoline on the highway
theorem miles_tankful_highway : ∃ (miles_tankful_highway : ℕ), miles_tankful_highway = (mpg_highway * (miles_tankful_city / mpg_city)) := 
by 
  exists (66 * (336 / 48)) -- Since 48 + 18 = 66 and 336 / 48 = 7, 66 * 7 = 462
  sorry

end NUMINAMATH_GPT_miles_tankful_highway_l258_25886


namespace NUMINAMATH_GPT_initial_ratio_milk_water_l258_25818

theorem initial_ratio_milk_water (M W : ℕ) (h1 : M + W = 165) (h2 : ∀ W', W' = W + 66 → M * 4 = 3 * W') : M / gcd M W = 3 ∧ W / gcd M W = 2 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_initial_ratio_milk_water_l258_25818


namespace NUMINAMATH_GPT_wrapping_paper_needed_l258_25808

-- Define the conditions as variables in Lean
def wrapping_paper_first := 3.5
def wrapping_paper_second := (2 / 3) * wrapping_paper_first
def wrapping_paper_third := wrapping_paper_second + 0.5 * wrapping_paper_second
def wrapping_paper_fourth := wrapping_paper_first + wrapping_paper_second
def wrapping_paper_fifth := wrapping_paper_third - 0.25 * wrapping_paper_third

-- Define the total wrapping paper needed
def total_wrapping_paper := wrapping_paper_first + wrapping_paper_second + wrapping_paper_third + wrapping_paper_fourth + wrapping_paper_fifth

-- Statement to prove the final equivalence
theorem wrapping_paper_needed : 
  total_wrapping_paper = 17.79 := 
sorry  -- Proof is omitted

end NUMINAMATH_GPT_wrapping_paper_needed_l258_25808


namespace NUMINAMATH_GPT_trigonometric_identity_application_l258_25898

theorem trigonometric_identity_application :
  2 * (Real.sin (35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) +
       Real.cos (35 * Real.pi / 180) * Real.cos (65 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_application_l258_25898


namespace NUMINAMATH_GPT_negation_of_proposition_l258_25816

theorem negation_of_proposition (x y : ℝ): (x + y > 0 → x > 0 ∧ y > 0) ↔ ¬ ((x + y ≤ 0) → (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l258_25816


namespace NUMINAMATH_GPT_popularity_order_l258_25884

def chess_popularity := 5 / 16
def drama_popularity := 7 / 24
def music_popularity := 11 / 32
def art_popularity := 13 / 48

theorem popularity_order :
  (31 / 96 < 34 / 96) ∧ (34 / 96 < 35 / 96) ∧ (35 / 96 < 36 / 96) ∧ 
  (chess_popularity < music_popularity) ∧ 
  (drama_popularity < music_popularity) ∧ 
  (music_popularity > art_popularity) ∧ 
  (chess_popularity > drama_popularity) ∧ 
  (drama_popularity > art_popularity) := 
sorry

end NUMINAMATH_GPT_popularity_order_l258_25884


namespace NUMINAMATH_GPT_norma_cards_lost_l258_25812

theorem norma_cards_lost (original_cards : ℕ) (current_cards : ℕ) (cards_lost : ℕ)
  (h1 : original_cards = 88) (h2 : current_cards = 18) :
  original_cards - current_cards = cards_lost →
  cards_lost = 70 := by
  sorry

end NUMINAMATH_GPT_norma_cards_lost_l258_25812


namespace NUMINAMATH_GPT_clown_balloon_count_l258_25873

theorem clown_balloon_count (b1 b2 : ℕ) (h1 : b1 = 47) (h2 : b2 = 13) : b1 + b2 = 60 := by
  sorry

end NUMINAMATH_GPT_clown_balloon_count_l258_25873


namespace NUMINAMATH_GPT_scoops_per_carton_l258_25871

-- Definitions for scoops required by everyone
def ethan_vanilla := 1
def ethan_chocolate := 1
def lucas_danny_connor_chocolate_each := 2
def lucas_danny_connor := 3
def olivia_vanilla := 1
def olivia_strawberry := 1
def shannon_vanilla := 2 * olivia_vanilla
def shannon_strawberry := 2 * olivia_strawberry

-- Definitions for total scoops taken
def total_vanilla_taken := ethan_vanilla + olivia_vanilla + shannon_vanilla
def total_chocolate_taken := ethan_chocolate + (lucas_danny_connor_chocolate_each * lucas_danny_connor)
def total_strawberry_taken := olivia_strawberry + shannon_strawberry
def total_scoops_taken := total_vanilla_taken + total_chocolate_taken + total_strawberry_taken

-- Definitions for remaining scoops and original total scoops
def remaining_scoops := 16
def original_scoops := total_scoops_taken + remaining_scoops

-- Definition for number of cartons
def total_cartons := 3

-- Proof goal: scoops per carton
theorem scoops_per_carton : original_scoops / total_cartons = 10 := 
by
  -- Add your proof steps here
  sorry

end NUMINAMATH_GPT_scoops_per_carton_l258_25871


namespace NUMINAMATH_GPT_Greatest_Percentage_Difference_l258_25805

def max_percentage_difference (B W P : ℕ) : ℕ :=
  ((max B (max W P) - min B (min W P)) * 100) / (min B (min W P))

def January_B : ℕ := 6
def January_W : ℕ := 4
def January_P : ℕ := 5

def February_B : ℕ := 7
def February_W : ℕ := 5
def February_P : ℕ := 6

def March_B : ℕ := 7
def March_W : ℕ := 7
def March_P : ℕ := 7

def April_B : ℕ := 5
def April_W : ℕ := 6
def April_P : ℕ := 7

def May_B : ℕ := 3
def May_W : ℕ := 4
def May_P : ℕ := 2

theorem Greatest_Percentage_Difference :
  max_percentage_difference May_B May_W May_P >
  max (max_percentage_difference January_B January_W January_P)
      (max (max_percentage_difference February_B February_W February_P)
           (max (max_percentage_difference March_B March_W March_P)
                (max_percentage_difference April_B April_W April_P))) :=
by
  sorry

end NUMINAMATH_GPT_Greatest_Percentage_Difference_l258_25805


namespace NUMINAMATH_GPT_problem_y_values_l258_25811

theorem problem_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 54) :
  ∃ y : ℝ, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 7.5 ∨ y = 4.5) := by
sorry

end NUMINAMATH_GPT_problem_y_values_l258_25811


namespace NUMINAMATH_GPT_largest_sum_faces_l258_25838

theorem largest_sum_faces (a b c d e f : ℕ)
  (h_ab : a + b ≤ 7) (h_ac : a + c ≤ 7) (h_ad : a + d ≤ 7) (h_ae : a + e ≤ 7) (h_af : a + f ≤ 7)
  (h_bc : b + c ≤ 7) (h_bd : b + d ≤ 7) (h_be : b + e ≤ 7) (h_bf : b + f ≤ 7)
  (h_cd : c + d ≤ 7) (h_ce : c + e ≤ 7) (h_cf : c + f ≤ 7)
  (h_de : d + e ≤ 7) (h_df : d + f ≤ 7)
  (h_ef : e + f ≤ 7) :
  ∃ x y z, 
  ((x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧ 
   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) ∧ 
   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e ∨ z = f)) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (x + y ≤ 7) ∧ (y + z ≤ 7) ∧ (x + z ≤ 7) ∧
  (x + y + z = 9) :=
sorry

end NUMINAMATH_GPT_largest_sum_faces_l258_25838


namespace NUMINAMATH_GPT_part1_part2_l258_25889

open Real

noncomputable def f (x : ℝ) : ℝ := abs ((2 / 3) * x + 1)

theorem part1 (a : ℝ) : (∀ x, f x ≥ -abs x + a) → a ≤ 1 :=
sorry

theorem part2 (x y : ℝ) (h1 : abs (x + y + 1) ≤ 1 / 3) (h2 : abs (y - 1 / 3) ≤ 2 / 3) : 
  f x ≤ 7 / 9 :=
sorry

end NUMINAMATH_GPT_part1_part2_l258_25889


namespace NUMINAMATH_GPT_interval_f_has_two_roots_l258_25817

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

theorem interval_f_has_two_roots (a : ℝ) : (∀ x : ℝ, f a x = 0 → ∃ u v : ℝ, u ≠ v ∧ f a u = 0 ∧ f a v = 0) ↔ 0 < a ∧ a < 1 / 8 := 
sorry

end NUMINAMATH_GPT_interval_f_has_two_roots_l258_25817


namespace NUMINAMATH_GPT_temperature_difference_l258_25824

theorem temperature_difference (T_high T_low : ℤ) (h_high : T_high = 11) (h_low : T_low = -11) :
  T_high - T_low = 22 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l258_25824
