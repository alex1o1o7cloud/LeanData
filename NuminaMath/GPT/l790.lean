import Mathlib

namespace NUMINAMATH_GPT_hyperbola_distance_property_l790_79009

theorem hyperbola_distance_property (P : ℝ × ℝ)
  (hP_on_hyperbola : (P.1 ^ 2 / 16) - (P.2 ^ 2 / 9) = 1)
  (h_dist_15 : dist P (5, 0) = 15) :
  dist P (-5, 0) = 7 ∨ dist P (-5, 0) = 23 := 
sorry

end NUMINAMATH_GPT_hyperbola_distance_property_l790_79009


namespace NUMINAMATH_GPT_find_range_of_m_l790_79093

-- Define properties of ellipses and hyperbolas
def isEllipseY (m : ℝ) : Prop := (8 - m > 2 * m - 1 ∧ 2 * m - 1 > 0)
def isHyperbola (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- The range of 'm' such that (p ∨ q) is true and (p ∧ q) is false
def p_or_q_true_p_and_q_false (m : ℝ) : Prop := 
  (isEllipseY m ∨ isHyperbola m) ∧ ¬ (isEllipseY m ∧ isHyperbola m)

-- The range of the real number 'm'
def range_of_m (m : ℝ) : Prop := 
  (-1 < m ∧ m ≤ 1/2) ∨ (2 ≤ m ∧ m < 3)

-- Prove that the above conditions imply the correct range for m
theorem find_range_of_m (m : ℝ) : p_or_q_true_p_and_q_false m → range_of_m m :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l790_79093


namespace NUMINAMATH_GPT_f_of_3_l790_79041

def f (x : ℕ) : ℤ :=
  if x = 0 then sorry else 2 * (x - 1) - 1  -- Define an appropriate value for f(0) later

theorem f_of_3 : f 3 = 3 := by
  sorry

end NUMINAMATH_GPT_f_of_3_l790_79041


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l790_79036

-- define the partitions function
def P (k l n : ℕ) : ℕ := sorry

-- Part (a) statement
theorem part_a (k l n : ℕ) :
  P k l n - P k (l - 1) n = P (k - 1) l (n - l) :=
sorry

-- Part (b) statement
theorem part_b (k l n : ℕ) :
  P k l n - P (k - 1) l n = P k (l - 1) (n - k) :=
sorry

-- Part (c) statement
theorem part_c (k l n : ℕ) :
  P k l n = P l k n :=
sorry

-- Part (d) statement
theorem part_d (k l n : ℕ) :
  P k l n = P k l (k * l - n) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l790_79036


namespace NUMINAMATH_GPT_speed_of_first_car_l790_79057

theorem speed_of_first_car (v : ℝ) (h1 : 2.5 * v + 2.5 * 45 = 175) : v = 25 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_first_car_l790_79057


namespace NUMINAMATH_GPT_roots_of_quadratic_sum_of_sixth_powers_l790_79099

theorem roots_of_quadratic_sum_of_sixth_powers {u v : ℝ} 
  (h₀ : u^2 - 2*u*Real.sqrt 3 + 1 = 0)
  (h₁ : v^2 - 2*v*Real.sqrt 3 + 1 = 0)
  : u^6 + v^6 = 970 := 
by 
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_sum_of_sixth_powers_l790_79099


namespace NUMINAMATH_GPT_solve_quadratic_eq_l790_79056

theorem solve_quadratic_eq (a b x : ℝ) :
  12 * a * b * x^2 - (16 * a^2 - 9 * b^2) * x - 12 * a * b = 0 ↔ (x = 4 * a / (3 * b)) ∨ (x = -3 * b / (4 * a)) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l790_79056


namespace NUMINAMATH_GPT_remaining_inventory_l790_79064

def initial_inventory : Int := 4500
def bottles_sold_mon : Int := 2445
def bottles_sold_tue : Int := 906
def bottles_sold_wed : Int := 215
def bottles_sold_thu : Int := 457
def bottles_sold_fri : Int := 312
def bottles_sold_sat : Int := 239
def bottles_sold_sun : Int := 188

def bottles_received_tue : Int := 350
def bottles_received_thu : Int := 750
def bottles_received_sat : Int := 981

def total_bottles_sold : Int := bottles_sold_mon + bottles_sold_tue + bottles_sold_wed + bottles_sold_thu + bottles_sold_fri + bottles_sold_sat + bottles_sold_sun
def total_bottles_received : Int := bottles_received_tue + bottles_received_thu + bottles_received_sat

theorem remaining_inventory (initial_inventory bottles_sold_mon bottles_sold_tue bottles_sold_wed bottles_sold_thu bottles_sold_fri bottles_sold_sat bottles_sold_sun bottles_received_tue bottles_received_thu bottles_received_sat total_bottles_sold total_bottles_received : Int) :
  initial_inventory - total_bottles_sold + total_bottles_received = 819 :=
by
  sorry

end NUMINAMATH_GPT_remaining_inventory_l790_79064


namespace NUMINAMATH_GPT_cubic_polynomial_roots_l790_79030

theorem cubic_polynomial_roots (a : ℚ) :
  (x^3 - 6*x^2 + a*x - 6 = 0) ∧ (x = 3) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_l790_79030


namespace NUMINAMATH_GPT_candy_difference_l790_79077

theorem candy_difference 
  (total_candies : ℕ)
  (strawberry_candies : ℕ)
  (total_eq : total_candies = 821)
  (strawberry_eq : strawberry_candies = 267) : 
  (total_candies - strawberry_candies - strawberry_candies = 287) :=
by
  sorry

end NUMINAMATH_GPT_candy_difference_l790_79077


namespace NUMINAMATH_GPT_ball_bounces_below_2_feet_l790_79092

theorem ball_bounces_below_2_feet :
  ∃ k : ℕ, 500 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ n < k, 500 * (2 / 3 : ℝ) ^ n ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_ball_bounces_below_2_feet_l790_79092


namespace NUMINAMATH_GPT_measure_of_angle_x_in_triangle_l790_79010

theorem measure_of_angle_x_in_triangle
  (x : ℝ)
  (h1 : x + 2 * x + 45 = 180) :
  x = 45 :=
sorry

end NUMINAMATH_GPT_measure_of_angle_x_in_triangle_l790_79010


namespace NUMINAMATH_GPT_games_bought_from_friend_is_21_l790_79001

-- Definitions from the conditions
def games_bought_at_garage_sale : ℕ := 8
def non_working_games : ℕ := 23
def good_games : ℕ := 6

-- The total number of games John has is the sum of good and non-working games
def total_games : ℕ := good_games + non_working_games

-- The number of games John bought from his friend
def games_from_friend : ℕ := total_games - games_bought_at_garage_sale

-- Statement to prove
theorem games_bought_from_friend_is_21 : games_from_friend = 21 := by
  sorry

end NUMINAMATH_GPT_games_bought_from_friend_is_21_l790_79001


namespace NUMINAMATH_GPT_total_distance_both_l790_79003

-- Define conditions
def speed_onur : ℝ := 35  -- km/h
def speed_hanil : ℝ := 45  -- km/h
def daily_hours_onur : ℝ := 7
def additional_distance_hanil : ℝ := 40
def days_in_week : ℕ := 7

-- Define the daily biking distance for Onur and Hanil
def distance_onur_daily : ℝ := speed_onur * daily_hours_onur
def distance_hanil_daily : ℝ := distance_onur_daily + additional_distance_hanil

-- Define the number of days Onur and Hanil bike in a week
def working_days_onur : ℕ := 5
def working_days_hanil : ℕ := 6

-- Define the total distance covered by Onur and Hanil in a week
def total_distance_onur_week : ℝ := distance_onur_daily * working_days_onur
def total_distance_hanil_week : ℝ := distance_hanil_daily * working_days_hanil

-- Proof statement
theorem total_distance_both : total_distance_onur_week + total_distance_hanil_week = 2935 := by
  sorry

end NUMINAMATH_GPT_total_distance_both_l790_79003


namespace NUMINAMATH_GPT_train_crossing_platform_time_l790_79071

theorem train_crossing_platform_time
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_signal_pole : ℝ)
  (speed : ℝ)
  (time_platform_cross : ℝ)
  (v := length_train / time_signal_pole)
  (d := length_train + length_platform)
  (t := d / v) :
  length_train = 300 →
  length_platform = 250 →
  time_signal_pole = 18 →
  time_platform_cross = 33 →
  t = time_platform_cross := by
  sorry

end NUMINAMATH_GPT_train_crossing_platform_time_l790_79071


namespace NUMINAMATH_GPT_computer_production_per_month_l790_79051

def days : ℕ := 28
def hours_per_day : ℕ := 24
def intervals_per_hour : ℕ := 2
def computers_per_interval : ℕ := 3

theorem computer_production_per_month : 
  (days * hours_per_day * intervals_per_hour * computers_per_interval = 4032) :=
by sorry

end NUMINAMATH_GPT_computer_production_per_month_l790_79051


namespace NUMINAMATH_GPT_average_of_P_Q_R_is_correct_l790_79090

theorem average_of_P_Q_R_is_correct (P Q R : ℝ) 
  (h1 : 1001 * R - 3003 * P = 6006) 
  (h2 : 2002 * Q + 4004 * P = 8008) : 
  (P + Q + R)/3 = (2 * (P + 5))/3 :=
sorry

end NUMINAMATH_GPT_average_of_P_Q_R_is_correct_l790_79090


namespace NUMINAMATH_GPT_point_three_units_away_from_A_is_negative_seven_or_negative_one_l790_79029

-- Defining the point A on the number line
def A : ℤ := -4

-- Definition of the condition where a point is 3 units away from A
def three_units_away (x : ℤ) : Prop := (x = A - 3) ∨ (x = A + 3)

-- The statement to be proved
theorem point_three_units_away_from_A_is_negative_seven_or_negative_one (x : ℤ) :
  three_units_away x → (x = -7 ∨ x = -1) :=
sorry

end NUMINAMATH_GPT_point_three_units_away_from_A_is_negative_seven_or_negative_one_l790_79029


namespace NUMINAMATH_GPT_range_estimate_of_expression_l790_79067

theorem range_estimate_of_expression : 
  6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
       (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 :=
by
  sorry

end NUMINAMATH_GPT_range_estimate_of_expression_l790_79067


namespace NUMINAMATH_GPT_jon_monthly_earnings_l790_79033

def earnings_per_person : ℝ := 0.10
def visits_per_hour : ℕ := 50
def hours_per_day : ℕ := 24
def days_per_month : ℕ := 30

theorem jon_monthly_earnings : 
  (earnings_per_person * visits_per_hour * hours_per_day * days_per_month) = 3600 :=
by
  sorry

end NUMINAMATH_GPT_jon_monthly_earnings_l790_79033


namespace NUMINAMATH_GPT_max_mn_value_l790_79025

theorem max_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (hA1 : ∀ k : ℝ, k * (-2) - (-1) + 2 * k - 1 = 0)
  (hA2 : m * (-2) + n * (-1) + 2 = 0) :
  mn ≤ 1/2 := sorry

end NUMINAMATH_GPT_max_mn_value_l790_79025


namespace NUMINAMATH_GPT_num_valid_n_l790_79091

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end NUMINAMATH_GPT_num_valid_n_l790_79091


namespace NUMINAMATH_GPT_ratio_of_heights_l790_79097

theorem ratio_of_heights (a b : ℝ) (area_ratio_is_9_4 : a / b = 9 / 4) :
  ∃ h₁ h₂ : ℝ, h₁ / h₂ = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_heights_l790_79097


namespace NUMINAMATH_GPT_obtain_half_not_obtain_one_l790_79038

theorem obtain_half (x : ℕ) : (10 + x) / (97 + x) = 1 / 2 ↔ x = 77 := 
by
  sorry

theorem not_obtain_one (x k : ℕ) : ¬ ((10 + x) / (97 + x) = 1 ∨ (10 * k) / (97 * k) = 1) := 
by
  sorry

end NUMINAMATH_GPT_obtain_half_not_obtain_one_l790_79038


namespace NUMINAMATH_GPT_shortest_paths_in_grid_l790_79070

-- Define a function that computes the binomial coefficient
def binom (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

-- Proof problem: Prove that the number of shortest paths in an m x n grid is binom(m, n)
theorem shortest_paths_in_grid (m n : ℕ) : binom m n = Nat.choose (m + n) n :=
by
  -- Intentionally left blank: proof is skipped
  sorry

end NUMINAMATH_GPT_shortest_paths_in_grid_l790_79070


namespace NUMINAMATH_GPT_min_marbles_to_draw_l790_79065

theorem min_marbles_to_draw (reds greens blues yellows oranges purples : ℕ)
  (h_reds : reds = 35)
  (h_greens : greens = 25)
  (h_blues : blues = 24)
  (h_yellows : yellows = 18)
  (h_oranges : oranges = 15)
  (h_purples : purples = 12)
  : ∃ n : ℕ, n = 103 ∧ (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r < 20 ∧ g < 20 ∧ b < 20 ∧ y < 20 ∧ o < 20 ∧ p < 20 → r + g + b + y + o + p < n) ∧
      (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r + g + b + y + o + p = n → r = 20 ∨ g = 20 ∨ b = 20 ∨ y = 20 ∨ o = 20 ∨ p = 20) :=
sorry

end NUMINAMATH_GPT_min_marbles_to_draw_l790_79065


namespace NUMINAMATH_GPT_find_x_of_equation_l790_79002

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end NUMINAMATH_GPT_find_x_of_equation_l790_79002


namespace NUMINAMATH_GPT_complex_power_equality_l790_79069

namespace ComplexProof

open Complex

noncomputable def cos5 : ℂ := cos (5 * Real.pi / 180)

theorem complex_power_equality (w : ℂ) (h : w + 1 / w = 2 * cos5) : 
  w ^ 1000 + 1 / (w ^ 1000) = -((Real.sqrt 5 + 1) / 2) :=
sorry

end ComplexProof

end NUMINAMATH_GPT_complex_power_equality_l790_79069


namespace NUMINAMATH_GPT_profit_is_correct_l790_79031

-- Define the constants for expenses
def cost_of_lemons : ℕ := 10
def cost_of_sugar : ℕ := 5
def cost_of_cups : ℕ := 3

-- Define the cost per cup of lemonade
def price_per_cup : ℕ := 4

-- Define the number of cups sold
def cups_sold : ℕ := 21

-- Define the total revenue
def total_revenue : ℕ := cups_sold * price_per_cup

-- Define the total expenses
def total_expenses : ℕ := cost_of_lemons + cost_of_sugar + cost_of_cups

-- Define the profit
def profit : ℕ := total_revenue - total_expenses

-- The theorem stating the profit
theorem profit_is_correct : profit = 66 := by
  sorry

end NUMINAMATH_GPT_profit_is_correct_l790_79031


namespace NUMINAMATH_GPT_train_speed_correct_l790_79059

def length_of_train := 280 -- in meters
def time_to_pass_tree := 16 -- in seconds
def speed_of_train := 63 -- in km/hr

theorem train_speed_correct :
  (length_of_train / time_to_pass_tree) * (3600 / 1000) = speed_of_train :=
sorry

end NUMINAMATH_GPT_train_speed_correct_l790_79059


namespace NUMINAMATH_GPT_area_difference_depends_only_on_bw_l790_79042

variable (b w n : ℕ)
variable (hb : b ≥ 2)
variable (hw : w ≥ 2)
variable (hn : n = b + w)

/-- Given conditions: 
1. \(b \geq 2\) 
2. \(w \geq 2\) 
3. \(n = b + w\)
4. There are \(2b\) identical black rods and \(2w\) identical white rods, each of side length 1. 
5. These rods form a regular \(2n\)-gon with parallel sides of the same color.
6. A convex \(2b\)-gon \(B\) is formed by translating the black rods. 
7. A convex \(2w\) A convex \(2w\)-gon \(W\) is formed by translating the white rods. 
Prove that the difference of the areas of \(B\) and \(W\) depends only on the numbers \(b\) and \(w\). -/
theorem area_difference_depends_only_on_bw :
  ∀ (A B W : ℝ), A - B = 2 * (b - w) :=
sorry

end NUMINAMATH_GPT_area_difference_depends_only_on_bw_l790_79042


namespace NUMINAMATH_GPT_greatest_perimeter_triangle_l790_79047

theorem greatest_perimeter_triangle :
  ∃ (x : ℕ), (x > (16 / 5)) ∧ (x < (16 / 3)) ∧ ((x = 4 ∨ x = 5) → 4 * x + x + 16 = 41) :=
by
  sorry

end NUMINAMATH_GPT_greatest_perimeter_triangle_l790_79047


namespace NUMINAMATH_GPT_Victor_more_scoops_l790_79081

def ground_almonds : ℝ := 1.56
def white_sugar : ℝ := 0.75

theorem Victor_more_scoops :
  ground_almonds - white_sugar = 0.81 :=
by
  sorry

end NUMINAMATH_GPT_Victor_more_scoops_l790_79081


namespace NUMINAMATH_GPT_automotive_test_l790_79084

noncomputable def total_distance (D : ℝ) (t : ℝ) : ℝ := 3 * D

theorem automotive_test (D : ℝ) (h_time : (D / 4 + D / 5 + D / 6 = 37)) : total_distance D 37 = 180 :=
  by
    -- This skips the proof, only the statement is given
    sorry

end NUMINAMATH_GPT_automotive_test_l790_79084


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l790_79012

variable (a b c : ℝ)

theorem inequality_for_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l790_79012


namespace NUMINAMATH_GPT_compare_f_values_l790_79011

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem compare_f_values :
  f 0.6 > f (-0.5) ∧ f (-0.5) > f 0 := by
  sorry

end NUMINAMATH_GPT_compare_f_values_l790_79011


namespace NUMINAMATH_GPT_ferry_journey_difference_l790_79032

theorem ferry_journey_difference
  (time_P : ℝ) (speed_P : ℝ) (mult_Q : ℝ) (speed_diff : ℝ)
  (dist_P : ℝ := time_P * speed_P)
  (dist_Q : ℝ := mult_Q * dist_P)
  (speed_Q : ℝ := speed_P + speed_diff)
  (time_Q : ℝ := dist_Q / speed_Q) :
  time_P = 3 ∧ speed_P = 6 ∧ mult_Q = 3 ∧ speed_diff = 3 → time_Q - time_P = 3 := by
  sorry

end NUMINAMATH_GPT_ferry_journey_difference_l790_79032


namespace NUMINAMATH_GPT_earliest_meeting_time_l790_79018

theorem earliest_meeting_time
    (charlie_lap : ℕ := 5)
    (ben_lap : ℕ := 8)
    (laura_lap_effective : ℕ := 11) :
    lcm (lcm charlie_lap ben_lap) laura_lap_effective = 440 := by
  sorry

end NUMINAMATH_GPT_earliest_meeting_time_l790_79018


namespace NUMINAMATH_GPT_new_sample_variance_l790_79024

-- Definitions based on conditions
def sample_size (original : Nat) : Prop := original = 7
def sample_average (original : ℝ) : Prop := original = 5
def sample_variance (original : ℝ) : Prop := original = 2
def new_data_point (point : ℝ) : Prop := point = 5

-- Statement to be proved
theorem new_sample_variance (original_size : Nat) (original_avg : ℝ) (original_var : ℝ) (new_point : ℝ) 
  (h₁ : sample_size original_size) 
  (h₂ : sample_average original_avg) 
  (h₃ : sample_variance original_var) 
  (h₄ : new_data_point new_point) : 
  (8 * original_var + 0) / 8 = 7 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_new_sample_variance_l790_79024


namespace NUMINAMATH_GPT_false_proposition_A_l790_79098

theorem false_proposition_A 
  (a b : ℝ)
  (root1_eq_1 : ∀ x, x^2 + a * x + b = 0 → x = 1)
  (root2_eq_3 : ∀ x, x^2 + a * x + b = 0 → x = 3)
  (sum_of_roots_eq_2 : -a = 2)
  (opposite_sign_roots : ∀ x1 x2, x1 * x2 < 0) :
  ∃ prop, prop = "A" :=
sorry

end NUMINAMATH_GPT_false_proposition_A_l790_79098


namespace NUMINAMATH_GPT_radio_range_l790_79068

-- Define constants for speeds and time
def speed_team_1 : ℝ := 20
def speed_team_2 : ℝ := 30
def time : ℝ := 2.5

-- Define the distances each team travels
def distance_team_1 := speed_team_1 * time
def distance_team_2 := speed_team_2 * time

-- Define the total distance which is the range of the radios
def total_distance := distance_team_1 + distance_team_2

-- Prove that the total distance when they lose radio contact is 125 miles
theorem radio_range : total_distance = 125 := by
  sorry

end NUMINAMATH_GPT_radio_range_l790_79068


namespace NUMINAMATH_GPT_brooke_social_studies_problems_l790_79083

theorem brooke_social_studies_problems :
  ∀ (math_problems science_problems total_minutes : Nat) 
    (math_time_per_problem science_time_per_problem soc_studies_time_per_problem : Nat)
    (soc_studies_problems : Nat),
  math_problems = 15 →
  science_problems = 10 →
  total_minutes = 48 →
  math_time_per_problem = 2 →
  science_time_per_problem = 3 / 2 → -- converting 1.5 minutes to a fraction
  soc_studies_time_per_problem = 1 / 2 → -- converting 30 seconds to a fraction
  math_problems * math_time_per_problem + science_problems * science_time_per_problem + soc_studies_problems * soc_studies_time_per_problem = 48 →
  soc_studies_problems = 6 :=
by
  intros math_problems science_problems total_minutes math_time_per_problem science_time_per_problem soc_studies_time_per_problem soc_studies_problems
  intros h_math_problems h_science_problems h_total_minutes h_math_time_per_problem h_science_time_per_problem h_soc_studies_time_per_problem h_eq
  sorry

end NUMINAMATH_GPT_brooke_social_studies_problems_l790_79083


namespace NUMINAMATH_GPT_purely_imaginary_complex_number_l790_79017

theorem purely_imaginary_complex_number (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 4 * m + 3 ≠ 0) → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_complex_number_l790_79017


namespace NUMINAMATH_GPT_tenth_number_in_row_1_sum_of_2023rd_numbers_l790_79072

noncomputable def a (n : ℕ) := (-2)^n
noncomputable def b (n : ℕ) := a n + (n + 1)

theorem tenth_number_in_row_1 : a 10 = (-2)^10 := 
sorry

theorem sum_of_2023rd_numbers : a 2023 + b 2023 = -(2^2024) + 2024 := 
sorry

end NUMINAMATH_GPT_tenth_number_in_row_1_sum_of_2023rd_numbers_l790_79072


namespace NUMINAMATH_GPT_gcd_lcm_find_other_number_l790_79014

theorem gcd_lcm_find_other_number {a b : ℕ} (h_gcd : Nat.gcd a b = 36) (h_lcm : Nat.lcm a b = 8820) (h_a : a = 360) : b = 882 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_find_other_number_l790_79014


namespace NUMINAMATH_GPT_a_is_perfect_square_l790_79000

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end NUMINAMATH_GPT_a_is_perfect_square_l790_79000


namespace NUMINAMATH_GPT_delta_value_l790_79052

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end NUMINAMATH_GPT_delta_value_l790_79052


namespace NUMINAMATH_GPT_smallest_number_l790_79015

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_number_l790_79015


namespace NUMINAMATH_GPT_trajectory_eq_l790_79066

theorem trajectory_eq {x y m : ℝ} (h : x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0) :
  x - 2 * y - 1 = 0 ∧ x ≠ 1 :=
sorry

end NUMINAMATH_GPT_trajectory_eq_l790_79066


namespace NUMINAMATH_GPT_cody_initial_marbles_l790_79023

theorem cody_initial_marbles (M : ℕ) (h1 : (2 / 3 : ℝ) * M - (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M) - (2 * (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M)) = 7) : M = 42 := 
  sorry

end NUMINAMATH_GPT_cody_initial_marbles_l790_79023


namespace NUMINAMATH_GPT_find_single_digit_A_l790_79027

theorem find_single_digit_A (A : ℕ) (h1 : A < 10) (h2 : (11 * A)^2 = 5929) : A = 7 := 
sorry

end NUMINAMATH_GPT_find_single_digit_A_l790_79027


namespace NUMINAMATH_GPT_abc_value_l790_79043

-- Define constants for the problem
variable (a b c k : ℕ)

-- Assumptions based on the given conditions
axiom h1 : a - b = 3
axiom h2 : a^2 + b^2 = 29
axiom h3 : a^2 + b^2 + c^2 = k
axiom pos_k : k > 0
axiom pos_a : a > 0

-- The goal is to prove that abc = 10
theorem abc_value : a * b * c = 10 :=
by
  sorry

end NUMINAMATH_GPT_abc_value_l790_79043


namespace NUMINAMATH_GPT_friends_raise_funds_l790_79076

theorem friends_raise_funds (total_amount friends_count min_amount amount_per_person: ℕ)
  (h1 : total_amount = 3000)
  (h2 : friends_count = 10)
  (h3 : min_amount = 300)
  (h4 : amount_per_person = total_amount / friends_count) :
  amount_per_person = min_amount :=
by
  sorry

end NUMINAMATH_GPT_friends_raise_funds_l790_79076


namespace NUMINAMATH_GPT_power_function_pass_through_point_l790_79005

theorem power_function_pass_through_point (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ a) (h_point : f 2 = 16) : a = 4 :=
sorry

end NUMINAMATH_GPT_power_function_pass_through_point_l790_79005


namespace NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l790_79079

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l790_79079


namespace NUMINAMATH_GPT_find_difference_l790_79026

-- Define the initial amounts each person paid.
def Alex_paid : ℕ := 95
def Tom_paid : ℕ := 140
def Dorothy_paid : ℕ := 110
def Sammy_paid : ℕ := 155

-- Define the total spent and the share per person.
def total_spent : ℕ := Alex_paid + Tom_paid + Dorothy_paid + Sammy_paid
def share : ℕ := total_spent / 4

-- Define how much each person needs to pay or should receive.
def Alex_balance : ℤ := share - Alex_paid
def Tom_balance : ℤ := Tom_paid - share
def Dorothy_balance : ℤ := share - Dorothy_paid
def Sammy_balance : ℤ := Sammy_paid - share

-- Define the values of t and d.
def t : ℤ := 0
def d : ℤ := 15

-- The proof goal
theorem find_difference : t - d = -15 := by
  sorry

end NUMINAMATH_GPT_find_difference_l790_79026


namespace NUMINAMATH_GPT_inequality_negatives_l790_79019

theorem inequality_negatives (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_inequality_negatives_l790_79019


namespace NUMINAMATH_GPT_find_fibonacci_x_l790_79075

def is_fibonacci (a b c : ℕ) : Prop :=
  c = a + b

theorem find_fibonacci_x (a b x : ℕ)
  (h₁ : a = 8)
  (h₂ : b = 13)
  (h₃ : is_fibonacci a b x) :
  x = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_fibonacci_x_l790_79075


namespace NUMINAMATH_GPT_amount_of_H2O_formed_l790_79053

-- Define the balanced chemical equation as a relation
def balanced_equation : Prop :=
  ∀ (naoh hcl nacl h2o : ℕ), 
    (naoh + hcl = nacl + h2o)

-- Define the reaction of 2 moles of NaOH and 2 moles of HCl
def reaction (naoh hcl : ℕ) : ℕ :=
  if (naoh = 2) ∧ (hcl = 2) then 2 else 0

theorem amount_of_H2O_formed :
  balanced_equation →
  reaction 2 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_H2O_formed_l790_79053


namespace NUMINAMATH_GPT_sqrt_four_ninths_l790_79022

theorem sqrt_four_ninths : 
  (∀ (x : ℚ), x * x = 4 / 9 → (x = 2 / 3 ∨ x = - (2 / 3))) :=
by sorry

end NUMINAMATH_GPT_sqrt_four_ninths_l790_79022


namespace NUMINAMATH_GPT_number_of_tables_large_meeting_l790_79063

-- Conditions
def table_length : ℕ := 2
def table_width : ℕ := 1
def side_length_large_meeting : ℕ := 7

-- To be proved: number of tables needed for a large meeting is 12.
theorem number_of_tables_large_meeting : 
  let tables_per_side := side_length_large_meeting / (table_length + table_width)
  ∃ total_tables, total_tables = 4 * tables_per_side ∧ total_tables = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tables_large_meeting_l790_79063


namespace NUMINAMATH_GPT_complex_abs_sum_eq_1_or_3_l790_79044

open Complex

theorem complex_abs_sum_eq_1_or_3 (a b c : ℂ) (ha : abs a = 1) (hb : abs b = 1) (hc : abs c = 1) 
  (h : a^3/(b^2 * c) + b^3/(a^2 * c) + c^3/(a^2 * b) = 1) : abs (a + b + c) = 1 ∨ abs (a + b + c) = 3 := 
by
  sorry

end NUMINAMATH_GPT_complex_abs_sum_eq_1_or_3_l790_79044


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l790_79074

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  {x : ℝ | 0 < x ∧ x ≤ 1} = {x : ℝ | ∃ ε > 0, ∀ y, y < x → f y > f x ∧ y > 0} :=
sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l790_79074


namespace NUMINAMATH_GPT_combined_total_time_l790_79035

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end NUMINAMATH_GPT_combined_total_time_l790_79035


namespace NUMINAMATH_GPT_division_scaling_l790_79082

theorem division_scaling (h : 204 / 12.75 = 16) : 2.04 / 1.275 = 16 :=
sorry

end NUMINAMATH_GPT_division_scaling_l790_79082


namespace NUMINAMATH_GPT_round_to_nearest_whole_l790_79096

theorem round_to_nearest_whole (x : ℝ) (hx : x = 12345.49999) : round x = 12345 := by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_round_to_nearest_whole_l790_79096


namespace NUMINAMATH_GPT_income_to_expenditure_ratio_l790_79088

-- Define the constants based on the conditions in step a)
def income : ℕ := 36000
def savings : ℕ := 4000

-- Define the expenditure as a function of income and savings
def expenditure (I S : ℕ) : ℕ := I - S

-- Define the ratio of two natural numbers
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to be proved
theorem income_to_expenditure_ratio : 
  ratio income (expenditure income savings) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_income_to_expenditure_ratio_l790_79088


namespace NUMINAMATH_GPT_find_primes_l790_79016

open Int

theorem find_primes (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p ^ x = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_primes_l790_79016


namespace NUMINAMATH_GPT_james_carrot_sticks_l790_79078

theorem james_carrot_sticks (x : ℕ) (h : x + 15 = 37) : x = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_james_carrot_sticks_l790_79078


namespace NUMINAMATH_GPT_range_of_m_minimum_value_ab_l790_79061

-- Define the given condition as a predicate on the real numbers
def domain_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Define the first part of the proof problem: range of m
theorem range_of_m :
  (∀ m : ℝ, domain_condition m) → ∀ m : ℝ, m ≤ 6 :=
sorry

-- Define the second part of the proof problem: minimum value of 4a + 7b
theorem minimum_value_ab (n : ℝ) (a b : ℝ) (h : n = 6) :
  (∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (4 / (a + 5 * b) + 1 / (3 * a + 2 * b) = n)) → 
  ∃ (a b : ℝ), 4 * a + 7 * b = 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_minimum_value_ab_l790_79061


namespace NUMINAMATH_GPT_remainder_2_pow_224_plus_104_l790_79039

theorem remainder_2_pow_224_plus_104 (x : ℕ) (h1 : x = 2 ^ 56) : 
  (2 ^ 224 + 104) % (2 ^ 112 + 2 ^ 56 + 1) = 103 := 
by
  sorry

end NUMINAMATH_GPT_remainder_2_pow_224_plus_104_l790_79039


namespace NUMINAMATH_GPT_remaining_soup_can_feed_adults_l790_79028

-- Define initial conditions
def cans_per_soup_for_children : ℕ := 6
def cans_per_soup_for_adults : ℕ := 4
def initial_cans : ℕ := 8
def children_to_feed : ℕ := 24

-- Define the problem statement in Lean
theorem remaining_soup_can_feed_adults :
  (initial_cans - (children_to_feed / cans_per_soup_for_children)) * cans_per_soup_for_adults = 16 := by
  sorry

end NUMINAMATH_GPT_remaining_soup_can_feed_adults_l790_79028


namespace NUMINAMATH_GPT_prove_monotonic_increasing_range_l790_79040

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end NUMINAMATH_GPT_prove_monotonic_increasing_range_l790_79040


namespace NUMINAMATH_GPT_soccer_team_total_games_l790_79085

variable (total_games : ℕ)
variable (won_games : ℕ)

-- Given conditions
def team_won_percentage (p : ℝ) := p = 0.60
def team_won_games (w : ℕ) := w = 78

-- The proof goal
theorem soccer_team_total_games 
    (h1 : team_won_percentage 0.60)
    (h2 : team_won_games 78) :
    total_games = 130 :=
sorry

end NUMINAMATH_GPT_soccer_team_total_games_l790_79085


namespace NUMINAMATH_GPT_probability_point_between_C_and_D_l790_79060

theorem probability_point_between_C_and_D :
  ∀ (A B C D E : ℝ), A < B ∧ C < D ∧
  (B - A = 4 * (D - A)) ∧ (B - A = 4 * (B - E)) ∧
  (D - A = C - D) ∧ (C - D = E - C) ∧ (E - C = B - E) →
  (B - A ≠ 0) → 
  (C - D) / (B - A) = 1 / 4 :=
by
  intros A B C D E hAB hNonZero
  sorry

end NUMINAMATH_GPT_probability_point_between_C_and_D_l790_79060


namespace NUMINAMATH_GPT_invalid_inverse_statement_l790_79046

/- Define the statements and their inverses -/

/-- Statement A: Vertical angles are equal. -/
def statement_A : Prop := ∀ {α β : ℝ}, α ≠ β → α = β

/-- Inverse of Statement A: If two angles are equal, then they are vertical angles. -/
def inverse_A : Prop := ∀ {α β : ℝ}, α = β → α ≠ β

/-- Statement B: If |a| = |b|, then a = b. -/
def statement_B (a b : ℝ) : Prop := abs a = abs b → a = b

/-- Inverse of Statement B: If a = b, then |a| = |b|. -/
def inverse_B (a b : ℝ) : Prop := a = b → abs a = abs b

/-- Statement C: If two lines are parallel, then the alternate interior angles are equal. -/
def statement_C (l1 l2 : Prop) : Prop := l1 → l2

/-- Inverse of Statement C: If the alternate interior angles are equal, then the two lines are parallel. -/
def inverse_C (l1 l2 : Prop) : Prop := l2 → l1

/-- Statement D: If a^2 = b^2, then a = b. -/
def statement_D (a b : ℝ) : Prop := a^2 = b^2 → a = b

/-- Inverse of Statement D: If a = b, then a^2 = b^2. -/
def inverse_D (a b : ℝ) : Prop := a = b → a^2 = b^2

/-- The statement that does not have a valid inverse among A, B, C, and D is statement A. -/
theorem invalid_inverse_statement : ¬inverse_A :=
by
sorry

end NUMINAMATH_GPT_invalid_inverse_statement_l790_79046


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l790_79048

theorem angle_in_third_quadrant
  (α : ℝ)
  (k : ℤ)
  (h : (π / 2) + 2 * (↑k) * π < α ∧ α < π + 2 * (↑k) * π) :
  π + 2 * (↑k) * π < (π / 2) + α ∧ (π / 2) + α < (3 * π / 2) + 2 * (↑k) * π :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l790_79048


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l790_79008

theorem distance_between_parallel_lines
  (line1 : ∀ (x y : ℝ), 3*x - 2*y - 1 = 0)
  (line2 : ∀ (x y : ℝ), 3*x - 2*y + 1 = 0) :
  ∃ d : ℝ, d = (2 * Real.sqrt 13) / 13 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l790_79008


namespace NUMINAMATH_GPT_total_animal_sightings_l790_79050

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end NUMINAMATH_GPT_total_animal_sightings_l790_79050


namespace NUMINAMATH_GPT_solutions_to_equation_l790_79062

variable (x : ℝ)

def original_eq : Prop :=
  (3 * x - 9) / (x^2 - 6 * x + 8) = (x + 1) / (x - 2)

theorem solutions_to_equation : (original_eq 1 ∧ original_eq 5) :=
by
  sorry

end NUMINAMATH_GPT_solutions_to_equation_l790_79062


namespace NUMINAMATH_GPT_John_reads_50_pages_per_hour_l790_79049

noncomputable def pages_per_hour (reads_daily hours : ℕ) (total_pages total_weeks : ℕ) : ℕ :=
  let days := total_weeks * 7
  let pages_per_day := total_pages / days
  pages_per_day / reads_daily

theorem John_reads_50_pages_per_hour :
  pages_per_hour 2 2800 4 = 50 := by
  sorry

end NUMINAMATH_GPT_John_reads_50_pages_per_hour_l790_79049


namespace NUMINAMATH_GPT_area_quotient_eq_correct_l790_79087

noncomputable def is_in_plane (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2

def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨ (x ≥ a ∧ y < b ∧ z ≥ c) ∨ (x < a ∧ y ≥ b ∧ z ≥ c)

def in_S (x y z : ℝ) : Prop :=
  is_in_plane x y z ∧ supports x y z 1 (2/3) (1/3)

noncomputable def area_S : ℝ := 
  -- Placeholder for the computed area of S
  sorry

noncomputable def area_T : ℝ := 
  -- Placeholder for the computed area of T
  sorry

theorem area_quotient_eq_correct :
  (area_S / area_T) = (3 / (8 * Real.sqrt 3)) := 
  sorry

end NUMINAMATH_GPT_area_quotient_eq_correct_l790_79087


namespace NUMINAMATH_GPT_ratio_of_sums_l790_79058

open Nat

def sum_multiples_of_3 (n : Nat) : Nat :=
  let m := n / 3
  m * (3 + 3 * m) / 2

def sum_first_n_integers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem ratio_of_sums :
  (sum_multiples_of_3 600) / (sum_first_n_integers 300) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l790_79058


namespace NUMINAMATH_GPT_zacks_friends_l790_79080

theorem zacks_friends (initial_marbles : ℕ) (marbles_kept : ℕ) (marbles_per_friend : ℕ) 
  (h_initial : initial_marbles = 65) (h_kept : marbles_kept = 5) 
  (h_per_friend : marbles_per_friend = 20) : (initial_marbles - marbles_kept) / marbles_per_friend = 3 :=
by
  sorry

end NUMINAMATH_GPT_zacks_friends_l790_79080


namespace NUMINAMATH_GPT_total_oranges_picked_l790_79007

theorem total_oranges_picked (mary_oranges : Nat) (jason_oranges : Nat) (hmary : mary_oranges = 122) (hjason : jason_oranges = 105) : mary_oranges + jason_oranges = 227 := by
  sorry

end NUMINAMATH_GPT_total_oranges_picked_l790_79007


namespace NUMINAMATH_GPT_intersection_A_B_is_1_and_2_l790_79021

def A : Set ℝ := {x | x ^ 2 - 3 * x - 4 < 0}
def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B_is_1_and_2 : A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_is_1_and_2_l790_79021


namespace NUMINAMATH_GPT_expected_value_8_sided_die_l790_79086

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end NUMINAMATH_GPT_expected_value_8_sided_die_l790_79086


namespace NUMINAMATH_GPT_avg_median_max_k_m_r_s_t_l790_79054

theorem avg_median_max_k_m_r_s_t (
  k m r s t : ℕ 
) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : 5 * 16 = k + m + r + s + t)
  (h6 : r = 17) : 
  t = 42 :=
by
  sorry

end NUMINAMATH_GPT_avg_median_max_k_m_r_s_t_l790_79054


namespace NUMINAMATH_GPT_milk_left_after_third_operation_l790_79037

theorem milk_left_after_third_operation :
  ∀ (initial_milk : ℝ), initial_milk > 0 →
  (initial_milk * 0.8 * 0.8 * 0.8 / initial_milk) * 100 = 51.2 :=
by
  intros initial_milk h_initial_milk_pos
  sorry

end NUMINAMATH_GPT_milk_left_after_third_operation_l790_79037


namespace NUMINAMATH_GPT_lcm_14_21_35_l790_79004

-- Define the numbers
def a : ℕ := 14
def b : ℕ := 21
def c : ℕ := 35

-- Define the prime factorizations
def prime_factors_14 : List (ℕ × ℕ) := [(2, 1), (7, 1)]
def prime_factors_21 : List (ℕ × ℕ) := [(3, 1), (7, 1)]
def prime_factors_35 : List (ℕ × ℕ) := [(5, 1), (7, 1)]

-- Prove the least common multiple
theorem lcm_14_21_35 : Nat.lcm (Nat.lcm a b) c = 210 := by
  sorry

end NUMINAMATH_GPT_lcm_14_21_35_l790_79004


namespace NUMINAMATH_GPT_cyclist_is_jean_l790_79055

theorem cyclist_is_jean (x x' y y' : ℝ) (hx : x' = 4 * x) (hy : y = 4 * y') : x < y :=
by
  sorry

end NUMINAMATH_GPT_cyclist_is_jean_l790_79055


namespace NUMINAMATH_GPT_susan_age_in_5_years_l790_79034

-- Definitions of the given conditions
def james_age_in_15_years : ℕ := 37
def years_until_james_is_37 : ℕ := 15
def years_ago_james_twice_janet : ℕ := 8
def susan_born_when_janet_turned : ℕ := 3
def years_to_future_susan_age : ℕ := 5

-- Calculate the current age of people involved
def james_current_age : ℕ := james_age_in_15_years - years_until_james_is_37
def james_age_8_years_ago : ℕ := james_current_age - years_ago_james_twice_janet
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def janet_current_age : ℕ := janet_age_8_years_ago + years_ago_james_twice_janet
def susan_current_age : ℕ := janet_current_age - susan_born_when_janet_turned

-- Prove that Susan will be 17 years old in 5 years
theorem susan_age_in_5_years (james_age_future : james_age_in_15_years = 37)
  (years_until_james_37 : years_until_james_is_37 = 15)
  (years_ago_twice_janet : years_ago_james_twice_janet = 8)
  (susan_born_janet : susan_born_when_janet_turned = 3)
  (years_future : years_to_future_susan_age = 5) :
  susan_current_age + years_to_future_susan_age = 17 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_susan_age_in_5_years_l790_79034


namespace NUMINAMATH_GPT_op_assoc_l790_79095

open Real

def op (x y : ℝ) : ℝ := x + y - x * y

theorem op_assoc (x y z : ℝ) : op (op x y) z = op x (op y z) := by
  sorry

end NUMINAMATH_GPT_op_assoc_l790_79095


namespace NUMINAMATH_GPT_complement_intersection_l790_79073

open Set -- Open the Set namespace to simplify notation for set operations

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def M : Set ℤ := {-1, 0, 1, 3}
def N : Set ℤ := {-2, 0, 2, 3}

theorem complement_intersection : (U \ M) ∩ N = ({-2, 2} : Set ℤ) :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l790_79073


namespace NUMINAMATH_GPT_atlantic_call_charge_l790_79094

theorem atlantic_call_charge :
  let united_base := 6.00
  let united_per_min := 0.25
  let atlantic_base := 12.00
  let same_bill_minutes := 120
  let atlantic_total (charge_per_minute : ℝ) := atlantic_base + charge_per_minute * same_bill_minutes
  let united_total := united_base + united_per_min * same_bill_minutes
  united_total = atlantic_total 0.20 :=
by
  sorry

end NUMINAMATH_GPT_atlantic_call_charge_l790_79094


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l790_79089

theorem average_gas_mileage_round_trip :
  let distance_to_conference := 150
  let distance_return_trip := 150
  let mpg_sedan := 25
  let mpg_hybrid := 40
  let total_distance := distance_to_conference + distance_return_trip
  let gas_used_sedan := distance_to_conference / mpg_sedan
  let gas_used_hybrid := distance_return_trip / mpg_hybrid
  let total_gas_used := gas_used_sedan + gas_used_hybrid
  let average_gas_mileage := total_distance / total_gas_used
  average_gas_mileage = 31 := by
    sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l790_79089


namespace NUMINAMATH_GPT_quadratic_function_proof_l790_79013

noncomputable def quadratic_function_condition (a b c : ℝ) :=
  ∀ x : ℝ, ((-3 ≤ x ∧ x ≤ 1) → (a * x^2 + b * x + c) ≤ 0) ∧
           ((x < -3 ∨ 1 < x) → (a * x^2 + b * x + c) > 0) ∧
           (a * 2^2 + b * 2 + c) = 5

theorem quadratic_function_proof (a b c : ℝ) (m : ℝ)
  (h : quadratic_function_condition a b c) :
  (a = 1 ∧ b = 2 ∧ c = -3) ∧ (m ≥ -7/9 ↔ ∃ x : ℝ, a * x^2 + b * x + c = 9 * m + 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_proof_l790_79013


namespace NUMINAMATH_GPT_calculate_expression_l790_79006

theorem calculate_expression :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l790_79006


namespace NUMINAMATH_GPT_us2_eq_3958_div_125_l790_79020

-- Definitions based on conditions
def t (x : ℚ) : ℚ := 5 * x - 12
def s (t_x : ℚ) : ℚ := (2 : ℚ) ^ 2 + 3 * 2 - 2
def u (s_t_x : ℚ) : ℚ := (14 : ℚ) / 5 ^ 3 + 2 * (14 / 5) ^ 2 - 14 / 5 + 4

-- Prove that u(s(2)) = 3958 / 125
theorem us2_eq_3958_div_125 : u (s (2)) = 3958 / 125 := by
  sorry

end NUMINAMATH_GPT_us2_eq_3958_div_125_l790_79020


namespace NUMINAMATH_GPT_estimate_yellow_balls_l790_79045

theorem estimate_yellow_balls (m : ℕ) (h1: (5 : ℝ) / (5 + m) = 0.2) : m = 20 :=
  sorry

end NUMINAMATH_GPT_estimate_yellow_balls_l790_79045
