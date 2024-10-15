import Mathlib

namespace NUMINAMATH_GPT_total_savings_eighteen_l2086_208654

theorem total_savings_eighteen :
  let fox_price := 15
  let pony_price := 18
  let discount_rate_sum := 50
  let fox_quantity := 3
  let pony_quantity := 2
  let pony_discount_rate := 50
  let total_price_without_discount := (fox_quantity * fox_price) + (pony_quantity * pony_price)
  let discounted_pony_price := (pony_price * (1 - (pony_discount_rate / 100)))
  let total_price_with_discount := (fox_quantity * fox_price) + (pony_quantity * discounted_pony_price)
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 18 :=
by sorry

end NUMINAMATH_GPT_total_savings_eighteen_l2086_208654


namespace NUMINAMATH_GPT_person_age_in_1893_l2086_208677

theorem person_age_in_1893 
    (x y : ℕ)
    (h1 : 0 ≤ x ∧ x < 10)
    (h2 : 0 ≤ y ∧ y < 10)
    (h3 : 1 + 8 + x + y = 93 - 10 * x - y) : 
    1893 - (1800 + 10 * x + y) = 24 :=
by
  sorry

end NUMINAMATH_GPT_person_age_in_1893_l2086_208677


namespace NUMINAMATH_GPT_roses_picked_second_time_l2086_208626

-- Define the initial conditions
def initial_roses : ℝ := 37.0
def first_pick : ℝ := 16.0
def total_roses_after_second_picking : ℝ := 72.0

-- Define the calculation after the first picking
def roses_after_first_picking : ℝ := initial_roses + first_pick

-- The Lean statement to prove the number of roses picked the second time
theorem roses_picked_second_time : total_roses_after_second_picking - roses_after_first_picking = 19.0 := 
by
  -- Use the facts stated in the conditions
  sorry

end NUMINAMATH_GPT_roses_picked_second_time_l2086_208626


namespace NUMINAMATH_GPT_chessboard_grains_difference_l2086_208642

open BigOperators

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), grains_on_square k

theorem chessboard_grains_difference : 
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := 
by 
  -- Proof of the statement goes here.
  sorry

end NUMINAMATH_GPT_chessboard_grains_difference_l2086_208642


namespace NUMINAMATH_GPT_ones_digit_of_73_pow_351_l2086_208624

theorem ones_digit_of_73_pow_351 : 
  (73 ^ 351) % 10 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_ones_digit_of_73_pow_351_l2086_208624


namespace NUMINAMATH_GPT_sum_1026_is_2008_l2086_208660

def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let groups_sum : ℕ := (n * n)
    let extra_2s := (2008 - groups_sum) / 2
    (n * (n + 1)) / 2 + extra_2s

theorem sum_1026_is_2008 : sequence_sum 1026 = 2008 :=
  sorry

end NUMINAMATH_GPT_sum_1026_is_2008_l2086_208660


namespace NUMINAMATH_GPT_number_of_positive_area_triangles_l2086_208613

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end NUMINAMATH_GPT_number_of_positive_area_triangles_l2086_208613


namespace NUMINAMATH_GPT_average_of_two_integers_l2086_208612

theorem average_of_two_integers {A B C D : ℕ} (h1 : A + B + C + D = 200) (h2 : C ≤ 130) : (A + B) / 2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_of_two_integers_l2086_208612


namespace NUMINAMATH_GPT_cone_volume_l2086_208691

theorem cone_volume (r h : ℝ) (h_cylinder_vol : π * r^2 * h = 72 * π) : 
  (1 / 3) * π * r^2 * (h / 2) = 12 * π := by
  sorry

end NUMINAMATH_GPT_cone_volume_l2086_208691


namespace NUMINAMATH_GPT_average_price_of_goat_l2086_208676

theorem average_price_of_goat (total_cost_goats_hens : ℕ) (num_goats num_hens : ℕ) (avg_price_hen : ℕ)
  (h1 : total_cost_goats_hens = 2500) (h2 : num_hens = 10) (h3 : avg_price_hen = 50) (h4 : num_goats = 5) :
  (total_cost_goats_hens - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end NUMINAMATH_GPT_average_price_of_goat_l2086_208676


namespace NUMINAMATH_GPT_part_one_min_f_value_part_two_range_a_l2086_208607

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x + a|

theorem part_one_min_f_value (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≥ (3/2) :=
  sorry

theorem part_two_range_a (a : ℝ) : (11/2 < a) ∧ (a < 4.5) :=
  sorry

end NUMINAMATH_GPT_part_one_min_f_value_part_two_range_a_l2086_208607


namespace NUMINAMATH_GPT_largest_B_div_by_4_l2086_208633

-- Given conditions
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- The seven-digit integer is 4B6792X
def number (B X : ℕ) : ℕ := 4000000 + B * 100000 + 60000 + 7000 + 900 + 20 + X

-- Problem statement: Prove that the largest digit B so that the seven-digit integer 4B6792X is divisible by 4
theorem largest_B_div_by_4 
(B X : ℕ) 
(hX : is_digit X)
(div_4 : divisible_by_4 (number B X)) : 
B = 9 := sorry

end NUMINAMATH_GPT_largest_B_div_by_4_l2086_208633


namespace NUMINAMATH_GPT_rabbit_clearing_10_square_yards_per_day_l2086_208637

noncomputable def area_cleared_by_one_rabbit_per_day (length width : ℕ) (rabbits : ℕ) (days : ℕ) : ℕ :=
  (length * width) / (3 * 3 * rabbits * days)

theorem rabbit_clearing_10_square_yards_per_day :
  area_cleared_by_one_rabbit_per_day 200 900 100 20 = 10 :=
by sorry

end NUMINAMATH_GPT_rabbit_clearing_10_square_yards_per_day_l2086_208637


namespace NUMINAMATH_GPT_solution_set_eq_l2086_208683

theorem solution_set_eq : { x : ℝ | |x| * (x - 2) ≥ 0 } = { x : ℝ | x ≥ 2 ∨ x = 0 } := by
  sorry

end NUMINAMATH_GPT_solution_set_eq_l2086_208683


namespace NUMINAMATH_GPT_expression_value_correct_l2086_208679

theorem expression_value_correct (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : -a - b^3 + a * b = -11 := by
  sorry

end NUMINAMATH_GPT_expression_value_correct_l2086_208679


namespace NUMINAMATH_GPT_students_6_to_8_hours_study_l2086_208655

-- Condition: 100 students were surveyed
def total_students : ℕ := 100

-- Hypothetical function representing the number of students studying for a specific range of hours based on the histogram
def histogram_students (lower_bound upper_bound : ℕ) : ℕ :=
  sorry  -- this would be defined based on actual histogram data

-- Question: Prove the number of students who studied for 6 to 8 hours
theorem students_6_to_8_hours_study : histogram_students 6 8 = 30 :=
  sorry -- the expected answer based on the histogram data

end NUMINAMATH_GPT_students_6_to_8_hours_study_l2086_208655


namespace NUMINAMATH_GPT_range_of_alpha_l2086_208640

theorem range_of_alpha :
  ∀ P : ℝ, 
  (∃ y : ℝ, y = 4 / (Real.exp P + 1)) →
  (∃ α : ℝ, α = Real.arctan (4 / (Real.exp P + 2 + 1 / Real.exp P)) ∧ (Real.tan α) ∈ Set.Ico (-1) 0) → 
  Set.Ico (3 * Real.pi / 4) Real.pi :=
by
  sorry

end NUMINAMATH_GPT_range_of_alpha_l2086_208640


namespace NUMINAMATH_GPT_factorize_xy_squared_minus_x_l2086_208662

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end NUMINAMATH_GPT_factorize_xy_squared_minus_x_l2086_208662


namespace NUMINAMATH_GPT_sum_of_integers_l2086_208649

theorem sum_of_integers {n : ℤ} (h : n + 2 = 9) : n + (n + 1) + (n + 2) = 24 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l2086_208649


namespace NUMINAMATH_GPT_polygonal_number_8_8_l2086_208648

-- Definitions based on conditions
def triangular_number (n : ℕ) : ℕ := (n^2 + n) / 2
def square_number (n : ℕ) : ℕ := n^2
def pentagonal_number (n : ℕ) : ℕ := (3 * n^2 - n) / 2
def hexagonal_number (n : ℕ) : ℕ := (4 * n^2 - 2 * n) / 2

-- General formula for k-sided polygonal number
def polygonal_number (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

-- The proposition to be proved
theorem polygonal_number_8_8 : polygonal_number 8 8 = 176 := by
  sorry

end NUMINAMATH_GPT_polygonal_number_8_8_l2086_208648


namespace NUMINAMATH_GPT_dogs_eat_each_day_l2086_208699

theorem dogs_eat_each_day (h1 : 0.125 + 0.125 = 0.25) : true := by
  sorry

end NUMINAMATH_GPT_dogs_eat_each_day_l2086_208699


namespace NUMINAMATH_GPT_danny_initial_wrappers_l2086_208666

def initial_wrappers (total_wrappers: ℕ) (found_wrappers: ℕ): ℕ :=
  total_wrappers - found_wrappers

theorem danny_initial_wrappers : initial_wrappers 57 30 = 27 :=
by
  exact rfl

end NUMINAMATH_GPT_danny_initial_wrappers_l2086_208666


namespace NUMINAMATH_GPT_gcd_176_88_l2086_208620

theorem gcd_176_88 : Nat.gcd 176 88 = 88 :=
by
  sorry

end NUMINAMATH_GPT_gcd_176_88_l2086_208620


namespace NUMINAMATH_GPT_inequality_hold_l2086_208670

theorem inequality_hold (n : ℕ) (h1 : n > 1) : 1 + n * 2^((n - 1 : ℕ) / 2) < 2^n :=
by
  sorry

end NUMINAMATH_GPT_inequality_hold_l2086_208670


namespace NUMINAMATH_GPT_max_value_of_3x_plus_4y_l2086_208647

theorem max_value_of_3x_plus_4y (x y : ℝ) 
(h : x^2 + y^2 = 14 * x + 6 * y + 6) : 
3 * x + 4 * y ≤ 73 := 
sorry

end NUMINAMATH_GPT_max_value_of_3x_plus_4y_l2086_208647


namespace NUMINAMATH_GPT_cyclist_distance_l2086_208621

theorem cyclist_distance
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1) * (3 * t / 4))
  (h3 : d = (x - 1) * (t + 3)) :
  d = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_cyclist_distance_l2086_208621


namespace NUMINAMATH_GPT_negative_linear_correlation_l2086_208650

theorem negative_linear_correlation (x y : ℝ) (h : y = 3 - 2 * x) : 
  ∃ c : ℝ, c < 0 ∧ y = 3 + c * x := 
by  
  sorry

end NUMINAMATH_GPT_negative_linear_correlation_l2086_208650


namespace NUMINAMATH_GPT_trigonometric_expression_l2086_208636

theorem trigonometric_expression
  (α : ℝ)
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) :
  2 + (2 / 3) * Real.sin α ^ 2 + (1 / 4) * Real.cos α ^ 2 = 21 / 8 := 
by sorry

end NUMINAMATH_GPT_trigonometric_expression_l2086_208636


namespace NUMINAMATH_GPT_pump_fills_tank_without_leak_l2086_208644

theorem pump_fills_tank_without_leak (T : ℝ) (h1 : 1 / 12 = 1 / T - 1 / 12) : T = 6 :=
sorry

end NUMINAMATH_GPT_pump_fills_tank_without_leak_l2086_208644


namespace NUMINAMATH_GPT_trapezoid_midsegment_inscribed_circle_l2086_208673

theorem trapezoid_midsegment_inscribed_circle (P : ℝ) (hP : P = 40) 
    (inscribed : Π (a b c d : ℝ), a + b = c + d) : 
    (∃ (c d : ℝ), (c + d) / 2 = 10) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_midsegment_inscribed_circle_l2086_208673


namespace NUMINAMATH_GPT_shaded_area_percentage_l2086_208661

theorem shaded_area_percentage (total_area shaded_area : ℕ) (h_total : total_area = 49) (h_shaded : shaded_area = 33) : 
  (shaded_area : ℚ) / total_area = 33 / 49 := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_percentage_l2086_208661


namespace NUMINAMATH_GPT_three_digit_multiples_of_seven_l2086_208656

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end NUMINAMATH_GPT_three_digit_multiples_of_seven_l2086_208656


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l2086_208669

variable (a b c d : ℝ)

theorem ratio_of_a_to_b (h1 : c = 0.20 * a) (h2 : c = 0.10 * b) : a = (1 / 2) * b :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l2086_208669


namespace NUMINAMATH_GPT_diane_stamp_combinations_l2086_208605

/-- Define the types of stamps Diane has --/
def diane_stamps : List ℕ := [1, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8]

/-- Define the condition for the correct number of different arrangements to sum exactly to 12 cents -/
noncomputable def count_arrangements (stamps : List ℕ) (sum : ℕ) : ℕ :=
  -- Implementation of the counting function goes here
  sorry

/-- Prove that the number of distinct arrangements to make exactly 12 cents is 13 --/
theorem diane_stamp_combinations : count_arrangements diane_stamps 12 = 13 :=
  sorry

end NUMINAMATH_GPT_diane_stamp_combinations_l2086_208605


namespace NUMINAMATH_GPT_stamps_count_l2086_208690

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end NUMINAMATH_GPT_stamps_count_l2086_208690


namespace NUMINAMATH_GPT_power_first_digits_l2086_208696

theorem power_first_digits (n : ℕ) (h1 : ∀ k : ℕ, n ≠ 10^k) : ∃ j k : ℕ, 1973 ≤ n^j / 10^k ∧ n^j / 10^k < 1974 := by
  sorry

end NUMINAMATH_GPT_power_first_digits_l2086_208696


namespace NUMINAMATH_GPT_weekly_car_mileage_l2086_208671

-- Definitions of the conditions
def dist_school := 2.5 
def dist_market := 2 
def school_days := 4
def school_trips_per_day := 2
def market_trips_per_week := 1

-- Proof statement
theorem weekly_car_mileage : 
  4 * 2 * (2.5 * 2) + (1 * (2 * 2)) = 44 :=
by
  -- The goal is to prove that 4 days of 2 round trips to school plus 1 round trip to market equals 44 miles
  sorry

end NUMINAMATH_GPT_weekly_car_mileage_l2086_208671


namespace NUMINAMATH_GPT_sum_of_remainders_mod_13_l2086_208653

theorem sum_of_remainders_mod_13 
  (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_13_l2086_208653


namespace NUMINAMATH_GPT_min_units_for_profitability_profitability_during_epidemic_l2086_208695

-- Conditions
def assembly_line_cost : ℝ := 1.8
def selling_price_per_product : ℝ := 0.1
def max_annual_output : ℕ := 100

noncomputable def production_cost (x : ℕ) : ℝ := 5 + 135 / (x + 1)

-- Part 1: Prove Minimum x for profitability
theorem min_units_for_profitability (x : ℕ) :
  (10 - (production_cost x)) * x - assembly_line_cost > 0 ↔ x ≥ 63 := sorry

-- Part 2: Profitability and max profit output during epidemic
theorem profitability_during_epidemic (x : ℕ) :
  (60 < x ∧ x ≤ max_annual_output) → 
  ((10 - (production_cost x)) * 60 - (x - 60) - assembly_line_cost > 0) ↔ x = 89 := sorry

end NUMINAMATH_GPT_min_units_for_profitability_profitability_during_epidemic_l2086_208695


namespace NUMINAMATH_GPT_paul_account_balance_after_transactions_l2086_208658

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end NUMINAMATH_GPT_paul_account_balance_after_transactions_l2086_208658


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_in_progressions_l2086_208639

theorem sum_of_inserted_numbers_in_progressions (x y : ℝ) (hx : 4 * (y / x) = x) (hy : 2 * y = x + 64) :
  x + y = 131 + 3 * Real.sqrt 129 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_in_progressions_l2086_208639


namespace NUMINAMATH_GPT_original_number_of_workers_l2086_208604

-- Definitions of the conditions given in the problem
def workers_days (W : ℕ) : ℕ := 35
def additional_workers : ℕ := 10
def reduced_days : ℕ := 10

-- The main theorem we need to prove
theorem original_number_of_workers (W : ℕ) (A : ℕ) 
  (h1 : W * workers_days W = (W + additional_workers) * (workers_days W - reduced_days)) :
  W = 25 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_workers_l2086_208604


namespace NUMINAMATH_GPT_at_least_12_lyamziks_rowed_l2086_208684

-- Define the lyamziks, their weights, and constraints
def LyamzikWeight1 : ℕ := 7
def LyamzikWeight2 : ℕ := 14
def LyamzikWeight3 : ℕ := 21
def LyamzikWeight4 : ℕ := 28
def totalLyamziks : ℕ := LyamzikWeight1 + LyamzikWeight2 + LyamzikWeight3 + LyamzikWeight4
def boatCapacity : ℕ := 10
def maxRowsPerLyamzik : ℕ := 2

-- Question to prove
theorem at_least_12_lyamziks_rowed : totalLyamziks ≥ 12 :=
  by sorry


end NUMINAMATH_GPT_at_least_12_lyamziks_rowed_l2086_208684


namespace NUMINAMATH_GPT_arithmetic_mean_three_fractions_l2086_208682

theorem arithmetic_mean_three_fractions :
  let a := (5 : ℚ) / 8
  let b := (7 : ℚ) / 8
  let c := (3 : ℚ) / 4
  (a + b) / 2 = c :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_three_fractions_l2086_208682


namespace NUMINAMATH_GPT_sugar_spilled_l2086_208619

-- Define the initial amount of sugar and the amount left
def initial_sugar : ℝ := 9.8
def remaining_sugar : ℝ := 4.6

-- State the problem as a theorem
theorem sugar_spilled :
  initial_sugar - remaining_sugar = 5.2 := 
sorry

end NUMINAMATH_GPT_sugar_spilled_l2086_208619


namespace NUMINAMATH_GPT_probability_all_same_room_probability_at_least_two_same_room_l2086_208645

/-- 
  Given that there are three people and each person is assigned to one of four rooms with equal probability,
  let P1 be the probability that all three people are assigned to the same room,
  and let P2 be the probability that at least two people are assigned to the same room.
  We need to prove:
  1. P1 = 1 / 16
  2. P2 = 5 / 8
-/
noncomputable def P1 : ℚ := sorry

noncomputable def P2 : ℚ := sorry

theorem probability_all_same_room :
  P1 = 1 / 16 :=
sorry

theorem probability_at_least_two_same_room :
  P2 = 5 / 8 :=
sorry

end NUMINAMATH_GPT_probability_all_same_room_probability_at_least_two_same_room_l2086_208645


namespace NUMINAMATH_GPT_Heechul_has_most_books_l2086_208643

namespace BookCollection

variables (Heejin Heechul Dongkyun : ℕ)

theorem Heechul_has_most_books (h_h : ℕ) (h_j : ℕ) (d : ℕ) 
  (h_h_eq : h_h = h_j + 2) (d_lt_h_j : d < h_j) : 
  h_h > h_j ∧ h_h > d := 
by
  sorry

end BookCollection

end NUMINAMATH_GPT_Heechul_has_most_books_l2086_208643


namespace NUMINAMATH_GPT_cannot_have_1970_minus_signs_in_grid_l2086_208681

theorem cannot_have_1970_minus_signs_in_grid :
  ∀ (k l : ℕ), k ≤ 100 → l ≤ 100 → (k+l)*50 - k*l ≠ 985 :=
by
  intros k l hk hl
  sorry

end NUMINAMATH_GPT_cannot_have_1970_minus_signs_in_grid_l2086_208681


namespace NUMINAMATH_GPT_interval_of_expression_l2086_208693

theorem interval_of_expression (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧ 
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
by sorry

end NUMINAMATH_GPT_interval_of_expression_l2086_208693


namespace NUMINAMATH_GPT_speed_of_first_boy_l2086_208603

-- Variables for speeds and time
variables (v : ℝ) (t : ℝ) (d : ℝ)

-- Given conditions
def initial_conditions := 
  v > 0 ∧ 
  7.5 > 0 ∧ 
  t = 10 ∧ 
  d = 20

-- Theorem statement with the conditions and the expected answer
theorem speed_of_first_boy
  (h : initial_conditions v t d) : 
  v = 9.5 :=
sorry

end NUMINAMATH_GPT_speed_of_first_boy_l2086_208603


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2086_208627

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = 3) :
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2086_208627


namespace NUMINAMATH_GPT_evaluate_expression_l2086_208680

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2086_208680


namespace NUMINAMATH_GPT_fraction_of_oranges_is_correct_l2086_208610

variable (O P A : ℕ)
variable (total_fruit : ℕ := 56)

theorem fraction_of_oranges_is_correct:
  (A = 35) →
  (P = O / 2) →
  (A = 5 * P) →
  (O + P + A = total_fruit) →
  (O / total_fruit = 1 / 4) :=
by
  -- proof to be filled in 
  sorry

end NUMINAMATH_GPT_fraction_of_oranges_is_correct_l2086_208610


namespace NUMINAMATH_GPT_ratio_of_Katie_to_Cole_l2086_208601

variable (K C : ℕ)

theorem ratio_of_Katie_to_Cole (h1 : 3 * K = 84) (h2 : C = 7) : K / C = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_Katie_to_Cole_l2086_208601


namespace NUMINAMATH_GPT_calculate_Y_payment_l2086_208628

-- Define the known constants
def total_payment : ℝ := 590
def x_to_y_ratio : ℝ := 1.2

-- Main theorem statement, asserting the value of Y's payment
theorem calculate_Y_payment (Y : ℝ) (X : ℝ) 
  (h1 : X = x_to_y_ratio * Y) 
  (h2 : X + Y = total_payment) : 
  Y = 268.18 :=
by
  sorry

end NUMINAMATH_GPT_calculate_Y_payment_l2086_208628


namespace NUMINAMATH_GPT_fish_eaten_by_new_fish_l2086_208692

def initial_original_fish := 14
def added_fish := 2
def exchange_new_fish := 3
def total_fish_now := 11

theorem fish_eaten_by_new_fish : initial_original_fish - (total_fish_now - exchange_new_fish) = 6 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_fish_eaten_by_new_fish_l2086_208692


namespace NUMINAMATH_GPT_gcf_7fact_8fact_l2086_208657

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end NUMINAMATH_GPT_gcf_7fact_8fact_l2086_208657


namespace NUMINAMATH_GPT_slices_per_pizza_l2086_208631

def num_pizzas : ℕ := 2
def total_slices : ℕ := 16

theorem slices_per_pizza : total_slices / num_pizzas = 8 := by
  sorry

end NUMINAMATH_GPT_slices_per_pizza_l2086_208631


namespace NUMINAMATH_GPT_Ray_has_4_nickels_left_l2086_208625

def Ray_initial_cents := 95
def Ray_cents_to_Peter := 25
def Ray_cents_to_Randi := 2 * Ray_cents_to_Peter

-- There are 5 cents in each nickel
def cents_per_nickel := 5

-- Nickels Ray originally has
def Ray_initial_nickels := Ray_initial_cents / cents_per_nickel
-- Nickels given to Peter
def Ray_nickels_to_Peter := Ray_cents_to_Peter / cents_per_nickel
-- Nickels given to Randi
def Ray_nickels_to_Randi := Ray_cents_to_Randi / cents_per_nickel
-- Total nickels given away
def Ray_nickels_given_away := Ray_nickels_to_Peter + Ray_nickels_to_Randi
-- Nickels left with Ray
def Ray_nickels_left := Ray_initial_nickels - Ray_nickels_given_away

theorem Ray_has_4_nickels_left :
  Ray_nickels_left = 4 :=
by
  sorry

end NUMINAMATH_GPT_Ray_has_4_nickels_left_l2086_208625


namespace NUMINAMATH_GPT_motorboat_time_to_C_l2086_208638

variables (r s p t_B : ℝ)

-- Condition declarations
def kayak_speed := r + s
def motorboat_speed := p
def meeting_time := 12

-- Problem statement: to prove the time it took for the motorboat to reach dock C before turning back
theorem motorboat_time_to_C :
  (2 * r + s) * t_B = r * 12 + s * 6 → t_B = (r * 12 + s * 6) / (2 * r + s) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_motorboat_time_to_C_l2086_208638


namespace NUMINAMATH_GPT_number_of_students_in_first_group_l2086_208629

def total_students : ℕ := 24
def second_group : ℕ := 8
def third_group : ℕ := 7
def fourth_group : ℕ := 4
def summed_other_groups : ℕ := second_group + third_group + fourth_group
def students_first_group : ℕ := total_students - summed_other_groups

theorem number_of_students_in_first_group :
  students_first_group = 5 :=
by
  -- proof required here
  sorry

end NUMINAMATH_GPT_number_of_students_in_first_group_l2086_208629


namespace NUMINAMATH_GPT_new_deck_card_count_l2086_208617

-- Define the conditions
def cards_per_time : ℕ := 30
def times_per_week : ℕ := 3
def weeks : ℕ := 11
def decks : ℕ := 18
def total_cards_tear_per_week : ℕ := cards_per_time * times_per_week
def total_cards_tear : ℕ := total_cards_tear_per_week * weeks
def total_cards_in_decks (cards_per_deck : ℕ) : ℕ := decks * cards_per_deck

-- Define the theorem we need to prove
theorem new_deck_card_count :
  ∃ (x : ℕ), total_cards_in_decks x = total_cards_tear ↔ x = 55 := by
  sorry

end NUMINAMATH_GPT_new_deck_card_count_l2086_208617


namespace NUMINAMATH_GPT_find_f_2008_l2086_208618

variable (f : ℝ → ℝ) 
variable (g : ℝ → ℝ) -- g is the inverse of f

def satisfies_conditions (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) ∧ 
  (f 9 = 18) ∧ (∀ x : ℝ, g (x + 1) = (f (x + 1)))

theorem find_f_2008 (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : satisfies_conditions f g) : f 2008 = -1981 :=
sorry

end NUMINAMATH_GPT_find_f_2008_l2086_208618


namespace NUMINAMATH_GPT_simplify_expression_l2086_208664

theorem simplify_expression (x : ℝ) :
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) = x^4 - 1 :=
  by 
    sorry

end NUMINAMATH_GPT_simplify_expression_l2086_208664


namespace NUMINAMATH_GPT_bianca_total_books_l2086_208663

theorem bianca_total_books (shelves_mystery shelves_picture books_per_shelf : ℕ) 
  (h1 : shelves_mystery = 5) 
  (h2 : shelves_picture = 4) 
  (h3 : books_per_shelf = 8) : 
  (shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf) = 72 := 
by 
  sorry

end NUMINAMATH_GPT_bianca_total_books_l2086_208663


namespace NUMINAMATH_GPT_tickets_needed_l2086_208634

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end NUMINAMATH_GPT_tickets_needed_l2086_208634


namespace NUMINAMATH_GPT_range_of_phi_l2086_208652

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + 2 * φ)

theorem range_of_phi :
  ∀ φ : ℝ,
  (0 < φ) ∧ (φ < π / 2) →
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 → g x φ ≤ g (x + π/6) φ) →
  (∃ x : ℝ, -π/6 < x ∧ x < 0 ∧ g x φ = 0) →
  φ ∈ Set.Ioc (π / 4) (π / 3) := 
by
  intros φ h1 h2 h3
  sorry

end NUMINAMATH_GPT_range_of_phi_l2086_208652


namespace NUMINAMATH_GPT_second_tap_empty_time_l2086_208606

theorem second_tap_empty_time :
  ∃ T : ℝ, (1 / 4 - 1 / T = 3 / 28) → T = 7 :=
by
  sorry

end NUMINAMATH_GPT_second_tap_empty_time_l2086_208606


namespace NUMINAMATH_GPT_movie_theorem_l2086_208609

variables (A B C D : Prop)

theorem movie_theorem 
  (h1 : (A → B))
  (h2 : (B → C))
  (h3 : (C → A))
  (h4 : (D → B)) 
  : ¬D := 
by
  sorry

end NUMINAMATH_GPT_movie_theorem_l2086_208609


namespace NUMINAMATH_GPT_student_marks_l2086_208608

theorem student_marks
(M P C : ℕ) -- the marks of Mathematics, Physics, and Chemistry are natural numbers
(h1 : C = P + 20)  -- Chemistry is 20 marks more than Physics
(h2 : (M + C) / 2 = 30)  -- The average marks in Mathematics and Chemistry is 30
: M + P = 40 := 
sorry

end NUMINAMATH_GPT_student_marks_l2086_208608


namespace NUMINAMATH_GPT_circle_equation_through_points_l2086_208678

theorem circle_equation_through_points (A B: ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (1, -1)) (hB : B = (-1, 1)) (hC : C.1 + C.2 = 2)
  (hAC : dist A C = dist B C) :
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_through_points_l2086_208678


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2086_208646

theorem simplify_and_evaluate : 
  ∀ (x y : ℚ), x = 1 / 2 → y = 2 / 3 →
  ((x - 2 * y)^2 + (x - 2 * y) * (x + 2 * y) - 3 * x * (2 * x - y)) / (2 * x) = -4 / 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2086_208646


namespace NUMINAMATH_GPT_rowing_speed_downstream_l2086_208659

/--
A man can row upstream at 25 kmph and downstream at a certain speed. 
The speed of the man in still water is 30 kmph. 
Prove that the speed of the man rowing downstream is 35 kmph.
-/
theorem rowing_speed_downstream (V_u V_sw V_s V_d : ℝ)
  (h1 : V_u = 25) 
  (h2 : V_sw = 30) 
  (h3 : V_u = V_sw - V_s) 
  (h4 : V_d = V_sw + V_s) :
  V_d = 35 :=
by
  sorry

end NUMINAMATH_GPT_rowing_speed_downstream_l2086_208659


namespace NUMINAMATH_GPT_ratio_of_r_l2086_208651

theorem ratio_of_r
  (total : ℕ) (r_amount : ℕ) (pq_amount : ℕ)
  (h_total : total = 7000 )
  (h_r_amount : r_amount = 2800 )
  (h_pq_amount : pq_amount = total - r_amount) :
  (r_amount / Nat.gcd r_amount pq_amount, pq_amount / Nat.gcd r_amount pq_amount) = (2, 3) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_r_l2086_208651


namespace NUMINAMATH_GPT_lcm_of_36_and_105_l2086_208672

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_GPT_lcm_of_36_and_105_l2086_208672


namespace NUMINAMATH_GPT_neg_prop_p_l2086_208698

-- Define the function f as a real-valued function
variable (f : ℝ → ℝ)

-- Definitions for the conditions in the problem
def prop_p := ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

-- Theorem stating the negation of proposition p
theorem neg_prop_p : ¬prop_p f ↔ ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by 
  sorry

end NUMINAMATH_GPT_neg_prop_p_l2086_208698


namespace NUMINAMATH_GPT_rational_linear_function_l2086_208667

theorem rational_linear_function (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
sorry

end NUMINAMATH_GPT_rational_linear_function_l2086_208667


namespace NUMINAMATH_GPT_initial_books_calculation_l2086_208689

-- Definitions based on conditions
def total_books : ℕ := 77
def additional_books : ℕ := 23

-- Statement of the problem
theorem initial_books_calculation : total_books - additional_books = 54 :=
by
  sorry

end NUMINAMATH_GPT_initial_books_calculation_l2086_208689


namespace NUMINAMATH_GPT_calculate_expression_l2086_208623

theorem calculate_expression :
  |1 - Real.sqrt 2| + (1/2)^(-2 : ℤ) - (Real.pi - 2023)^0 = Real.sqrt 2 + 2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2086_208623


namespace NUMINAMATH_GPT_number_of_pairs_l2086_208615

theorem number_of_pairs (h : ∀ (a : ℝ) (b : ℕ), 0 < a → 2 ≤ b ∧ b ≤ 200 → (Real.log a / Real.log b) ^ 2017 = Real.log (a ^ 2017) / Real.log b) :
  ∃ n, n = 597 ∧ ∀ b : ℕ, 2 ≤ b ∧ b ≤ 200 → 
    ∃ a1 a2 a3 : ℝ, 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 
      (Real.log a1 / Real.log b) = 0 ∧ 
      (Real.log a2 / Real.log b) = 2017^((1:ℝ)/2016) ∧ 
      (Real.log a3 / Real.log b) = -2017^((1:ℝ)/2016) :=
sorry

end NUMINAMATH_GPT_number_of_pairs_l2086_208615


namespace NUMINAMATH_GPT_different_pronunciation_in_group_C_l2086_208616

theorem different_pronunciation_in_group_C :
  let groupC := [("戏谑", "xuè"), ("虐待", "nüè"), ("瘠薄", "jí"), ("脊梁", "jǐ"), ("赝品", "yàn"), ("义愤填膺", "yīng")]
  ∀ {a : String} {b : String}, (a, b) ∈ groupC → a ≠ b :=
by
  intro groupC h
  sorry

end NUMINAMATH_GPT_different_pronunciation_in_group_C_l2086_208616


namespace NUMINAMATH_GPT_square_possible_length_l2086_208635

theorem square_possible_length (sticks : Finset ℕ) (H : sticks = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ s, s = 9 ∧
  ∃ (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a + b + c = 9 :=
by
  sorry

end NUMINAMATH_GPT_square_possible_length_l2086_208635


namespace NUMINAMATH_GPT_value_of_a_l2086_208694

theorem value_of_a (a : ℝ) : 
  (∀ (x : ℝ), (x < -4 ∨ x > 5) → x^2 + a * x + 20 > 0) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2086_208694


namespace NUMINAMATH_GPT_vasya_wins_l2086_208674

-- Definition of the game and players
inductive Player
| Vasya : Player
| Petya : Player

-- Define the problem conditions
structure Game where
  initial_piles : ℕ := 1      -- Initially, there is one pile
  players_take_turns : Bool := true
  take_or_divide : Bool := true
  remove_last_wins : Bool := true
  vasya_first_but_cannot_take_initially : Bool := true

-- Define the function to determine the winner
def winner_of_game (g : Game) : Player :=
  if g.initial_piles = 1 ∧ g.vasya_first_but_cannot_take_initially then Player.Vasya else Player.Petya

-- Define the theorem stating Vasya will win given the game conditions
theorem vasya_wins : ∀ (g : Game), g = {
    initial_piles := 1,
    players_take_turns := true,
    take_or_divide := true,
    remove_last_wins := true,
    vasya_first_but_cannot_take_initially := true
} → winner_of_game g = Player.Vasya := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_vasya_wins_l2086_208674


namespace NUMINAMATH_GPT_distance_to_larger_cross_section_l2086_208687

theorem distance_to_larger_cross_section
    (A B : ℝ)
    (a b : ℝ)
    (d : ℝ)
    (h : ℝ)
    (h_eq : h = 30):
  A = 300 * Real.sqrt 2 → 
  B = 675 * Real.sqrt 2 → 
  a = Real.sqrt (A / B) → 
  b = d / (1 - a) → 
  d = 10 → 
  b = h :=
by
  sorry

end NUMINAMATH_GPT_distance_to_larger_cross_section_l2086_208687


namespace NUMINAMATH_GPT_initial_deck_card_count_l2086_208611

-- Define the initial conditions
def initial_red_probability (r b : ℕ) : Prop := r * 4 = r + b
def added_black_probability (r b : ℕ) : Prop := r * 5 = 4 * r + 6

theorem initial_deck_card_count (r b : ℕ) (h1 : initial_red_probability r b) (h2 : added_black_probability r b) : r + b = 24 := 
by sorry

end NUMINAMATH_GPT_initial_deck_card_count_l2086_208611


namespace NUMINAMATH_GPT_common_chord_length_of_two_circles_l2086_208622

noncomputable def common_chord_length (r : ℝ) : ℝ :=
  if r = 10 then 10 * Real.sqrt 3 else sorry

theorem common_chord_length_of_two_circles (r : ℝ) (h : r = 10) :
  common_chord_length r = 10 * Real.sqrt 3 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_common_chord_length_of_two_circles_l2086_208622


namespace NUMINAMATH_GPT_family_ages_l2086_208632

theorem family_ages 
  (youngest : ℕ)
  (middle : ℕ := youngest + 2)
  (eldest : ℕ := youngest + 4)
  (mother : ℕ := 3 * youngest + 16)
  (father : ℕ := 4 * youngest + 18)
  (total_sum : youngest + middle + eldest + mother + father = 90) :
  youngest = 5 ∧ middle = 7 ∧ eldest = 9 ∧ mother = 31 ∧ father = 38 := 
by 
  sorry

end NUMINAMATH_GPT_family_ages_l2086_208632


namespace NUMINAMATH_GPT_greatest_A_satisfies_condition_l2086_208602

theorem greatest_A_satisfies_condition :
  ∃ (A : ℝ), A = 64 ∧ ∀ (s : Fin₇ → ℝ), (∀ i, 1 ≤ s i ∧ s i ≤ A) →
  ∃ (i j : Fin₇), i ≠ j ∧ (1 / 2 ≤ s i / s j ∧ s i / s j ≤ 2) :=
by 
  sorry

end NUMINAMATH_GPT_greatest_A_satisfies_condition_l2086_208602


namespace NUMINAMATH_GPT_clock_hands_angle_120_l2086_208686

-- We are only defining the problem statement and conditions. No need for proof steps or calculations.

def angle_between_clock_hands (hour minute : ℚ) : ℚ :=
  abs ((30 * hour + minute / 2) - (6 * minute))

-- Given conditions
def time_in_range (hour : ℚ) (minute : ℚ) := 7 ≤ hour ∧ hour < 8

-- Problem statement to be proved
theorem clock_hands_angle_120 (hour minute : ℚ) :
  time_in_range hour minute → angle_between_clock_hands hour minute = 120 :=
sorry

end NUMINAMATH_GPT_clock_hands_angle_120_l2086_208686


namespace NUMINAMATH_GPT_sqrt_205_between_14_and_15_l2086_208697

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_205_between_14_and_15_l2086_208697


namespace NUMINAMATH_GPT_choose_4_from_15_l2086_208641

theorem choose_4_from_15 : (Nat.choose 15 4) = 1365 :=
by
  sorry

end NUMINAMATH_GPT_choose_4_from_15_l2086_208641


namespace NUMINAMATH_GPT_abc_sum_seven_l2086_208685

theorem abc_sum_seven (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 7 :=
sorry

end NUMINAMATH_GPT_abc_sum_seven_l2086_208685


namespace NUMINAMATH_GPT_ratio_of_triangle_areas_l2086_208600

theorem ratio_of_triangle_areas 
  (r s : ℝ) (n : ℝ)
  (h_ratio : 3 * s = r) 
  (h_area : (3 / 2) * n = 1 / 2 * r * ((3 * n * 2) / r)) :
  3 / 3 = n :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_areas_l2086_208600


namespace NUMINAMATH_GPT_evaluate_expr_at_2_l2086_208614

def expr (x : ℝ) : ℝ := (2 * x + 3) * (2 * x - 3) + (x - 2) ^ 2 - 3 * x * (x - 1)

theorem evaluate_expr_at_2 : expr 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expr_at_2_l2086_208614


namespace NUMINAMATH_GPT_range_of_a_l2086_208665

theorem range_of_a (a : ℝ) : 1 ∉ {x : ℝ | x^2 - 2 * x + a > 0} → a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2086_208665


namespace NUMINAMATH_GPT_find_current_l2086_208688

theorem find_current (R Q t : ℝ) (hR : R = 8) (hQ : Q = 72) (ht : t = 2) :
  ∃ I : ℝ, Q = I^2 * R * t ∧ I = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_current_l2086_208688


namespace NUMINAMATH_GPT_complex_sum_l2086_208675

noncomputable def omega : ℂ := sorry
axiom h1 : omega^11 = 1
axiom h2 : omega ≠ 1

theorem complex_sum 
: omega^10 + omega^14 + omega^18 + omega^22 + omega^26 + omega^30 + omega^34 + omega^38 + omega^42 + omega^46 + omega^50 + omega^54 + omega^58 
= -omega^10 :=
sorry

end NUMINAMATH_GPT_complex_sum_l2086_208675


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l2086_208630

theorem ratio_of_larger_to_smaller 
    (x y : ℝ) 
    (hx : x > 0) 
    (hy : y > 0) 
    (h : x + y = 7 * (x - y)) : 
    x / y = 4 / 3 := 
by 
    sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l2086_208630


namespace NUMINAMATH_GPT_profit_function_and_optimal_price_l2086_208668

variable (cost selling base_units additional_units: ℝ)
variable (x: ℝ) (y: ℝ)

def profit (x: ℝ): ℝ := -20 * x^2 + 100 * x + 6000

theorem profit_function_and_optimal_price:
  (cost = 40) →
  (selling = 60) →
  (base_units = 300) →
  (additional_units = 20) →
  (0 ≤ x) →
  (x < 20) →
  (y = profit x) →
  exists x_max y_max: ℝ, (x_max = 2.5) ∧ (y_max = 6125) :=
by 
  sorry

end NUMINAMATH_GPT_profit_function_and_optimal_price_l2086_208668
