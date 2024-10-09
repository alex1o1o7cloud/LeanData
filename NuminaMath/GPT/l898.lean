import Mathlib

namespace Mona_grouped_with_one_player_before_in_second_group_l898_89828

/-- Mona plays in groups with four other players, joined 9 groups, and grouped with 33 unique players. 
    One of the groups included 2 players she had grouped with before. 
    Prove that the number of players she had grouped with before in the second group is 1. -/
theorem Mona_grouped_with_one_player_before_in_second_group 
    (total_groups : ℕ) (group_size : ℕ) (unique_players : ℕ) 
    (repeat_players_in_group1 : ℕ) : 
    (total_groups = 9) → (group_size = 5) → (unique_players = 33) → (repeat_players_in_group1 = 2) 
        → ∃ repeat_players_in_group2 : ℕ, repeat_players_in_group2 = 1 :=
by
    sorry

end Mona_grouped_with_one_player_before_in_second_group_l898_89828


namespace goods_train_length_l898_89845

theorem goods_train_length 
  (v_kmph : ℝ) (L_p : ℝ) (t : ℝ) (v_mps : ℝ) (d : ℝ) (L_t : ℝ) 
  (h1 : v_kmph = 96) 
  (h2 : L_p = 480) 
  (h3 : t = 36) 
  (h4 : v_mps = v_kmph * (5/18)) 
  (h5 : d = v_mps * t) : 
  L_t = d - L_p :=
sorry

end goods_train_length_l898_89845


namespace probability_of_selecting_same_gender_l898_89846

def number_of_ways_to_choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_of_selecting_same_gender (total_students male_students female_students : ℕ) (h1 : total_students = 10) (h2 : male_students = 2) (h3 : female_students = 8) : 
  let total_combinations := number_of_ways_to_choose_two total_students
  let male_combinations := number_of_ways_to_choose_two male_students
  let female_combinations := number_of_ways_to_choose_two female_students
  let favorable_combinations := male_combinations + female_combinations
  total_combinations = 45 ∧
  male_combinations = 1 ∧
  female_combinations = 28 ∧
  favorable_combinations = 29 ∧
  (favorable_combinations : ℚ) / total_combinations = 29 / 45 :=
by
  sorry

end probability_of_selecting_same_gender_l898_89846


namespace min_ab_l898_89839

theorem min_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 2 := by
  sorry

end min_ab_l898_89839


namespace inequalities_always_hold_l898_89853

theorem inequalities_always_hold (x y a b : ℝ) (hxy : x > y) (hab : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) :=
by
  sorry

end inequalities_always_hold_l898_89853


namespace number_of_female_students_l898_89817

-- Given conditions
variables (F : ℕ)

-- The average score of all students (90)
def avg_all_students := 90
-- The total number of male students (8)
def num_male_students := 8
-- The average score of male students (87)
def avg_male_students := 87
-- The average score of female students (92)
def avg_female_students := 92

-- We want to prove the following statement
theorem number_of_female_students :
  num_male_students * avg_male_students + F * avg_female_students = (num_male_students + F) * avg_all_students →
  F = 12 :=
sorry

end number_of_female_students_l898_89817


namespace quadratic_inequality_solution_l898_89821

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 9*x + 14 < 0) : 2 < x ∧ x < 7 :=
by
  sorry

end quadratic_inequality_solution_l898_89821


namespace abs_diff_squares_105_95_l898_89891

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l898_89891


namespace prove_N_value_l898_89842

theorem prove_N_value (x y N : ℝ) 
  (h1 : N = 4 * x + y) 
  (h2 : 3 * x - 4 * y = 5) 
  (h3 : 7 * x - 3 * y = 23) : 
  N = 86 / 3 := by
  sorry

end prove_N_value_l898_89842


namespace amount_after_two_years_l898_89837

-- Definition of initial amount and the rate of increase
def initial_value : ℝ := 32000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

-- The compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- The proof problem: Prove that after 2 years the amount is 40500
theorem amount_after_two_years : compound_interest initial_value rate_of_increase time_period = 40500 :=
sorry

end amount_after_two_years_l898_89837


namespace sector_area_l898_89876

theorem sector_area (theta : ℝ) (L : ℝ) (h_theta : theta = π / 3) (h_L : L = 4) :
  ∃ r : ℝ, (L = r * theta ∧ ∃ A : ℝ, A = 1/2 * r^2 * theta ∧ A = 24 / π) := by
  sorry

end sector_area_l898_89876


namespace integers_satisfy_equation_l898_89825

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end integers_satisfy_equation_l898_89825


namespace division_of_cubics_l898_89815

theorem division_of_cubics (c d : ℕ) (h1 : c = 7) (h2 : d = 3) : 
  (c^3 + d^3) / (c^2 - c * d + d^2) = 10 := by
  sorry

end division_of_cubics_l898_89815


namespace inequality_abc_l898_89863

theorem inequality_abc (a b c : ℝ) (h1 : a ∈ Set.Icc (-1 : ℝ) 2) (h2 : b ∈ Set.Icc (-1 : ℝ) 2) (h3 : c ∈ Set.Icc (-1 : ℝ) 2) : 
  a * b * c + 4 ≥ a * b + b * c + c * a := 
sorry

end inequality_abc_l898_89863


namespace geometric_sequence_sum_l898_89894

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 0 = 3)
(h_sum : a 0 + a 1 + a 2 = 21) (hq : ∀ n, a (n + 1) = a n * q) : a 2 + a 3 + a 4 = 84 := by
  sorry

end geometric_sequence_sum_l898_89894


namespace adjusted_distance_buoy_fourth_l898_89805

theorem adjusted_distance_buoy_fourth :
  let a1 := 20  -- distance to the first buoy
  let d := 4    -- common difference (distance between consecutive buoys)
  let ocean_current_effect := 3  -- effect of ocean current
  
  -- distances from the beach to buoys based on their sequence
  let a2 := a1 + d 
  let a3 := a2 + d
  let a4 := a3 + d
  
  -- distance to the fourth buoy without external factors
  let distance_to_fourth_buoy := a1 + 3 * d
  
  -- adjusted distance considering the ocean current
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  adjusted_distance = 29 := 
by
  let a1 := 20
  let d := 4
  let ocean_current_effect := 3
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let distance_to_fourth_buoy := a1 + 3 * d
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  sorry

end adjusted_distance_buoy_fourth_l898_89805


namespace power_inequality_l898_89859

variable {a b : ℝ}

theorem power_inequality (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := 
by sorry

end power_inequality_l898_89859


namespace fraction_to_decimal_l898_89800

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
sorry

end fraction_to_decimal_l898_89800


namespace min_value_condition_l898_89814

theorem min_value_condition 
  (a b : ℝ) 
  (h1 : 4 * a + b = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 1 - 4 * x → x = 16) := 
sorry

end min_value_condition_l898_89814


namespace period_six_l898_89867

variable {R : Type} [LinearOrderedField R]

def symmetric1 (f : R → R) : Prop := ∀ x : R, f (2 + x) = f (2 - x)
def symmetric2 (f : R → R) : Prop := ∀ x : R, f (5 + x) = f (5 - x)

theorem period_six (f : R → R) (h1 : symmetric1 f) (h2 : symmetric2 f) : ∀ x : R, f (x + 6) = f x :=
sorry

end period_six_l898_89867


namespace abs_eq_case_l898_89886

theorem abs_eq_case (x : ℝ) (h : |x - 3| = |x + 2|) : x = 1/2 :=
by
  sorry

end abs_eq_case_l898_89886


namespace no_point_on_line_y_eq_2x_l898_89869

theorem no_point_on_line_y_eq_2x
  (marked : Set (ℕ × ℕ))
  (initial_points : { p // p ∈ [(1, 1), (2, 3), (4, 5), (999, 111)] })
  (rule1 : ∀ a b, (a, b) ∈ marked → (b, a) ∈ marked ∧ (a - b, a + b) ∈ marked)
  (rule2 : ∀ a b c d, (a, b) ∈ marked ∧ (c, d) ∈ marked → (a * d + b * c, 4 * a * c - 4 * b * d) ∈ marked) :
  ∃ x, (x, 2 * x) ∈ marked → False := sorry

end no_point_on_line_y_eq_2x_l898_89869


namespace solve_bx2_ax_1_lt_0_l898_89804

noncomputable def quadratic_inequality_solution (a b : ℝ) (x : ℝ) : Prop :=
  x^2 + a * x + b > 0

theorem solve_bx2_ax_1_lt_0 (a b : ℝ) :
  (∀ x : ℝ, quadratic_inequality_solution a b x ↔ (x < -2 ∨ x > -1/2)) →
  (∀ x : ℝ, (x = -2 ∨ x = -1/2) → x^2 + a * x + b = 0) →
  (b * x^2 + a * x + 1 < 0) ↔ (-2 < x ∧ x < -1/2) :=
by
  sorry

end solve_bx2_ax_1_lt_0_l898_89804


namespace P_subset_Q_l898_89879

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_subset_Q : P ⊂ Q := by
  sorry

end P_subset_Q_l898_89879


namespace toms_total_cost_l898_89829

theorem toms_total_cost :
  let costA := 4 * 15
  let costB := 3 * 12
  let discountB := 0.20 * costB
  let costBDiscounted := costB - discountB
  let costC := 2 * 18
  costA + costBDiscounted + costC = 124.80 := 
by
  sorry

end toms_total_cost_l898_89829


namespace michael_twice_jacob_l898_89860

variable {J M Y : ℕ}

theorem michael_twice_jacob :
  (J + 4 = 13) → (M = J + 12) → (M + Y = 2 * (J + Y)) → (Y = 3) := by
  sorry

end michael_twice_jacob_l898_89860


namespace max_sum_condition_l898_89862

theorem max_sum_condition (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : Nat.gcd a b = 6) : a + b ≤ 186 :=
sorry

end max_sum_condition_l898_89862


namespace probability_at_least_one_alarm_on_time_l898_89835

noncomputable def P_alarm_A_on : ℝ := 0.80
noncomputable def P_alarm_B_on : ℝ := 0.90

theorem probability_at_least_one_alarm_on_time :
  (1 - (1 - P_alarm_A_on) * (1 - P_alarm_B_on)) = 0.98 :=
by
  sorry

end probability_at_least_one_alarm_on_time_l898_89835


namespace area_on_map_correct_l898_89884

namespace FieldMap

-- Given conditions
def actual_length_m : ℕ := 200
def actual_width_m : ℕ := 100
def scale_factor : ℕ := 2000

-- Conversion from meters to centimeters
def length_cm := actual_length_m * 100
def width_cm := actual_width_m * 100

-- Dimensions on the map
def length_map_cm := length_cm / scale_factor
def width_map_cm := width_cm / scale_factor

-- Area on the map
def area_map_cm2 := length_map_cm * width_map_cm

-- Statement to prove
theorem area_on_map_correct : area_map_cm2 = 50 := by
  sorry

end FieldMap

end area_on_map_correct_l898_89884


namespace min_xy_l898_89898

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : xy = x + 4 * y + 5) : xy ≥ 25 :=
sorry

end min_xy_l898_89898


namespace students_at_end_of_year_l898_89807

def n_start := 10
def n_left := 4
def n_new := 42

theorem students_at_end_of_year : n_start - n_left + n_new = 48 := by
  sorry

end students_at_end_of_year_l898_89807


namespace lucy_snowballs_eq_19_l898_89833

-- Define the conditions
def charlie_snowballs : ℕ := 50
def difference_charlie_lucy : ℕ := 31

-- Define what we want to prove, i.e., Lucy has 19 snowballs
theorem lucy_snowballs_eq_19 : (charlie_snowballs - difference_charlie_lucy = 19) :=
by
  -- We would provide the proof here, but it's not required for this prompt
  sorry

end lucy_snowballs_eq_19_l898_89833


namespace sequence_sum_a_b_l898_89883

theorem sequence_sum_a_b (a b : ℕ) (a_seq : ℕ → ℕ) 
  (h1 : a_seq 1 = a)
  (h2 : a_seq 2 = b)
  (h3 : ∀ n ≥ 1, a_seq (n+2) = (a_seq n + 2018) / (a_seq (n+1) + 1)) :
  a + b = 1011 ∨ a + b = 2019 :=
sorry

end sequence_sum_a_b_l898_89883


namespace max_value_k_eq_1_range_k_no_zeros_l898_89812

-- Define the function f(x)
noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Note: 'by' and 'sorry' are placeholders to skip the proof; actual proofs are not required.

-- Proof Problem 1: Prove that when k = 1, the maximum value of f(x) is 0.
theorem max_value_k_eq_1 : ∀ x : ℝ, 1 < x → f x 1 ≤ 0 := 
by
  sorry

-- Proof Problem 2: Prove that k ∈ (1, +∞) is the range such that f(x) has no zeros.
theorem range_k_no_zeros : ∀ k : ℝ, (∀ x : ℝ, 1 < x → f x k ≠ 0) → 1 < k :=
by
  sorry

end max_value_k_eq_1_range_k_no_zeros_l898_89812


namespace sqrt_product_simplification_l898_89871

theorem sqrt_product_simplification (p : ℝ) : 
  (Real.sqrt (42 * p)) * (Real.sqrt (14 * p)) * (Real.sqrt (7 * p)) = 14 * p * (Real.sqrt (21 * p)) := 
  sorry

end sqrt_product_simplification_l898_89871


namespace remaining_black_cards_l898_89880

theorem remaining_black_cards 
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (cards_taken_out : ℕ)
  (h1 : total_cards = 52)
  (h2 : black_cards = 26)
  (h3 : red_cards = 26)
  (h4 : cards_taken_out = 5) :
  black_cards - cards_taken_out = 21 := 
by {
  sorry
}

end remaining_black_cards_l898_89880


namespace quarters_addition_l898_89856

def original_quarters : ℝ := 783.0
def added_quarters : ℝ := 271.0
def total_quarters : ℝ := 1054.0

theorem quarters_addition :
  original_quarters + added_quarters = total_quarters :=
by
  sorry

end quarters_addition_l898_89856


namespace inequality_one_inequality_two_l898_89843

theorem inequality_one (x : ℝ) : 7 * x - 2 < 3 * (x + 2) → x < 2 :=
by
  sorry

theorem inequality_two (x : ℝ) : (x - 1) / 3 ≥ (x - 3) / 12 + 1 → x ≥ 13 / 3 :=
by
  sorry

end inequality_one_inequality_two_l898_89843


namespace pencils_in_each_box_l898_89887

open Nat

theorem pencils_in_each_box (boxes pencils_given_to_Lauren pencils_left pencils_each_box more_pencils : ℕ)
  (h1 : boxes = 2)
  (h2 : pencils_given_to_Lauren = 6)
  (h3 : pencils_left = 9)
  (h4 : more_pencils = 3)
  (h5 : pencils_given_to_Matt = pencils_given_to_Lauren + more_pencils)
  (h6 : pencils_each_box = (pencils_given_to_Lauren + pencils_given_to_Matt + pencils_left) / boxes) :
  pencils_each_box = 12 := by
  sorry

end pencils_in_each_box_l898_89887


namespace robot_distance_covered_l898_89881

theorem robot_distance_covered :
  let start1 := -3
  let end1 := -8
  let end2 := 6
  let distance1 := abs (end1 - start1)
  let distance2 := abs (end2 - end1)
  distance1 + distance2 = 19 := by
  sorry

end robot_distance_covered_l898_89881


namespace batting_average_is_60_l898_89854

-- Definitions for conditions:
def highest_score : ℕ := 179
def difference_highest_lowest : ℕ := 150
def average_44_innings : ℕ := 58
def innings_excluding_highest_lowest : ℕ := 44
def total_innings : ℕ := 46

-- Lowest score
def lowest_score : ℕ := highest_score - difference_highest_lowest

-- Total runs in 44 innings
def total_runs_44 : ℕ := average_44_innings * innings_excluding_highest_lowest

-- Total runs in 46 innings
def total_runs_46 : ℕ := total_runs_44 + highest_score + lowest_score

-- Batting average in 46 innings
def batting_average_46 : ℕ := total_runs_46 / total_innings

-- The theorem to prove
theorem batting_average_is_60 :
  batting_average_46 = 60 :=
sorry

end batting_average_is_60_l898_89854


namespace base9_add_subtract_l898_89851

theorem base9_add_subtract :
  let n1 := 3 * 9^2 + 5 * 9 + 1
  let n2 := 4 * 9^2 + 6 * 9 + 5
  let n3 := 1 * 9^2 + 3 * 9 + 2
  let n4 := 1 * 9^2 + 4 * 9 + 7
  (n1 + n2 + n3 - n4 = 8 * 9^2 + 4 * 9 + 7) :=
by
  sorry

end base9_add_subtract_l898_89851


namespace point_on_circle_l898_89878

noncomputable def x_value_on_circle : ℝ :=
  let a := (-3 : ℝ)
  let b := (21 : ℝ)
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  let y := 12
  Cx

theorem point_on_circle (x y : ℝ) (a b : ℝ) (ha : a = -3) (hb : b = 21) (hy : y = 12) :
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  (x - Cx) ^ 2 + y ^ 2 = radius ^ 2 → x = x_value_on_circle :=
by
  intros
  sorry

end point_on_circle_l898_89878


namespace monotonically_increasing_intervals_min_and_max_values_l898_89820

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.sin (2 * x + Real.pi / 4) + 1

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, 
    -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi → 
    f (x + 1) ≥ f x := sorry

theorem min_and_max_values :
  ∃ min max, 
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≥ min ∧ f x ≤ max) ∧ 
    (min = 0) ∧ 
    (max = Real.sqrt 2 + 1) := sorry

end monotonically_increasing_intervals_min_and_max_values_l898_89820


namespace original_number_is_1200_l898_89849

theorem original_number_is_1200 (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by
  sorry

end original_number_is_1200_l898_89849


namespace books_on_shelf_l898_89895

theorem books_on_shelf (total_books : ℕ) (sold_books : ℕ) (shelves : ℕ) (remaining_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 27 → sold_books = 6 → shelves = 3 → remaining_books = total_books - sold_books → books_per_shelf = remaining_books / shelves → books_per_shelf = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_on_shelf_l898_89895


namespace bob_distance_when_meet_l898_89841

-- Definitions of the variables and conditions
def distance_XY : ℝ := 40
def yolanda_rate : ℝ := 2  -- Yolanda's walking rate in miles per hour
def bob_rate : ℝ := 4      -- Bob's walking rate in miles per hour
def yolanda_start_time : ℝ := 1 -- Yolanda starts 1 hour earlier 

-- Prove that Bob has walked 25.33 miles when he meets Yolanda
theorem bob_distance_when_meet : 
  ∃ t : ℝ, 2 * (t + yolanda_start_time) + 4 * t = distance_XY ∧ (4 * t = 25.33) := 
by
  sorry

end bob_distance_when_meet_l898_89841


namespace min_sum_a_b_l898_89838

theorem min_sum_a_b (a b : ℝ) (h1 : 4 * a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  a + b ≥ 16 :=
sorry

end min_sum_a_b_l898_89838


namespace timber_logging_years_l898_89857

theorem timber_logging_years 
  (V0 : ℝ) (r : ℝ) (V : ℝ) (t : ℝ)
  (hV0 : V0 = 100000)
  (hr : r = 0.08)
  (hV : V = 400000)
  (hformula : V = V0 * (1 + r)^t)
  : t = (Real.log 4 / Real.log 1.08) :=
by
  sorry

end timber_logging_years_l898_89857


namespace find_a_l898_89861

theorem find_a :
  (∃ x1 x2, (x1 + x2 = -2 ∧ x1 * x2 = a) ∧ (∃ y1 y2, (y1 + y2 = - a ∧ y1 * y2 = 2) ∧ (x1^2 + x2^2 = y1^2 + y2^2))) → 
  (a = -4) := 
by
  sorry

end find_a_l898_89861


namespace average_monthly_income_is_2125_l898_89824

noncomputable def calculate_average_monthly_income (expenses_3_months: ℕ) (expenses_4_months: ℕ) (expenses_5_months: ℕ) (savings_per_year: ℕ) : ℕ :=
  (expenses_3_months * 3 + expenses_4_months * 4 + expenses_5_months * 5 + savings_per_year) / 12

theorem average_monthly_income_is_2125 :
  calculate_average_monthly_income 1700 1550 1800 5200 = 2125 :=
by
  sorry

end average_monthly_income_is_2125_l898_89824


namespace inequality_proof_l898_89852

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1) :=
by
  sorry

end inequality_proof_l898_89852


namespace algebraic_expression_evaluation_l898_89818

theorem algebraic_expression_evaluation
  (x y p q : ℝ)
  (h1 : x + y = 0)
  (h2 : p * q = 1) : (x + y) - 2 * (p * q) = -2 :=
by
  sorry

end algebraic_expression_evaluation_l898_89818


namespace volume_between_spheres_l898_89806

theorem volume_between_spheres (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 8) : 
  (4 / 3) * Real.pi * (r_large ^ 3) - (4 / 3) * Real.pi * (r_small ^ 3) = (1792 / 3) * Real.pi := 
by
  rw [h_small, h_large]
  sorry

end volume_between_spheres_l898_89806


namespace S_inter_T_eq_T_l898_89819

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l898_89819


namespace not_divisible_59_l898_89868

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end not_divisible_59_l898_89868


namespace jericho_altitude_300_l898_89808

def jericho_altitude (below_sea_level : Int) : Prop :=
  below_sea_level = -300

theorem jericho_altitude_300 (below_sea_level : Int)
  (h1 : below_sea_level = -300) : jericho_altitude below_sea_level :=
by
  sorry

end jericho_altitude_300_l898_89808


namespace m_value_l898_89888

open Polynomial

noncomputable def f (m : ℚ) : Polynomial ℚ := X^4 - 5*X^2 + 4*X - C m

theorem m_value (m : ℚ) : (2 * X + 1) ∣ f m ↔ m = -51/16 := by sorry

end m_value_l898_89888


namespace total_sum_step_l898_89848

-- Defining the conditions
def step_1_sum : ℕ := 2

-- Define the inductive process
def total_sum_labels (n : ℕ) : ℕ :=
  if n = 1 then step_1_sum
  else 2 * 3^(n - 1)

-- The theorem to prove
theorem total_sum_step (n : ℕ) : 
  total_sum_labels n = 2 * 3^(n - 1) :=
by
  sorry

end total_sum_step_l898_89848


namespace total_earnings_to_afford_car_l898_89870

-- Define the earnings per month
def monthlyEarnings : ℕ := 4000

-- Define the savings per month
def monthlySavings : ℕ := 500

-- Define the total amount needed to buy the car
def totalNeeded : ℕ := 45000

-- Define the number of months needed to save enough money
def monthsToSave : ℕ := totalNeeded / monthlySavings

-- Theorem stating the total money earned before he saves enough to buy the car
theorem total_earnings_to_afford_car : monthsToSave * monthlyEarnings = 360000 := by
  sorry

end total_earnings_to_afford_car_l898_89870


namespace factor_evaluate_l898_89864

theorem factor_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -2) : 
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end factor_evaluate_l898_89864


namespace probability_of_two_in_decimal_rep_of_eight_over_eleven_l898_89830

theorem probability_of_two_in_decimal_rep_of_eight_over_eleven : 
  (∃ B : List ℕ, (B = [7, 2]) ∧ (1 = (B.count 2) / (B.length)) ∧ 
  (0 + B.sum + 1) / 11 = 8 / 11) := sorry

end probability_of_two_in_decimal_rep_of_eight_over_eleven_l898_89830


namespace remainder_of_series_div_9_l898_89875

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end remainder_of_series_div_9_l898_89875


namespace final_value_of_A_l898_89899

theorem final_value_of_A (A : ℝ) (h1: A = 15) (h2: A = -A + 5) : A = -10 :=
sorry

end final_value_of_A_l898_89899


namespace tangent_line_to_circle_l898_89840

theorem tangent_line_to_circle (c : ℝ) (h : 0 < c) : 
  (∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c) ↔ c = 4 :=
by sorry

end tangent_line_to_circle_l898_89840


namespace tree_graph_probability_127_l898_89809

theorem tree_graph_probability_127 :
  let n := 5
  let p := 125
  let q := 1024
  q ^ (1/10) + p = 127 :=
by
  sorry

end tree_graph_probability_127_l898_89809


namespace find_coordinates_of_B_l898_89827

-- Define the conditions from the problem
def point_A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def point_B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- The proof problem: The coordinates of B are (4, -4)
theorem find_coordinates_of_B (a : ℝ) (h : point_A a = (0, a + 1)) : point_B a = (4, -4) := by
  -- This is skipping the proof part.
  sorry

end find_coordinates_of_B_l898_89827


namespace quadratic_decreases_after_vertex_l898_89896

theorem quadratic_decreases_after_vertex :
  ∀ x : ℝ, (x > 2) → (y = -(x - 2)^2 + 3) → ∃ k : ℝ, k < 0 :=
by
  sorry

end quadratic_decreases_after_vertex_l898_89896


namespace equilateral_triangle_of_arith_geo_seq_l898_89882

def triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :=
  (α + β + γ = Real.pi) ∧
  (2 * β = α + γ) ∧
  (b^2 = a * c)

theorem equilateral_triangle_of_arith_geo_seq
  (A B C : ℝ) (a b c α β γ : ℝ)
  (h1 : triangle A B C a b c α β γ)
  : (a = c) ∧ (A = B) ∧ (B = C) ∧ (a = b) :=
  sorry

end equilateral_triangle_of_arith_geo_seq_l898_89882


namespace goose_eggs_l898_89877

theorem goose_eggs (E : ℝ) :
  (E / 2 * 3 / 4 * 2 / 5 + (1 / 3 * (E / 2)) * 2 / 3 * 3 / 4 + (1 / 6 * (E / 2 + E / 6)) * 1 / 2 * 2 / 3 = 150) →
  E = 375 :=
by
  sorry

end goose_eggs_l898_89877


namespace part1_part2_l898_89847

variable (a b c x : ℝ)

-- Condition: lengths of the sides of the triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Quadratic equation
def quadratic_eq (x : ℝ) : ℝ := (a + c) * x^2 - 2 * b * x + (a - c)

-- Proof problem 1: If x = 1 is a root, then triangle ABC is isosceles
theorem part1 (h : quadratic_eq a b c 1 = 0) : a = b :=
by
  sorry

-- Proof problem 2: If triangle ABC is equilateral, then roots of the quadratic equation are 0 and 1
theorem part2 (h_eq : a = b ∧ b = c) :
  (quadratic_eq a a a 0 = 0) ∧ (quadratic_eq a a a 1 = 0) :=
by
  sorry

end part1_part2_l898_89847


namespace area_of_rectangle_abcd_l898_89865

-- Definition of the problem's conditions and question
def small_square_side_length : ℝ := 1
def large_square_side_length : ℝ := 1.5
def area_rectangle_abc : ℝ := 4.5

-- Lean 4 statement: Prove the area of rectangle ABCD is 4.5 square inches
theorem area_of_rectangle_abcd :
  (3 * small_square_side_length) * large_square_side_length = area_rectangle_abc :=
by
  sorry

end area_of_rectangle_abcd_l898_89865


namespace samantha_score_l898_89850

variables (correct_answers geometry_correct_answers incorrect_answers unanswered_questions : ℕ)
          (points_per_correct : ℝ := 1) (additional_geometry_points : ℝ := 0.5)

def total_score (correct_answers geometry_correct_answers : ℕ) : ℝ :=
  correct_answers * points_per_correct + geometry_correct_answers * additional_geometry_points

theorem samantha_score 
  (Samantha_correct : correct_answers = 15)
  (Samantha_geometry : geometry_correct_answers = 4)
  (Samantha_incorrect : incorrect_answers = 5)
  (Samantha_unanswered : unanswered_questions = 5) :
  total_score correct_answers geometry_correct_answers = 17 := 
by
  sorry

end samantha_score_l898_89850


namespace intersection_M_N_l898_89892

open Set

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

theorem intersection_M_N :
  M ∩ N = {-1, 3} := 
sorry

end intersection_M_N_l898_89892


namespace find_a13_l898_89816

variable (a_n : ℕ → ℝ)
variable (d : ℝ)
variable (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
variable (h_geo : a_n 9 ^ 2 = a_n 1 * a_n 5)
variable (h_sum : a_n 1 + 3 * a_n 5 + a_n 9 = 20)

theorem find_a13 (h_non_zero_d : d ≠ 0):
  a_n 13 = 28 :=
sorry

end find_a13_l898_89816


namespace solve_for_x_l898_89874

theorem solve_for_x : ∃ x : ℝ, x^4 + 10 * x^3 + 9 * x^2 - 50 * x - 56 = 0 ↔ x = -2 :=
by
  sorry

end solve_for_x_l898_89874


namespace tensor_identity_l898_89844

def tensor (a b : ℝ) : ℝ := a^3 - b

theorem tensor_identity (a : ℝ) : tensor a (tensor a (tensor a a)) = a^3 - a :=
by
  sorry

end tensor_identity_l898_89844


namespace min_value_x_plus_2y_l898_89803

theorem min_value_x_plus_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x + 2 * y ≥ 8 :=
sorry

end min_value_x_plus_2y_l898_89803


namespace beta_speed_l898_89885

theorem beta_speed (d : ℕ) (S_s : ℕ) (t : ℕ) (S_b : ℕ) :
  d = 490 ∧ S_s = 37 ∧ t = 7 ∧ (S_s * t) + (S_b * t) = d → S_b = 33 := by
  sorry

end beta_speed_l898_89885


namespace divisor_of_5025_is_5_l898_89832

/--
  Given an original number n which is 5026,
  and a resulting number after subtracting 1 from n,
  prove that the divisor of the resulting number is 5.
-/
theorem divisor_of_5025_is_5 (n : ℕ) (h₁ : n = 5026) (d : ℕ) (h₂ : (n - 1) % d = 0) : d = 5 :=
sorry

end divisor_of_5025_is_5_l898_89832


namespace find_root_and_m_l898_89893

theorem find_root_and_m {x : ℝ} {m : ℝ} (h : ∃ x1 x2 : ℝ, (x1 = 1) ∧ (x1 + x2 = -m) ∧ (x1 * x2 = 3)) :
  ∃ x2 : ℝ, (x2 = 3) ∧ (m = -4) :=
by
  obtain ⟨x1, x2, h1, h_sum, h_product⟩ := h
  have hx1 : x1 = 1 := h1
  rw [hx1] at h_product
  have hx2 : x2 = 3 := by linarith [h_product]
  have hm : m = -4 := by
    rw [hx1, hx2] at h_sum
    linarith
  exact ⟨x2, hx2, hm⟩

end find_root_and_m_l898_89893


namespace find_x6_l898_89811

-- Definition of the variables xi for i = 1, ..., 10.
variables {x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ}

-- Given conditions as equations.
axiom eq1 : (x2 + x4) / 2 = 3
axiom eq2 : (x4 + x6) / 2 = 5
axiom eq3 : (x6 + x8) / 2 = 7
axiom eq4 : (x8 + x10) / 2 = 9
axiom eq5 : (x10 + x2) / 2 = 1

axiom eq6 : (x1 + x3) / 2 = 2
axiom eq7 : (x3 + x5) / 2 = 4
axiom eq8 : (x5 + x7) / 2 = 6
axiom eq9 : (x7 + x9) / 2 = 8
axiom eq10 : (x9 + x1) / 2 = 10

-- The theorem to prove.
theorem find_x6 : x6 = 1 :=
by
  sorry

end find_x6_l898_89811


namespace probability_at_least_one_coordinate_greater_l898_89872

theorem probability_at_least_one_coordinate_greater (p : ℝ) :
  (∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ (x > p ∨ y > p))) ↔ p = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end probability_at_least_one_coordinate_greater_l898_89872


namespace frog_arrangements_l898_89802

theorem frog_arrangements :
  let total_frogs := 7
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let valid_sequences := 4
  let green_permutations := Nat.factorial green_frogs
  let red_permutations := Nat.factorial red_frogs
  let blue_permutations := Nat.factorial blue_frogs
  let total_permutations := valid_sequences * (green_permutations * red_permutations * blue_permutations)
  total_frogs = green_frogs + red_frogs + blue_frogs → 
  green_frogs = 2 ∧ red_frogs = 3 ∧ blue_frogs = 2 →
  valid_sequences = 4 →
  total_permutations = 96 := 
by
  -- Given conditions lead to the calculation of total permutations 
  sorry

end frog_arrangements_l898_89802


namespace circle_tangent_problem_solution_l898_89831

noncomputable def circle_tangent_problem
(radius : ℝ)
(center : ℝ × ℝ)
(point_A : ℝ × ℝ)
(distance_OA : ℝ)
(segment_BC : ℝ) : ℝ :=
  let r := radius
  let O := center
  let A := point_A
  let OA := distance_OA
  let BC := segment_BC
  let AT := Real.sqrt (OA^2 - r^2)
  2 * AT - BC

-- Definitions for the conditions
def radius : ℝ := 8
def center : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (17, 0)
def distance_OA : ℝ := 17
def segment_BC : ℝ := 12

-- Statement of the problem as an example theorem
theorem circle_tangent_problem_solution :
  circle_tangent_problem radius center point_A distance_OA segment_BC = 18 :=
by
  -- We would provide the proof here. The proof steps are not required as per the instructions.
  sorry

end circle_tangent_problem_solution_l898_89831


namespace grassy_plot_width_l898_89836

noncomputable def gravel_cost (L w p : ℝ) : ℝ :=
  0.80 * ((L + 2 * p) * (w + 2 * p) - L * w)

theorem grassy_plot_width
  (L : ℝ) 
  (p : ℝ) 
  (cost : ℝ) 
  (hL : L = 110) 
  (hp : p = 2.5) 
  (hcost : cost = 680) :
  ∃ w : ℝ, gravel_cost L w p = cost ∧ w = 97.5 :=
by
  sorry

end grassy_plot_width_l898_89836


namespace min_operations_to_reach_goal_l898_89897

-- Define the initial and final configuration of the letters
structure Configuration where
  A : Char := 'A'
  B : Char := 'B'
  C : Char := 'C'
  D : Char := 'D'
  E : Char := 'E'
  F : Char := 'F'
  G : Char := 'G'

-- Define a valid rotation operation
inductive Rotation
| rotate_ABC : Rotation
| rotate_ABD : Rotation
| rotate_DEF : Rotation
| rotate_EFC : Rotation

-- Function representing a single rotation
def applyRotation : Configuration -> Rotation -> Configuration
| config, Rotation.rotate_ABC => 
  { A := config.C, B := config.A, C := config.B, D := config.D, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_ABD => 
  { A := config.B, B := config.D, D := config.A, C := config.C, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_DEF => 
  { D := config.E, E := config.F, F := config.D, A := config.A, B := config.B, C := config.C, G := config.G }
| config, Rotation.rotate_EFC => 
  { E := config.F, F := config.C, C := config.E, A := config.A, B := config.B, D := config.D, G := config.G }

-- Define the goal configuration
def goalConfiguration : Configuration := 
  { A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G' }

-- Function to apply multiple rotations
def applyRotations (config : Configuration) (rotations : List Rotation) : Configuration :=
  rotations.foldl applyRotation config

-- Main theorem statement 
theorem min_operations_to_reach_goal : 
  ∃ rotations : List Rotation, rotations.length = 3 ∧ applyRotations {A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G'} rotations = goalConfiguration :=
sorry

end min_operations_to_reach_goal_l898_89897


namespace max_x_on_circle_l898_89801

theorem max_x_on_circle : 
  ∀ x y : ℝ,
  (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by
  intros x y h
  sorry

end max_x_on_circle_l898_89801


namespace range_of_function_l898_89834

theorem range_of_function : ∀ x : ℝ, 1 ≤ abs (Real.sin x) + 2 * abs (Real.cos x) ∧ abs (Real.sin x) + 2 * abs (Real.cos x) ≤ Real.sqrt 5 :=
by
  intro x
  sorry

end range_of_function_l898_89834


namespace solve_system_l898_89822

theorem solve_system :
  ∃ x y : ℝ, (x^3 + y^3) * (x^2 + y^2) = 64 ∧ x + y = 2 ∧ 
  ((x = 1 + Real.sqrt (5 / 3) ∧ y = 1 - Real.sqrt (5 / 3)) ∨ 
   (x = 1 - Real.sqrt (5 / 3) ∧ y = 1 + Real.sqrt (5 / 3))) :=
by
  sorry

end solve_system_l898_89822


namespace sequence_diff_exists_l898_89813

theorem sequence_diff_exists (x : ℕ → ℕ) (h1 : x 1 = 1) (h2 : ∀ n : ℕ, 1 ≤ n → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_exists_l898_89813


namespace max_dn_eq_401_l898_89866

open BigOperators

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_dn_eq_401 : ∃ n, d n = 401 ∧ ∀ m, d m ≤ 401 := by
  -- Proof will be filled here
  sorry

end max_dn_eq_401_l898_89866


namespace exponential_function_pass_through_point_l898_89826

theorem exponential_function_pass_through_point
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (a^(1 - 1) + 1 = 2) :=
by
  sorry

end exponential_function_pass_through_point_l898_89826


namespace linear_function_diff_l898_89823

noncomputable def g : ℝ → ℝ := sorry

theorem linear_function_diff (h_linear : ∀ x y z w : ℝ, (g y - g x) / (y - x) = (g w - g z) / (w - z))
                            (h_condition : g 8 - g 1 = 21) : 
  g 16 - g 1 = 45 := 
by 
  sorry

end linear_function_diff_l898_89823


namespace gab_score_ratio_l898_89889

theorem gab_score_ratio (S G C O : ℕ) (h1 : S = 20) (h2 : C = 2 * G) (h3 : O = 85) (h4 : S + G + C = O + 55) :
  G / S = 2 := 
by 
  sorry

end gab_score_ratio_l898_89889


namespace wedge_volume_calculation_l898_89873

theorem wedge_volume_calculation :
  let r := 5 
  let h := 8 
  let V := (1 / 3) * (Real.pi * r^2 * h) 
  V = (200 * Real.pi) / 3 :=
by
  let r := 5
  let h := 8
  let V := (1 / 3) * (Real.pi * r^2 * h)
  -- Prove the equality step is omitted as per the prompt
  sorry

end wedge_volume_calculation_l898_89873


namespace incenter_closest_to_median_l898_89810

variables (a b c : ℝ) (s_a s_b s_c d_a d_b d_c : ℝ)

noncomputable def median_length (a b c : ℝ) : ℝ := 
  Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

noncomputable def distance_to_median (x y median_length : ℝ) : ℝ := 
  (y - x) / (2 * median_length)

theorem incenter_closest_to_median
  (h₀ : a = 4) (h₁ : b = 5) (h₂ : c = 8) 
  (h₃ : s_a = median_length a b c)
  (h₄ : s_b = median_length b a c)
  (h₅ : s_c = median_length c a b)
  (h₆ : d_a = distance_to_median b c s_a)
  (h₇ : d_b = distance_to_median a c s_b)
  (h₈ : d_c = distance_to_median a b s_c) : 
  d_a = d_c := 
sorry

end incenter_closest_to_median_l898_89810


namespace final_score_is_correct_l898_89890

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end final_score_is_correct_l898_89890


namespace cuboid_volume_l898_89858

theorem cuboid_volume (a b c : ℕ) (h_incr_by_2_becomes_cube : c + 2 = a)
  (surface_area_incr : 2*a*(a + a + c + 2) - 2*a*(c + a + b) = 56) : a * b * c = 245 :=
sorry

end cuboid_volume_l898_89858


namespace least_positive_value_of_cubic_eq_l898_89855

theorem least_positive_value_of_cubic_eq (x y z w : ℕ) 
  (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hz : Nat.Prime z) (hw : Nat.Prime w) 
  (sum_lt_50 : x + y + z + w < 50) : 
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 :=
by
  sorry

end least_positive_value_of_cubic_eq_l898_89855
