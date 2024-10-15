import Mathlib

namespace NUMINAMATH_GPT_volume_of_cone_l2007_200731

noncomputable def lateral_surface_area : ℝ := 8 * Real.pi

theorem volume_of_cone (l r h : ℝ)
  (h_lateral_surface : l * Real.pi = 2 * lateral_surface_area)
  (h_radius : l = 2 * r)
  (h_height : h = Real.sqrt (l^2 - r^2)) :
  (1/3) * Real.pi * r^2 * h = (8 * Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cone_l2007_200731


namespace NUMINAMATH_GPT_minimize_PA_PB_l2007_200737

theorem minimize_PA_PB 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (5, 1)) : 
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ P' : ℝ × ℝ, P'.snd = 0 → (dist P A + dist P B) ≤ (dist P' A + dist P' B) :=
sorry

end NUMINAMATH_GPT_minimize_PA_PB_l2007_200737


namespace NUMINAMATH_GPT_shiny_pennies_probability_l2007_200735

theorem shiny_pennies_probability :
  ∃ (a b : ℕ), gcd a b = 1 ∧ a / b = 5 / 11 ∧ a + b = 16 :=
sorry

end NUMINAMATH_GPT_shiny_pennies_probability_l2007_200735


namespace NUMINAMATH_GPT_compute_diff_squares_l2007_200702

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_compute_diff_squares_l2007_200702


namespace NUMINAMATH_GPT_incorrect_option_c_l2007_200704

theorem incorrect_option_c (a b c d : ℝ)
  (h1 : a + b + c ≥ d)
  (h2 : a + b + d ≥ c)
  (h3 : a + c + d ≥ b)
  (h4 : b + c + d ≥ a) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) :=
by sorry

end NUMINAMATH_GPT_incorrect_option_c_l2007_200704


namespace NUMINAMATH_GPT_triangular_array_of_coins_l2007_200798

theorem triangular_array_of_coins (N : ℤ) (h : N * (N + 1) / 2 = 3003) : N = 77 :=
by
  sorry

end NUMINAMATH_GPT_triangular_array_of_coins_l2007_200798


namespace NUMINAMATH_GPT_volume_truncated_cone_l2007_200721

-- Define the geometric constants
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height_truncated_cone : ℝ := 8

-- The statement to prove the volume of the truncated cone
theorem volume_truncated_cone :
  let V_large := (1/3) * Real.pi * (large_base_radius^2) * (height_truncated_cone + height_truncated_cone)
  let V_small := (1/3) * Real.pi * (small_base_radius^2) * height_truncated_cone
  V_large - V_small = (1400/3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_truncated_cone_l2007_200721


namespace NUMINAMATH_GPT_symmetric_sum_eq_two_l2007_200776

-- Definitions and conditions
def symmetric (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

def P : ℝ × ℝ := (sorry, 1)
def Q : ℝ × ℝ := (-3, sorry)

-- Problem statement
theorem symmetric_sum_eq_two (h : symmetric P Q) : P.1 + Q.2 = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_symmetric_sum_eq_two_l2007_200776


namespace NUMINAMATH_GPT_min_value_fraction_l2007_200713

theorem min_value_fraction : ∃ (x : ℝ), (∀ y : ℝ, (y^2 + 9) / (Real.sqrt (y^2 + 5)) ≥ (9 * Real.sqrt 5) / 5)
  := sorry

end NUMINAMATH_GPT_min_value_fraction_l2007_200713


namespace NUMINAMATH_GPT_evaluate_fraction_l2007_200757

theorem evaluate_fraction : 3 / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2007_200757


namespace NUMINAMATH_GPT_units_digit_M_M12_l2007_200756

def modifiedLucas (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | 1     => 2
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem units_digit_M_M12 (n : ℕ) (H : modifiedLucas 12 = 555) : 
  (modifiedLucas (modifiedLucas 12) % 10) = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_M_M12_l2007_200756


namespace NUMINAMATH_GPT_abba_divisible_by_11_l2007_200796

-- Given any two-digit number with digits a and b
def is_divisible_by_11 (a b : ℕ) : Prop :=
  (1001 * a + 110 * b) % 11 = 0

theorem abba_divisible_by_11 (a b : ℕ) (ha : a < 10) (hb : b < 10) : is_divisible_by_11 a b :=
  sorry

end NUMINAMATH_GPT_abba_divisible_by_11_l2007_200796


namespace NUMINAMATH_GPT_slope_of_line_m_equals_neg_2_l2007_200768

theorem slope_of_line_m_equals_neg_2
  (m : ℝ)
  (h : (3 * m - 6) / (1 + m) = 12) :
  m = -2 :=
sorry

end NUMINAMATH_GPT_slope_of_line_m_equals_neg_2_l2007_200768


namespace NUMINAMATH_GPT_probability_of_two_germinates_is_48_over_125_l2007_200769

noncomputable def probability_of_exactly_two_germinates : ℚ :=
  let p := 4/5
  let n := 3
  let k := 2
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_of_two_germinates_is_48_over_125 :
  probability_of_exactly_two_germinates = 48/125 := by
    sorry

end NUMINAMATH_GPT_probability_of_two_germinates_is_48_over_125_l2007_200769


namespace NUMINAMATH_GPT_heather_bicycling_time_l2007_200733

theorem heather_bicycling_time (distance speed : ℝ) (h_distance : distance = 40) (h_speed : speed = 8) : (distance / speed) = 5 := 
by
  rw [h_distance, h_speed]
  norm_num

end NUMINAMATH_GPT_heather_bicycling_time_l2007_200733


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l2007_200762

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) (h_a5 : a 5 = 15) :
  a 3 + a 4 + a 6 + a 7 = 60 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l2007_200762


namespace NUMINAMATH_GPT_only_n_1_has_integer_solution_l2007_200729

theorem only_n_1_has_integer_solution :
  ∀ n : ℕ, (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 := 
by 
  sorry

end NUMINAMATH_GPT_only_n_1_has_integer_solution_l2007_200729


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l2007_200797

noncomputable def f (x : ℝ) : ℝ :=
  x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 18

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 22 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l2007_200797


namespace NUMINAMATH_GPT_exponential_equation_solution_l2007_200793

theorem exponential_equation_solution (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = (3 / 5)^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_exponential_equation_solution_l2007_200793


namespace NUMINAMATH_GPT_locus_of_point_M_l2007_200719

open Real

def distance (x y: ℝ × ℝ): ℝ :=
  ((x.1 - y.1)^2 + (x.2 - y.2)^2)^(1/2)

theorem locus_of_point_M :
  (∀ (M : ℝ × ℝ), 
     distance M (2, 0) + 1 = abs (M.1 + 3)) 
  → ∀ (M : ℝ × ℝ), M.2^2 = 8 * M.1 :=
sorry

end NUMINAMATH_GPT_locus_of_point_M_l2007_200719


namespace NUMINAMATH_GPT_stationary_train_length_l2007_200724

noncomputable def speed_train_kmh : ℝ := 144
noncomputable def speed_train_ms : ℝ := (speed_train_kmh * 1000) / 3600
noncomputable def time_to_pass_pole : ℝ := 8
noncomputable def time_to_pass_stationary : ℝ := 18
noncomputable def length_moving_train : ℝ := speed_train_ms * time_to_pass_pole
noncomputable def total_distance : ℝ := speed_train_ms * time_to_pass_stationary
noncomputable def length_stationary_train : ℝ := total_distance - length_moving_train

theorem stationary_train_length :
  length_stationary_train = 400 := by
  sorry

end NUMINAMATH_GPT_stationary_train_length_l2007_200724


namespace NUMINAMATH_GPT_minimum_jumps_l2007_200711

theorem minimum_jumps (dist_cm : ℕ) (jump_mm : ℕ) (dist_mm : ℕ) (cm_to_mm_conversion : dist_mm = dist_cm * 10) (leap_condition : ∃ n : ℕ, jump_mm * n ≥ dist_mm) : ∃ n : ℕ, 19 * n = 18120 → n = 954 :=
by
  sorry

end NUMINAMATH_GPT_minimum_jumps_l2007_200711


namespace NUMINAMATH_GPT_sqrt_six_ineq_l2007_200752

theorem sqrt_six_ineq : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_six_ineq_l2007_200752


namespace NUMINAMATH_GPT_each_friend_received_12_candies_l2007_200749

-- Define the number of friends and total candies given
def num_friends : ℕ := 35
def total_candies : ℕ := 420

-- Define the number of candies each friend received
def candies_per_friend : ℕ := total_candies / num_friends

theorem each_friend_received_12_candies :
  candies_per_friend = 12 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_each_friend_received_12_candies_l2007_200749


namespace NUMINAMATH_GPT_arithmetic_progression_terms_even_sums_l2007_200773

theorem arithmetic_progression_terms_even_sums (n a d : ℕ) (h_even : Even n) 
  (h_odd_sum : n * (a + (n - 2) * d) = 60) 
  (h_even_sum : n * (a + d + a + (n - 1) * d) = 72) 
  (h_last_first : (n - 1) * d = 12) : n = 8 := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_terms_even_sums_l2007_200773


namespace NUMINAMATH_GPT_factor_expression_l2007_200730

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2007_200730


namespace NUMINAMATH_GPT_doug_money_l2007_200738

def money_problem (J D B: ℝ) : Prop :=
  J + D + B = 68 ∧
  J = 2 * B ∧
  J = (3 / 4) * D

theorem doug_money (J D B: ℝ) (h: money_problem J D B): D = 36.27 :=
by sorry

end NUMINAMATH_GPT_doug_money_l2007_200738


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2007_200783

theorem arithmetic_sequence_sum (x y z d : ℤ)
  (h₀ : d = 10 - 3)
  (h₁ : 10 = 3 + d)
  (h₂ : 17 = 10 + d)
  (h₃ : x = 17 + d)
  (h₄ : y = x + d)
  (h₅ : 31 = y + d)
  (h₆ : z = 31 + d) :
  x + y + z = 93 := by
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2007_200783


namespace NUMINAMATH_GPT_shelves_fit_l2007_200741

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end NUMINAMATH_GPT_shelves_fit_l2007_200741


namespace NUMINAMATH_GPT_factor_expression_l2007_200725

theorem factor_expression (x : ℝ) :
  (16 * x ^ 7 + 36 * x ^ 4 - 9) - (4 * x ^ 7 - 6 * x ^ 4 - 9) = 6 * x ^ 4 * (2 * x ^ 3 + 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2007_200725


namespace NUMINAMATH_GPT_remainder_55_pow_55_plus_10_mod_8_l2007_200772

theorem remainder_55_pow_55_plus_10_mod_8 : (55 ^ 55 + 10) % 8 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_55_pow_55_plus_10_mod_8_l2007_200772


namespace NUMINAMATH_GPT_initially_calculated_avg_height_l2007_200789

theorem initially_calculated_avg_height
  (A : ℕ)
  (initially_calculated_total_height : ℕ := 35 * A)
  (wrong_height : ℕ := 166)
  (actual_height : ℕ := 106)
  (height_overestimation : ℕ := wrong_height - actual_height)
  (actual_avg_height : ℕ := 179)
  (correct_total_height : ℕ := 35 * actual_avg_height)
  (initially_calculate_total_height_is_more : initially_calculated_total_height = correct_total_height + height_overestimation) :
  A = 181 :=
by
  sorry

end NUMINAMATH_GPT_initially_calculated_avg_height_l2007_200789


namespace NUMINAMATH_GPT_scientific_notation_of_169200000000_l2007_200701

theorem scientific_notation_of_169200000000 : 169200000000 = 1.692 * 10^11 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_169200000000_l2007_200701


namespace NUMINAMATH_GPT_max_digit_sum_of_watch_display_l2007_200786

-- Define the problem conditions
def valid_hour (h : ℕ) : Prop := 0 ≤ h ∧ h < 24
def valid_minute (m : ℕ) : Prop := 0 ≤ m ∧ m < 60
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the proof problem
theorem max_digit_sum_of_watch_display : 
  ∃ h m : ℕ, valid_hour h ∧ valid_minute m ∧ (digit_sum h + digit_sum m = 24) :=
sorry

end NUMINAMATH_GPT_max_digit_sum_of_watch_display_l2007_200786


namespace NUMINAMATH_GPT_natural_number_pairs_sum_to_three_l2007_200774

theorem natural_number_pairs_sum_to_three :
  {p : ℕ × ℕ | p.1 + p.2 = 3} = {(1, 2), (2, 1)} :=
by
  sorry

end NUMINAMATH_GPT_natural_number_pairs_sum_to_three_l2007_200774


namespace NUMINAMATH_GPT_area_above_line_of_circle_l2007_200718

-- Define the circle equation
def circle_eq (x y : ℝ) := (x - 10)^2 + (y - 5)^2 = 50

-- Define the line equation
def line_eq (x y : ℝ) := y = x - 6

-- The area to determine
def area_above_line (R : ℝ) := 25 * R

-- Proof statement
theorem area_above_line_of_circle : area_above_line Real.pi = 25 * Real.pi :=
by
  -- mark the proof as sorry to skip the proof
  sorry

end NUMINAMATH_GPT_area_above_line_of_circle_l2007_200718


namespace NUMINAMATH_GPT_apples_pie_calculation_l2007_200751

-- Defining the conditions
def total_apples : ℕ := 34
def non_ripe_apples : ℕ := 6
def apples_per_pie : ℕ := 4 

-- Stating the problem
theorem apples_pie_calculation : (total_apples - non_ripe_apples) / apples_per_pie = 7 := by
  -- Proof would go here. For the structure of the task, we use sorry.
  sorry

end NUMINAMATH_GPT_apples_pie_calculation_l2007_200751


namespace NUMINAMATH_GPT_john_running_speed_l2007_200777

noncomputable def find_running_speed (x : ℝ) : Prop :=
  (12 / (3 * x + 2) + 8 / x = 2.2)

theorem john_running_speed : ∃ x : ℝ, find_running_speed x ∧ abs (x - 0.47) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_john_running_speed_l2007_200777


namespace NUMINAMATH_GPT_diff_between_percent_and_fraction_l2007_200795

theorem diff_between_percent_and_fraction :
  (0.75 * 800) - ((7 / 8) * 1200) = -450 :=
by
  sorry

end NUMINAMATH_GPT_diff_between_percent_and_fraction_l2007_200795


namespace NUMINAMATH_GPT_octal_67_equals_ternary_2001_l2007_200759

def octalToDecimal (n : Nat) : Nat :=
  -- Definition of octal to decimal conversion omitted
  sorry

def decimalToTernary (n : Nat) : Nat :=
  -- Definition of decimal to ternary conversion omitted
  sorry

theorem octal_67_equals_ternary_2001 : 
  decimalToTernary (octalToDecimal 67) = 2001 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_octal_67_equals_ternary_2001_l2007_200759


namespace NUMINAMATH_GPT_y_work_time_l2007_200712

theorem y_work_time (x_days : ℕ) (x_work_time : ℕ) (y_work_time : ℕ) :
  x_days = 40 ∧ x_work_time = 8 ∧ y_work_time = 20 →
  let x_rate := 1 / 40
  let work_done_by_x := 8 * x_rate
  let remaining_work := 1 - work_done_by_x
  let y_rate := remaining_work / 20
  y_rate * 25 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_y_work_time_l2007_200712


namespace NUMINAMATH_GPT_y_intercept_of_parallel_line_l2007_200740

theorem y_intercept_of_parallel_line (m : ℝ) (c1 c2 : ℝ) (x1 y1 : ℝ) (H_parallel : m = -3) (H_passing : (x1, y1) = (1, -4)) : 
    c2 = -1 :=
  sorry

end NUMINAMATH_GPT_y_intercept_of_parallel_line_l2007_200740


namespace NUMINAMATH_GPT_am_gm_inequality_l2007_200714

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem am_gm_inequality : (a / b) + (b / c) + (c / a) ≥ 3 := by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l2007_200714


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l2007_200799

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l2007_200799


namespace NUMINAMATH_GPT_lowest_point_in_fourth_quadrant_l2007_200767

theorem lowest_point_in_fourth_quadrant (k : ℝ) (h : k < -1) :
    let x := - (k + 1) / 2
    let y := (4 * k - (k + 1) ^ 2) / 4
    y < 0 ∧ x > 0 :=
by
  let x := - (k + 1) / 2
  let y := (4 * k - (k + 1) ^ 2) / 4
  sorry

end NUMINAMATH_GPT_lowest_point_in_fourth_quadrant_l2007_200767


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_six_terms_l2007_200726

noncomputable def sum_of_first_six_terms (a : ℤ) (d : ℤ) : ℤ :=
  let a1 := a
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let a5 := a4 + d
  let a6 := a5 + d
  a1 + a2 + a3 + a4 + a5 + a6

theorem arithmetic_sequence_sum_six_terms
  (a3 a4 a5 : ℤ)
  (h3 : a3 = 8)
  (h4 : a4 = 13)
  (h5 : a5 = 18)
  (d : ℤ) (a : ℤ)
  (h_d : d = a4 - a3)
  (h_a : a + 2 * d = 8) :
  sum_of_first_six_terms a d = 63 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_six_terms_l2007_200726


namespace NUMINAMATH_GPT_quadratic_root_eq_l2007_200720

theorem quadratic_root_eq {b : ℝ} (h : (2 : ℝ)^2 + b * 2 - 6 = 0) : b = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_eq_l2007_200720


namespace NUMINAMATH_GPT_janet_total_pockets_l2007_200744

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end NUMINAMATH_GPT_janet_total_pockets_l2007_200744


namespace NUMINAMATH_GPT_quadratic_roots_range_l2007_200763

theorem quadratic_roots_range (m : ℝ) : 
  (2 * x^2 - (m + 1) * x + m = 0) → 
  (m^2 - 6 * m + 1 > 0) → 
  (0 < m) → 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2 ∨ m > 3 + 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_range_l2007_200763


namespace NUMINAMATH_GPT_simplify_fraction_l2007_200781

theorem simplify_fraction (a b : ℕ) (h₁ : a = 84) (h₂ : b = 144) :
  a / gcd a b = 7 ∧ b / gcd a b = 12 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2007_200781


namespace NUMINAMATH_GPT_problem_statement_l2007_200775

theorem problem_statement (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
  sorry

end NUMINAMATH_GPT_problem_statement_l2007_200775


namespace NUMINAMATH_GPT_Kelly_remaining_games_l2007_200787

-- Definitions according to the conditions provided
def initial_games : ℝ := 121.0
def given_away : ℝ := 99.0
def remaining_games : ℝ := initial_games - given_away

-- The proof problem statement
theorem Kelly_remaining_games : remaining_games = 22.0 :=
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_Kelly_remaining_games_l2007_200787


namespace NUMINAMATH_GPT_smallest_positive_integer_x_l2007_200745

theorem smallest_positive_integer_x (x : ℕ) (h : 725 * x ≡ 1165 * x [MOD 35]) : x = 7 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_l2007_200745


namespace NUMINAMATH_GPT_two_pow_ge_n_cubed_l2007_200700

theorem two_pow_ge_n_cubed (n : ℕ) : 2^n ≥ n^3 ↔ n ≥ 10 := 
by sorry

end NUMINAMATH_GPT_two_pow_ge_n_cubed_l2007_200700


namespace NUMINAMATH_GPT_total_amount_paid_after_discount_l2007_200728

-- Define the given conditions
def marked_price_per_article : ℝ := 10
def discount_percentage : ℝ := 0.60
def number_of_articles : ℕ := 2

-- Proving the total amount paid
theorem total_amount_paid_after_discount : 
  (marked_price_per_article * number_of_articles) * (1 - discount_percentage) = 8 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_after_discount_l2007_200728


namespace NUMINAMATH_GPT_slope_of_line_l2007_200707

theorem slope_of_line (x y : ℝ) (h : 4 * x - 7 * y = 28) : (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 7) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_slope_of_line_l2007_200707


namespace NUMINAMATH_GPT_average_salary_techs_l2007_200708

noncomputable def total_salary := 20000
noncomputable def average_salary_all := 750
noncomputable def num_technicians := 5
noncomputable def average_salary_non_tech := 700
noncomputable def total_workers := 20

theorem average_salary_techs :
  (20000 - (num_technicians + average_salary_non_tech * (total_workers - num_technicians))) / num_technicians = 900 := by
  sorry

end NUMINAMATH_GPT_average_salary_techs_l2007_200708


namespace NUMINAMATH_GPT_total_cards_after_giveaway_l2007_200722

def ben_basketball_boxes := 8
def cards_per_basketball_box := 20
def ben_baseball_boxes := 10
def cards_per_baseball_box := 15
def ben_football_boxes := 12
def cards_per_football_box := 12

def alex_hockey_boxes := 6
def cards_per_hockey_box := 15
def alex_soccer_boxes := 9
def cards_per_soccer_box := 18

def cards_given_away := 175

def total_cards_for_ben := 
  (ben_basketball_boxes * cards_per_basketball_box) + 
  (ben_baseball_boxes * cards_per_baseball_box) + 
  (ben_football_boxes * cards_per_football_box)

def total_cards_for_alex := 
  (alex_hockey_boxes * cards_per_hockey_box) + 
  (alex_soccer_boxes * cards_per_soccer_box)

def total_cards_before_exchange := total_cards_for_ben + total_cards_for_alex

def ben_gives_to_alex := 
  (ben_basketball_boxes * (cards_per_basketball_box / 2)) + 
  (ben_baseball_boxes * (cards_per_baseball_box / 2))

def total_cards_remaining := total_cards_before_exchange - cards_given_away

theorem total_cards_after_giveaway :
  total_cards_before_exchange - cards_given_away = 531 := by
  sorry

end NUMINAMATH_GPT_total_cards_after_giveaway_l2007_200722


namespace NUMINAMATH_GPT_christine_savings_l2007_200748

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end NUMINAMATH_GPT_christine_savings_l2007_200748


namespace NUMINAMATH_GPT_consecutive_negative_product_sum_l2007_200754

theorem consecutive_negative_product_sum (n : ℤ) (h : n * (n + 1) = 2850) : n + (n + 1) = -107 :=
sorry

end NUMINAMATH_GPT_consecutive_negative_product_sum_l2007_200754


namespace NUMINAMATH_GPT_part1_part2_a_part2_b_part2_c_l2007_200743

noncomputable def f (x a : ℝ) := Real.exp x - x - a

theorem part1 (x : ℝ) : f x 0 > x := 
by 
  -- here would be the proof
  sorry

theorem part2_a (a : ℝ) : a > 1 → ∃ z₁ z₂ : ℝ, f z₁ a = 0 ∧ f z₂ a = 0 ∧ z₁ ≠ z₂ := 
by 
  -- here would be the proof
  sorry

theorem part2_b (a : ℝ) : a < 1 → ¬ (∃ z : ℝ, f z a = 0) := 
by 
  -- here would be the proof
  sorry

theorem part2_c : f 0 1 = 0 := 
by 
  -- here would be the proof
  sorry

end NUMINAMATH_GPT_part1_part2_a_part2_b_part2_c_l2007_200743


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_l2007_200770

open Real

theorem arithmetic_geometric_seq (a d : ℝ) (h₀ : d ≠ 0) 
  (h₁ : (a + d) * (a + 5 * d) = (a + 2 * d) ^ 2) : 
  (a + 2 * d) / (a + d) = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_l2007_200770


namespace NUMINAMATH_GPT_laptop_final_price_l2007_200717

theorem laptop_final_price (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  initial_price = 500 → first_discount = 10 → second_discount = 20 →
  (initial_price * (1 - first_discount / 100) * (1 - second_discount / 100)) = initial_price * 0.72 :=
by
  sorry

end NUMINAMATH_GPT_laptop_final_price_l2007_200717


namespace NUMINAMATH_GPT_factorization_from_left_to_right_l2007_200710

theorem factorization_from_left_to_right (a x y b : ℝ) :
  (a * (a + 1) = a^2 + a ∨
   a^2 + 3 * a - 1 = a * (a + 3) + 1 ∨
   x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) ∨
   (a - b)^3 = -(b - a)^3) →
  (x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)) := sorry

end NUMINAMATH_GPT_factorization_from_left_to_right_l2007_200710


namespace NUMINAMATH_GPT_square_perimeter_l2007_200764

theorem square_perimeter (A_total : ℕ) (A_common : ℕ) (A_circle : ℕ) 
  (H1 : A_total = 329)
  (H2 : A_common = 101)
  (H3 : A_circle = 234) :
  4 * (Int.sqrt (A_total - A_circle + A_common)) = 56 :=
by
  -- Since we are only required to provide the statement, we can skip the proof steps.
  -- sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_square_perimeter_l2007_200764


namespace NUMINAMATH_GPT_time_to_cross_platform_l2007_200734

-- Definitions from conditions
def train_speed_kmph : ℕ := 72
def speed_conversion_factor : ℕ := 1000 / 3600
def train_speed_mps : ℤ := train_speed_kmph * speed_conversion_factor
def time_cross_man_sec : ℕ := 16
def platform_length_meters : ℕ := 280

-- Proving the total time to cross platform
theorem time_to_cross_platform : ∃ t : ℕ, t = (platform_length_meters + (train_speed_mps * time_cross_man_sec)) / train_speed_mps ∧ t = 30 := 
by
  -- Since the proof isn't required, we add "sorry" to act as a placeholder.
  sorry

end NUMINAMATH_GPT_time_to_cross_platform_l2007_200734


namespace NUMINAMATH_GPT_point_D_coordinates_l2007_200760

-- Define the vectors and points
structure Point where
  x : Int
  y : Int

def vector_add (p1 p2 : Point) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

def scalar_multiply (k : Int) (p : Point) : Point :=
  { x := k * p.x, y := k * p.y }

def ab := Point.mk 5 (-3)
def c := Point.mk (-1) 3
def cd := scalar_multiply 2 ab

def D : Point := vector_add c cd

-- Theorem statement
theorem point_D_coordinates :
  D = Point.mk 9 (-3) :=
sorry

end NUMINAMATH_GPT_point_D_coordinates_l2007_200760


namespace NUMINAMATH_GPT_diff_squares_example_l2007_200780

theorem diff_squares_example :
  (311^2 - 297^2) / 14 = 608 :=
by
  -- The theorem statement directly follows from the conditions and question.
  sorry

end NUMINAMATH_GPT_diff_squares_example_l2007_200780


namespace NUMINAMATH_GPT_emily_patches_difference_l2007_200746

theorem emily_patches_difference (h p : ℕ) (h_eq : p = 3 * h) :
  (p * h) - ((p + 5) * (h - 3)) = (4 * h + 15) :=
by
  sorry

end NUMINAMATH_GPT_emily_patches_difference_l2007_200746


namespace NUMINAMATH_GPT_find_white_balls_l2007_200790

noncomputable def white_balls_in_bag (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) 
  (p_not_red_nor_purple : ℚ) : ℕ :=
total_balls - (red_balls + purple_balls) - (green_balls + yellow_balls)

theorem find_white_balls :
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  white_balls_in_bag total_balls green_balls yellow_balls red_balls purple_balls p_not_red_nor_purple = 21 :=
by
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  sorry

end NUMINAMATH_GPT_find_white_balls_l2007_200790


namespace NUMINAMATH_GPT_power_of_two_divides_factorial_iff_l2007_200732

theorem power_of_two_divides_factorial_iff (n : ℕ) (k : ℕ) : 2^(n - 1) ∣ n! ↔ n = 2^k := sorry

end NUMINAMATH_GPT_power_of_two_divides_factorial_iff_l2007_200732


namespace NUMINAMATH_GPT_calculate_expression_l2007_200706

theorem calculate_expression : 
  -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2007_200706


namespace NUMINAMATH_GPT_problem_x2_plus_y2_l2007_200782

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end NUMINAMATH_GPT_problem_x2_plus_y2_l2007_200782


namespace NUMINAMATH_GPT_right_triangle_perimeter_l2007_200761

noncomputable def perimeter_right_triangle (a b : ℝ) (hypotenuse : ℝ) : ℝ :=
  a + b + hypotenuse

theorem right_triangle_perimeter (a b : ℝ) (ha : a^2 + b^2 = 25) (hab : a * b = 10) (hhypotenuse : hypotenuse = 5) :
  perimeter_right_triangle a b hypotenuse = 5 + 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l2007_200761


namespace NUMINAMATH_GPT_joe_eggs_around_park_l2007_200794

variable (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ)

def joe_eggs (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ) : Prop :=
  total_eggs = club_house_eggs + town_hall_garden_eggs + park_eggs

theorem joe_eggs_around_park (h1 : total_eggs = 20) (h2 : club_house_eggs = 12) (h3 : town_hall_garden_eggs = 3) :
  ∃ park_eggs, joe_eggs total_eggs club_house_eggs town_hall_garden_eggs park_eggs ∧ park_eggs = 5 :=
by
  sorry

end NUMINAMATH_GPT_joe_eggs_around_park_l2007_200794


namespace NUMINAMATH_GPT_total_sequences_correct_l2007_200771

/-- 
Given 6 blocks arranged such that:
1. Block 1 must be removed first.
2. Blocks 2 and 3 become accessible after Block 1 is removed.
3. Blocks 4, 5, and 6 become accessible after Blocks 2 and 3 are removed.
4. A block can only be removed if no other block is stacked on top of it. 

Prove that the total number of possible sequences to remove all the blocks is 10.
-/
def total_sequences_to_remove_blocks : ℕ := 10

theorem total_sequences_correct : 
  total_sequences_to_remove_blocks = 10 :=
sorry

end NUMINAMATH_GPT_total_sequences_correct_l2007_200771


namespace NUMINAMATH_GPT_remaining_employees_earn_rate_l2007_200747

theorem remaining_employees_earn_rate
  (total_employees : ℕ)
  (employees_12_per_hour : ℕ)
  (employees_14_per_hour : ℕ)
  (total_cost : ℝ)
  (hourly_rate_12 : ℝ)
  (hourly_rate_14 : ℝ)
  (shift_hours : ℝ)
  (remaining_employees : ℕ)
  (remaining_hourly_rate : ℝ) :
  total_employees = 300 →
  employees_12_per_hour = 200 →
  employees_14_per_hour = 40 →
  total_cost = 31840 →
  hourly_rate_12 = 12 →
  hourly_rate_14 = 14 →
  shift_hours = 8 →
  remaining_employees = 60 →
  remaining_hourly_rate = 
    (total_cost - (employees_12_per_hour * hourly_rate_12 * shift_hours) - 
    (employees_14_per_hour * hourly_rate_14 * shift_hours)) / 
    (remaining_employees * shift_hours) →
  remaining_hourly_rate = 17 :=
by
  sorry

end NUMINAMATH_GPT_remaining_employees_earn_rate_l2007_200747


namespace NUMINAMATH_GPT_remainder_of_product_l2007_200742

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_l2007_200742


namespace NUMINAMATH_GPT_product_increase_l2007_200788

theorem product_increase (a b : ℝ) (h : (a + 1) * (b + 1) = 2 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a^2 - 1) * (b^2 - 1) = 4 * a * b) :=
sorry

end NUMINAMATH_GPT_product_increase_l2007_200788


namespace NUMINAMATH_GPT_unique_common_root_m_value_l2007_200753

theorem unique_common_root_m_value (m : ℝ) (h : m > 5) :
  (∃ x : ℝ, x^2 - 5 * x + 6 = 0 ∧ x^2 + 2 * x - 2 * m + 1 = 0) →
  m = 8 :=
by
  sorry

end NUMINAMATH_GPT_unique_common_root_m_value_l2007_200753


namespace NUMINAMATH_GPT_max_value_of_f_on_interval_exists_x_eq_min_1_l2007_200716

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f_on_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → f x ≤ 1 / 4 := sorry

theorem exists_x_eq_min_1 : 
  ∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f x = 1 / 4 := sorry

end NUMINAMATH_GPT_max_value_of_f_on_interval_exists_x_eq_min_1_l2007_200716


namespace NUMINAMATH_GPT_soap_bubble_radius_l2007_200727

/-- Given a spherical soap bubble that divides into two equal hemispheres, 
    each having a radius of 6 * (2 ^ (1 / 3)) cm, 
    show that the radius of the original bubble is also 6 * (2 ^ (1 / 3)) cm. -/
theorem soap_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h_r : r = 6 * (2 ^ (1 / 3)))
  (h_volume_eq : (4 / 3) * π * R^3 = (4 / 3) * π * r^3) : 
  R = 6 * (2 ^ (1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_soap_bubble_radius_l2007_200727


namespace NUMINAMATH_GPT_sum_of_coefficients_sum_even_odd_coefficients_l2007_200703

noncomputable def P (x : ℝ) : ℝ := (2 * x^2 - 2 * x + 1)^17 * (3 * x^2 - 3 * x + 1)^17

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

theorem sum_even_odd_coefficients :
  (P 1 + P (-1)) / 2 = (1 + 35^17) / 2 ∧ (P 1 - P (-1)) / 2 = (1 - 35^17) / 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_sum_even_odd_coefficients_l2007_200703


namespace NUMINAMATH_GPT_probability_picasso_consecutive_l2007_200715

-- Given Conditions
def total_pieces : Nat := 12
def picasso_paintings : Nat := 4

-- Desired probability calculation
theorem probability_picasso_consecutive :
  (Nat.factorial (total_pieces - picasso_paintings + 1) * Nat.factorial picasso_paintings) / 
  Nat.factorial total_pieces = 1 / 55 :=
by
  sorry

end NUMINAMATH_GPT_probability_picasso_consecutive_l2007_200715


namespace NUMINAMATH_GPT_chemistry_marks_l2007_200784

theorem chemistry_marks (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) 
                        (marks_biology : ℕ) (average_marks : ℚ) (marks_chemistry : ℕ) 
                        (h_english : marks_english = 70) 
                        (h_math : marks_math = 60) 
                        (h_physics : marks_physics = 78) 
                        (h_biology : marks_biology = 65) 
                        (h_average : average_marks = 66.6) 
                        (h_total: average_marks * 5 = marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) : 
  marks_chemistry = 60 :=
by sorry

end NUMINAMATH_GPT_chemistry_marks_l2007_200784


namespace NUMINAMATH_GPT_find_y_l2007_200709

variable (t : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := x = 3 - t
def condition2 : Prop := y = 2 * t + 11
def condition3 : Prop := x = 1

theorem find_y (h1 : condition1 x t) (h2 : condition2 t y) (h3 : condition3 x) : y = 15 := by
  sorry

end NUMINAMATH_GPT_find_y_l2007_200709


namespace NUMINAMATH_GPT_eccentricity_sum_cannot_be_2sqrt2_l2007_200791

noncomputable def e1 (a b : ℝ) := Real.sqrt (1 + (b^2) / (a^2))
noncomputable def e2 (a b : ℝ) := Real.sqrt (1 + (a^2) / (b^2))
noncomputable def e1_plus_e2 (a b : ℝ) := e1 a b + e2 a b

theorem eccentricity_sum_cannot_be_2sqrt2 (a b : ℝ) : e1_plus_e2 a b ≠ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_eccentricity_sum_cannot_be_2sqrt2_l2007_200791


namespace NUMINAMATH_GPT_even_function_increasing_l2007_200739

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * b * x + 1

theorem even_function_increasing (h_even : ∀ x : ℝ, f a b x = f a b (-x))
  (h_increasing : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → f a b x < f a b y) :
  f a b (a-2) < f a b (b+1) :=
sorry

end NUMINAMATH_GPT_even_function_increasing_l2007_200739


namespace NUMINAMATH_GPT_tangent_lines_to_circle_l2007_200705

-- Conditions
def regions_not_enclosed := 68
def num_lines := 30 - 4

-- Theorem statement
theorem tangent_lines_to_circle (h: regions_not_enclosed = 68) : num_lines = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_tangent_lines_to_circle_l2007_200705


namespace NUMINAMATH_GPT_equal_real_roots_quadratic_l2007_200723

theorem equal_real_roots_quadratic (k : ℝ) : (∀ x : ℝ, (x^2 + 2*x + k = 0)) → k = 1 :=
by
sorry

end NUMINAMATH_GPT_equal_real_roots_quadratic_l2007_200723


namespace NUMINAMATH_GPT_james_choices_count_l2007_200750

-- Define the conditions as Lean definitions
def isAscending (a b c d e : ℕ) : Prop := a < b ∧ b < c ∧ c < d ∧ d < e

def inRange (a b c d e : ℕ) : Prop := a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8 ∧ d ≤ 8 ∧ e ≤ 8

def meanEqualsMedian (a b c d e : ℕ) : Prop :=
  (a + b + c + d + e) / 5 = c

-- Define the problem statement
theorem james_choices_count :
  ∃ (s : Finset (ℕ × ℕ × ℕ × ℕ × ℕ)), 
    (∀ (a b c d e : ℕ), (a, b, c, d, e) ∈ s ↔ isAscending a b c d e ∧ inRange a b c d e ∧ meanEqualsMedian a b c d e) ∧
    s.card = 10 :=
sorry

end NUMINAMATH_GPT_james_choices_count_l2007_200750


namespace NUMINAMATH_GPT_incorrect_conclusions_l2007_200736

variables (a b : ℝ)

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem incorrect_conclusions :
  a > 0 → b > 0 → a ≠ 1 → b ≠ 1 → log_base a b > 1 →
  (a < 1 ∧ b > a ∨ (¬ (b < 1 ∧ b < a) ∧ ¬ (a < 1 ∧ a < b))) :=
by intros ha hb ha_ne1 hb_ne1 hlog; sorry

end NUMINAMATH_GPT_incorrect_conclusions_l2007_200736


namespace NUMINAMATH_GPT_sum_of_radii_of_tangent_circles_l2007_200758

theorem sum_of_radii_of_tangent_circles : 
  ∃ r1 r2 : ℝ, 
    r1 > 0 ∧
    r2 > 0 ∧
    ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
    ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧
    r1 + r2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_radii_of_tangent_circles_l2007_200758


namespace NUMINAMATH_GPT_domain_of_function_l2007_200765

noncomputable def domain_of_f : Set ℝ :=
  {x | x > -1/2 ∧ x ≠ 1}

theorem domain_of_function :
  (∀ x : ℝ, (2 * x + 1 ≥ 0) ∧ (2 * x^2 - x - 1 ≠ 0) ↔ (x > -1/2 ∧ x ≠ 1)) := by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2007_200765


namespace NUMINAMATH_GPT_remainder_when_sum_divided_mod7_l2007_200778

theorem remainder_when_sum_divided_mod7 (a b c : ℕ)
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (h7 : a * b * c % 7 = 2)
  (h8 : 3 * c % 7 = 1)
  (h9 : 4 * b % 7 = (2 + b) % 7) :
  (a + b + c) % 7 = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_mod7_l2007_200778


namespace NUMINAMATH_GPT_math_problem_l2007_200766

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39 / 37 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2007_200766


namespace NUMINAMATH_GPT_value_of_a_l2007_200792

theorem value_of_a (x a : ℤ) (h1 : x = 2) (h2 : 3 * x - a = -x + 7) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2007_200792


namespace NUMINAMATH_GPT_determine_angle_G_l2007_200785

theorem determine_angle_G 
  (C D E F G : ℝ)
  (hC : C = 120) 
  (h_linear_pair : C + D = 180)
  (hE : E = 50) 
  (hF : F = D) 
  (h_triangle_sum : E + F + G = 180) :
  G = 70 := 
sorry

end NUMINAMATH_GPT_determine_angle_G_l2007_200785


namespace NUMINAMATH_GPT_red_marbles_count_l2007_200755

noncomputable def total_marbles (R : ℕ) : ℕ := R + 16

noncomputable def P_blue (R : ℕ) : ℚ := 10 / (total_marbles R)

noncomputable def P_neither_blue (R : ℕ) : ℚ := (1 - P_blue R) * (1 - P_blue R)

noncomputable def P_either_blue (R : ℕ) : ℚ := 1 - P_neither_blue R

theorem red_marbles_count
  (R : ℕ) 
  (h1 : P_either_blue R = 0.75) :
  R = 4 :=
by
  sorry

end NUMINAMATH_GPT_red_marbles_count_l2007_200755


namespace NUMINAMATH_GPT_find_a_10_l2007_200779

theorem find_a_10 (a : ℕ → ℚ)
  (h0 : a 1 = 1)
  (h1 : ∀ n : ℕ, a (n + 1) = a n / (a n + 2)) :
  a 10 = 1 / 1023 :=
sorry

end NUMINAMATH_GPT_find_a_10_l2007_200779
