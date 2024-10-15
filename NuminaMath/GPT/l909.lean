import Mathlib

namespace NUMINAMATH_GPT_geomSeriesSum_eq_683_l909_90989

/-- Define the first term, common ratio, and number of terms -/
def firstTerm : ℤ := -1
def commonRatio : ℤ := -2
def numTerms : ℕ := 11

/-- Function to calculate the sum of the geometric series -/
def geomSeriesSum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r^n - 1) / (r - 1))

/-- The main theorem stating that the sum of the series equals 683 -/
theorem geomSeriesSum_eq_683 :
  geomSeriesSum firstTerm commonRatio numTerms = 683 :=
by sorry

end NUMINAMATH_GPT_geomSeriesSum_eq_683_l909_90989


namespace NUMINAMATH_GPT_compute_roots_sum_l909_90935

def roots_quadratic_eq_a_b (a b : ℂ) : Prop :=
  a^2 - 6 * a + 8 = 0 ∧ b^2 - 6 * b + 8 = 0

theorem compute_roots_sum (a b : ℂ) (ha : roots_quadratic_eq_a_b a b) :
  a^5 + a^3 * b^3 + b^5 = -568 := by
  sorry

end NUMINAMATH_GPT_compute_roots_sum_l909_90935


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l909_90983

theorem tenth_term_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), a 1 = 5/6 ∧ a 16 = 7/8 →
  a 10 = 103/120 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l909_90983


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l909_90937

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0} := 
by sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l909_90937


namespace NUMINAMATH_GPT_yardwork_payment_l909_90907

theorem yardwork_payment :
  let earnings := [15, 20, 25, 40]
  let total_earnings := List.sum earnings
  let equal_share := total_earnings / earnings.length
  let high_earner := 40
  high_earner - equal_share = 15 :=
by
  sorry

end NUMINAMATH_GPT_yardwork_payment_l909_90907


namespace NUMINAMATH_GPT_solve_inequality_l909_90954

theorem solve_inequality {x : ℝ} : (x^2 - 9 * x + 18 ≤ 0) ↔ 3 ≤ x ∧ x ≤ 6 :=
by
sorry

end NUMINAMATH_GPT_solve_inequality_l909_90954


namespace NUMINAMATH_GPT_annual_income_correct_l909_90990

def investment (amount : ℕ) := 6800
def dividend_rate (rate : ℕ) := 20
def stock_price (price : ℕ) := 136
def face_value : ℕ := 100
def calculate_annual_income (amount rate price value : ℕ) : ℕ := 
  let shares := amount / price
  let annual_income_per_share := value * rate / 100
  shares * annual_income_per_share

theorem annual_income_correct : calculate_annual_income (investment 6800) (dividend_rate 20) (stock_price 136) face_value = 1000 :=
by
  sorry

end NUMINAMATH_GPT_annual_income_correct_l909_90990


namespace NUMINAMATH_GPT_value_after_addition_l909_90977

theorem value_after_addition (x : ℕ) (h : x / 9 = 8) : x + 11 = 83 :=
by
  sorry

end NUMINAMATH_GPT_value_after_addition_l909_90977


namespace NUMINAMATH_GPT_number_of_oranges_l909_90923

theorem number_of_oranges (B T O : ℕ) (h₁ : B + T = 178) (h₂ : B + T + O = 273) : O = 95 :=
by
  -- Begin proof here
  sorry

end NUMINAMATH_GPT_number_of_oranges_l909_90923


namespace NUMINAMATH_GPT_exists_periodic_sequence_of_period_ge_two_l909_90972

noncomputable def periodic_sequence (x : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, x (n + p) = x n

theorem exists_periodic_sequence_of_period_ge_two :
  ∀ (p : ℕ), p ≥ 2 →
  ∃ (x : ℕ → ℝ), periodic_sequence x p ∧ 
  ∀ n, x (n + 1) = x n - (1 / x n) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_periodic_sequence_of_period_ge_two_l909_90972


namespace NUMINAMATH_GPT_no_three_in_range_l909_90958

theorem no_three_in_range (c : ℝ) : c > 4 → ¬ (∃ x : ℝ, x^2 + 2 * x + c = 3) :=
by
  sorry

end NUMINAMATH_GPT_no_three_in_range_l909_90958


namespace NUMINAMATH_GPT_janet_pills_monthly_l909_90980

def daily_intake_first_two_weeks := 2 + 3 -- 2 multivitamins + 3 calcium supplements
def daily_intake_last_two_weeks := 2 + 1 -- 2 multivitamins + 1 calcium supplement
def days_in_two_weeks := 2 * 7

theorem janet_pills_monthly :
  (daily_intake_first_two_weeks * days_in_two_weeks) + (daily_intake_last_two_weeks * days_in_two_weeks) = 112 :=
by
  sorry

end NUMINAMATH_GPT_janet_pills_monthly_l909_90980


namespace NUMINAMATH_GPT_valid_permutations_count_l909_90969

def num_permutations (seq : List ℕ) : ℕ :=
  -- A dummy implementation, the real function would calculate the number of valid permutations.
  sorry

theorem valid_permutations_count : num_permutations [1, 2, 3, 4, 5, 6] = 32 :=
by
  sorry

end NUMINAMATH_GPT_valid_permutations_count_l909_90969


namespace NUMINAMATH_GPT_wang_payment_correct_l909_90978

noncomputable def first_trip_payment (x : ℝ) : ℝ := 0.9 * x
noncomputable def second_trip_payment (y : ℝ) : ℝ := 300 * 0.9 + (y - 300) * 0.8

theorem wang_payment_correct (x y: ℝ) 
  (cond1: 0.1 * x = 19)
  (cond2: (x + y) - (0.9 * x + ((y - 300) * 0.8 + 300 * 0.9)) = 67) :
  first_trip_payment x = 171 ∧ second_trip_payment y = 342 := 
by
  sorry

end NUMINAMATH_GPT_wang_payment_correct_l909_90978


namespace NUMINAMATH_GPT_sum_of_series_eq_half_l909_90947

theorem sum_of_series_eq_half :
  (∑' k : ℕ, 3^(2^k) / (9^(2^k) - 1)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_series_eq_half_l909_90947


namespace NUMINAMATH_GPT_maximize_revenue_l909_90932

def revenue (p : ℝ) : ℝ := 150 * p - 4 * p^2

theorem maximize_revenue : 
  ∃ p, 0 ≤ p ∧ p ≤ 30 ∧ p = 18.75 ∧ (∀ q, 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue 18.75) :=
by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l909_90932


namespace NUMINAMATH_GPT_probability_at_least_one_boy_one_girl_l909_90942

def boys := 12
def girls := 18
def total_members := 30
def committee_size := 6

def total_ways := Nat.choose total_members committee_size
def all_boys_ways := Nat.choose boys committee_size
def all_girls_ways := Nat.choose girls committee_size
def all_boys_or_girls_ways := all_boys_ways + all_girls_ways
def complementary_probability := all_boys_or_girls_ways / total_ways
def desired_probability := 1 - complementary_probability

theorem probability_at_least_one_boy_one_girl :
  desired_probability = (574287 : ℚ) / 593775 :=
  sorry

end NUMINAMATH_GPT_probability_at_least_one_boy_one_girl_l909_90942


namespace NUMINAMATH_GPT_jack_sugar_usage_l909_90922

theorem jack_sugar_usage (initial_sugar bought_sugar final_sugar x : ℕ) 
  (h1 : initial_sugar = 65) 
  (h2 : bought_sugar = 50) 
  (h3 : final_sugar = 97) 
  (h4 : final_sugar = initial_sugar - x + bought_sugar) : 
  x = 18 := 
by 
  sorry

end NUMINAMATH_GPT_jack_sugar_usage_l909_90922


namespace NUMINAMATH_GPT_quadratic_function_properties_l909_90984

theorem quadratic_function_properties
    (f : ℝ → ℝ)
    (h_vertex : ∀ x, f x = -(x - 2)^2 + 1)
    (h_point : f (-1) = -8) :
  (∀ x, f x = -(x - 2)^2 + 1) ∧
  (f 1 = 0) ∧ (f 3 = 0) ∧ (f 0 = 1) :=
  by
    sorry

end NUMINAMATH_GPT_quadratic_function_properties_l909_90984


namespace NUMINAMATH_GPT_largest_angle_of_triangle_ABC_l909_90966

theorem largest_angle_of_triangle_ABC (a b c : ℝ)
  (h₁ : a + b + 2 * c = a^2) 
  (h₂ : a + b - 2 * c = -1) : 
  ∃ C : ℝ, C = 120 :=
sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_ABC_l909_90966


namespace NUMINAMATH_GPT_loss_percentage_is_26_l909_90908

/--
Given the cost price of a radio is Rs. 1500 and the selling price is Rs. 1110, 
prove that the loss percentage is 26%
-/
theorem loss_percentage_is_26 (cost_price selling_price : ℝ)
  (h₀ : cost_price = 1500)
  (h₁ : selling_price = 1110) :
  ((cost_price - selling_price) / cost_price) * 100 = 26 := 
by 
  sorry

end NUMINAMATH_GPT_loss_percentage_is_26_l909_90908


namespace NUMINAMATH_GPT_average_next_seven_consecutive_is_correct_l909_90955

-- Define the sum of seven consecutive integers starting at x.
def sum_seven_consecutive_integers (x : ℕ) : ℕ := 7 * x + 21

-- Define the next sequence of seven integers starting from y + 1.
def average_next_seven_consecutive_integers (x : ℕ) : ℕ :=
  let y := sum_seven_consecutive_integers x
  let start := y + 1
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) + (start + 6)) / 7

-- Problem statement
theorem average_next_seven_consecutive_is_correct (x : ℕ) : 
  average_next_seven_consecutive_integers x = 7 * x + 25 :=
by
  sorry

end NUMINAMATH_GPT_average_next_seven_consecutive_is_correct_l909_90955


namespace NUMINAMATH_GPT_green_dots_third_row_l909_90905

noncomputable def row_difference (a b : Nat) : Nat := b - a

theorem green_dots_third_row (a1 a2 a4 a5 a3 d : Nat)
  (h_a1 : a1 = 3)
  (h_a2 : a2 = 6)
  (h_a4 : a4 = 12)
  (h_a5 : a5 = 15)
  (h_d : row_difference a2 a1 = d)
  (h_d_consistent : row_difference a2 a1 = row_difference a4 a3) :
  a3 = 9 :=
sorry

end NUMINAMATH_GPT_green_dots_third_row_l909_90905


namespace NUMINAMATH_GPT_find_value_of_c_l909_90943

-- Mathematical proof problem in Lean 4 statement
theorem find_value_of_c (a b c d : ℝ)
  (h1 : a + c = 900)
  (h2 : b + c = 1100)
  (h3 : a + d = 700)
  (h4 : a + b + c + d = 2000) : 
  c = 200 :=
sorry

end NUMINAMATH_GPT_find_value_of_c_l909_90943


namespace NUMINAMATH_GPT_leila_total_cakes_l909_90902

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 :=
by sorry

end NUMINAMATH_GPT_leila_total_cakes_l909_90902


namespace NUMINAMATH_GPT_number_of_tulips_l909_90987

theorem number_of_tulips (T : ℕ) (roses : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (total_flowers : ℕ) (h1 : roses = 37) (h2 : used_flowers = 70) 
  (h3 : extra_flowers = 3) (h4: total_flowers = 73) 
  (h5 : T + roses = total_flowers) : T = 36 := 
by
  sorry

end NUMINAMATH_GPT_number_of_tulips_l909_90987


namespace NUMINAMATH_GPT_greatest_y_l909_90903

theorem greatest_y (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : y ≤ 24 :=
sorry

end NUMINAMATH_GPT_greatest_y_l909_90903


namespace NUMINAMATH_GPT_machine_made_8_shirts_today_l909_90930

-- Define the conditions
def shirts_per_minute : ℕ := 2
def minutes_worked_today : ℕ := 4

-- Define the expected number of shirts made today
def shirts_made_today : ℕ := shirts_per_minute * minutes_worked_today

-- The theorem stating that the shirts made today should be 8
theorem machine_made_8_shirts_today : shirts_made_today = 8 := by
  sorry

end NUMINAMATH_GPT_machine_made_8_shirts_today_l909_90930


namespace NUMINAMATH_GPT_initial_bottle_caps_l909_90945

theorem initial_bottle_caps (bought_caps total_caps initial_caps : ℕ) 
  (hb : bought_caps = 41) (ht : total_caps = 43):
  initial_caps = 2 :=
by
  have h : total_caps = initial_caps + bought_caps := sorry
  have ha : initial_caps = total_caps - bought_caps := sorry
  exact sorry

end NUMINAMATH_GPT_initial_bottle_caps_l909_90945


namespace NUMINAMATH_GPT_range_of_a_l909_90927

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ (-9 < a ∧ a < 5/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l909_90927


namespace NUMINAMATH_GPT_calculate_area_l909_90970

def leftmost_rectangle_area (height width : ℕ) : ℕ := height * width
def middle_rectangle_area (height width : ℕ) : ℕ := height * width
def rightmost_rectangle_area (height width : ℕ) : ℕ := height * width

theorem calculate_area : 
  let leftmost_segment_height := 7
  let bottom_width := 6
  let segment_above_3 := 3
  let segment_above_2 := 2
  let rightmost_width := 5
  leftmost_rectangle_area leftmost_segment_height bottom_width + 
  middle_rectangle_area segment_above_3 segment_above_3 + 
  rightmost_rectangle_area segment_above_2 rightmost_width = 
  61 := by
    sorry

end NUMINAMATH_GPT_calculate_area_l909_90970


namespace NUMINAMATH_GPT_intersection_A_B_l909_90906

def A := {x : ℝ | |x| < 1}
def B := {x : ℝ | -2 < x ∧ x < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l909_90906


namespace NUMINAMATH_GPT_total_cost_production_l909_90938

variable (FC MC : ℕ) (n : ℕ)

theorem total_cost_production : FC = 12000 → MC = 200 → n = 20 → (FC + MC * n = 16000) :=
by
  intro hFC hMC hn
  sorry

end NUMINAMATH_GPT_total_cost_production_l909_90938


namespace NUMINAMATH_GPT_mushroom_drying_l909_90957

theorem mushroom_drying (M M' : ℝ) (m1 m2 : ℝ) :
  M = 100 ∧ m1 = 0.01 * M ∧ m2 = 0.02 * M' ∧ m1 = 1 → M' = 50 :=
by
  sorry

end NUMINAMATH_GPT_mushroom_drying_l909_90957


namespace NUMINAMATH_GPT_max_value_of_function_is_seven_l909_90936

theorem max_value_of_function_is_seven:
  ∃ a: ℕ, (0 < a) ∧ 
  (∃ x: ℝ, (x + Real.sqrt (13 - 2 * a * x)) = 7 ∧
    ∀ y: ℝ, (y = x + Real.sqrt (13 - 2 * a * x)) → y ≤ 7) :=
sorry

end NUMINAMATH_GPT_max_value_of_function_is_seven_l909_90936


namespace NUMINAMATH_GPT_inverse_variation_y_at_x_l909_90934

variable (k x y : ℝ)

theorem inverse_variation_y_at_x :
  (∀ x y k, y = k / x → y = 6 → x = 3 → k = 18) → 
  k = 18 →
  x = 12 →
  y = 18 / 12 →
  y = 3 / 2 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_inverse_variation_y_at_x_l909_90934


namespace NUMINAMATH_GPT_group_1991_l909_90921

theorem group_1991 (n : ℕ) (h1 : 1 ≤ n) (h2 : 1991 = 2 * n ^ 2 - 1) : n = 32 := 
sorry

end NUMINAMATH_GPT_group_1991_l909_90921


namespace NUMINAMATH_GPT_four_digit_numbers_l909_90909

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end NUMINAMATH_GPT_four_digit_numbers_l909_90909


namespace NUMINAMATH_GPT_runners_speeds_and_track_length_l909_90912

/-- Given two runners α and β on a circular track starting at point P and running with uniform speeds,
when α reaches the halfway point Q, β is 16 meters behind α. At a later time, their positions are 
symmetric with respect to the diameter PQ. In 1 2/15 seconds, β reaches point Q, and 13 13/15 seconds later, 
α finishes the race. This theorem calculates the speeds of the runners and the distance of the lap. -/
theorem runners_speeds_and_track_length (x y : ℕ)
    (distance : ℝ)
    (runner_speed_alpha runner_speed_beta : ℝ) 
    (half_track_time_alpha half_track_time_beta : ℝ)
    (mirror_time_alpha mirror_time_beta : ℝ)
    (additional_time_beta : ℝ) :
    half_track_time_alpha = 16 ∧ 
    half_track_time_beta = (272/15) ∧ 
    mirror_time_alpha = (17/15) * (272/15 - 16/32) ∧ 
    mirror_time_beta = (17/15) ∧ 
    additional_time_beta = (13 + (13/15))  ∧ 
    runner_speed_beta = (15/2) ∧ 
    runner_speed_alpha = (85/10) ∧ 
    distance = 272 :=
  sorry

end NUMINAMATH_GPT_runners_speeds_and_track_length_l909_90912


namespace NUMINAMATH_GPT_find_numbers_l909_90988

theorem find_numbers :
  ∃ (a b c d : ℕ), 
  (a + 2 = 22) ∧ 
  (b - 2 = 22) ∧ 
  (c * 2 = 22) ∧ 
  (d / 2 = 22) ∧ 
  (a + b + c + d = 99) :=
sorry

end NUMINAMATH_GPT_find_numbers_l909_90988


namespace NUMINAMATH_GPT_travel_time_difference_in_minutes_l909_90941

/-
A bus travels at an average speed of 40 miles per hour.
We need to prove that the difference in travel time between a 360-mile trip and a 400-mile trip equals 60 minutes.
-/

theorem travel_time_difference_in_minutes 
  (speed : ℝ) (distance1 distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end NUMINAMATH_GPT_travel_time_difference_in_minutes_l909_90941


namespace NUMINAMATH_GPT_Chloe_wins_l909_90933

theorem Chloe_wins (C M : ℕ) (h_ratio : 8 * M = 3 * C) (h_Max : M = 9) : C = 24 :=
by {
    sorry
}

end NUMINAMATH_GPT_Chloe_wins_l909_90933


namespace NUMINAMATH_GPT_tangent_line_parallel_x_axis_l909_90900

def f (x : ℝ) : ℝ := x^4 - 4 * x

theorem tangent_line_parallel_x_axis :
  ∃ (m n : ℝ), (n = f m) ∧ (deriv f m = 0) ∧ (m, n) = (1, -3) := by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_x_axis_l909_90900


namespace NUMINAMATH_GPT_ratio_of_areas_l909_90928

theorem ratio_of_areas (side_length : ℝ) (h : side_length = 6) :
  let area_triangle := (side_length^2 * Real.sqrt 3) / 4
  let area_square := side_length^2
  (area_triangle / area_square) = Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l909_90928


namespace NUMINAMATH_GPT_inradius_plus_circumradius_le_height_l909_90993

theorem inradius_plus_circumradius_le_height {α β γ : ℝ} 
    (h : ℝ) (r R : ℝ)
    (h_triangle : α ≥ β ∧ β ≥ γ ∧ γ ≥ 0 ∧ α + β + γ = π )
    (h_non_obtuse : π / 2 ≥ α ∧ π / 2 ≥ β ∧ π / 2 ≥ γ)
    (h_greatest_height : true) -- Assuming this condition holds as given
    :
    r + R ≤ h :=
sorry

end NUMINAMATH_GPT_inradius_plus_circumradius_le_height_l909_90993


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l909_90926

variable (p : Prop)
variable (x : ℝ)

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l909_90926


namespace NUMINAMATH_GPT_card_game_fairness_l909_90999

theorem card_game_fairness :
  let deck_size := 52
  let aces := 2
  let total_pairings := Nat.choose deck_size aces  -- Number of ways to choose 2 positions from 52
  let tie_cases := deck_size - 1                  -- Number of ways for consecutive pairs
  let non_tie_outcomes := total_pairings - tie_cases
  non_tie_outcomes / 2 = non_tie_outcomes / 2
:= sorry

end NUMINAMATH_GPT_card_game_fairness_l909_90999


namespace NUMINAMATH_GPT_x_pow_12_eq_one_l909_90939

theorem x_pow_12_eq_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 :=
sorry

end NUMINAMATH_GPT_x_pow_12_eq_one_l909_90939


namespace NUMINAMATH_GPT_remainder_modulus_l909_90991

theorem remainder_modulus :
  (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_modulus_l909_90991


namespace NUMINAMATH_GPT_number_of_articles_l909_90918

-- Define the conditions
def gain := 1 / 9
def cp_one_article := 1  -- cost price of one article

-- Define the cost price for x articles
def cp (x : ℕ) := x * cp_one_article

-- Define the selling price for 45 articles
def sp (x : ℕ) := x / 45

-- Define the selling price equation considering gain
def sp_one_article := (cp_one_article * (1 + gain))

-- Main theorem to prove
theorem number_of_articles (x : ℕ) (h : sp x = sp_one_article) : x = 50 :=
by
  sorry

-- The theorem imports all necessary conditions and definitions and prepares the problem for proof.

end NUMINAMATH_GPT_number_of_articles_l909_90918


namespace NUMINAMATH_GPT_seq_an_general_term_and_sum_l909_90913

theorem seq_an_general_term_and_sum
  (a_n : ℕ → ℕ)
  (S : ℕ → ℕ)
  (T : ℕ → ℕ)
  (H1 : ∀ n, S n = 2 * a_n n - a_n 1)
  (H2 : ∃ d : ℕ, a_n 1 = d ∧ a_n 2 + 1 = a_n 1 + d ∧ a_n 3 = a_n 2 + d) :
  (∀ n, a_n n = 2^n) ∧ (∀ n, T n = n * 2^(n + 1) + 2 - 2^(n + 1)) := 
  by
  sorry

end NUMINAMATH_GPT_seq_an_general_term_and_sum_l909_90913


namespace NUMINAMATH_GPT_greatest_value_x_plus_y_l909_90981

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_greatest_value_x_plus_y_l909_90981


namespace NUMINAMATH_GPT_average_weight_of_boys_l909_90916

theorem average_weight_of_boys (n1 n2 : ℕ) (w1 w2 : ℚ) 
  (weight_avg_22_boys : w1 = 50.25) 
  (weight_avg_8_boys : w2 = 45.15) 
  (count_22_boys : n1 = 22) 
  (count_8_boys : n2 = 8) 
  : ((n1 * w1 + n2 * w2) / (n1 + n2) : ℚ) = 48.89 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_boys_l909_90916


namespace NUMINAMATH_GPT_percent_absent_is_correct_l909_90901

theorem percent_absent_is_correct (total_students boys girls absent_boys absent_girls : ℝ) 
(h1 : total_students = 100)
(h2 : boys = 50)
(h3 : girls = 50)
(h4 : absent_boys = boys * (1 / 5))
(h5 : absent_girls = girls * (1 / 4)):
  (absent_boys + absent_girls) / total_students * 100 = 22.5 :=
by 
  sorry

end NUMINAMATH_GPT_percent_absent_is_correct_l909_90901


namespace NUMINAMATH_GPT_find_x2_plus_y2_l909_90953

theorem find_x2_plus_y2 : ∀ (x y : ℝ),
  3 * x + 4 * y = 30 →
  x + 2 * y = 13 →
  x^2 + y^2 = 36.25 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l909_90953


namespace NUMINAMATH_GPT_area_percent_difference_l909_90967

theorem area_percent_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  percent_difference = 4 := 
by
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  sorry

end NUMINAMATH_GPT_area_percent_difference_l909_90967


namespace NUMINAMATH_GPT_divisibility_by_7_l909_90998

theorem divisibility_by_7 (n : ℕ) : (3^(2 * n + 1) + 2^(n + 2)) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_7_l909_90998


namespace NUMINAMATH_GPT_largest_angle_in_right_isosceles_triangle_l909_90995

theorem largest_angle_in_right_isosceles_triangle (X Y Z : Type) 
  (angle_X : ℝ) (angle_Y : ℝ) (angle_Z : ℝ) 
  (h1 : angle_X = 45) 
  (h2 : angle_Y = 90)
  (h3 : angle_Y + angle_X + angle_Z = 180) 
  (h4 : angle_X = angle_Z) : angle_Y = 90 := by 
  sorry

end NUMINAMATH_GPT_largest_angle_in_right_isosceles_triangle_l909_90995


namespace NUMINAMATH_GPT_avg_licks_l909_90996

theorem avg_licks (Dan Michael Sam David Lance : ℕ) 
  (hDan : Dan = 58) 
  (hMichael : Michael = 63) 
  (hSam : Sam = 70) 
  (hDavid : David = 70) 
  (hLance : Lance = 39) : 
  (Dan + Michael + Sam + David + Lance) / 5 = 60 :=
by 
  sorry

end NUMINAMATH_GPT_avg_licks_l909_90996


namespace NUMINAMATH_GPT_find_sum_A_B_C_l909_90979

theorem find_sum_A_B_C (A B C : ℤ)
  (h1 : ∀ x > 4, (x^2 : ℝ) / (A * x^2 + B * x + C) > 0.4)
  (h2 : A * (-2)^2 + B * (-2) + C = 0)
  (h3 : A * (3)^2 + B * (3) + C = 0)
  (h4 : 0.4 < 1 / (A : ℝ) ∧ 1 / (A : ℝ) < 1) :
  A + B + C = -12 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_A_B_C_l909_90979


namespace NUMINAMATH_GPT_minimum_value_of_y_l909_90904

noncomputable def y (x : ℝ) : ℝ :=
  x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_y : ∃ x > 0, y x = 49 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_y_l909_90904


namespace NUMINAMATH_GPT_candles_to_new_five_oz_l909_90986

theorem candles_to_new_five_oz 
  (h_wax_percent: ℝ)
  (h_candles_20oz_count: ℕ) 
  (h_candles_5oz_count: ℕ) 
  (h_candles_1oz_count: ℕ) 
  (h_candles_20oz_wax: ℝ) 
  (h_candles_5oz_wax: ℝ)
  (h_candles_1oz_wax: ℝ):
  h_wax_percent = 0.10 →
  h_candles_20oz_count = 5 →
  h_candles_5oz_count = 5 → 
  h_candles_1oz_count = 25 →
  h_candles_20oz_wax = 20 →
  h_candles_5oz_wax = 5 →
  h_candles_1oz_wax = 1 →
  (h_wax_percent * h_candles_20oz_wax * h_candles_20oz_count + 
   h_wax_percent * h_candles_5oz_wax * h_candles_5oz_count + 
   h_wax_percent * h_candles_1oz_wax * h_candles_1oz_count) / 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_candles_to_new_five_oz_l909_90986


namespace NUMINAMATH_GPT_polynomial_remainder_l909_90950

theorem polynomial_remainder (P : ℝ → ℝ) (h1 : P 19 = 16) (h2 : P 15 = 8) : 
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 15) * (x - 19) * Q x + 2 * x - 22 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l909_90950


namespace NUMINAMATH_GPT_min_value_fraction_sum_l909_90974

open Real

theorem min_value_fraction_sum (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 2) :
    ∃ m, m = (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ∧ m = 9/4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l909_90974


namespace NUMINAMATH_GPT_faster_train_pass_time_l909_90973

-- Defining the conditions
def length_of_train : ℕ := 45 -- length in meters
def speed_of_faster_train : ℕ := 45 -- speed in km/hr
def speed_of_slower_train : ℕ := 36 -- speed in km/hr

-- Define relative speed
def relative_speed := (speed_of_faster_train - speed_of_slower_train) * 5 / 18 -- converting km/hr to m/s

-- Total distance to pass (sum of lengths of both trains)
def total_passing_distance := (2 * length_of_train) -- 2 trains of 45 meters each

-- Calculate the time to pass the slower train
def time_to_pass := total_passing_distance / relative_speed

-- The theorem to prove
theorem faster_train_pass_time : time_to_pass = 36 := by
  -- This is where the proof would be placed
  sorry

end NUMINAMATH_GPT_faster_train_pass_time_l909_90973


namespace NUMINAMATH_GPT_age_of_vanya_and_kolya_l909_90949

theorem age_of_vanya_and_kolya (P V K : ℕ) (hP : P = 10)
  (hV : V = P - 1) (hK : K = P - 5 + 1) : V = 9 ∧ K = 6 :=
by
  sorry

end NUMINAMATH_GPT_age_of_vanya_and_kolya_l909_90949


namespace NUMINAMATH_GPT_translate_function_right_by_2_l909_90971

theorem translate_function_right_by_2 (x : ℝ) : 
  (∀ x, (x - 2) ^ 2 + (x - 2) = x ^ 2 - 3 * x + 2) := 
by 
  sorry

end NUMINAMATH_GPT_translate_function_right_by_2_l909_90971


namespace NUMINAMATH_GPT_part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l909_90985

-- Part I
def is_relevant_number (n m : ℕ) : Prop :=
  ∀ {P : Finset ℕ}, (P ⊆ (Finset.range (2*n + 1)) ∧ P.card = m) →
  ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem part_I_n_3_not_relevant :
  ¬ is_relevant_number 3 5 := sorry

theorem part_I_n_3_is_relevant :
  is_relevant_number 3 6 := sorry

-- Part II
theorem part_II (n m : ℕ) (h : is_relevant_number n m) : m - n - 3 ≥ 0 := sorry

-- Part III
theorem part_III_min_value_of_relevant_number (n : ℕ) : 
  ∃ m : ℕ, is_relevant_number n m ∧ ∀ k, is_relevant_number n k → m ≤ k := sorry

end NUMINAMATH_GPT_part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l909_90985


namespace NUMINAMATH_GPT_smaller_group_men_l909_90940

-- Define the main conditions of the problem
def men_work_days : ℕ := 36 * 18  -- 36 men for 18 days

-- Define the theorem we need to prove
theorem smaller_group_men (M : ℕ) (h: M * 72 = men_work_days) : M = 9 :=
by
  -- proof is not required
  sorry

end NUMINAMATH_GPT_smaller_group_men_l909_90940


namespace NUMINAMATH_GPT_geometric_sequence_m_value_l909_90992

theorem geometric_sequence_m_value 
  (a : ℕ → ℝ) (q : ℝ) (m : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a n = a 1 * q^(n-1))
  (h3 : |q| ≠ 1) 
  (h4 : a m = a 1 * a 2 * a 3 * a 4 * a 5) : 
  m = 11 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_m_value_l909_90992


namespace NUMINAMATH_GPT_smallest_b_for_undefined_inverse_mod_70_77_l909_90925

theorem smallest_b_for_undefined_inverse_mod_70_77 (b : ℕ) :
  (∀ k, k < b → k * 1 % 70 ≠ 1 ∧ k * 1 % 77 ≠ 1) ∧ (b * 1 % 70 ≠ 1) ∧ (b * 1 % 77 ≠ 1) → b = 7 :=
by sorry

end NUMINAMATH_GPT_smallest_b_for_undefined_inverse_mod_70_77_l909_90925


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_l909_90948

theorem sum_of_inserted_numbers (x y : ℝ) (r : ℝ) 
  (h1 : 4 * r = x) 
  (h2 : 4 * r^2 = y) 
  (h3 : (2 / y) = ((1 / x) + (1 / 16))) :
  x + y = 8 :=
sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_l909_90948


namespace NUMINAMATH_GPT_num_white_squares_in_24th_row_l909_90952

-- Define the function that calculates the total number of squares in the nth row
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

-- Define the function that calculates the number of white squares in the nth row
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2

-- Problem statement for the Lean 4 theorem
theorem num_white_squares_in_24th_row : white_squares 24 = 23 :=
by {
  -- Lean proof generation will be placed here
  sorry
}

end NUMINAMATH_GPT_num_white_squares_in_24th_row_l909_90952


namespace NUMINAMATH_GPT_final_probability_l909_90965

def total_cards := 52
def kings := 4
def aces := 4
def chosen_cards := 3

namespace probability

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def prob_three_kings : ℚ :=
  (4 / 52) * (3 / 51) * (2 / 50)

def prob_exactly_two_aces : ℚ :=
  (choose 4 2 * choose 48 1) / choose 52 3

def prob_exactly_three_aces : ℚ :=
  (choose 4 3) / choose 52 3

def prob_at_least_two_aces : ℚ :=
  prob_exactly_two_aces + prob_exactly_three_aces

def prob_three_kings_or_two_aces : ℚ :=
  prob_three_kings + prob_at_least_two_aces

theorem final_probability :
  prob_three_kings_or_two_aces = 6 / 425 :=
by
  sorry

end probability

end NUMINAMATH_GPT_final_probability_l909_90965


namespace NUMINAMATH_GPT_simplify_fraction_l909_90929

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l909_90929


namespace NUMINAMATH_GPT_taxi_fare_range_l909_90946

theorem taxi_fare_range (x : ℝ) (h : 12.5 + 2.4 * (x - 3) = 19.7) : 5 < x ∧ x ≤ 6 :=
by
  -- Given conditions and the equation, we need to prove the inequalities.
  have fare_eq : 12.5 + 2.4 * (x - 3) = 19.7 := h
  sorry

end NUMINAMATH_GPT_taxi_fare_range_l909_90946


namespace NUMINAMATH_GPT_tan_alpha_values_l909_90975

theorem tan_alpha_values (α : ℝ) (h : Real.sin α + Real.cos α = 7 / 5) : 
  (Real.tan α = 4 / 3) ∨ (Real.tan α = 3 / 4) := 
  sorry

end NUMINAMATH_GPT_tan_alpha_values_l909_90975


namespace NUMINAMATH_GPT_insects_legs_l909_90919

theorem insects_legs (n : ℕ) (l : ℕ) (h₁ : n = 6) (h₂ : l = 6) : n * l = 36 :=
by sorry

end NUMINAMATH_GPT_insects_legs_l909_90919


namespace NUMINAMATH_GPT_hypotenuse_of_isosceles_right_triangle_l909_90963

theorem hypotenuse_of_isosceles_right_triangle (a : ℝ) (hyp : a = 8) : 
  ∃ c : ℝ, c = a * Real.sqrt 2 :=
by
  use 8 * Real.sqrt 2
  sorry

end NUMINAMATH_GPT_hypotenuse_of_isosceles_right_triangle_l909_90963


namespace NUMINAMATH_GPT_correct_operation_is_C_l909_90960

/--
Given the following statements:
1. \( a^3 \cdot a^2 = a^6 \)
2. \( (2a^3)^3 = 6a^9 \)
3. \( -6x^5 \div 2x^3 = -3x^2 \)
4. \( (-x-2)(x-2) = x^2 - 4 \)

Prove that the correct statement is \( -6x^5 \div 2x^3 = -3x^2 \) and the other statements are incorrect.
-/
theorem correct_operation_is_C (a x : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧
  ((2 * a^3)^3 ≠ 6 * a^9) ∧
  (-6 * x^5 / (2 * x^3) = -3 * x^2) ∧
  ((-x - 2) * (x - 2) ≠ x^2 - 4) := by
  sorry

end NUMINAMATH_GPT_correct_operation_is_C_l909_90960


namespace NUMINAMATH_GPT_general_term_sequence_l909_90931

def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 3 ∧ a 1 = 9 ∧ ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2) - 4 * n + 2

theorem general_term_sequence (a : ℕ → ℤ) (h : seq a) : 
  ∀ n, a n = 3^n + n^2 + 3 * n + 2 :=
by
  sorry

end NUMINAMATH_GPT_general_term_sequence_l909_90931


namespace NUMINAMATH_GPT_total_children_count_l909_90976

theorem total_children_count (boys girls : ℕ) (hb : boys = 40) (hg : girls = 77) : boys + girls = 117 := by
  sorry

end NUMINAMATH_GPT_total_children_count_l909_90976


namespace NUMINAMATH_GPT_sqrt_condition_l909_90910

theorem sqrt_condition (x : ℝ) : (x - 3 ≥ 0) ↔ (x = 3) :=
by sorry

end NUMINAMATH_GPT_sqrt_condition_l909_90910


namespace NUMINAMATH_GPT_value_of_f_2018_l909_90914

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 3) * f x = -1
axiom initial_condition : f (-1) = 2

theorem value_of_f_2018 : f 2018 = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_2018_l909_90914


namespace NUMINAMATH_GPT_opposite_of_negative_2023_l909_90915

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_2023_l909_90915


namespace NUMINAMATH_GPT_jose_initial_caps_l909_90961

-- Definition of conditions and the problem
def jose_starting_caps : ℤ :=
  let final_caps := 9
  let caps_from_rebecca := 2
  final_caps - caps_from_rebecca

-- Lean theorem to state the required proof
theorem jose_initial_caps : jose_starting_caps = 7 := by
  -- skip proof
  sorry

end NUMINAMATH_GPT_jose_initial_caps_l909_90961


namespace NUMINAMATH_GPT_inverse_function_problem_l909_90920

theorem inverse_function_problem
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h₁ : ∀ x, f (f_inv x) = x)
  (h₂ : ∀ x, f_inv (f x) = x)
  (a b : ℝ)
  (h₃ : f_inv (a - 1) + f_inv (b - 1) = 1) :
  f (a * b) = 3 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_problem_l909_90920


namespace NUMINAMATH_GPT_union_of_sets_l909_90964

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5, 7}
def union_result : Set ℕ := {1, 3, 5, 7}

theorem union_of_sets : A ∪ B = union_result := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l909_90964


namespace NUMINAMATH_GPT_probability_first_hearts_second_ace_correct_l909_90994

noncomputable def probability_first_hearts_second_ace : ℚ :=
  let total_cards := 104
  let total_aces := 8 -- 4 aces per deck, 2 decks
  let hearts_count := 2 * 13 -- 13 hearts per deck, 2 decks
  let ace_of_hearts_count := 2

  -- Case 1: the first is an ace of hearts
  let prob_first_ace_of_hearts := (ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_ace_of_hearts := (total_aces - 1 : ℚ) / (total_cards - 1)

  -- Case 2: the first is a hearts but not an ace
  let prob_first_hearts_not_ace := (hearts_count - ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_hearts_not_ace := total_aces / (total_cards - 1)

  -- Combined probability
  (prob_first_ace_of_hearts * prob_second_ace_given_first_ace_of_hearts) +
  (prob_first_hearts_not_ace * prob_second_ace_given_first_hearts_not_ace)

theorem probability_first_hearts_second_ace_correct : 
  probability_first_hearts_second_ace = 7 / 453 := 
sorry

end NUMINAMATH_GPT_probability_first_hearts_second_ace_correct_l909_90994


namespace NUMINAMATH_GPT_fifty_third_card_is_A_l909_90982

noncomputable def card_seq : List String := 
  ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

theorem fifty_third_card_is_A : card_seq[(53 % 13)] = "A" := 
by 
  simp [card_seq] 
  sorry

end NUMINAMATH_GPT_fifty_third_card_is_A_l909_90982


namespace NUMINAMATH_GPT_muffins_baked_by_James_correct_l909_90944

noncomputable def muffins_baked_by_James (muffins_baked_by_Arthur : ℝ) (ratio : ℝ) : ℝ :=
  muffins_baked_by_Arthur / ratio

theorem muffins_baked_by_James_correct :
  muffins_baked_by_James 115.0 12.0 = 9.5833 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_muffins_baked_by_James_correct_l909_90944


namespace NUMINAMATH_GPT_parallel_ne_implies_value_l909_90959

theorem parallel_ne_implies_value 
  (x : ℝ) 
  (m : ℝ × ℝ := (2 * x, 7)) 
  (n : ℝ × ℝ := (6, x + 4)) 
  (h1 : 2 * x * (x + 4) = 42) 
  (h2 : m ≠ n) :
  x = -7 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_ne_implies_value_l909_90959


namespace NUMINAMATH_GPT_coffee_shop_lattes_l909_90951

theorem coffee_shop_lattes (x : ℕ) (number_of_teas number_of_lattes : ℕ)
  (h1 : number_of_teas = 6)
  (h2 : number_of_lattes = 32)
  (h3 : number_of_lattes = x * number_of_teas + 8) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_coffee_shop_lattes_l909_90951


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l909_90911

variable (height : ℝ) (alpha : ℝ)

theorem radius_of_inscribed_circle (h : ℝ) (α : ℝ) : 
∃ r : ℝ, r = (h / 2) * (Real.tan (Real.pi / 4 - α / 4)) ^ 2 := 
sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l909_90911


namespace NUMINAMATH_GPT_geometric_series_sum_eq_l909_90917

theorem geometric_series_sum_eq :
  let a := (1/3 : ℚ)
  let r := (1/3 : ℚ)
  let n := 8
  let S := a * (1 - r^n) / (1 - r)
  S = 3280 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_eq_l909_90917


namespace NUMINAMATH_GPT_wall_width_l909_90962

theorem wall_width
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ)
  (num_bricks : ℕ)
  (brick_volume : ℝ := brick_length * brick_width * brick_height)
  (total_volume : ℝ := num_bricks * brick_volume) :
  brick_length = 0.20 → brick_width = 0.10 → brick_height = 0.08 →
  wall_length = 10 → wall_height = 8 → num_bricks = 12250 →
  total_volume = wall_length * wall_height * (0.245 : ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_wall_width_l909_90962


namespace NUMINAMATH_GPT_vector_properties_l909_90968

/-- The vectors a, b, and c used in the problem. --/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-4, 2)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  ((∃ k : ℝ, b = k • a) ∧ (b.1 * c.1 + b.2 * c.2 = 0) ∧ (a.1*a.1 + a.2*a.2 = c.1*c.1 + c.2*c.2)) :=
  by sorry

end NUMINAMATH_GPT_vector_properties_l909_90968


namespace NUMINAMATH_GPT_quadratic_function_two_distinct_roots_l909_90956

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks the conditions for the quadratic to have two distinct real roots
theorem quadratic_function_two_distinct_roots (a : ℝ) : 
  (0 < a ∧ a < 2) → (discriminant a (-4) 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_two_distinct_roots_l909_90956


namespace NUMINAMATH_GPT_divisible_sum_or_difference_l909_90924

theorem divisible_sum_or_difference (a : Fin 52 → ℤ) :
  ∃ i j, (i ≠ j) ∧ (a i + a j) % 100 = 0 ∨ (a i - a j) % 100 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisible_sum_or_difference_l909_90924


namespace NUMINAMATH_GPT_factor_expression_l909_90997

theorem factor_expression (x : ℝ) : 
  3 * x^2 * (x - 5) + 4 * x * (x - 5) + 6 * (x - 5) = (3 * x^2 + 4 * x + 6) * (x - 5) :=
  sorry

end NUMINAMATH_GPT_factor_expression_l909_90997
