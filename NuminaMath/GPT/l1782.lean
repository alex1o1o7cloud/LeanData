import Mathlib

namespace NUMINAMATH_GPT_ratio_of_liquid_p_to_q_initial_l1782_178299

noncomputable def initial_ratio_of_p_to_q : ℚ :=
  let p := 20
  let q := 15
  p / q

theorem ratio_of_liquid_p_to_q_initial
  (p q : ℚ)
  (h1 : p + q = 35)
  (h2 : p / (q + 13) = 5 / 7) :
  p / q = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_liquid_p_to_q_initial_l1782_178299


namespace NUMINAMATH_GPT_john_makes_money_l1782_178286

-- Definitions of the conditions
def num_cars := 5
def time_first_3_cars := 3 * 40 -- 3 cars each take 40 minutes
def time_remaining_car := 40 * 3 / 2 -- Each remaining car takes 50% longer
def time_remaining_cars := 2 * time_remaining_car -- 2 remaining cars
def total_time_min := time_first_3_cars + time_remaining_cars
def total_time_hr := total_time_min / 60 -- Convert total time from minutes to hours
def rate_per_hour := 20

-- Theorem statement
theorem john_makes_money : total_time_hr * rate_per_hour = 80 := by
  sorry

end NUMINAMATH_GPT_john_makes_money_l1782_178286


namespace NUMINAMATH_GPT_cylinder_height_l1782_178206

theorem cylinder_height (base_area : ℝ) (h s : ℝ)
  (h_base : base_area > 0)
  (h_ratio : (1 / 3 * base_area * 4.5) / (base_area * h) = 1 / 6)
  (h_cone_height : s = 4.5) :
  h = 9 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cylinder_height_l1782_178206


namespace NUMINAMATH_GPT_lowest_total_points_l1782_178293

-- Five girls and their respective positions
inductive Girl where
  | Fiona
  | Gertrude
  | Hannah
  | India
  | Janice
  deriving DecidableEq, Repr, Inhabited

open Girl

-- Initial position mapping
def initial_position : Girl → Nat
  | Fiona => 1
  | Gertrude => 2
  | Hannah => 3
  | India => 4
  | Janice => 5

-- Final position mapping
def final_position : Girl → Nat
  | Fiona => 3
  | Gertrude => 2
  | Hannah => 5
  | India => 1
  | Janice => 4

-- Define a function to calculate points for given initial and final positions
def points_awarded (g : Girl) : Nat :=
  initial_position g - final_position g

-- Define a function to calculate the total number of points
def total_points : Nat :=
  points_awarded Fiona + points_awarded Gertrude + points_awarded Hannah + points_awarded India + points_awarded Janice

theorem lowest_total_points : total_points = 5 :=
by
  -- Placeholder to skip the proof steps
  sorry

end NUMINAMATH_GPT_lowest_total_points_l1782_178293


namespace NUMINAMATH_GPT_sqrt_sum_ge_two_l1782_178254

theorem sqrt_sum_ge_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_ge_two_l1782_178254


namespace NUMINAMATH_GPT_min_value_of_expression_min_value_achieved_at_l1782_178201

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 
  3 * Real.sqrt x + 4 / (x^2) ≥ 4 * 4^(1/5) :=
sorry

theorem min_value_achieved_at (x : ℝ) (hx : 0 < x) (h : x = 4^(2/5)) :
  3 * Real.sqrt x + 4 / (x^2) = 4 * 4^(1/5) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_min_value_achieved_at_l1782_178201


namespace NUMINAMATH_GPT_perfect_square_m_value_l1782_178289

theorem perfect_square_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ x : ℝ, (x^2 + (m : ℝ)*x + 1 : ℝ) = (x + (a : ℝ))^2) → m = 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_m_value_l1782_178289


namespace NUMINAMATH_GPT_percentage_divisible_by_7_l1782_178247

-- Define the total integers and the condition for being divisible by 7
def total_ints := 140
def divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

-- Calculate the number of integers between 1 and 140 that are divisible by 7
def count_divisible_by_7 : ℕ := Nat.succ (140 / 7)

-- The theorem to prove
theorem percentage_divisible_by_7 : (count_divisible_by_7 / total_ints : ℚ) * 100 = 14.29 := by
  sorry

end NUMINAMATH_GPT_percentage_divisible_by_7_l1782_178247


namespace NUMINAMATH_GPT_probability_correct_l1782_178284

noncomputable def probability_B1_eq_5_given_WB : ℚ :=
  let P_B1_eq_5 : ℚ := 1 / 8
  let P_WB : ℚ := 1 / 5
  let P_WB_given_B1_eq_5 : ℚ := 1 / 16 + 369 / 2048
  (P_B1_eq_5 * P_WB_given_B1_eq_5) / P_WB

theorem probability_correct :
  probability_B1_eq_5_given_WB = 115 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_probability_correct_l1782_178284


namespace NUMINAMATH_GPT_roots_difference_squared_l1782_178265

theorem roots_difference_squared
  {Φ ϕ : ℝ}
  (hΦ : Φ^2 - Φ - 2 = 0)
  (hϕ : ϕ^2 - ϕ - 2 = 0)
  (h_diff : Φ ≠ ϕ) :
  (Φ - ϕ)^2 = 9 :=
by sorry

end NUMINAMATH_GPT_roots_difference_squared_l1782_178265


namespace NUMINAMATH_GPT_ratio_a_to_c_l1782_178239

-- Declaring the variables a, b, c, and d as real numbers.
variables (a b c d : ℝ)

-- Define the conditions given in the problem.
def ratio_conditions : Prop :=
  (a / b = 5 / 4) ∧ (c / d = 4 / 3) ∧ (d / b = 1 / 5)

-- State the theorem we need to prove based on the conditions.
theorem ratio_a_to_c (h : ratio_conditions a b c d) : a / c = 75 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_a_to_c_l1782_178239


namespace NUMINAMATH_GPT_find_c_l1782_178212

theorem find_c 
  (b c : ℝ) 
  (h1 : 4 = 2 * (1:ℝ)^2 + b * (1:ℝ) + c)
  (h2 : 4 = 2 * (5:ℝ)^2 + b * (5:ℝ) + c) : 
  c = 14 := 
sorry

end NUMINAMATH_GPT_find_c_l1782_178212


namespace NUMINAMATH_GPT_time_after_10000_seconds_l1782_178252

def time_add_seconds (h m s : Nat) (t : Nat) : (Nat × Nat × Nat) :=
  let total_seconds := h * 3600 + m * 60 + s + t
  let hours := (total_seconds / 3600) % 24
  let minutes := (total_seconds % 3600) / 60
  let seconds := (total_seconds % 3600) % 60
  (hours, minutes, seconds)

theorem time_after_10000_seconds :
  time_add_seconds 5 45 0 10000 = (8, 31, 40) :=
by
  sorry

end NUMINAMATH_GPT_time_after_10000_seconds_l1782_178252


namespace NUMINAMATH_GPT_sequence_difference_l1782_178279

theorem sequence_difference : 
  (∃ (a : ℕ → ℤ) (S : ℕ → ℤ), 
    (∀ n : ℕ, S n = n^2 + 2 * n) ∧ 
    (∀ n : ℕ, n > 0 → a n = S n - S (n - 1) ) ∧ 
    (a 4 - a 2 = 4)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_difference_l1782_178279


namespace NUMINAMATH_GPT_ordered_quadruple_ellipse_l1782_178288

noncomputable def ellipse_quadruple := 
  let f₁ : (ℝ × ℝ) := (1, 1)
  let f₂ : (ℝ × ℝ) := (1, 7)
  let p : (ℝ × ℝ) := (12, -1)
  let a := (5 / 2) * (Real.sqrt 5 + Real.sqrt 37)
  let b := (1 / 2) * Real.sqrt (1014 + 50 * Real.sqrt 185)
  let h := 1
  let k := 4
  (a, b, h, k)

theorem ordered_quadruple_ellipse :
  let e : (ℝ × ℝ × ℝ × ℝ) := θse_quadruple
  e = ((5 / 2 * (Real.sqrt 5 + Real.sqrt 37)), (1 / 2 * Real.sqrt (1014 + 50 * Real.sqrt 185)), 1, 4) :=
by
  sorry

end NUMINAMATH_GPT_ordered_quadruple_ellipse_l1782_178288


namespace NUMINAMATH_GPT_similar_triangles_x_value_l1782_178259

theorem similar_triangles_x_value
  (x : ℝ)
  (h_similar : ∀ (AB BC DE EF : ℝ), AB / BC = DE / EF)
  (h_AB : AB = x)
  (h_BC : BC = 33)
  (h_DE : DE = 96)
  (h_EF : EF = 24) :
  x = 132 :=
by
  -- Proof steps will be here
  sorry

end NUMINAMATH_GPT_similar_triangles_x_value_l1782_178259


namespace NUMINAMATH_GPT_find_k_l1782_178258

theorem find_k (k : ℕ) (h_pos : k > 0) (h_coef : 15 * k^4 < 120) : k = 1 :=
sorry

end NUMINAMATH_GPT_find_k_l1782_178258


namespace NUMINAMATH_GPT_mary_total_nickels_l1782_178256

-- Define the initial number of nickels Mary had
def mary_initial_nickels : ℕ := 7

-- Define the number of nickels her dad gave her
def mary_received_nickels : ℕ := 5

-- The goal is to prove the total number of nickels Mary has now is 12
theorem mary_total_nickels : mary_initial_nickels + mary_received_nickels = 12 :=
by
  sorry

end NUMINAMATH_GPT_mary_total_nickels_l1782_178256


namespace NUMINAMATH_GPT_reciprocal_of_neg4_is_neg_one_fourth_l1782_178210

theorem reciprocal_of_neg4_is_neg_one_fourth (x : ℝ) (h : x * -4 = 1) : x = -1/4 := 
by 
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg4_is_neg_one_fourth_l1782_178210


namespace NUMINAMATH_GPT_range_of_a_in_third_quadrant_l1782_178216

theorem range_of_a_in_third_quadrant (a : ℝ) :
  let Z_re := a^2 - 2*a
  let Z_im := a^2 - a - 2
  (Z_re < 0 ∧ Z_im < 0) → 0 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_in_third_quadrant_l1782_178216


namespace NUMINAMATH_GPT_greatest_possible_x_l1782_178231

theorem greatest_possible_x : ∃ (x : ℕ), (x^2 + 5 < 30) ∧ ∀ (y : ℕ), (y^2 + 5 < 30) → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_x_l1782_178231


namespace NUMINAMATH_GPT_average_age_of_population_l1782_178277

theorem average_age_of_population
  (k : ℕ)
  (ratio_women_men : 7 * (k : ℕ) = 7 * (k : ℕ) + 5 * (k : ℕ) - 5 * (k : ℕ))
  (avg_age_women : ℝ := 38)
  (avg_age_men : ℝ := 36)
  : ( (7 * k * avg_age_women) + (5 * k * avg_age_men) ) / (12 * k) = 37 + (1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_population_l1782_178277


namespace NUMINAMATH_GPT_bucket_B_more_than_C_l1782_178257

-- Define the number of pieces of fruit in bucket B as a constant
def B := 12

-- Define the number of pieces of fruit in bucket C as a constant
def C := 9

-- Define the number of pieces of fruit in bucket A based on B
def A := B + 4

-- Define the total number of pieces of fruit in all three buckets
def total_fruit := A + B + C

-- Prove that bucket B has 3 more pieces of fruit than bucket C
theorem bucket_B_more_than_C : B - C = 3 := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_bucket_B_more_than_C_l1782_178257


namespace NUMINAMATH_GPT_proportional_parts_middle_l1782_178248

theorem proportional_parts_middle (x : ℚ) (hx : x + (1/2) * x + (1/4) * x = 120) : (1/2) * x = 240 / 7 :=
by
  sorry

end NUMINAMATH_GPT_proportional_parts_middle_l1782_178248


namespace NUMINAMATH_GPT_area_within_fence_l1782_178204

def length_rectangle : ℕ := 15
def width_rectangle : ℕ := 12
def side_cutout_square : ℕ := 3

theorem area_within_fence : (length_rectangle * width_rectangle) - (side_cutout_square * side_cutout_square) = 171 := by
  sorry

end NUMINAMATH_GPT_area_within_fence_l1782_178204


namespace NUMINAMATH_GPT_triangle_side_lengths_l1782_178241

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ (a^2 + b^2 + c^2 = 2050) ∧ (c^2 = a^2 + b^2)

theorem triangle_side_lengths :
  ∃ b c : ℝ, side_lengths 10 b c ∧ b = Real.sqrt 925 ∧ c = Real.sqrt 1025 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l1782_178241


namespace NUMINAMATH_GPT_range_of_a_l1782_178280

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def strictly_increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (m n : ℝ) (h_even : is_even_function f)
  (h_strict : strictly_increasing_on_nonnegative f)
  (h_m : m = 1/2) (h_f : ∀ x, m ≤ x ∧ x ≤ n → f (a * x + 1) ≤ f 2) :
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1782_178280


namespace NUMINAMATH_GPT_find_a_from_coefficient_l1782_178215

theorem find_a_from_coefficient :
  (∀ x : ℝ, (x + 1)^6 * (a*x - 1)^2 = 20 → a = 0 ∨ a = 5) :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_coefficient_l1782_178215


namespace NUMINAMATH_GPT_measure_of_angle_F_l1782_178243

theorem measure_of_angle_F (D E F : ℝ) (hD : D = E) 
  (hF : F = D + 40) (h_sum : D + E + F = 180) : F = 140 / 3 + 40 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_F_l1782_178243


namespace NUMINAMATH_GPT_quotient_of_0_009_div_0_3_is_0_03_l1782_178244

-- Statement:
theorem quotient_of_0_009_div_0_3_is_0_03 (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_0_009_div_0_3_is_0_03_l1782_178244


namespace NUMINAMATH_GPT_speed_of_boat_is_correct_l1782_178236

theorem speed_of_boat_is_correct (t : ℝ) (V_b : ℝ) (V_s : ℝ) 
  (h1 : V_s = 19) 
  (h2 : ∀ t, (V_b - V_s) * (2 * t) = (V_b + V_s) * t) :
  V_b = 57 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_speed_of_boat_is_correct_l1782_178236


namespace NUMINAMATH_GPT_derivative_at_one_l1782_178203

theorem derivative_at_one (f : ℝ → ℝ) (df : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2) 
  (h₂ : ∀ x, df x = 2 * x) : 
  df 1 = 2 :=
by sorry

end NUMINAMATH_GPT_derivative_at_one_l1782_178203


namespace NUMINAMATH_GPT_count_five_digit_multiples_of_five_l1782_178262

theorem count_five_digit_multiples_of_five : 
  ∃ (n : ℕ), n = 18000 ∧ (∀ x, 10000 ≤ x ∧ x ≤ 99999 ∧ x % 5 = 0 ↔ ∃ k, 10000 ≤ 5 * k ∧ 5 * k ≤ 99999) :=
by
  sorry

end NUMINAMATH_GPT_count_five_digit_multiples_of_five_l1782_178262


namespace NUMINAMATH_GPT_origami_papers_total_l1782_178271

-- Define the conditions as Lean definitions
def num_cousins : ℕ := 6
def papers_per_cousin : ℕ := 8

-- Define the total number of origami papers that Haley has to give away
def total_papers : ℕ := num_cousins * papers_per_cousin

-- Statement of the proof
theorem origami_papers_total : total_papers = 48 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_origami_papers_total_l1782_178271


namespace NUMINAMATH_GPT_num_paths_from_E_to_G_pass_through_F_l1782_178276

-- Definitions for the positions on the grid.
def E := (0, 4)
def G := (5, 0)
def F := (3, 3)

-- Function to calculate the number of combinations.
def binom (n k: ℕ) : ℕ := Nat.choose n k

-- The mathematical statement to be proven.
theorem num_paths_from_E_to_G_pass_through_F :
  (binom 4 1) * (binom 5 2) = 40 :=
by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_num_paths_from_E_to_G_pass_through_F_l1782_178276


namespace NUMINAMATH_GPT_swimming_pool_water_remaining_l1782_178260

theorem swimming_pool_water_remaining :
  let initial_water := 500 -- initial water in gallons
  let evaporation_rate := 1.5 -- water loss due to evaporation in gallons/day
  let leak_rate := 0.8 -- water loss due to leak in gallons/day
  let total_days := 20 -- total number of days

  let total_daily_loss := evaporation_rate + leak_rate -- total daily loss in gallons/day
  let total_loss := total_daily_loss * total_days -- total loss over the period in gallons
  let remaining_water := initial_water - total_loss -- remaining water after 20 days in gallons

  remaining_water = 454 :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_water_remaining_l1782_178260


namespace NUMINAMATH_GPT_scientific_notation_of_448000_l1782_178296

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_448000_l1782_178296


namespace NUMINAMATH_GPT_smallest_yellow_marbles_l1782_178298

def total_marbles (n : ℕ) := n

def blue_marbles (n : ℕ) := n / 3

def red_marbles (n : ℕ) := n / 4

def green_marbles := 6

def yellow_marbles (n : ℕ) := n - (blue_marbles n + red_marbles n + green_marbles)

theorem smallest_yellow_marbles (n : ℕ) (hn : n % 12 = 0) (blue : blue_marbles n = n / 3)
  (red : red_marbles n = n / 4) (green : green_marbles = 6) :
  yellow_marbles n = 4 ↔ n = 24 :=
by sorry

end NUMINAMATH_GPT_smallest_yellow_marbles_l1782_178298


namespace NUMINAMATH_GPT_transform_polynomial_l1782_178282

variables {x y : ℝ}

theorem transform_polynomial (h : y = x - 1 / x) :
  (x^6 + x^5 - 5 * x^4 + 2 * x^3 - 5 * x^2 + x + 1 = 0) ↔ (x^2 * (y^2 + y - 3) = 0) :=
sorry

end NUMINAMATH_GPT_transform_polynomial_l1782_178282


namespace NUMINAMATH_GPT_value_of_m_l1782_178255

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_m_l1782_178255


namespace NUMINAMATH_GPT_compound_interest_second_year_l1782_178269

variables {P r CI_2 CI_3 : ℝ}

-- Given conditions as definitions in Lean
def interest_rate : ℝ := 0.05
def year_3_interest : ℝ := 1260
def relation_between_CI2_and_CI3 (CI_2 CI_3 : ℝ) : Prop :=
  CI_3 = CI_2 * (1 + interest_rate)

-- The theorem to prove
theorem compound_interest_second_year :
  relation_between_CI2_and_CI3 CI_2 year_3_interest ∧
  r = interest_rate →
  CI_2 = 1200 := 
sorry

end NUMINAMATH_GPT_compound_interest_second_year_l1782_178269


namespace NUMINAMATH_GPT_find_a_l1782_178235

theorem find_a (a : ℝ) (h_pos : a > 0)
  (h_eq : ∀ (f g : ℝ → ℝ), (f = λ x => x^2 + 10) → (g = λ x => x^2 - 6) → f (g a) = 14) :
  a = 2 * Real.sqrt 2 ∨ a = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l1782_178235


namespace NUMINAMATH_GPT_min_h_for_circle_l1782_178214

theorem min_h_for_circle (h : ℝ) :
  (∀ x y : ℝ, (x - h)^2 + (y - 1)^2 = 1 → x + y + 1 ≥ 0) →
  h = Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_GPT_min_h_for_circle_l1782_178214


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l1782_178227

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l1782_178227


namespace NUMINAMATH_GPT_sum_of_ages_l1782_178230

theorem sum_of_ages (M C : ℝ) (h1 : M = C + 12) (h2 : M + 10 = 3 * (C - 6)) : M + C = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1782_178230


namespace NUMINAMATH_GPT_rhombus_area_l1782_178218

theorem rhombus_area : 
  ∃ (d1 d2 : ℝ), (∀ (x : ℝ), x^2 - 14 * x + 48 = 0 → x = d1 ∨ x = d2) ∧
  (∀ (A : ℝ), A = d1 * d2 / 2 → A = 24) :=
by 
sorry

end NUMINAMATH_GPT_rhombus_area_l1782_178218


namespace NUMINAMATH_GPT_choir_members_l1782_178297

theorem choir_members (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 11 = 6) (h3 : 200 ≤ n ∧ n ≤ 300) :
  n = 220 :=
sorry

end NUMINAMATH_GPT_choir_members_l1782_178297


namespace NUMINAMATH_GPT_remainder_8_pow_2023_div_5_l1782_178246

-- Definition for modulo operation
def mod_five (a : Nat) : Nat := a % 5

-- Key theorem to prove
theorem remainder_8_pow_2023_div_5 : mod_five (8 ^ 2023) = 2 :=
by
  sorry -- This is where the proof would go, but it's not required per the instructions

end NUMINAMATH_GPT_remainder_8_pow_2023_div_5_l1782_178246


namespace NUMINAMATH_GPT_concert_attendance_l1782_178200

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end NUMINAMATH_GPT_concert_attendance_l1782_178200


namespace NUMINAMATH_GPT_grading_combinations_l1782_178270

/-- There are 12 students in the class. -/
def num_students : ℕ := 12

/-- There are 4 possible grades (A, B, C, and D). -/
def num_grades : ℕ := 4

/-- The total number of ways to assign grades. -/
theorem grading_combinations : (num_grades ^ num_students) = 16777216 := 
by
  sorry

end NUMINAMATH_GPT_grading_combinations_l1782_178270


namespace NUMINAMATH_GPT_monthly_income_ratio_l1782_178275

noncomputable def A_annual_income : ℝ := 571200
noncomputable def C_monthly_income : ℝ := 17000
noncomputable def B_monthly_income : ℝ := C_monthly_income * 1.12
noncomputable def A_monthly_income : ℝ := A_annual_income / 12

theorem monthly_income_ratio :
  (A_monthly_income / B_monthly_income) = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_monthly_income_ratio_l1782_178275


namespace NUMINAMATH_GPT_relationship_among_neg_a_square_neg_a_cube_l1782_178292

theorem relationship_among_neg_a_square_neg_a_cube (a : ℝ) (h : -1 < a ∧ a < 0) : (-a > a^2 ∧ a^2 > -a^3) :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_neg_a_square_neg_a_cube_l1782_178292


namespace NUMINAMATH_GPT_profit_percentage_l1782_178251

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 625) : 
  ((SP - CP) / CP) * 100 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_profit_percentage_l1782_178251


namespace NUMINAMATH_GPT_initial_percentage_acid_l1782_178290

theorem initial_percentage_acid (P : ℝ) (h1 : 27 * P / 100 = 18 * 60 / 100) : P = 40 :=
sorry

end NUMINAMATH_GPT_initial_percentage_acid_l1782_178290


namespace NUMINAMATH_GPT_product_of_solutions_l1782_178249

theorem product_of_solutions :
  ∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) →
  (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x = x1 ∨ x = x2) → x1 * x2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l1782_178249


namespace NUMINAMATH_GPT_proof_of_k_bound_l1782_178226

noncomputable def sets_with_nonempty_intersection_implies_k_bound (k : ℝ) : Prop :=
  let M := {x : ℝ | -1 ≤ x ∧ x < 2}
  let N := {x : ℝ | x ≤ k + 3}
  M ∩ N ≠ ∅ → k ≥ -4

theorem proof_of_k_bound (k : ℝ) : sets_with_nonempty_intersection_implies_k_bound k := by
  intro h
  have : -1 ≤ k + 3 := sorry
  linarith

end NUMINAMATH_GPT_proof_of_k_bound_l1782_178226


namespace NUMINAMATH_GPT_tank_capacity_75_l1782_178219

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end NUMINAMATH_GPT_tank_capacity_75_l1782_178219


namespace NUMINAMATH_GPT_common_real_root_pair_l1782_178285

theorem common_real_root_pair (n : ℕ) (hn : n > 1) :
  ∃ x : ℝ, (∃ a b : ℤ, ((x^n + (a : ℝ) * x = 2008) ∧ (x^n + (b : ℝ) * x = 2009))) ↔
    ((a = 2007 ∧ b = 2008) ∨
     (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end NUMINAMATH_GPT_common_real_root_pair_l1782_178285


namespace NUMINAMATH_GPT_x_value_unique_l1782_178264

theorem x_value_unique (x : ℝ) (h : ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) :
  x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_x_value_unique_l1782_178264


namespace NUMINAMATH_GPT_proof_inequality_l1782_178268

noncomputable def proof_problem (x : ℝ) (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) : Prop :=
  let a := Real.log x
  let b := (1 / 2) ^ (Real.log x)
  let c := Real.exp (Real.log x)
  b > c ∧ c > a

theorem proof_inequality {x : ℝ} (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) :
  proof_problem x Hx :=
sorry

end NUMINAMATH_GPT_proof_inequality_l1782_178268


namespace NUMINAMATH_GPT_triangle_circle_distance_l1782_178222

open Real

theorem triangle_circle_distance 
  (DE DF EF : ℝ)
  (hDE : DE = 12) (hDF : DF = 16) (hEF : EF = 20) :
  let s := (DE + DF + EF) / 2
  let K := sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let ra := K / (s - EF)
  let DP := s - DF
  let DQ := s
  let DI := sqrt (DP^2 + r^2)
  let DE := sqrt (DQ^2 + ra^2)
  let distance := DE - DI
  distance = 24 * sqrt 2 - 4 * sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_triangle_circle_distance_l1782_178222


namespace NUMINAMATH_GPT_cannot_bisect_segment_with_ruler_l1782_178234

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end NUMINAMATH_GPT_cannot_bisect_segment_with_ruler_l1782_178234


namespace NUMINAMATH_GPT_expr_value_l1782_178238

theorem expr_value : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end NUMINAMATH_GPT_expr_value_l1782_178238


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_l1782_178245

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y - 1)^2 = 4) →
  (m = 6 ∨ m = 12) := by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_l1782_178245


namespace NUMINAMATH_GPT_rhombus_area_l1782_178283

-- Define the lengths of the diagonals
def d1 : ℝ := 6
def d2 : ℝ := 8

-- Problem statement: The area of the rhombus
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : (1 / 2) * d1 * d2 = 24 := by
  -- The proof is not required, so we use sorry.
  sorry

end NUMINAMATH_GPT_rhombus_area_l1782_178283


namespace NUMINAMATH_GPT_distance_BC_l1782_178253

theorem distance_BC (AB AC CD DA: ℝ) (hAB: AB = 50) (hAC: AC = 40) (hCD: CD = 25) (hDA: DA = 35):
  BC = 10 ∨ BC = 90 :=
by
  sorry

end NUMINAMATH_GPT_distance_BC_l1782_178253


namespace NUMINAMATH_GPT_man_twice_son_age_l1782_178221

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 27) (h2 : M = S + 29) (h3 : M + Y = 2 * (S + Y)) : Y = 2 := 
by sorry

end NUMINAMATH_GPT_man_twice_son_age_l1782_178221


namespace NUMINAMATH_GPT_swimmer_speed_is_4_4_l1782_178217

noncomputable def swimmer_speed_in_still_water (distance : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
(distance / time) + current_speed

theorem swimmer_speed_is_4_4 :
  swimmer_speed_in_still_water 7 2.5 3.684210526315789 = 4.4 :=
by
  -- This part would contain the proof to show that the calculated speed is 4.4
  sorry

end NUMINAMATH_GPT_swimmer_speed_is_4_4_l1782_178217


namespace NUMINAMATH_GPT_height_of_triangle_l1782_178232

theorem height_of_triangle (base height area : ℝ) (h1 : base = 6) (h2 : area = 24) (h3 : area = 1 / 2 * base * height) : height = 8 :=
by sorry

end NUMINAMATH_GPT_height_of_triangle_l1782_178232


namespace NUMINAMATH_GPT_find_other_root_l1782_178207

variable {m : ℝ} -- m is a real number
variable (x : ℝ)

theorem find_other_root (h : x^2 + m * x - 5 = 0) (hx1 : x = -1) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_other_root_l1782_178207


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1782_178295

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x - 4 * a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1782_178295


namespace NUMINAMATH_GPT_number_of_tiles_per_row_l1782_178209

-- Definitions of conditions
def area : ℝ := 320
def length : ℝ := 16
def tile_size : ℝ := 1

-- Theorem statement
theorem number_of_tiles_per_row : (area / length) / tile_size = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_tiles_per_row_l1782_178209


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1782_178273

theorem sum_of_arithmetic_sequence
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (hS : ∀ n : ℕ, S n = n * a n)
    (h_condition : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
    S 19 = -38 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1782_178273


namespace NUMINAMATH_GPT_johnny_fishes_l1782_178223

theorem johnny_fishes
  (total_fishes : ℕ)
  (sony_ratio : ℕ)
  (total_is_40 : total_fishes = 40)
  (sony_is_4x_johnny : sony_ratio = 4)
  : ∃ (johnny_fishes : ℕ), johnny_fishes + sony_ratio * johnny_fishes = total_fishes ∧ johnny_fishes = 8 :=
by
  sorry

end NUMINAMATH_GPT_johnny_fishes_l1782_178223


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1782_178213

theorem quadratic_inequality_solution :
  ∀ (x : ℝ), x^2 - 9 * x + 14 ≤ 0 → 2 ≤ x ∧ x ≤ 7 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1782_178213


namespace NUMINAMATH_GPT_largest_n_base_conditions_l1782_178211

theorem largest_n_base_conditions :
  ∃ n: ℕ, n < 10000 ∧ 
  (∃ a: ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b: ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c: ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1)) ∧
  n = 4369 :=
sorry

end NUMINAMATH_GPT_largest_n_base_conditions_l1782_178211


namespace NUMINAMATH_GPT_problem_statement_l1782_178294

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1782_178294


namespace NUMINAMATH_GPT_eggs_leftover_l1782_178237

theorem eggs_leftover (eggs_abigail eggs_beatrice eggs_carson cartons : ℕ)
  (h_abigail : eggs_abigail = 37)
  (h_beatrice : eggs_beatrice = 49)
  (h_carson : eggs_carson = 14)
  (h_cartons : cartons = 12) :
  ((eggs_abigail + eggs_beatrice + eggs_carson) % cartons) = 4 :=
by
  sorry

end NUMINAMATH_GPT_eggs_leftover_l1782_178237


namespace NUMINAMATH_GPT_find_value_of_a3_a6_a9_l1782_178233

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

variables {a : ℕ → ℤ} (d : ℤ)

-- Given conditions
axiom cond1 : a 1 + a 4 + a 7 = 45
axiom cond2 : a 2 + a 5 + a 8 = 29

-- Lean 4 Statement
theorem find_value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 13 :=
sorry

end NUMINAMATH_GPT_find_value_of_a3_a6_a9_l1782_178233


namespace NUMINAMATH_GPT_largest_digit_divisible_by_9_l1782_178278

theorem largest_digit_divisible_by_9 : ∀ (B : ℕ), B < 10 → (∃ n : ℕ, 9 * n = 5 + B + 4 + 8 + 6 + 1) → B = 9 := by
  sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_9_l1782_178278


namespace NUMINAMATH_GPT_option_C_true_l1782_178224

variable {a b : ℝ}

theorem option_C_true (h : a < b) : a / 3 < b / 3 := sorry

end NUMINAMATH_GPT_option_C_true_l1782_178224


namespace NUMINAMATH_GPT_quadratic_other_root_l1782_178267

theorem quadratic_other_root (k : ℝ) (h : ∀ x, x^2 - k*x - 4 = 0 → x = 2 ∨ x = -2) :
  ∀ x, x^2 - k*x - 4 = 0 → x = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_other_root_l1782_178267


namespace NUMINAMATH_GPT_rubber_band_problem_l1782_178287

noncomputable def a : ℤ := 4
noncomputable def b : ℤ := 12
noncomputable def c : ℤ := 3
noncomputable def band_length := a * Real.pi + b * Real.sqrt c

theorem rubber_band_problem (r1 r2 d : ℝ) (h1 : r1 = 3) (h2 : r2 = 9) (h3 : d = 12) :
  let a := 4
  let b := 12
  let c := 3
  let band_length := a * Real.pi + b * Real.sqrt c
  a + b + c = 19 :=
by
  sorry

end NUMINAMATH_GPT_rubber_band_problem_l1782_178287


namespace NUMINAMATH_GPT_increasing_m_range_l1782_178261

noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x - 2 * m * x

theorem increasing_m_range (m : ℝ) : 
  (∀ x > 0, (2 * x + 1 / x - 2 * m ≥ 0)) → m ≤ Real.sqrt 2 :=
by
  intros h
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_increasing_m_range_l1782_178261


namespace NUMINAMATH_GPT_intersection_A_B_l1782_178220

def A (x : ℝ) : Prop := x > 3
def B (x : ℝ) : Prop := x ≤ 4

theorem intersection_A_B : {x | A x} ∩ {x | B x} = {x | 3 < x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1782_178220


namespace NUMINAMATH_GPT_rockets_win_30_l1782_178242

-- Given conditions
def hawks_won (h : ℕ) (w : ℕ) : Prop := h > w
def rockets_won (r : ℕ) (k : ℕ) (l : ℕ) : Prop := r > k ∧ r < l
def knicks_at_least (k : ℕ) : Prop := k ≥ 15
def clippers_won (c : ℕ) (l : ℕ) : Prop := c < l

-- Possible number of games won
def possible_games : List ℕ := [15, 20, 25, 30, 35, 40]

-- Prove Rockets won 30 games
theorem rockets_win_30 (h w r k l c : ℕ) 
  (h_w: hawks_won h w)
  (r_kl : rockets_won r k l)
  (k_15: knicks_at_least k)
  (c_l : clippers_won c l)
  (h_mem : h ∈ possible_games)
  (w_mem : w ∈ possible_games)
  (r_mem : r ∈ possible_games)
  (k_mem : k ∈ possible_games)
  (l_mem : l ∈ possible_games)
  (c_mem : c ∈ possible_games) :
  r = 30 :=
sorry

end NUMINAMATH_GPT_rockets_win_30_l1782_178242


namespace NUMINAMATH_GPT_remainder_of_x50_div_x_plus_1_cubed_l1782_178266

theorem remainder_of_x50_div_x_plus_1_cubed (x : ℚ) : 
  (x ^ 50) % ((x + 1) ^ 3) = 1225 * x ^ 2 + 2450 * x + 1176 :=
by sorry

end NUMINAMATH_GPT_remainder_of_x50_div_x_plus_1_cubed_l1782_178266


namespace NUMINAMATH_GPT_initial_people_in_elevator_l1782_178250

theorem initial_people_in_elevator (W n : ℕ) (avg_initial_weight avg_new_weight new_person_weight : ℚ)
  (h1 : avg_initial_weight = 152)
  (h2 : avg_new_weight = 151)
  (h3 : new_person_weight = 145)
  (h4 : W = n * avg_initial_weight)
  (h5 : W + new_person_weight = (n + 1) * avg_new_weight) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_people_in_elevator_l1782_178250


namespace NUMINAMATH_GPT_masha_can_climb_10_steps_l1782_178228

def ways_to_climb_stairs : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => ways_to_climb_stairs (n + 1) + ways_to_climb_stairs n

theorem masha_can_climb_10_steps : ways_to_climb_stairs 10 = 89 :=
by
  -- proof omitted here as per instruction
  sorry

end NUMINAMATH_GPT_masha_can_climb_10_steps_l1782_178228


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l1782_178229

noncomputable def A := 29 / 15
noncomputable def B := 13 / 12
noncomputable def C := 37 / 15

theorem partial_fraction_decomposition :
  let ABC := A * B * C;
  ABC = 13949 / 2700 :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l1782_178229


namespace NUMINAMATH_GPT_S_6_equals_12_l1782_178202

noncomputable def S (n : ℕ) : ℝ := sorry -- Definition for the sum of the first n terms

axiom geometric_sequence_with_positive_terms (n : ℕ) : S n > 0

axiom S_3 : S 3 = 3

axiom S_9 : S 9 = 39

theorem S_6_equals_12 : S 6 = 12 := by
  sorry

end NUMINAMATH_GPT_S_6_equals_12_l1782_178202


namespace NUMINAMATH_GPT_pythagorean_theorem_mod_3_l1782_178291

theorem pythagorean_theorem_mod_3 {x y z : ℕ} (h : x^2 + y^2 = z^2) : x % 3 = 0 ∨ y % 3 = 0 ∨ z % 3 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_pythagorean_theorem_mod_3_l1782_178291


namespace NUMINAMATH_GPT_pictures_hung_in_new_galleries_l1782_178272

noncomputable def total_pencils_used : ℕ := 218
noncomputable def pencils_per_picture : ℕ := 5
noncomputable def pencils_per_exhibition : ℕ := 3

noncomputable def pictures_initial : ℕ := 9
noncomputable def galleries_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]
noncomputable def total_exhibitions : ℕ := 1 + galleries_requests.length

theorem pictures_hung_in_new_galleries :
  let total_pencils_for_signing := total_exhibitions * pencils_per_exhibition
  let total_pencils_for_drawing := total_pencils_used - total_pencils_for_signing
  let total_pictures_drawn := total_pencils_for_drawing / pencils_per_picture
  let pictures_in_new_galleries := total_pictures_drawn - pictures_initial
  pictures_in_new_galleries = 29 :=
by
  sorry

end NUMINAMATH_GPT_pictures_hung_in_new_galleries_l1782_178272


namespace NUMINAMATH_GPT_original_four_digit_number_l1782_178208

theorem original_four_digit_number : 
  ∃ x y z: ℕ, (x = 1 ∧ y = 9 ∧ z = 7 ∧ 1000 * x + 100 * y + 10 * z + y = 1979) ∧ 
  (1000 * y + 100 * z + 10 * y + x - (1000 * x + 100 * y + 10 * z + y) = 7812) ∧ 
  (1000 * y + 100 * z + 10 * y + x < 10000 ∧ 1000 * x + 100 * y + 10 * z + y < 10000) := 
sorry

end NUMINAMATH_GPT_original_four_digit_number_l1782_178208


namespace NUMINAMATH_GPT_find_a100_l1782_178225

noncomputable def S (k : ℝ) (n : ℤ) : ℝ := k * (n ^ 2) + n
noncomputable def a (k : ℝ) (n : ℤ) : ℝ := S k n - S k (n - 1)

theorem find_a100 (k : ℝ) 
  (h1 : a k 10 = 39) :
  a k 100 = 399 :=
sorry

end NUMINAMATH_GPT_find_a100_l1782_178225


namespace NUMINAMATH_GPT_alpha_half_in_II_IV_l1782_178240

theorem alpha_half_in_II_IV (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) : 
  (k * π - π / 4 < (α / 2) ∧ (α / 2) < k * π) :=
by
  sorry

end NUMINAMATH_GPT_alpha_half_in_II_IV_l1782_178240


namespace NUMINAMATH_GPT_sqrt_three_squared_l1782_178274

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_three_squared_l1782_178274


namespace NUMINAMATH_GPT_outer_boundary_diameter_l1782_178281

theorem outer_boundary_diameter (statue_width garden_width path_width fountain_diameter : ℝ) 
  (h_statue : statue_width = 2) 
  (h_garden : garden_width = 10) 
  (h_path : path_width = 8) 
  (h_fountain : fountain_diameter = 12) : 
  2 * ((fountain_diameter / 2 + statue_width) + garden_width + path_width) = 52 :=
by
  sorry

end NUMINAMATH_GPT_outer_boundary_diameter_l1782_178281


namespace NUMINAMATH_GPT_rebecca_eggs_l1782_178205

theorem rebecca_eggs (groups : ℕ) (eggs_per_group : ℕ) (total_eggs : ℕ) 
  (h1 : groups = 3) (h2 : eggs_per_group = 3) : total_eggs = 9 :=
by
  sorry

end NUMINAMATH_GPT_rebecca_eggs_l1782_178205


namespace NUMINAMATH_GPT_last_digit_B_l1782_178263

theorem last_digit_B 
  (B : ℕ) 
  (h : ∀ n : ℕ, n % 10 = (B - 287)^2 % 10 → n % 10 = 4) :
  (B = 5 ∨ B = 9) :=
sorry

end NUMINAMATH_GPT_last_digit_B_l1782_178263
