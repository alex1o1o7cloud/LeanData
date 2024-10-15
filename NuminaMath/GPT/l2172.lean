import Mathlib

namespace NUMINAMATH_GPT_total_weight_of_pumpkins_l2172_217241

def first_pumpkin_weight : ℝ := 12.6
def second_pumpkin_weight : ℝ := 23.4
def total_weight : ℝ := 36

theorem total_weight_of_pumpkins :
  first_pumpkin_weight + second_pumpkin_weight = total_weight :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_pumpkins_l2172_217241


namespace NUMINAMATH_GPT_jaewoong_ran_the_most_l2172_217219

def distance_jaewoong : ℕ := 20000 -- Jaewoong's distance in meters
def distance_seongmin : ℕ := 2600  -- Seongmin's distance in meters
def distance_eunseong : ℕ := 5000  -- Eunseong's distance in meters

theorem jaewoong_ran_the_most : distance_jaewoong > distance_seongmin ∧ distance_jaewoong > distance_eunseong := by
  sorry

end NUMINAMATH_GPT_jaewoong_ran_the_most_l2172_217219


namespace NUMINAMATH_GPT_two_trains_distance_before_meeting_l2172_217255

noncomputable def distance_one_hour_before_meeting (speed_A speed_B : ℕ) : ℕ :=
  speed_A + speed_B

theorem two_trains_distance_before_meeting (speed_A speed_B total_distance : ℕ) (h_speed_A : speed_A = 60) (h_speed_B : speed_B = 40) (h_total_distance : total_distance ≤ 250) :
  distance_one_hour_before_meeting speed_A speed_B = 100 :=
by
  sorry

end NUMINAMATH_GPT_two_trains_distance_before_meeting_l2172_217255


namespace NUMINAMATH_GPT_original_cost_of_car_l2172_217275

theorem original_cost_of_car (C : ℝ) 
  (repair_cost : ℝ := 15000)
  (selling_price : ℝ := 64900)
  (profit_percent : ℝ := 13.859649122807017) :
  C = 43837.21 :=
by
  have h1 : C + repair_cost = selling_price - (selling_price - (C + repair_cost)) := by sorry
  have h2 : profit_percent / 100 = (selling_price - (C + repair_cost)) / C := by sorry
  have h3 : C = 43837.21 := by sorry
  exact h3

end NUMINAMATH_GPT_original_cost_of_car_l2172_217275


namespace NUMINAMATH_GPT_weight_of_smallest_box_l2172_217285

variables (M S L : ℕ)

theorem weight_of_smallest_box
  (h1 : M + S = 83)
  (h2 : L + S = 85)
  (h3 : L + M = 86) :
  S = 41 :=
sorry

end NUMINAMATH_GPT_weight_of_smallest_box_l2172_217285


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l2172_217248

theorem geometric_progression_common_ratio (x y z w r : ℂ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h_geom : x * (y - w) = a ∧ y * (z - x) = a * r ∧ z * (w - y) = a * r^2 ∧ w * (x - z) = a * r^3) :
  1 + r + r^2 + r^3 = 0 :=
sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l2172_217248


namespace NUMINAMATH_GPT_maximum_xy_l2172_217280

theorem maximum_xy (x y : ℕ) (h1 : 7 * x + 2 * y = 110) : ∃ x y, (7 * x + 2 * y = 110) ∧ (x > 0) ∧ (y > 0) ∧ (x * y = 216) :=
by
  sorry

end NUMINAMATH_GPT_maximum_xy_l2172_217280


namespace NUMINAMATH_GPT_count_valid_c_l2172_217295

theorem count_valid_c : ∃ (count : ℕ), count = 670 ∧ 
  ∀ (c : ℤ), (-2007 ≤ c ∧ c ≤ 2007) → 
    (∃ (x : ℤ), (x^2 + c) % (2^2007) = 0) ↔ count = 670 :=
sorry

end NUMINAMATH_GPT_count_valid_c_l2172_217295


namespace NUMINAMATH_GPT_proof_mn_eq_9_l2172_217299

theorem proof_mn_eq_9 (m n : ℕ) (h1 : 2 * m + n = 8) (h2 : m - n = 1) : m^n = 9 :=
by {
  sorry 
}

end NUMINAMATH_GPT_proof_mn_eq_9_l2172_217299


namespace NUMINAMATH_GPT_max_det_A_l2172_217202

open Real

-- Define the matrix and the determinant expression
noncomputable def A (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, 1, 1],
    ![1, 1 + cos θ, 1],
    ![1 + sin θ, 1, 1]
  ]

-- Lean statement to prove the maximum value of the determinant of matrix A
theorem max_det_A : ∃ θ : ℝ, (Matrix.det (A θ)) ≤ 1/2 := by
  sorry

end NUMINAMATH_GPT_max_det_A_l2172_217202


namespace NUMINAMATH_GPT_smallest_d_for_divisibility_by_9_l2172_217214

theorem smallest_d_for_divisibility_by_9 : ∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (437003 + d * 100) % 9 = 0 ∧ ∀ d', 0 ≤ d' ∧ d' < d → ((437003 + d' * 100) % 9 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_for_divisibility_by_9_l2172_217214


namespace NUMINAMATH_GPT_fraction_of_jumbo_tiles_l2172_217242

-- Definitions for conditions
variables (L W : ℝ) -- Length and width of regular tiles
variables (n : ℕ) -- Number of regular tiles
variables (m : ℕ) -- Number of jumbo tiles

-- Conditions
def condition1 : Prop := (n : ℝ) * (L * W) = 40 -- Regular tiles cover 40 square feet
def condition2 : Prop := (n : ℝ) * (L * W) + (m : ℝ) * (3 * L * W) = 220 -- Entire wall is 220 square feet
def condition3 : Prop := ∃ (k : ℝ), (m : ℝ) = k * (n : ℝ) ∧ k = 1.5 -- Relationship ratio between jumbo and regular tiles

-- Theorem to be proved
theorem fraction_of_jumbo_tiles (L W : ℝ) (n m : ℕ)
  (h1 : condition1 L W n)
  (h2 : condition2 L W n m)
  (h3 : condition3 n m) :
  (m : ℝ) / ((n : ℝ) + (m : ℝ)) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_of_jumbo_tiles_l2172_217242


namespace NUMINAMATH_GPT_convex_polygon_interior_angle_l2172_217232

theorem convex_polygon_interior_angle (n : ℕ) (h1 : 3 ≤ n)
  (h2 : (n - 2) * 180 = 2570 + x) : x = 130 :=
sorry

end NUMINAMATH_GPT_convex_polygon_interior_angle_l2172_217232


namespace NUMINAMATH_GPT_circumference_to_diameter_ratio_l2172_217270

-- Definitions from the conditions
def r : ℝ := 15
def C : ℝ := 90
def D : ℝ := 2 * r

-- The proof goal
theorem circumference_to_diameter_ratio : C / D = 3 := 
by sorry

end NUMINAMATH_GPT_circumference_to_diameter_ratio_l2172_217270


namespace NUMINAMATH_GPT_oscar_leap_vs_elmer_stride_l2172_217289

/--
Given:
1. The 51st telephone pole is exactly 6600 feet from the first pole.
2. Elmer the emu takes 50 equal strides to walk between consecutive telephone poles.
3. Oscar the ostrich can cover the same distance in 15 equal leaps.
4. There are 50 gaps between the 51 poles.

Prove:
Oscar's leap is 6 feet longer than Elmer's stride.
-/
theorem oscar_leap_vs_elmer_stride : 
  let total_distance := 6600 
  let elmer_strides_per_gap := 50
  let oscar_leaps_per_gap := 15
  let num_gaps := 50
  let elmer_total_strides := elmer_strides_per_gap * num_gaps
  let oscar_total_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 6 := 
by {
  -- The proof would go here.
  sorry
}

end NUMINAMATH_GPT_oscar_leap_vs_elmer_stride_l2172_217289


namespace NUMINAMATH_GPT_no_intersection_points_l2172_217276

theorem no_intersection_points : ¬ ∃ x y : ℝ, y = x ∧ y = x - 2 := by
  sorry

end NUMINAMATH_GPT_no_intersection_points_l2172_217276


namespace NUMINAMATH_GPT_question_1_question_2_question_3_l2172_217274

def deck_size : Nat := 32

theorem question_1 :
  let hands_when_order_matters := deck_size * (deck_size - 1)
  hands_when_order_matters = 992 :=
by
  let hands_when_order_matters := deck_size * (deck_size - 1)
  sorry

theorem question_2 :
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  hands_when_order_does_not_matter = 496 :=
by
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  sorry

theorem question_3 :
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  hands_3_cards_order_does_not_matter = 4960 :=
by
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  sorry

end NUMINAMATH_GPT_question_1_question_2_question_3_l2172_217274


namespace NUMINAMATH_GPT_row_column_crossout_l2172_217210

theorem row_column_crossout (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ rows : Finset (Fin 1000), rows.card = 990 ∧ ∀ j : Fin 1000, ∃ i ∈ rowsᶜ, M i j = 1) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 990 ∧ ∀ i : Fin 1000, ∃ j ∈ colsᶜ, M i j = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_row_column_crossout_l2172_217210


namespace NUMINAMATH_GPT_find_number_of_shorts_l2172_217208

def price_of_shorts : ℕ := 7
def price_of_shoes : ℕ := 20
def total_spent : ℕ := 75

-- We represent the price of 4 tops as a variable
variable (T : ℕ)

theorem find_number_of_shorts (S : ℕ) (h : 7 * S + 4 * T + 20 = 75) : S = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_shorts_l2172_217208


namespace NUMINAMATH_GPT_sum_le_30_l2172_217296

variable (a b x y : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx_pos : 0 < x) (hy_pos : 0 < y)
variable (h1 : a * x ≤ 5) (h2 : a * y ≤ 10) (h3 : b * x ≤ 10) (h4 : b * y ≤ 10)

theorem sum_le_30 : a * x + a * y + b * x + b * y ≤ 30 := sorry

end NUMINAMATH_GPT_sum_le_30_l2172_217296


namespace NUMINAMATH_GPT_max_lines_between_points_l2172_217298

noncomputable def maxLines (points : Nat) := 
  let deg := [1, 2, 3, 4, 5]
  (1 * (points - 1) + 2 * (points - 2) + 3 * (points - 3) + 4 * (points - 4) + 5 * (points - 5)) / 2

theorem max_lines_between_points :
  ∀ (n : Nat), n = 15 → maxLines n = 85 :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_max_lines_between_points_l2172_217298


namespace NUMINAMATH_GPT_cos_7theta_l2172_217277

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -45682/8192 :=
by
  sorry

end NUMINAMATH_GPT_cos_7theta_l2172_217277


namespace NUMINAMATH_GPT_bankers_discount_l2172_217254

/-- The banker’s gain on a sum due 3 years hence at 12% per annum is Rs. 360.
   The banker's discount is to be determined. -/
theorem bankers_discount (BG BD TD : ℝ) (R : ℝ := 12 / 100) (T : ℝ := 3) 
  (h1 : BG = 360) (h2 : BG = (BD * TD) / (BD - TD)) (h3 : TD = (P * R * T) / 100) 
  (h4 : BG = (TD * R * T) / 100) :
  BD = 562.5 :=
sorry

end NUMINAMATH_GPT_bankers_discount_l2172_217254


namespace NUMINAMATH_GPT_binomial_12_3_eq_220_l2172_217259

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end NUMINAMATH_GPT_binomial_12_3_eq_220_l2172_217259


namespace NUMINAMATH_GPT_susan_min_packages_l2172_217288

theorem susan_min_packages (n : ℕ) (cost_per_package : ℕ := 5) (earnings_per_package : ℕ := 15) (initial_cost : ℕ := 1200) :
  15 * n - 5 * n ≥ 1200 → n ≥ 120 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_susan_min_packages_l2172_217288


namespace NUMINAMATH_GPT_ensure_two_of_each_kind_l2172_217233

def tablets_A := 10
def tablets_B := 14
def least_number_of_tablets_to_ensure_two_of_each := 12

theorem ensure_two_of_each_kind 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extracted : ℕ) 
  (hA : total_A = tablets_A) 
  (hB : total_B = tablets_B)
  (hExtract : extracted = least_number_of_tablets_to_ensure_two_of_each) : 
  ∃ (extracted : ℕ), extracted = least_number_of_tablets_to_ensure_two_of_each ∧ extracted ≥ tablets_A + 2 := 
sorry

end NUMINAMATH_GPT_ensure_two_of_each_kind_l2172_217233


namespace NUMINAMATH_GPT_probability_of_at_least_one_vowel_is_799_over_1024_l2172_217223

def Set1 : Set Char := {'a', 'e', 'i', 'b', 'c', 'd', 'f', 'g'}
def Set2 : Set Char := {'u', 'o', 'y', 'k', 'l', 'm', 'n', 'p'}
def Set3 : Set Char := {'e', 'u', 'v', 'r', 's', 't', 'w', 'x'}
def Set4 : Set Char := {'a', 'i', 'o', 'z', 'h', 'j', 'q', 'r'}

noncomputable def probability_of_at_least_one_vowel : ℚ :=
  1 - (5/8 : ℚ) * (3/4 : ℚ) * (3/4 : ℚ) * (5/8 : ℚ)

theorem probability_of_at_least_one_vowel_is_799_over_1024 :
  probability_of_at_least_one_vowel = 799 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_vowel_is_799_over_1024_l2172_217223


namespace NUMINAMATH_GPT_investment_ratio_l2172_217286

-- Define the investments
def A_investment (x : ℝ) : ℝ := 3 * x
def B_investment (x : ℝ) : ℝ := x
def C_investment (y : ℝ) : ℝ := y

-- Define the total profit and B's share of the profit
def total_profit : ℝ := 4400
def B_share : ℝ := 800

-- Define the ratio condition B's share based on investments
def B_share_cond (x y : ℝ) : Prop := (B_investment x / (A_investment x + B_investment x + C_investment y)) * total_profit = B_share

-- Define what we need to prove
theorem investment_ratio (x y : ℝ) (h : B_share_cond x y) : x / y = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_investment_ratio_l2172_217286


namespace NUMINAMATH_GPT_candy_distribution_l2172_217201

theorem candy_distribution (candies : ℕ) (family_members : ℕ) (required_candies : ℤ) :
  (candies = 45) ∧ (family_members = 5) →
  required_candies = 0 :=
by sorry

end NUMINAMATH_GPT_candy_distribution_l2172_217201


namespace NUMINAMATH_GPT_compute_binom_value_l2172_217256

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else x * binom (x - 1) (k - 1) / k

theorem compute_binom_value : 
  (binom (1/2) 2014 * 4^2014 / binom 4028 2014) = -1/4027 :=
by 
  sorry

end NUMINAMATH_GPT_compute_binom_value_l2172_217256


namespace NUMINAMATH_GPT_increasing_function_l2172_217257

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) (h : a ≥ 1) : 
  ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
by 
  sorry

end NUMINAMATH_GPT_increasing_function_l2172_217257


namespace NUMINAMATH_GPT_problem1_problem2_l2172_217266

-- Problem 1
theorem problem1 : ((- (1/2) - (1/3) + (3/4)) * -60) = 5 :=
by
  -- The proof steps would go here
  sorry

-- Problem 2
theorem problem2 : ((-1)^4 - (1/6) * (3 - (-3)^2)) = 2 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2172_217266


namespace NUMINAMATH_GPT_remainder_when_a6_divided_by_n_l2172_217235

theorem remainder_when_a6_divided_by_n (n : ℕ) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := 
sorry

end NUMINAMATH_GPT_remainder_when_a6_divided_by_n_l2172_217235


namespace NUMINAMATH_GPT_number_of_points_in_star_polygon_l2172_217293

theorem number_of_points_in_star_polygon :
  ∀ (n : ℕ) (D C : ℕ),
    (∀ i : ℕ, i < n → C = D - 15) →
    n * (D - (D - 15)) = 360 → n = 24 :=
by
  intros n D C h1 h2
  sorry

end NUMINAMATH_GPT_number_of_points_in_star_polygon_l2172_217293


namespace NUMINAMATH_GPT_wilson_theorem_application_l2172_217215

theorem wilson_theorem_application (h_prime : Nat.Prime 101) : 
  Nat.factorial 100 % 101 = 100 :=
by
  -- By Wilson's theorem, (p - 1)! ≡ -1 (mod p) for a prime p.
  -- Here p = 101, so (101 - 1)! ≡ -1 (mod 101).
  -- Therefore, 100! ≡ -1 (mod 101).
  -- Knowing that -1 ≡ 100 (mod 101), we can conclude that
  -- 100! ≡ 100 (mod 101).
  sorry

end NUMINAMATH_GPT_wilson_theorem_application_l2172_217215


namespace NUMINAMATH_GPT_find_x_l2172_217271

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * f x - 19 = f (x - 4)) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l2172_217271


namespace NUMINAMATH_GPT_range_of_abscissa_of_P_l2172_217225

noncomputable def point_lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 + 1 = 0

noncomputable def point_lies_on_circle_c (M N : ℝ × ℝ) : Prop :=
  (M.1 - 2)^2 + (M.2 - 1)^2 = 1 ∧ (N.1 - 2)^2 + (N.2 - 1)^2 = 1

noncomputable def angle_mpn_eq_60 (P M N : ℝ × ℝ) : Prop :=
  true -- This is a placeholder because we have to define the geometrical angle condition which is complex.

theorem range_of_abscissa_of_P :
  ∀ (P M N : ℝ × ℝ),
  point_lies_on_line P →
  point_lies_on_circle_c M N →
  angle_mpn_eq_60 P M N →
  0 ≤ P.1 ∧ P.1 ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_abscissa_of_P_l2172_217225


namespace NUMINAMATH_GPT_intersection_A_B_l2172_217247

def A : Set Int := {-1, 0, 1, 5, 8}
def B : Set Int := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2172_217247


namespace NUMINAMATH_GPT_circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l2172_217204

-- Define radii of the circles
def r1 : ℝ := 3
def r2 : ℝ := 5

-- Statement for first scenario (distance = 9)
theorem circles_do_not_intersect_first_scenario (d : ℝ) (h : d = 9) : ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

-- Statement for second scenario (distance = 1)
theorem circles_do_not_intersect_second_scenario (d : ℝ) (h : d = 1) : d < |r1 - r2| ∨ ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

end NUMINAMATH_GPT_circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l2172_217204


namespace NUMINAMATH_GPT_mountain_height_is_1700m_l2172_217258

noncomputable def height_of_mountain (temp_base : ℝ) (temp_summit : ℝ) (rate_decrease : ℝ) : ℝ :=
  ((temp_base - temp_summit) / rate_decrease) * 100

theorem mountain_height_is_1700m :
  height_of_mountain 26 14.1 0.7 = 1700 :=
by
  sorry

end NUMINAMATH_GPT_mountain_height_is_1700m_l2172_217258


namespace NUMINAMATH_GPT_slope_of_line_l2172_217291

theorem slope_of_line {x1 x2 y1 y2 : ℝ} 
  (h1 : (1 / x1 + 2 / y1 = 0)) 
  (h2 : (1 / x2 + 2 / y2 = 0)) 
  (h_neq : x1 ≠ x2) : 
  (y2 - y1) / (x2 - x1) = -2 := 
sorry

end NUMINAMATH_GPT_slope_of_line_l2172_217291


namespace NUMINAMATH_GPT_value_of_expression_l2172_217250

theorem value_of_expression (x Q : ℝ) (π : Real) (h : 5 * (3 * x - 4 * π) = Q) : 10 * (6 * x - 8 * π) = 4 * Q :=
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l2172_217250


namespace NUMINAMATH_GPT_Karls_Total_Travel_Distance_l2172_217222

theorem Karls_Total_Travel_Distance :
  let consumption_rate := 35
  let full_tank_gallons := 14
  let initial_miles := 350
  let added_gallons := 8
  let remaining_gallons := 7
  let net_gallons_consumed := (full_tank_gallons + added_gallons - remaining_gallons)
  let total_distance := net_gallons_consumed * consumption_rate
  total_distance = 525 := 
by 
  sorry

end NUMINAMATH_GPT_Karls_Total_Travel_Distance_l2172_217222


namespace NUMINAMATH_GPT_mushrooms_collected_l2172_217211

variable (P V : ℕ)

theorem mushrooms_collected (h1 : P = (V * 100) / (P + V)) (h2 : V % 2 = 1) :
  P + V = 25 ∨ P + V = 300 ∨ P + V = 525 ∨ P + V = 1900 ∨ P + V = 9900 := by
  sorry

end NUMINAMATH_GPT_mushrooms_collected_l2172_217211


namespace NUMINAMATH_GPT_distance_of_point_P_to_origin_l2172_217265

noncomputable def dist_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

theorem distance_of_point_P_to_origin :
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let y_P := 1 / 2
  ∃ x_P : ℝ, (x_P, y_P) = P ∧
    (dist_to_origin P = Real.sqrt 6 / 2) :=
by
  sorry

end NUMINAMATH_GPT_distance_of_point_P_to_origin_l2172_217265


namespace NUMINAMATH_GPT_find_nat_nums_satisfying_eq_l2172_217206

theorem find_nat_nums_satisfying_eq (m n : ℕ) (h_m : m = 3) (h_n : n = 3) : 2 ^ n + 1 = m ^ 2 :=
by
  rw [h_m, h_n]
  sorry

end NUMINAMATH_GPT_find_nat_nums_satisfying_eq_l2172_217206


namespace NUMINAMATH_GPT_assignment_statement_correct_l2172_217228

-- Definitions for the conditions:
def cond_A : Prop := ∀ M : ℕ, (M = M + 3)
def cond_B : Prop := ∀ M : ℕ, (M = M + (3 - M))
def cond_C : Prop := ∀ M : ℕ, (M = M + 3)
def cond_D : Prop := true ∧ cond_A ∧ cond_B ∧ cond_C

-- Theorem statement proving the correct interpretation of the assignment is condition B
theorem assignment_statement_correct : cond_B :=
by
  sorry

end NUMINAMATH_GPT_assignment_statement_correct_l2172_217228


namespace NUMINAMATH_GPT_compute_expression_l2172_217226

noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

theorem compute_expression : 5^(c / d) + 2^(d / c) = 2 * Real.sqrt 2 + 5^(2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2172_217226


namespace NUMINAMATH_GPT_pipe_q_fills_in_9_hours_l2172_217220

theorem pipe_q_fills_in_9_hours (x : ℝ) :
  (1 / 3 + 1 / x + 1 / 18 = 1 / 2) → x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_pipe_q_fills_in_9_hours_l2172_217220


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_plane_l2172_217292

variables {α β γ : Plane} {m n : Line}

-- Define the properties of perpendicular lines to planes and parallel lines
def perpendicular_to (l : Line) (p : Plane) : Prop := 
sorry -- definition skipped

def parallel_to (l1 l2 : Line) : Prop := 
sorry -- definition skipped

-- Theorem Statement (equivalent translation of the given question and its correct answer)
theorem lines_parallel_if_perpendicular_to_plane 
  (h1 : perpendicular_to m α) 
  (h2 : perpendicular_to n α) : parallel_to m n :=
sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_plane_l2172_217292


namespace NUMINAMATH_GPT_repeating_decimals_subtraction_l2172_217287

/--
Calculate the value of 0.\overline{234} - 0.\overline{567} - 0.\overline{891}.
Express your answer as a fraction in its simplest form.

Shown that:
Let x = 0.\overline{234}, y = 0.\overline{567}, z = 0.\overline{891},
Then 0.\overline{234} - 0.\overline{567} - 0.\overline{891} = -1224/999
-/
theorem repeating_decimals_subtraction : 
  let x : ℚ := 234 / 999
  let y : ℚ := 567 / 999
  let z : ℚ := 891 / 999
  x - y - z = -1224 / 999 := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_subtraction_l2172_217287


namespace NUMINAMATH_GPT_factorization_of_polynomial_l2172_217251

theorem factorization_of_polynomial (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := 
sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l2172_217251


namespace NUMINAMATH_GPT_quadratic_decreasing_then_increasing_l2172_217264

-- Define the given quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 6 * x + 10

-- Define the interval of interest
def interval (x : ℝ) : Prop := 2 < x ∧ x < 4

-- The main theorem to prove: the function is first decreasing on (2, 3] and then increasing on [3, 4)
theorem quadratic_decreasing_then_increasing :
  (∀ (x : ℝ), 2 < x ∧ x ≤ 3 → quadratic_function x > quadratic_function (x + ε) ∧ ε > 0) ∧
  (∀ (x : ℝ), 3 ≤ x ∧ x < 4 → quadratic_function x < quadratic_function (x + ε) ∧ ε > 0) :=
sorry

end NUMINAMATH_GPT_quadratic_decreasing_then_increasing_l2172_217264


namespace NUMINAMATH_GPT_solve_equation_l2172_217243

/-- 
  Given the equation:
    ∀ x, (x = 2 ∨ (3 < x ∧ x < 4)) ↔ (⌊(1/x) * ⌊x⌋^2⌋ = 2),
  where ⌊u⌋ represents the greatest integer less than or equal to u.
-/
theorem solve_equation (x : ℝ) : (x = 2 ∨ (3 < x ∧ x < 4)) ↔ ⌊(1/x) * ⌊x⌋^2⌋ = 2 := 
sorry

end NUMINAMATH_GPT_solve_equation_l2172_217243


namespace NUMINAMATH_GPT_sum_of_net_gains_is_correct_l2172_217238

namespace DepartmentRevenue

def revenueIncreaseA : ℝ := 0.1326
def revenueIncreaseB : ℝ := 0.0943
def revenueIncreaseC : ℝ := 0.7731
def taxRate : ℝ := 0.235
def initialRevenue : ℝ := 4.7 -- in millions

def netGain (revenueIncrease : ℝ) (taxRate : ℝ) (initialRevenue : ℝ) : ℝ :=
  (initialRevenue * (1 + revenueIncrease)) * (1 - taxRate)

def netGainA : ℝ := netGain revenueIncreaseA taxRate initialRevenue
def netGainB : ℝ := netGain revenueIncreaseB taxRate initialRevenue
def netGainC : ℝ := netGain revenueIncreaseC taxRate initialRevenue

def netGainSum : ℝ := netGainA + netGainB + netGainC

theorem sum_of_net_gains_is_correct :
  netGainSum = 14.38214 := by
    sorry

end DepartmentRevenue

end NUMINAMATH_GPT_sum_of_net_gains_is_correct_l2172_217238


namespace NUMINAMATH_GPT_bob_repayment_days_l2172_217273

theorem bob_repayment_days :
  ∃ (x : ℕ), (15 + 3 * x ≥ 45) ∧ (∀ y : ℕ, (15 + 3 * y ≥ 45) → x ≤ y) ∧ x = 10 := 
by
  sorry

end NUMINAMATH_GPT_bob_repayment_days_l2172_217273


namespace NUMINAMATH_GPT_find_initial_population_l2172_217227

noncomputable def population_first_year (P : ℝ) : ℝ :=
  let P1 := 0.90 * P    -- population after 1st year
  let P2 := 0.99 * P    -- population after 2nd year
  let P3 := 0.891 * P   -- population after 3rd year
  P3

theorem find_initial_population (h : population_first_year P = 4455) : P = 4455 / 0.891 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_population_l2172_217227


namespace NUMINAMATH_GPT_alex_and_zhu_probability_l2172_217239

theorem alex_and_zhu_probability :
  let num_students := 100
  let num_selected := 60
  let num_sections := 3
  let section_size := 20
  let P_alex_selected := 3 / 5
  let P_zhu_selected_given_alex_selected := 59 / 99
  let P_same_section_given_both_selected := 19 / 59
  (P_alex_selected * P_zhu_selected_given_alex_selected * P_same_section_given_both_selected) = 19 / 165 := 
by {
  sorry
}

end NUMINAMATH_GPT_alex_and_zhu_probability_l2172_217239


namespace NUMINAMATH_GPT_largest_prime_m_satisfying_quadratic_inequality_l2172_217272

theorem largest_prime_m_satisfying_quadratic_inequality :
  ∃ (m : ℕ), m = 5 ∧ m^2 - 11 * m + 28 < 0 ∧ Prime m :=
by sorry

end NUMINAMATH_GPT_largest_prime_m_satisfying_quadratic_inequality_l2172_217272


namespace NUMINAMATH_GPT_impossible_gather_all_coins_in_one_sector_l2172_217246

-- Definition of the initial condition with sectors and coins
def initial_coins_in_sectors := [1, 1, 1, 1, 1, 1] -- Each sector has one coin, represented by a list

-- Function to check if all coins are in one sector
def all_coins_in_one_sector (coins : List ℕ) := coins.count 6 == 1

-- Function to make a move (this is a helper; its implementation isn't necessary here but illustrates the idea)
def make_move (coins : List ℕ) (src dst : ℕ) : List ℕ := sorry

-- Proving that after 20 moves, coins cannot be gathered in one sector due to parity constraints
theorem impossible_gather_all_coins_in_one_sector : 
  ¬ ∃ (moves : List (ℕ × ℕ)), moves.length = 20 ∧ all_coins_in_one_sector (List.foldl (λ coins move => make_move coins move.1 move.2) initial_coins_in_sectors moves) :=
sorry

end NUMINAMATH_GPT_impossible_gather_all_coins_in_one_sector_l2172_217246


namespace NUMINAMATH_GPT_find_y_l2172_217284

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y (y : ℝ) (h : star 7 y = 47) : y = 5.4 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l2172_217284


namespace NUMINAMATH_GPT_cost_per_pancake_correct_l2172_217236

-- Define the daily rent expense
def daily_rent := 30

-- Define the daily supplies expense
def daily_supplies := 12

-- Define the number of pancakes needed to cover expenses
def number_of_pancakes := 21

-- Define the total daily expenses
def total_daily_expenses := daily_rent + daily_supplies

-- Define the cost per pancake calculation
def cost_per_pancake := total_daily_expenses / number_of_pancakes

-- The theorem to prove the cost per pancake
theorem cost_per_pancake_correct :
  cost_per_pancake = 2 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_pancake_correct_l2172_217236


namespace NUMINAMATH_GPT_sum_two_numbers_l2172_217213

theorem sum_two_numbers :
  let X := (2 * 10) + 6
  let Y := (4 * 10) + 1
  X + Y = 67 :=
by
  sorry

end NUMINAMATH_GPT_sum_two_numbers_l2172_217213


namespace NUMINAMATH_GPT_incorrect_option_C_l2172_217278

def line (α : Type*) := α → Prop
def plane (α : Type*) := α → Prop

variables {α : Type*} (m n : line α) (a b : plane α)

def parallel (m n : line α) : Prop := ∀ x, m x → n x
def perpendicular (m n : line α) : Prop := ∃ x, m x ∧ n x

def lies_in (m : line α) (a : plane α) : Prop := ∀ x, m x → a x

theorem incorrect_option_C (h : lies_in m a) : ¬ (parallel m n ∧ lies_in m a → parallel n a) :=
sorry

end NUMINAMATH_GPT_incorrect_option_C_l2172_217278


namespace NUMINAMATH_GPT_equivalent_octal_to_decimal_l2172_217200

def octal_to_decimal (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n+1 => (n % 10) + 8 * octal_to_decimal (n / 10)

theorem equivalent_octal_to_decimal : octal_to_decimal 753 = 491 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_octal_to_decimal_l2172_217200


namespace NUMINAMATH_GPT_red_sequence_2018th_num_l2172_217240

/-- Define the sequence of red-colored numbers based on the given conditions. -/
def red_sequenced_num (n : Nat) : Nat :=
  let k := Nat.sqrt (2 * n - 1) -- estimate block number
  let block_start := if k % 2 == 0 then (k - 1)*(k - 1) else k * (k - 1) + 1
  let position_in_block := n - (k * (k - 1) / 2) - 1
  if k % 2 == 0 then block_start + 2 * position_in_block else block_start + 2 * position_in_block

/-- Statement to assert the 2018th number is 3972 -/
theorem red_sequence_2018th_num : red_sequenced_num 2018 = 3972 := by
  sorry

end NUMINAMATH_GPT_red_sequence_2018th_num_l2172_217240


namespace NUMINAMATH_GPT_proposition_induction_l2172_217237

variable (P : ℕ → Prop)
variable (k : ℕ)

theorem proposition_induction (h : ∀ k : ℕ, P k → P (k + 1))
    (h9 : ¬ P 9) : ¬ P 8 :=
by
  sorry

end NUMINAMATH_GPT_proposition_induction_l2172_217237


namespace NUMINAMATH_GPT_number_of_chickens_l2172_217263

-- Definitions based on conditions
def totalAnimals := 100
def legDifference := 26

-- The problem statement to be proved
theorem number_of_chickens (x : Nat) (r : Nat) (legs_chickens : Nat) (legs_rabbits : Nat) (total : Nat := totalAnimals) (diff : Nat := legDifference) :
  x + r = total ∧ 2 * x + 4 * r - 4 * r = 2 * x + diff → x = 71 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_chickens_l2172_217263


namespace NUMINAMATH_GPT_total_payment_divisible_by_25_l2172_217209

theorem total_payment_divisible_by_25 (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 9) : 
  (2005 + B * 1000) % 25 = 0 :=
by
  sorry

end NUMINAMATH_GPT_total_payment_divisible_by_25_l2172_217209


namespace NUMINAMATH_GPT_situps_combined_l2172_217294

theorem situps_combined (peter_situps : ℝ) (greg_per_set : ℝ) (susan_per_set : ℝ) 
                        (peter_per_set : ℝ) (sets : ℝ) 
                        (peter_situps_performed : peter_situps = sets * peter_per_set) 
                        (greg_situps_performed : sets * greg_per_set = 4.5 * 6)
                        (susan_situps_performed : sets * susan_per_set = 3.75 * 6) :
    peter_situps = 37.5 ∧ greg_per_set = 4.5 ∧ susan_per_set = 3.75 ∧ peter_per_set = 6.25 → 
    4.5 * 6 + 3.75 * 6 = 49.5 :=
by
  sorry

end NUMINAMATH_GPT_situps_combined_l2172_217294


namespace NUMINAMATH_GPT_trig_expression_l2172_217234

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_trig_expression_l2172_217234


namespace NUMINAMATH_GPT_geometric_sequence_S5_equals_l2172_217231

theorem geometric_sequence_S5_equals :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    a 1 = 1 → 
    (a 3 + a 4) / (a 1 + a 2) = 4 → 
    ((S5 : ℤ) = 31 ∨ (S5 : ℤ) = 11) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S5_equals_l2172_217231


namespace NUMINAMATH_GPT_bacon_calories_percentage_l2172_217249

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end NUMINAMATH_GPT_bacon_calories_percentage_l2172_217249


namespace NUMINAMATH_GPT_no_prime_satisfies_condition_l2172_217230

theorem no_prime_satisfies_condition (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, (Real.sqrt (p + n) + Real.sqrt n) = k :=
by
  sorry

end NUMINAMATH_GPT_no_prime_satisfies_condition_l2172_217230


namespace NUMINAMATH_GPT_rational_function_sum_l2172_217207

-- Define the problem conditions and the target equality
theorem rational_function_sum (p q : ℝ → ℝ) :
  (∀ x, (p x) / (q x) = (x - 1) / ((x + 1) * (x - 1))) ∧
  (∀ x ≠ -1, q x ≠ 0) ∧
  (q 2 = 3) ∧
  (p 2 = 1) →
  (p x + q x = x^2 + x - 2) := by
  sorry

end NUMINAMATH_GPT_rational_function_sum_l2172_217207


namespace NUMINAMATH_GPT_train_passes_man_in_approximately_24_seconds_l2172_217245

noncomputable def train_length : ℝ := 880 -- length of the train in meters
noncomputable def train_speed_kmph : ℝ := 120 -- speed of the train in km/h
noncomputable def man_speed_kmph : ℝ := 12 -- speed of the man in km/h

noncomputable def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph
noncomputable def relative_speed : ℝ := train_speed_mps + man_speed_mps

noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem train_passes_man_in_approximately_24_seconds :
  abs (time_to_pass - 24) < 1 :=
sorry

end NUMINAMATH_GPT_train_passes_man_in_approximately_24_seconds_l2172_217245


namespace NUMINAMATH_GPT_possible_third_side_of_triangle_l2172_217229

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end NUMINAMATH_GPT_possible_third_side_of_triangle_l2172_217229


namespace NUMINAMATH_GPT_left_handed_women_percentage_l2172_217279

noncomputable section

variables (x y : ℕ) (percentage : ℝ)

-- Conditions
def right_handed_ratio := 3
def left_handed_ratio := 1
def men_ratio := 3
def women_ratio := 2

def total_population_by_hand := right_handed_ratio * x + left_handed_ratio * x -- i.e., 4x
def total_population_by_gender := men_ratio * y + women_ratio * y -- i.e., 5y

-- Main Statement
theorem left_handed_women_percentage (h1 : total_population_by_hand = total_population_by_gender) :
    percentage = 25 :=
by
  sorry

end NUMINAMATH_GPT_left_handed_women_percentage_l2172_217279


namespace NUMINAMATH_GPT_sanity_indeterminable_transylvanian_is_upyr_l2172_217203

noncomputable def transylvanianClaim := "I have lost my mind."

/-- Proving whether the sanity of the Transylvanian can be determined from the statement -/
theorem sanity_indeterminable (claim : String) : 
  claim = transylvanianClaim → 
  ¬ (∀ (sane : Prop), sane ∨ ¬ sane) := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

/-- Proving the nature of whether the Transylvanian is an upyr or human from the statement -/
theorem transylvanian_is_upyr (claim : String) : 
  claim = transylvanianClaim → 
  ∀ (human upyr : Prop), ¬ human ∧ upyr := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

end NUMINAMATH_GPT_sanity_indeterminable_transylvanian_is_upyr_l2172_217203


namespace NUMINAMATH_GPT_find_d_l2172_217268

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end NUMINAMATH_GPT_find_d_l2172_217268


namespace NUMINAMATH_GPT_find_P_nplus1_l2172_217297

-- Conditions
def P (n : ℕ) (k : ℕ) : ℚ :=
  1 / Nat.choose n k

-- Lean 4 statement for the proof
theorem find_P_nplus1 (n : ℕ) : (if Even n then P n (n+1) = 1 else P n (n+1) = 0) := by
  sorry

end NUMINAMATH_GPT_find_P_nplus1_l2172_217297


namespace NUMINAMATH_GPT_direct_proportion_conditions_l2172_217260

theorem direct_proportion_conditions (k b : ℝ) : 
  (y = (k - 4) * x + b → (k ≠ 4 ∧ b = 0)) ∧ ¬ (b ≠ 0 ∨ k ≠ 4) :=
sorry

end NUMINAMATH_GPT_direct_proportion_conditions_l2172_217260


namespace NUMINAMATH_GPT_plane_distance_last_10_seconds_l2172_217224

theorem plane_distance_last_10_seconds (s : ℝ → ℝ) (h : ∀ t, s t = 60 * t - 1.5 * t^2) : 
  s 20 - s 10 = 150 := 
by 
  sorry

end NUMINAMATH_GPT_plane_distance_last_10_seconds_l2172_217224


namespace NUMINAMATH_GPT_rowing_speed_l2172_217290

theorem rowing_speed :
  ∀ (initial_width final_width increase_per_10m : ℝ) (time_seconds : ℝ)
  (yards_to_meters : ℝ → ℝ) (width_increase_in_yards : ℝ) (distance_10m_segments : ℝ) 
  (total_distance : ℝ),
  initial_width = 50 →
  final_width = 80 →
  increase_per_10m = 2 →
  time_seconds = 30 →
  yards_to_meters 1 = 0.9144 →
  width_increase_in_yards = (final_width - initial_width) →
  width_increase_in_yards * (yards_to_meters 1) = 27.432 →
  distance_10m_segments = (width_increase_in_yards * (yards_to_meters 1)) / 10 →
  total_distance = distance_10m_segments * 10 →
  (total_distance / time_seconds) = 0.9144 :=
by
  intros initial_width final_width increase_per_10m time_seconds yards_to_meters 
        width_increase_in_yards distance_10m_segments total_distance
  sorry

end NUMINAMATH_GPT_rowing_speed_l2172_217290


namespace NUMINAMATH_GPT_number_divided_by_three_l2172_217218

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end NUMINAMATH_GPT_number_divided_by_three_l2172_217218


namespace NUMINAMATH_GPT_pipe_c_empty_time_l2172_217252

theorem pipe_c_empty_time (x : ℝ) :
  (4/20 + 4/30 + 4/x) * 3 = 1 → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_pipe_c_empty_time_l2172_217252


namespace NUMINAMATH_GPT_inversely_proportional_x_y_l2172_217217

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_x_y_l2172_217217


namespace NUMINAMATH_GPT_vertex_below_x_axis_l2172_217262

theorem vertex_below_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + a < 0) → a < 1 :=
by 
  sorry

end NUMINAMATH_GPT_vertex_below_x_axis_l2172_217262


namespace NUMINAMATH_GPT_problem1_problem2_l2172_217205

open Real -- Open the Real namespace to use real number trigonometric functions

-- Problem 1
theorem problem1 (α : ℝ) (hα : tan α = 3) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5/7 :=
sorry

-- Problem 2
theorem problem2 (θ : ℝ) (hθ : tan θ = -3/4) : 
  2 + sin θ * cos θ - cos θ ^ 2 = 22 / 25 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2172_217205


namespace NUMINAMATH_GPT_jan_drives_more_miles_than_ian_l2172_217269

-- Definitions of conditions
variables (s t d m: ℝ)

-- Ian's travel equation
def ian_distance := d = s * t

-- Han's travel equation
def han_distance := (d + 115) = (s + 8) * (t + 2)

-- Jan's travel equation
def jan_distance := m = (s + 12) * (t + 3)

-- The proof statement we want to prove
theorem jan_drives_more_miles_than_ian :
    (∀ (s t d m : ℝ),
    d = s * t →
    (d + 115) = (s + 8) * (t + 2) →
    m = (s + 12) * (t + 3) →
    (m - d) = 184.5) :=
    sorry

end NUMINAMATH_GPT_jan_drives_more_miles_than_ian_l2172_217269


namespace NUMINAMATH_GPT_truncated_cone_volume_correct_larger_cone_volume_correct_l2172_217282

def larger_base_radius : ℝ := 10 -- R
def smaller_base_radius : ℝ := 5  -- r
def height_truncated_cone : ℝ := 8 -- h
def height_small_cone : ℝ := 8 -- x

noncomputable def volume_truncated_cone : ℝ :=
  (1/3) * Real.pi * height_truncated_cone * 
  (larger_base_radius^2 + larger_base_radius * smaller_base_radius + smaller_base_radius^2)

theorem truncated_cone_volume_correct :
  volume_truncated_cone = 466 + 2/3 * Real.pi := sorry

noncomputable def total_height_larger_cone : ℝ :=
  height_small_cone + height_truncated_cone

noncomputable def volume_larger_cone : ℝ :=
  (1/3) * Real.pi * (larger_base_radius^2) * total_height_larger_cone

theorem larger_cone_volume_correct :
  volume_larger_cone = 533 + 1/3 * Real.pi := sorry

end NUMINAMATH_GPT_truncated_cone_volume_correct_larger_cone_volume_correct_l2172_217282


namespace NUMINAMATH_GPT_g_at_pi_over_4_l2172_217283

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem g_at_pi_over_4 : g (Real.pi / 4) = 3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_g_at_pi_over_4_l2172_217283


namespace NUMINAMATH_GPT_johns_website_visits_l2172_217253

theorem johns_website_visits (c: ℝ) (d: ℝ) (days: ℕ) (h1: c = 0.01) (h2: d = 10) (h3: days = 30) :
  d / c * days = 30000 :=
by
  sorry

end NUMINAMATH_GPT_johns_website_visits_l2172_217253


namespace NUMINAMATH_GPT_mushroom_pickers_l2172_217261

theorem mushroom_pickers (n : ℕ) (hn : n = 18) (total_mushrooms : ℕ) (h_total : total_mushrooms = 162) (h_each : ∀ i : ℕ, i < n → 0 < 1) : 
  ∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (total_mushrooms / n = (total_mushrooms / n)) :=
sorry

end NUMINAMATH_GPT_mushroom_pickers_l2172_217261


namespace NUMINAMATH_GPT_always_positive_expression_l2172_217281

variable (x a b : ℝ)

theorem always_positive_expression (h : ∀ x, (x - a)^2 + b > 0) : b > 0 :=
sorry

end NUMINAMATH_GPT_always_positive_expression_l2172_217281


namespace NUMINAMATH_GPT_min_a_plus_5b_l2172_217244

theorem min_a_plus_5b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a * b + b^2 = b + 1) : 
  a + 5 * b ≥ 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_a_plus_5b_l2172_217244


namespace NUMINAMATH_GPT_andrew_eggs_bought_l2172_217221

-- Define initial conditions
def initial_eggs : ℕ := 8
def final_eggs : ℕ := 70

-- Define the function to determine the number of eggs bought
def eggs_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

-- State the theorem we want to prove
theorem andrew_eggs_bought : eggs_bought initial_eggs final_eggs = 62 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_andrew_eggs_bought_l2172_217221


namespace NUMINAMATH_GPT_sum_of_possible_coefficient_values_l2172_217212

theorem sum_of_possible_coefficient_values :
  let pairs := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]
  let values := pairs.map (fun (r, s) => r + s)
  values.sum = 124 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_coefficient_values_l2172_217212


namespace NUMINAMATH_GPT_roots_greater_than_one_l2172_217216

def quadratic_roots_greater_than_one (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1

theorem roots_greater_than_one (a : ℝ) :
  -16/7 < a ∧ a < -1 → quadratic_roots_greater_than_one a :=
sorry

end NUMINAMATH_GPT_roots_greater_than_one_l2172_217216


namespace NUMINAMATH_GPT_squirrel_population_difference_l2172_217267

theorem squirrel_population_difference :
  ∀ (total_population scotland_population rest_uk_population : ℕ), 
  scotland_population = 120000 →
  120000 = 75 * total_population / 100 →
  rest_uk_population = total_population - scotland_population →
  scotland_population - rest_uk_population = 80000 :=
by
  intros total_population scotland_population rest_uk_population h1 h2 h3
  sorry

end NUMINAMATH_GPT_squirrel_population_difference_l2172_217267
