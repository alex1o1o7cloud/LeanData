import Mathlib

namespace NUMINAMATH_GPT_fraction_to_terminanting_decimal_l1741_174136

theorem fraction_to_terminanting_decimal : (47 / (5^4 * 2) : ℚ) = 0.0376 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_to_terminanting_decimal_l1741_174136


namespace NUMINAMATH_GPT_regular_polygon_sides_l1741_174191

theorem regular_polygon_sides (perimeter side_length : ℝ) (h1 : perimeter = 180) (h2 : side_length = 15) :
  perimeter / side_length = 12 :=
by sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1741_174191


namespace NUMINAMATH_GPT_sector_area_correct_l1741_174122

noncomputable def sector_area (θ r : ℝ) : ℝ :=
  (θ / (2 * Real.pi)) * (Real.pi * r^2)

theorem sector_area_correct : 
  sector_area (Real.pi / 3) 3 = (3 / 2) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_area_correct_l1741_174122


namespace NUMINAMATH_GPT_boxes_left_l1741_174187

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end NUMINAMATH_GPT_boxes_left_l1741_174187


namespace NUMINAMATH_GPT_find_D_l1741_174156

-- Definitions
def divides (a b : ℕ) : Prop := ∃ k, b = a * k
def remainder (a b r : ℕ) : Prop := ∃ k, a = b * k + r

-- Problem Statement
theorem find_D {N D : ℕ} (h1 : remainder N D 75) (h2 : remainder N 37 1) : 
  D = 112 :=
by
  sorry

end NUMINAMATH_GPT_find_D_l1741_174156


namespace NUMINAMATH_GPT_john_paid_after_tax_l1741_174104

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end NUMINAMATH_GPT_john_paid_after_tax_l1741_174104


namespace NUMINAMATH_GPT_johns_old_cards_l1741_174174

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 8

def total_cards := total_pages * cards_per_page
def old_cards := total_cards - new_cards

theorem johns_old_cards :
  old_cards = 16 :=
by
  -- Note: No specific solution steps needed here, just stating the theorem
  sorry

end NUMINAMATH_GPT_johns_old_cards_l1741_174174


namespace NUMINAMATH_GPT_pump1_half_drain_time_l1741_174103

-- Definitions and Conditions
def time_to_drain_half_pump1 (t : ℝ) : Prop :=
  ∃ rate1 rate2 : ℝ, 
    rate1 = 1 / (2 * t) ∧
    rate2 = 1 / 1.25 ∧
    rate1 + rate2 = 2

-- Equivalent Proof Problem
theorem pump1_half_drain_time (t : ℝ) : time_to_drain_half_pump1 t → t = 5 / 12 := sorry

end NUMINAMATH_GPT_pump1_half_drain_time_l1741_174103


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1741_174194

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  (1/x) + (1/y) = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1741_174194


namespace NUMINAMATH_GPT_batsman_average_increase_l1741_174138

theorem batsman_average_increase
  (prev_avg : ℝ) -- average before the 17th innings
  (total_runs_16 : ℝ := 16 * prev_avg) -- total runs scored in the first 16 innings
  (score_17th : ℝ := 85) -- score in the 17th innings
  (new_avg : ℝ := 37) -- new average after 17 innings
  (total_runs_17 : ℝ := total_runs_16 + score_17th) -- total runs after 17 innings
  (calc_total_runs_17 : ℝ := 17 * new_avg) -- new total runs calculated by the new average
  (h : total_runs_17 = calc_total_runs_17) -- given condition: total_runs_17 = calc_total_runs_17
  : (new_avg - prev_avg) = 3 := 
by
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l1741_174138


namespace NUMINAMATH_GPT_slope_of_line_inclination_angle_l1741_174148

theorem slope_of_line_inclination_angle 
  (k : ℝ) (θ : ℝ)
  (hθ1 : 30 * (π / 180) < θ)
  (hθ2 : θ < 90 * (π / 180)) :
  k = Real.tan θ → k > Real.tan (30 * (π / 180)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_slope_of_line_inclination_angle_l1741_174148


namespace NUMINAMATH_GPT_car_pass_time_l1741_174185

theorem car_pass_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) :
  length = 10 → 
  speed_kmph = 36 → 
  speed_mps = speed_kmph * (1000 / 3600) → 
  time = length / speed_mps → 
  time = 1 :=
by
  intros h_length h_speed_kmph h_speed_conversion h_time_calculation
  -- Here we would normally construct the proof
  sorry

end NUMINAMATH_GPT_car_pass_time_l1741_174185


namespace NUMINAMATH_GPT_power_of_i_l1741_174134

theorem power_of_i (i : ℂ) (h₀ : i^2 = -1) : i^(2016) = 1 :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_power_of_i_l1741_174134


namespace NUMINAMATH_GPT_coupon_probability_l1741_174107

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end NUMINAMATH_GPT_coupon_probability_l1741_174107


namespace NUMINAMATH_GPT_total_amount_paid_l1741_174189

/-- The owner's markup percentage and the cost price are given. 
We need to find out the total amount paid by the customer, which is equivalent to proving the total cost. -/
theorem total_amount_paid (markup_percentage : ℝ) (cost_price : ℝ) (markup : ℝ) (total_paid : ℝ) 
    (h1 : markup_percentage = 0.24) 
    (h2 : cost_price = 6425) 
    (h3 : markup = markup_percentage * cost_price) 
    (h4 : total_paid = cost_price + markup) : 
    total_paid = 7967 := 
sorry

end NUMINAMATH_GPT_total_amount_paid_l1741_174189


namespace NUMINAMATH_GPT_minimize_x_plus_y_on_circle_l1741_174105

theorem minimize_x_plus_y_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : x + y ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_x_plus_y_on_circle_l1741_174105


namespace NUMINAMATH_GPT_ellipse_centroid_locus_l1741_174123

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
noncomputable def centroid_locus (x y : ℝ) : Prop := (9 * x^2) / 4 + 3 * y^2 = 1 ∧ y ≠ 0

theorem ellipse_centroid_locus (x y : ℝ) (h : ellipse_equation x y) : centroid_locus (x / 3) (y / 3) :=
  sorry

end NUMINAMATH_GPT_ellipse_centroid_locus_l1741_174123


namespace NUMINAMATH_GPT_tangent_line_at_1_0_monotonic_intervals_l1741_174169

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * Real.log x

noncomputable def f_derivative (x : ℝ) (a : ℝ) : ℝ := (2 * x^2 - a * x + 2) / x

theorem tangent_line_at_1_0 (a : ℝ) (h : a = 1) :
  ∀ x y : ℝ, 
  (f x a, f 1 a) = (0, x - 1) → 
  y = 3 * x - 3 := 
sorry

theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, 0 < x → f_derivative x a ≥ 0) ↔ (a ≤ 4) ∧ 
  (∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < (a - Real.sqrt (a^2 - 16)) / 4) ∨ 
    ((a + Real.sqrt (a^2 - 16)) / 4 < x) 
  ) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_1_0_monotonic_intervals_l1741_174169


namespace NUMINAMATH_GPT_entree_cost_difference_l1741_174188

theorem entree_cost_difference 
  (total_cost : ℕ)
  (entree_cost : ℕ)
  (dessert_cost : ℕ)
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14)
  (h3 : total_cost = entree_cost + dessert_cost) :
  entree_cost - dessert_cost = 5 :=
by
  sorry

end NUMINAMATH_GPT_entree_cost_difference_l1741_174188


namespace NUMINAMATH_GPT_ratio_steel_to_tin_l1741_174126

def mass_copper (C : ℝ) := C = 90
def total_weight (S C T : ℝ) := 20 * S + 20 * C + 20 * T = 5100
def mass_steel (S C : ℝ) := S = C + 20

theorem ratio_steel_to_tin (S T C : ℝ)
  (hC : mass_copper C)
  (hTW : total_weight S C T)
  (hS : mass_steel S C) :
  S / T = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_steel_to_tin_l1741_174126


namespace NUMINAMATH_GPT_x_cubed_gt_y_squared_l1741_174121

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end NUMINAMATH_GPT_x_cubed_gt_y_squared_l1741_174121


namespace NUMINAMATH_GPT_fans_received_all_items_l1741_174133

theorem fans_received_all_items (n : ℕ) (h1 : (∀ k : ℕ, k * 45 ≤ n → (k * 45) ∣ n))
                                (h2 : (∀ k : ℕ, k * 50 ≤ n → (k * 50) ∣ n))
                                (h3 : (∀ k : ℕ, k * 100 ≤ n → (k * 100) ∣ n))
                                (capacity_full : n = 5000) :
  n / Nat.lcm 45 (Nat.lcm 50 100) = 5 :=
by
  sorry

end NUMINAMATH_GPT_fans_received_all_items_l1741_174133


namespace NUMINAMATH_GPT_find_constants_l1741_174113

noncomputable section

theorem find_constants (P Q R : ℝ)
  (h : ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
    (5*x^2 + 7*x) / ((x - 2) * (x - 4)^2) =
    P / (x - 2) + Q / (x - 4) + R / (x - 4)^2) :
  P = 3.5 ∧ Q = 1.5 ∧ R = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1741_174113


namespace NUMINAMATH_GPT_amount_subtracted_is_30_l1741_174109

-- Definitions based on conditions
def N : ℕ := 200
def subtracted_amount (A : ℕ) : Prop := 0.40 * (N : ℝ) - (A : ℝ) = 50

-- The theorem statement
theorem amount_subtracted_is_30 : subtracted_amount 30 :=
by 
  -- proof will be completed here
  sorry

end NUMINAMATH_GPT_amount_subtracted_is_30_l1741_174109


namespace NUMINAMATH_GPT_evaluate_g_sum_l1741_174139

def g (a b : ℚ) : ℚ :=
if a + b ≤ 5 then (a^2 * b - a + 3) / (3 * a) 
else (a * b^2 - b - 3) / (-3 * b)

theorem evaluate_g_sum : g 3 2 + g 3 3 = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_g_sum_l1741_174139


namespace NUMINAMATH_GPT_inequality_proof_l1741_174118

theorem inequality_proof (a b c x y z : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) :
  (a^2 * x^2 / ((b * y + c * z) * (b * z + c * y)) + 
   b^2 * y^2 / ((a * x + c * z) * (a * z + c * x)) +
   c^2 * z^2 / ((a * x + b * y) * (a * y + b * x))) ≥ 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1741_174118


namespace NUMINAMATH_GPT_choir_final_score_l1741_174152

theorem choir_final_score (content_score sing_score spirit_score : ℕ)
  (content_weight sing_weight spirit_weight : ℝ)
  (h_content : content_weight = 0.30) 
  (h_sing : sing_weight = 0.50) 
  (h_spirit : spirit_weight = 0.20) 
  (h_content_score : content_score = 90)
  (h_sing_score : sing_score = 94)
  (h_spirit_score : spirit_score = 95) :
  content_weight * content_score + sing_weight * sing_score + spirit_weight * spirit_score = 93 := by
  sorry

end NUMINAMATH_GPT_choir_final_score_l1741_174152


namespace NUMINAMATH_GPT_complex_sum_equals_one_l1741_174119

noncomputable def main (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1))

theorem complex_sum_equals_one (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : main x h1 h2 = 1 := by
  sorry

end NUMINAMATH_GPT_complex_sum_equals_one_l1741_174119


namespace NUMINAMATH_GPT_problem_statement_l1741_174163

theorem problem_statement (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 1) (x : ℝ) :
    (m * x^2 - 2 * x - m ≥ 2) ↔ (x ≤ -1) := sorry

end NUMINAMATH_GPT_problem_statement_l1741_174163


namespace NUMINAMATH_GPT_cricket_player_average_l1741_174154

theorem cricket_player_average (A : ℝ) (h1 : 10 * A + 84 = 11 * (A + 4)) : A = 40 :=
by
  sorry

end NUMINAMATH_GPT_cricket_player_average_l1741_174154


namespace NUMINAMATH_GPT_value_of_x_l1741_174182

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1741_174182


namespace NUMINAMATH_GPT_find_C_D_l1741_174181

theorem find_C_D (x C D : ℚ) 
  (h : 7 * x - 5 ≠ 0) -- Added condition to avoid zero denominator
  (hx : x^2 - 8 * x - 48 = (x - 12) * (x + 4))
  (h_eq : 7 * x - 5 = C * (x + 4) + D * (x - 12))
  (h_c : C = 79 / 16)
  (h_d : D = 33 / 16)
: 7 * x - 5 = 79 / 16 * (x + 4) + 33 / 16 * (x - 12) :=
by sorry

end NUMINAMATH_GPT_find_C_D_l1741_174181


namespace NUMINAMATH_GPT_first_thrilling_thursday_after_start_l1741_174128

theorem first_thrilling_thursday_after_start (start_date : ℕ) (school_start_month : ℕ) (school_start_day_of_week : ℤ) (month_length : ℕ → ℕ) (day_of_week_on_first_of_month : ℕ → ℤ) : 
    school_start_month = 9 ∧ school_start_day_of_week = 2 ∧ start_date = 12 ∧ month_length 9 = 30 ∧ day_of_week_on_first_of_month 10 = 0 → 
    ∃ day_of_thursday : ℕ, day_of_thursday = 26 :=
by
  sorry

end NUMINAMATH_GPT_first_thrilling_thursday_after_start_l1741_174128


namespace NUMINAMATH_GPT_max_minute_hands_l1741_174130

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end NUMINAMATH_GPT_max_minute_hands_l1741_174130


namespace NUMINAMATH_GPT_anne_total_bottle_caps_l1741_174125

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end NUMINAMATH_GPT_anne_total_bottle_caps_l1741_174125


namespace NUMINAMATH_GPT_area_of_square_efgh_proof_l1741_174172

noncomputable def area_of_square_efgh : ℝ :=
  let original_square_side_length := 3
  let radius_of_circles := (3 * Real.sqrt 2) / 2
  let efgh_side_length := original_square_side_length + 2 * radius_of_circles 
  efgh_side_length ^ 2

theorem area_of_square_efgh_proof :
  area_of_square_efgh = 27 + 18 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_efgh_proof_l1741_174172


namespace NUMINAMATH_GPT_rectangle_area_relation_l1741_174198

theorem rectangle_area_relation (x y : ℝ) (h : x * y = 4) (hx : x > 0) : y = 4 / x := 
sorry

end NUMINAMATH_GPT_rectangle_area_relation_l1741_174198


namespace NUMINAMATH_GPT_length_of_segment_l1741_174115

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_l1741_174115


namespace NUMINAMATH_GPT_max_writers_and_editors_l1741_174179

theorem max_writers_and_editors (total people writers editors y x : ℕ) 
  (h1 : total = 110) 
  (h2 : writers = 45) 
  (h3 : editors = 38 + y) 
  (h4 : y > 0) 
  (h5 : 45 + editors + 2 * x = 110) : 
  x = 13 := 
sorry

end NUMINAMATH_GPT_max_writers_and_editors_l1741_174179


namespace NUMINAMATH_GPT_election_valid_votes_l1741_174108

variable (V : ℕ)
variable (invalid_pct : ℝ)
variable (exceed_pct : ℝ)
variable (total_votes : ℕ)
variable (invalid_votes : ℝ)
variable (valid_votes : ℕ)
variable (A_votes : ℕ)
variable (B_votes : ℕ)

theorem election_valid_votes :
  V = 9720 →
  invalid_pct = 0.20 →
  exceed_pct = 0.15 →
  total_votes = V →
  invalid_votes = invalid_pct * V →
  valid_votes = total_votes - invalid_votes →
  A_votes = B_votes + exceed_pct * total_votes →
  A_votes + B_votes = valid_votes →
  B_votes = 3159 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_election_valid_votes_l1741_174108


namespace NUMINAMATH_GPT_number_of_boys_l1741_174165

variable (x y : ℕ)

theorem number_of_boys (h1 : x + y = 900) (h2 : y = (x / 100) * 900) : x = 90 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l1741_174165


namespace NUMINAMATH_GPT_probability_at_least_one_six_is_11_div_36_l1741_174151

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_six_is_11_div_36_l1741_174151


namespace NUMINAMATH_GPT_harmonic_mean_pairs_l1741_174199

open Nat

theorem harmonic_mean_pairs :
  ∃ n, n = 199 ∧ 
  (∀ (x y : ℕ), 0 < x → 0 < y → 
  x < y → (2 * x * y) / (x + y) = 6^10 → 
  x * y - (3^10 * 2^9) * (x - 1) - (3^10 * 2^9) * (y - 1) = 3^20 * 2^18) :=
sorry

end NUMINAMATH_GPT_harmonic_mean_pairs_l1741_174199


namespace NUMINAMATH_GPT_sqrt_ceil_eq_sqrt_sqrt_l1741_174158

theorem sqrt_ceil_eq_sqrt_sqrt (a : ℝ) (h : a > 1) : 
  (Int.floor (Real.sqrt (Int.floor (Real.sqrt a)))) = (Int.floor (Real.sqrt (Real.sqrt a))) :=
sorry

end NUMINAMATH_GPT_sqrt_ceil_eq_sqrt_sqrt_l1741_174158


namespace NUMINAMATH_GPT_line_intersects_curve_equal_segments_l1741_174196

theorem line_intersects_curve_equal_segments (k m : ℝ)
  (A B C : ℝ × ℝ)
  (hA_curve : A.2 = A.1^3 - 6 * A.1^2 + 13 * A.1 - 8)
  (hB_curve : B.2 = B.1^3 - 6 * B.1^2 + 13 * B.1 - 8)
  (hC_curve : C.2 = C.1^3 - 6 * C.1^2 + 13 * C.1 - 8)
  (h_lineA : A.2 = k * A.1 + m)
  (h_lineB : B.2 = k * B.1 + m)
  (h_lineC : C.2 = k * C.1 + m)
  (h_midpoint : 2 * B.1 = A.1 + C.1 ∧ 2 * B.2 = A.2 + C.2)
  : 2 * k + m = 2 :=
sorry

end NUMINAMATH_GPT_line_intersects_curve_equal_segments_l1741_174196


namespace NUMINAMATH_GPT_calc_f_g_3_minus_g_f_3_l1741_174193

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2

theorem calc_f_g_3_minus_g_f_3 :
  (f (g 3) - g (f 3)) = -96 :=
by
  sorry

end NUMINAMATH_GPT_calc_f_g_3_minus_g_f_3_l1741_174193


namespace NUMINAMATH_GPT_seq_general_term_l1741_174150

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 0 then 1/2
  else if n = 1 then 1/2
  else seq (n - 1) * 3 / (seq (n - 1) + 3)

theorem seq_general_term : ∀ n : ℕ, seq (n + 1) = 3 / (n + 6) :=
by
  intro n
  induction n with
  | zero => sorry
  | succ k ih => sorry

end NUMINAMATH_GPT_seq_general_term_l1741_174150


namespace NUMINAMATH_GPT_olive_needs_two_colours_l1741_174180

theorem olive_needs_two_colours (α : Type) [Finite α] (G : SimpleGraph α) (colour : α → Fin 2) :
  (∀ v : α, ∃! w : α, G.Adj v w ∧ colour v = colour w) → ∃ color_map : α → Fin 2, ∀ v, ∃! w, G.Adj v w ∧ color_map v = color_map w :=
sorry

end NUMINAMATH_GPT_olive_needs_two_colours_l1741_174180


namespace NUMINAMATH_GPT_sum_a_for_exactly_one_solution_l1741_174153

theorem sum_a_for_exactly_one_solution :
  (∀ a : ℝ, ∃ x : ℝ, 3 * x^2 + (a + 6) * x + 7 = 0) →
  ((-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12) :=
by
  sorry

end NUMINAMATH_GPT_sum_a_for_exactly_one_solution_l1741_174153


namespace NUMINAMATH_GPT_right_triangle_area_l1741_174131

theorem right_triangle_area (h b : ℝ) (hypotenuse : h = 5) (base : b = 3) :
  ∃ a : ℝ, a = 1 / 2 * b * (Real.sqrt (h^2 - b^2)) ∧ a = 6 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1741_174131


namespace NUMINAMATH_GPT_cost_of_four_dozen_l1741_174135

-- Defining the conditions
def cost_of_three_dozen (cost : ℚ) : Prop :=
  cost = 25.20

-- The theorem to prove the cost of four dozen apples at the same rate
theorem cost_of_four_dozen (cost : ℚ) :
  cost_of_three_dozen cost →
  (4 * (cost / 3) = 33.60) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_four_dozen_l1741_174135


namespace NUMINAMATH_GPT_min_satisfies_condition_only_for_x_eq_1_div_4_l1741_174141

theorem min_satisfies_condition_only_for_x_eq_1_div_4 (x : ℝ) (hx_nonneg : 0 ≤ x) :
  (min (Real.sqrt x) (min (x^2) x) = 1/16) ↔ (x = 1/4) :=
by sorry

end NUMINAMATH_GPT_min_satisfies_condition_only_for_x_eq_1_div_4_l1741_174141


namespace NUMINAMATH_GPT_total_questions_l1741_174100

theorem total_questions (S C I : ℕ) (h1 : S = 73) (h2 : C = 91) (h3 : S = C - 2 * I) : C + I = 100 :=
sorry

end NUMINAMATH_GPT_total_questions_l1741_174100


namespace NUMINAMATH_GPT_author_hardcover_percentage_l1741_174124

variable {TotalPaperCopies : Nat}
variable {PricePerPaperCopy : ℝ}
variable {TotalHardcoverCopies : Nat}
variable {PricePerHardcoverCopy : ℝ}
variable {PaperPercentage : ℝ}
variable {TotalEarnings : ℝ}

theorem author_hardcover_percentage (TotalPaperCopies : Nat)
  (PricePerPaperCopy : ℝ) (TotalHardcoverCopies : Nat)
  (PricePerHardcoverCopy : ℝ) (PaperPercentage TotalEarnings : ℝ)
  (h1 : TotalPaperCopies = 32000) (h2 : PricePerPaperCopy = 0.20)
  (h3 : TotalHardcoverCopies = 15000) (h4 : PricePerHardcoverCopy = 0.40)
  (h5 : PaperPercentage = 0.06) (h6 : TotalEarnings = 1104) :
  (720 / (15000 * 0.40) * 100) = 12 := by
  sorry

end NUMINAMATH_GPT_author_hardcover_percentage_l1741_174124


namespace NUMINAMATH_GPT_fundraiser_goal_eq_750_l1741_174114

def bronze_donations := 10 * 25
def silver_donations := 7 * 50
def gold_donations   := 1 * 100
def total_collected  := bronze_donations + silver_donations + gold_donations
def amount_needed    := 50
def total_goal       := total_collected + amount_needed

theorem fundraiser_goal_eq_750 : total_goal = 750 :=
by
  sorry

end NUMINAMATH_GPT_fundraiser_goal_eq_750_l1741_174114


namespace NUMINAMATH_GPT_M_gt_N_l1741_174178

-- Define M and N
def M (x y : ℝ) : ℝ := x^2 + y^2 + 1
def N (x y : ℝ) : ℝ := 2 * (x + y - 1)

-- State the theorem to prove M > N given the conditions
theorem M_gt_N (x y : ℝ) : M x y > N x y := by
  sorry

end NUMINAMATH_GPT_M_gt_N_l1741_174178


namespace NUMINAMATH_GPT_right_triangle_acute_angle_l1741_174143

theorem right_triangle_acute_angle (a b : ℝ) (h1 : a + b = 90) (h2 : a = 55) : b = 35 := 
by sorry

end NUMINAMATH_GPT_right_triangle_acute_angle_l1741_174143


namespace NUMINAMATH_GPT_factorization_correct_l1741_174184

theorem factorization_correct (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1741_174184


namespace NUMINAMATH_GPT_shaded_area_of_overlap_l1741_174192

structure Rectangle where
  width : ℕ
  height : ℕ

structure Parallelogram where
  base : ℕ
  height : ℕ

def area_of_rectangle (r : Rectangle) : ℕ :=
  r.width * r.height

def area_of_parallelogram (p : Parallelogram) : ℕ :=
  p.base * p.height

def overlapping_area_square (side : ℕ) : ℕ :=
  side * side

theorem shaded_area_of_overlap 
  (r : Rectangle)
  (p : Parallelogram)
  (overlapping_side : ℕ)
  (h1 : r.width = 4)
  (h2 : r.height = 12)
  (h3 : p.base = 10)
  (h4 : p.height = 4)
  (h5 : overlapping_side = 4) :
  area_of_rectangle r + area_of_parallelogram p - overlapping_area_square overlapping_side = 72 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_overlap_l1741_174192


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l1741_174166

theorem arithmetic_sequence_n_value (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  a 672 = 2014 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l1741_174166


namespace NUMINAMATH_GPT_simplify_expression_l1741_174177

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1741_174177


namespace NUMINAMATH_GPT_drums_needed_for_profit_l1741_174176

def cost_to_enter_contest : ℝ := 10
def money_per_drum : ℝ := 0.025
def money_needed_for_profit (drums_hit : ℝ) : Prop :=
  drums_hit * money_per_drum > cost_to_enter_contest

theorem drums_needed_for_profit : ∃ D : ℝ, money_needed_for_profit D ∧ D = 400 :=
  by
  use 400
  sorry

end NUMINAMATH_GPT_drums_needed_for_profit_l1741_174176


namespace NUMINAMATH_GPT_minimum_value_expression_l1741_174162

open Real

theorem minimum_value_expression : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1741_174162


namespace NUMINAMATH_GPT_num_of_loads_l1741_174127

theorem num_of_loads (n : ℕ) (h1 : 7 * n = 42) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_of_loads_l1741_174127


namespace NUMINAMATH_GPT_condition_sufficient_not_necessary_l1741_174157

theorem condition_sufficient_not_necessary (x : ℝ) : (1 < x ∧ x < 2) → ((x - 2) ^ 2 < 1) ∧ ¬ ((x - 2) ^ 2 < 1 → (1 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficient_not_necessary_l1741_174157


namespace NUMINAMATH_GPT_sum_of_midpoints_of_triangle_l1741_174161

theorem sum_of_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_of_triangle_l1741_174161


namespace NUMINAMATH_GPT_exists_positive_integer_pow_not_integer_l1741_174167

theorem exists_positive_integer_pow_not_integer
  (α β : ℝ)
  (hαβ : α ≠ β)
  (h_non_int : ¬(↑⌊α⌋ = α ∧ ↑⌊β⌋ = β)) :
  ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, α^n - β^n = k :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_pow_not_integer_l1741_174167


namespace NUMINAMATH_GPT_dice_probability_l1741_174168

/-- A standard six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

open Die

/-- Calculates the probability that after re-rolling four dice, at least four out of the six total dice show the same number,
given that initially six dice are rolled and there is no three-of-a-kind, and there is a pair of dice showing the same number
which are then set aside before re-rolling the remaining four dice. -/
theorem dice_probability (h1 : ∀ (d1 d2 d3 d4 d5 d6 : Die), 
  ¬ (d1 = d2 ∧ d2 = d3 ∨ d1 = d2 ∧ d2 = d4 ∨ d1 = d2 ∧ d2 = d5 ∨
     d1 = d2 ∧ d2 = d6 ∨ d1 = d3 ∧ d3 = d4 ∨ d1 = d3 ∧ d3 = d5 ∨
     d1 = d3 ∧ d3 = d6 ∨ d1 = d4 ∧ d4 = d5 ∨ d1 = d4 ∧ d4 = d6 ∨
     d1 = d5 ∧ d5 = d6 ∨ d2 = d3 ∧ d3 = d4 ∨ d2 = d3 ∧ d3 = d5 ∨
     d2 = d3 ∧ d3 = d6 ∨ d2 = d4 ∧ d4 = d5 ∨ d2 = d4 ∧ d4 = d6 ∨
     d2 = d5 ∧ d5 = d6 ∨ d3 = d4 ∧ d4 = d5 ∨ d3 = d4 ∧ d4 = d6 ∨ d3 = d5 ∧ d5 = d6 ∨ d4 = d5 ∧ d5 = d6))
    (h2 : ∃ (d1 d2 : Die) (d3 d4 d5 d6 : Die), d1 = d2 ∧ d3 ≠ d1 ∧ d4 ≠ d1 ∧ d5 ≠ d1 ∧ d6 ≠ d1): 
    ℚ := 
11 / 81

end NUMINAMATH_GPT_dice_probability_l1741_174168


namespace NUMINAMATH_GPT_orthocenter_PQR_is_correct_l1741_174159

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end NUMINAMATH_GPT_orthocenter_PQR_is_correct_l1741_174159


namespace NUMINAMATH_GPT_regular_polygon_sides_l1741_174183

theorem regular_polygon_sides (θ : ℝ) (h : θ = 20) : 360 / θ = 18 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1741_174183


namespace NUMINAMATH_GPT_May4th_Sunday_l1741_174112

theorem May4th_Sunday (x : ℕ) (h_sum : x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 80) : 
  (4 % 7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_May4th_Sunday_l1741_174112


namespace NUMINAMATH_GPT_weight_triangle_correct_weight_l1741_174170

noncomputable def area_square (side : ℝ) : ℝ := side ^ 2

noncomputable def area_triangle (side : ℝ) : ℝ := (side ^ 2 * Real.sqrt 3) / 4

noncomputable def weight (area : ℝ) (density : ℝ) := area * density

noncomputable def weight_equilateral_triangle (weight_square : ℝ) (side_square : ℝ) (side_triangle : ℝ) : ℝ :=
  let area_s := area_square side_square
  let area_t := area_triangle side_triangle
  let density := weight_square / area_s
  weight area_t density

theorem weight_triangle_correct_weight :
  weight_equilateral_triangle 8 4 6 = 9 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_GPT_weight_triangle_correct_weight_l1741_174170


namespace NUMINAMATH_GPT_emily_small_gardens_l1741_174146

theorem emily_small_gardens 
  (total_seeds : ℕ)
  (seeds_in_big_garden : ℕ)
  (seeds_per_small_garden : ℕ)
  (remaining_seeds := total_seeds - seeds_in_big_garden)
  (number_of_small_gardens := remaining_seeds / seeds_per_small_garden) :
  total_seeds = 41 → seeds_in_big_garden = 29 → seeds_per_small_garden = 4 → number_of_small_gardens = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_emily_small_gardens_l1741_174146


namespace NUMINAMATH_GPT_tom_age_ratio_l1741_174116

theorem tom_age_ratio (T N : ℝ) (h1 : T - N = 3 * (T - 4 * N)) : T / N = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_tom_age_ratio_l1741_174116


namespace NUMINAMATH_GPT_contrapositive_of_not_p_implies_q_l1741_174195

variable (p q : Prop)

theorem contrapositive_of_not_p_implies_q :
  (¬p → q) → (¬q → p) := by
  sorry

end NUMINAMATH_GPT_contrapositive_of_not_p_implies_q_l1741_174195


namespace NUMINAMATH_GPT_no_solution_l1741_174145

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end NUMINAMATH_GPT_no_solution_l1741_174145


namespace NUMINAMATH_GPT_ultratown_run_difference_l1741_174106

/-- In Ultratown, the streets are all 25 feet wide, 
and the blocks they enclose are rectangular with lengths of 500 feet and widths of 300 feet. 
Hannah runs around the block on the longer 500-foot side of the street, 
while Harry runs on the opposite, outward side of the street. 
Prove that Harry runs 200 more feet than Hannah does for every lap around the block.
-/ 
theorem ultratown_run_difference :
  let street_width : ℕ := 25
  let inner_length : ℕ := 500
  let inner_width : ℕ := 300
  let outer_length := inner_length + 2 * street_width
  let outer_width := inner_width + 2 * street_width
  let inner_perimeter := 2 * (inner_length + inner_width)
  let outer_perimeter := 2 * (outer_length + outer_width)
  (outer_perimeter - inner_perimeter) = 200 :=
by
  sorry

end NUMINAMATH_GPT_ultratown_run_difference_l1741_174106


namespace NUMINAMATH_GPT_solution_set_inequality_l1741_174171

theorem solution_set_inequality (x : ℝ) (h : x - 3 / x > 2) :
    -1 < x ∧ x < 0 ∨ x > 3 :=
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1741_174171


namespace NUMINAMATH_GPT_sin_neg_thirtyone_sixths_pi_l1741_174111

theorem sin_neg_thirtyone_sixths_pi : Real.sin (-31 / 6 * Real.pi) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_sin_neg_thirtyone_sixths_pi_l1741_174111


namespace NUMINAMATH_GPT_first_stack_height_is_seven_l1741_174102

-- Definitions of the conditions
def first_stack (h : ℕ) := h
def second_stack (h : ℕ) := h + 5
def third_stack (h : ℕ) := h + 12

-- Conditions on the blocks falling down
def blocks_fell_first_stack (h : ℕ) := h
def blocks_fell_second_stack (h : ℕ) := (h + 5) - 2
def blocks_fell_third_stack (h : ℕ) := (h + 12) - 3

-- Total blocks fell down
def total_blocks_fell (h : ℕ) := blocks_fell_first_stack h + blocks_fell_second_stack h + blocks_fell_third_stack h

-- Lean statement to prove the height of the first stack
theorem first_stack_height_is_seven (h : ℕ) (h_eq : total_blocks_fell h = 33) : h = 7 :=
by sorry

-- Testing the conditions hold for the solution h = 7
#eval total_blocks_fell 7 -- Expected: 33

end NUMINAMATH_GPT_first_stack_height_is_seven_l1741_174102


namespace NUMINAMATH_GPT_total_amount_is_200_l1741_174117

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_is_200_l1741_174117


namespace NUMINAMATH_GPT_find_t_l1741_174132

theorem find_t (t : ℤ) :
  ((t + 1) * (3 * t - 3)) = ((3 * t - 5) * (t + 2) + 2) → 
  t = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_t_l1741_174132


namespace NUMINAMATH_GPT_sum_of_squares_s_comp_r_l1741_174110

def r (x : ℝ) : ℝ := x^2 - 4
def s (x : ℝ) : ℝ := -|x + 1|
def s_comp_r (x : ℝ) : ℝ := s (r x)

theorem sum_of_squares_s_comp_r :
  (s_comp_r (-4))^2 + (s_comp_r (-3))^2 + (s_comp_r (-2))^2 + (s_comp_r (-1))^2 +
  (s_comp_r 0)^2 + (s_comp_r 1)^2 + (s_comp_r 2)^2 + (s_comp_r 3)^2 + (s_comp_r 4)^2 = 429 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_s_comp_r_l1741_174110


namespace NUMINAMATH_GPT_simplify_expression_l1741_174142

theorem simplify_expression :
  (64^(1/3) - 216^(1/3) = -2) :=
by
  have h1 : 64 = 4^3 := by norm_num
  have h2 : 216 = 6^3 := by norm_num
  sorry

end NUMINAMATH_GPT_simplify_expression_l1741_174142


namespace NUMINAMATH_GPT_fill_time_60_gallons_ten_faucets_l1741_174140

-- Define the problem parameters
def rate_of_five_faucets : ℚ := 150 / 8 -- in gallons per minute

def rate_of_one_faucet : ℚ := rate_of_five_faucets / 5

def rate_of_ten_faucets : ℚ := rate_of_one_faucet * 10

def time_to_fill_60_gallons_minutes : ℚ := 60 / rate_of_ten_faucets

def time_to_fill_60_gallons_seconds : ℚ := time_to_fill_60_gallons_minutes * 60

-- The main theorem to prove
theorem fill_time_60_gallons_ten_faucets : time_to_fill_60_gallons_seconds = 96 := by
  sorry

end NUMINAMATH_GPT_fill_time_60_gallons_ten_faucets_l1741_174140


namespace NUMINAMATH_GPT_exist_non_zero_function_iff_sum_zero_l1741_174175

theorem exist_non_zero_function_iff_sum_zero (a b c : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x y z : ℝ, a * f (x * y + f z) + b * f (y * z + f x) + c * f (z * x + f y) = 0) ∧ ¬ (∀ x : ℝ, f x = 0)) ↔ (a + b + c = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_exist_non_zero_function_iff_sum_zero_l1741_174175


namespace NUMINAMATH_GPT_expression_value_l1741_174160

theorem expression_value (x y z : ℕ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  ( (1 / (y : ℚ)) + (1 / (z : ℚ))) / (1 / (x : ℚ)) = 35 / 12 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1741_174160


namespace NUMINAMATH_GPT_opposite_of_neg_2_l1741_174197

noncomputable def opposite (a : ℤ) : ℤ := 
  a * (-1)

theorem opposite_of_neg_2 : opposite (-2) = 2 := by
  -- definition of opposite
  unfold opposite
  -- calculation using the definition
  rfl

end NUMINAMATH_GPT_opposite_of_neg_2_l1741_174197


namespace NUMINAMATH_GPT_calc_expr_l1741_174173

theorem calc_expr :
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_calc_expr_l1741_174173


namespace NUMINAMATH_GPT_min_distance_sum_l1741_174155

theorem min_distance_sum
  (A B C D E P : ℝ)
  (h_collinear : B = A + 2 ∧ C = B + 2 ∧ D = C + 3 ∧ E = D + 4)
  (h_bisector : P = (A + E) / 2) :
  (A - P)^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (E - P)^2 = 77.25 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_sum_l1741_174155


namespace NUMINAMATH_GPT_Daria_money_l1741_174190

theorem Daria_money (num_tickets : ℕ) (price_per_ticket : ℕ) (amount_needed : ℕ) (h1 : num_tickets = 4) (h2 : price_per_ticket = 90) (h3 : amount_needed = 171) : 
  (num_tickets * price_per_ticket) - amount_needed = 189 := 
by 
  sorry

end NUMINAMATH_GPT_Daria_money_l1741_174190


namespace NUMINAMATH_GPT_exists_v_satisfying_equation_l1741_174164

noncomputable def custom_operation (v : ℝ) : ℝ :=
  v - (v / 3) + Real.sin v

theorem exists_v_satisfying_equation :
  ∃ v : ℝ, custom_operation (custom_operation v) = 24 := 
sorry

end NUMINAMATH_GPT_exists_v_satisfying_equation_l1741_174164


namespace NUMINAMATH_GPT_all_real_K_have_real_roots_l1741_174120

noncomputable def quadratic_discriminant (K : ℝ) : ℝ :=
  let a := K ^ 3
  let b := -(4 * K ^ 3 + 1)
  let c := 3 * K ^ 3
  b ^ 2 - 4 * a * c

theorem all_real_K_have_real_roots : ∀ K : ℝ, quadratic_discriminant K ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_all_real_K_have_real_roots_l1741_174120


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1741_174129

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) (h₃ : a > b) : a + a + b = 15 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1741_174129


namespace NUMINAMATH_GPT_hilltop_high_students_l1741_174137

theorem hilltop_high_students : 
  ∀ (n_sophomore n_freshman n_junior : ℕ), 
  (n_sophomore : ℚ) / n_freshman = 7 / 4 ∧ (n_junior : ℚ) / n_sophomore = 6 / 7 → 
  n_sophomore + n_freshman + n_junior = 17 :=
by
  sorry

end NUMINAMATH_GPT_hilltop_high_students_l1741_174137


namespace NUMINAMATH_GPT_two_lines_perpendicular_to_same_line_are_parallel_l1741_174101

/- Define what it means for two lines to be perpendicular -/
def perpendicular (l m : Line) : Prop :=
  -- A placeholder definition for perpendicularity, replace with the actual definition
  sorry

/- Define what it means for two lines to be parallel -/
def parallel (l m : Line) : Prop :=
  -- A placeholder definition for parallelism, replace with the actual definition
  sorry

/- Given: Two lines l1 and l2 that are perpendicular to the same line l3 -/
variables (l1 l2 l3 : Line)
variable (h1 : perpendicular l1 l3)
variable (h2 : perpendicular l2 l3)

/- Prove: l1 and l2 are parallel to each other -/
theorem two_lines_perpendicular_to_same_line_are_parallel :
  parallel l1 l2 :=
  sorry

end NUMINAMATH_GPT_two_lines_perpendicular_to_same_line_are_parallel_l1741_174101


namespace NUMINAMATH_GPT_process_can_continue_indefinitely_l1741_174147

noncomputable def P (x : ℝ) : ℝ := x^3 - x^2 - x - 1

-- Assume the existence of t > 1 such that P(t) = 0
axiom exists_t : ∃ t : ℝ, t > 1 ∧ P t = 0

def triangle_inequality_fails (a b c : ℝ) : Prop :=
  ¬(a + b > c ∧ b + c > a ∧ c + a > b)

def shorten (a b : ℝ) : ℝ := a + b

def can_continue_indefinitely (a b c : ℝ) : Prop :=
  ∀ t, t > 0 → ∀ a b c, triangle_inequality_fails a b c → 
  (triangle_inequality_fails (shorten b c - shorten a b) b c ∧
   triangle_inequality_fails a (shorten a c - shorten b c) c ∧
   triangle_inequality_fails a b (shorten a b - shorten b c))

theorem process_can_continue_indefinitely (a b c : ℝ) (h : triangle_inequality_fails a b c) :
  can_continue_indefinitely a b c :=
sorry

end NUMINAMATH_GPT_process_can_continue_indefinitely_l1741_174147


namespace NUMINAMATH_GPT_number_of_crocodiles_l1741_174144

theorem number_of_crocodiles
  (f : ℕ) -- number of frogs
  (c : ℕ) -- number of crocodiles
  (total_eyes : ℕ) -- total number of eyes
  (frog_eyes : ℕ) -- number of eyes per frog
  (croc_eyes : ℕ) -- number of eyes per crocodile
  (h_f : f = 20) -- condition: there are 20 frogs
  (h_total_eyes : total_eyes = 52) -- condition: total number of eyes is 52
  (h_frog_eyes : frog_eyes = 2) -- condition: each frog has 2 eyes
  (h_croc_eyes : croc_eyes = 2) -- condition: each crocodile has 2 eyes
  :
  c = 6 := -- proof goal: number of crocodiles is 6
by
  sorry

end NUMINAMATH_GPT_number_of_crocodiles_l1741_174144


namespace NUMINAMATH_GPT_tip_percentage_l1741_174149

variable (L : ℝ) (T : ℝ)
 
theorem tip_percentage (h : L = 60.50) (h1 : T = 72.6) :
  ((T - L) / L) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_l1741_174149


namespace NUMINAMATH_GPT_sum_of_rel_prime_ints_l1741_174186

theorem sum_of_rel_prime_ints (a b : ℕ) (h1 : a < 15) (h2 : b < 15) (h3 : a * b + a + b = 71)
    (h4 : Nat.gcd a b = 1) : a + b = 16 := by
  sorry

end NUMINAMATH_GPT_sum_of_rel_prime_ints_l1741_174186
