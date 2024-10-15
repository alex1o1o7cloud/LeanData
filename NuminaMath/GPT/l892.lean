import Mathlib

namespace NUMINAMATH_GPT_speed_of_woman_in_still_water_l892_89258

noncomputable def V_w : ℝ := 5
variable (V_s : ℝ)

-- Conditions:
def downstream_condition : Prop := (V_w + V_s) * 6 = 54
def upstream_condition : Prop := (V_w - V_s) * 6 = 6

theorem speed_of_woman_in_still_water 
    (h1 : downstream_condition V_s) 
    (h2 : upstream_condition V_s) : 
    V_w = 5 :=
by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_speed_of_woman_in_still_water_l892_89258


namespace NUMINAMATH_GPT_no_nat_triplet_square_l892_89202

theorem no_nat_triplet_square (m n k : ℕ) : ¬ (∃ a b c : ℕ, m^2 + n + k = a^2 ∧ n^2 + k + m = b^2 ∧ k^2 + m + n = c^2) :=
by sorry

end NUMINAMATH_GPT_no_nat_triplet_square_l892_89202


namespace NUMINAMATH_GPT_no_integer_solution_xyz_l892_89200

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end NUMINAMATH_GPT_no_integer_solution_xyz_l892_89200


namespace NUMINAMATH_GPT_point_M_in_second_quadrant_l892_89271

-- Given conditions
def m : ℤ := -2
def n : ℤ := 1

-- Definitions to identify the quadrants
def point_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

-- Problem statement to prove
theorem point_M_in_second_quadrant : 
  point_in_second_quadrant m n :=
by
  sorry

end NUMINAMATH_GPT_point_M_in_second_quadrant_l892_89271


namespace NUMINAMATH_GPT_shaded_percentage_correct_l892_89228

def total_squares : ℕ := 6 * 6
def shaded_squares : ℕ := 18
def percentage_shaded (total shaded : ℕ) : ℕ := (shaded * 100) / total

theorem shaded_percentage_correct : percentage_shaded total_squares shaded_squares = 50 := by
  sorry

end NUMINAMATH_GPT_shaded_percentage_correct_l892_89228


namespace NUMINAMATH_GPT_n_multiple_of_40_and_infinite_solutions_l892_89211

theorem n_multiple_of_40_and_infinite_solutions 
  (n : ℤ)
  (h1 : ∃ k₁ : ℤ, 2 * n + 1 = k₁^2)
  (h2 : ∃ k₂ : ℤ, 3 * n + 1 = k₂^2)
  : ∃ (m : ℤ), n = 40 * m ∧ ∃ (seq : ℕ → ℤ), 
    (∀ i : ℕ, ∃ k₁ k₂ : ℤ, (2 * (seq i) + 1 = k₁^2) ∧ (3 * (seq i) + 1 = k₂^2) ∧ 
     (i ≠ 0 → seq i ≠ seq (i - 1))) :=
by sorry

end NUMINAMATH_GPT_n_multiple_of_40_and_infinite_solutions_l892_89211


namespace NUMINAMATH_GPT_right_triangle_inequality_equality_condition_l892_89234

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b ≤ 5 * c :=
by 
  sorry

theorem equality_condition (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b = 5 * c ↔ a / b = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_inequality_equality_condition_l892_89234


namespace NUMINAMATH_GPT_combined_molecular_weight_l892_89256

theorem combined_molecular_weight :
  let CaO_molecular_weight := 56.08
  let CO2_molecular_weight := 44.01
  let HNO3_molecular_weight := 63.01
  let moles_CaO := 5
  let moles_CO2 := 3
  let moles_HNO3 := 2
  moles_CaO * CaO_molecular_weight + moles_CO2 * CO2_molecular_weight + moles_HNO3 * HNO3_molecular_weight = 538.45 :=
by sorry

end NUMINAMATH_GPT_combined_molecular_weight_l892_89256


namespace NUMINAMATH_GPT_number_of_boys_l892_89277

theorem number_of_boys (n : ℕ)
    (incorrect_avg_weight : ℝ)
    (misread_weight new_weight : ℝ)
    (correct_avg_weight : ℝ)
    (h1 : incorrect_avg_weight = 58.4)
    (h2 : misread_weight = 56)
    (h3 : new_weight = 66)
    (h4 : correct_avg_weight = 58.9)
    (h5 : n * correct_avg_weight = n * incorrect_avg_weight + (new_weight - misread_weight)) :
  n = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l892_89277


namespace NUMINAMATH_GPT_y_share_is_correct_l892_89294

noncomputable def share_of_y (a : ℝ) := 0.45 * a

theorem y_share_is_correct :
  ∃ a : ℝ, (1 * a + 0.45 * a + 0.30 * a = 245) ∧ (share_of_y a = 63) :=
by
  sorry

end NUMINAMATH_GPT_y_share_is_correct_l892_89294


namespace NUMINAMATH_GPT_gcd_12345_6789_l892_89224

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_12345_6789_l892_89224


namespace NUMINAMATH_GPT_tap_B_fill_time_l892_89206

theorem tap_B_fill_time :
  ∃ t : ℝ, 
    (3 * 10 + (12 / t) * 10 = 36) →
    t = 20 :=
by
  sorry

end NUMINAMATH_GPT_tap_B_fill_time_l892_89206


namespace NUMINAMATH_GPT_rubber_ball_radius_l892_89241

theorem rubber_ball_radius (r : ℝ) (radius_exposed_section : ℝ) (depth : ℝ) 
  (h1 : radius_exposed_section = 20) 
  (h2 : depth = 12) 
  (h3 : (r - depth)^2 + radius_exposed_section^2 = r^2) : 
  r = 22.67 :=
by
  sorry

end NUMINAMATH_GPT_rubber_ball_radius_l892_89241


namespace NUMINAMATH_GPT_contradiction_method_at_most_one_positive_l892_89237

theorem contradiction_method_at_most_one_positive :
  (∃ a b c : ℝ, (a > 0 → (b ≤ 0 ∧ c ≤ 0)) ∧ (b > 0 → (a ≤ 0 ∧ c ≤ 0)) ∧ (c > 0 → (a ≤ 0 ∧ b ≤ 0))) → 
  (¬(∃ a b c : ℝ, (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (a > 0 ∧ c > 0))) :=
by sorry

end NUMINAMATH_GPT_contradiction_method_at_most_one_positive_l892_89237


namespace NUMINAMATH_GPT_number_of_prize_orders_l892_89227

/-- At the end of a professional bowling tournament, the top 6 bowlers have a playoff.
    - #6 and #5 play a game. The loser receives the 6th prize and the winner plays #4.
    - The loser of the second game receives the 5th prize and the winner plays #3.
    - The loser of the third game receives the 4th prize and the winner plays #2.
    - The loser of the fourth game receives the 3rd prize and the winner plays #1.
    - The winner of the final game gets 1st prize and the loser gets 2nd prize.

    We want to determine the number of possible orders in which the bowlers can receive the prizes.
-/
theorem number_of_prize_orders : 2^5 = 32 := by
  sorry

end NUMINAMATH_GPT_number_of_prize_orders_l892_89227


namespace NUMINAMATH_GPT_functional_equation_solution_l892_89272

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) ↔ (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l892_89272


namespace NUMINAMATH_GPT_scientific_notation_113700_l892_89240

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_113700_l892_89240


namespace NUMINAMATH_GPT_kite_area_is_28_l892_89295

noncomputable def area_of_kite : ℝ :=
  let base_upper := 8
  let height_upper := 2
  let base_lower := 8
  let height_lower := 5
  let area_upper := (1 / 2 : ℝ) * base_upper * height_upper
  let area_lower := (1 / 2 : ℝ) * base_lower * height_lower
  area_upper + area_lower

theorem kite_area_is_28 :
  area_of_kite = 28 :=
by
  simp [area_of_kite]
  sorry

end NUMINAMATH_GPT_kite_area_is_28_l892_89295


namespace NUMINAMATH_GPT_find_ratio_l892_89269

theorem find_ratio (a b : ℝ) (h1 : ∀ x, ax^2 + bx + 2 < 0 ↔ (x < -1/2 ∨ x > 1/3)) :
  (a - b) / a = 5 / 6 := 
sorry

end NUMINAMATH_GPT_find_ratio_l892_89269


namespace NUMINAMATH_GPT_solve_x_of_det_8_l892_89229

variable (x : ℝ)

def matrix_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem solve_x_of_det_8
  (h : matrix_det (x + 1) (1 - x) (1 - x) (x + 1) = 8) : x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_x_of_det_8_l892_89229


namespace NUMINAMATH_GPT_josanna_minimum_test_score_l892_89264

def test_scores := [90, 80, 70, 60, 85]

def target_average_increase := 3

def current_average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def sixth_test_score_needed (scores : List ℕ) (increase : ℚ) : ℚ :=
  let current_avg := current_average scores
  let target_avg := current_avg + increase
  target_avg * (scores.length + 1) - scores.sum

theorem josanna_minimum_test_score :
  sixth_test_score_needed test_scores target_average_increase = 95 := sorry

end NUMINAMATH_GPT_josanna_minimum_test_score_l892_89264


namespace NUMINAMATH_GPT_inv_mod_35_l892_89278

theorem inv_mod_35 : ∃ x : ℕ, 5 * x ≡ 1 [MOD 35] :=
by
  use 29
  sorry

end NUMINAMATH_GPT_inv_mod_35_l892_89278


namespace NUMINAMATH_GPT_negation_of_universal_l892_89215

theorem negation_of_universal (P : ∀ x : ℝ, x^2 > 0) : ¬ ( ∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_universal_l892_89215


namespace NUMINAMATH_GPT_hyperbola_through_point_has_asymptotes_l892_89263

-- Definitions based on condition (1)
def hyperbola_asymptotes (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Definition of the problem
def hyperbola_eqn (x y : ℝ) : Prop := (x^2 / 5) - (y^2 / 20) = 1

-- Main statement including all conditions and proving the correct answer
theorem hyperbola_through_point_has_asymptotes :
  ∀ x y : ℝ, hyperbola_eqn x y ↔ (hyperbola_asymptotes x y ∨ (x, y) = (-3, 4)) :=
by
  -- The proof part is skipped with sorry
  sorry

end NUMINAMATH_GPT_hyperbola_through_point_has_asymptotes_l892_89263


namespace NUMINAMATH_GPT_james_selling_price_l892_89214

variable (P : ℝ)  -- Selling price per candy bar

theorem james_selling_price 
  (boxes_sold : ℕ)
  (candy_bars_per_box : ℕ) 
  (cost_price_per_candy_bar : ℝ)
  (total_profit : ℝ)
  (H1 : candy_bars_per_box = 10)
  (H2 : boxes_sold = 5)
  (H3 : cost_price_per_candy_bar = 1)
  (H4 : total_profit = 25)
  (profit_eq : boxes_sold * candy_bars_per_box * (P - cost_price_per_candy_bar) = total_profit)
  : P = 1.5 :=
by 
  sorry

end NUMINAMATH_GPT_james_selling_price_l892_89214


namespace NUMINAMATH_GPT_min_expression_value_l892_89261

theorem min_expression_value (x y z : ℝ) : ∃ x y z : ℝ, (xy - z)^2 + (x + y + z)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l892_89261


namespace NUMINAMATH_GPT_franks_earnings_l892_89289

/-- Frank's earnings problem statement -/
theorem franks_earnings 
  (total_hours : ℕ) (days : ℕ) (regular_pay_rate : ℝ) (overtime_pay_rate : ℝ)
  (hours_first_day : ℕ) (overtime_first_day : ℕ)
  (hours_second_day : ℕ) (hours_third_day : ℕ)
  (hours_fourth_day : ℕ) (overtime_fourth_day : ℕ)
  (regular_hours_per_day : ℕ) :
  total_hours = 32 →
  days = 4 →
  regular_pay_rate = 15 →
  overtime_pay_rate = 22.50 →
  hours_first_day = 12 →
  overtime_first_day = 4 →
  hours_second_day = 8 →
  hours_third_day = 8 →
  hours_fourth_day = 12 →
  overtime_fourth_day = 4 →
  regular_hours_per_day = 8 →
  (32 * regular_pay_rate + 8 * overtime_pay_rate) = 660 := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_franks_earnings_l892_89289


namespace NUMINAMATH_GPT_perpendicular_vectors_l892_89245

theorem perpendicular_vectors (k : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (0, 2)) 
  (hb : b = (Real.sqrt 3, 1)) 
  (h : (a.1 - k * b.1) * (k * a.1 + b.1) + (a.2 - k * b.2) * (k * a.2 + b.2) = 0) :
  k = -1 ∨ k = 1 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l892_89245


namespace NUMINAMATH_GPT_expectedAdjacentBlackPairs_l892_89262

noncomputable def numberOfBlackPairsInCircleDeck (totalCards blackCards redCards : ℕ) : ℚ := 
  let probBlackNext := (blackCards - 1) / (totalCards - 1)
  blackCards * probBlackNext

theorem expectedAdjacentBlackPairs (totalCards blackCards redCards expectedPairs : ℕ) : 
  totalCards = 52 → 
  blackCards = 30 → 
  redCards = 22 → 
  expectedPairs = 870 / 51 → 
  numberOfBlackPairsInCircleDeck totalCards blackCards redCards = expectedPairs :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_expectedAdjacentBlackPairs_l892_89262


namespace NUMINAMATH_GPT_wrapping_paper_area_correct_l892_89225

-- Define the length, width, and height of the box
variables (l w h : ℝ)

-- Define the function to calculate the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ := 2 * (l + w + h) ^ 2

-- Statement problem that we need to prove
theorem wrapping_paper_area_correct :
  wrapping_paper_area l w h = 2 * (l + w + h) ^ 2 := 
sorry

end NUMINAMATH_GPT_wrapping_paper_area_correct_l892_89225


namespace NUMINAMATH_GPT_sale_price_per_bearing_before_bulk_discount_l892_89223

-- Define the given conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := machines * ball_bearings_per_machine

def normal_cost_per_bearing : ℝ := 1
def total_normal_cost : ℝ := total_ball_bearings * normal_cost_per_bearing

def bulk_discount : ℝ := 0.20
def sale_savings : ℝ := 120

-- The theorem we need to prove
theorem sale_price_per_bearing_before_bulk_discount (P : ℝ) :
  total_normal_cost - (total_ball_bearings * P * (1 - bulk_discount)) = sale_savings → 
  P = 0.75 :=
by sorry

end NUMINAMATH_GPT_sale_price_per_bearing_before_bulk_discount_l892_89223


namespace NUMINAMATH_GPT_polygon_sides_eq_seven_l892_89282

theorem polygon_sides_eq_seven (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_seven_l892_89282


namespace NUMINAMATH_GPT_weeks_jake_buys_papayas_l892_89220

theorem weeks_jake_buys_papayas
  (jake_papayas : ℕ)
  (brother_papayas : ℕ)
  (father_papayas : ℕ)
  (total_papayas : ℕ)
  (h1 : jake_papayas = 3)
  (h2 : brother_papayas = 5)
  (h3 : father_papayas = 4)
  (h4 : total_papayas = 48) :
  (total_papayas / (jake_papayas + brother_papayas + father_papayas) = 4) :=
by
  sorry

end NUMINAMATH_GPT_weeks_jake_buys_papayas_l892_89220


namespace NUMINAMATH_GPT_positive_integer_solution_lcm_eq_sum_l892_89283

def is_lcm (x y z m : Nat) : Prop :=
  ∃ (d : Nat), x = d * (Nat.gcd y z) ∧ y = d * (Nat.gcd x z) ∧ z = d * (Nat.gcd x y) ∧
  x * y * z / Nat.gcd x (Nat.gcd y z) = m

theorem positive_integer_solution_lcm_eq_sum :
  ∀ (a b c : Nat), 0 < a → 0 < b → 0 < c → is_lcm a b c (a + b + c) → (a, b, c) = (a, 2 * a, 3 * a) := by
    sorry

end NUMINAMATH_GPT_positive_integer_solution_lcm_eq_sum_l892_89283


namespace NUMINAMATH_GPT_final_amount_is_75139_84_l892_89290

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : ℝ :=
  P * (1 + r/n)^(n * t)

theorem final_amount_is_75139_84 (P : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) :
  P = 64000 → r = 1/12 → t = 2 → n = 12 → compoundInterest P r t n = 75139.84 :=
by
  intros hP hr ht hn
  sorry

end NUMINAMATH_GPT_final_amount_is_75139_84_l892_89290


namespace NUMINAMATH_GPT_mike_practice_hours_l892_89248

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end NUMINAMATH_GPT_mike_practice_hours_l892_89248


namespace NUMINAMATH_GPT_prism_faces_l892_89274

-- Define a structure for a prism with a given number of edges
def is_prism (edges : ℕ) := 
  ∃ (n : ℕ), 3 * n = edges

-- Define the theorem to prove the number of faces in a prism given it has 21 edges
theorem prism_faces (h : is_prism 21) : ∃ (faces : ℕ), faces = 9 :=
by
  sorry

end NUMINAMATH_GPT_prism_faces_l892_89274


namespace NUMINAMATH_GPT_product_of_three_numbers_l892_89238

theorem product_of_three_numbers (a b c : ℚ) 
  (h₁ : a + b + c = 30)
  (h₂ : a = 6 * (b + c))
  (h₃ : b = 5 * c) : 
  a * b * c = 22500 / 343 := 
sorry

end NUMINAMATH_GPT_product_of_three_numbers_l892_89238


namespace NUMINAMATH_GPT_find_particular_number_l892_89207

variable (x : ℝ)

theorem find_particular_number (h : 0.46 + x = 0.72) : x = 0.26 :=
sorry

end NUMINAMATH_GPT_find_particular_number_l892_89207


namespace NUMINAMATH_GPT_valid_p_interval_l892_89235

theorem valid_p_interval :
  ∀ p, (∀ q, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 0 ≤ p ∧ p < 4 :=
sorry

end NUMINAMATH_GPT_valid_p_interval_l892_89235


namespace NUMINAMATH_GPT_trace_bags_weight_l892_89249

theorem trace_bags_weight :
  ∀ (g1 g2 t1 t2 t3 t4 t5 : ℕ),
    g1 = 3 →
    g2 = 7 →
    (g1 + g2) = (t1 + t2 + t3 + t4 + t5) →
    (t1 = t2 ∧ t2 = t3 ∧ t3 = t4 ∧ t4 = t5) →
    t1 = 2 :=
by
  intros g1 g2 t1 t2 t3 t4 t5 hg1 hg2 hsum hsame
  sorry

end NUMINAMATH_GPT_trace_bags_weight_l892_89249


namespace NUMINAMATH_GPT_root_of_quadratic_l892_89267

theorem root_of_quadratic {x a : ℝ} (h : x = 2 ∧ x^2 - x + a = 0) : a = -2 := 
by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_l892_89267


namespace NUMINAMATH_GPT_total_ribbon_length_l892_89204

theorem total_ribbon_length (a b c d e f g h i : ℝ) 
  (H : a + b + c + d + e + f + g + h + i = 62) : 
  1.5 * (a + b + c + d + e + f + g + h + i) = 93 :=
by
  sorry

end NUMINAMATH_GPT_total_ribbon_length_l892_89204


namespace NUMINAMATH_GPT_sum_of_fractions_l892_89273

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l892_89273


namespace NUMINAMATH_GPT_farmer_total_cows_l892_89286

theorem farmer_total_cows (cows : ℕ) 
  (h1 : 1 / 3 + 1 / 6 + 1 / 8 = 5 / 8) 
  (h2 : (3 / 8) * cows = 15) : 
  cows = 40 := by
  -- Given conditions:
  -- h1: The first three sons receive a total of 5/8 of the cows.
  -- h2: The fourth son receives 3/8 of the cows, which is 15 cows.
  sorry

end NUMINAMATH_GPT_farmer_total_cows_l892_89286


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l892_89216

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem common_ratio_of_geometric_sequence
  (a1 d : ℝ) (h1 : d ≠ 0)
  (h2 : (a_n a1 d 5) * (a_n a1 d 20) = (a_n a1 d 10) ^ 2) :
  (a_n a1 d 10) / (a_n a1 d 5) = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l892_89216


namespace NUMINAMATH_GPT_num_divisible_by_33_l892_89285

theorem num_divisible_by_33 : ∀ (x y : ℕ), 
  (0 ≤ x ∧ x ≤ 9) → (0 ≤ y ∧ y ≤ 9) →
  (19 + x + y) % 3 = 0 →
  (x - y + 1) % 11 = 0 →
  ∃! (n : ℕ), (20070002008 * 100 + x * 10 + y) = n ∧ n % 33 = 0 :=
by
  intros x y hx hy h3 h11
  sorry

end NUMINAMATH_GPT_num_divisible_by_33_l892_89285


namespace NUMINAMATH_GPT_combined_collectors_edition_dolls_l892_89293

-- Definitions based on given conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def luna_dolls : ℕ := ivy_dolls - 10

-- Additional constraints based on the problem statement
def total_dolls : ℕ := dina_dolls + ivy_dolls + luna_dolls
def ivy_collectors_edition_dolls : ℕ := 2/3 * ivy_dolls
def luna_collectors_edition_dolls : ℕ := 1/2 * luna_dolls

-- Proof statement
theorem combined_collectors_edition_dolls :
  ivy_collectors_edition_dolls + luna_collectors_edition_dolls = 30 :=
sorry

end NUMINAMATH_GPT_combined_collectors_edition_dolls_l892_89293


namespace NUMINAMATH_GPT_four_digit_number_divisibility_l892_89233

theorem four_digit_number_divisibility 
  (E V I L : ℕ) 
  (hE : 0 ≤ E ∧ E < 10) 
  (hV : 0 ≤ V ∧ V < 10) 
  (hI : 0 ≤ I ∧ I < 10) 
  (hL : 0 ≤ L ∧ L < 10)
  (h1 : (1000 * E + 100 * V + 10 * I + L) % 73 = 0) 
  (h2 : (1000 * V + 100 * I + 10 * L + E) % 74 = 0)
  : 1000 * L + 100 * I + 10 * V + E = 5499 := 
  sorry

end NUMINAMATH_GPT_four_digit_number_divisibility_l892_89233


namespace NUMINAMATH_GPT_selling_price_l892_89239

theorem selling_price (cost_price : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 2400 ∧ profit_percent = 6 → selling_price = 2544 := by
  sorry

end NUMINAMATH_GPT_selling_price_l892_89239


namespace NUMINAMATH_GPT_find_greater_number_l892_89247

theorem find_greater_number (a b : ℕ) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 :=
sorry

end NUMINAMATH_GPT_find_greater_number_l892_89247


namespace NUMINAMATH_GPT_distance_is_twenty_cm_l892_89244

noncomputable def distance_between_pictures_and_board (picture_width: ℕ) (board_width_m: ℕ) (board_width_cm: ℕ) (number_of_pictures: ℕ) : ℕ :=
  let board_total_width := board_width_m * 100 + board_width_cm
  let total_pictures_width := number_of_pictures * picture_width
  let total_distance := board_total_width - total_pictures_width
  let total_gaps := number_of_pictures + 1
  total_distance / total_gaps

theorem distance_is_twenty_cm :
  distance_between_pictures_and_board 30 3 20 6 = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_is_twenty_cm_l892_89244


namespace NUMINAMATH_GPT_nonagon_area_l892_89284

noncomputable def area_of_nonagon (r : ℝ) : ℝ :=
  (9 / 2) * r^2 * Real.sin (Real.pi * 40 / 180)

theorem nonagon_area (r : ℝ) : 
  area_of_nonagon r = 2.891 * r^2 :=
by
  sorry

end NUMINAMATH_GPT_nonagon_area_l892_89284


namespace NUMINAMATH_GPT_min_abs_diff_is_11_l892_89253

noncomputable def min_abs_diff (k l : ℕ) : ℤ := abs (36^k - 5^l)

theorem min_abs_diff_is_11 :
  ∃ k l : ℕ, min_abs_diff k l = 11 :=
by
  sorry

end NUMINAMATH_GPT_min_abs_diff_is_11_l892_89253


namespace NUMINAMATH_GPT_range_of_alpha_minus_beta_l892_89270

theorem range_of_alpha_minus_beta (α β : Real) (h₁ : -180 < α) (h₂ : α < β) (h₃ : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_alpha_minus_beta_l892_89270


namespace NUMINAMATH_GPT_trees_to_plant_l892_89243

def road_length : ℕ := 156
def interval : ℕ := 6
def trees_needed (road_length interval : ℕ) := road_length / interval + 1

theorem trees_to_plant : trees_needed road_length interval = 27 := by
  sorry

end NUMINAMATH_GPT_trees_to_plant_l892_89243


namespace NUMINAMATH_GPT_curve_not_parabola_l892_89212

theorem curve_not_parabola (k : ℝ) : ¬ ∃ a b c t : ℝ, a * t^2 + b * t + c = x^2 + k * y^2 - 1 := sorry

end NUMINAMATH_GPT_curve_not_parabola_l892_89212


namespace NUMINAMATH_GPT_part1_part2_l892_89221

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end NUMINAMATH_GPT_part1_part2_l892_89221


namespace NUMINAMATH_GPT_A_inter_complement_RB_eq_l892_89299

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

def complement_RB : Set ℝ := {x | x ≥ 1}

theorem A_inter_complement_RB_eq : A ∩ complement_RB = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_A_inter_complement_RB_eq_l892_89299


namespace NUMINAMATH_GPT_traveling_zoo_l892_89219

theorem traveling_zoo (x y : ℕ) (h1 : x + y = 36) (h2 : 4 * x + 6 * y = 100) : x = 14 ∧ y = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_traveling_zoo_l892_89219


namespace NUMINAMATH_GPT_nth_term_arithmetic_seq_l892_89259

theorem nth_term_arithmetic_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : ∀ n : ℕ, ∃ m : ℝ, a (n + 1) = a n + m)
  (h_d_neg : d < 0)
  (h_condition1 : a 2 * a 4 = 12)
  (h_condition2 : a 2 + a 4 = 8):
  ∀ n : ℕ, a n = -2 * n + 10 :=
by
  sorry

end NUMINAMATH_GPT_nth_term_arithmetic_seq_l892_89259


namespace NUMINAMATH_GPT_geometric_sequence_min_l892_89232

theorem geometric_sequence_min (a : ℕ → ℝ) (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_condition : 2 * (a 4) + (a 3) - 2 * (a 2) - (a 1) = 8)
  (h_geometric : ∀ n, a (n+1) = a n * q) :
  ∃ min_val, min_val = 12 * Real.sqrt 3 ∧ min_val = 2 * (a 5) + (a 4) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_min_l892_89232


namespace NUMINAMATH_GPT_negation_of_universal_prop_l892_89222

theorem negation_of_universal_prop:
  (¬ (∀ x : ℝ, x ^ 3 - x ≥ 0)) ↔ (∃ x : ℝ, x ^ 3 - x < 0) := 
by 
sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l892_89222


namespace NUMINAMATH_GPT_apple_production_total_l892_89246

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end NUMINAMATH_GPT_apple_production_total_l892_89246


namespace NUMINAMATH_GPT_geo_seq_4th_term_l892_89257

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end NUMINAMATH_GPT_geo_seq_4th_term_l892_89257


namespace NUMINAMATH_GPT_labourer_income_l892_89252

noncomputable def monthly_income : ℤ := 75

theorem labourer_income:
  ∃ (I D : ℤ),
  (80 * 6 = 480) ∧
  (I * 6 - D + (I * 4) = 480 + 240 + D + 30) →
  I = monthly_income :=
by
  sorry

end NUMINAMATH_GPT_labourer_income_l892_89252


namespace NUMINAMATH_GPT_problem_I_number_of_zeros_problem_II_inequality_l892_89266

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x * Real.exp 1 - 1

theorem problem_I_number_of_zeros : 
  ∃! (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
sorry

theorem problem_II_inequality (a : ℝ) (h_a : a ≤ 0) (x : ℝ) (h_x : x ≥ 1) : 
  f x ≥ a * Real.log x - 1 := 
sorry

end NUMINAMATH_GPT_problem_I_number_of_zeros_problem_II_inequality_l892_89266


namespace NUMINAMATH_GPT_count_more_blue_l892_89218

-- Definitions derived from the provided conditions
variables (total_people more_green both neither : ℕ)
variable (more_blue : ℕ)

-- Condition 1: There are 150 people in total
axiom total_people_def : total_people = 150

-- Condition 2: 90 people believe that teal is "more green"
axiom more_green_def : more_green = 90

-- Condition 3: 35 people believe it is both "more green" and "more blue"
axiom both_def : both = 35

-- Condition 4: 25 people think that teal is neither "more green" nor "more blue"
axiom neither_def : neither = 25


-- Theorem statement
theorem count_more_blue (total_people more_green both neither more_blue : ℕ) 
  (total_people_def : total_people = 150)
  (more_green_def : more_green = 90)
  (both_def : both = 35)
  (neither_def : neither = 25) :
  more_blue = 70 :=
by
  sorry

end NUMINAMATH_GPT_count_more_blue_l892_89218


namespace NUMINAMATH_GPT_problem_lean_l892_89279

theorem problem_lean (k b : ℤ) : 
  ∃ n : ℤ, n = 25 ∧ n^2 = (k + 1)^4 - k^4 ∧ 3 * n + 100 = b^2 :=
sorry

end NUMINAMATH_GPT_problem_lean_l892_89279


namespace NUMINAMATH_GPT_darnell_saves_money_l892_89297

-- Define conditions
def current_plan_cost := 12
def text_cost := 1
def call_cost := 3
def texts_per_month := 60
def calls_per_month := 60
def texts_per_unit := 30
def calls_per_unit := 20

-- Define the costs for the alternative plan
def alternative_texting_cost := (text_cost * (texts_per_month / texts_per_unit))
def alternative_calling_cost := (call_cost * (calls_per_month / calls_per_unit))
def alternative_plan_cost := alternative_texting_cost + alternative_calling_cost

-- Define the problem to prove
theorem darnell_saves_money :
  current_plan_cost - alternative_plan_cost = 1 :=
by
  sorry

end NUMINAMATH_GPT_darnell_saves_money_l892_89297


namespace NUMINAMATH_GPT_electricity_price_increase_percentage_l892_89217

noncomputable def old_power_kW : ℝ := 0.8
noncomputable def additional_power_percent : ℝ := 50 / 100
noncomputable def old_price_per_kWh : ℝ := 0.12
noncomputable def cost_for_50_hours : ℝ := 9
noncomputable def total_hours : ℝ := 50
noncomputable def energy_consumed := old_power_kW * total_hours

theorem electricity_price_increase_percentage :
  ∃ P : ℝ, 
    (energy_consumed * P = cost_for_50_hours) ∧
    ((P - old_price_per_kWh) / old_price_per_kWh) * 100 = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_electricity_price_increase_percentage_l892_89217


namespace NUMINAMATH_GPT_calculateTotalProfit_l892_89281

-- Defining the initial investments and changes
def initialInvestmentA : ℕ := 5000
def initialInvestmentB : ℕ := 8000
def initialInvestmentC : ℕ := 9000

def additionalInvestmentA : ℕ := 2000
def withdrawnInvestmentB : ℕ := 1000
def additionalInvestmentC : ℕ := 3000

-- Defining the durations
def months1 : ℕ := 4
def months2 : ℕ := 8
def months3 : ℕ := 6

-- C's share of the profit
def shareOfC : ℕ := 45000

-- Total profit to be proved
def totalProfit : ℕ := 103571

-- Lean 4 theorem statement
theorem calculateTotalProfit :
  let ratioA := (initialInvestmentA * months1) + ((initialInvestmentA + additionalInvestmentA) * months2)
  let ratioB := (initialInvestmentB * months1) + ((initialInvestmentB - withdrawnInvestmentB) * months2)
  let ratioC := (initialInvestmentC * months3) + ((initialInvestmentC + additionalInvestmentC) * months3)
  let totalRatio := ratioA + ratioB + ratioC
  (shareOfC / ratioC : ℚ) = (totalProfit / totalRatio : ℚ) :=
sorry

end NUMINAMATH_GPT_calculateTotalProfit_l892_89281


namespace NUMINAMATH_GPT_reconstruct_quadrilateral_l892_89226

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A A'' B'' C'' D'' : V)

def trisect_segment (P Q R : V) : Prop :=
  Q = (1 / 3 : ℝ) • P + (2 / 3 : ℝ) • R

theorem reconstruct_quadrilateral
  (hB : trisect_segment A B A'')
  (hC : trisect_segment B C B'')
  (hD : trisect_segment C D C'')
  (hA : trisect_segment D A D'') :
  A = (2 / 26) • A'' + (6 / 26) • B'' + (6 / 26) • C'' + (12 / 26) • D'' :=
sorry

end NUMINAMATH_GPT_reconstruct_quadrilateral_l892_89226


namespace NUMINAMATH_GPT_hawks_points_l892_89210

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_points : total_points touchdowns points_per_touchdown = 21 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_hawks_points_l892_89210


namespace NUMINAMATH_GPT_find_meeting_time_l892_89280

-- Define the context and the problem parameters
def lisa_speed : ℝ := 9  -- Lisa's speed in mph
def adam_speed : ℝ := 7  -- Adam's speed in mph
def initial_distance : ℝ := 6  -- Initial distance in miles

-- The time in minutes for Lisa to meet Adam
theorem find_meeting_time : (initial_distance / (lisa_speed + adam_speed)) * 60 = 22.5 := by
  -- The proof is omitted for this statement
  sorry

end NUMINAMATH_GPT_find_meeting_time_l892_89280


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l892_89265

theorem algebraic_expression_evaluation (a b : ℝ) (h : -2 * a + 3 * b + 8 = 18) : 9 * b - 6 * a + 2 = 32 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l892_89265


namespace NUMINAMATH_GPT_max_value_quadratic_l892_89209

theorem max_value_quadratic (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : x * (1 - x) ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_quadratic_l892_89209


namespace NUMINAMATH_GPT_simplify_expression_l892_89275

variable (x y : ℤ)

theorem simplify_expression : 
  (15 * x + 45 * y) + (7 * x + 18 * y) - (6 * x + 35 * y) = 16 * x + 28 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l892_89275


namespace NUMINAMATH_GPT_rotation_transforms_and_sums_l892_89203

theorem rotation_transforms_and_sums 
    (D E F D' E' F' : (ℝ × ℝ))
    (hD : D = (0, 0)) (hE : E = (0, 20)) (hF : F = (30, 0)) 
    (hD' : D' = (-26, 23)) (hE' : E' = (-46, 23)) (hF' : F' = (-26, -7))
    (n : ℝ) (x y : ℝ)
    (rotation_condition : 0 < n ∧ n < 180)
    (angle_condition : n = 90) :
    n + x + y = 60.5 :=
by
  have hx : x = -49 := sorry
  have hy : y = 19.5 := sorry
  have hn : n = 90 := sorry
  sorry

end NUMINAMATH_GPT_rotation_transforms_and_sums_l892_89203


namespace NUMINAMATH_GPT_solution_is_three_l892_89255

def equation (x : ℝ) : Prop := 
  Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2

theorem solution_is_three : equation 3 :=
by sorry

end NUMINAMATH_GPT_solution_is_three_l892_89255


namespace NUMINAMATH_GPT_problem_b_l892_89201

theorem problem_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) : a + b ≥ 2 :=
sorry

end NUMINAMATH_GPT_problem_b_l892_89201


namespace NUMINAMATH_GPT_solution_set_l892_89236

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_deriv : ∀ x, deriv f x = f' x
axiom f_at_3 : f 3 = 1
axiom inequality : ∀ x, 3 * f x + x * f' x > 1

-- Goal to prove
theorem solution_set :
  {x : ℝ | (x - 2017) ^ 3 * f (x - 2017) - 27 > 0} = {x | 2020 < x} :=
  sorry

end NUMINAMATH_GPT_solution_set_l892_89236


namespace NUMINAMATH_GPT_total_price_is_correct_l892_89230

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end NUMINAMATH_GPT_total_price_is_correct_l892_89230


namespace NUMINAMATH_GPT_complement_union_sets_l892_89254

open Set

theorem complement_union_sets :
  ∀ (U A B : Set ℕ), (U = {1, 2, 3, 4}) → (A = {2, 3}) → (B = {3, 4}) → (U \ (A ∪ B) = {1}) :=
by
  intros U A B hU hA hB
  rw [hU, hA, hB]
  simp 
  sorry

end NUMINAMATH_GPT_complement_union_sets_l892_89254


namespace NUMINAMATH_GPT_remainder_2abc_mod_7_l892_89296

theorem remainder_2abc_mod_7
  (a b c : ℕ)
  (h₀ : 2 * a + 3 * b + c ≡ 1 [MOD 7])
  (h₁ : 3 * a + b + 2 * c ≡ 2 [MOD 7])
  (h₂ : a + b + c ≡ 3 [MOD 7])
  (ha : a < 7)
  (hb : b < 7)
  (hc : c < 7) :
  2 * a * b * c ≡ 0 [MOD 7] :=
sorry

end NUMINAMATH_GPT_remainder_2abc_mod_7_l892_89296


namespace NUMINAMATH_GPT_train_length_is_499_96_l892_89242

-- Define the conditions
def speed_train_kmh : ℕ := 75   -- Speed of the train in km/h
def speed_man_kmh : ℕ := 3     -- Speed of the man in km/h
def time_cross_s : ℝ := 24.998 -- Time taken for the train to cross the man in seconds

-- Define the conversion factors
def km_to_m : ℕ := 1000        -- Conversion from kilometers to meters
def hr_to_s : ℕ := 3600        -- Conversion from hours to seconds

-- Define relative speed in m/s
def relative_speed_ms : ℕ := (speed_train_kmh - speed_man_kmh) * km_to_m / hr_to_s

-- Prove the length of the train in meters
def length_of_train : ℝ := relative_speed_ms * time_cross_s

theorem train_length_is_499_96 : length_of_train = 499.96 := sorry

end NUMINAMATH_GPT_train_length_is_499_96_l892_89242


namespace NUMINAMATH_GPT_length_of_arc_l892_89276

def radius : ℝ := 5
def area_of_sector : ℝ := 10
def expected_length_of_arc : ℝ := 4

theorem length_of_arc (r : ℝ) (A : ℝ) (l : ℝ) (h₁ : r = radius) (h₂ : A = area_of_sector) : l = expected_length_of_arc := by
  sorry

end NUMINAMATH_GPT_length_of_arc_l892_89276


namespace NUMINAMATH_GPT_algebraic_expression_is_product_l892_89231

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_is_product_l892_89231


namespace NUMINAMATH_GPT_linda_age_difference_l892_89260

/-- 
Linda is some more than 2 times the age of Jane.
In five years, the sum of their ages will be 28.
Linda's age at present is 13.
Prove that Linda's age is 3 years more than 2 times Jane's age.
-/
theorem linda_age_difference {L J : ℕ} (h1 : L = 13)
  (h2 : (L + 5) + (J + 5) = 28) : L - 2 * J = 3 :=
by sorry

end NUMINAMATH_GPT_linda_age_difference_l892_89260


namespace NUMINAMATH_GPT_total_feet_is_140_l892_89287

def total_heads : ℕ := 48
def number_of_hens : ℕ := 26
def number_of_cows : ℕ := total_heads - number_of_hens
def feet_per_hen : ℕ := 2
def feet_per_cow : ℕ := 4

theorem total_feet_is_140 : ((number_of_hens * feet_per_hen) + (number_of_cows * feet_per_cow)) = 140 := by
  sorry

end NUMINAMATH_GPT_total_feet_is_140_l892_89287


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l892_89292

theorem solve_system_of_inequalities (x : ℝ) : 
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2) → -2 < x ∧ x < -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_of_inequalities_l892_89292


namespace NUMINAMATH_GPT_statement_l892_89208

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Condition 2: f(x-2) = -f(x) for all x
def satisfies_periodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x - 2) = -f x

-- Condition 3: f is decreasing on [0, 2]
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- The proof statement
theorem statement (h1 : is_odd_function f) (h2 : satisfies_periodicity f) (h3 : is_decreasing_on f 0 2) :
  f 5 < f 4 ∧ f 4 < f 3 :=
sorry

end NUMINAMATH_GPT_statement_l892_89208


namespace NUMINAMATH_GPT_hydrogen_moles_l892_89268

-- Define the balanced chemical reaction as a relation between moles
def balanced_reaction (NaH H₂O NaOH H₂ : ℕ) : Prop :=
  NaH = NaOH ∧ H₂ = NaOH ∧ NaH = H₂

-- Given conditions
def given_conditions (NaH H₂O : ℕ) : Prop :=
  NaH = 2 ∧ H₂O = 2

-- Problem statement to prove
theorem hydrogen_moles (NaH H₂O NaOH H₂ : ℕ)
  (h₁ : balanced_reaction NaH H₂O NaOH H₂)
  (h₂ : given_conditions NaH H₂O) :
  H₂ = 2 :=
by sorry

end NUMINAMATH_GPT_hydrogen_moles_l892_89268


namespace NUMINAMATH_GPT_SamBalloonsCount_l892_89251

-- Define the conditions
def FredBalloons : ℕ := 10
def DanBalloons : ℕ := 16
def TotalBalloons : ℕ := 72

-- Define the function to calculate Sam's balloons and the main theorem to prove
def SamBalloons := TotalBalloons - (FredBalloons + DanBalloons)

theorem SamBalloonsCount : SamBalloons = 46 := by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_SamBalloonsCount_l892_89251


namespace NUMINAMATH_GPT_linear_eq_with_one_variable_is_B_l892_89250

-- Define the equations
def eqA (x y : ℝ) : Prop := 2 * x = 3 * y
def eqB (x : ℝ) : Prop := 7 * x + 5 = 6 * (x - 1)
def eqC (x : ℝ) : Prop := x^2 + (1 / 2) * (x - 1) = 1
def eqD (x : ℝ) : Prop := (1 / x) - 2 = x

-- State the problem
theorem linear_eq_with_one_variable_is_B :
  ∃ x : ℝ, ¬ (∃ y : ℝ, eqA x y) ∧ eqB x ∧ ¬ eqC x ∧ ¬ eqD x :=
by {
  -- mathematical content goes here
  sorry
}

end NUMINAMATH_GPT_linear_eq_with_one_variable_is_B_l892_89250


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l892_89291

-- Definitions used in the conditions
variable (a b : ℝ)

-- The Lean 4 theorem statement for the proof problem
theorem necessary_but_not_sufficient : (a > b - 1) ∧ ¬ (a > b) ↔ a > b := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l892_89291


namespace NUMINAMATH_GPT_total_paint_area_eq_1060_l892_89288

/-- Define the dimensions of the stable and chimney -/
def stable_width := 12
def stable_length := 15
def stable_height := 6
def chimney_width := 2
def chimney_length := 2
def chimney_height := 2

/-- Define the area to be painted computation -/

def wall_area (width length height : ℕ) : ℕ :=
  (width * height * 2) * 2 + (length * height * 2) * 2

def roof_area (width length : ℕ) : ℕ :=
  width * length

def ceiling_area (width length : ℕ) : ℕ :=
  width * length

def chimney_area (width length height : ℕ) : ℕ :=
  (4 * (width * height)) + (width * length)

def total_paint_area : ℕ :=
  wall_area stable_width stable_length stable_height +
  roof_area stable_width stable_length +
  ceiling_area stable_width stable_length +
  chimney_area chimney_width chimney_length chimney_height

/-- Goal: Prove that the total paint area is 1060 sq. yd -/
theorem total_paint_area_eq_1060 : total_paint_area = 1060 := by
  sorry

end NUMINAMATH_GPT_total_paint_area_eq_1060_l892_89288


namespace NUMINAMATH_GPT_complement_union_A_B_l892_89298

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def U : Set ℝ := A ∪ B
def R : Set ℝ := univ

theorem complement_union_A_B : (R \ U) = {x | -2 < x ∧ x ≤ -1} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_l892_89298


namespace NUMINAMATH_GPT_multiple_properties_l892_89213

variables (a b : ℤ)

-- Definitions of the conditions
def is_multiple_of_4 (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k
def is_multiple_of_8 (x : ℤ) : Prop := ∃ k : ℤ, x = 8 * k

-- Problem statement
theorem multiple_properties (h1 : is_multiple_of_4 a) (h2 : is_multiple_of_8 b) :
  is_multiple_of_4 b ∧ is_multiple_of_4 (a + b) ∧ (∃ k : ℤ, a + b = 2 * k) :=
by
  sorry

end NUMINAMATH_GPT_multiple_properties_l892_89213


namespace NUMINAMATH_GPT_sofa_love_seat_cost_l892_89205

theorem sofa_love_seat_cost (love_seat_cost : ℕ) (sofa_cost : ℕ) 
    (h₁ : love_seat_cost = 148) (h₂ : sofa_cost = 2 * love_seat_cost) :
    love_seat_cost + sofa_cost = 444 := 
by
  sorry

end NUMINAMATH_GPT_sofa_love_seat_cost_l892_89205
