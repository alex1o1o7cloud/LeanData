import Mathlib

namespace NUMINAMATH_GPT_prove_fraction_l52_5270

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def forms_arithmetic_sequence (x y z : ℝ) : Prop :=
2 * y = x + z

theorem prove_fraction
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h_ar : forms_arithmetic_sequence (a 1 + 2 * b 1) (a 3 + 4 * b 3) (a 5 + 8 * b 5)) :
  (b 3 * b 7) / (b 4 ^ 2) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_prove_fraction_l52_5270


namespace NUMINAMATH_GPT_age_of_B_l52_5293

theorem age_of_B (A B C : ℕ) (h1 : A + B + C = 90)
                  (h2 : (A - 10) = (B - 10) / 2)
                  (h3 : (B - 10) / 2 = (C - 10) / 3) : 
                  B = 30 :=
by sorry

end NUMINAMATH_GPT_age_of_B_l52_5293


namespace NUMINAMATH_GPT_sam_hourly_rate_l52_5211

theorem sam_hourly_rate
  (first_month_earnings : ℕ)
  (second_month_earnings : ℕ)
  (total_hours : ℕ)
  (h1 : first_month_earnings = 200)
  (h2 : second_month_earnings = first_month_earnings + 150)
  (h3 : total_hours = 55) :
  (first_month_earnings + second_month_earnings) / total_hours = 10 := 
  by
  sorry

end NUMINAMATH_GPT_sam_hourly_rate_l52_5211


namespace NUMINAMATH_GPT_Jori_water_left_l52_5238

theorem Jori_water_left (a b : ℚ) (h1 : a = 7/2) (h2 : b = 7/4) : a - b = 7/4 := by
  sorry

end NUMINAMATH_GPT_Jori_water_left_l52_5238


namespace NUMINAMATH_GPT_water_remaining_l52_5290

theorem water_remaining (initial_water : ℕ) (evap_rate : ℕ) (days : ℕ) : 
  initial_water = 500 → evap_rate = 1 → days = 50 → 
  initial_water - evap_rate * days = 450 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_water_remaining_l52_5290


namespace NUMINAMATH_GPT_sufficient_material_for_box_l52_5296

theorem sufficient_material_for_box :
  ∃ (l w h : ℕ), l * w * h ≥ 1995 ∧ 2 * (l * w + w * h + h * l) ≤ 958 :=
  sorry

end NUMINAMATH_GPT_sufficient_material_for_box_l52_5296


namespace NUMINAMATH_GPT_items_sold_each_house_l52_5254

-- Define the conditions
def visits_day_one : ℕ := 20
def visits_day_two : ℕ := 2 * visits_day_one
def sale_percentage_day_two : ℝ := 0.8
def total_sales : ℕ := 104

-- Define the number of items sold at each house
variable (x : ℕ)

-- Define the main Lean 4 statement for the proof
theorem items_sold_each_house (h1 : 20 * x + 32 * x = 104) : x = 2 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_items_sold_each_house_l52_5254


namespace NUMINAMATH_GPT_shaded_area_concentric_circles_l52_5230

theorem shaded_area_concentric_circles (R : ℝ) (r : ℝ) (hR : π * R^2 = 100 * π) (hr : r = R / 2) :
  (1 / 2) * π * R^2 + (1 / 2) * π * r^2 = 62.5 * π :=
by
  -- Given conditions
  have R10 : R = 10 := sorry  -- Derived from hR
  have r5 : r = 5 := sorry    -- Derived from hr and R10
  -- Proof steps likely skipped
  sorry

end NUMINAMATH_GPT_shaded_area_concentric_circles_l52_5230


namespace NUMINAMATH_GPT_gear_q_revolutions_per_minute_is_40_l52_5207

-- Definitions corresponding to conditions
def gear_p_revolutions_per_minute : ℕ := 10
def gear_q_revolutions_per_minute (r : ℕ) : Prop :=
  ∃ (r : ℕ), (r * 20 / 60) - (10 * 20 / 60) = 10

-- Statement we need to prove
theorem gear_q_revolutions_per_minute_is_40 :
  gear_q_revolutions_per_minute 40 :=
sorry

end NUMINAMATH_GPT_gear_q_revolutions_per_minute_is_40_l52_5207


namespace NUMINAMATH_GPT_find_parenthesis_value_l52_5202

theorem find_parenthesis_value (x : ℝ) (h : x * (-2/3) = 2) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_parenthesis_value_l52_5202


namespace NUMINAMATH_GPT_quarter_more_than_whole_l52_5243

theorem quarter_more_than_whole (x : ℝ) (h : x / 4 = 9 + x) : x = -12 :=
by
  sorry

end NUMINAMATH_GPT_quarter_more_than_whole_l52_5243


namespace NUMINAMATH_GPT_tile_arrangement_probability_l52_5283

theorem tile_arrangement_probability :
  let X := 5
  let O := 4
  let total_tiles := 9
  (1 : ℚ) / (Nat.choose total_tiles X) = 1 / 126 :=
by
  sorry

end NUMINAMATH_GPT_tile_arrangement_probability_l52_5283


namespace NUMINAMATH_GPT_largest_integer_divides_difference_l52_5213

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end NUMINAMATH_GPT_largest_integer_divides_difference_l52_5213


namespace NUMINAMATH_GPT_first_number_eq_l52_5237

theorem first_number_eq (x y : ℝ) (h1 : x * 120 = 346) (h2 : y * 240 = 346) : x = 346 / 120 :=
by
  -- The final proof will be inserted here
  sorry

end NUMINAMATH_GPT_first_number_eq_l52_5237


namespace NUMINAMATH_GPT_horizontal_distance_travel_l52_5234

noncomputable def radius : ℝ := 2
noncomputable def angle_degrees : ℝ := 30
noncomputable def angle_radians : ℝ := angle_degrees * (Real.pi / 180)
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def cos_theta : ℝ := Real.cos angle_radians
noncomputable def horizontal_distance (r : ℝ) (θ : ℝ) : ℝ := (circumference r) * (Real.cos θ)

theorem horizontal_distance_travel (r : ℝ) (θ : ℝ) (h_radius : r = 2) (h_angle : θ = angle_radians) :
  horizontal_distance r θ = 2 * Real.pi * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_horizontal_distance_travel_l52_5234


namespace NUMINAMATH_GPT_remainder_444_power_444_mod_13_l52_5284

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_444_power_444_mod_13_l52_5284


namespace NUMINAMATH_GPT_tangency_point_exists_l52_5240

theorem tangency_point_exists :
  ∃ (x y : ℝ), y = x^2 + 18 * x + 47 ∧ x = y^2 + 36 * y + 323 ∧ x = -17 / 2 ∧ y = -35 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tangency_point_exists_l52_5240


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l52_5278

variables {a b : ℝ}

theorem sufficient_but_not_necessary (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l52_5278


namespace NUMINAMATH_GPT_quadratic_common_root_distinct_real_numbers_l52_5244

theorem quadratic_common_root_distinct_real_numbers:
  ∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0) ∧
  (∃ y, y^2 + a * y + b = 0 ∧ y^2 + b * y + c = 0) ∧
  (∃ z, z^2 + b * z + c = 0 ∧ z^2 + c * z + a = 0) →
  a^2 + b^2 + c^2 = 6 :=
by
  intros a b c h_distinct h_common_root
  sorry

end NUMINAMATH_GPT_quadratic_common_root_distinct_real_numbers_l52_5244


namespace NUMINAMATH_GPT_harry_morning_routine_time_l52_5255

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end NUMINAMATH_GPT_harry_morning_routine_time_l52_5255


namespace NUMINAMATH_GPT_possible_values_of_expression_l52_5297

theorem possible_values_of_expression (x : ℝ) (h : 3 ≤ x ∧ x ≤ 4) : 
  40 ≤ x^2 + 7 * x + 10 ∧ x^2 + 7 * x + 10 ≤ 54 := 
sorry

end NUMINAMATH_GPT_possible_values_of_expression_l52_5297


namespace NUMINAMATH_GPT_total_trophies_after_five_years_l52_5209

theorem total_trophies_after_five_years (michael_current_trophies : ℕ) (michael_increase : ℕ) (jack_multiplier : ℕ) (h1 : michael_current_trophies = 50) (h2 : michael_increase = 150) (h3 : jack_multiplier = 15) :
  let michael_five_years : ℕ := michael_current_trophies + michael_increase
  let jack_five_years : ℕ := jack_multiplier * michael_current_trophies
  michael_five_years + jack_five_years = 950 :=
by
  sorry

end NUMINAMATH_GPT_total_trophies_after_five_years_l52_5209


namespace NUMINAMATH_GPT_product_of_two_integers_l52_5295

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) (h3 : x > y) : x * y = 168 := by
  sorry

end NUMINAMATH_GPT_product_of_two_integers_l52_5295


namespace NUMINAMATH_GPT_winner_percentage_of_votes_l52_5235

theorem winner_percentage_of_votes (V W O : ℕ) (W_votes : W = 720) (won_by : W - O = 240) (total_votes : V = W + O) :
  (W * 100) / V = 60 :=
by
  sorry

end NUMINAMATH_GPT_winner_percentage_of_votes_l52_5235


namespace NUMINAMATH_GPT_percentage_both_questions_correct_l52_5267

-- Definitions for the conditions in the problem
def percentage_first_question_correct := 85
def percentage_second_question_correct := 65
def percentage_neither_question_correct := 5
def percentage_one_or_more_questions_correct := 100 - percentage_neither_question_correct

-- Theorem stating that 55 percent answered both questions correctly
theorem percentage_both_questions_correct :
  percentage_first_question_correct + percentage_second_question_correct - percentage_one_or_more_questions_correct = 55 :=
by
  sorry

end NUMINAMATH_GPT_percentage_both_questions_correct_l52_5267


namespace NUMINAMATH_GPT_total_sales_in_december_correct_l52_5208

def ear_muffs_sales_in_december : ℝ :=
  let typeB_sold := 3258
  let typeB_price := 6.9
  let typeC_sold := 3186
  let typeC_price := 7.4
  let total_typeB_sales := typeB_sold * typeB_price
  let total_typeC_sales := typeC_sold * typeC_price
  total_typeB_sales + total_typeC_sales

theorem total_sales_in_december_correct :
  ear_muffs_sales_in_december = 46056.6 :=
by
  sorry

end NUMINAMATH_GPT_total_sales_in_december_correct_l52_5208


namespace NUMINAMATH_GPT_find_quadratic_minimum_value_l52_5232

noncomputable def quadraticMinimumPoint (a b c : ℝ) : ℝ :=
  -b / (2 * a)

theorem find_quadratic_minimum_value :
  quadraticMinimumPoint 3 6 9 = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_quadratic_minimum_value_l52_5232


namespace NUMINAMATH_GPT_market_trips_l52_5299

theorem market_trips (d_school_round: ℝ) (d_market_round: ℝ) (num_school_trips_per_day: ℕ) (num_school_days_per_week: ℕ) (total_week_mileage: ℝ) :
  d_school_round = 5 →
  d_market_round = 4 →
  num_school_trips_per_day = 2 →
  num_school_days_per_week = 4 →
  total_week_mileage = 44 →
  (total_week_mileage - (d_school_round * num_school_trips_per_day * num_school_days_per_week)) / d_market_round = 1 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_market_trips_l52_5299


namespace NUMINAMATH_GPT_intersection_of_lines_l52_5227

theorem intersection_of_lines :
  ∃ x y : ℚ, 12 * x - 5 * y = 8 ∧ 10 * x + 2 * y = 20 ∧ x = 58 / 37 ∧ y = 667 / 370 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l52_5227


namespace NUMINAMATH_GPT_number_of_integer_values_for_a_l52_5282

theorem number_of_integer_values_for_a :
  (∃ (a : Int), ∃ (p q : Int), p * q = -12 ∧ p + q = a ∧ p ≠ q) →
  (∃ (n : Nat), n = 6) := by
  sorry

end NUMINAMATH_GPT_number_of_integer_values_for_a_l52_5282


namespace NUMINAMATH_GPT_sum_of_ages_is_55_l52_5269

def sum_of_ages (Y : ℕ) (interval : ℕ) (number_of_children : ℕ) : ℕ :=
  let ages := List.range number_of_children |>.map (λ i => Y + i * interval)
  ages.sum

theorem sum_of_ages_is_55 :
  sum_of_ages 7 2 5 = 55 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_55_l52_5269


namespace NUMINAMATH_GPT_greatest_value_of_squares_exists_max_value_of_squares_l52_5298

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
sorry

theorem exists_max_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 702 :=
sorry

end NUMINAMATH_GPT_greatest_value_of_squares_exists_max_value_of_squares_l52_5298


namespace NUMINAMATH_GPT_geometric_sequence_formula_l52_5272

variable {q : ℝ} -- Common ratio
variable {m n : ℕ} -- Positive natural numbers
variable {b : ℕ → ℝ} -- Geometric sequence

-- This is only necessary if importing Mathlib didn't bring it in
noncomputable def geom_sequence (m n : ℕ) (b : ℕ → ℝ) (q : ℝ) : Prop :=
  b n = b m * q^(n - m)

theorem geometric_sequence_formula (q : ℝ) (m n : ℕ) (b : ℕ → ℝ) 
  (hmn : 0 < m ∧ 0 < n) :
  geom_sequence m n b q :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_formula_l52_5272


namespace NUMINAMATH_GPT_fruit_vendor_sold_fruits_l52_5221

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end NUMINAMATH_GPT_fruit_vendor_sold_fruits_l52_5221


namespace NUMINAMATH_GPT_extremal_values_d_l52_5212

theorem extremal_values_d (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ → Prop)
  (hC : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1)
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : ∃ (x y : ℝ), C (x, y)) :
  ∃ (max_d min_d : ℝ), max_d = 14 ∧ min_d = 10 :=
by
  -- Necessary assumptions
  have h₁ : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1 := hC
  have h₂ : A = (-1, 0) := hA
  have h₃ : B = (1, 0) := hB
  have h₄ : ∃ (x y : ℝ), C (x, y) := hP
  sorry

end NUMINAMATH_GPT_extremal_values_d_l52_5212


namespace NUMINAMATH_GPT_discount_correct_l52_5229

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end NUMINAMATH_GPT_discount_correct_l52_5229


namespace NUMINAMATH_GPT_problem_inequality_l52_5241

variable (x y z : ℝ)

theorem problem_inequality (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := by
  sorry

end NUMINAMATH_GPT_problem_inequality_l52_5241


namespace NUMINAMATH_GPT_rachels_milk_consumption_l52_5217

theorem rachels_milk_consumption :
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  rachel_ratio * total_milk = (15 / 32 : ℚ) :=
by
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_rachels_milk_consumption_l52_5217


namespace NUMINAMATH_GPT_current_walnut_trees_l52_5279

theorem current_walnut_trees (x : ℕ) (h : x + 55 = 77) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_current_walnut_trees_l52_5279


namespace NUMINAMATH_GPT_fraction_of_area_above_line_l52_5252

open Real

-- Define the points and the line between them
noncomputable def pointA : (ℝ × ℝ) := (2, 3)
noncomputable def pointB : (ℝ × ℝ) := (5, 1)

-- Define the vertices of the square
noncomputable def square_vertices : List (ℝ × ℝ) := [(2, 1), (5, 1), (5, 4), (2, 4)]

-- Define the equation of the line
noncomputable def line_eq (x : ℝ) : ℝ :=
  (-2/3) * x + 13/3

-- Define the vertical and horizontal boundaries
noncomputable def x_min : ℝ := 2
noncomputable def x_max : ℝ := 5
noncomputable def y_min : ℝ := 1
noncomputable def y_max : ℝ := 4

-- Calculate the area of the triangle formed below the line
noncomputable def triangle_area : ℝ := 0.5 * 2 * 3

-- Calculate the area of the square
noncomputable def square_area : ℝ := 3 * 3

-- The fraction of the area above the line
noncomputable def area_fraction_above : ℝ := (square_area - triangle_area) / square_area

-- Prove the fraction of the area of the square above the line is 2/3
theorem fraction_of_area_above_line : area_fraction_above = 2 / 3 :=
  sorry

end NUMINAMATH_GPT_fraction_of_area_above_line_l52_5252


namespace NUMINAMATH_GPT_intersection_is_correct_l52_5289

noncomputable def M : Set ℝ := { x | 1 + x ≥ 0 }
noncomputable def N : Set ℝ := { x | 4 / (1 - x) > 0 }
noncomputable def intersection : Set ℝ := { x | -1 ≤ x ∧ x < 1 }

theorem intersection_is_correct : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l52_5289


namespace NUMINAMATH_GPT_correct_mean_l52_5264

theorem correct_mean (mean n incorrect_value correct_value : ℝ) 
  (hmean : mean = 150) (hn : n = 20) (hincorrect : incorrect_value = 135) (hcorrect : correct_value = 160):
  (mean * n - incorrect_value + correct_value) / n = 151.25 :=
by
  sorry

end NUMINAMATH_GPT_correct_mean_l52_5264


namespace NUMINAMATH_GPT_largest_num_consecutive_integers_sum_45_l52_5277

theorem largest_num_consecutive_integers_sum_45 : 
  ∃ n : ℕ, (0 < n) ∧ (n * (n + 1) / 2 = 45) ∧ (∀ m : ℕ, (0 < m) → m * (m + 1) / 2 = 45 → m ≤ n) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_num_consecutive_integers_sum_45_l52_5277


namespace NUMINAMATH_GPT_orchestra_club_members_l52_5218

theorem orchestra_club_members : ∃ (n : ℕ), 150 < n ∧ n < 250 ∧ n % 8 = 1 ∧ n % 6 = 2 ∧ n % 9 = 3 ∧ n = 169 := 
by {
  sorry
}

end NUMINAMATH_GPT_orchestra_club_members_l52_5218


namespace NUMINAMATH_GPT_mr_brown_financial_outcome_l52_5257

theorem mr_brown_financial_outcome :
  ∃ (C₁ C₂ : ℝ), (2.40 = 1.25 * C₁) ∧ (2.40 = 0.75 * C₂) ∧ ((2.40 + 2.40) - (C₁ + C₂) = -0.32) :=
by
  sorry

end NUMINAMATH_GPT_mr_brown_financial_outcome_l52_5257


namespace NUMINAMATH_GPT_find_dividend_l52_5205

noncomputable def quotient : ℕ := 2015
noncomputable def remainder : ℕ := 0
noncomputable def divisor : ℕ := 105

theorem find_dividend : quotient * divisor + remainder = 20685 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l52_5205


namespace NUMINAMATH_GPT_number_division_l52_5285

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end NUMINAMATH_GPT_number_division_l52_5285


namespace NUMINAMATH_GPT_minimum_possible_value_of_BC_l52_5253

def triangle_ABC_side_lengths_are_integers (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def angle_A_is_twice_angle_B (A B C : ℝ) : Prop :=
  A = 2 * B

def CA_is_nine (CA : ℕ) : Prop :=
  CA = 9

theorem minimum_possible_value_of_BC
  (a b c : ℕ) (A B C : ℝ) (CA : ℕ)
  (h1 : triangle_ABC_side_lengths_are_integers a b c)
  (h2 : angle_A_is_twice_angle_B A B C)
  (h3 : CA_is_nine CA) :
  ∃ (BC : ℕ), BC = 12 := 
sorry

end NUMINAMATH_GPT_minimum_possible_value_of_BC_l52_5253


namespace NUMINAMATH_GPT_car_rental_total_cost_l52_5275

theorem car_rental_total_cost 
  (rental_cost : ℕ)
  (gallons : ℕ)
  (cost_per_gallon : ℕ)
  (cost_per_mile : ℚ)
  (miles_driven : ℕ)
  (H1 : rental_cost = 150)
  (H2 : gallons = 8)
  (H3 : cost_per_gallon = 350 / 100)
  (H4 : cost_per_mile = 50 / 100)
  (H5 : miles_driven = 320) :
  rental_cost + gallons * cost_per_gallon + miles_driven * cost_per_mile = 338 :=
  sorry

end NUMINAMATH_GPT_car_rental_total_cost_l52_5275


namespace NUMINAMATH_GPT_original_planned_production_l52_5265

theorem original_planned_production (x : ℝ) (hx1 : x ≠ 0) (hx2 : 210 / x - 210 / (1.5 * x) = 5) : x = 14 :=
by sorry

end NUMINAMATH_GPT_original_planned_production_l52_5265


namespace NUMINAMATH_GPT_solve_abs_inequality_l52_5271

theorem solve_abs_inequality :
  { x : ℝ | 3 ≤ |x - 2| ∧ |x - 2| ≤ 6 } = { x : ℝ | -4 ≤ x ∧ x ≤ -1 } ∪ { x : ℝ | 5 ≤ x ∧ x ≤ 8 } :=
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l52_5271


namespace NUMINAMATH_GPT_evaluate_g_at_3_l52_5251

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem evaluate_g_at_3 : g 3 = 109 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l52_5251


namespace NUMINAMATH_GPT_quadratic_function_properties_l52_5291

noncomputable def f (x : ℝ) : ℝ := -5 / 2 * x^2 + 15 * x - 25 / 2

theorem quadratic_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, (f x = a * (x - 1) * (x - 5)) ∧ (f 3 = 10)) → 
  (f x = -5 / 2 * x^2 + 15 * x - 25 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_function_properties_l52_5291


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l52_5200

theorem problem1 : 2013^2 - 2012 * 2014 = 1 := 
by 
  sorry

variables (m n : ℤ)

theorem problem2 : ((m-n)^6 / (n-m)^4) * (m-n)^3 = (m-n)^5 :=
by 
  sorry

variables (a b c : ℤ)

theorem problem3 : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l52_5200


namespace NUMINAMATH_GPT_sin_x_lt_a_l52_5262

theorem sin_x_lt_a (a θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (hθ : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2 * n - 1) * Real.pi - θ < x ∧ x < 2 * n * Real.pi + θ} = {x : ℝ | Real.sin x < a} :=
sorry

end NUMINAMATH_GPT_sin_x_lt_a_l52_5262


namespace NUMINAMATH_GPT_part1_part2_l52_5273

-- Define the conditions for part (1)
def nonEmptyBoxes := ∀ i j k: Nat, (i ≠ j ∧ i ≠ k ∧ j ≠ k)
def ball3inBoxB := ∀ (b3: Nat) (B: Nat), b3 = 3 ∧ B > 0

-- Define the conditions for part (2)
def ball1notInBoxA := ∀ (b1: Nat) (A: Nat), b1 ≠ 1 ∧ A > 0
def ball2notInBoxB := ∀ (b2: Nat) (B: Nat), b2 ≠ 2 ∧ B > 0

-- Theorems to be proved
theorem part1 (h1: nonEmptyBoxes) (h2: ball3inBoxB) : ∃ n, n = 12 := by sorry

theorem part2 (h3: ball1notInBoxA) (h4: ball2notInBoxB) : ∃ n, n = 36 := by sorry

end NUMINAMATH_GPT_part1_part2_l52_5273


namespace NUMINAMATH_GPT_pauls_weekly_spending_l52_5260

def mowing_lawns : ℕ := 3
def weed_eating : ℕ := 3
def total_weeks : ℕ := 2
def total_money : ℕ := mowing_lawns + weed_eating
def spending_per_week : ℕ := total_money / total_weeks

theorem pauls_weekly_spending : spending_per_week = 3 := by
  sorry

end NUMINAMATH_GPT_pauls_weekly_spending_l52_5260


namespace NUMINAMATH_GPT_area_of_triangle_l52_5228

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h₁ : b = 2) (h₂ : c = 2 * Real.sqrt 2) (h₃ : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - B - C) = Real.sqrt 3 + 1 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l52_5228


namespace NUMINAMATH_GPT_problem1_problem2_l52_5201

variable (α : ℝ)

-- Equivalent problem 1
theorem problem1 (h : Real.tan α = 7) : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 := 
  sorry

-- Equivalent problem 2
theorem problem2 (h : Real.tan α = 7) : Real.sin α * Real.cos α = 7 / 50 := 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l52_5201


namespace NUMINAMATH_GPT_sovereign_states_upper_bound_l52_5250

theorem sovereign_states_upper_bound (n : ℕ) (k : ℕ) : 
  (∃ (lines : ℕ) (border_stop_moving : Prop) (countries_disappear : Prop)
     (create_un : Prop) (total_countries : ℕ),
        (lines = n)
        ∧ (border_stop_moving = true)
        ∧ (countries_disappear = true)
        ∧ (create_un = true)
        ∧ (total_countries = k)) 
  → k ≤ (n^3 + 5*n) / 6 + 1 := 
sorry

end NUMINAMATH_GPT_sovereign_states_upper_bound_l52_5250


namespace NUMINAMATH_GPT_count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l52_5245

open Nat

def num180Unchanged : Nat := 
  let valid_pairs := [(0, 0), (1, 1), (8, 8), (6, 9), (9, 6)];
  let middle_digits := [0, 1, 8];
  (valid_pairs.length) * ((valid_pairs.length + 1) * (valid_pairs.length + 1) * middle_digits.length)

def num180UnchangedDivBy4 : Nat :=
  let valid_div4_pairs := [(0, 0), (1, 6), (6, 0), (6, 8), (8, 0), (8, 8), (9, 6)];
  let middle_digits := [0, 1, 8];
  valid_div4_pairs.length * (valid_div4_pairs.length / 5) * middle_digits.length

def sum180UnchangedNumbers : Nat :=
   1959460200 -- The sum by the given problem

theorem count_7_digit_nums_180_reversible : num180Unchanged = 300 :=
sorry

theorem count_7_digit_nums_180_reversible_divis_by_4 : num180UnchangedDivBy4 = 75 :=
sorry

theorem sum_of_7_digit_nums_180_reversible : sum180UnchangedNumbers = 1959460200 :=
sorry

end NUMINAMATH_GPT_count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l52_5245


namespace NUMINAMATH_GPT_circle_equation_of_tangent_circle_l52_5247

theorem circle_equation_of_tangent_circle
  (h : ∀ x y: ℝ, x^2/4 - y^2 = 1 → (x = 2 ∨ x = -2) → y = 0)
  (asymptote : ∀ x y : ℝ, (y = (1/2)*x ∨ y = -(1/2)*x) → (x - 2)^2 + y^2 = (4/5))
  : ∃ k : ℝ, (∀ x y : ℝ, (x - 2)^2 + y^2 = k) → k = 4/5 := by
  sorry

end NUMINAMATH_GPT_circle_equation_of_tangent_circle_l52_5247


namespace NUMINAMATH_GPT_wire_cut_circle_square_area_eq_l52_5226

theorem wire_cut_circle_square_area_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (a^2 / (4 * π)) = ((b^2) / 16)) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_circle_square_area_eq_l52_5226


namespace NUMINAMATH_GPT_sampling_method_sequential_is_systematic_l52_5224

def is_sequential_ids (ids : List Nat) : Prop :=
  ids = [5, 10, 15, 20, 25, 30, 35, 40]

def is_systematic_sampling (sampling_method : Prop) : Prop :=
  sampling_method

theorem sampling_method_sequential_is_systematic :
  ∀ ids, is_sequential_ids ids → 
    is_systematic_sampling (ids = [5, 10, 15, 20, 25, 30, 35, 40]) :=
by
  intros
  apply id
  sorry

end NUMINAMATH_GPT_sampling_method_sequential_is_systematic_l52_5224


namespace NUMINAMATH_GPT_platform_length_eq_train_length_l52_5210

noncomputable def length_of_train : ℝ := 900
noncomputable def speed_of_train_kmh : ℝ := 108
noncomputable def speed_of_train_mpm : ℝ := (speed_of_train_kmh * 1000) / 60
noncomputable def crossing_time_min : ℝ := 1
noncomputable def total_distance_covered : ℝ := speed_of_train_mpm * crossing_time_min

theorem platform_length_eq_train_length :
  total_distance_covered - length_of_train = length_of_train :=
by
  sorry

end NUMINAMATH_GPT_platform_length_eq_train_length_l52_5210


namespace NUMINAMATH_GPT_find_sum_of_digits_in_base_l52_5220

theorem find_sum_of_digits_in_base (d A B : ℕ) (hd : d > 8) (hA : A < d) (hB : B < d) (h : (A * d + B) + (A * d + A) - (B * d + A) = 1 * d^2 + 8 * d + 0) : A + B = 10 :=
sorry

end NUMINAMATH_GPT_find_sum_of_digits_in_base_l52_5220


namespace NUMINAMATH_GPT_gcd_pow_sub_one_l52_5276

theorem gcd_pow_sub_one (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2000 - 1) : Nat.gcd m n = 2^24 - 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_pow_sub_one_l52_5276


namespace NUMINAMATH_GPT_probability_is_stable_frequency_l52_5204

/-- Definition of probability: the stable theoretical value reflecting the likelihood of event occurrence. -/
def probability (event : Type) : ℝ := sorry 

/-- Definition of frequency: the empirical count of how often an event occurs in repeated experiments. -/
def frequency (event : Type) (trials : ℕ) : ℝ := sorry 

/-- The statement that "probability is the stable value of frequency" is correct. -/
theorem probability_is_stable_frequency (event : Type) (trials : ℕ) :
  probability event = sorry ↔ true := 
by 
  -- This is where the proof would go, but is replaced with sorry for now. 
  sorry

end NUMINAMATH_GPT_probability_is_stable_frequency_l52_5204


namespace NUMINAMATH_GPT_rest_area_location_l52_5288

theorem rest_area_location :
  ∀ (A B : ℝ), A = 50 → B = 230 → (5 / 8 * (B - A) + A = 162.5) :=
by
  intros A B hA hB
  rw [hA, hB]
  -- doing the computation to show the rest area is at 162.5 km
  sorry

end NUMINAMATH_GPT_rest_area_location_l52_5288


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l52_5258

theorem parallel_vectors_x_value (x : ℝ) :
  (∀ k : ℝ, k ≠ 0 → (4, 2) = (k * x, k * (-3))) → x = -6 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l52_5258


namespace NUMINAMATH_GPT_primes_in_arithmetic_sequence_have_specific_ones_digit_l52_5287

-- Define the properties of the primes and the arithmetic sequence
theorem primes_in_arithmetic_sequence_have_specific_ones_digit
  (p q r s : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (prime_s : Nat.Prime s)
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4 ∧ s = r + 4)
  (p_gt_3 : p > 3) : 
  p % 10 = 9 := 
sorry

end NUMINAMATH_GPT_primes_in_arithmetic_sequence_have_specific_ones_digit_l52_5287


namespace NUMINAMATH_GPT_european_stamp_costs_l52_5214

theorem european_stamp_costs :
  let P_Italy := 0.07
  let P_Germany := 0.03
  let N_Italy := 9
  let N_Germany := 15
  N_Italy * P_Italy + N_Germany * P_Germany = 1.08 :=
by
  sorry

end NUMINAMATH_GPT_european_stamp_costs_l52_5214


namespace NUMINAMATH_GPT_simplify_expression_l52_5274

variable (a b c d x : ℝ)
variable (hab : a ≠ b)
variable (hac : a ≠ c)
variable (had : a ≠ d)
variable (hbc : b ≠ c)
variable (hbd : b ≠ d)
variable (hcd : c ≠ d)

theorem simplify_expression :
  ( ( (x + a)^4 / ((a - b)*(a - c)*(a - d)) )
  + ( (x + b)^4 / ((b - a)*(b - c)*(b - d)) )
  + ( (x + c)^4 / ((c - a)*(c - b)*(c - d)) )
  + ( (x + d)^4 / ((d - a)*(d - b)*(d - c)) ) = a + b + c + d + 4*x ) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l52_5274


namespace NUMINAMATH_GPT_semicircle_radius_l52_5236

theorem semicircle_radius (b h : ℝ) (base_eq_b : b = 16) (height_eq_h : h = 15) :
  let s := (2 * 17) / 2
  let area := 240 
  s * (r : ℝ) = area → r = 120 / 17 :=
  by
  intros s area
  sorry

end NUMINAMATH_GPT_semicircle_radius_l52_5236


namespace NUMINAMATH_GPT_compare_decimal_fraction_l52_5225

theorem compare_decimal_fraction : 0.8 - (1 / 2) = 0.3 := by
  sorry

end NUMINAMATH_GPT_compare_decimal_fraction_l52_5225


namespace NUMINAMATH_GPT_evaluate_expression_l52_5242

theorem evaluate_expression (a b x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
    (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l52_5242


namespace NUMINAMATH_GPT_find_s_l52_5248

theorem find_s (s : ℝ) :
  let P := (s - 3, 2)
  let Q := (1, s + 2)
  let M := ((s - 2) / 2, (s + 4) / 2)
  let dist_sq := (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2
  dist_sq = 3 * s^2 / 4 →
  s = -5 + 5 * Real.sqrt 2 ∨ s = -5 - 5 * Real.sqrt 2 :=
by
  intros P Q M dist_sq h
  sorry

end NUMINAMATH_GPT_find_s_l52_5248


namespace NUMINAMATH_GPT_vector_addition_l52_5246

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

-- State the problem as a theorem
theorem vector_addition : a + b = (-1, 5) := by
  -- the proof should go here
  sorry

end NUMINAMATH_GPT_vector_addition_l52_5246


namespace NUMINAMATH_GPT_class_groups_l52_5294

open Nat

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem class_groups (boys girls : ℕ) (group_size : ℕ) :
  boys = 9 → girls = 12 → group_size = 3 →
  (combinations boys 1 * combinations girls 2) + (combinations boys 2 * combinations girls 1) = 1026 :=
by
  intros
  sorry

end NUMINAMATH_GPT_class_groups_l52_5294


namespace NUMINAMATH_GPT_factor_expression_l52_5261

theorem factor_expression (b : ℤ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l52_5261


namespace NUMINAMATH_GPT_terminating_fraction_count_l52_5268

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end NUMINAMATH_GPT_terminating_fraction_count_l52_5268


namespace NUMINAMATH_GPT_last_digit_sum_l52_5259

theorem last_digit_sum (a b : ℕ) (exp : ℕ)
  (h₁ : a = 1993) (h₂ : b = 1995) (h₃ : exp = 2002) :
  ((a ^ exp + b ^ exp) % 10) = 4 := 
by
  sorry

end NUMINAMATH_GPT_last_digit_sum_l52_5259


namespace NUMINAMATH_GPT_no_integer_solutions_system_l52_5231

theorem no_integer_solutions_system :
  ¬∃ (x y z : ℤ), x^6 + x^3 + x^3 * y + y = 147^157 ∧ x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := 
sorry

end NUMINAMATH_GPT_no_integer_solutions_system_l52_5231


namespace NUMINAMATH_GPT_sum_of_sequence_l52_5239

theorem sum_of_sequence :
  3 + 15 + 27 + 53 + 65 + 17 + 29 + 41 + 71 + 83 = 404 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l52_5239


namespace NUMINAMATH_GPT_bamboo_node_volume_5_l52_5216

theorem bamboo_node_volume_5 {a_1 d : ℚ} :
  (a_1 + (a_1 + d) + (a_1 + 2 * d) + (a_1 + 3 * d) = 3) →
  ((a_1 + 6 * d) + (a_1 + 7 * d) + (a_1 + 8 * d) = 4) →
  (a_1 + 4 * d = 67 / 66) :=
by sorry

end NUMINAMATH_GPT_bamboo_node_volume_5_l52_5216


namespace NUMINAMATH_GPT_deepak_current_age_l52_5219

theorem deepak_current_age (x : ℕ) (rahul_age deepak_age : ℕ) :
  (rahul_age = 4 * x) →
  (deepak_age = 3 * x) →
  (rahul_age + 10 = 26) →
  deepak_age = 12 :=
by
  intros h1 h2 h3
  -- You would write the proof here
  sorry

end NUMINAMATH_GPT_deepak_current_age_l52_5219


namespace NUMINAMATH_GPT_convex_polyhedron_formula_l52_5286

theorem convex_polyhedron_formula
  (V E F t h T H : ℕ)
  (hF : F = 40)
  (hFaces : F = t + h)
  (hVertex : 2 * T + H = 7)
  (hEdges : E = (3 * t + 6 * h) / 2)
  (hEuler : V - E + F = 2)
  : 100 * H + 10 * T + V = 367 := 
sorry

end NUMINAMATH_GPT_convex_polyhedron_formula_l52_5286


namespace NUMINAMATH_GPT_marias_workday_ends_at_3_30_pm_l52_5263
open Nat

theorem marias_workday_ends_at_3_30_pm :
  let start_time := (7 : Nat)
  let lunch_start_time := (11 + (30 / 60))
  let work_duration := (8 : Nat)
  let lunch_break := (30 / 60 : Nat)
  let end_time := (15 + (30 / 60) : Nat)
  (start_time + work_duration + lunch_break) - (lunch_start_time - start_time) = end_time := by
  sorry

end NUMINAMATH_GPT_marias_workday_ends_at_3_30_pm_l52_5263


namespace NUMINAMATH_GPT_tom_read_books_l52_5249

theorem tom_read_books :
  let books_may := 2
  let books_june := 6
  let books_july := 10
  books_may + books_june + books_july = 18 := by
  sorry

end NUMINAMATH_GPT_tom_read_books_l52_5249


namespace NUMINAMATH_GPT_total_red_cards_l52_5203

def num_standard_decks : ℕ := 3
def num_special_decks : ℕ := 2
def num_custom_decks : ℕ := 2
def red_cards_standard_deck : ℕ := 26
def red_cards_special_deck : ℕ := 30
def red_cards_custom_deck : ℕ := 20

theorem total_red_cards : num_standard_decks * red_cards_standard_deck +
                          num_special_decks * red_cards_special_deck +
                          num_custom_decks * red_cards_custom_deck = 178 :=
by
  -- Calculation omitted
  sorry

end NUMINAMATH_GPT_total_red_cards_l52_5203


namespace NUMINAMATH_GPT_tommy_writing_time_l52_5206

def numUniqueLettersTommy : Nat := 5
def numRearrangementsPerMinute : Nat := 20
def totalRearrangements : Nat := numUniqueLettersTommy.factorial
def minutesToComplete : Nat := totalRearrangements / numRearrangementsPerMinute
def hoursToComplete : Rat := minutesToComplete / 60

theorem tommy_writing_time :
  hoursToComplete = 0.1 := by
  sorry

end NUMINAMATH_GPT_tommy_writing_time_l52_5206


namespace NUMINAMATH_GPT_remainder_product_191_193_197_mod_23_l52_5281

theorem remainder_product_191_193_197_mod_23 :
  (191 * 193 * 197) % 23 = 14 := by
  sorry

end NUMINAMATH_GPT_remainder_product_191_193_197_mod_23_l52_5281


namespace NUMINAMATH_GPT_num_dislikers_tv_books_games_is_correct_l52_5266

-- Definitions of the conditions as given in step A
def total_people : ℕ := 1500
def pct_dislike_tv : ℝ := 0.4
def pct_dislike_tv_books : ℝ := 0.15
def pct_dislike_tv_books_games : ℝ := 0.5

-- Calculate intermediate values
def num_tv_dislikers := pct_dislike_tv * total_people
def num_tv_books_dislikers := pct_dislike_tv_books * num_tv_dislikers
def num_tv_books_games_dislikers := pct_dislike_tv_books_games * num_tv_books_dislikers

-- Final proof statement ensuring the correctness of the solution
theorem num_dislikers_tv_books_games_is_correct :
  num_tv_books_games_dislikers = 45 := by
  -- Sorry placeholder for the proof. In actual Lean usage, this would require fulfilling the proof obligations.
  sorry

end NUMINAMATH_GPT_num_dislikers_tv_books_games_is_correct_l52_5266


namespace NUMINAMATH_GPT_year_when_P_costs_40_paise_more_than_Q_l52_5215

def price_of_P (n : ℕ) : ℝ := 4.20 + 0.40 * n
def price_of_Q (n : ℕ) : ℝ := 6.30 + 0.15 * n

theorem year_when_P_costs_40_paise_more_than_Q :
  ∃ n : ℕ, price_of_P n = price_of_Q n + 0.40 ∧ 2001 + n = 2011 :=
by
  sorry

end NUMINAMATH_GPT_year_when_P_costs_40_paise_more_than_Q_l52_5215


namespace NUMINAMATH_GPT_actual_average_speed_l52_5222

theorem actual_average_speed (v t : ℝ) (h1 : v > 0) (h2: t > 0) (h3 : (t / (t - (1 / 4) * t)) = ((v + 12) / v)) : v = 36 :=
by
  sorry

end NUMINAMATH_GPT_actual_average_speed_l52_5222


namespace NUMINAMATH_GPT_gcd_of_polynomials_l52_5223

theorem gcd_of_polynomials (n : ℕ) (h : n > 2^5) : gcd (n^3 + 5^2) (n + 6) = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l52_5223


namespace NUMINAMATH_GPT_min_transport_cost_l52_5233

theorem min_transport_cost :
  let large_truck_capacity := 7
  let large_truck_cost := 600
  let small_truck_capacity := 4
  let small_truck_cost := 400
  let total_goods := 20
  ∃ (n_large n_small : ℕ),
    n_large * large_truck_capacity + n_small * small_truck_capacity ≥ total_goods ∧ 
    (n_large * large_truck_cost + n_small * small_truck_cost) = 1800 :=
sorry

end NUMINAMATH_GPT_min_transport_cost_l52_5233


namespace NUMINAMATH_GPT_problem_solution_l52_5256

-- Define the conditions
variables {a c b d x y z q : Real}
axiom h1 : a^x = c^q ∧ c^q = b
axiom h2 : c^y = a^z ∧ a^z = d

-- State the theorem
theorem problem_solution : xy = zq :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l52_5256


namespace NUMINAMATH_GPT_question1_question2_l52_5280

variable (α : ℝ)

theorem question1 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = -15 / 23 := by
  sorry

theorem question2 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    Real.tan (α - 5 * π / 4) = -7 := by
  sorry

end NUMINAMATH_GPT_question1_question2_l52_5280


namespace NUMINAMATH_GPT_smallest_N_l52_5292

theorem smallest_N (N : ℕ) (h : 7 * N = 999999) : N = 142857 :=
sorry

end NUMINAMATH_GPT_smallest_N_l52_5292
