import Mathlib

namespace NUMINAMATH_GPT_evaluate_expression_l2072_207248

theorem evaluate_expression :
  (Int.floor ((Int.ceil ((11/5:ℚ)^2)) * (19/3:ℚ))) = 31 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2072_207248


namespace NUMINAMATH_GPT_sticks_predict_good_fortune_l2072_207293

def good_fortune_probability := 11 / 12

theorem sticks_predict_good_fortune:
  (∃ (α β: ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ 0 ≤ β ∧ β ≤ π / 2 ∧ (0 ≤ β ∧ β < π - α) ∧ (0 ≤ α ∧ α < π - β)) → 
  good_fortune_probability = 11 / 12 :=
sorry

end NUMINAMATH_GPT_sticks_predict_good_fortune_l2072_207293


namespace NUMINAMATH_GPT_g_f_x_not_quadratic_l2072_207254

open Real

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_f_x_not_quadratic (h : ∃ x : ℝ, x - f (g x) = 0) :
  ∀ x : ℝ, g (f x) ≠ x^2 + x + 1 / 5 := sorry

end NUMINAMATH_GPT_g_f_x_not_quadratic_l2072_207254


namespace NUMINAMATH_GPT_amy_local_calls_l2072_207287

-- Define the conditions as hypotheses
variable (L I : ℕ)
variable (h1 : L = (5 / 2 : ℚ) * I)
variable (h2 : L = (5 / 3 : ℚ) * (I + 3))

-- Statement of the theorem
theorem amy_local_calls : L = 15 := by
  sorry

end NUMINAMATH_GPT_amy_local_calls_l2072_207287


namespace NUMINAMATH_GPT_positive_integers_n_l2072_207204

theorem positive_integers_n (n a b : ℕ) (h1 : 2 < n) (h2 : n = a ^ 3 + b ^ 3) 
  (h3 : ∀ d, d > 1 ∧ d ∣ n → a ≤ d) (h4 : b ∣ n) : n = 16 ∨ n = 72 ∨ n = 520 :=
sorry

end NUMINAMATH_GPT_positive_integers_n_l2072_207204


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l2072_207274

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 3) : 3 ≤ x :=
by
  linarith

end NUMINAMATH_GPT_sqrt_meaningful_range_l2072_207274


namespace NUMINAMATH_GPT_rectangle_area_comparison_l2072_207268

theorem rectangle_area_comparison 
  {A A' B B' C C' D D': ℝ} 
  (h_A: A ≤ A') 
  (h_B: B ≤ B') 
  (h_C: C ≤ C') 
  (h_D: D ≤ B') : 
  A + B + C + D ≤ A' + B' + C' + D' := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_comparison_l2072_207268


namespace NUMINAMATH_GPT_age_of_father_l2072_207263

theorem age_of_father (F C : ℕ) 
  (h1 : F = C)
  (h2 : C + 5 * 15 = 2 * (F + 15)) : 
  F = 45 := 
by 
sorry

end NUMINAMATH_GPT_age_of_father_l2072_207263


namespace NUMINAMATH_GPT_series_sum_equals_one_l2072_207295

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 : ℝ)^(2 * (k + 1)) / ((3 : ℝ)^(2 * (k + 1)) - 1)

theorem series_sum_equals_one :
  series_sum = 1 :=
sorry

end NUMINAMATH_GPT_series_sum_equals_one_l2072_207295


namespace NUMINAMATH_GPT_eggs_total_l2072_207244

-- Definitions based on conditions
def isPackageSize (n : Nat) : Prop :=
  n = 6 ∨ n = 11

def numLargePacks : Nat := 5

def largePackSize : Nat := 11

-- Mathematical statement to prove
theorem eggs_total : ∃ totalEggs : Nat, totalEggs = numLargePacks * largePackSize :=
  by sorry

end NUMINAMATH_GPT_eggs_total_l2072_207244


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l2072_207200

theorem cone_lateral_surface_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 90) : 
  let base_circumference := 2 * Real.pi * r
  let R := 12
  let lateral_surface_area := (1 / 2) * base_circumference * R 
  lateral_surface_area = 36 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l2072_207200


namespace NUMINAMATH_GPT_megan_works_per_day_hours_l2072_207203

theorem megan_works_per_day_hours
  (h : ℝ)
  (earnings_per_hour : ℝ)
  (days_per_month : ℝ)
  (total_earnings_two_months : ℝ) :
  earnings_per_hour = 7.50 →
  days_per_month = 20 →
  total_earnings_two_months = 2400 →
  2 * days_per_month * earnings_per_hour * h = total_earnings_two_months →
  h = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_megan_works_per_day_hours_l2072_207203


namespace NUMINAMATH_GPT_commute_time_l2072_207216

theorem commute_time (d s1 s2 : ℝ) (h1 : s1 = 45) (h2 : s2 = 30) (h3 : d = 18) : (d / s1 + d / s2 = 1) :=
by
  -- Definitions and assumptions
  rw [h1, h2, h3]
  -- Total time calculation
  exact sorry

end NUMINAMATH_GPT_commute_time_l2072_207216


namespace NUMINAMATH_GPT_inscribed_circle_radius_l2072_207219

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (θ : ℝ) (tangent : ℝ) :
    θ = π / 3 →
    R = 5 →
    tangent = (5 : ℝ) * (Real.sqrt 2 - 1) →
    r * (1 + Real.sqrt 2) = R →
    r = 5 * (Real.sqrt 2 - 1) := 
by sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l2072_207219


namespace NUMINAMATH_GPT_robin_camera_pictures_l2072_207264

-- Given conditions
def pictures_from_phone : Nat := 35
def num_albums : Nat := 5
def pics_per_album : Nat := 8

-- Calculate total pictures and the number of pictures from the camera
theorem robin_camera_pictures : num_albums * pics_per_album - pictures_from_phone = 5 := by
  sorry

end NUMINAMATH_GPT_robin_camera_pictures_l2072_207264


namespace NUMINAMATH_GPT_dogs_with_flea_collars_l2072_207226

-- Conditions
def T : ℕ := 80
def Tg : ℕ := 45
def B : ℕ := 6
def N : ℕ := 1

-- Goal: prove the number of dogs with flea collars is 40 given the above conditions
theorem dogs_with_flea_collars : ∃ F : ℕ, F = 40 ∧ T = Tg + F - B + N := 
by
  use 40
  sorry

end NUMINAMATH_GPT_dogs_with_flea_collars_l2072_207226


namespace NUMINAMATH_GPT_manager_salary_correct_l2072_207298

-- Define the conditions of the problem
def total_salary_of_24_employees : ℕ := 24 * 2400
def new_average_salary_with_manager : ℕ := 2500
def number_of_people_with_manager : ℕ := 25

-- Define the manager's salary to be proved
def managers_salary : ℕ := 4900

-- Statement of the theorem to prove that the manager's salary is Rs. 4900
theorem manager_salary_correct :
  (number_of_people_with_manager * new_average_salary_with_manager) - total_salary_of_24_employees = managers_salary :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_manager_salary_correct_l2072_207298


namespace NUMINAMATH_GPT_evaluate_expression_l2072_207288

theorem evaluate_expression (x : ℝ) (h : 3 * x^3 - x = 1) : 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2072_207288


namespace NUMINAMATH_GPT_equivalent_multipliers_l2072_207275

variable (a b : ℝ)

theorem equivalent_multipliers (a b : ℝ) :
  let a_final := 0.93 * a
  let expr := a_final + 0.05 * b
  expr = 0.93 * a + 0.05 * b  :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_equivalent_multipliers_l2072_207275


namespace NUMINAMATH_GPT_daily_evaporation_l2072_207238

theorem daily_evaporation :
  ∀ (initial_amount : ℝ) (percentage_evaporated : ℝ) (days : ℕ),
  initial_amount = 10 →
  percentage_evaporated = 6 →
  days = 50 →
  (initial_amount * (percentage_evaporated / 100)) / days = 0.012 :=
by
  intros initial_amount percentage_evaporated days
  intros h_initial h_percentage h_days
  rw [h_initial, h_percentage, h_days]
  sorry

end NUMINAMATH_GPT_daily_evaporation_l2072_207238


namespace NUMINAMATH_GPT_range_of_q_l2072_207235

variable (a_n : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
variable (hg_seq : ∀ n : ℕ, n > 0 → ∃ a_1 : ℝ, S_n n = a_1 * (1 - q ^ n) / (1 - q))
variable (pos_sum : ∀ n : ℕ, n > 0 → S_n n > 0)

theorem range_of_q : q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ) := sorry

end NUMINAMATH_GPT_range_of_q_l2072_207235


namespace NUMINAMATH_GPT_temperature_difference_correct_l2072_207282

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_correct_l2072_207282


namespace NUMINAMATH_GPT_parallelogram_sides_l2072_207296

theorem parallelogram_sides (x y : ℝ) (h1 : 12 * y - 2 = 10) (h2 : 5 * x + 15 = 20) : x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_sides_l2072_207296


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l2072_207218

theorem arithmetic_sequence_ratio (a x b : ℝ) 
  (h1 : x - a = b - x)
  (h2 : 2 * x - b = b - x) :
  a / b = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l2072_207218


namespace NUMINAMATH_GPT_jen_profit_l2072_207246

-- Definitions based on the conditions
def cost_per_candy := 80 -- in cents
def sell_price_per_candy := 100 -- in cents
def total_candies_bought := 50
def total_candies_sold := 48

-- Total cost and total revenue calculations
def total_cost := cost_per_candy * total_candies_bought
def total_revenue := sell_price_per_candy * total_candies_sold

-- Profit calculation
def profit := total_revenue - total_cost

-- Main theorem to prove
theorem jen_profit : profit = 800 := by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_jen_profit_l2072_207246


namespace NUMINAMATH_GPT_total_matches_in_2006_world_cup_l2072_207236

-- Define relevant variables and conditions
def teams := 32
def groups := 8
def top2_from_each_group := 16

-- Calculate the number of matches in Group Stage
def matches_in_group_stage :=
  let matches_per_group := 6
  matches_per_group * groups

-- Calculate the number of matches in Knockout Stage
def matches_in_knockout_stage :=
  let first_round_matches := 8
  let quarter_final_matches := 4
  let semi_final_matches := 2
  let final_and_third_place_matches := 2
  first_round_matches + quarter_final_matches + semi_final_matches + final_and_third_place_matches

-- Total number of matches
theorem total_matches_in_2006_world_cup : matches_in_group_stage + matches_in_knockout_stage = 64 := by
  sorry

end NUMINAMATH_GPT_total_matches_in_2006_world_cup_l2072_207236


namespace NUMINAMATH_GPT_given_expression_equality_l2072_207213

theorem given_expression_equality (x : ℝ) (A ω φ b : ℝ) (hA : 0 < A)
  (h : 2 * (Real.cos x)^2 + Real.sin (2 * x) = A * Real.sin (ω * x + φ) + b) :
  A = Real.sqrt 2 ∧ b = 1 :=
sorry

end NUMINAMATH_GPT_given_expression_equality_l2072_207213


namespace NUMINAMATH_GPT_picnic_recyclable_collected_l2072_207292

theorem picnic_recyclable_collected :
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  soda_drinkers + sparkling_water_drinkers + juice_consumed = 115 :=
by
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  show soda_drinkers + sparkling_water_drinkers + juice_consumed = 115
  sorry

end NUMINAMATH_GPT_picnic_recyclable_collected_l2072_207292


namespace NUMINAMATH_GPT_gears_can_rotate_l2072_207214

theorem gears_can_rotate (n : ℕ) : (∃ f : ℕ → Prop, f 0 ∧ (∀ k, f (k+1) ↔ ¬f k) ∧ f n = f 0) ↔ (n % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_gears_can_rotate_l2072_207214


namespace NUMINAMATH_GPT_remainder_of_difference_l2072_207208

open Int

theorem remainder_of_difference (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 3) (h : a > b) : (a - b) % 6 = 5 :=
  sorry

end NUMINAMATH_GPT_remainder_of_difference_l2072_207208


namespace NUMINAMATH_GPT_seashells_total_l2072_207261

theorem seashells_total :
  let sally := 9.5
  let tom := 7.2
  let jessica := 5.3
  let alex := 12.8
  sally + tom + jessica + alex = 34.8 :=
by
  sorry

end NUMINAMATH_GPT_seashells_total_l2072_207261


namespace NUMINAMATH_GPT_hot_drinks_sales_l2072_207297

theorem hot_drinks_sales (x: ℝ) (h: x = 4) : abs ((-2.35 * x + 155.47) - 146) < 1 :=
by sorry

end NUMINAMATH_GPT_hot_drinks_sales_l2072_207297


namespace NUMINAMATH_GPT_sandy_grew_6_carrots_l2072_207262

theorem sandy_grew_6_carrots (sam_grew : ℕ) (total_grew : ℕ) (h1 : sam_grew = 3) (h2 : total_grew = 9) : ∃ sandy_grew : ℕ, sandy_grew = total_grew - sam_grew ∧ sandy_grew = 6 :=
by
  sorry

end NUMINAMATH_GPT_sandy_grew_6_carrots_l2072_207262


namespace NUMINAMATH_GPT_smallest_k_exists_l2072_207229

open Nat

theorem smallest_k_exists (n m k : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) (hk : k % 3 = 0) :
  (64^k + 32^m > 4^(16 + n^2)) ↔ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_exists_l2072_207229


namespace NUMINAMATH_GPT_six_digit_numbers_l2072_207285

def isNonPerfectPower (n : ℕ) : Prop :=
  ∀ m k : ℕ, m ≥ 2 → k ≥ 2 → m^k ≠ n

theorem six_digit_numbers : ∃ x : ℕ, 
  100000 ≤ x ∧ x < 1000000 ∧ 
  (∃ a b c: ℕ, x = (a^3 * b)^2 ∧ isNonPerfectPower a ∧ isNonPerfectPower b ∧ isNonPerfectPower c ∧ 
    (∃ k : ℤ, k > 1 ∧ 
      (x: ℤ) / (k^3 : ℤ) < 1 ∧ 
      ∃ num denom: ℕ, num < denom ∧ 
      num = n^3 ∧ denom = d^2 ∧ 
      isNonPerfectPower n ∧ isNonPerfectPower d)) := 
sorry

end NUMINAMATH_GPT_six_digit_numbers_l2072_207285


namespace NUMINAMATH_GPT_servings_per_guest_l2072_207252

-- Definitions based on conditions
def num_guests : ℕ := 120
def servings_per_bottle : ℕ := 6
def num_bottles : ℕ := 40

-- Theorem statement
theorem servings_per_guest : (num_bottles * servings_per_bottle) / num_guests = 2 := by
  sorry

end NUMINAMATH_GPT_servings_per_guest_l2072_207252


namespace NUMINAMATH_GPT_largest_possible_product_is_3886_l2072_207221

theorem largest_possible_product_is_3886 :
  ∃ a b c d : ℕ, 5 ≤ a ∧ a ≤ 8 ∧
               5 ≤ b ∧ b ≤ 8 ∧
               5 ≤ c ∧ c ≤ 8 ∧
               5 ≤ d ∧ d ≤ 8 ∧
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
               b ≠ c ∧ b ≠ d ∧
               c ≠ d ∧
               (max ((10 * a + b) * (10 * c + d))
                    ((10 * c + b) * (10 * a + d))) = 3886 :=
sorry

end NUMINAMATH_GPT_largest_possible_product_is_3886_l2072_207221


namespace NUMINAMATH_GPT_grill_ran_for_16_hours_l2072_207209

def coals_burn_time_A (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 15 * 20)) 0

def coals_burn_time_B (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 10 * 30)) 0

def total_grill_time (bags_A bags_B : List ℕ) : ℕ :=
  coals_burn_time_A bags_A + coals_burn_time_B bags_B

def bags_A : List ℕ := [60, 75, 45]
def bags_B : List ℕ := [50, 70, 40, 80]

theorem grill_ran_for_16_hours :
  total_grill_time bags_A bags_B = 960 / 60 :=
by
  unfold total_grill_time coals_burn_time_A coals_burn_time_B
  unfold bags_A bags_B
  norm_num
  sorry

end NUMINAMATH_GPT_grill_ran_for_16_hours_l2072_207209


namespace NUMINAMATH_GPT_find_m_l2072_207237

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition that a is parallel to m * b - c
def is_parallel (a : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  a.1 * v.2 = a.2 * v.1

-- The main theorem we want to prove
theorem find_m (m : ℝ) (h : is_parallel a (m * b.1 - c.1, m * b.2 - c.2)) : m = -3 :=
by {
  -- This will be filled in with the appropriate proof
  sorry
}

end NUMINAMATH_GPT_find_m_l2072_207237


namespace NUMINAMATH_GPT_fractional_expression_evaluation_l2072_207299

theorem fractional_expression_evaluation (a : ℝ) (h : a^3 + 3 * a^2 + a = 0) :
  ∃ b : ℝ, b = 0 ∨ b = 1 ∧ b = 2022 * a^2 / (a^4 + 2015 * a^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_fractional_expression_evaluation_l2072_207299


namespace NUMINAMATH_GPT_a2016_value_l2072_207249

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 1 - (1 / a n)

theorem a2016_value : ∃ a : ℕ → ℚ, seq a ∧ a 2016 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_a2016_value_l2072_207249


namespace NUMINAMATH_GPT_part1_part2_l2072_207205

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 - (4 / (2 * a^x + a))

theorem part1 (h₁ : ∀ x, f a x = -f a (-x)) (h₂ : a > 0) (h₃ : a ≠ 1) : a = 2 :=
  sorry

theorem part2 (h₁ : a = 2) (x : ℝ) (hx : 0 < x ∧ x ≤ 1) (t : ℝ) :
  t * (f a x) ≥ 2^x - 2 ↔ t ≥ 0 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l2072_207205


namespace NUMINAMATH_GPT_coins_after_tenth_hour_l2072_207269

-- Given variables representing the number of coins added or removed each hour.
def coins_put_in : ℕ :=
  20 + 30 + 30 + 40 + 50 + 60 + 70

def coins_taken_out : ℕ :=
  20 + 15 + 25

-- Definition of the full proof problem
theorem coins_after_tenth_hour :
  coins_put_in - coins_taken_out = 240 :=
by
  sorry

end NUMINAMATH_GPT_coins_after_tenth_hour_l2072_207269


namespace NUMINAMATH_GPT_geometric_sequence_product_l2072_207215

variable (a : ℕ → ℝ)

def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (h_geom : is_geometric_seq a) (h_a6 : a 6 = 3) :
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l2072_207215


namespace NUMINAMATH_GPT_scientific_notation_correct_l2072_207265

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2072_207265


namespace NUMINAMATH_GPT_trigonometric_identity_l2072_207240

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2072_207240


namespace NUMINAMATH_GPT_compute_expression_l2072_207271

theorem compute_expression :
  (143 + 29) * 2 + 25 + 13 = 382 :=
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l2072_207271


namespace NUMINAMATH_GPT_quadratic_has_one_solution_l2072_207220

theorem quadratic_has_one_solution (m : ℝ) : 3 * (49 / 12) - 7 * (49 / 12) + m = 0 → m = 49 / 12 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_l2072_207220


namespace NUMINAMATH_GPT_prob_sum_7_9_11_correct_l2072_207222

def die1 : List ℕ := [1, 2, 3, 3, 4, 4]
def die2 : List ℕ := [2, 2, 5, 6, 7, 8]

def prob_sum_7_9_11 : ℚ := 
  (1/6 * 1/6 + 1/6 * 1/6) + 2/6 * 3/6

theorem prob_sum_7_9_11_correct :
  prob_sum_7_9_11 = 4 / 9 := 
by
  sorry

end NUMINAMATH_GPT_prob_sum_7_9_11_correct_l2072_207222


namespace NUMINAMATH_GPT_fraction_sum_is_0_333_l2072_207259

theorem fraction_sum_is_0_333 : (3 / 10 : ℝ) + (3 / 100) + (3 / 1000) = 0.333 := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_is_0_333_l2072_207259


namespace NUMINAMATH_GPT_debut_show_tickets_l2072_207286

variable (P : ℕ) -- Number of people who bought tickets for the debut show

-- Conditions
def three_times_more (P : ℕ) : Bool := (3 * P = P + 2 * P)
def ticket_cost : ℕ := 25
def total_revenue (P : ℕ) : ℕ := 4 * P * ticket_cost

-- Main statement
theorem debut_show_tickets (h1 : three_times_more P = true) 
                           (h2 : total_revenue P = 20000) : P = 200 :=
by
  sorry

end NUMINAMATH_GPT_debut_show_tickets_l2072_207286


namespace NUMINAMATH_GPT_power_function_propositions_l2072_207277

theorem power_function_propositions : (∀ n : ℤ, n > 0 → ∀ x : ℝ, x > 0 → (x^n) < x) ∧
  (∀ n : ℤ, n < 0 → ∀ x : ℝ, x > 0 → (x^n) > x) :=
by
  sorry

end NUMINAMATH_GPT_power_function_propositions_l2072_207277


namespace NUMINAMATH_GPT_fraction_increase_invariance_l2072_207232

theorem fraction_increase_invariance (x y : ℝ) :
  (3 * (2 * y)) / (2 * x + 2 * y) = 3 * y / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_invariance_l2072_207232


namespace NUMINAMATH_GPT_binary_div_remainder_l2072_207294

theorem binary_div_remainder (n : ℕ) (h : n = 0b101011100101) : n % 8 = 5 :=
by sorry

end NUMINAMATH_GPT_binary_div_remainder_l2072_207294


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l2072_207283

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_arithmetic_sequence {a : ℕ → ℝ} 
  (h_arith : arithmetic_seq a)
  (h1 : a 2^2 + a 7^2 + 2 * a 2 * a 7 = 9)
  (h2 : ∀ n, a n < 0) : 
  S₁₀ = -15 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l2072_207283


namespace NUMINAMATH_GPT_obtuse_and_acute_angles_in_convex_octagon_l2072_207245

theorem obtuse_and_acute_angles_in_convex_octagon (m n : ℕ) (h₀ : n + m = 8) : m > n :=
sorry

end NUMINAMATH_GPT_obtuse_and_acute_angles_in_convex_octagon_l2072_207245


namespace NUMINAMATH_GPT_zero_in_interval_l2072_207279

theorem zero_in_interval (x y : ℝ) (hx_lt_0 : x < 0) (hy_gt_0 : 0 < y) (hy_lt_1 : y < 1) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) : x^5 < 0 ∧ 0 < y^8 :=
by
  sorry

end NUMINAMATH_GPT_zero_in_interval_l2072_207279


namespace NUMINAMATH_GPT_difference_of_digits_l2072_207267

theorem difference_of_digits (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h_diff : (10 * x + y) - (10 * y + x) = 54) : x - y = 6 :=
sorry

end NUMINAMATH_GPT_difference_of_digits_l2072_207267


namespace NUMINAMATH_GPT_baseball_card_ratio_l2072_207227

-- Define the conditions
variable (T : ℤ) -- Number of baseball cards on Tuesday

-- Given conditions
-- On Monday, Buddy has 30 baseball cards
def monday_cards : ℤ := 30

-- On Wednesday, Buddy has T + 12 baseball cards
def wednesday_cards : ℤ := T + 12

-- On Thursday, Buddy buys a third of what he had on Tuesday
def thursday_additional_cards : ℤ := T / 3

-- Total number of cards on Thursday is 32
def thursday_cards (T : ℤ) : ℤ := T + 12 + T / 3

-- We are given that Buddy has 32 baseball cards on Thursday
axiom thursday_total : thursday_cards T = 32

-- The theorem we want to prove: the ratio of Tuesday's to Monday's cards is 1:2
theorem baseball_card_ratio
  (T : ℤ)
  (htotal : thursday_cards T = 32)
  (hmon : monday_cards = 30) :
  T = 15 ∧ (T : ℚ) / monday_cards = 1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_baseball_card_ratio_l2072_207227


namespace NUMINAMATH_GPT_solutions_are__l2072_207273

def satisfies_system (x y z : ℝ) : Prop :=
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540

theorem solutions_are_ (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by
  sorry

end NUMINAMATH_GPT_solutions_are__l2072_207273


namespace NUMINAMATH_GPT_horner_eval_at_2_l2072_207260

def poly (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_eval_at_2 : poly 2 = 373 := by
  sorry

end NUMINAMATH_GPT_horner_eval_at_2_l2072_207260


namespace NUMINAMATH_GPT_find_a_maximize_profit_l2072_207225

-- Definition of parameters
def a := 260
def purchase_price_table := a
def purchase_price_chair := a - 140

-- Condition 1: The number of dining chairs purchased for 600 yuan is the same as the number of dining tables purchased for 1300 yuan.
def condition1 := (600 / (purchase_price_chair : ℚ)) = (1300 / (purchase_price_table : ℚ))

-- Given conditions for profit maximization
def qty_tables := 30
def qty_chairs := 5 * qty_tables + 20
def total_qty := qty_tables + qty_chairs

-- Condition: Total quantity of items does not exceed 200 units.
def condition2 := total_qty ≤ 200

-- Profit calculation
def profit := 280 * qty_tables + 800

-- Theorem statements
theorem find_a : condition1 → a = 260 := sorry

theorem maximize_profit : condition2 ∧ (8 * qty_tables + 800 > 0) → 
  (qty_tables = 30) ∧ (qty_chairs = 170) ∧ (profit = 9200) := sorry

end NUMINAMATH_GPT_find_a_maximize_profit_l2072_207225


namespace NUMINAMATH_GPT_train_can_speed_up_l2072_207257

theorem train_can_speed_up (d t_reduced v_increased v_safe : ℝ) 
  (h1 : d = 1600) (h2 : t_reduced = 4) (h3 : v_increased = 20) (h4 : v_safe = 140) :
  ∃ x : ℝ, (x > 0) ∧ (d / x) = (d / (x + v_increased) + t_reduced) ∧ ((x + v_increased) < v_safe) :=
by 
  sorry

end NUMINAMATH_GPT_train_can_speed_up_l2072_207257


namespace NUMINAMATH_GPT_ratio_is_one_half_l2072_207233

-- Define the problem conditions as constants
def robert_age_in_2_years : ℕ := 30
def years_until_robert_is_30 : ℕ := 2
def patrick_current_age : ℕ := 14

-- Using the conditions, set up the definitions for the proof
def robert_current_age : ℕ := robert_age_in_2_years - years_until_robert_is_30

-- Define the target ratio
def ratio_of_ages : ℚ := patrick_current_age / robert_current_age

-- Prove that the ratio of Patrick's age to Robert's age is 1/2
theorem ratio_is_one_half : ratio_of_ages = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_is_one_half_l2072_207233


namespace NUMINAMATH_GPT_relationship_among_ys_l2072_207253

-- Define the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ :=
  -2 * x + b

-- Define the points on the graph
def y1 (b : ℝ) : ℝ :=
  linear_function (-2) b

def y2 (b : ℝ) : ℝ :=
  linear_function (-1) b

def y3 (b : ℝ) : ℝ :=
  linear_function 1 b

-- Theorem to prove the relation among y1, y2, y3
theorem relationship_among_ys (b : ℝ) : y1 b > y2 b ∧ y2 b > y3 b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_ys_l2072_207253


namespace NUMINAMATH_GPT_count_prime_sum_112_l2072_207239

noncomputable def primeSum (primes : List ℕ) : ℕ :=
  if H : ∀ p ∈ primes, Nat.Prime p ∧ p > 10 then primes.sum else 0

theorem count_prime_sum_112 :
  ∃ (primes : List ℕ), primeSum primes = 112 ∧ primes.length = 6 := by
  sorry

end NUMINAMATH_GPT_count_prime_sum_112_l2072_207239


namespace NUMINAMATH_GPT_incorrect_statement_l2072_207280

-- Define the general rules of program flowcharts
def isValidStart (box : String) : Prop := box = "start"
def isValidEnd (box : String) : Prop := box = "end"
def isInputBox (box : String) : Prop := box = "input"
def isOutputBox (box : String) : Prop := box = "output"

-- Define the statement to be proved incorrect
def statement (boxes : List String) : Prop :=
  ∀ xs ys, boxes = xs ++ ["start", "input"] ++ ys ->
           ∀ zs ws, boxes = zs ++ ["output", "end"] ++ ws

-- The target theorem stating that the statement is incorrect
theorem incorrect_statement (boxes : List String) :
  ¬ statement boxes :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l2072_207280


namespace NUMINAMATH_GPT_solve_for_C_l2072_207201

-- Given constants and assumptions
def SumOfDigitsFirst (A B : ℕ) := 8 + 4 + A + 5 + 3 + B + 2 + 1
def SumOfDigitsSecond (A B C : ℕ) := 5 + 2 + 7 + A + B + 6 + 0 + C

theorem solve_for_C (A B C : ℕ) 
  (h1 : (SumOfDigitsFirst A B % 9) = 0)
  (h2 : (SumOfDigitsSecond A B C % 9) = 0) 
  : C = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_C_l2072_207201


namespace NUMINAMATH_GPT_length_of_AB_l2072_207207
-- Import the necessary libraries

-- Define the quadratic function
def quad (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define a predicate to state that x is a root of the quadratic
def is_root (x : ℝ) : Prop := quad x = 0

-- Define the length between the intersection points
theorem length_of_AB :
  (is_root (-1)) ∧ (is_root 3) → |3 - (-1)| = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_AB_l2072_207207


namespace NUMINAMATH_GPT_atLeastOneTrueRange_exactlyOneTrueRange_l2072_207272

-- Definitions of Proposition A and B
def propA (a : ℝ) : Prop := ∀ x, x^2 + (a - 1) * x + a^2 ≤ 0 → false
def propB (a : ℝ) : Prop := ∀ x, (2 * a^2 - a)^x < (2 * a^2 - a)^(x + 1)

-- At least one of A or B is true
def atLeastOneTrue (a : ℝ) : Prop :=
  propA a ∨ propB a

-- Exactly one of A or B is true
def exactlyOneTrue (a : ℝ) : Prop := 
  (propA a ∧ ¬ propB a) ∨ (¬ propA a ∧ propB a)

-- Theorems to prove
theorem atLeastOneTrueRange :
  ∃ a : ℝ, atLeastOneTrue a ↔ (a < -1/2 ∨ a > 1/3) := 
sorry

theorem exactlyOneTrueRange :
  ∃ a : ℝ, exactlyOneTrue a ↔ ((1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end NUMINAMATH_GPT_atLeastOneTrueRange_exactlyOneTrueRange_l2072_207272


namespace NUMINAMATH_GPT_tutoring_minutes_l2072_207256

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end NUMINAMATH_GPT_tutoring_minutes_l2072_207256


namespace NUMINAMATH_GPT_cost_milk_is_5_l2072_207210

-- Define the total cost the baker paid
def total_cost : ℕ := 80

-- Define the cost components
def cost_flour : ℕ := 3 * 3
def cost_eggs : ℕ := 3 * 10
def cost_baking_soda : ℕ := 2 * 3

-- Define the number of liters of milk
def liters_milk : ℕ := 7

-- Define the unknown cost per liter of milk
noncomputable def cost_per_liter_milk (c : ℕ) : Prop :=
  c * liters_milk = total_cost - (cost_flour + cost_eggs + cost_baking_soda)

-- State the theorem we want to prove
theorem cost_milk_is_5 : cost_per_liter_milk 5 := 
by
  sorry

end NUMINAMATH_GPT_cost_milk_is_5_l2072_207210


namespace NUMINAMATH_GPT_expression_for_x_l2072_207206

variable (A B C x y : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := C > B
def condition3 := B > 0
def condition4 := C = (1 + y / 100) * B
def condition5 := A = (1 + x / 100) * C

-- The theorem
theorem expression_for_x (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 B) (h4 : condition4 B C y) (h5 : condition5 A C x) :
    x = 100 * ((100 * (A - B)) / (100 + y)) :=
sorry

end NUMINAMATH_GPT_expression_for_x_l2072_207206


namespace NUMINAMATH_GPT_second_date_sum_eq_80_l2072_207278

theorem second_date_sum_eq_80 (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 = 80)
  (h2 : a2 = a1 + 1) (h3 : a3 = a2 + 1) (h4 : a4 = a3 + 1) (h5 : a5 = a4 + 1): a2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_second_date_sum_eq_80_l2072_207278


namespace NUMINAMATH_GPT_hamburgers_left_over_l2072_207212

-- Define the conditions as constants
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove that the number of hamburgers left over is 6
theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end NUMINAMATH_GPT_hamburgers_left_over_l2072_207212


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l2072_207255

theorem distance_between_parallel_lines 
  (d : ℝ) 
  (r : ℝ)
  (h1 : (42 * 21 + (d / 2) * 42 * (d / 2) = 42 * r^2))
  (h2 : (40 * 20 + (3 * d / 2) * 40 * (3 * d / 2) = 40 * r^2)) :
  d = 3 + 3 / 8 :=
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l2072_207255


namespace NUMINAMATH_GPT_number_of_black_boxcars_l2072_207242

def red_boxcars : Nat := 3
def blue_boxcars : Nat := 4
def black_boxcar_capacity : Nat := 4000
def boxcar_total_capacity : Nat := 132000

def blue_boxcar_capacity : Nat := 2 * black_boxcar_capacity
def red_boxcar_capacity : Nat := 3 * blue_boxcar_capacity

def red_boxcar_total_capacity : Nat := red_boxcars * red_boxcar_capacity
def blue_boxcar_total_capacity : Nat := blue_boxcars * blue_boxcar_capacity

def other_total_capacity : Nat := red_boxcar_total_capacity + blue_boxcar_total_capacity
def remaining_capacity : Nat := boxcar_total_capacity - other_total_capacity
def expected_black_boxcars : Nat := remaining_capacity / black_boxcar_capacity

theorem number_of_black_boxcars :
  expected_black_boxcars = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_black_boxcars_l2072_207242


namespace NUMINAMATH_GPT_smallest_floor_sum_l2072_207266

theorem smallest_floor_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋) = 9 :=
sorry

end NUMINAMATH_GPT_smallest_floor_sum_l2072_207266


namespace NUMINAMATH_GPT_fraction_calculation_l2072_207231

theorem fraction_calculation : (8 / 24) - (5 / 72) + (3 / 8) = 23 / 36 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l2072_207231


namespace NUMINAMATH_GPT_no_solution_abs_eq_l2072_207270

theorem no_solution_abs_eq (x : ℝ) (h : x > 0) : |x + 4| = 3 - x → false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_abs_eq_l2072_207270


namespace NUMINAMATH_GPT_gcf_180_270_l2072_207230

theorem gcf_180_270 : Int.gcd 180 270 = 90 :=
sorry

end NUMINAMATH_GPT_gcf_180_270_l2072_207230


namespace NUMINAMATH_GPT_ordering_eight_four_three_l2072_207241

noncomputable def eight_pow_ten := 8 ^ 10
noncomputable def four_pow_fifteen := 4 ^ 15
noncomputable def three_pow_twenty := 3 ^ 20

theorem ordering_eight_four_three :
  eight_pow_ten < three_pow_twenty ∧ three_pow_twenty < four_pow_fifteen :=
by
  sorry

end NUMINAMATH_GPT_ordering_eight_four_three_l2072_207241


namespace NUMINAMATH_GPT_largest_natural_number_not_sum_of_two_composites_l2072_207250

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end NUMINAMATH_GPT_largest_natural_number_not_sum_of_two_composites_l2072_207250


namespace NUMINAMATH_GPT_g_50_unique_l2072_207258

namespace Proof

-- Define the function g and the condition it should satisfy
variable (g : ℕ → ℕ)
variable (h : ∀ (a b : ℕ), 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b))

theorem g_50_unique : ∃ (m t : ℕ), m * t = 0 := by
  -- Existence of m and t fulfilling the condition
  -- Placeholder for the proof
  sorry

end Proof

end NUMINAMATH_GPT_g_50_unique_l2072_207258


namespace NUMINAMATH_GPT_erasers_total_l2072_207211

-- Define the initial amount of erasers
def initialErasers : Float := 95.0

-- Define the amount of erasers Marie buys
def boughtErasers : Float := 42.0

-- Define the total number of erasers Marie ends with
def totalErasers : Float := 137.0

-- The theorem that needs to be proven
theorem erasers_total 
  (initial : Float := initialErasers)
  (bought : Float := boughtErasers)
  (total : Float := totalErasers) :
  initial + bought = total :=
sorry

end NUMINAMATH_GPT_erasers_total_l2072_207211


namespace NUMINAMATH_GPT_find_number_l2072_207291

theorem find_number : ∃ x : ℝ, 3550 - (1002 / x) = 3500 ∧ x = 20.04 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2072_207291


namespace NUMINAMATH_GPT_IntervalForKTriangleLengths_l2072_207224

noncomputable def f (x k : ℝ) := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem IntervalForKTriangleLengths (k : ℝ) :
  (∀ (x : ℝ), 1 ≤ f x k ∧
              (k ≥ 1 → f x k ≤ (k + 2) / 3) ∧ 
              (k < 1 → f x k ≥ (k + 2) / 3)) →
  (∀ (a b c : ℝ), (f a k < f b k + f c k) ∧ 
                  (f b k < f a k + f c k) ∧ 
                  (f c k < f a k + f b k)) ↔ (-1/2 < k ∧ k < 4) :=
by sorry

#check f
#check IntervalForKTriangleLengths

end NUMINAMATH_GPT_IntervalForKTriangleLengths_l2072_207224


namespace NUMINAMATH_GPT_domain_f_l2072_207223

noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 2) + 1 / (x - 3)

theorem domain_f :
  {x : ℝ | x ≥ 2 ∧ x ≠ 3 } = {x : ℝ | (2 ≤ x ∧ x < 3) ∨ (3 < x)} :=
by
  sorry

end NUMINAMATH_GPT_domain_f_l2072_207223


namespace NUMINAMATH_GPT_complex_number_purely_imaginary_l2072_207247

theorem complex_number_purely_imaginary (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 1 ≠ 0) → m = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_complex_number_purely_imaginary_l2072_207247


namespace NUMINAMATH_GPT_second_hand_angle_after_2_minutes_l2072_207234

theorem second_hand_angle_after_2_minutes :
  ∀ angle_in_radians, (∀ rotations:ℝ, rotations = 2 → one_full_circle = 2 * Real.pi → angle_in_radians = - (rotations * one_full_circle)) →
  angle_in_radians = -4 * Real.pi :=
by
  intros
  sorry

end NUMINAMATH_GPT_second_hand_angle_after_2_minutes_l2072_207234


namespace NUMINAMATH_GPT_sum_of_coefficients_condition_l2072_207217

theorem sum_of_coefficients_condition 
  (t : ℕ → ℤ) 
  (d e f : ℤ) 
  (h0 : t 0 = 3) 
  (h1 : t 1 = 7) 
  (h2 : t 2 = 17) 
  (h3 : t 3 = 86)
  (rec_relation : ∀ k ≥ 2, t (k + 1) = d * t k + e * t (k - 1) + f * t (k - 2)) : 
  d + e + f = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_condition_l2072_207217


namespace NUMINAMATH_GPT_parallel_lines_slope_l2072_207251

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l2072_207251


namespace NUMINAMATH_GPT_sum_of_digits_third_smallest_multiple_l2072_207281

noncomputable def LCM_upto_7 : ℕ := Nat.lcm (Nat.lcm 1 2) (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))

noncomputable def third_smallest_multiple : ℕ := 3 * LCM_upto_7

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_third_smallest_multiple : sum_of_digits third_smallest_multiple = 9 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_third_smallest_multiple_l2072_207281


namespace NUMINAMATH_GPT_virginia_more_years_l2072_207243

variable {V A D x : ℕ}

theorem virginia_more_years (h1 : V + A + D = 75) (h2 : D = 34) (h3 : V = A + x) (h4 : V = D - x) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_virginia_more_years_l2072_207243


namespace NUMINAMATH_GPT_fraction_subtraction_simplified_l2072_207289

theorem fraction_subtraction_simplified : (7 / 17) - (4 / 51) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_simplified_l2072_207289


namespace NUMINAMATH_GPT_cuboid_unshaded_face_area_l2072_207276

theorem cuboid_unshaded_face_area 
  (x : ℝ)
  (h1 : ∀ a  : ℝ, a = 4*x) -- Condition: each unshaded face area = 4 * shaded face area
  (h2 : 18*x = 72)         -- Condition: total surface area = 72 cm²
  : 4*x = 16 :=            -- Conclusion: area of one visible unshaded face is 16 cm²
by
  sorry

end NUMINAMATH_GPT_cuboid_unshaded_face_area_l2072_207276


namespace NUMINAMATH_GPT_digit_sum_solution_l2072_207228

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end NUMINAMATH_GPT_digit_sum_solution_l2072_207228


namespace NUMINAMATH_GPT_sums_solved_correctly_l2072_207290

theorem sums_solved_correctly (x : ℕ) (h : x + 2 * x = 48) : x = 16 := by
  sorry

end NUMINAMATH_GPT_sums_solved_correctly_l2072_207290


namespace NUMINAMATH_GPT_negation_example_l2072_207202

theorem negation_example : 
  (¬ ∃ x_0 : ℚ, x_0 - 2 = 0) = (∀ x : ℚ, x - 2 ≠ 0) :=
by 
  sorry

end NUMINAMATH_GPT_negation_example_l2072_207202


namespace NUMINAMATH_GPT_am_gm_problem_l2072_207284

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end NUMINAMATH_GPT_am_gm_problem_l2072_207284
