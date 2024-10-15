import Mathlib

namespace NUMINAMATH_GPT_total_employees_l1666_166663

theorem total_employees (x : Nat) (h1 : x < 13) : 13 + 6 * x = 85 :=
by
  sorry

end NUMINAMATH_GPT_total_employees_l1666_166663


namespace NUMINAMATH_GPT_geometric_sequence_property_l1666_166644

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∀ m n : ℕ, a (m + n) = a m * a n / a 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h : geometric_sequence a) 
    (h4 : a 4 = 5) 
    (h8 : a 8 = 6) : 
    a 2 * a 10 = 30 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l1666_166644


namespace NUMINAMATH_GPT_option_a_option_d_l1666_166667

theorem option_a (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m = Nat.choose n (n - m) := 
sorry

theorem option_d (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m := 
sorry

end NUMINAMATH_GPT_option_a_option_d_l1666_166667


namespace NUMINAMATH_GPT_average_people_moving_l1666_166608

theorem average_people_moving (days : ℕ) (total_people : ℕ) 
    (h_days : days = 5) (h_total_people : total_people = 3500) : 
    (total_people / days) = 700 :=
by
  sorry

end NUMINAMATH_GPT_average_people_moving_l1666_166608


namespace NUMINAMATH_GPT_num_ways_to_make_change_l1666_166627

-- Define the standard U.S. coins
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the total amount
def total_amount : ℕ := 50

-- Condition to exclude two quarters
def valid_combination (num_pennies num_nickels num_dimes num_quarters : ℕ) : Prop :=
  (num_quarters != 2) ∧ (num_pennies + 5 * num_nickels + 10 * num_dimes + 25 * num_quarters = total_amount)

-- Prove that there are 39 ways to make change for 50 cents
theorem num_ways_to_make_change : 
  ∃ count : ℕ, count = 39 ∧ (∀ 
    (num_pennies num_nickels num_dimes num_quarters : ℕ),
    valid_combination num_pennies num_nickels num_dimes num_quarters → 
    (num_pennies, num_nickels, num_dimes, num_quarters) = count) :=
sorry

end NUMINAMATH_GPT_num_ways_to_make_change_l1666_166627


namespace NUMINAMATH_GPT_solution_exists_l1666_166642

noncomputable def find_p_q : Prop :=
  ∃ p q : ℕ, (p^q - q^p = 1927) ∧ (p = 2611) ∧ (q = 11)

theorem solution_exists : find_p_q :=
sorry

end NUMINAMATH_GPT_solution_exists_l1666_166642


namespace NUMINAMATH_GPT_f_20_equals_97_l1666_166623

noncomputable def f_rec (f : ℕ → ℝ) (n : ℕ) := (2 * f n + n) / 2

theorem f_20_equals_97 (f : ℕ → ℝ) (h₁ : f 1 = 2)
    (h₂ : ∀ n : ℕ, f (n + 1) = f_rec f n) : 
    f 20 = 97 :=
sorry

end NUMINAMATH_GPT_f_20_equals_97_l1666_166623


namespace NUMINAMATH_GPT_alcohol_by_volume_l1666_166668

/-- Solution x is 10% alcohol by volume and is 50 ml.
    Solution y is 30% alcohol by volume and is 150 ml.
    We must prove the final solution is 25% alcohol by volume. -/
theorem alcohol_by_volume (vol_x vol_y : ℕ) (conc_x conc_y : ℕ) (vol_mix : ℕ) (conc_mix : ℕ) :
  vol_x = 50 →
  conc_x = 10 →
  vol_y = 150 →
  conc_y = 30 →
  vol_mix = vol_x + vol_y →
  conc_mix = 100 * (vol_x * conc_x + vol_y * conc_y) / vol_mix →
  conc_mix = 25 :=
by
  intros h1 h2 h3 h4 h5 h_cons
  sorry

end NUMINAMATH_GPT_alcohol_by_volume_l1666_166668


namespace NUMINAMATH_GPT_large_beds_l1666_166687

theorem large_beds {L : ℕ} {M : ℕ} 
    (h1 : M = 2) 
    (h2 : ∀ (x : ℕ), 100 <= x → L = (320 - 60 * M) / 100) : 
  L = 2 :=
by
  sorry

end NUMINAMATH_GPT_large_beds_l1666_166687


namespace NUMINAMATH_GPT_total_hours_charged_l1666_166610

theorem total_hours_charged (K P M : ℕ) 
  (h₁ : P = 2 * K)
  (h₂ : P = (1 / 3 : ℚ) * (K + 80))
  (h₃ : M = K + 80) : K + P + M = 144 :=
by {
    sorry
}

end NUMINAMATH_GPT_total_hours_charged_l1666_166610


namespace NUMINAMATH_GPT_scale_of_map_l1666_166698

theorem scale_of_map 
  (map_distance : ℝ)
  (travel_time : ℝ)
  (average_speed : ℝ)
  (actual_distance : ℝ)
  (scale : ℝ)
  (h1 : map_distance = 5)
  (h2 : travel_time = 6.5)
  (h3 : average_speed = 60)
  (h4 : actual_distance = average_speed * travel_time)
  (h5 : scale = map_distance / actual_distance) :
  scale = 0.01282 :=
by
  sorry

end NUMINAMATH_GPT_scale_of_map_l1666_166698


namespace NUMINAMATH_GPT_select_student_B_l1666_166678

-- Define the average scores for the students A, B, C, D
def avg_A : ℝ := 85
def avg_B : ℝ := 90
def avg_C : ℝ := 90
def avg_D : ℝ := 85

-- Define the variances for the students A, B, C, D
def var_A : ℝ := 50
def var_B : ℝ := 42
def var_C : ℝ := 50
def var_D : ℝ := 42

-- Theorem stating the selected student should be B
theorem select_student_B (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ)
  (h_avg_A : avg_A = 85) (h_avg_B : avg_B = 90) (h_avg_C : avg_C = 90) (h_avg_D : avg_D = 85)
  (h_var_A : var_A = 50) (h_var_B : var_B = 42) (h_var_C : var_C = 50) (h_var_D : var_D = 42) :
  (avg_B = 90 ∧ avg_C = 90 ∧ avg_B ≥ avg_A ∧ avg_B ≥ avg_D ∧ var_B < var_C) → 
  (select_student = "B") :=
by
  sorry

end NUMINAMATH_GPT_select_student_B_l1666_166678


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l1666_166620

-- Part a
theorem part_a (x : ℝ) : 
  (5 / x - x / 3 = 1 / 6) ↔ x = 6 := 
by
  sorry

-- Part b
theorem part_b (a : ℝ) : 
  ¬ ∃ a, (1 / 2 + a / 4 = a / 4) := 
by
  sorry

-- Part c
theorem part_c (y : ℝ) : 
  (9 / y - y / 21 = 17 / 21) ↔ y = 7 := 
by
  sorry

-- Part d
theorem part_d (z : ℝ) : 
  (z / 8 - 1 / z = 3 / 8) ↔ z = 4 := 
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l1666_166620


namespace NUMINAMATH_GPT_part_i_l1666_166631

theorem part_i (n : ℕ) (h₁ : n ≥ 1) (h₂ : n ∣ (2^n - 1)) : n = 1 :=
sorry

end NUMINAMATH_GPT_part_i_l1666_166631


namespace NUMINAMATH_GPT_interest_rate_per_annum_l1666_166602

variable (P : ℝ := 1200) (T : ℝ := 1) (diff : ℝ := 2.999999999999936) (r : ℝ)
noncomputable def SI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * r * T
noncomputable def CI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * ((1 + r / 2) ^ (2 * T) - 1)

theorem interest_rate_per_annum :
  CI P r T - SI P r T = diff → r = 0.1 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l1666_166602


namespace NUMINAMATH_GPT_Carla_servings_l1666_166662

-- Define the volumes involved
def volume_watermelon : ℕ := 500
def volume_cream : ℕ := 100
def volume_per_serving : ℕ := 150

-- The total volume is the sum of the watermelon and cream volumes
def total_volume : ℕ := volume_watermelon + volume_cream

-- The number of servings is the total volume divided by the volume per serving
def n_servings : ℕ := total_volume / volume_per_serving

-- The theorem to prove that Carla can make 4 servings of smoothies
theorem Carla_servings : n_servings = 4 := by
  sorry

end NUMINAMATH_GPT_Carla_servings_l1666_166662


namespace NUMINAMATH_GPT_tan_of_cos_neg_five_thirteenth_l1666_166696

variable {α : Real}

theorem tan_of_cos_neg_five_thirteenth (hcos : Real.cos α = -5/13) (hα : π < α ∧ α < 3 * π / 2) : 
  Real.tan α = 12 / 5 := 
sorry

end NUMINAMATH_GPT_tan_of_cos_neg_five_thirteenth_l1666_166696


namespace NUMINAMATH_GPT_emily_lemon_juice_fraction_l1666_166626

/-- 
Emily places 6 ounces of tea into a twelve-ounce cup and 6 ounces of honey into a second cup
of the same size. Then she adds 3 ounces of lemon juice to the second cup. Next, she pours half
the tea from the first cup into the second, mixes thoroughly, and then pours one-third of the
mixture in the second cup back into the first. 
Prove that the fraction of the mixture in the first cup that is lemon juice is 1/7.
--/
theorem emily_lemon_juice_fraction :
  let cup1_tea := 6
  let cup2_honey := 6
  let cup2_lemon_juice := 3
  let cup1_tea_transferred := cup1_tea / 2
  let cup1 := cup1_tea - cup1_tea_transferred
  let cup2 := cup2_honey + cup2_lemon_juice + cup1_tea_transferred
  let mix_ratio (x y : ℕ) := (x : ℚ) / (x + y)
  let cup1_after_transfer := cup1 + (cup2 / 3)
  let cup2_tea := cup1_tea_transferred
  let cup2_honey := cup2_honey
  let cup2_lemon_juice := cup2_lemon_juice
  let cup1_lemon_transferred := 1
  cup1_tea + (cup2 / 3) = 3 + (cup2_tea * (1 / 3)) + 1 + (cup2_honey * (1 / 3)) + cup2_lemon_juice / 3 →
  cup1 / (cup1 + cup2_honey) = 1/7 :=
sorry

end NUMINAMATH_GPT_emily_lemon_juice_fraction_l1666_166626


namespace NUMINAMATH_GPT_samson_fuel_calculation_l1666_166672

def total_fuel_needed (main_distance : ℕ) (fuel_rate : ℕ) (hilly_distance : ℕ) (hilly_increase : ℚ)
                      (detours : ℕ) (detour_distance : ℕ) : ℚ :=
  let normal_distance := main_distance - hilly_distance
  let normal_fuel := (fuel_rate / 70) * normal_distance
  let hilly_fuel := (fuel_rate / 70) * hilly_distance * hilly_increase
  let detour_fuel := (fuel_rate / 70) * (detours * detour_distance)
  normal_fuel + hilly_fuel + detour_fuel

theorem samson_fuel_calculation :
  total_fuel_needed 140 10 30 1.2 2 5 = 22.28 :=
by sorry

end NUMINAMATH_GPT_samson_fuel_calculation_l1666_166672


namespace NUMINAMATH_GPT_smallest_geometric_number_l1666_166686

noncomputable def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def is_smallest_geometric_number (n : ℕ) : Prop :=
  n = 261

theorem smallest_geometric_number :
  ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10)) ∧
  (n / 100 = 2) ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  is_smallest_geometric_number n :=
by
  sorry

end NUMINAMATH_GPT_smallest_geometric_number_l1666_166686


namespace NUMINAMATH_GPT_difference_between_hit_and_unreleased_l1666_166684

-- Define the conditions as constants
def hit_songs : Nat := 25
def top_100_songs : Nat := hit_songs + 10
def total_songs : Nat := 80

-- Define the question, conditional on the definitions above
theorem difference_between_hit_and_unreleased : 
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_hit_and_unreleased_l1666_166684


namespace NUMINAMATH_GPT_mutually_exclusive_complementary_event_l1666_166614

-- Definitions of events
def hitting_target_at_least_once (shots: ℕ) : Prop := shots > 0
def not_hitting_target_at_all (shots: ℕ) : Prop := shots = 0

-- The statement to prove
theorem mutually_exclusive_complementary_event : 
  ∀ (shots: ℕ), (not_hitting_target_at_all shots ↔ ¬ hitting_target_at_least_once shots) :=
by 
  sorry

end NUMINAMATH_GPT_mutually_exclusive_complementary_event_l1666_166614


namespace NUMINAMATH_GPT_bacteria_reach_target_l1666_166621

def bacteria_growth (initial : ℕ) (target : ℕ) (doubling_time : ℕ) (delay : ℕ) : ℕ :=
  let doubling_count := Nat.log2 (target / initial)
  doubling_count * doubling_time + delay

theorem bacteria_reach_target : 
  bacteria_growth 800 25600 5 3 = 28 := by
  sorry

end NUMINAMATH_GPT_bacteria_reach_target_l1666_166621


namespace NUMINAMATH_GPT_find_a_value_l1666_166659

theorem find_a_value (a x y : ℝ) :
  (|y| + |y - x| ≤ a - |x - 1| ∧ (y - 4) * (y + 3) ≥ (4 - x) * (3 + x)) → a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l1666_166659


namespace NUMINAMATH_GPT_range_of_a_l1666_166681

open Real

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * exp x1 - a = 0) ∧ (x2 * exp x2 - a = 0)) ↔ -1 / exp 1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1666_166681


namespace NUMINAMATH_GPT_arithmetic_mean_a8_a11_l1666_166646

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_a8_a11 {a : ℕ → ℝ} (h1 : geometric_sequence a (-2)) 
    (h2 : a 2 * a 6 = 4 * a 3) :
  ((a 7 + a 10) / 2) = -56 :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_a8_a11_l1666_166646


namespace NUMINAMATH_GPT_enrollment_difference_l1666_166664

theorem enrollment_difference 
  (Varsity_enrollment : ℕ)
  (Northwest_enrollment : ℕ)
  (Central_enrollment : ℕ)
  (Greenbriar_enrollment : ℕ) 
  (h1 : Varsity_enrollment = 1300) 
  (h2 : Northwest_enrollment = 1500)
  (h3 : Central_enrollment = 1800)
  (h4 : Greenbriar_enrollment = 1600) : 
  Varsity_enrollment < Northwest_enrollment ∧ 
  Northwest_enrollment < Greenbriar_enrollment ∧ 
  Greenbriar_enrollment < Central_enrollment → 
    (Greenbriar_enrollment - Varsity_enrollment = 300) :=
by
  sorry

end NUMINAMATH_GPT_enrollment_difference_l1666_166664


namespace NUMINAMATH_GPT_complex_expression_l1666_166638

theorem complex_expression (z : ℂ) (h : z = (i + 1) / (i - 1)) : z^2 + z + 1 = -i := 
by 
  sorry

end NUMINAMATH_GPT_complex_expression_l1666_166638


namespace NUMINAMATH_GPT_find_sum_l1666_166630

theorem find_sum (a b : ℝ) 
  (h₁ : (a + Real.sqrt b) + (a - Real.sqrt b) = -8) 
  (h₂ : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) : 
  a + b = 8 := 
sorry

end NUMINAMATH_GPT_find_sum_l1666_166630


namespace NUMINAMATH_GPT_percentage_below_cost_l1666_166691

variable (CP SP : ℝ)

-- Given conditions
def cost_price : ℝ := 5625
def more_for_profit : ℝ := 1800
def profit_percentage : ℝ := 0.16
def expected_SP : ℝ := cost_price + (cost_price * profit_percentage)
def actual_SP : ℝ := expected_SP - more_for_profit

-- Statement to prove
theorem percentage_below_cost (h1 : CP = cost_price) (h2 : SP = actual_SP) :
  (CP - SP) / CP * 100 = 16 := by
sorry

end NUMINAMATH_GPT_percentage_below_cost_l1666_166691


namespace NUMINAMATH_GPT_candies_remaining_l1666_166636

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end NUMINAMATH_GPT_candies_remaining_l1666_166636


namespace NUMINAMATH_GPT_find_m_l1666_166649

theorem find_m (m : ℝ) (A : Set ℝ) (B : Set ℝ) (hA : A = { -1, 2, 2 * m - 1 }) (hB : B = { 2, m^2 }) (hSubset : B ⊆ A) : m = 1 := by
  sorry
 
end NUMINAMATH_GPT_find_m_l1666_166649


namespace NUMINAMATH_GPT_manuscript_age_in_decimal_l1666_166670

-- Given conditions
def octal_number : ℕ := 12345

-- Translate the problem statement into Lean:
theorem manuscript_age_in_decimal : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 :=
by
  sorry

end NUMINAMATH_GPT_manuscript_age_in_decimal_l1666_166670


namespace NUMINAMATH_GPT_valid_four_digit_numbers_count_l1666_166690

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end NUMINAMATH_GPT_valid_four_digit_numbers_count_l1666_166690


namespace NUMINAMATH_GPT_original_prices_correct_l1666_166609

-- Define the problem conditions
def Shirt_A_discount1 := 0.10
def Shirt_A_discount2 := 0.20
def Shirt_A_final_price := 420

def Shirt_B_discount1 := 0.15
def Shirt_B_discount2 := 0.25
def Shirt_B_final_price := 405

def Shirt_C_discount1 := 0.05
def Shirt_C_discount2 := 0.15
def Shirt_C_final_price := 680

def sales_tax := 0.05

-- Define the original prices for each shirt.
def original_price_A := 420 / (0.9 * 0.8)
def original_price_B := 405 / (0.85 * 0.75)
def original_price_C := 680 / (0.95 * 0.85)

-- Prove the original prices of the shirts
theorem original_prices_correct:
  original_price_A = 583.33 ∧ 
  original_price_B = 635 ∧ 
  original_price_C = 842.24 := 
by
  sorry

end NUMINAMATH_GPT_original_prices_correct_l1666_166609


namespace NUMINAMATH_GPT_greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l1666_166660

-- Define the given conditions
def totalOranges : ℕ := 81
def totalCookies : ℕ := 65
def numberOfChildren : ℕ := 7

-- Define the floor division for children
def orangesPerChild : ℕ := totalOranges / numberOfChildren
def cookiesPerChild : ℕ := totalCookies / numberOfChildren

-- Calculate leftover (donated) quantities
def orangesLeftover : ℕ := totalOranges % numberOfChildren
def cookiesLeftover : ℕ := totalCookies % numberOfChildren

-- Statements to prove
theorem greatest_number_of_donated_oranges : orangesLeftover = 4 := by {
    sorry
}

theorem greatest_number_of_donated_cookies : cookiesLeftover = 2 := by {
    sorry
}

end NUMINAMATH_GPT_greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l1666_166660


namespace NUMINAMATH_GPT_point_coordinates_l1666_166695

theorem point_coordinates (M : ℝ × ℝ) 
  (hx : abs M.2 = 3) 
  (hy : abs M.1 = 2) 
  (h_first_quadrant : 0 < M.1 ∧ 0 < M.2) : 
  M = (2, 3) := 
sorry

end NUMINAMATH_GPT_point_coordinates_l1666_166695


namespace NUMINAMATH_GPT_strawberries_for_mom_l1666_166641

-- Define the conditions as Lean definitions
def dozen : ℕ := 12
def strawberries_picked : ℕ := 2 * dozen
def strawberries_eaten : ℕ := 6

-- Define the statement to be proven
theorem strawberries_for_mom : (strawberries_picked - strawberries_eaten) = 18 := by
  sorry

end NUMINAMATH_GPT_strawberries_for_mom_l1666_166641


namespace NUMINAMATH_GPT_max_f_alpha_side_a_l1666_166634

noncomputable def a_vec (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)
noncomputable def b_vec (α : ℝ) : ℝ × ℝ := (6 * Real.sin α + Real.cos α, 7 * Real.sin α - 2 * Real.cos α)

noncomputable def f (α : ℝ) : ℝ := (a_vec α).1 * (b_vec α).1 + (a_vec α).2 * (b_vec α).2

theorem max_f_alpha : ∀ α : ℝ, f α ≤ 4 * Real.sqrt 2 + 2 :=
by
sorry

theorem side_a (A : ℝ) (b c : ℝ) (h1 : f A = 6) (h2 : 1/2 * b * c * Real.sin A = 3) (h3 : b + c = 2 + 3 * Real.sqrt 2) : 
  ∃ a : ℝ, a = Real.sqrt 10 :=
by
sorry

end NUMINAMATH_GPT_max_f_alpha_side_a_l1666_166634


namespace NUMINAMATH_GPT_parabola_distance_ratio_l1666_166633

open Real

theorem parabola_distance_ratio (p : ℝ) (M N : ℝ × ℝ)
  (h1 : p = 4)
  (h2 : M.snd ^ 2 = 2 * p * M.fst)
  (h3 : N.snd ^ 2 = 2 * p * N.fst)
  (h4 : (M.snd - 2 * N.snd) * (M.snd + 2 * N.snd) = 48) :
  |M.fst + 2| = 4 * |N.fst + 2| := sorry

end NUMINAMATH_GPT_parabola_distance_ratio_l1666_166633


namespace NUMINAMATH_GPT_eq1_solutions_eq2_solutions_l1666_166693

theorem eq1_solutions (x : ℝ) : x^2 - 6 * x + 3 = 0 ↔ (x = 3 + Real.sqrt 6) ∨ (x = 3 - Real.sqrt 6) :=
by {
  sorry
}

theorem eq2_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ (x = 2) ∨ (x = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_eq1_solutions_eq2_solutions_l1666_166693


namespace NUMINAMATH_GPT_find_number_A_l1666_166625

theorem find_number_A (A B : ℝ) (h₁ : A + B = 14.85) (h₂ : B = 10 * A) : A = 1.35 :=
sorry

end NUMINAMATH_GPT_find_number_A_l1666_166625


namespace NUMINAMATH_GPT_original_price_l1666_166665

theorem original_price (P S : ℝ) (h1 : S = 1.25 * P) (h2 : S - P = 625) : P = 2500 := by
  sorry

end NUMINAMATH_GPT_original_price_l1666_166665


namespace NUMINAMATH_GPT_lily_jog_time_l1666_166675

theorem lily_jog_time :
  (∃ (max_time : ℕ) (lily_miles_max : ℕ) (max_distance : ℕ) (lily_time_ratio : ℕ) (distance_wanted : ℕ)
      (expected_time : ℕ),
    max_time = 36 ∧
    lily_miles_max = 4 ∧
    max_distance = 6 ∧
    lily_time_ratio = 3 ∧
    distance_wanted = 7 ∧
    expected_time = 21 ∧
    lily_miles_max * lily_time_ratio = max_time ∧
    max_distance * lily_time_ratio = distance_wanted * expected_time) := 
sorry

end NUMINAMATH_GPT_lily_jog_time_l1666_166675


namespace NUMINAMATH_GPT_intersection_M_N_eq_l1666_166655

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N based on the given inequality
def N : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

-- The statement we want to prove
theorem intersection_M_N_eq {M N: Set ℝ} (hm: M = {0, 1, 2}) 
  (hn: N = {x | x^2 - 3 * x + 2 ≤ 0}) : 
  M ∩ N = {1, 2} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_eq_l1666_166655


namespace NUMINAMATH_GPT_number_of_tickets_l1666_166600

-- Define the given conditions
def initial_premium := 50 -- dollars per month
def premium_increase_accident (initial_premium : ℕ) := initial_premium / 10 -- 10% increase
def premium_increase_ticket := 5 -- dollars per month per ticket
def num_accidents := 1
def new_premium := 70 -- dollars per month

-- Define the target question
theorem number_of_tickets (tickets : ℕ) :
  initial_premium + premium_increase_accident initial_premium * num_accidents + premium_increase_ticket * tickets = new_premium → 
  tickets = 3 :=
by
   sorry

end NUMINAMATH_GPT_number_of_tickets_l1666_166600


namespace NUMINAMATH_GPT_percent_round_trip_tickets_l1666_166683

-- Define the main variables
variables (P R : ℝ)

-- Define the conditions based on the problem statement
def condition1 : Prop := 0.3 * P = 0.3 * R
 
-- State the theorem to prove
theorem percent_round_trip_tickets (h1 : condition1 P R) : R / P * 100 = 30 := by sorry

end NUMINAMATH_GPT_percent_round_trip_tickets_l1666_166683


namespace NUMINAMATH_GPT_question1_question2_question3_l1666_166689

variables {a x1 x2 : ℝ}

-- Definition of the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := a * x^2 + x + 1

-- Conditions
axiom a_positive : a > 0
axiom roots_exist : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0
axiom roots_real : x1 + x2 = -1 / a ∧ x1 * x2 = 1 / a

-- Question 1
theorem question1 : (1 + x1) * (1 + x2) = 1 :=
sorry

-- Question 2
theorem question2 : x1 < -1 ∧ x2 < -1 :=
sorry

-- Additional condition for question 3
axiom ratio_in_range : x1 / x2 ∈ Set.Icc (1 / 10 : ℝ) 10

-- Question 3
theorem question3 : a <= 1 / 4 :=
sorry

end NUMINAMATH_GPT_question1_question2_question3_l1666_166689


namespace NUMINAMATH_GPT_pen_price_ratio_l1666_166661

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end NUMINAMATH_GPT_pen_price_ratio_l1666_166661


namespace NUMINAMATH_GPT_average_a_b_l1666_166645

-- Defining the variables A, B, C
variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- The theorem stating that the average weight of a and b is 40 kg
theorem average_a_b (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : (A + B) / 2 = 40 :=
sorry

end NUMINAMATH_GPT_average_a_b_l1666_166645


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1666_166605

open Set

variable (U : Set ℕ) (A : Set ℕ)

theorem complement_of_A_in_U (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 6}) :
  (U \ A) = {1, 3, 5} := by 
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1666_166605


namespace NUMINAMATH_GPT_proof_x_bounds_l1666_166669

noncomputable def x : ℝ :=
  1 / Real.logb (1 / 3) (1 / 2) +
  1 / Real.logb (1 / 3) (1 / 4) +
  1 / Real.logb 7 (1 / 8)

theorem proof_x_bounds : 3 < x ∧ x < 3.5 := 
by
  sorry

end NUMINAMATH_GPT_proof_x_bounds_l1666_166669


namespace NUMINAMATH_GPT_grazing_b_l1666_166612

theorem grazing_b (A_oxen_months B_oxen_months C_oxen_months total_months total_rent C_rent B_oxen : ℕ) 
  (hA : A_oxen_months = 10 * 7)
  (hB : B_oxen_months = B_oxen * 5)
  (hC : C_oxen_months = 15 * 3)
  (htotal : total_months = A_oxen_months + B_oxen_months + C_oxen_months)
  (hrent : total_rent = 175)
  (hC_rent : C_rent = 45)
  (hC_share : C_oxen_months / total_months = C_rent / total_rent) :
  B_oxen = 12 :=
by
  sorry

end NUMINAMATH_GPT_grazing_b_l1666_166612


namespace NUMINAMATH_GPT_pattern_equation_l1666_166616

theorem pattern_equation (n : ℕ) : n^2 + n = n * (n + 1) := 
  sorry

end NUMINAMATH_GPT_pattern_equation_l1666_166616


namespace NUMINAMATH_GPT_calculate_expression_l1666_166653

theorem calculate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + (1 / 6)) = 57 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1666_166653


namespace NUMINAMATH_GPT_problem_1_problem_2_l1666_166618

theorem problem_1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 :=
sorry

theorem problem_2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 4) : 
  (1/4) * a^2 + (1/9) * b^2 + c^2 = 8 / 7 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1666_166618


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1666_166699

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a n < a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  q = 2 := 
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1666_166699


namespace NUMINAMATH_GPT_pick_three_different_cards_in_order_l1666_166643

theorem pick_three_different_cards_in_order :
  (52 * 51 * 50) = 132600 :=
by
  sorry

end NUMINAMATH_GPT_pick_three_different_cards_in_order_l1666_166643


namespace NUMINAMATH_GPT_result_of_operation_given_y_l1666_166674

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem result_of_operation_given_y :
  ∀ (y : ℤ), y = 11 → operation y 10 = 90 :=
by
  intros y hy
  rw [hy]
  show operation 11 10 = 90
  sorry

end NUMINAMATH_GPT_result_of_operation_given_y_l1666_166674


namespace NUMINAMATH_GPT_Hillary_sunday_minutes_l1666_166656

variable (total_minutes friday_minutes saturday_minutes : ℕ)

theorem Hillary_sunday_minutes 
  (h_total : total_minutes = 60) 
  (h_friday : friday_minutes = 16) 
  (h_saturday : saturday_minutes = 28) : 
  ∃ sunday_minutes : ℕ, total_minutes - (friday_minutes + saturday_minutes) = sunday_minutes ∧ sunday_minutes = 16 := 
by
  sorry

end NUMINAMATH_GPT_Hillary_sunday_minutes_l1666_166656


namespace NUMINAMATH_GPT_convex_cyclic_quadrilaterals_perimeter_40_l1666_166613

theorem convex_cyclic_quadrilaterals_perimeter_40 :
  ∃ (n : ℕ), n = 750 ∧ ∀ (a b c d : ℕ), a + b + c + d = 40 → a ≥ b → b ≥ c → c ≥ d →
  (a < b + c + d) ∧ (b < a + c + d) ∧ (c < a + b + d) ∧ (d < a + b + c) :=
sorry

end NUMINAMATH_GPT_convex_cyclic_quadrilaterals_perimeter_40_l1666_166613


namespace NUMINAMATH_GPT_find_other_root_l1666_166654

theorem find_other_root (m : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, (x = -6 → (x^2 + m * x - 6 = 0))) → (x^2 + m * x - 6 = (x + 6) * (x - 1)) → (∀ x : ℝ, (x^2 + 5 * x - 6 = 0) → (x = -6 ∨ x = 1))) :=
sorry

end NUMINAMATH_GPT_find_other_root_l1666_166654


namespace NUMINAMATH_GPT_therapy_charge_l1666_166628

-- Define the charges
def first_hour_charge (S : ℝ) : ℝ := S + 50
def subsequent_hour_charge (S : ℝ) : ℝ := S

-- Define the total charge before service fee for 8 hours
def total_charge_8_hours_before_fee (F S : ℝ) : ℝ := F + 7 * S

-- Define the total charge including the service fee for 8 hours
def total_charge_8_hours (F S : ℝ) : ℝ := 1.10 * (F + 7 * S)

-- Define the total charge before service fee for 3 hours
def total_charge_3_hours_before_fee (F S : ℝ) : ℝ := F + 2 * S

-- Define the total charge including the service fee for 3 hours
def total_charge_3_hours (F S : ℝ) : ℝ := 1.10 * (F + 2 * S)

theorem therapy_charge (S F : ℝ) :
  (F = S + 50) → (1.10 * (F + 7 * S) = 900) → (1.10 * (F + 2 * S) = 371.87) :=
by {
  sorry
}

end NUMINAMATH_GPT_therapy_charge_l1666_166628


namespace NUMINAMATH_GPT_max_z_value_l1666_166637

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x * y + y * z + z * x = -3) : z ≤ 2 := sorry

end NUMINAMATH_GPT_max_z_value_l1666_166637


namespace NUMINAMATH_GPT_value_of_f_at_3_l1666_166601

noncomputable def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

theorem value_of_f_at_3 : f 3 = 155 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_3_l1666_166601


namespace NUMINAMATH_GPT_age_of_youngest_child_l1666_166666

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : x = 6 :=
sorry

end NUMINAMATH_GPT_age_of_youngest_child_l1666_166666


namespace NUMINAMATH_GPT_two_common_points_with_x_axis_l1666_166658

noncomputable def func (x d : ℝ) : ℝ := x^3 - 3 * x + d

theorem two_common_points_with_x_axis (d : ℝ) :
(∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func x1 d = 0 ∧ func x2 d = 0) ↔ (d = 2 ∨ d = -2) :=
by
  sorry

end NUMINAMATH_GPT_two_common_points_with_x_axis_l1666_166658


namespace NUMINAMATH_GPT_solve_for_s_l1666_166611

theorem solve_for_s :
  let numerator := Real.sqrt (7^2 + 24^2)
  let denominator := Real.sqrt (64 + 36)
  let s := numerator / denominator
  s = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_s_l1666_166611


namespace NUMINAMATH_GPT_total_tiles_l1666_166624

theorem total_tiles (n : ℕ) (h : 3 * n - 2 = 55) : n^2 = 361 :=
by
  sorry

end NUMINAMATH_GPT_total_tiles_l1666_166624


namespace NUMINAMATH_GPT_find_k_l1666_166677

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

theorem find_k (k : ℝ) :
  let a : vector := (2, 3)
  let b : vector := (1, 4)
  let c : vector := (k, 3)
  orthogonal (a.1 + b.1, a.2 + b.2) c → k = -7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_k_l1666_166677


namespace NUMINAMATH_GPT_radio_price_and_total_items_l1666_166607

theorem radio_price_and_total_items :
  ∃ (n : ℕ) (p : ℝ),
    (∀ (i : ℕ), (1 ≤ i ∧ i ≤ n) → (i = 1 ∨ ∃ (j : ℕ), i = j + 1 ∧ p = 1 + (j * 0.50))) ∧
    (n - 49 = 85) ∧
    (p = 43) ∧
    (n = 134) :=
by {
  sorry
}

end NUMINAMATH_GPT_radio_price_and_total_items_l1666_166607


namespace NUMINAMATH_GPT_convert_to_spherical_l1666_166685

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then 3 * Real.pi / 2
           else if x > 0 then Real.arctan (y / x)
           else if y >= 0 then Real.arctan (y / x) + Real.pi
           else Real.arctan (y / x) - Real.pi
  (ρ, θ, φ)

theorem convert_to_spherical :
  rectangular_to_spherical (3 * Real.sqrt 2) (-4) 5 =
  (Real.sqrt 59, 2 * Real.pi + Real.arctan ((-4) / (3 * Real.sqrt 2)), Real.arccos (5 / Real.sqrt 59)) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_spherical_l1666_166685


namespace NUMINAMATH_GPT_company_members_and_days_l1666_166692

theorem company_members_and_days {t n : ℕ} (h : t = 6) :
    n = (t * (t - 1)) / 2 → n = 15 :=
by
  intro hn
  rw [h] at hn
  simp at hn
  exact hn

end NUMINAMATH_GPT_company_members_and_days_l1666_166692


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1666_166615

theorem solution_set_of_inequality 
  {f : ℝ → ℝ}
  (hf : ∀ x y : ℝ, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  {x : ℝ | |f (x - 2)| > 2 } = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1666_166615


namespace NUMINAMATH_GPT_cream_ratio_l1666_166694

theorem cream_ratio (john_coffee_initial jane_coffee_initial : ℕ)
  (john_drank john_added_cream jane_added_cream jane_drank : ℕ) :
  john_coffee_initial = 20 →
  jane_coffee_initial = 20 →
  john_drank = 3 →
  john_added_cream = 4 →
  jane_added_cream = 3 →
  jane_drank = 5 →
  john_added_cream / (jane_added_cream * 18 / (23 * 1)) = (46 / 27) := 
by
  intros
  sorry

end NUMINAMATH_GPT_cream_ratio_l1666_166694


namespace NUMINAMATH_GPT_square_of_binomial_l1666_166652

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end NUMINAMATH_GPT_square_of_binomial_l1666_166652


namespace NUMINAMATH_GPT_horizontal_shift_equivalence_l1666_166671

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)
noncomputable def resulting_function (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem horizontal_shift_equivalence :
  ∀ x : ℝ, resulting_function x = original_function (x + Real.pi / 3) :=
by sorry

end NUMINAMATH_GPT_horizontal_shift_equivalence_l1666_166671


namespace NUMINAMATH_GPT_trig_expression_value_l1666_166617

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
sorry

end NUMINAMATH_GPT_trig_expression_value_l1666_166617


namespace NUMINAMATH_GPT_smallest_n_l1666_166604

def in_interval (x y z : ℝ) (n : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ n ∧ 2 ≤ y ∧ y ≤ n ∧ 2 ≤ z ∧ z ≤ n

def no_two_within_one_unit (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 1 ∧ abs (y - z) ≥ 1 ∧ abs (z - x) ≥ 1

def more_than_two_units_apart (x y z : ℝ) (n : ℕ) : Prop :=
  x > 2 ∧ x < n - 2 ∧ y > 2 ∧ y < n - 2 ∧ z > 2 ∧ z < n - 2

def probability_condition (n : ℕ) : Prop :=
  (n-4)^3 / (n-2)^3 > 1/3

theorem smallest_n (n : ℕ) : 11 = n → (∃ x y z : ℝ, in_interval x y z n ∧ no_two_within_one_unit x y z ∧ more_than_two_units_apart x y z n ∧ probability_condition n) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1666_166604


namespace NUMINAMATH_GPT_inequality1_inequality2_l1666_166640

theorem inequality1 (x : ℝ) : 2 * x - 1 > x - 3 → x > -2 := by
  sorry

theorem inequality2 (x : ℝ) : 
  (x - 3 * (x - 2) ≥ 4) ∧ ((x - 1) / 5 < (x + 1) / 2) → -7 / 3 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_inequality1_inequality2_l1666_166640


namespace NUMINAMATH_GPT_total_pages_book_l1666_166680

-- Define the conditions
def reading_speed1 : ℕ := 10 -- pages per day for first half
def reading_speed2 : ℕ := 5 -- pages per day for second half
def total_days : ℕ := 75 -- total days spent reading

-- This is the main theorem we seek to prove:
theorem total_pages_book (P : ℕ) 
  (h1 : ∃ D1 D2 : ℕ, D1 + D2 = total_days ∧ D1 * reading_speed1 = P / 2 ∧ D2 * reading_speed2 = P / 2) : 
  P = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_book_l1666_166680


namespace NUMINAMATH_GPT_fixed_monthly_fee_l1666_166603

variable (x y : Real)

theorem fixed_monthly_fee :
  (x + y = 15.30) →
  (x + 1.5 * y = 20.55) →
  (x = 4.80) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l1666_166603


namespace NUMINAMATH_GPT_greatest_value_x_is_correct_l1666_166688

noncomputable def greatest_value_x : ℝ :=
-8 + Real.sqrt 6

theorem greatest_value_x_is_correct :
  ∀ x : ℝ, (x ≠ 9) → ((x^2 - x - 90) / (x - 9) = 2 / (x + 6)) → x ≤ greatest_value_x :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_x_is_correct_l1666_166688


namespace NUMINAMATH_GPT_find_angle_B_l1666_166629

def is_triangle (A B C : ℝ) : Prop :=
A + B > C ∧ B + C > A ∧ C + A > B

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Defining the problem conditions
lemma given_condition : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a := sorry
-- A triangle with sides a, b, c
lemma triangle_property : is_triangle a b c := sorry

-- The equivalent proof problem
theorem find_angle_B (h_triangle : is_triangle a b c) (h_cond : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) : 
    B = π / 6 := sorry

end NUMINAMATH_GPT_find_angle_B_l1666_166629


namespace NUMINAMATH_GPT_find_percentage_l1666_166635

noncomputable def percentage_solve (x : ℝ) : Prop :=
  0.15 * 40 = (x / 100) * 16 + 2

theorem find_percentage (x : ℝ) (h : percentage_solve x) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1666_166635


namespace NUMINAMATH_GPT_green_marbles_l1666_166682

theorem green_marbles :
  ∀ (total: ℕ) (blue: ℕ) (red: ℕ) (yellow: ℕ), 
  total = 164 →
  blue = total / 2 →
  red = total / 4 →
  yellow = 14 →
  (total - (blue + red + yellow)) = 27 :=
by
  intros total blue red yellow h_total h_blue h_red h_yellow
  sorry

end NUMINAMATH_GPT_green_marbles_l1666_166682


namespace NUMINAMATH_GPT_fruit_basket_l1666_166697

theorem fruit_basket :
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  oranges + apples + bananas + peaches = 28 :=
by
  intros oranges apples bananas peaches h_oranges h_apples h_bananas h_peaches
  rw [h_oranges, h_apples, h_bananas, h_peaches]
  sorry

end NUMINAMATH_GPT_fruit_basket_l1666_166697


namespace NUMINAMATH_GPT_smallest_base_for_80_l1666_166647

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end NUMINAMATH_GPT_smallest_base_for_80_l1666_166647


namespace NUMINAMATH_GPT_shelves_of_picture_books_l1666_166650

theorem shelves_of_picture_books
   (total_books : ℕ)
   (books_per_shelf : ℕ)
   (mystery_shelves : ℕ)
   (mystery_books : ℕ)
   (total_mystery_books : mystery_books = mystery_shelves * books_per_shelf)
   (total_books_condition : total_books = 32)
   (mystery_books_condition : mystery_books = 5 * books_per_shelf) :
   (total_books - mystery_books) / books_per_shelf = 3 :=
by
  sorry

end NUMINAMATH_GPT_shelves_of_picture_books_l1666_166650


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_problem_part_3_l1666_166639

open Set

-- Definitions for the given problem conditions
def U : Set ℕ := { x | x > 0 ∧ x < 10 }
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6}
def D : Set ℕ := B ∩ C

-- Prove each part of the problem
theorem problem_part_1 :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

theorem problem_part_2 :
  D = {3, 4} ∧
  (∀ (s : Set ℕ), s ⊆ D ↔ s = ∅ ∨ s = {3} ∨ s = {4} ∨ s = {3, 4}) := by
  sorry

theorem problem_part_3 :
  (U \ D) = {1, 2, 5, 6, 7, 8, 9} := by
  sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_problem_part_3_l1666_166639


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1666_166673

theorem maximum_value_of_expression (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) ≤ 3 := by
  sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1666_166673


namespace NUMINAMATH_GPT_find_k_l1666_166632

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (k : ℝ) : ℝ × ℝ := (2 * k, 3)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem find_k : ∃ k : ℝ, dot_product a (vector_add (scalar_mult 2 a) (b k)) = 0 ∧ k = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1666_166632


namespace NUMINAMATH_GPT_combined_weight_of_candles_l1666_166657

theorem combined_weight_of_candles (candles : ℕ) (weight_per_candle : ℕ) (total_weight : ℕ) :
  candles = 10 - 3 →
  weight_per_candle = 8 + 1 →
  total_weight = candles * weight_per_candle →
  total_weight = 63 :=
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_combined_weight_of_candles_l1666_166657


namespace NUMINAMATH_GPT_num_solutions_20_l1666_166622

def num_solutions (n : ℕ) : ℕ :=
  4 * n

theorem num_solutions_20 : num_solutions 20 = 80 := by
  sorry

end NUMINAMATH_GPT_num_solutions_20_l1666_166622


namespace NUMINAMATH_GPT_remainder_sum_is_74_l1666_166679

-- Defining the values from the given conditions
def num1 : ℕ := 1234567
def num2 : ℕ := 890123
def divisor : ℕ := 256

-- We state the theorem to capture the main problem
theorem remainder_sum_is_74 : (num1 + num2) % divisor = 74 := 
sorry

end NUMINAMATH_GPT_remainder_sum_is_74_l1666_166679


namespace NUMINAMATH_GPT_Maria_ate_2_cookies_l1666_166651

theorem Maria_ate_2_cookies : 
  ∀ (initial_cookies given_to_friend given_to_family remaining_after_eating : ℕ),
  initial_cookies = 19 →
  given_to_friend = 5 →
  given_to_family = (initial_cookies - given_to_friend) / 2 →
  remaining_after_eating = initial_cookies - given_to_friend - given_to_family - 2 →
  remaining_after_eating = 5 →
  2 = 2 := by
  intros
  sorry

end NUMINAMATH_GPT_Maria_ate_2_cookies_l1666_166651


namespace NUMINAMATH_GPT_sin_690_eq_neg_0_5_l1666_166676

theorem sin_690_eq_neg_0_5 : Real.sin (690 * Real.pi / 180) = -0.5 := by
  sorry

end NUMINAMATH_GPT_sin_690_eq_neg_0_5_l1666_166676


namespace NUMINAMATH_GPT_shorter_piece_length_correct_l1666_166619

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ := 
  total_length * ratio / (ratio + 1)

theorem shorter_piece_length_correct :
  shorter_piece_length 57.134 (3.25678 / 7.81945) = 16.790 :=
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_correct_l1666_166619


namespace NUMINAMATH_GPT_full_time_worked_year_l1666_166606

-- Define the conditions as constants
def total_employees : ℕ := 130
def full_time : ℕ := 80
def worked_year : ℕ := 100
def neither : ℕ := 20

-- Define the question as a theorem stating the correct answer
theorem full_time_worked_year : full_time + worked_year - total_employees + neither = 70 :=
by
  sorry

end NUMINAMATH_GPT_full_time_worked_year_l1666_166606


namespace NUMINAMATH_GPT_train_length_proof_l1666_166648

noncomputable def length_of_train : ℝ := 450.09

theorem train_length_proof
  (speed_kmh : ℝ := 60)
  (time_s : ℝ := 27) :
  (speed_kmh * (5 / 18) * time_s = length_of_train) :=
by
  sorry

end NUMINAMATH_GPT_train_length_proof_l1666_166648
