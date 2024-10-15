import Mathlib

namespace NUMINAMATH_GPT_hamburgers_left_over_l802_80205

theorem hamburgers_left_over (h_made : ℕ) (h_served : ℕ) (h_total : h_made = 9) (h_served_count : h_served = 3) : h_made - h_served = 6 :=
by
  sorry

end NUMINAMATH_GPT_hamburgers_left_over_l802_80205


namespace NUMINAMATH_GPT_largest_divisor_is_15_l802_80239

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def largest_divisor (n : ℕ) : ℕ :=
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)

theorem largest_divisor_is_15 : ∀ (n : ℕ), n > 0 → is_even n → 15 ∣ largest_divisor n ∧ (∀ m, m ∣ largest_divisor n → m ≤ 15) :=
by
  intros n pos even
  sorry

end NUMINAMATH_GPT_largest_divisor_is_15_l802_80239


namespace NUMINAMATH_GPT_add_fractions_l802_80288

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end NUMINAMATH_GPT_add_fractions_l802_80288


namespace NUMINAMATH_GPT_sequence_formula_l802_80210

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l802_80210


namespace NUMINAMATH_GPT_train_length_l802_80202

theorem train_length 
  (L : ℝ) -- Length of each train in meters.
  (speed_fast : ℝ := 56) -- Speed of the faster train in km/hr.
  (speed_slow : ℝ := 36) -- Speed of the slower train in km/hr.
  (time_pass : ℝ := 72) -- Time taken for the faster train to pass the slower train in seconds.
  (km_to_m_s : ℝ := 5 / 18) -- Conversion factor from km/hr to m/s.
  (relative_speed : ℝ := (speed_fast - speed_slow) * km_to_m_s) -- Relative speed in m/s.
  (distance_covered : ℝ := relative_speed * time_pass) -- Distance covered in meters.
  (equal_length : 2 * L = distance_covered) -- Condition of the problem: 2L = distance covered.
  : L = 200.16 :=
sorry

end NUMINAMATH_GPT_train_length_l802_80202


namespace NUMINAMATH_GPT_andy_late_minutes_l802_80262

theorem andy_late_minutes (school_starts_at : Nat) (normal_travel_time : Nat) 
  (stop_per_light : Nat) (red_lights : Nat) (construction_wait : Nat) 
  (left_house_at : Nat) : 
  let total_delay := (stop_per_light * red_lights) + construction_wait
  let total_travel_time := normal_travel_time + total_delay
  let arrive_time := left_house_at + total_travel_time
  let late_time := arrive_time - school_starts_at
  late_time = 7 :=
by
  sorry

end NUMINAMATH_GPT_andy_late_minutes_l802_80262


namespace NUMINAMATH_GPT_probability_10_coins_at_most_3_heads_l802_80273

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end NUMINAMATH_GPT_probability_10_coins_at_most_3_heads_l802_80273


namespace NUMINAMATH_GPT_Mary_ends_with_31_eggs_l802_80274

theorem Mary_ends_with_31_eggs (a b : ℕ) (h1 : a = 27) (h2 : b = 4) : a + b = 31 := by
  sorry

end NUMINAMATH_GPT_Mary_ends_with_31_eggs_l802_80274


namespace NUMINAMATH_GPT_ratio_of_a_over_3_to_b_over_2_l802_80240

theorem ratio_of_a_over_3_to_b_over_2 (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_over_3_to_b_over_2_l802_80240


namespace NUMINAMATH_GPT_tan_angle_PAB_correct_l802_80248

noncomputable def tan_angle_PAB (AB BC CA : ℝ) (P inside ABC : Prop) (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop) : ℝ :=
  180 / 329

theorem tan_angle_PAB_correct :
  ∀ (AB BC CA : ℝ)
    (P_inside_ABC : Prop)
    (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop),
    AB = 12 → BC = 15 → CA = 17 →
    (tan_angle_PAB AB BC CA P_inside_ABC PAB_angle_eq_PBC_angle_eq_PCA_angle) = 180 / 329 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tan_angle_PAB_correct_l802_80248


namespace NUMINAMATH_GPT_max_remainder_when_divided_by_7_l802_80255

theorem max_remainder_when_divided_by_7 (y : ℕ) (r : ℕ) (h : r = y % 7) : r ≤ 6 ∧ ∃ k, y = 7 * k + r :=
by
  sorry

end NUMINAMATH_GPT_max_remainder_when_divided_by_7_l802_80255


namespace NUMINAMATH_GPT_plates_difference_l802_80227

def num_plates_sunshine := 26^3 * 10^3
def num_plates_prairie := 26^2 * 10^4
def difference := num_plates_sunshine - num_plates_prairie

theorem plates_difference :
  difference = 10816000 := by sorry

end NUMINAMATH_GPT_plates_difference_l802_80227


namespace NUMINAMATH_GPT_f_f_2_eq_2_l802_80218

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 2 * Real.exp (x - 1)
else Real.log (x ^ 2 - 1) / Real.log 3

theorem f_f_2_eq_2 : f (f 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_f_f_2_eq_2_l802_80218


namespace NUMINAMATH_GPT_range_of_m_l802_80201

def prop_p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0
def prop_q (m : ℝ) : Prop := ∃ (x y : ℝ), (x^2) / (m-6) - (y^2) / (m+3) = 1

theorem range_of_m (m : ℝ) : ¬ (prop_p m ∧ prop_q m) → m ≥ -3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l802_80201


namespace NUMINAMATH_GPT_find_temperature_on_friday_l802_80247

variable (M T W Th F : ℕ)

def problem_conditions : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 46 ∧
  M = 44

theorem find_temperature_on_friday (h : problem_conditions M T W Th F) : F = 36 := by
  sorry

end NUMINAMATH_GPT_find_temperature_on_friday_l802_80247


namespace NUMINAMATH_GPT_find_speed_of_faster_train_l802_80204

noncomputable def speed_of_faster_train
  (length_each_train_m : ℝ)
  (speed_slower_kmph : ℝ)
  (time_pass_s : ℝ) : ℝ :=
  let distance_km := (2 * length_each_train_m / 1000)
  let time_pass_hr := (time_pass_s / 3600)
  let relative_speed_kmph := (distance_km / time_pass_hr)
  let speed_faster_kmph := (relative_speed_kmph - speed_slower_kmph)
  speed_faster_kmph

theorem find_speed_of_faster_train :
  speed_of_faster_train
    250   -- length_each_train_m
    30    -- speed_slower_kmph
    23.998080153587715 -- time_pass_s
  = 45 := sorry

end NUMINAMATH_GPT_find_speed_of_faster_train_l802_80204


namespace NUMINAMATH_GPT_simplify_fraction_l802_80259

theorem simplify_fraction (x : ℤ) : 
    (2 * x + 3) / 4 + (5 - 4 * x) / 3 = (-10 * x + 29) / 12 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l802_80259


namespace NUMINAMATH_GPT_Felipe_time_to_build_house_l802_80243

variables (E F : ℝ)
variables (Felipe_building_time_months : ℝ) (Combined_time : ℝ := 7.5) (Half_time_relation : F = 1 / 2 * E)

-- Felipe finished his house in 30 months
theorem Felipe_time_to_build_house :
  (F = 1 / 2 * E) →
  (F + E = Combined_time) →
  (Felipe_building_time_months = F * 12) →
  Felipe_building_time_months = 30 :=
by
  intros h1 h2 h3
  -- Combining the given conditions to prove the statement
  sorry

end NUMINAMATH_GPT_Felipe_time_to_build_house_l802_80243


namespace NUMINAMATH_GPT_tom_mileage_per_gallon_l802_80271

-- Definitions based on the given conditions
def daily_mileage : ℕ := 75
def cost_per_gallon : ℕ := 3
def amount_spent_in_10_days : ℕ := 45
def days : ℕ := 10

-- Main theorem to prove
theorem tom_mileage_per_gallon : 
  (amount_spent_in_10_days / cost_per_gallon) * 75 * days = 50 :=
by
  sorry

end NUMINAMATH_GPT_tom_mileage_per_gallon_l802_80271


namespace NUMINAMATH_GPT_incorrect_expression_l802_80284

variable (D : ℚ) (P Q : ℕ) (r s : ℕ)

-- D represents a repeating decimal.
-- P denotes the r figures of D which do not repeat themselves.
-- Q denotes the s figures of D which repeat themselves.

theorem incorrect_expression :
  10^r * (10^s - 1) * D ≠ Q * (P - 1) :=
sorry

end NUMINAMATH_GPT_incorrect_expression_l802_80284


namespace NUMINAMATH_GPT_sequence_is_geometric_and_general_formula_l802_80238

theorem sequence_is_geometric_and_general_formula (a : ℕ → ℝ) (h0 : a 1 = 2 / 3)
  (h1 : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (a (n + 1) + 1)) :
  ∃ r : ℝ, (0 < r ∧ r < 1 ∧ (∀ n : ℕ, a (n + 1) = (2:ℝ)^n / (1 + (2:ℝ)^n)) ∧
  ∀ n : ℕ, (1 / a (n + 1) - 1) = (1 / 2) * (1 / a n - 1)) := sorry

end NUMINAMATH_GPT_sequence_is_geometric_and_general_formula_l802_80238


namespace NUMINAMATH_GPT_min_value_of_a_l802_80256

noncomputable def x (t a : ℝ) : ℝ :=
  5 * (t + 1)^2 + a / (t + 1)^5

theorem min_value_of_a (a : ℝ) :
  (∀ t : ℝ, t ≥ 0 → x t a ≥ 24) ↔ a ≥ 2 * Real.sqrt ((24 / 7)^7) :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l802_80256


namespace NUMINAMATH_GPT_part_a_part_b_l802_80224

-- Define what it means for a coloring to be valid.
def valid_coloring (n : ℕ) (colors : Fin n → Fin 3) : Prop :=
  ∀ (i : Fin n),
  ∃ j k : Fin n, 
  ((i + 1) % n = j ∧ (i + 2) % n = k ∧ colors i ≠ colors j ∧ colors i ≠ colors k ∧ colors j ≠ colors k)

-- Part (a)
theorem part_a (n : ℕ) (hn : 3 ∣ n) : ∃ (colors : Fin n → Fin 3), valid_coloring n colors :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) : (∃ (colors : Fin n → Fin 3), valid_coloring n colors) → 3 ∣ n :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l802_80224


namespace NUMINAMATH_GPT_candy_bar_cost_l802_80290

def num_quarters := 4
def num_dimes := 3
def num_nickel := 1
def change_received := 4

def value_quarter := 25
def value_dime := 10
def value_nickel := 5

def total_paid := (num_quarters * value_quarter) + (num_dimes * value_dime) + (num_nickel * value_nickel)
def cost_candy_bar := total_paid - change_received

theorem candy_bar_cost : cost_candy_bar = 131 := by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l802_80290


namespace NUMINAMATH_GPT_saturday_earnings_l802_80293

variable (S : ℝ)
variable (totalEarnings : ℝ := 5182.50)
variable (difference : ℝ := 142.50)

theorem saturday_earnings : 
  S + (S - difference) = totalEarnings → S = 2662.50 := 
by 
  intro h 
  sorry

end NUMINAMATH_GPT_saturday_earnings_l802_80293


namespace NUMINAMATH_GPT_maximize_profit_at_200_l802_80260

noncomputable def cost (q : ℝ) : ℝ := 50000 + 200 * q
noncomputable def price (q : ℝ) : ℝ := 24200 - (1/5) * q^2
noncomputable def profit (q : ℝ) : ℝ := (price q) * q - (cost q)

theorem maximize_profit_at_200 : ∃ (q : ℝ), q = 200 ∧ ∀ (x : ℝ), x ≥ 0 → profit q ≥ profit x :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_at_200_l802_80260


namespace NUMINAMATH_GPT_marbles_per_friend_l802_80261

variable (initial_marbles remaining_marbles given_marbles_per_friend : ℕ)

-- conditions in a)
def condition_initial_marbles := initial_marbles = 500
def condition_remaining_marbles := 4 * remaining_marbles = 720
def condition_total_given_marbles := initial_marbles - remaining_marbles = 320
def condition_given_marbles_per_friend := given_marbles_per_friend * 4 = 320

-- question proof goal
theorem marbles_per_friend (initial_marbles: ℕ) (remaining_marbles: ℕ) (given_marbles_per_friend: ℕ) :
  (condition_initial_marbles initial_marbles) →
  (condition_remaining_marbles remaining_marbles) →
  (condition_total_given_marbles initial_marbles remaining_marbles) →
  (condition_given_marbles_per_friend given_marbles_per_friend) →
  given_marbles_per_friend = 80 :=
by
  intros hinitial hremaining htotal_given hgiven_per_friend
  sorry

end NUMINAMATH_GPT_marbles_per_friend_l802_80261


namespace NUMINAMATH_GPT_no_valid_triples_l802_80214

theorem no_valid_triples (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : 6 * (a * b + b * c + c * a) = a * b * c) : false :=
by
  sorry

end NUMINAMATH_GPT_no_valid_triples_l802_80214


namespace NUMINAMATH_GPT_find_digits_l802_80225

-- Define the digits range
def is_digit (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 9

-- Define the five-digit numbers
def num_abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 111 * c
def num_abbbb (a b : ℕ) : ℕ := 10000 * a + 1111 * b

-- Problem statement
theorem find_digits (a b c : ℕ) (h_da : is_digit a) (h_db : is_digit b) (h_dc : is_digit c) :
  (num_abccc a b c) + 1 = (num_abbbb a b) ↔
  (a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0) :=
sorry

end NUMINAMATH_GPT_find_digits_l802_80225


namespace NUMINAMATH_GPT_parking_garage_floors_l802_80285

theorem parking_garage_floors 
  (total_time : ℕ)
  (time_per_floor : ℕ)
  (gate_time : ℕ)
  (every_n_floors : ℕ) 
  (F : ℕ) 
  (h1 : total_time = 1440)
  (h2 : time_per_floor = 80)
  (h3 : gate_time = 120)
  (h4 : every_n_floors = 3)
  :
  F = 13 :=
by
  have total_id_time : ℕ := gate_time * ((F - 1) / every_n_floors)
  have total_drive_time : ℕ := time_per_floor * (F - 1)
  have total_time_calc : ℕ := total_drive_time + total_id_time
  have h5 := total_time_calc = total_time
  -- Now we simplify the algebraic equation given the problem conditions
  sorry

end NUMINAMATH_GPT_parking_garage_floors_l802_80285


namespace NUMINAMATH_GPT_sum_consecutive_integers_l802_80206

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_consecutive_integers_l802_80206


namespace NUMINAMATH_GPT_lisa_and_robert_total_photos_l802_80236

def claire_photos : Nat := 10
def lisa_photos (c : Nat) : Nat := 3 * c
def robert_photos (c : Nat) : Nat := c + 20

theorem lisa_and_robert_total_photos :
  let c := claire_photos
  let l := lisa_photos c
  let r := robert_photos c
  l + r = 60 :=
by
  sorry

end NUMINAMATH_GPT_lisa_and_robert_total_photos_l802_80236


namespace NUMINAMATH_GPT_socks_combinations_correct_l802_80263

noncomputable def num_socks_combinations (colors patterns pairs : ℕ) : ℕ :=
  colors * (colors - 1) * patterns * (patterns - 1)

theorem socks_combinations_correct :
  num_socks_combinations 5 4 20 = 240 :=
by
  sorry

end NUMINAMATH_GPT_socks_combinations_correct_l802_80263


namespace NUMINAMATH_GPT_g_at_3_l802_80257

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) - x * g (3 ^ (-x)) = x) : g 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_g_at_3_l802_80257


namespace NUMINAMATH_GPT_value_of_x_and_z_l802_80281

theorem value_of_x_and_z (x y z : ℤ) (h1 : x / y = 7 / 3) (h2 : y = 21) (h3 : z = 3 * y) : x = 49 ∧ z = 63 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_and_z_l802_80281


namespace NUMINAMATH_GPT_part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l802_80249

theorem part_a_max_cells_crossed (m n : ℕ) : 
  ∃ max_cells : ℕ, max_cells = m + n - 1 := sorry

theorem part_b_max_cells_crossed_by_needle : 
  ∃ max_cells : ℕ, max_cells = 285 := sorry

end NUMINAMATH_GPT_part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l802_80249


namespace NUMINAMATH_GPT_cone_altitude_to_radius_ratio_l802_80298

theorem cone_altitude_to_radius_ratio (r h : ℝ) (V_cone V_sphere : ℝ)
  (h1 : V_sphere = (4 / 3) * Real.pi * r^3)
  (h2 : V_cone = (1 / 3) * Real.pi * r^2 * h)
  (h3 : V_cone = (1 / 3) * V_sphere) :
  h / r = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_altitude_to_radius_ratio_l802_80298


namespace NUMINAMATH_GPT_simplify_expression_l802_80203

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l802_80203


namespace NUMINAMATH_GPT_cost_of_apple_l802_80234

variable (A O : ℝ)

theorem cost_of_apple :
  (6 * A + 3 * O = 1.77) ∧ (2 * A + 5 * O = 1.27) → A = 0.21 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_of_apple_l802_80234


namespace NUMINAMATH_GPT_annual_interest_rate_is_10_percent_l802_80244

noncomputable def principal (P : ℝ) := P = 1500
noncomputable def total_amount (A : ℝ) := A = 1815
noncomputable def time_period (t : ℝ) := t = 2
noncomputable def compounding_frequency (n : ℝ) := n = 1
noncomputable def interest_rate_compound_interest_formula (P A t n : ℝ) (r : ℝ) := 
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate_is_10_percent : 
  ∀ (P A t n : ℝ) (r : ℝ), principal P → total_amount A → time_period t → compounding_frequency n → 
  interest_rate_compound_interest_formula P A t n r → r = 0.1 :=
by
  intros P A t n r hP hA ht hn h_formula
  sorry

end NUMINAMATH_GPT_annual_interest_rate_is_10_percent_l802_80244


namespace NUMINAMATH_GPT_min_sticks_to_avoid_rectangles_l802_80265

noncomputable def min_stick_deletions (n : ℕ) : ℕ :=
  if n = 8 then 43 else 0 -- we define 43 as the minimum for an 8x8 chessboard

theorem min_sticks_to_avoid_rectangles : min_stick_deletions 8 = 43 :=
  by
    sorry

end NUMINAMATH_GPT_min_sticks_to_avoid_rectangles_l802_80265


namespace NUMINAMATH_GPT_find_a_if_parallel_l802_80226

-- Define the parallel condition for the given lines
def is_parallel (a : ℝ) : Prop :=
  let slope1 := -a / 2
  let slope2 := 3
  slope1 = slope2

-- Prove that a = -6 under the parallel condition
theorem find_a_if_parallel (a : ℝ) (h : is_parallel a) : a = -6 := by
  sorry

end NUMINAMATH_GPT_find_a_if_parallel_l802_80226


namespace NUMINAMATH_GPT_one_inch_cubes_with_red_paint_at_least_two_faces_l802_80209

theorem one_inch_cubes_with_red_paint_at_least_two_faces
  (number_of_one_inch_cubes : ℕ)
  (cubes_with_three_faces : ℕ)
  (cubes_with_two_faces : ℕ)
  (total_cubes_with_at_least_two_faces : ℕ) :
  number_of_one_inch_cubes = 64 →
  cubes_with_three_faces = 8 →
  cubes_with_two_faces = 24 →
  total_cubes_with_at_least_two_faces = cubes_with_three_faces + cubes_with_two_faces →
  total_cubes_with_at_least_two_faces = 32 :=
by
  sorry

end NUMINAMATH_GPT_one_inch_cubes_with_red_paint_at_least_two_faces_l802_80209


namespace NUMINAMATH_GPT_min_value_of_D_l802_80277

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a) ^ 2 + (Real.exp x - 2 * Real.sqrt a) ^ 2) + a + 2

theorem min_value_of_D (e : ℝ) (h_e : e = 2.71828) :
  ∀ a : ℝ, ∃ x : ℝ, D x a = Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_D_l802_80277


namespace NUMINAMATH_GPT_fill_pipe_fraction_l802_80200

theorem fill_pipe_fraction (t : ℕ) (f : ℝ) (h : t = 30) (h' : f = 1) : f = 1 :=
by
  sorry

end NUMINAMATH_GPT_fill_pipe_fraction_l802_80200


namespace NUMINAMATH_GPT_part_a_part_b_l802_80237

def balanced (V : Finset (ℝ × ℝ)) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ V → B ∈ V → A ≠ B → ∃ C : ℝ × ℝ, C ∈ V ∧ (dist C A = dist C B)

def center_free (V : Finset (ℝ × ℝ)) : Prop :=
  ¬ ∃ (A B C P : ℝ × ℝ), A ∈ V → B ∈ V → C ∈ V → P ∈ V →
                         A ≠ B ∧ B ≠ C ∧ A ≠ C →
                         (dist P A = dist P B ∧ dist P B = dist P C)

theorem part_a (n : ℕ) (hn : 3 ≤ n) :
  ∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V :=
by sorry

theorem part_b : ∀ n : ℕ, 3 ≤ n →
  (∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V ∧ center_free V ↔ n % 2 = 1) :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l802_80237


namespace NUMINAMATH_GPT_find_x_l802_80289

theorem find_x (x : ℝ) (h : 5.76 = 0.12 * 0.40 * x) : x = 120 := 
sorry

end NUMINAMATH_GPT_find_x_l802_80289


namespace NUMINAMATH_GPT_five_card_draw_probability_l802_80291

noncomputable def probability_at_least_one_card_from_each_suit : ℚ := 3 / 32

theorem five_card_draw_probability :
  let deck_size := 52
  let suits := 4
  let cards_drawn := 5
  (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4) = probability_at_least_one_card_from_each_suit := by
  sorry

end NUMINAMATH_GPT_five_card_draw_probability_l802_80291


namespace NUMINAMATH_GPT_problem_statement_l802_80264

noncomputable def expr (x y z : ℝ) : ℝ :=
  (x^2 * y^2) / ((x^2 - y*z) * (y^2 - x*z)) +
  (x^2 * z^2) / ((x^2 - y*z) * (z^2 - x*y)) +
  (y^2 * z^2) / ((y^2 - x*z) * (z^2 - x*y))

theorem problem_statement (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) (h₄ : x + y + z = -1) :
  expr x y z = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l802_80264


namespace NUMINAMATH_GPT_exists_unique_n_digit_number_with_one_l802_80228

def n_digit_number (n : ℕ) : Type := {l : List ℕ // l.length = n ∧ ∀ x ∈ l, x = 1 ∨ x = 2 ∨ x = 3}

theorem exists_unique_n_digit_number_with_one (n : ℕ) (hn : n > 0) :
  ∃ x : n_digit_number n, x.val.count 1 = 1 ∧ ∀ y : n_digit_number n, y ≠ x → x.val.append [1] ≠ y.val.append [1] :=
sorry

end NUMINAMATH_GPT_exists_unique_n_digit_number_with_one_l802_80228


namespace NUMINAMATH_GPT_min_value_of_f_l802_80280

noncomputable def f (x : ℝ) : ℝ := x^2 + 8 * x + 3

theorem min_value_of_f : ∃ x₀ : ℝ, (∀ x : ℝ, f x ≥ f x₀) ∧ f x₀ = -13 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l802_80280


namespace NUMINAMATH_GPT_digit_in_452nd_place_l802_80216

def repeating_sequence : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]
def repeat_length : Nat := 18

theorem digit_in_452nd_place :
  (repeating_sequence.get ⟨(452 % repeat_length) - 1, sorry⟩ = 6) :=
sorry

end NUMINAMATH_GPT_digit_in_452nd_place_l802_80216


namespace NUMINAMATH_GPT_can_form_triangle_l802_80215

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle :
  (is_triangle 3 5 7) ∧ ¬(is_triangle 3 3 7) ∧ ¬(is_triangle 4 4 8) ∧ ¬(is_triangle 4 5 9) :=
by
  -- Proof steps will be added here
  sorry

end NUMINAMATH_GPT_can_form_triangle_l802_80215


namespace NUMINAMATH_GPT_quadratic_sum_solutions_l802_80258

theorem quadratic_sum_solutions {a b : ℝ} (h : a ≥ b) (h1: a = 1 + Real.sqrt 17) (h2: b = 1 - Real.sqrt 17) :
  3 * a + 2 * b = 5 + Real.sqrt 17 := by
  sorry

end NUMINAMATH_GPT_quadratic_sum_solutions_l802_80258


namespace NUMINAMATH_GPT_problem_solution_l802_80283

theorem problem_solution :
  (-2: ℤ)^2004 + 3 * (-2: ℤ)^2003 = -2^2003 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l802_80283


namespace NUMINAMATH_GPT_log_base_equal_l802_80253

noncomputable def logx (b x : ℝ) := Real.log x / Real.log b

theorem log_base_equal {x : ℝ} (h : 0 < x ∧ x ≠ 1) :
  logx 81 x = logx 16 2 → x = 3 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_log_base_equal_l802_80253


namespace NUMINAMATH_GPT_factorization_correct_l802_80245

theorem factorization_correct (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end NUMINAMATH_GPT_factorization_correct_l802_80245


namespace NUMINAMATH_GPT_mrs_hilt_baked_pecan_pies_l802_80254

def total_pies (rows : ℕ) (pies_per_row : ℕ) : ℕ :=
  rows * pies_per_row

def pecan_pies (total_pies : ℕ) (apple_pies : ℕ) : ℕ :=
  total_pies - apple_pies

theorem mrs_hilt_baked_pecan_pies :
  let apple_pies := 14
  let rows := 6
  let pies_per_row := 5
  let total := total_pies rows pies_per_row
  pecan_pies total apple_pies = 16 :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_baked_pecan_pies_l802_80254


namespace NUMINAMATH_GPT_bottles_more_than_apples_l802_80211

def regular_soda : ℕ := 72
def diet_soda : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := regular_soda + diet_soda

theorem bottles_more_than_apples : total_bottles - apples = 26 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_bottles_more_than_apples_l802_80211


namespace NUMINAMATH_GPT_mask_price_reduction_l802_80275

theorem mask_price_reduction 
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (additional_sales_factor : ℝ)
  (desired_profit : ℝ)
  (x : ℝ)
  (h_initial_sales : initial_sales = 500)
  (h_initial_profit : initial_profit = 0.6)
  (h_additional_sales_factor : additional_sales_factor = 100 / 0.1)
  (h_desired_profit : desired_profit = 240) :
  (initial_profit - x) * (initial_sales + additional_sales_factor * x) = desired_profit → x = 0.3 :=
sorry

end NUMINAMATH_GPT_mask_price_reduction_l802_80275


namespace NUMINAMATH_GPT_probability_two_red_two_blue_l802_80294

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let total_ways_to_choose_4 := Nat.choose total_marbles 4
  let ways_to_choose_2_red := Nat.choose red_marbles 2
  let ways_to_choose_2_blue := Nat.choose blue_marbles 2
  (ways_to_choose_2_red * ways_to_choose_2_blue : ℚ) / total_ways_to_choose_4 = 56 / 147 := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_two_red_two_blue_l802_80294


namespace NUMINAMATH_GPT_max_playground_area_l802_80296

theorem max_playground_area
  (l w : ℝ)
  (h_fence : 2 * l + 2 * w = 400)
  (h_l_min : l ≥ 100)
  (h_w_min : w ≥ 50) :
  l * w ≤ 10000 :=
by
  sorry

end NUMINAMATH_GPT_max_playground_area_l802_80296


namespace NUMINAMATH_GPT_find_EF_squared_l802_80219

noncomputable def square_side := 15
noncomputable def BE := 6
noncomputable def DF := 6
noncomputable def AE := 14
noncomputable def CF := 14

theorem find_EF_squared (A B C D E F : ℝ) (AB BC CD DA : ℝ := square_side) :
  (BE = 6) → (DF = 6) → (AE = 14) → (CF = 14) → EF^2 = 72 :=
by
  -- Definitions and conditions usage according to (a)
  sorry

end NUMINAMATH_GPT_find_EF_squared_l802_80219


namespace NUMINAMATH_GPT_carols_rectangle_length_l802_80217

theorem carols_rectangle_length :
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  carol_length = 5 :=
by
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  show carol_length = 5
  sorry

end NUMINAMATH_GPT_carols_rectangle_length_l802_80217


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l802_80212

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 3) 
  (h2 : a + 11 * d = 9) : 
  a + 4 * d = -12 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l802_80212


namespace NUMINAMATH_GPT_b_squared_gt_4ac_l802_80241

theorem b_squared_gt_4ac (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
by
  sorry

end NUMINAMATH_GPT_b_squared_gt_4ac_l802_80241


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l802_80276

def P (x : ℝ) : Prop := 2 ≤ x ∧ x < 4
def Q (x : ℝ) : Prop := 3 * x - 7 ≥ 8 - 2 * x

theorem intersection_of_P_and_Q :
  ∀ x, P x ∧ Q x ↔ 3 ≤ x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l802_80276


namespace NUMINAMATH_GPT_diamond_2_3_l802_80233

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_2_3 : diamond 2 3 = 16 :=
by
  -- Imported definition and theorem structure.
  sorry

end NUMINAMATH_GPT_diamond_2_3_l802_80233


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l802_80252

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m < 1) → (∀ x y : ℝ, (x - m) ^ 2 + y ^ 2 = m ^ 2 → (x, y) ≠ (1, 1)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l802_80252


namespace NUMINAMATH_GPT_least_subtract_to_divisible_by_14_l802_80232

theorem least_subtract_to_divisible_by_14 (n : ℕ) (h : n = 7538): 
  (n % 14 = 6) -> ∃ m, (m = 6) ∧ ((n - m) % 14 = 0) :=
by
  sorry

end NUMINAMATH_GPT_least_subtract_to_divisible_by_14_l802_80232


namespace NUMINAMATH_GPT_tickets_used_to_buy_toys_l802_80297

-- Definitions for the conditions
def initial_tickets : ℕ := 13
def leftover_tickets : ℕ := 7

-- The theorem we want to prove
theorem tickets_used_to_buy_toys : initial_tickets - leftover_tickets = 6 :=
by
  sorry

end NUMINAMATH_GPT_tickets_used_to_buy_toys_l802_80297


namespace NUMINAMATH_GPT_solve_for_x_l802_80268

theorem solve_for_x (x : ℤ) (h : x + 1 = 10) : x = 9 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l802_80268


namespace NUMINAMATH_GPT_volume_of_rotated_solid_l802_80299

theorem volume_of_rotated_solid (unit_cylinder_r1 h1 r2 h2 : ℝ) :
  unit_cylinder_r1 = 6 → h1 = 1 → r2 = 3 → h2 = 4 → 
  (π * unit_cylinder_r1^2 * h1 + π * r2^2 * h2) = 72 * π :=
by 
-- We place the arguments and sorry for skipping the proof
  sorry

end NUMINAMATH_GPT_volume_of_rotated_solid_l802_80299


namespace NUMINAMATH_GPT_expression_for_f_pos_f_monotone_on_pos_l802_80269

section

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_neg : ∀ x, -1 ≤ x ∧ x < 0 → f x = 2 * x + 1 / x^2)

-- Part 1: Prove the expression for f(x) when x ∈ (0,1]
theorem expression_for_f_pos (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  f x = 2 * x - 1 / x^2 :=
sorry

-- Part 2: Prove the monotonicity of f(x) on (0,1]
theorem f_monotone_on_pos : 
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y ≤ 1 → f x < f y :=
sorry

end

end NUMINAMATH_GPT_expression_for_f_pos_f_monotone_on_pos_l802_80269


namespace NUMINAMATH_GPT_highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l802_80231

theorem highest_power_of_two_factor_13_pow_4_minus_11_pow_4 :
  ∃ n : ℕ, n = 5 ∧ (2 ^ n ∣ (13 ^ 4 - 11 ^ 4)) ∧ ¬ (2 ^ (n + 1) ∣ (13 ^ 4 - 11 ^ 4)) :=
sorry

end NUMINAMATH_GPT_highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l802_80231


namespace NUMINAMATH_GPT_calc_log_expression_l802_80266

theorem calc_log_expression : 2 * Real.log 5 + Real.log 4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_calc_log_expression_l802_80266


namespace NUMINAMATH_GPT_second_field_area_percent_greater_l802_80270

theorem second_field_area_percent_greater (r1 r2 : ℝ) (h : r1 / r2 = 2 / 5) : 
  (π * (r2^2) - π * (r1^2)) / (π * (r1^2)) * 100 = 525 := 
by
  sorry

end NUMINAMATH_GPT_second_field_area_percent_greater_l802_80270


namespace NUMINAMATH_GPT_zach_needs_more_money_l802_80230

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end NUMINAMATH_GPT_zach_needs_more_money_l802_80230


namespace NUMINAMATH_GPT_range_of_a_l802_80221

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 1 > 0) ↔ (-2 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l802_80221


namespace NUMINAMATH_GPT_distance_light_travels_100_years_l802_80242

def distance_light_travels_one_year : ℝ := 5870e9 * 10^3

theorem distance_light_travels_100_years : distance_light_travels_one_year * 100 = 587 * 10^12 :=
by
  rw [distance_light_travels_one_year]
  sorry

end NUMINAMATH_GPT_distance_light_travels_100_years_l802_80242


namespace NUMINAMATH_GPT_eight_letter_good_words_l802_80207

-- Definition of a good word sequence (only using A, B, and C)
inductive Letter
| A | B | C

-- Define the restriction condition for a good word
def is_valid_transition (a b : Letter) : Prop :=
  match a, b with
  | Letter.A, Letter.B => False
  | Letter.B, Letter.C => False
  | Letter.C, Letter.A => False
  | _, _ => True

-- Count the number of 8-letter good words
def count_good_words : ℕ :=
  let letters := [Letter.A, Letter.B, Letter.C]
  -- Initial 3 choices for the first letter
  let first_choices := letters.length
  -- Subsequent 7 letters each have 2 valid previous choices
  let subsequent_choices := 2 ^ 7
  first_choices * subsequent_choices

theorem eight_letter_good_words : count_good_words = 384 :=
by
  sorry

end NUMINAMATH_GPT_eight_letter_good_words_l802_80207


namespace NUMINAMATH_GPT_find_natural_numbers_l802_80222

theorem find_natural_numbers (n : ℕ) (h : n > 1) : 
  ((n - 1) ∣ (n^3 - 3)) ↔ (n = 2 ∨ n = 3) := 
by 
  sorry

end NUMINAMATH_GPT_find_natural_numbers_l802_80222


namespace NUMINAMATH_GPT_second_root_of_quadratic_l802_80272

theorem second_root_of_quadratic (p q r : ℝ) (quad_eqn : ∀ x, 2 * p * (q - r) * x^2 + 3 * q * (r - p) * x + 4 * r * (p - q) = 0) (root : 2 * p * (q - r) * 2^2 + 3 * q * (r - p) * 2 + 4 * r * (p - q) = 0) :
    ∃ r₂ : ℝ, r₂ = (r * (p - q)) / (p * (q - r)) :=
sorry

end NUMINAMATH_GPT_second_root_of_quadratic_l802_80272


namespace NUMINAMATH_GPT_fish_initial_numbers_l802_80295

theorem fish_initial_numbers (x y : ℕ) (h1 : x + y = 100) (h2 : x - 30 = y - 40) : x = 45 ∧ y = 55 :=
by
  sorry

end NUMINAMATH_GPT_fish_initial_numbers_l802_80295


namespace NUMINAMATH_GPT_pasta_ratio_l802_80278

theorem pasta_ratio (total_students : ℕ) (spaghetti : ℕ) (manicotti : ℕ) 
  (h1 : total_students = 650) 
  (h2 : spaghetti = 250) 
  (h3 : manicotti = 100) : 
  (spaghetti : ℤ) / (manicotti : ℤ) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_pasta_ratio_l802_80278


namespace NUMINAMATH_GPT_ram_task_completion_days_l802_80287

theorem ram_task_completion_days (R : ℕ) (h1 : ∀ k : ℕ, k = R / 2) (h2 : 1 / R + 2 / R = 1 / 12) : R = 36 :=
sorry

end NUMINAMATH_GPT_ram_task_completion_days_l802_80287


namespace NUMINAMATH_GPT_interchanged_digit_multiple_of_sum_l802_80286

theorem interchanged_digit_multiple_of_sum (n a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : n = 3 * (a + b)) 
  (h3 : 1 ≤ a) (h4 : a ≤ 9) 
  (h5 : 0 ≤ b) (h6 : b ≤ 9) : 
  10 * b + a = 8 * (a + b) := 
by 
  sorry

end NUMINAMATH_GPT_interchanged_digit_multiple_of_sum_l802_80286


namespace NUMINAMATH_GPT_abs_ineq_solution_l802_80229

theorem abs_ineq_solution (x : ℝ) : abs (x - 2) + abs (x - 3) < 9 ↔ -2 < x ∧ x < 7 :=
sorry

end NUMINAMATH_GPT_abs_ineq_solution_l802_80229


namespace NUMINAMATH_GPT_remaining_grandchild_share_l802_80208

theorem remaining_grandchild_share 
  (total : ℕ) 
  (half_share : ℕ) 
  (remaining : ℕ) 
  (n : ℕ) 
  (total_eq : total = 124600)
  (half_share_eq : half_share = total / 2)
  (remaining_eq : remaining = total - half_share)
  (n_eq : n = 10) 
  : remaining / n = 6230 := 
by sorry

end NUMINAMATH_GPT_remaining_grandchild_share_l802_80208


namespace NUMINAMATH_GPT_total_coins_l802_80246
-- Import the necessary library

-- Defining the conditions
def quarters := 22
def dimes := quarters + 3
def nickels := quarters - 6

-- Main theorem statement
theorem total_coins : (quarters + dimes + nickels) = 63 := by
  sorry

end NUMINAMATH_GPT_total_coins_l802_80246


namespace NUMINAMATH_GPT_train_length_is_correct_l802_80250

noncomputable def speed_kmhr : ℝ := 45
noncomputable def time_sec : ℝ := 30
noncomputable def bridge_length_m : ℝ := 235

noncomputable def speed_ms : ℝ := (speed_kmhr * 1000) / 3600
noncomputable def total_distance_m : ℝ := speed_ms * time_sec
noncomputable def train_length_m : ℝ := total_distance_m - bridge_length_m

theorem train_length_is_correct : train_length_m = 140 :=
by
  -- Placeholder to indicate that a proof should go here
  -- Proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l802_80250


namespace NUMINAMATH_GPT_boys_to_girls_ratio_l802_80213

theorem boys_to_girls_ratio (x y : ℕ) 
  (h1 : 149 * x + 144 * y = 147 * (x + y)) : 
  x = (3 / 2 : ℚ) * y :=
by
  sorry

end NUMINAMATH_GPT_boys_to_girls_ratio_l802_80213


namespace NUMINAMATH_GPT_determine_N_l802_80223

theorem determine_N (N : ℕ) : (Nat.choose N 5 = 3003) ↔ (N = 15) :=
by
  sorry

end NUMINAMATH_GPT_determine_N_l802_80223


namespace NUMINAMATH_GPT_exists_nat_with_digit_sum_1000_and_square_sum_1000000_l802_80292

-- Define a function to calculate the sum of digits in base-10
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem exists_nat_with_digit_sum_1000_and_square_sum_1000000 :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_exists_nat_with_digit_sum_1000_and_square_sum_1000000_l802_80292


namespace NUMINAMATH_GPT_study_tour_arrangement_l802_80267

def number_of_arrangements (classes routes : ℕ) (max_selected_route : ℕ) : ℕ :=
  if classes = 4 ∧ routes = 4 ∧ max_selected_route = 2 then 240 else 0

theorem study_tour_arrangement :
  number_of_arrangements 4 4 2 = 240 :=
by sorry

end NUMINAMATH_GPT_study_tour_arrangement_l802_80267


namespace NUMINAMATH_GPT_number_of_real_roots_l802_80279

theorem number_of_real_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b^2 + 1 = 0) :
  (c > 0 → ∃ x1 x2 x3 : ℝ, 
    (x1 = b * Real.sqrt c ∨ x1 = -b * Real.sqrt c ∨ x1 = -c / b) ∧
    (x2 = b * Real.sqrt c ∨ x2 = -b * Real.sqrt c ∨ x2 = -c / b) ∧
    (x3 = b * Real.sqrt c ∨ x3 = -b * Real.sqrt c ∨ x3 = -c / b)) ∧
  (c < 0 → ∃ x1 : ℝ, x1 = -c / b) :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_roots_l802_80279


namespace NUMINAMATH_GPT_linda_loan_interest_difference_l802_80220

theorem linda_loan_interest_difference :
  let P : ℝ := 8000
  let r : ℝ := 0.10
  let t : ℕ := 3
  let n_monthly : ℕ := 12
  let n_annual : ℕ := 1
  let A_monthly : ℝ := P * (1 + r / (n_monthly : ℝ))^(n_monthly * t)
  let A_annual : ℝ := P * (1 + r)^t
  A_monthly - A_annual = 151.07 :=
by
  sorry

end NUMINAMATH_GPT_linda_loan_interest_difference_l802_80220


namespace NUMINAMATH_GPT_cars_in_fourth_store_l802_80235

theorem cars_in_fourth_store
  (mean : ℝ) 
  (a1 a2 a3 a5 : ℝ) 
  (num_stores : ℝ) 
  (mean_value : mean = 20.8) 
  (a1_value : a1 = 30) 
  (a2_value : a2 = 14) 
  (a3_value : a3 = 14) 
  (a5_value : a5 = 25) 
  (num_stores_value : num_stores = 5) :
  ∃ x : ℝ, (a1 + a2 + a3 + x + a5) / num_stores = mean ∧ x = 21 :=
by
  sorry

end NUMINAMATH_GPT_cars_in_fourth_store_l802_80235


namespace NUMINAMATH_GPT_number_of_girls_l802_80251

-- Given conditions
def ratio_girls_boys_teachers (girls boys teachers : ℕ) : Prop :=
  3 * (girls + boys + teachers) = 3 * girls + 2 * boys + 1 * teachers

def total_people (total girls boys teachers : ℕ) : Prop :=
  total = girls + boys + teachers

-- Define the main theorem
theorem number_of_girls 
  (k total : ℕ)
  (h1 : ratio_girls_boys_teachers (3 * k) (2 * k) k)
  (h2 : total_people total (3 * k) (2 * k) k)
  (h_total : total = 60) : 
  3 * k = 30 :=
  sorry

end NUMINAMATH_GPT_number_of_girls_l802_80251


namespace NUMINAMATH_GPT_range_of_a_l802_80282

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

def has_pos_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, (3 + a * Real.exp (a * x) = 0) ∧ (x > 0)

theorem range_of_a (a : ℝ) : has_pos_extremum a → a < -3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l802_80282
