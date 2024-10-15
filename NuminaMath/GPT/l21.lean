import Mathlib

namespace NUMINAMATH_GPT_trajectory_of_M_l21_2142

theorem trajectory_of_M
  (A : ℝ × ℝ := (3, 0))
  (P_circle : ∀ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1)
  (M_midpoint : ∀ (P M : ℝ × ℝ), M = ((P.1 + 3) / 2, P.2 / 2) → M.1 = x ∧ M.2 = y) :
  (∀ (x y : ℝ), (x - 3/2)^2 + y^2 = 1/4) := 
sorry

end NUMINAMATH_GPT_trajectory_of_M_l21_2142


namespace NUMINAMATH_GPT_parker_total_weight_l21_2168

-- Define the number of initial dumbbells and their weight
def initial_dumbbells := 4
def weight_per_dumbbell := 20

-- Define the number of additional dumbbells
def additional_dumbbells := 2

-- Define the total weight calculation
def total_weight := initial_dumbbells * weight_per_dumbbell + additional_dumbbells * weight_per_dumbbell

-- Prove that the total weight is 120 pounds
theorem parker_total_weight : total_weight = 120 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_parker_total_weight_l21_2168


namespace NUMINAMATH_GPT_fill_digits_subtraction_correct_l21_2197

theorem fill_digits_subtraction_correct :
  ∀ (A B : ℕ), A236 - (B*100 + 97) = 5439 → A = 6 ∧ B = 7 :=
by
  sorry

end NUMINAMATH_GPT_fill_digits_subtraction_correct_l21_2197


namespace NUMINAMATH_GPT_isabel_total_problems_l21_2166

theorem isabel_total_problems
  (math_pages : ℕ)
  (reading_pages : ℕ)
  (problems_per_page : ℕ)
  (h1 : math_pages = 2)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 5) :
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end NUMINAMATH_GPT_isabel_total_problems_l21_2166


namespace NUMINAMATH_GPT_javier_total_time_spent_l21_2140

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end NUMINAMATH_GPT_javier_total_time_spent_l21_2140


namespace NUMINAMATH_GPT_geometric_inequality_l21_2130

variable {q : ℝ} {b : ℕ → ℝ}

def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_inequality
  (h_geometric : geometric_sequence b q)
  (h_q_gt_one : q > 1)
  (h_pos : ∀ n : ℕ, b n > 0) :
  b 4 + b 8 > b 5 + b 7 :=
by
  sorry

end NUMINAMATH_GPT_geometric_inequality_l21_2130


namespace NUMINAMATH_GPT_perpendicular_vectors_l21_2152

theorem perpendicular_vectors (a : ℝ) 
  (v1 : ℝ × ℝ := (4, -5))
  (v2 : ℝ × ℝ := (a, 2))
  (perpendicular : v1.fst * v2.fst + v1.snd * v2.snd = 0) :
  a = 5 / 2 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l21_2152


namespace NUMINAMATH_GPT_current_length_of_highway_l21_2101

def total_length : ℕ := 650
def miles_first_day : ℕ := 50
def miles_second_day : ℕ := 3 * miles_first_day
def miles_still_needed : ℕ := 250
def miles_built : ℕ := miles_first_day + miles_second_day

theorem current_length_of_highway :
  total_length - miles_still_needed = 400 :=
by
  sorry

end NUMINAMATH_GPT_current_length_of_highway_l21_2101


namespace NUMINAMATH_GPT_game_prob_comparison_l21_2192

theorem game_prob_comparison
  (P_H : ℚ) (P_T : ℚ) (h : P_H = 3/4 ∧ P_T = 1/4)
  (independent : ∀ (n : ℕ), (1 - P_H)^n = (1 - P_T)^n) :
  ((P_H^4 + P_T^4) = (P_H^3 * P_T^2 + P_T^3 * P_H^2) + 1/4) :=
by
  sorry

end NUMINAMATH_GPT_game_prob_comparison_l21_2192


namespace NUMINAMATH_GPT_liam_balloons_remainder_l21_2136

def balloons : Nat := 24 + 45 + 78 + 96
def friends : Nat := 10
def remainder := balloons % friends

theorem liam_balloons_remainder : remainder = 3 := by
  sorry

end NUMINAMATH_GPT_liam_balloons_remainder_l21_2136


namespace NUMINAMATH_GPT_find_monthly_fee_l21_2172

variable (monthly_fee : ℝ) (cost_per_minute : ℝ := 0.12) (minutes_used : ℕ := 178) (total_bill : ℝ := 23.36)

theorem find_monthly_fee
  (h1 : total_bill = monthly_fee + (cost_per_minute * minutes_used)) :
  monthly_fee = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_monthly_fee_l21_2172


namespace NUMINAMATH_GPT_abs_eq_solution_diff_l21_2161

theorem abs_eq_solution_diff : 
  ∀ x₁ x₂ : ℝ, 
  (2 * x₁ - 3 = 18 ∨ 2 * x₁ - 3 = -18) → 
  (2 * x₂ - 3 = 18 ∨ 2 * x₂ - 3 = -18) → 
  |x₁ - x₂| = 18 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_solution_diff_l21_2161


namespace NUMINAMATH_GPT_roof_area_l21_2198

-- Definitions based on conditions
variables (l w : ℝ)
def length_eq_five_times_width : Prop := l = 5 * w
def length_minus_width_eq_48 : Prop := l - w = 48

-- Proof goal
def area_of_roof : Prop := l * w = 720

-- Lean 4 statement asserting the mathematical problem
theorem roof_area (l w : ℝ) 
  (H1 : length_eq_five_times_width l w)
  (H2 : length_minus_width_eq_48 l w) : 
  area_of_roof l w := 
  by sorry

end NUMINAMATH_GPT_roof_area_l21_2198


namespace NUMINAMATH_GPT_shirt_original_price_l21_2174

theorem shirt_original_price (P : ℝ) : 
  (18 = P * 0.75 * 0.75 * 0.90 * 1.15) → 
  P = 18 / (0.75 * 0.75 * 0.90 * 1.15) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_shirt_original_price_l21_2174


namespace NUMINAMATH_GPT_find_original_selling_price_l21_2147

noncomputable def original_selling_price (purchase_price : ℝ) := 
  1.10 * purchase_price

noncomputable def new_selling_price (purchase_price : ℝ) := 
  1.17 * purchase_price

theorem find_original_selling_price (P : ℝ)
  (h1 : new_selling_price P - original_selling_price P = 56) :
  original_selling_price P = 880 := by 
  sorry

end NUMINAMATH_GPT_find_original_selling_price_l21_2147


namespace NUMINAMATH_GPT_cabbages_difference_l21_2193

noncomputable def numCabbagesThisYear : ℕ := 4096
noncomputable def numCabbagesLastYear : ℕ := 3969
noncomputable def diffCabbages : ℕ := numCabbagesThisYear - numCabbagesLastYear

theorem cabbages_difference :
  diffCabbages = 127 := by
  sorry

end NUMINAMATH_GPT_cabbages_difference_l21_2193


namespace NUMINAMATH_GPT_find_n_plus_m_l21_2110

noncomputable def f (x : ℝ) := abs (Real.log x / Real.log 2)

theorem find_n_plus_m (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n)
    (h4 : f m = f n) (h5 : ∀ x, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
    n + m = 5 / 2 := sorry

end NUMINAMATH_GPT_find_n_plus_m_l21_2110


namespace NUMINAMATH_GPT_gcd_m_n_l21_2173

def m : ℕ := 333333
def n : ℕ := 7777777

theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Mathematical steps have been omitted as they are not needed
  sorry

end NUMINAMATH_GPT_gcd_m_n_l21_2173


namespace NUMINAMATH_GPT_parabola_y_axis_intersection_l21_2120

theorem parabola_y_axis_intersection:
  (∀ x y : ℝ, y = -2 * (x - 1)^2 - 3 → x = 0 → y = -5) :=
by
  intros x y h_eq h_x
  sorry

end NUMINAMATH_GPT_parabola_y_axis_intersection_l21_2120


namespace NUMINAMATH_GPT_crescent_moon_falcata_area_l21_2114

/-
Prove that the area of the crescent moon falcata, which is bounded by:
1. A portion of the circle with radius 4 centered at (0,0) in the second quadrant.
2. A portion of the circle with radius 2 centered at (0,2) in the second quadrant.
3. The line segment from (0,0) to (-4,0).
is equal to 6π.
-/
theorem crescent_moon_falcata_area :
  let radius_large := 4
  let radius_small := 2
  let area_large := (1 / 2) * (π * (radius_large ^ 2))
  let area_small := (1 / 2) * (π * (radius_small ^ 2))
  (area_large - area_small) = 6 * π := by
  sorry

end NUMINAMATH_GPT_crescent_moon_falcata_area_l21_2114


namespace NUMINAMATH_GPT_percentage_of_women_attended_picnic_l21_2113

variable (E : ℝ) -- total number of employees
variable (M : ℝ) -- number of men
variable (W : ℝ) -- number of women

-- 45% of all employees are men
axiom h1 : M = 0.45 * E
-- Rest of employees are women
axiom h2 : W = E - M
-- 20% of men attended the picnic
variable (x : ℝ) -- percentage of women who attended the picnic
axiom h3 : 0.20 * M + (x / 100) * W = 0.31000000000000007 * E

theorem percentage_of_women_attended_picnic : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_women_attended_picnic_l21_2113


namespace NUMINAMATH_GPT_math_problem_l21_2146

variable {x y z : ℝ}

def condition1 (x : ℝ) := x = 1.2 * 40
def condition2 (x y : ℝ) := y = x - 0.35 * x
def condition3 (x y z : ℝ) := z = (x + y) / 2

theorem math_problem (x y z : ℝ) (h1 : condition1 x) (h2 : condition2 x y) (h3 : condition3 x y z) :
  z = 39.6 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l21_2146


namespace NUMINAMATH_GPT_sum_of_repeating_decimal_digits_of_five_thirteenths_l21_2148

theorem sum_of_repeating_decimal_digits_of_five_thirteenths 
  (a b : ℕ)
  (h1 : 5 / 13 = (a * 10 + b) / 99)
  (h2 : (a * 10 + b) = 38) :
  a + b = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_repeating_decimal_digits_of_five_thirteenths_l21_2148


namespace NUMINAMATH_GPT_hours_rained_l21_2118

theorem hours_rained (total_hours non_rain_hours rained_hours : ℕ)
 (h_total : total_hours = 8)
 (h_non_rain : non_rain_hours = 6)
 (h_rain_eq : rained_hours = total_hours - non_rain_hours) :
 rained_hours = 2 := 
by
  sorry

end NUMINAMATH_GPT_hours_rained_l21_2118


namespace NUMINAMATH_GPT_find_other_number_l21_2144

theorem find_other_number (LCM HCF number1 number2 : ℕ) 
  (hLCM : LCM = 7700) 
  (hHCF : HCF = 11) 
  (hNumber1 : number1 = 308)
  (hProductEquality : number1 * number2 = LCM * HCF) :
  number2 = 275 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_other_number_l21_2144


namespace NUMINAMATH_GPT_pqr_value_l21_2155

theorem pqr_value
  (p q r : ℤ) -- p, q, and r are integers
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) -- non-zero condition
  (h1 : p + q + r = 27) -- sum condition
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 300 / (p * q * r) = 1) -- equation condition
  : p * q * r = 984 := 
sorry 

end NUMINAMATH_GPT_pqr_value_l21_2155


namespace NUMINAMATH_GPT_function_is_decreasing_on_R_l21_2145

def is_decreasing (a : ℝ) : Prop := a - 1 < 0

theorem function_is_decreasing_on_R (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing a :=
by
  sorry

end NUMINAMATH_GPT_function_is_decreasing_on_R_l21_2145


namespace NUMINAMATH_GPT_option_d_correct_l21_2121

theorem option_d_correct (a b : ℝ) (h : a * b < 0) : 
  (a / b + b / a) ≤ -2 := by
  sorry

end NUMINAMATH_GPT_option_d_correct_l21_2121


namespace NUMINAMATH_GPT_value_of_a_g_odd_iff_m_eq_one_l21_2104

noncomputable def f (a x : ℝ) : ℝ := a ^ x

noncomputable def g (m x a : ℝ) : ℝ := m - 2 / (f a x + 1)

theorem value_of_a
  (a : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_diff : ∀ x y : ℝ, x ∈ (Set.Icc 1 2) → y ∈ (Set.Icc 1 2) → abs (f a x - f a y) = 2) :
  a = 2 :=
sorry

theorem g_odd_iff_m_eq_one
  (a m : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_a_eq : a = 2) :
  (∀ x : ℝ, g m x a = -g m (-x) a) ↔ m = 1 :=
sorry

end NUMINAMATH_GPT_value_of_a_g_odd_iff_m_eq_one_l21_2104


namespace NUMINAMATH_GPT_curve_passes_through_fixed_point_l21_2164

theorem curve_passes_through_fixed_point (k : ℝ) (x y : ℝ) (h : k ≠ -1) :
  (x ^ 2 + y ^ 2 + 2 * k * x + (4 * k + 10) * y + 10 * k + 20 = 0) → (x = 1 ∧ y = -3) :=
by
  sorry

end NUMINAMATH_GPT_curve_passes_through_fixed_point_l21_2164


namespace NUMINAMATH_GPT_remainder_of_power_sums_modulo_seven_l21_2111

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_power_sums_modulo_seven_l21_2111


namespace NUMINAMATH_GPT_piggy_bank_donation_l21_2115

theorem piggy_bank_donation (total_earnings : ℕ) (cost_of_ingredients : ℕ) 
  (total_donation_homeless_shelter : ℕ) : 
  (total_earnings = 400) → (cost_of_ingredients = 100) → (total_donation_homeless_shelter = 160) → 
  (total_donation_homeless_shelter - (total_earnings - cost_of_ingredients) / 2 = 10) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_piggy_bank_donation_l21_2115


namespace NUMINAMATH_GPT_apples_per_box_l21_2149

theorem apples_per_box (x : ℕ) (h1 : 10 * x > 0) (h2 : 3 * (10 * x) / 4 > 0) (h3 : (10 * x) / 4 = 750) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_apples_per_box_l21_2149


namespace NUMINAMATH_GPT_triangle_XYZ_median_l21_2194

theorem triangle_XYZ_median (XYZ : Triangle) (YZ : ℝ) (XM : ℝ) (XY2_add_XZ2 : ℝ) 
  (hYZ : YZ = 12) (hXM : XM = 7) : XY2_add_XZ2 = 170 → N - n = 0 := by
  sorry

end NUMINAMATH_GPT_triangle_XYZ_median_l21_2194


namespace NUMINAMATH_GPT_work_completion_days_l21_2182

variables (M D X : ℕ) (W : ℝ)

-- Original conditions
def original_men : ℕ := 15
def planned_days : ℕ := 40
def men_absent : ℕ := 5

-- Theorem to prove
theorem work_completion_days :
  M = original_men →
  D = planned_days →
  W > 0 →
  (M - men_absent) * X * W = M * D * W →
  X = 60 :=
by
  intros hM hD hW h_work
  sorry

end NUMINAMATH_GPT_work_completion_days_l21_2182


namespace NUMINAMATH_GPT_train_speed_l21_2132

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 700) (h_time : time = 40) : length / time = 17.5 :=
by
  -- length / time represents the speed of the train
  -- given length = 700 meters and time = 40 seconds
  -- we have to prove that 700 / 40 = 17.5
  sorry

end NUMINAMATH_GPT_train_speed_l21_2132


namespace NUMINAMATH_GPT_width_of_field_l21_2107

-- Definitions for the conditions
variables (W L : ℝ) (P : ℝ)
axiom length_condition : L = (7 / 5) * W
axiom perimeter_condition : P = 2 * L + 2 * W
axiom perimeter_value : P = 336

-- Theorem to be proved
theorem width_of_field : W = 70 :=
by
  -- Here will be the proof body
  sorry

end NUMINAMATH_GPT_width_of_field_l21_2107


namespace NUMINAMATH_GPT_bucket_weight_full_l21_2175

variable (p q x y : ℝ)

theorem bucket_weight_full (h1 : x + (3 / 4) * y = p)
                           (h2 : x + (1 / 3) * y = q) :
  x + y = (1 / 5) * (8 * p - 3 * q) :=
by
  sorry

end NUMINAMATH_GPT_bucket_weight_full_l21_2175


namespace NUMINAMATH_GPT_basketball_weight_l21_2162

theorem basketball_weight (b s : ℝ) (h1 : s = 20) (h2 : 5 * b = 4 * s) : b = 16 :=
by
  sorry

end NUMINAMATH_GPT_basketball_weight_l21_2162


namespace NUMINAMATH_GPT_number_of_students_passed_both_tests_l21_2131

theorem number_of_students_passed_both_tests 
  (total_students : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both_tests : ℕ) 
  (students_with_union : ℕ := total_students) :
  (students_with_union = passed_long_jump + passed_shot_put - passed_both_tests + failed_both_tests) 
  → passed_both_tests = 25 :=
by sorry

end NUMINAMATH_GPT_number_of_students_passed_both_tests_l21_2131


namespace NUMINAMATH_GPT_parabola_from_hyperbola_l21_2177

noncomputable def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

noncomputable def parabola_equation_1 (x y : ℝ) : Prop := y^2 = -24 * x

noncomputable def parabola_equation_2 (x y : ℝ) : Prop := y^2 = 24 * x

theorem parabola_from_hyperbola :
  (∃ x y : ℝ, hyperbola_equation x y) →
  (∃ x y : ℝ, parabola_equation_1 x y ∨ parabola_equation_2 x y) :=
by
  intro h
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_parabola_from_hyperbola_l21_2177


namespace NUMINAMATH_GPT_x_y_divisible_by_7_l21_2124

theorem x_y_divisible_by_7
  (x y a b : ℤ)
  (hx : 3 * x + 4 * y = a ^ 2)
  (hy : 4 * x + 3 * y = b ^ 2)
  (hx_pos : x > 0) (hy_pos : y > 0) :
  7 ∣ x ∧ 7 ∣ y :=
by
  sorry

end NUMINAMATH_GPT_x_y_divisible_by_7_l21_2124


namespace NUMINAMATH_GPT_calculation_correct_l21_2189

theorem calculation_correct : 1984 + 180 / 60 - 284 = 1703 := 
by 
  sorry

end NUMINAMATH_GPT_calculation_correct_l21_2189


namespace NUMINAMATH_GPT_time_to_walk_l21_2186

variable (v l r w : ℝ)
variable (h1 : l = 15 * (v + r))
variable (h2 : l = 30 * (v + w))
variable (h3 : l = 20 * r)

theorem time_to_walk (h1 : l = 15 * (v + r)) (h2 : l = 30 * (v + w)) (h3 : l = 20 * r) : l / w = 60 := 
by sorry

end NUMINAMATH_GPT_time_to_walk_l21_2186


namespace NUMINAMATH_GPT_op_value_l21_2191

def op (x y : ℕ) : ℕ := x^3 - 3*x*y^2 + y^3

theorem op_value :
  op 2 1 = 3 := by sorry

end NUMINAMATH_GPT_op_value_l21_2191


namespace NUMINAMATH_GPT_toms_total_score_l21_2127

def regular_enemy_points : ℕ := 10
def elite_enemy_points : ℕ := 25
def boss_enemy_points : ℕ := 50

def regular_enemy_bonus (kills : ℕ) : ℚ :=
  if 100 ≤ kills ∧ kills < 150 then 0.50
  else if 150 ≤ kills ∧ kills < 200 then 0.75
  else if kills ≥ 200 then 1.00
  else 0.00

def elite_enemy_bonus (kills : ℕ) : ℚ :=
  if 15 ≤ kills ∧ kills < 25 then 0.30
  else if 25 ≤ kills ∧ kills < 35 then 0.50
  else if kills >= 35 then 0.70
  else 0.00

def boss_enemy_bonus (kills : ℕ) : ℚ :=
  if 5 ≤ kills ∧ kills < 10 then 0.20
  else if kills ≥ 10 then 0.40
  else 0.00

noncomputable def total_score (regular_kills elite_kills boss_kills : ℕ) : ℚ :=
  let regular_points := regular_kills * regular_enemy_points
  let elite_points := elite_kills * elite_enemy_points
  let boss_points := boss_kills * boss_enemy_points
  let regular_total := regular_points + regular_points * regular_enemy_bonus regular_kills
  let elite_total := elite_points + elite_points * elite_enemy_bonus elite_kills
  let boss_total := boss_points + boss_points * boss_enemy_bonus boss_kills
  regular_total + elite_total + boss_total

theorem toms_total_score :
  total_score 160 20 8 = 3930 := by
  sorry

end NUMINAMATH_GPT_toms_total_score_l21_2127


namespace NUMINAMATH_GPT_split_numbers_cubic_l21_2154

theorem split_numbers_cubic (m : ℕ) (hm : 1 < m) (assumption : m^2 - m + 1 = 73) : m = 9 :=
sorry

end NUMINAMATH_GPT_split_numbers_cubic_l21_2154


namespace NUMINAMATH_GPT_puppy_cost_l21_2187

variable (P : ℕ)

theorem puppy_cost (hc : 2 * 50 = 100) (hd : 3 * 100 = 300) (htotal : 2 * 50 + 3 * 100 + 2 * P = 700) : P = 150 :=
by
  sorry

end NUMINAMATH_GPT_puppy_cost_l21_2187


namespace NUMINAMATH_GPT_inequality_abc_l21_2181

open Real

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b * c = 1) : 
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_abc_l21_2181


namespace NUMINAMATH_GPT_nested_fraction_l21_2158

theorem nested_fraction
  : 1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5))))))
  = 968 / 3191 := 
by
  sorry

end NUMINAMATH_GPT_nested_fraction_l21_2158


namespace NUMINAMATH_GPT_powers_of_2_form_6n_plus_8_l21_2119

noncomputable def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2 ^ k

def of_the_form (n : ℕ) : ℕ := 6 * n + 8

def is_odd_greater_than_one (k : ℕ) : Prop := k % 2 = 1 ∧ k > 1

theorem powers_of_2_form_6n_plus_8 (k : ℕ) (n : ℕ) :
  (2 ^ k = of_the_form n) ↔ is_odd_greater_than_one k :=
sorry

end NUMINAMATH_GPT_powers_of_2_form_6n_plus_8_l21_2119


namespace NUMINAMATH_GPT_steps_left_to_climb_l21_2179

-- Define the conditions
def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

-- The problem: Prove that the number of stairs left to climb is 22
theorem steps_left_to_climb : (total_stairs - climbed_stairs) = 22 :=
by 
  sorry

end NUMINAMATH_GPT_steps_left_to_climb_l21_2179


namespace NUMINAMATH_GPT_largest_integer_x_cubed_lt_three_x_squared_l21_2129

theorem largest_integer_x_cubed_lt_three_x_squared : 
  ∃ x : ℤ, x^3 < 3 * x^2 ∧ (∀ y : ℤ, y^3 < 3 * y^2 → y ≤ x) :=
  sorry

end NUMINAMATH_GPT_largest_integer_x_cubed_lt_three_x_squared_l21_2129


namespace NUMINAMATH_GPT_batsman_average_after_20th_innings_l21_2122

theorem batsman_average_after_20th_innings 
    (score_20th_innings : ℕ)
    (previous_avg_increase : ℕ)
    (total_innings : ℕ)
    (never_not_out : Prop)
    (previous_avg : ℕ)
    : score_20th_innings = 90 →
      previous_avg_increase = 2 →
      total_innings = 20 →
      previous_avg = (19 * previous_avg + score_20th_innings) / total_innings →
      ((19 * previous_avg + score_20th_innings) / total_innings) + previous_avg_increase = 52 :=
by 
  sorry

end NUMINAMATH_GPT_batsman_average_after_20th_innings_l21_2122


namespace NUMINAMATH_GPT_march_first_is_tuesday_l21_2156

theorem march_first_is_tuesday (march_15_tuesday : true) :
  true :=
sorry

end NUMINAMATH_GPT_march_first_is_tuesday_l21_2156


namespace NUMINAMATH_GPT_fixed_point_exists_line_intersects_circle_shortest_chord_l21_2116

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25
noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem fixed_point_exists : ∃ P : ℝ × ℝ, (∀ m : ℝ, line_l P.1 P.2 m) ∧ P = (3, 1) :=
by
  sorry

theorem line_intersects_circle : ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by
  sorry

theorem shortest_chord : ∃ m : ℝ, m = -3/4 ∧ (∀ x y, line_l x y m ↔ 2 * x - y - 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_exists_line_intersects_circle_shortest_chord_l21_2116


namespace NUMINAMATH_GPT_driving_hours_fresh_l21_2170

theorem driving_hours_fresh (x : ℚ) : (25 * x + 15 * (9 - x) = 152) → x = 17 / 10 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_driving_hours_fresh_l21_2170


namespace NUMINAMATH_GPT_prime_factors_difference_l21_2141

theorem prime_factors_difference (n : ℤ) (h₁ : n = 180181) : ∃ p q : ℤ, Prime p ∧ Prime q ∧ p > q ∧ n % p = 0 ∧ n % q = 0 ∧ (p - q) = 2 :=
by
  sorry

end NUMINAMATH_GPT_prime_factors_difference_l21_2141


namespace NUMINAMATH_GPT_Megan_full_folders_l21_2169

def initial_files : ℕ := 256
def deleted_files : ℕ := 67
def files_per_folder : ℕ := 12

def remaining_files : ℕ := initial_files - deleted_files
def number_of_folders : ℕ := remaining_files / files_per_folder

theorem Megan_full_folders : number_of_folders = 15 := by
  sorry

end NUMINAMATH_GPT_Megan_full_folders_l21_2169


namespace NUMINAMATH_GPT_inequality_must_hold_l21_2184

theorem inequality_must_hold (x y : ℝ) (h : x > y) : -2 * x < -2 * y :=
sorry

end NUMINAMATH_GPT_inequality_must_hold_l21_2184


namespace NUMINAMATH_GPT_noah_yearly_bill_l21_2138

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end NUMINAMATH_GPT_noah_yearly_bill_l21_2138


namespace NUMINAMATH_GPT_width_of_wall_l21_2102

theorem width_of_wall (l : ℕ) (w : ℕ) (hl : l = 170) (hw : w = 5 * l + 80) : w = 930 := 
by
  sorry

end NUMINAMATH_GPT_width_of_wall_l21_2102


namespace NUMINAMATH_GPT_white_paint_amount_l21_2128

theorem white_paint_amount (total_paint green_paint brown_paint : ℕ) 
  (h_total : total_paint = 69)
  (h_green : green_paint = 15)
  (h_brown : brown_paint = 34) :
  total_paint - (green_paint + brown_paint) = 20 := by
  sorry

end NUMINAMATH_GPT_white_paint_amount_l21_2128


namespace NUMINAMATH_GPT_value_of_x_l21_2176

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l21_2176


namespace NUMINAMATH_GPT_four_digit_numbers_count_l21_2151

open Nat

theorem four_digit_numbers_count :
  let valid_a := [5, 6]
  let valid_d := 0
  let valid_bc_pairs := [(3, 4), (3, 6)]
  valid_a.length * 1 * valid_bc_pairs.length = 4 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_count_l21_2151


namespace NUMINAMATH_GPT_min_value_of_xsquare_ysquare_l21_2160

variable {x y : ℝ}

theorem min_value_of_xsquare_ysquare (h : 5 * x^2 * y^2 + y^4 = 1) : x^2 + y^2 ≥ 4 / 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_xsquare_ysquare_l21_2160


namespace NUMINAMATH_GPT_good_numbers_count_l21_2150

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end NUMINAMATH_GPT_good_numbers_count_l21_2150


namespace NUMINAMATH_GPT_parrots_per_cage_l21_2153

theorem parrots_per_cage (P : ℕ) (parakeets_per_cage : ℕ) (cages : ℕ) (total_birds : ℕ) 
    (h1 : parakeets_per_cage = 7) (h2 : cages = 8) (h3 : total_birds = 72) 
    (h4 : total_birds = cages * P + cages * parakeets_per_cage) : 
    P = 2 :=
by
  sorry

end NUMINAMATH_GPT_parrots_per_cage_l21_2153


namespace NUMINAMATH_GPT_max_at_zero_l21_2159

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem max_at_zero : ∃ x, (∀ y, f y ≤ f x) ∧ x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_max_at_zero_l21_2159


namespace NUMINAMATH_GPT_range_of_set_l21_2139

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_set_l21_2139


namespace NUMINAMATH_GPT_total_players_count_l21_2117

def kabadi_players : ℕ := 10
def kho_kho_only_players : ℕ := 35
def both_games_players : ℕ := 5

theorem total_players_count : kabadi_players + kho_kho_only_players - both_games_players = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_players_count_l21_2117


namespace NUMINAMATH_GPT_heartsuit_zero_heartsuit_self_heartsuit_pos_l21_2126

def heartsuit (x y : Real) : Real := x^2 - y^2

theorem heartsuit_zero (x : Real) : heartsuit x 0 = x^2 :=
by
  sorry

theorem heartsuit_self (x : Real) : heartsuit x x = 0 :=
by
  sorry

theorem heartsuit_pos (x y : Real) (h : x > y) : heartsuit x y > 0 :=
by
  sorry

end NUMINAMATH_GPT_heartsuit_zero_heartsuit_self_heartsuit_pos_l21_2126


namespace NUMINAMATH_GPT_no_real_roots_l21_2123

def op (m n : ℝ) : ℝ := n^2 - m * n + 1

theorem no_real_roots (x : ℝ) : op 1 x = 0 → ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_real_roots_l21_2123


namespace NUMINAMATH_GPT_ap_sub_aq_l21_2196

variable {n : ℕ} (hn : n > 0)

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) (hn : n > 0) : ℕ :=
S n - S (n - 1)

theorem ap_sub_aq (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : p - q = 5) :
  a p hp - a q hq = 20 :=
sorry

end NUMINAMATH_GPT_ap_sub_aq_l21_2196


namespace NUMINAMATH_GPT_solve_for_x_l21_2112

theorem solve_for_x (x : ℝ) (h : 4^x = Real.sqrt 64) : x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l21_2112


namespace NUMINAMATH_GPT_borrowed_dimes_calculation_l21_2185

-- Define Sam's initial dimes and remaining dimes after borrowing
def original_dimes : ℕ := 8
def remaining_dimes : ℕ := 4

-- Statement to prove that the borrowed dimes is 4
theorem borrowed_dimes_calculation : (original_dimes - remaining_dimes) = 4 :=
by
  -- This is the proof section which follows by simple arithmetic computation
  sorry

end NUMINAMATH_GPT_borrowed_dimes_calculation_l21_2185


namespace NUMINAMATH_GPT_jane_can_buy_9_tickets_l21_2137

-- Definitions
def ticket_price : ℕ := 15
def jane_amount_initial : ℕ := 160
def scarf_cost : ℕ := 25
def jane_amount_after_scarf : ℕ := jane_amount_initial - scarf_cost
def max_tickets (amount : ℕ) (price : ℕ) := amount / price

-- The main statement
theorem jane_can_buy_9_tickets :
  max_tickets jane_amount_after_scarf ticket_price = 9 :=
by
  -- Proof goes here (proof steps would be outlined)
  sorry

end NUMINAMATH_GPT_jane_can_buy_9_tickets_l21_2137


namespace NUMINAMATH_GPT_midpoint_sum_l21_2100

theorem midpoint_sum :
  let x1 := 8
  let y1 := -4
  let z1 := 10
  let x2 := -2
  let y2 := 10
  let z2 := -6
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  let midpoint_z := (z1 + z2) / 2
  midpoint_x + midpoint_y + midpoint_z = 8 :=
by
  -- We just need to state the theorem, proof is not required
  sorry

end NUMINAMATH_GPT_midpoint_sum_l21_2100


namespace NUMINAMATH_GPT_probability_of_x_plus_y_less_than_4_l21_2133

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end NUMINAMATH_GPT_probability_of_x_plus_y_less_than_4_l21_2133


namespace NUMINAMATH_GPT_work_completion_days_l21_2105

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l21_2105


namespace NUMINAMATH_GPT_option_B_correct_l21_2125

-- Define the commutativity of multiplication
def commutativity_of_mul (a b : Nat) : Prop :=
  a * b = b * a

-- State the problem, which is to prove that 2ab + 3ba = 5ab given commutativity
theorem option_B_correct (a b : Nat) : commutativity_of_mul a b → 2 * (a * b) + 3 * (b * a) = 5 * (a * b) :=
by
  intro h_comm
  rw [←h_comm]
  sorry

end NUMINAMATH_GPT_option_B_correct_l21_2125


namespace NUMINAMATH_GPT_Problem_statements_l21_2167

theorem Problem_statements (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = a * b) :
  (a + b ≥ 4) ∧
  ¬(a * b ≤ 4) ∧
  (a + 4 * b ≥ 9) ∧
  (1 / a ^ 2 + 2 / b ^ 2 ≥ 2 / 3) :=
by sorry

end NUMINAMATH_GPT_Problem_statements_l21_2167


namespace NUMINAMATH_GPT_initial_amount_l21_2199

theorem initial_amount (M : ℝ) 
  (H1 : M * (2/3) * (4/5) * (3/4) * (5/7) * (5/6) = 200) : 
  M = 840 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_initial_amount_l21_2199


namespace NUMINAMATH_GPT_powers_greater_than_thresholds_l21_2135

theorem powers_greater_than_thresholds :
  (1.01^2778 > 1000000000000) ∧
  (1.001^27632 > 1000000000000) ∧
  (1.000001^27631000 > 1000000000000) ∧
  (1.01^4165 > 1000000000000000000) ∧
  (1.001^41447 > 1000000000000000000) ∧
  (1.000001^41446000 > 1000000000000000000) :=
by sorry

end NUMINAMATH_GPT_powers_greater_than_thresholds_l21_2135


namespace NUMINAMATH_GPT_company_total_payment_correct_l21_2178

def totalEmployees : Nat := 450
def firstGroup : Nat := 150
def secondGroup : Nat := 200
def thirdGroup : Nat := 100

def firstBaseSalary : Nat := 2000
def secondBaseSalary : Nat := 2500
def thirdBaseSalary : Nat := 3000

def firstInitialBonus : Nat := 500
def secondInitialBenefit : Nat := 400
def thirdInitialBenefit : Nat := 600

def firstLayoffRound1 : Nat := (20 * firstGroup) / 100
def secondLayoffRound1 : Nat := (25 * secondGroup) / 100
def thirdLayoffRound1 : Nat := (15 * thirdGroup) / 100

def remainingFirstGroupRound1 : Nat := firstGroup - firstLayoffRound1
def remainingSecondGroupRound1 : Nat := secondGroup - secondLayoffRound1
def remainingThirdGroupRound1 : Nat := thirdGroup - thirdLayoffRound1

def firstAdjustedBonusRound1 : Nat := 400
def secondAdjustedBenefitRound1 : Nat := 300

def firstLayoffRound2 : Nat := (10 * remainingFirstGroupRound1) / 100
def secondLayoffRound2 : Nat := (15 * remainingSecondGroupRound1) / 100
def thirdLayoffRound2 : Nat := (5 * remainingThirdGroupRound1) / 100

def remainingFirstGroupRound2 : Nat := remainingFirstGroupRound1 - firstLayoffRound2
def remainingSecondGroupRound2 : Nat := remainingSecondGroupRound1 - secondLayoffRound2
def remainingThirdGroupRound2 : Nat := remainingThirdGroupRound1 - thirdLayoffRound2

def thirdAdjustedBenefitRound2 : Nat := (80 * thirdInitialBenefit) / 100

def totalBaseSalary : Nat :=
  (remainingFirstGroupRound2 * firstBaseSalary)
  + (remainingSecondGroupRound2 * secondBaseSalary)
  + (remainingThirdGroupRound2 * thirdBaseSalary)

def totalBonusesAndBenefits : Nat :=
  (remainingFirstGroupRound2 * firstAdjustedBonusRound1)
  + (remainingSecondGroupRound2 * secondAdjustedBenefitRound1)
  + (remainingThirdGroupRound2 * thirdAdjustedBenefitRound2)

def totalPayment : Nat :=
  totalBaseSalary + totalBonusesAndBenefits

theorem company_total_payment_correct :
  totalPayment = 893200 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_company_total_payment_correct_l21_2178


namespace NUMINAMATH_GPT_bankers_gain_is_126_l21_2171

-- Define the given conditions
def present_worth : ℝ := 600
def interest_rate : ℝ := 0.10
def time_period : ℕ := 2

-- Define the formula for compound interest to find the amount due A
def amount_due (PW : ℝ) (R : ℝ) (T : ℕ) : ℝ := PW * (1 + R) ^ T

-- Define the banker's gain as the difference between the amount due and the present worth
def bankers_gain (A : ℝ) (PW : ℝ) : ℝ := A - PW

-- The theorem to prove that the banker's gain is Rs. 126 given the conditions
theorem bankers_gain_is_126 : bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 126 := by
  sorry

end NUMINAMATH_GPT_bankers_gain_is_126_l21_2171


namespace NUMINAMATH_GPT_maximize_value_l21_2106

def f (x : ℝ) : ℝ := -3 * x^2 - 8 * x + 18

theorem maximize_value : ∀ x : ℝ, f x ≤ f (-4/3) :=
by sorry

end NUMINAMATH_GPT_maximize_value_l21_2106


namespace NUMINAMATH_GPT_alex_serge_equiv_distinct_values_l21_2190

-- Defining the context and data structures
variable {n : ℕ} -- Number of boxes
variable {c : ℕ → ℕ} -- Function representing number of cookies in each box, indexed by box number
variable {m : ℕ} -- Number of plates
variable {p : ℕ → ℕ} -- Function representing number of cookies on each plate, indexed by plate number

-- Define the sets representing the unique counts recorded by Alex and Serge
def Alex_record (c : ℕ → ℕ) (n : ℕ) : Set ℕ := 
  { x | ∃ i, i < n ∧ c i = x }

def Serge_record (p : ℕ → ℕ) (m : ℕ) : Set ℕ := 
  { y | ∃ j, j < m ∧ p j = y }

-- The proof goal: Alex's record contains the same number of distinct values as Serge's record
theorem alex_serge_equiv_distinct_values
  (c : ℕ → ℕ) (n : ℕ) (p : ℕ → ℕ) (m : ℕ) :
  Alex_record c n = Serge_record p m :=
sorry

end NUMINAMATH_GPT_alex_serge_equiv_distinct_values_l21_2190


namespace NUMINAMATH_GPT_positive_numbers_l21_2143

theorem positive_numbers {a b c : ℝ} (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end NUMINAMATH_GPT_positive_numbers_l21_2143


namespace NUMINAMATH_GPT_number_line_is_line_l21_2188

-- Define the terms
def number_line : Type := ℝ -- Assume number line can be considered real numbers for simplicity
def is_line (l : Type) : Prop := l = ℝ

-- Proving that number line is a line.
theorem number_line_is_line : is_line number_line :=
by {
  -- by definition of the number_line and is_line
  sorry
}

end NUMINAMATH_GPT_number_line_is_line_l21_2188


namespace NUMINAMATH_GPT_fourth_year_students_without_glasses_l21_2157

theorem fourth_year_students_without_glasses (total_students: ℕ) (x: ℕ) (y: ℕ) 
  (h1: total_students = 1152) 
  (h2: total_students = 8 * x - 32) 
  (h3: x = 148) 
  (h4: 2 * y + 10 = x) 
  : y = 69 :=
by {
sorry
}

end NUMINAMATH_GPT_fourth_year_students_without_glasses_l21_2157


namespace NUMINAMATH_GPT_minimum_treasure_buried_l21_2103

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end NUMINAMATH_GPT_minimum_treasure_buried_l21_2103


namespace NUMINAMATH_GPT_find_f_8_6_l21_2163

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem find_f_8_6 (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_def : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = - (1 / 2) * x) :
  f 8.6 = 0.3 :=
sorry

end NUMINAMATH_GPT_find_f_8_6_l21_2163


namespace NUMINAMATH_GPT_solution_of_system_l21_2108

noncomputable def system_of_equations (x y : ℝ) :=
  x = 1.12 * y + 52.8 ∧ x = y + 50

theorem solution_of_system : 
  ∃ (x y : ℝ), system_of_equations x y ∧ y = -23.33 ∧ x = 26.67 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_system_l21_2108


namespace NUMINAMATH_GPT_train_speed_l21_2183

-- Definitions of the given conditions
def platform_length : ℝ := 250
def train_length : ℝ := 470.06
def time_taken : ℝ := 36

-- Definition of the total distance covered
def total_distance := platform_length + train_length

-- The proof problem: Prove that the calculated speed is approximately 20.0017 m/s
theorem train_speed :
  (total_distance / time_taken) = 20.0017 :=
by
  -- The actual proof goes here, but for now we leave it as sorry
  sorry

end NUMINAMATH_GPT_train_speed_l21_2183


namespace NUMINAMATH_GPT_inequality_for_natural_n_l21_2195

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n :=
by sorry

end NUMINAMATH_GPT_inequality_for_natural_n_l21_2195


namespace NUMINAMATH_GPT_problem_l21_2109

variable (a : ℕ → ℝ) (n m : ℕ)

-- Condition: non-negative sequence and a_{n+m} ≤ a_n + a_m
axiom condition (n m : ℕ) : a n ≥ 0 ∧ a (n + m) ≤ a n + a m

-- Theorem: for any n ≥ m
theorem problem (h : n ≥ m) : a n ≤ m * a 1 + ((n / m) - 1) * a m :=
sorry

end NUMINAMATH_GPT_problem_l21_2109


namespace NUMINAMATH_GPT_central_angle_agree_l21_2180

theorem central_angle_agree (ratio_agree : ℕ) (ratio_disagree : ℕ) (ratio_no_preference : ℕ) (total_angle : ℝ) :
  ratio_agree = 7 → ratio_disagree = 2 → ratio_no_preference = 1 → total_angle = 360 →
  (ratio_agree / (ratio_agree + ratio_disagree + ratio_no_preference) * total_angle = 252) :=
by
  -- conditions and assumptions
  intros h_agree h_disagree h_no_preference h_total_angle
  -- simplified steps here
  sorry

end NUMINAMATH_GPT_central_angle_agree_l21_2180


namespace NUMINAMATH_GPT_circle_symmetry_y_axis_eq_l21_2134

theorem circle_symmetry_y_axis_eq (x y : ℝ) :
  (x^2 + y^2 + 2 * x = 0) ↔ (x^2 + y^2 - 2 * x = 0) :=
sorry

end NUMINAMATH_GPT_circle_symmetry_y_axis_eq_l21_2134


namespace NUMINAMATH_GPT_solve_fraction_zero_l21_2165

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 16) / (4 - x) = 0) (h2 : 4 - x ≠ 0) : x = -4 :=
sorry

end NUMINAMATH_GPT_solve_fraction_zero_l21_2165
