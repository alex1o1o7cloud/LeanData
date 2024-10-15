import Mathlib

namespace NUMINAMATH_GPT_inverse_of_5_mod_35_l2_201

theorem inverse_of_5_mod_35 : (5 * 28) % 35 = 1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_5_mod_35_l2_201


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2_248

theorem problem1 : 12 - (-1) + (-7) = 6 := by
  sorry

theorem problem2 : -3.5 * (-3 / 4) / (7 / 8) = 3 := by
  sorry

theorem problem3 : (1 / 3 - 1 / 6 - 1 / 12) * (-12) = -1 := by
  sorry

theorem problem4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2_248


namespace NUMINAMATH_GPT_sheila_hourly_wage_l2_223

-- Sheila works 8 hours per day on Monday, Wednesday, and Friday
-- Sheila works 6 hours per day on Tuesday and Thursday
-- Sheila does not work on Saturday and Sunday
-- Sheila earns $288 per week

def hours_worked (monday_wednesday_friday_hours : Nat) (tuesday_thursday_hours : Nat) : Nat :=
  (monday_wednesday_friday_hours * 3) + (tuesday_thursday_hours * 2)

def weekly_earnings : Nat := 288
def total_hours_worked : Nat := hours_worked 8 6
def hourly_wage : Nat := weekly_earnings / total_hours_worked

theorem sheila_hourly_wage : hourly_wage = 8 := by
  -- Proof (omitted)
  sorry

end NUMINAMATH_GPT_sheila_hourly_wage_l2_223


namespace NUMINAMATH_GPT_sampling_prob_equal_l2_206

theorem sampling_prob_equal (N n : ℕ) (P_1 P_2 P_3 : ℝ)
  (H_random : ∀ i, 1 ≤ i ∧ i ≤ N → P_1 = 1 / N)
  (H_systematic : ∀ i, 1 ≤ i ∧ i ≤ N → P_2 = 1 / N)
  (H_stratified : ∀ i, 1 ≤ i ∧ i ≤ N → P_3 = 1 / N) :
  P_1 = P_2 ∧ P_2 = P_3 :=
by
  sorry

end NUMINAMATH_GPT_sampling_prob_equal_l2_206


namespace NUMINAMATH_GPT_calculate_t_minus_d_l2_229

def tom_pays : ℕ := 150
def dorothy_pays : ℕ := 190
def sammy_pays : ℕ := 240
def nancy_pays : ℕ := 320
def total_expenses := tom_pays + dorothy_pays + sammy_pays + nancy_pays
def individual_share := total_expenses / 4
def tom_needs_to_pay := individual_share - tom_pays
def dorothy_needs_to_pay := individual_share - dorothy_pays
def sammy_should_receive := sammy_pays - individual_share
def nancy_should_receive := nancy_pays - individual_share
def t := tom_needs_to_pay
def d := dorothy_needs_to_pay

theorem calculate_t_minus_d : t - d = 40 :=
by
  sorry

end NUMINAMATH_GPT_calculate_t_minus_d_l2_229


namespace NUMINAMATH_GPT_arun_age_l2_250

variable (A S G M : ℕ)

theorem arun_age (h1 : A - 6 = 18 * G)
                 (h2 : G + 2 = M)
                 (h3 : M = 5)
                 (h4 : S = A - 8) : A = 60 :=
by sorry

end NUMINAMATH_GPT_arun_age_l2_250


namespace NUMINAMATH_GPT_quadratic_root_value_l2_293

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end NUMINAMATH_GPT_quadratic_root_value_l2_293


namespace NUMINAMATH_GPT_proof_problem_l2_230

-- Definitions based on the conditions from the problem
def optionA (A : Set α) : Prop := ∅ ∩ A = ∅

def optionC : Prop := { y | ∃ x, y = 1 / x } = { z | ∃ t, z = 1 / t }

-- The main theorem statement
theorem proof_problem (A : Set α) : optionA A ∧ optionC := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_proof_problem_l2_230


namespace NUMINAMATH_GPT_cube_volume_l2_264

-- Define the condition: the surface area of the cube is 54
def surface_area_of_cube (x : ℝ) : Prop := 6 * x^2 = 54

-- Define the theorem that states the volume of the cube given the surface area condition
theorem cube_volume : ∃ (x : ℝ), surface_area_of_cube x ∧ x^3 = 27 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l2_264


namespace NUMINAMATH_GPT_wood_burned_afternoon_l2_275

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end NUMINAMATH_GPT_wood_burned_afternoon_l2_275


namespace NUMINAMATH_GPT_factor_expression_1_factor_expression_2_l2_221

theorem factor_expression_1 (a b c : ℝ) : a^2 + 2 * a * b + b^2 + a * c + b * c = (a + b) * (a + b + c) :=
  sorry

theorem factor_expression_2 (a x y : ℝ) : 4 * a^2 - x^2 + 4 * x * y - 4 * y^2 = (2 * a + x - 2 * y) * (2 * a - x + 2 * y) :=
  sorry

end NUMINAMATH_GPT_factor_expression_1_factor_expression_2_l2_221


namespace NUMINAMATH_GPT_Martha_needs_54_cakes_l2_246

theorem Martha_needs_54_cakes :
  let n_children := 3
  let n_cakes_per_child := 18
  let n_cakes_total := 54
  n_cakes_total = n_children * n_cakes_per_child :=
by
  sorry

end NUMINAMATH_GPT_Martha_needs_54_cakes_l2_246


namespace NUMINAMATH_GPT_h_of_neg2_eq_11_l2_215

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x ^ 2 + 1
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg2_eq_11 : h (-2) = 11 := by
  sorry

end NUMINAMATH_GPT_h_of_neg2_eq_11_l2_215


namespace NUMINAMATH_GPT_stickers_decorate_l2_291

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end NUMINAMATH_GPT_stickers_decorate_l2_291


namespace NUMINAMATH_GPT_even_function_and_monotonicity_l2_228

noncomputable def f (x : ℝ) : ℝ := sorry

theorem even_function_and_monotonicity (f_symm : ∀ x : ℝ, f x = f (-x))
  (f_inc_neg : ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → x1 ≤ 0 → x2 ≤ 0 → f x1 < f x2)
  (n : ℕ) (hn : n > 0) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := 
sorry

end NUMINAMATH_GPT_even_function_and_monotonicity_l2_228


namespace NUMINAMATH_GPT_dice_sum_probability_15_l2_204
open Nat

theorem dice_sum_probability_15 (n : ℕ) (h : n = 3432) : 
  ∃ d1 d2 d3 d4 d5 d6 d7 d8 : ℕ,
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ 
  (1 ≤ d3 ∧ d3 ≤ 6) ∧ (1 ≤ d4 ∧ d4 ≤ 6) ∧ 
  (1 ≤ d5 ∧ d5 ≤ 6) ∧ (1 ≤ d6 ∧ d6 ≤ 6) ∧ 
  (1 ≤ d7 ∧ d7 ≤ 6) ∧ (1 ≤ d8 ∧ d8 ≤ 6) ∧ 
  (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 = 15) :=
by
  sorry

end NUMINAMATH_GPT_dice_sum_probability_15_l2_204


namespace NUMINAMATH_GPT_gcd_gx_x_l2_208

noncomputable def g (x : ℕ) := (5 * x + 3) * (11 * x + 2) * (6 * x + 7) * (3 * x + 8)

theorem gcd_gx_x {x : ℕ} (hx : 36000 ∣ x) : Nat.gcd (g x) x = 144 := by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_l2_208


namespace NUMINAMATH_GPT_find_missing_values_l2_200

theorem find_missing_values :
  (∃ x y : ℕ, 4 / 5 = 20 / x ∧ 4 / 5 = y / 20 ∧ 4 / 5 = 80 / 100) →
  (x = 25 ∧ y = 16 ∧ 4 / 5 = 80 / 100) :=
by
  sorry

end NUMINAMATH_GPT_find_missing_values_l2_200


namespace NUMINAMATH_GPT_complementary_angles_positive_difference_l2_247

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_positive_difference_l2_247


namespace NUMINAMATH_GPT_jeff_boxes_filled_l2_296

def donuts_each_day : ℕ := 10
def days : ℕ := 12
def jeff_eats_per_day : ℕ := 1
def chris_eats : ℕ := 8
def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled : 
  (donuts_each_day * days - jeff_eats_per_day * days - chris_eats) / donuts_per_box = 10 :=
by
  sorry

end NUMINAMATH_GPT_jeff_boxes_filled_l2_296


namespace NUMINAMATH_GPT_second_player_win_strategy_l2_213

theorem second_player_win_strategy:
  ∃ strategy : (ℕ → ℕ) → ℕ, 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
    (strategy n + n = 1001) ∧
    (strategy n - n) % 13 = 0) :=
sorry

end NUMINAMATH_GPT_second_player_win_strategy_l2_213


namespace NUMINAMATH_GPT_range_of_m_l2_241

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then (1 / 3)^(-x) - 2 
  else 2 * Real.log x / Real.log 3

theorem range_of_m :
  {m : ℝ | f m > 1} = {m : ℝ | m < -Real.sqrt 3} ∪ {m : ℝ | 1 < m} :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2_241


namespace NUMINAMATH_GPT_range_of_a_l2_220

theorem range_of_a
  (a : ℝ)
  (h : ∀ (x : ℝ), 1 < x ∧ x < 4 → x^2 - 3 * x - 2 - a > 0) :
  a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2_220


namespace NUMINAMATH_GPT_first_day_exceeds_target_l2_245

-- Definitions based on the conditions
def initial_count : ℕ := 5
def daily_growth_factor : ℕ := 3
def target_count : ℕ := 200

-- The proof problem in Lean
theorem first_day_exceeds_target : ∃ n : ℕ, 5 * 3 ^ n > 200 ∧ ∀ m < n, ¬ (5 * 3 ^ m > 200) :=
by
  sorry

end NUMINAMATH_GPT_first_day_exceeds_target_l2_245


namespace NUMINAMATH_GPT_tangent_circles_distance_l2_207

-- Define the radii of the circles.
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 2

-- Define the condition that the circles are tangent.
def tangent (r1 r2 d : ℝ) : Prop :=
  d = r1 + r2 ∨ d = r1 - r2

-- State the theorem.
theorem tangent_circles_distance (d : ℝ) :
  tangent radius_O1 radius_O2 d → (d = 1 ∨ d = 5) :=
by
  sorry

end NUMINAMATH_GPT_tangent_circles_distance_l2_207


namespace NUMINAMATH_GPT_lines_do_not_form_triangle_l2_244

noncomputable def line1 (x y : ℝ) := 3 * x - y + 2 = 0
noncomputable def line2 (x y : ℝ) := 2 * x + y + 3 = 0
noncomputable def line3 (m x y : ℝ) := m * x + y = 0

theorem lines_do_not_form_triangle (m : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y) →
  (∀ x y : ℝ, (line1 x y → line3 m x y) ∨ (line2 x y → line3 m x y) ∨ 
    (line1 x y ∧ line2 x y → line3 m x y)) →
  (m = -3 ∨ m = 2 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_lines_do_not_form_triangle_l2_244


namespace NUMINAMATH_GPT_toothpaste_last_day_l2_287

theorem toothpaste_last_day (total_toothpaste : ℝ)
  (dad_use_per_brush : ℝ) (dad_brushes_per_day : ℕ)
  (mom_use_per_brush : ℝ) (mom_brushes_per_day : ℕ)
  (anne_use_per_brush : ℝ) (anne_brushes_per_day : ℕ)
  (brother_use_per_brush : ℝ) (brother_brushes_per_day : ℕ)
  (sister_use_per_brush : ℝ) (sister_brushes_per_day : ℕ)
  (grandfather_use_per_brush : ℝ) (grandfather_brushes_per_day : ℕ)
  (guest_use_per_brush : ℝ) (guest_brushes_per_day : ℕ) (guest_days : ℕ)
  (total_usage_per_day : ℝ) :
  total_toothpaste = 80 →
  dad_use_per_brush * dad_brushes_per_day = 16 →
  mom_use_per_brush * mom_brushes_per_day = 12 →
  anne_use_per_brush * anne_brushes_per_day = 8 →
  brother_use_per_brush * brother_brushes_per_day = 4 →
  sister_use_per_brush * sister_brushes_per_day = 2 →
  grandfather_use_per_brush * grandfather_brushes_per_day = 6 →
  guest_use_per_brush * guest_brushes_per_day * guest_days = 6 * 4 →
  total_usage_per_day = 54 →
  80 / 54 = 1 → 
  total_toothpaste / total_usage_per_day = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_toothpaste_last_day_l2_287


namespace NUMINAMATH_GPT_expiry_time_correct_l2_294

def factorial (n : Nat) : Nat := match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

def seconds_in_a_day : Nat := 86400
def seconds_in_an_hour : Nat := 3600
def donation_time_seconds : Nat := 8 * seconds_in_an_hour
def expiry_seconds : Nat := factorial 8

def time_of_expiry (donation_time : Nat) (expiry_time : Nat) : Nat :=
  (donation_time + expiry_time) % seconds_in_a_day

def time_to_HM (time_seconds : Nat) : Nat × Nat :=
  let hours := time_seconds / seconds_in_an_hour
  let minutes := (time_seconds % seconds_in_an_hour) / 60
  (hours, minutes)

def is_correct_expiry_time : Prop :=
  let (hours, minutes) := time_to_HM (time_of_expiry donation_time_seconds expiry_seconds)
  hours = 19 ∧ minutes = 12

theorem expiry_time_correct : is_correct_expiry_time := by
  sorry

end NUMINAMATH_GPT_expiry_time_correct_l2_294


namespace NUMINAMATH_GPT_no_valid_two_digit_N_exists_l2_285

def is_two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ (n : ℕ), n ^ 3 = x

def reverse_digits (N : ℕ) : ℕ :=
  match N / 10, N % 10 with
  | a, b => 10 * b + a

theorem no_valid_two_digit_N_exists : ∀ N : ℕ,
  is_two_digit_number N →
  (is_perfect_cube (N - reverse_digits N) ∧ (N - reverse_digits N) ≠ 27) → false :=
by sorry

end NUMINAMATH_GPT_no_valid_two_digit_N_exists_l2_285


namespace NUMINAMATH_GPT_daisy_dog_toys_l2_295

-- Given conditions
def dog_toys_monday : ℕ := 5
def dog_toys_tuesday_left : ℕ := 3
def dog_toys_tuesday_bought : ℕ := 3
def dog_toys_wednesday_all_found : ℕ := 13

-- The question we need to answer
def dog_toys_bought_wednesday : ℕ := 7

-- Statement to prove
theorem daisy_dog_toys :
  (dog_toys_monday - dog_toys_tuesday_left + dog_toys_tuesday_left + dog_toys_tuesday_bought + dog_toys_bought_wednesday = dog_toys_wednesday_all_found) :=
sorry

end NUMINAMATH_GPT_daisy_dog_toys_l2_295


namespace NUMINAMATH_GPT_scientific_notation_of_116_million_l2_236

theorem scientific_notation_of_116_million : 116000000 = 1.16 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_116_million_l2_236


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2_242

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x^2 - 2 * x < 0 → 0 < x ∧ x < 4)
  ∧ ¬(∀ (x : ℝ), 0 < x ∧ x < 4 → x^2 - 2 * x < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2_242


namespace NUMINAMATH_GPT_multiply_expand_l2_297

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end NUMINAMATH_GPT_multiply_expand_l2_297


namespace NUMINAMATH_GPT_joey_more_fish_than_peter_l2_205

-- Define the conditions
variables (A P J : ℕ)

-- Condition that Ali's fish weight is twice that of Peter's
def ali_double_peter (A P : ℕ) : Prop := A = 2 * P

-- Condition that Ali caught 12 kg of fish
def ali_caught_12 (A : ℕ) : Prop := A = 12

-- Condition that the total weight of the fish is 25 kg
def total_weight (A P J : ℕ) : Prop := A + P + J = 25

-- Prove that Joey caught 1 kg more fish than Peter
theorem joey_more_fish_than_peter (A P J : ℕ) :
  ali_double_peter A P → ali_caught_12 A → total_weight A P J → J = 1 :=
by 
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_joey_more_fish_than_peter_l2_205


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l2_288

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h0 : q ≠ 1) 
  (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q)) 
  (h2 : ∀ n, a n = a 0 * q^n) 
  (h3 : 2 * S 3 = 7 * a 2) :
  (S 5 / a 2 = 31 / 2) ∨ (S 5 / a 2 = 31 / 8) :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l2_288


namespace NUMINAMATH_GPT_smallest_second_term_l2_253

theorem smallest_second_term (a d : ℕ) (h1 : 5 * a + 10 * d = 95) (h2 : a > 0) (h3 : d > 0) : 
  a + d = 10 :=
sorry

end NUMINAMATH_GPT_smallest_second_term_l2_253


namespace NUMINAMATH_GPT_circle_radius_l2_279

/-
  Given:
  - The area of the circle x = π r^2
  - The circumference of the circle y = 2π r
  - The sum x + y = 72π

  Prove:
  The radius r = 6
-/
theorem circle_radius (r : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : x = π * r ^ 2) 
  (h₂ : y = 2 * π * r) 
  (h₃ : x + y = 72 * π) : 
  r = 6 := 
sorry

end NUMINAMATH_GPT_circle_radius_l2_279


namespace NUMINAMATH_GPT_vehicles_travelled_last_year_l2_249

theorem vehicles_travelled_last_year (V : ℕ) : 
  (∀ (x : ℕ), (96 : ℕ) * (V / 100000000) = 2880) → V = 3000000000 := 
by 
  sorry

end NUMINAMATH_GPT_vehicles_travelled_last_year_l2_249


namespace NUMINAMATH_GPT_income_of_m_l2_224

theorem income_of_m (M N O : ℝ)
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (M + O) / 2 = 5200) :
  M = 4000 :=
by
  -- sorry is used to skip the actual proof.
  sorry

end NUMINAMATH_GPT_income_of_m_l2_224


namespace NUMINAMATH_GPT_find_nonnegative_solutions_l2_268

theorem find_nonnegative_solutions :
  ∀ (x y z : ℕ), 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by
  sorry

end NUMINAMATH_GPT_find_nonnegative_solutions_l2_268


namespace NUMINAMATH_GPT_total_bill_calculation_l2_278

theorem total_bill_calculation (n : ℕ) (amount_per_person : ℝ) (total_amount : ℝ) :
  n = 9 → amount_per_person = 514.19 → total_amount = 4627.71 → 
  n * amount_per_person = total_amount :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_total_bill_calculation_l2_278


namespace NUMINAMATH_GPT_boys_at_reunion_l2_271

theorem boys_at_reunion (n : ℕ) (h : n * (n - 1) = 56) : n = 8 :=
sorry

end NUMINAMATH_GPT_boys_at_reunion_l2_271


namespace NUMINAMATH_GPT_circumscribed_quadrilateral_converse_arithmetic_progression_l2_261

theorem circumscribed_quadrilateral (a b c d : ℝ) (k : ℝ) (h1 : b = a + k) (h2 : d = a + 2 * k) (h3 : c = a + 3 * k) :
  a + c = b + d :=
by
  sorry

theorem converse_arithmetic_progression (a b c d : ℝ) (h : a + c = b + d) :
  ∃ k : ℝ, b = a + k ∧ d = a + 2 * k ∧ c = a + 3 * k :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_quadrilateral_converse_arithmetic_progression_l2_261


namespace NUMINAMATH_GPT_population_total_l2_254

theorem population_total (total_population layers : ℕ) (ratio_A ratio_B ratio_C : ℕ) 
(sample_capacity : ℕ) (prob_ab_in_C : ℚ) 
(h1 : ratio_A = 3)
(h2 : ratio_B = 6)
(h3 : ratio_C = 1)
(h4 : sample_capacity = 20)
(h5 : prob_ab_in_C = 1 / 21)
(h6 : total_population = 10 * ratio_C) :
  total_population = 70 := 
by 
  sorry

end NUMINAMATH_GPT_population_total_l2_254


namespace NUMINAMATH_GPT_salmon_total_l2_238

def num_male : ℕ := 712261
def num_female : ℕ := 259378
def num_total : ℕ := 971639

theorem salmon_total :
  num_male + num_female = num_total :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_salmon_total_l2_238


namespace NUMINAMATH_GPT_shadow_length_correct_l2_211

theorem shadow_length_correct :
  let light_source := (0, 16)
  let disc_center := (6, 10)
  let radius := 2
  let m := 4
  let n := 17
  let length_form := m * Real.sqrt n
  length_form = 4 * Real.sqrt 17 :=
by
  sorry

end NUMINAMATH_GPT_shadow_length_correct_l2_211


namespace NUMINAMATH_GPT_simplify_fractional_expression_l2_269

variable {a b c : ℝ}

theorem simplify_fractional_expression 
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0)
  (h_sum : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 
  3 / (2 * (-b - c + b * c)) :=
sorry

end NUMINAMATH_GPT_simplify_fractional_expression_l2_269


namespace NUMINAMATH_GPT_largest_number_is_A_l2_260

-- Definitions of the numbers
def numA := 8.45678
def numB := 8.456777777 -- This should be represented properly with an infinite sequence in a real formal proof
def numC := 8.456767676 -- This should be represented properly with an infinite sequence in a real formal proof
def numD := 8.456756756 -- This should be represented properly with an infinite sequence in a real formal proof
def numE := 8.456745674 -- This should be represented properly with an infinite sequence in a real formal proof

-- Lean statement to prove that numA is the largest number
theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE :=
by
  -- Proof not provided, sorry to skip
  sorry

end NUMINAMATH_GPT_largest_number_is_A_l2_260


namespace NUMINAMATH_GPT_bob_age_sum_digits_l2_232

theorem bob_age_sum_digits
  (A B C : ℕ)  -- Define ages for Alice (A), Bob (B), and Carl (C)
  (h1 : C = 2)  -- Carl's age is 2
  (h2 : B = A + 2)  -- Bob is 2 years older than Alice
  (h3 : ∃ n, A = 2 * n ∧ n > 0 ∧ n ≤ 8 )  -- Alice's age is a multiple of Carl's age today, marking the second of the 8 such multiples 
  : ∃ n, (B + n) % (C + n) = 0 ∧ (B + n) = 50 :=  -- Prove that the next time Bob's age is a multiple of Carl's, Bob's age will be 50
sorry

end NUMINAMATH_GPT_bob_age_sum_digits_l2_232


namespace NUMINAMATH_GPT_average_marks_correct_l2_274

-- Definitions used in the Lean 4 statement, reflecting conditions in the problem
def total_students_class1 : ℕ := 25 
def average_marks_class1 : ℕ := 40 
def total_students_class2 : ℕ := 30 
def average_marks_class2 : ℕ := 60 

-- Calculate the total marks for both classes
def total_marks_class1 : ℕ := total_students_class1 * average_marks_class1 
def total_marks_class2 : ℕ := total_students_class2 * average_marks_class2 
def total_marks : ℕ := total_marks_class1 + total_marks_class2 

-- Calculate the total number of students
def total_students : ℕ := total_students_class1 + total_students_class2 

-- Define the average of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_students 

-- The theorem to be proved
theorem average_marks_correct : average_marks_all_students = (2800 : ℚ) / 55 := 
by 
  sorry

end NUMINAMATH_GPT_average_marks_correct_l2_274


namespace NUMINAMATH_GPT_average_weight_l2_270

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_l2_270


namespace NUMINAMATH_GPT_find_radius_l2_231

def setA : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def setB (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem find_radius (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ setA ∧ p ∈ setB r) ↔ (r = 3 ∨ r = 7) :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l2_231


namespace NUMINAMATH_GPT_correct_operation_l2_234

theorem correct_operation :
  (∀ a : ℝ, (a^5 * a^3 = a^15) = false) ∧
  (∀ a : ℝ, (a^5 - a^3 = a^2) = false) ∧
  (∀ a : ℝ, ((-a^5)^2 = a^10) = true) ∧
  (∀ a : ℝ, (a^6 / a^3 = a^2) = false) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2_234


namespace NUMINAMATH_GPT_g_x_squared_plus_2_l2_272

namespace PolynomialProof

open Polynomial

noncomputable def g (x : ℚ) : ℚ := sorry

theorem g_x_squared_plus_2 (x : ℚ) (h : g (x^2 - 2) = x^4 - 6*x^2 + 8) :
  g (x^2 + 2) = x^4 + 2*x^2 + 2 :=
sorry

end PolynomialProof

end NUMINAMATH_GPT_g_x_squared_plus_2_l2_272


namespace NUMINAMATH_GPT_yanna_kept_apples_l2_257

-- Define the given conditions
def initial_apples : ℕ := 60
def percentage_given_to_zenny : ℝ := 0.40
def percentage_given_to_andrea : ℝ := 0.25

-- Prove the main statement
theorem yanna_kept_apples : 
  let apples_given_to_zenny := (percentage_given_to_zenny * initial_apples)
  let apples_remaining_after_zenny := (initial_apples - apples_given_to_zenny)
  let apples_given_to_andrea := (percentage_given_to_andrea * apples_remaining_after_zenny)
  let apples_kept := (apples_remaining_after_zenny - apples_given_to_andrea)
  apples_kept = 27 :=
by
  sorry

end NUMINAMATH_GPT_yanna_kept_apples_l2_257


namespace NUMINAMATH_GPT_average_infection_rate_l2_286

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_average_infection_rate_l2_286


namespace NUMINAMATH_GPT_train_speed_kmh_l2_217

def man_speed_kmh : ℝ := 3 -- The man's speed in km/h
def train_length_m : ℝ := 110 -- The train's length in meters
def passing_time_s : ℝ := 12 -- Time taken to pass the man in seconds

noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600 -- Convert man's speed to m/s

theorem train_speed_kmh :
  (110 / 12) - (5 / 6) * (3600 / 1000) = 30 := by
  -- Omitted steps will go here
  sorry

end NUMINAMATH_GPT_train_speed_kmh_l2_217


namespace NUMINAMATH_GPT_eval_polynomial_at_3_l2_252

def f (x : ℝ) : ℝ := 2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

theorem eval_polynomial_at_3 : f 3 = 130 :=
by
  -- proof can be completed here following proper steps or using Horner's method
  sorry

end NUMINAMATH_GPT_eval_polynomial_at_3_l2_252


namespace NUMINAMATH_GPT_area_OPA_l2_219

variable (x : ℝ)

def y (x : ℝ) : ℝ := -x + 6

def A : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (0, 0)
def P (x : ℝ) : ℝ × ℝ := (x, y x)

def area_triangle (O A P : ℝ × ℝ) : ℝ := 
  0.5 * abs (A.fst * P.snd + P.fst * O.snd + O.fst * A.snd - A.snd * P.fst - P.snd * O.fst - O.snd * A.fst)

theorem area_OPA : 0 < x ∧ x < 6 → area_triangle O A (P x) = 12 - 2 * x := by
  -- proof to be provided here
  sorry


end NUMINAMATH_GPT_area_OPA_l2_219


namespace NUMINAMATH_GPT_digits_to_replace_l2_235

theorem digits_to_replace (a b c d e f : ℕ) :
  (a = 1) →
  (b < 5) →
  (c = 8) →
  (d = 1) →
  (e = 0) →
  (f = 4) →
  (100 * a + 10 * b + c)^2 = 10000 * d + 1000 * e + 100 * f + 10 * f + f :=
  by
    intros ha hb hc hd he hf 
    sorry

end NUMINAMATH_GPT_digits_to_replace_l2_235


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l2_262

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_of_M_and_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ 2} :=
by sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l2_262


namespace NUMINAMATH_GPT_minimum_value_l2_282

open Real

theorem minimum_value (a : ℝ) (m n : ℝ) (h_a : a > 0) (h_a_not_one : a ≠ 1) 
                      (h_mn : m * n > 0) (h_point : -m - n + 1 = 0) :
  (1 / m + 2 / n) = 3 + 2 * sqrt 2 :=
by
  -- proof should go here
  sorry

end NUMINAMATH_GPT_minimum_value_l2_282


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l2_281

open Real

noncomputable def triangle_area (b c : ℝ) : ℝ :=
  (sqrt 2 / 4) * (sqrt (4 + b^2)) * (sqrt (4 + c^2))

theorem area_of_triangle_ABC (b c : ℝ) :
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  let A : ℝ × ℝ × ℝ := (2, 0, 0)
  let B : ℝ × ℝ × ℝ := (0, b, 0)
  let C : ℝ × ℝ × ℝ := (0, 0, c)
  let angle_BAC : ℝ := 45
  (cos (angle_BAC * π / 180) = sqrt 2 / 2) →
  (sin (angle_BAC * π / 180) = sqrt 2 / 2) →
  let AB := sqrt (2^2 + b^2)
  let AC := sqrt (2^2 + c^2)
  let area := (1/2) * AB * AC * (sin (45 * π / 180))
  area = triangle_area b c :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l2_281


namespace NUMINAMATH_GPT_total_coughs_after_20_minutes_l2_277

def coughs_in_n_minutes (rate_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  rate_per_minute * minutes

def total_coughs (georgia_rate_per_minute : ℕ) (minutes : ℕ) (multiplier : ℕ) : ℕ :=
  let georgia_coughs := coughs_in_n_minutes georgia_rate_per_minute minutes
  let robert_rate_per_minute := georgia_rate_per_minute * multiplier
  let robert_coughs := coughs_in_n_minutes robert_rate_per_minute minutes
  georgia_coughs + robert_coughs

theorem total_coughs_after_20_minutes :
  total_coughs 5 20 2 = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_coughs_after_20_minutes_l2_277


namespace NUMINAMATH_GPT_factor_expression_l2_222

theorem factor_expression (x : ℕ) : 63 * x + 54 = 9 * (7 * x + 6) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2_222


namespace NUMINAMATH_GPT_range_of_a_l2_239

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1) ∧ (5 * x > 3 * x + 2 * a) ↔ (x > 3)) ↔ (a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2_239


namespace NUMINAMATH_GPT_limping_rook_adjacent_sum_not_divisible_by_4_l2_289

/-- Problem statement: A limping rook traversed a 10 × 10 board,
visiting each square exactly once with numbers 1 through 100
written in the order visited.
Prove that the sum of the numbers in any two adjacent cells
is not divisible by 4. -/
theorem limping_rook_adjacent_sum_not_divisible_by_4 :
  ∀ (board : Fin 10 → Fin 10 → ℕ), 
  (∀ (i j : Fin 10), 1 ≤ board i j ∧ board i j ≤ 100) →
  (∀ (i j : Fin 10), (∃ (i' : Fin 10), i = i' + 1 ∨ i = i' - 1)
                 ∨ (∃ (j' : Fin 10), j = j' + 1 ∨ j = j' - 1)) →
  ((∀ (i j : Fin 10) (k l : Fin 10),
      (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      (board i j + board k l) % 4 ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_limping_rook_adjacent_sum_not_divisible_by_4_l2_289


namespace NUMINAMATH_GPT_average_first_21_multiples_of_6_l2_263

-- Define the arithmetic sequence and its conditions.
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

-- Define the problem statement.
theorem average_first_21_multiples_of_6 :
  let a1 := 6
  let d := 6
  let n := 21
  let an := arithmetic_sequence a1 d n
  (a1 + an) / 2 = 66 := by
  sorry

end NUMINAMATH_GPT_average_first_21_multiples_of_6_l2_263


namespace NUMINAMATH_GPT_anna_current_age_l2_216

theorem anna_current_age (A : ℕ) (Clara_now : ℕ) (years_ago : ℕ) (Clara_age_ago : ℕ) 
    (H1 : Clara_now = 80) 
    (H2 : years_ago = 41) 
    (H3 : Clara_age_ago = Clara_now - years_ago) 
    (H4 : Clara_age_ago = 3 * (A - years_ago)) : 
    A = 54 :=
by
  sorry

end NUMINAMATH_GPT_anna_current_age_l2_216


namespace NUMINAMATH_GPT_sum_of_inverses_inequality_l2_283

theorem sum_of_inverses_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum_eq : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end NUMINAMATH_GPT_sum_of_inverses_inequality_l2_283


namespace NUMINAMATH_GPT_solve_equation_l2_299

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2_299


namespace NUMINAMATH_GPT_inequality_amgm_l2_292

variable {a b c : ℝ}

theorem inequality_amgm (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) : 
  (1 / 2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) <= a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) ∧ 
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) <= (a - b)^2 + (b - c)^2 + (c - a)^2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_amgm_l2_292


namespace NUMINAMATH_GPT_tallest_building_model_height_l2_227

def height_campus : ℝ := 120
def volume_campus : ℝ := 30000
def volume_model : ℝ := 0.03
def height_model : ℝ := 1.2

theorem tallest_building_model_height :
  (volume_campus / volume_model)^(1/3) = (height_campus / height_model) :=
by
  sorry

end NUMINAMATH_GPT_tallest_building_model_height_l2_227


namespace NUMINAMATH_GPT_trigonometric_identity_l2_290

theorem trigonometric_identity 
  (deg7 deg37 deg83 : ℝ)
  (h7 : deg7 = 7) 
  (h37 : deg37 = 37) 
  (h83 : deg83 = 83) 
  : (Real.sin (deg7 * Real.pi / 180) * Real.cos (deg37 * Real.pi / 180) - Real.sin (deg83 * Real.pi / 180) * Real.sin (deg37 * Real.pi / 180) = -1/2) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l2_290


namespace NUMINAMATH_GPT_circumcircle_radius_of_right_triangle_l2_258

theorem circumcircle_radius_of_right_triangle (r : ℝ) (BC : ℝ) (R : ℝ) 
  (h1 : r = 3) (h2 : BC = 10) : R = 7.25 := 
by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_of_right_triangle_l2_258


namespace NUMINAMATH_GPT_crayon_boxes_needed_l2_212

theorem crayon_boxes_needed (total_crayons : ℕ) (crayons_per_box : ℕ) (h1 : total_crayons = 80) (h2 : crayons_per_box = 8) : (total_crayons / crayons_per_box) = 10 :=
by
  sorry

end NUMINAMATH_GPT_crayon_boxes_needed_l2_212


namespace NUMINAMATH_GPT_modified_pyramid_volume_l2_214

theorem modified_pyramid_volume (s h : ℝ) (V : ℝ) 
  (hV : V = 1/3 * s^2 * h) (hV_eq : V = 72) :
  (1/3) * (3 * s)^2 * (2 * h) = 1296 := by
  sorry

end NUMINAMATH_GPT_modified_pyramid_volume_l2_214


namespace NUMINAMATH_GPT_calculation_correct_l2_276

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l2_276


namespace NUMINAMATH_GPT_scott_runs_84_miles_in_a_month_l2_233

-- Define the number of miles Scott runs from Monday to Wednesday in a week.
def milesMonToWed : ℕ := 3 * 3

-- Define the number of miles Scott runs on Thursday and Friday in a week.
def milesThuFri : ℕ := 3 * 2 * 2

-- Define the total number of miles Scott runs in a week.
def totalMilesPerWeek : ℕ := milesMonToWed + milesThuFri

-- Define the number of weeks in a month.
def weeksInMonth : ℕ := 4

-- Define the total number of miles Scott runs in a month.
def totalMilesInMonth : ℕ := totalMilesPerWeek * weeksInMonth

-- Statement to prove that Scott runs 84 miles in a month with 4 weeks.
theorem scott_runs_84_miles_in_a_month : totalMilesInMonth = 84 := by
  -- The proof is omitted for this example.
  sorry

end NUMINAMATH_GPT_scott_runs_84_miles_in_a_month_l2_233


namespace NUMINAMATH_GPT_songs_today_is_14_l2_209

-- Define the number of songs Jeremy listened to yesterday
def songs_yesterday (x : ℕ) : ℕ := x

-- Define the number of songs Jeremy listened to today
def songs_today (x : ℕ) : ℕ := x + 5

-- Given conditions
def total_songs (x : ℕ) : Prop := songs_yesterday x + songs_today x = 23

-- Prove the number of songs Jeremy listened to today
theorem songs_today_is_14 : ∃ x: ℕ, total_songs x ∧ songs_today x = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_songs_today_is_14_l2_209


namespace NUMINAMATH_GPT_money_left_eq_l2_225

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end NUMINAMATH_GPT_money_left_eq_l2_225


namespace NUMINAMATH_GPT_cos_beta_zero_l2_267

theorem cos_beta_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : Real.cos α = 1 / 2) (h4 : Real.cos (α + β) = -1 / 2) : Real.cos β = 0 :=
sorry

end NUMINAMATH_GPT_cos_beta_zero_l2_267


namespace NUMINAMATH_GPT_correct_choice_2point5_l2_255

def set_M : Set ℝ := {x | -2 < x ∧ x < 3}

theorem correct_choice_2point5 : 2.5 ∈ set_M :=
by {
  -- sorry is added to close the proof for now
  sorry
}

end NUMINAMATH_GPT_correct_choice_2point5_l2_255


namespace NUMINAMATH_GPT_greatest_drop_in_price_is_august_l2_251

-- Define the months and their respective price changes
def price_changes : List (String × ℝ) :=
  [("January", -1.00), ("February", 1.50), ("March", -3.00), ("April", 2.50), 
   ("May", -0.75), ("June", -2.25), ("July", 1.00), ("August", -4.00)]

-- Define the statement that August has the greatest drop in price
theorem greatest_drop_in_price_is_august :
  ∀ month ∈ price_changes, month.snd ≤ -4.00 → month.fst = "August" :=
by
  sorry

end NUMINAMATH_GPT_greatest_drop_in_price_is_august_l2_251


namespace NUMINAMATH_GPT_distinct_units_digits_of_squares_mod_6_l2_240

theorem distinct_units_digits_of_squares_mod_6 : 
  ∃ (s : Finset ℕ), s = {0, 1, 4, 3} ∧ s.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_distinct_units_digits_of_squares_mod_6_l2_240


namespace NUMINAMATH_GPT_point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l2_256

-- Question (1): Proving that the point (-2,0) lies on the graph
theorem point_on_graph (k : ℝ) (hk : k ≠ 0) : k * (-2 + 2) = 0 := 
by sorry

-- Question (2): Finding the value of k given a shifted graph passing through a point
theorem find_k_shifted_graph_passing (k : ℝ) : (k * (1 + 2) + 2 = -2) → k = -4/3 := 
by sorry

-- Question (3): Proving the range of k for the function's y-intercept within given limits
theorem y_axis_intercept_range (k : ℝ) (hk : -2 < 2 * k ∧ 2 * k < 0) : -1 < k ∧ k < 0 := 
by sorry

end NUMINAMATH_GPT_point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l2_256


namespace NUMINAMATH_GPT_answer_keys_count_l2_266

theorem answer_keys_count 
  (test_questions : ℕ)
  (true_answers : ℕ)
  (false_answers : ℕ)
  (min_score : ℕ)
  (conditions : test_questions = 10 ∧ true_answers = 5 ∧ false_answers = 5 ∧ min_score >= 4) :
  ∃ (count : ℕ), count = 22 := by
  sorry

end NUMINAMATH_GPT_answer_keys_count_l2_266


namespace NUMINAMATH_GPT_parrots_are_red_l2_237

-- Definitions for fractions.
def total_parrots : ℕ := 160
def green_fraction : ℚ := 5 / 8
def blue_fraction : ℚ := 1 / 4

-- Definition for calculating the number of parrots.
def number_of_green_parrots : ℚ := green_fraction * total_parrots
def number_of_blue_parrots : ℚ := blue_fraction * total_parrots
def number_of_red_parrots : ℚ := total_parrots - number_of_green_parrots - number_of_blue_parrots

-- The theorem to prove.
theorem parrots_are_red : number_of_red_parrots = 20 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_parrots_are_red_l2_237


namespace NUMINAMATH_GPT_quadratic_roots_condition_l2_259

theorem quadratic_roots_condition (k : ℝ) : 
  (∀ (r s : ℝ), r + s = -k ∧ r * s = 12 → (r + 3) + (s + 3) = k) → k = 3 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l2_259


namespace NUMINAMATH_GPT_max_sum_x_y_l2_210

theorem max_sum_x_y (x y : ℝ) (h : (2015 + x^2) * (2015 + y^2) = 2 ^ 22) : 
  x + y ≤ 2 * Real.sqrt 33 :=
sorry

end NUMINAMATH_GPT_max_sum_x_y_l2_210


namespace NUMINAMATH_GPT_star_7_2_l2_273

def star (a b : ℕ) := 4 * a - 4 * b

theorem star_7_2 : star 7 2 = 20 := 
by
  sorry

end NUMINAMATH_GPT_star_7_2_l2_273


namespace NUMINAMATH_GPT_compare_abc_l2_218

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_compare_abc_l2_218


namespace NUMINAMATH_GPT_towers_remainder_l2_203

noncomputable def count_towers (k : ℕ) : ℕ := sorry

theorem towers_remainder : (count_towers 9) % 1000 = 768 := sorry

end NUMINAMATH_GPT_towers_remainder_l2_203


namespace NUMINAMATH_GPT_inequality_am_gm_l2_298

theorem inequality_am_gm (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l2_298


namespace NUMINAMATH_GPT_determine_a_l2_226

theorem determine_a (a b c : ℕ) (h_b : b = 5) (h_c : c = 6) (h_order : c > b ∧ b > a ∧ a > 2) :
(a - 2) * (b - 2) * (c - 2) = 4 * (b - 2) + 4 * (c - 2) → a = 4 :=
by 
  sorry

end NUMINAMATH_GPT_determine_a_l2_226


namespace NUMINAMATH_GPT_other_number_is_29_l2_202

theorem other_number_is_29
    (k : ℕ)
    (some_number : ℕ)
    (h1 : k = 2)
    (h2 : (5 + k) * (5 - k) = some_number - 2^3) :
    some_number = 29 :=
by
  sorry

end NUMINAMATH_GPT_other_number_is_29_l2_202


namespace NUMINAMATH_GPT_tax_free_amount_l2_280

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) 
(h1 : total_value = 1720) 
(h2 : tax_paid = 134.4) 
(h3 : tax_rate = 0.12) 
(h4 : tax_paid = tax_rate * (total_value - X)) 
: X = 600 := 
sorry

end NUMINAMATH_GPT_tax_free_amount_l2_280


namespace NUMINAMATH_GPT_find_a_l2_243

theorem find_a (x a : ℝ) (h₁ : x = 2) (h₂ : (4 - x) / 2 + a = 4) : a = 3 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_find_a_l2_243


namespace NUMINAMATH_GPT_bottom_row_bricks_l2_284

theorem bottom_row_bricks (n : ℕ) 
  (h1 : (n + (n-1) + (n-2) + (n-3) + (n-4) = 200)) : 
  n = 42 := 
by sorry

end NUMINAMATH_GPT_bottom_row_bricks_l2_284


namespace NUMINAMATH_GPT_arccos_sqrt_half_l2_265

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end NUMINAMATH_GPT_arccos_sqrt_half_l2_265
