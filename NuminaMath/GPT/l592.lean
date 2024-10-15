import Mathlib

namespace NUMINAMATH_GPT_inequality_for_average_daily_work_l592_59212

-- Given
def total_earthwork : ℕ := 300
def completed_earthwork_first_day : ℕ := 60
def scheduled_days : ℕ := 6
def days_ahead : ℕ := 2

-- To Prove
theorem inequality_for_average_daily_work (x : ℕ) :
  scheduled_days - days_ahead - 1 > 0 →
  (total_earthwork - completed_earthwork_first_day) ≤ x * (scheduled_days - days_ahead - 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_average_daily_work_l592_59212


namespace NUMINAMATH_GPT_sasha_added_num_l592_59248

theorem sasha_added_num (a b c : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a / b = 5 * (a + c) / (b * c)) : c = 6 ∨ c = -20 := 
sorry

end NUMINAMATH_GPT_sasha_added_num_l592_59248


namespace NUMINAMATH_GPT_remainder_is_correct_l592_59210

def P (x : ℝ) : ℝ := x^6 + 2 * x^5 - 3 * x^4 + x^2 - 8
def D (x : ℝ) : ℝ := x^2 - 1

theorem remainder_is_correct : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, P x = D x * q x + (2.5 * x - 9.5) :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_correct_l592_59210


namespace NUMINAMATH_GPT_quiz_true_false_questions_l592_59232

theorem quiz_true_false_questions (n : ℕ) 
  (h1 : 2^n - 2 ≠ 0) 
  (h2 : (2^n - 2) * 16 = 224) : 
  n = 4 := 
sorry

end NUMINAMATH_GPT_quiz_true_false_questions_l592_59232


namespace NUMINAMATH_GPT_stratified_sampling_total_results_l592_59252

theorem stratified_sampling_total_results :
  let junior_students := 400
  let senior_students := 200
  let total_students_to_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (Nat.choose junior_students junior_sample) * (Nat.choose senior_students senior_sample) = Nat.choose 400 40 * Nat.choose 200 20 :=
  sorry

end NUMINAMATH_GPT_stratified_sampling_total_results_l592_59252


namespace NUMINAMATH_GPT_sum_of_numbers_l592_59289

theorem sum_of_numbers (x y : ℕ) (hx : 100 ≤ x ∧ x < 1000) (hy : 1000 ≤ y ∧ y < 10000) (h : 10000 * x + y = 12 * x * y) :
  x + y = 1083 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l592_59289


namespace NUMINAMATH_GPT_sum_a5_a8_l592_59267

variable (a : ℕ → ℝ)
variable (r : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_a5_a8 (a1 a2 a3 a4 : ℝ) (q : ℝ)
  (h1 : a1 + a3 = 1)
  (h2 : a2 + a4 = 2)
  (h_seq : is_geometric_sequence a q)
  (a_def : ∀ n : ℕ, a n = a1 * q^n) :
  a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end NUMINAMATH_GPT_sum_a5_a8_l592_59267


namespace NUMINAMATH_GPT_find_LCM_of_numbers_l592_59290

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end NUMINAMATH_GPT_find_LCM_of_numbers_l592_59290


namespace NUMINAMATH_GPT_fraction_of_men_collected_dues_l592_59205

theorem fraction_of_men_collected_dues
  (M W : ℕ)
  (x : ℚ)
  (h1 : 45 * x * M + 5 * W = 17760)
  (h2 : M + W = 3552)
  (h3 : 1 / 12 * W = W / 12) :
  x = 1 / 9 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_fraction_of_men_collected_dues_l592_59205


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_iff_a_in_interval_l592_59283

theorem inequality_holds_for_all_x_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_iff_a_in_interval_l592_59283


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l592_59237

variable (x : ℝ)

def condition1 : Prop := x > 2
def condition2 : Prop := x^2 > 4

theorem sufficient_but_not_necessary :
  (condition1 x → condition2 x) ∧ (¬ (condition2 x → condition1 x)) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l592_59237


namespace NUMINAMATH_GPT_total_cost_pencils_and_pens_l592_59241

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end NUMINAMATH_GPT_total_cost_pencils_and_pens_l592_59241


namespace NUMINAMATH_GPT_minimize_on_interval_l592_59280

def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 2

theorem minimize_on_interval (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x a ≥ if a < 0 then -2 else if 0 ≤ a ∧ a ≤ 2 then -a^2 - 2 else 2 - 4*a) :=
by 
  sorry

end NUMINAMATH_GPT_minimize_on_interval_l592_59280


namespace NUMINAMATH_GPT_function_domain_exclusion_l592_59245

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end NUMINAMATH_GPT_function_domain_exclusion_l592_59245


namespace NUMINAMATH_GPT_x_minus_y_eq_neg_200_l592_59271

theorem x_minus_y_eq_neg_200 (x y : ℤ) (h1 : x + y = 290) (h2 : y = 245) : x - y = -200 := by
  sorry

end NUMINAMATH_GPT_x_minus_y_eq_neg_200_l592_59271


namespace NUMINAMATH_GPT_N_cannot_be_sum_of_three_squares_l592_59204

theorem N_cannot_be_sum_of_three_squares (K : ℕ) (L : ℕ) (N : ℕ) (h1 : N = 4^K * L) (h2 : L % 8 = 7) : ¬ ∃ (a b c : ℕ), N = a^2 + b^2 + c^2 := 
sorry

end NUMINAMATH_GPT_N_cannot_be_sum_of_three_squares_l592_59204


namespace NUMINAMATH_GPT_factor_expression_l592_59244

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l592_59244


namespace NUMINAMATH_GPT_problem1_problem2_l592_59254

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l592_59254


namespace NUMINAMATH_GPT_sheets_per_class_per_day_l592_59286

theorem sheets_per_class_per_day
  (weekly_sheets : ℕ)
  (school_days_per_week : ℕ)
  (num_classes : ℕ)
  (h1 : weekly_sheets = 9000)
  (h2 : school_days_per_week = 5)
  (h3 : num_classes = 9) :
  (weekly_sheets / school_days_per_week) / num_classes = 200 :=
by
  sorry

end NUMINAMATH_GPT_sheets_per_class_per_day_l592_59286


namespace NUMINAMATH_GPT_minions_mistake_score_l592_59225

theorem minions_mistake_score :
  (minions_left_phone_on_untrusted_website ∧
   downloaded_file_from_untrusted_source ∧
   guidelines_by_cellular_operators ∧
   avoid_sharing_personal_info ∧
   unverified_files_may_be_harmful ∧
   double_extensions_signify_malicious_software) →
  score = 21 :=
by
  -- Here we would provide the proof steps which we skip with sorry
  sorry

end NUMINAMATH_GPT_minions_mistake_score_l592_59225


namespace NUMINAMATH_GPT_train_passes_man_in_correct_time_l592_59291

-- Definitions for the given conditions
def platform_length : ℝ := 270
def train_length : ℝ := 180
def crossing_time : ℝ := 20

-- Theorem to prove the time taken to pass the man is 8 seconds
theorem train_passes_man_in_correct_time
  (p: ℝ) (l: ℝ) (t_cross: ℝ)
  (h1: p = platform_length)
  (h2: l = train_length)
  (h3: t_cross = crossing_time) :
  l / ((l + p) / t_cross) = 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_passes_man_in_correct_time_l592_59291


namespace NUMINAMATH_GPT_find_f_1_minus_a_l592_59270

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

theorem find_f_1_minus_a 
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_period : periodic_function f 2)
  (h_value : ∃ a : ℝ, f (1 + a) = 1) :
  ∃ a : ℝ, f (1 - a) = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_1_minus_a_l592_59270


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l592_59239

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, x^2 / 16 - y^2 / 9 = -1 → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l592_59239


namespace NUMINAMATH_GPT_min_a_condition_l592_59281

-- Definitions of the conditions
def real_numbers (x : ℝ) := true

def in_interval (a m n : ℝ) : Prop := 0 < n ∧ n < m ∧ m < 1 / a

def inequality (a m n : ℝ) : Prop :=
  (n^(1/m) / m^(1/n) > (n^a) / (m^a))

-- Lean statement
theorem min_a_condition (a m n : ℝ) (h1 : real_numbers m) (h2 : real_numbers n)
    (h3 : in_interval a m n) : inequality a m n ↔ 1 ≤ a :=
sorry

end NUMINAMATH_GPT_min_a_condition_l592_59281


namespace NUMINAMATH_GPT_side_length_sum_area_l592_59202

theorem side_length_sum_area (a b c d : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 12) :
  d = 13 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_side_length_sum_area_l592_59202


namespace NUMINAMATH_GPT_fill_trough_time_l592_59287

noncomputable def time_to_fill (T_old T_new T_third : ℕ) : ℝ :=
  let rate_old := (1 : ℝ) / T_old
  let rate_new := (1 : ℝ) / T_new
  let rate_third := (1 : ℝ) / T_third
  let total_rate := rate_old + rate_new + rate_third
  1 / total_rate

theorem fill_trough_time:
  time_to_fill 600 200 400 = 1200 / 11 := 
by
  sorry

end NUMINAMATH_GPT_fill_trough_time_l592_59287


namespace NUMINAMATH_GPT_problem_part_I_problem_part_II_l592_59292

-- Problem (I)
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (h1 : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : 
  a + b = 2 * c -> (a + b) = 2 * c :=
by
  intros h
  sorry

-- Problem (II)
theorem problem_part_II (a b c : ℝ) (A B C : ℝ) 
  (h1 : C = Real.pi / 3) 
  (h2 : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) 
  (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
  (h4 : a + b = 2 * c) : c = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_problem_part_I_problem_part_II_l592_59292


namespace NUMINAMATH_GPT_find_number_l592_59272

theorem find_number (x : ℝ) (h : (1/2) * x + 7 = 17) : x = 20 :=
sorry

end NUMINAMATH_GPT_find_number_l592_59272


namespace NUMINAMATH_GPT_calculate_decimal_l592_59253

theorem calculate_decimal : 3.59 + 2.4 - 1.67 = 4.32 := 
  by
  sorry

end NUMINAMATH_GPT_calculate_decimal_l592_59253


namespace NUMINAMATH_GPT_passing_probability_l592_59264

def probability_of_passing (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

theorem passing_probability :
  probability_of_passing 0.6 = 0.504 :=
by {
  sorry
}

end NUMINAMATH_GPT_passing_probability_l592_59264


namespace NUMINAMATH_GPT_exists_x_quadratic_eq_zero_iff_le_one_l592_59236

variable (a : ℝ)

theorem exists_x_quadratic_eq_zero_iff_le_one : (∃ x : ℝ, x^2 - 2 * x + a = 0) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_exists_x_quadratic_eq_zero_iff_le_one_l592_59236


namespace NUMINAMATH_GPT_fruit_seller_apples_l592_59203

theorem fruit_seller_apples (original_apples : ℝ) (sold_percent : ℝ) (remaining_apples : ℝ)
  (h1 : sold_percent = 0.40)
  (h2 : remaining_apples = 420)
  (h3 : original_apples * (1 - sold_percent) = remaining_apples) :
  original_apples = 700 :=
by
  sorry

end NUMINAMATH_GPT_fruit_seller_apples_l592_59203


namespace NUMINAMATH_GPT_total_players_l592_59279

-- Definitions of the given conditions
def K : ℕ := 10
def Kho_only : ℕ := 40
def Both : ℕ := 5

-- The lean statement that captures the problem of proving the total number of players equals 50
theorem total_players : (K - Both) + Kho_only + Both = 50 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_players_l592_59279


namespace NUMINAMATH_GPT_surface_area_of_z_eq_xy_over_a_l592_59249

noncomputable def surface_area (a : ℝ) (h : a > 0) : ℝ :=
  (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1)

theorem surface_area_of_z_eq_xy_over_a (a : ℝ) (h : a > 0) :
  surface_area a h = (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1) := 
sorry

end NUMINAMATH_GPT_surface_area_of_z_eq_xy_over_a_l592_59249


namespace NUMINAMATH_GPT_speed_of_man_rowing_upstream_l592_59216

theorem speed_of_man_rowing_upstream (Vm Vdownstream Vupstream : ℝ) (hVm : Vm = 40) (hVdownstream : Vdownstream = 45) : Vupstream = 35 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_rowing_upstream_l592_59216


namespace NUMINAMATH_GPT_circle_tangent_independence_l592_59293

noncomputable def e1 (r : ℝ) (β : ℝ) := r * Real.tan β
noncomputable def e2 (r : ℝ) (α : ℝ) := r * Real.tan α
noncomputable def e3 (r : ℝ) (β α : ℝ) := r * Real.tan (β - α)

theorem circle_tangent_independence 
  (O : ℝ) (r β α : ℝ) (hβ : β < π / 2) (hα : 0 < α) (hαβ : α < β) :
  (e1 r β) * (e2 r α) * (e3 r β α) / ((e1 r β) - (e2 r α) - (e3 r β α)) = r^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_independence_l592_59293


namespace NUMINAMATH_GPT_min_inv_sum_l592_59285

theorem min_inv_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 2 * a * 1 + b * 2 = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ (1/a) + (1/b) = 4 :=
sorry

end NUMINAMATH_GPT_min_inv_sum_l592_59285


namespace NUMINAMATH_GPT_intersection_M_N_l592_59218

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l592_59218


namespace NUMINAMATH_GPT_total_typing_cost_l592_59224

def typingCost (totalPages revisedOncePages revisedTwicePages : ℕ) (firstTimeCost revisionCost : ℕ) : ℕ := 
  let initialCost := totalPages * firstTimeCost
  let firstRevisionCost := revisedOncePages * revisionCost
  let secondRevisionCost := revisedTwicePages * (revisionCost * 2)
  initialCost + firstRevisionCost + secondRevisionCost

theorem total_typing_cost : typingCost 200 80 20 5 3 = 1360 := 
  by 
    rfl

end NUMINAMATH_GPT_total_typing_cost_l592_59224


namespace NUMINAMATH_GPT_derivative_at_zero_l592_59240

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem with the given conditions and expected result
theorem derivative_at_zero :
  (deriv f 0 = -120) :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l592_59240


namespace NUMINAMATH_GPT_determinant_of_matrix4x5_2x3_l592_59278

def matrix4x5_2x3 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, 5], ![2, 3]]

theorem determinant_of_matrix4x5_2x3 : matrix4x5_2x3.det = 2 := 
by
  sorry

end NUMINAMATH_GPT_determinant_of_matrix4x5_2x3_l592_59278


namespace NUMINAMATH_GPT_opposite_of_one_third_l592_59259

theorem opposite_of_one_third : -(1/3) = -1/3 := by
  sorry

end NUMINAMATH_GPT_opposite_of_one_third_l592_59259


namespace NUMINAMATH_GPT_rows_of_roses_l592_59242

variable (rows total_roses_per_row roses_per_row_red roses_per_row_non_red roses_per_row_white roses_per_row_pink total_pink_roses : ℕ)
variable (half_two_fifth three_fifth : ℚ)

-- Assume the conditions
axiom h1 : total_roses_per_row = 20
axiom h2 : roses_per_row_red = total_roses_per_row / 2
axiom h3 : roses_per_row_non_red = total_roses_per_row - roses_per_row_red
axiom h4 : roses_per_row_white = (3 / 5 : ℚ) * roses_per_row_non_red
axiom h5 : roses_per_row_pink = (2 / 5 : ℚ) * roses_per_row_non_red
axiom h6 : total_pink_roses = 40

-- Prove the number of rows in the garden
theorem rows_of_roses : rows = total_pink_roses / (roses_per_row_pink) :=
by
  sorry

end NUMINAMATH_GPT_rows_of_roses_l592_59242


namespace NUMINAMATH_GPT_dante_final_coconuts_l592_59235

theorem dante_final_coconuts
  (Paolo_coconuts : ℕ) (Dante_init_coconuts : ℝ)
  (Bianca_coconuts : ℕ) (Dante_final_coconuts : ℕ):
  Paolo_coconuts = 14 →
  Dante_init_coconuts = 1.5 * Real.sqrt Paolo_coconuts →
  Bianca_coconuts = 2 * (Paolo_coconuts + Int.floor Dante_init_coconuts) →
  Dante_final_coconuts = (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) - 
    (25 * (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) / 100) →
  Dante_final_coconuts = 3 :=
by
  sorry

end NUMINAMATH_GPT_dante_final_coconuts_l592_59235


namespace NUMINAMATH_GPT_sum_of_ages_3_years_ago_l592_59256

noncomputable def siblings_age_3_years_ago (R D S J : ℕ) : Prop :=
  R = D + 6 ∧
  D = S + 8 ∧
  J = R - 5 ∧
  R + 8 = 2 * (S + 8) ∧
  J + 10 = (D + 10) / 2 + 4 ∧
  S + 24 + J = 60 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43

theorem sum_of_ages_3_years_ago (R D S J : ℕ) :
  siblings_age_3_years_ago R D S J :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_ages_3_years_ago_l592_59256


namespace NUMINAMATH_GPT_sum_mod_9_l592_59201

theorem sum_mod_9 : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 :=
by sorry

end NUMINAMATH_GPT_sum_mod_9_l592_59201


namespace NUMINAMATH_GPT_A_not_on_transformed_plane_l592_59276

noncomputable def A : ℝ × ℝ × ℝ := (-3, -2, 4)
noncomputable def k : ℝ := -4/5
noncomputable def original_plane (x y z : ℝ) : Prop := 2 * x - 3 * y + z - 5 = 0

noncomputable def transformed_plane (x y z : ℝ) : Prop := 
  2 * x - 3 * y + z + (k * -5) = 0

theorem A_not_on_transformed_plane :
  ¬ transformed_plane (-3) (-2) 4 :=
by
  sorry

end NUMINAMATH_GPT_A_not_on_transformed_plane_l592_59276


namespace NUMINAMATH_GPT_train_speed_l592_59211

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end NUMINAMATH_GPT_train_speed_l592_59211


namespace NUMINAMATH_GPT_Ivan_pays_1_point_5_times_more_l592_59246

theorem Ivan_pays_1_point_5_times_more (x y : ℝ) (h : x = 2 * y) : 1.5 * (0.6 * x + 0.8 * y) = x + y :=
by
  sorry

end NUMINAMATH_GPT_Ivan_pays_1_point_5_times_more_l592_59246


namespace NUMINAMATH_GPT_trapezoid_side_lengths_l592_59273

theorem trapezoid_side_lengths
  (isosceles : ∀ (A B C D : ℝ) (height BE : ℝ), height = 2 → BE = 2 → A = 2 * Real.sqrt 2 → D = A → 12 = 0.5 * (B + C) * BE → A = D)
  (area : ∀ (BC AD : ℝ), 12 = 0.5 * (BC + AD) * 2)
  (height : ∀ (BE : ℝ), BE = 2)
  (intersect_right_angle : ∀ (A B C D : ℝ), 90 = 45 + 45) :
  ∃ A B C D, A = 2 * Real.sqrt 2 ∧ B = 4 ∧ C = 8 ∧ D = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_side_lengths_l592_59273


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_8_with_digit_sum_20_l592_59297

def sum_of_digits (n : Nat) : Nat :=
  Nat.digits 10 n |>.foldl (· + ·) 0

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20:
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ sum_of_digits n = 20 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 8 = 0 ∧ sum_of_digits m = 20 → n ≤ m :=
by { sorry }

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_8_with_digit_sum_20_l592_59297


namespace NUMINAMATH_GPT_range_of_m_min_of_squares_l592_59275

-- 1. Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 4)

-- 2. State the condition that f(x) ≤ -m^2 + 6m holds for all x
def condition (m : ℝ) : Prop := ∀ x : ℝ, f x ≤ -m^2 + 6 * m

-- 3. State the range of m to be proven
theorem range_of_m : ∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5 := 
sorry

-- 4. Auxiliary condition for part 2
def m_0 : ℝ := 5

-- 5. State the condition 3a + 4b + 5c = m_0
def sum_condition (a b c : ℝ) : Prop := 3 * a + 4 * b + 5 * c = m_0

-- 6. State the minimum value problem
theorem min_of_squares (a b c : ℝ) : sum_condition a b c → a^2 + b^2 + c^2 ≥ 1 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_min_of_squares_l592_59275


namespace NUMINAMATH_GPT_concyclic_projections_of_concyclic_quad_l592_59222

variables {A B C D A' B' C' D' : Type*}

def are_concyclic (p1 p2 p3 p4: Type*) : Prop :=
  sorry -- Assume we have a definition for concyclic property of points

def are_orthogonal_projection (x y : Type*) (l : Type*) : Type* :=
  sorry -- Assume we have a definition for orthogonal projection of a point on line

theorem concyclic_projections_of_concyclic_quad
  (hABCD : are_concyclic A B C D)
  (hA'_proj : are_orthogonal_projection A A' (BD))
  (hC'_proj : are_orthogonal_projection C C' (BD))
  (hB'_proj : are_orthogonal_projection B B' (AC))
  (hD'_proj : are_orthogonal_projection D D' (AC)) :
  are_concyclic A' B' C' D' :=
sorry

end NUMINAMATH_GPT_concyclic_projections_of_concyclic_quad_l592_59222


namespace NUMINAMATH_GPT_area_of_black_parts_l592_59282

theorem area_of_black_parts (x y : ℕ) (h₁ : x + y = 106) (h₂ : x + 2 * y = 170) : y = 64 :=
sorry

end NUMINAMATH_GPT_area_of_black_parts_l592_59282


namespace NUMINAMATH_GPT_hypotenuse_is_2_l592_59229

noncomputable def quadratic_trinomial_hypotenuse (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let xv := -b / (2 * a)
  let yv := a * xv^2 + b * xv + c
  if xv = (x1 + x2) / 2 then
    Real.sqrt 2 * abs (-b / a)
  else 0

theorem hypotenuse_is_2 {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  quadratic_trinomial_hypotenuse a b c = 2 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_is_2_l592_59229


namespace NUMINAMATH_GPT_negation_of_existential_l592_59231

def divisible_by (n x : ℤ) := ∃ k : ℤ, x = k * n
def odd (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

def P (x : ℤ) := divisible_by 7 x ∧ ¬ odd x

theorem negation_of_existential :
  (¬ ∃ x : ℤ, P x) ↔ ∀ x : ℤ, divisible_by 7 x → odd x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l592_59231


namespace NUMINAMATH_GPT_find_cost_price_l592_59215

theorem find_cost_price (SP : ℤ) (profit_percent : ℚ) (CP : ℤ) (h1 : SP = CP + (profit_percent * CP)) (h2 : SP = 240) (h3 : profit_percent = 0.25) : CP = 192 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l592_59215


namespace NUMINAMATH_GPT_no_descending_digits_multiple_of_111_l592_59250

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end NUMINAMATH_GPT_no_descending_digits_multiple_of_111_l592_59250


namespace NUMINAMATH_GPT_sum_non_solutions_is_neg21_l592_59209

noncomputable def sum_of_non_solutions (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2) : ℝ :=
  -21

theorem sum_non_solutions_is_neg21 (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) = 2) : 
  ∃! (x1 x2 : ℝ), ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2 → x = x1 ∨ x = x2 ∧ x1 + x2 = -21 :=
sorry

end NUMINAMATH_GPT_sum_non_solutions_is_neg21_l592_59209


namespace NUMINAMATH_GPT_kimberly_bought_skittles_l592_59234

-- Conditions
def initial_skittles : ℕ := 5
def total_skittles : ℕ := 12

-- Prove
theorem kimberly_bought_skittles : ∃ bought_skittles : ℕ, (total_skittles = initial_skittles + bought_skittles) ∧ bought_skittles = 7 :=
by
  sorry

end NUMINAMATH_GPT_kimberly_bought_skittles_l592_59234


namespace NUMINAMATH_GPT_Q_coordinates_l592_59268

def P : (ℝ × ℝ) := (2, -6)

def Q (x : ℝ) : (ℝ × ℝ) := (x, -6)

axiom PQ_parallel_to_x_axis : ∀ x, Q x = (x, -6)

axiom PQ_length : dist (Q 0) P = 2 ∨ dist (Q 4) P = 2

theorem Q_coordinates : Q 0 = (0, -6) ∨ Q 4 = (4, -6) :=
by {
  sorry
}

end NUMINAMATH_GPT_Q_coordinates_l592_59268


namespace NUMINAMATH_GPT_polynomial_expansion_l592_59298

theorem polynomial_expansion (z : ℤ) :
  (3 * z^3 + 6 * z^2 - 5 * z - 4) * (4 * z^4 - 3 * z^2 + 7) =
  12 * z^7 + 24 * z^6 - 29 * z^5 - 34 * z^4 + 36 * z^3 + 54 * z^2 + 35 * z - 28 := by
  -- Provide a proof here
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l592_59298


namespace NUMINAMATH_GPT_first_vessel_milk_water_l592_59208

variable (V : ℝ)

def vessel_ratio (v1 v2 : ℝ) : Prop := 
  v1 / v2 = 3 / 5

def vessel1_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 1 / 2

def vessel2_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 3 / 2

def mix_ratio (milk1 water1 milk2 water2 : ℝ) : Prop :=
  (milk1 + milk2) / (water1 + water2) = 1

theorem first_vessel_milk_water (V : ℝ) (v1 v2 : ℝ) (m1 w1 m2 w2 : ℝ)
  (hv : vessel_ratio v1 v2)
  (hv1 : vessel1_milk_water_ratio m1 w1)
  (hv2 : vessel2_milk_water_ratio m2 w2)
  (hmix : mix_ratio m1 w1 m2 w2) :
  vessel1_milk_water_ratio m1 w1 :=
  sorry

end NUMINAMATH_GPT_first_vessel_milk_water_l592_59208


namespace NUMINAMATH_GPT_imo_42nd_inequality_l592_59251

theorem imo_42nd_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 := by
  sorry

end NUMINAMATH_GPT_imo_42nd_inequality_l592_59251


namespace NUMINAMATH_GPT_possible_values_of_N_l592_59277

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_possible_values_of_N_l592_59277


namespace NUMINAMATH_GPT_sector_radius_l592_59284

theorem sector_radius (l : ℝ) (a : ℝ) (r : ℝ) (h1 : l = 2) (h2 : a = 4) (h3 : a = (1 / 2) * l * r) : r = 4 := by
  sorry

end NUMINAMATH_GPT_sector_radius_l592_59284


namespace NUMINAMATH_GPT_sum_50_to_75_l592_59220

-- Conditionally sum the series from 50 to 75
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_50_to_75 : sum_integers 50 75 = 1625 :=
by
  sorry

end NUMINAMATH_GPT_sum_50_to_75_l592_59220


namespace NUMINAMATH_GPT_projectile_height_reaches_45_at_t_0_5_l592_59258

noncomputable def quadratic (a b c : ℝ) : ℝ → ℝ :=
  λ t => a * t^2 + b * t + c

theorem projectile_height_reaches_45_at_t_0_5 :
  ∃ t : ℝ, quadratic (-16) 98.5 (-45) t = 45 ∧ 0 ≤ t ∧ t = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_projectile_height_reaches_45_at_t_0_5_l592_59258


namespace NUMINAMATH_GPT_merchant_loss_l592_59263

theorem merchant_loss (n m : ℝ) (h₁ : n ≠ m) : 
  let x := n / m
  let y := m / n
  x + y > 2 := by
sorry

end NUMINAMATH_GPT_merchant_loss_l592_59263


namespace NUMINAMATH_GPT_digit_place_value_ratio_l592_59294

theorem digit_place_value_ratio (n : ℚ) (h1 : n = 85247.2048) (h2 : ∃ d1 : ℚ, d1 * 0.1 = 0.2) (h3 : ∃ d2 : ℚ, d2 * 0.001 = 0.004) : 
  100 = 0.1 / 0.001 :=
by
  sorry

end NUMINAMATH_GPT_digit_place_value_ratio_l592_59294


namespace NUMINAMATH_GPT_fill_sink_time_l592_59266

theorem fill_sink_time {R1 R2 R T: ℝ} (h1: R1 = 1 / 210) (h2: R2 = 1 / 214) (h3: R = R1 + R2) (h4: T = 1 / R):
  T = 105.75 :=
by 
  sorry

end NUMINAMATH_GPT_fill_sink_time_l592_59266


namespace NUMINAMATH_GPT_rectangle_area_l592_59262

theorem rectangle_area
  (L B : ℕ)
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) : L * B = 2030 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l592_59262


namespace NUMINAMATH_GPT_range_of_a_l592_59213

theorem range_of_a (a : ℝ) : (a < 0 → (∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) ∧ 
                              (∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0) ↔ (x < -4 ∨ x ≥ -2)) ∧ 
                              ((¬(∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) 
                                → (¬(∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0))))
                            → (a ≤ -4 ∨ (a < 0 ∧ 3 * a >= -2)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l592_59213


namespace NUMINAMATH_GPT_minimum_value_of_C_over_D_is_three_l592_59295

variable (x : ℝ) (C D : ℝ)
variables (hxC : x^3 + 1/(x^3) = C) (hxD : x - 1/(x) = D)

theorem minimum_value_of_C_over_D_is_three (hC : C = D^3 + 3 * D) :
  ∃ x : ℝ, x^3 + 1/(x^3) = C ∧ x - 1/(x) = D → C / D ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_C_over_D_is_three_l592_59295


namespace NUMINAMATH_GPT_Z_real_axis_Z_first_quadrant_Z_on_line_l592_59207

-- Definitions based on the problem conditions
def Z_real (m : ℝ) : ℝ := m^2 + 5*m + 6
def Z_imag (m : ℝ) : ℝ := m^2 - 2*m - 15

-- Lean statement for the equivalent proof problem

theorem Z_real_axis (m : ℝ) :
  Z_imag m = 0 ↔ (m = -3 ∨ m = 5) := sorry

theorem Z_first_quadrant (m : ℝ) :
  (Z_real m > 0 ∧ Z_imag m > 0) ↔ (m > 5) := sorry

theorem Z_on_line (m : ℝ) :
  (Z_real m + Z_imag m + 5 = 0) ↔ (m = (-5 + Real.sqrt 41) / 2) := sorry

end NUMINAMATH_GPT_Z_real_axis_Z_first_quadrant_Z_on_line_l592_59207


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_l592_59206

noncomputable def a (n : ℕ) : ℤ := 2 * n - 4 -- General form of the arithmetic sequence

def is_geometric_sequence (s : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, (n > 1) → s (n+1) * s (n-1) = s n ^ 2

theorem arithmetic_geometric_seq:
  (∃ (d : ℤ) (a : ℕ → ℤ), a 5 = 6 ∧ 
  (∀ n, a n = 6 + (n - 5) * d) ∧ a (3) * a (11) = a (5) ^ 2 ∧
  (∀ k, 5 < k → is_geometric_sequence (fun n => a (k + n - 1)))) → 
  ∃ t : ℕ, ∀ n : ℕ, n <= 2015 → 
  (a n = 2 * n - 4 →  n = 7) := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_l592_59206


namespace NUMINAMATH_GPT_half_angle_quadrant_l592_59223

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 / 2 * Real.pi)
  : (k % 2 = 0 → k * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi + 3 / 4 * Real.pi) ∨
    (k % 2 = 1 → (k + 1) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k + 1) * Real.pi + 3 / 4 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_half_angle_quadrant_l592_59223


namespace NUMINAMATH_GPT_total_fish_caught_l592_59230

theorem total_fish_caught (C_trips : ℕ) (B_fish_per_trip : ℕ) (C_fish_per_trip : ℕ) (D_fish_per_trip : ℕ) (B_trips D_trips : ℕ) :
  C_trips = 10 →
  B_trips = 2 * C_trips →
  B_fish_per_trip = 400 →
  C_fish_per_trip = B_fish_per_trip * (1 + 2/5) →
  D_trips = 3 * C_trips →
  D_fish_per_trip = C_fish_per_trip * (1 + 50/100) →
  B_trips * B_fish_per_trip + C_trips * C_fish_per_trip + D_trips * D_fish_per_trip = 38800 := 
by
  sorry

end NUMINAMATH_GPT_total_fish_caught_l592_59230


namespace NUMINAMATH_GPT_remainder_of_3_pow_244_mod_5_l592_59247

theorem remainder_of_3_pow_244_mod_5 : (3^244) % 5 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_244_mod_5_l592_59247


namespace NUMINAMATH_GPT_green_peaches_eq_three_l592_59260

theorem green_peaches_eq_three (p r g : ℕ) (h1 : p = r + g) (h2 : r + 2 * g = p + 3) : g = 3 := 
by 
  sorry

end NUMINAMATH_GPT_green_peaches_eq_three_l592_59260


namespace NUMINAMATH_GPT_x_cubed_plus_y_cubed_l592_59226

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := 
by 
  sorry

end NUMINAMATH_GPT_x_cubed_plus_y_cubed_l592_59226


namespace NUMINAMATH_GPT_volume_of_parallelepiped_l592_59217

theorem volume_of_parallelepiped 
  (l w h : ℝ)
  (h1 : l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5)
  (h2 : h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13)
  (h3 : h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) 
  : l * w * h = 750 :=
sorry

end NUMINAMATH_GPT_volume_of_parallelepiped_l592_59217


namespace NUMINAMATH_GPT_decrease_is_75_86_percent_l592_59221

noncomputable def decrease_percent (x y z : ℝ) : ℝ :=
  let x' := 0.8 * x
  let y' := 0.75 * y
  let z' := 0.9 * z
  let original_value := x^2 * y^3 * z
  let new_value := (x')^2 * (y')^3 * z'
  let decrease_value := original_value - new_value
  decrease_value / original_value

theorem decrease_is_75_86_percent (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  decrease_percent x y z = 0.7586 :=
sorry

end NUMINAMATH_GPT_decrease_is_75_86_percent_l592_59221


namespace NUMINAMATH_GPT_initial_bottle_caps_l592_59296

theorem initial_bottle_caps (end_caps : ℕ) (eaten_caps : ℕ) (start_caps : ℕ) 
  (h1 : end_caps = 61) 
  (h2 : eaten_caps = 4) 
  (h3 : start_caps = end_caps + eaten_caps) : 
  start_caps = 65 := 
by 
  sorry

end NUMINAMATH_GPT_initial_bottle_caps_l592_59296


namespace NUMINAMATH_GPT_sequence_term_is_square_l592_59228

noncomputable def sequence_term (n : ℕ) : ℕ :=
  let part1 := (10 ^ (n + 1) - 1) / 9
  let part2 := (10 ^ (2 * n + 2) - 10 ^ (n + 1)) / 9
  1 + 4 * part1 + 4 * part2

theorem sequence_term_is_square (n : ℕ) : ∃ k : ℕ, k^2 = sequence_term n :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_is_square_l592_59228


namespace NUMINAMATH_GPT_find_k_l592_59238

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 3 = 2 * (n + k) + 5) : k = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_k_l592_59238


namespace NUMINAMATH_GPT_find_number_l592_59265

theorem find_number (x : ℝ) (h : x - (3/5) * x = 60) : x = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l592_59265


namespace NUMINAMATH_GPT_inverse_matrix_l592_59200

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![-1, -1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![-(1/3 : ℚ), -(7/3 : ℚ)], ![1/3, 4/3]]

theorem inverse_matrix : A.det ≠ 0 → A⁻¹ = A_inv := by
  sorry

end NUMINAMATH_GPT_inverse_matrix_l592_59200


namespace NUMINAMATH_GPT_symmetrical_point_correct_l592_59227

variables (x₁ y₁ : ℝ)

def symmetrical_point_x_axis (x y : ℝ) : ℝ × ℝ :=
(x, -y)

theorem symmetrical_point_correct : symmetrical_point_x_axis 3 2 = (3, -2) :=
by
  -- This is where we would provide the proof
  sorry

end NUMINAMATH_GPT_symmetrical_point_correct_l592_59227


namespace NUMINAMATH_GPT_find_m_value_l592_59255

theorem find_m_value (m : ℝ) (h₀ : m > 0) (h₁ : (4 - m) / (m - 2) = m) : m = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l592_59255


namespace NUMINAMATH_GPT_log_comparison_l592_59269

noncomputable def logBase (a x : ℝ) := Real.log x / Real.log a

theorem log_comparison
  (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) 
  (m : ℝ) (hm : m = logBase a (a^2 + 1))
  (n : ℝ) (hn : n = logBase a (a + 1))
  (p : ℝ) (hp : p = logBase a (2 * a)) :
  p > m ∧ m > n :=
by
  sorry

end NUMINAMATH_GPT_log_comparison_l592_59269


namespace NUMINAMATH_GPT_remaining_cookies_l592_59288

theorem remaining_cookies : 
  let naomi_cookies := 53
  let oliver_cookies := 67
  let penelope_cookies := 29
  let total_cookies := naomi_cookies + oliver_cookies + penelope_cookies
  let package_size := 15
  total_cookies % package_size = 14 :=
by
  sorry

end NUMINAMATH_GPT_remaining_cookies_l592_59288


namespace NUMINAMATH_GPT_abs_lt_2_sufficient_not_necessary_l592_59233

theorem abs_lt_2_sufficient_not_necessary (x : ℝ) :
  (|x| < 2 → x^2 - x - 6 < 0) ∧ ¬ (x^2 - x - 6 < 0 → |x| < 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_lt_2_sufficient_not_necessary_l592_59233


namespace NUMINAMATH_GPT_no_tangential_triangle_exists_l592_59214

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse C2
def C2 (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Additional condition that the point (1, 1) lies on C2
def point_on_C2 (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (1^2) / (a^2) + (1^2) / (b^2) = 1

-- The theorem to prove
theorem no_tangential_triangle_exists (a b : ℝ) (h : a > b ∧ b > 0) :
  point_on_C2 a b h →
  ¬ ∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C1 C.1 C.2) ∧ 
    (C2 a b A.1 A.2 h ∧ C2 a b B.1 B.2 h ∧ C2 a b C.1 C.2 h) :=
by sorry

end NUMINAMATH_GPT_no_tangential_triangle_exists_l592_59214


namespace NUMINAMATH_GPT_bob_age_is_725_l592_59219

theorem bob_age_is_725 (n : ℕ) (h1 : ∃ k : ℤ, n - 3 = k^2) (h2 : ∃ j : ℤ, n + 4 = j^3) : n = 725 :=
sorry

end NUMINAMATH_GPT_bob_age_is_725_l592_59219


namespace NUMINAMATH_GPT_cos_270_eq_zero_l592_59243

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end NUMINAMATH_GPT_cos_270_eq_zero_l592_59243


namespace NUMINAMATH_GPT_school_dance_boys_count_l592_59257

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end NUMINAMATH_GPT_school_dance_boys_count_l592_59257


namespace NUMINAMATH_GPT_vector_eq_to_slope_intercept_form_l592_59274

theorem vector_eq_to_slope_intercept_form :
  ∀ (x y : ℝ), (2 * (x - 4) + 5 * (y - 1)) = 0 → y = -(2 / 5) * x + 13 / 5 := 
by 
  intros x y h
  sorry

end NUMINAMATH_GPT_vector_eq_to_slope_intercept_form_l592_59274


namespace NUMINAMATH_GPT_friend_owns_10_bicycles_l592_59261

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_friend_owns_10_bicycles_l592_59261


namespace NUMINAMATH_GPT_find_x_l592_59299

theorem find_x (x : ℕ) (h : 2^x - 2^(x - 2) = 3 * 2^10) : x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l592_59299
