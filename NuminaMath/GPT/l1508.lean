import Mathlib

namespace NUMINAMATH_GPT_set_notation_nat_lt_3_l1508_150809

theorem set_notation_nat_lt_3 : {x : ℕ | x < 3} = {0, 1, 2} := 
sorry

end NUMINAMATH_GPT_set_notation_nat_lt_3_l1508_150809


namespace NUMINAMATH_GPT_interval_of_increase_l1508_150884

noncomputable def f (x : ℝ) : ℝ :=
  -abs x

theorem interval_of_increase :
  ∀ x, f x ≤ f (x + 1) ↔ x ≤ 0 := by
  sorry

end NUMINAMATH_GPT_interval_of_increase_l1508_150884


namespace NUMINAMATH_GPT_average_height_students_l1508_150880

/-- Given the average heights of female and male students, and the ratio of men to women, the average height -/
theorem average_height_students
  (avg_female_height : ℕ)
  (avg_male_height : ℕ)
  (ratio_men_women : ℕ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 182)
  (h3 : ratio_men_women = 5) :
  (avg_female_height + 5 * avg_male_height) / (1 + 5) = 180 :=
by
  sorry

end NUMINAMATH_GPT_average_height_students_l1508_150880


namespace NUMINAMATH_GPT_no_such_abc_exists_l1508_150861

theorem no_such_abc_exists :
  ¬ ∃ (a b c : ℝ), 
      ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0) ∨
       (a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∧ c < 0 ∧ a > 0)) ∧
      ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∨ c < 0 ∧ a > 0) ∨
       (a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_such_abc_exists_l1508_150861


namespace NUMINAMATH_GPT_Amy_balloons_l1508_150873

-- Defining the conditions
def James_balloons : ℕ := 1222
def more_balloons : ℕ := 208

-- Defining Amy's balloons as a proof goal
theorem Amy_balloons : ∀ (Amy_balloons : ℕ), James_balloons - more_balloons = Amy_balloons → Amy_balloons = 1014 :=
by
  intros Amy_balloons h
  sorry

end NUMINAMATH_GPT_Amy_balloons_l1508_150873


namespace NUMINAMATH_GPT_quadratic_equation_general_form_l1508_150893

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 2 * (x + 2)^2 + (x + 3) * (x - 2) = -11 ↔ 3 * x^2 + 9 * x + 13 = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_general_form_l1508_150893


namespace NUMINAMATH_GPT_fraction_A_BC_l1508_150889

-- Definitions for amounts A, B, C and the total T
variable (T : ℝ) (A : ℝ) (B : ℝ) (C : ℝ)

-- Given conditions
def conditions : Prop :=
  T = 300 ∧
  A = 120.00000000000001 ∧
  B = (6 / 9) * (A + C) ∧
  A + B + C = T

-- The fraction of the amount A gets compared to B and C together
def fraction (x : ℝ) : Prop :=
  A = x * (B + C)

-- The proof goal
theorem fraction_A_BC : conditions T A B C → fraction A B C (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_fraction_A_BC_l1508_150889


namespace NUMINAMATH_GPT_find_y_l1508_150856

-- Definitions of the angles
def angle_ABC : ℝ := 80
def angle_BAC : ℝ := 70
def angle_BCA : ℝ := 180 - angle_ABC - angle_BAC -- calculation of third angle in triangle ABC

-- Right angle in triangle CDE
def angle_ECD : ℝ := 90

-- Defining the proof problem
theorem find_y (y : ℝ) : 
  angle_BCA = 30 →
  angle_CDE = angle_BCA →
  angle_CDE + y + angle_ECD = 180 → 
  y = 60 := by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_y_l1508_150856


namespace NUMINAMATH_GPT_study_days_needed_l1508_150848

theorem study_days_needed :
  let math_chapters := 4
  let math_worksheets := 7
  let physics_chapters := 5
  let physics_worksheets := 9
  let chemistry_chapters := 6
  let chemistry_worksheets := 8

  let math_chapter_hours := 2.5
  let math_worksheet_hours := 1.5
  let physics_chapter_hours := 3.0
  let physics_worksheet_hours := 2.0
  let chemistry_chapter_hours := 3.5
  let chemistry_worksheet_hours := 1.75

  let daily_study_hours := 7.0
  let breaks_first_3_hours := 3 * 10 / 60.0
  let breaks_next_3_hours := 3 * 15 / 60.0
  let breaks_final_hour := 1 * 20 / 60.0
  let snack_breaks := 2 * 20 / 60.0
  let lunch_break := 45 / 60.0

  let break_time_per_day := breaks_first_3_hours + breaks_next_3_hours + breaks_final_hour + snack_breaks + lunch_break
  let effective_study_time_per_day := daily_study_hours - break_time_per_day

  let total_math_hours := (math_chapters * math_chapter_hours) + (math_worksheets * math_worksheet_hours)
  let total_physics_hours := (physics_chapters * physics_chapter_hours) + (physics_worksheets * physics_worksheet_hours)
  let total_chemistry_hours := (chemistry_chapters * chemistry_chapter_hours) + (chemistry_worksheets * chemistry_worksheet_hours)

  let total_study_hours := total_math_hours + total_physics_hours + total_chemistry_hours
  let total_study_days := total_study_hours / effective_study_time_per_day
  
  total_study_days.ceil = 23 := by sorry

end NUMINAMATH_GPT_study_days_needed_l1508_150848


namespace NUMINAMATH_GPT_tangent_line_values_l1508_150853

theorem tangent_line_values (m : ℝ) :
  (∃ s : ℝ, 3 * s^2 = 12 ∧ 12 * s + m = s^3 - 2) ↔ (m = -18 ∨ m = 14) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_values_l1508_150853


namespace NUMINAMATH_GPT_building_height_l1508_150860

noncomputable def height_of_building (H_f L_f L_b : ℝ) : ℝ :=
  (H_f * L_b) / L_f

theorem building_height (H_f L_f L_b H_b : ℝ)
  (H_f_val : H_f = 17.5)
  (L_f_val : L_f = 40.25)
  (L_b_val : L_b = 28.75)
  (H_b_val : H_b = 12.4375) :
  height_of_building H_f L_f L_b = H_b := by
  rw [H_f_val, L_f_val, L_b_val, H_b_val]
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_building_height_l1508_150860


namespace NUMINAMATH_GPT_shortest_distance_between_semicircles_l1508_150887

theorem shortest_distance_between_semicircles
  (ABCD : Type)
  (AD : ℝ)
  (shaded_area : ℝ)
  (is_rectangle : true)
  (AD_eq_10 : AD = 10)
  (shaded_area_eq_100 : shaded_area = 100) :
  ∃ d : ℝ, d = 2.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_between_semicircles_l1508_150887


namespace NUMINAMATH_GPT_sales_tax_difference_l1508_150885

theorem sales_tax_difference (price : ℝ) (rate1 rate2 : ℝ) : 
  rate1 = 0.085 → rate2 = 0.07 → price = 50 → 
  (price * rate1 - price * rate2) = 0.75 := 
by 
  intros h_rate1 h_rate2 h_price
  rw [h_rate1, h_rate2, h_price] 
  simp
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1508_150885


namespace NUMINAMATH_GPT_root_of_quadratic_eq_l1508_150869

theorem root_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, (x₁ = 0 ∧ x₂ = 2) ∧ ∀ x : ℝ, x^2 - 2 * x = 0 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_eq_l1508_150869


namespace NUMINAMATH_GPT_painting_time_l1508_150872

-- Definitions translated from conditions
def total_weight_tons := 5
def weight_per_ball_kg := 4
def number_of_students := 10
def balls_per_student_per_6_minutes := 5

-- Derived Definitions
def total_weight_kg := total_weight_tons * 1000
def total_balls := total_weight_kg / weight_per_ball_kg
def balls_painted_by_all_students_per_6_minutes := number_of_students * balls_per_student_per_6_minutes
def required_intervals := total_balls / balls_painted_by_all_students_per_6_minutes
def total_time_minutes := required_intervals * 6

-- The theorem statement
theorem painting_time : total_time_minutes = 150 := by
  sorry

end NUMINAMATH_GPT_painting_time_l1508_150872


namespace NUMINAMATH_GPT_find_quadratic_function_find_vertex_find_range_l1508_150864

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def satisfies_points (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = 0 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 2 = -3

theorem find_quadratic_function : ∃ a b c, satisfies_points a b c ∧ (a = 1 ∧ b = -2 ∧ c = -3) :=
sorry

theorem find_vertex (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∃ x y, x = 1 ∧ y = -4 ∧ ∀ x', x' > 1 → quadratic_function a b c x' > quadratic_function a b c x :=
sorry

theorem find_range (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∀ x, -1 < x ∧ x < 2 → -4 < quadratic_function a b c x ∧ quadratic_function a b c x < 0 :=
sorry

end NUMINAMATH_GPT_find_quadratic_function_find_vertex_find_range_l1508_150864


namespace NUMINAMATH_GPT_ratio_B_C_l1508_150894

variable (A B C : ℕ)
variable (h1 : A = B + 2)
variable (h2 : A + B + C = 37)
variable (h3 : B = 14)

theorem ratio_B_C : B / C = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_B_C_l1508_150894


namespace NUMINAMATH_GPT_kangaroo_can_jump_exact_200_in_30_jumps_l1508_150868

/-!
  A kangaroo can jump:
  - 3 meters using its left leg
  - 5 meters using its right leg
  - 7 meters using both legs
  - -3 meters backward
  We need to prove that the kangaroo can jump exactly 200 meters in 30 jumps.
 -/

theorem kangaroo_can_jump_exact_200_in_30_jumps :
  ∃ (n3 n5 n7 nm3 : ℕ),
    (n3 + n5 + n7 + nm3 = 30) ∧
    (3 * n3 + 5 * n5 + 7 * n7 - 3 * nm3 = 200) :=
sorry

end NUMINAMATH_GPT_kangaroo_can_jump_exact_200_in_30_jumps_l1508_150868


namespace NUMINAMATH_GPT_max_coins_identifiable_l1508_150823

theorem max_coins_identifiable (n : ℕ) : exists (c : ℕ), c = 2 * n^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_max_coins_identifiable_l1508_150823


namespace NUMINAMATH_GPT_N_is_composite_l1508_150877

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  sorry

end NUMINAMATH_GPT_N_is_composite_l1508_150877


namespace NUMINAMATH_GPT_original_number_is_842_l1508_150888

theorem original_number_is_842 (x y z : ℕ) (h1 : x * z = y^2)
  (h2 : 100 * z + x = 100 * x + z - 594)
  (h3 : 10 * z + y = 10 * y + z - 18)
  (hx : x = 8) (hy : y = 4) (hz : z = 2) :
  100 * x + 10 * y + z = 842 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_842_l1508_150888


namespace NUMINAMATH_GPT_sum_not_zero_l1508_150896

theorem sum_not_zero (a b c d : ℝ) (h1 : a * b * c - d = 1) (h2 : b * c * d - a = 2) 
  (h3 : c * d * a - b = 3) (h4 : d * a * b - c = -6) : a + b + c + d ≠ 0 :=
sorry

end NUMINAMATH_GPT_sum_not_zero_l1508_150896


namespace NUMINAMATH_GPT_evaluate_g_at_neg3_l1508_150811

def g (x : ℝ) : ℝ := 3 * x ^ 5 - 5 * x ^ 4 + 7 * x ^ 3 - 10 * x ^ 2 - 12 * x + 36

theorem evaluate_g_at_neg3 : g (-3) = -1341 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg3_l1508_150811


namespace NUMINAMATH_GPT_time_between_train_arrivals_l1508_150810

-- Define the conditions as given in the problem statement
def passengers_per_train : ℕ := 320 + 200
def total_passengers_per_hour : ℕ := 6240
def minutes_per_hour : ℕ := 60

-- Declare the statement to be proven
theorem time_between_train_arrivals: 
  (total_passengers_per_hour / passengers_per_train) = (minutes_per_hour / 5) := by 
  sorry

end NUMINAMATH_GPT_time_between_train_arrivals_l1508_150810


namespace NUMINAMATH_GPT_min_value_range_of_a_l1508_150813

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp (-2 * x) + a * (2 * x + 1) * Real.exp (-x) + x^2 + x

theorem min_value_range_of_a (a : ℝ) (h : a > 0)
  (min_f : ∃ x : ℝ, f a x = Real.log a ^ 2 + 3 * Real.log a + 2) :
  a ∈ Set.Ici (Real.exp (-3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_range_of_a_l1508_150813


namespace NUMINAMATH_GPT_stratified_sampling_l1508_150871

theorem stratified_sampling :
  let total_employees := 150
  let middle_managers := 30
  let senior_managers := 10
  let selected_employees := 30
  let selection_probability := selected_employees / total_employees
  let selected_middle_managers := middle_managers * selection_probability
  let selected_senior_managers := senior_managers * selection_probability
  selected_middle_managers = 6 ∧ selected_senior_managers = 2 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1508_150871


namespace NUMINAMATH_GPT_baby_frogs_on_rock_l1508_150897

theorem baby_frogs_on_rock (f_l f_L f_T : ℕ) (h1 : f_l = 5) (h2 : f_L = 3) (h3 : f_T = 32) : 
  f_T - (f_l + f_L) = 24 :=
by sorry

end NUMINAMATH_GPT_baby_frogs_on_rock_l1508_150897


namespace NUMINAMATH_GPT_savings_increase_is_100_percent_l1508_150892

variable (I : ℝ) -- Initial income
variable (S : ℝ) -- Initial savings
variable (I2 : ℝ) -- Income in the second year
variable (E1 : ℝ) -- Expenditure in the first year
variable (E2 : ℝ) -- Expenditure in the second year
variable (S2 : ℝ) -- Second year savings

-- Initial conditions
def initial_savings (I : ℝ) : ℝ := 0.25 * I
def first_year_expenditure (I : ℝ) (S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.25 * I

-- Total expenditure condition
def total_expenditure_condition (E1 : ℝ) (E2 : ℝ) : Prop := E1 + E2 = 2 * E1

-- Prove that the savings increase in the second year is 100%
theorem savings_increase_is_100_percent :
   ∀ (I S E1 I2 E2 S2 : ℝ),
     S = initial_savings I →
     E1 = first_year_expenditure I S →
     I2 = second_year_income I →
     total_expenditure_condition E1 E2 →
     S2 = I2 - E2 →
     ((S2 - S) / S) * 100 = 100 := by
  sorry

end NUMINAMATH_GPT_savings_increase_is_100_percent_l1508_150892


namespace NUMINAMATH_GPT_num_pos_three_digit_div_by_seven_l1508_150804

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end NUMINAMATH_GPT_num_pos_three_digit_div_by_seven_l1508_150804


namespace NUMINAMATH_GPT_evaluate_expression_l1508_150824

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a * b^2 = 59 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1508_150824


namespace NUMINAMATH_GPT_probability_of_dice_outcome_l1508_150815

theorem probability_of_dice_outcome : 
  let p_one_digit := 3 / 4
  let p_two_digit := 1 / 4
  let comb := Nat.choose 5 3
  (comb * (p_one_digit^3) * (p_two_digit^2)) = 135 / 512 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_dice_outcome_l1508_150815


namespace NUMINAMATH_GPT_first_negative_term_position_l1508_150838

def a1 : ℤ := 1031
def d : ℤ := -3
def nth_term (n : ℕ) : ℤ := a1 + (n - 1 : ℤ) * d

theorem first_negative_term_position : ∃ n : ℕ, nth_term n < 0 ∧ n = 345 := 
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_first_negative_term_position_l1508_150838


namespace NUMINAMATH_GPT_sqrt_sq_eq_abs_l1508_150846

theorem sqrt_sq_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| :=
sorry

end NUMINAMATH_GPT_sqrt_sq_eq_abs_l1508_150846


namespace NUMINAMATH_GPT_fraction_of_white_surface_area_l1508_150822

def larger_cube_edge : ℕ := 4
def number_of_smaller_cubes : ℕ := 64
def number_of_white_cubes : ℕ := 8
def number_of_red_cubes : ℕ := 56
def total_surface_area : ℕ := 6 * (larger_cube_edge * larger_cube_edge)
def minimized_white_surface_area : ℕ := 7

theorem fraction_of_white_surface_area :
  minimized_white_surface_area % total_surface_area = 7 % 96 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_white_surface_area_l1508_150822


namespace NUMINAMATH_GPT_suggested_bacon_students_l1508_150808

-- Definitions based on the given conditions
def students_mashed_potatoes : ℕ := 330
def students_tomatoes : ℕ := 76
def difference_bacon_mashed_potatoes : ℕ := 61

-- Lean 4 statement to prove the correct answer
theorem suggested_bacon_students : ∃ (B : ℕ), students_mashed_potatoes = B + difference_bacon_mashed_potatoes ∧ B = 269 := 
by
  sorry

end NUMINAMATH_GPT_suggested_bacon_students_l1508_150808


namespace NUMINAMATH_GPT_log_value_l1508_150843

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_value (x : ℝ) (h : log_base 3 (5 * x) = 3) : log_base x 125 = 3 / 2 :=
  by
  sorry

end NUMINAMATH_GPT_log_value_l1508_150843


namespace NUMINAMATH_GPT_slope_of_l4_l1508_150830

open Real

def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 6
def pointD : ℝ × ℝ := (0, -2)
def line2 (y : ℝ) : Prop := y = -1
def area_triangle_DEF := 4

theorem slope_of_l4 
  (l4_slope : ℝ)
  (H1 : ∃ x, line1 x (-1))
  (H2 : ∀ x y, 
         x ≠ 0 ∧
         y ≠ -2 ∧
         y ≠ -1 →
         line2 y →
         l4_slope = (y - (-2)) / (x - 0) →
         (1/2) * |(y + 1)| * (sqrt ((x-0) * (x-0) + (y-(-2)) * (y-(-2)))) = area_triangle_DEF ) :
  l4_slope = 1 / 8 :=
sorry

end NUMINAMATH_GPT_slope_of_l4_l1508_150830


namespace NUMINAMATH_GPT_greatest_a_inequality_l1508_150817

theorem greatest_a_inequality :
  ∃ a : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) ∧
          (∀ b : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) → b ≤ a) ∧
          a = 2 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_greatest_a_inequality_l1508_150817


namespace NUMINAMATH_GPT_tangent_line_b_value_l1508_150803

noncomputable def b_value : ℝ := Real.log 2 - 1

theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x > 0, (fun x => Real.log x) x = (1/2) * x + b → ∃ c : ℝ, c = b) → b = Real.log 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_b_value_l1508_150803


namespace NUMINAMATH_GPT_initial_temperature_l1508_150826

theorem initial_temperature (T_initial : ℝ) 
  (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ) 
  (T_heat : ℝ) (T_cool : ℝ) (T_target : ℝ) (T_final : ℝ) 
  (h1 : heating_rate = 5) (h2 : cooling_rate = 7)
  (h3 : T_target = 240) (h4 : T_final = 170) 
  (h5 : total_time = 46)
  (h6 : T_cool = (T_target - T_final) / cooling_rate)
  (h7: total_time = T_heat + T_cool)
  (h8 : T_heat = (T_target - T_initial) / heating_rate) :
  T_initial = 60 :=
by
  -- Proof yet to be filled in
  sorry

end NUMINAMATH_GPT_initial_temperature_l1508_150826


namespace NUMINAMATH_GPT_expression_calculates_to_l1508_150806

noncomputable def mixed_number : ℚ := 3 + 3 / 4

noncomputable def decimal_to_fraction : ℚ := 2 / 10

noncomputable def given_expression : ℚ := ((mixed_number * decimal_to_fraction) / 135) * 5.4

theorem expression_calculates_to : given_expression = 0.03 := by
  sorry

end NUMINAMATH_GPT_expression_calculates_to_l1508_150806


namespace NUMINAMATH_GPT_johns_tour_program_days_l1508_150876

/-- John has Rs 360 for his expenses. If he exceeds his days by 4 days, he must cut down daily expenses by Rs 3. Prove that the number of days of John's tour program is 20. -/
theorem johns_tour_program_days
    (d e : ℕ)
    (h1 : 360 = e * d)
    (h2 : 360 = (e - 3) * (d + 4)) : 
    d = 20 := 
  sorry

end NUMINAMATH_GPT_johns_tour_program_days_l1508_150876


namespace NUMINAMATH_GPT_max_value_of_a_l1508_150890

theorem max_value_of_a (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a < 3 * b) (h2 : b < 2 * c) (h3 : c < 5 * d) (h4 : d < 150) : a ≤ 4460 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1508_150890


namespace NUMINAMATH_GPT_combined_weight_difference_l1508_150841

def john_weight : ℕ := 81
def roy_weight : ℕ := 79
def derek_weight : ℕ := 91
def samantha_weight : ℕ := 72

theorem combined_weight_difference :
  derek_weight - samantha_weight = 19 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_difference_l1508_150841


namespace NUMINAMATH_GPT_incorrect_operation_in_list_l1508_150827

open Real

theorem incorrect_operation_in_list :
  ¬ (abs ((-2)^2) = -2) :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_incorrect_operation_in_list_l1508_150827


namespace NUMINAMATH_GPT_solve_for_x_l1508_150855

theorem solve_for_x (x : ℝ) : (1 + 2*x + 3*x^2) / (3 + 2*x + x^2) = 3 → x = -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1508_150855


namespace NUMINAMATH_GPT_skittles_problem_l1508_150849

def initial_skittles : ℕ := 76
def shared_skittles : ℕ := 72
def final_skittles (initial shared : ℕ) : ℕ := initial - shared

theorem skittles_problem : final_skittles initial_skittles shared_skittles = 4 := by
  sorry

end NUMINAMATH_GPT_skittles_problem_l1508_150849


namespace NUMINAMATH_GPT_square_area_from_circles_l1508_150879

theorem square_area_from_circles :
  (∀ (r : ℝ), r = 7 → ∀ (n : ℕ), n = 4 → (∃ (side_length : ℝ), side_length = 2 * (2 * r))) →
  ∀ (side_length : ℝ), side_length = 28 →
  (∃ (area : ℝ), area = side_length * side_length ∧ area = 784) :=
sorry

end NUMINAMATH_GPT_square_area_from_circles_l1508_150879


namespace NUMINAMATH_GPT_decryption_proof_l1508_150874

-- Definitions
def Original_Message := "МОСКВА"
def Encrypted_Text_1 := "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ"
def Encrypted_Text_2 := "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП"
def Encrypted_Text_3 := "РТПАИОМВСВТИЕОБПРОЕННИИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК"

noncomputable def Encrypted_Message_1 := "ЙМЫВОТСЬЛКЪГВЦАЯЯ"
noncomputable def Encrypted_Message_2 := "УКМАПОЧСРКЩВЗАХ"
noncomputable def Encrypted_Message_3 := "ШМФЭОГЧСЙЪКФЬВЫЕАКК"

def Decrypted_Message_1_and_3 := "ПОВТОРЕНИЕМАТЬУЧЕНИЯ"
def Decrypted_Message_2 := "СМОТРИВКОРЕНЬ"

-- Theorem statement
theorem decryption_proof :
  (Encrypted_Text_1 = Encrypted_Text_3 ∧ Original_Message = "МОСКВА" ∧ Encrypted_Message_1 = Encrypted_Message_3) →
  (Decrypted_Message_1_and_3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ Decrypted_Message_2 = "СМОТРИВКОРЕНЬ") :=
by 
  sorry

end NUMINAMATH_GPT_decryption_proof_l1508_150874


namespace NUMINAMATH_GPT_range_of_k_l1508_150852

noncomputable def f (k : ℝ) (x : ℝ) := (Real.exp x) / (x^2) + 2 * k * Real.log x - k * x

theorem range_of_k (k : ℝ) (h₁ : ∀ x > 0, (deriv (f k) x = 0) → x = 2) : k < Real.exp 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1508_150852


namespace NUMINAMATH_GPT_inequality_solution_set_l1508_150819

noncomputable def solution_set (a b : ℝ) := {x : ℝ | 2 < x ∧ x < 3}

theorem inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (ax^2 + 5 * x + b > 0)) →
  (∀ x : ℝ, (-6) * x^2 - 5 * x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1508_150819


namespace NUMINAMATH_GPT_parabola_2_second_intersection_x_l1508_150842

-- Definitions of the conditions in the problem
def parabola_1_intersects : Prop := 
  (∀ x : ℝ, (x = 10 ∨ x = 13) → (∃ y : ℝ, (x, y) ∈ ({p | p = (10, 0)} ∪ {p | p = (13, 0)})))

def parabola_2_intersects : Prop := 
  (∃ x : ℝ, x = 13)

def vertex_bisects_segment : Prop := 
  (∃ a : ℝ, 2 * 11.5 = a)

-- The theorem we want to prove
theorem parabola_2_second_intersection_x : 
  parabola_1_intersects ∧ parabola_2_intersects ∧ vertex_bisects_segment → 
  (∃ t : ℝ, t = 33) := 
  by
  sorry

end NUMINAMATH_GPT_parabola_2_second_intersection_x_l1508_150842


namespace NUMINAMATH_GPT_n_plus_floor_sqrt2_plus1_pow_n_is_odd_l1508_150839

theorem n_plus_floor_sqrt2_plus1_pow_n_is_odd (n : ℕ) (h : n > 0) : 
  Odd (n + ⌊(Real.sqrt 2 + 1) ^ n⌋) :=
by sorry

end NUMINAMATH_GPT_n_plus_floor_sqrt2_plus1_pow_n_is_odd_l1508_150839


namespace NUMINAMATH_GPT_ratio_volumes_of_spheres_l1508_150821

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end NUMINAMATH_GPT_ratio_volumes_of_spheres_l1508_150821


namespace NUMINAMATH_GPT_option_c_correct_l1508_150802

-- Statement of the problem: Prove that (x-3)^2 = x^2 - 6x + 9

theorem option_c_correct (x : ℝ) : (x - 3) ^ 2 = x ^ 2 - 6 * x + 9 :=
by
  sorry

end NUMINAMATH_GPT_option_c_correct_l1508_150802


namespace NUMINAMATH_GPT_tom_spending_is_correct_l1508_150847

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end NUMINAMATH_GPT_tom_spending_is_correct_l1508_150847


namespace NUMINAMATH_GPT_solve_cubic_equation_l1508_150858

theorem solve_cubic_equation : 
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^3 - y^3 = 999 ∧ (x, y) = (12, 9) ∨ (x, y) = (10, 1) := 
  by
  sorry

end NUMINAMATH_GPT_solve_cubic_equation_l1508_150858


namespace NUMINAMATH_GPT_solution1_solution2_l1508_150845

noncomputable def problem1 (a : ℝ) : Prop :=
  (∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + 4 < 0)

theorem solution1 (a : ℝ) : problem1 a ↔ a < -3 ∨ a ≥ -1 := 
  sorry

noncomputable def problem2 (a : ℝ) (x : ℝ) : Prop :=
  (-2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0)

noncomputable def condition2 (a x : ℝ) : Prop :=
  (2*a < x ∧ x < a+1)

theorem solution2 (a : ℝ) : (∀ x, condition2 a x → problem2 a x) → a ≥ -1/2 :=
  sorry

end NUMINAMATH_GPT_solution1_solution2_l1508_150845


namespace NUMINAMATH_GPT_alpha_plus_beta_l1508_150831

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π)
variable (hβ : 0 < β ∧ β < π)
variable (h1 : Real.sin (α - β) = 3 / 4)
variable (h2 : Real.tan α / Real.tan β = -5)

theorem alpha_plus_beta (h3 : α + β = 5 * π / 6) : α + β = 5 * π / 6 :=
by
  sorry

end NUMINAMATH_GPT_alpha_plus_beta_l1508_150831


namespace NUMINAMATH_GPT_distance_to_destination_l1508_150883

-- Conditions
def Speed : ℝ := 65 -- speed in km/hr
def Time : ℝ := 3   -- time in hours

-- Question to prove
theorem distance_to_destination : Speed * Time = 195 := by
  sorry

end NUMINAMATH_GPT_distance_to_destination_l1508_150883


namespace NUMINAMATH_GPT_fence_calculation_l1508_150881

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end NUMINAMATH_GPT_fence_calculation_l1508_150881


namespace NUMINAMATH_GPT_fundraiser_total_money_l1508_150818

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end NUMINAMATH_GPT_fundraiser_total_money_l1508_150818


namespace NUMINAMATH_GPT_div_by_5_factor_l1508_150800

theorem div_by_5_factor {x y z : ℤ} (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * 5 * (y - z) * (z - x) * (x - y) :=
sorry

end NUMINAMATH_GPT_div_by_5_factor_l1508_150800


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_inequality_l1508_150854

theorem necessary_but_not_sufficient_condition_for_inequality 
    {a b c : ℝ} (h : a * c^2 ≥ b * c^2) : ¬(a > b → (a * c^2 < b * c^2)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_inequality_l1508_150854


namespace NUMINAMATH_GPT_teammates_score_l1508_150829

def Lizzie_score := 4
def Nathalie_score := Lizzie_score + 3
def combined_Lizzie_Nathalie := Lizzie_score + Nathalie_score
def Aimee_score := 2 * combined_Lizzie_Nathalie
def total_team_score := 50
def total_combined_score := Lizzie_score + Nathalie_score + Aimee_score

theorem teammates_score : total_team_score - total_combined_score = 17 :=
by
  sorry

end NUMINAMATH_GPT_teammates_score_l1508_150829


namespace NUMINAMATH_GPT_cannot_form_figureB_l1508_150812

-- Define the pieces as terms
inductive Piece
| square : Piece
| rectangle : Π (h w : ℕ), Piece   -- h: height, w: width

-- Define the available pieces in a list (assuming these are predefined somewhere)
def pieces : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

-- Define the figures that can be formed
def figureA : List Piece := [Piece.square, Piece.square, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

def figureC : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, 
                             Piece.square, Piece.square]

def figureD : List Piece := [Piece.rectangle 2 2, Piece.square, Piece.square, Piece.square,
                              Piece.square]

def figureE : List Piece := [Piece.rectangle 3 1, Piece.square, Piece.square, Piece.square]

-- Define the figure B that we need to prove cannot be formed
def figureB : List Piece := [Piece.rectangle 5 1, Piece.square, Piece.square, Piece.square,
                              Piece.square]

theorem cannot_form_figureB :
  ¬(∃ arrangement : List Piece, arrangement ⊆ pieces ∧ arrangement = figureB) :=
sorry

end NUMINAMATH_GPT_cannot_form_figureB_l1508_150812


namespace NUMINAMATH_GPT_angela_height_l1508_150882

def height_of_Amy : ℕ := 150
def height_of_Helen : ℕ := height_of_Amy + 3
def height_of_Angela : ℕ := height_of_Helen + 4

theorem angela_height : height_of_Angela = 157 := by
  sorry

end NUMINAMATH_GPT_angela_height_l1508_150882


namespace NUMINAMATH_GPT_map_line_segments_l1508_150807

def point : Type := ℝ × ℝ

def transformation (f : point → point) (p q : point) : Prop := f p = q

def counterclockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

def clockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

theorem map_line_segments :
  (transformation counterclockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation counterclockwise_rotation_180 (2, -5) (-2, 5)) ∨
  (transformation clockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation clockwise_rotation_180 (2, -5) (-2, 5)) :=
by
  sorry

end NUMINAMATH_GPT_map_line_segments_l1508_150807


namespace NUMINAMATH_GPT_incorrect_statement_D_l1508_150844

theorem incorrect_statement_D :
  (∃ x : ℝ, x ^ 3 = -64 ∧ x = -4) ∧
  (∃ y : ℝ, y ^ 2 = 49 ∧ y = 7) ∧
  (∃ z : ℝ, z ^ 3 = 1 / 27 ∧ z = 1 / 3) ∧
  (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4 ∨ w = -1 / 4)
  → ¬ (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1508_150844


namespace NUMINAMATH_GPT_odd_factors_360_l1508_150895

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end NUMINAMATH_GPT_odd_factors_360_l1508_150895


namespace NUMINAMATH_GPT_cardinal_transitivity_l1508_150866

variable {α β γ : Cardinal}

theorem cardinal_transitivity (h1 : α < β) (h2 : β < γ) : α < γ :=
  sorry

end NUMINAMATH_GPT_cardinal_transitivity_l1508_150866


namespace NUMINAMATH_GPT_subset_implies_a_geq_4_l1508_150825

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + 3 ≤ 0}

theorem subset_implies_a_geq_4 (a : ℝ) :
  A ⊆ B a → a ≥ 4 := sorry

end NUMINAMATH_GPT_subset_implies_a_geq_4_l1508_150825


namespace NUMINAMATH_GPT_find_a_b_find_c_range_l1508_150857

noncomputable def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

theorem find_a_b (a b c : ℝ) (extreme_x1 extreme_x2 : ℝ) (h1 : extreme_x1 = 1) (h2 : extreme_x2 = 2) 
  (h3 : (deriv (f a b c) 1) = 0) (h4 : (deriv (f a b c) 2) = 0) : 
  a = -3 ∧ b = 4 :=
by sorry

theorem find_c_range (c : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f (-3) 4 c x < c^2) : 
  c ∈ Set.Iio (-1) ∪ Set.Ioi 9 :=
by sorry

end NUMINAMATH_GPT_find_a_b_find_c_range_l1508_150857


namespace NUMINAMATH_GPT_makeup_set_cost_l1508_150814

theorem makeup_set_cost (initial : ℕ) (gift : ℕ) (needed : ℕ) (total_cost : ℕ) :
  initial = 35 → gift = 20 → needed = 10 → total_cost = initial + gift + needed → total_cost = 65 :=
by
  intros h_init h_gift h_needed h_cost
  sorry

end NUMINAMATH_GPT_makeup_set_cost_l1508_150814


namespace NUMINAMATH_GPT_number_of_arrangements_BANANA_l1508_150805

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_BANANA_l1508_150805


namespace NUMINAMATH_GPT_setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l1508_150833

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end NUMINAMATH_GPT_setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l1508_150833


namespace NUMINAMATH_GPT_largest_e_l1508_150851

variable (a b c d e : ℤ)

theorem largest_e 
  (h1 : a - 1 = b + 2) 
  (h2 : a - 1 = c - 3)
  (h3 : a - 1 = d + 4)
  (h4 : a - 1 = e - 6) 
  : e > a ∧ e > b ∧ e > c ∧ e > d := 
sorry

end NUMINAMATH_GPT_largest_e_l1508_150851


namespace NUMINAMATH_GPT_solve_for_x_l1508_150837

theorem solve_for_x (x : ℝ) (h :  9 / x^2 = x / 25) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1508_150837


namespace NUMINAMATH_GPT_xiao_hua_correct_questions_l1508_150878

-- Definitions of the problem conditions
def n : Nat := 20
def p_correct : Int := 5
def p_wrong : Int := -2
def score : Int := 65

-- Theorem statement to prove the number of correct questions
theorem xiao_hua_correct_questions : 
  ∃ k : Nat, k = ((n : Int) - ((n * p_correct - score) / (p_correct - p_wrong))) ∧ 
               k = 15 :=
by
  sorry

end NUMINAMATH_GPT_xiao_hua_correct_questions_l1508_150878


namespace NUMINAMATH_GPT_min_sum_of_factors_l1508_150816

theorem min_sum_of_factors (x y z : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x * y * z = 3920) : x + y + z = 70 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l1508_150816


namespace NUMINAMATH_GPT_pickle_to_tomato_ratio_l1508_150850

theorem pickle_to_tomato_ratio 
  (mushrooms : ℕ) 
  (cherry_tomatoes : ℕ) 
  (pickles : ℕ) 
  (bacon_bits : ℕ) 
  (red_bacon_bits : ℕ) 
  (h1 : mushrooms = 3) 
  (h2 : cherry_tomatoes = 2 * mushrooms)
  (h3 : red_bacon_bits = 32)
  (h4 : bacon_bits = 3 * red_bacon_bits)
  (h5 : bacon_bits = 4 * pickles) : 
  pickles/cherry_tomatoes = 4 :=
by
  sorry

end NUMINAMATH_GPT_pickle_to_tomato_ratio_l1508_150850


namespace NUMINAMATH_GPT_instructors_meeting_l1508_150832

theorem instructors_meeting (R P E M : ℕ) (hR : R = 5) (hP : P = 8) (hE : E = 10) (hM : M = 9) :
  Nat.lcm (Nat.lcm R P) (Nat.lcm E M) = 360 :=
by
  rw [hR, hP, hE, hM]
  sorry

end NUMINAMATH_GPT_instructors_meeting_l1508_150832


namespace NUMINAMATH_GPT_exponent_multiplication_l1508_150862

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l1508_150862


namespace NUMINAMATH_GPT_new_ratio_books_to_clothes_l1508_150899

-- Given initial conditions
def initial_ratio := (7, 4, 3)
def electronics_weight : ℕ := 12
def clothes_removed : ℕ := 8

-- Definitions based on the problem
def part_weight : ℕ := electronics_weight / initial_ratio.2.2
def initial_books_weight : ℕ := initial_ratio.1 * part_weight
def initial_clothes_weight : ℕ := initial_ratio.2.1 * part_weight
def new_clothes_weight : ℕ := initial_clothes_weight - clothes_removed

-- Proof of the new ratio
theorem new_ratio_books_to_clothes : (initial_books_weight, new_clothes_weight) = (7 * part_weight, 2 * part_weight) :=
sorry

end NUMINAMATH_GPT_new_ratio_books_to_clothes_l1508_150899


namespace NUMINAMATH_GPT_small_beaker_salt_fraction_l1508_150834

theorem small_beaker_salt_fraction
  (S L : ℝ) 
  (h1 : L = 5 * S)
  (h2 : L * (1 / 5) = S)
  (h3 : L * 0.3 = S * 1.5)
  : (S * 0.5) / S = 0.5 :=
by 
  sorry

end NUMINAMATH_GPT_small_beaker_salt_fraction_l1508_150834


namespace NUMINAMATH_GPT_fraction_of_product_l1508_150820

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end NUMINAMATH_GPT_fraction_of_product_l1508_150820


namespace NUMINAMATH_GPT_complement_intersection_l1508_150801

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3, 4}
def A_complement : Set ℕ := U \ A

theorem complement_intersection :
  (A_complement ∩ B) = {2, 4} :=
by 
  sorry

end NUMINAMATH_GPT_complement_intersection_l1508_150801


namespace NUMINAMATH_GPT_john_needs_more_usd_l1508_150840

noncomputable def additional_usd (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ) : ℝ :=
  let eur_to_usd := 1 / 0.84
  let sgd_to_usd := 1 / 1.34
  let jpy_to_usd := 1 / 110.35
  let total_needed_usd := needed_eur * eur_to_usd + needed_sgd * sgd_to_usd
  let total_has_usd := has_usd + has_jpy * jpy_to_usd
  total_needed_usd - total_has_usd

theorem john_needs_more_usd :
  ∀ (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ),
    needed_eur = 7.50 → needed_sgd = 5.00 → has_usd = 2.00 → has_jpy = 500 →
    additional_usd needed_eur needed_sgd has_usd has_jpy = 6.13 :=
by
  intros needed_eur needed_sgd has_usd has_jpy
  intros hneeded_eur hneeded_sgd hhas_usd hhas_jpy
  unfold additional_usd
  rw [hneeded_eur, hneeded_sgd, hhas_usd, hhas_jpy]
  sorry

end NUMINAMATH_GPT_john_needs_more_usd_l1508_150840


namespace NUMINAMATH_GPT_sum_due_is_l1508_150886

-- Definitions and conditions from the problem
def BD : ℤ := 288
def TD : ℤ := 240
def face_value (FV : ℤ) : Prop := BD = TD + (TD * TD) / FV

-- Proof statement
theorem sum_due_is (FV : ℤ) (h : face_value FV) : FV = 1200 :=
sorry

end NUMINAMATH_GPT_sum_due_is_l1508_150886


namespace NUMINAMATH_GPT_cos_theta_plus_5π_div_6_l1508_150875

theorem cos_theta_plus_5π_div_6 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcond : Real.sin (θ / 2 + π / 6) = 3 / 5) :
  Real.cos (θ + 5 * π / 6) = -24 / 25 :=
by
  sorry -- Proof is skipped as instructed

end NUMINAMATH_GPT_cos_theta_plus_5π_div_6_l1508_150875


namespace NUMINAMATH_GPT_min_expression_l1508_150835

theorem min_expression 
  (a b c : ℝ)
  (ha : -1 < a ∧ a < 1)
  (hb : -1 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 1) :
  ∃ m, m = 2 ∧ ∀ x y z, (-1 < x ∧ x < 1) → (-1 < y ∧ y < 1) → (-1 < z ∧ z < 1) → 
  ( 1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)) ) ≥ m :=
sorry

end NUMINAMATH_GPT_min_expression_l1508_150835


namespace NUMINAMATH_GPT_kitty_vacuum_time_l1508_150867

theorem kitty_vacuum_time
  (weekly_toys : ℕ := 5)
  (weekly_windows : ℕ := 15)
  (weekly_furniture : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  (weeks : ℕ := 4)
  : (weekly_toys + weekly_windows + weekly_furniture) * weeks < total_cleaning_time ∧ ((total_cleaning_time - ((weekly_toys + weekly_windows + weekly_furniture) * weeks)) / weeks = 20)
  := by
  sorry

end NUMINAMATH_GPT_kitty_vacuum_time_l1508_150867


namespace NUMINAMATH_GPT_largest_value_among_given_numbers_l1508_150898

theorem largest_value_among_given_numbers :
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  d >= a ∧ d >= b ∧ d >= c ∧ d >= e :=
by
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  sorry

end NUMINAMATH_GPT_largest_value_among_given_numbers_l1508_150898


namespace NUMINAMATH_GPT_max_number_of_9_letter_palindromes_l1508_150870

theorem max_number_of_9_letter_palindromes : 26^5 = 11881376 :=
by sorry

end NUMINAMATH_GPT_max_number_of_9_letter_palindromes_l1508_150870


namespace NUMINAMATH_GPT_calculate_area_correct_l1508_150865

-- Define the side length of the square
def side_length : ℝ := 5

-- Define the rotation angles in degrees
def rotation_angles : List ℝ := [0, 30, 45, 60]

-- Define the area calculation function (to be implemented)
def calculate_overlap_area (s : ℝ) (angles : List ℝ) : ℝ := sorry

-- Define the proof that the calculated area is equal to 123.475
theorem calculate_area_correct : calculate_overlap_area side_length rotation_angles = 123.475 :=
by
  sorry

end NUMINAMATH_GPT_calculate_area_correct_l1508_150865


namespace NUMINAMATH_GPT_max_cake_pieces_l1508_150828

theorem max_cake_pieces (m n : ℕ) (h₁ : m ≥ 4) (h₂ : n ≥ 4)
    (h : (m-4)*(n-4) = m * n) :
    m * n = 72 :=
by
  sorry

end NUMINAMATH_GPT_max_cake_pieces_l1508_150828


namespace NUMINAMATH_GPT_least_whole_number_clock_equiv_l1508_150859

theorem least_whole_number_clock_equiv (h : ℕ) (h_gt_10 : h > 10) : 
  ∃ k, k = 12 ∧ (h^2 - h) % 12 = 0 ∧ h = 12 :=
by 
  sorry

end NUMINAMATH_GPT_least_whole_number_clock_equiv_l1508_150859


namespace NUMINAMATH_GPT_fine_per_day_of_absence_l1508_150863

theorem fine_per_day_of_absence :
  ∃ x: ℝ, ∀ (total_days work_wage total_received_days absent_days: ℝ),
  total_days = 30 →
  work_wage = 10 →
  total_received_days = 216 →
  absent_days = 7 →
  (total_days - absent_days) * work_wage - (absent_days * x) = total_received_days :=
sorry

end NUMINAMATH_GPT_fine_per_day_of_absence_l1508_150863


namespace NUMINAMATH_GPT_shoe_cost_on_monday_l1508_150891

theorem shoe_cost_on_monday 
  (price_thursday : ℝ) 
  (increase_rate : ℝ) 
  (decrease_rate : ℝ) 
  (price_thursday_eq : price_thursday = 40)
  (increase_rate_eq : increase_rate = 0.10)
  (decrease_rate_eq : decrease_rate = 0.10)
  :
  let price_friday := price_thursday * (1 + increase_rate)
  let discount := price_friday * decrease_rate
  let price_monday := price_friday - discount
  price_monday = 39.60 :=
by
  sorry

end NUMINAMATH_GPT_shoe_cost_on_monday_l1508_150891


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_progression_l1508_150836

theorem geometric_sequence_arithmetic_progression
  (q : ℝ) (h_q : q ≠ 1)
  (a : ℕ → ℝ) (m n p : ℕ)
  (h1 : ∃ a1, ∀ k, a k = a1 * q ^ (k - 1))
  (h2 : a n ^ 2 = a m * a p) :
  2 * n = m + p := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_progression_l1508_150836
