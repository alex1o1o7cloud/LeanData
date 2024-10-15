import Mathlib

namespace NUMINAMATH_GPT_correct_value_l1238_123892

theorem correct_value (x : ℝ) (h : x + 2.95 = 9.28) : x - 2.95 = 3.38 :=
by
  sorry

end NUMINAMATH_GPT_correct_value_l1238_123892


namespace NUMINAMATH_GPT_intersection_of_sets_l1238_123836

def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets :
  setA ∩ setB = {x | 0 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1238_123836


namespace NUMINAMATH_GPT_initial_breads_count_l1238_123808

theorem initial_breads_count :
  ∃ (X : ℕ), ((((X / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2 = 3 ∧ X = 127 :=
by sorry

end NUMINAMATH_GPT_initial_breads_count_l1238_123808


namespace NUMINAMATH_GPT_point_B_value_l1238_123895

theorem point_B_value :
  ∃ B : ℝ, (|B + 1| = 4) ∧ (B = 3 ∨ B = -5) := 
by
  sorry

end NUMINAMATH_GPT_point_B_value_l1238_123895


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1238_123816

theorem problem1 (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + a + 3 = 0) → (a ≤ -2 ∨ a ≥ 6) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a + 3 ≥ 4) → 
    (if a > 2 then 
      ∀ x : ℝ, ((x ≤ 1) ∨ (x ≥ a-1)) 
    else if a = 2 then 
      ∀ x : ℝ, true
    else 
      ∀ x : ℝ, ((x ≤ a - 1) ∨ (x ≥ 1))) :=
sorry

theorem problem3 (a : ℝ) :
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ x^2 - a*x + a + 3 = 0) → (6 ≤ a ∧ a ≤ 7) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1238_123816


namespace NUMINAMATH_GPT_work_completion_time_l1238_123870

theorem work_completion_time (d : ℕ) (h : d = 9) : 3 * d = 27 := by
  sorry

end NUMINAMATH_GPT_work_completion_time_l1238_123870


namespace NUMINAMATH_GPT_radius_of_circle_with_area_3_14_l1238_123819

theorem radius_of_circle_with_area_3_14 (A : ℝ) (π : ℝ) (hA : A = 3.14) (hπ : π = 3.14) (h_area : A = π * r^2) : r = 1 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_with_area_3_14_l1238_123819


namespace NUMINAMATH_GPT_parabola_intersection_square_l1238_123873

theorem parabola_intersection_square (p : ℝ) :
   (∃ (x : ℝ), (x = 1 ∨ x = 2) ∧ x^2 * p = 1 ∨ x^2 * p = 2)
   → (1 / 4 ≤ p ∧ p ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersection_square_l1238_123873


namespace NUMINAMATH_GPT_factorize_x2y_minus_4y_l1238_123898

variable {x y : ℝ}

theorem factorize_x2y_minus_4y : x^2 * y - 4 * y = y * (x + 2) * (x - 2) :=
sorry

end NUMINAMATH_GPT_factorize_x2y_minus_4y_l1238_123898


namespace NUMINAMATH_GPT_find_circumference_l1238_123854

theorem find_circumference
  (C : ℕ)
  (h1 : ∃ (vA vB : ℕ), C > 0 ∧ vA > 0 ∧ vB > 0 ∧ 
                        (120 * (C/2 + 80)) = ((C - 80) * (C/2 - 120)) ∧
                        (C - 240) / vA = (C + 240) / vB) :
  C = 520 := 
  sorry

end NUMINAMATH_GPT_find_circumference_l1238_123854


namespace NUMINAMATH_GPT_product_of_prs_eq_60_l1238_123805

theorem product_of_prs_eq_60 (p r s : ℕ) (h1 : 3 ^ p + 3 ^ 5 = 270) (h2 : 2 ^ r + 46 = 94) (h3 : 6 ^ s + 5 ^ 4 = 1560) :
  p * r * s = 60 :=
  sorry

end NUMINAMATH_GPT_product_of_prs_eq_60_l1238_123805


namespace NUMINAMATH_GPT_lucy_fish_moved_l1238_123833

theorem lucy_fish_moved (original_count moved_count remaining_count : ℝ)
  (h1: original_count = 212.0)
  (h2: remaining_count = 144.0) :
  moved_count = original_count - remaining_count :=
by sorry

end NUMINAMATH_GPT_lucy_fish_moved_l1238_123833


namespace NUMINAMATH_GPT_solve_equation_l1238_123878

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -1/2) ↔ (x / (x + 2) + 1 = 1 / (x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1238_123878


namespace NUMINAMATH_GPT_number_of_restaurants_l1238_123812

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end NUMINAMATH_GPT_number_of_restaurants_l1238_123812


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1238_123822

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1238_123822


namespace NUMINAMATH_GPT_option_a_is_odd_l1238_123814

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_a_is_odd (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_odd (a + 2 * b + 1) :=
by sorry

end NUMINAMATH_GPT_option_a_is_odd_l1238_123814


namespace NUMINAMATH_GPT_obtuse_triangle_condition_l1238_123872

theorem obtuse_triangle_condition
  (a b c : ℝ) 
  (h : ∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 ∧ a^2 + b^2 - c^2 < 0)
  : (∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 → a^2 + b^2 - c^2 < 0) := 
sorry

end NUMINAMATH_GPT_obtuse_triangle_condition_l1238_123872


namespace NUMINAMATH_GPT_cost_price_of_watch_l1238_123837

theorem cost_price_of_watch (CP : ℝ) (h_loss : 0.54 * CP = SP_loss)
                            (h_gain : 1.04 * CP = SP_gain)
                            (h_diff : SP_gain - SP_loss = 140) :
                            CP = 280 :=
by {
    sorry
}

end NUMINAMATH_GPT_cost_price_of_watch_l1238_123837


namespace NUMINAMATH_GPT_find_slope_l1238_123894

theorem find_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : ∃ m : ℝ, m = -4/7 ∧ (∀ x, y = m * x + 4) := 
by
  sorry

end NUMINAMATH_GPT_find_slope_l1238_123894


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1238_123867

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := 
by 
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1238_123867


namespace NUMINAMATH_GPT_probability_of_seeing_red_light_l1238_123883

def red_light_duration : ℝ := 30
def yellow_light_duration : ℝ := 5
def green_light_duration : ℝ := 40

def total_cycle_duration : ℝ := red_light_duration + yellow_light_duration + green_light_duration

theorem probability_of_seeing_red_light :
  (red_light_duration / total_cycle_duration) = 30 / 75 := by
  sorry

end NUMINAMATH_GPT_probability_of_seeing_red_light_l1238_123883


namespace NUMINAMATH_GPT_servings_required_l1238_123879

/-- Each serving of cereal is 2.0 cups, and 36 cups are needed. Prove that the number of servings required is 18. -/
theorem servings_required (cups_per_serving : ℝ) (total_cups : ℝ) (h1 : cups_per_serving = 2.0) (h2 : total_cups = 36.0) :
  total_cups / cups_per_serving = 18 :=
by
  sorry

end NUMINAMATH_GPT_servings_required_l1238_123879


namespace NUMINAMATH_GPT_amy_hours_per_week_l1238_123809

theorem amy_hours_per_week (hours_summer_per_week : ℕ) (weeks_summer : ℕ) (earnings_summer : ℕ)
  (weeks_school_year : ℕ) (earnings_school_year_goal : ℕ) :
  (hours_summer_per_week = 40) →
  (weeks_summer = 12) →
  (earnings_summer = 4800) →
  (weeks_school_year = 36) →
  (earnings_school_year_goal = 7200) →
  (∃ hours_school_year_per_week : ℕ, hours_school_year_per_week = 20) :=
by
  sorry

end NUMINAMATH_GPT_amy_hours_per_week_l1238_123809


namespace NUMINAMATH_GPT_sum_of_two_consecutive_squares_l1238_123801

variable {k m A : ℕ}

theorem sum_of_two_consecutive_squares :
  (∃ k : ℕ, A^2 = (k+1)^3 - k^3) → (∃ m : ℕ, A = m^2 + (m+1)^2) :=
by sorry

end NUMINAMATH_GPT_sum_of_two_consecutive_squares_l1238_123801


namespace NUMINAMATH_GPT_fraction_power_multiplication_l1238_123842

theorem fraction_power_multiplication :
  ( (5 / 8: ℚ) ^ 2 * (3 / 4) ^ 2 * (2 / 3) = 75 / 512) := 
  by
  sorry

end NUMINAMATH_GPT_fraction_power_multiplication_l1238_123842


namespace NUMINAMATH_GPT_KochCurve_MinkowskiDimension_l1238_123849

noncomputable def minkowskiDimensionOfKochCurve : ℝ :=
  let N (n : ℕ) := 3 * (4 ^ (n - 1))
  (Real.log 4) / (Real.log 3)

theorem KochCurve_MinkowskiDimension : minkowskiDimensionOfKochCurve = (Real.log 4) / (Real.log 3) := by
  sorry

end NUMINAMATH_GPT_KochCurve_MinkowskiDimension_l1238_123849


namespace NUMINAMATH_GPT_problem_statement_l1238_123899

variable (x P : ℝ)

theorem problem_statement
  (h1 : x^2 - 5 * x + 6 < 0)
  (h2 : P = x^2 + 5 * x + 6) :
  (20 < P) ∧ (P < 30) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1238_123899


namespace NUMINAMATH_GPT_reflection_of_P_across_y_axis_l1238_123876

-- Define the initial point P as a tuple
def P : ℝ × ℝ := (1, -2)

-- Define the reflection across the y-axis function
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- State the theorem that we want to prove
theorem reflection_of_P_across_y_axis :
  reflect_y_axis P = (-1, -2) :=
by
  -- placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_reflection_of_P_across_y_axis_l1238_123876


namespace NUMINAMATH_GPT_baseball_games_per_month_l1238_123896

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_baseball_games_per_month_l1238_123896


namespace NUMINAMATH_GPT_problem1_problem2_l1238_123840

theorem problem1 : (40 * Real.sqrt 3 - 18 * Real.sqrt 3 + 8 * Real.sqrt 3) / 6 = 5 * Real.sqrt 3 := 
by sorry

theorem problem2 : (Real.sqrt 3 - 2)^2023 * (Real.sqrt 3 + 2)^2023
                 - Real.sqrt 4 * Real.sqrt (1 / 2)
                 - (Real.pi - 1)^0
                = -2 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1238_123840


namespace NUMINAMATH_GPT_jorge_total_goals_l1238_123861

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end NUMINAMATH_GPT_jorge_total_goals_l1238_123861


namespace NUMINAMATH_GPT_distribute_paper_clips_l1238_123802

theorem distribute_paper_clips (total_clips : ℕ) (boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : boxes = 9) :
  total_clips / boxes = clips_per_box ↔ clips_per_box = 9 :=
by
  sorry

end NUMINAMATH_GPT_distribute_paper_clips_l1238_123802


namespace NUMINAMATH_GPT_temperature_on_friday_is_72_l1238_123853

-- Define the temperatures on specific days
def temp_sunday := 40
def temp_monday := 50
def temp_tuesday := 65
def temp_wednesday := 36
def temp_thursday := 82
def temp_saturday := 26

-- Average temperature over the week
def average_temp := 53

-- Total number of days in a week
def days_in_week := 7

-- Calculate the total sum of temperatures given the average temperature
def total_sum_temp : ℤ := average_temp * days_in_week

-- Sum of known temperatures from specific days
def known_sum_temp : ℤ := temp_sunday + temp_monday + temp_tuesday + temp_wednesday + temp_thursday + temp_saturday

-- Define the temperature on Friday
def temp_friday : ℤ := total_sum_temp - known_sum_temp

theorem temperature_on_friday_is_72 : temp_friday = 72 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_temperature_on_friday_is_72_l1238_123853


namespace NUMINAMATH_GPT_neg_p_equiv_l1238_123806

def p : Prop := ∃ x₀ : ℝ, x₀^2 + 1 > 3 * x₀

theorem neg_p_equiv :
  ¬ p ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x := by
  sorry

end NUMINAMATH_GPT_neg_p_equiv_l1238_123806


namespace NUMINAMATH_GPT_y_intercept_of_line_l1238_123845

theorem y_intercept_of_line (x y : ℝ) (h : 5 * x - 3 * y = 15) : (0, -5) = (0, (-5 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1238_123845


namespace NUMINAMATH_GPT_proof_problem_l1238_123868

open Real

noncomputable def set_A : Set ℝ :=
  {x | x = tan (-19 * π / 6) ∨ x = sin (-19 * π / 6)}

noncomputable def set_B : Set ℝ :=
  {m | 0 <= m ∧ m <= 4}

noncomputable def set_C (a : ℝ) : Set ℝ :=
  {x | a + 1 < x ∧ x < 2 * a}

theorem proof_problem (a : ℝ) :
  set_A = {-sqrt 3 / 3, -1 / 2} ∧
  set_B = {m | 0 <= m ∧ m <= 4} ∧
  (set_A ∪ set_B) = {-sqrt 3 / 3, -1 / 2, 0, 4} →
  (∀ a, set_C a ⊆ (set_A ∪ set_B) → 1 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1238_123868


namespace NUMINAMATH_GPT_total_cost_of_books_l1238_123893

theorem total_cost_of_books (total_children : ℕ) (n : ℕ) (extra_payment_per_child : ℕ) (cost : ℕ) :
  total_children = 12 →
  n = 2 →
  extra_payment_per_child = 10 →
  (total_children - n) * extra_payment_per_child = 100 →
  cost = 600 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_cost_of_books_l1238_123893


namespace NUMINAMATH_GPT_point_B_between_A_and_C_l1238_123862

theorem point_B_between_A_and_C (a b c : ℚ) (h_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end NUMINAMATH_GPT_point_B_between_A_and_C_l1238_123862


namespace NUMINAMATH_GPT_find_nine_day_segment_l1238_123803

/-- 
  Definitions:
  - ws_day: The Winter Solstice day, December 21, 2012.
  - j1_day: New Year's Day, January 1, 2013.
  - Calculate the total days difference between ws_day and j1_day.
  - Check that the distribution of days into 9-day segments leads to January 1, 2013, being the third day of the second segment.
-/
def ws_day : ℕ := 21
def j1_day : ℕ := 1
def days_in_december : ℕ := 31
def days_ws_to_end_dec : ℕ := days_in_december - ws_day + 1
def total_days : ℕ := days_ws_to_end_dec + j1_day

theorem find_nine_day_segment : (total_days % 9) = 3 ∧ (total_days / 9) = 1 := by
  sorry  -- Proof skipped

end NUMINAMATH_GPT_find_nine_day_segment_l1238_123803


namespace NUMINAMATH_GPT_problem_solution_l1238_123889

theorem problem_solution (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b = 1) :
  (a + 1 / b) ^ 2 + (b + 1 / a) ^ 2 ≥ 25 / 2 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1238_123889


namespace NUMINAMATH_GPT_prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l1238_123823

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end NUMINAMATH_GPT_prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l1238_123823


namespace NUMINAMATH_GPT_domain_of_f_l1238_123884

theorem domain_of_f (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1238_123884


namespace NUMINAMATH_GPT_triangle_area_in_circle_l1238_123825

theorem triangle_area_in_circle (r : ℝ) (arc1 arc2 arc3 : ℝ) 
  (circumference_eq : arc1 + arc2 + arc3 = 24)
  (radius_eq : 2 * Real.pi * r = 24) : 
  1 / 2 * (r ^ 2) * (Real.sin (105 * Real.pi / 180) + Real.sin (120 * Real.pi / 180) + Real.sin (135 * Real.pi / 180)) = 364.416 / (Real.pi ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_in_circle_l1238_123825


namespace NUMINAMATH_GPT_trigonometric_identity_l1238_123847

theorem trigonometric_identity (α : ℝ)
  (h1 : Real.sin (π + α) = 3 / 5)
  (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin ((π + α) / 2) - Real.cos ((π + α) / 2)) / 
  (Real.sin ((π - α) / 2) - Real.cos ((π - α) / 2)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1238_123847


namespace NUMINAMATH_GPT_surface_area_of_cube_given_sphere_surface_area_l1238_123804

noncomputable def edge_length_of_cube (sphere_surface_area : ℝ) : ℝ :=
  let a_square := 2
  Real.sqrt a_square

def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

theorem surface_area_of_cube_given_sphere_surface_area (sphere_surface_area : ℝ) :
  sphere_surface_area = 6 * Real.pi → 
  surface_area_of_cube (edge_length_of_cube sphere_surface_area) = 12 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_given_sphere_surface_area_l1238_123804


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l1238_123834

theorem lcm_of_two_numbers (a b : ℕ) (h_prod : a * b = 145862784) (h_hcf : Nat.gcd a b = 792) : Nat.lcm a b = 184256 :=
by {
  sorry
}

end NUMINAMATH_GPT_lcm_of_two_numbers_l1238_123834


namespace NUMINAMATH_GPT_hollow_iron_ball_diameter_l1238_123807

theorem hollow_iron_ball_diameter (R r : ℝ) (s : ℝ) (thickness : ℝ) 
  (h1 : thickness = 1) (h2 : s = 7.5) 
  (h3 : R - r = thickness) 
  (h4 : 4 / 3 * π * R^3 = 4 / 3 * π * s * (R^3 - r^3)) : 
  2 * R = 44.44 := 
sorry

end NUMINAMATH_GPT_hollow_iron_ball_diameter_l1238_123807


namespace NUMINAMATH_GPT_storybook_pages_l1238_123820

theorem storybook_pages :
  (10 + 5) / (1 - (1 / 5) * 2) = 25 := by
  sorry

end NUMINAMATH_GPT_storybook_pages_l1238_123820


namespace NUMINAMATH_GPT_all_statements_true_l1238_123838

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined (x : ℝ) : ∃ y, g x = y
axiom g_positive (x : ℝ) : g x > 0
axiom g_multiplicative (a b : ℝ) : g (a) * g (b) = g (a + b)
axiom g_div (a b : ℝ) (h : a > b) : g (a - b) = g (a) / g (b)

theorem all_statements_true :
  (g 0 = 1) ∧
  (∀ a, g (-a) = 1 / g (a)) ∧
  (∀ a, g (a) = (g (3 * a))^(1 / 3)) ∧
  (∀ a b, b > a → g (b - a) < g (b)) :=
by
  sorry

end NUMINAMATH_GPT_all_statements_true_l1238_123838


namespace NUMINAMATH_GPT_white_area_l1238_123850

/-- The area of a 5 by 17 rectangular sign. -/
def sign_area : ℕ := 5 * 17

/-- The area covered by the letter L. -/
def L_area : ℕ := 5 * 1 + 1 * 2

/-- The area covered by the letter O. -/
def O_area : ℕ := (3 * 3) - (1 * 1)

/-- The area covered by the letter V. -/
def V_area : ℕ := 2 * (3 * 1)

/-- The area covered by the letter E. -/
def E_area : ℕ := 3 * (1 * 3)

/-- The total area covered by the letters L, O, V, E. -/
def sum_black_area : ℕ := L_area + O_area + V_area + E_area

/-- The problem statement: Calculate the area of the white portion of the sign. -/
theorem white_area : sign_area - sum_black_area = 55 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_white_area_l1238_123850


namespace NUMINAMATH_GPT_angle_C_magnitude_area_triangle_l1238_123880

variable {a b c A B C : ℝ}

namespace triangle

-- Conditions and variable declarations
axiom condition1 : 2 * b * Real.cos C = a * Real.cos C + c * Real.cos A
axiom triangle_sides : a = 3 ∧ b = 2 ∧ c = Real.sqrt 7

-- Prove the magnitude of angle C is π/3
theorem angle_C_magnitude : C = Real.pi / 3 :=
by sorry

-- Prove that given b = 2 and c = sqrt(7), a = 3 and the area of triangle ABC is 3*sqrt(3)/2
theorem area_triangle :
  (b = 2 ∧ c = Real.sqrt 7 ∧ C = Real.pi / 3) → 
  (a = 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2)) :=
by sorry

end triangle

end NUMINAMATH_GPT_angle_C_magnitude_area_triangle_l1238_123880


namespace NUMINAMATH_GPT_team_problem_solved_probability_l1238_123844

-- Defining the probabilities
def P_A : ℚ := 1 / 5
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Defining the probability that the problem is solved
def P_s : ℚ := 3 / 5

-- Lean 4 statement to prove that the calculated probability matches the expected solution
theorem team_problem_solved_probability :
  1 - (1 - P_A) * (1 - P_B) * (1 - P_C) = P_s :=
by
  sorry

end NUMINAMATH_GPT_team_problem_solved_probability_l1238_123844


namespace NUMINAMATH_GPT_initial_music_files_eq_sixteen_l1238_123897

theorem initial_music_files_eq_sixteen (M : ℕ) :
  (M + 48 - 30 = 34) → (M = 16) :=
by
  sorry

end NUMINAMATH_GPT_initial_music_files_eq_sixteen_l1238_123897


namespace NUMINAMATH_GPT_count_paths_l1238_123827

theorem count_paths (m n : ℕ) : (n + m).choose m = (n + m).choose n :=
by
  sorry

end NUMINAMATH_GPT_count_paths_l1238_123827


namespace NUMINAMATH_GPT_midpoint_s2_l1238_123865

structure Point where
  x : ℤ
  y : ℤ

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def translate (p : Point) (dx dy : ℤ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem midpoint_s2 :
  let s1_p1 := ⟨6, -2⟩
  let s1_p2 := ⟨-4, 6⟩
  let s1_mid := midpoint s1_p1 s1_p2
  let s2_mid_translated := translate s1_mid (-3) (-4)
  s2_mid_translated = ⟨-2, -2⟩ := 
by
  sorry

end NUMINAMATH_GPT_midpoint_s2_l1238_123865


namespace NUMINAMATH_GPT_converse_and_inverse_false_l1238_123869

variable (Polygon : Type)
variable (RegularHexagon : Polygon → Prop)
variable (AllSidesEqual : Polygon → Prop)

theorem converse_and_inverse_false (p : Polygon → Prop) (q : Polygon → Prop)
  (h : ∀ x, RegularHexagon x → AllSidesEqual x) :
  ¬ (∀ x, AllSidesEqual x → RegularHexagon x) ∧ ¬ (∀ x, ¬ RegularHexagon x → ¬ AllSidesEqual x) :=
by
  sorry

end NUMINAMATH_GPT_converse_and_inverse_false_l1238_123869


namespace NUMINAMATH_GPT_complex_number_property_l1238_123848

noncomputable def imaginary_unit : Complex := Complex.I

theorem complex_number_property (n : ℕ) (hn : 4^n = 256) : (1 + imaginary_unit)^n = -4 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_property_l1238_123848


namespace NUMINAMATH_GPT_find_son_l1238_123877

variable (SonAge ManAge : ℕ)

def age_relationship (SonAge ManAge : ℕ) : Prop :=
  ManAge = SonAge + 20 ∧ ManAge + 2 = 2 * (SonAge + 2)

theorem find_son's_age (S M : ℕ) (h : age_relationship S M) : S = 18 :=
by
  unfold age_relationship at h
  obtain ⟨h1, h2⟩ := h
  sorry

end NUMINAMATH_GPT_find_son_l1238_123877


namespace NUMINAMATH_GPT_wheel_moves_distance_in_one_hour_l1238_123886

-- Definition of the given conditions
def rotations_per_minute : ℕ := 10
def distance_per_rotation : ℕ := 20
def minutes_per_hour : ℕ := 60

-- Theorem statement to prove the wheel moves 12000 cm in one hour
theorem wheel_moves_distance_in_one_hour : 
  rotations_per_minute * minutes_per_hour * distance_per_rotation = 12000 := 
by
  sorry

end NUMINAMATH_GPT_wheel_moves_distance_in_one_hour_l1238_123886


namespace NUMINAMATH_GPT_infinite_integer_solutions_iff_l1238_123839

theorem infinite_integer_solutions_iff
  (a b c d : ℤ) :
  (∃ inf_int_sol : (ℤ → ℤ) → Prop, ∀ (f : (ℤ → ℤ)), inf_int_sol f) ↔ (a^2 - 4*b = c^2 - 4*d) :=
by
  sorry

end NUMINAMATH_GPT_infinite_integer_solutions_iff_l1238_123839


namespace NUMINAMATH_GPT_lions_at_sanctuary_l1238_123835

variable (L C : ℕ)

noncomputable def is_solution :=
  C = 1 / 2 * (L + 14) ∧
  L + 14 + C = 39 ∧
  L = 12

theorem lions_at_sanctuary : is_solution L C :=
sorry

end NUMINAMATH_GPT_lions_at_sanctuary_l1238_123835


namespace NUMINAMATH_GPT_total_profit_l1238_123882

variable (A_s B_s C_s : ℝ)
variable (A_p : ℝ := 14700)
variable (P : ℝ)

theorem total_profit
  (h1 : A_s + B_s + C_s = 50000)
  (h2 : A_s = B_s + 4000)
  (h3 : B_s = C_s + 5000)
  (h4 : A_p = 14700) :
  P = 35000 :=
sorry

end NUMINAMATH_GPT_total_profit_l1238_123882


namespace NUMINAMATH_GPT_initial_bottle_caps_l1238_123885

theorem initial_bottle_caps (X : ℕ) (h1 : X - 60 + 58 = 67) : X = 69 := by
  sorry

end NUMINAMATH_GPT_initial_bottle_caps_l1238_123885


namespace NUMINAMATH_GPT_tim_prank_combinations_l1238_123890

def number_of_combinations (monday_choices : ℕ) (tuesday_choices : ℕ) (wednesday_choices : ℕ) (thursday_choices : ℕ) (friday_choices : ℕ) : ℕ :=
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations 2 3 0 6 1 = 0 :=
by
  -- Calculation yields 2 * 3 * 0 * 6 * 1 = 0
  sorry

end NUMINAMATH_GPT_tim_prank_combinations_l1238_123890


namespace NUMINAMATH_GPT_divisibility_by_65_product_of_four_natural_numbers_l1238_123829

def N : ℕ := 2^2022 + 1

theorem divisibility_by_65 : ∃ k : ℕ, N = 65 * k := by
  sorry

theorem product_of_four_natural_numbers :
  ∃ a b c d : ℕ, 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ N = a * b * c * d :=
  by sorry

end NUMINAMATH_GPT_divisibility_by_65_product_of_four_natural_numbers_l1238_123829


namespace NUMINAMATH_GPT_induction_inequality_l1238_123871

variable (n : ℕ) (h₁ : n ∈ Set.Icc 2 (2^n - 1))

theorem induction_inequality : 1 + 1/2 + 1/3 < 2 := 
  sorry

end NUMINAMATH_GPT_induction_inequality_l1238_123871


namespace NUMINAMATH_GPT_solution_set_l1238_123856

theorem solution_set (x : ℝ) : 
  1 < |x + 2| ∧ |x + 2| < 5 ↔ 
  (-7 < x ∧ x < -3) ∨ (-1 < x ∧ x < 3) := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_l1238_123856


namespace NUMINAMATH_GPT_quadratic_sequence_exists_l1238_123830

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), 
  a 0 = b ∧ 
  a n = c ∧ 
  ∀ i, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i^2 :=
sorry

end NUMINAMATH_GPT_quadratic_sequence_exists_l1238_123830


namespace NUMINAMATH_GPT_term_2012_of_T_is_2057_l1238_123874

-- Define a function that checks if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the sequence T as all natural numbers which are not perfect squares
def T (n : ℕ) : ℕ :=
  (n + Nat.sqrt (4 * n)) 

-- The theorem to state the mathematical proof problem
theorem term_2012_of_T_is_2057 :
  T 2012 = 2057 :=
sorry

end NUMINAMATH_GPT_term_2012_of_T_is_2057_l1238_123874


namespace NUMINAMATH_GPT_find_a_l1238_123866

noncomputable def tangent_to_circle_and_parallel (a : ℝ) : Prop := 
  let P := (2, 2)
  let circle_center := (1, 0)
  let on_circle := (P.1 - 1)^2 + P.2^2 = 5
  let perpendicular_slope := (P.2 - circle_center.2) / (P.1 - circle_center.1) * (1 / a) = -1
  on_circle ∧ perpendicular_slope

theorem find_a (a : ℝ) : tangent_to_circle_and_parallel a ↔ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1238_123866


namespace NUMINAMATH_GPT_min_moves_seven_chests_l1238_123810

/-
Problem:
Seven chests are placed in a circle, each containing a certain number of coins: [20, 15, 5, 6, 10, 17, 18].
Prove that the minimum number of moves required to equalize the number of coins in all chests is 22.
-/

def min_moves_to_equalize_coins (coins: List ℕ) : ℕ :=
  -- Function that would calculate the minimum number of moves
  sorry

theorem min_moves_seven_chests :
  min_moves_to_equalize_coins [20, 15, 5, 6, 10, 17, 18] = 22 :=
sorry

end NUMINAMATH_GPT_min_moves_seven_chests_l1238_123810


namespace NUMINAMATH_GPT_sum_of_first_3n_terms_l1238_123864

theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_3n_terms_l1238_123864


namespace NUMINAMATH_GPT_largest_base5_three_digit_to_base10_l1238_123841

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end NUMINAMATH_GPT_largest_base5_three_digit_to_base10_l1238_123841


namespace NUMINAMATH_GPT_least_integer_greater_than_sqrt_450_l1238_123818

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end NUMINAMATH_GPT_least_integer_greater_than_sqrt_450_l1238_123818


namespace NUMINAMATH_GPT_sum_of_roots_l1238_123875

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1238_123875


namespace NUMINAMATH_GPT_commute_time_abs_diff_l1238_123813

theorem commute_time_abs_diff (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  |x - y| = 4 := by
  sorry

end NUMINAMATH_GPT_commute_time_abs_diff_l1238_123813


namespace NUMINAMATH_GPT_gcd_at_most_3_digits_l1238_123846

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end NUMINAMATH_GPT_gcd_at_most_3_digits_l1238_123846


namespace NUMINAMATH_GPT_increasing_iff_a_le_0_l1238_123821

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - a * x + 1

theorem increasing_iff_a_le_0 : (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_increasing_iff_a_le_0_l1238_123821


namespace NUMINAMATH_GPT_inequality_xyz_geq_3_l1238_123828

theorem inequality_xyz_geq_3
  (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  (2 * x^2 - x + y + z) / (x + y^2 + z^2) +
  (2 * y^2 + x - y + z) / (x^2 + y + z^2) +
  (2 * z^2 + x + y - z) / (x^2 + y^2 + z) ≥ 3 := 
sorry

end NUMINAMATH_GPT_inequality_xyz_geq_3_l1238_123828


namespace NUMINAMATH_GPT_find_a1_l1238_123824

-- Definitions of the conditions
def Sn (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence
def a₁ : ℤ := sorry          -- First term of the sequence

axiom S_2016_eq_2016 : Sn 2016 = 2016
axiom diff_seq_eq_2000 : (Sn 2016 / 2016) - (Sn 16 / 16) = 2000

-- Proof statement
theorem find_a1 : a₁ = -2014 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_find_a1_l1238_123824


namespace NUMINAMATH_GPT_smallest_n_divides_24_and_1024_l1238_123832

theorem smallest_n_divides_24_and_1024 : ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ (∀ m : ℕ, (m > 0 ∧ (24 ∣ m^2) ∧ (1024 ∣ m^3)) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divides_24_and_1024_l1238_123832


namespace NUMINAMATH_GPT_smallest_integer_condition_l1238_123859

theorem smallest_integer_condition (x : ℝ) (hz : 9 = 9) (hineq : 27^9 > x^24) : x < 27 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_smallest_integer_condition_l1238_123859


namespace NUMINAMATH_GPT_space_shuttle_speed_l1238_123800

theorem space_shuttle_speed :
  ∀ (speed_kph : ℕ) (minutes_per_hour seconds_per_minute : ℕ),
    speed_kph = 32400 →
    minutes_per_hour = 60 →
    seconds_per_minute = 60 →
    (speed_kph / (minutes_per_hour * seconds_per_minute)) = 9 :=
by
  intros speed_kph minutes_per_hour seconds_per_minute
  intro h_speed
  intro h_minutes
  intro h_seconds
  sorry

end NUMINAMATH_GPT_space_shuttle_speed_l1238_123800


namespace NUMINAMATH_GPT_total_marbles_l1238_123857

theorem total_marbles (r b g : ℕ) (h_ratio : r = 1 ∧ b = 5 ∧ g = 3) (h_green : g = 27) :
  (r + b + g) * 3 = 81 :=
  sorry

end NUMINAMATH_GPT_total_marbles_l1238_123857


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l1238_123858

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l1238_123858


namespace NUMINAMATH_GPT_prove_a4_plus_1_div_a4_l1238_123817

theorem prove_a4_plus_1_div_a4 (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/(a^4) = 7 :=
by
  sorry

end NUMINAMATH_GPT_prove_a4_plus_1_div_a4_l1238_123817


namespace NUMINAMATH_GPT_find_number_l1238_123881

theorem find_number (x : ℕ) (h : x - 263 + 419 = 725) : x = 569 :=
sorry

end NUMINAMATH_GPT_find_number_l1238_123881


namespace NUMINAMATH_GPT_height_min_surface_area_l1238_123811

def height_of_box (x : ℝ) : ℝ := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem height_min_surface_area :
  ∀ x : ℝ, surface_area x ≥ 150 → x ≥ 5 → height_of_box x = 9 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_height_min_surface_area_l1238_123811


namespace NUMINAMATH_GPT_carol_initial_peanuts_l1238_123826

theorem carol_initial_peanuts (p_initial p_additional p_total : Nat) (h1 : p_additional = 5) (h2 : p_total = 7) (h3 : p_initial + p_additional = p_total) : p_initial = 2 :=
by
  sorry

end NUMINAMATH_GPT_carol_initial_peanuts_l1238_123826


namespace NUMINAMATH_GPT_real_solution_l1238_123888

theorem real_solution (x : ℝ) (h : x ≠ 3) :
  (x * (x + 2)) / ((x - 3)^2) ≥ 8 ↔ (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 48) :=
by
  sorry

end NUMINAMATH_GPT_real_solution_l1238_123888


namespace NUMINAMATH_GPT_sequence_value_l1238_123887

theorem sequence_value (a : ℕ → ℕ) (h₁ : ∀ n, a (2 * n) = a (2 * n - 1) + (-1 : ℤ)^n) 
                        (h₂ : ∀ n, a (2 * n + 1) = a (2 * n) + n)
                        (h₃ : a 1 = 1) : a 20 = 46 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_value_l1238_123887


namespace NUMINAMATH_GPT_percentage_music_l1238_123843

variable (students_total : ℕ)
variable (percent_dance percent_art percent_drama percent_sports percent_photography percent_music : ℝ)

-- Define the problem conditions
def school_conditions : Prop :=
  students_total = 3000 ∧
  percent_dance = 0.125 ∧
  percent_art = 0.22 ∧
  percent_drama = 0.135 ∧
  percent_sports = 0.15 ∧
  percent_photography = 0.08 ∧
  percent_music = 1 - (percent_dance + percent_art + percent_drama + percent_sports + percent_photography)

-- Define the proof statement
theorem percentage_music : school_conditions students_total percent_dance percent_art percent_drama percent_sports percent_photography percent_music → percent_music = 0.29 :=
by
  intros h
  rw [school_conditions] at h
  sorry

end NUMINAMATH_GPT_percentage_music_l1238_123843


namespace NUMINAMATH_GPT_extreme_points_inequality_l1238_123815

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 2) * x^2 + m * Real.log (1 - x)

theorem extreme_points_inequality (m x1 x2 : ℝ) 
  (h_m1 : 0 < m) (h_m2 : m < 1 / 4)
  (h_x1 : 0 < x1) (h_x2: x1 < 1 / 2)
  (h_x3: x2 > 1 / 2) (h_x4: x2 < 1)
  (h_x5 : x1 < x2)
  (h_sum : x1 + x2 = 1)
  (h_prod : x1 * x2 = m)
  : (1 / 4) - (1 / 2) * Real.log 2 < (f x1 m) / x2 ∧ (f x1 m) / x2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_extreme_points_inequality_l1238_123815


namespace NUMINAMATH_GPT_john_twice_as_old_in_x_years_l1238_123855

def frank_is_younger (john_age frank_age : ℕ) : Prop :=
  frank_age = john_age - 15

def frank_future_age (frank_age : ℕ) : ℕ :=
  frank_age + 4

def john_future_age (john_age : ℕ) : ℕ :=
  john_age + 4

theorem john_twice_as_old_in_x_years (john_age frank_age x : ℕ) 
  (h1 : frank_is_younger john_age frank_age)
  (h2 : frank_future_age frank_age = 16)
  (h3 : john_age = frank_age + 15) :
  (john_age + x) = 2 * (frank_age + x) → x = 3 :=
by 
  -- Skip the proof part
  sorry

end NUMINAMATH_GPT_john_twice_as_old_in_x_years_l1238_123855


namespace NUMINAMATH_GPT_negation_statement_l1238_123860

variables (Students : Type) (LeftHanded InChessClub : Students → Prop)

theorem negation_statement :
  (¬ ∃ x, LeftHanded x ∧ InChessClub x) ↔ (∃ x, LeftHanded x ∧ InChessClub x) :=
by
  sorry

end NUMINAMATH_GPT_negation_statement_l1238_123860


namespace NUMINAMATH_GPT_animals_per_aquarium_l1238_123863

theorem animals_per_aquarium
  (saltwater_aquariums : ℕ)
  (saltwater_animals : ℕ)
  (h1 : saltwater_aquariums = 56)
  (h2 : saltwater_animals = 2184)
  : saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end NUMINAMATH_GPT_animals_per_aquarium_l1238_123863


namespace NUMINAMATH_GPT_investment_in_scheme_B_l1238_123891

theorem investment_in_scheme_B 
    (yieldA : ℝ) (yieldB : ℝ) (investmentA : ℝ) (difference : ℝ) (totalA : ℝ) (totalB : ℝ):
    yieldA = 0.30 → yieldB = 0.50 → investmentA = 300 → difference = 90 
    → totalA = investmentA + (yieldA * investmentA) 
    → totalB = (1 + yieldB) * totalB 
    → totalA = totalB + difference 
    → totalB = 200 :=
by sorry

end NUMINAMATH_GPT_investment_in_scheme_B_l1238_123891


namespace NUMINAMATH_GPT_convert_kmph_to_mps_l1238_123851

theorem convert_kmph_to_mps (speed_kmph : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) : 
  speed_kmph = 56 → km_to_m = 1000 → hr_to_s = 3600 → 
  (speed_kmph * (km_to_m / hr_to_s) : ℝ) = 15.56 :=
by
  intros
  sorry

end NUMINAMATH_GPT_convert_kmph_to_mps_l1238_123851


namespace NUMINAMATH_GPT_probability_of_selected_member_l1238_123852

section Probability

variables {N : ℕ} -- Total number of members in the group

-- Conditions
-- Probabilities of selecting individuals by gender
def P_woman : ℝ := 0.70
def P_man : ℝ := 0.20
def P_non_binary : ℝ := 0.10

-- Conditional probabilities of occupations given gender
def P_engineer_given_woman : ℝ := 0.20
def P_doctor_given_man : ℝ := 0.20
def P_translator_given_non_binary : ℝ := 0.20

-- The main proof statement
theorem probability_of_selected_member :
  (P_woman * P_engineer_given_woman) + (P_man * P_doctor_given_man) + (P_non_binary * P_translator_given_non_binary) = 0.20 :=
by
  sorry

end Probability

end NUMINAMATH_GPT_probability_of_selected_member_l1238_123852


namespace NUMINAMATH_GPT_binomial_expansion_coeff_x10_sub_x5_eq_251_l1238_123831

open BigOperators Polynomial

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_coeff_x10_sub_x5_eq_251 :
  ∀ (a : Fin 11 → ℤ), (fun (x : ℤ) =>
    x^10 - x^5 - (a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
                  a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + 
                  a 5 * (x - 1)^5 + a 6 * (x - 1)^6 + 
                  a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
                  a 9 * (x - 1)^9 + a 10 * (x - 1)^10)) = 0 → 
  a 5 = 251 := 
by 
  sorry

end NUMINAMATH_GPT_binomial_expansion_coeff_x10_sub_x5_eq_251_l1238_123831
