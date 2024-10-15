import Mathlib

namespace NUMINAMATH_GPT_ott_fraction_of_total_money_l2328_232821

-- Definitions for the conditions
def Moe_initial_money (x : ℕ) : ℕ := 3 * x
def Loki_initial_money (x : ℕ) : ℕ := 5 * x
def Nick_initial_money (x : ℕ) : ℕ := 4 * x
def Total_initial_money (x : ℕ) : ℕ := Moe_initial_money x + Loki_initial_money x + Nick_initial_money x
def Ott_received_money (x : ℕ) : ℕ := 3 * x

-- Making the statement we want to prove
theorem ott_fraction_of_total_money (x : ℕ) : 
  (Ott_received_money x) / (Total_initial_money x) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_ott_fraction_of_total_money_l2328_232821


namespace NUMINAMATH_GPT_number_of_green_hats_l2328_232804

variables (B G : ℕ)

-- Given conditions as definitions
def totalHats : Prop := B + G = 85
def totalCost : Prop := 6 * B + 7 * G = 530

-- The statement we need to prove
theorem number_of_green_hats (h1 : totalHats B G) (h2 : totalCost B G) : G = 20 :=
sorry

end NUMINAMATH_GPT_number_of_green_hats_l2328_232804


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l2328_232879

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l2328_232879


namespace NUMINAMATH_GPT_possible_values_of_quadratic_expression_l2328_232840

theorem possible_values_of_quadratic_expression (x : ℝ) (h : 2 < x ∧ x < 3) : 
  20 < x^2 + 5 * x + 6 ∧ x^2 + 5 * x + 6 < 30 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_quadratic_expression_l2328_232840


namespace NUMINAMATH_GPT_inequality_transformation_l2328_232812

theorem inequality_transformation (x y : ℝ) (h : 2 * x - 5 < 2 * y - 5) : x < y := 
by 
  sorry

end NUMINAMATH_GPT_inequality_transformation_l2328_232812


namespace NUMINAMATH_GPT_alpha_in_first_quadrant_l2328_232881

theorem alpha_in_first_quadrant (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 2) < 0) 
  (h2 : Real.tan (Real.pi + α) > 0) : 
  (0 < α ∧ α < Real.pi / 2) ∨ (2 * Real.pi < α ∧ α < 5 * Real.pi / 2) := 
by
  sorry

end NUMINAMATH_GPT_alpha_in_first_quadrant_l2328_232881


namespace NUMINAMATH_GPT_problem_solution_l2328_232852

theorem problem_solution (x : ℝ) (h : 1 - 9 / x + 20 / x^2 = 0) : (2 / x = 1 / 2 ∨ 2 / x = 2 / 5) := 
  sorry

end NUMINAMATH_GPT_problem_solution_l2328_232852


namespace NUMINAMATH_GPT_factor_expression_l2328_232827

variable {R : Type*} [CommRing R]

theorem factor_expression (a b c : R) :
    a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
    (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) :=
sorry

end NUMINAMATH_GPT_factor_expression_l2328_232827


namespace NUMINAMATH_GPT_convex_polyhedron_triangular_face_or_three_edges_vertex_l2328_232808

theorem convex_polyhedron_triangular_face_or_three_edges_vertex
  (M N K : ℕ) 
  (euler_formula : N - M + K = 2) :
  ∃ (f : ℕ), (f ≤ N ∧ f = 3) ∨ ∃ (v : ℕ), (v ≤ K ∧ v = 3) := 
sorry

end NUMINAMATH_GPT_convex_polyhedron_triangular_face_or_three_edges_vertex_l2328_232808


namespace NUMINAMATH_GPT_find_side_b_l2328_232810

-- Given the side and angle conditions in the triangle
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ) 

-- Conditions provided in the problem
axiom side_a (h : a = 1) : True
axiom angle_B (h : B = Real.pi / 4) : True  -- 45 degrees in radians
axiom area_triangle (h : S = 2) : True

-- Final proof statement
theorem find_side_b (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : 
  b = 5 := sorry

end NUMINAMATH_GPT_find_side_b_l2328_232810


namespace NUMINAMATH_GPT_minimum_disks_needed_l2328_232818

theorem minimum_disks_needed :
  ∀ (n_files : ℕ) (disk_space : ℝ) (mb_files_1 : ℕ) (size_file_1 : ℝ) (mb_files_2 : ℕ) (size_file_2 : ℝ) (remaining_files : ℕ) (size_remaining_files : ℝ),
    n_files = 30 →
    disk_space = 1.5 →
    mb_files_1 = 4 →
    size_file_1 = 1.0 →
    mb_files_2 = 10 →
    size_file_2 = 0.6 →
    remaining_files = 16 →
    size_remaining_files = 0.5 →
    ∃ (min_disks : ℕ), min_disks = 13 :=
by
  sorry

end NUMINAMATH_GPT_minimum_disks_needed_l2328_232818


namespace NUMINAMATH_GPT_expand_polynomial_l2328_232823

theorem expand_polynomial :
  (3 * x ^ 2 - 4 * x + 3) * (-2 * x ^ 2 + 3 * x - 4) = -6 * x ^ 4 + 17 * x ^ 3 - 30 * x ^ 2 + 25 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l2328_232823


namespace NUMINAMATH_GPT_effective_annual_rate_correct_l2328_232845

noncomputable def nominal_annual_interest_rate : ℝ := 0.10
noncomputable def compounding_periods_per_year : ℕ := 2
noncomputable def effective_annual_rate : ℝ := (1 + nominal_annual_interest_rate / compounding_periods_per_year) ^ compounding_periods_per_year - 1

theorem effective_annual_rate_correct :
  effective_annual_rate = 0.1025 :=
by
  sorry

end NUMINAMATH_GPT_effective_annual_rate_correct_l2328_232845


namespace NUMINAMATH_GPT_line_intersects_circle_l2328_232872

variable (x₀ y₀ r : Real)

theorem line_intersects_circle (h : x₀^2 + y₀^2 > r^2) : 
  ∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = r^2) ∧ (x₀ * p.1 + y₀ * p.2 = r^2) := by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l2328_232872


namespace NUMINAMATH_GPT_dave_apps_left_l2328_232850

theorem dave_apps_left (initial_apps deleted_apps remaining_apps : ℕ)
  (h_initial : initial_apps = 23)
  (h_deleted : deleted_apps = 18)
  (h_calculation : remaining_apps = initial_apps - deleted_apps) :
  remaining_apps = 5 := 
by 
  sorry

end NUMINAMATH_GPT_dave_apps_left_l2328_232850


namespace NUMINAMATH_GPT_complex_exp_neg_ipi_on_real_axis_l2328_232856

theorem complex_exp_neg_ipi_on_real_axis :
  (Complex.exp (-Real.pi * Complex.I)).im = 0 :=
by 
  sorry

end NUMINAMATH_GPT_complex_exp_neg_ipi_on_real_axis_l2328_232856


namespace NUMINAMATH_GPT_max_tickets_l2328_232837

/-- Given the cost of each ticket and the total amount of money available, 
    prove that the maximum number of tickets that can be purchased is 8. -/
theorem max_tickets (ticket_cost : ℝ) (total_amount : ℝ) (h1 : ticket_cost = 18.75) (h2 : total_amount = 150) :
  (∃ n : ℕ, ticket_cost * n ≤ total_amount ∧ ∀ m : ℕ, ticket_cost * m ≤ total_amount → m ≤ n) ∧
  ∃ n : ℤ, (n : ℤ) = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_tickets_l2328_232837


namespace NUMINAMATH_GPT_solve_total_rainfall_l2328_232858

def rainfall_2010 : ℝ := 50.0
def increase_2011 : ℝ := 3.0
def increase_2012 : ℝ := 4.0

def monthly_rainfall_2011 : ℝ := rainfall_2010 + increase_2011
def monthly_rainfall_2012 : ℝ := monthly_rainfall_2011 + increase_2012

def total_rainfall_2011 : ℝ := monthly_rainfall_2011 * 12
def total_rainfall_2012 : ℝ := monthly_rainfall_2012 * 12

def total_rainfall_2011_2012 : ℝ := total_rainfall_2011 + total_rainfall_2012

theorem solve_total_rainfall :
  total_rainfall_2011_2012 = 1320.0 :=
sorry

end NUMINAMATH_GPT_solve_total_rainfall_l2328_232858


namespace NUMINAMATH_GPT_number_of_months_in_martian_calendar_l2328_232829

theorem number_of_months_in_martian_calendar
  (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) 
  (h2 : x + y = 74) :
  x + y = 74 := 
by
  sorry

end NUMINAMATH_GPT_number_of_months_in_martian_calendar_l2328_232829


namespace NUMINAMATH_GPT_derivative_y_l2328_232891

noncomputable def y (x : ℝ) : ℝ := 
  (Real.sqrt (49 * x^2 + 1) * Real.arctan (7 * x)) - 
  Real.log (7 * x + Real.sqrt (49 * x^2 + 1))

theorem derivative_y (x : ℝ) : 
  deriv y x = (7 * Real.arctan (7 * x)) / (2 * Real.sqrt (49 * x^2 + 1)) := by
  sorry

end NUMINAMATH_GPT_derivative_y_l2328_232891


namespace NUMINAMATH_GPT_complex_purely_imaginary_l2328_232811

theorem complex_purely_imaginary (a : ℂ) (h1 : a^2 - 3 * a + 2 = 0) (h2 : a - 1 ≠ 0) : a = 2 :=
sorry

end NUMINAMATH_GPT_complex_purely_imaginary_l2328_232811


namespace NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l2328_232899

noncomputable def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by 
  sorry

end NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l2328_232899


namespace NUMINAMATH_GPT_reciprocal_of_sum_of_repeating_decimals_l2328_232863

theorem reciprocal_of_sum_of_repeating_decimals :
  let x := 5 / 33
  let y := 1 / 3
  1 / (x + y) = 33 / 16 :=
by
  -- The following is the proof, but it will be skipped for this exercise.
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_of_repeating_decimals_l2328_232863


namespace NUMINAMATH_GPT_union_of_A_and_B_l2328_232806

def setA : Set ℝ := {x | 2 * x - 1 > 0}
def setB : Set ℝ := {x | abs x < 1}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | x > -1} := 
by {
  sorry
}

end NUMINAMATH_GPT_union_of_A_and_B_l2328_232806


namespace NUMINAMATH_GPT_quadratic_roots_property_l2328_232859

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_property_l2328_232859


namespace NUMINAMATH_GPT_same_volume_increase_rate_l2328_232833

def initial_radius := 10
def initial_height := 5 

def volume_increase_rate_new_radius (x : ℝ) :=
  let r' := initial_radius + 2 * x
  (r' ^ 2) * initial_height  - (initial_radius ^ 2) * initial_height

def volume_increase_rate_new_height (x : ℝ) :=
  let h' := initial_height + 3 * x
  (initial_radius ^ 2) * h' - (initial_radius ^ 2) * initial_height

theorem same_volume_increase_rate (x : ℝ) : volume_increase_rate_new_radius x = volume_increase_rate_new_height x → x = 5 := 
  by sorry

end NUMINAMATH_GPT_same_volume_increase_rate_l2328_232833


namespace NUMINAMATH_GPT_age_of_b_l2328_232877

variables {a b : ℕ}

theorem age_of_b (h₁ : a + 10 = 2 * (b - 10)) (h₂ : a = b + 11) : b = 41 :=
sorry

end NUMINAMATH_GPT_age_of_b_l2328_232877


namespace NUMINAMATH_GPT_blue_balls_taken_out_l2328_232886

theorem blue_balls_taken_out
  (x : ℕ) 
  (balls_initial : ℕ := 18)
  (blue_initial : ℕ := 6)
  (prob_blue : ℚ := 1/5)
  (total : ℕ := balls_initial - x)
  (blue_current : ℕ := blue_initial - x) :
  (↑blue_current / ↑total = prob_blue) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_blue_balls_taken_out_l2328_232886


namespace NUMINAMATH_GPT_find_x_l2328_232855

theorem find_x :
  ∃ x : ℝ, 8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 ∧ x = 1.464 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2328_232855


namespace NUMINAMATH_GPT_find_hall_length_l2328_232893

variable (W H total_cost cost_per_sqm : ℕ)

theorem find_hall_length
  (hW : W = 15)
  (hH : H = 5)
  (h_total_cost : total_cost = 57000)
  (h_cost_per_sqm : cost_per_sqm = 60)
  : (32 * W) + (2 * (H * 32)) + (2 * (H * W)) = total_cost / cost_per_sqm :=
by
  sorry

end NUMINAMATH_GPT_find_hall_length_l2328_232893


namespace NUMINAMATH_GPT_find_third_angle_l2328_232870

-- Definitions from the problem conditions
def triangle_angle_sum (a b c : ℝ) : Prop := a + b + c = 180

-- Statement of the proof problem
theorem find_third_angle (a b x : ℝ) (h1 : a = 50) (h2 : b = 45) (h3 : triangle_angle_sum a b x) : x = 85 := sorry

end NUMINAMATH_GPT_find_third_angle_l2328_232870


namespace NUMINAMATH_GPT_find_multiple_of_smaller_integer_l2328_232836

theorem find_multiple_of_smaller_integer (L S k : ℕ) 
  (h1 : S = 10) 
  (h2 : L + S = 30) 
  (h3 : 2 * L = k * S - 10) 
  : k = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_smaller_integer_l2328_232836


namespace NUMINAMATH_GPT_books_on_shelf_after_removal_l2328_232844

theorem books_on_shelf_after_removal :
  let initial_books : ℝ := 38.0
  let books_removed : ℝ := 10.0
  initial_books - books_removed = 28.0 :=
by 
  sorry

end NUMINAMATH_GPT_books_on_shelf_after_removal_l2328_232844


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_6_and_15_gt_40_l2328_232809

-- Define the LCM function to compute the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Define the statement of the proof problem
theorem smallest_positive_multiple_of_6_and_15_gt_40 : 
  ∃ a : ℕ, (a % 6 = 0) ∧ (a % 15 = 0) ∧ (a > 40) ∧ (∀ b : ℕ, (b % 6 = 0) ∧ (b % 15 = 0) ∧ (b > 40) → a ≤ b) :=
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_6_and_15_gt_40_l2328_232809


namespace NUMINAMATH_GPT_shyam_weight_increase_l2328_232897

theorem shyam_weight_increase (total_weight_after_increase : ℝ) (ram_initial_weight_ratio : ℝ) 
    (shyam_initial_weight_ratio : ℝ) (ram_increase_percent : ℝ) (total_increase_percent : ℝ) 
    (ram_total_weight_ratio : ram_initial_weight_ratio = 6) (shyam_initial_total_weight_ratio : shyam_initial_weight_ratio = 5) 
    (total_weight_after_increase_eq : total_weight_after_increase = 82.8) 
    (ram_increase_percent_eq : ram_increase_percent = 0.10) 
    (total_increase_percent_eq : total_increase_percent = 0.15) : 
  shyam_increase_percent = (21 : ℝ) :=
sorry

end NUMINAMATH_GPT_shyam_weight_increase_l2328_232897


namespace NUMINAMATH_GPT_thirty_percent_less_than_80_equals_one_fourth_more_l2328_232830

theorem thirty_percent_less_than_80_equals_one_fourth_more (n : ℝ) :
  80 * 0.30 = 24 → 80 - 24 = 56 → n + n / 4 = 56 → n = 224 / 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_thirty_percent_less_than_80_equals_one_fourth_more_l2328_232830


namespace NUMINAMATH_GPT_find_a_l2328_232898

def tangent_condition (x a : ℝ) : Prop := 2 * x - (Real.log x + a) + 1 = 0

def slope_condition (x : ℝ) : Prop := 2 = 1 / x

theorem find_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ tangent_condition x a ∧ slope_condition x) →
  a = -2 * Real.log 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l2328_232898


namespace NUMINAMATH_GPT_distance_home_to_school_l2328_232846

theorem distance_home_to_school
  (T T' : ℝ)
  (D : ℝ)
  (h1 : D = 6 * T)
  (h2 : D = 12 * T')
  (h3 : T - T' = 0.25) :
  D = 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_distance_home_to_school_l2328_232846


namespace NUMINAMATH_GPT_log_base_16_of_4_eq_half_l2328_232822

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end NUMINAMATH_GPT_log_base_16_of_4_eq_half_l2328_232822


namespace NUMINAMATH_GPT_percentage_difference_l2328_232801

noncomputable def P : ℝ := 40
variables {w x y z : ℝ}
variables (H1 : w = x * (1 - P / 100))
variables (H2 : x = 0.6 * y)
variables (H3 : z = 0.54 * y)
variables (H4 : z = 1.5 * w)

-- Goal
theorem percentage_difference : P = 40 :=
by sorry -- Proof omitted

end NUMINAMATH_GPT_percentage_difference_l2328_232801


namespace NUMINAMATH_GPT_fractions_equivalent_iff_x_eq_zero_l2328_232871

theorem fractions_equivalent_iff_x_eq_zero (x : ℝ) (h : (x + 1) / (x + 3) = 1 / 3) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_fractions_equivalent_iff_x_eq_zero_l2328_232871


namespace NUMINAMATH_GPT_sin_squared_minus_cos_squared_value_l2328_232896

noncomputable def sin_squared_minus_cos_squared : Real :=
  (Real.sin (Real.pi / 12))^2 - (Real.cos (Real.pi / 12))^2

theorem sin_squared_minus_cos_squared_value :
  sin_squared_minus_cos_squared = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_squared_minus_cos_squared_value_l2328_232896


namespace NUMINAMATH_GPT_baseball_games_in_season_l2328_232832

theorem baseball_games_in_season 
  (games_per_month : ℕ) 
  (months_in_season : ℕ)
  (h1 : games_per_month = 7) 
  (h2 : months_in_season = 2) :
  games_per_month * months_in_season = 14 := by
  sorry


end NUMINAMATH_GPT_baseball_games_in_season_l2328_232832


namespace NUMINAMATH_GPT_teacher_periods_per_day_l2328_232876

noncomputable def periods_per_day (days_per_month : ℕ) (months : ℕ) (period_rate : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_days := days_per_month * months
  let total_periods := total_earnings / period_rate
  let periods_per_day := total_periods / total_days
  periods_per_day

theorem teacher_periods_per_day :
  periods_per_day 24 6 5 3600 = 5 := by
  sorry

end NUMINAMATH_GPT_teacher_periods_per_day_l2328_232876


namespace NUMINAMATH_GPT_sallys_change_l2328_232878

-- Given conditions
def frames_bought : ℕ := 3
def cost_per_frame : ℕ := 3
def payment : ℕ := 20

-- The statement to prove
theorem sallys_change : payment - (frames_bought * cost_per_frame) = 11 := by
  sorry

end NUMINAMATH_GPT_sallys_change_l2328_232878


namespace NUMINAMATH_GPT_parry_position_probability_l2328_232887

theorem parry_position_probability :
    let total_members := 20
    let positions := ["President", "Vice President", "Secretary", "Treasurer"]
    let remaining_for_secretary := 18
    let remaining_for_treasurer := 17
    let prob_parry_secretary := (1 : ℚ) / remaining_for_secretary
    let prob_parry_treasurer_given_not_secretary := (1 : ℚ) / remaining_for_treasurer
    let overall_probability := prob_parry_secretary + prob_parry_treasurer_given_not_secretary * (remaining_for_treasurer / remaining_for_secretary)
    overall_probability = (1 : ℚ) / 9 := 
by
  sorry

end NUMINAMATH_GPT_parry_position_probability_l2328_232887


namespace NUMINAMATH_GPT_algebraic_expression_problem_l2328_232865

-- Define the conditions and the target statement to verify.
theorem algebraic_expression_problem (x : ℝ) 
  (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by 
  -- Add sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_algebraic_expression_problem_l2328_232865


namespace NUMINAMATH_GPT_tom_payment_l2328_232849

theorem tom_payment :
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  total_amount = 1190 :=
by
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  sorry

end NUMINAMATH_GPT_tom_payment_l2328_232849


namespace NUMINAMATH_GPT_triple_nested_application_l2328_232816

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2 * n + 3

theorem triple_nested_application : g (g (g 3)) = 49 := by
  sorry

end NUMINAMATH_GPT_triple_nested_application_l2328_232816


namespace NUMINAMATH_GPT_area_proof_l2328_232802

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end NUMINAMATH_GPT_area_proof_l2328_232802


namespace NUMINAMATH_GPT_min_sum_ab_max_product_ab_l2328_232815

theorem min_sum_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) : a + b ≥ 2 :=
by
  sorry

theorem max_product_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : a * b ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_ab_max_product_ab_l2328_232815


namespace NUMINAMATH_GPT_simple_interest_rate_l2328_232854

theorem simple_interest_rate (P : ℝ) (increase_time : ℝ) (increase_amount : ℝ) 
(hP : P = 2000) (h_increase_time : increase_time = 4) (h_increase_amount : increase_amount = 40) :
  ∃ R : ℝ, (2000 * R / 100 * (increase_time + 4) - 2000 * R / 100 * increase_time = increase_amount) ∧ (R = 0.5) := 
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2328_232854


namespace NUMINAMATH_GPT_solve_inequality_l2328_232875

theorem solve_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 ↔ x = -a) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2328_232875


namespace NUMINAMATH_GPT_power_of_product_l2328_232803

variable (x y: ℝ)

theorem power_of_product :
  (-2 * x * y^3)^2 = 4 * x^2 * y^6 := 
by
  sorry

end NUMINAMATH_GPT_power_of_product_l2328_232803


namespace NUMINAMATH_GPT_function_zeros_condition_l2328_232842

theorem function_zeros_condition (a : ℝ) (H : ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ 
  2 * Real.exp (2 * x1) - 2 * a * x1 + a - 2 * Real.exp 1 - 1 = 0 ∧ 
  2 * Real.exp (2 * x2) - 2 * a * x2 + a - 2 * Real.exp 1 - 1 = 0) :
  2 * Real.exp 1 - 1 < a ∧ a < 2 * Real.exp (2:ℝ) - 2 * Real.exp 1 - 1 := 
sorry

end NUMINAMATH_GPT_function_zeros_condition_l2328_232842


namespace NUMINAMATH_GPT_cubes_end_same_digits_l2328_232800

theorem cubes_end_same_digits (a b : ℕ) (h : a % 1000 = b % 1000) : (a^3) % 1000 = (b^3) % 1000 := by
  sorry

end NUMINAMATH_GPT_cubes_end_same_digits_l2328_232800


namespace NUMINAMATH_GPT_calories_peter_wants_to_eat_l2328_232885

-- Definitions for the conditions 
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def total_spent : ℕ := 4

-- Proven statement about the calories Peter wants to eat
theorem calories_peter_wants_to_eat : (total_spent / cost_per_bag) * (chips_per_bag * calories_per_chip) = 480 := by
  sorry

end NUMINAMATH_GPT_calories_peter_wants_to_eat_l2328_232885


namespace NUMINAMATH_GPT_latin_student_sophomore_probability_l2328_232826

variable (F S J SE : ℕ) -- freshmen, sophomores, juniors, seniors total
variable (FL SL JL SEL : ℕ) -- freshmen, sophomores, juniors, seniors taking latin
variable (p : ℚ) -- probability fraction
variable (m n : ℕ) -- relatively prime integers

-- Let the total number of students be 100 for simplicity in percentage calculations
-- Let us encode the given conditions
def conditions := 
  F = 40 ∧ 
  S = 30 ∧ 
  J = 20 ∧ 
  SE = 10 ∧ 
  FL = 40 ∧ 
  SL = S * 80 / 100 ∧ 
  JL = J * 50 / 100 ∧ 
  SEL = SE * 20 / 100

-- The probability calculation
def probability_sophomore (SL : ℕ) (FL SL JL SEL : ℕ) : ℚ := SL / (FL + SL + JL + SEL)

-- Target probability as a rational number
def target_probability := (6 : ℚ) / 19

theorem latin_student_sophomore_probability : 
  conditions F S J SE FL SL JL SEL → 
  probability_sophomore SL FL SL JL SEL = target_probability ∧ 
  m + n = 25 := 
by 
  sorry

end NUMINAMATH_GPT_latin_student_sophomore_probability_l2328_232826


namespace NUMINAMATH_GPT_find_ab_l2328_232864

variable (a b : ℝ)

def point_symmetric_about_line (Px Py Qx Qy : ℝ) (m n c : ℝ) : Prop :=
  ∃ xM yM : ℝ,
  xM = (Px + Qx) / 2 ∧ yM = (Py + Qy) / 2 ∧
  m * xM + n * yM = c ∧
  (Py - Qy) / (Px - Qx) * (-n/m) = -1

theorem find_ab (H : point_symmetric_about_line (a + 2) (b + 2) (b - a) (-b) 4 3 11) :
  a = 4 ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_find_ab_l2328_232864


namespace NUMINAMATH_GPT_time_to_carl_is_28_minutes_l2328_232847

variable (distance_to_julia : ℕ := 1) (time_to_julia : ℕ := 4)
variable (distance_to_carl : ℕ := 7)
variable (rate : ℕ := distance_to_julia * time_to_julia) -- Rate as product of distance and time

theorem time_to_carl_is_28_minutes : (distance_to_carl * time_to_julia) = 28 := by
  sorry

end NUMINAMATH_GPT_time_to_carl_is_28_minutes_l2328_232847


namespace NUMINAMATH_GPT_total_emails_received_l2328_232817

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_emails_received_l2328_232817


namespace NUMINAMATH_GPT_find_stock_face_value_l2328_232848

theorem find_stock_face_value
  (cost_price : ℝ) -- Definition for the cost price
  (discount_rate : ℝ) -- Definition for the discount rate
  (brokerage_rate : ℝ) -- Definition for the brokerage rate
  (h1 : cost_price = 98.2) -- Condition: The cost price is 98.2
  (h2 : discount_rate = 0.02) -- Condition: The discount rate is 2%
  (h3 : brokerage_rate = 0.002) -- Condition: The brokerage rate is 1/5% (0.002)
  : ∃ X : ℝ, 0.982 * X = cost_price ∧ X = 100 := -- Theorem statement to prove
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_stock_face_value_l2328_232848


namespace NUMINAMATH_GPT_three_digit_sum_26_l2328_232867

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_26 : 
  ∃! (n : ℕ), is_three_digit n ∧ digit_sum n = 26 := 
sorry

end NUMINAMATH_GPT_three_digit_sum_26_l2328_232867


namespace NUMINAMATH_GPT_mutual_acquainted_or_unacquainted_l2328_232825

theorem mutual_acquainted_or_unacquainted :
  ∀ (G : SimpleGraph (Fin 6)), 
  ∃ (V : Finset (Fin 6)), V.card = 3 ∧ ((∀ (u v : Fin 6), u ∈ V → v ∈ V → G.Adj u v) ∨ (∀ (u v : Fin 6), u ∈ V → v ∈ V → ¬G.Adj u v)) :=
by
  sorry

end NUMINAMATH_GPT_mutual_acquainted_or_unacquainted_l2328_232825


namespace NUMINAMATH_GPT_father_and_daughter_age_l2328_232869

-- A father's age is 5 times the daughter's age.
-- In 30 years, the father will be 3 times as old as the daughter.
-- Prove that the daughter's current age is 30 and the father's current age is 150.

theorem father_and_daughter_age :
  ∃ (d f : ℤ), (f = 5 * d) ∧ (f + 30 = 3 * (d + 30)) ∧ (d = 30 ∧ f = 150) :=
by
  sorry

end NUMINAMATH_GPT_father_and_daughter_age_l2328_232869


namespace NUMINAMATH_GPT_decimal_equiv_of_fraction_l2328_232820

theorem decimal_equiv_of_fraction : (1 / 5) ^ 2 = 0.04 := by
  sorry

end NUMINAMATH_GPT_decimal_equiv_of_fraction_l2328_232820


namespace NUMINAMATH_GPT_average_height_of_three_l2328_232894

theorem average_height_of_three (parker daisy reese : ℕ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : reese = 60) : 
  (parker + daisy + reese) / 3 = 64 := 
  sorry

end NUMINAMATH_GPT_average_height_of_three_l2328_232894


namespace NUMINAMATH_GPT_wire_length_around_square_field_l2328_232853

theorem wire_length_around_square_field (area : ℝ) (times : ℕ) (wire_length : ℝ) 
    (h1 : area = 69696) (h2 : times = 15) : wire_length = 15840 :=
by
  sorry

end NUMINAMATH_GPT_wire_length_around_square_field_l2328_232853


namespace NUMINAMATH_GPT_fold_creates_bisector_l2328_232835

-- Define an angle α with its vertex located outside the drawing (hence inaccessible)
structure Angle :=
  (theta1 theta2 : ℝ) -- theta1 and theta2 are the measures of the two angle sides

-- Define the condition: there exists an angle on transparent paper
variable (a: Angle)

-- Prove that folding such that the sides of the angle coincide results in the crease formed being the bisector
theorem fold_creates_bisector (a: Angle) :
  ∃ crease, crease = (a.theta1 + a.theta2) / 2 := 
sorry

end NUMINAMATH_GPT_fold_creates_bisector_l2328_232835


namespace NUMINAMATH_GPT_coordinates_of_foci_l2328_232895

-- Given conditions
def equation_of_hyperbola : Prop := ∃ (x y : ℝ), (x^2 / 4) - (y^2 / 5) = 1

-- The mathematical goal translated into a theorem
theorem coordinates_of_foci (x y : ℝ) (a b c : ℝ) (ha : a^2 = 4) (hb : b^2 = 5) (hc : c^2 = a^2 + b^2) :
  equation_of_hyperbola →
  ((x = 3 ∨ x = -3) ∧ y = 0) :=
sorry

end NUMINAMATH_GPT_coordinates_of_foci_l2328_232895


namespace NUMINAMATH_GPT_train_length_from_speed_l2328_232880

-- Definitions based on conditions
def seconds_to_cross_post : ℕ := 40
def seconds_to_cross_bridge : ℕ := 480
def bridge_length_meters : ℕ := 7200

-- Theorem statement to be proven
theorem train_length_from_speed :
  (bridge_length_meters / seconds_to_cross_bridge) * seconds_to_cross_post = 600 :=
sorry -- Proof is not provided

end NUMINAMATH_GPT_train_length_from_speed_l2328_232880


namespace NUMINAMATH_GPT_students_no_A_l2328_232814

theorem students_no_A
  (total_students : ℕ)
  (A_in_history : ℕ)
  (A_in_math : ℕ)
  (A_in_science : ℕ)
  (A_in_history_and_math : ℕ)
  (A_in_history_and_science : ℕ)
  (A_in_math_and_science : ℕ)
  (A_in_all_three : ℕ)
  (h_total_students : total_students = 40)
  (h_A_in_history : A_in_history = 10)
  (h_A_in_math : A_in_math = 15)
  (h_A_in_science : A_in_science = 8)
  (h_A_in_history_and_math : A_in_history_and_math = 5)
  (h_A_in_history_and_science : A_in_history_and_science = 3)
  (h_A_in_math_and_science : A_in_math_and_science = 4)
  (h_A_in_all_three : A_in_all_three = 2) :
  total_students - (A_in_history + A_in_math + A_in_science 
    - A_in_history_and_math - A_in_history_and_science - A_in_math_and_science 
    + A_in_all_three) = 17 := 
sorry

end NUMINAMATH_GPT_students_no_A_l2328_232814


namespace NUMINAMATH_GPT_Mary_work_hours_l2328_232851

variable (H : ℕ)
variable (weekly_earnings hourly_wage : ℕ)
variable (hours_Tuesday hours_Thursday : ℕ)

def weekly_hours (H : ℕ) : ℕ := 3 * H + hours_Tuesday + hours_Thursday

theorem Mary_work_hours:
  weekly_earnings = 11 * weekly_hours H → hours_Tuesday = 5 →
  hours_Thursday = 5 → weekly_earnings = 407 →
  hourly_wage = 11 → H = 9 :=
by
  intros earnings_eq tues_hours thurs_hours total_earn wage
  sorry

end NUMINAMATH_GPT_Mary_work_hours_l2328_232851


namespace NUMINAMATH_GPT_cubic_root_identity_l2328_232807

theorem cubic_root_identity (x1 x2 x3 : ℝ) (h1 : x1^3 - 3*x1 - 1 = 0) (h2 : x2^3 - 3*x2 - 1 = 0) (h3 : x3^3 - 3*x3 - 1 = 0) (h4 : x1 < x2) (h5 : x2 < x3) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end NUMINAMATH_GPT_cubic_root_identity_l2328_232807


namespace NUMINAMATH_GPT_exam_question_bound_l2328_232831

theorem exam_question_bound (n_students : ℕ) (k_questions : ℕ) (n_answers : ℕ) 
    (H_students : n_students = 25) (H_answers : n_answers = 5) 
    (H_condition : ∀ (i j : ℕ) (H1 : i < n_students) (H2 : j < n_students) (H_neq : i ≠ j), 
      ∀ q : ℕ, q < k_questions → ∀ ai aj : ℕ, ai < n_answers → aj < n_answers → 
      ((ai = aj) → (i = j ∨ q' > 1))) : 
    k_questions ≤ 6 := 
sorry

end NUMINAMATH_GPT_exam_question_bound_l2328_232831


namespace NUMINAMATH_GPT_interest_earned_l2328_232813

noncomputable def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T

noncomputable def T_years : ℚ :=
  5 + (8 / 12) + (12 / 365)

def principal : ℚ := 30000
def rate : ℚ := 23.7 / 100

theorem interest_earned :
  simple_interest principal rate T_years = 40524 := by
  sorry

end NUMINAMATH_GPT_interest_earned_l2328_232813


namespace NUMINAMATH_GPT_ratio_of_square_areas_l2328_232866

theorem ratio_of_square_areas (d s : ℝ)
  (h1 : d^2 = 2 * s^2) :
  (d^2) / (s^2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_square_areas_l2328_232866


namespace NUMINAMATH_GPT_circle_tangent_ellipse_l2328_232839

noncomputable def r : ℝ := (Real.sqrt 15) / 2

theorem circle_tangent_ellipse {x y : ℝ} (r : ℝ) (h₁ : r > 0) 
  (h₂ : ∀ x y, x^2 + 4*y^2 = 5 → ((x - r)^2 + y^2 = r^2 ∨ (x + r)^2 + y^2 = r^2))
  (h₃ : ∀ y, 4*(0 - r)^2 + (4*y^2) = 5 → ((-8*r)^2 - 4*3*(4*r^2 - 5) = 0)) :
  r = (Real.sqrt 15) / 2 :=
sorry

end NUMINAMATH_GPT_circle_tangent_ellipse_l2328_232839


namespace NUMINAMATH_GPT_percentage_increase_l2328_232884

theorem percentage_increase (D J : ℝ) (hD : D = 480) (hJ : J = 417.39) :
  ((D - J) / J) * 100 = 14.99 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l2328_232884


namespace NUMINAMATH_GPT_total_students_in_school_l2328_232834

theorem total_students_in_school (C1 C2 C3 C4 C5 : ℕ) 
  (h1 : C1 = 23)
  (h2 : C2 = C1 - 2)
  (h3 : C3 = C2 - 2)
  (h4 : C4 = C3 - 2)
  (h5 : C5 = C4 - 2)
  : C1 + C2 + C3 + C4 + C5 = 95 := 
by 
  -- proof details skipped with sorry
  sorry

end NUMINAMATH_GPT_total_students_in_school_l2328_232834


namespace NUMINAMATH_GPT_find_unit_price_B_l2328_232861

/-- Definitions based on the conditions --/
def total_cost_A := 7500
def total_cost_B := 4800
def quantity_difference := 30
def price_ratio : ℝ := 2.5

/-- Define the variable x as the unit price of B type soccer balls --/
def unit_price_B (x : ℝ) : Prop :=
  (total_cost_A / (price_ratio * x)) + 30 = (total_cost_B / x) ∧
  total_cost_A > 0 ∧ total_cost_B > 0 ∧ x > 0

/-- The main statement to prove --/
theorem find_unit_price_B (x : ℝ) : unit_price_B x ↔ x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_unit_price_B_l2328_232861


namespace NUMINAMATH_GPT_fraction_option_C_l2328_232805

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end NUMINAMATH_GPT_fraction_option_C_l2328_232805


namespace NUMINAMATH_GPT_equal_piece_length_l2328_232868

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end NUMINAMATH_GPT_equal_piece_length_l2328_232868


namespace NUMINAMATH_GPT_largest_divisible_by_6_ending_in_4_l2328_232860

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end NUMINAMATH_GPT_largest_divisible_by_6_ending_in_4_l2328_232860


namespace NUMINAMATH_GPT_solve_equation_l2328_232838

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48 ↔ x = 6 ∨ x = 8 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2328_232838


namespace NUMINAMATH_GPT_inverse_proportion_inequality_l2328_232862

variable (x1 x2 k : ℝ)
variable (y1 y2 : ℝ)

theorem inverse_proportion_inequality (h1 : x1 < 0) (h2 : 0 < x2) (hk : k < 0)
  (hy1 : y1 = k / x1) (hy2 : y2 = k / x2) : y2 < 0 ∧ 0 < y1 := 
by sorry

end NUMINAMATH_GPT_inverse_proportion_inequality_l2328_232862


namespace NUMINAMATH_GPT_maximize_annual_profit_l2328_232857

noncomputable def profit_function (x : ℝ) : ℝ :=
  - (1 / 3) * x^3 + 81 * x - 234

theorem maximize_annual_profit :
  ∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function x :=
sorry

end NUMINAMATH_GPT_maximize_annual_profit_l2328_232857


namespace NUMINAMATH_GPT_find_alpha_l2328_232882

def demand_function (p : ℝ) : ℝ := 150 - p
def supply_function (p : ℝ) : ℝ := 3 * p - 10

def new_demand_function (p : ℝ) (α : ℝ) : ℝ := α * (150 - p)

theorem find_alpha (α : ℝ) :
  (∃ p₀ p_new, demand_function p₀ = supply_function p₀ ∧ 
    p_new = p₀ * 1.25 ∧ 
    3 * p_new - 10 = new_demand_function p_new α) →
  α = 1.4 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_l2328_232882


namespace NUMINAMATH_GPT_legs_walking_on_ground_l2328_232889

def number_of_horses : ℕ := 14
def number_of_men : ℕ := number_of_horses
def legs_per_man : ℕ := 2
def legs_per_horse : ℕ := 4
def half (n : ℕ) : ℕ := n / 2

theorem legs_walking_on_ground :
  (half number_of_men) * legs_per_man + (half number_of_horses) * legs_per_horse = 42 :=
by
  sorry

end NUMINAMATH_GPT_legs_walking_on_ground_l2328_232889


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l2328_232873

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) :
  (|x - 1| < 2 → x ^ 2 - 5 * x - 6 < 0) ∧ ¬ (x ^ 2 - 5 * x - 6 < 0 → |x - 1| < 2) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l2328_232873


namespace NUMINAMATH_GPT_dynamic_load_L_value_l2328_232819

theorem dynamic_load_L_value (T H : ℝ) (hT : T = 3) (hH : H = 6) : 
  (L : ℝ) = (50 * T^3) / (H^3) -> L = 6.25 := 
by 
  sorry 

end NUMINAMATH_GPT_dynamic_load_L_value_l2328_232819


namespace NUMINAMATH_GPT_calculation_l2328_232890

theorem calculation :
  ((4.5 - 1.23) * 2.5 = 8.175) := 
by
  sorry

end NUMINAMATH_GPT_calculation_l2328_232890


namespace NUMINAMATH_GPT_square_side_length_l2328_232892

theorem square_side_length (a : ℝ) (n : ℕ) (P : ℝ) (h₀ : n = 5) (h₁ : 15 * (8 * a / 3) = P) (h₂ : P = 800) : a = 20 := 
by sorry

end NUMINAMATH_GPT_square_side_length_l2328_232892


namespace NUMINAMATH_GPT_int_solutions_fraction_l2328_232874

theorem int_solutions_fraction :
  ∀ n : ℤ, (∃ k : ℤ, (n - 2) / (n + 1) = k) ↔ n = 0 ∨ n = -2 ∨ n = 2 ∨ n = -4 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_int_solutions_fraction_l2328_232874


namespace NUMINAMATH_GPT_punger_needs_pages_l2328_232828

theorem punger_needs_pages (p c h : ℕ) (h_p : p = 60) (h_c : c = 7) (h_h : h = 10) : 
  (p * c) / h = 42 := 
by
  sorry

end NUMINAMATH_GPT_punger_needs_pages_l2328_232828


namespace NUMINAMATH_GPT_solve_for_X_l2328_232888

variable (X Y : ℝ)

def diamond (X Y : ℝ) := 4 * X + 3 * Y + 7

theorem solve_for_X (h : diamond X 5 = 75) : X = 53 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_X_l2328_232888


namespace NUMINAMATH_GPT_units_digit_3_pow_2004_l2328_232883

-- Definition of the observed pattern of the units digits of powers of 3.
def pattern_units_digits : List ℕ := [3, 9, 7, 1]

-- Theorem stating that the units digit of 3^2004 is 1.
theorem units_digit_3_pow_2004 : (3 ^ 2004) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_3_pow_2004_l2328_232883


namespace NUMINAMATH_GPT_expected_difference_l2328_232841

noncomputable def fair_eight_sided_die := [2, 3, 4, 5, 6, 7, 8]

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop := 
  n = 4 ∨ n = 6 ∨ n = 8

def unsweetened_cereal_days := (4 / 7) * 365
def sweetened_cereal_days := (3 / 7) * 365

theorem expected_difference :
  unsweetened_cereal_days - sweetened_cereal_days = 53 := by
  sorry

end NUMINAMATH_GPT_expected_difference_l2328_232841


namespace NUMINAMATH_GPT_andrew_donates_160_to_homeless_shelter_l2328_232824

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end NUMINAMATH_GPT_andrew_donates_160_to_homeless_shelter_l2328_232824


namespace NUMINAMATH_GPT_Addison_High_School_college_attendance_l2328_232843

theorem Addison_High_School_college_attendance:
  ∀ (G B : ℕ) (pG_not_college p_total_college : ℚ),
  G = 200 →
  B = 160 →
  pG_not_college = 0.4 →
  p_total_college = 0.6667 →
  ((B * 100) / 160) = 75 := 
by
  intro G B pG_not_college p_total_college G_eq B_eq pG_not_college_eq p_total_college_eq
  -- skipped proof
  sorry

end NUMINAMATH_GPT_Addison_High_School_college_attendance_l2328_232843
