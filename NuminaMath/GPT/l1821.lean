import Mathlib

namespace NUMINAMATH_GPT_plate_arrangement_l1821_182152

def arrangements_without_restriction : Nat :=
  Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 3)

def arrangements_adjacent_green : Nat :=
  (Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3)) * Nat.factorial 3

def allowed_arrangements : Nat :=
  arrangements_without_restriction - arrangements_adjacent_green

theorem plate_arrangement : 
  allowed_arrangements = 2520 := 
by
  sorry

end NUMINAMATH_GPT_plate_arrangement_l1821_182152


namespace NUMINAMATH_GPT_solutions_of_quadratic_l1821_182131

theorem solutions_of_quadratic (x : ℝ) : x^2 - x = 0 ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solutions_of_quadratic_l1821_182131


namespace NUMINAMATH_GPT_fraction_to_decimal_l1821_182133

theorem fraction_to_decimal : (31 : ℝ) / (2 * 5^6) = 0.000992 :=
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1821_182133


namespace NUMINAMATH_GPT_initial_investment_l1821_182116

theorem initial_investment (A r : ℝ) (n : ℕ) (P : ℝ) (hA : A = 630.25) (hr : r = 0.12) (hn : n = 5) :
  A = P * (1 + r) ^ n → P = 357.53 :=
by
  sorry

end NUMINAMATH_GPT_initial_investment_l1821_182116


namespace NUMINAMATH_GPT_smallest_x_y_sum_l1821_182166

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end NUMINAMATH_GPT_smallest_x_y_sum_l1821_182166


namespace NUMINAMATH_GPT_max_path_length_CQ_D_l1821_182161

noncomputable def maxCQDPathLength (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) : ℝ :=
  let r := dAB / 2
  let dCD := dAB - dAC - dBD
  2 * Real.sqrt (r^2 - (dCD / 2)^2)

theorem max_path_length_CQ_D 
  (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) (r := dAB / 2) (dCD := dAB - dAC - dBD) :
  dAB = 16 ∧ dAC = 3 ∧ dBD = 5 ∧ r = 8 ∧ dCD = 8
  → maxCQDPathLength 16 3 5 = 8 * Real.sqrt 3 :=
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_max_path_length_CQ_D_l1821_182161


namespace NUMINAMATH_GPT_problem_1_problem_2_l1821_182137

noncomputable def poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 := sorry

theorem problem_1 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  a₁ + a₂ + a₃ + a₄ = -80 :=
sorry

theorem problem_2 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 625 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1821_182137


namespace NUMINAMATH_GPT_find_constants_l1821_182135

-- Definitions based on the given problem
def inequality_in_x (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def roots_eq (a : ℝ) (r1 r2 : ℝ) : Prop :=
  a * r1^2 - 3 * r1 + 2 = 0 ∧ a * r2^2 - 3 * r2 + 2 = 0

def solution_set (a b : ℝ) (x : ℝ) : Prop :=
  x < 1 ∨ x > b

-- Problem statement: given conditions find a and b
theorem find_constants (a b : ℝ) (h1 : 1 < b) (h2 : 0 < a) :
  roots_eq a 1 b ∧ solution_set a b 1 ∧ solution_set a b b :=
sorry

end NUMINAMATH_GPT_find_constants_l1821_182135


namespace NUMINAMATH_GPT_find_x_when_y_is_20_l1821_182113

variable (x y k : ℝ)

axiom constant_ratio : (5 * 4 - 6) / (5 + 20) = k

theorem find_x_when_y_is_20 (h : (5 * x - 6) / (y + 20) = k) (hy : y = 20) : x = 5.68 := by
  sorry

end NUMINAMATH_GPT_find_x_when_y_is_20_l1821_182113


namespace NUMINAMATH_GPT_functional_equation_solution_l1821_182107

theorem functional_equation_solution {f : ℝ → ℝ} (h : ∀ x ≠ 1, (x - 1) * f (x + 1) - f x = x) :
    ∀ x, f x = 1 + 2 * x :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1821_182107


namespace NUMINAMATH_GPT_line_length_limit_l1821_182177

theorem line_length_limit : 
  ∑' n : ℕ, 1 / ((3 : ℝ) ^ n) + (1 / (3 ^ (n + 1))) * (Real.sqrt 3) = (3 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_line_length_limit_l1821_182177


namespace NUMINAMATH_GPT_y_range_for_conditions_l1821_182114

theorem y_range_for_conditions (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : -9 ≤ y ∧ y < -8 :=
sorry

end NUMINAMATH_GPT_y_range_for_conditions_l1821_182114


namespace NUMINAMATH_GPT_percent_less_than_m_plus_d_l1821_182191

-- Define the given conditions
variables (m d : ℝ) (distribution : ℝ → ℝ)

-- Assume the distribution is symmetric about the mean m
axiom symmetric_distribution :
  ∀ x, distribution (m + x) = distribution (m - x)

-- 84 percent of the distribution lies within one standard deviation d of the mean
axiom within_one_sd :
  ∫ x in -d..d, distribution (m + x) = 0.84

-- The goal is to prove that 42 percent of the distribution is less than m + d
theorem percent_less_than_m_plus_d : 
  ( ∫ x in -d..0, distribution (m + x) ) = 0.42 :=
by 
  sorry

end NUMINAMATH_GPT_percent_less_than_m_plus_d_l1821_182191


namespace NUMINAMATH_GPT_find_m_l1821_182127

def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
def C (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem find_m (m : ℝ) (h : A ∩ C m = C m) : 
  m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_find_m_l1821_182127


namespace NUMINAMATH_GPT_katie_clock_l1821_182163

theorem katie_clock (t_clock t_actual : ℕ) :
  t_clock = 540 →
  t_actual = (540 * 60) / 37 →
  8 * 60 + 875 = 22 * 60 + 36 :=
by
  intros h1 h2
  have h3 : 875 = (540 * 60 / 37) := sorry
  have h4 : 8 * 60 + 875 = 480 + 875 := sorry
  have h5 : 480 + 875 = 22 * 60 + 36 := sorry
  exact h5

end NUMINAMATH_GPT_katie_clock_l1821_182163


namespace NUMINAMATH_GPT_find_number_l1821_182194

theorem find_number (x : ℝ) (h : 0.9 * x = 0.0063) : x = 0.007 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l1821_182194


namespace NUMINAMATH_GPT_pencils_left_l1821_182183

-- Define initial count of pencils
def initial_pencils : ℕ := 20

-- Define pencils misplaced
def misplaced_pencils : ℕ := 7

-- Define pencils broken and thrown away
def broken_pencils : ℕ := 3

-- Define pencils found
def found_pencils : ℕ := 4

-- Define pencils bought
def bought_pencils : ℕ := 2

-- Define the final number of pencils
def final_pencils: ℕ := initial_pencils - misplaced_pencils - broken_pencils + found_pencils + bought_pencils

-- Prove that the final number of pencils is 16
theorem pencils_left : final_pencils = 16 :=
by
  -- The proof steps are omitted here
  sorry

end NUMINAMATH_GPT_pencils_left_l1821_182183


namespace NUMINAMATH_GPT_andrey_travel_distance_l1821_182192

theorem andrey_travel_distance:
  ∃ s t: ℝ, 
    (s = 60 * (t + 4/3) + 20  ∧ s = 90 * (t - 1/3) + 60) ∧ s = 180 :=
by
  sorry

end NUMINAMATH_GPT_andrey_travel_distance_l1821_182192


namespace NUMINAMATH_GPT_evaluate_expression_l1821_182119

theorem evaluate_expression : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1821_182119


namespace NUMINAMATH_GPT_laptop_weight_difference_is_3_67_l1821_182198

noncomputable def karen_tote_weight : ℝ := 8
noncomputable def kevin_empty_briefcase_weight : ℝ := karen_tote_weight / 2
noncomputable def umbrella_weight : ℝ := kevin_empty_briefcase_weight / 2
noncomputable def briefcase_full_weight_rainy_day : ℝ := 2 * karen_tote_weight
noncomputable def work_papers_weight : ℝ := (briefcase_full_weight_rainy_day - umbrella_weight) / 6
noncomputable def laptop_weight : ℝ := briefcase_full_weight_rainy_day - umbrella_weight - work_papers_weight
noncomputable def weight_difference : ℝ := laptop_weight - karen_tote_weight

theorem laptop_weight_difference_is_3_67 : weight_difference = 3.67 := by
  sorry

end NUMINAMATH_GPT_laptop_weight_difference_is_3_67_l1821_182198


namespace NUMINAMATH_GPT_complex_multiplication_l1821_182100

variable (i : ℂ)
axiom imag_unit : i^2 = -1

theorem complex_multiplication : (3 + i) * i = -1 + 3 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1821_182100


namespace NUMINAMATH_GPT_eval_expression_l1821_182176

variable {x : ℝ}

theorem eval_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 8 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1821_182176


namespace NUMINAMATH_GPT_train_crosses_pole_in_2point4_seconds_l1821_182115

noncomputable def time_to_cross (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / (speed_kmh * (5/18))

theorem train_crosses_pole_in_2point4_seconds :
  time_to_cross 120 180 = 2.4 := by
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_2point4_seconds_l1821_182115


namespace NUMINAMATH_GPT_grid_area_l1821_182167

-- Definitions based on problem conditions
def num_lines : ℕ := 36
def perimeter : ℕ := 72
def side_length : ℕ := perimeter / num_lines

-- Problem statement
theorem grid_area (h : num_lines = 36) (p : perimeter = 72)
  (s : side_length = 2) :
  let n_squares := (8 - 1) * (4 - 1)
  let area_square := side_length ^ 2
  let total_area := n_squares * area_square
  total_area = 84 :=
by {
  -- Skipping proof
  sorry
}

end NUMINAMATH_GPT_grid_area_l1821_182167


namespace NUMINAMATH_GPT_salary_percentage_l1821_182129

theorem salary_percentage (m n : ℝ) (P : ℝ) (h1 : m + n = 572) (h2 : n = 260) (h3 : m = (P / 100) * n) : P = 120 := 
by
  sorry

end NUMINAMATH_GPT_salary_percentage_l1821_182129


namespace NUMINAMATH_GPT_shawn_red_pebbles_l1821_182138

variable (Total : ℕ)
variable (B : ℕ)
variable (Y : ℕ)
variable (P : ℕ)
variable (G : ℕ)

theorem shawn_red_pebbles (h1 : Total = 40)
                          (h2 : B = 13)
                          (h3 : B - Y = 7)
                          (h4 : P = Y)
                          (h5 : G = Y)
                          (h6 : 3 * Y + B = Total)
                          : Total - (B + P + Y + G) = 9 :=
by
 sorry

end NUMINAMATH_GPT_shawn_red_pebbles_l1821_182138


namespace NUMINAMATH_GPT_eval_expression_l1821_182184

theorem eval_expression :
  16^3 + 3 * (16^2) * 2 + 3 * 16 * (2^2) + 2^3 = 5832 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1821_182184


namespace NUMINAMATH_GPT_inverse_function_correct_l1821_182109

noncomputable def inverse_function (y : ℝ) : ℝ := (1 / 2) * y - (3 / 2)

theorem inverse_function_correct :
  ∀ x ∈ Set.Icc (0 : ℝ) (5 : ℝ), (inverse_function (2 * x + 3) = x) ∧ (0 ≤ 2 * x + 3) ∧ (2 * x + 3 ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_correct_l1821_182109


namespace NUMINAMATH_GPT_monotonic_intervals_of_f_l1821_182182

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

-- Define the derivative f'
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

-- Prove the monotonicity intervals of the function f
theorem monotonic_intervals_of_f :
  (∀ x : ℝ, x < 0 → f' x < 0) ∧ (∀ x : ℝ, 0 < x → f' x > 0) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_of_f_l1821_182182


namespace NUMINAMATH_GPT_periodic_function_with_period_sqrt2_l1821_182160

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Definition of symmetry about x = sqrt(2)/2
def is_symmetric_about_line (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

-- Main theorem to prove
theorem periodic_function_with_period_sqrt2 (f : ℝ → ℝ) :
  is_even_function f → is_symmetric_about_line f (Real.sqrt 2 / 2) → ∃ T, T = Real.sqrt 2 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

end NUMINAMATH_GPT_periodic_function_with_period_sqrt2_l1821_182160


namespace NUMINAMATH_GPT_cube_volume_l1821_182126

theorem cube_volume
  (s : ℝ) 
  (surface_area_eq : 6 * s^2 = 54) :
  s^3 = 27 := 
by 
  sorry

end NUMINAMATH_GPT_cube_volume_l1821_182126


namespace NUMINAMATH_GPT_merchant_articles_l1821_182187

theorem merchant_articles (N CP SP : ℝ) 
  (h1 : N * CP = 16 * SP)
  (h2 : SP = CP * 1.375) : 
  N = 22 :=
by
  sorry

end NUMINAMATH_GPT_merchant_articles_l1821_182187


namespace NUMINAMATH_GPT_correct_calculation_option_D_l1821_182117

theorem correct_calculation_option_D (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_option_D_l1821_182117


namespace NUMINAMATH_GPT_initial_population_is_3162_l1821_182171

noncomputable def initial_population (P : ℕ) : Prop :=
  let after_bombardment := 0.95 * (P : ℝ)
  let after_fear := 0.85 * after_bombardment
  after_fear = 2553

theorem initial_population_is_3162 : initial_population 3162 :=
  by
    -- By our condition setup, we need to prove:
    -- let after_bombardment := 0.95 * 3162
    -- let after_fear := 0.85 * after_bombardment
    -- after_fear = 2553

    -- This can be directly stated and verified through concrete calculations as in the problem steps.
    sorry

end NUMINAMATH_GPT_initial_population_is_3162_l1821_182171


namespace NUMINAMATH_GPT_range_of_x_l1821_182193

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 2))) ↔ x > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1821_182193


namespace NUMINAMATH_GPT_bus_stop_time_l1821_182149

-- Usual time to walk to the bus stop
def usual_time (T : ℕ) := T

-- Usual speed
def usual_speed (S : ℕ) := S

-- New speed when walking at 4/5 of usual speed
def new_speed (S : ℕ) := (4 * S) / 5

-- Time relationship when walking at new speed
def time_relationship (T : ℕ) (S : ℕ) := (S / ((4 * S) / 5)) = (T + 10) / T

-- Prove the usual time T is 40 minutes
theorem bus_stop_time (T S : ℕ) (h1 : time_relationship T S) : T = 40 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l1821_182149


namespace NUMINAMATH_GPT_three_pow_gt_pow_three_for_n_ne_3_l1821_182105

theorem three_pow_gt_pow_three_for_n_ne_3 (n : ℕ) (h : n ≠ 3) : 3^n > n^3 :=
sorry

end NUMINAMATH_GPT_three_pow_gt_pow_three_for_n_ne_3_l1821_182105


namespace NUMINAMATH_GPT_polynomial_identity_l1821_182159

open Function

-- Define the polynomial terms
def f1 (x : ℝ) := 2*x^5 + 4*x^3 + 3*x + 4
def f2 (x : ℝ) := x^4 - 2*x^3 + 3
def g (x : ℝ) := -2*x^5 + x^4 - 6*x^3 - 3*x - 1

-- Lean theorem statement
theorem polynomial_identity :
  ∀ x : ℝ, f1 x + g x = f2 x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1821_182159


namespace NUMINAMATH_GPT_no_nonzero_integer_solution_l1821_182199

theorem no_nonzero_integer_solution (m n p : ℤ) :
  (m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0) → (m = 0 ∧ n = 0 ∧ p = 0) :=
by sorry

end NUMINAMATH_GPT_no_nonzero_integer_solution_l1821_182199


namespace NUMINAMATH_GPT_a_b_c_sum_l1821_182130

-- Definitions of the conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

theorem a_b_c_sum (a b c : ℝ) :
  (∀ x : ℝ, f (x + 4) a b c = 4 * x^2 + 9 * x + 5) ∧ (∀ x : ℝ, f x a b c = a * x^2 + b * x + c) →
  a + b + c = 14 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_a_b_c_sum_l1821_182130


namespace NUMINAMATH_GPT_value_of_a_l1821_182144

theorem value_of_a (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) (h3 : a > b) (h4 : a - b = 8) : a = 10 := 
by 
sorry

end NUMINAMATH_GPT_value_of_a_l1821_182144


namespace NUMINAMATH_GPT_second_machine_finishes_in_10_minutes_l1821_182168

-- Definitions for the conditions:
def time_to_clear_by_first_machine (t : ℝ) : Prop := t = 1
def time_to_clear_by_second_machine (t : ℝ) : Prop := t = 3 / 4
def time_first_machine_works (t : ℝ) : Prop := t = 1 / 3
def remaining_time (t : ℝ) : Prop := t = 1 / 6

-- Theorem statement:
theorem second_machine_finishes_in_10_minutes (t₁ t₂ t₃ t₄ : ℝ) 
  (h₁ : time_to_clear_by_first_machine t₁) 
  (h₂ : time_to_clear_by_second_machine t₂) 
  (h₃ : time_first_machine_works t₃) 
  (h₄ : remaining_time t₄) 
  : t₄ = 1 / 6 → t₄ * 60 = 10 := 
by
  -- here we can provide the proof steps, but the task does not require the proof
  sorry

end NUMINAMATH_GPT_second_machine_finishes_in_10_minutes_l1821_182168


namespace NUMINAMATH_GPT_parabola_focus_condition_l1821_182101

theorem parabola_focus_condition (m : ℝ) : (∃ (x y : ℝ), x + y - 2 = 0 ∧ y = (1 / (4 * m))) → m = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_condition_l1821_182101


namespace NUMINAMATH_GPT_fraction_undefined_at_one_l1821_182110

theorem fraction_undefined_at_one (x : ℤ) (h : x = 1) : (x / (x - 1) = 1) := by
  have h : 1 / (1 - 1) = 1 := sorry
  sorry

end NUMINAMATH_GPT_fraction_undefined_at_one_l1821_182110


namespace NUMINAMATH_GPT_passing_probability_l1821_182158

theorem passing_probability :
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  probability = 44 / 45 :=
by
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  have p_eq : probability = 44 / 45 := sorry
  exact p_eq

end NUMINAMATH_GPT_passing_probability_l1821_182158


namespace NUMINAMATH_GPT_parrots_fraction_l1821_182157

variable (P T : ℚ) -- P: fraction of parrots, T: fraction of toucans

def fraction_parrots (P T : ℚ) : Prop :=
  P + T = 1 ∧
  (2 / 3) * P + (1 / 4) * T = 0.5

theorem parrots_fraction (P T : ℚ) (h : fraction_parrots P T) : P = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_parrots_fraction_l1821_182157


namespace NUMINAMATH_GPT_factorization_a_minus_b_l1821_182175

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end NUMINAMATH_GPT_factorization_a_minus_b_l1821_182175


namespace NUMINAMATH_GPT_integral_of_x_squared_l1821_182134

-- Define the conditions
noncomputable def constant_term : ℝ := 3

-- Define the main theorem we want to prove
theorem integral_of_x_squared : ∫ (x : ℝ) in (1 : ℝ)..constant_term, x^2 = 26 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_integral_of_x_squared_l1821_182134


namespace NUMINAMATH_GPT_find_y_rotation_l1821_182136

def rotate_counterclockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry
def rotate_clockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry

variable {A B C : Point}
variable {y : ℝ}

theorem find_y_rotation
  (h1 : rotate_counterclockwise A B 450 = C)
  (h2 : rotate_clockwise A B y = C)
  (h3 : y < 360) :
  y = 270 :=
sorry

end NUMINAMATH_GPT_find_y_rotation_l1821_182136


namespace NUMINAMATH_GPT_factor_polynomial_l1821_182151

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1821_182151


namespace NUMINAMATH_GPT_expand_expression_l1821_182120

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1821_182120


namespace NUMINAMATH_GPT_tom_total_trip_cost_is_correct_l1821_182156

noncomputable def Tom_total_cost : ℝ :=
  let cost_vaccines := 10 * 45
  let cost_doctor := 250
  let total_medical := cost_vaccines + cost_doctor
  
  let insurance_coverage := 0.8 * total_medical
  let out_of_pocket_medical := total_medical - insurance_coverage
  
  let cost_flight := 1200

  let cost_lodging := 7 * 150
  let cost_transportation := 200
  let cost_food := 7 * 60
  let total_local_usd := cost_lodging + cost_transportation + cost_food
  let total_local_bbd := total_local_usd * 2

  let conversion_fee_bbd := 0.03 * total_local_bbd
  let conversion_fee_usd := conversion_fee_bbd / 2

  out_of_pocket_medical + cost_flight + total_local_usd + conversion_fee_usd

theorem tom_total_trip_cost_is_correct : Tom_total_cost = 3060.10 :=
  by
    -- Proof skipped
    sorry

end NUMINAMATH_GPT_tom_total_trip_cost_is_correct_l1821_182156


namespace NUMINAMATH_GPT_stratified_sampling_third_year_students_l1821_182150

theorem stratified_sampling_third_year_students 
  (total_students : ℕ)
  (sample_size : ℕ)
  (ratio_1st : ℕ)
  (ratio_2nd : ℕ)
  (ratio_3rd : ℕ)
  (ratio_4th : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 200)
  (h3 : ratio_1st = 4)
  (h4 : ratio_2nd = 3)
  (h5 : ratio_3rd = 2)
  (h6 : ratio_4th = 1) :
  (ratio_3rd : ℚ) / (ratio_1st + ratio_2nd + ratio_3rd + ratio_4th : ℚ) * sample_size = 40 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_year_students_l1821_182150


namespace NUMINAMATH_GPT_camel_steps_divisibility_l1821_182190

variables (A B : Type) (p q : ℕ)

-- Description of the conditions
-- let A, B be vertices
-- p and q be the steps to travel from A to B in different paths

theorem camel_steps_divisibility (h1: ∃ r : ℕ, p + r ≡ 0 [MOD 3])
                                  (h2: ∃ r : ℕ, q + r ≡ 0 [MOD 3]) : (p - q) % 3 = 0 := by
  sorry

end NUMINAMATH_GPT_camel_steps_divisibility_l1821_182190


namespace NUMINAMATH_GPT_sin_inequality_iff_angle_inequality_l1821_182178

section
variables {A B : ℝ} {a b : ℝ} (R : ℝ) (hA : A = Real.sin a) (hB : B = Real.sin b)

theorem sin_inequality_iff_angle_inequality (A B : ℝ) :
  (A > B) ↔ (Real.sin A > Real.sin B) :=
sorry
end

end NUMINAMATH_GPT_sin_inequality_iff_angle_inequality_l1821_182178


namespace NUMINAMATH_GPT_range_of_a_l1821_182179

theorem range_of_a (a : ℝ) : 
  (∀ P Q : ℝ × ℝ, P ≠ Q ∧ P.snd = a * P.fst ^ 2 - 1 ∧ Q.snd = a * Q.fst ^ 2 - 1 ∧ 
  P.fst + P.snd = -(Q.fst + Q.snd)) →
  a > 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1821_182179


namespace NUMINAMATH_GPT_g_value_at_100_l1821_182143

-- Given function g and its property
theorem g_value_at_100 (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y →
  x * g y - y * g x = g (x^2 / y)) : g 100 = 0 :=
sorry

end NUMINAMATH_GPT_g_value_at_100_l1821_182143


namespace NUMINAMATH_GPT_sum_of_coordinates_l1821_182148

variable (f : ℝ → ℝ)

/-- Given that the point (2, 3) is on the graph of y = f(x) / 3,
    show that (9, 2/3) must be on the graph of y = f⁻¹(x) / 3 and the
    sum of its coordinates is 29/3. -/
theorem sum_of_coordinates (h : 3 = f 2 / 3) : (9 : ℝ) + (2 / 3 : ℝ) = 29 / 3 :=
by
  have h₁ : f 2 = 9 := by
    linarith
    
  have h₂ : f⁻¹ 9 = 2 := by
    -- We assume that f has an inverse and it is well-defined
    sorry

  have point_on_graph : (9, (2 / 3)) ∈ { p : ℝ × ℝ | p.2 = f⁻¹ p.1 / 3 } := by
    sorry

  show 9 + 2 / 3 = 29 / 3
  norm_num

end NUMINAMATH_GPT_sum_of_coordinates_l1821_182148


namespace NUMINAMATH_GPT_luna_total_monthly_budget_l1821_182162

theorem luna_total_monthly_budget
  (H F phone_bill : ℝ)
  (h1 : F = 0.60 * H)
  (h2 : H + F = 240)
  (h3 : phone_bill = 0.10 * F) :
  H + F + phone_bill = 249 :=
by sorry

end NUMINAMATH_GPT_luna_total_monthly_budget_l1821_182162


namespace NUMINAMATH_GPT_chicken_burger_cost_l1821_182165

namespace BurgerCost

variables (C B : ℕ)

theorem chicken_burger_cost (h1 : B = C + 300) 
                            (h2 : 3 * B + 3 * C = 21000) : 
                            C = 3350 := 
sorry

end BurgerCost

end NUMINAMATH_GPT_chicken_burger_cost_l1821_182165


namespace NUMINAMATH_GPT_complementary_angle_measure_l1821_182104

theorem complementary_angle_measure (x : ℝ) (h1 : 0 < x) (h2 : 4*x + x = 90) : 4*x = 72 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angle_measure_l1821_182104


namespace NUMINAMATH_GPT_smoothie_ratios_l1821_182123

variable (initial_p initial_v m_p m_ratio_p_v: ℕ) (y_p y_v : ℕ)

-- Given conditions
theorem smoothie_ratios (h_initial_p : initial_p = 24) (h_initial_v : initial_v = 25) 
                        (h_m_p : m_p = 20) (h_m_ratio_p_v : m_ratio_p_v = 4)
                        (h_y_p : y_p = initial_p - m_p) (h_y_v : y_v = initial_v - m_p / m_ratio_p_v) :
  (y_p / gcd y_p y_v) = 1 ∧ (y_v / gcd y_p y_v) = 5 :=
by
  sorry

end NUMINAMATH_GPT_smoothie_ratios_l1821_182123


namespace NUMINAMATH_GPT_nine_sided_polygon_diagonals_l1821_182132

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end NUMINAMATH_GPT_nine_sided_polygon_diagonals_l1821_182132


namespace NUMINAMATH_GPT_clara_quarters_l1821_182128

theorem clara_quarters :
  ∃ q : ℕ, 8 < q ∧ q < 80 ∧ q % 3 = 1 ∧ q % 4 = 1 ∧ q % 5 = 1 ∧ q = 61 :=
by
  sorry

end NUMINAMATH_GPT_clara_quarters_l1821_182128


namespace NUMINAMATH_GPT_average_and_difference_l1821_182145

theorem average_and_difference
  (x y : ℚ) 
  (h1 : (15 + 24 + x + y) / 4 = 20)
  (h2 : x - y = 6) :
  x = 23.5 ∧ y = 17.5 := by
  sorry

end NUMINAMATH_GPT_average_and_difference_l1821_182145


namespace NUMINAMATH_GPT_cost_per_spool_l1821_182186

theorem cost_per_spool
  (p : ℕ) (f : ℕ) (y : ℕ) (t : ℕ) (n : ℕ)
  (hp : p = 15) (hf : f = 24) (hy : y = 5) (ht : t = 141) (hn : n = 2) :
  (t - (p + y * f)) / n = 3 :=
by sorry

end NUMINAMATH_GPT_cost_per_spool_l1821_182186


namespace NUMINAMATH_GPT_lcm_of_times_l1821_182189

-- Define the times each athlete takes to complete one lap
def time_A : Nat := 4
def time_B : Nat := 5
def time_C : Nat := 6

-- Prove that the LCM of 4, 5, and 6 is 60
theorem lcm_of_times : Nat.lcm time_A (Nat.lcm time_B time_C) = 60 := by
  sorry

end NUMINAMATH_GPT_lcm_of_times_l1821_182189


namespace NUMINAMATH_GPT_kim_monthly_revenue_l1821_182195

-- Define the cost to open the store
def initial_cost : ℤ := 25000

-- Define the monthly expenses
def monthly_expenses : ℤ := 1500

-- Define the number of months
def months : ℕ := 10

-- Define the revenue per month
def revenue_per_month (total_revenue : ℤ) (months : ℕ) : ℤ := total_revenue / months

theorem kim_monthly_revenue :
  ∃ r, revenue_per_month r months = 4000 :=
by 
  let total_expenses := monthly_expenses * months
  let total_revenue := initial_cost + total_expenses
  use total_revenue
  unfold revenue_per_month
  sorry

end NUMINAMATH_GPT_kim_monthly_revenue_l1821_182195


namespace NUMINAMATH_GPT_trucks_initial_count_l1821_182174

theorem trucks_initial_count (x : ℕ) (h : x - 13 = 38) : x = 51 :=
by sorry

end NUMINAMATH_GPT_trucks_initial_count_l1821_182174


namespace NUMINAMATH_GPT_square_of_volume_of_rect_box_l1821_182155

theorem square_of_volume_of_rect_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 18) 
  (h3 : z * x = 10) : (x * y * z) ^ 2 = 2700 :=
sorry

end NUMINAMATH_GPT_square_of_volume_of_rect_box_l1821_182155


namespace NUMINAMATH_GPT_ellipse_equation_l1821_182164

-- Definitions from conditions
def ecc (e : ℝ) := e = Real.sqrt 3 / 2
def parabola_focus (c : ℝ) (a : ℝ) := c = Real.sqrt 3 ∧ a = 2
def b_val (b a c : ℝ) := b = Real.sqrt (a^2 - c^2)

-- Main problem statement
theorem ellipse_equation (e a b c : ℝ) (x y : ℝ) :
  ecc e → parabola_focus c a → b_val b a c → (x^2 + y^2 / 4 = 1) := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1821_182164


namespace NUMINAMATH_GPT_carton_weight_l1821_182122

theorem carton_weight :
  ∀ (x : ℝ),
  (12 * 4 + 16 * x = 96) → 
  x = 3 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_carton_weight_l1821_182122


namespace NUMINAMATH_GPT_vertex_of_parabola_l1821_182118

theorem vertex_of_parabola :
  ∃ (h k : ℝ), (∀ x : ℝ, -2 * (x - h) ^ 2 + k = -2 * (x - 2) ^ 2 - 5) ∧ h = 2 ∧ k = -5 :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1821_182118


namespace NUMINAMATH_GPT_unique_rectangle_l1821_182103

theorem unique_rectangle (a b : ℝ) (h : a < b) :
  ∃! (x y : ℝ), (x < y) ∧ (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 4) := 
sorry

end NUMINAMATH_GPT_unique_rectangle_l1821_182103


namespace NUMINAMATH_GPT_delta_k_f_l1821_182181

open Nat

-- Define the function
def f (n : ℕ) : ℕ := 3^n

-- Define the discrete difference operator
def Δ (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

-- Define the k-th discrete difference
def Δk (g : ℕ → ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  if k = 0 then g n else Δk (Δ g) (k - 1) n

-- State the theorem
theorem delta_k_f (k : ℕ) (n : ℕ) (h : k ≥ 1) : Δk f k n = 2^k * 3^n := by
  sorry

end NUMINAMATH_GPT_delta_k_f_l1821_182181


namespace NUMINAMATH_GPT_initial_amount_l1821_182141

theorem initial_amount (X : ℝ) (h : 0.7 * X = 3500) : X = 5000 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l1821_182141


namespace NUMINAMATH_GPT_total_cost_after_discount_l1821_182139

noncomputable def mango_cost : ℝ := sorry
noncomputable def rice_cost : ℝ := sorry
noncomputable def flour_cost : ℝ := 21

theorem total_cost_after_discount :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (flour_cost = 21) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost) * 0.9 = 808.92 :=
by
  intros h1 h2 h3
  -- sorry as placeholder for actual proof
  sorry

end NUMINAMATH_GPT_total_cost_after_discount_l1821_182139


namespace NUMINAMATH_GPT_number_of_correct_statements_l1821_182146

def is_opposite (a b : ℤ) : Prop := a + b = 0

def statement1 : Prop := ∀ a b : ℤ, (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → is_opposite a b
def statement2 : Prop := ∀ n : ℤ, n = -n → n < 0
def statement3 : Prop := ∀ a b : ℤ, is_opposite a b → a + b = 0
def statement4 : Prop := ∀ a b : ℤ, is_opposite a b → (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)

theorem number_of_correct_statements : (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ↔ (∃n : ℕ, n = 1) :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_statements_l1821_182146


namespace NUMINAMATH_GPT_smallest_number_with_divisibility_condition_l1821_182124

theorem smallest_number_with_divisibility_condition :
  ∃ x : ℕ, (x + 7) % 24 = 0 ∧ (x + 7) % 36 = 0 ∧ (x + 7) % 50 = 0 ∧ (x + 7) % 56 = 0 ∧ (x + 7) % 81 = 0 ∧ x = 113393 :=
by {
  -- sorry is used to skip the proof.
  sorry
}

end NUMINAMATH_GPT_smallest_number_with_divisibility_condition_l1821_182124


namespace NUMINAMATH_GPT_f_at_2_lt_e6_l1821_182154

variable (f : ℝ → ℝ)

-- Specify the conditions
axiom derivable_f : Differentiable ℝ f
axiom condition_3f_gt_fpp : ∀ x : ℝ, 3 * f x > (deriv (deriv f)) x
axiom f_at_1 : f 1 = Real.exp 3

-- Conclusion to prove
theorem f_at_2_lt_e6 : f 2 < Real.exp 6 :=
sorry

end NUMINAMATH_GPT_f_at_2_lt_e6_l1821_182154


namespace NUMINAMATH_GPT_cos_double_angle_l1821_182125

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 1 / 3) :
  Real.cos (2 * α) = 7 / 9 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1821_182125


namespace NUMINAMATH_GPT_find_x_l1821_182108
-- Import all necessary libraries

-- Define the conditions
variables (x : ℝ) (log5x log6x log15x : ℝ)

-- Assume the edge lengths of the prism are logs with different bases
def edge_lengths (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  log5x = Real.logb 5 x ∧ log6x = Real.logb 6 x ∧ log15x = Real.logb 15 x

-- Define the ratio of Surface Area to Volume
def ratio_SA_to_V (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  let SA := 2 * (log5x * log6x + log5x * log15x + log6x * log15x)
  let V  := log5x * log6x * log15x
  SA / V = 10

-- Prove the value of x
theorem find_x (h1 : edge_lengths x log5x log6x log15x) (h2 : ratio_SA_to_V x log5x log6x log15x) :
  x = Real.rpow 450 (1/5) := 
sorry

end NUMINAMATH_GPT_find_x_l1821_182108


namespace NUMINAMATH_GPT_factorization_difference_of_squares_l1821_182197

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_factorization_difference_of_squares_l1821_182197


namespace NUMINAMATH_GPT_servings_left_proof_l1821_182180

-- Define the number of servings prepared
def total_servings : ℕ := 61

-- Define the number of guests
def total_guests : ℕ := 8

-- Define the fraction of servings the first 3 guests shared
def first_three_fraction : ℚ := 2 / 5

-- Define the fraction of servings the next 4 guests shared
def next_four_fraction : ℚ := 1 / 4

-- Define the number of servings consumed by the 8th guest
def eighth_guest_servings : ℕ := 5

-- Total consumed servings by the first three guests (rounded down)
def first_three_consumed := (first_three_fraction * total_servings).floor

-- Total consumed servings by the next four guests (rounded down)
def next_four_consumed := (next_four_fraction * total_servings).floor

-- Total consumed servings in total
def total_consumed := first_three_consumed + next_four_consumed + eighth_guest_servings

-- The number of servings left unconsumed
def servings_left_unconsumed := total_servings - total_consumed

-- The theorem stating there are 17 servings left unconsumed
theorem servings_left_proof : servings_left_unconsumed = 17 := by
  sorry

end NUMINAMATH_GPT_servings_left_proof_l1821_182180


namespace NUMINAMATH_GPT_find_a1_general_term_sum_of_terms_l1821_182172

-- Given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom h_condition : ∀ n, S n = (3 / 2) * a n - (1 / 2)

-- Specific condition for finding a1
axiom h_S1_eq_1 : S 1 = 1

-- Prove statements
theorem find_a1 : a 1 = 1 :=
by
  sorry

theorem general_term (n : ℕ) : n ≥ 1 → a n = 3 ^ (n - 1) :=
by
  sorry

theorem sum_of_terms (n : ℕ) : n ≥ 1 → S n = (3 ^ n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_general_term_sum_of_terms_l1821_182172


namespace NUMINAMATH_GPT_decrease_equation_l1821_182142

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_decrease_equation_l1821_182142


namespace NUMINAMATH_GPT_class_mean_calculation_correct_l1821_182173

variable (s1 s2 : ℕ) (mean1 mean2 : ℕ)
variable (n : ℕ) (mean_total : ℕ)

def overall_class_mean (s1 s2 mean1 mean2 : ℕ) : ℕ :=
  let total_score := (s1 * mean1) + (s2 * mean2)
  total_score / (s1 + s2)

theorem class_mean_calculation_correct
  (h1 : s1 = 40)
  (h2 : s2 = 10)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : n = 50)
  (h6 : mean_total = 82) :
  overall_class_mean s1 s2 mean1 mean2 = mean_total :=
  sorry

end NUMINAMATH_GPT_class_mean_calculation_correct_l1821_182173


namespace NUMINAMATH_GPT_price_per_glass_on_second_day_l1821_182147

 -- Definitions based on the conditions
def orangeade_first_day (O: ℝ) : ℝ := 2 * O -- Total volume on first day, O + O
def orangeade_second_day (O: ℝ) : ℝ := 3 * O -- Total volume on second day, O + 2O
def revenue_first_day (O: ℝ) (price_first_day: ℝ) : ℝ := 2 * O * price_first_day -- Revenue on first day
def revenue_second_day (O: ℝ) (P: ℝ) : ℝ := 3 * O * P -- Revenue on second day
def price_first_day: ℝ := 0.90 -- Given price per glass on the first day

 -- Statement to be proved
theorem price_per_glass_on_second_day (O: ℝ) (P: ℝ) (h: revenue_first_day O price_first_day = revenue_second_day O P) :
  P = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_price_per_glass_on_second_day_l1821_182147


namespace NUMINAMATH_GPT_func_has_extrema_l1821_182185

theorem func_has_extrema (a b c : ℝ) (h_a_nonzero : a ≠ 0) (h_discriminant_positive : b^2 + 8 * a * c > 0) 
    (h_pos_sum_roots : b / a > 0) (h_pos_product_roots : -2 * c / a > 0) : 
    (a * b > 0) ∧ (a * c < 0) :=
by 
  -- Proof skipped.
  sorry

end NUMINAMATH_GPT_func_has_extrema_l1821_182185


namespace NUMINAMATH_GPT_f_2010_eq_0_l1821_182153

theorem f_2010_eq_0 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, f (x + 2) = f x) : 
  f 2010 = 0 :=
by sorry

end NUMINAMATH_GPT_f_2010_eq_0_l1821_182153


namespace NUMINAMATH_GPT_reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l1821_182106

theorem reach_one_from_45 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 45 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_345 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 345 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_any_nat (n : ℕ) (h : n ≠ 0) : ∃ (k : ℕ), k = 1 :=
by
  -- Prove that starting from any non-zero natural number, you can reach 1.
  sorry

end NUMINAMATH_GPT_reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l1821_182106


namespace NUMINAMATH_GPT_discount_per_person_correct_l1821_182102

noncomputable def price_per_person : ℕ := 147
noncomputable def total_people : ℕ := 2
noncomputable def total_cost_with_discount : ℕ := 266

theorem discount_per_person_correct :
  let total_cost_without_discount := price_per_person * total_people
  let total_discount := total_cost_without_discount - total_cost_with_discount
  let discount_per_person := total_discount / total_people
  discount_per_person = 14 := by
  sorry

end NUMINAMATH_GPT_discount_per_person_correct_l1821_182102


namespace NUMINAMATH_GPT_day_53_days_from_friday_l1821_182196

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end NUMINAMATH_GPT_day_53_days_from_friday_l1821_182196


namespace NUMINAMATH_GPT_sin_alpha_terminal_point_l1821_182111

theorem sin_alpha_terminal_point :
  let alpha := (2 * Real.cos (120 * (π / 180)), Real.sqrt 2 * Real.sin (225 * (π / 180)))
  α = -π / 4 →
  α.sin = - Real.sqrt 2 / 2
:=
by
  intro α_definition
  sorry

end NUMINAMATH_GPT_sin_alpha_terminal_point_l1821_182111


namespace NUMINAMATH_GPT_find_x_in_coconut_grove_l1821_182112

theorem find_x_in_coconut_grove
  (x : ℕ)
  (h1 : (x + 2) * 30 + x * 120 + (x - 2) * 180 = 300 * x)
  (h2 : 3 * x ≠ 0) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_coconut_grove_l1821_182112


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1821_182121

theorem quadratic_two_distinct_real_roots (m : ℝ) : 
  ∀ x : ℝ, x^2 + m * x - 2 = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1821_182121


namespace NUMINAMATH_GPT_cost_per_charge_l1821_182188

theorem cost_per_charge
  (charges : ℕ) (budget left : ℝ) (cost_per_charge : ℝ)
  (charges_eq : charges = 4)
  (budget_eq : budget = 20)
  (left_eq : left = 6) :
  cost_per_charge = (budget - left) / charges :=
by
  apply sorry

end NUMINAMATH_GPT_cost_per_charge_l1821_182188


namespace NUMINAMATH_GPT_possible_n_values_l1821_182169

theorem possible_n_values (x y n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : n > 0)
  (top_box_eq : x * y * n^2 = 720) :
  ∃ k : ℕ,  k = 6 :=
by 
  sorry

end NUMINAMATH_GPT_possible_n_values_l1821_182169


namespace NUMINAMATH_GPT_dodgeballs_purchasable_l1821_182140

-- Definitions for the given conditions
def original_budget (B : ℝ) := B
def new_budget (B : ℝ) := 1.2 * B
def cost_per_dodgeball : ℝ := 5
def cost_per_softball : ℝ := 9
def softballs_purchased (B : ℝ) := 10

-- Theorem statement
theorem dodgeballs_purchasable {B : ℝ} (h : new_budget B = 90) : original_budget B / cost_per_dodgeball = 15 := 
by 
  sorry

end NUMINAMATH_GPT_dodgeballs_purchasable_l1821_182140


namespace NUMINAMATH_GPT_greatest_common_divisor_sum_arithmetic_sequence_l1821_182170

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_sum_arithmetic_sequence_l1821_182170
