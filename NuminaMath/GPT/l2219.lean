import Mathlib

namespace NUMINAMATH_GPT_lcm_Anthony_Bethany_Casey_Dana_l2219_221928

theorem lcm_Anthony_Bethany_Casey_Dana : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 10) = 120 := 
by
  sorry

end NUMINAMATH_GPT_lcm_Anthony_Bethany_Casey_Dana_l2219_221928


namespace NUMINAMATH_GPT_simplify_expression_l2219_221986

theorem simplify_expression (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2219_221986


namespace NUMINAMATH_GPT_area_of_shaded_region_l2219_221922

theorem area_of_shaded_region:
  let b := 10
  let h := 6
  let n := 14
  let rect_length := 2
  let rect_height := 1.5
  (n * rect_length * rect_height - (1/2 * b * h)) = 12 := 
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l2219_221922


namespace NUMINAMATH_GPT_triangle_has_at_most_one_obtuse_angle_l2219_221957

-- Definitions
def Triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def Obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

def Two_obtuse_angles (α β γ : ℝ) : Prop :=
  Obtuse_angle α ∧ Obtuse_angle β

-- Theorem Statement
theorem triangle_has_at_most_one_obtuse_angle (α β γ : ℝ) (h_triangle : Triangle α β γ) :
  ¬ Two_obtuse_angles α β γ := 
sorry

end NUMINAMATH_GPT_triangle_has_at_most_one_obtuse_angle_l2219_221957


namespace NUMINAMATH_GPT_f_1984_can_be_any_real_l2219_221952

noncomputable def f : ℤ → ℝ := sorry

axiom f_condition : ∀ (x y : ℤ), f (x - y^2) = f x + (y^2 - 2 * x) * f y

theorem f_1984_can_be_any_real
    (a : ℝ)
    (h : f 1 = a) : f 1984 = 1984^2 * a := sorry

end NUMINAMATH_GPT_f_1984_can_be_any_real_l2219_221952


namespace NUMINAMATH_GPT_correct_calculation_l2219_221934

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2219_221934


namespace NUMINAMATH_GPT_inequality_proof_l2219_221906

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2219_221906


namespace NUMINAMATH_GPT_most_accurate_method_is_independence_test_l2219_221931

-- Definitions and assumptions
inductive Methods
| contingency_table
| independence_test
| stacked_bar_chart
| others

def related_or_independent_method : Methods := Methods.independence_test

-- Proof statement
theorem most_accurate_method_is_independence_test :
  related_or_independent_method = Methods.independence_test :=
sorry

end NUMINAMATH_GPT_most_accurate_method_is_independence_test_l2219_221931


namespace NUMINAMATH_GPT_solve_quadratic_equation_l2219_221984

theorem solve_quadratic_equation (x : ℝ) : x^2 + 4 * x = 5 ↔ x = 1 ∨ x = -5 := sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l2219_221984


namespace NUMINAMATH_GPT_three_digit_repeated_digits_percentage_l2219_221977

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 900
  let non_repeated := 9 * 9 * 8
  let repeated := total_numbers - non_repeated
  (repeated / total_numbers) * 100

theorem three_digit_repeated_digits_percentage :
  percentage_repeated_digits = 28.0 := by
  sorry

end NUMINAMATH_GPT_three_digit_repeated_digits_percentage_l2219_221977


namespace NUMINAMATH_GPT_joan_apples_l2219_221912

theorem joan_apples (initial_apples : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : 
  initial_apples = 43 ∧ given_to_melanie = 27 ∧ given_to_sarah = 11 → (initial_apples - given_to_melanie - given_to_sarah) = 5 := 
by
  sorry

end NUMINAMATH_GPT_joan_apples_l2219_221912


namespace NUMINAMATH_GPT_total_area_of_triangles_l2219_221985

theorem total_area_of_triangles :
    let AB := 12
    let DE := 8 * Real.sqrt 2
    let area_ABC := (1 / 2) * AB * AB
    let area_DEF := (1 / 2) * DE * DE * 2
    area_ABC + area_DEF = 136 := by
  sorry

end NUMINAMATH_GPT_total_area_of_triangles_l2219_221985


namespace NUMINAMATH_GPT_find_x_value_l2219_221965

theorem find_x_value (x : ℚ) (h1 : 9 * x ^ 2 + 8 * x - 1 = 0) (h2 : 27 * x ^ 2 + 65 * x - 8 = 0) : x = 1 / 9 :=
sorry

end NUMINAMATH_GPT_find_x_value_l2219_221965


namespace NUMINAMATH_GPT_wickets_before_last_match_l2219_221959

theorem wickets_before_last_match
  (W : ℝ)  -- Number of wickets before last match
  (R : ℝ)  -- Total runs before last match
  (h1 : R = 12.4 * W)
  (h2 : (R + 26) / (W + 8) = 12.0)
  : W = 175 :=
sorry

end NUMINAMATH_GPT_wickets_before_last_match_l2219_221959


namespace NUMINAMATH_GPT_average_daily_net_income_correct_l2219_221968

-- Define the income, tips, and expenses for each day.
def day1_income := 300
def day1_tips := 50
def day1_expenses := 80

def day2_income := 150
def day2_tips := 20
def day2_expenses := 40

def day3_income := 750
def day3_tips := 100
def day3_expenses := 150

def day4_income := 200
def day4_tips := 30
def day4_expenses := 50

def day5_income := 600
def day5_tips := 70
def day5_expenses := 120

-- Define the net income for each day as income + tips - expenses.
def day1_net_income := day1_income + day1_tips - day1_expenses
def day2_net_income := day2_income + day2_tips - day2_expenses
def day3_net_income := day3_income + day3_tips - day3_expenses
def day4_net_income := day4_income + day4_tips - day4_expenses
def day5_net_income := day5_income + day5_tips - day5_expenses

-- Calculate the total net income over the 5 days.
def total_net_income := 
  day1_net_income + day2_net_income + day3_net_income + day4_net_income + day5_net_income

-- Define the number of days.
def number_of_days := 5

-- Calculate the average daily net income.
def average_daily_net_income := total_net_income / number_of_days

-- Statement to prove that the average daily net income is $366.
theorem average_daily_net_income_correct :
  average_daily_net_income = 366 := by
  sorry

end NUMINAMATH_GPT_average_daily_net_income_correct_l2219_221968


namespace NUMINAMATH_GPT_problem_statement_l2219_221943

variable {x y z : ℝ}

theorem problem_statement (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
    (h₁ : x^2 - y^2 = y * z) (h₂ : y^2 - z^2 = x * z) : 
    x^2 - z^2 = x * y := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2219_221943


namespace NUMINAMATH_GPT_lucas_age_correct_l2219_221923

variable (Noah_age : ℕ) (Mia_age : ℕ) (Lucas_age : ℕ)

-- Conditions
axiom h1 : Noah_age = 12
axiom h2 : Mia_age = Noah_age + 5
axiom h3 : Lucas_age = Mia_age - 6

-- Goal
theorem lucas_age_correct : Lucas_age = 11 := by
  sorry

end NUMINAMATH_GPT_lucas_age_correct_l2219_221923


namespace NUMINAMATH_GPT_sally_earnings_in_dozens_l2219_221948

theorem sally_earnings_in_dozens (earnings_per_house : ℕ) (houses_cleaned : ℕ) (dozens_of_dollars : ℕ) : 
  earnings_per_house = 25 ∧ houses_cleaned = 96 → dozens_of_dollars = 200 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_sally_earnings_in_dozens_l2219_221948


namespace NUMINAMATH_GPT_x_intercept_perpendicular_line_l2219_221946

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end NUMINAMATH_GPT_x_intercept_perpendicular_line_l2219_221946


namespace NUMINAMATH_GPT_value_of_expression_l2219_221918

-- Define the variables and conditions
variables (x y : ℝ)
axiom h1 : x + 2 * y = 4
axiom h2 : x * y = -8

-- Define the statement to be proven
theorem value_of_expression : x^2 + 4 * y^2 = 48 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2219_221918


namespace NUMINAMATH_GPT_standard_deviation_is_2point5_l2219_221933

noncomputable def mean : ℝ := 17.5
noncomputable def given_value : ℝ := 12.5

theorem standard_deviation_is_2point5 :
  ∀ (σ : ℝ), mean - 2 * σ = given_value → σ = 2.5 := by
  sorry

end NUMINAMATH_GPT_standard_deviation_is_2point5_l2219_221933


namespace NUMINAMATH_GPT_solve_system_equations_l2219_221981

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_system_equations :
  ∃ x y : ℝ, (y = 10^((log10 x)^(log10 x)) ∧ (log10 x)^(log10 (2 * x)) = (log10 y) * 10^((log10 (log10 x))^2))
  → ((x = 10 ∧ y = 10) ∨ (x = 100 ∧ y = 10000)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_equations_l2219_221981


namespace NUMINAMATH_GPT_initial_books_from_library_l2219_221941

-- Definitions of the problem conditions
def booksGivenAway : ℝ := 23.0
def booksLeft : ℝ := 31.0

-- Statement of the problem, proving that the initial number of books
def initialBooks (x : ℝ) : Prop :=
  x = booksGivenAway + booksLeft

-- Main theorem
theorem initial_books_from_library : initialBooks 54.0 :=
by
  -- Proof pending
  sorry

end NUMINAMATH_GPT_initial_books_from_library_l2219_221941


namespace NUMINAMATH_GPT_combined_salaries_l2219_221950
-- Import the required libraries

-- Define the salaries and conditions
def salary_c := 14000
def avg_salary_five := 8600
def num_individuals := 5
def total_salary := avg_salary_five * num_individuals

-- Define what we need to prove
theorem combined_salaries : total_salary - salary_c = 29000 :=
by
  -- The theorem statement
  sorry

end NUMINAMATH_GPT_combined_salaries_l2219_221950


namespace NUMINAMATH_GPT_g_of_x_l2219_221996

theorem g_of_x (f g : ℕ → ℕ) (h1 : ∀ x, f x = 2 * x + 3)
  (h2 : ∀ x, g (x + 2) = f x) : ∀ x, g x = 2 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_g_of_x_l2219_221996


namespace NUMINAMATH_GPT_fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l2219_221960

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Condition 1: a = 1, b = 5; the fixed points are x = -1 or x = -4
theorem fixed_points_a_one_b_five : 
  ∀ x : ℝ, is_fixed_point (f 1 5) x ↔ x = -1 ∨ x = -4 := by
  -- Proof goes here
  sorry

-- Condition 2: For any real b, f(x) always having two distinct fixed points implies 0 < a < 1
theorem range_of_a_two_distinct_fixed_points : 
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) ↔ 0 < a ∧ a < 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l2219_221960


namespace NUMINAMATH_GPT_sequence_is_k_plus_n_l2219_221980

theorem sequence_is_k_plus_n (a : ℕ → ℕ) (k : ℕ) (h : ∀ n : ℕ, a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1))
  (pos: ∀ n: ℕ, a n > 0) : ∀ n: ℕ, a n = k + n := 
sorry

end NUMINAMATH_GPT_sequence_is_k_plus_n_l2219_221980


namespace NUMINAMATH_GPT_youngest_child_age_l2219_221909

variables (child_ages : Fin 5 → ℕ)

def child_ages_eq_intervals (x : ℕ) : Prop :=
  child_ages 0 = x ∧ child_ages 1 = x + 8 ∧ child_ages 2 = x + 16 ∧ child_ages 3 = x + 24 ∧ child_ages 4 = x + 32

def sum_of_ages_eq (child_ages : Fin 5 → ℕ) (sum : ℕ) : Prop :=
  (Finset.univ : Finset (Fin 5)).sum child_ages = sum

theorem youngest_child_age (child_ages : Fin 5 → ℕ) (h1 : ∃ x, child_ages_eq_intervals child_ages x) (h2 : sum_of_ages_eq child_ages 90) :
  ∃ x, x = 2 ∧ child_ages 0 = x :=
sorry

end NUMINAMATH_GPT_youngest_child_age_l2219_221909


namespace NUMINAMATH_GPT_find_z_l2219_221974

open Complex

theorem find_z (z : ℂ) (h : (1 - I) * z = 2 * I) : z = -1 + I := by
  sorry

end NUMINAMATH_GPT_find_z_l2219_221974


namespace NUMINAMATH_GPT_conical_pile_volume_l2219_221951

noncomputable def volume_of_cone (d : ℝ) (h : ℝ) : ℝ :=
  (Real.pi * (d / 2) ^ 2 * h) / 3

theorem conical_pile_volume :
  let diameter := 10
  let height := 0.60 * diameter
  volume_of_cone diameter height = 50 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_conical_pile_volume_l2219_221951


namespace NUMINAMATH_GPT_base_of_second_exponent_l2219_221973

theorem base_of_second_exponent (a b : ℕ) (x : ℕ) 
  (h1 : (18^a) * (x^(3 * a - 1)) = (2^6) * (3^b)) 
  (h2 : a = 6) 
  (h3 : 0 < a)
  (h4 : 0 < b) : x = 3 := 
by
  sorry

end NUMINAMATH_GPT_base_of_second_exponent_l2219_221973


namespace NUMINAMATH_GPT_no_possible_values_for_n_l2219_221967

theorem no_possible_values_for_n (n a : ℤ) (h : n > 1) (d : ℤ := 3) (Sn : ℤ := 180) :
  ∃ n > 1, ∃ k : ℤ, a = k^2 ∧ Sn = n / 2 * (2 * a + (n - 1) * d) :=
sorry

end NUMINAMATH_GPT_no_possible_values_for_n_l2219_221967


namespace NUMINAMATH_GPT_tan_150_eq_neg_one_over_sqrt_three_l2219_221964

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_150_eq_neg_one_over_sqrt_three_l2219_221964


namespace NUMINAMATH_GPT_fraction_identity_l2219_221956

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_identity_l2219_221956


namespace NUMINAMATH_GPT_complement_event_l2219_221926

-- Definitions based on conditions
variables (shoot1 shoot2 : Prop) -- shoots the target on the first and second attempt

-- Definition based on the question and answer
def hits_at_least_once : Prop := shoot1 ∨ shoot2
def misses_both_times : Prop := ¬shoot1 ∧ ¬shoot2

-- Theorem statement based on the mathematical translation
theorem complement_event :
  misses_both_times shoot1 shoot2 = ¬hits_at_least_once shoot1 shoot2 :=
by sorry

end NUMINAMATH_GPT_complement_event_l2219_221926


namespace NUMINAMATH_GPT_pies_and_leftover_apples_l2219_221994

theorem pies_and_leftover_apples 
  (apples : ℕ) 
  (h : apples = 55) 
  (h1 : 15/3 = 5) :
  (apples / 5 = 11) ∧ (apples - 11 * 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_pies_and_leftover_apples_l2219_221994


namespace NUMINAMATH_GPT_lowest_possible_price_l2219_221925

theorem lowest_possible_price
  (MSRP : ℝ)
  (D1 : ℝ)
  (D2 : ℝ)
  (P_final : ℝ)
  (h1 : MSRP = 45.00)
  (h2 : 0.10 ≤ D1 ∧ D1 ≤ 0.30)
  (h3 : D2 = 0.20) :
  P_final = 25.20 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_price_l2219_221925


namespace NUMINAMATH_GPT_probability_hits_10_ring_l2219_221901

-- Definitions based on conditions
def total_shots : ℕ := 10
def hits_10_ring : ℕ := 2

-- Theorem stating the question and answer equivalence.
theorem probability_hits_10_ring : (hits_10_ring : ℚ) / total_shots = 0.2 := by
  -- We are skipping the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_probability_hits_10_ring_l2219_221901


namespace NUMINAMATH_GPT_max_principals_in_10_years_l2219_221944

theorem max_principals_in_10_years : ∀ term_length num_years,
  (term_length = 4) ∧ (num_years = 10) →
  ∃ max_principals, max_principals = 3
:=
  by intros term_length num_years h
     sorry

end NUMINAMATH_GPT_max_principals_in_10_years_l2219_221944


namespace NUMINAMATH_GPT_solve_for_m_l2219_221962

theorem solve_for_m (m x : ℝ) (h1 : 3 * m - 2 * x = 6) (h2 : x = 3) : m = 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_m_l2219_221962


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_part_e_l2219_221993

variable (n : ℤ)

theorem part_a : (n^3 - n) % 3 = 0 :=
  sorry

theorem part_b : (n^5 - n) % 5 = 0 :=
  sorry

theorem part_c : (n^7 - n) % 7 = 0 :=
  sorry

theorem part_d : (n^11 - n) % 11 = 0 :=
  sorry

theorem part_e : (n^13 - n) % 13 = 0 :=
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_part_e_l2219_221993


namespace NUMINAMATH_GPT_c_plus_d_l2219_221936

theorem c_plus_d (a b c d : ℝ) (h1 : a + b = 11) (h2 : b + c = 9) (h3 : a + d = 5) :
  c + d = 3 + b :=
by
  sorry

end NUMINAMATH_GPT_c_plus_d_l2219_221936


namespace NUMINAMATH_GPT_cow_spots_total_l2219_221938

theorem cow_spots_total
  (left_spots : ℕ) (right_spots : ℕ)
  (left_spots_eq : left_spots = 16)
  (right_spots_eq : right_spots = 3 * left_spots + 7) :
  left_spots + right_spots = 71 :=
by
  sorry

end NUMINAMATH_GPT_cow_spots_total_l2219_221938


namespace NUMINAMATH_GPT_stamps_sum_to_n_l2219_221998

noncomputable def selectStamps : Prop :=
  ∀ (n : ℕ) (k : ℕ), n > 0 → 
                      ∃ stamps : List ℕ, 
                      stamps.length = k ∧ 
                      n ≤ stamps.sum ∧ stamps.sum < 2 * k → 
                      ∃ (subset : List ℕ), 
                      subset.sum = n

theorem stamps_sum_to_n : selectStamps := sorry

end NUMINAMATH_GPT_stamps_sum_to_n_l2219_221998


namespace NUMINAMATH_GPT_warriors_games_won_l2219_221900

open Set

-- Define the variables for the number of games each team won
variables (games_L games_H games_W games_F games_R : ℕ)

-- Define the set of possible game scores
def game_scores : Set ℕ := {19, 23, 28, 32, 36}

-- Define the conditions as assumptions
axiom h1 : games_L > games_H
axiom h2 : games_W > games_F
axiom h3 : games_W < games_R
axiom h4 : games_F > 18
axiom h5 : ∃ min_games ∈ game_scores, min_games > games_H ∧ min_games < 20

-- Prove the main statement
theorem warriors_games_won : games_W = 32 :=
sorry

end NUMINAMATH_GPT_warriors_games_won_l2219_221900


namespace NUMINAMATH_GPT_slices_with_both_l2219_221954

theorem slices_with_both (n total_slices pepperoni_slices mushroom_slices other_slices : ℕ)
  (h1 : total_slices = 24) 
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 14)
  (h4 : (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices) :
  n = 5 :=
sorry

end NUMINAMATH_GPT_slices_with_both_l2219_221954


namespace NUMINAMATH_GPT_largest_w_l2219_221930

variable {x y z w : ℝ}

def x_value (x y z w : ℝ) := 
  x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4

theorem largest_w (h : x_value x y z w) : 
  max x (max y (max z w)) = w := 
sorry

end NUMINAMATH_GPT_largest_w_l2219_221930


namespace NUMINAMATH_GPT_train_times_l2219_221976

theorem train_times (t x : ℝ) : 
  (30 * t = 360) ∧ (36 * (t - x) = 360) → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_train_times_l2219_221976


namespace NUMINAMATH_GPT_average_annual_percent_change_l2219_221921

-- Define the initial and final population, and the time period
def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def decade_years : ℕ := 10

-- Define the theorem to find the resulting average percent change per year
theorem average_annual_percent_change
    (P₀ : ℕ := initial_population)
    (P₁₀ : ℕ := final_population)
    (years : ℕ := decade_years) :
    ((P₁₀ - P₀ : ℝ) / P₀ * 100) / years = 7 := by
        sorry

end NUMINAMATH_GPT_average_annual_percent_change_l2219_221921


namespace NUMINAMATH_GPT_possible_values_of_a_l2219_221913

theorem possible_values_of_a (a b c : ℝ) (h1 : a + b + c = 2005) (h2 : (a - 1 = a ∨ a - 1 = b ∨ a - 1 = c) ∧ (b + 1 = a ∨ b + 1 = b ∨ b + 1 = c) ∧ (c ^ 2 = a ∨ c ^ 2 = b ∨ c ^ 2 = c)) :
  a = 1003 ∨ a = 1002.5 :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_l2219_221913


namespace NUMINAMATH_GPT_circles_intersect_l2219_221955

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end NUMINAMATH_GPT_circles_intersect_l2219_221955


namespace NUMINAMATH_GPT_combined_rocket_height_l2219_221975

variable (h1 : ℕ) (h2 : ℕ)

-- Given conditions
def first_rocket_height : ℕ := 500
def second_rocket_height : ℕ := first_rocket_height * 2

-- Prove that the combined height is 1500 ft
theorem combined_rocket_height : first_rocket_height + second_rocket_height = 1500 := by
  sorry

end NUMINAMATH_GPT_combined_rocket_height_l2219_221975


namespace NUMINAMATH_GPT_infinitely_many_primes_of_form_6n_plus_5_l2219_221969

theorem infinitely_many_primes_of_form_6n_plus_5 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p % 6 = 5 :=
sorry

end NUMINAMATH_GPT_infinitely_many_primes_of_form_6n_plus_5_l2219_221969


namespace NUMINAMATH_GPT_joyce_gave_apples_l2219_221942

theorem joyce_gave_apples : 
  ∀ (initial_apples final_apples given_apples : ℕ), (initial_apples = 75) ∧ (final_apples = 23) → (given_apples = initial_apples - final_apples) → (given_apples = 52) :=
by
  intros
  sorry

end NUMINAMATH_GPT_joyce_gave_apples_l2219_221942


namespace NUMINAMATH_GPT_probability_x_lt_2y_l2219_221945

noncomputable def rectangle := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }

noncomputable def region_of_interest := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 2 * p.2 }

noncomputable def area_rectangle := 6 * 2

noncomputable def area_trapezoid := (1 / 2) * (4 + 6) * 2

theorem probability_x_lt_2y : (area_trapezoid / area_rectangle) = 5 / 6 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_probability_x_lt_2y_l2219_221945


namespace NUMINAMATH_GPT_value_of_expression_l2219_221963

variable {a : ℕ → ℤ}
variable {a₁ a₄ a₁₀ a₁₆ a₁₉ : ℤ}
variable {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + d * n

-- Given conditions
axiom h₀ : arithmetic_sequence a a₁ d
axiom h₁ : a₁ + a₄ + a₁₀ + a₁₆ + a₁₉ = 150

-- Prove the required statement
theorem value_of_expression :
  a 20 - a 26 + a 16 = 30 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l2219_221963


namespace NUMINAMATH_GPT_exists_unique_line_prime_x_intercept_positive_y_intercept_l2219_221917

/-- There is exactly one line with x-intercept that is a prime number less than 10 and y-intercept that is a positive integer not equal to 5, which passes through the point (5, 4) -/
theorem exists_unique_line_prime_x_intercept_positive_y_intercept (x_intercept : ℕ) (hx : Nat.Prime x_intercept) (hx_lt_10 : x_intercept < 10) (y_intercept : ℕ) (hy_pos : y_intercept > 0) (hy_ne_5 : y_intercept ≠ 5) :
  (∃ (a b : ℕ), a = x_intercept ∧ b = y_intercept ∧ (∀ p q : ℕ, p = 5 ∧ q = 4 → (p / a) + (q / b) = 1)) :=
sorry

end NUMINAMATH_GPT_exists_unique_line_prime_x_intercept_positive_y_intercept_l2219_221917


namespace NUMINAMATH_GPT_proportional_segments_l2219_221990

theorem proportional_segments (a b c d : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) (h4 : a / b = c / d) : d = 3 / 2 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_proportional_segments_l2219_221990


namespace NUMINAMATH_GPT_cost_of_five_trip_ticket_l2219_221903

-- Variables for the costs of the tickets
variables (x y z : ℕ)

-- Conditions from the problem
def condition1 : Prop := 5 * x > y
def condition2 : Prop := 4 * y > z
def condition3 : Prop := z + 3 * y = 33
def condition4 : Prop := 20 + 3 * 5 = 35

-- The theorem to prove
theorem cost_of_five_trip_ticket (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z y) (h4 : condition4) : y = 5 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_five_trip_ticket_l2219_221903


namespace NUMINAMATH_GPT_elena_marco_sum_ratio_l2219_221997

noncomputable def sum_odds (n : Nat) : Nat := (n / 2 + 1) * n

noncomputable def sum_integers (n : Nat) : Nat := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odds 499) / (sum_integers 250) = 2 :=
by
  sorry

end NUMINAMATH_GPT_elena_marco_sum_ratio_l2219_221997


namespace NUMINAMATH_GPT_trigonometric_identity_x1_trigonometric_identity_x2_l2219_221949

noncomputable def x1 (n : ℤ) : ℝ := (2 * n + 1) * (Real.pi / 4)
noncomputable def x2 (k : ℤ) : ℝ := ((-1)^(k + 1)) * (Real.pi / 8) + k * (Real.pi / 2)

theorem trigonometric_identity_x1 (n : ℤ) : 
  (Real.cos (4 * x1 n) * Real.cos (Real.pi + 2 * x1 n) - 
   Real.sin (2 * x1 n) * Real.cos (Real.pi / 2 - 4 * x1 n)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x1 n) := 
by
  sorry

theorem trigonometric_identity_x2 (k : ℤ) : 
  (Real.cos (4 * x2 k) * Real.cos (Real.pi + 2 * x2 k) - 
   Real.sin (2 * x2 k) * Real.cos (Real.pi / 2 - 4 * x2 k)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x2 k) := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_x1_trigonometric_identity_x2_l2219_221949


namespace NUMINAMATH_GPT_D_72_l2219_221910

/-- D(n) denotes the number of ways of writing the positive integer n
    as a product n = f1 * f2 * ... * fk, where k ≥ 1, the fi are integers
    strictly greater than 1, and the order in which the factors are
    listed matters. -/
def D (n : ℕ) : ℕ := sorry

theorem D_72 : D 72 = 43 := sorry

end NUMINAMATH_GPT_D_72_l2219_221910


namespace NUMINAMATH_GPT_find_max_min_find_angle_C_l2219_221907

open Real

noncomputable def f (x : ℝ) : ℝ :=
  12 * sin (x + π / 6) * cos x - 3

theorem find_max_min (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) :
  let fx := f x 
  (∀ a, a = abs (fx - 6)) -> (∀ b, b = abs (fx - 3)) -> fx = 6 ∨ fx = 3 := sorry

theorem find_angle_C (AC BC CD : ℝ) (hAC : AC = 6) (hBC : BC = 3) (hCD : CD = 2 * sqrt 2) :
  ∃ C : ℝ, C = π / 2 := sorry

end NUMINAMATH_GPT_find_max_min_find_angle_C_l2219_221907


namespace NUMINAMATH_GPT_average_of_second_pair_l2219_221927

theorem average_of_second_pair (S : ℝ) (S1 : ℝ) (S3 : ℝ) (S2 : ℝ) (avg : ℝ) :
  (S / 6 = 3.95) →
  (S1 / 2 = 3.8) →
  (S3 / 2 = 4.200000000000001) →
  (S = S1 + S2 + S3) →
  (avg = S2 / 2) →
  avg = 3.85 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_average_of_second_pair_l2219_221927


namespace NUMINAMATH_GPT_factorize_expression_l2219_221983

theorem factorize_expression (x : ℝ) : 
  x^4 + 324 = (x^2 - 18 * x + 162) * (x^2 + 18 * x + 162) := 
sorry

end NUMINAMATH_GPT_factorize_expression_l2219_221983


namespace NUMINAMATH_GPT_simplify_fraction_l2219_221987

noncomputable def simplified_expression (x y : ℝ) : ℝ :=
  (x^2 - (4 / y)) / (y^2 - (4 / x))

theorem simplify_fraction {x y : ℝ} (h : x * y ≠ 4) :
  simplified_expression x y = x / y := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2219_221987


namespace NUMINAMATH_GPT_point_coordinates_with_respect_to_origin_l2219_221904

theorem point_coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_with_respect_to_origin_l2219_221904


namespace NUMINAMATH_GPT_add_words_to_meet_requirement_l2219_221932

-- Definitions required by the problem
def yvonne_words : ℕ := 400
def janna_extra_words : ℕ := 150
def words_removed : ℕ := 20
def requirement : ℕ := 1000

-- Derived values based on the conditions
def janna_words : ℕ := yvonne_words + janna_extra_words
def initial_words : ℕ := yvonne_words + janna_words
def words_after_removal : ℕ := initial_words - words_removed
def words_added : ℕ := 2 * words_removed
def total_words_after_editing : ℕ := words_after_removal + words_added
def words_to_add : ℕ := requirement - total_words_after_editing

-- The theorem to prove
theorem add_words_to_meet_requirement : words_to_add = 30 := by
  sorry

end NUMINAMATH_GPT_add_words_to_meet_requirement_l2219_221932


namespace NUMINAMATH_GPT_determine_a_if_fx_odd_l2219_221966

theorem determine_a_if_fx_odd (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = 2^x + a * 2^(-x)) (h2 : ∀ x, f (-x) = -f x) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_if_fx_odd_l2219_221966


namespace NUMINAMATH_GPT_smallest_sphere_radius_l2219_221929

noncomputable def radius_smallest_sphere : ℝ := 2 * Real.sqrt 3 + 2

theorem smallest_sphere_radius (r : ℝ) (h : r = 2) : radius_smallest_sphere = 2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_GPT_smallest_sphere_radius_l2219_221929


namespace NUMINAMATH_GPT_julia_cakes_remaining_l2219_221920

namespace CakeProblem

def cakes_per_day : ℕ := 5 - 1
def days_baked : ℕ := 6
def total_cakes_baked : ℕ := cakes_per_day * days_baked
def days_clifford_eats : ℕ := days_baked / 2
def cakes_eaten_by_clifford : ℕ := days_clifford_eats

theorem julia_cakes_remaining : total_cakes_baked - cakes_eaten_by_clifford = 21 :=
by
  -- proof goes here
  sorry

end CakeProblem

end NUMINAMATH_GPT_julia_cakes_remaining_l2219_221920


namespace NUMINAMATH_GPT_john_sells_percentage_of_newspapers_l2219_221988

theorem john_sells_percentage_of_newspapers
    (n_newspapers : ℕ)
    (selling_price : ℝ)
    (cost_price_discount : ℝ)
    (profit : ℝ)
    (sold_percentage : ℝ)
    (h1 : n_newspapers = 500)
    (h2 : selling_price = 2)
    (h3 : cost_price_discount = 0.75)
    (h4 : profit = 550)
    (h5 : sold_percentage = 80) : 
    ( ∃ (sold_n : ℕ), 
      sold_n / n_newspapers * 100 = sold_percentage ∧
      sold_n * selling_price = 
        n_newspapers * selling_price * (1 - cost_price_discount) + profit) :=
by
  sorry

end NUMINAMATH_GPT_john_sells_percentage_of_newspapers_l2219_221988


namespace NUMINAMATH_GPT_sum_of_consecutive_odds_l2219_221970

theorem sum_of_consecutive_odds (N1 N2 N3 : ℕ) (h1 : N1 % 2 = 1) (h2 : N2 % 2 = 1) (h3 : N3 % 2 = 1)
  (h_consec1 : N2 = N1 + 2) (h_consec2 : N3 = N2 + 2) (h_max : N3 = 27) : 
  N1 + N2 + N3 = 75 := by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_odds_l2219_221970


namespace NUMINAMATH_GPT_negation_universal_statement_l2219_221908

theorem negation_universal_statement :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by sorry

end NUMINAMATH_GPT_negation_universal_statement_l2219_221908


namespace NUMINAMATH_GPT_tickets_spent_correct_l2219_221991

/-- Tom won 32 tickets playing 'whack a mole'. -/
def tickets_whack_mole : ℕ := 32

/-- Tom won 25 tickets playing 'skee ball'. -/
def tickets_skee_ball : ℕ := 25

/-- Tom is left with 50 tickets after spending some on a hat. -/
def tickets_left : ℕ := 50

/-- The total number of tickets Tom won from both games. -/
def tickets_total : ℕ := tickets_whack_mole + tickets_skee_ball

/-- The number of tickets Tom spent on the hat. -/
def tickets_spent : ℕ := tickets_total - tickets_left

-- Prove that the number of tickets Tom spent on the hat is 7.
theorem tickets_spent_correct : tickets_spent = 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tickets_spent_correct_l2219_221991


namespace NUMINAMATH_GPT_remainder_when_divided_l2219_221953

theorem remainder_when_divided (P D Q R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D * D') = R + R' * D :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l2219_221953


namespace NUMINAMATH_GPT_cube_edge_length_l2219_221924

-- Define the edge length 'a'
variable (a : ℝ)

-- Given conditions: 6a^2 = 24
theorem cube_edge_length (h : 6 * a^2 = 24) : a = 2 :=
by {
  -- The actual proof would go here, but we use sorry to skip it as per instructions.
  sorry
}

end NUMINAMATH_GPT_cube_edge_length_l2219_221924


namespace NUMINAMATH_GPT_aziz_age_l2219_221947

-- Definitions of the conditions
def year_moved : ℕ := 1982
def years_before_birth : ℕ := 3
def current_year : ℕ := 2021

-- Prove the main statement
theorem aziz_age : current_year - (year_moved + years_before_birth) = 36 :=
by
  sorry

end NUMINAMATH_GPT_aziz_age_l2219_221947


namespace NUMINAMATH_GPT_find_remainder_l2219_221940

def mod_condition : Prop :=
  (764251 % 31 = 5) ∧
  (1095223 % 31 = 6) ∧
  (1487719 % 31 = 1) ∧
  (263311 % 31 = 0) ∧
  (12097 % 31 = 25) ∧
  (16817 % 31 = 26) ∧
  (23431 % 31 = 0) ∧
  (305643 % 31 = 20)

theorem find_remainder (h : mod_condition) : 
  ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_remainder_l2219_221940


namespace NUMINAMATH_GPT_var_power_eight_l2219_221999

variable (k j : ℝ)
variable {x y z : ℝ}

theorem var_power_eight (hx : x = k * y^4) (hy : y = j * z^2) : ∃ c : ℝ, x = c * z^8 :=
by
  sorry

end NUMINAMATH_GPT_var_power_eight_l2219_221999


namespace NUMINAMATH_GPT_trolley_length_l2219_221978

theorem trolley_length (L F : ℝ) (h1 : 4 * L + 3 * F = 108) (h2 : 10 * L + 9 * F = 168) : L = 78 := 
by
  sorry

end NUMINAMATH_GPT_trolley_length_l2219_221978


namespace NUMINAMATH_GPT_measure_of_angle_F_l2219_221916

theorem measure_of_angle_F (D E F : ℝ) (h₁ : D = 85) (h₂ : E = 4 * F + 15) (h₃ : D + E + F = 180) : 
  F = 16 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_F_l2219_221916


namespace NUMINAMATH_GPT_evaluate_complex_modulus_l2219_221958

namespace ComplexProblem

open Complex

theorem evaluate_complex_modulus : 
  abs ((1 / 2 : ℂ) - (3 / 8) * Complex.I) = 5 / 8 :=
by
  sorry

end ComplexProblem

end NUMINAMATH_GPT_evaluate_complex_modulus_l2219_221958


namespace NUMINAMATH_GPT_coffee_shop_ratio_l2219_221911

theorem coffee_shop_ratio (morning_usage afternoon_multiplier weekly_usage days_per_week : ℕ) (r : ℕ) 
  (h_morning : morning_usage = 3)
  (h_afternoon : afternoon_multiplier = 3)
  (h_weekly : weekly_usage = 126)
  (h_days : days_per_week = 7):
  weekly_usage = days_per_week * (morning_usage + afternoon_multiplier * morning_usage + r * morning_usage) →
  r = 2 :=
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_coffee_shop_ratio_l2219_221911


namespace NUMINAMATH_GPT_sum_of_three_pairwise_rel_prime_integers_l2219_221939

theorem sum_of_three_pairwise_rel_prime_integers (a b c : ℕ)
  (h1: 1 < a) (h2: 1 < b) (h3: 1 < c)
  (prod: a * b * c = 216000)
  (rel_prime_ab : Nat.gcd a b = 1)
  (rel_prime_ac : Nat.gcd a c = 1)
  (rel_prime_bc : Nat.gcd b c = 1) : 
  a + b + c = 184 := 
sorry

end NUMINAMATH_GPT_sum_of_three_pairwise_rel_prime_integers_l2219_221939


namespace NUMINAMATH_GPT_inequality_holds_for_positive_x_l2219_221979

theorem inequality_holds_for_positive_x (x : ℝ) (h : 0 < x) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_positive_x_l2219_221979


namespace NUMINAMATH_GPT_find_M_for_same_asymptotes_l2219_221961

theorem find_M_for_same_asymptotes :
  ∃ M : ℝ, ∀ x y : ℝ,
    (x^2 / 16 - y^2 / 25 = 1) →
    (y^2 / 50 - x^2 / M = 1) →
    (∀ x : ℝ, ∃ k : ℝ, y = k * x ↔ k = 5 / 4) →
    M = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_M_for_same_asymptotes_l2219_221961


namespace NUMINAMATH_GPT_remainder_when_55_times_57_divided_by_8_l2219_221902

theorem remainder_when_55_times_57_divided_by_8 :
  (55 * 57) % 8 = 7 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_remainder_when_55_times_57_divided_by_8_l2219_221902


namespace NUMINAMATH_GPT_amy_tickets_initial_l2219_221937

theorem amy_tickets_initial (x : ℕ) (h1 : x + 21 = 54) : x = 33 :=
by sorry

end NUMINAMATH_GPT_amy_tickets_initial_l2219_221937


namespace NUMINAMATH_GPT_find_target_number_l2219_221914

theorem find_target_number : ∃ n ≥ 0, (∀ k < 5, ∃ m, 0 ≤ m ∧ m ≤ n ∧ m % 11 = 3 ∧ m = 3 + k * 11) ∧ n = 47 :=
by
  sorry

end NUMINAMATH_GPT_find_target_number_l2219_221914


namespace NUMINAMATH_GPT_exists_monochromatic_triangle_l2219_221989

theorem exists_monochromatic_triangle (points : Fin 6 → Point) (color : (Point × Point) → Color) :
  ∃ (a b c : Point), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (color (a, b) = color (b, c) ∧ color (b, c) = color (c, a)) :=
by
  sorry

end NUMINAMATH_GPT_exists_monochromatic_triangle_l2219_221989


namespace NUMINAMATH_GPT_find_a_b_extreme_points_l2219_221971

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : deriv (f a b) 2 = 0) (h₃ : f a b 2 = 8) : 
  a = 4 ∧ b = 24 :=
by
  sorry

noncomputable def f_deriv (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

theorem extreme_points (a : ℝ) (h₁ : a > 0) : 
  (∃ x: ℝ, f_deriv a x = 0 ∧ 
      ((x = -Real.sqrt a ∧ f a 24 x = 40) ∨ 
       (x = Real.sqrt a ∧ f a 24 x = 16))) := 
by
  sorry

end NUMINAMATH_GPT_find_a_b_extreme_points_l2219_221971


namespace NUMINAMATH_GPT_least_number_to_subtract_l2219_221992

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : 
  ∃ k, (n - k) % 10 = 0 ∧ k = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l2219_221992


namespace NUMINAMATH_GPT_correct_system_of_equations_l2219_221972

-- Define the given problem conditions.
def cost_doll : ℝ := 60
def cost_keychain : ℝ := 20
def total_cost : ℝ := 5000

-- Define the condition that each gift set needs 1 doll and 2 keychains.
def gift_set_relation (x y : ℝ) : Prop := 2 * x = y

-- Define the system of equations representing the problem.
def system_of_equations (x y : ℝ) : Prop :=
  2 * x = y ∧
  60 * x + 20 * y = total_cost

-- State the theorem to prove that the given system correctly models the problem.
theorem correct_system_of_equations (x y : ℝ) :
  system_of_equations x y ↔ (2 * x = y ∧ 60 * x + 20 * y = 5000) :=
by sorry

end NUMINAMATH_GPT_correct_system_of_equations_l2219_221972


namespace NUMINAMATH_GPT_simplify_fraction_l2219_221905

-- Define the fractions and the product
def fraction1 : ℚ := 18 / 11
def fraction2 : ℚ := -42 / 45
def product : ℚ := 15 * fraction1 * fraction2

-- State the theorem to prove the correctness of the simplification
theorem simplify_fraction : product = -23 + 1 / 11 :=
by
  -- Adding this as a placeholder. The proof would go here.
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2219_221905


namespace NUMINAMATH_GPT_expression_value_l2219_221995

theorem expression_value :
  (6^2 - 3^2)^4 = 531441 := by
  -- Proof steps were omitted
  sorry

end NUMINAMATH_GPT_expression_value_l2219_221995


namespace NUMINAMATH_GPT_geometric_difference_l2219_221935

def is_geometric_sequence (n : ℕ) : Prop :=
∃ (a b c : ℤ), n = a * 100 + b * 10 + c ∧
a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
(b^2 = a * c) ∧
(b % 2 = 1)

theorem geometric_difference :
  ∃ (n1 n2 : ℕ), is_geometric_sequence n1 ∧ is_geometric_sequence n2 ∧
  n2 > n1 ∧
  n2 - n1 = 220 :=
sorry

end NUMINAMATH_GPT_geometric_difference_l2219_221935


namespace NUMINAMATH_GPT_johns_weekly_earnings_increase_l2219_221919

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_johns_weekly_earnings_increase_l2219_221919


namespace NUMINAMATH_GPT_probability_heads_l2219_221982

variable (p : ℝ)
variable (h1 : 0 ≤ p)
variable (h2 : p ≤ 1)
variable (h3 : p * (1 - p) ^ 4 = 0.03125)

theorem probability_heads :
  p = 0.5 :=
sorry

end NUMINAMATH_GPT_probability_heads_l2219_221982


namespace NUMINAMATH_GPT_solution_set_of_inequality_eq_l2219_221915

noncomputable def inequality_solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem solution_set_of_inequality_eq :
  {x : ℝ | (2 * x) / (x - 1) < 1} = inequality_solution_set := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_eq_l2219_221915
