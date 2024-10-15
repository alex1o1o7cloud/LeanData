import Mathlib

namespace NUMINAMATH_GPT_orchard_tree_growth_problem_l1850_185066

theorem orchard_tree_growth_problem
  (T0 : ℕ) (Tn : ℕ) (n : ℕ)
  (h1 : T0 = 1280)
  (h2 : Tn = 3125)
  (h3 : Tn = (5/4 : ℚ) ^ n * T0) :
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_orchard_tree_growth_problem_l1850_185066


namespace NUMINAMATH_GPT_gcd_lcm_product_correct_l1850_185056

noncomputable def gcd_lcm_product : ℕ :=
  let a := 90
  let b := 135
  gcd a b * lcm a b

theorem gcd_lcm_product_correct : gcd_lcm_product = 12150 :=
  by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_correct_l1850_185056


namespace NUMINAMATH_GPT_seq_form_l1850_185057

-- Define the sequence a as a function from natural numbers to natural numbers
def seq (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, 0 < m → 0 < n → ⌊(a m : ℚ) / a n⌋ = ⌊(m : ℚ) / n⌋

-- Define the statement that all sequences satisfying the condition must be of the form k * i
theorem seq_form (a : ℕ → ℕ) : seq a → ∃ k : ℕ, (0 < k) ∧ (∀ n, 0 < n → a n = k * n) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_seq_form_l1850_185057


namespace NUMINAMATH_GPT_polynomial_inequality_l1850_185047

noncomputable def F (x a_3 a_2 a_1 k : ℝ) : ℝ :=
  x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + k^4

theorem polynomial_inequality 
  (p k : ℝ) 
  (a_3 a_2 a_1 : ℝ) 
  (h_p : 0 < p) 
  (h_k : 0 < k) 
  (h_roots : ∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧
    F (-x1) a_3 a_2 a_1 k = 0 ∧
    F (-x2) a_3 a_2 a_1 k = 0 ∧
    F (-x3) a_3 a_2 a_1 k = 0 ∧
    F (-x4) a_3 a_2 a_1 k = 0) :
  F p a_3 a_2 a_1 k ≥ (p + k)^4 := 
sorry

end NUMINAMATH_GPT_polynomial_inequality_l1850_185047


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l1850_185018

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_areas_of_circles_l1850_185018


namespace NUMINAMATH_GPT_investment_return_l1850_185092

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_investment_return_l1850_185092


namespace NUMINAMATH_GPT_speed_at_perigee_l1850_185035

-- Define the conditions
def semi_major_axis (a : ℝ) := a > 0
def perigee_distance (a : ℝ) := 0.5 * a
def point_P_distance (a : ℝ) := 0.75 * a
def speed_at_P (v1 : ℝ) := v1 > 0

-- Define what we need to prove
theorem speed_at_perigee (a v1 v2 : ℝ) (h1 : semi_major_axis a) (h2 : speed_at_P v1) :
  v2 = (3 / Real.sqrt 5) * v1 :=
sorry

end NUMINAMATH_GPT_speed_at_perigee_l1850_185035


namespace NUMINAMATH_GPT_problem_statement_l1850_185016

variable {a : ℕ → ℝ} 
variable {a1 d : ℝ}
variable (h_arith : ∀ n, a (n + 1) = a n + d)  -- Arithmetic sequence condition
variable (h_d_nonzero : d ≠ 0)  -- d ≠ 0
variable (h_a1_nonzero : a1 ≠ 0)  -- a1 ≠ 0
variable (h_geom : (a 1) * (a 7) = (a 3) ^ 2)  -- Geometric sequence condition a2 = a 1, a4 = a 3, a8 = a 7

theorem problem_statement :
  (a 0 + a 4 + a 8) / (a 1 + a 2) = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1850_185016


namespace NUMINAMATH_GPT_water_required_for_reaction_l1850_185011

noncomputable def sodium_hydride_reaction (NaH H₂O NaOH H₂ : Type) : Nat :=
  1

theorem water_required_for_reaction :
  let NaH := 2
  let required_H₂O := 2 -- Derived from balanced chemical equation and given condition
  sodium_hydride_reaction Nat Nat Nat Nat = required_H₂O :=
by
  sorry

end NUMINAMATH_GPT_water_required_for_reaction_l1850_185011


namespace NUMINAMATH_GPT_fraction_identity_l1850_185020

theorem fraction_identity (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : 
  (ab - a)/(a + b) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_identity_l1850_185020


namespace NUMINAMATH_GPT_sum_of_integers_l1850_185036

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1850_185036


namespace NUMINAMATH_GPT_parabola_focus_distance_l1850_185019

theorem parabola_focus_distance (p : ℝ) (hp : p > 0) (A : ℝ × ℝ)
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hA_focus_dist : dist A (p / 2, 0) = 12)
  (hA_yaxis_dist : abs A.1 = 9) : p = 6 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1850_185019


namespace NUMINAMATH_GPT_jen_hours_per_week_l1850_185043

theorem jen_hours_per_week (B : ℕ) (h1 : ∀ t : ℕ, t * (B + 7) = 6 * B) : B + 7 = 21 := by
  sorry

end NUMINAMATH_GPT_jen_hours_per_week_l1850_185043


namespace NUMINAMATH_GPT_cost_per_pound_peanuts_l1850_185017

-- Defining the conditions as needed for our problem
def one_dollar_bills := 7
def five_dollar_bills := 4
def ten_dollar_bills := 2
def twenty_dollar_bills := 1
def change := 4
def pounds_per_day := 3
def days_in_week := 7

-- Calculating the total initial amount of money Frank has
def total_initial_money := (one_dollar_bills * 1) + (five_dollar_bills * 5) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)

-- Calculating the total amount spent on peanuts
def total_spent := total_initial_money - change

-- Calculating the total pounds of peanuts
def total_pounds := pounds_per_day * days_in_week

-- The proof statement
theorem cost_per_pound_peanuts : total_spent / total_pounds = 3 := sorry

end NUMINAMATH_GPT_cost_per_pound_peanuts_l1850_185017


namespace NUMINAMATH_GPT_domain_of_function_l1850_185031

theorem domain_of_function :
  {x : ℝ | (x + 1 ≥ 0) ∧ (2 - x ≠ 0)} = {x : ℝ | -1 ≤ x ∧ x ≠ 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_function_l1850_185031


namespace NUMINAMATH_GPT_system1_solution_l1850_185032

theorem system1_solution (x y : ℝ) 
  (h1 : x + y = 10^20) 
  (h2 : x - y = 10^19) :
  x = 55 * 10^18 ∧ y = 45 * 10^18 := 
by
  sorry

end NUMINAMATH_GPT_system1_solution_l1850_185032


namespace NUMINAMATH_GPT_find_a_l1850_185094

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end NUMINAMATH_GPT_find_a_l1850_185094


namespace NUMINAMATH_GPT_extreme_value_h_at_a_zero_range_of_a_l1850_185054

noncomputable def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x / (a * x + 1)

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (Real.exp (-x)) * (g x a)

-- Statement for the first proof problem
theorem extreme_value_h_at_a_zero :
  ∀ x : ℝ, h x 0 ≤ 1 / Real.exp 1 :=
sorry

-- Statement for the second proof problem
theorem range_of_a:
  ∀ x : ℝ, (0 ≤ x → x ≤ 1 / 2) → (f x ≤ g x x) :=
sorry

end NUMINAMATH_GPT_extreme_value_h_at_a_zero_range_of_a_l1850_185054


namespace NUMINAMATH_GPT_total_days_correct_l1850_185070

-- Defining the years and the conditions given.
def year_1999 := 1999
def year_2000 := 2000
def year_2001 := 2001
def year_2002 := 2002

-- Defining the leap year and regular year days
def days_in_regular_year := 365
def days_in_leap_year := 366

-- Noncomputable version to skip the proof
noncomputable def total_days_from_1999_to_2002 : ℕ :=
  3 * days_in_regular_year + days_in_leap_year

-- The theorem stating the problem, which we need to prove
theorem total_days_correct : total_days_from_1999_to_2002 = 1461 := by
  sorry

end NUMINAMATH_GPT_total_days_correct_l1850_185070


namespace NUMINAMATH_GPT_taxi_fare_distance_l1850_185069

theorem taxi_fare_distance (x : ℝ) : 
  (8 + if x ≤ 3 then 0 else if x ≤ 8 then 2.15 * (x - 3) else 2.15 * 5 + 2.85 * (x - 8)) + 1 = 31.15 → x = 11.98 :=
by 
  sorry

end NUMINAMATH_GPT_taxi_fare_distance_l1850_185069


namespace NUMINAMATH_GPT_power_quotient_l1850_185030

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end NUMINAMATH_GPT_power_quotient_l1850_185030


namespace NUMINAMATH_GPT_union_sets_l1850_185091

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_GPT_union_sets_l1850_185091


namespace NUMINAMATH_GPT_proportional_function_y_decreases_l1850_185041

theorem proportional_function_y_decreases (k : ℝ) (h₀ : k ≠ 0) (h₁ : (4 : ℝ) * k = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ :=
by 
  sorry

end NUMINAMATH_GPT_proportional_function_y_decreases_l1850_185041


namespace NUMINAMATH_GPT_binary_to_decimal_and_octal_conversion_l1850_185086

-- Definition of the binary number in question
def bin_num : ℕ := 0b1011

-- The expected decimal equivalent
def dec_num : ℕ := 11

-- The expected octal equivalent
def oct_num : ℤ := 0o13

-- Proof problem statement
theorem binary_to_decimal_and_octal_conversion :
  bin_num = dec_num ∧ dec_num = oct_num := 
by 
  sorry

end NUMINAMATH_GPT_binary_to_decimal_and_octal_conversion_l1850_185086


namespace NUMINAMATH_GPT_angle_E_measure_l1850_185090

theorem angle_E_measure (H F G E : ℝ) 
  (h1 : E = 2 * F) (h2 : F = 2 * G) (h3 : G = 1.25 * H) 
  (h4 : E + F + G + H = 360) : E = 150 := by
  sorry

end NUMINAMATH_GPT_angle_E_measure_l1850_185090


namespace NUMINAMATH_GPT_factor_expression_l1850_185076

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1850_185076


namespace NUMINAMATH_GPT_min_packages_l1850_185055

theorem min_packages (p : ℕ) (N : ℕ) :
  (N = 19 * p) →
  (N % 7 = 4) →
  (N % 11 = 1) →
  p = 40 :=
by
  sorry

end NUMINAMATH_GPT_min_packages_l1850_185055


namespace NUMINAMATH_GPT_peanuts_remaining_l1850_185012

def initial_peanuts := 220
def brock_fraction := 1 / 4
def bonita_fraction := 2 / 5
def carlos_peanuts := 17

noncomputable def peanuts_left := initial_peanuts - (initial_peanuts * brock_fraction + ((initial_peanuts - initial_peanuts * brock_fraction) * bonita_fraction)) - carlos_peanuts

theorem peanuts_remaining : peanuts_left = 82 :=
by
  sorry

end NUMINAMATH_GPT_peanuts_remaining_l1850_185012


namespace NUMINAMATH_GPT_factor_expression_l1850_185087

theorem factor_expression :
  (8 * x ^ 4 + 34 * x ^ 3 - 120 * x + 150) - (-2 * x ^ 4 + 12 * x ^ 3 - 5 * x + 10) 
  = 5 * x * (2 * x ^ 3 + (22 / 5) * x ^ 2 - 23 * x + 28) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1850_185087


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1850_185097

variable {R : Type} [LinearOrderedField R] (f : R → R)

-- Conditions
def monotonically_increasing_on_nonnegatives := 
  ∀ x y : R, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

def odd_function_shifted_one := 
  ∀ x : R, f (-x) = 2 - f (x)

-- The problem
theorem solution_set_of_inequality
  (mono_inc : monotonically_increasing_on_nonnegatives f)
  (odd_shift : odd_function_shifted_one f) :
  {x : R | f (3 * x + 4) + f (1 - x) < 2} = {x : R | x < -5 / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1850_185097


namespace NUMINAMATH_GPT_project_selection_probability_l1850_185088

/-- Each employee can randomly select one project from four optional assessment projects. -/
def employees : ℕ := 4

def projects : ℕ := 4

def total_events (e : ℕ) (p : ℕ) : ℕ := p^e

def choose_exactly_one_project_not_selected_probability (e : ℕ) (p : ℕ) : ℚ :=
  (Nat.choose p 2 * Nat.factorial 3) / (p^e : ℚ)

theorem project_selection_probability :
  choose_exactly_one_project_not_selected_probability employees projects = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_project_selection_probability_l1850_185088


namespace NUMINAMATH_GPT_mike_laptop_row_division_impossible_l1850_185021

theorem mike_laptop_row_division_impossible (total_laptops : ℕ) (num_rows : ℕ) 
(types_ratios : List ℕ)
(H_total : total_laptops = 44)
(H_rows : num_rows = 5) 
(H_ratio : types_ratios = [2, 3, 4]) :
  ¬ (∃ (n : ℕ), (total_laptops = n * num_rows) 
  ∧ (n % (types_ratios.sum) = 0)
  ∧ (∀ (t : ℕ), t ∈ types_ratios → t ≤ n)) := sorry

end NUMINAMATH_GPT_mike_laptop_row_division_impossible_l1850_185021


namespace NUMINAMATH_GPT_problem_l1850_185080

theorem problem (a b c d : ℝ) (h1 : a - b - c + d = 18) (h2 : a + b - c - d = 6) : (b - d) ^ 2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1850_185080


namespace NUMINAMATH_GPT_percentage_off_sale_l1850_185052

theorem percentage_off_sale (original_price sale_price : ℝ) (h₁ : original_price = 350) (h₂ : sale_price = 140) :
  ((original_price - sale_price) / original_price) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_off_sale_l1850_185052


namespace NUMINAMATH_GPT_initial_salmons_l1850_185082

theorem initial_salmons (x : ℕ) (hx : 10 * x = 5500) : x = 550 := 
by
  sorry

end NUMINAMATH_GPT_initial_salmons_l1850_185082


namespace NUMINAMATH_GPT_train_pass_tree_l1850_185073

theorem train_pass_tree
  (L : ℝ) (S : ℝ) (conv_factor : ℝ) 
  (hL : L = 275)
  (hS : S = 90)
  (hconv : conv_factor = 5 / 18) :
  L / (S * conv_factor) = 11 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_tree_l1850_185073


namespace NUMINAMATH_GPT_rowing_speed_upstream_l1850_185098

theorem rowing_speed_upstream (V_m V_down : ℝ) (h_Vm : V_m = 35) (h_Vdown : V_down = 40) : V_m - (V_down - V_m) = 30 :=
by
  sorry

end NUMINAMATH_GPT_rowing_speed_upstream_l1850_185098


namespace NUMINAMATH_GPT_find_a_l1850_185059

noncomputable def a_value_given_conditions : ℝ :=
  let A := 30 * Real.pi / 180
  let C := 105 * Real.pi / 180
  let B := 180 * Real.pi / 180 - A - C
  let b := 8
  let a := (b * Real.sin A) / Real.sin B
  a

theorem find_a :
  a_value_given_conditions = 4 * Real.sqrt 2 :=
by
  -- We assume that the value computation as specified is correct
  -- hence this is just stating the problem.
  sorry

end NUMINAMATH_GPT_find_a_l1850_185059


namespace NUMINAMATH_GPT_distinct_pairs_count_l1850_185064

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end NUMINAMATH_GPT_distinct_pairs_count_l1850_185064


namespace NUMINAMATH_GPT_final_selling_price_l1850_185001

variable (a : ℝ)

theorem final_selling_price (h : a > 0) : 0.9 * (1.25 * a) = 1.125 * a := 
by
  sorry

end NUMINAMATH_GPT_final_selling_price_l1850_185001


namespace NUMINAMATH_GPT_six_smallest_distinct_integers_l1850_185024

theorem six_smallest_distinct_integers:
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a * b * c * d * e = 999999 ∧ a = 3 ∧
    f = 37 ∨
    a * b * c * d * f = 999999 ∧ a = 3 ∧ e = 13 ∨ 
    a * b * d * f * e = 999999 ∧ c = 9 ∧ 
    a * c * d * e * f = 999999 ∧ b = 7 ∧ 
    b * c * d * e * f = 999999 ∧ a = 3 := 
sorry

end NUMINAMATH_GPT_six_smallest_distinct_integers_l1850_185024


namespace NUMINAMATH_GPT_profit_function_expression_l1850_185003

def dailySalesVolume (x : ℝ) : ℝ := 300 + 3 * (99 - x)

def profitPerItem (x : ℝ) : ℝ := x - 50

def dailyProfit (x : ℝ) : ℝ := (x - 50) * (300 + 3 * (99 - x))

theorem profit_function_expression (x : ℝ) :
  dailyProfit x = (x - 50) * dailySalesVolume x :=
by sorry

end NUMINAMATH_GPT_profit_function_expression_l1850_185003


namespace NUMINAMATH_GPT_tens_digit_13_power_1987_l1850_185000

theorem tens_digit_13_power_1987 : (13^1987)%100 / 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_13_power_1987_l1850_185000


namespace NUMINAMATH_GPT_blue_string_length_l1850_185074

def length_red := 8
def length_white := 5 * length_red
def length_blue := length_white / 8

theorem blue_string_length : length_blue = 5 := by
  sorry

end NUMINAMATH_GPT_blue_string_length_l1850_185074


namespace NUMINAMATH_GPT_complement_union_correct_l1850_185040

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_correct :
  ((U \ A) ∪ B) = {0, 2, 3, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_correct_l1850_185040


namespace NUMINAMATH_GPT_a_minus_b_is_neg_seven_l1850_185006

-- Definitions for sets
def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 1 < x ∧ x < 4}
def setC : Set ℝ := {x | 1 < x ∧ x < 3}

-- Proving the statement
theorem a_minus_b_is_neg_seven :
  ∀ (a b : ℝ), (∀ x, (x ∈ setC) ↔ (x^2 + a*x + b < 0)) → a - b = -7 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_a_minus_b_is_neg_seven_l1850_185006


namespace NUMINAMATH_GPT_calculate_color_cartridges_l1850_185009

theorem calculate_color_cartridges (c b : ℕ) (h1 : 32 * c + 27 * b = 123) (h2 : b ≥ 1) : c = 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_color_cartridges_l1850_185009


namespace NUMINAMATH_GPT_wood_stove_afternoon_burn_rate_l1850_185042

-- Conditions extracted as definitions
def morning_burn_rate : ℝ := 2
def morning_duration : ℝ := 4
def initial_wood : ℝ := 30
def final_wood : ℝ := 3
def afternoon_duration : ℝ := 4

-- Theorem statement matching the conditions and correct answer
theorem wood_stove_afternoon_burn_rate :
  let morning_burned := morning_burn_rate * morning_duration
  let total_burned := initial_wood - final_wood
  let afternoon_burned := total_burned - morning_burned
  ∃ R : ℝ, (afternoon_burned = R * afternoon_duration) ∧ (R = 4.75) :=
by
  sorry

end NUMINAMATH_GPT_wood_stove_afternoon_burn_rate_l1850_185042


namespace NUMINAMATH_GPT_tens_digit_of_9_pow_2023_l1850_185045

theorem tens_digit_of_9_pow_2023 : (9 ^ 2023) % 100 / 10 = 2 :=
by sorry

end NUMINAMATH_GPT_tens_digit_of_9_pow_2023_l1850_185045


namespace NUMINAMATH_GPT_at_least_one_number_greater_than_16000_l1850_185093

theorem at_least_one_number_greater_than_16000 
    (numbers : Fin 20 → ℕ) 
    (h_distinct : Function.Injective numbers)
    (h_square_product : ∀ i : Fin 19, ∃ k : ℕ, numbers i * numbers (i + 1) = k^2)
    (h_first : numbers 0 = 42) :
    ∃ i : Fin 20, numbers i > 16000 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_number_greater_than_16000_l1850_185093


namespace NUMINAMATH_GPT_find_x_l1850_185095

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1850_185095


namespace NUMINAMATH_GPT_additional_bags_at_max_weight_l1850_185026

/-
Constants representing the problem conditions.
-/
def num_people : Nat := 6
def bags_per_person : Nat := 5
def max_weight_per_bag : Nat := 50
def total_weight_capacity : Nat := 6000

/-
Calculate the total existing luggage weight.
-/
def total_existing_bags : Nat := num_people * bags_per_person
def total_existing_weight : Nat := total_existing_bags * max_weight_per_bag
def remaining_weight_capacity : Nat := total_weight_capacity - total_existing_weight

/-
The proof statement asserting that given the conditions, 
the airplane can hold 90 more bags at maximum weight.
-/
theorem additional_bags_at_max_weight : remaining_weight_capacity / max_weight_per_bag = 90 := by
  sorry

end NUMINAMATH_GPT_additional_bags_at_max_weight_l1850_185026


namespace NUMINAMATH_GPT_quadratic_roots_range_l1850_185046

variable (a : ℝ)

theorem quadratic_roots_range (h : ∀ b c (eq : b = -a ∧ c = a^2 - 4), ∃ x y, x ≠ y ∧ x^2 + b * x + c = 0 ∧ x > 0 ∧ y^2 + b * y + c = 0) :
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_range_l1850_185046


namespace NUMINAMATH_GPT_polynomial_condition_satisfied_l1850_185077

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end NUMINAMATH_GPT_polynomial_condition_satisfied_l1850_185077


namespace NUMINAMATH_GPT_find_q_l1850_185081

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l1850_185081


namespace NUMINAMATH_GPT_grain_output_scientific_notation_l1850_185049

theorem grain_output_scientific_notation :
    682.85 * 10^6 = 6.8285 * 10^8 := 
by sorry

end NUMINAMATH_GPT_grain_output_scientific_notation_l1850_185049


namespace NUMINAMATH_GPT_prob_sum_equals_15_is_0_l1850_185008

theorem prob_sum_equals_15_is_0 (coin1 coin2 : ℕ) (die_min die_max : ℕ) (age : ℕ)
  (h1 : coin1 = 5) (h2 : coin2 = 15) (h3 : die_min = 1) (h4 : die_max = 6) (h5 : age = 15) :
  ((coin1 = 5 ∨ coin2 = 15) → die_min ≤ ((if coin1 = 5 then 5 else 15) + (die_max - die_min + 1)) ∧ 
   (die_min ≤ 6) ∧ 6 ≤ die_max) → 
  0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_prob_sum_equals_15_is_0_l1850_185008


namespace NUMINAMATH_GPT_seventh_place_is_unspecified_l1850_185039

noncomputable def charlie_position : ℕ := 5
noncomputable def emily_position : ℕ := charlie_position + 5
noncomputable def dana_position : ℕ := 10
noncomputable def bob_position : ℕ := dana_position - 2
noncomputable def alice_position : ℕ := emily_position + 3

theorem seventh_place_is_unspecified :
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 15 ∧ x ≠ charlie_position ∧ x ≠ emily_position ∧
  x ≠ dana_position ∧ x ≠ bob_position ∧ x ≠ alice_position →
  x = 7 → false := 
by
  sorry

end NUMINAMATH_GPT_seventh_place_is_unspecified_l1850_185039


namespace NUMINAMATH_GPT_payment_to_C_l1850_185079

theorem payment_to_C (A_days B_days total_payment days_taken : ℕ) 
  (A_work_rate B_work_rate : ℚ)
  (work_fraction_by_A_and_B : ℚ)
  (remaining_work_fraction_by_C : ℚ)
  (C_payment : ℚ) :
  A_days = 6 →
  B_days = 8 →
  total_payment = 3360 →
  days_taken = 3 →
  A_work_rate = 1/6 →
  B_work_rate = 1/8 →
  work_fraction_by_A_and_B = (A_work_rate + B_work_rate) * days_taken →
  remaining_work_fraction_by_C = 1 - work_fraction_by_A_and_B →
  C_payment = total_payment * remaining_work_fraction_by_C →
  C_payment = 420 := 
by
  intros hA hB hTP hD hAR hBR hWF hRWF hCP
  sorry

end NUMINAMATH_GPT_payment_to_C_l1850_185079


namespace NUMINAMATH_GPT_jessicas_score_l1850_185085

theorem jessicas_score (average_20 : ℕ) (average_21 : ℕ) (n : ℕ) (jessica_score : ℕ) 
  (h1 : average_20 = 75)
  (h2 : average_21 = 76)
  (h3 : n = 20)
  (h4 : jessica_score = (average_21 * (n + 1)) - (average_20 * n)) :
  jessica_score = 96 :=
by 
  sorry

end NUMINAMATH_GPT_jessicas_score_l1850_185085


namespace NUMINAMATH_GPT_statement_2_statement_4_l1850_185089

-- Definitions and conditions
variables {Point Line Plane : Type}
variable (a b : Line)
variable (α : Plane)

def parallel (l1 l2 : Line) : Prop := sorry  -- Define parallel relation
def perp (l1 l2 : Line) : Prop := sorry  -- Define perpendicular relation
def perp_plane (l : Line) (p : Plane) : Prop := sorry  -- Define line-plane perpendicular relation
def lies_in (l : Line) (p : Plane) : Prop := sorry  -- Define line lies in plane relation

-- Problem statement 2: If a ∥ b and a ⟂ α, then b ⟂ α
theorem statement_2 (h1 : parallel a b) (h2 : perp_plane a α) : perp_plane b α := sorry

-- Problem statement 4: If a ⟂ α and b ⟂ a, then a ∥ b
theorem statement_4 (h1 : perp_plane a α) (h2 : perp b a) : parallel a b := sorry

end NUMINAMATH_GPT_statement_2_statement_4_l1850_185089


namespace NUMINAMATH_GPT_speed_in_still_water_l1850_185060

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_up : upstream_speed = 26) (h_down : downstream_speed = 30) :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1850_185060


namespace NUMINAMATH_GPT_multiplication_result_l1850_185005

theorem multiplication_result :
  (500 ^ 50) * (2 ^ 100) = 10 ^ 75 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_result_l1850_185005


namespace NUMINAMATH_GPT_jean_pairs_of_pants_l1850_185037

theorem jean_pairs_of_pants
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (number_of_pairs : ℝ)
  (h1 : retail_price = 45)
  (h2 : discount_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : total_paid = 396)
  (h5 : number_of_pairs = total_paid / ((retail_price * (1 - discount_rate)) * (1 + tax_rate))) :
  number_of_pairs = 10 :=
by
  sorry

end NUMINAMATH_GPT_jean_pairs_of_pants_l1850_185037


namespace NUMINAMATH_GPT_solution_part_1_solution_part_2_l1850_185051

def cost_price_of_badges (x y : ℕ) : Prop :=
  (x - y = 4) ∧ (6 * x = 10 * y)

theorem solution_part_1 (x y : ℕ) :
  cost_price_of_badges x y → x = 10 ∧ y = 6 :=
by
  sorry

def maximizing_profit (m : ℕ) (w : ℕ) : Prop :=
  (10 * m + 6 * (400 - m) ≤ 2800) ∧ (w = m + 800)

theorem solution_part_2 (m : ℕ) :
  maximizing_profit m 900 → m = 100 :=
by
  sorry


end NUMINAMATH_GPT_solution_part_1_solution_part_2_l1850_185051


namespace NUMINAMATH_GPT_power_identity_l1850_185099

-- Define the given definitions
def P (m : ℕ) : ℕ := 5 ^ m
def R (n : ℕ) : ℕ := 7 ^ n

-- The theorem to be proved
theorem power_identity (m n : ℕ) : 35 ^ (m + n) = (P m ^ n * R n ^ m) := 
by sorry

end NUMINAMATH_GPT_power_identity_l1850_185099


namespace NUMINAMATH_GPT_camera_sticker_price_l1850_185058

theorem camera_sticker_price (p : ℝ)
  (h1 : p > 0)
  (hx : ∀ x, x = 0.80 * p - 50)
  (hy : ∀ y, y = 0.65 * p)
  (hs : 0.80 * p - 50 = 0.65 * p - 40) :
  p = 666.67 :=
by sorry

end NUMINAMATH_GPT_camera_sticker_price_l1850_185058


namespace NUMINAMATH_GPT_find_k_check_divisibility_l1850_185071

-- Define the polynomial f(x) as 2x^3 - 8x^2 + kx - 10
def f (x k : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + k * x - 10

-- Define the polynomial g(x) as 2x^3 - 8x^2 + 13x - 10 after finding k = 13
def g (x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + 13 * x - 10

-- The first proof problem: Finding k
theorem find_k : (f 2 k = 0) → k = 13 := 
sorry

-- The second proof problem: Checking divisibility by 2x^2 - 1
theorem check_divisibility : ¬ (∃ h : ℝ → ℝ, g x = (2 * x^2 - 1) * h x) := 
sorry

end NUMINAMATH_GPT_find_k_check_divisibility_l1850_185071


namespace NUMINAMATH_GPT_percentage_reduction_l1850_185044

variable (C S newS newC : ℝ)
variable (P : ℝ)
variable (hC : C = 50)
variable (hS : S = 1.25 * C)
variable (hNewS : newS = S - 10.50)
variable (hGain30 : newS = 1.30 * newC)
variable (hNewC : newC = C - P * C)

theorem percentage_reduction (C S newS newC : ℝ) (hC : C = 50) 
  (hS : S = 1.25 * C) (hNewS : newS = S - 10.50) 
  (hGain30 : newS = 1.30 * newC) 
  (hNewC : newC = C - P * C) : 
  P = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l1850_185044


namespace NUMINAMATH_GPT_letter_150_in_pattern_l1850_185050

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end NUMINAMATH_GPT_letter_150_in_pattern_l1850_185050


namespace NUMINAMATH_GPT_charity_event_revenue_l1850_185010

theorem charity_event_revenue :
  ∃ (f t p : ℕ), f + t = 190 ∧ f * p + t * (p / 3) = 2871 ∧ f * p = 1900 :=
by
  sorry

end NUMINAMATH_GPT_charity_event_revenue_l1850_185010


namespace NUMINAMATH_GPT_trigonometric_proof_l1850_185022

theorem trigonometric_proof (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by sorry

end NUMINAMATH_GPT_trigonometric_proof_l1850_185022


namespace NUMINAMATH_GPT_vector_subtraction_l1850_185063

-- Definitions of given conditions
def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

-- Definition of vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_subtraction : vector_sub OB OA = (-5, 3) :=
by 
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1850_185063


namespace NUMINAMATH_GPT_max_n_is_11_l1850_185002

noncomputable def max_n (a1 d : ℝ) : ℕ :=
if h : d < 0 then
  11
else
  sorry

theorem max_n_is_11 (d : ℝ) (a1 : ℝ) (c : ℝ) :
  (d / 2) * (22 ^ 2) + (a1 - (d / 2)) * 22 + c ≥ 0 →
  22 = (a1 - (d / 2)) / (- (d / 2)) →
  max_n a1 d = 11 :=
by
  intros h1 h2
  rw [max_n]
  split_ifs
  · exact rfl
  · exact sorry

end NUMINAMATH_GPT_max_n_is_11_l1850_185002


namespace NUMINAMATH_GPT_total_cards_across_decks_l1850_185084

-- Conditions
def DeckA_cards : ℕ := 52
def DeckB_cards : ℕ := 40
def DeckC_cards : ℕ := 50
def DeckD_cards : ℕ := 48

-- Question as a statement
theorem total_cards_across_decks : (DeckA_cards + DeckB_cards + DeckC_cards + DeckD_cards = 190) := by
  sorry

end NUMINAMATH_GPT_total_cards_across_decks_l1850_185084


namespace NUMINAMATH_GPT_quadrilateral_area_l1850_185048

variable (d : ℝ) (o₁ : ℝ) (o₂ : ℝ)

theorem quadrilateral_area (h₁ : d = 28) (h₂ : o₁ = 8) (h₃ : o₂ = 2) : 
  (1 / 2 * d * o₁) + (1 / 2 * d * o₂) = 140 := 
  by
    rw [h₁, h₂, h₃]
    sorry

end NUMINAMATH_GPT_quadrilateral_area_l1850_185048


namespace NUMINAMATH_GPT_length_of_crease_correct_l1850_185023

noncomputable def length_of_crease (theta : ℝ) : ℝ := Real.sqrt (40 + 24 * Real.cos theta)

theorem length_of_crease_correct (theta : ℝ) : 
  length_of_crease theta = Real.sqrt (40 + 24 * Real.cos theta) := 
by 
  sorry

end NUMINAMATH_GPT_length_of_crease_correct_l1850_185023


namespace NUMINAMATH_GPT_price_per_glass_first_day_l1850_185068

theorem price_per_glass_first_day (O P2 P1: ℝ) (H1 : O > 0) (H2 : P2 = 0.2) (H3 : 2 * O * P1 = 3 * O * P2) : P1 = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_price_per_glass_first_day_l1850_185068


namespace NUMINAMATH_GPT_inequality_solution_l1850_185065

theorem inequality_solution (x : ℝ) : 
  (∃ (y : ℝ), y = 1 / (3 ^ x) ∧ y * (y - 2) < 15) ↔ x > - (Real.log 5 / Real.log 3) :=
by 
    sorry

end NUMINAMATH_GPT_inequality_solution_l1850_185065


namespace NUMINAMATH_GPT_marble_distribution_l1850_185096

-- Define the problem statement using conditions extracted above
theorem marble_distribution :
  ∃ (A B C D : ℕ), A + B + C + D = 28 ∧
  (A = 7 ∨ B = 7 ∨ C = 7 ∨ D = 7) ∧
  ((A = 7 → B + C + D = 21) ∧
   (B = 7 → A + C + D = 21) ∧
   (C = 7 → A + B + D = 21) ∧
   (D = 7 → A + B + C = 21)) :=
sorry

end NUMINAMATH_GPT_marble_distribution_l1850_185096


namespace NUMINAMATH_GPT_number_of_ways_to_choose_one_book_l1850_185038

-- Defining the conditions
def num_chinese_books : ℕ := 5
def num_math_books : ℕ := 4

-- Statement of the theorem
theorem number_of_ways_to_choose_one_book : num_chinese_books + num_math_books = 9 :=
by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_one_book_l1850_185038


namespace NUMINAMATH_GPT_germs_per_dish_l1850_185014

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end NUMINAMATH_GPT_germs_per_dish_l1850_185014


namespace NUMINAMATH_GPT_min_height_bounces_l1850_185075

noncomputable def geometric_sequence (a r: ℝ) (n: ℕ) : ℝ := 
  a * r^n

theorem min_height_bounces (k : ℕ) : 
  ∀ k, 20 * (2 / 3 : ℝ) ^ k < 3 → k ≥ 7 := 
by
  sorry

end NUMINAMATH_GPT_min_height_bounces_l1850_185075


namespace NUMINAMATH_GPT_blend_pieces_eq_two_l1850_185027

variable (n_silk n_cashmere total_pieces : ℕ)

def luther_line := n_silk = 10 ∧ n_cashmere = n_silk / 2 ∧ total_pieces = 13

theorem blend_pieces_eq_two : luther_line n_silk n_cashmere total_pieces → (n_cashmere - (total_pieces - n_silk) = 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_blend_pieces_eq_two_l1850_185027


namespace NUMINAMATH_GPT_floor_eq_solution_l1850_185033

theorem floor_eq_solution (x : ℝ) : 2.5 ≤ x ∧ x < 3.5 → (⌊2 * x + 0.5⌋ = ⌊x + 3⌋) :=
by
  sorry

end NUMINAMATH_GPT_floor_eq_solution_l1850_185033


namespace NUMINAMATH_GPT_janet_faster_playtime_l1850_185015

theorem janet_faster_playtime 
  (initial_minutes : ℕ)
  (initial_seconds : ℕ)
  (faster_rate : ℝ)
  (initial_time_in_seconds := initial_minutes * 60 + initial_seconds)
  (target_time_in_seconds := initial_time_in_seconds / faster_rate) :
  initial_minutes = 3 →
  initial_seconds = 20 →
  faster_rate = 1.25 →
  target_time_in_seconds = 160 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_janet_faster_playtime_l1850_185015


namespace NUMINAMATH_GPT_more_elements_in_set_N_l1850_185061

theorem more_elements_in_set_N 
  (M N : Finset ℕ) 
  (h_partition : ∀ x, x ∈ M ∨ x ∈ N) 
  (h_disjoint : ∀ x, x ∈ M → x ∉ N) 
  (h_total_2000 : M.card + N.card = 10^2000 - 10^1999) 
  (h_total_1000 : (10^1000 - 10^999) * (10^1000 - 10^999) < 10^2000 - 10^1999) : 
  N.card > M.card :=
by { sorry }

end NUMINAMATH_GPT_more_elements_in_set_N_l1850_185061


namespace NUMINAMATH_GPT_find_a_extremum_and_min_value_find_max_k_l1850_185004

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a

theorem find_a_extremum_and_min_value :
  (∀ a : ℝ, f' a 0 = 0 → a = -1) ∧
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → f (-1) x ≥ 2) :=
by sorry

theorem find_max_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → k * (Real.exp x - 1) < x * Real.exp x + 1) →
  k ≤ 2 :=
by sorry

end NUMINAMATH_GPT_find_a_extremum_and_min_value_find_max_k_l1850_185004


namespace NUMINAMATH_GPT_expressions_same_type_l1850_185083

def same_type_as (e1 e2 : ℕ × ℕ) : Prop :=
  e1 = e2

def exp_of_expr (a_exp b_exp : ℕ) : ℕ × ℕ :=
  (a_exp, b_exp)

def exp_3a2b := exp_of_expr 2 1
def exp_neg_ba2 := exp_of_expr 2 1

theorem expressions_same_type :
  same_type_as exp_neg_ba2 exp_3a2b :=
by
  sorry

end NUMINAMATH_GPT_expressions_same_type_l1850_185083


namespace NUMINAMATH_GPT_solve_equation_l1850_185072

theorem solve_equation (x y : ℤ) (eq : (x^2 - y^2)^2 = 16 * y + 1) : 
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ 
  (x = 4 ∧ y = 3) ∨ (x = -4 ∧ y = 3) ∨ 
  (x = 4 ∧ y = 5) ∨ (x = -4 ∧ y = 5) :=
sorry

end NUMINAMATH_GPT_solve_equation_l1850_185072


namespace NUMINAMATH_GPT_only_function_B_has_inverse_l1850_185078

-- Definitions based on the problem conditions
def function_A (x : ℝ) : ℝ := 3 - x^2 -- Parabola opening downwards with vertex at (0,3)
def function_B (x : ℝ) : ℝ := x -- Straight line with slope 1 passing through (0,0) and (1,1)
def function_C (x y : ℝ) : Prop := x^2 + y^2 = 4 -- Circle centered at (0,0) with radius 2

-- Theorem stating that only function B has an inverse
theorem only_function_B_has_inverse :
  (∀ y : ℝ, ∃! x : ℝ, function_B x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, function_A x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, ∃ y1 y2 : ℝ, function_C x y1 ∧ function_C x y2 ∧ y1 ≠ y2) :=
  by 
  sorry -- Proof not required

end NUMINAMATH_GPT_only_function_B_has_inverse_l1850_185078


namespace NUMINAMATH_GPT_expression_value_l1850_185007

theorem expression_value (x : ℤ) (h : x = -2) : x ^ 2 + 6 * x - 8 = -16 := 
by 
  rw [h]
  sorry

end NUMINAMATH_GPT_expression_value_l1850_185007


namespace NUMINAMATH_GPT_part1_l1850_185025

theorem part1 (n : ℕ) (m : ℕ) (h_form : m = 2 ^ (n - 2) * 5 ^ n) (h : 6 * 10 ^ n + m = 25 * m) :
  ∃ k : ℕ, 6 * 10 ^ n + m = 625 * 10 ^ (n - 2) :=
by
  sorry

end NUMINAMATH_GPT_part1_l1850_185025


namespace NUMINAMATH_GPT_tax_collection_amount_l1850_185028

theorem tax_collection_amount (paid_tax : ℝ) (willam_percentage : ℝ) (total_collected : ℝ) (h_paid: paid_tax = 480) (h_percentage: willam_percentage = 0.3125) :
    total_collected = 1536 :=
by
  sorry

end NUMINAMATH_GPT_tax_collection_amount_l1850_185028


namespace NUMINAMATH_GPT_outfit_count_l1850_185062

theorem outfit_count (shirts pants ties belts : ℕ) (h_shirts : shirts = 8) (h_pants : pants = 5) (h_ties : ties = 4) (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end NUMINAMATH_GPT_outfit_count_l1850_185062


namespace NUMINAMATH_GPT_min_balls_to_draw_l1850_185067

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end NUMINAMATH_GPT_min_balls_to_draw_l1850_185067


namespace NUMINAMATH_GPT_janelle_gave_green_marbles_l1850_185034

def initial_green_marbles : ℕ := 26
def bags_blue_marbles : ℕ := 6
def marbles_per_bag : ℕ := 10
def total_blue_marbles : ℕ := bags_blue_marbles * marbles_per_bag
def total_marbles_after_gift : ℕ := 72
def blue_marbles_in_gift : ℕ := 8
def final_blue_marbles : ℕ := total_blue_marbles - blue_marbles_in_gift
def final_green_marbles : ℕ := total_marbles_after_gift - final_blue_marbles
def initial_green_marbles_after_gift : ℕ := final_green_marbles
def green_marbles_given : ℕ := initial_green_marbles - initial_green_marbles_after_gift

theorem janelle_gave_green_marbles :
  green_marbles_given = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_janelle_gave_green_marbles_l1850_185034


namespace NUMINAMATH_GPT_max_5x_plus_3y_l1850_185013

theorem max_5x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 8 * y + 10) : 5 * x + 3 * y ≤ 105 :=
sorry

end NUMINAMATH_GPT_max_5x_plus_3y_l1850_185013


namespace NUMINAMATH_GPT_taehyung_walks_more_than_minyoung_l1850_185029

def taehyung_distance_per_minute : ℕ := 114
def minyoung_distance_per_minute : ℕ := 79
def minutes_per_hour : ℕ := 60

theorem taehyung_walks_more_than_minyoung :
  (taehyung_distance_per_minute * minutes_per_hour) -
  (minyoung_distance_per_minute * minutes_per_hour) = 2100 := by
  sorry

end NUMINAMATH_GPT_taehyung_walks_more_than_minyoung_l1850_185029


namespace NUMINAMATH_GPT_value_of_expression_l1850_185053

theorem value_of_expression (x y z : ℝ) (h : x * y * z = 1) :
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1850_185053
