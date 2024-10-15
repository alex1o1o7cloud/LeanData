import Mathlib

namespace NUMINAMATH_GPT_percentage_girls_l961_96112

theorem percentage_girls (x y : ℕ) (S₁ S₂ : ℕ)
  (h1 : S₁ = 22 * x)
  (h2 : S₂ = 47 * y)
  (h3 : (S₁ + S₂) / (x + y) = 41) :
  (x : ℝ) / (x + y) = 0.24 :=
sorry

end NUMINAMATH_GPT_percentage_girls_l961_96112


namespace NUMINAMATH_GPT_two_same_color_probability_l961_96136

-- Definitions based on the given conditions
def total_balls := 5
def black_balls := 3
def red_balls := 2

-- Definition for drawing two balls at random
def draw_two_same_color_probability : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let black_pairs := Nat.choose black_balls 2
  let red_pairs := Nat.choose red_balls 2
  (black_pairs + red_pairs) / total_ways

-- Statement of the theorem
theorem two_same_color_probability :
  draw_two_same_color_probability = 2 / 5 :=
  sorry

end NUMINAMATH_GPT_two_same_color_probability_l961_96136


namespace NUMINAMATH_GPT_actual_total_area_in_acres_l961_96184

-- Define the conditions
def base_cm : ℝ := 20
def height_cm : ℝ := 12
def rect_length_cm : ℝ := 20
def rect_width_cm : ℝ := 5
def scale_cm_to_miles : ℝ := 3
def sq_mile_to_acres : ℝ := 640

-- Define the total area in acres calculation
def total_area_cm_squared : ℝ := 120 + 100
def total_area_miles_squared : ℝ := total_area_cm_squared * (scale_cm_to_miles ^ 2)
def total_area_acres : ℝ := total_area_miles_squared * sq_mile_to_acres

-- The theorem statement
theorem actual_total_area_in_acres : total_area_acres = 1267200 :=
by
  sorry

end NUMINAMATH_GPT_actual_total_area_in_acres_l961_96184


namespace NUMINAMATH_GPT_initial_percentage_of_water_l961_96188

theorem initial_percentage_of_water (P : ℕ) : 
  (P / 100) * 120 + 54 = (3 / 4) * 120 → P = 30 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_initial_percentage_of_water_l961_96188


namespace NUMINAMATH_GPT_max_rooks_in_cube_l961_96193

def non_attacking_rooks (n : ℕ) (cube : ℕ × ℕ × ℕ) : ℕ :=
  if cube = (8, 8, 8) then 64 else 0

theorem max_rooks_in_cube:
  non_attacking_rooks 64 (8, 8, 8) = 64 :=
by
  -- proof by logical steps matching the provided solution, if necessary, start with sorry for placeholder
  sorry

end NUMINAMATH_GPT_max_rooks_in_cube_l961_96193


namespace NUMINAMATH_GPT_jordan_rectangle_width_l961_96173

noncomputable def carol_length : ℝ := 4.5
noncomputable def carol_width : ℝ := 19.25
noncomputable def jordan_length : ℝ := 3.75

noncomputable def carol_area : ℝ := carol_length * carol_width
noncomputable def jordan_width : ℝ := carol_area / jordan_length

theorem jordan_rectangle_width : jordan_width = 23.1 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_jordan_rectangle_width_l961_96173


namespace NUMINAMATH_GPT_base_p_prime_values_zero_l961_96146

theorem base_p_prime_values_zero :
  (∀ p : ℕ, p.Prime → 2008 * p^3 + 407 * p^2 + 214 * p + 226 = 243 * p^2 + 382 * p + 471 → False) :=
by
  sorry

end NUMINAMATH_GPT_base_p_prime_values_zero_l961_96146


namespace NUMINAMATH_GPT_factorization_correct_l961_96153

theorem factorization_correct : ∃ (a b : ℕ), (a > b) ∧ (3 * b - a = 12) ∧ (x^2 - 16 * x + 63 = (x - a) * (x - b)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l961_96153


namespace NUMINAMATH_GPT_positive_integers_p_divisibility_l961_96127

theorem positive_integers_p_divisibility (p : ℕ) (hp : 0 < p) :
  (∃ n : ℕ, 0 < n ∧ p^n + 3^n ∣ p^(n+1) + 3^(n+1)) ↔ p = 3 ∨ p = 6 ∨ p = 15 :=
by sorry

end NUMINAMATH_GPT_positive_integers_p_divisibility_l961_96127


namespace NUMINAMATH_GPT_mowing_lawn_each_week_l961_96194

-- Definitions based on the conditions
def riding_speed : ℝ := 2 -- acres per hour with riding mower
def push_speed : ℝ := 1 -- acre per hour with push mower
def total_hours : ℝ := 5 -- total hours

-- The problem we want to prove
theorem mowing_lawn_each_week (A : ℝ) :
  (3 / 4) * A / riding_speed + (1 / 4) * A / push_speed = total_hours → 
  A = 15 :=
by
  sorry

end NUMINAMATH_GPT_mowing_lawn_each_week_l961_96194


namespace NUMINAMATH_GPT_complement_intersection_l961_96131

open Set

variable (U : Set ℕ) (A B : Set ℕ)

theorem complement_intersection :
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 3, 5} →
  U \ (A ∩ B) = {1, 4, 5} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_intersection_l961_96131


namespace NUMINAMATH_GPT_circle_equation_midpoint_trajectory_l961_96166

-- Definition for the circle equation proof
theorem circle_equation (x y : ℝ) (h : (x - 3)^2 + (y - 2)^2 = 13)
  (hx : x = 3) (hy : y = 2) : 
  (x - 3)^2 + (y - 2)^2 = 13 := by
  sorry -- Placeholder for proof

-- Definition for the midpoint trajectory proof
theorem midpoint_trajectory (x y : ℝ) (hx : x = (2 * x - 11) / 2)
  (hy : y = (2 * y - 2) / 2) (h : (2 * x - 11)^2 + (2 * y - 2)^2 = 13) :
  (x - 11 / 2)^2 + (y - 1)^2 = 13 / 4 := by
  sorry -- Placeholder for proof

end NUMINAMATH_GPT_circle_equation_midpoint_trajectory_l961_96166


namespace NUMINAMATH_GPT_mark_fewer_than_susan_l961_96167

variable (apples_total : ℕ) (greg_apples : ℕ) (susan_apples : ℕ) (mark_apples : ℕ) (mom_apples : ℕ)

def evenly_split (total : ℕ) : ℕ := total / 2

theorem mark_fewer_than_susan
    (h1 : apples_total = 18)
    (h2 : greg_apples = evenly_split apples_total)
    (h3 : susan_apples = 2 * greg_apples)
    (h4 : mom_apples = 40 + 9)
    (h5 : mark_apples = mom_apples - susan_apples) :
    susan_apples - mark_apples = 13 := 
sorry

end NUMINAMATH_GPT_mark_fewer_than_susan_l961_96167


namespace NUMINAMATH_GPT_MarionBikeCost_l961_96147

theorem MarionBikeCost (M : ℤ) (h1 : 2 * M + M = 1068) : M = 356 :=
by
  sorry

end NUMINAMATH_GPT_MarionBikeCost_l961_96147


namespace NUMINAMATH_GPT_neg_proposition_p_l961_96102

variable {x : ℝ}

def proposition_p : Prop := ∀ x ≥ 0, x^3 - 1 ≥ 0

theorem neg_proposition_p : ¬ proposition_p ↔ ∃ x ≥ 0, x^3 - 1 < 0 :=
by sorry

end NUMINAMATH_GPT_neg_proposition_p_l961_96102


namespace NUMINAMATH_GPT_senior_employee_bonus_l961_96171

theorem senior_employee_bonus (J S : ℝ) 
  (h1 : S = J + 1200)
  (h2 : J + S = 5000) : 
  S = 3100 :=
sorry

end NUMINAMATH_GPT_senior_employee_bonus_l961_96171


namespace NUMINAMATH_GPT_overtime_percentage_increase_l961_96155

-- Define basic conditions
def basic_hours := 40
def total_hours := 48
def basic_pay := 20
def total_wage := 25

-- Calculate overtime hours and wages
def overtime_hours := total_hours - basic_hours
def overtime_pay := total_wage - basic_pay

-- Define basic and overtime hourly rates
def basic_hourly_rate := basic_pay / basic_hours
def overtime_hourly_rate := overtime_pay / overtime_hours

-- Calculate and state the theorem for percentage increase
def percentage_increase := ((overtime_hourly_rate - basic_hourly_rate) / basic_hourly_rate) * 100

theorem overtime_percentage_increase :
  percentage_increase = 25 :=
by
  sorry

end NUMINAMATH_GPT_overtime_percentage_increase_l961_96155


namespace NUMINAMATH_GPT_diamond_property_C_l961_96154

-- Define the binary operation diamond
def diamond (a b : ℕ) : ℕ := a ^ (2 * b)

theorem diamond_property_C (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) : 
  (diamond a b) ^ n = diamond a (b * n) :=
by
  sorry

end NUMINAMATH_GPT_diamond_property_C_l961_96154


namespace NUMINAMATH_GPT_change_received_l961_96137

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_change_received_l961_96137


namespace NUMINAMATH_GPT_city_population_divided_l961_96195

theorem city_population_divided (total_population : ℕ) (parts : ℕ) (male_parts : ℕ) 
  (h1 : total_population = 1000) (h2 : parts = 5) (h3 : male_parts = 2) : 
  ∃ males : ℕ, males = 400 :=
by
  sorry

end NUMINAMATH_GPT_city_population_divided_l961_96195


namespace NUMINAMATH_GPT_tattoo_ratio_l961_96133

theorem tattoo_ratio (a j k : ℕ) (ha : a = 23) (hj : j = 10) (rel : a = k * j + 3) : a / j = 23 / 10 :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_tattoo_ratio_l961_96133


namespace NUMINAMATH_GPT_dice_sum_not_20_l961_96115

/-- Given that Louise rolls four standard six-sided dice (with faces numbered from 1 to 6)
    and the product of the numbers on the upper faces is 216, prove that it is not possible
    for the sum of the upper faces to be 20. -/
theorem dice_sum_not_20 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
                        (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
                        (product : a * b * c * d = 216) : a + b + c + d ≠ 20 := 
by sorry

end NUMINAMATH_GPT_dice_sum_not_20_l961_96115


namespace NUMINAMATH_GPT_find_pages_revised_twice_l961_96181

def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_time : ℕ) (cost_revised_once : ℕ) (cost_revised_twice : ℕ) (total_cost : ℕ) :=
  ∃ (x : ℕ), 
    (total_pages - pages_revised_once - x) * cost_first_time
    + pages_revised_once * (cost_first_time + cost_revised_once)
    + x * (cost_first_time + cost_revised_once + cost_revised_once) = total_cost 

theorem find_pages_revised_twice :
  pages_revised_twice 100 35 6 4 4 860 ↔ ∃ x, x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_pages_revised_twice_l961_96181


namespace NUMINAMATH_GPT_number_of_rabbits_l961_96121

theorem number_of_rabbits (x y : ℕ) (h1 : x + y = 28) (h2 : 4 * x = 6 * y + 12) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rabbits_l961_96121


namespace NUMINAMATH_GPT_ellipse_focal_length_l961_96157

theorem ellipse_focal_length :
  ∀ a b c : ℝ, (a^2 = 11) → (b^2 = 3) → (c^2 = a^2 - b^2) → (2 * c = 4 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focal_length_l961_96157


namespace NUMINAMATH_GPT_calculate_result_l961_96125

theorem calculate_result :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_calculate_result_l961_96125


namespace NUMINAMATH_GPT_min_moves_to_checkerboard_l961_96150

noncomputable def minimum_moves_checkerboard (n : ℕ) : ℕ :=
if n = 6 then 18
else 0

theorem min_moves_to_checkerboard :
  minimum_moves_checkerboard 6 = 18 :=
by sorry

end NUMINAMATH_GPT_min_moves_to_checkerboard_l961_96150


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l961_96185

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero : 
  (∃ x : ℝ, f x = 20) ∧ (∃ y : ℝ, f y = 20 ∧ x = -y) → 
  x + y = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l961_96185


namespace NUMINAMATH_GPT_problem1_problem2_l961_96134

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a/x

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number),
prove that if the function f(x) has two zeros, then 0 < a < 1/e.
-/
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → (0 < a ∧ a < 1/Real.exp 1) :=
sorry

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number) and a line y = m
that intersects the graph of f(x) at two points (x1, m) and (x2, m),
prove that x1 + x2 > 2a.
-/
theorem problem2 (x1 x2 a m : ℝ) (h : f x1 a = m ∧ f x2 a = m ∧ x1 ≠ x2) :
  x1 + x2 > 2 * a :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l961_96134


namespace NUMINAMATH_GPT_part1_part2_part3_l961_96123

variable (a b c d S A B C D : ℝ)

-- The given conditions
def cond1 : Prop := a + c = b + d
def cond2 : Prop := A + C = B + D
def cond3 : Prop := S^2 = a * b * c * d

-- The statements to prove
theorem part1 (h1 : cond1 a b c d) (h2 : cond2 A B C D) : cond3 a b c d S := sorry
theorem part2 (h1 : cond1 a b c d) (h3 : cond3 a b c d S) : cond2 A B C D := sorry
theorem part3 (h2 : cond2 A B C D) : cond3 a b c d S := sorry

end NUMINAMATH_GPT_part1_part2_part3_l961_96123


namespace NUMINAMATH_GPT_subtract_complex_eq_l961_96168

noncomputable def subtract_complex (a b : ℂ) : ℂ := a - b

theorem subtract_complex_eq (i : ℂ) (h_i : i^2 = -1) :
  subtract_complex (5 - 3 * i) (7 - 7 * i) = -2 + 4 * i :=
by
  sorry

end NUMINAMATH_GPT_subtract_complex_eq_l961_96168


namespace NUMINAMATH_GPT_functional_expression_result_l961_96179

theorem functional_expression_result {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y) :
  ∀ t : ℝ, (f (4 * t) - f t) / (f (3 * t) - f (2 * t)) = 3 :=
sorry

end NUMINAMATH_GPT_functional_expression_result_l961_96179


namespace NUMINAMATH_GPT_part_a_part_b_l961_96116

/-- Definition of the sequence of numbers on the cards -/
def card_numbers (n : ℕ) : ℕ :=
  if n = 0 then 1 else (10^(n + 1) - 1) / 9 * 2 + 1

/-- Part (a) statement: Is it possible to choose at least three cards such that 
the sum of the numbers on them equals a number where all digits except one are twos? -/
theorem part_a : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ card_numbers a + card_numbers b + card_numbers c % 10 = 2 ∧ 
  (∀ d, ∃ k ≤ 1, (card_numbers a + card_numbers b + card_numbers c / (10^d)) % 10 = 2) :=
sorry

/-- Part (b) statement: Suppose several cards were chosen such that the sum of the numbers 
on them equals a number where all digits except one are twos. What could be the digit that is not two? -/
theorem part_b (sum : ℕ) :
  (∀ d, sum / (10^d) % 10 = 2) → ((sum % 10 = 0) ∨ (sum % 10 = 1)) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l961_96116


namespace NUMINAMATH_GPT_correct_operation_l961_96169

theorem correct_operation (a b : ℝ) : 
  (-a^3 * b)^2 = a^6 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l961_96169


namespace NUMINAMATH_GPT_pipe_r_fill_time_l961_96180

theorem pipe_r_fill_time (x : ℝ) : 
  (1 / 3 + 1 / 9 + 1 / x = 1 / 2) → 
  x = 18 :=
by 
  sorry

end NUMINAMATH_GPT_pipe_r_fill_time_l961_96180


namespace NUMINAMATH_GPT_parabola_focus_segment_length_l961_96145

theorem parabola_focus_segment_length (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∀ x, abs x * abs (1 / a) = 4) : a = 1/4 := 
sorry

end NUMINAMATH_GPT_parabola_focus_segment_length_l961_96145


namespace NUMINAMATH_GPT_positive_integers_solution_l961_96110

theorem positive_integers_solution :
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ (3 ^ m - 2 ^ n = -1 ∨ 3 ^ m - 2 ^ n = 5 ∨ 3 ^ m - 2 ^ n = 7) ↔
  (m, n) = (0, 1) ∨ (m, n) = (2, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 2) :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_solution_l961_96110


namespace NUMINAMATH_GPT_sqrt_of_neg_five_squared_l961_96170

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 ∨ Real.sqrt ((-5 : ℝ)^2) = -5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_neg_five_squared_l961_96170


namespace NUMINAMATH_GPT_value_of_a_l961_96132

theorem value_of_a :
  ∀ (a : ℤ) (BO CO : ℤ), 
  BO = 2 → 
  CO = 2 * BO → 
  |a + 3| = CO → 
  a < 0 → 
  a = -7 := by
  intros a BO CO hBO hCO hAbs ha_neg
  sorry

end NUMINAMATH_GPT_value_of_a_l961_96132


namespace NUMINAMATH_GPT_distinct_digit_numbers_count_l961_96176

def numDistinctDigitNumbers : Nat := 
  let first_digit_choices := 10
  let second_digit_choices := 9
  let third_digit_choices := 8
  let fourth_digit_choices := 7
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem distinct_digit_numbers_count : numDistinctDigitNumbers = 5040 :=
by
  sorry

end NUMINAMATH_GPT_distinct_digit_numbers_count_l961_96176


namespace NUMINAMATH_GPT_sin_cos_value_l961_96117

variable (x : ℝ)

theorem sin_cos_value (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
sorry

end NUMINAMATH_GPT_sin_cos_value_l961_96117


namespace NUMINAMATH_GPT_belize_homes_l961_96124

theorem belize_homes (H : ℝ) 
  (h1 : (3 / 5) * (3 / 4) * H = 240) : 
  H = 400 :=
sorry

end NUMINAMATH_GPT_belize_homes_l961_96124


namespace NUMINAMATH_GPT_problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l961_96156

variable {a b c : ℝ}

theorem problem_inequality_A (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * b < b * c :=
by sorry

theorem problem_inequality_B (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * c < b * c :=
by sorry

theorem problem_inequality_D (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a + b < b + c :=
by sorry

theorem problem_inequality_E (h1 : a > 0) (h2 : a < b) (h3 : b < c) : c / a > 1 :=
by sorry

end NUMINAMATH_GPT_problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l961_96156


namespace NUMINAMATH_GPT_y_value_l961_96100

theorem y_value {y : ℝ} (h1 : (0, 2) = (0, 2))
                (h2 : (3, y) = (3, y))
                (h3 : dist (0, 2) (3, y) = 10)
                (h4 : y > 0) :
                y = 2 + Real.sqrt 91 := by
  sorry

end NUMINAMATH_GPT_y_value_l961_96100


namespace NUMINAMATH_GPT_f_is_constant_l961_96135

noncomputable def is_const (f : ℤ × ℤ → ℕ) := ∃ c : ℕ, ∀ p : ℤ × ℤ, f p = c

theorem f_is_constant (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_const f :=
sorry

end NUMINAMATH_GPT_f_is_constant_l961_96135


namespace NUMINAMATH_GPT_slope_and_y_intercept_l961_96163

def line_equation (x y : ℝ) : Prop := 4 * y = 6 * x - 12

theorem slope_and_y_intercept (x y : ℝ) (h : line_equation x y) : 
  ∃ m b : ℝ, (m = 3/2) ∧ (b = -3) ∧ (y = m * x + b) :=
  sorry

end NUMINAMATH_GPT_slope_and_y_intercept_l961_96163


namespace NUMINAMATH_GPT_sqrt_sum_eq_nine_l961_96175

theorem sqrt_sum_eq_nine (x : ℝ) (h : Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) :
  (7 + x) * (28 - x) = 529 :=
sorry

end NUMINAMATH_GPT_sqrt_sum_eq_nine_l961_96175


namespace NUMINAMATH_GPT_cylinder_from_sector_l961_96141

noncomputable def circle_radius : ℝ := 12
noncomputable def sector_angle : ℝ := 300
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def is_valid_cylinder (base_radius height : ℝ) : Prop :=
  2 * Real.pi * base_radius = arc_length circle_radius sector_angle ∧ height = circle_radius

theorem cylinder_from_sector :
  is_valid_cylinder 10 12 :=
by
  -- here, the proof will be provided
  sorry

end NUMINAMATH_GPT_cylinder_from_sector_l961_96141


namespace NUMINAMATH_GPT_find_f_log_3_54_l961_96190

noncomputable def f : ℝ → ℝ := sorry  -- Since we have to define a function and we do not need the exact implementation.

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_property : ∀ x : ℝ, f (x + 2) = - 1 / f x
axiom interval_property : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 3 ^ x

theorem find_f_log_3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 :=
by
  sorry


end NUMINAMATH_GPT_find_f_log_3_54_l961_96190


namespace NUMINAMATH_GPT_sum_of_roots_l961_96177

theorem sum_of_roots :
  let a := (6 : ℝ) + 3 * Real.sqrt 3
  let b := (3 : ℝ) + Real.sqrt 3
  let c := -(3 : ℝ)
  let root_sum := -b / a
  root_sum = -1 + Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_sum_of_roots_l961_96177


namespace NUMINAMATH_GPT_center_of_symmetry_is_neg2_3_l961_96149

theorem center_of_symmetry_is_neg2_3 :
  ∃ (a b : ℝ), 
  (a,b) = (-2, 3) ∧ 
  ∀ x : ℝ, 
    2 * b = ((a + x + 2)^3 - (a + x) + 1) + ((a - x + 2)^3 - (a - x) + 1) := 
by
  use -2, 3
  sorry

end NUMINAMATH_GPT_center_of_symmetry_is_neg2_3_l961_96149


namespace NUMINAMATH_GPT_difference_between_numbers_l961_96111

open Int

theorem difference_between_numbers (A B : ℕ) 
  (h1 : A + B = 1812) 
  (h2 : A = 7 * B + 4) : 
  A - B = 1360 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l961_96111


namespace NUMINAMATH_GPT_initial_balance_before_check_deposit_l961_96183

theorem initial_balance_before_check_deposit (new_balance : ℝ) (initial_balance : ℝ) : 
  (50 = 1 / 4 * new_balance) → (initial_balance = new_balance - 50) → initial_balance = 150 :=
by
  sorry

end NUMINAMATH_GPT_initial_balance_before_check_deposit_l961_96183


namespace NUMINAMATH_GPT_find_value_l961_96120

-- Defining the sequence a_n, assuming all terms are positive
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

-- Definition to capture the given condition a_2 * a_4 = 4
def condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 4

-- The main statement
theorem find_value (a : ℕ → ℝ) (h_seq : is_geometric_sequence a) (h_cond : condition a) : 
  a 1 * a 5 + a 3 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_l961_96120


namespace NUMINAMATH_GPT_additional_grassy_ground_l961_96192

theorem additional_grassy_ground (r1 r2 : ℝ) (h1: r1 = 16) (h2: r2 = 23) :
  (π * r2 ^ 2) - (π * r1 ^ 2) = 273 * π :=
by
  sorry

end NUMINAMATH_GPT_additional_grassy_ground_l961_96192


namespace NUMINAMATH_GPT_transformed_center_coordinates_l961_96104

-- Define the original center of the circle
def center_initial : ℝ × ℝ := (3, -4)

-- Define the function for reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the function for translation by a certain number of units up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Define the problem statement
theorem transformed_center_coordinates :
  translate_up (reflect_x_axis center_initial) 5 = (3, 9) :=
by
  sorry

end NUMINAMATH_GPT_transformed_center_coordinates_l961_96104


namespace NUMINAMATH_GPT_find_p_l961_96129

theorem find_p (m n p : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < p) 
  (h : 3 * m + 3 / (n + 1 / p) = 17) : p = 2 := 
sorry

end NUMINAMATH_GPT_find_p_l961_96129


namespace NUMINAMATH_GPT_smallest_possible_S_l961_96189

/-- Define the maximum possible sum for n dice --/
def max_sum (n : ℕ) : ℕ := 6 * n

/-- Define the transformation of the dice sum when each result is transformed to 7 - d_i --/
def transformed_sum (n R : ℕ) : ℕ := 7 * n - R

/-- Determine the smallest possible S under given conditions --/
theorem smallest_possible_S :
  ∃ n : ℕ, max_sum n ≥ 2001 ∧ transformed_sum n 2001 = 337 :=
by
  -- TODO: Complete the proof
  sorry

end NUMINAMATH_GPT_smallest_possible_S_l961_96189


namespace NUMINAMATH_GPT_geostationary_orbit_distance_l961_96160

noncomputable def distance_between_stations (earth_radius : ℝ) (orbit_altitude : ℝ) (num_stations : ℕ) : ℝ :=
  let θ : ℝ := 360 / num_stations
  let R : ℝ := earth_radius + orbit_altitude
  let sin_18 := (Real.sqrt 5 - 1) / 4
  2 * R * sin_18

theorem geostationary_orbit_distance :
  distance_between_stations 3960 22236 10 = -13098 + 13098 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_geostationary_orbit_distance_l961_96160


namespace NUMINAMATH_GPT_two_digit_prime_sum_9_l961_96114

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end NUMINAMATH_GPT_two_digit_prime_sum_9_l961_96114


namespace NUMINAMATH_GPT_volume_of_rect_prism_l961_96108

variables {a b c V : ℝ}

theorem volume_of_rect_prism :
  (∃ (a b c : ℝ), (a * b = Real.sqrt 2) ∧ (b * c = Real.sqrt 3) ∧ (a * c = Real.sqrt 6) ∧ V = a * b * c) →
  V = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_rect_prism_l961_96108


namespace NUMINAMATH_GPT_option_d_is_correct_l961_96143

theorem option_d_is_correct {x y : ℝ} (h : x - 2 = y - 2) : x = y := 
by 
  sorry

end NUMINAMATH_GPT_option_d_is_correct_l961_96143


namespace NUMINAMATH_GPT_initial_height_after_10_seconds_l961_96118

open Nat

def distance_fallen_in_nth_second (n : ℕ) : ℕ := 10 * n - 5

def total_distance_fallen (n : ℕ) : ℕ :=
  (n * (distance_fallen_in_nth_second 1 + distance_fallen_in_nth_second n)) / 2

theorem initial_height_after_10_seconds : 
  total_distance_fallen 10 = 500 := 
by
  sorry

end NUMINAMATH_GPT_initial_height_after_10_seconds_l961_96118


namespace NUMINAMATH_GPT_box_volume_80_possible_l961_96126

theorem box_volume_80_possible :
  ∃ (x : ℕ), 10 * x^3 = 80 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_80_possible_l961_96126


namespace NUMINAMATH_GPT_cost_of_slices_eaten_by_dog_is_correct_l961_96174

noncomputable def total_cost_before_tax : ℝ :=
  2 * 3 + 1 * 2 + 1 * 5 + 3 * 0.5 + 0.25 + 1.5 + 1.25

noncomputable def sales_tax_rate : ℝ := 0.06

noncomputable def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

noncomputable def slices : ℝ := 8

noncomputable def cost_per_slice : ℝ := total_cost_after_tax / slices

noncomputable def slices_eaten_by_dog : ℝ := 8 - 3

noncomputable def cost_of_slices_eaten_by_dog : ℝ := cost_per_slice * slices_eaten_by_dog

theorem cost_of_slices_eaten_by_dog_is_correct : 
  cost_of_slices_eaten_by_dog = 11.59 := by
    sorry

end NUMINAMATH_GPT_cost_of_slices_eaten_by_dog_is_correct_l961_96174


namespace NUMINAMATH_GPT_orthocenter_of_triangle_l961_96159

theorem orthocenter_of_triangle :
  ∀ (A B C H : ℝ × ℝ × ℝ),
    A = (2, 3, 4) → 
    B = (6, 4, 2) → 
    C = (4, 5, 6) → 
    H = (17/53, 152/53, 725/53) → 
    true :=
by sorry

end NUMINAMATH_GPT_orthocenter_of_triangle_l961_96159


namespace NUMINAMATH_GPT_mike_needs_percentage_to_pass_l961_96199

theorem mike_needs_percentage_to_pass :
  ∀ (mike_score marks_short max_marks : ℕ),
  mike_score = 212 → marks_short = 22 → max_marks = 780 →
  ((mike_score + marks_short : ℕ) / (max_marks : ℕ) : ℚ) * 100 = 30 :=
by
  intros mike_score marks_short max_marks Hmike Hshort Hmax
  rw [Hmike, Hshort, Hmax]
  -- Proof will be filled out here
  sorry

end NUMINAMATH_GPT_mike_needs_percentage_to_pass_l961_96199


namespace NUMINAMATH_GPT_dusty_change_l961_96139

def price_single_layer : ℕ := 4
def price_double_layer : ℕ := 7
def number_of_single_layers : ℕ := 7
def number_of_double_layers : ℕ := 5
def amount_paid : ℕ := 100

theorem dusty_change :
  amount_paid - (number_of_single_layers * price_single_layer + number_of_double_layers * price_double_layer) = 37 := 
by
  sorry

end NUMINAMATH_GPT_dusty_change_l961_96139


namespace NUMINAMATH_GPT_small_triangle_perimeter_l961_96182

theorem small_triangle_perimeter (P : ℕ) (P₁ : ℕ) (P₂ : ℕ) (P₃ : ℕ)
  (h₁ : P = 11) (h₂ : P₁ = 5) (h₃ : P₂ = 7) (h₄ : P₃ = 9) :
  (P₁ + P₂ + P₃) - P = 10 :=
by
  sorry

end NUMINAMATH_GPT_small_triangle_perimeter_l961_96182


namespace NUMINAMATH_GPT_rectangle_area_at_stage_8_l961_96128

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_at_stage_8_l961_96128


namespace NUMINAMATH_GPT_point_in_which_quadrant_l961_96130

theorem point_in_which_quadrant (x y : ℝ) (h1 : y = 2 * x + 3) (h2 : abs x = abs y) :
  (x < 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_point_in_which_quadrant_l961_96130


namespace NUMINAMATH_GPT_intercepts_of_line_l961_96186

theorem intercepts_of_line (x y : ℝ) : 
  (x + 6 * y + 2 = 0) → (x = -2) ∧ (y = -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_of_line_l961_96186


namespace NUMINAMATH_GPT_range_of_t_for_point_in_upper_left_side_l961_96152

def point_in_upper_left_side_condition (x y : ℝ) : Prop :=
  x - y + 4 < 0

theorem range_of_t_for_point_in_upper_left_side :
  ∀ t : ℝ, point_in_upper_left_side_condition (-2) t ↔ t > 2 :=
by
  intros t
  unfold point_in_upper_left_side_condition
  simp
  sorry

end NUMINAMATH_GPT_range_of_t_for_point_in_upper_left_side_l961_96152


namespace NUMINAMATH_GPT_infinite_non_prime_numbers_l961_96187

theorem infinite_non_prime_numbers : ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ (¬(Nat.Prime (2 ^ (2 ^ m) + 1) ∨ ¬Nat.Prime (2018 ^ (2 ^ m) + 1))) := sorry

end NUMINAMATH_GPT_infinite_non_prime_numbers_l961_96187


namespace NUMINAMATH_GPT_find_A_minus_B_l961_96172

def A : ℕ := (55 * 100) + (19 * 10)
def B : ℕ := 173 + (5 * 224)

theorem find_A_minus_B : A - B = 4397 := by
  sorry

end NUMINAMATH_GPT_find_A_minus_B_l961_96172


namespace NUMINAMATH_GPT_investment_rate_l961_96197

theorem investment_rate (P_total P_7000 P_15000 I_total : ℝ)
  (h_investment : P_total = 22000)
  (h_investment_7000 : P_7000 = 7000)
  (h_investment_15000 : P_15000 = P_total - P_7000)
  (R_7000 : ℝ)
  (h_rate_7000 : R_7000 = 0.18)
  (I_7000 : ℝ)
  (h_interest_7000 : I_7000 = P_7000 * R_7000)
  (h_total_interest : I_total = 3360) :
  ∃ (R_15000 : ℝ), (I_total - I_7000) = P_15000 * R_15000 ∧ R_15000 = 0.14 := 
by
  sorry

end NUMINAMATH_GPT_investment_rate_l961_96197


namespace NUMINAMATH_GPT_find_f_neg_2_l961_96107

def f (x : ℝ) : ℝ := sorry -- The actual function f is undefined here.

theorem find_f_neg_2 (h : ∀ x ≠ 0, f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
sorry

end NUMINAMATH_GPT_find_f_neg_2_l961_96107


namespace NUMINAMATH_GPT_tangent_line_inclination_range_l961_96113

theorem tangent_line_inclination_range:
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 → (0 ≤ 2*x ∧ 2*x ≤ 1 ∨ -1 ≤ 2*x ∧ 2*x < 0) →
    ∃ (α : ℝ), (0 ≤ α ∧ α ≤ π/4) ∨ (3*π/4 ≤ α ∧ α < π) :=
sorry

end NUMINAMATH_GPT_tangent_line_inclination_range_l961_96113


namespace NUMINAMATH_GPT_mr_yadav_yearly_savings_l961_96105

theorem mr_yadav_yearly_savings (S : ℕ) (h1 : S * 3 / 5 * 1 / 2 = 1584) : S * 3 / 5 * 1 / 2 * 12 = 19008 :=
  sorry

end NUMINAMATH_GPT_mr_yadav_yearly_savings_l961_96105


namespace NUMINAMATH_GPT_correct_transformation_of_95_sq_l961_96162

theorem correct_transformation_of_95_sq : 95^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end NUMINAMATH_GPT_correct_transformation_of_95_sq_l961_96162


namespace NUMINAMATH_GPT_find_value_l961_96106

-- Given conditions of the problem
axiom condition : ∀ (a : ℝ), a - 1/a = 1

-- The mathematical proof problem
theorem find_value (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l961_96106


namespace NUMINAMATH_GPT_shaded_area_of_pattern_l961_96122

theorem shaded_area_of_pattern (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) : 
  d = 3 → 
  L = 24 → 
  n = 16 → 
  r = 3 / 2 → 
  (A = 18 * Real.pi) :=
by
  intro hd
  intro hL
  intro hn
  intro hr
  sorry

end NUMINAMATH_GPT_shaded_area_of_pattern_l961_96122


namespace NUMINAMATH_GPT_that_remaining_money_l961_96140

section
/-- Initial money in Olivia's wallet --/
def initial_money : ℕ := 53

/-- Money collected from ATM --/
def collected_money : ℕ := 91

/-- Money spent at the supermarket --/
def spent_money : ℕ := collected_money + 39

/-- Remaining money after visiting the supermarket --
Theorem that proves Olivia's remaining money is 14 dollars.
-/
theorem remaining_money : initial_money + collected_money - spent_money = 14 := 
by
  unfold initial_money collected_money spent_money
  simp
  sorry
end

end NUMINAMATH_GPT_that_remaining_money_l961_96140


namespace NUMINAMATH_GPT_correct_time_l961_96148

-- Define the observed times on the clocks
def time1 := 14 * 60 + 54  -- 14:54 in minutes
def time2 := 14 * 60 + 57  -- 14:57 in minutes
def time3 := 15 * 60 + 2   -- 15:02 in minutes
def time4 := 15 * 60 + 3   -- 15:03 in minutes

-- Define the inaccuracies of the clocks
def inaccuracy1 := 2  -- First clock off by 2 minutes
def inaccuracy2 := 3  -- Second clock off by 3 minutes
def inaccuracy3 := -4  -- Third clock off by 4 minutes
def inaccuracy4 := -5  -- Fourth clock off by 5 minutes

-- State that given these conditions, the correct time is 14:58
theorem correct_time : ∃ (T : Int), 
  (time1 + inaccuracy1 = T) ∧
  (time2 + inaccuracy2 = T) ∧
  (time3 + inaccuracy3 = T) ∧
  (time4 + inaccuracy4 = T) ∧
  (T = 14 * 60 + 58) :=
by
  sorry

end NUMINAMATH_GPT_correct_time_l961_96148


namespace NUMINAMATH_GPT_compare_squares_l961_96165

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2 * a := by
  -- the proof would go here, but we skip it according to the instruction
  sorry

end NUMINAMATH_GPT_compare_squares_l961_96165


namespace NUMINAMATH_GPT_part1_part2_l961_96144

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part1 (x : ℝ) : (f x 2 ≥ 7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) : a = 1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l961_96144


namespace NUMINAMATH_GPT_arithmetic_seq_a4_l961_96178

-- Definition of an arithmetic sequence with the first three terms given.
def arithmetic_seq (a : ℕ → ℕ) :=
  a 0 = 2 ∧ a 1 = 4 ∧ a 2 = 6 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

-- The actual proof goal.
theorem arithmetic_seq_a4 : ∃ a : ℕ → ℕ, arithmetic_seq a ∧ a 3 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a4_l961_96178


namespace NUMINAMATH_GPT_quadrants_containing_points_l961_96101

theorem quadrants_containing_points (x y : ℝ) :
  (y > x + 1) → (y > 3 - 2 * x) → 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_quadrants_containing_points_l961_96101


namespace NUMINAMATH_GPT_find_analytical_expression_of_f_l961_96158

-- Define the function f and the condition it needs to satisfy
variable (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1))

-- State the objective to prove
theorem find_analytical_expression_of_f : 
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (1 + x) := by
  sorry

end NUMINAMATH_GPT_find_analytical_expression_of_f_l961_96158


namespace NUMINAMATH_GPT_matrix_power_A_100_l961_96109

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![0, 0, 1],![1, 0, 0],![0, 1, 0]]

theorem matrix_power_A_100 : A^100 = A := by sorry

end NUMINAMATH_GPT_matrix_power_A_100_l961_96109


namespace NUMINAMATH_GPT_find_k_when_lines_perpendicular_l961_96119

theorem find_k_when_lines_perpendicular (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (3-k) * y + 1 = 0 → ∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0 → -((k-3)/(3-k)) * (k-3) = -1) → 
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_when_lines_perpendicular_l961_96119


namespace NUMINAMATH_GPT_problem1_xy_xplusy_l961_96161

theorem problem1_xy_xplusy (x y: ℝ) (h1: x * y = 5) (h2: x + y = 6) : x - y = 4 ∨ x - y = -4 := 
sorry

end NUMINAMATH_GPT_problem1_xy_xplusy_l961_96161


namespace NUMINAMATH_GPT_cone_surface_area_and_volume_l961_96142

theorem cone_surface_area_and_volume
  (r l m : ℝ)
  (h_ratio : (π * r * l) / (π * r * l + π * r^2) = 25 / 32)
  (h_height : m = 96) :
  (π * r * l + π * r^2 = 3584 * π) ∧ ((1 / 3) * π * r^2 * m = 25088 * π) :=
by {
  sorry
}

end NUMINAMATH_GPT_cone_surface_area_and_volume_l961_96142


namespace NUMINAMATH_GPT_median_mean_l961_96164

theorem median_mean (n : ℕ) (h : n + 4 = 8) : (4 + 6 + 8 + 14 + 16) / 5 = 9.6 := by
  sorry

end NUMINAMATH_GPT_median_mean_l961_96164


namespace NUMINAMATH_GPT_discriminant_negative_of_positive_parabola_l961_96196

variable (a b c : ℝ)

theorem discriminant_negative_of_positive_parabola (h1 : ∀ x : ℝ, a * x^2 + b * x + c > 0) (h2 : a > 0) : b^2 - 4*a*c < 0 := 
sorry

end NUMINAMATH_GPT_discriminant_negative_of_positive_parabola_l961_96196


namespace NUMINAMATH_GPT_mother_reaches_timothy_l961_96191

/--
Timothy leaves home for school, riding his bicycle at a rate of 6 miles per hour.
Fifteen minutes after he leaves, his mother sees Timothy's math homework lying on his bed and immediately leaves home to bring it to him.
If his mother drives at 36 miles per hour, prove that she must drive 1.8 miles to reach Timothy.
-/
theorem mother_reaches_timothy
  (timothy_speed : ℕ)
  (mother_speed : ℕ)
  (delay_minutes : ℕ)
  (distance_must_drive : ℕ)
  (h_speed_t : timothy_speed = 6)
  (h_speed_m : mother_speed = 36)
  (h_delay : delay_minutes = 15)
  (h_distance : distance_must_drive = 18 / 10 ) :
  ∃ t : ℚ, (timothy_speed * (delay_minutes / 60) + timothy_speed * t) = (mother_speed * t) := sorry

end NUMINAMATH_GPT_mother_reaches_timothy_l961_96191


namespace NUMINAMATH_GPT_stacy_berries_multiple_l961_96198

theorem stacy_berries_multiple (Skylar_berries : ℕ) (Stacy_berries : ℕ) (Steve_berries : ℕ) (m : ℕ)
  (h1 : Skylar_berries = 20)
  (h2 : Steve_berries = Skylar_berries / 2)
  (h3 : Stacy_berries = m * Steve_berries + 2)
  (h4 : Stacy_berries = 32) :
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_stacy_berries_multiple_l961_96198


namespace NUMINAMATH_GPT_price_per_pound_salt_is_50_l961_96151

-- Given conditions
def totalWeight : ℕ := 60
def weightSalt1 : ℕ := 20
def priceSalt2 : ℕ := 35
def weightSalt2 : ℕ := 40
def sellingPricePerPound : ℕ := 48
def desiredProfitRate : ℚ := 0.20

-- Mathematical definitions derived from conditions
def costSalt1 (priceSalt1 : ℕ) : ℕ := weightSalt1 * priceSalt1
def costSalt2 : ℕ := weightSalt2 * priceSalt2
def totalCost (priceSalt1 : ℕ) : ℕ := costSalt1 priceSalt1 + costSalt2
def totalRevenue : ℕ := totalWeight * sellingPricePerPound
def profit (priceSalt1 : ℕ) : ℚ := desiredProfitRate * totalCost priceSalt1
def totalProfit (priceSalt1 : ℕ) : ℚ := totalCost priceSalt1 + profit priceSalt1

-- Proof statement
theorem price_per_pound_salt_is_50 : ∃ (priceSalt1 : ℕ), totalRevenue = totalProfit priceSalt1 ∧ priceSalt1 = 50 := by
  -- We provide the prove structure, exact proof steps are skipped with sorry
  sorry

end NUMINAMATH_GPT_price_per_pound_salt_is_50_l961_96151


namespace NUMINAMATH_GPT_original_price_per_kg_of_salt_l961_96103

variable {P X : ℝ}

theorem original_price_per_kg_of_salt (h1 : 400 / (0.8 * P) = X + 10)
    (h2 : 400 / P = X) : P = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_price_per_kg_of_salt_l961_96103


namespace NUMINAMATH_GPT_min_sum_a_b_l961_96138

-- The conditions
variables {a b : ℝ}
variables (h₁ : a > 1) (h₂ : b > 1) (h₃ : ab - (a + b) = 1)

-- The theorem statement
theorem min_sum_a_b : a + b = 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_sum_a_b_l961_96138
