import Mathlib

namespace NUMINAMATH_GPT_original_expression_equals_l221_22172

noncomputable def evaluate_expression (a : ℝ) : ℝ :=
  ( (a / (a + 2) + 1 / (a^2 - 4)) / ( (a - 1) / (a + 2) + 1 / (a - 2) ))

theorem original_expression_equals (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  evaluate_expression a = (Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_GPT_original_expression_equals_l221_22172


namespace NUMINAMATH_GPT_negation_statement_contrapositive_statement_l221_22165

variable (x y : ℝ)

theorem negation_statement :
  (¬ ((x-1) * (y+2) ≠ 0 → x ≠ 1 ∧ y ≠ -2)) ↔ ((x-1) * (y+2) = 0 → x = 1 ∨ y = -2) :=
by sorry

theorem contrapositive_statement :
  (x = 1 ∨ y = -2) → ((x-1) * (y+2) = 0) :=
by sorry

end NUMINAMATH_GPT_negation_statement_contrapositive_statement_l221_22165


namespace NUMINAMATH_GPT_zhang_hua_repayment_l221_22111

noncomputable def principal_amount : ℕ := 480000
noncomputable def repayment_period : ℕ := 240
noncomputable def monthly_interest_rate : ℝ := 0.004
noncomputable def principal_payment : ℝ := principal_amount / repayment_period -- 2000, but keeping general form

noncomputable def interest (month : ℕ) : ℝ :=
  (principal_amount - (month - 1) * principal_payment) * monthly_interest_rate

noncomputable def monthly_repayment (month : ℕ) : ℝ :=
  principal_payment + interest month

theorem zhang_hua_repayment (n : ℕ) (h : 1 ≤ n ∧ n ≤ repayment_period) :
  monthly_repayment n = 3928 - 8 * n := 
by
  -- proof would be placed here
  sorry

end NUMINAMATH_GPT_zhang_hua_repayment_l221_22111


namespace NUMINAMATH_GPT_find_c_l221_22138

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 8 = 6) : c = 3 / 2 := 
sorry

end NUMINAMATH_GPT_find_c_l221_22138


namespace NUMINAMATH_GPT_side_length_irrational_l221_22174

theorem side_length_irrational (s : ℝ) (h : s^2 = 3) : ¬∃ (r : ℚ), s = r := by
  sorry

end NUMINAMATH_GPT_side_length_irrational_l221_22174


namespace NUMINAMATH_GPT_student_count_before_new_student_l221_22181

variable {W : ℝ} -- total weight of students before the new student joined
variable {n : ℕ} -- number of students before the new student joined
variable {W_new : ℝ} -- total weight including the new student
variable {n_new : ℕ} -- number of students including the new student

theorem student_count_before_new_student 
  (h1 : W = n * 28) 
  (h2 : W_new = W + 7) 
  (h3 : n_new = n + 1) 
  (h4 : W_new / n_new = 27.3) : n = 29 := 
by
  sorry

end NUMINAMATH_GPT_student_count_before_new_student_l221_22181


namespace NUMINAMATH_GPT_gain_percentage_is_30_l221_22184

def sellingPrice : ℕ := 195
def gain : ℕ := 45
def costPrice : ℕ := sellingPrice - gain

def gainPercentage : ℚ := (gain : ℚ) / (costPrice : ℚ) * 100

theorem gain_percentage_is_30 :
  gainPercentage = 30 := 
sorry

end NUMINAMATH_GPT_gain_percentage_is_30_l221_22184


namespace NUMINAMATH_GPT_five_fourths_of_x_over_3_l221_22190

theorem five_fourths_of_x_over_3 (x : ℚ) : (5/4) * (x/3) = 5 * x / 12 :=
by
  sorry

end NUMINAMATH_GPT_five_fourths_of_x_over_3_l221_22190


namespace NUMINAMATH_GPT_bridge_length_l221_22120

theorem bridge_length (lorry_length : ℝ) (lorry_speed_kmph : ℝ) (cross_time_seconds : ℝ) : 
  lorry_length = 200 ∧ lorry_speed_kmph = 80 ∧ cross_time_seconds = 17.998560115190784 →
  lorry_length + lorry_speed_kmph * (1000 / 3600) * cross_time_seconds = 400 → 
  400 - lorry_length = 200 :=
by
  intro h₁ h₂
  cases h₁
  sorry

end NUMINAMATH_GPT_bridge_length_l221_22120


namespace NUMINAMATH_GPT_original_number_is_1212_or_2121_l221_22179

theorem original_number_is_1212_or_2121 (x y z t : ℕ) (h₁ : t ≠ 0)
  (h₂ : 1000 * x + 100 * y + 10 * z + t + 1000 * t + 100 * x + 10 * y + z = 3333) : 
  (1000 * x + 100 * y + 10 * z + t = 1212) ∨ (1000 * x + 100 * y + 10 * z + t = 2121) :=
sorry

end NUMINAMATH_GPT_original_number_is_1212_or_2121_l221_22179


namespace NUMINAMATH_GPT_tax_difference_is_correct_l221_22161

-- Define the original price and discount rate as constants
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10

-- Define the state and local sales tax rates as constants
def state_sales_tax_rate : ℝ := 0.075
def local_sales_tax_rate : ℝ := 0.07

-- Calculate the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Calculate state and local sales taxes after discount
def state_sales_tax : ℝ := discounted_price * state_sales_tax_rate
def local_sales_tax : ℝ := discounted_price * local_sales_tax_rate

-- Calculate the difference between state and local sales taxes
def tax_difference : ℝ := state_sales_tax - local_sales_tax

-- The proof to show that the difference is 0.225
theorem tax_difference_is_correct : tax_difference = 0.225 := by
  sorry

end NUMINAMATH_GPT_tax_difference_is_correct_l221_22161


namespace NUMINAMATH_GPT_shekar_marks_in_math_l221_22156

theorem shekar_marks_in_math (M : ℕ) : 
  (65 + 82 + 67 + 75 + M) / 5 = 73 → M = 76 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_shekar_marks_in_math_l221_22156


namespace NUMINAMATH_GPT_range_of_a_l221_22102

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- State the main theorem
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime a x1 = 0 ∧ f_prime a x2 = 0) →
  a < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l221_22102


namespace NUMINAMATH_GPT_least_k_l221_22173

noncomputable def u : ℕ → ℝ
| 0 => 1 / 8
| (n + 1) => 3 * u n - 3 * (u n) ^ 2

theorem least_k :
  ∃ k : ℕ, |u k - (1 / 3)| ≤ 1 / 2 ^ 500 ∧ ∀ m < k, |u m - (1 / 3)| > 1 / 2 ^ 500 :=
by
  sorry

end NUMINAMATH_GPT_least_k_l221_22173


namespace NUMINAMATH_GPT_max_expr_value_l221_22199

noncomputable def expr (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_expr_value : 
  ∃ (a b c d : ℝ),
    a ∈ Set.Icc (-5 : ℝ) 5 ∧
    b ∈ Set.Icc (-5 : ℝ) 5 ∧
    c ∈ Set.Icc (-5 : ℝ) 5 ∧
    d ∈ Set.Icc (-5 : ℝ) 5 ∧
    expr a b c d = 110 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_max_expr_value_l221_22199


namespace NUMINAMATH_GPT_add_to_any_integer_l221_22115

theorem add_to_any_integer (y : ℤ) : (∀ x : ℤ, y + x = x) → y = 0 :=
  by
  sorry

end NUMINAMATH_GPT_add_to_any_integer_l221_22115


namespace NUMINAMATH_GPT_perimeter_of_specific_figure_l221_22176

-- Define the grid size and additional column properties as given in the problem
structure Figure :=
  (rows : ℕ)
  (cols : ℕ)
  (additionalCols : ℕ)
  (additionalRows : ℕ)

-- The specific figure properties from the problem statement
def specificFigure : Figure := {
  rows := 3,
  cols := 4,
  additionalCols := 1,
  additionalRows := 2
}

-- Define the perimeter computation
def computePerimeter (fig : Figure) : ℕ :=
  2 * (fig.rows + fig.cols + fig.additionalCols) + fig.additionalRows

theorem perimeter_of_specific_figure : computePerimeter specificFigure = 13 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_specific_figure_l221_22176


namespace NUMINAMATH_GPT_total_amount_l221_22198

theorem total_amount (A B C T : ℝ)
  (h1 : A = 1 / 4 * (B + C))
  (h2 : B = 3 / 5 * (A + C))
  (h3 : A = 20) :
  T = A + B + C → T = 100 := by
  sorry

end NUMINAMATH_GPT_total_amount_l221_22198


namespace NUMINAMATH_GPT_initial_number_of_employees_l221_22168

variables (E : ℕ)
def hourly_rate : ℕ := 12
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def extra_employees : ℕ := 200
def total_payroll : ℕ := 1680000

-- Total hours worked by each employee per month
def monthly_hours_per_employee : ℕ := hours_per_day * days_per_week * weeks_per_month

-- Monthly salary per employee
def monthly_salary_per_employee : ℕ := monthly_hours_per_employee * hourly_rate

-- Condition expressing the constraint given in the problem
def payroll_equation : Prop :=
  (E + extra_employees) * monthly_salary_per_employee = total_payroll

-- The statement we are proving
theorem initial_number_of_employees :
  payroll_equation E → E = 500 :=
by
  -- Proof not required
  intros
  sorry

end NUMINAMATH_GPT_initial_number_of_employees_l221_22168


namespace NUMINAMATH_GPT_find_range_of_m_l221_22194

def equation1 (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0

def equation2 (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 → false

theorem find_range_of_m (m : ℝ) (h1 : equation1 m → m > 2) (h2 : equation2 m → 1 < m ∧ m < 3) :
  (equation1 m ∨ equation2 m) ∧ ¬(equation1 m ∧ equation2 m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l221_22194


namespace NUMINAMATH_GPT_not_divisible_by_1000_pow_m_minus_1_l221_22185

theorem not_divisible_by_1000_pow_m_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_1000_pow_m_minus_1_l221_22185


namespace NUMINAMATH_GPT_sum_of_square_roots_l221_22166

theorem sum_of_square_roots :
  (Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4)) = 
  (1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10) := 
sorry

end NUMINAMATH_GPT_sum_of_square_roots_l221_22166


namespace NUMINAMATH_GPT_aubrey_travel_time_l221_22114

def aubrey_time_to_school (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem aubrey_travel_time :
  aubrey_time_to_school 88 22 = 4 := by
  sorry

end NUMINAMATH_GPT_aubrey_travel_time_l221_22114


namespace NUMINAMATH_GPT_compute_volume_of_cube_l221_22107

-- Define the conditions and required properties
variable (s V : ℝ)

-- Given condition: the surface area of the cube is 384 sq cm
def surface_area (s : ℝ) : Prop := 6 * s^2 = 384

-- Define the volume of the cube
def volume (s : ℝ) (V : ℝ) : Prop := V = s^3

-- Theorem statement to prove the volume is correctly computed
theorem compute_volume_of_cube (h₁ : surface_area s) : volume s 512 :=
  sorry

end NUMINAMATH_GPT_compute_volume_of_cube_l221_22107


namespace NUMINAMATH_GPT_mean_median_mode_relation_l221_22146

-- Defining the data set of the number of fish caught in twelve outings.
def fish_catches : List ℕ := [3, 0, 2, 2, 1, 5, 3, 0, 1, 4, 3, 3]

-- Proof statement to show the relationship among mean, median and mode.
theorem mean_median_mode_relation (hs : fish_catches = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]) :
  let mean := (fish_catches.sum : ℚ) / fish_catches.length
  let median := (fish_catches.nthLe 5 sorry + fish_catches.nthLe 6 sorry : ℚ) / 2
  let mode := 3
  mean < median ∧ median < mode := by
  -- Placeholder for the proof. Details are skipped here.
  sorry

end NUMINAMATH_GPT_mean_median_mode_relation_l221_22146


namespace NUMINAMATH_GPT_smallest_m_l221_22178

theorem smallest_m (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  ∃ m, (∀ (a b c : ℝ), a + b + c = 1 → 0 < a → 0 < b → 0 < c → m * (a ^ 3 + b ^ 3 + c ^ 3) ≥ 6 * (a ^ 2 + b ^ 2 + c ^ 2) + 1) ↔ m = 27 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_l221_22178


namespace NUMINAMATH_GPT_students_answered_both_correctly_l221_22118

theorem students_answered_both_correctly 
(total_students : ℕ) 
(did_not_answer_A_correctly : ℕ) 
(answered_A_correctly_but_not_B : ℕ) 
(h1 : total_students = 50) 
(h2 : did_not_answer_A_correctly = 12) 
(h3 : answered_A_correctly_but_not_B = 30) : 
    (total_students - did_not_answer_A_correctly - answered_A_correctly_but_not_B) = 8 :=
by
    sorry

end NUMINAMATH_GPT_students_answered_both_correctly_l221_22118


namespace NUMINAMATH_GPT_union_example_l221_22162

open Set

variable (A B : Set ℤ)
variable (AB : Set ℤ)

theorem union_example (hA : A = {-3, 1, 2})
                      (hB : B = {0, 1, 2, 3}) :
                      A ∪ B = {-3, 0, 1, 2, 3} :=
by
  rw [hA, hB]
  ext
  simp
  sorry

end NUMINAMATH_GPT_union_example_l221_22162


namespace NUMINAMATH_GPT_number_of_buses_in_month_l221_22141

-- Given conditions
def weekday_buses := 36
def saturday_buses := 24
def sunday_holiday_buses := 12
def num_weekdays := 18
def num_saturdays := 4
def num_sundays_holidays := 6

-- Statement to prove
theorem number_of_buses_in_month : 
  num_weekdays * weekday_buses + num_saturdays * saturday_buses + num_sundays_holidays * sunday_holiday_buses = 816 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_buses_in_month_l221_22141


namespace NUMINAMATH_GPT_decreasing_function_l221_22136

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem decreasing_function (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1) : 
  f x₁ > f x₂ :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_decreasing_function_l221_22136


namespace NUMINAMATH_GPT_coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l221_22106

/- The problem's conditions and questions rephrased for Lean:
  1. Prove: if \( n \) is divisible by 3, then a valid coloring is possible.
  2. Prove: if a valid coloring is possible, then \( n \) is divisible by 3.
-/

def is_colorable (n : ℕ) : Prop :=
  ∃ (colors : Fin 3 → Fin n → Fin 3),
    ∀ (i j : Fin n), i ≠ j → (colors 0 i ≠ colors 0 j ∧ colors 1 i ≠ colors 1 j ∧ colors 2 i ≠ colors 2 j)

theorem coloring_possible_if_divisible_by_three (n : ℕ) (h : n % 3 = 0) : is_colorable n :=
  sorry

theorem divisible_by_three_if_coloring_possible (n : ℕ) (h : is_colorable n) : n % 3 = 0 :=
  sorry

end NUMINAMATH_GPT_coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l221_22106


namespace NUMINAMATH_GPT_part1_part2_l221_22171

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem part1 : {x : ℝ | f x ≤ 5} = {x : ℝ | -7 / 4 ≤ x ∧ x ≤ 3 / 4} :=
sorry

theorem part2 (h : ∃ x : ℝ, f x < |m - 2|) : m > 6 ∨ m < -2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l221_22171


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l221_22134

theorem simplify_and_evaluate_expr (a b : ℤ) (h₁ : a = -1) (h₂ : b = 2) :
  (2 * a + b - 2 * (3 * a - 2 * b)) = 14 := by
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l221_22134


namespace NUMINAMATH_GPT_students_tried_out_l221_22135

theorem students_tried_out (x : ℕ) (h1 : 8 * (x - 17) = 384) : x = 65 := 
by
  sorry

end NUMINAMATH_GPT_students_tried_out_l221_22135


namespace NUMINAMATH_GPT_compute_fraction_sum_l221_22105

theorem compute_fraction_sum
  (a b c : ℝ)
  (h : a^3 - 6 * a^2 + 11 * a = 12)
  (h : b^3 - 6 * b^2 + 11 * b = 12)
  (h : c^3 - 6 * c^2 + 11 * c = 12) :
  (ab : ℝ) / c + (bc : ℝ) / a + (ca : ℝ) / b = -23 / 12 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_sum_l221_22105


namespace NUMINAMATH_GPT_polynomial_solutions_l221_22196

-- Define the type of the polynomials and statement of the problem
def P1 (x : ℝ) : ℝ := x
def P2 (x : ℝ) : ℝ := x^2 + 1
def P3 (x : ℝ) : ℝ := x^4 + 2*x^2 + 2

theorem polynomial_solutions :
  (∀ x : ℝ, P1 (x^2 + 1) = P1 x^2 + 1) ∧
  (∀ x : ℝ, P2 (x^2 + 1) = P2 x^2 + 1) ∧
  (∀ x : ℝ, P3 (x^2 + 1) = P3 x^2 + 1) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_polynomial_solutions_l221_22196


namespace NUMINAMATH_GPT_lock_probability_l221_22126

/-- The probability of correctly guessing the last digit of a three-digit combination lock,
given that the first two digits are correctly set and each digit ranges from 0 to 9. -/
theorem lock_probability : 
  ∀ (d1 d2 : ℕ), 
  (0 ≤ d1 ∧ d1 < 10) ∧ (0 ≤ d2 ∧ d2 < 10) →
  (0 ≤ d3 ∧ d3 < 10) → 
  (1/10 : ℝ) = (1 : ℝ) / (10 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_lock_probability_l221_22126


namespace NUMINAMATH_GPT_personal_trainer_cost_proof_l221_22108

-- Define the conditions
def hourly_wage_before_raise : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 5
def old_bills_per_week : ℝ := 600
def leftover_money : ℝ := 980

-- Define the question
def new_hourly_wage : ℝ := hourly_wage_before_raise * (1 + raise_percentage)
def weekly_hours : ℕ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def total_weekly_expenses : ℝ := weekly_earnings - leftover_money
def personal_trainer_cost_per_week : ℝ := total_weekly_expenses - old_bills_per_week

-- Theorem statement
theorem personal_trainer_cost_proof : personal_trainer_cost_per_week = 100 := 
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_personal_trainer_cost_proof_l221_22108


namespace NUMINAMATH_GPT_speed_limit_correct_l221_22197

def speed_limit_statement (v : ℝ) : Prop :=
  v ≤ 70

theorem speed_limit_correct (v : ℝ) (h : v ≤ 70) : speed_limit_statement v :=
by
  exact h

#print axioms speed_limit_correct

end NUMINAMATH_GPT_speed_limit_correct_l221_22197


namespace NUMINAMATH_GPT_verify_expressions_l221_22169

variable (x y : ℝ)
variable (h : x / y = 5 / 3)

theorem verify_expressions :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / -7 ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
sorry

end NUMINAMATH_GPT_verify_expressions_l221_22169


namespace NUMINAMATH_GPT_percentage_of_cars_on_monday_compared_to_tuesday_l221_22128

theorem percentage_of_cars_on_monday_compared_to_tuesday : 
  ∀ (cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun : ℕ),
    cars_mon + cars_tue + cars_wed + cars_thu + cars_fri + cars_sat + cars_sun = 97 →
    cars_tue = 25 →
    cars_wed = cars_mon + 2 →
    cars_thu = 10 →
    cars_fri = 10 →
    cars_sat = 5 →
    cars_sun = 5 →
    (cars_mon * 100 / cars_tue = 80) :=
by
  intros cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun
  intro h_total
  intro h_tue
  intro h_wed
  intro h_thu
  intro h_fri
  intro h_sat
  intro h_sun
  sorry

end NUMINAMATH_GPT_percentage_of_cars_on_monday_compared_to_tuesday_l221_22128


namespace NUMINAMATH_GPT_odd_pair_exists_k_l221_22132

theorem odd_pair_exists_k (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) : 
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := 
sorry

end NUMINAMATH_GPT_odd_pair_exists_k_l221_22132


namespace NUMINAMATH_GPT_circle_length_l221_22104

theorem circle_length (n : ℕ) (arm_span : ℝ) (overlap : ℝ) (contribution : ℝ) (total_length : ℝ) :
  n = 16 ->
  arm_span = 10.4 ->
  overlap = 3.5 ->
  contribution = arm_span - overlap ->
  total_length = n * contribution ->
  total_length = 110.4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_circle_length_l221_22104


namespace NUMINAMATH_GPT_average_buns_per_student_l221_22151

theorem average_buns_per_student (packages_class1 packages_class2 packages_class3 packages_class4 : ℕ)
    (buns_per_package students_per_class stale_buns uneaten_buns : ℕ)
    (h1 : packages_class1 = 20)
    (h2 : packages_class2 = 25)
    (h3 : packages_class3 = 30)
    (h4 : packages_class4 = 35)
    (h5 : buns_per_package = 8)
    (h6 : students_per_class = 30)
    (h7 : stale_buns = 16)
    (h8 : uneaten_buns = 20) :
  let total_buns_class1 := packages_class1 * buns_per_package
  let total_buns_class2 := packages_class2 * buns_per_package
  let total_buns_class3 := packages_class3 * buns_per_package
  let total_buns_class4 := packages_class4 * buns_per_package
  let total_uneaten_buns := stale_buns + uneaten_buns
  let uneaten_buns_per_class := total_uneaten_buns / 4
  let remaining_buns_class1 := total_buns_class1 - uneaten_buns_per_class
  let remaining_buns_class2 := total_buns_class2 - uneaten_buns_per_class
  let remaining_buns_class3 := total_buns_class3 - uneaten_buns_per_class
  let remaining_buns_class4 := total_buns_class4 - uneaten_buns_per_class
  let avg_buns_class1 := remaining_buns_class1 / students_per_class
  let avg_buns_class2 := remaining_buns_class2 / students_per_class
  let avg_buns_class3 := remaining_buns_class3 / students_per_class
  let avg_buns_class4 := remaining_buns_class4 / students_per_class
  avg_buns_class1 = 5 ∧ avg_buns_class2 = 6 ∧ avg_buns_class3 = 7 ∧ avg_buns_class4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_buns_per_student_l221_22151


namespace NUMINAMATH_GPT_log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l221_22195

theorem log_one_plus_xsq_lt_xsq_over_one_plus_xsq (x : ℝ) (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 / (1 + x^2) :=
sorry

end NUMINAMATH_GPT_log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l221_22195


namespace NUMINAMATH_GPT_angle_measure_l221_22112

-- Define the problem conditions
def angle (x : ℝ) : Prop :=
  let complement := 3 * x + 6
  x + complement = 90

-- The theorem to prove
theorem angle_measure : ∃ x : ℝ, angle x ∧ x = 21 := 
sorry

end NUMINAMATH_GPT_angle_measure_l221_22112


namespace NUMINAMATH_GPT_solution_set_16_sin_pi_x_cos_pi_x_l221_22145

theorem solution_set_16_sin_pi_x_cos_pi_x (x : ℝ) :
  (x = 1 / 4 ∨ x = -1 / 4) ↔ 16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x :=
sorry

end NUMINAMATH_GPT_solution_set_16_sin_pi_x_cos_pi_x_l221_22145


namespace NUMINAMATH_GPT_cubic_identity_l221_22167

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 40) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1575 := 
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l221_22167


namespace NUMINAMATH_GPT_closest_fraction_l221_22148

theorem closest_fraction :
  let won_france := (23 : ℝ) / 120
  let fractions := [ (1 : ℝ) / 4, (1 : ℝ) / 5, (1 : ℝ) / 6, (1 : ℝ) / 7, (1 : ℝ) / 8 ]
  ∃ closest : ℝ, closest ∈ fractions ∧ ∀ f ∈ fractions, abs (won_france - closest) ≤ abs (won_france - f)  :=
  sorry

end NUMINAMATH_GPT_closest_fraction_l221_22148


namespace NUMINAMATH_GPT_rationalize_denominator_sum_l221_22127

noncomputable def rationalize_denominator (x y z : ℤ) :=
  x = 4 ∧ y = 49 ∧ z = 35 ∧ y ∣ 343 ∧ z > 0 

theorem rationalize_denominator_sum : 
  ∃ A B C : ℤ, rationalize_denominator A B C ∧ A + B + C = 88 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_sum_l221_22127


namespace NUMINAMATH_GPT_parabola_find_m_l221_22192

theorem parabola_find_m
  (p m : ℝ) (h_p_pos : p > 0) (h_point_on_parabola : (2 * p * m) = 8)
  (h_chord_length : (m + (2 / m))^2 - m^2 = 7) : m = (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_GPT_parabola_find_m_l221_22192


namespace NUMINAMATH_GPT_john_mary_game_l221_22121

theorem john_mary_game (n : ℕ) (h : n ≥ 3) :
  ∃ S : ℕ, S = n * (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_john_mary_game_l221_22121


namespace NUMINAMATH_GPT_annual_interest_correct_l221_22130

-- Define the conditions
def Rs_total : ℝ := 3400
def P1 : ℝ := 1300
def P2 : ℝ := Rs_total - P1
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

-- Define the interests
def Interest1 : ℝ := P1 * Rate1
def Interest2 : ℝ := P2 * Rate2

-- The total interest
def Total_Interest : ℝ := Interest1 + Interest2

-- The theorem to prove
theorem annual_interest_correct :
  Total_Interest = 144 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_correct_l221_22130


namespace NUMINAMATH_GPT_maria_baggies_l221_22149

-- Definitions of the conditions
def total_cookies (chocolate_chip : Nat) (oatmeal : Nat) : Nat :=
  chocolate_chip + oatmeal

def cookies_per_baggie : Nat :=
  3

def number_of_baggies (total_cookies : Nat) (cookies_per_baggie : Nat) : Nat :=
  total_cookies / cookies_per_baggie

-- Proof statement
theorem maria_baggies :
  number_of_baggies (total_cookies 2 16) cookies_per_baggie = 6 := 
sorry

end NUMINAMATH_GPT_maria_baggies_l221_22149


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l221_22143

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 720 ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → 
    (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ∉ [2, 5, 7, 9])) := 
sorry

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l221_22143


namespace NUMINAMATH_GPT_crayons_end_of_school_year_l221_22183

-- Definitions based on conditions
def crayons_after_birthday : Float := 479.0
def total_crayons_now : Float := 613.0

-- The mathematically equivalent proof problem statement
theorem crayons_end_of_school_year : (total_crayons_now - crayons_after_birthday = 134.0) :=
by
  sorry

end NUMINAMATH_GPT_crayons_end_of_school_year_l221_22183


namespace NUMINAMATH_GPT_dawn_lemonade_price_l221_22152

theorem dawn_lemonade_price (x : ℕ) : 
  (10 * 25) = (8 * x) + 26 → x = 28 :=
by 
  sorry

end NUMINAMATH_GPT_dawn_lemonade_price_l221_22152


namespace NUMINAMATH_GPT_certain_events_l221_22122

-- Define the idioms and their classifications
inductive Event
| impossible
| certain
| unlikely

-- Definitions based on the given conditions
def scooping_moon := Event.impossible
def rising_tide := Event.certain
def waiting_by_stump := Event.unlikely
def catching_turtles := Event.certain
def pulling_seeds := Event.impossible

-- The theorem statement
theorem certain_events :
  (rising_tide = Event.certain) ∧ (catching_turtles = Event.certain) := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_certain_events_l221_22122


namespace NUMINAMATH_GPT_max_min_of_f_l221_22170

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * Real.pi + x) + 
  Real.sqrt 3 * Real.cos (2 * Real.pi - x) -
  Real.sin (2013 * Real.pi + Real.pi / 6)

theorem max_min_of_f : 
  - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 →
  (-1 / 2) ≤ f x ∧ f x ≤ 5 / 2 :=
sorry

end NUMINAMATH_GPT_max_min_of_f_l221_22170


namespace NUMINAMATH_GPT_slope_point_on_line_l221_22116

theorem slope_point_on_line (b : ℝ) (h1 : ∃ x, x + b = 30) (h2 : (b / (30 - b)) = 4) : b = 24 :=
  sorry

end NUMINAMATH_GPT_slope_point_on_line_l221_22116


namespace NUMINAMATH_GPT_greatest_ratio_AB_CD_on_circle_l221_22140

/-- The statement proving the greatest possible value of the ratio AB/CD for points A, B, C, D lying on the 
circle x^2 + y^2 = 16 with integer coordinates and unequal distances AB and CD is sqrt 10 / 3. -/
theorem greatest_ratio_AB_CD_on_circle :
  ∀ (A B C D : ℤ × ℤ), A ≠ B → C ≠ D → 
  A.1^2 + A.2^2 = 16 → B.1^2 + B.2^2 = 16 → 
  C.1^2 + C.2^2 = 16 → D.1^2 + D.2^2 = 16 → 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let ratio := AB / CD
  AB ≠ CD →
  ratio ≤ Real.sqrt 10 / 3 :=
sorry

end NUMINAMATH_GPT_greatest_ratio_AB_CD_on_circle_l221_22140


namespace NUMINAMATH_GPT_find_pairs_l221_22182

theorem find_pairs (n k : ℕ) (h_pos_n : 0 < n) (h_cond : n! + n = n ^ k) : 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) := 
by 
  sorry

end NUMINAMATH_GPT_find_pairs_l221_22182


namespace NUMINAMATH_GPT_part1_part2_part3_l221_22160

-- Definition of a companion point
structure Point where
  x : ℝ
  y : ℝ

def isCompanion (P Q : Point) : Prop :=
  Q.x = P.x + 2 ∧ Q.y = P.y - 4

-- Part (1) proof statement
theorem part1 (P Q : Point) (hPQ : isCompanion P Q) (hP : P = ⟨2, -1⟩) (hQ : Q.y = -20 / Q.x) : Q.x = 4 ∧ Q.y = -5 ∧ -20 / 4 = -5 :=
  sorry

-- Part (2) proof statement
theorem part2 (P Q : Point) (hPQ : isCompanion P Q) (hPLine : P.y = P.x - (-5)) (hQ : Q = ⟨-1, -2⟩) : P.x = -3 ∧ P.y = -3 - (-5) ∧ Q.x = -1 ∧ Q.y = -2 :=
  sorry

-- Part (3) proof statement
noncomputable def line2 (Q : Point) := 2*Q.x - 5

theorem part3 (P Q : Point) (hPQ : isCompanion P Q) (hP : P.y = 2*P.x + 3) (hQLine : Q.y = line2 Q) : line2 Q = 2*(P.x + 2) - 5 :=
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l221_22160


namespace NUMINAMATH_GPT_longest_chord_in_circle_l221_22177

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end NUMINAMATH_GPT_longest_chord_in_circle_l221_22177


namespace NUMINAMATH_GPT_percentage_discount_total_amount_paid_l221_22142

variable (P Q : ℝ)

theorem percentage_discount (h₁ : P > Q) (h₂ : Q > 0) :
  100 * ((P - Q) / P) = 100 * (P - Q) / P :=
sorry

theorem total_amount_paid (h₁ : P > Q) (h₂ : Q > 0) :
  10 * Q = 10 * Q :=
sorry

end NUMINAMATH_GPT_percentage_discount_total_amount_paid_l221_22142


namespace NUMINAMATH_GPT_value_of_D_l221_22186

theorem value_of_D (E F D : ℕ) (cond1 : E + F + D = 15) (cond2 : F + E = 11) : D = 4 := 
by
  sorry

end NUMINAMATH_GPT_value_of_D_l221_22186


namespace NUMINAMATH_GPT_math_problem_l221_22150

noncomputable def x : ℝ := 24

theorem math_problem : ∀ (x : ℝ), x = 3/8 * x + 15 → x = 24 := 
by 
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_math_problem_l221_22150


namespace NUMINAMATH_GPT_scientific_notation_of_8_5_million_l221_22124

theorem scientific_notation_of_8_5_million :
  (8.5 * 10^6) = 8500000 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_8_5_million_l221_22124


namespace NUMINAMATH_GPT_trig_identity_l221_22123

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l221_22123


namespace NUMINAMATH_GPT_original_price_eq_36_l221_22131

-- Definitions for the conditions
def first_cup_price (x : ℕ) : ℕ := x
def second_cup_price (x : ℕ) : ℕ := x / 2
def third_cup_price : ℕ := 3
def total_cost (x : ℕ) : ℕ := x + (x / 2) + third_cup_price
def average_price (total : ℕ) : ℕ := total / 3

-- The proof statement
theorem original_price_eq_36 (x : ℕ) (h : total_cost x = 57) : x = 36 :=
  sorry

end NUMINAMATH_GPT_original_price_eq_36_l221_22131


namespace NUMINAMATH_GPT_number_to_add_l221_22189

theorem number_to_add (a b n : ℕ) (h_a : a = 425897) (h_b : b = 456) (h_n : n = 47) : 
  (a + n) % b = 0 :=
by
  rw [h_a, h_b, h_n]
  sorry

end NUMINAMATH_GPT_number_to_add_l221_22189


namespace NUMINAMATH_GPT_remainder_is_6910_l221_22193

def polynomial (x : ℝ) : ℝ := 5 * x^7 - 3 * x^6 - 8 * x^5 + 3 * x^3 + 5 * x^2 - 20

def divisor (x : ℝ) : ℝ := 3 * x - 9

theorem remainder_is_6910 : polynomial 3 = 6910 := by
  sorry

end NUMINAMATH_GPT_remainder_is_6910_l221_22193


namespace NUMINAMATH_GPT_point_above_line_l221_22119

/-- Given the point (-2, t) lies above the line x - 2y + 4 = 0,
    we want to prove t ∈ (1, +∞) -/
theorem point_above_line (t : ℝ) : (-2 - 2 * t + 4 > 0) → t > 1 :=
sorry

end NUMINAMATH_GPT_point_above_line_l221_22119


namespace NUMINAMATH_GPT_intersection_of_sets_l221_22133

def M (x : ℝ) : Prop := (x - 2) / (x - 3) < 0
def N (x : ℝ) : Prop := Real.log (x - 2) / Real.log (1 / 2) ≥ 1 

theorem intersection_of_sets : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 2 < x ∧ x ≤ 5 / 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l221_22133


namespace NUMINAMATH_GPT_tea_blend_ratio_l221_22113

theorem tea_blend_ratio (x y : ℝ)
  (h1 : 18 * x + 20 * y = (21 * (x + y)) / 1.12)
  (h2 : x + y ≠ 0) :
  x / y = 5 / 3 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_tea_blend_ratio_l221_22113


namespace NUMINAMATH_GPT_find_abc_sum_l221_22154

theorem find_abc_sum {U : Type} 
  (a b c : ℕ)
  (ha : a = 26)
  (hb : b = 1)
  (hc : c = 32)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  a + b + c = 59 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_sum_l221_22154


namespace NUMINAMATH_GPT_inradius_inequality_l221_22163

/-- Given a point P inside the triangle ABC, where da, db, and dc are the distances from P to the sides BC, CA, and AB respectively,
 and r is the inradius of the triangle ABC, prove the inequality -/
theorem inradius_inequality (a b c da db dc : ℝ) (r : ℝ) 
  (h1 : 0 < da) (h2 : 0 < db) (h3 : 0 < dc)
  (h4 : r = (a * da + b * db + c * dc) / (a + b + c)) :
  2 / (1 / da + 1 / db + 1 / dc) < r ∧ r < (da + db + dc) / 2 :=
  sorry

end NUMINAMATH_GPT_inradius_inequality_l221_22163


namespace NUMINAMATH_GPT_radius_of_intersection_l221_22175

noncomputable def sphere_radius := 2 * Real.sqrt 17

theorem radius_of_intersection (s : ℝ) 
  (h1 : (3:ℝ)=(3:ℝ)) (h2 : (5:ℝ)=(5:ℝ)) (h3 : (0-3:ℝ)^2 + (5-5:ℝ)^2 + (s-(-8+8))^2 = sphere_radius^2) :
  s = Real.sqrt 59 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_intersection_l221_22175


namespace NUMINAMATH_GPT_cost_per_bag_l221_22157

theorem cost_per_bag (C : ℝ)
  (total_bags : ℕ := 20)
  (price_per_bag_original : ℝ := 6)
  (sold_original : ℕ := 15)
  (price_per_bag_discounted : ℝ := 4)
  (sold_discounted : ℕ := 5)
  (net_profit : ℝ := 50) :
  sold_original * price_per_bag_original + sold_discounted * price_per_bag_discounted - net_profit = total_bags * C →
  C = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cost_per_bag_l221_22157


namespace NUMINAMATH_GPT_number_of_paths_from_C_to_D_l221_22155

-- Define the grid and positions
def C := (0,0)  -- Bottom-left corner
def D := (7,3)  -- Top-right corner
def gridWidth : ℕ := 7
def gridHeight : ℕ := 3

-- Define the binomial coefficient function
-- Note: Lean already has binomial coefficient defined in Mathlib, use Nat.choose for that

-- The statement to prove
theorem number_of_paths_from_C_to_D : Nat.choose (gridWidth + gridHeight) gridHeight = 120 :=
by
  sorry

end NUMINAMATH_GPT_number_of_paths_from_C_to_D_l221_22155


namespace NUMINAMATH_GPT_max_unique_dance_counts_l221_22158

theorem max_unique_dance_counts (boys girls : ℕ) (positive_boys : boys = 29) (positive_girls : girls = 15) 
  (dances : ∀ b g, b ≤ boys → g ≤ girls → ℕ) :
  ∃ num_dances, num_dances = 29 := 
by
  sorry

end NUMINAMATH_GPT_max_unique_dance_counts_l221_22158


namespace NUMINAMATH_GPT_value_of_x_l221_22164

theorem value_of_x (x : ℕ) : (1 / 16) * (2 ^ 20) = 4 ^ x → x = 8 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l221_22164


namespace NUMINAMATH_GPT_area_of_triangle_is_correct_l221_22180

def line_1 (x y : ℝ) : Prop := y - 5 * x = -4
def line_2 (x y : ℝ) : Prop := 4 * y + 2 * x = 16

def y_axis (x y : ℝ) : Prop := x = 0

def satisfies_y_intercepts (f : ℝ → ℝ) : Prop :=
f 0 = -4 ∧ f 0 = 4

noncomputable def area_of_triangle (height base : ℝ) : ℝ :=
(1 / 2) * base * height

theorem area_of_triangle_is_correct :
  ∃ (x y : ℝ), line_1 x y ∧ line_2 x y ∧ y_axis 0 8 ∧ area_of_triangle (16 / 11) 8 = (64 / 11) := 
sorry

end NUMINAMATH_GPT_area_of_triangle_is_correct_l221_22180


namespace NUMINAMATH_GPT_range_of_a_l221_22100

theorem range_of_a (x a : ℝ) :
  (∀ x, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l221_22100


namespace NUMINAMATH_GPT_mr_slinkums_shipments_l221_22103

theorem mr_slinkums_shipments 
  (T : ℝ) 
  (h : (3 / 4) * T = 150) : 
  T = 200 := 
sorry

end NUMINAMATH_GPT_mr_slinkums_shipments_l221_22103


namespace NUMINAMATH_GPT_base_length_of_prism_l221_22137

theorem base_length_of_prism (V : ℝ) (hV : V = 36 * Real.pi) : ∃ (AB : ℝ), AB = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_base_length_of_prism_l221_22137


namespace NUMINAMATH_GPT_find_coordinates_of_B_l221_22101

theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (h1 : ∃ (C1 C2 : ℝ × ℝ), C1.2 = 0 ∧ C2.2 = 0 ∧ (dist C1 A = dist C1 B) ∧ (dist C2 A = dist C2 B) ∧ (A ≠ B))
  (h2 : A = (-3, 2)) :
  B = (-3, -2) :=
sorry

end NUMINAMATH_GPT_find_coordinates_of_B_l221_22101


namespace NUMINAMATH_GPT_max_points_of_intersection_l221_22191

-- Definitions based on the conditions in a)
def intersects_circle (l : ℕ) : ℕ := 2 * l  -- Each line intersects the circle at most twice
def intersects_lines (n : ℕ) : ℕ := n * (n - 1) / 2  -- Number of intersection points between lines (combinatorial)

-- The main statement that needs to be proved
theorem max_points_of_intersection (lines circle : ℕ) (h_lines_distinct : lines = 3) (h_no_parallel : ∀ (i j : ℕ), i ≠ j → i < lines → j < lines → true) (h_no_common_point : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(true)) : (intersects_circle lines + intersects_lines lines = 9) := 
  by
    sorry

end NUMINAMATH_GPT_max_points_of_intersection_l221_22191


namespace NUMINAMATH_GPT_ellipse_condition_l221_22125

theorem ellipse_condition (k : ℝ) : 
  (k > 1 ↔ 
  (k - 1 > 0 ∧ k + 1 > 0 ∧ k - 1 ≠ k + 1)) :=
by sorry

end NUMINAMATH_GPT_ellipse_condition_l221_22125


namespace NUMINAMATH_GPT_complement_of_M_l221_22144

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 2*x > 0 }
def complement (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

theorem complement_of_M :
  complement U M = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l221_22144


namespace NUMINAMATH_GPT_rectangle_side_length_l221_22147

theorem rectangle_side_length (a c : ℝ) (h_ratio : a / c = 3 / 4) (hc : c = 4) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_length_l221_22147


namespace NUMINAMATH_GPT_sum_nth_beginning_end_l221_22188

theorem sum_nth_beginning_end (n : ℕ) (F L : ℤ) (M : ℤ) 
  (consecutive : ℤ → ℤ) (median : M = 60) 
  (median_formula : M = (F + L) / 2) :
  n = n → F + L = 120 :=
by
  sorry

end NUMINAMATH_GPT_sum_nth_beginning_end_l221_22188


namespace NUMINAMATH_GPT_solve_inequality_l221_22117

theorem solve_inequality (x : ℝ) : 
  (3 * x - 6 > 12 - 2 * x + x^2) ↔ (-1 < x ∧ x < 6) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l221_22117


namespace NUMINAMATH_GPT_find_m_for_integer_solution_l221_22187

theorem find_m_for_integer_solution :
  ∀ (m x : ℤ), (x^3 - m*x^2 + m*x - (m^2 + 1) = 0) → (m = -3 ∨ m = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_integer_solution_l221_22187


namespace NUMINAMATH_GPT_find_max_marks_l221_22139

variable (marks_scored : ℕ) -- 212
variable (shortfall : ℕ) -- 22
variable (pass_percentage : ℝ) -- 0.30

theorem find_max_marks (h_marks : marks_scored = 212) 
                       (h_short : shortfall = 22) 
                       (h_pass : pass_percentage = 0.30) : 
  ∃ M : ℝ, M = 780 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_max_marks_l221_22139


namespace NUMINAMATH_GPT_maximize_area_l221_22110

theorem maximize_area (P L W : ℝ) (h1 : P = 2 * L + 2 * W) (h2 : 0 < P) : 
  (L = P / 4) ∧ (W = P / 4) :=
by
  sorry

end NUMINAMATH_GPT_maximize_area_l221_22110


namespace NUMINAMATH_GPT_interest_rate_eq_ten_l221_22153

theorem interest_rate_eq_ten (R : ℝ) (P : ℝ) (SI CI : ℝ) :
  P = 1400 ∧
  SI = 14 * R ∧
  CI = 1400 * ((1 + R / 200) ^ 2 - 1) ∧
  CI - SI = 3.50 → 
  R = 10 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_eq_ten_l221_22153


namespace NUMINAMATH_GPT_find_product_of_M1_M2_l221_22109

theorem find_product_of_M1_M2 (x M1 M2 : ℝ) 
  (h : (27 * x - 19) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) : 
  M1 * M2 = -2170 := 
sorry

end NUMINAMATH_GPT_find_product_of_M1_M2_l221_22109


namespace NUMINAMATH_GPT_complex_number_quadrant_l221_22129

def imaginary_unit := Complex.I

def complex_simplification (z : Complex) : Complex :=
  z

theorem complex_number_quadrant :
  ∃ z : Complex, z = (5 * imaginary_unit) / (2 + imaginary_unit ^ 9) ∧ (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l221_22129


namespace NUMINAMATH_GPT_opponents_team_points_l221_22159

theorem opponents_team_points (M D V O : ℕ) (hM : M = 5) (hD : D = 3) 
    (hV : V = 2 * (M + D)) (hO : O = (M + D + V) + 16) : O = 40 := by
  sorry

end NUMINAMATH_GPT_opponents_team_points_l221_22159
