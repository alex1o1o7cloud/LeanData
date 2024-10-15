import Mathlib

namespace NUMINAMATH_CALUDE_intersection_M_N_l658_65873

def M : Set ℝ := {x | Real.log (x + 1) > 0}
def N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l658_65873


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l658_65876

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l658_65876


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l658_65860

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + a*x + 1

-- Define what it means for x to be a fixed point of f
def is_fixed_point (a x : ℝ) : Prop := f a x = x

-- Theorem statement
theorem no_fixed_points_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬(is_fixed_point a x)) ↔ (-1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l658_65860


namespace NUMINAMATH_CALUDE_jaces_debt_jaces_debt_value_l658_65841

theorem jaces_debt (earned : ℝ) (gave_away_cents : ℕ) (current_balance : ℝ) : ℝ :=
  let gave_away : ℝ := (gave_away_cents : ℝ) / 100
  let debt : ℝ := earned - (current_balance + gave_away)
  debt

theorem jaces_debt_value : jaces_debt 1000 358 642 = 354.42 := by sorry

end NUMINAMATH_CALUDE_jaces_debt_jaces_debt_value_l658_65841


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l658_65889

theorem sqrt_equation_solution :
  ∃! (x : ℝ), Real.sqrt (3 * x + 7) - Real.sqrt (2 * x - 1) + 2 = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l658_65889


namespace NUMINAMATH_CALUDE_max_min_sum_absolute_value_l658_65831

theorem max_min_sum_absolute_value (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0)
  (h2 : x + y - 1 ≥ 0)
  (h3 : 3 * x - y - 3 ≤ 0) :
  ∃ (z_max z_min : ℝ),
    (∀ (x' y' : ℝ), 
      x' - y' + 1 ≥ 0 → 
      x' + y' - 1 ≥ 0 → 
      3 * x' - y' - 3 ≤ 0 → 
      |x' - 4 * y' + 1| ≤ z_max ∧ 
      |x' - 4 * y' + 1| ≥ z_min) ∧
    z_max + z_min = 11 / Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_absolute_value_l658_65831


namespace NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l658_65807

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) : (100 ^ 10 : ℕ) = n * 10 → n = 10 ^ 19 := by
  sorry

end NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l658_65807


namespace NUMINAMATH_CALUDE_expression_equality_l658_65843

theorem expression_equality : (2^1501 + 5^1500)^2 - (2^1501 - 5^1500)^2 = 8 * 10^1500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l658_65843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l658_65839

-- Define the sequence a_n
def a (n : ℕ+) : ℕ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ+) : ℕ := n * n

-- State the theorem
theorem arithmetic_sequence_inequality (m k p : ℕ+) (h : m + p = 2 * k) :
  1 / S m + 1 / S p ≥ 2 / S k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l658_65839


namespace NUMINAMATH_CALUDE_correct_number_of_choices_l658_65864

/-- Represents the number of junior boys or girls -/
def num_juniors : ℕ := 7

/-- Represents the number of senior boys or girls -/
def num_seniors : ℕ := 8

/-- Represents the number of genders (boys and girls) -/
def num_genders : ℕ := 2

/-- Calculates the number of ways to choose a president and vice-president -/
def ways_to_choose_leaders : ℕ :=
  num_genders * (num_juniors * num_seniors + num_seniors * num_juniors)

/-- Theorem stating that the number of ways to choose leaders is 224 -/
theorem correct_number_of_choices : ways_to_choose_leaders = 224 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_choices_l658_65864


namespace NUMINAMATH_CALUDE_solve_for_a_l658_65857

-- Define the operation *
def star_op (a b : ℚ) : ℚ := 2*a - b^2

-- Theorem statement
theorem solve_for_a :
  ∀ a : ℚ, star_op a 7 = -20 → a = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l658_65857


namespace NUMINAMATH_CALUDE_certain_number_problem_l658_65878

theorem certain_number_problem (x : ℚ) : 
  (((x + 5) * 2) / 5) - 5 = 44 / 2 → x = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l658_65878


namespace NUMINAMATH_CALUDE_vector_parallel_value_l658_65874

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_parallel_value (x : ℝ) :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, x - 1)
  parallel a b → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_value_l658_65874


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l658_65801

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 200) → (∀ m : ℕ, m > n → m * (m + 1) ≥ 200) → n + (n + 1) = 27 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l658_65801


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l658_65859

theorem simplify_radical_expression :
  Real.sqrt 18 - Real.sqrt 50 + 3 * Real.sqrt (1/2) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l658_65859


namespace NUMINAMATH_CALUDE_mateo_deducted_salary_l658_65856

/-- Calculates the deducted salary for a worker given their weekly salary and number of absent days. -/
def deducted_salary (weekly_salary : ℚ) (absent_days : ℕ) : ℚ :=
  weekly_salary - (weekly_salary / 5 * absent_days)

/-- Proves that Mateo's deducted salary is correct given his weekly salary and absent days. -/
theorem mateo_deducted_salary :
  deducted_salary 791 4 = 158.2 := by
  sorry

end NUMINAMATH_CALUDE_mateo_deducted_salary_l658_65856


namespace NUMINAMATH_CALUDE_count_integer_pairs_l658_65886

theorem count_integer_pairs : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2 + p.2 = p.1 * p.2 + 1) ∧ 
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l658_65886


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l658_65868

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - x - m = 0

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, quadratic_equation x m

-- Theorem statement
theorem quadratic_real_roots_range (m : ℝ) : has_real_roots m → m ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l658_65868


namespace NUMINAMATH_CALUDE_multiply_b_is_eight_l658_65879

theorem multiply_b_is_eight (a b x : ℝ) 
  (h1 : 7 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 8) / (b / 7) = 1) : 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiply_b_is_eight_l658_65879


namespace NUMINAMATH_CALUDE_school_arrival_time_l658_65898

/-- Represents the problem of calculating how late a boy arrived at school. -/
theorem school_arrival_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (early_time : ℝ) : 
  distance = 2.5 ∧ 
  speed_day1 = 5 ∧ 
  speed_day2 = 10 ∧ 
  early_time = 10/60 →
  (distance / speed_day1) * 60 - ((distance / speed_day2) * 60 + early_time * 60) = 5 := by
  sorry

#check school_arrival_time

end NUMINAMATH_CALUDE_school_arrival_time_l658_65898


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l658_65815

theorem triangle_angle_relation (A B C C₁ C₂ : ℝ) : 
  B = 2 * A →
  C + A + B = Real.pi →
  C₁ + A = Real.pi / 2 →
  C₂ + B = Real.pi / 2 →
  C = C₁ + C₂ →
  C₁ - C₂ = A :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l658_65815


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l658_65885

theorem power_equality_implies_exponent (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l658_65885


namespace NUMINAMATH_CALUDE_inverse_inequality_l658_65890

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l658_65890


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l658_65812

/-- The solution set of the inequality (m-1)x^2 + (m-1)x + 2 > 0 is ℝ -/
def solution_set_is_real (m : ℝ) : Prop :=
  ∀ x, (m - 1) * x^2 + (m - 1) * x + 2 > 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 9

theorem inequality_solution_implies_m_range :
  ∀ m : ℝ, solution_set_is_real m → m_range m :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l658_65812


namespace NUMINAMATH_CALUDE_original_raw_silk_amount_l658_65819

/-- Given information about silk drying process, prove the original amount of raw silk. -/
theorem original_raw_silk_amount 
  (initial_wet : ℚ) 
  (water_loss : ℚ) 
  (final_dry : ℚ) 
  (h1 : initial_wet = 30) 
  (h2 : water_loss = 3) 
  (h3 : final_dry = 12) : 
  (initial_wet * final_dry) / (initial_wet - water_loss) = 40 / 3 := by
  sorry

#check original_raw_silk_amount

end NUMINAMATH_CALUDE_original_raw_silk_amount_l658_65819


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l658_65848

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a^2 + b^2 = 164) : 
  a * b = -50 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l658_65848


namespace NUMINAMATH_CALUDE_red_balloon_probability_l658_65847

/-- Calculates the probability of selecting a red balloon given the initial and additional counts of red and blue balloons. -/
theorem red_balloon_probability
  (initial_red : ℕ)
  (initial_blue : ℕ)
  (additional_red : ℕ)
  (additional_blue : ℕ)
  (h1 : initial_red = 2)
  (h2 : initial_blue = 4)
  (h3 : additional_red = 2)
  (h4 : additional_blue = 2) :
  (initial_red + additional_red : ℚ) / ((initial_red + additional_red + initial_blue + additional_blue) : ℚ) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_red_balloon_probability_l658_65847


namespace NUMINAMATH_CALUDE_coin_flip_probability_l658_65825

theorem coin_flip_probability : 
  let n : ℕ := 12  -- Total number of coins
  let k : ℕ := 3   -- Maximum number of heads we're interested in
  let favorable_outcomes : ℕ := (Finset.range (k + 1)).sum (λ i => Nat.choose n i)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 299 / 4096 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l658_65825


namespace NUMINAMATH_CALUDE_project_work_difference_l658_65866

/-- Represents the work hours of three people on a project -/
structure ProjectWork where
  person1 : ℝ
  person2 : ℝ
  person3 : ℝ

/-- The conditions of the project work -/
def validProjectWork (work : ProjectWork) : Prop :=
  work.person1 > 0 ∧ work.person2 > 0 ∧ work.person3 > 0 ∧
  work.person2 = 2 * work.person1 ∧
  work.person3 = 3 * work.person1 ∧
  work.person1 + work.person2 + work.person3 = 120

theorem project_work_difference (work : ProjectWork) 
  (h : validProjectWork work) : 
  work.person3 - work.person1 = 40 := by
  sorry

#check project_work_difference

end NUMINAMATH_CALUDE_project_work_difference_l658_65866


namespace NUMINAMATH_CALUDE_function_value_plus_derivative_l658_65830

open Real

/-- Given a differentiable function f : ℝ → ℝ satisfying f x = 2 * x * f.deriv 1 + log x for all x > 0,
    prove that f 1 + f.deriv 1 = -3 -/
theorem function_value_plus_derivative (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x > 0, f x = 2 * x * (deriv f 1) + log x) :
  f 1 + deriv f 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_plus_derivative_l658_65830


namespace NUMINAMATH_CALUDE_power_sum_equality_l658_65875

theorem power_sum_equality : (-2)^2007 + (-2)^2008 = 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l658_65875


namespace NUMINAMATH_CALUDE_percentage_of_older_female_students_l658_65814

/-- Represents the percentage of female students who are 25 years old or older -/
def P : ℝ := 30

theorem percentage_of_older_female_students :
  let total_students : ℝ := 100
  let male_percentage : ℝ := 40
  let female_percentage : ℝ := 100 - male_percentage
  let older_male_percentage : ℝ := 40
  let younger_probability : ℝ := 0.66
  
  (male_percentage / 100 * (100 - older_male_percentage) / 100 +
   female_percentage / 100 * (100 - P) / 100) * total_students = younger_probability * total_students :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_older_female_students_l658_65814


namespace NUMINAMATH_CALUDE_quadratic_root_square_condition_l658_65835

theorem quadratic_root_square_condition (p q r : ℝ) (α β : ℝ) : 
  (p * α^2 + q * α + r = 0) →  -- α is a root of the quadratic equation
  (p * β^2 + q * β + r = 0) →  -- β is a root of the quadratic equation
  (β = α^2) →                  -- one root is the square of the other
  (p - 4*q ≥ 0) :=             -- the relationship between coefficients
by sorry

end NUMINAMATH_CALUDE_quadratic_root_square_condition_l658_65835


namespace NUMINAMATH_CALUDE_course_selection_theorem_l658_65895

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 + 
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l658_65895


namespace NUMINAMATH_CALUDE_no_solution_cubic_system_l658_65805

theorem no_solution_cubic_system (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬∃ x : ℝ, (x^3 - a*x^2 + b^3 = 0) ∧ (x^3 - b*x^2 + c^3 = 0) ∧ (x^3 - c*x^2 + a^3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cubic_system_l658_65805


namespace NUMINAMATH_CALUDE_integral_x_over_sqrt_5_minus_x_l658_65846

theorem integral_x_over_sqrt_5_minus_x (x : ℝ) :
  HasDerivAt (λ x => (2/3) * (5 - x)^(3/2) - 10 * (5 - x)^(1/2)) 
             (x / (5 - x)^(1/2)) 
             x :=
sorry

end NUMINAMATH_CALUDE_integral_x_over_sqrt_5_minus_x_l658_65846


namespace NUMINAMATH_CALUDE_cosine_odd_function_phi_l658_65861

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cosine_odd_function_phi (φ : ℝ) :
  is_odd_function (λ x => Real.cos (x + φ + π/3)) → φ = π/6 ∨ ∃ k : ℤ, φ = k * π + π/6 :=
by sorry

end NUMINAMATH_CALUDE_cosine_odd_function_phi_l658_65861


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l658_65821

/-- Given two employees X and Y with a total pay of 572, where Y is paid 260,
    prove that X's pay as a percentage of Y's pay is 120%. -/
theorem employee_pay_percentage (X Y : ℝ) : 
  Y = 260 → X + Y = 572 → (X / Y) * 100 = 120 := by sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l658_65821


namespace NUMINAMATH_CALUDE_unique_divisible_digit_l658_65851

def is_single_digit (n : ℕ) : Prop := n < 10

def number_with_A (A : ℕ) : ℕ := 653802 * 10 + A

theorem unique_divisible_digit :
  ∃! A : ℕ, is_single_digit A ∧
    (∀ d : ℕ, d ∈ [2, 3, 4, 6, 8, 9, 25] → (number_with_A A) % d = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l658_65851


namespace NUMINAMATH_CALUDE_banana_survey_l658_65816

theorem banana_survey (total_students : ℕ) (banana_percentage : ℚ) : 
  total_students = 100 →
  banana_percentage = 1/5 →
  (banana_percentage * total_students : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_banana_survey_l658_65816


namespace NUMINAMATH_CALUDE_john_tax_difference_l658_65880

/-- Represents the tax rates and incomes before and after the change -/
structure TaxData where
  old_rate : ℝ
  new_rate : ℝ
  old_income : ℝ
  new_income : ℝ

/-- Calculates the difference in tax payments given the tax data -/
def tax_difference (data : TaxData) : ℝ :=
  data.new_rate * data.new_income - data.old_rate * data.old_income

/-- The specific tax data for John's situation -/
def john_tax_data : TaxData :=
  { old_rate := 0.20
    new_rate := 0.30
    old_income := 1000000
    new_income := 1500000 }

/-- Theorem stating that the difference in John's tax payments is $250,000 -/
theorem john_tax_difference :
  tax_difference john_tax_data = 250000 := by
  sorry

end NUMINAMATH_CALUDE_john_tax_difference_l658_65880


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l658_65871

theorem quadratic_root_zero (a : ℝ) :
  (∃ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l658_65871


namespace NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l658_65870

/-- A polynomial of degree 5 with leading coefficient 1 -/
def Polynomial5 : Type := ℝ → ℝ

/-- The difference of two polynomials of degree 5 -/
def PolynomialDifference (p q : Polynomial5) : ℝ → ℝ := fun x => p x - q x

theorem max_intersections_fifth_degree_polynomials (p q : Polynomial5) 
  (h_diff : p ≠ q) : 
  (∃ (S : Finset ℝ), ∀ x : ℝ, p x = q x ↔ x ∈ S) ∧ 
  (∀ (S : Finset ℝ), (∀ x : ℝ, p x = q x ↔ x ∈ S) → S.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l658_65870


namespace NUMINAMATH_CALUDE_equation_solutions_l658_65892

theorem equation_solutions :
  ∀ x : ℝ, x ≥ 4 →
    ((x / (2 * Real.sqrt 2) + 5 * Real.sqrt 2 / 2) * Real.sqrt (x^3 - 64*x + 200) = x^2 + 6*x - 40) ↔
    (x = 6 ∨ x = Real.sqrt 13 + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l658_65892


namespace NUMINAMATH_CALUDE_probability_of_purple_marble_l658_65818

theorem probability_of_purple_marble (blue_prob green_prob purple_prob : ℝ) :
  blue_prob = 0.25 →
  green_prob = 0.35 →
  blue_prob + green_prob + purple_prob = 1 →
  purple_prob = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_purple_marble_l658_65818


namespace NUMINAMATH_CALUDE_unique_base_nine_l658_65803

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem unique_base_nine :
  ∃! b : Nat, b > 1 ∧ 
    to_decimal [1, 5, 2] b + to_decimal [1, 4, 3] b = to_decimal [3, 0, 5] b :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_nine_l658_65803


namespace NUMINAMATH_CALUDE_successive_discounts_l658_65813

theorem successive_discounts (original_price : ℝ) (first_discount second_discount : ℝ) 
  (h1 : first_discount = 0.25)
  (h2 : second_discount = 0.10) :
  (original_price * (1 - first_discount) * (1 - second_discount)) / original_price = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_l658_65813


namespace NUMINAMATH_CALUDE_fourth_operation_result_l658_65896

def pattern_result (a b : ℕ) : ℕ := a * b + a * (b - a)

theorem fourth_operation_result : pattern_result 5 8 = 55 := by
  sorry

end NUMINAMATH_CALUDE_fourth_operation_result_l658_65896


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l658_65881

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem f_satisfies_conditions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x → x < 2 → f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l658_65881


namespace NUMINAMATH_CALUDE_probability_consecutive_points_l658_65802

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of points in the quadrilateral -/
def q : ℕ := 4

/-- The number of points in the triangle -/
def t : ℕ := 3

/-- The number of ways to select 3 consecutive points from n points on a circle -/
def consecutive_selections (n : ℕ) : ℕ := n

/-- The total number of ways to select 3 points from n points -/
def total_selections (n : ℕ) : ℕ := n.choose 3

/-- The probability of selecting 3 consecutive points out of 7 points on a circle,
    given that 4 points have already been selected to form a quadrilateral -/
theorem probability_consecutive_points : 
  (consecutive_selections n : ℚ) / (total_selections n : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_consecutive_points_l658_65802


namespace NUMINAMATH_CALUDE_complex_fraction_power_l658_65806

theorem complex_fraction_power (i : ℂ) (h : i^2 = -1) :
  ((1 + i) / (1 - i))^2006 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l658_65806


namespace NUMINAMATH_CALUDE_quadratic_roots_angles_l658_65852

theorem quadratic_roots_angles (Az m n φ ψ : ℝ) (hAz : Az ≠ 0) :
  (∀ x, Az * x^2 - m * x + n = 0 ↔ x = Real.tan φ ∨ x = Real.tan ψ) →
  Real.tan (φ + ψ) = m / (1 - n) ∧ Real.tan (φ - ψ) = Real.sqrt (m^2 - 4*n) / (1 + n) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_angles_l658_65852


namespace NUMINAMATH_CALUDE_expected_sales_theorem_l658_65834

/-- Represents the number of vehicles sold for each type -/
structure VehicleSales where
  sports_cars : ℕ
  sedans : ℕ
  trucks : ℕ

/-- The ratio of vehicle sales -/
def sales_ratio : VehicleSales :=
  { sports_cars := 3
    sedans := 5
    trucks := 4 }

/-- The expected number of sports cars to be sold -/
def expected_sports_cars : ℕ := 36

/-- Calculates the expected sales based on the ratio and expected sports car sales -/
def calculate_expected_sales (ratio : VehicleSales) (sports_cars : ℕ) : VehicleSales :=
  { sports_cars := sports_cars
    sedans := (sports_cars * ratio.sedans) / ratio.sports_cars
    trucks := (sports_cars * ratio.trucks) / ratio.sports_cars }

theorem expected_sales_theorem :
  let expected_sales := calculate_expected_sales sales_ratio expected_sports_cars
  expected_sales.sedans = 60 ∧ expected_sales.trucks = 48 := by
  sorry

end NUMINAMATH_CALUDE_expected_sales_theorem_l658_65834


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l658_65862

-- Define geometric bodies
structure GeometricBody where
  height : ℝ
  crossSectionalArea : ℝ → ℝ
  volume : ℝ

-- Define the Gougu Principle
def gougu_principle (A B : GeometricBody) : Prop :=
  A.height = B.height →
  (∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) →
  A.volume = B.volume

-- Define the relationship between p and q
theorem p_necessary_not_sufficient_for_q (A B : GeometricBody) :
  (∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) →
  A.volume = B.volume ∧
  ¬(A.volume = B.volume →
    ∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) :=
by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l658_65862


namespace NUMINAMATH_CALUDE_range_of_m_l658_65838

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m^2 + 3*m - 3

/-- Proposition p: The minimum value of f(x) is less than 0 -/
def p (m : ℝ) : Prop := ∃ x, f m x < 0

/-- Proposition q: The equation represents an ellipse with foci on the x-axis -/
def q (m : ℝ) : Prop := 5*m - 1 > 0 ∧ m - 2 < 0 ∧ 5*m - 1 > -(m - 2)

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) (h1 : ¬(p m ∨ q m)) (h2 : ¬(p m ∧ q m)) : 
  m ≤ -4 ∨ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l658_65838


namespace NUMINAMATH_CALUDE_abs_z_squared_l658_65804

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z + |z| = 3 + 12i
def condition (z : ℂ) : Prop := z + Complex.abs z = 3 + 12 * Complex.I

-- Theorem statement
theorem abs_z_squared (h : condition z) : Complex.abs z ^ 2 = 650.25 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_squared_l658_65804


namespace NUMINAMATH_CALUDE_rationalize_denominator_l658_65828

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l658_65828


namespace NUMINAMATH_CALUDE_unique_point_in_S_l658_65836

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3)*p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_point_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end NUMINAMATH_CALUDE_unique_point_in_S_l658_65836


namespace NUMINAMATH_CALUDE_fraction_contradiction_l658_65894

theorem fraction_contradiction : ¬∃ (x : ℚ), (8 * x = 4) ∧ ((1/4) * 16 = 10 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_contradiction_l658_65894


namespace NUMINAMATH_CALUDE_sin_equality_in_range_l658_65844

theorem sin_equality_in_range (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → Real.sin (n * π / 180) = Real.sin (750 * π / 180) → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_in_range_l658_65844


namespace NUMINAMATH_CALUDE_M_necessary_not_sufficient_for_N_l658_65833

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem M_necessary_not_sufficient_for_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_M_necessary_not_sufficient_for_N_l658_65833


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l658_65832

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l658_65832


namespace NUMINAMATH_CALUDE_triangle_arithmetic_geometric_is_equilateral_l658_65850

/-- A triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The property that the angles form an arithmetic sequence -/
def Triangle.angles_arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : ℝ, (t.B - t.A = d ∧ t.C - t.B = d) ∨ (t.A - t.B = d ∧ t.B - t.C = d) ∨ (t.C - t.A = d ∧ t.A - t.B = d)

/-- The property that the sides form a geometric sequence -/
def Triangle.sides_geometric_sequence (t : Triangle) : Prop :=
  (t.b^2 = t.a * t.c) ∨ (t.a^2 = t.b * t.c) ∨ (t.c^2 = t.a * t.b)

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The main theorem -/
theorem triangle_arithmetic_geometric_is_equilateral (t : Triangle) :
  t.angles_arithmetic_sequence → t.sides_geometric_sequence → t.is_equilateral :=
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_geometric_is_equilateral_l658_65850


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l658_65899

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℚ
  hari_initial : ℚ
  total_months : ℕ
  hari_join_month : ℕ

/-- Calculates the effective contribution of a partner -/
def effective_contribution (initial : ℚ) (months : ℕ) : ℚ :=
  initial * months

/-- Theorem stating the profit-sharing ratio between Praveen and Hari -/
theorem profit_sharing_ratio (p : Partnership) 
  (h1 : p.praveen_initial = 3780)
  (h2 : p.hari_initial = 9720)
  (h3 : p.total_months = 12)
  (h4 : p.hari_join_month = 5) :
  (effective_contribution p.praveen_initial p.total_months) / 
  (effective_contribution p.hari_initial (p.total_months - p.hari_join_month)) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_profit_sharing_ratio_l658_65899


namespace NUMINAMATH_CALUDE_product_evaluation_l658_65869

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l658_65869


namespace NUMINAMATH_CALUDE_ali_boxes_calculation_l658_65867

/-- The number of boxes Ali used for each of his circles -/
def ali_boxes_per_circle : ℕ := 14

/-- The total number of boxes -/
def total_boxes : ℕ := 80

/-- The number of circles Ali made -/
def ali_circles : ℕ := 5

/-- The number of boxes Ernie used for his circle -/
def ernie_boxes : ℕ := 10

theorem ali_boxes_calculation :
  ali_boxes_per_circle * ali_circles + ernie_boxes = total_boxes :=
by sorry

end NUMINAMATH_CALUDE_ali_boxes_calculation_l658_65867


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l658_65842

theorem triangle_cosine_inequality (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  (Real.cos A / Real.cos B)^2 + (Real.cos B / Real.cos C)^2 + (Real.cos C / Real.cos A)^2 
  ≥ 4 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l658_65842


namespace NUMINAMATH_CALUDE_max_min_sum_difference_l658_65827

def three_digit_integer (a b c : ℕ) : Prop :=
  100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c < 1000

def all_different (a b c d e f g h i : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem max_min_sum_difference :
  ∀ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℕ),
  three_digit_integer a₁ b₁ c₁ →
  three_digit_integer a₂ b₂ c₂ →
  three_digit_integer a₃ b₃ c₃ →
  all_different a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ →
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ : ℕ),
    three_digit_integer x₁ y₁ z₁ →
    three_digit_integer x₂ y₂ z₂ →
    three_digit_integer x₃ y₃ z₃ →
    all_different x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ →
    (a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃) ≥
    (x₁ * 100 + y₁ * 10 + z₁) + (x₂ * 100 + y₂ * 10 + z₂) + (x₃ * 100 + y₃ * 10 + z₃)) →
  (∀ (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℕ),
    three_digit_integer p₁ q₁ r₁ →
    three_digit_integer p₂ q₂ r₂ →
    three_digit_integer p₃ q₃ r₃ →
    all_different p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ →
    (p₁ * 100 + q₁ * 10 + r₁) + (p₂ * 100 + q₂ * 10 + r₂) + (p₃ * 100 + q₃ * 10 + r₃) ≥
    (a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃)) →
  ((a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃)) -
  ((p₁ * 100 + q₁ * 10 + r₁) + (p₂ * 100 + q₂ * 10 + r₂) + (p₃ * 100 + q₃ * 10 + r₃)) = 1845 :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_difference_l658_65827


namespace NUMINAMATH_CALUDE_probability_13_11_l658_65897

/-- Represents a table tennis player -/
inductive Player : Type
| MaLong : Player
| FanZhendong : Player

/-- The probability of a player scoring when serving -/
def scoreProbability (server : Player) : ℚ :=
  match server with
  | Player.MaLong => 2/3
  | Player.FanZhendong => 1/2

/-- The probability of a player scoring when receiving -/
def receiveProbability (receiver : Player) : ℚ :=
  match receiver with
  | Player.MaLong => 1/2
  | Player.FanZhendong => 1/3

/-- Theorem stating the probability of reaching 13:11 score -/
theorem probability_13_11 :
  let initialServer := Player.MaLong
  let prob13_11 := (scoreProbability initialServer * receiveProbability Player.FanZhendong * scoreProbability initialServer * receiveProbability Player.FanZhendong) +
                   (receiveProbability Player.FanZhendong * scoreProbability Player.FanZhendong * scoreProbability initialServer * receiveProbability Player.FanZhendong) +
                   (receiveProbability Player.FanZhendong * scoreProbability Player.FanZhendong * receiveProbability initialServer * scoreProbability Player.FanZhendong) +
                   (scoreProbability initialServer * receiveProbability Player.FanZhendong * receiveProbability initialServer * scoreProbability Player.FanZhendong)
  prob13_11 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_13_11_l658_65897


namespace NUMINAMATH_CALUDE_will_had_28_bottles_l658_65884

/-- The number of bottles Will had -/
def bottles : ℕ := sorry

/-- The number of days the bottles would last -/
def days : ℕ := 4

/-- The number of bottles Will would drink per day -/
def bottles_per_day : ℕ := 7

/-- Theorem stating that Will had 28 bottles -/
theorem will_had_28_bottles : bottles = 28 := by
  sorry

end NUMINAMATH_CALUDE_will_had_28_bottles_l658_65884


namespace NUMINAMATH_CALUDE_max_difference_theorem_l658_65808

/-- The maximum difference between the sum of ball numbers for two people --/
def maxDifference : ℕ := 9644

/-- The total number of balls --/
def totalBalls : ℕ := 200

/-- The starting number of the balls --/
def startNumber : ℕ := 101

/-- The ending number of the balls --/
def endNumber : ℕ := 300

/-- The number of balls each person takes --/
def ballsPerPerson : ℕ := 100

/-- The ball number that person A takes --/
def ballA : ℕ := 102

/-- The ball number that person B takes --/
def ballB : ℕ := 280

theorem max_difference_theorem :
  ∀ (sumA sumB : ℕ),
  sumA ≤ (startNumber + endNumber) * ballsPerPerson / 2 - (ballB - ballA) →
  sumB ≥ (startNumber + endNumber - totalBalls + 1) * ballsPerPerson / 2 + (ballB - ballA) →
  sumA - sumB ≤ maxDifference :=
sorry

end NUMINAMATH_CALUDE_max_difference_theorem_l658_65808


namespace NUMINAMATH_CALUDE_committee_selection_l658_65837

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection :
  choose 20 3 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l658_65837


namespace NUMINAMATH_CALUDE_hockey_games_per_month_l658_65853

/-- Proves that the number of hockey games played each month is 13,
    given that there are 182 hockey games in a 14-month season. -/
theorem hockey_games_per_month :
  let total_games : ℕ := 182
  let season_months : ℕ := 14
  let games_per_month : ℕ := total_games / season_months
  games_per_month = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_hockey_games_per_month_l658_65853


namespace NUMINAMATH_CALUDE_complex_i_minus_one_in_third_quadrant_l658_65849

theorem complex_i_minus_one_in_third_quadrant :
  let z : ℂ := Complex.I * (Complex.I - 1)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_i_minus_one_in_third_quadrant_l658_65849


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l658_65883

theorem choose_four_from_seven : 
  Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l658_65883


namespace NUMINAMATH_CALUDE_range_of_a_l658_65891

/-- The range of a given the conditions in the problem -/
theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 4*a*x + 3*a^2 < 0 → |x - 3| > 1) ∧ 
  (∃ x, |x - 3| > 1 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) ∧ 
  (a > 0) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry


end NUMINAMATH_CALUDE_range_of_a_l658_65891


namespace NUMINAMATH_CALUDE_fraction_value_l658_65893

/-- Given a, b, c, d are real numbers satisfying certain relationships,
    prove that (a * c) / (b * d) = 15 -/
theorem fraction_value (a b c d : ℝ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * c) 
    (h3 : c = 5 * d) 
    (h4 : b ≠ 0) 
    (h5 : d ≠ 0) : 
  (a * c) / (b * d) = 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l658_65893


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l658_65829

theorem chess_tournament_participants (x : ℕ) (y : ℕ) : 
  (2 * y + 8 = (x + 2) * (x + 1) / 2) →
  (x * y + 8 = (x + 2) * (x + 1) / 2) →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l658_65829


namespace NUMINAMATH_CALUDE_cohen_bird_count_l658_65858

/-- The total number of fish-eater birds Cohen saw over three days -/
def total_birds (day1 : ℕ) (day2_factor : ℕ) (day3_reduction : ℕ) : ℕ :=
  day1 + day1 * day2_factor + (day1 * day2_factor - day3_reduction)

/-- Theorem stating the total number of fish-eater birds Cohen saw over three days -/
theorem cohen_bird_count :
  total_birds 300 2 200 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_cohen_bird_count_l658_65858


namespace NUMINAMATH_CALUDE_trigonometric_equality_l658_65810

theorem trigonometric_equality (a b c : ℝ) (α β : ℝ) 
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (h3 : ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  Real.sin (α - β) ^ 2 = c^2 / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l658_65810


namespace NUMINAMATH_CALUDE_trapezoid_area_l658_65854

/-- The area of a trapezoid given the areas of triangles formed by its diagonals -/
theorem trapezoid_area (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) : 
  ∃ (A : ℝ), A = (Real.sqrt S₁ + Real.sqrt S₂)^2 ∧ A > 0 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l658_65854


namespace NUMINAMATH_CALUDE_sandy_savings_l658_65877

theorem sandy_savings (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (h1 : last_year_savings_rate > 0)
  (h2 : last_year_savings_rate < 1)
  (h3 : (1.1 * last_year_salary) * 0.09 = 1.65 * (last_year_salary * last_year_savings_rate)) :
  last_year_savings_rate = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_savings_l658_65877


namespace NUMINAMATH_CALUDE_comparison_of_powers_l658_65855

theorem comparison_of_powers : 
  let a : ℝ := Real.rpow 0.6 0.6
  let b : ℝ := Real.rpow 0.6 1.2
  let c : ℝ := Real.rpow 1.2 0.6
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l658_65855


namespace NUMINAMATH_CALUDE_magnificent_class_size_l658_65826

theorem magnificent_class_size :
  ∀ (girls boys chocolates_given : ℕ),
    girls + boys = 33 →
    boys = girls + 3 →
    girls * girls + boys * boys = chocolates_given →
    chocolates_given = 540 - 12 →
    True :=
by
  sorry

end NUMINAMATH_CALUDE_magnificent_class_size_l658_65826


namespace NUMINAMATH_CALUDE_emilys_dogs_l658_65820

theorem emilys_dogs (food_per_dog_per_day : ℕ) (vacation_days : ℕ) (total_food_kg : ℕ) :
  food_per_dog_per_day = 250 →
  vacation_days = 14 →
  total_food_kg = 14 →
  (total_food_kg * 1000) / (food_per_dog_per_day * vacation_days) = 4 :=
by sorry

end NUMINAMATH_CALUDE_emilys_dogs_l658_65820


namespace NUMINAMATH_CALUDE_common_chord_equation_l658_65872

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- The equation of the common chord -/
def common_chord (x y : ℝ) : Prop := x - 2*y + 5 = 0

/-- Theorem stating that the common chord of the two circles is x - 2y + 5 = 0 -/
theorem common_chord_equation :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l658_65872


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l658_65845

/-- The ring toss game made this amount in the first 44 days -/
def first_period_earnings : ℕ := 382

/-- The ring toss game made this amount in the remaining 10 days -/
def second_period_earnings : ℕ := 374

/-- The total earnings of the ring toss game -/
def total_earnings : ℕ := first_period_earnings + second_period_earnings

theorem ring_toss_earnings : total_earnings = 756 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l658_65845


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l658_65863

/-- Represents the fishing schedule in the coastal village --/
structure FishingSchedule where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow given the fishing schedule --/
def fishersTomorrow (schedule : FishingSchedule) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow --/
theorem fifteen_fishers_tomorrow :
  let schedule := FishingSchedule.mk 7 8 3 12 10
  fishersTomorrow schedule = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l658_65863


namespace NUMINAMATH_CALUDE_cat_adoption_rate_is_25_percent_l658_65824

def initial_dogs : ℕ := 30
def initial_cats : ℕ := 28
def initial_lizards : ℕ := 20
def dog_adoption_rate : ℚ := 1/2
def lizard_adoption_rate : ℚ := 1/5
def new_pets : ℕ := 13
def total_pets_after_month : ℕ := 65

theorem cat_adoption_rate_is_25_percent :
  let dogs_adopted := (initial_dogs : ℚ) * dog_adoption_rate
  let lizards_adopted := (initial_lizards : ℚ) * lizard_adoption_rate
  let remaining_dogs := initial_dogs - dogs_adopted.floor
  let remaining_lizards := initial_lizards - lizards_adopted.floor
  let remaining_pets := remaining_dogs + remaining_lizards + new_pets
  let remaining_cats := total_pets_after_month - remaining_pets
  let cats_adopted := initial_cats - remaining_cats
  (cats_adopted : ℚ) / initial_cats = 1/4 := by
    sorry

end NUMINAMATH_CALUDE_cat_adoption_rate_is_25_percent_l658_65824


namespace NUMINAMATH_CALUDE_roots_of_equation_l658_65840

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * (x - 3) * (x + 2)

theorem roots_of_equation : 
  {x : ℝ | equation x = 0} = {-2, 2, 3} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l658_65840


namespace NUMINAMATH_CALUDE_triangle_theorem_l658_65888

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.tan t.C / Real.tan t.B = -t.c / (2 * t.a + t.c))
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l658_65888


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l658_65822

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line) : Prop := l₁.slope = l₂.slope

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l₁ : Line := ⟨1, a/2⟩
  let l₂ : Line := ⟨a^2 - 3, 1⟩
  parallel l₁ l₂ → a = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l658_65822


namespace NUMINAMATH_CALUDE_exponent_division_l658_65800

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l658_65800


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l658_65882

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

-- Theorem statement
theorem circle_radius_is_sqrt_2 :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l658_65882


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l658_65887

theorem polar_to_rectangular_conversion :
  let r : ℝ := 7
  let θ : ℝ := Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l658_65887


namespace NUMINAMATH_CALUDE_no_half_rectangle_exists_l658_65811

theorem no_half_rectangle_exists (a b : ℝ) (h : 0 < a ∧ a < b) :
  ¬ ∃ (x y : ℝ), 
    x < a / 2 ∧ 
    y < a / 2 ∧ 
    2 * (x + y) = a + b ∧ 
    x * y = a * b / 2 :=
by sorry

end NUMINAMATH_CALUDE_no_half_rectangle_exists_l658_65811


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l658_65865

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Calculates the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1)

theorem arithmetic_sequence_problem :
  ∃ (row col1 col2 : ArithmeticSequence),
    row.first = 15 ∧
    row.nthTerm 4 = 2 ∧
    col1.nthTerm 2 = 14 ∧
    col1.nthTerm 3 = 10 ∧
    col2.nthTerm 5 = -21 ∧
    col2.first = -13.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l658_65865


namespace NUMINAMATH_CALUDE_systematic_sampling_first_group_l658_65809

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalStudents : Nat
  numGroups : Nat
  sampleSize : Nat
  groupSize : Nat
  sixteenthGroupDraw : Nat

/-- Theorem for systematic sampling -/
theorem systematic_sampling_first_group
  (setup : SystematicSampling)
  (h1 : setup.totalStudents = 160)
  (h2 : setup.numGroups = 20)
  (h3 : setup.sampleSize = 20)
  (h4 : setup.groupSize = setup.totalStudents / setup.numGroups)
  (h5 : setup.sixteenthGroupDraw = 126) :
  ∃ (firstGroupDraw : Nat), firstGroupDraw = 6 ∧
    setup.sixteenthGroupDraw = (16 - 1) * setup.groupSize + firstGroupDraw :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_group_l658_65809


namespace NUMINAMATH_CALUDE_unique_solution_l658_65823

/-- 
Given two positive integers x and y, prove that if they satisfy the equations
x^y + 4 = y^x and 3x^y = y^x + 10, then x = 7 and y = 1.
-/
theorem unique_solution (x y : ℕ+) 
  (h1 : x^(y:ℕ) + 4 = y^(x:ℕ)) 
  (h2 : 3 * x^(y:ℕ) = y^(x:ℕ) + 10) : 
  x = 7 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l658_65823


namespace NUMINAMATH_CALUDE_straight_line_angle_value_l658_65817

/-- The sum of angles in a straight line is 180 degrees -/
def straight_line_angle_sum : ℝ := 180

/-- The angles along the straight line ABC -/
def angle1 (x : ℝ) : ℝ := x
def angle2 : ℝ := 21
def angle3 : ℝ := 21
def angle4 (x : ℝ) : ℝ := 2 * x
def angle5 : ℝ := 57

/-- Theorem: Given a straight line ABC with angles x°, 21°, 21°, 2x°, and 57°, the value of x is 27° -/
theorem straight_line_angle_value :
  ∀ x : ℝ, 
  angle1 x + angle2 + angle3 + angle4 x + angle5 = straight_line_angle_sum → 
  x = 27 := by
sorry


end NUMINAMATH_CALUDE_straight_line_angle_value_l658_65817
