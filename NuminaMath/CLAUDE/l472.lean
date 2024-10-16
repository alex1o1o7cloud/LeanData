import Mathlib

namespace NUMINAMATH_CALUDE_height_after_changes_l472_47214

-- Define the initial height in centimeters
def initial_height : ℝ := 167.64

-- Define the growth and decrease percentages
def first_growth : ℝ := 0.15
def second_growth : ℝ := 0.07
def decrease : ℝ := 0.04

-- Define the final height calculation
def final_height : ℝ :=
  initial_height * (1 + first_growth) * (1 + second_growth) * (1 - decrease)

-- State the theorem
theorem height_after_changes :
  ∃ ε > 0, |final_height - 198.03| < ε :=
sorry

end NUMINAMATH_CALUDE_height_after_changes_l472_47214


namespace NUMINAMATH_CALUDE_third_part_time_l472_47224

/-- Represents the time taken for each part of the assignment -/
def timeTaken (k : ℕ) : ℕ := 25 * k

/-- The total time available for the assignment in minutes -/
def totalTimeAvailable : ℕ := 150

/-- The time taken for the first break -/
def firstBreak : ℕ := 10

/-- The time taken for the second break -/
def secondBreak : ℕ := 15

/-- Theorem stating that the time taken for the third part is 50 minutes -/
theorem third_part_time : 
  totalTimeAvailable - (timeTaken 1 + firstBreak + timeTaken 2 + secondBreak) = 50 := by
  sorry


end NUMINAMATH_CALUDE_third_part_time_l472_47224


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l472_47216

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Given a two-digit number, insert_zero inserts a 0 between its digits. -/
def insert_zero (n : ℕ) : ℕ := (n / 10) * 100 + (n % 10)

/-- The set of numbers that satisfy the condition in the problem. -/
def solution_set : Set ℕ := {80, 81, 82, 83, 84, 85, 86, 87, 88, 89}

/-- The main theorem that proves the solution to the problem. -/
theorem two_digit_number_theorem (n : ℕ) : 
  TwoDigitNumber n → (insert_zero n = n + 720) → n ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l472_47216


namespace NUMINAMATH_CALUDE_M_congruent_to_1_mod_47_l472_47295

def M : ℕ := sorry -- Definition of M as the 81-digit number

theorem M_congruent_to_1_mod_47 :
  M % 47 = 1 := by sorry

end NUMINAMATH_CALUDE_M_congruent_to_1_mod_47_l472_47295


namespace NUMINAMATH_CALUDE_boat_reachable_area_l472_47259

/-- Represents the speed of the boat in miles per hour -/
structure BoatSpeed where
  river : ℝ
  land : ℝ

/-- Calculates the area reachable by the boat given its speed and time limit -/
def reachable_area (speed : BoatSpeed) (time_limit : ℝ) : ℝ :=
  sorry

theorem boat_reachable_area :
  let speed : BoatSpeed := { river := 40, land := 10 }
  let time_limit : ℝ := 12 / 60 -- 12 minutes in hours
  reachable_area speed time_limit = 232 * π / 6 := by
  sorry

#eval (232 + 6 : Nat)

end NUMINAMATH_CALUDE_boat_reachable_area_l472_47259


namespace NUMINAMATH_CALUDE_min_sum_areas_two_triangles_l472_47235

/-- The minimum sum of areas of two equilateral triangles formed from a 12cm wire -/
theorem min_sum_areas_two_triangles : 
  ∃ (f : ℝ → ℝ), 
    (∀ x, 0 ≤ x ∧ x ≤ 12 → 
      f x = (Real.sqrt 3 / 36) * (x^2 + (12 - x)^2)) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 12 → f x ≥ 2 * Real.sqrt 3) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 12 ∧ f x = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_areas_two_triangles_l472_47235


namespace NUMINAMATH_CALUDE_complex_inequality_l472_47234

theorem complex_inequality (a b c d : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hab : 2 * Complex.abs (a - b) ≤ Complex.abs b)
  (hbc : 2 * Complex.abs (b - c) ≤ Complex.abs c)
  (hcd : 2 * Complex.abs (c - d) ≤ Complex.abs d)
  (hda : 2 * Complex.abs (d - a) ≤ Complex.abs a) :
  (7 : ℝ) / 2 < Complex.abs (b / a + c / b + d / c + a / d) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l472_47234


namespace NUMINAMATH_CALUDE_power_mod_eleven_l472_47230

theorem power_mod_eleven : 7^308 ≡ 9 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l472_47230


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l472_47283

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Main theorem: If a line is perpendicular to a plane and parallel to another line,
    then the other line is also perpendicular to the plane -/
theorem perpendicular_parallel_transitive
  (m n : Line3D) (α : Plane3D)
  (h1 : perpendicular m α)
  (h2 : parallel m n) :
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l472_47283


namespace NUMINAMATH_CALUDE_expression_evaluation_l472_47282

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/2
  (x * (x - 4*y) + (2*x + y) * (2*x - y) - (2*x - y)^2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l472_47282


namespace NUMINAMATH_CALUDE_largest_operator_result_l472_47252

theorem largest_operator_result : 
  let expr := (5 * Real.sqrt 2 - Real.sqrt 2)
  (expr * Real.sqrt 2 > expr + Real.sqrt 2) ∧
  (expr * Real.sqrt 2 > expr - Real.sqrt 2) ∧
  (expr * Real.sqrt 2 > expr / Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_operator_result_l472_47252


namespace NUMINAMATH_CALUDE_drum_oil_capacity_l472_47221

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) : 
  C > 0 → -- Capacity of Drum X is positive
  Y ≥ 0 → -- Initial amount of oil in Drum Y is non-negative
  Y + (1/2 * C) = 0.65 * (2 * C) → -- After pouring, Drum Y is filled to 0.65 capacity
  Y = 0.8 * (2 * C) -- Initial fill level of Drum Y is 0.8 of its capacity
  := by sorry

end NUMINAMATH_CALUDE_drum_oil_capacity_l472_47221


namespace NUMINAMATH_CALUDE_sandwich_cost_is_two_l472_47269

/-- Calculates the cost per sandwich given the prices and discounts for ingredients -/
def cost_per_sandwich (bread_price : ℚ) (meat_price : ℚ) (cheese_price : ℚ) 
  (meat_discount : ℚ) (cheese_discount : ℚ) (num_sandwiches : ℕ) : ℚ :=
  let total_cost := bread_price + 2 * meat_price + 2 * cheese_price - meat_discount - cheese_discount
  total_cost / num_sandwiches

/-- Proves that the cost per sandwich is $2.00 given the specified conditions -/
theorem sandwich_cost_is_two :
  cost_per_sandwich 4 5 4 1 1 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_two_l472_47269


namespace NUMINAMATH_CALUDE_remaining_requests_after_seven_days_l472_47228

/-- The number of days -/
def days : ℕ := 7

/-- The number of requests received per day -/
def requests_per_day : ℕ := 8

/-- The number of requests completed per day -/
def completed_per_day : ℕ := 4

/-- The number of remaining requests after a given number of days -/
def remaining_requests (d : ℕ) : ℕ :=
  (requests_per_day - completed_per_day) * d + requests_per_day * d

theorem remaining_requests_after_seven_days :
  remaining_requests days = 84 := by
  sorry

end NUMINAMATH_CALUDE_remaining_requests_after_seven_days_l472_47228


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l472_47268

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_root_sum (a b c d : ℝ) :
  g a b c d (3*I) = 0 ∧ g a b c d (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l472_47268


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l472_47257

theorem sqrt_equation_solution (a : ℝ) (h : a ≥ -1/4) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (a + Real.sqrt (a + x)) = x ∧ x = (1 + Real.sqrt (1 + 4*a)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l472_47257


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l472_47276

theorem largest_digit_divisible_by_six :
  ∀ N : ℕ, N ≤ 9 → (4517 * 10 + N) % 6 = 0 → N ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l472_47276


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l472_47200

/-- The height of a cylinder resulting from a melted sphere, given constant volume --/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3)
  (h_cylinder : r_cylinder = 10) :
  (4 / 3 * π * r_sphere ^ 3) / (π * r_cylinder ^ 2) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_melted_ice_cream_height_l472_47200


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l472_47265

/-- The number M as defined in the problem -/
def M : ℕ := 36 * 36 * 95 * 400

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of M -/
theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l472_47265


namespace NUMINAMATH_CALUDE_area_PTW_approx_34_l472_47292

-- Define the areas of triangles as functions of x
def area_PUW (x : ℝ) : ℝ := 4*x + 4
def area_SUW (x : ℝ) : ℝ := 2*x + 20
def area_SVW (x : ℝ) : ℝ := 5*x + 20
def area_SVR (x : ℝ) : ℝ := 5*x + 11
def area_QVR (x : ℝ) : ℝ := 8*x + 32
def area_QVW (x : ℝ) : ℝ := 8*x + 50

-- Define the equation for solving x
def solve_for_x (x : ℝ) : Prop :=
  (area_QVW x) / (area_SVW x) = (area_QVR x) / (area_SVR x)

-- Define the area of triangle PTW
noncomputable def area_PTW (x : ℝ) : ℝ := 
  sorry  -- The exact formula is not provided in the problem

-- State the theorem
theorem area_PTW_approx_34 :
  ∃ (x : ℝ), solve_for_x x ∧ 
  (∀ (y : ℝ), abs (area_PTW x - 34) ≤ abs (area_PTW x - y) ∨ y = 34) :=
sorry

end NUMINAMATH_CALUDE_area_PTW_approx_34_l472_47292


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l472_47247

theorem max_value_of_exponential_difference (x : ℝ) :
  ∃ (max : ℝ), max = 1/4 ∧ ∀ (y : ℝ), 5^y - 25^y ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l472_47247


namespace NUMINAMATH_CALUDE_updated_mean_l472_47238

/-- Given 50 observations with an original mean of 200 and a decrement of 47 from each observation,
    the updated mean is 153. -/
theorem updated_mean (n : ℕ) (original_mean decrement : ℚ) (h1 : n = 50) (h2 : original_mean = 200) (h3 : decrement = 47) :
  let total_sum := n * original_mean
  let total_decrement := n * decrement
  let updated_sum := total_sum - total_decrement
  let updated_mean := updated_sum / n
  updated_mean = 153 := by
sorry

end NUMINAMATH_CALUDE_updated_mean_l472_47238


namespace NUMINAMATH_CALUDE_solve_for_y_l472_47212

theorem solve_for_y (x y : ℤ) (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l472_47212


namespace NUMINAMATH_CALUDE_company_income_analysis_l472_47240

structure Company where
  employees : ℕ
  max_income : ℕ
  avg_income : ℕ
  min_income : ℕ
  mid_50_low : ℕ
  mid_50_high : ℕ

def is_high_income (c : Company) (income : ℕ) : Prop :=
  income > c.avg_income

def is_sufficient_info (c : Company) : Prop :=
  c.mid_50_low > 0 ∧ c.mid_50_high > c.mid_50_low

def estimate_median (c : Company) : ℕ :=
  (c.mid_50_low + c.mid_50_high) / 2

theorem company_income_analysis (c : Company) 
  (h1 : c.employees = 50)
  (h2 : c.max_income = 1000000)
  (h3 : c.avg_income = 35000)
  (h4 : c.min_income = 5000)
  (h5 : c.mid_50_low = 10000)
  (h6 : c.mid_50_high = 30000) :
  ¬is_high_income c 25000 ∧
  ¬is_sufficient_info {employees := c.employees, max_income := c.max_income, avg_income := c.avg_income, min_income := c.min_income, mid_50_low := 0, mid_50_high := 0} ∧
  is_sufficient_info c ∧
  estimate_median c < c.avg_income := by
  sorry

#check company_income_analysis

end NUMINAMATH_CALUDE_company_income_analysis_l472_47240


namespace NUMINAMATH_CALUDE_min_value_quadratic_l472_47271

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4*x^2 + 8*x + 10 → y ≥ y_min ∧ y_min = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l472_47271


namespace NUMINAMATH_CALUDE_sixty_degrees_in_clerts_l472_47256

/-- The number of clerts in a full circle on Venus -/
def venus_full_circle : ℚ := 800

/-- The number of degrees in a full circle on Earth -/
def earth_full_circle : ℚ := 360

/-- Converts degrees to clerts on Venus -/
def degrees_to_clerts (degrees : ℚ) : ℚ :=
  (degrees / earth_full_circle) * venus_full_circle

/-- Theorem: 60 degrees is equivalent to 133.3 (repeating) clerts on Venus -/
theorem sixty_degrees_in_clerts :
  degrees_to_clerts 60 = 133 + 1/3 := by sorry

end NUMINAMATH_CALUDE_sixty_degrees_in_clerts_l472_47256


namespace NUMINAMATH_CALUDE_shaded_area_problem_l472_47250

theorem shaded_area_problem (diagonal : ℝ) (num_squares : ℕ) : 
  diagonal = 10 → num_squares = 25 → 
  (diagonal^2 / 2) = (num_squares : ℝ) * (diagonal^2 / (2 * num_squares : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l472_47250


namespace NUMINAMATH_CALUDE_polynomial_factorization_l472_47286

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 - 2 * b * (c - a)^3 + 3 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * a - 4 * b - 3 * c) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l472_47286


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l472_47294

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 1 + a 3 = 2 * a 2) ∧
  (∃ a : ℕ → ℝ, a 1 + a 3 = 2 * a 2 ∧ ¬is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l472_47294


namespace NUMINAMATH_CALUDE_sin_cos_product_l472_47248

theorem sin_cos_product (α : Real) 
  (h : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7) : 
  Real.sin α * Real.cos α = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_product_l472_47248


namespace NUMINAMATH_CALUDE_no_rational_squares_l472_47266

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_rational_squares :
  ∀ n : ℕ, ∀ r : ℚ, sequence_a n ≠ r^2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_squares_l472_47266


namespace NUMINAMATH_CALUDE_inequality_proof_l472_47245

theorem inequality_proof (a b c x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) : 
  (a^2 * x^2) / ((b*y + c*z) * (b*z + c*y)) + 
  (b^2 * y^2) / ((a*x + c*z) * (a*z + c*x)) + 
  (c^2 * z^2) / ((a*x + b*y) * (a*y + b*x)) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l472_47245


namespace NUMINAMATH_CALUDE_equal_debt_days_l472_47239

/-- The number of days for two borrowers to owe the same amount -/
def days_to_equal_debt (
  morgan_initial : ℚ)
  (morgan_rate : ℚ)
  (olivia_initial : ℚ)
  (olivia_rate : ℚ) : ℚ :=
  (olivia_initial - morgan_initial) / (morgan_rate * morgan_initial - olivia_rate * olivia_initial)

/-- Proof that Morgan and Olivia will owe the same amount after 25/3 days -/
theorem equal_debt_days :
  days_to_equal_debt 200 (12/100) 300 (4/100) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_equal_debt_days_l472_47239


namespace NUMINAMATH_CALUDE_teacher_age_teacher_age_problem_l472_47262

/-- Given a class of students and their teacher, calculate the teacher's age based on how it affects the class average. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℚ) (new_avg_age : ℚ) : ℚ :=
  let total_student_age := num_students * student_avg_age
  let total_new_age := (num_students + 1) * new_avg_age
  total_new_age - total_student_age

/-- Prove that for a class of 25 students with an average age of 26 years, 
    if including the teacher's age increases the average by 1 year, 
    then the teacher's age is 52 years. -/
theorem teacher_age_problem : teacher_age 25 26 27 = 52 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_teacher_age_problem_l472_47262


namespace NUMINAMATH_CALUDE_fraction_addition_simplest_form_l472_47253

theorem fraction_addition : 8 / 15 + 7 / 10 = 37 / 30 := by sorry

theorem simplest_form : ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → Nat.gcd n m = 1 → (n : ℚ) / m = 37 / 30 → n = 37 ∧ m = 30 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplest_form_l472_47253


namespace NUMINAMATH_CALUDE_binomial_mode_is_one_l472_47260

/-- The number of blind boxes in a box -/
def n : ℕ := 6

/-- The probability of drawing a hidden item from one blind box -/
def p : ℚ := 1/6

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℚ :=
  Nat.choose n k * p^k * (1-p)^(n-k)

/-- Theorem: The mode of the binomial distribution with n=6 and p=1/6 is 1 -/
theorem binomial_mode_is_one :
  ∀ k : ℕ, k ≠ 1 → k ≤ n → binomialPMF 1 ≥ binomialPMF k :=
sorry

end NUMINAMATH_CALUDE_binomial_mode_is_one_l472_47260


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_nine_l472_47288

theorem sum_of_solutions_is_nine : 
  let f (x : ℝ) := (12 * x) / (x^2 - 4) - (3 * x) / (x + 2) + 9 / (x - 2)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_nine_l472_47288


namespace NUMINAMATH_CALUDE_sine_cosine_roots_product_l472_47249

theorem sine_cosine_roots_product (α β a b c d : Real) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = Real.sin α ∨ x = Real.sin β) →
  (∀ x, x^2 - c*x + d = 0 ↔ x = Real.cos α ∨ x = Real.cos β) →
  c * d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_product_l472_47249


namespace NUMINAMATH_CALUDE_amusement_park_spending_l472_47263

theorem amusement_park_spending (total : ℕ) (admission : ℕ) (food : ℕ) 
  (h1 : total = 77)
  (h2 : admission = 45)
  (h3 : total = admission + food)
  (h4 : food < admission) :
  admission - food = 13 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l472_47263


namespace NUMINAMATH_CALUDE_walter_chores_l472_47293

theorem walter_chores (total_days : ℕ) (total_earnings : ℕ) 
  (regular_pay : ℕ) (exceptional_pay : ℕ) :
  total_days = 15 →
  total_earnings = 47 →
  regular_pay = 3 →
  exceptional_pay = 4 →
  ∃ (regular_days exceptional_days : ℕ),
    regular_days + exceptional_days = total_days ∧
    regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 2 :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l472_47293


namespace NUMINAMATH_CALUDE_count_injective_functions_count_non_injective_functions_no_surjective_function_l472_47225

/-- Set A with 3 elements -/
def A : Type := Fin 3

/-- Set B with 4 elements -/
def B : Type := Fin 4

/-- The number of injective functions from A to B is 24 -/
theorem count_injective_functions : (A → B) → Nat :=
  fun _ => 24

/-- The number of non-injective functions from A to B is 40 -/
theorem count_non_injective_functions : (A → B) → Nat :=
  fun _ => 40

/-- There does not exist a surjective function from A to B -/
theorem no_surjective_function : ¬∃ (f : A → B), Function.Surjective f := by
  sorry

end NUMINAMATH_CALUDE_count_injective_functions_count_non_injective_functions_no_surjective_function_l472_47225


namespace NUMINAMATH_CALUDE_rope_fraction_proof_l472_47261

theorem rope_fraction_proof (total_ropes : ℕ) (avg_all avg_short avg_long : ℝ) 
  (h_total : total_ropes = 6)
  (h_avg_all : avg_all = 80)
  (h_avg_short : avg_short = 70)
  (h_avg_long : avg_long = 85) :
  ∃ (f : ℝ), 0 < f ∧ f < 1 ∧
  f * total_ropes * avg_short + (1 - f) * total_ropes * avg_long = total_ropes * avg_all ∧
  f = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rope_fraction_proof_l472_47261


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l472_47204

theorem parallelogram_side_length
  (s : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (angle : ℝ)
  (area : ℝ)
  (h1 : side1 = s)
  (h2 : side2 = 3 * s)
  (h3 : angle = π / 3)  -- 60 degrees in radians
  (h4 : area = 27 * Real.sqrt 3)
  (h5 : area = side1 * side2 * Real.sin angle) :
  s = 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l472_47204


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_for_polynomial_l472_47222

theorem smallest_root_of_unity_for_polynomial : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → m < n → ∃ z : ℂ, z^5 - z^3 + 1 = 0 ∧ z^m ≠ 1) ∧
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_for_polynomial_l472_47222


namespace NUMINAMATH_CALUDE_circle_radius_zero_l472_47270

theorem circle_radius_zero (x y : ℝ) :
  x^2 - 4*x + y^2 - 6*y + 13 = 0 → (∃ r : ℝ, r = 0 ∧ (x - 2)^2 + (y - 3)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l472_47270


namespace NUMINAMATH_CALUDE_tiffany_album_distribution_l472_47213

/-- Calculates the number of pictures in each album given the total number of pictures and the number of albums. -/
def pictures_per_album (phone_pics camera_pics num_albums : ℕ) : ℕ :=
  (phone_pics + camera_pics) / num_albums

/-- Proves that given the conditions in the problem, the number of pictures in each album is 4. -/
theorem tiffany_album_distribution :
  let phone_pics := 7
  let camera_pics := 13
  let num_albums := 5
  pictures_per_album phone_pics camera_pics num_albums = 4 := by
sorry

#eval pictures_per_album 7 13 5

end NUMINAMATH_CALUDE_tiffany_album_distribution_l472_47213


namespace NUMINAMATH_CALUDE_binomial_18_6_l472_47205

theorem binomial_18_6 : Nat.choose 18 6 = 13260 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l472_47205


namespace NUMINAMATH_CALUDE_fraction_value_l472_47264

theorem fraction_value (a b c : ℚ) (h1 : a/b = 3) (h2 : b/c = 2) : (a-b)/(c-b) = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l472_47264


namespace NUMINAMATH_CALUDE_intersection_parallel_line_equation_l472_47284

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line. -/
theorem intersection_parallel_line_equation 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop)
  (l_parallel : ℝ → ℝ → Prop)
  (result_line : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 2 * x - 3 * y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x - 4 * y - 2 = 0)
  (h_parallel : ∀ x y, l_parallel x y ↔ 4 * x - 2 * y + 7 = 0)
  (h_result : ∀ x y, result_line x y ↔ 2 * x - y - 18 = 0) :
  ∃ (x₀ y₀ : ℝ), 
    (l₁ x₀ y₀ ∧ l₂ x₀ y₀) ∧ 
    (∃ (k : ℝ), ∀ x y, result_line x y ↔ l_parallel (x - x₀) (y - y₀)) ∧
    result_line x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_equation_l472_47284


namespace NUMINAMATH_CALUDE_find_y_value_l472_47232

theorem find_y_value (x y z : ℚ) : 
  x + y + z = 150 → x + 8 = y - 8 → x + 8 = 4 * z → y = 224 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l472_47232


namespace NUMINAMATH_CALUDE_roses_to_grandmother_l472_47237

/-- Given that Ian had a certain number of roses and distributed them in a specific way,
    this theorem proves how many roses he gave to his grandmother. -/
theorem roses_to_grandmother (total : ℕ) (to_mother : ℕ) (to_sister : ℕ) (kept : ℕ) 
    (h1 : total = 20)
    (h2 : to_mother = 6)
    (h3 : to_sister = 4)
    (h4 : kept = 1) :
    total - (to_mother + to_sister + kept) = 9 := by
  sorry

end NUMINAMATH_CALUDE_roses_to_grandmother_l472_47237


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l472_47201

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l472_47201


namespace NUMINAMATH_CALUDE_last_three_digits_are_427_l472_47219

/-- A function that generates the nth digit in the list of increasing positive integers 
    starting with 2 and containing all numbers with a first digit of 2 -/
def nthDigitInList (n : ℕ) : ℕ := sorry

/-- The last three digits of the 2000-digit sequence -/
def lastThreeDigits : ℕ × ℕ × ℕ := (nthDigitInList 1998, nthDigitInList 1999, nthDigitInList 2000)

theorem last_three_digits_are_427 : lastThreeDigits = (4, 2, 7) := by sorry

end NUMINAMATH_CALUDE_last_three_digits_are_427_l472_47219


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l472_47226

/-- Given a geometric sequence of positive integers where the first term is 5 and the fourth term is 405,
    prove that the fifth term is 405. -/
theorem fifth_term_of_geometric_sequence (a : ℕ+) (r : ℕ+) : 
  a = 5 → a * r^3 = 405 → a * r^4 = 405 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l472_47226


namespace NUMINAMATH_CALUDE_max_value_linear_program_l472_47217

theorem max_value_linear_program (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + 2*y ≤ 3) 
  (h3 : x - 2*y ≤ 1) : 
  ∀ z, z = x + 6*y → z ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_linear_program_l472_47217


namespace NUMINAMATH_CALUDE_third_bouquet_carnations_l472_47210

/-- Theorem: Given three bouquets of carnations with specific conditions, 
    the third bouquet contains 13 carnations. -/
theorem third_bouquet_carnations 
  (total_bouquets : ℕ)
  (first_bouquet : ℕ)
  (second_bouquet : ℕ)
  (average_carnations : ℕ)
  (h1 : total_bouquets = 3)
  (h2 : first_bouquet = 9)
  (h3 : second_bouquet = 14)
  (h4 : average_carnations = 12)
  (h5 : average_carnations * total_bouquets = first_bouquet + second_bouquet + (total_bouquets - 2)) :
  total_bouquets - 2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_third_bouquet_carnations_l472_47210


namespace NUMINAMATH_CALUDE_distance_to_origin_l472_47244

theorem distance_to_origin : ∃ (M : ℝ × ℝ), 
  M = (-5, 12) ∧ Real.sqrt ((-5)^2 + 12^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l472_47244


namespace NUMINAMATH_CALUDE_total_albums_l472_47287

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina

/-- The theorem to be proved -/
theorem total_albums (a : Albums) (h : album_conditions a) : 
  a.adele + a.bridget + a.katrina + a.miriam = 585 := by
  sorry


end NUMINAMATH_CALUDE_total_albums_l472_47287


namespace NUMINAMATH_CALUDE_john_pushups_l472_47207

theorem john_pushups (zachary_pushups : ℕ) (david_more_than_zachary : ℕ) (john_less_than_david : ℕ)
  (h1 : zachary_pushups = 51)
  (h2 : david_more_than_zachary = 22)
  (h3 : john_less_than_david = 4) :
  zachary_pushups + david_more_than_zachary - john_less_than_david = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_john_pushups_l472_47207


namespace NUMINAMATH_CALUDE_inequality_solution_l472_47272

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 12) / (x - 3) < 0 ↔ (x > -2 ∧ x < 3) ∨ (x > 3 ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l472_47272


namespace NUMINAMATH_CALUDE_coloring_book_problem_l472_47277

/-- The number of pictures colored given the initial count and remaining count --/
def pictures_colored (initial_count : ℕ) (remaining_count : ℕ) : ℕ :=
  initial_count - remaining_count

/-- Theorem stating that given two coloring books with 44 pictures each and 68 pictures left to color, 
    the number of pictures colored is 20 --/
theorem coloring_book_problem :
  let book1_count : ℕ := 44
  let book2_count : ℕ := 44
  let total_count : ℕ := book1_count + book2_count
  let remaining_count : ℕ := 68
  pictures_colored total_count remaining_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l472_47277


namespace NUMINAMATH_CALUDE_added_amount_proof_l472_47280

theorem added_amount_proof (x y : ℝ) : x = 16 ∧ 3 * (2 * x + y) = 111 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_proof_l472_47280


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l472_47278

/-- Given a quadratic equation x^2 + mx + (m^2 - 3m + 3) = 0, 
    if its roots are reciprocals of each other, then m = 2 -/
theorem reciprocal_roots_quadratic (m : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
   x^2 + m*x + (m^2 - 3*m + 3) = 0 ∧
   y^2 + m*y + (m^2 - 3*m + 3) = 0 ∧
   x*y = 1) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l472_47278


namespace NUMINAMATH_CALUDE_congruent_count_l472_47267

theorem congruent_count (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 1) (Finset.range 251)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_congruent_count_l472_47267


namespace NUMINAMATH_CALUDE_village_population_problem_l472_47202

theorem village_population_problem (X : ℝ) : 
  (X > 0) →
  (0.9 * X * 0.75 + 0.9 * X * 0.25 * 0.15 = 5265) →
  X = 7425 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l472_47202


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l472_47297

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then x else 3
def b : Fin 2 → ℝ := λ i => if i = 0 then 3 else 1

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, dot_product (a x) b = 0 → x = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l472_47297


namespace NUMINAMATH_CALUDE_no_function_satisfying_condition_l472_47296

open Real

-- Define the type for positive real numbers
def PositiveReal := {x : ℝ // x > 0}

-- State the theorem
theorem no_function_satisfying_condition :
  ¬ ∃ (f : PositiveReal → PositiveReal),
    ∀ (x y : PositiveReal),
      (f (⟨x.val + y.val, sorry⟩)).val ^ 2 ≥ (f x).val ^ 2 * (1 + y.val * (f x).val) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfying_condition_l472_47296


namespace NUMINAMATH_CALUDE_chess_tournament_rounds_l472_47281

theorem chess_tournament_rounds (total_games : ℕ) (h : total_games = 224) :
  ∃ (participants rounds : ℕ),
    participants > 1 ∧
    rounds > 0 ∧
    participants * (participants - 1) * rounds = 2 * total_games ∧
    rounds = 8 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_rounds_l472_47281


namespace NUMINAMATH_CALUDE_ratio_and_mean_problem_l472_47209

theorem ratio_and_mean_problem (a b c : ℕ+) (h_ratio : (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 4)
  (h_mean : (a + b + c : ℚ) / 3 = 42) : a = 28 := by
  sorry

end NUMINAMATH_CALUDE_ratio_and_mean_problem_l472_47209


namespace NUMINAMATH_CALUDE_bunny_teddy_ratio_l472_47236

def initial_teddies : ℕ := 5
def koala_bears : ℕ := 1
def additional_teddies_per_bunny : ℕ := 2
def total_mascots : ℕ := 51

def bunnies : ℕ := (total_mascots - initial_teddies - koala_bears) / (additional_teddies_per_bunny + 1)

theorem bunny_teddy_ratio :
  bunnies / initial_teddies = 3 ∧ bunnies % initial_teddies = 0 := by
  sorry

end NUMINAMATH_CALUDE_bunny_teddy_ratio_l472_47236


namespace NUMINAMATH_CALUDE_michael_fish_count_l472_47258

theorem michael_fish_count (initial_fish : ℕ) (given_fish : ℕ) : 
  initial_fish = 31 → given_fish = 18 → initial_fish + given_fish = 49 := by
sorry

end NUMINAMATH_CALUDE_michael_fish_count_l472_47258


namespace NUMINAMATH_CALUDE_apple_eraser_distribution_l472_47227

/-- Given a total of 84 items consisting of apples and erasers, prove the number of apples each friend receives and the number of erasers the teacher receives. -/
theorem apple_eraser_distribution (a e : ℕ) (h : a + e = 84) :
  ∃ (friend_apples teacher_erasers : ℚ),
    friend_apples = a / 3 ∧
    teacher_erasers = e / 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_eraser_distribution_l472_47227


namespace NUMINAMATH_CALUDE_triangle_formation_l472_47273

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 7 ∧
  ¬can_form_triangle 1 3 4 ∧
  ¬can_form_triangle 2 2 7 ∧
  ¬can_form_triangle 3 3 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l472_47273


namespace NUMINAMATH_CALUDE_max_spheres_in_frustum_l472_47241

structure Frustum where
  height : ℝ

structure Sphere where
  radius : ℝ

def is_tangent_to_frustum (s : Sphere) (f : Frustum) : Prop := sorry

def is_tangent_to_sphere (s1 s2 : Sphere) : Prop := sorry

def can_fit_inside_frustum (s : Sphere) (f : Frustum) : Prop := sorry

theorem max_spheres_in_frustum (f : Frustum) (o1 o2 : Sphere) 
  (h_height : f.height = 8)
  (h_o1_radius : o1.radius = 2)
  (h_o2_radius : o2.radius = 3)
  (h_o1_tangent : is_tangent_to_frustum o1 f)
  (h_o2_tangent : is_tangent_to_frustum o2 f)
  (h_o1_o2_tangent : is_tangent_to_sphere o1 o2) :
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬∃ (spheres : Fin m → Sphere), 
      (∀ i, (spheres i).radius = 3 ∧ 
            can_fit_inside_frustum (spheres i) f ∧
            (∀ j, i ≠ j → is_tangent_to_sphere (spheres i) (spheres j)))) :=
sorry

end NUMINAMATH_CALUDE_max_spheres_in_frustum_l472_47241


namespace NUMINAMATH_CALUDE_gcd_3060_561_l472_47229

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3060_561_l472_47229


namespace NUMINAMATH_CALUDE_intersection_M_N_l472_47231

-- Define the sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- State the theorem
theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l472_47231


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_plus_b_l472_47218

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 7]

-- Theorem statement
theorem a_perpendicular_to_a_plus_b :
  (a 0 * (a 0 + b 0) + a 1 * (a 1 + b 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_plus_b_l472_47218


namespace NUMINAMATH_CALUDE_xy_value_l472_47215

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^(Real.sqrt y) = 27) (h2 : (Real.sqrt x)^y = 9) : 
  x * y = 12 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l472_47215


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l472_47211

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def line_l : Set (ℝ × ℝ) := {p | p.1 = -3}

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the properties of the moving circle
def is_valid_circle (c : Circle) : Prop :=
  (c.center.1 - 3)^2 + c.center.2^2 = c.radius^2 ∧  -- passes through A(3,0)
  c.center.1 + c.radius = -3                        -- tangent to x = -3

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ c : Circle, is_valid_circle c →
  ∃ x y : ℝ, c.center = (x, y) ∧ y^2 = 12 * x :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l472_47211


namespace NUMINAMATH_CALUDE_fraction_increase_l472_47243

theorem fraction_increase (x y : ℝ) (h : x + y ≠ 0) :
  (3 * (2 * x) * (2 * y)) / ((2 * x) + (2 * y)) = 2 * ((3 * x * y) / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l472_47243


namespace NUMINAMATH_CALUDE_min_rectangles_cover_l472_47255

/-- A point in the unit square -/
structure Point where
  x : Real
  y : Real
  x_in_unit : 0 < x ∧ x < 1
  y_in_unit : 0 < y ∧ y < 1

/-- A rectangle with sides parallel to the unit square -/
structure Rectangle where
  left : Real
  right : Real
  bottom : Real
  top : Real
  valid : 0 ≤ left ∧ left < right ∧ right ≤ 1 ∧
          0 ≤ bottom ∧ bottom < top ∧ top ≤ 1

/-- The theorem statement -/
theorem min_rectangles_cover (n : Nat) (S : Finset Point) :
  S.card = n →
  ∃ (k : Nat) (R : Finset Rectangle),
    R.card = k ∧
    (∀ p ∈ S, ∀ r ∈ R, ¬(r.left < p.x ∧ p.x < r.right ∧ r.bottom < p.y ∧ p.y < r.top)) ∧
    (∀ x y : Real, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
      (∀ p ∈ S, p.x ≠ x ∨ p.y ≠ y) →
      ∃ r ∈ R, r.left < x ∧ x < r.right ∧ r.bottom < y ∧ y < r.top) ∧
    k = 2 * n + 2 ∧
    (∀ m : Nat, m < k →
      ¬∃ (R' : Finset Rectangle),
        R'.card = m ∧
        (∀ p ∈ S, ∀ r ∈ R', ¬(r.left < p.x ∧ p.x < r.right ∧ r.bottom < p.y ∧ p.y < r.top)) ∧
        (∀ x y : Real, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
          (∀ p ∈ S, p.x ≠ x ∨ p.y ≠ y) →
          ∃ r ∈ R', r.left < x ∧ x < r.right ∧ r.bottom < y ∧ y < r.top)) :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_cover_l472_47255


namespace NUMINAMATH_CALUDE_average_leaves_theorem_l472_47298

/-- The number of leaves that fell in the first hour -/
def leaves_first_hour : ℕ := 7

/-- The rate of leaves falling per hour for the second and third hour -/
def leaves_rate_later : ℕ := 4

/-- The total number of hours of observation -/
def total_hours : ℕ := 3

/-- The total number of leaves that fell during the observation period -/
def total_leaves : ℕ := leaves_first_hour + leaves_rate_later * (total_hours - 1)

/-- The average number of leaves falling per hour -/
def average_leaves_per_hour : ℚ := total_leaves / total_hours

theorem average_leaves_theorem : average_leaves_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_leaves_theorem_l472_47298


namespace NUMINAMATH_CALUDE_min_games_for_condition_l472_47246

/-- Represents a football championship. -/
structure Championship where
  teams : Nat
  games_played : Nat

/-- Calculates the total number of possible games in a championship. -/
def total_possible_games (c : Championship) : Nat :=
  c.teams * (c.teams - 1) / 2

/-- Defines the property that among any three teams, at least two have played against each other. -/
def satisfies_condition (c : Championship) : Prop :=
  ∀ (a b d : Fin c.teams), a ≠ b ∧ b ≠ d ∧ a ≠ d →
    ∃ (x y : Fin c.teams), (x = a ∧ y = b) ∨ (x = b ∧ y = d) ∨ (x = a ∧ y = d)

/-- The main theorem to be proved. -/
theorem min_games_for_condition (c : Championship) 
  (h1 : c.teams = 20)
  (h2 : c.games_played ≥ 90)
  (h3 : ∀ (c' : Championship), c'.teams = 20 ∧ c'.games_played < 90 → ¬satisfies_condition c') :
  satisfies_condition c :=
sorry

end NUMINAMATH_CALUDE_min_games_for_condition_l472_47246


namespace NUMINAMATH_CALUDE_unique_number_l472_47233

/-- A structure representing the statements made by a boy -/
structure BoyStatements where
  statement1 : Nat → Prop
  statement2 : Nat → Prop

/-- The set of statements made by each boy -/
def boyStatements : Fin 3 → BoyStatements
  | 0 => ⟨λ n => n % 10 = 6, λ n => n % 7 = 0⟩  -- Andrey
  | 1 => ⟨λ n => n > 26, λ n => n % 10 = 8⟩     -- Borya
  | 2 => ⟨λ n => n % 13 = 0, λ n => n < 27⟩     -- Sasha
  | _ => ⟨λ _ => False, λ _ => False⟩           -- Unreachable case

/-- The theorem stating that 91 is the only two-digit number satisfying all conditions -/
theorem unique_number : ∃! n : Nat, 10 ≤ n ∧ n < 100 ∧
  (∀ i : Fin 3, (boyStatements i).statement1 n ≠ (boyStatements i).statement2 n) ∧
  (∀ i : Fin 3, (boyStatements i).statement1 n ∨ (boyStatements i).statement2 n) :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l472_47233


namespace NUMINAMATH_CALUDE_truck_capacity_l472_47279

theorem truck_capacity (total_boxes : ℕ) (num_trips : ℕ) (h1 : total_boxes = 871) (h2 : num_trips = 218) :
  total_boxes / num_trips = 4 := by
sorry

end NUMINAMATH_CALUDE_truck_capacity_l472_47279


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l472_47223

theorem multiplicative_inverse_203_mod_301 : ∃ a : ℕ, 0 ≤ a ∧ a < 301 ∧ (203 * a) % 301 = 1 :=
by
  use 238
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l472_47223


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l472_47254

theorem power_three_mod_eleven : 3^2048 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l472_47254


namespace NUMINAMATH_CALUDE_words_exceeded_proof_l472_47291

def word_limit : ℕ := 1000
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650

theorem words_exceeded_proof :
  (saturday_words + sunday_words) - word_limit = 100 := by
  sorry

end NUMINAMATH_CALUDE_words_exceeded_proof_l472_47291


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l472_47208

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 32 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l472_47208


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l472_47285

theorem square_sum_from_difference_and_product (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a * b = 50) : 
  a^2 + b^2 = 164 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l472_47285


namespace NUMINAMATH_CALUDE_complement_of_union_equals_interval_l472_47274

-- Define the universal set U
def U : Set ℝ := {x | -5 < x ∧ x < 5}

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- State the theorem
theorem complement_of_union_equals_interval :
  (U \ (A ∪ B)) = {x | -5 < x ∧ x ≤ -2} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_interval_l472_47274


namespace NUMINAMATH_CALUDE_wireless_internet_percentage_l472_47275

/-- The percentage of major airline companies that offer free on-board snacks -/
def snacks_percentage : ℝ := 70

/-- The greatest possible percentage of major airline companies that offer both wireless internet and free on-board snacks -/
def both_services_percentage : ℝ := 50

/-- The percentage of major airline companies that equip their planes with wireless internet access -/
def wireless_percentage : ℝ := 50

theorem wireless_internet_percentage :
  wireless_percentage = 50 :=
sorry

end NUMINAMATH_CALUDE_wireless_internet_percentage_l472_47275


namespace NUMINAMATH_CALUDE_david_cindy_walk_difference_l472_47203

theorem david_cindy_walk_difference (AC CB : ℝ) (h1 : AC = 8) (h2 : CB = 15) :
  let AB : ℝ := Real.sqrt (AC^2 + CB^2)
  AC + CB - AB = 6 := by sorry

end NUMINAMATH_CALUDE_david_cindy_walk_difference_l472_47203


namespace NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l472_47206

-- Define the color type
inductive Color
  | Blue
  | Green
  | Red

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- State the existence of at least one point of each color
axiom exists_blue : ∃ p : Point, coloring p = Color.Blue
axiom exists_green : ∃ p : Point, coloring p = Color.Green
axiom exists_red : ∃ p : Point, coloring p = Color.Red

-- Define a right triangle
def is_right_triangle (p q r : Point) : Prop := sorry

-- State the theorem
theorem exists_right_triangle_with_different_colors :
  ∃ p q r : Point, is_right_triangle p q r ∧
    coloring p ≠ coloring q ∧
    coloring q ≠ coloring r ∧
    coloring r ≠ coloring p :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l472_47206


namespace NUMINAMATH_CALUDE_proposition_is_false_l472_47251

theorem proposition_is_false : 
  ¬(∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l472_47251


namespace NUMINAMATH_CALUDE_vector_projection_l472_47289

/-- The projection of vector a in the direction of vector b is equal to √65/5 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (2, 3) → b = (-4, 7) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l472_47289


namespace NUMINAMATH_CALUDE_total_jogging_distance_l472_47220

def monday_distance : ℕ := 2
def tuesday_distance : ℕ := 5
def wednesday_distance : ℕ := 9

theorem total_jogging_distance :
  monday_distance + tuesday_distance + wednesday_distance = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_jogging_distance_l472_47220


namespace NUMINAMATH_CALUDE_croissants_for_breakfast_l472_47242

theorem croissants_for_breakfast (total_items cakes pizzas : ℕ) 
  (h1 : total_items = 110)
  (h2 : cakes = 18)
  (h3 : pizzas = 30) :
  total_items - cakes - pizzas = 62 := by
  sorry

end NUMINAMATH_CALUDE_croissants_for_breakfast_l472_47242


namespace NUMINAMATH_CALUDE_fourth_grade_classrooms_difference_l472_47299

theorem fourth_grade_classrooms_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 20 →
  guinea_pigs_per_class = 3 →
  num_classes = 5 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_classrooms_difference_l472_47299


namespace NUMINAMATH_CALUDE_base_8_to_16_digit_count_l472_47290

theorem base_8_to_16_digit_count :
  ∀ n : ℕ,
  (1000 ≤ n ∧ n ≤ 7777) →  -- 4 digits in base 8
  (512 ≤ n ∧ n ≤ 4095) →   -- Equivalent range in decimal
  (0x200 ≤ n ∧ n ≤ 0xFFF)  -- 3 digits in base 16
  := by sorry

end NUMINAMATH_CALUDE_base_8_to_16_digit_count_l472_47290
