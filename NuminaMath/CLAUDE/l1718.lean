import Mathlib

namespace NUMINAMATH_CALUDE_not_one_zero_pronounced_l1718_171879

def number_of_pronounced_zeros (n : Nat) : Nat :=
  sorry -- Implementation of counting pronounced zeros

theorem not_one_zero_pronounced (n : Nat) (h : n = 83721000) : 
  number_of_pronounced_zeros n ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_not_one_zero_pronounced_l1718_171879


namespace NUMINAMATH_CALUDE_simplify_expression_l1718_171805

theorem simplify_expression (a b : ℝ) : a * (4 * a - b) - (2 * a + b) * (2 * a - b) = b^2 - a * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1718_171805


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1718_171891

theorem solve_system_of_equations (x y : ℝ) : 
  (2 * x - y = 12) → (x = 5) → (y = -2) := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1718_171891


namespace NUMINAMATH_CALUDE_function_range_theorem_l1718_171818

/-- Given a function f(x) = |2x - 1| + |x - 2a|, if for all x ∈ [1, 2], f(x) ≤ 4,
    then the range of real values for a is [1/2, 3/2]. -/
theorem function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x = |2 * x - 1| + |x - 2 * a|) →
  (∀ x ∈ Set.Icc 1 2, f x ≤ 4) →
  a ∈ Set.Icc (1/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l1718_171818


namespace NUMINAMATH_CALUDE_remaining_apples_l1718_171871

def initial_apples : ℕ := 127
def given_apples : ℕ := 88

theorem remaining_apples : initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_remaining_apples_l1718_171871


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1718_171864

theorem trigonometric_identities :
  -- Part 1
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = (Real.sqrt 5 - 1) / 32 ∧
  -- Part 2
  ∀ α : Real, 
    π / 2 < α ∧ α < π →  -- α is in the second quadrant
    Real.sin α = Real.sqrt 15 / 4 →
    Real.sin (α + π / 4) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1718_171864


namespace NUMINAMATH_CALUDE_conic_section_is_hyperbola_l1718_171887

/-- The equation (x+7)^2 = (5y-6)^2 + 125 defines a hyperbola -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), (x + 7)^2 = (5*y - 6)^2 + 125 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0 :=
by sorry

end NUMINAMATH_CALUDE_conic_section_is_hyperbola_l1718_171887


namespace NUMINAMATH_CALUDE_system_solution_l1718_171804

theorem system_solution (x y : ℚ) (h1 : x + 2*y = -1) (h2 : 2*x + y = 3) : x + y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1718_171804


namespace NUMINAMATH_CALUDE_x_value_is_five_l1718_171868

theorem x_value_is_five (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_is_five_l1718_171868


namespace NUMINAMATH_CALUDE_digit_123_is_1_l1718_171898

/-- The decimal representation of 47/740 -/
def decimal_rep : ℚ := 47 / 740

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 12

/-- The position we're interested in -/
def target_position : ℕ := 123

/-- The function that returns the nth digit after the decimal point in the decimal representation of 47/740 -/
noncomputable def nth_digit (n : ℕ) : ℕ :=
  sorry

theorem digit_123_is_1 : nth_digit target_position = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_123_is_1_l1718_171898


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l1718_171800

theorem complex_magnitude_one (z : ℂ) (r : ℝ) 
  (h1 : |r| < 2) 
  (h2 : z + z⁻¹ = r) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l1718_171800


namespace NUMINAMATH_CALUDE_product_inequality_l1718_171890

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1718_171890


namespace NUMINAMATH_CALUDE_movie_admission_problem_l1718_171807

theorem movie_admission_problem (total_admitted : ℕ) 
  (west_side_total : ℕ) (west_side_denied_percent : ℚ)
  (mountaintop_total : ℕ) (mountaintop_denied_percent : ℚ)
  (first_school_denied_percent : ℚ) :
  total_admitted = 148 →
  west_side_total = 90 →
  west_side_denied_percent = 70/100 →
  mountaintop_total = 50 →
  mountaintop_denied_percent = 1/2 →
  first_school_denied_percent = 20/100 →
  ∃ (first_school_total : ℕ),
    first_school_total = 120 ∧
    total_admitted = 
      (first_school_total * (1 - first_school_denied_percent)).floor +
      (west_side_total * (1 - west_side_denied_percent)).floor +
      (mountaintop_total * (1 - mountaintop_denied_percent)).floor :=
by sorry

end NUMINAMATH_CALUDE_movie_admission_problem_l1718_171807


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l1718_171872

/-- Calculates the daily evaporation rate of water in a glass -/
theorem water_evaporation_rate 
  (initial_amount : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) :
  initial_amount = 12 →
  evaporation_period = 22 →
  evaporation_percentage = 5.5 →
  (initial_amount * evaporation_percentage / 100) / evaporation_period = 0.03 :=
by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l1718_171872


namespace NUMINAMATH_CALUDE_race_probability_l1718_171823

theorem race_probability (pX pY pZ : ℚ) : 
  pX = 1/4 → pY = 1/12 → pZ = 1/7 → 
  (pX + pY + pZ : ℚ) = 10/21 := by sorry

end NUMINAMATH_CALUDE_race_probability_l1718_171823


namespace NUMINAMATH_CALUDE_cellphone_survey_rate_increase_is_30_percent_l1718_171832

/-- Calculates the percentage increase in pay rate for cellphone surveys -/
def cellphone_survey_rate_increase (regular_rate : ℚ) (total_surveys : ℕ) 
  (cellphone_surveys : ℕ) (total_earnings : ℚ) : ℚ :=
  let regular_earnings := regular_rate * total_surveys
  let additional_earnings := total_earnings - regular_earnings
  let additional_rate := additional_earnings / cellphone_surveys
  let cellphone_rate := regular_rate + additional_rate
  (cellphone_rate - regular_rate) / regular_rate * 100

/-- Theorem stating the percentage increase in pay rate for cellphone surveys -/
theorem cellphone_survey_rate_increase_is_30_percent :
  cellphone_survey_rate_increase 10 100 60 1180 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cellphone_survey_rate_increase_is_30_percent_l1718_171832


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l1718_171877

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the composition function g(x) = f(|x+2|)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (|x + 2|)

-- State the theorem
theorem monotonic_increase_interval
  (f : ℝ → ℝ) (h : DecreasingFunction f) :
  StrictMonoOn (g f) (Set.Iio (-2)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l1718_171877


namespace NUMINAMATH_CALUDE_induction_step_for_even_numbers_l1718_171881

theorem induction_step_for_even_numbers (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) :
  let n := k + 2
  Even n ∧ n > k :=
by sorry

end NUMINAMATH_CALUDE_induction_step_for_even_numbers_l1718_171881


namespace NUMINAMATH_CALUDE_tan_315_degrees_l1718_171885

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l1718_171885


namespace NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l1718_171888

/-- The number of pills needed to meet the weekly recommended amount of Vitamin A -/
def pills_needed (vitamin_per_pill : ℕ) (daily_recommended : ℕ) (days_per_week : ℕ) : ℕ :=
  (daily_recommended * days_per_week) / vitamin_per_pill

/-- Proof that 28 pills are needed per week to meet the recommended Vitamin A intake -/
theorem vitamin_a_weekly_pills : pills_needed 50 200 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l1718_171888


namespace NUMINAMATH_CALUDE_monas_weekly_miles_l1718_171878

/-- Represents the days of the week Mona bikes --/
inductive BikingDay
| Monday
| Wednesday
| Saturday

/-- Represents Mona's biking schedule --/
structure BikingSchedule where
  monday_miles : ℕ
  wednesday_miles : ℕ
  saturday_miles : ℕ

/-- Mona's actual biking schedule --/
def monas_schedule : BikingSchedule :=
  { monday_miles := 6
  , wednesday_miles := 12
  , saturday_miles := 12 }

/-- The total miles Mona bikes in a week --/
def total_miles (schedule : BikingSchedule) : ℕ :=
  schedule.monday_miles + schedule.wednesday_miles + schedule.saturday_miles

/-- Theorem stating that Mona bikes 30 miles each week --/
theorem monas_weekly_miles :
  total_miles monas_schedule = 30 ∧
  monas_schedule.wednesday_miles = 12 ∧
  monas_schedule.saturday_miles = 2 * monas_schedule.monday_miles ∧
  monas_schedule.monday_miles = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_monas_weekly_miles_l1718_171878


namespace NUMINAMATH_CALUDE_stella_annual_income_l1718_171813

/-- Calculates the annual income given monthly income and months of unpaid leave -/
def annual_income (monthly_income : ℕ) (unpaid_leave_months : ℕ) : ℕ :=
  monthly_income * (12 - unpaid_leave_months)

/-- Theorem: Given Stella's monthly income and unpaid leave, her annual income is 49190 dollars -/
theorem stella_annual_income :
  annual_income 4919 2 = 49190 := by
  sorry

end NUMINAMATH_CALUDE_stella_annual_income_l1718_171813


namespace NUMINAMATH_CALUDE_no_three_digit_base_7_equals_two_digit_base_6_l1718_171873

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if a number is representable as a two-digit number in base 6 --/
def is_two_digit_base_6 (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), d1 < 6 ∧ d2 < 6 ∧ n = to_base_10 [d1, d2] 6

theorem no_three_digit_base_7_equals_two_digit_base_6 :
  ¬ ∃ (d1 d2 d3 : ℕ), 
    d1 > 0 ∧ d1 < 7 ∧ d2 < 7 ∧ d3 < 7 ∧ 
    is_two_digit_base_6 (to_base_10 [d1, d2, d3] 7) :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_base_7_equals_two_digit_base_6_l1718_171873


namespace NUMINAMATH_CALUDE_tommy_truck_count_l1718_171855

/-- The number of trucks Tommy saw -/
def num_trucks : ℕ := 12

/-- The number of cars Tommy saw -/
def num_cars : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := 100

/-- The number of wheels per vehicle -/
def wheels_per_vehicle : ℕ := 4

theorem tommy_truck_count :
  num_trucks * wheels_per_vehicle + num_cars * wheels_per_vehicle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_tommy_truck_count_l1718_171855


namespace NUMINAMATH_CALUDE_five_digit_divisibility_count_l1718_171883

/-- The count of 5-digit numbers with a specific divisibility property -/
theorem five_digit_divisibility_count : 
  (Finset.filter 
    (fun n : ℕ => 
      10000 ≤ n ∧ n ≤ 99999 ∧ 
      (n / 50 + n % 50) % 7 = 0)
    (Finset.range 100000)).card = 14400 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_count_l1718_171883


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1718_171860

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * Real.cos (4 / (3 * x)) + x^2 / 2 else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1718_171860


namespace NUMINAMATH_CALUDE_function_equation_solution_l1718_171849

/-- Given a function f : ℝ → ℝ satisfying the equation
    f(x) + f(2x+y) + 7xy = f(3x - 2y) + 3x^2 + 2
    for all real numbers x and y, prove that f(15) = 1202 -/
theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 7*x*y = f (3*x - 2*y) + 3*x^2 + 2) : 
  f 15 = 1202 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1718_171849


namespace NUMINAMATH_CALUDE_function_increasing_range_l1718_171882

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then Real.log x / Real.log a else a * x - 2

theorem function_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_range_l1718_171882


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l1718_171835

theorem ceiling_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l1718_171835


namespace NUMINAMATH_CALUDE_sixth_power_sum_l1718_171838

theorem sixth_power_sum (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6*a^4*b + 9*a^2*b^2 - 2*b^3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l1718_171838


namespace NUMINAMATH_CALUDE_memory_sequence_increment_prime_l1718_171822

def memory_sequence : ℕ → ℕ
  | 0 => 6
  | n + 1 => memory_sequence n + Nat.gcd (memory_sequence n) (n + 1)

theorem memory_sequence_increment_prime (n : ℕ) :
  n > 0 → (memory_sequence n - memory_sequence (n - 1) = 1) ∨
          Nat.Prime (memory_sequence n - memory_sequence (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_memory_sequence_increment_prime_l1718_171822


namespace NUMINAMATH_CALUDE_chord_length_l1718_171837

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 4*y + 6 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x - y - 5 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), circle_equation x y → line_equation x y → 
      ∃ (x1 y1 x2 y2 : ℝ), 
        circle_equation x1 y1 ∧ circle_equation x2 y2 ∧
        line_equation x1 y1 ∧ line_equation x2 y2 ∧
        (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) ∧
    chord_length = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1718_171837


namespace NUMINAMATH_CALUDE_total_books_l1718_171848

-- Define the number of books for each person
def betty_books (x : ℚ) : ℚ := x

def sister_books (x : ℚ) : ℚ := x + (1/4) * x

def cousin_books (x : ℚ) : ℚ := 2 * (sister_books x)

def friend_books (x y : ℚ) : ℚ := 
  betty_books x + sister_books x + cousin_books x - y

-- Theorem statement
theorem total_books (x y : ℚ) : 
  betty_books x + sister_books x + cousin_books x + friend_books x y = (19/2) * x - y := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1718_171848


namespace NUMINAMATH_CALUDE_beach_house_pool_problem_l1718_171870

theorem beach_house_pool_problem (total_people : ℕ) (legs_in_pool : ℕ) (legs_per_person : ℕ) :
  total_people = 26 →
  legs_in_pool = 34 →
  legs_per_person = 2 →
  total_people - (legs_in_pool / legs_per_person) = 9 := by
sorry

end NUMINAMATH_CALUDE_beach_house_pool_problem_l1718_171870


namespace NUMINAMATH_CALUDE_range_of_a_l1718_171831

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (a ∈ Set.Icc (-1) 1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1718_171831


namespace NUMINAMATH_CALUDE_certain_number_is_six_l1718_171853

theorem certain_number_is_six : ∃ x : ℝ, (7 * x - 6 - 12 = 4 * x) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_six_l1718_171853


namespace NUMINAMATH_CALUDE_rent_increase_proof_l1718_171809

theorem rent_increase_proof (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (increase_rate : ℝ) 
  (h1 : n = 4)
  (h2 : initial_avg = 800)
  (h3 : new_avg = 870)
  (h4 : increase_rate = 0.2) :
  ∃ (original_rent : ℝ), 
    (n * new_avg - n * initial_avg) / increase_rate = original_rent ∧ 
    original_rent = 1400 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_proof_l1718_171809


namespace NUMINAMATH_CALUDE_smallest_three_star_number_three_star_common_divisor_with_30_l1718_171844

/-- A three-star number is a three-digit positive integer that is the product of three distinct prime numbers. -/
def IsThreeStarNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r

/-- The smallest three-star number is 102. -/
theorem smallest_three_star_number : 
  IsThreeStarNumber 102 ∧ ∀ n, IsThreeStarNumber n → 102 ≤ n :=
sorry

/-- Every three-star number has a common divisor with 30 greater than 1. -/
theorem three_star_common_divisor_with_30 (n : ℕ) (h : IsThreeStarNumber n) : 
  ∃ d : ℕ, d > 1 ∧ d ∣ n ∧ d ∣ 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_star_number_three_star_common_divisor_with_30_l1718_171844


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l1718_171815

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number of visitors -/
def visitors : ℕ := 876000

/-- The scientific notation representation of the number of visitors -/
def visitors_scientific : ScientificNotation :=
  { coefficient := 8.76
  , exponent := 5
  , h1 := by sorry }

theorem visitors_in_scientific_notation :
  (visitors : ℝ) = visitors_scientific.coefficient * (10 : ℝ) ^ visitors_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l1718_171815


namespace NUMINAMATH_CALUDE_sixth_sampled_item_is_101_l1718_171842

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  startNumber : ℕ

/-- Calculates the nth sampled item number in a systematic sampling -/
def nthSampledItem (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.startNumber + (s.totalItems / s.sampleSize) * (n - 1)

/-- The main theorem to prove -/
theorem sixth_sampled_item_is_101 :
  let s : SystematicSampling := {
    totalItems := 1000,
    sampleSize := 50,
    startNumber := 1
  }
  nthSampledItem s 6 = 101 := by sorry

end NUMINAMATH_CALUDE_sixth_sampled_item_is_101_l1718_171842


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1718_171858

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 13 * x + b * y + c * z = 0)
  (eq2 : a * x + 23 * y + c * z = 0)
  (eq3 : a * x + b * y + 42 * z = 0)
  (ha : a ≠ 13)
  (hx : x ≠ 0) :
  13 / (a - 13) + 23 / (b - 23) + 42 / (c - 42) = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1718_171858


namespace NUMINAMATH_CALUDE_golf_tournament_total_cost_l1718_171806

/-- The cost of the golf tournament given the electricity bill cost and additional expenses -/
def golf_tournament_cost (electricity_bill : ℝ) (cell_phone_additional : ℝ) : ℝ :=
  let cell_phone_expense := electricity_bill + cell_phone_additional
  let tournament_additional_cost := 0.2 * cell_phone_expense
  cell_phone_expense + tournament_additional_cost

/-- Theorem stating the total cost of the golf tournament -/
theorem golf_tournament_total_cost :
  golf_tournament_cost 800 400 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_golf_tournament_total_cost_l1718_171806


namespace NUMINAMATH_CALUDE_jim_siblings_l1718_171866

-- Define the characteristics
inductive EyeColor
| Blue
| Brown

inductive HairColor
| Blond
| Black

inductive GlassesWorn
| Yes
| No

-- Define a student structure
structure Student where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  glassesWorn : GlassesWorn

-- Define the list of students
def students : List Student := [
  ⟨"Benjamin", EyeColor.Blue, HairColor.Blond, GlassesWorn.Yes⟩,
  ⟨"Jim", EyeColor.Brown, HairColor.Blond, GlassesWorn.No⟩,
  ⟨"Nadeen", EyeColor.Brown, HairColor.Black, GlassesWorn.Yes⟩,
  ⟨"Austin", EyeColor.Blue, HairColor.Black, GlassesWorn.No⟩,
  ⟨"Tevyn", EyeColor.Blue, HairColor.Blond, GlassesWorn.Yes⟩,
  ⟨"Sue", EyeColor.Brown, HairColor.Blond, GlassesWorn.No⟩
]

-- Define a function to check if two students share at least one characteristic
def shareCharacteristic (s1 s2 : Student) : Prop :=
  s1.eyeColor = s2.eyeColor ∨ s1.hairColor = s2.hairColor ∨ s1.glassesWorn = s2.glassesWorn

-- Define a function to check if three students are siblings
def areSiblings (s1 s2 s3 : Student) : Prop :=
  shareCharacteristic s1 s2 ∧ shareCharacteristic s2 s3 ∧ shareCharacteristic s1 s3

-- Theorem statement
theorem jim_siblings :
  ∃ (jim sue benjamin : Student),
    jim ∈ students ∧ sue ∈ students ∧ benjamin ∈ students ∧
    jim.name = "Jim" ∧ sue.name = "Sue" ∧ benjamin.name = "Benjamin" ∧
    areSiblings jim sue benjamin ∧
    (∀ (other : Student), other ∈ students → other.name ≠ "Jim" → other.name ≠ "Sue" → other.name ≠ "Benjamin" →
      ¬(areSiblings jim sue other ∨ areSiblings jim benjamin other ∨ areSiblings sue benjamin other)) :=
sorry

end NUMINAMATH_CALUDE_jim_siblings_l1718_171866


namespace NUMINAMATH_CALUDE_l_shapes_on_8x8_board_l1718_171899

/-- Represents a square checkerboard -/
structure Checkerboard :=
  (size : Nat)

/-- Represents an L-shape on the checkerboard -/
structure LShape :=
  (x : Nat) (y : Nat) (orientation : Nat)

/-- The number of different L-shapes on a checkerboard -/
def count_l_shapes (board : Checkerboard) : Nat :=
  sorry

theorem l_shapes_on_8x8_board :
  ∃ (board : Checkerboard),
    board.size = 8 ∧ count_l_shapes board = 196 :=
  sorry

end NUMINAMATH_CALUDE_l_shapes_on_8x8_board_l1718_171899


namespace NUMINAMATH_CALUDE_odd_sum_probability_l1718_171808

structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  even_plus_odd : even + odd = total

def probability_odd_sum (a b : Wheel) : ℚ :=
  (a.even * b.odd + a.odd * b.even : ℚ) / (a.total * b.total : ℚ)

theorem odd_sum_probability 
  (a b : Wheel)
  (ha : a.even = a.odd)
  (hb : b.even = 3 * b.odd)
  (hta : a.total = 8)
  (htb : b.total = 8) :
  probability_odd_sum a b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l1718_171808


namespace NUMINAMATH_CALUDE_standard_deviation_value_l1718_171863

-- Define a symmetric distribution
structure SymmetricDistribution (μ : ℝ) where
  -- The distribution function
  F : ℝ → ℝ
  -- Symmetry property
  symmetric : ∀ x, F (μ + x) + F (μ - x) = 1

-- Define the standard normal distribution function
noncomputable def Φ : ℝ → ℝ := sorry

-- Theorem statement
theorem standard_deviation_value 
  (μ : ℝ) (x : ℝ) (D : SymmetricDistribution μ) :
  (D.F (μ + x) - D.F (μ - x) = 0.68) →
  (D.F (μ + x) = 0.84) →
  (x = 1) :=
sorry

end NUMINAMATH_CALUDE_standard_deviation_value_l1718_171863


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_79_l1718_171874

theorem gcd_of_powers_of_79 : 
  Nat.Prime 79 → Nat.gcd (79^7 + 1) (79^7 + 79^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_79_l1718_171874


namespace NUMINAMATH_CALUDE_watch_price_proof_l1718_171857

/-- The sticker price of the watch in dollars -/
def stickerPrice : ℝ := 250

/-- The price at store X after discounts -/
def priceX (price : ℝ) : ℝ := 0.8 * price - 50

/-- The price at store Y after discount -/
def priceY (price : ℝ) : ℝ := 0.9 * price

theorem watch_price_proof :
  priceY stickerPrice - priceX stickerPrice = 25 :=
sorry

end NUMINAMATH_CALUDE_watch_price_proof_l1718_171857


namespace NUMINAMATH_CALUDE_shoe_repair_cost_l1718_171861

theorem shoe_repair_cost (new_shoe_cost : ℝ) (new_shoe_lifespan : ℝ) (repaired_shoe_lifespan : ℝ) (cost_difference_percentage : ℝ) :
  new_shoe_cost = 30 →
  new_shoe_lifespan = 2 →
  repaired_shoe_lifespan = 1 →
  cost_difference_percentage = 42.857142857142854 →
  ∃ repair_cost : ℝ,
    repair_cost = 10.5 ∧
    (new_shoe_cost / new_shoe_lifespan) = repair_cost * (1 + cost_difference_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_shoe_repair_cost_l1718_171861


namespace NUMINAMATH_CALUDE_tangent_lines_proof_l1718_171859

-- Define the curves
def f (x : ℝ) : ℝ := x^3 + x^2 + 1
def g (x : ℝ) : ℝ := x^2

-- Define the points
def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (3, 5)

-- Define the tangent line equations
def tangent_line1 (x y : ℝ) : Prop := x - y + 2 = 0
def tangent_line2 (x y : ℝ) : Prop := 2*x - y - 1 = 0
def tangent_line3 (x y : ℝ) : Prop := 10*x - y - 25 = 0

theorem tangent_lines_proof :
  (∀ x y : ℝ, y = f x → (x, y) = P1 → tangent_line1 x y) ∧
  (∀ x y : ℝ, y = g x → (x, y) = P2 → (tangent_line2 x y ∨ tangent_line3 x y)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_proof_l1718_171859


namespace NUMINAMATH_CALUDE_evaluate_expression_l1718_171847

theorem evaluate_expression : 
  Real.sqrt ((16^6 + 8^8) / (16^3 + 8^9)) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1718_171847


namespace NUMINAMATH_CALUDE_alice_investment_ratio_l1718_171812

theorem alice_investment_ratio (initial_investment : ℝ) 
  (alice_final : ℝ) (bob_final : ℝ) :
  initial_investment = 2000 →
  bob_final = 6 * initial_investment →
  bob_final = alice_final + 8000 →
  alice_final / initial_investment = 2 :=
by sorry

end NUMINAMATH_CALUDE_alice_investment_ratio_l1718_171812


namespace NUMINAMATH_CALUDE_system_solution_l1718_171829

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 0 ∧ 3*x - 4*y = 5) ↔ (x = 1 ∧ y = -1/2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1718_171829


namespace NUMINAMATH_CALUDE_bad_carrots_count_l1718_171841

theorem bad_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  nancy_carrots = 38 → mom_carrots = 47 → good_carrots = 71 →
  nancy_carrots + mom_carrots - good_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l1718_171841


namespace NUMINAMATH_CALUDE_courier_packages_l1718_171830

theorem courier_packages (x : ℕ) (h1 : x + 2*x = 240) : x = 80 := by
  sorry

end NUMINAMATH_CALUDE_courier_packages_l1718_171830


namespace NUMINAMATH_CALUDE_prob_different_ranks_value_l1718_171840

/-- The number of cards in a standard deck --/
def deck_size : ℕ := 52

/-- The number of ranks in a standard deck --/
def num_ranks : ℕ := 13

/-- The number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- The probability of drawing two cards of different ranks from a standard deck --/
def prob_different_ranks : ℚ :=
  (deck_size * (deck_size - 1) - num_ranks * (num_suits * (num_suits - 1))) /
  (deck_size * (deck_size - 1))

theorem prob_different_ranks_value : prob_different_ranks = 208 / 221 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_ranks_value_l1718_171840


namespace NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l1718_171817

/-- A quadrilateral inscribed in a semi-circle -/
structure InscribedQuadrilateral (r : ℝ) where
  vertices : Fin 4 → ℝ × ℝ
  inside_semicircle : ∀ i, (vertices i).1^2 + (vertices i).2^2 ≤ r^2 ∧ (vertices i).2 ≥ 0

/-- The area of a quadrilateral -/
def area (q : InscribedQuadrilateral r) : ℝ :=
  sorry

/-- The shape of a half regular hexagon -/
def half_regular_hexagon (r : ℝ) : InscribedQuadrilateral r :=
  sorry

theorem max_area_inscribed_quadrilateral (r : ℝ) (hr : r > 0) :
  (∀ q : InscribedQuadrilateral r, area q ≤ (3 * Real.sqrt 3 / 4) * r^2) ∧
  area (half_regular_hexagon r) = (3 * Real.sqrt 3 / 4) * r^2 :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l1718_171817


namespace NUMINAMATH_CALUDE_car_average_speed_l1718_171843

/-- The average speed of a car traveling 60 km in the first hour and 30 km in the second hour is 45 km/h. -/
theorem car_average_speed : 
  let speed1 : ℝ := 60 -- Speed in the first hour (km/h)
  let speed2 : ℝ := 30 -- Speed in the second hour (km/h)
  let time : ℝ := 2 -- Total time (hours)
  let total_distance : ℝ := speed1 + speed2 -- Total distance (km)
  let average_speed : ℝ := total_distance / time -- Average speed (km/h)
  average_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l1718_171843


namespace NUMINAMATH_CALUDE_checkerboard_ratio_l1718_171834

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The sum of squares from 1 to n -/
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles on an n x n checkerboard -/
def num_rectangles (n : ℕ) : ℕ := (choose_2 (n + 1)) ^ 2

/-- The number of squares on an n x n checkerboard -/
def num_squares (n : ℕ) : ℕ := sum_squares n

theorem checkerboard_ratio :
  (num_squares 9 : ℚ) / (num_rectangles 9 : ℚ) = 19 / 135 := by sorry

end NUMINAMATH_CALUDE_checkerboard_ratio_l1718_171834


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1718_171802

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2/x₀ + 1/y₀ = 1 ∧ x₀ + 2*y₀ = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1718_171802


namespace NUMINAMATH_CALUDE_blood_expiry_time_l1718_171893

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a date -/
structure Date where
  month : ℕ
  day : ℕ
  year : ℕ

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : Time

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def addSeconds (dt : DateTime) (seconds : ℕ) : DateTime :=
  sorry -- Implementation not required for the statement

theorem blood_expiry_time 
  (donation_time : DateTime)
  (expiry_seconds : ℕ)
  (h_donation_time : donation_time = ⟨⟨1, 1, 2023⟩, ⟨8, 0, sorry, sorry⟩⟩)
  (h_expiry_seconds : expiry_seconds = factorial 8) :
  addSeconds donation_time expiry_seconds = ⟨⟨1, 1, 2023⟩, ⟨19, 12, sorry, sorry⟩⟩ :=
sorry

end NUMINAMATH_CALUDE_blood_expiry_time_l1718_171893


namespace NUMINAMATH_CALUDE_factorization_condition_l1718_171801

-- Define the polynomial
def polynomial (m : ℤ) (x y : ℤ) : ℤ := x^2 + 5*x*y + 2*x + m*y - 2*m

-- Define what it means for a polynomial to have two linear factors with integer coefficients
def has_two_linear_factors (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), 
    ∀ (x y : ℤ), polynomial m x y = (a*x + b*y + c) * (d*x + e*y + f)

-- State the theorem
theorem factorization_condition (m : ℤ) : 
  has_two_linear_factors m ↔ (m = 0 ∨ m = 10) := by sorry

end NUMINAMATH_CALUDE_factorization_condition_l1718_171801


namespace NUMINAMATH_CALUDE_optimal_advertising_plan_l1718_171820

/-- Represents the advertising plan for a company --/
structure AdvertisingPlan where
  timeA : ℝ  -- Time allocated to TV station A in minutes
  timeB : ℝ  -- Time allocated to TV station B in minutes

/-- Calculates the total advertising time for a given plan --/
def totalTime (plan : AdvertisingPlan) : ℝ :=
  plan.timeA + plan.timeB

/-- Calculates the total advertising cost for a given plan --/
def totalCost (plan : AdvertisingPlan) : ℝ :=
  500 * plan.timeA + 200 * plan.timeB

/-- Calculates the total revenue for a given plan --/
def totalRevenue (plan : AdvertisingPlan) : ℝ :=
  0.3 * plan.timeA + 0.2 * plan.timeB

/-- Theorem stating the optimal advertising plan and maximum revenue --/
theorem optimal_advertising_plan :
  ∃ (plan : AdvertisingPlan),
    totalTime plan ≤ 300 ∧
    totalCost plan ≤ 90000 ∧
    plan.timeA = 100 ∧
    plan.timeB = 200 ∧
    totalRevenue plan = 70 ∧
    ∀ (other : AdvertisingPlan),
      totalTime other ≤ 300 →
      totalCost other ≤ 90000 →
      totalRevenue other ≤ totalRevenue plan :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_advertising_plan_l1718_171820


namespace NUMINAMATH_CALUDE_intersection_empty_range_l1718_171852

theorem intersection_empty_range (a : ℝ) : 
  let A := {x : ℝ | |x - a| < 1}
  let B := {x : ℝ | 1 < x ∧ x < 5}
  (A ∩ B = ∅) ↔ (a ≤ 0 ∨ a ≥ 6) := by sorry

end NUMINAMATH_CALUDE_intersection_empty_range_l1718_171852


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1718_171869

theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 1 / 6 →
  combined_fill_time = 30 →
  1 / fill_time - empty_rate = 1 / combined_fill_time →
  fill_time = 5 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1718_171869


namespace NUMINAMATH_CALUDE_shortest_path_length_l1718_171810

/-- The length of the shortest path from (0,0) to (20,21) avoiding a circle --/
theorem shortest_path_length (start : ℝ × ℝ) (end_point : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : 
  start = (0, 0) →
  end_point = (20, 21) →
  center = (10, 10.5) →
  radius = 6 →
  ∃ (path_length : ℝ),
    path_length = 26.4 + 2 * Real.pi ∧
    ∀ (other_path : ℝ),
      (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 ≥ radius^2 → 
        (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (x, y) = (start.1 + t * (end_point.1 - start.1), start.2 + t * (end_point.2 - start.2)))) →
      other_path ≥ path_length :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_length_l1718_171810


namespace NUMINAMATH_CALUDE_circle_locus_is_spherical_triangle_l1718_171875

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  vertex : Point3D
  isRightAngled : Bool

/-- The locus of circle centers touching the faces of a right-angled trihedral angle -/
def circleLocus (t : TrihedralAngle) (r : ℝ) : Set Point3D :=
  {p : Point3D | ∃ (c : Circle3D), c.radius = r ∧ 
    c.center = p ∧ 
    (c.center.x ≤ r ∧ c.center.y ≤ r ∧ c.center.z ≤ r) ∧
    (c.center.x ≥ 0 ∧ c.center.y ≥ 0 ∧ c.center.z ≥ 0) ∧
    (c.center.x ^ 2 + c.center.y ^ 2 + c.center.z ^ 2 = 2 * r ^ 2)}

theorem circle_locus_is_spherical_triangle (t : TrihedralAngle) (r : ℝ) 
  (h : t.isRightAngled = true) :
  circleLocus t r = {p : Point3D | 
    p.x ^ 2 + p.y ^ 2 + p.z ^ 2 = 2 * r ^ 2 ∧
    p.x ≤ r ∧ p.y ≤ r ∧ p.z ≤ r ∧
    p.x ≥ 0 ∧ p.y ≥ 0 ∧ p.z ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_circle_locus_is_spherical_triangle_l1718_171875


namespace NUMINAMATH_CALUDE_rectangle_division_l1718_171876

theorem rectangle_division (w₁ h₁ w₂ h₂ : ℝ) :
  w₁ > 0 ∧ h₁ > 0 ∧ w₂ > 0 ∧ h₂ > 0 →
  w₁ * h₁ = 6 →
  w₂ * h₁ = 15 →
  w₂ * h₂ = 25 →
  w₁ * h₂ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l1718_171876


namespace NUMINAMATH_CALUDE_divisibility_conditions_solutions_l1718_171816

theorem divisibility_conditions_solutions (a b : ℕ+) : 
  (a ∣ b^2) → (b ∣ a^2) → ((a + 1) ∣ (b^2 + 1)) → 
  (∃ q : ℕ+, (a = q^2 ∧ b = q) ∨ 
             (a = q^2 ∧ b = q^3) ∨ 
             (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_conditions_solutions_l1718_171816


namespace NUMINAMATH_CALUDE_tan_120_degrees_l1718_171827

theorem tan_120_degrees : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_120_degrees_l1718_171827


namespace NUMINAMATH_CALUDE_fencing_calculation_l1718_171886

/-- The total fencing length for a square playground and a rectangular garden -/
def total_fencing (playground_side : ℝ) (garden_length garden_width : ℝ) : ℝ :=
  4 * playground_side + 2 * (garden_length + garden_width)

/-- Theorem: The total fencing for a playground with side 27 yards and a garden of 12 by 9 yards is 150 yards -/
theorem fencing_calculation :
  total_fencing 27 12 9 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l1718_171886


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1718_171814

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let d : ℝ := 10  -- diameter in meters
  let r : ℝ := d / 2  -- radius in meters
  let area : ℝ := π * r^2  -- area formula
  area = 25 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1718_171814


namespace NUMINAMATH_CALUDE_sequence_equality_l1718_171846

/-- Given a sequence a₀, a₁, a₂, ..., prove that aₙ = 10ⁿ for all natural numbers n,
    if the following equation holds for all real t:
    ∑_{n=0}^∞ aₙ * t^n / n! = (∑_{n=0}^∞ 2^n * t^n / n!)² * (∑_{n=0}^∞ 3^n * t^n / n!)² -/
theorem sequence_equality (a : ℕ → ℝ) :
  (∀ t : ℝ, ∑' n, a n * t^n / n.factorial = (∑' n, 2^n * t^n / n.factorial)^2 * (∑' n, 3^n * t^n / n.factorial)^2) →
  ∀ n : ℕ, a n = 10^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_equality_l1718_171846


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1718_171892

theorem right_triangle_shorter_leg (a b c m : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0 →  -- Positive lengths
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = 2 * a →  -- One leg is twice the other
  m = 15 →  -- Median to hypotenuse is 15
  m^2 = (c^2) / 4 + (a^2 + b^2) / 4 →  -- Median formula
  a = 6 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1718_171892


namespace NUMINAMATH_CALUDE_large_sphere_radius_twelve_small_to_one_large_l1718_171836

/-- The radius of a single sphere made from the same amount of material as multiple smaller spheres -/
theorem large_sphere_radius (n : ℕ) (r : ℝ) (h : n > 0) :
  (((n : ℝ) * (4 / 3 * Real.pi * r^3)) / (4 / 3 * Real.pi))^(1/3) = n^(1/3) * r :=
by sorry

/-- The radius of a single sphere made from the same amount of material as 12 spheres of radius 0.5 -/
theorem twelve_small_to_one_large :
  (((12 : ℝ) * (4 / 3 * Real.pi * (1/2)^3)) / (4 / 3 * Real.pi))^(1/3) = (3/2)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_large_sphere_radius_twelve_small_to_one_large_l1718_171836


namespace NUMINAMATH_CALUDE_tenths_vs_thousandths_l1718_171821

def number : ℚ := 85247.2048

theorem tenths_vs_thousandths :
  (number - number.floor) * 10 % 1 * 10 = 
  100 * ((number - number.floor) * 1000 % 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_tenths_vs_thousandths_l1718_171821


namespace NUMINAMATH_CALUDE_bread_cost_l1718_171825

theorem bread_cost (butter_cost juice_cost bread_cost total_spent : ℝ) : 
  butter_cost = 3 →
  juice_cost = 2 * bread_cost →
  total_spent = 9 →
  bread_cost + butter_cost + juice_cost = total_spent →
  bread_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_bread_cost_l1718_171825


namespace NUMINAMATH_CALUDE_minimum_packaging_volume_l1718_171845

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the packaging problem parameters -/
structure PackagingProblem where
  boxDimensions : BoxDimensions
  costPerBox : ℝ
  minTotalCost : ℝ

theorem minimum_packaging_volume (p : PackagingProblem) 
  (h1 : p.boxDimensions.length = 20)
  (h2 : p.boxDimensions.width = 20)
  (h3 : p.boxDimensions.height = 12)
  (h4 : p.costPerBox = 0.4)
  (h5 : p.minTotalCost = 200) :
  (p.minTotalCost / p.costPerBox) * boxVolume p.boxDimensions = 2400000 := by
  sorry

#check minimum_packaging_volume

end NUMINAMATH_CALUDE_minimum_packaging_volume_l1718_171845


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1718_171819

/-- Given a right triangle with sides 5, 12, and 13 (13 being the hypotenuse),
    a square of side length x inscribed with one side along the leg of length 12,
    and another square of side length y inscribed with one side along the hypotenuse,
    the ratio of x to y is 12/13. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 → y > 0 →
  x^2 + x^2 = 5 * x →
  y^2 + y^2 = 13 * y →
  x / y = 12 / 13 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1718_171819


namespace NUMINAMATH_CALUDE_student_group_composition_l1718_171826

theorem student_group_composition (total : Nat) (boys : Nat) (ways : Nat) : 
  total = 8 → 
  ways = 180 → 
  (boys * (boys - 1) / 2) * (total - boys) * 6 = ways → 
  (boys = 5 ∨ boys = 6) := by
  sorry

end NUMINAMATH_CALUDE_student_group_composition_l1718_171826


namespace NUMINAMATH_CALUDE_simplest_form_count_l1718_171811

-- Define the fractions
def fraction1 (a b : ℚ) : ℚ := b / (8 * a)
def fraction2 (a b : ℚ) : ℚ := (a + b) / (a - b)
def fraction3 (x y : ℚ) : ℚ := (x - y) / (x^2 - y^2)
def fraction4 (x y : ℚ) : ℚ := (x - y) / (x^2 + 2*x*y + y^2)

-- Define a function to check if a fraction is in simplest form
def isSimplestForm (f : ℚ → ℚ → ℚ) : Prop := 
  ∀ a b, a ≠ 0 → b ≠ 0 → (∃ c, f a b = c) → 
    ¬∃ d e, d ≠ 0 ∧ e ≠ 0 ∧ f (a*d) (b*e) = f a b

-- Theorem statement
theorem simplest_form_count : 
  (isSimplestForm fraction1) ∧ 
  (isSimplestForm fraction2) ∧ 
  ¬(isSimplestForm fraction3) ∧
  (isSimplestForm fraction4) := by sorry

end NUMINAMATH_CALUDE_simplest_form_count_l1718_171811


namespace NUMINAMATH_CALUDE_hiker_catches_cyclist_l1718_171856

/-- Proves that a hiker catches up to a cyclist in 30 minutes under specific conditions -/
theorem hiker_catches_cyclist (hiker_speed cyclist_speed : ℝ) (cyclist_travel_time : ℝ) : 
  hiker_speed = 4 →
  cyclist_speed = 24 →
  cyclist_travel_time = 5 / 60 →
  let cyclist_distance := cyclist_speed * cyclist_travel_time
  let catchup_time := cyclist_distance / hiker_speed
  catchup_time * 60 = 30 := by sorry

end NUMINAMATH_CALUDE_hiker_catches_cyclist_l1718_171856


namespace NUMINAMATH_CALUDE_difference_x_y_l1718_171824

theorem difference_x_y : ∀ (x y : ℤ), x + y = 250 → y = 225 → |x - y| = 200 := by
  sorry

end NUMINAMATH_CALUDE_difference_x_y_l1718_171824


namespace NUMINAMATH_CALUDE_sequence_problem_l1718_171862

def S (n : ℕ) (k : ℕ) : ℚ := -1/2 * n^2 + k*n

theorem sequence_problem (k : ℕ) (h1 : k > 0) 
  (h2 : ∀ n : ℕ, S n k ≤ 8) 
  (h3 : ∃ n : ℕ, S n k = 8) :
  k = 4 ∧ ∀ n : ℕ, n ≥ 1 → ((-1/2 : ℚ) * n^2 + 4*n) - ((-1/2 : ℚ) * (n-1)^2 + 4*(n-1)) = 9/2 - n :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1718_171862


namespace NUMINAMATH_CALUDE_inequality_proof_l1718_171896

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1718_171896


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1718_171895

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a - Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1718_171895


namespace NUMINAMATH_CALUDE_dormitory_allocation_l1718_171889

/-- The number of ways to assign n students to two dormitories with at least k students in each -/
def allocation_schemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange r items from n items -/
def arrange (n : ℕ) (r : ℕ) : ℕ := sorry

theorem dormitory_allocation :
  allocation_schemes 7 2 = 112 :=
by
  sorry

#check dormitory_allocation

end NUMINAMATH_CALUDE_dormitory_allocation_l1718_171889


namespace NUMINAMATH_CALUDE_scientific_notation_2150000_l1718_171854

theorem scientific_notation_2150000 : 
  ∃ (a : ℝ) (n : ℤ), 2150000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_2150000_l1718_171854


namespace NUMINAMATH_CALUDE_sequence_proof_l1718_171850

theorem sequence_proof (a : Fin 8 → ℕ) 
  (h1 : ∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 100)
  (h2 : a 0 = 20)
  (h3 : a 7 = 16) :
  a = ![20, 16, 64, 20, 16, 64, 20, 16] := by
sorry

end NUMINAMATH_CALUDE_sequence_proof_l1718_171850


namespace NUMINAMATH_CALUDE_evaluate_expression_l1718_171865

theorem evaluate_expression : (4^4 - 4*(4-2)^4)^4 = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1718_171865


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1718_171839

theorem quadratic_root_problem (m : ℝ) : 
  ((-5)^2 + m*(-5) - 10 = 0) → (2^2 + m*2 - 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1718_171839


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l1718_171884

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l1718_171884


namespace NUMINAMATH_CALUDE_astroid_length_l1718_171867

/-- The astroid curve -/
def astroid (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^(2/3) + p.2^(2/3) = a^(2/3)) ∧ a > 0}

/-- The length of a curve -/
noncomputable def curveLength (C : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The length of the astroid x^(2/3) + y^(2/3) = a^(2/3) is 6a -/
theorem astroid_length (a : ℝ) (h : a > 0) : 
  curveLength (astroid a) = 6 * a := by sorry

end NUMINAMATH_CALUDE_astroid_length_l1718_171867


namespace NUMINAMATH_CALUDE_even_function_m_value_l1718_171828

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 12)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l1718_171828


namespace NUMINAMATH_CALUDE_equalize_foma_ierema_l1718_171880

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equalize_foma_ierema (w : MerchantWealth) 
  (h : problem_conditions w) : 
  ∃ (x : ℕ), w.foma - x = w.ierema + x ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_equalize_foma_ierema_l1718_171880


namespace NUMINAMATH_CALUDE_different_color_probability_l1718_171803

/-- The probability of drawing two balls of different colors from a box -/
theorem different_color_probability (total : ℕ) (red : ℕ) (yellow : ℕ) : 
  total = red + yellow →
  red = 3 →
  yellow = 2 →
  (red.choose 1 * yellow.choose 1 : ℚ) / total.choose 2 = 3/5 :=
sorry

end NUMINAMATH_CALUDE_different_color_probability_l1718_171803


namespace NUMINAMATH_CALUDE_givenEquationIsQuadratic_l1718_171851

/-- Represents a polynomial equation with one variable -/
structure PolynomialEquation :=
  (a b c : ℝ)

/-- Defines a quadratic equation with one variable -/
def IsQuadraticOneVariable (eq : PolynomialEquation) : Prop :=
  eq.a ≠ 0

/-- The specific equation we're considering -/
def givenEquation : PolynomialEquation :=
  { a := 1, b := 1, c := 3 }

/-- Theorem stating that the given equation is a quadratic equation with one variable -/
theorem givenEquationIsQuadratic : IsQuadraticOneVariable givenEquation := by
  sorry


end NUMINAMATH_CALUDE_givenEquationIsQuadratic_l1718_171851


namespace NUMINAMATH_CALUDE_video_game_sales_theorem_l1718_171894

/-- Given a total number of video games, number of non-working games, and a price per working game,
    calculate the total money that can be earned by selling the working games. -/
def total_money_earned (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Theorem stating that given 10 total games, 8 non-working games, and a price of $6 per working game,
    the total money earned is $12. -/
theorem video_game_sales_theorem :
  total_money_earned 10 8 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_theorem_l1718_171894


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1718_171897

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1718_171897


namespace NUMINAMATH_CALUDE_project_duration_calculation_l1718_171833

/-- The number of weeks a project lasts based on breakfast expenses -/
def project_duration (people : ℕ) (days_per_week : ℕ) (meal_cost : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (people * days_per_week * meal_cost)

theorem project_duration_calculation :
  let people : ℕ := 4
  let days_per_week : ℕ := 5
  let meal_cost : ℚ := 4
  let total_spent : ℚ := 1280
  project_duration people days_per_week meal_cost total_spent = 16 := by
  sorry

end NUMINAMATH_CALUDE_project_duration_calculation_l1718_171833
