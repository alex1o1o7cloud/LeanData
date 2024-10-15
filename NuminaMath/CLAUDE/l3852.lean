import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_equation_l3852_385224

/-- The standard equation of an ellipse with foci on the coordinate axes and passing through points A(√3, -2) and B(-2√3, 1) is x²/15 + y²/5 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 15 + y^2 / 5 = 1)) ∧
  (x^2 / 15 + y^2 / 5 = 1 → x = Real.sqrt 3 ∧ y = -2 ∨ x = -2 * Real.sqrt 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3852_385224


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l3852_385240

theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℝ)
  (excluded_count : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 30)
  (h2 : all_average = 80)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 92)
  : (total_students * all_average - (total_students - excluded_count) * remaining_average) / excluded_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l3852_385240


namespace NUMINAMATH_CALUDE_parallel_vectors_l3852_385211

def a : ℝ × ℝ := (-1, 1)
def b (m : ℝ) : ℝ × ℝ := (3, m)

theorem parallel_vectors (m : ℝ) : 
  (∃ (k : ℝ), a = k • (a + b m)) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3852_385211


namespace NUMINAMATH_CALUDE_extreme_value_M_inequality_condition_l3852_385256

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := x + 1/x + a * (1/x)

noncomputable def M (x : ℝ) : ℝ := x - log x + 2/x

theorem extreme_value_M :
  (∀ x > 0, M x ≥ 3 - log 2) ∧
  (M 2 = 3 - log 2) ∧
  (∀ b : ℝ, ∃ x > 0, M x > b) :=
sorry

theorem inequality_condition (m : ℝ) :
  (∀ x > 0, 1 / (x + 1/x) ≤ 1 / (2 + m * (log x)^2)) ↔ 0 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_M_inequality_condition_l3852_385256


namespace NUMINAMATH_CALUDE_kolya_purchase_l3852_385264

/-- Represents the price of an item in kopecks -/
def ItemPrice (rubles : ℕ) : ℕ := 100 * rubles + 99

/-- Represents Kolya's total purchase in kopecks -/
def TotalPurchase : ℕ := 200 * 100 + 83

/-- Predicate to check if a given number of items satisfies the purchase conditions -/
def ValidPurchase (n : ℕ) : Prop :=
  ∃ (rubles : ℕ), n * ItemPrice rubles = TotalPurchase

theorem kolya_purchase :
  {n : ℕ | ValidPurchase n} = {17, 117} := by sorry

end NUMINAMATH_CALUDE_kolya_purchase_l3852_385264


namespace NUMINAMATH_CALUDE_complex_number_system_l3852_385267

theorem complex_number_system (x y z : ℂ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 3)
  (h3 : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_system_l3852_385267


namespace NUMINAMATH_CALUDE_solution_set_for_m_equals_one_m_range_for_inequality_l3852_385261

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Theorem 1
theorem solution_set_for_m_equals_one (x : ℝ) :
  f 1 x ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1 := by sorry

-- Theorem 2
theorem m_range_for_inequality (m : ℝ) (h1 : m > 0) :
  (∀ x ∈ Set.Icc m (2*m^2), (1/2) * f m x ≤ |x + 1|) →
  1/2 < m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_m_equals_one_m_range_for_inequality_l3852_385261


namespace NUMINAMATH_CALUDE_apple_group_addition_l3852_385231

/-- Given a basket of apples divided among a group, prove the number of people who joined --/
theorem apple_group_addition (total_apples : ℕ) (original_per_person : ℕ) (new_per_person : ℕ) :
  total_apples = 1430 →
  original_per_person = 22 →
  new_per_person = 13 →
  ∃ (original_group : ℕ) (joined_group : ℕ),
    original_group * original_per_person = total_apples ∧
    (original_group + joined_group) * new_per_person = total_apples ∧
    joined_group = 45 := by
  sorry


end NUMINAMATH_CALUDE_apple_group_addition_l3852_385231


namespace NUMINAMATH_CALUDE_lawrence_county_camp_kids_l3852_385257

/-- The number of kids going to camp during summer break in Lawrence county --/
def kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) : ℕ :=
  total_kids - kids_at_home

/-- Theorem stating the number of kids going to camp in Lawrence county --/
theorem lawrence_county_camp_kids :
  kids_at_camp 313473 274865 = 38608 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_kids_l3852_385257


namespace NUMINAMATH_CALUDE_largest_value_l3852_385201

theorem largest_value (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : x + y = 1) :
  (1/2 < x^2 + y^2) ∧ (2*x*y < x^2 + y^2) ∧ (x < x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3852_385201


namespace NUMINAMATH_CALUDE_length_eb_is_two_l3852_385292

/-- An equilateral triangle with points on its sides -/
structure TriangleWithPoints where
  -- The side length of the equilateral triangle
  side : ℝ
  -- Lengths of segments
  ad : ℝ
  de : ℝ
  ef : ℝ
  fa : ℝ
  -- Conditions
  equilateral : side > 0
  d_on_ab : ad ≤ side
  e_on_bc : de ≤ side
  f_on_ca : fa ≤ side
  ad_value : ad = 4
  de_value : de = 8
  ef_value : ef = 10
  fa_value : fa = 6

/-- The length of segment EB in the triangle -/
def length_eb (t : TriangleWithPoints) : ℝ := 2

/-- Theorem: The length of EB is 2 -/
theorem length_eb_is_two (t : TriangleWithPoints) : length_eb t = 2 := by
  sorry

end NUMINAMATH_CALUDE_length_eb_is_two_l3852_385292


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l3852_385223

theorem power_function_not_through_origin (m : ℝ) : 
  (m = 1 ∨ m = 2) → 
  ∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^((m^2 - m - 2)/2) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l3852_385223


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3852_385244

theorem sum_of_x_solutions_is_zero (x y : ℝ) : 
  y = 9 → x^2 + y^2 = 169 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ x₁^2 + y^2 = 169 ∧ x₂^2 + y^2 = 169 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3852_385244


namespace NUMINAMATH_CALUDE_sally_lemonade_sales_l3852_385280

/-- Calculates the total number of lemonade cups sold over two weeks -/
def total_lemonade_cups (last_week : ℕ) (percent_increase : ℕ) : ℕ :=
  let this_week := last_week + (last_week * percent_increase) / 100
  last_week + this_week

/-- Proves that given the conditions, Sally sold 46 cups of lemonade in total -/
theorem sally_lemonade_sales : total_lemonade_cups 20 30 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sally_lemonade_sales_l3852_385280


namespace NUMINAMATH_CALUDE_parallel_line_plane_condition_l3852_385270

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_plane_condition 
  (α β : Plane) (m : Line) :
  parallel_planes α β → 
  ¬subset_line_plane m β → 
  parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_condition_l3852_385270


namespace NUMINAMATH_CALUDE_concatenated_integers_divisible_by_55_l3852_385232

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

theorem concatenated_integers_divisible_by_55 :
  ∃ k : ℕ, concatenate_integers 55 = 55 * k := by
  sorry

end NUMINAMATH_CALUDE_concatenated_integers_divisible_by_55_l3852_385232


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_is_126_l3852_385266

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each fourth-grade classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The number of fourth-grade classrooms -/
def number_of_classrooms : ℕ := 6

/-- The difference between the total number of students and the total number of guinea pigs -/
def student_guinea_pig_difference : ℕ :=
  (students_per_classroom * number_of_classrooms) - (guinea_pigs_per_classroom * number_of_classrooms)

theorem student_guinea_pig_difference_is_126 :
  student_guinea_pig_difference = 126 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_is_126_l3852_385266


namespace NUMINAMATH_CALUDE_sylvia_earnings_l3852_385275

-- Define the work durations
def monday_hours : ℚ := 5/2
def tuesday_minutes : ℕ := 40
def wednesday_start : ℕ := 9 * 60 + 15  -- 9:15 AM in minutes
def wednesday_end : ℕ := 11 * 60 + 50   -- 11:50 AM in minutes
def thursday_minutes : ℕ := 45

-- Define the hourly rate
def hourly_rate : ℚ := 4

-- Define the function to calculate total earnings
def total_earnings : ℚ :=
  let total_minutes : ℚ := 
    monday_hours * 60 + 
    tuesday_minutes + 
    (wednesday_end - wednesday_start) + 
    thursday_minutes
  let total_hours : ℚ := total_minutes / 60
  total_hours * hourly_rate

-- Theorem statement
theorem sylvia_earnings : total_earnings = 26 := by
  sorry

end NUMINAMATH_CALUDE_sylvia_earnings_l3852_385275


namespace NUMINAMATH_CALUDE_shirt_price_proof_l3852_385218

-- Define the original price
def original_price : ℝ := 32

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the final price
def final_price : ℝ := 18

-- Theorem statement
theorem shirt_price_proof :
  (1 - discount_rate) * (1 - discount_rate) * original_price = final_price :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l3852_385218


namespace NUMINAMATH_CALUDE_women_in_room_l3852_385279

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  24 = 2 * (initial_women - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l3852_385279


namespace NUMINAMATH_CALUDE_lindas_savings_l3852_385206

/-- Given that Linda spent 3/4 of her savings on furniture and the rest on a TV costing $500,
    prove that her original savings were $2000. -/
theorem lindas_savings (savings : ℝ) : 
  (3/4 : ℝ) * savings + 500 = savings → savings = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l3852_385206


namespace NUMINAMATH_CALUDE_company_employee_increase_l3852_385248

theorem company_employee_increase (january_employees : ℝ) (increase_percentage : ℝ) :
  january_employees = 434.7826086956522 →
  increase_percentage = 15 →
  january_employees * (1 + increase_percentage / 100) = 500 := by
sorry

end NUMINAMATH_CALUDE_company_employee_increase_l3852_385248


namespace NUMINAMATH_CALUDE_students_left_early_l3852_385273

theorem students_left_early (original_groups : ℕ) (students_per_group : ℕ) (remaining_students : ℕ)
  (h1 : original_groups = 3)
  (h2 : students_per_group = 8)
  (h3 : remaining_students = 22) :
  original_groups * students_per_group - remaining_students = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_left_early_l3852_385273


namespace NUMINAMATH_CALUDE_profit_per_meter_l3852_385251

/-- Calculate the profit per meter of cloth -/
theorem profit_per_meter (meters_sold : ℕ) (selling_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_sold = 45 →
  selling_price = 4500 →
  cost_price_per_meter = 88 →
  (selling_price - meters_sold * cost_price_per_meter) / meters_sold = 12 :=
by sorry

end NUMINAMATH_CALUDE_profit_per_meter_l3852_385251


namespace NUMINAMATH_CALUDE_david_average_marks_l3852_385233

def david_marks : List ℕ := [86, 85, 82, 87, 85]

theorem david_average_marks :
  (List.sum david_marks) / (List.length david_marks) = 85 := by
  sorry

end NUMINAMATH_CALUDE_david_average_marks_l3852_385233


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_eq_neg_two_monotone_increasing_implies_a_nonneg_l3852_385242

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

theorem local_minimum_implies_a_eq_neg_two (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) → a = -2 := by sorry

theorem monotone_increasing_implies_a_nonneg (a : ℝ) :
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a x < f a y) → a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_eq_neg_two_monotone_increasing_implies_a_nonneg_l3852_385242


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l3852_385245

/-- Represents a wheel with spokes -/
structure Wheel where
  spokes : ℕ

/-- Represents a configuration of wheels -/
structure WheelConfiguration where
  wheels : List Wheel
  total_spokes : ℕ
  max_visible_spokes : ℕ

/-- Checks if a configuration is valid based on the problem conditions -/
def is_valid_configuration (config : WheelConfiguration) : Prop :=
  config.total_spokes ≥ 7 ∧
  config.max_visible_spokes ≤ 3 ∧
  (∀ w ∈ config.wheels, w.spokes ≤ config.max_visible_spokes)

theorem wheel_configuration_theorem :
  ∃ (config_three : WheelConfiguration),
    config_three.wheels.length = 3 ∧
    is_valid_configuration config_three ∧
  ¬∃ (config_two : WheelConfiguration),
    config_two.wheels.length = 2 ∧
    is_valid_configuration config_two :=
sorry

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l3852_385245


namespace NUMINAMATH_CALUDE_passing_percentage_is_fifty_l3852_385263

def student_score : ℕ := 200
def failed_by : ℕ := 20
def max_marks : ℕ := 440

def passing_score : ℕ := student_score + failed_by

def passing_percentage : ℚ := (passing_score : ℚ) / (max_marks : ℚ) * 100

theorem passing_percentage_is_fifty : passing_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_fifty_l3852_385263


namespace NUMINAMATH_CALUDE_smallest_x_value_l3852_385281

theorem smallest_x_value (x y : ℝ) : 
  4 ≤ x ∧ x < 6 →
  6 < y ∧ y < 10 →
  (∃ (n : ℤ), n = ⌊y - x⌋ ∧ n ≤ 5 ∧ ∀ (m : ℤ), m = ⌊y - x⌋ → m ≤ n) →
  x ≥ 4 ∧ ∀ (z : ℝ), (4 ≤ z ∧ z < 6 ∧ 
    (∃ (w : ℝ), 6 < w ∧ w < 10 ∧ 
      (∃ (n : ℤ), n = ⌊w - z⌋ ∧ n ≤ 5 ∧ ∀ (m : ℤ), m = ⌊w - z⌋ → m ≤ n))) →
    z ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3852_385281


namespace NUMINAMATH_CALUDE_gloves_with_pair_count_l3852_385253

-- Define the number of glove pairs
def num_pairs : ℕ := 4

-- Define the total number of gloves
def total_gloves : ℕ := 2 * num_pairs

-- Define the number of gloves to pick
def gloves_to_pick : ℕ := 4

-- Theorem statement
theorem gloves_with_pair_count :
  (Nat.choose total_gloves gloves_to_pick) - (2^num_pairs) = 54 := by
  sorry

end NUMINAMATH_CALUDE_gloves_with_pair_count_l3852_385253


namespace NUMINAMATH_CALUDE_prime_iff_totient_and_divisor_sum_condition_l3852_385295

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Divisor sum function -/
def σ : ℕ → ℕ := sorry

/-- An integer n ≥ 2 is prime if and only if φ(n) divides (n - 1) and (n + 1) divides σ(n) -/
theorem prime_iff_totient_and_divisor_sum_condition (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ (φ n ∣ n - 1) ∧ (n + 1 ∣ σ n) := by sorry

end NUMINAMATH_CALUDE_prime_iff_totient_and_divisor_sum_condition_l3852_385295


namespace NUMINAMATH_CALUDE_max_digit_sum_is_37_l3852_385239

/-- Represents a two-digit display --/
structure TwoDigitDisplay where
  tens : Nat
  ones : Nat
  valid : tens ≤ 9 ∧ ones ≤ 9

/-- Represents a time display in 12-hour format --/
structure TimeDisplay where
  hours : TwoDigitDisplay
  minutes : TwoDigitDisplay
  seconds : TwoDigitDisplay
  valid_hours : hours.tens * 10 + hours.ones ≥ 1 ∧ hours.tens * 10 + hours.ones ≤ 12
  valid_minutes : minutes.tens * 10 + minutes.ones ≤ 59
  valid_seconds : seconds.tens * 10 + seconds.ones ≤ 59

/-- Calculates the sum of digits in a TwoDigitDisplay --/
def digitSum (d : TwoDigitDisplay) : Nat :=
  d.tens + d.ones

/-- Calculates the total sum of digits in a TimeDisplay --/
def totalDigitSum (t : TimeDisplay) : Nat :=
  digitSum t.hours + digitSum t.minutes + digitSum t.seconds

/-- The maximum possible sum of digits in a 12-hour format digital watch display --/
def maxDigitSum : Nat := 37

/-- Theorem: The maximum sum of digits in a 12-hour format digital watch display is 37 --/
theorem max_digit_sum_is_37 :
  ∀ t : TimeDisplay, totalDigitSum t ≤ maxDigitSum :=
by
  sorry  -- The proof would go here

#check max_digit_sum_is_37

end NUMINAMATH_CALUDE_max_digit_sum_is_37_l3852_385239


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3852_385216

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3852_385216


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3852_385247

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 1, 1]

theorem inverse_of_B_cubed :
  let B_inv := !![3, -1; 1, 1]
  (B_inv^3)⁻¹ = !![20, -12; 12, -4] := by sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3852_385247


namespace NUMINAMATH_CALUDE_sons_age_l3852_385236

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 16 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 14 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3852_385236


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3852_385213

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f x - y * f y = (x - y) * f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax + b for some real a and b -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3852_385213


namespace NUMINAMATH_CALUDE_vorontsova_dashkova_lifespan_l3852_385212

/-- Represents a person's lifespan across two centuries -/
structure Lifespan where
  total : ℕ
  diff_18th_19th : ℕ
  years_19th : ℕ
  birth_year : ℕ
  death_year : ℕ

/-- Theorem about E.P. Vorontsova-Dashkova's lifespan -/
theorem vorontsova_dashkova_lifespan :
  ∃ (l : Lifespan),
    l.total = 66 ∧
    l.diff_18th_19th = 46 ∧
    l.years_19th = 10 ∧
    l.birth_year = 1744 ∧
    l.death_year = 1810 ∧
    l.total = l.years_19th + (l.years_19th + l.diff_18th_19th) ∧
    l.birth_year + l.total = l.death_year ∧
    l.birth_year + (l.total - l.years_19th) = 1800 :=
by
  sorry


end NUMINAMATH_CALUDE_vorontsova_dashkova_lifespan_l3852_385212


namespace NUMINAMATH_CALUDE_reciprocal_plus_two_product_l3852_385234

theorem reciprocal_plus_two_product (x y : ℝ) : 
  x ≠ y → x = 1/x + 2 → y = 1/y + 2 → x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_plus_two_product_l3852_385234


namespace NUMINAMATH_CALUDE_divisibility_implication_l3852_385271

theorem divisibility_implication (a b : ℕ) (h1 : a < 1000) (h2 : b^10 ∣ a^21) : b ∣ a^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l3852_385271


namespace NUMINAMATH_CALUDE_symbol_equation_solution_l3852_385208

theorem symbol_equation_solution :
  ∀ (star square circle : ℕ),
    star + square = 24 →
    square + circle = 30 →
    circle + star = 36 →
    square = 9 ∧ circle = 21 ∧ star = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_symbol_equation_solution_l3852_385208


namespace NUMINAMATH_CALUDE_consecutive_zeros_in_prime_power_l3852_385297

theorem consecutive_zeros_in_prime_power (p : Nat) (h : Nat.Prime p) :
  ∃ n : Nat, n > 0 ∧ p^n % 10^2002 = 0 ∧ p^n % 10^2003 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_zeros_in_prime_power_l3852_385297


namespace NUMINAMATH_CALUDE_record_collection_problem_l3852_385205

theorem record_collection_problem (shared_records : ℕ) (emily_total : ℕ) (mark_unique : ℕ) : 
  shared_records = 15 → emily_total = 25 → mark_unique = 10 →
  emily_total - shared_records + mark_unique = 20 := by
sorry

end NUMINAMATH_CALUDE_record_collection_problem_l3852_385205


namespace NUMINAMATH_CALUDE_no_upper_limit_for_q_q_determines_side_ratio_l3852_385219

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- The combined area of two rotated congruent rectangles -/
noncomputable def combined_area (r : Rectangle) : ℝ :=
  if r.b / r.a ≥ Real.sqrt 2 - 1 then
    (1 - 1 / Real.sqrt 2) * (r.a + r.b)^2
  else
    2 * r.a * r.b - Real.sqrt 2 * r.b^2

/-- The ratio of combined area to single rectangle area -/
noncomputable def area_ratio (r : Rectangle) : ℝ :=
  combined_area r / r.area

theorem no_upper_limit_for_q :
  ∀ M : ℝ, ∃ r : Rectangle, area_ratio r > M :=
sorry

theorem q_determines_side_ratio {r : Rectangle} (h : Real.sqrt 2 ≤ area_ratio r ∧ area_ratio r < 2) :
  r.b / r.a = (2 - area_ratio r) / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_no_upper_limit_for_q_q_determines_side_ratio_l3852_385219


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l3852_385290

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x ≥ 0, f x = x^2 + 2*x) :
  ∀ x < 0, f x = -x^2 + 2*x :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l3852_385290


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l3852_385277

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_half : reciprocal (-1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l3852_385277


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l3852_385284

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : n = 5 ∧ d = 1600) :
  (n : ℚ) / d = 0.003125 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l3852_385284


namespace NUMINAMATH_CALUDE_largest_choir_size_l3852_385221

theorem largest_choir_size : 
  ∃ (n s : ℕ), 
    n * s < 150 ∧ 
    n * s + 3 = (n + 2) * (s - 3) ∧ 
    ∀ (m n' s' : ℕ), 
      m < 150 → 
      m + 3 = n' * s' → 
      m = (n' + 2) * (s' - 3) → 
      m ≤ n * s :=
by sorry

end NUMINAMATH_CALUDE_largest_choir_size_l3852_385221


namespace NUMINAMATH_CALUDE_time_sum_after_duration_l3852_385202

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the result on a 12-hour clock -/
def addDuration (startTime : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

theorem time_sum_after_duration (startTime : Time) (durationHours durationMinutes durationSeconds : Nat) :
  let endTime := addDuration startTime durationHours durationMinutes durationSeconds
  startTime.hours = 3 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 ∧
  durationHours = 315 ∧ durationMinutes = 58 ∧ durationSeconds = 16 →
  endTime.hours + endTime.minutes + endTime.seconds = 77 := by
  sorry

end NUMINAMATH_CALUDE_time_sum_after_duration_l3852_385202


namespace NUMINAMATH_CALUDE_cubic_inequality_l3852_385265

theorem cubic_inequality (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3852_385265


namespace NUMINAMATH_CALUDE_train_length_l3852_385254

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : Real) (pass_time : Real) (platform_length : Real) :
  train_speed = 45 * (5/18) ∧ 
  pass_time = 44 ∧ 
  platform_length = 190 →
  (train_speed * pass_time) - platform_length = 360 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l3852_385254


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3852_385286

theorem divisibility_implies_equality (a b : ℕ) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3852_385286


namespace NUMINAMATH_CALUDE_square_side_difference_sum_l3852_385296

theorem square_side_difference_sum (a b c : ℝ) 
  (ha : a^2 = 25) (hb : b^2 = 81) (hc : c^2 = 64) : 
  (b - a) + (b - c) = 5 := by sorry

end NUMINAMATH_CALUDE_square_side_difference_sum_l3852_385296


namespace NUMINAMATH_CALUDE_max_value_of_symmetric_f_l3852_385283

def f (a b x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

theorem max_value_of_symmetric_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-4 - x)) →
  (∃ x : ℝ, f a b x = 0 ∧ f a b (-4 - x) = 0) →
  (∃ m : ℝ, ∀ x : ℝ, f a b x ≤ m ∧ ∃ x₀ : ℝ, f a b x₀ = m) →
  (∃ m : ℝ, (∀ x : ℝ, f a b x ≤ m) ∧ (∃ x₀ : ℝ, f a b x₀ = m) ∧ m = 16) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_symmetric_f_l3852_385283


namespace NUMINAMATH_CALUDE_A_value_l3852_385289

/-- Rounds a natural number down to the nearest tens -/
def round_down_to_tens (n : ℕ) : ℕ :=
  (n / 10) * 10

/-- Given a natural number n = A567 where A is unknown, 
    if n rounds down to 2560, then A = 2 -/
theorem A_value (n : ℕ) (h : round_down_to_tens n = 2560) : 
  n / 1000 = 2 := by sorry

end NUMINAMATH_CALUDE_A_value_l3852_385289


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l3852_385293

-- (1) (3x-1)^2 = (x+1)^2
theorem equation_one_solutions (x : ℝ) :
  (3*x - 1)^2 = (x + 1)^2 ↔ x = 0 ∨ x = 1 := by sorry

-- (2) (x-1)^2+2x(x-1)=0
theorem equation_two_solutions (x : ℝ) :
  (x - 1)^2 + 2*x*(x - 1) = 0 ↔ x = 1 ∨ x = 1/3 := by sorry

-- (3) x^2 - 4x + 1 = 0
theorem equation_three_solutions (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

-- (4) 2x^2 + 7x - 4 = 0
theorem equation_four_solutions (x : ℝ) :
  2*x^2 + 7*x - 4 = 0 ↔ x = 1/2 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l3852_385293


namespace NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l3852_385229

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_first_four_composites_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l3852_385229


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_l3852_385255

theorem sqrt_sum_equation (x : ℝ) 
  (h : Real.sqrt (49 - x^2) - Real.sqrt (25 - x^2) = 3) : 
  Real.sqrt (49 - x^2) + Real.sqrt (25 - x^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_l3852_385255


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3852_385260

theorem absolute_value_equality (x : ℝ) (h : x < -2) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3852_385260


namespace NUMINAMATH_CALUDE_equation_solution_l3852_385226

theorem equation_solution :
  ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3852_385226


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l3852_385237

theorem circle_equation_through_points :
  let general_circle_eq (x y D E F : ℝ) := x^2 + y^2 + D*x + E*y + F = 0
  let specific_circle_eq (x y : ℝ) := x^2 + y^2 - 4*x - 6*y = 0
  (∀ x y, general_circle_eq x y (-4) (-6) 0 ↔ specific_circle_eq x y) ∧
  specific_circle_eq 0 0 ∧
  specific_circle_eq 4 0 ∧
  specific_circle_eq (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l3852_385237


namespace NUMINAMATH_CALUDE_rectangle_tileability_l3852_385262

/-- A rectangle can be tiled with 1 × b tiles -/
def IsTileable (m n b : ℕ) : Prop := sorry

/-- For an even b, there exists M such that for all m, n > M with mn even, 
    an m × n rectangle is (1, b)-tileable -/
theorem rectangle_tileability (b : ℕ) (h_even : Even b) : 
  ∃ M : ℕ, ∀ m n : ℕ, m > M → n > M → Even (m * n) → IsTileable m n b := by sorry

end NUMINAMATH_CALUDE_rectangle_tileability_l3852_385262


namespace NUMINAMATH_CALUDE_find_k_l3852_385222

theorem find_k : ∃ k : ℝ, (5 * 2 - k * 3 - 7 = 0) ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3852_385222


namespace NUMINAMATH_CALUDE_initial_investment_interest_rate_l3852_385282

/-- Given an initial investment and an additional investment with their respective interest rates,
    proves that the interest rate of the initial investment is 5% when the total annual income
    equals 6% of the entire investment. -/
theorem initial_investment_interest_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 3000)
  (h2 : additional_investment = 1499.9999999999998)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : ∃ r : ℝ, initial_investment * r + additional_investment * additional_rate =
                 (initial_investment + additional_investment) * total_rate) :
  ∃ r : ℝ, r = 0.05 ∧
    initial_investment * r + additional_investment * additional_rate =
    (initial_investment + additional_investment) * total_rate :=
sorry

end NUMINAMATH_CALUDE_initial_investment_interest_rate_l3852_385282


namespace NUMINAMATH_CALUDE_hexagon_probability_l3852_385241

/-- Represents a hexagonal checkerboard -/
structure HexBoard :=
  (total_hexagons : ℕ)
  (side_length : ℕ)

/-- Calculates the number of hexagons on the perimeter of the board -/
def perimeter_hexagons (board : HexBoard) : ℕ :=
  6 * board.side_length - 6

/-- Calculates the number of hexagons not on the perimeter of the board -/
def inner_hexagons (board : HexBoard) : ℕ :=
  board.total_hexagons - perimeter_hexagons board

/-- Theorem: The probability of a randomly chosen hexagon not touching the outer edge -/
theorem hexagon_probability (board : HexBoard) 
  (h1 : board.total_hexagons = 91)
  (h2 : board.side_length = 5) :
  (inner_hexagons board : ℚ) / board.total_hexagons = 67 / 91 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_probability_l3852_385241


namespace NUMINAMATH_CALUDE_unique_pen_distribution_l3852_385252

/-- Represents a distribution of pens among students -/
structure PenDistribution where
  num_students : ℕ
  pens_per_student : ℕ → ℕ
  total_pens : ℕ

/-- The condition that among any four pens, at least two belong to the same person -/
def four_pens_condition (d : PenDistribution) : Prop :=
  ∀ (s : Finset ℕ), s.card = 4 → ∃ i ∈ s, d.pens_per_student i ≥ 2

/-- The condition that among any five pens, no more than three belong to the same person -/
def five_pens_condition (d : PenDistribution) : Prop :=
  ∀ (s : Finset ℕ), s.card = 5 → ∀ i ∈ s, d.pens_per_student i ≤ 3

/-- The theorem stating the unique distribution satisfying the given conditions -/
theorem unique_pen_distribution :
  ∀ (d : PenDistribution),
    d.total_pens = 9 →
    four_pens_condition d →
    five_pens_condition d →
    d.num_students = 3 ∧ (∀ i, i < d.num_students → d.pens_per_student i = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_pen_distribution_l3852_385252


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1_l3852_385274

theorem tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1 
  (α : ℝ) (h : Real.tan (α + π/4) = -3) : 
  Real.cos (2*α) + 2 * Real.sin (2*α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1_l3852_385274


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l3852_385259

theorem wage_increase_percentage (old_wage new_wage : ℝ) (h1 : old_wage = 20) (h2 : new_wage = 28) :
  (new_wage - old_wage) / old_wage * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l3852_385259


namespace NUMINAMATH_CALUDE_calculation_proof_l3852_385276

theorem calculation_proof : (3.6 * 0.3) / 0.2 = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3852_385276


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3852_385269

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 3) * (2 - x) < 0} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3852_385269


namespace NUMINAMATH_CALUDE_rain_probability_l3852_385299

/-- The probability of rain on three consecutive days --/
theorem rain_probability (p_sat p_sun p_mon_given_sat : ℝ) 
  (h_sat : p_sat = 0.7)
  (h_sun : p_sun = 0.5)
  (h_mon_given_sat : p_mon_given_sat = 0.4) :
  p_sat * p_sun * p_mon_given_sat = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3852_385299


namespace NUMINAMATH_CALUDE_megan_initial_bottles_l3852_385294

/-- The number of water bottles Megan had initially -/
def initial_bottles : ℕ := sorry

/-- The number of water bottles Megan drank -/
def bottles_drank : ℕ := 3

/-- The number of water bottles Megan had left -/
def bottles_left : ℕ := 14

theorem megan_initial_bottles : 
  initial_bottles = bottles_left + bottles_drank :=
by sorry

end NUMINAMATH_CALUDE_megan_initial_bottles_l3852_385294


namespace NUMINAMATH_CALUDE_melanie_dimes_l3852_385209

/-- The number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Proof that Melanie has 19 dimes after receiving dimes from her parents -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l3852_385209


namespace NUMINAMATH_CALUDE_ellipse_equation_l3852_385272

theorem ellipse_equation (x y : ℝ) : 
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m ≠ n ∧
    m * 2^2 + n * Real.sqrt 2^2 = 1 ∧
    m * (Real.sqrt 2)^2 + n * (Real.sqrt 3)^2 = 1) →
  (x^2 / 8 + y^2 / 4 = 1 ↔ m * x^2 + n * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3852_385272


namespace NUMINAMATH_CALUDE_mass_of_Fe2CO3_3_l3852_385278

/-- The molar mass of iron in g/mol -/
def molar_mass_Fe : ℝ := 55.845

/-- The molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.011

/-- The molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 15.999

/-- The number of moles of Fe2(CO3)3 -/
def num_moles : ℝ := 12

/-- The molar mass of Fe2(CO3)3 in g/mol -/
def molar_mass_Fe2CO3_3 : ℝ :=
  2 * molar_mass_Fe + 3 * molar_mass_C + 9 * molar_mass_O

/-- The mass of Fe2(CO3)3 in grams -/
def mass_Fe2CO3_3 : ℝ := num_moles * molar_mass_Fe2CO3_3

theorem mass_of_Fe2CO3_3 : mass_Fe2CO3_3 = 3500.568 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_Fe2CO3_3_l3852_385278


namespace NUMINAMATH_CALUDE_find_k_value_l3852_385204

theorem find_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4) → k = -17 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l3852_385204


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3852_385200

theorem scientific_notation_equivalence : 
  8200000 = 8.2 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3852_385200


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3852_385217

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), n^300 < 3^500 ∧ ∀ (m : ℕ), m^300 < 3^500 → m ≤ n :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3852_385217


namespace NUMINAMATH_CALUDE_workshop_workers_l3852_385203

/-- The number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- The average salary of all workers in the workshop -/
def avg_salary_all : ℕ := 8000

/-- The average salary of technicians in the workshop -/
def avg_salary_tech : ℕ := 12000

/-- The average salary of non-technicians in the workshop -/
def avg_salary_nontech : ℕ := 6000

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 21

theorem workshop_workers :
  (total_workers * avg_salary_all = 
   num_technicians * avg_salary_tech + 
   (total_workers - num_technicians) * avg_salary_nontech) ∧
  (total_workers = 21) := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l3852_385203


namespace NUMINAMATH_CALUDE_mayor_harvey_flowers_l3852_385235

/-- Represents the quantities of flowers for an institution -/
structure FlowerQuantities :=
  (roses : ℕ)
  (tulips : ℕ)
  (lilies : ℕ)

/-- Calculates the total number of flowers for given quantities -/
def totalFlowers (quantities : FlowerQuantities) : ℕ :=
  quantities.roses + quantities.tulips + quantities.lilies

/-- Theorem: The total number of flowers Mayor Harvey needs to buy is 855 -/
theorem mayor_harvey_flowers :
  let nursing_home : FlowerQuantities := ⟨90, 80, 100⟩
  let shelter : FlowerQuantities := ⟨120, 75, 95⟩
  let maternity_ward : FlowerQuantities := ⟨100, 110, 85⟩
  totalFlowers nursing_home + totalFlowers shelter + totalFlowers maternity_ward = 855 :=
by
  sorry

#eval let nursing_home : FlowerQuantities := ⟨90, 80, 100⟩
      let shelter : FlowerQuantities := ⟨120, 75, 95⟩
      let maternity_ward : FlowerQuantities := ⟨100, 110, 85⟩
      totalFlowers nursing_home + totalFlowers shelter + totalFlowers maternity_ward

end NUMINAMATH_CALUDE_mayor_harvey_flowers_l3852_385235


namespace NUMINAMATH_CALUDE_crazy_silly_school_theorem_l3852_385258

/-- Represents the 'Crazy Silly School' series collection --/
structure CrazySillyCollection where
  books : ℕ
  movies : ℕ
  videoGames : ℕ
  audiobooks : ℕ

/-- Represents the completed items in the collection --/
structure CompletedItems where
  booksRead : ℕ
  moviesWatched : ℕ
  gamesPlayed : ℕ
  audiobooksListened : ℕ
  halfReadBooks : ℕ
  halfWatchedMovies : ℕ

/-- Calculates the portions left to complete in the collection --/
def portionsLeftToComplete (collection : CrazySillyCollection) (completed : CompletedItems) : ℚ :=
  let totalPortions := collection.books + collection.movies + collection.videoGames + collection.audiobooks
  let completedPortions := completed.booksRead - completed.halfReadBooks / 2 +
                           completed.moviesWatched - completed.halfWatchedMovies / 2 +
                           completed.gamesPlayed +
                           completed.audiobooksListened
  totalPortions - completedPortions

/-- Theorem stating the number of portions left to complete in the 'Crazy Silly School' series --/
theorem crazy_silly_school_theorem (collection : CrazySillyCollection) (completed : CompletedItems) :
  collection.books = 22 ∧
  collection.movies = 10 ∧
  collection.videoGames = 8 ∧
  collection.audiobooks = 15 ∧
  completed.booksRead = 12 ∧
  completed.moviesWatched = 6 ∧
  completed.gamesPlayed = 3 ∧
  completed.audiobooksListened = 7 ∧
  completed.halfReadBooks = 2 ∧
  completed.halfWatchedMovies = 1 →
  portionsLeftToComplete collection completed = 28.5 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_theorem_l3852_385258


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3852_385291

theorem quadratic_solution_difference_squared :
  ∀ d e : ℝ,
  (4 * d^2 + 8 * d - 48 = 0) →
  (4 * e^2 + 8 * e - 48 = 0) →
  d ≠ e →
  (d - e)^2 = 49 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3852_385291


namespace NUMINAMATH_CALUDE_two_digit_number_swap_sum_theorem_l3852_385268

/-- Represents a two-digit number with distinct non-zero digits -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_not_zero : tens ≠ 0
  units_not_zero : units ≠ 0
  distinct_digits : tens ≠ units
  is_two_digit : tens < 10 ∧ units < 10

/-- The value of a TwoDigitNumber -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The value of a TwoDigitNumber with swapped digits -/
def TwoDigitNumber.swapped_value (n : TwoDigitNumber) : Nat :=
  10 * n.units + n.tens

theorem two_digit_number_swap_sum_theorem 
  (a b c : TwoDigitNumber) 
  (h : a.value + b.value + c.value = 41) :
  a.swapped_value + b.swapped_value + c.swapped_value = 113 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_swap_sum_theorem_l3852_385268


namespace NUMINAMATH_CALUDE_tshirt_cost_l3852_385207

theorem tshirt_cost (initial_amount : ℕ) (sweater_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 91 →
  sweater_cost = 24 →
  shoes_cost = 11 →
  remaining_amount = 50 →
  initial_amount - remaining_amount - sweater_cost - shoes_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l3852_385207


namespace NUMINAMATH_CALUDE_hens_count_l3852_385288

/-- Represents the number of hens in the farm -/
def num_hens : ℕ := sorry

/-- Represents the number of cows in the farm -/
def num_cows : ℕ := sorry

/-- The total number of heads in the farm -/
def total_heads : ℕ := 48

/-- The total number of feet in the farm -/
def total_feet : ℕ := 140

/-- Each hen has 1 head and 2 feet -/
def hen_head_feet : ℕ × ℕ := (1, 2)

/-- Each cow has 1 head and 4 feet -/
def cow_head_feet : ℕ × ℕ := (1, 4)

theorem hens_count : num_hens = 26 :=
  by sorry

end NUMINAMATH_CALUDE_hens_count_l3852_385288


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3852_385298

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3852_385298


namespace NUMINAMATH_CALUDE_customs_waiting_time_l3852_385250

/-- The time Jack waited to get through customs, given total waiting time and quarantine days. -/
theorem customs_waiting_time (total_hours quarantine_days : ℕ) : 
  total_hours = 356 ∧ quarantine_days = 14 → 
  total_hours - (quarantine_days * 24) = 20 := by
  sorry

end NUMINAMATH_CALUDE_customs_waiting_time_l3852_385250


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3852_385238

theorem quadratic_roots_sum_product (α β : ℝ) : 
  α^2 + α - 1 = 0 → β^2 + β - 1 = 0 → α ≠ β → α*β + α + β = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3852_385238


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3852_385210

theorem absolute_value_simplification : |-4^2 + 7| = 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3852_385210


namespace NUMINAMATH_CALUDE_root_sum_fourth_power_l3852_385227

theorem root_sum_fourth_power (a b c s : ℝ) : 
  (x^3 - 6*x^2 + 14*x - 6 = 0 → (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 12*s^2 - 24*s = 20 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fourth_power_l3852_385227


namespace NUMINAMATH_CALUDE_condition1_coordinates_condition2_coordinates_l3852_385285

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point A with coordinates dependent on parameter a -/
def A (a : ℝ) : Point := ⟨3*a + 2, 2*a - 4⟩

/-- Fixed point B -/
def B : Point := ⟨3, 4⟩

/-- Condition 1: Line AB is parallel to x-axis -/
def parallel_to_x_axis (A B : Point) : Prop :=
  A.y = B.y

/-- Condition 2: Distance from A to both coordinate axes is equal -/
def equal_distance_to_axes (A : Point) : Prop :=
  |A.x| = |A.y|

/-- Theorem for Condition 1 -/
theorem condition1_coordinates :
  ∃ a : ℝ, parallel_to_x_axis (A a) B → A a = ⟨14, 4⟩ := by sorry

/-- Theorem for Condition 2 -/
theorem condition2_coordinates :
  ∃ a : ℝ, equal_distance_to_axes (A a) → 
    (A a = ⟨-16, -16⟩ ∨ A a = ⟨3.2, -3.2⟩) := by sorry

end NUMINAMATH_CALUDE_condition1_coordinates_condition2_coordinates_l3852_385285


namespace NUMINAMATH_CALUDE_function_inequality_and_ratio_l3852_385246

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Define the maximum value T
def T : ℝ := 3

-- Theorem statement
theorem function_inequality_and_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = T) :
  (∀ x : ℝ, f x ≥ T) ∧ (2 / (1/a + 1/b) ≤ Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_and_ratio_l3852_385246


namespace NUMINAMATH_CALUDE_soccer_shoe_price_l3852_385228

theorem soccer_shoe_price (total_pairs : Nat) (total_price : Nat) :
  total_pairs = 99 →
  total_price % 100 = 76 →
  total_price < 20000 →
  ∃ (price_per_pair : Nat), 
    price_per_pair * total_pairs = total_price ∧
    price_per_pair = 124 :=
by sorry

end NUMINAMATH_CALUDE_soccer_shoe_price_l3852_385228


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l3852_385230

theorem least_subtrahend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution :
  let n := 13603
  let d := 87
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 ∧ k = 31 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l3852_385230


namespace NUMINAMATH_CALUDE_problem_statement_l3852_385215

/-- Given xw + yz = 8 and (2x + y)(2z + w) = 20, prove that xz + yw = 1 -/
theorem problem_statement (x y z w : ℝ) 
  (h1 : x * w + y * z = 8)
  (h2 : (2 * x + y) * (2 * z + w) = 20) :
  x * z + y * w = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3852_385215


namespace NUMINAMATH_CALUDE_wheel_speed_l3852_385220

/-- The speed (in miles per hour) of a wheel with a 10-foot circumference -/
def r : ℝ := sorry

/-- The time (in hours) for one complete rotation of the wheel -/
def t : ℝ := sorry

/-- Relation between speed, time, and distance for one rotation -/
axiom speed_time_relation : r * t = (10 / 5280)

/-- Relation between original and new speed and time -/
axiom speed_time_change : (r + 5) * (t - 1 / (3 * 3600)) = (10 / 5280)

theorem wheel_speed : r = 10 := by sorry

end NUMINAMATH_CALUDE_wheel_speed_l3852_385220


namespace NUMINAMATH_CALUDE_line_parameterization_vector_l3852_385287

/-- The line equation y = (4x - 7) / 3 -/
def line_equation (x y : ℝ) : Prop := y = (4 * x - 7) / 3

/-- The parameterization of the line -/
def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

/-- The distance constraint -/
def distance_constraint (p : ℝ × ℝ) (t : ℝ) : Prop :=
  p.1 ≥ 5 → ‖(p.1 - 5, p.2 - 2)‖ = 2 * t

/-- The theorem statement -/
theorem line_parameterization_vector :
  ∃ (v : ℝ × ℝ), ∀ (x y t : ℝ),
    let p := parameterization v (6/5, 8/5) t
    line_equation p.1 p.2 ∧
    distance_constraint p t :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_vector_l3852_385287


namespace NUMINAMATH_CALUDE_seven_n_representable_l3852_385214

theorem seven_n_representable (n a b : ℤ) (h : n = a^2 + a*b + b^2) :
  ∃ x y : ℤ, 7*n = x^2 + x*y + y^2 := by sorry

end NUMINAMATH_CALUDE_seven_n_representable_l3852_385214


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_l3852_385249

theorem cos_shift_equals_sin (x : ℝ) : 
  Real.cos (x + π/3) = Real.sin (x + 5*π/6) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_l3852_385249


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3852_385225

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ + a₃ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3852_385225


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3852_385243

theorem decimal_point_problem : ∃ x : ℝ, x > 0 ∧ 1000 * x = 9 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3852_385243
