import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3537_353721

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions,
    prove that the sum of its third and fourth terms is 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h1 : isArithmeticSequence a) 
    (h2 : a 1 + a 2 = 10) 
    (h3 : a 4 = a 3 + 2) : 
  a 3 + a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3537_353721


namespace NUMINAMATH_CALUDE_swimming_laps_per_day_l3537_353747

/-- Proves that swimming 300 laps in 5 weeks, 5 days per week, results in 12 laps per day -/
theorem swimming_laps_per_day 
  (total_laps : ℕ) 
  (weeks : ℕ) 
  (days_per_week : ℕ) 
  (h1 : total_laps = 300) 
  (h2 : weeks = 5) 
  (h3 : days_per_week = 5) : 
  total_laps / (weeks * days_per_week) = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimming_laps_per_day_l3537_353747


namespace NUMINAMATH_CALUDE_factorization_x4_plus_324_l3537_353781

theorem factorization_x4_plus_324 (x : ℝ) : 
  x^4 + 324 = (x^2 - 18*x + 162) * (x^2 + 18*x + 162) := by sorry

end NUMINAMATH_CALUDE_factorization_x4_plus_324_l3537_353781


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3537_353740

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7, one side is 3
  a + b + c = 17 :=        -- The perimeter is 17
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3537_353740


namespace NUMINAMATH_CALUDE_min_quotient_l3537_353763

/-- A three-digit number with distinct non-zero digits that sum to 10 -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h3 : a + b + c = 10

/-- The quotient of the number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : ℚ :=
  (100 * n.a + 10 * n.b + n.c) / (n.a + n.b + n.c)

/-- The minimum value of the quotient is 12.7 -/
theorem min_quotient :
  ∀ n : ThreeDigitNumber, quotient n ≥ 127/10 ∧ ∃ m : ThreeDigitNumber, quotient m = 127/10 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_l3537_353763


namespace NUMINAMATH_CALUDE_point_five_units_from_origin_l3537_353753

theorem point_five_units_from_origin (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_five_units_from_origin_l3537_353753


namespace NUMINAMATH_CALUDE_five_lapping_points_l3537_353777

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- The circular track model -/
def CircularTrack := Unit

/-- Calculates the number of lapping points on a circular track -/
def numberOfLappingPoints (track : CircularTrack) (a b : Runner) : ℕ :=
  sorry

theorem five_lapping_points (track : CircularTrack) (a b : Runner) :
  a.speed > 0 ∧ b.speed > 0 ∧
  a.initialPosition = b.initialPosition + 10 ∧
  b.speed * 22 = a.speed * 32 →
  numberOfLappingPoints track a b = 5 :=
sorry

end NUMINAMATH_CALUDE_five_lapping_points_l3537_353777


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3537_353741

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := y - 2 = m * x + m

-- Theorem statement
theorem fixed_point_on_line :
  ∀ m : ℝ, line_equation (-1) 2 m :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3537_353741


namespace NUMINAMATH_CALUDE_initial_female_percentage_calculation_l3537_353704

/-- Represents a company's workforce statistics -/
structure Company where
  initial_employees : ℕ
  initial_female_percentage : ℚ
  hired_male_workers : ℕ
  final_employees : ℕ
  final_female_percentage : ℚ

/-- Theorem stating the relationship between initial and final workforce statistics -/
theorem initial_female_percentage_calculation (c : Company) 
  (h1 : c.hired_male_workers = 20)
  (h2 : c.final_employees = 240)
  (h3 : c.final_female_percentage = 55/100)
  (h4 : c.initial_employees + c.hired_male_workers = c.final_employees)
  (h5 : c.initial_employees * c.initial_female_percentage = 
        c.final_employees * c.final_female_percentage) :
  c.initial_female_percentage = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_initial_female_percentage_calculation_l3537_353704


namespace NUMINAMATH_CALUDE_fabric_cutting_l3537_353738

theorem fabric_cutting (initial_length : ℚ) (cut_length : ℚ) (desired_length : ℚ) :
  initial_length = 2/3 →
  cut_length = 1/6 →
  desired_length = 1/2 →
  initial_length - cut_length = desired_length :=
by sorry

end NUMINAMATH_CALUDE_fabric_cutting_l3537_353738


namespace NUMINAMATH_CALUDE_part_one_part_two_l3537_353776

-- Part 1
theorem part_one (x : ℝ) (h1 : x^2 - 4*x + 3 < 0) (h2 : |x - 3| < 1) :
  2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h_pos : a > 0) 
  (h_subset : {x : ℝ | x^2 - 4*a*x + 3*a^2 ≥ 0} ⊂ {x : ℝ | |x - 3| ≥ 1}) :
  4/3 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3537_353776


namespace NUMINAMATH_CALUDE_min_value_z_l3537_353759

theorem min_value_z (x y : ℝ) : x^2 + 3*y^2 + 8*x - 6*y + 30 ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3537_353759


namespace NUMINAMATH_CALUDE_raine_steps_theorem_l3537_353795

/-- The number of steps Raine takes to walk to school -/
def steps_to_school : ℕ := 150

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- The total number of steps Raine takes in the given number of days -/
def total_steps : ℕ := 2 * steps_to_school * days

theorem raine_steps_theorem : total_steps = 1500 := by
  sorry

end NUMINAMATH_CALUDE_raine_steps_theorem_l3537_353795


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_999_l3537_353769

/-- The sum of all digits of integers from 0 to 999 inclusive -/
def sumOfDigits : ℕ := sorry

/-- The range of integers we're considering -/
def integerRange : Set ℕ := { n | 0 ≤ n ∧ n ≤ 999 }

theorem sum_of_digits_0_to_999 : sumOfDigits = 13500 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_0_to_999_l3537_353769


namespace NUMINAMATH_CALUDE_cars_triangle_right_angle_l3537_353765

/-- Represents a car traveling on a triangular path -/
structure Car where
  speedAB : ℝ
  speedBC : ℝ
  speedCA : ℝ

/-- Represents a triangle with three cars traveling on its sides -/
structure TriangleWithCars where
  -- Lengths of the sides of the triangle
  ab : ℝ
  bc : ℝ
  ca : ℝ
  -- The three cars
  car1 : Car
  car2 : Car
  car3 : Car

/-- The theorem stating that if three cars travel on a triangle and return at the same time, 
    the angle ABC is 90 degrees -/
theorem cars_triangle_right_angle (t : TriangleWithCars) : 
  (t.ab / t.car1.speedAB + t.bc / t.car1.speedBC + t.ca / t.car1.speedCA = 
   t.ab / t.car2.speedAB + t.bc / t.car2.speedBC + t.ca / t.car2.speedCA) ∧
  (t.ab / t.car1.speedAB + t.bc / t.car1.speedBC + t.ca / t.car1.speedCA = 
   t.ab / t.car3.speedAB + t.bc / t.car3.speedBC + t.ca / t.car3.speedCA) ∧
  (t.car1.speedAB = 12) ∧ (t.car1.speedBC = 10) ∧ (t.car1.speedCA = 15) ∧
  (t.car2.speedAB = 15) ∧ (t.car2.speedBC = 15) ∧ (t.car2.speedCA = 10) ∧
  (t.car3.speedAB = 10) ∧ (t.car3.speedBC = 20) ∧ (t.car3.speedCA = 12) →
  ∃ (A B C : ℝ × ℝ), 
    let angleABC := Real.arccos ((t.ab^2 + t.bc^2 - t.ca^2) / (2 * t.ab * t.bc))
    angleABC = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_triangle_right_angle_l3537_353765


namespace NUMINAMATH_CALUDE_f_is_increasing_on_reals_l3537_353705

-- Define the function
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem f_is_increasing_on_reals :
  (∀ x, x ∈ Set.univ → f x ∈ Set.univ) ∧
  (∀ x y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_increasing_on_reals_l3537_353705


namespace NUMINAMATH_CALUDE_students_without_A_l3537_353742

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ)
  (h_total : total = 40)
  (h_history : history = 10)
  (h_math : math = 18)
  (h_both : both = 6) :
  total - (history + math - both) = 18 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l3537_353742


namespace NUMINAMATH_CALUDE_two_year_increase_l3537_353755

/-- Calculates the final value after two years of percentage increases -/
def final_value (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating the final value after two years of specific increases -/
theorem two_year_increase (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : initial = 65000)
  (h2 : rate1 = 0.12)
  (h3 : rate2 = 0.08) :
  final_value initial rate1 rate2 = 78624 := by
  sorry

#eval final_value 65000 0.12 0.08

end NUMINAMATH_CALUDE_two_year_increase_l3537_353755


namespace NUMINAMATH_CALUDE_equation_solution_l3537_353730

theorem equation_solution :
  ∃ x : ℝ, (2*x - 1)^2 - (1 - 3*x)^2 = 5*(1 - x)*(x + 1) ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3537_353730


namespace NUMINAMATH_CALUDE_givenPoint_in_first_quadrant_l3537_353788

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point -/
def givenPoint : Point2D :=
  { x := 6, y := 2 }

/-- Theorem stating that the given point is in the first quadrant -/
theorem givenPoint_in_first_quadrant :
  isInFirstQuadrant givenPoint :=
by
  sorry

end NUMINAMATH_CALUDE_givenPoint_in_first_quadrant_l3537_353788


namespace NUMINAMATH_CALUDE_remaining_money_is_63_10_l3537_353772

/-- Calculates the remaining money after hotel stays --/
def remaining_money (initial_amount : ℝ) 
  (hotel1_night_rate hotel1_morning_rate : ℝ)
  (hotel1_night_hours hotel1_morning_hours : ℝ)
  (hotel2_night_rate hotel2_morning_rate : ℝ)
  (hotel2_night_hours hotel2_morning_hours : ℝ)
  (tax_rate service_fee : ℝ) : ℝ :=
  let hotel1_subtotal := hotel1_night_rate * hotel1_night_hours + hotel1_morning_rate * hotel1_morning_hours
  let hotel2_subtotal := hotel2_night_rate * hotel2_night_hours + hotel2_morning_rate * hotel2_morning_hours
  let hotel1_total := hotel1_subtotal * (1 + tax_rate) + service_fee
  let hotel2_total := hotel2_subtotal * (1 + tax_rate) + service_fee
  initial_amount - (hotel1_total + hotel2_total)

/-- Theorem stating that the remaining money after hotel stays is $63.10 --/
theorem remaining_money_is_63_10 :
  remaining_money 160 2.5 3 6 4 3.5 4 8 6 0.1 5 = 63.1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_63_10_l3537_353772


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l3537_353703

theorem other_solution_quadratic (x : ℚ) :
  56 * (5/7)^2 + 27 = 89 * (5/7) - 8 →
  56 * (7/8)^2 + 27 = 89 * (7/8) - 8 :=
by sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l3537_353703


namespace NUMINAMATH_CALUDE_completing_square_result_l3537_353766

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x - 1 = 0) ↔ ((x - 2)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_result_l3537_353766


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_y_l3537_353712

-- Define the function f(x, y) = x + y
def f (x y : ℝ) := x + y

-- State the theorem
theorem min_value_of_x_plus_y (x y : ℝ) (h1 : x > 1) (h2 : x * y = 2 * x + y + 2) :
  ∃ (m : ℝ), m = 7 ∧ ∀ (x' y' : ℝ), x' > 1 → x' * y' = 2 * x' + y' + 2 → f x' y' ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_y_l3537_353712


namespace NUMINAMATH_CALUDE_center_after_transformations_l3537_353711

-- Define the initial center coordinates
def initial_center : ℝ × ℝ := (3, -4)

-- Define the reflection across x-axis function
def reflect_x (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Define the translation function
def translate_right (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 + units, point.2)

-- Theorem statement
theorem center_after_transformations :
  let reflected := reflect_x initial_center
  let final := translate_right reflected 5
  final = (8, 4) := by sorry

end NUMINAMATH_CALUDE_center_after_transformations_l3537_353711


namespace NUMINAMATH_CALUDE_sharas_age_l3537_353724

theorem sharas_age (jaymee_age shara_age : ℕ) : 
  jaymee_age = 22 →
  jaymee_age = 2 * shara_age + 2 →
  shara_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_sharas_age_l3537_353724


namespace NUMINAMATH_CALUDE_room_width_calculation_l3537_353785

/-- Given a room with specified dimensions and paving costs, calculate its width -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 600)
  (h3 : total_cost = 12375) :
  total_cost / cost_per_sqm / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l3537_353785


namespace NUMINAMATH_CALUDE_T_equality_l3537_353727

theorem T_equality (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = (x - 1)^4 + 1 := by
sorry

end NUMINAMATH_CALUDE_T_equality_l3537_353727


namespace NUMINAMATH_CALUDE_lawnmower_depreciation_l3537_353796

theorem lawnmower_depreciation (initial_value : ℝ) (first_depreciation_rate : ℝ) (second_depreciation_rate : ℝ) :
  initial_value = 100 →
  first_depreciation_rate = 0.25 →
  second_depreciation_rate = 0.20 →
  initial_value * (1 - first_depreciation_rate) * (1 - second_depreciation_rate) = 60 := by
sorry

end NUMINAMATH_CALUDE_lawnmower_depreciation_l3537_353796


namespace NUMINAMATH_CALUDE_max_red_balls_l3537_353746

theorem max_red_balls 
  (total : ℕ) 
  (green : ℕ) 
  (h1 : total = 28) 
  (h2 : green = 12) 
  (h3 : ∀ red : ℕ, red + green < 24) : 
  ∃ max_red : ℕ, max_red = 11 ∧ ∀ red : ℕ, red ≤ max_red := by
sorry

end NUMINAMATH_CALUDE_max_red_balls_l3537_353746


namespace NUMINAMATH_CALUDE_surrounding_circle_area_l3537_353774

/-- Given a circle of radius R surrounded by four equal circles, each touching 
    the given circle and each other, the area of one surrounding circle is πR²(3 + 2√2) -/
theorem surrounding_circle_area (R : ℝ) (R_pos : R > 0) : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    (R + r)^2 + (R + r)^2 = (2*r)^2 ∧ 
    r = R * (1 + Real.sqrt 2) ∧
    π * r^2 = π * R^2 * (3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_surrounding_circle_area_l3537_353774


namespace NUMINAMATH_CALUDE_smallest_result_is_16_l3537_353760

def S : Finset Nat := {2, 3, 5, 7, 11, 13}

theorem smallest_result_is_16 :
  ∃ (a b c : Nat), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b) * c = 16 ∧
  ∀ (x y z : Nat), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  (x + y) * z ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_smallest_result_is_16_l3537_353760


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l3537_353756

/-- The area of a triangle inscribed in a circle, given the circle's radius and the ratio of the triangle's sides. -/
theorem inscribed_triangle_area
  (r : ℝ) -- radius of the circle
  (a b c : ℝ) -- ratios of the triangle's sides
  (h_positive : r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0) -- positivity conditions
  (h_ratio : a^2 + b^2 = c^2) -- Pythagorean theorem condition for the ratios
  (h_diameter : c * (a + b + c)⁻¹ * 2 * r = c) -- condition relating the longest side to the diameter
  : (1/2 * a * b * (a + b + c)⁻¹ * 2 * r)^2 = 216/25 ∧ r = 3 ∧ (a, b, c) = (3, 4, 5) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l3537_353756


namespace NUMINAMATH_CALUDE_cubic_expression_equals_1000_l3537_353783

theorem cubic_expression_equals_1000 (α : ℝ) (h : α = 6) : 
  α^3 + 3*α^2*4 + 3*α*16 + 64 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equals_1000_l3537_353783


namespace NUMINAMATH_CALUDE_reflection_line_is_x_equals_zero_l3537_353745

-- Define the points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (5, 7)
def R : ℝ × ℝ := (-2, 5)
def P' : ℝ × ℝ := (-1, 2)
def Q' : ℝ × ℝ := (-5, 7)
def R' : ℝ × ℝ := (2, 5)

-- Define the reflection line
def M : Set (ℝ × ℝ) := {(x, y) | x = 0}

-- Theorem statement
theorem reflection_line_is_x_equals_zero :
  (∀ (x y : ℝ), (x, y) ∈ M ↔ x = 0) ∧
  (P.1 + P'.1 = 0) ∧ (P.2 = P'.2) ∧
  (Q.1 + Q'.1 = 0) ∧ (Q.2 = Q'.2) ∧
  (R.1 + R'.1 = 0) ∧ (R.2 = R'.2) :=
sorry


end NUMINAMATH_CALUDE_reflection_line_is_x_equals_zero_l3537_353745


namespace NUMINAMATH_CALUDE_rock_age_count_l3537_353758

/-- The set of digits used to form the rock's age -/
def rock_age_digits : Finset Nat := {2, 3, 7, 9}

/-- The number of occurrences of each digit in the rock's age -/
def digit_occurrences : Nat → Nat
  | 2 => 3
  | 3 => 1
  | 7 => 1
  | 9 => 1
  | _ => 0

/-- The set of odd digits that can start the rock's age -/
def odd_start_digits : Finset Nat := {3, 7, 9}

/-- The length of the rock's age in digits -/
def age_length : Nat := 6

/-- The number of possibilities for the rock's age -/
def rock_age_possibilities : Nat := 60

theorem rock_age_count :
  (Finset.card odd_start_digits) *
  (Nat.factorial (age_length - 1)) /
  (Nat.factorial (digit_occurrences 2)) =
  rock_age_possibilities := by sorry

end NUMINAMATH_CALUDE_rock_age_count_l3537_353758


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l3537_353754

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := 
  sorry

/-- The specific line passing through (-1, 1) and (0, 3) -/
def specific_line : Line := { x₁ := -1, y₁ := 1, x₂ := 0, y₂ := 3 }

theorem x_intercept_of_specific_line : 
  x_intercept specific_line = -3/2 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l3537_353754


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3537_353799

def f (x : ℝ) : ℝ := x^2

def B : Set ℝ := {1, 4}

theorem intersection_of_A_and_B (A : Set ℝ) (h : ∀ x ∈ A, f x ∈ B) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3537_353799


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3537_353782

/-- The length of the major axis of an ellipse with given foci and y-axis tangency -/
theorem ellipse_major_axis_length : 
  let f₁ : ℝ × ℝ := (1, -3 + 2 * Real.sqrt 3)
  let f₂ : ℝ × ℝ := (1, -3 - 2 * Real.sqrt 3)
  ∀ (e : Set (ℝ × ℝ)), 
    (∃ (p : ℝ × ℝ), p ∈ e ∧ p.1 = 0) →  -- Tangent to y-axis
    (∀ (q : ℝ × ℝ), q ∈ e → ∃ (a b : ℝ), a * (q.1 - f₁.1)^2 + b * (q.2 - f₁.2)^2 = 1 ∧ 
                                         a * (q.1 - f₂.1)^2 + b * (q.2 - f₂.2)^2 = 1) →
    (∃ (major_axis : ℝ), major_axis = 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3537_353782


namespace NUMINAMATH_CALUDE_m_arit_fib_seq_periodic_l3537_353748

/-- An m-arithmetic Fibonacci sequence -/
def MAritFibSeq (m : ℕ) := ℕ → Fin m

/-- The period of an m-arithmetic Fibonacci sequence -/
def Period (m : ℕ) (v : MAritFibSeq m) (r : ℕ) : Prop :=
  ∀ n, v n = v (n + r)

theorem m_arit_fib_seq_periodic (m : ℕ) (v : MAritFibSeq m) :
  ∃ r : ℕ, r ≤ m^2 ∧ Period m v r := by
  sorry

end NUMINAMATH_CALUDE_m_arit_fib_seq_periodic_l3537_353748


namespace NUMINAMATH_CALUDE_extreme_value_difference_l3537_353793

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x

theorem extreme_value_difference (a b : ℝ) :
  (∃ x, x = 2 ∧ (deriv (f a b)) x = 0) →
  (deriv (f a b)) 1 = -3 →
  ∃ max min, (∀ x, f a b x ≤ max) ∧ 
              (∀ x, f a b x ≥ min) ∧ 
              max - min = 4 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_difference_l3537_353793


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3537_353736

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 = (X^2 + 3*X + 2) * q + (-15*X - 14) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3537_353736


namespace NUMINAMATH_CALUDE_inequality_proof_l3537_353784

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.exp (2 * x^3) - 2*x > 2*(x+1)*Real.log x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3537_353784


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l3537_353719

/-- Given a function f(x) = ax³ + b sin(x) + 1 where f(1) = 5, prove that f(-1) = -3 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 1) 
  (h2 : f 1 = 5) : 
  f (-1) = -3 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l3537_353719


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l3537_353786

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8|

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }

-- State the theorem
theorem sum_of_max_and_min_is_two :
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max + min = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l3537_353786


namespace NUMINAMATH_CALUDE_flour_to_add_l3537_353768

/-- Given the total amount of flour required for a recipe and the amount already added,
    this theorem proves that the amount of flour needed to be added is the difference
    between the total required and the amount already added. -/
theorem flour_to_add (total_flour : ℕ) (flour_added : ℕ) :
  total_flour ≥ flour_added →
  total_flour - flour_added = total_flour - flour_added := by
  sorry

#check flour_to_add

end NUMINAMATH_CALUDE_flour_to_add_l3537_353768


namespace NUMINAMATH_CALUDE_quadratic_min_max_l3537_353708

theorem quadratic_min_max (x : ℝ) (n : ℝ) :
  (∀ x, x^2 - 4*x - 3 ≥ -7) ∧
  (n = 6 - x → ∀ x, Real.sqrt (x^2 - 2*n^2) ≤ 6 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_min_max_l3537_353708


namespace NUMINAMATH_CALUDE_flyers_left_to_hand_out_l3537_353752

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed : ℕ) 
  (rose_handed : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed = 120)
  (h3 : rose_handed = 320) :
  total_flyers - (jack_handed + rose_handed) = 796 :=
by sorry

end NUMINAMATH_CALUDE_flyers_left_to_hand_out_l3537_353752


namespace NUMINAMATH_CALUDE_driver_comparison_l3537_353798

theorem driver_comparison (d : ℝ) (h : d > 0) : d / 40 < 8 * d / 315 := by
  sorry

#check driver_comparison

end NUMINAMATH_CALUDE_driver_comparison_l3537_353798


namespace NUMINAMATH_CALUDE_d_value_approx_l3537_353725

-- Define the equation
def equation (d : ℝ) : Prop :=
  4 * ((3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5)) = 3200.0000000000005

-- Theorem statement
theorem d_value_approx :
  ∃ d : ℝ, equation d ∧ abs (d - 0.3) < 0.0000001 :=
sorry

end NUMINAMATH_CALUDE_d_value_approx_l3537_353725


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3537_353733

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (4 + x) / (4 - 2*x)) ↔ x ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3537_353733


namespace NUMINAMATH_CALUDE_sum_of_ages_l3537_353751

theorem sum_of_ages (maria_age jose_age : ℕ) : 
  maria_age = 14 → 
  jose_age = maria_age + 12 → 
  maria_age + jose_age = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3537_353751


namespace NUMINAMATH_CALUDE_equation_solution_l3537_353731

theorem equation_solution : 
  ∃ x : ℝ, (4 : ℝ) ^ x = 2 ^ (x + 1) - 1 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3537_353731


namespace NUMINAMATH_CALUDE_permutations_of_377353752_div_by_5_l3537_353702

def original_number : ℕ := 377353752

-- Function to count occurrences of a digit in a number
def count_digit (n : ℕ) (d : ℕ) : ℕ := sorry

-- Function to calculate factorial
def factorial (n : ℕ) : ℕ := sorry

-- Function to calculate permutations of multiset
def permutations_multiset (n : ℕ) (counts : List ℕ) : ℕ := sorry

theorem permutations_of_377353752_div_by_5 :
  let digits := [3, 3, 3, 7, 7, 7, 5, 2]
  let n := digits.length
  let counts := [
    count_digit original_number 3,
    count_digit original_number 7,
    count_digit original_number 5,
    count_digit original_number 2
  ]
  permutations_multiset n counts = 1120 :=
by sorry

end NUMINAMATH_CALUDE_permutations_of_377353752_div_by_5_l3537_353702


namespace NUMINAMATH_CALUDE_boat_current_speed_l3537_353709

/-- Proves that given a boat with a speed of 15 km/hr in still water,
    traveling 3.6 km downstream in 12 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_speed (boat_speed : ℝ) (downstream_distance : ℝ) (time_minutes : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_distance = 3.6)
  (h3 : time_minutes = 12) : 
  let time_hours : ℝ := time_minutes / 60
  let current_speed : ℝ := downstream_distance / time_hours - boat_speed
  current_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_speed_l3537_353709


namespace NUMINAMATH_CALUDE_binomial_200_200_l3537_353770

theorem binomial_200_200 : Nat.choose 200 200 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_200_200_l3537_353770


namespace NUMINAMATH_CALUDE_password_count_l3537_353744

/-- The number of case-insensitive English letters -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letters in the password -/
def num_password_letters : ℕ := 2

/-- The number of digits in the password -/
def num_password_digits : ℕ := 2

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of possible passwords -/
def num_passwords : ℕ := 
  permutations num_letters num_password_letters * permutations num_digits num_password_digits

theorem password_count : 
  num_passwords = permutations num_letters num_password_letters * permutations num_digits num_password_digits :=
by sorry

end NUMINAMATH_CALUDE_password_count_l3537_353744


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3537_353728

/-- Tetrahedron with given edge lengths -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BC : ℝ
  BD : ℝ
  CD : ℝ

/-- Volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- The theorem to be proved -/
theorem tetrahedron_volume (t : Tetrahedron) 
  (h1 : t.AB = 4)
  (h2 : t.AC = 5)
  (h3 : t.AD = 6)
  (h4 : t.BC = 2 * Real.sqrt 7)
  (h5 : t.BD = 5)
  (h6 : t.CD = Real.sqrt 34) :
  volume t = 6 * Real.sqrt 1301 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3537_353728


namespace NUMINAMATH_CALUDE_even_function_quadratic_l3537_353771

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 1, f a b x = f a b (-x)) →
  a + 2 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_quadratic_l3537_353771


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3537_353750

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 11) = 14 → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3537_353750


namespace NUMINAMATH_CALUDE_unique_prime_with_few_divisors_l3537_353773

theorem unique_prime_with_few_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ (Nat.card (Nat.divisors (p^2 + 11)) < 11) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_few_divisors_l3537_353773


namespace NUMINAMATH_CALUDE_A_inverse_proof_l3537_353710

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -5; -3, 2]
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-2, -5; -3, -7]

theorem A_inverse_proof : A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_A_inverse_proof_l3537_353710


namespace NUMINAMATH_CALUDE_triangle_transformation_result_l3537_353726

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

/-- Reflects a point over the x-axis -/
def reflectOverX (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Translates a point vertically by a given amount -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

/-- Applies all transformations to a single point -/
def applyAllTransformations (p : Point) : Point :=
  rotate180 (translateVertical (reflectOverX (rotate90Clockwise p)) 3)

/-- The main theorem stating the result of the transformations -/
theorem triangle_transformation_result :
  let initial := Triangle.mk ⟨1, 2⟩ ⟨4, 2⟩ ⟨1, 5⟩
  let final := Triangle.mk (applyAllTransformations initial.A)
                           (applyAllTransformations initial.B)
                           (applyAllTransformations initial.C)
  final = Triangle.mk ⟨-2, -4⟩ ⟨-2, -7⟩ ⟨-5, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_result_l3537_353726


namespace NUMINAMATH_CALUDE_root_product_sum_l3537_353729

theorem root_product_sum (p q r : ℝ) : 
  (4 * p^3 - 6 * p^2 + 17 * p - 10 = 0) →
  (4 * q^3 - 6 * q^2 + 17 * q - 10 = 0) →
  (4 * r^3 - 6 * r^2 + 17 * r - 10 = 0) →
  p * q + p * r + q * r = 17/4 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l3537_353729


namespace NUMINAMATH_CALUDE_johns_height_l3537_353790

/-- Given the heights of various people and their relationships, prove John's height. -/
theorem johns_height (carl becky amy helen angela tom mary john : ℝ) 
  (h1 : carl = 120)
  (h2 : becky = 2 * carl)
  (h3 : amy = 1.2 * becky)
  (h4 : helen = amy + 3)
  (h5 : angela = helen + 4)
  (h6 : tom = angela - 70)
  (h7 : mary = 2 * tom)
  (h8 : john = 1.5 * mary) : 
  john = 675 := by sorry

end NUMINAMATH_CALUDE_johns_height_l3537_353790


namespace NUMINAMATH_CALUDE_shared_triangle_angle_measure_l3537_353700

-- Define the angle measures
def angle1 : Real := 58
def angle2 : Real := 35
def angle3 : Real := 42

-- Define the theorem
theorem shared_triangle_angle_measure :
  ∃ (angle4 angle5 angle6 : Real),
    -- The sum of angles in the first triangle is 180°
    angle1 + angle2 + angle5 = 180 ∧
    -- The sum of angles in the second triangle is 180°
    angle3 + angle5 + angle6 = 180 ∧
    -- The sum of angles in the third triangle (with the unknown angle) is 180°
    angle4 + angle5 + angle6 = 180 ∧
    -- The measure of the unknown angle (angle4) is 135°
    angle4 = 135 := by
  sorry

end NUMINAMATH_CALUDE_shared_triangle_angle_measure_l3537_353700


namespace NUMINAMATH_CALUDE_expansion_simplification_l3537_353715

theorem expansion_simplification (a b : ℝ) : (a + b) * (3 * a - b) - b * (a - b) = 3 * a^2 + a * b := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l3537_353715


namespace NUMINAMATH_CALUDE_exponential_inequality_l3537_353792

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3537_353792


namespace NUMINAMATH_CALUDE_songs_learned_correct_l3537_353787

/-- The number of songs Vincent knew before summer camp -/
def songs_before : ℕ := 56

/-- The number of songs Vincent knows after summer camp -/
def songs_after : ℕ := 74

/-- The number of songs Vincent learned at summer camp -/
def songs_learned : ℕ := songs_after - songs_before

theorem songs_learned_correct : songs_learned = 18 := by sorry

end NUMINAMATH_CALUDE_songs_learned_correct_l3537_353787


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3537_353761

-- Define the arithmetic progression
def arithmetic_progression (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_progression a d n + d

-- Define the geometric progression
def geometric_progression (g_1 g_2 g_3 : ℝ) : Prop :=
  g_2 ^ 2 = g_1 * g_3

-- Theorem statement
theorem smallest_third_term_of_geometric_progression :
  ∀ d : ℝ,
  let a := arithmetic_progression 9 d
  let g_1 := 9
  let g_2 := a 1 + 5
  let g_3 := a 2 + 30
  geometric_progression g_1 g_2 g_3 →
  ∃ min_g_3 : ℝ, min_g_3 = 29 - 20 * Real.sqrt 2 ∧
  ∀ other_g_3 : ℝ, geometric_progression g_1 g_2 other_g_3 → min_g_3 ≤ other_g_3 :=
sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3537_353761


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3537_353780

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, a (n + 2) - a n = 6) :
  a 11 = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3537_353780


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3537_353794

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * (l + w) = 72) →  -- perimeter condition
  (l = 5/2 * w) →       -- ratio condition
  Real.sqrt (l^2 + w^2) = 194/7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3537_353794


namespace NUMINAMATH_CALUDE_cos_difference_value_l3537_353734

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by sorry

end NUMINAMATH_CALUDE_cos_difference_value_l3537_353734


namespace NUMINAMATH_CALUDE_triangle_x_coordinate_l3537_353722

/-- 
Given a triangle with vertices (x, 0), (7, 4), and (7, -4),
if the area of the triangle is 32, then x = -1.
-/
theorem triangle_x_coordinate (x : ℝ) : 
  let v1 : ℝ × ℝ := (x, 0)
  let v2 : ℝ × ℝ := (7, 4)
  let v3 : ℝ × ℝ := (7, -4)
  let base : ℝ := |v2.2 - v3.2|
  let height : ℝ := |7 - x|
  let area : ℝ := (1/2) * base * height
  area = 32 → x = -1 := by
sorry

end NUMINAMATH_CALUDE_triangle_x_coordinate_l3537_353722


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3537_353791

theorem power_of_two_equality (m : ℤ) : 
  2^1999 - 2^1998 - 2^1997 + 2^1996 - 2^1995 = m * 2^1995 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3537_353791


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3537_353735

def A : Set Char := {'a', 'b', 'c', 'd', 'e'}
def B : Set Char := {'d', 'f', 'g'}

theorem intersection_of_A_and_B :
  A ∩ B = {'d'} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3537_353735


namespace NUMINAMATH_CALUDE_centrally_symmetric_multiple_symmetry_axes_l3537_353717

/-- A polygon in a 2D plane. -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- Represents a line in a 2D plane. -/
structure Line where
  -- Add necessary fields for a line

/-- Predicate to check if a polygon is centrally symmetric. -/
def is_centrally_symmetric (p : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line is a symmetry axis of a polygon. -/
def is_symmetry_axis (l : Line) (p : Polygon) : Prop :=
  sorry

/-- The number of symmetry axes a polygon has. -/
def num_symmetry_axes (p : Polygon) : Nat :=
  sorry

/-- Theorem: A centrally symmetric polygon with at least one symmetry axis must have more than one symmetry axis. -/
theorem centrally_symmetric_multiple_symmetry_axes (p : Polygon) :
  is_centrally_symmetric p → (∃ l : Line, is_symmetry_axis l p) → num_symmetry_axes p > 1 :=
by sorry

end NUMINAMATH_CALUDE_centrally_symmetric_multiple_symmetry_axes_l3537_353717


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3537_353778

/-- The area of a square with adjacent vertices at (-1, 4) and (2, -3) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (-1, 4)
  let p2 : ℝ × ℝ := (2, -3)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  distance_squared = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3537_353778


namespace NUMINAMATH_CALUDE_quadrilateral_angle_combinations_l3537_353714

/-- Represents the type of an angle in a quadrilateral -/
inductive AngleType
| Acute
| Right
| Obtuse

/-- Represents a combination of angles in a quadrilateral -/
structure AngleCombination :=
  (acute : Nat)
  (right : Nat)
  (obtuse : Nat)

/-- A convex quadrilateral has exactly four angles -/
def total_angles : Nat := 4

/-- The sum of interior angles in a quadrilateral is 360 degrees -/
def angle_sum : Nat := 360

/-- Theorem: The only possible combinations of internal angles in a convex quadrilateral
    are the seven combinations listed. -/
theorem quadrilateral_angle_combinations :
  ∃ (valid_combinations : List AngleCombination),
    (valid_combinations.length = 7) ∧
    (∀ combo : AngleCombination,
      (combo.acute + combo.right + combo.obtuse = total_angles) →
      (combo.right * 90 + combo.acute * 89 + combo.obtuse * 91 ≤ angle_sum) →
      (combo.right * 90 + combo.acute * 1 + combo.obtuse * 91 ≥ angle_sum) →
      (combo ∈ valid_combinations)) ∧
    (∀ combo : AngleCombination,
      combo ∈ valid_combinations →
      (combo.acute + combo.right + combo.obtuse = total_angles) ∧
      (combo.right * 90 + combo.acute * 89 + combo.obtuse * 91 ≤ angle_sum) ∧
      (combo.right * 90 + combo.acute * 1 + combo.obtuse * 91 ≥ angle_sum)) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_combinations_l3537_353714


namespace NUMINAMATH_CALUDE_perfect_square_identification_l3537_353723

theorem perfect_square_identification :
  let a := 3^6 * 7^7 * 8^8
  let b := 3^8 * 7^6 * 8^7
  let c := 3^7 * 7^8 * 8^6
  let d := 3^7 * 7^7 * 8^8
  let e := 3^8 * 7^8 * 8^8
  ∃ n : ℕ, e = n^2 ∧ 
  (∀ m : ℕ, a ≠ m^2) ∧ 
  (∀ m : ℕ, b ≠ m^2) ∧ 
  (∀ m : ℕ, c ≠ m^2) ∧ 
  (∀ m : ℕ, d ≠ m^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_identification_l3537_353723


namespace NUMINAMATH_CALUDE_chocolate_division_theorem_l3537_353757

/-- Represents a piece of chocolate -/
structure ChocolatePiece where
  area : ℝ

/-- Represents the chocolate bar -/
structure ChocolateBar where
  length : ℝ
  width : ℝ
  pieces : Fin 4 → ChocolatePiece

/-- The chocolate bar is divided as described in the problem -/
def is_divided_as_described (bar : ChocolateBar) : Prop :=
  bar.length = 6 ∧ bar.width = 4 ∧
  ∃ (p1 p2 p3 p4 : ChocolatePiece),
    bar.pieces 0 = p1 ∧ bar.pieces 1 = p2 ∧ bar.pieces 2 = p3 ∧ bar.pieces 3 = p4 ∧
    p1.area + p2.area + p3.area + p4.area = bar.length * bar.width

/-- All pieces have equal area -/
def all_pieces_equal (bar : ChocolateBar) : Prop :=
  ∀ i j : Fin 4, (bar.pieces i).area = (bar.pieces j).area

/-- Main theorem: If the chocolate bar is divided as described, all pieces have equal area -/
theorem chocolate_division_theorem (bar : ChocolateBar) :
  is_divided_as_described bar → all_pieces_equal bar := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_theorem_l3537_353757


namespace NUMINAMATH_CALUDE_equation_solution_l3537_353720

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3537_353720


namespace NUMINAMATH_CALUDE_alice_original_seat_was_six_l3537_353707

/-- Represents the number of seats -/
def num_seats : Nat := 7

/-- Represents the seat Alice ends up in -/
def alice_final_seat : Nat := 4

/-- Represents the net movement of all other friends -/
def net_movement : Int := 2

/-- Calculates Alice's original seat given her final seat and the net movement of others -/
def alice_original_seat (final_seat : Nat) (net_move : Int) : Nat :=
  final_seat + net_move.toNat

/-- Theorem stating Alice's original seat was 6 -/
theorem alice_original_seat_was_six :
  alice_original_seat alice_final_seat net_movement = 6 := by
  sorry

#eval alice_original_seat alice_final_seat net_movement

end NUMINAMATH_CALUDE_alice_original_seat_was_six_l3537_353707


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3537_353767

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 - Complex.I)^2 ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3537_353767


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3537_353797

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 3*x₁ - 5 = 0 →
  x₂^2 - 3*x₂ - 5 = 0 →
  x₁ + x₂ - x₁ * x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3537_353797


namespace NUMINAMATH_CALUDE_garrison_provision_days_l3537_353762

/-- Calculates the initial number of days provisions were supposed to last for a garrison -/
def initialProvisionDays (initialGarrison : ℕ) (reinforcement : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  ((initialGarrison + reinforcement) * daysAfterReinforcement + initialGarrison * daysBeforeReinforcement) / initialGarrison

theorem garrison_provision_days :
  initialProvisionDays 1000 1250 15 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provision_days_l3537_353762


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3537_353718

/-- Calculates the speed of a train given the parameters of a passing goods train -/
theorem train_speed_calculation (goods_train_speed : ℝ) (goods_train_length : ℝ) (passing_time : ℝ) : 
  goods_train_speed = 108 →
  goods_train_length = 340 →
  passing_time = 8 →
  ∃ (man_train_speed : ℝ), man_train_speed = 45 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3537_353718


namespace NUMINAMATH_CALUDE_total_charge_2_hours_l3537_353732

/-- Represents the pricing structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyPricing where
  /-- The charge for the first hour of therapy -/
  first_hour_charge : ℕ
  /-- The charge for each additional hour of therapy -/
  additional_hour_charge : ℕ
  /-- The difference between the first hour charge and additional hour charge -/
  charge_difference : first_hour_charge = additional_hour_charge + 25
  /-- The total charge for 5 hours of therapy -/
  total_charge_5_hours : first_hour_charge + 4 * additional_hour_charge = 250

/-- Theorem stating that the total charge for 2 hours of therapy is $115 -/
theorem total_charge_2_hours (p : TherapyPricing) : 
  p.first_hour_charge + p.additional_hour_charge = 115 := by
  sorry


end NUMINAMATH_CALUDE_total_charge_2_hours_l3537_353732


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_line_l3537_353706

theorem min_sum_of_reciprocal_line (a b : ℝ) : 
  a > 0 → b > 0 → (1 : ℝ) / a + (1 : ℝ) / b = 1 → (a + b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_line_l3537_353706


namespace NUMINAMATH_CALUDE_smallest_M_inequality_l3537_353701

theorem smallest_M_inequality (a b c : ℝ) : ∃ (M : ℝ), 
  (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ M*(x^2 + y^2 + z^2)^2) ∧ 
  (M = (9 * Real.sqrt 2) / 64) ∧
  (∀ (N : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) → M ≤ N) :=
by sorry

end NUMINAMATH_CALUDE_smallest_M_inequality_l3537_353701


namespace NUMINAMATH_CALUDE_triangle_formation_l3537_353743

/-- Triangle inequality check for three sides -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 3 6 ∧
  ¬ can_form_triangle 2 5 7 ∧
  can_form_triangle 4 5 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3537_353743


namespace NUMINAMATH_CALUDE_hyperbola_intersection_range_l3537_353716

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def distinct_intersection (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂, x₁ ≠ x₂ →
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the dot product condition
def dot_product_condition (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂,
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ →
    x₁ * x₂ + y₁ * y₂ > 0

-- Main theorem
theorem hyperbola_intersection_range :
  ∀ k : ℝ, distinct_intersection k ∧ dot_product_condition k ↔
    (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_range_l3537_353716


namespace NUMINAMATH_CALUDE_average_apples_picked_l3537_353789

theorem average_apples_picked (maggie kelsey layla : ℕ) (h1 : maggie = 40) (h2 : kelsey = 28) (h3 : layla = 22) :
  (maggie + kelsey + layla) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_picked_l3537_353789


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3537_353737

/-- Given a geometric sequence {a_n} with a_1 = 1/2 and a_4 = 4, 
    the common ratio q is equal to 2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = 4) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3537_353737


namespace NUMINAMATH_CALUDE_total_amount_not_unique_l3537_353739

/-- Represents the investment scenario with two different interest rates -/
structure Investment where
  x : ℝ  -- Amount invested at 10%
  y : ℝ  -- Amount invested at 8%
  T : ℝ  -- Total amount invested

/-- The conditions of the investment problem -/
def investment_conditions (inv : Investment) : Prop :=
  inv.x * 0.10 - inv.y * 0.08 = 65 ∧ inv.x + inv.y = inv.T

/-- Theorem stating that the total amount T cannot be uniquely determined -/
theorem total_amount_not_unique :
  ∃ (inv1 inv2 : Investment), 
    investment_conditions inv1 ∧ 
    investment_conditions inv2 ∧ 
    inv1.T ≠ inv2.T :=
sorry

#check total_amount_not_unique

end NUMINAMATH_CALUDE_total_amount_not_unique_l3537_353739


namespace NUMINAMATH_CALUDE_bread_for_double_meat_sandwiches_l3537_353713

/-- Given the following conditions:
  - Two pieces of bread are needed for one regular sandwich.
  - Three pieces of bread are needed for a double meat sandwich.
  - There are 14 regular sandwiches.
  - A total of 64 pieces of bread are used.
Prove that the number of bread pieces used for double meat sandwiches is 36. -/
theorem bread_for_double_meat_sandwiches :
  let regular_sandwich_bread : ℕ := 2
  let double_meat_sandwich_bread : ℕ := 3
  let regular_sandwiches : ℕ := 14
  let total_bread : ℕ := 64
  let double_meat_bread := total_bread - regular_sandwich_bread * regular_sandwiches
  double_meat_bread = 36 := by
  sorry

end NUMINAMATH_CALUDE_bread_for_double_meat_sandwiches_l3537_353713


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3537_353779

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  -- Angle between faces ABC and BCD
  angle : ℝ
  -- Area of face ABC
  area_ABC : ℝ
  -- Area of face BCD
  area_BCD : ℝ
  -- Length of edge BC
  length_BC : ℝ

/-- The volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the tetrahedron with given properties -/
theorem tetrahedron_volume :
  ∀ t : Tetrahedron,
    t.angle = π/4 ∧
    t.area_ABC = 150 ∧
    t.area_BCD = 100 ∧
    t.length_BC = 12 →
    volume t = (1250 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3537_353779


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l3537_353775

/-- Hyperbola properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  eccentricity : ℝ
  vertex_to_asymptote : ℝ

/-- Theorem: Given a hyperbola with specific eccentricity and vertex-to-asymptote distance, prove its parameters -/
theorem hyperbola_parameters (h : Hyperbola) 
  (h_eccentricity : h.eccentricity = Real.sqrt 6 / 2)
  (h_vertex_to_asymptote : h.vertex_to_asymptote = 2 * Real.sqrt 6 / 3) :
  h.a = 2 * Real.sqrt 2 ∧ h.b = 2 := by
  sorry

#check hyperbola_parameters

end NUMINAMATH_CALUDE_hyperbola_parameters_l3537_353775


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l3537_353749

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : 1 ≤ a - 2*b ∧ a - 2*b ≤ 3) : 
  -11/3 ≤ a + 3*b ∧ a + 3*b ≤ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l3537_353749


namespace NUMINAMATH_CALUDE_root_sum_product_l3537_353764

theorem root_sum_product (p q : ℝ) : 
  (∃ x, x^4 - 6*x - 2 = 0) → 
  (p^4 - 6*p - 2 = 0) →
  (q^4 - 6*q - 2 = 0) →
  p ≠ q →
  pq + p + q = 1 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l3537_353764
