import Mathlib

namespace NUMINAMATH_CALUDE_count_odd_increasing_integers_l402_40200

/-- A three-digit integer with odd digits in strictly increasing order -/
structure OddIncreasingInteger where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_odd_hundreds : Odd hundreds
  is_odd_tens : Odd tens
  is_odd_ones : Odd ones
  is_increasing : hundreds < tens ∧ tens < ones
  is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000

/-- The count of three-digit integers with odd digits in strictly increasing order -/
def countOddIncreasingIntegers : Nat := sorry

/-- Theorem stating that there are exactly 10 three-digit integers with odd digits in strictly increasing order -/
theorem count_odd_increasing_integers :
  countOddIncreasingIntegers = 10 := by sorry

end NUMINAMATH_CALUDE_count_odd_increasing_integers_l402_40200


namespace NUMINAMATH_CALUDE_equation_solution_l402_40295

theorem equation_solution : 
  ∃ x : ℝ, (2 * x / (x - 2) + 3 / (2 - x) = 1) ∧ (x = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l402_40295


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l402_40227

theorem closest_integer_to_cube_root_150 : 
  ∀ n : ℤ, |n - (150 : ℝ)^(1/3)| ≥ |6 - (150 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l402_40227


namespace NUMINAMATH_CALUDE_max_value_of_expression_l402_40299

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^3*(b+c) + b^3*(c+a) + c^3*(a+b)) / ((a+b+c)^4 - 79*(a*b*c)^(4/3))
  A ≤ 3 ∧ (A = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l402_40299


namespace NUMINAMATH_CALUDE_base7UnitsDigitIs6_l402_40252

/-- The units digit of the base-7 representation of the product of 328 and 57 -/
def base7UnitsDigit : ℕ :=
  (328 * 57) % 7

/-- Theorem stating that the units digit of the base-7 representation of the product of 328 and 57 is 6 -/
theorem base7UnitsDigitIs6 : base7UnitsDigit = 6 := by
  sorry

end NUMINAMATH_CALUDE_base7UnitsDigitIs6_l402_40252


namespace NUMINAMATH_CALUDE_number_divided_by_three_l402_40241

theorem number_divided_by_three : ∃ x : ℝ, x / 3 = 3 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l402_40241


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l402_40280

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmeticSequence a d)
  (h_d_neg : d < 0)
  (h_prod : a 2 * a 4 = 12)
  (h_sum : a 2 + a 4 = 8) :
  ∀ n : ℕ+, a n = -2 * n + 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l402_40280


namespace NUMINAMATH_CALUDE_share_calculation_l402_40233

/-- Represents the share of each party in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
theorem share_calculation (s : Share) : 
  s.x + s.y + s.z = 175 →  -- Total sum is 175
  s.z = 0.3 * s.x →        -- z gets 0.3 for each rupee x gets
  s.x > 0 →                -- Ensure x's share is positive
  s.y = 173.7 :=           -- y's share is 173.7
by sorry

end NUMINAMATH_CALUDE_share_calculation_l402_40233


namespace NUMINAMATH_CALUDE_percentage_of_blue_shirts_l402_40292

theorem percentage_of_blue_shirts (total_students : ℕ) 
  (red_percent green_percent : ℚ) (other_count : ℕ) : 
  total_students = 700 →
  red_percent = 23/100 →
  green_percent = 15/100 →
  other_count = 119 →
  (1 - (red_percent + green_percent + (other_count : ℚ) / total_students)) * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_blue_shirts_l402_40292


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l402_40244

theorem quadratic_root_k_value : ∃ k : ℝ, 3^2 - k*3 - 6 = 0 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l402_40244


namespace NUMINAMATH_CALUDE_project_hours_l402_40237

theorem project_hours (x y z : ℕ) (h1 : y = (5 * x) / 3) (h2 : z = 2 * x) (h3 : z = x + 30) :
  x + y + z = 140 :=
by sorry

end NUMINAMATH_CALUDE_project_hours_l402_40237


namespace NUMINAMATH_CALUDE_joan_bought_72_eggs_l402_40298

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- Theorem: Joan bought 72 eggs -/
theorem joan_bought_72_eggs : dozens_bought * eggs_per_dozen = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_72_eggs_l402_40298


namespace NUMINAMATH_CALUDE_symmetric_parabolas_product_l402_40247

/-- Given two parabolas that are symmetric with respect to a line, 
    prove that the product of their parameters is -3 -/
theorem symmetric_parabolas_product (a p m : ℝ) : 
  a ≠ 0 → p > 0 → 
  (∀ x y : ℝ, y = a * x^2 - 3 * x + 3 ↔ 
    ∃ x' y', y' = x + m ∧ x = y' - m ∧ y'^2 = 2 * p * x') →
  a * p * m = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_parabolas_product_l402_40247


namespace NUMINAMATH_CALUDE_triangle_angle_sine_equivalence_l402_40293

theorem triangle_angle_sine_equivalence (A B C : Real) (h : A > 0 ∧ B > 0 ∧ C > 0) :
  (A > B ↔ Real.sin A > Real.sin B) :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_equivalence_l402_40293


namespace NUMINAMATH_CALUDE_little_red_journey_l402_40206

-- Define the parameters
def total_distance : ℝ := 1500  -- in meters
def total_time : ℝ := 18  -- in minutes
def uphill_speed : ℝ := 2  -- in km/h
def downhill_speed : ℝ := 3  -- in km/h

-- Define variables for uphill and downhill time
variable (x y : ℝ)

-- Theorem statement
theorem little_red_journey :
  (x + y = total_time) ∧
  ((uphill_speed / 60) * x + (downhill_speed / 60) * y = total_distance / 1000) :=
by sorry

end NUMINAMATH_CALUDE_little_red_journey_l402_40206


namespace NUMINAMATH_CALUDE_complex_equality_l402_40224

theorem complex_equality (z : ℂ) : z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l402_40224


namespace NUMINAMATH_CALUDE_rationalize_denominator_l402_40254

theorem rationalize_denominator : 7 / Real.sqrt 98 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l402_40254


namespace NUMINAMATH_CALUDE_pascals_triangle_25th_number_l402_40226

theorem pascals_triangle_25th_number (n : ℕ) (k : ℕ) : 
  n = 27 ∧ k = 24 → Nat.choose n k = 2925 :=
by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_25th_number_l402_40226


namespace NUMINAMATH_CALUDE_fourth_animal_is_sheep_l402_40287

def animals : List String := ["Horses", "Cows", "Pigs", "Sheep", "Rabbits", "Squirrels"]

theorem fourth_animal_is_sheep : animals[3] = "Sheep" := by
  sorry

end NUMINAMATH_CALUDE_fourth_animal_is_sheep_l402_40287


namespace NUMINAMATH_CALUDE_line_equation_problem_l402_40212

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem
theorem line_equation_problem (l : Line) (P : Point) :
  (P.x = 2 ∧ P.y = 3) →
  (
    (l.slope = -Real.sqrt 3) ∨
    (l.slope = -2) ∨
    (l.slope = 3/2 ∧ l.y_intercept = 0) ∨
    (l.slope = 1 ∧ l.y_intercept = -1)
  ) →
  (
    (Real.sqrt 3 * P.x + P.y - 3 - 2 * Real.sqrt 3 = 0) ∨
    (2 * P.x + P.y - 7 = 0) ∨
    (3 * P.x - 2 * P.y = 0) ∨
    (P.x - P.y + 1 = 0)
  ) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_problem_l402_40212


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l402_40283

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l402_40283


namespace NUMINAMATH_CALUDE_work_completion_proof_l402_40216

/-- The original number of men working on a task -/
def original_men : ℕ := 20

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 10

/-- The number of men removed from the original group -/
def removed_men : ℕ := 10

/-- The number of additional days it takes to complete the work with fewer men -/
def additional_days : ℕ := 10

theorem work_completion_proof :
  (original_men * original_days = (original_men - removed_men) * (original_days + additional_days)) →
  original_men = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_proof_l402_40216


namespace NUMINAMATH_CALUDE_min_values_theorem_l402_40263

theorem min_values_theorem (a b c : ℕ+) (h : a^2 + b^2 - c^2 = 2018) :
  (∀ x y z : ℕ+, x^2 + y^2 - z^2 = 2018 → a + b - c ≤ x + y - z) ∧
  (∀ x y z : ℕ+, x^2 + y^2 - z^2 = 2018 → a + b + c ≤ x + y + z) ∧
  a + b - c = 2 ∧ a + b + c = 52 :=
sorry

end NUMINAMATH_CALUDE_min_values_theorem_l402_40263


namespace NUMINAMATH_CALUDE_unique_c_complex_magnitude_l402_40250

theorem unique_c_complex_magnitude : ∃! c : ℝ, Complex.abs (1 - (c + 1) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_complex_magnitude_l402_40250


namespace NUMINAMATH_CALUDE_max_correct_is_23_l402_40232

/-- Represents the scoring system and Amy's exam results -/
structure ExamResults where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResults) : ℕ :=
  sorry

/-- Theorem stating that given the exam conditions, the maximum number of correct answers is 23 -/
theorem max_correct_is_23 (exam : ExamResults) 
  (h1 : exam.total_questions = 30)
  (h2 : exam.correct_score = 4)
  (h3 : exam.incorrect_score = -1)
  (h4 : exam.total_score = 85) :
  max_correct_answers exam = 23 :=
sorry

end NUMINAMATH_CALUDE_max_correct_is_23_l402_40232


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l402_40271

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = 4/3 * 90
  -- One angle is 36° larger than the other
  → b = a + 36
  -- All angles are non-negative
  → a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0
  -- Sum of all angles in a triangle is 180°
  → a + b + c = 180
  -- The largest angle is 78°
  → max a (max b c) = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l402_40271


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l402_40270

-- Define the set of all functions
variable (F : Type)

-- Define the property of being a logarithmic function
variable (isLogarithmic : F → Prop)

-- Define the property of being a monotonic function
variable (isMonotonic : F → Prop)

-- The theorem to prove
theorem negation_of_universal_proposition :
  (¬ ∀ f : F, isLogarithmic f → isMonotonic f) ↔ 
  (∃ f : F, isLogarithmic f ∧ ¬isMonotonic f) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l402_40270


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l402_40219

theorem quadratic_is_perfect_square :
  ∃ (a b : ℝ), ∀ x, 9 * x^2 - 30 * x + 25 = (a * x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l402_40219


namespace NUMINAMATH_CALUDE_rectangle_construction_solutions_l402_40236

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomRight : Point
  bottomLeft : Point

/-- Check if a point lies on any side of the rectangle -/
def pointOnRectangle (p : Point) (r : Rectangle) : Prop :=
  (p.x = r.topLeft.x ∧ p.y ≥ r.bottomLeft.y ∧ p.y ≤ r.topLeft.y) ∨
  (p.x = r.topRight.x ∧ p.y ≥ r.bottomRight.y ∧ p.y ≤ r.topRight.y) ∨
  (p.y = r.topLeft.y ∧ p.x ≥ r.topLeft.x ∧ p.x ≤ r.topRight.x) ∨
  (p.y = r.bottomLeft.y ∧ p.x ≥ r.bottomLeft.x ∧ p.x ≤ r.bottomRight.x)

/-- Check if the rectangle has a side of length 'a' -/
def hasLengthA (r : Rectangle) (a : ℝ) : Prop :=
  (r.topRight.x - r.topLeft.x = a) ∨
  (r.topRight.y - r.bottomRight.y = a)

/-- The main theorem -/
theorem rectangle_construction_solutions 
  (A B C D : Point) (a : ℝ) (h : a > 0) :
  ∃ (solutions : Finset Rectangle), 
    solutions.card = 12 ∧
    ∀ r ∈ solutions, 
      pointOnRectangle A r ∧
      pointOnRectangle B r ∧
      pointOnRectangle C r ∧
      pointOnRectangle D r ∧
      hasLengthA r a :=
sorry

end NUMINAMATH_CALUDE_rectangle_construction_solutions_l402_40236


namespace NUMINAMATH_CALUDE_reflect_x_axis_l402_40282

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflect_x_axis (p : Point) (h : p = Point.mk 3 1) :
  reflectX p = Point.mk 3 (-1) := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_axis_l402_40282


namespace NUMINAMATH_CALUDE_age_problem_l402_40242

/-- The age problem involving Sebastian, his siblings, and their father. -/
theorem age_problem (sebastian_age : ℕ) (sister_age_diff : ℕ) (brother_age_diff : ℕ) : 
  sebastian_age = 40 →
  sister_age_diff = 10 →
  brother_age_diff = 7 →
  (sebastian_age - 5 + (sebastian_age - sister_age_diff - 5) + 
   (sebastian_age - sister_age_diff - brother_age_diff - 5) : ℚ) = 
   (3 / 4 : ℚ) * ((109 : ℕ) - 5) →
  109 = sebastian_age + 69 := by
  sorry

#check age_problem

end NUMINAMATH_CALUDE_age_problem_l402_40242


namespace NUMINAMATH_CALUDE_line_parameterization_values_l402_40204

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 3 * x + 2

/-- The parameterization of the line -/
def parameterization (t s m : ℝ) : ℝ × ℝ :=
  (-4 + t * m, s + t * (-7))

/-- Theorem stating the values of s and m for the given line and parameterization -/
theorem line_parameterization_values :
  ∃ (s m : ℝ), 
    (∀ t, line_equation (parameterization t s m).1 (parameterization t s m).2) ∧
    s = -10 ∧ 
    m = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_values_l402_40204


namespace NUMINAMATH_CALUDE_rate_categories_fractions_l402_40217

/-- Represents the three rate categories for electricity usage --/
inductive RateCategory
  | A
  | B
  | C

/-- Total hours in a week --/
def hoursInWeek : ℕ := 7 * 24

/-- Hours that Category A applies in a week --/
def categoryAHours : ℕ := 12 * 5

/-- Hours that Category B applies in a week --/
def categoryBHours : ℕ := 10 * 2

/-- Hours that Category C applies in a week --/
def categoryCHours : ℕ := hoursInWeek - (categoryAHours + categoryBHours)

/-- Function to get the fraction of the week a category applies to --/
def categoryFraction (c : RateCategory) : ℚ :=
  match c with
  | RateCategory.A => categoryAHours / hoursInWeek
  | RateCategory.B => categoryBHours / hoursInWeek
  | RateCategory.C => categoryCHours / hoursInWeek

theorem rate_categories_fractions :
  categoryFraction RateCategory.A = 5 / 14 ∧
  categoryFraction RateCategory.B = 5 / 42 ∧
  categoryFraction RateCategory.C = 11 / 21 ∧
  categoryFraction RateCategory.A + categoryFraction RateCategory.B + categoryFraction RateCategory.C = 1 := by
  sorry


end NUMINAMATH_CALUDE_rate_categories_fractions_l402_40217


namespace NUMINAMATH_CALUDE_elena_garden_petals_l402_40214

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals on each lily -/
def petals_per_lily : ℕ := 6

/-- The number of petals on each tulip -/
def petals_per_tulip : ℕ := 3

/-- The total number of flower petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end NUMINAMATH_CALUDE_elena_garden_petals_l402_40214


namespace NUMINAMATH_CALUDE_shirts_not_washed_l402_40209

theorem shirts_not_washed 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 40)
  (h2 : long_sleeve = 23)
  (h3 : washed = 29) :
  short_sleeve + long_sleeve - washed = 34 := by
sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l402_40209


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l402_40267

/-- The area of the shaded region formed by semicircles -/
theorem shaded_area_semicircles (UV VW WX XY YZ : ℝ) 
  (h_UV : UV = 3) 
  (h_VW : VW = 5) 
  (h_WX : WX = 4) 
  (h_XY : XY = 6) 
  (h_YZ : YZ = 7) : 
  let UZ := UV + VW + WX + XY + YZ
  let area_large := (π / 8) * UZ^2
  let area_small := (π / 8) * (UV^2 + VW^2 + WX^2 + XY^2 + YZ^2)
  area_large - area_small = (247 / 4) * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l402_40267


namespace NUMINAMATH_CALUDE_averageIs295_l402_40222

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 295 -/
theorem averageIs295 (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 570) (h2 : otherDayVisitors = 240) : 
    averageVisitorsPerDay sundayVisitors otherDayVisitors = 295 := by
  sorry

end NUMINAMATH_CALUDE_averageIs295_l402_40222


namespace NUMINAMATH_CALUDE_x_younger_than_w_l402_40262

-- Define the ages of the individuals
variable (w_years x_years y_years z_years : ℤ)

-- Define the conditions
axiom sum_condition : w_years + x_years = y_years + z_years + 15
axiom difference_condition : |w_years - x_years| = 2 * |y_years - z_years|
axiom w_z_relation : w_years = z_years + 30

-- Theorem to prove
theorem x_younger_than_w : x_years = w_years - 45 := by
  sorry

end NUMINAMATH_CALUDE_x_younger_than_w_l402_40262


namespace NUMINAMATH_CALUDE_connors_garage_wheels_l402_40266

/-- The number of wheels in Connor's garage -/
def total_wheels (bicycles cars motorcycles : ℕ) : ℕ :=
  2 * bicycles + 4 * cars + 2 * motorcycles

/-- Theorem: The total number of wheels in Connor's garage is 90 -/
theorem connors_garage_wheels :
  total_wheels 20 10 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_connors_garage_wheels_l402_40266


namespace NUMINAMATH_CALUDE_september_first_was_wednesday_l402_40249

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of lessons Vasya skips on a given day -/
def lessonsSkipped (day : DayOfWeek) : Nat :=
  match day with
  | DayOfWeek.Monday => 1
  | DayOfWeek.Tuesday => 2
  | DayOfWeek.Wednesday => 3
  | DayOfWeek.Thursday => 4
  | DayOfWeek.Friday => 5
  | _ => 0

/-- Calculates the day of the week for a given date in September -/
def dayOfWeekForDate (date : Nat) (sept1 : DayOfWeek) : DayOfWeek :=
  sorry

/-- Calculates the total number of lessons Vasya skipped in September -/
def totalLessonsSkipped (sept1 : DayOfWeek) : Nat :=
  sorry

theorem september_first_was_wednesday :
  totalLessonsSkipped DayOfWeek.Wednesday = 64 :=
by sorry

end NUMINAMATH_CALUDE_september_first_was_wednesday_l402_40249


namespace NUMINAMATH_CALUDE_distribute_four_teachers_three_schools_l402_40248

/-- Number of ways to distribute n distinct teachers among k distinct schools,
    with each school receiving at least one teacher -/
def distribute_teachers (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 4 distinct teachers among 3 distinct schools,
    with each school receiving at least one teacher, is 36 -/
theorem distribute_four_teachers_three_schools :
  distribute_teachers 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_four_teachers_three_schools_l402_40248


namespace NUMINAMATH_CALUDE_soccer_team_strikers_l402_40218

theorem soccer_team_strikers (goalies defenders midfielders strikers total : ℕ) : 
  goalies = 3 →
  defenders = 10 →
  midfielders = 2 * defenders →
  total = 40 →
  strikers = total - (goalies + defenders + midfielders) →
  strikers = 7 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_strikers_l402_40218


namespace NUMINAMATH_CALUDE_equation_solution_l402_40291

theorem equation_solution : ∃ x : ℝ, (x / (2 * x - 3) + 5 / (3 - 2 * x) = 4) ∧ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l402_40291


namespace NUMINAMATH_CALUDE_two_vertices_same_degree_l402_40296

-- Define a graph
def Graph (α : Type) := α → α → Prop

-- Define the degree of a vertex in a graph
def degree {α : Type} (G : Graph α) (v : α) : ℕ := sorry

theorem two_vertices_same_degree {α : Type} (G : Graph α) (n : ℕ) (h : Fintype α) :
  (Fintype.card α = n) →
  (∀ v : α, degree G v < n) →
  ∃ u v : α, u ≠ v ∧ degree G u = degree G v :=
sorry

end NUMINAMATH_CALUDE_two_vertices_same_degree_l402_40296


namespace NUMINAMATH_CALUDE_inequality_system_solution_l402_40279

theorem inequality_system_solution (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l402_40279


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_ratio_l402_40281

/-- An inscribed acute-angled isosceles triangle in a circle -/
structure IsoscelesTriangle (α : ℝ) :=
  (angle_base : 0 < α ∧ α < π/2)

/-- An inscribed trapezoid in a circle -/
structure Trapezoid (α : ℝ) :=
  (base_is_diameter : True)
  (sides_parallel_to_triangle : True)

/-- The theorem stating that the area of the trapezoid equals the area of the triangle -/
theorem trapezoid_triangle_area_ratio 
  (α : ℝ) 
  (triangle : IsoscelesTriangle α) 
  (trapezoid : Trapezoid α) : 
  ∃ (area_trapezoid area_triangle : ℝ), 
    area_trapezoid / area_triangle = 1 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_ratio_l402_40281


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l402_40253

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 15.999

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ :=
  hydrogen_count * hydrogen_weight +
  chlorine_count * chlorine_weight +
  oxygen_count * oxygen_weight

theorem compound_molecular_weight :
  molecular_weight = 68.456 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l402_40253


namespace NUMINAMATH_CALUDE_matrix_commutation_fraction_l402_40211

/-- Given two matrices A and B, where A is fixed and B has variable entries,
    if AB = BA and 4b ≠ c, then (a - d) / (c - 4b) = 3/8 -/
theorem matrix_commutation_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ c) → ((a - d) / (c - 4 * b) = 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_fraction_l402_40211


namespace NUMINAMATH_CALUDE_abc_inequality_l402_40285

theorem abc_inequality (a b c : ℝ) (sum_eq_one : a + b + c = 1) (prod_pos : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l402_40285


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l402_40256

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℕ, x n = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l402_40256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l402_40260

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l402_40260


namespace NUMINAMATH_CALUDE_fair_division_of_walls_l402_40215

/-- The number of people in Amanda's family -/
def family_size : ℕ := 5

/-- The number of rooms with 4 walls -/
def rooms_with_4_walls : ℕ := 5

/-- The number of rooms with 5 walls -/
def rooms_with_5_walls : ℕ := 4

/-- The total number of walls in the house -/
def total_walls : ℕ := rooms_with_4_walls * 4 + rooms_with_5_walls * 5

/-- The number of walls each person should paint for fair division -/
def walls_per_person : ℕ := total_walls / family_size

theorem fair_division_of_walls :
  walls_per_person = 8 := by sorry

end NUMINAMATH_CALUDE_fair_division_of_walls_l402_40215


namespace NUMINAMATH_CALUDE_m_range_l402_40255

-- Define the condition function
def condition (m : ℝ) : Set ℝ := {x | 1 - m < x ∧ x < 1 + m}

-- Define the inequality function
def inequality : Set ℝ := {x | (x - 1)^2 < 1}

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, (condition m ⊆ inequality ∧ condition m ≠ inequality) → m ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l402_40255


namespace NUMINAMATH_CALUDE_sum_of_segments_9_9_l402_40286

/-- The sum of lengths of all line segments formed by dividing a line segment into equal parts -/
def sum_of_segments (total_length : ℕ) (num_divisions : ℕ) : ℕ :=
  let unit_length := total_length / num_divisions
  let sum_short_segments := (num_divisions - 1) * num_divisions * unit_length
  let sum_long_segments := (num_divisions * (num_divisions + 1) * unit_length) / 2
  sum_short_segments + sum_long_segments

/-- Theorem: The sum of lengths of all line segments formed by dividing a line segment of length 9 into 9 equal parts is equal to 165 -/
theorem sum_of_segments_9_9 :
  sum_of_segments 9 9 = 165 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_segments_9_9_l402_40286


namespace NUMINAMATH_CALUDE_investment_ratio_proof_l402_40208

/-- Represents the investment and return for an investor -/
structure Investor where
  investment : ℝ
  returnRate : ℝ

/-- Proves that the ratio of investments is 6:5:4 given the problem conditions -/
theorem investment_ratio_proof 
  (a b c : Investor)
  (return_ratio : a.returnRate / b.returnRate = 6/5 ∧ b.returnRate / c.returnRate = 5/4)
  (b_earns_more : b.investment * b.returnRate = a.investment * a.returnRate + 100)
  (total_earnings : a.investment * a.returnRate + b.investment * b.returnRate + c.investment * c.returnRate = 2900)
  : a.investment / b.investment = 6/5 ∧ b.investment / c.investment = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_proof_l402_40208


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l402_40259

theorem simplify_and_evaluate (a b : ℝ) : 
  (a^2 + a - 6 = 0) → 
  (b^2 + b - 6 = 0) → 
  a ≠ b →
  ((a / (a^2 - b^2) - 1 / (a + b)) / (1 / (a^2 - a * b))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l402_40259


namespace NUMINAMATH_CALUDE_subset_pairs_count_l402_40284

/-- Given a fixed set S with n elements, this theorem states that the number of ordered pairs (A, B) 
    where A and B are subsets of S and A ⊆ B is equal to 3^n. -/
theorem subset_pairs_count (n : ℕ) : 
  (Finset.powerset (Finset.range n)).card = 3^n := by sorry

end NUMINAMATH_CALUDE_subset_pairs_count_l402_40284


namespace NUMINAMATH_CALUDE_meadowood_58_impossible_l402_40223

/-- Represents the village of Meadowood with its animal and people relationships -/
structure Meadowood where
  sheep : ℕ
  horses : ℕ
  ducks : ℕ := 5 * sheep
  cows : ℕ := 2 * horses
  people : ℕ := 4 * ducks

/-- The total population in Meadowood -/
def Meadowood.total (m : Meadowood) : ℕ :=
  m.people + m.horses + m.sheep + m.cows + m.ducks

/-- Theorem stating that 58 cannot be the total population in Meadowood -/
theorem meadowood_58_impossible : ¬∃ m : Meadowood, m.total = 58 := by
  sorry

end NUMINAMATH_CALUDE_meadowood_58_impossible_l402_40223


namespace NUMINAMATH_CALUDE_combined_males_below_50_l402_40275

/-- Represents an office branch with employee information -/
structure Branch where
  total_employees : ℕ
  male_percentage : ℚ
  male_over_50_percentage : ℚ

/-- Calculates the number of males below 50 in a branch -/
def males_below_50 (b : Branch) : ℚ :=
  b.total_employees * b.male_percentage * (1 - b.male_over_50_percentage)

/-- The given information about the three branches -/
def branch_A : Branch :=
  { total_employees := 4500
  , male_percentage := 60 / 100
  , male_over_50_percentage := 40 / 100 }

def branch_B : Branch :=
  { total_employees := 3500
  , male_percentage := 50 / 100
  , male_over_50_percentage := 55 / 100 }

def branch_C : Branch :=
  { total_employees := 2200
  , male_percentage := 35 / 100
  , male_over_50_percentage := 70 / 100 }

/-- The main theorem stating the combined number of males below 50 -/
theorem combined_males_below_50 :
  ⌊males_below_50 branch_A + males_below_50 branch_B + males_below_50 branch_C⌋ = 2638 := by
  sorry

end NUMINAMATH_CALUDE_combined_males_below_50_l402_40275


namespace NUMINAMATH_CALUDE_f_is_linear_l402_40245

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

def f (x : ℝ) : ℝ := -2 * x

theorem f_is_linear : is_linear f := by sorry

end NUMINAMATH_CALUDE_f_is_linear_l402_40245


namespace NUMINAMATH_CALUDE_x_sixth_power_is_one_l402_40289

theorem x_sixth_power_is_one (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_power_is_one_l402_40289


namespace NUMINAMATH_CALUDE_complex_equation_solution_l402_40243

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 + z) → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l402_40243


namespace NUMINAMATH_CALUDE_arithmetic_mean_log_implies_geometric_mean_but_not_conversely_l402_40264

open Real

theorem arithmetic_mean_log_implies_geometric_mean_but_not_conversely 
  (x y z : ℝ) : 
  (2 * log y = log x + log z → y ^ 2 = x * z) ∧
  ¬(y ^ 2 = x * z → 2 * log y = log x + log z) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_log_implies_geometric_mean_but_not_conversely_l402_40264


namespace NUMINAMATH_CALUDE_part_one_part_two_l402_40276

-- Define the new operation *
def star (a b : ℚ) : ℚ := 4 * a * b

-- Theorem for part (1)
theorem part_one : star 3 (-4) = -48 := by sorry

-- Theorem for part (2)
theorem part_two : star (-2) (star 6 3) = -576 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l402_40276


namespace NUMINAMATH_CALUDE_contest_prime_problem_l402_40203

theorem contest_prime_problem : ∃! p : ℕ,
  Prime p ∧
  100 < p ∧ p < 500 ∧
  (∃ e : ℕ,
    e > 100 ∧
    e ≡ 2016 [ZMOD (p - 1)] ∧
    e - (p - 1) / 2 = 21 ∧
    2^2016 ≡ -(2^21) [ZMOD p]) ∧
  p = 211 := by
sorry

end NUMINAMATH_CALUDE_contest_prime_problem_l402_40203


namespace NUMINAMATH_CALUDE_mean_temperature_l402_40272

def temperatures : List ℝ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = 85.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l402_40272


namespace NUMINAMATH_CALUDE_five_thirteenths_period_l402_40257

def decimal_expansion_period (n d : ℕ) : ℕ :=
  sorry

theorem five_thirteenths_period :
  decimal_expansion_period 5 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_thirteenths_period_l402_40257


namespace NUMINAMATH_CALUDE_runners_meeting_time_l402_40274

/-- Represents a runner with their lap time in minutes -/
structure Runner where
  name : String
  lapTime : Nat

/-- Calculates the earliest time (in minutes) when all runners meet at the starting point -/
def earliestMeetingTime (runners : List Runner) : Nat :=
  sorry

theorem runners_meeting_time :
  let runners : List Runner := [
    { name := "Laura", lapTime := 5 },
    { name := "Maria", lapTime := 8 },
    { name := "Charlie", lapTime := 10 },
    { name := "Zoe", lapTime := 2 }
  ]
  earliestMeetingTime runners = 40 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l402_40274


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l402_40273

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 2 → 
  exterior_angle = 20 → 
  (n : ℝ) * exterior_angle = 360 →
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l402_40273


namespace NUMINAMATH_CALUDE_octal_subtraction_example_l402_40221

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSubtract (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_subtraction_example :
  octalSubtract (toOctal 641) (toOctal 324) = toOctal 317 := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_example_l402_40221


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l402_40228

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∀ t : ℕ, t > 0 → t ∣ n → t ≤ 12 ∧ 12 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l402_40228


namespace NUMINAMATH_CALUDE_distance_between_centers_l402_40258

/-- Two circles in the first quadrant, both tangent to both coordinate axes and passing through (4,1) -/
structure TangentCircles where
  C₁ : ℝ × ℝ  -- Center of first circle
  C₂ : ℝ × ℝ  -- Center of second circle
  h₁ : C₁.1 = C₁.2  -- Centers lie on angle bisector
  h₂ : C₂.1 = C₂.2  -- Centers lie on angle bisector
  h₃ : C₁.1 > 0 ∧ C₁.2 > 0  -- First circle in first quadrant
  h₄ : C₂.1 > 0 ∧ C₂.2 > 0  -- Second circle in first quadrant
  h₅ : (C₁.1 - 4)^2 + (C₁.2 - 1)^2 = C₁.1^2  -- First circle passes through (4,1)
  h₆ : (C₂.1 - 4)^2 + (C₂.2 - 1)^2 = C₂.1^2  -- Second circle passes through (4,1)

/-- The distance between the centers of two tangent circles is 8 -/
theorem distance_between_centers (tc : TangentCircles) : 
  Real.sqrt ((tc.C₁.1 - tc.C₂.1)^2 + (tc.C₁.2 - tc.C₂.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l402_40258


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l402_40234

theorem sum_of_two_numbers (x y : ℝ) : 
  0.5 * x + 0.3333 * y = 11 → 
  max x y = 15 → 
  x + y = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l402_40234


namespace NUMINAMATH_CALUDE_binary_to_septal_conversion_l402_40246

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its septal (base 7) representation -/
def decimal_to_septal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_to_septal_conversion :
  let binary := [true, false, true, false, true, true]
  let decimal := binary_to_decimal binary
  let septal := decimal_to_septal decimal
  decimal = 53 ∧ septal = [1, 0, 4] :=
by sorry

end NUMINAMATH_CALUDE_binary_to_septal_conversion_l402_40246


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_quadratic_form_components_l402_40269

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (x + 1)^2 + (x - 2) * (x + 2) = 1 ↔ 2 * x^2 + 2 * x - 4 = 0 :=
by sorry

-- Definitions for the components of the quadratic equation
def quadratic_term (x : ℝ) : ℝ := 2 * x^2
def quadratic_coefficient : ℝ := 2
def linear_term (x : ℝ) : ℝ := 2 * x
def linear_coefficient : ℝ := 2
def constant_term : ℝ := -4

-- Theorem stating that the transformed equation is in the general form of a quadratic equation
theorem quadratic_form_components (x : ℝ) :
  2 * x^2 + 2 * x - 4 = quadratic_term x + linear_term x + constant_term :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_quadratic_form_components_l402_40269


namespace NUMINAMATH_CALUDE_rosie_pies_l402_40290

/-- Represents the number of pies that can be made given ingredients and their ratios -/
def pies_made (apples_per_pie oranges_per_pie available_apples available_oranges : ℚ) : ℚ :=
  min (available_apples / apples_per_pie) (available_oranges / oranges_per_pie)

/-- Theorem stating that Rosie can make 9 pies with the given ingredients -/
theorem rosie_pies :
  let apples_per_pie : ℚ := 12 / 3
  let oranges_per_pie : ℚ := 6 / 3
  let available_apples : ℚ := 36
  let available_oranges : ℚ := 18
  pies_made apples_per_pie oranges_per_pie available_apples available_oranges = 9 := by
  sorry

#eval pies_made (12 / 3) (6 / 3) 36 18

end NUMINAMATH_CALUDE_rosie_pies_l402_40290


namespace NUMINAMATH_CALUDE_mika_stickers_problem_l402_40235

/-- The number of stickers Mika's mother gave her -/
def mothers_stickers (initial : Float) (bought : Float) (birthday : Float) (sister : Float) (final_total : Float) : Float :=
  final_total - (initial + bought + birthday + sister)

theorem mika_stickers_problem (initial : Float) (bought : Float) (birthday : Float) (sister : Float) (final_total : Float)
  (h1 : initial = 20.0)
  (h2 : bought = 26.0)
  (h3 : birthday = 20.0)
  (h4 : sister = 6.0)
  (h5 : final_total = 130.0) :
  mothers_stickers initial bought birthday sister final_total = 58.0 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_problem_l402_40235


namespace NUMINAMATH_CALUDE_four_dice_same_number_probability_l402_40231

/-- The probability of a single die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same number -/
def all_same_prob : ℚ := single_die_prob ^ (num_dice - 1)

theorem four_dice_same_number_probability :
  all_same_prob = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_number_probability_l402_40231


namespace NUMINAMATH_CALUDE_wednesday_work_time_l402_40297

/-- Represents the work time in minutes for each day of the week -/
structure WorkWeek where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- Calculates the total work time for the week in minutes -/
def totalWorkTime (w : WorkWeek) : ℚ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Converts hours to minutes -/
def hoursToMinutes (hours : ℚ) : ℚ :=
  hours * 60

theorem wednesday_work_time (w : WorkWeek) : 
  w.monday = hoursToMinutes (3/4) ∧ 
  w.tuesday = hoursToMinutes (1/2) ∧ 
  w.thursday = hoursToMinutes (5/6) ∧ 
  w.friday = 75 ∧ 
  totalWorkTime w = hoursToMinutes 4 → 
  w.wednesday = 40 := by
sorry

end NUMINAMATH_CALUDE_wednesday_work_time_l402_40297


namespace NUMINAMATH_CALUDE_beef_not_used_in_soup_l402_40210

-- Define the variables
def total_beef : ℝ := 4
def vegetables_used : ℝ := 6

-- Define the theorem
theorem beef_not_used_in_soup :
  ∃ (beef_used beef_not_used : ℝ),
    beef_used = vegetables_used / 2 ∧
    beef_not_used = total_beef - beef_used ∧
    beef_not_used = 1 := by
  sorry

end NUMINAMATH_CALUDE_beef_not_used_in_soup_l402_40210


namespace NUMINAMATH_CALUDE_sin_alpha_eq_neg_half_l402_40213

theorem sin_alpha_eq_neg_half (α : Real) 
  (h : Real.sin (α/2 - Real.pi/4) * Real.cos (α/2 + Real.pi/4) = -3/4) : 
  Real.sin α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_eq_neg_half_l402_40213


namespace NUMINAMATH_CALUDE_quartic_sum_l402_40201

/-- A quartic polynomial Q with specific values at 0, 1, and -1 -/
def QuarticPolynomial (m : ℝ) : ℝ → ℝ := sorry

/-- Properties of the QuarticPolynomial -/
axiom quartic_prop_0 (m : ℝ) : QuarticPolynomial m 0 = m
axiom quartic_prop_1 (m : ℝ) : QuarticPolynomial m 1 = 3 * m
axiom quartic_prop_neg1 (m : ℝ) : QuarticPolynomial m (-1) = 2 * m

/-- Theorem: For a quartic polynomial Q with Q(0) = m, Q(1) = 3m, and Q(-1) = 2m, Q(3) + Q(-3) = 56m -/
theorem quartic_sum (m : ℝ) : 
  QuarticPolynomial m 3 + QuarticPolynomial m (-3) = 56 * m := by sorry

end NUMINAMATH_CALUDE_quartic_sum_l402_40201


namespace NUMINAMATH_CALUDE_cubic_equation_integer_roots_l402_40277

theorem cubic_equation_integer_roots :
  ∀ x : ℤ, x^3 - 3*x^2 - 10*x + 20 = 0 ↔ x = -2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_roots_l402_40277


namespace NUMINAMATH_CALUDE_picnic_blankets_theorem_l402_40202

/-- The area of a blanket after a given number of folds -/
def folded_area (initial_area : ℕ) (num_folds : ℕ) : ℕ :=
  initial_area / 2^num_folds

/-- The total area of multiple blankets after folding -/
def total_folded_area (num_blankets : ℕ) (initial_area : ℕ) (num_folds : ℕ) : ℕ :=
  num_blankets * folded_area initial_area num_folds

theorem picnic_blankets_theorem :
  total_folded_area 3 64 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_picnic_blankets_theorem_l402_40202


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l402_40261

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/4, -3/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 3 = -9 * x

theorem intersection_point_is_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point := by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l402_40261


namespace NUMINAMATH_CALUDE_expression_simplification_l402_40251

theorem expression_simplification (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 + 6 = 45*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l402_40251


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_five_halves_l402_40278

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The length of the larger base -/
  a : ℕ
  /-- The length of the smaller base -/
  b : ℕ
  /-- The height of the trapezoid -/
  h : ℕ
  /-- The radius of the inscribed circle -/
  r : ℚ
  /-- The area of the upper part divided by the median -/
  upper_area : ℕ
  /-- The area of the lower part divided by the median -/
  lower_area : ℕ
  /-- Ensure the bases are different (it's a trapezoid) -/
  base_diff : a > b
  /-- The total area of the trapezoid -/
  total_area : (a + b) * h / 2 = upper_area + lower_area
  /-- The median divides the trapezoid into two parts -/
  median_division : upper_area = 15 ∧ lower_area = 30
  /-- The radius is half the height (property of inscribed circle in trapezoid) -/
  radius_height_relation : r = h / 2

/-- Theorem stating that the radius of the inscribed circle is 5/2 -/
theorem inscribed_circle_radius_is_five_halves (t : InscribedCircleTrapezoid) : t.r = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_is_five_halves_l402_40278


namespace NUMINAMATH_CALUDE_supplement_of_beta_l402_40238

def complementary_angles (α β : Real) : Prop := α + β = 90

theorem supplement_of_beta (α β : Real) 
  (h1 : complementary_angles α β) 
  (h2 : α = 30) : 
  180 - β = 120 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_beta_l402_40238


namespace NUMINAMATH_CALUDE_triangle_properties_l402_40225

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c ∧ Real.cos (t.A - t.C) = Real.cos t.B + 1/2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π/3 ∧ t.A = π/3 ∧
  ∀ (CD : ℝ), CD = 6 → 
    (∃ (max_perimeter : ℝ), max_perimeter = 4 * Real.sqrt 3 + 6 ∧
      ∀ (perimeter : ℝ), perimeter ≤ max_perimeter) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l402_40225


namespace NUMINAMATH_CALUDE_floor_y_length_l402_40205

/-- Represents a rectangular floor with length and width -/
structure RectangularFloor where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular floor -/
def area (floor : RectangularFloor) : ℝ :=
  floor.length * floor.width

theorem floor_y_length 
  (floor_x floor_y : RectangularFloor)
  (equal_area : area floor_x = area floor_y)
  (x_dimensions : floor_x.length = 10 ∧ floor_x.width = 18)
  (y_width : floor_y.width = 9) :
  floor_y.length = 20 := by
sorry

end NUMINAMATH_CALUDE_floor_y_length_l402_40205


namespace NUMINAMATH_CALUDE_h_is_smallest_l402_40239

/-- Definition of the partition property for h(n) -/
def has_partition_property (h n : ℕ) : Prop :=
  ∀ (A : Fin n → Set ℕ), 
    (∀ i j, i ≠ j → A i ∩ A j = ∅) → 
    (⋃ i, A i) = Finset.range h →
    ∃ (a x y : ℕ), 
      1 ≤ x ∧ x ≤ y ∧ y ≤ h ∧
      ∃ i, {a + x, a + y, a + x + y} ⊆ A i

/-- The function h(n) -/
def h (n : ℕ) : ℕ := Nat.choose n (n / 2)

/-- Main theorem: h(n) is the smallest positive integer satisfying the partition property -/
theorem h_is_smallest (n : ℕ) (hn : 0 < n) : 
  has_partition_property (h n) n ∧ 
  ∀ m, 0 < m ∧ m < h n → ¬has_partition_property m n :=
sorry

end NUMINAMATH_CALUDE_h_is_smallest_l402_40239


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l402_40229

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 25) 
  (h_a2 : a 2 = 3) :
  a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l402_40229


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l402_40265

/-- Given vectors in R², prove that if they satisfy certain conditions, then x = 1/2 -/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  let u : Fin 2 → ℝ := a + 2 • b
  let v : Fin 2 → ℝ := 2 • a - b
  (∃ (k : ℝ), k ≠ 0 ∧ u = k • v) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l402_40265


namespace NUMINAMATH_CALUDE_arrange_four_on_eight_l402_40288

/-- The number of ways to arrange n people on m chairs in a row,
    such that no two people sit next to each other -/
def arrangePeople (n m : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 120 ways to arrange 4 people on 8 chairs in a row,
    such that no two people sit next to each other -/
theorem arrange_four_on_eight :
  arrangePeople 4 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_four_on_eight_l402_40288


namespace NUMINAMATH_CALUDE_problem_1_l402_40220

theorem problem_1 : (1 : ℝ) * (1 - 2 * Real.sqrt 3) * (1 + 2 * Real.sqrt 3) - (1 + Real.sqrt 3)^2 = -15 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l402_40220


namespace NUMINAMATH_CALUDE_pencil_sorting_l402_40230

theorem pencil_sorting (box2 box3 box4 box5 : ℕ) : 
  box2 = 87 →
  box3 = box2 + 9 →
  box4 = box3 + 9 →
  box5 = box4 + 9 →
  box5 = 114 →
  box2 - 9 = 78 := by
sorry

end NUMINAMATH_CALUDE_pencil_sorting_l402_40230


namespace NUMINAMATH_CALUDE_simplify_expression_l402_40207

theorem simplify_expression (x : ℝ) : (2*x - 5)*(x + 6) - (x + 4)*(2*x - 1) = -26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l402_40207


namespace NUMINAMATH_CALUDE_equation_solution_l402_40294

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ * (x₁ + 2) = -3 * (x₁ + 2)) ∧ 
  (x₂ * (x₂ + 2) = -3 * (x₂ + 2)) ∧ 
  x₁ = -2 ∧ x₂ = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l402_40294


namespace NUMINAMATH_CALUDE_sin_B_value_l402_40240

-- Define a right triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2

-- Define the given triangle
def given_triangle : RightTriangle where
  a := 3
  b := 4
  c := 5
  right_angle := by norm_num

-- Theorem to prove
theorem sin_B_value (triangle : RightTriangle) (h1 : triangle.a = 3) (h2 : triangle.b = 4) :
  Real.sin (Real.arcsin (triangle.b / triangle.c)) = 4/5 := by
  sorry

#check sin_B_value given_triangle rfl rfl

end NUMINAMATH_CALUDE_sin_B_value_l402_40240


namespace NUMINAMATH_CALUDE_problem_1_l402_40268

theorem problem_1 : -3⁻¹ * Real.sqrt 27 + |1 - Real.sqrt 3| + (-1)^2023 = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l402_40268
