import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_log_function_l1050_105029

-- Define the logarithm function for any base a > 0 and a ≠ 1
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f(x) = log_a(x+2) + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 2) + 1

-- State the theorem
theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_log_function_l1050_105029


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l1050_105007

theorem tangent_sum_identity (α β γ : ℝ) (h : α + β + γ = Real.pi / 2) :
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l1050_105007


namespace NUMINAMATH_CALUDE_largest_band_size_l1050_105042

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- The total number of band members --/
def totalMembers (f : BandFormation) : ℕ := f.rows * f.membersPerRow

/-- Conditions for the band formations --/
def validFormations (original new : BandFormation) (total : ℕ) : Prop :=
  total < 100 ∧
  totalMembers original + 4 = total ∧
  totalMembers new = total ∧
  new.membersPerRow = original.membersPerRow + 2 ∧
  new.rows + 3 = original.rows

/-- The theorem stating that the largest possible number of band members is 88 --/
theorem largest_band_size :
  ∀ original new : BandFormation,
  ∀ total : ℕ,
  validFormations original new total →
  total ≤ 88 :=
sorry

end NUMINAMATH_CALUDE_largest_band_size_l1050_105042


namespace NUMINAMATH_CALUDE_jeans_pricing_l1050_105015

theorem jeans_pricing (cost : ℝ) (cost_positive : cost > 0) :
  let retailer_price := cost * (1 + 0.4)
  let customer_price := retailer_price * (1 + 0.15)
  (customer_price - cost) / cost = 0.61 := by
  sorry

end NUMINAMATH_CALUDE_jeans_pricing_l1050_105015


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l1050_105059

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l1050_105059


namespace NUMINAMATH_CALUDE_congcong_carbon_emissions_l1050_105039

/-- Carbon dioxide emissions calculation for household tap water -/
def carbon_emissions (water_usage : ℝ) : ℝ := water_usage * 0.91

/-- Congcong's water usage in a certain month (in tons) -/
def congcong_water_usage : ℝ := 6

/-- Theorem stating the carbon dioxide emissions from Congcong's tap water for a certain month -/
theorem congcong_carbon_emissions :
  carbon_emissions congcong_water_usage = 5.46 := by
  sorry

end NUMINAMATH_CALUDE_congcong_carbon_emissions_l1050_105039


namespace NUMINAMATH_CALUDE_simple_interest_growth_factor_l1050_105040

/-- The growth factor for simple interest -/
def growth_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

/-- Theorem: The growth factor for a 5% simple interest rate over 20 years is 2 -/
theorem simple_interest_growth_factor : 
  growth_factor (5 / 100) 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_growth_factor_l1050_105040


namespace NUMINAMATH_CALUDE_storage_unit_area_l1050_105087

theorem storage_unit_area :
  let total_units : ℕ := 42
  let total_area : ℕ := 5040
  let known_units : ℕ := 20
  let known_unit_length : ℕ := 8
  let known_unit_width : ℕ := 4
  let remaining_units := total_units - known_units
  let known_units_area := known_units * known_unit_length * known_unit_width
  let remaining_area := total_area - known_units_area
  remaining_area / remaining_units = 200 := by
sorry

end NUMINAMATH_CALUDE_storage_unit_area_l1050_105087


namespace NUMINAMATH_CALUDE_mess_expenditure_original_mess_expenditure_l1050_105024

/-- Calculates the original daily expenditure of a mess given initial conditions. -/
theorem mess_expenditure (initial_students : ℕ) (new_students : ℕ) (expense_increase : ℕ) (avg_decrease : ℕ) : ℕ :=
  let total_students : ℕ := initial_students + new_students
  let original_expenditure : ℕ := initial_students * (total_students * expense_increase) / (total_students * avg_decrease)
  original_expenditure

/-- Proves that the original daily expenditure of the mess was 420 given the specified conditions. -/
theorem original_mess_expenditure :
  mess_expenditure 35 7 42 1 = 420 := by
  sorry

end NUMINAMATH_CALUDE_mess_expenditure_original_mess_expenditure_l1050_105024


namespace NUMINAMATH_CALUDE_smallest_m_is_30_l1050_105080

def probability_condition (m : ℕ) : Prop :=
  (1 / 6) * ((m - 4) ^ 3 : ℚ) / (m ^ 3 : ℚ) > 3 / 5

theorem smallest_m_is_30 :
  ∀ k : ℕ, k > 0 → (probability_condition k → k ≥ 30) ∧
  probability_condition 30 := by sorry

end NUMINAMATH_CALUDE_smallest_m_is_30_l1050_105080


namespace NUMINAMATH_CALUDE_students_using_red_color_l1050_105031

theorem students_using_red_color 
  (total_students : ℕ) 
  (green_users : ℕ) 
  (both_colors : ℕ) 
  (h1 : total_students = 70) 
  (h2 : green_users = 52) 
  (h3 : both_colors = 38) : 
  total_students + both_colors - green_users = 56 := by
  sorry

end NUMINAMATH_CALUDE_students_using_red_color_l1050_105031


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_thirty_l1050_105094

theorem largest_multiple_of_seven_less_than_negative_thirty :
  ∃ (n : ℤ), n * 7 = -35 ∧ 
  n * 7 < -30 ∧ 
  ∀ (m : ℤ), m * 7 < -30 → m * 7 ≤ -35 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_thirty_l1050_105094


namespace NUMINAMATH_CALUDE_sheet_width_calculation_l1050_105057

theorem sheet_width_calculation (paper_length : Real) (margin : Real) (picture_area : Real) :
  paper_length = 10 ∧ margin = 1.5 ∧ picture_area = 38.5 →
  ∃ (paper_width : Real), 
    paper_width = 8.5 ∧
    (paper_width - 2 * margin) * (paper_length - 2 * margin) = picture_area :=
by sorry

end NUMINAMATH_CALUDE_sheet_width_calculation_l1050_105057


namespace NUMINAMATH_CALUDE_equation_real_root_implies_m_value_l1050_105023

theorem equation_real_root_implies_m_value (x m : ℝ) (i : ℂ) :
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) →
  m = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_equation_real_root_implies_m_value_l1050_105023


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1050_105002

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64 / 9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1050_105002


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_third_l1050_105001

open Real

theorem derivative_f_at_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = x + Real.sin x) :
  deriv f (π / 3) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_third_l1050_105001


namespace NUMINAMATH_CALUDE_bruce_lost_eggs_main_theorem_l1050_105022

/-- Proof that Bruce lost 70 eggs -/
theorem bruce_lost_eggs : ℕ → ℕ → ℕ → Prop :=
  fun initial_eggs remaining_eggs lost_eggs =>
    initial_eggs = 75 →
    remaining_eggs = 5 →
    lost_eggs = initial_eggs - remaining_eggs →
    lost_eggs = 70

/-- Main theorem statement -/
theorem main_theorem : ∃ lost_eggs : ℕ, bruce_lost_eggs 75 5 lost_eggs := by
  sorry

end NUMINAMATH_CALUDE_bruce_lost_eggs_main_theorem_l1050_105022


namespace NUMINAMATH_CALUDE_purely_imaginary_Z_implies_m_equals_two_l1050_105062

-- Define the complex number Z as a function of m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 2) (m^2 - 2*m - 3)

-- State the theorem
theorem purely_imaginary_Z_implies_m_equals_two :
  ∀ m : ℝ, (Z m).re = 0 ∧ (Z m).im ≠ 0 → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_Z_implies_m_equals_two_l1050_105062


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1050_105000

theorem arithmetic_equality : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1050_105000


namespace NUMINAMATH_CALUDE_complementary_angle_of_30_28_l1050_105033

/-- Represents an angle in degrees and minutes -/
structure DegreeMinute where
  degree : ℕ
  minute : ℕ

/-- Converts DegreeMinute to a rational number -/
def DegreeMinute.toRational (dm : DegreeMinute) : ℚ :=
  dm.degree + dm.minute / 60

/-- Theorem: The complementary angle of 30°28' is 59°32' -/
theorem complementary_angle_of_30_28 :
  let angle1 : DegreeMinute := ⟨30, 28⟩
  let complement : DegreeMinute := ⟨59, 32⟩
  DegreeMinute.toRational angle1 + DegreeMinute.toRational complement = 90 := by
  sorry


end NUMINAMATH_CALUDE_complementary_angle_of_30_28_l1050_105033


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l1050_105085

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l1050_105085


namespace NUMINAMATH_CALUDE_area_between_circles_l1050_105005

theorem area_between_circles (r : ℝ) (R : ℝ) : 
  r = 3 →                   -- radius of smaller circle
  R = 3 * r →               -- radius of larger circle is three times the smaller
  π * R^2 - π * r^2 = 72*π  -- area between circles is 72π
  := by sorry

end NUMINAMATH_CALUDE_area_between_circles_l1050_105005


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1050_105097

theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    (3 * p.1 - 2 * p.2 - 9 = 0) ∧
    (6 * p.1 + 4 * p.2 - 12 = 0) ∧
    (p.1 = 3) ∧
    (p.2 = -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1050_105097


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l1050_105038

-- Define the type for positions
inductive Position : Type
  | A | B | C | D | E | F

-- Define the function type for digit assignments
def DigitAssignment := Position → Fin 6

-- Define the condition that all digits are used exactly once
def allDigitsUsedOnce (assignment : DigitAssignment) : Prop :=
  ∀ d : Fin 6, ∃! p : Position, assignment p = d

-- Define the sum conditions
def sumConditions (assignment : DigitAssignment) : Prop :=
  (assignment Position.A).val + (assignment Position.D).val + (assignment Position.E).val = 15 ∧
  7 + (assignment Position.C).val + (assignment Position.E).val = 15 ∧
  9 + (assignment Position.C).val + (assignment Position.A).val = 15 ∧
  (assignment Position.A).val + 8 + (assignment Position.F).val = 15 ∧
  7 + (assignment Position.D).val + (assignment Position.F).val = 15 ∧
  9 + (assignment Position.D).val + (assignment Position.B).val = 15 ∧
  (assignment Position.B).val + (assignment Position.C).val + (assignment Position.F).val = 15

-- Define the correct assignment
def correctAssignment : DigitAssignment :=
  λ p => match p with
  | Position.A => 3  -- 4 - 1 (Fin 6 is 0-based)
  | Position.B => 0  -- 1 - 1
  | Position.C => 1  -- 2 - 1
  | Position.D => 4  -- 5 - 1
  | Position.E => 5  -- 6 - 1
  | Position.F => 2  -- 3 - 1

-- Theorem statement
theorem unique_digit_arrangement :
  ∀ assignment : DigitAssignment,
    allDigitsUsedOnce assignment ∧ sumConditions assignment →
    assignment = correctAssignment :=
sorry

end NUMINAMATH_CALUDE_unique_digit_arrangement_l1050_105038


namespace NUMINAMATH_CALUDE_coin_difference_l1050_105083

def coin_values : List ℕ := [5, 10, 25]

def total_amount : ℕ := 35

def min_coins (values : List ℕ) (amount : ℕ) : ℕ :=
  sorry

def max_coins (values : List ℕ) (amount : ℕ) : ℕ :=
  sorry

theorem coin_difference :
  max_coins coin_values total_amount - min_coins coin_values total_amount = 5 :=
sorry

end NUMINAMATH_CALUDE_coin_difference_l1050_105083


namespace NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1050_105088

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 34) 
  (product_condition : x * y = 240) : 
  |x - y| = 14 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1050_105088


namespace NUMINAMATH_CALUDE_unique_divisible_by_eight_l1050_105099

theorem unique_divisible_by_eight : ∃! n : ℕ, 70 < n ∧ n < 80 ∧ n % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_eight_l1050_105099


namespace NUMINAMATH_CALUDE_total_students_is_47_l1050_105096

/-- The number of students supposed to be in Miss Smith's English class -/
def total_students : ℕ :=
  let tables : ℕ := 6
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_multiplier : ℕ := 3
  let new_groups : ℕ := 2
  let students_per_new_group : ℕ := 4
  let german_students : ℕ := 3
  let french_students : ℕ := 3
  let norwegian_students : ℕ := 3

  let current_students := tables * students_per_table
  let missing_students := bathroom_students + (canteen_multiplier * bathroom_students)
  let new_group_students := new_groups * students_per_new_group
  let exchange_students := german_students + french_students + norwegian_students

  current_students + missing_students + new_group_students + exchange_students

theorem total_students_is_47 : total_students = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_47_l1050_105096


namespace NUMINAMATH_CALUDE_donalds_apples_l1050_105004

theorem donalds_apples (marin_apples total_apples : ℕ) 
  (h1 : marin_apples = 9)
  (h2 : total_apples = 11) :
  total_apples - marin_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_donalds_apples_l1050_105004


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1050_105081

/-- The smallest positive integer x such that 420x is a perfect square -/
def x : ℕ := 105

/-- The smallest positive integer y such that 420y is a perfect cube -/
def y : ℕ := 22050

/-- 420 * x is a perfect square -/
axiom x_square : ∃ n : ℕ, 420 * x = n * n

/-- 420 * y is a perfect cube -/
axiom y_cube : ∃ n : ℕ, 420 * y = n * n * n

/-- x is the smallest positive integer such that 420x is a perfect square -/
axiom x_smallest : ∀ z : ℕ, z > 0 → z < x → ¬∃ n : ℕ, 420 * z = n * n

/-- y is the smallest positive integer such that 420y is a perfect cube -/
axiom y_smallest : ∀ z : ℕ, z > 0 → z < y → ¬∃ n : ℕ, 420 * z = n * n * n

theorem sum_of_x_and_y : x + y = 22155 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1050_105081


namespace NUMINAMATH_CALUDE_regression_line_at_12_l1050_105034

def regression_line (x_mean y_mean slope : ℝ) (x : ℝ) : ℝ :=
  slope * (x - x_mean) + y_mean

theorem regression_line_at_12 
  (x_mean : ℝ) 
  (y_mean : ℝ) 
  (slope : ℝ) 
  (h1 : x_mean = 10) 
  (h2 : y_mean = 4) 
  (h3 : slope = 0.6) :
  regression_line x_mean y_mean slope 12 = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_at_12_l1050_105034


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l1050_105051

theorem min_shift_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)) →
  φ > 0 →
  (∀ x, f (x - φ) = f (π / 6 - x)) →
  φ ≥ 5 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l1050_105051


namespace NUMINAMATH_CALUDE_peters_age_l1050_105073

theorem peters_age (P Q : ℝ) 
  (h1 : Q - P = P / 2)
  (h2 : P + Q = 35) :
  Q = 21 := by sorry

end NUMINAMATH_CALUDE_peters_age_l1050_105073


namespace NUMINAMATH_CALUDE_rational_sum_squares_l1050_105028

theorem rational_sum_squares (a b c : ℚ) :
  1 / (b - c)^2 + 1 / (c - a)^2 + 1 / (a - b)^2 = (1 / (a - b) + 1 / (b - c) + 1 / (c - a))^2 :=
by sorry

end NUMINAMATH_CALUDE_rational_sum_squares_l1050_105028


namespace NUMINAMATH_CALUDE_exposed_sides_is_30_l1050_105014

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the arrangement of polygons -/
structure PolygonArrangement :=
  (triangle : RegularPolygon)
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (octagon : RegularPolygon)
  (nonagon : RegularPolygon)

/-- Calculates the number of exposed sides in the arrangement -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.triangle.sides +
  arrangement.square.sides +
  arrangement.pentagon.sides +
  arrangement.hexagon.sides +
  arrangement.heptagon.sides +
  arrangement.octagon.sides +
  arrangement.nonagon.sides -
  12 -- Subtracting the shared sides

/-- The specific arrangement described in the problem -/
def specificArrangement : PolygonArrangement :=
  { triangle := ⟨3⟩
  , square := ⟨4⟩
  , pentagon := ⟨5⟩
  , hexagon := ⟨6⟩
  , heptagon := ⟨7⟩
  , octagon := ⟨8⟩
  , nonagon := ⟨9⟩ }

/-- Theorem stating that the number of exposed sides in the specific arrangement is 30 -/
theorem exposed_sides_is_30 : exposedSides specificArrangement = 30 := by
  sorry

end NUMINAMATH_CALUDE_exposed_sides_is_30_l1050_105014


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l1050_105020

/-- The value of 'a' for a parabola y^2 = ax whose focus coincides with 
    the left focus of the ellipse x^2/6 + y^2/2 = 1 -/
theorem parabola_ellipse_focus_coincide : ∃ (a : ℝ), 
  (∀ (x y : ℝ), y^2 = a*x → x^2/6 + y^2/2 = 1 → 
    (x = -2 ∧ y = 0)) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l1050_105020


namespace NUMINAMATH_CALUDE_unique_equation_solution_l1050_105056

/-- A function that checks if a list of integers contains exactly the digits 1 to 9 --/
def isValidDigitList (lst : List Int) : Prop :=
  lst.length = 9 ∧ (∀ n, n ∈ lst → 1 ≤ n ∧ n ≤ 9) ∧ lst.toFinset.card = 9

/-- A function that converts a list of three integers to a three-digit number --/
def toThreeDigitNumber (a b c : Int) : Int :=
  100 * a + 10 * b + c

/-- A function that converts a list of two integers to a two-digit number --/
def toTwoDigitNumber (a b : Int) : Int :=
  10 * a + b

theorem unique_equation_solution :
  ∃! (digits : List Int),
    isValidDigitList digits ∧
    7 ∈ digits ∧
    let abc := toThreeDigitNumber (digits[0]!) (digits[1]!) (digits[2]!)
    let de := toTwoDigitNumber (digits[3]!) (digits[4]!)
    let f := digits[5]!
    let h := digits[8]!
    abc / de = f ∧ f = h - 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_equation_solution_l1050_105056


namespace NUMINAMATH_CALUDE_coffee_purchase_l1050_105084

theorem coffee_purchase (gift_card : ℝ) (coffee_price : ℝ) (remaining : ℝ) 
  (h1 : gift_card = 70)
  (h2 : coffee_price = 8.58)
  (h3 : remaining = 35.68) :
  (gift_card - remaining) / coffee_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_purchase_l1050_105084


namespace NUMINAMATH_CALUDE_geologists_probability_l1050_105070

/-- Represents a circular field with radial roads -/
structure CircularField where
  numRoads : ℕ
  radius : ℝ

/-- Represents a geologist's position -/
structure GeologistPosition where
  road : ℕ
  distance : ℝ

/-- Calculates the distance between two geologists -/
def distanceBetweenGeologists (field : CircularField) (pos1 pos2 : GeologistPosition) : ℝ :=
  sorry

/-- Calculates the probability of two geologists being at least a certain distance apart -/
def probabilityOfDistance (field : CircularField) (speed time minDistance : ℝ) : ℝ :=
  sorry

theorem geologists_probability (field : CircularField) :
  field.numRoads = 6 →
  probabilityOfDistance field 4 1 6 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_geologists_probability_l1050_105070


namespace NUMINAMATH_CALUDE_total_visitors_is_440_l1050_105036

/-- Represents the survey results from a modern art museum --/
structure SurveyResults where
  totalVisitors : ℕ
  notEnjoyedNotUnderstood : ℕ
  enjoyedAndUnderstood : ℕ
  visitorsBelowFortyRatio : ℚ
  visitorFortyAndAboveRatio : ℚ
  expertRatio : ℚ
  nonExpertRatio : ℚ
  enjoyedAndUnderstoodRatio : ℚ
  fortyAndAboveEnjoyedRatio : ℚ

/-- Theorem stating the total number of visitors based on survey conditions --/
theorem total_visitors_is_440 (survey : SurveyResults) :
  survey.totalVisitors = 440 ∧
  survey.notEnjoyedNotUnderstood = 110 ∧
  survey.enjoyedAndUnderstood = survey.totalVisitors - survey.notEnjoyedNotUnderstood ∧
  survey.visitorsBelowFortyRatio = 2 * survey.visitorFortyAndAboveRatio ∧
  survey.expertRatio = 3/5 ∧
  survey.nonExpertRatio = 2/5 ∧
  survey.enjoyedAndUnderstoodRatio = 3/4 ∧
  survey.fortyAndAboveEnjoyedRatio = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_total_visitors_is_440_l1050_105036


namespace NUMINAMATH_CALUDE_P_in_xoz_plane_l1050_105048

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Point P with coordinates (-2, 0, 3) -/
def P : Point3D :=
  ⟨-2, 0, 3⟩

theorem P_in_xoz_plane : P ∈ xoz_plane := by
  sorry


end NUMINAMATH_CALUDE_P_in_xoz_plane_l1050_105048


namespace NUMINAMATH_CALUDE_tan_225_degrees_l1050_105043

theorem tan_225_degrees : Real.tan (225 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_225_degrees_l1050_105043


namespace NUMINAMATH_CALUDE_probability_select_one_coastal_l1050_105075

/-- Represents a city that can be either coastal or inland -/
inductive City
| coastal : City
| inland : City

/-- The set of all cities -/
def allCities : Finset City := sorry

/-- The set of coastal cities -/
def coastalCities : Finset City := sorry

theorem probability_select_one_coastal :
  (2 : ℕ) = Finset.card coastalCities →
  (4 : ℕ) = Finset.card allCities →
  (1 : ℚ) / 2 = Finset.card coastalCities / Finset.card allCities := by
  sorry

end NUMINAMATH_CALUDE_probability_select_one_coastal_l1050_105075


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1050_105050

theorem quadratic_roots_relation (b c p q r s : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 + p*x + q = 0 ↔ x = r^2 ∨ x = s^2) →
  p = 2*c - b^2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1050_105050


namespace NUMINAMATH_CALUDE_total_spots_l1050_105098

def cow_spots (left_spots : ℕ) (right_spots : ℕ) : Prop :=
  (left_spots = 16) ∧ 
  (right_spots = 3 * left_spots + 7)

theorem total_spots : ∀ left_spots right_spots : ℕ, 
  cow_spots left_spots right_spots → left_spots + right_spots = 71 := by
sorry

end NUMINAMATH_CALUDE_total_spots_l1050_105098


namespace NUMINAMATH_CALUDE_algebraic_expression_transformation_l1050_105016

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x, x^2 - 6*x + b = (x - a)^2 - 1) → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_transformation_l1050_105016


namespace NUMINAMATH_CALUDE_frac_one_over_x_is_fraction_l1050_105066

-- Define what a fraction is
def is_fraction (expr : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, expr x = (n x) / (d x)

-- State the theorem
theorem frac_one_over_x_is_fraction :
  is_fraction (λ x => 1 / x) :=
sorry

end NUMINAMATH_CALUDE_frac_one_over_x_is_fraction_l1050_105066


namespace NUMINAMATH_CALUDE_square_IJKL_side_length_l1050_105027

-- Define the side lengths of squares ABCD and EFGH
def side_ABCD : ℝ := 3
def side_EFGH : ℝ := 9

-- Define the right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the arrangement of triangles
def triangle_arrangement (t : RightTriangle) : Prop :=
  t.leg1 - t.leg2 = side_ABCD ∧ t.leg1 + t.leg2 = side_EFGH

-- Theorem statement
theorem square_IJKL_side_length 
  (t : RightTriangle) 
  (h : triangle_arrangement t) : 
  t.hypotenuse = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_IJKL_side_length_l1050_105027


namespace NUMINAMATH_CALUDE_twentieth_term_is_400_l1050_105090

/-- A second-order arithmetic sequence -/
def SecondOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) - (a (n-1) - a (n-2)) = 2

/-- The sequence starts with 1, 4, 9, 16 -/
def SequenceStart (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 4 ∧ a 3 = 9 ∧ a 4 = 16

theorem twentieth_term_is_400 (a : ℕ → ℕ) 
  (h1 : SecondOrderArithmeticSequence a) 
  (h2 : SequenceStart a) : 
  a 20 = 400 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_400_l1050_105090


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l1050_105019

theorem min_value_of_function (x : ℝ) (h : x > 0) : x^2 + 2/x ≥ 3 := by sorry

theorem min_value_achievable : ∃ x > 0, x^2 + 2/x = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l1050_105019


namespace NUMINAMATH_CALUDE_school_trip_speed_l1050_105065

/-- Proves that the speed for the first half of a round trip is 6 km/hr,
    given the specified conditions. -/
theorem school_trip_speed
  (total_distance : ℝ)
  (return_speed : ℝ)
  (total_time : ℝ)
  (h1 : total_distance = 48)
  (h2 : return_speed = 4)
  (h3 : total_time = 10)
  : ∃ (going_speed : ℝ),
    going_speed = 6 ∧
    total_time = (total_distance / 2) / going_speed + (total_distance / 2) / return_speed :=
sorry

end NUMINAMATH_CALUDE_school_trip_speed_l1050_105065


namespace NUMINAMATH_CALUDE_product_of_numbers_l1050_105093

theorem product_of_numbers (x y : ℝ) : 
  x + y = 22 → x^2 + y^2 = 460 → x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1050_105093


namespace NUMINAMATH_CALUDE_triple_transmission_better_for_zero_l1050_105046

-- Define the channel parameters
variable (α β : ℝ)

-- Define the conditions
variable (h1 : 0 < α)
variable (h2 : α < 0.5)
variable (h3 : 0 < β)
variable (h4 : β < 1)

-- Define the probabilities for single and triple transmission
def P_single_0 : ℝ := 1 - α
def P_triple_0 : ℝ := 3 * α * (1 - α)^2 + (1 - α)^3

-- State the theorem
theorem triple_transmission_better_for_zero :
  P_triple_0 α > P_single_0 α := by sorry

end NUMINAMATH_CALUDE_triple_transmission_better_for_zero_l1050_105046


namespace NUMINAMATH_CALUDE_min_even_integers_l1050_105079

theorem min_even_integers (a b c d e f g h : ℤ) : 
  a + b + c = 36 →
  a + b + c + d + e + f = 60 →
  a + b + c + d + e + f + g + h = 76 →
  g * h = 48 →
  ∃ (count : ℕ), count ≥ 1 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) + 
            (if Even g then 1 else 0) + 
            (if Even h then 1 else 0) :=
by
  sorry

end NUMINAMATH_CALUDE_min_even_integers_l1050_105079


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_l1050_105018

/-- The expression (x^2-4x+4)/(9-x^3) is nonnegative if and only if x ≤ 3 -/
theorem expression_nonnegative_iff (x : ℝ) : (x^2 - 4*x + 4) / (9 - x^3) ≥ 0 ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_l1050_105018


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1050_105052

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 2}

theorem intersection_of_M_and_N : M ∩ N = {(2, 0)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1050_105052


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l1050_105009

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number --/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l1050_105009


namespace NUMINAMATH_CALUDE_non_defective_engines_count_l1050_105008

def total_engines (num_batches : ℕ) (engines_per_batch : ℕ) : ℕ :=
  num_batches * engines_per_batch

def non_defective_fraction : ℚ := 3/4

theorem non_defective_engines_count 
  (num_batches : ℕ) 
  (engines_per_batch : ℕ) 
  (h1 : num_batches = 5) 
  (h2 : engines_per_batch = 80) :
  ↑(total_engines num_batches engines_per_batch) * non_defective_fraction = 300 := by
  sorry

end NUMINAMATH_CALUDE_non_defective_engines_count_l1050_105008


namespace NUMINAMATH_CALUDE_right_triangle_with_special_sides_l1050_105013

theorem right_triangle_with_special_sides : ∃ (a b c : ℕ), 
  (a * a + b * b = c * c) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  ((a % 4 = 0 ∨ b % 4 = 0) ∧ 
   (a % 3 = 0 ∨ b % 3 = 0) ∧ 
   (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_special_sides_l1050_105013


namespace NUMINAMATH_CALUDE_perimeter_is_twenty_l1050_105032

/-- The perimeter of a six-sided figure with specified side lengths -/
def perimeter_of_figure (h1 h2 v1 v2 v3 v4 : ℕ) : ℕ :=
  h1 + h2 + v1 + v2 + v3 + v4

/-- Theorem: The perimeter of the given figure is 20 units -/
theorem perimeter_is_twenty :
  ∃ (h1 h2 v1 v2 v3 v4 : ℕ),
    h1 + h2 = 5 ∧
    v1 = 2 ∧ v2 = 3 ∧ v3 = 3 ∧ v4 = 2 ∧
    perimeter_of_figure h1 h2 v1 v2 v3 v4 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_perimeter_is_twenty_l1050_105032


namespace NUMINAMATH_CALUDE_simplify_expression_l1050_105053

theorem simplify_expression : (9 * 10^10) / (3 * 10^3 - 2 * 10^3) = 9 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1050_105053


namespace NUMINAMATH_CALUDE_permutations_count_l1050_105074

def original_number : List Nat := [1, 1, 2, 3, 4, 5, 6, 7]
def target_number : List Nat := [4, 6, 7, 5, 3, 2, 1, 1]

def count_permutations_less_than_or_equal (original : List Nat) (target : List Nat) : Nat :=
  sorry

theorem permutations_count :
  count_permutations_less_than_or_equal original_number target_number = 12240 :=
sorry

end NUMINAMATH_CALUDE_permutations_count_l1050_105074


namespace NUMINAMATH_CALUDE_tan_half_product_l1050_105095

theorem tan_half_product (a b : Real) :
  3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = -4 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l1050_105095


namespace NUMINAMATH_CALUDE_square_divisibility_l1050_105077

theorem square_divisibility (n : ℕ) (h1 : n > 0) (h2 : ∀ d : ℕ, d ∣ n → d ≤ 6) :
  36 ∣ n^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l1050_105077


namespace NUMINAMATH_CALUDE_linear_system_determinant_l1050_105071

/-- 
Given integers a, b, c, d such that the system of equations
  ax + by = m
  cx + dy = n
has integer solutions for all integer values of m and n,
prove that |ad - bc| = 1
-/
theorem linear_system_determinant (a b c d : ℤ) 
  (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
  |a * d - b * c| = 1 :=
sorry

end NUMINAMATH_CALUDE_linear_system_determinant_l1050_105071


namespace NUMINAMATH_CALUDE_jug_pouring_l1050_105017

/-- Represents the state of two jugs after pouring from two equal full jugs -/
structure JugState where
  x_capacity : ℚ
  y_capacity : ℚ
  x_filled : ℚ
  y_filled : ℚ
  h_x_filled : x_filled = 1/4 * x_capacity
  h_y_filled : y_filled = 2/3 * y_capacity
  h_equal_initial : x_filled + y_filled = x_capacity

/-- The fraction of jug X that contains water after filling jug Y -/
def final_x_fraction (state : JugState) : ℚ :=
  1/8

theorem jug_pouring (state : JugState) :
  final_x_fraction state = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_jug_pouring_l1050_105017


namespace NUMINAMATH_CALUDE_train_speed_problem_l1050_105091

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  crossing_time = 4 →
  ∃ (speed : ℝ),
    speed * crossing_time = 2 * train_length ∧
    speed * 3.6 = 108 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1050_105091


namespace NUMINAMATH_CALUDE_common_tangents_l1050_105092

/-- The first curve: 9x^2 + 16y^2 = 144 -/
def curve1 (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

/-- The second curve: 7x^2 - 32y^2 = 224 -/
def curve2 (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

/-- A common tangent line: ax + by + c = 0 -/
def is_tangent (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, (curve1 x y ∨ curve2 x y) → (a * x + b * y + c = 0 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      ((x' - x)^2 + (y' - y)^2 < δ^2) → 
      ((curve1 x' y' ∨ curve2 x' y') → (a * x' + b * y' + c ≠ 0)))

/-- The theorem stating that the given equations are common tangents -/
theorem common_tangents : 
  (is_tangent 1 1 5 ∧ is_tangent 1 1 (-5) ∧ is_tangent 1 (-1) 5 ∧ is_tangent 1 (-1) (-5)) :=
sorry

end NUMINAMATH_CALUDE_common_tangents_l1050_105092


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1050_105047

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h : a > b)
  (h' : b > 0)

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- A point on the ellipse -/
def Point_on_ellipse (e : Ellipse a b) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The right vertex of the ellipse -/
def right_vertex (e : Ellipse a b) : ℝ × ℝ := (a, 0)

/-- The left focus of the ellipse -/
def left_focus (e : Ellipse a b) : ℝ × ℝ := sorry

/-- Predicate to check if a vector is twice another vector -/
def is_twice (v w : ℝ × ℝ) : Prop := v = 2 • w

/-- Theorem stating the eccentricity of the ellipse under given conditions -/
theorem ellipse_eccentricity (a b : ℝ) (e : Ellipse a b) 
  (B : ℝ × ℝ) (P : ℝ × ℝ) 
  (h_B : Point_on_ellipse e B.1 B.2)
  (h_BF_perp : (B.1 - (left_focus e).1) = 0)
  (h_P_on_y : P.1 = 0)
  (h_AP_PB : is_twice (right_vertex e - P) (P - B)) :
  eccentricity e = 1/2 := sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1050_105047


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1050_105082

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ ^ 2 - Real.sin θ ^ 2) = 0

/-- The Cartesian equation of two intersecting straight lines -/
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 = y^2

/-- Theorem stating that the polar equation represents two intersecting straight lines -/
theorem polar_to_cartesian :
  ∀ x y : ℝ, ∃ ρ θ : ℝ, polar_equation ρ θ ↔ cartesian_equation x y :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1050_105082


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l1050_105058

/-- Given a cubic function f(x) with an extreme value at x = 1, prove that a + b = -7 -/
theorem extreme_value_cubic (a b : ℝ) : 
  let f := fun x : ℝ ↦ x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧ 
  (∃ (ε : ℝ), ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  f 1 = 10 →
  a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l1050_105058


namespace NUMINAMATH_CALUDE_triangle_area_l1050_105037

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (∀ x y z, (x = a ∧ y = b ∧ z = c) → 
    (x = 2 * (y * Real.sin C) * (z * Real.sin B) / (Real.sin A)) ∧
    (y^2 + z^2 - x^2 = 8)) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1050_105037


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1050_105035

theorem complex_fraction_evaluation :
  2 - (1 / (2 + (1 / (2 - (1 / 3))))) = 21 / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1050_105035


namespace NUMINAMATH_CALUDE_star_identity_l1050_105086

/-- The binary operation * on pairs of real numbers -/
def star (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * q.1, p.1 * q.2 + p.2 * q.1)

/-- The identity element for the star operation -/
def identity_element : ℝ × ℝ := (1, 0)

/-- Theorem stating that (1, 0) is the unique identity element for the star operation -/
theorem star_identity :
  ∀ p : ℝ × ℝ, star p identity_element = p ∧
  (∀ q : ℝ × ℝ, (∀ p : ℝ × ℝ, star p q = p) → q = identity_element) := by
  sorry

end NUMINAMATH_CALUDE_star_identity_l1050_105086


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l1050_105069

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1010_is_10 :
  binary_to_decimal [true, false, true, false] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l1050_105069


namespace NUMINAMATH_CALUDE_water_level_accurate_l1050_105045

/-- Represents the water level function for a reservoir -/
def water_level (x : ℝ) : ℝ := 6 + 0.3 * x

/-- Theorem stating that the water level function accurately describes the reservoir's water level -/
theorem water_level_accurate (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  water_level x = 6 + 0.3 * x ∧ water_level x ≥ 6 ∧ water_level x ≤ 7.5 := by
  sorry

end NUMINAMATH_CALUDE_water_level_accurate_l1050_105045


namespace NUMINAMATH_CALUDE_exterior_bisector_theorem_l1050_105060

/-- Represents a triangle with angles given in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- The triangle formed by the exterior angle bisectors -/
def exterior_bisector_triangle : Triangle :=
  { angle1 := 52
    angle2 := 61
    angle3 := 67
    sum_180 := by norm_num }

/-- The original triangle whose exterior angle bisectors form the given triangle -/
def original_triangle : Triangle :=
  { angle1 := 76
    angle2 := 58
    angle3 := 46
    sum_180 := by norm_num }

theorem exterior_bisector_theorem (t : Triangle) :
  t = exterior_bisector_triangle →
  ∃ (orig : Triangle), orig = original_triangle ∧
    (90 - orig.angle2 / 2) + (90 - orig.angle3 / 2) = t.angle1 ∧
    (90 - orig.angle1 / 2) + (90 - orig.angle3 / 2) = t.angle2 ∧
    (90 - orig.angle1 / 2) + (90 - orig.angle2 / 2) = t.angle3 :=
by
  sorry

end NUMINAMATH_CALUDE_exterior_bisector_theorem_l1050_105060


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1050_105044

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_sq_eq : a^2 + b^2 + c^2 = 6)
  (sum_cube_eq : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1050_105044


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1050_105078

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1050_105078


namespace NUMINAMATH_CALUDE_jordan_fourth_period_shots_l1050_105010

/-- Given Jordan's shot-blocking performance in a hockey game, prove the number of shots blocked in the fourth period. -/
theorem jordan_fourth_period_shots 
  (first_period : ℕ) 
  (second_period : ℕ)
  (third_period : ℕ)
  (total_shots : ℕ)
  (h1 : first_period = 4)
  (h2 : second_period = 2 * first_period)
  (h3 : third_period = second_period - 3)
  (h4 : total_shots = 21)
  : total_shots - (first_period + second_period + third_period) = 4 := by
  sorry

end NUMINAMATH_CALUDE_jordan_fourth_period_shots_l1050_105010


namespace NUMINAMATH_CALUDE_pizza_pasta_price_difference_l1050_105067

/-- The price difference between a Pizza and a Pasta -/
def price_difference (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  pizza_price - pasta_price

/-- The total cost of the Smith family's purchase -/
def smith_purchase (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  2 * pizza_price + 3 * chilli_price + 4 * pasta_price

/-- The total cost of the Patel family's purchase -/
def patel_purchase (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  5 * pizza_price + 6 * chilli_price + 7 * pasta_price

theorem pizza_pasta_price_difference 
  (pizza_price chilli_price pasta_price : ℚ) 
  (h1 : smith_purchase pizza_price chilli_price pasta_price = 53)
  (h2 : patel_purchase pizza_price chilli_price pasta_price = 107) :
  price_difference pizza_price chilli_price pasta_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pasta_price_difference_l1050_105067


namespace NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l1050_105026

theorem tan_plus_four_sin_twenty_degrees :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l1050_105026


namespace NUMINAMATH_CALUDE_magazine_page_height_l1050_105025

/-- Given advertising costs and dimensions, calculate the height of a magazine page -/
theorem magazine_page_height 
  (cost_per_sq_inch : ℝ) 
  (ad_fraction : ℝ) 
  (page_width : ℝ) 
  (total_cost : ℝ) 
  (h : cost_per_sq_inch = 8)
  (h₁ : ad_fraction = 1/2)
  (h₂ : page_width = 12)
  (h₃ : total_cost = 432) :
  ∃ (page_height : ℝ), 
    page_height = 9 ∧ 
    ad_fraction * page_height * page_width * cost_per_sq_inch = total_cost := by
  sorry

end NUMINAMATH_CALUDE_magazine_page_height_l1050_105025


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1050_105006

theorem min_value_of_expression (x y : ℝ) : (x*y - 1)^3 + (x + y)^3 ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1050_105006


namespace NUMINAMATH_CALUDE_battery_current_at_12_ohms_l1050_105063

/-- Given a battery with voltage 48V and a relationship between current I and resistance R,
    prove that when R = 12Ω, I = 4A. -/
theorem battery_current_at_12_ohms :
  let voltage : ℝ := 48
  let R : ℝ := 12
  let I : ℝ := voltage / R
  I = 4 := by sorry

end NUMINAMATH_CALUDE_battery_current_at_12_ohms_l1050_105063


namespace NUMINAMATH_CALUDE_count_triangles_l1050_105030

/-- A point in the plane with coordinates that are multiples of 3 -/
structure Point :=
  (x : ℤ)
  (y : ℤ)
  (x_multiple : 3 ∣ x)
  (y_multiple : 3 ∣ y)

/-- The equation 47x + y = 2353 -/
def satisfies_equation (p : Point) : Prop :=
  47 * p.x + p.y = 2353

/-- The area of triangle OPQ where O is the origin -/
def triangle_area (p q : Point) : ℚ :=
  (p.x * q.y - q.x * p.y : ℚ) / 2

/-- The main theorem -/
theorem count_triangles :
  ∃ (triangle_set : Finset (Point × Point)),
    (∀ (p q : Point), (p, q) ∈ triangle_set →
      p ≠ q ∧
      satisfies_equation p ∧
      satisfies_equation q ∧
      (triangle_area p q).num ≠ 0 ∧
      (triangle_area p q).den = 1) ∧
    triangle_set.card = 64 ∧
    ∀ (p q : Point),
      p ≠ q →
      satisfies_equation p →
      satisfies_equation q →
      (triangle_area p q).num ≠ 0 →
      (triangle_area p q).den = 1 →
      (p, q) ∈ triangle_set :=
sorry

end NUMINAMATH_CALUDE_count_triangles_l1050_105030


namespace NUMINAMATH_CALUDE_troll_ratio_l1050_105054

/-- Given the number of trolls in different locations, prove the ratio of trolls in the plains to trolls under the bridge -/
theorem troll_ratio (path bridge plains : ℕ) : 
  path = 6 ∧ 
  bridge = 4 * path - 6 ∧ 
  path + bridge + plains = 33 →
  plains * 2 = bridge := by
sorry

end NUMINAMATH_CALUDE_troll_ratio_l1050_105054


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1050_105061

def A : Set Nat := {3, 5, 6, 8}
def B : Set Nat := {4, 5, 7, 8}

theorem intersection_of_A_and_B : A ∩ B = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1050_105061


namespace NUMINAMATH_CALUDE_probability_jack_queen_king_hearts_l1050_105012

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (face_cards_per_suit : Nat)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { cards := 52,
    suits := 4,
    face_cards_per_suit := 3 }

/-- The probability of drawing a specific set of cards from a deck -/
def probability (d : Deck) (favorable_outcomes : Nat) : ℚ :=
  favorable_outcomes / d.cards

theorem probability_jack_queen_king_hearts (d : Deck := standard_deck) :
  probability d d.face_cards_per_suit = 3 / 52 := by
  sorry

#eval probability standard_deck standard_deck.face_cards_per_suit

end NUMINAMATH_CALUDE_probability_jack_queen_king_hearts_l1050_105012


namespace NUMINAMATH_CALUDE_no_prime_solution_l1050_105068

theorem no_prime_solution : 
  ¬∃ (p : ℕ), Nat.Prime p ∧ 
  (p ^ 3 + 7) + (3 * p ^ 2 + 6) + (p ^ 2 + p + 3) + (p ^ 2 + 2 * p + 5) + 6 = 
  (p ^ 2 + 4 * p + 2) + (2 * p ^ 2 + 7 * p + 1) + (3 * p ^ 2 + 6 * p) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1050_105068


namespace NUMINAMATH_CALUDE_student_count_l1050_105076

theorem student_count (x : ℕ) (h : x > 0) :
  (Nat.choose x 4 : ℚ) / (x * (x - 1)) = 13 / 2 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1050_105076


namespace NUMINAMATH_CALUDE_vicente_shopping_cost_l1050_105055

/-- Calculates the total amount spent in US dollars given the following conditions:
  - 5 kg of rice at €2 per kg with a 10% discount
  - 3 pounds of meat at £5 per pound with a 5% sales tax
  - Exchange rates: €1 = $1.20 and £1 = $1.35
-/
theorem vicente_shopping_cost :
  let rice_kg : ℝ := 5
  let rice_price_euro : ℝ := 2
  let meat_lb : ℝ := 3
  let meat_price_pound : ℝ := 5
  let rice_discount : ℝ := 0.1
  let meat_tax : ℝ := 0.05
  let euro_to_usd : ℝ := 1.20
  let pound_to_usd : ℝ := 1.35
  
  let rice_cost : ℝ := rice_kg * rice_price_euro * (1 - rice_discount) * euro_to_usd
  let meat_cost : ℝ := meat_lb * meat_price_pound * (1 + meat_tax) * pound_to_usd
  let total_cost : ℝ := rice_cost + meat_cost

  total_cost = 32.06 := by sorry

end NUMINAMATH_CALUDE_vicente_shopping_cost_l1050_105055


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1050_105089

/-- A quadratic equation in one variable x is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 3x^2 + 2x + 4 = 0 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 4

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1050_105089


namespace NUMINAMATH_CALUDE_total_ways_is_2531_l1050_105072

/-- The number of different types of cookies -/
def num_cookie_types : ℕ := 6

/-- The number of different types of milk -/
def num_milk_types : ℕ := 4

/-- The total number of product types -/
def total_product_types : ℕ := num_cookie_types + num_milk_types

/-- The number of products they purchase collectively -/
def total_purchases : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of ways Charlie and Delta can leave the store with 4 products collectively -/
def total_ways : ℕ :=
  -- Charlie 4, Delta 0
  choose total_product_types 4 +
  -- Charlie 3, Delta 1
  choose total_product_types 3 * num_cookie_types +
  -- Charlie 2, Delta 2
  choose total_product_types 2 * (choose num_cookie_types 2 + num_cookie_types) +
  -- Charlie 1, Delta 3
  total_product_types * (choose num_cookie_types 3 + num_cookie_types * (num_cookie_types - 1) + num_cookie_types) +
  -- Charlie 0, Delta 4
  (choose num_cookie_types 4 + num_cookie_types * (num_cookie_types - 1) + choose num_cookie_types 2 * 3 + num_cookie_types)

theorem total_ways_is_2531 : total_ways = 2531 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_2531_l1050_105072


namespace NUMINAMATH_CALUDE_min_colors_needed_l1050_105064

/-- Represents a coloring of a 2023 x 2023 grid --/
def Coloring := Fin 2023 → Fin 2023 → ℕ

/-- Checks if a coloring satisfies the given condition --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ color row i,
    (∀ j ∈ Finset.range 6, c row (i + j) = color) →
    (∀ k < i, ∀ l > i + 5, ∀ m, c m k ≠ color ∧ c m l ≠ color)

/-- The main theorem stating the smallest number of colors needed --/
theorem min_colors_needed :
  (∃ (c : Coloring) (n : ℕ), n = 338 ∧ (∀ i j, c i j < n) ∧ valid_coloring c) ∧
  (∀ (c : Coloring) (n : ℕ), (∀ i j, c i j < n) ∧ valid_coloring c → n ≥ 338) :=
sorry

end NUMINAMATH_CALUDE_min_colors_needed_l1050_105064


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l1050_105049

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.0010101 * (10 : ℝ) ^ m ≤ 100)) ∧ 
  (0.0010101 * (10 : ℝ) ^ k > 100) → 
  k = 6 := by sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l1050_105049


namespace NUMINAMATH_CALUDE_puppy_feeding_schedule_l1050_105021

-- Define the feeding schedule and amounts
def total_days : ℕ := 28 -- 4 weeks
def today_food : ℚ := 1/2
def last_two_weeks_daily : ℚ := 1
def total_food : ℚ := 25

-- Define the unknown amount for the first two weeks
def first_two_weeks_per_meal : ℚ := 1/4

-- Theorem statement
theorem puppy_feeding_schedule :
  let first_two_weeks_total := 14 * 3 * first_two_weeks_per_meal
  let last_two_weeks_total := 14 * last_two_weeks_daily
  today_food + first_two_weeks_total + last_two_weeks_total = total_food :=
by sorry

end NUMINAMATH_CALUDE_puppy_feeding_schedule_l1050_105021


namespace NUMINAMATH_CALUDE_distinct_values_count_l1050_105011

-- Define a type for expressions
inductive Expr
  | Num : ℕ → Expr
  | Pow : Expr → Expr → Expr

-- Define a function to evaluate expressions
def eval : Expr → ℕ
  | Expr.Num n => n
  | Expr.Pow a b => (eval a) ^ (eval b)

-- Define the base expression
def baseExpr : Expr :=
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3)))

-- Define all possible parenthesizations
def parenthesizations : List Expr := [
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3))),
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Num 3)),
  Expr.Pow (Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Num 3)) (Expr.Num 3),
  Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3))) (Expr.Num 3),
  Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Pow (Expr.Num 3) (Expr.Num 3))
]

-- Theorem: The number of distinct values is 3
theorem distinct_values_count :
  (parenthesizations.map eval).toFinset.card = 3 := by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l1050_105011


namespace NUMINAMATH_CALUDE_max_draws_at_23_l1050_105003

/-- Represents a lottery draw as a list of distinct integers -/
def LotteryDraw := List Nat

/-- The number of numbers drawn in each lottery draw -/
def drawSize : Nat := 5

/-- The maximum number that can be drawn -/
def maxNumber : Nat := 90

/-- Function to calculate the number of possible draws for a given second smallest number -/
def countDraws (secondSmallest : Nat) : Nat :=
  (secondSmallest - 1) * (maxNumber - secondSmallest) * (maxNumber - secondSmallest - 1) * (maxNumber - secondSmallest - 2)

theorem max_draws_at_23 :
  ∀ m, m ≠ 23 → countDraws 23 ≥ countDraws m :=
sorry

end NUMINAMATH_CALUDE_max_draws_at_23_l1050_105003


namespace NUMINAMATH_CALUDE_equation_solution_l1050_105041

theorem equation_solution : ∃ x : ℝ, 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1050_105041
