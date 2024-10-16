import Mathlib

namespace NUMINAMATH_CALUDE_floor_ceil_inequality_l3274_327419

theorem floor_ceil_inequality (a b c : ℝ) 
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_inequality_l3274_327419


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l3274_327436

theorem min_value_of_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094 :=
sorry

theorem equality_achieved : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l3274_327436


namespace NUMINAMATH_CALUDE_gerald_remaining_pfennigs_l3274_327491

/-- Represents the number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

/-- Represents the number of farthings Gerald has -/
def geralds_farthings : ℕ := 54

/-- Represents the cost of a meat pie in pfennigs -/
def meat_pie_cost : ℕ := 2

/-- Calculates the number of pfennigs Gerald will have left after buying the pie -/
def remaining_pfennigs : ℕ :=
  geralds_farthings / farthings_per_pfennig - meat_pie_cost

/-- Theorem stating that Gerald will have 7 pfennigs left after buying the pie -/
theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 7 := by sorry

end NUMINAMATH_CALUDE_gerald_remaining_pfennigs_l3274_327491


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l3274_327443

/-- Calculates the corrected mean of a set of observations after fixing recording errors -/
theorem corrected_mean_calculation (n : ℕ) (original_mean : ℝ) 
  (error1_recorded error1_actual : ℝ)
  (error2_recorded error2_actual : ℝ)
  (error3_recorded error3_actual : ℝ)
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : error1_recorded = 23 ∧ error1_actual = 48)
  (h4 : error2_recorded = 42 ∧ error2_actual = 36)
  (h5 : error3_recorded = 28 ∧ error3_actual = 55) :
  let corrected_sum := n * original_mean + 
    (error1_actual - error1_recorded) + 
    (error2_actual - error2_recorded) + 
    (error3_actual - error3_recorded)
  (corrected_sum / n) = 41.92 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l3274_327443


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3274_327412

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  c_eq : c = 7/2
  area_eq : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2
  tan_eq : Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1)

/-- Theorem about the properties of the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.C = π/3 ∧ t.a + t.b = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l3274_327412


namespace NUMINAMATH_CALUDE_work_hours_per_day_l3274_327401

theorem work_hours_per_day 
  (total_hours : ℝ) 
  (total_days : ℝ) 
  (h1 : total_hours = 8.0) 
  (h2 : total_days = 4.0) 
  (h3 : total_days > 0) : 
  total_hours / total_days = 2.0 := by
sorry

end NUMINAMATH_CALUDE_work_hours_per_day_l3274_327401


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l3274_327438

theorem angle_sum_around_point (y : ℝ) : 
  y > 0 ∧ 150 + y + y = 360 → y = 105 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l3274_327438


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3274_327493

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- State the theorem
theorem set_intersection_equality : M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3274_327493


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l3274_327404

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 4

/-- Theorem: The number of intersection points between the parabola y = -x^2 + 4x - 4 
    and the coordinate axes is equal to 2 -/
theorem parabola_intersection_points : 
  (∃! x : ℝ, f x = 0) ∧ (∃! y : ℝ, f 0 = y) ∧ 
  (∀ x y : ℝ, (x = 0 ∨ y = 0) → (y = f x) → (x = 0 ∧ y = f 0) ∨ (y = 0 ∧ f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l3274_327404


namespace NUMINAMATH_CALUDE_perimeter_of_square_region_l3274_327472

/-- The perimeter of a region formed by 8 congruent squares arranged in a 2x4 rectangle,
    given that the total area of the region is 512 square centimeters. -/
theorem perimeter_of_square_region (total_area : ℝ) (num_squares : ℕ) (rows cols : ℕ) :
  total_area = 512 →
  num_squares = 8 →
  rows = 2 →
  cols = 4 →
  let square_side := Real.sqrt (total_area / num_squares)
  let perimeter := 2 * square_side * (rows + cols)
  perimeter = 128 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_region_l3274_327472


namespace NUMINAMATH_CALUDE_A_union_B_eq_B_l3274_327437

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

theorem A_union_B_eq_B : A ∪ B = B := by
  sorry

end NUMINAMATH_CALUDE_A_union_B_eq_B_l3274_327437


namespace NUMINAMATH_CALUDE_last_student_theorem_l3274_327482

/-- The number of students initially in the line -/
def initial_students : ℕ := 196

/-- The process of removing students at odd positions and renumbering -/
def remove_odd_positions (n : ℕ) : ℕ :=
  if n ≤ 1 then n else (n + 1) / 2

/-- The initial position of the last remaining student -/
def last_student_position : ℕ := 128

/-- Theorem stating that the last remaining student had initial position 128 -/
theorem last_student_theorem :
  (Nat.iterate remove_odd_positions (Nat.log2 initial_students) initial_students) = last_student_position :=
sorry

end NUMINAMATH_CALUDE_last_student_theorem_l3274_327482


namespace NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l3274_327450

-- Define the variables
def before_dinner : ℕ := 22
def total : ℕ := 37

-- Define the theorem
theorem carrot_sticks_after_dinner :
  total - before_dinner = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l3274_327450


namespace NUMINAMATH_CALUDE_time_to_install_one_window_l3274_327441

/-- Proves that the time to install one window is 5 hours -/
theorem time_to_install_one_window
  (total_windows : ℕ)
  (installed_windows : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_windows = 10)
  (h2 : installed_windows = 6)
  (h3 : time_for_remaining = 20)
  : (time_for_remaining : ℚ) / (total_windows - installed_windows : ℚ) = 5 := by
  sorry


end NUMINAMATH_CALUDE_time_to_install_one_window_l3274_327441


namespace NUMINAMATH_CALUDE_sin_beta_value_l3274_327432

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 1 / 7)
  (h4 : Real.cos (α + β) = -11 / 14) :
  Real.sin β = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3274_327432


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l3274_327409

def f (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 21 / 7 ∧
  f r = 0 ∧
  ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l3274_327409


namespace NUMINAMATH_CALUDE_f_symmetry_l3274_327468

/-- Given a function f(x) = x^2005 + ax^3 - b/x - 8, where a and b are real constants,
    if f(-2) = 10, then f(2) = -26 -/
theorem f_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2005 + a*x^3 - b/x - 8
  f (-2) = 10 → f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_f_symmetry_l3274_327468


namespace NUMINAMATH_CALUDE_marshmallow_total_l3274_327481

def marshmallow_challenge (haley michael brandon : ℕ) : Prop :=
  haley = 8 ∧
  michael = 3 * haley ∧
  brandon = michael / 2 ∧
  haley + michael + brandon = 44

theorem marshmallow_total : ∃ (haley michael brandon : ℕ), marshmallow_challenge haley michael brandon :=
  sorry

end NUMINAMATH_CALUDE_marshmallow_total_l3274_327481


namespace NUMINAMATH_CALUDE_peach_distribution_problem_l3274_327417

/-- Represents the distribution of peaches among monkeys -/
structure PeachDistribution where
  total_peaches : ℕ
  num_monkeys : ℕ

/-- Checks if the distribution satisfies the first condition -/
def satisfies_condition1 (d : PeachDistribution) : Prop :=
  2 * 4 + (d.num_monkeys - 2) * 2 + 4 = d.total_peaches

/-- Checks if the distribution satisfies the second condition -/
def satisfies_condition2 (d : PeachDistribution) : Prop :=
  1 * 6 + (d.num_monkeys - 1) * 4 = d.total_peaches + 12

/-- The theorem to be proved -/
theorem peach_distribution_problem :
  ∃ (d : PeachDistribution),
    d.total_peaches = 26 ∧
    d.num_monkeys = 9 ∧
    satisfies_condition1 d ∧
    satisfies_condition2 d :=
by sorry

end NUMINAMATH_CALUDE_peach_distribution_problem_l3274_327417


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3274_327492

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of (2,3) and (10,7) lies on the line y = mx + b
    y = m * x + b ∧ 
    x = (2 + 10) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The slope of the line is perpendicular to the slope of the segment connecting (2,3) and (10,7)
    m * ((7 - 3) / (10 - 2)) = -1) →
  m + b = 15 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l3274_327492


namespace NUMINAMATH_CALUDE_complementary_angles_adjustment_l3274_327486

theorem complementary_angles_adjustment (x y : ℝ) (h1 : x + y = 90) (h2 : x / y = 3 / 7) :
  let new_x := x * 1.2
  let new_y := 90 - new_x
  (y - new_y) / y * 100 = 8.57143 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_adjustment_l3274_327486


namespace NUMINAMATH_CALUDE_fast_food_cost_l3274_327469

theorem fast_food_cost (burger shake cola : ℝ) : 
  (3 * burger + 7 * shake + cola = 120) →
  (4 * burger + 10 * shake + cola = 160.5) →
  (burger + shake + cola = 39) :=
by sorry

end NUMINAMATH_CALUDE_fast_food_cost_l3274_327469


namespace NUMINAMATH_CALUDE_circle_properties_l3274_327407

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the bisecting line
def bisecting_line (x y : ℝ) : Prop := x - y = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (∀ x y : ℝ, circle_equation x y → bisecting_line x y) ∧
    (∃ x y : ℝ, circle_equation x y ∧ tangent_line x y) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3274_327407


namespace NUMINAMATH_CALUDE_investment_satisfies_profit_ratio_q_investment_is_correct_l3274_327477

/-- Represents the investment amounts and profit ratio of two business partners -/
structure BusinessInvestment where
  p_investment : ℝ
  q_investment : ℝ
  p_profit_ratio : ℝ
  q_profit_ratio : ℝ

/-- The business investment scenario described in the problem -/
def problem_investment : BusinessInvestment where
  p_investment := 50000
  q_investment := 66666.67
  p_profit_ratio := 3
  q_profit_ratio := 4

/-- Theorem stating that the given investment amounts satisfy the profit ratio condition -/
theorem investment_satisfies_profit_ratio (bi : BusinessInvestment) :
  bi.p_investment / bi.q_investment = bi.p_profit_ratio / bi.q_profit_ratio →
  bi = problem_investment :=
by
  sorry

/-- Main theorem proving that q's investment is correct given the conditions -/
theorem q_investment_is_correct :
  ∃ (bi : BusinessInvestment),
    bi.p_investment = 50000 ∧
    bi.p_profit_ratio = 3 ∧
    bi.q_profit_ratio = 4 ∧
    bi.p_investment / bi.q_investment = bi.p_profit_ratio / bi.q_profit_ratio ∧
    bi.q_investment = 66666.67 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_satisfies_profit_ratio_q_investment_is_correct_l3274_327477


namespace NUMINAMATH_CALUDE_chairs_to_remove_l3274_327465

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 15

/-- The initial number of chairs set up -/
def initial_chairs : ℕ := 225

/-- The number of expected attendees -/
def expected_attendees : ℕ := 180

/-- Theorem: The number of chairs to be removed is 45 -/
theorem chairs_to_remove :
  ∃ (removed : ℕ),
    removed = initial_chairs - expected_attendees ∧
    removed % chairs_per_row = 0 ∧
    (initial_chairs - removed) ≥ expected_attendees ∧
    (initial_chairs - removed) % chairs_per_row = 0 ∧
    removed = 45 := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l3274_327465


namespace NUMINAMATH_CALUDE_repeating_decimal_one_point_foursix_equals_fraction_l3274_327487

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to its rational representation. -/
def repeating_decimal_to_rational (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (99 : ℚ)

theorem repeating_decimal_one_point_foursix_equals_fraction :
  repeating_decimal_to_rational ⟨1, 46⟩ = 145 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_one_point_foursix_equals_fraction_l3274_327487


namespace NUMINAMATH_CALUDE_total_tiles_on_floor_l3274_327410

/-- Represents a square floor with a border of tiles -/
structure BorderedFloor where
  /-- The side length of the square floor -/
  side_length : ℕ
  /-- The width of the border in tiles -/
  border_width : ℕ
  /-- The number of tiles in the border -/
  border_tiles : ℕ

/-- Theorem: Given a square floor with a 1-tile wide black border containing 204 tiles, 
    the total number of tiles on the floor is 2704 -/
theorem total_tiles_on_floor (floor : BorderedFloor) 
  (h1 : floor.border_width = 1)
  (h2 : floor.border_tiles = 204) : 
  floor.side_length^2 = 2704 := by
  sorry

#check total_tiles_on_floor

end NUMINAMATH_CALUDE_total_tiles_on_floor_l3274_327410


namespace NUMINAMATH_CALUDE_part_one_part_two_l3274_327488

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2 * x + 1|

-- Part I: Prove that when m = -1, f(x) ≤ 3 is equivalent to -1 ≤ x ≤ 1
theorem part_one : 
  ∀ x : ℝ, f (-1) x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

-- Part II: Prove that the minimum value of f(x) is |m - 1/2|
theorem part_two (m : ℝ) : 
  ∃ x : ℝ, ∀ y : ℝ, f m x ≤ f m y ∧ f m x = |m - 1/2| := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3274_327488


namespace NUMINAMATH_CALUDE_sqrt_squared_eq_self_sqrt_784_squared_l3274_327452

theorem sqrt_squared_eq_self (x : ℝ) (h : x ≥ 0) : (Real.sqrt x) ^ 2 = x := by sorry

theorem sqrt_784_squared : (Real.sqrt 784) ^ 2 = 784 := by sorry

end NUMINAMATH_CALUDE_sqrt_squared_eq_self_sqrt_784_squared_l3274_327452


namespace NUMINAMATH_CALUDE_distribute_seven_to_twelve_l3274_327449

/-- The number of ways to distribute distinct items to recipients -/
def distribute_ways (n_items : ℕ) (n_recipients : ℕ) : ℕ :=
  n_recipients ^ n_items

/-- Theorem: The number of ways to distribute 7 distinct items to 12 recipients,
    where each recipient can receive multiple items, is equal to 12^7 -/
theorem distribute_seven_to_twelve :
  distribute_ways 7 12 = 35831808 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_to_twelve_l3274_327449


namespace NUMINAMATH_CALUDE_line_y_intercept_l3274_327459

/-- A line with slope -3 and x-intercept (7,0) has y-intercept (0, 21) -/
theorem line_y_intercept (f : ℝ → ℝ) (h1 : ∀ x y, f y - f x = -3 * (y - x)) 
  (h2 : f 7 = 0) : f 0 = 21 := by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_l3274_327459


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l3274_327433

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 :=
sorry

theorem min_value_attained (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) = 1 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l3274_327433


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l3274_327476

theorem power_tower_mod_1000 : 5^(5^(5^5)) % 1000 = 125 := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l3274_327476


namespace NUMINAMATH_CALUDE_license_plate_count_l3274_327456

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- The total number of characters available (letters + digits) -/
def num_chars : ℕ := num_letters + num_digits

/-- The format of the license plate -/
inductive LicensePlateChar
| Letter
| Digit
| Any

/-- The structure of the license plate -/
def license_plate_format : List LicensePlateChar :=
  [LicensePlateChar.Letter, LicensePlateChar.Digit, LicensePlateChar.Any, LicensePlateChar.Digit]

/-- 
  The number of ways to create a 4-character license plate 
  where the format is a letter followed by a digit, then any character, and ending with a digit,
  ensuring that exactly two characters on the license plate are the same.
-/
theorem license_plate_count : 
  (num_letters * num_digits * num_chars) = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3274_327456


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3274_327427

theorem rectangle_max_area (length width : ℝ) :
  length > 0 → width > 0 → length + width = 18 →
  length * width ≤ 81 ∧
  (length * width = 81 ↔ length = 9 ∧ width = 9) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3274_327427


namespace NUMINAMATH_CALUDE_f_derivative_correct_l3274_327464

/-- The exponential function -/
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

/-- The function f(x) = e^(-2x) -/
noncomputable def f (x : ℝ) : ℝ := exp (-2 * x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := -2 * exp (-2 * x)

theorem f_derivative_correct :
  ∀ x : ℝ, deriv f x = f_derivative x :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_correct_l3274_327464


namespace NUMINAMATH_CALUDE_light_source_height_l3274_327458

/-- Given a cube with edge length 3 cm, illuminated by a light source x cm directly
    above and 3 cm horizontally from a top vertex, if the shadow area outside the
    cube's base is 75 square cm, then x = 7 cm. -/
theorem light_source_height (x : ℝ) : 
  let cube_edge : ℝ := 3
  let horizontal_distance : ℝ := 3
  let shadow_area : ℝ := 75
  let total_area : ℝ := cube_edge^2 + shadow_area
  let shadow_side : ℝ := Real.sqrt total_area
  let height_increase : ℝ := shadow_side - cube_edge
  x = (cube_edge * (cube_edge + height_increase)) / height_increase → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_light_source_height_l3274_327458


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l3274_327483

theorem not_divisible_by_169 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 7*n - 4 = 169*k := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l3274_327483


namespace NUMINAMATH_CALUDE_smallest_average_is_16_5_l3274_327435

/-- A function that generates all valid combinations of three single-digit
    and three double-digit numbers from the digits 1 to 9 without repetition -/
def generateValidCombinations : List (List ℕ) := sorry

/-- Calculates the average of a list of numbers -/
def average (numbers : List ℕ) : ℚ :=
  (numbers.sum : ℚ) / numbers.length

/-- Theorem stating that the smallest possible average is 16.5 -/
theorem smallest_average_is_16_5 :
  let allCombinations := generateValidCombinations
  let averages := allCombinations.map average
  averages.minimum? = some (33/2) := by sorry

end NUMINAMATH_CALUDE_smallest_average_is_16_5_l3274_327435


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3274_327489

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 3*x*y - 2 = 0) :
  ∃ (m : ℝ), m = 4/3 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + 3*a*b - 2 = 0 → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3274_327489


namespace NUMINAMATH_CALUDE_survey_sample_size_l3274_327462

/-- Represents a survey conducted on students -/
structure Survey where
  numSelected : ℕ

/-- Definition of sample size for a survey -/
def sampleSize (s : Survey) : ℕ := s.numSelected

/-- Theorem stating that the sample size of the survey is 200 -/
theorem survey_sample_size :
  ∃ (s : Survey), s.numSelected = 200 ∧ sampleSize s = 200 := by
  sorry

end NUMINAMATH_CALUDE_survey_sample_size_l3274_327462


namespace NUMINAMATH_CALUDE_incircle_radius_l3274_327416

/-- An isosceles triangle with base 10 and height 12 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  is_isosceles : base = 10 ∧ height = 12

/-- The incircle of a triangle -/
def incircle (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: The radius of the incircle of the given isosceles triangle is 10/3 -/
theorem incircle_radius (t : IsoscelesTriangle) : incircle t = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_l3274_327416


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_l3274_327463

theorem product_and_reciprocal_relation (x y : ℝ) :
  x > 0 ∧ y > 0 ∧ x * y = 12 ∧ 1 / x = 3 * (1 / y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_l3274_327463


namespace NUMINAMATH_CALUDE_multiples_count_l3274_327446

def count_multiples (n : ℕ) : ℕ := 
  (Finset.filter (λ x => x % 2 = 0 ∨ x % 3 = 0) (Finset.range (n + 1))).card

def count_multiples_not_five (n : ℕ) : ℕ := 
  (Finset.filter (λ x => (x % 2 = 0 ∨ x % 3 = 0) ∧ x % 5 ≠ 0) (Finset.range (n + 1))).card

theorem multiples_count : count_multiples_not_five 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_l3274_327446


namespace NUMINAMATH_CALUDE_metal_square_weight_relation_l3274_327442

/-- Represents the properties of a square metal slab -/
structure MetalSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two metal squares of the same material and thickness -/
theorem metal_square_weight_relation 
  (uniformDensity : ℝ → ℝ → ℝ) -- Function representing uniform density
  (square1 : MetalSquare) 
  (square2 : MetalSquare) 
  (h1 : square1.side_length = 4) 
  (h2 : square1.weight = 16) 
  (h3 : square2.side_length = 6) 
  (h4 : ∀ s w, uniformDensity s w = w / (s * s)) -- Density is weight divided by area
  : square2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_metal_square_weight_relation_l3274_327442


namespace NUMINAMATH_CALUDE_square_root_product_plus_one_l3274_327455

theorem square_root_product_plus_one (a : ℕ) (n : ℕ) : 
  a = 2020 ∧ n = 4086461 → a * (a + 1) * (a + 2) * (a + 3) + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_plus_one_l3274_327455


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l3274_327471

-- Define the functions
def y₁ (x : ℝ) : ℝ := x^2 + 2*x + 1
def y₂ (x b : ℝ) : ℝ := x^2 + b*x + 2
def y₃ (x c : ℝ) : ℝ := x^2 + c*x + 3

-- Define the number of roots for each function
def M₁ : ℕ := 1
def M₂ : ℕ := 1
def M₃ : ℕ := 2

-- Theorem statement
theorem intersection_points_theorem 
  (b c : ℝ) 
  (hb : b > 0) 
  (hc : c > 0) 
  (h_bc : b^2 = 2*c) 
  (h_M₁ : ∃! x, y₁ x = 0) 
  (h_M₂ : ∃! x, y₂ x b = 0) : 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ y₃ x₁ c = 0 ∧ y₃ x₂ c = 0 ∧ ∀ x, y₃ x c = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l3274_327471


namespace NUMINAMATH_CALUDE_drink_equality_l3274_327484

theorem drink_equality (x : ℝ) : 
  let eric_initial := x
  let sara_initial := 1.4 * x
  let eric_consumed := (2/3) * eric_initial
  let sara_consumed := (2/3) * sara_initial
  let eric_remaining := eric_initial - eric_consumed
  let sara_remaining := sara_initial - sara_consumed
  let transfer := (1/2) * sara_remaining + 3
  let eric_final := eric_consumed + transfer
  let sara_final := sara_consumed + (sara_remaining - transfer)
  eric_final = sara_final ∧ eric_final = 23 ∧ sara_final = 23 :=
by sorry

#check drink_equality

end NUMINAMATH_CALUDE_drink_equality_l3274_327484


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3274_327415

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let P : Point2D := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3274_327415


namespace NUMINAMATH_CALUDE_expression_evaluation_l3274_327490

theorem expression_evaluation : 3 - 5 * (6 - 2^3) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3274_327490


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3274_327451

/-- Given a hyperbola and a circle, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∀ x y : ℝ, (x - 2)^2 + y^2 = 1 → 
    ∃ k : ℝ, (x = k ∧ y = k * (b / a)) ∨ (x = -k ∧ y = k * (b / a))) →
  ∃ c : ℝ, c^2 = 3 ∧ (∀ x y : ℝ, x + c * y = 0 ∨ x - c * y = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3274_327451


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3274_327473

theorem quadratic_roots_problem (x₁ x₂ m : ℝ) : 
  (x₁^2 - 2*(m+1)*x₁ + m^2 - 3 = 0) →
  (x₂^2 - 2*(m+1)*x₂ + m^2 - 3 = 0) →
  (x₁^2 + x₂^2 - x₁*x₂ = 33) →
  (m = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3274_327473


namespace NUMINAMATH_CALUDE_ana_guarantee_l3274_327480

/-- The hat game setup -/
structure HatGame where
  n : ℕ
  h_n_gt_1 : n > 1

/-- The minimum number of correct guesses Ana can guarantee -/
def min_correct_guesses (game : HatGame) : ℕ :=
  (game.n - 1) / 2

/-- The theorem stating Ana's guarantee -/
theorem ana_guarantee (game : HatGame) :
  ∃ (strategy : Type),
    ∀ (bob_distribution : Type),
      ∃ (correct_guesses : ℕ),
        correct_guesses ≥ min_correct_guesses game :=
sorry


end NUMINAMATH_CALUDE_ana_guarantee_l3274_327480


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3274_327444

theorem cubic_equation_solutions :
  ∀ x : ℝ, (x ^ (1/3) = 15 / (8 - x ^ (1/3))) ↔ (x = 27 ∨ x = 125) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3274_327444


namespace NUMINAMATH_CALUDE_inequality_sine_square_l3274_327457

theorem inequality_sine_square (x : ℝ) (h : x ∈ Set.Ioo 0 (π / 2)) : 
  0 < (1 / Real.sin x ^ 2) - (1 / x ^ 2) ∧ (1 / Real.sin x ^ 2) - (1 / x ^ 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_sine_square_l3274_327457


namespace NUMINAMATH_CALUDE_cat_puppy_weight_difference_l3274_327440

theorem cat_puppy_weight_difference : 
  let num_puppies : ℕ := 4
  let puppy_weight : ℝ := 7.5
  let num_cats : ℕ := 14
  let cat_weight : ℝ := 2.5
  let total_puppy_weight := (num_puppies : ℝ) * puppy_weight
  let total_cat_weight := (num_cats : ℝ) * cat_weight
  total_cat_weight - total_puppy_weight = 5
  := by sorry

end NUMINAMATH_CALUDE_cat_puppy_weight_difference_l3274_327440


namespace NUMINAMATH_CALUDE_car_catch_up_time_l3274_327499

/-- The time it takes for a car to catch up to a truck on a highway -/
theorem car_catch_up_time (truck_speed car_speed : ℝ) (head_start : ℝ) : 
  truck_speed = 45 →
  car_speed = 60 →
  head_start = 1 →
  ∃ t : ℝ, t = 6 ∧ car_speed * t = truck_speed * (t + head_start) + truck_speed * head_start :=
by
  sorry


end NUMINAMATH_CALUDE_car_catch_up_time_l3274_327499


namespace NUMINAMATH_CALUDE_same_type_square_root_l3274_327422

theorem same_type_square_root (k : ℕ) : (∃ n : ℕ, 2 * k - 4 = 3 * n ^ 2) ↔ k = 8 := by sorry

end NUMINAMATH_CALUDE_same_type_square_root_l3274_327422


namespace NUMINAMATH_CALUDE_rectangle_to_cylinder_surface_area_l3274_327447

/-- The surface area of a cylinder formed by rolling a rectangle -/
def cylinderSurfaceArea (length width : Real) : Set Real :=
  let baseArea1 := Real.pi * (length / (2 * Real.pi))^2
  let baseArea2 := Real.pi * (width / (2 * Real.pi))^2
  let lateralArea := length * width
  {lateralArea + 2 * baseArea1, lateralArea + 2 * baseArea2}

theorem rectangle_to_cylinder_surface_area :
  cylinderSurfaceArea (4 * Real.pi) (8 * Real.pi) = {32 * Real.pi^2 + 8 * Real.pi, 32 * Real.pi^2 + 32 * Real.pi} := by
  sorry

#check rectangle_to_cylinder_surface_area

end NUMINAMATH_CALUDE_rectangle_to_cylinder_surface_area_l3274_327447


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3274_327454

theorem cryptarithmetic_puzzle (D E F G : ℕ) : 
  (∀ (X Y : ℕ), (X = D ∨ X = E ∨ X = F ∨ X = G) ∧ (Y = D ∨ Y = E ∨ Y = F ∨ Y = G) ∧ X ≠ Y → X ≠ Y) →
  F - E = D - 1 →
  D + E + F = 16 →
  F - E = D →
  G = F - E →
  G = 5 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3274_327454


namespace NUMINAMATH_CALUDE_sequence_second_term_l3274_327461

/-- Given a sequence {a_n} with sum of first n terms S_n, prove a_2 = 4 -/
theorem sequence_second_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * (a n - 1)) → a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_second_term_l3274_327461


namespace NUMINAMATH_CALUDE_victoria_remaining_balance_l3274_327439

/-- Calculates Victoria's remaining balance after shopping --/
theorem victoria_remaining_balance :
  let initial_amount : ℕ := 500
  let rice_price : ℕ := 20
  let rice_quantity : ℕ := 2
  let wheat_price : ℕ := 25
  let wheat_quantity : ℕ := 3
  let soda_price : ℕ := 150
  let soda_quantity : ℕ := 1
  let total_spent : ℕ := rice_price * rice_quantity + wheat_price * wheat_quantity + soda_price * soda_quantity
  let remaining_balance : ℕ := initial_amount - total_spent
  remaining_balance = 235 := by
  sorry

end NUMINAMATH_CALUDE_victoria_remaining_balance_l3274_327439


namespace NUMINAMATH_CALUDE_exponential_inequality_l3274_327453

theorem exponential_inequality (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  Real.exp ((x₁ + x₂) / 2) < (Real.exp x₁ + Real.exp x₂) / (x₁ - x₂) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3274_327453


namespace NUMINAMATH_CALUDE_consecutive_non_divisors_l3274_327479

theorem consecutive_non_divisors (n : ℕ) (k : ℕ) : 
  (∀ i ∈ Finset.range 250, i ≠ k ∧ i ≠ k + 1 → n % i = 0) →
  (n % k ≠ 0 ∧ n % (k + 1) ≠ 0) →
  1 ≤ k →
  k ≤ 249 →
  k = 127 := by
sorry

end NUMINAMATH_CALUDE_consecutive_non_divisors_l3274_327479


namespace NUMINAMATH_CALUDE_three_circles_middle_radius_l3274_327402

/-- Configuration of three circles with two common tangent lines -/
structure ThreeCirclesConfig where
  r_large : ℝ  -- radius of the largest circle
  r_small : ℝ  -- radius of the smallest circle
  r_middle : ℝ  -- radius of the middle circle
  tangent_lines : ℕ  -- number of common tangent lines

/-- Theorem: In a configuration of three circles with two common tangent lines,
    if the radius of the largest circle is 18 and the radius of the smallest circle is 8,
    then the radius of the middle circle is 12. -/
theorem three_circles_middle_radius 
  (config : ThreeCirclesConfig) 
  (h1 : config.r_large = 18) 
  (h2 : config.r_small = 8) 
  (h3 : config.tangent_lines = 2) : 
  config.r_middle = 12 := by
  sorry

end NUMINAMATH_CALUDE_three_circles_middle_radius_l3274_327402


namespace NUMINAMATH_CALUDE_tan_beta_value_l3274_327430

open Real

theorem tan_beta_value (α β : ℝ) 
  (h1 : tan (α + β) = 3) 
  (h2 : tan (α + π/4) = 2) : 
  tan β = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3274_327430


namespace NUMINAMATH_CALUDE_sachin_age_l3274_327423

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age * 12 = rahul_age * 5) :
  sachin_age = 5 := by
sorry

end NUMINAMATH_CALUDE_sachin_age_l3274_327423


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l3274_327421

theorem multiplication_division_equality : (3.6 * 0.25) / 0.5 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l3274_327421


namespace NUMINAMATH_CALUDE_guitar_ratio_l3274_327497

/-- The ratio of Davey's guitars to Barbeck's guitars is 1:1 -/
theorem guitar_ratio (davey barbeck : ℕ) : 
  davey = 18 → davey = barbeck → davey / barbeck = 1 := by
  sorry

end NUMINAMATH_CALUDE_guitar_ratio_l3274_327497


namespace NUMINAMATH_CALUDE_electrician_wage_l3274_327434

theorem electrician_wage (total_hours : ℝ) (bricklayer_wage : ℝ) (total_payment : ℝ) (individual_hours : ℝ)
  (h1 : total_hours = 90)
  (h2 : bricklayer_wage = 12)
  (h3 : total_payment = 1350)
  (h4 : individual_hours = 22.5) :
  (total_payment - bricklayer_wage * individual_hours) / individual_hours = 48 := by
sorry

end NUMINAMATH_CALUDE_electrician_wage_l3274_327434


namespace NUMINAMATH_CALUDE_tech_club_enrollment_l3274_327466

theorem tech_club_enrollment (total : ℕ) (cs : ℕ) (electronics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 90)
  (h3 : electronics = 60)
  (h4 : both = 20) :
  total - (cs + electronics - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tech_club_enrollment_l3274_327466


namespace NUMINAMATH_CALUDE_x_value_l3274_327467

theorem x_value (x : ℝ) (h1 : x > 0) (h2 : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3274_327467


namespace NUMINAMATH_CALUDE_inequality_relationship_l3274_327403

theorem inequality_relationship (x : ℝ) :
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l3274_327403


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_one_l3274_327498

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The lines ax + 2y - 2 = 0 and x + (a+1)y + 1 = 0 are parallel if and only if a = 1 -/
theorem lines_parallel_iff_a_eq_one (a : ℝ) :
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, 
    (a * x + 2 * y - 2 = 0 ↔ y = (-a/2) * x + b1) ∧ 
    (x + (a+1) * y + 1 = 0 ↔ y = (-1/(a+1)) * x + b2)) ↔ 
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_one_l3274_327498


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l3274_327448

theorem complex_addition_simplification :
  (4 : ℂ) + 3*I + (-7 : ℂ) + 5*I = -3 + 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l3274_327448


namespace NUMINAMATH_CALUDE_least_whole_number_ratio_l3274_327460

theorem least_whole_number_ratio (x : ℕ) : 
  (x > 0 ∧ (6 - x : ℚ) / (7 - x) < 16 / 21) ↔ x ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_least_whole_number_ratio_l3274_327460


namespace NUMINAMATH_CALUDE_birds_on_fence_l3274_327414

theorem birds_on_fence (initial_birds : ℕ) : 
  initial_birds + 8 = 20 → initial_birds = 12 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3274_327414


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_range_l3274_327424

theorem sine_cosine_inequality_range (θ : Real) :
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  Real.sin θ ^ 3 - Real.cos θ ^ 3 > (Real.cos θ ^ 5 - Real.sin θ ^ 5) / 7 →
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_range_l3274_327424


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3274_327428

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ 6 * (8 * x^2 + 7 * x + 11) - x * (8 * x - 45)
  ∃ x₀ : ℝ, f x₀ = 0 ∧ (∀ x : ℝ, f x = 0 → x₀ ≤ x) ∧ x₀ = -11/8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3274_327428


namespace NUMINAMATH_CALUDE_profit_equation_l3274_327413

/-- Given a profit equation P = (1/m)S - (1/n)C, prove that P = (m-n)/(mn) * S -/
theorem profit_equation (m n : ℝ) (m_ne_zero : m ≠ 0) (n_ne_zero : n ≠ 0) :
  ∀ (S C P : ℝ), P = (1/m) * S - (1/n) * C → P = (m-n)/(m*n) * S :=
by sorry

end NUMINAMATH_CALUDE_profit_equation_l3274_327413


namespace NUMINAMATH_CALUDE_students_6_to_8_hours_l3274_327470

/-- Represents a frequency distribution histogram for study times -/
structure StudyTimeHistogram where
  total_students : ℕ
  freq_6_to_8 : ℕ
  -- Other fields for other time intervals could be added here

/-- Theorem stating that in a given histogram of 100 students, 30 studied for 6 to 8 hours -/
theorem students_6_to_8_hours (h : StudyTimeHistogram) 
  (h_total : h.total_students = 100) : h.freq_6_to_8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_6_to_8_hours_l3274_327470


namespace NUMINAMATH_CALUDE_distribute_6_4_l3274_327494

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 4 indistinguishable boxes is 262 -/
theorem distribute_6_4 : distribute 6 4 = 262 := by sorry

end NUMINAMATH_CALUDE_distribute_6_4_l3274_327494


namespace NUMINAMATH_CALUDE_greatest_divisor_of_p_plus_one_l3274_327420

theorem greatest_divisor_of_p_plus_one (n : ℕ+) : 
  ∃ (d : ℕ), d = 6 ∧ 
  (∀ (p : ℕ), Prime p → p % 3 = 2 → ¬(p ∣ n) → d ∣ (p + 1)) ∧
  (∀ (k : ℕ), k > d → ∃ (p : ℕ), Prime p ∧ p % 3 = 2 ∧ ¬(p ∣ n) ∧ ¬(k ∣ (p + 1))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_p_plus_one_l3274_327420


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l3274_327485

/-- The minimum value of a for which x^2 + ax + 1 ≥ 0 holds for all x ∈ (0, 1] is -2 -/
theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l3274_327485


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3274_327405

theorem multiply_and_simplify (x : ℝ) : (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3274_327405


namespace NUMINAMATH_CALUDE_inscribed_dodecagon_radius_inscribed_dodecagon_radius_proof_l3274_327478

/-- The radius of a circle circumscribing a convex dodecagon with alternating side lengths of √2 and √24 is √38. -/
theorem inscribed_dodecagon_radius : ℝ → ℝ → ℝ → Prop :=
  fun (r : ℝ) (side1 : ℝ) (side2 : ℝ) =>
    side1 = Real.sqrt 2 ∧
    side2 = Real.sqrt 24 ∧
    r = Real.sqrt 38

/-- Proof of the theorem -/
theorem inscribed_dodecagon_radius_proof :
  ∃ (r : ℝ), inscribed_dodecagon_radius r (Real.sqrt 2) (Real.sqrt 24) :=
by
  sorry

#check inscribed_dodecagon_radius
#check inscribed_dodecagon_radius_proof

end NUMINAMATH_CALUDE_inscribed_dodecagon_radius_inscribed_dodecagon_radius_proof_l3274_327478


namespace NUMINAMATH_CALUDE_voucher_distribution_l3274_327431

-- Define the number of representatives and vouchers
def num_representatives : ℕ := 5
def num_vouchers : ℕ := 4

-- Define the distribution method
def distribution_method (n m : ℕ) : ℕ := Nat.choose n m

-- Theorem statement
theorem voucher_distribution :
  distribution_method num_representatives num_vouchers = 5 := by
  sorry

end NUMINAMATH_CALUDE_voucher_distribution_l3274_327431


namespace NUMINAMATH_CALUDE_count_sevens_up_to_2017_l3274_327425

/-- Count of digit 7 in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Sum of count_sevens for all numbers from 1 to n -/
def sum_count_sevens (n : ℕ) : ℕ := sorry

/-- The main theorem stating the count of digit 7 in numbers from 1 to 2017 -/
theorem count_sevens_up_to_2017 : sum_count_sevens 2017 = 602 := by sorry

end NUMINAMATH_CALUDE_count_sevens_up_to_2017_l3274_327425


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l3274_327495

/-- Given two points M and N in a plane, and P as the midpoint of MN, 
    prove that P has the specified coordinates. -/
theorem midpoint_coordinates (M N P : ℝ × ℝ) : 
  M = (3, -2) → N = (-5, -1) → P = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) → 
  P = (-1, -3/2) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l3274_327495


namespace NUMINAMATH_CALUDE_k_value_at_4_l3274_327475

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - x + 1

-- Define k as a function of h's roots
def k (α β γ : ℝ) (x : ℝ) : ℝ := -(x - α^3) * (x - β^3) * (x - γ^3)

theorem k_value_at_4 (α β γ : ℝ) :
  h α = 0 → h β = 0 → h γ = 0 →  -- α, β, γ are roots of h
  k α β γ 0 = 1 →                -- k(0) = 1
  k α β γ 4 = -61 :=             -- k(4) = -61
by sorry

end NUMINAMATH_CALUDE_k_value_at_4_l3274_327475


namespace NUMINAMATH_CALUDE_prasanna_speed_l3274_327408

-- Define the speeds and distance
def laxmi_speed : ℝ := 40
def total_distance : ℝ := 78
def time : ℝ := 1

-- Theorem to prove Prasanna's speed
theorem prasanna_speed : 
  ∃ (prasanna_speed : ℝ), 
    laxmi_speed * time + prasanna_speed * time = total_distance ∧ 
    prasanna_speed = 38 := by
  sorry

end NUMINAMATH_CALUDE_prasanna_speed_l3274_327408


namespace NUMINAMATH_CALUDE_a_range_f_result_l3274_327429

noncomputable section

variables (a x x₁ x₂ t : ℝ)

def f (x : ℝ) := Real.exp x - a * x + a

def f' (x : ℝ) := Real.exp x - a

axiom a_positive : a > 0

axiom x₁_less_x₂ : x₁ < x₂

axiom f_roots : f a x₁ = 0 ∧ f a x₂ = 0

axiom t_def : Real.sqrt ((x₂ - 1) / (x₁ - 1)) = t

axiom isosceles_right_triangle : ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo x₁ x₂ ∧ 
  f a x₀ = (x₁ - x₂) / 2

theorem a_range : a > Real.exp 2 := by sorry

theorem f'_negative : f' a (Real.sqrt (x₁ * x₂)) < 0 := by sorry

theorem result : (a - 1) * (t - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_a_range_f_result_l3274_327429


namespace NUMINAMATH_CALUDE_additional_grazing_area_l3274_327496

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 18^2 - π * 12^2 = 180 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l3274_327496


namespace NUMINAMATH_CALUDE_pictures_deleted_vacation_pictures_deleted_l3274_327474

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) :
  zoo_pics + museum_pics - remaining_pics =
  (zoo_pics + museum_pics) - remaining_pics :=
by sorry

theorem vacation_pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) :
  zoo_pics = 15 →
  museum_pics = 18 →
  remaining_pics = 2 →
  zoo_pics + museum_pics - remaining_pics = 31 :=
by sorry

end NUMINAMATH_CALUDE_pictures_deleted_vacation_pictures_deleted_l3274_327474


namespace NUMINAMATH_CALUDE_vector_calculation_l3274_327426

/-- Given vectors a, b, c, and e in a vector space, 
    where a = 5e, b = -3e, and c = 4e,
    prove that 2a - 3b + c = 23e -/
theorem vector_calculation 
  (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e : V) 
  (a b c : V) 
  (ha : a = 5 • e) 
  (hb : b = -3 • e) 
  (hc : c = 4 • e) : 
  2 • a - 3 • b + c = 23 • e := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l3274_327426


namespace NUMINAMATH_CALUDE_no_solution_for_equal_ratios_l3274_327400

theorem no_solution_for_equal_ratios :
  ¬∃ (x : ℝ), (4 + x) / (5 + x) = (1 + x) / (2 + x) := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_equal_ratios_l3274_327400


namespace NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l3274_327418

/-- Given that x is the largest solution to the equation log_{2x^3} 2 + log_{4x^4} 2 = -1,
    prove that 1/x^6 = 4 -/
theorem largest_solution_reciprocal_sixth_power (x : ℝ) 
  (h : x > 0)
  (eq : Real.log 2 / Real.log (2 * x^3) + Real.log 2 / Real.log (4 * x^4) = -1)
  (largest : ∀ y > 0, Real.log 2 / Real.log (2 * y^3) + Real.log 2 / Real.log (4 * y^4) = -1 → y ≤ x) :
  1 / x^6 = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l3274_327418


namespace NUMINAMATH_CALUDE_y2_greater_than_y1_l3274_327445

/-- The parabola equation y = x² - 2x + 3 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 2*x + 3

theorem y2_greater_than_y1 (y1 y2 : ℝ) 
  (h1 : parabola (-1) y1)
  (h2 : parabola (-2) y2) : 
  y2 > y1 := by
  sorry

end NUMINAMATH_CALUDE_y2_greater_than_y1_l3274_327445


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l3274_327411

theorem maintenance_check_increase (original_time new_time : ℝ) 
  (h1 : original_time = 50)
  (h2 : new_time = 60) :
  (new_time - original_time) / original_time * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l3274_327411


namespace NUMINAMATH_CALUDE_absolute_difference_bound_l3274_327406

theorem absolute_difference_bound (x y s t : ℝ) 
  (hx : |x - s| < t) (hy : |y - s| < t) : |x - y| < 2*t := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_bound_l3274_327406
