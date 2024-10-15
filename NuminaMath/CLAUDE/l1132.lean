import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_expression_a_range_l1132_113281

-- Define the quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for the inequality solution
def inequality_solution (a b c : ℝ) : Prop :=
  ∀ x, quadratic_function a b c x > -2 * x ↔ 1 < x ∧ x < 3

-- Theorem 1
theorem quadratic_expression
  (a b c : ℝ)
  (h1 : inequality_solution a b c)
  (h2 : ∃ x, quadratic_function a b c x + 6 * a = 0 ∧
              ∀ y, quadratic_function a b c y + 6 * a = 0 → y = x) :
  ∃ x, quadratic_function (-1/5) (-6/5) (-3/5) x = quadratic_function a b c x :=
sorry

-- Theorem 2
theorem a_range
  (a b c : ℝ)
  (h1 : inequality_solution a b c)
  (h2 : ∃ m, ∀ x, quadratic_function a b c x ≤ m ∧ m > 0) :
  a > -2 + Real.sqrt 3 ∨ a < -2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_expression_a_range_l1132_113281


namespace NUMINAMATH_CALUDE_choose_two_from_ten_l1132_113297

/-- The number of ways to choose 2 colors out of 10 colors -/
def choose_colors (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: Choosing 2 colors out of 10 results in 45 combinations -/
theorem choose_two_from_ten :
  choose_colors 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_ten_l1132_113297


namespace NUMINAMATH_CALUDE_ramanujan_number_l1132_113208

theorem ramanujan_number (h r : ℂ) : 
  h * r = 40 - 24 * I ∧ h = 4 + 4 * I → r = 2 - 8 * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l1132_113208


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l1132_113289

/-- Given a geometric sequence {a_n} where a_4 = 7 and a_6 = 21, prove that a_8 = 63 -/
theorem geometric_sequence_a8 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a4 : a 4 = 7) (h_a6 : a 6 = 21) : a 8 = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l1132_113289


namespace NUMINAMATH_CALUDE_cubes_volume_percentage_l1132_113235

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : Cube) : ℕ :=
  cube.sideLength * cube.sideLength * cube.sideLength

/-- Calculates the number of cubes that can fit along a given dimension -/
def cubesFitInDimension (dimension : ℕ) (cube : Cube) : ℕ :=
  dimension / cube.sideLength

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubesFit (box : BoxDimensions) (cube : Cube) : ℕ :=
  (cubesFitInDimension box.length cube) *
  (cubesFitInDimension box.width cube) *
  (cubesFitInDimension box.height cube)

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a 8x6x12 inch box is 66.67% -/
theorem cubes_volume_percentage :
  let box := BoxDimensions.mk 8 6 12
  let cube := Cube.mk 4
  let cubesVolume := (totalCubesFit box cube) * (cubeVolume cube)
  let totalVolume := boxVolume box
  let percentage := (cubesVolume : ℚ) / (totalVolume : ℚ) * 100
  percentage = 200/3 := by
  sorry

end NUMINAMATH_CALUDE_cubes_volume_percentage_l1132_113235


namespace NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l1132_113257

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of Mets to Red Sox fans -/
theorem mets_to_red_sox_ratio (fans : FanCount) 
  (total_fans : fans.yankees + fans.mets + fans.red_sox = 330)
  (yankees_to_mets : Ratio)
  (yankees_mets_ratio : yankees_to_mets.numerator * fans.mets = yankees_to_mets.denominator * fans.yankees)
  (yankees_mets_values : yankees_to_mets.numerator = 3 ∧ yankees_to_mets.denominator = 2)
  (mets_count : fans.mets = 88) :
  ∃ (r : Ratio), r.numerator = 4 ∧ r.denominator = 5 ∧ 
    r.numerator * fans.red_sox = r.denominator * fans.mets :=
sorry

end NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l1132_113257


namespace NUMINAMATH_CALUDE_fraction_sum_equals_negative_one_l1132_113244

theorem fraction_sum_equals_negative_one (a : ℝ) (h : 1 - 2*a ≠ 0) :
  a / (1 - 2*a) + (a - 1) / (1 - 2*a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_negative_one_l1132_113244


namespace NUMINAMATH_CALUDE_largest_special_number_l1132_113226

def has_distinct_digits (n : ℕ) : Prop := sorry

def divisible_by_digits (n : ℕ) : Prop := sorry

def contains_digit (n : ℕ) (d : ℕ) : Prop := sorry

theorem largest_special_number :
  ∀ n : ℕ,
    has_distinct_digits n ∧
    divisible_by_digits n ∧
    contains_digit n 5 →
    n ≤ 9735 :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_l1132_113226


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1132_113237

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_arithmetic : a 1 - (1/2 * a 3) = (1/2 * a 3) - (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1132_113237


namespace NUMINAMATH_CALUDE_percentage_problem_l1132_113294

theorem percentage_problem (x : ℝ) (h : 25 = 0.4 * x) : x = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1132_113294


namespace NUMINAMATH_CALUDE_complex_number_equality_l1132_113265

theorem complex_number_equality : ∃ (z : ℂ), z = (2 * Complex.I) / (1 - Complex.I) ∧ z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1132_113265


namespace NUMINAMATH_CALUDE_remainder_of_polynomial_l1132_113223

theorem remainder_of_polynomial (n : ℤ) (k : ℤ) : 
  n = 100 * k - 1 → (n^2 + 3*n + 4) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_polynomial_l1132_113223


namespace NUMINAMATH_CALUDE_larger_integer_proof_l1132_113224

theorem larger_integer_proof (x : ℤ) : 
  (x > 0) →  -- Ensure x is positive
  (x + 6 : ℚ) / (4 * x : ℚ) = 1 / 3 → 
  4 * x = 72 :=
by sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l1132_113224


namespace NUMINAMATH_CALUDE_twin_birthday_product_l1132_113268

theorem twin_birthday_product (age : ℕ) (h : age = 5) :
  (age + 1) * (age + 1) - age * age = 11 := by
  sorry

end NUMINAMATH_CALUDE_twin_birthday_product_l1132_113268


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l1132_113256

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l1132_113256


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1132_113262

/-- Given a telescope that increases the visual range by 150 percent from an original range of 60 kilometers, 
    the new visual range is 150 kilometers. -/
theorem telescope_visual_range : 
  let original_range : ℝ := 60
  let increase_percent : ℝ := 150
  let new_range : ℝ := original_range * (1 + increase_percent / 100)
  new_range = 150 := by sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1132_113262


namespace NUMINAMATH_CALUDE_vikki_take_home_pay_l1132_113219

def vikki_problem (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : Prop :=
  let gross_earnings := hours_worked * hourly_rate
  let tax_deduction := tax_rate * gross_earnings
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := tax_deduction + insurance_deduction + union_dues
  let take_home_pay := gross_earnings - total_deductions
  take_home_pay = 310

theorem vikki_take_home_pay :
  vikki_problem 42 10 (20/100) (5/100) 5 :=
sorry

end NUMINAMATH_CALUDE_vikki_take_home_pay_l1132_113219


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1132_113288

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^10 + x^8 - 6*x^6 + 27*x^4 + 64) ≤ 1/8.38 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1132_113288


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1132_113220

-- Define the initial conditions
def initial_purchase_price : ℝ := 10
def initial_selling_price : ℝ := 18
def initial_daily_sales : ℝ := 60

-- Define the price-sales relationships
def price_increase_effect (price_change : ℝ) : ℝ := -5 * price_change
def price_decrease_effect (price_change : ℝ) : ℝ := 10 * price_change

-- Define the profit functions
def profit_function_high (x : ℝ) : ℝ := -5 * (x - 20)^2 + 500
def profit_function_low (x : ℝ) : ℝ := -10 * (x - 17)^2 + 490

-- Theorem statement
theorem optimal_selling_price :
  ∃ (x : ℝ), x = 20 ∧
  ∀ (y : ℝ), y ≥ initial_selling_price →
    profit_function_high y ≤ profit_function_high x ∧
  ∀ (z : ℝ), z < initial_selling_price →
    profit_function_low z ≤ profit_function_high x :=
by sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l1132_113220


namespace NUMINAMATH_CALUDE_triangle_angle_not_greater_than_60_l1132_113203

theorem triangle_angle_not_greater_than_60 (a b c : ℝ) (h_triangle : a + b + c = 180) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_greater_than_60_l1132_113203


namespace NUMINAMATH_CALUDE_sodium_reduction_proof_l1132_113290

def salt_teaspoons : ℕ := 2
def initial_parmesan_ounces : ℕ := 8
def sodium_per_salt_teaspoon : ℕ := 50
def sodium_per_parmesan_ounce : ℕ := 25
def reduction_factor : ℚ := 1/3

def total_sodium (parmesan_ounces : ℕ) : ℕ :=
  salt_teaspoons * sodium_per_salt_teaspoon + parmesan_ounces * sodium_per_parmesan_ounce

def reduced_parmesan_ounces : ℕ := initial_parmesan_ounces - 4

theorem sodium_reduction_proof :
  (total_sodium initial_parmesan_ounces : ℚ) * (1 - reduction_factor) =
  (total_sodium reduced_parmesan_ounces : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sodium_reduction_proof_l1132_113290


namespace NUMINAMATH_CALUDE_car_travel_distance_l1132_113298

/-- Proves that two cars traveling at different speeds for different times cover the same distance of 600 miles -/
theorem car_travel_distance :
  ∀ (distance : ℝ) (time_R : ℝ) (speed_R : ℝ),
    speed_R = 50 →
    distance = speed_R * time_R →
    distance = (speed_R + 10) * (time_R - 2) →
    distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l1132_113298


namespace NUMINAMATH_CALUDE_speed_ratio_is_four_thirds_l1132_113250

/-- Two runners in a race where one gets a head start -/
structure Race where
  length : ℝ
  speed_a : ℝ
  speed_b : ℝ
  head_start : ℝ

/-- The race ends in a dead heat -/
def dead_heat (r : Race) : Prop :=
  r.length / r.speed_a = (r.length - r.head_start) / r.speed_b

/-- The head start is 0.25 of the race length -/
def quarter_head_start (r : Race) : Prop :=
  r.head_start = 0.25 * r.length

theorem speed_ratio_is_four_thirds (r : Race) 
  (h1 : dead_heat r) (h2 : quarter_head_start r) : 
  r.speed_a / r.speed_b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_four_thirds_l1132_113250


namespace NUMINAMATH_CALUDE_exponent_product_square_l1132_113263

theorem exponent_product_square (x y : ℝ) : (3 * x * y)^2 = 9 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_square_l1132_113263


namespace NUMINAMATH_CALUDE_one_third_green_faces_iff_three_l1132_113225

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of green faces on unit cubes after cutting a large painted cube -/
def green_faces (c : Cube n) : ℕ := 6 * n^2

/-- The total number of faces on all unit cubes after cutting -/
def total_faces (c : Cube n) : ℕ := 6 * n^3

/-- Theorem stating that exactly one-third of faces are green iff n = 3 -/
theorem one_third_green_faces_iff_three (c : Cube n) :
  3 * green_faces c = total_faces c ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_one_third_green_faces_iff_three_l1132_113225


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_point_five_l1132_113217

theorem floor_plus_self_eq_seventeen_point_five (s : ℝ) : 
  ⌊s⌋ + s = 17.5 ↔ s = 8.5 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_point_five_l1132_113217


namespace NUMINAMATH_CALUDE_snow_probability_l1132_113252

theorem snow_probability (p : ℝ) (n : ℕ) 
  (h_p : p = 3/4) 
  (h_n : n = 4) : 
  1 - (1 - p)^n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1132_113252


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_18_times_5_smallest_positive_multiple_is_90_l1132_113272

theorem smallest_positive_multiple_of_18_times_5 :
  ∀ n : ℕ+, n * (18 * 5) ≥ 18 * 5 :=
by
  sorry

theorem smallest_positive_multiple_is_90 :
  ∃ (n : ℕ+), n * (18 * 5) = 90 ∧ ∀ (m : ℕ+), m * (18 * 5) ≥ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_18_times_5_smallest_positive_multiple_is_90_l1132_113272


namespace NUMINAMATH_CALUDE_situp_rate_difference_l1132_113254

-- Define the given conditions
def diana_rate : ℕ := 4
def diana_situps : ℕ := 40
def total_situps : ℕ := 110

-- Define the theorem
theorem situp_rate_difference : ℕ := by
  -- The difference between Hani's and Diana's situp rates is 3
  sorry

end NUMINAMATH_CALUDE_situp_rate_difference_l1132_113254


namespace NUMINAMATH_CALUDE_unattainable_value_l1132_113238

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃ x, (2 - x) / (3 * x + 4) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_unattainable_value_l1132_113238


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1132_113200

theorem quadratic_inequality_condition (x : ℝ) : 
  (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1132_113200


namespace NUMINAMATH_CALUDE_min_box_value_l1132_113296

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + a) = 45*x^2 + Box*x + 45) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  ∀ Box', (∀ x, (∃ a' b', a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
                 (a'*x + b') * (b'*x + a') = 45*x^2 + Box'*x + 45)) →
  Box' ≥ Box →
  Box ≥ 106 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l1132_113296


namespace NUMINAMATH_CALUDE_largest_angle_right_triangle_l1132_113201

theorem largest_angle_right_triangle (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 90) (h3 : b / c = 7 / 2) : max a (max b c) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_right_triangle_l1132_113201


namespace NUMINAMATH_CALUDE_diana_work_hours_l1132_113277

/-- Represents Diana's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday combined
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday combined
  weekly_earnings : ℕ    -- Weekly earnings in dollars
  hourly_rate : ℕ        -- Hourly rate in dollars

/-- Theorem stating Diana's work hours on Monday, Wednesday, and Friday --/
theorem diana_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.tue_thu_hours = 30)  -- 15 hours each on Tuesday and Thursday
  (h2 : schedule.weekly_earnings = 1800)
  (h3 : schedule.hourly_rate = 30)
  : schedule.mon_wed_fri_hours = 30 := by
  sorry


end NUMINAMATH_CALUDE_diana_work_hours_l1132_113277


namespace NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l1132_113295

/-- Given the cost of copying 5 pages is 7 cents, this theorem proves that
    the number of pages that can be copied for $35 is 2500. -/
theorem pages_copied_for_35_dollars : 
  let cost_per_5_pages : ℚ := 7 / 100  -- 7 cents in dollars
  let dollars : ℚ := 35
  let pages_per_dollar : ℚ := 5 / cost_per_5_pages
  ⌊dollars * pages_per_dollar⌋ = 2500 :=
by sorry

end NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l1132_113295


namespace NUMINAMATH_CALUDE_pentagon_covers_half_l1132_113251

/-- Represents a tiling of a plane with large squares -/
structure PlaneTiling where
  /-- The number of smaller squares in each row/column of a large square -/
  grid_size : ℕ
  /-- The number of smaller squares that are part of pentagons in each large square -/
  pentagon_squares : ℕ

/-- The percentage of the plane enclosed by pentagons -/
def pentagon_percentage (tiling : PlaneTiling) : ℚ :=
  (tiling.pentagon_squares : ℚ) / (tiling.grid_size^2 : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 50% -/
theorem pentagon_covers_half (tiling : PlaneTiling) 
  (h1 : tiling.grid_size = 4)
  (h2 : tiling.pentagon_squares = 8) : 
  pentagon_percentage tiling = 50 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_covers_half_l1132_113251


namespace NUMINAMATH_CALUDE_quadratic_sum_l1132_113205

/-- Given a quadratic function f(x) = ax^2 + bx + c where a = 2, b = -3, c = 4,
    and f(1) = 3, prove that 2a - b + c = 11 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℤ) :
  a = 2 ∧ b = -3 ∧ c = 4 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  f 1 = 3 →
  2 * a - b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1132_113205


namespace NUMINAMATH_CALUDE_square_root_of_16_l1132_113202

theorem square_root_of_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_16_l1132_113202


namespace NUMINAMATH_CALUDE_mutually_inscribed_tetrahedra_exist_l1132_113267

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a tetrahedron as a set of four points
structure Tetrahedron where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

-- Define the property of being coplanar
def coplanar (p q r s : Point3D) : Prop := sorry

-- Define the property of a tetrahedron being inscribed in another
def inscribed (t1 t2 : Tetrahedron) : Prop :=
  coplanar t2.a t1.b t1.c t1.d ∧
  coplanar t2.b t1.a t1.c t1.d ∧
  coplanar t2.c t1.a t1.b t1.d ∧
  coplanar t2.d t1.a t1.b t1.c

-- Define the property of two tetrahedra not sharing vertices
def no_shared_vertices (t1 t2 : Tetrahedron) : Prop :=
  t1.a ≠ t2.a ∧ t1.a ≠ t2.b ∧ t1.a ≠ t2.c ∧ t1.a ≠ t2.d ∧
  t1.b ≠ t2.a ∧ t1.b ≠ t2.b ∧ t1.b ≠ t2.c ∧ t1.b ≠ t2.d ∧
  t1.c ≠ t2.a ∧ t1.c ≠ t2.b ∧ t1.c ≠ t2.c ∧ t1.c ≠ t2.d ∧
  t1.d ≠ t2.a ∧ t1.d ≠ t2.b ∧ t1.d ≠ t2.c ∧ t1.d ≠ t2.d

-- The theorem to be proved
theorem mutually_inscribed_tetrahedra_exist : 
  ∃ (t1 t2 : Tetrahedron), inscribed t1 t2 ∧ inscribed t2 t1 ∧ no_shared_vertices t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_mutually_inscribed_tetrahedra_exist_l1132_113267


namespace NUMINAMATH_CALUDE_ship_length_in_emily_steps_l1132_113284

theorem ship_length_in_emily_steps :
  ∀ (emily_speed ship_speed : ℝ) (emily_steps_forward emily_steps_backward : ℕ),
    emily_speed > ship_speed →
    emily_steps_forward = 300 →
    emily_steps_backward = 60 →
    ship_speed > 0 →
    ∃ (ship_length : ℝ),
      ship_length = emily_steps_forward * emily_speed / (emily_speed + ship_speed) +
                    emily_steps_backward * emily_speed / (emily_speed - ship_speed) ∧
      ship_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_ship_length_in_emily_steps_l1132_113284


namespace NUMINAMATH_CALUDE_building_shadow_length_l1132_113292

/-- Given a flagpole and a building under similar shadow-casting conditions,
    calculate the length of the building's shadow. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_height_pos : 0 < building_height)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building : building_height = 20) :
  (building_height * flagpole_shadow) / flagpole_height = 50 := by
  sorry


end NUMINAMATH_CALUDE_building_shadow_length_l1132_113292


namespace NUMINAMATH_CALUDE_overtime_to_regular_pay_ratio_l1132_113271

/-- Proves that the ratio of overtime to regular pay rate is 2:1 given the problem conditions --/
theorem overtime_to_regular_pay_ratio :
  ∀ (regular_rate overtime_rate total_pay : ℚ) (regular_hours overtime_hours : ℕ),
    regular_rate = 3 →
    regular_hours = 40 →
    overtime_hours = 12 →
    total_pay = 192 →
    total_pay = regular_rate * regular_hours + overtime_rate * overtime_hours →
    overtime_rate / regular_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_overtime_to_regular_pay_ratio_l1132_113271


namespace NUMINAMATH_CALUDE_no_convex_function_exists_l1132_113273

theorem no_convex_function_exists : 
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
by sorry

end NUMINAMATH_CALUDE_no_convex_function_exists_l1132_113273


namespace NUMINAMATH_CALUDE_hannah_seashell_distribution_l1132_113246

theorem hannah_seashell_distribution (noah liam hannah : ℕ) : 
  hannah = 4 * liam ∧ 
  liam = 3 * noah → 
  (7 : ℚ) / 36 = (hannah + liam + noah) / 3 - liam / hannah :=
by sorry

end NUMINAMATH_CALUDE_hannah_seashell_distribution_l1132_113246


namespace NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l1132_113213

theorem alternating_sum_of_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
  (∀ x : ℝ, (x + 1)^2 * (x^2 - 7)^3 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
                                      a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + 
                                      a₇*(x+2)^7 + a₈*(x+2)^8) →
  a₁ - a₂ + a₃ - a₄ + a₅ - a₆ + a₇ = -58 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l1132_113213


namespace NUMINAMATH_CALUDE_sausage_pieces_l1132_113230

/-- Given a sausage with red, yellow, and green rings, prove that cutting along all rings results in 21 pieces. -/
theorem sausage_pieces (red_pieces yellow_pieces green_pieces : ℕ) 
  (h_red : red_pieces = 5)
  (h_yellow : yellow_pieces = 7)
  (h_green : green_pieces = 11) :
  red_pieces + yellow_pieces + green_pieces - 2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_sausage_pieces_l1132_113230


namespace NUMINAMATH_CALUDE_sequence_sum_2017_l1132_113243

theorem sequence_sum_2017 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = 2 * n - 1) →
  (∀ n : ℕ, n > 0 → S n = S (n - 1) + a n) →
  a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_2017_l1132_113243


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l1132_113279

theorem complex_arithmetic_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l1132_113279


namespace NUMINAMATH_CALUDE_f_properties_l1132_113229

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
    ∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  (∀ k : ℤ, ∀ x : ℝ, -π/3 + k * π ≤ x ∧ x ≤ π/6 + k * π → 
    ∀ y : ℝ, -π/3 + k * π ≤ y ∧ y ≤ x → f y ≤ f x) ∧
  (∀ A B C a b c : ℝ, 
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    A + B + C = π ∧
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
    (a + 2*c) * Real.cos B = -b * Real.cos A →
    2 < f A ∧ f A ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1132_113229


namespace NUMINAMATH_CALUDE_smallest_cubic_divisible_by_810_l1132_113207

theorem smallest_cubic_divisible_by_810 : ∃ (a : ℕ), 
  (∀ (n : ℕ), n < a → ¬(∃ (k : ℕ), n = k^3 ∧ 810 ∣ n)) ∧
  (∃ (k : ℕ), a = k^3) ∧ 
  (810 ∣ a) ∧
  a = 729000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cubic_divisible_by_810_l1132_113207


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1132_113227

def M : Set ℝ := {x | 2 * x - 1 > 0}
def N : Set ℝ := {x | Real.sqrt x < 2}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1/2 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1132_113227


namespace NUMINAMATH_CALUDE_phone_call_probability_l1132_113249

/-- The probability of answering a phone call at the first ring -/
def p_first : ℝ := 0.1

/-- The probability of answering a phone call at the second ring -/
def p_second : ℝ := 0.2

/-- The probability of answering a phone call at the third ring -/
def p_third : ℝ := 0.4

/-- The probability of answering a phone call at the fourth ring -/
def p_fourth : ℝ := 0.1

/-- The events of answering at each ring are mutually exclusive -/
axiom mutually_exclusive : True

/-- The probability of answering a phone call within the first four rings -/
def p_within_four_rings : ℝ := p_first + p_second + p_third + p_fourth

theorem phone_call_probability : p_within_four_rings = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_phone_call_probability_l1132_113249


namespace NUMINAMATH_CALUDE_sum_of_odd_terms_l1132_113221

theorem sum_of_odd_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n, S n = n^2 + n) → 
  a 1 + a 3 + a 5 + a 7 + a 9 = 50 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_terms_l1132_113221


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l1132_113286

theorem complex_subtraction_simplification :
  (4 - 3 * Complex.I) - (7 - 5 * Complex.I) = -3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l1132_113286


namespace NUMINAMATH_CALUDE_vacation_tents_l1132_113266

/-- Calculates the number of tents needed given the total number of people,
    the number of people the house can sleep, and the number of people per tent. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (people_per_tent : ℕ) : ℕ :=
  ((total_people - house_capacity) + (people_per_tent - 1)) / people_per_tent

theorem vacation_tents :
  tents_needed 14 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_tents_l1132_113266


namespace NUMINAMATH_CALUDE_set_union_problem_l1132_113210

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1132_113210


namespace NUMINAMATH_CALUDE_bouquets_calculation_l1132_113231

/-- Given the initial number of flowers, flowers per bouquet, and wilted flowers,
    calculates the number of bouquets that can be made. -/
def calculate_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

/-- Proves that given 88 initial flowers, 5 flowers per bouquet, and 48 wilted flowers,
    the number of bouquets that can be made is equal to 8. -/
theorem bouquets_calculation :
  calculate_bouquets 88 5 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bouquets_calculation_l1132_113231


namespace NUMINAMATH_CALUDE_circle_center_l1132_113218

/-- The polar equation of a circle is given by ρ = √2(cos θ + sin θ).
    This theorem proves that the center of this circle is at the point (1, π/4) in polar coordinates. -/
theorem circle_center (ρ θ : ℝ) : 
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ) → 
  ∃ (r θ₀ : ℝ), r = 1 ∧ θ₀ = π / 4 ∧ 
    (∀ (x y : ℝ), x = r * Real.cos θ₀ ∧ y = r * Real.sin θ₀ → 
      (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1132_113218


namespace NUMINAMATH_CALUDE_root_sum_problem_l1132_113253

theorem root_sum_problem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → 
  b^2 - 5*b + 6 = 0 → 
  a^3 + a^4*b^2 + a^2*b^4 + b^3 + a*b*(a+b) = 533 := by
sorry

end NUMINAMATH_CALUDE_root_sum_problem_l1132_113253


namespace NUMINAMATH_CALUDE_exponential_decreasing_l1132_113275

theorem exponential_decreasing (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_l1132_113275


namespace NUMINAMATH_CALUDE_equation_solutions_l1132_113241

theorem equation_solutions :
  (∀ x, x * (x - 5) = 3 * x - 15 ↔ x = 5 ∨ x = 3) ∧
  (∀ y, 2 * y^2 - 9 * y + 5 = 0 ↔ y = (9 + Real.sqrt 41) / 4 ∨ y = (9 - Real.sqrt 41) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1132_113241


namespace NUMINAMATH_CALUDE_complement_of_A_l1132_113206

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ x^2 + x - 2 < 0}

theorem complement_of_A : (U \ A) = {-2, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1132_113206


namespace NUMINAMATH_CALUDE_greater_solution_quadratic_l1132_113299

theorem greater_solution_quadratic (x : ℝ) : 
  (x^2 + 15*x - 54 = 0) → (∃ y : ℝ, y^2 + 15*y - 54 = 0 ∧ y ≠ x) → 
  (x ≥ y ↔ x = 3) :=
sorry

end NUMINAMATH_CALUDE_greater_solution_quadratic_l1132_113299


namespace NUMINAMATH_CALUDE_angle_sum_equals_pi_l1132_113274

theorem angle_sum_equals_pi (x y : Real) : 
  x > 0 → x < π / 2 → y > 0 → y < π / 2 →
  4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 2 →
  4 * Real.cos (2 * x) + 3 * Real.cos (2 * y) = 1 →
  2 * x + y = π := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equals_pi_l1132_113274


namespace NUMINAMATH_CALUDE_geometric_sequence_determinant_l1132_113239

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_determinant
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a5 : a 5 = 3) :
  let det := a 2 * a 8 + a 7 * a 3
  det = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_determinant_l1132_113239


namespace NUMINAMATH_CALUDE_jar_weight_percentage_l1132_113212

theorem jar_weight_percentage (jar_weight bean_weight : ℝ) 
  (h1 : jar_weight + 0.5 * bean_weight = 0.6 * (jar_weight + bean_weight)) : 
  jar_weight / (jar_weight + bean_weight) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_jar_weight_percentage_l1132_113212


namespace NUMINAMATH_CALUDE_tangent_segments_area_l1132_113261

/-- The area of the region formed by all line segments of length 2 units 
    tangent to a circle with radius 3 units is equal to 4π. -/
theorem tangent_segments_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 2) : 
  let outer_radius := Real.sqrt (r^2 + l^2)
  π * outer_radius^2 - π * r^2 = 4 * π := by
sorry

end NUMINAMATH_CALUDE_tangent_segments_area_l1132_113261


namespace NUMINAMATH_CALUDE_two_numbers_sum_l1132_113283

theorem two_numbers_sum : ∃ (x y : ℝ), x * 15 = x + 196 ∧ y * 50 = y + 842 ∧ x + y = 31.2 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_l1132_113283


namespace NUMINAMATH_CALUDE_quality_difference_significant_frequency_machine_A_frequency_machine_B_l1132_113280

-- Define the contingency table
def machine_A_first_class : ℕ := 150
def machine_A_second_class : ℕ := 50
def machine_B_first_class : ℕ := 120
def machine_B_second_class : ℕ := 80
def total_products : ℕ := 400

-- Define the K^2 formula
def K_squared (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value_99_percent : ℚ := 6635 / 1000

-- Theorem statement
theorem quality_difference_significant :
  K_squared machine_A_first_class machine_A_second_class
            machine_B_first_class machine_B_second_class
            total_products > critical_value_99_percent := by
  sorry

-- Frequencies of first-class products
theorem frequency_machine_A : (machine_A_first_class : ℚ) / (machine_A_first_class + machine_A_second_class) = 3 / 4 := by
  sorry

theorem frequency_machine_B : (machine_B_first_class : ℚ) / (machine_B_first_class + machine_B_second_class) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_quality_difference_significant_frequency_machine_A_frequency_machine_B_l1132_113280


namespace NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l1132_113215

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3 - x) / (x - 4) + 1 / (4 - x) = 1

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  2 * (x + 1) > x ∧ 1 - 2 * x ≥ (x + 7) / 2

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ x = 3 :=
sorry

theorem inequality_system_solution :
  ∃ x : ℝ, inequality_system x ↔ -2 < x ∧ x ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l1132_113215


namespace NUMINAMATH_CALUDE_gcd_power_difference_l1132_113259

theorem gcd_power_difference (a b n : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (hn : n ≥ 2) (hgcd : Nat.gcd a b = 1) :
  Nat.gcd ((a^n - b^n) / (a - b)) (a - b) = Nat.gcd (a - b) n := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_difference_l1132_113259


namespace NUMINAMATH_CALUDE_equation_solution_existence_l1132_113282

theorem equation_solution_existence (z : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ 1 ≤ |z| ∧ |z| ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l1132_113282


namespace NUMINAMATH_CALUDE_special_deck_probability_l1132_113211

/-- A deck of cards with specified properties -/
structure Deck :=
  (total : ℕ)
  (red_non_joker : ℕ)
  (black_or_joker : ℕ)

/-- The probability of drawing a red non-joker card first and a black or joker card second -/
def draw_probability (d : Deck) : ℚ :=
  (d.red_non_joker : ℚ) * (d.black_or_joker : ℚ) / ((d.total : ℚ) * (d.total - 1 : ℚ))

/-- Theorem stating the probability for the specific deck described in the problem -/
theorem special_deck_probability :
  let d := Deck.mk 60 26 40
  draw_probability d = 5 / 17 := by sorry

end NUMINAMATH_CALUDE_special_deck_probability_l1132_113211


namespace NUMINAMATH_CALUDE_circle_has_most_symmetry_lines_l1132_113240

/-- Represents the number of lines of symmetry for a geometrical figure. -/
inductive SymmetryCount
  | Finite (n : ℕ)
  | Infinite

/-- Represents the geometrical figures mentioned in the problem. -/
inductive GeometricalFigure
  | Circle
  | Semicircle
  | EquilateralTriangle
  | RegularPentagon
  | Ellipse

/-- Returns the number of lines of symmetry for a given geometrical figure. -/
def symmetryLines (figure : GeometricalFigure) : SymmetryCount :=
  match figure with
  | GeometricalFigure.Circle => SymmetryCount.Infinite
  | GeometricalFigure.Semicircle => SymmetryCount.Finite 1
  | GeometricalFigure.EquilateralTriangle => SymmetryCount.Finite 3
  | GeometricalFigure.RegularPentagon => SymmetryCount.Finite 5
  | GeometricalFigure.Ellipse => SymmetryCount.Finite 2

/-- Compares two SymmetryCount values. -/
def symmetryCountLe (a b : SymmetryCount) : Prop :=
  match a, b with
  | SymmetryCount.Finite n, SymmetryCount.Finite m => n ≤ m
  | _, SymmetryCount.Infinite => True
  | SymmetryCount.Infinite, SymmetryCount.Finite _ => False

/-- States that the circle has the greatest number of lines of symmetry among the given figures. -/
theorem circle_has_most_symmetry_lines :
    ∀ (figure : GeometricalFigure),
      symmetryCountLe (symmetryLines figure) (symmetryLines GeometricalFigure.Circle) :=
  sorry

end NUMINAMATH_CALUDE_circle_has_most_symmetry_lines_l1132_113240


namespace NUMINAMATH_CALUDE_complex_modulus_l1132_113234

theorem complex_modulus (z : ℂ) (h : (3 + 2*I) * z = 5 - I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1132_113234


namespace NUMINAMATH_CALUDE_xiaotong_grade_l1132_113232

/-- Represents the grading system for a physical education course -/
structure GradingSystem where
  maxScore : ℝ
  classroomWeight : ℝ
  midtermWeight : ℝ
  finalWeight : ℝ

/-- Represents a student's scores in the physical education course -/
structure StudentScores where
  classroom : ℝ
  midterm : ℝ
  final : ℝ

/-- Calculates the final grade based on the grading system and student scores -/
def calculateGrade (sys : GradingSystem) (scores : StudentScores) : ℝ :=
  sys.classroomWeight * scores.classroom +
  sys.midtermWeight * scores.midterm +
  sys.finalWeight * scores.final

/-- The theorem stating that Xiaotong's grade is 55 given the specified grading system and scores -/
theorem xiaotong_grade :
  let sys : GradingSystem := {
    maxScore := 60,
    classroomWeight := 0.2,
    midtermWeight := 0.3,
    finalWeight := 0.5
  }
  let scores : StudentScores := {
    classroom := 60,
    midterm := 50,
    final := 56
  }
  calculateGrade sys scores = 55 := by
  sorry

end NUMINAMATH_CALUDE_xiaotong_grade_l1132_113232


namespace NUMINAMATH_CALUDE_perpendicular_line_through_vertex_l1132_113260

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 4

/-- The given line equation -/
def given_line (x y : ℝ) : Prop := x/4 + y/3 = 1

/-- The perpendicular line equation to be proved -/
def perp_line (x y : ℝ) : Prop := y = (4/3)*x - 8/3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

theorem perpendicular_line_through_vertex :
  ∃ (m b : ℝ), 
    (∀ x y, perp_line x y ↔ y = m*x + b) ∧ 
    perp_line vertex.1 vertex.2 ∧
    (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ → 
      (y₂ - y₁)/(x₂ - x₁) * m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_vertex_l1132_113260


namespace NUMINAMATH_CALUDE_row_swap_matrix_exists_l1132_113245

open Matrix

theorem row_swap_matrix_exists : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
  N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] := by
  sorry

end NUMINAMATH_CALUDE_row_swap_matrix_exists_l1132_113245


namespace NUMINAMATH_CALUDE_prism_15_edges_l1132_113248

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular lateral faces. -/
structure Prism where
  edges : ℕ
  faces : ℕ
  vertices : ℕ

/-- Theorem: A prism with 15 edges has 7 faces and 10 vertices. -/
theorem prism_15_edges (p : Prism) (h : p.edges = 15) : p.faces = 7 ∧ p.vertices = 10 := by
  sorry

end NUMINAMATH_CALUDE_prism_15_edges_l1132_113248


namespace NUMINAMATH_CALUDE_vector_magnitude_l1132_113285

/-- Given two vectors a and b in ℝ² with an angle of π/3 between them,
    where a = (1, √3) and |a - 2b| = 2√3, prove that |b| = 2 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 1 ∧ a.2 = Real.sqrt 3) →  -- a = (1, √3)
  (a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 * Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) →  -- angle between a and b is π/3
  ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 12) →  -- |a - 2b| = 2√3
  Real.sqrt (b.1^2 + b.2^2) = 2  -- |b| = 2
:= by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1132_113285


namespace NUMINAMATH_CALUDE_sum_difference_zero_l1132_113270

theorem sum_difference_zero (x y z : ℝ) 
  (h : (2*x^2 + 8*x + 11)*(y^2 - 10*y + 29)*(3*z^2 - 18*z + 32) ≤ 60) : 
  x + y - z = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_zero_l1132_113270


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l1132_113247

/-- The minimum distance between a point on the line x - y - 4 = 0 and a point on the parabola x² = 4y is (3√2)/2 -/
theorem min_distance_line_parabola :
  let line := {p : ℝ × ℝ | p.1 - p.2 = 4}
  let parabola := {p : ℝ × ℝ | p.1^2 = 4 * p.2}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ parabola ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ parabola →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l1132_113247


namespace NUMINAMATH_CALUDE_animal_shelter_problem_l1132_113236

theorem animal_shelter_problem (initial_cats initial_lizards : ℕ)
  (dog_adoption_rate cat_adoption_rate lizard_adoption_rate : ℚ)
  (new_pets total_pets_after_month : ℕ) :
  initial_cats = 28 →
  initial_lizards = 20 →
  dog_adoption_rate = 1/2 →
  cat_adoption_rate = 1/4 →
  lizard_adoption_rate = 1/5 →
  new_pets = 13 →
  total_pets_after_month = 65 →
  ∃ (initial_dogs : ℕ),
    initial_dogs = 30 ∧
    (1 - dog_adoption_rate) * initial_dogs +
    (1 - cat_adoption_rate) * initial_cats +
    (1 - lizard_adoption_rate) * initial_lizards +
    new_pets = total_pets_after_month :=
by sorry

end NUMINAMATH_CALUDE_animal_shelter_problem_l1132_113236


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1132_113242

/-- A geometric sequence is a sequence where the ratio between consecutive terms is constant. -/
def IsGeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- Given a geometric sequence b where b₂ * b₃ * b₄ = 8, prove that b₃ = 2 -/
theorem geometric_sequence_product (b : ℕ → ℝ) (h_geo : IsGeometricSequence b) 
    (h_prod : b 2 * b 3 * b 4 = 8) : b 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1132_113242


namespace NUMINAMATH_CALUDE_probability_allison_wins_l1132_113293

def allison_cube : Fin 6 → ℕ := λ _ => 6

def brian_cube : Fin 6 → ℕ := λ i => i.val + 1

def noah_cube : Fin 6 → ℕ
| 0 => 3
| 1 => 3
| _ => 5

def prob_brian_less_or_equal_5 : ℚ := 5 / 6

def prob_noah_less_or_equal_5 : ℚ := 1

theorem probability_allison_wins : ℚ := by
  sorry

end NUMINAMATH_CALUDE_probability_allison_wins_l1132_113293


namespace NUMINAMATH_CALUDE_tinas_savings_l1132_113209

/-- Tina's savings problem -/
theorem tinas_savings (x : ℕ) : 
  (x + 14 + 21) - (5 + 17) = 40 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_tinas_savings_l1132_113209


namespace NUMINAMATH_CALUDE_robins_gum_problem_l1132_113291

theorem robins_gum_problem (initial_gum : ℝ) (total_gum : ℕ) (h1 : initial_gum = 18.0) (h2 : total_gum = 62) :
  (total_gum : ℝ) - initial_gum = 44 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_problem_l1132_113291


namespace NUMINAMATH_CALUDE_gear_alignment_theorem_l1132_113216

/-- Represents a gear with a certain number of teeth and ground-off pairs -/
structure Gear where
  initial_teeth : Nat
  ground_off_pairs : Nat

/-- Calculates the number of remaining teeth on a gear -/
def remaining_teeth (g : Gear) : Nat :=
  g.initial_teeth - g.ground_off_pairs

/-- Calculates the number of possible alignment positions -/
def alignment_positions (g : Gear) : Nat :=
  g.initial_teeth - g.ground_off_pairs + 1

/-- Theorem stating that there exists exactly one position where a hole in one gear
    aligns with a whole tooth on the other gear -/
theorem gear_alignment_theorem (g1 g2 : Gear)
  (h1 : g1.initial_teeth = 32)
  (h2 : g2.initial_teeth = 32)
  (h3 : g1.ground_off_pairs = 6)
  (h4 : g2.ground_off_pairs = 6)
  : ∃! position, position ≤ alignment_positions g1 ∧
    (position ≠ 0 → 
      (∃ hole_in_g1 whole_tooth_in_g2, 
        hole_in_g1 ≤ g1.ground_off_pairs ∧
        whole_tooth_in_g2 ≤ remaining_teeth g2 ∧
        hole_in_g1 ≠ whole_tooth_in_g2)) :=
  sorry

end NUMINAMATH_CALUDE_gear_alignment_theorem_l1132_113216


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_polynomial_l1132_113269

theorem sum_of_roots_cubic_polynomial : 
  let p (x : ℝ) := 3 * x^3 - 9 * x^2 - 72 * x - 18
  ∃ (r s t : ℝ), p r = 0 ∧ p s = 0 ∧ p t = 0 ∧ r + s + t = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_polynomial_l1132_113269


namespace NUMINAMATH_CALUDE_nine_fourth_equals_three_two_m_l1132_113276

theorem nine_fourth_equals_three_two_m (m : ℕ) : 9^4 = 3^(2*m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_nine_fourth_equals_three_two_m_l1132_113276


namespace NUMINAMATH_CALUDE_bread_price_is_four_l1132_113255

/-- The price of a loaf of bread -/
def bread_price : ℝ := 4

/-- The price of a pastry -/
def pastry_price : ℝ := 2

/-- The usual number of pastries sold daily -/
def usual_pastries : ℕ := 20

/-- The usual number of loaves of bread sold daily -/
def usual_bread : ℕ := 10

/-- The number of pastries sold today -/
def today_pastries : ℕ := 14

/-- The number of loaves of bread sold today -/
def today_bread : ℕ := 25

/-- The difference between today's sales and the usual daily average -/
def sales_difference : ℝ := 48

theorem bread_price_is_four :
  (today_pastries : ℝ) * pastry_price + today_bread * bread_price -
  (usual_pastries : ℝ) * pastry_price - usual_bread * bread_price = sales_difference ∧
  bread_price = 4 := by sorry

end NUMINAMATH_CALUDE_bread_price_is_four_l1132_113255


namespace NUMINAMATH_CALUDE_stating_max_pairs_remaining_l1132_113228

/-- Represents the number of shoe types -/
def num_types : ℕ := 5

/-- Represents the number of shoe colors -/
def num_colors : ℕ := 5

/-- Represents the initial number of shoe pairs -/
def initial_pairs : ℕ := 25

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- 
Theorem stating that given the initial conditions, the maximum number of 
complete pairs remaining after losing shoes is 22
-/
theorem max_pairs_remaining : 
  ∀ (remaining_pairs : ℕ),
  remaining_pairs ≤ initial_pairs ∧
  remaining_pairs ≥ initial_pairs - shoes_lost / 2 →
  remaining_pairs ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_pairs_remaining_l1132_113228


namespace NUMINAMATH_CALUDE_fifth_number_in_specific_pascal_row_l1132_113204

/-- Represents a row in Pascal's triangle -/
def PascalRow (n : ℕ) := Fin (n + 1) → ℕ

/-- The nth row of Pascal's triangle -/
def nthPascalRow (n : ℕ) : PascalRow n := 
  fun k => Nat.choose n k.val

/-- The condition that a row starts with 1 and then 15 -/
def startsWithOneAndFifteen (row : PascalRow 15) : Prop :=
  row 0 = 1 ∧ row 1 = 15

theorem fifth_number_in_specific_pascal_row : 
  ∀ (row : PascalRow 15), 
    startsWithOneAndFifteen row → 
    row 4 = Nat.choose 15 4 ∧ 
    Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_specific_pascal_row_l1132_113204


namespace NUMINAMATH_CALUDE_perimeter_T_shape_specific_l1132_113264

/-- Calculates the perimeter of a T shape formed by two rectangles with given dimensions and overlap. -/
def perimeter_T_shape (rect1_length rect1_width rect2_length rect2_width overlap : ℝ) : ℝ :=
  2 * (rect1_length + rect1_width) + 2 * (rect2_length + rect2_width) - 2 * overlap

/-- The perimeter of a T shape formed by two rectangles (3 inch × 5 inch and 2 inch × 6 inch) with a 1-inch overlap is 30 inches. -/
theorem perimeter_T_shape_specific : perimeter_T_shape 3 5 2 6 1 = 30 := by
  sorry

#eval perimeter_T_shape 3 5 2 6 1

end NUMINAMATH_CALUDE_perimeter_T_shape_specific_l1132_113264


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1132_113287

theorem sum_of_fractions : 
  (1 : ℚ) / 2 + 1 / 6 + 1 / 12 + 1 / 20 + 1 / 30 + 1 / 42 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1132_113287


namespace NUMINAMATH_CALUDE_current_age_problem_l1132_113278

theorem current_age_problem (my_age brother_age : ℕ) : 
  (my_age + 10 = 2 * (brother_age + 10)) →
  ((my_age + 10) + (brother_age + 10) = 45) →
  my_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_current_age_problem_l1132_113278


namespace NUMINAMATH_CALUDE_exists_min_value_l1132_113233

/-- The function we want to minimize -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

/-- Theorem stating that there exists a minimum value for the function -/
theorem exists_min_value :
  ∃ (y : ℝ), ∃ (min_val : ℝ), ∀ (x : ℝ), f x y ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_exists_min_value_l1132_113233


namespace NUMINAMATH_CALUDE_subtract_decimals_l1132_113222

theorem subtract_decimals : (145.23 : ℝ) - 0.07 = 145.16 := by
  sorry

end NUMINAMATH_CALUDE_subtract_decimals_l1132_113222


namespace NUMINAMATH_CALUDE_set_M_characterization_l1132_113258

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 ∨ y = 1}

-- Define the set of valid x values
def valid_x : Set ℝ := {x | x ≠ 1 ∧ x ≠ -1}

-- Theorem statement
theorem set_M_characterization : 
  ∀ x : ℝ, (x^2 ∈ M ∧ 1 ∈ M ∧ x^2 ≠ 1) ↔ x ∈ valid_x :=
sorry

end NUMINAMATH_CALUDE_set_M_characterization_l1132_113258


namespace NUMINAMATH_CALUDE_vector_magnitude_range_function_f_range_l1132_113214

noncomputable section

def x : ℝ := sorry

-- Define vector a
def a : ℝ × ℝ := (Real.sin x + Real.cos x, Real.sqrt 2 * Real.cos x)

-- Define vector b
def b : ℝ × ℝ := (Real.cos x - Real.sin x, Real.sqrt 2 * Real.sin x)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the function f(x)
def f : ℝ → ℝ := λ x => dot_product a b - magnitude a

theorem vector_magnitude_range :
  x ∈ Set.Icc (-Real.pi / 8) 0 →
  magnitude a ∈ Set.Icc (Real.sqrt 2) (Real.sqrt 3) :=
sorry

theorem function_f_range :
  x ∈ Set.Icc (-Real.pi / 8) 0 →
  f x ∈ Set.Icc (-Real.sqrt 2) (1 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_function_f_range_l1132_113214
