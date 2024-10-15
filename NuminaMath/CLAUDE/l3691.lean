import Mathlib

namespace NUMINAMATH_CALUDE_factor_expression_l3691_369167

theorem factor_expression (b : ℝ) : 43 * b^2 + 129 * b = 43 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3691_369167


namespace NUMINAMATH_CALUDE_max_type_c_test_tubes_l3691_369141

/-- Represents the number of test tubes of each type -/
structure TestTubes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the solution percentages are valid -/
def validSolution (t : TestTubes) : Prop :=
  10 * t.a + 20 * t.b + 90 * t.c = 2017 * (t.a + t.b + t.c)

/-- Checks if the total number of test tubes is 1000 -/
def totalIs1000 (t : TestTubes) : Prop :=
  t.a + t.b + t.c = 1000

/-- Checks if test tubes of the same type are not used consecutively -/
def noConsecutiveSameType (t : TestTubes) : Prop :=
  7 * t.c ≤ 517 ∧ 8 * t.c ≥ 518 ∧ t.c ≤ 500

/-- Theorem: The maximum number of type C test tubes is 73 -/
theorem max_type_c_test_tubes :
  ∃ (t : TestTubes),
    validSolution t ∧
    totalIs1000 t ∧
    noConsecutiveSameType t ∧
    (∀ (t' : TestTubes),
      validSolution t' ∧ totalIs1000 t' ∧ noConsecutiveSameType t' →
      t'.c ≤ t.c) ∧
    t.c = 73 :=
  sorry

end NUMINAMATH_CALUDE_max_type_c_test_tubes_l3691_369141


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l3691_369103

/-- Rachel's homework problem -/
theorem rachel_homework_difference :
  ∀ (math_pages reading_pages biology_pages : ℕ),
    math_pages = 9 →
    reading_pages = 2 →
    biology_pages = 96 →
    math_pages - reading_pages = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l3691_369103


namespace NUMINAMATH_CALUDE_middle_school_students_l3691_369152

theorem middle_school_students (band_percentage : ℚ) (band_students : ℕ) 
  (h1 : band_percentage = 1/5) 
  (h2 : band_students = 168) : 
  ∃ total_students : ℕ, 
    (band_percentage * total_students = band_students) ∧ 
    total_students = 840 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_students_l3691_369152


namespace NUMINAMATH_CALUDE_polynomial_existence_l3691_369181

theorem polynomial_existence : 
  ∃ (p : ℝ → ℝ), 
    (∃ (a b c : ℝ), ∀ x, p x = a * x^2 + b * x + c) ∧ 
    p 0 = 100 ∧ 
    p 1 = 90 ∧ 
    p 2 = 70 ∧ 
    p 3 = 40 ∧ 
    p 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_existence_l3691_369181


namespace NUMINAMATH_CALUDE_complex_geometry_problem_l3691_369175

/-- A complex number z with specific properties -/
def z : ℂ :=
  sorry

/-- The condition that |z| = √2 -/
axiom z_norm : Complex.abs z = Real.sqrt 2

/-- The condition that the imaginary part of z² is 2 -/
axiom z_sq_im : Complex.im (z ^ 2) = 2

/-- The condition that z is in the first quadrant -/
axiom z_first_quadrant : Complex.re z > 0 ∧ Complex.im z > 0

/-- Point A corresponds to z -/
def A : ℂ := z

/-- Point B corresponds to z² -/
def B : ℂ := z ^ 2

/-- Point C corresponds to z - z² -/
def C : ℂ := z - z ^ 2

/-- The main theorem to be proved -/
theorem complex_geometry_problem :
  z = 1 + Complex.I ∧
  Real.cos (Complex.arg (B - A) - Complex.arg (C - B)) = -2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_complex_geometry_problem_l3691_369175


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_20_l3691_369191

/-- A regular polygon with exterior angles measuring 20 degrees has 18 sides. -/
theorem regular_polygon_exterior_angle_20 : 
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 20 → 
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_20_l3691_369191


namespace NUMINAMATH_CALUDE_new_drive_size_l3691_369145

/-- Calculates the size of a new external drive based on initial drive conditions and file operations -/
theorem new_drive_size
  (initial_free : ℝ)
  (initial_used : ℝ)
  (deleted_size : ℝ)
  (new_files_size : ℝ)
  (new_free_space : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : deleted_size = 4.6)
  (h4 : new_files_size = 2)
  (h5 : new_free_space = 10) :
  initial_used - deleted_size + new_files_size + new_free_space = 20 := by
  sorry

#check new_drive_size

end NUMINAMATH_CALUDE_new_drive_size_l3691_369145


namespace NUMINAMATH_CALUDE_max_a_for_zero_points_l3691_369194

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / x^2 - x - a/x + 2*Real.exp 1

theorem max_a_for_zero_points :
  (∃ a : ℝ, ∃ x : ℝ, x > 0 ∧ f a x = 0) →
  (∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1) ∧
  (∃ x : ℝ, x > 0 ∧ f (Real.exp 2 + 1 / Real.exp 1) x = 0) := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_zero_points_l3691_369194


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3691_369166

theorem binomial_coefficient_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ : ℝ) :
  (∀ x : ℝ, (1 + x)^14 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + 
    a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ + 8*a₈ + 9*a₉ + 10*a₁₀ + 
    11*a₁₁ + 12*a₁₂ + 13*a₁₃ + 14*a₁₄ = 7 * 2^14 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3691_369166


namespace NUMINAMATH_CALUDE_probability_odd_product_sum_div_5_l3691_369135

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧
  is_odd a ∧ is_odd b ∧ is_divisible_by_5 (a + b)

def total_pairs : ℕ := 190

def valid_pairs : ℕ := 6

theorem probability_odd_product_sum_div_5 :
  (valid_pairs : ℚ) / total_pairs = 3 / 95 := by sorry

end NUMINAMATH_CALUDE_probability_odd_product_sum_div_5_l3691_369135


namespace NUMINAMATH_CALUDE_specific_det_value_det_equation_solution_l3691_369198

-- Define the determinant of order 2
def det2 (a b c d : ℤ) : ℤ := a * d - b * c

-- Theorem 1: The value of the specific determinant is 1
theorem specific_det_value : det2 2022 2023 2021 2022 = 1 := by sorry

-- Theorem 2: If the given determinant equals 32, then m = 4
theorem det_equation_solution (m : ℤ) : 
  det2 (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 := by sorry

end NUMINAMATH_CALUDE_specific_det_value_det_equation_solution_l3691_369198


namespace NUMINAMATH_CALUDE_bus_rows_theorem_l3691_369148

/-- Represents the state of passengers on a bus -/
structure BusState where
  initial : Nat
  first_stop_board : Nat
  first_stop_leave : Nat
  second_stop_board : Nat
  second_stop_leave : Nat
  empty_seats : Nat
  seats_per_row : Nat

/-- Calculates the number of rows on the bus given its state -/
def calculate_rows (state : BusState) : Nat :=
  let total_passengers := state.initial + 
    (state.first_stop_board - state.first_stop_leave) + 
    (state.second_stop_board - state.second_stop_leave)
  let total_seats := total_passengers + state.empty_seats
  total_seats / state.seats_per_row

/-- Theorem stating that given the problem conditions, the bus has 23 rows -/
theorem bus_rows_theorem (state : BusState) 
  (h1 : state.initial = 16)
  (h2 : state.first_stop_board = 15)
  (h3 : state.first_stop_leave = 3)
  (h4 : state.second_stop_board = 17)
  (h5 : state.second_stop_leave = 10)
  (h6 : state.empty_seats = 57)
  (h7 : state.seats_per_row = 4) :
  calculate_rows state = 23 := by
  sorry

#eval calculate_rows {
  initial := 16,
  first_stop_board := 15,
  first_stop_leave := 3,
  second_stop_board := 17,
  second_stop_leave := 10,
  empty_seats := 57,
  seats_per_row := 4
}

end NUMINAMATH_CALUDE_bus_rows_theorem_l3691_369148


namespace NUMINAMATH_CALUDE_all_statements_imply_negation_l3691_369115

theorem all_statements_imply_negation (p q r : Prop) :
  (p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (¬p ∧ q ∧ r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (¬p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ ¬r) :=
by sorry

#check all_statements_imply_negation

end NUMINAMATH_CALUDE_all_statements_imply_negation_l3691_369115


namespace NUMINAMATH_CALUDE_dog_food_cost_l3691_369165

def initial_amount : ℕ := 167
def meat_cost : ℕ := 17
def chicken_cost : ℕ := 22
def veggie_cost : ℕ := 43
def egg_cost : ℕ := 5
def remaining_amount : ℕ := 35

theorem dog_food_cost : 
  initial_amount - (meat_cost + chicken_cost + veggie_cost + egg_cost + remaining_amount) = 45 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_cost_l3691_369165


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3691_369111

theorem sum_of_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3691_369111


namespace NUMINAMATH_CALUDE_medical_team_selection_l3691_369153

theorem medical_team_selection (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 6) (h2 : female_doctors = 5) :
  (Nat.choose male_doctors 2) * (Nat.choose female_doctors 1) = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l3691_369153


namespace NUMINAMATH_CALUDE_square_area_16m_l3691_369133

/-- The area of a square with side length 16 meters is 256 square meters. -/
theorem square_area_16m (side_length : ℝ) (h : side_length = 16) : 
  side_length * side_length = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_area_16m_l3691_369133


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_29_l3691_369142

theorem sum_of_coefficients_equals_negative_29 :
  let p (x : ℝ) := 5 * (2 * x^8 - 9 * x^3 + 6) - 4 * (x^6 + 8 * x^3 - 3)
  (p 1) = -29 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_29_l3691_369142


namespace NUMINAMATH_CALUDE_square_side_length_l3691_369190

-- Define the rectangle's dimensions
def rectangle_length : ℝ := 7
def rectangle_width : ℝ := 5

-- Define the theorem
theorem square_side_length : 
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side := rectangle_perimeter / 4
  square_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3691_369190


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l3691_369171

theorem arithmetic_geometric_mean_ratio : ∃ (c d : ℝ), 
  c > d ∧ d > 0 ∧ 
  (c + d) / 2 = 3 * Real.sqrt (c * d) ∧
  |(c / d) - 34| < 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l3691_369171


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l3691_369184

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday
  weekly_earnings : ℕ    -- Weekly earnings in dollars

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly rate --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 8
  , tue_thu_hours := 6
  , weekly_earnings := 288 }

/-- Theorem stating that Sheila's hourly rate is $8 --/
theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l3691_369184


namespace NUMINAMATH_CALUDE_range_of_function_range_tight_l3691_369102

theorem range_of_function (x : ℝ) :
  ∃ (y : ℝ), y = |2 * Real.sin x + 3 * Real.cos x + 4| ∧
  4 - Real.sqrt 13 ≤ y ∧ y ≤ 4 + Real.sqrt 13 :=
by sorry

theorem range_tight :
  ∃ (x₁ x₂ : ℝ), 
    |2 * Real.sin x₁ + 3 * Real.cos x₁ + 4| = 4 - Real.sqrt 13 ∧
    |2 * Real.sin x₂ + 3 * Real.cos x₂ + 4| = 4 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_range_tight_l3691_369102


namespace NUMINAMATH_CALUDE_ivy_coverage_l3691_369138

/-- The amount of ivy Cary strips each day, in feet -/
def daily_strip : ℕ := 6

/-- The amount of ivy that grows back each night, in feet -/
def nightly_growth : ℕ := 2

/-- The number of days it takes Cary to strip all the ivy -/
def days_to_strip : ℕ := 10

/-- The net amount of ivy stripped per day, in feet -/
def net_strip_per_day : ℕ := daily_strip - nightly_growth

/-- The total amount of ivy covering the tree, in feet -/
def total_ivy : ℕ := net_strip_per_day * days_to_strip

theorem ivy_coverage : total_ivy = 40 := by
  sorry

end NUMINAMATH_CALUDE_ivy_coverage_l3691_369138


namespace NUMINAMATH_CALUDE_door_opening_probability_l3691_369169

/-- The probability of opening a door on the second try given specific conditions -/
theorem door_opening_probability (total_keys : ℕ) (working_keys : ℕ) : 
  total_keys = 4 → working_keys = 2 → 
  (working_keys : ℚ) / total_keys * working_keys / (total_keys - 1) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_door_opening_probability_l3691_369169


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3691_369137

/-- If the roots of the quadratic equation 5x^2 + 4x + k are (-4 ± i√379) / 10, then k = 19.75 -/
theorem quadratic_root_problem (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 4 * x + k = 0 ↔ x = (-4 + ℍ * Real.sqrt 379) / 10 ∨ x = (-4 - ℍ * Real.sqrt 379) / 10) →
  k = 19.75 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3691_369137


namespace NUMINAMATH_CALUDE_inequality_proof_l3691_369146

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : x > 1) (hn : n > 1) :
  1 + (x - 1) / (n * x) < x^(1/n) ∧ x^(1/n) < 1 + (x - 1) / n :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3691_369146


namespace NUMINAMATH_CALUDE_square_perimeter_relationship_l3691_369144

/-- Given two squares C and D, where C has a perimeter of 32 cm and D has an area
    equal to one-third the area of C, the perimeter of D is (32√3)/3 cm. -/
theorem square_perimeter_relationship (C D : Real → Real → Prop) :
  (∃ (side_c : Real), C side_c side_c ∧ 4 * side_c = 32) →
  (∃ (side_d : Real), D side_d side_d ∧ side_d^2 = (side_c^2) / 3) →
  (∃ (perimeter_d : Real), perimeter_d = 32 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relationship_l3691_369144


namespace NUMINAMATH_CALUDE_unique_solution_system_l3691_369108

theorem unique_solution_system : ∃! (x y : ℕ+), 
  (x.val : ℝ) ^ (y.val : ℝ) + 1 = (y.val : ℝ) ^ (x.val : ℝ) ∧ 
  2 * (x.val : ℝ) ^ (y.val : ℝ) = (y.val : ℝ) ^ (x.val : ℝ) + 7 ∧
  x.val = 2 ∧ y.val = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3691_369108


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3691_369182

theorem triangle_angle_problem (A B C : Real) (BC AC : Real) :
  BC = Real.sqrt 3 →
  AC = Real.sqrt 2 →
  A = π / 3 →
  B = π / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3691_369182


namespace NUMINAMATH_CALUDE_solve_equation_l3691_369174

theorem solve_equation (x n : ℝ) (h1 : x / 4 - (x - 3) / n = 1) (h2 : x = 6) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3691_369174


namespace NUMINAMATH_CALUDE_brandon_gecko_sales_l3691_369188

/-- The number of geckos Brandon sold in the first half of last year -/
def first_half_last_year : ℕ := 46

/-- The number of geckos Brandon sold in the second half of last year -/
def second_half_last_year : ℕ := 55

/-- The number of geckos Brandon sold in the first half two years ago -/
def first_half_two_years_ago : ℕ := 3 * first_half_last_year

/-- The number of geckos Brandon sold in the second half two years ago -/
def second_half_two_years_ago : ℕ := 117

/-- The total number of geckos Brandon sold in the last two years -/
def total_geckos : ℕ := first_half_last_year + second_half_last_year + first_half_two_years_ago + second_half_two_years_ago

theorem brandon_gecko_sales : total_geckos = 356 := by
  sorry

end NUMINAMATH_CALUDE_brandon_gecko_sales_l3691_369188


namespace NUMINAMATH_CALUDE_machine_input_l3691_369104

/-- A machine that processes numbers -/
def Machine (x : ℤ) : ℤ := x + 15 - 6

/-- Theorem: If the machine outputs 35, the input must have been 26 -/
theorem machine_input (x : ℤ) : Machine x = 35 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_machine_input_l3691_369104


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3691_369157

/-- Given two similar triangles with an area ratio of 1:9 and the smaller triangle
    having a height of 5 cm, prove that the corresponding height of the larger triangle
    is 15 cm. -/
theorem similar_triangles_height (small_height large_height : ℝ) :
  small_height = 5 →
  (9 : ℝ) * small_height^2 = large_height^2 →
  large_height = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3691_369157


namespace NUMINAMATH_CALUDE_men_in_second_scenario_l3691_369173

/-- Calculates the number of men working in the second scenario given the conditions --/
theorem men_in_second_scenario 
  (hours_per_day_first : ℕ) 
  (hours_per_day_second : ℕ)
  (men_first : ℕ)
  (earnings_first : ℚ)
  (earnings_second : ℚ)
  (days_per_week : ℕ) :
  hours_per_day_first = 10 →
  hours_per_day_second = 6 →
  men_first = 4 →
  earnings_first = 1400 →
  earnings_second = 1890.0000000000002 →
  days_per_week = 7 →
  ∃ (men_second : ℕ), men_second = 9 ∧ 
    (men_second * hours_per_day_second * days_per_week : ℚ) * 
    (earnings_first / (men_first * hours_per_day_first * days_per_week : ℚ)) = 
    earnings_second :=
by sorry

end NUMINAMATH_CALUDE_men_in_second_scenario_l3691_369173


namespace NUMINAMATH_CALUDE_rectangleWithHoleAreaTheorem_l3691_369120

/-- The area of a rectangle with a hole, given the dimensions of both rectangles -/
def rectangleWithHoleArea (x : ℝ) : ℝ :=
  let largeRectLength := x + 7
  let largeRectWidth := x + 5
  let holeLength := 2*x - 3
  let holeWidth := x - 2
  (largeRectLength * largeRectWidth) - (holeLength * holeWidth)

/-- Theorem stating that the area of the rectangle with a hole is equal to -x^2 + 19x + 29 -/
theorem rectangleWithHoleAreaTheorem (x : ℝ) :
  rectangleWithHoleArea x = -x^2 + 19*x + 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangleWithHoleAreaTheorem_l3691_369120


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3691_369149

/-- Given a rectangle with perimeter 80 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is sqrt(46400)/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 80) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = Real.sqrt 46400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3691_369149


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3691_369156

theorem geometric_sequence_sum (n : ℕ) : 
  (1/3 : ℝ) * (1 - (1/3)^n) / (1 - 1/3) = 728/729 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3691_369156


namespace NUMINAMATH_CALUDE_num_tuba_players_l3691_369107

/-- The weight carried by each trumpet or clarinet player -/
def trumpet_clarinet_weight : ℕ := 5

/-- The weight carried by each trombone player -/
def trombone_weight : ℕ := 10

/-- The weight carried by each tuba player -/
def tuba_weight : ℕ := 20

/-- The weight carried by each drum player -/
def drum_weight : ℕ := 15

/-- The number of trumpet players -/
def num_trumpets : ℕ := 6

/-- The number of clarinet players -/
def num_clarinets : ℕ := 9

/-- The number of trombone players -/
def num_trombones : ℕ := 8

/-- The number of drum players -/
def num_drums : ℕ := 2

/-- The total weight carried by the marching band -/
def total_weight : ℕ := 245

/-- Theorem: The number of tuba players in the marching band is 3 -/
theorem num_tuba_players : 
  ∃ (n : ℕ), n * tuba_weight = 
    total_weight - 
    (num_trumpets * trumpet_clarinet_weight + 
     num_clarinets * trumpet_clarinet_weight + 
     num_trombones * trombone_weight + 
     num_drums * drum_weight) ∧ 
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_tuba_players_l3691_369107


namespace NUMINAMATH_CALUDE_current_speed_l3691_369129

/-- Given a boat that moves upstream at 1 km in 40 minutes and downstream at 1 km in 12 minutes,
    prove that the speed of the current is 1.75 km/h. -/
theorem current_speed (upstream_speed : ℝ) (downstream_speed : ℝ)
    (h1 : upstream_speed = 1 / (40 / 60))  -- 1 km in 40 minutes converted to km/h
    (h2 : downstream_speed = 1 / (12 / 60))  -- 1 km in 12 minutes converted to km/h
    : (downstream_speed - upstream_speed) / 2 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3691_369129


namespace NUMINAMATH_CALUDE_f_equals_g_l3691_369161

/-- Two functions are considered the same if they have the same domain, codomain, and function value for all inputs. -/
def same_function (α β : Type) (f g : α → β) : Prop :=
  ∀ x, f x = g x

/-- Function f defined as f(x) = x - 1 -/
def f : ℝ → ℝ := λ x ↦ x - 1

/-- Function g defined as g(t) = t - 1 -/
def g : ℝ → ℝ := λ t ↦ t - 1

/-- Theorem stating that f and g are the same function -/
theorem f_equals_g : same_function ℝ ℝ f g := by
  sorry


end NUMINAMATH_CALUDE_f_equals_g_l3691_369161


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l3691_369125

theorem kendall_driving_distance (distance_with_mother distance_with_father : Real) 
  (h1 : distance_with_mother = 0.17)
  (h2 : distance_with_father = 0.5) : 
  distance_with_mother + distance_with_father = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l3691_369125


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l3691_369122

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l3691_369122


namespace NUMINAMATH_CALUDE_roots_cubic_reciprocal_sum_l3691_369150

/-- Given a quadratic equation px^2 + qx + m = 0 with roots r and s,
    prove that 1/r^3 + 1/s^3 = (-q^3 + 3qm) / m^3 -/
theorem roots_cubic_reciprocal_sum (p q m : ℝ) (hp : p ≠ 0) (hm : m ≠ 0) :
  ∃ (r s : ℝ), (p * r^2 + q * r + m = 0) ∧ 
               (p * s^2 + q * s + m = 0) ∧ 
               (1 / r^3 + 1 / s^3 = (-q^3 + 3*q*m) / m^3) := by
  sorry

end NUMINAMATH_CALUDE_roots_cubic_reciprocal_sum_l3691_369150


namespace NUMINAMATH_CALUDE_baseball_card_pages_l3691_369172

theorem baseball_card_pages (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 8)
  (h3 : old_cards = 10) :
  (new_cards + old_cards) / cards_per_page = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l3691_369172


namespace NUMINAMATH_CALUDE_profit_starts_fourth_year_option_two_more_profitable_l3691_369151

def initial_investment : ℕ := 81
def annual_rental_income : ℕ := 30
def first_year_renovation : ℕ := 1
def yearly_renovation_increase : ℕ := 2

def total_renovation_cost (n : ℕ) : ℕ := n^2

def total_income (n : ℕ) : ℕ := annual_rental_income * n

def profit (n : ℕ) : ℤ := (total_income n : ℤ) - (initial_investment : ℤ) - (total_renovation_cost n : ℤ)

def average_profit (n : ℕ) : ℚ := (profit n : ℚ) / n

theorem profit_starts_fourth_year :
  ∀ n : ℕ, n < 4 → profit n ≤ 0 ∧ profit 4 > 0 := by sorry

theorem option_two_more_profitable :
  profit 15 + 10 < profit 9 + 50 := by sorry

#eval profit 4
#eval profit 15 + 10
#eval profit 9 + 50

end NUMINAMATH_CALUDE_profit_starts_fourth_year_option_two_more_profitable_l3691_369151


namespace NUMINAMATH_CALUDE_only_points_in_circle_form_set_l3691_369197

-- Define a type for the objects in question
inductive Object
| MaleStudents
| DifficultProblems
| OutgoingGirls
| PointsInCircle

-- Define a predicate for whether an object can form a set
def CanFormSet (obj : Object) : Prop :=
  match obj with
  | Object.PointsInCircle => True
  | _ => False

-- State the theorem
theorem only_points_in_circle_form_set :
  ∀ (obj : Object), CanFormSet obj ↔ obj = Object.PointsInCircle :=
by sorry

end NUMINAMATH_CALUDE_only_points_in_circle_form_set_l3691_369197


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l3691_369100

/-- Proves that adding a specific amount of water to a given mixture results in a new mixture with the expected water percentage. -/
theorem water_mixture_percentage 
  (initial_volume : ℝ) 
  (initial_water_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_volume = 200)
  (h2 : initial_water_percentage = 20)
  (h3 : added_water = 13.333333333333334)
  : (initial_water_percentage / 100 * initial_volume + added_water) / (initial_volume + added_water) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l3691_369100


namespace NUMINAMATH_CALUDE_percentage_both_correct_l3691_369116

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered both questions correctly. -/
theorem percentage_both_correct
  (p_first : ℝ)  -- Probability of answering the first question correctly
  (p_second : ℝ) -- Probability of answering the second question correctly
  (p_neither : ℝ) -- Probability of answering neither question correctly
  (h1 : p_first = 0.65)  -- 65% answered the first question correctly
  (h2 : p_second = 0.55) -- 55% answered the second question correctly
  (h3 : p_neither = 0.20) -- 20% answered neither question correctly
  : p_first + p_second - (1 - p_neither) = 0.40 := by
  sorry

#check percentage_both_correct

end NUMINAMATH_CALUDE_percentage_both_correct_l3691_369116


namespace NUMINAMATH_CALUDE_house_rent_percentage_l3691_369186

-- Define the percentages as real numbers
def food_percentage : ℝ := 0.50
def education_percentage : ℝ := 0.15
def remaining_percentage : ℝ := 0.175

-- Define the theorem
theorem house_rent_percentage :
  let total_income : ℝ := 100
  let remaining_after_food_education : ℝ := total_income * (1 - food_percentage - education_percentage)
  let spent_on_rent : ℝ := remaining_after_food_education - (total_income * remaining_percentage)
  (spent_on_rent / remaining_after_food_education) = 0.5 := by sorry

end NUMINAMATH_CALUDE_house_rent_percentage_l3691_369186


namespace NUMINAMATH_CALUDE_max_available_is_two_l3691_369192

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the colleagues
inductive Colleague
| Alice
| Bob
| Charlie
| Diana

-- Define a function that represents the availability of a colleague on a given day
def isAvailable (c : Colleague) (d : Day) : Bool :=
  match c, d with
  | Colleague.Alice, Day.Monday => false
  | Colleague.Alice, Day.Tuesday => true
  | Colleague.Alice, Day.Wednesday => false
  | Colleague.Alice, Day.Thursday => true
  | Colleague.Alice, Day.Friday => false
  | Colleague.Bob, Day.Monday => true
  | Colleague.Bob, Day.Tuesday => false
  | Colleague.Bob, Day.Wednesday => true
  | Colleague.Bob, Day.Thursday => false
  | Colleague.Bob, Day.Friday => true
  | Colleague.Charlie, Day.Monday => false
  | Colleague.Charlie, Day.Tuesday => false
  | Colleague.Charlie, Day.Wednesday => true
  | Colleague.Charlie, Day.Thursday => true
  | Colleague.Charlie, Day.Friday => false
  | Colleague.Diana, Day.Monday => true
  | Colleague.Diana, Day.Tuesday => true
  | Colleague.Diana, Day.Wednesday => false
  | Colleague.Diana, Day.Thursday => false
  | Colleague.Diana, Day.Friday => true

-- Define a function that counts the number of available colleagues on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun c => isAvailable c d) [Colleague.Alice, Colleague.Bob, Colleague.Charlie, Colleague.Diana]).length

-- Theorem: The maximum number of available colleagues on any day is 2
theorem max_available_is_two :
  (List.map countAvailable [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]).maximum? = some 2 := by
  sorry


end NUMINAMATH_CALUDE_max_available_is_two_l3691_369192


namespace NUMINAMATH_CALUDE_ivan_payment_l3691_369177

/-- The total amount paid for discounted Uno Giant Family Cards -/
def total_paid (original_price discount quantity : ℕ) : ℕ :=
  (original_price - discount) * quantity

/-- Theorem: Ivan paid $100 for 10 Uno Giant Family Cards with a $2 discount each -/
theorem ivan_payment :
  let original_price : ℕ := 12
  let discount : ℕ := 2
  let quantity : ℕ := 10
  total_paid original_price discount quantity = 100 := by
sorry

end NUMINAMATH_CALUDE_ivan_payment_l3691_369177


namespace NUMINAMATH_CALUDE_boys_ratio_in_class_l3691_369114

theorem boys_ratio_in_class (n_boys n_girls : ℕ) (h_prob : n_boys / (n_boys + n_girls) = 2/3 * (n_girls / (n_boys + n_girls))) :
  n_boys / (n_boys + n_girls) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_boys_ratio_in_class_l3691_369114


namespace NUMINAMATH_CALUDE_mollys_age_l3691_369154

/-- Given Sandy's age and the ratio of Sandy's age to Molly's age, calculate Molly's age -/
theorem mollys_age (sandy_age : ℕ) (ratio : ℚ) (h1 : sandy_age = 49) (h2 : ratio = 7/9) :
  sandy_age / ratio = 63 :=
sorry

end NUMINAMATH_CALUDE_mollys_age_l3691_369154


namespace NUMINAMATH_CALUDE_cab_delay_l3691_369136

theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) (delay : ℝ) : 
  usual_time = 60 →
  speed_ratio = 5/6 →
  delay = usual_time * (1 / speed_ratio - 1) →
  delay = 12 := by
sorry

end NUMINAMATH_CALUDE_cab_delay_l3691_369136


namespace NUMINAMATH_CALUDE_range_of_a_l3691_369187

theorem range_of_a (a : ℝ) : 
  (∀ t ∈ Set.Ioo 0 2, t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2) → 
  a ∈ Set.Icc (2/13) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3691_369187


namespace NUMINAMATH_CALUDE_min_value_a_l3691_369106

theorem min_value_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) :
  (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) →
  a ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3691_369106


namespace NUMINAMATH_CALUDE_probability_of_one_red_ball_l3691_369132

/-- The probability of drawing exactly one red ball from a bag containing 2 yellow balls, 3 red balls, and 5 white balls is 3/10. -/
theorem probability_of_one_red_ball (yellow_balls red_balls white_balls : ℕ) 
  (h_yellow : yellow_balls = 2)
  (h_red : red_balls = 3)
  (h_white : white_balls = 5) : 
  (red_balls : ℚ) / (yellow_balls + red_balls + white_balls) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_red_ball_l3691_369132


namespace NUMINAMATH_CALUDE_largest_absolute_value_l3691_369185

theorem largest_absolute_value : let S : Finset Int := {2, 3, -3, -4}
  ∃ x ∈ S, ∀ y ∈ S, |y| ≤ |x| ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_largest_absolute_value_l3691_369185


namespace NUMINAMATH_CALUDE_part1_part2_l3691_369183

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Part 1
theorem part1 (k : ℝ) : 
  (∀ x, f k x < 0 ↔ 2 < x ∧ x < 3) → k = 2/5 := by sorry

-- Part 2
theorem part2 (k : ℝ) :
  k > 0 ∧ (∀ x, 2 < x ∧ x < 3 → f k x < 0) → 0 < k ∧ k ≤ 2/5 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3691_369183


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l3691_369127

theorem power_mod_thirteen : 777^777 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l3691_369127


namespace NUMINAMATH_CALUDE_mozzarella_count_l3691_369176

def cheese_pack (cheddar pepperjack mozzarella : ℕ) : Prop :=
  cheddar = 15 ∧ 
  pepperjack = 45 ∧ 
  (pepperjack : ℚ) / (cheddar + pepperjack + mozzarella) = 1/2

theorem mozzarella_count : ∃ m : ℕ, cheese_pack 15 45 m ∧ m = 30 := by
  sorry

end NUMINAMATH_CALUDE_mozzarella_count_l3691_369176


namespace NUMINAMATH_CALUDE_max_min_sum_implies_a_value_l3691_369195

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

theorem max_min_sum_implies_a_value (a : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 2 3, f a x ≤ max) ∧
    (∃ y ∈ Set.Icc 2 3, f a y = max) ∧
    (∀ x ∈ Set.Icc 2 3, min ≤ f a x) ∧
    (∃ y ∈ Set.Icc 2 3, f a y = min) ∧
    max + min = 5) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_max_min_sum_implies_a_value_l3691_369195


namespace NUMINAMATH_CALUDE_decimal_to_fraction_times_three_l3691_369160

theorem decimal_to_fraction_times_three :
  (2.36 : ℚ) * 3 = 177 / 25 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_times_three_l3691_369160


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3691_369124

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/8))^(1/4))^12 * (((a^16)^(1/4))^(1/8))^12 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3691_369124


namespace NUMINAMATH_CALUDE_five_students_two_groups_l3691_369121

/-- The number of ways to assign students to groups -/
def assignment_count (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem: There are 32 ways to assign 5 students to 2 groups -/
theorem five_students_two_groups :
  assignment_count 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_five_students_two_groups_l3691_369121


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3691_369117

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3691_369117


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_chord_bounds_l3691_369131

/-- Given an ellipse (x²/a²) + (y²/b²) = 1 with a > b > 0, for any two points A and B on the ellipse
    such that OA ⊥ OB, the distance |AB| satisfies (ab / √(a² + b²)) ≤ |AB| ≤ √(a² + b²) -/
theorem ellipse_perpendicular_chord_bounds (a b : ℝ) (ha : 0 < b) (hab : b < a) :
  ∀ (A B : ℝ × ℝ),
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) →
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (a * b / Real.sqrt (a^2 + b^2) ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_chord_bounds_l3691_369131


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_integer_l3691_369180

/-- Represents three consecutive even integers -/
structure ConsecutiveEvenIntegers where
  middle : ℕ
  is_even : Even middle

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The property that the sum of the integers is one-fifth of their product -/
def sum_is_one_fifth_of_product (integers : ConsecutiveEvenIntegers) : Prop :=
  (integers.middle - 2) + integers.middle + (integers.middle + 2) = 
    ((integers.middle - 2) * integers.middle * (integers.middle + 2)) / 5

theorem smallest_consecutive_even_integer :
  ∃ (integers : ConsecutiveEvenIntegers),
    (is_two_digit (integers.middle - 2)) ∧
    (is_two_digit integers.middle) ∧
    (is_two_digit (integers.middle + 2)) ∧
    (sum_is_one_fifth_of_product integers) ∧
    (integers.middle - 2 = 86) := by
  sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_integer_l3691_369180


namespace NUMINAMATH_CALUDE_samoa_price_is_4_l3691_369179

/-- The price of a box of samoas -/
def samoa_price : ℝ := sorry

/-- The number of boxes of samoas sold -/
def samoa_boxes : ℕ := 3

/-- The price of a box of thin mints -/
def thin_mint_price : ℝ := 3.5

/-- The number of boxes of thin mints sold -/
def thin_mint_boxes : ℕ := 2

/-- The price of a box of fudge delights -/
def fudge_delight_price : ℝ := 5

/-- The number of boxes of fudge delights sold -/
def fudge_delight_boxes : ℕ := 1

/-- The price of a box of sugar cookies -/
def sugar_cookie_price : ℝ := 2

/-- The number of boxes of sugar cookies sold -/
def sugar_cookie_boxes : ℕ := 9

/-- The total sales amount -/
def total_sales : ℝ := 42

theorem samoa_price_is_4 : 
  samoa_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_samoa_price_is_4_l3691_369179


namespace NUMINAMATH_CALUDE_bacteria_population_after_15_days_l3691_369178

/-- Calculates the population of bacteria cells after a given number of days -/
def bacteriaPopulation (initialCells : ℕ) (daysPerDivision : ℕ) (totalDays : ℕ) : ℕ :=
  initialCells * (3 ^ (totalDays / daysPerDivision))

/-- Theorem stating that the bacteria population after 15 days is 1215 cells -/
theorem bacteria_population_after_15_days :
  bacteriaPopulation 5 3 15 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_population_after_15_days_l3691_369178


namespace NUMINAMATH_CALUDE_problem_solution_l3691_369119

theorem problem_solution : ∀ x y : ℝ,
  x = 88 * (1 + 0.3) →
  y = x * (1 - 0.15) →
  y = 97.24 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3691_369119


namespace NUMINAMATH_CALUDE_inequality_proof_l3691_369199

theorem inequality_proof (a b c : ℝ) (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  (a + b + c ≥ 3) ∧ (a * b * c ≤ 1) ∧
  ((a + b + c = 3 ∧ a * b * c = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3691_369199


namespace NUMINAMATH_CALUDE_janice_age_proof_l3691_369163

/-- Calculates a person's age given their birth year and the current year -/
def calculate_age (birth_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - birth_year

theorem janice_age_proof (current_year : ℕ) (mark_birth_year : ℕ) :
  current_year = 2021 →
  mark_birth_year = 1976 →
  let mark_age := calculate_age mark_birth_year current_year
  let graham_age := mark_age - 3
  let janice_age := graham_age / 2
  janice_age = 21 := by sorry

end NUMINAMATH_CALUDE_janice_age_proof_l3691_369163


namespace NUMINAMATH_CALUDE_project_work_time_difference_l3691_369170

theorem project_work_time_difference (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 90 →
  2 * t1 = 3 * t2 →
  3 * t2 = 4 * t3 →
  t3 - t1 = 20 := by
sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l3691_369170


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3691_369158

theorem smallest_integer_with_given_remainders :
  ∃ (n : ℕ), n > 0 ∧
    n % 5 = 4 ∧
    n % 7 = 5 ∧
    n % 11 = 9 ∧
    n % 13 = 11 ∧
    (∀ m : ℕ, m > 0 ∧
      m % 5 = 4 ∧
      m % 7 = 5 ∧
      m % 11 = 9 ∧
      m % 13 = 11 → m ≥ n) ∧
    n = 999 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3691_369158


namespace NUMINAMATH_CALUDE_triangle_special_cosine_identity_l3691_369164

theorem triangle_special_cosine_identity (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  Real.sin A = Real.cos B ∧ 
  Real.sin A = Real.tan C → 
  Real.cos A ^ 3 + Real.cos A ^ 2 - Real.cos A = 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_cosine_identity_l3691_369164


namespace NUMINAMATH_CALUDE_circle_equation_l3691_369110

theorem circle_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ = 0 ∧ y₀ = 0) ∨ (x₀ = 4 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = 1) → 
    x₀^2 + y₀^2 - 4*x₀ - 6*y₀ = 0) ↔
  x^2 + y^2 - 4*x - 6*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3691_369110


namespace NUMINAMATH_CALUDE_thirtieth_triangular_and_difference_l3691_369140

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_and_difference :
  (triangular_number 30 = 465) ∧
  (triangular_number 30 - triangular_number 29 = 30) := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_and_difference_l3691_369140


namespace NUMINAMATH_CALUDE_problem_statement_l3691_369134

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, 2*x - 2 ≥ m^2 - 3*m

def q (m : ℝ) : Prop := ∃ x ∈ Set.Icc (-1) 1, m ≤ x

theorem problem_statement (m : ℝ) :
  (p m ↔ m ∈ Set.Icc 1 2) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) ↔ m < 1 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3691_369134


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l3691_369128

theorem range_of_2a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 5) (hb : 5 < b ∧ b < 12) :
  -10 < 2 * a - b ∧ 2 * a - b < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l3691_369128


namespace NUMINAMATH_CALUDE_orange_price_problem_l3691_369130

/-- Proof of the orange price problem --/
theorem orange_price_problem 
  (apple_price : ℚ) 
  (total_fruits : ℕ) 
  (initial_avg_price : ℚ) 
  (oranges_removed : ℕ) 
  (final_avg_price : ℚ) 
  (h1 : apple_price = 40/100)
  (h2 : total_fruits = 10)
  (h3 : initial_avg_price = 56/100)
  (h4 : oranges_removed = 6)
  (h5 : final_avg_price = 50/100) :
  ∃ (orange_price : ℚ), orange_price = 60/100 := by
sorry


end NUMINAMATH_CALUDE_orange_price_problem_l3691_369130


namespace NUMINAMATH_CALUDE_egg_cost_l3691_369189

/-- The cost of breakfast items and breakfasts for Dale and Andrew -/
structure BreakfastCosts where
  toast : ℝ  -- Cost of a slice of toast
  egg : ℝ    -- Cost of an egg
  dale : ℝ   -- Cost of Dale's breakfast
  andrew : ℝ  -- Cost of Andrew's breakfast
  total : ℝ  -- Total cost of both breakfasts

/-- Theorem stating the cost of an egg given the breakfast costs -/
theorem egg_cost (b : BreakfastCosts) 
  (h_toast : b.toast = 1)
  (h_dale : b.dale = 2 * b.toast + 2 * b.egg)
  (h_andrew : b.andrew = b.toast + 2 * b.egg)
  (h_total : b.total = b.dale + b.andrew)
  (h_total_value : b.total = 15) :
  b.egg = 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_cost_l3691_369189


namespace NUMINAMATH_CALUDE_mabels_tomatoes_l3691_369155

/-- The number of tomatoes Mabel has -/
def total_tomatoes (plant1 plant2 plant3 plant4 : ℕ) : ℕ :=
  plant1 + plant2 + plant3 + plant4

/-- Theorem stating the total number of tomatoes Mabel has -/
theorem mabels_tomatoes :
  ∃ (plant1 plant2 plant3 plant4 : ℕ),
    plant1 = 8 ∧
    plant2 = plant1 + 4 ∧
    plant3 = 3 * (plant1 + plant2) ∧
    plant4 = 3 * (plant1 + plant2) ∧
    total_tomatoes plant1 plant2 plant3 plant4 = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_mabels_tomatoes_l3691_369155


namespace NUMINAMATH_CALUDE_markup_percentage_is_30_l3691_369162

/-- Represents the markup percentage applied by a merchant -/
def markup_percentage : ℝ → ℝ := sorry

/-- Represents the discount percentage applied to the marked price -/
def discount_percentage : ℝ := 10

/-- Represents the profit percentage after discount -/
def profit_percentage : ℝ := 17

/-- Theorem stating that given the conditions, the markup percentage is 30% -/
theorem markup_percentage_is_30 :
  ∀ (cost_price : ℝ),
  cost_price > 0 →
  let marked_price := cost_price * (1 + markup_percentage cost_price / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + profit_percentage / 100) →
  markup_percentage cost_price = 30 :=
by sorry

end NUMINAMATH_CALUDE_markup_percentage_is_30_l3691_369162


namespace NUMINAMATH_CALUDE_sum_of_primes_even_l3691_369126

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem sum_of_primes_even 
  (A B C : ℕ) 
  (hA : isPrime A) 
  (hB : isPrime B) 
  (hC : isPrime C) 
  (hAB_minus : isPrime (A - B)) 
  (hAB_plus : isPrime (A + B)) 
  (hABC : isPrime (A + B + C)) : 
  Even (A + B + C + (A - B) + (A + B) + (A + B + C)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_even_l3691_369126


namespace NUMINAMATH_CALUDE_circle_intersection_trajectory_l3691_369193

/-- Given two circles with centers at (a₁, 0) and (a₂, 0), both passing through (1, 0),
    intersecting the positive y-axis at (0, y₁) and (0, y₂) respectively,
    prove that the trajectory of (1/a₁, 1/a₂) is a straight line when ln y₁ + ln y₂ = 0 -/
theorem circle_intersection_trajectory (a₁ a₂ y₁ y₂ : ℝ) 
    (h1 : (1 - a₁)^2 = a₁^2 + y₁^2)
    (h2 : (1 - a₂)^2 = a₂^2 + y₂^2)
    (h3 : Real.log y₁ + Real.log y₂ = 0) :
    ∃ (m b : ℝ), ∀ (x y : ℝ), (x = 1/a₁ ∧ y = 1/a₂) → y = m*x + b :=
  sorry


end NUMINAMATH_CALUDE_circle_intersection_trajectory_l3691_369193


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3691_369101

theorem simplify_fraction_product : 
  4 * (18 / 5) * (25 / -45) * (10 / 8) = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3691_369101


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_zoo_ticket_cost_example_l3691_369147

/-- Calculate the total cost of zoo tickets for a group with a discount --/
theorem zoo_ticket_cost (num_children num_adults num_seniors : ℕ)
                        (child_price adult_price senior_price : ℚ)
                        (discount_rate : ℚ) : ℚ :=
  let total_before_discount := num_children * child_price +
                               num_adults * adult_price +
                               num_seniors * senior_price
  let discount_amount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  total_after_discount

/-- Prove that the total cost of zoo tickets for the given group is $227.80 --/
theorem zoo_ticket_cost_example : zoo_ticket_cost 6 10 4 10 16 12 (15/100) = 227.8 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_zoo_ticket_cost_example_l3691_369147


namespace NUMINAMATH_CALUDE_fires_put_out_l3691_369112

/-- The number of fires Doug put out -/
def doug_fires : ℕ := 20

/-- The number of fires Kai put out -/
def kai_fires : ℕ := 3 * doug_fires

/-- The number of fires Eli put out -/
def eli_fires : ℕ := kai_fires / 2

/-- The total number of fires put out by Doug, Kai, and Eli -/
def total_fires : ℕ := doug_fires + kai_fires + eli_fires

theorem fires_put_out : total_fires = 110 := by
  sorry

end NUMINAMATH_CALUDE_fires_put_out_l3691_369112


namespace NUMINAMATH_CALUDE_parsnip_box_ratio_l3691_369113

/-- Represents the number of parsnips in a full box -/
def full_box_capacity : ℕ := 20

/-- Represents the total number of boxes in an average harvest -/
def total_boxes : ℕ := 20

/-- Represents the total number of parsnips in an average harvest -/
def total_parsnips : ℕ := 350

/-- Represents the number of full boxes -/
def full_boxes : ℕ := 15

/-- Represents the number of half-full boxes -/
def half_full_boxes : ℕ := total_boxes - full_boxes

theorem parsnip_box_ratio :
  (full_boxes : ℚ) / total_boxes = 3 / 4 ∧
  full_boxes + half_full_boxes = total_boxes ∧
  full_boxes * full_box_capacity + half_full_boxes * (full_box_capacity / 2) = total_parsnips :=
by sorry

end NUMINAMATH_CALUDE_parsnip_box_ratio_l3691_369113


namespace NUMINAMATH_CALUDE_max_cookies_andy_l3691_369118

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  andy : ℕ
  alexa : ℕ
  john : ℕ

/-- Checks if a cookie distribution is valid according to the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.alexa = 2 * d.andy + 2 ∧
  d.john = d.andy - 3 ∧
  d.andy + d.alexa + d.john = 30

/-- Theorem stating that the maximum number of cookies Andy can eat is 7 -/
theorem max_cookies_andy :
  ∀ d : CookieDistribution, isValidDistribution d → d.andy ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l3691_369118


namespace NUMINAMATH_CALUDE_relay_race_time_difference_l3691_369168

theorem relay_race_time_difference 
  (total_time : ℕ) 
  (jen_time : ℕ) 
  (susan_time : ℕ) 
  (mary_time : ℕ) 
  (tiffany_time : ℕ) :
  total_time = 223 →
  jen_time = 30 →
  susan_time = jen_time + 10 →
  mary_time = 2 * susan_time →
  tiffany_time < mary_time →
  total_time = mary_time + susan_time + jen_time + tiffany_time →
  mary_time - tiffany_time = 7 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_time_difference_l3691_369168


namespace NUMINAMATH_CALUDE_production_increase_l3691_369123

theorem production_increase (n : ℕ) (old_avg new_avg : ℚ) (today_production : ℚ) : 
  n = 19 →
  old_avg = 50 →
  new_avg = 52 →
  today_production = n * old_avg + today_production →
  (n + 1) * new_avg = n * old_avg + today_production →
  today_production = 90 := by
  sorry

end NUMINAMATH_CALUDE_production_increase_l3691_369123


namespace NUMINAMATH_CALUDE_log_equation_solution_l3691_369196

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1)
  (h_eq : Real.log x / (3 * Real.log b) + Real.log b / (3 * Real.log x) = 1) :
  x = b ∨ x = b ^ ((3 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3691_369196


namespace NUMINAMATH_CALUDE_product_of_polynomials_l3691_369143

theorem product_of_polynomials (p x y : ℝ) : 
  (2 * p^2 - 5 * p + x) * (5 * p^2 + y * p - 10) = 10 * p^4 + 5 * p^3 - 65 * p^2 + 40 * p + 40 →
  x + y = -6.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l3691_369143


namespace NUMINAMATH_CALUDE_simplify_fraction_l3691_369109

theorem simplify_fraction (a : ℚ) (h : a = 3) : 10 * a^3 / (55 * a^2) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3691_369109


namespace NUMINAMATH_CALUDE_reverse_order_product_sum_l3691_369159

/-- Checks if two positive integers have reverse digit order -/
def are_reverse_order (a b : ℕ) : Prop := sorry

/-- Given two positive integers m and n with reverse digit order and m * n = 1446921630, prove m + n = 79497 -/
theorem reverse_order_product_sum (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : are_reverse_order m n) 
  (h4 : m * n = 1446921630) : 
  m + n = 79497 := by sorry

end NUMINAMATH_CALUDE_reverse_order_product_sum_l3691_369159


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3691_369139

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (1, 2)
  let n : ℝ → ℝ × ℝ := λ x ↦ (x, 2 - 2*x)
  ∀ x : ℝ, are_parallel m (n x) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3691_369139


namespace NUMINAMATH_CALUDE_equilateral_triangle_product_l3691_369105

/-- Given that (0,0), (a,19), and (b,61) are vertices of an equilateral triangle, prove that ab = 7760/9 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (∀ (z : ℂ), z^3 = 1 ∧ z ≠ 1 → (a + 19*I) * z = b + 61*I) → 
  a * b = 7760 / 9 := by
sorry


end NUMINAMATH_CALUDE_equilateral_triangle_product_l3691_369105
