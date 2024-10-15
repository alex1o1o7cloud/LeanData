import Mathlib

namespace NUMINAMATH_CALUDE_line_through_point_perpendicular_to_line_l3373_337393

/-- Given a point A and two lines l₁ and l₂, this theorem states that
    l₂ passes through A and is perpendicular to l₁. -/
theorem line_through_point_perpendicular_to_line
  (A : ℝ × ℝ)  -- Point A
  (l₁ : ℝ → ℝ → Prop)  -- Line l₁
  (l₂ : ℝ → ℝ → Prop)  -- Line l₂
  (h₁ : l₁ = fun x y ↦ 2 * x + 3 * y + 4 = 0)  -- Equation of l₁
  (h₂ : l₂ = fun x y ↦ 3 * x - 2 * y + 7 = 0)  -- Equation of l₂
  (h₃ : A = (-1, 2))  -- Coordinates of point A
  : (l₂ (A.1) (A.2)) ∧  -- l₂ passes through A
    (∀ (x y : ℝ), l₁ x y → l₂ x y → (2 * 3 + 3 * (-2) = 0)) :=  -- l₁ ⊥ l₂
by sorry

end NUMINAMATH_CALUDE_line_through_point_perpendicular_to_line_l3373_337393


namespace NUMINAMATH_CALUDE_rubies_in_chest_l3373_337367

theorem rubies_in_chest (diamonds : ℕ) (difference : ℕ) (rubies : ℕ) : 
  diamonds = 421 → difference = 44 → diamonds = rubies + difference → rubies = 377 := by
  sorry

end NUMINAMATH_CALUDE_rubies_in_chest_l3373_337367


namespace NUMINAMATH_CALUDE_f_increasing_and_not_in_second_quadrant_l3373_337386

-- Define the function
def f (x : ℝ) : ℝ := 2 * x - 5

-- State the theorem
theorem f_increasing_and_not_in_second_quadrant :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x y : ℝ, x < 0 ∧ y > 0 → ¬(f x = y)) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_and_not_in_second_quadrant_l3373_337386


namespace NUMINAMATH_CALUDE_intersection_theorem_l3373_337368

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x^2 < 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem :
  A_intersect_B = {x | -1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3373_337368


namespace NUMINAMATH_CALUDE_min_value_of_function_l3373_337350

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ 5/2 ∧ 
  ∃ y : ℝ, (y^2 + 5) / Real.sqrt (y^2 + 4) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3373_337350


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3373_337354

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ ∀ (a b c : ℝ), a + 2*b + c = 1 → a^2 + 4*b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3373_337354


namespace NUMINAMATH_CALUDE_cubic_equation_with_geometric_progression_roots_l3373_337356

theorem cubic_equation_with_geometric_progression_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 11*x^2 + a*x - 8 = 0 ∧
    y^3 - 11*y^2 + a*y - 8 = 0 ∧
    z^3 - 11*z^2 + a*z - 8 = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ y = x*q ∧ z = y*q) →
  a = 22 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_geometric_progression_roots_l3373_337356


namespace NUMINAMATH_CALUDE_sin_linear_dependence_l3373_337335

theorem sin_linear_dependence :
  ∃ (α₁ α₂ α₃ : ℝ), (α₁ ≠ 0 ∨ α₂ ≠ 0 ∨ α₃ ≠ 0) ∧
  ∀ x : ℝ, α₁ * Real.sin x + α₂ * Real.sin (x + π/8) + α₃ * Real.sin (x - π/8) = 0 := by
sorry

end NUMINAMATH_CALUDE_sin_linear_dependence_l3373_337335


namespace NUMINAMATH_CALUDE_five_by_five_perimeter_l3373_337352

/-- The number of points on the perimeter of a square grid -/
def perimeterPoints (n : ℕ) : ℕ := 4 * n - 4

/-- Theorem: The number of points on the perimeter of a 5x5 grid is 16 -/
theorem five_by_five_perimeter : perimeterPoints 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_by_five_perimeter_l3373_337352


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l3373_337374

/-- Represents a 9x9 checkerboard filled with numbers 1 through 81 in order across rows. -/
def Checkerboard := Fin 9 → Fin 9 → Nat

/-- The value at position (i, j) on the checkerboard. -/
def checkerboardValue (i j : Fin 9) : Nat :=
  i.val * 9 + j.val + 1

/-- The sum of the values in the four corners of the checkerboard. -/
def cornerSum (board : Checkerboard) : Nat :=
  board 0 0 + board 0 8 + board 8 0 + board 8 8

/-- Theorem stating that the sum of the numbers in the four corners of the checkerboard is 164. -/
theorem corner_sum_is_164 (board : Checkerboard) :
  (∀ i j : Fin 9, board i j = checkerboardValue i j) →
  cornerSum board = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l3373_337374


namespace NUMINAMATH_CALUDE_sam_and_mary_balloons_l3373_337323

/-- The total number of yellow balloons Sam and Mary have together -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (mary : ℝ) : ℝ :=
  (sam_initial - sam_given) + mary

/-- Proof that Sam and Mary have 8.0 yellow balloons in total -/
theorem sam_and_mary_balloons :
  total_balloons 6.0 5.0 7.0 = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_and_mary_balloons_l3373_337323


namespace NUMINAMATH_CALUDE_min_value_of_fraction_min_value_achieved_l3373_337342

theorem min_value_of_fraction (x : ℝ) (h : x > 9) : 
  (x^2) / (x - 9) ≥ 36 := by
sorry

theorem min_value_achieved (x : ℝ) (h : x > 9) : 
  (x^2) / (x - 9) = 36 ↔ x = 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_min_value_achieved_l3373_337342


namespace NUMINAMATH_CALUDE_no_three_common_tangents_l3373_337346

/-- Two circles in the same plane with different radii -/
structure TwoCircles where
  plane : Type*
  circle1 : Set plane
  circle2 : Set plane
  radius1 : ℝ
  radius2 : ℝ
  different_radii : radius1 ≠ radius2

/-- A common tangent to two circles -/
def CommonTangent (tc : TwoCircles) (line : Set tc.plane) : Prop := sorry

/-- The number of common tangents to two circles -/
def NumCommonTangents (tc : TwoCircles) : ℕ := sorry

/-- Theorem: Two circles with different radii cannot have exactly 3 common tangents -/
theorem no_three_common_tangents (tc : TwoCircles) : 
  NumCommonTangents tc ≠ 3 := by sorry

end NUMINAMATH_CALUDE_no_three_common_tangents_l3373_337346


namespace NUMINAMATH_CALUDE_cookies_per_day_l3373_337312

-- Define the problem parameters
def cookie_cost : ℕ := 19
def total_spent : ℕ := 2356
def days_in_march : ℕ := 31

-- Define the theorem
theorem cookies_per_day :
  (total_spent / cookie_cost) / days_in_march = 4 := by
  sorry


end NUMINAMATH_CALUDE_cookies_per_day_l3373_337312


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l3373_337333

theorem polynomial_root_relation (p q r s : ℝ) (h_p : p ≠ 0) :
  (p * (4 : ℝ)^3 + q * (4 : ℝ)^2 + r * (4 : ℝ) + s = 0) →
  (p * (-3 : ℝ)^3 + q * (-3 : ℝ)^2 + r * (-3 : ℝ) + s = 0) →
  (q + r) / p = -13 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l3373_337333


namespace NUMINAMATH_CALUDE_corresponding_angles_relationships_l3373_337315

/-- Two angles are corresponding if they occupy the same relative position at each intersection where a straight line crosses two others. -/
def corresponding_angles (α β : Real) : Prop := sorry

/-- The statement that all relationships (equal, greater than, less than) are possible for corresponding angles. -/
theorem corresponding_angles_relationships (α β : Real) (h : corresponding_angles α β) :
  (∃ (α₁ β₁ : Real), corresponding_angles α₁ β₁ ∧ α₁ = β₁) ∧
  (∃ (α₂ β₂ : Real), corresponding_angles α₂ β₂ ∧ α₂ > β₂) ∧
  (∃ (α₃ β₃ : Real), corresponding_angles α₃ β₃ ∧ α₃ < β₃) :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_relationships_l3373_337315


namespace NUMINAMATH_CALUDE_tetrahedron_cube_volume_ratio_l3373_337344

theorem tetrahedron_cube_volume_ratio :
  let cube_side : ℝ := x
  let cube_volume := cube_side ^ 3
  let tetrahedron_side := cube_side * Real.sqrt 3 / 2
  let tetrahedron_volume := tetrahedron_side ^ 3 * Real.sqrt 2 / 12
  tetrahedron_volume / cube_volume = Real.sqrt 6 / 32 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_cube_volume_ratio_l3373_337344


namespace NUMINAMATH_CALUDE_set_difference_N_M_l3373_337314

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 2, 3, 7}

theorem set_difference_N_M : N \ M = {7} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_N_M_l3373_337314


namespace NUMINAMATH_CALUDE_gravel_cost_theorem_l3373_337343

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 4

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 8

/-- The total cost of gravel for a given volume in cubic yards -/
def total_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yards_to_cubic_feet * gravel_cost_per_cubic_foot

theorem gravel_cost_theorem : total_cost gravel_volume_cubic_yards = 864 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_theorem_l3373_337343


namespace NUMINAMATH_CALUDE_workshop_workers_l3373_337369

/-- The total number of workers in a workshop, given specific salary conditions -/
theorem workshop_workers (average_salary : ℚ) (technician_salary : ℚ) (other_salary : ℚ)
  (h1 : average_salary = 8000)
  (h2 : technician_salary = 18000)
  (h3 : other_salary = 6000) :
  ∃ (total_workers : ℕ), 
    (7 : ℚ) * technician_salary + (total_workers - 7 : ℚ) * other_salary = (total_workers : ℚ) * average_salary ∧
    total_workers = 42 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l3373_337369


namespace NUMINAMATH_CALUDE_polynomial_equality_l3373_337340

theorem polynomial_equality (q : Polynomial ℝ) :
  (q + (2 * X^4 - 5 * X^2 + 8 * X + 3) = 10 * X^3 - 7 * X^2 + 15 * X + 6) →
  q = -2 * X^4 + 10 * X^3 - 2 * X^2 + 7 * X + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3373_337340


namespace NUMINAMATH_CALUDE_intersection_union_when_m_3_intersection_equals_B_implies_m_range_l3373_337302

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 1}

-- Part 1
theorem intersection_union_when_m_3 :
  (A ∩ B 3) = {x | 1 ≤ x ∧ x ≤ 3} ∧
  (A ∪ B 3) = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2
theorem intersection_equals_B_implies_m_range (m : ℝ) :
  A ∩ B m = B m → 1 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_union_when_m_3_intersection_equals_B_implies_m_range_l3373_337302


namespace NUMINAMATH_CALUDE_factors_of_expression_l3373_337399

theorem factors_of_expression : 
  ∃ (a b : ℕ), 60 < a ∧ a < b ∧ b < 70 ∧ 
  (29 * 26 * (2^48 - 1)) % a = 0 ∧ 
  (29 * 26 * (2^48 - 1)) % b = 0 ∧
  (∀ c, 60 < c ∧ c < 70 ∧ (29 * 26 * (2^48 - 1)) % c = 0 → c = a ∨ c = b) ∧
  a = 63 ∧ b = 65 :=
sorry

end NUMINAMATH_CALUDE_factors_of_expression_l3373_337399


namespace NUMINAMATH_CALUDE_yonder_license_plates_l3373_337336

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The total number of possible license plates in Yonder -/
def total_license_plates : ℕ := num_letters ^ 3 * num_digits ^ 3

theorem yonder_license_plates :
  total_license_plates = 17576000 := by sorry

end NUMINAMATH_CALUDE_yonder_license_plates_l3373_337336


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3373_337311

theorem divisibility_equivalence (n : ℤ) : 
  let A := n % 1000
  let B := n / 1000
  let k := A - B
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3373_337311


namespace NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_16_to_30_main_theorem_l3373_337306

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_first_15 :
  sum_of_squares 15 = 1240 :=
by sorry

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8205 :=
by sorry

theorem main_theorem :
  sum_of_squares 15 = 1240 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_16_to_30_main_theorem_l3373_337306


namespace NUMINAMATH_CALUDE_correct_num_students_l3373_337362

/-- The number of students in a class with incorrectly entered marks -/
def num_students : ℕ := by sorry

/-- The total increase in marks due to incorrect entry -/
def total_mark_increase : ℕ := 44

/-- The increase in class average due to incorrect entry -/
def average_increase : ℚ := 1/2

theorem correct_num_students :
  num_students = 88 ∧
  (total_mark_increase : ℚ) = num_students * average_increase := by sorry

end NUMINAMATH_CALUDE_correct_num_students_l3373_337362


namespace NUMINAMATH_CALUDE_student_selection_and_advancement_probability_l3373_337328

-- Define the scores for students A and B
def scores_A : List ℕ := [100, 90, 120, 130, 105, 115]
def scores_B : List ℕ := [95, 125, 110, 95, 100, 135]

-- Define a function to calculate the average score
def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Define a function to calculate the variance
def variance (scores : List ℕ) : ℚ :=
  let avg := average scores
  (scores.map (λ x => ((x : ℚ) - avg) ^ 2)).sum / scores.length

-- Define a function to select the student with lower variance
def select_student (scores_A scores_B : List ℕ) : Bool :=
  variance scores_A < variance scores_B

-- Define the probability of advancing to the final round
def probability_advance : ℚ := 7 / 10

-- Theorem statement
theorem student_selection_and_advancement_probability 
  (scores_A scores_B : List ℕ) 
  (h_scores_A : scores_A = [100, 90, 120, 130, 105, 115])
  (h_scores_B : scores_B = [95, 125, 110, 95, 100, 135]) :
  select_student scores_A scores_B = true ∧ 
  probability_advance = 7 / 10 := by
  sorry


end NUMINAMATH_CALUDE_student_selection_and_advancement_probability_l3373_337328


namespace NUMINAMATH_CALUDE_f_at_2_l3373_337388

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem f_at_2 : f 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l3373_337388


namespace NUMINAMATH_CALUDE_stream_speed_l3373_337358

/-- Proves that given a boat with a speed of 13 km/hr in still water,
    traveling 68 km downstream in 4 hours, the speed of the stream is 4 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 13 →
  distance = 68 →
  time = 4 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l3373_337358


namespace NUMINAMATH_CALUDE_sqrt_63_minus_7_sqrt_one_seventh_l3373_337359

theorem sqrt_63_minus_7_sqrt_one_seventh (x : ℝ) : 
  Real.sqrt 63 - 7 * Real.sqrt (1 / 7) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_63_minus_7_sqrt_one_seventh_l3373_337359


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l3373_337313

theorem square_area_with_four_circles (r : ℝ) (h : r = 3) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l3373_337313


namespace NUMINAMATH_CALUDE_commission_allocation_l3373_337341

theorem commission_allocation (commission_rate : ℚ) (total_sales : ℚ) (amount_saved : ℚ)
  (h1 : commission_rate = 12 / 100)
  (h2 : total_sales = 24000)
  (h3 : amount_saved = 1152) :
  (total_sales * commission_rate - amount_saved) / (total_sales * commission_rate) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_commission_allocation_l3373_337341


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3373_337317

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℚ, x^4 = (x^3 + 3*x^2 + 2*x + 1) * q + (-x^2 - x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3373_337317


namespace NUMINAMATH_CALUDE_area_of_region_l3373_337380

-- Define the circle and chord properties
def circle_radius : ℝ := 50
def chord_length : ℝ := 84
def intersection_distance : ℝ := 24

-- Define the area calculation function
def area_calculation (r : ℝ) (c : ℝ) (d : ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_region :
  area_calculation circle_radius chord_length intersection_distance = 1250 * Real.sqrt 3 + (1250 / 3) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l3373_337380


namespace NUMINAMATH_CALUDE_twenty_knocks_to_knicks_l3373_337309

-- Define the units
variable (knick knack knock : ℚ)

-- Define the given conditions
axiom knicks_to_knacks : 8 * knick = 3 * knack
axiom knacks_to_knocks : 4 * knack = 5 * knock

-- State the theorem
theorem twenty_knocks_to_knicks : 
  20 * knock = 64 / 3 * knick :=
sorry

end NUMINAMATH_CALUDE_twenty_knocks_to_knicks_l3373_337309


namespace NUMINAMATH_CALUDE_cos_54_degrees_l3373_337360

theorem cos_54_degrees : Real.cos (54 * π / 180) = Real.sqrt ((3 + Real.sqrt 5) / 8) := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l3373_337360


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l3373_337316

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) :
  n = 15 →
  ∃ k : ℕ, (1 : ℚ) / 3^n = k / 10 + (0 : ℚ) / 10 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l3373_337316


namespace NUMINAMATH_CALUDE_english_only_enrollment_l3373_337382

/-- Represents the number of students in different enrollment categories -/
structure EnrollmentCount where
  total : ℕ
  bothSubjects : ℕ
  germanTotal : ℕ

/-- Calculates the number of students enrolled only in English -/
def studentsOnlyEnglish (e : EnrollmentCount) : ℕ :=
  e.total - e.germanTotal

/-- Theorem: Given the enrollment conditions, 28 students are enrolled only in English -/
theorem english_only_enrollment (e : EnrollmentCount) 
  (h1 : e.total = 50)
  (h2 : e.bothSubjects = 12)
  (h3 : e.germanTotal = 22)
  (h4 : e.total ≥ e.germanTotal) : 
  studentsOnlyEnglish e = 28 := by
  sorry

#check english_only_enrollment

end NUMINAMATH_CALUDE_english_only_enrollment_l3373_337382


namespace NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l3373_337318

/-- Lagrange interpolation polynomial for the given points -/
def P₂ (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 8

/-- The x-coordinates of the interpolation points -/
def x₀ : ℝ := -3
def x₁ : ℝ := -1
def x₂ : ℝ := 2

/-- The y-coordinates of the interpolation points -/
def y₀ : ℝ := -5
def y₁ : ℝ := -11
def y₂ : ℝ := 10

/-- Theorem stating that P₂ is the Lagrange interpolation polynomial for the given points -/
theorem lagrange_interpolation_polynomial :
  P₂ x₀ = y₀ ∧ P₂ x₁ = y₁ ∧ P₂ x₂ = y₂ := by
  sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l3373_337318


namespace NUMINAMATH_CALUDE_sum_of_base3_digits_333_l3373_337334

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits in the base-3 representation of 333 is 3 -/
theorem sum_of_base3_digits_333 : sumDigits (toBase3 333) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base3_digits_333_l3373_337334


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3373_337310

def A : Set (ℝ × ℝ) := {p | p.1 + p.2 = 5}
def B : Set (ℝ × ℝ) := {p | p.1 - p.2 = 1}

theorem intersection_of_A_and_B : A ∩ B = {(3, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3373_337310


namespace NUMINAMATH_CALUDE_prank_combinations_l3373_337324

theorem prank_combinations (choices : List Nat) : 
  choices = [1, 3, 5, 6, 2] → choices.prod = 180 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l3373_337324


namespace NUMINAMATH_CALUDE_z_absolute_value_range_l3373_337327

open Complex

theorem z_absolute_value_range (t : ℝ) :
  let z : ℂ := (sin t / Real.sqrt 2 + I * cos t) / (sin t - I * cos t / Real.sqrt 2)
  1 / Real.sqrt 2 ≤ abs z ∧ abs z ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_z_absolute_value_range_l3373_337327


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l3373_337371

theorem geometric_sequence_solution :
  ∃! (x : ℝ), x > 0 ∧ (∃ (r : ℝ), 12 * r = x ∧ x * r = 2/3) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l3373_337371


namespace NUMINAMATH_CALUDE_total_cost_l3373_337376

/-- The cost of an enchilada -/
def e : ℝ := sorry

/-- The cost of a taco -/
def t : ℝ := sorry

/-- The cost of a burrito -/
def b : ℝ := sorry

/-- The first condition: 4 enchiladas, 5 tacos, and 2 burritos cost $8.20 -/
axiom condition1 : 4 * e + 5 * t + 2 * b = 8.20

/-- The second condition: 6 enchiladas, 3 tacos, and 4 burritos cost $9.40 -/
axiom condition2 : 6 * e + 3 * t + 4 * b = 9.40

/-- Theorem stating that the total cost of 5 enchiladas, 6 tacos, and 3 burritos is $12.20 -/
theorem total_cost : 5 * e + 6 * t + 3 * b = 12.20 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_l3373_337376


namespace NUMINAMATH_CALUDE_norris_savings_l3373_337347

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Norris saved in December -/
def december_savings : ℕ := 35

/-- The amount of money Norris saved in January -/
def january_savings : ℕ := 40

/-- The total amount of money Norris saved from September to January -/
def total_savings : ℕ := september_savings + october_savings + november_savings + december_savings + january_savings

theorem norris_savings : total_savings = 160 := by
  sorry

end NUMINAMATH_CALUDE_norris_savings_l3373_337347


namespace NUMINAMATH_CALUDE_unique_rational_pair_l3373_337387

theorem unique_rational_pair : 
  ∀ (a b r s : ℚ), 
    a ≠ b → 
    r ≠ s → 
    (∀ (z : ℚ), (z - r) * (z - s) = (z - a*r) * (z - b*s)) → 
    ∃! (p : ℚ × ℚ), p.1 ≠ p.2 ∧ 
      ∀ (z : ℚ), (z - r) * (z - s) = (z - p.1*r) * (z - p.2*s) :=
by sorry

end NUMINAMATH_CALUDE_unique_rational_pair_l3373_337387


namespace NUMINAMATH_CALUDE_preimage_of_neg_one_plus_two_i_l3373_337345

/-- The complex transformation f(Z) = (1+i)Z -/
def f (Z : ℂ) : ℂ := (1 + Complex.I) * Z

/-- Theorem: The pre-image of -1+2i under f is (1+3i)/2 -/
theorem preimage_of_neg_one_plus_two_i :
  f ((1 + 3 * Complex.I) / 2) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_neg_one_plus_two_i_l3373_337345


namespace NUMINAMATH_CALUDE_function_property_l3373_337322

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

def domain (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem function_property (f : ℝ → ℝ) (h : ∀ x y, domain x → domain y → f x + f y = f ((x + y) / (1 + x * y))) :
  ∀ x, domain x → f x = Real.log ((1 - x) / (1 + x)) →
  (∀ x, domain x → f (-x) = -f x) ∧
  ∃ a b, domain a ∧ domain b ∧ 
    f ((a + b) / (1 + a * b)) = 1 ∧ 
    f ((a - b) / (1 - a * b)) = 2 ∧
    f a = 3/2 ∧ f b = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l3373_337322


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3373_337325

theorem girls_to_boys_ratio (girls boys : ℕ) (h1 : girls = 10) (h2 : boys = 20) :
  (girls : ℚ) / (boys : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3373_337325


namespace NUMINAMATH_CALUDE_sequence_2013_value_l3373_337390

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ (p k : ℕ), Nat.Prime p → k > 0 → a (p * k + 1) = p * a k - 3 * a p + 13

theorem sequence_2013_value (a : ℕ → ℤ) (h : is_valid_sequence a) : a 2013 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2013_value_l3373_337390


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3373_337353

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) ↔ (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3373_337353


namespace NUMINAMATH_CALUDE_inverse_sum_modulo_eleven_l3373_337301

theorem inverse_sum_modulo_eleven :
  (((3⁻¹ : ZMod 11) + (5⁻¹ : ZMod 11) + (7⁻¹ : ZMod 11))⁻¹ : ZMod 11) = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_modulo_eleven_l3373_337301


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3373_337332

theorem necessary_not_sufficient (p q : Prop) :
  (¬p → ¬(p ∨ q)) ∧ ¬(¬p → ¬(p ∨ q)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3373_337332


namespace NUMINAMATH_CALUDE_extreme_points_cubic_l3373_337339

/-- A function f(x) = x^3 + ax has exactly two extreme points on R if and only if a < 0 -/
theorem extreme_points_cubic (a : ℝ) :
  (∃! (p q : ℝ), p ≠ q ∧ 
    (∀ x : ℝ, (3 * x^2 + a = 0) ↔ (x = p ∨ x = q))) ↔ 
  a < 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_cubic_l3373_337339


namespace NUMINAMATH_CALUDE_max_sum_of_product_48_l3373_337379

theorem max_sum_of_product_48 :
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧
  ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_product_48_l3373_337379


namespace NUMINAMATH_CALUDE_female_contestant_probability_l3373_337372

theorem female_contestant_probability :
  let total_contestants : ℕ := 8
  let female_contestants : ℕ := 4
  let male_contestants : ℕ := 4
  let chosen_contestants : ℕ := 2
  
  (female_contestants.choose chosen_contestants : ℚ) / (total_contestants.choose chosen_contestants) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_female_contestant_probability_l3373_337372


namespace NUMINAMATH_CALUDE_function_value_at_sine_l3373_337300

/-- Given a function f(x) = 4x² + 2x, prove that f(sin(7π/6)) = 0 -/
theorem function_value_at_sine (f : ℝ → ℝ) : 
  (∀ x, f x = 4 * x^2 + 2 * x) → f (Real.sin (7 * π / 6)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_sine_l3373_337300


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l3373_337384

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l3373_337384


namespace NUMINAMATH_CALUDE_triangle_area_l3373_337308

/-- Given a triangle with perimeter 32 cm and inradius 3.5 cm, its area is 56 cm². -/
theorem triangle_area (p r A : ℝ) (h1 : p = 32) (h2 : r = 3.5) (h3 : A = r * p / 2) : A = 56 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3373_337308


namespace NUMINAMATH_CALUDE_no_solution_base_conversion_l3373_337329

theorem no_solution_base_conversion : ¬∃ (d : ℕ), d ≤ 9 ∧ d * 5 + 2 = d * 9 + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_base_conversion_l3373_337329


namespace NUMINAMATH_CALUDE_base7_sum_property_l3373_337397

/-- A digit in base 7 is a natural number less than 7 -/
def Digit7 : Type := { n : ℕ // n < 7 }

/-- Convert a three-digit number in base 7 to its decimal representation -/
def toDecimal (d e f : Digit7) : ℕ := 49 * d.val + 7 * e.val + f.val

/-- The sum of three permutations of a three-digit number in base 7 -/
def sumPermutations (d e f : Digit7) : ℕ :=
  toDecimal d e f + toDecimal e f d + toDecimal f d e

theorem base7_sum_property (d e f : Digit7) 
  (h_distinct : d ≠ e ∧ d ≠ f ∧ e ≠ f) 
  (h_nonzero : d.val ≠ 0 ∧ e.val ≠ 0 ∧ f.val ≠ 0)
  (h_sum : sumPermutations d e f = 400 * d.val) :
  e.val + f.val = 6 :=
sorry

end NUMINAMATH_CALUDE_base7_sum_property_l3373_337397


namespace NUMINAMATH_CALUDE_area_trapezoid_equals_rectangle_l3373_337361

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the shapes
def rectangle_PQRS : Set (ℝ × ℝ) := sorry
def trapezoid_TQSR : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- State the theorem
theorem area_trapezoid_equals_rectangle
  (h1 : area rectangle_PQRS = 20)
  (h2 : trapezoid_TQSR ⊆ rectangle_PQRS)
  (h3 : P = (0, 0))
  (h4 : Q = (5, 0))
  (h5 : R = (5, 4))
  (h6 : S = (0, 4))
  (h7 : T = (2, 4)) :
  area trapezoid_TQSR = area rectangle_PQRS :=
by sorry

end NUMINAMATH_CALUDE_area_trapezoid_equals_rectangle_l3373_337361


namespace NUMINAMATH_CALUDE_variance_of_doubled_data_l3373_337383

-- Define a set of data as a list of real numbers
def DataSet := List ℝ

-- Define the standard deviation of a data set
noncomputable def standardDeviation (data : DataSet) : ℝ := sorry

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define a function to double each element in a data set
def doubleData (data : DataSet) : DataSet := data.map (· * 2)

-- Theorem statement
theorem variance_of_doubled_data (data : DataSet) :
  let s := standardDeviation data
  variance (doubleData data) = 4 * (s ^ 2) := by sorry

end NUMINAMATH_CALUDE_variance_of_doubled_data_l3373_337383


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l3373_337391

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the points E and F on the sides of the triangle
def E (triangle : Triangle) : Point := sorry
def F (triangle : Triangle) : Point := sorry

-- Define the angles
def angle_ABC (triangle : Triangle) : ℝ := sorry
def angle_AEC (triangle : Triangle) (circle : Circle) : ℝ := sorry
def angle_BAF (triangle : Triangle) (circle : Circle) : ℝ := sorry

-- Define the length of AC
def length_AC (triangle : Triangle) : ℝ := sorry

-- State the theorem
theorem circle_radius_theorem (triangle : Triangle) (circle : Circle) :
  angle_ABC triangle = 72 →
  angle_AEC triangle circle = 5 * angle_BAF triangle circle →
  length_AC triangle = 6 →
  circle.radius = 3 := by sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l3373_337391


namespace NUMINAMATH_CALUDE_marbles_left_theorem_l3373_337319

def initial_marbles : ℕ := 87
def marbles_given_away : ℕ := 8

theorem marbles_left_theorem : 
  initial_marbles - marbles_given_away = 79 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_theorem_l3373_337319


namespace NUMINAMATH_CALUDE_min_perimeter_57_triangle_hexagon_exists_57_triangle_hexagon_with_perimeter_19_l3373_337326

/-- Represents a hexagon formed by unit equilateral triangles -/
structure TriangleHexagon where
  /-- The number of unit equilateral triangles used to form the hexagon -/
  num_triangles : ℕ
  /-- The perimeter of the hexagon -/
  perimeter : ℕ
  /-- Assertion that the hexagon is formed without gaps or overlaps -/
  no_gaps_or_overlaps : Prop
  /-- Assertion that all internal angles of the hexagon are not greater than 180 degrees -/
  angles_not_exceeding_180 : Prop

/-- Theorem stating the minimum perimeter of a hexagon formed by 57 unit equilateral triangles -/
theorem min_perimeter_57_triangle_hexagon :
  ∀ h : TriangleHexagon,
    h.num_triangles = 57 →
    h.no_gaps_or_overlaps →
    h.angles_not_exceeding_180 →
    h.perimeter ≥ 19 := by
  sorry

/-- Existence of a hexagon with perimeter 19 formed by 57 unit equilateral triangles -/
theorem exists_57_triangle_hexagon_with_perimeter_19 :
  ∃ h : TriangleHexagon,
    h.num_triangles = 57 ∧
    h.perimeter = 19 ∧
    h.no_gaps_or_overlaps ∧
    h.angles_not_exceeding_180 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_57_triangle_hexagon_exists_57_triangle_hexagon_with_perimeter_19_l3373_337326


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l3373_337349

theorem sum_of_three_consecutive_integers (a b c : ℕ) : 
  (a + 1 = b ∧ b + 1 = c) → c = 7 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l3373_337349


namespace NUMINAMATH_CALUDE_system_solution_l3373_337330

theorem system_solution (x y : ℝ) : 2*x - y = 5 ∧ x - 2*y = 1 → x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3373_337330


namespace NUMINAMATH_CALUDE_divisibility_proof_l3373_337381

theorem divisibility_proof (k : ℕ) (p : ℕ) (m : ℕ) 
  (h1 : k > 1) 
  (h2 : p = 6 * k + 1) 
  (h3 : Nat.Prime p) 
  (h4 : m = 2^p - 1) : 
  (127 * m) ∣ (2^(m-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3373_337381


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l3373_337366

/-- Greatest prime factor of a positive integer -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- The theorem states that there exists exactly one positive integer n > 1
    satisfying both conditions simultaneously -/
theorem unique_n_satisfying_conditions : ∃! n : ℕ, n > 1 ∧ 
  (greatest_prime_factor n = n.sqrt) ∧ 
  (greatest_prime_factor (n + 72) = (n + 72).sqrt) := by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l3373_337366


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l3373_337378

theorem company_picnic_attendance 
  (total_employees : ℕ) 
  (men_percentage : ℝ) 
  (women_percentage : ℝ) 
  (men_attendance_rate : ℝ) 
  (women_attendance_rate : ℝ) 
  (h1 : men_percentage = 0.35) 
  (h2 : women_percentage = 1 - men_percentage) 
  (h3 : men_attendance_rate = 0.2) 
  (h4 : women_attendance_rate = 0.4) : 
  (men_percentage * men_attendance_rate + women_percentage * women_attendance_rate) * 100 = 33 := by
  sorry

#check company_picnic_attendance

end NUMINAMATH_CALUDE_company_picnic_attendance_l3373_337378


namespace NUMINAMATH_CALUDE_bugs_and_flowers_l3373_337304

theorem bugs_and_flowers (total_bugs : ℝ) (total_flowers : ℝ) 
  (h1 : total_bugs = 2.0) 
  (h2 : total_flowers = 3.0) : 
  total_flowers / total_bugs = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_bugs_and_flowers_l3373_337304


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3373_337365

/-- Given a hyperbola with equation 9x^2 - 4y^2 = -36, 
    its asymptotes are y = ±(3/2)(-ix) -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℂ), 9 * x^2 - 4 * y^2 = -36 →
  ∃ (k : ℂ), k = (3 / 2) * Complex.I ∧
  (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3373_337365


namespace NUMINAMATH_CALUDE_computer_sticker_price_l3373_337355

theorem computer_sticker_price : 
  ∀ (sticker_price : ℝ),
    (sticker_price * 0.85 - 90 = sticker_price * 0.75 - 15) →
    sticker_price = 750 := by
  sorry

end NUMINAMATH_CALUDE_computer_sticker_price_l3373_337355


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3373_337373

def A : Set Int := {0, 1, 2}
def B : Set Int := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3373_337373


namespace NUMINAMATH_CALUDE_exponent_division_l3373_337398

theorem exponent_division (a : ℝ) : a^4 / a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3373_337398


namespace NUMINAMATH_CALUDE_solve_for_y_l3373_337375

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3373_337375


namespace NUMINAMATH_CALUDE_line_intersection_point_l3373_337305

theorem line_intersection_point :
  ∃! p : ℝ × ℝ, 
    5 * p.1 - 3 * p.2 = 15 ∧ 
    4 * p.1 + 2 * p.2 = 14 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_line_intersection_point_l3373_337305


namespace NUMINAMATH_CALUDE_original_water_amount_l3373_337363

/-- Proves that the original amount of water in a glass is 15 ounces, given the daily evaporation rate,
    evaporation period, and the percentage of water evaporated. -/
theorem original_water_amount
  (daily_evaporation : ℝ)
  (evaporation_period : ℕ)
  (evaporation_percentage : ℝ)
  (h1 : daily_evaporation = 0.05)
  (h2 : evaporation_period = 15)
  (h3 : evaporation_percentage = 0.05)
  (h4 : daily_evaporation * ↑evaporation_period = evaporation_percentage * original_amount) :
  original_amount = 15 :=
by
  sorry

#check original_water_amount

end NUMINAMATH_CALUDE_original_water_amount_l3373_337363


namespace NUMINAMATH_CALUDE_number_above_210_l3373_337377

/-- The number of elements in the k-th row of the triangular array -/
def row_size (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The sum of elements up to and including the k-th row -/
def sum_up_to_row (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6

/-- The first element in the k-th row -/
def first_in_row (k : ℕ) : ℕ := sum_up_to_row (k - 1) + 1

/-- The last element in the k-th row -/
def last_in_row (k : ℕ) : ℕ := sum_up_to_row k

theorem number_above_210 :
  ∃ (k : ℕ), 
    (first_in_row k ≤ 210) ∧ 
    (210 ≤ last_in_row k) ∧ 
    (210 - first_in_row k + 1 = row_size k) ∧
    (last_in_row (k - 1) = 165) := by
  sorry

#eval sum_up_to_row 9  -- Expected: 165
#eval sum_up_to_row 10 -- Expected: 220
#eval first_in_row 10  -- Expected: 166
#eval last_in_row 9    -- Expected: 165

end NUMINAMATH_CALUDE_number_above_210_l3373_337377


namespace NUMINAMATH_CALUDE_water_level_rise_l3373_337395

/-- The rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise
  (cube_edge : ℝ)
  (vessel_length : ℝ)
  (vessel_width : ℝ)
  (h_cube_edge : cube_edge = 15)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l3373_337395


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3373_337394

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3373_337394


namespace NUMINAMATH_CALUDE_solutions_for_20_l3373_337337

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ :=
  4 * n

/-- The property that |x| + |y| = 1 has 4 solutions -/
axiom base_case : num_solutions 1 = 4

/-- The property that the number of solutions increases by 4 for each unit increase -/
axiom induction_step : ∀ n : ℕ, num_solutions (n + 1) = num_solutions n + 4

/-- The theorem to be proved -/
theorem solutions_for_20 : num_solutions 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_20_l3373_337337


namespace NUMINAMATH_CALUDE_count_valid_configurations_l3373_337331

/-- Represents a configuration of 8's with + signs inserted -/
structure Configuration where
  singles : ℕ  -- number of individual 8's
  doubles : ℕ  -- number of 88's
  triples : ℕ  -- number of 888's

/-- The total number of 8's used in a configuration -/
def Configuration.total_eights (c : Configuration) : ℕ :=
  c.singles + 2 * c.doubles + 3 * c.triples

/-- The sum of a configuration -/
def Configuration.sum (c : Configuration) : ℕ :=
  8 * c.singles + 88 * c.doubles + 888 * c.triples

/-- A configuration is valid if its sum is 8880 -/
def Configuration.is_valid (c : Configuration) : Prop :=
  c.sum = 8880

theorem count_valid_configurations :
  (∃ (s : Finset ℕ), s.card = 119 ∧
    (∀ n, n ∈ s ↔ ∃ c : Configuration, c.is_valid ∧ c.total_eights = n)) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_configurations_l3373_337331


namespace NUMINAMATH_CALUDE_f_composition_value_l3373_337320

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then |x - 1| - 2 else 1 / (1 + x^2)

theorem f_composition_value : f (f 3) = -11/10 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3373_337320


namespace NUMINAMATH_CALUDE_positive_t_value_l3373_337364

theorem positive_t_value (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = 5) →
  (a * b = t - 3 * Complex.I) →
  (t > 0) →
  t = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_positive_t_value_l3373_337364


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l3373_337338

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l3373_337338


namespace NUMINAMATH_CALUDE_tia_walking_time_l3373_337303

/-- Represents a person's walking characteristics and time to destination -/
structure Walker where
  steps_per_minute : ℝ
  step_length : ℝ
  time_to_destination : ℝ

/-- Calculates the distance walked based on walking characteristics and time -/
def distance (w : Walker) : ℝ :=
  w.steps_per_minute * w.step_length * w.time_to_destination

theorem tia_walking_time (ella tia : Walker)
  (h1 : ella.steps_per_minute = 80)
  (h2 : ella.step_length = 80)
  (h3 : ella.time_to_destination = 20)
  (h4 : tia.steps_per_minute = 120)
  (h5 : tia.step_length = 70)
  (h6 : distance ella = distance tia) :
  tia.time_to_destination = 15.24 := by
  sorry

end NUMINAMATH_CALUDE_tia_walking_time_l3373_337303


namespace NUMINAMATH_CALUDE_linear_system_fraction_sum_l3373_337307

theorem linear_system_fraction_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_system_fraction_sum_l3373_337307


namespace NUMINAMATH_CALUDE_tablecloth_black_percentage_l3373_337392

/-- Represents a square tablecloth -/
structure Tablecloth :=
  (size : ℕ)
  (black_outer_ratio : ℚ)

/-- Calculates the percentage of black area on the tablecloth -/
def black_percentage (t : Tablecloth) : ℚ :=
  let total_squares := t.size * t.size
  let outer_squares := 4 * (t.size - 1)
  let black_squares := (outer_squares : ℚ) * t.black_outer_ratio
  (black_squares / total_squares) * 100

/-- Theorem stating that a 5x5 tablecloth with half of each outer square black is 32% black -/
theorem tablecloth_black_percentage :
  let t : Tablecloth := ⟨5, 1/2⟩
  black_percentage t = 32 := by
  sorry

end NUMINAMATH_CALUDE_tablecloth_black_percentage_l3373_337392


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l3373_337321

/-- Given the costs of different combinations of pencils and pens, 
    calculate the cost of three pencils and three pens. -/
theorem pencil_pen_cost (pencil pen : ℝ) 
  (h1 : 3 * pencil + 2 * pen = 3.60)
  (h2 : 2 * pencil + 3 * pen = 3.15) :
  3 * pencil + 3 * pen = 4.05 := by
  sorry


end NUMINAMATH_CALUDE_pencil_pen_cost_l3373_337321


namespace NUMINAMATH_CALUDE_isolated_sets_intersection_empty_l3373_337357

def is_isolated_element (x : ℤ) (A : Set ℤ) : Prop :=
  x ∈ A ∧ (x - 1) ∉ A ∧ (x + 1) ∉ A

def isolated_set (A : Set ℤ) : Set ℤ :=
  {x | is_isolated_element x A}

def M : Set ℤ := {0, 1, 3}
def N : Set ℤ := {0, 3, 4}

theorem isolated_sets_intersection_empty :
  (isolated_set M) ∩ (isolated_set N) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_isolated_sets_intersection_empty_l3373_337357


namespace NUMINAMATH_CALUDE_example_implicit_function_l3373_337370

-- Define the concept of an implicit function
def is_implicit_function (F : ℝ → ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), F x y = 0

-- Define our specific function
def F (x y : ℝ) : ℝ := 2 * x - 3 * y - 1

-- Theorem statement
theorem example_implicit_function :
  is_implicit_function F :=
sorry

end NUMINAMATH_CALUDE_example_implicit_function_l3373_337370


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3373_337385

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 3 / 2)
  (hy : y / z = 4 / 3)
  (hz : z / x = 1 / 5) :
  w / y = 45 / 8 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3373_337385


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3373_337348

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3373_337348


namespace NUMINAMATH_CALUDE_log_equality_l3373_337389

theorem log_equality (y : ℝ) : y = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3) → Real.log y / Real.log 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3373_337389


namespace NUMINAMATH_CALUDE_smallest_multiple_45_div_3_l3373_337351

theorem smallest_multiple_45_div_3 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 3 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_45_div_3_l3373_337351


namespace NUMINAMATH_CALUDE_three_digit_self_repeating_powers_l3373_337396

theorem three_digit_self_repeating_powers : 
  {N : ℕ | 100 ≤ N ∧ N < 1000 ∧ ∀ k : ℕ, k ≥ 1 → N^k % 1000 = N} = {376, 625} := by
  sorry

end NUMINAMATH_CALUDE_three_digit_self_repeating_powers_l3373_337396
