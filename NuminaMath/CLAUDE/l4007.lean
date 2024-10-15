import Mathlib

namespace NUMINAMATH_CALUDE_notebook_buyers_difference_l4007_400794

theorem notebook_buyers_difference (notebook_cost : ℕ) 
  (fifth_grade_total : ℕ) (fourth_grade_total : ℕ) 
  (fourth_grade_count : ℕ) :
  notebook_cost > 0 ∧ 
  notebook_cost * 100 ∣ fifth_grade_total ∧ 
  notebook_cost * 100 ∣ fourth_grade_total ∧
  fifth_grade_total = 210 ∧
  fourth_grade_total = 252 ∧
  fourth_grade_count = 28 ∧
  fourth_grade_count ≥ fourth_grade_total / (notebook_cost * 100) →
  (fourth_grade_total / (notebook_cost * 100)) - 
  (fifth_grade_total / (notebook_cost * 100)) = 2 :=
sorry

end NUMINAMATH_CALUDE_notebook_buyers_difference_l4007_400794


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l4007_400755

/-- Proves that in an isosceles triangle where one angle is 40% larger than a right angle,
    the measure of one of the two smallest angles is 27°. -/
theorem isosceles_triangle_angle_measure :
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One angle is 40% larger than a right angle (90°)
  c = 90 + 0.4 * 90 →
  -- One of the two smallest angles measures 27°
  a = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l4007_400755


namespace NUMINAMATH_CALUDE_angle_inequality_l4007_400718

theorem angle_inequality (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) := by
sorry

end NUMINAMATH_CALUDE_angle_inequality_l4007_400718


namespace NUMINAMATH_CALUDE_john_total_distance_l4007_400763

-- Define the driving segments
def segment1_speed : ℝ := 55
def segment1_time : ℝ := 2.5
def segment2_speed : ℝ := 65
def segment2_time : ℝ := 3.25
def segment3_speed : ℝ := 50
def segment3_time : ℝ := 4

-- Define the total distance function
def total_distance : ℝ :=
  segment1_speed * segment1_time +
  segment2_speed * segment2_time +
  segment3_speed * segment3_time

-- Theorem statement
theorem john_total_distance :
  total_distance = 548.75 := by
  sorry

end NUMINAMATH_CALUDE_john_total_distance_l4007_400763


namespace NUMINAMATH_CALUDE_cost_price_correct_l4007_400732

/-- The cost price of a product satisfying given conditions -/
def cost_price : ℝ := 90

/-- The marked price of the product -/
def marked_price : ℝ := 120

/-- The discount rate applied to the product -/
def discount_rate : ℝ := 0.1

/-- The profit rate relative to the cost price -/
def profit_rate : ℝ := 0.2

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct : 
  cost_price * (1 + profit_rate) = marked_price * (1 - discount_rate) := by
  sorry

#eval cost_price -- Should output 90

end NUMINAMATH_CALUDE_cost_price_correct_l4007_400732


namespace NUMINAMATH_CALUDE_haley_garden_problem_l4007_400782

def seeds_in_big_garden (total_seeds small_gardens seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - small_gardens * seeds_per_small_garden

theorem haley_garden_problem (total_seeds small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : small_gardens = 7)
  (h3 : seeds_per_small_garden = 3) :
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 := by
  sorry

end NUMINAMATH_CALUDE_haley_garden_problem_l4007_400782


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_25_l4007_400750

theorem smallest_divisible_by_18_and_25 : Nat.lcm 18 25 = 450 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_25_l4007_400750


namespace NUMINAMATH_CALUDE_count_even_factors_l4007_400771

def n : ℕ := 2^3 * 3^2 * 5

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 18 :=
sorry

end NUMINAMATH_CALUDE_count_even_factors_l4007_400771


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4007_400724

theorem complex_fraction_simplification :
  2017 * (2016 / 2017) / (2019 * (1 / 2016)) + 1 / 2017 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4007_400724


namespace NUMINAMATH_CALUDE_only_statement1_is_true_l4007_400700

-- Define the statements
def statement1 (a x y : ℝ) : Prop := a * (x - y) = a * x - a * y
def statement2 (a x y : ℝ) : Prop := a^(x - y) = a^x - a^y
def statement3 (x y : ℝ) : Prop := x > y ∧ y > 0 → Real.log (x - y) = Real.log x - Real.log y
def statement4 (x y : ℝ) : Prop := x > 0 ∧ y > 0 → Real.log x / Real.log y = Real.log x - Real.log y
def statement5 (a x y : ℝ) : Prop := a * (x * y) = (a * x) * (a * y)

-- Theorem stating that only statement1 is true among all statements
theorem only_statement1_is_true :
  (∀ a x y : ℝ, statement1 a x y) ∧
  (∃ a x y : ℝ, ¬ statement2 a x y) ∧
  (∃ x y : ℝ, ¬ statement3 x y) ∧
  (∃ x y : ℝ, ¬ statement4 x y) ∧
  (∃ a x y : ℝ, ¬ statement5 a x y) :=
sorry

end NUMINAMATH_CALUDE_only_statement1_is_true_l4007_400700


namespace NUMINAMATH_CALUDE_abs_equality_l4007_400744

theorem abs_equality (x : ℝ) : 
  (|x| = Real.sqrt (x^2)) ∧ 
  (|x| = if x ≥ 0 then x else -x) := by sorry

end NUMINAMATH_CALUDE_abs_equality_l4007_400744


namespace NUMINAMATH_CALUDE_system_solution_l4007_400762

/-- The system of equations:
    1. 3x² - xy = 1
    2. 9xy + y² = 22
    has exactly four solutions: (1,2), (-1,-2), (-1/6, 5.5), and (1/6, -5.5) -/
theorem system_solution :
  let f (x y : ℝ) := 3 * x^2 - x * y - 1
  let g (x y : ℝ) := 9 * x * y + y^2 - 22
  ∀ x y : ℝ, f x y = 0 ∧ g x y = 0 ↔
    (x = 1 ∧ y = 2) ∨
    (x = -1 ∧ y = -2) ∨
    (x = -1/6 ∧ y = 11/2) ∨
    (x = 1/6 ∧ y = -11/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4007_400762


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l4007_400751

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = 2 * Real.sqrt 3) :
  let side : ℝ := 2 * h / Real.sqrt 3
  let area : ℝ := 1/2 * side * h
  area = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l4007_400751


namespace NUMINAMATH_CALUDE_star_calculation_l4007_400710

-- Define the ☆ operation for rational numbers
def star (a b : ℚ) : ℚ := 2 * a - b + 1

-- Theorem statement
theorem star_calculation : star 1 (star 2 (-3)) = -5 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l4007_400710


namespace NUMINAMATH_CALUDE_scientific_notation_of_1340000000_l4007_400758

theorem scientific_notation_of_1340000000 :
  ∃ (a : ℝ) (n : ℤ), 1340000000 = a * (10 : ℝ)^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.34 ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1340000000_l4007_400758


namespace NUMINAMATH_CALUDE_tangent_line_parabola_l4007_400734

/-- The equation of the tangent line to the parabola y = x^2 that is parallel to the line y = 2x is 2x - y - 1 = 0 -/
theorem tangent_line_parabola (x y : ℝ) : 
  (y = x^2) →  -- parabola equation
  (∃ m : ℝ, m = 2) →  -- parallel to y = 2x
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ 
    ∀ x₀ y₀ : ℝ, y₀ = x₀^2 → (y₀ - (x₀^2) = m * (x - x₀))) →  -- tangent line equation
  (2 * x - y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_l4007_400734


namespace NUMINAMATH_CALUDE_parabola_vertex_l4007_400792

/-- A parabola defined by y = x^2 - 2ax + b passing through (1, 1) and intersecting the x-axis at only one point -/
structure Parabola where
  a : ℝ
  b : ℝ
  point_condition : 1 = 1^2 - 2*a*1 + b
  single_intersection : ∃! x, x^2 - 2*a*x + b = 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.a, p.a^2 - p.b)

theorem parabola_vertex (p : Parabola) : vertex p = (0, 0) ∨ vertex p = (2, 0) := by
  sorry


end NUMINAMATH_CALUDE_parabola_vertex_l4007_400792


namespace NUMINAMATH_CALUDE_cora_reading_schedule_l4007_400786

/-- The number of pages Cora needs to read on Thursday to finish her book -/
def pages_to_read_thursday (total_pages : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) (pages_wednesday : ℕ) : ℕ :=
  let pages_thursday := (total_pages - pages_monday - pages_tuesday - pages_wednesday) / 3
  pages_thursday

theorem cora_reading_schedule :
  pages_to_read_thursday 158 23 38 61 = 12 := by
  sorry

#eval pages_to_read_thursday 158 23 38 61

end NUMINAMATH_CALUDE_cora_reading_schedule_l4007_400786


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l4007_400737

theorem existence_of_special_integers : 
  ∃ (a b : ℕ+), 
    (¬ (7 ∣ a.val)) ∧ 
    (¬ (7 ∣ b.val)) ∧ 
    (¬ (7 ∣ (a.val + b.val))) ∧ 
    (7^7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l4007_400737


namespace NUMINAMATH_CALUDE_lines_theorem_l4007_400715

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2*x + 3*y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₃ (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 2*x + y - 3 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

theorem lines_theorem :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →  -- P exists as intersection of l₁ and l₂
  (parallel_line (P.1) (P.2)) ∧    -- Parallel line passes through P
  (∀ (x y : ℝ), parallel_line x y → (∃ (k : ℝ), y - P.2 = k * (x - P.1) ∧ y = -2*x + 5)) ∧  -- Parallel line is parallel to l₃
  (perpendicular_line (P.1) (P.2)) ∧  -- Perpendicular line passes through P
  (∀ (x y : ℝ), perpendicular_line x y → 
    (∃ (k₁ k₂ : ℝ), y - P.2 = k₁ * (x - P.1) ∧ y = k₂ * x - 5/2 ∧ k₁ * k₂ = -1)) -- Perpendicular line is perpendicular to l₃
  := by sorry

end NUMINAMATH_CALUDE_lines_theorem_l4007_400715


namespace NUMINAMATH_CALUDE_sin_arctan_reciprocal_square_l4007_400790

theorem sin_arctan_reciprocal_square (x : ℝ) (h_pos : x > 0) (h_eq : Real.sin (Real.arctan x) = 1 / x) : x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_arctan_reciprocal_square_l4007_400790


namespace NUMINAMATH_CALUDE_max_value_inequality_l4007_400789

theorem max_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^3 * (b + c)^3) ≤ 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l4007_400789


namespace NUMINAMATH_CALUDE_social_practice_problem_l4007_400779

/-- Represents the number of students -/
def num_students : ℕ := sorry

/-- Represents the number of 35-seat buses needed to exactly fit all students -/
def num_35_seat_buses : ℕ := sorry

/-- Represents the number of 55-seat buses needed -/
def num_55_seat_buses : ℕ := sorry

/-- Cost of renting a 35-seat bus -/
def cost_35_seat : ℕ := 320

/-- Cost of renting a 55-seat bus -/
def cost_55_seat : ℕ := 400

/-- Total number of buses to rent -/
def total_buses : ℕ := 4

/-- Maximum budget for bus rental -/
def max_budget : ℕ := 1500

/-- Theorem stating the conditions and the result to be proven -/
theorem social_practice_problem :
  num_students = 35 * num_35_seat_buses ∧
  num_students = 55 * num_55_seat_buses - 45 ∧
  num_55_seat_buses = num_35_seat_buses - 1 ∧
  num_students = 175 ∧
  ∃ (x y : ℕ), x + y = total_buses ∧
               x * cost_35_seat + y * cost_55_seat ≤ max_budget ∧
               x * cost_35_seat + y * cost_55_seat = 1440 :=
by sorry

end NUMINAMATH_CALUDE_social_practice_problem_l4007_400779


namespace NUMINAMATH_CALUDE_weeding_rate_calculation_l4007_400702

/-- The hourly rate for mowing lawns -/
def mowing_rate : ℝ := 4

/-- The number of hours spent mowing lawns in September -/
def mowing_hours : ℝ := 25

/-- The number of hours spent pulling weeds in September -/
def weeding_hours : ℝ := 3

/-- The total earnings for September and October -/
def total_earnings : ℝ := 248

/-- The hourly rate for pulling weeds -/
def weeding_rate : ℝ := 8

theorem weeding_rate_calculation :
  2 * (mowing_rate * mowing_hours + weeding_rate * weeding_hours) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_weeding_rate_calculation_l4007_400702


namespace NUMINAMATH_CALUDE_james_steak_purchase_l4007_400760

/-- Represents the buy one get one free deal -/
def buyOneGetOneFree (x : ℝ) : ℝ := 2 * x

/-- Represents the price per pound in dollars -/
def pricePerPound : ℝ := 15

/-- Represents the total amount James paid in dollars -/
def totalPaid : ℝ := 150

/-- Theorem stating that James bought 20 pounds of steaks -/
theorem james_steak_purchase :
  ∃ (x : ℝ), x > 0 ∧ x * pricePerPound = totalPaid ∧ buyOneGetOneFree x = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_james_steak_purchase_l4007_400760


namespace NUMINAMATH_CALUDE_line_equation_l4007_400711

/-- Proves that the line represented by the given parametric equations has the equation y = 2x - 4 -/
theorem line_equation (t : ℝ) :
  let x := 3 * t + 1
  let y := 6 * t - 2
  y = 2 * x - 4 := by sorry

end NUMINAMATH_CALUDE_line_equation_l4007_400711


namespace NUMINAMATH_CALUDE_sams_test_score_l4007_400722

theorem sams_test_score (initial_students : ℕ) (initial_average : ℚ) (new_average : ℚ) 
  (h1 : initial_students = 19)
  (h2 : initial_average = 85)
  (h3 : new_average = 86) :
  (initial_students + 1) * new_average - initial_students * initial_average = 105 :=
by sorry

end NUMINAMATH_CALUDE_sams_test_score_l4007_400722


namespace NUMINAMATH_CALUDE_total_people_shook_hands_l4007_400731

/-- The number of schools participating in the debate -/
def num_schools : ℕ := 5

/-- The number of students in the fourth school -/
def students_fourth : ℕ := 150

/-- The number of faculty members per school -/
def faculty_per_school : ℕ := 10

/-- The number of event staff per school -/
def event_staff_per_school : ℕ := 5

/-- Calculate the number of students in the third school -/
def students_third : ℕ := (3 * students_fourth) / 2

/-- Calculate the number of students in the second school -/
def students_second : ℕ := students_third + 50

/-- Calculate the number of students in the first school -/
def students_first : ℕ := 2 * students_second

/-- Calculate the number of students in the fifth school -/
def students_fifth : ℕ := students_fourth - 120

/-- Calculate the total number of students -/
def total_students : ℕ := students_first + students_second + students_third + students_fourth + students_fifth

/-- Calculate the total number of faculty and staff -/
def total_faculty_staff : ℕ := num_schools * (faculty_per_school + event_staff_per_school)

/-- The theorem to prove -/
theorem total_people_shook_hands : total_students + total_faculty_staff = 1305 := by
  sorry

end NUMINAMATH_CALUDE_total_people_shook_hands_l4007_400731


namespace NUMINAMATH_CALUDE_inscribed_hexagon_diagonal_sum_l4007_400756

/-- A hexagon inscribed in a circle with five sides of length 90 and one side of length 36 -/
structure InscribedHexagon where
  /-- The length of five sides of the hexagon -/
  regularSideLength : ℝ
  /-- The length of the sixth side of the hexagon -/
  irregularSideLength : ℝ
  /-- The hexagon is inscribed in a circle -/
  inscribed : Bool
  /-- Five sides have the same length -/
  fiveSidesEqual : regularSideLength = 90
  /-- The sixth side has a different length -/
  sixthSideDifferent : irregularSideLength = 36
  /-- The hexagon is actually inscribed in a circle -/
  isInscribed : inscribed = true

/-- The sum of the lengths of the three diagonals drawn from one vertex of the hexagon -/
def diagonalSum (h : InscribedHexagon) : ℝ := 428.4

/-- Theorem: The sum of the lengths of the three diagonals drawn from one vertex
    of the inscribed hexagon with the given properties is 428.4 -/
theorem inscribed_hexagon_diagonal_sum (h : InscribedHexagon) :
  diagonalSum h = 428.4 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_diagonal_sum_l4007_400756


namespace NUMINAMATH_CALUDE_polynomial_equality_l4007_400726

theorem polynomial_equality : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4007_400726


namespace NUMINAMATH_CALUDE_simplify_2M_minus_N_value_at_specific_points_independence_condition_l4007_400765

-- Define the polynomials M and N
def M (x y : ℝ) : ℝ := x^2 + x*y + 2*y - 2
def N (x y : ℝ) : ℝ := 2*x^2 - 2*x*y + x - 4

-- Theorem 1: Simplification of 2M - N
theorem simplify_2M_minus_N (x y : ℝ) :
  2 * M x y - N x y = 4*x*y + 4*y - x :=
sorry

-- Theorem 2: Value of 2M - N when x = -2 and y = -4
theorem value_at_specific_points :
  2 * M (-2) (-4) - N (-2) (-4) = 18 :=
sorry

-- Theorem 3: Condition for 2M - N to be independent of x
theorem independence_condition (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, 2 * M x y - N x y = c) ↔ y = 1/4 :=
sorry

end NUMINAMATH_CALUDE_simplify_2M_minus_N_value_at_specific_points_independence_condition_l4007_400765


namespace NUMINAMATH_CALUDE_square_sum_divided_l4007_400738

theorem square_sum_divided : (2005^2 + 2 * 2005 * 1995 + 1995^2) / 800 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_divided_l4007_400738


namespace NUMINAMATH_CALUDE_sequence_problem_l4007_400759

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  a 4 = 16 →
  arithmetic_sequence b →
  a 3 = b 3 →
  a 5 = b 5 →
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ n : ℕ, b n = 12*n - 28) ∧
  (∀ n : ℕ, S n = (3*n - 10) * 2^(n+3) - 80) :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l4007_400759


namespace NUMINAMATH_CALUDE_color_assignment_count_l4007_400742

theorem color_assignment_count : ∀ (n m : ℕ), n = 5 ∧ m = 3 →
  (n * (n - 1) * (n - 2)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_color_assignment_count_l4007_400742


namespace NUMINAMATH_CALUDE_length_width_ratio_l4007_400798

-- Define the rectangle
def rectangle (width : ℝ) (length : ℝ) : Prop :=
  width > 0 ∧ length > 0

-- Define the area of the rectangle
def area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

-- Theorem statement
theorem length_width_ratio (width : ℝ) (length : ℝ) :
  rectangle width length →
  width = 6 →
  area width length = 108 →
  length / width = 3 := by
  sorry


end NUMINAMATH_CALUDE_length_width_ratio_l4007_400798


namespace NUMINAMATH_CALUDE_min_buses_second_group_l4007_400780

theorem min_buses_second_group 
  (total_students : ℕ) 
  (bus_capacity : ℕ) 
  (max_buses_first_group : ℕ) 
  (min_buses_second_group : ℕ) : 
  total_students = 550 → 
  bus_capacity = 45 → 
  max_buses_first_group = 8 → 
  min_buses_second_group = 5 → 
  (max_buses_first_group * bus_capacity + min_buses_second_group * bus_capacity ≥ total_students) ∧
  ((min_buses_second_group - 1) * bus_capacity < total_students - max_buses_first_group * bus_capacity) :=
by
  sorry

#check min_buses_second_group

end NUMINAMATH_CALUDE_min_buses_second_group_l4007_400780


namespace NUMINAMATH_CALUDE_special_numbers_count_l4007_400704

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Counts the number of two-digit integers x for which digit_sum(digit_sum(x)) = 4 -/
def count_special_numbers : ℕ := sorry

theorem special_numbers_count : count_special_numbers = 10 := by sorry

end NUMINAMATH_CALUDE_special_numbers_count_l4007_400704


namespace NUMINAMATH_CALUDE_simplify_power_sum_l4007_400733

theorem simplify_power_sum : (-2)^2003 + 2^2004 + (-2)^2005 - 2^2006 = 5 * 2^2003 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_sum_l4007_400733


namespace NUMINAMATH_CALUDE_correct_change_l4007_400747

/-- The change Bomi should receive after buying candy and chocolate -/
def bomi_change (candy_cost chocolate_cost paid : ℕ) : ℕ :=
  paid - (candy_cost + chocolate_cost)

/-- Theorem stating the correct change Bomi should receive -/
theorem correct_change : bomi_change 350 500 1000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l4007_400747


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l4007_400774

open Complex

theorem complex_exponential_sum (α β γ : ℝ) :
  exp (I * α) + exp (I * β) + exp (I * γ) = 1 + I →
  exp (-I * α) + exp (-I * β) + exp (-I * γ) = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l4007_400774


namespace NUMINAMATH_CALUDE_leo_current_weight_l4007_400721

def leo_weight_problem (leo_weight kendra_weight : ℝ) : Prop :=
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 150)

theorem leo_current_weight :
  ∃ (leo_weight kendra_weight : ℝ),
    leo_weight_problem leo_weight kendra_weight ∧
    leo_weight = 86 := by
  sorry

end NUMINAMATH_CALUDE_leo_current_weight_l4007_400721


namespace NUMINAMATH_CALUDE_line_parameterization_l4007_400770

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = (2/3) * x + 5

-- Define the parameterization
def parameterization (x y s l t : ℝ) : Prop :=
  (x = -3 + t * l) ∧ (y = s - 6 * t)

-- Theorem statement
theorem line_parameterization (s l : ℝ) :
  (∀ x y t : ℝ, line_equation x y ↔ parameterization x y s l t) →
  s = 3 ∧ l = -9 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l4007_400770


namespace NUMINAMATH_CALUDE_infimum_attained_by_uniform_distribution_l4007_400757

-- Define the set of Borel functions
def BorelFunction (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being an increasing function
def Increasing (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- Define a random variable
def RandomVariable (X : ℝ → ℝ) : Prop := sorry

-- Define the property of density not exceeding 1/2
def DensityNotExceedingHalf (X : ℝ → ℝ) : Prop := sorry

-- Define uniform distribution on [-1, 1]
def UniformDistributionOnUnitInterval (U : ℝ → ℝ) : Prop := sorry

-- Define expected value
def ExpectedValue (f : ℝ → ℝ) (X : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem infimum_attained_by_uniform_distribution
  (f : ℝ → ℝ) (X U : ℝ → ℝ) :
  BorelFunction f →
  Increasing f →
  RandomVariable X →
  DensityNotExceedingHalf X →
  UniformDistributionOnUnitInterval U →
  ExpectedValue (fun x => f (abs x)) X ≥ ExpectedValue (fun x => f (abs x)) U :=
sorry

end NUMINAMATH_CALUDE_infimum_attained_by_uniform_distribution_l4007_400757


namespace NUMINAMATH_CALUDE_amanda_lost_notebooks_l4007_400720

/-- The number of notebooks Amanda lost -/
def notebooks_lost (initial : ℕ) (ordered : ℕ) (current : ℕ) : ℕ :=
  initial + ordered - current

theorem amanda_lost_notebooks : notebooks_lost 10 6 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_amanda_lost_notebooks_l4007_400720


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l4007_400767

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

def box : Dimensions := ⟨3, 4, 3⟩
def block : Dimensions := ⟨3, 1, 1⟩

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := volume box / volume block

theorem max_blocks_in_box : max_blocks = 12 := by sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l4007_400767


namespace NUMINAMATH_CALUDE_unique_modular_solution_l4007_400783

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [ZMOD 11] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l4007_400783


namespace NUMINAMATH_CALUDE_quadratic_roots_positive_conditions_l4007_400730

theorem quadratic_roots_positive_conditions (a b c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) →
  (b^2 - 4*a*c ≥ 0 ∧ a*c > 0 ∧ a*b < 0) ∧
  ¬(b^2 - 4*a*c ≥ 0 ∧ a*c > 0 ∧ a*b < 0 → 
    ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_positive_conditions_l4007_400730


namespace NUMINAMATH_CALUDE_hex_multiplication_l4007_400705

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a hexadecimal digit to its decimal value --/
def hex_to_dec (d : HexDigit) : Nat :=
  match d with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Represents a two-digit hexadecimal number --/
structure HexNumber :=
  (msb : HexDigit)
  (lsb : HexDigit)

/-- Converts a two-digit hexadecimal number to its decimal value --/
def hex_number_to_dec (h : HexNumber) : Nat :=
  16 * (hex_to_dec h.msb) + (hex_to_dec h.lsb)

/-- The main theorem to prove --/
theorem hex_multiplication :
  let a := HexNumber.mk HexDigit.A HexDigit.A
  let b := HexNumber.mk HexDigit.B HexDigit.B
  let result := HexNumber.mk HexDigit.D6 HexDigit.E
  hex_number_to_dec a * hex_number_to_dec b = hex_number_to_dec result := by
  sorry

end NUMINAMATH_CALUDE_hex_multiplication_l4007_400705


namespace NUMINAMATH_CALUDE_book_sale_profit_l4007_400708

/-- Represents the profit calculation for a book sale with and without discount -/
theorem book_sale_profit (cost_price : ℝ) (discount_percent : ℝ) (profit_with_discount_percent : ℝ) :
  discount_percent = 5 →
  profit_with_discount_percent = 23.5 →
  let selling_price_with_discount := cost_price * (1 + profit_with_discount_percent / 100 - discount_percent / 100)
  let selling_price_without_discount := selling_price_with_discount + cost_price * (discount_percent / 100)
  let profit_without_discount_percent := (selling_price_without_discount - cost_price) / cost_price * 100
  profit_without_discount_percent = 23.5 :=
by sorry

end NUMINAMATH_CALUDE_book_sale_profit_l4007_400708


namespace NUMINAMATH_CALUDE_prob_non_expired_single_draw_prob_expired_two_draws_l4007_400776

/-- Represents the total number of bottles --/
def total_bottles : ℕ := 6

/-- Represents the number of expired bottles --/
def expired_bottles : ℕ := 2

/-- Represents the number of non-expired bottles --/
def non_expired_bottles : ℕ := total_bottles - expired_bottles

/-- Theorem for the probability of drawing a non-expired bottle in a single draw --/
theorem prob_non_expired_single_draw : 
  (non_expired_bottles : ℚ) / total_bottles = 2 / 3 := by sorry

/-- Theorem for the probability of drawing at least one expired bottle in two draws --/
theorem prob_expired_two_draws : 
  1 - (non_expired_bottles * (non_expired_bottles - 1) : ℚ) / (total_bottles * (total_bottles - 1)) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_non_expired_single_draw_prob_expired_two_draws_l4007_400776


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4007_400775

theorem sum_of_fractions : (1 : ℚ) / 3 + (1 : ℚ) / 4 = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4007_400775


namespace NUMINAMATH_CALUDE_find_b_l4007_400743

def is_valid_set (x b : ℕ) : Prop :=
  x > 0 ∧ x + 2 > 0 ∧ x + b > 0 ∧ x + 7 > 0 ∧ x + 32 > 0

def median (x b : ℕ) : ℚ := x + b

def mean (x b : ℕ) : ℚ := (x + (x + 2) + (x + b) + (x + 7) + (x + 32)) / 5

theorem find_b (x : ℕ) :
  ∃ b : ℕ, is_valid_set x b ∧ mean x b = median x b + 5 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l4007_400743


namespace NUMINAMATH_CALUDE_chefs_flour_calculation_l4007_400712

theorem chefs_flour_calculation (recipe_ratio : ℚ) (eggs_needed : ℕ) (flour_used : ℚ) : 
  recipe_ratio = 7 / 2 →
  eggs_needed = 28 →
  flour_used = eggs_needed / recipe_ratio →
  flour_used = 8 := by
sorry

end NUMINAMATH_CALUDE_chefs_flour_calculation_l4007_400712


namespace NUMINAMATH_CALUDE_malcom_brandon_card_difference_l4007_400735

theorem malcom_brandon_card_difference :
  ∀ (brandon_cards malcom_cards_initial malcom_cards_after : ℕ),
    brandon_cards = 20 →
    malcom_cards_initial > brandon_cards →
    malcom_cards_after = 14 →
    malcom_cards_after * 2 = malcom_cards_initial →
    malcom_cards_initial - brandon_cards = 8 :=
by sorry

end NUMINAMATH_CALUDE_malcom_brandon_card_difference_l4007_400735


namespace NUMINAMATH_CALUDE_intersection_midpoint_l4007_400746

/-- The midpoint of the line segment connecting the intersection points of y = x and y^2 = 4x is (2,2) -/
theorem intersection_midpoint :
  let line := {(x, y) : ℝ × ℝ | y = x}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}
  let intersection := line ∩ parabola
  ∃ (a b : ℝ × ℝ), a ∈ intersection ∧ b ∈ intersection ∧ a ≠ b ∧
    (a.1 + b.1) / 2 = 2 ∧ (a.2 + b.2) / 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_midpoint_l4007_400746


namespace NUMINAMATH_CALUDE_floor_length_is_sqrt_150_l4007_400753

/-- Represents a rectangular floor with specific properties -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  total_paint_cost : ℝ
  paint_rate_per_sqm : ℝ

/-- The length is 200% more than the breadth -/
def length_breadth_relation (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The total paint cost divided by the rate per sqm gives the area -/
def area_from_paint_cost (floor : RectangularFloor) : Prop :=
  floor.total_paint_cost / floor.paint_rate_per_sqm = floor.length * floor.breadth

/-- Theorem stating the length of the floor -/
theorem floor_length_is_sqrt_150 (floor : RectangularFloor) 
  (h1 : length_breadth_relation floor)
  (h2 : area_from_paint_cost floor)
  (h3 : floor.total_paint_cost = 100)
  (h4 : floor.paint_rate_per_sqm = 2) : 
  floor.length = Real.sqrt 150 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_is_sqrt_150_l4007_400753


namespace NUMINAMATH_CALUDE_journey_average_speed_l4007_400778

/-- Calculates the average speed of a journey with two segments -/
def average_speed (speed1 : ℝ) (time1_fraction : ℝ) (speed2 : ℝ) (time2_fraction : ℝ) : ℝ :=
  speed1 * time1_fraction + speed2 * time2_fraction

theorem journey_average_speed :
  let speed1 := 10
  let speed2 := 50
  let time1_fraction := 0.25
  let time2_fraction := 0.75
  average_speed speed1 time1_fraction speed2 time2_fraction = 40 := by
sorry

end NUMINAMATH_CALUDE_journey_average_speed_l4007_400778


namespace NUMINAMATH_CALUDE_katie_game_difference_l4007_400761

theorem katie_game_difference (katie_games friends_games : ℕ) 
  (h1 : katie_games = 81) (h2 : friends_games = 59) : 
  katie_games - friends_games = 22 := by
sorry

end NUMINAMATH_CALUDE_katie_game_difference_l4007_400761


namespace NUMINAMATH_CALUDE_expression_evaluation_l4007_400727

theorem expression_evaluation :
  let a : ℚ := -1/3
  let b : ℚ := -3
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4007_400727


namespace NUMINAMATH_CALUDE_price_calculation_equivalence_l4007_400728

theorem price_calculation_equivalence 
  (initial_price tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) 
  (discount_rate_pos : 0 < discount_rate) 
  (tax_rate_bound : tax_rate < 1) 
  (discount_rate_bound : discount_rate < 1) :
  initial_price * (1 + tax_rate) * (1 - discount_rate) = 
  initial_price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_price_calculation_equivalence_l4007_400728


namespace NUMINAMATH_CALUDE_expression_evaluation_l4007_400781

theorem expression_evaluation : 
  let cos_45 : ℝ := Real.sqrt 2 / 2
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (cos_45 - 3)) = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4007_400781


namespace NUMINAMATH_CALUDE_three_color_circle_existence_l4007_400703

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- State the theorem
theorem three_color_circle_existence 
  (coloring : Coloring) 
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c) :
  ∃ circ : Circle, ∀ c : Color, ∃ p : Point, 
    coloring p = c ∧ (p.x - circ.center.x)^2 + (p.y - circ.center.y)^2 ≤ circ.radius^2 :=
sorry

end NUMINAMATH_CALUDE_three_color_circle_existence_l4007_400703


namespace NUMINAMATH_CALUDE_bisection_method_theorem_l4007_400768

/-- The bisection method theorem -/
theorem bisection_method_theorem 
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_continuous : Continuous f) 
  (h_unique_zero : ∃! x, x ∈ Set.Ioo a b ∧ f x = 0) 
  (h_interval : b - a = 0.1) :
  ∃ n : ℕ, n ≤ 10 ∧ (0.1 / 2^n : ℝ) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_bisection_method_theorem_l4007_400768


namespace NUMINAMATH_CALUDE_expression_simplification_l4007_400796

theorem expression_simplification (x : ℝ) (h : x = 5) :
  (2 / (x^2 - 2*x) - (x - 6) / (x^2 - 4*x + 4) / ((x - 6) / (x - 2))) = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l4007_400796


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l4007_400740

theorem cards_given_to_jeff (initial_cards : ℕ) (cards_to_john : ℕ) (cards_left : ℕ) :
  initial_cards = 573 →
  cards_to_john = 195 →
  cards_left = 210 →
  initial_cards - cards_to_john - cards_left = 168 :=
by sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l4007_400740


namespace NUMINAMATH_CALUDE_shortest_path_on_parallelepiped_l4007_400764

/-- The shortest path on the surface of a rectangular parallelepiped -/
theorem shortest_path_on_parallelepiped (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let surface_paths := [
    Real.sqrt ((a + c + a)^2 + b^2),
    Real.sqrt ((a + b + a)^2 + c^2),
    Real.sqrt ((b + a + b)^2 + c^2)
  ]
  ∃ (path : ℝ), path ∈ surface_paths ∧ path = Real.sqrt 125 ∧ ∀ x ∈ surface_paths, path ≤ x :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_on_parallelepiped_l4007_400764


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l4007_400799

open Real

theorem largest_n_for_sin_cos_inequality :
  ∃ (n : ℕ), n = 3 ∧
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → sin x ^ n + cos x ^ n > 1 / 2) ∧
  ¬(∀ x : ℝ, 0 < x ∧ x < π / 2 → sin x ^ (n + 1) + cos x ^ (n + 1) > 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l4007_400799


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l4007_400741

theorem sine_cosine_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < π / 2) (h3 : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l4007_400741


namespace NUMINAMATH_CALUDE_line_length_difference_l4007_400788

/-- Conversion rate from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Length of the white line in inches -/
def white_line_inch : ℝ := 7.666666666666667

/-- Length of the blue line in inches -/
def blue_line_inch : ℝ := 3.3333333333333335

/-- Converts a length from inches to centimeters -/
def to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

/-- The difference in length between the white and blue lines in centimeters -/
theorem line_length_difference : 
  to_cm white_line_inch - to_cm blue_line_inch = 11.005555555555553 := by
  sorry

end NUMINAMATH_CALUDE_line_length_difference_l4007_400788


namespace NUMINAMATH_CALUDE_game_draw_probability_l4007_400797

theorem game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3) 
  (h_not_lose : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_game_draw_probability_l4007_400797


namespace NUMINAMATH_CALUDE_number_difference_l4007_400707

theorem number_difference (L S : ℝ) (h1 : L = 1650) (h2 : L = 6 * S + 15) : 
  L - S = 1377.5 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l4007_400707


namespace NUMINAMATH_CALUDE_factorial_ratio_l4007_400716

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l4007_400716


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4007_400729

theorem max_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (2 * a) / (a^2 + b) + b / (a + b^2) ≤ (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4007_400729


namespace NUMINAMATH_CALUDE_twins_age_product_difference_l4007_400785

theorem twins_age_product_difference (current_age : ℕ) (h : current_age = 8) : 
  (current_age + 1) * (current_age + 1) - current_age * current_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_product_difference_l4007_400785


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_l4007_400752

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-1, 4)  -- Derived from a + b = (0, 2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := vector_a
  let b : ℝ × ℝ := vector_b
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_2b_l4007_400752


namespace NUMINAMATH_CALUDE_negation_equivalence_l4007_400739

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4007_400739


namespace NUMINAMATH_CALUDE_function_shape_is_graph_l4007_400784

/-- A function from real numbers to real numbers -/
def RealFunction := ℝ → ℝ

/-- A point in the Cartesian coordinate system -/
def CartesianPoint := ℝ × ℝ

/-- The set of all points representing a function in the Cartesian coordinate system -/
def FunctionPoints (f : RealFunction) : Set CartesianPoint :=
  {p : CartesianPoint | ∃ x : ℝ, p = (x, f x)}

/-- The graph of a function is the set of all points representing that function -/
def Graph (f : RealFunction) : Set CartesianPoint := FunctionPoints f

/-- Theorem: The shape formed by all points plotted in the Cartesian coordinate system 
    that represent a function is called the graph of the function -/
theorem function_shape_is_graph (f : RealFunction) : 
  FunctionPoints f = Graph f := by sorry

end NUMINAMATH_CALUDE_function_shape_is_graph_l4007_400784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4007_400719

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_fifth : a 5 = 8)
  (h_sum : a 1 + a 2 + a 3 = 6) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4007_400719


namespace NUMINAMATH_CALUDE_division_problem_l4007_400795

theorem division_problem (x y : ℤ) (hx : x > 0) : 
  (∃ q : ℤ, x = 11 * y + 4 ∧ q * 11 + 4 = x) →
  (∃ q : ℤ, 2 * x = 6 * (3 * y) + 1 ∧ q * 6 + 1 = 2 * x) →
  7 * y - x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4007_400795


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4007_400717

/-- An isosceles triangle with two sides of 7 cm each and a perimeter of 23 cm has a base of 9 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
    base > 0 → 
    7 + 7 + base = 23 → 
    base = 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4007_400717


namespace NUMINAMATH_CALUDE_peggy_initial_dolls_l4007_400713

theorem peggy_initial_dolls :
  ∀ (initial : ℕ) (grandmother_gift : ℕ) (birthday_christmas : ℕ),
    grandmother_gift = 30 →
    birthday_christmas = grandmother_gift / 2 →
    initial + grandmother_gift + birthday_christmas = 51 →
    initial = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_peggy_initial_dolls_l4007_400713


namespace NUMINAMATH_CALUDE_greatest_sum_of_digits_l4007_400709

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the sum of digits for a given time -/
def sumOfDigits (t : Time) : Nat :=
  (t.hours / 10) + (t.hours % 10) + (t.minutes / 10) + (t.minutes % 10)

/-- States that 19:59 has the greatest sum of digits among all possible times -/
theorem greatest_sum_of_digits :
  ∀ t : Time, sumOfDigits t ≤ sumOfDigits ⟨19, 59, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_of_digits_l4007_400709


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_six_l4007_400736

/-- A three-digit number in the form 2a3 -/
def number_2a3 (a : ℕ) : ℕ := 200 + 10 * a + 3

/-- A three-digit number in the form 5b9 -/
def number_5b9 (b : ℕ) : ℕ := 500 + 10 * b + 9

/-- Proposition: If 2a3 + 326 = 5b9 and 5b9 is a multiple of 9, then a + b = 6 -/
theorem sum_of_a_and_b_is_six (a b : ℕ) :
  number_2a3 a + 326 = number_5b9 b →
  (∃ k : ℕ, number_5b9 b = 9 * k) →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_six_l4007_400736


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l4007_400749

theorem subtraction_of_decimals : (25.50 : ℝ) - 3.245 = 22.255 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l4007_400749


namespace NUMINAMATH_CALUDE_complex_number_properties_l4007_400766

theorem complex_number_properties (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z := x + Complex.I * y
  (0 < z.re ∧ 0 < z.im) ∧ Complex.abs z = Real.sqrt 2 ∧ z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l4007_400766


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4007_400714

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4007_400714


namespace NUMINAMATH_CALUDE_balloon_distribution_l4007_400772

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 400) 
  (h2 : num_friends = 10) : 
  (total_balloons / num_friends) - ((total_balloons / num_friends) * 3 / 5) = 16 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l4007_400772


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l4007_400745

def normal_distribution (μ σ : ℝ) : Type := ℝ

def probability {α : Type} (p : Set α) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : normal_distribution 0 3) : 
  probability {x : ℝ | -3 < x ∧ x < 6} = 0.8185 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l4007_400745


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l4007_400706

theorem number_of_girls_in_school (total_boys : ℕ) (total_sections : ℕ) 
  (h1 : total_boys = 408)
  (h2 : total_sections = 27)
  (h3 : total_boys % total_sections = 0) -- Boys are divided into equal sections
  : ∃ (total_girls : ℕ), 
    total_girls = 324 ∧ 
    total_girls % total_sections = 0 ∧ -- Girls are divided into equal sections
    (total_boys / total_sections + total_girls / total_sections = total_sections) :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_girls_in_school_l4007_400706


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_6_l4007_400701

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def digit_sum (n : ℕ) : ℕ := 
  let digits := n.digits 10
  digits.sum

theorem five_digit_multiple_of_6 (n : ℕ) : 
  (∃ d : ℕ, n = 84370 + d ∧ d < 10) → 
  is_multiple_of_6 n → 
  (n.mod 10 = 2 ∨ n.mod 10 = 8) :=
sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_6_l4007_400701


namespace NUMINAMATH_CALUDE_cycling_trip_tailwind_time_l4007_400748

theorem cycling_trip_tailwind_time 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_with_tailwind : ℝ) 
  (speed_against_wind : ℝ) 
  (h1 : total_distance = 150) 
  (h2 : total_time = 12) 
  (h3 : speed_with_tailwind = 15) 
  (h4 : speed_against_wind = 10) : 
  ∃ (time_with_tailwind : ℝ), 
    time_with_tailwind = 6 ∧ 
    speed_with_tailwind * time_with_tailwind + 
    speed_against_wind * (total_time - time_with_tailwind) = total_distance := by
  sorry

end NUMINAMATH_CALUDE_cycling_trip_tailwind_time_l4007_400748


namespace NUMINAMATH_CALUDE_newspaper_collection_ratio_l4007_400787

def chris_newspapers : ℕ := 42
def lily_extra_newspapers : ℕ := 23

def lily_newspapers : ℕ := chris_newspapers + lily_extra_newspapers

theorem newspaper_collection_ratio :
  (chris_newspapers : ℚ) / (lily_newspapers : ℚ) = 42 / 65 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_collection_ratio_l4007_400787


namespace NUMINAMATH_CALUDE_product_of_reals_l4007_400723

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l4007_400723


namespace NUMINAMATH_CALUDE_expression_evaluation_l4007_400773

theorem expression_evaluation (b x : ℝ) (h : x = b + 4) :
  2*x - b + 5 = b + 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4007_400773


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4007_400754

theorem expression_simplification_and_evaluation :
  ∀ x y : ℤ, x = -1 ∧ y = 2 →
  (x * y + (3 * x * y - 4 * x^2) - 2 * (x * y - 2 * x^2)) = 2 * x * y ∧
  2 * x * y = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4007_400754


namespace NUMINAMATH_CALUDE_negation_of_square_sum_nonnegative_l4007_400769

theorem negation_of_square_sum_nonnegative :
  (¬ ∀ x y : ℝ, x^2 + y^2 ≥ 0) ↔ (∃ x y : ℝ, x^2 + y^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_sum_nonnegative_l4007_400769


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l4007_400791

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  ¬(7 ∣ (a.val * b.val * (a.val + b.val))) ∧ 
  (7^7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l4007_400791


namespace NUMINAMATH_CALUDE_bertha_family_females_without_daughters_l4007_400793

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  each_daughter_has_equal_children : Bool
  no_great_grandchildren : Bool

/-- Calculates the number of females with no daughters in Bertha's family -/
def females_without_daughters (family : BerthaFamily) : ℕ :=
  family.total_descendants - family.daughters

/-- Theorem stating that the number of females with no daughters in Bertha's family is 32 -/
theorem bertha_family_females_without_daughters :
  ∀ (family : BerthaFamily),
    family.daughters = 8 ∧
    family.total_descendants = 40 ∧
    family.each_daughter_has_equal_children = true ∧
    family.no_great_grandchildren = true →
    females_without_daughters family = 32 := by
  sorry

end NUMINAMATH_CALUDE_bertha_family_females_without_daughters_l4007_400793


namespace NUMINAMATH_CALUDE_toy_ratio_l4007_400725

def total_toys : ℕ := 240
def elder_son_toys : ℕ := 60

def younger_son_toys : ℕ := total_toys - elder_son_toys

theorem toy_ratio :
  (younger_son_toys : ℚ) / elder_son_toys = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_toy_ratio_l4007_400725


namespace NUMINAMATH_CALUDE_isosceles_base_length_l4007_400777

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An equilateral triangle is a triangle where all sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- An isosceles triangle is a triangle where at least two sides are equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The perimeter of a triangle is the sum of its side lengths -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Given an equilateral triangle with perimeter 45 and an isosceles triangle with perimeter 40,
    where at least one side of the isosceles triangle is equal to the side of the equilateral triangle,
    prove that the base of the isosceles triangle is 10 units -/
theorem isosceles_base_length
  (equilateral : Triangle)
  (isosceles : Triangle)
  (h_equilateral : equilateral.isEquilateral)
  (h_isosceles : isosceles.isIsosceles)
  (h_equilateral_perimeter : equilateral.perimeter = 45)
  (h_isosceles_perimeter : isosceles.perimeter = 40)
  (h_shared_side : isosceles.a = equilateral.a ∨ isosceles.b = equilateral.a ∨ isosceles.c = equilateral.a) :
  isosceles.c = 10 ∨ isosceles.b = 10 ∨ isosceles.a = 10 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_base_length_l4007_400777
