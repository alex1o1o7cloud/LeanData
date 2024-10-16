import Mathlib

namespace NUMINAMATH_CALUDE_zoey_holidays_l3032_303258

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of holidays Zoey takes per month -/
def holidays_per_month : ℕ := 2

/-- The total number of holidays Zoey takes in a year -/
def total_holidays : ℕ := months_in_year * holidays_per_month

theorem zoey_holidays : total_holidays = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoey_holidays_l3032_303258


namespace NUMINAMATH_CALUDE_overlapping_strips_area_l3032_303264

/-- Represents a rectangular strip with given length and width -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Calculates the number of overlaps between n strips -/
def numOverlaps (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The total area covered by 5 overlapping strips -/
theorem overlapping_strips_area :
  let strips : List Strip := List.replicate 5 ⟨12, 1⟩
  let totalStripArea := (strips.map stripArea).sum
  let overlapArea := numOverlaps 5
  totalStripArea - overlapArea = 50 := by
  sorry


end NUMINAMATH_CALUDE_overlapping_strips_area_l3032_303264


namespace NUMINAMATH_CALUDE_probability_log_base_2_equal_1_l3032_303274

def dice_face := Fin 6

def is_valid_roll (x y : dice_face) : Prop :=
  (y.val : ℝ) = 2 * (x.val : ℝ)

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 3

theorem probability_log_base_2_equal_1 :
  (favorable_outcomes : ℝ) / total_outcomes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_log_base_2_equal_1_l3032_303274


namespace NUMINAMATH_CALUDE_base_13_conversion_l3032_303269

-- Define a function to convert a base 10 number to base 13
def toBase13 (n : ℕ) : String :=
  sorry

-- Define a function to convert a base 13 string to base 10
def fromBase13 (s : String) : ℕ :=
  sorry

-- Theorem statement
theorem base_13_conversion :
  toBase13 136 = "A6" ∧ fromBase13 "A6" = 136 :=
sorry

end NUMINAMATH_CALUDE_base_13_conversion_l3032_303269


namespace NUMINAMATH_CALUDE_cross_shaped_graph_paper_rectangles_l3032_303248

/-- Calculates the number of rectangles in a grid --/
def rectangleCount (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Calculates the sum of squares from 1 to n --/
def sumOfSquares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

/-- The side length of the original square graph paper in mm --/
def originalSideLength : ℕ := 30

/-- The side length of the cut-away corner squares in mm --/
def cornerSideLength : ℕ := 10

/-- The total number of smallest squares in the original graph paper --/
def totalSmallestSquares : ℕ := 900

theorem cross_shaped_graph_paper_rectangles :
  let totalRectangles := rectangleCount originalSideLength originalSideLength
  let cornerRectangles := 4 * rectangleCount cornerSideLength originalSideLength
  let remainingSquares := 2 * sumOfSquares originalSideLength - sumOfSquares (originalSideLength - 2 * cornerSideLength)
  totalRectangles - cornerRectangles - remainingSquares = 144130 := by
  sorry

end NUMINAMATH_CALUDE_cross_shaped_graph_paper_rectangles_l3032_303248


namespace NUMINAMATH_CALUDE_employee_gross_pay_l3032_303255

/-- Calculate the gross pay for an employee given regular and overtime hours and rates -/
def calculate_gross_pay (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (overtime_hours : ℚ) : ℚ :=
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that the employee's gross pay for the week is $622 -/
theorem employee_gross_pay :
  let regular_rate : ℚ := 11.25
  let regular_hours : ℚ := 40
  let overtime_rate : ℚ := 16
  let overtime_hours : ℚ := 10.75
  calculate_gross_pay regular_rate regular_hours overtime_rate overtime_hours = 622 := by
  sorry

#eval calculate_gross_pay (11.25 : ℚ) (40 : ℚ) (16 : ℚ) (10.75 : ℚ)

end NUMINAMATH_CALUDE_employee_gross_pay_l3032_303255


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3032_303270

/-- A circle passing through three points -/
structure Circle where
  D : ℝ
  E : ℝ

/-- Check if a point lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y = 0

/-- The specific circle we're interested in -/
def our_circle : Circle := { D := -4, E := -6 }

/-- Theorem stating that our_circle passes through the given points -/
theorem circle_passes_through_points : 
  (our_circle.contains 0 0) ∧ 
  (our_circle.contains 4 0) ∧ 
  (our_circle.contains (-1) 1) := by
  sorry


end NUMINAMATH_CALUDE_circle_passes_through_points_l3032_303270


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3032_303263

/-- Given a geometric sequence {a_n} satisfying the condition
    a_4 · a_6 + 2a_5 · a_7 + a_6 · a_8 = 36, prove that a_5 + a_7 = ±6 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_condition : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  (a 5 + a 7 = 6) ∨ (a 5 + a 7 = -6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3032_303263


namespace NUMINAMATH_CALUDE_maximal_k_for_triangle_l3032_303290

theorem maximal_k_for_triangle : ∃ (k : ℝ), k = 5 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → k * a * b * c > a^3 + b^3 + c^3 → 
    a + b > c ∧ b + c > a ∧ c + a > b) ∧
  (∀ (k' : ℝ), k' > k → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ k' * a * b * c > a^3 + b^3 + c^3 ∧
      (a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b)) :=
sorry

end NUMINAMATH_CALUDE_maximal_k_for_triangle_l3032_303290


namespace NUMINAMATH_CALUDE_combined_nuts_bill_harry_l3032_303230

theorem combined_nuts_bill_harry (sue_nuts : ℕ) (harry_nuts : ℕ) (bill_nuts : ℕ) 
  (h1 : sue_nuts = 48)
  (h2 : harry_nuts = 2 * sue_nuts)
  (h3 : bill_nuts = 6 * harry_nuts) :
  bill_nuts + harry_nuts = 672 := by
  sorry

end NUMINAMATH_CALUDE_combined_nuts_bill_harry_l3032_303230


namespace NUMINAMATH_CALUDE_second_quadrant_point_coordinates_l3032_303214

/-- A point in the second quadrant of a coordinate plane. -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_pos : y > 0

/-- The theorem stating that a point in the second quadrant with given distances to the axes has specific coordinates. -/
theorem second_quadrant_point_coordinates (P : SecondQuadrantPoint) 
  (dist_x_axis : |P.y| = 4)
  (dist_y_axis : |P.x| = 5) :
  P.x = -5 ∧ P.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_point_coordinates_l3032_303214


namespace NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l3032_303287

theorem terminal_side_in_second_quadrant (α : Real) 
  (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ x y : Real, x < 0 ∧ y > 0 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ 
  Real.sin α = y / Real.sqrt (x^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l3032_303287


namespace NUMINAMATH_CALUDE_tyler_initial_money_l3032_303268

def scissors_cost : ℕ := 8 * 5
def erasers_cost : ℕ := 10 * 4
def remaining_money : ℕ := 20

theorem tyler_initial_money :
  scissors_cost + erasers_cost + remaining_money = 100 :=
by sorry

end NUMINAMATH_CALUDE_tyler_initial_money_l3032_303268


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l3032_303201

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by
  sorry

theorem negation_of_squared_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l3032_303201


namespace NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l3032_303218

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 - x) / (x - 2) + 1 / (2 - x) = 1
def equation2 (x : ℝ) : Prop := 3 / (x^2 - 4) + 2 / (x + 2) = 1 / (x - 2)

-- Theorem for equation 1
theorem no_solution_equation1 : ¬∃ x : ℝ, equation1 x := by sorry

-- Theorem for equation 2
theorem unique_solution_equation2 : ∃! x : ℝ, equation2 x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l3032_303218


namespace NUMINAMATH_CALUDE_remainder_problem_l3032_303298

theorem remainder_problem (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3032_303298


namespace NUMINAMATH_CALUDE_patricks_age_to_roberts_age_ratio_l3032_303249

/-- Given that Robert will turn 30 after 2 years and Patrick is 14 years old now,
    prove that the ratio of Patrick's age to Robert's age is 1:2 -/
theorem patricks_age_to_roberts_age_ratio :
  ∀ (roberts_age patricks_age : ℕ),
  roberts_age + 2 = 30 →
  patricks_age = 14 →
  patricks_age / roberts_age = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_patricks_age_to_roberts_age_ratio_l3032_303249


namespace NUMINAMATH_CALUDE_rational_number_properties_l3032_303231

theorem rational_number_properties (a b : ℚ) 
  (h_product : a * b < 0)
  (h_sum : a + b < 0) :
  (abs a > abs b ∧ a < 0 ∧ b > 0) ∨ (abs b > abs a ∧ b < 0 ∧ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_number_properties_l3032_303231


namespace NUMINAMATH_CALUDE_return_probability_eight_reflections_l3032_303225

/-- A square with a point at its center -/
structure CenteredSquare where
  /-- The square -/
  square : Set (ℝ × ℝ)
  /-- The center point -/
  center : ℝ × ℝ
  /-- The center point is in the square -/
  center_in_square : center ∈ square

/-- A reflection over a line in a square -/
def reflect (s : CenteredSquare) (line : Set (ℝ × ℝ)) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The sequence of points generated by reflections -/
def reflection_sequence (s : CenteredSquare) (n : ℕ) : ℝ × ℝ := sorry

/-- The probability of returning to the center after n reflections -/
def return_probability (s : CenteredSquare) (n : ℕ) : ℚ := sorry

theorem return_probability_eight_reflections (s : CenteredSquare) :
  return_probability s 8 = 1225 / 16384 := by sorry

end NUMINAMATH_CALUDE_return_probability_eight_reflections_l3032_303225


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3032_303261

theorem complex_equation_solution :
  ∃ (x y : ℝ), (-5 + 2 * Complex.I) * x - (3 - 4 * Complex.I) * y = 2 - Complex.I ∧
  x = -5/14 ∧ y = -1/14 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3032_303261


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_main_theorem_l3032_303216

/-- Represents a tetrahedron A-BCD with specific properties -/
structure Tetrahedron where
  /-- Base BCD is an equilateral triangle with side length 2 -/
  base_side_length : ℝ
  base_is_equilateral : base_side_length = 2
  /-- Projection of vertex A onto base BCD is the center of triangle BCD -/
  vertex_projection_is_center : Bool
  /-- E is the midpoint of side BC -/
  e_is_midpoint : Bool
  /-- Sine of angle formed by line AE with base BCD is 2√2 -/
  sine_angle_ae_base : ℝ
  sine_angle_ae_base_value : sine_angle_ae_base = 2 * Real.sqrt 2

/-- The surface area of the circumscribed sphere of the tetrahedron is 6π -/
theorem circumscribed_sphere_surface_area (t : Tetrahedron) : ℝ := by
  sorry

/-- Main theorem: The surface area of the circumscribed sphere is 6π -/
theorem main_theorem (t : Tetrahedron) : circumscribed_sphere_surface_area t = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_main_theorem_l3032_303216


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l3032_303217

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The point on the circle -/
def point_on_circle : ℝ × ℝ := (4, 1)

/-- The proposed tangent line equation -/
def tangent_line_equation (x y : ℝ) : Prop := 3*x + 4*y - 16 = 0 ∨ x = 4

/-- Theorem stating that the proposed equation represents the tangent line -/
theorem tangent_line_is_correct :
  tangent_line_equation (point_on_circle.1) (point_on_circle.2) ∧
  ∀ (x y : ℝ), circle_equation x y →
    tangent_line_equation x y →
    (x, y) = point_on_circle ∨
    ∃ (t : ℝ), (x, y) = (point_on_circle.1 + t, point_on_circle.2 + t) ∧ t ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l3032_303217


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3032_303289

theorem complex_number_quadrant : 
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3032_303289


namespace NUMINAMATH_CALUDE_cows_eating_grass_l3032_303293

-- Define the amount of hectares a cow eats per week
def cow_eat_rate : ℚ := 1/2

-- Define the amount of hectares of grass that grows per week
def grass_growth_rate : ℚ := 1/2

-- Define the function that calculates the amount of grass eaten
def grass_eaten (cows : ℕ) (weeks : ℕ) : ℚ :=
  (cows : ℚ) * cow_eat_rate * (weeks : ℚ)

-- Define the function that calculates the amount of grass regrown
def grass_regrown (hectares : ℕ) (weeks : ℕ) : ℚ :=
  (hectares : ℚ) * grass_growth_rate * (weeks : ℚ)

-- Theorem statement
theorem cows_eating_grass (cows : ℕ) : 
  (grass_eaten 3 2 - grass_regrown 2 2 = 2) →
  (grass_eaten 2 4 - grass_regrown 2 4 = 2) →
  (grass_eaten cows 6 - grass_regrown 6 6 = 6) →
  cows = 3 := by
  sorry

end NUMINAMATH_CALUDE_cows_eating_grass_l3032_303293


namespace NUMINAMATH_CALUDE_school_sampling_l3032_303286

theorem school_sampling (total_students sample_size : ℕ) 
  (h_total : total_students = 1200)
  (h_sample : sample_size = 200)
  (h_stratified : ∃ (boys girls : ℕ), 
    boys + girls = sample_size ∧ 
    boys = girls + 10 ∧ 
    (boys : ℚ) / total_students = (boys : ℚ) / sample_size) :
  ∃ (school_boys : ℕ), school_boys = 630 ∧ 
    (school_boys : ℚ) / total_students = 
    ((sample_size / 2 + 5) : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_school_sampling_l3032_303286


namespace NUMINAMATH_CALUDE_q_prime_div_p_prime_eq_550_l3032_303219

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def distinct_numbers : ℕ := 12

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p' : ℚ := 12 / (total_slips.choose drawn_slips)

/-- The probability of drawing three slips with one number and two with another -/
def q' : ℚ := (6600 : ℚ) / (total_slips.choose drawn_slips)

/-- The main theorem stating the ratio of q' to p' -/
theorem q_prime_div_p_prime_eq_550 : q' / p' = 550 := by sorry

end NUMINAMATH_CALUDE_q_prime_div_p_prime_eq_550_l3032_303219


namespace NUMINAMATH_CALUDE_rectangle_area_l3032_303275

/-- The area of a rectangle with sides 1.5 meters and 0.75 meters is 1.125 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 1.5
  let width : ℝ := 0.75
  length * width = 1.125 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3032_303275


namespace NUMINAMATH_CALUDE_rice_sales_problem_l3032_303220

/-- Represents the daily rice sales function -/
structure RiceSales where
  k : ℝ
  b : ℝ
  y : ℝ → ℝ
  h1 : ∀ x, y x = k * x + b
  h2 : y 5 = 950
  h3 : y 6 = 900

/-- Calculates the profit for a given price -/
def profit (price : ℝ) (sales : ℝ) : ℝ := (price - 4) * sales

theorem rice_sales_problem (rs : RiceSales) :
  (rs.y = λ x => -50 * x + 1200) ∧
  (∃ x ∈ Set.Icc 4 7, profit x (rs.y x) = 1800 ∧ x = 6) ∧
  (∀ x ∈ Set.Icc 4 7, profit x (rs.y x) ≤ 2550) ∧
  (profit 7 (rs.y 7) = 2550) := by
  sorry

end NUMINAMATH_CALUDE_rice_sales_problem_l3032_303220


namespace NUMINAMATH_CALUDE_six_digit_number_remainder_l3032_303291

/-- Represents a 6-digit number in the form 6x62y4 -/
def SixDigitNumber (x y : Nat) : Nat :=
  600000 + 10000 * x + 6200 + 10 * y + 4

theorem six_digit_number_remainder (x y : Nat) :
  x < 10 → y < 10 →
  (SixDigitNumber x y) % 11 = 0 →
  (SixDigitNumber x y) % 9 = 6 →
  (SixDigitNumber x y) % 13 = 6 := by
sorry

end NUMINAMATH_CALUDE_six_digit_number_remainder_l3032_303291


namespace NUMINAMATH_CALUDE_three_digit_number_operations_l3032_303250

theorem three_digit_number_operations (a b c : Nat) 
  (h1 : a > 0) 
  (h2 : a < 10) 
  (h3 : b < 10) 
  (h4 : c < 10) : 
  ((2 * a + 3) * 5 + b) * 10 + c - 150 = 100 * a + 10 * b + c := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_operations_l3032_303250


namespace NUMINAMATH_CALUDE_max_vertex_sum_l3032_303285

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h_T_pos : T > 0
  h_passes_through : ∀ x y, y = a * x * (x - T) → 
    ((x = 0 ∧ y = 0) ∨ (x = T ∧ y = 0) ∨ (x = T + 1 ∧ y = 50))

/-- The sum of coordinates of the vertex of the parabola -/
def vertexSum (p : Parabola) : ℚ :=
  p.T / 2 - (p.a * p.T^2) / 4

/-- The theorem stating the maximum value of the vertex sum -/
theorem max_vertex_sum (p : Parabola) : 
  vertexSum p ≤ -23/2 :=
sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l3032_303285


namespace NUMINAMATH_CALUDE_prime_sum_2003_l3032_303253

theorem prime_sum_2003 (a b : ℕ) (ha : Prime a) (hb : Prime b) (heq : a^2 + b = 2003) :
  a + b = 2001 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_2003_l3032_303253


namespace NUMINAMATH_CALUDE_f_range_l3032_303208

def closest_multiple (k : ℤ) (n : ℤ) : ℤ :=
  n * round (k / n)

def f (k : ℤ) : ℤ :=
  closest_multiple k 3 + closest_multiple (2*k) 5 + closest_multiple (3*k) 7 - 6*k

theorem f_range :
  (∀ k : ℤ, -6 ≤ f k ∧ f k ≤ 6) ∧
  (∀ m : ℤ, -6 ≤ m ∧ m ≤ 6 → ∃ k : ℤ, f k = m) :=
sorry

end NUMINAMATH_CALUDE_f_range_l3032_303208


namespace NUMINAMATH_CALUDE_revenue_is_78_l3032_303252

/-- The revenue per t-shirt for a shop selling t-shirts during two games -/
def revenue_per_tshirt (total_tshirts : ℕ) (first_game_tshirts : ℕ) (second_game_revenue : ℕ) : ℚ :=
  second_game_revenue / (total_tshirts - first_game_tshirts)

/-- Theorem stating that the revenue per t-shirt is $78 given the specified conditions -/
theorem revenue_is_78 :
  revenue_per_tshirt 186 172 1092 = 78 := by
  sorry

#eval revenue_per_tshirt 186 172 1092

end NUMINAMATH_CALUDE_revenue_is_78_l3032_303252


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3032_303215

theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) : 
  (∀ x : ℝ, ax^2 - 6*x + a^2 < 0 ↔ 1 < x ∧ x < m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3032_303215


namespace NUMINAMATH_CALUDE_rahim_average_book_price_l3032_303212

/-- The average price of books bought by Rahim -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2 : ℚ)

/-- Theorem: The average price Rahim paid per book is 85 rupees -/
theorem rahim_average_book_price :
  average_price 65 35 6500 2000 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rahim_average_book_price_l3032_303212


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3032_303223

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 25 * x + 9 = 0) :
  ∃ x, b * x^2 + 25 * x + 9 = 0 ∧ x = -18/25 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3032_303223


namespace NUMINAMATH_CALUDE_f_inequality_l3032_303205

/-- A quadratic function with positive leading coefficient and axis of symmetry at x=1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) > f(2^x) for x > 0 -/
theorem f_inequality (a b c : ℝ) (h_a : a > 0) (h_sym : ∀ x, f a b c (2 - x) = f a b c x) :
  ∀ x > 0, f a b c (3^x) > f a b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3032_303205


namespace NUMINAMATH_CALUDE_pyramid_properties_l3032_303265

-- Define the cone and pyramid
structure Cone where
  height : ℝ
  slantHeight : ℝ

structure Pyramid where
  cone : Cone
  OB : ℝ

-- Define the properties of the cone and pyramid
def isValidCone (c : Cone) : Prop :=
  c.height = 4 ∧ c.slantHeight = 5

def isValidPyramid (p : Pyramid) : Prop :=
  isValidCone p.cone ∧ p.OB = 3

-- Define the properties to be proved
def pyramidVolume (p : Pyramid) : ℝ := sorry

def dihedralAngleAB (p : Pyramid) : ℝ := sorry

def circumscribedSphereRadius (p : Pyramid) : ℝ := sorry

-- Main theorem
theorem pyramid_properties (p : Pyramid) 
  (h : isValidPyramid p) : 
  ∃ (v d r : ℝ),
    pyramidVolume p = v ∧
    dihedralAngleAB p = d ∧
    circumscribedSphereRadius p = r :=
  sorry

end NUMINAMATH_CALUDE_pyramid_properties_l3032_303265


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3032_303247

/-- Given that the solution set of ax^2 + 5x + b > 0 is {x | 2 < x < 3},
    prove that the solution set of bx^2 - 5x + a > 0 is (-1/2, -1/3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x + b > 0 ↔ 2 < x ∧ x < 3) →
  (∀ x : ℝ, b*x^2 - 5*x + a > 0 ↔ -1/2 < x ∧ x < -1/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3032_303247


namespace NUMINAMATH_CALUDE_building_heights_sum_l3032_303239

theorem building_heights_sum (h1 h2 h3 h4 : ℝ) : 
  h1 = 100 →
  h2 = h1 / 2 →
  h3 = h2 / 2 →
  h4 = h3 / 5 →
  h1 + h2 + h3 + h4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l3032_303239


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_neg_two_l3032_303240

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

/-- Theorem: If z(a) is purely imaginary, then a = -2 -/
theorem purely_imaginary_implies_a_eq_neg_two :
  ∀ a : ℝ, isPurelyImaginary (z a) → a = -2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_neg_two_l3032_303240


namespace NUMINAMATH_CALUDE_mean_height_of_players_l3032_303251

def heights : List ℕ := [145, 149, 151, 151, 157, 158, 163, 163, 164, 167, 168, 169, 170, 175]

def total_players : ℕ := heights.length

def sum_of_heights : ℕ := heights.sum

theorem mean_height_of_players :
  (sum_of_heights : ℚ) / (total_players : ℚ) = 160.714 := by sorry

end NUMINAMATH_CALUDE_mean_height_of_players_l3032_303251


namespace NUMINAMATH_CALUDE_total_planting_area_is_2600_l3032_303271

/-- Represents the number of trees to be planted for each tree chopped -/
structure PlantingRatio :=
  (oak : ℕ)
  (pine : ℕ)

/-- Represents the number of trees chopped in each half of the year -/
structure TreesChopped :=
  (oak : ℕ)
  (pine : ℕ)

/-- Represents the space required for planting each type of tree -/
structure PlantingSpace :=
  (oak : ℕ)
  (pine : ℕ)

/-- Calculates the total area needed for tree planting during the entire year -/
def totalPlantingArea (ratio : PlantingRatio) (firstHalf : TreesChopped) (secondHalf : TreesChopped) (space : PlantingSpace) : ℕ :=
  let oakArea := (firstHalf.oak * ratio.oak * space.oak)
  let pineArea := ((firstHalf.pine + secondHalf.pine) * ratio.pine * space.pine)
  oakArea + pineArea

/-- Theorem stating that the total area needed for tree planting is 2600 m² -/
theorem total_planting_area_is_2600 (ratio : PlantingRatio) (firstHalf : TreesChopped) (secondHalf : TreesChopped) (space : PlantingSpace) :
  ratio.oak = 4 →
  ratio.pine = 2 →
  firstHalf.oak = 100 →
  firstHalf.pine = 100 →
  secondHalf.oak = 150 →
  secondHalf.pine = 150 →
  space.oak = 4 →
  space.pine = 2 →
  totalPlantingArea ratio firstHalf secondHalf space = 2600 :=
by
  sorry

end NUMINAMATH_CALUDE_total_planting_area_is_2600_l3032_303271


namespace NUMINAMATH_CALUDE_expression_value_l3032_303278

theorem expression_value : 
  let a : ℝ := 5
  let b : ℝ := 7
  let c : ℝ := 3
  (2*a - (3*b - 4*c)) - ((2*a - 3*b) - 4*c) = 24 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3032_303278


namespace NUMINAMATH_CALUDE_camp_III_selected_count_l3032_303203

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat
  campIIIStart : Nat
  campIIIEnd : Nat

/-- Calculates the number of students selected from Camp III in a systematic sample -/
def countCampIIISelected (s : SystematicSample) : Nat :=
  let interval := s.totalStudents / s.sampleSize
  let firstCampIII := s.startNumber + interval * ((s.campIIIStart - s.startNumber + interval - 1) / interval)
  let lastSelected := s.startNumber + interval * (s.sampleSize - 1)
  if firstCampIII > s.campIIIEnd then 0
  else ((min lastSelected s.campIIIEnd) - firstCampIII) / interval + 1

theorem camp_III_selected_count (s : SystematicSample) 
  (h1 : s.totalStudents = 600) 
  (h2 : s.sampleSize = 50) 
  (h3 : s.startNumber = 3) 
  (h4 : s.campIIIStart = 496) 
  (h5 : s.campIIIEnd = 600) : 
  countCampIIISelected s = 8 := by
  sorry

end NUMINAMATH_CALUDE_camp_III_selected_count_l3032_303203


namespace NUMINAMATH_CALUDE_area_comparison_l3032_303235

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define a function to check if a triangle is inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define a function to find the points where angle bisectors meet the circle
def angleBisectorPoints (t : Triangle) (c : Circle) : Triangle := sorry

-- Theorem statement
theorem area_comparison 
  (t : Triangle) (c : Circle) 
  (h : isInscribed t c) : 
  let t' := angleBisectorPoints t c
  triangleArea t ≤ triangleArea t' := by sorry

end NUMINAMATH_CALUDE_area_comparison_l3032_303235


namespace NUMINAMATH_CALUDE_circle_radius_equals_one_l3032_303226

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define a right triangle
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧ 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define an equilateral triangle
def EquilateralTriangle (A D E : ℝ × ℝ) : Prop :=
  Triangle A D E ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (E.1 - A.1)^2 + (E.2 - A.2)^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2

-- Define a point on a line segment
def PointOnSegment (P X Y : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  P.1 = X.1 + t * (Y.1 - X.1) ∧
  P.2 = X.2 + t * (Y.2 - X.2)

-- Main theorem
theorem circle_radius_equals_one 
  (A B C D E : ℝ × ℝ) :
  RightTriangle A B C →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1 →
  PointOnSegment D B C →
  PointOnSegment E A C →
  EquilateralTriangle A D E →
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_equals_one_l3032_303226


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_of_powers_l3032_303236

theorem cube_root_unity_sum_of_powers : 
  let ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  (ω ^ 3 = 1) → (ω ≠ 1) →
  (ω ^ 8 + (ω ^ 2) ^ 8 = -2) := by
sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_of_powers_l3032_303236


namespace NUMINAMATH_CALUDE_k_range_when_f_less_than_bound_l3032_303292

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + (1-k)*x - k * Real.log x

theorem k_range_when_f_less_than_bound (k : ℝ) (h_k_pos : k > 0) :
  (∃ x₀ : ℝ, f k x₀ < 3/2 - k^2) → 0 < k ∧ k < 1 := by sorry

end NUMINAMATH_CALUDE_k_range_when_f_less_than_bound_l3032_303292


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3032_303222

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 10 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3032_303222


namespace NUMINAMATH_CALUDE_reflection_composition_maps_points_l3032_303297

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define reflection operations
def reflectY (p : Point) : Point :=
  (-p.1, p.2)

def reflectX (p : Point) : Point :=
  (p.1, -p.2)

-- Define the composition of reflections
def reflectYX (p : Point) : Point :=
  reflectX (reflectY p)

-- Theorem statement
theorem reflection_composition_maps_points :
  let C : Point := (3, -2)
  let D : Point := (4, -5)
  let C' : Point := (-3, 2)
  let D' : Point := (-4, 5)
  reflectYX C = C' ∧ reflectYX D = D' := by sorry

end NUMINAMATH_CALUDE_reflection_composition_maps_points_l3032_303297


namespace NUMINAMATH_CALUDE_square_rectangle_intersection_l3032_303238

theorem square_rectangle_intersection (EFGH_side_length MO LO shaded_area : ℝ) :
  EFGH_side_length = 8 →
  MO = 12 →
  LO = 8 →
  shaded_area = (MO * LO) / 2 →
  shaded_area = EFGH_side_length * (EFGH_side_length - EM) →
  EM = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_intersection_l3032_303238


namespace NUMINAMATH_CALUDE_association_confidence_level_l3032_303229

-- Define the χ² value
def chi_squared : ℝ := 6.825

-- Define the degrees of freedom for a 2x2 contingency table
def degrees_of_freedom : ℕ := 1

-- Define the critical value for 99% confidence level with 1 degree of freedom
def critical_value : ℝ := 6.635

-- Define the confidence level we want to prove
def target_confidence_level : ℝ := 99

-- Theorem statement
theorem association_confidence_level :
  chi_squared > critical_value →
  (∃ (confidence_level : ℝ), confidence_level ≥ target_confidence_level) :=
sorry

end NUMINAMATH_CALUDE_association_confidence_level_l3032_303229


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3032_303279

theorem max_value_of_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h_sum : x + y + z = 3) :
  (x * y / (x + y + 1) + x * z / (x + z + 1) + y * z / (y + z + 1)) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    x * y / (x + y + 1) + x * z / (x + z + 1) + y * z / (y + z + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3032_303279


namespace NUMINAMATH_CALUDE_min_sum_squares_l3032_303295

theorem min_sum_squares (x y z : ℝ) (h : 2*x + y + 2*z = 6) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), 2*a + b + 2*c = 6 → a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3032_303295


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3032_303234

theorem equal_roots_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (x * (x - 1) - (m^2 + 2*m + 1)) / ((x - 1) * (m^2 - 1) + 1) = x / m) →
  (∃! x : ℝ, x * (x - 1) - (m^2 + 2*m + 1) = 0) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3032_303234


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3032_303277

open Real

theorem trigonometric_inequality : 
  let a : ℝ := sin (46 * π / 180)
  let b : ℝ := cos (46 * π / 180)
  let c : ℝ := tan (46 * π / 180)
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3032_303277


namespace NUMINAMATH_CALUDE_product_digit_sum_base7_l3032_303244

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digit_sum_base7 :
  sumOfDigitsBase7 (toBase7 (toBase10 35 * toBase10 13)) = 11 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base7_l3032_303244


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3032_303294

theorem quadratic_rewrite (b : ℝ) (h1 : b < 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 2/3 = (x + n)^2 + 1/4) →
  b = -Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3032_303294


namespace NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l3032_303233

theorem ab_nonzero_sufficient_not_necessary_for_a_nonzero (a b : ℝ) :
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ ab = 0) :=
by sorry

end NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l3032_303233


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3032_303273

/-- Given a hyperbola with equation x²/m - y²/3 = 1 where m > 0,
    if one of its asymptotic lines is y = (1/2)x, then m = 12 -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) : 
  (∃ (x y : ℝ), x^2 / m - y^2 / 3 = 1 ∧ y = (1/2) * x) → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3032_303273


namespace NUMINAMATH_CALUDE_f_less_than_g_iff_m_in_range_l3032_303276

-- Define the functions f and g
def f (x m : ℝ) : ℝ := |x - 1| + |x + m|
def g (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem f_less_than_g_iff_m_in_range :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (-m) 1, f x m < g x) ↔ -1 < m ∧ m < -2/3 := by sorry

end NUMINAMATH_CALUDE_f_less_than_g_iff_m_in_range_l3032_303276


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3032_303283

theorem quadratic_equation_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + 2 = 0 ∧ x₂^2 - 4*x₂ + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3032_303283


namespace NUMINAMATH_CALUDE_parabola_b_value_l3032_303224

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, -p), where p ≠ 0, 
    prove that b = 4. -/
theorem parabola_b_value (a b c p : ℝ) (hp : p ≠ 0) 
  (h_vertex : ∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p)
  (h_y_intercept : c = -p) : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3032_303224


namespace NUMINAMATH_CALUDE_simplify_fraction_l3032_303257

theorem simplify_fraction : (45 : ℚ) / 75 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3032_303257


namespace NUMINAMATH_CALUDE_isabella_hair_length_l3032_303266

/-- The length of Isabella's hair before the haircut -/
def hair_length_before : ℕ := sorry

/-- The length of Isabella's hair after the haircut -/
def hair_length_after : ℕ := 9

/-- The length of hair that was cut off -/
def hair_length_cut : ℕ := 9

/-- Theorem stating that the length of Isabella's hair before the haircut
    is equal to the sum of the length after the haircut and the length cut off -/
theorem isabella_hair_length : hair_length_before = hair_length_after + hair_length_cut := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_length_l3032_303266


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3032_303211

/-- Given two parallel vectors a and b, prove that x = 1/2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (∃ (k : ℝ), a = k • b) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3032_303211


namespace NUMINAMATH_CALUDE_meter_to_km_conversion_kg_to_g_conversion_cm_to_dm_conversion_time_to_minutes_conversion_l3032_303280

-- Define conversion rates
def meter_to_km : ℕ → ℕ := λ m => m / 1000
def kg_to_g : ℕ → ℕ := λ kg => kg * 1000
def cm_to_dm : ℕ → ℕ := λ cm => cm / 10
def hours_to_minutes : ℕ → ℕ := λ h => h * 60

-- Theorem statements
theorem meter_to_km_conversion : meter_to_km 6000 = 6 := by sorry

theorem kg_to_g_conversion : kg_to_g (5 + 2) = 7000 := by sorry

theorem cm_to_dm_conversion : cm_to_dm (58 + 32) = 9 := by sorry

theorem time_to_minutes_conversion : hours_to_minutes 3 + 30 = 210 := by sorry

end NUMINAMATH_CALUDE_meter_to_km_conversion_kg_to_g_conversion_cm_to_dm_conversion_time_to_minutes_conversion_l3032_303280


namespace NUMINAMATH_CALUDE_pyramid_height_formula_l3032_303299

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base -/
  base_perimeter : ℝ
  /-- The distance from the apex to each vertex of the square base -/
  apex_to_vertex : ℝ

/-- The height of the pyramid from its peak to the center of the square base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

theorem pyramid_height_formula (p : RightPyramid) 
  (h1 : p.base_perimeter = 40)
  (h2 : p.apex_to_vertex = 15) : 
  pyramid_height p = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_formula_l3032_303299


namespace NUMINAMATH_CALUDE_quadratic_roots_same_sign_l3032_303237

theorem quadratic_roots_same_sign (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + 2*x₁ + m = 0 ∧ 
   x₂^2 + 2*x₂ + m = 0 ∧
   (x₁ > 0 ∧ x₂ > 0 ∨ x₁ < 0 ∧ x₂ < 0)) →
  (0 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_same_sign_l3032_303237


namespace NUMINAMATH_CALUDE_eleventhNumberWithSumOfDigits12Is156_l3032_303228

-- Define a function to check if the sum of digits of a number is 12
def sumOfDigitsIs12 (n : ℕ) : Prop := sorry

-- Define a function to get the nth number in the sequence
def nthNumberWithSumOfDigits12 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem eleventhNumberWithSumOfDigits12Is156 : 
  nthNumberWithSumOfDigits12 11 = 156 := by sorry

end NUMINAMATH_CALUDE_eleventhNumberWithSumOfDigits12Is156_l3032_303228


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l3032_303262

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing k balls of a specific color from a bag -/
def prob_draw (bag : Bag) (color : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose color k : ℚ) / (Nat.choose (bag.white + bag.black) k)

/-- The probability of drawing all black balls from both bags -/
def prob_all_black (bagA bagB : Bag) : ℚ :=
  (prob_draw bagA bagA.black 2) * (prob_draw bagB bagB.black 2)

/-- The probability of drawing exactly one white ball from both bags -/
def prob_one_white (bagA bagB : Bag) : ℚ :=
  (prob_draw bagA bagA.black 2) * (prob_draw bagB bagB.white 1) * (prob_draw bagB bagB.black 1) +
  (prob_draw bagA bagA.white 1) * (prob_draw bagA bagA.black 1) * (prob_draw bagB bagB.black 2)

theorem ball_probability_theorem (bagA bagB : Bag) 
  (hA : bagA = ⟨2, 4⟩) (hB : bagB = ⟨1, 4⟩) : 
  prob_all_black bagA bagB = 6/25 ∧ prob_one_white bagA bagB = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l3032_303262


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3032_303288

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + Complex.I) * z = -1 + 5 * Complex.I → z = 2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3032_303288


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l3032_303241

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l3032_303241


namespace NUMINAMATH_CALUDE_marcus_final_cards_l3032_303245

def marcus_initial_cards : ℕ := 2100
def carter_initial_cards : ℕ := 3040
def carter_gift_cards : ℕ := 750
def carter_gift_percentage : ℚ := 125 / 1000

theorem marcus_final_cards : 
  marcus_initial_cards + carter_gift_cards + 
  (carter_initial_cards * carter_gift_percentage).floor = 3230 :=
by sorry

end NUMINAMATH_CALUDE_marcus_final_cards_l3032_303245


namespace NUMINAMATH_CALUDE_mushroom_collection_l3032_303296

theorem mushroom_collection (total_mushrooms : ℕ) (h1 : total_mushrooms = 289) :
  ∃ (num_children : ℕ) (mushrooms_per_child : ℕ),
    num_children > 0 ∧
    mushrooms_per_child > 0 ∧
    num_children * mushrooms_per_child = total_mushrooms ∧
    num_children = 17 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l3032_303296


namespace NUMINAMATH_CALUDE_emilys_final_score_l3032_303272

/-- A trivia game with 5 rounds and specific scoring rules -/
def triviaGame (round1 round2 round3 round4Base round5Base lastRoundLoss : ℕ) 
               (round4Multiplier round5Multiplier : ℕ) : ℕ :=
  round1 + round2 + round3 + 
  (round4Base * round4Multiplier) + 
  (round5Base * round5Multiplier) - 
  lastRoundLoss

/-- The final score of Emily's trivia game -/
theorem emilys_final_score : 
  triviaGame 16 33 21 10 4 48 2 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_emilys_final_score_l3032_303272


namespace NUMINAMATH_CALUDE_cos_sum_fifteenth_l3032_303204

theorem cos_sum_fifteenth : Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (8 * Real.pi / 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_fifteenth_l3032_303204


namespace NUMINAMATH_CALUDE_calculation_proof_l3032_303227

theorem calculation_proof : (15200 * 3^2) / 12 / (6^3 * 5) = 10.5555555556 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3032_303227


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l3032_303200

/-- A quadratic polynomial. -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given point. -/
def evaluate (q : QuadraticPolynomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The property that [q(x)]^3 - x is divisible by (x - 2)(x + 2)(x - 5). -/
def hasDivisibilityProperty (q : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, (x = 2 ∨ x = -2 ∨ x = 5) → (evaluate q x)^3 = x

theorem quadratic_polynomial_property (q : QuadraticPolynomial) 
  (h : hasDivisibilityProperty q) : evaluate q 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l3032_303200


namespace NUMINAMATH_CALUDE_percent_equality_l3032_303259

theorem percent_equality (x : ℝ) : (70 / 100 * 600 = 40 / 100 * x) → x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3032_303259


namespace NUMINAMATH_CALUDE_principal_is_20000_l3032_303246

/-- Calculates the principal amount given the interest rate, time, and total interest -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 20000 -/
theorem principal_is_20000 : 
  let rate : ℚ := 12
  let time : ℕ := 3
  let interest : ℚ := 7200
  calculate_principal rate time interest = 20000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_20000_l3032_303246


namespace NUMINAMATH_CALUDE_f_properties_l3032_303282

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (x + 2) * Real.exp (-x) - 2
  else (x - 2) * Real.exp x + 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x ≤ 0, f x = (x + 2) * Real.exp (-x) - 2) →
  (∀ x > 0, f x = (x - 2) * Real.exp x + 2) ∧
  (∀ m : ℝ, (∃ x ∈ Set.Icc 0 2, f x = m) ↔ m ∈ Set.Icc (2 - Real.exp 1) 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3032_303282


namespace NUMINAMATH_CALUDE_point_on_axes_l3032_303281

theorem point_on_axes (a : ℝ) :
  let P : ℝ × ℝ := (2*a - 1, a + 2)
  (P.1 = 0 ∨ P.2 = 0) → (P = (-5, 0) ∨ P = (0, 2.5)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_axes_l3032_303281


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l3032_303284

theorem volunteer_allocation_schemes (n : ℕ) (k : ℕ) :
  n = 5 ∧ k = 4 →
  (Nat.choose n 2) * (Nat.factorial k) = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l3032_303284


namespace NUMINAMATH_CALUDE_race_distance_proof_l3032_303210

/-- The length of the race track in feet -/
def track_length : ℕ := 5000

/-- The distance Alex and Max run evenly at the start -/
def even_start : ℕ := 200

/-- The distance Alex gets ahead after the even start -/
def alex_first_lead : ℕ := 300

/-- The distance Alex gets ahead at the end -/
def alex_final_lead : ℕ := 440

/-- The distance left for Max to catch up at the end -/
def max_remaining : ℕ := 3890

/-- The unknown distance Max gets ahead of Alex -/
def max_lead : ℕ := 170

theorem race_distance_proof :
  even_start + alex_first_lead + max_lead + alex_final_lead = track_length - max_remaining :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l3032_303210


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l3032_303260

/-- Two vectors in ℝ³ -/
def v1 : Fin 3 → ℝ := ![3, -1, 4]
def v2 (x : ℝ) : Fin 3 → ℝ := ![x, 4, -2]

/-- Dot product of two vectors in ℝ³ -/
def dot_product (u v : Fin 3 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

/-- The theorem stating that x = 4 makes v1 and v2 orthogonal -/
theorem orthogonal_vectors :
  ∃ x : ℝ, dot_product v1 (v2 x) = 0 ∧ x = 4 := by
  sorry

#check orthogonal_vectors

end NUMINAMATH_CALUDE_orthogonal_vectors_l3032_303260


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_squares_min_reciprocal_sum_squares_achieved_l3032_303213

theorem min_reciprocal_sum_squares (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x^2 + 1 / y^2) ≥ 2 / 25 :=
by sorry

theorem min_reciprocal_sum_squares_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 20 ∧ 1 / x^2 + 1 / y^2 < 2 / 25 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_squares_min_reciprocal_sum_squares_achieved_l3032_303213


namespace NUMINAMATH_CALUDE_honey_barrel_distribution_l3032_303267

/-- Represents a barrel of honey -/
inductive Barrel
  | Full
  | Half
  | Empty

/-- Represents a distribution of barrels to a person -/
structure Distribution :=
  (full : ℕ)
  (half : ℕ)
  (empty : ℕ)

/-- Calculates the amount of honey in a distribution -/
def honey_amount (d : Distribution) : ℚ :=
  d.full + d.half / 2

/-- Calculates the total number of barrels in a distribution -/
def barrel_count (d : Distribution) : ℕ :=
  d.full + d.half + d.empty

/-- Checks if a distribution is valid (7 barrels and 3.5 units of honey) -/
def is_valid_distribution (d : Distribution) : Prop :=
  barrel_count d = 7 ∧ honey_amount d = 7/2

/-- Represents a solution to the honey distribution problem -/
structure Solution :=
  (person1 : Distribution)
  (person2 : Distribution)
  (person3 : Distribution)

/-- Checks if a solution is valid -/
def is_valid_solution (s : Solution) : Prop :=
  is_valid_distribution s.person1 ∧
  is_valid_distribution s.person2 ∧
  is_valid_distribution s.person3 ∧
  s.person1.full + s.person2.full + s.person3.full = 7 ∧
  s.person1.half + s.person2.half + s.person3.half = 7 ∧
  s.person1.empty + s.person2.empty + s.person3.empty = 7

theorem honey_barrel_distribution :
  ∃ (s : Solution), is_valid_solution s :=
sorry

end NUMINAMATH_CALUDE_honey_barrel_distribution_l3032_303267


namespace NUMINAMATH_CALUDE_triangle_properties_l3032_303207

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.a * t.c * Real.sin t.B = t.b^2 - (t.a - t.c)^2) :
  (Real.sin t.B = 4/5) ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → t.b^2 / (x^2 + y^2) ≥ 2/5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3032_303207


namespace NUMINAMATH_CALUDE_min_saltwater_animals_is_1136_l3032_303202

/-- The minimum number of saltwater animals Tyler has -/
def min_saltwater_animals : ℕ :=
  let freshwater_aquariums : ℕ := 52
  let full_freshwater_aquariums : ℕ := 38
  let animals_per_full_freshwater : ℕ := 64
  let total_freshwater_animals : ℕ := 6310
  let saltwater_aquariums : ℕ := 28
  let full_saltwater_aquariums : ℕ := 18
  let animals_per_full_saltwater : ℕ := 52
  let min_animals_per_saltwater : ℕ := 20
  
  let full_saltwater_animals : ℕ := full_saltwater_aquariums * animals_per_full_saltwater
  let min_remaining_saltwater_animals : ℕ := (saltwater_aquariums - full_saltwater_aquariums) * min_animals_per_saltwater
  
  full_saltwater_animals + min_remaining_saltwater_animals

theorem min_saltwater_animals_is_1136 : min_saltwater_animals = 1136 := by
  sorry

end NUMINAMATH_CALUDE_min_saltwater_animals_is_1136_l3032_303202


namespace NUMINAMATH_CALUDE_quadratic_root_difference_sum_l3032_303209

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 7 * x - 10 = 0

-- Define the condition for m (positive integer not divisible by the square of any prime)
def is_squarefree (m : ℕ) : Prop :=
  m > 0 ∧ ∀ p : ℕ, Prime p → (p^2 ∣ m → False)

-- Main theorem
theorem quadratic_root_difference_sum (m n : ℤ) : 
  (∃ r₁ r₂ : ℝ, quadratic_equation r₁ ∧ quadratic_equation r₂ ∧ |r₁ - r₂| = (Real.sqrt (m : ℝ)) / (n : ℝ)) →
  is_squarefree (m.natAbs) →
  m + n = 254 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_root_difference_sum_l3032_303209


namespace NUMINAMATH_CALUDE_andrew_grape_purchase_l3032_303243

/-- The amount of grapes Andrew purchased in kg -/
def G : ℝ := by sorry

/-- The price of grapes per kg -/
def grape_price : ℝ := 70

/-- The amount of mangoes Andrew purchased in kg -/
def mango_amount : ℝ := 9

/-- The price of mangoes per kg -/
def mango_price : ℝ := 55

/-- The total amount Andrew paid -/
def total_paid : ℝ := 1055

theorem andrew_grape_purchase :
  G * grape_price + mango_amount * mango_price = total_paid ∧ G = 8 := by sorry

end NUMINAMATH_CALUDE_andrew_grape_purchase_l3032_303243


namespace NUMINAMATH_CALUDE_total_cars_is_32_l3032_303221

/-- The number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ

/-- The conditions of car ownership -/
def car_ownership_conditions (c : CarOwnership) : Prop :=
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.cathy = 5

/-- The total number of cars owned by all four people -/
def total_cars (c : CarOwnership) : ℕ :=
  c.cathy + c.lindsey + c.carol + c.susan

/-- Theorem stating that the total number of cars is 32 -/
theorem total_cars_is_32 (c : CarOwnership) (h : car_ownership_conditions c) :
  total_cars c = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_32_l3032_303221


namespace NUMINAMATH_CALUDE_mark_households_visited_mark_collection_proof_l3032_303256

theorem mark_households_visited (days : ℕ) (total_collected : ℕ) (donation : ℕ) : ℕ :=
  let households_per_day := 20
  have days_collecting := 5
  have half_households_donate := households_per_day / 2
  have donation_amount := 2 * 20
  have total_collected_calculated := days_collecting * half_households_donate * donation_amount
  households_per_day

theorem mark_collection_proof 
  (days : ℕ) 
  (total_collected : ℕ) 
  (donation : ℕ) 
  (h1 : days = 5) 
  (h2 : donation = 2 * 20) 
  (h3 : total_collected = 2000) :
  mark_households_visited days total_collected donation = 20 := by
  sorry

end NUMINAMATH_CALUDE_mark_households_visited_mark_collection_proof_l3032_303256


namespace NUMINAMATH_CALUDE_pencil_distribution_l3032_303206

theorem pencil_distribution (n : Nat) (k : Nat) : 
  n = 6 → k = 3 → (Nat.choose (n - k + k - 1) (k - 1)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3032_303206


namespace NUMINAMATH_CALUDE_arrangement_count_l3032_303232

/-- The number of representatives in unit A -/
def unitA : ℕ := 7

/-- The number of representatives in unit B -/
def unitB : ℕ := 3

/-- The total number of elements to arrange (treating unit B as one element) -/
def totalElements : ℕ := unitA + 1

/-- The number of possible arrangements -/
def numArrangements : ℕ := (Nat.factorial totalElements) * (Nat.factorial unitB)

theorem arrangement_count : numArrangements = 241920 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3032_303232


namespace NUMINAMATH_CALUDE_tim_keys_needed_l3032_303242

/-- Calculates the total number of keys needed for apartment complexes -/
def total_keys (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_lock : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_lock

/-- Proves that for Tim's specific case, the total number of keys needed is 72 -/
theorem tim_keys_needed :
  total_keys 2 12 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tim_keys_needed_l3032_303242


namespace NUMINAMATH_CALUDE_oliver_quarters_problem_l3032_303254

theorem oliver_quarters_problem (initial_cash : ℝ) (quarters_given : ℕ) (final_amount : ℝ) :
  initial_cash = 40 →
  quarters_given = 120 →
  final_amount = 55 →
  ∃ (Q : ℕ), 
    (initial_cash + 0.25 * Q) - (5 + 0.25 * quarters_given) = final_amount ∧
    Q = 200 :=
by sorry

end NUMINAMATH_CALUDE_oliver_quarters_problem_l3032_303254
