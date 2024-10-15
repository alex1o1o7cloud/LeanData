import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l4012_401235

theorem problem_statement (a b : ℝ) (h : 2*a + b + 1 = 0) : 1 + 4*a + 2*b = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4012_401235


namespace NUMINAMATH_CALUDE_music_class_size_l4012_401268

theorem music_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_music_class_size_l4012_401268


namespace NUMINAMATH_CALUDE_product_of_square_roots_equals_one_l4012_401295

theorem product_of_square_roots_equals_one :
  let P := Real.sqrt 2012 + Real.sqrt 2013
  let Q := -Real.sqrt 2012 - Real.sqrt 2013
  let R := Real.sqrt 2012 - Real.sqrt 2013
  let S := Real.sqrt 2013 - Real.sqrt 2012
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_equals_one_l4012_401295


namespace NUMINAMATH_CALUDE_rain_probability_l4012_401225

theorem rain_probability (monday_rain : ℝ) (tuesday_rain : ℝ) (no_rain : ℝ)
  (h1 : monday_rain = 0.7)
  (h2 : tuesday_rain = 0.5)
  (h3 : no_rain = 0.2) :
  monday_rain + tuesday_rain - (1 - no_rain) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l4012_401225


namespace NUMINAMATH_CALUDE_length_ae_is_10_l4012_401282

/-- A quadrilateral with the properties of an isosceles trapezoid and a rectangle -/
structure QuadrilateralABCDE where
  /-- AB is a side of the quadrilateral -/
  ab : ℝ
  /-- EC is a side of the quadrilateral -/
  ec : ℝ
  /-- ABCE is an isosceles trapezoid -/
  abce_isosceles_trapezoid : Bool
  /-- ACDE is a rectangle -/
  acde_rectangle : Bool

/-- The length of AE in the quadrilateral ABCDE -/
def length_ae (q : QuadrilateralABCDE) : ℝ :=
  sorry

/-- Theorem stating that the length of AE is 10 under given conditions -/
theorem length_ae_is_10 (q : QuadrilateralABCDE) 
  (h1 : q.ab = 10) 
  (h2 : q.ec = 20) 
  (h3 : q.abce_isosceles_trapezoid = true) 
  (h4 : q.acde_rectangle = true) : 
  length_ae q = 10 :=
sorry

end NUMINAMATH_CALUDE_length_ae_is_10_l4012_401282


namespace NUMINAMATH_CALUDE_area_common_to_translated_triangles_l4012_401258

theorem area_common_to_translated_triangles : 
  let hypotenuse : ℝ := 10
  let translation : ℝ := 2
  let short_leg : ℝ := hypotenuse / 2
  let long_leg : ℝ := short_leg * Real.sqrt 3
  let overlap_height : ℝ := long_leg - translation
  let common_area : ℝ := (1 / 2) * hypotenuse * overlap_height
  common_area = 25 * Real.sqrt 3 - 10 := by
sorry

end NUMINAMATH_CALUDE_area_common_to_translated_triangles_l4012_401258


namespace NUMINAMATH_CALUDE_same_color_probability_l4012_401234

/-- The probability of drawing two balls of the same color from a box of 6 balls -/
theorem same_color_probability : ℝ := by
  -- Define the number of balls of each color
  let red_balls : ℕ := 3
  let yellow_balls : ℕ := 2
  let blue_balls : ℕ := 1

  -- Define the total number of balls
  let total_balls : ℕ := red_balls + yellow_balls + blue_balls

  -- Define the probability of drawing two balls of the same color
  let prob : ℝ := 4 / 15

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l4012_401234


namespace NUMINAMATH_CALUDE_ratio_S4_a3_l4012_401233

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℚ := 2^n - 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℚ := S n - S (n-1)

/-- The theorem to prove -/
theorem ratio_S4_a3 : S 4 / a 3 = 15/4 := by sorry

end NUMINAMATH_CALUDE_ratio_S4_a3_l4012_401233


namespace NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_plus_one_equals_negative_three_l4012_401222

theorem x_squared_minus_four_y_squared_plus_one_equals_negative_three 
  (x y : ℝ) (h1 : x + 2*y = 4) (h2 : x - 2*y = -1) : 
  x^2 - 4*y^2 + 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_plus_one_equals_negative_three_l4012_401222


namespace NUMINAMATH_CALUDE_ab_value_l4012_401241

theorem ab_value (a b c : ℝ) 
  (eq1 : a - b = 5)
  (eq2 : a^2 + b^2 = 34)
  (eq3 : a^3 - b^3 = 30)
  (eq4 : a^2 + b^2 - c^2 = 50) : 
  a * b = 4.5 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l4012_401241


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l4012_401280

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l4012_401280


namespace NUMINAMATH_CALUDE_cube_cutting_l4012_401220

theorem cube_cutting (n : ℕ) : n > 0 → 6 * (n - 2)^2 = 54 → n^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l4012_401220


namespace NUMINAMATH_CALUDE_number_difference_l4012_401269

theorem number_difference (a b : ℕ) (h1 : b = 10 * a) (h2 : a + b = 30000) :
  b - a = 24543 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4012_401269


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4012_401294

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- Define the second inequality
def g (a b c : ℝ) (x : ℝ) := a * (x^2 + 1) + b * (x + 1) + c - 3 * a * x

theorem quadratic_inequality_solution 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : solution_set a b c = Set.Ioo (-2) 1) :
  {x : ℝ | g a b c x < 0} = Set.Iic 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4012_401294


namespace NUMINAMATH_CALUDE_work_completion_time_equivalence_l4012_401267

/-- Represents the work rate of a single worker per day -/
def work_rate : ℝ := 1

/-- Calculates the total work done given the number of workers, work rate, and days -/
def work_done (workers : ℕ) (rate : ℝ) (days : ℕ) : ℝ :=
  (workers : ℝ) * rate * (days : ℝ)

/-- Theorem stating that if the work is completed in 40 days with varying workforce,
    it would take 45 days with a constant workforce -/
theorem work_completion_time_equivalence :
  let total_work := work_done 100 work_rate 35 + work_done 200 work_rate 5
  ∃ (days : ℕ), days = 45 ∧ work_done 100 work_rate days = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_equivalence_l4012_401267


namespace NUMINAMATH_CALUDE_function_monotonicity_l4012_401204

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_monotonicity (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : periodic_two f)
  (h3 : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_l4012_401204


namespace NUMINAMATH_CALUDE_square_of_sum_l4012_401293

theorem square_of_sum (x y : ℝ) : (x + 2*y)^2 = x^2 + 4*x*y + 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l4012_401293


namespace NUMINAMATH_CALUDE_vector_expression_not_equal_AD_l4012_401205

/-- Given vectors in a plane or space, prove that the expression
    (MB + AD) - BM is not equal to AD. -/
theorem vector_expression_not_equal_AD
  (A B C D M O : EuclideanSpace ℝ (Fin n)) :
  (M - B + (A - D)) - (B - M) ≠ A - D := by sorry

end NUMINAMATH_CALUDE_vector_expression_not_equal_AD_l4012_401205


namespace NUMINAMATH_CALUDE_melanie_dimes_given_to_dad_l4012_401246

theorem melanie_dimes_given_to_dad (initial_dimes : ℕ) (dimes_from_mother : ℕ) (final_dimes : ℕ) :
  initial_dimes = 7 →
  dimes_from_mother = 4 →
  final_dimes = 3 →
  initial_dimes + dimes_from_mother - final_dimes = 8 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_given_to_dad_l4012_401246


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l4012_401278

theorem quadratic_coefficient_sum (k : ℤ) : 
  (∃ x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + k*x + 25 = 0 ∧ y^2 + k*y + 25 = 0) → 
  k = 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l4012_401278


namespace NUMINAMATH_CALUDE_digital_earth_technologies_l4012_401262

-- Define the set of all possible technologies
def AllTechnologies : Set String :=
  {"Sustainable development", "Global positioning technology", "Geographic information system",
   "Global positioning system", "Virtual technology", "Network technology"}

-- Define the digital Earth as a complex computer technology system
structure DigitalEarth where
  technologies : Set String
  complex : Bool
  integrates_various_tech : Bool

-- Define the supporting technologies for the digital Earth
def SupportingTechnologies (de : DigitalEarth) : Set String := de.technologies

-- Theorem statement
theorem digital_earth_technologies (de : DigitalEarth) 
  (h1 : de.complex = true) 
  (h2 : de.integrates_various_tech = true) : 
  SupportingTechnologies de = AllTechnologies := by
  sorry

end NUMINAMATH_CALUDE_digital_earth_technologies_l4012_401262


namespace NUMINAMATH_CALUDE_nell_card_collection_l4012_401281

/-- Represents the number and types of cards Nell has --/
structure CardCollection where
  baseball : ℕ
  ace : ℕ
  pokemon : ℕ

/-- Represents the initial state of Nell's card collection --/
def initial_collection : CardCollection := {
  baseball := 438,
  ace := 18,
  pokemon := 312
}

/-- Represents the state of Nell's card collection after giving away cards --/
def after_giveaway (c : CardCollection) : CardCollection := {
  baseball := c.baseball - c.baseball / 2,
  ace := c.ace - c.ace / 3,
  pokemon := c.pokemon
}

/-- Represents the final state of Nell's card collection after trading --/
def final_collection (c : CardCollection) : CardCollection := {
  baseball := c.baseball,
  ace := c.ace + 37,
  pokemon := c.pokemon - 52
}

/-- The main theorem to prove --/
theorem nell_card_collection :
  let final := final_collection (after_giveaway initial_collection)
  (final.baseball - final.ace = 170) ∧
  (final.baseball : ℚ) / 219 = (final.ace : ℚ) / 49 ∧
  (final.ace : ℚ) / 49 = (final.pokemon : ℚ) / 260 := by
  sorry


end NUMINAMATH_CALUDE_nell_card_collection_l4012_401281


namespace NUMINAMATH_CALUDE_distinct_points_difference_l4012_401291

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop := y^2 + x^4 = 3 * x^2 * y + 2

-- Define the constant e
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem distinct_points_difference (a b : ℝ) 
  (ha : graph_equation (Real.sqrt e) a)
  (hb : graph_equation (Real.sqrt e) b)
  (hab : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 8) := by sorry

end NUMINAMATH_CALUDE_distinct_points_difference_l4012_401291


namespace NUMINAMATH_CALUDE_joan_books_count_l4012_401275

/-- The number of books Joan sold in the yard sale -/
def books_sold : ℕ := 26

/-- The number of books Joan has left after the sale -/
def books_left : ℕ := 7

/-- The total number of books Joan gathered to sell -/
def total_books : ℕ := books_sold + books_left

theorem joan_books_count : total_books = 33 := by sorry

end NUMINAMATH_CALUDE_joan_books_count_l4012_401275


namespace NUMINAMATH_CALUDE_profit_percentage_l4012_401208

theorem profit_percentage (C S : ℝ) (h : 19 * C = 16 * S) : 
  (S - C) / C * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l4012_401208


namespace NUMINAMATH_CALUDE_square_side_length_l4012_401215

theorem square_side_length (r s : ℕ) : 
  (2*r + s = 2000) →
  (2*r + 5*s = 3030) →
  s = 258 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l4012_401215


namespace NUMINAMATH_CALUDE_a_completion_time_l4012_401264

def job_completion_time (a b c : ℝ) : Prop :=
  (1 / b = 8) ∧ 
  (1 / c = 12) ∧ 
  (2340 / (1 / a + 1 / b + 1 / c) = 780 / (1 / b))

theorem a_completion_time (a b c : ℝ) : 
  job_completion_time a b c → 1 / a = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_completion_time_l4012_401264


namespace NUMINAMATH_CALUDE_greatest_3digit_base7_divisible_by_7_l4012_401265

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (a b c : Nat) : Nat :=
  a * 7^2 + b * 7 + c

/-- Checks if a number is a valid 3-digit base 7 number --/
def isValidBase7 (a b c : Nat) : Prop :=
  a > 0 ∧ a < 7 ∧ b < 7 ∧ c < 7

/-- The proposed solution in base 7 --/
def solution : (Nat × Nat × Nat) := (6, 6, 0)

theorem greatest_3digit_base7_divisible_by_7 :
  let (a, b, c) := solution
  isValidBase7 a b c ∧
  base7ToBase10 a b c % 7 = 0 ∧
  ∀ x y z, isValidBase7 x y z → 
    base7ToBase10 x y z % 7 = 0 → 
    base7ToBase10 x y z ≤ base7ToBase10 a b c :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base7_divisible_by_7_l4012_401265


namespace NUMINAMATH_CALUDE_complex_squared_i_positive_l4012_401244

theorem complex_squared_i_positive (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (Complex.I * (a + Complex.I)^2 = x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_squared_i_positive_l4012_401244


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4012_401217

theorem complex_equation_solution :
  ∃ z : ℂ, (4 : ℂ) - 2 * Complex.I * z = 3 + 5 * Complex.I * z ∧ z = (1 / 7 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4012_401217


namespace NUMINAMATH_CALUDE_smallest_square_tiling_l4012_401206

/-- The smallest square perfectly tiled by 3x4 rectangles -/
def smallest_tiled_square : ℕ := 12

/-- The number of 3x4 rectangles needed to tile the smallest square -/
def num_rectangles : ℕ := 9

/-- The area of a 3x4 rectangle -/
def rectangle_area : ℕ := 3 * 4

theorem smallest_square_tiling :
  (smallest_tiled_square * smallest_tiled_square) % rectangle_area = 0 ∧
  num_rectangles * rectangle_area = smallest_tiled_square * smallest_tiled_square ∧
  ∀ n : ℕ, n < smallest_tiled_square → (n * n) % rectangle_area ≠ 0 := by
  sorry

#check smallest_square_tiling

end NUMINAMATH_CALUDE_smallest_square_tiling_l4012_401206


namespace NUMINAMATH_CALUDE_circle_properties_l4012_401299

/-- Given a circle with area 16π, prove its diameter is 8 and circumference is 8π -/
theorem circle_properties (r : ℝ) (h : π * r^2 = 16 * π) :
  2 * r = 8 ∧ 2 * π * r = 8 * π := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l4012_401299


namespace NUMINAMATH_CALUDE_exists_isosceles_right_triangle_same_color_l4012_401229

/-- A color type with three possible values -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def ColoringFunction := GridPoint → Color

/-- An isosceles right triangle in the grid -/
structure IsoscelesRightTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  is_isosceles : (a.x - b.x)^2 + (a.y - b.y)^2 = (a.x - c.x)^2 + (a.y - c.y)^2
  is_right : (b.x - c.x) * (a.x - c.x) + (b.y - c.y) * (a.y - c.y) = 0

/-- The main theorem: There exists an isosceles right triangle with vertices of the same color -/
theorem exists_isosceles_right_triangle_same_color (coloring : ColoringFunction) :
  ∃ (t : IsoscelesRightTriangle), coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end NUMINAMATH_CALUDE_exists_isosceles_right_triangle_same_color_l4012_401229


namespace NUMINAMATH_CALUDE_congruence_problem_l4012_401240

theorem congruence_problem (y : ℤ) 
  (h1 : (4 + y) % (2^4) = 3^2 % (2^4))
  (h2 : (6 + y) % (3^4) = 2^3 % (3^4))
  (h3 : (8 + y) % (5^4) = 7^2 % (5^4)) :
  y % 360 = 317 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4012_401240


namespace NUMINAMATH_CALUDE_toms_remaining_balloons_l4012_401228

/-- Theorem: Tom's remaining violet balloons -/
theorem toms_remaining_balloons (initial_balloons : ℕ) (given_balloons : ℕ) 
  (h1 : initial_balloons = 30)
  (h2 : given_balloons = 16) :
  initial_balloons - given_balloons = 14 := by
  sorry

end NUMINAMATH_CALUDE_toms_remaining_balloons_l4012_401228


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_2_intersection_complement_empty_iff_l4012_401226

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3*m - 4 ∨ x ≥ 8 + m}

-- Theorem for part 1
theorem intersection_complement_when_m_2 :
  A ∩ (Set.univ \ B 2) = {x | 2 < x ∧ x < 4} := by sorry

-- Theorem for part 2
theorem intersection_complement_empty_iff (m : ℝ) :
  m < 6 →
  (A ∩ (Set.univ \ B m) = ∅ ↔ m ≤ -7 ∨ (8/3 ≤ m ∧ m < 6)) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_2_intersection_complement_empty_iff_l4012_401226


namespace NUMINAMATH_CALUDE_min_abs_z_l4012_401249

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2) + Complex.abs (z - 7*I) = 10) :
  Complex.abs z ≥ 1.4 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l4012_401249


namespace NUMINAMATH_CALUDE_log_4_30_l4012_401283

theorem log_4_30 (a c : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 5 / Real.log 10 = c) :
  Real.log 30 / Real.log 4 = 1 / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_log_4_30_l4012_401283


namespace NUMINAMATH_CALUDE_dividend_calculation_l4012_401253

theorem dividend_calculation (quotient divisor remainder : ℕ) : 
  quotient = 15000 → 
  divisor = 82675 → 
  remainder = 57801 → 
  quotient * divisor + remainder = 1240182801 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4012_401253


namespace NUMINAMATH_CALUDE_painting_difference_l4012_401238

/-- Represents a 5x5x5 cube -/
structure Cube :=
  (size : Nat)
  (h_size : size = 5)

/-- Counts the number of unit cubes with at least one painted face when two opposite faces and one additional face are painted -/
def count_painted_opposite_plus_one (c : Cube) : Nat :=
  c.size * c.size + (c.size - 2) * c.size + c.size * c.size

/-- Counts the number of unit cubes with at least one painted face when three adjacent faces sharing one vertex are painted -/
def count_painted_adjacent (c : Cube) : Nat :=
  (c.size - 1) * 9 + c.size * c.size

/-- The difference between the two painting configurations is 4 -/
theorem painting_difference (c : Cube) : 
  count_painted_opposite_plus_one c - count_painted_adjacent c = 4 := by
  sorry


end NUMINAMATH_CALUDE_painting_difference_l4012_401238


namespace NUMINAMATH_CALUDE_marble_difference_l4012_401245

theorem marble_difference : ∀ (total_marbles : ℕ),
  -- Conditions
  (total_marbles > 0) →  -- Ensure there are marbles
  (∃ (blue1 green1 blue2 green2 : ℕ),
    -- Jar 1 ratio
    7 * green1 = 3 * blue1 ∧
    -- Jar 2 ratio
    5 * green2 = 4 * blue2 ∧
    -- Same total in each jar
    blue1 + green1 = blue2 + green2 ∧
    -- Total green marbles
    green1 + green2 = 140 ∧
    -- Total marbles in each jar
    blue1 + green1 = total_marbles) →
  -- Conclusion
  ∃ (blue1 blue2 : ℕ), blue1 - blue2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l4012_401245


namespace NUMINAMATH_CALUDE_fraction_problem_l4012_401298

theorem fraction_problem (x : ℚ) :
  (x / (4 * x + 5) = 3 / 7) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4012_401298


namespace NUMINAMATH_CALUDE_imaginary_difference_condition_l4012_401212

def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_difference_condition (z₁ z₂ : ℂ) :
  (is_imaginary (z₁ - z₂) → (is_imaginary z₁ ∨ is_imaginary z₂)) ∧
  ∃ z₁ z₂ : ℂ, (is_imaginary z₁ ∨ is_imaginary z₂) ∧ ¬is_imaginary (z₁ - z₂) :=
sorry

end NUMINAMATH_CALUDE_imaginary_difference_condition_l4012_401212


namespace NUMINAMATH_CALUDE_taco_truck_profit_l4012_401297

/-- Calculate the profit for a taco truck given the total beef, beef per taco, selling price, and cost to make. -/
theorem taco_truck_profit
  (total_beef : ℝ)
  (beef_per_taco : ℝ)
  (selling_price : ℝ)
  (cost_to_make : ℝ)
  (h1 : total_beef = 100)
  (h2 : beef_per_taco = 0.25)
  (h3 : selling_price = 2)
  (h4 : cost_to_make = 1.5) :
  (total_beef / beef_per_taco) * (selling_price - cost_to_make) = 200 :=
by
  sorry

#check taco_truck_profit

end NUMINAMATH_CALUDE_taco_truck_profit_l4012_401297


namespace NUMINAMATH_CALUDE_largest_two_digit_one_less_multiple_l4012_401296

theorem largest_two_digit_one_less_multiple : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n + 1 = 60 * k) ∧
  (∀ m : ℕ, m > n → m < 100 → ¬∃ j : ℕ, m + 1 = 60 * j) ∧
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_one_less_multiple_l4012_401296


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l4012_401260

/-- Conversion from polar to rectangular coordinates -/
theorem polar_to_rectangular (r θ : ℝ) :
  r > 0 →
  θ = 11 * π / 6 →
  (r * Real.cos θ, r * Real.sin θ) = (5 * Real.sqrt 3, -5) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l4012_401260


namespace NUMINAMATH_CALUDE_average_speed_two_part_trip_l4012_401263

/-- Calculates the average speed of a two-part trip -/
theorem average_speed_two_part_trip
  (total_distance : ℝ)
  (distance1 : ℝ)
  (speed1 : ℝ)
  (distance2 : ℝ)
  (speed2 : ℝ)
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 35)
  (h3 : distance2 = 35)
  (h4 : speed1 = 48)
  (h5 : speed2 = 24)
  (h6 : total_distance = 70) :
  ∃ (avg_speed : ℝ), abs (avg_speed - 32) < 0.1 ∧
  avg_speed = total_distance / (distance1 / speed1 + distance2 / speed2) := by
  sorry


end NUMINAMATH_CALUDE_average_speed_two_part_trip_l4012_401263


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l4012_401290

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 210 →
  B = 330 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l4012_401290


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l4012_401250

-- Define set M
def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}

-- Define set N
def N : Set ℝ := {x : ℝ | x < -5 ∨ x > 5}

-- Theorem statement
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < -5 ∨ x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l4012_401250


namespace NUMINAMATH_CALUDE_song_time_is_125_minutes_l4012_401247

/-- Represents the duration of a radio show in minutes -/
def total_show_time : ℕ := 3 * 60

/-- Represents the duration of a single talking segment in minutes -/
def talking_segment_duration : ℕ := 10

/-- Represents the duration of a single ad break in minutes -/
def ad_break_duration : ℕ := 5

/-- Represents the number of talking segments in the show -/
def num_talking_segments : ℕ := 3

/-- Represents the number of ad breaks in the show -/
def num_ad_breaks : ℕ := 5

/-- Calculates the total time spent on talking segments -/
def total_talking_time : ℕ := talking_segment_duration * num_talking_segments

/-- Calculates the total time spent on ad breaks -/
def total_ad_time : ℕ := ad_break_duration * num_ad_breaks

/-- Calculates the total time spent on non-song content -/
def total_non_song_time : ℕ := total_talking_time + total_ad_time

/-- Theorem: The remaining time for songs in the radio show is 125 minutes -/
theorem song_time_is_125_minutes : 
  total_show_time - total_non_song_time = 125 := by sorry

end NUMINAMATH_CALUDE_song_time_is_125_minutes_l4012_401247


namespace NUMINAMATH_CALUDE_power_calculation_l4012_401255

theorem power_calculation : 2^300 + 9^3 / 9^2 - 3^4 = 2^300 - 72 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l4012_401255


namespace NUMINAMATH_CALUDE_expression_evaluation_l4012_401224

theorem expression_evaluation : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4012_401224


namespace NUMINAMATH_CALUDE_inequality_solution_l4012_401202

theorem inequality_solution (x : ℝ) : 
  3 - 2 / (3 * x + 2) < 5 ↔ x < -1 ∨ x > -2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4012_401202


namespace NUMINAMATH_CALUDE_complex_polynomial_roots_l4012_401201

theorem complex_polynomial_roots (c : ℂ) : 
  (∃ (P : ℂ → ℂ), P = (fun x ↦ (x^2 - 2*x + 2) * (x^2 - c*x + 4) * (x^2 - 4*x + 8)) ∧ 
   (∃ (r1 r2 r3 r4 : ℂ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧
    ∀ x, P x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4)) →
  Complex.abs c = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_polynomial_roots_l4012_401201


namespace NUMINAMATH_CALUDE_max_ballpoint_pens_l4012_401273

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- The cost of each type of pen in rubles -/
def penCosts : PenCounts := { ballpoint := 10, gel := 30, fountain := 60 }

/-- The total cost of a given combination of pens -/
def totalCost (counts : PenCounts) : ℕ :=
  counts.ballpoint * penCosts.ballpoint +
  counts.gel * penCosts.gel +
  counts.fountain * penCosts.fountain

/-- The total number of pens -/
def totalPens (counts : PenCounts) : ℕ :=
  counts.ballpoint + counts.gel + counts.fountain

/-- Predicate for a valid pen combination -/
def isValidCombination (counts : PenCounts) : Prop :=
  totalPens counts = 20 ∧
  totalCost counts = 500 ∧
  counts.ballpoint > 0 ∧
  counts.gel > 0 ∧
  counts.fountain > 0

/-- Theorem: The maximum number of ballpoint pens is 11 -/
theorem max_ballpoint_pens :
  ∃ (counts : PenCounts), isValidCombination counts ∧
    counts.ballpoint = 11 ∧
    ∀ (other : PenCounts), isValidCombination other →
      other.ballpoint ≤ counts.ballpoint :=
by sorry

end NUMINAMATH_CALUDE_max_ballpoint_pens_l4012_401273


namespace NUMINAMATH_CALUDE_chocolate_cake_price_is_12_l4012_401214

/-- The price of a chocolate cake given the order details and total payment -/
def chocolate_cake_price (num_chocolate : ℕ) (num_strawberry : ℕ) (strawberry_price : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment - num_strawberry * strawberry_price) / num_chocolate

theorem chocolate_cake_price_is_12 :
  chocolate_cake_price 3 6 22 168 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cake_price_is_12_l4012_401214


namespace NUMINAMATH_CALUDE_gary_money_calculation_l4012_401259

/-- Calculates Gary's final amount of money after a series of transactions -/
def gary_final_amount (initial_amount snake_sale_price hamster_cost supplies_cost : ℝ) : ℝ :=
  initial_amount + snake_sale_price - hamster_cost - supplies_cost

/-- Theorem stating that Gary's final amount is 90.60 dollars -/
theorem gary_money_calculation :
  gary_final_amount 73.25 55.50 25.75 12.40 = 90.60 := by
  sorry

end NUMINAMATH_CALUDE_gary_money_calculation_l4012_401259


namespace NUMINAMATH_CALUDE_max_abs_quadratic_on_interval_l4012_401231

/-- The function f(x) = |x^2 - 2x - t| with maximum value 2 on [0, 3] implies t = 1 -/
theorem max_abs_quadratic_on_interval (t : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| = 2) →
  t = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_quadratic_on_interval_l4012_401231


namespace NUMINAMATH_CALUDE_min_coins_for_alex_l4012_401213

/-- The minimum number of additional coins needed for distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for the given scenario. -/
theorem min_coins_for_alex : min_additional_coins 15 63 = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_for_alex_l4012_401213


namespace NUMINAMATH_CALUDE_cistern_theorem_l4012_401257

/-- Represents the cistern problem -/
def cistern_problem (capacity : ℝ) (leak_time : ℝ) (tap_rate : ℝ) : Prop :=
  let leak_rate : ℝ := capacity / leak_time
  let net_rate : ℝ := leak_rate - tap_rate
  let emptying_time : ℝ := capacity / net_rate
  emptying_time = 24

/-- The theorem statement for the cistern problem -/
theorem cistern_theorem :
  cistern_problem 480 20 4 := by sorry

end NUMINAMATH_CALUDE_cistern_theorem_l4012_401257


namespace NUMINAMATH_CALUDE_edric_work_hours_l4012_401239

/-- Calculates the number of hours worked per day given monthly salary, days worked per week, and hourly rate -/
def hours_per_day (monthly_salary : ℕ) (days_per_week : ℕ) (hourly_rate : ℕ) : ℕ :=
  let days_per_month := days_per_week * 4
  let total_hours := monthly_salary / hourly_rate
  total_hours / days_per_month

theorem edric_work_hours :
  hours_per_day 576 6 3 = 8 := by
  sorry

#eval hours_per_day 576 6 3

end NUMINAMATH_CALUDE_edric_work_hours_l4012_401239


namespace NUMINAMATH_CALUDE_solution_set_inequality_l4012_401254

theorem solution_set_inequality (x : ℝ) : 
  (((2 * x - 1) / (x + 2)) > 1) ↔ (x < -2 ∨ x > 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l4012_401254


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l4012_401274

theorem polynomial_identity_sum (a b c d e f : ℤ) :
  (∀ x : ℤ, (3 * x + 1)^5 = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) →
  a - b + c - d + e - f = 32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l4012_401274


namespace NUMINAMATH_CALUDE_orange_box_problem_l4012_401203

theorem orange_box_problem (box1_capacity box2_capacity : ℕ) 
  (box1_fill_fraction : ℚ) (total_oranges : ℕ) :
  box1_capacity = 80 →
  box2_capacity = 50 →
  box1_fill_fraction = 3/4 →
  total_oranges = 90 →
  ∃ (box2_fill_fraction : ℚ),
    box2_fill_fraction = 3/5 ∧
    (box1_capacity : ℚ) * box1_fill_fraction + (box2_capacity : ℚ) * box2_fill_fraction = total_oranges := by
  sorry

end NUMINAMATH_CALUDE_orange_box_problem_l4012_401203


namespace NUMINAMATH_CALUDE_min_value_expression_l4012_401284

theorem min_value_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a + 2 * b = 1) :
  ∃ (m : ℝ), m = 2/3 ∧ ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 3 * x + 2 * y = 1 → 
    1 / (12 * x + 1) + 1 / (8 * y + 1) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4012_401284


namespace NUMINAMATH_CALUDE_nickel_difference_l4012_401271

/-- Given that Alice has 3p + 2 nickels and Bob has 2p + 6 nickels,
    the difference in their money in pennies is 5p - 20 --/
theorem nickel_difference (p : ℤ) : 
  let alice_nickels : ℤ := 3 * p + 2
  let bob_nickels : ℤ := 2 * p + 6
  let nickel_value : ℤ := 5  -- value of a nickel in pennies
  5 * p - 20 = nickel_value * (alice_nickels - bob_nickels) :=
by sorry

end NUMINAMATH_CALUDE_nickel_difference_l4012_401271


namespace NUMINAMATH_CALUDE_line_slope_l4012_401276

theorem line_slope (x y : ℝ) : 3 * y = 4 * x - 12 → (y - (-4)) / (x - 0) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l4012_401276


namespace NUMINAMATH_CALUDE_problem_solution_l4012_401248

def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 2|

theorem problem_solution :
  (∃ (M : ℝ), (∀ x, f x ≥ M) ∧ (∃ x, f x = M) ∧ M = 3) ∧
  (∀ x, f x < 3 + |2*x + 2| ↔ -1 < x ∧ x < 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 = 3 → 2*a + b ≤ 3*Real.sqrt 6 / 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 = 3 ∧ 2*a + b = 3*Real.sqrt 6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4012_401248


namespace NUMINAMATH_CALUDE_prob_exactly_two_of_three_l4012_401261

/-- The probability of exactly two out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_exactly_two_of_three (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/5) (h_B : p_B = 1/4) (h_C : p_C = 1/3) :
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_exactly_two_of_three_l4012_401261


namespace NUMINAMATH_CALUDE_candies_remaining_l4012_401251

def vasya_eat (n : ℕ) : ℕ := n - (1 + (n - 9) / 7)

def petya_eat (n : ℕ) : ℕ := n - (1 + (n - 7) / 9)

theorem candies_remaining (initial_candies : ℕ) : 
  initial_candies = 1000 → petya_eat (vasya_eat initial_candies) = 761 := by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_l4012_401251


namespace NUMINAMATH_CALUDE_cubic_integer_roots_l4012_401287

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of integer roots of a cubic polynomial, counting multiplicity -/
def num_integer_roots (p : CubicPolynomial) : ℕ := sorry

/-- The theorem stating the possible values for the number of integer roots -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_l4012_401287


namespace NUMINAMATH_CALUDE_hall_breadth_proof_l4012_401272

/-- Given a rectangular hall and stones with specified dimensions, 
    prove that the breadth of the hall is 15 meters. -/
theorem hall_breadth_proof (hall_length : ℝ) (stone_length : ℝ) (stone_width : ℝ) 
                            (num_stones : ℕ) :
  hall_length = 36 →
  stone_length = 0.4 →
  stone_width = 0.5 →
  num_stones = 2700 →
  hall_length * (num_stones * stone_length * stone_width / hall_length) = 15 := by
  sorry

#check hall_breadth_proof

end NUMINAMATH_CALUDE_hall_breadth_proof_l4012_401272


namespace NUMINAMATH_CALUDE_shaded_area_square_with_triangles_l4012_401218

/-- The area of the shaded region in a square with two unshaded triangles -/
theorem shaded_area_square_with_triangles : 
  let square_side : ℝ := 50
  let triangle1_base : ℝ := 20
  let triangle1_height : ℝ := 20
  let triangle2_base : ℝ := 20
  let triangle2_height : ℝ := 20
  let square_area := square_side * square_side
  let triangle1_area := (1/2) * triangle1_base * triangle1_height
  let triangle2_area := (1/2) * triangle2_base * triangle2_height
  let total_triangle_area := triangle1_area + triangle2_area
  let shaded_area := square_area - total_triangle_area
  shaded_area = 2100 := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_triangles_l4012_401218


namespace NUMINAMATH_CALUDE_roberts_spending_l4012_401236

theorem roberts_spending (total : ℝ) : 
  total = 100 + 125 + 0.1 * total → total = 250 :=
by sorry

end NUMINAMATH_CALUDE_roberts_spending_l4012_401236


namespace NUMINAMATH_CALUDE_conference_handshakes_l4012_401286

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total_people : ℕ)
  (group1_size : ℕ)
  (group2_size : ℕ)
  (h_total : total_people = group1_size + group2_size)

/-- Calculates the number of handshakes in the conference -/
def handshakes (conf : Conference) : ℕ :=
  let group2_external := conf.group2_size * (conf.total_people - 1)
  let group2_internal := (conf.group2_size * (conf.group2_size - 1)) / 2
  group2_external + group2_internal

/-- Theorem stating the number of handshakes in the specific conference scenario -/
theorem conference_handshakes :
  ∃ (conf : Conference),
    conf.total_people = 50 ∧
    conf.group1_size = 30 ∧
    conf.group2_size = 20 ∧
    handshakes conf = 1170 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l4012_401286


namespace NUMINAMATH_CALUDE_clock_shows_ten_to_five_l4012_401270

/-- Represents a clock hand --/
inductive ClockHand
  | A
  | B
  | C

/-- Represents the position of a clock hand --/
inductive HandPosition
  | ExactHourMark
  | SlightlyOffHourMark

/-- Represents a clock with three hands --/
structure Clock :=
  (hands : Fin 3 → ClockHand)
  (positions : ClockHand → HandPosition)

/-- The time shown on the clock --/
structure Time :=
  (hours : Nat)
  (minutes : Nat)

/-- Checks if the given clock configuration is valid --/
def isValidClock (c : Clock) : Prop :=
  ∃ (h1 h2 : ClockHand), h1 ≠ h2 ∧ 
    c.positions h1 = HandPosition.ExactHourMark ∧
    c.positions h2 = HandPosition.ExactHourMark ∧
    (∀ h, h ≠ h1 → h ≠ h2 → c.positions h = HandPosition.SlightlyOffHourMark)

/-- The main theorem --/
theorem clock_shows_ten_to_five (c : Clock) : 
  isValidClock c → ∃ (t : Time), t.hours = 4 ∧ t.minutes = 50 :=
sorry

end NUMINAMATH_CALUDE_clock_shows_ten_to_five_l4012_401270


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l4012_401252

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ≥ 2, Monotone (f a)) → a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l4012_401252


namespace NUMINAMATH_CALUDE_exists_winning_strategy_for_first_player_l4012_401219

/-- Represents the state of the orange game -/
structure GameState :=
  (oranges : ℕ)
  (player_turn : Bool)

/-- Defines a valid move in the game -/
def valid_move (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 5

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : ℕ) : GameState :=
  { oranges := state.oranges - move
  , player_turn := ¬state.player_turn }

/-- Determines if a game state is winning for the current player -/
def is_winning_state (state : GameState) : Prop :=
  state.oranges = 0

/-- Defines a winning strategy for the game -/
def winning_strategy (strategy : GameState → ℕ) : Prop :=
  ∀ (state : GameState),
    valid_move (strategy state) ∧
    (is_winning_state (apply_move state (strategy state)) ∨
     ∀ (opponent_move : ℕ),
       valid_move opponent_move →
       ¬is_winning_state (apply_move (apply_move state (strategy state)) opponent_move))

/-- Theorem stating that there exists a winning strategy for the first player in the 100-orange game -/
theorem exists_winning_strategy_for_first_player :
  ∃ (strategy : GameState → ℕ),
    winning_strategy strategy ∧
    strategy { oranges := 100, player_turn := true } = 4 :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_for_first_player_l4012_401219


namespace NUMINAMATH_CALUDE_polynomial_identity_l4012_401277

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l4012_401277


namespace NUMINAMATH_CALUDE_triangle_side_length_l4012_401285

/-- Given a triangle ABC with side a = 8, angle B = 30°, and angle C = 105°, 
    prove that the length of side b is equal to 4√2. -/
theorem triangle_side_length (a b : ℝ) (A B C : ℝ) : 
  a = 8 → B = 30 * π / 180 → C = 105 * π / 180 → 
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  b = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4012_401285


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l4012_401256

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmh = 36)
  (h3 : bridge_length = 132) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 24.2 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l4012_401256


namespace NUMINAMATH_CALUDE_gecko_count_l4012_401211

theorem gecko_count : 
  ∀ (gecko_count : ℕ) (lizard_count : ℕ) (insects_per_gecko : ℕ) (total_insects : ℕ),
    lizard_count = 3 →
    insects_per_gecko = 6 →
    total_insects = 66 →
    total_insects = gecko_count * insects_per_gecko + lizard_count * (2 * insects_per_gecko) →
    gecko_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_gecko_count_l4012_401211


namespace NUMINAMATH_CALUDE_dividend_calculation_l4012_401288

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 47.5)
  (h2 : quotient = 24.3)
  (h3 : remainder = 32.4) :
  divisor * quotient + remainder = 1186.15 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4012_401288


namespace NUMINAMATH_CALUDE_triple_equality_l4012_401221

theorem triple_equality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h1 : x * y * (x + y) = y * z * (y + z)) 
  (h2 : y * z * (y + z) = z * x * (z + x)) : 
  (x = y ∧ y = z) ∨ x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_triple_equality_l4012_401221


namespace NUMINAMATH_CALUDE_oil_measurement_l4012_401292

theorem oil_measurement (initial_oil : ℚ) (added_oil : ℚ) (total_oil : ℚ) :
  initial_oil = 17/100 →
  added_oil = 67/100 →
  total_oil = initial_oil + added_oil →
  total_oil = 84/100 := by
sorry

end NUMINAMATH_CALUDE_oil_measurement_l4012_401292


namespace NUMINAMATH_CALUDE_distance_to_hypotenuse_l4012_401237

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- The length of one leg of the triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the triangle -/
  leg2 : ℝ
  /-- The distance from the intersection point of the medians to one leg -/
  dist1 : ℝ
  /-- The distance from the intersection point of the medians to the other leg -/
  dist2 : ℝ
  /-- Ensure the triangle is not degenerate -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2
  /-- The distances from the intersection point to the legs are positive -/
  dist1_pos : 0 < dist1
  dist2_pos : 0 < dist2
  /-- The given distances from the intersection point to the legs -/
  dist1_eq : dist1 = 3
  dist2_eq : dist2 = 4

/-- The theorem to be proved -/
theorem distance_to_hypotenuse (t : RightTriangle) : 
  let hypotenuse := Real.sqrt (t.leg1^2 + t.leg2^2)
  let area := t.leg1 * t.leg2 / 2
  area / hypotenuse = 12/5 := by sorry

end NUMINAMATH_CALUDE_distance_to_hypotenuse_l4012_401237


namespace NUMINAMATH_CALUDE_larger_number_proof_l4012_401289

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1345)
  (h2 : L = 6 * S + 15) : 
  L = 1611 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4012_401289


namespace NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_two_l4012_401210

theorem correct_operation_is_multiplication_by_two (N : ℝ) (x : ℝ) :
  (N / 10 = (5 / 100) * (N * x)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_two_l4012_401210


namespace NUMINAMATH_CALUDE_differential_at_zero_l4012_401216

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 3)

theorem differential_at_zero (x : ℝ) : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_differential_at_zero_l4012_401216


namespace NUMINAMATH_CALUDE_bouquet_calculation_l4012_401209

def max_bouquets (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / flowers_per_bouquet

theorem bouquet_calculation (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) 
  (h1 : total_flowers = 53)
  (h2 : flowers_per_bouquet = 7)
  (h3 : wilted_flowers = 18) :
  max_bouquets total_flowers flowers_per_bouquet wilted_flowers = 5 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_calculation_l4012_401209


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4012_401242

theorem algebraic_expression_value (a b c : ℤ) 
  (h1 : a - b = 3) 
  (h2 : b + c = -5) : 
  a * c - b * c + a^2 - a * b = -6 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4012_401242


namespace NUMINAMATH_CALUDE_max_value_of_f_l4012_401232

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-1) 4 → f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4012_401232


namespace NUMINAMATH_CALUDE_exponential_inequality_l4012_401207

theorem exponential_inequality (a b c : ℝ) : 
  a^b > a^c ∧ a^c > 1 ∧ b < c → b < c ∧ c < 0 ∧ 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l4012_401207


namespace NUMINAMATH_CALUDE_area_between_circle_and_squares_l4012_401279

theorem area_between_circle_and_squares :
  let outer_square_side : ℝ := 2
  let circle_radius : ℝ := 1/2
  let inner_square_side : ℝ := 1.8
  let outer_square_area : ℝ := outer_square_side^2
  let inner_square_area : ℝ := inner_square_side^2
  let circle_area : ℝ := π * circle_radius^2
  let area_between : ℝ := outer_square_area - inner_square_area - (outer_square_area - circle_area)
  area_between = 0.76 := by sorry

end NUMINAMATH_CALUDE_area_between_circle_and_squares_l4012_401279


namespace NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l4012_401230

theorem remainder_777_pow_777_mod_13 : 777^777 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l4012_401230


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l4012_401243

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 3*x^2 - 4*x + 12

-- State the theorem
theorem partial_fraction_decomposition_sum (a b c D E F : ℝ) : 
  -- a, b, c are distinct roots of p
  p a = 0 → p b = 0 → p c = 0 → a ≠ b → b ≠ c → a ≠ c →
  -- Partial fraction decomposition holds
  (∀ s : ℝ, s ≠ a → s ≠ b → s ≠ c → 
    1 / (s^3 - 3*s^2 - 4*s + 12) = D / (s - a) + E / (s - b) + F / (s - c)) →
  -- Conclusion
  1 / D + 1 / E + 1 / F + a * b * c = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l4012_401243


namespace NUMINAMATH_CALUDE_cubic_root_product_l4012_401223

theorem cubic_root_product : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 9 * x^2 + 5 * x - 10
  ∀ a b c : ℝ, f a = 0 → f b = 0 → f c = 0 → a * b * c = 10 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_product_l4012_401223


namespace NUMINAMATH_CALUDE_rectangle_width_three_l4012_401266

/-- A rectangle with length twice its width and area equal to perimeter has width 3. -/
theorem rectangle_width_three (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 6 * w) → w = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_three_l4012_401266


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l4012_401227

theorem fractional_equation_simplification (x : ℝ) (h : x ≠ 2) : 
  (x / (x - 2) - 2 = 3 / (2 - x)) ↔ (x - 2 * (x - 2) = -3) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l4012_401227


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4012_401200

/-- The lateral surface area of a cylinder with a square axial cross-section -/
theorem cylinder_lateral_surface_area (s : ℝ) (h : s = 10) :
  let circumference := s * Real.pi
  let height := s
  height * circumference = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4012_401200
