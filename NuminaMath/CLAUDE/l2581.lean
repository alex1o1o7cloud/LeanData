import Mathlib

namespace NUMINAMATH_CALUDE_servant_service_duration_l2581_258135

/-- Calculates the number of months served given the total yearly payment and the received payment -/
def months_served (total_yearly_payment : ℚ) (received_payment : ℚ) : ℚ :=
  (received_payment / (total_yearly_payment / 12))

/-- Theorem stating that for the given payment conditions, the servant served approximately 6 months -/
theorem servant_service_duration :
  let total_yearly_payment : ℚ := 800
  let received_payment : ℚ := 400
  abs (months_served total_yearly_payment received_payment - 6) < 0.1 := by
sorry

end NUMINAMATH_CALUDE_servant_service_duration_l2581_258135


namespace NUMINAMATH_CALUDE_cube_sum_equality_l2581_258169

theorem cube_sum_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (square_fourth_equality : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) :
  a^3 + b^3 + c^3 = -3*a*b*(a+b) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l2581_258169


namespace NUMINAMATH_CALUDE_cunningham_white_lambs_l2581_258168

/-- The number of white lambs owned by farmer Cunningham -/
def white_lambs (total : ℕ) (black : ℕ) : ℕ := total - black

theorem cunningham_white_lambs :
  white_lambs 6048 5855 = 193 :=
by sorry

end NUMINAMATH_CALUDE_cunningham_white_lambs_l2581_258168


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2581_258132

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x - 2*y + x^2 = 8

/-- The y-intercept of the line -/
def y_intercept : ℝ := -4

/-- Theorem: The y-intercept of the line described by the equation x - 2y + x^2 = 8 is -4 -/
theorem y_intercept_of_line :
  line_equation 0 y_intercept := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2581_258132


namespace NUMINAMATH_CALUDE_cube_coloring_count_l2581_258174

/-- The number of distinct orientations of a cube -/
def cubeOrientations : ℕ := 24

/-- The number of ways to permute 6 colors -/
def colorPermutations : ℕ := 720

/-- The number of distinct ways to paint a cube's faces with 6 different colors,
    where each color appears exactly once and rotations are considered identical -/
def distinctCubeColorings : ℕ := colorPermutations / cubeOrientations

theorem cube_coloring_count :
  distinctCubeColorings = 30 := by sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l2581_258174


namespace NUMINAMATH_CALUDE_lisa_flight_time_l2581_258143

/-- Given a distance of 256 miles and a speed of 32 miles per hour, 
    the time taken to travel this distance is 8 hours. -/
theorem lisa_flight_time : 
  ∀ (distance speed time : ℝ), 
    distance = 256 → 
    speed = 32 → 
    time = distance / speed → 
    time = 8 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_time_l2581_258143


namespace NUMINAMATH_CALUDE_bob_show_dogs_count_l2581_258156

/-- The number of show dogs Bob bought -/
def num_show_dogs : ℕ := 2

/-- The cost of each show dog in dollars -/
def cost_per_show_dog : ℕ := 250

/-- The number of puppies -/
def num_puppies : ℕ := 6

/-- The selling price of each puppy in dollars -/
def price_per_puppy : ℕ := 350

/-- The total profit in dollars -/
def total_profit : ℕ := 1600

theorem bob_show_dogs_count :
  num_puppies * price_per_puppy - num_show_dogs * cost_per_show_dog = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bob_show_dogs_count_l2581_258156


namespace NUMINAMATH_CALUDE_face_mask_selling_price_l2581_258133

theorem face_mask_selling_price
  (num_boxes : ℕ)
  (masks_per_box : ℕ)
  (total_cost : ℚ)
  (total_profit : ℚ)
  (h_num_boxes : num_boxes = 3)
  (h_masks_per_box : masks_per_box = 20)
  (h_total_cost : total_cost = 15)
  (h_total_profit : total_profit = 15) :
  (total_cost + total_profit) / (num_boxes * masks_per_box : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_face_mask_selling_price_l2581_258133


namespace NUMINAMATH_CALUDE_sarah_weed_pulling_l2581_258120

def tuesday_weeds : ℕ := 25

def wednesday_weeds : ℕ := 3 * tuesday_weeds

def thursday_weeds : ℕ := wednesday_weeds / 5

def friday_weeds : ℕ := thursday_weeds - 10

def total_weeds : ℕ := tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds

theorem sarah_weed_pulling :
  total_weeds = 120 :=
sorry

end NUMINAMATH_CALUDE_sarah_weed_pulling_l2581_258120


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l2581_258164

def fruit_survey (apples bananas cherries oranges grapes : ℕ) : Prop :=
  let total := apples + bananas + cherries + oranges + grapes
  let apple_percentage := (apples : ℚ) / (total : ℚ) * 100
  apple_percentage = 26.67

theorem apple_preference_percentage :
  fruit_survey 80 90 50 40 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l2581_258164


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l2581_258103

theorem irrationality_of_sqrt_two_and_rationality_of_others : 
  (∃ (a b : ℤ), (a : ℝ) / (b : ℝ) = Real.sqrt 2) ∧ 
  (∃ (c d : ℤ), (c : ℝ) / (d : ℝ) = 3.14) ∧
  (∃ (e f : ℤ), (e : ℝ) / (f : ℝ) = -2) ∧
  (∃ (g h : ℤ), (g : ℝ) / (h : ℝ) = 1/3) ∧
  (¬∃ (i j : ℤ), (i : ℝ) / (j : ℝ) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l2581_258103


namespace NUMINAMATH_CALUDE_complex_multiplication_l2581_258111

theorem complex_multiplication : (1 + Complex.I) * (2 + Complex.I) * (3 + Complex.I) = 10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2581_258111


namespace NUMINAMATH_CALUDE_min_value_inequality_l2581_258158

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 6) : 1/a^2 + 2/b^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2581_258158


namespace NUMINAMATH_CALUDE_pipe_ratio_l2581_258157

theorem pipe_ratio (total_length shorter_length : ℕ) 
  (h1 : total_length = 177)
  (h2 : shorter_length = 59)
  (h3 : shorter_length < total_length) :
  (total_length - shorter_length) / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_pipe_ratio_l2581_258157


namespace NUMINAMATH_CALUDE_sneakers_price_l2581_258134

/-- Given a pair of sneakers with an unknown original price, if applying a $10 coupon 
followed by a 10% membership discount results in a final price of $99, 
then the original price of the sneakers was $120. -/
theorem sneakers_price (original_price : ℝ) : 
  (original_price - 10) * 0.9 = 99 → original_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_price_l2581_258134


namespace NUMINAMATH_CALUDE_count_solutions_eq_51_l2581_258179

/-- The number of distinct ordered pairs (a, b) of non-negative integers that satisfy a + b = 50 -/
def count_solutions : ℕ := 
  (Finset.range 51).card

/-- Theorem: There are 51 distinct ordered pairs (a, b) of non-negative integers that satisfy a + b = 50 -/
theorem count_solutions_eq_51 : count_solutions = 51 := by
  sorry

#eval count_solutions  -- This should output 51

end NUMINAMATH_CALUDE_count_solutions_eq_51_l2581_258179


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2581_258117

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d) ∧
  a 4 = -8 ∧
  a 8 = 2

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) (h : arithmetic_sequence a) :
  a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2581_258117


namespace NUMINAMATH_CALUDE_probability_is_three_fiftieths_l2581_258182

/-- Represents a 5x5x5 cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)

/-- Represents the count of different types of unit cubes -/
structure CubeCounts :=
  (two_blue : ℕ)
  (unpainted : ℕ)

/-- Calculates the probability of selecting specific cube types -/
def probability_two_blue_and_unpainted (cube : PaintedCube) (counts : CubeCounts) : ℚ :=
  let total_combinations := (cube.total_cubes.choose 2 : ℚ)
  let favorable_outcomes := (counts.two_blue * counts.unpainted : ℚ)
  favorable_outcomes / total_combinations

/-- The main theorem to be proved -/
theorem probability_is_three_fiftieths (cube : PaintedCube) (counts : CubeCounts) : 
  cube.size = 5 ∧ 
  cube.total_cubes = 125 ∧ 
  cube.blue_faces = 2 ∧ 
  cube.red_faces = 1 ∧
  counts.two_blue = 9 ∧
  counts.unpainted = 51 →
  probability_two_blue_and_unpainted cube counts = 3 / 50 :=
sorry

end NUMINAMATH_CALUDE_probability_is_three_fiftieths_l2581_258182


namespace NUMINAMATH_CALUDE_triangular_pyramid_can_be_oblique_l2581_258170

/-- A pyramid with a regular triangular base and isosceles triangular lateral faces -/
structure TriangularPyramid where
  /-- The base of the pyramid is a regular triangle -/
  base_is_regular : Bool
  /-- Each lateral face is an isosceles triangle -/
  lateral_faces_isosceles : Bool

/-- Definition of an oblique pyramid -/
def is_oblique_pyramid (p : TriangularPyramid) : Prop :=
  ∃ (lateral_edge base_edge : ℝ), lateral_edge ≠ base_edge

/-- Theorem stating that a TriangularPyramid can be an oblique pyramid -/
theorem triangular_pyramid_can_be_oblique (p : TriangularPyramid) 
  (h1 : p.base_is_regular = true) 
  (h2 : p.lateral_faces_isosceles = true) : 
  ∃ (q : TriangularPyramid), is_oblique_pyramid q :=
sorry

end NUMINAMATH_CALUDE_triangular_pyramid_can_be_oblique_l2581_258170


namespace NUMINAMATH_CALUDE_tom_has_sixteen_robots_l2581_258162

/-- The number of animal robots Michael has -/
def michael_robots : ℕ := 8

/-- The number of animal robots Tom has -/
def tom_robots : ℕ := 2 * michael_robots

/-- Theorem stating that Tom has 16 animal robots -/
theorem tom_has_sixteen_robots : tom_robots = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_has_sixteen_robots_l2581_258162


namespace NUMINAMATH_CALUDE_range_of_a_l2581_258199

theorem range_of_a (a : ℝ) : a > 0 →
  (((∀ x y : ℝ, x < y → a^x > a^y) ↔ ¬(∀ x : ℝ, x^2 - 3*a*x + 1 > 0)) ↔
   (2/3 ≤ a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2581_258199


namespace NUMINAMATH_CALUDE_expression_evaluation_l2581_258124

theorem expression_evaluation :
  let x : ℚ := -1/2
  (3 * x^4 - 2 * x^3) / (-x) - (x - x^2) * 3 * x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2581_258124


namespace NUMINAMATH_CALUDE_read_book_in_12_days_l2581_258175

/-- Represents the number of days it takes to read a book -/
def days_to_read_book (total_pages : ℕ) (weekday_pages : ℕ) (weekend_pages : ℕ) : ℕ :=
  let pages_per_week := 5 * weekday_pages + 2 * weekend_pages
  let full_weeks := total_pages / pages_per_week
  let remaining_pages := total_pages % pages_per_week
  let additional_days := 
    if remaining_pages ≤ 5 * weekday_pages
    then (remaining_pages + weekday_pages - 1) / weekday_pages
    else 5 + (remaining_pages - 5 * weekday_pages + weekend_pages - 1) / weekend_pages
  7 * full_weeks + additional_days

/-- Theorem stating that it takes 12 days to read the book under given conditions -/
theorem read_book_in_12_days : 
  days_to_read_book 285 23 35 = 12 := by
  sorry

end NUMINAMATH_CALUDE_read_book_in_12_days_l2581_258175


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2581_258125

theorem least_positive_integer_with_remainders : ∃! b : ℕ+, 
  (b : ℕ) % 3 = 2 ∧ 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 5 = 4 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ x : ℕ+, 
    (x : ℕ) % 3 = 2 → 
    (x : ℕ) % 4 = 3 → 
    (x : ℕ) % 5 = 4 → 
    (x : ℕ) % 6 = 5 → 
    b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2581_258125


namespace NUMINAMATH_CALUDE_texasCityGDP2009_scientific_notation_l2581_258115

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The GDP of Texas City in 2009 in billion yuan -/
def texasCityGDP2009 : ℝ := 1545.35

theorem texasCityGDP2009_scientific_notation :
  toScientificNotation (texasCityGDP2009 * 1000000000) 3 =
    ScientificNotation.mk 1.55 11 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_texasCityGDP2009_scientific_notation_l2581_258115


namespace NUMINAMATH_CALUDE_knights_and_liars_l2581_258173

-- Define the inhabitants
inductive Inhabitant : Type
| A
| B
| C

-- Define the possible types of inhabitants
inductive InhabitantType : Type
| Knight
| Liar

-- Define a function to determine if an inhabitant is a knight or liar
def isKnight : Inhabitant → Bool
| Inhabitant.A => true  -- We assume A is a knight based on the solution
| Inhabitant.B => true  -- To be proved
| Inhabitant.C => false -- To be proved

-- Define what B and C claim about A's statement
def B_claim : Prop := isKnight Inhabitant.A = true
def C_claim : Prop := isKnight Inhabitant.A = false

-- The main theorem to prove
theorem knights_and_liars :
  (B_claim ∧ ¬C_claim) →
  (isKnight Inhabitant.B = true ∧ isKnight Inhabitant.C = false) := by
  sorry


end NUMINAMATH_CALUDE_knights_and_liars_l2581_258173


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2581_258108

theorem real_roots_of_polynomial (x : ℝ) : 
  (x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0) ↔ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2581_258108


namespace NUMINAMATH_CALUDE_abs_negative_eleven_l2581_258177

theorem abs_negative_eleven : abs (-11 : ℤ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_eleven_l2581_258177


namespace NUMINAMATH_CALUDE_coin_drawing_probability_l2581_258138

/-- The number of shiny coins in the box -/
def shiny_coins : ℕ := 3

/-- The number of dull coins in the box -/
def dull_coins : ℕ := 4

/-- The total number of coins in the box -/
def total_coins : ℕ := shiny_coins + dull_coins

/-- The probability of needing more than 4 draws to select all shiny coins -/
def prob_more_than_four_draws : ℚ := 31 / 35

theorem coin_drawing_probability :
  let p := 1 - (Nat.choose shiny_coins shiny_coins * 
    (Nat.choose dull_coins 1 * Nat.choose shiny_coins shiny_coins + 
    Nat.choose (total_coins - 1) 3)) / Nat.choose total_coins 4
  p = prob_more_than_four_draws := by sorry

end NUMINAMATH_CALUDE_coin_drawing_probability_l2581_258138


namespace NUMINAMATH_CALUDE_redistribution_impossible_l2581_258112

/-- Represents the distribution of balls in boxes -/
structure BallDistribution where
  white_boxes : ℕ
  black_boxes : ℕ
  balls_per_white : ℕ
  balls_per_black : ℕ

/-- The initial distribution of balls -/
def initial_distribution : BallDistribution :=
  { white_boxes := 0,  -- We don't know the exact number, so we use 0
    black_boxes := 0,  -- We don't know the exact number, so we use 0
    balls_per_white := 31,
    balls_per_black := 26 }

/-- The distribution after adding 3 boxes -/
def new_distribution : BallDistribution :=
  { white_boxes := initial_distribution.white_boxes + 3,  -- Total boxes increased by 3
    black_boxes := initial_distribution.black_boxes,      -- Assuming all new boxes are white
    balls_per_white := 21,
    balls_per_black := 16 }

/-- The desired final distribution -/
def desired_distribution : BallDistribution :=
  { white_boxes := 0,  -- We don't know the exact number
    black_boxes := 0,  -- We don't know the exact number
    balls_per_white := 15,
    balls_per_black := 10 }

theorem redistribution_impossible :
  ∀ (final_distribution : BallDistribution),
  (final_distribution.balls_per_white = desired_distribution.balls_per_white ∧
   final_distribution.balls_per_black = desired_distribution.balls_per_black) →
  (final_distribution.white_boxes * final_distribution.balls_per_white +
   final_distribution.black_boxes * final_distribution.balls_per_black =
   new_distribution.white_boxes * new_distribution.balls_per_white +
   new_distribution.black_boxes * new_distribution.balls_per_black) →
  False :=
sorry

end NUMINAMATH_CALUDE_redistribution_impossible_l2581_258112


namespace NUMINAMATH_CALUDE_paint_intensity_after_replacement_l2581_258195

/-- Calculates the new paint intensity after partial replacement -/
def new_paint_intensity (initial_intensity : ℝ) (replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

/-- Theorem: Given the specified conditions, the new paint intensity is 0.4 (40%) -/
theorem paint_intensity_after_replacement :
  let initial_intensity : ℝ := 0.5
  let replacement_intensity : ℝ := 0.25
  let replacement_fraction : ℝ := 0.4
  new_paint_intensity initial_intensity replacement_intensity replacement_fraction = 0.4 := by
sorry

#eval new_paint_intensity 0.5 0.25 0.4

end NUMINAMATH_CALUDE_paint_intensity_after_replacement_l2581_258195


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2581_258181

/-- Given r is the positive real solution to x³ - x² + ¼x - 1 = 0,
    prove that the infinite sum r³ + 2r⁶ + 3r⁹ + 4r¹² + ... equals 16r -/
theorem cubic_root_sum (r : ℝ) (hr : r > 0) (hroot : r^3 - r^2 + (1/4)*r - 1 = 0) :
  (∑' n, (n : ℝ) * r^(3*n)) = 16*r := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2581_258181


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l2581_258142

theorem sufficient_condition_range (a x : ℝ) : 
  (∀ x, (a ≤ x ∧ x < a + 2) → x ≤ -1) ∧ 
  (∃ x, x ≤ -1 ∧ ¬(a ≤ x ∧ x < a + 2)) →
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l2581_258142


namespace NUMINAMATH_CALUDE_roots_equation_r_value_l2581_258187

theorem roots_equation_r_value (m p : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  (r = 16/3) := by
sorry

end NUMINAMATH_CALUDE_roots_equation_r_value_l2581_258187


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l2581_258198

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 5

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ x < 1 → f x < f y) ∧
  (∀ x y, 3 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < 3 → f x > f y) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → f x < f 1) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → f x > f 3) ∧
  f 1 = -1 ∧
  f 3 = -5 := by
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l2581_258198


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2581_258137

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q (a : ℕ → ℝ) : ℝ := sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  S 4 a = -5 →
  S 6 a = 21 * S 2 a →
  S 8 a = -85 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2581_258137


namespace NUMINAMATH_CALUDE_equation_solution_l2581_258161

theorem equation_solution (x y : ℝ) : ∃ z : ℝ, 0.65 * x * y - z = 0.2 * 747.50 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2581_258161


namespace NUMINAMATH_CALUDE_lindas_cookies_l2581_258119

theorem lindas_cookies (classmates : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (remaining_batches : ℕ) :
  classmates = 24 →
  cookies_per_student = 10 →
  cookies_per_batch = 48 →
  chocolate_chip_batches = 2 →
  remaining_batches = 2 →
  (classmates * cookies_per_student - chocolate_chip_batches * cookies_per_batch) / cookies_per_batch - remaining_batches = 1 :=
by sorry

end NUMINAMATH_CALUDE_lindas_cookies_l2581_258119


namespace NUMINAMATH_CALUDE_cube_root_of_product_l2581_258183

theorem cube_root_of_product (a b c : ℕ) : 
  (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l2581_258183


namespace NUMINAMATH_CALUDE_butterflies_fraction_l2581_258185

theorem butterflies_fraction (initial : ℕ) (remaining : ℕ) : 
  initial = 9 → remaining = 6 → (initial - remaining : ℚ) / initial = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_fraction_l2581_258185


namespace NUMINAMATH_CALUDE_jimmy_earnings_theorem_l2581_258172

/-- Calculates Jimmy's total earnings from selling all his action figures --/
def jimmy_total_earnings : ℕ := by
  -- Define the number of each type of action figure
  let num_type_a : ℕ := 5
  let num_type_b : ℕ := 4
  let num_type_c : ℕ := 3

  -- Define the original value of each type of action figure
  let value_type_a : ℕ := 20
  let value_type_b : ℕ := 30
  let value_type_c : ℕ := 40

  -- Define the discount for each type of action figure
  let discount_type_a : ℕ := 7
  let discount_type_b : ℕ := 10
  let discount_type_c : ℕ := 12

  -- Calculate the selling price for each type of action figure
  let sell_price_a := value_type_a - discount_type_a
  let sell_price_b := value_type_b - discount_type_b
  let sell_price_c := value_type_c - discount_type_c

  -- Calculate the total earnings
  let total := num_type_a * sell_price_a + num_type_b * sell_price_b + num_type_c * sell_price_c

  exact total

/-- Theorem stating that Jimmy's total earnings is 229 --/
theorem jimmy_earnings_theorem : jimmy_total_earnings = 229 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_earnings_theorem_l2581_258172


namespace NUMINAMATH_CALUDE_value_of_x_l2581_258146

theorem value_of_x (x y z : ℤ) 
  (eq1 : 4*x + y + z = 80) 
  (eq2 : 2*x - y - z = 40) 
  (eq3 : 3*x + y - z = 20) : 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2581_258146


namespace NUMINAMATH_CALUDE_basketball_max_score_l2581_258150

def max_individual_score (n : ℕ) (total_points : ℕ) (min_points : ℕ) : ℕ :=
  total_points - (n - 1) * min_points

theorem basketball_max_score :
  max_individual_score 12 100 7 = 23 :=
by sorry

end NUMINAMATH_CALUDE_basketball_max_score_l2581_258150


namespace NUMINAMATH_CALUDE_twenty_percent_value_l2581_258193

theorem twenty_percent_value (x : ℝ) (h : 1.2 * x = 1200) : 0.2 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_value_l2581_258193


namespace NUMINAMATH_CALUDE_false_conjunction_implication_l2581_258163

theorem false_conjunction_implication : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_false_conjunction_implication_l2581_258163


namespace NUMINAMATH_CALUDE_restaurant_cooks_l2581_258151

theorem restaurant_cooks (initial_cooks : ℕ) (initial_waiters : ℕ) : 
  initial_cooks / initial_waiters = 3 / 11 →
  initial_cooks / (initial_waiters + 12) = 1 / 5 →
  initial_cooks = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_l2581_258151


namespace NUMINAMATH_CALUDE_polygon_sides_l2581_258106

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1440 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2581_258106


namespace NUMINAMATH_CALUDE_circle_tangent_k_range_l2581_258167

/-- Represents a circle in the 2D plane --/
structure Circle where
  k : ℝ

/-- Represents a point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is outside the circle --/
def isOutside (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + 2*p.x + 2*p.y + c.k > 0

/-- Checks if two tangents can be drawn from a point to the circle --/
def hasTwoTangents (p : Point) (c : Circle) : Prop :=
  isOutside p c

/-- The main theorem --/
theorem circle_tangent_k_range (c : Circle) :
  let p : Point := ⟨1, -1⟩
  hasTwoTangents p c → -2 < c.k ∧ c.k < 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_k_range_l2581_258167


namespace NUMINAMATH_CALUDE_min_value_z_l2581_258114

theorem min_value_z (x y : ℝ) : x^2 + 2*y^2 + 6*x - 4*y + 22 ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l2581_258114


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l2581_258144

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) :
  m ≠ n →
  parallel n m →
  perpendicular n α →
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l2581_258144


namespace NUMINAMATH_CALUDE_austin_picked_24_bags_l2581_258153

/-- The number of bags of fruit Austin picked in total -/
def austin_total (dallas_apples dallas_pears austin_apples_diff austin_pears_diff : ℕ) : ℕ :=
  (dallas_apples + austin_apples_diff) + (dallas_pears - austin_pears_diff)

/-- Theorem stating that Austin picked 24 bags of fruit in total -/
theorem austin_picked_24_bags
  (dallas_apples : ℕ)
  (dallas_pears : ℕ)
  (austin_apples_diff : ℕ)
  (austin_pears_diff : ℕ)
  (h1 : dallas_apples = 14)
  (h2 : dallas_pears = 9)
  (h3 : austin_apples_diff = 6)
  (h4 : austin_pears_diff = 5) :
  austin_total dallas_apples dallas_pears austin_apples_diff austin_pears_diff = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_austin_picked_24_bags_l2581_258153


namespace NUMINAMATH_CALUDE_centric_sequence_bound_and_extremal_points_l2581_258140

/-- The set of points (x, y) in R^2 such that x^2 + y^2 ≤ 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

/-- A sequence of points in R^2 -/
def Sequence := ℕ → ℝ × ℝ

/-- The circumcenter of a triangle formed by three points -/
noncomputable def circumcenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A centric sequence satisfies the given properties -/
def IsCentric (A : Sequence) : Prop :=
  A 0 = (0, 0) ∧ A 1 = (1, 0) ∧
  ∀ n : ℕ, circumcenter (A n) (A (n+1)) (A (n+2)) ∈ C

theorem centric_sequence_bound_and_extremal_points :
  ∀ A : Sequence, IsCentric A →
    (A 2012).1^2 + (A 2012).2^2 ≤ 4048144 ∧
    (∀ x y : ℝ, x^2 + y^2 = 4048144 →
      (∃ A : Sequence, IsCentric A ∧ A 2012 = (x, y)) →
      ((x = -1006 ∧ y = 1006 * Real.sqrt 3) ∨
       (x = -1006 ∧ y = -1006 * Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_centric_sequence_bound_and_extremal_points_l2581_258140


namespace NUMINAMATH_CALUDE_golden_retriever_age_problem_l2581_258126

/-- The age of a golden retriever given its weight gain per year and current weight -/
def golden_retriever_age (weight_gain_per_year : ℕ) (current_weight : ℕ) : ℕ :=
  current_weight / weight_gain_per_year

/-- Theorem: The age of a golden retriever that gains 11 pounds each year and currently weighs 88 pounds is 8 years -/
theorem golden_retriever_age_problem :
  golden_retriever_age 11 88 = 8 := by
  sorry

end NUMINAMATH_CALUDE_golden_retriever_age_problem_l2581_258126


namespace NUMINAMATH_CALUDE_right_triangle_30_deg_side_half_hypotenuse_l2581_258178

/-- Theorem: In a right-angled triangle with one angle of 30°, 
    the length of the side opposite to the 30° angle is equal to 
    half the length of the hypotenuse. -/
theorem right_triangle_30_deg_side_half_hypotenuse 
  (A B C : ℝ × ℝ) -- Three points representing the vertices of the triangle
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) -- Right angle condition
  (angle_30_deg : ∃ i j k, i^2 + j^2 = k^2 ∧ i / k = 1 / 2) -- 30° angle condition
  : ∃ side hypotenuse, side = hypotenuse / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_30_deg_side_half_hypotenuse_l2581_258178


namespace NUMINAMATH_CALUDE_circle_through_points_is_valid_circle_equation_l2581_258128

/-- Given three points in 2D space, this function returns true if they lie on the circle
    described by the equation x^2 + y^2 - 4x - 6y = 0 -/
def points_on_circle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let f := fun (x y : ℝ) => x^2 + y^2 - 4*x - 6*y
  f p1.1 p1.2 = 0 ∧ f p2.1 p2.2 = 0 ∧ f p3.1 p3.2 = 0

/-- The theorem states that the points (0,0), (4,0), and (-1,1) lie on the circle
    described by the equation x^2 + y^2 - 4x - 6y = 0 -/
theorem circle_through_points :
  points_on_circle (0, 0) (4, 0) (-1, 1) := by
  sorry

/-- The general equation of a circle is x^2 + y^2 + Dx + Ey + F = 0 -/
def is_circle_equation (D E F : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = x^2 + y^2 + D*x + E*y + F

/-- This theorem states that the equation x^2 + y^2 - 4x - 6y = 0 is a valid circle equation -/
theorem is_valid_circle_equation :
  is_circle_equation (-4) (-6) 0 (fun x y => x^2 + y^2 - 4*x - 6*y) := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_is_valid_circle_equation_l2581_258128


namespace NUMINAMATH_CALUDE_store_discount_income_increase_l2581_258131

theorem store_discount_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (quantity_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : quantity_increase_rate = 0.2) : 
  let new_price := original_price * (1 - discount_rate)
  let new_quantity := original_quantity * (1 + quantity_increase_rate)
  let original_income := original_price * original_quantity
  let new_income := new_price * new_quantity
  (new_income - original_income) / original_income = 0.08 := by
sorry

end NUMINAMATH_CALUDE_store_discount_income_increase_l2581_258131


namespace NUMINAMATH_CALUDE_sum_repeating_decimals_eq_l2581_258188

/-- The sum of the repeating decimals 0.141414... and 0.272727... -/
def sum_repeating_decimals : ℚ :=
  let a : ℚ := 14 / 99  -- 0.141414...
  let b : ℚ := 27 / 99  -- 0.272727...
  a + b

/-- Theorem: The sum of the repeating decimals 0.141414... and 0.272727... is 41/99 -/
theorem sum_repeating_decimals_eq :
  sum_repeating_decimals = 41 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_repeating_decimals_eq_l2581_258188


namespace NUMINAMATH_CALUDE_right_triangle_area_l2581_258141

theorem right_triangle_area (leg1 leg2 : ℝ) (h1 : leg1 = 45) (h2 : leg2 = 48) :
  (1/2 : ℝ) * leg1 * leg2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2581_258141


namespace NUMINAMATH_CALUDE_project_hours_difference_l2581_258186

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 144) 
  (kate_hours pat_hours mark_hours : ℕ) 
  (h_pat_kate : pat_hours = 2 * kate_hours)
  (h_pat_mark : pat_hours * 3 = mark_hours)
  (h_sum : kate_hours + pat_hours + mark_hours = total_hours) :
  mark_hours - kate_hours = 80 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2581_258186


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l2581_258180

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem unique_solution_floor_equation :
  ∃! x : ℝ, (floor (x - 1/2) : ℝ) = 3*x - 5 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l2581_258180


namespace NUMINAMATH_CALUDE_equation_solution_l2581_258189

theorem equation_solution : 
  ∃ x : ℝ, (10 : ℝ)^(2*x) * (100 : ℝ)^(3*x) = (1000 : ℝ)^7 ∧ x = 21/8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2581_258189


namespace NUMINAMATH_CALUDE_gcd_cube_plus_eight_and_n_plus_three_l2581_258196

theorem gcd_cube_plus_eight_and_n_plus_three (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 + 2^3) (n + 3) = 9 := by sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_eight_and_n_plus_three_l2581_258196


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_touching_circle_l2581_258149

/-- A triangle with sides a, b, c and medians m_a, m_b, m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ

/-- A circle touching two sides and two medians of a triangle -/
structure TouchingCircle (T : Triangle) where
  touches_side_a : Bool
  touches_side_b : Bool
  touches_median_a : Bool
  touches_median_b : Bool

/-- 
If a circle touches two sides of a triangle and their corresponding medians,
then the triangle is isosceles.
-/
theorem isosceles_triangle_from_touching_circle (T : Triangle) 
  (C : TouchingCircle T) (h1 : C.touches_side_a) (h2 : C.touches_side_b) 
  (h3 : C.touches_median_a) (h4 : C.touches_median_b) : 
  T.a = T.b := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_from_touching_circle_l2581_258149


namespace NUMINAMATH_CALUDE_divided_square_area_l2581_258130

/-- Represents a square divided into rectangles -/
structure DividedSquare where
  side_length : ℝ
  vertical_lines : ℕ
  horizontal_lines : ℕ

/-- Calculates the total perimeter of all rectangles in a divided square -/
def total_perimeter (s : DividedSquare) : ℝ :=
  4 * s.side_length + 2 * s.side_length * (s.vertical_lines * (s.horizontal_lines + 1) + s.horizontal_lines * (s.vertical_lines + 1))

/-- The main theorem -/
theorem divided_square_area (s : DividedSquare) 
  (h1 : s.vertical_lines = 5)
  (h2 : s.horizontal_lines = 3)
  (h3 : (s.vertical_lines + 1) * (s.horizontal_lines + 1) = 24)
  (h4 : total_perimeter s = 24) :
  s.side_length ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divided_square_area_l2581_258130


namespace NUMINAMATH_CALUDE_sector_angle_l2581_258191

/-- A circular sector with arc length and area both equal to 4 has a central angle of 2 radians -/
theorem sector_angle (R : ℝ) (α : ℝ) (h1 : α * R = 4) (h2 : (1/2) * α * R^2 = 4) : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2581_258191


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l2581_258121

-- Define an equilateral triangle ABC with side length s
def equilateral_triangle (A B C : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Define the extended points B', C', and A'
def extended_points (A B C A' B' C' : ℝ × ℝ) (s : ℝ) : Prop :=
  dist B B' = 2*s ∧ dist C C' = 3*s ∧ dist A A' = 4*s

-- Define the area of a triangle given its vertices
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_theorem (A B C A' B' C' : ℝ × ℝ) (s : ℝ) :
  equilateral_triangle A B C s →
  extended_points A B C A' B' C' s →
  triangle_area A' B' C' / triangle_area A B C = 60 := by sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l2581_258121


namespace NUMINAMATH_CALUDE_reflection_property_l2581_258100

/-- A reflection in R² --/
structure Reflection where
  /-- The reflection function --/
  reflect : ℝ × ℝ → ℝ × ℝ

/-- Given a reflection that maps (2, -3) to (-2, 9), it also maps (3, 1) to (-3, 1) --/
theorem reflection_property (r : Reflection) 
  (h1 : r.reflect (2, -3) = (-2, 9)) : 
  r.reflect (3, 1) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_property_l2581_258100


namespace NUMINAMATH_CALUDE_geometric_sequence_special_property_l2581_258104

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₂ · a₄ = 2a₃ - 1, then a₃ = 1 -/
theorem geometric_sequence_special_property (a : ℕ → ℝ) :
  geometric_sequence a → a 2 * a 4 = 2 * a 3 - 1 → a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_property_l2581_258104


namespace NUMINAMATH_CALUDE_willie_initial_stickers_l2581_258102

/-- The number of stickers Willie gave to Emily -/
def stickers_given : ℕ := 7

/-- The number of stickers Willie had left after giving some to Emily -/
def stickers_left : ℕ := 29

/-- The initial number of stickers Willie had -/
def initial_stickers : ℕ := stickers_given + stickers_left

theorem willie_initial_stickers : initial_stickers = 36 := by
  sorry

end NUMINAMATH_CALUDE_willie_initial_stickers_l2581_258102


namespace NUMINAMATH_CALUDE_hare_tortoise_race_l2581_258155

theorem hare_tortoise_race (v : ℝ) (x : ℝ) (y : ℝ) (h_v_pos : v > 0) :
  v > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  x + y = 25 ∧ 
  x^2 + 5^2 = y^2 →
  y = 13 :=
by sorry

end NUMINAMATH_CALUDE_hare_tortoise_race_l2581_258155


namespace NUMINAMATH_CALUDE_distance_between_points_l2581_258105

theorem distance_between_points : Real.sqrt 89 = Real.sqrt ((1 - (-4))^2 + (-3 - 5)^2) := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2581_258105


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l2581_258192

theorem min_value_product_quotient (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) ≥ 343 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l2581_258192


namespace NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l2581_258190

theorem consecutive_numbers_divisibility (n : ℕ) :
  n ≥ 4 ∧
  n ∣ ((n - 3) * (n - 2) * (n - 1)) →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l2581_258190


namespace NUMINAMATH_CALUDE_kelly_games_to_give_away_l2581_258160

theorem kelly_games_to_give_away (initial_games : ℕ) (remaining_games : ℕ) : 
  initial_games - remaining_games = 15 :=
by
  sorry

#check kelly_games_to_give_away 50 35

end NUMINAMATH_CALUDE_kelly_games_to_give_away_l2581_258160


namespace NUMINAMATH_CALUDE_volume_of_specific_open_box_l2581_258159

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem stating that the volume of the specific open box is 5120 cubic meters. -/
theorem volume_of_specific_open_box :
  openBoxVolume 48 36 8 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_open_box_l2581_258159


namespace NUMINAMATH_CALUDE_tank_fill_time_xy_l2581_258165

/-- Represents the time (in hours) to fill a tank given specific valve configurations -/
structure TankFillTime where
  all : ℝ
  xz : ℝ
  yz : ℝ

/-- Proves that given specific fill times for different valve configurations, 
    the time to fill the tank with only valves X and Y open is 2.4 hours -/
theorem tank_fill_time_xy (t : TankFillTime) 
  (h_all : t.all = 2)
  (h_xz : t.xz = 3)
  (h_yz : t.yz = 4) :
  1 / (1 / t.all - 1 / t.yz) + 1 / (1 / t.all - 1 / t.xz) = 2.4 := by
  sorry

#check tank_fill_time_xy

end NUMINAMATH_CALUDE_tank_fill_time_xy_l2581_258165


namespace NUMINAMATH_CALUDE_product_and_ratio_implies_y_value_l2581_258152

theorem product_and_ratio_implies_y_value 
  (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = 9) 
  (h4 : x / y = 36) : 
  y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_product_and_ratio_implies_y_value_l2581_258152


namespace NUMINAMATH_CALUDE_stone_length_proof_l2581_258129

/-- Given a hall and stones with specific dimensions, prove the length of each stone --/
theorem stone_length_proof (hall_length hall_width : ℝ) (stone_width : ℝ) (num_stones : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_width = 0.5)
  (h4 : num_stones = 1800) :
  (hall_length * hall_width * 100) / (stone_width * 10 * num_stones) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stone_length_proof_l2581_258129


namespace NUMINAMATH_CALUDE_bills_age_l2581_258136

theorem bills_age (bill eric : ℕ) 
  (h1 : bill = eric + 4) 
  (h2 : bill + eric = 28) : 
  bill = 16 := by
  sorry

end NUMINAMATH_CALUDE_bills_age_l2581_258136


namespace NUMINAMATH_CALUDE_special_triangle_area_property_l2581_258107

/-- A triangle with side length PQ = 30 and its incircle trisecting the median PS in ratio 1:2 -/
structure SpecialTriangle where
  -- Points of the triangle
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- Incircle center
  I : ℝ × ℝ
  -- Point where median PS intersects QR
  S : ℝ × ℝ
  -- Points where incircle touches the sides
  T : ℝ × ℝ  -- on QR
  U : ℝ × ℝ  -- on RP
  V : ℝ × ℝ  -- on PQ
  -- Properties
  pq_length : dist P Q = 30
  trisect_median : dist P T = (1/3) * dist P S ∧ dist T S = (2/3) * dist P S
  incircle_tangent : dist I T = dist I U ∧ dist I U = dist I V

/-- The area of the special triangle can be expressed as x√y where x and y are integers -/
def area_expression (t : SpecialTriangle) : ℕ × ℕ :=
  sorry

/-- Predicate to check if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

theorem special_triangle_area_property (t : SpecialTriangle) :
  let (x, y) := area_expression t
  (x > 0 ∧ y > 0) ∧ not_divisible_by_prime_square y ∧ ∃ (k : ℕ), x + y = k :=
sorry

end NUMINAMATH_CALUDE_special_triangle_area_property_l2581_258107


namespace NUMINAMATH_CALUDE_martha_cakes_per_child_l2581_258118

/-- Given that Martha has 3.0 children and needs to buy 54 cakes in total,
    prove that each child will get 18 cakes. -/
theorem martha_cakes_per_child :
  let num_children : ℝ := 3.0
  let total_cakes : ℕ := 54
  (total_cakes : ℝ) / num_children = 18 :=
by sorry

end NUMINAMATH_CALUDE_martha_cakes_per_child_l2581_258118


namespace NUMINAMATH_CALUDE_parabola_min_y_l2581_258194

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  y + x = (y - x)^2 + 3*(y - x) + 3

/-- Theorem stating the minimum value of y for points on the parabola -/
theorem parabola_min_y :
  (∀ x y : ℝ, parabola_eq x y → y ≥ -1/2) ∧
  (∃ x y : ℝ, parabola_eq x y ∧ y = -1/2) := by
sorry

end NUMINAMATH_CALUDE_parabola_min_y_l2581_258194


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2581_258101

/-- Race problem statement -/
theorem race_speed_ratio :
  ∀ (vA vB : ℝ) (d : ℝ),
  d > 0 →
  d / vA = 2 →
  d / vB = 1.5 →
  vA / vB = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2581_258101


namespace NUMINAMATH_CALUDE_rainfall_difference_l2581_258147

def monday_count : ℕ := 10
def tuesday_count : ℕ := 12
def wednesday_count : ℕ := 8
def thursday_count : ℕ := 6

def monday_rain : ℝ := 1.25
def tuesday_rain : ℝ := 2.15
def wednesday_rain : ℝ := 1.60
def thursday_rain : ℝ := 2.80

theorem rainfall_difference :
  (tuesday_count * tuesday_rain + thursday_count * thursday_rain) -
  (monday_count * monday_rain + wednesday_count * wednesday_rain) = 17.3 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l2581_258147


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2581_258113

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * s = 68 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2581_258113


namespace NUMINAMATH_CALUDE_calculation_part1_sum_first_25_odd_numbers_l2581_258145

-- Part 1
theorem calculation_part1 : 0.45 * 2.5 + 4.5 * 0.65 + 0.45 = 4.5 := by
  sorry

-- Part 2
def first_n_odd_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * i + 1)

theorem sum_first_25_odd_numbers :
  (first_n_odd_numbers 25).sum = 625 := by
  sorry

end NUMINAMATH_CALUDE_calculation_part1_sum_first_25_odd_numbers_l2581_258145


namespace NUMINAMATH_CALUDE_sum_of_variables_l2581_258154

theorem sum_of_variables (x y z : ℚ) 
  (eq1 : y + z = 18 - 4*x)
  (eq2 : x + z = 16 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  3*x + 3*y + 3*z = 43/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l2581_258154


namespace NUMINAMATH_CALUDE_ae_length_l2581_258123

/-- Triangle ABC and ADE share vertex A and angle A --/
structure NestedTriangles where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  AE : ℝ
  k : ℝ
  area_proportion : AB * AC = k * AD * AE

/-- The specific nested triangles in the problem --/
def problem_triangles : NestedTriangles where
  AB := 5
  AC := 7
  AD := 2
  AE := 17.5
  k := 1
  area_proportion := by sorry

theorem ae_length (t : NestedTriangles) (h1 : t.AB = 5) (h2 : t.AC = 7) (h3 : t.AD = 2) (h4 : t.k = 1) :
  t.AE = 17.5 := by
  sorry

#check ae_length problem_triangles

end NUMINAMATH_CALUDE_ae_length_l2581_258123


namespace NUMINAMATH_CALUDE_curve_fixed_point_l2581_258116

/-- The curve C: x^2 + y^2 + 2kx + (4k+10)y + 10k + 20 = 0 passes through the fixed point (1, -3) for all k ≠ -1 -/
theorem curve_fixed_point (k : ℝ) (h : k ≠ -1) :
  let C (x y : ℝ) := x^2 + y^2 + 2*k*x + (4*k+10)*y + 10*k + 20
  C 1 (-3) = 0 := by sorry

end NUMINAMATH_CALUDE_curve_fixed_point_l2581_258116


namespace NUMINAMATH_CALUDE_reciprocal_of_one_fifth_l2581_258139

theorem reciprocal_of_one_fifth (x : ℚ) : 
  (x * (1 / x) = 1) → ((1 / (1 / 5)) = 5) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_fifth_l2581_258139


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2581_258109

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 32) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2581_258109


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2581_258197

/-- Given a hyperbola with the specified properties, prove its equation is x²/8 - y²/8 = 1 -/
theorem hyperbola_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (c / a = Real.sqrt 2) →                   -- Eccentricity is √2
  (4 / c = 1) →                             -- Slope of line through F(-c,0) and P(0,4) is 1
  (a = b) →                                 -- Equilateral hyperbola
  (∀ x y : ℝ, x^2 / 8 - y^2 / 8 = 1) :=     -- Resulting equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2581_258197


namespace NUMINAMATH_CALUDE_xy_divided_by_three_l2581_258171

theorem xy_divided_by_three (x y : ℚ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : x + 2 * y = 5) : 
  (x + y) / 3 = 1.222222222222222 := by
sorry

end NUMINAMATH_CALUDE_xy_divided_by_three_l2581_258171


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_inequality_l2581_258110

theorem lcm_gcd_sum_inequality (a b k : ℕ+) (hk : k > 1) 
  (h : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : 
  a + b ≥ 4 * k := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_inequality_l2581_258110


namespace NUMINAMATH_CALUDE_total_mass_of_water_l2581_258122

/-- The total mass of water in two glasses on an unequal-arm scale -/
theorem total_mass_of_water (L m l : ℝ) (hL : L > 0) (hm : m > 0) (hl : l ≠ 0) : ∃ total_mass : ℝ,
  (∃ m₁ m₂ l₁ : ℝ, 
    -- Initial balance condition
    m₁ * l₁ = m₂ * (L - l₁) ∧
    -- Balance condition after transfer
    (m₁ - m) * (l₁ + l) = (m₂ + m) * (L - l₁ - l) ∧
    -- Total mass definition
    total_mass = m₁ + m₂) ∧
  total_mass = m * L / l :=
sorry

end NUMINAMATH_CALUDE_total_mass_of_water_l2581_258122


namespace NUMINAMATH_CALUDE_neil_initial_games_neil_had_two_games_l2581_258127

theorem neil_initial_games (henry_initial : ℕ) (games_given : ℕ) (henry_neil_ratio : ℕ) : ℕ :=
  let henry_final := henry_initial - games_given
  let neil_final := henry_final / henry_neil_ratio
  neil_final - games_given

theorem neil_had_two_games : neil_initial_games 33 5 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_neil_initial_games_neil_had_two_games_l2581_258127


namespace NUMINAMATH_CALUDE_books_in_bargain_bin_l2581_258166

theorem books_in_bargain_bin 
  (initial_books : ℕ) 
  (books_sold : ℕ) 
  (books_added : ℕ) 
  (h1 : initial_books ≥ books_sold) : 
  initial_books - books_sold + books_added = 
    initial_books + books_added - books_sold :=
by sorry

end NUMINAMATH_CALUDE_books_in_bargain_bin_l2581_258166


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l2581_258184

theorem parallelepiped_volume (j : ℝ) :
  j > 0 →
  (abs (3 * (j^2 - 9) - 2 * (4*j - 15) + 2 * (12 - 5*j)) = 36) →
  j = (9 + Real.sqrt 585) / 6 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l2581_258184


namespace NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_deriv_l2581_258176

/-- The function f(x) = -x^3 + ax^2 - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f_deriv (a : ℝ) :
  (∃ (x : ℝ), f_deriv a x = 0 ∧ x = 2) →
  (∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 → n' ∈ Set.Icc (-1 : ℝ) 1 →
      f a m + f_deriv a n ≤ f a m' + f_deriv a n') →
  ∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_deriv a n = -13 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_deriv_l2581_258176


namespace NUMINAMATH_CALUDE_polynomial_roots_l2581_258148

theorem polynomial_roots : 
  let p (x : ℝ) := 3 * x^4 - 2 * x^3 - 4 * x^2 - 2 * x + 3
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = (-1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2581_258148
