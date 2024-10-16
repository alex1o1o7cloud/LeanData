import Mathlib

namespace NUMINAMATH_CALUDE_mary_remaining_sheep_l3360_336013

def initial_sheep : ℕ := 400

def sheep_after_sister (initial : ℕ) : ℕ :=
  initial - (initial / 4)

def sheep_after_brother (after_sister : ℕ) : ℕ :=
  after_sister - (after_sister / 2)

theorem mary_remaining_sheep :
  sheep_after_brother (sheep_after_sister initial_sheep) = 150 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_sheep_l3360_336013


namespace NUMINAMATH_CALUDE_price_reduction_l3360_336047

theorem price_reduction (initial_price : ℝ) (first_reduction : ℝ) : 
  first_reduction > 0 ∧ first_reduction < 100 →
  (1 - first_reduction / 100) * (1 - 0.3) = (1 - 0.475) →
  first_reduction = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l3360_336047


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l3360_336056

theorem surrounding_circles_radius (r : ℝ) : 
  (∃ (A B C D : ℝ × ℝ),
    -- Define the centers of the four surrounding circles
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (2*r)^2 ∧
    -- Ensure the surrounding circles touch the central circle
    (A.1^2 + A.2^2 = (r + 2)^2) ∧
    (B.1^2 + B.2^2 = (r + 2)^2) ∧
    (C.1^2 + C.2^2 = (r + 2)^2) ∧
    (D.1^2 + D.2^2 = (r + 2)^2) ∧
    -- Ensure the surrounding circles touch each other
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (2*r)^2) →
  r = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l3360_336056


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3360_336052

/-- For an infinite geometric series with common ratio -1/3 and sum 12, the first term is 16 -/
theorem infinite_geometric_series_first_term :
  ∀ (a : ℝ), 
    (∃ (S : ℝ), S = a / (1 - (-1/3))) →  -- Infinite geometric series formula
    (a / (1 - (-1/3)) = 12) →             -- Sum of the series is 12
    a = 16 :=                             -- First term is 16
by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3360_336052


namespace NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l3360_336026

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem 1: A ⊆ B iff a < -1
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a < -1 := by sorry

-- Theorem 2: A ∩ B ≠ ∅ iff a < 3
theorem nonempty_intersection (a : ℝ) : (A ∩ B a).Nonempty ↔ a < 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l3360_336026


namespace NUMINAMATH_CALUDE_min_value_f_min_value_sum_squares_l3360_336024

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the minimum value of f(x)
theorem min_value_f : 
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 5 :=
sorry

-- Theorem for the minimum value of a^2 + 2b^2 + 3c^2
theorem min_value_sum_squares :
  ∃ m : ℝ, m = 15/2 ∧
  (∀ a b c : ℝ, a + 2*b + c = 5 → a^2 + 2*b^2 + 3*c^2 ≥ m) ∧
  (∃ a b c : ℝ, a + 2*b + c = 5 ∧ a^2 + 2*b^2 + 3*c^2 = m) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_sum_squares_l3360_336024


namespace NUMINAMATH_CALUDE_a_minus_c_value_l3360_336012

/-- Given that A = 742, B = A + 397, and B = C + 693, prove that A - C = 296 -/
theorem a_minus_c_value (A B C : ℤ) 
  (h1 : A = 742)
  (h2 : B = A + 397)
  (h3 : B = C + 693) : 
  A - C = 296 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_value_l3360_336012


namespace NUMINAMATH_CALUDE_circle_tangent_to_hyperbola_radius_l3360_336009

/-- The radius of a circle tangent to a hyperbola --/
theorem circle_tangent_to_hyperbola_radius 
  (a b : ℝ) 
  (h_a : a = 6) 
  (h_b : b = 5) : 
  let c := Real.sqrt (a^2 + b^2)
  let r := |a - c|
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x - c)^2 + y^2 ≥ r^2) ∧ 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_hyperbola_radius_l3360_336009


namespace NUMINAMATH_CALUDE_bag_cost_is_eight_l3360_336074

/-- Represents the coffee consumption and cost for Maddie's mom --/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  ounces_per_bag : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  total_weekly_cost : ℚ

/-- Calculates the cost of a bag of coffee based on the given consumption data --/
def bag_cost (c : CoffeeConsumption) : ℚ :=
  let ounces_per_week := c.cups_per_day * c.ounces_per_cup * 7
  let bags_per_week := ounces_per_week / c.ounces_per_bag
  let milk_cost_per_week := c.milk_gallons_per_week * c.milk_cost_per_gallon
  let coffee_cost_per_week := c.total_weekly_cost - milk_cost_per_week
  coffee_cost_per_week / bags_per_week

/-- Theorem stating that the cost of a bag of coffee is $8 --/
theorem bag_cost_is_eight (c : CoffeeConsumption) 
  (h1 : c.cups_per_day = 2)
  (h2 : c.ounces_per_cup = 3/2)
  (h3 : c.ounces_per_bag = 21/2)
  (h4 : c.milk_gallons_per_week = 1/2)
  (h5 : c.milk_cost_per_gallon = 4)
  (h6 : c.total_weekly_cost = 18) :
  bag_cost c = 8 := by
  sorry

#eval bag_cost {
  cups_per_day := 2,
  ounces_per_cup := 3/2,
  ounces_per_bag := 21/2,
  milk_gallons_per_week := 1/2,
  milk_cost_per_gallon := 4,
  total_weekly_cost := 18
}

end NUMINAMATH_CALUDE_bag_cost_is_eight_l3360_336074


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3360_336022

/-- Given a rectangle with area 20 sq. cm and perimeter 18 cm, prove its diagonal is √41 cm. -/
theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  Real.sqrt (l^2 + w^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3360_336022


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l3360_336063

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareConfig where
  s : ℝ  -- Side length of the inner square
  x : ℝ  -- Shorter side of the rectangle
  y : ℝ  -- Longer side of the rectangle

/-- Theorem: If four congruent rectangles are placed around a central square such that 
    the area of the outer square is 9 times the area of the inner square, 
    then the ratio of the longer side to the shorter side of each rectangle is 2. -/
theorem rectangle_square_ratio 
  (config : RectangleSquareConfig) 
  (h1 : config.s > 0)  -- Inner square has positive side length
  (h2 : config.x > 0)  -- Rectangle has positive width
  (h3 : config.y > 0)  -- Rectangle has positive height
  (h4 : config.s + 2 * config.x = 3 * config.s)  -- Outer square side length relation
  (h5 : config.y + config.x = 3 * config.s)  -- Outer square side length relation (alternative)
  : config.y / config.x = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_ratio_l3360_336063


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l3360_336043

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 3)^2 - 4*(a 3) + 3 = 0 →
  (a 7)^2 - 4*(a 7) + 3 = 0 →
  a 5 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l3360_336043


namespace NUMINAMATH_CALUDE_solve_for_a_l3360_336061

theorem solve_for_a (x a : ℝ) (h1 : 2 * x + a - 9 = 0) (h2 : x = 2) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3360_336061


namespace NUMINAMATH_CALUDE_apple_distribution_l3360_336029

theorem apple_distribution (students : ℕ) (apples : ℕ) : 
  (apples = 4 * students + 3) ∧ 
  (6 * (students - 1) ≤ apples) ∧ 
  (apples ≤ 6 * (students - 1) + 2) →
  (students = 4 ∧ apples = 19) :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3360_336029


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3360_336044

/-- For a line y = mx + b with negative slope m and positive y-intercept b, 
    the product mb satisfies -1 < mb < 0 -/
theorem line_slope_intercept_product (m b : ℝ) (h1 : m < 0) (h2 : b > 0) : 
  -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3360_336044


namespace NUMINAMATH_CALUDE_cost_of_paints_l3360_336081

def cost_paintbrush : ℚ := 2.40
def cost_easel : ℚ := 6.50
def rose_has : ℚ := 7.10
def rose_needs : ℚ := 11.00

theorem cost_of_paints :
  let total_cost := rose_has + rose_needs
  let cost_paints := total_cost - (cost_paintbrush + cost_easel)
  cost_paints = 9.20 := by sorry

end NUMINAMATH_CALUDE_cost_of_paints_l3360_336081


namespace NUMINAMATH_CALUDE_alice_bob_meet_after_5_turns_l3360_336095

/-- Represents the number of points on the circle -/
def num_points : ℕ := 15

/-- Represents Alice's clockwise movement per turn -/
def alice_move : ℕ := 4

/-- Represents Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 8

/-- Calculates the position after a given number of moves -/
def position_after_moves (start : ℕ) (move : ℕ) (turns : ℕ) : ℕ :=
  (start + move * turns) % num_points

/-- Theorem stating that Alice and Bob meet after 5 turns -/
theorem alice_bob_meet_after_5_turns :
  ∃ (meeting_point : ℕ),
    position_after_moves num_points alice_move 5 = meeting_point ∧
    position_after_moves num_points (num_points - bob_move) 5 = meeting_point :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_meet_after_5_turns_l3360_336095


namespace NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l3360_336007

/-- Represents a repeating decimal with a repeating part and a period -/
def RepeatingDecimal (repeating_part : ℕ) (period : ℕ) : ℚ :=
  repeating_part / (10^period - 1)

/-- The sum of three specific repeating decimals -/
theorem sum_of_specific_repeating_decimals :
  RepeatingDecimal 12 2 + RepeatingDecimal 34 3 + RepeatingDecimal 567 5 = 16133 / 99999 := by
  sorry

#eval RepeatingDecimal 12 2 + RepeatingDecimal 34 3 + RepeatingDecimal 567 5

end NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l3360_336007


namespace NUMINAMATH_CALUDE_remaining_credit_after_call_prove_remaining_credit_l3360_336098

/-- Calculates the remaining credit on a prepaid phone card after a call. -/
theorem remaining_credit_after_call 
  (initial_value : ℝ) 
  (cost_per_minute : ℝ) 
  (call_duration : ℕ) 
  (remaining_credit : ℝ) : Prop :=
  initial_value = 30 ∧ 
  cost_per_minute = 0.16 ∧ 
  call_duration = 22 ∧ 
  remaining_credit = initial_value - (cost_per_minute * call_duration) → 
  remaining_credit = 26.48

/-- Proof of the remaining credit calculation. -/
theorem prove_remaining_credit : 
  ∃ (initial_value cost_per_minute : ℝ) (call_duration : ℕ) (remaining_credit : ℝ),
    remaining_credit_after_call initial_value cost_per_minute call_duration remaining_credit :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_credit_after_call_prove_remaining_credit_l3360_336098


namespace NUMINAMATH_CALUDE_remaining_area_after_cutouts_cutout_area_is_nine_dark_gray_rectangle_area_light_gray_rectangle_area_l3360_336019

/-- The area of a square with side length 6, minus specific triangular cutouts, equals 27 -/
theorem remaining_area_after_cutouts (square_side : ℝ) (cutout_area : ℝ) : 
  square_side = 6 → 
  cutout_area = 9 → 
  square_side^2 - cutout_area = 27 := by
  sorry

/-- The area of triangular cutouts in a 6x6 square equals 9 -/
theorem cutout_area_is_nine (dark_gray_rect_area light_gray_rect_area : ℝ) :
  dark_gray_rect_area = 3 →
  light_gray_rect_area = 6 →
  dark_gray_rect_area + light_gray_rect_area = 9 := by
  sorry

/-- The area of a rectangle formed by dark gray triangles is 3 -/
theorem dark_gray_rectangle_area (length width : ℝ) :
  length = 1 →
  width = 3 →
  length * width = 3 := by
  sorry

/-- The area of a rectangle formed by light gray triangles is 6 -/
theorem light_gray_rectangle_area (length width : ℝ) :
  length = 2 →
  width = 3 →
  length * width = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_cutouts_cutout_area_is_nine_dark_gray_rectangle_area_light_gray_rectangle_area_l3360_336019


namespace NUMINAMATH_CALUDE_apartment_number_l3360_336080

theorem apartment_number : ∃! n : ℕ, n ≥ 10 ∧ n < 100 ∧ n = 17 * (n % 10) := by
  sorry

end NUMINAMATH_CALUDE_apartment_number_l3360_336080


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l3360_336031

theorem greatest_integer_with_gcf_five : ∃ n : ℕ, n < 200 ∧ Nat.gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 → Nat.gcd m 30 = 5 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l3360_336031


namespace NUMINAMATH_CALUDE_orange_packing_problem_l3360_336036

/-- Given a total number of oranges and the capacity of each box, 
    calculate the number of boxes needed. -/
def boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) : ℕ :=
  total_oranges / oranges_per_box

/-- Theorem stating that 265 boxes are needed to pack 2650 oranges 
    when each box holds 10 oranges. -/
theorem orange_packing_problem : 
  boxes_needed 2650 10 = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_problem_l3360_336036


namespace NUMINAMATH_CALUDE_checkerboard_corner_sum_l3360_336060

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- The number in the top-left corner -/
def topLeft : Nat := 1

/-- The number in the top-right corner -/
def topRight : Nat := boardSize

/-- The number in the bottom-left corner -/
def bottomLeft : Nat := totalSquares - boardSize + 1

/-- The number in the bottom-right corner -/
def bottomRight : Nat := totalSquares

/-- The sum of the numbers in the four corners of the checkerboard -/
def cornerSum : Nat := topLeft + topRight + bottomLeft + bottomRight

theorem checkerboard_corner_sum : cornerSum = 164 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_corner_sum_l3360_336060


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l3360_336067

theorem two_digit_reverse_sum (x y m : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a) →  -- y is obtained by reversing the digits of x
  (0 < m) →  -- m is a positive integer
  (x^2 - y^2 = 9 * m^2) →  -- given equation
  x + y + 2 * m = 143 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l3360_336067


namespace NUMINAMATH_CALUDE_factorization_a5_minus_a3b2_l3360_336059

theorem factorization_a5_minus_a3b2 (a b : ℝ) : 
  a^5 - a^3 * b^2 = a^3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_a5_minus_a3b2_l3360_336059


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l3360_336003

theorem two_digit_integer_problem :
  ∃ (m n : ℕ),
    10 ≤ m ∧ m < 100 ∧
    10 ≤ n ∧ n < 100 ∧
    (m + n : ℚ) / 2 = n + m / 100 ∧
    m + n < 150 ∧
    m = 50 ∧ n = 49 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l3360_336003


namespace NUMINAMATH_CALUDE_total_trophies_is_950_l3360_336077

/-- The total number of trophies Jack and Michael will have after five years -/
def totalTrophies (michaelCurrent : ℕ) (michaelIncrease : ℕ) (jackMultiplier : ℕ) : ℕ :=
  (michaelCurrent + michaelIncrease) + (jackMultiplier * michaelCurrent)

/-- Proof that the total number of trophies is 950 -/
theorem total_trophies_is_950 :
  totalTrophies 50 150 15 = 950 := by
  sorry

end NUMINAMATH_CALUDE_total_trophies_is_950_l3360_336077


namespace NUMINAMATH_CALUDE_distance_equals_radius_l3360_336018

/-- A circle resting on the x-axis and tangent to the line x=3 -/
structure TangentCircle where
  /-- The x-coordinate of the circle's center -/
  h : ℝ
  /-- The radius of the circle -/
  r : ℝ
  /-- The circle rests on the x-axis and is tangent to x=3 -/
  tangent_condition : r = |3 - h|

/-- The distance from the center to the point of tangency equals the radius -/
theorem distance_equals_radius (c : TangentCircle) :
  |3 - c.h| = c.r := by sorry

end NUMINAMATH_CALUDE_distance_equals_radius_l3360_336018


namespace NUMINAMATH_CALUDE_function_zero_point_l3360_336084

theorem function_zero_point
  (f : ℝ → ℝ)
  (h_mono : Monotone f)
  (h_prop : ∀ x : ℝ, f (f x - 2^x) = -1/2) :
  ∃! x : ℝ, f x = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_point_l3360_336084


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l3360_336058

theorem partial_fraction_sum (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l3360_336058


namespace NUMINAMATH_CALUDE_min_value_expression_l3360_336086

theorem min_value_expression (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x' y' : ℝ), x' * y' + 3 * x' = 3 → 0 < x' → x' < 1/2 → 3 / x' + 1 / (y' - 3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3360_336086


namespace NUMINAMATH_CALUDE_factor_expression_l3360_336066

theorem factor_expression (x : ℝ) : 16 * x^3 + 4 * x^2 = 4 * x^2 * (4 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3360_336066


namespace NUMINAMATH_CALUDE_sheela_deposit_l3360_336064

/-- Sheela's monthly income in Rupees -/
def monthly_income : ℕ := 25000

/-- The percentage of monthly income deposited -/
def deposit_percentage : ℚ := 20 / 100

/-- Calculate the deposit amount based on monthly income and deposit percentage -/
def deposit_amount (income : ℕ) (percentage : ℚ) : ℚ :=
  percentage * income

/-- Theorem stating that Sheela's deposit amount is 5000 Rupees -/
theorem sheela_deposit :
  deposit_amount monthly_income deposit_percentage = 5000 := by
  sorry

end NUMINAMATH_CALUDE_sheela_deposit_l3360_336064


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_2theta_l3360_336035

/-- Given two parallel vectors a and b, prove that cos(2θ) = -1/3 -/
theorem parallel_vectors_cos_2theta (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (Real.cos θ, 1)) 
  (hb : b = (1, 3 * Real.cos θ)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  Real.cos (2 * θ) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_2theta_l3360_336035


namespace NUMINAMATH_CALUDE_knights_and_knaves_l3360_336057

-- Define the type for individuals
inductive Person : Type
| A
| B
| C

-- Define the type for knight/knave status
inductive Status : Type
| Knight
| Knave

-- Function to determine if a person is a knight
def isKnight (p : Person) (s : Person → Status) : Prop :=
  s p = Status.Knight

-- Function to determine if a person is a knave
def isKnave (p : Person) (s : Person → Status) : Prop :=
  s p = Status.Knave

-- A's statement
def A_statement (s : Person → Status) : Prop :=
  isKnight Person.C s → isKnave Person.B s

-- C's statement
def C_statement (s : Person → Status) : Prop :=
  (isKnight Person.A s ∧ isKnave Person.C s) ∨ (isKnave Person.A s ∧ isKnight Person.C s)

-- Main theorem
theorem knights_and_knaves :
  ∃ (s : Person → Status),
    (∀ p, (isKnight p s → A_statement s = true) ∧ (isKnave p s → A_statement s = false)) ∧
    (∀ p, (isKnight p s → C_statement s = true) ∧ (isKnave p s → C_statement s = false)) ∧
    isKnave Person.A s ∧ isKnight Person.B s ∧ isKnight Person.C s :=
sorry

end NUMINAMATH_CALUDE_knights_and_knaves_l3360_336057


namespace NUMINAMATH_CALUDE_shipping_cost_proof_l3360_336042

/-- Calculates the total shipping cost for fish -/
def total_shipping_cost (total_weight : ℕ) (crate_weight : ℕ) (cost_per_crate : ℚ) (surcharge_per_crate : ℚ) (flat_fee : ℚ) : ℚ :=
  let num_crates : ℕ := total_weight / crate_weight
  let crate_total_cost : ℚ := (cost_per_crate + surcharge_per_crate) * num_crates
  crate_total_cost + flat_fee

/-- Proves that the total shipping cost for the given conditions is $46.00 -/
theorem shipping_cost_proof :
  total_shipping_cost 540 30 (3/2) (1/2) 10 = 46 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_proof_l3360_336042


namespace NUMINAMATH_CALUDE_plate_on_square_table_l3360_336071

/-- Given a square table with a round plate, if the distances from the plate to the table edges
    on one side are 10 cm and 63 cm, and on the opposite side are 20 cm and x cm,
    then x = 53 cm. -/
theorem plate_on_square_table (x : ℝ) : x = 53 := by
  sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l3360_336071


namespace NUMINAMATH_CALUDE_same_color_probability_l3360_336001

def total_balls : ℕ := 5 + 8 + 4 + 3

def green_balls : ℕ := 5
def white_balls : ℕ := 8
def blue_balls : ℕ := 4
def red_balls : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem same_color_probability :
  (choose green_balls 4 + choose white_balls 4 + choose blue_balls 4) / choose total_balls 4 = 76 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3360_336001


namespace NUMINAMATH_CALUDE_equal_probability_l3360_336073

/-- The number of black gloves in the pocket -/
def black_gloves : ℕ := 15

/-- The number of white gloves in the pocket -/
def white_gloves : ℕ := 10

/-- The total number of gloves in the pocket -/
def total_gloves : ℕ := black_gloves + white_gloves

/-- The number of ways to choose 2 gloves from n gloves -/
def choose (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of drawing two gloves of the same color -/
def prob_same_color : ℚ :=
  (choose black_gloves + choose white_gloves) / choose total_gloves

/-- The probability of drawing two gloves of different colors -/
def prob_diff_color : ℚ :=
  (black_gloves * white_gloves) / choose total_gloves

theorem equal_probability : prob_same_color = prob_diff_color := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_l3360_336073


namespace NUMINAMATH_CALUDE_standard_equation_of_M_no_B_on_circle_and_M_l3360_336011

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus F₁ and vertex C
def F₁ : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (-2, 0)

-- Theorem for part I
theorem standard_equation_of_M : 
  ∀ x y : ℝ, ellipse_M x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- Theorem for part II
theorem no_B_on_circle_and_M :
  ¬ ∃ x₀ y₀ : ℝ, 
    ellipse_M x₀ y₀ ∧ 
    -2 < x₀ ∧ x₀ < 2 ∧
    (x₀ + 1)^2 + y₀^2 = (x₀ + 2)^2 + y₀^2 :=
sorry

end NUMINAMATH_CALUDE_standard_equation_of_M_no_B_on_circle_and_M_l3360_336011


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l3360_336092

theorem decimal_to_percentage (x : ℝ) (h : x = 0.005) : x * 100 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l3360_336092


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l3360_336079

/-- g(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 65 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m : ℕ, 0 < m → m < 65 → g m ≠ 4) ∧ g 65 = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l3360_336079


namespace NUMINAMATH_CALUDE_gcf_of_90_and_126_l3360_336091

theorem gcf_of_90_and_126 : Nat.gcd 90 126 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_and_126_l3360_336091


namespace NUMINAMATH_CALUDE_rectangles_on_clock_face_l3360_336068

/-- The number of equally spaced points on a circle -/
def n : ℕ := 12

/-- A function that calculates the number of rectangles that can be formed
    by selecting 4 vertices from n equally spaced points on a circle -/
def count_rectangles (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of rectangles formed is 15 when n = 12 -/
theorem rectangles_on_clock_face : count_rectangles n = 15 := by sorry

end NUMINAMATH_CALUDE_rectangles_on_clock_face_l3360_336068


namespace NUMINAMATH_CALUDE_order_of_abc_l3360_336025

-- Define the constants
noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := Real.cos 2
noncomputable def c : ℝ := 2 ^ (1 / 5)

-- State the theorem
theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l3360_336025


namespace NUMINAMATH_CALUDE_tangent_value_l3360_336033

theorem tangent_value (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_value_l3360_336033


namespace NUMINAMATH_CALUDE_equation_solution_l3360_336034

-- Define the equation
def equation (n : ℕ) (x : ℝ) : Prop :=
  (Real.cos x)^n - (Real.sin x)^n = 1

-- Define the solution set
def solution_set (n : ℕ) : Set ℝ :=
  if n % 2 = 0 then
    {x | ∃ k : ℤ, x = k * Real.pi}
  else
    {x | ∃ k : ℤ, x = 2 * k * Real.pi ∨ x = (3 / 2 + 2 * k) * Real.pi}

-- State the theorem
theorem equation_solution (n : ℕ) :
  ∀ x : ℝ, equation n x ↔ x ∈ solution_set n :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3360_336034


namespace NUMINAMATH_CALUDE_factor_of_a_l3360_336065

theorem factor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a := by sorry

end NUMINAMATH_CALUDE_factor_of_a_l3360_336065


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3360_336005

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * (a^2 + b*c))/(b + c) + (b * (b^2 + a*c))/(a + c) + (c * (c^2 + a*b))/(a + b) ≥
  a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3360_336005


namespace NUMINAMATH_CALUDE_prob_no_eight_correct_l3360_336000

/-- The probability of selecting a number from 1 to 10000 that doesn't contain the digit 8 -/
def prob_no_eight : ℚ :=
  (9^4 : ℚ) / 10000

/-- Theorem stating that the probability of selecting a number from 1 to 10000
    that doesn't contain the digit 8 is equal to (9^4) / 10000 -/
theorem prob_no_eight_correct :
  prob_no_eight = (9^4 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_prob_no_eight_correct_l3360_336000


namespace NUMINAMATH_CALUDE_line_translation_proof_l3360_336008

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The vertical translation distance between two lines with the same slope -/
def verticalTranslation (l1 l2 : Line) : ℝ :=
  l2.yIntercept - l1.yIntercept

theorem line_translation_proof (l1 l2 : Line) 
  (h1 : l1.slope = 3 ∧ l1.yIntercept = -1)
  (h2 : l2.slope = 3 ∧ l2.yIntercept = 6)
  : verticalTranslation l1 l2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_proof_l3360_336008


namespace NUMINAMATH_CALUDE_find_divisor_l3360_336070

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3360_336070


namespace NUMINAMATH_CALUDE_cos_squared_derivative_l3360_336048

theorem cos_squared_derivative (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = (Real.cos (2 * x))^2) →
  (deriv f) x = -2 * Real.sin (4 * x) := by
sorry

end NUMINAMATH_CALUDE_cos_squared_derivative_l3360_336048


namespace NUMINAMATH_CALUDE_symposium_partition_exists_l3360_336088

/-- Represents a symposium with delegates and their acquaintances. -/
structure Symposium where
  delegates : Finset Nat
  acquainted : Nat → Nat → Prop
  acquainted_symmetric : ∀ a b, acquainted a b ↔ acquainted b a
  acquainted_irreflexive : ∀ a, ¬acquainted a a
  has_acquaintance : ∀ a ∈ delegates, ∃ b ∈ delegates, a ≠ b ∧ acquainted a b
  not_all_acquainted : ∀ a ∈ delegates, ∃ b ∈ delegates, a ≠ b ∧ ¬acquainted a b

/-- Represents a partition of delegates into two groups. -/
structure Partition (s : Symposium) where
  group1 : Finset Nat
  group2 : Finset Nat
  covers : group1 ∪ group2 = s.delegates
  disjoint : group1 ∩ group2 = ∅
  nonempty : group1.Nonempty ∧ group2.Nonempty

/-- The main theorem stating that a valid partition exists for any symposium. -/
theorem symposium_partition_exists (s : Symposium) :
  ∃ p : Partition s, ∀ a ∈ s.delegates,
    (a ∈ p.group1 → ∃ b ∈ p.group1, a ≠ b ∧ s.acquainted a b) ∧
    (a ∈ p.group2 → ∃ b ∈ p.group2, a ≠ b ∧ s.acquainted a b) :=
  sorry

end NUMINAMATH_CALUDE_symposium_partition_exists_l3360_336088


namespace NUMINAMATH_CALUDE_eliana_steps_l3360_336089

def steps_day1 (x : ℕ) := 200 + x
def steps_day2 (x : ℕ) := 2 * steps_day1 x
def steps_day3 (x : ℕ) := steps_day2 x + 100

theorem eliana_steps (x : ℕ) :
  steps_day1 x + steps_day2 x + steps_day3 x = 1600 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_l3360_336089


namespace NUMINAMATH_CALUDE_fourth_term_is_six_l3360_336037

/-- An increasing sequence of positive integers satisfying a_{a_n} = 2n + 1 -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, n < m → a n < a m) ∧ 
  (∀ n : ℕ+, a (a n) = 2*n + 1)

/-- The fourth term of the special sequence is 6 -/
theorem fourth_term_is_six (a : ℕ → ℕ) (h : SpecialSequence a) : a 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_six_l3360_336037


namespace NUMINAMATH_CALUDE_hailstone_conjecture_instance_l3360_336085

def hailstone_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = if a n % 2 = 0 then a n / 2 else 3 * a n + 1

theorem hailstone_conjecture_instance : ∃ a₁ : ℕ, a₁ < 50 ∧
  (∃ a : ℕ → ℕ, hailstone_seq a ∧ a 1 = a₁ ∧ a 10 = 1 ∧ ∀ i ∈ Finset.range 9, a (i + 1) ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_hailstone_conjecture_instance_l3360_336085


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l3360_336090

theorem set_equality_implies_difference (a b : ℝ) :
  ({a, 1} : Set ℝ) = {0, a + b} → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l3360_336090


namespace NUMINAMATH_CALUDE_sum_of_divisors_154_l3360_336017

/-- The sum of all positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all positive divisors of 154 is 288 -/
theorem sum_of_divisors_154 : sum_of_divisors 154 = 288 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_154_l3360_336017


namespace NUMINAMATH_CALUDE_problem_solution_l3360_336040

theorem problem_solution (a : ℝ) (h : a^2 + a = 0) : a^2011 + a^2010 + 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3360_336040


namespace NUMINAMATH_CALUDE_find_m_l3360_336016

def A (m : ℕ) : Set ℝ := {x : ℝ | (m * x - 1) / x < 0}

def B : Set ℝ := {x : ℝ | 2 * x^2 - x < 0}

def is_necessary_not_sufficient (A B : Set ℝ) : Prop :=
  B ⊆ A ∧ A ≠ B

theorem find_m :
  ∃ (m : ℕ), m > 0 ∧ m < 6 ∧ is_necessary_not_sufficient (A m) B ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_find_m_l3360_336016


namespace NUMINAMATH_CALUDE_work_increase_percentage_l3360_336030

/-- Proves that when 1/7 of the members are absent in an office, 
    the percentage increase in work for each remaining person is 100/6. -/
theorem work_increase_percentage (p : ℝ) (p_pos : p > 0) : 
  let absent_fraction : ℝ := 1/7
  let remaining_fraction : ℝ := 1 - absent_fraction
  let work_increase_ratio : ℝ := 1 / remaining_fraction
  let percentage_increase : ℝ := (work_increase_ratio - 1) * 100
  percentage_increase = 100/6 := by
sorry

#eval (100 : ℚ) / 6  -- To show the approximate decimal value

end NUMINAMATH_CALUDE_work_increase_percentage_l3360_336030


namespace NUMINAMATH_CALUDE_flippers_win_probability_l3360_336097

theorem flippers_win_probability :
  let n : ℕ := 6  -- Total number of games
  let k : ℕ := 4  -- Number of games to win
  let p : ℚ := 3/5  -- Probability of winning a single game
  Nat.choose n k * p^k * (1-p)^(n-k) = 4860/15625 := by
sorry

end NUMINAMATH_CALUDE_flippers_win_probability_l3360_336097


namespace NUMINAMATH_CALUDE_cabin_rental_security_deposit_l3360_336083

/-- Calculate the security deposit for a cabin rental --/
theorem cabin_rental_security_deposit :
  let rental_period : ℕ := 14 -- 2 weeks
  let daily_rate : ℚ := 125
  let pet_fee : ℚ := 100
  let service_fee_rate : ℚ := 1/5 -- 20%
  let security_deposit_rate : ℚ := 1/2 -- 50%

  let rental_cost := rental_period * daily_rate
  let subtotal := rental_cost + pet_fee
  let service_fee := subtotal * service_fee_rate
  let total_cost := subtotal + service_fee
  let security_deposit := total_cost * security_deposit_rate

  security_deposit = 1110
  := by sorry

end NUMINAMATH_CALUDE_cabin_rental_security_deposit_l3360_336083


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3360_336002

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → (a^2 + b^2 = 48) := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3360_336002


namespace NUMINAMATH_CALUDE_rectangle_sections_3x5_l3360_336032

/-- The number of rectangular sections (including squares) in a grid --/
def rectangleCount (width height : ℕ) : ℕ :=
  let squareCount := (width * (width + 1) * height * (height + 1)) / 4
  let rectangleCount := (width * (width + 1) * height * (height + 1)) / 4 - (width * height)
  squareCount + rectangleCount

/-- Theorem stating that the number of rectangular sections in a 3x5 grid is 72 --/
theorem rectangle_sections_3x5 :
  rectangleCount 3 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sections_3x5_l3360_336032


namespace NUMINAMATH_CALUDE_jane_sunflower_seeds_l3360_336054

/-- Calculates the total number of sunflower seeds given the number of cans and seeds per can. -/
def total_seeds (num_cans : ℕ) (seeds_per_can : ℕ) : ℕ :=
  num_cans * seeds_per_can

/-- Theorem stating that 9 cans with 6 seeds each results in 54 total seeds. -/
theorem jane_sunflower_seeds :
  total_seeds 9 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_jane_sunflower_seeds_l3360_336054


namespace NUMINAMATH_CALUDE_m_range_l3360_336075

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 - 2*m*x + 7*m - 10 ≠ 0

def q (m : ℝ) : Prop := ∀ x > 0, x^2 - m*x + 4 ≥ 0

-- State the theorem
theorem m_range (m : ℝ) 
  (h1 : p m ∨ q m) 
  (h2 : p m ∧ q m) : 
  m ∈ Set.Ioo 2 4 ∪ {4} :=
sorry

end NUMINAMATH_CALUDE_m_range_l3360_336075


namespace NUMINAMATH_CALUDE_vasil_can_win_more_l3360_336015

/-- Represents the possible objects on a coin side -/
inductive Object
| Scissors
| Paper
| Rock

/-- Represents a coin with two sides -/
structure Coin where
  side1 : Object
  side2 : Object

/-- The set of available coins -/
def coins : List Coin := [
  ⟨Object.Scissors, Object.Paper⟩,
  ⟨Object.Rock, Object.Scissors⟩,
  ⟨Object.Paper, Object.Rock⟩
]

/-- Determines if object1 beats object2 -/
def beats (object1 object2 : Object) : Bool :=
  match object1, object2 with
  | Object.Scissors, Object.Paper => true
  | Object.Paper, Object.Rock => true
  | Object.Rock, Object.Scissors => true
  | _, _ => false

/-- Calculates the probability of Vasil winning against Asya -/
def winProbability (asyaCoin vasilCoin : Coin) : Rat :=
  sorry

/-- Theorem stating that Vasil can choose a coin to have a higher winning probability -/
theorem vasil_can_win_more : ∃ (strategy : Coin → Coin),
  ∀ (asyaChoice : Coin),
    asyaChoice ∈ coins →
    winProbability asyaChoice (strategy asyaChoice) > 1/2 :=
  sorry

end NUMINAMATH_CALUDE_vasil_can_win_more_l3360_336015


namespace NUMINAMATH_CALUDE_function_properties_l3360_336069

/-- Given a function f(x) = a*sin(x) + b*cos(x) where ab ≠ 0, a and b are real numbers,
    and for all x in ℝ, f(x) ≥ f(5π/6), then:
    1. f(π/3) = 0
    2. The line passing through (a, b) intersects the graph of f(x) -/
theorem function_properties (a b : ℝ) (h1 : a * b ≠ 0) :
  let f := fun (x : ℝ) ↦ a * Real.sin x + b * Real.cos x
  (∀ x : ℝ, f x ≥ f (5 * Real.pi / 6)) →
  (f (Real.pi / 3) = 0) ∧
  (∃ x : ℝ, f x = a * x + b) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3360_336069


namespace NUMINAMATH_CALUDE_not_zero_necessary_not_sufficient_for_positive_l3360_336023

theorem not_zero_necessary_not_sufficient_for_positive (x : ℝ) :
  (∃ x, x ≠ 0 ∧ x ≤ 0) ∧ (∀ x, x > 0 → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_not_zero_necessary_not_sufficient_for_positive_l3360_336023


namespace NUMINAMATH_CALUDE_complex_modulus_sum_l3360_336094

theorem complex_modulus_sum : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 5*Complex.I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_l3360_336094


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3360_336004

theorem sum_of_numbers : let numbers := [0.8, 1/2, 0.5]
  (∀ x ∈ numbers, x ≤ 2) →
  numbers.sum = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3360_336004


namespace NUMINAMATH_CALUDE_caravan_keepers_l3360_336039

/-- The number of keepers in a caravan with hens, goats, and camels. -/
def num_keepers : ℕ := by sorry

theorem caravan_keepers :
  let hens : ℕ := 50
  let goats : ℕ := 45
  let camels : ℕ := 8
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_head : ℕ := 1
  let keeper_feet : ℕ := 2
  let total_animal_feet : ℕ := hens * hen_feet + goats * goat_feet + camels * camel_feet
  let total_animal_heads : ℕ := hens + goats + camels
  let extra_feet : ℕ := 224
  num_keepers * keeper_feet + total_animal_feet = num_keepers * keeper_head + total_animal_heads + extra_feet →
  num_keepers = 15 := by sorry

end NUMINAMATH_CALUDE_caravan_keepers_l3360_336039


namespace NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3360_336096

/-- Calculates the length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- Proves that the platform length is 340 m given the specific parameters --/
theorem platform_length_proof :
  platform_length 160 72 25 = 340 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3360_336096


namespace NUMINAMATH_CALUDE_hyperbola_focus_directrix_distance_example_l3360_336076

/-- The distance between the right focus and left directrix of a hyperbola -/
def hyperbola_focus_directrix_distance (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  5

/-- Theorem: The distance between the right focus and left directrix of the hyperbola x²/4 - y²/12 = 1 is 5 -/
theorem hyperbola_focus_directrix_distance_example :
  hyperbola_focus_directrix_distance 2 (2 * Real.sqrt 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_directrix_distance_example_l3360_336076


namespace NUMINAMATH_CALUDE_correct_ages_l3360_336055

/-- Represents the ages of Albert, Mary, Betty, and Carol -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ
  carol : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 10 ∧
  ages.carol = ages.betty + 3 ∧
  ages.carol = ages.mary / 2

/-- The theorem to prove -/
theorem correct_ages :
  ∃ (ages : Ages), satisfiesConditions ages ∧
    ages.albert = 20 ∧
    ages.mary = 10 ∧
    ages.betty = 2 ∧
    ages.carol = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_ages_l3360_336055


namespace NUMINAMATH_CALUDE_circumradius_side_ratio_not_unique_l3360_336053

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The length of a side of a triangle -/
def side_length (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- The shape of a triangle, represented by its angles -/
def triangle_shape (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The ratio of circumradius to one side does not uniquely determine triangle shape -/
theorem circumradius_side_ratio_not_unique (r : ℝ) (side : Fin 3) :
  ∃ t1 t2 : Triangle, 
    circumradius t1 / side_length t1 side = r ∧
    circumradius t2 / side_length t2 side = r ∧
    triangle_shape t1 ≠ triangle_shape t2 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_side_ratio_not_unique_l3360_336053


namespace NUMINAMATH_CALUDE_victoria_wins_l3360_336082

/-- Represents a player in the game -/
inductive Player : Type
| Harry : Player
| Victoria : Player

/-- Represents a line segment on the grid -/
inductive Segment : Type
| EastWest : Segment
| NorthSouth : Segment

/-- Represents the state of the game -/
structure GameState :=
(turn : Player)
(harry_score : Nat)
(victoria_score : Nat)
(moves : List Segment)

/-- Represents a strategy for a player -/
def Strategy := GameState → Segment

/-- Determines if a move is valid for a given player -/
def valid_move (player : Player) (segment : Segment) : Bool :=
  match player, segment with
  | Player.Harry, Segment.EastWest => true
  | Player.Victoria, Segment.NorthSouth => true
  | _, _ => false

/-- Determines if a move completes a square -/
def completes_square (state : GameState) (segment : Segment) : Bool :=
  sorry -- Implementation details omitted

/-- Applies a move to the game state -/
def apply_move (state : GameState) (segment : Segment) : GameState :=
  sorry -- Implementation details omitted

/-- Determines if the game is over -/
def game_over (state : GameState) : Bool :=
  sorry -- Implementation details omitted

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Player :=
  sorry -- Implementation details omitted

/-- Victoria's winning strategy -/
def victoria_strategy : Strategy :=
  sorry -- Implementation details omitted

/-- Theorem stating that Victoria has a winning strategy -/
theorem victoria_wins :
  ∀ (harry_strategy : Strategy),
  ∃ (final_state : GameState),
  (game_over final_state = true) ∧
  (winner final_state = some Player.Victoria) :=
sorry

end NUMINAMATH_CALUDE_victoria_wins_l3360_336082


namespace NUMINAMATH_CALUDE_locus_is_ellipse_and_angle_bisector_l3360_336010

-- Define the circle A
def circle_A (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 15 = 0

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the locus of points N
def locus_N (x y : ℝ) : Prop :=
  ∃ (mx my : ℝ),
    circle_A mx my ∧
    (x - mx)^2 + (y - my)^2 = (x - point_B.1)^2 + (y - point_B.2)^2 ∧
    (x - mx) * (1 - mx) + (y - my) * (0 - my) = 0

-- Theorem statement
theorem locus_is_ellipse_and_angle_bisector :
  (∀ x y, locus_N x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ k x₁ y₁ x₂ y₂,
    x₁^2/4 + y₁^2/3 = 1 ∧
    x₂^2/4 + y₂^2/3 = 1 ∧
    y₁ = k*(x₁ - 1) ∧
    y₂ = k*(x₂ - 1) →
    (y₁ / (x₁ - 4) + y₂ / (x₂ - 4) = 0)) :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_and_angle_bisector_l3360_336010


namespace NUMINAMATH_CALUDE_boots_discounted_price_l3360_336078

/-- Calculates the discounted price of an item given its original price and discount percentage. -/
def discountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Proves that the discounted price of boots with an original price of $90 and a 20% discount is $72. -/
theorem boots_discounted_price :
  discountedPrice 90 20 = 72 := by
  sorry

end NUMINAMATH_CALUDE_boots_discounted_price_l3360_336078


namespace NUMINAMATH_CALUDE_ball_probability_l3360_336062

theorem ball_probability (P_A P_B P_C : ℝ) 
  (h1 : P_A + P_B = 0.4)
  (h2 : P_A + P_C = 0.9)
  (h3 : P_A + P_B + P_C = 1) : 
  P_B + P_C = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3360_336062


namespace NUMINAMATH_CALUDE_passing_marks_l3360_336038

/-- Proves that the passing marks is 120 given the conditions of the problem -/
theorem passing_marks (T : ℝ) (P : ℝ) 
  (h1 : 0.30 * T = P - 30)
  (h2 : 0.45 * T = P + 15) : P = 120 := by
  sorry

end NUMINAMATH_CALUDE_passing_marks_l3360_336038


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3360_336049

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (6, 3)
  let b : ℝ → ℝ × ℝ := fun m ↦ (m, 2)
  ∀ m : ℝ, are_parallel a (b m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3360_336049


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_max_sum_on_C₂_l3360_336014

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (ρ θ : ℝ) : Prop := ρ * (Real.sin θ + Real.cos θ) = 1
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
    C₁ ρ₁ θ₁ ∧ C₂ ρ₁ θ₁ ∧
    C₁ ρ₂ θ₂ ∧ C₂ ρ₂ θ₂ ∧
    A = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧
    B = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    A ≠ B

-- Theorem 1: Distance between intersection points
theorem distance_between_intersection_points
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
sorry

-- Define a point on C₂ in Cartesian coordinates
def point_on_C₂ (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), C₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem 2: Maximum value of x + y for points on C₂
theorem max_sum_on_C₂ :
  ∃ (M : ℝ), M = Real.sqrt 10 - 1 ∧
  (∀ x y, point_on_C₂ x y → x + y ≤ M) ∧
  (∃ x y, point_on_C₂ x y ∧ x + y = M) :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_max_sum_on_C₂_l3360_336014


namespace NUMINAMATH_CALUDE_degree_of_product_polynomial_l3360_336046

/-- The degree of a polynomial (x^2+1)^5 * (x^3+1)^2 * (x+1)^3 -/
theorem degree_of_product_polynomial : ∃ (p : Polynomial ℝ), 
  p = (X^2 + 1)^5 * (X^3 + 1)^2 * (X + 1)^3 ∧ 
  Polynomial.degree p = 19 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_product_polynomial_l3360_336046


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3360_336020

theorem expression_equals_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) :
  (x + 1/x) * (y - 1/y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3360_336020


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l3360_336087

theorem lcm_24_36_45 : Nat.lcm 24 (Nat.lcm 36 45) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l3360_336087


namespace NUMINAMATH_CALUDE_rhombus_area_l3360_336093

/-- The area of a rhombus with side length 4 cm and an angle of 45 degrees between adjacent sides is 8√2 square centimeters. -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 4 →
  angle = π / 4 →
  let area := side_length * side_length * Real.sin angle
  area = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3360_336093


namespace NUMINAMATH_CALUDE_drill_bits_purchase_cost_l3360_336072

/-- The total cost of a purchase with tax -/
def total_cost (num_sets : ℕ) (price_per_set : ℚ) (tax_rate : ℚ) : ℚ :=
  let pre_tax_cost := num_sets * price_per_set
  let tax := pre_tax_cost * tax_rate
  pre_tax_cost + tax

/-- Theorem: The total cost for 5 sets of drill bits at $6 each with 10% tax is $33 -/
theorem drill_bits_purchase_cost :
  total_cost 5 6 (1/10) = 33 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_purchase_cost_l3360_336072


namespace NUMINAMATH_CALUDE_max_surface_area_30_cubes_l3360_336028

/-- Represents a configuration of connected unit cubes -/
structure CubeConfiguration where
  num_cubes : ℕ
  surface_area : ℕ

/-- The number of cubes in our problem -/
def total_cubes : ℕ := 30

/-- Function to calculate the surface area of a linear arrangement of cubes -/
def linear_arrangement_surface_area (n : ℕ) : ℕ :=
  if n ≤ 1 then 6 * n else 2 + 4 * n

/-- Theorem stating that the maximum surface area for 30 connected unit cubes is 122 -/
theorem max_surface_area_30_cubes :
  (∀ c : CubeConfiguration, c.num_cubes = total_cubes → c.surface_area ≤ 122) ∧
  (∃ c : CubeConfiguration, c.num_cubes = total_cubes ∧ c.surface_area = 122) := by
  sorry

#eval linear_arrangement_surface_area total_cubes

end NUMINAMATH_CALUDE_max_surface_area_30_cubes_l3360_336028


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3360_336099

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 + I) / (2 - I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3360_336099


namespace NUMINAMATH_CALUDE_power_of_power_three_l3360_336045

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l3360_336045


namespace NUMINAMATH_CALUDE_alphametic_puzzle_solution_l3360_336021

theorem alphametic_puzzle_solution :
  ∃! (A R K : ℕ),
    A < 10 ∧ R < 10 ∧ K < 10 ∧
    A ≠ R ∧ A ≠ K ∧ R ≠ K ∧
    1000 * A + 100 * R + 10 * K + A +
    100 * R + 10 * K + A +
    10 * K + A +
    A = 2014 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_solution_l3360_336021


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_six_l3360_336051

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 21

theorem probability_sum_greater_than_six :
  (favorable_outcomes : ℚ) / dice_outcomes = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_six_l3360_336051


namespace NUMINAMATH_CALUDE_prime_triplet_existence_l3360_336050

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_triplet_existence :
  (∃ n : ℕ, isPrime (n - 96) ∧ isPrime n ∧ isPrime (n + 96)) ∧
  (¬∃ n : ℕ, isPrime (n - 1996) ∧ isPrime n ∧ isPrime (n + 1996)) :=
sorry

end NUMINAMATH_CALUDE_prime_triplet_existence_l3360_336050


namespace NUMINAMATH_CALUDE_box_length_l3360_336027

/-- The length of a rectangular box given specific conditions --/
theorem box_length (width : ℝ) (volume_gallons : ℝ) (height_inches : ℝ) (conversion_factor : ℝ) : 
  width = 25 →
  volume_gallons = 4687.5 →
  height_inches = 6 →
  conversion_factor = 7.5 →
  ∃ (length : ℝ), length = 50 := by
sorry

end NUMINAMATH_CALUDE_box_length_l3360_336027


namespace NUMINAMATH_CALUDE_negation_equivalence_l3360_336006

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x < x) ↔ (∀ x : ℝ, Real.exp x ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3360_336006


namespace NUMINAMATH_CALUDE_sanity_determination_question_exists_l3360_336041

/-- Represents the sanity state of a guest -/
inductive Sanity
| Sane
| Insane

/-- Represents possible answers to a question -/
inductive Answer
| Yes
| Ball

/-- A function representing how a guest answers a question based on their sanity -/
def guest_answer (s : Sanity) : Answer :=
  match s with
  | Sanity.Sane => Answer.Ball
  | Sanity.Insane => Answer.Yes

/-- The theorem stating that there exists a question that can determine a guest's sanity -/
theorem sanity_determination_question_exists :
  ∃ (question : Sanity → Answer),
    (∀ s : Sanity, question s = guest_answer s) ∧
    (∀ s₁ s₂ : Sanity, question s₁ = question s₂ → s₁ = s₂) :=
by sorry

end NUMINAMATH_CALUDE_sanity_determination_question_exists_l3360_336041
