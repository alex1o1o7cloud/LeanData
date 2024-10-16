import Mathlib

namespace NUMINAMATH_CALUDE_older_ate_twelve_l1770_177077

/-- Represents the pancake eating scenario -/
structure PancakeScenario where
  initial_pancakes : ℕ
  final_pancakes : ℕ
  younger_eats : ℕ
  older_eats : ℕ
  grandma_bakes : ℕ

/-- Calculates the number of pancakes eaten by the older grandchild -/
def older_grandchild_pancakes (scenario : PancakeScenario) : ℕ :=
  let net_reduction := scenario.younger_eats + scenario.older_eats - scenario.grandma_bakes
  let cycles := (scenario.initial_pancakes - scenario.final_pancakes) / net_reduction
  scenario.older_eats * cycles

/-- Theorem stating that the older grandchild ate 12 pancakes in the given scenario -/
theorem older_ate_twelve (scenario : PancakeScenario) 
  (h1 : scenario.initial_pancakes = 19)
  (h2 : scenario.final_pancakes = 11)
  (h3 : scenario.younger_eats = 1)
  (h4 : scenario.older_eats = 3)
  (h5 : scenario.grandma_bakes = 2) :
  older_grandchild_pancakes scenario = 12 := by
  sorry

#eval older_grandchild_pancakes { 
  initial_pancakes := 19, 
  final_pancakes := 11, 
  younger_eats := 1, 
  older_eats := 3, 
  grandma_bakes := 2 
}

end NUMINAMATH_CALUDE_older_ate_twelve_l1770_177077


namespace NUMINAMATH_CALUDE_segment_existence_l1770_177048

theorem segment_existence (pencil_length eraser_length : ℝ) 
  (h_pencil : pencil_length > 0) (h_eraser : eraser_length > 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = pencil_length ∧ Real.sqrt (x * y) = eraser_length :=
sorry

end NUMINAMATH_CALUDE_segment_existence_l1770_177048


namespace NUMINAMATH_CALUDE_interior_edge_sum_is_twenty_l1770_177090

/-- A rectangular picture frame with specific properties -/
structure PictureFrame where
  outer_length : ℝ
  outer_width : ℝ
  frame_width : ℝ
  frame_area : ℝ
  outer_length_given : outer_length = 7
  frame_width_given : frame_width = 1
  frame_area_given : frame_area = 24
  positive_dimensions : outer_length > 0 ∧ outer_width > 0

/-- The sum of the interior edge lengths of the picture frame -/
def interior_edge_sum (frame : PictureFrame) : ℝ :=
  2 * (frame.outer_length - 2 * frame.frame_width) + 2 * (frame.outer_width - 2 * frame.frame_width)

/-- Theorem stating that the sum of interior edge lengths is 20 -/
theorem interior_edge_sum_is_twenty (frame : PictureFrame) :
  interior_edge_sum frame = 20 := by
  sorry


end NUMINAMATH_CALUDE_interior_edge_sum_is_twenty_l1770_177090


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1770_177094

theorem cube_equation_solution :
  ∃! x : ℝ, (x + 3)^3 = (1/27)⁻¹ :=
by
  -- The unique solution is x = 0
  use 0
  constructor
  · -- Prove that x = 0 satisfies the equation
    sorry
  · -- Prove that this is the only solution
    sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1770_177094


namespace NUMINAMATH_CALUDE_fabian_accessories_cost_l1770_177064

def mouse_cost : ℕ := 16

def keyboard_cost (m : ℕ) : ℕ := 3 * m

def total_cost (m k : ℕ) : ℕ := m + k

theorem fabian_accessories_cost :
  total_cost mouse_cost (keyboard_cost mouse_cost) = 64 := by
  sorry

end NUMINAMATH_CALUDE_fabian_accessories_cost_l1770_177064


namespace NUMINAMATH_CALUDE_expression_simplification_l1770_177018

theorem expression_simplification (a b c : ℝ) 
  (ha : a ≠ 2) (hb : b ≠ 3) (hc : c ≠ 6) : 
  ((a - 2) / (6 - c)) * ((b - 3) / (2 - a)) * ((c - 6) / (3 - b)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1770_177018


namespace NUMINAMATH_CALUDE_probability_three_ones_or_twos_in_five_rolls_l1770_177035

-- Define the probability of rolling a 1 or 2 on a fair six-sided die
def prob_one_or_two : ℚ := 1 / 3

-- Define the probability of not rolling a 1 or 2 on a fair six-sided die
def prob_not_one_or_two : ℚ := 2 / 3

-- Define the number of rolls
def num_rolls : ℕ := 5

-- Define the number of times we want to roll a 1 or 2
def target_rolls : ℕ := 3

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem probability_three_ones_or_twos_in_five_rolls :
  (binomial num_rolls target_rolls : ℚ) * prob_one_or_two ^ target_rolls * prob_not_one_or_two ^ (num_rolls - target_rolls) = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_ones_or_twos_in_five_rolls_l1770_177035


namespace NUMINAMATH_CALUDE_bird_puzzle_solution_l1770_177057

theorem bird_puzzle_solution :
  ∃! (x y z : ℕ),
    x + y + z = 30 ∧
    (x : ℚ) / 3 + (y : ℚ) / 2 + 2 * (z : ℚ) = 30 ∧
    x = 9 ∧ y = 10 ∧ z = 11 := by
  sorry

end NUMINAMATH_CALUDE_bird_puzzle_solution_l1770_177057


namespace NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half_l1770_177080

theorem tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half :
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half_l1770_177080


namespace NUMINAMATH_CALUDE_box_width_is_15_l1770_177017

/-- Given a rectangular box with length 8 cm and height 5 cm, built using 10 cubic cm cubes,
    and requiring a minimum of 60 cubes, prove that the width of the box is 15 cm. -/
theorem box_width_is_15 (length : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) :
  length = 8 →
  height = 5 →
  cube_volume = 10 →
  min_cubes = 60 →
  (min_cubes : ℝ) * cube_volume / (length * height) = 15 := by
  sorry

end NUMINAMATH_CALUDE_box_width_is_15_l1770_177017


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1770_177005

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 5) :
  let R := (A - P) / (P * T) * 100
  R = 4 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1770_177005


namespace NUMINAMATH_CALUDE_a_share_is_288_l1770_177049

/-- Calculates the share of profit for investor A given the initial investments,
    changes after 8 months, and total profit over a year. -/
def calculate_share_a (a_initial : ℕ) (b_initial : ℕ) (a_change : ℕ) (b_change : ℕ) (total_profit : ℕ) : ℕ :=
  let a_investment_months := a_initial * 8 + (a_initial - a_change) * 4
  let b_investment_months := b_initial * 8 + (b_initial + b_change) * 4
  let total_investment_months := a_investment_months + b_investment_months
  let a_ratio := a_investment_months * total_profit / total_investment_months
  a_ratio

/-- Theorem stating that A's share of the profit is 288 Rs given the problem conditions. -/
theorem a_share_is_288 :
  calculate_share_a 3000 4000 1000 1000 756 = 288 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_288_l1770_177049


namespace NUMINAMATH_CALUDE_number_equality_l1770_177078

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (49/216) * (1/x)) : x = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1770_177078


namespace NUMINAMATH_CALUDE_equation_equality_l1770_177092

theorem equation_equality (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) 
  (hac : a + c = 10) : 
  (10 * b + a) * (10 * b + c) = 100 * b * (b + 1) + a * c := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1770_177092


namespace NUMINAMATH_CALUDE_square_point_probability_l1770_177020

/-- Represents a point on the 3x3 square grid --/
inductive SquarePoint
| Corner
| Midpoint
| Center

/-- The set of all points on the 3x3 square --/
def square_points : Finset SquarePoint := sorry

/-- Two points are considered adjacent if they are one unit apart --/
def adjacent (p q : SquarePoint) : Prop := sorry

/-- The number of pairs of adjacent points --/
def num_adjacent_pairs : Nat := sorry

theorem square_point_probability :
  let total_pairs := Nat.choose (Finset.card square_points) 2
  Nat.cast num_adjacent_pairs / Nat.cast total_pairs = 16 / 45 := by sorry

end NUMINAMATH_CALUDE_square_point_probability_l1770_177020


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1770_177086

/-- The quadratic function f(x) = x^2 + 4x - 5 has a minimum value of -9 at x = -2 -/
theorem quadratic_minimum : ∃ (f : ℝ → ℝ), 
  (∀ x, f x = x^2 + 4*x - 5) ∧ 
  (∀ x, f x ≥ f (-2)) ∧
  f (-2) = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1770_177086


namespace NUMINAMATH_CALUDE_fixed_point_of_arcsin_function_l1770_177043

theorem fixed_point_of_arcsin_function (m : ℝ) :
  ∃ (P : ℝ × ℝ), P = (0, -1) ∧ ∀ x : ℝ, m * Real.arcsin x - 1 = P.2 ↔ x = P.1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_arcsin_function_l1770_177043


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_exact_l1770_177025

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 4/w = 1 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_exact (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + 2*y = 9 + 4*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_exact_l1770_177025


namespace NUMINAMATH_CALUDE_unique_solution_for_circ_equation_l1770_177082

-- Define the operation ∘
def circ (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

-- Theorem statement
theorem unique_solution_for_circ_equation :
  ∃! y : ℝ, circ 2 y = 10 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_circ_equation_l1770_177082


namespace NUMINAMATH_CALUDE_decimal_division_l1770_177026

theorem decimal_division (x y : ℚ) (hx : x = 0.12) (hy : y = 0.04) :
  x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l1770_177026


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1770_177068

/-- The function g(x) defined as x^2 + bx + 1 -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- Theorem stating that -3 is not in the range of g(x) if and only if b is in the open interval (-4, 4) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, g b x ≠ -3) ↔ b ∈ Set.Ioo (-4 : ℝ) 4 :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1770_177068


namespace NUMINAMATH_CALUDE_equation_solution_l1770_177037

theorem equation_solution (x : ℝ) : (x + 2)^(x + 3) = 1 → x = -3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1770_177037


namespace NUMINAMATH_CALUDE_base4_calculation_l1770_177038

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplication in base 4 --/
def mulBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a * base4ToBase10 b)

/-- Division in base 4 --/
def divBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a / base4ToBase10 b)

theorem base4_calculation : 
  mulBase4 (divBase4 321 3) 21 = 2223 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l1770_177038


namespace NUMINAMATH_CALUDE_nested_fraction_value_l1770_177006

theorem nested_fraction_value : 1 + 2 / (3 + 4/5) = 29/19 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_value_l1770_177006


namespace NUMINAMATH_CALUDE_ap_length_l1770_177015

/-- Square with inscribed circle -/
structure SquareWithCircle where
  /-- Side length of the square -/
  side_length : ℝ
  /-- The square ABCD -/
  square : Set (ℝ × ℝ)
  /-- The inscribed circle ω -/
  circle : Set (ℝ × ℝ)
  /-- Point A of the square -/
  A : ℝ × ℝ
  /-- Point M where the circle intersects CD -/
  M : ℝ × ℝ
  /-- Point P where AM intersects the circle (different from M) -/
  P : ℝ × ℝ
  /-- The side length is 2 -/
  h_side_length : side_length = 2
  /-- A is a vertex of the square -/
  h_A_in_square : A ∈ square
  /-- M is on the circle and on the side CD -/
  h_M_on_circle_and_CD : M ∈ circle ∧ M.2 = -1
  /-- P is on the circle and on line AM -/
  h_P_on_circle_and_AM : P ∈ circle ∧ P ≠ M ∧ ∃ t : ℝ, P = (1 - t) • A + t • M

/-- The length of AP in a square with inscribed circle is √5/5 -/
theorem ap_length (swc : SquareWithCircle) : Real.sqrt 5 / 5 = ‖swc.A - swc.P‖ := by
  sorry

end NUMINAMATH_CALUDE_ap_length_l1770_177015


namespace NUMINAMATH_CALUDE_common_tangents_count_l1770_177034

/-- The number of common tangents between two circles -/
def num_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ := sorry

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- Theorem stating that the number of common tangents between C₁ and C₂ is 4 -/
theorem common_tangents_count : num_common_tangents C₁ C₂ = 4 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1770_177034


namespace NUMINAMATH_CALUDE_function_properties_l1770_177069

/-- The function f(x) = 2|x-1| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x - 1) - a

/-- The function g(x) = -|x+m| -/
def g (m : ℝ) (x : ℝ) : ℝ := -(abs (x + m))

/-- The statement that g(x) > -1 has exactly one integer solution, which is -3 -/
def has_unique_integer_solution (m : ℝ) : Prop :=
  ∃! (n : ℤ), g m (n : ℝ) > -1 ∧ n = -3

theorem function_properties (a m : ℝ) 
  (h_unique : has_unique_integer_solution m) :
  m = 3 ∧ (∀ x, f a x > g m x) → a < 4 := by sorry

end NUMINAMATH_CALUDE_function_properties_l1770_177069


namespace NUMINAMATH_CALUDE_mike_seashell_count_l1770_177051

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of seashells Mike found later -/
def later_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := initial_seashells + later_seashells

theorem mike_seashell_count : total_seashells = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_mike_seashell_count_l1770_177051


namespace NUMINAMATH_CALUDE_mike_total_games_l1770_177065

/-- The number of video games Mike had initially -/
def total_games : ℕ := sorry

/-- The number of non-working games -/
def non_working_games : ℕ := 8

/-- The price of each working game in dollars -/
def price_per_game : ℕ := 7

/-- The total amount earned from selling working games in dollars -/
def total_earned : ℕ := 56

/-- Theorem stating that the total number of video games Mike had initially is 16 -/
theorem mike_total_games : total_games = 16 := by sorry

end NUMINAMATH_CALUDE_mike_total_games_l1770_177065


namespace NUMINAMATH_CALUDE_fruit_group_sizes_l1770_177072

theorem fruit_group_sizes (total_bananas total_oranges total_apples : ℕ)
                          (banana_groups orange_groups apple_groups : ℕ)
                          (h1 : total_bananas = 142)
                          (h2 : total_oranges = 356)
                          (h3 : total_apples = 245)
                          (h4 : banana_groups = 47)
                          (h5 : orange_groups = 178)
                          (h6 : apple_groups = 35) :
  ∃ (B O A : ℕ),
    banana_groups * B = total_bananas ∧
    orange_groups * O = total_oranges ∧
    apple_groups * A = total_apples ∧
    B = 3 ∧ O = 2 ∧ A = 7 := by
  sorry

end NUMINAMATH_CALUDE_fruit_group_sizes_l1770_177072


namespace NUMINAMATH_CALUDE_function_solution_l1770_177050

open Real

-- Define the function property
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x ≠ 0, f (1 / x) + (5 / x) * f x = 3 / x^3

-- State the theorem
theorem function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
    ∀ x ≠ 0, f x = 5 / (8 * x^2) - x^3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_solution_l1770_177050


namespace NUMINAMATH_CALUDE_number_of_pens_purchased_l1770_177029

/-- Given the total cost of pens and pencils, the number of pencils, and the prices of pens and pencils,
    prove that the number of pens purchased is 30. -/
theorem number_of_pens_purchased 
  (total_cost : ℝ) 
  (num_pencils : ℕ) 
  (price_pencil : ℝ) 
  (price_pen : ℝ) 
  (h1 : total_cost = 510)
  (h2 : num_pencils = 75)
  (h3 : price_pencil = 2)
  (h4 : price_pen = 12) :
  (total_cost - num_pencils * price_pencil) / price_pen = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_pens_purchased_l1770_177029


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_bound_l1770_177054

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, its 12th term is less than or equal to 7. -/
theorem arithmetic_sequence_12th_term_bound
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_8 : a 8 ≥ 15)
  (h_9 : a 9 ≤ 13) :
  a 12 ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_bound_l1770_177054


namespace NUMINAMATH_CALUDE_correct_substitution_proof_l1770_177045

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 5
def equation2 (x y : ℝ) : Prop := y = 1 + x

-- Define the correct substitution
def correct_substitution (x : ℝ) : Prop := 2 * x - 1 - x = 5

-- Theorem statement
theorem correct_substitution_proof :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → correct_substitution x :=
by sorry

end NUMINAMATH_CALUDE_correct_substitution_proof_l1770_177045


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_l1770_177021

theorem regular_triangular_pyramid (a : ℝ) : 
  (∃ h : ℝ, h = (a * Real.sqrt 3) / 3) → -- height in terms of base side
  (∃ V : ℝ, V = (1 / 3) * ((a^2 * Real.sqrt 3) / 4) * ((a * Real.sqrt 3) / 3)) → -- volume formula
  V = 18 → -- given volume
  a = 6 := by sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_l1770_177021


namespace NUMINAMATH_CALUDE_april_rainfall_calculation_l1770_177074

/-- The amount of rainfall in March, in inches -/
def march_rainfall : ℝ := 0.81

/-- The difference in rainfall between March and April, in inches -/
def rainfall_difference : ℝ := 0.35

/-- The amount of rainfall in April, in inches -/
def april_rainfall : ℝ := march_rainfall - rainfall_difference

theorem april_rainfall_calculation :
  april_rainfall = 0.46 := by sorry

end NUMINAMATH_CALUDE_april_rainfall_calculation_l1770_177074


namespace NUMINAMATH_CALUDE_square_difference_l1770_177085

theorem square_difference (x y : ℚ) 
  (sum_eq : x + y = 9/13) 
  (diff_eq : x - y = 5/13) : 
  x^2 - y^2 = 45/169 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1770_177085


namespace NUMINAMATH_CALUDE_monday_greatest_range_l1770_177040

/-- Temperature range for a day -/
def temp_range (high low : Int) : Int := high - low

/-- Temperature data for each day -/
def monday_high : Int := 6
def monday_low : Int := -4
def tuesday_high : Int := 3
def tuesday_low : Int := -6
def wednesday_high : Int := 4
def wednesday_low : Int := -2
def thursday_high : Int := 4
def thursday_low : Int := -5
def friday_high : Int := 8
def friday_low : Int := 0

/-- Theorem: Monday has the greatest temperature range -/
theorem monday_greatest_range :
  let monday_range := temp_range monday_high monday_low
  let tuesday_range := temp_range tuesday_high tuesday_low
  let wednesday_range := temp_range wednesday_high wednesday_low
  let thursday_range := temp_range thursday_high thursday_low
  let friday_range := temp_range friday_high friday_low
  (monday_range > tuesday_range) ∧
  (monday_range > wednesday_range) ∧
  (monday_range > thursday_range) ∧
  (monday_range > friday_range) :=
by sorry

end NUMINAMATH_CALUDE_monday_greatest_range_l1770_177040


namespace NUMINAMATH_CALUDE_cannot_tile_figure_l1770_177061

/-- A figure that can be colored such that each 1 × 3 strip covers exactly one colored cell. -/
structure ColoredFigure where
  colored_cells : ℕ

/-- A strip used for tiling. -/
structure Strip where
  width : ℕ
  height : ℕ

/-- Predicate to check if a figure can be tiled with given strips. -/
def CanBeTiled (f : ColoredFigure) (s : Strip) : Prop :=
  f.colored_cells % s.width = 0

theorem cannot_tile_figure (f : ColoredFigure) (s : Strip) 
  (h1 : f.colored_cells = 7)
  (h2 : s.width = 3)
  (h3 : s.height = 1) : 
  ¬CanBeTiled f s := by
  sorry

end NUMINAMATH_CALUDE_cannot_tile_figure_l1770_177061


namespace NUMINAMATH_CALUDE_dividend_divisor_problem_l1770_177062

theorem dividend_divisor_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x = 6 ∧ x + y + 6 = 216 → x = 30 ∧ y = 180 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_problem_l1770_177062


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l1770_177073

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) :
  (fibonacci (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + 1 / ((fibonacci n : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l1770_177073


namespace NUMINAMATH_CALUDE_stating_max_areas_theorem_l1770_177097

/-- Represents a circular disk divided by radii, a secant line, and a non-central chord -/
structure DividedDisk where
  n : ℕ
  radii_count : n > 0

/-- 
Calculates the maximum number of non-overlapping areas in a divided disk.
-/
def max_areas (disk : DividedDisk) : ℕ :=
  4 * disk.n + 1

/-- 
Theorem stating that the maximum number of non-overlapping areas in a divided disk
is equal to 4n + 1, where n is the number of equally spaced radii.
-/
theorem max_areas_theorem (disk : DividedDisk) :
  max_areas disk = 4 * disk.n + 1 := by sorry

end NUMINAMATH_CALUDE_stating_max_areas_theorem_l1770_177097


namespace NUMINAMATH_CALUDE_new_people_count_l1770_177011

/-- The number of people born in the country last year -/
def born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def immigrated : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := born + immigrated

/-- Theorem stating that the total number of new people is 106,491 -/
theorem new_people_count : total_new_people = 106491 := by
  sorry

end NUMINAMATH_CALUDE_new_people_count_l1770_177011


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l1770_177060

theorem max_ratio_two_digit_integers (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 → -- x and y are two-digit positive integers
  (x + y) / 2 = 65 → -- mean is 65
  x * y = 1950 → -- product is 1950
  ∀ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (a + b) / 2 = 65 ∧ a * b = 1950 →
    (a : ℚ) / b ≤ 99 / 31 :=
by sorry

#check max_ratio_two_digit_integers

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l1770_177060


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1770_177030

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * (2 ^ position)

/-- Represents the binary number 110011 -/
def binaryNumber : List Nat := [1, 1, 0, 0, 1, 1]

/-- Converts a list of binary digits to its decimal representation -/
def listBinaryToDecimal (bits : List Nat) : Nat :=
  (List.zipWith binaryToDecimal bits (List.range bits.length)).sum

theorem binary_110011_equals_51 : listBinaryToDecimal binaryNumber = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1770_177030


namespace NUMINAMATH_CALUDE_new_student_weight_l1770_177003

theorem new_student_weight (n : ℕ) (original_avg replaced_weight new_avg : ℝ) :
  n = 5 →
  replaced_weight = 72 →
  new_avg = original_avg - 12 →
  n * original_avg - replaced_weight = n * new_avg - (n * original_avg - n * new_avg) →
  n * original_avg - replaced_weight = 12 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l1770_177003


namespace NUMINAMATH_CALUDE_max_profit_appliance_business_l1770_177088

/-- Represents the cost and profit structure for small electrical appliances --/
structure ApplianceBusiness where
  cost_a : ℝ  -- Cost of one unit of type A
  cost_b : ℝ  -- Cost of one unit of type B
  profit_a : ℝ  -- Profit from selling one unit of type A
  profit_b : ℝ  -- Profit from selling one unit of type B

/-- Theorem stating the maximum profit for the given business scenario --/
theorem max_profit_appliance_business 
  (business : ApplianceBusiness)
  (h1 : 2 * business.cost_a + 3 * business.cost_b = 90)
  (h2 : 3 * business.cost_a + business.cost_b = 65)
  (h3 : business.profit_a = 3)
  (h4 : business.profit_b = 4)
  (h5 : ∀ a : ℕ, 30 ≤ a ∧ a ≤ 50 → 
    2750 ≤ a * business.cost_a + (150 - a) * business.cost_b ∧
    a * business.cost_a + (150 - a) * business.cost_b ≤ 2850)
  (h6 : ∀ a : ℕ, 30 ≤ a ∧ a ≤ 35 → 
    565 ≤ a * business.profit_a + (150 - a) * business.profit_b) :
  ∃ (max_profit : ℝ), 
    max_profit = 30 * business.profit_a + 120 * business.profit_b ∧
    max_profit = 570 ∧
    ∀ (a : ℕ), 30 ≤ a ∧ a ≤ 35 → 
      a * business.profit_a + (150 - a) * business.profit_b ≤ max_profit :=
by sorry


end NUMINAMATH_CALUDE_max_profit_appliance_business_l1770_177088


namespace NUMINAMATH_CALUDE_money_sharing_l1770_177031

theorem money_sharing (emma finn grace total : ℕ) : 
  emma = 45 →
  emma + finn + grace = total →
  3 * finn = 4 * emma →
  3 * grace = 5 * emma →
  total = 180 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l1770_177031


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1770_177024

/-- Given a geometric sequence {aₙ} with a₁ = 1/16 and a₃a₇ = 2a₅ - 1, prove that a₃ = 1/4. -/
theorem geometric_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 / 16 →
  a 3 * a 7 = 2 * a 5 - 1 →
  a 3 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1770_177024


namespace NUMINAMATH_CALUDE_correct_annual_increase_l1770_177007

/-- The annual percentage increase in Bobby's toy cars -/
def annual_increase : ℝ := 0.5

/-- The initial number of toy cars Bobby has -/
def initial_cars : ℕ := 16

/-- The number of years that pass -/
def years : ℕ := 3

/-- The final number of toy cars Bobby has -/
def final_cars : ℕ := 54

/-- Theorem stating that the annual increase is correct -/
theorem correct_annual_increase :
  (initial_cars : ℝ) * (1 + annual_increase)^years = final_cars := by
  sorry


end NUMINAMATH_CALUDE_correct_annual_increase_l1770_177007


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_17_l1770_177053

theorem modular_inverse_of_5_mod_17 : 
  ∃! x : ℕ, x ∈ Finset.range 17 ∧ (5 * x) % 17 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_17_l1770_177053


namespace NUMINAMATH_CALUDE_cos_negative_120_degrees_l1770_177091

theorem cos_negative_120_degrees : Real.cos (-(120 * Real.pi / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_120_degrees_l1770_177091


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l1770_177095

/-- If a, b, c are the sides of a triangle and satisfy the given equation,
    then the triangle is isosceles. -/
theorem isosceles_triangle_condition (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0 →
  (a = b ∨ b = c ∨ c = a) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l1770_177095


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1770_177013

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the equation x^2 - 2x - 3 = 0 -/
def roots_equation (x y : ℝ) : Prop :=
  x^2 - 2*x - 3 = 0 ∧ y^2 - 2*y - 3 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  roots_equation (a 1) (a 4) →
  a 2 * a 3 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1770_177013


namespace NUMINAMATH_CALUDE_sum_of_digits_properties_l1770_177071

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_properties :
  (∀ n : ℕ, sum_of_digits (2 * n) ≤ 2 * sum_of_digits n) ∧
  (∀ n : ℕ, 2 * sum_of_digits n ≤ 10 * sum_of_digits (2 * n)) ∧
  (∃ k : ℕ, sum_of_digits k = 1996 * sum_of_digits (3 * k)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_properties_l1770_177071


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1770_177042

/-- Given an article sold at $1800 with a 20% profit, prove that the cost price is $1500. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 1800)
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1500 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1770_177042


namespace NUMINAMATH_CALUDE_jeromes_contact_list_l1770_177010

theorem jeromes_contact_list (classmates : ℕ) (out_of_school_friends : ℕ) (family_members : ℕ) : 
  classmates = 20 →
  out_of_school_friends = classmates / 2 →
  family_members = 3 →
  classmates + out_of_school_friends + family_members = 33 := by
  sorry

end NUMINAMATH_CALUDE_jeromes_contact_list_l1770_177010


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1770_177075

theorem rectangle_dimensions (length width : ℝ) : 
  length > 0 → width > 0 → 
  length * width = 120 → 
  2 * (length + width) = 46 → 
  min length width = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1770_177075


namespace NUMINAMATH_CALUDE_compute_expression_l1770_177022

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1770_177022


namespace NUMINAMATH_CALUDE_lipschitz_periodic_bound_l1770_177076

/-- A function f is k-Lipschitz if |f(x) - f(y)| ≤ k|x - y| for all x, y in the domain -/
def is_k_lipschitz (f : ℝ → ℝ) (k : ℝ) :=
  ∀ x y, |f x - f y| ≤ k * |x - y|

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def is_periodic (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

theorem lipschitz_periodic_bound
  (f : ℝ → ℝ)
  (h_lipschitz : is_k_lipschitz f 1)
  (h_periodic : is_periodic f 2) :
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_lipschitz_periodic_bound_l1770_177076


namespace NUMINAMATH_CALUDE_binary_repr_25_l1770_177084

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

theorem binary_repr_25 : binary_repr 25 = [true, false, false, true, true] := by sorry

end NUMINAMATH_CALUDE_binary_repr_25_l1770_177084


namespace NUMINAMATH_CALUDE_triangle_problem_l1770_177096

noncomputable def Triangle (a b c : ℝ) (A B C : ℝ) := True

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h_c : c = Real.sqrt 7)
  (h_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :
  C = π/3 ∧ a + b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1770_177096


namespace NUMINAMATH_CALUDE_village_population_l1770_177041

/-- The population change over two years -/
def population_change (initial : ℝ) : ℝ := initial * 1.3 * 0.7

/-- The problem statement -/
theorem village_population : 
  ∃ (initial : ℝ), 
    population_change initial = 13650 ∧ 
    initial = 15000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1770_177041


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1770_177016

-- Part 1
theorem problem_1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (π / 3) = 1 - Real.sqrt 3 := by sorry

-- Part 2
theorem problem_2 (a b : ℝ) (h : a ≠ b) : (a - b) / (a + b) / (b - a) = -1 / (a + b) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1770_177016


namespace NUMINAMATH_CALUDE_parking_lot_capacity_l1770_177027

/-- Calculates the number of vehicles that can still park in a lot -/
def remainingParkingSpaces (totalSpaces : ℕ) (caravanSpaces : ℕ) (caravansParked : ℕ) : ℕ :=
  totalSpaces - (caravanSpaces * caravansParked)

/-- Theorem: Given the conditions, 24 vehicles can still park -/
theorem parking_lot_capacity : remainingParkingSpaces 30 2 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_capacity_l1770_177027


namespace NUMINAMATH_CALUDE_four_propositions_l1770_177047

-- Define the propositions
def opposite_numbers (x y : ℝ) : Prop := x = -y

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of congruent triangles

def equal_areas (t1 t2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of equal areas for triangles

def right_triangle (t : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of right triangle

def has_two_acute_angles (t : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of triangle with two acute angles

-- Theorem to prove
theorem four_propositions :
  (∀ x y : ℝ, opposite_numbers x y → x + y = 0) ∧
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) ∧
  ¬(∀ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) → ¬(equal_areas t1 t2)) ∧
  ¬(∀ t : Set ℝ × Set ℝ, has_two_acute_angles t → right_triangle t) :=
by
  sorry

end NUMINAMATH_CALUDE_four_propositions_l1770_177047


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l1770_177052

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 → 
  Odd n → 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) → 
  x = 16808 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l1770_177052


namespace NUMINAMATH_CALUDE_line_points_k_value_l1770_177009

/-- 
Given two points (m, n) and (m + 5, n + k) on a line with equation x = 2y + 5,
prove that k = 2.5
-/
theorem line_points_k_value 
  (m n k : ℝ) 
  (point1_on_line : m = 2 * n + 5)
  (point2_on_line : m + 5 = 2 * (n + k) + 5) :
  k = 2.5 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1770_177009


namespace NUMINAMATH_CALUDE_number_square_problem_l1770_177014

theorem number_square_problem : ∃! x : ℝ, x^2 + 64 = (x - 16)^2 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_number_square_problem_l1770_177014


namespace NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l1770_177019

/-- A fifteen-digit integer formed by repeating a five-digit integer three times -/
def repeatedNumber (n : ℕ) : ℕ := n * 10000100001

/-- The set of all such fifteen-digit numbers -/
def S : Set ℕ := {m : ℕ | ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ m = repeatedNumber n}

theorem gcd_of_repeated_numbers : 
  ∃ d : ℕ, d > 0 ∧ ∀ m ∈ S, d ∣ m ∧ ∀ k : ℕ, (∀ m ∈ S, k ∣ m) → k ∣ d :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l1770_177019


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l1770_177089

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 212 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a = 98 ∧
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l1770_177089


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l1770_177087

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 6 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 18 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l1770_177087


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l1770_177058

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l1770_177058


namespace NUMINAMATH_CALUDE_lower_bound_second_inequality_l1770_177098

theorem lower_bound_second_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : ∃ n, n < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  ∀ n, n < x → n ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_lower_bound_second_inequality_l1770_177098


namespace NUMINAMATH_CALUDE_power_division_equals_27_l1770_177046

theorem power_division_equals_27 : 3^12 / 27^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_27_l1770_177046


namespace NUMINAMATH_CALUDE_science_project_cans_l1770_177001

def empty_cans_problem (alyssa_cans abigail_cans more_needed : ℕ) : Prop :=
  alyssa_cans + abigail_cans + more_needed = 100

theorem science_project_cans : empty_cans_problem 30 43 27 := by
  sorry

end NUMINAMATH_CALUDE_science_project_cans_l1770_177001


namespace NUMINAMATH_CALUDE_distributive_property_fraction_l1770_177036

theorem distributive_property_fraction (x y : ℝ) :
  (x + y) / 2 = x / 2 + y / 2 := by sorry

end NUMINAMATH_CALUDE_distributive_property_fraction_l1770_177036


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1770_177079

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 5 = 0) → 
  (b^3 - 2*b^2 + 3*b - 5 = 0) → 
  (c^3 - 2*c^2 + 3*c - 5 = 0) → 
  a^3 + b^3 + c^3 = 5 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1770_177079


namespace NUMINAMATH_CALUDE_profit_division_l1770_177055

theorem profit_division (profit_x profit_y total_profit : ℚ) : 
  profit_x / profit_y = 1/2 / (1/3) →
  profit_x - profit_y = 100 →
  profit_x + profit_y = total_profit →
  total_profit = 500 := by
sorry

end NUMINAMATH_CALUDE_profit_division_l1770_177055


namespace NUMINAMATH_CALUDE_fraction_order_l1770_177012

theorem fraction_order : 
  let f₁ : ℚ := 16 / 12
  let f₂ : ℚ := 20 / 16
  let f₃ : ℚ := 18 / 14
  let f₄ : ℚ := 22 / 17
  f₂ < f₃ ∧ f₃ < f₄ ∧ f₄ < f₁ :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1770_177012


namespace NUMINAMATH_CALUDE_second_month_sale_proof_l1770_177093

/-- Calculates the sale in the second month given the sales of other months and the required total sales. -/
def second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (total_sales : ℕ) : ℕ :=
  total_sales - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Proves that the sale in the second month is 11690 given the specific sales figures. -/
theorem second_month_sale_proof :
  second_month_sale 5266 5678 6029 5922 4937 33600 = 11690 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_proof_l1770_177093


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1770_177070

theorem fraction_sum_equality : 
  (1/4 - 1/5) / (2/5 - 1/4) + (1/6) / (1/3 - 1/4) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1770_177070


namespace NUMINAMATH_CALUDE_inequality_proof_l1770_177059

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a + b > |c - d|) 
  (h2 : c + d > |a - b|) : 
  a + c > |b - d| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1770_177059


namespace NUMINAMATH_CALUDE_expression_equals_73_l1770_177056

def x : ℤ := 2
def y : ℤ := -3
def z : ℤ := 6

theorem expression_equals_73 : x^2 + y^2 + z^2 + 2*x*y - 2*y*z = 73 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_73_l1770_177056


namespace NUMINAMATH_CALUDE_certain_number_problem_l1770_177099

theorem certain_number_problem : ∃ x : ℝ, 45 * x = 0.45 * 900 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1770_177099


namespace NUMINAMATH_CALUDE_two_p_plus_q_value_l1770_177063

theorem two_p_plus_q_value (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_value_l1770_177063


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1770_177028

/-- Given a rectangle with perimeter 160 feet and integer side lengths, 
    the maximum possible area is 1600 square feet. -/
theorem max_area_rectangle (l w : ℕ) : 
  2 * (l + w) = 160 → l * w ≤ 1600 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1770_177028


namespace NUMINAMATH_CALUDE_composition_ratio_l1770_177033

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 3 * x - 2

theorem composition_ratio : (f (g (f 2))) / (g (f (g 2))) = 41 / 31 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l1770_177033


namespace NUMINAMATH_CALUDE_adults_trekking_l1770_177083

/-- The number of adults who went for trekking -/
def numAdults : ℕ := 56

/-- The number of children who went for trekking -/
def numChildren : ℕ := 70

/-- The number of adults the meal can feed -/
def mealAdults : ℕ := 70

/-- The number of children the meal can feed -/
def mealChildren : ℕ := 90

/-- The number of adults who have already eaten -/
def adultsEaten : ℕ := 14

/-- The number of children that can be fed with remaining food after some adults eat -/
def remainingChildren : ℕ := 72

theorem adults_trekking :
  numAdults = mealAdults - adultsEaten ∧
  numChildren = 70 ∧
  mealAdults = 70 ∧
  mealChildren = 90 ∧
  adultsEaten = 14 ∧
  remainingChildren = 72 ∧
  mealChildren = remainingChildren + adultsEaten * mealChildren / mealAdults :=
by sorry

end NUMINAMATH_CALUDE_adults_trekking_l1770_177083


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1770_177067

theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 10 * Real.sqrt 3 → s^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1770_177067


namespace NUMINAMATH_CALUDE_existence_of_incommensurable_segments_l1770_177008

-- Define incommensurability
def incommensurable (x y : ℝ) : Prop :=
  ∀ k : ℚ, k ≠ 0 → x ≠ k * y

-- State the theorem
theorem existence_of_incommensurable_segments :
  ∃ (a b c d : ℝ),
    a + b + c = d ∧
    incommensurable a d ∧
    incommensurable b d ∧
    incommensurable c d :=
by sorry

end NUMINAMATH_CALUDE_existence_of_incommensurable_segments_l1770_177008


namespace NUMINAMATH_CALUDE_ratio_problem_l1770_177039

theorem ratio_problem (x y : ℚ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : 
  x / y = -13 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1770_177039


namespace NUMINAMATH_CALUDE_min_double_rooms_part1_min_triple_rooms_part2_l1770_177032

/-- Represents a hotel room configuration --/
structure RoomConfig where
  double_rooms : ℕ
  triple_rooms : ℕ

/-- Calculates the total number of students that can be accommodated --/
def total_students (config : RoomConfig) : ℕ :=
  2 * config.double_rooms + 3 * config.triple_rooms

/-- Calculates the total cost of the room configuration --/
def total_cost (config : RoomConfig) (double_price : ℕ) (triple_price : ℕ) : ℕ :=
  config.double_rooms * double_price + config.triple_rooms * triple_price

/-- Theorem for part (1) --/
theorem min_double_rooms_part1 (male_students female_students : ℕ) 
  (h_total : male_students + female_students = 50)
  (h_male : male_students = 27)
  (h_female : female_students = 23)
  (double_price triple_price : ℕ)
  (h_double_price : double_price = 200)
  (h_triple_price : triple_price = 250) :
  ∃ (config : RoomConfig), 
    total_students config ≥ 50 ∧ 
    config.double_rooms = 1 ∧
    (∀ (other_config : RoomConfig), 
      total_students other_config ≥ 50 → 
      total_cost config double_price triple_price ≤ total_cost other_config double_price triple_price) :=
sorry

/-- Theorem for part (2) --/
theorem min_triple_rooms_part2 (male_students female_students : ℕ) 
  (h_total : male_students + female_students = 50)
  (h_male : male_students = 27)
  (h_female : female_students = 23)
  (double_price triple_price : ℕ)
  (h_double_price : double_price = 160)  -- 20% discount applied
  (h_triple_price : triple_price = 250)
  (max_double_rooms : ℕ)
  (h_max_double : max_double_rooms = 15) :
  ∃ (config : RoomConfig), 
    total_students config ≥ 50 ∧ 
    config.double_rooms ≤ max_double_rooms ∧
    config.triple_rooms = 8 ∧
    (∀ (other_config : RoomConfig), 
      total_students other_config ≥ 50 ∧ 
      other_config.double_rooms ≤ max_double_rooms → 
      total_cost config double_price triple_price ≤ total_cost other_config double_price triple_price) :=
sorry

end NUMINAMATH_CALUDE_min_double_rooms_part1_min_triple_rooms_part2_l1770_177032


namespace NUMINAMATH_CALUDE_line_curve_intersection_l1770_177044

-- Define the line
def line (x : ℝ) : ℝ := x + 3

-- Define the curve
def curve (x y : ℝ) : Prop := y^2 / 9 - x * abs x / 4 = 1

-- State the theorem
theorem line_curve_intersection :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = 3 ∧ 
    (∀ p ∈ points, curve p.1 p.2 ∧ p.2 = line p.1) ∧
    (∀ x y, curve x y ∧ y = line x → (x, y) ∈ points) :=
sorry

end NUMINAMATH_CALUDE_line_curve_intersection_l1770_177044


namespace NUMINAMATH_CALUDE_smallest_integer_l1770_177004

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 28) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 28 → b ≤ c → b = 105 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l1770_177004


namespace NUMINAMATH_CALUDE_investment_amount_from_interest_difference_l1770_177002

/-- Proves that given two equal investments with specific interest rates and time period, 
    the investment amount can be determined from the interest difference. -/
theorem investment_amount_from_interest_difference 
  (P : ℝ) -- The amount invested (same for both investments)
  (r1 : ℝ) -- Interest rate for first investment
  (r2 : ℝ) -- Interest rate for second investment
  (t : ℝ) -- Time period in years
  (diff : ℝ) -- Difference in interest earned
  (h1 : r1 = 0.04) -- First interest rate is 4%
  (h2 : r2 = 0.045) -- Second interest rate is 4.5%
  (h3 : t = 7) -- Time period is 7 years
  (h4 : P * r2 * t - P * r1 * t = diff) -- Difference in interest equation
  (h5 : diff = 31.5) -- Interest difference is $31.50
  : P = 900 := by
  sorry

#check investment_amount_from_interest_difference

end NUMINAMATH_CALUDE_investment_amount_from_interest_difference_l1770_177002


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1770_177023

theorem rope_cutting_problem (a b c : ℕ) (ha : a = 39) (hb : b = 52) (hc : c = 65) :
  Nat.gcd a (Nat.gcd b c) = 13 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1770_177023


namespace NUMINAMATH_CALUDE_ae_length_l1770_177000

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- The property that AD = BC -/
  ad_eq_bc : True
  /-- Point E such that BC = EC -/
  bc_eq_ec : True
  /-- AE is perpendicular to EC -/
  ae_perp_ec : True

/-- The main theorem stating the length of AE in the specific isosceles trapezoid -/
theorem ae_length (t : IsoscelesTrapezoid) (h1 : t.ab = 3) (h2 : t.cd = 8) : 
  ∃ ae : ℝ, ae = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ae_length_l1770_177000


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1770_177066

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ 3 * x^2 - x + m = 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1770_177066


namespace NUMINAMATH_CALUDE_root_of_polynomial_l1770_177081

theorem root_of_polynomial (b : ℝ) (h : b^5 = 2 - Real.sqrt 3) :
  (b + (2 + Real.sqrt 3)^(1/5 : ℝ))^5 - 5*(b + (2 + Real.sqrt 3)^(1/5 : ℝ))^3 + 
  5*(b + (2 + Real.sqrt 3)^(1/5 : ℝ)) - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l1770_177081
