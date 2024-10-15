import Mathlib

namespace NUMINAMATH_CALUDE_find_x_l3680_368056

theorem find_x : ∃ x : ℚ, x * 9999 = 724827405 ∧ x = 72492.75 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3680_368056


namespace NUMINAMATH_CALUDE_couponA_greatest_discount_specific_prices_l3680_368092

/-- Represents the discount amount for Coupon A -/
def couponA (p : ℝ) : ℝ := 0.15 * p

/-- Represents the discount amount for Coupon B -/
def couponB : ℝ := 30

/-- Represents the discount amount for Coupon C -/
def couponC (p : ℝ) : ℝ := 0.25 * (p - 150)

/-- Theorem stating when Coupon A offers the greatest discount -/
theorem couponA_greatest_discount (p : ℝ) :
  (couponA p > couponB ∧ couponA p > couponC p) ↔ (200 < p ∧ p < 375) :=
sorry

/-- Function to check if a price satisfies the condition for Coupon A being the best -/
def is_couponA_best (p : ℝ) : Prop := 200 < p ∧ p < 375

/-- Theorem for the specific price points given in the problem -/
theorem specific_prices :
  is_couponA_best 209.95 ∧
  is_couponA_best 229.95 ∧
  is_couponA_best 249.95 ∧
  ¬is_couponA_best 169.95 ∧
  ¬is_couponA_best 189.95 :=
sorry

end NUMINAMATH_CALUDE_couponA_greatest_discount_specific_prices_l3680_368092


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l3680_368058

theorem percentage_increase_decrease (x : ℝ) (h : x > 0) :
  x * (1 + 0.25) * (1 - 0.20) = x := by
  sorry

#check percentage_increase_decrease

end NUMINAMATH_CALUDE_percentage_increase_decrease_l3680_368058


namespace NUMINAMATH_CALUDE_max_value_expression_l3680_368001

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_eq : c^2 = a^2 + b^2) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) ≤ 2 * a^2 + b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) = 2 * a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3680_368001


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3680_368063

-- Define the inequality function
def f (x : ℝ) := (x - 2) * (x + 1)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 0} = Set.Ioo (-1 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3680_368063


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3680_368024

theorem arithmetic_expression_equality : 
  8.1 * 1.3 + 8 / 1.3 + 1.9 * 1.3 - 11.9 / 1.3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3680_368024


namespace NUMINAMATH_CALUDE_money_distribution_l3680_368037

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 340) :
  c = 40 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3680_368037


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3680_368052

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 3) :
  2^a + 2^b ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3680_368052


namespace NUMINAMATH_CALUDE_tangent_line_constant_l3680_368035

/-- The value of m for which y = -x + m is tangent to y = x^2 - 3ln(x) -/
theorem tangent_line_constant (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 
    x^2 - 3 * Real.log x = -x + m ∧ 
    2 * x - 3 / x = -1) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_constant_l3680_368035


namespace NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l3680_368007

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series_equals_three_fourths :
  (∑' k : ℕ, (k : ℝ) / 3^k) = 3/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l3680_368007


namespace NUMINAMATH_CALUDE_complement_of_union_l3680_368016

def U : Finset ℕ := {1, 3, 5, 9}
def A : Finset ℕ := {1, 3, 9}
def B : Finset ℕ := {1, 9}

theorem complement_of_union :
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3680_368016


namespace NUMINAMATH_CALUDE_bridget_profit_is_40_l3680_368064

/-- Calculates Bridget's profit from baking and selling bread --/
def bridget_profit (
  total_loaves : ℕ)
  (morning_price afternoon_price late_afternoon_price : ℚ)
  (production_cost fixed_cost : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let afternoon_sales := (total_loaves - morning_sales) / 2
  let late_afternoon_sales := total_loaves - morning_sales - afternoon_sales
  let total_revenue := 
    morning_sales * morning_price + 
    afternoon_sales * afternoon_price + 
    late_afternoon_sales * late_afternoon_price
  let total_cost := total_loaves * production_cost + fixed_cost
  total_revenue - total_cost

/-- Theorem stating that Bridget's profit is $40 given the problem conditions --/
theorem bridget_profit_is_40 :
  bridget_profit 60 3 (3/2) 1 1 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bridget_profit_is_40_l3680_368064


namespace NUMINAMATH_CALUDE_basketball_game_total_points_l3680_368034

theorem basketball_game_total_points : 
  ∀ (adam_2pt adam_3pt mada_2pt mada_3pt : ℕ),
    adam_2pt + adam_3pt = 10 →
    mada_2pt + mada_3pt = 11 →
    adam_2pt = mada_3pt →
    2 * adam_2pt + 3 * adam_3pt = 3 * mada_3pt + 2 * mada_2pt →
    2 * adam_2pt + 3 * adam_3pt + 3 * mada_3pt + 2 * mada_2pt = 52 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_total_points_l3680_368034


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l3680_368033

/-- A line tangent to the unit circle with intercepts summing to √3 forms a triangle with area 3/2 --/
theorem tangent_line_triangle_area :
  ∀ (a b : ℝ),
  (a > 0 ∧ b > 0) →  -- Positive intercepts
  (a + b = Real.sqrt 3) →  -- Sum of intercepts
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*y + b*x = a*b) →  -- Tangent to unit circle
  (1/2 * a * b = 3/2) :=  -- Area of triangle
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l3680_368033


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l3680_368014

-- Define the repeating decimal 0.6̄
def repeating_decimal : ℚ := 2/3

-- State the theorem
theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l3680_368014


namespace NUMINAMATH_CALUDE_fiftieth_day_of_previous_year_l3680_368030

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℤ
  isLeapYear : Bool

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (dayNumber : ℕ) : DayOfWeek :=
  sorry

/-- Determines if two days of the week are equal -/
def dayOfWeekEqual (d1 d2 : DayOfWeek) : Prop :=
  sorry

theorem fiftieth_day_of_previous_year
  (N : Year)
  (h1 : N.isLeapYear = true)
  (h2 : dayOfWeekEqual (dayOfWeek N 250) DayOfWeek.Monday = true)
  (h3 : dayOfWeekEqual (dayOfWeek (Year.mk (N.number + 1) false) 150) DayOfWeek.Tuesday = true) :
  dayOfWeekEqual (dayOfWeek (Year.mk (N.number - 1) false) 50) DayOfWeek.Wednesday = true :=
sorry

end NUMINAMATH_CALUDE_fiftieth_day_of_previous_year_l3680_368030


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l3680_368036

theorem alcohol_water_ratio (alcohol_fraction water_fraction : ℚ) 
  (h1 : alcohol_fraction = 3/5)
  (h2 : water_fraction = 2/5)
  (h3 : alcohol_fraction + water_fraction = 1) :
  alcohol_fraction / water_fraction = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l3680_368036


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l3680_368097

theorem sum_of_two_smallest_numbers (a b c d : ℝ) : 
  a / b = 3 / 5 ∧ 
  b / c = 5 / 7 ∧ 
  c / d = 7 / 9 ∧ 
  (a + b + c + d) / 4 = 30 →
  a + b = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l3680_368097


namespace NUMINAMATH_CALUDE_cone_slant_height_l3680_368060

theorem cone_slant_height (V : ℝ) (θ : ℝ) (l : ℝ) : 
  V = 9 * Real.pi → θ = Real.pi / 4 → l = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3680_368060


namespace NUMINAMATH_CALUDE_butterfly_collection_l3680_368008

/-- Given a collection of butterflies with specific conditions, prove the number of black butterflies. -/
theorem butterfly_collection (total : ℕ) (blue : ℕ) (yellow : ℕ) (black : ℕ)
  (h_total : total = 19)
  (h_blue : blue = 6)
  (h_yellow_ratio : yellow * 2 = blue)
  (h_sum : total = blue + yellow + black) :
  black = 10 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l3680_368008


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3680_368094

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3680_368094


namespace NUMINAMATH_CALUDE_greatest_power_under_600_l3680_368009

theorem greatest_power_under_600 (a b : ℕ) : 
  a > 0 → b > 1 → a^b < 600 → 
  (∀ c d : ℕ, c > 0 → d > 1 → c^d < 600 → c^d ≤ a^b) →
  a + b = 26 := by
sorry

end NUMINAMATH_CALUDE_greatest_power_under_600_l3680_368009


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_equation_l3680_368005

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_through_point 
  (P : Point2D)
  (given_line : Line2D)
  (result_line : Line2D) : Prop :=
  P.x = -1 ∧ 
  P.y = 3 ∧ 
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  result_line.a = 2 ∧ 
  result_line.b = 1 ∧ 
  result_line.c = -1 ∧
  perpendicular given_line result_line ∧
  passes_through result_line P ∧
  ∀ (other_line : Line2D), 
    perpendicular given_line other_line ∧ 
    passes_through other_line P → 
    other_line = result_line

theorem perpendicular_line_equation : 
  perpendicular_line_through_point 
    (Point2D.mk (-1) 3) 
    (Line2D.mk 1 (-2) 3) 
    (Line2D.mk 2 1 (-1)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_equation_l3680_368005


namespace NUMINAMATH_CALUDE_square_sum_and_product_l3680_368010

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l3680_368010


namespace NUMINAMATH_CALUDE_hexagon_percentage_l3680_368006

-- Define the tiling structure
structure Tiling where
  smallSquareArea : ℝ
  largeSquareArea : ℝ
  hexagonArea : ℝ
  smallSquaresPerLarge : ℕ
  hexagonsPerLarge : ℕ
  smallSquaresInHexagons : ℝ

-- Define the tiling conditions
def tilingConditions (t : Tiling) : Prop :=
  t.smallSquaresPerLarge = 16 ∧
  t.hexagonsPerLarge = 3 ∧
  t.largeSquareArea = 16 * t.smallSquareArea ∧
  t.hexagonArea = 2 * t.smallSquareArea ∧
  t.smallSquaresInHexagons = 3 * t.hexagonArea

-- Theorem to prove
theorem hexagon_percentage (t : Tiling) (h : tilingConditions t) :
  (t.smallSquaresInHexagons / t.largeSquareArea) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_percentage_l3680_368006


namespace NUMINAMATH_CALUDE_factor_expression_l3680_368067

theorem factor_expression (x y a b : ℝ) :
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3680_368067


namespace NUMINAMATH_CALUDE_smallest_n_proof_l3680_368012

/-- The number of boxes -/
def num_boxes : ℕ := 2010

/-- The probability of stopping after drawing exactly n marbles -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

/-- The smallest n for which P(n) < 1/2010 -/
def smallest_n : ℕ := 45

theorem smallest_n_proof :
  (∀ k < smallest_n, P k ≥ threshold) ∧
  P smallest_n < threshold :=
sorry

#check smallest_n_proof

end NUMINAMATH_CALUDE_smallest_n_proof_l3680_368012


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3680_368049

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3680_368049


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3680_368066

theorem polynomial_simplification :
  ∀ x : ℝ, 3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3680_368066


namespace NUMINAMATH_CALUDE_blue_face_cubes_count_l3680_368072

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with more than one blue face in a painted block -/
def count_multi_blue_face_cubes (b : Block) : Nat :=
  sorry

/-- The main theorem stating that a 5x3x1 block has 10 cubes with more than one blue face -/
theorem blue_face_cubes_count :
  let block := Block.mk 5 3 1
  count_multi_blue_face_cubes block = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_cubes_count_l3680_368072


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3680_368038

/-- The fixed point of the function f(x) = 2a^(x+1) - 3, where a > 0 and a ≠ 1, is (-1, -1). -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 ∧ ∀ x : ℝ, f x = x → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3680_368038


namespace NUMINAMATH_CALUDE_range_of_m_l3680_368078

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x + 1

-- Define the condition for two real roots
def has_two_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  has_two_real_roots m ↔ m ≤ 2 ∧ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3680_368078


namespace NUMINAMATH_CALUDE_sqrt_form_existence_l3680_368099

def has_sqrt_form (a : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x^2 + 2*y^2 = a) ∧ (2*x*y = 12)

theorem sqrt_form_existence :
  has_sqrt_form 17 ∧
  has_sqrt_form 22 ∧
  has_sqrt_form 38 ∧
  has_sqrt_form 73 ∧
  ¬(has_sqrt_form 54) :=
sorry

end NUMINAMATH_CALUDE_sqrt_form_existence_l3680_368099


namespace NUMINAMATH_CALUDE_solution1_solution2_a_solution2_b_l3680_368026

-- Part 1
def equation1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y + 13 = 0

theorem solution1 : equation1 2 (-3) := by sorry

-- Part 2
def equation2 (x y : ℝ) : Prop :=
  x*y - 1 = x - y

theorem solution2_a (y : ℝ) : equation2 1 y := by sorry

theorem solution2_b (x : ℝ) (h : x ≠ 1) : equation2 x 1 := by sorry

end NUMINAMATH_CALUDE_solution1_solution2_a_solution2_b_l3680_368026


namespace NUMINAMATH_CALUDE_base_prime_repr_294_l3680_368046

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of natural numbers is a valid base prime representation -/
def is_valid_base_prime_repr (repr : List ℕ) : Prop :=
  sorry

theorem base_prime_repr_294 :
  let repr := base_prime_repr 294
  is_valid_base_prime_repr repr ∧ repr = [2, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_prime_repr_294_l3680_368046


namespace NUMINAMATH_CALUDE_certain_number_proof_l3680_368039

/-- The certain number that, when multiplied by the smallest positive integer a
    that makes the product a square, equals 14 -/
def certain_number : ℕ := 14

theorem certain_number_proof (a : ℕ) (h1 : a = 14) 
  (h2 : ∀ k : ℕ, k < a → ¬∃ m : ℕ, k * certain_number = m * m) 
  (h3 : ∃ m : ℕ, a * certain_number = m * m) : 
  certain_number = 14 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3680_368039


namespace NUMINAMATH_CALUDE_price_conditions_max_basketballs_l3680_368015

/-- Represents the price of a basketball -/
def basketball_price : ℕ := 80

/-- Represents the price of a soccer ball -/
def soccer_price : ℕ := 50

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 60

/-- The maximum allowed total cost -/
def max_cost : ℕ := 4000

/-- Verifies that the prices satisfy the given conditions -/
theorem price_conditions : 
  2 * basketball_price + 3 * soccer_price = 310 ∧
  5 * basketball_price + 2 * soccer_price = 500 := by sorry

/-- Proves that the maximum number of basketballs that can be purchased is 33 -/
theorem max_basketballs :
  ∀ m : ℕ, 
    m ≤ total_balls ∧ 
    m * basketball_price + (total_balls - m) * soccer_price ≤ max_cost →
    m ≤ 33 := by sorry

end NUMINAMATH_CALUDE_price_conditions_max_basketballs_l3680_368015


namespace NUMINAMATH_CALUDE_james_berets_l3680_368061

/-- The number of spools required to make one beret -/
def spools_per_beret : ℕ := 3

/-- The number of red yarn spools James has -/
def red_spools : ℕ := 12

/-- The number of black yarn spools James has -/
def black_spools : ℕ := 15

/-- The number of blue yarn spools James has -/
def blue_spools : ℕ := 6

/-- The total number of spools James has -/
def total_spools : ℕ := red_spools + black_spools + blue_spools

/-- The number of berets James can make -/
def berets_made : ℕ := total_spools / spools_per_beret

theorem james_berets :
  berets_made = 11 := by sorry

end NUMINAMATH_CALUDE_james_berets_l3680_368061


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3680_368003

theorem scientific_notation_equality : 0.0000012 = 1.2 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3680_368003


namespace NUMINAMATH_CALUDE_equation_solution_l3680_368031

theorem equation_solution (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) :
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3680_368031


namespace NUMINAMATH_CALUDE_pyramid_volume_l3680_368081

/-- The volume of a square-based pyramid with given dimensions -/
theorem pyramid_volume (base_side : ℝ) (apex_distance : ℝ) (volume : ℝ) : 
  base_side = 24 →
  apex_distance = Real.sqrt 364 →
  volume = (1 / 3) * base_side^2 * Real.sqrt ((apex_distance^2) - (1/2 * base_side * Real.sqrt 2)^2) →
  volume = 384 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3680_368081


namespace NUMINAMATH_CALUDE_back_seat_capacity_l3680_368065

def bus_capacity := 88
def left_side_seats := 15
def seat_capacity := 3

theorem back_seat_capacity : 
  ∀ (right_side_seats : ℕ) (back_seat_capacity : ℕ),
    right_side_seats = left_side_seats - 3 →
    bus_capacity = left_side_seats * seat_capacity + right_side_seats * seat_capacity + back_seat_capacity →
    back_seat_capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_back_seat_capacity_l3680_368065


namespace NUMINAMATH_CALUDE_slant_base_angle_is_36_degrees_l3680_368076

/-- A regular pentagonal pyramid where the slant height is equal to the base edge -/
structure RegularPentagonalPyramid where
  /-- The base of the pyramid is a regular pentagon -/
  base : RegularPentagon
  /-- The slant height of the pyramid -/
  slant_height : ℝ
  /-- The base edge of the pyramid -/
  base_edge : ℝ
  /-- The slant height is equal to the base edge -/
  slant_height_eq_base_edge : slant_height = base_edge

/-- The angle between a slant height and a non-intersecting, non-perpendicular base edge -/
def slant_base_angle (p : RegularPentagonalPyramid) : Angle := sorry

/-- Theorem: The angle between a slant height and a non-intersecting, non-perpendicular base edge is 36° -/
theorem slant_base_angle_is_36_degrees (p : RegularPentagonalPyramid) :
  slant_base_angle p = 36 * π / 180 := by sorry

end NUMINAMATH_CALUDE_slant_base_angle_is_36_degrees_l3680_368076


namespace NUMINAMATH_CALUDE_smallest_n_with_perfect_square_sum_l3680_368017

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def partition_with_perfect_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ),
    (A ∪ B = Finset.range n) →
    (A ∩ B = ∅) →
    (A ≠ ∅) →
    (B ≠ ∅) →
    (∃ (x y : ℕ), (x ∈ A ∧ y ∈ A ∧ is_perfect_square (x + y)) ∨
                  (x ∈ B ∧ y ∈ B ∧ is_perfect_square (x + y)))

theorem smallest_n_with_perfect_square_sum : 
  (∀ k < 15, ¬ partition_with_perfect_square_sum k) ∧ 
  partition_with_perfect_square_sum 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_perfect_square_sum_l3680_368017


namespace NUMINAMATH_CALUDE_change_calculation_l3680_368087

def egg_cost : ℕ := 3
def pancake_cost : ℕ := 2
def cocoa_cost : ℕ := 2
def tax : ℕ := 1
def initial_order_cost : ℕ := egg_cost + pancake_cost + 2 * cocoa_cost + tax
def additional_order_cost : ℕ := pancake_cost + cocoa_cost
def total_paid : ℕ := 15

theorem change_calculation :
  total_paid - (initial_order_cost + additional_order_cost) = 1 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l3680_368087


namespace NUMINAMATH_CALUDE_two_intersection_points_l3680_368019

/-- Represents a line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * X + b * Y = c

/-- Represents a point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- The three lines given in the problem --/
def line1 : Line := ⟨4, -3, 2, by sorry⟩
def line2 : Line := ⟨1, 3, 3, by sorry⟩
def line3 : Line := ⟨3, -4, 3, by sorry⟩

/-- Theorem stating that there are exactly two distinct intersection points --/
theorem two_intersection_points : 
  ∃! (p1 p2 : Point), 
    (pointOnLine p1 line1 ∧ pointOnLine p1 line2) ∨ 
    (pointOnLine p1 line1 ∧ pointOnLine p1 line3) ∨ 
    (pointOnLine p1 line2 ∧ pointOnLine p1 line3) ∧
    (pointOnLine p2 line1 ∧ pointOnLine p2 line2) ∨ 
    (pointOnLine p2 line1 ∧ pointOnLine p2 line3) ∨ 
    (pointOnLine p2 line2 ∧ pointOnLine p2 line3) ∧
    p1 ≠ p2 :=
  sorry

end NUMINAMATH_CALUDE_two_intersection_points_l3680_368019


namespace NUMINAMATH_CALUDE_carter_has_152_cards_l3680_368013

/-- The number of baseball cards Marcus has -/
def marcus_cards : ℕ := 210

/-- The difference between Marcus's and Carter's cards -/
def marcus_carter_diff : ℕ := 58

/-- The difference between Carter's and Jenny's cards -/
def carter_jenny_diff : ℕ := 35

/-- Carter's number of baseball cards -/
def carter_cards : ℕ := marcus_cards - marcus_carter_diff

theorem carter_has_152_cards : carter_cards = 152 := by
  sorry

end NUMINAMATH_CALUDE_carter_has_152_cards_l3680_368013


namespace NUMINAMATH_CALUDE_rational_equation_solution_no_solution_rational_equation_l3680_368059

-- Problem 1
theorem rational_equation_solution (x : ℝ) :
  x ≠ 2 →
  ((2*x - 5) / (x - 2) = 3 / (2 - x)) ↔ (x = 4) :=
sorry

-- Problem 2
theorem no_solution_rational_equation (x : ℝ) :
  x ≠ 3 →
  x ≠ -3 →
  ¬(12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) :=
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_no_solution_rational_equation_l3680_368059


namespace NUMINAMATH_CALUDE_magnitude_of_a_plus_bi_l3680_368045

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (a b : ℝ) : Prop :=
  a / (1 - i) = 1 - b * i

-- State the theorem
theorem magnitude_of_a_plus_bi (a b : ℝ) :
  given_equation a b → Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_a_plus_bi_l3680_368045


namespace NUMINAMATH_CALUDE_sweater_cost_l3680_368051

/-- Given shopping information, prove the cost of a sweater --/
theorem sweater_cost (initial_amount : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 91)
  (h2 : tshirt_cost = 6)
  (h3 : shoes_cost = 11)
  (h4 : remaining_amount = 50) :
  initial_amount - remaining_amount - tshirt_cost - shoes_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_sweater_cost_l3680_368051


namespace NUMINAMATH_CALUDE_congruence_solution_l3680_368053

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 5 % 47 → n % 47 = 4 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l3680_368053


namespace NUMINAMATH_CALUDE_jane_max_tickets_l3680_368077

/-- The maximum number of tickets that can be bought with a given budget, 
    given a regular price, discounted price, and discount threshold. -/
def maxTickets (budget : ℕ) (regularPrice discountPrice : ℕ) (discountThreshold : ℕ) : ℕ :=
  let regularTickets := budget / regularPrice
  let discountedTotal := 
    discountThreshold * regularPrice + 
    (budget - discountThreshold * regularPrice) / discountPrice
  max regularTickets discountedTotal

/-- Theorem: Given the specific conditions of the problem, 
    the maximum number of tickets Jane can buy is 11. -/
theorem jane_max_tickets : 
  maxTickets 150 15 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l3680_368077


namespace NUMINAMATH_CALUDE_solution_existence_l3680_368023

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- State the theorem
theorem solution_existence (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) ↔ m ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_solution_existence_l3680_368023


namespace NUMINAMATH_CALUDE_valid_tree_arrangement_exists_l3680_368018

/-- Represents a tree type -/
inductive TreeType
| Apple
| Pear
| Plum
| Apricot
| Cherry
| Almond

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point

/-- Represents the arrangement of trees -/
structure TreeArrangement where
  triangles : List EquilateralTriangle
  treeAssignment : Point → Option TreeType

/-- The main theorem stating that a valid tree arrangement exists -/
theorem valid_tree_arrangement_exists : ∃ (arrangement : TreeArrangement), 
  (arrangement.triangles.length = 6) ∧ 
  (∀ t ∈ arrangement.triangles, 
    ∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    arrangement.treeAssignment t.vertex1 = some t1 ∧
    arrangement.treeAssignment t.vertex2 = some t2 ∧
    arrangement.treeAssignment t.vertex3 = some t3) ∧
  (∀ p : Point, (∃ t ∈ arrangement.triangles, p ∈ [t.vertex1, t.vertex2, t.vertex3]) →
    ∃! treeType : TreeType, arrangement.treeAssignment p = some treeType) :=
by sorry

end NUMINAMATH_CALUDE_valid_tree_arrangement_exists_l3680_368018


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l3680_368055

theorem solve_cubic_equation (m : ℝ) : (m - 3)^3 = (1/27)⁻¹ → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l3680_368055


namespace NUMINAMATH_CALUDE_platform_length_l3680_368043

/-- Given a train passing a platform, calculate the platform length -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) 
  (h1 : train_length = 360) 
  (h2 : train_speed_kmh = 45) 
  (h3 : time_to_pass = 51.99999999999999) : 
  ∃ platform_length : ℝ, platform_length = 290 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3680_368043


namespace NUMINAMATH_CALUDE_f_composition_negative_four_l3680_368062

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else (1/2)^x

theorem f_composition_negative_four : f (f (-4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_four_l3680_368062


namespace NUMINAMATH_CALUDE_violet_necklace_problem_l3680_368025

theorem violet_necklace_problem (x : ℝ) 
  (h1 : (1/2 : ℝ) * x + 30 = (3/4 : ℝ) * x) : 
  (1/4 : ℝ) * x = 30 := by
  sorry

end NUMINAMATH_CALUDE_violet_necklace_problem_l3680_368025


namespace NUMINAMATH_CALUDE_max_NF_value_slope_AN_l3680_368057

-- Define the ellipse parameters
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
axiom major_minor_ratio : 2*a = (3*Real.sqrt 5 / 5) * (2*b)
axiom point_D_on_ellipse : ellipse a b (-1) (2*Real.sqrt 10 / 3)
axiom vector_relation (x₀ y₀ x₁ y₁ : ℝ) (hx₀ : x₀ > 0) (hy₀ : y₀ > 0) :
  ellipse a b x₀ y₀ → ellipse a b x₁ y₁ → (x₀, y₀) = 2 * (x₁ + 3, y₁)

-- State the theorems to be proved
theorem max_NF_value :
  ∃ (c : ℝ), c^2 = a^2 - b^2 ∧ a + c = 5 :=
sorry

theorem slope_AN :
  ∃ (x₀ y₀ : ℝ), ellipse a b x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0 ∧ y₀ / x₀ = 5 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_NF_value_slope_AN_l3680_368057


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l3680_368020

theorem dinner_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : Real) (total_bill : Real) : 
  total_friends = 10 →
  paying_friends = 9 →
  extra_payment = 3 →
  paying_friends * (total_bill / total_friends + extra_payment) = total_bill →
  total_bill = 270 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l3680_368020


namespace NUMINAMATH_CALUDE_distinct_fraction_equality_l3680_368083

theorem distinct_fraction_equality (a b c : ℝ) (k : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_distinct_fraction_equality_l3680_368083


namespace NUMINAMATH_CALUDE_ones_digit_of_36_power_ones_digit_of_36_to_large_power_l3680_368074

theorem ones_digit_of_36_power (n : ℕ) : (36 ^ n) % 10 = 6 := by sorry

theorem ones_digit_of_36_to_large_power :
  (36 ^ (36 * (5 ^ 5))) % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_36_power_ones_digit_of_36_to_large_power_l3680_368074


namespace NUMINAMATH_CALUDE_derivative_sin_cos_x_l3680_368084

theorem derivative_sin_cos_x (x : ℝ) : 
  deriv (fun x => Real.sin x * Real.cos x) x = Real.cos x ^ 2 - Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_x_l3680_368084


namespace NUMINAMATH_CALUDE_unique_solution_star_l3680_368093

/-- The ⋆ operation -/
def star (x y : ℝ) : ℝ := 5*x - 2*y + 3*x*y

/-- Theorem stating that 2 ⋆ y = 10 has a unique solution y = 0 -/
theorem unique_solution_star :
  ∃! y : ℝ, star 2 y = 10 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_star_l3680_368093


namespace NUMINAMATH_CALUDE_function_properties_l3680_368041

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →  -- g is an odd function
  (∃ c, ∀ x, f a b x = -1/3 * x^3 + x^2 + c) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≤ 4 * Real.sqrt 2 / 3) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≥ 4 / 3) ∧
  (∃ x ∈ Set.Icc 1 2, g (-1/3) 0 x = 4 * Real.sqrt 2 / 3) ∧
  (∃ x ∈ Set.Icc 1 2, g (-1/3) 0 x = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3680_368041


namespace NUMINAMATH_CALUDE_counterexample_exists_l3680_368079

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Statement of the theorem
theorem counterexample_exists : ∃ n : ℕ, 
  (sumOfDigits n % 9 = 0) ∧ (n % 9 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3680_368079


namespace NUMINAMATH_CALUDE_days_missed_difference_l3680_368090

/-- Represents the number of students who missed a certain number of days -/
structure DaysMissed where
  days : ℕ
  count : ℕ

/-- The histogram data -/
def histogram : List DaysMissed := [
  ⟨0, 3⟩, ⟨1, 1⟩, ⟨2, 4⟩, ⟨3, 1⟩, ⟨4, 1⟩, ⟨5, 5⟩
]

/-- The total number of students -/
def totalStudents : ℕ := 15

/-- Calculates the median number of days missed -/
def median (h : List DaysMissed) (total : ℕ) : ℚ := sorry

/-- Calculates the mean number of days missed -/
def mean (h : List DaysMissed) (total : ℕ) : ℚ := sorry

/-- The main theorem -/
theorem days_missed_difference :
  mean histogram totalStudents - median histogram totalStudents = 11 / 15 := by sorry

end NUMINAMATH_CALUDE_days_missed_difference_l3680_368090


namespace NUMINAMATH_CALUDE_expression_value_at_negative_two_l3680_368042

theorem expression_value_at_negative_two :
  let x : ℝ := -2
  let expr := (x - 5 + 16 / (x + 3)) / ((x - 1) / (x^2 - 9))
  expr = 15 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_two_l3680_368042


namespace NUMINAMATH_CALUDE_train_speed_constant_l3680_368089

/-- A train crossing a stationary man on a platform --/
structure Train :=
  (initial_length : ℝ)
  (initial_speed : ℝ)
  (length_increase_rate : ℝ)

/-- The final speed of the train after crossing the man --/
def final_speed (t : Train) : ℝ := t.initial_speed

theorem train_speed_constant (t : Train) 
  (h1 : t.initial_length = 160)
  (h2 : t.initial_speed = 30)
  (h3 : t.length_increase_rate = 2)
  (h4 : final_speed t = t.initial_speed) :
  final_speed t = 30 := by sorry

end NUMINAMATH_CALUDE_train_speed_constant_l3680_368089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_four_l3680_368095

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_before_four :
  ∀ n : ℕ, n ≤ 30 → arithmetic_sequence 92 (-3) n > 4 ∧
  arithmetic_sequence 92 (-3) 31 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_four_l3680_368095


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3680_368044

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - 2*x

theorem tangent_slope_at_zero :
  deriv f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3680_368044


namespace NUMINAMATH_CALUDE_chocolate_ice_cream_orders_l3680_368071

theorem chocolate_ice_cream_orders (total_orders : ℕ) (vanilla_ratio : ℚ) 
  (h1 : total_orders = 220)
  (h2 : vanilla_ratio = 1/5)
  (h3 : vanilla_ratio * total_orders = 2 * (total_orders - vanilla_ratio * total_orders - (total_orders - vanilla_ratio * total_orders))) :
  total_orders - vanilla_ratio * total_orders - (total_orders - vanilla_ratio * total_orders) = 22 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_ice_cream_orders_l3680_368071


namespace NUMINAMATH_CALUDE_dice_sum_multiple_of_5_prob_correct_l3680_368002

/-- The probability that the sum of n rolls of a 6-sided die is a multiple of 5 -/
def dice_sum_multiple_of_5_prob (n : ℕ) : ℚ :=
  if 5 ∣ n then
    (6^n + 4) / (5 * 6^n)
  else
    (6^n - 1) / (5 * 6^n)

/-- Theorem: The probability that the sum of n rolls of a 6-sided die is a multiple of 5
    is (6^n - 1) / (5 * 6^n) if 5 doesn't divide n, and (6^n + 4) / (5 * 6^n) if 5 divides n -/
theorem dice_sum_multiple_of_5_prob_correct (n : ℕ) :
  dice_sum_multiple_of_5_prob n =
    if 5 ∣ n then
      (6^n + 4) / (5 * 6^n)
    else
      (6^n - 1) / (5 * 6^n) := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_multiple_of_5_prob_correct_l3680_368002


namespace NUMINAMATH_CALUDE_jakes_third_test_score_l3680_368082

/-- Proof that Jake scored 65 marks in the third test given the conditions -/
theorem jakes_third_test_score :
  ∀ (third_test_score : ℕ),
  (∃ (first_test : ℕ) (second_test : ℕ) (fourth_test : ℕ),
    first_test = 80 ∧
    second_test = first_test + 10 ∧
    fourth_test = third_test_score ∧
    (first_test + second_test + third_test_score + fourth_test) / 4 = 75) →
  third_test_score = 65 := by
sorry

end NUMINAMATH_CALUDE_jakes_third_test_score_l3680_368082


namespace NUMINAMATH_CALUDE_calculate_expression_solve_inequality_system_l3680_368032

-- Part 1
theorem calculate_expression : (π - 3) ^ 0 + (-1) ^ 2023 - Real.sqrt 8 = -2 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem solve_inequality_system {x : ℝ} : (4 * x - 3 > 9 ∧ 2 + x ≥ 0) ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_inequality_system_l3680_368032


namespace NUMINAMATH_CALUDE_solution_value_l3680_368000

theorem solution_value (x a : ℝ) (h : 5 * 3 - a = 8) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3680_368000


namespace NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l3680_368011

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13]

def remove_number (l : List ℕ) (n : ℕ) : List ℕ :=
  l.filter (· ≠ n)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem remove_one_gives_average_seven_point_five :
  average (remove_number original_list 1) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l3680_368011


namespace NUMINAMATH_CALUDE_max_value_sum_product_l3680_368022

theorem max_value_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d + d * a ≤ 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_product_l3680_368022


namespace NUMINAMATH_CALUDE_rectangle_diagonals_plus_three_l3680_368069

theorem rectangle_diagonals_plus_three (rectangle_diagonals : ℕ) : 
  rectangle_diagonals = 2 → rectangle_diagonals + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonals_plus_three_l3680_368069


namespace NUMINAMATH_CALUDE_correct_passwords_count_l3680_368028

def total_passwords : ℕ := 10000

def invalid_passwords : ℕ := 10

theorem correct_passwords_count :
  total_passwords - invalid_passwords = 9990 :=
by sorry

end NUMINAMATH_CALUDE_correct_passwords_count_l3680_368028


namespace NUMINAMATH_CALUDE_birds_in_tree_l3680_368096

theorem birds_in_tree (initial_birds final_birds : ℕ) : 
  initial_birds = 231 → final_birds = 312 → 
  final_birds - initial_birds = 81 := by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3680_368096


namespace NUMINAMATH_CALUDE_tan_product_less_than_one_l3680_368075

theorem tan_product_less_than_one (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  π / 2 < C ∧ C < π →  -- Angle C is obtuse
  Real.tan A * Real.tan B < 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_less_than_one_l3680_368075


namespace NUMINAMATH_CALUDE_line_relationship_l3680_368086

-- Define a type for lines in space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder definition

-- Define parallel relationship between lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of parallel lines

-- Define intersection relationship between lines
def intersects (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of intersecting lines

-- Define skew relationship between lines
def skew (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of skew lines

-- Theorem statement
theorem line_relationship (a b c : Line3D) 
  (h1 : parallel a b) 
  (h2 : intersects a c) :
  skew b c ∨ intersects b c :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3680_368086


namespace NUMINAMATH_CALUDE_first_player_win_prob_l3680_368088

/-- The probability of the first player getting heads -/
def p1 : ℚ := 1/3

/-- The probability of the second player getting heads -/
def p2 : ℚ := 2/5

/-- The game where two players flip coins alternately until one gets heads -/
def coin_flip_game (p1 p2 : ℚ) : Prop :=
  p1 > 0 ∧ p1 < 1 ∧ p2 > 0 ∧ p2 < 1

/-- The probability of the first player winning the game -/
noncomputable def win_prob (p1 p2 : ℚ) : ℚ :=
  p1 / (1 - (1 - p1) * (1 - p2))

/-- Theorem stating that the probability of the first player winning is 5/9 -/
theorem first_player_win_prob :
  coin_flip_game p1 p2 → win_prob p1 p2 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_first_player_win_prob_l3680_368088


namespace NUMINAMATH_CALUDE_g_of_three_equals_fourteen_l3680_368085

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 2 * x + 4

-- State the theorem
theorem g_of_three_equals_fourteen : g 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_equals_fourteen_l3680_368085


namespace NUMINAMATH_CALUDE_circle_max_distance_squared_l3680_368048

theorem circle_max_distance_squared (x y : ℝ) (h : x^2 + (y - 2)^2 = 1) :
  x^2 + y^2 ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_circle_max_distance_squared_l3680_368048


namespace NUMINAMATH_CALUDE_volume_formula_correct_l3680_368047

/-- A solid formed by the union of a sphere and a truncated cone -/
structure SphereConeUnion where
  R : ℝ  -- radius of the sphere
  S : ℝ  -- total surface area of the solid

/-- The sphere is tangent to one base of the truncated cone -/
axiom sphere_tangent_base (solid : SphereConeUnion) : True

/-- The sphere is tangent to the lateral surface of the cone along a circle -/
axiom sphere_tangent_lateral (solid : SphereConeUnion) : True

/-- The circle of tangency coincides with the other base of the cone -/
axiom tangency_coincides_base (solid : SphereConeUnion) : True

/-- The volume of the solid formed by the union of the cone and the sphere -/
noncomputable def volume (solid : SphereConeUnion) : ℝ :=
  (1 / 3) * solid.S * solid.R

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula_correct (solid : SphereConeUnion) :
  volume solid = (1 / 3) * solid.S * solid.R := by sorry

end NUMINAMATH_CALUDE_volume_formula_correct_l3680_368047


namespace NUMINAMATH_CALUDE_even_quadratic_max_value_l3680_368054

/-- A quadratic function f(x) = ax^2 + bx + 1 defined on [-1-a, 2a] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The domain of the function -/
def domain (a : ℝ) : Set ℝ := Set.Icc (-1 - a) (2 * a)

/-- Theorem: If f is even on its domain, its maximum value is 5 -/
theorem even_quadratic_max_value (a b : ℝ) :
  (∀ x ∈ domain a, f a b x = f a b (-x)) →
  (∃ x ∈ domain a, ∀ y ∈ domain a, f a b y ≤ f a b x) →
  (∃ x ∈ domain a, f a b x = 5) :=
sorry

end NUMINAMATH_CALUDE_even_quadratic_max_value_l3680_368054


namespace NUMINAMATH_CALUDE_cyclic_sum_nonnegative_l3680_368070

theorem cyclic_sum_nonnegative (r : ℝ) (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x^r * (x - y) * (x - z) + y^r * (y - x) * (y - z) + z^r * (z - x) * (z - y) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_nonnegative_l3680_368070


namespace NUMINAMATH_CALUDE_f_decreasing_area_is_one_l3680_368029

/-- A function that is directly proportional to x-1 and passes through (-1, 4) -/
def f (x : ℝ) : ℝ := -2 * x + 2

/-- The property that f is directly proportional to x-1 -/
axiom f_prop (x : ℝ) : ∃ k : ℝ, f x = k * (x - 1)

/-- The property that f(-1) = 4 -/
axiom f_point : f (-1) = 4

/-- For any two x-values, if x1 > x2, then f(x1) < f(x2) -/
theorem f_decreasing (x1 x2 : ℝ) (h : x1 > x2) : f x1 < f x2 := by sorry

/-- The area of the triangle formed by shifting f down by 4 units -/
def triangle_area : ℝ := 1

/-- The area of the triangle formed by shifting f down by 4 units is 1 -/
theorem area_is_one : triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_area_is_one_l3680_368029


namespace NUMINAMATH_CALUDE_jean_needs_four_more_packs_l3680_368098

/-- Represents the number of cupcakes in a small pack -/
def small_pack : ℕ := 10

/-- Represents the number of cupcakes in a large pack -/
def large_pack : ℕ := 15

/-- Represents the number of large packs Jean initially bought -/
def initial_packs : ℕ := 4

/-- Represents the total number of children in the orphanage -/
def total_children : ℕ := 100

/-- Calculates the number of additional packs of 10 cupcakes Jean needs to buy -/
def additional_packs_needed : ℕ :=
  (total_children - initial_packs * large_pack) / small_pack

theorem jean_needs_four_more_packs :
  additional_packs_needed = 4 :=
sorry

end NUMINAMATH_CALUDE_jean_needs_four_more_packs_l3680_368098


namespace NUMINAMATH_CALUDE_increasing_function_condition_l3680_368040

/-- A function f(x) = (1/2)mx^2 + ln x - 2x is increasing on its domain (x > 0) if and only if m ≥ 1 -/
theorem increasing_function_condition (m : ℝ) :
  (∀ x > 0, Monotone (fun x => (1/2) * m * x^2 + Real.log x - 2*x)) ↔ m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l3680_368040


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l3680_368068

theorem candy_sampling_percentage (caught_sampling : Real) (total_sampling : Real) 
  (h1 : caught_sampling = 22)
  (h2 : total_sampling = 25.88235294117647) :
  total_sampling - caught_sampling = 3.88235294117647 := by
sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l3680_368068


namespace NUMINAMATH_CALUDE_value_of_expression_l3680_368021

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem value_of_expression (a b c d : ℝ) :
  f a b c d (-2) = -3 → 10 * a - 5 * b + 3 * c - 2 * d = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3680_368021


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l3680_368050

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : M ⊆ N a → a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l3680_368050


namespace NUMINAMATH_CALUDE_differential_equation_classification_l3680_368073

-- Define a type for equations
inductive Equation
| A : Equation  -- y' + 3x = 0
| B : Equation  -- y² + x² = 5
| C : Equation  -- y = e^x
| D : Equation  -- y = ln|x| + C
| E : Equation  -- y'y - x = 0
| F : Equation  -- 2dy + 3xdx = 0

-- Define a predicate for differential equations
def isDifferentialEquation : Equation → Prop
| Equation.A => True
| Equation.B => False
| Equation.C => False
| Equation.D => False
| Equation.E => True
| Equation.F => True

-- Theorem statement
theorem differential_equation_classification :
  (isDifferentialEquation Equation.A ∧
   isDifferentialEquation Equation.E ∧
   isDifferentialEquation Equation.F) ∧
  (¬isDifferentialEquation Equation.B ∧
   ¬isDifferentialEquation Equation.C ∧
   ¬isDifferentialEquation Equation.D) :=
by sorry

end NUMINAMATH_CALUDE_differential_equation_classification_l3680_368073


namespace NUMINAMATH_CALUDE_shortest_side_length_l3680_368027

/-- Represents a triangle with angles in the ratio 1:2:3 and longest side of length 6 -/
structure SpecialTriangle where
  /-- The smallest angle of the triangle -/
  smallest_angle : ℝ
  /-- The ratio of angles is 1:2:3 -/
  angle_ratio : smallest_angle > 0 ∧ smallest_angle + 2 * smallest_angle + 3 * smallest_angle = 180
  /-- The length of the longest side is 6 -/
  longest_side : ℝ
  longest_side_eq : longest_side = 6

/-- The length of the shortest side in a SpecialTriangle is 3 -/
theorem shortest_side_length (t : SpecialTriangle) : ∃ (shortest_side : ℝ), shortest_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l3680_368027


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3680_368080

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + 50) / 3 = ((x + 40 + 6) / 3) + 8 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3680_368080


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3680_368004

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3680_368004


namespace NUMINAMATH_CALUDE_pet_store_birds_count_l3680_368091

theorem pet_store_birds_count :
  let num_cages : ℕ := 8
  let parrots_per_cage : ℕ := 2
  let parakeets_per_cage : ℕ := 7
  let total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)
  total_birds = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_count_l3680_368091
