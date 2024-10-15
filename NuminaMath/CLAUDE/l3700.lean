import Mathlib

namespace NUMINAMATH_CALUDE_car_speed_problem_l3700_370021

/-- Given a car traveling for two hours, where its speed in the second hour is 30 km/h
    and its average speed over the two hours is 25 km/h, prove that the speed of the car
    in the first hour must be 20 km/h. -/
theorem car_speed_problem (first_hour_speed : ℝ) : 
  (first_hour_speed + 30) / 2 = 25 → first_hour_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3700_370021


namespace NUMINAMATH_CALUDE_point_slope_theorem_l3700_370049

theorem point_slope_theorem (k : ℝ) (h1 : k > 0) : 
  (2 - k) / (k - 1) = k^2 → k = 1 := by sorry

end NUMINAMATH_CALUDE_point_slope_theorem_l3700_370049


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3700_370068

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3700_370068


namespace NUMINAMATH_CALUDE_amusement_park_group_composition_l3700_370083

theorem amusement_park_group_composition :
  let total_cost : ℕ := 720
  let adult_price : ℕ := 15
  let child_price : ℕ := 8
  let num_children : ℕ := 15
  let num_adults : ℕ := (total_cost - child_price * num_children) / adult_price
  num_adults - num_children = 25 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_group_composition_l3700_370083


namespace NUMINAMATH_CALUDE_xy_value_l3700_370006

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3700_370006


namespace NUMINAMATH_CALUDE_product_of_numbers_l3700_370032

theorem product_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 20) 
  (sum_squares_condition : x^2 + y^2 = 200) : 
  x * y = 100 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3700_370032


namespace NUMINAMATH_CALUDE_lobster_theorem_l3700_370001

/-- The total pounds of lobster in three harbors -/
def total_lobster (hooper_bay other1 other2 : ℕ) : ℕ := hooper_bay + other1 + other2

/-- Theorem stating the total pounds of lobster in the three harbors -/
theorem lobster_theorem (hooper_bay other1 other2 : ℕ) 
  (h1 : hooper_bay = 2 * (other1 + other2)) 
  (h2 : other1 = 80) 
  (h3 : other2 = 80) : 
  total_lobster hooper_bay other1 other2 = 480 := by
  sorry

#check lobster_theorem

end NUMINAMATH_CALUDE_lobster_theorem_l3700_370001


namespace NUMINAMATH_CALUDE_polynomial_equality_constant_l3700_370047

theorem polynomial_equality_constant (s : ℚ) : 
  (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 
    15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) → 
  s = -95/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_constant_l3700_370047


namespace NUMINAMATH_CALUDE_exam_mean_score_l3700_370097

theorem exam_mean_score (score_below mean score_above : ℝ) 
  (h1 : score_below = mean - 7 * (score_above - mean) / 3)
  (h2 : score_above = mean + 3 * (score_above - mean) / 3)
  (h3 : score_below = 86)
  (h4 : score_above = 90) :
  mean = 88.8 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3700_370097


namespace NUMINAMATH_CALUDE_fruit_cost_prices_l3700_370037

/-- Represents the cost and selling prices of fruits -/
structure FruitPrices where
  apple_sell : ℚ
  orange_sell : ℚ
  banana_sell : ℚ
  apple_loss : ℚ
  orange_loss : ℚ
  banana_gain : ℚ

/-- Calculates the cost price given selling price and loss/gain percentage -/
def cost_price (sell : ℚ) (loss_gain : ℚ) (is_gain : Bool) : ℚ :=
  if is_gain then
    sell / (1 + loss_gain)
  else
    sell / (1 - loss_gain)

/-- Theorem stating the correct cost prices for the fruits -/
theorem fruit_cost_prices (prices : FruitPrices)
  (h_apple_sell : prices.apple_sell = 20)
  (h_orange_sell : prices.orange_sell = 15)
  (h_banana_sell : prices.banana_sell = 6)
  (h_apple_loss : prices.apple_loss = 1/6)
  (h_orange_loss : prices.orange_loss = 1/4)
  (h_banana_gain : prices.banana_gain = 1/8) :
  cost_price prices.apple_sell prices.apple_loss false = 24 ∧
  cost_price prices.orange_sell prices.orange_loss false = 20 ∧
  cost_price prices.banana_sell prices.banana_gain true = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_prices_l3700_370037


namespace NUMINAMATH_CALUDE_stream_current_is_three_l3700_370011

/-- Represents the rowing scenario described in the problem -/
structure RowingScenario where
  r : ℝ  -- man's rowing speed in still water (miles per hour)
  c : ℝ  -- speed of the stream's current (miles per hour)
  distance : ℝ  -- distance traveled (miles)
  timeDiffNormal : ℝ  -- time difference between upstream and downstream at normal rate (hours)
  timeDiffTripled : ℝ  -- time difference between upstream and downstream at tripled rate (hours)

/-- The theorem stating that given the problem conditions, the stream's current is 3 mph -/
theorem stream_current_is_three 
  (scenario : RowingScenario)
  (h1 : scenario.distance = 20)
  (h2 : scenario.timeDiffNormal = 6)
  (h3 : scenario.timeDiffTripled = 1.5)
  (h4 : scenario.distance / (scenario.r + scenario.c) + scenario.timeDiffNormal = 
        scenario.distance / (scenario.r - scenario.c))
  (h5 : scenario.distance / (3 * scenario.r + scenario.c) + scenario.timeDiffTripled = 
        scenario.distance / (3 * scenario.r - scenario.c))
  : scenario.c = 3 := by
  sorry

#check stream_current_is_three

end NUMINAMATH_CALUDE_stream_current_is_three_l3700_370011


namespace NUMINAMATH_CALUDE_barycentric_coords_exist_and_unique_l3700_370050

-- Define a triangle in 2D space
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

-- Define barycentric coordinates
structure BarycentricCoords where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- State the theorem
theorem barycentric_coords_exist_and_unique (t : Triangle) (X : Point) :
  ∃! (b : BarycentricCoords),
    b.m₁ + b.m₂ + b.m₃ = 1 ∧
    X = (b.m₁ * t.A₁.1 + b.m₂ * t.A₂.1 + b.m₃ * t.A₃.1,
         b.m₁ * t.A₁.2 + b.m₂ * t.A₂.2 + b.m₃ * t.A₃.2) :=
  sorry

end NUMINAMATH_CALUDE_barycentric_coords_exist_and_unique_l3700_370050


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3700_370060

/-- The expression ax^2 + 2bxy + cy^2 - k(x^2 + y^2) is a perfect square if and only if 
    k = (a+c)/2 ± (1/2)√((a-c)^2 + 4b^2), where a, b, c are real constants. -/
theorem perfect_square_condition (a b c k : ℝ) :
  (∃ (f : ℝ → ℝ → ℝ), ∀ (x y : ℝ), a * x^2 + 2 * b * x * y + c * y^2 - k * (x^2 + y^2) = (f x y)^2) ↔
  (k = (a + c) / 2 + (1 / 2) * Real.sqrt ((a - c)^2 + 4 * b^2) ∨
   k = (a + c) / 2 - (1 / 2) * Real.sqrt ((a - c)^2 + 4 * b^2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3700_370060


namespace NUMINAMATH_CALUDE_smallest_positive_angle_for_neg1990_l3700_370098

-- Define the concept of angle equivalence
def angle_equivalent (a b : Int) : Prop :=
  ∃ k : Int, b - a = k * 360

-- Define the smallest positive equivalent angle
def smallest_positive_equivalent (a : Int) : Int :=
  let b := a % 360
  if b < 0 then b + 360 else b

-- Theorem statement
theorem smallest_positive_angle_for_neg1990 :
  smallest_positive_equivalent (-1990) = 170 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_for_neg1990_l3700_370098


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_of_570_l3700_370023

theorem gcd_of_polynomial_and_multiple_of_570 (b : ℤ) : 
  (∃ k : ℤ, b = 570 * k) → Int.gcd (4 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_of_570_l3700_370023


namespace NUMINAMATH_CALUDE_star_wars_earnings_value_l3700_370058

/-- The cost to make The Lion King in millions of dollars -/
def lion_king_cost : ℝ := 10

/-- The box office earnings of The Lion King in millions of dollars -/
def lion_king_earnings : ℝ := 200

/-- The cost to make Star Wars in millions of dollars -/
def star_wars_cost : ℝ := 25

/-- The profit of The Lion King in millions of dollars -/
def lion_king_profit : ℝ := lion_king_earnings - lion_king_cost

/-- The profit of Star Wars in millions of dollars -/
def star_wars_profit : ℝ := 2 * lion_king_profit

/-- The earnings of Star Wars in millions of dollars -/
def star_wars_earnings : ℝ := star_wars_cost + star_wars_profit

theorem star_wars_earnings_value : star_wars_earnings = 405 := by
  sorry

#eval star_wars_earnings

end NUMINAMATH_CALUDE_star_wars_earnings_value_l3700_370058


namespace NUMINAMATH_CALUDE_real_square_nonnegative_and_no_real_square_root_of_negative_one_l3700_370045

theorem real_square_nonnegative_and_no_real_square_root_of_negative_one :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬(∃ x : ℝ, x^2 = -1) := by
  sorry

end NUMINAMATH_CALUDE_real_square_nonnegative_and_no_real_square_root_of_negative_one_l3700_370045


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l3700_370034

/-- Represents the number of years until a man's age is twice his son's age. -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  let x := man_age + 2 - 2 * (son_age + 2)
  2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2,
    given the son's current age and the age difference between the man and his son. -/
theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
    (h1 : son_age = 28) (h2 : age_difference = 30) : 
    years_until_double_age son_age age_difference = 2 := by
  sorry

#eval years_until_double_age 28 30

end NUMINAMATH_CALUDE_double_age_in_two_years_l3700_370034


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3700_370077

theorem fraction_equation_solution (x : ℚ) :
  2/5 - 1/4 = 1/x → x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3700_370077


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_no_solution_product_equation_l3700_370029

/-- Given x and y are positive real numbers satisfying x^2 + y^2 = x + y -/
def satisfies_equation (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 + y^2 = x + y

/-- The minimum value of 1/x + 1/y is 2 -/
theorem min_value_sum_reciprocals {x y : ℝ} (h : satisfies_equation x y) :
  1/x + 1/y ≥ 2 := by
  sorry

/-- There do not exist x and y satisfying (x+1)(y+1) = 5 -/
theorem no_solution_product_equation {x y : ℝ} (h : satisfies_equation x y) :
  (x + 1) * (y + 1) ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_no_solution_product_equation_l3700_370029


namespace NUMINAMATH_CALUDE_hyperbola_third_point_x_squared_l3700_370081

/-- A hyperbola is defined by its center, orientation, and three points it passes through. -/
structure Hyperbola where
  center : ℝ × ℝ
  opens_horizontally : Bool
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The theorem states that for a specific hyperbola, the square of the x-coordinate of its third point is 361/36. -/
theorem hyperbola_third_point_x_squared (h : Hyperbola) 
  (h_center : h.center = (1, 0))
  (h_orientation : h.opens_horizontally = true)
  (h_point1 : h.point1 = (0, 3))
  (h_point2 : h.point2 = (1, -4))
  (h_point3 : h.point3.2 = -1) :
  (h.point3.1)^2 = 361/36 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_third_point_x_squared_l3700_370081


namespace NUMINAMATH_CALUDE_constant_equation_solution_l3700_370040

theorem constant_equation_solution (n : ℝ) : 
  (∃ m : ℝ, 21 * (m + n) + 21 = 21 * (-m + n) + 21) → 
  (∃ m : ℝ, m = 0 ∧ 21 * (m + n) + 21 = 21 * (-m + n) + 21) :=
by sorry

end NUMINAMATH_CALUDE_constant_equation_solution_l3700_370040


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3700_370079

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (∀ x : ℝ, x^2 + 16*x = 100 ↔ x = Real.sqrt a - b ∨ x = -Real.sqrt a - b) ∧ 
  (Real.sqrt a - b > 0) ∧
  (a + b = 172) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3700_370079


namespace NUMINAMATH_CALUDE_doughnuts_given_away_l3700_370009

def doughnuts_per_box : ℕ := 10
def total_doughnuts : ℕ := 300
def boxes_sold : ℕ := 27

theorem doughnuts_given_away : ℕ := by
  have h1 : total_doughnuts % doughnuts_per_box = 0 := by sorry
  have h2 : total_doughnuts / doughnuts_per_box > boxes_sold := by sorry
  exact (total_doughnuts / doughnuts_per_box - boxes_sold) * doughnuts_per_box

end NUMINAMATH_CALUDE_doughnuts_given_away_l3700_370009


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3700_370014

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.2 : ℝ)⌉ = 27 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3700_370014


namespace NUMINAMATH_CALUDE_abs_inequality_iff_inequality_l3700_370070

theorem abs_inequality_iff_inequality (a b : ℝ) : a > b ↔ a * |a| > b * |b| := by sorry

end NUMINAMATH_CALUDE_abs_inequality_iff_inequality_l3700_370070


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l3700_370010

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → 
  n ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l3700_370010


namespace NUMINAMATH_CALUDE_derivative_of_f_tangent_line_at_one_l3700_370078

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x * Real.log x

-- State the theorem for the derivative of f
theorem derivative_of_f :
  deriv f = fun x => 2 * x + Real.log x + 1 :=
sorry

-- Define the tangent line function
def tangent_line (x y : ℝ) : ℝ := 3 * x - y - 2

-- State the theorem for the tangent line at x=1
theorem tangent_line_at_one :
  ∀ x y, f 1 = y → deriv f 1 * (x - 1) + y = tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_derivative_of_f_tangent_line_at_one_l3700_370078


namespace NUMINAMATH_CALUDE_road_trip_distance_ratio_l3700_370024

theorem road_trip_distance_ratio : 
  ∀ (total_distance first_day_distance second_day_distance third_day_distance : ℝ),
  total_distance = 525 →
  first_day_distance = 200 →
  second_day_distance = 3/4 * first_day_distance →
  third_day_distance = total_distance - (first_day_distance + second_day_distance) →
  third_day_distance / (first_day_distance + second_day_distance) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_road_trip_distance_ratio_l3700_370024


namespace NUMINAMATH_CALUDE_base_b_square_l3700_370030

theorem base_b_square (b : ℕ) (h : b > 1) : 
  (3 * b^2 + 4 * b + 3 = (b + 3)^2) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l3700_370030


namespace NUMINAMATH_CALUDE_smallest_constant_term_l3700_370055

theorem smallest_constant_term (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -3 ∨ x = 6 ∨ x = 10 ∨ x = -1/4) →
  e > 0 →
  e ≥ 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_term_l3700_370055


namespace NUMINAMATH_CALUDE_exists_color_with_all_distances_l3700_370084

-- Define a type for colors
inductive Color
| Yellow
| Red

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a distance function between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem exists_color_with_all_distances :
  ∃ c : Color, ∀ x : ℝ, x > 0 → ∃ p1 p2 : Point,
    coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by sorry

end NUMINAMATH_CALUDE_exists_color_with_all_distances_l3700_370084


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3700_370002

def total_books : ℕ := 7
def science_books : ℕ := 2
def math_books : ℕ := 2
def unique_books : ℕ := total_books - science_books - math_books

def arrangements : ℕ := (total_books.factorial) / (science_books.factorial * math_books.factorial)

def highlighted_arrangements : ℕ := arrangements * (total_books.choose 2)

theorem book_arrangement_theorem :
  arrangements = 1260 ∧ highlighted_arrangements = 26460 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3700_370002


namespace NUMINAMATH_CALUDE_input_statement_separator_l3700_370054

/-- Represents the possible separators in an input statement -/
inductive Separator
  | Comma
  | Space
  | Semicolon
  | Pause

/-- Represents the general format of an input statement -/
structure InputStatement where
  separator : Separator

/-- The correct separator for multiple variables in an input statement -/
def correctSeparator : Separator := Separator.Comma

/-- Theorem stating that the correct separator in the general format of an input statement is a comma -/
theorem input_statement_separator :
  ∀ (stmt : InputStatement), stmt.separator = correctSeparator :=
sorry


end NUMINAMATH_CALUDE_input_statement_separator_l3700_370054


namespace NUMINAMATH_CALUDE_socks_cost_theorem_l3700_370061

/-- The cost of each red pair of socks -/
def red_cost : ℝ := 3

/-- The number of red sock pairs -/
def red_pairs : ℕ := 4

/-- The number of blue sock pairs -/
def blue_pairs : ℕ := 6

/-- The cost of each blue pair of socks -/
def blue_cost : ℝ := 5

/-- The total cost of all socks -/
def total_cost : ℝ := 42

theorem socks_cost_theorem :
  red_cost * red_pairs + blue_cost * blue_pairs = total_cost :=
by sorry

end NUMINAMATH_CALUDE_socks_cost_theorem_l3700_370061


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3700_370052

theorem quadratic_equation_roots (a : ℝ) :
  a = 1 →
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    x₁^2 + (1 - a) * x₁ - 1 = 0 ∧
    x₂^2 + (1 - a) * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3700_370052


namespace NUMINAMATH_CALUDE_function_properties_l3700_370041

def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def isOddOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f x

def isLinearOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ m k, ∀ x, a ≤ x ∧ x ≤ b → f x = m * x + k

def isQuadraticOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ a₀ a₁ a₂, ∀ x, a ≤ x ∧ x ≤ b → f x = a₂ * x^2 + a₁ * x + a₀

theorem function_properties (f : ℝ → ℝ) 
    (h1 : isPeriodic f 5)
    (h2 : isOddOn f (-1) 1)
    (h3 : isLinearOn f 0 1)
    (h4 : isQuadraticOn f 1 4)
    (h5 : ∃ x, x = 2 ∧ f x = -5 ∧ ∀ y, f y ≥ -5) :
  (f 1 + f 4 = 0) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = 5/3 * (x - 2)^2 - 5) ∧
  (∀ x, 4 ≤ x ∧ x ≤ 6 → f x = -10/3 * x + 50/3) ∧
  (∀ x, 6 < x ∧ x ≤ 9 → f x = 5/3 * (x - 7)^2 - 5) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3700_370041


namespace NUMINAMATH_CALUDE_expressions_evaluation_l3700_370093

theorem expressions_evaluation :
  let expr1 := (1) * (Real.sqrt 48 - 4 * Real.sqrt (1/8)) - (2 * Real.sqrt (1/3) - 2 * Real.sqrt 0.5)
  let expr2 := Real.sqrt ((-2)^2) - |1 - Real.sqrt 3| + (3 - Real.sqrt 3) * (1 + 1 / Real.sqrt 3)
  (expr1 = (10/3) * Real.sqrt 3) ∧ (expr2 = 5 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_expressions_evaluation_l3700_370093


namespace NUMINAMATH_CALUDE_apples_count_l3700_370025

/-- The number of apples in the market -/
def apples : ℕ := sorry

/-- The number of oranges in the market -/
def oranges : ℕ := sorry

/-- The number of bananas in the market -/
def bananas : ℕ := sorry

/-- There are 27 more apples than oranges -/
axiom apples_oranges_diff : apples = oranges + 27

/-- There are 11 more oranges than bananas -/
axiom oranges_bananas_diff : oranges = bananas + 11

/-- The total number of fruits is 301 -/
axiom total_fruits : apples + oranges + bananas = 301

/-- The number of apples in the market is 122 -/
theorem apples_count : apples = 122 := by sorry

end NUMINAMATH_CALUDE_apples_count_l3700_370025


namespace NUMINAMATH_CALUDE_sugar_amount_proof_l3700_370072

/-- The amount of sugar in pounds in the first combination -/
def sugar_amount : ℝ := 39

/-- The cost per pound of sugar and flour in dollars -/
def cost_per_pound : ℝ := 0.45

/-- The cost of the first combination in dollars -/
def cost_first : ℝ := 26

/-- The cost of the second combination in dollars -/
def cost_second : ℝ := 26

/-- The amount of flour in the first combination in pounds -/
def flour_first : ℝ := 16

/-- The amount of sugar in the second combination in pounds -/
def sugar_second : ℝ := 30

/-- The amount of flour in the second combination in pounds -/
def flour_second : ℝ := 25

theorem sugar_amount_proof :
  cost_per_pound * sugar_amount + cost_per_pound * flour_first = cost_first ∧
  cost_per_pound * sugar_second + cost_per_pound * flour_second = cost_second ∧
  sugar_amount + flour_first = sugar_second + flour_second :=
by sorry

end NUMINAMATH_CALUDE_sugar_amount_proof_l3700_370072


namespace NUMINAMATH_CALUDE_binomial_nine_choose_five_l3700_370031

theorem binomial_nine_choose_five : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_nine_choose_five_l3700_370031


namespace NUMINAMATH_CALUDE_greenhouse_renovation_l3700_370048

/-- Greenhouse renovation problem -/
theorem greenhouse_renovation 
  (cost_2A_vs_1B : ℝ) 
  (cost_1A_2B : ℝ) 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (total_greenhouses : ℕ) 
  (max_budget : ℝ) 
  (max_days : ℝ)
  (h1 : cost_2A_vs_1B = 6)
  (h2 : cost_1A_2B = 48)
  (h3 : days_A = 5)
  (h4 : days_B = 3)
  (h5 : total_greenhouses = 8)
  (h6 : max_budget = 128)
  (h7 : max_days = 35) :
  ∃ (cost_A cost_B : ℝ),
    cost_A = 12 ∧ 
    cost_B = 18 ∧
    2 * cost_A = cost_B + cost_2A_vs_1B ∧
    cost_A + 2 * cost_B = cost_1A_2B ∧
    (∀ m : ℕ, 
      (m ≤ total_greenhouses ∧
       m * cost_A + (total_greenhouses - m) * cost_B ≤ max_budget ∧
       m * days_A + (total_greenhouses - m) * days_B ≤ max_days) 
      ↔ m ∈ ({3, 4, 5} : Set ℕ)) :=
sorry

end NUMINAMATH_CALUDE_greenhouse_renovation_l3700_370048


namespace NUMINAMATH_CALUDE_smurfs_gold_coins_l3700_370075

theorem smurfs_gold_coins (total : ℕ) (smurfs : ℕ) (gargamel : ℕ) 
  (h1 : total = 200)
  (h2 : smurfs + gargamel = total)
  (h3 : (2 : ℚ) / 3 * smurfs = (4 : ℚ) / 5 * gargamel + 38) :
  smurfs = 135 := by
  sorry

end NUMINAMATH_CALUDE_smurfs_gold_coins_l3700_370075


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3700_370056

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_power_sum : i^23 + i^45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3700_370056


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l3700_370035

theorem max_product_under_constraint (a b : ℝ) :
  a > 0 → b > 0 → 5 * a + 8 * b = 80 → ab ≤ 40 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 5 * a + 8 * b = 80 ∧ a * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l3700_370035


namespace NUMINAMATH_CALUDE_sandy_fingernail_record_l3700_370020

/-- Calculates the length of fingernails after a given number of years -/
def fingernail_length (current_age : ℕ) (target_age : ℕ) (current_length : ℝ) (growth_rate : ℝ) : ℝ :=
  current_length + (target_age - current_age) * 12 * growth_rate

/-- Proves that Sandy's fingernails will be 26 inches long at age 32, given the initial conditions -/
theorem sandy_fingernail_record :
  fingernail_length 12 32 2 0.1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fingernail_record_l3700_370020


namespace NUMINAMATH_CALUDE_log_equality_l3700_370005

theorem log_equality (x : ℝ) (h : x > 0) :
  (Real.log (2 * x) / Real.log (5 * x) = Real.log (8 * x) / Real.log (625 * x)) →
  (Real.log x / Real.log 2 = Real.log 5 / (3 * (Real.log 2 - Real.log 5))) := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3700_370005


namespace NUMINAMATH_CALUDE_largest_m_l3700_370071

/-- A three-digit positive integer that is the product of three distinct prime factors --/
def m (x y : ℕ) : ℕ := x * y * (10 * x + y)

/-- The proposition that m is the largest possible value given the conditions --/
theorem largest_m : ∀ x y : ℕ, 
  x < 10 → y < 10 → x ≠ y → 
  Nat.Prime x → Nat.Prime y → Nat.Prime (10 * x + y) →
  m x y ≤ 795 ∧ m x y < 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_m_l3700_370071


namespace NUMINAMATH_CALUDE_nba_division_impossibility_l3700_370019

theorem nba_division_impossibility : ∀ (A B : ℕ),
  A + B = 30 →
  A * B ≠ (30 * 82) / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_nba_division_impossibility_l3700_370019


namespace NUMINAMATH_CALUDE_bicycle_inventory_decrease_is_58_l3700_370089

/-- Calculates the decrease in bicycle inventory from January 1 to October 1 -/
def bicycle_inventory_decrease : ℕ :=
  let initial_inventory : ℕ := 200
  let feb_to_june_decrease : ℕ := 4 + 6 + 8 + 10 + 12
  let july_decrease : ℕ := 14
  let august_decrease : ℕ := 16 + 20  -- Including sales event
  let september_decrease : ℕ := 18
  let new_shipment : ℕ := 50
  (feb_to_june_decrease + july_decrease + august_decrease + september_decrease) - new_shipment

/-- Theorem stating that the bicycle inventory decrease from January 1 to October 1 is 58 -/
theorem bicycle_inventory_decrease_is_58 : bicycle_inventory_decrease = 58 := by
  sorry

#eval bicycle_inventory_decrease

end NUMINAMATH_CALUDE_bicycle_inventory_decrease_is_58_l3700_370089


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3700_370004

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3700_370004


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3700_370012

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3700_370012


namespace NUMINAMATH_CALUDE_max_value_abc_l3700_370065

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  a^4 * b^3 * c^2 ≤ 1 / 6561 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3700_370065


namespace NUMINAMATH_CALUDE_oliver_age_l3700_370099

/-- Given the ages of Mark, Nina, and Oliver, prove that Oliver is 22 years old. -/
theorem oliver_age (m n o : ℕ) : 
  (m + n + o) / 3 = 12 →  -- Average age is 12
  o - 5 = 2 * n →  -- Five years ago, Oliver was twice Nina's current age
  m + 2 = (4 * (n + 2)) / 5 →  -- In 2 years, Mark's age will be 4/5 of Nina's
  m + 4 + n + 4 + o + 4 = 60 →  -- In 4 years, total age will be 60
  o = 22 := by
  sorry

end NUMINAMATH_CALUDE_oliver_age_l3700_370099


namespace NUMINAMATH_CALUDE_surface_area_ratio_l3700_370092

/-- The ratio of the total surface area of n³ unit cubes to the surface area of a cube with edge length n is equal to n. -/
theorem surface_area_ratio (n : ℕ) (h : n > 0) :
  (n^3 * (6 : ℝ)) / (6 * n^2) = n :=
sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l3700_370092


namespace NUMINAMATH_CALUDE_red_balls_count_l3700_370094

def bag_sizes : List Nat := [7, 15, 16, 10, 23]

def total_balls : Nat := bag_sizes.sum

structure BallConfiguration where
  red : Nat
  yellow : Nat
  blue : Nat

def is_valid_configuration (config : BallConfiguration) : Prop :=
  config.red ∈ bag_sizes ∧
  config.yellow + config.blue = total_balls - config.red ∧
  config.yellow = 2 * config.blue

theorem red_balls_count : ∃ (config : BallConfiguration), 
  is_valid_configuration config ∧ config.red = 23 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3700_370094


namespace NUMINAMATH_CALUDE_ratio_of_bases_l3700_370038

/-- An isosceles trapezoid circumscribed about a circle -/
structure CircumscribedTrapezoid where
  /-- The longer base of the trapezoid -/
  AD : ℝ
  /-- The shorter base of the trapezoid -/
  BC : ℝ
  /-- The ratio of AN to NM, where N is the intersection of AM and the circle -/
  k : ℝ
  /-- AD is longer than BC -/
  h_AD_gt_BC : AD > BC
  /-- The trapezoid is isosceles -/
  h_isosceles : True
  /-- The trapezoid is circumscribed about a circle -/
  h_circumscribed : True
  /-- The circle touches one of the non-parallel sides -/
  h_touches_side : True
  /-- AM intersects the circle at N -/
  h_AM_intersects : True

/-- The ratio of the longer base to the shorter base in a circumscribed isosceles trapezoid -/
theorem ratio_of_bases (t : CircumscribedTrapezoid) : t.AD / t.BC = 8 * t.k - 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_bases_l3700_370038


namespace NUMINAMATH_CALUDE_area_of_AEC_l3700_370063

-- Define the triangle ABC and its area
def triangle_ABC : Real := 40

-- Define the points on the sides of the triangle
def point_D : Real := 3
def point_B : Real := 5

-- Define the equality of areas
def area_equality : Prop := true

-- Theorem to prove
theorem area_of_AEC (triangle_ABC : Real) (point_D point_B : Real) (area_equality : Prop) :
  (3 : Real) / 8 * triangle_ABC = 15 := by
  sorry

end NUMINAMATH_CALUDE_area_of_AEC_l3700_370063


namespace NUMINAMATH_CALUDE_montague_population_fraction_l3700_370046

/-- The fraction of the population living in Montague province -/
def montague_fraction : ℝ := sorry

/-- The fraction of the population living in Capulet province -/
def capulet_fraction : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem montague_population_fraction :
  -- Conditions
  (montague_fraction + capulet_fraction = 1) ∧
  (0.8 * montague_fraction + 0.3 * capulet_fraction = 0.7 * capulet_fraction / (7/11)) →
  -- Conclusion
  montague_fraction = 2/3 := by sorry

end NUMINAMATH_CALUDE_montague_population_fraction_l3700_370046


namespace NUMINAMATH_CALUDE_exam_grading_rules_l3700_370062

-- Define the types
def Student : Type := String
def Grade : Type := String
def Essay : Type := Bool

-- Define the predicates
def all_mc_correct (s : Student) : Prop := sorry
def satisfactory_essay (s : Student) : Prop := sorry
def grade_is (s : Student) (g : Grade) : Prop := sorry

-- State the theorem
theorem exam_grading_rules (s : Student) :
  -- Condition 1
  (∀ s, all_mc_correct s → grade_is s "B" ∨ grade_is s "A") →
  -- Condition 2
  (∀ s, all_mc_correct s ∧ satisfactory_essay s → grade_is s "A") →
  -- Statement D
  (grade_is s "A" → all_mc_correct s) ∧
  -- Statement E
  (all_mc_correct s ∧ satisfactory_essay s → grade_is s "A") := by
  sorry


end NUMINAMATH_CALUDE_exam_grading_rules_l3700_370062


namespace NUMINAMATH_CALUDE_different_color_probability_l3700_370027

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

theorem different_color_probability : 
  let prob_blue_not_blue := (blue_chips : ℚ) / total_chips * ((red_chips + yellow_chips) : ℚ) / total_chips
  let prob_red_not_red := (red_chips : ℚ) / total_chips * ((blue_chips + yellow_chips) : ℚ) / total_chips
  let prob_yellow_not_yellow := (yellow_chips : ℚ) / total_chips * ((blue_chips + red_chips) : ℚ) / total_chips
  prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow = 148 / 225 :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l3700_370027


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3700_370057

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p > 30 ∧ q > 30 ∧
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ r s : ℕ, is_prime r → is_prime s → r > 30 → s > 30 → r ≠ s → p * q ≤ r * s :=
sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3700_370057


namespace NUMINAMATH_CALUDE_base_eight_perfect_square_c_is_one_l3700_370028

/-- Represents a number in base 8 with the form 1b27c -/
def BaseEightNumber (b c : ℕ) : ℕ := 1024 + 64 * b + 16 + 7 + c

/-- A number is a perfect square if there exists an integer whose square is that number -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The last digit of a perfect square in base 8 can only be 0, 1, or 4 -/
axiom perfect_square_mod_8 (n : ℕ) : IsPerfectSquare n → n % 8 ∈ ({0, 1, 4} : Set ℕ)

theorem base_eight_perfect_square_c_is_one (b : ℕ) :
  IsPerfectSquare (BaseEightNumber b 1) →
  ∀ c : ℕ, IsPerfectSquare (BaseEightNumber b c) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_perfect_square_c_is_one_l3700_370028


namespace NUMINAMATH_CALUDE_inequality_proof_l3700_370082

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) ≤ (Real.sqrt 33 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3700_370082


namespace NUMINAMATH_CALUDE_amount_with_r_l3700_370022

theorem amount_with_r (total : ℝ) (p q r : ℝ) : 
  total = 9000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 3600 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l3700_370022


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3700_370017

theorem remainder_of_large_number (n : ℕ) (d : ℕ) (h : n = 123456789012 ∧ d = 126) :
  n % d = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3700_370017


namespace NUMINAMATH_CALUDE_max_decreasing_votes_is_five_l3700_370013

/-- A movie rating system with integer ratings from 0 to 10 -/
structure MovieRating where
  ratings : List ℕ
  valid_ratings : ∀ r ∈ ratings, r ≤ 10

/-- Calculate the current rating as the average of all ratings -/
def current_rating (mr : MovieRating) : ℚ :=
  (mr.ratings.sum : ℚ) / mr.ratings.length

/-- The maximum number of consecutive votes that can decrease the rating by 1 each time -/
def max_consecutive_decreasing_votes (mr : MovieRating) : ℕ :=
  sorry

/-- Theorem: The maximum number of consecutive votes that can decrease 
    an integer rating by 1 each time is 5 -/
theorem max_decreasing_votes_is_five (mr : MovieRating) 
  (h : ∃ n : ℕ, current_rating mr = n) :
  max_consecutive_decreasing_votes mr = 5 :=
sorry

end NUMINAMATH_CALUDE_max_decreasing_votes_is_five_l3700_370013


namespace NUMINAMATH_CALUDE_owen_profit_l3700_370003

/-- Calculate Owen's overall profit from selling face masks --/
theorem owen_profit : 
  let cheap_boxes := 8
  let expensive_boxes := 4
  let cheap_box_price := 9
  let expensive_box_price := 12
  let masks_per_box := 50
  let small_packets := 100
  let small_packet_price := 5
  let small_packet_size := 25
  let large_packets := 28
  let large_packet_price := 12
  let large_packet_size := 100
  let remaining_cheap := 150
  let remaining_expensive := 150
  let remaining_cheap_price := 3
  let remaining_expensive_price := 4

  let total_cost := cheap_boxes * cheap_box_price + expensive_boxes * expensive_box_price
  let total_masks := (cheap_boxes + expensive_boxes) * masks_per_box
  let repacked_revenue := small_packets * small_packet_price + large_packets * large_packet_price
  let remaining_revenue := remaining_cheap * remaining_cheap_price + remaining_expensive * remaining_expensive_price
  let total_revenue := repacked_revenue + remaining_revenue
  let profit := total_revenue - total_cost

  profit = 1766 := by sorry

end NUMINAMATH_CALUDE_owen_profit_l3700_370003


namespace NUMINAMATH_CALUDE_sharon_in_middle_l3700_370088

-- Define the set of people
inductive Person : Type
  | Maren : Person
  | Aaron : Person
  | Sharon : Person
  | Darren : Person
  | Karen : Person

-- Define the seating arrangement as a function from position (1 to 5) to Person
def Seating := Fin 5 → Person

-- Define the constraints
def satisfies_constraints (s : Seating) : Prop :=
  -- Maren sat in the last car
  s 5 = Person.Maren ∧
  -- Aaron sat directly behind Sharon
  (∃ i : Fin 4, s i = Person.Sharon ∧ s (i.succ) = Person.Aaron) ∧
  -- Darren sat directly behind Karen
  (∃ i : Fin 4, s i = Person.Karen ∧ s (i.succ) = Person.Darren) ∧
  -- At least one person sat between Aaron and Maren
  (∃ i j : Fin 5, i < j ∧ j < 5 ∧ s i = Person.Aaron ∧ s j ≠ Person.Maren ∧ s (j+1) = Person.Maren)

-- Theorem: Given the constraints, Sharon must be in the middle car
theorem sharon_in_middle (s : Seating) (h : satisfies_constraints s) : s 3 = Person.Sharon :=
sorry

end NUMINAMATH_CALUDE_sharon_in_middle_l3700_370088


namespace NUMINAMATH_CALUDE_min_value_of_function_l3700_370008

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  (x^2 + x + 1) / (x - 1) ≥ 3 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3700_370008


namespace NUMINAMATH_CALUDE_area_at_stage_5_is_24_l3700_370016

/-- Calculates the length of the rectangle at a given stage -/
def length_at_stage (stage : ℕ) : ℕ := 4 + 2 * (stage - 1)

/-- Calculates the area of the rectangle at a given stage -/
def area_at_stage (stage : ℕ) : ℕ := length_at_stage stage * 2

/-- Theorem stating that the area at Stage 5 is 24 square inches -/
theorem area_at_stage_5_is_24 : area_at_stage 5 = 24 := by
  sorry

#eval area_at_stage 5  -- This should output 24

end NUMINAMATH_CALUDE_area_at_stage_5_is_24_l3700_370016


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3700_370000

/-- The usual time to catch the bus, given that walking at 3/5 of the usual speed results in missing the bus by 5 minutes -/
theorem usual_time_to_catch_bus : ∃ (T : ℝ), T > 0 ∧ (5/3 * T = T + 5) ∧ T = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3700_370000


namespace NUMINAMATH_CALUDE_constant_vertex_l3700_370026

/-- The function f(x) = a^(x-2) + 1 always passes through the point (2, 2) for a > 0 and a ≠ 1 -/
theorem constant_vertex (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_vertex_l3700_370026


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l3700_370095

theorem linear_function_not_in_first_quadrant :
  ∀ x y : ℝ, y = -2 * x - 1 → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l3700_370095


namespace NUMINAMATH_CALUDE_shop_pricing_l3700_370074

theorem shop_pricing (CP : ℝ) 
  (h1 : CP * 0.5 = 320) : CP * 1.25 = 800 := by
  sorry

end NUMINAMATH_CALUDE_shop_pricing_l3700_370074


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3700_370091

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 36 = 0) ↔ (m = 12 * Real.sqrt 3 ∨ m = -12 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3700_370091


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3700_370059

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

theorem vector_sum_magnitude (x : ℝ) 
  (h : vector_a • vector_b x = -3) : 
  ‖vector_a + vector_b x‖ = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3700_370059


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l3700_370044

/-- Prove that Maxwell's walking speed is 4 km/h given the conditions of the problem -/
theorem maxwell_walking_speed :
  ∀ (maxwell_speed : ℝ),
    maxwell_speed > 0 →
    (3 * maxwell_speed) + (2 * 6) = 24 →
    maxwell_speed = 4 :=
by
  sorry

#check maxwell_walking_speed

end NUMINAMATH_CALUDE_maxwell_walking_speed_l3700_370044


namespace NUMINAMATH_CALUDE_tetrahedron_altitude_volume_inequality_l3700_370087

/-- A tetrahedron with volume and altitudes -/
structure Tetrahedron where
  volume : ℝ
  altitude : Fin 4 → ℝ

/-- Predicate to check if a tetrahedron is right-angled -/
def isRightAngled (t : Tetrahedron) : Prop := sorry

/-- Theorem stating the relationship between altitudes and volume of a tetrahedron -/
theorem tetrahedron_altitude_volume_inequality (t : Tetrahedron) :
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    t.altitude i * t.altitude j * t.altitude k ≤ 6 * t.volume ∧
    (t.altitude i * t.altitude j * t.altitude k = 6 * t.volume ↔ isRightAngled t) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_altitude_volume_inequality_l3700_370087


namespace NUMINAMATH_CALUDE_fifteen_to_binary_l3700_370086

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem fifteen_to_binary :
  decimal_to_binary 15 = [1, 1, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_fifteen_to_binary_l3700_370086


namespace NUMINAMATH_CALUDE_diagonal_passes_at_least_length_squares_l3700_370090

/-- Represents an irregular hexagon composed of unit squares -/
structure IrregularHexagon where
  total_squares : ℕ
  length : ℕ
  width1 : ℕ
  width2 : ℕ

/-- The minimum number of squares a diagonal passes through -/
def diagonal_squares_count (h : IrregularHexagon) : ℕ :=
  h.length

/-- Theorem stating that the diagonal passes through at least as many squares as the length -/
theorem diagonal_passes_at_least_length_squares (h : IrregularHexagon)
  (h_total : h.total_squares = 78)
  (h_length : h.length = 12)
  (h_width1 : h.width1 = 8)
  (h_width2 : h.width2 = 6) :
  diagonal_squares_count h ≥ h.length :=
sorry

end NUMINAMATH_CALUDE_diagonal_passes_at_least_length_squares_l3700_370090


namespace NUMINAMATH_CALUDE_square_division_negative_numbers_l3700_370067

theorem square_division_negative_numbers : (-128)^2 / (-64)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_division_negative_numbers_l3700_370067


namespace NUMINAMATH_CALUDE_relay_race_distance_l3700_370053

theorem relay_race_distance (total_distance : ℕ) (team_members : ℕ) (individual_distance : ℕ) : 
  total_distance = 150 ∧ team_members = 5 ∧ individual_distance * team_members = total_distance →
  individual_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_l3700_370053


namespace NUMINAMATH_CALUDE_martha_troubleshooting_time_l3700_370069

/-- The total time Martha spent on router troubleshooting activities -/
def total_time (router_time hold_time yelling_time : ℕ) : ℕ :=
  router_time + hold_time + yelling_time

/-- Theorem stating the total time Martha spent on activities -/
theorem martha_troubleshooting_time :
  ∃ (router_time hold_time yelling_time : ℕ),
    router_time = 10 ∧
    hold_time = 6 * router_time ∧
    yelling_time = hold_time / 2 ∧
    total_time router_time hold_time yelling_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_martha_troubleshooting_time_l3700_370069


namespace NUMINAMATH_CALUDE_greg_sisters_count_l3700_370066

def number_of_sisters (total_bars : ℕ) (days_in_week : ℕ) (traded_bars : ℕ) (bars_per_sister : ℕ) : ℕ :=
  (total_bars - days_in_week - traded_bars) / bars_per_sister

theorem greg_sisters_count :
  let total_bars : ℕ := 20
  let days_in_week : ℕ := 7
  let traded_bars : ℕ := 3
  let bars_per_sister : ℕ := 5
  number_of_sisters total_bars days_in_week traded_bars bars_per_sister = 2 := by
  sorry

end NUMINAMATH_CALUDE_greg_sisters_count_l3700_370066


namespace NUMINAMATH_CALUDE_dans_remaining_green_marbles_l3700_370064

def dans_initial_green_marbles : ℕ := 32
def mikes_taken_green_marbles : ℕ := 23

theorem dans_remaining_green_marbles :
  dans_initial_green_marbles - mikes_taken_green_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_green_marbles_l3700_370064


namespace NUMINAMATH_CALUDE_power_multiplication_l3700_370043

theorem power_multiplication (a b : ℕ) : (10 : ℕ) ^ 85 * (10 : ℕ) ^ 84 = (10 : ℕ) ^ (85 + 84) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3700_370043


namespace NUMINAMATH_CALUDE_emilia_berry_cobbler_l3700_370085

/-- The number of cartons of berries needed for Emilia's berry cobbler -/
def total_cartons (strawberry_cartons blueberry_cartons additional_cartons : ℕ) : ℕ :=
  strawberry_cartons + blueberry_cartons + additional_cartons

/-- Theorem stating that the total number of cartons is 42 given the specific quantities -/
theorem emilia_berry_cobbler : total_cartons 2 7 33 = 42 := by
  sorry

end NUMINAMATH_CALUDE_emilia_berry_cobbler_l3700_370085


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l3700_370076

theorem unique_number_with_conditions : ∃! N : ℤ,
  35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l3700_370076


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3700_370080

theorem cube_volume_problem (s : ℝ) : 
  s > 0 → 
  (s + 2) * (s - 3) * s - s^3 = 26 → 
  s^3 = 343 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3700_370080


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3700_370018

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3700_370018


namespace NUMINAMATH_CALUDE_factor_tree_problem_l3700_370051

theorem factor_tree_problem (X Y Z F G : ℕ) :
  X = Y * Z ∧
  Y = 5 * F ∧
  Z = 7 * G ∧
  F = 5 * 3 ∧
  G = 7 * 3 →
  X = 11025 :=
by sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l3700_370051


namespace NUMINAMATH_CALUDE_sin_5pi_6_minus_2alpha_l3700_370042

theorem sin_5pi_6_minus_2alpha (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_5pi_6_minus_2alpha_l3700_370042


namespace NUMINAMATH_CALUDE_andrews_sleepover_donuts_l3700_370073

/-- The number of donuts Andrew's mother needs to buy for a sleepover --/
def donuts_for_sleepover (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts_per_friend : ℕ) : ℕ :=
  let total_friends := initial_friends + additional_friends
  let donuts_for_friends := total_friends * (donuts_per_friend + extra_donuts_per_friend)
  let donuts_for_andrew := donuts_per_friend + extra_donuts_per_friend
  donuts_for_friends + donuts_for_andrew

/-- Theorem: Andrew's mother needs to buy 20 donuts for the sleepover --/
theorem andrews_sleepover_donuts :
  donuts_for_sleepover 2 2 3 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_andrews_sleepover_donuts_l3700_370073


namespace NUMINAMATH_CALUDE_books_redistribution_l3700_370036

theorem books_redistribution (mark_initial : ℕ) (alice_initial : ℕ) (books_given : ℕ) : 
  mark_initial = 105 →
  alice_initial = 15 →
  books_given = 15 →
  mark_initial - books_given = 3 * (alice_initial + books_given) :=
by
  sorry

end NUMINAMATH_CALUDE_books_redistribution_l3700_370036


namespace NUMINAMATH_CALUDE_fuji_fraction_l3700_370096

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.crossPollinated = o.total / 10 ∧
  o.fuji + o.crossPollinated = 170 ∧
  o.gala = 30 ∧
  o.total = o.fuji + o.gala + o.crossPollinated

/-- The theorem stating that 3/4 of the trees are pure Fuji -/
theorem fuji_fraction (o : Orchard) (h : orchard_conditions o) : 
  o.fuji = 3 * o.total / 4 := by
  sorry

#check fuji_fraction

end NUMINAMATH_CALUDE_fuji_fraction_l3700_370096


namespace NUMINAMATH_CALUDE_expression_evaluation_l3700_370033

theorem expression_evaluation : 
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) + 
  Real.sqrt (7 + 2 * Real.sqrt 10) - Real.sqrt (7 - 2 * Real.sqrt 10) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3700_370033


namespace NUMINAMATH_CALUDE_roll_distribution_probability_l3700_370015

/-- The number of guests -/
def num_guests : ℕ := 4

/-- The number of roll types -/
def num_roll_types : ℕ := 4

/-- The total number of rolls -/
def total_rolls : ℕ := 16

/-- The number of rolls per type -/
def rolls_per_type : ℕ := 4

/-- The number of rolls given to each guest -/
def rolls_per_guest : ℕ := 4

/-- The probability of each guest receiving one of each type of roll -/
def probability_each_guest_gets_one_of_each : ℚ := 1 / 6028032000

theorem roll_distribution_probability :
  probability_each_guest_gets_one_of_each = 
    (rolls_per_type / total_rolls) *
    ((rolls_per_type - 1) / (total_rolls - 1)) *
    ((rolls_per_type - 2) / (total_rolls - 2)) *
    ((rolls_per_type - 3) / (total_rolls - 3)) *
    ((rolls_per_type - 1) / (total_rolls - 4)) *
    ((rolls_per_type - 2) / (total_rolls - 5)) *
    ((rolls_per_type - 3) / (total_rolls - 6)) *
    ((rolls_per_type - 2) / (total_rolls - 8)) *
    ((rolls_per_type - 3) / (total_rolls - 9)) *
    ((rolls_per_type - 3) / (total_rolls - 12)) := by
  sorry

#eval probability_each_guest_gets_one_of_each

end NUMINAMATH_CALUDE_roll_distribution_probability_l3700_370015


namespace NUMINAMATH_CALUDE_jack_driving_distance_l3700_370007

/-- Calculates the number of miles driven every four months given the total years of driving and total miles driven. -/
def miles_per_four_months (years : ℕ) (total_miles : ℕ) : ℕ :=
  total_miles / (years * 3)

/-- Theorem stating that driving for 9 years and covering 999,000 miles results in driving 37,000 miles every four months. -/
theorem jack_driving_distance :
  miles_per_four_months 9 999000 = 37000 := by
  sorry

end NUMINAMATH_CALUDE_jack_driving_distance_l3700_370007


namespace NUMINAMATH_CALUDE_negation_of_square_nonnegative_l3700_370039

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x ^ 2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀ ^ 2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_square_nonnegative_l3700_370039
