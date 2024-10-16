import Mathlib

namespace NUMINAMATH_CALUDE_max_non_managers_l481_48108

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 36 →
  ∃ (max_non_managers : ℕ), max_non_managers = 36 ∧ 
    ∀ n : ℕ, n > max_non_managers → (managers : ℚ) / n ≤ 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_max_non_managers_l481_48108


namespace NUMINAMATH_CALUDE_fifth_color_count_l481_48190

/-- Represents the number of marbles of each color in a box -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  fifth : ℕ

/-- Defines the properties of the marble counts as given in the problem -/
def valid_marble_count (m : MarbleCount) : Prop :=
  m.red = 25 ∧
  m.green = 3 * m.red ∧
  m.yellow = m.green / 5 ∧
  m.blue = 2 * m.yellow ∧
  m.fifth = (m.red + m.blue) + (m.red + m.blue) / 2 ∧
  m.red + m.green + m.yellow + m.blue + m.fifth = 4 * m.green

theorem fifth_color_count (m : MarbleCount) (h : valid_marble_count m) : m.fifth = 155 := by
  sorry

end NUMINAMATH_CALUDE_fifth_color_count_l481_48190


namespace NUMINAMATH_CALUDE_add_five_sixteen_base7_l481_48153

/-- Converts a base 7 number to decimal --/
def toDecimal (b₇ : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (d : ℕ) : ℕ := sorry

/-- Addition in base 7 --/
def addBase7 (a₇ b₇ : ℕ) : ℕ := 
  toBase7 (toDecimal a₇ + toDecimal b₇)

theorem add_five_sixteen_base7 : 
  addBase7 5 16 = 24 := by sorry

end NUMINAMATH_CALUDE_add_five_sixteen_base7_l481_48153


namespace NUMINAMATH_CALUDE_floor_sum_example_l481_48147

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_example_l481_48147


namespace NUMINAMATH_CALUDE_closed_polyline_theorem_l481_48183

/-- Represents a rectangle on a unit grid --/
structure Rectangle where
  m : ℕ  -- Width
  n : ℕ  -- Height

/-- Determines if a closed polyline exists for a given rectangle --/
def closedPolylineExists (rect : Rectangle) : Prop :=
  Odd rect.m ∨ Odd rect.n

/-- Calculates the length of the closed polyline if it exists --/
def polylineLength (rect : Rectangle) : ℕ :=
  (rect.n + 1) * (rect.m + 1)

/-- Main theorem about the existence and length of the closed polyline --/
theorem closed_polyline_theorem (rect : Rectangle) :
  closedPolylineExists rect ↔ 
    ∃ (length : ℕ), length = polylineLength rect ∧ 
      (∀ (i j : ℕ), i ≤ rect.m ∧ j ≤ rect.n → 
        ∃ (unique_visit : Prop), unique_visit) :=
by sorry

end NUMINAMATH_CALUDE_closed_polyline_theorem_l481_48183


namespace NUMINAMATH_CALUDE_football_shaped_area_l481_48127

/-- The area of two quarter-circle sectors minus two right triangles in a square with side length 4 -/
theorem football_shaped_area (π : ℝ) (h_π : π = Real.pi) : 
  let side_length : ℝ := 4
  let diagonal : ℝ := side_length * Real.sqrt 2
  let quarter_circle_area : ℝ := (π * diagonal^2) / 4
  let triangle_area : ℝ := side_length^2 / 2
  2 * (quarter_circle_area - triangle_area) = 16 * π - 16 := by
sorry

end NUMINAMATH_CALUDE_football_shaped_area_l481_48127


namespace NUMINAMATH_CALUDE_one_quarter_of_seven_point_two_l481_48145

theorem one_quarter_of_seven_point_two : 
  (7.2 : ℚ) / 4 = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_one_quarter_of_seven_point_two_l481_48145


namespace NUMINAMATH_CALUDE_vector_sum_equality_l481_48151

theorem vector_sum_equality (a b c : ℝ × ℝ) :
  a = (1, -1) →
  b = (-1, 1) →
  c = (5, 1) →
  c + a + b = c := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l481_48151


namespace NUMINAMATH_CALUDE_sum_of_squared_fractions_l481_48110

theorem sum_of_squared_fractions (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_fractions_l481_48110


namespace NUMINAMATH_CALUDE_equation_satisfied_l481_48194

theorem equation_satisfied (a b c : ℤ) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 3 ↔ (a = c + 1 ∧ b - 1 = a) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_l481_48194


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l481_48187

/-- A cyclic quadrilateral with side lengths a, b, c, d, area Q, and circumradius R -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  Q : ℝ
  R : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ Q > 0 ∧ R > 0

/-- The main theorem about cyclic quadrilaterals -/
theorem cyclic_quadrilateral_theorem (ABCD : CyclicQuadrilateral) :
  ABCD.R^2 = ((ABCD.a * ABCD.b + ABCD.c * ABCD.d) * (ABCD.a * ABCD.c + ABCD.b * ABCD.d) * (ABCD.a * ABCD.d + ABCD.b * ABCD.c)) / (16 * ABCD.Q^2) ∧
  ABCD.R ≥ ((ABCD.a * ABCD.b * ABCD.c * ABCD.d)^(3/4)) / (ABCD.Q * Real.sqrt 2) ∧
  (ABCD.R = ((ABCD.a * ABCD.b * ABCD.c * ABCD.d)^(3/4)) / (ABCD.Q * Real.sqrt 2) ↔ ABCD.a = ABCD.b ∧ ABCD.b = ABCD.c ∧ ABCD.c = ABCD.d) :=
by sorry


end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l481_48187


namespace NUMINAMATH_CALUDE_exists_double_application_square_l481_48105

theorem exists_double_application_square :
  ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n^2 := by sorry

end NUMINAMATH_CALUDE_exists_double_application_square_l481_48105


namespace NUMINAMATH_CALUDE_four_points_left_of_origin_l481_48166

theorem four_points_left_of_origin : 
  let points : List ℝ := [-(-8), (-1)^2023, -(3^2), -1-11, -2/5]
  (points.filter (· < 0)).length = 4 := by
sorry

end NUMINAMATH_CALUDE_four_points_left_of_origin_l481_48166


namespace NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l481_48165

def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k > 0 ∧ n = (k * (2 * a + k - 1)) / 2

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (s : ℕ), n = 2^s

theorem sum_of_consecutive_iff_not_power_of_two (n : ℕ) :
  ¬(is_sum_of_consecutive n) ↔ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l481_48165


namespace NUMINAMATH_CALUDE_fraction_difference_l481_48149

theorem fraction_difference (a b c d : ℚ) : 
  a = 3/4 ∧ b = 7/8 ∧ c = 13/16 ∧ d = 1/2 →
  max a (max b (max c d)) - min a (min b (min c d)) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l481_48149


namespace NUMINAMATH_CALUDE_jazmin_dolls_count_l481_48131

theorem jazmin_dolls_count (geraldine_dolls : ℝ) (difference : ℕ) :
  geraldine_dolls = 2186.0 →
  difference = 977 →
  geraldine_dolls - difference = 1209 :=
by sorry

end NUMINAMATH_CALUDE_jazmin_dolls_count_l481_48131


namespace NUMINAMATH_CALUDE_base_nine_solution_l481_48103

/-- Convert a list of digits in base b to its decimal representation -/
def toDecimal (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Check if the equation is valid in base b -/
def isValidEquation (b : ℕ) : Prop :=
  toDecimal [5, 7, 4, 2] b + toDecimal [6, 9, 3, 1] b = toDecimal [1, 2, 7, 7, 3] b

theorem base_nine_solution :
  ∃ (b : ℕ), b > 1 ∧ isValidEquation b ∧ ∀ (x : ℕ), x > 1 ∧ x ≠ b → ¬isValidEquation x :=
by sorry

end NUMINAMATH_CALUDE_base_nine_solution_l481_48103


namespace NUMINAMATH_CALUDE_money_sum_l481_48121

/-- Given two people a and b with some amount of money, 
    if 2/3 of a's amount equals 1/2 of b's amount, 
    and b has 484 rupees, then their total amount is 847 rupees. -/
theorem money_sum (a b : ℕ) (h1 : 2 * a = 3 * (b / 2)) (h2 : b = 484) : 
  a + b = 847 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l481_48121


namespace NUMINAMATH_CALUDE_volume_of_sphere_containing_pyramid_l481_48170

/-- Regular triangular pyramid with base on sphere -/
structure RegularTriangularPyramid where
  /-- Base edge length -/
  baseEdge : ℝ
  /-- Volume of the pyramid -/
  volume : ℝ
  /-- Radius of the circumscribed sphere -/
  sphereRadius : ℝ

/-- Theorem: Volume of sphere containing regular triangular pyramid -/
theorem volume_of_sphere_containing_pyramid (p : RegularTriangularPyramid) 
  (h1 : p.baseEdge = 2 * Real.sqrt 3)
  (h2 : p.volume = Real.sqrt 3) :
  (4 / 3) * Real.pi * p.sphereRadius ^ 3 = (20 * Real.sqrt 5 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_containing_pyramid_l481_48170


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l481_48186

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 1 < 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l481_48186


namespace NUMINAMATH_CALUDE_jackson_keeps_120_lollipops_l481_48102

/-- The number of lollipops Jackson keeps for himself -/
def lollipops_kept (apple banana cherry dragon_fruit : ℕ) (friends : ℕ) : ℕ :=
  cherry

/-- Theorem stating that Jackson keeps 120 lollipops for himself -/
theorem jackson_keeps_120_lollipops :
  lollipops_kept 53 62 120 15 13 = 120 := by
  sorry

#eval lollipops_kept 53 62 120 15 13

end NUMINAMATH_CALUDE_jackson_keeps_120_lollipops_l481_48102


namespace NUMINAMATH_CALUDE_visible_yellow_bus_length_l481_48101

/-- Proves that the visible length of the yellow bus is 18 feet --/
theorem visible_yellow_bus_length (red_bus_length green_truck_length yellow_bus_length orange_car_length : ℝ) :
  red_bus_length = 48 →
  red_bus_length = 4 * orange_car_length →
  yellow_bus_length = 3.5 * orange_car_length →
  green_truck_length = 2 * orange_car_length →
  yellow_bus_length - green_truck_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_visible_yellow_bus_length_l481_48101


namespace NUMINAMATH_CALUDE_higher_variance_greater_fluctuations_l481_48143

-- Define the properties of the two data sets
def mean_A : ℝ := 5
def mean_B : ℝ := 5
def variance_A : ℝ := 0.1
def variance_B : ℝ := 0.2

-- Define a function to represent fluctuations based on variance
def fluctuations (variance : ℝ) : ℝ := variance

-- Theorem stating that higher variance implies greater fluctuations
theorem higher_variance_greater_fluctuations :
  variance_A < variance_B →
  fluctuations variance_A < fluctuations variance_B :=
by sorry

end NUMINAMATH_CALUDE_higher_variance_greater_fluctuations_l481_48143


namespace NUMINAMATH_CALUDE_carol_and_alex_peanuts_l481_48171

def peanut_distribution (initial_peanuts : ℕ) (multiplier : ℕ) (num_people : ℕ) : ℕ :=
  (initial_peanuts + initial_peanuts * multiplier) / num_people

theorem carol_and_alex_peanuts :
  peanut_distribution 2 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_carol_and_alex_peanuts_l481_48171


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l481_48148

/-- The function y = (a + 1)x^2 - 2x + 3 is quadratic with respect to x -/
def is_quadratic (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (a + 1) * x^2 - 2 * x + 3

/-- The range of values for a in the quadratic function y = (a + 1)x^2 - 2x + 3 -/
theorem quadratic_function_a_range :
  ∀ a : ℝ, is_quadratic a ↔ a ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l481_48148


namespace NUMINAMATH_CALUDE_symmetric_circle_l481_48142

/-- Given a circle C with equation x^2 + y^2 = 25 and a point of symmetry (-3, 4),
    the symmetric circle has the equation (x + 6)^2 + (y - 8)^2 = 25 -/
theorem symmetric_circle (x y : ℝ) :
  (∀ x y, x^2 + y^2 = 25 → (x + 6)^2 + (y - 8)^2 = 25) ∧
  (∃ x₀ y₀, x₀^2 + y₀^2 = 25 ∧ 
    2 * (-3) = x₀ + (x₀ - 6) ∧
    2 * 4 = y₀ + (y₀ - (-8))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l481_48142


namespace NUMINAMATH_CALUDE_mean_ice_cream_sales_l481_48185

def ice_cream_sales : List ℕ := [100, 92, 109, 96, 103, 96, 105]

theorem mean_ice_cream_sales :
  (ice_cream_sales.sum : ℚ) / ice_cream_sales.length = 100.14 := by
  sorry

end NUMINAMATH_CALUDE_mean_ice_cream_sales_l481_48185


namespace NUMINAMATH_CALUDE_prob_two_green_marbles_l481_48118

/-- The probability of drawing two green marbles without replacement from a bag containing 5 blue marbles and 7 green marbles is 7/22. -/
theorem prob_two_green_marbles (blue_marbles green_marbles : ℕ) 
  (h_blue : blue_marbles = 5) (h_green : green_marbles = 7) :
  let total_marbles := blue_marbles + green_marbles
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_green_marbles_l481_48118


namespace NUMINAMATH_CALUDE_reciprocal_opposite_square_minus_product_l481_48114

theorem reciprocal_opposite_square_minus_product (a b c d : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_square_minus_product_l481_48114


namespace NUMINAMATH_CALUDE_circle_radius_problem_l481_48172

theorem circle_radius_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (π * x^2 = π * y^2) →   -- Circles have the same area
  (2 * π * x = 12 * π) →  -- Circumference of circle x is 12π
  (∃ v, y = 2 * v) →      -- Radius of circle y is twice some value v
  (∃ v, y = 2 * v ∧ v = 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l481_48172


namespace NUMINAMATH_CALUDE_girls_fraction_in_class_l481_48169

theorem girls_fraction_in_class (T G B : ℚ) (h1 : T > 0) (h2 : G > 0) (h3 : B > 0)
  (h4 : T = G + B) (h5 : B / G = 5 / 3) :
  ∃ X : ℚ, X * G = (1 / 4) * T ∧ X = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_girls_fraction_in_class_l481_48169


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1010th_term_l481_48162

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (p r : ℚ) : ℕ → ℚ
| 0 => p / 2
| 1 => 18
| 2 => 2 * p - r
| 3 => 2 * p + r
| n + 4 => arithmetic_sequence p r 3 + (n + 1) * (arithmetic_sequence p r 3 - arithmetic_sequence p r 2)

/-- The 1010th term of the sequence is 72774/11 -/
theorem arithmetic_sequence_1010th_term (p r : ℚ) :
  arithmetic_sequence p r 1009 = 72774 / 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1010th_term_l481_48162


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l481_48135

theorem square_sum_reciprocal (x : ℝ) (h : x + (1/x) = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l481_48135


namespace NUMINAMATH_CALUDE_central_angle_values_l481_48181

/-- A circular sector with given perimeter and area -/
structure CircularSector where
  perimeter : ℝ
  area : ℝ

/-- The central angle of a circular sector in radians -/
def central_angle (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, 
    s.area = 1/2 * r^2 * θ ∧ 
    s.perimeter = 2 * r + r * θ}

/-- Theorem: For a circular sector with perimeter 3 cm and area 1/2 cm², 
    the central angle is either 1 or 4 radians -/
theorem central_angle_values (s : CircularSector) 
  (h_perimeter : s.perimeter = 3)
  (h_area : s.area = 1/2) : 
  central_angle s = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_central_angle_values_l481_48181


namespace NUMINAMATH_CALUDE_circle_symmetry_l481_48133

/-- The equation of a circle -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x - 2*a*y = 0

/-- The line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x + y = 0

/-- Theorem stating that the circle is symmetric with respect to the line x + y = 0 -/
theorem circle_symmetry (a : ℝ) (h : a ≠ 0) :
  ∃ (r : ℝ), r > 0 ∧
  ∀ (x y : ℝ), circle_equation x y a ↔
    ∃ (x₀ y₀ : ℝ), symmetry_line x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l481_48133


namespace NUMINAMATH_CALUDE_derivative_x_minus_inverse_x_l481_48178

open Real

theorem derivative_x_minus_inverse_x (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => x - 1 / x) x = 1 + 1 / x^2 :=
sorry

end NUMINAMATH_CALUDE_derivative_x_minus_inverse_x_l481_48178


namespace NUMINAMATH_CALUDE_complex_product_real_condition_l481_48132

theorem complex_product_real_condition (a b c d : ℝ) :
  (Complex.mk a b * Complex.mk c d).im = 0 ↔ a * d + b * c = 0 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_condition_l481_48132


namespace NUMINAMATH_CALUDE_book_pages_proof_l481_48155

/-- Proves that a book has 500 pages given specific writing and damage conditions -/
theorem book_pages_proof (total_pages : ℕ) : 
  (150 : ℕ) < total_pages →
  (0.8 * 0.7 * (total_pages - 150 : ℕ) : ℝ) = 196 →
  total_pages = 500 := by
sorry

end NUMINAMATH_CALUDE_book_pages_proof_l481_48155


namespace NUMINAMATH_CALUDE_all_quadrilaterals_diagonals_bisect_l481_48150

-- Define a type for quadrilaterals
inductive Quadrilateral
  | Parallelogram
  | Rectangle
  | Rhombus
  | Square

-- Define a function to check if diagonals bisect each other
def diagonalsBisectEachOther (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Parallelogram => true
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Rhombus => true
  | Quadrilateral.Square => true

-- Theorem statement
theorem all_quadrilaterals_diagonals_bisect :
  ∀ q : Quadrilateral, diagonalsBisectEachOther q := by
  sorry

end NUMINAMATH_CALUDE_all_quadrilaterals_diagonals_bisect_l481_48150


namespace NUMINAMATH_CALUDE_extreme_point_property_g_maximum_bound_l481_48109

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x - b

-- Define the function g
def g (a b x : ℝ) : ℝ := |f a b x|

theorem extreme_point_property (a b x₀ x₁ : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f a b x₀ ≤ f a b x) →
  f a b x₁ = f a b x₀ →
  x₁ ≠ x₀ →
  x₁ + 2*x₀ = 0 := by sorry

theorem g_maximum_bound (a b : ℝ) (ha : a > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, g a b x ≥ (1/4 : ℝ) ∧ g a b x ≥ g a b y := by sorry

end NUMINAMATH_CALUDE_extreme_point_property_g_maximum_bound_l481_48109


namespace NUMINAMATH_CALUDE_petyas_coins_l481_48117

/-- Represents the denominations of coins --/
inductive Coin
  | OneRuble
  | TwoRubles
  | Other

/-- Represents Petya's pocket of coins --/
structure Pocket where
  coins : List Coin

/-- Checks if a list of coins contains at least one 1 ruble coin --/
def hasOneRuble (coins : List Coin) : Prop :=
  Coin.OneRuble ∈ coins

/-- Checks if a list of coins contains at least one 2 rubles coin --/
def hasTwoRubles (coins : List Coin) : Prop :=
  Coin.TwoRubles ∈ coins

/-- The main theorem to prove --/
theorem petyas_coins (p : Pocket) :
  (∀ (subset : List Coin), subset ⊆ p.coins → subset.length = 3 → hasOneRuble subset) →
  (∀ (subset : List Coin), subset ⊆ p.coins → subset.length = 4 → hasTwoRubles subset) →
  p.coins.length = 5 →
  p.coins = [Coin.OneRuble, Coin.OneRuble, Coin.OneRuble, Coin.TwoRubles, Coin.TwoRubles] :=
by sorry

end NUMINAMATH_CALUDE_petyas_coins_l481_48117


namespace NUMINAMATH_CALUDE_workers_paid_four_fifties_is_31_l481_48199

/-- Represents the payment structure for workers -/
structure PaymentStructure where
  total_workers : Nat
  payment_per_worker : Nat
  hundred_bills : Nat
  fifty_bills : Nat
  workers_paid_two_hundreds : Nat

/-- Calculates the number of workers paid with four $50 bills -/
def workers_paid_four_fifties (p : PaymentStructure) : Nat :=
  let remaining_hundreds := p.hundred_bills - 2 * p.workers_paid_two_hundreds
  let workers_paid_mixed := remaining_hundreds
  let fifties_for_mixed := 2 * workers_paid_mixed
  let remaining_fifties := p.fifty_bills - fifties_for_mixed
  remaining_fifties / 4

/-- Theorem stating that given the specific payment structure, 31 workers are paid with four $50 bills -/
theorem workers_paid_four_fifties_is_31 :
  let p : PaymentStructure := {
    total_workers := 108,
    payment_per_worker := 200,
    hundred_bills := 122,
    fifty_bills := 188,
    workers_paid_two_hundreds := 45
  }
  workers_paid_four_fifties p = 31 := by
  sorry

end NUMINAMATH_CALUDE_workers_paid_four_fifties_is_31_l481_48199


namespace NUMINAMATH_CALUDE_smallest_y_value_l481_48157

theorem smallest_y_value (x y : ℝ) 
  (h1 : 2 < x ∧ x < y)
  (h2 : 2 + x ≤ y)
  (h3 : 1 / x + 1 / y ≤ 1) :
  y ≥ 2 + Real.sqrt 2 ∧ ∀ z, (2 < z ∧ 2 + z ≤ y ∧ 1 / z + 1 / y ≤ 1) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l481_48157


namespace NUMINAMATH_CALUDE_second_number_value_l481_48141

theorem second_number_value (A B C : ℚ) 
  (sum_eq : A + B + C = 98)
  (ratio_AB : A = (2/3) * B)
  (ratio_BC : C = (8/5) * B) : 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l481_48141


namespace NUMINAMATH_CALUDE_arrangement_theorem_l481_48126

/-- The number of ways to arrange 5 people in 5 seats with exactly 2 matching --/
def arrangement_count : ℕ := 20

/-- The number of ways to choose 2 items from 5 --/
def choose_two_from_five : ℕ := 10

/-- The number of ways to arrange the remaining 3 people --/
def arrange_remaining : ℕ := 2

theorem arrangement_theorem : 
  arrangement_count = choose_two_from_five * arrange_remaining := by
  sorry


end NUMINAMATH_CALUDE_arrangement_theorem_l481_48126


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l481_48106

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x + 5| = 3 * x - 1 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l481_48106


namespace NUMINAMATH_CALUDE_cell_phone_customers_l481_48193

theorem cell_phone_customers (us_customers other_customers : ℕ) 
  (h1 : us_customers = 723)
  (h2 : other_customers = 6699) :
  us_customers + other_customers = 7422 := by
sorry

end NUMINAMATH_CALUDE_cell_phone_customers_l481_48193


namespace NUMINAMATH_CALUDE_total_peaches_l481_48130

/-- The number of baskets -/
def num_baskets : ℕ := 11

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 10

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 18

/-- Theorem: The total number of peaches in all baskets is 308 -/
theorem total_peaches :
  (num_baskets * (red_peaches_per_basket + green_peaches_per_basket)) = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l481_48130


namespace NUMINAMATH_CALUDE_x_value_l481_48176

def x : ℚ := (320 / 2) / 3

theorem x_value : x = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l481_48176


namespace NUMINAMATH_CALUDE_rectangle_area_l481_48125

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) 
  (h1 : square_area = 900)
  (h2 : rectangle_width = 10)
  (h3 : ∃ (circle_radius : ℝ), circle_radius = Real.sqrt square_area)
  (h4 : ∃ (rectangle_length : ℝ), rectangle_length = (2/5) * Real.sqrt square_area) :
  ∃ (rectangle_area : ℝ), rectangle_area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l481_48125


namespace NUMINAMATH_CALUDE_rod_cutting_l481_48113

/-- Given a rod of length 42.5 meters that can be cut into 50 equal pieces,
    prove that the length of each piece is 0.85 meters. -/
theorem rod_cutting (rod_length : Real) (num_pieces : Nat) (piece_length : Real) 
    (h1 : rod_length = 42.5)
    (h2 : num_pieces = 50)
    (h3 : piece_length * num_pieces = rod_length) : 
  piece_length = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l481_48113


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l481_48184

theorem cos_alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.cos (α + π/3) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l481_48184


namespace NUMINAMATH_CALUDE_golden_apples_weight_l481_48120

theorem golden_apples_weight (kidney_apples : ℕ) (canada_apples : ℕ) (sold_apples : ℕ) (left_apples : ℕ) 
  (h1 : kidney_apples = 23)
  (h2 : canada_apples = 14)
  (h3 : sold_apples = 36)
  (h4 : left_apples = 38) :
  ∃ golden_apples : ℕ, 
    kidney_apples + golden_apples + canada_apples = sold_apples + left_apples ∧ 
    golden_apples = 37 := by
  sorry

end NUMINAMATH_CALUDE_golden_apples_weight_l481_48120


namespace NUMINAMATH_CALUDE_rectangle_length_l481_48158

/-- Represents a rectangle with perimeter P, width W, length L, and area A. -/
structure Rectangle where
  P : ℝ  -- Perimeter
  W : ℝ  -- Width
  L : ℝ  -- Length
  A : ℝ  -- Area
  h1 : P = 2 * (L + W)  -- Perimeter formula
  h2 : A = L * W        -- Area formula
  h3 : P / W = 5        -- Given ratio
  h4 : A = 150          -- Given area

/-- Proves that a rectangle with the given properties has a length of 15. -/
theorem rectangle_length (rect : Rectangle) : rect.L = 15 := by
  sorry

#check rectangle_length

end NUMINAMATH_CALUDE_rectangle_length_l481_48158


namespace NUMINAMATH_CALUDE_pie_slices_served_yesterday_l481_48173

def slices_served_yesterday (lunch_today dinner_today total_today : ℕ) : ℕ :=
  total_today - (lunch_today + dinner_today)

theorem pie_slices_served_yesterday : 
  slices_served_yesterday 7 5 12 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_pie_slices_served_yesterday_l481_48173


namespace NUMINAMATH_CALUDE_triangular_pyramid_base_balls_l481_48156

/-- The number of balls in a triangular pyramid with n rows -/
def triangular_pyramid_balls (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of balls in the base of a triangular pyramid with n rows -/
def base_balls (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: In a regular triangular pyramid with 165 tightly packed identical balls,
    the number of balls in the base is 45 -/
theorem triangular_pyramid_base_balls :
  ∃ n : ℕ, triangular_pyramid_balls n = 165 ∧ base_balls n = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_base_balls_l481_48156


namespace NUMINAMATH_CALUDE_f_value_at_2_l481_48195

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 4

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 2 → f a b 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l481_48195


namespace NUMINAMATH_CALUDE_remainder_987_pow_654_mod_13_l481_48144

theorem remainder_987_pow_654_mod_13 : 987^654 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987_pow_654_mod_13_l481_48144


namespace NUMINAMATH_CALUDE_parabola_sum_l481_48159

/-- Represents a parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-6) = 7 → p.x_coord (-4) = 5 → p.a + p.b + p.c = -42 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l481_48159


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l481_48164

theorem rectangular_plot_breadth :
  ∀ (length breadth area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 2028 →
  breadth = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l481_48164


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l481_48128

theorem algebraic_expression_symmetry (a b c : ℝ) : 
  a * (-5)^4 + b * (-5)^2 + c = 3 → a * 5^4 + b * 5^2 + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l481_48128


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l481_48104

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - (2 * Q.quadrilateral_faces)

/-- Theorem: A convex polyhedron Q with 30 vertices, 70 edges, 42 faces
    (30 triangular and 12 quadrilateral) has 341 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l481_48104


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l481_48191

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a*b + b*c + c*a = 50) : 
  a + b + c = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l481_48191


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l481_48138

-- Problem 1
theorem problem_1 (x : ℝ) : (-2 * x^2)^3 + 4 * x^3 * x^3 = -4 * x^6 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3 * x^2 - x + 1) * (-4 * x) = -12 * x^3 + 4 * x^2 - 4 * x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l481_48138


namespace NUMINAMATH_CALUDE_cube_sum_equals_110_l481_48168

theorem cube_sum_equals_110 (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_110_l481_48168


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l481_48119

theorem consecutive_integers_product_sum : ∃ (n : ℕ), 
  n > 0 ∧ n * (n + 1) = 1080 ∧ n + (n + 1) = 65 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l481_48119


namespace NUMINAMATH_CALUDE_student_616_selected_l481_48100

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  first_selected : ℕ

/-- Checks if a student number is selected in the systematic sampling -/
def is_selected (s : SystematicSampling) (student : ℕ) : Prop :=
  ∃ k : ℕ, student = s.first_selected + k * (s.population / s.sample_size)

theorem student_616_selected (s : SystematicSampling)
  (h_pop : s.population = 1000)
  (h_sample : s.sample_size = 100)
  (h_46_selected : is_selected s 46) :
  is_selected s 616 := by
sorry

end NUMINAMATH_CALUDE_student_616_selected_l481_48100


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l481_48146

theorem largest_constant_inequality :
  ∃ (C : ℝ), (C = 2 / Real.sqrt 3) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + 2*z^2 + 1 ≥ C*(x + y + z)) ∧
  (∀ (C' : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + 2*z^2 + 1 ≥ C'*(x + y + z)) → C' ≤ C) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l481_48146


namespace NUMINAMATH_CALUDE_polynomial_real_root_l481_48116

theorem polynomial_real_root 
  (P : ℝ → ℝ) 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_nonzero : a₁ * a₂ * a₃ ≠ 0)
  (h_poly : ∀ x : ℝ, P (a₁ * x + b₁) + P (a₂ * x + b₂) = P (a₃ * x + b₃)) :
  ∃ r : ℝ, P r = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l481_48116


namespace NUMINAMATH_CALUDE_evaluate_expression_l481_48111

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l481_48111


namespace NUMINAMATH_CALUDE_geese_remaining_l481_48152

theorem geese_remaining (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 51 → flew_away = 28 → remaining = initial - flew_away → remaining = 23 := by
  sorry

end NUMINAMATH_CALUDE_geese_remaining_l481_48152


namespace NUMINAMATH_CALUDE_three_lines_equidistant_l481_48136

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- Distance between a point and a line --/
def distance_point_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

theorem three_lines_equidistant (A B : ℝ × ℝ) (h : dist A B = 5) :
  ∃! (s : Finset Line), s.card = 3 ∧ 
    (∀ l ∈ s, distance_point_line A l = 2 ∧ distance_point_line B l = 3) :=
sorry

end NUMINAMATH_CALUDE_three_lines_equidistant_l481_48136


namespace NUMINAMATH_CALUDE_tony_squat_weight_l481_48192

/-- Represents Tony's weight lifting capabilities -/
structure WeightLifter where
  curl_weight : ℕ
  military_press_multiplier : ℕ
  squat_multiplier : ℕ

/-- Calculates the weight Tony can lift in the squat exercise -/
def squat_weight (lifter : WeightLifter) : ℕ :=
  lifter.curl_weight * lifter.military_press_multiplier * lifter.squat_multiplier

/-- Theorem: Tony can lift 900 pounds in the squat exercise -/
theorem tony_squat_weight :
  ∃ (tony : WeightLifter),
    tony.curl_weight = 90 ∧
    tony.military_press_multiplier = 2 ∧
    tony.squat_multiplier = 5 ∧
    squat_weight tony = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_squat_weight_l481_48192


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l481_48140

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 2) → 
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) → 
  z₁ * z₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l481_48140


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l481_48189

/-- A regular octagon with a square constructed outward on one side -/
structure RegularOctagonWithSquare where
  /-- The vertices of the octagon -/
  vertices : Fin 8 → ℝ × ℝ
  /-- The square constructed outward on one side -/
  square : Fin 4 → ℝ × ℝ
  /-- The octagon is regular -/
  regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8))
  /-- The square is connected to the octagon -/
  square_connected : ∃ i : Fin 8, square 0 = vertices i ∧ square 1 = vertices ((i + 1) % 8)

/-- Point B where two diagonals intersect inside the octagon -/
def intersection_point (o : RegularOctagonWithSquare) : ℝ × ℝ := sorry

/-- Angle ABC in the octagon -/
def angle_ABC (o : RegularOctagonWithSquare) : ℝ := sorry

/-- Theorem: The measure of angle ABC is 22.5° -/
theorem angle_ABC_measure (o : RegularOctagonWithSquare) : angle_ABC o = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l481_48189


namespace NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l481_48175

theorem largest_m_satisfying_inequality :
  ∀ m : ℕ, (1 : ℚ) / 4 + (m : ℚ) / 6 < 3 / 2 ↔ m ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l481_48175


namespace NUMINAMATH_CALUDE_toms_final_amount_l481_48197

/-- Calculates the final amount of money Tom has after washing cars -/
def final_amount (initial_amount : ℝ) (supply_percentage : ℝ) (total_earnings : ℝ) (earnings_percentage : ℝ) : ℝ :=
  let amount_after_supplies := initial_amount * (1 - supply_percentage)
  let earnings := total_earnings * earnings_percentage
  amount_after_supplies + earnings

/-- Theorem stating that Tom's final amount is 114.5 dollars -/
theorem toms_final_amount :
  final_amount 74 0.15 86 0.6 = 114.5 := by
  sorry

end NUMINAMATH_CALUDE_toms_final_amount_l481_48197


namespace NUMINAMATH_CALUDE_markov_equation_solution_l481_48167

theorem markov_equation_solution (m n p : ℕ) : 
  m^2 + n^2 + p^2 = m * n * p → 
  ∃ m₁ n₁ p₁ : ℕ, m = 3 * m₁ ∧ n = 3 * n₁ ∧ p = 3 * p₁ ∧ 
  m₁^2 + n₁^2 + p₁^2 = 3 * m₁ * n₁ * p₁ := by
sorry

end NUMINAMATH_CALUDE_markov_equation_solution_l481_48167


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l481_48179

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧
  (a * (a + 2) = -1) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l481_48179


namespace NUMINAMATH_CALUDE_star_drawing_probability_l481_48174

def total_stars : ℕ := 12
def red_stars : ℕ := 3
def gold_stars : ℕ := 4
def silver_stars : ℕ := 5
def stars_drawn : ℕ := 6

theorem star_drawing_probability : 
  (red_stars / total_stars) * 
  (Nat.choose gold_stars 3 * Nat.choose silver_stars 2) / 
  (Nat.choose (total_stars - 1) (stars_drawn - 1)) = 5 / 231 := by
  sorry

end NUMINAMATH_CALUDE_star_drawing_probability_l481_48174


namespace NUMINAMATH_CALUDE_double_reflection_of_H_l481_48137

-- Define the point type
def Point := ℝ × ℝ

-- Define the parallelogram
def E : Point := (3, 6)
def F : Point := (5, 10)
def G : Point := (7, 6)
def H : Point := (5, 2)

-- Define reflection across x-axis
def reflect_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Define reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : Point) : Point :=
  (p.2 - 2, p.1 + 2)

-- Define the composition of the two reflections
def double_reflection (p : Point) : Point :=
  reflect_y_eq_x_plus_2 (reflect_x_axis p)

-- Theorem statement
theorem double_reflection_of_H :
  double_reflection H = (-4, 7) := by sorry

end NUMINAMATH_CALUDE_double_reflection_of_H_l481_48137


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_three_l481_48129

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that i is the imaginary unit, if the complex number z=(m^2+2m-3)+(m-1)i
    is a pure imaginary number, then m = -3. -/
theorem pure_imaginary_implies_m_eq_neg_three (m : ℝ) :
  IsPureImaginary (Complex.mk (m^2 + 2*m - 3) (m - 1)) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_three_l481_48129


namespace NUMINAMATH_CALUDE_work_completion_proof_l481_48161

/-- The number of days A takes to complete the work alone -/
def a_days : ℕ := 45

/-- The number of days B takes to complete the work alone -/
def b_days : ℕ := 40

/-- The number of days B takes to complete the remaining work after A leaves -/
def b_remaining_days : ℕ := 23

/-- The number of days A works before leaving -/
def x : ℕ := 9

theorem work_completion_proof :
  let total_work := 1
  let a_rate := total_work / a_days
  let b_rate := total_work / b_days
  x * (a_rate + b_rate) + b_remaining_days * b_rate = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l481_48161


namespace NUMINAMATH_CALUDE_circle_area_from_points_l481_48107

/-- The area of a circle with diameter endpoints at A(-1,3) and B'(13,12) is 277π/4 square units. -/
theorem circle_area_from_points :
  let A : ℝ × ℝ := (-1, 3)
  let B' : ℝ × ℝ := (13, 12)
  let diameter := Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2)
  let radius := diameter / 2
  let area := π * radius^2
  area = 277 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_from_points_l481_48107


namespace NUMINAMATH_CALUDE_bike_ride_time_l481_48198

/-- Given a consistent bike riding speed where 1 mile takes 4 minutes,
    prove that the time required to ride 4.5 miles is 18 minutes. -/
theorem bike_ride_time (speed : ℝ) (distance_to_park : ℝ) : 
  speed = 1 / 4 →  -- Speed in miles per minute
  distance_to_park = 4.5 → -- Distance to park in miles
  distance_to_park / speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_time_l481_48198


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l481_48139

theorem arctan_sum_equals_pi_over_four :
  ∃ n : ℕ+, 
    Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/(n : ℝ)) = π/4 ∧
    n = 56 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l481_48139


namespace NUMINAMATH_CALUDE_helios_population_2060_l481_48177

/-- The population growth function for Helios -/
def helios_population (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (2 ^ (years_passed / 20))

/-- Theorem stating the population of Helios in 2060 -/
theorem helios_population_2060 :
  helios_population 250 60 = 2000 := by
  sorry

#eval helios_population 250 60

end NUMINAMATH_CALUDE_helios_population_2060_l481_48177


namespace NUMINAMATH_CALUDE_line_erased_length_l481_48163

theorem line_erased_length (initial_length : ℝ) (final_length : ℝ) (erased_length : ℝ) : 
  initial_length = 1 →
  final_length = 0.67 →
  erased_length = initial_length * 100 - final_length * 100 →
  erased_length = 33 := by
sorry

end NUMINAMATH_CALUDE_line_erased_length_l481_48163


namespace NUMINAMATH_CALUDE_total_weight_of_balls_l481_48115

theorem total_weight_of_balls (blue_weight brown_weight : ℝ) :
  blue_weight = 6 → brown_weight = 3.12 →
  blue_weight + brown_weight = 9.12 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_balls_l481_48115


namespace NUMINAMATH_CALUDE_odd_divisors_of_factorial_20_l481_48123

/-- The factorial of 20 -/
def factorial_20 : ℕ := 2432902008176640000

/-- The total number of natural divisors of 20! -/
def total_divisors : ℕ := 41040

/-- Theorem: The number of odd natural divisors of 20! is 2160 -/
theorem odd_divisors_of_factorial_20 : 
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors factorial_20)).card = 2160 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_of_factorial_20_l481_48123


namespace NUMINAMATH_CALUDE_log_equation_solution_l481_48124

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h2 : x > 0) :
  (Real.log x / Real.log k) * (Real.log (k^2) / Real.log 5) = 3 →
  x = 5 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l481_48124


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l481_48196

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) 
  (incorrect_value1 incorrect_value2 incorrect_value3 : ℚ)
  (correct_value1 correct_value2 correct_value3 : ℚ) :
  n = 50 ∧ 
  initial_mean = 350 ∧
  incorrect_value1 = 150 ∧ correct_value1 = 180 ∧
  incorrect_value2 = 200 ∧ correct_value2 = 235 ∧
  incorrect_value3 = 270 ∧ correct_value3 = 290 →
  (n : ℚ) * initial_mean + (correct_value1 - incorrect_value1) + 
  (correct_value2 - incorrect_value2) + (correct_value3 - incorrect_value3) = n * 351.7 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l481_48196


namespace NUMINAMATH_CALUDE_circle_area_when_eight_times_reciprocal_circumference_equals_diameter_l481_48122

theorem circle_area_when_eight_times_reciprocal_circumference_equals_diameter :
  ∀ (r : ℝ), r > 0 → (8 * (1 / (2 * Real.pi * r)) = 2 * r) → (Real.pi * r^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_when_eight_times_reciprocal_circumference_equals_diameter_l481_48122


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l481_48180

/-- The repeating decimal 0.8̄23 as a rational number -/
def repeating_decimal : ℚ := 0.8 + 23 / 99

/-- The expected fraction representation of 0.8̄23 -/
def expected_fraction : ℚ := 511 / 495

/-- Theorem stating that the repeating decimal 0.8̄23 is equal to 511/495 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = expected_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l481_48180


namespace NUMINAMATH_CALUDE_total_time_first_to_seventh_l481_48160

/-- Represents the travel times between stations in hours -/
def travel_times : List Real := [3, 2, 1.5, 4, 1, 2.5]

/-- Represents the break times at stations in minutes -/
def break_times : List Real := [45, 30, 15]

/-- Converts hours to minutes -/
def hours_to_minutes (hours : Real) : Real := hours * 60

/-- Calculates the total travel time in minutes -/
def total_travel_time : Real := (travel_times.map hours_to_minutes).sum

/-- Calculates the total break time in minutes -/
def total_break_time : Real := break_times.sum

/-- Theorem stating the total time from first to seventh station -/
theorem total_time_first_to_seventh : 
  total_travel_time + total_break_time = 930 := by sorry

end NUMINAMATH_CALUDE_total_time_first_to_seventh_l481_48160


namespace NUMINAMATH_CALUDE_weeks_in_month_is_four_l481_48188

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := sorry

/-- The standard work hours per week -/
def standard_hours_per_week : ℕ := 20

/-- The number of months worked -/
def months_worked : ℕ := 2

/-- The additional hours worked due to covering a shift -/
def additional_hours : ℕ := 20

/-- The total hours worked over the period -/
def total_hours_worked : ℕ := 180

theorem weeks_in_month_is_four :
  weeks_in_month = 4 :=
by sorry

end NUMINAMATH_CALUDE_weeks_in_month_is_four_l481_48188


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l481_48154

theorem sum_of_a_and_b (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 4) (h3 : a * b < 0) :
  a + b = 3 ∨ a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l481_48154


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l481_48134

theorem pure_imaginary_complex (m : ℝ) : 
  (m * (m - 1) : ℂ) + m * Complex.I = (0 : ℝ) + (b : ℝ) * Complex.I ∧ b ≠ 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l481_48134


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l481_48112

theorem fitness_center_membership_ratio :
  ∀ (f m c : ℕ), 
  (f > 0) → (m > 0) → (c > 0) →
  (35 * f + 30 * m + 10 * c : ℝ) / (f + m + c : ℝ) = 25 →
  ∃ (k : ℕ), k > 0 ∧ f = 3 * k ∧ m = 6 * k ∧ c = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l481_48112


namespace NUMINAMATH_CALUDE_time_expression_l481_48182

/-- Given V = 3gt + V₀ and S = (3/2)gt² + V₀t + (1/2)at², where a is another constant acceleration,
    prove that t = 9gS / (2(V - V₀)² + 3V₀(V - V₀)) -/
theorem time_expression (V V₀ g a S t : ℝ) 
  (hV : V = 3 * g * t + V₀)
  (hS : S = (3/2) * g * t^2 + V₀ * t + (1/2) * a * t^2) :
  t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by sorry

end NUMINAMATH_CALUDE_time_expression_l481_48182
