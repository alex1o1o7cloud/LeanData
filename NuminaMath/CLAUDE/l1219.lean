import Mathlib

namespace NUMINAMATH_CALUDE_soccer_balls_per_class_l1219_121978

theorem soccer_balls_per_class 
  (num_schools : ℕ)
  (elementary_classes_per_school : ℕ)
  (middle_classes_per_school : ℕ)
  (total_soccer_balls : ℕ)
  (h1 : num_schools = 2)
  (h2 : elementary_classes_per_school = 4)
  (h3 : middle_classes_per_school = 5)
  (h4 : total_soccer_balls = 90) :
  total_soccer_balls / (num_schools * (elementary_classes_per_school + middle_classes_per_school)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_per_class_l1219_121978


namespace NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l1219_121973

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted strip on a cube face --/
structure PaintedStrip where
  width : ℕ
  height : ℕ

/-- Calculates the number of unpainted unit cubes in a cube with painted strips --/
def unpainted_cubes (c : Cube 4) (strip : PaintedStrip) : ℕ :=
  sorry

theorem unpainted_cubes_4x4x4 :
  ∀ (c : Cube 4) (strip : PaintedStrip),
    strip.width = 2 ∧ strip.height = c.side_length →
    unpainted_cubes c strip = 40 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l1219_121973


namespace NUMINAMATH_CALUDE_tom_running_distance_l1219_121962

def base_twelve_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 12^3 + ((n / 100) % 10) * 12^2 + ((n / 10) % 10) * 12^1 + (n % 10)

def average_per_week (total : ℕ) (weeks : ℕ) : ℚ :=
  (total : ℚ) / (weeks : ℚ)

theorem tom_running_distance :
  let base_twelve_distance : ℕ := 3847
  let decimal_distance : ℕ := base_twelve_to_decimal base_twelve_distance
  let weeks : ℕ := 4
  decimal_distance = 6391 ∧ average_per_week decimal_distance weeks = 1597.75 := by
  sorry

end NUMINAMATH_CALUDE_tom_running_distance_l1219_121962


namespace NUMINAMATH_CALUDE_prob_ace_ten_queen_correct_l1219_121971

/-- The probability of drawing an Ace, then a 10, then a Queen from a standard 52-card deck without replacement -/
def prob_ace_ten_queen : ℚ := 8 / 16575

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of 10s in a standard deck -/
def num_tens : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

theorem prob_ace_ten_queen_correct (d : Deck) : 
  (num_aces : ℚ) / 52 * (num_tens : ℚ) / 51 * (num_queens : ℚ) / 50 = prob_ace_ten_queen :=
sorry

end NUMINAMATH_CALUDE_prob_ace_ten_queen_correct_l1219_121971


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l1219_121947

/-- Given x = (3 + √8)^1001, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1001
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l1219_121947


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1219_121922

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 18 / (9 - x ^ (1/4))) ↔ (x = 81 ∨ x = 1296) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1219_121922


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1219_121952

theorem complex_equation_sum (x y : ℝ) : 
  (x + y * Complex.I) / (1 + Complex.I) = (2 : ℂ) + Complex.I → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1219_121952


namespace NUMINAMATH_CALUDE_largest_zip_code_l1219_121937

def phone_number : List Nat := [4, 6, 5, 3, 2, 7, 1]

def is_valid_zip_code (zip : List Nat) : Prop :=
  zip.length = 4 ∧ 
  zip.toFinset.card = 4 ∧
  zip.sum = phone_number.sum

def zip_code_value (zip : List Nat) : Nat :=
  zip.foldl (fun acc d => acc * 10 + d) 0

theorem largest_zip_code :
  ∀ zip : List Nat, is_valid_zip_code zip →
  zip_code_value zip ≤ 9865 :=
sorry

end NUMINAMATH_CALUDE_largest_zip_code_l1219_121937


namespace NUMINAMATH_CALUDE_multiplication_equality_l1219_121950

-- Define the digits as natural numbers
def A : ℕ := 6
def B : ℕ := 7
def C : ℕ := 4
def D : ℕ := 2
def E : ℕ := 5
def F : ℕ := 9
def H : ℕ := 3
def J : ℕ := 8

-- Define the numbers ABCD and EF
def ABCD : ℕ := A * 1000 + B * 100 + C * 10 + D
def EF : ℕ := E * 10 + F

-- Define the result HFBBBJ
def HFBBBJ : ℕ := H * 100000 + F * 10000 + B * 1000 + B * 100 + B * 10 + J

-- State the theorem
theorem multiplication_equality :
  ABCD * EF = HFBBBJ :=
sorry

end NUMINAMATH_CALUDE_multiplication_equality_l1219_121950


namespace NUMINAMATH_CALUDE_min_colors_for_grid_l1219_121945

-- Define the grid as a type alias for pairs of integers
def Grid := ℤ × ℤ

-- Define the distance function between two cells
def distance (a b : Grid) : ℕ :=
  max (Int.natAbs (a.1 - b.1)) (Int.natAbs (a.2 - b.2))

-- Define the color function
def color (cell : Grid) : Fin 4 :=
  Fin.ofNat ((cell.1 + cell.2).natAbs % 4)

-- State the theorem
theorem min_colors_for_grid : 
  (∀ a b : Grid, distance a b = 6 → color a ≠ color b) ∧
  (∀ n : ℕ, n < 4 → ∃ a b : Grid, distance a b = 6 ∧ 
    Fin.ofNat (n % 4) = color a ∧ Fin.ofNat (n % 4) = color b) :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_grid_l1219_121945


namespace NUMINAMATH_CALUDE_system_solution_l1219_121909

theorem system_solution :
  ∀ (a b c d n m : ℚ),
    a / 7 + b / 8 = n →
    b = 3 * a - 2 →
    c / 9 + d / 10 = m →
    d = 4 * c + 1 →
    a = 3 →
    c = 2 →
    n = 73 / 56 ∧ m = 101 / 90 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1219_121909


namespace NUMINAMATH_CALUDE_tiling_impossibility_l1219_121981

/-- Represents a rectangular area that can be tiled. -/
structure TileableArea where
  width : ℕ
  height : ℕ

/-- Represents the count of each type of tile. -/
structure TileCount where
  two_by_two : ℕ
  one_by_four : ℕ

/-- Checks if a given area can be tiled with the given tile counts. -/
def can_tile (area : TileableArea) (tiles : TileCount) : Prop :=
  2 * tiles.two_by_two + 4 * tiles.one_by_four = area.width * area.height

/-- Theorem stating that if an area can be tiled, it becomes impossible
    to tile after replacing one 2x2 tile with a 1x4 tile. -/
theorem tiling_impossibility (area : TileableArea) (initial_tiles : TileCount) :
  can_tile area initial_tiles →
  ¬can_tile area { two_by_two := initial_tiles.two_by_two - 1,
                   one_by_four := initial_tiles.one_by_four + 1 } :=
by sorry

end NUMINAMATH_CALUDE_tiling_impossibility_l1219_121981


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1219_121988

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q : P ∩ Q = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1219_121988


namespace NUMINAMATH_CALUDE_toy_cost_l1219_121917

theorem toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_money = 57 →
  game_cost = 27 →
  num_toys = 5 →
  (initial_money - game_cost) % num_toys = 0 →
  (initial_money - game_cost) / num_toys = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l1219_121917


namespace NUMINAMATH_CALUDE_find_N_l1219_121967

theorem find_N : ∃ N : ℕ+, (22 ^ 2 * 55 ^ 2 : ℕ) = 10 ^ 2 * N ^ 2 ∧ N = 121 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l1219_121967


namespace NUMINAMATH_CALUDE_intersection_point_when_a_is_one_parallel_when_a_is_three_halves_l1219_121989

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := x + a * y - a + 2 = 0
def l₂ (a x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Theorem for the intersection point when a = 1
theorem intersection_point_when_a_is_one :
  ∃ (x y : ℝ), l₁ 1 x y ∧ l₂ 1 x y ∧ x = -4 ∧ y = 3 :=
sorry

-- Definition of parallel lines
def parallel (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (1 : ℝ) / (a : ℝ) = k * (2 * a) / (a + 3) ∧
  (a ≠ -3)

-- Theorem for parallel lines when a = 3/2
theorem parallel_when_a_is_three_halves :
  parallel (3/2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_when_a_is_one_parallel_when_a_is_three_halves_l1219_121989


namespace NUMINAMATH_CALUDE_hannahs_pay_l1219_121987

/-- Calculates the final pay for an employee given their hourly rate, hours worked, late penalty, and number of times late. -/
def calculate_final_pay (hourly_rate : ℕ) (hours_worked : ℕ) (late_penalty : ℕ) (times_late : ℕ) : ℕ :=
  hourly_rate * hours_worked - late_penalty * times_late

/-- Proves that Hannah's final pay is $525 given her work conditions. -/
theorem hannahs_pay :
  calculate_final_pay 30 18 5 3 = 525 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_pay_l1219_121987


namespace NUMINAMATH_CALUDE_simplify_fraction_l1219_121926

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1219_121926


namespace NUMINAMATH_CALUDE_unique_digits_for_multiple_of_99_l1219_121923

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem unique_digits_for_multiple_of_99 :
  ∀ α β : ℕ,
  0 ≤ α ∧ α ≤ 9 →
  0 ≤ β ∧ β ≤ 9 →
  is_divisible_by_99 (62 * 10000 + α * 1000 + β * 100 + 427) →
  α = 2 ∧ β = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_digits_for_multiple_of_99_l1219_121923


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1219_121976

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 5 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1219_121976


namespace NUMINAMATH_CALUDE_rolling_coin_curve_length_l1219_121975

/-- The length of the curve traced by the center of a rolling coin -/
theorem rolling_coin_curve_length 
  (coin_circumference : ℝ) 
  (quadrilateral_perimeter : ℝ) : 
  coin_circumference = 5 →
  quadrilateral_perimeter = 20 →
  (curve_length : ℝ) = quadrilateral_perimeter + coin_circumference →
  curve_length = 25 :=
by sorry

end NUMINAMATH_CALUDE_rolling_coin_curve_length_l1219_121975


namespace NUMINAMATH_CALUDE_intersection_locus_l1219_121992

/-- The locus of the intersection point of two lines in a Cartesian coordinate system -/
theorem intersection_locus (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  ∀ (x y : ℝ), 
  (∃ c : ℝ, c ≠ 0 ∧ 
    (y = (a / c) * x) ∧ 
    (x / b + y / c = 1)) →
  ((x - b / 2)^2 / (b^2 / 4) + y^2 / (a * b / 4) = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_l1219_121992


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l1219_121995

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def move_first_digit_to_end (n : ℕ) : ℕ :=
  (n % 10000) * 10 + (n / 10000)

theorem unique_five_digit_number : ∃! n : ℕ,
  is_five_digit n ∧
  move_first_digit_to_end n = n + 34767 ∧
  move_first_digit_to_end n + n = 86937 ∧
  n = 26035 := by
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l1219_121995


namespace NUMINAMATH_CALUDE_expression_defined_iff_l1219_121933

def expression_defined (x : ℝ) : Prop :=
  x > 2 ∧ x < 5

theorem expression_defined_iff (x : ℝ) :
  expression_defined x ↔ (∃ y : ℝ, y = (Real.log (5 - x)) / Real.sqrt (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l1219_121933


namespace NUMINAMATH_CALUDE_greatest_average_speed_l1219_121919

/-- Checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The initial odometer reading -/
def initialReading : ℕ := 12321

/-- The duration of the drive in hours -/
def driveDuration : ℝ := 4

/-- The speed limit in miles per hour -/
def speedLimit : ℝ := 85

/-- The greatest possible average speed in miles per hour -/
def greatestAverageSpeed : ℝ := 75

/-- Theorem stating the greatest possible average speed given the conditions -/
theorem greatest_average_speed :
  isPalindrome initialReading →
  ∃ (finalReading : ℕ),
    isPalindrome finalReading ∧
    finalReading > initialReading ∧
    (finalReading - initialReading : ℝ) / driveDuration ≤ speedLimit ∧
    (finalReading - initialReading : ℝ) / driveDuration = greatestAverageSpeed :=
  sorry

end NUMINAMATH_CALUDE_greatest_average_speed_l1219_121919


namespace NUMINAMATH_CALUDE_honey_water_percentage_l1219_121953

/-- Given that 1.7 kg of flower-nectar containing 50% water yields 1 kg of honey,
    prove that the percentage of water in the resulting honey is 15%. -/
theorem honey_water_percentage
  (nectar_weight : ℝ)
  (honey_weight : ℝ)
  (nectar_water_percentage : ℝ)
  (h1 : nectar_weight = 1.7)
  (h2 : honey_weight = 1)
  (h3 : nectar_water_percentage = 50)
  : (honey_weight - (nectar_weight * (1 - nectar_water_percentage / 100))) / honey_weight * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_honey_water_percentage_l1219_121953


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1219_121901

/-- The function f(x) = a^(2x-1) + 1 passes through (1/2, 2) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) + 1
  f (1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1219_121901


namespace NUMINAMATH_CALUDE_frustum_surface_area_l1219_121925

/-- The surface area of a frustum of a regular pyramid with square bases -/
theorem frustum_surface_area (top_side : ℝ) (bottom_side : ℝ) (slant_height : ℝ) :
  top_side = 2 →
  bottom_side = 4 →
  slant_height = 2 →
  let lateral_area := (top_side + bottom_side) * slant_height * 2
  let top_area := top_side ^ 2
  let bottom_area := bottom_side ^ 2
  lateral_area + top_area + bottom_area = 12 * Real.sqrt 3 + 20 := by
  sorry

end NUMINAMATH_CALUDE_frustum_surface_area_l1219_121925


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1219_121956

theorem ellipse_eccentricity (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 / (b^2 + 1) = 1 → 
    b / Real.sqrt (b^2 + 1) = Real.sqrt 10 / 10) → 
  b = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1219_121956


namespace NUMINAMATH_CALUDE_range_of_a_l1219_121984

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ∈ Set.Iic (-2) ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1219_121984


namespace NUMINAMATH_CALUDE_exactly_two_valid_numbers_l1219_121948

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_number (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  is_perfect_square (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) ∧
  is_perfect_square ((n / 10 % 10) + (n % 10)) ∧
  is_perfect_square ((n / 10 % 10) - (n % 10)) ∧
  is_perfect_square (n % 10) ∧
  is_perfect_square ((n / 100) % 100) ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) > 0) ∧
  ((n / 10 % 10) + (n % 10) > 0) ∧
  ((n / 10 % 10) - (n % 10) > 0) ∧
  (n % 10 > 0)

theorem exactly_two_valid_numbers :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n ∈ s, valid_number n :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_numbers_l1219_121948


namespace NUMINAMATH_CALUDE_horner_f_at_5_v2_eq_21_l1219_121938

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 - 5x^4 - 4x^3 + 3x^2 - 6x + 7 -/
def f : List ℝ := [2, -5, -4, 3, -6, 7]

/-- Theorem: Horner's method for f(x) at x = 5 yields v_2 = 21 -/
theorem horner_f_at_5_v2_eq_21 :
  let v := horner f 5
  let v0 := 2
  let v1 := v0 * 5 - 5
  let v2 := v1 * 5 - 4
  v2 = 21 := by sorry

end NUMINAMATH_CALUDE_horner_f_at_5_v2_eq_21_l1219_121938


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l1219_121964

open Real

theorem min_value_of_trig_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (∀ y, 0 < y ∧ y < π / 2 → 
    (1 + cos (2 * y) + 8 * sin y ^ 2) / sin (2 * y) ≥ 
    (1 + cos (2 * x) + 8 * sin x ^ 2) / sin (2 * x)) →
  (1 + cos (2 * x) + 8 * sin x ^ 2) / sin (2 * x) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l1219_121964


namespace NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l1219_121963

theorem triangles_with_fixed_vertex (n : ℕ) (h : n = 9) :
  Nat.choose (n - 1) 2 = 28 :=
sorry

end NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l1219_121963


namespace NUMINAMATH_CALUDE_beths_shopping_multiple_l1219_121968

/-- The problem of Beth's shopping for peas and corn -/
theorem beths_shopping_multiple (peas corn : ℕ) (multiple : ℚ) 
  (h1 : peas = corn * multiple + 15)
  (h2 : peas = 35)
  (h3 : corn = 10) :
  multiple = 2 := by
  sorry

end NUMINAMATH_CALUDE_beths_shopping_multiple_l1219_121968


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l1219_121998

/-- A type representing lines in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A type representing planes in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem perpendicular_parallel_implies_perpendicular 
  (b c : Line3D) (α : Plane3D) :
  perpendicular_line_plane b α → 
  parallel_line_plane c α → 
  perpendicular_lines b c :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l1219_121998


namespace NUMINAMATH_CALUDE_oldest_child_age_l1219_121903

/-- Represents the ages of 7 children -/
def ChildrenAges := Fin 7 → ℕ

/-- The property that each child has a different age -/
def AllDifferent (ages : ChildrenAges) : Prop :=
  ∀ i j : Fin 7, i ≠ j → ages i ≠ ages j

/-- The property that the difference in age between consecutive children is 1 year -/
def ConsecutiveDifference (ages : ChildrenAges) : Prop :=
  ∀ i : Fin 6, ages (Fin.succ i) = ages i + 1

/-- The average age of the children is 8 years -/
def AverageAge (ages : ChildrenAges) : Prop :=
  (ages 0 + ages 1 + ages 2 + ages 3 + ages 4 + ages 5 + ages 6) / 7 = 8

theorem oldest_child_age
  (ages : ChildrenAges)
  (h_diff : AllDifferent ages)
  (h_cons : ConsecutiveDifference ages)
  (h_avg : AverageAge ages) :
  ages 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l1219_121903


namespace NUMINAMATH_CALUDE_intersection_line_equation_l1219_121912

/-- Given two lines L₁ and L₂ in the plane, and a third line L that intersects both L₁ and L₂,
    if the midpoint of the line segment formed by these intersections is the origin,
    then L has the equation x + 6y = 0. -/
theorem intersection_line_equation (L₁ L₂ L : Set (ℝ × ℝ)) :
  L₁ = {p : ℝ × ℝ | 4 * p.1 + p.2 + 6 = 0} →
  L₂ = {p : ℝ × ℝ | 3 * p.1 - 5 * p.2 - 6 = 0} →
  (∃ A B : ℝ × ℝ, A ∈ L ∩ L₁ ∧ B ∈ L ∩ L₂ ∧ (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0) →
  L = {p : ℝ × ℝ | p.1 + 6 * p.2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l1219_121912


namespace NUMINAMATH_CALUDE_M_subset_P_l1219_121970

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = -x^2 + 1}
def P : Set ℝ := Set.univ

-- State the theorem
theorem M_subset_P : M ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_M_subset_P_l1219_121970


namespace NUMINAMATH_CALUDE_average_visitors_per_day_l1219_121972

def visitor_counts : List ℕ := [583, 246, 735, 492, 639]
def num_days : ℕ := 5

theorem average_visitors_per_day :
  (visitor_counts.sum / num_days : ℚ) = 539 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_per_day_l1219_121972


namespace NUMINAMATH_CALUDE_correct_statements_count_l1219_121979

theorem correct_statements_count (x : ℝ) : 
  (((x > 0) → (x^2 > 0)) ∧ ((x^2 ≤ 0) → (x ≤ 0)) ∧ ¬((x ≤ 0) → (x^2 ≤ 0))) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_count_l1219_121979


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l1219_121940

theorem complex_in_fourth_quadrant (m : ℝ) (z : ℂ) 
  (h1 : m < 1) 
  (h2 : z = 2 + (m - 1) * Complex.I) : 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l1219_121940


namespace NUMINAMATH_CALUDE_smallest_four_digit_not_dividing_l1219_121921

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_not_dividing :
  ∃ (n : ℕ), is_four_digit n ∧
    ¬(sum_of_first_n n ∣ product_of_first_n n) ∧
    (∀ m, is_four_digit m ∧ m < n →
      sum_of_first_n m ∣ product_of_first_n m) ∧
    n = 1002 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_not_dividing_l1219_121921


namespace NUMINAMATH_CALUDE_west_movement_l1219_121949

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (dir : Direction) (distance : ℤ) : ℤ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_movement :
  (movement Direction.East 50 = 50) →
  (∀ (d : Direction) (x : ℤ), movement d x = -movement (match d with
    | Direction.East => Direction.West
    | Direction.West => Direction.East) x) →
  (movement Direction.West 60 = -60) :=
by
  sorry

end NUMINAMATH_CALUDE_west_movement_l1219_121949


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l1219_121908

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the line perpendicular to the tangent
def perp_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 2 = 0

-- Theorem statement
theorem tangent_perpendicular_to_line :
  ∀ (x₀ y₀ : ℝ),
  y₀ = curve x₀ →
  (∃ (m : ℝ), ∀ (x y : ℝ), y - y₀ = m * (x - x₀) → 
    (perp_line x y ↔ (x - x₀) * 1 + (y - y₀) * 4 = 0)) →
  tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l1219_121908


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1219_121931

def z : ℂ := Complex.I + Complex.I^6

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1219_121931


namespace NUMINAMATH_CALUDE_sodium_chloride_moles_l1219_121911

-- Define the chemical reaction components
structure ChemicalReaction where
  NaCl : ℕ  -- moles of Sodium chloride
  HNO3 : ℕ  -- moles of Nitric acid
  NaNO3 : ℕ  -- moles of Sodium nitrate
  HCl : ℕ   -- moles of Hydrochloric acid

-- Define the theorem
theorem sodium_chloride_moles (reaction : ChemicalReaction) :
  reaction.NaNO3 = 2 →  -- Condition 1
  reaction.HCl = 2 →    -- Condition 2
  reaction.HNO3 = reaction.NaNO3 →  -- Condition 3
  reaction.NaCl = 2 :=  -- Conclusion
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_sodium_chloride_moles_l1219_121911


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1219_121974

theorem min_value_of_expression (x y : ℝ) : (2*x*y - 3)^2 + (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1219_121974


namespace NUMINAMATH_CALUDE_solution_set_correct_inequality_factorization_l1219_121902

/-- The solution set of the quadratic inequality ax^2 + (a-2)x - 2 ≤ 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Ici (-1)
  else if a > 0 then Set.Icc (-1) (2/a)
  else if -2 < a ∧ a < 0 then Set.Iic (2/a) ∪ Set.Ici (-1)
  else if a < -2 then Set.Iic (-1) ∪ Set.Ici (2/a)
  else Set.univ

/-- Theorem stating that the solution_set function correctly solves the quadratic inequality -/
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 + (a - 2) * x - 2 ≤ 0 := by
  sorry

/-- Theorem stating that the quadratic inequality can be rewritten as a product of linear factors -/
theorem inequality_factorization (a : ℝ) (x : ℝ) :
  a * x^2 + (a - 2) * x - 2 = (a * x - 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_inequality_factorization_l1219_121902


namespace NUMINAMATH_CALUDE_range_of_m_l1219_121994

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - x₁ + m - 4 = 0 ∧ 
              x₂^2 - x₂ + m - 4 = 0 ∧ 
              x₁ * x₂ < 0

-- Main theorem
theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬p m) :
  m ≤ 1 - Real.sqrt 2 ∨ (1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1219_121994


namespace NUMINAMATH_CALUDE_inverse_88_mod_89_l1219_121985

theorem inverse_88_mod_89 : ∃ x : ℕ, x ≤ 88 ∧ (88 * x) % 89 = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_inverse_88_mod_89_l1219_121985


namespace NUMINAMATH_CALUDE_circles_intersect_l1219_121996

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Theorem stating that the circles intersect
theorem circles_intersect : ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l1219_121996


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1219_121982

theorem smallest_prime_divisor_of_sum : ∃ k : ℕ, 4^15 + 6^17 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1219_121982


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1219_121936

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of five consecutive terms equals 450 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SumCondition a → a 2 + a 8 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1219_121936


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1219_121920

-- Define the properties of function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Main theorem
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on_nonneg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f (2 * x - 1) > 0} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1219_121920


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1219_121916

theorem quadratic_factorization (x : ℝ) : x^2 - 5*x + 6 = (x - 2) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1219_121916


namespace NUMINAMATH_CALUDE_cafeteria_problem_l1219_121927

theorem cafeteria_problem (n : ℕ) (h : n = 6) :
  (∃ (max_days : ℕ) (avg_dishes : ℚ),
    max_days = 2^n ∧
    avg_dishes = n / 2 ∧
    max_days = 64 ∧
    avg_dishes = 3) := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_problem_l1219_121927


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1219_121932

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (percentage_longer : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + percentage_longer)

/-- Theorem: The length of the major axis of the ellipse is 6.4 --/
theorem ellipse_major_axis_length :
  major_axis_length 2 0.6 = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1219_121932


namespace NUMINAMATH_CALUDE_f_2018_equals_neg_2018_l1219_121905

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -1 / f (x + 3)

theorem f_2018_equals_neg_2018
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_eq : satisfies_equation f)
  (h_f4 : f 4 = -2018) :
  f 2018 = -2018 :=
sorry

end NUMINAMATH_CALUDE_f_2018_equals_neg_2018_l1219_121905


namespace NUMINAMATH_CALUDE_polynomial_form_l1219_121939

/-- A real-coefficient polynomial function -/
def RealPolynomial := ℝ → ℝ

/-- The condition that needs to be satisfied by the polynomial -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), a * b + b * c + c * a = 0 →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form (P : RealPolynomial) 
    (h : SatisfiesCondition P) : 
    ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_form_l1219_121939


namespace NUMINAMATH_CALUDE_hidden_dots_count_l1219_121990

/-- Represents a standard six-sided die -/
def StandardDie := Fin 6

/-- The sum of dots on all faces of a standard die -/
def sumOfDots : ℕ := (List.range 6).sum + 6

/-- The list of visible face values -/
def visibleFaces : List ℕ := [1, 2, 3, 4, 5, 4, 6, 5, 3]

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 4

/-- The number of visible faces -/
def numberOfVisibleFaces : ℕ := 9

theorem hidden_dots_count :
  (numberOfDice * sumOfDots) - visibleFaces.sum = 51 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l1219_121990


namespace NUMINAMATH_CALUDE_product_equals_99999919_l1219_121951

theorem product_equals_99999919 : 103 * 97 * 10009 = 99999919 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_99999919_l1219_121951


namespace NUMINAMATH_CALUDE_quadratic_polynomial_theorem_l1219_121906

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on the graph of a quadratic polynomial -/
def pointLiesOnPolynomial (p : Point) (q : QuadraticPolynomial) : Prop :=
  p.y = q.a * p.x^2 + q.b * p.x + q.c

/-- The main theorem -/
theorem quadratic_polynomial_theorem 
  (points : Finset Point) 
  (h_count : points.card = 100)
  (h_four_points : ∀ (p₁ p₂ p₃ p₄ : Point), p₁ ∈ points → p₂ ∈ points → p₃ ∈ points → p₄ ∈ points →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₁ ≠ p₄ → p₂ ≠ p₃ → p₂ ≠ p₄ → p₃ ≠ p₄ →
    ∃ (q : QuadraticPolynomial), pointLiesOnPolynomial p₁ q ∧ pointLiesOnPolynomial p₂ q ∧
      pointLiesOnPolynomial p₃ q ∧ pointLiesOnPolynomial p₄ q) :
  ∃ (q : QuadraticPolynomial), ∀ (p : Point), p ∈ points → pointLiesOnPolynomial p q :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_theorem_l1219_121906


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1219_121991

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x - 1/x) - 2 * Real.log x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 + 2/(x^2) - 2/x

-- Theorem statement
theorem tangent_line_at_one (x y : ℝ) :
  (y = f x) → (x = 1) → (2*x - y - 2 = 0) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_l1219_121991


namespace NUMINAMATH_CALUDE_partner_C_investment_l1219_121997

/-- Represents the investment and profit structure of a business partnership --/
structure BusinessPartnership where
  investment_A : ℕ
  investment_B : ℕ
  profit_share_B : ℕ
  profit_diff_AC : ℕ

/-- Calculates the investment of partner C given the business partnership details --/
def calculate_investment_C (bp : BusinessPartnership) : ℕ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating that given the specific business partnership details, 
    partner C's investment is 120000 --/
theorem partner_C_investment 
  (bp : BusinessPartnership) 
  (h1 : bp.investment_A = 8000)
  (h2 : bp.investment_B = 10000)
  (h3 : bp.profit_share_B = 1400)
  (h4 : bp.profit_diff_AC = 560) : 
  calculate_investment_C bp = 120000 := by
  sorry

end NUMINAMATH_CALUDE_partner_C_investment_l1219_121997


namespace NUMINAMATH_CALUDE_g_sum_zero_l1219_121954

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l1219_121954


namespace NUMINAMATH_CALUDE_rank_from_bottom_calculation_l1219_121957

/-- Represents a student's ranking in a class. -/
structure StudentRanking where
  totalStudents : Nat
  rankFromTop : Nat
  rankFromBottom : Nat

/-- Calculates the rank from the bottom given the total number of students and rank from the top. -/
def calculateRankFromBottom (total : Nat) (rankFromTop : Nat) : Nat :=
  total - rankFromTop + 1

/-- Theorem stating that for a class of 53 students, a student ranking 5th from the top
    will rank 49th from the bottom. -/
theorem rank_from_bottom_calculation (s : StudentRanking)
    (h1 : s.totalStudents = 53)
    (h2 : s.rankFromTop = 5)
    (h3 : s.rankFromBottom = calculateRankFromBottom s.totalStudents s.rankFromTop) :
  s.rankFromBottom = 49 := by
  sorry

#check rank_from_bottom_calculation

end NUMINAMATH_CALUDE_rank_from_bottom_calculation_l1219_121957


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1219_121944

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_prod : a 5 * a 11 = 3)
  (h_sum : a 3 + a 13 = 4) :
  ∃ r : ℝ, (r = 3 ∨ r = -3) ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1219_121944


namespace NUMINAMATH_CALUDE_recliner_sales_increase_l1219_121910

/-- Proves that a 20% price reduction and 28% gross revenue increase results in a 60% increase in sales volume -/
theorem recliner_sales_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (new_price : ℝ) 
  (new_quantity : ℝ) 
  (h1 : new_price = 0.80 * original_price) 
  (h2 : new_price * new_quantity = 1.28 * (original_price * original_quantity)) : 
  (new_quantity - original_quantity) / original_quantity = 0.60 := by
sorry

end NUMINAMATH_CALUDE_recliner_sales_increase_l1219_121910


namespace NUMINAMATH_CALUDE_notebook_cost_l1219_121966

theorem notebook_cost (book_cost : ℝ) (binders_cost : ℝ) (num_notebooks : ℕ) (total_cost : ℝ)
  (h1 : book_cost = 16)
  (h2 : binders_cost = 6)
  (h3 : num_notebooks = 6)
  (h4 : total_cost = 28)
  : (total_cost - (book_cost + binders_cost)) / num_notebooks = 1 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1219_121966


namespace NUMINAMATH_CALUDE_area_constant_circle_final_equation_minimum_distance_l1219_121941

noncomputable section

variable (t : ℝ)
variable (h : t ≠ 0)

def C : ℝ × ℝ := (t, 2/t)
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2*t, 0)
def B : ℝ × ℝ := (0, 4/t)

def circle_equation (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2/t)^2 = t^2 + 4/t^2

def line_equation (x y : ℝ) : Prop :=
  2*x + y - 4 = 0

def line_l_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

theorem area_constant :
  (1/2) * |2*t| * |4/t| = 4 :=
sorry

theorem circle_final_equation (x y : ℝ) :
  (∃ M N : ℝ × ℝ, 
    circle_equation t x y ∧ 
    line_equation (M.1) (M.2) ∧ 
    line_equation (N.1) (N.2) ∧
    (M.1 - O.1)^2 + (M.2 - O.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2) →
  (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

theorem minimum_distance (h_pos : t > 0) :
  let B : ℝ × ℝ := (0, 2)
  ∃ P Q : ℝ × ℝ,
    line_l_equation P.1 P.2 ∧
    circle_equation t Q.1 Q.2 ∧
    (∀ P' Q' : ℝ × ℝ, 
      line_l_equation P'.1 P'.2 → 
      circle_equation t Q'.1 Q'.2 →
      Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤
      Real.sqrt ((P'.1 - B.1)^2 + (P'.2 - B.2)^2) + Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 5 ∧
    P.1 = -4/3 ∧ P.2 = -2/3 :=
sorry

end NUMINAMATH_CALUDE_area_constant_circle_final_equation_minimum_distance_l1219_121941


namespace NUMINAMATH_CALUDE_ellipse_properties_l1219_121983

/-- Definition of an ellipse passing through a point with given foci -/
def is_ellipse_through_point (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f2.1 - f1.1)^2 + (f2.2 - f1.2)^2) / 2
  ∃ a : ℝ, a > c ∧ d1 + d2 = 2 * a

/-- The equation of an ellipse in standard form -/
def ellipse_equation (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  let f1 : ℝ × ℝ := (0, 0)
  let f2 : ℝ × ℝ := (0, 8)
  let p : ℝ × ℝ := (7, 4)
  let a : ℝ := 8 * Real.sqrt 2
  let b : ℝ := 8 * Real.sqrt 7
  let h : ℝ := 0
  let k : ℝ := 4
  is_ellipse_through_point f1 f2 p →
  (∀ x y : ℝ, ellipse_equation a b h k x y ↔ 
    ((x - 0)^2 / (8 * Real.sqrt 2)^2 + (y - 4)^2 / (8 * Real.sqrt 7)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1219_121983


namespace NUMINAMATH_CALUDE_select_twelve_students_l1219_121946

/-- Represents the number of students in each course -/
structure CourseDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sampling information -/
structure SamplingInfo where
  total_students : ℕ
  selected_students : ℕ

/-- Checks if the course distribution forms an arithmetic sequence with the given common difference -/
def is_arithmetic_sequence (dist : CourseDistribution) (diff : ℤ) : Prop :=
  dist.second = dist.first - diff ∧ dist.third = dist.second - diff

/-- Calculates the number of students to be selected from the first course -/
def students_to_select (dist : CourseDistribution) (info : SamplingInfo) : ℕ :=
  (dist.first * info.selected_students) / info.total_students

/-- Main theorem: Given the conditions, prove that 12 students should be selected from the first course -/
theorem select_twelve_students 
  (dist : CourseDistribution)
  (info : SamplingInfo)
  (h1 : dist.first + dist.second + dist.third = info.total_students)
  (h2 : info.total_students = 600)
  (h3 : info.selected_students = 30)
  (h4 : is_arithmetic_sequence dist (-40)) :
  students_to_select dist info = 12 := by
  sorry

end NUMINAMATH_CALUDE_select_twelve_students_l1219_121946


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1219_121914

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_third_term : a 3 = 16)
  (h_seventh_term : a 7 = 2) :
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1219_121914


namespace NUMINAMATH_CALUDE_range_of_a_l1219_121935

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - x - 2 ≥ 0 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ x^2 - x - 2 < 0) → 
  a ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1219_121935


namespace NUMINAMATH_CALUDE_h3po4_naoh_reaction_results_l1219_121980

/-- Represents a chemical compound in a reaction --/
structure Compound where
  name : String
  moles : ℝ

/-- Represents a balanced chemical equation --/
structure BalancedEquation where
  reactant1 : Compound
  reactant2 : Compound
  product1 : Compound
  product2 : Compound
  stoichiometry : ℝ

/-- Determines the limiting reactant and calculates reaction results --/
def reactionResults (eq : BalancedEquation) : Compound × Compound × Compound := sorry

/-- Theorem stating the reaction results for H3PO4 and NaOH --/
theorem h3po4_naoh_reaction_results :
  let h3po4 := Compound.mk "H3PO4" 2.5
  let naoh := Compound.mk "NaOH" 3
  let equation := BalancedEquation.mk h3po4 naoh (Compound.mk "Na3PO4" 0) (Compound.mk "H2O" 0) 3
  let (h2o_formed, limiting_reactant, unreacted_h3po4) := reactionResults equation
  h2o_formed.moles = 3 ∧
  limiting_reactant.name = "NaOH" ∧
  unreacted_h3po4.moles = 1.5 := by sorry

end NUMINAMATH_CALUDE_h3po4_naoh_reaction_results_l1219_121980


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l1219_121942

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 2)^2
def parabola2 (x y : ℝ) : Prop := x + 3 = (y - 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_sum_zero_l1219_121942


namespace NUMINAMATH_CALUDE_reciprocals_inversely_proportional_l1219_121986

/-- Two real numbers are inversely proportional if their product is constant --/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

/-- Theorem: If x and y are inversely proportional, then their reciprocals are also inversely proportional --/
theorem reciprocals_inversely_proportional
  (x y : ℝ → ℝ)
  (h : InverselyProportional x y)
  (hx : ∀ t, x t ≠ 0)
  (hy : ∀ t, y t ≠ 0) :
  InverselyProportional (fun t ↦ 1 / x t) (fun t ↦ 1 / y t) :=
by
  sorry

end NUMINAMATH_CALUDE_reciprocals_inversely_proportional_l1219_121986


namespace NUMINAMATH_CALUDE_school_enrollment_increase_l1219_121960

-- Define the variables and constants
def last_year_total : ℕ := 4000
def last_year_YY : ℕ := 2400
def XX_percent_increase : ℚ := 7 / 100
def extra_growth_XX : ℕ := 40

-- Define the theorem
theorem school_enrollment_increase : 
  ∃ (p : ℚ), 
    (p ≥ 0) ∧ 
    (p ≤ 1) ∧
    (XX_percent_increase * (last_year_total - last_year_YY) = 
     (p * last_year_YY) + extra_growth_XX) ∧
    (p = 3 / 100) := by
  sorry

end NUMINAMATH_CALUDE_school_enrollment_increase_l1219_121960


namespace NUMINAMATH_CALUDE_total_weight_of_pets_l1219_121915

/-- The total weight of four pets given specific weight relationships -/
theorem total_weight_of_pets (evan_dog : ℝ) (ivan_dog : ℝ) (kara_cat : ℝ) (lisa_parrot : ℝ) 
  (h1 : evan_dog = 63)
  (h2 : evan_dog = 7 * ivan_dog)
  (h3 : kara_cat = 5 * (evan_dog + ivan_dog))
  (h4 : lisa_parrot = 3 * (evan_dog + ivan_dog + kara_cat)) :
  evan_dog + ivan_dog + kara_cat + lisa_parrot = 1728 := by
  sorry

#check total_weight_of_pets

end NUMINAMATH_CALUDE_total_weight_of_pets_l1219_121915


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1219_121928

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_property
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h_m_pos : m > 0)
  (h_sum_m : seq.S m = -2)
  (h_sum_m1 : seq.S (m + 1) = 0)
  (h_sum_m2 : seq.S (m + 2) = 3) :
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1219_121928


namespace NUMINAMATH_CALUDE_physical_exercise_test_results_l1219_121959

/-- Represents a school in the physical exercise test --/
structure School where
  name : String
  total_students : Nat
  sampled_students : Nat
  average_score : Float
  median_score : Float
  mode_score : Nat

/-- Represents the score distribution for a school --/
structure ScoreDistribution where
  school : School
  scores : List (Nat × Nat)  -- (score_range_start, count)

theorem physical_exercise_test_results 
  (school_a school_b : School)
  (dist_a : ScoreDistribution)
  (h1 : school_a.name = "School A")
  (h2 : school_b.name = "School B")
  (h3 : school_a.total_students = 180)
  (h4 : school_b.total_students = 180)
  (h5 : school_a.sampled_students = 30)
  (h6 : school_b.sampled_students = 30)
  (h7 : school_a.average_score = 96.35)
  (h8 : school_a.mode_score = 99)
  (h9 : school_b.average_score = 95.85)
  (h10 : school_b.median_score = 97.5)
  (h11 : school_b.mode_score = 99)
  (h12 : dist_a.school = school_a)
  (h13 : dist_a.scores = [(90, 2), (92, 3), (94, 5), (96, 10), (98, 10)]) :
  school_a.median_score = 96.5 ∧ 
  (((school_a.total_students * 20) / 30 : Nat) * 2 - 100 = 140) := by
  sorry

end NUMINAMATH_CALUDE_physical_exercise_test_results_l1219_121959


namespace NUMINAMATH_CALUDE_set_operations_l1219_121958

def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x | 4 < x ∧ x < 6}) ∧
  (Set.univ \ B = {x | x ≥ 6 ∨ x ≤ -6}) ∧
  (A \ B = {x | x ≥ 6}) ∧
  (A \ (A \ B) = {x | 4 < x ∧ x < 6}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l1219_121958


namespace NUMINAMATH_CALUDE_race_head_start_l1219_121955

/-- Proof of head start time in a race --/
theorem race_head_start 
  (race_distance : ℝ) 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (catch_up_time : ℝ)
  (h1 : race_distance = 500)
  (h2 : cristina_speed = 5)
  (h3 : nicky_speed = 3)
  (h4 : catch_up_time = 30) :
  let distance_covered := nicky_speed * catch_up_time
  let cristina_time := distance_covered / cristina_speed
  let head_start := catch_up_time - cristina_time
  head_start = 12 := by sorry

end NUMINAMATH_CALUDE_race_head_start_l1219_121955


namespace NUMINAMATH_CALUDE_isabella_read_250_pages_l1219_121999

/-- The number of pages Isabella read in a week -/
def total_pages (pages_first_three : ℕ) (pages_next_three : ℕ) (pages_last_day : ℕ) : ℕ :=
  3 * pages_first_three + 3 * pages_next_three + pages_last_day

/-- Theorem stating that Isabella read 250 pages in total -/
theorem isabella_read_250_pages : 
  total_pages 36 44 10 = 250 := by
  sorry

#check isabella_read_250_pages

end NUMINAMATH_CALUDE_isabella_read_250_pages_l1219_121999


namespace NUMINAMATH_CALUDE_wall_length_is_800_l1219_121918

-- Define the dimensions of a single brick
def brick_length : ℝ := 125
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the known dimensions of the wall
def wall_width : ℝ := 600
def wall_height : ℝ := 22.5

-- Define the number of bricks needed
def num_bricks : ℕ := 1280

-- Theorem statement
theorem wall_length_is_800 :
  ∃ (wall_length : ℝ),
    wall_length = 800 ∧
    (brick_length * brick_width * brick_height) * num_bricks =
    wall_length * wall_width * wall_height :=
by
  sorry


end NUMINAMATH_CALUDE_wall_length_is_800_l1219_121918


namespace NUMINAMATH_CALUDE_x_value_l1219_121977

theorem x_value : ∃ x : ℝ, (3 * x = (20 - x) + 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1219_121977


namespace NUMINAMATH_CALUDE_union_of_sets_l1219_121969

def set_A : Set ℝ := {x | |x - 1| < 3}
def set_B : Set ℝ := {x | x^2 - 4*x < 0}

theorem union_of_sets : set_A ∪ set_B = Set.Ioo (-2) 4 := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1219_121969


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1219_121934

/-- The inclination angle of a line with point-slope form y - 2 = -√3(x - 1) is π/3 -/
theorem line_inclination_angle (x y : ℝ) :
  y - 2 = -Real.sqrt 3 * (x - 1) → ∃ α : ℝ, α = π / 3 ∧ Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1219_121934


namespace NUMINAMATH_CALUDE_range_of_a_l1219_121993

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1219_121993


namespace NUMINAMATH_CALUDE_number_of_boys_l1219_121930

/-- Given a school with a total of 1396 people, 315 girls, and 772 teachers,
    prove that there are 309 boys in the school. -/
theorem number_of_boys (total : ℕ) (girls : ℕ) (teachers : ℕ) 
    (h1 : total = 1396) 
    (h2 : girls = 315) 
    (h3 : teachers = 772) : 
  total - girls - teachers = 309 := by
  sorry


end NUMINAMATH_CALUDE_number_of_boys_l1219_121930


namespace NUMINAMATH_CALUDE_nicky_running_time_l1219_121904

-- Define the race parameters
def race_distance : ℝ := 400
def head_start : ℝ := 12
def cristina_speed : ℝ := 5
def nicky_speed : ℝ := 3

-- Define the theorem
theorem nicky_running_time (t : ℝ) : 
  t * cristina_speed = head_start * nicky_speed + (t + head_start) * nicky_speed → 
  t + head_start = 48 :=
by sorry

end NUMINAMATH_CALUDE_nicky_running_time_l1219_121904


namespace NUMINAMATH_CALUDE_right_triangle_with_consecutive_sides_l1219_121907

theorem right_triangle_with_consecutive_sides (a b c : ℕ) : 
  a = 11 → b + 1 = c → a^2 + b^2 = c^2 → c = 61 := by sorry

end NUMINAMATH_CALUDE_right_triangle_with_consecutive_sides_l1219_121907


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1219_121924

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 10) : 
  3 * a 5 + a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1219_121924


namespace NUMINAMATH_CALUDE_equation_solutions_l1219_121961

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 + 6 * x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x + 2)^2 = 3 * (x + 2)
  let sol1 : Set ℝ := {(-3 + Real.sqrt 3) / 2, (-3 - Real.sqrt 3) / 2}
  let sol2 : Set ℝ := {-2, 1}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1219_121961


namespace NUMINAMATH_CALUDE_problem_statement_l1219_121913

theorem problem_statement (x y : ℤ) (hx : x = 1) (hy : y = 630) : 2019 * x - 3 * y - 9 = 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1219_121913


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l1219_121929

theorem similar_triangles_leg_length 
  (leg1 : ℝ) 
  (hyp1 : ℝ) 
  (hyp2 : ℝ) 
  (h1 : leg1 = 15) 
  (h2 : hyp1 = 17) 
  (h3 : hyp2 = 51) : 
  (leg1 * hyp2 / hyp1) = 45 :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l1219_121929


namespace NUMINAMATH_CALUDE_log_expression_equality_l1219_121900

theorem log_expression_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 + 8^(1/4) * 2^(1/4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1219_121900


namespace NUMINAMATH_CALUDE_ben_remaining_money_l1219_121943

def calculate_remaining_money (initial amount : ℕ) (cheque debtor_payment maintenance_cost : ℕ) : ℕ :=
  initial - cheque + debtor_payment - maintenance_cost

theorem ben_remaining_money :
  calculate_remaining_money 2000 600 800 1200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ben_remaining_money_l1219_121943


namespace NUMINAMATH_CALUDE_largest_difference_l1219_121965

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 2005^2006)
  (hB : B = 2005^2006)
  (hC : C = 2004 * 2005^2005)
  (hD : D = 3 * 2005^2005)
  (hE : E = 2005^2005)
  (hF : F = 2005^2004) :
  A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l1219_121965
