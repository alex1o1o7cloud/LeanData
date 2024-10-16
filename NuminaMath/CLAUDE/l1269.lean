import Mathlib

namespace NUMINAMATH_CALUDE_factor_implies_m_equals_one_l1269_126943

theorem factor_implies_m_equals_one (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 42 = (x + 6) * k) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_m_equals_one_l1269_126943


namespace NUMINAMATH_CALUDE_medianDivisionReassembly_l1269_126951

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents the result of dividing a triangle by its medians -/
structure DividedTriangle where
  original : Triangle
  parts : Finset Triangle

/-- Predicate to check if a set of triangles can form a single triangle -/
def canFormTriangle (ts : Finset Triangle) : Prop :=
  ∃ t : Triangle, t.area = (ts.sum (λ t' ↦ t'.area))

/-- Theorem stating that it's possible to form one triangle from the six resulting triangles -/
theorem medianDivisionReassembly (dt : DividedTriangle) : 
  dt.original.area > 0 → 
  dt.parts.card = 6 → 
  (∀ t ∈ dt.parts, t.area = dt.original.area / 6) → 
  canFormTriangle dt.parts :=
sorry

end NUMINAMATH_CALUDE_medianDivisionReassembly_l1269_126951


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1269_126938

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 - 2 / y)^(1/3 : ℝ) = -3 ∧ y = 1/16 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1269_126938


namespace NUMINAMATH_CALUDE_no_such_function_l1269_126919

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l1269_126919


namespace NUMINAMATH_CALUDE_smallest_AC_l1269_126982

/-- Represents a right triangle ABC with a point D on AC -/
structure RightTriangleWithPoint where
  AC : ℕ  -- Length of AC
  CD : ℕ  -- Length of CD
  bd_squared : ℕ  -- Square of length BD

/-- Defines the conditions for the right triangle and point D -/
def valid_triangle (t : RightTriangleWithPoint) : Prop :=
  t.AC > 0 ∧ t.CD > 0 ∧ t.CD < t.AC ∧ t.bd_squared = 36 ∧
  2 * t.AC * t.CD = t.CD * t.CD + t.bd_squared

/-- Theorem: The smallest possible value of AC is 6 -/
theorem smallest_AC :
  ∀ t : RightTriangleWithPoint, valid_triangle t → t.AC ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_AC_l1269_126982


namespace NUMINAMATH_CALUDE_wanda_blocks_calculation_l1269_126991

theorem wanda_blocks_calculation (initial_blocks : ℕ) (theresa_percentage : ℚ) (give_away_fraction : ℚ) : 
  initial_blocks = 2450 →
  theresa_percentage = 35 / 100 →
  give_away_fraction = 1 / 8 →
  (initial_blocks + Int.floor (theresa_percentage * initial_blocks) - 
   Int.floor (give_away_fraction * (initial_blocks + Int.floor (theresa_percentage * initial_blocks)))) = 2894 := by
  sorry

end NUMINAMATH_CALUDE_wanda_blocks_calculation_l1269_126991


namespace NUMINAMATH_CALUDE_volume_of_specific_cuboid_l1269_126940

/-- The volume of a cuboid formed by two identical cubes in a line --/
def cuboid_volume (edge_length : ℝ) : ℝ :=
  2 * (edge_length ^ 3)

/-- Theorem: The volume of a cuboid formed by two cubes with edge length 5 cm is 250 cm³ --/
theorem volume_of_specific_cuboid :
  cuboid_volume 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_cuboid_l1269_126940


namespace NUMINAMATH_CALUDE_min_values_xy_l1269_126920

theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9*x + y - x*y = 0) :
  (∃ (min_sum : ℝ), min_sum = 16 ∧ 
    (∀ (a b : ℝ), a > 0 → b > 0 → 9*a + b - a*b = 0 → a + b ≥ min_sum) ∧
    (x + y = min_sum ↔ x = 4 ∧ y = 12)) ∧
  (∃ (min_prod : ℝ), min_prod = 36 ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → 9*a + b - a*b = 0 → a*b ≥ min_prod) ∧
    (x*y = min_prod ↔ x = 2 ∧ y = 18)) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_l1269_126920


namespace NUMINAMATH_CALUDE_centroid_projections_sum_l1269_126915

/-- Given a triangle XYZ with sides of length 4, 3, and 5, 
    this theorem states that the sum of the distances from 
    the centroid to each side of the triangle is 47/15. -/
theorem centroid_projections_sum (X Y Z G : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d X Y = 4) → (d X Z = 3) → (d Y Z = 5) →
  (G.1 = (X.1 + Y.1 + Z.1) / 3) → (G.2 = (X.2 + Y.2 + Z.2) / 3) →
  let dist_point_to_line := λ p a b : ℝ × ℝ => 
    |((b.2 - a.2) * p.1 - (b.1 - a.1) * p.2 + b.1 * a.2 - b.2 * a.1) / d a b|
  (dist_point_to_line G Y Z + dist_point_to_line G X Z + dist_point_to_line G X Y = 47/15) := by
sorry

end NUMINAMATH_CALUDE_centroid_projections_sum_l1269_126915


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1269_126927

/-- Given a rectangle with perimeter 30 inches and one side 3 inches longer than the other,
    the maximum possible area is 54 square inches. -/
theorem rectangle_max_area :
  ∀ x : ℝ,
  x > 0 →
  2 * (x + (x + 3)) = 30 →
  x * (x + 3) ≤ 54 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1269_126927


namespace NUMINAMATH_CALUDE_white_smallest_probability_l1269_126984

def total_balls : ℕ := 16
def red_balls : ℕ := 9
def black_balls : ℕ := 5
def white_balls : ℕ := 2

theorem white_smallest_probability :
  (white_balls : ℚ) / total_balls < (red_balls : ℚ) / total_balls ∧
  (white_balls : ℚ) / total_balls < (black_balls : ℚ) / total_balls :=
by sorry

end NUMINAMATH_CALUDE_white_smallest_probability_l1269_126984


namespace NUMINAMATH_CALUDE_eighty_factorial_zeroes_l1269_126957

/-- Count the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

theorem eighty_factorial_zeroes :
  trailingZeroes 73 = 16 → trailingZeroes 80 = 19 := by sorry

end NUMINAMATH_CALUDE_eighty_factorial_zeroes_l1269_126957


namespace NUMINAMATH_CALUDE_order_of_operations_l1269_126968

theorem order_of_operations (a b c : ℕ) : a - b * c = a - (b * c) := by
  sorry

end NUMINAMATH_CALUDE_order_of_operations_l1269_126968


namespace NUMINAMATH_CALUDE_impossible_cube_permutation_l1269_126907

/-- Represents a position in the 3x3x3 cube -/
structure Position :=
  (x y z : Fin 3)

/-- Represents a labeling of the 27 unit cubes -/
def Labeling := Fin 27 → Position

/-- Represents a move: swapping cube 27 with a neighbor -/
inductive Move
  | swap : Position → Move

/-- The parity of a position (even sum of coordinates is black, odd is white) -/
def Position.parity (p : Position) : Bool :=
  (p.x + p.y + p.z) % 2 = 0

/-- The final permutation required by the problem -/
def finalPermutation (n : Fin 27) : Fin 27 :=
  if n = 27 then 27 else 27 - n

/-- Theorem stating the impossibility of the required sequence of moves -/
theorem impossible_cube_permutation (initial : Labeling) :
  ¬ ∃ (moves : List Move), 
    (∀ n : Fin 27, 
      (initial n).parity = (initial (finalPermutation n)).parity) ∧
    (moves.length % 2 = 0) :=
  sorry

end NUMINAMATH_CALUDE_impossible_cube_permutation_l1269_126907


namespace NUMINAMATH_CALUDE_chessboard_dark_light_difference_l1269_126903

/-- Represents a square on the chessboard -/
inductive Square
| Dark
| Light

/-- Represents a row on the chessboard -/
def Row := Vector Square 9

/-- Generates a row starting with the given square color -/
def generateRow (startSquare : Square) : Row := sorry

/-- The chessboard, consisting of 9 rows -/
def Chessboard := Vector Row 9

/-- Generates the chessboard with alternating row starts -/
def generateChessboard : Chessboard := sorry

/-- Counts the number of dark squares in a row -/
def countDarkSquares (row : Row) : Nat := sorry

/-- Counts the number of light squares in a row -/
def countLightSquares (row : Row) : Nat := sorry

/-- Counts the total number of dark squares on the chessboard -/
def totalDarkSquares (board : Chessboard) : Nat := sorry

/-- Counts the total number of light squares on the chessboard -/
def totalLightSquares (board : Chessboard) : Nat := sorry

theorem chessboard_dark_light_difference :
  let board := generateChessboard
  totalDarkSquares board = totalLightSquares board + 1 := by sorry

end NUMINAMATH_CALUDE_chessboard_dark_light_difference_l1269_126903


namespace NUMINAMATH_CALUDE_prime_in_sequence_l1269_126923

theorem prime_in_sequence (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∃ n : ℕ, p = Int.sqrt (24 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_in_sequence_l1269_126923


namespace NUMINAMATH_CALUDE_position_of_2010_l1269_126993

/-- The sum of the first n terms of the arithmetic sequence representing the number of integers in each row -/
def rowSum (n : ℕ) : ℕ := n^2

/-- The first number in the nth row -/
def firstInRow (n : ℕ) : ℕ := rowSum (n - 1) + 1

/-- The position of a number in the table -/
structure Position where
  row : ℕ
  column : ℕ

/-- Find the position of a number in the table -/
def findPosition (num : ℕ) : Position :=
  let row := (Nat.sqrt (num - 1) + 1)
  let column := num - firstInRow row + 1
  ⟨row, column⟩

theorem position_of_2010 : findPosition 2010 = ⟨45, 74⟩ := by
  sorry

end NUMINAMATH_CALUDE_position_of_2010_l1269_126993


namespace NUMINAMATH_CALUDE_grapes_per_day_calculation_l1269_126901

/-- The number of pickers -/
def num_pickers : ℕ := 235

/-- The number of drums of raspberries filled per day -/
def raspberries_per_day : ℕ := 100

/-- The number of days -/
def num_days : ℕ := 77

/-- The total number of drums filled in 77 days -/
def total_drums : ℕ := 17017

/-- The number of drums of grapes filled per day -/
def grapes_per_day : ℕ := 121

theorem grapes_per_day_calculation :
  grapes_per_day = (total_drums - raspberries_per_day * num_days) / num_days :=
by sorry

end NUMINAMATH_CALUDE_grapes_per_day_calculation_l1269_126901


namespace NUMINAMATH_CALUDE_cosine_graph_minimum_l1269_126974

theorem cosine_graph_minimum (c : ℝ) (h1 : c > 0) : 
  (∀ x : ℝ, 3 * Real.cos (5 * x + c) ≥ 3 * Real.cos c) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo 0 ε, 3 * Real.cos (5 * x + c) > 3 * Real.cos c) → 
  c = Real.pi := by
sorry

end NUMINAMATH_CALUDE_cosine_graph_minimum_l1269_126974


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1269_126939

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1269_126939


namespace NUMINAMATH_CALUDE_johns_final_push_time_l1269_126928

/-- The time of John's final push in a race, given specific conditions --/
theorem johns_final_push_time (john_initial_lag : ℝ) (john_speed : ℝ) (steve_speed : ℝ) (john_final_lead : ℝ)
  (h1 : john_initial_lag = 15)
  (h2 : john_speed = 4.2)
  (h3 : steve_speed = 3.7)
  (h4 : john_final_lead = 2) :
  (john_initial_lag + john_final_lead) / john_speed = 17 / 4.2 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_push_time_l1269_126928


namespace NUMINAMATH_CALUDE_log_inequality_equiv_interval_l1269_126934

-- Define the logarithm function with base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_inequality_equiv_interval (x : ℝ) :
  (log2 (4 - x) > log2 (3 * x)) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_interval_l1269_126934


namespace NUMINAMATH_CALUDE_partition_exists_l1269_126976

/-- The set of weights from 1 to 101 grams -/
def weights : Finset ℕ := Finset.range 101

/-- The sum of all weights from 1 to 101 grams -/
def total_sum : ℕ := weights.sum id

/-- The remaining weights after removing the 19-gram weight -/
def remaining_weights : Finset ℕ := weights.erase 19

/-- The sum of remaining weights -/
def remaining_sum : ℕ := remaining_weights.sum id

/-- A partition of the remaining weights into two subsets -/
structure Partition :=
  (subset1 subset2 : Finset ℕ)
  (partition_complete : subset1 ∪ subset2 = remaining_weights)
  (partition_disjoint : subset1 ∩ subset2 = ∅)
  (equal_size : subset1.card = subset2.card)
  (size_fifty : subset1.card = 50)

/-- The theorem stating that a valid partition exists -/
theorem partition_exists : ∃ (p : Partition), p.subset1.sum id = p.subset2.sum id :=
sorry

end NUMINAMATH_CALUDE_partition_exists_l1269_126976


namespace NUMINAMATH_CALUDE_batsman_average_l1269_126929

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℚ) : 
  total_innings = 25 →
  last_innings_score = 95 →
  average_increase = 5/2 →
  (∃ (previous_average : ℚ), 
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings = previous_average + average_increase) →
  (∃ (final_average : ℚ), final_average = 35) := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l1269_126929


namespace NUMINAMATH_CALUDE_sqrt_pattern_l1269_126947

theorem sqrt_pattern (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 + (2 * n - 1) / (n^2 : ℝ)) = (n + 1 : ℝ) / n :=
by sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l1269_126947


namespace NUMINAMATH_CALUDE_function_satisfying_equation_l1269_126941

theorem function_satisfying_equation (f : ℕ → ℕ) : 
  (∀ (a b c : ℕ), a ≥ 2 → b ≥ 2 → c ≥ 2 → 
    (f^[a*b*c - a] (a*b*c) + f^[a*b*c - b] (a*b*c) + f^[a*b*c - c] (a*b*c) = a + b + c)) →
  (∀ n : ℕ, n ≥ 3 → f n = n - 1) ∧ 
  (∀ m : ℕ, m = 1 ∨ m = 2 → True) :=
by sorry

#check function_satisfying_equation

end NUMINAMATH_CALUDE_function_satisfying_equation_l1269_126941


namespace NUMINAMATH_CALUDE_edward_money_proof_l1269_126918

/-- The amount of money Edward had before spending, given his expenses and remaining money. -/
def edward_initial_money (books_cost pens_cost remaining : ℕ) : ℕ :=
  books_cost + pens_cost + remaining

/-- Theorem stating that Edward's initial money was $41 given the problem conditions. -/
theorem edward_money_proof :
  edward_initial_money 6 16 19 = 41 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_proof_l1269_126918


namespace NUMINAMATH_CALUDE_wire_service_reporters_theorem_l1269_126956

/-- Represents the percentage of reporters in a wire service -/
structure ReporterPercentage where
  local_politics : Real
  not_politics : Real
  politics_not_local : Real

/-- Given the percentages of reporters covering local politics and not covering politics,
    calculates the percentage of reporters covering politics but not local politics -/
def calculate_politics_not_local (rp : ReporterPercentage) : Real :=
  100 - rp.not_politics - rp.local_politics

/-- Theorem stating that given the specific percentages in the problem,
    the percentage of reporters covering politics but not local politics is 2.14285714285714% -/
theorem wire_service_reporters_theorem (rp : ReporterPercentage)
  (h1 : rp.local_politics = 5)
  (h2 : rp.not_politics = 92.85714285714286) :
  calculate_politics_not_local rp = 2.14285714285714 := by
  sorry

#eval calculate_politics_not_local { local_politics := 5, not_politics := 92.85714285714286, politics_not_local := 0 }

end NUMINAMATH_CALUDE_wire_service_reporters_theorem_l1269_126956


namespace NUMINAMATH_CALUDE_factor_expression_l1269_126936

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1269_126936


namespace NUMINAMATH_CALUDE_circular_garden_ratio_l1269_126931

theorem circular_garden_ratio (r : ℝ) (h : r = 10) : 
  (2 * π * r) / (π * r^2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_ratio_l1269_126931


namespace NUMINAMATH_CALUDE_new_books_bought_l1269_126980

/-- Given Kaleb's initial number of books, the number of books he sold, and his final number of books,
    prove that the number of new books he bought is equal to the difference between his final number
    of books and the number of books he had after selling some. -/
theorem new_books_bought (initial_books sold_books final_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  final_books = 24 →
  final_books - (initial_books - sold_books) = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_books_bought_l1269_126980


namespace NUMINAMATH_CALUDE_range_of_m_l1269_126965

/-- Condition p: |1 - (x-1)/3| < 2 -/
def p (x : ℝ) : Prop := |1 - (x-1)/3| < 2

/-- Condition q: (x-1)^2 < m^2 -/
def q (x m : ℝ) : Prop := (x-1)^2 < m^2

/-- q is a sufficient condition for p -/
def q_sufficient (m : ℝ) : Prop := ∀ x, q x m → p x

/-- q is not a necessary condition for p -/
def q_not_necessary (m : ℝ) : Prop := ∃ x, p x ∧ ¬q x m

theorem range_of_m :
  (∀ m, q_sufficient m ∧ q_not_necessary m) →
  (∀ m, m ∈ Set.Icc (-3 : ℝ) 3) ∧ 
  (∃ m₁ m₂, m₁ ∈ Set.Ioo (-3 : ℝ) 3 ∧ m₂ ∈ Set.Ioo (-3 : ℝ) 3 ∧ m₁ ≠ m₂) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1269_126965


namespace NUMINAMATH_CALUDE_probability_one_of_each_type_l1269_126913

def total_silverware : ℕ := 30
def forks : ℕ := 10
def spoons : ℕ := 10
def knives : ℕ := 10

theorem probability_one_of_each_type (total_silverware forks spoons knives : ℕ) :
  total_silverware = forks + spoons + knives →
  (Nat.choose total_silverware 3 : ℚ) ≠ 0 →
  (forks * spoons * knives : ℚ) / Nat.choose total_silverware 3 = 500 / 203 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_type_l1269_126913


namespace NUMINAMATH_CALUDE_b_k_divisible_by_9_count_l1269_126917

/-- The sequence b_n is defined as the number obtained by concatenating
    integers from 1 to n and subtracting n -/
def b (n : ℕ) : ℕ := sorry

/-- g(n) represents the sum of digits of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of b_k divisible by 9 for 1 ≤ k ≤ 100 -/
def count_divisible_by_9 : ℕ := sorry

theorem b_k_divisible_by_9_count :
  count_divisible_by_9 = 22 := by sorry

end NUMINAMATH_CALUDE_b_k_divisible_by_9_count_l1269_126917


namespace NUMINAMATH_CALUDE_binomial_coefficient_1999000_l1269_126987

theorem binomial_coefficient_1999000 :
  ∀ x : ℕ+, (∃ y : ℕ+, Nat.choose x.val y.val = 1999000) ↔ (x.val = 1999000 ∨ x.val = 2000) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1999000_l1269_126987


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1269_126967

theorem x_squared_plus_reciprocal (x : ℝ) (h : 35 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1269_126967


namespace NUMINAMATH_CALUDE_teal_more_blue_l1269_126979

/-- The number of people surveyed -/
def total_surveyed : ℕ := 150

/-- The number of people who believe teal is "more green" -/
def more_green : ℕ := 80

/-- The number of people who believe teal is both "more green" and "more blue" -/
def both : ℕ := 40

/-- The number of people who think teal is neither "more green" nor "more blue" -/
def neither : ℕ := 20

/-- The number of people who believe teal is "more blue" -/
def more_blue : ℕ := total_surveyed - (more_green - both) - both - neither

theorem teal_more_blue : more_blue = 90 := by
  sorry

end NUMINAMATH_CALUDE_teal_more_blue_l1269_126979


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l1269_126970

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- A circle with center at the origin and radius 2 -/
def Circle := {p : Point | p.x^2 + p.y^2 ≤ 4}

/-- The endpoints of a diameter of the circle -/
def diameterEndpoints : (Point × Point) :=
  ({x := -2, y := 0}, {x := 2, y := 0})

/-- The condition for a point P to satisfy the sum of squares property -/
def satisfiesSumOfSquares (p : Point) : Prop :=
  let (a, b) := diameterEndpoints
  distanceSquared p a + distanceSquared p b = 8

/-- The set of points satisfying the condition -/
def SatisfyingPoints : Set Point :=
  {p ∈ Circle | satisfiesSumOfSquares p}

theorem infinitely_many_satisfying_points :
  Set.Infinite SatisfyingPoints :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l1269_126970


namespace NUMINAMATH_CALUDE_probability_at_least_two_correct_l1269_126945

theorem probability_at_least_two_correct (n : ℕ) (p : ℚ) : 
  n = 6 → p = 1/6 → 
  1 - (Nat.choose n 0 * p^0 * (1-p)^n + Nat.choose n 1 * p^1 * (1-p)^(n-1)) = 34369/58420 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_correct_l1269_126945


namespace NUMINAMATH_CALUDE_certain_amount_proof_l1269_126986

theorem certain_amount_proof (x : ℝ) (A : ℝ) (h1 : x = 840) (h2 : 0.25 * x = 0.15 * 1500 - A) : A = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l1269_126986


namespace NUMINAMATH_CALUDE_binomial_expansion_max_term_max_term_for_sqrt11_expansion_l1269_126959

theorem binomial_expansion_max_term (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

theorem max_term_for_sqrt11_expansion :
  let n : ℕ := 208
  let x : ℝ := Real.sqrt 11
  ∃ k : ℕ, k = 160 ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_max_term_max_term_for_sqrt11_expansion_l1269_126959


namespace NUMINAMATH_CALUDE_triangle_ratio_l1269_126906

/-- Given an acute triangle ABC with a point D inside it, 
    if ∠ADB = ∠ACB + 90° and AC * BD = AD * BC, 
    then (AB * CD) / (AC * BD) = √2 -/
theorem triangle_ratio (A B C D : ℂ) : 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧  -- A, B, C form a triangle
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = t*B + (1-t)*C) ∧  -- D is inside triangle ABC
  Complex.arg ((D - B) / (D - A)) = Complex.arg ((C - B) / (C - A)) + Real.pi / 2 ∧  -- ∠ADB = ∠ACB + 90°
  Complex.abs (C - A) * Complex.abs (D - B) = Complex.abs (D - A) * Complex.abs (C - B) →  -- AC * BD = AD * BC
  Complex.abs ((B - A) * (D - C)) / (Complex.abs (C - A) * Complex.abs (D - B)) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1269_126906


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1269_126981

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) (h : isGeometric a) :
  a 4 * a 6 * a 8 * a 10 * a 12 = 32 → a 10^2 / a 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1269_126981


namespace NUMINAMATH_CALUDE_complement_of_67_is_23_l1269_126910

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := by sorry

end NUMINAMATH_CALUDE_complement_of_67_is_23_l1269_126910


namespace NUMINAMATH_CALUDE_power_function_increasing_condition_l1269_126932

theorem power_function_increasing_condition (m : ℝ) : 
  (m^2 - m - 1 = 1) ∧ (m^2 + m - 3 > 0) ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_increasing_condition_l1269_126932


namespace NUMINAMATH_CALUDE_odd_factors_of_252_l1269_126950

def number_of_odd_factors (n : ℕ) : ℕ := sorry

theorem odd_factors_of_252 : number_of_odd_factors 252 = 6 := by sorry

end NUMINAMATH_CALUDE_odd_factors_of_252_l1269_126950


namespace NUMINAMATH_CALUDE_bank_account_withdrawal_l1269_126975

theorem bank_account_withdrawal (initial_balance deposit1 deposit2 final_balance_increase : ℕ) :
  initial_balance = 150 →
  deposit1 = 17 →
  deposit2 = 21 →
  final_balance_increase = 16 →
  ∃ withdrawal : ℕ, 
    initial_balance + deposit1 - withdrawal + deposit2 = initial_balance + final_balance_increase ∧
    withdrawal = 22 :=
by sorry

end NUMINAMATH_CALUDE_bank_account_withdrawal_l1269_126975


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1269_126926

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x - 2*y = 8 ∧ x = -2 ∧ y = -7) ∧
  -- System 2
  (∃ x y : ℝ, 3*x + 4*y = 5 ∧ 5*x - 2*y = 30 ∧ x = 5 ∧ y = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1269_126926


namespace NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l1269_126988

theorem derivative_zero_at_negative_one (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 - 4) * (x - t)
  let f' : ℝ → ℝ := λ x ↦ 2*x*(x - t) + (x^2 - 4)
  f' (-1) = 0 → t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l1269_126988


namespace NUMINAMATH_CALUDE_present_age_ratio_l1269_126911

theorem present_age_ratio (R M : ℝ) (h1 : M - R = 7.5) (h2 : (R + 10) / (M + 10) = 2 / 3) 
  (h3 : R > 0) (h4 : M > 0) : R / M = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_present_age_ratio_l1269_126911


namespace NUMINAMATH_CALUDE_mountain_distance_l1269_126954

/-- Represents the mountain climbing scenario -/
structure MountainClimb where
  /-- Distance from bottom to top of the mountain in meters -/
  total_distance : ℝ
  /-- A's ascending speed in meters per hour -/
  speed_a_up : ℝ
  /-- B's ascending speed in meters per hour -/
  speed_b_up : ℝ
  /-- Distance from top where A and B meet in meters -/
  meeting_point : ℝ
  /-- Assumption that descending speed is 3 times ascending speed -/
  descent_speed_multiplier : ℝ
  /-- Assumption that A reaches bottom when B is halfway down -/
  b_halfway_when_a_bottom : Bool

/-- Main theorem: The distance from bottom to top is 1550 meters -/
theorem mountain_distance (climb : MountainClimb) 
  (h1 : climb.meeting_point = 150)
  (h2 : climb.descent_speed_multiplier = 3)
  (h3 : climb.b_halfway_when_a_bottom = true) :
  climb.total_distance = 1550 := by
  sorry

end NUMINAMATH_CALUDE_mountain_distance_l1269_126954


namespace NUMINAMATH_CALUDE_playground_total_l1269_126990

/-- The number of people on a playground --/
structure Playground where
  girls : ℕ
  boys : ℕ
  thirdGradeGirls : ℕ
  thirdGradeBoys : ℕ
  teachers : ℕ
  maleTeachers : ℕ
  femaleTeachers : ℕ

/-- The total number of people on the playground is 67 --/
theorem playground_total (p : Playground)
  (h1 : p.girls = 28)
  (h2 : p.boys = 35)
  (h3 : p.thirdGradeGirls = 15)
  (h4 : p.thirdGradeBoys = 18)
  (h5 : p.teachers = 4)
  (h6 : p.maleTeachers = 2)
  (h7 : p.femaleTeachers = 2)
  (h8 : p.teachers = p.maleTeachers + p.femaleTeachers) :
  p.girls + p.boys + p.teachers = 67 := by
  sorry

#check playground_total

end NUMINAMATH_CALUDE_playground_total_l1269_126990


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1269_126960

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1269_126960


namespace NUMINAMATH_CALUDE_english_only_students_l1269_126973

theorem english_only_students (total : Nat) (max_liz : Nat) (english : Nat) (french : Nat) : 
  total = 25 → 
  max_liz = 2 → 
  total = english + french - max_liz → 
  english = 2 * french → 
  english - french = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_english_only_students_l1269_126973


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l1269_126914

theorem reciprocal_of_negative_one_point_five :
  ((-1.5)⁻¹ : ℝ) = -2/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l1269_126914


namespace NUMINAMATH_CALUDE_congruence_solution_l1269_126994

theorem congruence_solution (n : ℤ) : 
  4 ≤ n ∧ n ≤ 10 ∧ n ≡ 11783 [ZMOD 7] → n = 5 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1269_126994


namespace NUMINAMATH_CALUDE_square_difference_l1269_126962

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : 
  (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1269_126962


namespace NUMINAMATH_CALUDE_arctan_of_tan_difference_l1269_126949

-- Define the problem parameters
def angle₁ : Real := 80
def angle₂ : Real := 30

-- Define the theorem
theorem arctan_of_tan_difference (h : 0 ≤ angle₁ ∧ angle₁ ≤ 180 ∧ 0 ≤ angle₂ ∧ angle₂ ≤ 180) :
  Real.arctan (Real.tan (angle₁ * π / 180) - 3 * Real.tan (angle₂ * π / 180)) * 180 / π = angle₁ := by
  sorry


end NUMINAMATH_CALUDE_arctan_of_tan_difference_l1269_126949


namespace NUMINAMATH_CALUDE_ab_greater_than_e_squared_l1269_126997

theorem ab_greater_than_e_squared (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : e = Real.exp 1) (h4 : a^b = b^a) : a * b > e^2 := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_e_squared_l1269_126997


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1269_126912

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def nth_term (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_nth_term 
  (x : ℝ) (n : ℕ) 
  (h1 : arithmetic_sequence (3*x - 4) (7*x - 14) (4*x + 5))
  (h2 : ∃ (a d : ℝ), nth_term a d n = 4013 ∧ a = 3*x - 4 ∧ d = (7*x - 14) - (3*x - 4)) :
  n = 610 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1269_126912


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l1269_126925

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l1269_126925


namespace NUMINAMATH_CALUDE_number_of_towns_l1269_126955

theorem number_of_towns (n : ℕ) : Nat.choose n 2 = 15 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_towns_l1269_126955


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1269_126900

-- Define the perimeter of square A
def perimeterA : ℝ := 36

-- Define the relationship between areas of square A and B
def areaRelation (areaA areaB : ℝ) : Prop := areaB = areaA / 3

-- State the theorem
theorem square_perimeter_relation (sideA sideB : ℝ) 
  (h1 : sideA * 4 = perimeterA)
  (h2 : areaRelation (sideA * sideA) (sideB * sideB)) :
  4 * sideB = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1269_126900


namespace NUMINAMATH_CALUDE_triangle_midpoint_vector_l1269_126964

/-- Given a triangle ABC with vertices A(-1, 0), B(0, 2), and C(2, 0),
    and D is the midpoint of BC, prove that vector AD equals (2, 1) -/
theorem triangle_midpoint_vector (A B C D : ℝ × ℝ) : 
  A = (-1, 0) → B = (0, 2) → C = (2, 0) → D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (D.1 - A.1, D.2 - A.2) = (2, 1) := by
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_vector_l1269_126964


namespace NUMINAMATH_CALUDE_prob_both_3_l1269_126999

-- Define the number of sides for each die
def die1_sides : ℕ := 6
def die2_sides : ℕ := 7

-- Define the probability of rolling a 3 on each die
def prob_3_die1 : ℚ := 1 / die1_sides
def prob_3_die2 : ℚ := 1 / die2_sides

-- Theorem: The probability of rolling a 3 on both dice is 1/42
theorem prob_both_3 : prob_3_die1 * prob_3_die2 = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_3_l1269_126999


namespace NUMINAMATH_CALUDE_expression_values_l1269_126937

theorem expression_values (a b : ℝ) (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 3 ∨ (3 * a - b) / (a + 5 * b) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l1269_126937


namespace NUMINAMATH_CALUDE_proportion_sum_l1269_126977

theorem proportion_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 84 / Q → P + Q = 175 := by
  sorry

end NUMINAMATH_CALUDE_proportion_sum_l1269_126977


namespace NUMINAMATH_CALUDE_intersection_singleton_iff_a_in_range_l1269_126969

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * |p.1|}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + a}

theorem intersection_singleton_iff_a_in_range (a : ℝ) :
  (∃! p : ℝ × ℝ, p ∈ set_A a ∩ set_B a) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_singleton_iff_a_in_range_l1269_126969


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l1269_126933

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 0.2

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 720 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l1269_126933


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1269_126966

theorem sqrt_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 / y) + Real.sqrt (y^2 / x) ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1269_126966


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1269_126953

theorem sum_remainder_mod_seven (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2*n) % 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1269_126953


namespace NUMINAMATH_CALUDE_percentage_decrease_l1269_126998

/-- Given a percentage increase P in production value from one year to the next,
    calculate the percentage decrease from the latter year to the former year. -/
theorem percentage_decrease (P : ℝ) : 
  P > -100 → (100 * (1 - 1 / (1 + P / 100))) = P / (1 + P / 100) := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l1269_126998


namespace NUMINAMATH_CALUDE_equation_solution_l1269_126944

theorem equation_solution : 
  ∃ x : ℝ, ((0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + 0.052^2 + x^2) = 100) ∧ x = 0.0035 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1269_126944


namespace NUMINAMATH_CALUDE_product_of_numbers_l1269_126995

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 12) (h3 : x/y = 3/2) :
  x * y = 1244.16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1269_126995


namespace NUMINAMATH_CALUDE_sandbox_perimeter_l1269_126958

/-- The perimeter of a rectangular sandbox with width 5 feet and length twice the width is 30 feet. -/
theorem sandbox_perimeter : 
  ∀ (width length perimeter : ℝ), 
  width = 5 → 
  length = 2 * width → 
  perimeter = 2 * (length + width) → 
  perimeter = 30 := by sorry

end NUMINAMATH_CALUDE_sandbox_perimeter_l1269_126958


namespace NUMINAMATH_CALUDE_range_of_q_l1269_126963

-- Define the function q(x)
def q (x : ℝ) : ℝ := (x^2 + 2)^3

-- State the theorem
theorem range_of_q : 
  {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ q x = y} = {y : ℝ | y ≥ 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_q_l1269_126963


namespace NUMINAMATH_CALUDE_correct_average_weight_l1269_126935

/-- Given a class of 20 boys with an initial average weight and a misread weight,
    calculate the correct average weight. -/
theorem correct_average_weight
  (num_boys : ℕ)
  (initial_avg : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : num_boys = 20)
  (h2 : initial_avg = 58.4)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 62) :
  (num_boys : ℝ) * initial_avg + (correct_weight - misread_weight) = num_boys * 58.7 :=
by sorry

#check correct_average_weight

end NUMINAMATH_CALUDE_correct_average_weight_l1269_126935


namespace NUMINAMATH_CALUDE_couples_satisfy_handshake_equation_l1269_126909

/-- The number of couples at a gathering where each person shakes hands with everyone
    except themselves and their partner, resulting in a total of 31,000 handshakes. -/
def num_couples : ℕ := 125

/-- The total number of handshakes at the gathering. -/
def total_handshakes : ℕ := 31000

/-- Theorem stating that the number of couples satisfies the equation derived from
    the handshake conditions. -/
theorem couples_satisfy_handshake_equation :
  2 * (num_couples * num_couples) - 2 * num_couples = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_couples_satisfy_handshake_equation_l1269_126909


namespace NUMINAMATH_CALUDE_point_outside_circle_l1269_126948

theorem point_outside_circle (r OA : ℝ) (h1 : r = 3) (h2 : OA = 5) :
  OA > r := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1269_126948


namespace NUMINAMATH_CALUDE_horner_rule_equality_f_at_two_equals_62_l1269_126961

/-- Horner's Rule representation of a polynomial -/
def horner_form (a b c d e : ℝ) (x : ℝ) : ℝ :=
  x * (x * (x * (a * x + b) + c) + d) + e

/-- Original polynomial function -/
def f (x : ℝ) : ℝ :=
  2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_equality :
  ∀ x : ℝ, f x = horner_form 2 3 0 5 (-4) x :=
sorry

theorem f_at_two_equals_62 : f 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_equality_f_at_two_equals_62_l1269_126961


namespace NUMINAMATH_CALUDE_andreas_erasers_l1269_126916

theorem andreas_erasers (andrea_erasers : ℕ) : 
  (4 * andrea_erasers = andrea_erasers + 12) → andrea_erasers = 4 := by
  sorry

end NUMINAMATH_CALUDE_andreas_erasers_l1269_126916


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l1269_126972

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The number of trees planted by 6th graders -/
def trees_6th : ℕ := 3 * trees_5th - 30

/-- The total number of trees planted by all three grades -/
def total_trees : ℕ := trees_4th + trees_5th + trees_6th

theorem tree_planting_theorem : total_trees = 240 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l1269_126972


namespace NUMINAMATH_CALUDE_arithmetic_sequence_40th_term_l1269_126908

/-- Given an arithmetic sequence where the first term is 3 and the twentieth term is 63,
    prove that the fortieth term is 126. -/
theorem arithmetic_sequence_40th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                                 -- first term is 3
    a 19 = 63 →                               -- twentieth term is 63
    a 39 = 126 := by                          -- fortieth term is 126
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_40th_term_l1269_126908


namespace NUMINAMATH_CALUDE_inequality_range_l1269_126902

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ a ≤ -1 ∨ a ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1269_126902


namespace NUMINAMATH_CALUDE_unique_m_solution_l1269_126904

theorem unique_m_solution : 
  ∀ m : ℕ+, 
  (∃ a b c : ℕ+, (a.val * b.val * c.val * m.val : ℕ) = 1 + a.val^2 + b.val^2 + c.val^2) ↔ 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_m_solution_l1269_126904


namespace NUMINAMATH_CALUDE_urn_problem_l1269_126978

theorem urn_problem (N : ℕ) : 
  (6 : ℝ) / 10 * 10 / (10 + N) + (4 : ℝ) / 10 * N / (10 + N) = 1 / 2 → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l1269_126978


namespace NUMINAMATH_CALUDE_third_factor_proof_l1269_126952

theorem third_factor_proof (w : ℕ) (h1 : w = 168) (h2 : 2^5 ∣ (936 * w)) (h3 : 3^3 ∣ (936 * w)) :
  (936 * w) / (2^5 * 3^3) = 182 := by
  sorry

end NUMINAMATH_CALUDE_third_factor_proof_l1269_126952


namespace NUMINAMATH_CALUDE_mityas_age_l1269_126992

theorem mityas_age (shura_age mitya_age : ℚ) : 
  (mitya_age = shura_age + 11) →
  (mitya_age - shura_age = 2 * (shura_age - (mitya_age - shura_age))) →
  mitya_age = 27.5 := by sorry

end NUMINAMATH_CALUDE_mityas_age_l1269_126992


namespace NUMINAMATH_CALUDE_square_difference_l1269_126996

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1269_126996


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l1269_126971

/-- Calculates the intensity of a paint mixture -/
def mixturePaintIntensity (originalIntensity : ℚ) (addedIntensity : ℚ) (replacedFraction : ℚ) : ℚ :=
  (1 - replacedFraction) * originalIntensity + replacedFraction * addedIntensity

/-- Theorem stating that mixing 50% intensity paint with 20% intensity paint in a 2:1 ratio results in 40% intensity -/
theorem paint_mixture_intensity :
  mixturePaintIntensity (1/2) (1/5) (1/3) = (2/5) := by
  sorry

#eval mixturePaintIntensity (1/2) (1/5) (1/3)

end NUMINAMATH_CALUDE_paint_mixture_intensity_l1269_126971


namespace NUMINAMATH_CALUDE_water_needed_for_solution_l1269_126922

theorem water_needed_for_solution (total_volume : ℝ) (water_ratio : ℝ) (desired_volume : ℝ) :
  water_ratio = 1/3 →
  desired_volume = 0.48 →
  water_ratio * desired_volume = 0.16 :=
by sorry

end NUMINAMATH_CALUDE_water_needed_for_solution_l1269_126922


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l1269_126985

-- Define the function f
def f (x : ℝ) : ℝ := x^6 + x^2 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l1269_126985


namespace NUMINAMATH_CALUDE_total_paintable_area_l1269_126946

/-- Calculate the total square feet of walls to be painted in bedrooms and hallway -/
theorem total_paintable_area (
  num_bedrooms : ℕ)
  (bedroom_length bedroom_width bedroom_height : ℝ)
  (hallway_length hallway_width hallway_height : ℝ)
  (unpaintable_area_per_bedroom : ℝ)
  (h1 : num_bedrooms = 4)
  (h2 : bedroom_length = 14)
  (h3 : bedroom_width = 11)
  (h4 : bedroom_height = 9)
  (h5 : hallway_length = 20)
  (h6 : hallway_width = 7)
  (h7 : hallway_height = 9)
  (h8 : unpaintable_area_per_bedroom = 70) :
  (num_bedrooms * (2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area_per_bedroom)) +
  (2 * (hallway_length * hallway_height + hallway_width * hallway_height)) = 2006 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l1269_126946


namespace NUMINAMATH_CALUDE_remove_one_for_avg_eight_point_five_l1269_126905

theorem remove_one_for_avg_eight_point_five (n : Nat) (h : n = 15) :
  let list := List.range n
  let sum := n * (n + 1) / 2
  let removed := 1
  let remaining_sum := sum - removed
  let remaining_count := n - 1
  (remaining_sum : ℚ) / remaining_count = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_for_avg_eight_point_five_l1269_126905


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1269_126921

theorem right_triangle_hypotenuse (shorter_leg : ℝ) (longer_leg : ℝ) (area : ℝ) :
  shorter_leg > 0 →
  longer_leg = 3 * shorter_leg - 3 →
  area = (1 / 2) * shorter_leg * longer_leg →
  area = 84 →
  (shorter_leg ^ 2 + longer_leg ^ 2).sqrt = Real.sqrt 505 := by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1269_126921


namespace NUMINAMATH_CALUDE_base_sum_problem_l1269_126989

theorem base_sum_problem (G₁ G₂ : ℚ) : ∃! (S₁ S₂ : ℕ+),
  (G₁ = (4 * S₁ + 8) / (S₁^2 - 1) ∧ G₁ = (3 * S₂ + 6) / (S₂^2 - 1)) ∧
  (G₂ = (8 * S₁ + 4) / (S₁^2 - 1) ∧ G₂ = (6 * S₂ + 3) / (S₂^2 - 1)) ∧
  S₁ + S₂ = 23 := by
  sorry

end NUMINAMATH_CALUDE_base_sum_problem_l1269_126989


namespace NUMINAMATH_CALUDE_consecutive_primes_integral_roots_properties_l1269_126930

-- Define consecutive primes
def consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

-- Define the quadratic equation with integral roots
def has_integral_roots (p q : ℕ) : Prop :=
  ∃ x y : ℤ, x^2 - (p + q : ℤ) * x + (p * q : ℤ) = 0 ∧
             y^2 - (p + q : ℤ) * y + (p * q : ℤ) = 0 ∧
             x ≠ y

theorem consecutive_primes_integral_roots_properties
  (p q : ℕ) (h1 : consecutive_primes p q) (h2 : has_integral_roots p q) :
  (∃ x y : ℤ, x + y = p + q ∧ Even (x + y)) ∧  -- Sum of roots is even
  (∀ x : ℤ, x^2 - (p + q : ℤ) * x + (p * q : ℤ) = 0 → x ≥ p) ∧  -- Each root ≥ p
  ¬Nat.Prime (p + q) :=  -- p+q is composite
by sorry

end NUMINAMATH_CALUDE_consecutive_primes_integral_roots_properties_l1269_126930


namespace NUMINAMATH_CALUDE_quadratic_polynomial_special_value_l1269_126983

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Property: [q(x)]^2 - x^2 is divisible by (x - 2)(x + 2)(x - 5) -/
def HasSpecialDivisibility (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ r : ℝ → ℝ, ∀ x : ℝ, (q x)^2 - x^2 = (x - 2) * (x + 2) * (x - 5) * (r x)

theorem quadratic_polynomial_special_value 
  (q : QuadraticPolynomial ℝ) 
  (h : HasSpecialDivisibility q) : 
  q 10 = 110 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_special_value_l1269_126983


namespace NUMINAMATH_CALUDE_pool_depth_l1269_126924

/-- Represents the dimensions and properties of a rectangular pool -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (chlorine_coverage : ℝ)
  (chlorine_cost : ℝ)
  (money_spent : ℝ)

/-- Theorem stating the depth of the pool given the conditions -/
theorem pool_depth (p : Pool) 
  (h1 : p.length = 10)
  (h2 : p.width = 8)
  (h3 : p.chlorine_coverage = 120)
  (h4 : p.chlorine_cost = 3)
  (h5 : p.money_spent = 12) :
  p.depth = 6 := by
  sorry

#check pool_depth

end NUMINAMATH_CALUDE_pool_depth_l1269_126924


namespace NUMINAMATH_CALUDE_math_only_students_l1269_126942

/-- Represents the number of students in each subject and the total --/
structure ClassSizes where
  science : ℕ
  math : ℕ
  total : ℕ

/-- The conditions of the problem --/
def problemConditions (c : ClassSizes) : Prop :=
  c.total = 30 ∧
  c.math = 3 * c.science ∧
  c.science + c.math + 2 = c.total

/-- The theorem to prove --/
theorem math_only_students (c : ClassSizes) :
  problemConditions c → c.math - 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_only_students_l1269_126942
