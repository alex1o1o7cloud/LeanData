import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l410_41078

/-- A positive arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), d > 0 ∧ ∀ n, a n = a₁ + (n - 1) * d

/-- A positive geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), r > 1 ∧ ∀ n, b n = b₁ * r^(n - 1)

theorem arithmetic_less_than_geometric
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_eq1 : a 1 = b 1)
  (h_eq2 : a 2 = b 2) :
  ∀ n ≥ 3, a n < b n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l410_41078


namespace NUMINAMATH_CALUDE_circle_equation_proof_l410_41025

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = (x - h)^2 + (y - k)^2 - r^2

/-- A point (x, y) is on a line ax + by + c = 0 if it satisfies the equation -/
def point_on_line (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- A point (x, y) is on a circle if it satisfies the circle's equation -/
def point_on_circle (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  f x y = 0

theorem circle_equation_proof (f : ℝ → ℝ → ℝ) :
  is_circle_equation 1 1 2 f →
  (∀ x y, point_on_line 1 1 (-2) x y → point_on_circle f x y) →
  point_on_circle f 1 (-1) →
  point_on_circle f (-1) 1 →
  ∀ x y, f x y = (x - 1)^2 + (y - 1)^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l410_41025


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l410_41080

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax² + bx + c = 0 are given by the quadratic formula:
    x = (-b ± √(b² - 4ac)) / (2a) -/
def isRootOfQuadratic (x k : ℝ) : Prop := x^2 - 72*x + k = 0

theorem unique_k_for_prime_roots : 
  ∃! k : ℝ, ∃ p q : ℕ, 
    isPrime p ∧ 
    isPrime q ∧ 
    isRootOfQuadratic p k ∧ 
    isRootOfQuadratic q k ∧
    k = 335 := by sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l410_41080


namespace NUMINAMATH_CALUDE_tangent_line_inverse_l410_41048

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
variable (h_inverse : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f)

-- Define a point x
variable (x : ℝ)

-- Define the tangent line to f at (x, f(x))
def tangent_line_f (t : ℝ) : ℝ := 2 * t - 3

-- State the theorem
theorem tangent_line_inverse (h_tangent : ∀ t, f x = tangent_line_f t - t) :
  ∀ t, x = t - 2 * (f_inv t) - 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_inverse_l410_41048


namespace NUMINAMATH_CALUDE_missing_angles_sum_l410_41072

-- Define the properties of our polygon
def ConvexPolygon (n : ℕ) (knownSum missingSum : ℝ) : Prop :=
  -- The polygon has n sides
  n > 2 ∧
  -- The sum of known angles is 1620°
  knownSum = 1620 ∧
  -- There are two missing angles
  -- The total sum (known + missing) is divisible by 180°
  ∃ (k : ℕ), (knownSum + missingSum) = 180 * k

-- State the theorem
theorem missing_angles_sum (n : ℕ) (knownSum missingSum : ℝ) 
  (h : ConvexPolygon n knownSum missingSum) : missingSum = 180 := by
  sorry

end NUMINAMATH_CALUDE_missing_angles_sum_l410_41072


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l410_41065

theorem complex_sum_of_powers (i : ℂ) : i^2 = -1 → i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l410_41065


namespace NUMINAMATH_CALUDE_candy_spending_l410_41059

/-- The fraction of a dollar spent on candy given initial quarters and remaining cents -/
def fraction_spent (initial_quarters : ℕ) (remaining_cents : ℕ) : ℚ :=
  (initial_quarters * 25 - remaining_cents) / 100

/-- Theorem stating that given 14 quarters initially and 300 cents remaining,
    the fraction of a dollar spent on candy is 1/2 -/
theorem candy_spending :
  fraction_spent 14 300 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_candy_spending_l410_41059


namespace NUMINAMATH_CALUDE_cricketer_wickets_before_match_l410_41041

/-- Represents a cricketer's bowling statistics -/
structure CricketerStats where
  wickets : ℕ
  runs : ℕ
  avg : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : CricketerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

theorem cricketer_wickets_before_match 
  (stats : CricketerStats)
  (h1 : stats.avg = 12.4)
  (h2 : newAverage stats 5 26 = 12) :
  stats.wickets = 85 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_wickets_before_match_l410_41041


namespace NUMINAMATH_CALUDE_carbon_copies_invariant_l410_41055

/-- Represents a stack of sheets with carbon paper -/
structure CarbonPaperStack :=
  (num_sheets : ℕ)
  (carbons_between : ℕ)

/-- Calculates the number of carbon copies produced by a stack -/
def carbon_copies (stack : CarbonPaperStack) : ℕ :=
  max 0 (stack.num_sheets - 1)

/-- Represents a folding operation on the stack -/
inductive FoldOperation
  | UpperLower
  | LeftRight
  | BackFront

/-- Applies a sequence of folding operations to a stack -/
def apply_folds (stack : CarbonPaperStack) (folds : List FoldOperation) : CarbonPaperStack :=
  stack

theorem carbon_copies_invariant (initial_stack : CarbonPaperStack) (folds : List FoldOperation) :
  initial_stack.num_sheets = 6 ∧ initial_stack.carbons_between = 2 →
  carbon_copies initial_stack = carbon_copies (apply_folds initial_stack folds) ∧
  carbon_copies initial_stack = 5 :=
sorry

end NUMINAMATH_CALUDE_carbon_copies_invariant_l410_41055


namespace NUMINAMATH_CALUDE_runners_capture_probability_l410_41096

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool -- true for counterclockwise, false for clockwise
  lap_time : ℕ -- time to complete one lap in seconds

/-- Represents the photographer's capture area -/
structure CaptureArea where
  fraction : ℚ -- fraction of the track captured
  center : ℚ -- position of the center of the capture area (0 ≤ center < 1)

/-- Calculates the probability of both runners being in the capture area -/
def probability_both_in_picture (runner1 runner2 : Runner) (capture : CaptureArea) 
  (start_time end_time : ℕ) : ℚ :=
sorry

theorem runners_capture_probability :
  let jenna : Runner := { direction := true, lap_time := 75 }
  let jonathan : Runner := { direction := false, lap_time := 60 }
  let capture : CaptureArea := { fraction := 1/3, center := 0 }
  probability_both_in_picture jenna jonathan capture (15 * 60) (16 * 60) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_runners_capture_probability_l410_41096


namespace NUMINAMATH_CALUDE_minimum_blocks_for_wall_l410_41067

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℝ
  length1 : ℝ
  length2 : ℝ

/-- Calculates the minimum number of blocks needed for a wall -/
def minimumBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specific wall -/
theorem minimum_blocks_for_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 2 1.5
  minimumBlocksNeeded wall block = 604 :=
by sorry

end NUMINAMATH_CALUDE_minimum_blocks_for_wall_l410_41067


namespace NUMINAMATH_CALUDE_count_rectangles_3x3_grid_l410_41051

/-- The number of rectangles that can be formed on a 3x3 grid -/
def rectangles_on_3x3_grid : ℕ := 9

/-- Theorem stating that the number of rectangles on a 3x3 grid is 9 -/
theorem count_rectangles_3x3_grid : 
  rectangles_on_3x3_grid = 9 := by sorry

end NUMINAMATH_CALUDE_count_rectangles_3x3_grid_l410_41051


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l410_41068

/-- Vovochka's addition method for three-digit numbers -/
def vovochkaAdd (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Regular addition for three-digit numbers -/
def regularAdd (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's addition and regular addition -/
def addDifference (a b c d e f : ℕ) : ℤ :=
  (vovochkaAdd a b c d e f : ℤ) - (regularAdd a b c d e f : ℤ)

/-- Theorem: The smallest positive difference between Vovochka's addition and regular addition is 1800 -/
theorem smallest_positive_difference :
  ∃ (a b c d e f : ℕ),
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10) ∧
    (a + d > 0) ∧
    (addDifference a b c d e f > 0) ∧
    (∀ (x y z u v w : ℕ),
      (x < 10 ∧ y < 10 ∧ z < 10 ∧ u < 10 ∧ v < 10 ∧ w < 10) →
      (x + u > 0) →
      (addDifference x y z u v w > 0) →
      (addDifference a b c d e f ≤ addDifference x y z u v w)) ∧
    (addDifference a b c d e f = 1800) :=
  sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l410_41068


namespace NUMINAMATH_CALUDE_rectangle_ratio_l410_41039

/-- A rectangle with a circle passing through two vertices and touching one side. -/
structure RectangleWithCircle where
  /-- Length of the longer side of the rectangle -/
  x : ℝ
  /-- Length of the shorter side of the rectangle -/
  y : ℝ
  /-- Radius of the circle -/
  R : ℝ
  /-- The perimeter of the rectangle is 4 times the radius of the circle -/
  h_perimeter : x + y = 2 * R
  /-- The circle passes through two vertices and touches one side -/
  h_circle_touch : y = R + Real.sqrt (R^2 - (x/2)^2)
  /-- The sides are positive -/
  h_positive : x > 0 ∧ y > 0 ∧ R > 0

/-- The ratio of the sides of the rectangle is 4:1 -/
theorem rectangle_ratio (rect : RectangleWithCircle) : rect.x / rect.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l410_41039


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l410_41056

theorem fixed_point_on_line (a : ℝ) : 
  let line := fun (x y : ℝ) => a * x + y + a + 1 = 0
  line (-1) (-1) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l410_41056


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l410_41033

theorem sum_a_b_equals_negative_one (a b : ℝ) : 
  (|a + 3| + (b - 2)^2 = 0) → (a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l410_41033


namespace NUMINAMATH_CALUDE_mars_other_elements_weight_l410_41086

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The ratio of iron in the moon's composition -/
def iron_ratio : ℝ := 0.5

/-- The ratio of carbon in the moon's composition -/
def carbon_ratio : ℝ := 0.2

/-- The ratio of Mars' weight to the moon's weight -/
def mars_moon_weight_ratio : ℝ := 2

theorem mars_other_elements_weight :
  let other_ratio : ℝ := 1 - iron_ratio - carbon_ratio
  let moon_other_weight : ℝ := other_ratio * moon_weight
  let mars_other_weight : ℝ := mars_moon_weight_ratio * moon_other_weight
  mars_other_weight = 150 := by sorry

end NUMINAMATH_CALUDE_mars_other_elements_weight_l410_41086


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l410_41092

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l410_41092


namespace NUMINAMATH_CALUDE_line_curve_intersection_implies_m_geq_3_l410_41090

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- Theorem statement
theorem line_curve_intersection_implies_m_geq_3 (k m : ℝ) :
  (∃ x y : ℝ, line k x = y ∧ curve x y m) → m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_line_curve_intersection_implies_m_geq_3_l410_41090


namespace NUMINAMATH_CALUDE_f_max_values_l410_41001

noncomputable def f (x θ : Real) : Real :=
  Real.sin x ^ 2 + Real.sqrt 3 * Real.tan θ * Real.cos x + (Real.sqrt 3 / 8) * Real.tan θ - 3/2

theorem f_max_values (θ : Real) (h : θ ∈ Set.Icc 0 (Real.pi / 3)) :
  (∃ (x : Real), f x (Real.pi / 3) ≤ f x (Real.pi / 3) ∧ f x (Real.pi / 3) = 15/8) ∧
  (∃ (θ' : Real) (h' : θ' ∈ Set.Icc 0 (Real.pi / 3)), 
    (∃ (x : Real), ∀ (y : Real), f y θ' ≤ f x θ' ∧ f x θ' = -1/8) ∧ 
    θ' = Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_f_max_values_l410_41001


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l410_41061

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel_line_line (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect_planes (p₁ p₂ : Plane) (l : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem parallel_lines_theorem 
  (a b c : Line) 
  (α β : Plane) 
  (h_non_overlapping_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_non_overlapping_planes : α ≠ β)
  (h1 : parallel_line_plane a α)
  (h2 : intersect_planes α β b)
  (h3 : line_in_plane a β) :
  parallel_line_line a b :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l410_41061


namespace NUMINAMATH_CALUDE_gross_monthly_salary_l410_41020

theorem gross_monthly_salary (rent food_expenses mortgage savings taxes gross_salary : ℚ) : 
  rent = 600 →
  food_expenses = (3/5) * rent →
  mortgage = 3 * food_expenses →
  savings = 2000 →
  taxes = (2/5) * savings →
  gross_salary = rent + food_expenses + mortgage + taxes + savings →
  gross_salary = 4840 := by
sorry

end NUMINAMATH_CALUDE_gross_monthly_salary_l410_41020


namespace NUMINAMATH_CALUDE_batsman_average_runs_l410_41098

/-- Calculates the average runs for a batsman given their performance in two sets of matches -/
def average_runs (matches1 : ℕ) (avg1 : ℚ) (matches2 : ℕ) (avg2 : ℚ) : ℚ :=
  ((matches1 : ℚ) * avg1 + (matches2 : ℚ) * avg2) / ((matches1 + matches2) : ℚ)

/-- Proves that the average runs for 30 matches is approximately 33.33 -/
theorem batsman_average_runs : 
  let matches1 := 20
  let avg1 := 40
  let matches2 := 10
  let avg2 := 20
  let total_matches := matches1 + matches2
  let result := average_runs matches1 avg1 matches2 avg2
  ∃ ε > 0, |result - 100/3| < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_runs_l410_41098


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l410_41081

theorem movie_ticket_cost (ticket_count : ℕ) (borrowed_movie_cost change paid : ℚ) : 
  ticket_count = 2 → 
  borrowed_movie_cost = 679/100 → 
  change = 137/100 → 
  paid = 20 → 
  ∃ (ticket_cost : ℚ), 
    ticket_cost * ticket_count + borrowed_movie_cost = paid - change ∧ 
    ticket_cost = 592/100 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l410_41081


namespace NUMINAMATH_CALUDE_probNotAllSame_l410_41046

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability that all dice show the same number -/
def probAllSame : ℚ := 1 / numSides^(numDice - 1)

/-- The probability that not all dice show the same number -/
theorem probNotAllSame : (1 : ℚ) - probAllSame = 215 / 216 := by sorry

end NUMINAMATH_CALUDE_probNotAllSame_l410_41046


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l410_41000

/-- The number of routes on a grid from (0, m) to (n, 0) moving only right or down -/
def num_routes (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The dimensions of the grid -/
def grid_width : ℕ := 3
def grid_height : ℕ := 2

theorem routes_on_3x2_grid :
  num_routes grid_height grid_width = Nat.choose (grid_height + grid_width) grid_height :=
by sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l410_41000


namespace NUMINAMATH_CALUDE_expression_simplification_l410_41057

theorem expression_simplification : 
  ((3 + 4 + 5 + 6) / 2) + ((3 * 6 + 9) / 3 + 1) = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l410_41057


namespace NUMINAMATH_CALUDE_robot_max_score_l410_41005

def initial_iq : ℕ := 25

def point_range : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 10}

def solve_problem (iq : ℕ) (points : ℕ) : Option ℕ :=
  if iq ≥ points then some (iq - points + 1) else none

def max_score (problems : List ℕ) : ℕ :=
  problems.foldl (λ acc p => acc + p) 0

theorem robot_max_score :
  ∃ (problems : List ℕ),
    (∀ p ∈ problems, p ∈ point_range) ∧
    (problems.foldl (λ iq p => (solve_problem iq p).getD 0) initial_iq ≠ 0) ∧
    (max_score problems = 31) ∧
    (∀ (other_problems : List ℕ),
      (∀ p ∈ other_problems, p ∈ point_range) →
      (other_problems.foldl (λ iq p => (solve_problem iq p).getD 0) initial_iq ≠ 0) →
      max_score other_problems ≤ 31) :=
by sorry

end NUMINAMATH_CALUDE_robot_max_score_l410_41005


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l410_41038

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (4 * a^2 - 8 * a - 21 = 0) →
  (4 * b^2 - 8 * b - 21 = 0) →
  (a - b)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l410_41038


namespace NUMINAMATH_CALUDE_fayes_age_l410_41013

/-- Given the ages of four people and their relationships, prove Faye's age -/
theorem fayes_age 
  (C D E F : ℕ) -- Chad's, Diana's, Eduardo's, and Faye's ages
  (h1 : D = E - 2) -- Diana is two years younger than Eduardo
  (h2 : E = C + 5) -- Eduardo is five years older than Chad
  (h3 : F = C + 4) -- Faye is four years older than Chad
  (h4 : D = 15) -- Diana is 15 years old
  : F = 16 := by
  sorry

end NUMINAMATH_CALUDE_fayes_age_l410_41013


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_345_l410_41003

/-- Returns the binary representation of a natural number as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 2) :: go (m / 2)
  go n

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_binary_digits_345 : sumList (toBinary 345) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_345_l410_41003


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l410_41045

theorem triangle_angle_bounds (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π) (h5 : A ≤ B) (h6 : B ≤ C) :
  (0 < A ∧ A ≤ π/3) ∧
  (0 < B ∧ B < π/2) ∧
  (π/3 ≤ C ∧ C < π) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l410_41045


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l410_41016

theorem units_digit_of_seven_to_sixth (n : ℕ) : n = 7^6 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l410_41016


namespace NUMINAMATH_CALUDE_box_width_l410_41002

/-- The width of a rectangular box given its filling rate, dimensions, and filling time. -/
theorem box_width
  (fill_rate : ℝ)  -- Filling rate in cubic feet per hour
  (length : ℝ)     -- Length of the box in feet
  (depth : ℝ)      -- Depth of the box in feet
  (fill_time : ℝ)  -- Time to fill the box in hours
  (h1 : fill_rate = 3)
  (h2 : length = 5)
  (h3 : depth = 3)
  (h4 : fill_time = 20) :
  ∃ (width : ℝ), width = 4 ∧ fill_rate * fill_time = length * width * depth :=
by
  sorry

end NUMINAMATH_CALUDE_box_width_l410_41002


namespace NUMINAMATH_CALUDE_angle_sixty_degrees_l410_41030

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem angle_sixty_degrees (t : Triangle) 
  (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) : 
  t.A = 60 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_angle_sixty_degrees_l410_41030


namespace NUMINAMATH_CALUDE_find_a_value_l410_41042

/-- The problem statement translated to Lean 4 --/
theorem find_a_value (a : ℝ) :
  (∀ x y : ℝ, 2*x - y + a ≥ 0 ∧ 3*x + y - 3 ≤ 0 →
    4*x + 3*y ≤ 8) ∧
  (∃ x y : ℝ, 2*x - y + a ≥ 0 ∧ 3*x + y - 3 ≤ 0 ∧
    4*x + 3*y = 8) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l410_41042


namespace NUMINAMATH_CALUDE_division_problem_l410_41095

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 171 →
  quotient = 8 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  divisor = 21 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l410_41095


namespace NUMINAMATH_CALUDE_cultivation_equation_correct_l410_41032

/-- Represents the cultivation problem of a farmer --/
structure CultivationProblem where
  paddy_area : ℝ
  dry_area : ℝ
  dry_rate_difference : ℝ
  time_ratio : ℝ

/-- The equation representing the cultivation problem --/
def cultivation_equation (p : CultivationProblem) (x : ℝ) : Prop :=
  p.paddy_area / x = 2 * (p.dry_area / (x + p.dry_rate_difference))

/-- Theorem stating that the given equation correctly represents the cultivation problem --/
theorem cultivation_equation_correct (p : CultivationProblem) :
  p.paddy_area = 36 ∧ 
  p.dry_area = 30 ∧ 
  p.dry_rate_difference = 4 ∧ 
  p.time_ratio = 2 →
  ∃ x : ℝ, cultivation_equation p x :=
by sorry

end NUMINAMATH_CALUDE_cultivation_equation_correct_l410_41032


namespace NUMINAMATH_CALUDE_max_distance_braking_car_l410_41085

/-- The distance function for a braking car -/
def s (b : ℝ) (t : ℝ) : ℝ := -6 * t^2 + b * t

/-- The theorem stating the maximum distance traveled by the braking car -/
theorem max_distance_braking_car :
  ∃ b : ℝ, (s b (1/2) = 6) ∧ (∀ t : ℝ, s b t ≤ 75/8) ∧ (∃ t : ℝ, s b t = 75/8) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_braking_car_l410_41085


namespace NUMINAMATH_CALUDE_fraction_addition_equivalence_l410_41047

theorem fraction_addition_equivalence (a b : ℤ) (h : b > 0) :
  ∀ x y : ℤ, y > 0 →
    (a / b + x / y = (a + x) / (b + y)) ↔
    ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by sorry

end NUMINAMATH_CALUDE_fraction_addition_equivalence_l410_41047


namespace NUMINAMATH_CALUDE_males_in_band_not_in_orchestra_l410_41026

/-- Given the information about band and orchestra membership, prove that the number of males in the band who are not in the orchestra is 10. -/
theorem males_in_band_not_in_orchestra : 
  ∀ (female_band male_band female_orch male_orch female_both total_students : ℕ),
    female_band = 100 →
    male_band = 80 →
    female_orch = 80 →
    male_orch = 100 →
    female_both = 60 →
    total_students = 230 →
    ∃ (male_both : ℕ),
      female_band + female_orch - female_both + male_band + male_orch - male_both = total_students ∧
      male_band - male_both = 10 :=
by sorry

end NUMINAMATH_CALUDE_males_in_band_not_in_orchestra_l410_41026


namespace NUMINAMATH_CALUDE_multiplication_division_equivalence_l410_41062

theorem multiplication_division_equivalence (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (14/5) := by
sorry

end NUMINAMATH_CALUDE_multiplication_division_equivalence_l410_41062


namespace NUMINAMATH_CALUDE_weeks_to_save_for_games_l410_41034

/-- Calculates the minimum number of weeks required to save for a games console and a video game -/
theorem weeks_to_save_for_games (console_cost video_game_cost initial_savings weekly_allowance : ℚ)
  (tax_rate : ℚ) (h_console : console_cost = 282)
  (h_video_game : video_game_cost = 75) (h_tax : tax_rate = 0.1)
  (h_initial : initial_savings = 42) (h_allowance : weekly_allowance = 24) :
  ⌈(console_cost + video_game_cost * (1 + tax_rate) - initial_savings) / weekly_allowance⌉ = 14 := by
sorry

end NUMINAMATH_CALUDE_weeks_to_save_for_games_l410_41034


namespace NUMINAMATH_CALUDE_superman_game_cost_l410_41044

/-- The cost of Tom's video game purchases -/
def total_spent : ℝ := 18.66

/-- The cost of the Batman game -/
def batman_cost : ℝ := 13.6

/-- The number of games Tom already owns -/
def existing_games : ℕ := 2

/-- The cost of the Superman game -/
def superman_cost : ℝ := total_spent - batman_cost

theorem superman_game_cost : superman_cost = 5.06 := by
  sorry

end NUMINAMATH_CALUDE_superman_game_cost_l410_41044


namespace NUMINAMATH_CALUDE_complex_calculation_l410_41024

theorem complex_calculation : 
  (2/3 * Real.sqrt 180) * (0.4 * 300)^3 - (0.4 * 180 - 1/3 * (0.4 * 180)) = 15454377.6 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l410_41024


namespace NUMINAMATH_CALUDE_fourth_region_area_l410_41027

/-- A regular hexagon divided into four regions by three line segments -/
structure DividedHexagon where
  /-- The total area of the hexagon -/
  total_area : ℝ
  /-- The areas of the four regions -/
  region_areas : Fin 4 → ℝ
  /-- The hexagon is regular and divided into four regions -/
  is_regular_divided : total_area = region_areas 0 + region_areas 1 + region_areas 2 + region_areas 3

/-- The theorem stating the area of the fourth region -/
theorem fourth_region_area (h : DividedHexagon) 
  (h1 : h.region_areas 0 = 2)
  (h2 : h.region_areas 1 = 3)
  (h3 : h.region_areas 2 = 4) :
  h.region_areas 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_region_area_l410_41027


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l410_41074

/-- Given an arithmetic sequence of 6 terms where the first term is 11 and the last term is 51,
    prove that the third term is 27. -/
theorem arithmetic_sequence_third_term :
  ∀ (seq : Fin 6 → ℝ),
    (∀ i j : Fin 6, seq (i + 1) - seq i = seq (j + 1) - seq j) →  -- arithmetic sequence
    seq 0 = 11 →  -- first term is 11
    seq 5 = 51 →  -- last term is 51
    seq 2 = 27 :=  -- third term is 27
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l410_41074


namespace NUMINAMATH_CALUDE_charity_share_is_75_l410_41058

-- Define the quantities of each baked good (in dozens)
def cookie_dozens : ℕ := 6
def brownie_dozens : ℕ := 4
def muffin_dozens : ℕ := 3

-- Define the selling prices (in dollars)
def cookie_price : ℚ := 3/2
def brownie_price : ℚ := 2
def muffin_price : ℚ := 5/2

-- Define the costs to make each item (in dollars)
def cookie_cost : ℚ := 1/4
def brownie_cost : ℚ := 1/2
def muffin_cost : ℚ := 3/4

-- Define the number of charities
def num_charities : ℕ := 3

-- Define a function to calculate the profit for each type of baked good
def profit_per_type (dozens : ℕ) (price : ℚ) (cost : ℚ) : ℚ :=
  (dozens * 12 : ℚ) * (price - cost)

-- Define the total profit
def total_profit : ℚ :=
  profit_per_type cookie_dozens cookie_price cookie_cost +
  profit_per_type brownie_dozens brownie_price brownie_cost +
  profit_per_type muffin_dozens muffin_price muffin_cost

-- Theorem to prove
theorem charity_share_is_75 :
  total_profit / num_charities = 75 := by sorry

end NUMINAMATH_CALUDE_charity_share_is_75_l410_41058


namespace NUMINAMATH_CALUDE_cos_A_value_cos_2A_plus_pi_over_4_l410_41053

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = C ∧ 2 * b = Real.sqrt 3 * a

-- Theorem 1: cos A = 1/3
theorem cos_A_value (A B C : ℝ) (a b c : ℝ) 
  (h : triangle A B C a b c) : Real.cos A = 1 / 3 := by
  sorry

-- Theorem 2: cos(2A + π/4) = -(8 + 7√2)/18
theorem cos_2A_plus_pi_over_4 (A B C : ℝ) (a b c : ℝ) 
  (h : triangle A B C a b c) : 
  Real.cos (2 * A + Real.pi / 4) = -(8 + 7 * Real.sqrt 2) / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_value_cos_2A_plus_pi_over_4_l410_41053


namespace NUMINAMATH_CALUDE_fuel_cost_theorem_l410_41035

theorem fuel_cost_theorem (x : ℝ) : 
  (x / 4 - x / 6 = 8) → x = 96 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_theorem_l410_41035


namespace NUMINAMATH_CALUDE_calligraphy_students_l410_41052

theorem calligraphy_students (x : ℕ) : 
  (50 : ℕ) = (2 * x - 1) + x + (51 - 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_calligraphy_students_l410_41052


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l410_41076

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 6

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

/-- Theorem stating the condition for f to have both maximum and minimum -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ x y : ℝ, ∀ z : ℝ, f a z ≤ f a x ∧ f a y ≤ f a z) ↔ a < 1/3 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l410_41076


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l410_41011

def U : Set ℕ := {x : ℕ | (x + 1 : ℚ) / (x - 5 : ℚ) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l410_41011


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l410_41050

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l410_41050


namespace NUMINAMATH_CALUDE_mary_payment_l410_41049

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_threshold : ℕ := 5
def discount_amount : ℕ := 1

def mary_apples : ℕ := 5
def mary_oranges : ℕ := 3
def mary_bananas : ℕ := 2

def total_fruits : ℕ := mary_apples + mary_oranges + mary_bananas

def fruit_cost : ℕ := mary_apples * apple_cost + mary_oranges * orange_cost + mary_bananas * banana_cost

def discount_sets : ℕ := total_fruits / discount_threshold

def total_discount : ℕ := discount_sets * discount_amount

def final_cost : ℕ := fruit_cost - total_discount

theorem mary_payment : final_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_payment_l410_41049


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l410_41015

def U : Set Int := Set.univ

def A : Set Int := {-1, 0, 1, 2}

def B : Set Int := {x | x^2 ≠ x}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l410_41015


namespace NUMINAMATH_CALUDE_divisors_121_divisors_1000_divisors_1000000000_l410_41031

-- Define a function to calculate the number of divisors given prime factorization
def num_divisors (factorization : List (Nat × Nat)) : Nat :=
  factorization.foldl (fun acc (_, exp) => acc * (exp + 1)) 1

-- Theorem for 121
theorem divisors_121 :
  num_divisors [(11, 2)] = 3 := by sorry

-- Theorem for 1000
theorem divisors_1000 :
  num_divisors [(2, 3), (5, 3)] = 16 := by sorry

-- Theorem for 1000000000
theorem divisors_1000000000 :
  num_divisors [(2, 9), (5, 9)] = 100 := by sorry

end NUMINAMATH_CALUDE_divisors_121_divisors_1000_divisors_1000000000_l410_41031


namespace NUMINAMATH_CALUDE_race_length_correct_l410_41021

/-- Represents the race scenario -/
structure Race where
  length : ℝ
  samTime : ℝ
  johnTime : ℝ
  headStart : ℝ

/-- The given race conditions -/
def givenRace : Race where
  length := 126
  samTime := 13
  johnTime := 18
  headStart := 35

/-- Theorem stating that the given race length satisfies all conditions -/
theorem race_length_correct (r : Race) : 
  r.samTime = 13 ∧ 
  r.johnTime = r.samTime + 5 ∧ 
  r.headStart = 35 ∧
  r.length / r.samTime * r.samTime = r.length / r.johnTime * r.samTime + r.headStart →
  r.length = 126 := by
  sorry

#check race_length_correct givenRace

end NUMINAMATH_CALUDE_race_length_correct_l410_41021


namespace NUMINAMATH_CALUDE_car_travel_distance_l410_41023

def initial_distance : ℝ := 192
def initial_gallons : ℝ := 6
def efficiency_increase : ℝ := 0.1
def new_gallons : ℝ := 8

theorem car_travel_distance : 
  let initial_mpg := initial_distance / initial_gallons
  let new_mpg := initial_mpg * (1 + efficiency_increase)
  new_mpg * new_gallons = 281.6 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l410_41023


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l410_41087

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (hr : abs r < 1) :
  (a / (1 - r)) = 8 * (a * r^2 / (1 - r)) →
  r = 1 / (2 * Real.sqrt 2) ∨ r = -1 / (2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l410_41087


namespace NUMINAMATH_CALUDE_weighted_average_price_approximation_l410_41008

def large_bottles : ℕ := 1365
def small_bottles : ℕ := 720
def medium_bottles : ℕ := 450
def extra_large_bottles : ℕ := 275

def large_price : ℚ := 189 / 100
def small_price : ℚ := 142 / 100
def medium_price : ℚ := 162 / 100
def extra_large_price : ℚ := 209 / 100

def total_bottles : ℕ := large_bottles + small_bottles + medium_bottles + extra_large_bottles

def total_cost : ℚ := 
  large_bottles * large_price + 
  small_bottles * small_price + 
  medium_bottles * medium_price + 
  extra_large_bottles * extra_large_price

def weighted_average_price : ℚ := total_cost / total_bottles

theorem weighted_average_price_approximation : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |weighted_average_price - 175/100| < ε :=
sorry

end NUMINAMATH_CALUDE_weighted_average_price_approximation_l410_41008


namespace NUMINAMATH_CALUDE_unique_integral_solution_l410_41014

/-- Given a system of equations, prove that there is only one integral solution -/
theorem unique_integral_solution :
  ∃! (x y z : ℤ),
    (z : ℝ) ^ (x : ℝ) = (y : ℝ) ^ (2 * x : ℝ) ∧
    (2 : ℝ) ^ (z : ℝ) = 2 * (8 : ℝ) ^ (x : ℝ) ∧
    x + y + z = 18 ∧
    x = 8 ∧ y = 5 ∧ z = 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l410_41014


namespace NUMINAMATH_CALUDE_mall_garage_third_level_spaces_l410_41028

/-- Represents the number of parking spaces on each level of a four-story parking garage -/
structure ParkingGarage :=
  (level1 : ℕ)
  (level2 : ℕ)
  (level3 : ℕ)
  (level4 : ℕ)

/-- Calculates the total number of parking spaces in the garage -/
def total_spaces (g : ParkingGarage) : ℕ := g.level1 + g.level2 + g.level3 + g.level4

/-- Represents the parking garage described in the problem -/
def mall_garage (x : ℕ) : ParkingGarage :=
  { level1 := 90
  , level2 := 90 + 8
  , level3 := 90 + 8 + x
  , level4 := 90 + 8 + x - 9 }

/-- The theorem to be proved -/
theorem mall_garage_third_level_spaces :
  ∃ x : ℕ, 
    total_spaces (mall_garage x) = 399 ∧ 
    x = 12 := by sorry

end NUMINAMATH_CALUDE_mall_garage_third_level_spaces_l410_41028


namespace NUMINAMATH_CALUDE_bc_plus_ce_is_one_third_of_ad_l410_41010

-- Define the points and lengths
variable (A B C D E : ℝ)
variable (AB AC AE BD CD ED BC CE AD : ℝ)

-- State the conditions
variable (h1 : B < C)
variable (h2 : C < E)
variable (h3 : E < D)
variable (h4 : AB = 3 * BD)
variable (h5 : AC = 7 * CD)
variable (h6 : AE = 5 * ED)
variable (h7 : AD = AB + BD + CD + ED)
variable (h8 : BC = AC - AB)
variable (h9 : CE = AE - AC)

-- State the theorem
theorem bc_plus_ce_is_one_third_of_ad :
  (BC + CE) / AD = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_bc_plus_ce_is_one_third_of_ad_l410_41010


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l410_41084

theorem rectangle_dimensions : ∃! x : ℝ, x > 3 ∧ (x - 3) * (3 * x + 4) = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l410_41084


namespace NUMINAMATH_CALUDE_cloth_selling_price_l410_41091

/-- Calculates the total selling price of cloth given the quantity sold, profit per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  quantity * (profit_per_meter + cost_price_per_meter)

/-- Proves that the total selling price of 66 meters of cloth with a profit of Rs. 5 per meter and a cost price of Rs. 5 per meter is Rs. 660. -/
theorem cloth_selling_price :
  total_selling_price 66 5 5 = 660 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l410_41091


namespace NUMINAMATH_CALUDE_hover_solution_l410_41036

def hover_problem (central_day1 : ℝ) : Prop :=
  let mountain_day1 : ℝ := 3
  let eastern_day1 : ℝ := 2
  let extra_day2 : ℝ := 2
  let total_time : ℝ := 24
  let mountain_day2 : ℝ := mountain_day1 + extra_day2
  let central_day2 : ℝ := central_day1 + extra_day2
  let eastern_day2 : ℝ := eastern_day1 + extra_day2
  mountain_day1 + central_day1 + eastern_day1 + mountain_day2 + central_day2 + eastern_day2 = total_time

theorem hover_solution : hover_problem 4 := by
  sorry

end NUMINAMATH_CALUDE_hover_solution_l410_41036


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l410_41094

theorem integer_solutions_quadratic_equation :
  ∀ a b : ℤ, 7 * a + 14 * b = 5 * a^2 + 5 * a * b + 5 * b^2 ↔
    (a = -1 ∧ b = 3) ∨ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l410_41094


namespace NUMINAMATH_CALUDE_prob_sum_leq_8_is_13_18_l410_41060

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum ≤ 8) when rolling two dice -/
def favorable_outcomes : ℕ := 26

/-- The probability of the sum being less than or equal to 8 when two dice are tossed -/
def prob_sum_leq_8 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_leq_8_is_13_18 : prob_sum_leq_8 = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_leq_8_is_13_18_l410_41060


namespace NUMINAMATH_CALUDE_range_of_g_l410_41064

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {π/12, -π/12} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l410_41064


namespace NUMINAMATH_CALUDE_balance_theorem_l410_41083

/-- Represents the weight of a ball in an arbitrary unit -/
@[ext] structure BallWeight where
  weight : ℚ

/-- Defines the weight relationships between different colored balls -/
structure BallWeights where
  red : BallWeight
  blue : BallWeight
  orange : BallWeight
  purple : BallWeight
  red_blue_balance : 4 * red.weight = 8 * blue.weight
  orange_blue_balance : 3 * orange.weight = 15/2 * blue.weight
  blue_purple_balance : 8 * blue.weight = 6 * purple.weight

/-- Theorem stating the balance of 68.5/3 blue balls with 5 red, 3 orange, and 4 purple balls -/
theorem balance_theorem (weights : BallWeights) :
  (68.5/3) * weights.blue.weight = 5 * weights.red.weight + 3 * weights.orange.weight + 4 * weights.purple.weight :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l410_41083


namespace NUMINAMATH_CALUDE_thirty_six_times_sum_of_digits_l410_41069

def sum_of_digits (x : ℕ) : ℕ := sorry

theorem thirty_six_times_sum_of_digits :
  ∀ x : ℕ, x = 36 * sum_of_digits x ↔ x = 324 ∨ x = 648 := by sorry

end NUMINAMATH_CALUDE_thirty_six_times_sum_of_digits_l410_41069


namespace NUMINAMATH_CALUDE_pants_price_l410_41019

theorem pants_price (total coat pants shoes : ℕ) 
  (h1 : total = 700)
  (h2 : total = coat + pants + shoes)
  (h3 : coat = pants + 340)
  (h4 : coat = shoes + pants + 180) :
  pants = 100 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_l410_41019


namespace NUMINAMATH_CALUDE_problem_stack_total_logs_l410_41004

/-- Represents a stack of logs -/
structure LogStack where
  bottomRowCount : ℕ
  topRowCount : ℕ
  rowDifference : ℕ

/-- Calculates the total number of logs in the stack -/
def totalLogs (stack : LogStack) : ℕ :=
  sorry

/-- The specific log stack described in the problem -/
def problemStack : LogStack :=
  { bottomRowCount := 20
  , topRowCount := 4
  , rowDifference := 2 }

theorem problem_stack_total_logs :
  totalLogs problemStack = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_stack_total_logs_l410_41004


namespace NUMINAMATH_CALUDE_conference_handshakes_l410_41088

def number_of_attendees : ℕ := 10

def handshake (a b : ℕ) : Prop := a ≠ b

def total_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

theorem conference_handshakes :
  total_handshakes number_of_attendees = 45 :=
sorry

end NUMINAMATH_CALUDE_conference_handshakes_l410_41088


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l410_41006

def A : Set ℚ := {x | ∃ k : ℕ, x = 3 * k + 1}
def B : Set ℚ := {x | x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 4, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l410_41006


namespace NUMINAMATH_CALUDE_min_perimeter_of_cross_sectional_triangle_l410_41082

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  baseEdgeLength : ℝ
  lateralEdgeLength : ℝ

/-- Cross-sectional triangle in the pyramid -/
structure CrossSectionalTriangle (p : RegularTriangularPyramid) where
  intersectsLateralEdges : Bool

/-- The minimum perimeter of the cross-sectional triangle -/
def minPerimeter (p : RegularTriangularPyramid) (t : CrossSectionalTriangle p) : ℝ :=
  sorry

/-- Theorem: Minimum perimeter of cross-sectional triangle in given pyramid -/
theorem min_perimeter_of_cross_sectional_triangle 
  (p : RegularTriangularPyramid) 
  (t : CrossSectionalTriangle p)
  (h1 : p.baseEdgeLength = 4)
  (h2 : p.lateralEdgeLength = 8)
  (h3 : t.intersectsLateralEdges = true) :
  minPerimeter p t = 11 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_of_cross_sectional_triangle_l410_41082


namespace NUMINAMATH_CALUDE_complex_equation_solution_l410_41099

theorem complex_equation_solution (a : ℂ) :
  a / (1 - Complex.I) = (1 + Complex.I) / Complex.I → a = -2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l410_41099


namespace NUMINAMATH_CALUDE_coffee_shop_weekly_total_l410_41043

/-- Represents a coffee shop with its brewing characteristics -/
structure CoffeeShop where
  weekday_rate : ℕ  -- Cups brewed per hour on weekdays
  weekend_total : ℕ  -- Total cups brewed over the weekend
  daily_hours : ℕ  -- Hours open per day

/-- Calculates the total number of coffee cups brewed in one week -/
def weekly_total (shop : CoffeeShop) : ℕ :=
  (shop.weekday_rate * shop.daily_hours * 5) + shop.weekend_total

/-- Theorem stating that a coffee shop with given characteristics brews 370 cups per week -/
theorem coffee_shop_weekly_total :
  ∀ (shop : CoffeeShop),
    shop.weekday_rate = 10 ∧
    shop.weekend_total = 120 ∧
    shop.daily_hours = 5 →
    weekly_total shop = 370 := by
  sorry


end NUMINAMATH_CALUDE_coffee_shop_weekly_total_l410_41043


namespace NUMINAMATH_CALUDE_train_speed_l410_41089

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l410_41089


namespace NUMINAMATH_CALUDE_max_third_side_length_l410_41097

theorem max_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 11) :
  ∀ c : ℝ, (c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) → c ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_max_third_side_length_l410_41097


namespace NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l410_41070

theorem smallest_factor_of_32_with_sum_3 :
  ∃ (a b c : ℤ),
    a * b * c = 32 ∧
    a + b + c = 3 ∧
    (∀ (x y z : ℤ), x * y * z = 32 → x + y + z = 3 → min a (min b c) ≤ min x (min y z)) ∧
    min a (min b c) = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l410_41070


namespace NUMINAMATH_CALUDE_smaller_number_is_ten_l410_41018

theorem smaller_number_is_ten (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) (h3 : x ≤ y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_is_ten_l410_41018


namespace NUMINAMATH_CALUDE_factorization_equality_l410_41077

theorem factorization_equality (x y : ℝ) : 2*x^2 - 8*x*y + 8*y^2 = 2*(x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l410_41077


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l410_41040

/-- Given a cubic equation and a polynomial P satisfying certain conditions, 
    prove that P has a specific form. -/
theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) : 
  (∀ x, x^3 - 4*x^2 + x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  P a = b + c →
  P b = a + c →
  P c = a + b →
  P (a + b + c) = -20 →
  ∀ x, P x = (-20*x^3 + 80*x^2 - 23*x + 32) / 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l410_41040


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ninth_l410_41007

theorem last_digit_of_one_over_three_to_ninth (n : ℕ) : n = 3^9 → (1000000000 / n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ninth_l410_41007


namespace NUMINAMATH_CALUDE_joans_remaining_books_l410_41009

/-- Calculates the number of remaining books after a sale. -/
def remaining_books (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem stating that Joan's remaining books is 7. -/
theorem joans_remaining_books :
  remaining_books 33 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joans_remaining_books_l410_41009


namespace NUMINAMATH_CALUDE_function_inequality_l410_41066

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f is differentiable
variable (hf : Differentiable ℝ f)

-- Define the condition f'(x) + f(x) < 0
variable (hf' : ∀ x, HasDerivAt f (f x) x → (deriv f x + f x < 0))

-- Define m as a real number
variable (m : ℝ)

-- State the theorem
theorem function_inequality :
  f (m - m^2) > Real.exp (m^2 - m + 1) * f 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l410_41066


namespace NUMINAMATH_CALUDE_minimize_expression_l410_41037

theorem minimize_expression (x : ℝ) (h : x > 1) :
  (2 + 3*x + 4/(x - 1)) ≥ 4*Real.sqrt 3 + 5 ∧
  (2 + 3*x + 4/(x - 1) = 4*Real.sqrt 3 + 5 ↔ x = 2/3*Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_minimize_expression_l410_41037


namespace NUMINAMATH_CALUDE_interest_rate_increase_specific_interest_rate_increase_l410_41029

-- Define the initial interest rate
def last_year_rate : ℝ := 9.90990990990991

-- Define the increase percentage
def increase_percent : ℝ := 10

-- Theorem to prove
theorem interest_rate_increase (last_year_rate : ℝ) (increase_percent : ℝ) :
  last_year_rate * (1 + increase_percent / 100) = 10.9009009009009 := by
  sorry

-- Apply the theorem to our specific values
theorem specific_interest_rate_increase :
  last_year_rate * (1 + increase_percent / 100) = 10.9009009009009 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_increase_specific_interest_rate_increase_l410_41029


namespace NUMINAMATH_CALUDE_factor_expression_l410_41017

theorem factor_expression (b c : ℝ) : 55 * b^2 + 165 * b * c = 55 * b * (b + 3 * c) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l410_41017


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l410_41093

theorem average_of_six_numbers (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1) / 2 = 1.1)
  (h2 : (numbers 2 + numbers 3) / 2 = 1.4)
  (h3 : (numbers 4 + numbers 5) / 2 = 5) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l410_41093


namespace NUMINAMATH_CALUDE_gcd_38_23_l410_41054

theorem gcd_38_23 : Nat.gcd 38 23 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_38_23_l410_41054


namespace NUMINAMATH_CALUDE_equation_equivalence_l410_41079

theorem equation_equivalence (x : ℝ) : x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l410_41079


namespace NUMINAMATH_CALUDE_total_video_game_cost_l410_41073

/-- The cost of the basketball game -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game -/
def racing_cost : ℚ := 4.23

/-- The total cost of the video games -/
def total_cost : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total cost of video games is $9.43 -/
theorem total_video_game_cost : total_cost = 9.43 := by sorry

end NUMINAMATH_CALUDE_total_video_game_cost_l410_41073


namespace NUMINAMATH_CALUDE_intersection_point_modulo_9_l410_41071

theorem intersection_point_modulo_9 :
  ∀ x : ℕ, (3 * x + 6) % 9 = (7 * x + 3) % 9 → x % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_modulo_9_l410_41071


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_theorem_l410_41022

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem about the perimeter of a non-shaded region in a composite figure -/
theorem non_shaded_perimeter_theorem 
  (large_rect : Rectangle)
  (small_rect : Rectangle)
  (shaded_area : ℝ)
  (h1 : large_rect.length = 12)
  (h2 : large_rect.width = 10)
  (h3 : small_rect.length = 4)
  (h4 : small_rect.width = 3)
  (h5 : shaded_area = 104) : 
  ∃ (non_shaded_rect : Rectangle), 
    abs (perimeter non_shaded_rect - 21.34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_theorem_l410_41022


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_l410_41063

/-- A function that computes the sum ABC + CA + B -/
def digit_sum (A B C : ℕ) : ℕ := 101 * A + 11 * B + 11 * C

/-- A predicate that checks if three natural numbers are different digits -/
def are_different_digits (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10

theorem largest_three_digit_sum :
  ∃ A B C : ℕ, are_different_digits A B C ∧ 
  digit_sum A B C = 986 ∧
  ∀ X Y Z : ℕ, are_different_digits X Y Z → 
  digit_sum X Y Z ≤ 986 ∧ digit_sum X Y Z < 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_l410_41063


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l410_41012

theorem product_of_sums_equals_difference_of_powers : 
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * 
  (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 3^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l410_41012


namespace NUMINAMATH_CALUDE_two_integer_b_values_l410_41075

theorem two_integer_b_values : 
  ∃! (S : Finset ℤ), 
    (Finset.card S = 2) ∧ 
    (∀ b ∈ S, ∃! (T : Finset ℤ), 
      (Finset.card T = 2) ∧ 
      (∀ x ∈ T, x^2 + b*x + 3 ≤ 0) ∧
      (∀ x : ℤ, x^2 + b*x + 3 ≤ 0 → x ∈ T)) := by
sorry

end NUMINAMATH_CALUDE_two_integer_b_values_l410_41075
