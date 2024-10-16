import Mathlib

namespace NUMINAMATH_CALUDE_part1_part2_l238_23855

-- Definition of arithmetic sequence sum
def S (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * d

-- Part 1
theorem part1 : ∃! k : ℕ+, S (3/2) 1 (k^2) = (S (3/2) 1 k)^2 := by sorry

-- Part 2
theorem part2 : ∀ a1 d : ℚ, 
  (∀ k : ℕ+, S a1 d (k^2) = (S a1 d k)^2) ↔ 
  ((a1 = 0 ∧ d = 0) ∨ (a1 = 1 ∧ d = 0) ∨ (a1 = 1 ∧ d = 2)) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l238_23855


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l238_23872

theorem consecutive_integers_product_sum (a b c d : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ a * b * c * d = 5040 → a + b + c + d = 34 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l238_23872


namespace NUMINAMATH_CALUDE_right_triangle_existence_l238_23879

theorem right_triangle_existence (α β : ℝ) :
  (∃ (x y z h : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧ h > 0 ∧
    x^2 + y^2 = z^2 ∧
    x * y = z * h ∧
    x - y = α ∧
    z - h = β) ↔
  β > α :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l238_23879


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l238_23820

theorem triangle_abc_properties (A B C : ℝ) (AB : ℝ) :
  2 * Real.sin (2 * C) * Real.cos C - Real.sin (3 * C) = Real.sqrt 3 * (1 - Real.cos C) →
  AB = 2 →
  Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A) →
  C = π / 3 ∧ (1 / 2) * AB * Real.sin C * Real.sqrt ((4 - AB^2) / (4 * Real.sin C^2)) = (2 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l238_23820


namespace NUMINAMATH_CALUDE_least_prime_factor_of_p6_minus_p5_l238_23891

theorem least_prime_factor_of_p6_minus_p5 (p : ℕ) (hp : Nat.Prime p) :
  Nat.minFac (p^6 - p^5) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_p6_minus_p5_l238_23891


namespace NUMINAMATH_CALUDE_quadratic_function_zeros_range_l238_23840

theorem quadratic_function_zeros_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (x₁ ∈ Set.Ioo (-2) 0 ∧ x₂ ∈ Set.Ioo 2 3) ∧
    (x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0)) →
  a ∈ Set.Ioo (-3) 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_zeros_range_l238_23840


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l238_23810

theorem diophantine_equation_solution : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l238_23810


namespace NUMINAMATH_CALUDE_magic_square_d_plus_e_l238_23817

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  sum_eq : sum = 30 + e + 15
         ∧ sum = 10 + c + d
         ∧ sum = a + 25 + b
         ∧ sum = 30 + 10 + a
         ∧ sum = e + c + 25
         ∧ sum = 15 + d + b
         ∧ sum = 30 + c + b
         ∧ sum = a + c + e
         ∧ sum = 15 + 25 + a

theorem magic_square_d_plus_e (sq : MagicSquare) : sq.d + sq.e = 25 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_d_plus_e_l238_23817


namespace NUMINAMATH_CALUDE_all_configurations_exist_l238_23894

-- Define the geometric shapes
structure Rectangle where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  all_right_angles : ∀ i, angles i = 90
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3

structure Rhombus where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j, sides i = sides j
  opposite_angles_equal : angles 0 = angles 2 ∧ angles 1 = angles 3

structure Parallelogram where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3
  adjacent_angles_supplementary : ∀ i, angles i + angles ((i + 1) % 4) = 180

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ

structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : angles 0 + angles 1 + angles 2 = 180

-- Theorem stating that all configurations can exist
theorem all_configurations_exist :
  (∃ r : Rectangle, r.sides 0 ≠ r.sides 1) ∧
  (∃ rh : Rhombus, ∀ i, rh.angles i = 90) ∧
  (∃ p : Parallelogram, True) ∧
  (∃ q : Quadrilateral, (∀ i, q.angles i = 90) ∧ q.sides 0 ≠ q.sides 1) ∧
  (∃ t : Triangle, t.angles 0 = 100 ∧ t.angles 1 = 40 ∧ t.angles 2 = 40) :=
by sorry

end NUMINAMATH_CALUDE_all_configurations_exist_l238_23894


namespace NUMINAMATH_CALUDE_selections_equal_sixteen_l238_23844

/-- The number of ways to select 3 people from 2 females and 4 males, with at least 1 female -/
def selectionsWithFemale (totalStudents femaleStudents maleStudents selectCount : ℕ) : ℕ :=
  Nat.choose totalStudents selectCount - Nat.choose maleStudents selectCount

/-- Proof that the number of selections is 16 -/
theorem selections_equal_sixteen :
  selectionsWithFemale 6 2 4 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_selections_equal_sixteen_l238_23844


namespace NUMINAMATH_CALUDE_grade_distribution_l238_23870

theorem grade_distribution (thompson_total : ℕ) (thompson_a : ℕ) (thompson_b : ℕ) (carter_total : ℕ)
  (h1 : thompson_total = 20)
  (h2 : thompson_a = 12)
  (h3 : thompson_b = 5)
  (h4 : carter_total = 30)
  (h5 : thompson_a + thompson_b ≤ thompson_total) :
  ∃ (carter_a carter_b : ℕ),
    carter_a + carter_b ≤ carter_total ∧
    carter_a * thompson_total = thompson_a * carter_total ∧
    carter_b * (thompson_total - thompson_a) = thompson_b * (carter_total - carter_a) ∧
    carter_a = 18 ∧
    carter_b = 8 :=
by sorry

end NUMINAMATH_CALUDE_grade_distribution_l238_23870


namespace NUMINAMATH_CALUDE_sum_p_q_equals_expected_p_condition_q_condition_l238_23867

/-- A linear function p(x) satisfying p(-1) = -2 -/
def p (x : ℝ) : ℝ := 4 * x - 2

/-- A quadratic function q(x) satisfying q(1) = 3 -/
def q (x : ℝ) : ℝ := 1.5 * x^2 - 1.5

/-- Theorem stating that p(x) + q(x) = 1.5x^2 + 4x - 3.5 -/
theorem sum_p_q_equals_expected : 
  ∀ x : ℝ, p x + q x = 1.5 * x^2 + 4 * x - 3.5 := by
  sorry

/-- Verification of the conditions -/
theorem p_condition : p (-1) = -2 := by
  sorry

theorem q_condition : q 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_p_q_equals_expected_p_condition_q_condition_l238_23867


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l238_23850

-- Define a function to check if a number is a four-digit number
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a function to check if the first two digits are equal
def firstTwoEqual (n : ℕ) : Prop :=
  (n / 1000) = ((n / 100) % 10)

-- Define a function to check if the last two digits are equal
def lastTwoEqual (n : ℕ) : Prop :=
  ((n / 10) % 10) = (n % 10)

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Theorem statement
theorem unique_four_digit_square :
  ∃! n : ℕ, isFourDigit n ∧ firstTwoEqual n ∧ lastTwoEqual n ∧ isPerfectSquare n ∧ n = 7744 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l238_23850


namespace NUMINAMATH_CALUDE_max_score_is_six_l238_23886

/-- Represents a 5x5 game board -/
def GameBoard : Type := Fin 5 → Fin 5 → Bool

/-- Calculates the sum of a 3x3 sub-square starting at (i, j) -/
def subSquareSum (board : GameBoard) (i j : Fin 3) : ℕ :=
  (Finset.sum (Finset.range 3) fun x =>
    Finset.sum (Finset.range 3) fun y =>
      if board (i + x) (j + y) then 1 else 0)

/-- Calculates the score of a given board (maximum sum of any 3x3 sub-square) -/
def boardScore (board : GameBoard) : ℕ :=
  Finset.sup (Finset.range 3) fun i =>
    Finset.sup (Finset.range 3) fun j =>
      subSquareSum board i j

/-- Represents a strategy for Player 2 -/
def Player2Strategy : Type := GameBoard → Fin 5 → Fin 5

/-- Represents the game play with both players' moves -/
def gamePlay (p2strat : Player2Strategy) : GameBoard :=
  sorry -- Implementation of game play

theorem max_score_is_six :
  ∀ (p2strat : Player2Strategy),
    boardScore (gamePlay p2strat) ≤ 6 ∧
    ∃ (optimal_p2strat : Player2Strategy),
      boardScore (gamePlay optimal_p2strat) = 6 :=
sorry

end NUMINAMATH_CALUDE_max_score_is_six_l238_23886


namespace NUMINAMATH_CALUDE_elena_allowance_spending_l238_23800

theorem elena_allowance_spending (A : ℝ) : ∃ (m s : ℝ),
  m = (1/4) * (A - s) ∧
  s = (1/10) * (A - m) ∧
  m + s = (4/13) * A :=
by sorry

end NUMINAMATH_CALUDE_elena_allowance_spending_l238_23800


namespace NUMINAMATH_CALUDE_geometric_propositions_l238_23895

/-- Two lines in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ
  nonzero : normal ≠ (0, 0, 0)

/-- Perpendicularity between lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Parallelism between lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallelism between planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem geometric_propositions :
  ∃ (l1 l2 l3 : Line3D) (p1 p2 p3 : Plane3D),
    (¬∀ l1 l2 l3, perpendicular_lines l1 l3 → perpendicular_lines l2 l3 → parallel_lines l1 l2) ∧
    (∀ l1 l2 p, perpendicular_line_plane l1 p → perpendicular_line_plane l2 p → parallel_lines l1 l2) ∧
    (∀ p1 p2 l, perpendicular_line_plane l p1 → perpendicular_line_plane l p2 → parallel_planes p1 p2) ∧
    (¬∀ p1 p2 p3, perpendicular_planes p1 p3 → perpendicular_planes p2 p3 → perpendicular_planes p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_propositions_l238_23895


namespace NUMINAMATH_CALUDE_largest_number_value_l238_23805

theorem largest_number_value (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 5) :
  c = 41.67 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_value_l238_23805


namespace NUMINAMATH_CALUDE_tina_total_pens_l238_23846

/-- The number of pens Tina has -/
structure PenCount where
  pink : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ

/-- Conditions on Tina's pen count -/
def tina_pen_conditions (p : PenCount) : Prop :=
  p.pink = 15 ∧
  p.green = p.pink - 9 ∧
  p.blue = p.green + 3 ∧
  p.yellow = p.pink + p.green - 5

/-- Theorem stating the total number of pens Tina has -/
theorem tina_total_pens (p : PenCount) (h : tina_pen_conditions p) :
  p.pink + p.green + p.blue + p.yellow = 46 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_pens_l238_23846


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l238_23832

theorem multiply_and_simplify (x : ℝ) (h : x ≠ 0) :
  (18 * x^3) * (4 * x^2) * (1 / (2*x)^3) = 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l238_23832


namespace NUMINAMATH_CALUDE_nuts_cost_correct_l238_23877

/-- The cost of nuts per kilogram -/
def cost_of_nuts : ℝ := 12

/-- The cost of dried fruits per kilogram -/
def cost_of_dried_fruits : ℝ := 8

/-- The amount of nuts bought in kilograms -/
def amount_of_nuts : ℝ := 3

/-- The amount of dried fruits bought in kilograms -/
def amount_of_dried_fruits : ℝ := 2.5

/-- The total cost of the purchase -/
def total_cost : ℝ := 56

theorem nuts_cost_correct : 
  cost_of_nuts * amount_of_nuts + cost_of_dried_fruits * amount_of_dried_fruits = total_cost :=
by sorry

end NUMINAMATH_CALUDE_nuts_cost_correct_l238_23877


namespace NUMINAMATH_CALUDE_smallest_positive_value_cubic_expression_l238_23884

theorem smallest_positive_value_cubic_expression (a b c : ℕ+) :
  a^3 + b^3 + c^3 - 3*a*b*c ≥ 4 ∧ ∃ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_value_cubic_expression_l238_23884


namespace NUMINAMATH_CALUDE_sandwich_cost_l238_23841

theorem sandwich_cost (total_cost soda_cost : ℝ) 
  (h1 : total_cost = 10.46)
  (h2 : soda_cost = 0.87) : 
  ∃ sandwich_cost : ℝ, 
    sandwich_cost = 3.49 ∧ 
    2 * sandwich_cost + 4 * soda_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_sandwich_cost_l238_23841


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l238_23892

theorem unique_function_satisfying_condition :
  ∃! f : ℝ → ℝ, (∀ x y z : ℝ, f (x * y) + f (x * z) + f x * f (y * z) ≥ 3) ∧
  (∀ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l238_23892


namespace NUMINAMATH_CALUDE_binomial_expansion_101_2_l238_23839

theorem binomial_expansion_101_2 : 
  101^3 + 3*(101^2)*2 + 3*101*(2^2) + 2^3 = 1092727 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_101_2_l238_23839


namespace NUMINAMATH_CALUDE_total_flowers_l238_23874

theorem total_flowers (roses tulips lilies : ℕ) : 
  roses = 58 ∧ 
  tulips = roses - 15 ∧ 
  lilies = roses + 25 → 
  roses + tulips + lilies = 184 := by
sorry

end NUMINAMATH_CALUDE_total_flowers_l238_23874


namespace NUMINAMATH_CALUDE_milburg_children_count_l238_23835

/-- The number of children in Milburg -/
def children_count (total_population : ℕ) (adult_count : ℕ) : ℕ :=
  total_population - adult_count

/-- Theorem: The number of children in Milburg is 2987 -/
theorem milburg_children_count :
  children_count 5256 2269 = 2987 := by
  sorry

end NUMINAMATH_CALUDE_milburg_children_count_l238_23835


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l238_23838

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b = 0}

-- State the theorem
theorem unique_quadratic_root (a b : ℝ) : A a b = {1} → a = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l238_23838


namespace NUMINAMATH_CALUDE_digit_sum_of_squared_palindrome_l238_23825

theorem digit_sum_of_squared_palindrome (r : ℕ) (x : ℕ) (p q : ℕ) :
  r ≤ 400 →
  x = p * r^3 + p * r^2 + q * r + q →
  7 * q = 17 * p →
  ∃ (a b c : ℕ),
    x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a →
  2 * (a + b + c) = 400 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_of_squared_palindrome_l238_23825


namespace NUMINAMATH_CALUDE_p_on_x_axis_equal_distance_to_axes_l238_23873

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (8 - 2*m, m - 1)

-- Part 1: P lies on the x-axis implies m = 1
theorem p_on_x_axis (m : ℝ) : (P m).2 = 0 → m = 1 := by sorry

-- Part 2: Equal distance to both axes implies P(2,2) or P(-6,6)
theorem equal_distance_to_axes (m : ℝ) : 
  |8 - 2*m| = |m - 1| → (P m = (2, 2) ∨ P m = (-6, 6)) := by sorry

end NUMINAMATH_CALUDE_p_on_x_axis_equal_distance_to_axes_l238_23873


namespace NUMINAMATH_CALUDE_extreme_value_condition_monotonicity_intervals_min_value_on_interval_l238_23834

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

-- Theorem 1: f(x) has an extreme value at x = 1 if and only if a = -1
theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) ↔ a = -1 :=
sorry

-- Theorem 2: Monotonicity intervals depend on the value of a
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ (x y : ℝ), x < y → f a x < f a y) ∧
  (a > 0 → ∀ (x y : ℝ), (x < y ∧ y < -a) ∨ (x > 0 ∧ y > x) → f a x < f a y) ∧
  (a > 0 → ∀ (x y : ℝ), -a < x ∧ x < y ∧ y < 0 → f a x > f a y) ∧
  (a < 0 → ∀ (x y : ℝ), (x < y ∧ y < 0) ∨ (x > -a ∧ y > x) → f a x < f a y) ∧
  (a < 0 → ∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < -a → f a x > f a y) :=
sorry

-- Theorem 3: Minimum value on [0, 2] depends on the value of a
theorem min_value_on_interval (a : ℝ) :
  (a ≥ 0 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a 0) ∧
  (-2 < a ∧ a < 0 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a (-a)) ∧
  (a ≤ -2 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a 2) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_monotonicity_intervals_min_value_on_interval_l238_23834


namespace NUMINAMATH_CALUDE_garland_theorem_l238_23875

/-- The number of ways to arrange light bulbs in a garland -/
def garland_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red + 1) white * Nat.choose (blue + red) blue

/-- Theorem: The number of ways to arrange 9 blue, 7 red, and 14 white light bulbs
    in a garland, such that no two white light bulbs are adjacent, is 7,779,200 -/
theorem garland_theorem :
  garland_arrangements 9 7 14 = 7779200 := by
  sorry

#eval garland_arrangements 9 7 14

end NUMINAMATH_CALUDE_garland_theorem_l238_23875


namespace NUMINAMATH_CALUDE_drone_production_equations_correct_l238_23854

/-- Represents the number of drones of type A and B produced by a company -/
structure DroneProduction where
  x : ℝ  -- number of type A drones
  y : ℝ  -- number of type B drones

/-- The system of equations representing the drone production conditions -/
def satisfiesConditions (p : DroneProduction) : Prop :=
  p.x = (1/2) * (p.x + p.y) + 11 ∧ p.y = (1/3) * (p.x + p.y) - 2

/-- Theorem stating that the system of equations correctly represents the given conditions -/
theorem drone_production_equations_correct (p : DroneProduction) :
  satisfiesConditions p ↔
    (p.x = (1/2) * (p.x + p.y) + 11 ∧   -- Type A drones condition
     p.y = (1/3) * (p.x + p.y) - 2) :=  -- Type B drones condition
by sorry

end NUMINAMATH_CALUDE_drone_production_equations_correct_l238_23854


namespace NUMINAMATH_CALUDE_boxes_to_brother_l238_23883

def total_boxes : ℕ := 45
def boxes_to_sister : ℕ := 9
def boxes_to_cousin : ℕ := 7
def boxes_left : ℕ := 17

theorem boxes_to_brother :
  total_boxes - boxes_to_sister - boxes_to_cousin - boxes_left = 12 := by
  sorry

end NUMINAMATH_CALUDE_boxes_to_brother_l238_23883


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l238_23814

/-- The number of games required in a single-elimination tournament -/
def games_required (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 21 teams, 20 games are required to declare a winner -/
theorem single_elimination_tournament_games :
  games_required 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l238_23814


namespace NUMINAMATH_CALUDE_black_faces_alignment_l238_23871

/-- Represents a cube with one black face and five white faces -/
structure Cube where
  blackFace : Fin 6

/-- Represents the 8x8 grid of cubes -/
def Grid := Fin 8 → Fin 8 → Cube

/-- Rotates all cubes in a given row -/
def rotateRow (g : Grid) (row : Fin 8) : Grid :=
  sorry

/-- Rotates all cubes in a given column -/
def rotateColumn (g : Grid) (col : Fin 8) : Grid :=
  sorry

/-- Checks if all cubes have their black faces pointing in the same direction -/
def allFacingSameDirection (g : Grid) : Prop :=
  sorry

/-- The main theorem stating that it's always possible to make all black faces point in the same direction -/
theorem black_faces_alignment (g : Grid) :
  ∃ (ops : List (Sum (Fin 8) (Fin 8))), 
    let finalGrid := ops.foldl (λ acc op => match op with
      | Sum.inl row => rotateRow acc row
      | Sum.inr col => rotateColumn acc col) g
    allFacingSameDirection finalGrid :=
  sorry

end NUMINAMATH_CALUDE_black_faces_alignment_l238_23871


namespace NUMINAMATH_CALUDE_abs_c_value_l238_23856

def polynomial (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_value (a b c : ℤ) : 
  polynomial a b c (3 - Complex.I) = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 129 :=
sorry

end NUMINAMATH_CALUDE_abs_c_value_l238_23856


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_seven_l238_23845

theorem product_of_repeating_decimal_and_seven :
  ∃ (x : ℚ), (∀ n : ℕ, (x * 10^(3*n+3) - x * 10^(3*n)).num = 456 ∧ 
              (x * 10^(3*n+3) - x * 10^(3*n)).den = 10^3 - 1) →
  x * 7 = 1064 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_seven_l238_23845


namespace NUMINAMATH_CALUDE_area_gray_quadrilateral_l238_23878

/-- The Stomachion puzzle square --/
def stomachion_square : ℝ := 12

/-- The length of side AB in the gray quadrilateral --/
def side_AB : ℝ := 6

/-- The height of triangle ABD --/
def height_ABD : ℝ := 3

/-- The length of side BC in the gray quadrilateral --/
def side_BC : ℝ := 3

/-- The height of triangle BCD --/
def height_BCD : ℝ := 2

/-- The area of the gray quadrilateral ABCD in the Stomachion puzzle --/
theorem area_gray_quadrilateral : 
  (1/2 * side_AB * height_ABD) + (1/2 * side_BC * height_BCD) = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_gray_quadrilateral_l238_23878


namespace NUMINAMATH_CALUDE_complex_modulus_product_l238_23828

theorem complex_modulus_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l238_23828


namespace NUMINAMATH_CALUDE_profit_calculation_l238_23863

/-- Profit calculation for a product with variable price reduction --/
theorem profit_calculation 
  (price_tag : ℕ) 
  (discount : ℚ) 
  (initial_profit : ℕ) 
  (initial_sales : ℕ) 
  (sales_increase : ℕ) 
  (x : ℕ) 
  (h1 : price_tag = 80)
  (h2 : discount = 1/5)
  (h3 : initial_profit = 24)
  (h4 : initial_sales = 220)
  (h5 : sales_increase = 20) :
  ∃ y : ℤ, y = (24 - x) * (initial_sales + sales_increase * x) :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_l238_23863


namespace NUMINAMATH_CALUDE_largest_sum_is_ten_l238_23899

/-- A structure representing a set of five positive integers -/
structure FiveIntegers where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+

/-- The property that the sum of the five integers equals their product -/
def hasSumProductProperty (x : FiveIntegers) : Prop :=
  x.a + x.b + x.c + x.d + x.e = x.a * x.b * x.c * x.d * x.e

/-- The sum of the five integers -/
def sum (x : FiveIntegers) : ℕ :=
  x.a + x.b + x.c + x.d + x.e

/-- The theorem stating that (1, 1, 1, 2, 5) has the largest sum among all valid sets -/
theorem largest_sum_is_ten :
  ∀ x : FiveIntegers, hasSumProductProperty x → sum x ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_largest_sum_is_ten_l238_23899


namespace NUMINAMATH_CALUDE_m_always_composite_l238_23849

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The function m defined as n^4 + 64 -/
def m (n : ℕ) : ℕ :=
  n^4 + 64

/-- Theorem stating that m(n) is always composite for any natural number n -/
theorem m_always_composite :
  ∀ n : ℕ, IsComposite (m n) :=
by
  sorry


end NUMINAMATH_CALUDE_m_always_composite_l238_23849


namespace NUMINAMATH_CALUDE_non_negative_sequence_l238_23861

theorem non_negative_sequence (a : Fin 100 → ℝ) 
  (h1 : ∀ i : Fin 98, a i - 2 * a (i + 1) + a (i + 2) ≤ 0)
  (h2 : a 0 = a 99)
  (h3 : a 0 ≥ 0) : 
  ∀ i : Fin 100, a i ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_non_negative_sequence_l238_23861


namespace NUMINAMATH_CALUDE_inequality_proof_l238_23812

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + b^2 + c^2 = 14) : a^5 + (1/8)*b^5 + (1/27)*c^5 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l238_23812


namespace NUMINAMATH_CALUDE_geometric_sum_divisors_l238_23887

/-- The sum of geometric series from 0 to n with ratio a -/
def geometric_sum (a : ℕ) (n : ℕ) : ℕ :=
  (a^(n+1) - 1) / (a - 1)

/-- The set of all divisors of geometric_sum a n for some n -/
def divisor_set (a : ℕ) : Set ℕ :=
  {m : ℕ | ∃ n : ℕ, (geometric_sum a n) % m = 0}

/-- The set of all natural numbers relatively prime to a -/
def coprime_set (a : ℕ) : Set ℕ :=
  {m : ℕ | Nat.gcd m a = 1}

theorem geometric_sum_divisors (a : ℕ) (h : a > 1) :
  divisor_set a = coprime_set a :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_divisors_l238_23887


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l238_23823

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The given conditions for the triangle -/
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c ∧
  Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C

/-- Theorem: If a triangle satisfies the given conditions, then it is equilateral -/
theorem triangle_is_equilateral (t : Triangle) 
  (h : satisfiesConditions t) : t.isEquilateral := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l238_23823


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l238_23847

/-- Given a line with equation y - 5 = 3(x - 9), the sum of its x-intercept and y-intercept is -44/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 5 = 3 * (x - 9)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 5 = 3 * (x_int - 9)) ∧ 
    (0 - 5 = 3 * (x_int - 9)) ∧ 
    (y_int - 5 = 3 * (0 - 9)) ∧ 
    (x_int + y_int = -44/3)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l238_23847


namespace NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l238_23860

theorem unique_solution_to_diophantine_equation :
  ∃! (x y z n : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ n ≥ 2 ∧
    z ≤ 5 * 2^(2*n) ∧
    x^(2*n + 1) - y^(2*n + 1) = x * y * z + 2^(2*n + 1) ∧
    x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l238_23860


namespace NUMINAMATH_CALUDE_cm_per_inch_l238_23836

/-- Theorem: Given the map scale and measured distance, prove the number of centimeters in one inch -/
theorem cm_per_inch (map_scale_inches : Real) (map_scale_miles : Real) 
  (measured_cm : Real) (measured_miles : Real) :
  map_scale_inches = 1.5 →
  map_scale_miles = 24 →
  measured_cm = 47 →
  measured_miles = 296.06299212598424 →
  (measured_cm / (measured_miles / (map_scale_miles / map_scale_inches))) = 2.54 :=
by sorry

end NUMINAMATH_CALUDE_cm_per_inch_l238_23836


namespace NUMINAMATH_CALUDE_math_club_team_selection_l238_23808

theorem math_club_team_selection (boys girls team_size : ℕ) 
  (h1 : boys = 7) 
  (h2 : girls = 9) 
  (h3 : team_size = 5) : 
  Nat.choose (boys + girls) team_size = 4368 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l238_23808


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l238_23896

theorem polynomial_coefficients_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁ * (x - 1)^4 + a₂ * (x - 1)^3 + a₃ * (x - 1)^2 + a₄ * (x - 1) + a₅ = x^4) →
  a₂ + a₃ + a₄ = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l238_23896


namespace NUMINAMATH_CALUDE_triangle_properties_l238_23833

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  2 * a * Real.sin (C + π / 6) = b + c →
  B = π / 4 →
  b - a = Real.sqrt 2 - Real.sqrt 3 →
  A = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l238_23833


namespace NUMINAMATH_CALUDE_expression_simplification_l238_23890

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l238_23890


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l238_23876

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 - r2)^2

/-- The equation of the first circle: x^2 + y^2 - 2x = 0 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The equation of the second circle: x^2 + y^2 - 2x - 6y - 6 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 6 = 0

theorem circles_internally_tangent :
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ),
    (∀ x y, circle1 x y ↔ (x - c1.1)^2 + (y - c1.2)^2 = r1^2) ∧
    (∀ x y, circle2 x y ↔ (x - c2.1)^2 + (y - c2.2)^2 = r2^2) ∧
    internally_tangent c1 c2 r1 r2 :=
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l238_23876


namespace NUMINAMATH_CALUDE_cindy_solution_l238_23882

def cindy_problem (x : ℝ) : Prop :=
  (x - 12) / 4 = 32 →
  round ((x - 7) / 5) = 27

theorem cindy_solution : ∃ x : ℝ, cindy_problem x := by
  sorry

end NUMINAMATH_CALUDE_cindy_solution_l238_23882


namespace NUMINAMATH_CALUDE_prob_at_most_two_heads_prove_prob_at_most_two_heads_l238_23853

/-- The probability of getting at most 2 heads when tossing three unbiased coins -/
theorem prob_at_most_two_heads : ℚ :=
  7 / 8

/-- Prove that the probability of getting at most 2 heads when tossing three unbiased coins is 7/8 -/
theorem prove_prob_at_most_two_heads :
  prob_at_most_two_heads = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_two_heads_prove_prob_at_most_two_heads_l238_23853


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l238_23826

/-- The number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans in the Senate committee -/
def num_republicans : ℕ := 4

/-- The total number of politicians in the Senate committee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- The number of gaps between Democrats where Republicans can be placed -/
def num_gaps : ℕ := num_democrats

/-- Function to calculate the number of valid seating arrangements -/
def seating_arrangements (d r : ℕ) : ℕ :=
  (Nat.factorial (d - 1)) * (Nat.choose d r) * (Nat.factorial r)

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_arrangements :
  seating_arrangements num_democrats num_republicans = 43200 := by
  sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l238_23826


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l238_23807

theorem nested_fraction_equality : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l238_23807


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l238_23862

/-- Given a geometric sequence with third term 12 and fourth term 18, prove that the first term is 16/3 and the second term is 8. -/
theorem geometric_sequence_terms (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 12 →                    -- Third term is 12
  a 4 = 18 →                    -- Fourth term is 18
  a 1 = 16 / 3 ∧ a 2 = 8 :=     -- First term is 16/3 and second term is 8
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l238_23862


namespace NUMINAMATH_CALUDE_hyperbola_foci_and_incenter_l238_23880

/-- Definition of the hyperbola C -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-5, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (5, 0)

/-- Definition of a point being on the left branch of the hyperbola -/
def on_left_branch (x y : ℝ) : Prop :=
  hyperbola x y ∧ x < 0

/-- The center of the incircle of a triangle -/
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- Definition of incenter calculation

theorem hyperbola_foci_and_incenter :
  (∀ x y : ℝ, hyperbola x y → 
    (F₁ = (-5, 0) ∧ F₂ = (5, 0))) ∧
  (∀ x y : ℝ, on_left_branch x y →
    (incenter F₁ (x, y) F₂).1 = -3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_and_incenter_l238_23880


namespace NUMINAMATH_CALUDE_max_cake_pieces_l238_23809

def cakeSize : ℕ := 50
def pieceSize1 : ℕ := 4
def pieceSize2 : ℕ := 6
def pieceSize3 : ℕ := 8

theorem max_cake_pieces :
  let maxLargePieces := (cakeSize / pieceSize3) ^ 2
  let remainingWidth := cakeSize - (cakeSize / pieceSize3) * pieceSize3
  let maxSmallPieces := 2 * (cakeSize / pieceSize1)
  maxLargePieces + maxSmallPieces = 60 :=
by sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l238_23809


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l238_23898

/-- The volume of a tetrahedron formed by non-adjacent vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side : ℝ) (h : cube_side = 8) :
  let tetrahedron_volume := (cube_side^3 * Real.sqrt 2) / 3
  tetrahedron_volume = (512 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l238_23898


namespace NUMINAMATH_CALUDE_sqrt_three_diamond_sqrt_three_l238_23813

-- Define the operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_three_diamond_sqrt_three : diamond (Real.sqrt 3) (Real.sqrt 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_diamond_sqrt_three_l238_23813


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l238_23802

/-- A hyperbola with one asymptote defined by x±y=0 and passing through (-1,-2) -/
structure Hyperbola where
  /-- One asymptote of the hyperbola is defined by x±y=0 -/
  asymptote : ∀ (x y : ℝ), x = y ∨ x = -y
  /-- The hyperbola passes through the point (-1,-2) -/
  passes_through : ∃ (f : ℝ → ℝ → ℝ), f (-1) (-2) = 0

/-- The standard equation of the hyperbola is y²/3 - x²/3 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  ∃ (f : ℝ → ℝ → ℝ), (∀ x y, f x y = y^2/3 - x^2/3 - 1) ∧ (∀ x y, f x y = 0 ↔ h.passes_through.choose x y = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l238_23802


namespace NUMINAMATH_CALUDE_discount_calculation_l238_23830

/-- Proves that a product with given cost and original prices, sold at a specific profit margin, 
    results in a particular discount percentage. -/
theorem discount_calculation (cost_price original_price : ℝ) 
  (profit_margin : ℝ) (discount_percentage : ℝ) : 
  cost_price = 200 → 
  original_price = 300 → 
  profit_margin = 0.05 →
  discount_percentage = 0.7 →
  (original_price * discount_percentage - cost_price) / cost_price = profit_margin :=
by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l238_23830


namespace NUMINAMATH_CALUDE_percentage_problem_l238_23866

theorem percentage_problem (P : ℝ) : 
  (0.3 * 200 = P / 100 * 50 + 30) → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l238_23866


namespace NUMINAMATH_CALUDE_max_fourth_number_l238_23864

def numbers : Finset Nat := {39, 41, 44, 45, 47, 52, 55}

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.toFinset = numbers ∧
  ∀ i, i + 2 < arr.length → (arr[i]! + arr[i+1]! + arr[i+2]!) % 3 = 0

theorem max_fourth_number :
  ∃ (arr : List Nat), is_valid_arrangement arr ∧
    ∀ (other_arr : List Nat), is_valid_arrangement other_arr →
      arr[3]! ≥ other_arr[3]! ∧ arr[3]! = 47 :=
sorry

end NUMINAMATH_CALUDE_max_fourth_number_l238_23864


namespace NUMINAMATH_CALUDE_tan_range_problem_l238_23842

open Real Set

theorem tan_range_problem (m : ℝ) : 
  (∃ x ∈ Icc 0 (π/4), ¬(tan x < m)) ↔ m ∈ Iic 1 :=
sorry

end NUMINAMATH_CALUDE_tan_range_problem_l238_23842


namespace NUMINAMATH_CALUDE_tiling_ways_2x12_l238_23869

/-- The number of ways to tile a 2 × n rectangle with 1 × 2 dominoes -/
def tiling_ways : ℕ → ℕ
  | 0 => 0  -- Added for completeness
  | 1 => 1
  | 2 => 2
  | n+3 => tiling_ways (n+2) + tiling_ways (n+1)

/-- Theorem: The number of ways to tile a 2 × 12 rectangle with 1 × 2 dominoes is 233 -/
theorem tiling_ways_2x12 : tiling_ways 12 = 233 := by
  sorry


end NUMINAMATH_CALUDE_tiling_ways_2x12_l238_23869


namespace NUMINAMATH_CALUDE_F_and_G_increasing_l238_23843

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define F and G
def F (x : ℝ) := f x + g x
def G (x : ℝ) := f x - g x

-- State the theorem
theorem F_and_G_increasing
  (h_f_increasing : ∀ x y, x < y → f x < f y)
  (h_inequality : ∀ x y, x ≠ y → (f x - f y)^2 > (g x - g y)^2) :
  (∀ x y, x < y → F f g x < F f g y) ∧ (∀ x y, x < y → G f g x < G f g y) :=
sorry

end NUMINAMATH_CALUDE_F_and_G_increasing_l238_23843


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_14_9_l238_23824

/-- A quadratic function f(x) = ax^2 + bx + c with vertex at (6, -2) and passing through (3, 0) -/
def QuadraticFunction (a b c : ℚ) : ℚ → ℚ :=
  fun x ↦ a * x^2 + b * x + c

theorem sum_of_coefficients_equals_14_9 (a b c : ℚ) :
  (QuadraticFunction a b c 6 = -2) →  -- vertex at (6, -2)
  (QuadraticFunction a b c 3 = 0) →   -- passes through (3, 0)
  (∀ x, QuadraticFunction a b c (12 - x) = QuadraticFunction a b c x) →  -- vertical symmetry
  a + b + c = 14 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_14_9_l238_23824


namespace NUMINAMATH_CALUDE_quadratic_factorization_l238_23818

/-- Factorization of a quadratic expression -/
theorem quadratic_factorization (a : ℝ) : a^2 - 8*a + 16 = (a - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l238_23818


namespace NUMINAMATH_CALUDE_pictures_per_album_l238_23806

theorem pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) (pictures_per_album : ℕ) : 
  total_pictures = 24 → 
  num_albums = 4 → 
  total_pictures = num_albums * pictures_per_album →
  pictures_per_album = 6 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l238_23806


namespace NUMINAMATH_CALUDE_min_sticks_removal_part_a_result_part_b_result_l238_23804

/-- Represents a rectangular fence made of sticks -/
structure Fence where
  m : Nat
  n : Nat
  sticks : Nat

/-- The number of ants in a fence is equal to the number of 1x1 squares -/
def num_ants (f : Fence) : Nat := f.m * f.n

/-- The minimum number of sticks to remove for all ants to escape -/
def min_sticks_to_remove (f : Fence) : Nat := num_ants f

/-- Theorem: The minimum number of sticks to remove for all ants to escape
    is equal to the number of ants in the fence -/
theorem min_sticks_removal (f : Fence) :
  min_sticks_to_remove f = num_ants f :=
by sorry

/-- Corollary: For a 1x4 fence with 13 sticks, 4 sticks need to be removed -/
theorem part_a_result :
  min_sticks_to_remove ⟨1, 4, 13⟩ = 4 :=
by sorry

/-- Corollary: For a 4x4 fence with 24 sticks, 9 sticks need to be removed -/
theorem part_b_result :
  min_sticks_to_remove ⟨4, 4, 24⟩ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sticks_removal_part_a_result_part_b_result_l238_23804


namespace NUMINAMATH_CALUDE_down_payment_amount_l238_23893

/-- Given a purchase with a payment plan, prove the down payment amount. -/
theorem down_payment_amount
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (num_payments : ℕ)
  (interest_rate : ℝ)
  (h1 : purchase_price = 110)
  (h2 : monthly_payment = 10)
  (h3 : num_payments = 12)
  (h4 : interest_rate = 9.090909090909092 / 100) :
  ∃ (down_payment : ℝ),
    down_payment + num_payments * monthly_payment =
      purchase_price + interest_rate * purchase_price ∧
    down_payment = 0 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_amount_l238_23893


namespace NUMINAMATH_CALUDE_room_volume_example_l238_23881

/-- The volume of a rectangular room -/
def room_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a room with dimensions 100 m * 10 m * 10 m is 10,000 cubic meters -/
theorem room_volume_example : room_volume 100 10 10 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_room_volume_example_l238_23881


namespace NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l238_23865

/-- Calculates the additional money needed for Mrs. Smith's shopping --/
theorem additional_money_needed (total_budget dress_budget shoe_budget accessory_budget : ℚ)
  (dress_discount shoe_discount accessory_discount : ℚ) : ℚ :=
  let dress_needed := dress_budget * (1 + 2/5)
  let shoe_needed := shoe_budget * (1 + 2/5)
  let accessory_needed := accessory_budget * (1 + 2/5)
  let dress_discounted := dress_needed * (1 - dress_discount)
  let shoe_discounted := shoe_needed * (1 - shoe_discount)
  let accessory_discounted := accessory_needed * (1 - accessory_discount)
  let total_needed := dress_discounted + shoe_discounted + accessory_discounted
  total_needed - total_budget

/-- Proves that Mrs. Smith needs $84.50 more to complete her shopping --/
theorem mrs_smith_shopping :
  additional_money_needed 500 300 150 50 (20/100) (10/100) (15/100) = 169/2 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l238_23865


namespace NUMINAMATH_CALUDE_gcd_binomial_divisibility_l238_23897

theorem gcd_binomial_divisibility (m n : ℕ) (h1 : 0 < m) (h2 : m ≤ n) : 
  ∃ k : ℤ, (Int.gcd m n : ℚ) / n * (n.choose m) = k := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_divisibility_l238_23897


namespace NUMINAMATH_CALUDE_cube_volume_proof_l238_23815

theorem cube_volume_proof (a b c : ℝ) 
  (h1 : a^2 + b^2 = 81)
  (h2 : a^2 + c^2 = 169)
  (h3 : c^2 + b^2 = 196) :
  a * b * c = 18 * Real.sqrt 71 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_proof_l238_23815


namespace NUMINAMATH_CALUDE_minimum_value_and_inequality_l238_23819

def f (x : ℝ) : ℝ := |x + 3| + |x - 1|

theorem minimum_value_and_inequality (p q r : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 4) ∧
  (p^2 + 2*q^2 + r^2 = 4 → q*(p + r) ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_and_inequality_l238_23819


namespace NUMINAMATH_CALUDE_tom_dancing_hours_l238_23801

/-- Calculates the total dancing hours over multiple years -/
def total_dancing_hours (sessions_per_week : ℕ) (hours_per_session : ℕ) (years : ℕ) (weeks_per_year : ℕ) : ℕ :=
  sessions_per_week * hours_per_session * years * weeks_per_year

/-- Proves that Tom's total dancing hours over 10 years is 4160 -/
theorem tom_dancing_hours : 
  total_dancing_hours 4 2 10 52 = 4160 := by
  sorry

#eval total_dancing_hours 4 2 10 52

end NUMINAMATH_CALUDE_tom_dancing_hours_l238_23801


namespace NUMINAMATH_CALUDE_money_combination_l238_23852

theorem money_combination (raquel nataly tom sam : ℝ) : 
  tom = (1/4) * nataly →
  nataly = 3 * raquel →
  sam = 2 * nataly →
  raquel = 40 →
  tom + raquel + nataly + sam = 430 :=
by sorry

end NUMINAMATH_CALUDE_money_combination_l238_23852


namespace NUMINAMATH_CALUDE_cone_lateral_area_l238_23859

/-- The lateral area of a cone with a central angle of 60° and base radius of 8 is 384π. -/
theorem cone_lateral_area : 
  ∀ (r : ℝ) (central_angle : ℝ) (lateral_area : ℝ),
  r = 8 →
  central_angle = 60 * π / 180 →
  lateral_area = π * r * (2 * π * r) / central_angle →
  lateral_area = 384 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l238_23859


namespace NUMINAMATH_CALUDE_sixth_result_proof_l238_23889

theorem sixth_result_proof (total_results : ℕ) (first_group : ℕ) (last_group : ℕ)
  (total_average : ℚ) (first_average : ℚ) (last_average : ℚ)
  (h1 : total_results = 11)
  (h2 : first_group = 6)
  (h3 : last_group = 6)
  (h4 : total_average = 60)
  (h5 : first_average = 58)
  (h6 : last_average = 63) :
  ∃ (sixth_result : ℚ), sixth_result = 66 := by
sorry

end NUMINAMATH_CALUDE_sixth_result_proof_l238_23889


namespace NUMINAMATH_CALUDE_donut_combinations_l238_23851

theorem donut_combinations : 
  let total_donuts : ℕ := 8
  let donut_types : ℕ := 5
  let remaining_donuts : ℕ := total_donuts - donut_types
  Nat.choose (remaining_donuts + donut_types - 1) (donut_types - 1) = 35 :=
by sorry

end NUMINAMATH_CALUDE_donut_combinations_l238_23851


namespace NUMINAMATH_CALUDE_apple_transport_trucks_l238_23803

theorem apple_transport_trucks (total_apples : ℕ) (transported_apples : ℕ) (truck_capacity : ℕ) 
  (h1 : total_apples = 80)
  (h2 : transported_apples = 56)
  (h3 : truck_capacity = 4)
  : (total_apples - transported_apples) / truck_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_transport_trucks_l238_23803


namespace NUMINAMATH_CALUDE_second_number_is_30_l238_23822

theorem second_number_is_30 (x y : ℤ) : 
  y = x + 4 →  -- The second number is 4 more than the first
  x + y = 56 → -- The sum of the two numbers is 56
  y = 30       -- The second number is 30
:= by sorry

end NUMINAMATH_CALUDE_second_number_is_30_l238_23822


namespace NUMINAMATH_CALUDE_inequality_equivalence_l238_23821

theorem inequality_equivalence (x : ℝ) : 
  -1 < (x^2 - 10*x + 9) / (x^2 - 4*x + 5) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 5) < 1 ↔ x > 5.3 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l238_23821


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l238_23857

theorem dividend_percentage_calculation 
  (face_value : ℝ) 
  (purchase_price : ℝ) 
  (roi : ℝ) 
  (h1 : face_value = 50) 
  (h2 : purchase_price = 25) 
  (h3 : roi = 0.25) :
  let dividend_per_share := roi * purchase_price
  let dividend_percentage := (dividend_per_share / face_value) * 100
  dividend_percentage = 12.5 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l238_23857


namespace NUMINAMATH_CALUDE_gas_station_distance_l238_23829

theorem gas_station_distance (x : ℝ) : 
  (¬ (x ≥ 10)) →   -- Adam's statement is false
  (¬ (x ≤ 7)) →    -- Betty's statement is false
  (¬ (x < 5)) →    -- Carol's statement is false
  (¬ (x ≤ 9)) →    -- Dave's statement is false
  x > 9 :=         -- Conclusion: x is in the interval (9, ∞)
by
  sorry            -- Proof is omitted

#check gas_station_distance

end NUMINAMATH_CALUDE_gas_station_distance_l238_23829


namespace NUMINAMATH_CALUDE_placemat_length_l238_23816

theorem placemat_length (R : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) :
  R = 5 →
  n = 8 →
  w = 2 →
  y = 2 * R * Real.sin (π / (2 * n)) →
  y = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_placemat_length_l238_23816


namespace NUMINAMATH_CALUDE_sum_reciprocals_squared_l238_23827

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

-- State the theorem
theorem sum_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 96/529 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_squared_l238_23827


namespace NUMINAMATH_CALUDE_number_times_a_equals_7b_l238_23858

theorem number_times_a_equals_7b (a b x : ℝ) : 
  x * a = 7 * b → 
  x * a = 20 → 
  7 * b = 20 → 
  84 * a * b = 800 → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_number_times_a_equals_7b_l238_23858


namespace NUMINAMATH_CALUDE_quadratic_factorization_l238_23888

theorem quadratic_factorization (m n : ℝ) : 
  (∃ (x : ℝ), x^2 - m*x + n = 0) ∧ 
  (3 : ℝ)^2 - m*(3 : ℝ) + n = 0 ∧ 
  (-4 : ℝ)^2 - m*(-4 : ℝ) + n = 0 →
  ∀ (x : ℝ), x^2 - m*x + n = (x - 3)*(x + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l238_23888


namespace NUMINAMATH_CALUDE_systematic_sampling_524_l238_23831

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  populationSize : Nat
  samplingInterval : Nat

/-- Checks if the sampling interval divides the population size evenly -/
def SystematicSampling.isValidInterval (s : SystematicSampling) : Prop :=
  s.populationSize % s.samplingInterval = 0

theorem systematic_sampling_524 :
  ∃ (s : SystematicSampling), s.populationSize = 524 ∧ s.samplingInterval = 4 ∧ s.isValidInterval :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_524_l238_23831


namespace NUMINAMATH_CALUDE_average_of_distinct_t_is_22_3_l238_23868

/-- Given a polynomial x^2 - 6x + t with only positive integer roots,
    this function returns the average of all distinct possible values of t. -/
def averageOfDistinctT : ℚ :=
  22 / 3

/-- The polynomial x^2 - 6x + t has only positive integer roots. -/
axiom has_positive_integer_roots (t : ℤ) : 
  ∃ (r₁ r₂ : ℕ+), r₁.val * r₁.val - 6 * r₁.val + t = 0 ∧ 
                  r₂.val * r₂.val - 6 * r₂.val + t = 0

/-- The main theorem stating that the average of all distinct possible values of t
    for the polynomial x^2 - 6x + t with only positive integer roots is 22/3. -/
theorem average_of_distinct_t_is_22_3 :
  averageOfDistinctT = 22 / 3 :=
sorry

end NUMINAMATH_CALUDE_average_of_distinct_t_is_22_3_l238_23868


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l238_23885

/-- The minimum number of gloves required for a hiking team -/
def min_gloves (participants : ℕ) : ℕ := 2 * participants

/-- Theorem: For 82 participants, the minimum number of gloves required is 164 -/
theorem hiking_team_gloves : min_gloves 82 = 164 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_gloves_l238_23885


namespace NUMINAMATH_CALUDE_additional_discount_calculation_l238_23837

-- Define the manufacturer's suggested retail price (MSRP)
def MSRP : ℝ := 30

-- Define the regular discount range
def regularDiscountMin : ℝ := 0.1
def regularDiscountMax : ℝ := 0.3

-- Define the lowest possible price after all discounts
def lowestPrice : ℝ := 16.8

-- Define the additional discount percentage
def additionalDiscount : ℝ := 0.2

-- Theorem statement
theorem additional_discount_calculation :
  ∃ (regularDiscount : ℝ),
    regularDiscountMin ≤ regularDiscount ∧ 
    regularDiscount ≤ regularDiscountMax ∧
    MSRP * (1 - regularDiscount) * (1 - additionalDiscount) = lowestPrice :=
  sorry

end NUMINAMATH_CALUDE_additional_discount_calculation_l238_23837


namespace NUMINAMATH_CALUDE_james_purchase_cost_l238_23848

theorem james_purchase_cost : 
  let num_shirts : ℕ := 10
  let num_pants : ℕ := num_shirts / 2
  let shirt_cost : ℕ := 6
  let pants_cost : ℕ := 8
  let total_cost : ℕ := num_shirts * shirt_cost + num_pants * pants_cost
  total_cost = 100 := by sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l238_23848


namespace NUMINAMATH_CALUDE_unique_prime_solution_l238_23811

theorem unique_prime_solution :
  ∃! (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r ∧
    p < q ∧ q < r ∧
    25 * p * q + r = 2004 ∧
    ∃ m : ℕ, p * q * r + 1 = m * m ∧
    p = 7 ∧ q = 11 ∧ r = 79 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l238_23811
