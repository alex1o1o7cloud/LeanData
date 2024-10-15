import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_in_third_quadrant_l295_29546

/-- A line passes through the first, second, and third quadrants -/
structure LineInQuadrants (a b : ℝ) : Prop :=
  (passes_through_123 : a > 0 ∧ b > 0)

/-- A circle with center (-a, -b) and radius r -/
structure Circle (a b r : ℝ) : Prop :=
  (positive_radius : r > 0)

/-- The third quadrant -/
def ThirdQuadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem circle_center_in_third_quadrant
  (a b r : ℝ) (line : LineInQuadrants a b) (circle : Circle a b r) :
  ThirdQuadrant (-a) (-b) :=
sorry

end NUMINAMATH_CALUDE_circle_center_in_third_quadrant_l295_29546


namespace NUMINAMATH_CALUDE_adults_group_size_l295_29594

/-- The number of children in each group -/
def children_per_group : ℕ := 15

/-- The minimum number of adults (and children) attending -/
def min_attendees : ℕ := 255

/-- The number of adults in each group -/
def adults_per_group : ℕ := 15

theorem adults_group_size :
  (min_attendees % children_per_group = 0) →
  (min_attendees % adults_per_group = 0) →
  (min_attendees / children_per_group = min_attendees / adults_per_group) →
  adults_per_group = 15 := by
  sorry

end NUMINAMATH_CALUDE_adults_group_size_l295_29594


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l295_29539

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l295_29539


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l295_29501

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x - 11

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | polynomial a₂ a₁ x = 0} ⊆ {-11, -1, 1, 11} :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l295_29501


namespace NUMINAMATH_CALUDE_compound_weight_l295_29592

-- Define the atomic weights
def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

-- Define the total molecular weight
def total_weight : ℝ := 68

-- Define the number of oxygen atoms
def n : ℕ := 2

-- Theorem statement
theorem compound_weight :
  atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O = total_weight :=
by sorry

end NUMINAMATH_CALUDE_compound_weight_l295_29592


namespace NUMINAMATH_CALUDE_functional_equation_solution_l295_29562

theorem functional_equation_solution (f : ℕ → ℕ) 
  (h : ∀ x y : ℕ, f (x + y) = f x + f y) : 
  ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l295_29562


namespace NUMINAMATH_CALUDE_range_a_theorem_l295_29551

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (a > -2 ∧ a < -1) ∨ a ≥ 1

-- Theorem statement
theorem range_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l295_29551


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_rectangle_l295_29553

/-- Eccentricity of an ellipse with foci at opposite corners of a 4x3 rectangle 
    and passing through the other two corners -/
theorem ellipse_eccentricity_rectangle (a b c : ℝ) : 
  a = 4 →
  b = 3 →
  c = 2 →
  b^2 = 3*a →
  a^2 - c^2 = b^2 →
  c/a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_rectangle_l295_29553


namespace NUMINAMATH_CALUDE_age_problem_l295_29550

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l295_29550


namespace NUMINAMATH_CALUDE_largest_common_term_l295_29581

def isInFirstSequence (x : ℕ) : Prop := ∃ n : ℕ, x = 3 + 8 * n

def isInSecondSequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 9 * m

theorem largest_common_term : 
  (∃ x : ℕ, x ≤ 200 ∧ isInFirstSequence x ∧ isInSecondSequence x ∧ 
    ∀ y : ℕ, y ≤ 200 → isInFirstSequence y → isInSecondSequence y → y ≤ x) ∧
  (∃ x : ℕ, x = 131 ∧ x ≤ 200 ∧ isInFirstSequence x ∧ isInSecondSequence x) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l295_29581


namespace NUMINAMATH_CALUDE_trees_after_typhoon_l295_29558

/-- The number of trees Haley initially grew -/
def initial_trees : ℕ := 17

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 5

/-- Theorem stating that the number of trees left after the typhoon is 12 -/
theorem trees_after_typhoon : initial_trees - dead_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_trees_after_typhoon_l295_29558


namespace NUMINAMATH_CALUDE_max_gcd_lcm_product_l295_29555

theorem max_gcd_lcm_product (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a₀ b₀ c₀ : ℕ), Nat.gcd (Nat.lcm a₀ b₀) c₀ = 10 ∧
    Nat.gcd (Nat.lcm a₀ b₀) c₀ * Nat.lcm (Nat.gcd a₀ b₀) c₀ = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_product_l295_29555


namespace NUMINAMATH_CALUDE_carters_reading_rate_l295_29590

/-- Given reading rates for Oliver, Lucy, and Carter, prove Carter's reading rate -/
theorem carters_reading_rate 
  (oliver_rate : ℕ) 
  (lucy_rate : ℕ) 
  (carter_rate : ℕ) 
  (h1 : oliver_rate = 40)
  (h2 : lucy_rate = oliver_rate + 20)
  (h3 : carter_rate = lucy_rate / 2) : 
  carter_rate = 30 := by
sorry

end NUMINAMATH_CALUDE_carters_reading_rate_l295_29590


namespace NUMINAMATH_CALUDE_expression_equality_l295_29569

theorem expression_equality (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l295_29569


namespace NUMINAMATH_CALUDE_red_white_flowers_l295_29588

/-- Represents the number of flowers of each color combination --/
structure FlowerCounts where
  total : ℕ
  yellowWhite : ℕ
  redYellow : ℕ
  redWhite : ℕ

/-- The difference between flowers containing red and white --/
def redWhiteDifference (f : FlowerCounts) : ℤ :=
  (f.redYellow + f.redWhite : ℤ) - (f.yellowWhite + f.redWhite : ℤ)

/-- Theorem stating the number of red and white flowers --/
theorem red_white_flowers (f : FlowerCounts) 
  (h_total : f.total = 44)
  (h_yellowWhite : f.yellowWhite = 13)
  (h_redYellow : f.redYellow = 17)
  (h_redWhiteDiff : redWhiteDifference f = 4) :
  f.redWhite = 14 := by
  sorry

end NUMINAMATH_CALUDE_red_white_flowers_l295_29588


namespace NUMINAMATH_CALUDE_constant_segments_am_plus_bn_equals_11_am_equals_bn_l295_29571

-- Define the points on the number line
def A (t : ℝ) : ℝ := -1 + 2*t
def M (t : ℝ) : ℝ := t
def N (t : ℝ) : ℝ := t + 2
def B (t : ℝ) : ℝ := 11 - t

-- Theorem for part 1
theorem constant_segments :
  ∀ x t : ℝ, abs (B t - A t) = 12 ∧ abs (N t - M t) = 2 :=
sorry

-- Theorem for part 2, question 1
theorem am_plus_bn_equals_11 :
  ∃ t : ℝ, abs (M t - A t) + abs (B t - N t) = 11 ∧ t = 9.5 :=
sorry

-- Theorem for part 2, question 2
theorem am_equals_bn :
  ∃ t₁ t₂ : ℝ, 
    abs (M t₁ - A t₁) = abs (B t₁ - N t₁) ∧
    abs (M t₂ - A t₂) = abs (B t₂ - N t₂) ∧
    t₁ = 10/3 ∧ t₂ = 8 :=
sorry

end NUMINAMATH_CALUDE_constant_segments_am_plus_bn_equals_11_am_equals_bn_l295_29571


namespace NUMINAMATH_CALUDE_irreducible_fraction_l295_29579

theorem irreducible_fraction (n : ℕ) :
  (Nat.gcd (n^3 + n) (2*n + 1) = 1) ↔ (n % 5 ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l295_29579


namespace NUMINAMATH_CALUDE_rectangle_diagonals_not_always_perpendicular_and_equal_l295_29536

/-- A rectangle is a quadrilateral with four right angles -/
structure Rectangle where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  is_right_angle : ∀ i, angles i = π / 2

/-- Diagonals of a shape -/
def diagonals (r : Rectangle) : Fin 2 → ℝ := sorry

/-- Two real numbers are equal -/
def are_equal (a b : ℝ) : Prop := a = b

/-- Two lines are perpendicular if they form a right angle -/
def are_perpendicular (a b : ℝ) : Prop := sorry

theorem rectangle_diagonals_not_always_perpendicular_and_equal (r : Rectangle) : 
  ¬(are_equal (diagonals r 0) (diagonals r 1) ∧ are_perpendicular (diagonals r 0) (diagonals r 1)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonals_not_always_perpendicular_and_equal_l295_29536


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l295_29559

/-- A point in the second quadrant with |x| = 2 and |y| = 3 -/
structure PointM where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  abs_x_eq_two : |x| = 2
  abs_y_eq_three : |y| = 3

/-- The coordinates of a point symmetric to M with respect to the y-axis -/
def symmetric_point (m : PointM) : ℝ × ℝ := (-m.x, m.y)

theorem symmetric_point_coordinates (m : PointM) : 
  symmetric_point m = (2, 3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l295_29559


namespace NUMINAMATH_CALUDE_matrix_vector_multiplication_l295_29535

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; -3, 4]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![3; -1]

theorem matrix_vector_multiplication :
  A * v = !![7; -13] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_multiplication_l295_29535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_1000_l295_29523

def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_2_to_1000 :
  arithmetic_sequence_sum 2 1000 2 = 250500 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_1000_l295_29523


namespace NUMINAMATH_CALUDE_factorization_problems_l295_29552

theorem factorization_problems (x : ℝ) : 
  (9 * x^2 - 6 * x + 1 = (3 * x - 1)^2) ∧ 
  (x^3 - x = x * (x + 1) * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l295_29552


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_function_l295_29538

/-- A linear function of the form y = kx + k + 2 -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + k + 2

/-- The fixed point of the linear function -/
def fixedPoint : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the linear function for all k -/
theorem fixed_point_satisfies_function :
  ∀ k : ℝ, linearFunction k (fixedPoint.1) = fixedPoint.2 := by
  sorry

#check fixed_point_satisfies_function

end NUMINAMATH_CALUDE_fixed_point_satisfies_function_l295_29538


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l295_29585

theorem greatest_integer_satisfying_inequality :
  ∀ n : ℤ, (∀ x : ℤ, 3 * |2 * x + 1| + 10 > 28 → x ≤ n) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l295_29585


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l295_29591

theorem rhombus_diagonal_length (d1 : ℝ) (d2 : ℝ) (square_side : ℝ) 
  (h1 : d1 = 16)
  (h2 : square_side = 8)
  (h3 : d1 * d2 / 2 = square_side ^ 2) :
  d2 = 8 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l295_29591


namespace NUMINAMATH_CALUDE_triangle_inequality_l295_29586

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (Real.sqrt (b + c - a) / (Real.sqrt b + Real.sqrt c - Real.sqrt a)) +
  (Real.sqrt (c + a - b) / (Real.sqrt c + Real.sqrt a - Real.sqrt b)) +
  (Real.sqrt (a + b - c) / (Real.sqrt a + Real.sqrt b - Real.sqrt c)) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l295_29586


namespace NUMINAMATH_CALUDE_circplus_assoc_l295_29577

/-- The custom operation ⊕ on real numbers -/
def circplus (x y : ℝ) : ℝ := x + y - x * y

/-- Theorem stating that the ⊕ operation is associative -/
theorem circplus_assoc :
  ∀ (x y z : ℝ), circplus (circplus x y) z = circplus x (circplus y z) := by
  sorry

end NUMINAMATH_CALUDE_circplus_assoc_l295_29577


namespace NUMINAMATH_CALUDE_library_budget_is_3000_l295_29516

-- Define the total budget
def total_budget : ℝ := 20000

-- Define the library budget percentage
def library_percentage : ℝ := 0.15

-- Define the parks budget percentage
def parks_percentage : ℝ := 0.24

-- Define the remaining budget
def remaining_budget : ℝ := 12200

-- Theorem to prove
theorem library_budget_is_3000 :
  library_percentage * total_budget = 3000 :=
by
  sorry


end NUMINAMATH_CALUDE_library_budget_is_3000_l295_29516


namespace NUMINAMATH_CALUDE_range_of_m_l295_29576

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1
def C₂ (m : ℝ) (x y : ℝ) : Prop := y^2 = 2*(x + m)

-- Define the condition for a single common point above x-axis
def single_common_point (a m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, C₁ a p.1 p.2 ∧ C₂ m p.1 p.2 ∧ p.2 > 0

-- State the theorem
theorem range_of_m (a : ℝ) (h : a > 0) :
  (∀ m : ℝ, single_common_point a m →
    ((0 < a ∧ a < 1 → m = (a^2 + 1)/2 ∨ (-a < m ∧ m ≤ a)) ∧
     (a ≥ 1 → -a < m ∧ m < a))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l295_29576


namespace NUMINAMATH_CALUDE_smallest_fraction_divisible_l295_29545

theorem smallest_fraction_divisible (f1 f2 f3 : Rat) (h1 : f1 = 6/7) (h2 : f2 = 5/14) (h3 : f3 = 10/21) :
  (∀ q : Rat, (∃ n1 n2 n3 : ℤ, f1 * q = n1 ∧ f2 * q = n2 ∧ f3 * q = n3) →
    (1 : Rat) / 42 ≤ q) ∧
  (∃ n1 n2 n3 : ℤ, f1 * (1/42 : Rat) = n1 ∧ f2 * (1/42 : Rat) = n2 ∧ f3 * (1/42 : Rat) = n3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_divisible_l295_29545


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l295_29570

/-- Theorem: Relationship between heights of two cylinders with equal volume and different radii -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l295_29570


namespace NUMINAMATH_CALUDE_equidistant_arrangement_exists_l295_29526

/-- A move on a circular track -/
structure Move where
  person1 : Fin n
  person2 : Fin n
  distance : ℝ

/-- The state of people on a circular track -/
def TrackState (n : ℕ) := Fin n → ℝ

/-- Apply a move to a track state -/
def applyMove (state : TrackState n) (move : Move) : TrackState n :=
  fun i => if i = move.person1 then state i + move.distance
           else if i = move.person2 then state i - move.distance
           else state i

/-- Check if a track state is equidistant -/
def isEquidistant (state : TrackState n) : Prop :=
  ∀ i j : Fin n, (state i - state j) % 1 = (i - j : ℝ) / n

/-- Main theorem: it's possible to reach an equidistant state in at most n-1 moves -/
theorem equidistant_arrangement_exists (n : ℕ) (initial : TrackState n) :
  ∃ (moves : List Move), moves.length ≤ n - 1 ∧
    isEquidistant (moves.foldl applyMove initial) :=
sorry

end NUMINAMATH_CALUDE_equidistant_arrangement_exists_l295_29526


namespace NUMINAMATH_CALUDE_min_value_abc_l295_29531

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l295_29531


namespace NUMINAMATH_CALUDE_second_player_can_prevent_win_l295_29565

/-- Represents a position on the infinite grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a move in the game -/
inductive Move
  | X (pos : Position)
  | O (pos : Position)

/-- Represents the game state -/
def GameState := List Move

/-- A strategy for the second player -/
def Strategy := GameState → Position

/-- Checks if a list of positions contains 11 consecutive X's -/
def hasElevenConsecutiveXs (positions : List Position) : Prop :=
  sorry

/-- Checks if a game state has a winning condition for the first player -/
def isWinningState (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that the second player can prevent the first player from winning -/
theorem second_player_can_prevent_win :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      ¬(isWinningState game) :=
sorry

end NUMINAMATH_CALUDE_second_player_can_prevent_win_l295_29565


namespace NUMINAMATH_CALUDE_negation_equivalence_l295_29504

theorem negation_equivalence (a b : ℝ) : 
  (¬(a + b = 1 → a^2 + b^2 > 1)) ↔ (∃ a b : ℝ, a + b = 1 ∧ a^2 + b^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l295_29504


namespace NUMINAMATH_CALUDE_bobs_grade_l295_29561

theorem bobs_grade (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = jason_grade / 2 →
  bob_grade = 35 := by
  sorry

end NUMINAMATH_CALUDE_bobs_grade_l295_29561


namespace NUMINAMATH_CALUDE_number_exists_l295_29513

theorem number_exists : ∃ N : ℝ, (N / 10 - N / 1000) = 700 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l295_29513


namespace NUMINAMATH_CALUDE_percentage_relation_l295_29500

theorem percentage_relation (x y z : ℝ) (h1 : y = 0.6 * z) (h2 : x = 0.78 * z) :
  x = y * (1 + 0.3) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relation_l295_29500


namespace NUMINAMATH_CALUDE_blue_paint_cans_l295_29528

def blue_to_green_ratio : ℚ := 4 / 3
def total_cans : ℕ := 35

theorem blue_paint_cans : ℕ := by
  -- The number of cans of blue paint is 20
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l295_29528


namespace NUMINAMATH_CALUDE_line_slope_l295_29518

theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l295_29518


namespace NUMINAMATH_CALUDE_bo_number_l295_29587

theorem bo_number (a b : ℂ) : 
  a * b = 52 - 28 * I ∧ a = 7 + 4 * I → b = 476 / 65 - 404 / 65 * I :=
by sorry

end NUMINAMATH_CALUDE_bo_number_l295_29587


namespace NUMINAMATH_CALUDE_egg_count_and_weight_l295_29519

/-- Conversion factor from ounces to grams -/
def ouncesToGrams : ℝ := 28.3495

/-- Initial number of eggs -/
def initialEggs : ℕ := 47

/-- Number of whole eggs added -/
def addedEggs : ℕ := 5

/-- Total weight of eggs in ounces -/
def totalWeightOunces : ℝ := 143.5

theorem egg_count_and_weight :
  (initialEggs + addedEggs = 52) ∧
  (abs (totalWeightOunces * ouncesToGrams - 4067.86) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_egg_count_and_weight_l295_29519


namespace NUMINAMATH_CALUDE_horner_v1_equals_22_l295_29521

/-- Horner's Method for polynomial evaluation -/
def horner_step (coeff : ℝ) (x : ℝ) (prev : ℝ) : ℝ :=
  prev * x + coeff

/-- The polynomial f(x) = 4x⁵ + 2x⁴ + 3.5x³ - 2.6x² + 1.7x - 0.8 -/
def f (x : ℝ) : ℝ :=
  4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Theorem: The value of V₁ when calculating f(5) using Horner's Method is 22 -/
theorem horner_v1_equals_22 :
  let v0 := 4  -- Initialize V₀ with the coefficient of the highest degree term
  let v1 := horner_step 2 5 v0  -- Calculate V₁
  v1 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_v1_equals_22_l295_29521


namespace NUMINAMATH_CALUDE_train_pass_bridge_time_l295_29549

/-- Time for a train to pass a bridge -/
theorem train_pass_bridge_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 860)
  (h2 : train_speed_kmh = 85)
  (h3 : bridge_length = 450) :
  ∃ (t : ℝ), abs (t - 55.52) < 0.01 ∧ 
  t = (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) :=
sorry

end NUMINAMATH_CALUDE_train_pass_bridge_time_l295_29549


namespace NUMINAMATH_CALUDE_area_under_arcsin_cos_l295_29547

noncomputable def f (x : ℝ) := Real.arcsin (Real.cos x)

theorem area_under_arcsin_cos : ∫ x in (0)..(3 * Real.pi), |f x| = (3 * Real.pi^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_under_arcsin_cos_l295_29547


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l295_29541

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s, s = -(b / a) ∧ ∀ x y, f x = 0 → f y = 0 → x + y = s) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2023 * x - 2024
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s, s = -2023 ∧ ∀ x y, f x = 0 → f y = 0 → x + y = s) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l295_29541


namespace NUMINAMATH_CALUDE_painted_cube_probability_l295_29544

/-- The size of the cube's side -/
def cube_side : ℕ := 5

/-- The total number of unit cubes in the larger cube -/
def total_cubes : ℕ := cube_side ^ 3

/-- The number of unit cubes with exactly three painted faces -/
def three_painted_faces : ℕ := 1

/-- The number of unit cubes with no painted faces -/
def no_painted_faces : ℕ := (cube_side - 2) ^ 3

/-- The number of ways to choose two cubes out of the total -/
def total_combinations : ℕ := total_cubes.choose 2

/-- The number of successful outcomes -/
def successful_outcomes : ℕ := three_painted_faces * no_painted_faces

theorem painted_cube_probability :
  (successful_outcomes : ℚ) / total_combinations = 9 / 2583 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l295_29544


namespace NUMINAMATH_CALUDE_min_value_on_circle_l295_29560

theorem min_value_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) :
  ∃ (min : ℝ), (∀ (a b : ℝ), (a - 2)^2 + (b - 1)^2 = 1 → a^2 + b^2 ≥ min) ∧ min = 6 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l295_29560


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l295_29567

/-- Proves that for a group of 7 people with an average age of 30 and the youngest being 8 years old,
    the average age of the group when the youngest was born was 22 years. -/
theorem average_age_when_youngest_born
  (num_people : ℕ)
  (current_average_age : ℝ)
  (youngest_age : ℕ)
  (h_num_people : num_people = 7)
  (h_current_average : current_average_age = 30)
  (h_youngest : youngest_age = 8) :
  (num_people * current_average_age - num_people * youngest_age) / num_people = 22 :=
sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l295_29567


namespace NUMINAMATH_CALUDE_calculate_S_l295_29582

-- Define the relationship between R, S, and T
def relation (c : ℝ) (R S T : ℝ) : Prop :=
  R = c * (S^2 / T^2)

-- Define the theorem
theorem calculate_S (c : ℝ) (R₁ S₁ T₁ R₂ T₂ : ℝ) :
  relation c R₁ S₁ T₁ →
  R₁ = 9 →
  S₁ = 2 →
  T₁ = 3 →
  R₂ = 16 →
  T₂ = 4 →
  ∃ S₂, relation c R₂ S₂ T₂ ∧ S₂ = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_S_l295_29582


namespace NUMINAMATH_CALUDE_age_ratio_ten_years_ago_l295_29580

-- Define Alice's current age
def alice_current_age : ℕ := 30

-- Define the age difference between Alice and Tom
def age_difference : ℕ := 15

-- Define the number of years that have passed
def years_passed : ℕ := 10

-- Define Tom's current age
def tom_current_age : ℕ := alice_current_age - age_difference

-- Define Alice's age 10 years ago
def alice_past_age : ℕ := alice_current_age - years_passed

-- Define Tom's age 10 years ago
def tom_past_age : ℕ := tom_current_age - years_passed

-- Theorem to prove
theorem age_ratio_ten_years_ago :
  alice_past_age / tom_past_age = 4 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_ten_years_ago_l295_29580


namespace NUMINAMATH_CALUDE_selene_purchase_l295_29542

theorem selene_purchase (camera_price : ℝ) (frame_price : ℝ) (discount_rate : ℝ) (total_paid : ℝ) :
  camera_price = 110 →
  frame_price = 120 →
  discount_rate = 0.05 →
  total_paid = 551 →
  ∃ num_frames : ℕ,
    (1 - discount_rate) * (2 * camera_price + num_frames * frame_price) = total_paid ∧
    num_frames = 3 :=
by sorry

end NUMINAMATH_CALUDE_selene_purchase_l295_29542


namespace NUMINAMATH_CALUDE_maurice_cookout_beef_per_package_l295_29593

/-- Calculates the amount of ground beef per package for Maurice's cookout -/
theorem maurice_cookout_beef_per_package 
  (total_people : ℕ) 
  (beef_per_person : ℕ) 
  (num_packages : ℕ) 
  (h1 : total_people = 10) 
  (h2 : beef_per_person = 2) 
  (h3 : num_packages = 4) : 
  (total_people * beef_per_person) / num_packages = 5 := by
  sorry

#check maurice_cookout_beef_per_package

end NUMINAMATH_CALUDE_maurice_cookout_beef_per_package_l295_29593


namespace NUMINAMATH_CALUDE_larger_integer_value_l295_29507

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  a = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l295_29507


namespace NUMINAMATH_CALUDE_work_completion_l295_29595

theorem work_completion (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 6 → absent_men = 4 → final_days = 12 →
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_l295_29595


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l295_29543

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 3 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l295_29543


namespace NUMINAMATH_CALUDE_map_scale_l295_29598

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (scale : ℝ → ℝ) (h1 : scale 15 = 90) : scale 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l295_29598


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l295_29563

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l295_29563


namespace NUMINAMATH_CALUDE_tommys_quarters_l295_29511

/-- Tommy's coin collection problem -/
theorem tommys_quarters (P D N Q : ℕ) 
  (dimes_pennies : D = P + 10)
  (nickels_dimes : N = 2 * D)
  (pennies_quarters : P = 10 * Q)
  (total_nickels : N = 100) : Q = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommys_quarters_l295_29511


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l295_29537

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| > 4}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 6}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = Set.Ioo 4 6 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l295_29537


namespace NUMINAMATH_CALUDE_rhombus_area_l295_29534

/-- The area of a rhombus with specific side length and diagonal difference -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 →
  diag_diff = 8 →
  area = 6 * Real.sqrt 210 - 12 →
  ∃ (d1 d2 : ℝ), 
    d1 > 0 ∧ d2 > 0 ∧
    d2 - d1 = diag_diff ∧
    d1 * d2 / 2 = area ∧
    d1^2 / 4 + side^2 = (d2 / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l295_29534


namespace NUMINAMATH_CALUDE_trigonometric_identities_l295_29583

theorem trigonometric_identities :
  (∃ (tan10 tan20 tan23 tan37 : ℝ),
    tan10 = Real.tan (10 * π / 180) ∧
    tan20 = Real.tan (20 * π / 180) ∧
    tan23 = Real.tan (23 * π / 180) ∧
    tan37 = Real.tan (37 * π / 180) ∧
    tan10 * tan20 + Real.sqrt 3 * (tan10 + tan20) = 1 ∧
    tan23 + tan37 + Real.sqrt 3 * tan23 * tan37 = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l295_29583


namespace NUMINAMATH_CALUDE_K_change_implies_equilibrium_shift_l295_29589

-- Define the equilibrium constant as a function of temperature
def K (temperature : ℝ) : ℝ := sorry

-- Define a predicate for equilibrium shift
def equilibrium_shift (initial_state final_state : ℝ) : Prop :=
  initial_state ≠ final_state

-- Define a predicate for K change
def K_change (initial_K final_K : ℝ) : Prop :=
  initial_K ≠ final_K

-- Theorem statement
theorem K_change_implies_equilibrium_shift
  (initial_temp final_temp : ℝ)
  (h_K_change : K_change (K initial_temp) (K final_temp)) :
  equilibrium_shift initial_temp final_temp :=
sorry

end NUMINAMATH_CALUDE_K_change_implies_equilibrium_shift_l295_29589


namespace NUMINAMATH_CALUDE_exam_questions_unique_solution_l295_29506

theorem exam_questions_unique_solution (n : ℕ) : 
  (15 + (n - 20) / 3 : ℚ) / n = 1 / 2 → n = 50 :=
by sorry

end NUMINAMATH_CALUDE_exam_questions_unique_solution_l295_29506


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l295_29517

/-- Given that 9 oranges weigh the same as 6 apples, prove that 36 oranges weigh the same as 24 apples -/
theorem orange_apple_weight_equivalence 
  (orange_weight apple_weight : ℚ) 
  (h : 9 * orange_weight = 6 * apple_weight) : 
  36 * orange_weight = 24 * apple_weight := by
sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l295_29517


namespace NUMINAMATH_CALUDE_square_plus_eight_divisible_by_eleven_l295_29572

theorem square_plus_eight_divisible_by_eleven : 
  ∃ k : ℤ, 5^2 + 8 = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_square_plus_eight_divisible_by_eleven_l295_29572


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l295_29568

theorem simplify_fraction_expression :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l295_29568


namespace NUMINAMATH_CALUDE_two_digit_number_value_l295_29520

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Theorem: The value of a two-digit number is 10a + b, where a is the tens digit and b is the ones digit -/
theorem two_digit_number_value (n : TwoDigitNumber) : 
  n.value = 10 * n.tens + n.ones := by sorry

end NUMINAMATH_CALUDE_two_digit_number_value_l295_29520


namespace NUMINAMATH_CALUDE_problem_statement_l295_29512

theorem problem_statement (a b c t : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : t ≥ 1)
  (h5 : a + b + c = 1/2)
  (h6 : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6*t) / 2) :
  a^(2*t) + b^(2*t) + c^(2*t) = 1/12 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l295_29512


namespace NUMINAMATH_CALUDE_num_2d_faces_6cube_l295_29533

/-- The number of 2-D square faces in a 6-dimensional cube of side length 6 -/
def num_2d_faces (n : ℕ) (side_length : ℕ) : ℕ :=
  (Nat.choose n 4) * (side_length + 1)^4 * side_length^2

/-- Theorem stating the number of 2-D square faces in a 6-cube of side length 6 -/
theorem num_2d_faces_6cube :
  num_2d_faces 6 6 = 1296150 := by
  sorry

end NUMINAMATH_CALUDE_num_2d_faces_6cube_l295_29533


namespace NUMINAMATH_CALUDE_toms_hourly_wage_l295_29505

/-- Tom's hourly wage calculation --/
theorem toms_hourly_wage :
  let item_cost : ℝ := 25.35 + 70.69 + 85.96
  let hours_worked : ℕ := 31
  let savings_rate : ℝ := 0.1
  let hourly_wage : ℝ := item_cost / ((1 - savings_rate) * hours_worked)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |hourly_wage - 6.52| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_toms_hourly_wage_l295_29505


namespace NUMINAMATH_CALUDE_camera_cost_proof_l295_29509

/-- The cost of the old camera model --/
def old_camera_cost : ℝ := 4000

/-- The cost of the new camera model --/
def new_camera_cost : ℝ := old_camera_cost * 1.3

/-- The original price of the lens --/
def lens_original_price : ℝ := 400

/-- The discount on the lens --/
def lens_discount : ℝ := 200

/-- The discounted price of the lens --/
def lens_discounted_price : ℝ := lens_original_price - lens_discount

/-- The total amount paid for the new camera and the discounted lens --/
def total_paid : ℝ := 5400

theorem camera_cost_proof : 
  new_camera_cost + lens_discounted_price = total_paid ∧ 
  old_camera_cost = 4000 := by
  sorry

end NUMINAMATH_CALUDE_camera_cost_proof_l295_29509


namespace NUMINAMATH_CALUDE_percentage_6_plus_years_l295_29532

-- Define the number of marks for each year range
def marks : List Nat := [10, 4, 6, 5, 8, 3, 5, 4, 2, 2]

-- Define the total number of marks
def total_marks : Nat := marks.sum

-- Define the number of marks for 6 years or more
def marks_6_plus : Nat := (marks.drop 6).sum

-- Theorem to prove
theorem percentage_6_plus_years (ε : Real) (hε : ε > 0) :
  ∃ (p : Real), abs (p - 26.53) < ε ∧ p = (marks_6_plus * 100 : Real) / total_marks :=
sorry

end NUMINAMATH_CALUDE_percentage_6_plus_years_l295_29532


namespace NUMINAMATH_CALUDE_reading_time_proof_l295_29530

/-- Calculates the number of weeks needed to read a series of books -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  1 + (total_books - first_week + subsequent_weeks - 1) / subsequent_weeks

/-- Proves that reading 70 books takes 11 weeks when reading 5 books in the first week and 7 books per week thereafter -/
theorem reading_time_proof :
  weeks_to_read 70 5 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_proof_l295_29530


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l295_29515

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℝ := 60 + 11 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_price (x : ℕ) : ℝ := 20 + 15 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < gamma_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ gamma_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l295_29515


namespace NUMINAMATH_CALUDE_certain_number_value_l295_29525

theorem certain_number_value (a : ℝ) (x : ℝ) 
  (h1 : -6 * a^2 = x * (4 * a + 2)) 
  (h2 : -6 * 1^2 = x * (4 * 1 + 2)) : 
  x = -1 := by sorry

end NUMINAMATH_CALUDE_certain_number_value_l295_29525


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l295_29524

theorem percentage_of_defective_meters
  (total_meters : ℕ)
  (rejected_meters : ℕ)
  (h1 : total_meters = 8000)
  (h2 : rejected_meters = 4) :
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l295_29524


namespace NUMINAMATH_CALUDE_swimming_championship_races_swimming_championship_proof_l295_29508

/-- Calculate the number of races needed to determine a champion in a swimming competition. -/
theorem swimming_championship_races (total_swimmers : ℕ) 
  (swimmers_per_race : ℕ) (advancing_swimmers : ℕ) : ℕ :=
  let eliminated_per_race := swimmers_per_race - advancing_swimmers
  let total_eliminations := total_swimmers - 1
  ⌈(total_eliminations : ℚ) / eliminated_per_race⌉.toNat

/-- Prove that 53 races are required for 300 swimmers with 8 per race and 2 advancing. -/
theorem swimming_championship_proof : 
  swimming_championship_races 300 8 2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_swimming_championship_races_swimming_championship_proof_l295_29508


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l295_29510

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l295_29510


namespace NUMINAMATH_CALUDE_trebled_resultant_l295_29514

theorem trebled_resultant (x : ℕ) : x = 20 → 3 * ((2 * x) + 5) = 135 := by
  sorry

end NUMINAMATH_CALUDE_trebled_resultant_l295_29514


namespace NUMINAMATH_CALUDE_octagon_diagonals_l295_29503

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l295_29503


namespace NUMINAMATH_CALUDE_smallest_m_is_one_l295_29529

/-- The largest prime with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : q ≥ 10^2022 ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → (p ≥ 10^2022 ∧ p < 10^2023) → p ≤ q

/-- The smallest positive integer m such that q^2 - m is divisible by 15 -/
def m : ℕ := sorry

theorem smallest_m_is_one : m = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_m_is_one_l295_29529


namespace NUMINAMATH_CALUDE_range_of_a_l295_29584

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2*a) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l295_29584


namespace NUMINAMATH_CALUDE_probability_three_correct_is_one_sixth_l295_29540

/-- The probability of exactly 3 out of 5 packages being delivered to the correct houses in a random delivery -/
def probability_three_correct_deliveries : ℚ :=
  (Nat.choose 5 3 * 2) / Nat.factorial 5

/-- Theorem stating that the probability of exactly 3 out of 5 packages being delivered to the correct houses is 1/6 -/
theorem probability_three_correct_is_one_sixth :
  probability_three_correct_deliveries = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_correct_is_one_sixth_l295_29540


namespace NUMINAMATH_CALUDE_circle_in_second_quadrant_implies_a_range_l295_29557

/-- Definition of the circle equation -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0

/-- Definition of a point being in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem stating that if all points on the circle are in the second quadrant,
    then a is between 0 and 3 -/
theorem circle_in_second_quadrant_implies_a_range :
  (∀ x y : ℝ, circle_equation x y a → in_second_quadrant x y) →
  0 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_circle_in_second_quadrant_implies_a_range_l295_29557


namespace NUMINAMATH_CALUDE_safe_cracking_l295_29554

def Password := Fin 10 → Fin 10

def isValidPassword (p : Password) : Prop :=
  (∀ i j : Fin 7, i ≠ j → p i ≠ p j) ∧ (∀ i : Fin 7, p i < 10)

def Attempt := Fin 7 → Fin 10

def isSuccessfulAttempt (p : Password) (a : Attempt) : Prop :=
  ∃ i : Fin 7, p i = a i

theorem safe_cracking (p : Password) (h : isValidPassword p) :
  ∃ attempts : Fin 6 → Attempt,
    ∃ i : Fin 6, isSuccessfulAttempt p (attempts i) :=
sorry

end NUMINAMATH_CALUDE_safe_cracking_l295_29554


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l295_29574

theorem x_squared_plus_reciprocal (x : ℝ) (h : 20 = x^6 + 1/x^6) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l295_29574


namespace NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l295_29548

theorem square_cut_into_three_rectangles (square_side : ℝ) (cut_length : ℝ) : 
  square_side = 36 →
  ∃ (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℝ),
    -- The three rectangles have equal areas
    rect1_width * rect1_height = rect2_width * rect2_height ∧
    rect2_width * rect2_height = rect3_width * rect3_height ∧
    -- The rectangles fit within the square
    rect1_width + rect2_width ≤ square_side ∧
    rect1_height ≤ square_side ∧
    rect2_height ≤ square_side ∧
    rect3_width ≤ square_side ∧
    rect3_height ≤ square_side ∧
    -- The rectangles have common boundaries
    (rect1_width = rect2_width ∨ rect1_height = rect2_height) ∧
    (rect2_width = rect3_width ∨ rect2_height = rect3_height) ∧
    (rect1_width = rect3_width ∨ rect1_height = rect3_height) →
  cut_length = 60 :=
by sorry

end NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l295_29548


namespace NUMINAMATH_CALUDE_ellipse_k_range_l295_29597

/-- Represents an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  (equation : ∀ x y : ℝ, x^2 + k*y^2 = 2)
  (is_ellipse : k ≠ 0)
  (foci_on_y : k < 1)

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l295_29597


namespace NUMINAMATH_CALUDE_line_bisects_circle_l295_29578

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_bisects_circle :
  line_eq (circle_center.1) (circle_center.2) ∧
  ∃ (r : ℝ), ∀ (x y : ℝ), circle_eq x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l295_29578


namespace NUMINAMATH_CALUDE_petya_win_probability_l295_29522

/-- The game "Pile of Stones" --/
structure PileOfStones where
  initialStones : Nat
  minTake : Nat
  maxTake : Nat

/-- The optimal strategy for the game --/
def optimalStrategy (game : PileOfStones) : Nat → Nat :=
  sorry

/-- The probability of winning when playing randomly --/
def randomWinProbability (game : PileOfStones) : ℚ :=
  sorry

/-- The theorem stating the probability of Petya winning --/
theorem petya_win_probability :
  let game : PileOfStones := {
    initialStones := 16,
    minTake := 1,
    maxTake := 4
  }
  randomWinProbability game = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_petya_win_probability_l295_29522


namespace NUMINAMATH_CALUDE_proposition_problem_l295_29575

theorem proposition_problem (a : ℝ) :
  ((∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + a < 0) ∨
   (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + a*x + 1 = 0 ∧ y^2 + a*y + 1 = 0)) →
  (a > 2 ∨ a < 1) :=
sorry

end NUMINAMATH_CALUDE_proposition_problem_l295_29575


namespace NUMINAMATH_CALUDE_laura_shopping_cost_l295_29556

/-- Calculates the total cost of Laura's shopping trip given the prices and quantities of items. -/
def shopping_cost (salad_price : ℚ) (juice_price : ℚ) : ℚ :=
  let beef_price := 2 * salad_price
  let potato_price := salad_price / 3
  let mixed_veg_price := beef_price / 2 + 0.5
  let tomato_sauce_price := salad_price * 3 / 4
  let pasta_price := juice_price + mixed_veg_price
  2 * salad_price +
  2 * beef_price +
  1 * potato_price +
  2 * juice_price +
  3 * mixed_veg_price +
  5 * tomato_sauce_price +
  4 * pasta_price

theorem laura_shopping_cost :
  shopping_cost 3 1.5 = 63.75 := by
  sorry

end NUMINAMATH_CALUDE_laura_shopping_cost_l295_29556


namespace NUMINAMATH_CALUDE_equation_solutions_l295_29573

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 14*x - 10) + 1 / (x^2 + 3*x - 10) + 1 / (x^2 - 16*x - 10)
  {x : ℝ | f x = 0} = {5, -2, 2, -5} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l295_29573


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_l295_29599

theorem hyperbola_parabola_focus (a : ℝ) (h1 : a > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/3 = 1) ∧ 
  (∃ (x : ℝ), (2, 0) = (x, 0) ∧ x^2/a^2 - 0^2/3 = 1) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_l295_29599


namespace NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l295_29596

/-- A right triangle is a triangle with one right angle -/
structure RightTriangle where
  /-- The measure of the right angle in degrees -/
  right_angle : ℝ
  /-- The right angle measures 90 degrees -/
  is_right : right_angle = 90

/-- The number of right angles in a right triangle -/
def num_right_angles (t : RightTriangle) : ℕ := 1

theorem right_triangle_has_one_right_angle (t : RightTriangle) : 
  num_right_angles t = 1 := by
  sorry

#check right_triangle_has_one_right_angle

end NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l295_29596


namespace NUMINAMATH_CALUDE_theresa_final_week_hours_l295_29502

/-- The number of weeks Theresa needs to work -/
def total_weeks : ℕ := 6

/-- The required average number of hours per week -/
def required_average : ℚ := 10

/-- The list of hours worked in the first 5 weeks -/
def hours_worked : List ℚ := [8, 11, 7, 12, 10]

/-- The sum of hours worked in the first 5 weeks -/
def sum_first_five : ℚ := hours_worked.sum

/-- The number of hours Theresa needs to work in the final week -/
def hours_final_week : ℚ := required_average * total_weeks - sum_first_five

theorem theresa_final_week_hours :
  hours_final_week = 12 := by sorry

end NUMINAMATH_CALUDE_theresa_final_week_hours_l295_29502


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l295_29566

theorem continued_fraction_sum (v w x y z : ℕ+) : 
  (v : ℚ) + 1 / ((w : ℚ) + 1 / ((x : ℚ) + 1 / ((y : ℚ) + 1 / (z : ℚ)))) = 222 / 155 →
  10^4 * v.val + 10^3 * w.val + 10^2 * x.val + 10 * y.val + z.val = 12354 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l295_29566


namespace NUMINAMATH_CALUDE_complex_equation_solution_l295_29527

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : z = -1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l295_29527


namespace NUMINAMATH_CALUDE_probability_purple_marble_l295_29564

theorem probability_purple_marble (blue_prob green_prob : ℝ) 
  (h1 : blue_prob = 0.3)
  (h2 : green_prob = 0.4)
  (h3 : ∃ purple_prob : ℝ, blue_prob + green_prob + purple_prob = 1) :
  ∃ purple_prob : ℝ, purple_prob = 0.3 ∧ blue_prob + green_prob + purple_prob = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_purple_marble_l295_29564
