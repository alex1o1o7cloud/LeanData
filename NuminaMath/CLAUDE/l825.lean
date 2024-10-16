import Mathlib

namespace NUMINAMATH_CALUDE_line_bisected_by_point_m_prove_line_bisected_by_point_m_l825_82592

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def Point.isMidpointOf (m : Point) (p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- The theorem to be proved -/
theorem line_bisected_by_point_m (l1 l2 : Line) (m : Point) : Prop :=
  let desired_line : Line := { a := 1, b := 4, c := -4 }
  let point_m : Point := { x := 0, y := 1 }
  l1 = { a := 1, b := -3, c := 10 } →
  l2 = { a := 2, b := 1, c := -8 } →
  m = point_m →
  ∃ (p1 p2 : Point),
    p1.liesOn l1 ∧
    p2.liesOn l2 ∧
    m.isMidpointOf p1 p2 ∧
    p1.liesOn desired_line ∧
    p2.liesOn desired_line ∧
    m.liesOn desired_line

/-- Proof of the theorem -/
theorem prove_line_bisected_by_point_m (l1 l2 : Line) (m : Point) :
  line_bisected_by_point_m l1 l2 m := by
  sorry

end NUMINAMATH_CALUDE_line_bisected_by_point_m_prove_line_bisected_by_point_m_l825_82592


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l825_82510

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 2 * Real.sqrt 3 * x + 1 = 0 → (x - 1/x = 2 * Real.sqrt 2 ∨ x - 1/x = -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l825_82510


namespace NUMINAMATH_CALUDE_function_properties_l825_82532

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_not_constant : ∃ x y, f x ≠ f y)
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_symmetry : ∀ x, f (x + 1) = f (1 - x)) :
  is_even_function f ∧ is_periodic_function f 2 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l825_82532


namespace NUMINAMATH_CALUDE_smallest_a_value_l825_82583

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a ≥ 15 ∧ ∀ a' ≥ 0, (∀ x : ℝ, Real.sin (a' * x + b) = Real.sin (15 * x)) → a' ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l825_82583


namespace NUMINAMATH_CALUDE_max_intersection_area_point_l825_82512

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Calculates the ratio of two line segments -/
def ratio (P Q R : Point) : ℝ :=
  sorry

/-- Calculates the area of a triangle -/
def triangleArea (P Q R : Point) : ℝ :=
  sorry

/-- Calculates the area of the intersection of two triangles -/
def intersectionArea (P Q R S T U : Point) : ℝ :=
  sorry

/-- Theorem: The point M on BC that maximizes the intersection area satisfies BM/MC = AK/KD -/
theorem max_intersection_area_point (ABCD : Trapezoid) (K : Point) :
  ∃ (M : Point),
    (∀ (M' : Point), intersectionArea ABCD.A ABCD.B ABCD.C ABCD.D K M ≥ 
                      intersectionArea ABCD.A ABCD.B ABCD.C ABCD.D K M') ↔
    (ratio ABCD.B M ABCD.C = ratio ABCD.A K ABCD.D) :=
  sorry

end NUMINAMATH_CALUDE_max_intersection_area_point_l825_82512


namespace NUMINAMATH_CALUDE_unique_prime_in_sequence_l825_82567

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (A : ℕ) : ℕ := 205100 + A

theorem unique_prime_in_sequence :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) ∧ number A = 205103 := by sorry

end NUMINAMATH_CALUDE_unique_prime_in_sequence_l825_82567


namespace NUMINAMATH_CALUDE_lower_price_calculation_l825_82575

/-- The lower selling price of an article -/
def lower_price : ℚ := 348

/-- The higher selling price of an article -/
def higher_price : ℚ := 350

/-- The cost price of the article -/
def cost_price : ℚ := 40

/-- The percentage difference in profit between the two selling prices -/
def profit_difference_percentage : ℚ := 5 / 100

theorem lower_price_calculation :
  (higher_price - cost_price) = (lower_price - cost_price) + profit_difference_percentage * cost_price :=
by sorry

end NUMINAMATH_CALUDE_lower_price_calculation_l825_82575


namespace NUMINAMATH_CALUDE_orange_profit_l825_82521

theorem orange_profit : 
  let buy_rate : ℚ := 10 / 11  -- Cost in r per orange when buying
  let sell_rate : ℚ := 11 / 10  -- Revenue in r per orange when selling
  let num_oranges : ℕ := 110
  let cost : ℚ := buy_rate * num_oranges
  let revenue : ℚ := sell_rate * num_oranges
  let profit : ℚ := revenue - cost
  profit = 21 := by sorry

end NUMINAMATH_CALUDE_orange_profit_l825_82521


namespace NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_2_l825_82505

theorem gcd_13m_plus_4_7m_plus_2_max_2 :
  (∀ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) ≤ 2) ∧
  (∃ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_2_l825_82505


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l825_82536

def fibonacci_like_sequence (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n + 1)

theorem first_term_of_sequence (a b : ℕ) :
  fibonacci_like_sequence a b 5 = 21 ∧
  fibonacci_like_sequence a b 6 = 34 ∧
  fibonacci_like_sequence a b 7 = 55 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l825_82536


namespace NUMINAMATH_CALUDE_handshake_count_l825_82585

theorem handshake_count (n : ℕ) (h : n = 10) : n * (n - 1) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l825_82585


namespace NUMINAMATH_CALUDE_polygon_sides_l825_82533

theorem polygon_sides (sum_interior_angles sum_exterior_angles : ℕ) : 
  sum_interior_angles - sum_exterior_angles = 720 →
  sum_exterior_angles = 360 →
  (∃ n : ℕ, sum_interior_angles = (n - 2) * 180 ∧ n = 8) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l825_82533


namespace NUMINAMATH_CALUDE_pats_pool_ratio_l825_82566

theorem pats_pool_ratio : 
  let total_pools : ℕ := 800
  let ark_pools : ℕ := 200
  let supply_pools : ℕ := total_pools - ark_pools
  supply_pools / ark_pools = 3 := by
  sorry

end NUMINAMATH_CALUDE_pats_pool_ratio_l825_82566


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l825_82564

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3/4) :
  (4/x + 1/y) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l825_82564


namespace NUMINAMATH_CALUDE_all_lines_pass_through_point_l825_82529

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Defines a geometric progression for three real numbers -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

/-- The theorem stating that all lines with a, b, c in geometric progression pass through (0, 1) -/
theorem all_lines_pass_through_point :
  ∀ l : Line, isGeometricProgression l.a l.b l.c → l.contains 0 1 :=
sorry

end NUMINAMATH_CALUDE_all_lines_pass_through_point_l825_82529


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l825_82598

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 5 * width →
  width * length = 720 →
  length - width = 48 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l825_82598


namespace NUMINAMATH_CALUDE_folded_square_perimeter_l825_82546

/-- A square with side length 2 is folded so that vertex A meets edge BC at A',
    and edge AB intersects edge CD at F. Given BA' = 1/2,
    prove that the perimeter of triangle CFA' is (3 + √17) / 2. -/
theorem folded_square_perimeter (A B C D A' F : ℝ × ℝ) : 
  let square_side : ℝ := 2
  let BA'_length : ℝ := 1/2
  -- Define the square
  (A = (0, square_side) ∧ B = (0, 0) ∧ C = (square_side, 0) ∧ D = (square_side, square_side)) →
  -- A' is on BC
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A' = (t * square_side, 0)) →
  -- BA' length is 1/2
  (Real.sqrt ((A'.1 - B.1)^2 + (A'.2 - B.2)^2) = BA'_length) →
  -- F is on CD
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (square_side, s * square_side)) →
  -- F is also on AB
  (∃ r : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ F = ((1-r) * A.1 + r * B.1, (1-r) * A.2 + r * B.2)) →
  -- Conclusion: Perimeter of CFA' is (3 + √17) / 2
  let CF := Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2)
  let FA' := Real.sqrt ((F.1 - A'.1)^2 + (F.2 - A'.2)^2)
  let CA' := Real.sqrt ((C.1 - A'.1)^2 + (C.2 - A'.2)^2)
  CF + FA' + CA' = (3 + Real.sqrt 17) / 2 := by
sorry

end NUMINAMATH_CALUDE_folded_square_perimeter_l825_82546


namespace NUMINAMATH_CALUDE_garden_area_l825_82528

/-- The area of a garden surrounding a circular ground -/
theorem garden_area (d : ℝ) (w : ℝ) (h1 : d = 34) (h2 : w = 2) :
  let r := d / 2
  let R := r + w
  π * (R^2 - r^2) = π * 72 := by sorry

end NUMINAMATH_CALUDE_garden_area_l825_82528


namespace NUMINAMATH_CALUDE_geometric_progression_middle_term_l825_82574

theorem geometric_progression_middle_term 
  (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_geometric : b^2 = a * c) 
  (h_a : a = 5 + 2 * Real.sqrt 6) 
  (h_c : c = 5 - 2 * Real.sqrt 6) : 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_middle_term_l825_82574


namespace NUMINAMATH_CALUDE_digits_s_200_l825_82579

/-- s(n) is the number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- count_digits n is the number of digits in the decimal representation of n -/
def count_digits (n : ℕ) : ℕ := sorry

/-- The number of digits in s(200) is 492 -/
theorem digits_s_200 : count_digits (s 200) = 492 := by sorry

end NUMINAMATH_CALUDE_digits_s_200_l825_82579


namespace NUMINAMATH_CALUDE_special_sequence_sum_l825_82571

/-- A sequence where the sum of two terms with a term between them increases by a constant amount -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2) + a (n + 3) = a n + a (n + 1) + d

theorem special_sequence_sum (a : ℕ → ℝ) (h : SpecialSequence a)
    (h1 : a 2 + a 3 = 4) (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_l825_82571


namespace NUMINAMATH_CALUDE_winning_probability_l825_82563

theorem winning_probability (total_products winning_products : ℕ) 
  (h1 : total_products = 6)
  (h2 : winning_products = 2) :
  (winning_products : ℚ) / total_products = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_winning_probability_l825_82563


namespace NUMINAMATH_CALUDE_committee_election_count_l825_82559

def group_size : ℕ := 15
def women_count : ℕ := 5
def committee_size : ℕ := 4
def min_women : ℕ := 2

def elect_committee : ℕ := sorry

theorem committee_election_count : 
  elect_committee = 555 := by sorry

end NUMINAMATH_CALUDE_committee_election_count_l825_82559


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l825_82520

-- Define the cube structure
structure Cube where
  faces : Fin 6 → Fin 4

-- Define the probability of a continuous stripe
def probability_continuous_stripe (c : Cube) : ℚ :=
  3 * (1 / 4) ^ 12

-- Theorem statement
theorem continuous_stripe_probability :
  ∀ c : Cube, probability_continuous_stripe c = 3 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l825_82520


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l825_82503

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l825_82503


namespace NUMINAMATH_CALUDE_hyperbola_t_squared_l825_82506

/-- A hyperbola is defined by its center, orientation, and three points it passes through. -/
structure Hyperbola where
  center : ℝ × ℝ
  horizontalOpening : Bool
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Given a hyperbola with specific properties, calculate t². -/
def calculateTSquared (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating that for a hyperbola with given properties, t² = 45/4. -/
theorem hyperbola_t_squared 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_opening : h.horizontalOpening = true)
  (h_point1 : h.point1 = (-3, 4))
  (h_point2 : h.point2 = (-3, 0))
  (h_point3 : ∃ t : ℝ, h.point3 = (t, 3)) :
  calculateTSquared h = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_t_squared_l825_82506


namespace NUMINAMATH_CALUDE_probability_three_digit_ending_4_divisible_by_3_l825_82599

/-- A three-digit positive integer ending in 4 -/
def ThreeDigitEndingIn4 : Type := { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 4 }

/-- The count of three-digit positive integers ending in 4 -/
def totalCount : ℕ := 90

/-- The count of three-digit positive integers ending in 4 that are divisible by 3 -/
def divisibleBy3Count : ℕ := 33

/-- The probability that a three-digit positive integer ending in 4 is divisible by 3 -/
def probabilityDivisibleBy3 : ℚ := divisibleBy3Count / totalCount

theorem probability_three_digit_ending_4_divisible_by_3 :
  probabilityDivisibleBy3 = 11 / 30 := by sorry

end NUMINAMATH_CALUDE_probability_three_digit_ending_4_divisible_by_3_l825_82599


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l825_82584

-- Define the sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- State the theorem
theorem union_of_M_and_N :
  M ∪ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l825_82584


namespace NUMINAMATH_CALUDE_deepak_age_l825_82517

theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) (years_ahead : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 2 →
  rahul_future_age = 26 →
  years_ahead = 10 →
  ∃ (x : ℕ), rahul_ratio * x + years_ahead = rahul_future_age ∧ deepak_ratio * x = 8 :=
by sorry

end NUMINAMATH_CALUDE_deepak_age_l825_82517


namespace NUMINAMATH_CALUDE_set_operations_l825_82502

open Set

-- Define the universe set U
def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ A) ∩ (U \ B) = {3}) ∧
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 5, 6, 7, 8, 9, 10}) ∧
  ((A ∩ B) ∩ C = ∅) ∧
  ((A ∪ B) ∩ C = {5, 7}) := by
sorry


end NUMINAMATH_CALUDE_set_operations_l825_82502


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l825_82550

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c² = (a-b)² + 6 and the area of the triangle is 3√3/2,
    then the measure of angle C is π/3 -/
theorem angle_measure_in_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c^2 = (a - b)^2 + 6 →
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l825_82550


namespace NUMINAMATH_CALUDE_board_number_problem_l825_82513

theorem board_number_problem (x : ℕ) : 
  x > 0 ∧ x < 2022 ∧ 
  (∀ n : ℕ, n ≤ 10 → (2022 + x) % (2^n) = 0) →
  x = 998 := by
sorry

end NUMINAMATH_CALUDE_board_number_problem_l825_82513


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l825_82578

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x - 2) = 9 / x) ∧ (x ≠ 2) ∧ (x ≠ 0)

def equation2 (x : ℝ) : Prop := (x / (x + 1) = 2 * x / (3 * x + 3) - 1) ∧ (x ≠ -1) ∧ (3 * x + 3 ≠ 0)

-- State the theorems
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 3 := by sorry

theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -3/4 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l825_82578


namespace NUMINAMATH_CALUDE_mixture_replacement_l825_82507

/-- Given a mixture of liquids A and B, this theorem proves that
    replacing a certain amount of the mixture with liquid B
    results in the specified final ratio. -/
theorem mixture_replacement (initial_a initial_b replacement : ℚ) :
  initial_a = 16 →
  initial_b = 4 →
  (initial_a - 4/5 * replacement) / (initial_b + 4/5 * replacement) = 2/3 →
  replacement = 10 := by
  sorry

end NUMINAMATH_CALUDE_mixture_replacement_l825_82507


namespace NUMINAMATH_CALUDE_reciprocal_sum_l825_82538

theorem reciprocal_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l825_82538


namespace NUMINAMATH_CALUDE_min_wrapping_paper_dimensions_l825_82556

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular wrapping paper -/
structure WrappingPaperDimensions where
  width : ℝ
  length : ℝ

/-- Checks if the wrapping paper can cover the box completely -/
def canCoverBox (box : BoxDimensions) (paper : WrappingPaperDimensions) : Prop :=
  paper.width ≥ box.width + 2 * box.height ∧
  paper.length ≥ box.length + 2 * box.height

/-- The main theorem stating the minimum dimensions of wrapping paper required -/
theorem min_wrapping_paper_dimensions (w : ℝ) (hw : w > 0) :
  ∀ paper : WrappingPaperDimensions,
    let box : BoxDimensions := ⟨w, 2*w, w⟩
    canCoverBox box paper →
    paper.width ≥ 3*w ∧ paper.length ≥ 4*w :=
  sorry

end NUMINAMATH_CALUDE_min_wrapping_paper_dimensions_l825_82556


namespace NUMINAMATH_CALUDE_base_difference_calculation_l825_82504

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- The main theorem stating the result of the calculation -/
theorem base_difference_calculation :
  base6ToBase10 52143 - base7ToBase10 4310 = 5449 := by sorry

end NUMINAMATH_CALUDE_base_difference_calculation_l825_82504


namespace NUMINAMATH_CALUDE_book_sale_revenue_l825_82541

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  total_books > 0 ∧ 
  price_per_book > 0 ∧
  (total_books : ℚ) / 3 = 30 ∧ 
  price_per_book = 17/4 → 
  (2 : ℚ) / 3 * (total_books : ℚ) * price_per_book = 255 := by
sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l825_82541


namespace NUMINAMATH_CALUDE_min_pie_pieces_correct_l825_82549

/-- The minimum number of pieces a pie can be cut into to be equally divided among either 10 or 11 guests -/
def min_pie_pieces : ℕ := 20

/-- The number of expected guests -/
def possible_guests : Set ℕ := {10, 11}

/-- A function that checks if a given number of pieces can be equally divided among a given number of guests -/
def is_divisible (pieces : ℕ) (guests : ℕ) : Prop :=
  ∃ (k : ℕ), pieces = k * guests

theorem min_pie_pieces_correct :
  (∀ g ∈ possible_guests, is_divisible min_pie_pieces g) ∧
  (∀ p < min_pie_pieces, ∃ g ∈ possible_guests, ¬is_divisible p g) :=
sorry

end NUMINAMATH_CALUDE_min_pie_pieces_correct_l825_82549


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l825_82553

/-- Two points symmetric about the y-axis in a Cartesian coordinate system -/
structure SymmetricPoints where
  m : ℝ
  n : ℝ
  symmetric : m + 4 = 0 ∧ n = 3

/-- The theorem stating that for symmetric points A(m,3) and B(4,n), (m+n)^2023 = -1 -/
theorem symmetric_points_sum_power (p : SymmetricPoints) : (p.m + p.n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l825_82553


namespace NUMINAMATH_CALUDE_A_intersect_B_l825_82576

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def B : Set ℝ := {x | ∃ k : ℤ, x = 2*k}

theorem A_intersect_B : A ∩ B = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l825_82576


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l825_82508

/-- Given two vectors a and b in ℝ³, where a = (2, -3, 1) and b = (4, -6, x),
    if a is perpendicular to b, then x = -26. -/
theorem perpendicular_vectors_x_value :
  let a : Fin 3 → ℝ := ![2, -3, 1]
  let b : Fin 3 → ℝ := ![4, -6, x]
  (∀ i : Fin 3, a i * b i = 0) → x = -26 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l825_82508


namespace NUMINAMATH_CALUDE_zero_in_interval_l825_82557

/-- Given two positive real numbers a and b where a > b > 0 and |log a| = |log b|,
    there exists an x in the interval (-1, 0) such that a^x + x - b = 0 -/
theorem zero_in_interval (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : |Real.log a| = |Real.log b|) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ a^x + x - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l825_82557


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l825_82514

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 * k + 1 ∧ p.2^2 = 4 * p.1) → 
  k = -1 ∨ k = 0 ∨ k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l825_82514


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l825_82516

theorem sufficiency_not_necessity (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 → x₁ + x₂ > 2 ∧ x₁ * x₂ > 1) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ + x₂ > 2 ∧ x₁ * x₂ > 1 ∧ ¬(x₁ > 1 ∧ x₂ > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l825_82516


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l825_82524

theorem fraction_equals_zero (x : ℝ) : 
  (x^2 - 1) / (1 - x) = 0 ∧ 1 - x ≠ 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l825_82524


namespace NUMINAMATH_CALUDE_task_completion_probability_l825_82547

theorem task_completion_probability (p_task1 p_task1_not_task2 : ℝ) 
  (h1 : p_task1 = 5/8)
  (h2 : p_task1_not_task2 = 1/4)
  (h3 : 0 ≤ p_task1 ∧ p_task1 ≤ 1)
  (h4 : 0 ≤ p_task1_not_task2 ∧ p_task1_not_task2 ≤ 1) :
  ∃ p_task2 : ℝ, p_task2 = 3/5 ∧ p_task1 * (1 - p_task2) = p_task1_not_task2 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l825_82547


namespace NUMINAMATH_CALUDE_custom_op_equation_solution_l825_82539

-- Define the custom operation *
def custom_op (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem custom_op_equation_solution :
  ∃ x : ℚ, custom_op 3 (custom_op 6 x) = -2 ∧ x = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_equation_solution_l825_82539


namespace NUMINAMATH_CALUDE_probability_three_blue_marbles_specific_l825_82562

/-- Represents the probability of drawing 3 blue marbles from a jar --/
def probability_three_blue_marbles (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  (blue / total) * ((blue - 1) / (total - 1)) * ((blue - 2) / (total - 2))

/-- Theorem stating the probability of drawing 3 blue marbles from a specific jar configuration --/
theorem probability_three_blue_marbles_specific :
  probability_three_blue_marbles 3 4 13 = 1 / 285 := by
  sorry

#eval probability_three_blue_marbles 3 4 13

end NUMINAMATH_CALUDE_probability_three_blue_marbles_specific_l825_82562


namespace NUMINAMATH_CALUDE_parabola_intersection_properties_l825_82545

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2
def g (x : ℝ) : ℝ := 2 * x - 3
def h (x : ℝ) : ℝ := 2

-- Define the theorem
theorem parabola_intersection_properties
  (a : ℝ)
  (ha : a ≠ 0)
  (h_intersection : f a 1 = g 1) :
  (a = -1) ∧
  (∀ x, f a x = -x^2) ∧
  (∀ x, x < 0 → (∀ y, y < x → f a y < f a x)) ∧
  (let x₁ := Real.sqrt 2
   let x₂ := -Real.sqrt 2
   let area := (1/2) * (x₁ - x₂) * (h x₁ - f a 0)
   area = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_properties_l825_82545


namespace NUMINAMATH_CALUDE_valid_rectangles_count_l825_82522

/-- Represents a square array of dots -/
structure DotArray where
  size : ℕ

/-- Represents a rectangle in the dot array -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Returns true if the rectangle has an area greater than 1 -/
def Rectangle.areaGreaterThanOne (r : Rectangle) : Prop :=
  r.width * r.height > 1

/-- Returns the number of valid rectangles in the dot array -/
def countValidRectangles (arr : DotArray) : ℕ :=
  sorry

theorem valid_rectangles_count (arr : DotArray) :
  arr.size = 5 → countValidRectangles arr = 84 := by
  sorry

end NUMINAMATH_CALUDE_valid_rectangles_count_l825_82522


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l825_82569

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h_side : a * b = 20) 
  (h_front : b * c = 12) 
  (h_bottom : a * c = 15) : 
  a * b * c = 60 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l825_82569


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l825_82560

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -12 * x^2 + 3 * x - 5 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l825_82560


namespace NUMINAMATH_CALUDE_principal_amount_is_875_l825_82519

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (interest_rate : ℚ) (time : ℕ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (interest_rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 875. -/
theorem principal_amount_is_875 :
  let interest_rate : ℚ := 12
  let time : ℕ := 20
  let total_interest : ℚ := 2100
  calculate_principal interest_rate time total_interest = 875 := by sorry

end NUMINAMATH_CALUDE_principal_amount_is_875_l825_82519


namespace NUMINAMATH_CALUDE_fathers_age_is_38_l825_82580

/-- The age of the son 5 years ago -/
def sons_age_5_years_ago : ℕ := 14

/-- The current age of the son -/
def sons_current_age : ℕ := sons_age_5_years_ago + 5

/-- The age of the father when the son was born -/
def fathers_age_at_sons_birth : ℕ := sons_current_age

/-- The current age of the father -/
def fathers_current_age : ℕ := fathers_age_at_sons_birth + sons_current_age

theorem fathers_age_is_38 : fathers_current_age = 38 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_is_38_l825_82580


namespace NUMINAMATH_CALUDE_product_H₁_H₂_is_square_l825_82525

/-- For any positive integer n, H₁ is the set of odd numbers from 1 to 2n-1 -/
def H₁ (n : ℕ+) : Finset ℕ :=
  Finset.range n |>.image (fun i => 2*i + 1)

/-- For any positive integers n and k, H₂ is the set obtained by adding k to each element of H₁ -/
def H₂ (n : ℕ+) (k : ℕ+) : Finset ℕ :=
  H₁ n |>.image (fun x => x + k)

/-- The product of all elements in the union of H₁ and H₂ -/
def product_H₁_H₂ (n : ℕ+) (k : ℕ+) : ℕ :=
  (H₁ n ∪ H₂ n k).prod id

/-- For any positive integer n, when k = 2n + 1, the product of all elements in H₁ ∪ H₂ is a perfect square -/
theorem product_H₁_H₂_is_square (n : ℕ+) :
  ∃ m : ℕ, product_H₁_H₂ n (2*n + 1) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_product_H₁_H₂_is_square_l825_82525


namespace NUMINAMATH_CALUDE_derivative_of_y_l825_82594

noncomputable def y (x : ℝ) : ℝ := 
  -1 / (3 * Real.sin x ^ 3) - 1 / Real.sin x + 1 / 2 * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

theorem derivative_of_y (x : ℝ) (h : x ∉ Set.range (fun n => n * π)) :
  deriv y x = 1 / (Real.cos x * Real.sin x ^ 4) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_y_l825_82594


namespace NUMINAMATH_CALUDE_total_rectangles_3x3_grid_l825_82590

/-- Represents a grid of points -/
structure Grid where
  rows : Nat
  cols : Nat

/-- Represents a rectangle on the grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Counts the number of rectangles of a given size on the grid -/
def countRectangles (g : Grid) (r : Rectangle) : Nat :=
  sorry

/-- The total number of rectangles on a 3x3 grid -/
def totalRectangles : Nat :=
  let g : Grid := { rows := 3, cols := 3 }
  (countRectangles g { width := 1, height := 1 }) +
  (countRectangles g { width := 1, height := 2 }) +
  (countRectangles g { width := 1, height := 3 }) +
  (countRectangles g { width := 2, height := 1 }) +
  (countRectangles g { width := 2, height := 2 }) +
  (countRectangles g { width := 2, height := 3 }) +
  (countRectangles g { width := 3, height := 1 }) +
  (countRectangles g { width := 3, height := 2 })

/-- Theorem stating that the total number of rectangles on a 3x3 grid is 124 -/
theorem total_rectangles_3x3_grid :
  totalRectangles = 124 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_3x3_grid_l825_82590


namespace NUMINAMATH_CALUDE_f_properties_l825_82595

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 + 4*x + 3 else Real.log (x - 1) + 1

-- Theorem statement
theorem f_properties :
  (f (Real.exp 1 + 1) = 2) ∧
  (Set.range f = Set.Ici (-1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l825_82595


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l825_82530

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l825_82530


namespace NUMINAMATH_CALUDE_fraction_split_l825_82500

theorem fraction_split (n d a b : ℕ) (h1 : d = a * b) (h2 : Nat.gcd a b = 1) (h3 : n = 58) (h4 : d = 77) (h5 : a = 11) (h6 : b = 7) :
  ∃ (x y : ℤ), (n : ℚ) / d = (x : ℚ) / b + (y : ℚ) / a :=
sorry

end NUMINAMATH_CALUDE_fraction_split_l825_82500


namespace NUMINAMATH_CALUDE_M_equals_N_l825_82589

def M : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k - 1) * Real.pi}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l825_82589


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l825_82597

theorem decimal_sum_to_fraction :
  (0.3 : ℚ) + 0.04 + 0.005 + 0.0006 + 0.00007 = 34567 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l825_82597


namespace NUMINAMATH_CALUDE_max_value_inequality_l825_82548

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 + 2*c^2 = 1) : 
  Real.sqrt 2 * a * b + 2 * b * c + 7 * a * c ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l825_82548


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l825_82543

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l825_82543


namespace NUMINAMATH_CALUDE_interior_angles_ratio_l825_82518

-- Define a triangle type
structure Triangle where
  -- Define exterior angles
  ext_angle1 : ℝ
  ext_angle2 : ℝ
  ext_angle3 : ℝ
  -- Condition: exterior angles sum to 360°
  sum_ext_angles : ext_angle1 + ext_angle2 + ext_angle3 = 360
  -- Condition: ratio of exterior angles is 3:4:5
  ratio_ext_angles : ∃ (x : ℝ), ext_angle1 = 3*x ∧ ext_angle2 = 4*x ∧ ext_angle3 = 5*x

-- Define interior angles
def interior_angle1 (t : Triangle) : ℝ := 180 - t.ext_angle1
def interior_angle2 (t : Triangle) : ℝ := 180 - t.ext_angle2
def interior_angle3 (t : Triangle) : ℝ := 180 - t.ext_angle3

-- Theorem statement
theorem interior_angles_ratio (t : Triangle) :
  ∃ (k : ℝ), interior_angle1 t = 3*k ∧ interior_angle2 t = 2*k ∧ interior_angle3 t = k := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_ratio_l825_82518


namespace NUMINAMATH_CALUDE_audit_options_l825_82509

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem audit_options (initial_OR initial_GTU first_week_OR first_week_GTU : ℕ) 
  (h1 : initial_OR = 13)
  (h2 : initial_GTU = 15)
  (h3 : first_week_OR = 2)
  (h4 : first_week_GTU = 3) :
  (choose (initial_OR - first_week_OR) first_week_OR) * 
  (choose (initial_GTU - first_week_GTU) first_week_GTU) = 12100 := by
  sorry

end NUMINAMATH_CALUDE_audit_options_l825_82509


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l825_82568

/-- A quadratic equation with reciprocal roots whose sum is four times their product -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_reciprocal : ∃ (r : ℝ), r ≠ 0 ∧ r + 1/r = -b/a
  sum_four_times_product : -b/a = 4 * (c/a)

/-- The coefficients of the quadratic equation satisfy a = c and b = -4a -/
theorem quadratic_equation_coefficients (eq : QuadraticEquation) : eq.a = eq.c ∧ eq.b = -4 * eq.a := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l825_82568


namespace NUMINAMATH_CALUDE_factorization_mx_minus_my_l825_82515

theorem factorization_mx_minus_my (m x y : ℝ) : m * x - m * y = m * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_mx_minus_my_l825_82515


namespace NUMINAMATH_CALUDE_solve_for_z_l825_82542

theorem solve_for_z (x z : ℝ) 
  (h1 : x = 102) 
  (h2 : x^4*z - 3*x^3*z + 2*x^2*z = 1075648000) : 
  z = 1.024 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l825_82542


namespace NUMINAMATH_CALUDE_joey_caught_one_kg_more_than_peter_l825_82535

/-- Given three fishers Ali, Peter, and Joey, prove that Joey caught 1 kg more fish than Peter -/
theorem joey_caught_one_kg_more_than_peter 
  (total_catch : ℝ)
  (ali_catch : ℝ)
  (peter_catch : ℝ)
  (joey_catch : ℝ)
  (h1 : total_catch = 25)
  (h2 : ali_catch = 12)
  (h3 : ali_catch = 2 * peter_catch)
  (h4 : joey_catch = peter_catch + (joey_catch - peter_catch))
  (h5 : total_catch = ali_catch + peter_catch + joey_catch) :
  joey_catch - peter_catch = 1 := by
  sorry

end NUMINAMATH_CALUDE_joey_caught_one_kg_more_than_peter_l825_82535


namespace NUMINAMATH_CALUDE_combined_output_in_five_minutes_l825_82586

/-- The rate at which Machine A fills boxes (boxes per minute) -/
def machine_a_rate : ℚ := 24 / 60

/-- The rate at which Machine B fills boxes (boxes per minute) -/
def machine_b_rate : ℚ := 36 / 60

/-- The combined rate of both machines (boxes per minute) -/
def combined_rate : ℚ := machine_a_rate + machine_b_rate

/-- The time period we're interested in (minutes) -/
def time_period : ℚ := 5

theorem combined_output_in_five_minutes :
  combined_rate * time_period = 5 := by sorry

end NUMINAMATH_CALUDE_combined_output_in_five_minutes_l825_82586


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l825_82544

/-- Given a store's coloring book inventory and sales, calculate the number of shelves needed to display the remaining books. -/
theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) :
  initial_stock = 86 →
  books_sold = 37 →
  books_per_shelf = 7 →
  (initial_stock - books_sold) / books_per_shelf = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l825_82544


namespace NUMINAMATH_CALUDE_smallest_number_in_arithmetic_sequence_l825_82534

theorem smallest_number_in_arithmetic_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 29 →
  b = 30 →
  c = b + 5 →
  a < b ∧ b < c →
  a = 22 := by sorry

end NUMINAMATH_CALUDE_smallest_number_in_arithmetic_sequence_l825_82534


namespace NUMINAMATH_CALUDE_cantor_is_founder_l825_82523

/-- Represents a mathematician -/
inductive Mathematician
  | Gauss
  | Dedekind
  | Weierstrass
  | Cantor

/-- Represents the founder of modern set theory -/
def founder_of_modern_set_theory : Mathematician := Mathematician.Cantor

/-- Theorem stating that Cantor is the founder of modern set theory -/
theorem cantor_is_founder : 
  founder_of_modern_set_theory = Mathematician.Cantor := by sorry

end NUMINAMATH_CALUDE_cantor_is_founder_l825_82523


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l825_82570

theorem infinite_solutions_condition (c : ℝ) : 
  (∀ x, 5 * (3 * x - c) = 3 * (5 * x + 20)) ↔ c = -12 := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l825_82570


namespace NUMINAMATH_CALUDE_quadratic_inequality_l825_82526

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c < 0) : 
  b / a < c / a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l825_82526


namespace NUMINAMATH_CALUDE_zebras_total_games_l825_82537

theorem zebras_total_games : 
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →  -- 40% win rate initially
    ∀ (final_games : ℕ) (final_wins : ℕ),
      final_games = initial_games + 11 →  -- 8 won + 3 lost = 11 more games
      final_wins = initial_wins + 8 →     -- 8 more wins
      final_wins = (11 * final_games / 20) →  -- 55% win rate finally
      final_games = 24 := by
sorry

end NUMINAMATH_CALUDE_zebras_total_games_l825_82537


namespace NUMINAMATH_CALUDE_immersed_cone_specific_gravity_l825_82577

/-- Represents an equilateral cone immersed in water -/
structure ImmersedCone where
  -- Radius of the base of the cone
  baseRadius : ℝ
  -- Height of the cone
  height : ℝ
  -- Height of the cone above water
  heightAboveWater : ℝ
  -- Specific gravity of the cone material
  specificGravity : ℝ
  -- The cone is equilateral
  equilateral : height = baseRadius * Real.sqrt 3
  -- The area of the water surface circle is one-third of the base area
  waterSurfaceArea : π * (heightAboveWater / 3)^2 = π * baseRadius^2 / 3
  -- The angle between water surface and cone side is 120°
  waterSurfaceAngle : Real.cos (2 * π / 3) = heightAboveWater / (2 * baseRadius)

/-- Theorem stating the specific gravity of the cone -/
theorem immersed_cone_specific_gravity (c : ImmersedCone) :
  c.specificGravity = 1 - Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_immersed_cone_specific_gravity_l825_82577


namespace NUMINAMATH_CALUDE_legoland_kangaroos_l825_82581

theorem legoland_kangaroos (koalas kangaroos : ℕ) : 
  kangaroos = 5 * koalas →
  koalas + kangaroos = 216 →
  kangaroos = 180 := by
sorry

end NUMINAMATH_CALUDE_legoland_kangaroos_l825_82581


namespace NUMINAMATH_CALUDE_antonio_winning_strategy_l825_82501

/-- Represents the game state with two piles of chips -/
structure GameState where
  m : ℕ
  n : ℕ

/-- Defines the possible moves in the game -/
inductive Move
  | TakeOne : Bool → Move  -- True for first pile, False for second
  | TakeBoth : Move
  | Transfer : Bool → Move  -- True for first to second, False for second to first

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeOne first => 
      if first then ⟨state.m - 1, state.n⟩ else ⟨state.m, state.n - 1⟩
  | Move.TakeBoth => ⟨state.m - 1, state.n - 1⟩
  | Move.Transfer first => 
      if first then ⟨state.m - 1, state.n + 1⟩ else ⟨state.m + 1, state.n - 1⟩

/-- Determines if a move is valid for a given state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.TakeOne first => if first then state.m > 0 else state.n > 0
  | Move.TakeBoth => state.m > 0 ∧ state.n > 0
  | Move.Transfer first => if first then state.m > 0 else state.n > 0

/-- Determines if the game is over (no valid moves) -/
def isGameOver (state : GameState) : Bool :=
  state.m = 0 ∧ state.n = 0

/-- Theorem: The first player (Antonio) has a winning strategy if and only if at least one of m or n is odd -/
theorem antonio_winning_strategy (initialState : GameState) :
  (initialState.m % 2 = 1 ∨ initialState.n % 2 = 1) ↔ 
  ∃ (strategy : GameState → Move), 
    (∀ (state : GameState), 
      ¬isGameOver state → 
      isValidMove state (strategy state) ∧ 
      ¬∃ (counterStrategy : GameState → Move), 
        (∀ (state : GameState), 
          ¬isGameOver state → 
          isValidMove state (counterStrategy state) ∧ 
          isGameOver (applyMove (applyMove state (strategy state)) (counterStrategy (applyMove state (strategy state)))))) :=
sorry

end NUMINAMATH_CALUDE_antonio_winning_strategy_l825_82501


namespace NUMINAMATH_CALUDE_quadratic_roots_property_largest_a_value_l825_82591

theorem quadratic_roots_property (a : ℝ) : 
  let f (x : ℝ) := (1/3) * x^2 + (a + 1/2) * x + (a^2 + a)
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁^3 + x₂^3 = 3 * x₁ * x₂) →
  a ≤ -1/4 :=
by sorry

theorem largest_a_value :
  ∃ a : ℝ, a = -1/4 ∧
  let f (x : ℝ) := (1/3) * x^2 + (a + 1/2) * x + (a^2 + a)
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁^3 + x₂^3 = 3 * x₁ * x₂) ∧
  ∀ b > a, ¬(∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁^3 + x₂^3 = 3 * x₁ * x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_largest_a_value_l825_82591


namespace NUMINAMATH_CALUDE_pallet_weight_l825_82554

/-- Given a pallet with 3 boxes, where each box weighs 89 kilograms,
    the total weight of the pallet is 267 kilograms. -/
theorem pallet_weight (num_boxes : ℕ) (weight_per_box : ℕ) (total_weight : ℕ) : 
  num_boxes = 3 → weight_per_box = 89 → total_weight = num_boxes * weight_per_box → 
  total_weight = 267 := by
  sorry

end NUMINAMATH_CALUDE_pallet_weight_l825_82554


namespace NUMINAMATH_CALUDE_range_of_a_l825_82565

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → a ∈ Set.Iic 1 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l825_82565


namespace NUMINAMATH_CALUDE_fourth_student_number_l825_82588

def class_size : ℕ := 54
def sample_size : ℕ := 4

def systematic_sample (start : ℕ) : Fin 4 → ℕ :=
  λ i => (start + i.val * 13) % class_size + 1

theorem fourth_student_number (h : ∃ start : ℕ, 
  (systematic_sample start 1 = 3 ∧ 
   systematic_sample start 2 = 29 ∧ 
   systematic_sample start 3 = 42)) : 
  ∃ start : ℕ, systematic_sample start 0 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_number_l825_82588


namespace NUMINAMATH_CALUDE_stating_probability_same_district_l825_82558

/-- Represents the four districts available for housing applications. -/
inductive District : Type
  | A
  | B
  | C
  | D

/-- The number of districts available. -/
def num_districts : ℕ := 4

/-- Represents an application scenario for two applicants. -/
def ApplicationScenario : Type := District × District

/-- The total number of possible application scenarios for two applicants. -/
def total_scenarios : ℕ := num_districts * num_districts

/-- Predicate to check if two applicants applied for the same district. -/
def same_district (scenario : ApplicationScenario) : Prop :=
  scenario.1 = scenario.2

/-- The number of scenarios where two applicants apply for the same district. -/
def num_same_district_scenarios : ℕ := num_districts

/-- 
Theorem stating that the probability of two applicants choosing the same district
is 1/4, given that there are four equally likely choices for each applicant.
-/
theorem probability_same_district :
  (num_same_district_scenarios : ℚ) / total_scenarios = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_stating_probability_same_district_l825_82558


namespace NUMINAMATH_CALUDE_pie_slices_yesterday_l825_82552

theorem pie_slices_yesterday (total : ℕ) (today : ℕ) (yesterday : ℕ) : 
  total = 7 → today = 2 → yesterday = total - today → yesterday = 5 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_yesterday_l825_82552


namespace NUMINAMATH_CALUDE_two_sqrt_two_minus_three_is_negative_l825_82572

theorem two_sqrt_two_minus_three_is_negative : 2 * Real.sqrt 2 - 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_two_minus_three_is_negative_l825_82572


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_45_4095_l825_82511

theorem gcd_lcm_sum_45_4095 : 
  (Nat.gcd 45 4095) + (Nat.lcm 45 4095) = 4140 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_45_4095_l825_82511


namespace NUMINAMATH_CALUDE_profit_margin_calculation_l825_82555

/-- The originally anticipated profit margin given a 6.4% decrease in purchase price
    and an 8 percentage point increase in profit margin -/
def original_profit_margin : ℝ := 117

/-- The decrease in purchase price as a percentage -/
def price_decrease : ℝ := 6.4

/-- The increase in profit margin in percentage points -/
def margin_increase : ℝ := 8

theorem profit_margin_calculation :
  let new_purchase_price : ℝ := 100 - price_decrease
  let new_profit_margin : ℝ := original_profit_margin + margin_increase
  (100 + original_profit_margin) * 100 = new_purchase_price * (100 + new_profit_margin) := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_calculation_l825_82555


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l825_82551

/-- A rhombus with an inscribed square -/
structure RhombusWithSquare where
  /-- Length of the first diagonal of the rhombus -/
  d1 : ℝ
  /-- Length of the second diagonal of the rhombus -/
  d2 : ℝ
  /-- The first diagonal is positive -/
  d1_pos : 0 < d1
  /-- The second diagonal is positive -/
  d2_pos : 0 < d2
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed with sides parallel to rhombus diagonals -/
  inscribed : square_side > 0

/-- Theorem stating the side length of the inscribed square in a rhombus with given diagonals -/
theorem inscribed_square_side_length (r : RhombusWithSquare) (h1 : r.d1 = 8) (h2 : r.d2 = 12) : 
  r.square_side = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l825_82551


namespace NUMINAMATH_CALUDE_table_movement_l825_82527

theorem table_movement (table_length table_width : ℝ) 
  (hl : table_length = 12) (hw : table_width = 9) : 
  let diagonal := Real.sqrt (table_length^2 + table_width^2)
  ∀ L W : ℕ, 
    (L ≥ diagonal ∧ W ≥ diagonal ∧ L ≥ table_length) → 
    (∀ L' W' : ℕ, (L' < L ∨ W' < W) → 
      ¬(L' ≥ diagonal ∧ W' ≥ diagonal ∧ L' ≥ table_length)) → 
    L = 15 ∧ W = 15 :=
by sorry

end NUMINAMATH_CALUDE_table_movement_l825_82527


namespace NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l825_82531

theorem sum_geq_three_cube_root_three (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) : 
  a + b + c ≥ 3 * (3 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l825_82531


namespace NUMINAMATH_CALUDE_intersection_distance_l825_82582

/-- The distance between the points of intersection of x^2 + y = 12 and x + y = 12 is √2 -/
theorem intersection_distance : 
  ∃ (p₁ p₂ : ℝ × ℝ), 
    (p₁.1^2 + p₁.2 = 12 ∧ p₁.1 + p₁.2 = 12) ∧ 
    (p₂.1^2 + p₂.2 = 12 ∧ p₂.1 + p₂.2 = 12) ∧ 
    p₁ ≠ p₂ ∧
    Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l825_82582


namespace NUMINAMATH_CALUDE_min_value_interval_min_value_interval_converse_l825_82593

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The theorem stating the possible values of a -/
theorem min_value_interval (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧
  (∃ x ∈ Set.Icc a (a + 6), f x = 9) →
  a = 2 ∨ a = -10 := by
  sorry

/-- The converse theorem -/
theorem min_value_interval_converse :
  ∀ a : ℝ, (a = 2 ∨ a = -10) →
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧
  (∃ x ∈ Set.Icc a (a + 6), f x = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_interval_min_value_interval_converse_l825_82593


namespace NUMINAMATH_CALUDE_delivery_pay_difference_l825_82540

/-- Calculates the difference in pay between two delivery workers given their delivery counts and pay rate. -/
theorem delivery_pay_difference 
  (oula_deliveries : ℕ) 
  (tona_deliveries_ratio : ℚ) 
  (pay_per_delivery : ℕ) 
  (h1 : oula_deliveries = 96)
  (h2 : tona_deliveries_ratio = 3/4)
  (h3 : pay_per_delivery = 100) :
  (oula_deliveries * pay_per_delivery : ℕ) - (((tona_deliveries_ratio * oula_deliveries) : ℚ).floor * pay_per_delivery) = 2400 := by
  sorry

#check delivery_pay_difference

end NUMINAMATH_CALUDE_delivery_pay_difference_l825_82540


namespace NUMINAMATH_CALUDE_square_side_length_l825_82587

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) :
  perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l825_82587


namespace NUMINAMATH_CALUDE_total_drawing_time_l825_82573

/-- Given Bianca's and Lucas's drawing times, prove their total drawing time is 86 minutes -/
theorem total_drawing_time 
  (bianca_school : Nat) 
  (bianca_home : Nat)
  (lucas_school : Nat)
  (lucas_home : Nat)
  (h1 : bianca_school = 22)
  (h2 : bianca_home = 19)
  (h3 : lucas_school = 10)
  (h4 : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := by
  sorry

#check total_drawing_time

end NUMINAMATH_CALUDE_total_drawing_time_l825_82573


namespace NUMINAMATH_CALUDE_prob_sum_24_is_prob_four_sixes_l825_82561

/-- Represents a fair, standard six-sided die -/
def Die : Type := Fin 6

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : Die) : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 24

/-- The probability of rolling four 6s with four fair, standard six-sided dice -/
def prob_four_sixes : ℚ := (1 / 6) ^ 4

theorem prob_sum_24_is_prob_four_sixes : 
  prob_four_sixes = 1 / 1296 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_24_is_prob_four_sixes_l825_82561


namespace NUMINAMATH_CALUDE_same_number_of_digits_l825_82596

/-- 
Given a natural number n, if k is the number of digits in 1974^n,
then 1974^n + 2^n < 10^k.
This implies that 1974^n and 1974^n + 2^n have the same number of digits.
-/
theorem same_number_of_digits (n : ℕ) : 
  let k := (Nat.log 10 (1974^n) + 1)
  1974^n + 2^n < 10^k := by
  sorry

end NUMINAMATH_CALUDE_same_number_of_digits_l825_82596
