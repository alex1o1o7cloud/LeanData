import Mathlib

namespace NUMINAMATH_CALUDE_elsa_marbles_proof_l4055_405573

/-- The number of marbles in Elsa's new bag -/
def new_bag_marbles : ℕ := by sorry

theorem elsa_marbles_proof :
  let initial_marbles : ℕ := 40
  let lost_at_breakfast : ℕ := 3
  let given_to_susie : ℕ := 5
  let final_marbles : ℕ := 54
  
  new_bag_marbles = 
    final_marbles - 
    (initial_marbles - lost_at_breakfast - given_to_susie + 2 * given_to_susie) := by sorry

end NUMINAMATH_CALUDE_elsa_marbles_proof_l4055_405573


namespace NUMINAMATH_CALUDE_f_increasing_f_comparison_l4055_405525

noncomputable section

-- Define the function f with the given property
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → a * f a + b * f b > a * f b + b * f a

-- Theorem 1: f is monotonically increasing
theorem f_increasing (f : ℝ → ℝ) (hf : f_property f) :
  Monotone f := by sorry

-- Theorem 2: f(x+y) > f(6) under given conditions
theorem f_comparison (f : ℝ → ℝ) (hf : f_property f) (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_eq : 4/x + 9/y = 4) :
  f (x + y) > f 6 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_comparison_l4055_405525


namespace NUMINAMATH_CALUDE_smallest_b_for_inequality_l4055_405583

theorem smallest_b_for_inequality : ∃ b : ℕ, (∀ k : ℕ, 27^k > 3^24 → k ≥ b) ∧ 27^b > 3^24 :=
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_inequality_l4055_405583


namespace NUMINAMATH_CALUDE_base12_addition_correct_l4055_405591

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Digit12 to its decimal value --/
def toDecimal (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal value --/
def base12ToDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => toDecimal d + 12 * acc) 0

/-- Addition in base 12 --/
def addBase12 (a b : Base12) : Base12 :=
  sorry -- Implementation details omitted

theorem base12_addition_correct :
  addBase12 [Digit12.D8, Digit12.A, Digit12.D2] [Digit12.D3, Digit12.B, Digit12.D7] =
  [Digit12.D1, Digit12.D0, Digit12.D9, Digit12.D9] :=
by sorry

end NUMINAMATH_CALUDE_base12_addition_correct_l4055_405591


namespace NUMINAMATH_CALUDE_cube_side_length_ratio_l4055_405554

/-- Given two cubes of the same material, if the weight of the second cube
    is 8 times the weight of the first cube, then the ratio of the side length
    of the second cube to the side length of the first cube is 2:1. -/
theorem cube_side_length_ratio (s1 s2 : ℝ) (w1 w2 : ℝ) : 
  s1 > 0 → s2 > 0 → w1 > 0 → w2 > 0 →
  (w2 = 8 * w1) →
  (w1 = s1^3) →
  (w2 = s2^3) →
  s2 / s1 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_side_length_ratio_l4055_405554


namespace NUMINAMATH_CALUDE_simplify_sum_of_roots_l4055_405559

theorem simplify_sum_of_roots : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_roots_l4055_405559


namespace NUMINAMATH_CALUDE_exists_distinct_diagonal_products_l4055_405522

/-- A type representing the vertices of a nonagon -/
inductive Vertex : Type
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9

/-- A function type representing an arrangement of numbers on the nonagon vertices -/
def Arrangement := Vertex → Fin 9

/-- The set of all diagonals in a nonagon -/
def Diagonals : Set (Vertex × Vertex) := sorry

/-- Calculate the product of numbers at the ends of a diagonal -/
def diagonalProduct (arr : Arrangement) (d : Vertex × Vertex) : Nat := sorry

/-- Theorem stating that there exists an arrangement with all distinct diagonal products -/
theorem exists_distinct_diagonal_products :
  ∃ (arr : Arrangement), Function.Injective (diagonalProduct arr) := by sorry

end NUMINAMATH_CALUDE_exists_distinct_diagonal_products_l4055_405522


namespace NUMINAMATH_CALUDE_expression_simplification_l4055_405527

theorem expression_simplification (m n : ℚ) 
  (hm : m = 2) 
  (hn : n = -1/2) : 
  3 * (m^2 - m + n^2) - 2 * (1/2 * m^2 - m*n + 3/2 * n^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4055_405527


namespace NUMINAMATH_CALUDE_green_packs_count_l4055_405533

-- Define the number of balls per pack
def balls_per_pack : ℕ := 10

-- Define the number of packs for red and yellow balls
def red_packs : ℕ := 4
def yellow_packs : ℕ := 8

-- Define the total number of balls
def total_balls : ℕ := 160

-- Define the number of packs of green balls
def green_packs : ℕ := (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack

-- Theorem statement
theorem green_packs_count : green_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_packs_count_l4055_405533


namespace NUMINAMATH_CALUDE_area_is_60_perimeter_is_40_l4055_405579

/-- Triangle with side lengths 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The area of the right triangle is 60 -/
theorem area_is_60 (t : RightTriangle) : (1/2) * t.a * t.b = 60 := by sorry

/-- The perimeter of the right triangle is 40 -/
theorem perimeter_is_40 (t : RightTriangle) : t.a + t.b + t.c = 40 := by sorry

end NUMINAMATH_CALUDE_area_is_60_perimeter_is_40_l4055_405579


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l4055_405532

theorem binomial_expansion_arithmetic_sequence (n : ℕ) : 
  (∃ d : ℚ, 1 + d = n / 2 ∧ n / 2 + d = n * (n - 1) / 8) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l4055_405532


namespace NUMINAMATH_CALUDE_sets_intersection_union_l4055_405568

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- Define the open interval (2009, 2010]
def openInterval : Set ℝ := Set.Ioc 2009 2010

-- State the theorem
theorem sets_intersection_union (a b : ℝ) : 
  (M ∪ N a b = Set.univ) ∧ (M ∩ N a b = openInterval) → a = 2009 ∧ b = 2010 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_union_l4055_405568


namespace NUMINAMATH_CALUDE_solve_chips_problem_l4055_405560

def chips_problem (total father_chips brother_chips : ℕ) : Prop :=
  total = 800 ∧ father_chips = 268 ∧ brother_chips = 182 →
  total - (father_chips + brother_chips) = 350

theorem solve_chips_problem :
  ∀ (total father_chips brother_chips : ℕ),
    chips_problem total father_chips brother_chips :=
by
  sorry

end NUMINAMATH_CALUDE_solve_chips_problem_l4055_405560


namespace NUMINAMATH_CALUDE_no_solution_exists_l4055_405523

theorem no_solution_exists : ¬∃ (f c₁ c₂ : ℕ), 
  (f > 0) ∧ (c₁ > 0) ∧ (c₂ > 0) ∧ 
  (∃ k : ℕ, f = k * (c₁ + c₂)) ∧
  (f + 5 = 2 * ((c₁ + 5) + (c₂ + 5))) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4055_405523


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l4055_405507

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z = (2 - Complex.I) / (2 + Complex.I) ∧ 
  0 < z.re ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l4055_405507


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l4055_405574

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_power_of_3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

def is_power_of_7 (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

def digit_in_number (d : ℕ) (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 10 + d + b * 100 ∧ d < 10

theorem cross_number_puzzle :
  ∃! d : ℕ, 
    (∃ n : ℕ, is_three_digit n ∧ is_power_of_3 n ∧ digit_in_number d n) ∧
    (∃ m : ℕ, is_three_digit m ∧ is_power_of_7 m ∧ digit_in_number d m) ∧
    d = 4 :=
sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l4055_405574


namespace NUMINAMATH_CALUDE_hidden_dots_sum_l4055_405563

/-- The sum of numbers on a single die --/
def single_die_sum : ℕ := 21

/-- The total number of dice --/
def total_dice : ℕ := 4

/-- The number of visible faces --/
def visible_faces : ℕ := 10

/-- The sum of visible numbers --/
def visible_sum : ℕ := 37

theorem hidden_dots_sum :
  single_die_sum * total_dice - visible_sum = 47 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_sum_l4055_405563


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l4055_405553

theorem original_price_after_discounts (price : ℝ) : 
  price * (1 - 0.2) * (1 - 0.1) * (1 - 0.05) = 6840 → price = 10000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discounts_l4055_405553


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l4055_405537

theorem arithmetic_mean_greater_than_geometric_mean
  (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≠ b) (ha_pos : a ≠ 0) (hb_pos : b ≠ 0) :
  (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l4055_405537


namespace NUMINAMATH_CALUDE_cord_length_proof_l4055_405557

theorem cord_length_proof (n : ℕ) (longest shortest : ℝ) : 
  n = 19 → 
  longest = 8 → 
  shortest = 2 → 
  (n : ℝ) * (longest / 2 + shortest) = 114 :=
by
  sorry

end NUMINAMATH_CALUDE_cord_length_proof_l4055_405557


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l4055_405593

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let m := n % 100
  if m ≠ 0 then m else last_two_nonzero_digits (n / 10)

theorem last_two_nonzero_digits_70_factorial :
  ∃ n : ℕ, last_two_nonzero_digits (factorial 70) = n ∧ n < 100 := by
  sorry

#eval last_two_nonzero_digits (factorial 70)

end NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l4055_405593


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l4055_405535

theorem halfway_point_between_fractions :
  (1 / 12 + 1 / 14) / 2 = 13 / 168 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l4055_405535


namespace NUMINAMATH_CALUDE_power_approximations_l4055_405506

theorem power_approximations : 
  (|((1.02 : ℝ)^30 - 1.8114)| < 0.00005) ∧ 
  (|((0.996 : ℝ)^13 - 0.9492)| < 0.00005) := by
  sorry

end NUMINAMATH_CALUDE_power_approximations_l4055_405506


namespace NUMINAMATH_CALUDE_hexagon_count_l4055_405581

theorem hexagon_count (initial_sheets : ℕ) (cuts : ℕ) (initial_sides_per_sheet : ℕ) :
  initial_sheets = 15 →
  cuts = 60 →
  initial_sides_per_sheet = 4 →
  let final_sheets := initial_sheets + cuts
  let total_sides := initial_sheets * initial_sides_per_sheet + cuts * 4
  let hexagon_count := (total_sides - 3 * final_sheets) / 3
  hexagon_count = 25 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_count_l4055_405581


namespace NUMINAMATH_CALUDE_heat_engine_efficiency_l4055_405501

theorem heat_engine_efficiency
  (η₀ η₁ η₂ α : ℝ)
  (h1 : η₁ < η₀)
  (h2 : η₂ < η₀)
  (h3 : η₀ < 1)
  (h4 : η₁ < 1)
  (h5 : η₂ = (η₀ - η₁) / (1 - η₁))
  (h6 : η₁ = (1 - 0.01 * α) * η₀) :
  η₂ = α / (100 - (100 - α) * η₀) := by
sorry

end NUMINAMATH_CALUDE_heat_engine_efficiency_l4055_405501


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l4055_405590

/-- The number of letters in an alphabet with specific dot and line properties -/
theorem alphabet_letter_count :
  let dot_and_line : ℕ := 13  -- Letters with both dot and line
  let line_only : ℕ := 24     -- Letters with line but no dot
  let dot_only : ℕ := 3       -- Letters with dot but no line
  let total : ℕ := dot_and_line + line_only + dot_only
  total = 40 := by sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l4055_405590


namespace NUMINAMATH_CALUDE_intersection_length_l4055_405520

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle_O₂ (x y m : ℝ) : Prop := (x + m)^2 + y^2 = 20

-- Define the intersection points
structure IntersectionPoints (m : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : circle_O₁ A.1 A.2
  h₂ : circle_O₂ A.1 A.2 m
  h₃ : circle_O₁ B.1 B.2
  h₄ : circle_O₂ B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicular_tangents (m : ℝ) (A : ℝ × ℝ) : Prop :=
  circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2 m ∧
  (∃ t₁ t₂ : ℝ × ℝ, 
    (t₁.1 * t₂.1 + t₁.2 * t₂.2 = 0) ∧  -- Perpendicular condition
    (t₁.1 * A.1 + t₁.2 * A.2 = 0) ∧    -- Tangent to O₁
    (t₂.1 * (A.1 + m) + t₂.2 * A.2 = 0)) -- Tangent to O₂

-- Theorem statement
theorem intersection_length (m : ℝ) (points : IntersectionPoints m) :
  perpendicular_tangents m points.A →
  Real.sqrt ((points.A.1 - points.B.1)^2 + (points.A.2 - points.B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_length_l4055_405520


namespace NUMINAMATH_CALUDE_adjacent_probability_in_row_of_five_l4055_405597

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The probability of two specific people sitting adjacent in a row of 5 people -/
theorem adjacent_probability_in_row_of_five :
  let total_arrangements := factorial 5
  let adjacent_arrangements := 2 * factorial 4
  (adjacent_arrangements : ℚ) / total_arrangements = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_in_row_of_five_l4055_405597


namespace NUMINAMATH_CALUDE_inequality_proof_l4055_405580

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^3 + b^3 + c^3 + (a*b)/(a^2 + b^2) + (b*c)/(b^2 + c^2) + (c*a)/(c^2 + a^2) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4055_405580


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l4055_405576

/-- Given two points A and B symmetric about the y-axis, prove that m-n = -4 -/
theorem symmetric_points_difference (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (m - 2, 3) ∧ 
    B = (4, n + 1) ∧ 
    (A.1 = -B.1) ∧  -- x-coordinates are opposite
    (A.2 = B.2))    -- y-coordinates are equal
  → m - n = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l4055_405576


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l4055_405513

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 6 →
  c = 4 →
  Real.sin (B / 2) = Real.sqrt 3 / 3 →
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l4055_405513


namespace NUMINAMATH_CALUDE_intersection_of_intervals_l4055_405550

open Set

-- Define the sets A and B
def A : Set ℝ := Ioo (-1) 2
def B : Set ℝ := Ioi 0

-- State the theorem
theorem intersection_of_intervals : A ∩ B = Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_l4055_405550


namespace NUMINAMATH_CALUDE_smallest_x_value_l4055_405548

theorem smallest_x_value (x : ℝ) : x ≠ 1/4 →
  ((20 * x^2 - 49 * x + 20) / (4 * x - 1) + 7 * x = 3 * x + 2) →
  x ≥ 2/9 ∧ (∃ y : ℝ, y ≠ 1/4 ∧ ((20 * y^2 - 49 * y + 20) / (4 * y - 1) + 7 * y = 3 * y + 2) ∧ y = 2/9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l4055_405548


namespace NUMINAMATH_CALUDE_factorial_sum_equals_36018_l4055_405517

theorem factorial_sum_equals_36018 : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 5 = 36018 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_36018_l4055_405517


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l4055_405511

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 5) :
  ∀ x, x^2 - 12*x + 25 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l4055_405511


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l4055_405570

theorem polygon_interior_angles (n : ℕ) (h1 : n > 0) : 
  (n - 2) * 180 = n * 177 → n = 120 := by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l4055_405570


namespace NUMINAMATH_CALUDE_complex_product_sum_l4055_405528

theorem complex_product_sum (i : ℂ) (h : i^2 = -1) : 
  let z := (1 + i) * (1 - i)
  ∃ p q : ℝ, z = p + q * i ∧ p + q = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l4055_405528


namespace NUMINAMATH_CALUDE_first_triangle_height_l4055_405578

/-- Given two triangles where the second has double the area of the first,
    prove that the height of the first triangle is 12 cm. -/
theorem first_triangle_height
  (base1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_base2 : base2 = 20)
  (h_height2 : height2 = 18)
  (h_area_relation : base2 * height2 = 2 * base1 * (12 : ℝ)) :
  ∃ (height1 : ℝ), height1 = 12 ∧ base1 * height1 = (1/2) * base2 * height2 :=
by sorry

end NUMINAMATH_CALUDE_first_triangle_height_l4055_405578


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_half_l4055_405564

theorem trigonometric_sum_equals_half : 
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) - 
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_half_l4055_405564


namespace NUMINAMATH_CALUDE_existence_of_indices_with_inequalities_l4055_405531

theorem existence_of_indices_with_inequalities 
  (a b c : ℕ → ℕ) : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end NUMINAMATH_CALUDE_existence_of_indices_with_inequalities_l4055_405531


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_l4055_405582

theorem range_of_x_minus_2y (x y : ℝ) (hx : -1 ≤ x ∧ x < 2) (hy : 0 < y ∧ y ≤ 1) :
  ∃ (z : ℝ), -3 ≤ z ∧ z < 2 ∧ ∃ (x' y' : ℝ), -1 ≤ x' ∧ x' < 2 ∧ 0 < y' ∧ y' ≤ 1 ∧ z = x' - 2*y' :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_l4055_405582


namespace NUMINAMATH_CALUDE_class_selection_theorem_l4055_405588

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of boys in the class. -/
def num_boys : ℕ := 13

/-- The total number of girls in the class. -/
def num_girls : ℕ := 10

/-- The number of boys selected. -/
def boys_selected : ℕ := 2

/-- The number of girls selected. -/
def girls_selected : ℕ := 1

/-- The total number of possible combinations. -/
def total_combinations : ℕ := 780

theorem class_selection_theorem :
  choose num_boys boys_selected * choose num_girls girls_selected = total_combinations :=
sorry

end NUMINAMATH_CALUDE_class_selection_theorem_l4055_405588


namespace NUMINAMATH_CALUDE_min_value_theorem_l4055_405552

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 10*x + 100/x^2 ≥ 79 ∧ ∃ y > 0, y^2 + 10*y + 100/y^2 = 79 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4055_405552


namespace NUMINAMATH_CALUDE_fraction_transformation_l4055_405519

theorem fraction_transformation (a b c d x : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + x) = c / d) : 
  x = (a * d - b * c) / (c - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l4055_405519


namespace NUMINAMATH_CALUDE_cube_preserves_order_l4055_405524

theorem cube_preserves_order (a b c : ℝ) (h : b > a) : b^3 > a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l4055_405524


namespace NUMINAMATH_CALUDE_circle_area_outside_square_is_zero_l4055_405589

/-- A square with an inscribed circle -/
structure SquareWithCircle where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- The circle is inscribed in the square -/
  circle_inscribed : radius = side_length / 2

/-- The area of the portion of the circle outside the square is zero -/
theorem circle_area_outside_square_is_zero (s : SquareWithCircle) (h : s.side_length = 10) :
  Real.pi * s.radius ^ 2 - s.side_length ^ 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_circle_area_outside_square_is_zero_l4055_405589


namespace NUMINAMATH_CALUDE_circle_area_through_two_points_l4055_405558

/-- The area of a circle with center P(2, -1) passing through Q(-4, 5) is 72π. -/
theorem circle_area_through_two_points :
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 5)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 72 * π := by
  sorry


end NUMINAMATH_CALUDE_circle_area_through_two_points_l4055_405558


namespace NUMINAMATH_CALUDE_score_difference_is_seven_l4055_405504

-- Define the score distribution
def score_distribution : List (Float × Float) := [
  (0.20, 60),
  (0.30, 70),
  (0.25, 85),
  (0.25, 95)
]

-- Define the mean score
def mean_score : Float :=
  (score_distribution.map (λ p => p.1 * p.2)).sum

-- Define the median score
def median_score : Float := 85

-- Theorem statement
theorem score_difference_is_seven :
  median_score - mean_score = 7 := by
  sorry


end NUMINAMATH_CALUDE_score_difference_is_seven_l4055_405504


namespace NUMINAMATH_CALUDE_hash_sum_plus_five_l4055_405500

-- Define the operation #
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- Theorem statement
theorem hash_sum_plus_five (a b : ℕ) : hash a b = 100 → (a + b) + 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hash_sum_plus_five_l4055_405500


namespace NUMINAMATH_CALUDE_sin_minus_cos_value_l4055_405526

theorem sin_minus_cos_value (x : ℝ) (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) : 
  Real.sin x - Real.cos x = -1 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_value_l4055_405526


namespace NUMINAMATH_CALUDE_y_range_given_x_constraints_l4055_405542

theorem y_range_given_x_constraints (x y : ℝ) 
  (h1 : 2 ≤ |x - 5| ∧ |x - 5| ≤ 9) 
  (h2 : y = 3 * x + 2) : 
  y ∈ Set.Icc (-10) 11 ∪ Set.Icc 23 44 := by
  sorry

end NUMINAMATH_CALUDE_y_range_given_x_constraints_l4055_405542


namespace NUMINAMATH_CALUDE_total_notes_count_l4055_405514

def total_amount : ℕ := 10350
def note_50_value : ℕ := 50
def note_500_value : ℕ := 500
def num_50_notes : ℕ := 17

theorem total_notes_count : 
  ∃ (num_500_notes : ℕ), 
    num_50_notes * note_50_value + num_500_notes * note_500_value = total_amount ∧
    num_50_notes + num_500_notes = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_notes_count_l4055_405514


namespace NUMINAMATH_CALUDE_infinite_cube_square_triples_l4055_405567

theorem infinite_cube_square_triples :
  ∃ S : Set (ℤ × ℤ × ℤ), Set.Infinite S ∧ 
  ∀ (x y z : ℤ), (x, y, z) ∈ S → x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_cube_square_triples_l4055_405567


namespace NUMINAMATH_CALUDE_R_final_coordinates_l4055_405572

/-- Reflect a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflect a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The initial point R -/
def R : ℝ × ℝ := (0, -5)

/-- The sequence of reflections applied to R -/
def R_transformed : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x R))

theorem R_final_coordinates :
  R_transformed = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_R_final_coordinates_l4055_405572


namespace NUMINAMATH_CALUDE_sin_intersection_sum_l4055_405592

open Real

theorem sin_intersection_sum (f : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) :
  (∀ x ∈ Set.Icc 0 (7 * π / 6), f x = Real.sin (2 * x + π / 6)) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₁ ∈ Set.Icc 0 (7 * π / 6) →
  x₂ ∈ Set.Icc 0 (7 * π / 6) →
  x₃ ∈ Set.Icc 0 (7 * π / 6) →
  f x₁ = f x₂ →
  f x₂ = f x₃ →
  x₁ + 2 * x₂ + x₃ = 5 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_sin_intersection_sum_l4055_405592


namespace NUMINAMATH_CALUDE_pairs_count_l4055_405512

/-- S(n) denotes the sum of the digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of pairs <m, n> satisfying the given conditions -/
def count_pairs : ℕ := sorry

theorem pairs_count :
  count_pairs = 99 ∧
  ∀ m n : ℕ,
    m < 100 →
    n < 100 →
    m > n →
    m + S n = n + 2 * S m →
    (m, n) ∈ (Finset.filter (fun p : ℕ × ℕ => 
      p.1 < 100 ∧
      p.2 < 100 ∧
      p.1 > p.2 ∧
      p.1 + S p.2 = p.2 + 2 * S p.1)
    (Finset.product (Finset.range 100) (Finset.range 100))) :=
by sorry

end NUMINAMATH_CALUDE_pairs_count_l4055_405512


namespace NUMINAMATH_CALUDE_logarithm_order_comparison_l4055_405549

theorem logarithm_order_comparison : 
  Real.log 4 / Real.log 3 > Real.log 3 / Real.log 4 ∧ 
  Real.log 3 / Real.log 4 > Real.log (3/4) / Real.log (4/3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_order_comparison_l4055_405549


namespace NUMINAMATH_CALUDE_extreme_values_and_increasing_condition_l4055_405584

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem extreme_values_and_increasing_condition :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → f (-1/2) x ≤ f (-1/2) x₀) ∧
  f (-1/2) x₀ = 0 ∧
  (∀ (y : ℝ), y > 0 → ∃ (z : ℝ), z > y ∧ f (-1/2) z > f (-1/2) y) ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), 0 < x ∧ x < y → f a x < f a y) ↔ a ≥ 1 / (2 * Real.exp 2)) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_increasing_condition_l4055_405584


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_l4055_405509

/-- A line intersects a circle -/
structure LineCircleIntersection where
  /-- Slope of the line y = kx + 3 -/
  k : ℝ
  /-- The line intersects the circle (x-1)^2 + (y-2)^2 = 9 at two points -/
  intersects : k > 1
  /-- The distance between the two intersection points is 12√5/5 -/
  distance : ℝ
  distance_eq : distance = 12 * Real.sqrt 5 / 5

/-- The slope k of the line is 2 -/
theorem line_circle_intersection_k (lci : LineCircleIntersection) : lci.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_k_l4055_405509


namespace NUMINAMATH_CALUDE_gcf_of_48_180_98_l4055_405515

theorem gcf_of_48_180_98 : Nat.gcd 48 (Nat.gcd 180 98) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_48_180_98_l4055_405515


namespace NUMINAMATH_CALUDE_joyce_initial_apples_l4055_405556

/-- The number of apples Joyce started with -/
def initial_apples : ℕ := sorry

/-- The number of apples Larry gave to Joyce -/
def apples_from_larry : ℚ := 52.0

/-- The total number of apples Joyce has after receiving apples from Larry -/
def total_apples : ℕ := 127

/-- Theorem stating that Joyce started with 75 apples -/
theorem joyce_initial_apples :
  initial_apples = 75 :=
by sorry

end NUMINAMATH_CALUDE_joyce_initial_apples_l4055_405556


namespace NUMINAMATH_CALUDE_six_digit_difference_not_divisible_l4055_405508

theorem six_digit_difference_not_divisible (A B : ℕ) : 
  100 ≤ A ∧ A < 1000 → 100 ≤ B ∧ B < 1000 → A ≠ B → 
  ¬(∃ k : ℤ, 999 * (A - B) = 1976 * k) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_difference_not_divisible_l4055_405508


namespace NUMINAMATH_CALUDE_min_ratio_digit_difference_l4055_405505

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def ratio (n : ℕ) : ℚ := n / (digit_sum n)

def ten_thousands_digit (n : ℕ) : ℕ := n / 10000

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

theorem min_ratio_digit_difference :
  ∃ (n : ℕ), is_five_digit n ∧
  (∀ (m : ℕ), is_five_digit m → ratio n ≤ ratio m) ∧
  (thousands_digit n - ten_thousands_digit n = 8) :=
sorry

end NUMINAMATH_CALUDE_min_ratio_digit_difference_l4055_405505


namespace NUMINAMATH_CALUDE_triangle_area_change_l4055_405565

theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = 1.4 * base) 
  (h2 : height_new = 0.6 * height) 
  (h3 : area = (base * height) / 2) 
  (h4 : area_new = (base_new * height_new) / 2) : 
  area_new = 0.42 * area := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l4055_405565


namespace NUMINAMATH_CALUDE_horner_method_correct_l4055_405575

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 1 + x + 2x^2 + 3x^3 + 4x^4 + 5x^5 -/
def f (x : ℝ) : ℝ :=
  1 + x + 2*x^2 + 3*x^3 + 4*x^4 + 5*x^5

theorem horner_method_correct :
  horner [5, 4, 3, 2, 1, 1] (-1) = f (-1) ∧ f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_correct_l4055_405575


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4055_405571

theorem imaginary_part_of_complex_fraction : Complex.im ((2 * Complex.I) / (1 - Complex.I) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4055_405571


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l4055_405596

theorem largest_angle_in_triangle (x : ℝ) : 
  x + 40 + 60 = 180 → 
  max x (max 40 60) = 80 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l4055_405596


namespace NUMINAMATH_CALUDE_expression_evaluation_l4055_405545

theorem expression_evaluation (b : ℝ) (h : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / (2 * b) = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4055_405545


namespace NUMINAMATH_CALUDE_min_value_expression_l4055_405562

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2)) / (a * b * c) ≥ 343 ∧
  ((1^2 + 5*1 + 2) * (1^2 + 5*1 + 2) * (1^2 + 5*1 + 2)) / (1 * 1 * 1) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4055_405562


namespace NUMINAMATH_CALUDE_private_teacher_cost_l4055_405546

/-- Calculates the amount each parent must pay for a private teacher --/
theorem private_teacher_cost 
  (former_salary : ℕ) 
  (raise_percentage : ℚ) 
  (num_kids : ℕ) 
  (h1 : former_salary = 45000)
  (h2 : raise_percentage = 1/5)
  (h3 : num_kids = 9) :
  (former_salary + former_salary * raise_percentage) / num_kids = 6000 := by
  sorry

#check private_teacher_cost

end NUMINAMATH_CALUDE_private_teacher_cost_l4055_405546


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4055_405547

/-- A geometric sequence is a sequence where the ratio between successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Nat.factorial n)

theorem geometric_sequence_first_term :
  ∀ a : ℕ → ℝ,
  IsGeometricSequence a →
  a 4 = factorial 6 →
  a 6 = factorial 7 →
  a 1 = (720 : ℝ) * Real.sqrt 7 / 49 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4055_405547


namespace NUMINAMATH_CALUDE_sqrt_comparison_l4055_405587

theorem sqrt_comparison : Real.sqrt 7 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l4055_405587


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l4055_405538

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_9 : fib 150 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l4055_405538


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l4055_405510

theorem cos_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I →
  Complex.exp (δ * Complex.I) = -(5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I →
  Real.cos (γ + δ) = -(56 / 65) := by
sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l4055_405510


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l4055_405539

theorem two_red_two_blue_probability (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 12)
  (h3 : blue_marbles = 8) :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 : ℚ) / Nat.choose total_marbles 4 = 1848 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l4055_405539


namespace NUMINAMATH_CALUDE_luis_red_socks_l4055_405551

/-- The number of pairs of red socks Luis bought -/
def red_socks : ℕ := sorry

/-- The number of pairs of blue socks Luis bought -/
def blue_socks : ℕ := 6

/-- The cost of each pair of red socks in dollars -/
def red_sock_cost : ℕ := 3

/-- The cost of each pair of blue socks in dollars -/
def blue_sock_cost : ℕ := 5

/-- The total amount Luis spent in dollars -/
def total_spent : ℕ := 42

/-- Theorem stating that Luis bought 4 pairs of red socks -/
theorem luis_red_socks : 
  red_socks * red_sock_cost + blue_socks * blue_sock_cost = total_spent → 
  red_socks = 4 := by sorry

end NUMINAMATH_CALUDE_luis_red_socks_l4055_405551


namespace NUMINAMATH_CALUDE_min_value_approx_l4055_405595

-- Define the function to be minimized
def f (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / (a + b + c)

-- Define the conditions
def valid_digits (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ b > 3

-- Theorem statement
theorem min_value_approx (a b c : ℕ) (h : valid_digits a b c) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ f a b c ≥ 19.62 - ε :=
sorry

end NUMINAMATH_CALUDE_min_value_approx_l4055_405595


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l4055_405521

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (5 : ℚ) / 8 → q' ≥ q) →
  q - p = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l4055_405521


namespace NUMINAMATH_CALUDE_no_leftover_eggs_l4055_405544

/-- The number of eggs Abigail has -/
def abigail_eggs : ℕ := 28

/-- The number of eggs Beatrice has -/
def beatrice_eggs : ℕ := 53

/-- The number of eggs Carson has -/
def carson_eggs : ℕ := 19

/-- The number of eggs in each carton -/
def carton_size : ℕ := 10

/-- Theorem stating that the remainder of the total number of eggs divided by the carton size is 0 -/
theorem no_leftover_eggs : (abigail_eggs + beatrice_eggs + carson_eggs) % carton_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_leftover_eggs_l4055_405544


namespace NUMINAMATH_CALUDE_coin_problem_l4055_405503

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def total_coins : ℕ := 11
def total_value : ℕ := 118

theorem coin_problem (p n d q : ℕ) : 
  p ≥ 1 → n ≥ 1 → d ≥ 1 → q ≥ 1 →
  p + n + d + q = total_coins →
  p * penny + n * nickel + d * dime + q * quarter = total_value →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l4055_405503


namespace NUMINAMATH_CALUDE_number_value_l4055_405530

theorem number_value (x : ℚ) (n : ℚ) : 
  x = 12 → n + 7 / x = 6 - 5 / x → n = 5 := by sorry

end NUMINAMATH_CALUDE_number_value_l4055_405530


namespace NUMINAMATH_CALUDE_polynomial_division_result_l4055_405516

variables {a p x : ℝ}

theorem polynomial_division_result :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l4055_405516


namespace NUMINAMATH_CALUDE_function_equation_solution_l4055_405518

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2) - f (y^2) + 2*x + 1 = f (x + y) * f (x - y)) : 
  (∀ x : ℝ, f x = x + 1) ∨ (∀ x : ℝ, f x = -x - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l4055_405518


namespace NUMINAMATH_CALUDE_sin_double_angle_on_unit_circle_l4055_405555

/-- Given a point B on the unit circle with coordinates (-3/5, 4/5), 
    prove that sin(2α) = -24/25, where α is the angle formed by OA and OB, 
    and O is the origin and A is the point (1,0) on the unit circle. -/
theorem sin_double_angle_on_unit_circle 
  (B : ℝ × ℝ) 
  (h_B_on_circle : B.1^2 + B.2^2 = 1) 
  (h_B_coords : B = (-3/5, 4/5)) 
  (α : ℝ) 
  (h_α_def : α = Real.arccos B.1) : 
  Real.sin (2 * α) = -24/25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_on_unit_circle_l4055_405555


namespace NUMINAMATH_CALUDE_x_equals_nine_l4055_405585

/-- The star operation defined as a ⭐ b = 5a - 3b -/
def star (a b : ℝ) : ℝ := 5 * a - 3 * b

/-- Theorem stating that X = 9 given the condition X ⭐ (3 ⭐ 2) = 18 -/
theorem x_equals_nine : ∃ X : ℝ, star X (star 3 2) = 18 ∧ X = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_nine_l4055_405585


namespace NUMINAMATH_CALUDE_percentage_of_male_students_l4055_405534

theorem percentage_of_male_students (M F : ℝ) : 
  M + F = 100 →
  0.60 * M + 0.70 * F = 66 →
  M = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_male_students_l4055_405534


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l4055_405569

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 216 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 2180 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l4055_405569


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4055_405594

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∀ x, f x = 0 ↔ (x - sum_of_roots / 2)^2 = (sum_of_roots^2 - 4 * (b^2 - 4*a*c) / (4*a)) / 4) →
  sum_of_roots = 5 ↔ a = 1 ∧ b = -5 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4055_405594


namespace NUMINAMATH_CALUDE_pencil_price_after_discount_l4055_405561

/-- Given a pencil with an original cost and a discount, calculate the final price. -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem stating that for a pencil with an original cost of 4 dollars and a discount of 0.63 dollars, the final price is 3.37 dollars. -/
theorem pencil_price_after_discount :
  final_price 4 0.63 = 3.37 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_after_discount_l4055_405561


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_problem_solution_l4055_405599

/-- Represents a polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  { vertices := sorry }  -- Implementation details omitted

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  List.sum (List.map Prod.fst p.vertices)

theorem midpoint_sum_invariant (p : Polygon) (h : p.vertices.length = 50) :
  let p2 := midpointPolygon p
  let p3 := midpointPolygon p2
  sumXCoordinates p3 = sumXCoordinates p := by
  sorry

/-- The main theorem that proves the result for the specific case in the problem -/
theorem problem_solution (p : Polygon) (h1 : p.vertices.length = 50) (h2 : sumXCoordinates p = 1005) :
  let p2 := midpointPolygon p
  let p3 := midpointPolygon p2
  sumXCoordinates p3 = 1005 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_problem_solution_l4055_405599


namespace NUMINAMATH_CALUDE_inequality_solution_l4055_405577

theorem inequality_solution : 
  ∀ x y : ℝ, y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y → 
  ((x = 0 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2)) ∧
  (x = 0 ∧ y = 0 → y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y) ∧
  (x = 1/2 ∧ y = 1/2 → y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4055_405577


namespace NUMINAMATH_CALUDE_students_with_B_grade_l4055_405566

def total_students : ℕ := 40

def prob_A (x : ℕ) : ℚ := (3 : ℚ) / 5 * x
def prob_B (x : ℕ) : ℚ := x
def prob_C (x : ℕ) : ℚ := (6 : ℚ) / 5 * x

theorem students_with_B_grade :
  ∃ x : ℕ, 
    x ≤ total_students ∧
    (prob_A x + prob_B x + prob_C x : ℚ) = total_students ∧
    x = 14 := by sorry

end NUMINAMATH_CALUDE_students_with_B_grade_l4055_405566


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l4055_405502

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ,
    x ≤ y →
    x^2 + y^2 = 3 * 2016^z + 77 →
    ((x = 4 ∧ y = 8 ∧ z = 0) ∨
     (x = 14 ∧ y = 77 ∧ z = 1) ∨
     (x = 35 ∧ y = 70 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l4055_405502


namespace NUMINAMATH_CALUDE_cubic_root_function_l4055_405541

/-- Given a function y = kx^(1/3) where y = 5√2 when x = 64, 
    prove that y = 2.5√2 when x = 8 -/
theorem cubic_root_function (k : ℝ) : 
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 5 * Real.sqrt 2) →
  k * 8^(1/3) = 2.5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_cubic_root_function_l4055_405541


namespace NUMINAMATH_CALUDE_kaleb_chocolate_bars_l4055_405598

/-- The number of chocolate bars Kaleb needs to sell -/
def total_chocolate_bars (bars_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  bars_per_box * num_boxes

/-- Theorem stating the total number of chocolate bars Kaleb needs to sell -/
theorem kaleb_chocolate_bars :
  total_chocolate_bars 5 142 = 710 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_chocolate_bars_l4055_405598


namespace NUMINAMATH_CALUDE_expression_evaluation_l4055_405543

theorem expression_evaluation (b : ℝ) (a : ℝ) (h1 : b = 2) (h2 : a = b + 3) :
  (a^2 + b)^2 - (a^2 - b)^2 = 200 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4055_405543


namespace NUMINAMATH_CALUDE_value_range_equivalence_l4055_405540

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem value_range_equivalence (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a x ∈ Set.Icc (f a a) (f a 4)) ↔ 
  a ∈ Set.Icc (-2 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_value_range_equivalence_l4055_405540


namespace NUMINAMATH_CALUDE_ant_final_position_l4055_405536

/-- Represents the vertices of the rectangle --/
inductive Vertex : Type
  | A : Vertex
  | B : Vertex
  | C : Vertex
  | D : Vertex

/-- Represents a single movement of the ant --/
def next_vertex : Vertex → Vertex
  | Vertex.A => Vertex.B
  | Vertex.B => Vertex.C
  | Vertex.C => Vertex.D
  | Vertex.D => Vertex.A

/-- Represents multiple movements of the ant --/
def ant_position (start : Vertex) (moves : Nat) : Vertex :=
  match moves with
  | 0 => start
  | n + 1 => next_vertex (ant_position start n)

/-- The main theorem to prove --/
theorem ant_final_position :
  ant_position Vertex.A 2018 = Vertex.C := by
  sorry

end NUMINAMATH_CALUDE_ant_final_position_l4055_405536


namespace NUMINAMATH_CALUDE_trigonometric_identities_l4055_405529

theorem trigonometric_identities (α : Real) (h : Real.tan α = -3/4) :
  (Real.sin (2 * Real.pi - α) + Real.cos (5/2 * Real.pi + α)) / Real.sin (α - Real.pi/2) = -3/2 ∧
  (Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l4055_405529


namespace NUMINAMATH_CALUDE_max_product_constrained_l4055_405586

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 8 * b = 48) :
  a * b ≤ 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 8 * b₀ = 48 ∧ a₀ * b₀ = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l4055_405586
