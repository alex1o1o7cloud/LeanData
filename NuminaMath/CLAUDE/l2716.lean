import Mathlib

namespace NUMINAMATH_CALUDE_max_true_statements_four_true_statements_possible_l2716_271695

theorem max_true_statements (a b : ℝ) : 
  ¬(1/a < 1/b ∧ a^3 < b^3 ∧ a < b ∧ a < 0 ∧ b < 0) :=
by sorry

theorem four_true_statements_possible (a b : ℝ) : 
  ∃ (a b : ℝ), a^3 < b^3 ∧ a < b ∧ a < 0 ∧ b < 0 :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_four_true_statements_possible_l2716_271695


namespace NUMINAMATH_CALUDE_compare_powers_l2716_271611

theorem compare_powers : 2^2023 * 7^2023 < 3^2023 * 5^2023 := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l2716_271611


namespace NUMINAMATH_CALUDE_equation_solution_l2716_271647

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (3 / x + (4 / x) / (8 / x) = 1.5) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2716_271647


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l2716_271684

/-- Represents a standard die with opposite faces summing to 7 -/
structure StandardDie :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- Represents the 4x4x4 cube constructed from standard dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → StandardDie

/-- Calculates the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the smallest possible sum of visible faces is 144 -/
theorem smallest_visible_sum (cube : LargeCube) : 
  ∃ (min_cube : LargeCube), visibleSum min_cube = 144 ∧ ∀ (c : LargeCube), visibleSum c ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l2716_271684


namespace NUMINAMATH_CALUDE_max_true_statements_l2716_271619

theorem max_true_statements (a b : ℝ) : ∃ a b : ℝ,
  (a^2 + b^2 < (a + b)^2) ∧
  (a * b > 0) ∧
  (a > b) ∧
  (a > 0) ∧
  (b > 0) := by
sorry

end NUMINAMATH_CALUDE_max_true_statements_l2716_271619


namespace NUMINAMATH_CALUDE_jam_distribution_l2716_271658

/-- The jam distribution problem -/
theorem jam_distribution (total_jam : ℝ) (ponchik_hypothetical_days : ℝ) (syrupchik_hypothetical_days : ℝ)
  (h_total : total_jam = 100)
  (h_ponchik : ponchik_hypothetical_days = 45)
  (h_syrupchik : syrupchik_hypothetical_days = 20) :
  ∃ (ponchik_jam syrupchik_jam ponchik_rate syrupchik_rate : ℝ),
    ponchik_jam + syrupchik_jam = total_jam ∧
    ponchik_jam = 40 ∧
    syrupchik_jam = 60 ∧
    ponchik_rate = 4/3 ∧
    syrupchik_rate = 2 ∧
    ponchik_jam / ponchik_rate = syrupchik_jam / syrupchik_rate ∧
    syrupchik_jam / ponchik_hypothetical_days = ponchik_rate ∧
    ponchik_jam / syrupchik_hypothetical_days = syrupchik_rate :=
by sorry

end NUMINAMATH_CALUDE_jam_distribution_l2716_271658


namespace NUMINAMATH_CALUDE_valid_3x3_grid_exists_l2716_271634

/-- Represents a county with a diagonal road -/
inductive County
  | NorthEast
  | SouthWest

/-- Represents a 3x3 grid of counties -/
def Grid := Fin 3 → Fin 3 → County

/-- Checks if two adjacent counties have compatible road directions -/
def compatible (c1 c2 : County) : Bool :=
  match c1, c2 with
  | County.NorthEast, County.SouthWest => true
  | County.SouthWest, County.NorthEast => true
  | _, _ => false

/-- Checks if the grid forms a valid closed path -/
def isValidPath (g : Grid) : Bool :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that a valid 3x3 grid configuration exists -/
theorem valid_3x3_grid_exists : ∃ g : Grid, isValidPath g := by
  sorry

end NUMINAMATH_CALUDE_valid_3x3_grid_exists_l2716_271634


namespace NUMINAMATH_CALUDE_flagpole_break_height_l2716_271600

theorem flagpole_break_height (h : ℝ) (b : ℝ) (break_height : ℝ) :
  h = 8 →
  b = 3 →
  break_height = (Real.sqrt (h^2 + b^2)) / 2 →
  break_height = Real.sqrt 73 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l2716_271600


namespace NUMINAMATH_CALUDE_quadratic_with_irrational_root_l2716_271653

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a = 1 ∧ 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = Real.sqrt 3 - 2 ∨ x = -Real.sqrt 3 - 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_with_irrational_root_l2716_271653


namespace NUMINAMATH_CALUDE_prism_volume_l2716_271614

/-- The volume of a right rectangular prism with face areas 15, 10, and 30 -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 30) :
  l * w * h = 30 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2716_271614


namespace NUMINAMATH_CALUDE_perfect_squares_is_good_l2716_271623

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def perfect_squares : Set ℕ := {n : ℕ | is_perfect_square n}

def is_good (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → 
    ∀ p q : ℕ, Prime p → Prime q → p ≠ q → p ∣ n → q ∣ n →
      ¬(n - p ∈ A ∧ n - q ∈ A)

theorem perfect_squares_is_good : is_good perfect_squares :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_is_good_l2716_271623


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2716_271638

theorem problem_1 : (1) - 2^2 + (-1/2)^4 + (3 - Real.pi)^0 = -47/16 := by sorry

theorem problem_2 : 5^2022 * (-1/5)^2023 = -1/5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2716_271638


namespace NUMINAMATH_CALUDE_sum_inequality_l2716_271678

theorem sum_inequality (a b c : ℕ+) (h : (a * b * c : ℚ) = 1) :
  (1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) : ℚ) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2716_271678


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2716_271694

theorem fraction_subtraction : (7 : ℚ) / 3 - 5 / 6 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2716_271694


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2716_271645

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 :=
by sorry

theorem min_value_achieved (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (a₀ b₀ c₀ : ℝ), 2 ≤ a₀ ∧ a₀ ≤ b₀ ∧ b₀ ≤ c₀ ∧ c₀ ≤ 5 ∧
    (a₀ - 2)^2 + (b₀/a₀ - 1)^2 + (c₀/b₀ - 1)^2 + (5/c₀ - 1)^2 = 4 * (5^(1/4) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2716_271645


namespace NUMINAMATH_CALUDE_minimum_chocolates_l2716_271646

theorem minimum_chocolates (n : ℕ) : n ≥ 118 →
  (n % 6 = 4 ∧ n % 8 = 6 ∧ n % 10 = 8) →
  ∃ (m : ℕ), m < n → ¬(m % 6 = 4 ∧ m % 8 = 6 ∧ m % 10 = 8) :=
by sorry

end NUMINAMATH_CALUDE_minimum_chocolates_l2716_271646


namespace NUMINAMATH_CALUDE_isosceles_non_equilateral_distinct_lines_l2716_271696

/-- A triangle in a 2D Euclidean space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is isosceles --/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Predicate to check if a triangle is equilateral --/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Function to count distinct lines representing altitudes, medians, and interior angle bisectors --/
def countDistinctLines (t : Triangle) : ℕ := sorry

/-- Theorem stating that an isosceles non-equilateral triangle has 5 distinct lines --/
theorem isosceles_non_equilateral_distinct_lines (t : Triangle) :
  isIsosceles t ∧ ¬isEquilateral t → countDistinctLines t = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_non_equilateral_distinct_lines_l2716_271696


namespace NUMINAMATH_CALUDE_combined_figure_area_l2716_271687

/-- Regular pentagon with side length 3 -/
structure RegularPentagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_square : side_length = 3)

/-- Combined figure of a regular pentagon and a square -/
structure CombinedFigure :=
  (pentagon : RegularPentagon)
  (square : Square)
  (shared_side : pentagon.side_length = square.side_length)

/-- Area of the combined figure -/
def area (figure : CombinedFigure) : ℝ := sorry

/-- Theorem stating the area of the combined figure -/
theorem combined_figure_area (figure : CombinedFigure) :
  area figure = Real.sqrt 81 + Real.sqrt 27 := by sorry

end NUMINAMATH_CALUDE_combined_figure_area_l2716_271687


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l2716_271635

/-- Given a point Q on the unit circle 30° counterclockwise from (1,0),
    and E as the foot of the altitude from Q to the x-axis,
    prove that sin(30°) = 1/2 -/
theorem sin_thirty_degrees (Q : ℝ × ℝ) (E : ℝ × ℝ) :
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Q.1 = Real.cos (30 * π / 180)) →  -- Q is 30° counterclockwise from (1,0)
  (Q.2 = Real.sin (30 * π / 180)) →
  (E.1 = Q.1 ∧ E.2 = 0) →  -- E is the foot of the altitude from Q to the x-axis
  Real.sin (30 * π / 180) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l2716_271635


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2716_271604

def M (a : ℤ) : Set ℤ := {a, 0}

def N : Set ℤ := {x : ℤ | x^2 - 3*x < 0}

theorem intersection_implies_a_value (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2716_271604


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2716_271630

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3, 4}
  let N : Set ℕ := {0, 1, 2, 3}
  M ∩ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2716_271630


namespace NUMINAMATH_CALUDE_function_value_proof_l2716_271667

theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x - 1)
  (h2 : f 2 = 3) :
  f 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l2716_271667


namespace NUMINAMATH_CALUDE_correct_calculation_l2716_271606

theorem correct_calculation (a : ℝ) : (-a + 3) * (-3 - a) = a^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2716_271606


namespace NUMINAMATH_CALUDE_log_problem_l2716_271679

theorem log_problem : Real.log (648 * Real.rpow 6 (1/3)) / Real.log (Real.rpow 6 (1/3)) = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2716_271679


namespace NUMINAMATH_CALUDE_sum_squared_geq_three_l2716_271612

theorem sum_squared_geq_three (a b c : ℝ) (h : a * b + b * c + a * c = 1) :
  (a + b + c)^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_geq_three_l2716_271612


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2716_271672

/-- The number of terms in the arithmetic sequence 2.5, 7.5, 12.5, ..., 57.5, 62.5 -/
def sequenceLength : ℕ := 13

/-- The first term of the sequence -/
def firstTerm : ℚ := 2.5

/-- The last term of the sequence -/
def lastTerm : ℚ := 62.5

/-- The common difference of the sequence -/
def commonDifference : ℚ := 5

theorem arithmetic_sequence_length :
  sequenceLength = (lastTerm - firstTerm) / commonDifference + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2716_271672


namespace NUMINAMATH_CALUDE_square_figure_perimeter_l2716_271627

/-- A figure composed of four identical squares with a specific arrangement -/
structure SquareFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The number of squares in the figure -/
  num_squares : ℕ
  /-- The number of exposed sides in the figure's perimeter -/
  exposed_sides : ℕ
  /-- Assertion that the figure is composed of four squares -/
  h_four_squares : num_squares = 4
  /-- Assertion that the total area is 144 cm² -/
  h_total_area : total_area = 144
  /-- Assertion that the exposed sides count is 9 based on the specific arrangement -/
  h_exposed_sides : exposed_sides = 9
  /-- Assertion that the total area is the sum of the areas of individual squares -/
  h_area_sum : total_area = num_squares * square_side ^ 2

/-- The perimeter of the SquareFigure -/
def perimeter (f : SquareFigure) : ℝ :=
  f.exposed_sides * f.square_side

/-- Theorem stating that the perimeter of the SquareFigure is 54 cm -/
theorem square_figure_perimeter (f : SquareFigure) : perimeter f = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_figure_perimeter_l2716_271627


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2716_271693

theorem smallest_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 1) ∧ 
  (a % 5 = 3) ∧
  (∀ (b : ℕ), b > 0 ∧ b % 3 = 2 ∧ b % 4 = 1 ∧ b % 5 = 3 → a ≤ b) ∧
  (a = 53) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2716_271693


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2716_271622

/-- Given squares A, B, and C, prove that the perimeter of C is 48 -/
theorem square_perimeter_problem (A B C : ℝ) : 
  (4 * A = 16) →  -- Perimeter of A is 16
  (4 * B = 32) →  -- Perimeter of B is 32
  (C = A + B) →   -- Side length of C is sum of side lengths of A and B
  (4 * C = 48) := by  -- Perimeter of C is 48
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2716_271622


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2716_271621

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : x*y + x*z + y*z = 32) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2716_271621


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2716_271657

/-- Given two congruent squares with side length 20 that overlap to form a 20 by 30 rectangle,
    the percentage of the area of the rectangle that is shaded is 100/3%. -/
theorem shaded_area_percentage (square_side : ℝ) (rect_width rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 20 →
  rect_length = 30 →
  (((2 * square_side - rect_length) * square_side) / (rect_width * rect_length)) * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2716_271657


namespace NUMINAMATH_CALUDE_tournament_300_players_l2716_271651

/-- A single-elimination tournament with initial players and power-of-2 rounds -/
structure Tournament :=
  (initial_players : ℕ)
  (is_power_of_two : ℕ → Prop)

/-- Calculate the number of byes and total games in a tournament -/
def tournament_results (t : Tournament) : ℕ × ℕ :=
  sorry

theorem tournament_300_players 
  (t : Tournament) 
  (h1 : t.initial_players = 300) 
  (h2 : ∀ n, t.is_power_of_two n ↔ ∃ k, n = 2^k) : 
  tournament_results t = (44, 255) :=
sorry

end NUMINAMATH_CALUDE_tournament_300_players_l2716_271651


namespace NUMINAMATH_CALUDE_closest_root_is_point_four_l2716_271607

/-- Quadratic function f(x) = 3x^2 - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 6 * x + c

/-- The constant c in the quadratic function -/
def c : ℝ := 2.24  -- f(0) = 2.24, so c = 2.24

theorem closest_root_is_point_four :
  let options : List ℝ := [0.2, 0.4, 0.6, 0.8]
  ∃ (root : ℝ), f c root = 0 ∧
    ∀ (x : ℝ), x ∈ options → |x - root| ≥ |0.4 - root| :=
by sorry

end NUMINAMATH_CALUDE_closest_root_is_point_four_l2716_271607


namespace NUMINAMATH_CALUDE_system_of_equations_l2716_271665

theorem system_of_equations (x y a b : ℝ) (h1 : 4*x - 2*y = a) (h2 : 6*y - 12*x = b) (h3 : b ≠ 0) : a/b = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l2716_271665


namespace NUMINAMATH_CALUDE_line_direction_vector_l2716_271613

theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) :
  p1 = (4, -3) →
  p2 = (-1, 6) →
  ∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1)) →
  b = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2716_271613


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2716_271642

theorem nested_fraction_equality : 
  (1 : ℝ) / (2 - 1 / (2 - 1 / (2 - 1 / 3))) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2716_271642


namespace NUMINAMATH_CALUDE_nine_steps_climb_l2716_271659

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def ways_to_climb (n : ℕ) : ℕ := fibonacci (n + 1)

theorem nine_steps_climb : ways_to_climb 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_nine_steps_climb_l2716_271659


namespace NUMINAMATH_CALUDE_quadratic_completed_square_l2716_271663

theorem quadratic_completed_square (b : ℝ) (m : ℝ) :
  (∀ x, x^2 + b*x + 1/6 = (x + m)^2 + 1/12) → b = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completed_square_l2716_271663


namespace NUMINAMATH_CALUDE_max_n_is_81_l2716_271641

/-- The maximum value of n given the conditions -/
def max_n : ℕ := 81

/-- The set of numbers from 1 to 500 -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

/-- The probability of selecting a divisor of n from S -/
def prob_divisor (n : ℕ) : ℚ := (Finset.filter (· ∣ n) (Finset.range 500)).card / 500

/-- The theorem stating that 81 is the maximum value satisfying the conditions -/
theorem max_n_is_81 :
  ∀ n : ℕ, n ∈ S → prob_divisor n = 1/100 → n ≤ max_n :=
sorry

end NUMINAMATH_CALUDE_max_n_is_81_l2716_271641


namespace NUMINAMATH_CALUDE_money_division_l2716_271620

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 4400 →
  r - q = 5500 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2716_271620


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2716_271680

theorem quadratic_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2716_271680


namespace NUMINAMATH_CALUDE_prob_heads_11th_toss_l2716_271677

/-- A fair coin is a coin with equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting heads on a single toss of a fair coin -/
def prob_heads (p : ℝ) : ℝ := p

/-- The number of tosses -/
def num_tosses : ℕ := 10

/-- The number of heads observed -/
def heads_observed : ℕ := 7

/-- Theorem: The probability of getting heads on the 11th toss of a fair coin is 0.5,
    given that the coin was tossed 10 times with 7 heads as the result -/
theorem prob_heads_11th_toss (p : ℝ) (h : fair_coin p) :
  prob_heads p = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_prob_heads_11th_toss_l2716_271677


namespace NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_4_l2716_271685

theorem remainder_of_n_squared_plus_2n_plus_4 (n : ℤ) (k : ℤ) 
  (h : n = 75 * k - 1) : 
  (n^2 + 2*n + 4) % 75 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_4_l2716_271685


namespace NUMINAMATH_CALUDE_bank_savings_exceed_target_l2716_271689

/-- Geometric sequence sum function -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- Starting amount in cents -/
def initial_deposit : ℚ := 2

/-- Daily multiplication factor -/
def daily_factor : ℚ := 2

/-- Target amount in cents -/
def target_amount : ℚ := 400

theorem bank_savings_exceed_target :
  ∀ n : ℕ, n < 8 → geometric_sum initial_deposit daily_factor n < target_amount ∧
  geometric_sum initial_deposit daily_factor 8 ≥ target_amount :=
by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_target_l2716_271689


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_28_l2716_271617

theorem largest_five_digit_congruent_to_17_mod_28 : ∃ x : ℕ, 
  (x ≥ 10000 ∧ x < 100000) ∧ 
  x ≡ 17 [MOD 28] ∧
  (∀ y : ℕ, (y ≥ 10000 ∧ y < 100000) ∧ y ≡ 17 [MOD 28] → y ≤ x) ∧
  x = 99947 := by
sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_28_l2716_271617


namespace NUMINAMATH_CALUDE_range_of_m_l2716_271644

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 8 - m ∧ b = 2*m - 1

def q (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- Define the theorem
theorem range_of_m : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo (-1 : ℝ) (1/2) ∪ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2716_271644


namespace NUMINAMATH_CALUDE_shaded_area_square_triangle_l2716_271640

/-- The area of the shaded region formed by a square and an equilateral triangle -/
theorem shaded_area_square_triangle : 
  let square_side : ℝ := 12
  let triangle_base : ℝ := 12
  let square_vertex : ℝ × ℝ := (0, square_side)
  let triangle_vertex : ℝ × ℝ := (24, 6 * Real.sqrt 3)
  let intersection_x : ℝ := 24 * square_side / (-6 * Real.sqrt 3 + 12)
  let shaded_area : ℝ := (1/2) * triangle_base * (square_side - ((-6 * Real.sqrt 3 + 12) / 24) * intersection_x)
  shaded_area = 6 * (24 / (-6 * Real.sqrt 3 + 12)) := by
sorry


end NUMINAMATH_CALUDE_shaded_area_square_triangle_l2716_271640


namespace NUMINAMATH_CALUDE_tire_usage_proof_l2716_271602

/-- Represents the number of miles each tire is used when seven tires are used equally over a total distance --/
def miles_per_tire (total_miles : ℕ) : ℚ :=
  (4 * total_miles : ℚ) / 7

/-- Proves that given the conditions of the problem, each tire is used for 25,714 miles --/
theorem tire_usage_proof (total_miles : ℕ) (h1 : total_miles = 45000) :
  ⌊miles_per_tire total_miles⌋ = 25714 := by
  sorry

#eval ⌊miles_per_tire 45000⌋

end NUMINAMATH_CALUDE_tire_usage_proof_l2716_271602


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2716_271686

theorem arithmetic_operations : 
  (400 / 5 = 80) ∧ (3 * 230 = 690) := by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2716_271686


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l2716_271671

/-- Given a man's speed in still water and downstream speed, calculate his upstream speed -/
theorem mans_upstream_speed (v_still : ℝ) (v_downstream : ℝ) (h1 : v_still = 75) (h2 : v_downstream = 90) :
  v_still - (v_downstream - v_still) = 60 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_speed_l2716_271671


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2716_271601

def M : Set ℝ := {2, 4, 6, 8}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2716_271601


namespace NUMINAMATH_CALUDE_set_size_comparison_l2716_271618

/-- The size of set A for a given n -/
def size_A (n : ℕ) : ℕ := n^3 + n^5 + n^7 + n^9

/-- The size of set B for a given m -/
def size_B (m : ℕ) : ℕ := m^2 + m^4 + m^6 + m^8

/-- Theorem stating the condition for |B| ≥ |A| when n = 6 -/
theorem set_size_comparison (m : ℕ) :
  size_B m ≥ size_A 6 ↔ m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_set_size_comparison_l2716_271618


namespace NUMINAMATH_CALUDE_incorrect_division_result_l2716_271662

theorem incorrect_division_result (dividend : ℕ) :
  dividend / 36 = 32 →
  dividend / 48 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_division_result_l2716_271662


namespace NUMINAMATH_CALUDE_ages_proof_l2716_271628

/-- Represents the current age of Grant -/
def grant_age : ℕ := 25

/-- Represents the current age of the hospital -/
def hospital_age : ℕ := 40

/-- Represents the current age of the university -/
def university_age : ℕ := 30

/-- Represents the current age of the town library -/
def town_library_age : ℕ := 50

theorem ages_proof :
  (grant_age + 5 = (2 * (hospital_age + 5)) / 3) ∧
  (university_age = hospital_age - 10) ∧
  (town_library_age = university_age + 20) ∧
  (hospital_age < town_library_age) :=
by sorry

end NUMINAMATH_CALUDE_ages_proof_l2716_271628


namespace NUMINAMATH_CALUDE_half_of_number_l2716_271616

theorem half_of_number (N : ℚ) : 
  (4/15 * 5/7 * N) - (4/9 * 2/5 * N) = 24 → N/2 = 945 := by
sorry

end NUMINAMATH_CALUDE_half_of_number_l2716_271616


namespace NUMINAMATH_CALUDE_exponent_division_l2716_271668

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^4 / a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2716_271668


namespace NUMINAMATH_CALUDE_distribute_negative_three_l2716_271699

theorem distribute_negative_three (a : ℝ) : -3 * (a - 1) = 3 - 3 * a := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_three_l2716_271699


namespace NUMINAMATH_CALUDE_gray_area_is_65_l2716_271609

/-- Given two overlapping rectangles, calculates the area of the gray part -/
def gray_area (width1 length1 width2 length2 black_area : ℕ) : ℕ :=
  width2 * length2 - (width1 * length1 - black_area)

/-- Theorem stating that the area of the gray part is 65 -/
theorem gray_area_is_65 :
  gray_area 8 10 12 9 37 = 65 := by
  sorry

end NUMINAMATH_CALUDE_gray_area_is_65_l2716_271609


namespace NUMINAMATH_CALUDE_lcm_of_6_10_15_l2716_271673

theorem lcm_of_6_10_15 : Nat.lcm (Nat.lcm 6 10) 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_6_10_15_l2716_271673


namespace NUMINAMATH_CALUDE_five_and_half_hours_in_seconds_l2716_271636

/-- Converts hours to seconds -/
def hours_to_seconds (hours : ℝ) : ℝ :=
  hours * 60 * 60

/-- Theorem: 5.5 hours is equal to 19800 seconds -/
theorem five_and_half_hours_in_seconds : 
  hours_to_seconds 5.5 = 19800 := by sorry

end NUMINAMATH_CALUDE_five_and_half_hours_in_seconds_l2716_271636


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l2716_271691

-- Define the cost price and selling price
def cost_price : ℚ := 1500
def selling_price : ℚ := 1200

-- Define the loss percentage calculation
def loss_percentage (cp sp : ℚ) : ℚ := (cp - sp) / cp * 100

-- Theorem statement
theorem loss_percentage_calculation :
  loss_percentage cost_price selling_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l2716_271691


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l2716_271692

/-- Two cyclists on a circular track meet at the starting point -/
theorem cyclists_meet_time (v1 v2 C : ℝ) (h1 : v1 = 7) (h2 : v2 = 8) (h3 : C = 600) :
  C / (v1 + v2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meet_time_l2716_271692


namespace NUMINAMATH_CALUDE_absolute_value_condition_l2716_271664

theorem absolute_value_condition (x : ℝ) : |x - 1| = 1 - x → x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_condition_l2716_271664


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l2716_271608

theorem absolute_value_sum_zero (a b : ℝ) :
  |a - 3| + |b + 6| = 0 → (a + b - 2 = -5 ∧ a - b - 2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l2716_271608


namespace NUMINAMATH_CALUDE_cubic_sum_plus_eight_l2716_271697

theorem cubic_sum_plus_eight (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 8 = 978 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_plus_eight_l2716_271697


namespace NUMINAMATH_CALUDE_train_length_calculation_l2716_271670

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) → 
  time_to_pass = 44 →
  bridge_length = 140 →
  train_speed * time_to_pass - bridge_length = 410 :=
by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2716_271670


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2716_271633

theorem sqrt_equation_solution :
  ∃ x : ℝ, x = 196 ∧ Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2716_271633


namespace NUMINAMATH_CALUDE_wrench_force_calculation_l2716_271632

/-- Represents the force required to loosen a bolt with a wrench of a given length -/
structure WrenchForce where
  length : ℝ
  force : ℝ

/-- The inverse relationship between force and wrench length -/
def inverseProportion (w1 w2 : WrenchForce) : Prop :=
  w1.force * w1.length = w2.force * w2.length

theorem wrench_force_calculation 
  (w1 w2 : WrenchForce)
  (h1 : w1.length = 12)
  (h2 : w1.force = 300)
  (h3 : w2.length = 18)
  (h4 : inverseProportion w1 w2) :
  w2.force = 200 := by
  sorry

end NUMINAMATH_CALUDE_wrench_force_calculation_l2716_271632


namespace NUMINAMATH_CALUDE_magnitude_product_complex_l2716_271648

theorem magnitude_product_complex : Complex.abs ((7 - 4 * Complex.I) * (3 + 10 * Complex.I)) = Real.sqrt 7085 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_product_complex_l2716_271648


namespace NUMINAMATH_CALUDE_ratio_common_value_l2716_271661

theorem ratio_common_value (x y z : ℝ) (k : ℝ) 
  (h1 : (x + y) / z = k)
  (h2 : (x + z) / y = k)
  (h3 : (y + z) / x = k)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0) :
  k = -1 ∨ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_common_value_l2716_271661


namespace NUMINAMATH_CALUDE_negative_y_implies_m_gt_2_smallest_m_solution_l2716_271681

-- Define the equation
def equation (y m : ℝ) : Prop := 4 * y + 2 * m + 1 = 2 * y + 5

-- Define the inequality
def inequality (x m : ℝ) : Prop := x - 1 > (m * x + 1) / 2

theorem negative_y_implies_m_gt_2 :
  (∃ y, y < 0 ∧ equation y m) → m > 2 :=
sorry

theorem smallest_m_solution :
  m = 3 → (∀ x, inequality x m ↔ x < -3) :=
sorry

end NUMINAMATH_CALUDE_negative_y_implies_m_gt_2_smallest_m_solution_l2716_271681


namespace NUMINAMATH_CALUDE_farm_has_55_cows_l2716_271603

/-- Given information about husk consumption by cows on a dairy farm -/
structure DairyFarm where
  totalBags : ℕ -- Total bags of husk consumed by the group
  totalDays : ℕ -- Total days for group consumption
  singleCowDays : ℕ -- Days for one cow to consume one bag

/-- Calculate the number of cows on the farm -/
def numberOfCows (farm : DairyFarm) : ℕ :=
  farm.totalBags * farm.singleCowDays / farm.totalDays

/-- Theorem stating that the number of cows is 55 under given conditions -/
theorem farm_has_55_cows (farm : DairyFarm)
  (h1 : farm.totalBags = 55)
  (h2 : farm.totalDays = 55)
  (h3 : farm.singleCowDays = 55) :
  numberOfCows farm = 55 := by
  sorry

end NUMINAMATH_CALUDE_farm_has_55_cows_l2716_271603


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2716_271655

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (18 * x) = 30 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2716_271655


namespace NUMINAMATH_CALUDE_infinite_solutions_l2716_271660

/-- The equation that x, y, and z must satisfy -/
def satisfies_equation (x y z : ℕ+) : Prop :=
  (x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)

/-- The set of all positive integer solutions to the equation -/
def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {xyz | satisfies_equation xyz.1 xyz.2.1 xyz.2.2}

/-- The main theorem stating that the solution set is infinite -/
theorem infinite_solutions : Set.Infinite solution_set := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l2716_271660


namespace NUMINAMATH_CALUDE_two_distinct_roots_implies_k_values_l2716_271610

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| + 1
def g (k : ℝ) (x : ℝ) : ℝ := k * x

-- State the theorem
theorem two_distinct_roots_implies_k_values (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g k x₁ ∧ f x₂ = g k x₂) →
  (k = 1/2 ∨ k = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_implies_k_values_l2716_271610


namespace NUMINAMATH_CALUDE_children_who_got_off_bus_l2716_271669

theorem children_who_got_off_bus (initial_children : ℕ) (remaining_children : ℕ) 
  (h1 : initial_children = 43) 
  (h2 : remaining_children = 21) : 
  initial_children - remaining_children = 22 := by
  sorry

end NUMINAMATH_CALUDE_children_who_got_off_bus_l2716_271669


namespace NUMINAMATH_CALUDE_work_completion_time_l2716_271631

def work_rate (days : ℕ) : ℚ := 1 / days

def johnson_rate : ℚ := work_rate 10
def vincent_rate : ℚ := work_rate 40
def alice_rate : ℚ := work_rate 20
def bob_rate : ℚ := work_rate 30

def day1_rate : ℚ := johnson_rate + vincent_rate
def day2_rate : ℚ := alice_rate + bob_rate

def two_day_cycle_rate : ℚ := day1_rate + day2_rate

theorem work_completion_time : ∃ n : ℕ, n * two_day_cycle_rate ≥ 1 ∧ n * 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2716_271631


namespace NUMINAMATH_CALUDE_triangle_centroid_property_l2716_271688

/-- Triangle with centroid -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  h_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Distance squared between two points -/
def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of squared distances from a point to triangle vertices -/
def sum_dist_sq (t : Triangle) (M : ℝ × ℝ) : ℝ :=
  dist_sq M t.A + dist_sq M t.B + dist_sq M t.C

/-- Theorem statement -/
theorem triangle_centroid_property (t : Triangle) :
  (∀ M : ℝ × ℝ, sum_dist_sq t M ≥ sum_dist_sq t t.G) ∧
  (∀ M : ℝ × ℝ, sum_dist_sq t M = sum_dist_sq t t.G ↔ M = t.G) ∧
  (∀ k : ℝ, k > sum_dist_sq t t.G →
    ∃ r : ℝ, r = Real.sqrt ((k - sum_dist_sq t t.G) / 3) ∧
      {M : ℝ × ℝ | sum_dist_sq t M = k} = {M : ℝ × ℝ | dist_sq M t.G = r^2}) :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_property_l2716_271688


namespace NUMINAMATH_CALUDE_friends_total_points_l2716_271675

/-- The total points scored by four friends in table football games -/
def total_points (darius matt marius sofia : ℕ) : ℕ :=
  darius + matt + marius + sofia

/-- Theorem stating the total points scored by the four friends -/
theorem friends_total_points :
  ∀ (darius matt marius sofia : ℕ),
    darius = 10 →
    marius = darius + 3 →
    darius = matt - 5 →
    sofia = 2 * matt →
    total_points darius matt marius sofia = 68 := by
  sorry

#check friends_total_points

end NUMINAMATH_CALUDE_friends_total_points_l2716_271675


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2716_271637

theorem no_x_squared_term (m : ℚ) : 
  (∀ x, (x + 1) * (x^2 + 5*m*x + 3) = x^3 + (3 + 5*m)*x + 3) → m = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2716_271637


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2716_271624

theorem sum_of_decimals : 0.001 + 1.01 + 0.11 = 1.121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2716_271624


namespace NUMINAMATH_CALUDE_georges_walk_speed_l2716_271626

/-- Proves that given the conditions, George must walk at 6 mph for the last segment to arrive on time -/
theorem georges_walk_speed (total_distance : Real) (normal_speed : Real) (first_half_distance : Real) (first_half_speed : Real) :
  total_distance = 1.5 →
  normal_speed = 3 →
  first_half_distance = 0.75 →
  first_half_speed = 2 →
  (total_distance / normal_speed - first_half_distance / first_half_speed) / (total_distance - first_half_distance) = 6 := by
  sorry


end NUMINAMATH_CALUDE_georges_walk_speed_l2716_271626


namespace NUMINAMATH_CALUDE_outer_prism_width_is_ten_l2716_271650

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The inner prism dimensions satisfy the given conditions -/
def inner_prism_conditions (d : PrismDimensions) : Prop :=
  d.length * d.width * d.height = 128 ∧
  d.width = 2 * d.length ∧
  d.width = 2 * d.height

/-- The outer prism dimensions are one unit larger in each dimension -/
def outer_prism_dimensions (d : PrismDimensions) : PrismDimensions :=
  { length := d.length + 2
  , width := d.width + 2
  , height := d.height + 2 }

/-- The width of the outer prism is 10 inches -/
theorem outer_prism_width_is_ten (d : PrismDimensions) 
  (h : inner_prism_conditions d) : 
  (outer_prism_dimensions d).width = 10 := by
  sorry

end NUMINAMATH_CALUDE_outer_prism_width_is_ten_l2716_271650


namespace NUMINAMATH_CALUDE_boys_average_age_l2716_271674

theorem boys_average_age 
  (total_students : ℕ) 
  (girls_avg_age : ℝ) 
  (school_avg_age : ℝ) 
  (num_girls : ℕ) 
  (h1 : total_students = 604)
  (h2 : girls_avg_age = 11)
  (h3 : school_avg_age = 11.75)
  (h4 : num_girls = 151) :
  let num_boys : ℕ := total_students - num_girls
  let boys_total_age : ℝ := school_avg_age * total_students - girls_avg_age * num_girls
  boys_total_age / num_boys = 5411 / 453 :=
by sorry

end NUMINAMATH_CALUDE_boys_average_age_l2716_271674


namespace NUMINAMATH_CALUDE_f_properties_l2716_271605

noncomputable def f (x : ℝ) := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 2) ∧
  f (3 * Real.pi / 8) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2716_271605


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2716_271682

theorem solution_set_equivalence (m n : ℝ) 
  (h : ∀ x : ℝ, m * x + n > 0 ↔ x < 1/3) : 
  ∀ x : ℝ, n * x - m < 0 ↔ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2716_271682


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2716_271654

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50*x + 576 ≤ 16} = Set.Icc 20 28 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2716_271654


namespace NUMINAMATH_CALUDE_range_of_m_l2716_271625

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the property that ¬p is sufficient but not necessary for ¬q
def neg_p_sufficient_not_necessary_for_neg_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ∃ x, ¬(q x m) ∧ p x

-- Theorem statement
theorem range_of_m :
  ∀ m, neg_p_sufficient_not_necessary_for_neg_q m ↔ -3 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2716_271625


namespace NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l2716_271643

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle in 2D space represented by its three vertices -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Function to create a triangle from a line that intersects the coordinate axes -/
def triangleFromLine (l : Line) : Triangle := sorry

/-- Function to calculate the sum of altitudes of a triangle -/
def sumOfAltitudes (t : Triangle) : ℝ := sorry

/-- Theorem stating that for the given line, the sum of altitudes of the formed triangle
    is equal to 23 + 60/√409 -/
theorem sum_of_altitudes_for_specific_line :
  let l : Line := { a := 20, b := 3, c := 60 }
  let t : Triangle := triangleFromLine l
  sumOfAltitudes t = 23 + 60 / Real.sqrt 409 := by sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l2716_271643


namespace NUMINAMATH_CALUDE_triangle_problem_l2716_271676

open Real

theorem triangle_problem (A B C : ℝ) (s t : ℝ × ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Condition on dot products
  (B - C) * (C - A) = (C - A) * (A - B) →
  -- Definitions of vectors s and t
  s = (2 * sin C, -Real.sqrt 3) ∧
  t = (sin (2 * C), 2 * (cos (C / 2))^2 - 1) →
  -- Vectors s and t are parallel
  ∃ (k : ℝ), s.1 * t.2 = s.2 * t.1 →
  -- Given value of sin A
  sin A = 1 / 3 →
  -- Conclusion
  sin (π / 3 - B) = (2 * Real.sqrt 6 - 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2716_271676


namespace NUMINAMATH_CALUDE_cashier_bills_problem_l2716_271615

theorem cashier_bills_problem (total_bills : ℕ) (total_value : ℕ) 
  (h_total_bills : total_bills = 126)
  (h_total_value : total_value = 840) :
  ∃ (some_dollar_bills ten_dollar_bills : ℕ),
    some_dollar_bills + ten_dollar_bills = total_bills ∧
    some_dollar_bills + 10 * ten_dollar_bills = total_value ∧
    some_dollar_bills = 47 := by
  sorry

end NUMINAMATH_CALUDE_cashier_bills_problem_l2716_271615


namespace NUMINAMATH_CALUDE_vasily_salary_higher_than_fedor_l2716_271649

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  high : ℝ  -- Proportion earning 60,000 rubles
  very_high : ℝ  -- Proportion earning 80,000 rubles
  low : ℝ  -- Proportion earning 25,000 rubles (not in field)
  medium : ℝ  -- Proportion earning 40,000 rubles

/-- Calculates the expected salary given a salary distribution --/
def expected_salary (dist : SalaryDistribution) : ℝ :=
  60000 * dist.high + 80000 * dist.very_high + 25000 * dist.low + 40000 * dist.medium

/-- Calculates Fedor's salary after a given number of years --/
def fedor_salary (years : ℕ) : ℝ :=
  25000 + 3000 * years

/-- Main theorem statement --/
theorem vasily_salary_higher_than_fedor :
  let total_students : ℝ := 300
  let successful_students : ℝ := 270
  let grad_prob : ℝ := successful_students / total_students
  let salary_dist : SalaryDistribution := {
    high := 1/5,
    very_high := 1/10,
    low := 1/20,
    medium := 1 - (1/5 + 1/10 + 1/20)
  }
  let vasily_expected_salary : ℝ := 
    grad_prob * expected_salary salary_dist + (1 - grad_prob) * 25000
  let fedor_final_salary : ℝ := fedor_salary 4
  vasily_expected_salary = 45025 ∧ 
  vasily_expected_salary - fedor_final_salary = 8025 := by
  sorry


end NUMINAMATH_CALUDE_vasily_salary_higher_than_fedor_l2716_271649


namespace NUMINAMATH_CALUDE_chocolate_boxes_sold_l2716_271690

/-- The number of chocolate biscuit boxes sold by Kaylee -/
def chocolate_boxes : ℕ :=
  let total_boxes : ℕ := 33
  let lemon_boxes : ℕ := 12
  let oatmeal_boxes : ℕ := 4
  let remaining_boxes : ℕ := 12
  total_boxes - (lemon_boxes + oatmeal_boxes + remaining_boxes)

theorem chocolate_boxes_sold :
  chocolate_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_sold_l2716_271690


namespace NUMINAMATH_CALUDE_bezdikovPopulationTheorem_l2716_271639

/-- Represents the population of Bezdíkov -/
structure BezdikovPopulation where
  women1966 : ℕ
  men1966 : ℕ
  womenNow : ℕ
  menNow : ℕ

/-- Conditions for the Bezdíkov population problem -/
def bezdikovConditions (p : BezdikovPopulation) : Prop :=
  p.women1966 = p.men1966 + 30 ∧
  p.womenNow = p.women1966 / 4 ∧
  p.menNow = p.men1966 - 196 ∧
  p.womenNow = p.menNow + 10

/-- The theorem stating that the current total population of Bezdíkov is 134 -/
theorem bezdikovPopulationTheorem (p : BezdikovPopulation) 
  (h : bezdikovConditions p) : p.womenNow + p.menNow = 134 :=
by
  sorry

#check bezdikovPopulationTheorem

end NUMINAMATH_CALUDE_bezdikovPopulationTheorem_l2716_271639


namespace NUMINAMATH_CALUDE_magazine_cost_is_one_l2716_271683

/-- The cost of a magazine in dollars -/
def magazine_cost : ℝ := sorry

/-- The cost of a chocolate bar in dollars -/
def chocolate_cost : ℝ := sorry

/-- Theorem stating the cost of one magazine is $1 -/
theorem magazine_cost_is_one :
  (4 * chocolate_cost = 8 * magazine_cost) →
  (12 * chocolate_cost = 24) →
  magazine_cost = 1 := by sorry

end NUMINAMATH_CALUDE_magazine_cost_is_one_l2716_271683


namespace NUMINAMATH_CALUDE_thickness_after_four_folds_l2716_271629

def blanket_thickness (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

theorem thickness_after_four_folds :
  blanket_thickness 3 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_thickness_after_four_folds_l2716_271629


namespace NUMINAMATH_CALUDE_expression_equals_one_l2716_271666

theorem expression_equals_one : 
  (120^2 - 13^2) / (90^2 - 19^2) * ((90-19)*(90+19)) / ((120-13)*(120+13)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2716_271666


namespace NUMINAMATH_CALUDE_vector_sum_l2716_271698

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b : ℝ → ℝ × ℝ := λ x ↦ (2, x)
def c : ℝ → ℝ × ℝ := λ m ↦ (m, -3)

-- Define the parallel and perpendicular conditions
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem vector_sum (x m : ℝ) 
  (h1 : parallel a (b x)) 
  (h2 : perpendicular (b x) (c m)) : 
  x + m = -10 := by sorry

end NUMINAMATH_CALUDE_vector_sum_l2716_271698


namespace NUMINAMATH_CALUDE_kylie_total_beads_l2716_271652

-- Define the number of items made
def necklaces_monday : ℕ := 10
def necklaces_tuesday : ℕ := 2
def bracelets : ℕ := 5
def earrings : ℕ := 7

-- Define the number of beads needed for each item
def beads_per_necklace : ℕ := 20
def beads_per_bracelet : ℕ := 10
def beads_per_earring : ℕ := 5

-- Define the total number of beads used
def total_beads : ℕ := 
  (necklaces_monday + necklaces_tuesday) * beads_per_necklace +
  bracelets * beads_per_bracelet +
  earrings * beads_per_earring

-- Theorem statement
theorem kylie_total_beads : total_beads = 325 := by
  sorry

end NUMINAMATH_CALUDE_kylie_total_beads_l2716_271652


namespace NUMINAMATH_CALUDE_babblian_word_count_l2716_271656

def alphabet_size : ℕ := 6
def max_word_length : ℕ := 3

def count_words (alphabet_size : ℕ) (max_word_length : ℕ) : ℕ :=
  (alphabet_size^1 + alphabet_size^2 + alphabet_size^3)

theorem babblian_word_count :
  count_words alphabet_size max_word_length = 258 := by
  sorry

end NUMINAMATH_CALUDE_babblian_word_count_l2716_271656
