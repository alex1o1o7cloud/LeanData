import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l1266_126624

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + x^2 - 3*x - x / Real.exp x

theorem f_properties (h : ∀ x, x > 0 → f x = x * Real.log x + x^2 - 3*x - x / Real.exp x) :
  (∃ x₀ > 0, ∀ x > 0, f x ≥ f x₀ ∧ f x₀ = -2 - 1 / Real.exp 1) ∧
  (∀ x, Real.exp x ≥ x + 1) ∧
  (∀ x y, 0 < x ∧ x < y → (deriv f x) < (deriv f y)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1266_126624


namespace NUMINAMATH_CALUDE_rectangle_segment_ratio_l1266_126660

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle ABCD -/
structure Rectangle :=
  (A B C D : Point)
  (AB_length : ℝ)
  (BC_length : ℝ)

/-- Represents the ratio of segments -/
structure Ratio :=
  (r s t u : ℕ)

def is_on_segment (P Q R : Point) : Prop := sorry

def intersect (P Q R S : Point) : Point := sorry

def parallel (P Q R S : Point) : Prop := sorry

theorem rectangle_segment_ratio 
  (ABCD : Rectangle)
  (E F G : Point)
  (P Q R : Point)
  (h1 : ABCD.AB_length = 8)
  (h2 : ABCD.BC_length = 4)
  (h3 : is_on_segment ABCD.B E ABCD.C)
  (h4 : is_on_segment ABCD.B F ABCD.C)
  (h5 : is_on_segment ABCD.C G ABCD.D)
  (h6 : (ABCD.B.x - E.x) / (E.x - ABCD.C.x) = 1 / 2)
  (h7 : (ABCD.B.x - F.x) / (F.x - ABCD.C.x) = 2 / 1)
  (h8 : P = intersect ABCD.A E ABCD.B ABCD.D)
  (h9 : Q = intersect ABCD.A F ABCD.B ABCD.D)
  (h10 : R = intersect ABCD.A G ABCD.B ABCD.D)
  (h11 : parallel ABCD.A G ABCD.B ABCD.C) :
  ∃ (ratio : Ratio), 
    ratio.r = 3 ∧ 
    ratio.s = 2 ∧ 
    ratio.t = 6 ∧ 
    ratio.u = 6 ∧
    ratio.r + ratio.s + ratio.t + ratio.u = 17 := by sorry

end NUMINAMATH_CALUDE_rectangle_segment_ratio_l1266_126660


namespace NUMINAMATH_CALUDE_pairwise_product_inequality_l1266_126650

theorem pairwise_product_inequality (a b c : ℕ+) : 
  (a * b : ℕ) + (b * c : ℕ) + (a * c : ℕ) ≤ 3 * (a * b * c : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_product_inequality_l1266_126650


namespace NUMINAMATH_CALUDE_mrs_hilt_nickels_l1266_126615

/-- Represents the number of coins of each type a person has -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents given a CoinCount -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Mrs. Hilt's coin count, with unknown number of nickels -/
def mrsHilt (n : ℕ) : CoinCount :=
  { pennies := 2, nickels := n, dimes := 2 }

/-- Jacob's coin count -/
def jacob : CoinCount :=
  { pennies := 4, nickels := 1, dimes := 1 }

/-- Theorem stating that Mrs. Hilt must have 2 nickels -/
theorem mrs_hilt_nickels :
  ∃ n : ℕ, totalValue (mrsHilt n) - totalValue jacob = 13 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_nickels_l1266_126615


namespace NUMINAMATH_CALUDE_equation_solutions_l1266_126699

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6) ∧
  (∀ x : ℝ, (1/2) * (x - 1)^3 = -4 ↔ x = -1) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1266_126699


namespace NUMINAMATH_CALUDE_minesweeper_sum_invariant_l1266_126625

/-- Represents a cell in the Minesweeper grid -/
inductive Cell
| Mine : Cell
| Number (n : ℕ) : Cell

/-- A 10x10 Minesweeper grid -/
def MinesweeperGrid := Fin 10 → Fin 10 → Cell

/-- Calculates the sum of all numbers in a Minesweeper grid -/
def gridSum (grid : MinesweeperGrid) : ℕ := sorry

/-- Flips the state of all cells in a Minesweeper grid -/
def flipGrid (grid : MinesweeperGrid) : MinesweeperGrid := sorry

/-- Theorem stating that the sum of numbers remains constant after flipping the grid -/
theorem minesweeper_sum_invariant (grid : MinesweeperGrid) : 
  gridSum grid = gridSum (flipGrid grid) := by sorry

end NUMINAMATH_CALUDE_minesweeper_sum_invariant_l1266_126625


namespace NUMINAMATH_CALUDE_mushroom_collection_l1266_126610

theorem mushroom_collection 
  (N I A V : ℝ) 
  (h_non_negative : 0 ≤ N ∧ 0 ≤ I ∧ 0 ≤ A ∧ 0 ≤ V)
  (h_natasha_most : N > I ∧ N > A ∧ N > V)
  (h_ira_least : I ≤ N ∧ I ≤ A ∧ I ≤ V)
  (h_alyosha_more : A > V) : 
  N + I > A + V := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l1266_126610


namespace NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l1266_126664

theorem sunglasses_and_hats_probability
  (total_sunglasses : ℕ)
  (total_hats : ℕ)
  (prob_hat_and_sunglasses : ℚ)
  (h1 : total_sunglasses = 80)
  (h2 : total_hats = 50)
  (h3 : prob_hat_and_sunglasses = 3/5) :
  (prob_hat_and_sunglasses * total_hats) / total_sunglasses = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l1266_126664


namespace NUMINAMATH_CALUDE_oliver_learning_time_l1266_126601

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The total number of days Oliver needs to learn all vowels -/
def total_days : ℕ := 25

/-- The number of days Oliver needs to learn one vowel -/
def days_per_vowel : ℕ := total_days / num_vowels

theorem oliver_learning_time : days_per_vowel = 5 := by
  sorry

end NUMINAMATH_CALUDE_oliver_learning_time_l1266_126601


namespace NUMINAMATH_CALUDE_scientific_notation_2023_l1266_126674

/-- Scientific notation representation with a specified number of significant figures -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  sigFigs : ℕ

/-- Convert a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Check if a ScientificNotation representation is valid -/
def isValidScientificNotation (sn : ScientificNotation) : Prop :=
  1 ≤ sn.coefficient ∧ sn.coefficient < 10 ∧ sn.sigFigs > 0

theorem scientific_notation_2023 :
  let sn := toScientificNotation 2023 2
  isValidScientificNotation sn ∧ sn.coefficient = 2.0 ∧ sn.exponent = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_2023_l1266_126674


namespace NUMINAMATH_CALUDE_power_tower_comparison_l1266_126644

theorem power_tower_comparison : 3^(3^(3^3)) > 2^(2^(2^(2^2))) := by
  sorry

end NUMINAMATH_CALUDE_power_tower_comparison_l1266_126644


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1266_126682

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_composition_of_three : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1266_126682


namespace NUMINAMATH_CALUDE_expression_value_l1266_126636

theorem expression_value (a : ℝ) (h : a^2 + 2*a + 2 - Real.sqrt 3 = 0) :
  1 / (a + 1) - (a + 3) / (a^2 - 1) * (a^2 - 2*a + 1) / (a^2 + 4*a + 3) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1266_126636


namespace NUMINAMATH_CALUDE_intersection_implies_sum_zero_l1266_126600

theorem intersection_implies_sum_zero (α β : ℝ) :
  (∃ x₀ : ℝ, x₀ / (Real.sin α + Real.sin β) + (-x₀) / (Real.sin α + Real.cos β) = 1 ∧
              x₀ / (Real.cos α + Real.sin β) + (-x₀) / (Real.cos α + Real.cos β) = 1) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_zero_l1266_126600


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1266_126639

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 = 5 ∧
  a 3 + a 5 = 2

/-- The common difference of an arithmetic sequence with given conditions is -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1266_126639


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l1266_126646

-- Define the given conditions
def north_students : ℕ := 1800
def north_tennis_percentage : ℚ := 25 / 100
def south_students : ℕ := 2700
def south_tennis_percentage : ℚ := 35 / 100

-- Define the theorem
theorem combined_tennis_percentage :
  let north_tennis := (north_students : ℚ) * north_tennis_percentage
  let south_tennis := (south_students : ℚ) * south_tennis_percentage
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_students + south_students : ℚ)
  (total_tennis / total_students) * 100 = 31 := by
  sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l1266_126646


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1266_126655

/-- Given a point M on the line 2x + y - 1 = 0 and points (3,0) and (0,1) on a circle centered at M,
    prove that the equation of this circle is (x-1)² + (y+1)² = 5 -/
theorem circle_equation_proof (M : ℝ × ℝ) :
  (2 * M.1 + M.2 - 1 = 0) →
  ((3 : ℝ) - M.1)^2 + (0 - M.2)^2 = ((0 : ℝ) - M.1)^2 + (1 - M.2)^2 →
  ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ↔ (x - M.1)^2 + (y - M.2)^2 = ((3 : ℝ) - M.1)^2 + (0 - M.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1266_126655


namespace NUMINAMATH_CALUDE_initial_amount_of_A_l1266_126611

/-- Represents the money exchange problem with three people --/
structure MoneyExchange where
  a : ℕ  -- Initial amount of A
  b : ℕ  -- Initial amount of B
  c : ℕ  -- Initial amount of C

/-- Predicate that checks if the money exchange satisfies the problem conditions --/
def satisfies_conditions (m : MoneyExchange) : Prop :=
  -- After all exchanges, everyone has 16 dollars
  4 * (m.a - m.b - m.c) = 16 ∧
  6 * m.b - 2 * m.a - 2 * m.c = 16 ∧
  7 * m.c - m.a - m.b = 16

/-- Theorem stating that if the conditions are satisfied, A's initial amount was 29 --/
theorem initial_amount_of_A (m : MoneyExchange) :
  satisfies_conditions m → m.a = 29 := by
  sorry


end NUMINAMATH_CALUDE_initial_amount_of_A_l1266_126611


namespace NUMINAMATH_CALUDE_numbers_satisfying_conditions_l1266_126647

def satisfies_conditions (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 1000 ∧
  ∃ k : ℕ, n = 7 * k ∧
  ((∃ m : ℕ, n = 4 * m + 3) ∨ (∃ m : ℕ, n = 9 * m + 3))

theorem numbers_satisfying_conditions :
  {n : ℕ | satisfies_conditions n} = {147, 399, 651, 903} := by
  sorry

end NUMINAMATH_CALUDE_numbers_satisfying_conditions_l1266_126647


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l1266_126638

theorem chicken_rabbit_problem (c r : ℕ) : 
  c = r - 20 → 
  4 * r = 3 * (2 * c) + 10 → 
  c = 35 :=
by sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l1266_126638


namespace NUMINAMATH_CALUDE_four_x_plus_g_is_odd_l1266_126670

theorem four_x_plus_g_is_odd (x g : ℤ) (h : 2 * x - g = 11) : 
  ∃ k : ℤ, 4 * x + g = 2 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_four_x_plus_g_is_odd_l1266_126670


namespace NUMINAMATH_CALUDE_max_slope_product_l1266_126609

theorem max_slope_product (m₁ m₂ : ℝ) : 
  m₁ ≠ 0 → m₂ ≠ 0 →                           -- non-horizontal, non-vertical lines
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 →         -- 45° angle intersection
  m₂ = 6 * m₁ →                               -- one slope is 6 times the other
  ∃ (p : ℝ), p = m₁ * m₂ ∧ p ≤ (3/2 : ℝ) ∧ 
  ∀ (q : ℝ), (∃ (n₁ n₂ : ℝ), n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ 
              |((n₂ - n₁) / (1 + n₁ * n₂))| = 1 ∧ 
              n₂ = 6 * n₁ ∧ q = n₁ * n₂) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_max_slope_product_l1266_126609


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1266_126678

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  Polynomial.degree q = 9 →
  r = 5 * X^4 + 6 * X^3 - 2 * X + 7 →
  f = d * q + r →
  Polynomial.degree d = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1266_126678


namespace NUMINAMATH_CALUDE_squared_roots_polynomial_l1266_126656

theorem squared_roots_polynomial (x : ℝ) : 
  let f (x : ℝ) := x^3 + x^2 - 2*x - 1
  let g (x : ℝ) := x^3 - 5*x^2 + 6*x - 1
  ∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2 :=
by sorry

end NUMINAMATH_CALUDE_squared_roots_polynomial_l1266_126656


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_numerator_denominator_l1266_126658

def repeating_decimal : ℚ := 345 / 999

theorem repeating_decimal_fraction :
  repeating_decimal = 115 / 111 :=
sorry

theorem sum_numerator_denominator :
  115 + 111 = 226 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_numerator_denominator_l1266_126658


namespace NUMINAMATH_CALUDE_temperature_drop_l1266_126621

theorem temperature_drop (initial_temp final_temp drop : ℤ) :
  initial_temp = -6 ∧ drop = 5 → final_temp = initial_temp - drop → final_temp = -11 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_l1266_126621


namespace NUMINAMATH_CALUDE_score_difference_l1266_126628

theorem score_difference (chuck_team_score red_team_score : ℕ) 
  (h1 : chuck_team_score = 95) 
  (h2 : red_team_score = 76) : 
  chuck_team_score - red_team_score = 19 := by
sorry

end NUMINAMATH_CALUDE_score_difference_l1266_126628


namespace NUMINAMATH_CALUDE_count_integer_points_l1266_126683

def point_A : ℤ × ℤ := (2, 3)
def point_B : ℤ × ℤ := (150, 903)

def is_between (p q r : ℤ × ℤ) : Prop :=
  (p.1 < q.1 ∧ q.1 < r.1) ∨ (r.1 < q.1 ∧ q.1 < p.1)

def on_line (p q r : ℤ × ℤ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

def integer_points_between : Prop :=
  ∃ (S : Finset (ℤ × ℤ)),
    S.card = 4 ∧
    (∀ p ∈ S, is_between point_A p point_B ∧ on_line point_A point_B p) ∧
    (∀ p : ℤ × ℤ, is_between point_A p point_B ∧ on_line point_A point_B p → p ∈ S)

theorem count_integer_points : integer_points_between := by
  sorry

end NUMINAMATH_CALUDE_count_integer_points_l1266_126683


namespace NUMINAMATH_CALUDE_equal_perimeters_if_base_equals_height_base_equals_height_if_equal_perimeters_l1266_126618

/-- Represents a triangle ABC with inscribed rectangles --/
structure TriangleWithRectangles where
  -- The length of the base AB
  base : ℝ
  -- The height of the triangle corresponding to base AB
  height : ℝ
  -- A function that given a real number between 0 and 1, returns the dimensions of the inscribed rectangle
  rectangleDimensions : ℝ → (ℝ × ℝ)

/-- The perimeter of a rectangle given its dimensions --/
def rectanglePerimeter (dimensions : ℝ × ℝ) : ℝ :=
  2 * (dimensions.1 + dimensions.2)

/-- Theorem: If base equals height, then all inscribed rectangles have equal perimeters --/
theorem equal_perimeters_if_base_equals_height (triangle : TriangleWithRectangles) :
  triangle.base = triangle.height →
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
  rectanglePerimeter (triangle.rectangleDimensions x) = rectanglePerimeter (triangle.rectangleDimensions y) :=
sorry

/-- Theorem: If all inscribed rectangles have equal perimeters, then base equals height --/
theorem base_equals_height_if_equal_perimeters (triangle : TriangleWithRectangles) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
   rectanglePerimeter (triangle.rectangleDimensions x) = rectanglePerimeter (triangle.rectangleDimensions y)) →
  triangle.base = triangle.height :=
sorry

end NUMINAMATH_CALUDE_equal_perimeters_if_base_equals_height_base_equals_height_if_equal_perimeters_l1266_126618


namespace NUMINAMATH_CALUDE_H_triple_2_l1266_126686

/-- The function H defined as H(x) = 2x - 1 for all real x -/
def H (x : ℝ) : ℝ := 2 * x - 1

/-- Theorem stating that H(H(H(2))) = 9 -/
theorem H_triple_2 : H (H (H 2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_H_triple_2_l1266_126686


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1266_126666

/-- Given a cubic polynomial g(x) = cx³ + 5x² + dx + 7, prove that if g(2) = 19 and g(-1) = -9, 
    then c = -25/3 and d = 88/3 -/
theorem polynomial_remainder (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 + 5 * x^2 + d * x + 7
  (g 2 = 19 ∧ g (-1) = -9) → c = -25/3 ∧ d = 88/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1266_126666


namespace NUMINAMATH_CALUDE_geometric_progression_cubic_roots_l1266_126662

theorem geometric_progression_cubic_roots (x y z r p q : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  y^2 = r * x^2 →
  z^2 = r * y^2 →
  x^3 - p*x^2 + q*x - r = 0 →
  y^3 - p*y^2 + q*y - r = 0 →
  z^3 - p*z^2 + q*z - r = 0 →
  r^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_cubic_roots_l1266_126662


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l1266_126672

theorem percentage_of_red_non_honda_cars
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (honda_red_ratio : ℚ)
  (total_red_ratio : ℚ)
  (h1 : total_cars = 9000)
  (h2 : honda_cars = 5000)
  (h3 : honda_red_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100)
  : (((total_red_ratio * total_cars) - (honda_red_ratio * honda_cars)) /
     (total_cars - honda_cars)) = 225 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l1266_126672


namespace NUMINAMATH_CALUDE_interest_calculation_years_l1266_126643

/-- Proves that the number of years for which the interest is calculated on the first part is 8 --/
theorem interest_calculation_years (total_sum : ℚ) (second_part : ℚ) 
  (interest_rate_first : ℚ) (interest_rate_second : ℚ) (time_second : ℚ) :
  total_sum = 2795 →
  second_part = 1720 →
  interest_rate_first = 3 / 100 →
  interest_rate_second = 5 / 100 →
  time_second = 3 →
  let first_part := total_sum - second_part
  let interest_second := second_part * interest_rate_second * time_second
  ∃ (time_first : ℚ), first_part * interest_rate_first * time_first = interest_second ∧ time_first = 8 :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_years_l1266_126643


namespace NUMINAMATH_CALUDE_smallest_distance_to_target_l1266_126669

def jump_distance_1 : ℕ := 364
def jump_distance_2 : ℕ := 715
def target_point : ℕ := 2010

theorem smallest_distance_to_target : 
  ∃ (x y : ℤ), 
    (∀ (a b : ℤ), |target_point - (jump_distance_1 * a + jump_distance_2 * b)| ≥ 
                   |target_point - (jump_distance_1 * x + jump_distance_2 * y)|) ∧
    |target_point - (jump_distance_1 * x + jump_distance_2 * y)| = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_to_target_l1266_126669


namespace NUMINAMATH_CALUDE_integral_equals_four_l1266_126635

theorem integral_equals_four : ∫ x in (1:ℝ)..2, (3*x^2 - 2*x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_four_l1266_126635


namespace NUMINAMATH_CALUDE_y_value_l1266_126606

theorem y_value (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 8) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1266_126606


namespace NUMINAMATH_CALUDE_circle_equation_standard_form_tangent_line_b_value_l1266_126612

open Real

/-- A line ax + by = c is tangent to a circle (x - h)^2 + (y - k)^2 = r^2 if and only if
    the distance from the center (h, k) to the line is equal to the radius r. -/
def is_tangent_line_to_circle (a b c h k r : ℝ) : Prop :=
  (|a * h + b * k - c| / sqrt (a^2 + b^2)) = r

/-- The equation of the circle x^2 + y^2 - 2x - 2y + 1 = 0 in standard form is (x - 1)^2 + (y - 1)^2 = 1 -/
theorem circle_equation_standard_form :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 2*y + 1 = 0 ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

/-- The main theorem: If the line 3x + 4y = b is tangent to the circle x^2 + y^2 - 2x - 2y + 1 = 0,
    then b = 2 or b = 12 -/
theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, is_tangent_line_to_circle 3 4 b 1 1 1) → (b = 2 ∨ b = 12) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_standard_form_tangent_line_b_value_l1266_126612


namespace NUMINAMATH_CALUDE_line_through_point_l1266_126681

/-- Given a line with equation 3kx - 2 = 4y passing through the point (-1/2, -5),
    prove that k = 12. -/
theorem line_through_point (k : ℝ) : 
  (3 * k * (-1/2) - 2 = 4 * (-5)) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1266_126681


namespace NUMINAMATH_CALUDE_x_value_proof_l1266_126602

theorem x_value_proof (x : ℝ) : (5 * x - 3)^3 = Real.sqrt 64 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1266_126602


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l1266_126631

theorem gcd_of_powers_of_two : Nat.gcd (2^1010 - 1) (2^1000 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l1266_126631


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1266_126640

/-- The quadratic function f(x) = -2(x+1)^2 + k -/
def f (k : ℝ) (x : ℝ) : ℝ := -2 * (x + 1)^2 + k

/-- Theorem stating the inequality between f(2), f(-3), and f(-0.5) -/
theorem quadratic_inequality (k : ℝ) : f k 2 < f k (-3) ∧ f k (-3) < f k (-0.5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1266_126640


namespace NUMINAMATH_CALUDE_smallest_winning_number_l1266_126641

theorem smallest_winning_number :
  ∃ N : ℕ,
    N ≤ 499 ∧
    (∀ m : ℕ, m < N →
      (3 * m < 500 ∧
       3 * m + 25 < 500 ∧
       3 * (3 * m + 25) < 500 ∧
       3 * (3 * m + 25) + 25 ≥ 500 →
       False)) ∧
    3 * N < 500 ∧
    3 * N + 25 < 500 ∧
    3 * (3 * N + 25) < 500 ∧
    3 * (3 * N + 25) + 25 ≥ 500 ∧
    N = 45 :=
by sorry

#eval (45 / 10 + 45 % 10) -- Sum of digits of 45

end NUMINAMATH_CALUDE_smallest_winning_number_l1266_126641


namespace NUMINAMATH_CALUDE_x_value_l1266_126671

theorem x_value (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1266_126671


namespace NUMINAMATH_CALUDE_unique_polynomial_coefficients_l1266_126649

theorem unique_polynomial_coefficients :
  ∃! (a b c : ℕ+),
  let x : ℝ := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)
  x^100 = 3*x^98 + 15*x^96 + 12*x^94 - x^50 + (a:ℝ)*x^46 + (b:ℝ)*x^44 + (c:ℝ)*x^40 ∧
  a + b + c = 5824 := by
sorry

end NUMINAMATH_CALUDE_unique_polynomial_coefficients_l1266_126649


namespace NUMINAMATH_CALUDE_translation_property_l1266_126627

-- Define a translation as a function from ℂ to ℂ
def Translation := ℂ → ℂ

-- Define the property of a translation taking one point to another
def TranslatesTo (T : Translation) (z w : ℂ) : Prop := T z = w

theorem translation_property (T : Translation) :
  TranslatesTo T (1 - 2*I) (4 + 3*I) →
  TranslatesTo T (2 + 4*I) (5 + 9*I) := by
  sorry

end NUMINAMATH_CALUDE_translation_property_l1266_126627


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_to_7_l1266_126690

theorem smallest_multiple_of_3_to_7 : 
  ∃ (N : ℕ), N > 0 ∧ 
    (∀ (k : ℕ), k > 0 ∧ k < N → 
      ¬(3 ∣ k ∧ 4 ∣ k ∧ 5 ∣ k ∧ 6 ∣ k ∧ 7 ∣ k)) ∧
    (3 ∣ N ∧ 4 ∣ N ∧ 5 ∣ N ∧ 6 ∣ N ∧ 7 ∣ N) ∧
    N = 420 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_to_7_l1266_126690


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l1266_126663

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the center of a circle
def is_center (h k : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ r, ∀ x y, C x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Define tangency between a line and a circle
def is_tangent (l : ℝ → ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∃! p, (∃ x y, l x y m ∧ C x y ∧ p = (x, y))

-- Theorem statement
theorem circle_and_line_properties :
  (is_center 0 1 circle_C) ∧
  (∀ m, is_tangent line_l circle_C m ↔ (m = 3 ∨ m = -1)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l1266_126663


namespace NUMINAMATH_CALUDE_decimal_place_of_13_over_17_l1266_126687

/-- The decimal representation of 13/17 repeats every 17 digits -/
def decimal_period : ℕ := 17

/-- The repeating sequence of digits in the decimal representation of 13/17 -/
def repeating_sequence : List ℕ := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7]

/-- The position we're interested in -/
def target_position : ℕ := 250

theorem decimal_place_of_13_over_17 :
  (repeating_sequence.get! ((target_position - 1) % decimal_period)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_of_13_over_17_l1266_126687


namespace NUMINAMATH_CALUDE_squares_after_seven_dwarfs_l1266_126605

/-- Represents the process of a dwarf cutting a square --/
def dwarf_cut (n : ℕ) : ℕ := n + 3

/-- Calculates the number of squares after n dwarfs have performed their cuts --/
def squares_after_cuts (n : ℕ) : ℕ := 
  Nat.iterate dwarf_cut n 1

/-- The theorem stating that after 7 dwarfs, there are 22 squares --/
theorem squares_after_seven_dwarfs : 
  squares_after_cuts 7 = 22 := by sorry

end NUMINAMATH_CALUDE_squares_after_seven_dwarfs_l1266_126605


namespace NUMINAMATH_CALUDE_rectangular_to_polar_sqrt2_l1266_126673

theorem rectangular_to_polar_sqrt2 :
  ∃ (r : ℝ) (θ : ℝ), 
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = Real.sqrt 2 ∧
    r * Real.sin θ = -Real.sqrt 2 ∧
    r = 2 ∧
    θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_sqrt2_l1266_126673


namespace NUMINAMATH_CALUDE_area_of_enclosed_region_l1266_126645

/-- The curve defined by |x-1| + |y-1| = 1 -/
def curve (x y : ℝ) : Prop := |x - 1| + |y - 1| = 1

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2}

/-- The area of the region enclosed by the curve is 2 -/
theorem area_of_enclosed_region : MeasureTheory.volume enclosed_region = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_enclosed_region_l1266_126645


namespace NUMINAMATH_CALUDE_profit_increase_l1266_126633

theorem profit_increase (cost_price selling_price : ℝ) (a : ℝ) 
  (h1 : cost_price > 0)
  (h2 : selling_price > cost_price)
  (h3 : (selling_price - cost_price) / cost_price = a / 100)
  (h4 : (selling_price - cost_price * 0.95) / (cost_price * 0.95) = (a + 15) / 100) :
  a = 185 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l1266_126633


namespace NUMINAMATH_CALUDE_gcd_5800_14025_l1266_126604

theorem gcd_5800_14025 : Nat.gcd 5800 14025 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5800_14025_l1266_126604


namespace NUMINAMATH_CALUDE_min_shots_theorem_l1266_126667

/-- The hit rate for each shot -/
def hit_rate : ℝ := 0.25

/-- The desired probability of hitting the target at least once -/
def desired_probability : ℝ := 0.75

/-- The probability of hitting the target at least once given n shots -/
def prob_hit_at_least_once (n : ℕ) : ℝ := 1 - (1 - hit_rate) ^ n

/-- The minimum number of shots required to achieve the desired probability -/
def min_shots : ℕ := 5

theorem min_shots_theorem :
  (∀ k < min_shots, prob_hit_at_least_once k < desired_probability) ∧
  prob_hit_at_least_once min_shots ≥ desired_probability :=
by sorry

end NUMINAMATH_CALUDE_min_shots_theorem_l1266_126667


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1266_126696

/-- The lateral surface area of a cone with base radius 1 and height √3 is 2π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let A : ℝ := π * r * l
  A = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1266_126696


namespace NUMINAMATH_CALUDE_sum_assigned_values_zero_l1266_126698

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

/-- Assigns 1 or -1 to a number based on its prime factorization -/
def assignValue (n : ℕ) : Int := sorry

/-- The sum of assigned values for all divisors of a number -/
def sumAssignedValues (n : ℕ) : Int := sorry

/-- Theorem: The sum of assigned values for divisors of the product of first k primes is 0 -/
theorem sum_assigned_values_zero (k : ℕ) : sumAssignedValues (primeProduct k) = 0 := by sorry

end NUMINAMATH_CALUDE_sum_assigned_values_zero_l1266_126698


namespace NUMINAMATH_CALUDE_infinitely_many_rational_solutions_l1266_126665

theorem infinitely_many_rational_solutions :
  ∃ f : ℕ → ℚ × ℚ,
    (∀ n : ℕ, (f n).1^3 + (f n).2^3 = 9) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_rational_solutions_l1266_126665


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1266_126677

/-- A line y = mx + (2m+1) always passes through the point (-2, 1) for any real m -/
theorem line_passes_through_fixed_point (m : ℝ) : 
  let f : ℝ → ℝ := fun x => m * x + (2 * m + 1)
  f (-2) = 1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1266_126677


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l1266_126668

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The transformation of the random variable -/
def eta (X : BinomialRV) : ℝ → ℝ := fun x ↦ -2 * x + 1

theorem variance_of_transformed_binomial (X : BinomialRV) 
  (h_n : X.n = 6) (h_p : X.p = 0.4) : 
  variance X * 4 = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l1266_126668


namespace NUMINAMATH_CALUDE_quadratic_inequality_1_l1266_126617

theorem quadratic_inequality_1 (x : ℝ) :
  x^2 - 7*x + 12 > 0 ↔ x < 3 ∨ x > 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_1_l1266_126617


namespace NUMINAMATH_CALUDE_biology_class_boys_l1266_126695

theorem biology_class_boys (girls_to_boys_ratio : ℚ) (physics_students : ℕ) (biology_to_physics_ratio : ℚ) :
  girls_to_boys_ratio = 3 →
  physics_students = 200 →
  biology_to_physics_ratio = 1/2 →
  (physics_students : ℚ) * biology_to_physics_ratio / (1 + girls_to_boys_ratio) = 25 :=
by sorry

end NUMINAMATH_CALUDE_biology_class_boys_l1266_126695


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1266_126623

theorem consecutive_even_integers_sum (n : ℤ) : 
  (n % 2 = 0) ∧ (n * (n + 2) * (n + 4) = 480) → n + (n + 2) + (n + 4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1266_126623


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1266_126688

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1266_126688


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1266_126659

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ), x = 6 ∧ y = 2 * Real.sqrt 3 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 4 * Real.sqrt 3 ∧ θ = Real.pi / 6 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1266_126659


namespace NUMINAMATH_CALUDE_added_classes_l1266_126632

theorem added_classes (initial_classes : ℕ) (students_per_class : ℕ) (new_total_students : ℕ)
  (h1 : initial_classes = 15)
  (h2 : students_per_class = 20)
  (h3 : new_total_students = 400) :
  (new_total_students - initial_classes * students_per_class) / students_per_class = 5 := by
sorry

end NUMINAMATH_CALUDE_added_classes_l1266_126632


namespace NUMINAMATH_CALUDE_triangle_coloring_theorem_l1266_126629

-- Define the set of colors
inductive Color
| Blue
| Red
| Yellow

-- Define a point with a color
structure Point where
  color : Color

-- Define a triangle
structure Triangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point

-- Define the main theorem
theorem triangle_coloring_theorem 
  (K P S : Point)
  (A B C D E F : Point)
  (h1 : K.color = Color.Blue)
  (h2 : P.color = Color.Red)
  (h3 : S.color = Color.Yellow)
  (h4 : (A.color = K.color ∨ A.color = S.color) ∧
        (B.color = K.color ∨ B.color = S.color) ∧
        (C.color = K.color ∨ C.color = P.color) ∧
        (D.color = P.color ∨ D.color = S.color) ∧
        (E.color = P.color ∨ E.color = S.color) ∧
        (F.color = K.color ∨ F.color = P.color)) :
  ∃ (t : Triangle), t.vertex1.color ≠ t.vertex2.color ∧ 
                    t.vertex2.color ≠ t.vertex3.color ∧ 
                    t.vertex3.color ≠ t.vertex1.color :=
by sorry

end NUMINAMATH_CALUDE_triangle_coloring_theorem_l1266_126629


namespace NUMINAMATH_CALUDE_sum_mod_nine_l1266_126676

theorem sum_mod_nine :
  (2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l1266_126676


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_24_and_7_l1266_126675

theorem pythagorean_triple_with_24_and_7 : 
  ∃ (x : ℕ), x > 0 ∧ x^2 + 7^2 = 24^2 ∨ x^2 = 24^2 + 7^2 → x = 25 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_24_and_7_l1266_126675


namespace NUMINAMATH_CALUDE_inequality_proof_l1266_126626

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1266_126626


namespace NUMINAMATH_CALUDE_factors_of_23232_l1266_126614

theorem factors_of_23232 : Nat.card (Nat.divisors 23232) = 42 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_23232_l1266_126614


namespace NUMINAMATH_CALUDE_set_relationship_l1266_126648

theorem set_relationship (A B C : Set α) (hAnonempty : A.Nonempty) (hBnonempty : B.Nonempty) (hCnonempty : C.Nonempty)
  (hUnion : A ∪ B = C) (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ ¬(∀ x, x ∈ C → x ∈ A) := by
sorry

end NUMINAMATH_CALUDE_set_relationship_l1266_126648


namespace NUMINAMATH_CALUDE_arun_weight_average_l1266_126684

-- Define Arun's weight as a real number
def arun_weight : ℝ := sorry

-- Define the conditions on Arun's weight
def condition1 : Prop := 61 < arun_weight ∧ arun_weight < 72
def condition2 : Prop := 60 < arun_weight ∧ arun_weight < 70
def condition3 : Prop := arun_weight ≤ 64
def condition4 : Prop := 62 < arun_weight ∧ arun_weight < 73
def condition5 : Prop := 59 < arun_weight ∧ arun_weight < 68

-- Theorem stating that the average of possible weights is 63.5
theorem arun_weight_average :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  (63 + 64) / 2 = 63.5 :=
by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l1266_126684


namespace NUMINAMATH_CALUDE_inscribed_equilateral_triangle_side_length_l1266_126622

theorem inscribed_equilateral_triangle_side_length 
  (diameter : ℝ) (side_length : ℝ) 
  (h1 : diameter = 2000) 
  (h2 : side_length = 1732 + 1/20) : 
  side_length = diameter / 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_equilateral_triangle_side_length_l1266_126622


namespace NUMINAMATH_CALUDE_max_sum_of_squares_exists_max_sum_of_squares_l1266_126608

/-- Given a quadruple (a, b, c, d) satisfying certain conditions, 
    the sum of their squares is at most 254. -/
theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 47 →
  a * d + b * c = 88 →
  c * d = 54 →
  a^2 + b^2 + c^2 + d^2 ≤ 254 := by
  sorry

/-- There exists a quadruple (a, b, c, d) satisfying the conditions 
    where the sum of their squares equals 254. -/
theorem exists_max_sum_of_squares : 
  ∃ (a b c d : ℝ), 
    a + b = 12 ∧
    a * b + c + d = 47 ∧
    a * d + b * c = 88 ∧
    c * d = 54 ∧
    a^2 + b^2 + c^2 + d^2 = 254 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_exists_max_sum_of_squares_l1266_126608


namespace NUMINAMATH_CALUDE_cos_90_degrees_is_zero_l1266_126693

theorem cos_90_degrees_is_zero :
  let cos_36 : ℝ := (1 + Real.sqrt 5) / 4
  let cos_54 : ℝ := (1 - Real.sqrt 5) / 4
  let sin_36 : ℝ := Real.sqrt (10 - 2 * Real.sqrt 5) / 4
  let sin_54 : ℝ := Real.sqrt (10 + 2 * Real.sqrt 5) / 4
  let cos_sum := cos_36 * cos_54 - sin_36 * sin_54
  cos_sum = 0 := by sorry

end NUMINAMATH_CALUDE_cos_90_degrees_is_zero_l1266_126693


namespace NUMINAMATH_CALUDE_leftover_value_is_650_l1266_126697

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a person's coin collection --/
structure CoinCollection where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of coins in dollars --/
def coinValue (quarters dimes : Nat) : Rat :=
  (quarters * 25 + dimes * 10) / 100

/-- Calculates the number and value of leftover coins --/
def leftoverCoins (emily jack : CoinCollection) (roll : RollSize) : Rat :=
  let totalQuarters := emily.quarters + jack.quarters
  let totalDimes := emily.dimes + jack.dimes
  let leftoverQuarters := totalQuarters % roll.quarters
  let leftoverDimes := totalDimes % roll.dimes
  coinValue leftoverQuarters leftoverDimes

/-- The main theorem --/
theorem leftover_value_is_650 :
  let roll : RollSize := { quarters := 45, dimes := 60 }
  let emily : CoinCollection := { quarters := 105, dimes := 215 }
  let jack : CoinCollection := { quarters := 140, dimes := 340 }
  leftoverCoins emily jack roll = 13/2 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_650_l1266_126697


namespace NUMINAMATH_CALUDE_binomial_10_0_l1266_126694

theorem binomial_10_0 : (10 : ℕ).choose 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_0_l1266_126694


namespace NUMINAMATH_CALUDE_min_value_n_plus_32_over_n_squared_l1266_126603

theorem min_value_n_plus_32_over_n_squared (n : ℝ) (h : n > 0) :
  n + 32 / n^2 ≥ 6 ∧ ∃ n₀ > 0, n₀ + 32 / n₀^2 = 6 := by sorry

end NUMINAMATH_CALUDE_min_value_n_plus_32_over_n_squared_l1266_126603


namespace NUMINAMATH_CALUDE_gcd_problem_l1266_126692

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 97 * (2 * k)) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1266_126692


namespace NUMINAMATH_CALUDE_sector_area_l1266_126630

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 6) (h2 : central_angle = 2) :
  let radius := arc_length / central_angle
  (1/2) * radius^2 * central_angle = 9 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l1266_126630


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l1266_126653

theorem quadratic_root_existence (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (x₃ ∈ Set.Icc x₁ x₂ ∨ x₃ ∈ Set.Icc x₂ x₁) ∧ 
        (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l1266_126653


namespace NUMINAMATH_CALUDE_flowchart_output_l1266_126652

def iterate_add_two (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => iterate_add_two (x + 2) n

theorem flowchart_output : iterate_add_two 10 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_flowchart_output_l1266_126652


namespace NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l1266_126651

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)
  (is_triangle : A + B + C = Real.pi)

-- State the theorem
theorem triangle_angle_A_is_30_degrees
  (abc : Triangle)
  (h1 : abc.a^2 - abc.b^2 = Real.sqrt 3 * abc.b * abc.c)
  (h2 : Real.sin abc.C = 2 * Real.sqrt 3 * Real.sin abc.B) :
  abc.A = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l1266_126651


namespace NUMINAMATH_CALUDE_selection_methods_eq_twelve_l1266_126679

/-- Represents the number of teachers available for selection -/
def total_teachers : ℕ := 4

/-- Represents the number of teachers to be selected -/
def selected_teachers : ℕ := 3

/-- Represents the number of phases in the training -/
def training_phases : ℕ := 3

/-- Represents the number of teachers who cannot participate in the first phase -/
def restricted_teachers : ℕ := 2

/-- Calculates the number of different selection methods -/
def selection_methods : ℕ := sorry

/-- Theorem stating that the number of selection methods is 12 -/
theorem selection_methods_eq_twelve : selection_methods = 12 := by sorry

end NUMINAMATH_CALUDE_selection_methods_eq_twelve_l1266_126679


namespace NUMINAMATH_CALUDE_determinant_of_special_matrix_l1266_126619

open Matrix Real

theorem determinant_of_special_matrix (α β : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, cos α, sin α;
                                       sin α, 0, cos β;
                                       -cos α, -sin β, 0]
  det M = cos (β - 2*α) := by
sorry

end NUMINAMATH_CALUDE_determinant_of_special_matrix_l1266_126619


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l1266_126661

/-- Given a rectangle with dimensions 5 and 7 inches, if reducing one side by 2 inches
    results in an area of 21 square inches, then reducing the other side by 2 inches
    will result in an area of 25 square inches. -/
theorem rectangle_area_reduction (w h : ℝ) : 
  w = 5 ∧ h = 7 ∧ (w - 2) * h = 21 → w * (h - 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l1266_126661


namespace NUMINAMATH_CALUDE_girls_attending_event_l1266_126616

theorem girls_attending_event (total_students : ℕ) (total_attending : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1800 →
  total_attending = 1110 →
  girls + boys = total_students →
  (3 * girls) / 4 + (2 * boys) / 3 = total_attending →
  (3 * girls) / 4 = 690 :=
by sorry

end NUMINAMATH_CALUDE_girls_attending_event_l1266_126616


namespace NUMINAMATH_CALUDE_unique_equal_sum_existence_l1266_126637

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The statement that there exists exactly one positive integer n such that
    the sum of the first n terms of the arithmetic sequence (8, 12, ...)
    equals the sum of the first n terms of the arithmetic sequence (17, 19, ...) -/
theorem unique_equal_sum_existence : ∃! (n : ℕ), n > 0 ∧ 
  arithmetic_sum 8 4 n = arithmetic_sum 17 2 n := by sorry

end NUMINAMATH_CALUDE_unique_equal_sum_existence_l1266_126637


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l1266_126654

theorem product_of_five_consecutive_integers_not_square (n : ℕ+) :
  ¬∃ k : ℕ, (n : ℕ) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2 :=
sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l1266_126654


namespace NUMINAMATH_CALUDE_rectangular_envelope_foldable_l1266_126657

-- Define a rectangular envelope
structure RectangularEnvelope where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define a tetrahedron
structure Tetrahedron where
  surface_area : ℝ
  surface_area_positive : surface_area > 0

-- Define the property of being able to fold into two congruent tetrahedrons
def can_fold_into_congruent_tetrahedrons (env : RectangularEnvelope) : Prop :=
  ∃ (t : Tetrahedron), 
    t.surface_area = (env.length * env.width) / 2 ∧ 
    env.length ≠ env.width

-- State the theorem
theorem rectangular_envelope_foldable (env : RectangularEnvelope) :
  can_fold_into_congruent_tetrahedrons env :=
sorry

end NUMINAMATH_CALUDE_rectangular_envelope_foldable_l1266_126657


namespace NUMINAMATH_CALUDE_equation_solution_l1266_126620

theorem equation_solution (x : ℝ) (h : x ≥ 0) :
  (2021 * x = 2022 * (x^(2021/2022)) - 1) ↔ (x = 1) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1266_126620


namespace NUMINAMATH_CALUDE_sqrt_100_equals_10_l1266_126642

theorem sqrt_100_equals_10 : Real.sqrt 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_100_equals_10_l1266_126642


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1266_126689

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 5 = x + y + 3

/-- The main theorem stating that the function f(x) = x + 4 satisfies the functional equation -/
theorem functional_equation_solution :
  ∃ f : ℝ → ℝ, FunctionalEquation f ∧ ∀ x : ℝ, f x = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1266_126689


namespace NUMINAMATH_CALUDE_joydens_number_difference_l1266_126634

theorem joydens_number_difference (m j c : ℕ) : 
  m = j + 20 →
  j < c →
  c = 80 →
  m + j + c = 180 →
  c - j = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_joydens_number_difference_l1266_126634


namespace NUMINAMATH_CALUDE_f_properties_l1266_126685

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos (2 * x) + 3

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π/4 → f x ≤ M) ∧
  f (π/6) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1266_126685


namespace NUMINAMATH_CALUDE_complex_number_properties_l1266_126607

theorem complex_number_properties (z : ℂ) (h : z = (2 * Complex.I) / (-1 - Complex.I)) : 
  z ^ 2 = 2 * Complex.I ∧ z.im = -1 := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1266_126607


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l1266_126680

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 56

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 2184 :=
by sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l1266_126680


namespace NUMINAMATH_CALUDE_root_in_interval_l1266_126613

theorem root_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ Real.exp x + Real.log x = 0 := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1266_126613


namespace NUMINAMATH_CALUDE_mMobileCheaperByEleven_l1266_126691

/-- Calculates the cost of a mobile plan given the base cost for two lines, 
    the cost per additional line, and the total number of lines. -/
def mobilePlanCost (baseCost : ℕ) (additionalLineCost : ℕ) (totalLines : ℕ) : ℕ :=
  baseCost + (max (totalLines - 2) 0) * additionalLineCost

/-- Proves that M-Mobile is $11 cheaper than T-Mobile for a family plan with 5 lines. -/
theorem mMobileCheaperByEleven : 
  mobilePlanCost 50 16 5 - mobilePlanCost 45 14 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mMobileCheaperByEleven_l1266_126691
