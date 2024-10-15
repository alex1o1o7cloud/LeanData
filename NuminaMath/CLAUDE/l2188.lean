import Mathlib

namespace NUMINAMATH_CALUDE_stating_max_squares_correct_max_squares_1000_l2188_218890

/-- 
Represents the maximum number of squares that can be chosen on an m × n chessboard 
such that no three chosen squares have two in the same row and two in the same column.
-/
def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

/-- 
Theorem stating that max_squares gives the correct maximum number of squares
that can be chosen on an m × n chessboard under the given constraints.
-/
theorem max_squares_correct (m n : ℕ) (h : m ≤ n) :
  max_squares m n = 
    if m = 1 
    then n
    else m + n - 2 :=
by sorry

/-- 
Corollary for the specific case of a 1000 × 1000 chessboard.
-/
theorem max_squares_1000 : max_squares 1000 1000 = 1998 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_squares_correct_max_squares_1000_l2188_218890


namespace NUMINAMATH_CALUDE_square_of_1033_l2188_218887

theorem square_of_1033 : (1033 : ℕ)^2 = 1067089 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1033_l2188_218887


namespace NUMINAMATH_CALUDE_solve_equation_l2188_218820

theorem solve_equation (y : ℝ) : 7 - y = 12 ↔ y = -5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2188_218820


namespace NUMINAMATH_CALUDE_minimum_m_for_inequality_l2188_218868

open Real

theorem minimum_m_for_inequality (m : ℝ) :
  (∀ x > 0, (log x - (1/2) * m * x^2 + x) ≤ m * x - 1) ↔ m ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_minimum_m_for_inequality_l2188_218868


namespace NUMINAMATH_CALUDE_soap_bubble_thickness_l2188_218874

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem soap_bubble_thickness : toScientificNotation 0.0000007 = 
  { coefficient := 7,
    exponent := -7,
    is_valid := by sorry } := by sorry

end NUMINAMATH_CALUDE_soap_bubble_thickness_l2188_218874


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l2188_218866

theorem merchant_profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) :
  (S - C) / C * 100 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l2188_218866


namespace NUMINAMATH_CALUDE_evaluate_expression_l2188_218811

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 1) :
  y^2 * (y - 4*x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2188_218811


namespace NUMINAMATH_CALUDE_trig_identity_l2188_218879

theorem trig_identity (α : Real) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2188_218879


namespace NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l2188_218875

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem f_has_one_zero_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l2188_218875


namespace NUMINAMATH_CALUDE_problem_statement_l2188_218846

theorem problem_statement (x : ℝ) (h : x + 2/x = 4) :
  -5*x / (x^2 + 2) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2188_218846


namespace NUMINAMATH_CALUDE_sony_games_to_give_away_l2188_218810

theorem sony_games_to_give_away (current_sony_games : ℕ) (target_sony_games : ℕ) :
  current_sony_games = 132 →
  target_sony_games = 31 →
  current_sony_games - target_sony_games = 101 :=
by
  sorry

#check sony_games_to_give_away

end NUMINAMATH_CALUDE_sony_games_to_give_away_l2188_218810


namespace NUMINAMATH_CALUDE_apple_difference_l2188_218821

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 125

/-- The number of apples Adam has -/
def adam_apples : ℕ := 98

/-- The number of apples Laura has -/
def laura_apples : ℕ := 173

/-- The difference between Laura's apples and the sum of Jackie's and Adam's apples -/
theorem apple_difference : Int.ofNat laura_apples - Int.ofNat (jackie_apples + adam_apples) = -50 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l2188_218821


namespace NUMINAMATH_CALUDE_solution_abs_difference_l2188_218870

theorem solution_abs_difference (x y : ℝ) : 
  (Int.floor x : ℝ) + (y - Int.floor y) = 3.7 →
  (x - Int.floor x) + (Int.floor y : ℝ) = 4.2 →
  |x - 2*y| = 6.2 := by
sorry

end NUMINAMATH_CALUDE_solution_abs_difference_l2188_218870


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_five_l2188_218833

theorem absolute_value_sqrt_five (x : ℝ) : 
  |x| = Real.sqrt 5 → x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_five_l2188_218833


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2188_218828

theorem complex_on_imaginary_axis (a : ℝ) : ∃ y : ℝ, (a + I) * (1 + a * I) = y * I := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2188_218828


namespace NUMINAMATH_CALUDE_determinant_transformation_l2188_218853

theorem determinant_transformation (x y z w : ℝ) :
  (x * w - y * z = 3) →
  (x * (9 * z + 4 * w) - z * (9 * x + 4 * y) = 12) := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l2188_218853


namespace NUMINAMATH_CALUDE_complement_of_union_l2188_218883

open Set

def U : Finset ℕ := {1, 2, 3, 4}
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2188_218883


namespace NUMINAMATH_CALUDE_total_players_count_l2188_218808

/-- The number of players who play kabaddi -/
def kabaddi_players : ℕ := 10

/-- The number of players who play kho-kho only -/
def kho_kho_only_players : ℕ := 20

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabaddi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_players_count_l2188_218808


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l2188_218860

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 45) 
  (h2 : Nat.gcd a b = 9) : 
  a * b = 405 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l2188_218860


namespace NUMINAMATH_CALUDE_river_crossing_drift_l2188_218819

/-- Given a river crossing scenario, calculate the drift of the boat. -/
theorem river_crossing_drift (river_width : ℝ) (boat_speed : ℝ) (crossing_time : ℝ) 
  (h1 : river_width = 400)
  (h2 : boat_speed = 10)
  (h3 : crossing_time = 50) :
  boat_speed * crossing_time - river_width = 100 := by
  sorry

#check river_crossing_drift

end NUMINAMATH_CALUDE_river_crossing_drift_l2188_218819


namespace NUMINAMATH_CALUDE_equation_solutions_l2188_218858

theorem equation_solutions (x : ℝ) : 
  (7.331 * (Real.log x / Real.log 3 - 1) / (Real.log (x/3) / Real.log 3) - 
   2 * Real.log (Real.sqrt x) / Real.log 3 + 
   (Real.log x / Real.log 3)^2 = 3) ↔ 
  (x = 1/3 ∨ x = 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2188_218858


namespace NUMINAMATH_CALUDE_expression_value_l2188_218895

theorem expression_value : 
  (128^2 - 5^2) / (72^2 - 13^2) * ((72-13)*(72+13)) / ((128-5)*(128+5)) * (128+5) / (72+13) = 133/85 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2188_218895


namespace NUMINAMATH_CALUDE_equality_condition_l2188_218880

theorem equality_condition (x : ℝ) (h1 : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) = 15 ↔ x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2188_218880


namespace NUMINAMATH_CALUDE_combined_bus_capacity_l2188_218842

/-- The capacity of the train -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of one bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem: The combined capacity of the two buses is 40 people -/
theorem combined_bus_capacity :
  (num_buses : ℚ) * (bus_capacity_fraction * train_capacity) = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_bus_capacity_l2188_218842


namespace NUMINAMATH_CALUDE_money_distribution_l2188_218805

theorem money_distribution (a b : ℚ) : 
  (a + b / 2 = 50) → 
  (b + 2 * a / 3 = 50) → 
  (a = 37.5 ∧ b = 25) := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2188_218805


namespace NUMINAMATH_CALUDE_like_terms_imply_value_l2188_218892

theorem like_terms_imply_value (m n : ℤ) : 
  (m + 2 = 6 ∧ n + 1 = 3) → (-m)^3 + n^2 = -60 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_value_l2188_218892


namespace NUMINAMATH_CALUDE_no_two_digit_prime_sum_9_div_3_l2188_218882

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_prime_sum_9_div_3 :
  ¬ ∃ (n : ℕ), is_two_digit n ∧ Nat.Prime n ∧ sum_of_digits n = 9 ∧ n % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_prime_sum_9_div_3_l2188_218882


namespace NUMINAMATH_CALUDE_equivalent_operation_l2188_218856

theorem equivalent_operation (x : ℝ) : (x * (2/3)) / (5/6) = x * (4/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l2188_218856


namespace NUMINAMATH_CALUDE_two_students_adjacent_probability_l2188_218843

theorem two_students_adjacent_probability (n : ℕ) (h : n = 10) :
  (2 * Nat.factorial (n - 1)) / Nat.factorial n = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_two_students_adjacent_probability_l2188_218843


namespace NUMINAMATH_CALUDE_pool_capacity_exceeds_max_l2188_218873

-- Define the constants from the problem
def totalMaxCapacity : ℝ := 5000

-- Define the capacities of each section
def sectionACapacity : ℝ := 3000
def sectionBCapacity : ℝ := 2333.33
def sectionCCapacity : ℝ := 2000

-- Define the theorem
theorem pool_capacity_exceeds_max : 
  sectionACapacity + sectionBCapacity + sectionCCapacity > totalMaxCapacity :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_exceeds_max_l2188_218873


namespace NUMINAMATH_CALUDE_flammable_ice_scientific_notation_l2188_218864

theorem flammable_ice_scientific_notation :
  (800 * 10^9 : ℝ) = 8 * 10^11 := by sorry

end NUMINAMATH_CALUDE_flammable_ice_scientific_notation_l2188_218864


namespace NUMINAMATH_CALUDE_major_premise_for_increasing_cubic_l2188_218878

-- Define the function y = x³
def f (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

-- State the theorem
theorem major_premise_for_increasing_cubic :
  (∀ g : ℝ → ℝ, IsIncreasing g ↔ (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)) →
  IsIncreasing f :=
by sorry

end NUMINAMATH_CALUDE_major_premise_for_increasing_cubic_l2188_218878


namespace NUMINAMATH_CALUDE_alice_winning_condition_l2188_218801

/-- Game state representing the numbers on the board -/
structure GameState where
  numbers : List ℚ
  deriving Repr

/-- Player type -/
inductive Player
| Alice
| Bob
deriving Repr

/-- Result of the game -/
inductive GameResult
| AliceWins
| BobWins
deriving Repr

/-- Perform a move in the game -/
def makeMove (state : GameState) : GameState :=
  sorry

/-- Play the game with given parameters -/
def playGame (n : ℕ) (c : ℚ) (initialNumbers : List ℕ) : GameResult :=
  sorry

/-- Alice's winning condition -/
def aliceWins (c : ℚ) : Prop :=
  ∀ n₀ : ℕ, ∃ n : ℕ, n ≥ n₀ ∧ ∀ initialNumbers : List ℕ,
    initialNumbers.length = n → (∃ x y : ℕ, x ∈ initialNumbers ∧ y ∈ initialNumbers ∧ x ≠ y) →
      playGame n c initialNumbers = GameResult.AliceWins

theorem alice_winning_condition (c : ℚ) :
  aliceWins c ↔ c ≥ (1/2 : ℚ) :=
  sorry

end NUMINAMATH_CALUDE_alice_winning_condition_l2188_218801


namespace NUMINAMATH_CALUDE_product_equals_sum_l2188_218852

theorem product_equals_sum (g h : ℚ) : 
  (∀ d : ℚ, (5 * d^2 - 4 * d + g) * (4 * d^2 + h * d - 5) = 
    20 * d^4 - 31 * d^3 - 17 * d^2 + 23 * d - 10) → 
  g + h = (7 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_product_equals_sum_l2188_218852


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l2188_218827

/-- Given a mixture of raisins and nuts with specific quantities and price ratios,
    prove that the cost of raisins is 1/4 of the total cost. -/
theorem raisin_cost_fraction (raisin_pounds almond_pounds cashew_pounds : ℕ) 
                              (raisin_price : ℚ) :
  raisin_pounds = 4 →
  almond_pounds = 3 →
  cashew_pounds = 2 →
  raisin_price > 0 →
  (raisin_pounds * raisin_price) / 
  (raisin_pounds * raisin_price + 
   almond_pounds * (2 * raisin_price) + 
   cashew_pounds * (3 * raisin_price)) = 1 / 4 := by
  sorry

#check raisin_cost_fraction

end NUMINAMATH_CALUDE_raisin_cost_fraction_l2188_218827


namespace NUMINAMATH_CALUDE_total_marbles_count_l2188_218848

/-- Represents the colors of marbles in the bag -/
inductive MarbleColor
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents the bag of marbles -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- The ratio of marbles in the bag -/
def marbleRatio : MarbleBag := {
  red := 2,
  blue := 4,
  green := 3,
  yellow := 1
}

/-- The number of green marbles in the bag -/
def greenMarbleCount : ℕ := 24

/-- Theorem stating the total number of marbles in the bag -/
theorem total_marbles_count (bag : MarbleBag) 
  (h1 : bag.red = 2 * bag.green / 3)
  (h2 : bag.blue = 4 * bag.green / 3)
  (h3 : bag.yellow = bag.green / 3)
  (h4 : bag.green = greenMarbleCount) :
  bag.red + bag.blue + bag.green + bag.yellow = 80 := by
  sorry

#check total_marbles_count

end NUMINAMATH_CALUDE_total_marbles_count_l2188_218848


namespace NUMINAMATH_CALUDE_min_midpoint_for_transformed_sine_l2188_218829

theorem min_midpoint_for_transformed_sine (f g : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, f x = Real.sin (x + π/3)) →
  (∀ x, g x = Real.sin (2*x + π/3)) →
  (x₁ ≠ x₂) →
  (g x₁ * g x₂ = -1) →
  (∃ m, m = |(x₁ + x₂)/2| ∧ ∀ y₁ y₂, y₁ ≠ y₂ → g y₁ * g y₂ = -1 → m ≤ |(y₁ + y₂)/2|) →
  |(x₁ + x₂)/2| = π/6 :=
by sorry

end NUMINAMATH_CALUDE_min_midpoint_for_transformed_sine_l2188_218829


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l2188_218886

theorem cubic_equation_sum (a b c : ℝ) : 
  a^3 - 7*a^2 + 10*a = 12 →
  b^3 - 7*b^2 + 10*b = 12 →
  c^3 - 7*c^2 + 10*c = 12 →
  (a*b)/c + (b*c)/a + (c*a)/b = -17/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l2188_218886


namespace NUMINAMATH_CALUDE_circle_tangent_line_m_values_l2188_218816

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the translation vector
def translation_vector : ℝ × ℝ := (2, 1)

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop :=
  original_circle (x - translation_vector.1) (y - translation_vector.2)

-- Define the tangent line
def tangent_line (x y m : ℝ) : Prop := x + y + m = 0

-- Theorem statement
theorem circle_tangent_line_m_values :
  ∃ m : ℝ, (m = -1 ∨ m = -5) ∧
  ∀ x y : ℝ, translated_circle x y →
  (∃ p : ℝ × ℝ, p.1 + p.2 + m = 0 ∧
  ∀ q : ℝ × ℝ, q.1 + q.2 + m = 0 →
  (p.1 - x)^2 + (p.2 - y)^2 ≤ (q.1 - x)^2 + (q.2 - y)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_m_values_l2188_218816


namespace NUMINAMATH_CALUDE_shaded_area_sum_l2188_218839

/-- Given an equilateral triangle with side length 10 cm and an inscribed circle
    whose diameter is a side of the triangle, the sum of the areas of the two regions
    between the circle and the triangle can be expressed as a*π - b*√c,
    where a + b + c = 143/6. -/
theorem shaded_area_sum (a b c : ℝ) : 
  let side_length : ℝ := 10
  let triangle_area := side_length^2 * Real.sqrt 3 / 4
  let circle_radius := side_length / 2
  let sector_area := π * circle_radius^2 / 3
  let shaded_area := 2 * (sector_area - triangle_area / 2)
  (∃ (a b c : ℝ), shaded_area = a * π - b * Real.sqrt c ∧ a + b + c = 143/6) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l2188_218839


namespace NUMINAMATH_CALUDE_sum_irreducible_fractions_integer_l2188_218877

theorem sum_irreducible_fractions_integer (a b c d A : ℤ) 
  (h1 : b ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : Nat.gcd a.natAbs b.natAbs = 1) 
  (h4 : Nat.gcd c.natAbs d.natAbs = 1) 
  (h5 : a / b + c / d = A) : 
  b = d := by
sorry

end NUMINAMATH_CALUDE_sum_irreducible_fractions_integer_l2188_218877


namespace NUMINAMATH_CALUDE_production_time_theorem_l2188_218807

-- Define the time ratios for parts A, B, and C
def time_ratio_A : ℝ := 1
def time_ratio_B : ℝ := 2
def time_ratio_C : ℝ := 3

-- Define the number of parts produced in 10 hours
def parts_A_10h : ℕ := 2
def parts_B_10h : ℕ := 3
def parts_C_10h : ℕ := 4

-- Define the number of parts to be produced
def parts_A_target : ℕ := 14
def parts_B_target : ℕ := 10
def parts_C_target : ℕ := 2

-- Theorem to prove
theorem production_time_theorem :
  ∃ (x : ℝ),
    x > 0 ∧
    x * time_ratio_A * parts_A_10h + x * time_ratio_B * parts_B_10h + x * time_ratio_C * parts_C_10h = 10 ∧
    x * time_ratio_A * parts_A_target + x * time_ratio_B * parts_B_target + x * time_ratio_C * parts_C_target = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_production_time_theorem_l2188_218807


namespace NUMINAMATH_CALUDE_horner_method_operations_l2188_218871

/-- Polynomial coefficients in descending order of degree -/
def poly_coeffs : List ℤ := [5, 4, 1, 3, -81, 9, -1]

/-- Degree of the polynomial -/
def poly_degree : ℕ := poly_coeffs.length - 1

/-- Horner's method evaluation point -/
def x : ℤ := 2

/-- Number of additions in Horner's method -/
def num_additions : ℕ := poly_degree

/-- Number of multiplications in Horner's method -/
def num_multiplications : ℕ := poly_degree

theorem horner_method_operations :
  num_additions = 6 ∧ num_multiplications = 6 := by sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2188_218871


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2188_218841

theorem quadratic_factorization (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2188_218841


namespace NUMINAMATH_CALUDE_negation_equivalence_l2188_218899

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Icc (0 : ℝ) 1, x^3 + x^2 > 1) ↔
  (∀ x ∈ Set.Icc (0 : ℝ) 1, x^3 + x^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2188_218899


namespace NUMINAMATH_CALUDE_specific_composite_square_perimeter_l2188_218804

/-- Represents a square composed of four rectangles and an inner square -/
structure CompositeSquare where
  /-- Total area of the four rectangles -/
  rectangle_area : ℝ
  /-- Area of the square formed by the inner vertices of the rectangles -/
  inner_square_area : ℝ

/-- Calculates the total perimeter of the four rectangles in a CompositeSquare -/
def total_perimeter (cs : CompositeSquare) : ℝ :=
  sorry

/-- Theorem stating that for a specific CompositeSquare, the total perimeter is 48 -/
theorem specific_composite_square_perimeter :
  ∃ (cs : CompositeSquare),
    cs.rectangle_area = 32 ∧
    cs.inner_square_area = 20 ∧
    total_perimeter cs = 48 :=
  sorry

end NUMINAMATH_CALUDE_specific_composite_square_perimeter_l2188_218804


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2188_218849

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (3, -4)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2188_218849


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2188_218824

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - x -/
def f (x : ℝ) : ℝ := x^2 - x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2188_218824


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2188_218891

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2188_218891


namespace NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l2188_218894

def carrot_weight : ℝ := 250
def cucumber_multiplier : ℝ := 2.5

theorem total_weight_carrots_cucumbers : 
  carrot_weight + cucumber_multiplier * carrot_weight = 875 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l2188_218894


namespace NUMINAMATH_CALUDE_greendale_final_score_l2188_218855

/-- Roosevelt High School's basketball tournament scoring --/
def roosevelt_tournament (first_game : ℕ) (bonus : ℕ) : ℕ :=
  let second_game := first_game / 2
  let third_game := second_game * 3
  first_game + second_game + third_game + bonus

/-- Greendale High School's total points --/
def greendale_points (roosevelt_total : ℕ) : ℕ :=
  roosevelt_total - 10

/-- Theorem stating Greendale's final score --/
theorem greendale_final_score :
  greendale_points (roosevelt_tournament 30 50) = 130 := by
  sorry

end NUMINAMATH_CALUDE_greendale_final_score_l2188_218855


namespace NUMINAMATH_CALUDE_point_inside_ellipse_l2188_218889

/-- A point A(a, 1) is inside the ellipse x²/4 + y²/2 = 1 if and only if -√2 < a < √2 -/
theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) ↔ (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_point_inside_ellipse_l2188_218889


namespace NUMINAMATH_CALUDE_total_vegetables_l2188_218898

def garden_vegetables (potatoes cucumbers peppers : ℕ) : Prop :=
  (cucumbers = potatoes - 60) ∧
  (peppers = 2 * cucumbers) ∧
  (potatoes + cucumbers + peppers = 768)

theorem total_vegetables : ∃ (cucumbers peppers : ℕ), 
  garden_vegetables 237 cucumbers peppers := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_l2188_218898


namespace NUMINAMATH_CALUDE_veranda_width_l2188_218851

/-- Proves that the width of a veranda surrounding a 20 m × 12 m rectangular room is 2 m,
    given that the area of the veranda is 144 m². -/
theorem veranda_width (room_length : ℝ) (room_width : ℝ) (veranda_area : ℝ) :
  room_length = 20 →
  room_width = 12 →
  veranda_area = 144 →
  ∃ w : ℝ, w > 0 ∧ (room_length + 2*w) * (room_width + 2*w) - room_length * room_width = veranda_area ∧ w = 2 :=
by sorry

end NUMINAMATH_CALUDE_veranda_width_l2188_218851


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l2188_218831

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -8) : 
  x + y = -1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l2188_218831


namespace NUMINAMATH_CALUDE_gcd_of_2750_and_9450_l2188_218830

theorem gcd_of_2750_and_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_2750_and_9450_l2188_218830


namespace NUMINAMATH_CALUDE_total_shark_teeth_l2188_218826

/-- The number of teeth a tiger shark has -/
def tiger_teeth : ℕ := 180

/-- The number of teeth a hammerhead shark has -/
def hammerhead_teeth : ℕ := tiger_teeth / 6

/-- The number of teeth a great white shark has -/
def great_white_teeth : ℕ := 2 * (tiger_teeth + hammerhead_teeth)

/-- The number of teeth a mako shark has -/
def mako_teeth : ℕ := (5 * hammerhead_teeth) / 3

/-- The total number of teeth for all four sharks -/
def total_teeth : ℕ := tiger_teeth + hammerhead_teeth + great_white_teeth + mako_teeth

theorem total_shark_teeth : total_teeth = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_shark_teeth_l2188_218826


namespace NUMINAMATH_CALUDE_largest_c_for_seven_in_range_l2188_218865

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

theorem largest_c_for_seven_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = 7) → d ≤ c) ∧
  (∃ (x : ℝ), f (37/4) x = 7) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_seven_in_range_l2188_218865


namespace NUMINAMATH_CALUDE_only_setC_forms_right_triangle_l2188_218869

-- Define a function to check if three numbers can form a right triangle
def canFormRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def setA : List ℕ := [4, 5, 6]
def setB : List ℕ := [5, 7, 9]
def setC : List ℕ := [6, 8, 10]
def setD : List ℕ := [7, 8, 9]

-- Theorem stating that only set C can form a right triangle
theorem only_setC_forms_right_triangle :
  (¬ canFormRightTriangle setA[0] setA[1] setA[2]) ∧
  (¬ canFormRightTriangle setB[0] setB[1] setB[2]) ∧
  (canFormRightTriangle setC[0] setC[1] setC[2]) ∧
  (¬ canFormRightTriangle setD[0] setD[1] setD[2]) :=
by
  sorry

#check only_setC_forms_right_triangle

end NUMINAMATH_CALUDE_only_setC_forms_right_triangle_l2188_218869


namespace NUMINAMATH_CALUDE_triangle_properties_l2188_218850

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 4)
  (h2 : t.b = 6)
  (h3 : Real.sin t.A = Real.sin (2 * t.B)) :
  Real.cos t.B = 1/3 ∧ 
  1/2 * t.a * t.c * Real.sin t.B = 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2188_218850


namespace NUMINAMATH_CALUDE_converse_A_false_others_true_l2188_218881

-- Define the basic geometric concepts
structure Triangle where
  angles : Fin 3 → ℝ
  sides : Fin 3 → ℝ

def is_congruent (t1 t2 : Triangle) : Prop := sorry

def is_right_triangle (t : Triangle) : Prop := sorry

def is_equilateral (t : Triangle) : Prop := sorry

def are_complementary (a b : ℝ) : Prop := sorry

-- Define the statements and their converses
def statement_A (t1 t2 : Triangle) : Prop :=
  is_congruent t1 t2 → ∀ i : Fin 3, t1.angles i = t2.angles i

def converse_A (t1 t2 : Triangle) : Prop :=
  (∀ i : Fin 3, t1.angles i = t2.angles i) → is_congruent t1 t2

def statement_B (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.angles i = t.angles j) → (∀ i j : Fin 3, t.sides i = t.sides j)

def converse_B (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.sides i = t.sides j) → (∀ i j : Fin 3, t.angles i = t.angles j)

def statement_C (t : Triangle) : Prop :=
  is_right_triangle t → are_complementary (t.angles 0) (t.angles 1)

def converse_C (t : Triangle) : Prop :=
  are_complementary (t.angles 0) (t.angles 1) → is_right_triangle t

def statement_D (t : Triangle) : Prop :=
  is_equilateral t → (∀ i j : Fin 3, t.angles i = t.angles j)

def converse_D (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.angles i = t.angles j) → is_equilateral t

-- Main theorem
theorem converse_A_false_others_true :
  (∃ t1 t2 : Triangle, converse_A t1 t2 = false) ∧
  (∀ t : Triangle, converse_B t = true) ∧
  (∀ t : Triangle, converse_C t = true) ∧
  (∀ t : Triangle, converse_D t = true) := by sorry

end NUMINAMATH_CALUDE_converse_A_false_others_true_l2188_218881


namespace NUMINAMATH_CALUDE_height_of_specific_block_l2188_218823

/-- Represents a rectangular block --/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of the block in cubic centimeters --/
def volume (block : RectangularBlock) : ℕ :=
  block.length * block.width * block.height

/-- The perimeter of the base of the block in centimeters --/
def basePerimeter (block : RectangularBlock) : ℕ :=
  2 * (block.length + block.width)

theorem height_of_specific_block :
  ∃ (block : RectangularBlock),
    volume block = 42 ∧
    basePerimeter block = 18 ∧
    block.height = 3 :=
by
  sorry

#check height_of_specific_block

end NUMINAMATH_CALUDE_height_of_specific_block_l2188_218823


namespace NUMINAMATH_CALUDE_unique_b_value_l2188_218809

theorem unique_b_value (a b : ℕ+) (h1 : (3 ^ a.val) ^ b.val = 3 ^ 3) (h2 : 3 ^ a.val * 3 ^ b.val = 81) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l2188_218809


namespace NUMINAMATH_CALUDE_small_jar_capacity_l2188_218845

theorem small_jar_capacity 
  (total_jars : ℕ) 
  (large_jar_capacity : ℕ) 
  (total_capacity : ℕ) 
  (small_jars : ℕ) 
  (h1 : total_jars = 100)
  (h2 : large_jar_capacity = 5)
  (h3 : total_capacity = 376)
  (h4 : small_jars = 62) :
  (total_capacity - (total_jars - small_jars) * large_jar_capacity) / small_jars = 3 := by
  sorry

end NUMINAMATH_CALUDE_small_jar_capacity_l2188_218845


namespace NUMINAMATH_CALUDE_longest_segment_is_BD_l2188_218818

/-- Given a triangle ABC, returns true if AC > AB > BC -/
def triangleInequalityOrder (angleA angleB angleC : ℝ) : Prop :=
  angleA > angleB ∧ angleB > angleC

theorem longest_segment_is_BD 
  (angleABD angleADB angleCBD angleBDC : ℝ)
  (h1 : angleABD = 50)
  (h2 : angleADB = 45)
  (h3 : angleCBD = 70)
  (h4 : angleBDC = 65)
  (h5 : triangleInequalityOrder (180 - angleABD - angleADB) angleABD angleADB)
  (h6 : triangleInequalityOrder angleCBD angleBDC (180 - angleCBD - angleBDC)) :
  ∃ (lengthAB lengthBC lengthCD lengthAD lengthBD : ℝ),
    lengthAD < lengthAB ∧ 
    lengthAB < lengthBC ∧ 
    lengthBC < lengthCD ∧ 
    lengthCD < lengthBD :=
by sorry

end NUMINAMATH_CALUDE_longest_segment_is_BD_l2188_218818


namespace NUMINAMATH_CALUDE_expand_expression_l2188_218835

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2188_218835


namespace NUMINAMATH_CALUDE_opposite_sides_iff_in_set_l2188_218803

/-- The set of real numbers a for which points A and B lie on opposite sides of the line 3x - y = 4 -/
def opposite_sides_set : Set ℝ :=
  {a | a < -1 ∨ (-1/3 < a ∧ a < 0) ∨ a > 8/3}

/-- Point A coordinates satisfy the given equation -/
def point_A (a x y : ℝ) : Prop :=
  26 * a^2 - 22 * a * x - 20 * a * y + 5 * x^2 + 8 * x * y + 4 * y^2 = 0

/-- Parabola equation with vertex at point B -/
def parabola (a x y : ℝ) : Prop :=
  a * x^2 + 2 * a^2 * x - a * y + a^3 + 1 = 0

/-- Line equation -/
def line (x y : ℝ) : Prop :=
  3 * x - y = 4

/-- Main theorem: A and B lie on opposite sides of the line if and only if a is in the opposite_sides_set -/
theorem opposite_sides_iff_in_set (a : ℝ) :
  (∃ x_a y_a x_b y_b : ℝ,
    point_A a x_a y_a ∧
    parabola a x_b y_b ∧
    ¬line x_a y_a ∧
    ¬line x_b y_b ∧
    (3 * x_a - y_a - 4) * (3 * x_b - y_b - 4) < 0) ↔
  a ∈ opposite_sides_set := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_iff_in_set_l2188_218803


namespace NUMINAMATH_CALUDE_mikes_games_last_year_l2188_218893

/-- The number of basketball games Mike went to this year -/
def games_this_year : ℕ := 15

/-- The number of basketball games Mike missed this year -/
def games_missed : ℕ := 41

/-- The total number of basketball games Mike went to -/
def total_games : ℕ := 54

/-- The number of basketball games Mike went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem mikes_games_last_year : games_last_year = 39 := by
  sorry

end NUMINAMATH_CALUDE_mikes_games_last_year_l2188_218893


namespace NUMINAMATH_CALUDE_factorial_sum_division_l2188_218896

theorem factorial_sum_division : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_division_l2188_218896


namespace NUMINAMATH_CALUDE_age_problem_solution_l2188_218888

/-- Represents the ages of two people --/
structure Ages where
  your_age : ℕ
  my_age : ℕ

/-- The conditions of the age problem --/
def age_conditions (ages : Ages) : Prop :=
  -- Condition 1: I am twice as old as you were when I was as old as you are now
  ages.your_age = 2 * (2 * ages.my_age - ages.your_age) ∧
  -- Condition 2: When you are as old as I am now, the sum of our ages will be 140 years
  ages.my_age + (2 * ages.my_age - ages.your_age) = 140

/-- The theorem stating the solution to the age problem --/
theorem age_problem_solution :
  ∃ (ages : Ages), age_conditions ages ∧ ages.your_age = 112 ∧ ages.my_age = 84 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_solution_l2188_218888


namespace NUMINAMATH_CALUDE_exist_irrational_with_natural_power_l2188_218884

theorem exist_irrational_with_natural_power : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Irrational a ∧ Irrational b ∧ ∃ (n : ℕ), a^b = n :=
sorry

end NUMINAMATH_CALUDE_exist_irrational_with_natural_power_l2188_218884


namespace NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l2188_218861

/-- The cost price of an article given profit and loss conditions -/
theorem cost_price_from_profit_loss_equality (cost_price : ℝ) : 
  (66 - cost_price = cost_price - 22) → cost_price = 44 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l2188_218861


namespace NUMINAMATH_CALUDE_pizza_theorem_l2188_218857

def pizza_problem (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ) : Prop :=
  craig_day1 = 40 ∧
  craig_day2 = craig_day1 + 60 ∧
  heather_day1 = 4 * craig_day1 ∧
  heather_day2 = craig_day2 - 20 ∧
  craig_day1 + craig_day2 + heather_day1 + heather_day2 = 380

theorem pizza_theorem : ∃ craig_day1 craig_day2 heather_day1 heather_day2 : ℕ,
  pizza_problem craig_day1 craig_day2 heather_day1 heather_day2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l2188_218857


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l2188_218844

theorem polygon_interior_exterior_angle_relation (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l2188_218844


namespace NUMINAMATH_CALUDE_final_cake_count_l2188_218885

-- Define the problem parameters
def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

-- Theorem statement
theorem final_cake_count :
  initial_cakes - cakes_sold + additional_cakes = 111 := by
  sorry

end NUMINAMATH_CALUDE_final_cake_count_l2188_218885


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l2188_218867

theorem trig_expression_simplification (α β : ℝ) :
  (Real.cos α * Real.cos β - Real.cos (α + β)) / (Real.cos (α - β) - Real.sin α * Real.sin β) = Real.tan α * Real.tan β :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l2188_218867


namespace NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l2188_218813

/-- Prove that given vectors a = (1,2) and b = (-4,m), if a ⊥ b, then m = 2 -/
theorem orthogonal_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-4, m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l2188_218813


namespace NUMINAMATH_CALUDE_triangle_value_l2188_218859

theorem triangle_value (triangle p : ℤ) 
  (h1 : triangle + p = 75)
  (h2 : 3 * (triangle + p) - p = 198) : 
  triangle = 48 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l2188_218859


namespace NUMINAMATH_CALUDE_solution_set_correct_l2188_218840

def solution_set : Set (ℚ × ℚ) :=
  {(-2/3, 1), (1, 1), (-1/3, -3), (-1/3, 2)}

def satisfies_equations (p : ℚ × ℚ) : Prop :=
  let x := p.1
  let y := p.2
  (3*x - y - 3*x*y = -1) ∧ (9*x^2*y^2 + 9*x^2 + y^2 - 6*x*y = 13)

theorem solution_set_correct :
  ∀ p : ℚ × ℚ, p ∈ solution_set ↔ satisfies_equations p :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2188_218840


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2188_218863

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the directrices of C₁
def l₁ : ℝ := -4
def l₂ : ℝ := 4

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = -16 * x

-- Define the intersection points A and B
def A : ℝ × ℝ := (-4, 8)
def B : ℝ × ℝ := (-4, -8)

-- Theorem statement
theorem ellipse_parabola_intersection :
  (∀ x y, C₁ x y → (x = l₁ ∨ x = l₂)) ∧
  (∀ x y, C₂ x y → (x = 0 ∨ x = l₂)) ∧
  (C₂ A.1 A.2 ∧ C₂ B.1 B.2) ∧
  (A.1 = l₁ ∧ B.1 = l₁) →
  (∀ x y, C₂ x y ↔ y^2 = -16 * x) ∧
  (A.2 - B.2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2188_218863


namespace NUMINAMATH_CALUDE_paul_initial_strawberries_l2188_218847

/-- The number of strawberries Paul initially had -/
def initial_strawberries : ℕ := sorry

/-- The number of strawberries Paul picked -/
def picked_strawberries : ℕ := 35

/-- The total number of strawberries Paul had after picking more -/
def total_strawberries : ℕ := 63

theorem paul_initial_strawberries : 
  initial_strawberries = 28 :=
by
  have h : initial_strawberries + picked_strawberries = total_strawberries := sorry
  sorry

end NUMINAMATH_CALUDE_paul_initial_strawberries_l2188_218847


namespace NUMINAMATH_CALUDE_equal_cost_sharing_l2188_218838

theorem equal_cost_sharing (A B : ℝ) (h : A < B) :
  (B - A) / 2 = (A + B) / 2 - A := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_sharing_l2188_218838


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_1_range_of_a_when_not_p_implies_not_q_l2188_218806

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x_when_a_is_1 :
  ∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  (∀ x a : ℝ, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x a : ℝ, ¬(q x) ∧ p x a) →
  ∀ a : ℝ, 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_1_range_of_a_when_not_p_implies_not_q_l2188_218806


namespace NUMINAMATH_CALUDE_range_of_m_satisfies_conditions_l2188_218817

/-- Given two functions f and g, prove that the range of m satisfies the given conditions -/
theorem range_of_m_satisfies_conditions (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 + m) →
  (∀ x, g x = 2^x - m) →
  (∀ x₁ ∈ Set.Icc (-1) 2, ∃ x₂ ∈ Set.Icc 0 3, f x₁ = g x₂) →
  m ∈ Set.Icc (1/2) 2 := by
  sorry

#check range_of_m_satisfies_conditions

end NUMINAMATH_CALUDE_range_of_m_satisfies_conditions_l2188_218817


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_fifteen_l2188_218802

theorem remainder_when_divided_by_fifteen (r : ℕ) (h : r / 15 = 82 / 10) : r % 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_fifteen_l2188_218802


namespace NUMINAMATH_CALUDE_school_dinner_theatre_tickets_l2188_218854

theorem school_dinner_theatre_tickets (child_price adult_price total_tickets total_revenue : ℕ) 
  (h1 : child_price = 6)
  (h2 : adult_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_revenue = 1875) :
  ∃ (children adults : ℕ),
    children + adults = total_tickets ∧
    child_price * children + adult_price * adults = total_revenue ∧
    children = 50 := by
  sorry

end NUMINAMATH_CALUDE_school_dinner_theatre_tickets_l2188_218854


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l2188_218825

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio_two 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) 
  (h_ratio : ∀ n : ℕ, a (n + 1) = 2 * a n) : 
  (2 * a 1 + a 2) / (2 * a 3 + a 4) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l2188_218825


namespace NUMINAMATH_CALUDE_construction_cost_is_212900_l2188_218872

-- Define the cost components
def land_cost_per_sqm : ℚ := 60
def land_area : ℚ := 2500
def brick_cost_per_1000 : ℚ := 120
def brick_quantity : ℚ := 15000
def roof_tile_cost : ℚ := 12
def roof_tile_quantity : ℚ := 800
def cement_bag_cost : ℚ := 8
def cement_bag_quantity : ℚ := 250
def wooden_beam_cost_per_m : ℚ := 25
def wooden_beam_length : ℚ := 1000
def steel_bar_cost_per_m : ℚ := 15
def steel_bar_length : ℚ := 500
def electrical_wiring_cost_per_m : ℚ := 2
def electrical_wiring_length : ℚ := 2000
def plumbing_pipe_cost_per_m : ℚ := 4
def plumbing_pipe_length : ℚ := 3000

-- Define the total construction cost function
def total_construction_cost : ℚ :=
  land_cost_per_sqm * land_area +
  brick_cost_per_1000 * brick_quantity / 1000 +
  roof_tile_cost * roof_tile_quantity +
  cement_bag_cost * cement_bag_quantity +
  wooden_beam_cost_per_m * wooden_beam_length +
  steel_bar_cost_per_m * steel_bar_length +
  electrical_wiring_cost_per_m * electrical_wiring_length +
  plumbing_pipe_cost_per_m * plumbing_pipe_length

-- Theorem statement
theorem construction_cost_is_212900 :
  total_construction_cost = 212900 := by
  sorry

end NUMINAMATH_CALUDE_construction_cost_is_212900_l2188_218872


namespace NUMINAMATH_CALUDE_population_in_scientific_notation_l2188_218862

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a number to scientific notation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem population_in_scientific_notation :
  let population_millions : ℝ := 141178
  let population : ℝ := population_millions * 1000000
  let scientific_form := toScientificNotation population
  scientific_form.coefficient = 1.41178 ∧ scientific_form.exponent = 9 :=
sorry

end NUMINAMATH_CALUDE_population_in_scientific_notation_l2188_218862


namespace NUMINAMATH_CALUDE_travel_equation_correct_l2188_218836

/-- Represents the scenario of Confucius and his students traveling to a school -/
structure TravelScenario where
  distance : ℝ
  student_speed : ℝ
  cart_speed_multiplier : ℝ
  head_start : ℝ

/-- The equation representing the travel times is correct for the given scenario -/
theorem travel_equation_correct (scenario : TravelScenario) 
  (h_distance : scenario.distance = 30)
  (h_cart_speed : scenario.cart_speed_multiplier = 1.5)
  (h_head_start : scenario.head_start = 1)
  (h_student_speed_pos : scenario.student_speed > 0) :
  scenario.distance / scenario.student_speed = 
    scenario.distance / (scenario.cart_speed_multiplier * scenario.student_speed) + scenario.head_start :=
sorry

end NUMINAMATH_CALUDE_travel_equation_correct_l2188_218836


namespace NUMINAMATH_CALUDE_total_time_is_ten_years_l2188_218837

/-- The total time taken to find two artifacts given the research and expedition time for the first artifact, and a multiplier for the second artifact. -/
def total_time_for_artifacts (research_time_1 : ℝ) (expedition_time_1 : ℝ) (multiplier : ℝ) : ℝ :=
  let time_1 := research_time_1 + expedition_time_1
  let time_2 := time_1 * multiplier
  time_1 + time_2

/-- Theorem stating that the total time to find both artifacts is 10 years -/
theorem total_time_is_ten_years :
  total_time_for_artifacts 0.5 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_ten_years_l2188_218837


namespace NUMINAMATH_CALUDE_safari_animal_count_l2188_218897

theorem safari_animal_count (total animals : ℕ) (antelopes rabbits hyenas wild_dogs leopards : ℕ) :
  total = 605 →
  antelopes = 80 →
  rabbits = antelopes + 34 →
  hyenas = antelopes + rabbits - 42 →
  wild_dogs > hyenas →
  leopards * 2 = rabbits →
  total = antelopes + rabbits + hyenas + wild_dogs + leopards →
  wild_dogs - hyenas = 50 := by
  sorry

end NUMINAMATH_CALUDE_safari_animal_count_l2188_218897


namespace NUMINAMATH_CALUDE_smallest_square_for_five_disks_l2188_218815

/-- A disk with radius 1 -/
structure UnitDisk where
  center : ℝ × ℝ

/-- A square with side length a -/
structure Square (a : ℝ) where
  center : ℝ × ℝ

/-- Predicate to check if two disks overlap -/
def disks_overlap (d1 d2 : UnitDisk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 < 4

/-- Predicate to check if a disk is contained in a square -/
def disk_in_square (d : UnitDisk) (s : Square a) : Prop :=
  abs (d.center.1 - s.center.1) ≤ a/2 - 1 ∧ abs (d.center.2 - s.center.2) ≤ a/2 - 1

/-- The main theorem -/
theorem smallest_square_for_five_disks :
  ∀ a : ℝ,
  (∃ (s : Square a) (d1 d2 d3 d4 d5 : UnitDisk),
    disk_in_square d1 s ∧ disk_in_square d2 s ∧ disk_in_square d3 s ∧ disk_in_square d4 s ∧ disk_in_square d5 s ∧
    ¬disks_overlap d1 d2 ∧ ¬disks_overlap d1 d3 ∧ ¬disks_overlap d1 d4 ∧ ¬disks_overlap d1 d5 ∧
    ¬disks_overlap d2 d3 ∧ ¬disks_overlap d2 d4 ∧ ¬disks_overlap d2 d5 ∧
    ¬disks_overlap d3 d4 ∧ ¬disks_overlap d3 d5 ∧
    ¬disks_overlap d4 d5) →
  a ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_for_five_disks_l2188_218815


namespace NUMINAMATH_CALUDE_cuboid_height_l2188_218822

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edges of a cuboid -/
def Cuboid.sumOfEdges (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

theorem cuboid_height (c : Cuboid) 
  (h_sum : c.sumOfEdges = 224)
  (h_width : c.width = 30)
  (h_length : c.length = 22) :
  c.height = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_l2188_218822


namespace NUMINAMATH_CALUDE_bean_garden_columns_l2188_218832

/-- A garden with bean plants arranged in rows and columns. -/
structure BeanGarden where
  rows : ℕ
  columns : ℕ
  total_plants : ℕ
  h_total : total_plants = rows * columns

/-- The number of columns in a bean garden with 52 rows and 780 total plants is 15. -/
theorem bean_garden_columns (garden : BeanGarden) 
    (h_rows : garden.rows = 52) 
    (h_total : garden.total_plants = 780) : 
    garden.columns = 15 := by
  sorry

end NUMINAMATH_CALUDE_bean_garden_columns_l2188_218832


namespace NUMINAMATH_CALUDE_unique_divisor_1058_l2188_218812

theorem unique_divisor_1058 : ∃! d : ℕ, d ≠ 1 ∧ d ∣ 1058 := by sorry

end NUMINAMATH_CALUDE_unique_divisor_1058_l2188_218812


namespace NUMINAMATH_CALUDE_min_angle_in_prime_triangle_l2188_218814

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem min_angle_in_prime_triangle (a b c : ℕ) : 
  (a + b + c = 180) →
  (is_prime a) →
  (is_prime b) →
  (is_prime c) →
  (a > b) →
  (b > c) →
  c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_angle_in_prime_triangle_l2188_218814


namespace NUMINAMATH_CALUDE_complex_moduli_product_l2188_218876

theorem complex_moduli_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_moduli_product_l2188_218876


namespace NUMINAMATH_CALUDE_complex_expression_squared_l2188_218834

theorem complex_expression_squared (x y z p : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 15)
  (h2 : x * y = 3)
  (h3 : x * z = 4)
  (h4 : Real.cos x + Real.sin y + Real.tan z = p) :
  (x - y - z)^2 = (Real.sqrt ((15 + 5 * Real.sqrt 5) / 2) - 
                   3 / Real.sqrt ((15 + 5 * Real.sqrt 5) / 2) - 
                   4 / Real.sqrt ((15 + 5 * Real.sqrt 5) / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_squared_l2188_218834


namespace NUMINAMATH_CALUDE_min_value_of_f_l2188_218800

def f (x : ℝ) : ℝ := 
  Finset.sum (Finset.range 2015) (fun i => (i + 1) * x^(2014 - i))

theorem min_value_of_f :
  ∃ (min : ℝ), min = 1008 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2188_218800
