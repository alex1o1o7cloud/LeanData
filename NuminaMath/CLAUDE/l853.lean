import Mathlib

namespace NUMINAMATH_CALUDE_complex_power_36_l853_85375

theorem complex_power_36 :
  (Complex.exp (160 * π / 180 * Complex.I))^36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_power_36_l853_85375


namespace NUMINAMATH_CALUDE_circular_table_seating_l853_85368

theorem circular_table_seating (n : ℕ) (a : Fin (2*n) → Fin (2*n)) 
  (h_perm : Function.Bijective a) :
  ∃ i j : Fin (2*n), i ≠ j ∧ 
    (a i - a j : ℤ) % (2*n) = (i - j : ℤ) % (2*n) ∨
    (a i - a j : ℤ) % (2*n) = (i - j - 2*n : ℤ) % (2*n) :=
by sorry

end NUMINAMATH_CALUDE_circular_table_seating_l853_85368


namespace NUMINAMATH_CALUDE_nickel_count_is_three_l853_85386

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total value of coins in cents -/
def totalValue (c : CoinCount) : ℕ :=
  c.pennies * 1 + c.nickels * 5 + c.dimes * 10

/-- The total number of coins -/
def totalCoins (c : CoinCount) : ℕ :=
  c.pennies + c.nickels + c.dimes

theorem nickel_count_is_three :
  ∃ (c : CoinCount),
    totalCoins c = 8 ∧
    totalValue c = 47 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    (∀ (c' : CoinCount),
      totalCoins c' = 8 →
      totalValue c' = 47 →
      c'.pennies ≥ 1 →
      c'.nickels ≥ 1 →
      c'.dimes ≥ 1 →
      c'.nickels = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_nickel_count_is_three_l853_85386


namespace NUMINAMATH_CALUDE_unique_prime_product_power_l853_85304

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

theorem unique_prime_product_power :
  ∀ k : ℕ, k > 0 →
    (∃ a n : ℕ, n > 1 ∧ primeProduct k - 1 = a^n) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_product_power_l853_85304


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l853_85384

/-- The imaginary part of the product of two complex numbers -/
theorem imaginary_part_of_product (ω₁ ω₂ : ℂ) : 
  let z := ω₁ * ω₂
  ω₁ = -1/2 + (Real.sqrt 3/2) * I →
  ω₂ = Complex.exp (I * (π/12)) →
  z.im = Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l853_85384


namespace NUMINAMATH_CALUDE_friendly_snakes_not_green_l853_85334

structure Snake where
  friendly : Bool
  green : Bool
  can_multiply : Bool
  can_divide : Bool

def Tom_snakes : Finset Snake := sorry

theorem friendly_snakes_not_green :
  ∀ s ∈ Tom_snakes,
  (s.friendly → s.can_multiply) ∧
  (s.green → ¬s.can_divide) ∧
  (¬s.can_divide → ¬s.can_multiply) →
  (s.friendly → ¬s.green) :=
by sorry

end NUMINAMATH_CALUDE_friendly_snakes_not_green_l853_85334


namespace NUMINAMATH_CALUDE_mean_score_calculation_l853_85305

theorem mean_score_calculation (f s : ℕ) (F S : ℝ) : 
  F = 92 →
  S = 78 →
  f = 2 * s / 3 →
  (F * f + S * s) / (f + s) = 83.6 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_score_calculation_l853_85305


namespace NUMINAMATH_CALUDE_number_975_in_column_B_l853_85380

/-- Represents the columns in the arrangement --/
inductive Column
| A | B | C | D | E | F

/-- Determines if a given row number is odd --/
def isOddRow (n : ℕ) : Bool :=
  n % 2 = 1

/-- Calculates the column for a given number in the arrangement --/
def columnForNumber (n : ℕ) : Column :=
  let adjustedN := n - 1
  let rowNumber := (adjustedN / 6) + 1
  let positionInRow := adjustedN % 6
  if isOddRow rowNumber then
    match positionInRow with
    | 0 => Column.A
    | 1 => Column.B
    | 2 => Column.C
    | 3 => Column.D
    | 4 => Column.E
    | _ => Column.F
  else
    match positionInRow with
    | 0 => Column.F
    | 1 => Column.E
    | 2 => Column.D
    | 3 => Column.C
    | 4 => Column.B
    | _ => Column.A

/-- Theorem: The integer 975 is in column B in the given arrangement --/
theorem number_975_in_column_B : columnForNumber 975 = Column.B := by
  sorry

end NUMINAMATH_CALUDE_number_975_in_column_B_l853_85380


namespace NUMINAMATH_CALUDE_expression_multiple_of_six_l853_85364

theorem expression_multiple_of_six (n : ℕ) (h : n ≥ 12) :
  ∃ k : ℤ, ((n + 3).factorial - 2 * (n + 2).factorial) / (n + 1).factorial = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_expression_multiple_of_six_l853_85364


namespace NUMINAMATH_CALUDE_green_balls_count_l853_85356

/-- The number of green balls in a bag with specific conditions -/
def num_green_balls (total : ℕ) (white : ℕ) (yellow : ℕ) (red : ℕ) (purple : ℕ) 
    (prob_not_red_purple : ℚ) : ℕ :=
  total - (white + yellow + red + purple)

theorem green_balls_count :
  let total := 60
  let white := 22
  let yellow := 2
  let red := 15
  let purple := 3
  let prob_not_red_purple := 7/10
  num_green_balls total white yellow red purple prob_not_red_purple = 18 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l853_85356


namespace NUMINAMATH_CALUDE_vector_relation_l853_85344

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem vector_relation (h : B - A = 2 • (D - C)) :
  B - D = A - C - (3/2 : ℝ) • (B - A) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_l853_85344


namespace NUMINAMATH_CALUDE_plane_at_distance_from_point_and_through_axis_l853_85393

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

structure Sphere where
  center : Point
  radius : ℝ

def ProjectionAxis : Set Point := sorry

-- Define the distance between a point and a plane
def distancePointPlane (p : Point) (plane : Plane) : ℝ := sorry

-- Define a predicate for a plane passing through the projection axis
def passesThroughProjectionAxis (plane : Plane) : Prop := sorry

-- Define a predicate for a plane being tangent to a sphere
def isTangentTo (plane : Plane) (sphere : Sphere) : Prop := sorry

-- The main theorem
theorem plane_at_distance_from_point_and_through_axis
  (A : Point) (d : ℝ) (P : Plane) :
  (distancePointPlane A P = d ∧ passesThroughProjectionAxis P) ↔
  (isTangentTo P (Sphere.mk A d) ∧ passesThroughProjectionAxis P) := by
  sorry

end NUMINAMATH_CALUDE_plane_at_distance_from_point_and_through_axis_l853_85393


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l853_85392

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n} where a_3 + a_11 = 40, 
    the value of a_6 - a_7 + a_8 is equal to 20 -/
theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 11 = 40) : 
  a 6 - a 7 + a 8 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l853_85392


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l853_85371

open Real

/-- The minimum distance between two points on different curves -/
theorem min_distance_between_curves (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), 
    a = 3 * x₁ + 3 ∧ 
    a = 2 * x₂ + log x₂ ∧
    ∀ (y₁ y₂ : ℝ), 
      (a = 3 * y₁ + 3 ∧ a = 2 * y₂ + log y₂) → 
      |x₂ - x₁| ≤ |y₂ - y₁| ∧
      |x₂ - x₁| = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l853_85371


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l853_85385

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -2/3 * x^3 + 2 * x^2 - 8/3 * x - 16/3
  (q 1 = -6) ∧ (q 2 = -8) ∧ (q 3 = -14) ∧ (q 4 = -28) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l853_85385


namespace NUMINAMATH_CALUDE_expression_simplification_l853_85377

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x * (y^2 + 1) / y) - ((x^2 - 1) / y * (y^3 - 1) / x) =
  (x^3 * y^2 - x^2 * y^3 + x^3 + x^2 + y^2 + y^3) / (x * y) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l853_85377


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l853_85354

open Real

theorem partial_fraction_sum : ∃ (A B C D E : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) ∧
  A + B + C + D + E = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l853_85354


namespace NUMINAMATH_CALUDE_product_difference_implies_sum_l853_85307

theorem product_difference_implies_sum (p q : ℤ) 
  (h1 : p * q = 1764) 
  (h2 : p - q = 20) : 
  p + q = 86 := by
sorry

end NUMINAMATH_CALUDE_product_difference_implies_sum_l853_85307


namespace NUMINAMATH_CALUDE_complex_modulus_and_argument_l853_85359

open Complex

theorem complex_modulus_and_argument : 
  let z : ℂ := -Complex.sin (π/8) - Complex.I * Complex.cos (π/8)
  (abs z = 1) ∧ (arg z = -5*π/8) := by sorry

end NUMINAMATH_CALUDE_complex_modulus_and_argument_l853_85359


namespace NUMINAMATH_CALUDE_goldfish_death_rate_l853_85397

/-- The number of goldfish that die each week -/
def goldfish_deaths_per_week : ℕ := 5

/-- The initial number of goldfish -/
def initial_goldfish : ℕ := 18

/-- The number of goldfish purchased each week -/
def goldfish_purchased_per_week : ℕ := 3

/-- The number of weeks -/
def weeks : ℕ := 7

/-- The final number of goldfish -/
def final_goldfish : ℕ := 4

theorem goldfish_death_rate : 
  initial_goldfish + (goldfish_purchased_per_week * weeks) - (goldfish_deaths_per_week * weeks) = final_goldfish :=
by sorry

end NUMINAMATH_CALUDE_goldfish_death_rate_l853_85397


namespace NUMINAMATH_CALUDE_melody_reading_pages_l853_85366

def english_pages : ℕ := 20
def science_pages : ℕ := 16
def civics_pages : ℕ := 8
def chinese_pages : ℕ := 12

def pages_to_read (total_pages : ℕ) : ℕ := total_pages / 4

theorem melody_reading_pages : 
  pages_to_read english_pages + 
  pages_to_read science_pages + 
  pages_to_read civics_pages + 
  pages_to_read chinese_pages = 14 := by
  sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l853_85366


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l853_85300

theorem quadratic_function_properties (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 1 3 ↔ a * x^2 + b * x + c > -2 * x) →
  (a < 0 ∧
   b = -4 * a - 2 ∧
   (∀ x : ℝ, (a * x^2 + b * x + c + 6 * a = 0 → 
    ∃! r : ℝ, a * r^2 + b * r + c + 6 * a = 0) → a = -1/5)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l853_85300


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l853_85318

def N : ℕ := 18 * 52 * 75 * 98

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N) * 30 = sum_even_divisors N :=
sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l853_85318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l853_85326

/-- Given an arithmetic sequence {a_n} with first term a_1 = 1 and common difference d = 3,
    if a_n = 2005, then n = 669. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) : 
  a 1 = 1 →                                 -- First term is 1
  (∀ k, a (k + 1) - a k = 3) →              -- Common difference is 3
  a n = 2005 →                              -- nth term is 2005
  n = 669 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l853_85326


namespace NUMINAMATH_CALUDE_solve_for_a_l853_85308

theorem solve_for_a : ∃ a : ℝ, (2 * 1 - a * (-1) = 3) ∧ a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l853_85308


namespace NUMINAMATH_CALUDE_double_average_marks_l853_85398

theorem double_average_marks (n : ℕ) (initial_avg : ℝ) (h1 : n = 30) (h2 : initial_avg = 45) :
  let total_marks := n * initial_avg
  let new_total_marks := 2 * total_marks
  let new_avg := new_total_marks / n
  new_avg = 90 := by sorry

end NUMINAMATH_CALUDE_double_average_marks_l853_85398


namespace NUMINAMATH_CALUDE_john_profit_is_13100_l853_85317

/-- Calculates the profit made by John from chopping trees and selling tables -/
def john_profit : ℕ := by
  -- Define the number of trees in each group
  let trees_group1 : ℕ := 10
  let trees_group2 : ℕ := 10
  let trees_group3 : ℕ := 10

  -- Define the number of planks per tree in each group
  let planks_per_tree_group1 : ℕ := 20
  let planks_per_tree_group2 : ℕ := 25
  let planks_per_tree_group3 : ℕ := 30

  -- Define the labor cost per tree in each group
  let labor_cost_group1 : ℕ := 120
  let labor_cost_group2 : ℕ := 80
  let labor_cost_group3 : ℕ := 60

  -- Define the number of planks required to make a table
  let planks_per_table : ℕ := 15

  -- Define the selling price for each group of tables
  let price_tables_1_10 : ℕ := 350
  let price_tables_11_30 : ℕ := 325
  let price_decrease_per_5_tables : ℕ := 10

  -- Calculate the total number of planks
  let total_planks : ℕ := 
    trees_group1 * planks_per_tree_group1 +
    trees_group2 * planks_per_tree_group2 +
    trees_group3 * planks_per_tree_group3

  -- Calculate the total number of tables
  let total_tables : ℕ := total_planks / planks_per_table

  -- Calculate the total labor cost
  let total_labor_cost : ℕ := 
    trees_group1 * labor_cost_group1 +
    trees_group2 * labor_cost_group2 +
    trees_group3 * labor_cost_group3

  -- Calculate the total revenue
  let total_revenue : ℕ := 
    10 * price_tables_1_10 +
    20 * price_tables_11_30 +
    5 * (price_tables_11_30 - price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 2 * price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 3 * price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 4 * price_decrease_per_5_tables)

  -- Calculate the profit
  let profit : ℕ := total_revenue - total_labor_cost

  -- Prove that the profit is equal to 13100
  sorry

theorem john_profit_is_13100 : john_profit = 13100 := by sorry

end NUMINAMATH_CALUDE_john_profit_is_13100_l853_85317


namespace NUMINAMATH_CALUDE_system_solution_l853_85339

theorem system_solution : 
  ∃ (x y u v : ℝ), 
    (5 * x^7 + 3 * y^2 + 5 * u + 4 * v^4 = -2) ∧
    (2 * x^7 + 8 * y^2 + 7 * u + 4 * v^4 = 6^5 / (3^4 * 4^2)) ∧
    (8 * x^7 + 2 * y^2 + 3 * u + 6 * v^4 = -6) ∧
    (5 * x^7 + 7 * y^2 + 7 * u + 8 * v^4 = 8^3 / (2^6 * 4)) ∧
    ((x = -1 ∧ (y = 1 ∨ y = -1) ∧ u = 0 ∧ v = 0) ∨
     (x = 1 ∧ (y = 1 ∨ y = -1) ∧ u = 0 ∧ v = 0)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l853_85339


namespace NUMINAMATH_CALUDE_digit_sum_power_property_l853_85383

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The property that the fifth power of the sum of digits equals the square of the number -/
def has_property (n : ℕ) : Prop := (sum_of_digits n)^5 = n^2

/-- Theorem stating that only 1 and 243 satisfy the property -/
theorem digit_sum_power_property :
  ∀ n : ℕ, has_property n ↔ n = 1 ∨ n = 243 := by sorry

end NUMINAMATH_CALUDE_digit_sum_power_property_l853_85383


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l853_85340

/-- Given two points A(-2, y₁) and B(-1, y₂) on the inverse proportion function y = 2/x,
    prove that y₁ > y₂ -/
theorem inverse_proportion_comparison :
  ∀ y₁ y₂ : ℝ,
  y₁ = 2 / (-2) →
  y₂ = 2 / (-1) →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l853_85340


namespace NUMINAMATH_CALUDE_max_value_abcd_l853_85357

theorem max_value_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^2 * (c + d)^2) ≤ (1 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abcd_l853_85357


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l853_85399

theorem binomial_coefficient_sum (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 3^7 - 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l853_85399


namespace NUMINAMATH_CALUDE_assignment_ways_20_3_l853_85365

/-- The number of ways to assign 3 distinct items from a set of 20 items -/
def assignmentWays (n : ℕ) (k : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem: The number of ways to assign 3 distinct items from a set of 20 items is 6840 -/
theorem assignment_ways_20_3 :
  assignmentWays 20 3 = 6840 := by
  sorry

#eval assignmentWays 20 3

end NUMINAMATH_CALUDE_assignment_ways_20_3_l853_85365


namespace NUMINAMATH_CALUDE_cookie_sugar_measurement_l853_85309

-- Define the amount of sugar needed
def sugar_needed : ℚ := 15/4

-- Define the capacity of the measuring cup
def cup_capacity : ℚ := 1/3

-- Theorem to prove
theorem cookie_sugar_measurement :
  ⌈sugar_needed / cup_capacity⌉ = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sugar_measurement_l853_85309


namespace NUMINAMATH_CALUDE_theta_value_l853_85315

theorem theta_value : ∃! θ : ℕ, θ ∈ Finset.range 10 ∧ θ ≠ 0 ∧ 294 / θ = 30 + 4 * θ := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l853_85315


namespace NUMINAMATH_CALUDE_prob_at_least_two_long_specific_l853_85360

/-- Represents the probability of a road being at least 5 miles long -/
structure RoadProbability where
  ab : ℚ  -- Probability for road A to B
  bc : ℚ  -- Probability for road B to C
  cd : ℚ  -- Probability for road C to D

/-- Calculates the probability of selecting at least two roads that are at least 5 miles long -/
def prob_at_least_two_long (p : RoadProbability) : ℚ :=
  p.ab * p.bc * (1 - p.cd) +  -- A to B and B to C are long, C to D is not
  p.ab * (1 - p.bc) * p.cd +  -- A to B and C to D are long, B to C is not
  (1 - p.ab) * p.bc * p.cd +  -- B to C and C to D are long, A to B is not
  p.ab * p.bc * p.cd          -- All three roads are long

theorem prob_at_least_two_long_specific : 
  let p : RoadProbability := { ab := 3/4, bc := 2/3, cd := 1/2 }
  prob_at_least_two_long p = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_long_specific_l853_85360


namespace NUMINAMATH_CALUDE_tall_trees_indeterminate_l853_85302

/-- Represents the number of trees in the park -/
structure ParkTrees where
  short_current : ℕ
  short_planted : ℕ
  short_after : ℕ
  tall : ℕ

/-- The given information about the trees in the park -/
def park_info : ParkTrees where
  short_current := 41
  short_planted := 57
  short_after := 98
  tall := 0  -- We use 0 as a placeholder since the number is unknown

/-- Theorem stating that the number of tall trees cannot be determined -/
theorem tall_trees_indeterminate (park : ParkTrees) 
    (h1 : park.short_current = park_info.short_current)
    (h2 : park.short_planted = park_info.short_planted)
    (h3 : park.short_after = park_info.short_after)
    (h4 : park.short_after = park.short_current + park.short_planted) :
    ∀ n : ℕ, ∃ p : ParkTrees, p.short_current = park.short_current ∧ 
                               p.short_planted = park.short_planted ∧ 
                               p.short_after = park.short_after ∧ 
                               p.tall = n :=
by sorry

end NUMINAMATH_CALUDE_tall_trees_indeterminate_l853_85302


namespace NUMINAMATH_CALUDE_valentines_given_to_children_l853_85327

/-- The number of Valentines Mrs. Wong had initially -/
def initial_valentines : ℕ := 30

/-- The number of Valentines Mrs. Wong was left with -/
def remaining_valentines : ℕ := 22

/-- The number of Valentines Mrs. Wong gave to her children -/
def given_valentines : ℕ := initial_valentines - remaining_valentines

theorem valentines_given_to_children :
  given_valentines = 8 :=
by sorry

end NUMINAMATH_CALUDE_valentines_given_to_children_l853_85327


namespace NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l853_85329

/-- Given that one kilometer equals 1000 meters, prove that one cubic kilometer equals 1,000,000,000 cubic meters. -/
theorem cubic_kilometer_to_cubic_meters :
  (1 : ℝ) * (kilometer ^ 3) = 1000000000 * (meter ^ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l853_85329


namespace NUMINAMATH_CALUDE_age_ratio_problem_l853_85319

/-- Mike's current age -/
def m : ℕ := sorry

/-- Ana's current age -/
def a : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_problem :
  (m - 3 = 4 * (a - 3)) ∧ 
  (m - 7 = 5 * (a - 7)) →
  x = 77 ∧ 
  (m + x) * 2 = (a + x) * 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l853_85319


namespace NUMINAMATH_CALUDE_fermat_like_equation_power_l853_85345

theorem fermat_like_equation_power (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ l : ℕ, n = p^l :=
by sorry

end NUMINAMATH_CALUDE_fermat_like_equation_power_l853_85345


namespace NUMINAMATH_CALUDE_complement_of_A_l853_85350

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l853_85350


namespace NUMINAMATH_CALUDE_initial_music_files_count_l853_85391

/-- The number of music files Vanessa initially had -/
def initial_music_files : ℕ := sorry

/-- The number of video files Vanessa initially had -/
def initial_video_files : ℕ := 48

/-- The number of files Vanessa deleted -/
def deleted_files : ℕ := 30

/-- The number of files remaining after deletion -/
def remaining_files : ℕ := 34

/-- Theorem stating that the initial number of music files is 16 -/
theorem initial_music_files_count : initial_music_files = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_music_files_count_l853_85391


namespace NUMINAMATH_CALUDE_batsman_average_increase_l853_85331

theorem batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) :
  total_innings = 12 →
  last_innings_score = 65 →
  final_average = 43 →
  (total_innings * final_average - last_innings_score) / (total_innings - 1) = 41 →
  final_average - (total_innings * final_average - last_innings_score) / (total_innings - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l853_85331


namespace NUMINAMATH_CALUDE_distance_after_rest_l853_85335

/-- The length of a football field in meters -/
def football_field_length : ℝ := 168

/-- The distance Nate ran before resting, in meters -/
def distance_before_rest : ℝ := 4 * football_field_length

/-- The total distance Nate ran, in meters -/
def total_distance : ℝ := 1172

/-- Theorem: The distance Nate ran after resting is 500 meters -/
theorem distance_after_rest :
  total_distance - distance_before_rest = 500 := by sorry

end NUMINAMATH_CALUDE_distance_after_rest_l853_85335


namespace NUMINAMATH_CALUDE_function_properties_l853_85358

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * Real.sin (ω * x / 2) ^ 2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem function_properties
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : is_periodic (f ω) (3 * Real.pi))
  (h_min_period : ∀ T, 0 < T ∧ T < 3 * Real.pi → ¬ is_periodic (f ω) T)
  (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_f_C : f ω C = 1)
  (h_trig_eq : 2 * Real.sin (2 * B) = Real.cos B + Real.cos (A - C)) :
  (∃ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), ∀ y ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f ω x ≤ f ω y) ∧
  f ω (Real.pi / 2) = Real.sqrt 3 - 1 ∧
  Real.sin A = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l853_85358


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l853_85312

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

-- Define the union of A and B
def AUnionB : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l853_85312


namespace NUMINAMATH_CALUDE_sins_match_prayers_l853_85361

structure Sin :=
  (teDeum : ℕ)
  (paterNoster : ℕ)
  (credo : ℕ)

def pride : Sin := ⟨1, 2, 0⟩
def slander : Sin := ⟨0, 2, 7⟩
def sloth : Sin := ⟨2, 0, 0⟩
def adultery : Sin := ⟨10, 10, 10⟩
def gluttony : Sin := ⟨1, 0, 0⟩
def selfishness : Sin := ⟨0, 3, 1⟩
def jealousy : Sin := ⟨0, 3, 0⟩
def evilSpeaking : Sin := ⟨0, 7, 2⟩

def totalPrayers (sins : List Sin) : Sin :=
  sins.foldl (λ acc sin => ⟨acc.teDeum + sin.teDeum, acc.paterNoster + sin.paterNoster, acc.credo + sin.credo⟩) ⟨0, 0, 0⟩

theorem sins_match_prayers :
  let sins := [slander] ++ List.replicate 2 evilSpeaking ++ [selfishness] ++ List.replicate 9 gluttony
  totalPrayers sins = ⟨9, 12, 10⟩ := by sorry

end NUMINAMATH_CALUDE_sins_match_prayers_l853_85361


namespace NUMINAMATH_CALUDE_apple_pear_difference_l853_85324

theorem apple_pear_difference :
  let num_apples : ℕ := 17
  let num_pears : ℕ := 9
  num_apples - num_pears = 8 := by sorry

end NUMINAMATH_CALUDE_apple_pear_difference_l853_85324


namespace NUMINAMATH_CALUDE_sum_equals_product_l853_85353

theorem sum_equals_product (x : ℝ) (h : x ≠ 1) :
  ∃! y : ℝ, x + y = x * y ∧ y = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_sum_equals_product_l853_85353


namespace NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l853_85320

variable {A B : Type}
variable (f : A → B)

theorem mapping_not_necessarily_injective : 
  ¬(∀ (x y : A), f x = f y → x = y) :=
sorry

end NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l853_85320


namespace NUMINAMATH_CALUDE_comparison_arithmetic_geometric_mean_l853_85388

theorem comparison_arithmetic_geometric_mean (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬(∀ a b c, (a + b + c) / 3 ≥ (a^2 * b * b * c * c * a)^(1/3)) ∧ 
  ¬(∀ a b c, (a + b + c) / 3 ≤ (a^2 * b * b * c * c * a)^(1/3)) ∧ 
  ¬(∀ a b c, (a + b + c) / 3 = (a^2 * b * b * c * c * a)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_comparison_arithmetic_geometric_mean_l853_85388


namespace NUMINAMATH_CALUDE_triangle_angle_sum_equivalent_to_parallel_postulate_l853_85395

-- Define Euclidean geometry
axiom EuclideanGeometry : Type

-- Define the parallel postulate
axiom parallel_postulate : EuclideanGeometry → Prop

-- Define the triangle angle sum theorem
axiom triangle_angle_sum : EuclideanGeometry → Prop

-- Theorem statement
theorem triangle_angle_sum_equivalent_to_parallel_postulate :
  ∀ (E : EuclideanGeometry), triangle_angle_sum E ↔ parallel_postulate E :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_equivalent_to_parallel_postulate_l853_85395


namespace NUMINAMATH_CALUDE_fraction_equality_l853_85341

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 4 / 3) 
  (h2 : r / t = 9 / 14) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l853_85341


namespace NUMINAMATH_CALUDE_blue_regular_polygon_l853_85396

/-- A circle with some red points and the rest blue -/
structure ColoredCircle where
  redPoints : Finset ℝ
  (red_count : redPoints.card = 2016)

/-- A regular n-gon inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Finset ℝ
  (vertex_count : vertices.card = n)

/-- The theorem statement -/
theorem blue_regular_polygon
  (circle : ColoredCircle)
  (n : ℕ)
  (h : n ≥ 3) :
  ∃ (poly : RegularPolygon n), poly.vertices ∩ circle.redPoints = ∅ :=
sorry

end NUMINAMATH_CALUDE_blue_regular_polygon_l853_85396


namespace NUMINAMATH_CALUDE_order_of_f_values_l853_85376

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem order_of_f_values :
  let a := f (Real.sqrt 2 / 2)
  let b := f (Real.sqrt 3 / 2)
  let c := f (Real.sqrt 6 / 2)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_order_of_f_values_l853_85376


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l853_85301

/-- Given two points A and B in the plane, this theorem states that the equation of the circle
    for which the segment AB is a diameter is (x-1)^2+(y+3)^2=116. -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (-4, -5) →
  B = (6, -1) →
  ∀ x y : ℝ, (x - 1)^2 + (y + 3)^2 = 116 ↔
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    x = -4 * (1 - t) + 6 * t ∧
    y = -5 * (1 - t) - 1 * t :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l853_85301


namespace NUMINAMATH_CALUDE_unique_even_solution_l853_85355

def f (n : ℤ) : ℤ :=
  if n < 0 then n^2 + 4*n + 4 else 3*n - 15

theorem unique_even_solution :
  ∃! a : ℤ, Even a ∧ f (-3) + f 3 + f a = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_even_solution_l853_85355


namespace NUMINAMATH_CALUDE_llama_cost_increase_l853_85316

/-- Proves that the percentage increase in the cost of each llama compared to each goat is 50% -/
theorem llama_cost_increase (goat_cost : ℝ) (total_cost : ℝ) : 
  goat_cost = 400 →
  total_cost = 4800 →
  let num_goats : ℕ := 3
  let num_llamas : ℕ := 2 * num_goats
  let total_goat_cost : ℝ := goat_cost * num_goats
  let total_llama_cost : ℝ := total_cost - total_goat_cost
  let llama_cost : ℝ := total_llama_cost / num_llamas
  (llama_cost - goat_cost) / goat_cost * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_llama_cost_increase_l853_85316


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l853_85374

theorem custom_operation_theorem (a b : ℚ) : 
  a ≠ 0 → b ≠ 0 → a - b = 9 → a / b = 20 → 1 / a + 1 / b = 19 / 60 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l853_85374


namespace NUMINAMATH_CALUDE_practice_time_ratio_l853_85381

/-- Represents the practice time in minutes for each day of the week -/
structure PracticeTime where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the ratio of practice time on Monday to Tuesday is 2:1 -/
theorem practice_time_ratio (p : PracticeTime) : 
  p.thursday = 50 ∧ 
  p.wednesday = p.thursday + 5 ∧ 
  p.tuesday = p.wednesday - 10 ∧ 
  p.friday = 60 ∧ 
  p.monday + p.tuesday + p.wednesday + p.thursday + p.friday = 300 →
  p.monday = 2 * p.tuesday :=
by sorry

end NUMINAMATH_CALUDE_practice_time_ratio_l853_85381


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_l853_85323

theorem angle_with_same_terminal_side : ∃ k : ℤ, 2019 + k * 360 = -141 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_l853_85323


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l853_85314

theorem smallest_prime_divisor_of_sum (p : Nat) : 
  Prime p → p ∣ (2^14 + 7^9) → p > 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l853_85314


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l853_85367

theorem negative_three_less_than_negative_two : -3 < -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l853_85367


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l853_85373

theorem interest_difference_theorem (P : ℝ) : 
  let r : ℝ := 0.04  -- 4% annual interest rate
  let t : ℕ := 2     -- 2 years time period
  let compound_interest := P * (1 + r)^t - P
  let simple_interest := P * r * t
  compound_interest - simple_interest = 1 → P = 625 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l853_85373


namespace NUMINAMATH_CALUDE_fourth_root_logarithm_equality_l853_85349

theorem fourth_root_logarithm_equality : 
  (16 ^ 3) ^ (1/4) - (25/4) ^ (1/2) + (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_logarithm_equality_l853_85349


namespace NUMINAMATH_CALUDE_detergent_per_pound_l853_85333

/-- Given that 18 ounces of detergent are used for 9 pounds of clothes,
    prove that 2 ounces of detergent are used per pound of clothes. -/
theorem detergent_per_pound (total_detergent : ℝ) (total_clothes : ℝ) 
  (h1 : total_detergent = 18) (h2 : total_clothes = 9) :
  total_detergent / total_clothes = 2 := by
  sorry

end NUMINAMATH_CALUDE_detergent_per_pound_l853_85333


namespace NUMINAMATH_CALUDE_first_term_greater_than_2017_l853_85310

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem first_term_greater_than_2017 :
  ∃ n : ℕ, 
    arithmetic_sequence 5 7 n > 2017 ∧
    arithmetic_sequence 5 7 n = 2021 ∧
    ∀ m : ℕ, m < n → arithmetic_sequence 5 7 m ≤ 2017 :=
by sorry

end NUMINAMATH_CALUDE_first_term_greater_than_2017_l853_85310


namespace NUMINAMATH_CALUDE_probability_age_less_than_20_l853_85332

theorem probability_age_less_than_20 (total : ℕ) (age_over_30 : ℕ) (age_under_20 : ℕ) :
  total = 120 →
  age_over_30 = 90 →
  age_under_20 = total - age_over_30 →
  (age_under_20 : ℚ) / (total : ℚ) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_probability_age_less_than_20_l853_85332


namespace NUMINAMATH_CALUDE_min_sum_squares_l853_85348

/-- Parabola defined by y² = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Line passing through (4, 0) -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

/-- Intersection points of the line and parabola -/
def Intersection (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  Parabola x₁ y₁ ∧ Parabola x₂ y₂ ∧ Line k x₁ y₁ ∧ Line k x₂ y₂ ∧ x₁ ≠ x₂

theorem min_sum_squares :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  Intersection k x₁ y₁ x₂ y₂ →
  y₁^2 + y₂^2 ≥ 32 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l853_85348


namespace NUMINAMATH_CALUDE_unwashed_shirts_l853_85311

theorem unwashed_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 40 → long_sleeve = 23 → washed = 29 → 
  short_sleeve + long_sleeve - washed = 34 := by
  sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l853_85311


namespace NUMINAMATH_CALUDE_solution_a_is_correct_l853_85330

/-- The amount of Solution A used in milliliters -/
def solution_a : ℝ := 100

/-- The amount of Solution B used in milliliters -/
def solution_b : ℝ := solution_a + 500

/-- The alcohol percentage in Solution A -/
def alcohol_percent_a : ℝ := 0.16

/-- The alcohol percentage in Solution B -/
def alcohol_percent_b : ℝ := 0.10

/-- The total amount of pure alcohol in the resulting mixture in milliliters -/
def total_pure_alcohol : ℝ := 76

theorem solution_a_is_correct :
  solution_a * alcohol_percent_a + solution_b * alcohol_percent_b = total_pure_alcohol :=
sorry

end NUMINAMATH_CALUDE_solution_a_is_correct_l853_85330


namespace NUMINAMATH_CALUDE_chord_length_is_four_l853_85372

/-- A circle with center at (0, 1) and radius 2, tangent to the line y = -1 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_eq : center = (0, 1)
  radius_eq : radius = 2
  tangent_to_line : ∀ (x y : ℝ), y = -1 → (x - center.1)^2 + (y - center.2)^2 ≥ radius^2

/-- The length of the chord intercepted by the circle on the y-axis -/
def chord_length (c : Circle) : ℝ :=
  let y₁ := c.center.2 + c.radius
  let y₂ := c.center.2 - c.radius
  y₁ - y₂

theorem chord_length_is_four (c : Circle) : chord_length c = 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l853_85372


namespace NUMINAMATH_CALUDE_susan_reading_time_l853_85306

/-- Given Susan's free time activities ratio and time spent with friends, calculate reading time -/
theorem susan_reading_time (swimming reading friends : ℕ) 
  (ratio : swimming + reading + friends = 15) 
  (swim_ratio : swimming = 1)
  (read_ratio : reading = 4)
  (friend_ratio : friends = 10)
  (friend_time : ℕ) 
  (friend_hours : friend_time = 20) : 
  (friend_time * reading) / friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_reading_time_l853_85306


namespace NUMINAMATH_CALUDE_area_lower_bound_l853_85336

/-- A plane convex polygon with given projections -/
structure ConvexPolygon where
  /-- Projection onto OX axis -/
  proj_ox : ℝ
  /-- Projection onto bisector of 1st and 3rd coordinate angles -/
  proj_bisector13 : ℝ
  /-- Projection onto OY axis -/
  proj_oy : ℝ
  /-- Projection onto bisector of 2nd and 4th coordinate angles -/
  proj_bisector24 : ℝ
  /-- Area of the polygon -/
  area : ℝ
  /-- Convexity property (simplified) -/
  convex : True

/-- Theorem: The area of a convex polygon with given projections is at least 10 -/
theorem area_lower_bound (p : ConvexPolygon)
  (h1 : p.proj_ox = 4)
  (h2 : p.proj_bisector13 = 3 * Real.sqrt 2)
  (h3 : p.proj_oy = 5)
  (h4 : p.proj_bisector24 = 4 * Real.sqrt 2) :
  p.area ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_area_lower_bound_l853_85336


namespace NUMINAMATH_CALUDE_binary_calculation_l853_85322

-- Define binary numbers as natural numbers
def bin110110 : ℕ := 54  -- 110110 in base 2 is 54 in base 10
def bin101110 : ℕ := 46  -- 101110 in base 2 is 46 in base 10
def bin100 : ℕ := 4      -- 100 in base 2 is 4 in base 10
def bin11100011110 : ℕ := 1886  -- 11100011110 in base 2 is 1886 in base 10

-- State the theorem
theorem binary_calculation :
  (bin110110 / bin100) * bin101110 = bin11100011110 := by
  sorry

end NUMINAMATH_CALUDE_binary_calculation_l853_85322


namespace NUMINAMATH_CALUDE_white_l_shapes_imply_all_white_2x2_l853_85363

/-- Represents a grid cell that can be either black or white -/
inductive Color
| Black
| White

/-- Represents an m × n grid -/
def Grid (m n : ℕ) := Fin m → Fin n → Color

/-- Counts the number of L-shapes with exactly three white squares in a grid -/
def countWhiteLShapes (g : Grid m n) : ℕ := sorry

/-- Checks if there exists a 2 × 2 grid with all white squares -/
def existsAllWhite2x2 (g : Grid m n) : Prop := sorry

/-- Main theorem: If the number of L-shapes with three white squares is at least mn/3,
    then there exists a 2 × 2 grid with all white squares -/
theorem white_l_shapes_imply_all_white_2x2 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (g : Grid m n) :
  countWhiteLShapes g ≥ m * n / 3 → existsAllWhite2x2 g :=
sorry

end NUMINAMATH_CALUDE_white_l_shapes_imply_all_white_2x2_l853_85363


namespace NUMINAMATH_CALUDE_fraction_multiplication_l853_85370

theorem fraction_multiplication : (2 : ℚ) / 5 * (7 : ℚ) / 10 = (7 : ℚ) / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l853_85370


namespace NUMINAMATH_CALUDE_total_weight_is_5040_l853_85378

/-- The weight of all settings for a catering event. -/
def total_weight_of_settings : ℕ :=
  let silverware_weight_per_piece : ℕ := 4
  let silverware_pieces_per_setting : ℕ := 3
  let plate_weight : ℕ := 12
  let plates_per_setting : ℕ := 2
  let tables : ℕ := 15
  let settings_per_table : ℕ := 8
  let backup_settings : ℕ := 20
  
  let total_settings : ℕ := tables * settings_per_table + backup_settings
  let weight_per_setting : ℕ := silverware_weight_per_piece * silverware_pieces_per_setting + 
                                 plate_weight * plates_per_setting
  
  total_settings * weight_per_setting

theorem total_weight_is_5040 : total_weight_of_settings = 5040 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_5040_l853_85378


namespace NUMINAMATH_CALUDE_bruce_mangoes_purchase_l853_85342

theorem bruce_mangoes_purchase :
  let grapes_kg : ℕ := 8
  let grapes_price : ℕ := 70
  let mango_price : ℕ := 55
  let total_paid : ℕ := 1165
  let mango_kg : ℕ := (total_paid - grapes_kg * grapes_price) / mango_price
  mango_kg = 11 := by sorry

end NUMINAMATH_CALUDE_bruce_mangoes_purchase_l853_85342


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l853_85338

theorem fixed_point_of_exponential_function 
  (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 1
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l853_85338


namespace NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l853_85382

theorem inscribed_circle_square_side_length 
  (circle_area : ℝ) 
  (h_area : circle_area = 36 * Real.pi) : 
  ∃ (square_side : ℝ), 
    square_side = 12 ∧ 
    circle_area = Real.pi * (square_side / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l853_85382


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l853_85347

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 6/5 of a right angle
  a + b = 6/5 * 90 →
  -- One angle is 30° larger than the other
  b = a + 30 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 72°
  max a (max b c) = 72 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l853_85347


namespace NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l853_85321

/-- Given a bowl of sweets with red, green, and other colors, calculate the number of sweets that are neither red nor green. -/
theorem sweets_neither_red_nor_green 
  (total : ℕ) 
  (red : ℕ) 
  (green : ℕ) 
  (h_total : total = 285) 
  (h_red : red = 49) 
  (h_green : green = 59) :
  total - (red + green) = 177 := by
  sorry

end NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l853_85321


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l853_85390

/-- Given an arithmetic sequence where:
  a₁ = 3 (first term)
  a₂ = 7 (second term)
  a₃ = 11 (third term)
  Prove that a₅ = 19 (fifth term)
-/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℤ), 
    (a 1 = 3) →  -- First term
    (a 2 = 7) →  -- Second term
    (a 3 = 11) → -- Third term
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → -- Arithmetic sequence property
    a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l853_85390


namespace NUMINAMATH_CALUDE_john_biking_distance_john_biking_distance_proof_l853_85325

theorem john_biking_distance (bike_speed walking_speed : ℝ) 
  (walking_distance total_time : ℝ) : ℝ :=
  let total_biking_distance := 
    (total_time - walking_distance / walking_speed) * bike_speed + walking_distance
  total_biking_distance

#check john_biking_distance 15 4 3 (7/6) = 9.25

theorem john_biking_distance_proof :
  john_biking_distance 15 4 3 (7/6) = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_john_biking_distance_john_biking_distance_proof_l853_85325


namespace NUMINAMATH_CALUDE_grid_bottom_right_value_l853_85352

/-- Represents a 3x3 grid with some known values -/
structure Grid :=
  (a b c d e f g h i : ℕ)
  (b_eq_6 : b = 6)
  (c_eq_3 : c = 3)
  (h_eq_2 : h = 2)
  (all_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0)

/-- The product of each row, column, and diagonal is the same -/
def grid_property (grid : Grid) : Prop :=
  let p := grid.a * grid.b * grid.c
  p = grid.d * grid.e * grid.f ∧
  p = grid.g * grid.h * grid.i ∧
  p = grid.a * grid.d * grid.g ∧
  p = grid.b * grid.e * grid.h ∧
  p = grid.c * grid.f * grid.i ∧
  p = grid.a * grid.e * grid.i ∧
  p = grid.c * grid.e * grid.g

theorem grid_bottom_right_value (grid : Grid) (h : grid_property grid) : grid.i = 36 := by
  sorry

end NUMINAMATH_CALUDE_grid_bottom_right_value_l853_85352


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l853_85379

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  let c := Real.sqrt (a^2 + b^2)
  let area := (1/2) * a * b
  area = 35 := by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l853_85379


namespace NUMINAMATH_CALUDE_problem_solution_l853_85387

def A : Set ℝ := {x | (x - 1/2) * (x - 3) = 0}

def B (a : ℝ) : Set ℝ := {x | Real.log (x^2 + a*x + a + 9/4) = 0}

theorem problem_solution :
  (∀ a : ℝ, (∃! x : ℝ, x ∈ B a) → (a = 5 ∨ a = -1)) ∧
  (∀ a : ℝ, B a ⊂ A → a ∈ Set.Icc (-1) 5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l853_85387


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_ten_l853_85351

def is_smallest_square_multiplier (y : ℕ) (n : ℕ) : Prop :=
  y > 0 ∧ ∃ (m : ℕ), y * n = m^2 ∧
  ∀ (k : ℕ), k > 0 → k < y → ¬∃ (m : ℕ), k * n = m^2

theorem smallest_square_multiplier_ten (n : ℕ) :
  is_smallest_square_multiplier 10 n → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_ten_l853_85351


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l853_85346

theorem absolute_value_inequality (x : ℝ) : 
  3 ≤ |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l853_85346


namespace NUMINAMATH_CALUDE_right_triangles_count_l853_85303

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if a triangle is right-angled with vertex c as the right angle -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Checks if a point is a lattice point (has integer coordinates) -/
def isLatticePoint (p : Point) : Prop := sorry

/-- Calculates the incenter of a triangle -/
def incenter (t : Triangle) : Point := sorry

/-- Counts the number of right triangles satisfying the given conditions -/
def countRightTriangles (p : ℕ) (isPrime : Nat.Prime p) : ℕ := sorry

/-- The main theorem -/
theorem right_triangles_count (p : ℕ) (isPrime : Nat.Prime p) :
  let m := Point.mk (p * 1994) (7 * p * 1994)
  countRightTriangles p isPrime =
    if p = 2 then 18
    else if p = 997 then 20
    else 36 := by
  sorry

end NUMINAMATH_CALUDE_right_triangles_count_l853_85303


namespace NUMINAMATH_CALUDE_expression_range_l853_85369

/-- The quadratic equation in terms of x with parameter m -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m-2)*x + m^2 + 4

/-- Predicate to check if the quadratic equation has two real roots -/
def has_two_real_roots (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

/-- The expression we want to find the range of -/
def expression (x₁ x₂ : ℝ) : ℝ := x₁^2 + x₂^2 - x₁*x₂

theorem expression_range :
  ∀ m : ℝ, has_two_real_roots m →
    (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ 
      expression x₁ x₂ ≥ 4 ∧ 
      ∀ ε > 0, ∃ m' : ℝ, has_two_real_roots m' ∧ 
        ∃ y₁ y₂ : ℝ, quadratic m' y₁ = 0 ∧ quadratic m' y₂ = 0 ∧ 
          expression y₁ y₂ < 4 + ε) :=
sorry

end NUMINAMATH_CALUDE_expression_range_l853_85369


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l853_85389

theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 70 ∧ 
  length > breadth ∧ 
  26.50 * (2 * length + 2 * breadth) = 5300 → 
  length - breadth = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l853_85389


namespace NUMINAMATH_CALUDE_candle_arrangement_l853_85362

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of candles that satisfies the given conditions -/
def num_candles : ℕ := 4

theorem candle_arrangement :
  (∀ c : ℕ, (choose_2 c * 9 = 54) → c = num_candles) :=
by sorry

end NUMINAMATH_CALUDE_candle_arrangement_l853_85362


namespace NUMINAMATH_CALUDE_elvis_songwriting_time_l853_85328

/-- Given Elvis's album recording scenario, prove that the time spent writing each song is 15 minutes. -/
theorem elvis_songwriting_time (total_songs : ℕ) (studio_time : ℕ) (recording_time_per_song : ℕ) (total_editing_time : ℕ) :
  total_songs = 10 →
  studio_time = 5 * 60 →
  recording_time_per_song = 12 →
  total_editing_time = 30 →
  (studio_time - (total_songs * recording_time_per_song + total_editing_time)) / total_songs = 15 :=
by sorry

end NUMINAMATH_CALUDE_elvis_songwriting_time_l853_85328


namespace NUMINAMATH_CALUDE_balls_per_bag_l853_85343

theorem balls_per_bag (total_balls : ℕ) (num_bags : ℕ) (balls_per_bag : ℕ) 
  (h1 : total_balls = 36)
  (h2 : num_bags = 9)
  (h3 : total_balls = num_bags * balls_per_bag) :
  balls_per_bag = 4 := by
sorry

end NUMINAMATH_CALUDE_balls_per_bag_l853_85343


namespace NUMINAMATH_CALUDE_gcd_84_126_l853_85394

theorem gcd_84_126 : Nat.gcd 84 126 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_126_l853_85394


namespace NUMINAMATH_CALUDE_cubic_function_property_l853_85313

/-- Given a cubic function f(x) = ax³ + bx + 8 where f(-2) = 10, prove that f(2) = 6 -/
theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l853_85313


namespace NUMINAMATH_CALUDE_odell_kershaw_passing_l853_85337

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- track radius in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing :
  let odell : Runner := { speed := 260, radius := 55, direction := 1 }
  let kershaw : Runner := { speed := 310, radius := 65, direction := -1 }
  passingCount odell kershaw 35 = 52 :=
sorry

end NUMINAMATH_CALUDE_odell_kershaw_passing_l853_85337
