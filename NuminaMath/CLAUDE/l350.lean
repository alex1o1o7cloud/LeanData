import Mathlib

namespace NUMINAMATH_CALUDE_unique_base_representation_l350_35066

/-- The repeating base-k representation of a rational number -/
def repeatingBaseK (n d k : ℕ) : ℚ :=
  (4 : ℚ) / k + (7 : ℚ) / k^2

/-- The condition for the repeating base-k representation to equal the given fraction -/
def isValidK (k : ℕ) : Prop :=
  k > 0 ∧ repeatingBaseK 11 77 k = 11 / 77

theorem unique_base_representation :
  ∃! k : ℕ, isValidK k ∧ k = 17 :=
sorry

end NUMINAMATH_CALUDE_unique_base_representation_l350_35066


namespace NUMINAMATH_CALUDE_two_points_explain_phenomena_l350_35024

-- Define the type for phenomena
inductive Phenomenon : Type
| RiverChannel
| WoodenStrips
| TreePlanting
| WallFixing

-- Define a function to check if a phenomenon can be explained by "two points determine a straight line"
def explainedByTwoPoints : Phenomenon → Prop
| Phenomenon.RiverChannel => false
| Phenomenon.WoodenStrips => true
| Phenomenon.TreePlanting => true
| Phenomenon.WallFixing => true

-- State the theorem
theorem two_points_explain_phenomena :
  (explainedByTwoPoints Phenomenon.RiverChannel = false) ∧
  (explainedByTwoPoints Phenomenon.WoodenStrips = true) ∧
  (explainedByTwoPoints Phenomenon.TreePlanting = true) ∧
  (explainedByTwoPoints Phenomenon.WallFixing = true) :=
by sorry

end NUMINAMATH_CALUDE_two_points_explain_phenomena_l350_35024


namespace NUMINAMATH_CALUDE_pairwise_sums_not_distinct_l350_35005

theorem pairwise_sums_not_distinct (n : ℕ+) (A : Finset (ZMod n)) :
  A.card > 1 + Real.sqrt (n + 4) →
  ∃ (a b c d : ZMod n), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c) ∧ a + b = c + d :=
by sorry

end NUMINAMATH_CALUDE_pairwise_sums_not_distinct_l350_35005


namespace NUMINAMATH_CALUDE_percentage_married_employees_l350_35068

theorem percentage_married_employees (total : ℝ) (total_pos : 0 < total) : 
  let women_ratio : ℝ := 0.76
  let men_ratio : ℝ := 1 - women_ratio
  let married_women_ratio : ℝ := 0.6842
  let single_men_ratio : ℝ := 2/3
  let married_men_ratio : ℝ := 1 - single_men_ratio
  let married_ratio : ℝ := women_ratio * married_women_ratio + men_ratio * married_men_ratio
  married_ratio = 0.600392 :=
sorry

end NUMINAMATH_CALUDE_percentage_married_employees_l350_35068


namespace NUMINAMATH_CALUDE_inequality_condition_l350_35065

theorem inequality_condition (a b : ℝ) :
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l350_35065


namespace NUMINAMATH_CALUDE_sin_double_angle_when_tan_is_half_l350_35092

theorem sin_double_angle_when_tan_is_half (α : Real) (h : Real.tan α = 1/2) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_when_tan_is_half_l350_35092


namespace NUMINAMATH_CALUDE_min_value_expression_l350_35039

theorem min_value_expression (m n : ℝ) (h1 : m > 1) (h2 : n > 0) (h3 : m^2 - 3*m + n = 0) :
  ∃ (min_val : ℝ), min_val = 9/2 ∧ ∀ (x : ℝ), (4/(m-1) + m/n) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l350_35039


namespace NUMINAMATH_CALUDE_puzzle_solution_l350_35057

/-- Represents the pieces of the puzzle -/
inductive Piece
| Two
| One
| Zero
| Minus

/-- Represents the arrangement of pieces -/
def Arrangement := List Piece

/-- Checks if an arrangement forms a valid subtraction equation -/
def isValidArrangement (arr : Arrangement) : Prop := sorry

/-- Calculates the result of a valid arrangement -/
def calculateResult (arr : Arrangement) : Int := sorry

/-- The main theorem: The correct arrangement results in -100 -/
theorem puzzle_solution :
  ∃ (arr : Arrangement),
    isValidArrangement arr ∧ calculateResult arr = -100 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l350_35057


namespace NUMINAMATH_CALUDE_frame_diameter_l350_35037

theorem frame_diameter (d_y : ℝ) (uncovered_fraction : ℝ) (d_x : ℝ) : 
  d_y = 12 →
  uncovered_fraction = 0.4375 →
  d_x = 16 →
  (π * (d_x / 2)^2) = (π * (d_y / 2)^2) + uncovered_fraction * (π * (d_x / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_frame_diameter_l350_35037


namespace NUMINAMATH_CALUDE_service_provider_assignment_l350_35055

theorem service_provider_assignment (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 :=
by sorry

end NUMINAMATH_CALUDE_service_provider_assignment_l350_35055


namespace NUMINAMATH_CALUDE_equation_solution_l350_35076

theorem equation_solution : ∃ x : ℝ, 15 * 2 = 3 + x ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l350_35076


namespace NUMINAMATH_CALUDE_tangent_line_intersection_at_minus_one_range_of_a_l350_35077

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x^2 + a, where a is a parameter -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g_derivative (x : ℝ) : ℝ := 2 * x

/-- Theorem stating that when x₁ = -1, a = 3 -/
theorem tangent_line_intersection_at_minus_one (a : ℝ) :
  (∃ x₂ : ℝ, f_derivative (-1) = g_derivative x₂ ∧ 
    f (-1) - f_derivative (-1) * (-1) = g a x₂ - g_derivative x₂ * x₂) →
  a = 3 :=
sorry

/-- Theorem stating that a ≥ -1 -/
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f_derivative x₁ = g_derivative x₂ ∧ 
    f x₁ - f_derivative x₁ * x₁ = g a x₂ - g_derivative x₂ * x₂) →
  a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_at_minus_one_range_of_a_l350_35077


namespace NUMINAMATH_CALUDE_union_of_sets_l350_35019

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 2, 3}
  let B : Set ℕ := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l350_35019


namespace NUMINAMATH_CALUDE_total_cost_of_all_lawns_l350_35085

structure Lawn where
  length : ℕ
  breadth : ℕ
  lengthRoadWidth : ℕ
  breadthRoadWidth : ℕ
  costPerSqMeter : ℕ

def totalRoadArea (l : Lawn) : ℕ :=
  l.length * l.lengthRoadWidth + l.breadth * l.breadthRoadWidth

def totalCost (l : Lawn) : ℕ :=
  totalRoadArea l * l.costPerSqMeter

def lawnA : Lawn := ⟨80, 70, 8, 6, 3⟩
def lawnB : Lawn := ⟨120, 50, 12, 10, 4⟩
def lawnC : Lawn := ⟨150, 90, 15, 9, 5⟩

theorem total_cost_of_all_lawns :
  totalCost lawnA + totalCost lawnB + totalCost lawnC = 26240 := by
  sorry

#eval totalCost lawnA + totalCost lawnB + totalCost lawnC

end NUMINAMATH_CALUDE_total_cost_of_all_lawns_l350_35085


namespace NUMINAMATH_CALUDE_solve_for_a_l350_35074

theorem solve_for_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 18 - 6 * a) : a = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l350_35074


namespace NUMINAMATH_CALUDE_mateen_backyard_area_l350_35016

/-- A rectangular backyard with specific walking distances -/
structure Backyard where
  length : ℝ
  width : ℝ
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ

/-- The conditions of Mateen's backyard -/
def mateen_backyard : Backyard where
  length := 40
  width := 10
  total_distance := 1200
  length_walks := 30
  perimeter_walks := 12

/-- Theorem stating the area of Mateen's backyard -/
theorem mateen_backyard_area :
  let b := mateen_backyard
  b.length * b.width = 400 ∧
  b.length_walks * b.length = b.total_distance ∧
  b.perimeter_walks * (2 * b.length + 2 * b.width) = b.total_distance :=
by sorry

end NUMINAMATH_CALUDE_mateen_backyard_area_l350_35016


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_product_squares_l350_35054

theorem consecutive_odd_numbers_sum_product_squares : 
  ∃ (a : ℤ), 
    let sequence := List.range 25 |>.map (λ i => a + 2*i - 24)
    ∃ (s p : ℤ), 
      (sequence.sum = s^2) ∧ 
      (sequence.prod = p^2) ∧
      (∀ n ∈ sequence, n % 2 = 1 ∨ n % 2 = -1) := by
  sorry

#check consecutive_odd_numbers_sum_product_squares

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_product_squares_l350_35054


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l350_35042

/-- Calculates the amount after two years of compound interest with different rates for each year. -/
def amountAfterTwoYears (initialAmount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amountAfterFirstYear := initialAmount * (1 + rate1)
  amountAfterFirstYear * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated. -/
theorem compound_interest_calculation (initialAmount : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : initialAmount = 9828) 
  (h2 : rate1 = 0.04) 
  (h3 : rate2 = 0.05) :
  amountAfterTwoYears initialAmount rate1 rate2 = 10732.176 := by
  sorry

#eval amountAfterTwoYears 9828 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l350_35042


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l350_35086

theorem quadratic_equation_roots (m : ℝ) :
  (2 * (2 : ℝ)^2 - 5 * 2 - m = 0) →
  (m = -2 ∧ ∃ (x : ℝ), x ≠ 2 ∧ 2 * x^2 - 5 * x - m = 0 ∧ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l350_35086


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l350_35070

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 52 → x*y + y*z + z*x = 27 → x + y + z = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l350_35070


namespace NUMINAMATH_CALUDE_probability_six_spades_correct_l350_35008

/-- The number of cards in a standard deck of poker cards (excluding jokers) -/
def deck_size : ℕ := 52

/-- The number of spades in a standard deck of poker cards -/
def spades_count : ℕ := 13

/-- The number of cards each player receives when 4 people play -/
def cards_per_player : ℕ := deck_size / 4

/-- The probability of a person getting exactly 6 spades when 4 people play with a standard deck -/
def probability_six_spades : ℚ :=
  (Nat.choose spades_count 6 * Nat.choose (deck_size - spades_count) (cards_per_player - 6)) /
  Nat.choose deck_size cards_per_player

theorem probability_six_spades_correct :
  probability_six_spades = (Nat.choose 13 6 * Nat.choose 39 7) / Nat.choose 52 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_spades_correct_l350_35008


namespace NUMINAMATH_CALUDE_max_value_of_f_l350_35051

-- Define the parabola function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 3

-- Theorem: The maximum value of f is 3
theorem max_value_of_f : ∀ x : ℝ, f x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l350_35051


namespace NUMINAMATH_CALUDE_wax_sculpture_problem_l350_35004

theorem wax_sculpture_problem (large_animal_wax : ℕ) (small_animal_wax : ℕ) 
  (small_animal_total_wax : ℕ) (total_wax : ℕ) :
  large_animal_wax = 4 →
  small_animal_wax = 2 →
  small_animal_total_wax = 12 →
  total_wax = 20 →
  total_wax = small_animal_total_wax + (total_wax - small_animal_total_wax) :=
by sorry

end NUMINAMATH_CALUDE_wax_sculpture_problem_l350_35004


namespace NUMINAMATH_CALUDE_base_b_divisibility_l350_35062

theorem base_b_divisibility (b : ℤ) : b = 7 ↔ ¬(5 ∣ (b^2 * (3*b - 2))) ∧ 
  (b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 → 5 ∣ (b^2 * (3*b - 2))) := by
  sorry

end NUMINAMATH_CALUDE_base_b_divisibility_l350_35062


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l350_35001

theorem circle_line_distance_range (a : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x + y = a}
  let distance_to_line (p : ℝ × ℝ) := |p.1 + p.2 - a| / Real.sqrt 2
  (∃ p1 p2 : ℝ × ℝ, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2 ∧ 
    distance_to_line p1 = 1 ∧ distance_to_line p2 = 1) →
  a ∈ Set.Ioo (-3 * Real.sqrt 2) (3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l350_35001


namespace NUMINAMATH_CALUDE_gcd_228_1995_l350_35007

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l350_35007


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l350_35025

theorem constant_term_binomial_expansion :
  (Finset.sum (Finset.range 10) (fun k => Nat.choose 9 k * (1 : ℝ)^k * (1 : ℝ)^(9 - k))) = 84 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l350_35025


namespace NUMINAMATH_CALUDE_age_difference_l350_35046

theorem age_difference (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →  -- Ensuring a and b are single digits
  (10 * a + b) + 5 = 3 * ((10 * b + a) + 5) → -- Condition after 5 years
  (10 * a + b) - (10 * b + a) = 63 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l350_35046


namespace NUMINAMATH_CALUDE_sqrt2_fractional_part_bounds_l350_35036

theorem sqrt2_fractional_part_bounds :
  (∀ n : ℕ, n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ > 1 / (2 * n * Real.sqrt 2)) ∧
  (∀ ε > 0, ∃ n : ℕ, n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ < 1 / (2 * n * Real.sqrt 2) + ε) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_fractional_part_bounds_l350_35036


namespace NUMINAMATH_CALUDE_triangular_grid_edges_l350_35027

theorem triangular_grid_edges (n : ℕ) (h : n = 1001) : 
  let total_squares := n * (n + 1) / 2
  let total_edges_without_sharing := 4 * total_squares
  let shared_edges := (n - 1) * n / 2 - 1
  total_edges_without_sharing - 2 * shared_edges = 1006004 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_edges_l350_35027


namespace NUMINAMATH_CALUDE_special_function_18_48_l350_35072

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ+ → ℕ+) : Prop :=
  (∀ x : ℕ+, f x x = x) ∧
  (∀ x y : ℕ+, f x y = f y x) ∧
  (∀ x y : ℕ+, (x + y) * (f x y) = x * (f x (x + y)))

/-- The main theorem -/
theorem special_function_18_48 (f : ℕ+ → ℕ+ → ℕ+) (h : special_function f) :
  f 18 48 = 48 := by
  sorry

end NUMINAMATH_CALUDE_special_function_18_48_l350_35072


namespace NUMINAMATH_CALUDE_mrs_hilts_snow_amount_l350_35038

def snow_at_mrs_hilts_house : ℕ := 29
def snow_at_brecknock_school : ℕ := 17

theorem mrs_hilts_snow_amount : snow_at_mrs_hilts_house = 29 := by sorry

end NUMINAMATH_CALUDE_mrs_hilts_snow_amount_l350_35038


namespace NUMINAMATH_CALUDE_house_distance_ratio_l350_35064

/-- Given three points on a road representing houses, proves the ratio of distances -/
theorem house_distance_ratio (K D M : ℝ) : 
  let KD := |K - D|
  let DM := |D - M|
  KD = 4 → KD + DM + DM + KD = 12 → KD / DM = 2 := by
  sorry

end NUMINAMATH_CALUDE_house_distance_ratio_l350_35064


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l350_35045

theorem apple_pear_equivalence (apple_value pear_value : ℚ) :
  (3/4 : ℚ) * 12 * apple_value = 10 * pear_value →
  (2/3 : ℚ) * 9 * apple_value = (20/3 : ℚ) * pear_value :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l350_35045


namespace NUMINAMATH_CALUDE_x₂_integer_part_sum_of_arctans_l350_35000

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 - 17*x - 18

-- Define the roots and their properties
axiom x₁ : ℝ
axiom x₂ : ℝ
axiom x₃ : ℝ
axiom x₁_range : -4 < x₁ ∧ x₁ < -3
axiom x₃_range : 4 < x₃ ∧ x₃ < 5
axiom roots_property : cubic_equation x₁ = 0 ∧ cubic_equation x₂ = 0 ∧ cubic_equation x₃ = 0

-- Theorem for the integer part of x₂
theorem x₂_integer_part : ⌊x₂⌋ = -2 := by sorry

-- Theorem for the sum of arctangents
theorem sum_of_arctans : Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = -π/4 := by sorry

end NUMINAMATH_CALUDE_x₂_integer_part_sum_of_arctans_l350_35000


namespace NUMINAMATH_CALUDE_unique_solution_range_l350_35048

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (a x : ℝ) : Prop :=
  lg (a * x + 1) = lg (x - 1) + lg (2 - x)

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  (a > -1 ∧ a ≤ -1/2) ∨ a = 3 - 2 * Real.sqrt 3

-- Theorem statement
theorem unique_solution_range :
  ∀ a : ℝ, (∃! x : ℝ, equation a x) ↔ a_range a := by sorry

end NUMINAMATH_CALUDE_unique_solution_range_l350_35048


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l350_35033

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - x + a > 0) → a > (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l350_35033


namespace NUMINAMATH_CALUDE_isosceles_iff_equal_angle_bisectors_l350_35075

/-- Given a triangle with sides a, b, c, and angle bisectors l_α and l_β, 
    prove that the triangle is isosceles (a = b) if and only if l_α = l_β -/
theorem isosceles_iff_equal_angle_bisectors 
  (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (l_α : ℝ := (1 / (b + c)) * Real.sqrt (b * c * ((b + c)^2 - a^2)))
  (l_β : ℝ := (1 / (c + a)) * Real.sqrt (c * a * ((c + a)^2 - b^2))) :
  a = b ↔ l_α = l_β := by
  sorry

end NUMINAMATH_CALUDE_isosceles_iff_equal_angle_bisectors_l350_35075


namespace NUMINAMATH_CALUDE_complex_multiplication_l350_35059

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 - i) * (-2 + i) = -3 + 4*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l350_35059


namespace NUMINAMATH_CALUDE_inequality_proof_l350_35098

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l350_35098


namespace NUMINAMATH_CALUDE_fraction_increase_possible_l350_35050

theorem fraction_increase_possible : ∃ (a b : ℕ+), (a + 1 : ℚ) / (b + 100) > (a : ℚ) / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_possible_l350_35050


namespace NUMINAMATH_CALUDE_mayor_approval_probability_l350_35003

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem mayor_approval_probability :
  binomial_probability p n k = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_mayor_approval_probability_l350_35003


namespace NUMINAMATH_CALUDE_nicky_card_trade_loss_l350_35083

/-- Calculates the profit or loss from a card trade with tax -/
def card_trade_profit (
  cards_given_value1 : ℝ)
  (cards_given_count1 : ℕ)
  (cards_given_value2 : ℝ)
  (cards_given_count2 : ℕ)
  (cards_received_value1 : ℝ)
  (cards_received_count1 : ℕ)
  (cards_received_value2 : ℝ)
  (cards_received_count2 : ℕ)
  (tax_rate : ℝ) : ℝ :=
  let total_given := cards_given_value1 * cards_given_count1 + cards_given_value2 * cards_given_count2
  let total_received := cards_received_value1 * cards_received_count1 + cards_received_value2 * cards_received_count2
  let total_trade_value := total_given + total_received
  let tax := tax_rate * total_trade_value
  total_received - total_given - tax

theorem nicky_card_trade_loss :
  card_trade_profit 8 2 5 3 21 1 6 2 0.05 = -1.20 := by
  sorry

end NUMINAMATH_CALUDE_nicky_card_trade_loss_l350_35083


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l350_35079

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l350_35079


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l350_35041

/-- Represents a three-digit number where the first and last digits are the same -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.a

theorem three_digit_divisibility_by_seven (n : ThreeDigitNumber) :
  (n.toNum % 7 = 0) ↔ ((n.a + n.b) % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l350_35041


namespace NUMINAMATH_CALUDE_min_m_plus_n_l350_35052

theorem min_m_plus_n (m n : ℕ+) (h : 75 * m = n^3) : 
  ∀ (m' n' : ℕ+), 75 * m' = n'^3 → m + n ≤ m' + n' :=
by sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l350_35052


namespace NUMINAMATH_CALUDE_number_rewriting_l350_35043

theorem number_rewriting :
  (29800000 = 2980 * 10000) ∧ (14000000000 = 140 * 100000000) := by
  sorry

end NUMINAMATH_CALUDE_number_rewriting_l350_35043


namespace NUMINAMATH_CALUDE_fewer_puzzles_than_kits_difference_is_nine_l350_35095

/-- The Smart Mart sells educational toys -/
structure SmartMart where
  science_kits : ℕ
  puzzles : ℕ

/-- The number of science kits sold is 45 -/
def science_kits_sold : ℕ := 45

/-- The number of puzzles sold is 36 -/
def puzzles_sold : ℕ := 36

/-- The Smart Mart sold fewer puzzles than science kits -/
theorem fewer_puzzles_than_kits (sm : SmartMart) :
  sm.puzzles < sm.science_kits :=
sorry

/-- The difference between science kits and puzzles sold is 9 -/
theorem difference_is_nine (sm : SmartMart) 
  (h1 : sm.science_kits = science_kits_sold) 
  (h2 : sm.puzzles = puzzles_sold) : 
  sm.science_kits - sm.puzzles = 9 :=
sorry

end NUMINAMATH_CALUDE_fewer_puzzles_than_kits_difference_is_nine_l350_35095


namespace NUMINAMATH_CALUDE_complex_subtraction_l350_35069

theorem complex_subtraction : (4 : ℂ) - 3*I - ((2 : ℂ) + 5*I) = (2 : ℂ) - 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l350_35069


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l350_35022

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 6 + a 7 = 15) : 
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l350_35022


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l350_35091

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x - (x + 1)^2

theorem extreme_value_and_range :
  (∃ x : ℝ, ∀ y : ℝ, f (-1) y ≤ f (-1) x ∧ f (-1) x = 1 / Real.exp 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≤ 0) ↔ a ∈ Set.Icc 0 (4 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l350_35091


namespace NUMINAMATH_CALUDE_product_325_3_base7_l350_35010

-- Define a function to convert from base 7 to base 10
def base7ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 7
def base10ToBase7 (n : ℕ) : ℕ := sorry

-- Define the multiplication operation in base 7
def multBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a * base7ToBase10 b)

-- State the theorem
theorem product_325_3_base7 : 
  multBase7 325 3 = 3111 := by sorry

end NUMINAMATH_CALUDE_product_325_3_base7_l350_35010


namespace NUMINAMATH_CALUDE_find_lighter_orange_l350_35081

/-- Represents a group of objects that can be weighed -/
structure WeightGroup where
  objects : Finset ℕ
  size : ℕ
  h_size : objects.card = size

/-- Represents the result of weighing two groups -/
inductive WeighResult
  | Left
  | Right
  | Equal

/-- Represents a balance scale that can compare two groups -/
def Balance := WeightGroup → WeightGroup → WeighResult

/-- The problem setup with 8 objects, 7 of equal weight and 1 lighter -/
structure OrangeSetup where
  total_objects : ℕ
  h_total : total_objects = 8
  equal_weight_objects : ℕ
  h_equal : equal_weight_objects = 7
  h_lighter : total_objects = equal_weight_objects + 1

/-- The theorem stating that the lighter object can be found in at most 2 measurements -/
theorem find_lighter_orange (setup : OrangeSetup) :
  ∃ (strategy : Balance → Balance → ℕ),
    ∀ (b : Balance), strategy b b < setup.total_objects ∧ 
    (strategy b b) ∈ Finset.range setup.total_objects := by
  sorry


end NUMINAMATH_CALUDE_find_lighter_orange_l350_35081


namespace NUMINAMATH_CALUDE_inequality_solution_set_l350_35089

theorem inequality_solution_set (x : ℝ) : 
  1 - 7 / (2 * x - 1) < 0 ↔ 1/2 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l350_35089


namespace NUMINAMATH_CALUDE_vector_problem_l350_35020

/-- Given vectors a and b in ℝ², if vector c satisfies the conditions
    (c + b) ⊥ a and (c - a) ∥ b, then c = (2, 1). -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (1, -1) → 
  b = (1, 2) → 
  ((c.1 + b.1, c.2 + b.2) • a = 0) →  -- (c + b) ⊥ a
  (∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2)) →  -- (c - a) ∥ b
  c = (2, 1) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l350_35020


namespace NUMINAMATH_CALUDE_positive_real_inequality_l350_35060

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 1/x + 1/y + 1/z) :
  x + y + z ≥ Real.sqrt ((x*y + 1)/2) + Real.sqrt ((y*z + 1)/2) + Real.sqrt ((z*x + 1)/2) :=
by sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l350_35060


namespace NUMINAMATH_CALUDE_linear_function_proof_l350_35084

theorem linear_function_proof (f : ℝ → ℝ) :
  (∀ x y : ℝ, ∃ k b : ℝ, f x = k * x + b) →
  (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) →
  (∀ x : ℝ, f x = x + 3) := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l350_35084


namespace NUMINAMATH_CALUDE_max_students_distribution_l350_35011

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1230) (h2 : pencils = 920) :
  (∃ (students : ℕ), students > 0 ∧ 
   pens % students = 0 ∧ 
   pencils % students = 0 ∧
   ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔ 
  (Nat.gcd pens pencils = 10) :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l350_35011


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_one_l350_35031

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, then k = 1 -/
theorem parallel_vectors_imply_k_equals_one (a b c : ℝ × ℝ) (k : ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (0, 1) →
  c = (k, Real.sqrt 3) →
  ∃ (t : ℝ), t • (a + 2 • b) = c →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_one_l350_35031


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l350_35009

theorem waitress_income_fraction (salary : ℝ) (tips : ℝ) (income : ℝ) : 
  salary > 0 →
  tips = (7 / 4) * salary →
  income = salary + tips →
  tips / income = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l350_35009


namespace NUMINAMATH_CALUDE_unknown_number_proof_l350_35015

theorem unknown_number_proof : 
  ∃ x : ℝ, (45 * x = 0.4 * 900) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l350_35015


namespace NUMINAMATH_CALUDE_ellipse_properties_l350_35053

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The ellipse satisfies the given conditions -/
def EllipseConditions (e : Ellipse) : Prop :=
  (e.a ^ 2 - e.b ^ 2) / e.a ^ 2 = 3 / 4 ∧  -- eccentricity is √3/2
  e.a - (e.a ^ 2 - e.b ^ 2).sqrt = 2       -- distance from upper vertex to focus is 2

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) (h : EllipseConditions e) :
  e.a = 2 ∧ e.b = 1 ∧
  (∀ k : ℝ, k = 1 → 
    (∃ S : ℝ → ℝ, (∀ m : ℝ, S m ≤ 1) ∧ (∃ m : ℝ, S m = 1))) ∧
  (∀ k : ℝ, (∀ m : ℝ, ∃ C : ℝ, 
    (∀ x : ℝ, (x - m)^2 + (k * (x - m))^2 + 
      ((4 * (k^2 * m^2 - 1)) / (1 + 4 * k^2) - x)^2 + 
      (k * ((4 * (k^2 * m^2 - 1)) / (1 + 4 * k^2) - x))^2 = C)) → 
    k = 1/2 ∨ k = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l350_35053


namespace NUMINAMATH_CALUDE_picnic_class_size_l350_35029

theorem picnic_class_size : ∃ (x : ℕ), 
  x > 0 ∧ 
  (x / 2 + x / 3 + x / 4 : ℚ) = 65 ∧ 
  x = 60 :=
by sorry

end NUMINAMATH_CALUDE_picnic_class_size_l350_35029


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l350_35018

/-- The base 6 number represented as a list of digits -/
def base_6_number : List Nat := [1, 0, 2, 1, 1, 1, 0, 1, 1]

/-- Convert a list of digits in base 6 to a natural number -/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The number we're working with -/
def n : Nat := to_base_10 base_6_number

/-- A number is prime if it has exactly two distinct divisors -/
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ m : Nat, m > 0 → m < p → (p % m = 0 → m = 1)

/-- p divides n -/
def divides (p n : Nat) : Prop := n % p = 0

theorem largest_prime_divisor :
  ∃ (p : Nat), is_prime p ∧ divides p n ∧
  ∀ (q : Nat), is_prime q → divides q n → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l350_35018


namespace NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l350_35017

/-- Given that i is the imaginary unit and (x+i)i = y-i where x and y are real numbers,
    prove that the point (x, y) lies in the third quadrant of the complex plane. -/
theorem complex_point_in_third_quadrant (x y : ℝ) (i : ℂ) 
  (h_i : i * i = -1) 
  (h_eq : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l350_35017


namespace NUMINAMATH_CALUDE_complement_of_union_l350_35093

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l350_35093


namespace NUMINAMATH_CALUDE_power_five_mod_eighteen_l350_35071

theorem power_five_mod_eighteen : 5^100 % 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_eighteen_l350_35071


namespace NUMINAMATH_CALUDE_cos_150_degrees_l350_35094

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l350_35094


namespace NUMINAMATH_CALUDE_coins_per_stack_l350_35063

theorem coins_per_stack (total_coins : ℕ) (num_stacks : ℕ) (coins_per_stack : ℕ) : 
  total_coins = 15 → num_stacks = 5 → total_coins = num_stacks * coins_per_stack → coins_per_stack = 3 := by
  sorry

end NUMINAMATH_CALUDE_coins_per_stack_l350_35063


namespace NUMINAMATH_CALUDE_collinear_probability_l350_35090

/-- Represents a rectangular grid of dots -/
structure DotGrid :=
  (rows : ℕ)
  (columns : ℕ)

/-- Calculates the total number of dots in the grid -/
def DotGrid.total_dots (g : DotGrid) : ℕ := g.rows * g.columns

/-- Calculates the number of ways to choose 4 dots from n dots -/
def choose_four (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Represents the number of collinear sets in the grid -/
def collinear_sets (g : DotGrid) : ℕ := 
  g.rows * 2 + g.columns + 4

/-- The main theorem stating the probability of four randomly chosen dots being collinear -/
theorem collinear_probability (g : DotGrid) (h1 : g.rows = 4) (h2 : g.columns = 5) : 
  (collinear_sets g : ℚ) / (choose_four (g.total_dots) : ℚ) = 17 / 4845 := by
  sorry

#eval collinear_sets (DotGrid.mk 4 5)
#eval choose_four 20

end NUMINAMATH_CALUDE_collinear_probability_l350_35090


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_half_l350_35044

theorem tangent_slope_at_pi_half :
  let f (x : ℝ) := Real.tan (x / 2)
  (deriv f) (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_half_l350_35044


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l350_35047

/-- The perimeter of a trapezoid JKLM with given coordinates is 34 units. -/
theorem trapezoid_perimeter : 
  let j : ℝ × ℝ := (-2, -4)
  let k : ℝ × ℝ := (-2, 1)
  let l : ℝ × ℝ := (6, 7)
  let m : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist j k + dist k l + dist l m + dist m j
  perimeter = 34 := by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l350_35047


namespace NUMINAMATH_CALUDE_binomial_12_9_l350_35061

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l350_35061


namespace NUMINAMATH_CALUDE_smallest_common_multiple_9_15_gt_50_l350_35013

theorem smallest_common_multiple_9_15_gt_50 : ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < n → (m % 9 = 0 ∧ m % 15 = 0 → m ≤ 50)) ∧
  n % 9 = 0 ∧ n % 15 = 0 ∧ n > 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_9_15_gt_50_l350_35013


namespace NUMINAMATH_CALUDE_train_speed_calculation_l350_35040

/-- Given two trains moving in opposite directions, prove the speed of one train given the lengths, speed of the other train, and time to cross. -/
theorem train_speed_calculation (length1 length2 speed2 time_to_cross : ℝ) 
  (h1 : length1 = 300)
  (h2 : length2 = 200.04)
  (h3 : speed2 = 80)
  (h4 : time_to_cross = 9 / 3600) : 
  ∃ speed1 : ℝ, speed1 = 120.016 ∧ 
  (length1 + length2) / 1000 = (speed1 + speed2) * time_to_cross := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l350_35040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l350_35058

/-- Definition of arithmetic sequence sum -/
def arithmetic_sequence_sum (n : ℕ) : ℝ := sorry

/-- Theorem: For an arithmetic sequence with sum S_n, if S_3 = 15 and S_9 = 153, then S_6 = 66 -/
theorem arithmetic_sequence_sum_property :
  (arithmetic_sequence_sum 3 = 15) →
  (arithmetic_sequence_sum 9 = 153) →
  (arithmetic_sequence_sum 6 = 66) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l350_35058


namespace NUMINAMATH_CALUDE_bagel_store_spending_l350_35028

/-- The total amount spent by Ben and David in the bagel store -/
def total_spent (b d : ℝ) : ℝ := b + d

/-- Ben's spending is $15 more than David's spending -/
def ben_spent_more (b d : ℝ) : Prop := b = d + 15

/-- David's spending is half of Ben's spending -/
def david_spent_half (b d : ℝ) : Prop := d = b / 2

theorem bagel_store_spending (b d : ℝ) 
  (h1 : david_spent_half b d) 
  (h2 : ben_spent_more b d) : 
  total_spent b d = 45 := by
  sorry

end NUMINAMATH_CALUDE_bagel_store_spending_l350_35028


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l350_35012

/-- Two parallel planar vectors have a specific sum magnitude -/
theorem parallel_vectors_sum_magnitude :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -3]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  ‖(a + b)‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l350_35012


namespace NUMINAMATH_CALUDE_davids_math_marks_l350_35035

theorem davids_math_marks (english physics chemistry biology average : ℕ) 
  (h1 : english = 86)
  (h2 : physics = 92)
  (h3 : chemistry = 87)
  (h4 : biology = 95)
  (h5 : average = 89)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_math_marks_l350_35035


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l350_35097

theorem sibling_ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 180 → a + b + c = 26 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l350_35097


namespace NUMINAMATH_CALUDE_unique_solution_l350_35021

theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ (180 / x) + ((5 * 12) / x) + 80 = 81 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l350_35021


namespace NUMINAMATH_CALUDE_certain_number_calculation_l350_35049

theorem certain_number_calculation (y : ℝ) : (0.65 * 210 = 0.20 * y) → y = 682.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l350_35049


namespace NUMINAMATH_CALUDE_stating_count_valid_outfits_l350_35032

/-- Represents the number of red shirts -/
def red_shirts : ℕ := 7

/-- Represents the number of green shirts -/
def green_shirts : ℕ := 5

/-- Represents the number of pants -/
def pants : ℕ := 6

/-- Represents the number of green hats -/
def green_hats : ℕ := 9

/-- Represents the number of red hats -/
def red_hats : ℕ := 7

/-- Represents the total number of valid outfits -/
def total_outfits : ℕ := 1152

/-- 
Theorem stating that the number of valid outfits is 1152.
A valid outfit consists of one shirt, one pair of pants, and one hat,
where either the shirt and hat don't share the same color,
or the pants and hat don't share the same color.
-/
theorem count_valid_outfits : 
  (red_shirts * pants * green_hats) + 
  (green_shirts * pants * red_hats) + 
  (red_shirts * red_hats * pants) +
  (green_shirts * green_hats * pants) = total_outfits :=
sorry

end NUMINAMATH_CALUDE_stating_count_valid_outfits_l350_35032


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l350_35080

/-- Given a square of side length 2, divided into a central square and two congruent trapezoids,
    if the areas are equal, then the longer parallel side of a trapezoid is 1. -/
theorem trapezoid_side_length (s : ℝ) : 
  2 > 0 ∧                             -- Square side length is positive
  s > 0 ∧                             -- Central square side length is positive
  s < 2 ∧                             -- Central square fits inside the larger square
  s^2 = (1 + s) / 2 →                 -- Areas are equal
  s = 1 :=                            -- Longer parallel side of trapezoid is 1
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l350_35080


namespace NUMINAMATH_CALUDE_house_work_payment_l350_35067

/-- Represents the total payment for work on a house --/
def total_payment (bricklayer_rate : ℝ) (electrician_rate : ℝ) (hours_worked : ℝ) : ℝ :=
  bricklayer_rate * hours_worked + electrician_rate * hours_worked

/-- Proves that the total payment for the work is $630 --/
theorem house_work_payment : 
  let bricklayer_rate : ℝ := 12
  let electrician_rate : ℝ := 16
  let hours_worked : ℝ := 22.5
  total_payment bricklayer_rate electrician_rate hours_worked = 630 := by
  sorry

#eval total_payment 12 16 22.5

end NUMINAMATH_CALUDE_house_work_payment_l350_35067


namespace NUMINAMATH_CALUDE_fraction_multiplication_l350_35002

theorem fraction_multiplication : 
  (7 / 8 : ℚ) * (1 / 3 : ℚ) * (3 / 7 : ℚ) = 0.12499999999999997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l350_35002


namespace NUMINAMATH_CALUDE_sum_of_numbers_l350_35026

theorem sum_of_numbers (a b : ℕ) : 
  100 ≤ a ∧ a ≤ 999 →   -- a is a three-digit number
  10 ≤ b ∧ b ≤ 99 →     -- b is a two-digit number
  a - b = 989 →         -- their difference is 989
  a + b = 1009 :=       -- prove their sum is 1009
by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l350_35026


namespace NUMINAMATH_CALUDE_computer_games_count_l350_35099

def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def computer_game_price : ℕ := 90
def polo_shirt_count : ℕ := 3
def necklace_count : ℕ := 2
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

theorem computer_games_count :
  ∃ (n : ℕ), 
    n * computer_game_price + 
    polo_shirt_count * polo_shirt_price + 
    necklace_count * necklace_price - 
    rebate = total_cost_after_rebate ∧ 
    n = 1 := by sorry

end NUMINAMATH_CALUDE_computer_games_count_l350_35099


namespace NUMINAMATH_CALUDE_prob_A_at_edge_is_two_thirds_l350_35023

/-- The number of students -/
def num_students : ℕ := 3

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := num_students.factorial

/-- The number of arrangements with A at the edge -/
def arrangements_with_A_at_edge : ℕ := 2 * (num_students - 1).factorial

/-- The probability of A standing at the edge -/
def prob_A_at_edge : ℚ := arrangements_with_A_at_edge / total_arrangements

theorem prob_A_at_edge_is_two_thirds : prob_A_at_edge = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_at_edge_is_two_thirds_l350_35023


namespace NUMINAMATH_CALUDE_homework_problems_exist_l350_35006

theorem homework_problems_exist : ∃ (a b c d : ℤ), 
  (a ≤ -1) ∧ (b ≤ -1) ∧ (c ≤ -1) ∧ (d ≤ -1) ∧ 
  (a * b = -(a + b)) ∧ 
  (c * d = -182 * (1 / (c + d))) :=
sorry

end NUMINAMATH_CALUDE_homework_problems_exist_l350_35006


namespace NUMINAMATH_CALUDE_largest_distinct_digits_divisible_by_99_l350_35073

def is_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n.digits 10).nthLe i (by sorry) ≠ (n.digits 10).nthLe j (by sorry)

theorem largest_distinct_digits_divisible_by_99 :
  ∀ n : ℕ, n > 9876524130 → ¬(is_distinct_digits n ∧ n % 99 = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_distinct_digits_divisible_by_99_l350_35073


namespace NUMINAMATH_CALUDE_octal_55_to_binary_l350_35034

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 8 + (n % 10)

/-- Converts a decimal number to binary --/
def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

/-- Represents a binary number as a natural number --/
def binary_to_nat (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 2 * acc + d) 0

theorem octal_55_to_binary : 
  binary_to_nat (decimal_to_binary (octal_to_decimal 55)) = binary_to_nat [1,0,1,1,0,1] := by
  sorry

end NUMINAMATH_CALUDE_octal_55_to_binary_l350_35034


namespace NUMINAMATH_CALUDE_cube_center_pyramids_l350_35056

/-- Given a cube with edge length a, prove the volume and surface area of the pyramids formed by connecting the center to all vertices. -/
theorem cube_center_pyramids (a : ℝ) (h : a > 0) :
  ∃ (volume surface_area : ℝ),
    volume = a^3 / 6 ∧
    surface_area = a^2 * (1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_cube_center_pyramids_l350_35056


namespace NUMINAMATH_CALUDE_base_conversion_3275_to_octal_l350_35096

theorem base_conversion_3275_to_octal :
  (6 * 8^3 + 3 * 8^2 + 2 * 8^1 + 3 * 8^0 : ℕ) = 3275 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_3275_to_octal_l350_35096


namespace NUMINAMATH_CALUDE_ratio_problem_l350_35087

/-- Given ratios for x, y, and z, prove their values -/
theorem ratio_problem (x y z : ℚ) : 
  (x / 12 = 5 / 1) → 
  (y / 21 = 7 / 3) → 
  (z / 16 = 4 / 2) → 
  (x = 60 ∧ y = 49 ∧ z = 32) :=
by sorry

end NUMINAMATH_CALUDE_ratio_problem_l350_35087


namespace NUMINAMATH_CALUDE_change_received_l350_35078

/-- Represents the cost of a basic calculator in dollars -/
def basic_cost : ℕ := 8

/-- Represents the total amount of money the teacher had in dollars -/
def total_money : ℕ := 100

/-- Calculates the cost of a scientific calculator -/
def scientific_cost : ℕ := 2 * basic_cost

/-- Calculates the cost of a graphing calculator -/
def graphing_cost : ℕ := 3 * scientific_cost

/-- Calculates the total cost of buying one of each calculator -/
def total_cost : ℕ := basic_cost + scientific_cost + graphing_cost

/-- Theorem stating that the change received is $28 -/
theorem change_received : total_money - total_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l350_35078


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l350_35082

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + 2*(m-1)*x + m^2 - 1

-- Theorem statement
theorem quadratic_roots_properties :
  -- The equation has two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 →
  -- The range of m is m < 1
  (∀ m : ℝ, (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0) → m < 1) ∧
  -- There exists a value of m such that the product of the roots is zero, and that value is m = -1
  (∃ m : ℝ, m = -1 ∧ ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁ * x₂ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_properties_l350_35082


namespace NUMINAMATH_CALUDE_cindy_lisa_marble_difference_l350_35014

theorem cindy_lisa_marble_difference :
  ∀ (lisa_initial : ℕ),
  let cindy_initial : ℕ := 20
  let cindy_after : ℕ := cindy_initial - 12
  let lisa_after : ℕ := lisa_initial + 12
  lisa_after = cindy_after + 19 →
  cindy_initial - lisa_initial = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_lisa_marble_difference_l350_35014


namespace NUMINAMATH_CALUDE_kelly_apples_l350_35088

theorem kelly_apples (initial : ℕ) (second_day : ℕ) (third_day : ℕ) (eaten : ℕ) : 
  initial = 56 → second_day = 105 → third_day = 84 → eaten = 23 →
  initial + second_day + third_day - eaten = 222 :=
by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l350_35088


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l350_35030

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧ A ≠ B

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles 
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  ∀ x y, perpendicular_bisector x y ↔ 
    (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ∧
    2*x = A.1 + B.1 ∧ 2*y = A.2 + B.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l350_35030
