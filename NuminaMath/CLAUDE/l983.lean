import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l983_98380

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l983_98380


namespace NUMINAMATH_CALUDE_min_ratio_logarithmic_intersections_l983_98378

theorem min_ratio_logarithmic_intersections (m : ℝ) (h : m > 0) :
  let f (m : ℝ) := (2^m - 2^(8/(2*m+1))) / (2^(-m) - 2^(-8/(2*m+1)))
  ∀ x > 0, f m ≥ 8 * Real.sqrt 2 ∧ ∃ m₀ > 0, f m₀ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_logarithmic_intersections_l983_98378


namespace NUMINAMATH_CALUDE_toy_sale_proof_l983_98311

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the gain in terms of number of toys. -/
def totalSellingPrice (numToysSold : ℕ) (costPrice : ℕ) (gainInToys : ℕ) : ℕ :=
  (numToysSold + gainInToys) * costPrice

/-- Proves that the total selling price for 18 toys with a cost price of 1200
    and a gain equal to the cost of 3 toys is 25200. -/
theorem toy_sale_proof :
  totalSellingPrice 18 1200 3 = 25200 := by
  sorry

end NUMINAMATH_CALUDE_toy_sale_proof_l983_98311


namespace NUMINAMATH_CALUDE_function_behavior_l983_98361

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2*x - 6

-- Theorem statement
theorem function_behavior :
  ∃ c ∈ Set.Ioo 2 4, 
    (∀ x ∈ Set.Ioo 2 c, (f' x < 0)) ∧ 
    (∀ x ∈ Set.Ioo c 4, (f' x > 0)) :=
sorry


end NUMINAMATH_CALUDE_function_behavior_l983_98361


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l983_98382

theorem trig_expression_equals_negative_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l983_98382


namespace NUMINAMATH_CALUDE_greatest_b_satisfying_inequality_l983_98348

def quadratic_inequality (b : ℝ) : Prop :=
  b^2 - 14*b + 45 ≤ 0

theorem greatest_b_satisfying_inequality :
  ∃ (b : ℝ), quadratic_inequality b ∧
    ∀ (x : ℝ), quadratic_inequality x → x ≤ b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_b_satisfying_inequality_l983_98348


namespace NUMINAMATH_CALUDE_sally_coins_theorem_l983_98365

def initial_pennies : ℕ := 8
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def penny_value : ℕ := 1
def nickel_value : ℕ := 5

theorem sally_coins_theorem :
  let total_nickels := initial_nickels + dad_nickels + mom_nickels
  let total_value := initial_pennies * penny_value + total_nickels * nickel_value
  total_nickels = 18 ∧ total_value = 98 := by sorry

end NUMINAMATH_CALUDE_sally_coins_theorem_l983_98365


namespace NUMINAMATH_CALUDE_absolute_curve_sufficient_not_necessary_l983_98363

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the property of being on the curve y = |x|
def onAbsoluteCurve (p : Point2D) : Prop :=
  p.y = |p.x|

-- Define the property of equal distance to both axes
def equalDistanceToAxes (p : Point2D) : Prop :=
  |p.x| = |p.y|

-- Theorem statement
theorem absolute_curve_sufficient_not_necessary :
  (∀ p : Point2D, onAbsoluteCurve p → equalDistanceToAxes p) ∧
  (∃ p : Point2D, equalDistanceToAxes p ∧ ¬onAbsoluteCurve p) :=
sorry

end NUMINAMATH_CALUDE_absolute_curve_sufficient_not_necessary_l983_98363


namespace NUMINAMATH_CALUDE_expression_evaluation_l983_98397

theorem expression_evaluation :
  (5^500 + 6^501)^2 - (5^500 - 6^501)^2 = 24 * 30^500 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l983_98397


namespace NUMINAMATH_CALUDE_total_marbles_l983_98324

/-- The total number of marbles for three people given specific conditions -/
theorem total_marbles (my_marbles : ℕ) (brother_marbles : ℕ) (friend_marbles : ℕ) 
  (h1 : my_marbles = 16)
  (h2 : my_marbles - 2 = 2 * (brother_marbles + 2))
  (h3 : friend_marbles = 3 * (my_marbles - 2)) :
  my_marbles + brother_marbles + friend_marbles = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l983_98324


namespace NUMINAMATH_CALUDE_product_multiple_of_16_probability_l983_98304

def S : Finset ℕ := {3, 4, 8, 16}

theorem product_multiple_of_16_probability :
  let pairs := S.powerset.filter (λ p : Finset ℕ => p.card = 2)
  let valid_pairs := pairs.filter (λ p : Finset ℕ => (p.prod id) % 16 = 0)
  (valid_pairs.card : ℚ) / pairs.card = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_product_multiple_of_16_probability_l983_98304


namespace NUMINAMATH_CALUDE_bryan_continents_l983_98314

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := 122

/-- The total number of books Bryan collected from all continents -/
def total_books : ℕ := 488

/-- The number of continents Bryan collected books from -/
def num_continents : ℕ := total_books / books_per_continent

theorem bryan_continents :
  num_continents = 4 := by sorry

end NUMINAMATH_CALUDE_bryan_continents_l983_98314


namespace NUMINAMATH_CALUDE_parabola_intersection_right_angle_l983_98309

-- Define the line equation
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem parabola_intersection_right_angle :
  ∃ (A B C : ℝ × ℝ),
    A ≠ B ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    parabola C.1 C.2 ∧
    angle A C B = π/2 →
    C = (1, -2) ∨ C = (9, -6) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_right_angle_l983_98309


namespace NUMINAMATH_CALUDE_longest_segment_proof_l983_98370

/-- The total length of all segments in the rectangular spiral -/
def total_length : ℕ := 3000

/-- Predicate to check if a given length satisfies the spiral condition -/
def satisfies_spiral_condition (n : ℕ) : Prop :=
  n * (n + 1) ≤ total_length

/-- The longest line segment in the rectangular spiral -/
def longest_segment : ℕ := 54

theorem longest_segment_proof :
  satisfies_spiral_condition longest_segment ∧
  ∀ m : ℕ, m > longest_segment → ¬satisfies_spiral_condition m :=
by sorry

end NUMINAMATH_CALUDE_longest_segment_proof_l983_98370


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l983_98325

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x ≤ 20 →
  x + 4*x > 20 →
  x + 20 > 4*x →
  4*x + 20 > x →
  ∀ y : ℕ,
  y > 0 →
  y ≤ 20 →
  y + 4*y > 20 →
  y + 20 > 4*y →
  4*y + 20 > y →
  x + 4*x + 20 ≥ y + 4*y + 20 →
  x + 4*x + 20 ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l983_98325


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l983_98356

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n = 990 ∧ 
  (∀ m : ℕ, m < 1000 → m % 5 = 0 → m % 6 = 0 → m ≤ n) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l983_98356


namespace NUMINAMATH_CALUDE_special_number_exists_l983_98306

theorem special_number_exists : ∃ n : ℕ+, 
  (Nat.digits 10 n.val).length = 1000 ∧ 
  0 ∉ Nat.digits 10 n.val ∧
  ∃ pairs : List (ℕ × ℕ), 
    pairs.length = 500 ∧
    (pairs.map (λ p => p.1 * p.2)).sum ∣ n.val ∧
    ∀ d ∈ Nat.digits 10 n.val, ∃ p ∈ pairs, d = p.1 ∨ d = p.2 :=
by sorry

end NUMINAMATH_CALUDE_special_number_exists_l983_98306


namespace NUMINAMATH_CALUDE_maple_trees_planted_l983_98391

theorem maple_trees_planted (initial : ℕ) (final : ℕ) (h1 : initial = 53) (h2 : final = 64) :
  final - initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_planted_l983_98391


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l983_98313

/-- Proves that a compound with 6 C atoms, 8 H atoms, and a molecular weight of 192
    contains 7 O atoms, given the atomic weights of C, H, and O. -/
theorem compound_oxygen_atoms 
  (atomic_weight_C : ℝ) 
  (atomic_weight_H : ℝ) 
  (atomic_weight_O : ℝ) 
  (h1 : atomic_weight_C = 12.01)
  (h2 : atomic_weight_H = 1.008)
  (h3 : atomic_weight_O = 16.00)
  (h4 : (6 * atomic_weight_C + 8 * atomic_weight_H + 7 * atomic_weight_O) = 192) :
  ∃ n : ℕ, n = 7 ∧ (6 * atomic_weight_C + 8 * atomic_weight_H + n * atomic_weight_O) = 192 :=
by
  sorry


end NUMINAMATH_CALUDE_compound_oxygen_atoms_l983_98313


namespace NUMINAMATH_CALUDE_box_width_proof_l983_98317

theorem box_width_proof (length width height : ℕ) (cubes : ℕ) : 
  length = 15 → height = 13 → cubes = 3120 → cubes = length * width * height → width = 16 := by
  sorry

end NUMINAMATH_CALUDE_box_width_proof_l983_98317


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l983_98398

theorem sum_of_three_numbers (a b c : ℝ) : 
  a + b = 35 ∧ b + c = 50 ∧ c + a = 60 → a + b + c = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l983_98398


namespace NUMINAMATH_CALUDE_expression_evaluation_l983_98319

theorem expression_evaluation :
  let a : ℚ := -1/3
  let expr := (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2*a)
  expr = -2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l983_98319


namespace NUMINAMATH_CALUDE_water_consumption_l983_98352

theorem water_consumption (initial_water : ℝ) : 
  initial_water > 0 →
  let remaining_day1 := initial_water * (1 - 7/15)
  let remaining_day2 := remaining_day1 * (1 - 5/8)
  let remaining_day3 := remaining_day2 * (1 - 2/3)
  remaining_day3 = 2.6 →
  initial_water = 39 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_l983_98352


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l983_98302

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'X', 'O', 'O', 'X', 'O', 'X']

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l983_98302


namespace NUMINAMATH_CALUDE_steak_cost_calculation_l983_98399

/-- Calculate the total cost of steaks with a buy two get one free offer and a discount --/
theorem steak_cost_calculation (price_per_pound : ℝ) (pounds_bought : ℝ) (discount_rate : ℝ) : 
  price_per_pound = 15 →
  pounds_bought = 24 →
  discount_rate = 0.1 →
  (pounds_bought * price_per_pound) * (1 - discount_rate) = 324 := by
sorry

end NUMINAMATH_CALUDE_steak_cost_calculation_l983_98399


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l983_98355

/-- The minimum distance between a point on the line x - y + 1 = 0 
    and a point on the circle (x - 1)² + y² = 1 is √2 - 1 -/
theorem min_distance_line_circle :
  let line := {(x, y) : ℝ × ℝ | x - y + 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧ 
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ circle → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) ∧
    (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_line_circle_l983_98355


namespace NUMINAMATH_CALUDE_touching_balls_theorem_l983_98303

/-- Represents a spherical ball with a given radius -/
structure Ball where
  radius : ℝ

/-- Represents two touching balls on the ground -/
structure TouchingBalls where
  ball1 : Ball
  ball2 : Ball
  contactHeight : ℝ

/-- The radius of the other ball given the conditions -/
def otherBallRadius (balls : TouchingBalls) : ℝ := 6

theorem touching_balls_theorem (balls : TouchingBalls) 
  (h1 : balls.ball1.radius = 4)
  (h2 : balls.contactHeight = 6) :
  otherBallRadius balls = balls.ball2.radius :=
sorry

end NUMINAMATH_CALUDE_touching_balls_theorem_l983_98303


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l983_98333

theorem solution_set_of_inequality (x : ℝ) :
  Set.Ioo (-1 : ℝ) 2 = {x | |x^2 - x| < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l983_98333


namespace NUMINAMATH_CALUDE_stating_exists_k_no_carries_l983_98315

/-- 
Given two positive integers a and b, returns true if adding a to b
results in no carries during the whole calculation in base 10.
-/
def no_carries (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (a / 10^d % 10 + b / 10^d % 10 < 10)

/-- 
Theorem stating that there exists a positive integer k such that
adding 1996k to 1997k results in no carries during the whole calculation.
-/
theorem exists_k_no_carries : ∃ k : ℕ, k > 0 ∧ no_carries (1996 * k) (1997 * k) := by
  sorry

end NUMINAMATH_CALUDE_stating_exists_k_no_carries_l983_98315


namespace NUMINAMATH_CALUDE_quadratic_function_b_value_l983_98336

/-- Given a quadratic function f(x) = ax² + bx + c, if f(2) - f(-2) = 8, then b = 2 -/
theorem quadratic_function_b_value (a b c : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 8 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_b_value_l983_98336


namespace NUMINAMATH_CALUDE_sum_of_constants_l983_98305

/-- Given a function y(x) = a + b/x, where a and b are constants,
    prove that a + b = -34 if y(-2) = 2 and y(-4) = 8 -/
theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 2 ↔ x = -2) ∧ (a + b / x = 8 ↔ x = -4)) →
  a + b = -34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_constants_l983_98305


namespace NUMINAMATH_CALUDE_exists_n_for_all_k_l983_98351

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 
  Real.sqrt (n + 1981^k) + Real.sqrt n = (Real.sqrt 1982 + 1)^k := by
  sorry

end NUMINAMATH_CALUDE_exists_n_for_all_k_l983_98351


namespace NUMINAMATH_CALUDE_x_eighteenth_equals_negative_one_l983_98308

theorem x_eighteenth_equals_negative_one (x : ℂ) (h : x + 1/x = Real.sqrt 3) : x^18 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_eighteenth_equals_negative_one_l983_98308


namespace NUMINAMATH_CALUDE_remove_500th_digit_of_3_7_is_greater_l983_98334

/-- Represents a decimal expansion with a finite number of digits -/
def DecimalExpansion := List Nat

/-- Converts a rational number to its decimal expansion with a given number of digits -/
def rationalToDecimal (n d : Nat) (digits : Nat) : DecimalExpansion :=
  sorry

/-- Removes the nth digit from a decimal expansion -/
def removeNthDigit (n : Nat) (d : DecimalExpansion) : DecimalExpansion :=
  sorry

/-- Converts a decimal expansion back to a rational number -/
def decimalToRational (d : DecimalExpansion) : Rat :=
  sorry

theorem remove_500th_digit_of_3_7_is_greater :
  let original := (3 : Rat) / 7
  let decimalExp := rationalToDecimal 3 7 1000
  let modified := removeNthDigit 500 decimalExp
  decimalToRational modified > original := by
  sorry

end NUMINAMATH_CALUDE_remove_500th_digit_of_3_7_is_greater_l983_98334


namespace NUMINAMATH_CALUDE_tangent_line_slope_l983_98358

/-- Given a curve y = x^3 and its tangent line y = kx + 2, prove that k = 3 -/
theorem tangent_line_slope (x : ℝ) :
  let f : ℝ → ℝ := fun x => x^3
  let f' : ℝ → ℝ := fun x => 3 * x^2
  let tangent_line (k : ℝ) (x : ℝ) := k * x + 2
  ∃ m : ℝ, f m = tangent_line k m ∧ f' m = k → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l983_98358


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l983_98385

/-- Two real numbers vary inversely if their product is constant. -/
def VaryInversely (p q : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ x : ℝ, p x * q x = k

theorem inverse_variation_problem (p q : ℝ → ℝ) 
    (h_inverse : VaryInversely p q)
    (h_initial : p 1 = 800 ∧ q 1 = 0.5) :
    p 2 = 1600 → q 2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l983_98385


namespace NUMINAMATH_CALUDE_function_difference_implies_m_value_l983_98327

/-- The function f(x) = 4x^2 - 3x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5

/-- The function g(x) = x^2 - mx - 8, parameterized by m -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 8

/-- Theorem stating that if f(5) - g(5) = 15, then m = -11.6 -/
theorem function_difference_implies_m_value :
  ∃ m : ℝ, f 5 - g m 5 = 15 → m = -11.6 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_implies_m_value_l983_98327


namespace NUMINAMATH_CALUDE_weight_difference_l983_98384

/-- Given Mildred weighs 59 pounds and Carol weighs 9 pounds, 
    prove that Mildred is 50 pounds heavier than Carol. -/
theorem weight_difference (mildred_weight carol_weight : ℕ) 
  (h1 : mildred_weight = 59) 
  (h2 : carol_weight = 9) : 
  mildred_weight - carol_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l983_98384


namespace NUMINAMATH_CALUDE_muffins_per_box_l983_98322

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) (muffins_per_box : ℕ) : 
  total_muffins = 96 →
  num_boxes = 8 →
  total_muffins = num_boxes * muffins_per_box →
  muffins_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_muffins_per_box_l983_98322


namespace NUMINAMATH_CALUDE_min_colors_tessellation_valid_coloring_uses_three_colors_l983_98372

/-- Represents a tile in the tessellation -/
inductive Tile
  | Rectangle : Tile
  | Circle : Tile

/-- Represents a color used in the coloring -/
inductive Color
  | Red : Color
  | Blue : Color
  | Green : Color

/-- Represents the tessellation structure -/
structure Tessellation where
  tiles : List Tile
  adjacent : Tile → Tile → Prop
  overlapping : Tile → Tile → Prop

/-- A valid coloring of the tessellation -/
def ValidColoring (t : Tessellation) (coloring : Tile → Color) : Prop :=
  ∀ x y, (t.adjacent x y ∨ t.overlapping x y) → coloring x ≠ coloring y

/-- The main theorem stating that 3 is the minimum number of colors required -/
theorem min_colors_tessellation (t : Tessellation) :
  (∃ coloring, ValidColoring t coloring) ∧
  (∀ coloring, ValidColoring t coloring → (Set.range coloring).ncard ≥ 3) :=
sorry

/-- Helper theorem: Any valid coloring uses at least 3 colors -/
theorem valid_coloring_uses_three_colors (t : Tessellation) (coloring : Tile → Color) :
  ValidColoring t coloring → (Set.range coloring).ncard ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_colors_tessellation_valid_coloring_uses_three_colors_l983_98372


namespace NUMINAMATH_CALUDE_water_consumption_percentage_difference_l983_98394

theorem water_consumption_percentage_difference : 
  let yesterday_consumption : ℝ := 48
  let two_days_ago_consumption : ℝ := 50
  let difference := two_days_ago_consumption - yesterday_consumption
  let percentage_difference := (difference / two_days_ago_consumption) * 100
  percentage_difference = 4 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_percentage_difference_l983_98394


namespace NUMINAMATH_CALUDE_part1_part2_l983_98346

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 4) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part1 (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, ¬(q m x) → ¬(p x)) 
  (h3 : ∃ x, ¬(p x) ∧ q m x) : 
  m ≥ 4 := by sorry

-- Part 2
theorem part2 (x : ℝ) 
  (h1 : p x ∨ q 5 x) 
  (h2 : ¬(p x ∧ q 5 x)) : 
  x ∈ Set.Icc (-3 : ℝ) (-2) ∪ Set.Ioc 4 7 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l983_98346


namespace NUMINAMATH_CALUDE_more_non_products_than_products_l983_98321

/-- The number of ten-digit numbers -/
def ten_digit_count : ℕ := 9 * 10^9

/-- The number of five-digit numbers -/
def five_digit_count : ℕ := 90000

/-- The estimated number of products of two five-digit numbers that are ten-digit numbers -/
def ten_digit_products : ℕ := (five_digit_count * (five_digit_count - 1) / 2 + five_digit_count) / 2

theorem more_non_products_than_products : ten_digit_count - ten_digit_products > ten_digit_products := by
  sorry

end NUMINAMATH_CALUDE_more_non_products_than_products_l983_98321


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1540_l983_98393

theorem sum_of_extreme_prime_factors_of_1540 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ 
    largest.Prime ∧
    smallest ∣ 1540 ∧ 
    largest ∣ 1540 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1540 → p ≥ smallest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1540 → p ≤ largest) ∧
    smallest + largest = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1540_l983_98393


namespace NUMINAMATH_CALUDE_number_exists_l983_98323

theorem number_exists : ∃ x : ℝ, (2/3 * x)^3 - 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l983_98323


namespace NUMINAMATH_CALUDE_visibility_time_correct_l983_98362

/-- The time when Steve and Laura can see each other again -/
def visibility_time : ℝ := 45

/-- Steve's walking speed in feet per second -/
def steve_speed : ℝ := 3

/-- Laura's walking speed in feet per second -/
def laura_speed : ℝ := 1

/-- Distance between Steve and Laura's parallel paths in feet -/
def path_distance : ℝ := 240

/-- Diameter of the circular art installation in feet -/
def installation_diameter : ℝ := 80

/-- Initial separation between Steve and Laura when hidden by the art installation in feet -/
def initial_separation : ℝ := 230

/-- Theorem stating that the visibility time is correct given the problem conditions -/
theorem visibility_time_correct :
  ∃ (steve_pos laura_pos : ℝ × ℝ),
    let steve_final := (steve_pos.1 + steve_speed * visibility_time, steve_pos.2)
    let laura_final := (laura_pos.1 + laura_speed * visibility_time, laura_pos.2)
    (steve_pos.2 - laura_pos.2 = path_distance) ∧
    ((steve_pos.1 - laura_pos.1)^2 + (steve_pos.2 - laura_pos.2)^2 = initial_separation^2) ∧
    (∃ (center : ℝ × ℝ), 
      (center.1 - steve_pos.1)^2 + ((center.2 - steve_pos.2) - path_distance/2)^2 = (installation_diameter/2)^2 ∧
      (center.1 - laura_pos.1)^2 + ((center.2 - laura_pos.2) + path_distance/2)^2 = (installation_diameter/2)^2) ∧
    ((steve_final.1 - laura_final.1)^2 + (steve_final.2 - laura_final.2)^2 > 
     (steve_pos.1 - laura_pos.1)^2 + (steve_pos.2 - laura_pos.2)^2) ∧
    (∀ t : ℝ, 0 < t → t < visibility_time →
      ∃ (x y : ℝ), 
        x^2 + y^2 = (installation_diameter/2)^2 ∧
        (y - steve_pos.2) * (steve_pos.1 + steve_speed * t - x) = 
        (x - steve_pos.1 - steve_speed * t) * (steve_pos.2 - y) ∧
        (y - laura_pos.2) * (laura_pos.1 + laura_speed * t - x) = 
        (x - laura_pos.1 - laura_speed * t) * (laura_pos.2 - y)) :=
by sorry

end NUMINAMATH_CALUDE_visibility_time_correct_l983_98362


namespace NUMINAMATH_CALUDE_minutes_to_seconds_l983_98345

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we're converting to seconds -/
def minutes : ℚ := 12.5

/-- Theorem stating that 12.5 minutes is equal to 750 seconds -/
theorem minutes_to_seconds : (minutes * seconds_per_minute : ℚ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_l983_98345


namespace NUMINAMATH_CALUDE_problem_1_l983_98318

theorem problem_1 (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 5) (hy : y = Real.sqrt 3 - Real.sqrt 5) :
  2 * x^2 - 4 * x * y + 2 * y^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l983_98318


namespace NUMINAMATH_CALUDE_ice_palace_staircase_steps_l983_98374

theorem ice_palace_staircase_steps 
  (time_for_20_steps : ℕ) 
  (steps_20 : ℕ) 
  (total_time : ℕ) 
  (h1 : time_for_20_steps = 120)
  (h2 : steps_20 = 20)
  (h3 : total_time = 180) :
  (total_time * steps_20) / time_for_20_steps = 30 :=
by sorry

end NUMINAMATH_CALUDE_ice_palace_staircase_steps_l983_98374


namespace NUMINAMATH_CALUDE_profit_maximized_at_100_yuan_optimal_selling_price_l983_98342

/-- Profit function given price increase -/
def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x)

/-- The price increase that maximizes profit -/
def optimal_price_increase : ℝ := 10

theorem profit_maximized_at_100_yuan :
  ∀ x : ℝ, profit x ≤ profit optimal_price_increase :=
sorry

/-- The optimal selling price is 100 yuan -/
theorem optimal_selling_price :
  90 + optimal_price_increase = 100 :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_100_yuan_optimal_selling_price_l983_98342


namespace NUMINAMATH_CALUDE_pipe_length_problem_l983_98307

theorem pipe_length_problem (total_length : ℝ) (short_length : ℝ) (long_length : ℝ) : 
  total_length = 177 →
  long_length = 2 * short_length →
  total_length = short_length + long_length →
  long_length = 118 := by
sorry

end NUMINAMATH_CALUDE_pipe_length_problem_l983_98307


namespace NUMINAMATH_CALUDE_rationalize_denominator_l983_98354

theorem rationalize_denominator :
  (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l983_98354


namespace NUMINAMATH_CALUDE_age_difference_l983_98337

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 12) : A - C = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l983_98337


namespace NUMINAMATH_CALUDE_half_meter_cut_l983_98390

theorem half_meter_cut (initial_length : ℚ) (cut_length : ℚ) (result_length : ℚ) : 
  initial_length = 8/15 →
  cut_length = 1/30 →
  result_length = initial_length - cut_length →
  result_length = 1/2 :=
by
  sorry

#check half_meter_cut

end NUMINAMATH_CALUDE_half_meter_cut_l983_98390


namespace NUMINAMATH_CALUDE_greatest_k_for_100_factorial_l983_98347

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_of_2 (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1).log2) 0

def highest_power_of_5 (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (Nat.log 5 (x + 1))) 0

theorem greatest_k_for_100_factorial (a b : ℕ) (k : ℕ) :
  a = factorial 100 →
  b = 100^k →
  (∀ m : ℕ, m > k → ¬(100^m ∣ a)) →
  (100^k ∣ a) →
  k = 12 := by sorry

end NUMINAMATH_CALUDE_greatest_k_for_100_factorial_l983_98347


namespace NUMINAMATH_CALUDE_inequality_holds_l983_98330

theorem inequality_holds (r s : ℝ) (hr : 0 ≤ r ∧ r < 2) (hs : s > 0) :
  (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) > 3 * r^2 * s := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l983_98330


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l983_98332

theorem sum_of_a_and_b (a b : ℝ) : (2*a + 2*b - 1) * (2*a + 2*b + 1) = 99 → a + b = 5 ∨ a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l983_98332


namespace NUMINAMATH_CALUDE_sum_real_coefficients_binomial_expansion_l983_98366

theorem sum_real_coefficients_binomial_expansion (i : ℂ) :
  let x : ℂ := Complex.I
  let n : ℕ := 1010
  let T : ℝ := (Finset.range (n + 1)).sum (λ k => if k % 2 = 0 then (n.choose k : ℝ) else 0)
  T = 2^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_sum_real_coefficients_binomial_expansion_l983_98366


namespace NUMINAMATH_CALUDE_sarah_today_cans_l983_98343

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- The difference in total cans collected between yesterday and today -/
def fewer_today : ℕ := 20

/-- Theorem: Sarah collected 40 cans today -/
theorem sarah_today_cans : 
  sarah_yesterday + (sarah_yesterday + lara_extra_yesterday) - fewer_today - lara_today = 40 := by
  sorry

end NUMINAMATH_CALUDE_sarah_today_cans_l983_98343


namespace NUMINAMATH_CALUDE_autograph_value_change_l983_98353

theorem autograph_value_change (initial_value : ℝ) : 
  initial_value = 100 → 
  (initial_value * (1 - 0.3) * (1 + 0.4)) = 98 := by
  sorry

end NUMINAMATH_CALUDE_autograph_value_change_l983_98353


namespace NUMINAMATH_CALUDE_heart_equation_solution_l983_98312

/-- The heart operation defined on two real numbers -/
def heart (A B : ℝ) : ℝ := 4*A + A*B + 3*B + 6

/-- Theorem stating that 60/7 is the unique solution to A ♥ 3 = 75 -/
theorem heart_equation_solution :
  ∃! A : ℝ, heart A 3 = 75 ∧ A = 60/7 := by sorry

end NUMINAMATH_CALUDE_heart_equation_solution_l983_98312


namespace NUMINAMATH_CALUDE_area_ratio_is_459_625_l983_98369

/-- Triangle XYZ with points P and Q -/
structure TriangleXYZ where
  /-- Side length XY -/
  xy : ℝ
  /-- Side length YZ -/
  yz : ℝ
  /-- Side length XZ -/
  xz : ℝ
  /-- Length XP -/
  xp : ℝ
  /-- Length XQ -/
  xq : ℝ
  /-- xy is positive -/
  xy_pos : 0 < xy
  /-- yz is positive -/
  yz_pos : 0 < yz
  /-- xz is positive -/
  xz_pos : 0 < xz
  /-- xp is positive and less than xy -/
  xp_bounds : 0 < xp ∧ xp < xy
  /-- xq is positive and less than xz -/
  xq_bounds : 0 < xq ∧ xq < xz

/-- The ratio of areas in the triangle -/
def areaRatio (t : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the ratio of areas -/
theorem area_ratio_is_459_625 (t : TriangleXYZ) 
  (h1 : t.xy = 30) (h2 : t.yz = 45) (h3 : t.xz = 51) 
  (h4 : t.xp = 18) (h5 : t.xq = 15) : 
  areaRatio t = 459 / 625 := by sorry

end NUMINAMATH_CALUDE_area_ratio_is_459_625_l983_98369


namespace NUMINAMATH_CALUDE_total_cost_is_649_70_l983_98364

/-- Calculates the total cost of a guitar and amplifier in dollars --/
def total_cost_in_dollars (guitar_price : ℝ) (amplifier_price : ℝ) 
  (guitar_discount : ℝ) (amplifier_discount : ℝ) (vat_rate : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_guitar := guitar_price * (1 - guitar_discount)
  let discounted_amplifier := amplifier_price * (1 - amplifier_discount)
  let total_with_vat := (discounted_guitar + discounted_amplifier) * (1 + vat_rate)
  total_with_vat * exchange_rate

/-- Theorem stating that the total cost is equal to $649.70 --/
theorem total_cost_is_649_70 :
  total_cost_in_dollars 330 220 0.10 0.05 0.07 1.20 = 649.70 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_649_70_l983_98364


namespace NUMINAMATH_CALUDE_initial_loss_percentage_l983_98329

-- Define the cost price of a pencil
def cost_price : ℚ := 1 / 13

-- Define the selling price when selling 20 pencils for 1 rupee
def selling_price_20 : ℚ := 1 / 20

-- Define the selling price when selling 10 pencils for 1 rupee (30% gain)
def selling_price_10 : ℚ := 1 / 10

-- Define the percentage loss
def percentage_loss : ℚ := ((cost_price - selling_price_20) / cost_price) * 100

-- Theorem stating the initial loss percentage
theorem initial_loss_percentage : 
  (selling_price_10 = cost_price + 0.3 * cost_price) → 
  (percentage_loss = 35) :=
by
  sorry

end NUMINAMATH_CALUDE_initial_loss_percentage_l983_98329


namespace NUMINAMATH_CALUDE_range_of_a_l983_98387

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 < 0) ↔ -8 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l983_98387


namespace NUMINAMATH_CALUDE_bert_stamp_collection_l983_98386

theorem bert_stamp_collection (stamps_bought : ℕ) (stamps_before : ℕ) : 
  stamps_bought = 300 →
  stamps_before = stamps_bought / 2 →
  stamps_before + stamps_bought = 450 := by
sorry

end NUMINAMATH_CALUDE_bert_stamp_collection_l983_98386


namespace NUMINAMATH_CALUDE_function_composition_equality_l983_98396

theorem function_composition_equality (A C D : ℝ) (h_C : C ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x - 3 * C^2
  let g : ℝ → ℝ := λ x ↦ C * x + D
  f (g 2) = 0 → A = (3 * C^2) / (2 * C + D) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l983_98396


namespace NUMINAMATH_CALUDE_divides_condition_l983_98331

theorem divides_condition (a b : ℕ) : 
  (a^b + b) ∣ (a^(2*b) + 2*b) ↔ 
  (a = 0) ∨ (b = 0) ∨ (a = 2 ∧ b = 1) := by
  sorry

-- Define 0^0 = 1
axiom zero_pow_zero : (0 : ℕ)^(0 : ℕ) = 1

end NUMINAMATH_CALUDE_divides_condition_l983_98331


namespace NUMINAMATH_CALUDE_congruence_properties_l983_98301

theorem congruence_properties (a b c d : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧ 
  (a - b ≡ a - c [ZMOD d]) ∧ 
  (a * b ≡ a * c [ZMOD d]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_properties_l983_98301


namespace NUMINAMATH_CALUDE_voltage_meter_max_value_l983_98357

/-- Represents a voltage meter with a maximum recordable value -/
structure VoltageMeter where
  max_value : ℝ
  records_nonnegative : 0 ≤ max_value

/-- Theorem: Given the conditions, the maximum recordable value is 14 volts -/
theorem voltage_meter_max_value (meter : VoltageMeter) 
  (avg_recording : ℝ) 
  (min_recording : ℝ) 
  (h1 : avg_recording = 6)
  (h2 : min_recording = 2)
  (h3 : ∃ (a b c : ℝ), 
    0 ≤ a ∧ a ≤ meter.max_value ∧
    0 ≤ b ∧ b ≤ meter.max_value ∧
    0 ≤ c ∧ c ≤ meter.max_value ∧
    (a + b + c) / 3 = avg_recording ∧
    min_recording ≤ a ∧ min_recording ≤ b ∧ min_recording ≤ c) :
  meter.max_value = 14 := by
sorry

end NUMINAMATH_CALUDE_voltage_meter_max_value_l983_98357


namespace NUMINAMATH_CALUDE_tangent_chord_existence_l983_98326

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a line intersects a circle to form a chord of given length -/
def formsChord (l : Line) (c : Circle) (length : ℝ) : Prop := sorry

/-- Main theorem: Given two circles and a length, there exists a tangent to the larger circle
    that forms a chord of the given length in the smaller circle -/
theorem tangent_chord_existence (largeCircle smallCircle : Circle) (chordLength : ℝ) :
  ∃ (tangentLine : Line),
    isTangent tangentLine largeCircle ∧
    formsChord tangentLine smallCircle chordLength :=
  sorry

end NUMINAMATH_CALUDE_tangent_chord_existence_l983_98326


namespace NUMINAMATH_CALUDE_sector_angle_values_l983_98310

-- Define a sector
structure Sector where
  radius : ℝ
  centralAngle : ℝ

-- Define the perimeter and area of a sector
def perimeter (s : Sector) : ℝ := 2 * s.radius + s.radius * s.centralAngle
def area (s : Sector) : ℝ := 0.5 * s.radius * s.radius * s.centralAngle

-- Theorem statement
theorem sector_angle_values :
  ∃ s : Sector, perimeter s = 6 ∧ area s = 2 ∧ (s.centralAngle = 1 ∨ s.centralAngle = 4) :=
sorry

end NUMINAMATH_CALUDE_sector_angle_values_l983_98310


namespace NUMINAMATH_CALUDE_zoo_field_trip_count_l983_98375

/-- Represents the number of individuals at the zoo during the field trip -/
def ZooFieldTrip : Type :=
  { n : ℕ // n ≤ 100 }

/-- The initial class size -/
def initial_class_size : ℕ := 10

/-- The number of parents who volunteered as chaperones -/
def parent_chaperones : ℕ := 5

/-- The number of teachers who joined -/
def teachers : ℕ := 2

/-- The number of students who left -/
def students_left : ℕ := 10

/-- The number of chaperones who left -/
def chaperones_left : ℕ := 2

/-- Function to calculate the final number of individuals at the zoo -/
def final_zoo_count (init_class : ℕ) (parents : ℕ) (teachers : ℕ) (students_gone : ℕ) (chaperones_gone : ℕ) : ZooFieldTrip :=
  ⟨2 * init_class + parents + teachers - students_gone - chaperones_gone, by sorry⟩

/-- Theorem stating that the final number of individuals at the zoo is 15 -/
theorem zoo_field_trip_count :
  (final_zoo_count initial_class_size parent_chaperones teachers students_left chaperones_left).val = 15 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_count_l983_98375


namespace NUMINAMATH_CALUDE_arcsin_arccos_inequality_l983_98340

theorem arcsin_arccos_inequality (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.arcsin ((5 / (2 * Real.pi)) * Real.arccos x) > Real.arccos ((10 / (3 * Real.pi)) * Real.arcsin x)) ↔
  (x ∈ Set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪
   Set.Ioo (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5))) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_inequality_l983_98340


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l983_98377

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Main theorem
theorem fibonacci_divisibility (A B h k : ℕ) : 
  A > 0 → B > 0 → 
  (∃ m : ℕ, B^93 = m * A^19) →
  (∃ n : ℕ, A^93 = n * B^19) →
  (∃ i : ℕ, h = fib i ∧ k = fib (i + 1)) →
  (∃ p : ℕ, (A^4 + B^8)^k = p * (A * B)^h) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l983_98377


namespace NUMINAMATH_CALUDE_standard_deviation_reflects_fluctuation_amplitude_l983_98381

/-- Standard deviation of a sample -/
def standard_deviation (sample : List ℝ) : ℝ := sorry

/-- Fluctuation amplitude of a population -/
def fluctuation_amplitude (population : List ℝ) : ℝ := sorry

/-- The standard deviation of a sample approximately reflects 
    the fluctuation amplitude of a population -/
theorem standard_deviation_reflects_fluctuation_amplitude 
  (sample : List ℝ) (population : List ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ |standard_deviation sample - fluctuation_amplitude population| < ε :=
sorry

end NUMINAMATH_CALUDE_standard_deviation_reflects_fluctuation_amplitude_l983_98381


namespace NUMINAMATH_CALUDE_M_properties_l983_98367

def M (n : ℕ+) : ℤ := (-2) ^ n.val

theorem M_properties :
  (M 5 + M 6 = 32) ∧
  (2 * M 2015 + M 2016 = 0) ∧
  (∀ n : ℕ+, 2 * M n + M (n + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_M_properties_l983_98367


namespace NUMINAMATH_CALUDE_gcd_of_198_and_286_l983_98338

theorem gcd_of_198_and_286 : Nat.gcd 198 286 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_198_and_286_l983_98338


namespace NUMINAMATH_CALUDE_first_class_students_l983_98300

/-- The number of students in the first class -/
def x : ℕ := 24

/-- The number of students in the second class -/
def second_class_students : ℕ := 50

/-- The average marks of the first class -/
def first_class_avg : ℚ := 40

/-- The average marks of the second class -/
def second_class_avg : ℚ := 60

/-- The average marks of all students combined -/
def total_avg : ℚ := 53513513513513516 / 1000000000000000

theorem first_class_students :
  (x * first_class_avg + second_class_students * second_class_avg) / (x + second_class_students) = total_avg := by
  sorry

end NUMINAMATH_CALUDE_first_class_students_l983_98300


namespace NUMINAMATH_CALUDE_linear_function_x_intercept_l983_98368

/-- A linear function f(x) = -x + 2 -/
def f (x : ℝ) : ℝ := -x + 2

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intercept : ℝ := 2

theorem linear_function_x_intercept :
  f x_intercept = 0 ∧ x_intercept = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_x_intercept_l983_98368


namespace NUMINAMATH_CALUDE_niles_win_probability_l983_98359

/-- Represents a die with six faces. -/
structure Die :=
  (faces : Fin 6 → ℕ)

/-- Billie's die -/
def billie_die : Die :=
  { faces := λ i => i.val + 1 }

/-- Niles' die -/
def niles_die : Die :=
  { faces := λ i => if i.val < 3 then 4 else 5 }

/-- The probability that Niles wins when rolling against Billie -/
def niles_win_prob : ℚ :=
  7 / 12

theorem niles_win_probability :
  let p := niles_win_prob.num
  let q := niles_win_prob.den
  7 * p + 11 * q = 181 := by sorry

end NUMINAMATH_CALUDE_niles_win_probability_l983_98359


namespace NUMINAMATH_CALUDE_lucky_years_2020_to_2024_l983_98320

def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem lucky_years_2020_to_2024 :
  isLuckyYear 2020 ∧
  isLuckyYear 2021 ∧
  isLuckyYear 2022 ∧
  ¬isLuckyYear 2023 ∧
  isLuckyYear 2024 := by
  sorry

end NUMINAMATH_CALUDE_lucky_years_2020_to_2024_l983_98320


namespace NUMINAMATH_CALUDE_circle_center_correct_l983_98392

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (1, -1)

/-- Theorem: The center of the circle defined by CircleEquation is CircleCenter -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l983_98392


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l983_98379

theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 5*x₁ + 6 = 0 ∧ x₂^2 - 5*x₂ + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l983_98379


namespace NUMINAMATH_CALUDE_smallest_valid_coloring_distance_l983_98316

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of points inside and on the edges of a regular hexagon with side length 1 -/
def S : Set Point := sorry

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- A 3-coloring of points -/
def Coloring := Point → Fin 3

/-- A valid coloring respecting the distance r -/
def valid_coloring (c : Coloring) (r : ℝ) : Prop :=
  ∀ p q : Point, p ∈ S → q ∈ S → c p = c q → distance p q < r

/-- The existence of a valid coloring -/
def exists_valid_coloring (r : ℝ) : Prop :=
  ∃ c : Coloring, valid_coloring c r

/-- The theorem stating that 3/2 is the smallest r for which a valid 3-coloring exists -/
theorem smallest_valid_coloring_distance :
  (∀ r < 3/2, ¬ exists_valid_coloring r) ∧ exists_valid_coloring (3/2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_coloring_distance_l983_98316


namespace NUMINAMATH_CALUDE_amusement_park_average_cost_l983_98389

/-- Represents the cost and trips data for a child's season pass -/
structure ChildData where
  pass_cost : ℕ
  trips : ℕ

/-- Calculates the average cost per trip given a list of ChildData -/
def average_cost_per_trip (children : List ChildData) : ℚ :=
  let total_cost := children.map (λ c => c.pass_cost) |>.sum
  let total_trips := children.map (λ c => c.trips) |>.sum
  (total_cost : ℚ) / total_trips

/-- The main theorem stating the average cost per trip for the given scenario -/
theorem amusement_park_average_cost :
  let children : List ChildData := [
    { pass_cost := 100, trips := 35 },
    { pass_cost := 90, trips := 25 },
    { pass_cost := 80, trips := 20 },
    { pass_cost := 70, trips := 15 }
  ]
  abs (average_cost_per_trip children - 3.58) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_amusement_park_average_cost_l983_98389


namespace NUMINAMATH_CALUDE_a_value_proof_l983_98344

theorem a_value_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l983_98344


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l983_98371

theorem complex_number_in_first_quadrant :
  let z : ℂ := Complex.I / (2 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l983_98371


namespace NUMINAMATH_CALUDE_divisibility_by_480_l983_98376

theorem divisibility_by_480 (n : ℤ) 
  (h2 : ¬ 2 ∣ n) 
  (h3 : ¬ 3 ∣ n) 
  (h5 : ¬ 5 ∣ n) : 
  480 ∣ (n^8 - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_480_l983_98376


namespace NUMINAMATH_CALUDE_not_necessarily_equal_proportion_l983_98388

theorem not_necessarily_equal_proportion (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a * d = b * c) : 
  ¬(∀ a b c d, (a + 1) / b = (c + 1) / d) :=
by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_equal_proportion_l983_98388


namespace NUMINAMATH_CALUDE_five_letter_word_count_l983_98350

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of five-letter words that begin and end with the same letter
    and have a vowel as the third letter -/
def word_count : ℕ := alphabet_size * alphabet_size * vowel_count * alphabet_size

theorem five_letter_word_count : word_count = 87880 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_word_count_l983_98350


namespace NUMINAMATH_CALUDE_ice_cube_calculation_l983_98383

theorem ice_cube_calculation (cubes_per_tray : ℕ) (num_trays : ℕ) 
  (h1 : cubes_per_tray = 9) 
  (h2 : num_trays = 8) : 
  cubes_per_tray * num_trays = 72 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_calculation_l983_98383


namespace NUMINAMATH_CALUDE_first_year_after_2100_digit_sum_15_l983_98349

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Check if a year is after 2100 -/
def is_after_2100 (year : ℕ) : Prop :=
  year > 2100

/-- First year after 2100 with digit sum 15 -/
def first_year_after_2100_with_digit_sum_15 : ℕ := 2139

theorem first_year_after_2100_digit_sum_15 :
  (is_after_2100 first_year_after_2100_with_digit_sum_15) ∧
  (sum_of_digits first_year_after_2100_with_digit_sum_15 = 15) ∧
  (∀ y : ℕ, is_after_2100 y ∧ y < first_year_after_2100_with_digit_sum_15 →
    sum_of_digits y ≠ 15) :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2100_digit_sum_15_l983_98349


namespace NUMINAMATH_CALUDE_no_100_digit_page_numbering_l983_98341

theorem no_100_digit_page_numbering :
  ¬ ∃ (n : ℕ), n > 0 ∧ (
    let single_digit_sum := min n 9
    let double_digit_sum := if n > 9 then 2 * (n - 9) else 0
    single_digit_sum + double_digit_sum = 100
  ) := by
  sorry

end NUMINAMATH_CALUDE_no_100_digit_page_numbering_l983_98341


namespace NUMINAMATH_CALUDE_conference_support_percentage_l983_98335

theorem conference_support_percentage
  (total_attendees : ℕ)
  (male_attendees : ℕ)
  (female_attendees : ℕ)
  (male_support_rate : ℚ)
  (female_support_rate : ℚ)
  (h1 : total_attendees = 1000)
  (h2 : male_attendees = 150)
  (h3 : female_attendees = 850)
  (h4 : male_support_rate = 70 / 100)
  (h5 : female_support_rate = 75 / 100)
  (h6 : total_attendees = male_attendees + female_attendees) :
  let total_supporters : ℚ :=
    male_support_rate * male_attendees + female_support_rate * female_attendees
  (total_supporters / total_attendees) * 100 = 74.2 := by
  sorry


end NUMINAMATH_CALUDE_conference_support_percentage_l983_98335


namespace NUMINAMATH_CALUDE_cosine_sum_less_than_sum_of_cosines_l983_98328

theorem cosine_sum_less_than_sum_of_cosines (α β : Real) 
  (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_less_than_sum_of_cosines_l983_98328


namespace NUMINAMATH_CALUDE_derivative_of_exp_neg_x_l983_98395

theorem derivative_of_exp_neg_x (x : ℝ) : 
  deriv (fun x => Real.exp (-x)) x = -Real.exp (-x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_neg_x_l983_98395


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l983_98339

theorem divisibility_of_power_plus_one (n : ℕ) :
  ∃ k : ℤ, 2^(3^n) + 1 = k * 3^(n + 1) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l983_98339


namespace NUMINAMATH_CALUDE_monthly_profit_calculation_l983_98360

/-- Calculates the monthly profit for John's computer assembly business --/
theorem monthly_profit_calculation (cost_per_computer : ℝ) (markup : ℝ) 
  (computers_per_month : ℕ) (monthly_rent : ℝ) (monthly_non_rent_expenses : ℝ) :
  cost_per_computer = 800 →
  markup = 1.4 →
  computers_per_month = 60 →
  monthly_rent = 5000 →
  monthly_non_rent_expenses = 3000 →
  let selling_price := cost_per_computer * markup
  let total_revenue := selling_price * computers_per_month
  let total_component_cost := cost_per_computer * computers_per_month
  let total_expenses := monthly_rent + monthly_non_rent_expenses
  let profit := total_revenue - total_component_cost - total_expenses
  profit = 11200 := by
sorry

end NUMINAMATH_CALUDE_monthly_profit_calculation_l983_98360


namespace NUMINAMATH_CALUDE_max_value_constraint_l983_98373

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + 2*b + 3*c = 1) :
  a + b^3 + c^4 ≤ 0.125 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l983_98373
