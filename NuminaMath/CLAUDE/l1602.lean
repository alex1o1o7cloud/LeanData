import Mathlib

namespace NUMINAMATH_CALUDE_count_perfect_square_factors_360_l1602_160253

/-- The number of perfect square factors of 360 -/
def perfect_square_factors_360 : ℕ :=
  let prime_factorization := (2, 3, 3, 2, 5, 1)
  4

theorem count_perfect_square_factors_360 :
  perfect_square_factors_360 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_360_l1602_160253


namespace NUMINAMATH_CALUDE_horner_method_operation_count_l1602_160292

/-- Polynomial coefficients in descending order of degree -/
def coefficients : List ℝ := [3, 4, -5, -6, 7, -8, 1]

/-- The point at which to evaluate the polynomial -/
def x : ℝ := 0.4

/-- Count of operations in Horner's method -/
structure OperationCount where
  multiplications : ℕ
  additions : ℕ

/-- Horner's method for polynomial evaluation -/
def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ := sorry

/-- Count operations in Horner's method -/
def count_operations (coeffs : List ℝ) : OperationCount := sorry

/-- Theorem: Horner's method for the given polynomial requires 6 multiplications and 6 additions -/
theorem horner_method_operation_count :
  let count := count_operations coefficients
  count.multiplications = 6 ∧ count.additions = 6 := by sorry

end NUMINAMATH_CALUDE_horner_method_operation_count_l1602_160292


namespace NUMINAMATH_CALUDE_unique_integer_sum_property_l1602_160266

theorem unique_integer_sum_property : ∃! (A : ℕ), A > 0 ∧ 
  ∃ (B : ℕ), B < 1000 ∧ 1000 * A + B = A * (A + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sum_property_l1602_160266


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l1602_160248

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 1

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (2, 0)
def radius2 : ℝ := 3

-- Define the distance between centers
def center_distance : ℝ := 2

-- Theorem stating that the circles are internally tangent
theorem circles_internally_tangent :
  center_distance = abs (radius2 - radius1) ∧
  center_distance < radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l1602_160248


namespace NUMINAMATH_CALUDE_function_property_l1602_160204

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the constant k
variable (k : ℝ)

-- State the theorem
theorem function_property (h1 : ∀ x : ℝ, f x + f (1 - x) = k)
                          (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x)
                          (h3 : ∀ x : ℝ, f x + f (-x) = 7) :
  k = 10 := by sorry

end NUMINAMATH_CALUDE_function_property_l1602_160204


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1602_160234

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) :
  2 * (a + 2 * b) - (3 * a + 5 * b) + 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1602_160234


namespace NUMINAMATH_CALUDE_hair_cut_ratio_l1602_160263

/-- Given the initial hair length, growth after first cut, second cut length, and final hair length,
    prove that the ratio of the initial cut to the original length is 1/2. -/
theorem hair_cut_ratio 
  (initial_length : ℝ) 
  (growth : ℝ) 
  (second_cut : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 24)
  (h2 : growth = 4)
  (h3 : second_cut = 2)
  (h4 : final_length = 14)
  : (initial_length - (final_length - growth + second_cut)) / initial_length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_ratio_l1602_160263


namespace NUMINAMATH_CALUDE_least_even_p_for_square_three_is_solution_least_even_p_is_three_l1602_160299

theorem least_even_p_for_square (p : ℕ) : 
  (∃ n : ℕ, 300 * p = n^2) ∧ Even p → p ≥ 3 :=
by sorry

theorem three_is_solution : 
  (∃ n : ℕ, 300 * 3 = n^2) ∧ Even 3 :=
by sorry

theorem least_even_p_is_three : 
  (∃ p : ℕ, (∃ n : ℕ, 300 * p = n^2) ∧ Even p ∧ 
  (∀ q : ℕ, (∃ m : ℕ, 300 * q = m^2) ∧ Even q → p ≤ q)) ∧
  (∀ p : ℕ, (∃ n : ℕ, 300 * p = n^2) ∧ Even p ∧ 
  (∀ q : ℕ, (∃ m : ℕ, 300 * q = m^2) ∧ Even q → p ≤ q) → p = 3) :=
by sorry

end NUMINAMATH_CALUDE_least_even_p_for_square_three_is_solution_least_even_p_is_three_l1602_160299


namespace NUMINAMATH_CALUDE_equation_solution_l1602_160212

theorem equation_solution (x : ℚ) : 
  x ≠ -5 → ((x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1602_160212


namespace NUMINAMATH_CALUDE_cost_price_satisfies_conditions_l1602_160217

/-- The cost price of a book satisfying the given conditions. -/
def cost_price : ℝ := 3000

/-- The selling price at 10% profit. -/
def selling_price_10_percent : ℝ := cost_price * 1.1

/-- The selling price at 15% profit. -/
def selling_price_15_percent : ℝ := cost_price * 1.15

/-- Theorem stating that the cost price satisfies the given conditions. -/
theorem cost_price_satisfies_conditions :
  (selling_price_15_percent - selling_price_10_percent = 150) ∧
  (selling_price_10_percent = cost_price * 1.1) ∧
  (selling_price_15_percent = cost_price * 1.15) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_satisfies_conditions_l1602_160217


namespace NUMINAMATH_CALUDE_tan_product_one_implies_sin_square_sum_one_l1602_160227

/-- In a triangle ABC, if tan A * tan B = 1, then sin²A + sin²B = 1, but the converse is not always true. -/
theorem tan_product_one_implies_sin_square_sum_one (A B C : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_tan_product : Real.tan A * Real.tan B = 1) : 
  (Real.sin A)^2 + (Real.sin B)^2 = 1 ∧ 
  ¬(∀ A B : ℝ, (Real.sin A)^2 + (Real.sin B)^2 = 1 → Real.tan A * Real.tan B = 1) :=
by sorry

end NUMINAMATH_CALUDE_tan_product_one_implies_sin_square_sum_one_l1602_160227


namespace NUMINAMATH_CALUDE_quadratic_two_positive_roots_l1602_160233

theorem quadratic_two_positive_roots (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (1 - a) * x^2 + (a + 2) * x - 4 = 0 ∧
    (1 - a) * y^2 + (a + 2) * y - 4 = 0) ↔
  (1 < a ∧ a ≤ 2) ∨ a ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_positive_roots_l1602_160233


namespace NUMINAMATH_CALUDE_choose_3_from_10_l1602_160268

-- Define the number of items to choose from
def n : ℕ := 10

-- Define the number of items to be chosen
def k : ℕ := 3

-- Theorem stating that choosing 3 out of 10 items results in 120 possibilities
theorem choose_3_from_10 : Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_10_l1602_160268


namespace NUMINAMATH_CALUDE_inverse_tangent_identity_l1602_160223

/-- For all real x, if g(x) = arctan(x), then g((5x - x^5) / (1 + 5x^4)) = 5g(x) - g(x)^5 -/
theorem inverse_tangent_identity (g : ℝ → ℝ) (h : ∀ x, g x = Real.arctan x) :
  ∀ x, g ((5 * x - x^5) / (1 + 5 * x^4)) = 5 * g x - (g x)^5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_tangent_identity_l1602_160223


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1602_160205

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 1 + a 9 = 180) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1602_160205


namespace NUMINAMATH_CALUDE_innings_count_l1602_160290

/-- Represents the batting statistics of a batsman -/
structure BattingStats where
  n : ℕ  -- number of innings
  T : ℕ  -- total runs
  H : ℕ  -- highest score
  L : ℕ  -- lowest score

/-- The conditions given in the problem -/
def batting_conditions (stats : BattingStats) : Prop :=
  stats.T = 63 * stats.n ∧  -- batting average is 63
  stats.H - stats.L = 150 ∧  -- difference between highest and lowest score
  stats.H = 248 ∧  -- highest score
  (stats.T - stats.H - stats.L) / (stats.n - 2) = 58  -- average excluding highest and lowest

/-- The theorem to prove -/
theorem innings_count (stats : BattingStats) : 
  batting_conditions stats → stats.n = 46 := by
  sorry


end NUMINAMATH_CALUDE_innings_count_l1602_160290


namespace NUMINAMATH_CALUDE_xy_value_l1602_160294

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^(Real.sqrt y) = 27) (h2 : (Real.sqrt x)^y = 9) : 
  x * y = 12 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l1602_160294


namespace NUMINAMATH_CALUDE_can_capacity_l1602_160254

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Represents a can with its contents and capacity -/
structure Can where
  contents : CanContents
  capacity : ℝ

/-- The theorem stating the capacity of the can given the conditions -/
theorem can_capacity (initial : CanContents) (final : CanContents) : 
  (initial.milk / initial.water = 5 / 3) →
  (final.milk / final.water = 2 / 1) →
  (final.milk = initial.milk + 8) →
  (final.water = initial.water) →
  (∃ (can : Can), can.contents = final ∧ can.capacity = 72) :=
by sorry

end NUMINAMATH_CALUDE_can_capacity_l1602_160254


namespace NUMINAMATH_CALUDE_zoe_country_albums_l1602_160225

/-- The number of pop albums Zoe bought -/
def pop_albums : ℕ := 5

/-- The number of songs per album -/
def songs_per_album : ℕ := 3

/-- The total number of songs Zoe bought -/
def total_songs : ℕ := 24

/-- The number of country albums Zoe bought -/
def country_albums : ℕ := (total_songs - pop_albums * songs_per_album) / songs_per_album

theorem zoe_country_albums : country_albums = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoe_country_albums_l1602_160225


namespace NUMINAMATH_CALUDE_cube_derivative_three_l1602_160237

theorem cube_derivative_three (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (deriv f x₀ = 3) →
  (x₀ = 1 ∨ x₀ = -1) := by
sorry

end NUMINAMATH_CALUDE_cube_derivative_three_l1602_160237


namespace NUMINAMATH_CALUDE_golf_cart_capacity_l1602_160255

/-- The number of patrons that can fit in a golf cart -/
def patrons_per_cart (patrons_from_cars : ℕ) (patrons_from_bus : ℕ) (total_carts : ℕ) : ℕ :=
  (patrons_from_cars + patrons_from_bus) / total_carts

/-- Theorem: Given the conditions from the problem, prove that 3 patrons can fit in a golf cart -/
theorem golf_cart_capacity :
  patrons_per_cart 12 27 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_golf_cart_capacity_l1602_160255


namespace NUMINAMATH_CALUDE_exists_monomial_neg5_deg2_l1602_160201

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial where
  coefficient : ℤ
  degree : ℕ

/-- Checks if a given monomial has the specified coefficient and degree. -/
def has_coeff_and_degree (m : Monomial) (c : ℤ) (d : ℕ) : Prop :=
  m.coefficient = c ∧ m.degree = d

/-- There exists a monomial with coefficient -5 and degree 2. -/
theorem exists_monomial_neg5_deg2 : ∃ m : Monomial, has_coeff_and_degree m (-5) 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_monomial_neg5_deg2_l1602_160201


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1602_160213

theorem arithmetic_geometric_sequence_ratio 
  (a b c : ℝ) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_arith : b - a = c - b) 
  (h_geom : c^2 = a * b) : 
  ∃ (k : ℝ), k ≠ 0 ∧ a = 4*k ∧ b = k ∧ c = -2*k := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1602_160213


namespace NUMINAMATH_CALUDE_november_rainfall_total_november_rainfall_l1602_160258

/-- Calculates the total rainfall in November given specific conditions -/
theorem november_rainfall (days_in_november : ℕ) 
                          (first_period : ℕ) 
                          (daily_rainfall_first_period : ℝ) 
                          (rainfall_ratio_second_period : ℝ) : ℝ :=
  let second_period := days_in_november - first_period
  let daily_rainfall_second_period := daily_rainfall_first_period * rainfall_ratio_second_period
  let total_rainfall_first_period := (first_period : ℝ) * daily_rainfall_first_period
  let total_rainfall_second_period := (second_period : ℝ) * daily_rainfall_second_period
  total_rainfall_first_period + total_rainfall_second_period

/-- The total rainfall in November is 180 inches -/
theorem total_november_rainfall : 
  november_rainfall 30 15 4 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_november_rainfall_total_november_rainfall_l1602_160258


namespace NUMINAMATH_CALUDE_exponential_function_not_in_second_quadrant_l1602_160245

/-- A function f: ℝ → ℝ does not pass through the second quadrant if for all x < 0, f(x) ≤ 0 -/
def not_in_second_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x ≤ 0

/-- The main theorem stating the condition for f(x) = 2^x + b - 1 to not pass through the second quadrant -/
theorem exponential_function_not_in_second_quadrant (b : ℝ) :
  not_in_second_quadrant (fun x ↦ 2^x + b - 1) ↔ b ≤ 0 := by
  sorry

#check exponential_function_not_in_second_quadrant

end NUMINAMATH_CALUDE_exponential_function_not_in_second_quadrant_l1602_160245


namespace NUMINAMATH_CALUDE_complex_fraction_equals_221_l1602_160219

theorem complex_fraction_equals_221 : 
  (((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324)) : ℚ) /
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) = 221 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_221_l1602_160219


namespace NUMINAMATH_CALUDE_height_after_changes_l1602_160293

-- Define the initial height in centimeters
def initial_height : ℝ := 167.64

-- Define the growth and decrease percentages
def first_growth : ℝ := 0.15
def second_growth : ℝ := 0.07
def decrease : ℝ := 0.04

-- Define the final height calculation
def final_height : ℝ :=
  initial_height * (1 + first_growth) * (1 + second_growth) * (1 - decrease)

-- State the theorem
theorem height_after_changes :
  ∃ ε > 0, |final_height - 198.03| < ε :=
sorry

end NUMINAMATH_CALUDE_height_after_changes_l1602_160293


namespace NUMINAMATH_CALUDE_ellipse_equation_l1602_160252

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

-- Define the conditions
theorem ellipse_equation :
  ∃ (a b : ℝ), 
    -- Condition 1: Foci on X-axis (implied by standard form)
    -- Condition 2: Major axis is three times minor axis
    a = 3 * b ∧
    -- Condition 3: Passes through (3,0)
    (3, 0) ∈ Ellipse a b ∧
    -- Condition 4: Center at origin (implied by standard form)
    -- Condition 5: Coordinate axes are axes of symmetry (implied by standard form)
    -- Condition 6: Passes through (√6,1) and (-√3,-√2)
    (Real.sqrt 6, 1) ∈ Ellipse a b ∧
    (-Real.sqrt 3, -Real.sqrt 2) ∈ Ellipse a b →
    -- Conclusion: The equation of the ellipse is x²/9 + y²/3 = 1
    Ellipse 3 (Real.sqrt 3) = {p : ℝ × ℝ | (p.1 ^ 2 / 9) + (p.2 ^ 2 / 3) = 1} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1602_160252


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1602_160274

-- Define the repeating decimal 2.36̄
def repeating_decimal : ℚ := 2 + 36 / 99

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 26 / 11 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1602_160274


namespace NUMINAMATH_CALUDE_length_of_trisected_segment_l1602_160257

/-- Given a line segment AD with points B, C, and M, where:
  * B and C trisect AD
  * M is one-third the way from A to D
  * The length of MC is 5
  Prove that the length of AD is 15. -/
theorem length_of_trisected_segment (A B C D M : ℝ) : 
  (B - A = C - B) ∧ (C - B = D - C) →  -- B and C trisect AD
  (M - A = (1/3) * (D - A)) →          -- M is one-third the way from A to D
  (C - M = 5) →                        -- MC = 5
  (D - A = 15) :=                      -- AD = 15
by sorry

end NUMINAMATH_CALUDE_length_of_trisected_segment_l1602_160257


namespace NUMINAMATH_CALUDE_xiaoming_ticket_arrangements_l1602_160214

/-- The number of children with 1 yuan each -/
def num_friends : ℕ := 6

/-- The minimum number of friends that must be before Xiaoming -/
def min_friends_before : ℕ := 4

/-- The total number of children including Xiaoming -/
def total_children : ℕ := num_friends + 1

/-- The number of ways to arrange the children so Xiaoming can get change -/
def valid_arrangements : ℕ := Nat.choose num_friends min_friends_before * Nat.factorial num_friends

theorem xiaoming_ticket_arrangements : valid_arrangements = 10800 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_ticket_arrangements_l1602_160214


namespace NUMINAMATH_CALUDE_convex_curves_length_area_difference_l1602_160280

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields and properties here
  length : ℝ
  area : ℝ

/-- The distance between two convex curves -/
def distance (K₁ K₂ : ConvexCurve) : ℝ := 
  sorry

theorem convex_curves_length_area_difference 
  (K₁ K₂ : ConvexCurve) (r : ℝ) (hr : distance K₁ K₂ ≤ r) :
  let L := max K₁.length K₂.length
  ∃ (L₁ L₂ S₁ S₂ : ℝ),
    L₁ = K₁.length ∧ 
    L₂ = K₂.length ∧
    S₁ = K₁.area ∧ 
    S₂ = K₂.area ∧
    |L₂ - L₁| ≤ 2 * Real.pi * r ∧
    |S₂ - S₁| ≤ L * r + Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_convex_curves_length_area_difference_l1602_160280


namespace NUMINAMATH_CALUDE_train_speed_problem_l1602_160271

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 → 
  crossing_time = 16 → 
  ∃ (speed : ℝ), speed = 27 ∧ 
    (2 * train_length) / crossing_time * 3.6 = 2 * speed := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1602_160271


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l1602_160218

def small_bottle : ℕ := 25
def medium_bottle : ℕ := 75
def large_bottle : ℕ := 600

theorem min_bottles_to_fill :
  (large_bottle / medium_bottle : ℕ) = 8 ∧
  large_bottle % medium_bottle = 0 ∧
  (∀ n : ℕ, n < 8 → n * medium_bottle < large_bottle) :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l1602_160218


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1602_160270

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0 → a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1602_160270


namespace NUMINAMATH_CALUDE_remainder_of_n_l1602_160238

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 5 = 1) 
  (h2 : n^3 % 5 = 4) : 
  n % 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l1602_160238


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_500_mod_7_l1602_160242

theorem remainder_of_3_pow_500_mod_7 : 3^500 % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_500_mod_7_l1602_160242


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1602_160239

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 235) 
  (h2 : train_speed_kmh = 64) 
  (h3 : crossing_time = 45) : 
  ∃ (bridge_length : ℝ), 
    (bridge_length ≥ 565) ∧ 
    (bridge_length < 566) :=
by
  sorry


end NUMINAMATH_CALUDE_bridge_length_calculation_l1602_160239


namespace NUMINAMATH_CALUDE_power_mod_eleven_l1602_160230

theorem power_mod_eleven : 7^308 ≡ 9 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l1602_160230


namespace NUMINAMATH_CALUDE_f_properties_l1602_160208

/-- A function with a local minimum at x = 1 -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 2*b*x

/-- The function has a local minimum of -1 at x = 1 -/
def has_local_min (a b : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - 1| < ε → f a b x ≥ f a b 1 ∧ f a b 1 = -1

/-- The range of f on [0,2] -/
def range_f (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc 0 2, f a b x = y}

theorem f_properties :
  ∃ a b : ℝ, has_local_min a b ∧ a = 1/3 ∧ b = -1/2 ∧ range_f a b = Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1602_160208


namespace NUMINAMATH_CALUDE_sequence_problem_l1602_160211

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a)
  (h_arith : is_arithmetic_sequence b)
  (h_a : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_b : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l1602_160211


namespace NUMINAMATH_CALUDE_at_least_one_genuine_certain_l1602_160283

def total_products : ℕ := 12
def genuine_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem at_least_one_genuine_certain :
  (1 : ℚ) = 1 - (defective_products.choose selected_products : ℚ) / (total_products.choose selected_products : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_genuine_certain_l1602_160283


namespace NUMINAMATH_CALUDE_value_of_2a_minus_b_l1602_160260

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_minus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →  -- h is the inverse of x + 9
  2 * a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_minus_b_l1602_160260


namespace NUMINAMATH_CALUDE_sweets_distribution_l1602_160261

theorem sweets_distribution (total_sweets : ℕ) (remaining_sweets : ℕ) (alt_children : ℕ) (alt_remaining : ℕ) :
  total_sweets = 358 →
  remaining_sweets = 8 →
  alt_children = 28 →
  alt_remaining = 22 →
  ∃ (children : ℕ), 
    children * ((total_sweets - remaining_sweets) / children) + remaining_sweets = total_sweets ∧
    alt_children * ((total_sweets - alt_remaining) / alt_children) + alt_remaining = total_sweets ∧
    children = 29 :=
by sorry

end NUMINAMATH_CALUDE_sweets_distribution_l1602_160261


namespace NUMINAMATH_CALUDE_circle_inequality_l1602_160241

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a*y₁ + b*y₂ + c*y₃ + d*y₄)^2 + (a*x₄ + b*x₃ + c*x₂ + d*x₁)^2 
    ≤ 2*((a^2 + b^2)/(a*b) + (c^2 + d^2)/(c*d)) := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l1602_160241


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l1602_160297

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Converts a Time to minutes since midnight -/
def timeToMinutes (t : Time) : Nat :=
  t.hours * 60 + t.minutes

/-- Converts minutes since midnight to Time -/
def minutesToTime (m : Nat) : Time :=
  { hours := m / 60, minutes := m % 60 }

theorem sunset_time_calculation (sunrise : Time) (daylight_length : Time) :
  let sunrise_minutes := timeToMinutes sunrise
  let daylight_minutes := timeToMinutes daylight_length
  let sunset_minutes := sunrise_minutes + daylight_minutes
  let sunset := minutesToTime sunset_minutes
  sunrise.hours = 7 ∧ sunrise.minutes = 15 ∧
  daylight_length.hours = 11 ∧ daylight_length.minutes = 36 →
  sunset.hours = 18 ∧ sunset.minutes = 51 := by
  sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l1602_160297


namespace NUMINAMATH_CALUDE_blinks_per_minute_l1602_160229

theorem blinks_per_minute (x : ℚ) 
  (h1 : x - (3/5 : ℚ) * x = 10) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_blinks_per_minute_l1602_160229


namespace NUMINAMATH_CALUDE_sydney_more_suitable_l1602_160267

/-- Represents a city with its time difference from Beijing -/
structure City where
  name : String
  timeDiff : Int

/-- Determines if a given hour is suitable for communication -/
def isSuitableHour (hour : Int) : Bool :=
  8 ≤ hour ∧ hour ≤ 22

/-- Calculates the local time given Beijing time and time difference -/
def localTime (beijingTime hour : Int) : Int :=
  (beijingTime + hour + 24) % 24

/-- Theorem: Sydney is more suitable for communication when it's 18:00 in Beijing -/
theorem sydney_more_suitable (sydney : City) (la : City) :
  sydney.name = "Sydney" →
  sydney.timeDiff = 2 →
  la.name = "Los Angeles" →
  la.timeDiff = -15 →
  isSuitableHour (localTime 18 sydney.timeDiff) ∧
  ¬isSuitableHour (localTime 18 la.timeDiff) :=
by
  sorry

#check sydney_more_suitable

end NUMINAMATH_CALUDE_sydney_more_suitable_l1602_160267


namespace NUMINAMATH_CALUDE_intersection_point_l1602_160250

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1602_160250


namespace NUMINAMATH_CALUDE_business_ownership_l1602_160206

theorem business_ownership (total_value : ℝ) (sold_fraction : ℝ) (sold_value : ℝ) 
  (h1 : total_value = 75000)
  (h2 : sold_fraction = 3/5)
  (h3 : sold_value = 15000) :
  (sold_value / sold_fraction) / total_value = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_business_ownership_l1602_160206


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l1602_160272

/-- Represents the number of trees of each type -/
structure TreeCounts where
  apricot : ℕ
  peach : ℕ
  cherry : ℕ

/-- Calculates the ratio of trees given the counts -/
def tree_ratio (counts : TreeCounts) : ℕ × ℕ × ℕ :=
  let gcd := Nat.gcd (Nat.gcd counts.apricot counts.peach) counts.cherry
  (counts.apricot / gcd, counts.peach / gcd, counts.cherry / gcd)

theorem tree_planting_ratio :
  ∀ (yard_size : ℕ) (space_per_tree : ℕ),
    yard_size = 2000 →
    space_per_tree = 10 →
    ∃ (counts : TreeCounts),
      counts.apricot = 58 ∧
      counts.peach = 3 * counts.apricot ∧
      counts.cherry = 5 * counts.peach ∧
      tree_ratio counts = (1, 3, 15) :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_ratio_l1602_160272


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_over_11_l1602_160220

theorem largest_integer_less_than_150_over_11 : 
  ∃ (x : ℤ), (∀ (y : ℤ), 11 * y < 150 → y ≤ x) ∧ (11 * x < 150) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_over_11_l1602_160220


namespace NUMINAMATH_CALUDE_min_value_expression_l1602_160202

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 2) :
  (a + b)^2 / (Real.sqrt (a^2 - 1) + Real.sqrt (b^2 - 4)) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1602_160202


namespace NUMINAMATH_CALUDE_locus_of_E_l1602_160235

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/48 + y^2/16 = 1

-- Define a point on the ellipse
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : C x y

-- Define symmetric points
def symmetric_y (p : PointOnC) : PointOnC :=
  ⟨-p.x, p.y, by sorry⟩

def symmetric_origin (p : PointOnC) : PointOnC :=
  ⟨-p.x, -p.y, by sorry⟩

def symmetric_x (p : PointOnC) : PointOnC :=
  ⟨p.x, -p.y, by sorry⟩

-- Define perpendicularity
def perpendicular (p q r s : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0

-- Define the locus of E
def locus_E (x y : ℝ) : Prop := x^2/12 + y^2/4 = 1

-- Main theorem
theorem locus_of_E (M : PointOnC) (N : PointOnC) 
  (h_diff : (M.x, M.y) ≠ (N.x, N.y))
  (h_perp : perpendicular (M.x, M.y) (N.x, N.y) (M.x, M.y) (-M.x, -M.y)) :
  ∃ (E : ℝ × ℝ), locus_E E.1 E.2 := by sorry

end NUMINAMATH_CALUDE_locus_of_E_l1602_160235


namespace NUMINAMATH_CALUDE_probability_theorem_l1602_160222

structure StudyGroup where
  total_members : ℕ
  women_percentage : ℚ
  men_percentage : ℚ
  women_lawyer_percentage : ℚ
  women_doctor_percentage : ℚ
  women_engineer_percentage : ℚ
  women_architect_percentage : ℚ
  women_finance_percentage : ℚ
  men_lawyer_percentage : ℚ
  men_doctor_percentage : ℚ
  men_engineer_percentage : ℚ
  men_architect_percentage : ℚ
  men_finance_percentage : ℚ

def probability_female_engineer_male_doctor_male_lawyer (group : StudyGroup) : ℚ :=
  group.women_percentage * group.women_engineer_percentage +
  group.men_percentage * group.men_doctor_percentage +
  group.men_percentage * group.men_lawyer_percentage

theorem probability_theorem (group : StudyGroup) 
  (h1 : group.women_percentage = 3/5)
  (h2 : group.men_percentage = 2/5)
  (h3 : group.women_engineer_percentage = 1/5)
  (h4 : group.men_doctor_percentage = 1/4)
  (h5 : group.men_lawyer_percentage = 3/10) :
  probability_female_engineer_male_doctor_male_lawyer group = 17/50 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1602_160222


namespace NUMINAMATH_CALUDE_function_value_problem_l1602_160262

theorem function_value_problem (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x - 1) :
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_problem_l1602_160262


namespace NUMINAMATH_CALUDE_tables_needed_l1602_160277

theorem tables_needed (invited : ℕ) (no_shows : ℕ) (capacity : ℕ) : 
  invited = 24 → no_shows = 10 → capacity = 7 → 
  (invited - no_shows + capacity - 1) / capacity = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tables_needed_l1602_160277


namespace NUMINAMATH_CALUDE_linda_needs_four_more_batches_l1602_160259

/-- Represents the number of cookies in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of classmates Linda has -/
def classmates : ℕ := 36

/-- Represents the number of cookies Linda wants to give each classmate -/
def cookies_per_classmate : ℕ := 12

/-- Represents the number of dozens of cookies made by one batch of chocolate chip cookies -/
def choc_chip_dozens_per_batch : ℕ := 3

/-- Represents the number of dozens of cookies made by one batch of oatmeal raisin cookies -/
def oatmeal_dozens_per_batch : ℕ := 4

/-- Represents the number of dozens of cookies made by one batch of peanut butter cookies -/
def pb_dozens_per_batch : ℕ := 5

/-- Represents the number of batches of chocolate chip cookies Linda made -/
def choc_chip_batches : ℕ := 3

/-- Represents the number of batches of oatmeal raisin cookies Linda made -/
def oatmeal_batches : ℕ := 2

/-- Calculates the total number of cookies needed -/
def total_cookies_needed : ℕ := classmates * cookies_per_classmate

/-- Calculates the number of cookies already made -/
def cookies_already_made : ℕ := 
  (choc_chip_batches * choc_chip_dozens_per_batch * dozen) + 
  (oatmeal_batches * oatmeal_dozens_per_batch * dozen)

/-- Calculates the number of cookies still needed -/
def cookies_still_needed : ℕ := total_cookies_needed - cookies_already_made

/-- Represents the number of cookies made by one batch of peanut butter cookies -/
def cookies_per_pb_batch : ℕ := pb_dozens_per_batch * dozen

/-- Theorem stating that Linda needs to bake 4 more batches of peanut butter cookies -/
theorem linda_needs_four_more_batches : 
  (cookies_still_needed + cookies_per_pb_batch - 1) / cookies_per_pb_batch = 4 := by
  sorry

end NUMINAMATH_CALUDE_linda_needs_four_more_batches_l1602_160259


namespace NUMINAMATH_CALUDE_inequality_relations_l1602_160285

theorem inequality_relations (a b : ℝ) (h : a > b) : (a - 3 > b - 3) ∧ (-4 * a < -4 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relations_l1602_160285


namespace NUMINAMATH_CALUDE_students_who_didnt_pass_l1602_160273

def total_students : ℕ := 804
def pass_percentage : ℚ := 75 / 100

theorem students_who_didnt_pass (total : ℕ) (pass_rate : ℚ) : 
  total - (total * pass_rate).floor = 201 :=
by sorry

end NUMINAMATH_CALUDE_students_who_didnt_pass_l1602_160273


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l1602_160295

theorem sine_cosine_sum (α : Real) : 
  (∃ (x y : Real), x = 3 ∧ y = -4 ∧ x = 5 * Real.cos α ∧ y = 5 * Real.sin α) →
  Real.sin α + 2 * Real.cos α = 2/5 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l1602_160295


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1602_160210

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 48) 
  (h5 : x*y + y*z + z*x = 30) : 
  x + y + z = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1602_160210


namespace NUMINAMATH_CALUDE_average_price_approximately_85_85_l1602_160236

/-- Represents a bookstore with its purchase details -/
structure Bookstore where
  books : ℕ
  total : ℚ
  discount : ℚ
  tax : ℚ
  specialDeal : ℚ

/-- Calculates the effective price per book for a given bookstore -/
def effectivePrice (store : Bookstore) : ℚ :=
  sorry

/-- The list of bookstores Rahim visited -/
def bookstores : List Bookstore := [
  { books := 25, total := 1600, discount := 15/100, tax := 5/100, specialDeal := 0 },
  { books := 35, total := 3200, discount := 0, tax := 0, specialDeal := 8/35 },
  { books := 40, total := 3800, discount := 1/100, tax := 7/100, specialDeal := 0 },
  { books := 30, total := 2400, discount := 1/60, tax := 6/100, specialDeal := 0 },
  { books := 20, total := 1800, discount := 8/100, tax := 4/100, specialDeal := 0 }
]

/-- Calculates the average price per book across all bookstores -/
def averagePrice (stores : List Bookstore) : ℚ :=
  sorry

theorem average_price_approximately_85_85 :
  abs (averagePrice bookstores - 85.85) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_price_approximately_85_85_l1602_160236


namespace NUMINAMATH_CALUDE_largest_value_with_brackets_l1602_160281

def original_expression : List Int := [2, 0, 1, 9]

def insert_brackets (expr : List Int) (pos : Nat) : Int :=
  match pos with
  | 0 => (expr[0]! - expr[1]!) - expr[2]! - expr[3]!
  | 1 => expr[0]! - (expr[1]! - expr[2]!) - expr[3]!
  | 2 => expr[0]! - expr[1]! - (expr[2]! - expr[3]!)
  | _ => 0

def all_possible_values (expr : List Int) : List Int :=
  [insert_brackets expr 0, insert_brackets expr 1, insert_brackets expr 2]

theorem largest_value_with_brackets :
  (all_possible_values original_expression).maximum? = some 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_with_brackets_l1602_160281


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1602_160296

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1602_160296


namespace NUMINAMATH_CALUDE_addition_problem_l1602_160209

theorem addition_problem (m n p q : ℕ) : 
  (1 ≤ m ∧ m ≤ 9) →
  (1 ≤ n ∧ n ≤ 9) →
  (1 ≤ p ∧ p ≤ 9) →
  (1 ≤ q ∧ q ≤ 9) →
  3 + 2 + q = 12 →
  1 + 6 + p + 8 = 24 →
  2 + n + 7 + 5 = 20 →
  m = 2 →
  m + n + p + q = 24 := by
sorry

end NUMINAMATH_CALUDE_addition_problem_l1602_160209


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1602_160282

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 11) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 184 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1602_160282


namespace NUMINAMATH_CALUDE_part_one_evaluation_part_two_evaluation_l1602_160207

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Part I
theorem part_one_evaluation : 
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) = 1/2 := by sorry

-- Part II
theorem part_two_evaluation :
  lg 14 - 2 * lg (7/3) + lg 7 - lg 18 = 0 := by sorry

end NUMINAMATH_CALUDE_part_one_evaluation_part_two_evaluation_l1602_160207


namespace NUMINAMATH_CALUDE_festival_line_up_l1602_160215

/-- Represents the minimum number of Gennadys required for the festival line-up. -/
def min_gennadys (alexanders borises vasilies : ℕ) : ℕ :=
  (borises - 1) - (alexanders + vasilies)

/-- Theorem stating the minimum number of Gennadys required for the festival. -/
theorem festival_line_up (alexanders borises vasilies : ℕ) 
  (h1 : alexanders = 45)
  (h2 : borises = 122)
  (h3 : vasilies = 27) :
  min_gennadys alexanders borises vasilies = 49 := by
  sorry

#eval min_gennadys 45 122 27

end NUMINAMATH_CALUDE_festival_line_up_l1602_160215


namespace NUMINAMATH_CALUDE_upstream_speed_l1602_160226

/-- The speed of a man rowing upstream, given his speed in still water and downstream speed -/
theorem upstream_speed (still_water_speed downstream_speed : ℝ) : 
  still_water_speed = 40 →
  downstream_speed = 48 →
  still_water_speed - (downstream_speed - still_water_speed) = 32 := by
  sorry

#check upstream_speed

end NUMINAMATH_CALUDE_upstream_speed_l1602_160226


namespace NUMINAMATH_CALUDE_sandys_shorts_cost_l1602_160298

theorem sandys_shorts_cost (total_spent shirt_cost jacket_cost : ℚ)
  (h1 : shirt_cost = 12.14)
  (h2 : jacket_cost = 7.43)
  (h3 : total_spent = 33.56) :
  total_spent - shirt_cost - jacket_cost = 13.99 := by
sorry

end NUMINAMATH_CALUDE_sandys_shorts_cost_l1602_160298


namespace NUMINAMATH_CALUDE_solution_properties_l1602_160279

-- Define the equation (1) as a function
def equation_one (a b : ℝ) : Prop := a^2 - 5*b^2 = 1

-- Theorem statement
theorem solution_properties (a b : ℝ) (h : equation_one a b) :
  (equation_one a (-b)) ∧ (1 / (a + b * Real.sqrt 5) = a - b * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_properties_l1602_160279


namespace NUMINAMATH_CALUDE_parabola_opens_downwards_l1602_160231

/-- A parabola y = (a-1)x^2 + 2x opens downwards if and only if a < 1 -/
theorem parabola_opens_downwards (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + 2*x ≤ (a - 1) * 0^2 + 2*0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_opens_downwards_l1602_160231


namespace NUMINAMATH_CALUDE_square_perimeter_from_rearranged_rectangles_l1602_160251

/-- 
Given a square cut into four equal rectangles, which are then arranged to form a shape 
with perimeter 56, prove that the perimeter of the original square is 32.
-/
theorem square_perimeter_from_rearranged_rectangles 
  (rectangle_width : ℝ) 
  (rectangle_length : ℝ) 
  (h1 : rectangle_length = 4 * rectangle_width) 
  (h2 : 28 * rectangle_width = 56) : 
  4 * (2 * rectangle_length) = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rearranged_rectangles_l1602_160251


namespace NUMINAMATH_CALUDE_inequality_proof_l1602_160284

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hx_le_1 : x ≤ 1) :
  x * y + y + 2 * z ≥ 4 * Real.sqrt (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1602_160284


namespace NUMINAMATH_CALUDE_sixth_row_correct_l1602_160228

def number_table : Nat → List Nat
  | 0 => [3, 6]
  | n + 1 => 
    let prev := number_table n
    3 :: (List.zipWith (·+·) prev (prev.tail!)) ++ [6]

theorem sixth_row_correct : 
  number_table 5 = [3, 21, 60, 90, 75, 33, 6] := by
  sorry

end NUMINAMATH_CALUDE_sixth_row_correct_l1602_160228


namespace NUMINAMATH_CALUDE_at_least_one_wrong_probability_l1602_160224

/-- The probability of getting a single multiple-choice question wrong -/
def p_wrong : ℝ := 0.1

/-- The number of multiple-choice questions -/
def n : ℕ := 3

/-- The probability of getting at least one question wrong out of n questions -/
def p_at_least_one_wrong : ℝ := 1 - (1 - p_wrong) ^ n

theorem at_least_one_wrong_probability :
  p_at_least_one_wrong = 0.271 := by sorry

end NUMINAMATH_CALUDE_at_least_one_wrong_probability_l1602_160224


namespace NUMINAMATH_CALUDE_f_is_fraction_l1602_160288

-- Define what a fraction is
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, x ≠ 0 → f x = (n x) / (d x) ∧ d x ≠ 0

-- Define the specific function we're examining
def f (x : ℚ) : ℚ := (x + 3) / x

-- Theorem statement
theorem f_is_fraction : is_fraction f := by sorry

end NUMINAMATH_CALUDE_f_is_fraction_l1602_160288


namespace NUMINAMATH_CALUDE_system_solution_l1602_160216

theorem system_solution (x y : ℝ) : 
  x - 2*y = -5 → 3*x + 6*y = 7 → x + y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1602_160216


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_existence_l1602_160249

theorem quadratic_inequality_solution_existence (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (c > 0 ∧ c < 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_existence_l1602_160249


namespace NUMINAMATH_CALUDE_butterfat_mixture_l1602_160264

theorem butterfat_mixture (x : ℝ) : 
  let initial_volume : ℝ := 8
  let initial_butterfat_percentage : ℝ := 50
  let added_volume : ℝ := 24
  let final_butterfat_percentage : ℝ := 20
  let final_volume : ℝ := initial_volume + added_volume
  let initial_butterfat : ℝ := initial_volume * (initial_butterfat_percentage / 100)
  let added_butterfat : ℝ := added_volume * (x / 100)
  let final_butterfat : ℝ := final_volume * (final_butterfat_percentage / 100)
  initial_butterfat + added_butterfat = final_butterfat → x = 10 := by
sorry

end NUMINAMATH_CALUDE_butterfat_mixture_l1602_160264


namespace NUMINAMATH_CALUDE_power_ratio_simplification_l1602_160278

theorem power_ratio_simplification : (10^2003 + 10^2001) / (10^2002 + 10^2002) = 101/20 := by
  sorry

end NUMINAMATH_CALUDE_power_ratio_simplification_l1602_160278


namespace NUMINAMATH_CALUDE_taller_cylinder_radius_l1602_160275

/-- Represents a cylindrical container --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Given two cylinders with the same volume, one with height four times the other,
    and the shorter cylinder having a radius of 10 units,
    prove that the radius of the taller cylinder is 40 units. --/
theorem taller_cylinder_radius
  (c1 c2 : Cylinder) -- Two cylindrical containers
  (h : ℝ) -- Height variable
  (volume_eq : c1.radius ^ 2 * c1.height = c2.radius ^ 2 * c2.height) -- Same volume
  (height_relation : c1.height = 4 * c2.height) -- One height is four times the other
  (shorter_radius : c2.radius = 10) -- Radius of shorter cylinder is 10 units
  : c1.radius = 40 := by
  sorry

end NUMINAMATH_CALUDE_taller_cylinder_radius_l1602_160275


namespace NUMINAMATH_CALUDE_g_1993_of_4_eq_11_26_l1602_160269

-- Define the function g
def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

-- Define the recursive function gₙ
def g_n : ℕ → (ℚ → ℚ)
| 0 => id
| (n + 1) => g ∘ (g_n n)

-- Theorem statement
theorem g_1993_of_4_eq_11_26 : g_n 1993 4 = 11/26 := by
  sorry

end NUMINAMATH_CALUDE_g_1993_of_4_eq_11_26_l1602_160269


namespace NUMINAMATH_CALUDE_job_completion_time_l1602_160256

theorem job_completion_time (rateA rateB rateC : ℝ) 
  (h1 : 3 * (rateA + rateB) = 1)
  (h2 : 6 * (rateB + rateC) = 1)
  (h3 : 3.6 * (rateA + rateC) = 1) :
  1 / rateC = 18 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1602_160256


namespace NUMINAMATH_CALUDE_value_of_D_l1602_160243

theorem value_of_D (A B C D : ℝ) 
  (h1 : A + A = 6)
  (h2 : B - A = 4)
  (h3 : C + B = 9)
  (h4 : D - C = 7)
  (h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) : 
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_value_of_D_l1602_160243


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1602_160244

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1602_160244


namespace NUMINAMATH_CALUDE_product_sum_max_l1602_160203

theorem product_sum_max (a b c d : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 5) (h4 : d = 6) :
  a * b + b * c + c * d + d * a = 63 := by sorry

end NUMINAMATH_CALUDE_product_sum_max_l1602_160203


namespace NUMINAMATH_CALUDE_quadratic_properties_is_vertex_l1602_160265

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x - 1

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧ 
  (f 2 = -5) ∧ 
  (∀ x : ℝ, f x ≥ -5) := by
  sorry

-- Define the vertex of the quadratic function
def vertex : ℝ × ℝ := (2, -5)

-- Theorem stating that the defined point is indeed the vertex
theorem is_vertex : 
  ∀ x : ℝ, f x ≥ f vertex.1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_is_vertex_l1602_160265


namespace NUMINAMATH_CALUDE_seeds_per_can_l1602_160276

def total_seeds : ℝ := 54.0
def num_cans : ℝ := 9.0

theorem seeds_per_can :
  total_seeds / num_cans = 6.0 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_can_l1602_160276


namespace NUMINAMATH_CALUDE_function_range_l1602_160246

theorem function_range (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : 
    ∀ x : ℝ, f x ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1602_160246


namespace NUMINAMATH_CALUDE_h_max_value_f_leq_g_condition_l1602_160232

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
def g (a : ℝ) (x : ℝ) : ℝ := x / (a * x + 1)
def h (x : ℝ) : ℝ := x * Real.exp (-x)

-- Theorem for the maximum value of h(x)
theorem h_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), h y ≤ h x ∧ h x = 1 / Real.exp 1 :=
sorry

-- Theorem for the range of a
theorem f_leq_g_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ g a x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_h_max_value_f_leq_g_condition_l1602_160232


namespace NUMINAMATH_CALUDE_divisibility_rule_37_l1602_160200

/-- Represents a natural number as a list of its digits -/
def toDigits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Calculates the sum for the divisibility rule of 37 -/
def divSum (digits : List ℕ) : ℤ :=
  let rec aux (l : List ℕ) (k : ℕ) : ℤ :=
    match l with
    | [] => 0
    | [a] => a
    | [a, b] => a + 10 * b
    | a :: b :: c :: rest => a + 10 * b - 11 * c + aux rest (k + 1)
  aux digits 0

/-- The divisibility rule for 37 -/
theorem divisibility_rule_37 (n : ℕ) :
  n % 37 = 0 ↔ (divSum (toDigits n)) % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_rule_37_l1602_160200


namespace NUMINAMATH_CALUDE_brick_height_is_7_point_5_cm_l1602_160221

/-- Proves that the height of a brick is 7.5 cm given the dimensions of the wall,
    the number of bricks, and the length and width of a single brick. -/
theorem brick_height_is_7_point_5_cm
  (brick_length : ℝ)
  (brick_width : ℝ)
  (wall_length : ℝ)
  (wall_width : ℝ)
  (wall_height : ℝ)
  (num_bricks : ℕ)
  (h_brick_length : brick_length = 20)
  (h_brick_width : brick_width = 10)
  (h_wall_length : wall_length = 2600)
  (h_wall_width : wall_width = 200)
  (h_wall_height : wall_height = 75)
  (h_num_bricks : num_bricks = 26000) :
  ∃ (brick_height : ℝ), brick_height = 7.5 ∧
    num_bricks * brick_length * brick_width * brick_height =
    wall_length * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_brick_height_is_7_point_5_cm_l1602_160221


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_l1602_160247

theorem absolute_value_equals_sqrt (x : ℝ) : 2 * |x| = Real.sqrt (4 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_l1602_160247


namespace NUMINAMATH_CALUDE_inverse_quadratic_sum_l1602_160289

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The inverse function f^(-1)(x) = cx^2 + bx + a -/
def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

/-- Theorem: If f and f_inv are inverse functions, then a + c = -1 -/
theorem inverse_quadratic_sum (a b c : ℝ) :
  (∀ x, f a b c (f_inv a b c x) = x) → a + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_quadratic_sum_l1602_160289


namespace NUMINAMATH_CALUDE_misha_additional_earnings_l1602_160287

/-- Represents Misha's savings situation -/
def MishaSavings (x : ℝ) (y : ℝ) (z : ℝ) : Prop :=
  x + (y / 100 * x) * z = 47

theorem misha_additional_earnings :
  ∀ (y z : ℝ), MishaSavings 34 y z → 47 - 34 = 13 := by
  sorry

end NUMINAMATH_CALUDE_misha_additional_earnings_l1602_160287


namespace NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l1602_160240

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 17 * n ≡ 5678 [ZMOD 11] → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 17 * 4 ≡ 5678 [ZMOD 11] :=
by sorry

theorem four_is_smallest : ∀ m : ℕ, m > 0 ∧ m < 4 → ¬(17 * m ≡ 5678 [ZMOD 11]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 17 * n ≡ 5678 [ZMOD 11] ∧ ∀ m : ℕ, m > 0 ∧ 17 * m ≡ 5678 [ZMOD 11] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l1602_160240


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l1602_160286

/-- Given plane vectors a, b, c, prove that k = -8 -/
theorem vector_parallel_problem (a b c : ℝ × ℝ) (k : ℝ) : 
  a = (-1, 1) →
  b = (2, 3) →
  c = (-2, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = (t * c.1, t * c.2)) →
  k = -8 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l1602_160286


namespace NUMINAMATH_CALUDE_concurrency_condition_l1602_160291

/-- An isosceles triangle ABC with geometric progressions on its sides -/
structure GeometricTriangle where
  -- Triangle side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Isosceles condition
  isosceles : BC = CA
  -- Specific side lengths
  AB_length : AB = 4
  BC_length : BC = 6
  -- Geometric progressions on sides
  X : ℕ → ℝ  -- Points on AB
  Y : ℕ → ℝ  -- Points on CB
  Z : ℕ → ℝ  -- Points on AC
  -- Geometric progression conditions
  X_gp : ∀ n : ℕ, X (n + 1) - X n = 3 * (1/4)^n
  Y_gp : ∀ n : ℕ, Y (n + 1) - Y n = 3 * (1/2)^n
  Z_gp : ∀ n : ℕ, Z (n + 1) - Z n = 3 * (1/2)^n
  -- Initial conditions
  X_init : X 0 = 0
  Y_init : Y 0 = 0
  Z_init : Z 0 = 0

/-- The concurrency condition for the triangle -/
def concurrent (T : GeometricTriangle) (a b c : ℕ+) : Prop :=
  4^c.val - 1 = (2^a.val - 1) * (2^b.val - 1)

/-- The main theorem stating the conditions for concurrency -/
theorem concurrency_condition (T : GeometricTriangle) :
  ∀ a b c : ℕ+, concurrent T a b c ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_concurrency_condition_l1602_160291
