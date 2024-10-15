import Mathlib

namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l1928_192812

/-- Parabola equation: x = 3y^2 - 9y + 5 -/
def parabola_eq (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- X-intercept of the parabola -/
def x_intercept (a : ℝ) : Prop := parabola_eq a 0

/-- Y-intercepts of the parabola -/
def y_intercepts (b c : ℝ) : Prop := parabola_eq 0 b ∧ parabola_eq 0 c ∧ b ≠ c

theorem parabola_intercepts_sum :
  ∀ a b c : ℝ, x_intercept a → y_intercepts b c → a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l1928_192812


namespace NUMINAMATH_CALUDE_jerry_stickers_l1928_192845

theorem jerry_stickers (fred_stickers : ℕ) (george_stickers : ℕ) (jerry_stickers : ℕ) : 
  fred_stickers = 18 →
  george_stickers = fred_stickers - 6 →
  jerry_stickers = 3 * george_stickers →
  jerry_stickers = 36 := by
sorry

end NUMINAMATH_CALUDE_jerry_stickers_l1928_192845


namespace NUMINAMATH_CALUDE_lee_savings_l1928_192887

theorem lee_savings (initial_savings : ℕ) (num_figures : ℕ) (price_per_figure : ℕ) (sneaker_cost : ℕ) : 
  initial_savings = 15 →
  num_figures = 10 →
  price_per_figure = 10 →
  sneaker_cost = 90 →
  initial_savings + num_figures * price_per_figure - sneaker_cost = 25 := by
sorry

end NUMINAMATH_CALUDE_lee_savings_l1928_192887


namespace NUMINAMATH_CALUDE_quadratic_rational_roots_l1928_192806

theorem quadratic_rational_roots 
  (n p q : ℚ) 
  (h : p = n + q / n) : 
  ∃ (x y : ℚ), x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_roots_l1928_192806


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1928_192841

theorem complex_modulus_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := (i / (1 - i))^2
  Complex.abs z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1928_192841


namespace NUMINAMATH_CALUDE_fraction_equality_l1928_192859

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) : 
  18 / 7 + ((2 * q - p) / (2 * q + p)) = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1928_192859


namespace NUMINAMATH_CALUDE_remainder_x_plus_2_pow_2022_l1928_192863

theorem remainder_x_plus_2_pow_2022 (x : ℤ) :
  (x^3 % (x^2 + x + 1) = 1) →
  ((x + 2)^2022 % (x^2 + x + 1) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_x_plus_2_pow_2022_l1928_192863


namespace NUMINAMATH_CALUDE_sum_A_B_equals_twice_cube_l1928_192844

/-- The sum of numbers in the nth group of positive integers -/
def A (n : ℕ) : ℕ :=
  (2 * n - 1) * (n^2 - n + 1)

/-- The difference between the latter and former number in the nth group of cubes -/
def B (n : ℕ) : ℕ :=
  3 * n^2 - 3 * n + 1

/-- Theorem stating that A_n + B_n = 2n^3 for all positive integers n -/
theorem sum_A_B_equals_twice_cube (n : ℕ) :
  A n + B n = 2 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_A_B_equals_twice_cube_l1928_192844


namespace NUMINAMATH_CALUDE_min_sum_squares_l1928_192877

/-- The polynomial equation we're considering -/
def P (a b x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + 1

/-- The condition that the polynomial has at least one real root -/
def has_real_root (a b : ℝ) : Prop := ∃ x : ℝ, P a b x = 0

/-- The theorem statement -/
theorem min_sum_squares (a b : ℝ) (h : has_real_root a b) :
  ∃ (a₀ b₀ : ℝ), has_real_root a₀ b₀ ∧ a₀^2 + b₀^2 = 4/5 ∧ 
  ∀ (a' b' : ℝ), has_real_root a' b' → a'^2 + b'^2 ≥ 4/5 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1928_192877


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_not_constant_l1928_192875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  a : Point
  b : Point
  c : Point
  equalSideLength : ℝ
  baseSideLength : ℝ

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : IsoscelesTriangle) : Prop := sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
def perpendicularDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Theorem: The sum of perpendiculars is not constant for all points inside the triangle -/
theorem sum_of_perpendiculars_not_constant (t : IsoscelesTriangle)
  (h1 : t.equalSideLength = 10)
  (h2 : t.baseSideLength = 8) :
  ∃ p1 p2 : Point,
    isInside p1 t ∧ isInside p2 t ∧
    perpendicularDistance p1 t.a t.b + perpendicularDistance p1 t.b t.c + perpendicularDistance p1 t.c t.a ≠
    perpendicularDistance p2 t.a t.b + perpendicularDistance p2 t.b t.c + perpendicularDistance p2 t.c t.a :=
by sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_not_constant_l1928_192875


namespace NUMINAMATH_CALUDE_inequality_proof_l1928_192804

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1928_192804


namespace NUMINAMATH_CALUDE_bear_laps_in_scenario_l1928_192871

/-- Represents the number of laps completed by the bear in one hour -/
def bear_laps (lake_perimeter : ℝ) (salmon1_speed : ℝ) (salmon2_speed : ℝ) (bear_speed : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of laps completed by the bear in the given scenario -/
theorem bear_laps_in_scenario : 
  bear_laps 1000 500 750 200 = 7 := by sorry

end NUMINAMATH_CALUDE_bear_laps_in_scenario_l1928_192871


namespace NUMINAMATH_CALUDE_company_y_installation_charge_l1928_192862

-- Define the given constants
def company_x_price : ℝ := 575
def company_x_surcharge_rate : ℝ := 0.04
def company_x_installation : ℝ := 82.50
def company_y_price : ℝ := 530
def company_y_surcharge_rate : ℝ := 0.03
def total_charge_difference : ℝ := 41.60

-- Define the function to calculate total cost
def total_cost (price surcharge_rate installation : ℝ) : ℝ :=
  price + price * surcharge_rate + installation

-- State the theorem
theorem company_y_installation_charge :
  ∃ (company_y_installation : ℝ),
    company_y_installation = 93 ∧
    total_cost company_x_price company_x_surcharge_rate company_x_installation -
    total_cost company_y_price company_y_surcharge_rate company_y_installation =
    total_charge_difference :=
by
  sorry

end NUMINAMATH_CALUDE_company_y_installation_charge_l1928_192862


namespace NUMINAMATH_CALUDE_complex_multiplication_l1928_192847

theorem complex_multiplication (i : ℂ) : i * i = -1 → 2 * i * (1 - i) = 2 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1928_192847


namespace NUMINAMATH_CALUDE_distance_not_equal_sum_l1928_192846

theorem distance_not_equal_sum : ∀ (a b : ℤ), 
  a = -2 ∧ b = 10 → |b - a| ≠ -2 + 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_not_equal_sum_l1928_192846


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1928_192830

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2*x + 14

theorem quadratic_minimum :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1928_192830


namespace NUMINAMATH_CALUDE_ladder_problem_l1928_192808

theorem ladder_problem (ladder_length height : Real) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : Real, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1928_192808


namespace NUMINAMATH_CALUDE_inequality_solution_l1928_192825

theorem inequality_solution (x : ℝ) : 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9 ↔ x > 45 / 26 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1928_192825


namespace NUMINAMATH_CALUDE_inequality_proof_l1928_192882

theorem inequality_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) : 
  1 / (a + b) < 1 / (a * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1928_192882


namespace NUMINAMATH_CALUDE_max_x_value_l1928_192886

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_sum_eq : x*y + x*z + y*z = 12) :
  x ≤ (13 + Real.sqrt 160) / 6 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l1928_192886


namespace NUMINAMATH_CALUDE_sam_pennies_l1928_192807

theorem sam_pennies (initial : ℕ) (spent : ℕ) (remaining : ℕ) : 
  initial = 98 → spent = 93 → remaining = initial - spent → remaining = 5 :=
by sorry

end NUMINAMATH_CALUDE_sam_pennies_l1928_192807


namespace NUMINAMATH_CALUDE_james_net_income_l1928_192833

def rental_rate : ℕ := 20
def monday_wednesday_hours : ℕ := 8
def friday_hours : ℕ := 6
def sunday_hours : ℕ := 5
def maintenance_cost : ℕ := 35
def insurance_fee : ℕ := 15
def rental_days : ℕ := 4

def total_rental_income : ℕ := 
  rental_rate * (2 * monday_wednesday_hours + friday_hours + sunday_hours)

def total_expenses : ℕ := maintenance_cost + insurance_fee * rental_days

def net_income : ℕ := total_rental_income - total_expenses

theorem james_net_income : net_income = 445 := by
  sorry

end NUMINAMATH_CALUDE_james_net_income_l1928_192833


namespace NUMINAMATH_CALUDE_gcf_of_180_240_300_l1928_192864

theorem gcf_of_180_240_300 : Nat.gcd 180 (Nat.gcd 240 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_240_300_l1928_192864


namespace NUMINAMATH_CALUDE_coupon1_best_discount_best_prices_l1928_192883

/-- Represents the discount offered by Coupon 1 -/
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

/-- Represents the discount offered by Coupon 2 -/
def coupon2_discount : ℝ := 50

/-- Represents the discount offered by Coupon 3 -/
def coupon3_discount (x : ℝ) : ℝ := 0.25 * (x - 250)

/-- Theorem stating when Coupon 1 offers the best discount -/
theorem coupon1_best_discount (x : ℝ) :
  (x ≥ 200 ∧ x ≥ 250) →
  (coupon1_discount x > coupon2_discount ∧
   coupon1_discount x > coupon3_discount x) ↔
  (333.33 < x ∧ x < 625) :=
by sorry

/-- Checks if a given price satisfies the condition for Coupon 1 being the best -/
def is_coupon1_best (price : ℝ) : Prop :=
  333.33 < price ∧ price < 625

/-- Theorem stating which of the given prices satisfy the condition -/
theorem best_prices :
  is_coupon1_best 349.95 ∧
  is_coupon1_best 399.95 ∧
  is_coupon1_best 449.95 ∧
  is_coupon1_best 499.95 ∧
  ¬is_coupon1_best 299.95 :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_discount_best_prices_l1928_192883


namespace NUMINAMATH_CALUDE_log_product_equality_l1928_192894

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log y^4) * (Real.log y^3 / Real.log x^3) *
  (Real.log x^4 / Real.log y^5) * (Real.log y^4 / Real.log x^2) *
  (Real.log x^3 / Real.log y^3) = (1/5) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l1928_192894


namespace NUMINAMATH_CALUDE_simplify_expression_l1928_192890

theorem simplify_expression (x y : ℝ) : 3*x + 4*x - 2*x + 5*y - y = 5*x + 4*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1928_192890


namespace NUMINAMATH_CALUDE_area_r_is_twelve_point_five_percent_l1928_192802

/-- Represents a circular spinner with specific properties -/
structure CircularSpinner where
  /-- Diameter PQ passes through the center -/
  has_diameter_through_center : Bool
  /-- Areas R and S are equal -/
  r_equals_s : Bool
  /-- R and S together form a quadrant -/
  r_plus_s_is_quadrant : Bool

/-- Calculates the percentage of the total area occupied by region R -/
def area_percentage_r (spinner : CircularSpinner) : ℝ :=
  sorry

/-- Theorem stating that the area of region R is 12.5% of the total circle area -/
theorem area_r_is_twelve_point_five_percent (spinner : CircularSpinner) 
  (h1 : spinner.has_diameter_through_center = true)
  (h2 : spinner.r_equals_s = true)
  (h3 : spinner.r_plus_s_is_quadrant = true) : 
  area_percentage_r spinner = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_area_r_is_twelve_point_five_percent_l1928_192802


namespace NUMINAMATH_CALUDE_regression_properties_l1928_192840

-- Define the data points
def data_points : List (ℝ × ℝ) := [(5, 17), (6, 20), (8, 25), (9, 28), (12, 35)]

-- Define the empirical regression equation
def regression_equation (x : ℝ) : ℝ := 2.6 * x + 4.2

-- Theorem to prove the three statements
theorem regression_properties :
  -- 1. The point (8, 25) lies on the regression line
  regression_equation 8 = 25 ∧
  -- 2. The y-intercept of the regression line is 4.2
  regression_equation 0 = 4.2 ∧
  -- 3. The residual for x = 5 is -0.2
  17 - regression_equation 5 = -0.2 := by
  sorry

end NUMINAMATH_CALUDE_regression_properties_l1928_192840


namespace NUMINAMATH_CALUDE_range_of_m_solution_sets_l1928_192896

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m
theorem range_of_m :
  {m : ℝ | ∀ x, y m x < 0} = Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution sets
theorem solution_sets (m : ℝ) :
  {x : ℝ | y m x < (1 - m) * x - 1} =
    if m = 0 then
      {x : ℝ | x > 0}
    else if m > 0 then
      {x : ℝ | 0 < x ∧ x < 1 / m}
    else
      {x : ℝ | x < 1 / m ∨ x > 0} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_sets_l1928_192896


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1928_192823

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ -16 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1928_192823


namespace NUMINAMATH_CALUDE_probability_three_red_jellybeans_l1928_192858

/-- Represents the probability of selecting exactly 3 red jellybeans from a bowl -/
def probability_three_red (total : ℕ) (red : ℕ) (blue : ℕ) (white : ℕ) : ℚ :=
  let total_combinations := Nat.choose total 4
  let favorable_outcomes := Nat.choose red 3 * Nat.choose (blue + white) 1
  favorable_outcomes / total_combinations

/-- Theorem stating the probability of selecting exactly 3 red jellybeans -/
theorem probability_three_red_jellybeans :
  probability_three_red 15 6 3 6 = 4 / 91 := by
  sorry

#eval probability_three_red 15 6 3 6

end NUMINAMATH_CALUDE_probability_three_red_jellybeans_l1928_192858


namespace NUMINAMATH_CALUDE_grain_distance_equation_l1928_192892

/-- The distance between the two towers in feet -/
def tower_distance : ℝ := 400

/-- The height of the church tower in feet -/
def church_tower_height : ℝ := 180

/-- The height of the cathedral tower in feet -/
def cathedral_tower_height : ℝ := 240

/-- The speed of the bird from the church tower in ft/s -/
def church_bird_speed : ℝ := 20

/-- The speed of the bird from the cathedral tower in ft/s -/
def cathedral_bird_speed : ℝ := 25

/-- The theorem stating the equation for the distance of the grain from the church tower -/
theorem grain_distance_equation (x : ℝ) :
  x ≥ 0 ∧ x ≤ tower_distance →
  25 * x = 20 * (tower_distance - x) :=
sorry

end NUMINAMATH_CALUDE_grain_distance_equation_l1928_192892


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1928_192817

/-- Expresses the repeating decimal 3.464646... as a rational number -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 46 / 99) ∧ (x = 343 / 99) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1928_192817


namespace NUMINAMATH_CALUDE_max_areas_theorem_l1928_192895

/-- Represents the number of non-overlapping areas in a circular disk -/
def max_areas (n : ℕ+) : ℕ :=
  3 * n + 3

/-- 
Theorem: The maximum number of non-overlapping areas in a circular disk 
divided by 2n equally spaced radii and two optimally placed secant lines 
is 3n + 3, where n is a positive integer.
-/
theorem max_areas_theorem (n : ℕ+) : 
  max_areas n = 3 * n + 3 := by
  sorry

#check max_areas_theorem

end NUMINAMATH_CALUDE_max_areas_theorem_l1928_192895


namespace NUMINAMATH_CALUDE_triangle_side_range_l1928_192852

theorem triangle_side_range (x : ℝ) : 
  x > 0 → 
  (4 + 5 > x ∧ 4 + x > 5 ∧ 5 + x > 4) → 
  1 < x ∧ x < 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1928_192852


namespace NUMINAMATH_CALUDE_binomial_square_polynomial_l1928_192839

theorem binomial_square_polynomial : ∃ (r s : ℝ), (r * X + s) ^ 2 = 4 * X ^ 2 + 12 * X + 9 :=
sorry

end NUMINAMATH_CALUDE_binomial_square_polynomial_l1928_192839


namespace NUMINAMATH_CALUDE_square_side_significant_digits_l1928_192827

/-- Given a square with area 0.12321 m², the number of significant digits in its side length is 5 -/
theorem square_side_significant_digits :
  ∀ (s : ℝ), 
  (s^2 ≥ 0.123205 ∧ s^2 < 0.123215) →  -- Area to the nearest ten-thousandth
  (∃ (n : ℕ), n ≥ 10000 ∧ n < 100000 ∧ s = (n : ℝ) / 100000) := by
  sorry

#check square_side_significant_digits

end NUMINAMATH_CALUDE_square_side_significant_digits_l1928_192827


namespace NUMINAMATH_CALUDE_weight_increase_percentage_l1928_192865

/-- Calculates the percentage increase in weight on the lowering portion of an exercise machine. -/
theorem weight_increase_percentage
  (num_plates : ℕ)
  (plate_weight : ℝ)
  (lowered_weight : ℝ)
  (h1 : num_plates = 10)
  (h2 : plate_weight = 30)
  (h3 : lowered_weight = 360) :
  ((lowered_weight - num_plates * plate_weight) / (num_plates * plate_weight)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_weight_increase_percentage_l1928_192865


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l1928_192843

theorem rational_inequality_solution (x : ℝ) :
  (2 * x - 1) / (x + 1) < 0 ↔ -1 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l1928_192843


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1928_192816

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  6 * x / ((x - 4) * (x - 2)^2) = 6 / (x - 4) + (-6) / (x - 2) + (-6) / (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1928_192816


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1928_192835

theorem square_area_perimeter_ratio :
  ∀ s₁ s₂ : ℝ,
  s₁ > 0 → s₂ > 0 →
  s₁^2 / s₂^2 = 16 / 81 →
  (4 * s₁) / (4 * s₂) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1928_192835


namespace NUMINAMATH_CALUDE_inequality_proof_l1928_192899

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1928_192899


namespace NUMINAMATH_CALUDE_sum_of_differences_correct_l1928_192818

def number : ℕ := 84125398

def place_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def sum_of_differences (n : ℕ) : ℕ :=
  let ones_thousands := place_value 1 3
  let ones_tens := place_value 1 1
  let eights_hundred_millions := place_value 8 8
  let eights_tens := place_value 8 1
  (eights_hundred_millions - ones_thousands) + (eights_tens - ones_tens)

theorem sum_of_differences_correct :
  sum_of_differences number = 79999070 := by sorry

end NUMINAMATH_CALUDE_sum_of_differences_correct_l1928_192818


namespace NUMINAMATH_CALUDE_polynomial_equality_l1928_192836

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (3*x + Real.sqrt 7)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1928_192836


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1928_192856

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 4 / (m - 1)) ↔ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1928_192856


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1928_192880

theorem quadratic_roots_condition (c : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + c = 0 ∧ 
  x₂^2 - 2*x₂ + c = 0 ∧ 
  7*x₂ - 4*x₁ = 47 →
  c = -15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1928_192880


namespace NUMINAMATH_CALUDE_complex_square_l1928_192829

theorem complex_square (a b : ℝ) (i : ℂ) (h : i * i = -1) (eq : a + i = 2 - b * i) :
  (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_l1928_192829


namespace NUMINAMATH_CALUDE_mixing_solutions_theorem_l1928_192849

/-- Proves that mixing 300 mL of 10% alcohol solution with 900 mL of 30% alcohol solution 
    results in a 25% alcohol solution -/
theorem mixing_solutions_theorem (x_volume y_volume : ℝ) 
  (x_concentration y_concentration final_concentration : ℝ) :
  x_volume = 300 →
  y_volume = 900 →
  x_concentration = 0.1 →
  y_concentration = 0.3 →
  final_concentration = 0.25 →
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = final_concentration :=
by
  sorry

#check mixing_solutions_theorem

end NUMINAMATH_CALUDE_mixing_solutions_theorem_l1928_192849


namespace NUMINAMATH_CALUDE_line_y_intercept_implies_m_l1928_192809

/-- Given a line equation x + my + 3 - 2m = 0 with y-intercept -1, prove that m = 1 -/
theorem line_y_intercept_implies_m (m : ℝ) :
  (∀ x y : ℝ, x + m * y + 3 - 2 * m = 0) →  -- Line equation
  (0 + m * (-1) + 3 - 2 * m = 0) →          -- y-intercept is -1
  m = 1 :=                                  -- Conclusion: m = 1
by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_implies_m_l1928_192809


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1928_192857

/-- A geometric sequence. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * q ^ (n - 1)

/-- The theorem stating the properties of the specific geometric sequence. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_diff1 : a 5 - a 1 = 15) 
    (h_diff2 : a 4 - a 2 = 6) : 
    a 3 = 4 ∨ a 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1928_192857


namespace NUMINAMATH_CALUDE_find_x_l1928_192822

def numbers : List ℕ := [201, 202, 204, 205, 206, 209, 209, 210]

theorem find_x (x : ℕ) :
  let all_numbers := numbers ++ [x]
  (all_numbers.sum / all_numbers.length : ℚ) = 207 →
  x = 217 := by sorry

end NUMINAMATH_CALUDE_find_x_l1928_192822


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l1928_192854

theorem triangle_circles_area_sum : 
  ∀ (u v w : ℝ),
  u > 0 ∧ v > 0 ∧ w > 0 →
  u + v = 6 →
  u + w = 8 →
  v + w = 10 →
  π * (u^2 + v^2 + w^2) = 56 * π :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l1928_192854


namespace NUMINAMATH_CALUDE_martine_has_sixteen_peaches_l1928_192824

/-- Given the number of peaches Gabrielle has -/
def gabrielle_peaches : ℕ := 15

/-- Benjy's peaches in terms of Gabrielle's -/
def benjy_peaches : ℕ := gabrielle_peaches / 3

/-- Martine's peaches in terms of Benjy's -/
def martine_peaches : ℕ := 2 * benjy_peaches + 6

/-- Theorem: Martine has 16 peaches -/
theorem martine_has_sixteen_peaches : martine_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_martine_has_sixteen_peaches_l1928_192824


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l1928_192848

/-- A quadratic radical is considered simpler if it cannot be further simplified 
    by extracting perfect square factors or rationalizing the denominator. -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop := sorry

theorem simplest_quadratic_radical : 
  IsSimplestQuadraticRadical (-Real.sqrt 2) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt (3/2)) ∧
  ¬IsSimplestQuadraticRadical (1 / Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l1928_192848


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l1928_192870

theorem solution_of_linear_equation :
  ∀ x : ℝ, x - 2 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l1928_192870


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1928_192803

theorem unique_integer_solution :
  ∃! (a b c : ℤ), a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1928_192803


namespace NUMINAMATH_CALUDE_count_3digit_even_no_repeat_is_360_l1928_192801

/-- A function that counts the number of 3-digit even numbers with no repeated digits -/
def count_3digit_even_no_repeat : ℕ :=
  let first_digit_options := 9  -- 1 to 9
  let second_digit_options := 8  -- Any digit except the first
  let last_digit_zero := first_digit_options * second_digit_options
  let last_digit_even_not_zero := first_digit_options * second_digit_options * 4
  last_digit_zero + last_digit_even_not_zero

/-- Theorem stating that the count of 3-digit even numbers with no repeated digits is 360 -/
theorem count_3digit_even_no_repeat_is_360 :
  count_3digit_even_no_repeat = 360 := by
  sorry

end NUMINAMATH_CALUDE_count_3digit_even_no_repeat_is_360_l1928_192801


namespace NUMINAMATH_CALUDE_lego_set_cost_lego_set_cost_is_20_l1928_192855

/-- The cost of each lego set when Tonya buys Christmas gifts for her sisters -/
theorem lego_set_cost (doll_cost : ℕ) (num_dolls : ℕ) (num_lego_sets : ℕ) : ℕ :=
  let total_doll_cost := doll_cost * num_dolls
  total_doll_cost / num_lego_sets

/-- Proof that each lego set costs $20 -/
theorem lego_set_cost_is_20 :
  lego_set_cost 15 4 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lego_set_cost_lego_set_cost_is_20_l1928_192855


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1928_192884

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1928_192884


namespace NUMINAMATH_CALUDE_raw_silk_calculation_l1928_192891

/-- The amount of raw silk that results in 12 pounds of dried silk -/
def original_raw_silk : ℚ := 96 / 7

/-- The weight loss during drying in pounds -/
def weight_loss : ℚ := 3 + 12 / 16

theorem raw_silk_calculation (initial_raw : ℚ) (dried : ℚ) 
  (h1 : initial_raw = 30)
  (h2 : dried = 12)
  (h3 : initial_raw - weight_loss = dried) :
  original_raw_silk * (initial_raw - weight_loss) = dried * initial_raw :=
sorry

end NUMINAMATH_CALUDE_raw_silk_calculation_l1928_192891


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l1928_192826

theorem triangle_side_sum_max (a b c : ℝ) (A : ℝ) :
  a = 4 → A = π / 3 → b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l1928_192826


namespace NUMINAMATH_CALUDE_total_red_pencils_l1928_192898

/-- The number of packs of colored pencils Johnny bought -/
def total_packs : ℕ := 35

/-- The number of packs with 3 extra red pencils -/
def packs_with_3_extra_red : ℕ := 7

/-- The number of packs with 2 extra blue pencils and 1 extra red pencil -/
def packs_with_2_extra_blue_1_extra_red : ℕ := 4

/-- The number of packs with 1 extra green pencil and 2 extra red pencils -/
def packs_with_1_extra_green_2_extra_red : ℕ := 10

/-- The number of red pencils in each pack without extra pencils -/
def red_pencils_per_pack : ℕ := 1

/-- Theorem: The total number of red pencils Johnny bought is 59 -/
theorem total_red_pencils : 
  total_packs * red_pencils_per_pack + 
  packs_with_3_extra_red * 3 + 
  packs_with_2_extra_blue_1_extra_red * 1 + 
  packs_with_1_extra_green_2_extra_red * 2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_red_pencils_l1928_192898


namespace NUMINAMATH_CALUDE_binomial_properties_l1928_192819

/-- A random variable following a binomial distribution B(n,p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (ξ : BinomialRV)

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- The probability of getting 0 successes in a binomial distribution -/
def prob_zero (ξ : BinomialRV) : ℝ := (1 - ξ.p) ^ ξ.n

theorem binomial_properties (ξ : BinomialRV) 
  (h2 : 3 * expectation ξ + 2 = 9.2)
  (h3 : 9 * variance ξ = 12.96) :
  ξ.n = 6 ∧ ξ.p = 0.4 ∧ prob_zero ξ = 0.6^6 := by sorry

end NUMINAMATH_CALUDE_binomial_properties_l1928_192819


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_squares_l1928_192866

theorem gcd_sum_and_sum_squares (a b : ℕ) (h : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_squares_l1928_192866


namespace NUMINAMATH_CALUDE_position_of_three_fifths_l1928_192837

def sequence_sum (n : ℕ) : ℕ := n - 1

def position_in_group (n m : ℕ) : ℕ := 
  (sequence_sum n * (sequence_sum n + 1)) / 2 + m

theorem position_of_three_fifths : 
  position_in_group 8 3 = 24 := by sorry

end NUMINAMATH_CALUDE_position_of_three_fifths_l1928_192837


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1928_192814

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- The focal length and eccentricity of a hyperbola -/
structure HyperbolaProperties where
  focal_length : ℝ
  eccentricity : ℝ

theorem hyperbola_properties (C : Hyperbola) (l : Line) :
  l.m = Real.sqrt 3 ∧ 
  l.c = -4 * Real.sqrt 3 ∧ 
  (∃ (x y : ℝ), x^2 / C.a^2 - y^2 / C.b^2 = 1 ∧ y = l.m * x + l.c) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / C.a^2 - y₁^2 / C.b^2 = 1 ∧ y₁ = l.m * x₁ + l.c ∧ 
    x₂^2 / C.a^2 - y₂^2 / C.b^2 = 1 ∧ y₂ = l.m * x₂ + l.c → 
    x₁ = x₂ ∧ y₁ = y₂) →
  ∃ (props : HyperbolaProperties), 
    props.focal_length = 8 ∧ 
    props.eccentricity = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1928_192814


namespace NUMINAMATH_CALUDE_expected_mass_of_disks_l1928_192878

/-- The expected mass of 100 metal disks with manufacturing errors -/
theorem expected_mass_of_disks (
  perfect_diameter : ℝ) 
  (perfect_mass : ℝ) 
  (radius_std_dev : ℝ) 
  (num_disks : ℕ) 
  (h1 : perfect_diameter = 1) 
  (h2 : perfect_mass = 100) 
  (h3 : radius_std_dev = 0.01) 
  (h4 : num_disks = 100) : 
  ∃ (expected_mass : ℝ), expected_mass = 10004 := by
  sorry

end NUMINAMATH_CALUDE_expected_mass_of_disks_l1928_192878


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1928_192842

theorem algebraic_expression_value : ∀ a b : ℝ,
  (a * 1^3 + b * 1 + 2022 = 2020) →
  (a * (-1)^3 + b * (-1) + 2023 = 2025) :=
by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1928_192842


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1928_192867

theorem contrapositive_equivalence (p q : Prop) : (p → q) → (¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1928_192867


namespace NUMINAMATH_CALUDE_tiger_catch_deer_distance_tiger_catch_deer_distance_is_800_l1928_192838

/-- The distance a tiger runs to catch a deer under specific conditions -/
theorem tiger_catch_deer_distance (tiger_leaps_behind : ℕ) 
  (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ)
  (tiger_meters_per_leap : ℕ) (deer_meters_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_meters_per_leap
  let tiger_distance_per_minute := tiger_leaps_per_minute * tiger_meters_per_leap
  let deer_distance_per_minute := deer_leaps_per_minute * deer_meters_per_leap
  let gain_per_minute := tiger_distance_per_minute - deer_distance_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_distance_per_minute

/-- The distance a tiger runs to catch a deer is 800 meters under the given conditions -/
theorem tiger_catch_deer_distance_is_800 : 
  tiger_catch_deer_distance 50 5 4 8 5 = 800 := by
  sorry

end NUMINAMATH_CALUDE_tiger_catch_deer_distance_tiger_catch_deer_distance_is_800_l1928_192838


namespace NUMINAMATH_CALUDE_game_correct_answers_l1928_192876

theorem game_correct_answers (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3) :
  ∃ (x : ℕ), x * correct_reward = (total_questions - x) * incorrect_penalty ∧ x = 15 := by
sorry

end NUMINAMATH_CALUDE_game_correct_answers_l1928_192876


namespace NUMINAMATH_CALUDE_foreign_language_speakers_l1928_192893

theorem foreign_language_speakers (M F : ℕ) : 
  M = F →  -- number of male students equals number of female students
  (3 : ℚ) / 5 * M + (2 : ℚ) / 3 * F = (19 : ℚ) / 30 * (M + F) := by
  sorry

end NUMINAMATH_CALUDE_foreign_language_speakers_l1928_192893


namespace NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l1928_192888

theorem power_sum_reciprocal_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
sorry

end NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l1928_192888


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1928_192869

/-- Given vectors e₁ and e₂ forming an angle of 2π/3, prove that |e₁ + 2e₂| = √3 -/
theorem magnitude_of_vector_sum (e₁ e₂ : ℝ × ℝ) : 
  e₁ • e₁ = 1 → 
  e₂ • e₂ = 1 → 
  e₁ • e₂ = -1/2 → 
  let a := e₁ + 2 • e₂ 
  ‖a‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1928_192869


namespace NUMINAMATH_CALUDE_joy_visits_grandma_l1928_192874

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours until Joy sees her grandma -/
def hours_until_visit : ℕ := 48

/-- The number of days until Joy sees her grandma -/
def days_until_visit : ℕ := hours_until_visit / hours_per_day

theorem joy_visits_grandma : days_until_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_joy_visits_grandma_l1928_192874


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l1928_192853

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope 3 and x-intercept (7, 0), the y-intercept is (0, -21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := 3, x_intercept := (7, 0) }
  y_intercept l = (0, -21) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l1928_192853


namespace NUMINAMATH_CALUDE_invisible_square_exists_l1928_192805

theorem invisible_square_exists (n : ℕ) : 
  ∃ (a b : ℤ), ∀ (i j : ℕ), i < n → j < n → Nat.gcd (Int.toNat (a + i)) (Int.toNat (b + j)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l1928_192805


namespace NUMINAMATH_CALUDE_remainder_777_444_mod_13_l1928_192821

theorem remainder_777_444_mod_13 : 777^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_444_mod_13_l1928_192821


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1928_192897

theorem quadratic_equation_m_value 
  (x₁ x₂ m : ℝ) 
  (h1 : x₁^2 - 5*x₁ + m = 0)
  (h2 : x₂^2 - 5*x₂ + m = 0)
  (h3 : 3*x₁ - 2*x₂ = 5) :
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1928_192897


namespace NUMINAMATH_CALUDE_prob_face_or_ace_two_draws_l1928_192873

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (special_cards : ℕ)

/-- Calculates the probability of drawing at least one special card in two draws with replacement -/
def prob_at_least_one_special (d : Deck) : ℚ :=
  1 - (1 - d.special_cards / d.total_cards) ^ 2

/-- Theorem: The probability of drawing at least one face card or ace in two draws with replacement from a standard 52-card deck is 88/169 -/
theorem prob_face_or_ace_two_draws :
  let standard_deck : Deck := ⟨52, 16⟩
  prob_at_least_one_special standard_deck = 88 / 169 := by
  sorry


end NUMINAMATH_CALUDE_prob_face_or_ace_two_draws_l1928_192873


namespace NUMINAMATH_CALUDE_max_f_value_l1928_192820

/-- The function f(n) is the greatest common divisor of all numbers 
    obtained by permuting the digits of n, including permutations 
    with leading zeroes. -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem: The maximum value of f(n) for positive integers n 
    where f(n) ≠ n is 81. -/
theorem max_f_value : 
  (∃ (n : ℕ+), f n = 81 ∧ f n ≠ n) ∧ 
  (∀ (n : ℕ+), f n ≠ n → f n ≤ 81) :=
sorry

end NUMINAMATH_CALUDE_max_f_value_l1928_192820


namespace NUMINAMATH_CALUDE_remainder_sum_l1928_192885

theorem remainder_sum (n : ℤ) (h : n % 18 = 4) : (n % 2 + n % 9 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1928_192885


namespace NUMINAMATH_CALUDE_tetrahedron_analogy_l1928_192834

-- Define the types of reasoning
inductive ReasoningType
  | Deductive
  | Inductive
  | Analogy

-- Define a structure for a reasoning example
structure ReasoningExample where
  description : String
  type : ReasoningType

-- Define the specific example we're interested in
def tetrahedronExample : ReasoningExample :=
  { description := "Inferring the properties of a tetrahedron in space from the properties of a plane triangle"
  , type := ReasoningType.Analogy }

-- Theorem statement
theorem tetrahedron_analogy :
  tetrahedronExample.type = ReasoningType.Analogy :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_analogy_l1928_192834


namespace NUMINAMATH_CALUDE_metal_argument_is_deductive_l1928_192832

-- Define the structure of a logical argument
structure Argument where
  premises : List String
  conclusion : String

-- Define the property of being deductive
def is_deductive (arg : Argument) : Prop :=
  ∀ (world : Type) (interpretation : String → world → Prop),
    (∀ p ∈ arg.premises, ∀ w, interpretation p w) →
    (∀ w, interpretation arg.conclusion w)

-- Define the argument about metals and uranium
def metal_argument : Argument :=
  { premises := ["All metals can conduct electricity", "Uranium is a metal"],
    conclusion := "Uranium can conduct electricity" }

-- Theorem statement
theorem metal_argument_is_deductive :
  is_deductive metal_argument :=
sorry

end NUMINAMATH_CALUDE_metal_argument_is_deductive_l1928_192832


namespace NUMINAMATH_CALUDE_triangle_cosA_value_l1928_192881

theorem triangle_cosA_value (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : (a^2 + b^2) * Real.tan C = 8 * S)
  (h2 : Real.sin A * Real.cos B = 2 * Real.cos A * Real.sin B)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : S = (1/2) * a * b * Real.sin C) :
  Real.cos A = Real.sqrt 30 / 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosA_value_l1928_192881


namespace NUMINAMATH_CALUDE_basketball_team_starters_l1928_192828

theorem basketball_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 7 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 11220 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l1928_192828


namespace NUMINAMATH_CALUDE_modulus_z_l1928_192813

theorem modulus_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : Complex.abs w = Real.sqrt 13) :
  Complex.abs z = (25 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l1928_192813


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l1928_192811

def angle_set (k : ℤ) : ℝ := k * 360 - 1560

theorem same_terminal_side_angles :
  (∃ k₁ : ℤ, angle_set k₁ = 240) ∧
  (∃ k₂ : ℤ, angle_set k₂ = -120) ∧
  (∀ α : ℝ, (∃ k : ℤ, angle_set k = α) →
    (α > 0 → α ≥ 240) ∧
    (α < 0 → α ≤ -120)) :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l1928_192811


namespace NUMINAMATH_CALUDE_tilly_star_count_l1928_192861

theorem tilly_star_count (east_stars : ℕ) : 
  east_stars + 6 * east_stars = 840 → east_stars = 120 := by
  sorry

end NUMINAMATH_CALUDE_tilly_star_count_l1928_192861


namespace NUMINAMATH_CALUDE_third_candidate_votes_l1928_192815

theorem third_candidate_votes (total_votes : ℕ) (john_votes : ℕ) (james_percentage : ℚ) : 
  total_votes = 1150 →
  john_votes = 150 →
  james_percentage = 70 / 100 →
  ∃ (third_votes : ℕ), 
    third_votes = total_votes - john_votes - (james_percentage * (total_votes - john_votes)).floor ∧
    third_votes = john_votes + 150 :=
by sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l1928_192815


namespace NUMINAMATH_CALUDE_two_numbers_sum_squares_and_product_l1928_192810

theorem two_numbers_sum_squares_and_product : ∃ u v : ℝ, 
  u^2 + v^2 = 20 ∧ u * v = 8 ∧ ((u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_squares_and_product_l1928_192810


namespace NUMINAMATH_CALUDE_work_done_cyclic_process_work_done_equals_665J_l1928_192800

/-- Represents a point in the P-T diagram -/
structure Point where
  pressure : ℝ
  temperature : ℝ

/-- Represents the cyclic process abca -/
structure CyclicProcess where
  a : Point
  b : Point
  c : Point

/-- The gas constant -/
def R : ℝ := 8.314

/-- Theorem: Work done in the cyclic process -/
theorem work_done_cyclic_process (process : CyclicProcess) : ℝ :=
  let T₀ : ℝ := 320
  have h1 : process.a.temperature = T₀ := by sorry
  have h2 : process.c.temperature = T₀ := by sorry
  have h3 : process.a.pressure = process.c.pressure / 2 := by sorry
  have h4 : process.b.pressure = process.a.pressure := by sorry
  have h5 : (process.b.temperature - process.a.temperature) * process.a.pressure > 0 := by sorry
  (1/2) * R * T₀

/-- Main theorem: The work done is equal to 665 J -/
theorem work_done_equals_665J (process : CyclicProcess) : 
  work_done_cyclic_process process = 665 := by sorry

end NUMINAMATH_CALUDE_work_done_cyclic_process_work_done_equals_665J_l1928_192800


namespace NUMINAMATH_CALUDE_prob_advance_four_shots_value_l1928_192879

/-- The probability of a successful shot -/
def p : ℝ := 0.6

/-- The probability of advancing after exactly four shots in a basketball contest -/
def prob_advance_four_shots : ℝ :=
  (1 : ℝ) * (1 - p) * p * p

/-- Theorem stating the probability of advancing after exactly four shots -/
theorem prob_advance_four_shots_value :
  prob_advance_four_shots = 18 / 125 := by
  sorry

end NUMINAMATH_CALUDE_prob_advance_four_shots_value_l1928_192879


namespace NUMINAMATH_CALUDE_total_combinations_eq_40_l1928_192889

/-- Represents the number of helper options for each day of the week --/
def helperOptions : Fin 5 → ℕ
  | 0 => 1  -- Monday
  | 1 => 2  -- Tuesday
  | 2 => 4  -- Wednesday
  | 3 => 5  -- Thursday
  | 4 => 1  -- Friday

/-- The total number of different combinations of helpers for the week --/
def totalCombinations : ℕ := (List.range 5).map helperOptions |>.prod

/-- Theorem stating that the total number of combinations is 40 --/
theorem total_combinations_eq_40 : totalCombinations = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_eq_40_l1928_192889


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_root_distance_l1928_192851

theorem quadratic_intersection_and_root_distance 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b * x₁ + c = 0 ∧ a * x₂^2 + 2*b * x₂ + c = 0) ∧
  (∀ x₁ x₂ : ℝ, a * x₁^2 + 2*b * x₁ + c = 0 → a * x₂^2 + 2*b * x₂ + c = 0 → 
    Real.sqrt 3 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_root_distance_l1928_192851


namespace NUMINAMATH_CALUDE_ticket_price_increase_l1928_192868

theorem ticket_price_increase (last_year_income : ℝ) (club_share_last_year : ℝ) 
  (club_share_this_year : ℝ) (rental_cost : ℝ) : 
  club_share_last_year = 0.1 * last_year_income →
  rental_cost = 0.9 * last_year_income →
  club_share_this_year = 0.2 →
  (((rental_cost / (1 - club_share_this_year)) / last_year_income) - 1) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_increase_l1928_192868


namespace NUMINAMATH_CALUDE_determinant_fraction_equality_l1928_192831

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem determinant_fraction_equality (θ : ℝ) : 
  det (Real.sin θ) 2 (Real.cos θ) 3 = 0 →
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_determinant_fraction_equality_l1928_192831


namespace NUMINAMATH_CALUDE_modulus_of_complex_l1928_192860

theorem modulus_of_complex (i : ℂ) (a : ℝ) : 
  i * i = -1 →
  (1 - i) * (1 - a * i) ∈ {z : ℂ | z.re = 0} →
  Complex.abs (1 - a * i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l1928_192860


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l1928_192850

theorem discount_profit_calculation (discount : ℝ) (no_discount_profit : ℝ) (with_discount_profit : ℝ) :
  discount = 0.04 →
  no_discount_profit = 0.4375 →
  with_discount_profit = (1 + no_discount_profit) * (1 - discount) - 1 →
  with_discount_profit = 0.38 := by
sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l1928_192850


namespace NUMINAMATH_CALUDE_cricket_team_size_l1928_192872

/-- A cricket team with the following properties:
  * The team is 25 years old
  * The wicket keeper is 3 years older than the team
  * The average age of the remaining players (excluding team and wicket keeper) is 1 year less than the average age of the whole team
  * The average age of the team is 22 years
-/
structure CricketTeam where
  n : ℕ  -- number of team members
  team_age : ℕ
  wicket_keeper_age : ℕ
  avg_age : ℕ
  team_age_eq : team_age = 25
  wicket_keeper_age_eq : wicket_keeper_age = team_age + 3
  avg_age_eq : avg_age = 22
  remaining_avg_age : (n * avg_age - team_age - wicket_keeper_age) / (n - 2) = avg_age - 1

/-- The number of members in the cricket team is 11 -/
theorem cricket_team_size (team : CricketTeam) : team.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l1928_192872
