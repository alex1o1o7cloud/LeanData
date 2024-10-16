import Mathlib

namespace NUMINAMATH_CALUDE_fence_posts_for_grazing_area_l3830_383067

/-- The minimum number of fence posts required to enclose a rectangular area -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * length + width
  let num_intervals := perimeter / post_spacing
  num_intervals + 1

theorem fence_posts_for_grazing_area :
  min_fence_posts 60 36 12 = 12 :=
sorry

end NUMINAMATH_CALUDE_fence_posts_for_grazing_area_l3830_383067


namespace NUMINAMATH_CALUDE_fraction_comparison_l3830_383006

theorem fraction_comparison : (5555553 : ℚ) / 5555557 > (6666664 : ℚ) / 6666669 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3830_383006


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3830_383094

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity √3,
    where the line x = -a²/c (c is the semi-latus rectum) coincides with the latus rectum
    of the parabola y² = 4x, prove that the equation of this hyperbola is x²/3 - y²/6 = 1. -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (c / a = Real.sqrt 3) → (a^2 / c = 1) → (x^2 / a^2 - y^2 / b^2 = 1) →
  (x^2 / 3 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3830_383094


namespace NUMINAMATH_CALUDE_double_value_points_range_l3830_383033

/-- A point (k, 2k) is a double value point -/
def DoubleValuePoint (k : ℝ) : ℝ × ℝ := (k, 2 * k)

/-- The quadratic function -/
def QuadraticFunction (t s : ℝ) (x : ℝ) : ℝ := (t + 1) * x^2 + (t + 2) * x + s

theorem double_value_points_range (t s : ℝ) (h : t ≠ -1) :
  (∀ k₁ k₂ : ℝ, k₁ ≠ k₂ → 
    ∃ (p₁ p₂ : ℝ × ℝ), p₁ = DoubleValuePoint k₁ ∧ p₂ = DoubleValuePoint k₂ ∧
    QuadraticFunction t s (p₁.1) = p₁.2 ∧ QuadraticFunction t s (p₂.1) = p₂.2) →
  -1 < s ∧ s < 0 := by
  sorry

end NUMINAMATH_CALUDE_double_value_points_range_l3830_383033


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l3830_383037

theorem inscribed_square_area_ratio : 
  ∀ (large_square_side : ℝ) (inscribed_square_side : ℝ),
    large_square_side = 4 →
    inscribed_square_side = 2 →
    (inscribed_square_side^2) / (large_square_side^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l3830_383037


namespace NUMINAMATH_CALUDE_prob_same_color_is_17_25_l3830_383064

def num_green_balls : ℕ := 8
def num_red_balls : ℕ := 2
def total_balls : ℕ := num_green_balls + num_red_balls

def prob_same_color : ℚ := (num_green_balls / total_balls)^2 + (num_red_balls / total_balls)^2

theorem prob_same_color_is_17_25 : prob_same_color = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_17_25_l3830_383064


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_triangles_l3830_383019

/-- Represents a trapezoid with given area and bases -/
structure Trapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Represents the areas of triangles formed by diagonals in a trapezoid -/
structure DiagonalTriangles where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- 
Given a trapezoid with area 3 and bases 1 and 2, 
the areas of the triangles formed by its diagonals are 1/3, 2/3, 2/3, and 4/3
-/
theorem trapezoid_diagonal_triangles (t : Trapezoid) 
  (h1 : t.area = 3) 
  (h2 : t.base1 = 1) 
  (h3 : t.base2 = 2) : 
  ∃ (d : DiagonalTriangles), 
    d.area1 = 1/3 ∧ 
    d.area2 = 2/3 ∧ 
    d.area3 = 2/3 ∧ 
    d.area4 = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_diagonal_triangles_l3830_383019


namespace NUMINAMATH_CALUDE_course_selection_ways_l3830_383044

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_ways (type_a type_b total : ℕ) : 
  type_a = 3 → type_b = 4 → total = 3 →
  (choose type_a 2 * choose type_b 1 + choose type_a 1 * choose type_b 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_ways_l3830_383044


namespace NUMINAMATH_CALUDE_function_equation_zero_l3830_383061

theorem function_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y = f (f x * f y)) : 
  ∀ x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_function_equation_zero_l3830_383061


namespace NUMINAMATH_CALUDE_expression_evaluation_l3830_383028

theorem expression_evaluation : (25 - (3010 - 260)) * (1500 - (100 - 25)) = -3885625 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3830_383028


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3830_383032

theorem complex_magnitude_proof : 
  Complex.abs ((11/13 : ℂ) + (12/13 : ℂ) * Complex.I)^12 = (Real.sqrt 265 / 13)^12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3830_383032


namespace NUMINAMATH_CALUDE_rectangle_tangent_circles_l3830_383029

/-- Given a rectangle ABCD with side lengths a and b, and two externally tangent circles
    inside the rectangle, one tangent to AB and AD, the other tangent to CB and CD,
    this theorem proves properties about the distance between circle centers and
    the locus of their tangency point. -/
theorem rectangle_tangent_circles
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let d := (Real.sqrt a - Real.sqrt b) ^ 2
  let m := min a b
  let p₁ := (a - m / 2, b - m / 2)
  let p₂ := (m / 2 + Real.sqrt (2 * a * b) - b, m / 2 + Real.sqrt (2 * a * b) - a)
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    -- c₁ and c₂ are the centers of the circles
    -- r₁ and r₂ are the radii of the circles
    -- The circles are inside the rectangle
    c₁.1 ∈ Set.Icc 0 a ∧ c₁.2 ∈ Set.Icc 0 b ∧
    c₂.1 ∈ Set.Icc 0 a ∧ c₂.2 ∈ Set.Icc 0 b ∧
    -- The circles are tangent to the sides of the rectangle
    c₁.1 = r₁ ∧ c₁.2 = r₁ ∧
    c₂.1 = a - r₂ ∧ c₂.2 = b - r₂ ∧
    -- The circles are externally tangent to each other
    (c₁.1 - c₂.1) ^ 2 + (c₁.2 - c₂.2) ^ 2 = (r₁ + r₂) ^ 2 ∧
    -- The distance between the centers is d
    (c₁.1 - c₂.1) ^ 2 + (c₁.2 - c₂.2) ^ 2 = d ^ 2 ∧
    -- The locus of the tangency point is a line segment
    ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧
      let p := (1 - t) • p₁ + t • p₂
      p.1 = (r₁ * (a - r₁ - r₂)) / (r₁ + r₂) + r₁ ∧
      p.2 = (r₁ * (b - r₁ - r₂)) / (r₁ + r₂) + r₁ :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_tangent_circles_l3830_383029


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l3830_383066

theorem prime_square_minus_one_divisible_by_thirty (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  30 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l3830_383066


namespace NUMINAMATH_CALUDE_bob_discount_percentage_l3830_383023

def bob_bill : ℝ := 30
def kate_bill : ℝ := 25
def total_after_discount : ℝ := 53

theorem bob_discount_percentage :
  let total_before_discount := bob_bill + kate_bill
  let discount_amount := total_before_discount - total_after_discount
  let discount_percentage := (discount_amount / bob_bill) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |discount_percentage - 6.67| < ε :=
sorry

end NUMINAMATH_CALUDE_bob_discount_percentage_l3830_383023


namespace NUMINAMATH_CALUDE_inequality_solution_l3830_383014

theorem inequality_solution (x : ℝ) : 
  (x^2 + 1)/(x-2) ≥ 3/(x+2) + 2/3 ↔ x ∈ Set.Ioo (-2 : ℝ) (5/3 : ℝ) ∪ Set.Ioi (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3830_383014


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3830_383087

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordered numbers
  b = 10 ∧  -- Median is 10
  (a + b + c) / 3 = a + 15 ∧  -- Mean is 15 more than least
  (a + b + c) / 3 = c - 20  -- Mean is 20 less than greatest
  → a + b + c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3830_383087


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l3830_383008

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (x^3 - 18*x^2 + 19*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) → 
  (1 + a) * (1 + b) * (1 + c) = 46 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l3830_383008


namespace NUMINAMATH_CALUDE_instrument_probability_l3830_383024

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 2 / 5 →
  two_or_more = 96 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l3830_383024


namespace NUMINAMATH_CALUDE_oscar_review_questions_l3830_383027

/-- Calculates the total number of questions Professor Oscar must review. -/
def total_questions (questions_per_exam : ℕ) (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  questions_per_exam * num_classes * students_per_class

/-- Proves that Professor Oscar must review 1750 questions in total. -/
theorem oscar_review_questions :
  total_questions 10 5 35 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_oscar_review_questions_l3830_383027


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l3830_383088

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y : ℝ, y > 0 ∧ y < 1/6 → (y - 3) / (12 * y^2 - 50 * y + 12) ≠ 0) ∧ 
  ((1/6 : ℝ) - 3) / (12 * (1/6)^2 - 50 * (1/6) + 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l3830_383088


namespace NUMINAMATH_CALUDE_widget_sales_sum_l3830_383003

def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 1

def sum_arithmetic_sequence (n : ℕ) : ℕ :=
  n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

theorem widget_sales_sum :
  sum_arithmetic_sequence 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_widget_sales_sum_l3830_383003


namespace NUMINAMATH_CALUDE_logarithm_difference_equals_two_l3830_383056

theorem logarithm_difference_equals_two :
  (Real.log 80 / Real.log 2) / (Real.log 40 / Real.log 2) -
  (Real.log 160 / Real.log 2) / (Real.log 20 / Real.log 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_difference_equals_two_l3830_383056


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3830_383054

/-- The area of a rectangle with perimeter 20 meters and one side length x meters --/
def rectangle_area (x : ℝ) : ℝ := x * (10 - x)

/-- Theorem: The area of a rectangle with perimeter 20 meters and one side length x meters is x(10 - x) square meters --/
theorem rectangle_area_theorem (x : ℝ) (h : x > 0 ∧ x < 10) : 
  rectangle_area x = x * (10 - x) ∧ 
  2 * (x + (10 - x)) = 20 := by
  sorry

#check rectangle_area_theorem

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3830_383054


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_one_l3830_383071

theorem at_least_one_not_greater_than_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b ≤ 1) ∨ (b / c ≤ 1) ∨ (c / a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_one_l3830_383071


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3830_383015

theorem trigonometric_inequality (a b A B : ℝ) :
  (∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) →
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3830_383015


namespace NUMINAMATH_CALUDE_count_solutions_power_diff_l3830_383039

/-- The number of solutions to x^n - y^n = 2^100 where x, y, n are positive integers and n > 1 -/
theorem count_solutions_power_diff : 
  (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      let (x, y, n) := t
      x > 0 ∧ y > 0 ∧ n > 1 ∧ x^n - y^n = 2^100)
    (Finset.product (Finset.range (2^100 + 1)) 
      (Finset.product (Finset.range (2^100 + 1)) (Finset.range 101)))).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_power_diff_l3830_383039


namespace NUMINAMATH_CALUDE_target_miss_probability_l3830_383020

theorem target_miss_probability 
  (p_I p_II p_III : ℝ) 
  (h_I : p_I = 0.35) 
  (h_II : p_II = 0.30) 
  (h_III : p_III = 0.25) : 
  1 - (p_I + p_II + p_III) = 0.1 := by
sorry

end NUMINAMATH_CALUDE_target_miss_probability_l3830_383020


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4290_l3830_383005

theorem largest_prime_factor_of_4290 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4290 ∧ ∀ q, Nat.Prime q → q ∣ 4290 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4290_l3830_383005


namespace NUMINAMATH_CALUDE_max_m_for_right_angle_l3830_383073

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = 2 * x + 2 * m

-- Define the rectangular coordinates of circle C
def circle_C_rect (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2^2

-- Theorem statement
theorem max_m_for_right_angle (m : ℝ) :
  (∃ x y : ℝ, circle_C_rect x y ∧ line_l x y m) →
  m ≤ Real.sqrt 5 - 2 :=
sorry

end NUMINAMATH_CALUDE_max_m_for_right_angle_l3830_383073


namespace NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l3830_383053

/-- Given a line segment with one endpoint (6, 2) and midpoint (5, 7),
    the sum of the coordinates of the other endpoint is 16. -/
theorem endpoint_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 2) ∧
    midpoint = (5, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 16

/-- Proof of the theorem -/
theorem endpoint_sum_proof : ∃ (endpoint2 : ℝ × ℝ), endpoint_sum (6, 2) (5, 7) endpoint2 :=
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l3830_383053


namespace NUMINAMATH_CALUDE_modulus_of_z_l3830_383011

theorem modulus_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := i / (1 + i)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3830_383011


namespace NUMINAMATH_CALUDE_satellite_selection_probabilities_l3830_383031

/-- The number of geostationary Earth orbit (GEO) satellites -/
def num_geo : ℕ := 3

/-- The number of inclined geosynchronous orbit (IGSO) satellites -/
def num_igso : ℕ := 3

/-- The total number of satellites to select -/
def num_select : ℕ := 2

/-- The probability of selecting exactly one GEO satellite and one IGSO satellite -/
def prob_one_geo_one_igso : ℚ := 3/5

/-- The probability of selecting at least one IGSO satellite -/
def prob_at_least_one_igso : ℚ := 4/5

theorem satellite_selection_probabilities :
  (num_geo = 3 ∧ num_igso = 3 ∧ num_select = 2) →
  (prob_one_geo_one_igso = 3/5 ∧ prob_at_least_one_igso = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_satellite_selection_probabilities_l3830_383031


namespace NUMINAMATH_CALUDE_black_fraction_of_triangle_l3830_383051

theorem black_fraction_of_triangle (total_parts : ℕ) (black_parts : ℕ) :
  total_parts = 64 →
  black_parts = 27 →
  (black_parts : ℚ) / total_parts = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_black_fraction_of_triangle_l3830_383051


namespace NUMINAMATH_CALUDE_B_spend_percent_is_85_percent_l3830_383097

def total_salary : ℝ := 7000
def A_salary : ℝ := 5250
def A_spend_percent : ℝ := 0.95

def B_salary : ℝ := total_salary - A_salary
def A_savings : ℝ := A_salary * (1 - A_spend_percent)

theorem B_spend_percent_is_85_percent :
  ∃ (B_spend_percent : ℝ),
    B_spend_percent = 0.85 ∧
    A_savings = B_salary * (1 - B_spend_percent) := by
  sorry

end NUMINAMATH_CALUDE_B_spend_percent_is_85_percent_l3830_383097


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3830_383086

-- Define the polynomial and divisor
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 9*x - 6
def g (x : ℝ) : ℝ := x^2 - x + 4

-- Define the quotient and remainder
def q (x : ℝ) : ℝ := x - 3
def r (x : ℝ) : ℝ := 2*x + 6

-- Theorem statement
theorem polynomial_division_theorem :
  ∀ x : ℝ, f x = g x * q x + r x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3830_383086


namespace NUMINAMATH_CALUDE_boat_distance_calculation_l3830_383043

theorem boat_distance_calculation (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 2)
  (h3 : total_time = 960) :
  ∃ distance : ℝ, 
    distance = 7590 ∧ 
    total_time = distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_calculation_l3830_383043


namespace NUMINAMATH_CALUDE_mean_temperature_l3830_383041

def temperatures : List ℝ := [82, 80, 83, 88, 84, 90, 92, 85, 89, 90]

theorem mean_temperature (temps := temperatures) : 
  (temps.sum / temps.length : ℝ) = 86.3 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3830_383041


namespace NUMINAMATH_CALUDE_number_division_remainder_l3830_383013

theorem number_division_remainder (n : ℕ) : 
  (n / 8 = 8 ∧ n % 8 = 0) → n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainder_l3830_383013


namespace NUMINAMATH_CALUDE_hexagonal_prism_edge_sum_specific_l3830_383090

/-- Calculates the sum of lengths of all edges of a regular hexagonal prism -/
def hexagonal_prism_edge_sum (base_side_length : ℝ) (height : ℝ) : ℝ :=
  2 * (6 * base_side_length) + 6 * height

theorem hexagonal_prism_edge_sum_specific : 
  hexagonal_prism_edge_sum 6 11 = 138 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_edge_sum_specific_l3830_383090


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3830_383042

/-- A quadratic function f(x) = ax^2 + bx + 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The solution set of f(x) > 0 -/
def solution_set (a b : ℝ) : Set ℝ := {x | f a b x > 0}

theorem quadratic_inequality_solution (a b : ℝ) :
  solution_set a b = {x | -1 < x ∧ x < 1/3} → a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3830_383042


namespace NUMINAMATH_CALUDE_f_10_eq_3_div_5_l3830_383035

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) (h : x > 0) : f x = 2 * f (1/x) * Real.log x + 1

theorem f_10_eq_3_div_5 : f 10 = 3/5 := by sorry

end NUMINAMATH_CALUDE_f_10_eq_3_div_5_l3830_383035


namespace NUMINAMATH_CALUDE_staircase_cube_construction_l3830_383082

/-- A staircase-brick with 3 steps of width 2, made of 12 unit cubes -/
structure StaircaseBrick where
  steps : Nat
  width : Nat
  volume : Nat
  steps_eq : steps = 3
  width_eq : width = 2
  volume_eq : volume = 12

/-- Predicate to check if a cube of side n can be built using staircase-bricks -/
def canBuildCube (n : Nat) : Prop :=
  ∃ (k : Nat), n^3 = k * 12

/-- Theorem stating that a cube of side n can be built using staircase-bricks
    if and only if n is a multiple of 12 -/
theorem staircase_cube_construction (n : Nat) :
  canBuildCube n ↔ ∃ (m : Nat), n = 12 * m :=
by sorry

end NUMINAMATH_CALUDE_staircase_cube_construction_l3830_383082


namespace NUMINAMATH_CALUDE_toll_constant_is_half_dollar_l3830_383059

/-- The number of axles on an 18-wheel truck with 2 wheels on its front axle and 4 wheels on each other axle -/
def truck_axles : ℕ := 5

/-- The toll formula for a truck -/
def toll (constant : ℝ) (x : ℕ) : ℝ := 2.50 + constant * (x - 2)

/-- The theorem stating that the constant in the toll formula is 0.50 -/
theorem toll_constant_is_half_dollar :
  ∃ (constant : ℝ), toll constant truck_axles = 4 ∧ constant = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_toll_constant_is_half_dollar_l3830_383059


namespace NUMINAMATH_CALUDE_max_value_problem_l3830_383075

theorem max_value_problem (x y : ℝ) 
  (h1 : x - y ≥ 2) 
  (h2 : x + y ≤ 3) 
  (h3 : x ≥ 0) 
  (h4 : y ≥ 0) : 
  ∃ (z : ℝ), z = 6 ∧ ∀ (w : ℝ), w = 2*x - 3*y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l3830_383075


namespace NUMINAMATH_CALUDE_quadratic_equation_with_prime_coefficients_l3830_383099

theorem quadratic_equation_with_prime_coefficients (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x : ℤ, x^2 - p*x + q = 0) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_prime_coefficients_l3830_383099


namespace NUMINAMATH_CALUDE_roots_are_imaginary_l3830_383009

theorem roots_are_imaginary (k : ℝ) : 
  let quadratic (x : ℝ) := x^2 - 3*k*x + 2*k^2 - 1
  ∀ r₁ r₂ : ℝ, quadratic r₁ = 0 ∧ quadratic r₂ = 0 → r₁ * r₂ = 8 →
  ∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ x : ℝ, quadratic x = 0 ↔ x = Complex.mk a b ∨ x = Complex.mk a (-b)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_imaginary_l3830_383009


namespace NUMINAMATH_CALUDE_entrance_exam_correct_answers_l3830_383091

theorem entrance_exam_correct_answers 
  (total_questions : ℕ) 
  (correct_marks : ℤ) 
  (wrong_marks : ℤ) 
  (total_score : ℤ) : 
  total_questions = 70 → 
  correct_marks = 3 → 
  wrong_marks = -1 → 
  total_score = 38 → 
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_marks + (total_questions - correct_answers) * wrong_marks = total_score ∧ 
    correct_answers = 27 := by
  sorry

end NUMINAMATH_CALUDE_entrance_exam_correct_answers_l3830_383091


namespace NUMINAMATH_CALUDE_dads_dimes_proof_l3830_383063

/-- The number of dimes Melanie's dad gave her -/
def dads_dimes : ℕ := 83 - (19 + 25)

/-- Melanie's initial number of dimes -/
def initial_dimes : ℕ := 19

/-- Number of dimes Melanie's mother gave her -/
def mothers_dimes : ℕ := 25

/-- Melanie's total number of dimes after receiving from both parents -/
def total_dimes : ℕ := 83

theorem dads_dimes_proof : 
  dads_dimes = total_dimes - (initial_dimes + mothers_dimes) := by
  sorry

end NUMINAMATH_CALUDE_dads_dimes_proof_l3830_383063


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l3830_383080

def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_equation_solution (x : ℕ) : 
  (3 * A 8 x = 4 * A 9 (x - 1)) → x ≤ 8 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l3830_383080


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3830_383089

/-- A geometric sequence with specified terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_2 : a 2 = 2)
  (h_3 : a 3 = -4) :
  a 5 = -16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3830_383089


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l3830_383001

theorem trigonometric_inequalities (α β γ : ℝ) (h : α + β + γ = 0) :
  (|Real.cos (α + β)| ≤ |Real.cos α| + |Real.sin β|) ∧
  (|Real.sin (α + β)| ≤ |Real.cos α| + |Real.cos β|) ∧
  (|Real.cos α| + |Real.cos β| + |Real.cos γ| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l3830_383001


namespace NUMINAMATH_CALUDE_olivia_supermarket_spending_l3830_383069

/-- The amount Olivia spent at the supermarket -/
def amount_spent (initial_amount : ℕ) (amount_left : ℕ) : ℕ :=
  initial_amount - amount_left

/-- Theorem: Olivia spent $25 at the supermarket -/
theorem olivia_supermarket_spending :
  amount_spent 54 29 = 25 := by
  sorry

end NUMINAMATH_CALUDE_olivia_supermarket_spending_l3830_383069


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3830_383092

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of Chromium atoms in the compound -/
def num_Cr : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := 
  (num_H : ℝ) * atomic_weight_H + 
  (num_Cr : ℝ) * atomic_weight_Cr + 
  (num_O : ℝ) * atomic_weight_O

theorem molecular_weight_calculation : 
  molecular_weight = 118.008 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3830_383092


namespace NUMINAMATH_CALUDE_equation_D_is_linear_l3830_383002

/-- Definition of a linear equation in two variables -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

/-- The specific equation we want to prove is linear -/
def equation_D (x y : ℝ) : ℝ := 2 * x + y - 5

/-- Theorem stating that equation_D is a linear equation in two variables -/
theorem equation_D_is_linear : is_linear_equation equation_D := by
  sorry


end NUMINAMATH_CALUDE_equation_D_is_linear_l3830_383002


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3830_383018

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 3*x < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3830_383018


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3830_383007

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f (y * f x - 1) = x^2 * f y - f x) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3830_383007


namespace NUMINAMATH_CALUDE_percentage_calculation_l3830_383004

theorem percentage_calculation (p : ℝ) : 
  (p / 100) * 2348 / 4.98 = 528.0642570281125 → 
  ∃ ε > 0, |p - 112| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3830_383004


namespace NUMINAMATH_CALUDE_fractional_to_linear_equation_l3830_383072

/-- Given the fractional equation 2/x = 1/(x-1), prove that multiplying both sides
    by x(x-1) results in a linear equation. -/
theorem fractional_to_linear_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  ∃ (a b : ℝ), x * (x - 1) * (2 / x) = a * x + b :=
sorry

end NUMINAMATH_CALUDE_fractional_to_linear_equation_l3830_383072


namespace NUMINAMATH_CALUDE_perfect_square_15AB9_l3830_383045

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_form_15AB9 (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ n = 15000 + A * 100 + B * 10 + 9

theorem perfect_square_15AB9 (n : ℕ) (h1 : is_five_digit n) (h2 : has_form_15AB9 n) (h3 : is_perfect_square n) :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ n = 15000 + A * 100 + B * 10 + 9 ∧ A + B = 3 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_15AB9_l3830_383045


namespace NUMINAMATH_CALUDE_oddSum_not_prime_l3830_383095

def oddSum (n : Nat) : Nat :=
  List.sum (List.map (fun i => 2 * i - 1) (List.range n))

theorem oddSum_not_prime (n : Nat) (h : 2 ≤ n ∧ n ≤ 5) : ¬ Nat.Prime (oddSum n) := by
  sorry

end NUMINAMATH_CALUDE_oddSum_not_prime_l3830_383095


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l3830_383084

/-- Represents a 5x5x5 cube with painted faces -/
structure PaintedCube where
  size : Nat
  painted_squares_per_face : Nat
  total_cubes : Nat
  painted_pattern_size : Nat

/-- Calculates the number of unpainted cubes in the PaintedCube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_cubes - (cube.painted_squares_per_face * 6 - (cube.painted_pattern_size - 1) * 4 * 3)

/-- Theorem stating that the number of unpainted cubes is 83 -/
theorem unpainted_cubes_count (cube : PaintedCube) 
  (h1 : cube.size = 5)
  (h2 : cube.painted_squares_per_face = 9)
  (h3 : cube.total_cubes = 125)
  (h4 : cube.painted_pattern_size = 3) : 
  unpainted_cubes cube = 83 := by
  sorry

#eval unpainted_cubes { size := 5, painted_squares_per_face := 9, total_cubes := 125, painted_pattern_size := 3 }

end NUMINAMATH_CALUDE_unpainted_cubes_count_l3830_383084


namespace NUMINAMATH_CALUDE_players_joined_equals_two_l3830_383010

/-- The number of players who joined an online game --/
def players_joined (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  (total_lives / lives_per_player) - initial_players

/-- Theorem: The number of players who joined the game is 2 --/
theorem players_joined_equals_two :
  players_joined 2 6 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_players_joined_equals_two_l3830_383010


namespace NUMINAMATH_CALUDE_cookies_milk_proportion_l3830_383048

/-- Given that 24 cookies require 5 quarts of milk and 1 quart equals 4 cups,
    prove that 8 cookies require 20/3 cups of milk. -/
theorem cookies_milk_proportion :
  let cookies_24 : ℕ := 24
  let quarts_24 : ℕ := 5
  let cups_per_quart : ℕ := 4
  let cookies_8 : ℕ := 8
  cookies_24 * (cups_per_quart * quarts_24) / cookies_8 = 20 / 3 := by sorry

end NUMINAMATH_CALUDE_cookies_milk_proportion_l3830_383048


namespace NUMINAMATH_CALUDE_min_value_of_y_l3830_383046

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 2) :
  ∀ y : ℝ, y = 4*a + b → y ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_y_l3830_383046


namespace NUMINAMATH_CALUDE_number_of_lineups_l3830_383017

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def cant_play_together : ℕ := 3
def injured : ℕ := 1

theorem number_of_lineups :
  (Nat.choose (team_size - cant_play_together - injured) lineup_size) +
  (cant_play_together * Nat.choose (team_size - cant_play_together - injured) (lineup_size - 1)) = 1452 :=
by sorry

end NUMINAMATH_CALUDE_number_of_lineups_l3830_383017


namespace NUMINAMATH_CALUDE_odd_function_interval_l3830_383012

/-- A function f is odd on an interval [a, b] if and only if
    the interval is symmetric around the origin -/
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f (-x) = -f x ∧ -x ∈ Set.Icc a b

theorem odd_function_interval (f : ℝ → ℝ) (b : ℝ) :
  is_odd_on_interval f (b - 1) 2 → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_interval_l3830_383012


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_neg_two_l3830_383096

/-- A quadratic function with given coordinate values -/
structure QuadraticFunction where
  f : ℝ → ℝ
  coords : List (ℝ × ℝ)
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The axis of symmetry of a quadratic function -/
def axis_of_symmetry (qf : QuadraticFunction) : ℝ := sorry

/-- The given quadratic function from the problem -/
def given_function : QuadraticFunction where
  f := sorry
  coords := [(-3, -3), (-2, -2), (-1, -3), (0, -6), (1, -11)]
  is_quadratic := sorry

/-- Theorem stating that the axis of symmetry of the given function is -2 -/
theorem axis_of_symmetry_is_neg_two :
  axis_of_symmetry given_function = -2 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_neg_two_l3830_383096


namespace NUMINAMATH_CALUDE_current_speed_calculation_l3830_383098

-- Define the given conditions
def downstream_distance : ℝ := 96
def downstream_time : ℝ := 8
def upstream_distance : ℝ := 8
def upstream_time : ℝ := 2

-- Define the speed of the current
def current_speed : ℝ := 4

-- Theorem statement
theorem current_speed_calculation :
  let boat_speed := (downstream_distance / downstream_time + upstream_distance / upstream_time) / 2
  current_speed = boat_speed - upstream_distance / upstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_speed_calculation_l3830_383098


namespace NUMINAMATH_CALUDE_lawn_area_20_l3830_383077

/-- The area of a rectangular lawn with given width and length -/
def lawn_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of a rectangular lawn with width 5 feet and length 4 feet is 20 square feet -/
theorem lawn_area_20 : lawn_area 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lawn_area_20_l3830_383077


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3830_383025

theorem fraction_subtraction : 
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (6 : ℚ) / 10 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3830_383025


namespace NUMINAMATH_CALUDE_old_phone_plan_cost_l3830_383057

theorem old_phone_plan_cost 
  (new_plan_cost : ℝ) 
  (price_increase_percentage : ℝ) 
  (h1 : new_plan_cost = 195) 
  (h2 : price_increase_percentage = 0.30) : 
  new_plan_cost / (1 + price_increase_percentage) = 150 := by
sorry

end NUMINAMATH_CALUDE_old_phone_plan_cost_l3830_383057


namespace NUMINAMATH_CALUDE_total_people_in_program_l3830_383021

theorem total_people_in_program (parents pupils teachers staff volunteers : ℕ) 
  (h1 : parents = 105)
  (h2 : pupils = 698)
  (h3 : teachers = 35)
  (h4 : staff = 20)
  (h5 : volunteers = 50) :
  parents + pupils + teachers + staff + volunteers = 908 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l3830_383021


namespace NUMINAMATH_CALUDE_box_weight_is_42_l3830_383016

/-- The weight of a box of books -/
def box_weight (book_weight : ℕ) (num_books : ℕ) : ℕ :=
  book_weight * num_books

/-- Theorem: The weight of a box of books is 42 pounds -/
theorem box_weight_is_42 :
  box_weight 3 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_is_42_l3830_383016


namespace NUMINAMATH_CALUDE_root_interval_implies_a_range_l3830_383052

theorem root_interval_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 3 ∧ x^2 - 3*x + a = 0) →
  0 < a ∧ a ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_root_interval_implies_a_range_l3830_383052


namespace NUMINAMATH_CALUDE_bonnets_theorem_l3830_383022

def bonnets_problem (monday thursday friday : ℕ) : Prop :=
  let tuesday_wednesday := 2 * monday
  let total_mon_to_thu := monday + tuesday_wednesday + thursday
  let total_sent := 11 * 5
  thursday = monday + 5 ∧
  total_sent = total_mon_to_thu + friday ∧
  thursday - friday = 5

theorem bonnets_theorem : 
  ∃ (monday thursday friday : ℕ), 
    monday = 10 ∧ 
    bonnets_problem monday thursday friday :=
sorry

end NUMINAMATH_CALUDE_bonnets_theorem_l3830_383022


namespace NUMINAMATH_CALUDE_banana_bread_recipe_l3830_383055

/-- Banana bread recipe problem -/
theorem banana_bread_recipe 
  (bananas_per_mush : ℕ) 
  (total_bananas : ℕ) 
  (total_flour : ℕ) 
  (h1 : bananas_per_mush = 4)
  (h2 : total_bananas = 20)
  (h3 : total_flour = 15) :
  (total_flour : ℚ) / ((total_bananas : ℚ) / (bananas_per_mush : ℚ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_recipe_l3830_383055


namespace NUMINAMATH_CALUDE_circle_circumference_radius_increase_l3830_383026

/-- If the circumference of a circle increases by 0.628 cm, then its radius increases by 0.1 cm. -/
theorem circle_circumference_radius_increase : 
  ∀ (r : ℝ) (Δr : ℝ), 
  2 * Real.pi * Δr = 0.628 → Δr = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_radius_increase_l3830_383026


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3830_383000

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3830_383000


namespace NUMINAMATH_CALUDE_min_value_expression_l3830_383030

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a * b * c) ≥ 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3830_383030


namespace NUMINAMATH_CALUDE_diagonal_cut_square_dimensions_l3830_383060

/-- Given a square with side length 10 units that is cut diagonally,
    prove that the resulting triangles have dimensions 10, 10, and 10√2 units. -/
theorem diagonal_cut_square_dimensions :
  let square_side : ℝ := 10
  let diagonal : ℝ := square_side * Real.sqrt 2
  ∀ triangle : Set (ℝ × ℝ × ℝ),
    (∃ (a b c : ℝ), triangle = {(a, b, c)} ∧
      a = square_side ∧
      b = square_side ∧
      c = diagonal) →
    triangle = {(10, 10, 10 * Real.sqrt 2)} :=
by sorry

end NUMINAMATH_CALUDE_diagonal_cut_square_dimensions_l3830_383060


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3830_383050

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3830_383050


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3830_383068

theorem unique_solution_exponential_equation :
  ∃! y : ℝ, (3 : ℝ) ^ (4 * y + 2) * (9 : ℝ) ^ (2 * y + 3) = (27 : ℝ) ^ (3 * y + 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3830_383068


namespace NUMINAMATH_CALUDE_parabola_points_order_l3830_383062

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 9

-- Define the points on the parabola
def A : ℝ × ℝ := (-2, f (-2))
def B : ℝ × ℝ := (1, f 1)
def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_points_order : y₃ > y₂ ∧ y₂ > y₁ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_order_l3830_383062


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l3830_383079

theorem arctan_sum_special_case : 
  ∀ (a b : ℝ), 
    a = -1/3 → 
    (2*a + 1)*(2*b + 1) = 1 → 
    Real.arctan a + Real.arctan b = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l3830_383079


namespace NUMINAMATH_CALUDE_y_derivative_l3830_383081

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x + Real.sqrt x + 2

theorem y_derivative (x : ℝ) (h : x ≠ 0) : 
  deriv y x = (x * Real.cos x - Real.sin x) / x^2 + 1 / (2 * Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3830_383081


namespace NUMINAMATH_CALUDE_function_bound_l3830_383085

theorem function_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : 
  |a * x^2 + x - a| ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_function_bound_l3830_383085


namespace NUMINAMATH_CALUDE_angle_xpy_is_45_deg_l3830_383036

/-- A rectangle WXYZ with a point P on side WZ -/
structure RectangleWithPoint where
  /-- Length of side WZ -/
  wz : ℝ
  /-- Length of side XY -/
  xy : ℝ
  /-- Distance from W to P -/
  wp : ℝ
  /-- Angle WPY in radians -/
  angle_wpy : ℝ
  /-- Angle XPY in radians -/
  angle_xpy : ℝ
  /-- WZ is positive -/
  wz_pos : 0 < wz
  /-- XY is positive -/
  xy_pos : 0 < xy
  /-- P is on WZ -/
  wp_le_wz : 0 ≤ wp ∧ wp ≤ wz
  /-- Sine ratio condition -/
  sine_ratio : Real.sin angle_wpy / Real.sin angle_xpy = 2

/-- Theorem: If WZ = 8, XY = 4, and the sine ratio condition holds, then ∠XPY = 45° -/
theorem angle_xpy_is_45_deg (r : RectangleWithPoint) 
  (h1 : r.wz = 8) (h2 : r.xy = 4) : r.angle_xpy = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_xpy_is_45_deg_l3830_383036


namespace NUMINAMATH_CALUDE_component_reliability_l3830_383076

/-- Represents the service life of an electronic component in years -/
def ServiceLife : Type := ℝ

/-- The probability that a single electronic component works normally for more than 9 years -/
def ProbSingleComponentWorksOver9Years : ℝ := 0.2

/-- The number of electronic components in parallel -/
def NumComponents : ℕ := 3

/-- The probability that the component (made up of 3 parallel electronic components) 
    can work normally for more than 9 years -/
def ProbComponentWorksOver9Years : ℝ :=
  1 - (1 - ProbSingleComponentWorksOver9Years) ^ NumComponents

theorem component_reliability :
  ProbComponentWorksOver9Years = 0.488 :=
sorry

end NUMINAMATH_CALUDE_component_reliability_l3830_383076


namespace NUMINAMATH_CALUDE_gcd_properties_l3830_383038

theorem gcd_properties (a b : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  (Nat.gcd (a + b).natAbs (a * b).natAbs = 1 ∧
   Nat.gcd (a - b).natAbs (a * b).natAbs = 1) ∧
  (Nat.gcd (a + b).natAbs (a - b).natAbs = 1 ∨
   Nat.gcd (a + b).natAbs (a - b).natAbs = 2) := by
  sorry

end NUMINAMATH_CALUDE_gcd_properties_l3830_383038


namespace NUMINAMATH_CALUDE_sum_of_eight_numbers_l3830_383093

/-- Given a list of 8 real numbers with an average of 5.7, prove that their sum is 45.6 -/
theorem sum_of_eight_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 8)
  (h2 : numbers.sum / numbers.length = 5.7) : 
  numbers.sum = 45.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eight_numbers_l3830_383093


namespace NUMINAMATH_CALUDE_f_10_eq_756_l3830_383065

/-- The polynomial function f(x) = x^3 - 2x^2 - 5x + 6 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

/-- Theorem: f(10) = 756 -/
theorem f_10_eq_756 : f 10 = 756 := by
  sorry

end NUMINAMATH_CALUDE_f_10_eq_756_l3830_383065


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l3830_383040

/-- Represents a rectangular park with given dimensions and fencing cost -/
structure RectangularPark where
  ratio : Rat  -- Ratio of longer side to shorter side
  area : ℝ     -- Area in square meters
  fencingCost : ℝ  -- Cost of fencing per meter in paise

/-- Calculates the cost of fencing a rectangular park -/
def calculateFencingCost (park : RectangularPark) : ℝ :=
  sorry

/-- Theorem: The cost of fencing the given park is 175 rupees -/
theorem fencing_cost_theorem (park : RectangularPark) 
  (h1 : park.ratio = 3/2)
  (h2 : park.area = 7350)
  (h3 : park.fencingCost = 50) : 
  calculateFencingCost park = 175 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l3830_383040


namespace NUMINAMATH_CALUDE_worker_count_l3830_383078

theorem worker_count : ∃ (x : ℕ), 
  x > 0 ∧ 
  (7200 / x + 400) * (x - 3) = 7200 ∧ 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_worker_count_l3830_383078


namespace NUMINAMATH_CALUDE_expanded_figure_perimeter_l3830_383049

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the figure composed of squares -/
structure ExpandedFigure where
  squares : List Square
  bottomRowCount : ℕ
  topRowCount : ℕ

/-- Calculates the perimeter of the expanded figure -/
def perimeter (figure : ExpandedFigure) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem expanded_figure_perimeter :
  ∀ (figure : ExpandedFigure),
    (∀ s ∈ figure.squares, s.sideLength = 2) →
    figure.bottomRowCount = 3 →
    figure.topRowCount = 1 →
    figure.squares.length = 4 →
    perimeter figure = 20 :=
  sorry

end NUMINAMATH_CALUDE_expanded_figure_perimeter_l3830_383049


namespace NUMINAMATH_CALUDE_parallelogram_area_l3830_383083

/-- The area of a parallelogram with base 28 cm and height 32 cm is 896 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 28 → 
  height = 32 → 
  area = base * height →
  area = 896 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3830_383083


namespace NUMINAMATH_CALUDE_expression_value_l3830_383070

theorem expression_value (a b c d x : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |x| = 3) : 
  3 * (a + b) - (-c * d) ^ 2021 + x = 4 ∨ 3 * (a + b) - (-c * d) ^ 2021 + x = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3830_383070


namespace NUMINAMATH_CALUDE_even_sum_probability_l3830_383047

-- Define the possible outcomes for each spinner
def X : Finset ℕ := {2, 5, 7}
def Y : Finset ℕ := {2, 4, 6}
def Z : Finset ℕ := {1, 2, 3, 4}

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define the probability of getting an even sum
def probEvenSum : ℚ := sorry

-- Theorem statement
theorem even_sum_probability :
  probEvenSum = 1/2 := by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l3830_383047


namespace NUMINAMATH_CALUDE_circle_segment_perimeter_l3830_383058

/-- Given a circle with radius 7 and a central angle of 270°, 
    the perimeter of the segment formed by this angle is equal to 14 + 10.5π. -/
theorem circle_segment_perimeter (r : ℝ) (angle : ℝ) : 
  r = 7 → angle = 270 * π / 180 → 
  2 * r + (angle / (2 * π)) * (2 * π * r) = 14 + 10.5 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_perimeter_l3830_383058


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3830_383034

theorem arithmetic_evaluation : 5 + 2 * (8 - 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3830_383034


namespace NUMINAMATH_CALUDE_min_staircase_steps_l3830_383074

theorem min_staircase_steps (a b : ℕ+) :
  ∃ (n : ℕ), n = a + b - Nat.gcd a b ∧
  (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), k * a = m ∨ k * a = m + b) ∧
  (∃ (k l : ℕ), k * a = n ∧ l * a = n + b) :=
sorry

end NUMINAMATH_CALUDE_min_staircase_steps_l3830_383074
