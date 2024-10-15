import Mathlib

namespace NUMINAMATH_CALUDE_clown_balloons_l1820_182001

/-- The number of balloons a clown had initially, given the number of boys and girls who bought balloons, and the number of balloons remaining. -/
def initial_balloons (boys girls remaining : ℕ) : ℕ :=
  boys + girls + remaining

/-- Theorem stating that the clown initially had 36 balloons -/
theorem clown_balloons : initial_balloons 3 12 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l1820_182001


namespace NUMINAMATH_CALUDE_intersection_complement_equals_interval_l1820_182039

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 1}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (U \ B) = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_interval_l1820_182039


namespace NUMINAMATH_CALUDE_rectangle_quadrilateral_area_l1820_182086

/-- Given a rectangle with sides 5 cm and 48 cm, where the longer side is divided into three equal parts
    and the midpoint of the shorter side is connected to the first division point on the longer side,
    the area of the resulting smaller quadrilateral is 90 cm². -/
theorem rectangle_quadrilateral_area :
  let short_side : ℝ := 5
  let long_side : ℝ := 48
  let division_point : ℝ := long_side / 3
  let midpoint : ℝ := short_side / 2
  let total_area : ℝ := short_side * long_side
  let part_area : ℝ := short_side * division_point
  let quadrilateral_area : ℝ := part_area + (part_area / 2)
  quadrilateral_area = 90
  := by sorry

end NUMINAMATH_CALUDE_rectangle_quadrilateral_area_l1820_182086


namespace NUMINAMATH_CALUDE_g_of_60_l1820_182095

/-- Given a function g satisfying the specified properties, prove that g(60) = 11.25 -/
theorem g_of_60 (g : ℝ → ℝ) 
    (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y)
    (h2 : g 45 = 15) : 
  g 60 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_g_of_60_l1820_182095


namespace NUMINAMATH_CALUDE_division_theorem_l1820_182009

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 131 → divisor = 14 → remainder = 5 → 
  dividend = divisor * quotient + remainder → quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l1820_182009


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1820_182090

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1820_182090


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l1820_182027

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l1820_182027


namespace NUMINAMATH_CALUDE_rectangle_configuration_CG_length_l1820_182075

/-- A configuration of two rectangles ABCD and EFGH with parallel sides -/
structure RectangleConfiguration where
  /-- The length of segment AE -/
  AE : ℝ
  /-- The length of segment BF -/
  BF : ℝ
  /-- The length of segment DH -/
  DH : ℝ
  /-- The length of segment CG -/
  CG : ℝ

/-- The theorem stating the length of CG given the other lengths -/
theorem rectangle_configuration_CG_length 
  (config : RectangleConfiguration) 
  (h1 : config.AE = 10) 
  (h2 : config.BF = 20) 
  (h3 : config.DH = 30) : 
  config.CG = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_configuration_CG_length_l1820_182075


namespace NUMINAMATH_CALUDE_jerome_money_left_l1820_182023

def jerome_problem (initial_half : ℕ) (meg_amount : ℕ) : Prop :=
  let initial_total := 2 * initial_half
  let after_meg := initial_total - meg_amount
  let bianca_amount := 3 * meg_amount
  let final_amount := after_meg - bianca_amount
  final_amount = 54

theorem jerome_money_left : jerome_problem 43 8 := by
  sorry

end NUMINAMATH_CALUDE_jerome_money_left_l1820_182023


namespace NUMINAMATH_CALUDE_solve_equation_l1820_182006

theorem solve_equation (k l q : ℚ) : 
  (3/4 : ℚ) = k/108 ∧ 
  (3/4 : ℚ) = (l+k)/126 ∧ 
  (3/4 : ℚ) = (q-l)/180 → 
  q = 148.5 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1820_182006


namespace NUMINAMATH_CALUDE_root_values_l1820_182000

theorem root_values (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_root_values_l1820_182000


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1820_182085

theorem simple_interest_rate : 
  ∀ (P : ℝ) (R : ℝ),
  P > 0 →
  (P * R * 10) / 100 = (3 / 5) * P →
  R = 6 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1820_182085


namespace NUMINAMATH_CALUDE_min_abs_w_l1820_182074

theorem min_abs_w (w : ℂ) (h : Complex.abs (w + 2 - 2*I) + Complex.abs (w - 5*I) = 7) :
  Complex.abs w ≥ 10/7 ∧ ∃ w₀ : ℂ, Complex.abs (w₀ + 2 - 2*I) + Complex.abs (w₀ - 5*I) = 7 ∧ Complex.abs w₀ = 10/7 :=
sorry

end NUMINAMATH_CALUDE_min_abs_w_l1820_182074


namespace NUMINAMATH_CALUDE_midpoint_endpoint_product_l1820_182050

/-- Given a segment CD with midpoint M and endpoint C, proves that the product of D's coordinates is -63 -/
theorem midpoint_endpoint_product (M C D : ℝ × ℝ) : 
  M = (4, -2) → C = (-1, 3) → M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → D.1 * D.2 = -63 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_endpoint_product_l1820_182050


namespace NUMINAMATH_CALUDE_direct_proportion_equation_l1820_182069

/-- A direct proportion function passing through (-1, 2) -/
def direct_proportion_through_neg1_2 (k : ℝ) (x : ℝ) : Prop :=
  k ≠ 0 ∧ 2 = k * (-1)

/-- The equation of the direct proportion function -/
def equation_of_direct_proportion (x : ℝ) : ℝ := -2 * x

theorem direct_proportion_equation :
  ∀ k : ℝ, direct_proportion_through_neg1_2 k x →
  ∀ x : ℝ, k * x = equation_of_direct_proportion x :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_equation_l1820_182069


namespace NUMINAMATH_CALUDE_subset_implies_all_elements_in_l1820_182030

theorem subset_implies_all_elements_in : 
  ∀ (A B : Set α), A.Nonempty → B.Nonempty → A ⊆ B → ∀ x ∈ A, x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_all_elements_in_l1820_182030


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1820_182099

/-- Two lines are parallel if and only if their slopes are equal -/
theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, (a + 2) * x + (a + 3) * y - 5 = 0 ∧ 
               6 * x + (2 * a - 1) * y - 5 = 0) →
  (a + 2) / 6 = (a + 3) / (2 * a - 1) →
  a = -5/2 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_slope_l1820_182099


namespace NUMINAMATH_CALUDE_odd_function_a_range_l1820_182044

/-- An odd function f: ℝ → ℝ satisfying given conditions -/
def OddFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, f x = 9*x + a^2/x + 7) ∧
  (∀ x ≥ 0, f x ≥ 0)

/-- Theorem stating the range of values for a -/
theorem odd_function_a_range (f : ℝ → ℝ) (a : ℝ) 
  (h : OddFunction f a) : 
  a ≥ 7/6 ∨ a ≤ -7/6 :=
sorry

end NUMINAMATH_CALUDE_odd_function_a_range_l1820_182044


namespace NUMINAMATH_CALUDE_min_value_theorem_l1820_182008

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 1/y = 2 → 2*x*y + 1/x ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1820_182008


namespace NUMINAMATH_CALUDE_sum_of_ages_l1820_182021

/-- The sum of Eunji and Yuna's ages given their age difference -/
theorem sum_of_ages (eunji_age : ℕ) (age_difference : ℕ) : 
  eunji_age = 7 → age_difference = 5 → eunji_age + (eunji_age + age_difference) = 19 := by
  sorry

#check sum_of_ages

end NUMINAMATH_CALUDE_sum_of_ages_l1820_182021


namespace NUMINAMATH_CALUDE_infinite_valid_points_l1820_182045

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 25}

-- Define the center of the circle
def Center : ℝ × ℝ := (0, 0)

-- Define the diameter endpoints
def DiameterEndpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((-5, 0), (5, 0))

-- Define the condition for points P
def ValidPoint (p : ℝ × ℝ) : Prop :=
  let (a, b) := DiameterEndpoints
  ((p.1 - a.1)^2 + (p.2 - a.2)^2) + ((p.1 - b.1)^2 + (p.2 - b.2)^2) = 50 ∧
  (p.1^2 + p.2^2 < 25)

-- Theorem statement
theorem infinite_valid_points : ∃ (S : Set (ℝ × ℝ)), Set.Infinite S ∧ ∀ p ∈ S, p ∈ Circle ∧ ValidPoint p :=
sorry

end NUMINAMATH_CALUDE_infinite_valid_points_l1820_182045


namespace NUMINAMATH_CALUDE_smallest_b_value_l1820_182017

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1 / a + 1 / b ≤ 2) →
  b ≥ (3 + Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1820_182017


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1820_182061

theorem sum_of_squares_problem (a b c d k p : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 390)
  (h2 : a*b + b*c + c*a + a*d + b*d + c*d = 5)
  (h3 : a*d + b*d + c*d = k)
  (h4 : a^2*b^2*c^2*d^2 = p) :
  a + b + c + d = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1820_182061


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l1820_182033

open Real

theorem triangle_abc_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b ∧ sin B / b = sin C / c →
  cos (2 * C) - cos (2 * A) = 2 * sin (π / 3 + C) * sin (π / 3 - C) →
  a = sqrt 3 →
  b ≥ a →
  A = π / 3 ∧ sqrt 3 ≤ 2 * b - c ∧ 2 * b - c < 2 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l1820_182033


namespace NUMINAMATH_CALUDE_product_of_roots_is_4y_squared_l1820_182031

-- Define a quadratic function f
variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Assumptions
axiom f_is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
axiom root_of_f_x_minus_y : f (2*y - y) = 0
axiom root_of_f_x_plus_y : f (3*y + y) = 0

-- Theorem statement
theorem product_of_roots_is_4y_squared :
  (∃ (r₁ r₂ : ℝ), ∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂)) →
  (∃ (r₁ r₂ : ℝ), (∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂)) ∧ r₁ * r₂ = 4 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_is_4y_squared_l1820_182031


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l1820_182011

/-- The perimeter of the shape ABFCDE formed by cutting a right triangle from a square and
    repositioning it on the left side of the square. -/
theorem perimeter_of_modified_square (
  square_perimeter : ℝ)
  (triangle_leg : ℝ)
  (h1 : square_perimeter = 48)
  (h2 : triangle_leg = 12) : ℝ :=
by
  -- The perimeter of the new shape ABFCDE is 60 inches
  sorry

#check perimeter_of_modified_square

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l1820_182011


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1820_182028

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1820_182028


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1820_182002

theorem at_least_one_geq_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1820_182002


namespace NUMINAMATH_CALUDE_andrew_payment_l1820_182052

/-- The total amount Andrew paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_price grape_weight mango_price mango_weight : ℕ) : ℕ :=
  grape_price * grape_weight + mango_price * mango_weight

/-- Theorem stating that Andrew paid 1428 to the shopkeeper -/
theorem andrew_payment :
  total_amount 98 11 50 7 = 1428 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l1820_182052


namespace NUMINAMATH_CALUDE_centroid_division_area_difference_l1820_182068

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Represents the centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Represents a line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Calculates the areas of two parts of a triangle divided by a line -/
def dividedAreas (t : Triangle) (l : Line) : ℝ × ℝ := sorry

/-- Theorem: The difference in areas of two parts of a triangle divided by a line
    through its centroid is not greater than 1/9 of the triangle's total area -/
theorem centroid_division_area_difference (t : Triangle) (l : Line) :
  let (A1, A2) := dividedAreas t l
  let G := centroid t
  l.point = G →
  |A1 - A2| ≤ (1/9) * triangleArea t :=
sorry

end NUMINAMATH_CALUDE_centroid_division_area_difference_l1820_182068


namespace NUMINAMATH_CALUDE_bookstore_discount_proof_l1820_182081

/-- Calculates the final price as a percentage of the original price
    given an initial discount and an additional discount on the already discounted price. -/
def final_price_percentage (initial_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  (1 - initial_discount) * (1 - additional_discount) * 100

/-- Proves that with a 40% initial discount and an additional 20% discount,
    the final price is 48% of the original price. -/
theorem bookstore_discount_proof :
  final_price_percentage 0.4 0.2 = 48 := by
sorry

end NUMINAMATH_CALUDE_bookstore_discount_proof_l1820_182081


namespace NUMINAMATH_CALUDE_angle_identities_l1820_182065

/-- Given that α is an angle in the second quadrant and cos(α + π) = 3/13,
    prove that tan α = -4√10/3 and sin(α - π/2) * sin(-α - π) = -12√10/169 -/
theorem angle_identities (α : Real) 
    (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
    (h2 : Real.cos (α + π) = 3/13) :
    Real.tan α = -4 * Real.sqrt 10 / 3 ∧ 
    Real.sin (α - π/2) * Real.sin (-α - π) = -12 * Real.sqrt 10 / 169 := by
  sorry

end NUMINAMATH_CALUDE_angle_identities_l1820_182065


namespace NUMINAMATH_CALUDE_two_cookies_eaten_l1820_182096

/-- Given an initial number of cookies and the number of cookies left,
    calculate the number of cookies eaten. -/
def cookies_eaten (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Theorem: Given 7 initial cookies and 5 cookies left,
    prove that 2 cookies were eaten. -/
theorem two_cookies_eaten :
  cookies_eaten 7 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cookies_eaten_l1820_182096


namespace NUMINAMATH_CALUDE_b_value_function_comparison_l1820_182010

/-- A quadratic function with the given properties -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry property of the function -/
axiom symmetry_property (b c : ℝ) : ∀ x : ℝ, f b c (2 + x) = f b c (2 - x)

/-- The value of b in the quadratic function -/
theorem b_value : ∃ b : ℝ, (∀ c x : ℝ, f b c (2 + x) = f b c (2 - x)) ∧ b = 4 :=
sorry

/-- Comparison of function values -/
theorem function_comparison (b c : ℝ) 
  (h : ∀ x : ℝ, f b c (2 + x) = f b c (2 - x)) : 
  ∀ a : ℝ, f b c (5/4) < f b c (-a^2 - a + 1) :=
sorry

end NUMINAMATH_CALUDE_b_value_function_comparison_l1820_182010


namespace NUMINAMATH_CALUDE_system_solution_l1820_182094

/-- A solution to the system of equations is a triple (x, y, z) that satisfies all three equations. -/
def IsSolution (x y z : ℝ) : Prop :=
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34

/-- The theorem states that the only solutions to the system of equations are (2, 3, 1) and (3, 2, 1). -/
theorem system_solution :
  ∀ x y z : ℝ, IsSolution x y z ↔ ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1820_182094


namespace NUMINAMATH_CALUDE_cylinder_volume_height_relation_l1820_182003

theorem cylinder_volume_height_relation (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let v := π * r^2 * h
  let r' := 2 * r
  let v' := 2 * v
  ∃ h', v' = π * r'^2 * h' ∧ h' = h / 4 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_height_relation_l1820_182003


namespace NUMINAMATH_CALUDE_fib_F15_units_digit_l1820_182019

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem fib_F15_units_digit :
  unitsDigit (fib (fib 15)) = 5 := by sorry

end NUMINAMATH_CALUDE_fib_F15_units_digit_l1820_182019


namespace NUMINAMATH_CALUDE_ellipse_tangent_max_area_l1820_182056

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the xy-plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a line is tangent to an ellipse -/
def isTangent (l : Line) (e : Ellipse) : Prop :=
  sorry -- Definition of tangent line to ellipse

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry -- Formula for triangle area

/-- The main theorem -/
theorem ellipse_tangent_max_area
  (e : Ellipse)
  (A B C : Point)
  (h1 : e.a^2 = 1 ∧ e.b^2 = 3)
  (h2 : A.x = 1 ∧ A.y = 1)
  (h3 : pointOnEllipse A e)
  (h4 : pointOnEllipse B e)
  (h5 : pointOnEllipse C e)
  (h6 : ∃ (l : Line), isTangent l e ∧ l.a * B.x + l.b * B.y + l.c = 0 ∧ l.a * C.x + l.b * C.y + l.c = 0)
  (h7 : ∀ (B' C' : Point), pointOnEllipse B' e → pointOnEllipse C' e →
        triangleArea A B' C' ≤ triangleArea A B C) :
  ∃ (l : Line), l.a = 1 ∧ l.b = 3 ∧ l.c = 2 ∧
    isTangent l e ∧
    l.a * B.x + l.b * B.y + l.c = 0 ∧
    l.a * C.x + l.b * C.y + l.c = 0 ∧
    triangleArea A B C = 3 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_max_area_l1820_182056


namespace NUMINAMATH_CALUDE_rehabilitation_centers_count_rehabilitation_centers_count_proof_l1820_182055

theorem rehabilitation_centers_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun lisa jude han jane =>
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 →
    lisa + jude + han + jane = 27

#check rehabilitation_centers_count

-- The proof is omitted
theorem rehabilitation_centers_count_proof :
  ∃ (lisa jude han jane : ℕ),
    rehabilitation_centers_count lisa jude han jane :=
sorry

end NUMINAMATH_CALUDE_rehabilitation_centers_count_rehabilitation_centers_count_proof_l1820_182055


namespace NUMINAMATH_CALUDE_circle_properties_l1820_182041

/-- The parabola to which the circle is tangent -/
def parabola (x y : ℝ) : Prop := y^2 = 5*x + 9

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := 2*x^2 - 10*x*y - 31*y^2 + 175*x - 6*y + 297 = 0

/-- Points through which the circle passes -/
def point_P : ℝ × ℝ := (0, 3)
def point_Q : ℝ × ℝ := (-1, -2)
def point_A : ℝ × ℝ := (-2, 1)

theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, (x - 0)^2 + (y - 0)^2 = r^2) ∧
  (parabola (point_P.1) (point_P.2)) ∧
  (parabola (point_Q.1) (point_Q.2)) ∧
  (circle_equation (point_P.1) (point_P.2)) ∧
  (circle_equation (point_Q.1) (point_Q.2)) ∧
  (circle_equation (point_A.1) (point_A.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1820_182041


namespace NUMINAMATH_CALUDE_base_x_is_8_l1820_182073

/-- The base of the numeral system in which 1728 (decimal) is represented as 3362 -/
def base_x : ℕ :=
  sorry

/-- The representation of 1728 in base x -/
def representation : List ℕ :=
  [3, 3, 6, 2]

theorem base_x_is_8 :
  (base_x ^ 3 * representation[0]! +
   base_x ^ 2 * representation[1]! +
   base_x ^ 1 * representation[2]! +
   base_x ^ 0 * representation[3]!) = 1728 ∧
  base_x = 8 :=
sorry

end NUMINAMATH_CALUDE_base_x_is_8_l1820_182073


namespace NUMINAMATH_CALUDE_lee_ribbons_left_l1820_182024

/-- The number of ribbons Mr. Lee had left after giving away ribbons in the morning and afternoon -/
def ribbons_left (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : ℕ :=
  initial - (morning + afternoon)

/-- Theorem stating that Mr. Lee had 8 ribbons left -/
theorem lee_ribbons_left : ribbons_left 38 14 16 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lee_ribbons_left_l1820_182024


namespace NUMINAMATH_CALUDE_bank_queue_properties_l1820_182014

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧
  expected_wasted_time q = 84 :=
by sorry

end NUMINAMATH_CALUDE_bank_queue_properties_l1820_182014


namespace NUMINAMATH_CALUDE_l_shaped_area_is_ten_l1820_182084

/-- The area of an L-shaped region formed by subtracting four smaller squares from a larger square -/
def l_shaped_area (large_side : ℝ) (small_side1 small_side2 : ℝ) : ℝ :=
  large_side^2 - 2 * small_side1^2 - 2 * small_side2^2

/-- Theorem stating that the area of the L-shaped region is 10 -/
theorem l_shaped_area_is_ten :
  l_shaped_area 6 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_is_ten_l1820_182084


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1820_182060

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Calculates the stratified sample sizes for each age group -/
def stratifiedSample (total : ℕ) (ratio : EmployeeCount) (sampleSize : ℕ) : EmployeeCount :=
  let totalRatio := ratio.middleAged + ratio.young + ratio.elderly
  { middleAged := (sampleSize * ratio.middleAged) / totalRatio,
    young := (sampleSize * ratio.young) / totalRatio,
    elderly := (sampleSize * ratio.elderly) / totalRatio }

theorem correct_stratified_sample :
  let total := 3200
  let ratio := EmployeeCount.mk 5 3 2
  let sampleSize := 400
  let result := stratifiedSample total ratio sampleSize
  result.middleAged = 200 ∧ result.young = 120 ∧ result.elderly = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1820_182060


namespace NUMINAMATH_CALUDE_honey_servings_l1820_182026

/-- Calculates the number of full servings of honey in a container -/
def fullServings (containerAmount : Rat) (servingSize : Rat) : Rat :=
  containerAmount / servingSize

/-- Proves that a container with 47 2/3 tablespoons of honey provides 14 1/5 full servings when each serving is 3 1/3 tablespoons -/
theorem honey_servings :
  let containerAmount : Rat := 47 + 2/3
  let servingSize : Rat := 3 + 1/3
  fullServings containerAmount servingSize = 14 + 1/5 := by
sorry

#eval fullServings (47 + 2/3) (3 + 1/3)

end NUMINAMATH_CALUDE_honey_servings_l1820_182026


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1820_182040

/-- The number of ways to arrange books on a shelf -/
def arrange_books (arabic : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  Nat.factorial (arabic + german + spanish - 2) * Nat.factorial arabic * Nat.factorial spanish

/-- Theorem stating the number of arrangements for the given book configuration -/
theorem book_arrangement_count :
  arrange_books 2 3 4 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1820_182040


namespace NUMINAMATH_CALUDE_bananas_per_box_l1820_182012

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 10) :
  total_bananas / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l1820_182012


namespace NUMINAMATH_CALUDE_product_plus_one_square_l1820_182070

theorem product_plus_one_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_square_l1820_182070


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1820_182035

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1820_182035


namespace NUMINAMATH_CALUDE_min_purchase_price_l1820_182049

/-- Represents the coin denominations available on the Moon -/
def moon_coins : List Nat := [1, 15, 50]

/-- Theorem stating the minimum possible price of a purchase on the Moon -/
theorem min_purchase_price :
  ∀ (payment : List Nat) (change : List Nat),
    (∀ c ∈ payment, c ∈ moon_coins) →
    (∀ c ∈ change, c ∈ moon_coins) →
    (change.length = payment.length + 1) →
    (payment.sum - change.sum ≥ 6) →
    ∃ (p : List Nat) (c : List Nat),
      (∀ x ∈ p, x ∈ moon_coins) ∧
      (∀ x ∈ c, x ∈ moon_coins) ∧
      (c.length = p.length + 1) ∧
      (p.sum - c.sum = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_purchase_price_l1820_182049


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1820_182071

theorem min_value_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let A := (a^2 + b^2)^4 / (c*d)^4 + (b^2 + c^2)^4 / (a*d)^4 + (c^2 + d^2)^4 / (a*b)^4 + (d^2 + a^2)^4 / (b*c)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1820_182071


namespace NUMINAMATH_CALUDE_family_reunion_food_l1820_182004

/-- The total amount of food Peter buys for the family reunion -/
def total_food (chicken : ℝ) (hamburger_ratio : ℝ) (hotdog_difference : ℝ) (sides_ratio : ℝ) : ℝ :=
  let hamburger := chicken * hamburger_ratio
  let hotdog := hamburger + hotdog_difference
  let sides := hotdog * sides_ratio
  chicken + hamburger + hotdog + sides

/-- Theorem stating the total amount of food Peter will buy -/
theorem family_reunion_food :
  total_food 16 (1/2) 2 (1/2) = 39 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_food_l1820_182004


namespace NUMINAMATH_CALUDE_gift_wrap_sale_total_l1820_182057

/-- Calculates the total amount of money collected from selling gift wrap rolls -/
def total_amount_collected (total_rolls : ℕ) (print_rolls : ℕ) (solid_price : ℚ) (print_price : ℚ) : ℚ :=
  (print_rolls * print_price) + ((total_rolls - print_rolls) * solid_price)

/-- Proves that the total amount collected from selling gift wrap rolls is $2340 -/
theorem gift_wrap_sale_total : 
  total_amount_collected 480 210 4 6 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrap_sale_total_l1820_182057


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1820_182058

/-- Given a geometric sequence {a_n} where 2a_3, a_5/2, 3a_1 forms an arithmetic sequence,
    prove that (a_2 + a_5) / (a_9 + a_6) = 1/9 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) (hq : q > 0) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (2 * a 3 - a 5 / 2 = a 5 / 2 - 3 * a 1) →  -- arithmetic sequence condition
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1820_182058


namespace NUMINAMATH_CALUDE_heartsuit_properties_l1820_182037

def heartsuit (x y : ℝ) : ℝ := |x - y| + 1

theorem heartsuit_properties :
  (∀ x y : ℝ, heartsuit x y = heartsuit y x) ∧
  (∃ x y : ℝ, 2 * (heartsuit x y) ≠ heartsuit (2 * x) (2 * y)) ∧
  (∃ x : ℝ, heartsuit x 0 ≠ x + 1) ∧
  (∀ x : ℝ, heartsuit x x = 1) ∧
  (∀ x y : ℝ, x ≠ y → heartsuit x y > 1) :=
by sorry

end NUMINAMATH_CALUDE_heartsuit_properties_l1820_182037


namespace NUMINAMATH_CALUDE_expression_evaluation_l1820_182048

theorem expression_evaluation :
  (2^(2+1) - 2*(2-1)^(2+1))^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1820_182048


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l1820_182088

theorem coordinate_sum_of_point_B (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3/4 →
  B.1 + B.2 = 35/3 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l1820_182088


namespace NUMINAMATH_CALUDE_slide_count_l1820_182059

theorem slide_count (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 22 → additional = 13 → total = initial + additional → total = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_count_l1820_182059


namespace NUMINAMATH_CALUDE_midpoint_triangle_half_area_l1820_182097

/-- A rectangle with midpoints on longer sides -/
structure RectangleWithMidpoints where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  p : ℝ × ℝ
  q : ℝ × ℝ
  p_midpoint : p = (0, length / 2)
  q_midpoint : q = (length, length / 2)

/-- The area of the triangle formed by midpoints and corner is half the rectangle area -/
theorem midpoint_triangle_half_area (r : RectangleWithMidpoints) :
    let triangle_area := (r.length * r.length / 2) / 2
    let rectangle_area := r.length * r.width
    triangle_area = rectangle_area / 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_half_area_l1820_182097


namespace NUMINAMATH_CALUDE_baker_initial_cakes_l1820_182067

/-- 
Given that Baker made some initial cakes, then made 149 more, 
sold 144, and still has 67 cakes, prove that he initially made 62 cakes.
-/
theorem baker_initial_cakes : 
  ∀ (initial : ℕ), 
  (initial + 149 : ℕ) - 144 = 67 → 
  initial = 62 := by
sorry

end NUMINAMATH_CALUDE_baker_initial_cakes_l1820_182067


namespace NUMINAMATH_CALUDE_smallest_fifth_prime_term_l1820_182015

/-- An arithmetic sequence of five prime numbers -/
structure PrimeArithmeticSequence :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference
  (h1 : 0 < d)  -- Ensure the sequence is increasing
  (h2 : ∀ i : Fin 5, Prime (a + i.val * d))  -- All 5 terms are prime

/-- The fifth term of a prime arithmetic sequence -/
def fifthTerm (seq : PrimeArithmeticSequence) : ℕ :=
  seq.a + 4 * seq.d

theorem smallest_fifth_prime_term :
  (∃ seq : PrimeArithmeticSequence, fifthTerm seq = 29) ∧
  (∀ seq : PrimeArithmeticSequence, 29 ≤ fifthTerm seq) :=
sorry

end NUMINAMATH_CALUDE_smallest_fifth_prime_term_l1820_182015


namespace NUMINAMATH_CALUDE_factor_expression_l1820_182066

theorem factor_expression (c : ℝ) : 210 * c^3 + 35 * c^2 = 35 * c^2 * (6 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1820_182066


namespace NUMINAMATH_CALUDE_intersection_quadrilateral_perimeter_bounds_l1820_182029

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  a_pos : 0 < a

/-- A quadrilateral formed by the intersection of a plane and a regular tetrahedron -/
structure IntersectionQuadrilateral (t : RegularTetrahedron) where
  perimeter : ℝ

/-- The theorem stating that the perimeter of the intersection quadrilateral
    is bounded between 2a and 3a -/
theorem intersection_quadrilateral_perimeter_bounds
  (t : RegularTetrahedron) (q : IntersectionQuadrilateral t) :
  2 * t.a ≤ q.perimeter ∧ q.perimeter ≤ 3 * t.a :=
sorry

end NUMINAMATH_CALUDE_intersection_quadrilateral_perimeter_bounds_l1820_182029


namespace NUMINAMATH_CALUDE_oliver_age_l1820_182087

/-- Given the ages of Oliver, Mia, and Lucas, prove that Oliver is 18 years old. -/
theorem oliver_age :
  ∀ (oliver_age mia_age lucas_age : ℕ),
    oliver_age = mia_age - 2 →
    mia_age = lucas_age + 5 →
    lucas_age = 15 →
    oliver_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_oliver_age_l1820_182087


namespace NUMINAMATH_CALUDE_train_length_l1820_182005

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 6 → speed * time * (1000 / 3600) = 150 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1820_182005


namespace NUMINAMATH_CALUDE_longest_chord_length_l1820_182053

theorem longest_chord_length (r : ℝ) (h : r = 11) : 
  2 * r = 22 := by sorry

end NUMINAMATH_CALUDE_longest_chord_length_l1820_182053


namespace NUMINAMATH_CALUDE_area_enclosed_by_trajectory_l1820_182079

def f (x : ℝ) : ℝ := x^2 + 1

theorem area_enclosed_by_trajectory (a b : ℝ) (h1 : a < b) 
  (h2 : Set.range f = Set.Icc 1 5) 
  (h3 : Set.Icc a b = f⁻¹' (Set.Icc 1 5)) : 
  (b - a) * 1 = 4 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_trajectory_l1820_182079


namespace NUMINAMATH_CALUDE_cory_fruit_arrangements_l1820_182034

/-- The number of ways to arrange indistinguishable objects of different types -/
def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial ks))

/-- The number of distinct arrangements of Cory's fruit -/
theorem cory_fruit_arrangements :
  let total_fruit : ℕ := 7
  let fruit_counts : List ℕ := [3, 2, 2]
  multinomial_coefficient total_fruit fruit_counts = 210 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_arrangements_l1820_182034


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l1820_182063

/-- The nth term of a geometric sequence -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 10th term of a geometric sequence with first term 4 and common ratio 5/3 -/
theorem tenth_term_of_specific_sequence :
  geometric_sequence 4 (5/3) 10 = 7812500/19683 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l1820_182063


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1820_182043

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 8 * a 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1820_182043


namespace NUMINAMATH_CALUDE_calendar_reuse_2080_l1820_182038

/-- A year is reusable after a fixed number of years if both years have the same leap year status and start on the same day of the week. -/
def is_calendar_reusable (initial_year target_year : ℕ) : Prop :=
  (initial_year % 4 = 0 ↔ target_year % 4 = 0) ∧
  (initial_year + 1) % 7 = (target_year + 1) % 7

/-- The theorem states that the calendar of the year 2080 can be reused after 28 years. -/
theorem calendar_reuse_2080 :
  let initial_year := 2080
  let year_difference := 28
  is_calendar_reusable initial_year (initial_year + year_difference) :=
by sorry

end NUMINAMATH_CALUDE_calendar_reuse_2080_l1820_182038


namespace NUMINAMATH_CALUDE_square_root_expression_evaluation_l1820_182054

theorem square_root_expression_evaluation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000001 ∧ 
  |22 + Real.sqrt (-4 + 6 * 4 * 3) - 30.246211251| < ε := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_evaluation_l1820_182054


namespace NUMINAMATH_CALUDE_min_domain_length_l1820_182036

open Real

theorem min_domain_length (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x = sin x * sin (x + π/3) - 1/4) →
  m < n →
  Set.range f = Set.Icc (-1/2) (1/4) →
  n - m ≥ 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_min_domain_length_l1820_182036


namespace NUMINAMATH_CALUDE_man_age_difference_l1820_182082

/-- Proves that a man is 37 years older than his son given the conditions. -/
theorem man_age_difference (son_age man_age : ℕ) : 
  son_age = 35 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 37 := by
  sorry

end NUMINAMATH_CALUDE_man_age_difference_l1820_182082


namespace NUMINAMATH_CALUDE_digit_appearance_l1820_182091

def digit_free (n : ℕ) (d : Finset ℕ) : Prop :=
  ∀ (i : ℕ), i ∈ d → (n / 10^i % 10 ≠ i)

def contains_digit (n : ℕ) (d : Finset ℕ) : Prop :=
  ∃ (i : ℕ), i ∈ d ∧ (n / 10^i % 10 = i)

theorem digit_appearance (n : ℕ) (h1 : n ≥ 1) (h2 : digit_free n {1, 2, 9}) :
  contains_digit (3 * n) {1, 2, 9} := by
  sorry

end NUMINAMATH_CALUDE_digit_appearance_l1820_182091


namespace NUMINAMATH_CALUDE_total_fruits_l1820_182007

def persimmons : ℕ := 2
def apples : ℕ := 7

theorem total_fruits : persimmons + apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l1820_182007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l1820_182022

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_2017 :
  arithmetic_sequence 4 3 672 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l1820_182022


namespace NUMINAMATH_CALUDE_merchant_markup_theorem_l1820_182020

/-- Proves the required markup percentage for a merchant to achieve a specific profit --/
theorem merchant_markup_theorem (list_price : ℝ) (h_list_price_pos : 0 < list_price) :
  let cost_price := 0.7 * list_price
  let selling_price := list_price
  let marked_price := (5/4) * list_price
  (cost_price = 0.7 * selling_price) →
  (selling_price = 0.8 * marked_price) →
  (marked_price = 1.25 * list_price) :=
by
  sorry

#check merchant_markup_theorem

end NUMINAMATH_CALUDE_merchant_markup_theorem_l1820_182020


namespace NUMINAMATH_CALUDE_bridge_painting_l1820_182078

theorem bridge_painting (painted_section : ℚ) : 
  (1.2 * painted_section = 1/2) → 
  (1/2 - painted_section = 1/12) := by
sorry

end NUMINAMATH_CALUDE_bridge_painting_l1820_182078


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l1820_182092

theorem quadratic_real_solutions_range (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + a = 0) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l1820_182092


namespace NUMINAMATH_CALUDE_money_ratio_problem_l1820_182062

theorem money_ratio_problem (a b : ℕ) (ha : a = 800) (hb : b = 500) : 
  (a : ℚ) / b = 8 / 5 ∧ 
  ((a - 50 : ℚ) / (b + 100) = 5 / 4) := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l1820_182062


namespace NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l1820_182051

/-- Given two positive integers with a ratio of 3:4 and HCF of 4, their LCM is 48 -/
theorem ratio_hcf_to_lcm (a b : ℕ+) (h_ratio : a.val * 4 = b.val * 3) (h_hcf : Nat.gcd a.val b.val = 4) :
  Nat.lcm a.val b.val = 48 := by
  sorry

end NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l1820_182051


namespace NUMINAMATH_CALUDE_book_pages_calculation_l1820_182080

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := 10

/-- The number of pages Sally reads on weekends -/
def weekend_pages : ℕ := 20

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

theorem book_pages_calculation :
  total_pages = 
    weeks_to_finish * (weekdays_per_week * weekday_pages + weekend_days_per_week * weekend_pages) :=
by sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l1820_182080


namespace NUMINAMATH_CALUDE_sin_70_degrees_l1820_182098

theorem sin_70_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_degrees_l1820_182098


namespace NUMINAMATH_CALUDE_exists_valid_configuration_two_thirds_not_exists_valid_configuration_three_fourths_not_exists_valid_configuration_seven_tenths_l1820_182077

/-- Represents the fraction of difficult problems and well-performing students -/
structure TestConfiguration (α : ℚ) where
  difficultProblems : ℚ
  wellPerformingStudents : ℚ
  difficultProblems_ge : difficultProblems ≥ α
  wellPerformingStudents_ge : wellPerformingStudents ≥ α

/-- Theorem stating the existence of a valid configuration for α = 2/3 -/
theorem exists_valid_configuration_two_thirds :
  ∃ (config : TestConfiguration (2/3)), True :=
sorry

/-- Theorem stating the non-existence of a valid configuration for α = 3/4 -/
theorem not_exists_valid_configuration_three_fourths :
  ¬ ∃ (config : TestConfiguration (3/4)), True :=
sorry

/-- Theorem stating the non-existence of a valid configuration for α = 7/10 -/
theorem not_exists_valid_configuration_seven_tenths :
  ¬ ∃ (config : TestConfiguration (7/10)), True :=
sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_two_thirds_not_exists_valid_configuration_three_fourths_not_exists_valid_configuration_seven_tenths_l1820_182077


namespace NUMINAMATH_CALUDE_twenty_seven_power_minus_log_eight_two_equals_zero_l1820_182093

theorem twenty_seven_power_minus_log_eight_two_equals_zero :
  Real.rpow 27 (-1/3) - Real.log 2 / Real.log 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_seven_power_minus_log_eight_two_equals_zero_l1820_182093


namespace NUMINAMATH_CALUDE_min_score_game_12_is_42_l1820_182025

/-- Represents the scores of a football player over a season -/
structure FootballScores where
  first_seven : ℕ  -- Total score for first 7 games
  game_8 : ℕ := 18
  game_9 : ℕ := 25
  game_10 : ℕ := 10
  game_11 : ℕ := 22
  game_12 : ℕ

/-- The minimum score for game 12 that satisfies all conditions -/
def min_score_game_12 (scores : FootballScores) : Prop :=
  let total_8_to_11 := scores.game_8 + scores.game_9 + scores.game_10 + scores.game_11
  let avg_8_to_11 : ℚ := total_8_to_11 / 4
  let total_12_games := scores.first_seven + total_8_to_11 + scores.game_12
  (scores.first_seven / 7 : ℚ) < (total_12_games - scores.game_12) / 11 ∧ 
  (total_12_games : ℚ) / 12 > 20 ∧
  scores.game_12 = 42 ∧
  ∀ x : ℕ, x < 42 → 
    let total_with_x := scores.first_seven + total_8_to_11 + x
    (total_with_x : ℚ) / 12 ≤ 20 ∨ (scores.first_seven / 7 : ℚ) ≥ (total_with_x - x) / 11

theorem min_score_game_12_is_42 (scores : FootballScores) :
  min_score_game_12 scores := by sorry

end NUMINAMATH_CALUDE_min_score_game_12_is_42_l1820_182025


namespace NUMINAMATH_CALUDE_stratified_sampling_second_grade_l1820_182018

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of students -/
def totalStudents (dist : GradeDistribution) : ℕ :=
  dist.first + dist.second + dist.third

/-- Calculates the number of students to be sampled from a specific grade -/
def sampleSize (dist : GradeDistribution) (totalSample : ℕ) (grade : ℕ) : ℕ :=
  match grade with
  | 1 => (dist.first * totalSample) / (totalStudents dist)
  | 2 => (dist.second * totalSample) / (totalStudents dist)
  | 3 => (dist.third * totalSample) / (totalStudents dist)
  | _ => 0

theorem stratified_sampling_second_grade 
  (dist : GradeDistribution) 
  (h1 : dist.first = 1200) 
  (h2 : dist.second = 900) 
  (h3 : dist.third = 1500) 
  (h4 : totalStudents dist = 3600) 
  (h5 : sampleSize dist 720 2 = 480) : 
  sampleSize dist 720 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_grade_l1820_182018


namespace NUMINAMATH_CALUDE_unique_card_combination_l1820_182072

/-- Represents the colors of the cards --/
inductive Color
  | Green
  | Yellow
  | Blue
  | Red

/-- Represents the set of cards with their numbers --/
def CardSet := Color → Nat

/-- Checks if a number is a valid card number (positive integer less than 10) --/
def isValidCardNumber (n : Nat) : Prop := 0 < n ∧ n < 10

/-- Checks if all numbers in a card set are valid --/
def allValidNumbers (cards : CardSet) : Prop :=
  ∀ c, isValidCardNumber (cards c)

/-- Checks if all numbers in a card set are different --/
def allDifferentNumbers (cards : CardSet) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → cards c₁ ≠ cards c₂

/-- Checks if the product of green and yellow numbers is the green number --/
def greenYellowProduct (cards : CardSet) : Prop :=
  cards Color.Green * cards Color.Yellow = cards Color.Green

/-- Checks if the blue number is the same as the red number --/
def blueRedSame (cards : CardSet) : Prop :=
  cards Color.Blue = cards Color.Red

/-- Checks if the product of red and blue numbers forms a two-digit number
    with green and yellow digits in that order --/
def redBlueProductCondition (cards : CardSet) : Prop :=
  cards Color.Red * cards Color.Blue = 10 * cards Color.Green + cards Color.Yellow

/-- The main theorem stating that the only valid combination is 8, 1, 9, 9 --/
theorem unique_card_combination :
  ∀ cards : CardSet,
    allValidNumbers cards →
    allDifferentNumbers cards →
    greenYellowProduct cards →
    blueRedSame cards →
    redBlueProductCondition cards →
    (cards Color.Green = 8 ∧
     cards Color.Yellow = 1 ∧
     cards Color.Blue = 9 ∧
     cards Color.Red = 9) :=
by sorry

end NUMINAMATH_CALUDE_unique_card_combination_l1820_182072


namespace NUMINAMATH_CALUDE_frogs_eat_pests_l1820_182083

/-- The number of pests a single frog eats per day -/
def pests_per_frog_per_day : ℕ := 80

/-- The number of frogs -/
def num_frogs : ℕ := 5

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: 5 frogs eat 2800 pests in a week -/
theorem frogs_eat_pests : 
  pests_per_frog_per_day * num_frogs * days_in_week = 2800 := by
  sorry

end NUMINAMATH_CALUDE_frogs_eat_pests_l1820_182083


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l1820_182064

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the solution set of f(x) ≤ 0
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≤ 0}

-- Theorem 1
theorem solution_set_of_inequality (a : ℝ) :
  solution_set a = Set.Icc 1 2 →
  {x : ℝ | f a x ≥ 1 - x^2} = Set.Iic (1/2) ∪ Set.Ici 1 :=
sorry

-- Theorem 2
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
  a ∈ Set.Iic (1/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l1820_182064


namespace NUMINAMATH_CALUDE_flu_spreads_indefinitely_flu_stops_spreading_l1820_182013

-- Define the population as a finite type
variable {Population : Type} [Finite Population]

-- Define the state of a person
inductive State
  | Healthy
  | Infected
  | Immune

-- Define the friendship relation
variable (friends : Population → Population → Prop)

-- Define the state of the population on a given day
variable (state : ℕ → Population → State)

-- Define the condition that each person visits their friends daily
axiom daily_visits : ∀ (d : ℕ) (p q : Population), friends p q → True

-- Define the condition that healthy people become ill after visiting sick friends
axiom infection_spread : ∀ (d : ℕ) (p : Population), 
  state d p = State.Healthy → 
  (∃ (q : Population), friends p q ∧ state d q = State.Infected) → 
  state (d + 1) p = State.Infected

-- Define the condition that illness lasts one day, followed by immunity
axiom illness_duration : ∀ (d : ℕ) (p : Population),
  state d p = State.Infected → state (d + 1) p = State.Immune

-- Define the condition that immunity lasts at least one day
axiom immunity_duration : ∀ (d : ℕ) (p : Population),
  state d p = State.Immune → state (d + 1) p ≠ State.Infected

-- Theorem 1: If some people have immunity initially, the flu can spread indefinitely
theorem flu_spreads_indefinitely (h : ∃ (p : Population), state 0 p = State.Immune) :
  ∀ (n : ℕ), ∃ (d : ℕ) (p : Population), d ≥ n ∧ state d p = State.Infected :=
sorry

-- Theorem 2: If no one has immunity initially, the flu will eventually stop spreading
theorem flu_stops_spreading (h : ∀ (p : Population), state 0 p ≠ State.Immune) :
  ∃ (n : ℕ), ∀ (d : ℕ) (p : Population), d ≥ n → state d p ≠ State.Infected :=
sorry

end NUMINAMATH_CALUDE_flu_spreads_indefinitely_flu_stops_spreading_l1820_182013


namespace NUMINAMATH_CALUDE_constant_function_invariant_l1820_182076

theorem constant_function_invariant (g : ℝ → ℝ) (h : ∀ x : ℝ, g x = -3) :
  ∀ x : ℝ, g (3 * x - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l1820_182076


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1820_182032

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1820_182032


namespace NUMINAMATH_CALUDE_detergent_amount_in_new_solution_l1820_182046

/-- Represents a solution with bleach, detergent, and water -/
structure Solution where
  bleach : ℝ
  detergent : ℝ
  water : ℝ

/-- The original ratio of the solution -/
def original_ratio : Solution :=
  { bleach := 2, detergent := 40, water := 100 }

/-- The new ratio after adjustments -/
def new_ratio (s : Solution) : Solution :=
  { bleach := 3 * s.bleach,
    detergent := s.detergent,
    water := 2 * s.water }

/-- The theorem stating the amount of detergent in the new solution -/
theorem detergent_amount_in_new_solution :
  let s := new_ratio original_ratio
  let water_amount := 300
  let detergent_amount := (s.detergent / s.water) * water_amount
  detergent_amount = 120 := by sorry

end NUMINAMATH_CALUDE_detergent_amount_in_new_solution_l1820_182046


namespace NUMINAMATH_CALUDE_curve_tangent_parallel_l1820_182016

/-- The curve C: y = ax^3 + bx^2 + d -/
def C (a b d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + d

/-- The derivative of C with respect to x -/
def C' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem curve_tangent_parallel (a b d : ℝ) :
  C a b d 1 = 1 →  -- Point A(1,1) lies on the curve
  C a b d (-1) = -3 →  -- Point B(-1,-3) lies on the curve
  C' a b 1 = C' a b (-1) →  -- Tangents at A and B are parallel
  a^3 + b^2 + d = 7 := by sorry

end NUMINAMATH_CALUDE_curve_tangent_parallel_l1820_182016


namespace NUMINAMATH_CALUDE_function_range_l1820_182089

def f (x : ℝ) : ℝ := -x^2 + 3*x + 1

theorem function_range :
  ∃ (a b : ℝ), a = -3 ∧ b = 13/4 ∧
  (∀ x, x ∈ Set.Icc (-1) 2 → f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc (-1) 2, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l1820_182089


namespace NUMINAMATH_CALUDE_georgie_initial_avocados_l1820_182042

/-- The number of avocados needed per serving of guacamole -/
def avocados_per_serving : ℕ := 3

/-- The number of avocados Georgie's sister buys -/
def sister_bought : ℕ := 4

/-- The number of servings Georgie can make -/
def servings : ℕ := 3

/-- Georgie's initial number of avocados -/
def initial_avocados : ℕ := servings * avocados_per_serving - sister_bought

theorem georgie_initial_avocados : initial_avocados = 5 := by
  sorry

end NUMINAMATH_CALUDE_georgie_initial_avocados_l1820_182042


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l1820_182047

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.sin x + 3 * Real.cos x
  ∃ M : ℝ, M = Real.sqrt 13 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l1820_182047
