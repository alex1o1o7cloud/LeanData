import Mathlib

namespace NUMINAMATH_CALUDE_jacket_cost_value_l690_69078

/-- The amount spent on clothing at the mall -/
def total_spent : ℚ := 19.02

/-- The amount spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount spent on the jacket -/
def jacket_cost : ℚ := total_spent - shorts_cost

theorem jacket_cost_value : jacket_cost = 4.74 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_value_l690_69078


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l690_69051

theorem algebraic_expression_value (a : ℤ) (h : a = -2) : a + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l690_69051


namespace NUMINAMATH_CALUDE_marbles_lost_fraction_l690_69031

theorem marbles_lost_fraction (initial_marbles : ℕ) (additional_marbles : ℕ) (new_marbles : ℕ) (final_marbles : ℕ)
  (h1 : initial_marbles = 12)
  (h2 : additional_marbles = 10)
  (h3 : new_marbles = 25)
  (h4 : final_marbles = 41) :
  (initial_marbles - (final_marbles - additional_marbles - new_marbles)) / initial_marbles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_fraction_l690_69031


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l690_69026

/-- The reciprocal of the common fraction form of 0.353535... is 99/35 -/
theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 35 / 99  -- Common fraction form of 0.353535...
  (1 : ℚ) / x = 99 / 35 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l690_69026


namespace NUMINAMATH_CALUDE_parabola_minimum_point_l690_69008

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points A, B, C
def A : ℝ × ℝ := (-1, -3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem parabola_minimum_point (a b c : ℝ) :
  ∃ (m n : ℝ),
    -- The parabola passes through points A, B, C
    parabola a b c A.1 = A.2 ∧
    parabola a b c B.1 = B.2 ∧
    parabola a b c C.1 = C.2 ∧
    -- P(m, n) is on the axis of symmetry
    m = -b / (2 * a) ∧
    -- P(m, n) minimizes PA + PC
    ∀ (x y : ℝ), x = m → parabola a b c x = y →
      (Real.sqrt ((x - A.1)^2 + (y - A.2)^2) +
       Real.sqrt ((x - C.1)^2 + (y - C.2)^2)) ≥
      (Real.sqrt ((m - A.1)^2 + (n - A.2)^2) +
       Real.sqrt ((m - C.1)^2 + (n - C.2)^2)) →
    -- The y-coordinate of P is 0
    n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_minimum_point_l690_69008


namespace NUMINAMATH_CALUDE_regular_polygon_162_degrees_l690_69045

/-- A regular polygon with interior angles measuring 162 degrees has 20 sides -/
theorem regular_polygon_162_degrees : ∀ n : ℕ, 
  n > 2 → 
  (180 * (n - 2) : ℝ) / n = 162 → 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_162_degrees_l690_69045


namespace NUMINAMATH_CALUDE_product_of_large_integers_l690_69066

theorem product_of_large_integers : ∃ (a b : ℤ), 
  a > 10^2009 ∧ b > 10^2009 ∧ a * b = 3^(4^5) + 4^(5^6) := by
  sorry

end NUMINAMATH_CALUDE_product_of_large_integers_l690_69066


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l690_69080

/-- The magnitude of the complex number (2+4i)/(1+i) is √10 -/
theorem magnitude_of_complex_fraction :
  let z : ℂ := (2 + 4 * Complex.I) / (1 + Complex.I)
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l690_69080


namespace NUMINAMATH_CALUDE_tangent_and_below_and_two_zeros_l690_69089

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + 1

def tangent_line (x y : ℝ) : Prop := (1 - a) * x - y = 0

def g (x : ℝ) : ℝ := 1/2 * a * x^2 - (f a x + a * x)

theorem tangent_and_below_and_two_zeros :
  (∀ y, tangent_line a 1 y ↔ y = f a 1) ∧
  (∀ x > 0, x ≠ 1 → f a x < (1 - a) * x) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ ∀ x, x ≠ x₁ → x ≠ x₂ → g a x ≠ 0) ↔
  0 < a ∧ a < Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_and_below_and_two_zeros_l690_69089


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l690_69098

theorem diophantine_equation_solution :
  ∀ x y : ℕ+,
  let d := Nat.gcd x.val y.val
  x.val * y.val * d = x.val + y.val + d^2 →
  (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l690_69098


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l690_69076

/-- The area of shaded region formed by semicircles in a pattern --/
theorem semicircle_pattern_area (d : ℝ) (l : ℝ) (h1 : d = 4) (h2 : l = 24) : 
  (l / d) / 2 * (π * (d / 2)^2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l690_69076


namespace NUMINAMATH_CALUDE_locus_empty_near_origin_l690_69011

/-- Represents a polynomial of degree 3 in two variables -/
structure Polynomial3 (α : Type*) [Ring α] where
  A : α
  B : α
  C : α
  D : α
  E : α
  F : α
  G : α

/-- Evaluates the polynomial at a given point (x, y) -/
def eval_poly (p : Polynomial3 ℝ) (x y : ℝ) : ℝ :=
  p.A * x^2 + p.B * x * y + p.C * y^2 + p.D * x^3 + p.E * x^2 * y + p.F * x * y^2 + p.G * y^3

theorem locus_empty_near_origin (p : Polynomial3 ℝ) (h : p.B^2 - 4 * p.A * p.C < 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x y : ℝ, 0 < x^2 + y^2 ∧ x^2 + y^2 < δ^2 → eval_poly p x y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_empty_near_origin_l690_69011


namespace NUMINAMATH_CALUDE_race_course_length_race_course_length_proof_l690_69086

/-- Given two runners A and B, where A runs 4 times as fast as B and gives B a 63-meter head start,
    the length of the race course that allows both runners to finish at the same time is 84 meters. -/
theorem race_course_length : ℝ → ℝ → Prop :=
  fun (speed_B : ℝ) (course_length : ℝ) =>
    speed_B > 0 →
    course_length > 63 →
    course_length / (4 * speed_B) = (course_length - 63) / speed_B →
    course_length = 84

/-- Proof of the race_course_length theorem -/
theorem race_course_length_proof : ∃ (speed_B : ℝ) (course_length : ℝ),
  race_course_length speed_B course_length :=
by
  sorry

end NUMINAMATH_CALUDE_race_course_length_race_course_length_proof_l690_69086


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l690_69070

theorem diophantine_equation_solution (k : ℕ+) : 
  (∃ (x y : ℕ+), x^2 + y^2 = k * x * y - 1) ↔ k = 3 := by
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l690_69070


namespace NUMINAMATH_CALUDE_geometric_sum_of_root_l690_69052

theorem geometric_sum_of_root (x : ℝ) : 
  x^10 - 3*x + 2 = 0 → x ≠ 1 → x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_of_root_l690_69052


namespace NUMINAMATH_CALUDE_augmented_matrix_sum_l690_69050

/-- Given an augmented matrix representing a system of linear equations and its solution,
    prove that the sum of certain elements in the matrix equals 10. -/
theorem augmented_matrix_sum (m n : ℝ) : 
  (∃ (A : Matrix (Fin 2) (Fin 3) ℝ), 
    A = ![![m, 0, 6],
         ![0, 3, n]] ∧ 
    (∀ (x y : ℝ), x = -3 ∧ y = 4 → m * x = 6 ∧ 3 * y = n)) →
  m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_sum_l690_69050


namespace NUMINAMATH_CALUDE_larger_number_problem_l690_69040

theorem larger_number_problem (x y : ℤ) : 
  x + y = 56 → y = x + 12 → y = 34 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l690_69040


namespace NUMINAMATH_CALUDE_floor_times_x_equals_48_l690_69016

theorem floor_times_x_equals_48 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 48 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_equals_48_l690_69016


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l690_69069

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + p * x - 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, 3 * y^2 + p * y - 6 = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l690_69069


namespace NUMINAMATH_CALUDE_fraction_under_eleven_l690_69090

theorem fraction_under_eleven (total : ℕ) (between_eleven_and_thirteen : ℚ) (thirteen_and_above : ℕ) :
  total = 45 →
  between_eleven_and_thirteen = 2 / 5 →
  thirteen_and_above = 12 →
  (total : ℚ) - between_eleven_and_thirteen * total - (thirteen_and_above : ℚ) = 1 / 3 * total :=
by sorry

end NUMINAMATH_CALUDE_fraction_under_eleven_l690_69090


namespace NUMINAMATH_CALUDE_triangle_ratio_l690_69049

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 3 * a →
  b / a = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l690_69049


namespace NUMINAMATH_CALUDE_largest_points_with_empty_square_fifteen_points_optimal_l690_69068

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Checks if a point is inside a square -/
def is_point_inside_square (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

/-- The main theorem -/
theorem largest_points_with_empty_square :
  ∀ (points : List Point),
    (∀ p ∈ points, 0 < p.x ∧ p.x < 4 ∧ 0 < p.y ∧ p.y < 4) →
    points.length ≤ 15 →
    ∃ (s : Square),
      s.side_length = 1 ∧
      0 ≤ s.center.x ∧ s.center.x ≤ 3 ∧
      0 ≤ s.center.y ∧ s.center.y ≤ 3 ∧
      ∀ p ∈ points, ¬is_point_inside_square p s :=
by sorry

/-- The optimality of 15 -/
theorem fifteen_points_optimal :
  ∃ (points : List Point),
    points.length = 16 ∧
    (∀ p ∈ points, 0 < p.x ∧ p.x < 4 ∧ 0 < p.y ∧ p.y < 4) ∧
    ∀ (s : Square),
      s.side_length = 1 →
      0 ≤ s.center.x ∧ s.center.x ≤ 3 →
      0 ≤ s.center.y ∧ s.center.y ≤ 3 →
      ∃ p ∈ points, is_point_inside_square p s :=
by sorry

end NUMINAMATH_CALUDE_largest_points_with_empty_square_fifteen_points_optimal_l690_69068


namespace NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l690_69094

theorem max_y_coordinate_ellipse :
  let f (x y : ℝ) := x^2 / 49 + (y + 3)^2 / 25
  ∀ x y : ℝ, f x y = 1 → y ≤ 2 ∧ ∃ x₀ : ℝ, f x₀ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l690_69094


namespace NUMINAMATH_CALUDE_complex_absolute_value_sum_l690_69013

theorem complex_absolute_value_sum : 
  Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) + Complex.abs (1 + 5*I) = 2 * Real.sqrt 34 + Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_sum_l690_69013


namespace NUMINAMATH_CALUDE_birthday_money_theorem_l690_69077

/-- Given an initial amount of money and an amount spent, 
    calculate the remaining amount. -/
def remaining_amount (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that given 67 dollars initially and 
    spending 34 dollars, the remaining amount is 33 dollars. -/
theorem birthday_money_theorem :
  remaining_amount 67 34 = 33 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_theorem_l690_69077


namespace NUMINAMATH_CALUDE_unique_zero_point_implies_same_sign_l690_69073

theorem unique_zero_point_implies_same_sign (f : ℝ → ℝ) :
  Continuous f →
  (∃! x, x ∈ (Set.Ioo 0 2) ∧ f x = 0) →
  f 2 * f 16 > 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_implies_same_sign_l690_69073


namespace NUMINAMATH_CALUDE_square_fraction_below_line_l690_69099

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line passing through two points -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- Represents a square defined by its corners -/
structure Square :=
  (bottomLeft : Point)
  (topRight : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  sorry

/-- Finds the intersection of a line with the right edge of a square -/
def rightEdgeIntersection (l : Line) (s : Square) : Point :=
  sorry

/-- Main theorem: The fraction of the square's area below the line is 1/18 -/
theorem square_fraction_below_line :
  let s := Square.mk (Point.mk 2 0) (Point.mk 5 3)
  let l := Line.mk (Point.mk 2 3) (Point.mk 5 1)
  let intersection := rightEdgeIntersection l s
  let belowArea := triangleArea (Point.mk 2 0) (Point.mk 5 0) intersection
  let totalArea := squareArea s
  belowArea / totalArea = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_below_line_l690_69099


namespace NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l690_69017

theorem power_three_plus_four_mod_five : 3^101 + 4 ≡ 2 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l690_69017


namespace NUMINAMATH_CALUDE_parabola_equation_l690_69091

/-- A parabola with the given properties has the equation y² = 4x -/
theorem parabola_equation (p : ℝ) (h₁ : p > 0) : 
  (∃ M : ℝ × ℝ, M.1 = 3 ∧ 
   ∃ F : ℝ × ℝ, F.1 = p/2 ∧ F.2 = 0 ∧ 
   (M.1 - F.1)^2 + (M.2 - F.2)^2 = (2*p)^2) →
  (∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 4*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l690_69091


namespace NUMINAMATH_CALUDE_remainder_of_n_mod_500_l690_69085

/-- The set S containing elements from 1 to 12 -/
def S : Finset ℕ := Finset.range 12

/-- The number of sets of two non-empty disjoint subsets of S -/
def n : ℕ := ((3^12 - 2 * 2^12 + 1) / 2 : ℕ)

/-- Theorem stating that the remainder of n divided by 500 is 125 -/
theorem remainder_of_n_mod_500 : n % 500 = 125 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_mod_500_l690_69085


namespace NUMINAMATH_CALUDE_delivery_driver_net_pay_l690_69021

/-- Calculates the net rate of pay for a delivery driver --/
theorem delivery_driver_net_pay 
  (travel_time : ℝ) 
  (speed : ℝ) 
  (fuel_efficiency : ℝ) 
  (earnings_per_mile : ℝ) 
  (gasoline_price : ℝ) 
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : earnings_per_mile = 0.60)
  (h5 : gasoline_price = 2.50) : 
  (earnings_per_mile * speed * travel_time - 
   (speed * travel_time / fuel_efficiency) * gasoline_price) / travel_time = 25 := by
  sorry

#check delivery_driver_net_pay

end NUMINAMATH_CALUDE_delivery_driver_net_pay_l690_69021


namespace NUMINAMATH_CALUDE_absolute_value_and_roots_l690_69079

theorem absolute_value_and_roots : |-3| + (Real.sqrt 2 - 1)^0 - (Real.sqrt 3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_roots_l690_69079


namespace NUMINAMATH_CALUDE_original_fraction_l690_69075

theorem original_fraction (x y : ℚ) : 
  x / (y + 1) = 1 / 2 → (x + 1) / y = 1 → x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l690_69075


namespace NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l690_69054

/-- The number of amoebas after n days, given an initial population of 1 and a tripling growth rate each day. -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of amoebas after 10 days is equal to 3^10. -/
theorem amoeba_count_after_ten_days : amoeba_count 10 = 3^10 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l690_69054


namespace NUMINAMATH_CALUDE_hayley_stickers_l690_69039

theorem hayley_stickers (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (h1 : num_friends = 9) (h2 : stickers_per_friend = 8) : 
  num_friends * stickers_per_friend = 72 := by
  sorry

end NUMINAMATH_CALUDE_hayley_stickers_l690_69039


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l690_69095

/-- A point in a plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points form an isosceles triangle -/
def isIsosceles (p1 p2 p3 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d31 := (p3.x - p1.x)^2 + (p3.y - p1.y)^2
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- A configuration of five points in a plane -/
def Configuration := Fin 5 → Point

/-- Check if a configuration satisfies the isosceles condition for all triplets -/
def validConfiguration (config : Configuration) : Prop :=
  ∀ i j k, i < j → j < k → isIsosceles (config i) (config j) (config k)

/-- There exists a configuration of five points satisfying the isosceles condition -/
theorem exists_valid_configuration : ∃ (config : Configuration), validConfiguration config := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l690_69095


namespace NUMINAMATH_CALUDE_lcm_gcd_product_8_16_l690_69027

theorem lcm_gcd_product_8_16 : Nat.lcm 8 16 * Nat.gcd 8 16 = 128 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_8_16_l690_69027


namespace NUMINAMATH_CALUDE_fraction_subtraction_l690_69096

theorem fraction_subtraction (a b c d x : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a - x) / (b - x) = c / d) 
  (h4 : d ≠ c) : 
  x = (b * c - a * d) / (d - c) := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l690_69096


namespace NUMINAMATH_CALUDE_digit_before_y_l690_69038

/-- Given a number of the form xy86038 where x and y are single digits,
    if y = 3 and the number is divisible by 11,
    then x = 6 -/
theorem digit_before_y (x y : ℕ) : 
  y = 3 →
  x < 10 →
  y < 10 →
  (x * 1000000 + y * 100000 + 86038) % 11 = 0 →
  (∀ z < y, (x * 1000000 + z * 100000 + 86038) % 11 ≠ 0) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_digit_before_y_l690_69038


namespace NUMINAMATH_CALUDE_garden_carnations_percentage_l690_69033

theorem garden_carnations_percentage 
  (total : ℕ) 
  (pink : ℕ) 
  (white : ℕ) 
  (pink_roses : ℕ) 
  (red_carnations : ℕ) 
  (h_pink : pink = 3 * total / 5)
  (h_white : white = total / 5)
  (h_pink_roses : pink_roses = pink / 2)
  (h_red_carnations : red_carnations = (total - pink - white) / 2) :
  (pink - pink_roses + red_carnations + white) * 100 = 60 * total :=
sorry

end NUMINAMATH_CALUDE_garden_carnations_percentage_l690_69033


namespace NUMINAMATH_CALUDE_division_problem_l690_69034

theorem division_problem (x : ℤ) : (64 / x = 4) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l690_69034


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l690_69023

/-- A quadratic equation in x is of the form ax² + bx + c = 0 where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

/-- The coefficients of a general second-degree polynomial equation -/
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the condition for an equation to be quadratic -/
theorem quadratic_equation_condition (coeff : QuadraticCoefficients) :
  is_quadratic_equation coeff.a coeff.b coeff.c ↔ coeff.a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l690_69023


namespace NUMINAMATH_CALUDE_smallest_n_for_integral_solutions_l690_69015

theorem smallest_n_for_integral_solutions : 
  ∀ n : ℕ+, 
  (∃ x : ℤ, 12 * x^2 - n * x + 576 = 0) → 
  n ≥ 168 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integral_solutions_l690_69015


namespace NUMINAMATH_CALUDE_simple_interest_principal_l690_69014

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 0.05 →
  time = 1 →
  interest = 500 →
  principal * rate * time = interest →
  principal = 10000 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l690_69014


namespace NUMINAMATH_CALUDE_x_seventh_minus_27x_squared_l690_69046

theorem x_seventh_minus_27x_squared (x : ℝ) (h : x^3 - 3*x = 6) :
  x^7 - 27*x^2 = 9*(x + 1)*(x + 6) := by
  sorry

end NUMINAMATH_CALUDE_x_seventh_minus_27x_squared_l690_69046


namespace NUMINAMATH_CALUDE_remainder_1732_base12_div_9_l690_69074

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 1732₁₂ --/
def number_1732_base12 : List Nat := [2, 3, 7, 1]

theorem remainder_1732_base12_div_9 :
  (base12ToBase10 number_1732_base12) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1732_base12_div_9_l690_69074


namespace NUMINAMATH_CALUDE_incorrect_complex_analogy_l690_69056

def complex_square_property (z : ℂ) : Prop :=
  Complex.abs z ^ 2 = z ^ 2

theorem incorrect_complex_analogy :
  ∃ z : ℂ, ¬(complex_square_property z) :=
sorry

end NUMINAMATH_CALUDE_incorrect_complex_analogy_l690_69056


namespace NUMINAMATH_CALUDE_university_tuition_cost_l690_69001

def cost_first_8_years : ℕ := 8 * 10000
def cost_next_10_years : ℕ := 10 * 20000
def total_raising_cost : ℕ := cost_first_8_years + cost_next_10_years
def johns_contribution : ℕ := total_raising_cost / 2
def total_cost_with_tuition : ℕ := 265000

theorem university_tuition_cost :
  total_cost_with_tuition - johns_contribution = 125000 :=
by sorry

end NUMINAMATH_CALUDE_university_tuition_cost_l690_69001


namespace NUMINAMATH_CALUDE_simple_sampling_methods_correct_l690_69035

/-- The set of methods for implementing simple sampling -/
def SimpleSamplingMethods : Set String :=
  {"Lottery method", "Random number table method"}

/-- Theorem stating that the set of methods for implementing simple sampling
    contains exactly the lottery method and random number table method -/
theorem simple_sampling_methods_correct :
  SimpleSamplingMethods = {"Lottery method", "Random number table method"} := by
  sorry

end NUMINAMATH_CALUDE_simple_sampling_methods_correct_l690_69035


namespace NUMINAMATH_CALUDE_readers_all_genres_l690_69005

theorem readers_all_genres (total : ℕ) (sci_fi : ℕ) (literary : ℕ) (non_fiction : ℕ)
  (sci_fi_literary : ℕ) (sci_fi_non_fiction : ℕ) (literary_non_fiction : ℕ) :
  total = 500 →
  sci_fi = 320 →
  literary = 200 →
  non_fiction = 150 →
  sci_fi_literary = 120 →
  sci_fi_non_fiction = 80 →
  literary_non_fiction = 60 →
  ∃ (all_genres : ℕ),
    all_genres = 90 ∧
    total = sci_fi + literary + non_fiction -
      (sci_fi_literary + sci_fi_non_fiction + literary_non_fiction) + all_genres :=
by
  sorry

end NUMINAMATH_CALUDE_readers_all_genres_l690_69005


namespace NUMINAMATH_CALUDE_smallest_valid_number_l690_69002

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≥ 1 ∧ d ≤ 9 → ∃ k : ℕ, n / (10^k) % 10 = d

def is_smallest_valid_number (n : ℕ) : Prop :=
  n % 72 = 0 ∧
  contains_all_digits n ∧
  ∀ m : ℕ, m < n → ¬(m % 72 = 0 ∧ contains_all_digits m)

theorem smallest_valid_number : is_smallest_valid_number 123457968 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l690_69002


namespace NUMINAMATH_CALUDE_game_positions_after_359_moves_l690_69007

/-- Represents the four positions of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the twelve positions of the mouse -/
inductive MousePosition
  | TopLeft | TopMiddle | TopRight
  | RightTop | RightMiddle | RightBottom
  | BottomRight | BottomMiddle | BottomLeft
  | LeftBottom | LeftMiddle | LeftTop

/-- Calculates the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : ℕ) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.TopRight
  | 2 => CatPosition.BottomRight
  | _ => CatPosition.BottomLeft

/-- Calculates the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : ℕ) : MousePosition :=
  match moves % 12 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightTop
  | 4 => MousePosition.RightMiddle
  | 5 => MousePosition.RightBottom
  | 6 => MousePosition.BottomRight
  | 7 => MousePosition.BottomMiddle
  | 8 => MousePosition.BottomLeft
  | 9 => MousePosition.LeftBottom
  | 10 => MousePosition.LeftMiddle
  | _ => MousePosition.LeftTop

theorem game_positions_after_359_moves :
  catPositionAfterMoves 359 = CatPosition.BottomRight ∧
  mousePositionAfterMoves 359 = MousePosition.LeftMiddle :=
by sorry

end NUMINAMATH_CALUDE_game_positions_after_359_moves_l690_69007


namespace NUMINAMATH_CALUDE_ellipse_symmetric_point_range_l690_69053

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

/-- Definition of symmetry with respect to y = 2x -/
def symmetric_points (x₀ y₀ x₁ y₁ : ℝ) : Prop :=
  (y₀ - y₁) / (x₀ - x₁) = -1/2 ∧ (y₀ + y₁) / 2 = 2 * ((x₀ + x₁) / 2)

/-- The main theorem -/
theorem ellipse_symmetric_point_range :
  ∀ x₀ y₀ x₁ y₁ : ℝ,
  ellipse_C x₀ y₀ →
  symmetric_points x₀ y₀ x₁ y₁ →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_point_range_l690_69053


namespace NUMINAMATH_CALUDE_ad_arrangement_count_l690_69087

/-- The number of ways to arrange n items, taking r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways to arrange 6 advertisements (4 commercial and 2 public service) 
    where the 2 public service ads cannot be consecutive -/
def ad_arrangements : ℕ :=
  permutations 4 4 * permutations 5 2

theorem ad_arrangement_count : 
  ad_arrangements = permutations 4 4 * permutations 5 2 := by
  sorry

end NUMINAMATH_CALUDE_ad_arrangement_count_l690_69087


namespace NUMINAMATH_CALUDE_roses_planted_is_difference_l690_69061

/-- The number of rose bushes planted in a park --/
def rosesBushesPlanted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of rose bushes planted is the difference between final and initial counts --/
theorem roses_planted_is_difference (initial final : ℕ) (h : final ≥ initial) :
  rosesBushesPlanted initial final = final - initial :=
by
  sorry

/-- Specific instance for the given problem --/
example : rosesBushesPlanted 2 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_planted_is_difference_l690_69061


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l690_69048

/-- Given a cubic polynomial x^3 + ax^2 + bx + c, returns the sum of its roots -/
def sumOfRoots (a b c : ℝ) : ℝ := -a

/-- Given a cubic polynomial x^3 + ax^2 + bx + c, returns the product of its roots -/
def productOfRoots (a b c : ℝ) : ℝ := -c

theorem cubic_roots_problem (p q r u v w : ℝ) :
  (∀ x, x^3 + 2*x^2 + 5*x - 8 = (x - p)*(x - q)*(x - r)) →
  (∀ x, x^3 + u*x^2 + v*x + w = (x - (p + q))*(x - (q + r))*(x - (r + p))) →
  w = 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l690_69048


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l690_69036

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 - 5*x - 6 < 0
def inequality2 (x : ℝ) : Prop := (x - 1) / (x + 2) ≤ 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -1 < x ∧ x < 6}
def solution_set2 : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 :=
sorry

theorem inequality2_solution : 
  ∀ x : ℝ, x ≠ -2 → (inequality2 x ↔ x ∈ solution_set2) :=
sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l690_69036


namespace NUMINAMATH_CALUDE_smallest_number_l690_69010

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Represents the number 85 in base 9 --/
def num1 : List Nat := [8, 5]

/-- Represents the number 1000 in base 4 --/
def num2 : List Nat := [1, 0, 0, 0]

/-- Represents the number 111111 in base 2 --/
def num3 : List Nat := [1, 1, 1, 1, 1, 1]

theorem smallest_number :
  to_base_10 num3 2 ≤ to_base_10 num1 9 ∧
  to_base_10 num3 2 ≤ to_base_10 num2 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l690_69010


namespace NUMINAMATH_CALUDE_xavier_speed_increase_time_l690_69020

/-- Represents the journey of Xavier from p to q -/
structure Journey where
  initialSpeed : ℝ  -- Initial speed in km/h
  speedIncrease : ℝ  -- Speed increase in km/h
  totalDistance : ℝ  -- Total distance in km
  totalTime : ℝ  -- Total time in hours

/-- Calculates the time at which Xavier increases his speed -/
def timeOfSpeedIncrease (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that Xavier increases his speed after 24 minutes -/
theorem xavier_speed_increase_time (j : Journey) 
  (h1 : j.initialSpeed = 50)
  (h2 : j.speedIncrease = 10)
  (h3 : j.totalDistance = 52)
  (h4 : j.totalTime = 48 / 60) : 
  timeOfSpeedIncrease j = 24 / 60 := by
  sorry

end NUMINAMATH_CALUDE_xavier_speed_increase_time_l690_69020


namespace NUMINAMATH_CALUDE_sum_other_y_coordinates_specific_parallelogram_l690_69093

/-- A parallelogram with two opposite corners given -/
structure Parallelogram where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- The sum of y-coordinates of the other two vertices of the parallelogram -/
def sumOtherYCoordinates (p : Parallelogram) : ℝ :=
  (p.corner1.2 + p.corner2.2)

theorem sum_other_y_coordinates_specific_parallelogram :
  let p := Parallelogram.mk (2, 15) (8, -6)
  sumOtherYCoordinates p = 9 := by
  sorry

#check sum_other_y_coordinates_specific_parallelogram

end NUMINAMATH_CALUDE_sum_other_y_coordinates_specific_parallelogram_l690_69093


namespace NUMINAMATH_CALUDE_triangle_inequality_l690_69018

theorem triangle_inequality (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l690_69018


namespace NUMINAMATH_CALUDE_probability_purple_face_l690_69072

/-- The probability of rolling a purple face on a 10-sided die with 3 purple faces is 3/10. -/
theorem probability_purple_face (total_faces : ℕ) (purple_faces : ℕ) 
  (h1 : total_faces = 10) (h2 : purple_faces = 3) : 
  (purple_faces : ℚ) / total_faces = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_purple_face_l690_69072


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l690_69058

theorem right_triangle_side_length 
  (west_distance : ℝ) 
  (total_distance : ℝ) 
  (h1 : west_distance = 10) 
  (h2 : total_distance = 14.142135623730951) : 
  ∃ (north_distance : ℝ), 
    north_distance^2 + west_distance^2 = total_distance^2 ∧ 
    north_distance = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l690_69058


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l690_69000

theorem greatest_multiple_under_1000 : ∃ (n : ℕ), n = 945 ∧ 
  n < 1000 ∧ 
  3 ∣ n ∧ 
  5 ∣ n ∧ 
  7 ∣ n ∧ 
  ∀ m : ℕ, m < 1000 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l690_69000


namespace NUMINAMATH_CALUDE_functional_equation_solution_l690_69060

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of being twice differentiable with continuous second derivative
def TwiceDifferentiableContinuous (f : RealFunction) : Prop :=
  Differentiable ℝ f ∧ 
  Differentiable ℝ (deriv f) ∧ 
  Continuous (deriv (deriv f))

-- Define the functional equation
def SatisfiesFunctionalEquation (f : RealFunction) : Prop :=
  ∀ t : ℝ, f t ^ 2 = f (t * Real.sqrt 2)

-- Main theorem
theorem functional_equation_solution 
  (f : RealFunction) 
  (h1 : TwiceDifferentiableContinuous f) 
  (h2 : SatisfiesFunctionalEquation f) : 
  (∃ c : ℝ, ∀ x : ℝ, f x = Real.exp (c * x^2)) ∨ 
  (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l690_69060


namespace NUMINAMATH_CALUDE_repair_cost_is_2400_l690_69037

/-- The total cost of car repairs given labor rate, labor hours, and part cost. -/
def total_repair_cost (labor_rate : ℕ) (labor_hours : ℕ) (part_cost : ℕ) : ℕ :=
  labor_rate * labor_hours + part_cost

/-- Theorem stating that the total repair cost is $2400 given the specified conditions. -/
theorem repair_cost_is_2400 :
  total_repair_cost 75 16 1200 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_2400_l690_69037


namespace NUMINAMATH_CALUDE_final_value_of_A_l690_69024

theorem final_value_of_A : ∀ (A : ℤ), A = 15 → -A + 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l690_69024


namespace NUMINAMATH_CALUDE_smallest_class_size_l690_69012

theorem smallest_class_size (n : ℕ) : 
  (4*n + 2 > 40) ∧ 
  (∀ m : ℕ, m < n → 4*m + 2 ≤ 40) → 
  4*n + 2 = 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l690_69012


namespace NUMINAMATH_CALUDE_richard_david_age_difference_l690_69057

-- Define the ages of the three sons
def david_age : ℕ := 14
def scott_age : ℕ := david_age - 8
def richard_age : ℕ := scott_age * 2 + 8

-- Define the conditions
theorem richard_david_age_difference :
  richard_age - david_age = 6 :=
by
  -- Proof goes here
  sorry

#check richard_david_age_difference

end NUMINAMATH_CALUDE_richard_david_age_difference_l690_69057


namespace NUMINAMATH_CALUDE_original_price_from_discounted_l690_69083

/-- 
Given a shirt sold at a discounted price with a known discount percentage, 
this theorem proves the original selling price.
-/
theorem original_price_from_discounted (discounted_price : ℝ) (discount_percent : ℝ) 
  (h1 : discounted_price = 560) 
  (h2 : discount_percent = 20) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percent / 100) = discounted_price ∧ 
    original_price = 700 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_discounted_l690_69083


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l690_69084

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given a geometric sequence {aₙ} satisfying a₁ + a₂ = 3 and a₂ + a₃ = 6, prove that a₇ = 64 -/
theorem geometric_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_geom : IsGeometricSequence a) 
  (h_sum1 : a 1 + a 2 = 3) 
  (h_sum2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l690_69084


namespace NUMINAMATH_CALUDE_select_blocks_count_l690_69055

/-- The number of ways to select 4 blocks from a 6x6 grid such that no two blocks are in the same row or column -/
def select_blocks : ℕ :=
  Nat.choose 6 4 * Nat.choose 6 4 * Nat.factorial 4

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    such that no two blocks are in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l690_69055


namespace NUMINAMATH_CALUDE_sales_executive_target_earning_l690_69041

/-- Calculates the target monthly earning for a sales executive --/
def target_monthly_earning (fixed_salary : ℝ) (commission_rate : ℝ) (required_sales : ℝ) : ℝ :=
  fixed_salary + commission_rate * required_sales

/-- Proves that the target monthly earning is $5000 given the specified conditions --/
theorem sales_executive_target_earning :
  target_monthly_earning 1000 0.05 80000 = 5000 := by
sorry

end NUMINAMATH_CALUDE_sales_executive_target_earning_l690_69041


namespace NUMINAMATH_CALUDE_group_size_calculation_l690_69029

theorem group_size_calculation (n : ℕ) : 
  (n * 14 + 34) / (n + 1) = 16 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l690_69029


namespace NUMINAMATH_CALUDE_parking_lot_capacity_l690_69006

theorem parking_lot_capacity (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425)
  (h2 : num_levels = 5)
  (h3 : parked_cars = 23) :
  (total_capacity / num_levels) - parked_cars = 62 := by
  sorry

#check parking_lot_capacity

end NUMINAMATH_CALUDE_parking_lot_capacity_l690_69006


namespace NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l690_69042

/-- Represents a right triangle with a perpendicular from the right angle to the hypotenuse -/
structure RightTriangleWithAltitude where
  /-- Length of the shorter leg -/
  short_leg : ℝ
  /-- Length of the longer leg -/
  long_leg : ℝ
  /-- The longer leg is 3 times the shorter leg -/
  leg_ratio : long_leg = 3 * short_leg
  /-- Length of the segment of the hypotenuse adjacent to the shorter leg -/
  hyp_short : ℝ
  /-- Length of the segment of the hypotenuse adjacent to the longer leg -/
  hyp_long : ℝ
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : short_leg ^ 2 + long_leg ^ 2 = (hyp_short + hyp_long) ^ 2

/-- The main theorem: the ratio of hypotenuse segments is 9:1 -/
theorem hypotenuse_segment_ratio (t : RightTriangleWithAltitude) : 
  t.hyp_long / t.hyp_short = 9 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l690_69042


namespace NUMINAMATH_CALUDE_sum_of_partial_fractions_coefficients_l690_69043

theorem sum_of_partial_fractions_coefficients (A B C D E : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 ∧ x ≠ -6 →
    (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
    A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_partial_fractions_coefficients_l690_69043


namespace NUMINAMATH_CALUDE_hash_difference_l690_69064

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 4 2 - hash 2 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l690_69064


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l690_69097

theorem y_intercept_of_line (x y : ℝ) :
  x + 2*y - 1 = 0 → x = 0 → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l690_69097


namespace NUMINAMATH_CALUDE_partner_b_share_l690_69062

/-- Calculates the share of a partner in a partnership. -/
def calculate_share (total_profit : ℚ) (investment : ℚ) (total_investment : ℚ) : ℚ :=
  (investment / total_investment) * total_profit

theorem partner_b_share 
  (investment_a investment_b investment_c : ℚ)
  (share_a : ℚ)
  (h1 : investment_a = 7000)
  (h2 : investment_b = 11000)
  (h3 : investment_c = 18000)
  (h4 : share_a = 560) :
  calculate_share 
    ((share_a * (investment_a + investment_b + investment_c)) / investment_a)
    investment_b
    (investment_a + investment_b + investment_c) = 880 := by
  sorry

#eval calculate_share (560 * 36 / 7) 11000 36000

end NUMINAMATH_CALUDE_partner_b_share_l690_69062


namespace NUMINAMATH_CALUDE_janet_action_figures_l690_69009

/-- Calculates the final number of action figures Janet has -/
def final_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  let after_selling := initial - sold
  let after_buying := after_selling + bought
  let brothers_collection := 2 * after_buying
  after_buying + brothers_collection

/-- Proves that Janet ends up with 24 action figures given the initial conditions -/
theorem janet_action_figures :
  final_action_figures 10 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l690_69009


namespace NUMINAMATH_CALUDE_equation_solutions_l690_69065

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 6 ∧ 
    ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 3/4 ∧ 
    ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l690_69065


namespace NUMINAMATH_CALUDE_bus_speed_problem_l690_69028

/-- Given two buses traveling in opposite directions, this theorem proves
    the speed of the second bus given the conditions of the problem. -/
theorem bus_speed_problem (east_speed : ℝ) (time : ℝ) (total_distance : ℝ)
  (h1 : east_speed = 55)
  (h2 : time = 4)
  (h3 : total_distance = 460) :
  ∃ west_speed : ℝ, 
    west_speed * time + east_speed * time = total_distance ∧
    west_speed = 60 :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l690_69028


namespace NUMINAMATH_CALUDE_cube_with_corners_removed_faces_l690_69092

-- Define the properties of the cube
def cube_side_length : ℝ := 3
def small_cube_side_length : ℝ := 1
def initial_faces : ℕ := 6
def corners_in_cube : ℕ := 8
def new_faces_per_corner : ℕ := 3

-- Theorem statement
theorem cube_with_corners_removed_faces :
  initial_faces + corners_in_cube * new_faces_per_corner = 30 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_corners_removed_faces_l690_69092


namespace NUMINAMATH_CALUDE_smallest_n_squared_plus_n_divisibility_l690_69059

theorem smallest_n_squared_plus_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), (1 ≤ k) ∧ (k ≤ n) ∧ ((n^2 + n) % k = 0)) ∧
  (∃ (k : ℕ), (1 ≤ k) ∧ (k ≤ n) ∧ ((n^2 + n) % k ≠ 0)) ∧
  (∀ (m : ℕ), (m > 0) ∧ (m < n) → 
    (∀ (k : ℕ), (1 ≤ k) ∧ (k ≤ m) → ((m^2 + m) % k = 0)) ∨
    (∀ (k : ℕ), (1 ≤ k) ∧ (k ≤ m) → ((m^2 + m) % k ≠ 0))) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_squared_plus_n_divisibility_l690_69059


namespace NUMINAMATH_CALUDE_remainder_a_fourth_plus_four_l690_69030

theorem remainder_a_fourth_plus_four (a : ℤ) (h : ¬ (5 ∣ a)) : (a^4 + 4) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_a_fourth_plus_four_l690_69030


namespace NUMINAMATH_CALUDE_remainder_sum_l690_69063

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l690_69063


namespace NUMINAMATH_CALUDE_arrangements_with_non_adjacent_students_l690_69082

def number_of_students : ℕ := 5

-- Total number of permutations for n students
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Number of permutations where A and B are adjacent
def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem arrangements_with_non_adjacent_students :
  total_permutations number_of_students - adjacent_permutations number_of_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_non_adjacent_students_l690_69082


namespace NUMINAMATH_CALUDE_blossom_room_area_l690_69081

/-- Represents the length of a side of a square room in feet -/
def room_side_length : ℕ := 10

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Calculates the area of a square room in square inches -/
def room_area_sq_inches (side_length : ℕ) (inches_per_foot : ℕ) : ℕ :=
  (side_length * inches_per_foot) ^ 2

/-- Theorem stating that the area of Blossom's room is 14400 square inches -/
theorem blossom_room_area :
  room_area_sq_inches room_side_length inches_per_foot = 14400 := by
  sorry

end NUMINAMATH_CALUDE_blossom_room_area_l690_69081


namespace NUMINAMATH_CALUDE_target_hit_probability_l690_69088

theorem target_hit_probability (hit_rate_A hit_rate_B : ℚ) 
  (h1 : hit_rate_A = 4/5)
  (h2 : hit_rate_B = 5/6) :
  1 - (1 - hit_rate_A) * (1 - hit_rate_B) = 29/30 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l690_69088


namespace NUMINAMATH_CALUDE_wall_volume_theorem_l690_69004

/-- Calculates the volume of a rectangular wall given its width and height-to-width and length-to-height ratios -/
def wall_volume (width : ℝ) (height_ratio : ℝ) (length_ratio : ℝ) : ℝ :=
  width * (height_ratio * width) * (length_ratio * height_ratio * width)

/-- Theorem: The volume of a wall with width 4m, height 6 times its width, and length 7 times its height is 16128 cubic meters -/
theorem wall_volume_theorem :
  wall_volume 4 6 7 = 16128 := by
  sorry

end NUMINAMATH_CALUDE_wall_volume_theorem_l690_69004


namespace NUMINAMATH_CALUDE_f_value_at_2017_5_l690_69022

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def f_on_unit_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

theorem f_value_at_2017_5 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period_2 f) 
  (h_unit : f_on_unit_interval f) : 
  f 2017.5 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_2017_5_l690_69022


namespace NUMINAMATH_CALUDE_cost_of_scooter_l690_69071

def scooter_cost (megan_money tara_money : ℕ) : Prop :=
  (tara_money = megan_money + 4) ∧
  (tara_money = 15) ∧
  (megan_money + tara_money = 26)

theorem cost_of_scooter :
  ∃ (megan_money tara_money : ℕ), scooter_cost megan_money tara_money :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_scooter_l690_69071


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l690_69003

theorem bernoulli_inequality (c x : ℝ) (p : ℤ) 
  (hc : c > 0) (hp : p > 1) (hx1 : x > -1) (hx2 : x ≠ 0) : 
  (1 + x)^p > 1 + p * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l690_69003


namespace NUMINAMATH_CALUDE_cookies_per_bag_l690_69025

/-- Proves that the number of cookies in each bag is 20, given the conditions of the problem. -/
theorem cookies_per_bag (bags_per_box : ℕ) (total_calories : ℕ) (calories_per_cookie : ℕ)
  (h1 : bags_per_box = 4)
  (h2 : total_calories = 1600)
  (h3 : calories_per_cookie = 20) :
  total_calories / (bags_per_box * calories_per_cookie) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l690_69025


namespace NUMINAMATH_CALUDE_monopoly_houses_theorem_l690_69032

structure Player where
  name : String
  initialHouses : ℕ
  deriving Repr

def seanTransactions (houses : ℕ) : ℕ :=
  houses - 15 + 18

def karenTransactions (houses : ℕ) : ℕ :=
  0 + 10 + 8 + 15

def markTransactions (houses : ℕ) : ℕ :=
  houses + 12 - 25 - 15

def lucyTransactions (houses : ℕ) : ℕ :=
  houses - 8 + 6 - 20

def finalHouses (player : Player) : ℕ :=
  match player.name with
  | "Sean" => seanTransactions player.initialHouses
  | "Karen" => karenTransactions player.initialHouses
  | "Mark" => markTransactions player.initialHouses
  | "Lucy" => lucyTransactions player.initialHouses
  | _ => player.initialHouses

theorem monopoly_houses_theorem (sean karen mark lucy : Player)
  (h1 : sean.name = "Sean" ∧ sean.initialHouses = 45)
  (h2 : karen.name = "Karen" ∧ karen.initialHouses = 30)
  (h3 : mark.name = "Mark" ∧ mark.initialHouses = 55)
  (h4 : lucy.name = "Lucy" ∧ lucy.initialHouses = 35) :
  finalHouses sean = 48 ∧
  finalHouses karen = 33 ∧
  finalHouses mark = 27 ∧
  finalHouses lucy = 13 := by
  sorry

end NUMINAMATH_CALUDE_monopoly_houses_theorem_l690_69032


namespace NUMINAMATH_CALUDE_no_equilateral_from_splice_l690_69067

/-- Represents a triangle with a 45° angle -/
structure Triangle45 where
  -- We only need to define two sides, as the third is determined by the right angle
  side1 : ℝ
  side2 : ℝ
  positive_sides : 0 < side1 ∧ 0 < side2

/-- Represents the result of splicing two Triangle45 objects -/
inductive SplicedShape
  | Equilateral
  | Other

/-- Function to splice two Triangle45 objects -/
def splice (t1 t2 : Triangle45) : SplicedShape :=
  sorry

/-- Theorem stating that splicing two Triangle45 objects cannot result in an equilateral triangle -/
theorem no_equilateral_from_splice (t1 t2 : Triangle45) :
  splice t1 t2 ≠ SplicedShape.Equilateral :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_from_splice_l690_69067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l690_69044

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_difference (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  (3 * a 6 = a 3 + a 4 + a 5 + 6) →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l690_69044


namespace NUMINAMATH_CALUDE_seating_solution_l690_69019

/-- Represents the seating arrangement problem with changing conditions --/
structure SeatingProblem where
  total_seats : ℕ
  initial_gap : ℕ
  final_gap : ℕ

/-- Calculates the minimum number of occupied seats required --/
def min_occupied_seats (problem : SeatingProblem) : ℕ :=
  sorry

/-- The theorem stating the solution to the specific problem --/
theorem seating_solution :
  let problem : SeatingProblem := {
    total_seats := 150,
    initial_gap := 2,
    final_gap := 1
  }
  min_occupied_seats problem = 57 := by sorry

end NUMINAMATH_CALUDE_seating_solution_l690_69019


namespace NUMINAMATH_CALUDE_sandwich_combinations_l690_69047

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) : 
  meat_types = 12 → cheese_types = 8 → 
  (meat_types.choose 2) * cheese_types = 528 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l690_69047
