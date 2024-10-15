import Mathlib

namespace NUMINAMATH_CALUDE_vasya_fraction_simplification_l1506_150607

theorem vasya_fraction_simplification (n : ℕ) : 
  (n = (2 * (10^1990 - 1)) / 3) → 
  (10^1990 + n) / (10 * n + 4) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_vasya_fraction_simplification_l1506_150607


namespace NUMINAMATH_CALUDE_rectangle_width_l1506_150690

/-- Given a rectangle with area 300 square meters and perimeter 70 meters, prove its width is 15 meters. -/
theorem rectangle_width (length width : ℝ) : 
  length * width = 300 ∧ 
  2 * (length + width) = 70 → 
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1506_150690


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l1506_150610

theorem candy_sampling_percentage (caught_percentage : ℝ) (total_percentage : ℝ)
  (h1 : caught_percentage = 22)
  (h2 : total_percentage = 23.157894736842106) :
  total_percentage - caught_percentage = 1.157894736842106 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l1506_150610


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_value_l1506_150662

/-- A quadratic function f(x) = ax² + bx + c with a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function f(x) defined by the quadratic function -/
def f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The derivative of f(x) -/
def f' (q : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * q.a * x + q.b

theorem quadratic_function_minimum_value (q : QuadraticFunction)
  (h1 : f' q 0 > 0)
  (h2 : ∀ x : ℝ, f q x ≥ 0) :
  2 ≤ (f q 1) / (f' q 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_value_l1506_150662


namespace NUMINAMATH_CALUDE_inequality_theorem_l1506_150673

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hk₁ : x₁ * y₁ - z₁^2 > 0) (hk₂ : x₂ * y₂ - z₂^2 > 0) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
    x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂ ∧ x₁ * y₁ - z₁^2 = x₂ * y₂ - z₂^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1506_150673


namespace NUMINAMATH_CALUDE_gcd_problem_l1506_150678

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 360 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 72) b = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1506_150678


namespace NUMINAMATH_CALUDE_line_equation_coordinate_form_l1506_150670

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line passing through the origin -/
structure Line where
  direction : Vector3D

/-- The direction vector of a line is a unit vector -/
def Line.isUnitVector (l : Line) : Prop :=
  l.direction.x^2 + l.direction.y^2 + l.direction.z^2 = 1

/-- The components of the direction vector are cosines of angles with coordinate axes -/
def Line.directionCosines (l : Line) (α β γ : ℝ) : Prop :=
  l.direction.x = Real.cos α ∧
  l.direction.y = Real.cos β ∧
  l.direction.z = Real.cos γ

/-- A point on the line -/
def Line.pointOnLine (l : Line) (t : ℝ) : Vector3D :=
  { x := t * l.direction.x,
    y := t * l.direction.y,
    z := t * l.direction.z }

/-- The coordinate form of the line equation -/
def Line.coordinateForm (l : Line) (α β γ : ℝ) : Prop :=
  ∀ (p : Vector3D), p ∈ Set.range (l.pointOnLine) →
    p.x / Real.cos α = p.y / Real.cos β ∧
    p.y / Real.cos β = p.z / Real.cos γ

/-- The main theorem: proving the coordinate form of the line equation -/
theorem line_equation_coordinate_form (l : Line) (α β γ : ℝ) :
  l.isUnitVector →
  l.directionCosines α β γ →
  l.coordinateForm α β γ := by
  sorry


end NUMINAMATH_CALUDE_line_equation_coordinate_form_l1506_150670


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1506_150685

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 2 ∧ 2 < x₂ ∧ 
   x₁^2 + 2*a*x₁ - 9 = 0 ∧ 
   x₂^2 + 2*a*x₂ - 9 = 0) → 
  a < 5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1506_150685


namespace NUMINAMATH_CALUDE_equation_system_result_l1506_150654

theorem equation_system_result (x y z : ℝ) 
  (eq1 : 2*x + y + z = 6) 
  (eq2 : x + 2*y + z = 7) : 
  5*x^2 + 8*x*y + 5*y^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_equation_system_result_l1506_150654


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l1506_150696

/-- Given a triangle ABC, this theorem proves the position of point Q
    based on the given ratios of points G and H on the sides of the triangle. -/
theorem intersection_point_coordinates (A B C G H Q : ℝ × ℝ) : 
  (∃ t : ℝ, G = (1 - t) • A + t • B ∧ t = 2/5) →
  (∃ s : ℝ, H = (1 - s) • B + s • C ∧ s = 3/4) →
  (∃ r : ℝ, Q = (1 - r) • A + r • G) →
  (∃ u : ℝ, Q = (1 - u) • C + u • H) →
  Q = (3/8) • A + (1/4) • B + (3/8) • C :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l1506_150696


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l1506_150698

theorem correct_quadratic_equation 
  (b c : ℝ) 
  (h1 : 5 + 1 = -b) 
  (h2 : (-6) * (-4) = c) : 
  b = -10 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l1506_150698


namespace NUMINAMATH_CALUDE_decimal_expansion_four_seventeenths_l1506_150657

/-- The decimal expansion of 4/17 has a repeating block of 235. -/
theorem decimal_expansion_four_seventeenths :
  ∃ (a b : ℕ), (4 : ℚ) / 17 = (a : ℚ) / 999 + (b : ℚ) / (999 * 1000) ∧ a = 235 ∧ b < 999 := by
  sorry

end NUMINAMATH_CALUDE_decimal_expansion_four_seventeenths_l1506_150657


namespace NUMINAMATH_CALUDE_fish_for_white_duck_l1506_150693

/-- The number of fish for each white duck -/
def fish_per_white_duck : ℕ := sorry

/-- The number of fish for each black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish for each multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := 157

theorem fish_for_white_duck :
  fish_per_white_duck * white_ducks +
  fish_per_black_duck * black_ducks +
  fish_per_multicolor_duck * multicolor_ducks = total_fish ∧
  fish_per_white_duck = 5 := by sorry

end NUMINAMATH_CALUDE_fish_for_white_duck_l1506_150693


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1506_150666

theorem smallest_integer_with_remainder (k : ℕ) : k = 275 ↔ 
  k > 1 ∧ 
  k % 13 = 2 ∧ 
  k % 7 = 2 ∧ 
  k % 3 = 2 ∧ 
  ∀ m : ℕ, m > 1 → m % 13 = 2 → m % 7 = 2 → m % 3 = 2 → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1506_150666


namespace NUMINAMATH_CALUDE_sum_difference_with_triangular_problem_solution_l1506_150641

def even_sum (n : ℕ) : ℕ := n * (n + 1)

def odd_sum (n : ℕ) : ℕ := n * n

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sum_difference_with_triangular (n : ℕ) :
  even_sum n - odd_sum n + triangular_sum n = n * (n * n + 3) / 3 :=
by sorry

theorem problem_solution : 
  even_sum 1500 - odd_sum 1500 + triangular_sum 1500 = 563628000 :=
by sorry

end NUMINAMATH_CALUDE_sum_difference_with_triangular_problem_solution_l1506_150641


namespace NUMINAMATH_CALUDE_equation_solution_l1506_150604

theorem equation_solution (a b : ℝ) :
  (∀ x y : ℝ, y = a + b / x) →
  (3 = a + b / 2) →
  (-1 = a + b / (-4)) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1506_150604


namespace NUMINAMATH_CALUDE_haunted_mansion_scenarios_l1506_150622

theorem haunted_mansion_scenarios (windows : ℕ) (rooms : ℕ) : windows = 8 → rooms = 3 → windows * (windows - 1) * rooms = 168 := by
  sorry

end NUMINAMATH_CALUDE_haunted_mansion_scenarios_l1506_150622


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_distinct_roots_l1506_150638

/-- The quadratic equation (a-3)x^2 - 4x - 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop :=
  (a - 3) * x^2 - 4 * x - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ :=
  16 - 4 * (a - 3) * (-1)

theorem quadratic_equation_roots (a : ℝ) :
  (∃ x : ℝ, quadratic_equation a x ∧
    (∀ y : ℝ, quadratic_equation a y → y = x)) →
  a = -1 ∧ (∀ x : ℝ, quadratic_equation a x → x = -1/2) :=
sorry

theorem quadratic_equation_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x ∧ quadratic_equation a y) →
  (a > -1 ∧ a ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_distinct_roots_l1506_150638


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l1506_150643

/-- Given two points P and Q in a 2D plane, where Q is symmetric to P with respect to the x-axis,
    this theorem proves that the x-coordinate of Q is the same as P, and the y-coordinate of Q
    is the negative of P's y-coordinate. -/
theorem symmetric_point_x_axis 
  (P Q : ℝ × ℝ) 
  (h_symmetric : Q.1 = P.1 ∧ Q.2 = -P.2) 
  (h_P : P = (-3, 1)) : 
  Q = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l1506_150643


namespace NUMINAMATH_CALUDE_james_beats_record_l1506_150637

/-- The number of points James beat the old record by -/
def points_above_record (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (old_record : ℕ) : ℕ :=
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + 
                      two_point_conversions * 2
  total_points - old_record

/-- Theorem stating that James beat the old record by 72 points -/
theorem james_beats_record : points_above_record 4 6 15 6 300 = 72 := by
  sorry

end NUMINAMATH_CALUDE_james_beats_record_l1506_150637


namespace NUMINAMATH_CALUDE_acorn_problem_l1506_150655

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
def total_acorns (shawna sheila danny : ℕ) : ℕ := shawna + sheila + danny

/-- Theorem stating the total number of acorns given the problem conditions -/
theorem acorn_problem (shawna sheila danny : ℕ) 
  (h1 : shawna = 7)
  (h2 : sheila = 5 * shawna)
  (h3 : danny = sheila + 3) :
  total_acorns shawna sheila danny = 80 := by
  sorry

end NUMINAMATH_CALUDE_acorn_problem_l1506_150655


namespace NUMINAMATH_CALUDE_square_orientation_after_1011_transformations_l1506_150691

/-- Represents the possible orientations of the square -/
inductive SquareOrientation
  | ABCD
  | DABC
  | BADC
  | DCBA

/-- Applies the 90-degree clockwise rotation -/
def rotate90 (s : SquareOrientation) : SquareOrientation :=
  match s with
  | SquareOrientation.ABCD => SquareOrientation.DABC
  | SquareOrientation.DABC => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.DCBA
  | SquareOrientation.DCBA => SquareOrientation.ABCD

/-- Applies the 180-degree rotation -/
def rotate180 (s : SquareOrientation) : SquareOrientation :=
  match s with
  | SquareOrientation.ABCD => SquareOrientation.BADC
  | SquareOrientation.DABC => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.ABCD
  | SquareOrientation.DCBA => SquareOrientation.ABCD

/-- Applies both rotations in sequence -/
def applyTransformations (s : SquareOrientation) : SquareOrientation :=
  rotate180 (rotate90 s)

/-- Applies the transformations n times -/
def applyNTimes (s : SquareOrientation) (n : Nat) : SquareOrientation :=
  match n with
  | 0 => s
  | n + 1 => applyTransformations (applyNTimes s n)

theorem square_orientation_after_1011_transformations :
  applyNTimes SquareOrientation.ABCD 1011 = SquareOrientation.DCBA := by
  sorry


end NUMINAMATH_CALUDE_square_orientation_after_1011_transformations_l1506_150691


namespace NUMINAMATH_CALUDE_second_parallel_line_length_l1506_150624

/-- Given a triangle with base length 18 and three parallel lines dividing it into four equal areas,
    the length of the second parallel line from the base is 9√2. -/
theorem second_parallel_line_length (base : ℝ) (l₁ l₂ l₃ : ℝ) :
  base = 18 →
  l₁ < l₂ ∧ l₂ < l₃ →
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ base → (x * l₁) = (x * l₂) ∧ (x * l₂) = (x * l₃) ∧ (x * l₃) = (x * base / 4)) →
  l₂ = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_second_parallel_line_length_l1506_150624


namespace NUMINAMATH_CALUDE_angle_expression_equality_l1506_150609

theorem angle_expression_equality (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (3 * Real.pi / 2 + θ) + Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_equality_l1506_150609


namespace NUMINAMATH_CALUDE_speed_conversion_l1506_150684

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The initial speed in meters per second -/
def initial_speed : ℝ := 5

/-- Theorem: 5 mps is equal to 18 kmph -/
theorem speed_conversion : initial_speed * mps_to_kmph = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l1506_150684


namespace NUMINAMATH_CALUDE_circle_equation_equivalence_l1506_150618

theorem circle_equation_equivalence :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_equivalence_l1506_150618


namespace NUMINAMATH_CALUDE_morse_code_symbols_l1506_150615

/-- The number of possible symbols (dot, dash, space) -/
def num_symbols : ℕ := 3

/-- The maximum length of a sequence -/
def max_length : ℕ := 3

/-- Calculates the number of distinct sequences for a given length -/
def sequences_of_length (n : ℕ) : ℕ := num_symbols ^ n

/-- The total number of distinct symbols that can be represented -/
def total_distinct_symbols : ℕ :=
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3)

/-- Theorem: The total number of distinct symbols that can be represented is 39 -/
theorem morse_code_symbols : total_distinct_symbols = 39 := by
  sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l1506_150615


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1506_150646

/-- A quadratic function is a function of the form f(x) = ax² + bx + c, where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (h_quad : QuadraticFunction f)
  (h_f_2 : f 2 = -1)
  (h_f_neg1 : f (-1) = -1)
  (h_max : ∃ x_max, ∀ x, f x ≤ f x_max ∧ f x_max = 8) :
  ∀ x, f x = -4 * x^2 + 4 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1506_150646


namespace NUMINAMATH_CALUDE_tiling_problem_l1506_150687

/-- Number of ways to tile a 3 × n rectangle -/
def tiling_ways (n : ℕ) : ℚ :=
  (2^(n+2) + (-1)^(n+1)) / 3

/-- Proof of the tiling problem -/
theorem tiling_problem (n : ℕ) (h : n > 3) :
  tiling_ways n = (2^(n+2) + (-1)^(n+1)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_tiling_problem_l1506_150687


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1506_150628

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a = 2 →
  c = 3 →
  B = 2 * π / 3 →  -- 120° in radians
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →  -- Law of Cosines
  b = Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1506_150628


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1506_150649

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Define the point the hyperbola passes through
def passes_through : Prop := hyperbola 3 (-2 * Real.sqrt 3)

-- Define the intersection line
def intersection_line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 3)

-- State the theorem
theorem hyperbola_properties :
  (∀ x y, asymptotes x y → hyperbola x y) ∧
  passes_through ∧
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    intersection_line A.1 A.2 ∧
    intersection_line B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1506_150649


namespace NUMINAMATH_CALUDE_opposite_definition_opposite_of_eight_l1506_150651

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of a number added to the original number equals zero -/
theorem opposite_definition (x : ℝ) : x + opposite x = 0 := by sorry

/-- The opposite of 8 is -8 -/
theorem opposite_of_eight : opposite 8 = -8 := by sorry

end NUMINAMATH_CALUDE_opposite_definition_opposite_of_eight_l1506_150651


namespace NUMINAMATH_CALUDE_x_less_than_two_necessary_not_sufficient_l1506_150644

theorem x_less_than_two_necessary_not_sufficient :
  (∀ x : ℝ, |x - 1| < 1 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ |x - 1| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_two_necessary_not_sufficient_l1506_150644


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1506_150672

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ x y : ℝ, x^2 = 4*y ∧ y = k*x - 2 ∧ k = (1/2)*x) → k^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1506_150672


namespace NUMINAMATH_CALUDE_grandchildren_gender_probability_l1506_150699

theorem grandchildren_gender_probability :
  let n : ℕ := 12  -- total number of grandchildren
  let p : ℚ := 1/2  -- probability of a grandchild being male (or female)
  let equal_prob := (n.choose (n/2)) / 2^n  -- probability of equal number of grandsons and granddaughters
  1 - equal_prob = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_grandchildren_gender_probability_l1506_150699


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_is_14pi_l1506_150629

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a right triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  side_ab : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 3
  side_bc : Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) = 4
  side_ca : Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2) = 5
  right_angle : (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0

/-- Checks if two circles are externally tangent -/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = c1.radius + c2.radius

/-- Theorem: The sum of the areas of three mutually externally tangent circles
    centered at the vertices of a 3-4-5 right triangle is 14π -/
theorem sum_of_circle_areas_is_14pi (t : RightTriangle) 
    (c1 : Circle) (c2 : Circle) (c3 : Circle)
    (h1 : c1.center = t.a) (h2 : c2.center = t.b) (h3 : c3.center = t.c)
    (h4 : areExternallyTangent c1 c2)
    (h5 : areExternallyTangent c2 c3)
    (h6 : areExternallyTangent c3 c1) :
    π * (c1.radius^2 + c2.radius^2 + c3.radius^2) = 14 * π := by
  sorry


end NUMINAMATH_CALUDE_sum_of_circle_areas_is_14pi_l1506_150629


namespace NUMINAMATH_CALUDE_left_focus_coordinates_l1506_150680

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 2 = 1

/-- The left focus of the hyperbola -/
def left_focus : ℝ × ℝ := (-2, 0)

/-- Theorem: The coordinates of the left focus of the given hyperbola are (-2,0) -/
theorem left_focus_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → left_focus = (-2, 0) := by
  sorry

end NUMINAMATH_CALUDE_left_focus_coordinates_l1506_150680


namespace NUMINAMATH_CALUDE_basketball_team_points_l1506_150645

theorem basketball_team_points (x : ℚ) (z : ℕ) : 
  (1 / 3 : ℚ) * x + (3 / 8 : ℚ) * x + 18 + z = x →
  z ≤ 27 →
  z = 21 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_points_l1506_150645


namespace NUMINAMATH_CALUDE_four_weighings_sufficient_l1506_150606

/-- Represents the weight of a coin in grams -/
inductive CoinWeight
  | One
  | Two
  | Three
  | Four

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing action -/
def Weighing := (List CoinWeight) → (List CoinWeight) → WeighingResult

/-- The set of four coins with weights 1, 2, 3, and 4 grams -/
def CoinSet : Set CoinWeight := {CoinWeight.One, CoinWeight.Two, CoinWeight.Three, CoinWeight.Four}

/-- A strategy is a sequence of weighings -/
def Strategy := List Weighing

/-- Checks if a strategy can identify all coins uniquely -/
def canIdentifyAllCoins (s : Strategy) (coins : Set CoinWeight) : Prop := sorry

/-- Main theorem: There exists a strategy with at most 4 weighings that can identify all coins -/
theorem four_weighings_sufficient :
  ∃ (s : Strategy), s.length ≤ 4 ∧ canIdentifyAllCoins s CoinSet := by sorry

end NUMINAMATH_CALUDE_four_weighings_sufficient_l1506_150606


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_pairs_l1506_150623

def S (r : ℕ) (x y z : ℝ) : ℝ := x^r + y^r + z^r

theorem equation_holds_iff_specific_pairs (m n : ℕ) (x y z : ℝ) 
  (h : x + y + z = 0) :
  (∀ (x y z : ℝ), x + y + z = 0 → 
    S (m + n) x y z / (m + n : ℝ) = (S m x y z / m) * (S n x y z / n)) ↔ 
  ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2)) :=
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_pairs_l1506_150623


namespace NUMINAMATH_CALUDE_book_selling_loss_l1506_150635

/-- Calculates the loss from buying and selling books -/
theorem book_selling_loss 
  (books_per_month : ℕ) 
  (book_cost : ℕ) 
  (months : ℕ) 
  (selling_price : ℕ) : 
  books_per_month * months * book_cost - selling_price = 220 :=
by
  sorry

#check book_selling_loss 3 20 12 500

end NUMINAMATH_CALUDE_book_selling_loss_l1506_150635


namespace NUMINAMATH_CALUDE_impossible_parking_space_l1506_150668

theorem impossible_parking_space (L W : ℝ) : 
  L = 99 ∧ L + 2 * W = 37 → False :=
by sorry

end NUMINAMATH_CALUDE_impossible_parking_space_l1506_150668


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1506_150695

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s := 2^(n+1) + 2
  let r := 3^s - 2*s + 1
  r = 387420454 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1506_150695


namespace NUMINAMATH_CALUDE_equation_solutions_l1506_150608

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (-3 + Real.sqrt 13) / 2 ∧ x2 = (-3 - Real.sqrt 13) / 2 ∧
    x1^2 + 3*x1 - 1 = 0 ∧ x2^2 + 3*x2 - 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -2 ∧ x2 = -1 ∧
    (x1 + 2)^2 = x1 + 2 ∧ (x2 + 2)^2 = x2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1506_150608


namespace NUMINAMATH_CALUDE_isosceles_triangle_30_angle_diff_l1506_150634

-- Define an isosceles triangle with one angle of 30 degrees
structure IsoscelesTriangle30 where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  isosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)
  has_30 : angles 0 = 30 ∨ angles 1 = 30 ∨ angles 2 = 30

-- State the theorem
theorem isosceles_triangle_30_angle_diff 
  (t : IsoscelesTriangle30) : 
  ∃ (i j : Fin 3), i ≠ j ∧ (t.angles i - t.angles j = 90 ∨ t.angles i - t.angles j = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_30_angle_diff_l1506_150634


namespace NUMINAMATH_CALUDE_function_inequality_l1506_150653

open Real

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x : ℝ, f x > deriv f x) : 
  (ℯ^2016 * f (-2016) > f 0) ∧ (f 2016 < ℯ^2016 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1506_150653


namespace NUMINAMATH_CALUDE_largest_solution_value_l1506_150688

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log 10 / Real.log (x^2) + Real.log 10 / Real.log (x^4) + Real.log 10 / Real.log (9*x^5) = 0

-- Define the set of solutions
def solution_set := { x : ℝ | equation x ∧ x > 0 }

-- State the theorem
theorem largest_solution_value :
  ∃ (x : ℝ), x ∈ solution_set ∧ 
  (∀ (y : ℝ), y ∈ solution_set → y ≤ x) ∧
  (1 / x^18 = 9^93) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_value_l1506_150688


namespace NUMINAMATH_CALUDE_marla_horse_purchase_l1506_150659

/-- The number of bottle caps equivalent to one lizard -/
def bottlecaps_per_lizard : ℕ := 8

/-- The number of lizards equivalent to 5 gallons of water -/
def lizards_per_five_gallons : ℕ := 3

/-- The number of gallons of water equivalent to one horse -/
def gallons_per_horse : ℕ := 80

/-- The number of bottle caps Marla can scavenge per day -/
def daily_scavenge : ℕ := 20

/-- The number of bottle caps Marla pays per night for food and shelter -/
def daily_expense : ℕ := 4

/-- The number of days it takes Marla to collect enough bottle caps to buy a horse -/
def days_to_buy_horse : ℕ := 24

theorem marla_horse_purchase :
  days_to_buy_horse * (daily_scavenge - daily_expense) =
  (gallons_per_horse * lizards_per_five_gallons * bottlecaps_per_lizard) / 5 :=
by sorry

end NUMINAMATH_CALUDE_marla_horse_purchase_l1506_150659


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1506_150612

theorem ferris_wheel_capacity 
  (total_seats : ℕ) 
  (people_per_seat : ℕ) 
  (broken_seats : ℕ) 
  (h1 : total_seats = 18) 
  (h2 : people_per_seat = 15) 
  (h3 : broken_seats = 10) :
  (total_seats - broken_seats) * people_per_seat = 120 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1506_150612


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1506_150636

/-- The area of a quadrilateral formed by four squares arranged in a specific manner -/
theorem quadrilateral_area (s₁ s₂ s₃ s₄ : ℝ) (h₁ : s₁ = 1) (h₂ : s₂ = 3) (h₃ : s₃ = 5) (h₄ : s₄ = 7) :
  let total_length := s₁ + s₂ + s₃ + s₄
  let height_ratio := s₄ / total_length
  let height₂ := s₂ * height_ratio
  let height₃ := s₃ * height_ratio
  let quadrilateral_height := s₃ - s₂
  (height₂ + height₃) * quadrilateral_height / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1506_150636


namespace NUMINAMATH_CALUDE_percentage_problem_l1506_150656

theorem percentage_problem (x : ℝ) : 
  (0.12 * 160) - (x / 100 * 80) = 11.2 ↔ x = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1506_150656


namespace NUMINAMATH_CALUDE_sandwiches_theorem_l1506_150639

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth's brother ate -/
def brother_ate : ℕ := 2

/-- The number of sandwiches Ruth's first cousin ate -/
def first_cousin_ate : ℕ := 2

/-- The number of sandwiches each of Ruth's other two cousins ate -/
def other_cousins_ate_each : ℕ := 1

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- Theorem stating that the total number of sandwiches Ruth prepared
    is equal to the sum of sandwiches eaten by everyone and those left -/
theorem sandwiches_theorem :
  total_sandwiches = ruth_ate + brother_ate + first_cousin_ate +
    (2 * other_cousins_ate_each) + sandwiches_left :=
by
  sorry

end NUMINAMATH_CALUDE_sandwiches_theorem_l1506_150639


namespace NUMINAMATH_CALUDE_person_A_age_l1506_150674

theorem person_A_age (current_age_A current_age_B past_age_A past_age_B years_ago : ℕ) : 
  current_age_A + current_age_B = 70 →
  current_age_A - years_ago = current_age_B →
  past_age_B = past_age_A / 2 →
  past_age_A = current_age_A →
  past_age_B = current_age_B - years_ago →
  current_age_A = 42 := by
  sorry

end NUMINAMATH_CALUDE_person_A_age_l1506_150674


namespace NUMINAMATH_CALUDE_toucan_count_l1506_150660

/-- Given that there are initially 2 toucans on a tree limb and 1 more toucan joins them,
    prove that the total number of toucans is 3. -/
theorem toucan_count (initial : ℕ) (joined : ℕ) (h1 : initial = 2) (h2 : joined = 1) :
  initial + joined = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l1506_150660


namespace NUMINAMATH_CALUDE_A_value_l1506_150621

noncomputable def A (x y : ℝ) : ℝ :=
  (Real.sqrt (x^3 + 2*x^2*y) + Real.sqrt (x^4 + 2 - x^3) - (x^(3/2) + x^2)) /
  (Real.sqrt (2*(x + y - Real.sqrt (x^2 + 2*x*y))) * (x^(2/3) - x^(5/6) + x))

theorem A_value (x y : ℝ) (hx : x > 0) :
  (y > 0 → A x y = x^(1/3) + x^(1/2)) ∧
  (-x/2 ≤ y ∧ y < 0 → A x y = -(x^(1/3) + x^(1/2))) :=
by sorry

end NUMINAMATH_CALUDE_A_value_l1506_150621


namespace NUMINAMATH_CALUDE_probability_x_gt_5y_l1506_150611

/-- The probability of selecting a point (x,y) from a rectangle with vertices
    (0,0), (2020,0), (2020,2021), and (0,2021) such that x > 5y is 101/1011. -/
theorem probability_x_gt_5y : 
  let rectangle_area := 2020 * 2021
  let triangle_area := (1 / 2) * 2020 * 404
  triangle_area / rectangle_area = 101 / 1011 := by
sorry

end NUMINAMATH_CALUDE_probability_x_gt_5y_l1506_150611


namespace NUMINAMATH_CALUDE_paige_score_l1506_150671

/-- Given a dodgeball team with the following properties:
  * The team has 5 players
  * The team scored a total of 41 points
  * 4 players scored 6 points each
  Prove that the remaining player (Paige) scored 17 points. -/
theorem paige_score (team_size : ℕ) (total_score : ℕ) (other_player_score : ℕ) :
  team_size = 5 →
  total_score = 41 →
  other_player_score = 6 →
  total_score - (team_size - 1) * other_player_score = 17 := by
  sorry


end NUMINAMATH_CALUDE_paige_score_l1506_150671


namespace NUMINAMATH_CALUDE_william_final_napkins_l1506_150620

def napkin_problem (initial_napkins : ℕ) (olivia_napkins : ℕ) : ℕ :=
  let amelia_napkins := 2 * olivia_napkins
  let charlie_napkins := amelia_napkins / 2
  let georgia_napkins := 3 * charlie_napkins
  initial_napkins + olivia_napkins + amelia_napkins + charlie_napkins + georgia_napkins

theorem william_final_napkins :
  napkin_problem 15 10 = 85 := by
  sorry

end NUMINAMATH_CALUDE_william_final_napkins_l1506_150620


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l1506_150681

def original_houses : ℕ := 20817
def new_houses : ℕ := 97741

theorem lincoln_county_houses : original_houses + new_houses = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l1506_150681


namespace NUMINAMATH_CALUDE_pies_sold_in_week_l1506_150692

/-- The number of pies sold daily -/
def daily_sales : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def weekly_sales : ℕ := daily_sales * days_in_week

theorem pies_sold_in_week : weekly_sales = 56 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_in_week_l1506_150692


namespace NUMINAMATH_CALUDE_greatest_b_value_l1506_150682

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ 9) ∧ 
  (9^2 - 14*9 + 45 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l1506_150682


namespace NUMINAMATH_CALUDE_paper_completion_days_l1506_150648

theorem paper_completion_days (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) : 
  total_pages = 81 → pages_per_day = 27 → days * pages_per_day = total_pages → days = 3 := by
  sorry

end NUMINAMATH_CALUDE_paper_completion_days_l1506_150648


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1506_150602

theorem geometric_sequence_problem (a : ℝ) :
  a > 0 ∧
  (∃ (r : ℝ), 210 * r = a ∧ a * r = 63 / 40) →
  a = 18.1875 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1506_150602


namespace NUMINAMATH_CALUDE_common_prime_root_quadratics_l1506_150689

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * p + b = 0 ∧ 
    (p : ℤ)^2 + b * p + 1100 = 0) → 
  a = 274 ∨ a = 40 := by
sorry

end NUMINAMATH_CALUDE_common_prime_root_quadratics_l1506_150689


namespace NUMINAMATH_CALUDE_happy_point_properties_l1506_150679

/-- A point (m, n+2) is a "happy point" if 2m = 8 + n --/
def is_happy_point (m n : ℝ) : Prop := 2 * m = 8 + n

/-- The point B(4,5) --/
def B : ℝ × ℝ := (4, 5)

/-- The point M(a, a-1) --/
def M (a : ℝ) : ℝ × ℝ := (a, a - 1)

theorem happy_point_properties :
  (¬ is_happy_point B.1 (B.2 - 2)) ∧
  (∀ a : ℝ, is_happy_point (M a).1 ((M a).2 - 2) → a > 0 ∧ a - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_happy_point_properties_l1506_150679


namespace NUMINAMATH_CALUDE_subset_intersection_theorem_l1506_150658

theorem subset_intersection_theorem (α : ℝ) 
  (h_pos : α > 0) (h_bound : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ+) (S T : Finset (Finset (Fin n))),
    p > α * 2^(n : ℝ) ∧
    S.card = p ∧
    T.card = p ∧
    (∀ s ∈ S, ∀ t ∈ T, (s ∩ t).Nonempty) :=
sorry

end NUMINAMATH_CALUDE_subset_intersection_theorem_l1506_150658


namespace NUMINAMATH_CALUDE_roi_difference_emma_briana_l1506_150650

/-- Calculates the difference in return-on-investment between two investors after a given time period. -/
def roi_difference (emma_investment briana_investment : ℝ) 
                   (emma_yield_rate briana_yield_rate : ℝ) 
                   (years : ℕ) : ℝ :=
  (briana_investment * briana_yield_rate * years) - (emma_investment * emma_yield_rate * years)

/-- Theorem stating the difference in return-on-investment between Briana and Emma after 2 years. -/
theorem roi_difference_emma_briana : 
  roi_difference 300 500 0.15 0.10 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_roi_difference_emma_briana_l1506_150650


namespace NUMINAMATH_CALUDE_area_of_circle_with_diameter_6_l1506_150631

-- Define the circle
def circle_diameter : ℝ := 6

-- Theorem statement
theorem area_of_circle_with_diameter_6 :
  (π * (circle_diameter / 2)^2) = 9 * π := by sorry

end NUMINAMATH_CALUDE_area_of_circle_with_diameter_6_l1506_150631


namespace NUMINAMATH_CALUDE_cos_product_equation_l1506_150603

theorem cos_product_equation (α : Real) (h : Real.tan α = 2) :
  Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 + α) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_equation_l1506_150603


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l1506_150614

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let radius := area / s
  2 * radius = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l1506_150614


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l1506_150605

/-- Represents the repeating decimal 0.090909... -/
def a : ℚ := 1 / 11

/-- Represents the repeating decimal 0.777777... -/
def b : ℚ := 7 / 9

/-- The product of the repeating decimals 0.090909... and 0.777777... equals 7/99 -/
theorem product_of_repeating_decimals : a * b = 7 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l1506_150605


namespace NUMINAMATH_CALUDE_binary_to_base5_l1506_150600

-- Define the binary number
def binary_num : List Bool := [true, true, false, true, false, true, true]

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : ℕ :=
  bin.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert decimal to base 5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_to_base5 :
  decimal_to_base5 (binary_to_decimal binary_num) = [4, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_binary_to_base5_l1506_150600


namespace NUMINAMATH_CALUDE_no_x3_term_condition_l1506_150642

/-- The coefficient of x^3 in the expansion of ((x+a)^2(2x-1/x)^5) -/
def coeff_x3 (a : ℝ) : ℝ := 80 - 80 * a^2

/-- The theorem stating that the value of 'a' for which the expansion of 
    ((x+a)^2(2x-1/x)^5) does not contain the x^3 term is ±1 -/
theorem no_x3_term_condition (a : ℝ) : 
  coeff_x3 a = 0 ↔ a = 1 ∨ a = -1 := by
  sorry

#check no_x3_term_condition

end NUMINAMATH_CALUDE_no_x3_term_condition_l1506_150642


namespace NUMINAMATH_CALUDE_debt_amount_l1506_150601

/-- Represents the savings of the three girls and the debt amount -/
structure Savings where
  lulu : ℕ
  nora : ℕ
  tamara : ℕ
  debt : ℕ

/-- Theorem stating the debt amount given the conditions -/
theorem debt_amount (s : Savings) :
  s.lulu = 6 ∧
  s.nora = 5 * s.lulu ∧
  s.nora = 3 * s.tamara ∧
  s.lulu + s.nora + s.tamara = s.debt + 6 →
  s.debt = 40 := by
  sorry


end NUMINAMATH_CALUDE_debt_amount_l1506_150601


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l1506_150661

/-- The curve function f(x) = x^4 + x -/
def f (x : ℝ) : ℝ := x^4 + x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * x^3 + 1

theorem tangent_point_coordinates :
  ∃ (x y : ℝ), f y = f x ∧ f' x = -3 → x = -1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l1506_150661


namespace NUMINAMATH_CALUDE_negative_rational_and_fraction_l1506_150676

-- Define the number -0.3
def num : ℚ := -3/10

-- Theorem statement
theorem negative_rational_and_fraction (n : ℚ) (h : n = -3/10) :
  n < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b :=
sorry

end NUMINAMATH_CALUDE_negative_rational_and_fraction_l1506_150676


namespace NUMINAMATH_CALUDE_race_head_start_l1506_150686

theorem race_head_start (L : ℝ) (va vb : ℝ) (h : va = 20/13 * vb) :
  let H := (L - H) / vb + 0.6 * L / vb - L / va
  H = 19/20 * L := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l1506_150686


namespace NUMINAMATH_CALUDE_b_time_approx_l1506_150663

/-- The time it takes for A to complete the work alone -/
def a_time : ℝ := 20

/-- The time it takes for A and B to complete the work together -/
def ab_time : ℝ := 12.727272727272728

/-- The time it takes for B to complete the work alone -/
noncomputable def b_time : ℝ := (a_time * ab_time) / (a_time - ab_time)

/-- Theorem stating that B can complete the work in approximately 34.90909090909091 days -/
theorem b_time_approx : 
  ∃ ε > 0, |b_time - 34.90909090909091| < ε :=
sorry

end NUMINAMATH_CALUDE_b_time_approx_l1506_150663


namespace NUMINAMATH_CALUDE_geometric_sequence_bounded_ratio_counterexample_l1506_150619

theorem geometric_sequence_bounded_ratio_counterexample :
  ¬ (∀ (a₁ : ℝ) (q : ℝ) (a : ℝ),
    (a₁ > 0 ∧ q > 0) →
    (∀ n : ℕ, a₁ * q^(n - 1) < a) →
    (q > 0 ∧ q < 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_bounded_ratio_counterexample_l1506_150619


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l1506_150613

/-- Represents a cube with given side length -/
structure Cube where
  side : ℝ
  side_pos : side > 0

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def original_cube : Cube := ⟨4, by norm_num⟩

/-- Represents the corner cube to be removed -/
def corner_cube : Cube := ⟨2, by norm_num⟩

/-- Number of corners in a cube -/
def num_corners : ℕ := 8

/-- Theorem stating that the surface area remains unchanged after removing corner cubes -/
theorem surface_area_unchanged : 
  surface_area original_cube = surface_area original_cube := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l1506_150613


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l1506_150617

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def share_digit (a b : ℕ) : Prop :=
  (a / 10 = b / 10) ∨ (a % 10 = b % 10)

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem two_digit_numbers_problem (a b : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧
  a = b + 14 ∧
  share_digit a b ∧
  sum_of_digits a = 2 * sum_of_digits b →
  ((a = 37 ∧ b = 23) ∨ (a = 31 ∧ b = 17)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l1506_150617


namespace NUMINAMATH_CALUDE_max_areas_formula_max_areas_for_n_3_l1506_150647

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii_count : ℕ
  secant_lines : ℕ
  h_positive : n > 0
  h_radii : radii_count = 2 * n
  h_secants : secant_lines = 2

/-- Calculates the maximum number of non-overlapping areas in a divided disk -/
def max_areas (d : DividedDisk) : ℕ :=
  4 * d.n + 4

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_formula (d : DividedDisk) :
  max_areas d = 4 * d.n + 4 :=
by sorry

/-- Specific case for n = 3 -/
theorem max_areas_for_n_3 :
  ∃ (d : DividedDisk), d.n = 3 ∧ max_areas d = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_areas_formula_max_areas_for_n_3_l1506_150647


namespace NUMINAMATH_CALUDE_bankers_gain_example_l1506_150664

/-- Calculates the banker's gain given present worth, interest rate, and time period. -/
def bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  present_worth * (1 + interest_rate) ^ years - present_worth

/-- Theorem stating that the banker's gain is 126 given the specific conditions. -/
theorem bankers_gain_example : bankers_gain 600 0.1 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_example_l1506_150664


namespace NUMINAMATH_CALUDE_common_tangent_sum_l1506_150640

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 52/5

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 25/10

/-- Common tangent line L -/
def L (a b c x y : ℝ) : Prop := a*x + b*y = c

/-- Theorem stating the sum of a, b, and c for the common tangent line -/
theorem common_tangent_sum (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧ L a b c x₁ y₁ ∧ L a b c x₂ y₂) →
  (∃ (k : ℚ), a = k * b) →
  Nat.gcd a (Nat.gcd b c) = 1 →
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l1506_150640


namespace NUMINAMATH_CALUDE_a_range_for_two_positive_zeros_l1506_150627

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

/-- The condition for f to have two positive zeros -/
def has_two_positive_zeros (a : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- Theorem stating the range of a for f to have two positive zeros -/
theorem a_range_for_two_positive_zeros :
  ∀ a : ℝ, has_two_positive_zeros a ↔ a > 3 :=
sorry

end NUMINAMATH_CALUDE_a_range_for_two_positive_zeros_l1506_150627


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1506_150669

theorem quadratic_solution_difference_squared : 
  ∀ a b : ℝ, (5 * a^2 - 6 * a - 55 = 0) → 
             (5 * b^2 - 6 * b - 55 = 0) → 
             (a ≠ b) →
             (a - b)^2 = 1296 / 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1506_150669


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1506_150697

theorem baseball_card_value_decrease : 
  ∀ (initial_value : ℝ), initial_value > 0 →
  let first_year_value := initial_value * (1 - 0.5)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.55 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1506_150697


namespace NUMINAMATH_CALUDE_different_color_probability_l1506_150694

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def yellow_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem different_color_probability :
  (Nat.choose red_balls 1 * Nat.choose yellow_balls 1) / Nat.choose total_balls drawn_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l1506_150694


namespace NUMINAMATH_CALUDE_janes_number_l1506_150677

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of positive divisors of a natural number -/
def sum_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of prime divisors of a natural number -/
def sum_prime_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is uniquely determined by its sum of divisors -/
def is_unique_by_sum_divisors (n : ℕ) : Prop := 
  ∀ m : ℕ, sum_divisors m = sum_divisors n → m = n

/-- A function that checks if a number is uniquely determined by its sum of prime divisors -/
def is_unique_by_sum_prime_divisors (n : ℕ) : Prop := 
  ∀ m : ℕ, sum_prime_divisors m = sum_prime_divisors n → m = n

theorem janes_number : 
  ∃! n : ℕ, 
    500 < n ∧ 
    n < 1000 ∧ 
    num_divisors n = 20 ∧ 
    ¬ is_unique_by_sum_divisors n ∧ 
    ¬ is_unique_by_sum_prime_divisors n ∧ 
    n = 880 := by sorry

end NUMINAMATH_CALUDE_janes_number_l1506_150677


namespace NUMINAMATH_CALUDE_all_crop_to_diagonal_l1506_150626

/-- A symmetric kite-shaped field -/
structure KiteField where
  long_side : ℝ
  short_side : ℝ
  angle : ℝ
  long_side_positive : 0 < long_side
  short_side_positive : 0 < short_side
  angle_range : 0 < angle ∧ angle < π

/-- The fraction of the field area closer to the longer diagonal -/
def fraction_closer_to_diagonal (k : KiteField) : ℝ :=
  1 -- Definition, not proof

/-- The theorem statement -/
theorem all_crop_to_diagonal (k : KiteField) 
  (h1 : k.long_side = 100)
  (h2 : k.short_side = 70)
  (h3 : k.angle = 2 * π / 3) :
  fraction_closer_to_diagonal k = 1 := by
  sorry

end NUMINAMATH_CALUDE_all_crop_to_diagonal_l1506_150626


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l1506_150652

theorem sum_of_squares_theorem (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 2)
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l1506_150652


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1506_150632

theorem smaller_root_of_quadratic (x : ℝ) :
  x^2 + 10*x - 24 = 0 → (x = -12 ∨ x = 2) ∧ -12 < 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1506_150632


namespace NUMINAMATH_CALUDE_asymptote_sum_l1506_150633

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) with integer coefficients A, B, C,
    if it has vertical asymptotes at x = -3, 0, and 4, then A + B + C = -13 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 4 → 
    x / (x^3 + A*x^2 + B*x + C) ≠ 0) →
  A + B + C = -13 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1506_150633


namespace NUMINAMATH_CALUDE_smallest_valid_tournament_l1506_150675

/-- A tournament is valid if for any two players, there exists a third player who beat both of them -/
def is_valid_tournament (k : ℕ) (tournament : Fin k → Fin k → Bool) : Prop :=
  k > 1 ∧
  (∀ i j, i ≠ j → tournament i j = !tournament j i) ∧
  (∀ i j, i ≠ j → ∃ m, m ≠ i ∧ m ≠ j ∧ tournament m i ∧ tournament m j)

/-- The smallest k for which a valid tournament exists is 7 -/
theorem smallest_valid_tournament : 
  (∃ k : ℕ, ∃ tournament : Fin k → Fin k → Bool, is_valid_tournament k tournament) ∧
  (∀ k : ℕ, k < 7 → ¬∃ tournament : Fin k → Fin k → Bool, is_valid_tournament k tournament) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_tournament_l1506_150675


namespace NUMINAMATH_CALUDE_factoring_expression_l1506_150625

theorem factoring_expression (y : ℝ) : 
  5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1506_150625


namespace NUMINAMATH_CALUDE_car_capacities_and_rental_plans_l1506_150683

/-- The capacity of a type A car in tons -/
def capacity_A : ℕ := 3

/-- The capacity of a type B car in tons -/
def capacity_B : ℕ := 4

/-- The total weight of goods to be transported -/
def total_weight : ℕ := 31

/-- A rental plan is a pair of natural numbers (a, b) where a is the number of type A cars and b is the number of type B cars -/
def RentalPlan := ℕ × ℕ

/-- The set of all valid rental plans -/
def valid_rental_plans : Set RentalPlan :=
  {plan | plan.1 * capacity_A + plan.2 * capacity_B = total_weight}

theorem car_capacities_and_rental_plans :
  (2 * capacity_A + capacity_B = 10) ∧
  (capacity_A + 2 * capacity_B = 11) ∧
  (valid_rental_plans = {(1, 7), (5, 4), (9, 1)}) := by
  sorry


end NUMINAMATH_CALUDE_car_capacities_and_rental_plans_l1506_150683


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1506_150630

/-- Proves that each friend receives 24/7 pounds of chocolate given the initial conditions -/
theorem chocolate_distribution (total : ℚ) (initial_piles : ℕ) (friends : ℕ) : 
  total = 60 / 7 →
  initial_piles = 5 →
  friends = 2 →
  (total - (total / initial_piles)) / friends = 24 / 7 := by
sorry

#eval (60 / 7 : ℚ)
#eval (24 / 7 : ℚ)

end NUMINAMATH_CALUDE_chocolate_distribution_l1506_150630


namespace NUMINAMATH_CALUDE_slope_angles_of_line_l_l1506_150667

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- Line l in parametric form -/
def line_l (x y t α : ℝ) : Prop := x = 1 + t * Real.cos α ∧ y = t * Real.sin α

/-- Intersection condition -/
def intersection_condition (t α : ℝ) : Prop := t^2 - 2*t*Real.cos α - 3 = 0

/-- Main theorem -/
theorem slope_angles_of_line_l (α : ℝ) :
  (∃ ρ θ x y t, curve_C ρ θ ∧ line_l x y t α ∧ intersection_condition t α) →
  α = π/4 ∨ α = 3*π/4 :=
sorry

end NUMINAMATH_CALUDE_slope_angles_of_line_l_l1506_150667


namespace NUMINAMATH_CALUDE_balance_theorem_l1506_150616

/-- The number of blue balls that balance one green ball -/
def green_to_blue : ℚ := 2

/-- The number of blue balls that balance one yellow ball -/
def yellow_to_blue : ℚ := 2.5

/-- The number of blue balls that balance one white ball -/
def white_to_blue : ℚ := 10/7

/-- The number of blue balls that balance 5 green, 4 yellow, and 3 white balls -/
def total_blue_balls : ℚ := 170/7

theorem balance_theorem :
  5 * green_to_blue + 4 * yellow_to_blue + 3 * white_to_blue = total_blue_balls :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l1506_150616


namespace NUMINAMATH_CALUDE_function_is_periodic_l1506_150665

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the parameters a and b
variable (a b : ℝ)

-- State the conditions
axiom cond1 : ∀ x, f x = f (2 * b - x)
axiom cond2 : ∀ x, f (a + x) = -f (a - x)
axiom cond3 : a ≠ b

-- State the theorem
theorem function_is_periodic : ∀ x, f x = f (x + 4 * (a - b)) := by sorry

end NUMINAMATH_CALUDE_function_is_periodic_l1506_150665
