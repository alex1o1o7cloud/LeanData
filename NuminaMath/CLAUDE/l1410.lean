import Mathlib

namespace NUMINAMATH_CALUDE_green_leaves_remaining_l1410_141033

theorem green_leaves_remaining (num_plants : ℕ) (initial_leaves : ℕ) (falling_fraction : ℚ) : 
  num_plants = 3 → 
  initial_leaves = 18 → 
  falling_fraction = 1/3 → 
  (num_plants * initial_leaves * (1 - falling_fraction) : ℚ) = 36 := by
sorry

end NUMINAMATH_CALUDE_green_leaves_remaining_l1410_141033


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l1410_141049

theorem max_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ x * y = 1000000 ∧ 
  ∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l1410_141049


namespace NUMINAMATH_CALUDE_largest_integer_in_range_l1410_141032

theorem largest_integer_in_range : ∃ (x : ℤ), 
  (1 / 4 : ℚ) < (x : ℚ) / 7 ∧ 
  (x : ℚ) / 7 < (2 / 3 : ℚ) ∧ 
  ∀ (y : ℤ), (1 / 4 : ℚ) < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < (2 / 3 : ℚ) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_range_l1410_141032


namespace NUMINAMATH_CALUDE_inner_square_probability_is_16_25_l1410_141099

/-- The size of the checkerboard -/
def board_size : ℕ := 10

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the board -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not touching the outer edge -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not touching the outer edge -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem inner_square_probability_is_16_25 : 
  inner_square_probability = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_probability_is_16_25_l1410_141099


namespace NUMINAMATH_CALUDE_xy_difference_squared_l1410_141048

theorem xy_difference_squared (x y b c : ℝ) 
  (h1 : x * y = c^2) 
  (h2 : 1 / x^2 + 1 / y^2 = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := by
sorry

end NUMINAMATH_CALUDE_xy_difference_squared_l1410_141048


namespace NUMINAMATH_CALUDE_max_weight_proof_l1410_141004

def max_weight_single_trip : ℕ := 8750

theorem max_weight_proof (crate_weight_min crate_weight_max : ℕ) 
  (weight_8_crates weight_12_crates : ℕ) :
  crate_weight_min = 150 →
  crate_weight_max = 250 →
  weight_8_crates ≤ 1300 →
  weight_12_crates ≤ 2100 →
  max_weight_single_trip = 8750 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_proof_l1410_141004


namespace NUMINAMATH_CALUDE_circle_area_through_points_l1410_141009

/-- The area of a circle with center P(5, -2) passing through point Q(-7, 6) is 208π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (5, -2)
  let Q : ℝ × ℝ := (-7, 6)
  let r := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  π * r^2 = 208 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l1410_141009


namespace NUMINAMATH_CALUDE_set_operations_l1410_141045

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {0, 1, 4}

-- Define set B
def B : Finset Nat := {0, 1, 3}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {0, 1}) ∧ (A ∪ B = {0, 1, 3, 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1410_141045


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l1410_141010

/-- Proves that the discount percentage is 15% given the regular price and discounted price for soda cans. -/
theorem soda_discount_percentage 
  (regular_price : ℝ) 
  (discounted_price : ℝ) 
  (can_count : ℕ) :
  regular_price = 0.30 →
  discounted_price = 18.36 →
  can_count = 72 →
  (1 - discounted_price / (regular_price * can_count)) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l1410_141010


namespace NUMINAMATH_CALUDE_triangle_prime_angles_l1410_141070

theorem triangle_prime_angles (a b c : ℕ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c →  -- All angles are prime
  a = 2 ∨ b = 2 ∨ c = 2 :=  -- One angle must be 2 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_prime_angles_l1410_141070


namespace NUMINAMATH_CALUDE_line_properties_l1410_141080

/-- A line in the 2D plane represented by the equation y = k(x-1) --/
structure Line where
  k : ℝ

/-- The point (1,0) in the 2D plane --/
def point : ℝ × ℝ := (1, 0)

/-- Checks if a given line passes through the point (1,0) --/
def passes_through_point (l : Line) : Prop :=
  0 = l.k * (point.1 - 1)

/-- Checks if a given line is not perpendicular to the x-axis --/
def not_perpendicular_to_x_axis (l : Line) : Prop :=
  l.k ≠ 0

/-- Theorem stating that all lines represented by y = k(x-1) pass through (1,0) and are not perpendicular to the x-axis --/
theorem line_properties (l : Line) : 
  passes_through_point l ∧ not_perpendicular_to_x_axis l :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1410_141080


namespace NUMINAMATH_CALUDE_sequence_properties_l1410_141023

/-- Arithmetic sequence with a₈ = 6 and a₁₀ = 0 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  30 - 3 * n

/-- Geometric sequence with a₁ = 1/2 and a₄ = 4 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 2)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  2^(n - 1) - 1/2

theorem sequence_properties :
  (arithmetic_sequence 8 = 6 ∧ arithmetic_sequence 10 = 0) ∧
  (geometric_sequence 1 = 1/2 ∧ geometric_sequence 4 = 4) ∧
  (∀ n : ℕ, geometric_sum n = (geometric_sequence 1) * (1 - (2^n)) / (1 - 2)) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l1410_141023


namespace NUMINAMATH_CALUDE_min_value_problem_l1410_141053

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log x + Real.log y = 2) :
  5 * x + 2 * y ≥ 20 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1410_141053


namespace NUMINAMATH_CALUDE_james_bed_purchase_l1410_141078

theorem james_bed_purchase (bed_frame_price : ℝ) (discount_rate : ℝ) : 
  bed_frame_price = 75 →
  discount_rate = 0.2 →
  let bed_price := 10 * bed_frame_price
  let total_before_discount := bed_frame_price + bed_price
  let discount_amount := discount_rate * total_before_discount
  let final_price := total_before_discount - discount_amount
  final_price = 660 := by sorry

end NUMINAMATH_CALUDE_james_bed_purchase_l1410_141078


namespace NUMINAMATH_CALUDE_inequalities_hold_l1410_141041

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (a^2 + b^2 ≥ 8) ∧ (1/(a*b) ≥ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1410_141041


namespace NUMINAMATH_CALUDE_emily_seeds_count_l1410_141098

/-- The number of seeds Emily planted in the big garden -/
def big_garden_seeds : ℕ := 29

/-- The number of small gardens Emily has -/
def num_small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden -/
def seeds_per_small_garden : ℕ := 4

/-- The total number of seeds Emily planted -/
def total_seeds : ℕ := big_garden_seeds + num_small_gardens * seeds_per_small_garden

theorem emily_seeds_count : total_seeds = 41 := by
  sorry

end NUMINAMATH_CALUDE_emily_seeds_count_l1410_141098


namespace NUMINAMATH_CALUDE_max_prism_pyramid_elements_l1410_141051

/-- A shape formed by fusing a rectangular prism with a pyramid on one of its faces -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_new_faces : ℕ
  pyramid_new_edges : ℕ
  pyramid_new_vertex : ℕ

/-- The sum of exterior faces, vertices, and edges of a PrismPyramid -/
def total_elements (pp : PrismPyramid) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_new_faces) +
  (pp.prism_edges + pp.pyramid_new_edges) +
  (pp.prism_vertices + pp.pyramid_new_vertex)

/-- Theorem stating that the maximum sum of exterior faces, vertices, and edges is 34 -/
theorem max_prism_pyramid_elements :
  ∃ (pp : PrismPyramid), 
    pp.prism_faces = 6 ∧
    pp.prism_edges = 12 ∧
    pp.prism_vertices = 8 ∧
    pp.pyramid_new_faces = 4 ∧
    pp.pyramid_new_edges = 4 ∧
    pp.pyramid_new_vertex = 1 ∧
    total_elements pp = 34 ∧
    ∀ (pp' : PrismPyramid), total_elements pp' ≤ 34 :=
  sorry

end NUMINAMATH_CALUDE_max_prism_pyramid_elements_l1410_141051


namespace NUMINAMATH_CALUDE_popsicle_sticks_count_l1410_141018

theorem popsicle_sticks_count 
  (num_groups : ℕ) 
  (sticks_per_group : ℕ) 
  (sticks_left : ℕ) 
  (h1 : num_groups = 10)
  (h2 : sticks_per_group = 15)
  (h3 : sticks_left = 20) :
  num_groups * sticks_per_group + sticks_left = 170 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_count_l1410_141018


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l1410_141027

-- Define the triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_construction_theorem 
  (A B P : ℝ) 
  (h1 : 0 < A) (h2 : 0 < B) (h3 : A + B < 180) (h4 : 0 < P) :
  ∃ (t : Triangle), 
    t.A = A ∧ 
    t.B = B ∧ 
    t.C = 180 - (A + B) ∧ 
    t.a + t.b + t.c = P :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l1410_141027


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1410_141087

/-- The perimeter of a rhombus given the lengths of its diagonals -/
theorem rhombus_perimeter (d1 d2 θ : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : 0 < θ ∧ θ < π) :
  ∃ (P : ℝ), P = 2 * Real.sqrt (d1^2 + d2^2) ∧ P > 0 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1410_141087


namespace NUMINAMATH_CALUDE_taxi_speed_is_60_l1410_141069

/-- The speed of the taxi in mph -/
def taxi_speed : ℝ := 60

/-- The speed of the bus in mph -/
def bus_speed : ℝ := taxi_speed - 30

/-- The time difference between the bus and taxi departure in hours -/
def time_difference : ℝ := 3

/-- The time it takes for the taxi to overtake the bus in hours -/
def overtake_time : ℝ := 3

theorem taxi_speed_is_60 :
  (taxi_speed * overtake_time = bus_speed * (time_difference + overtake_time)) →
  taxi_speed = 60 := by
  sorry

#check taxi_speed_is_60

end NUMINAMATH_CALUDE_taxi_speed_is_60_l1410_141069


namespace NUMINAMATH_CALUDE_triangle_circumcircle_radius_l1410_141042

theorem triangle_circumcircle_radius 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π) -- Sum of angles in a triangle
  (h5 : Real.sin C + Real.sin B = 4 * Real.sin A) -- Given condition
  (h6 : a = 2) -- Given condition
  (h7 : a = 2 * Real.sin (A/2) * R) -- Relation between side and circumradius
  (h8 : b = 2 * Real.sin (B/2) * R) -- Relation between side and circumradius
  (h9 : c = 2 * Real.sin (C/2) * R) -- Relation between side and circumradius
  (h10 : ∀ R' > 0, R ≤ R') -- R is the minimum possible radius
  : R = 8 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_radius_l1410_141042


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1410_141002

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - x - 6 ≥ 0}
def B : Set ℝ := {x | (1 - x) / (x - 3) ≥ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ 1 ∨ x ≤ -3/2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1410_141002


namespace NUMINAMATH_CALUDE_pyramid_volume_is_2000_div_3_l1410_141063

/-- A triangle in 2D space --/
structure Triangle where
  A : (ℝ × ℝ)
  B : (ℝ × ℝ)
  C : (ℝ × ℝ)

/-- The triangle described in the problem --/
def problemTriangle : Triangle :=
  { A := (0, 0),
    B := (30, 0),
    C := (15, 20) }

/-- Function to calculate the volume of the pyramid formed by folding the triangle --/
def pyramidVolume (t : Triangle) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the volume of the pyramid is 2000/3 --/
theorem pyramid_volume_is_2000_div_3 :
  pyramidVolume problemTriangle = 2000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_2000_div_3_l1410_141063


namespace NUMINAMATH_CALUDE_parabola_sum_l1410_141052

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate stating that a point (x, y) is on the parabola -/
def on_parabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Predicate stating that (h, k) is the vertex of the parabola -/
def has_vertex (p : Parabola) (h k : ℝ) : Prop :=
  ∀ x, p.a * (x - h)^2 + k = p.a * x^2 + p.b * x + p.c

/-- The axis of symmetry is vertical when x = h, where (h, k) is the vertex -/
def has_vertical_axis_of_symmetry (p : Parabola) (h : ℝ) : Prop :=
  ∀ x y, on_parabola p x y ↔ on_parabola p (2*h - x) y

theorem parabola_sum (p : Parabola) :
  has_vertex p 4 4 →
  has_vertical_axis_of_symmetry p 4 →
  on_parabola p 3 0 →
  p.a + p.b + p.c = -32 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l1410_141052


namespace NUMINAMATH_CALUDE_boys_in_class_l1410_141068

theorem boys_in_class (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 485 →
  diff = 69 →
  total = boys + (boys + diff) →
  boys = 208 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l1410_141068


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l1410_141086

/-- Given an ellipse and a hyperbola sharing a common focus, prove that a² = 11 under specific conditions --/
theorem ellipse_hyperbola_intersection (a b : ℝ) : 
  a > b → b > 0 →
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1) →  -- Ellipse C1
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / 2 - (y t)^2 / 8 = 1) →  -- Hyperbola C2
  (a^2 - b^2 = 10) →  -- Common focus condition
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = a^2) ∧ (B.1^2 + B.2^2 = a^2) ∧  -- A and B on circle
    (∃ (k : ℝ), A.2 = k * A.1 ∧ B.2 = k * B.1) ∧  -- A and B on asymptote
    (∃ (C D : ℝ × ℝ), 
      C.1^2 / a^2 + C.2^2 / b^2 = 1 ∧
      D.1^2 / a^2 + D.2^2 / b^2 = 1 ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*a/3)^2)) →  -- C1 divides AB into three equal parts
  a^2 = 11 := by
sorry


end NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l1410_141086


namespace NUMINAMATH_CALUDE_odd_function_sum_l1410_141036

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1410_141036


namespace NUMINAMATH_CALUDE_chlorine_and_hcl_moles_l1410_141008

/-- Represents the stoichiometric coefficients of the chemical reaction:
    C2H6 + 6Cl2 → C2Cl6 + 6HCl -/
structure ReactionCoefficients where
  ethane : ℕ
  chlorine : ℕ
  hexachloroethane : ℕ
  hydrochloric_acid : ℕ

/-- The given chemical reaction -/
def reaction : ReactionCoefficients :=
  { ethane := 1
  , chlorine := 6
  , hexachloroethane := 1
  , hydrochloric_acid := 6 }

/-- The number of moles of ethane given -/
def ethane_moles : ℕ := 3

/-- Theorem stating the number of moles of chlorine required and hydrochloric acid formed -/
theorem chlorine_and_hcl_moles :
  (ethane_moles * reaction.chlorine = 18) ∧
  (ethane_moles * reaction.hydrochloric_acid = 18) := by
  sorry

end NUMINAMATH_CALUDE_chlorine_and_hcl_moles_l1410_141008


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_sqrt_three_l1410_141056

theorem smallest_integer_greater_than_sqrt_three : 
  ∀ n : ℤ, n > Real.sqrt 3 → n ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_sqrt_three_l1410_141056


namespace NUMINAMATH_CALUDE_non_zero_digits_after_decimal_l1410_141076

theorem non_zero_digits_after_decimal (n : ℕ) (d : ℕ) : 
  (720 : ℚ) / (2^5 * 5^9) = n / (10^d) ∧ 
  n % 10 ≠ 0 ∧
  n < 10^4 ∧ 
  n ≥ 10^3 →
  d = 8 :=
sorry

end NUMINAMATH_CALUDE_non_zero_digits_after_decimal_l1410_141076


namespace NUMINAMATH_CALUDE_expression_simplification_l1410_141016

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x^2 - 1) / (x^2 + x) / (x - (2*x - 1) / x) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1410_141016


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1410_141015

theorem root_sum_theorem : ∃ (a b : ℝ), 
  (∃ (x y : ℝ), x ≠ y ∧ 
    ((a * x^2 - 24 * x + b) / (x^2 - 1) = x) ∧ 
    ((a * y^2 - 24 * y + b) / (y^2 - 1) = y) ∧
    x + y = 12) ∧
  ((a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819)) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1410_141015


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1410_141020

theorem hyperbola_equation (x y : ℝ) :
  (∀ t : ℝ, y = (2/3) * x ∨ y = -(2/3) * x) →  -- asymptotes condition
  (∃ x₀ y₀ : ℝ, x₀ = 3 ∧ y₀ = 4 ∧ (y₀^2 / 12 - x₀^2 / 27 = 1)) →  -- point condition
  (y^2 / 12 - x^2 / 27 = 1) :=  -- equation of the hyperbola
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1410_141020


namespace NUMINAMATH_CALUDE_factor_expression_l1410_141074

theorem factor_expression (x : ℝ) : 75 * x^13 + 450 * x^26 = 75 * x^13 * (1 + 6 * x^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1410_141074


namespace NUMINAMATH_CALUDE_damaged_polynomial_satisfies_equation_damaged_polynomial_value_l1410_141059

-- Define the damaged polynomial
def damaged_polynomial (x y : ℚ) : ℚ := -3 * x + y^2

-- Define the given equation
def equation_holds (x y : ℚ) : Prop :=
  damaged_polynomial x y + 2 * (x - 1/3 * y^2) = -x + 1/3 * y^2

-- Theorem 1: The damaged polynomial satisfies the equation
theorem damaged_polynomial_satisfies_equation :
  ∀ x y : ℚ, equation_holds x y :=
sorry

-- Theorem 2: The value of the damaged polynomial for given x and y
theorem damaged_polynomial_value :
  damaged_polynomial (-3) (3/2) = 45/4 :=
sorry

end NUMINAMATH_CALUDE_damaged_polynomial_satisfies_equation_damaged_polynomial_value_l1410_141059


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1410_141043

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}
def B : Set ℝ := {x | (x-3)*(2-x) ≥ 0}

-- State the theorem
theorem necessary_not_sufficient_condition (a : ℝ) (h1 : a > 0) 
  (h2 : B ⊂ A a) (h3 : A a ≠ B) : a ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1410_141043


namespace NUMINAMATH_CALUDE_library_books_count_l1410_141022

theorem library_books_count (children_percentage : ℝ) (adult_count : ℕ) : 
  children_percentage = 35 →
  adult_count = 104 →
  ∃ (total : ℕ), (total : ℝ) * (1 - children_percentage / 100) = adult_count ∧ total = 160 :=
by sorry

end NUMINAMATH_CALUDE_library_books_count_l1410_141022


namespace NUMINAMATH_CALUDE_exactly_three_primes_39p_plus_1_perfect_square_l1410_141095

theorem exactly_three_primes_39p_plus_1_perfect_square :
  ∃! (s : Finset Nat), 
    (∀ p ∈ s, Nat.Prime p ∧ ∃ n : Nat, 39 * p + 1 = n^2) ∧ 
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_primes_39p_plus_1_perfect_square_l1410_141095


namespace NUMINAMATH_CALUDE_sum_of_squares_l1410_141061

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 23 → 
  a * b + b * c + a * c = 131 → 
  a^2 + b^2 + c^2 = 267 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1410_141061


namespace NUMINAMATH_CALUDE_boat_speed_problem_l1410_141031

/-- A problem about a man navigating a boat through different river currents -/
theorem boat_speed_problem 
  (speed_with_current : ℝ) 
  (first_current_speed : ℝ)
  (perpendicular_current_speed : ℝ)
  (final_current_speed : ℝ)
  (h1 : speed_with_current = 15)
  (h2 : first_current_speed = 2.8)
  (h3 : perpendicular_current_speed = 3)
  (h4 : final_current_speed = 4.5) :
  let actual_speed := speed_with_current - first_current_speed
  actual_speed = 12.2 ∧ actual_speed - final_current_speed = 7.7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_problem_l1410_141031


namespace NUMINAMATH_CALUDE_digit_placement_combinations_l1410_141005

def grid_size : ℕ := 6
def num_digits : ℕ := 4

theorem digit_placement_combinations : 
  (grid_size * (grid_size - 1) * (grid_size - 2) * (grid_size - 3) * (grid_size - 4)) = 720 :=
by sorry

end NUMINAMATH_CALUDE_digit_placement_combinations_l1410_141005


namespace NUMINAMATH_CALUDE_constant_term_value_l1410_141064

theorem constant_term_value (y : ℝ) (c : ℝ) : 
  y = 2 → 5 * y^2 - 8 * y + c = 59 → c = 55 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l1410_141064


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1410_141017

/-- The probability of a coin landing tails up -/
def ProbTails (coin : Nat) : ℚ :=
  match coin with
  | 1 => 3/4  -- Coin A
  | 2 => 1/2  -- Coin B
  | 3 => 1/4  -- Coin C
  | _ => 0    -- Invalid coin number

/-- The probability of the desired outcome -/
def DesiredOutcome : ℚ :=
  ProbTails 1 * ProbTails 2 * (1 - ProbTails 3)

theorem coin_flip_probability :
  DesiredOutcome = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1410_141017


namespace NUMINAMATH_CALUDE_sequence_partition_sequence_partition_general_l1410_141085

-- Define the type for our sequence
def Sequence := ℕ → Set ℝ

-- Define what it means for a sequence to be in [0, 1)
def InUnitInterval (s : Sequence) : Prop :=
  ∀ n, ∀ x ∈ s n, 0 ≤ x ∧ x < 1

-- Define what it means for a set to contain infinitely many elements of a sequence
def ContainsInfinitelyMany (A : Set ℝ) (s : Sequence) : Prop :=
  ∀ N, ∃ n ≥ N, ∃ x ∈ s n, x ∈ A

theorem sequence_partition (s : Sequence) (h : InUnitInterval s) :
  ContainsInfinitelyMany (Set.Icc 0 (1/2)) s ∨ ContainsInfinitelyMany (Set.Ico (1/2) 1) s :=
sorry

theorem sequence_partition_general (s : Sequence) (h : InUnitInterval s) :
  ∀ n : ℕ, n ≥ 1 →
    ∃ k : ℕ, k < 2^n ∧
      ContainsInfinitelyMany (Set.Ico (k / 2^n) ((k + 1) / 2^n)) s :=
sorry

end NUMINAMATH_CALUDE_sequence_partition_sequence_partition_general_l1410_141085


namespace NUMINAMATH_CALUDE_root_product_theorem_l1410_141096

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - x^3 + 2*x^2 - x + 1

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0) (h₄ : f x₄ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ = 667 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1410_141096


namespace NUMINAMATH_CALUDE_total_toll_for_week_l1410_141006

/-- Calculate the total toll for a week for an 18-wheel truck -/
theorem total_toll_for_week (total_wheels : Nat) (front_axle_wheels : Nat) (other_axle_wheels : Nat)
  (weekday_base_toll : Real) (weekday_rate : Real) (weekend_base_toll : Real) (weekend_rate : Real) :
  total_wheels = 18 →
  front_axle_wheels = 2 →
  other_axle_wheels = 4 →
  weekday_base_toll = 2.50 →
  weekday_rate = 0.70 →
  weekend_base_toll = 3.00 →
  weekend_rate = 0.80 →
  let total_axles := (total_wheels - front_axle_wheels) / other_axle_wheels + 1
  let weekday_toll := weekday_base_toll + weekday_rate * (total_axles - 1)
  let weekend_toll := weekend_base_toll + weekend_rate * (total_axles - 1)
  let total_toll := 5 * weekday_toll + 2 * weekend_toll
  total_toll = 38.90 := by
  sorry

end NUMINAMATH_CALUDE_total_toll_for_week_l1410_141006


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1410_141028

-- Define propositions p and q
def p (x y : ℝ) : Prop := x > 0 ∧ y > 0
def q (x y : ℝ) : Prop := x * y > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1410_141028


namespace NUMINAMATH_CALUDE_smallest_positive_w_l1410_141038

theorem smallest_positive_w (y w : Real) (h1 : Real.sin y = 0) (h2 : Real.sin (y + w) = Real.sqrt 3 / 2) :
  ∃ (w_min : Real), w_min > 0 ∧ w_min = π / 3 ∧ ∀ (w' : Real), w' > 0 ∧ Real.sin y = 0 ∧ Real.sin (y + w') = Real.sqrt 3 / 2 → w' ≥ w_min :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_w_l1410_141038


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l1410_141019

theorem fourth_root_equation_solution (x : ℝ) :
  (x^3)^(1/4) = 81 * 81^(1/16) → x = 243 * 9^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l1410_141019


namespace NUMINAMATH_CALUDE_correct_total_paths_l1410_141058

/-- The number of paths from Wolfburg to the Green Meadows -/
def paths_wolfburg_to_meadows : ℕ := 6

/-- The number of paths from the Green Meadows to Sheep Village -/
def paths_meadows_to_village : ℕ := 20

/-- Wolfburg and Sheep Village are separated by the Green Meadows -/
axiom separated_by_meadows : True

/-- The number of different ways to travel from Wolfburg to Sheep Village -/
def total_paths : ℕ := paths_wolfburg_to_meadows * paths_meadows_to_village

theorem correct_total_paths : total_paths = 120 := by sorry

end NUMINAMATH_CALUDE_correct_total_paths_l1410_141058


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1410_141000

-- Define the types for points and circles
variable (Point Circle : Type)
-- Define the predicate for a point lying on a circle
variable (lies_on : Point → Circle → Prop)
-- Define the predicate for two circles intersecting
variable (intersect : Circle → Circle → Prop)
-- Define the predicate for a circle being tangent to another circle
variable (tangent : Circle → Circle → Prop)
-- Define the predicate for a point being the intersection of a line and a circle
variable (line_circle_intersection : Point → Point → Circle → Point → Prop)
-- Define the predicate for four points being concyclic
variable (concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_theorem 
  (Γ₁ Γ₂ Γ : Circle) 
  (A B C D E F G H I : Point) :
  intersect Γ₁ Γ₂ →
  lies_on A Γ₁ ∧ lies_on A Γ₂ →
  lies_on B Γ₁ ∧ lies_on B Γ₂ →
  tangent Γ Γ₁ ∧ tangent Γ Γ₂ →
  lies_on D Γ ∧ lies_on D Γ₁ →
  lies_on E Γ ∧ lies_on E Γ₂ →
  line_circle_intersection A B Γ C →
  line_circle_intersection E C Γ₂ F →
  line_circle_intersection D C Γ₁ G →
  line_circle_intersection E D Γ₁ H →
  line_circle_intersection E D Γ₂ I →
  concyclic F G H I := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1410_141000


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l1410_141071

theorem log_equality_implies_y_value (m y : ℝ) (hm : m > 0) (hy : y > 0) :
  (Real.log y / Real.log m) * (Real.log m / Real.log 7) = 4 → y = 2401 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l1410_141071


namespace NUMINAMATH_CALUDE_pieces_per_box_l1410_141065

/-- Proves that the number of pieces per box is 6 given the initial conditions --/
theorem pieces_per_box (initial_boxes : Real) (boxes_given_away : Real) (remaining_pieces : ℕ) :
  initial_boxes = 14.0 →
  boxes_given_away = 7.0 →
  remaining_pieces = 42 →
  (remaining_pieces : Real) / (initial_boxes - boxes_given_away) = 6 := by
  sorry

#check pieces_per_box

end NUMINAMATH_CALUDE_pieces_per_box_l1410_141065


namespace NUMINAMATH_CALUDE_problem_solution_l1410_141003

theorem problem_solution (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1410_141003


namespace NUMINAMATH_CALUDE_tenth_black_ball_probability_l1410_141013

/-- Represents the probability of drawing a black ball on the tenth draw from a box of colored balls. -/
def probability_tenth_black_ball (total_balls : ℕ) (black_balls : ℕ) : ℚ :=
  black_balls / total_balls

/-- Theorem stating that the probability of drawing a black ball on the tenth draw
    from a box with specific numbers of colored balls is 4/30. -/
theorem tenth_black_ball_probability :
  let red_balls : ℕ := 7
  let black_balls : ℕ := 4
  let yellow_balls : ℕ := 5
  let green_balls : ℕ := 6
  let white_balls : ℕ := 8
  let total_balls : ℕ := red_balls + black_balls + yellow_balls + green_balls + white_balls
  probability_tenth_black_ball total_balls black_balls = 4 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_black_ball_probability_l1410_141013


namespace NUMINAMATH_CALUDE_fuel_refills_l1410_141094

theorem fuel_refills (total_spent : ℕ) (cost_per_refill : ℕ) (h1 : total_spent = 40) (h2 : cost_per_refill = 10) :
  total_spent / cost_per_refill = 4 := by
  sorry

end NUMINAMATH_CALUDE_fuel_refills_l1410_141094


namespace NUMINAMATH_CALUDE_susan_walk_distance_l1410_141025

theorem susan_walk_distance (total_distance : ℝ) (erin_susan_diff : ℝ) (daniel_susan_ratio : ℝ) :
  total_distance = 32 ∧
  erin_susan_diff = 3 ∧
  daniel_susan_ratio = 2 →
  ∃ susan_distance : ℝ,
    susan_distance + (susan_distance - erin_susan_diff) + (daniel_susan_ratio * susan_distance) = total_distance ∧
    susan_distance = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_susan_walk_distance_l1410_141025


namespace NUMINAMATH_CALUDE_student_count_third_row_l1410_141040

/-- The number of students in the first row -/
def students_first_row : ℕ := 12

/-- The number of students in the second row -/
def students_second_row : ℕ := 12

/-- The change in average age (in weeks) for the first row after rearrangement -/
def change_first_row : ℤ := 1

/-- The change in average age (in weeks) for the second row after rearrangement -/
def change_second_row : ℤ := 2

/-- The change in average age (in weeks) for the third row after rearrangement -/
def change_third_row : ℤ := -4

/-- The number of students in the third row -/
def students_third_row : ℕ := 9

theorem student_count_third_row : 
  students_first_row * change_first_row + 
  students_second_row * change_second_row + 
  students_third_row * change_third_row = 0 :=
by sorry

end NUMINAMATH_CALUDE_student_count_third_row_l1410_141040


namespace NUMINAMATH_CALUDE_lychee_production_increase_l1410_141055

theorem lychee_production_increase (x : ℝ) : 
  let increase_factor := 1 + x / 100
  let two_year_increase := increase_factor ^ 2 - 1
  two_year_increase = ((1 + x / 100) ^ 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_lychee_production_increase_l1410_141055


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1410_141090

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (z : ℂ), z = m^2 - 1 + (m + 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1410_141090


namespace NUMINAMATH_CALUDE_movies_on_shelves_l1410_141039

theorem movies_on_shelves (total_movies : ℕ) (num_shelves : ℕ) (h1 : total_movies = 999) (h2 : num_shelves = 5) :
  ∃ (additional_movies : ℕ), 
    additional_movies = 1 ∧ 
    (total_movies + additional_movies) % num_shelves = 0 :=
by sorry

end NUMINAMATH_CALUDE_movies_on_shelves_l1410_141039


namespace NUMINAMATH_CALUDE_max_nSn_l1410_141057

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  sum : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The problem statement -/
theorem max_nSn (seq : ArithmeticSequence) 
  (h1 : seq.sum 6 = 26)
  (h2 : seq.a 7 = 2) :
  ∃ m : ℚ, m = 338 ∧ ∀ n : ℕ, n * seq.sum n ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_nSn_l1410_141057


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1410_141079

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  1 / x + 4 / y ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1410_141079


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l1410_141054

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define perpendicularity
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

-- Define a point being on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define a line passing through a point
def line_through_point (l : Line) (p : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem perpendicular_line_exists 
  (C : Circle) (A B M : ℝ × ℝ) (diameter : Line) :
  point_on_circle A C →
  point_on_circle B C →
  point_on_line A diameter →
  point_on_line B diameter →
  ∃ (L : Line), line_through_point L M ∧ perpendicular L diameter :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l1410_141054


namespace NUMINAMATH_CALUDE_total_ants_is_twenty_l1410_141082

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := abe_ants + abe_ants / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_is_twenty : total_ants = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_is_twenty_l1410_141082


namespace NUMINAMATH_CALUDE_profit_percentage_problem_l1410_141001

/-- Calculates the profit percentage given the cost price and selling price -/
def profit_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the profit percentage is 25% for the given problem -/
theorem profit_percentage_problem : profit_percentage 96 120 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_problem_l1410_141001


namespace NUMINAMATH_CALUDE_horner_method_f_3_f_3_equals_328_l1410_141083

/-- Horner's method representation of a polynomial -/
def horner_rep (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

/-- The polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_method_f_3 :
  f 3 = horner_rep [1, 0, 2, 3, 1, 1] 3 := by
  sorry

theorem f_3_equals_328 : f 3 = 328 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_3_f_3_equals_328_l1410_141083


namespace NUMINAMATH_CALUDE_range_a_characterization_l1410_141081

/-- The range of values for a where "p or q" is true and "p and q" is false -/
def range_a : Set ℝ := Set.union (Set.Ioc 0 0.5) (Set.Ico 1 2)

/-- p is true when 0 < a < 1 -/
def p_true (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- q is true when 0.5 < a < 2 -/
def q_true (a : ℝ) : Prop := 0.5 < a ∧ a < 2

theorem range_a_characterization (a : ℝ) (h : a > 0) :
  a ∈ range_a ↔ (p_true a ∨ q_true a) ∧ ¬(p_true a ∧ q_true a) :=
by sorry

end NUMINAMATH_CALUDE_range_a_characterization_l1410_141081


namespace NUMINAMATH_CALUDE_parabola_focus_l1410_141014

/-- The focus of a parabola with equation y^2 = -6x has coordinates (-3/2, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -6*x → (x + 3/2)^2 + y^2 = (3/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1410_141014


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1410_141060

-- Define the vectors
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1410_141060


namespace NUMINAMATH_CALUDE_laundry_time_calculation_l1410_141077

theorem laundry_time_calculation (loads : ℕ) (wash_time : ℕ) (dry_time : ℕ) :
  loads = 8 ∧ wash_time = 45 ∧ dry_time = 60 →
  (loads * (wash_time + dry_time)) / 60 = 14 := by
  sorry

end NUMINAMATH_CALUDE_laundry_time_calculation_l1410_141077


namespace NUMINAMATH_CALUDE_applicant_a_wins_l1410_141084

/-- Represents an applicant with their test scores -/
structure Applicant where
  education : ℝ
  experience : ℝ
  work_attitude : ℝ

/-- Calculates the final score of an applicant given the weights -/
def final_score (a : Applicant) (w_edu w_exp w_att : ℝ) : ℝ :=
  a.education * w_edu + a.experience * w_exp + a.work_attitude * w_att

/-- Theorem stating that Applicant A's final score is higher than Applicant B's -/
theorem applicant_a_wins (applicant_a applicant_b : Applicant)
    (h_a_edu : applicant_a.education = 7)
    (h_a_exp : applicant_a.experience = 8)
    (h_a_att : applicant_a.work_attitude = 9)
    (h_b_edu : applicant_b.education = 10)
    (h_b_exp : applicant_b.experience = 7)
    (h_b_att : applicant_b.work_attitude = 8) :
    final_score applicant_a (1/6) (1/3) (1/2) > final_score applicant_b (1/6) (1/3) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_applicant_a_wins_l1410_141084


namespace NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l1410_141097

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let c1_center : ℝ × ℝ := (5, 5)
  let c2_center : ℝ × ℝ := (14, 5)
  let radius : ℝ := 3
  let rectangle_area : ℝ := (14 - 5) * 5
  let circle_segment_area : ℝ := 2 * (π * radius^2 / 4)
  rectangle_area - circle_segment_area = 45 - 9 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l1410_141097


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1410_141046

/-- Given a line in vector form, prove it's equivalent to slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y + 4) = 0 ↔ y = 2 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1410_141046


namespace NUMINAMATH_CALUDE_rectangular_shingle_area_l1410_141089

/-- The area of a rectangular roof shingle -/
theorem rectangular_shingle_area (length width : ℝ) (h1 : length = 10) (h2 : width = 7) :
  length * width = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_shingle_area_l1410_141089


namespace NUMINAMATH_CALUDE_y_value_proof_l1410_141093

theorem y_value_proof (x y z a b c : ℝ) 
  (ha : x * y / (x + y) = a)
  (hb : x * z / (x + z) = b)
  (hc : y * z / (y + z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  y = 2 * a * b * c / (b * c + a * c - a * b) :=
sorry

end NUMINAMATH_CALUDE_y_value_proof_l1410_141093


namespace NUMINAMATH_CALUDE_cube_volume_and_surface_area_l1410_141072

/-- Represents a cube with edge length in centimeters -/
structure Cube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The sum of all edge lengths of the cube -/
def Cube.sumEdgeLength (c : Cube) : ℝ := 12 * c.edgeLength

/-- The volume of the cube -/
def Cube.volume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- The surface area of the cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength ^ 2

theorem cube_volume_and_surface_area 
  (c : Cube) 
  (h : c.sumEdgeLength = 72) : 
  c.volume = 216 ∧ c.surfaceArea = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_and_surface_area_l1410_141072


namespace NUMINAMATH_CALUDE_arctan_sum_identity_l1410_141007

theorem arctan_sum_identity : 
  Real.arctan (3/4) + 2 * Real.arctan (4/3) = π - Real.arctan (3/4) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_identity_l1410_141007


namespace NUMINAMATH_CALUDE_circle_tangent_to_semicircles_radius_bounds_l1410_141047

/-- Given a triangle ABC with semiperimeter s and inradius r, and semicircles drawn on its sides,
    the radius t of the circle tangent to all three semicircles satisfies:
    s/2 < t ≤ s/2 + (1 - √3/2)r -/
theorem circle_tangent_to_semicircles_radius_bounds
  (s r t : ℝ) -- semiperimeter, inradius, and radius of tangent circle
  (h_s_pos : 0 < s)
  (h_r_pos : 0 < r)
  (h_t_pos : 0 < t)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ s = (a + b + c) / 2)
  (h_inradius : ∃ (area : ℝ), area > 0 ∧ r = area / s)
  (h_tangent : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
               t + x/2 = t + y/2 ∧ t + y/2 = t + z/2 ∧ x + y + z = 2 * s) :
  s / 2 < t ∧ t ≤ s / 2 + (1 - Real.sqrt 3 / 2) * r := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_semicircles_radius_bounds_l1410_141047


namespace NUMINAMATH_CALUDE_hot_drink_sales_at_2_degrees_l1410_141092

/-- Represents the linear regression equation for hot drink sales -/
def hot_drink_sales (x : ℝ) : ℝ := -2.35 * x + 147.77

/-- Theorem stating that when the temperature is 2℃, approximately 143 hot drinks are sold -/
theorem hot_drink_sales_at_2_degrees :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |hot_drink_sales 2 - 143| < ε :=
sorry

end NUMINAMATH_CALUDE_hot_drink_sales_at_2_degrees_l1410_141092


namespace NUMINAMATH_CALUDE_abc_product_l1410_141050

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = (11 + Real.sqrt 117) / 2 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1410_141050


namespace NUMINAMATH_CALUDE_fraction_simplification_l1410_141021

theorem fraction_simplification : (1 : ℚ) / (2 + 2/3) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1410_141021


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1410_141030

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/144
  let r : ℚ := a₂ / a₁
  r = -10/21 ∧ a₃ / a₂ = r := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1410_141030


namespace NUMINAMATH_CALUDE_inequality_proof_l1410_141012

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1410_141012


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1410_141026

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - x^2 + 4*x - 30

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, polynomial x = (divisor x) * (q x) + (-34) := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1410_141026


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1410_141034

/-- The standard equation of an ellipse given its focal distance and sum of distances from a point to foci -/
theorem ellipse_standard_equation (focal_distance sum_distances : ℝ) :
  focal_distance = 8 →
  sum_distances = 10 →
  (∃ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) ∨ (∃ x y : ℝ, x^2 / 9 + y^2 / 25 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1410_141034


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1410_141088

theorem binomial_expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 9 = 54 → n = 4 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1410_141088


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1410_141066

/-- The area of a rectangular field with one side of 4 meters and a diagonal of 5 meters is 12 square meters. -/
theorem rectangular_field_area : ∀ (w l : ℝ), 
  w = 4 → 
  w^2 + l^2 = 5^2 → 
  w * l = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1410_141066


namespace NUMINAMATH_CALUDE_f_properties_l1410_141029

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x / Real.log 2

theorem f_properties :
  (∀ x > 0, f (-x) ≠ -f x ∧ f (-x) ≠ f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1410_141029


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l1410_141044

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  (∃ k : ℕ, a = 15 * k ∨ b = 15 * k) →  -- one angle is multiple of 15
  |a - b| = 45 := by
sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l1410_141044


namespace NUMINAMATH_CALUDE_ways_to_pay_100_l1410_141024

/-- Represents the available coin denominations -/
def CoinDenominations : List Nat := [1, 2, 10, 20, 50]

/-- Calculates the number of ways to pay a given amount using the available coin denominations -/
def waysToPayAmount (amount : Nat) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that there are 784 ways to pay 100 using the given coin denominations -/
theorem ways_to_pay_100 : waysToPayAmount 100 = 784 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_pay_100_l1410_141024


namespace NUMINAMATH_CALUDE_palindrome_count_l1410_141035

/-- Represents a time on a 12-hour digital clock --/
structure Time where
  hour : Nat
  minute : Nat
  hour_valid : 1 ≤ hour ∧ hour ≤ 12
  minute_valid : minute < 60

/-- Checks if a given time is a palindrome --/
def isPalindrome (t : Time) : Bool :=
  let digits := 
    if t.hour < 10 then
      [t.hour, t.minute / 10, t.minute % 10]
    else
      [t.hour / 10, t.hour % 10, t.minute / 10, t.minute % 10]
  digits = digits.reverse

/-- The set of all valid palindrome times on a 12-hour digital clock --/
def palindromeTimes : Finset Time :=
  sorry

theorem palindrome_count : palindromeTimes.card = 57 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_count_l1410_141035


namespace NUMINAMATH_CALUDE_lateral_surface_area_is_4S_l1410_141073

/-- A regular quadrilateral pyramid with specific properties -/
structure RegularQuadPyramid where
  -- The dihedral angle at the lateral edge
  dihedral_angle : ℝ
  -- The area of the diagonal section
  diagonal_section_area : ℝ
  -- Condition that the dihedral angle is 120°
  angle_is_120 : dihedral_angle = 120 * π / 180

/-- The lateral surface area of a regular quadrilateral pyramid -/
def lateral_surface_area (p : RegularQuadPyramid) : ℝ := 4 * p.diagonal_section_area

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid with a 120° dihedral angle
    at the lateral edge is 4 times the area of its diagonal section -/
theorem lateral_surface_area_is_4S (p : RegularQuadPyramid) :
  lateral_surface_area p = 4 * p.diagonal_section_area := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_is_4S_l1410_141073


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1410_141037

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : is_geometric_sequence a)
  (h_prod1 : a 1 * a 2 * a 3 = 4)
  (h_prod2 : a 4 * a 5 * a 6 = 12)
  (h_prod3 : ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324) :
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324 ∧ n = 14 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1410_141037


namespace NUMINAMATH_CALUDE_shadow_length_proportion_l1410_141011

/-- Represents a pot with its height and shadow length -/
structure Pot where
  height : ℝ
  shadowLength : ℝ

/-- Theorem stating the relationship between pot heights and shadow lengths -/
theorem shadow_length_proportion (pot1 pot2 : Pot)
  (h1 : pot1.height = 20)
  (h2 : pot1.shadowLength = 10)
  (h3 : pot2.height = 40)
  (h4 : pot2.shadowLength = 20)
  (h5 : pot2.height = 2 * pot1.height)
  (h6 : pot2.shadowLength = 2 * pot1.shadowLength) :
  pot1.shadowLength = pot2.shadowLength / 2 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_proportion_l1410_141011


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1410_141075

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 + 2 * x - 5 < 8} = {x : ℝ | -2 * Real.sqrt 10 / 6 - 1 / 3 < x ∧ x < 2 * Real.sqrt 10 / 6 - 1 / 3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1410_141075


namespace NUMINAMATH_CALUDE_square_units_digits_correct_l1410_141062

/-- The set of all possible units digits of squares of whole numbers -/
def square_units_digits : Set Nat :=
  {0, 1, 4, 5, 6, 9}

/-- Function to get the units digit of a number -/
def units_digit (n : Nat) : Nat :=
  n % 10

/-- Theorem stating that the set of all possible units digits of squares of whole numbers
    is exactly {0, 1, 4, 5, 6, 9} -/
theorem square_units_digits_correct :
  ∀ n : Nat, ∃ m : Nat, units_digit (m * m) ∈ square_units_digits ∧
  ∀ k : Nat, units_digit (k * k) ∈ square_units_digits := by
  sorry

#check square_units_digits_correct

end NUMINAMATH_CALUDE_square_units_digits_correct_l1410_141062


namespace NUMINAMATH_CALUDE_complex_product_equality_complex_product_equality_proof_l1410_141091

theorem complex_product_equality : Complex → Prop :=
  fun i =>
    i * i = -1 →
    (1 + i) * (2 - i) = 3 + i

-- The proof is omitted
theorem complex_product_equality_proof : complex_product_equality Complex.I :=
  sorry

end NUMINAMATH_CALUDE_complex_product_equality_complex_product_equality_proof_l1410_141091


namespace NUMINAMATH_CALUDE_circle_radius_zero_l1410_141067

theorem circle_radius_zero (x y : ℝ) :
  25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l1410_141067
