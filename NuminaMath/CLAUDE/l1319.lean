import Mathlib

namespace NUMINAMATH_CALUDE_probability_is_175_323_l1319_131971

-- Define the number of black and white balls
def black_balls : ℕ := 10
def white_balls : ℕ := 9
def total_balls : ℕ := black_balls + white_balls

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the probability function
def probability_at_least_two_black : ℚ :=
  (Nat.choose black_balls 2 * Nat.choose white_balls 1 +
   Nat.choose black_balls 3) /
  Nat.choose total_balls drawn_balls

-- Theorem statement
theorem probability_is_175_323 :
  probability_at_least_two_black = 175 / 323 :=
by sorry

end NUMINAMATH_CALUDE_probability_is_175_323_l1319_131971


namespace NUMINAMATH_CALUDE_amount_subtracted_l1319_131984

theorem amount_subtracted (number : ℝ) (result : ℝ) (amount : ℝ) : 
  number = 150 →
  result = 50 →
  0.60 * number - amount = result →
  amount = 40 := by
sorry

end NUMINAMATH_CALUDE_amount_subtracted_l1319_131984


namespace NUMINAMATH_CALUDE_function_through_point_l1319_131932

/-- Proves that if the function y = k/x passes through the point (3, -1), then k = -3 -/
theorem function_through_point (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f 3 = -1) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l1319_131932


namespace NUMINAMATH_CALUDE_circle_area_difference_l1319_131913

theorem circle_area_difference (L W : ℝ) (h1 : L > 0) (h2 : W > 0) 
  (h3 : L * π = 704) (h4 : W * π = 396) : 
  (π * (L / 2)^2 - π * (W / 2)^2) = (L^2 - W^2) / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1319_131913


namespace NUMINAMATH_CALUDE_quadrilateral_area_in_regular_octagon_l1319_131964

-- Define a regular octagon
structure RegularOctagon :=
  (side_length : ℝ)

-- Define the area of a quadrilateral formed by two adjacent vertices and two diagonal intersections
def quadrilateral_area (octagon : RegularOctagon) : ℝ :=
  sorry

-- Theorem statement
theorem quadrilateral_area_in_regular_octagon 
  (octagon : RegularOctagon) 
  (h : octagon.side_length = 5) : 
  quadrilateral_area octagon = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_in_regular_octagon_l1319_131964


namespace NUMINAMATH_CALUDE_total_squares_blocked_5x6_grid_l1319_131952

/-- Represents a partially blocked grid -/
structure PartialGrid :=
  (rows : Nat)
  (cols : Nat)
  (squares_1x1 : Nat)
  (squares_2x2 : Nat)
  (squares_3x3 : Nat)
  (squares_4x4 : Nat)

/-- Calculates the total number of squares in a partially blocked grid -/
def total_squares (grid : PartialGrid) : Nat :=
  grid.squares_1x1 + grid.squares_2x2 + grid.squares_3x3 + grid.squares_4x4

/-- Theorem: The total number of squares in the given partially blocked 5x6 grid is 57 -/
theorem total_squares_blocked_5x6_grid :
  ∃ (grid : PartialGrid),
    grid.rows = 5 ∧
    grid.cols = 6 ∧
    grid.squares_1x1 = 30 ∧
    grid.squares_2x2 = 18 ∧
    grid.squares_3x3 = 7 ∧
    grid.squares_4x4 = 2 ∧
    total_squares grid = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_total_squares_blocked_5x6_grid_l1319_131952


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1319_131944

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - x^2 / 25 = 1

/-- The asymptote equation -/
def asymptote (m b x y : ℝ) : Prop :=
  y = m * x + b

/-- Theorem stating that the given equations are the asymptotes of the hyperbola -/
theorem hyperbola_asymptotes :
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), hyperbola x y → 
      (asymptote m b x y ∨ asymptote (-m) b x y)) ∧
    m = 4/5 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1319_131944


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l1319_131965

theorem mariela_get_well_cards (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403)
  (h2 : cards_at_home = 287) :
  cards_in_hospital + cards_at_home = 690 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l1319_131965


namespace NUMINAMATH_CALUDE_a_closed_form_l1319_131986

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => a n - (n + 1 : ℚ) / (Nat.factorial (n + 2))

theorem a_closed_form (n : ℕ) :
  a n = (Nat.factorial (n + 1) + 1 : ℚ) / Nat.factorial (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_a_closed_form_l1319_131986


namespace NUMINAMATH_CALUDE_difference_of_squares_l1319_131979

theorem difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1319_131979


namespace NUMINAMATH_CALUDE_choose_three_from_seven_l1319_131958

/-- The number of ways to choose 3 distinct people from a group of 7 to fill 3 distinct positions -/
def ways_to_choose_officers (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem stating that choosing 3 distinct people from a group of 7 to fill 3 distinct positions can be done in 210 ways -/
theorem choose_three_from_seven :
  ways_to_choose_officers 7 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_seven_l1319_131958


namespace NUMINAMATH_CALUDE_odd_function_increasing_function_symmetry_more_than_two_roots_l1319_131991

-- Define the function f
def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

-- Theorem statements
theorem odd_function (b : ℝ) :
  ∀ x, f b 0 x = -f b 0 (-x) := by sorry

theorem increasing_function (c : ℝ) :
  ∀ x y, x < y → f 0 c x < f 0 c y := by sorry

theorem symmetry (b c : ℝ) :
  ∀ x, f b c x - c = -(f b c (-x) - c) := by sorry

theorem more_than_two_roots :
  ∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_increasing_function_symmetry_more_than_two_roots_l1319_131991


namespace NUMINAMATH_CALUDE_average_population_is_1000_l1319_131921

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

theorem average_population_is_1000 : 
  (village_populations.sum / village_populations.length : ℚ) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_average_population_is_1000_l1319_131921


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1319_131947

theorem quadratic_equation_solution (x m n : ℝ) : 
  (x^2 + x + m = (x - n)^2) → (m = 1/4 ∧ n = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1319_131947


namespace NUMINAMATH_CALUDE_pear_trees_count_total_trees_sum_apple_tree_exists_pear_tree_exists_l1319_131990

/-- The number of trees in the garden -/
def total_trees : ℕ := 46

/-- The number of pear trees in the garden -/
def pear_trees : ℕ := 27

/-- The number of apple trees in the garden -/
def apple_trees : ℕ := total_trees - pear_trees

/-- Theorem stating that the number of pear trees is 27 -/
theorem pear_trees_count : pear_trees = 27 := by sorry

/-- Theorem stating that the sum of apple and pear trees equals the total number of trees -/
theorem total_trees_sum : apple_trees + pear_trees = total_trees := by sorry

/-- Theorem stating that among any 28 trees, there is at least one apple tree -/
theorem apple_tree_exists (subset : Finset ℕ) (h : subset.card = 28) (h2 : subset ⊆ Finset.range total_trees) : 
  ∃ (tree : ℕ), tree ∈ subset ∧ tree < apple_trees := by sorry

/-- Theorem stating that among any 20 trees, there is at least one pear tree -/
theorem pear_tree_exists (subset : Finset ℕ) (h : subset.card = 20) (h2 : subset ⊆ Finset.range total_trees) : 
  ∃ (tree : ℕ), tree ∈ subset ∧ tree ≥ apple_trees := by sorry

end NUMINAMATH_CALUDE_pear_trees_count_total_trees_sum_apple_tree_exists_pear_tree_exists_l1319_131990


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l1319_131976

theorem prop_a_necessary_not_sufficient :
  ¬(∀ x y : ℤ, (x ≠ 1000 ∨ y ≠ 1002) ↔ (x + y ≠ 2002)) ∧
  (∀ x y : ℤ, (x + y ≠ 2002) → (x ≠ 1000 ∨ y ≠ 1002)) :=
by sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l1319_131976


namespace NUMINAMATH_CALUDE_factor_sum_l1319_131989

theorem factor_sum (a b : ℝ) : 
  (∃ m n : ℝ, ∀ x : ℝ, x^4 + a*x^2 + b = (x^2 + 2*x + 5) * (x^2 + m*x + n)) →
  a + b = 31 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l1319_131989


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1319_131910

theorem fixed_point_of_exponential_function 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 2
  f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1319_131910


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1319_131959

theorem factorial_divisibility (a : ℕ) : 
  (a.factorial + (a + 2).factorial) ∣ (a + 4).factorial ↔ a = 0 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1319_131959


namespace NUMINAMATH_CALUDE_no_largest_non_expressible_l1319_131974

-- Define a function to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the property of being expressible as the sum of a multiple of 36 and a non-square
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 36 * a + b ∧ b > 0 ∧ ¬(is_square b)

-- Theorem statement
theorem no_largest_non_expressible :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ¬(is_expressible m) :=
sorry

end NUMINAMATH_CALUDE_no_largest_non_expressible_l1319_131974


namespace NUMINAMATH_CALUDE_flat_cost_calculation_l1319_131961

theorem flat_cost_calculation (x : ℝ) : 
  x > 0 →  -- Assuming the cost is positive
  0.11 * x - (-0.11 * x) = 1.21 →
  x = 5.50 := by
  sorry

end NUMINAMATH_CALUDE_flat_cost_calculation_l1319_131961


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_l1319_131999

theorem triangle_angle_cosine (A B C : Real) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle is π radians
  A + C = 2 * B →
  1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B →
  Real.cos ((A - C) / 2) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_l1319_131999


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1319_131977

-- Define the shape
structure CubeShape :=
  (base_length : ℕ)
  (base_width : ℕ)
  (column_height : ℕ)

-- Define the properties of our specific shape
def our_shape : CubeShape :=
  { base_length := 3
  , base_width := 3
  , column_height := 3 }

-- Calculate the volume of the shape
def volume (shape : CubeShape) : ℕ :=
  shape.base_length * shape.base_width + shape.column_height - 1

-- Calculate the surface area of the shape
def surface_area (shape : CubeShape) : ℕ :=
  let base_area := 2 * shape.base_length * shape.base_width
  let side_area := 2 * shape.base_length * shape.column_height + 2 * shape.base_width * shape.column_height
  let column_area := 4 * (shape.column_height - 1)
  base_area + side_area + column_area - (shape.base_length * shape.base_width - 1)

-- Theorem: The ratio of volume to surface area is 9:40
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  shape = our_shape → volume shape * 40 = surface_area shape * 9 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1319_131977


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l1319_131903

/-- A convex quadrilateral with angles α, β, γ, δ -/
structure ConvexQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ

/-- Definition of a parallelogram -/
def IsParallelogram (q : ConvexQuadrilateral) : Prop :=
  q.α + q.γ = 180 ∧ q.β + q.δ = 180

/-- Definition of a trapezoid -/
def IsTrapezoid (q : ConvexQuadrilateral) : Prop :=
  q.α + q.β = 180 ∨ q.β + q.γ = 180 ∨ q.γ + q.δ = 180 ∨ q.δ + q.α = 180

theorem quadrilateral_with_equal_sine_sums (q : ConvexQuadrilateral) 
  (h : Real.sin q.α + Real.sin q.γ = Real.sin q.β + Real.sin q.δ) :
  IsParallelogram q ∨ IsTrapezoid q := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l1319_131903


namespace NUMINAMATH_CALUDE_pants_original_price_l1319_131980

theorem pants_original_price 
  (total_spent : ℝ)
  (jacket_discount : ℝ)
  (pants_discount : ℝ)
  (jacket_original : ℝ)
  (h1 : total_spent = 306)
  (h2 : jacket_discount = 0.7)
  (h3 : pants_discount = 0.8)
  (h4 : jacket_original = 300) :
  ∃ (pants_original : ℝ), 
    jacket_original * jacket_discount + pants_original * pants_discount = total_spent ∧ 
    pants_original = 120 :=
by sorry

end NUMINAMATH_CALUDE_pants_original_price_l1319_131980


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1319_131929

/-- Atomic weight of Potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Potassium atoms in the compound -/
def num_K : ℕ := 1

/-- Number of Bromine atoms in the compound -/
def num_Br : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound -/
def molecular_weight : ℝ := num_K * atomic_weight_K + num_Br * atomic_weight_Br + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 167.00 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1319_131929


namespace NUMINAMATH_CALUDE_trapezoidal_sequence_624_l1319_131918

/-- The trapezoidal sequence -/
def trapezoidal_sequence : ℕ → ℕ
| 0 => 5
| n + 1 => trapezoidal_sequence n + (n + 4)

/-- The 624th term of the trapezoidal sequence is 196250 -/
theorem trapezoidal_sequence_624 : trapezoidal_sequence 623 = 196250 := by
  sorry

end NUMINAMATH_CALUDE_trapezoidal_sequence_624_l1319_131918


namespace NUMINAMATH_CALUDE_courtyard_area_difference_l1319_131998

/-- The difference in area between a circular courtyard and a rectangular courtyard -/
theorem courtyard_area_difference :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter : ℝ := 2 * (rect_length + rect_width)
  let rect_area : ℝ := rect_length * rect_width
  let circle_radius : ℝ := rect_perimeter / (2 * Real.pi)
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  circle_area - rect_area = (6400 - 1200 * Real.pi) / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_courtyard_area_difference_l1319_131998


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1319_131902

theorem no_positive_integer_solutions
  (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) (n : ℕ+) :
  ¬∃ (x y : ℕ+), p^(n : ℕ) = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1319_131902


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l1319_131973

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l1319_131973


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1319_131967

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 0 → 
  exterior_angle = 18 → 
  (n : ℝ) * exterior_angle = 360 →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1319_131967


namespace NUMINAMATH_CALUDE_factorial_ratio_l1319_131962

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1319_131962


namespace NUMINAMATH_CALUDE_age_difference_l1319_131937

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 20 → 
  man_age + 2 = 2 * (son_age + 2) → 
  man_age - son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1319_131937


namespace NUMINAMATH_CALUDE_inequality_proof_l1319_131985

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4*x + y + 2*z) * (2*x + y + 8*z) ≥ (375/2) * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1319_131985


namespace NUMINAMATH_CALUDE_quadratic_point_condition_l1319_131946

/-- The quadratic function y = -(x-1)² + n -/
def f (x n : ℝ) : ℝ := -(x - 1)^2 + n

theorem quadratic_point_condition (m y₁ y₂ n : ℝ) :
  f m n = y₁ →
  f (m + 1) n = y₂ →
  y₁ > y₂ →
  m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_condition_l1319_131946


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1319_131936

theorem polynomial_simplification (y : ℝ) : 
  (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) = 
  2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1319_131936


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l1319_131972

theorem rectangle_ratio_theorem (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + y = 3*s) → (x / y = 2) := by
  sorry

#check rectangle_ratio_theorem

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l1319_131972


namespace NUMINAMATH_CALUDE_polynomial_product_constraint_l1319_131953

theorem polynomial_product_constraint (a b : ℝ) : 
  (∀ x, (a * x + b) * (2 * x + 1) = 2 * a * x^2 + b) ∧ b = 6 → a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_constraint_l1319_131953


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1319_131988

/-- Given positive real numbers a, b, and c such that the minimum value of |x - a| + |x + b| + c is 1 -/
def min_condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x, |x - a| + |x + b| + c ≥ 1) ∧ (∃ x, |x - a| + |x + b| + c = 1)

theorem min_value_and_inequality (a b c : ℝ) (h : min_condition a b c) :
  (∀ x y z, 9*x^2 + 4*y^2 + (1/4)*z^2 ≥ 36/157) ∧
  (9*a^2 + 4*b^2 + (1/4)*c^2 = 36/157) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) + Real.sqrt (c^2 + c*a + a^2) > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1319_131988


namespace NUMINAMATH_CALUDE_unique_m_for_even_f_l1319_131994

/-- A function f is even if f(-x) = f(x) for all x in the domain of f -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + (m-1)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m-1)*x + 3

/-- Theorem: The unique value of m that makes f an even function is 1 -/
theorem unique_m_for_even_f :
  ∃! m, IsEven (f m) := by sorry

end NUMINAMATH_CALUDE_unique_m_for_even_f_l1319_131994


namespace NUMINAMATH_CALUDE_exactly_six_valid_tuples_l1319_131908

def is_valid_tuple (t : Fin 4 → Fin 4) : Prop :=
  (∃ (σ : Equiv (Fin 4) (Fin 4)), ∀ i, t i = σ i) ∧
  (t 0 = 1 ∨ t 1 ≠ 1 ∨ t 2 = 2 ∨ t 3 ≠ 4) ∧
  ¬(t 0 = 1 ∧ t 1 ≠ 1) ∧
  ¬(t 0 = 1 ∧ t 2 = 2) ∧
  ¬(t 0 = 1 ∧ t 3 ≠ 4) ∧
  ¬(t 1 ≠ 1 ∧ t 2 = 2) ∧
  ¬(t 1 ≠ 1 ∧ t 3 ≠ 4) ∧
  ¬(t 2 = 2 ∧ t 3 ≠ 4)

theorem exactly_six_valid_tuples :
  ∃! (s : Finset (Fin 4 → Fin 4)), s.card = 6 ∧ ∀ t, t ∈ s ↔ is_valid_tuple t :=
sorry

end NUMINAMATH_CALUDE_exactly_six_valid_tuples_l1319_131908


namespace NUMINAMATH_CALUDE_solution_set_ax_gt_b_l1319_131992

theorem solution_set_ax_gt_b (a b : ℝ) :
  let S := {x : ℝ | a * x > b}
  (a > 0 → S = {x : ℝ | x > b / a}) ∧
  (a < 0 → S = {x : ℝ | x < b / a}) ∧
  (a = 0 ∧ b ≥ 0 → S = ∅) ∧
  (a = 0 ∧ b < 0 → S = Set.univ) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_ax_gt_b_l1319_131992


namespace NUMINAMATH_CALUDE_average_snack_sales_theorem_l1319_131950

/-- Represents the sales data for snacks per 6 movie tickets sold -/
structure SnackSales where
  crackers_quantity : ℕ
  crackers_price : ℚ
  beverage_quantity : ℕ
  beverage_price : ℚ
  chocolate_quantity : ℕ
  chocolate_price : ℚ

/-- Calculates the average snack sales per movie ticket -/
def average_snack_sales_per_ticket (sales : SnackSales) : ℚ :=
  let total_sales := sales.crackers_quantity * sales.crackers_price +
                     sales.beverage_quantity * sales.beverage_price +
                     sales.chocolate_quantity * sales.chocolate_price
  total_sales / 6

/-- The main theorem stating the average snack sales per movie ticket -/
theorem average_snack_sales_theorem (sales : SnackSales) 
  (h1 : sales.crackers_quantity = 3)
  (h2 : sales.crackers_price = 9/4)
  (h3 : sales.beverage_quantity = 4)
  (h4 : sales.beverage_price = 3/2)
  (h5 : sales.chocolate_quantity = 4)
  (h6 : sales.chocolate_price = 1) :
  average_snack_sales_per_ticket sales = 279/100 := by
  sorry

end NUMINAMATH_CALUDE_average_snack_sales_theorem_l1319_131950


namespace NUMINAMATH_CALUDE_dimes_percentage_l1319_131975

theorem dimes_percentage (num_nickels num_dimes : ℕ) 
  (nickel_value dime_value : ℕ) : 
  num_nickels = 40 → 
  num_dimes = 30 → 
  nickel_value = 5 → 
  dime_value = 10 → 
  (num_dimes * dime_value : ℚ) / 
  (num_nickels * nickel_value + num_dimes * dime_value) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_dimes_percentage_l1319_131975


namespace NUMINAMATH_CALUDE_root_product_l1319_131927

def f (x : ℂ) : ℂ := x^5 - x^3 + x + 1

def g (x : ℂ) : ℂ := x^2 - 3

theorem root_product (x₁ x₂ x₃ x₄ x₅ : ℂ) 
  (hf : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = 146 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l1319_131927


namespace NUMINAMATH_CALUDE_students_favor_both_issues_l1319_131904

theorem students_favor_both_issues (total : ℕ) (favor_first : ℕ) (favor_second : ℕ) (against_both : ℕ)
  (h1 : total = 500)
  (h2 : favor_first = 375)
  (h3 : favor_second = 275)
  (h4 : against_both = 40) :
  total - against_both = favor_first + favor_second - 190 := by
  sorry

end NUMINAMATH_CALUDE_students_favor_both_issues_l1319_131904


namespace NUMINAMATH_CALUDE_marathon_time_l1319_131934

/-- Calculates the total time to complete a marathon given specific conditions -/
theorem marathon_time (total_distance : ℝ) (initial_distance : ℝ) (initial_time : ℝ) (remaining_pace_factor : ℝ) :
  total_distance = 26 →
  initial_distance = 10 →
  initial_time = 1 →
  remaining_pace_factor = 0.8 →
  let initial_pace := initial_distance / initial_time
  let remaining_distance := total_distance - initial_distance
  let remaining_pace := initial_pace * remaining_pace_factor
  let remaining_time := remaining_distance / remaining_pace
  initial_time + remaining_time = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_marathon_time_l1319_131934


namespace NUMINAMATH_CALUDE_expression_simplification_l1319_131923

theorem expression_simplification (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  (((x / (x + b)) + (b / (x - b))) / ((b / (x + b)) - (x / (x - b)))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1319_131923


namespace NUMINAMATH_CALUDE_badminton_cost_theorem_l1319_131996

/-- Represents the cost of badminton equipment under different purchasing options -/
def BadmintonCost (x : ℕ) : Prop :=
  let racket_price : ℕ := 40
  let shuttlecock_price : ℕ := 10
  let racket_quantity : ℕ := 10
  let shuttlecock_quantity : ℕ := x
  let option1_cost : ℕ := 10 * x + 300
  let option2_cost : ℕ := 9 * x + 360
  x > 10 ∧
  option1_cost = racket_price * racket_quantity + shuttlecock_price * (shuttlecock_quantity - racket_quantity) ∧
  option2_cost = (racket_price * racket_quantity + shuttlecock_price * shuttlecock_quantity) * 9 / 10 ∧
  (x = 30 → option1_cost < option2_cost) ∧
  ∃ (better_cost : ℕ), x = 30 → better_cost < option1_cost ∧ better_cost < option2_cost

theorem badminton_cost_theorem : 
  ∀ x : ℕ, BadmintonCost x :=
sorry

end NUMINAMATH_CALUDE_badminton_cost_theorem_l1319_131996


namespace NUMINAMATH_CALUDE_purchase_decision_l1319_131981

/-- Represents the prices and conditions for the company's purchase decision --/
structure PurchaseScenario where
  tablet_price : ℝ
  speaker_price : ℝ
  total_items : ℕ
  discount_rate1 : ℝ
  discount_threshold2 : ℝ
  discount_rate2 : ℝ

/-- Theorem stating the correct prices and cost-effective decision based on the number of tablets --/
theorem purchase_decision (p : PurchaseScenario) 
  (h1 : 2 * p.tablet_price + 3 * p.speaker_price = 7600)
  (h2 : 3 * p.tablet_price = 5 * p.speaker_price)
  (h3 : p.total_items = 30)
  (h4 : p.discount_rate1 = 0.1)
  (h5 : p.discount_threshold2 = 24000)
  (h6 : p.discount_rate2 = 0.2) :
  p.tablet_price = 2000 ∧ 
  p.speaker_price = 1200 ∧ 
  (∀ a : ℕ, a < 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) < 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) ∧
  (∀ a : ℕ, a = 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) = 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) ∧
  (∀ a : ℕ, a > 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) > 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) :=
by sorry

end NUMINAMATH_CALUDE_purchase_decision_l1319_131981


namespace NUMINAMATH_CALUDE_megan_folders_l1319_131931

/-- Given the initial number of files, number of deleted files, and files per folder,
    calculate the number of folders needed to store all remaining files. -/
def folders_needed (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  let remaining_files := initial_files - deleted_files
  (remaining_files + files_per_folder - 1) / files_per_folder

/-- Prove that given 237 initial files, 53 deleted files, and 12 files per folder,
    the number of folders needed is 16. -/
theorem megan_folders : folders_needed 237 53 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l1319_131931


namespace NUMINAMATH_CALUDE_equation_solution_l1319_131978

theorem equation_solution : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1319_131978


namespace NUMINAMATH_CALUDE_inequality_condition_l1319_131954

theorem inequality_condition :
  (∀ a b c d : ℝ, (a > b ∧ c > d) → (a + c > b + d)) ∧
  (∃ a b c d : ℝ, (a + c > b + d) ∧ ¬(a > b ∧ c > d)) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l1319_131954


namespace NUMINAMATH_CALUDE_combined_average_age_l1319_131920

theorem combined_average_age (x_count y_count z_count : ℕ) 
  (x_avg y_avg z_avg : ℝ) : 
  x_count = 5 → 
  y_count = 3 → 
  z_count = 2 → 
  x_avg = 35 → 
  y_avg = 30 → 
  z_avg = 45 → 
  (x_count * x_avg + y_count * y_avg + z_count * z_avg) / (x_count + y_count + z_count) = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l1319_131920


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l1319_131915

theorem students_playing_neither_sport
  (total : ℕ)
  (football : ℕ)
  (tennis : ℕ)
  (both : ℕ)
  (h1 : total = 50)
  (h2 : football = 32)
  (h3 : tennis = 28)
  (h4 : both = 22) :
  total - (football + tennis - both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l1319_131915


namespace NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l1319_131956

def zero_sequence : ℕ → ℝ := λ _ => 0

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem zero_sequence_arithmetic_not_geometric :
  is_arithmetic_sequence zero_sequence ∧ ¬is_geometric_sequence zero_sequence := by
  sorry

end NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l1319_131956


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l1319_131969

theorem shirt_cost_problem (total_cost : ℕ) (num_shirts : ℕ) (num_known_shirts : ℕ) (known_shirt_cost : ℕ) (h1 : total_cost = 85) (h2 : num_shirts = 5) (h3 : num_known_shirts = 3) (h4 : known_shirt_cost = 15) :
  let remaining_shirts := num_shirts - num_known_shirts
  let remaining_cost := total_cost - (num_known_shirts * known_shirt_cost)
  remaining_cost / remaining_shirts = 20 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l1319_131969


namespace NUMINAMATH_CALUDE_expansion_sum_coefficients_l1319_131917

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the expression
def expansion_sum (x : ℕ) : ℕ :=
  (binomial_coeff x 1 + binomial_coeff x 2 + binomial_coeff x 3 + binomial_coeff x 4) ^ 2

-- Theorem statement
theorem expansion_sum_coefficients :
  ∃ x, expansion_sum x = 225 := by sorry

end NUMINAMATH_CALUDE_expansion_sum_coefficients_l1319_131917


namespace NUMINAMATH_CALUDE_analogical_conclusions_correctness_l1319_131922

theorem analogical_conclusions_correctness :
  (∃! i : Fin 3, 
    (i = 0 → ∀ (a b : ℝ) (n : ℕ), (a + b)^n = a^n + b^n) ∨
    (i = 1 → ∀ (α β : ℝ), Real.sin (α + β) = Real.sin α * Real.sin β) ∨
    (i = 2 → ∀ (a b : ℝ), (a + b)^2 = a^2 + 2*a*b + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_analogical_conclusions_correctness_l1319_131922


namespace NUMINAMATH_CALUDE_power_mod_eleven_l1319_131925

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l1319_131925


namespace NUMINAMATH_CALUDE_number_puzzle_l1319_131942

theorem number_puzzle : ∃! x : ℝ, (x / 12) * 24 = x + 36 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1319_131942


namespace NUMINAMATH_CALUDE_min_students_in_both_clubs_l1319_131928

theorem min_students_in_both_clubs
  (total_students : ℕ)
  (club1_students : ℕ)
  (club2_students : ℕ)
  (h1 : total_students = 33)
  (h2 : club1_students ≥ 24)
  (h3 : club2_students ≥ 24) :
  ∃ (intersection : ℕ), intersection ≥ 15 ∧
    intersection ≤ min club1_students club2_students ∧
    intersection ≤ total_students :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_both_clubs_l1319_131928


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l1319_131945

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010_is_10 :
  binary_to_decimal [false, true, false, true] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l1319_131945


namespace NUMINAMATH_CALUDE_sum_of_squares_is_four_l1319_131966

/-- Represents a rectangle ABCD with an inscribed ellipse K and a point P on K. -/
structure RectangleWithEllipse where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Angle parameter for point P on the ellipse -/
  θ : ℝ
  /-- AB = 2 -/
  h_ab : ab = 2
  /-- AD < √2 -/
  h_ad : ad < Real.sqrt 2

/-- The sum of squares of AM and LB is always 4 -/
theorem sum_of_squares_is_four (rect : RectangleWithEllipse) :
  let x_M := (Real.sqrt 2 * (Real.cos rect.θ - 1)) / (Real.sqrt 2 - Real.sin rect.θ) + 1
  let x_L := (Real.sqrt 2 * (1 + Real.cos rect.θ)) / (Real.sqrt 2 - Real.sin rect.θ) - 1
  (1 + x_M)^2 + (1 - x_L)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_four_l1319_131966


namespace NUMINAMATH_CALUDE_intersection_and_inequality_l1319_131905

theorem intersection_and_inequality (m : ℝ) : 
  (∃ x, 2 * x + m = 0 ∧ x = -1) → 
  (∀ x, 2 * x + m ≤ 0 ↔ x ≤ -1) := by
sorry

end NUMINAMATH_CALUDE_intersection_and_inequality_l1319_131905


namespace NUMINAMATH_CALUDE_three_distinct_roots_transformation_l1319_131940

/-- Given an equation a x^5 + b x^4 + c = 0 with three distinct roots,
    prove that c x^5 + b x + a = 0 also has three distinct roots -/
theorem three_distinct_roots_transformation (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    a * x^5 + b * x^4 + c = 0 ∧
    a * y^5 + b * y^4 + c = 0 ∧
    a * z^5 + b * z^4 + c = 0) →
  (∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c * u^5 + b * u + a = 0 ∧
    c * v^5 + b * v + a = 0 ∧
    c * w^5 + b * w + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_roots_transformation_l1319_131940


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l1319_131939

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l1319_131939


namespace NUMINAMATH_CALUDE_min_quotient_value_l1319_131926

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0) ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem min_quotient_value :
  ∀ n : ℕ, is_valid_number n → (n : ℚ) / (digit_sum n : ℚ) ≥ 105 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_value_l1319_131926


namespace NUMINAMATH_CALUDE_sphere_division_l1319_131912

/-- The maximum number of parts into which the surface of a sphere can be divided by n great circles -/
def max_sphere_parts (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating that max_sphere_parts gives the correct number of maximum parts -/
theorem sphere_division (n : ℕ) :
  max_sphere_parts n = n^2 - n + 2 := by sorry

end NUMINAMATH_CALUDE_sphere_division_l1319_131912


namespace NUMINAMATH_CALUDE_ordering_abc_l1319_131943

theorem ordering_abc (a b c : ℝ) (ha : a = 1.01^(1/2 : ℝ)) (hb : b = 1.01^(3/5 : ℝ)) (hc : c = 0.6^(1/2 : ℝ)) : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l1319_131943


namespace NUMINAMATH_CALUDE_bus_students_l1319_131916

theorem bus_students (initial : Real) (got_on : Real) (total : Real) : 
  initial = 10.0 → got_on = 3.0 → total = initial + got_on → total = 13.0 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_l1319_131916


namespace NUMINAMATH_CALUDE_union_of_M_and_N_N_is_possible_set_l1319_131914

def M : Set ℕ := {1, 2}
def N : Set ℕ := {1, 3}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3} := by sorry

theorem N_is_possible_set :
  M = {1, 2} → M ∪ N = {1, 2, 3} → N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_N_is_possible_set_l1319_131914


namespace NUMINAMATH_CALUDE_books_left_over_l1319_131997

theorem books_left_over (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1335 →
  books_per_initial_box = 39 →
  books_per_new_box = 40 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 25 := by
  sorry

end NUMINAMATH_CALUDE_books_left_over_l1319_131997


namespace NUMINAMATH_CALUDE_cubic_expansion_result_l1319_131909

theorem cubic_expansion_result (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 3 * x - Real.sqrt 2)^3 = a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_result_l1319_131909


namespace NUMINAMATH_CALUDE_solve_bus_problem_l1319_131919

def bus_problem (initial : ℕ) (stop_a_off stop_a_on : ℕ) (stop_b_off stop_b_on : ℕ) 
                 (stop_c_off stop_c_on : ℕ) (stop_d_off : ℕ) (final : ℕ) : Prop :=
  let after_a := initial - stop_a_off + stop_a_on
  let after_b := after_a - stop_b_off + stop_b_on
  let after_c := after_b - stop_c_off + stop_c_on
  let after_d := after_c - stop_d_off
  ∃ (stop_d_on : ℕ), after_d + stop_d_on = final ∧ stop_d_on = 10

theorem solve_bus_problem : 
  bus_problem 64 8 12 4 6 14 22 10 78 := by
  sorry

end NUMINAMATH_CALUDE_solve_bus_problem_l1319_131919


namespace NUMINAMATH_CALUDE_man_son_age_difference_l1319_131941

/-- Given a man and his son, where the son's present age is 16, and in two years
    the man's age will be twice the age of his son, prove that the man is 18 years
    older than his son. -/
theorem man_son_age_difference (man_age son_age : ℕ) : 
  son_age = 16 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l1319_131941


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1319_131906

/-- Parabola E with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line intersecting the parabola -/
structure IntersectingLine where
  k : ℝ

/-- Point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem about parabola intersection and slope relations -/
theorem parabola_intersection_theorem 
  (E : Parabola) 
  (L : IntersectingLine) 
  (A B : ParabolaPoint) 
  (h_on_parabola : A.x^2 = 2*E.p*A.y ∧ B.x^2 = 2*E.p*B.y)
  (h_on_line : A.y = L.k*A.x + 2 ∧ B.y = L.k*B.x + 2)
  (h_dot_product : A.x*B.x + A.y*B.y = 2) :
  (∃ (k₁ k₂ : ℝ), 
    k₁ = (A.y + 2) / A.x ∧ 
    k₂ = (B.y + 2) / B.x ∧ 
    k₁^2 + k₂^2 - 2*L.k^2 = 16) ∧
  E.p = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1319_131906


namespace NUMINAMATH_CALUDE_man_speed_man_speed_specific_l1319_131993

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : Real) (train_speed_kmh : Real) (pass_time : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / pass_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man given specific values -/
theorem man_speed_specific : 
  man_speed 150 83.99280057595394 6 = 6.007199827245052 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_specific_l1319_131993


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1319_131911

/-- The standard equation of a hyperbola with foci on the y-axis, given a + c = 9 and b = 3 -/
theorem hyperbola_standard_equation (a c b : ℝ) (h1 : a + c = 9) (h2 : b = 3) :
  ∃ (x y : ℝ), y^2 / 16 - x^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1319_131911


namespace NUMINAMATH_CALUDE_difference_of_squares_l1319_131995

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1319_131995


namespace NUMINAMATH_CALUDE_orange_pounds_value_l1319_131938

/-- The price of a dozen eggs -/
def egg_price : ℝ := sorry

/-- The price per pound of oranges -/
def orange_price : ℝ := sorry

/-- The number of pounds of oranges -/
def orange_pounds : ℝ := sorry

/-- The current prices of eggs and oranges are equal -/
axiom price_equality : egg_price = orange_price * orange_pounds

/-- The price increase equation -/
axiom price_increase : 0.09 * egg_price + 0.06 * orange_price * orange_pounds = 15

theorem orange_pounds_value : orange_pounds = 100 := by sorry

end NUMINAMATH_CALUDE_orange_pounds_value_l1319_131938


namespace NUMINAMATH_CALUDE_a_4_equals_18_l1319_131907

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |>.sum

theorem a_4_equals_18 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = sequence_sum a n) →
  a 1 = 1 →
  (∀ n : ℕ+, a (n + 1) = 2 * S n) →
  a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_a_4_equals_18_l1319_131907


namespace NUMINAMATH_CALUDE_guppy_ratio_l1319_131935

/-- The number of guppies Haylee has -/
def hayleeGuppies : ℕ := 36

/-- The number of guppies Jose has -/
def joseGuppies : ℕ := hayleeGuppies / 2

/-- The number of guppies Charliz has -/
def charlizGuppies : ℕ := 6

/-- The number of guppies Nicolai has -/
def nicolaiGuppies : ℕ := 4 * charlizGuppies

/-- The total number of guppies all four friends have -/
def totalGuppies : ℕ := 84

/-- Theorem stating that the ratio of Charliz's guppies to Jose's guppies is 1:3 -/
theorem guppy_ratio :
  charlizGuppies * 3 = joseGuppies ∧
  hayleeGuppies + joseGuppies + charlizGuppies + nicolaiGuppies = totalGuppies :=
by sorry

end NUMINAMATH_CALUDE_guppy_ratio_l1319_131935


namespace NUMINAMATH_CALUDE_problem_solution_l1319_131968

theorem problem_solution (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1319_131968


namespace NUMINAMATH_CALUDE_sector_max_area_l1319_131983

/-- The maximum area of a sector with circumference 40 is 100 -/
theorem sector_max_area (C : ℝ) (h : C = 40) : 
  ∃ (A : ℝ), A = 100 ∧ ∀ (r θ : ℝ), r > 0 → θ > 0 → r * θ + 2 * r = C → 
    (1/2 : ℝ) * r^2 * θ ≤ A := by sorry

end NUMINAMATH_CALUDE_sector_max_area_l1319_131983


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_and_validity_l1319_131930

/-- Given a triangle with sides a, b, c, and a similar triangle with sides k*a, k*b, k*c and perimeter p,
    prove that the longest side of the similar triangle is 60 and that it forms a valid triangle. -/
theorem similar_triangle_longest_side_and_validity 
  (a b c : ℝ) 
  (h_original : a = 8 ∧ b = 10 ∧ c = 12) 
  (k : ℝ) 
  (h_perimeter : k*a + k*b + k*c = 150) :
  (max (k*a) (max (k*b) (k*c)) = 60) ∧ 
  (k*a + k*b > k*c ∧ k*a + k*c > k*b ∧ k*b + k*c > k*a) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_and_validity_l1319_131930


namespace NUMINAMATH_CALUDE_entrance_exam_questions_entrance_exam_questions_is_70_l1319_131924

/-- Proves that the total number of questions in an entrance exam is 70,
    given the specified scoring system and student performance. -/
theorem entrance_exam_questions : ℕ :=
  let correct_marks : ℕ := 3
  let wrong_marks : ℤ := -1
  let total_score : ℤ := 38
  let correct_answers : ℕ := 27
  let total_questions : ℕ := 70
  
  have h1 : (correct_answers : ℤ) * correct_marks + 
            (total_questions - correct_answers : ℤ) * wrong_marks = total_score := by sorry
  
  total_questions

/-- The proof that the number of questions in the entrance exam is 70. -/
theorem entrance_exam_questions_is_70 : entrance_exam_questions = 70 := by sorry

end NUMINAMATH_CALUDE_entrance_exam_questions_entrance_exam_questions_is_70_l1319_131924


namespace NUMINAMATH_CALUDE_sample_size_is_ten_l1319_131933

/-- Represents a collection of products -/
structure ProductCollection where
  total : Nat
  selected : Nat
  random_selection : selected ≤ total

/-- Definition of sample size for a product collection -/
def sample_size (pc : ProductCollection) : Nat := pc.selected

/-- Theorem: For a product collection with 80 total products and 10 randomly selected,
    the sample size is 10 -/
theorem sample_size_is_ten (pc : ProductCollection) 
  (h1 : pc.total = 80) 
  (h2 : pc.selected = 10) : 
  sample_size pc = 10 := by
  sorry


end NUMINAMATH_CALUDE_sample_size_is_ten_l1319_131933


namespace NUMINAMATH_CALUDE_min_n_value_l1319_131955

/-- The set A containing numbers from 1 to 6 -/
def A : Finset ℕ := Finset.range 6

/-- The set B containing numbers from 7 to n -/
def B (n : ℕ) : Finset ℕ := Finset.Icc 7 n

/-- A function that generates a set A_i -/
def generate_A_i (i : ℕ) (n : ℕ) : Finset ℕ := sorry

/-- The proposition that all A_i sets satisfy the given conditions -/
def valid_A_i_sets (n : ℕ) : Prop :=
  ∃ (f : ℕ → Finset ℕ), 
    (∀ i, i ≤ 20 → (f i).card = 8) ∧
    (∀ i, i ≤ 20 → (f i ∩ A).card = 3) ∧
    (∀ i, i ≤ 20 → (f i ∩ B n).card = 5) ∧
    (∀ i j, i < j ∧ j ≤ 20 → (f i ∩ f j).card ≤ 2)

theorem min_n_value : 
  (∀ n < 41, ¬ valid_A_i_sets n) ∧ valid_A_i_sets 41 := by sorry

end NUMINAMATH_CALUDE_min_n_value_l1319_131955


namespace NUMINAMATH_CALUDE_sports_equipment_store_problem_l1319_131948

/-- Sports equipment store problem -/
theorem sports_equipment_store_problem 
  (total_balls : ℕ) 
  (budget : ℕ) 
  (basketball_cost : ℕ) 
  (volleyball_cost : ℕ) 
  (basketball_price_ratio : ℚ) 
  (school_basketball_purchase : ℕ) 
  (school_volleyball_purchase : ℕ) 
  (school_basketball_count : ℕ) 
  (school_volleyball_count : ℕ) :
  total_balls = 200 →
  budget ≤ 5000 →
  basketball_cost = 30 →
  volleyball_cost = 24 →
  basketball_price_ratio = 3/2 →
  school_basketball_purchase = 1800 →
  school_volleyball_purchase = 1500 →
  school_volleyball_count = school_basketball_count + 10 →
  ∃ (basketball_price volleyball_price : ℕ) 
    (optimal_basketball optimal_volleyball : ℕ),
    basketball_price = 45 ∧
    volleyball_price = 30 ∧
    optimal_basketball = 33 ∧
    optimal_volleyball = 167 ∧
    optimal_basketball + optimal_volleyball = total_balls ∧
    optimal_basketball * basketball_cost + optimal_volleyball * volleyball_cost ≤ budget ∧
    ∀ (b v : ℕ), 
      b + v = total_balls →
      b * basketball_cost + v * volleyball_cost ≤ budget →
      (basketball_price - 3 - basketball_cost) * b + (volleyball_price - 2 - volleyball_cost) * v ≤
      (basketball_price - 3 - basketball_cost) * optimal_basketball + 
      (volleyball_price - 2 - volleyball_cost) * optimal_volleyball :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_store_problem_l1319_131948


namespace NUMINAMATH_CALUDE_complex_power_multiply_l1319_131900

theorem complex_power_multiply (i : ℂ) : i^2 = -1 → i^13 * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_multiply_l1319_131900


namespace NUMINAMATH_CALUDE_equation_solution_l1319_131970

theorem equation_solution : ∃ X : ℝ, 
  (1.5 * ((X * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002) ∧ 
  (abs (X - 3.6000000000000005) < 1e-10) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1319_131970


namespace NUMINAMATH_CALUDE_total_students_l1319_131982

theorem total_students (middle_school : ℕ) (elementary_school : ℕ) (high_school : ℕ) : 
  middle_school = 50 → 
  elementary_school = 4 * middle_school - 3 → 
  high_school = 2 * elementary_school → 
  elementary_school + middle_school + high_school = 641 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l1319_131982


namespace NUMINAMATH_CALUDE_angle_QRS_is_150_degrees_l1319_131963

/-- A quadrilateral PQRS with specific side lengths and angles -/
structure Quadrilateral :=
  (PQ : ℝ)
  (RS : ℝ)
  (PS : ℝ)
  (angle_QPS : ℝ)
  (angle_RSP : ℝ)

/-- The theorem stating the condition for ∠QRS in the given quadrilateral -/
theorem angle_QRS_is_150_degrees (q : Quadrilateral) 
  (h1 : q.PQ = 40)
  (h2 : q.RS = 20)
  (h3 : q.PS = 60)
  (h4 : q.angle_QPS = 60)
  (h5 : q.angle_RSP = 60) :
  ∃ (angle_QRS : ℝ), angle_QRS = 150 := by
  sorry

end NUMINAMATH_CALUDE_angle_QRS_is_150_degrees_l1319_131963


namespace NUMINAMATH_CALUDE_max_min_difference_l1319_131987

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
               (∀ x ∈ I, m ≤ f x) ∧ 
               (∃ x₁ ∈ I, f x₁ = M) ∧ 
               (∃ x₂ ∈ I, f x₂ = m) ∧ 
               M - m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l1319_131987


namespace NUMINAMATH_CALUDE_complement_intersection_eq_singleton_l1319_131951

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq_singleton :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_eq_singleton_l1319_131951


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1319_131960

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x ≥ x + 1) ↔ (∃ x₀ : ℝ, Real.exp x₀ < x₀ + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1319_131960


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1319_131957

theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 32 ∧ ∀ (x : ℝ), a * x * (x - 2)^2 ≤ M) →
  a = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1319_131957


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l1319_131901

theorem rachel_apple_picking (trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : 
  trees = 4 → 
  apples_per_tree = 7 → 
  remaining_apples = 29 → 
  trees * apples_per_tree = 28 :=
by sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l1319_131901


namespace NUMINAMATH_CALUDE_judy_spending_l1319_131949

-- Define the prices and quantities
def carrot_price : ℚ := 1
def carrot_quantity : ℕ := 8
def milk_price : ℚ := 3
def milk_quantity : ℕ := 4
def pineapple_regular_price : ℚ := 4
def pineapple_quantity : ℕ := 3
def flour_price : ℚ := 5
def flour_quantity : ℕ := 3
def ice_cream_price : ℚ := 7
def ice_cream_quantity : ℕ := 2

-- Define the discount conditions
def discount_threshold : ℚ := 50
def discount_rate : ℚ := 0.1
def coupon_value : ℚ := 5
def coupon_threshold : ℚ := 30

-- Calculate the total before discounts
def total_before_discounts : ℚ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  (pineapple_regular_price / 2) * pineapple_quantity +
  flour_price * flour_quantity +
  ice_cream_price * ice_cream_quantity

-- Apply discounts
def final_total : ℚ :=
  let discounted_total := 
    if total_before_discounts > discount_threshold
    then total_before_discounts * (1 - discount_rate)
    else total_before_discounts
  if discounted_total ≥ coupon_threshold
  then discounted_total - coupon_value
  else discounted_total

-- Theorem statement
theorem judy_spending : final_total = 44.5 := by sorry

end NUMINAMATH_CALUDE_judy_spending_l1319_131949
