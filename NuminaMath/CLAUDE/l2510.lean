import Mathlib

namespace NUMINAMATH_CALUDE_projection_relations_l2510_251036

-- Define a plane
structure Plane where
  -- Add necessary fields

-- Define a line
structure Line where
  -- Add necessary fields

-- Define the projection of a line onto a plane
def project (l : Line) (p : Plane) : Line :=
  sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define coincident lines
def coincident (l1 l2 : Line) : Prop :=
  sorry

theorem projection_relations (α : Plane) (m n : Line) :
  let m1 := project m α
  let n1 := project n α
  -- All four propositions are false
  (¬ (parallel m1 n1 → parallel m n)) ∧
  (¬ (parallel m n → (parallel m1 n1 ∨ coincident m1 n1))) ∧
  (¬ (perpendicular m1 n1 → perpendicular m n)) ∧
  (¬ (perpendicular m n → perpendicular m1 n1)) :=
by
  sorry

end NUMINAMATH_CALUDE_projection_relations_l2510_251036


namespace NUMINAMATH_CALUDE_image_of_two_is_five_l2510_251066

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem image_of_two_is_five : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_is_five_l2510_251066


namespace NUMINAMATH_CALUDE_greatest_number_of_baskets_l2510_251056

theorem greatest_number_of_baskets (oranges pears bananas : ℕ) 
  (h_oranges : oranges = 18) 
  (h_pears : pears = 27) 
  (h_bananas : bananas = 12) : 
  (Nat.gcd oranges (Nat.gcd pears bananas)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_baskets_l2510_251056


namespace NUMINAMATH_CALUDE_pyramid_theorem_l2510_251018

structure Pyramid where
  S₁ : ℝ  -- Area of face ABD
  S₂ : ℝ  -- Area of face BCD
  S₃ : ℝ  -- Area of face CAD
  Q : ℝ   -- Area of face ABC
  α : ℝ   -- Dihedral angle at edge AB
  β : ℝ   -- Dihedral angle at edge BC
  γ : ℝ   -- Dihedral angle at edge AC
  h₁ : S₁ > 0
  h₂ : S₂ > 0
  h₃ : S₃ > 0
  h₄ : Q > 0
  h₅ : 0 < α ∧ α < π
  h₆ : 0 < β ∧ β < π
  h₇ : 0 < γ ∧ γ < π
  h₈ : Real.cos α = S₁ / Q
  h₉ : Real.cos β = S₂ / Q
  h₁₀ : Real.cos γ = S₃ / Q

theorem pyramid_theorem (p : Pyramid) : 
  p.S₁^2 + p.S₂^2 + p.S₃^2 = p.Q^2 ∧ 
  Real.cos (2 * p.α) + Real.cos (2 * p.β) + Real.cos (2 * p.γ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l2510_251018


namespace NUMINAMATH_CALUDE_max_regular_lines_six_points_l2510_251068

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Enumeration of possible regular line types -/
inductive RegularLineType
  | Horizontal
  | Vertical
  | LeftDiagonal
  | RightDiagonal

/-- A regular line in the 2D plane -/
structure RegularLine where
  type : RegularLineType
  offset : ℝ

/-- Function to check if a point lies on a regular line -/
def pointOnRegularLine (p : Point2D) (l : RegularLine) : Prop :=
  match l.type with
  | RegularLineType.Horizontal => p.y = l.offset
  | RegularLineType.Vertical => p.x = l.offset
  | RegularLineType.LeftDiagonal => p.y - p.x = l.offset
  | RegularLineType.RightDiagonal => p.y + p.x = l.offset

/-- The main theorem stating the maximum number of regular lines -/
theorem max_regular_lines_six_points (points : Fin 6 → Point2D) :
  (∃ (lines : Finset RegularLine), 
    (∀ l ∈ lines, ∃ i j, i ≠ j ∧ pointOnRegularLine (points i) l ∧ pointOnRegularLine (points j) l) ∧
    lines.card = 11) ∧
  (∀ (lines : Finset RegularLine),
    (∀ l ∈ lines, ∃ i j, i ≠ j ∧ pointOnRegularLine (points i) l ∧ pointOnRegularLine (points j) l) →
    lines.card ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_max_regular_lines_six_points_l2510_251068


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l2510_251042

theorem divisible_by_thirteen (n : ℤ) : 
  13 ∣ (n^2 - 6*n - 4) ↔ n ≡ 3 [ZMOD 13] := by
sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l2510_251042


namespace NUMINAMATH_CALUDE_unique_solution_l2510_251081

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 1

/-- The theorem stating that the function g(x) = 2x + 3 is the unique solution -/
theorem unique_solution :
  ∀ g : ℝ → ℝ, FunctionalEquation g → (∀ x : ℝ, g x = 2 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2510_251081


namespace NUMINAMATH_CALUDE_smallest_positive_a_for_parabola_l2510_251084

theorem smallest_positive_a_for_parabola :
  ∀ (a b c : ℚ),
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ y + 5/4 = a * (x - 1/2)^2) →
  a > 0 →
  ∃ n : ℤ, a + b + c = n →
  (∀ a' : ℚ, a' > 0 → (∃ b' c' : ℚ, (∀ x y : ℚ, y = a' * x^2 + b' * x + c' ↔ y + 5/4 = a' * (x - 1/2)^2) ∧ 
                      (∃ n' : ℤ, a' + b' + c' = n')) → a' ≥ a) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_a_for_parabola_l2510_251084


namespace NUMINAMATH_CALUDE_power_product_equality_l2510_251012

theorem power_product_equality : 3^5 * 7^5 = 4084101 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2510_251012


namespace NUMINAMATH_CALUDE_book_arrangement_combinations_l2510_251041

-- Define the number of each type of book
def geometry_books : ℕ := 4
def number_theory_books : ℕ := 5

-- Define the total number of books
def total_books : ℕ := geometry_books + number_theory_books

-- Define the number of remaining spots after placing the first geometry book
def remaining_spots : ℕ := total_books - 1

-- Define the number of remaining geometry books to place
def remaining_geometry_books : ℕ := geometry_books - 1

-- Theorem statement
theorem book_arrangement_combinations :
  (remaining_spots.choose remaining_geometry_books) = 56 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_combinations_l2510_251041


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_longest_obtuse_triangle_longest_side_l2510_251017

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < α ∧ 0 < β ∧ 0 < γ
  h_sum_angles : α + β + γ = π

-- Theorem for right-angled triangle
theorem right_triangle_hypotenuse_longest (t : Triangle) (h_right : t.γ = π/2) :
  t.c ≥ t.a ∧ t.c ≥ t.b := by sorry

-- Theorem for obtuse-angled triangle
theorem obtuse_triangle_longest_side (t : Triangle) (h_obtuse : t.γ > π/2) :
  t.c > t.a ∧ t.c > t.b := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_longest_obtuse_triangle_longest_side_l2510_251017


namespace NUMINAMATH_CALUDE_find_number_l2510_251065

theorem find_number : ∃! x : ℕ+, 
  (172 / x.val : ℚ) = 172 / 4 - 28 ∧ 
  172 % x.val = 7 ∧ 
  x = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2510_251065


namespace NUMINAMATH_CALUDE_april_coffee_cost_l2510_251059

/-- The number of coffees Jon buys per day -/
def coffees_per_day : ℕ := 2

/-- The cost of one coffee in dollars -/
def cost_per_coffee : ℕ := 2

/-- The number of days in April -/
def days_in_april : ℕ := 30

/-- The total cost of coffee for Jon in April -/
def total_cost : ℕ := coffees_per_day * cost_per_coffee * days_in_april

theorem april_coffee_cost : total_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_april_coffee_cost_l2510_251059


namespace NUMINAMATH_CALUDE_fraction_sum_minus_five_equals_negative_four_l2510_251030

theorem fraction_sum_minus_five_equals_negative_four (a b : ℝ) (h : a ≠ b) :
  a / (a - b) + b / (b - a) - 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_minus_five_equals_negative_four_l2510_251030


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2510_251082

theorem polygon_sides_count (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 3 * 360 → n = 8 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2510_251082


namespace NUMINAMATH_CALUDE_sin_bounded_difference_l2510_251061

theorem sin_bounded_difference (a : ℝ) : 
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → |Real.sin x - a| ≤ |Real.sin x₀ - a| := by
  sorry

end NUMINAMATH_CALUDE_sin_bounded_difference_l2510_251061


namespace NUMINAMATH_CALUDE_min_sum_squares_l2510_251073

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + z = 1) : 
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + c = 1 → a^2 + b^2 + c^2 ≥ m) ∧ 
  (∃ p q r : ℝ, p + 2*q + r = 1 ∧ p^2 + q^2 + r^2 = m) ∧ 
  m = 1/6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2510_251073


namespace NUMINAMATH_CALUDE_prime_power_sum_l2510_251007

theorem prime_power_sum (p q : Nat) (m n : Nat) : 
  Nat.Prime p → Nat.Prime q → p < q →
  (∃ c : Nat, (p^(m+1) - 1) / (p - 1) = q^c) →
  (∃ d : Nat, (q^(n+1) - 1) / (q - 1) = p^d) →
  (p = 2 ∧ ∃ t : Nat, Nat.Prime t ∧ q = 2^t - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2510_251007


namespace NUMINAMATH_CALUDE_forty_is_twenty_percent_of_two_hundred_l2510_251070

theorem forty_is_twenty_percent_of_two_hundred (x : ℝ) : 40 = (20 / 100) * x → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_forty_is_twenty_percent_of_two_hundred_l2510_251070


namespace NUMINAMATH_CALUDE_smallest_radii_sum_squares_l2510_251079

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Check if four points lie on a circle -/
def onCircle (A B C D : Point) : Prop := sorry

/-- The theorem to be proved -/
theorem smallest_radii_sum_squares
  (A : Point) (B : Point) (C : Point) (D : Point)
  (h1 : A = ⟨0, 0⟩)
  (h2 : B = ⟨-1, -1⟩)
  (h3 : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ C = ⟨x, y⟩ ∧ D = ⟨x + 1, y⟩)
  (h4 : onCircle A B C D) :
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > r₁ ∧
    (∀ (r : ℝ), onCircle A B C D → r ≥ r₁) ∧
    (∀ (r : ℝ), onCircle A B C D ∧ r ≠ r₁ → r ≥ r₂) ∧
    r₁^2 + r₂^2 = 1381 := by
  sorry

end NUMINAMATH_CALUDE_smallest_radii_sum_squares_l2510_251079


namespace NUMINAMATH_CALUDE_demand_decrease_with_price_increase_l2510_251055

theorem demand_decrease_with_price_increase (P Q : ℝ) (P_new Q_new : ℝ) :
  P > 0 → Q > 0 →
  P_new = 1.5 * P →
  P_new * Q_new = 1.2 * P * Q →
  Q_new = 0.8 * Q :=
by
  sorry

#check demand_decrease_with_price_increase

end NUMINAMATH_CALUDE_demand_decrease_with_price_increase_l2510_251055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_probability_l2510_251049

/-- The set of numbers from which we select -/
def NumberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

/-- A function to check if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop := a + c = 2 * b

/-- The total number of ways to choose 3 numbers from the set -/
def totalSelections : ℕ := Nat.choose 20 3

/-- The number of ways to choose 3 numbers that form an arithmetic sequence -/
def arithmeticSequenceSelections : ℕ := 90

/-- The probability of selecting 3 numbers that form an arithmetic sequence -/
def probability : ℚ := arithmeticSequenceSelections / totalSelections

theorem arithmetic_sequence_probability :
  probability = 3 / 38 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_probability_l2510_251049


namespace NUMINAMATH_CALUDE_max_tangent_segments_2017_l2510_251019

/-- Given a number of circles, calculates the maximum number of tangent segments -/
def max_tangent_segments (n : ℕ) : ℕ := 3 * (n * (n - 1)) / 2

/-- Theorem: The maximum number of tangent segments for 2017 circles is 6,051,252 -/
theorem max_tangent_segments_2017 :
  max_tangent_segments 2017 = 6051252 := by
  sorry

#eval max_tangent_segments 2017

end NUMINAMATH_CALUDE_max_tangent_segments_2017_l2510_251019


namespace NUMINAMATH_CALUDE_rectangle_combination_perimeter_l2510_251087

theorem rectangle_combination_perimeter : ∀ (l w : ℝ),
  l = 4 ∧ w = 2 →
  ∃ (new_l new_w : ℝ),
    ((new_l = l + l ∧ new_w = w) ∨ (new_l = l ∧ new_w = w + w)) ∧
    (2 * (new_l + new_w) = 20 ∨ 2 * (new_l + new_w) = 16) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_combination_perimeter_l2510_251087


namespace NUMINAMATH_CALUDE_regular_polygon_lattice_points_regular_polyhedra_lattice_points_l2510_251043

-- Define a 3D lattice point
def LatticePoint := ℤ × ℤ × ℤ

-- Define a regular polygon
structure RegularPolygon (n : ℕ) where
  vertices : List LatticePoint
  is_regular : Bool  -- This should be a proof in a real implementation
  vertex_count : vertices.length = n

-- Define a regular polyhedron
structure RegularPolyhedron where
  vertices : List LatticePoint
  is_regular : Bool  -- This should be a proof in a real implementation

-- Theorem for regular polygons
theorem regular_polygon_lattice_points :
  ∀ n : ℕ, (∃ p : RegularPolygon n, True) ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

-- Define the Platonic solids
inductive PlatonicSolid
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

-- Function to check if a Platonic solid can have lattice point vertices
def has_lattice_vertices : PlatonicSolid → Prop
  | PlatonicSolid.Tetrahedron => True
  | PlatonicSolid.Cube => True
  | PlatonicSolid.Octahedron => True
  | PlatonicSolid.Dodecahedron => False
  | PlatonicSolid.Icosahedron => False

-- Theorem for regular polyhedra
theorem regular_polyhedra_lattice_points :
  ∀ s : PlatonicSolid, has_lattice_vertices s ↔
    (s = PlatonicSolid.Tetrahedron ∨ s = PlatonicSolid.Cube ∨ s = PlatonicSolid.Octahedron) :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_lattice_points_regular_polyhedra_lattice_points_l2510_251043


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2510_251089

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 784.3 →
  percentage = 28.5 →
  final = initial * (1 + percentage / 100) →
  final = 1007.8255 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2510_251089


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2510_251013

/-- Calculates the length of a platform given train characteristics and crossing time -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmph = 84 →
  crossing_time = 16 →
  ∃ (platform_length : ℝ), abs (platform_length - 233.33) < 0.01 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l2510_251013


namespace NUMINAMATH_CALUDE_mean_car_sales_l2510_251050

def car_sales : List Nat := [8, 3, 10, 4, 4, 4]

theorem mean_car_sales :
  (car_sales.sum : ℚ) / car_sales.length = 5.5 := by sorry

end NUMINAMATH_CALUDE_mean_car_sales_l2510_251050


namespace NUMINAMATH_CALUDE_total_cloud_count_l2510_251076

def cloud_count (carson_count : ℕ) (brother_multiplier : ℕ) (sister_divisor : ℕ) : ℕ :=
  carson_count + (carson_count * brother_multiplier) + (carson_count / sister_divisor)

theorem total_cloud_count :
  cloud_count 12 5 2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_total_cloud_count_l2510_251076


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2510_251057

/-- Given a trader sells cloth with the following conditions:
    - Total length of cloth sold is 75 meters
    - Total selling price is Rs. 4950
    - Profit per meter is Rs. 15
    This theorem proves that the cost price per meter is Rs. 51 -/
theorem cloth_cost_price (total_length : ℕ) (total_selling_price : ℕ) (profit_per_meter : ℕ) :
  total_length = 75 →
  total_selling_price = 4950 →
  profit_per_meter = 15 →
  (total_selling_price - total_length * profit_per_meter) / total_length = 51 :=
by sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2510_251057


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2510_251004

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 5/2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2510_251004


namespace NUMINAMATH_CALUDE_min_value_f_exists_min_f_l2510_251045

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Theorem for the minimum value of f(x)
theorem min_value_f (a : ℝ) :
  (a = 1 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 5) ∧
  (a ≤ -1 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 6 + 2*a) ∧
  (-1 < a ∧ a < 0 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 5 - a^2) :=
by sorry

-- Theorem for the existence of x that achieves the minimum
theorem exists_min_f (a : ℝ) :
  (a = 1 → ∃ x ∈ Set.Icc (-1) 0, f a x = 5) ∧
  (a ≤ -1 → ∃ x ∈ Set.Icc (-1) 0, f a x = 6 + 2*a) ∧
  (-1 < a ∧ a < 0 → ∃ x ∈ Set.Icc (-1) 0, f a x = 5 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_exists_min_f_l2510_251045


namespace NUMINAMATH_CALUDE_sum_of_symmetric_points_l2510_251086

/-- Two points M(a, -3) and N(4, b) are symmetric with respect to the origin -/
def symmetric_points (a b : ℝ) : Prop :=
  (a = -4) ∧ (b = 3)

/-- Theorem: If M(a, -3) and N(4, b) are symmetric with respect to the origin, then a + b = -1 -/
theorem sum_of_symmetric_points (a b : ℝ) (h : symmetric_points a b) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_points_l2510_251086


namespace NUMINAMATH_CALUDE_last_five_digits_of_product_l2510_251037

theorem last_five_digits_of_product : 
  (99 * 10101 * 111 * 1001001) % (100000 : ℕ) = 88889 := by
  sorry

end NUMINAMATH_CALUDE_last_five_digits_of_product_l2510_251037


namespace NUMINAMATH_CALUDE_cylinder_water_properties_l2510_251003

/-- Represents a cylindrical tank lying on its side -/
structure HorizontalCylinder where
  radius : ℝ
  height : ℝ

/-- Represents the water level in the tank -/
def WaterLevel : ℝ := 3

/-- The volume of water in the cylindrical tank -/
def waterVolume (c : HorizontalCylinder) (h : ℝ) : ℝ := sorry

/-- The submerged surface area of the cylindrical side of the tank -/
def submergedSurfaceArea (c : HorizontalCylinder) (h : ℝ) : ℝ := sorry

theorem cylinder_water_properties :
  let c : HorizontalCylinder := { radius := 5, height := 10 }
  (waterVolume c WaterLevel = 290.7 * Real.pi - 40 * Real.sqrt 6) ∧
  (submergedSurfaceArea c WaterLevel = 91.5) := by sorry

end NUMINAMATH_CALUDE_cylinder_water_properties_l2510_251003


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2510_251046

-- Define the binary operation
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Define the properties of the operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) ^ c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) : diamond a a = 1

-- State the theorem
theorem diamond_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ diamond 2048 (diamond 4 x) = 16 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2510_251046


namespace NUMINAMATH_CALUDE_box_weights_l2510_251027

theorem box_weights (a b c : ℝ) 
  (hab : a + b = 122)
  (hbc : b + c = 125)
  (hca : c + a = 127) : 
  a + b + c = 187 := by
  sorry

end NUMINAMATH_CALUDE_box_weights_l2510_251027


namespace NUMINAMATH_CALUDE_orange_harvest_orange_harvest_solution_l2510_251078

theorem orange_harvest (discarded : ℕ) (days : ℕ) (remaining : ℕ) : ℕ :=
  let harvested := (remaining + days * discarded) / days
  harvested

theorem orange_harvest_solution :
  orange_harvest 71 51 153 = 74 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_orange_harvest_solution_l2510_251078


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l2510_251026

theorem continued_fraction_sum (a b c d : ℕ+) :
  (147 : ℚ) / 340 = 1 / (a + 1 / (b + 1 / (c + 1 / d))) →
  (a : ℕ) + b + c + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l2510_251026


namespace NUMINAMATH_CALUDE_third_angle_is_40_l2510_251074

/-- A geometric configuration with an isosceles triangle connected to a right-angled triangle -/
structure GeometricConfig where
  -- Angles of the isosceles triangle
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Angles of the right-angled triangle
  δ : ℝ
  ε : ℝ
  ζ : ℝ

/-- Properties of the geometric configuration -/
def is_valid_config (c : GeometricConfig) : Prop :=
  c.α = 65 ∧ c.β = 65 ∧  -- Two angles of isosceles triangle are 65°
  c.α + c.β + c.γ = 180 ∧  -- Sum of angles in isosceles triangle is 180°
  c.δ = 90 ∧  -- One angle of right-angled triangle is 90°
  c.γ = c.ε ∧  -- Vertically opposite angles are equal
  c.δ + c.ε + c.ζ = 180  -- Sum of angles in right-angled triangle is 180°

/-- Theorem stating that the third angle of the right-angled triangle is 40° -/
theorem third_angle_is_40 (c : GeometricConfig) (h : is_valid_config c) : c.ζ = 40 :=
sorry

end NUMINAMATH_CALUDE_third_angle_is_40_l2510_251074


namespace NUMINAMATH_CALUDE_blasting_safety_condition_l2510_251095

/-- Represents the parameters of a blasting operation safety scenario -/
structure BlastingSafety where
  safetyDistance : ℝ
  fuseSpeed : ℝ
  blasterSpeed : ℝ

/-- Defines the safety condition for a blasting operation -/
def isSafe (params : BlastingSafety) (fuseLength : ℝ) : Prop :=
  fuseLength / params.fuseSpeed > (params.safetyDistance - fuseLength) / params.blasterSpeed

/-- Theorem stating the safety condition for a specific blasting scenario -/
theorem blasting_safety_condition :
  let params : BlastingSafety := {
    safetyDistance := 50,
    fuseSpeed := 0.2,
    blasterSpeed := 3
  }
  ∀ x : ℝ, isSafe params x ↔ x / 0.2 > (50 - x) / 3 := by
  sorry


end NUMINAMATH_CALUDE_blasting_safety_condition_l2510_251095


namespace NUMINAMATH_CALUDE_unit_digit_15_power_100_l2510_251038

theorem unit_digit_15_power_100 : (15 ^ 100) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_15_power_100_l2510_251038


namespace NUMINAMATH_CALUDE_symmetry_proof_l2510_251047

/-- Given two lines in the 2D plane represented by their equations,
    this function returns true if they are symmetric with respect to the line y = -x. -/
def are_symmetric (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 (-y) (-x)

/-- The equation of the original line: 3x - 4y + 5 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the symmetric line: 4x - 3y + 5 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line x + y = 0 -/
theorem symmetry_proof : are_symmetric original_line symmetric_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_proof_l2510_251047


namespace NUMINAMATH_CALUDE_percent_relation_l2510_251015

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y) 
  (h2 : z = 2 * x) : 
  y = 0.75 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l2510_251015


namespace NUMINAMATH_CALUDE_root_sum_inverse_complement_l2510_251048

def cubic_polynomial (x : ℝ) : ℝ := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem root_sum_inverse_complement (a b c : ℝ) : 
  (cubic_polynomial a = 0) → 
  (cubic_polynomial b = 0) → 
  (cubic_polynomial c = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) → 
  (0 < a) → (a < 1) → 
  (0 < b) → (b < 1) → 
  (0 < c) → (c < 1) → 
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_inverse_complement_l2510_251048


namespace NUMINAMATH_CALUDE_min_even_integers_l2510_251054

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 28 →
  a + b + c + d = 45 →
  a + b + c + d + e + f = 63 →
  ∃ (n : ℕ), n ≥ 1 ∧ 
    ∀ (m : ℕ), (∃ (evens : Finset ℤ), evens.card = m ∧ 
      (∀ x ∈ evens, x % 2 = 0) ∧ 
      evens ⊆ {a, b, c, d, e, f}) → 
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l2510_251054


namespace NUMINAMATH_CALUDE_liam_money_left_l2510_251016

/-- Calculates the amount of money Liam has left after paying his bills -/
def money_left_after_bills (monthly_savings : ℕ) (savings_duration_months : ℕ) (bills : ℕ) : ℕ :=
  monthly_savings * savings_duration_months - bills

/-- Theorem stating that Liam will have $8,500 left after paying his bills -/
theorem liam_money_left :
  money_left_after_bills 500 24 3500 = 8500 := by
  sorry

#eval money_left_after_bills 500 24 3500

end NUMINAMATH_CALUDE_liam_money_left_l2510_251016


namespace NUMINAMATH_CALUDE_find_n_l2510_251024

theorem find_n (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : ¬3 ∣ n) (h3 : ¬2 ∣ m) : n = 230 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2510_251024


namespace NUMINAMATH_CALUDE_panda_pregnancy_percentage_l2510_251098

/-- The number of pandas in the zoo -/
def total_pandas : ℕ := 16

/-- The number of panda babies born -/
def babies_born : ℕ := 2

/-- The number of panda couples in the zoo -/
def total_couples : ℕ := total_pandas / 2

/-- The number of couples that got pregnant -/
def pregnant_couples : ℕ := babies_born

/-- The percentage of panda couples that get pregnant after mating -/
def pregnancy_percentage : ℚ := pregnant_couples / total_couples * 100

theorem panda_pregnancy_percentage :
  pregnancy_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_panda_pregnancy_percentage_l2510_251098


namespace NUMINAMATH_CALUDE_number_puzzle_l2510_251006

theorem number_puzzle : ∃ x : ℚ, (x / 5 + 6 = x / 4 - 6) ∧ x = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2510_251006


namespace NUMINAMATH_CALUDE_max_graduates_proof_l2510_251094

theorem max_graduates_proof (x : ℕ) : 
  x ≤ 210 ∧ 
  (49 + ((x - 50) / 8) * 7 : ℝ) / x > 0.9 ∧ 
  ∀ y : ℕ, y > 210 → (49 + ((y - 50) / 8) * 7 : ℝ) / y ≤ 0.9 := by
  sorry

end NUMINAMATH_CALUDE_max_graduates_proof_l2510_251094


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2510_251031

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem volume_of_inscribed_sphere (π : ℝ) :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = (256 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2510_251031


namespace NUMINAMATH_CALUDE_metal_waste_l2510_251058

theorem metal_waste (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let circle_area := Real.pi * (s/2)^2
  let inner_square_side := s / Real.sqrt 2
  let inner_square_area := inner_square_side^2
  let waste := square_area - circle_area + (circle_area - inner_square_area)
  waste = square_area / 2 :=
by sorry

end NUMINAMATH_CALUDE_metal_waste_l2510_251058


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l2510_251064

noncomputable def f (x : ℝ) : ℝ := Real.cos x - x / 2

theorem tangent_line_at_zero (x y : ℝ) :
  (f 0 = 1) →
  (∀ x, HasDerivAt f (-Real.sin x - 1/2) x) →
  (y = -1/2 * x + 1) →
  (x + 2*y = 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l2510_251064


namespace NUMINAMATH_CALUDE_vacuum_pump_usage_l2510_251040

/-- The fraction of air remaining after each use of the pump -/
def remaining_fraction : ℝ := 0.4

/-- The target fraction of air to reach -/
def target_fraction : ℝ := 0.005

/-- The minimum number of pump uses required -/
def min_pump_uses : ℕ := 6

theorem vacuum_pump_usage (n : ℕ) :
  n ≥ min_pump_uses ↔ remaining_fraction ^ n < target_fraction :=
sorry

end NUMINAMATH_CALUDE_vacuum_pump_usage_l2510_251040


namespace NUMINAMATH_CALUDE_smallest_number_last_three_digits_l2510_251032

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def consists_of_2_and_7 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 2 ∨ d = 7

def has_at_least_one_2_and_7 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem smallest_number_last_three_digits :
  ∃ m : ℕ, 
    (∀ k : ℕ, k < m → 
      ¬(is_divisible_by k 6 ∧ 
        is_divisible_by k 8 ∧ 
        consists_of_2_and_7 k ∧ 
        has_at_least_one_2_and_7 k)) ∧
    is_divisible_by m 6 ∧
    is_divisible_by m 8 ∧
    consists_of_2_and_7 m ∧
    has_at_least_one_2_and_7 m ∧
    last_three_digits m = 722 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_last_three_digits_l2510_251032


namespace NUMINAMATH_CALUDE_outfit_combinations_l2510_251014

/-- The number of different outfits that can be created given a set of clothing items -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem: Given 8 shirts, 4 pairs of pants, 5 ties, and 2 types of belts,
    where an outfit requires a shirt and pants, and can optionally include a tie and/or a belt,
    the total number of different outfits that can be created is 576. -/
theorem outfit_combinations : number_of_outfits 8 4 5 2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2510_251014


namespace NUMINAMATH_CALUDE_power_function_through_point_l2510_251033

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 3 = 27 → ∀ x : ℝ, f x = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2510_251033


namespace NUMINAMATH_CALUDE_world_book_day_purchase_l2510_251051

theorem world_book_day_purchase (planned_spending : ℝ) (price_reduction : ℝ) (additional_books_ratio : ℝ) :
  planned_spending = 180 →
  price_reduction = 9 →
  additional_books_ratio = 1/4 →
  ∃ (planned_books actual_books : ℝ),
    planned_books > 0 ∧
    actual_books = planned_books * (1 + additional_books_ratio) ∧
    planned_spending / planned_books - planned_spending / actual_books = price_reduction ∧
    actual_books = 5 := by
  sorry

end NUMINAMATH_CALUDE_world_book_day_purchase_l2510_251051


namespace NUMINAMATH_CALUDE_complex_modulus_cos_sin_three_l2510_251034

theorem complex_modulus_cos_sin_three : 
  let z : ℂ := Complex.mk (Real.cos 3) (Real.sin 3)
  |(Complex.abs z - 1)| < 1e-10 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_cos_sin_three_l2510_251034


namespace NUMINAMATH_CALUDE_marias_purse_value_l2510_251044

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of nickels in Maria's purse -/
def num_nickels : ℕ := 2

/-- The number of dimes in Maria's purse -/
def num_dimes : ℕ := 3

/-- The number of quarters in Maria's purse -/
def num_quarters : ℕ := 2

theorem marias_purse_value :
  (num_nickels * nickel_value + num_dimes * dime_value + num_quarters * quarter_value) * 100 / cents_per_dollar = 90 := by
  sorry

end NUMINAMATH_CALUDE_marias_purse_value_l2510_251044


namespace NUMINAMATH_CALUDE_minimal_ratio_is_two_thirds_l2510_251071

/-- Represents a point on the square tablecloth -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square tablecloth with dark spots -/
structure Tablecloth where
  side_length : ℝ
  spots : Set Point

/-- The total area of dark spots on the tablecloth -/
def total_spot_area (t : Tablecloth) : ℝ := sorry

/-- The visible area of spots when folded along a specified line -/
def visible_area_when_folded (t : Tablecloth) (fold_type : Nat) : ℝ := sorry

theorem minimal_ratio_is_two_thirds (t : Tablecloth) :
  let S := total_spot_area t
  let S₁ := visible_area_when_folded t 1  -- Folding along first median or diagonal
  (∀ (i : Nat), i ≤ 3 → visible_area_when_folded t i = S₁) ∧  -- First three folds result in S₁
  (visible_area_when_folded t 4 = S) →  -- Fourth fold (other diagonal) results in S
  ∃ (ratio : ℝ), (∀ (r : ℝ), S₁ / S ≥ r → r ≤ ratio) ∧ ratio = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_minimal_ratio_is_two_thirds_l2510_251071


namespace NUMINAMATH_CALUDE_mans_rate_l2510_251053

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 12) :
  (speed_with_stream + speed_against_stream) / 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l2510_251053


namespace NUMINAMATH_CALUDE_total_value_after_depreciation_l2510_251029

def calculate_depreciated_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

theorem total_value_after_depreciation 
  (machine1_value : ℝ) (machine2_value : ℝ) (machine3_value : ℝ)
  (machine1_rate : ℝ) (machine2_rate : ℝ) (machine3_rate : ℝ)
  (years : ℕ) :
  machine1_value = 2500 →
  machine2_value = 3500 →
  machine3_value = 4500 →
  machine1_rate = 0.05 →
  machine2_rate = 0.07 →
  machine3_rate = 0.04 →
  years = 3 →
  (calculate_depreciated_value machine1_value machine1_rate years +
   calculate_depreciated_value machine2_value machine2_rate years +
   calculate_depreciated_value machine3_value machine3_rate years) = 8940 := by
  sorry

end NUMINAMATH_CALUDE_total_value_after_depreciation_l2510_251029


namespace NUMINAMATH_CALUDE_cube_to_sphere_surface_area_ratio_l2510_251088

theorem cube_to_sphere_surface_area_ratio :
  ∀ (a R : ℝ), a > 0 → R > 0 →
  (a^3 = (4/3) * π * R^3) →
  ((6 * a^2) / (4 * π * R^2) = 3 * (6/π)) :=
by sorry

end NUMINAMATH_CALUDE_cube_to_sphere_surface_area_ratio_l2510_251088


namespace NUMINAMATH_CALUDE_trajectory_of_A_l2510_251052

-- Define the points B and C
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 0)

-- Define the perimeter of triangle ABC
def perimeter : ℝ := 16

-- Define the trajectory of point A
def trajectory (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1 ∧ y ≠ 0

-- Theorem statement
theorem trajectory_of_A (A : ℝ × ℝ) :
  (dist A B + dist A C + dist B C = perimeter) →
  trajectory A.1 A.2 :=
by sorry


end NUMINAMATH_CALUDE_trajectory_of_A_l2510_251052


namespace NUMINAMATH_CALUDE_tom_seashells_per_day_l2510_251091

/-- Represents the number of seashells Tom found each day -/
def seashells_per_day (total_seashells : ℕ) (days_at_beach : ℕ) : ℕ :=
  total_seashells / days_at_beach

/-- Theorem stating that Tom found 7 seashells per day -/
theorem tom_seashells_per_day :
  seashells_per_day 35 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_per_day_l2510_251091


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2510_251011

/-- Calculates the selling price of an article given its cost price and profit percentage -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem: The selling price of an article with cost price 480 and profit percentage 25% is 600 -/
theorem selling_price_calculation :
  selling_price 480 25 = 600 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l2510_251011


namespace NUMINAMATH_CALUDE_expression_equals_minus_15i_l2510_251069

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z : ℂ := (1 + i) / (1 - i)

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The expression to be evaluated -/
noncomputable def expression : ℂ := 
  binomial 8 1 + 
  binomial 8 2 * z + 
  binomial 8 3 * z^2 + 
  binomial 8 4 * z^3 + 
  binomial 8 5 * z^4 + 
  binomial 8 6 * z^5 + 
  binomial 8 7 * z^6 + 
  binomial 8 8 * z^7

theorem expression_equals_minus_15i : expression = -15 * i := by sorry

end NUMINAMATH_CALUDE_expression_equals_minus_15i_l2510_251069


namespace NUMINAMATH_CALUDE_equation_solution_l2510_251093

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∀ x : ℝ, (a * Real.sin x + b) / (b * Real.cos x + a) = (a * Real.cos x + b) / (b * Real.sin x + a) ↔
  ∃ k : ℤ, x = π/4 + π * k :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2510_251093


namespace NUMINAMATH_CALUDE_square_area_12m_l2510_251072

theorem square_area_12m (side_length : ℝ) (area : ℝ) : 
  side_length = 12 → area = side_length^2 → area = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_12m_l2510_251072


namespace NUMINAMATH_CALUDE_tomatoes_left_l2510_251092

/-- Theorem: Given a farmer with 160 tomatoes who picks 56 tomatoes, the number of tomatoes left is equal to 104. -/
theorem tomatoes_left (total : ℕ) (picked : ℕ) (h1 : total = 160) (h2 : picked = 56) :
  total - picked = 104 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l2510_251092


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l2510_251035

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l2510_251035


namespace NUMINAMATH_CALUDE_tom_seashell_collection_l2510_251028

/-- The number of days Tom spent at the beach -/
def days_at_beach : ℕ := 5

/-- The number of seashells Tom found each day -/
def seashells_per_day : ℕ := 7

/-- The total number of seashells Tom found during his beach trip -/
def total_seashells : ℕ := days_at_beach * seashells_per_day

theorem tom_seashell_collection :
  total_seashells = 35 :=
by sorry

end NUMINAMATH_CALUDE_tom_seashell_collection_l2510_251028


namespace NUMINAMATH_CALUDE_valentine_spending_percentage_l2510_251021

def total_students : ℕ := 30
def valentine_percentage : ℚ := 60 / 100
def valentine_cost : ℚ := 2
def total_money : ℚ := 40

theorem valentine_spending_percentage :
  (↑total_students * valentine_percentage * valentine_cost) / total_money * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_valentine_spending_percentage_l2510_251021


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2510_251090

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2510_251090


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2510_251039

/-- The perimeter of a semi-circle with radius 2.1 cm is π * 2.1 + 4.2 cm. -/
theorem semicircle_perimeter :
  let r : ℝ := 2.1
  (π * r + 2 * r) = π * 2.1 + 4.2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2510_251039


namespace NUMINAMATH_CALUDE_stratified_sample_middle_school_l2510_251005

/-- Represents the number of students in a school -/
structure School :=
  (students : ℕ)

/-- Represents a stratified sampling plan -/
structure StratifiedSample :=
  (schoolA : School)
  (schoolB : School)
  (schoolC : School)
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (isArithmeticSequence : schoolA.students + schoolC.students = 2 * schoolB.students)

/-- The theorem statement -/
theorem stratified_sample_middle_school 
  (sample : StratifiedSample)
  (h1 : sample.totalStudents = 1500)
  (h2 : sample.sampleSize = 120) :
  ∃ (d : ℕ), 
    sample.schoolA.students = 40 - d ∧ 
    sample.schoolB.students = 40 ∧ 
    sample.schoolC.students = 40 + d :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_middle_school_l2510_251005


namespace NUMINAMATH_CALUDE_sum_of_five_numbers_l2510_251060

theorem sum_of_five_numbers : 1357 + 2468 + 3579 + 4680 + 5791 = 17875 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_numbers_l2510_251060


namespace NUMINAMATH_CALUDE_max_value_f_l2510_251080

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem max_value_f (a : ℝ) (h : a ≤ 0) :
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc 0 1 → f a x ≥ f a y) ∧
  (∃ (max_val : ℝ), 
    (a = 0 → max_val = 1) ∧
    (-2 < a ∧ a < 0 → max_val = Real.exp a) ∧
    (a ≤ -2 → max_val = 4 / (a^2 * Real.exp 2))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_l2510_251080


namespace NUMINAMATH_CALUDE_abs_c_equals_181_l2510_251008

def f (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_equals_181 (a b c : ℤ) :
  (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs) = 1) →
  (f a b c (3 + 2*Complex.I) = 0) →
  (c.natAbs = 181) :=
sorry

end NUMINAMATH_CALUDE_abs_c_equals_181_l2510_251008


namespace NUMINAMATH_CALUDE_rope_cutting_ratio_l2510_251096

/-- Given a rope of initial length 100 feet, prove that after specific cuts,
    the ratio of the final piece to its parent piece is 1:5 -/
theorem rope_cutting_ratio :
  ∀ (initial_length : ℝ) (final_piece_length : ℝ),
    initial_length = 100 →
    final_piece_length = 5 →
    ∃ (second_cut_length : ℝ),
      second_cut_length = initial_length / 4 ∧
      final_piece_length / second_cut_length = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_ratio_l2510_251096


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2510_251000

theorem min_value_quadratic (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*k*x₁ + k^2 + k + 3 = 0 ∧ x₂^2 + 2*k*x₂ + k^2 + k + 3 = 0) →
  (∀ k' : ℝ, k'^2 + k' + 3 ≥ 9) ∧ (∃ k₀ : ℝ, k₀^2 + k₀ + 3 = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2510_251000


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l2510_251085

/-- Represents a binary sequence -/
def BinarySequence := List Bool

/-- Checks if a binary sequence contains two consecutive 1s -/
def hasConsecutiveOnes : BinarySequence → Bool :=
  fun seq => sorry

/-- Generates all valid 12-digit binary sequences starting with 1 -/
def generateSequences : List BinarySequence :=
  sorry

/-- Counts the number of sequences without consecutive 1s -/
def countValidSequences : Nat :=
  sorry

/-- The total number of possible 12-digit sequences starting with 1 -/
def totalSequences : Nat := 2^11

theorem probability_no_consecutive_ones :
  (countValidSequences : ℚ) / totalSequences = 233 / 2048 :=
sorry

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l2510_251085


namespace NUMINAMATH_CALUDE_simplify_square_root_l2510_251001

theorem simplify_square_root : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by sorry

end NUMINAMATH_CALUDE_simplify_square_root_l2510_251001


namespace NUMINAMATH_CALUDE_crayons_left_l2510_251025

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : 
  initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l2510_251025


namespace NUMINAMATH_CALUDE_root_sum_square_problem_l2510_251002

theorem root_sum_square_problem (α β : ℝ) : 
  (α^2 + 2*α - 2025 = 0) → 
  (β^2 + 2*β - 2025 = 0) → 
  α^2 + 3*α + β = 2023 := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_problem_l2510_251002


namespace NUMINAMATH_CALUDE_total_chase_time_distance_equality_at_capture_l2510_251099

/-- Represents the chase scenario between Black Cat Detective and One-Ear --/
structure ChaseScenario where
  v : ℝ  -- One-Ear's speed
  initial_time : ℝ  -- Time before chase begins
  chase_time : ℝ  -- Time of chase

/-- Conditions of the chase scenario --/
def chase_conditions (s : ChaseScenario) : Prop :=
  s.initial_time = 13 ∧
  s.chase_time = 1 ∧
  s.v > 0

/-- The theorem stating the total time of the chase --/
theorem total_chase_time (s : ChaseScenario) 
  (h : chase_conditions s) : s.initial_time + s.chase_time = 14 := by
  sorry

/-- The theorem proving the distance equality at the point of capture --/
theorem distance_equality_at_capture (s : ChaseScenario) 
  (h : chase_conditions s) : 
  (5 * s.v + s.v) * s.initial_time = (7.5 * s.v - s.v) * s.chase_time := by
  sorry

end NUMINAMATH_CALUDE_total_chase_time_distance_equality_at_capture_l2510_251099


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_zero_l2510_251083

theorem sum_of_x_and_y_is_zero (x y : ℝ) 
  (h : (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1) : 
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_zero_l2510_251083


namespace NUMINAMATH_CALUDE_min_sum_m_n_l2510_251022

theorem min_sum_m_n (m n : ℕ+) (h : 108 * m = n ^ 3) : 
  ∀ (k l : ℕ+), 108 * k = l ^ 3 → m + n ≤ k + l :=
sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l2510_251022


namespace NUMINAMATH_CALUDE_tom_current_blue_tickets_l2510_251062

/-- Represents the number of tickets Tom has -/
structure TomTickets where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the conversion rates between ticket types -/
structure TicketConversion where
  yellow_to_red : ℕ
  red_to_blue : ℕ

/-- Theorem: Given the conditions, Tom currently has 7 blue tickets -/
theorem tom_current_blue_tickets 
  (total_yellow_needed : ℕ)
  (conversion : TicketConversion)
  (tom_tickets : TomTickets)
  (additional_blue_needed : ℕ)
  (h1 : total_yellow_needed = 10)
  (h2 : conversion.yellow_to_red = 10)
  (h3 : conversion.red_to_blue = 10)
  (h4 : tom_tickets.yellow = 8)
  (h5 : tom_tickets.red = 3)
  (h6 : additional_blue_needed = 163) :
  tom_tickets.blue = 7 := by
  sorry

#check tom_current_blue_tickets

end NUMINAMATH_CALUDE_tom_current_blue_tickets_l2510_251062


namespace NUMINAMATH_CALUDE_candy_bowl_problem_l2510_251077

theorem candy_bowl_problem (talitha_pieces solomon_pieces remaining_pieces : ℕ) 
  (h1 : talitha_pieces = 108)
  (h2 : solomon_pieces = 153)
  (h3 : remaining_pieces = 88) :
  talitha_pieces + solomon_pieces + remaining_pieces = 349 := by
  sorry

end NUMINAMATH_CALUDE_candy_bowl_problem_l2510_251077


namespace NUMINAMATH_CALUDE_tank_capacity_l2510_251097

theorem tank_capacity : ∃ (x : ℝ), x > 0 ∧ (3/4 * x - 1/3 * x = 18) ∧ x = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2510_251097


namespace NUMINAMATH_CALUDE_square_root_of_25_l2510_251023

theorem square_root_of_25 : ∃ (a b : ℝ), a ≠ b ∧ a^2 = 25 ∧ b^2 = 25 := by
  sorry

#check square_root_of_25

end NUMINAMATH_CALUDE_square_root_of_25_l2510_251023


namespace NUMINAMATH_CALUDE_min_expression_value_l2510_251067

-- Define the type for our permutation
def Permutation := Fin 9 → Fin 9

-- Define our expression as a function of a permutation
def expression (p : Permutation) : ℕ :=
  let x₁ := (p 0).val + 1
  let x₂ := (p 1).val + 1
  let x₃ := (p 2).val + 1
  let y₁ := (p 3).val + 1
  let y₂ := (p 4).val + 1
  let y₃ := (p 5).val + 1
  let z₁ := (p 6).val + 1
  let z₂ := (p 7).val + 1
  let z₃ := (p 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃

-- State the theorem
theorem min_expression_value :
  (∀ p : Permutation, expression p ≥ 214) ∧
  (∃ p : Permutation, expression p = 214) := by
  sorry

end NUMINAMATH_CALUDE_min_expression_value_l2510_251067


namespace NUMINAMATH_CALUDE_new_person_weight_l2510_251063

/-- Given a group of 8 people, if replacing a person weighing 45 kg with a new person
    increases the average weight by 2.5 kg, then the weight of the new person is 65 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2510_251063


namespace NUMINAMATH_CALUDE_arithmetic_progression_term_position_l2510_251009

theorem arithmetic_progression_term_position
  (a d : ℚ)  -- first term and common difference
  (sum_two_terms : a + 11 * d + a + (x - 1) * d = 20)  -- sum of 12th and x-th term is 20
  (sum_ten_terms : 10 * a + 45 * d = 100)  -- sum of first 10 terms is 100
  (x : ℕ)  -- position of the other term
  : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_term_position_l2510_251009


namespace NUMINAMATH_CALUDE_different_color_probability_l2510_251075

theorem different_color_probability (total : ℕ) (white : ℕ) (black : ℕ) (drawn : ℕ) :
  total = white + black →
  total = 5 →
  white = 3 →
  black = 2 →
  drawn = 2 →
  (Nat.choose white 1 * Nat.choose black 1 : ℚ) / Nat.choose total drawn = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2510_251075


namespace NUMINAMATH_CALUDE_sinusoidal_midline_l2510_251020

theorem sinusoidal_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_midline_l2510_251020


namespace NUMINAMATH_CALUDE_slope_of_line_l2510_251010

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ x₂) 
  (h₁ : (4 : ℝ) / x₁ + (6 : ℝ) / y₁ = 0) 
  (h₂ : (4 : ℝ) / x₂ + (6 : ℝ) / y₂ = 0) : 
  (y₂ - y₁) / (x₂ - x₁) = -(3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2510_251010
