import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l675_67596

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 60) (h2 : time = 7) :
  ∃ length : ℝ, abs (length - 116.69) < 0.01 ∧ length = speed * (1000 / 3600) * time := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l675_67596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l675_67594

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := Real.sin (x / 2 - 5 * Real.pi / 12)

-- Theorem statement
theorem function_transformation (x : ℝ) : 
  g x = f (2 * (x / 2 + Real.pi / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l675_67594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l675_67547

/-- Given 15 points on the positive x-axis and 6 points on the positive y-axis
    (with non-integer coordinates), the maximum number of intersection points
    in the first quadrant formed by segments connecting points from x-axis
    to y-axis is 1575. -/
theorem max_intersection_points
  (x_points : Finset ℝ)
  (y_points : Finset ℝ)
  (hx : x_points.card = 15)
  (hy : y_points.card = 6)
  (hx_pos : ∀ x ∈ x_points, x > 0)
  (hy_pos : ∀ y ∈ y_points, y > 0)
  (hy_non_int : ∀ y ∈ y_points, ¬ ∃ n : ℤ, y = n) :
  (Nat.choose x_points.card 2 * Nat.choose y_points.card 2 : ℕ) = 1575 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l675_67547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_symmetric_to_M_l675_67501

/-- The line about which the symmetry occurs -/
def symmetry_line (x y : ℝ) : Prop := x + 2 * y - 1 = 0

/-- The point M -/
noncomputable def M : ℝ × ℝ := (-1, 0)

/-- The point M' -/
noncomputable def M' : ℝ × ℝ := (-1/5, 8/5)

/-- Checks if two points are symmetric about a line -/
def is_symmetric (p1 p2 : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) / (p2.1 - p1.1) = -2

/-- Theorem stating that M' is symmetric to M about the given line -/
theorem M_symmetric_to_M' : is_symmetric M M' symmetry_line := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_symmetric_to_M_l675_67501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_omega_bound_l675_67540

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - Real.sin (ω * x)

theorem monotonic_decreasing_omega_bound 
  (ω : ℝ) 
  (h_pos : ω > 0) 
  (h_monotonic : StrictMonoOn (f ω) (Set.Ioo (-Real.pi/2) (Real.pi/2))) :
  ω ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_omega_bound_l675_67540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l675_67523

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin (3*x + Real.pi/3) + Real.cos (3*x + Real.pi/6) + m * Real.sin (3*x)

-- Theorem for part 1
theorem part_one (m : ℝ) : f m (17*Real.pi/18) = -1 → m = 1 := by sorry

-- Theorem for part 2
theorem part_two (A B C : ℝ) (a b c : ℝ) :
  f 1 (B/3) = Real.sqrt 3 →
  a^2 = 2*c^2 + b^2 →
  Real.tan A = -3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l675_67523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_theorem_l675_67529

theorem sqrt_sum_theorem : Real.sqrt 9 + (-1)^2 + (27 : Real)^(1/3) + Real.sqrt 36 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_theorem_l675_67529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_terminal_point_l675_67507

theorem sine_terminal_point (α : ℝ) (y : ℝ) :
  Real.sin α = -1/2 →
  (2 : ℝ) ^ 2 + y ^ 2 = (2 / Real.sin α) ^ 2 →
  y = -(2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_terminal_point_l675_67507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_distances_constant_l675_67587

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  a : ℝ  -- Side length
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Function to calculate the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Predicate to check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  distanceSquared p c.center = c.radius^2

/-- Theorem: Sum of squares of distances from a point on the circle to triangle vertices is constant -/
theorem sum_squares_distances_constant
  (triangle : EquilateralTriangle)
  (circle : Circle)
  (h1 : circle.radius = triangle.a / Real.sqrt 3)
  (h2 : ∀ (side : Point × Point), side ∈ [(triangle.A, triangle.B), (triangle.B, triangle.C), (triangle.C, triangle.A)] →
        ∃ (p1 p2 : Point), p1 ≠ p2 ∧ p1 ≠ side.1 ∧ p2 ≠ side.2 ∧
        distanceSquared p1 side.1 = distanceSquared p1 p2 ∧
        distanceSquared p2 side.2 = distanceSquared p1 p2 ∧
        isOnCircle p1 circle ∧ isOnCircle p2 circle)
  (P : Point)
  (h3 : isOnCircle P circle) :
  distanceSquared P triangle.A + distanceSquared P triangle.B + distanceSquared P triangle.C = 2 * triangle.a^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_distances_constant_l675_67587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l675_67549

noncomputable def g (x : Real) : Real := Real.sin (2 * x - Real.pi / 3)

theorem g_properties :
  (∀ x : Real, g (Real.pi / 3 - x) = -g (Real.pi / 3 + x)) ∧
  (∀ x ∈ Set.Icc (Real.pi / 12) (5 * Real.pi / 12), 
    ∀ y ∈ Set.Icc (Real.pi / 12) (5 * Real.pi / 12), 
    x < y → g x < g y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l675_67549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_sides_l675_67517

noncomputable def α : ℝ := Real.arctan (1/3)
def t : ℝ := 2

def point1 : ℝ × ℝ := (3, 1)
def point2 : ℝ × ℝ := (t, 4)

theorem angle_terminal_sides :
  point1.1 = 3 ∧ point1.2 = 1 ∧
  point2.1 = t ∧ point2.2 = 4 ∧
  Real.tan α = 1/3 ∧
  Real.tan (α + Real.pi/4) = 4/t →
  t = 2 := by
  sorry

#check angle_terminal_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_sides_l675_67517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_product_l675_67592

noncomputable section

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the fixed circle B
def circle_B (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define a point on the trajectory
def point_on_trajectory (Q : ℝ × ℝ) : Prop := trajectory_E Q.1 Q.2

-- Define the angle AQB
noncomputable def angle_AQB (Q : ℝ × ℝ) : ℝ := 
  Real.arccos ((Q.1 + 2)^2 + Q.2^2 - 16) / (2 * Real.sqrt ((Q.1 + 2)^2 + Q.2^2) * Real.sqrt ((Q.1 - 2)^2 + Q.2^2))

-- Theorem statement
theorem trajectory_and_product :
  ∀ (P : ℝ × ℝ), 
    (∀ (x y : ℝ), (x, y) = P → trajectory_E x y) ∧
    (∀ (Q : ℝ × ℝ), point_on_trajectory Q → angle_AQB Q = π / 3 → 
      Real.sqrt ((Q.1 + 2)^2 + Q.2^2) * Real.sqrt ((Q.1 - 2)^2 + Q.2^2) = 20 / 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_product_l675_67592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_line_intersection_inequality_l675_67536

/-- 
Given a triangle ABC with centroid G and a line through G intersecting AB at P and AC at Q,
prove that (PB/PA) * (QC/QA) ≤ 1/4
-/
theorem centroid_line_intersection_inequality 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (G : EuclideanSpace ℝ (Fin 2)) 
  (P : EuclideanSpace ℝ (Fin 2)) 
  (Q : EuclideanSpace ℝ (Fin 2)) 
  (h_centroid : G = (1/3 : ℝ) • (A + B + C)) 
  (h_P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) 
  (h_Q_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • A + s • C) 
  (h_G_on_PQ : ∃ r : ℝ, G = (1 - r) • P + r • Q) : 
  (dist P B / dist P A) * (dist Q C / dist Q A) ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_line_intersection_inequality_l675_67536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_one_not_plus_minus_one_l675_67516

-- Definition of cube root (marked as noncomputable)
noncomputable def cube_root (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem cube_root_of_one_not_plus_minus_one :
  ¬(cube_root 1 = 1 ∧ cube_root 1 = -1) :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_one_not_plus_minus_one_l675_67516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_f_over_x_geq_2_iff_a_geq_1_g_increasing_iff_a_geq_1_16_l675_67575

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + (1 - (a - 1) * x^2) / x

-- Theorem 1
theorem max_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧ f 2 x = 5 ∧ ∀ y ∈ Set.Icc (-1) 1, f 2 y ≤ 5 :=
by sorry

-- Theorem 2
theorem f_over_x_geq_2_iff_a_geq_1 :
  ∀ a : ℝ, (∀ x, x ∈ Set.Icc 1 2 → f a x / x ≥ 2) ↔ a ≥ 1 :=
by sorry

-- Theorem 3
theorem g_increasing_iff_a_geq_1_16 :
  ∀ a : ℝ, (∀ x y, x ∈ Set.Ioo 2 3 → y ∈ Set.Ioo 2 3 → x < y → g a x < g a y) ↔ a ≥ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_f_over_x_geq_2_iff_a_geq_1_g_increasing_iff_a_geq_1_16_l675_67575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l675_67588

/-- Represents a convex quadrilateral with vertices A, B, C, D -/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

/-- Represents that diagonals of quadrilateral ABCD intersect at point O -/
def DiagonalsIntersectAt (A B C D O : ℝ × ℝ) : Prop := sorry

/-- Represents that a line through O intersects AB at M -/
def LineIntersectsAt (O A B M : ℝ × ℝ) : Prop := sorry

/-- Calculates the area of a triangle with vertices P, Q, R -/
noncomputable def TriangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- Given a convex quadrilateral ABCD with diagonals intersecting at O and a line through O intersecting AB at M and CD at N, 
    if the area of triangle OMB is greater than the area of triangle OND,
    and the area of triangle OCN is greater than the area of triangle OAM,
    then the sum of areas of triangles OAM, OBC, and OND is greater than 
    the sum of areas of triangles OAD, OBM, and OCN. -/
theorem quadrilateral_area_inequality 
  (A B C D O M N : ℝ × ℝ) 
  (h_convex : ConvexQuadrilateral A B C D)
  (h_diagonals : DiagonalsIntersectAt A B C D O)
  (h_line : LineIntersectsAt O A B M)
  (h_line' : LineIntersectsAt O C D N)
  (h_area1 : TriangleArea O M B > TriangleArea O N D)
  (h_area2 : TriangleArea O C N > TriangleArea O A M) :
  TriangleArea O A M + TriangleArea O B C + TriangleArea O N D >
  TriangleArea O A D + TriangleArea O B M + TriangleArea O C N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l675_67588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l675_67542

theorem triangle_side_count : 
  let possible_x := {x : ℕ | x > 0 ∧ x + 15 > 37 ∧ x + 37 > 15 ∧ 15 + 37 > x}
  Finset.card (Finset.filter (λ x => x > 0 ∧ x + 15 > 37 ∧ x + 37 > 15 ∧ 15 + 37 > x) (Finset.range 52)) = 29 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l675_67542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l675_67584

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x + a * Real.log x - 2

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x + x - b

theorem problem_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = -1 → x = 1) ∧
    (∃ x₁ x₂ : ℝ, Real.exp (-1) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.exp 1 ∧ 
      g a b x₁ = 0 ∧ g a b x₂ = 0 → 1 < b ∧ b ≤ 2 / Real.exp 1 + Real.exp 1 - 1) ∧
    (∀ x : ℝ, x > 0 → 
      (∀ t : ℝ, |t| ≤ 2 → Real.pi ^ (f a x) > (1 / Real.pi) ^ (1 + x - Real.log x)) →
      (x < 2 - Real.sqrt 2 ∨ x > 2 + Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l675_67584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_change_l675_67598

theorem village_population_change (P : ℝ) : 
  P * 1.15 * 0.90 * 1.20 * 0.75 = 7575 → 
  ⌊P⌋ = 12199 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_change_l675_67598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_radius_l675_67562

/-- The radius of the circumcircle of a triangle with two sides of length a and one side of length b -/
noncomputable def circumcircle_radius (a b : ℝ) : ℝ := a^2 / Real.sqrt (4 * a^2 - b^2)

/-- Theorem: For a triangle with two sides of length a and one side of length b,
    the radius of its circumcircle is a²/√(4a² - b²) -/
theorem triangle_circumcircle_radius (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b < 2*a) :
  ∃ (R : ℝ), R = circumcircle_radius a b ∧ R > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_radius_l675_67562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l675_67541

/-- Represents the time required to fill a pool with two valves -/
noncomputable def fill_time (pool_capacity : ℝ) (valve1_time : ℝ) (valve2_extra_rate : ℝ) : ℝ :=
  let valve1_rate := pool_capacity / valve1_time
  let valve2_rate := valve1_rate + valve2_extra_rate
  let total_rate := valve1_rate + valve2_rate
  pool_capacity / total_rate

/-- Theorem stating the time required to fill the pool under given conditions -/
theorem pool_fill_time :
  fill_time 12000 120 50 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l675_67541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l675_67581

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  4 * a = Real.sqrt 5 * c →
  Real.cos C = 3 / 5 →
  b = 11 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  (1 / 2) * a * b * Real.sin C = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l675_67581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_prices_and_schemes_l675_67561

/-- Represents the purchase details of prizes A and B -/
structure Purchase where
  a : ℕ  -- quantity of A
  b : ℕ  -- quantity of B
  cost : ℕ  -- total cost in yuan
  deriving BEq

/-- Represents the unit prices of prizes A and B -/
structure Prices where
  a : ℕ  -- unit price of A in yuan
  b : ℕ  -- unit price of B in yuan
  deriving BEq

/-- Represents a purchasing scheme for the third round -/
structure ThirdRoundScheme where
  a : ℕ  -- quantity of A
  b : ℕ  -- quantity of B
  deriving BEq

def first_purchase : Purchase := { a := 15, b := 20, cost := 520 }
def second_purchase : Purchase := { a := 20, b := 17, cost := 616 }

theorem prize_prices_and_schemes :
  ∃ (prices : Prices) (schemes : List ThirdRoundScheme),
    -- The prices satisfy the purchase equations
    (prices.a * first_purchase.a + prices.b * first_purchase.b = first_purchase.cost) ∧
    (prices.a * second_purchase.a + prices.b * second_purchase.b = second_purchase.cost) ∧
    -- The prices are 24 and 8
    prices.a = 24 ∧ prices.b = 8 ∧
    -- There are exactly 3 possible schemes
    schemes.length = 3 ∧
    -- Each scheme satisfies the constraints
    (∀ scheme ∈ schemes,
      -- Total pieces in three rounds is 100
      (first_purchase.a + second_purchase.a + scheme.a +
       first_purchase.b + second_purchase.b + scheme.b = 100) ∧
      -- Quantity of A ≥ Quantity of B in third round
      scheme.a ≥ scheme.b ∧
      -- Total cost of third round ≤ 480
      prices.a * scheme.a + prices.b * scheme.b ≤ 480) ∧
    -- The schemes are exactly the ones we found
    (schemes.contains { a := 14, b := 14 } ∧
     schemes.contains { a := 15, b := 13 } ∧
     schemes.contains { a := 16, b := 12 }) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_prices_and_schemes_l675_67561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_l675_67504

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

-- State the theorem
theorem odd_function_range (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2 + 2*x) →  -- f(x) = x^2 + 2x for x ≥ 0
  f (3 - a^2) > f (2*a) →  -- given condition
  -3 < a ∧ a < 1 :=  -- conclusion
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_l675_67504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_inequality_l675_67502

/-- The function f(x) = (1-x) / (1+x) -/
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

/-- The set of real numbers a for which f(a) + 1 ≥ f(a+1) -/
def A : Set ℝ := {a : ℝ | f a + 1 ≥ f (a + 1)}

/-- The theorem stating that A is equal to (-∞, -2) ∪ (-1, +∞) -/
theorem range_of_f_inequality :
  A = {a : ℝ | a < -2 ∨ a > -1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_inequality_l675_67502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l675_67519

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSeq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_properties (a₁ d : ℝ) :
  let a := arithmeticSeq a₁ d
  let S := arithmeticSum a₁ d
  (S 5 = 4 * (a 2) + (a 7)) ∧
  (∀ n : ℕ, ∃ k : ℝ, ∀ m : ℕ, Real.sqrt (S m) = arithmeticSeq k (Real.sqrt (S 1)) m → d = 2 * a₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l675_67519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l675_67558

-- Define the central angle in degrees
def central_angle : ℝ := 150

-- Define the radius of the sector
def sector_radius : ℝ := 12

-- Theorem statement
theorem cone_base_radius :
  let arc_length := central_angle / 360 * 2 * Real.pi * sector_radius
  let base_radius := arc_length / (2 * Real.pi)
  base_radius = 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l675_67558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_odd_distances_l675_67525

theorem no_four_points_odd_distances :
  ¬ ∃ (A B C D : ℝ × ℝ),
    (∀ (P Q : ℝ × ℝ), P ∈ ({A, B, C, D} : Set (ℝ × ℝ)) → Q ∈ ({A, B, C, D} : Set (ℝ × ℝ)) → P ≠ Q →
      ∃ (n : ℕ), dist P Q = 2 * n + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_odd_distances_l675_67525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l675_67511

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Define the property of f being increasing on the domain
def increasing_on_domain (f : ℝ → ℝ) (S : Set ℝ) :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

-- State the theorem
theorem solution_set_of_inequality 
  (h1 : increasing_on_domain f domain) :
  { x : ℝ | f (x + 1/2) ≤ f (1/(x-1)) } = { x : ℝ | -3/2 ≤ x ∧ x ≤ -1 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l675_67511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l675_67518

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem range_of_b (a b : ℝ) (h : f a = g b) : b ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l675_67518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l675_67508

theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_eccentricity : Real.sqrt (a^2 + b^2) / a = Real.sqrt 10 / 2) 
  (h_point : 4 / a^2 - 3 / b^2 = 1) :
  a^2 = 2 ∧ b^2 = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l675_67508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_six_hours_l675_67543

/-- Represents the properties of a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  inletRate : ℝ
  emptyingTime : ℝ

/-- Calculates the time it takes for the leak alone to empty the tank. -/
noncomputable def leakEmptyTime (t : Tank) : ℝ :=
  t.capacity / ((t.capacity / t.emptyingTime) + t.inletRate * 60)

/-- Theorem stating that for a specific tank configuration, 
    the leak alone empties the tank in 6 hours. -/
theorem leak_empty_time_is_six_hours :
  let t : Tank := {
    capacity := 4320,
    inletRate := 6,
    emptyingTime := 12
  }
  leakEmptyTime t = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_six_hours_l675_67543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l675_67510

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2 else x + 2

/-- Function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  f a (Real.log x + 1/x) - a

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∃ x > 0, g a x = 0) → a ∈ Set.Icc (-1) 2 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l675_67510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l675_67578

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle is given by S = (√3/4)(a² + c² - b²) -/
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)

/-- The sides a, b, c form a geometric sequence -/
def is_geometric_sequence (t : Triangle) : Prop := t.a * t.c = t.b^2

/-- Main theorem -/
theorem triangle_properties (t : Triangle) 
  (h_area : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)) 
  (h_geometric : is_geometric_sequence t) : 
  t.B = π/3 ∧ Real.sin t.A * Real.sin t.C = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l675_67578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_sum_l675_67559

/-- The sum of the areas of two small shaded regions for an equilateral triangle inscribed in a circle -/
theorem shaded_areas_sum (side_length : ℝ) (h_side : side_length = 16) : 
  ∃ (a b c : ℝ), 
    2 * ((1/6) * π * (side_length/2)^2 - (Real.sqrt 3 / 4) * side_length^2) = a * π - b * Real.sqrt c ∧ 
    a + b + c = 152.34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_sum_l675_67559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_A_calculation_l675_67528

/-- The average weight of section A in a class with two sections -/
noncomputable def average_weight_A (num_students_A num_students_B : ℕ) 
                     (avg_weight_B avg_weight_total : ℝ) : ℝ :=
  ((avg_weight_total * (num_students_A + num_students_B : ℝ)) - 
   (avg_weight_B * (num_students_B : ℝ))) / (num_students_A : ℝ)

theorem average_weight_A_calculation :
  let num_students_A : ℕ := 60
  let num_students_B : ℕ := 70
  let avg_weight_B : ℝ := 80
  let avg_weight_total : ℝ := 70.77
  abs (average_weight_A num_students_A num_students_B avg_weight_B avg_weight_total - 59.985) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_A_calculation_l675_67528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_num_roots_diff_value_l675_67556

noncomputable def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

noncomputable def mean_coeff : ℝ := (3 + (-5) + 4) / 3

noncomputable def Q (x : ℝ) : ℝ := mean_coeff * x^3 - mean_coeff * x + mean_coeff

/-- The number of real roots of a cubic polynomial -/
def num_real_roots (f : ℝ → ℝ) : ℕ := sorry

theorem same_num_roots_diff_value : 
  (num_real_roots P = num_real_roots Q) ∧ (P 1 ≠ Q 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_num_roots_diff_value_l675_67556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l675_67577

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  -- Given equation
  c * (a * Real.cos B - b / 2) = a^2 - b^2

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_problem a b c A B C) : 
  A = Real.pi / 3 ∧ 
  (∀ x y : ℝ, triangle_problem a x y A B C → Real.sin B + Real.sin C ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l675_67577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_bug_stays_or_neighbors_l675_67571

/-- Represents a position on the 3x3 grid --/
inductive Position
  | One
  | Two
  | Three
  | Four
  | Five

/-- Represents the T shape on the 3x3 grid --/
def T : Set Position :=
  {Position.One, Position.Two, Position.Three, Position.Four, Position.Five}

/-- Defines the neighboring positions for each position in T --/
def neighbors : Position → Set Position
  | Position.One => {Position.One, Position.Two, Position.Four}
  | Position.Two => {Position.One, Position.Two, Position.Three, Position.Four}
  | Position.Three => {Position.Two, Position.Three, Position.Four}
  | Position.Four => {Position.One, Position.Two, Position.Three, Position.Four, Position.Five}
  | Position.Five => {Position.Four, Position.Five}

/-- A function representing the movement of bugs --/
def bugMovement : Position → Position := sorry

theorem at_least_one_bug_stays_or_neighbors :
  ∃ p : Position, p ∈ T ∧ bugMovement p ∈ neighbors p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_bug_stays_or_neighbors_l675_67571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_subjects_time_percentage_l675_67553

/-- Represents the homework subjects -/
inductive Subject
  | Math
  | Reading
  | Biology
  | History
  | Physics
  | Chemistry

/-- Represents the number of pages for each subject -/
def pages : Subject → ℕ
  | Subject.Math => 2
  | Subject.Reading => 3
  | Subject.Biology => 10
  | Subject.History => 4
  | Subject.Physics => 5
  | Subject.Chemistry => 8

/-- The percentage of total study time spent on biology -/
def biology_time_percentage : ℚ := 30 / 100

/-- Theorem stating that the percentage of total study time spent on math, history, physics, and chemistry combined is 40% -/
theorem combined_subjects_time_percentage :
  let total_pages := (List.map pages [Subject.Math, Subject.Reading, Subject.Biology, Subject.History, Subject.Physics, Subject.Chemistry]).sum
  let other_subjects := [Subject.Math, Subject.History, Subject.Physics, Subject.Chemistry]
  let other_pages := (List.map pages other_subjects).sum
  (other_pages : ℚ) / total_pages = 40 / 100 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_subjects_time_percentage_l675_67553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_for_sine_transformation_l675_67520

theorem max_interval_for_sine_transformation (a b : ℝ) :
  let f := λ x : ℝ => (1/2 : ℝ) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x
  (∀ x ∈ Set.Icc a b, -1/2 ≤ f x ∧ f x ≤ 1) →
  b - a ≤ 4 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_for_sine_transformation_l675_67520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_three_fourths_plus_five_twelfths_i_l675_67567

theorem complex_magnitude_three_fourths_plus_five_twelfths_i :
  Complex.abs (3/4 + (5/12) * Complex.I) = Real.sqrt 106 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_three_fourths_plus_five_twelfths_i_l675_67567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l675_67597

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4) - 1

theorem f_properties :
  let period : ℝ := Real.pi
  let axis_of_symmetry (k : ℤ) : ℝ := (k : ℝ) * Real.pi / 2 + Real.pi / 8
  let max_value : ℝ := 2
  let max_x (k : ℤ) : ℝ := (k : ℝ) * Real.pi + Real.pi / 8
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, f (axis_of_symmetry k + x) = f (axis_of_symmetry k - x)) ∧
  (∀ x : ℝ, f x ≤ max_value) ∧
  (∀ k : ℤ, f (max_x k) = max_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l675_67597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pr_range_l675_67505

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Type where
  is_triangle : Prop

-- Define the angle bisector
def angle_bisector (P Q R S : ℝ × ℝ) : Prop :=
  sorry

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem triangle_pr_range (P Q R S : ℝ × ℝ) :
  Triangle P Q R →
  distance P Q = 12 →
  angle_bisector P Q R S →
  distance Q S = 4 →
  ∃ (m n : ℝ),
    m = 4 ∧ 
    n = 18 ∧ 
    m + n = 22 ∧
    ∀ x, m < x ∧ x < n ↔ distance P R = x :=
by
  sorry

#check triangle_pr_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pr_range_l675_67505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_A_geq_B_l675_67579

/-- The set of colors used to color elements of M -/
inductive Color
  | Red
  | Blue
  | Yellow

/-- A function that assigns a color to each element of M -/
def coloring (n : ℕ) := Fin n → Color

/-- The set A of triples (x,y,z) with the same color and sum congruent to 0 mod n -/
def A (n : ℕ) (c : coloring n) : Set (Fin n × Fin n × Fin n) :=
  {(x, y, z) | (x.val + y.val + z.val) % n = 0 ∧ c x = c y ∧ c y = c z}

/-- The set B of triples (x,y,z) with distinct colors and sum congruent to 0 mod n -/
def B (n : ℕ) (c : coloring n) : Set (Fin n × Fin n × Fin n) :=
  {(x, y, z) | (x.val + y.val + z.val) % n = 0 ∧ 
               c x ≠ c y ∧ c y ≠ c z ∧ c x ≠ c z}

/-- The main theorem stating that 2|A| ≥ |B| -/
theorem two_A_geq_B (n : ℕ) (c : coloring n) : 2 * Finset.card (A n c).toFinite.toFinset ≥ Finset.card (B n c).toFinite.toFinset := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_A_geq_B_l675_67579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_polar_sum_l675_67512

/-- Proves that for an ellipse with equation x²/16 + y²/4 = 1,
    if A(ρ₁,θ) and B(ρ₂,θ+π/2) are two points on the ellipse in polar coordinates,
    then 1/ρ₁² + 1/ρ₂² = 5/16 -/
theorem ellipse_polar_sum (ρ₁ ρ₂ θ : ℝ) : 
  (ρ₁ * Real.cos θ)^2 / 16 + (ρ₁ * Real.sin θ)^2 / 4 = 1 →
  (ρ₂ * Real.sin θ)^2 / 16 + (ρ₂ * Real.cos θ)^2 / 4 = 1 →
  1 / ρ₁^2 + 1 / ρ₂^2 = 5 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_polar_sum_l675_67512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_N_l675_67566

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

-- Define the condition for point N
def point_N (x y : ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ my_circle (t*x) (t*y) ∧ t ≤ 1/2

-- Theorem statement
theorem trajectory_of_N :
  ∀ x y : ℝ, point_N x y → x^2 + y*x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_N_l675_67566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_four_days_l675_67595

/-- The number of days it takes for B to finish the remaining work after A leaves -/
noncomputable def days_for_B_to_finish (days_A : ℝ) (days_B : ℝ) (days_together : ℝ) : ℝ :=
  let rate_A := 1 / days_A
  let rate_B := 1 / days_B
  let combined_rate := rate_A + rate_B
  let work_done_together := combined_rate * days_together
  let remaining_work := 1 - work_done_together
  remaining_work / rate_B

/-- Theorem stating that B will take 4 days to finish the remaining work -/
theorem b_finishes_in_four_days :
  days_for_B_to_finish 5 10 2 = 4 := by
  -- Unfold the definition of days_for_B_to_finish
  unfold days_for_B_to_finish
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_four_days_l675_67595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l675_67500

/-- Given two vectors m and n in ℝ³, where m = (3, 1, 3) and n = (-1, l, -1),
    and m is parallel to n, prove that l = -1/3 -/
theorem parallel_vectors_lambda (l : ℝ) :
  let m : Fin 3 → ℝ := ![3, 1, 3]
  let n : Fin 3 → ℝ := ![-1, l, -1]
  (∃ (k : ℝ), ∀ i, n i = k * m i) →
  l = -1/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l675_67500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_card_probability_l675_67583

-- Define the set of shapes
inductive Shape
| Circle
| Parallelogram
| IsoscelesTriangle
| Rectangle
| Square

-- Define a function to check if a shape is both axially and centrally symmetric
def isAxiallyCentrallySymmetric (s : Shape) : Bool :=
  match s with
  | Shape.Circle => true
  | Shape.Parallelogram => false
  | Shape.IsoscelesTriangle => false
  | Shape.Rectangle => true
  | Shape.Square => true

-- Define the set of cards
def cards : List Shape := [Shape.Circle, Shape.Parallelogram, Shape.IsoscelesTriangle, Shape.Rectangle, Shape.Square]

-- State the theorem
theorem symmetric_card_probability :
  (cards.filter isAxiallyCentrallySymmetric).length / cards.length = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_card_probability_l675_67583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_sets_l675_67545

def is_valid_set (S : Finset ℕ) : Prop :=
  S.Nonempty ∧ 
  ∀ i j, i ∈ S → j ∈ S → (i + j) / Nat.gcd i j ∈ S

theorem characterize_valid_sets :
  ∀ S : Finset ℕ, is_valid_set S →
    (∃ n : ℕ, S = {n}) ∨
    (∃ n : ℕ, n > 2 ∧ S = {n, n * (n - 1)}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_sets_l675_67545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_over_a_l675_67535

open Real

theorem range_of_b_over_a (a b : ℝ) (x₀ : ℝ) :
  a > 0 →
  x₀ > 1 →
  let f := λ x => (a * x - b / x - 2 * a) * Real.exp x
  let f' := λ x => (b / (x^2) + a * x - b / x - a) * Real.exp x
  (f x₀ + f' x₀ = 0) →
  b / a > -1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_over_a_l675_67535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l675_67527

theorem tan_double_angle (α : ℝ) 
  (h1 : α ∈ Set.Ioo (-π/2) 0)
  (h2 : Real.cos α = 4/5) : 
  Real.tan (2 * α) = -24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l675_67527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l675_67531

theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l675_67531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_y_l675_67513

-- Define the function f with domain [-1, 1]
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-1) 1

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := f (x + 1) / Real.sqrt (x^2 - 2*x - 3)

-- State the theorem
theorem domain_of_y :
  {x : ℝ | y x ∈ Set.range y} = Set.Ioc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_y_l675_67513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_BC_l675_67515

/-- Given vector AB, normal vector n, and dot product of n and AC, prove that n · BC = 2 -/
theorem dot_product_BC (AB n : ℝ × ℝ) (dot_nAC : ℝ) :
  AB = (3, -1) →
  n = (2, 1) →
  dot_nAC = 7 →
  ∃ AC : ℝ × ℝ, n.1 * AC.1 + n.2 * AC.2 = dot_nAC ∧
    n.1 * (AC.1 - AB.1) + n.2 * (AC.2 - AB.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_BC_l675_67515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_AXY_l675_67521

structure TriangularPyramid where
  A : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)
  D : EuclideanSpace ℝ (Fin 3)

structure Sphere where
  center : EuclideanSpace ℝ (Fin 3)
  radius : ℝ

def touches (s : Sphere) (p : EuclideanSpace ℝ (Fin 3)) : Prop :=
  dist s.center p = s.radius

def plane (p q r : EuclideanSpace ℝ (Fin 3)) : Set (EuclideanSpace ℝ (Fin 3)) :=
  {x | ∃ (a b c : ℝ), x = a • p + b • q + c • r ∧ a + b + c = 1}

def is_obtuse_triangle (a b c : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (angle : ℝ), angle > Real.pi / 2 ∧ 
    (angle = Real.arccos ((dist b a)^2 + (dist c a)^2 - (dist b c)^2) / (2 * dist b a * dist c a) ∨
     angle = Real.arccos ((dist a b)^2 + (dist c b)^2 - (dist a c)^2) / (2 * dist a b * dist c b) ∨
     angle = Real.arccos ((dist a c)^2 + (dist b c)^2 - (dist a b)^2) / (2 * dist a c * dist b c))

theorem obtuse_triangle_AXY (ABCD : TriangularPyramid)
  (inscribed : Sphere) (exscribed : Sphere)
  (X Y : EuclideanSpace ℝ (Fin 3)) :
  touches inscribed X →
  touches exscribed Y →
  X ≠ Y →
  X ∈ plane ABCD.B ABCD.C ABCD.D →
  Y ∈ plane ABCD.B ABCD.C ABCD.D →
  is_obtuse_triangle ABCD.A X Y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_AXY_l675_67521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_7528758090625_l675_67539

theorem sixth_root_of_7528758090625 : (7528758090625 : ℝ) ^ (1/6) = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_7528758090625_l675_67539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_equal_22510_l675_67572

/-- Represents the earnings of an Italian restaurant for a month. -/
structure RestaurantEarnings where
  weekdayEarnings : ℚ
  weekendMinEarnings : ℚ
  weekendMaxEarnings : ℚ
  mondayDiscount : ℚ
  monthlySpecialEvent : ℚ
  holidayMinEarnings : ℚ
  holidayMaxEarnings : ℚ
  totalWeekdays : ℕ
  specialOccasions : ℕ
  discountDays : ℕ
  totalWeekends : ℕ
  promotionalWeekends : ℕ

/-- Calculates the total earnings for the restaurant given the conditions. -/
def calculateTotalEarnings (r : RestaurantEarnings) : ℚ :=
  let regularWeekdays := r.totalWeekdays - r.specialOccasions - r.discountDays
  let regularWeekdayEarnings := r.weekdayEarnings * regularWeekdays
  let discountedEarnings := r.weekdayEarnings * (1 - r.mondayDiscount) * r.discountDays
  let specialOccasionEarnings := ((r.holidayMinEarnings + r.holidayMaxEarnings) / 2) * r.specialOccasions
  let regularWeekendEarnings := r.weekendMinEarnings * (r.totalWeekends - r.promotionalWeekends)
  let promotionalWeekendEarnings := ((r.weekendMinEarnings + r.weekendMaxEarnings) / 2) * r.promotionalWeekends
  regularWeekdayEarnings + discountedEarnings + specialOccasionEarnings +
  regularWeekendEarnings + promotionalWeekendEarnings + r.monthlySpecialEvent

/-- Theorem stating that the total earnings for the given conditions equal $22,510. -/
theorem total_earnings_equal_22510 :
  let r : RestaurantEarnings := {
    weekdayEarnings := 600,
    weekendMinEarnings := 1000,
    weekendMaxEarnings := 1500,
    mondayDiscount := 1/10,
    monthlySpecialEvent := 500,
    holidayMinEarnings := 600,
    holidayMaxEarnings := 900,
    totalWeekdays := 22,
    specialOccasions := 2,
    discountDays := 4,
    totalWeekends := 8,
    promotionalWeekends := 3
  }
  calculateTotalEarnings r = 22510 := by
  sorry

#eval calculateTotalEarnings {
  weekdayEarnings := 600,
  weekendMinEarnings := 1000,
  weekendMaxEarnings := 1500,
  mondayDiscount := 1/10,
  monthlySpecialEvent := 500,
  holidayMinEarnings := 600,
  holidayMaxEarnings := 900,
  totalWeekdays := 22,
  specialOccasions := 2,
  discountDays := 4,
  totalWeekends := 8,
  promotionalWeekends := 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_equal_22510_l675_67572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleanup_drive_result_l675_67537

/-- Represents the cleanup drive data and calculates the garbage collected per mile per hour -/
noncomputable def cleanup_drive (hours : ℝ) (lizzie_group : ℝ) (second_group_diff : ℝ) (third_group_oz : ℝ) 
                  (oz_per_pound : ℝ) (miles : ℝ) : ℝ :=
  let second_group := lizzie_group - second_group_diff
  let third_group := third_group_oz / oz_per_pound
  let total_garbage := lizzie_group + second_group + third_group
  let garbage_per_mile := total_garbage / miles
  garbage_per_mile / hours

/-- Theorem stating that the cleanup drive results in 24.0625 pounds of garbage collected per mile per hour -/
theorem cleanup_drive_result : 
  cleanup_drive 4 387 39 560 16 8 = 24.0625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleanup_drive_result_l675_67537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_has_minimum_l675_67563

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem f_is_even_and_has_minimum :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∃ m : ℝ, f 0 = m ∧ ∀ x : ℝ, f x ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_has_minimum_l675_67563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parallelograms_in_hexagon_l675_67506

/-- The area of an equilateral triangle with side length 1 -/
noncomputable def s : ℝ := Real.sqrt 3 / 4

/-- The area of a regular hexagon with side length 3 -/
noncomputable def hexagon_area : ℝ := 54 * s

/-- The area of a parallelogram with sides 1 and 2, and angles 60° and 120° -/
noncomputable def parallelogram_area : ℝ := 4 * s

/-- The number of black triangles in the coloring strategy -/
def black_triangles : ℕ := 12

/-- The maximum number of parallelograms that can be placed in the hexagon -/
noncomputable def max_parallelograms : ℕ := 
  min (Int.floor (hexagon_area / parallelogram_area)).toNat black_triangles

theorem max_parallelograms_in_hexagon : 
  max_parallelograms = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parallelograms_in_hexagon_l675_67506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_napkin_placements_l675_67585

/-- The number of ways to place napkins around a round table. -/
def napkin_placements (n : ℕ) : ℕ :=
  2^n + 2 * ((-1 : ℤ)^n).toNat

/-- The actual number of valid napkin placements (assumed to be defined elsewhere). -/
def number_of_valid_placements (n : ℕ) : ℕ :=
  sorry -- This would be defined based on the problem conditions

/-- Theorem stating the number of valid napkin placements around a round table. -/
theorem count_napkin_placements (n : ℕ) (h : n ≥ 2) :
  napkin_placements n = number_of_valid_placements n :=
by
  sorry -- Proof to be implemented

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_napkin_placements_l675_67585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l675_67546

open Function Real

theorem function_identity (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = x^2 + 2*x*(deriv f 2)) : 
  ∀ x, f x = x^2 + 12*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l675_67546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l675_67568

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := 3 * Real.sqrt (x^2 - 4*x + 4) + |x - m|

-- Define the theorem
theorem problem_solution :
  ∃ (m n : ℝ), 
    (∀ x, f x m < 3 ↔ 1 < x ∧ x < n) ∧
    m = 1 ∧
    n = 5/2 ∧
    (∀ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = m →
      a^4 / (b^2 + 1) + b^4 / (c^2 + 1) + c^4 / (a^2 + 1) ≥ 1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l675_67568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_age_is_36_l675_67538

-- Define Chris's and Dave's ages
def chris_age : ℕ → ℕ := sorry
def dave_age : ℕ → ℕ := sorry

-- The sum of their present ages is 56
axiom sum_of_ages : chris_age 0 + dave_age 0 = 56

-- Define the relationship between their ages
axiom age_relationship : ∃ t₁ t₂ t₃ : ℕ, 
  (t₁ < t₂ ∧ t₂ < t₃) ∧
  (dave_age 0 = chris_age t₁) ∧
  (dave_age t₁ = chris_age t₂) ∧
  (dave_age t₂ = chris_age t₃ / 3)

-- Theorem to prove
theorem chris_age_is_36 : chris_age 0 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_age_is_36_l675_67538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_problem_l675_67551

noncomputable def f (x : ℝ) (φ : ℝ) := 4 * Real.cos (3 * x + φ)

theorem function_symmetry_problem 
  (φ : ℝ) 
  (h_φ : |φ| < π / 2) 
  (h_sym : ∀ x, f x φ = f (11 * π / 6 - x) φ) 
  (x₁ x₂ : ℝ) 
  (h_x₁ : -7 * π / 12 < x₁ ∧ x₁ < -π / 12) 
  (h_x₂ : -7 * π / 12 < x₂ ∧ x₂ < -π / 12) 
  (h_neq : x₁ ≠ x₂) 
  (h_eq : f x₁ φ = f x₂ φ) : 
  f (x₁ + x₂) φ = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_problem_l675_67551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_larger_triangle_l675_67555

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points --/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- A set of n points on a plane --/
def PointSet (n : ℕ) := Fin n → Point

/-- Predicate to check if a point is inside or on the boundary of a triangle --/
noncomputable def pointInTriangle (p t1 t2 t3 : Point) : Prop :=
  let a1 := triangleArea p t1 t2
  let a2 := triangleArea p t2 t3
  let a3 := triangleArea p t3 t1
  let a := triangleArea t1 t2 t3
  a1 + a2 + a3 ≤ a

theorem points_in_larger_triangle (n : ℕ) (points : PointSet n) 
  (h : ∀ (i j k : Fin n), triangleArea (points i) (points j) (points k) ≤ 1) :
  ∃ (t1 t2 t3 : Point), (∀ (i : Fin n), pointInTriangle (points i) t1 t2 t3) ∧ 
    triangleArea t1 t2 t3 ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_larger_triangle_l675_67555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_altitude_length_l675_67590

/-- Triangle XYZ with given side lengths -/
structure Triangle (X Y Z : ℝ × ℝ) where
  xy_length : dist X Y = 15
  xz_length : dist X Z = 17
  yz_length : dist Y Z = 24

/-- Centroid of a triangle -/
noncomputable def centroid (X Y Z : ℝ × ℝ) : ℝ × ℝ :=
  ((X.1 + Y.1 + Z.1) / 3, (X.2 + Y.2 + Z.2) / 3)

/-- Foot of the altitude from a point to a line segment -/
noncomputable def altitudeFoot (P A B : ℝ × ℝ) : ℝ × ℝ := sorry

theorem centroid_altitude_length (X Y Z : ℝ × ℝ) (h : Triangle X Y Z) :
  let G := centroid X Y Z
  let Q := altitudeFoot G Y Z
  dist G Q = 3.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_altitude_length_l675_67590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_condition_l675_67565

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

-- State the theorem
theorem two_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
   ∀ z : ℝ, z ≠ x ∧ z ≠ y → f a z ≠ 0) ↔ 
  (0 < a ∧ a < 1/8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_condition_l675_67565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_point_exists_l675_67573

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure MovingPoint where
  start : Point
  velocity : ℝ
  direction : Line

-- Helper function to calculate distance between two points
noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the theorem
theorem constant_ratio_point_exists (A B : MovingPoint) (k : ℝ) 
    (h1 : A.velocity ≠ B.velocity) 
    (h2 : k = A.velocity / B.velocity) 
    (h3 : A.direction ≠ B.direction) : 
  ∃ P : Point, ∀ t : ℝ, 
    let A_t : Point := ⟨A.start.x + t * A.velocity * A.direction.a, A.start.y + t * A.velocity * A.direction.b⟩
    let B_t : Point := ⟨B.start.x + t * B.velocity * B.direction.a, B.start.y + t * B.velocity * B.direction.b⟩
    dist P A_t / dist P B_t = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_point_exists_l675_67573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_five_l675_67574

def S : Finset ℕ := {1, 2, 3, 4}

def pairs : Finset (ℕ × ℕ) := S.product S

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (λ p => p.1 < p.2)

def sum_is_five (p : ℕ × ℕ) : Bool := p.1 + p.2 = 5

def favorable_outcomes : Finset (ℕ × ℕ) := valid_pairs.filter (λ p => sum_is_five p)

theorem probability_sum_is_five :
  (favorable_outcomes.card : ℚ) / valid_pairs.card = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_five_l675_67574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l675_67530

/-- Given an ellipse with equation 4x^2 + y^2 = 1 and a hyperbola sharing the same foci with this ellipse,
    if one of the asymptotes of the hyperbola is given by y = √2x,
    then the equation of the hyperbola is 2y^2 - 4x^2 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ (x y : ℝ), 4 * x^2 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    (∃ (m : ℝ), 0 < m ∧ m < 3/4 ∧
      (∀ (x y : ℝ), y = Real.sqrt 2 * x ↔ y^2 / m = x^2 / (3/4 - m)))) →
  (2 * y^2 - 4 * x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l675_67530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l675_67544

theorem tan_pi_4_minus_alpha (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α = 4/5) :
  Real.tan (π/4 - α) = -1/7 ∨ Real.tan (π/4 - α) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l675_67544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_right_to_left_equiv_standard_l675_67534

/-- Evaluates an expression from right to left -/
noncomputable def rightToLeftEval (m n p q s : ℝ) : ℝ :=
  m / (n - p + q * s)

/-- Standard algebraic evaluation -/
noncomputable def standardEval (m n p q s : ℝ) : ℝ :=
  m / (n - p - q * s)

/-- 
Theorem stating that right-to-left evaluation of the expression
m ÷ n - p + q × s is equivalent to its standard algebraic evaluation
-/
theorem right_to_left_equiv_standard 
  (m n p q s : ℝ) (h : n - p + q * s ≠ 0) : 
  rightToLeftEval m n p q s = standardEval m n p q s := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_right_to_left_equiv_standard_l675_67534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vincents_earnings_l675_67591

/-- Represents the price of a fantasy book in dollars -/
def fantasy_price : ℚ := 4

/-- Represents the price of a literature book in dollars -/
def literature_price : ℚ := fantasy_price / 2

/-- Represents the number of fantasy books sold per day -/
def fantasy_sales_per_day : ℕ := 5

/-- Represents the number of literature books sold per day -/
def literature_sales_per_day : ℕ := 8

/-- Represents the number of days -/
def days : ℕ := 5

/-- Theorem stating Vincent's earnings after 5 days -/
theorem vincents_earnings : 
  (fantasy_price * fantasy_sales_per_day + 
   literature_price * literature_sales_per_day) * days = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vincents_earnings_l675_67591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l675_67589

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + f y) + f (y + f x) = 2 * f (x * f y)) →
  (∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l675_67589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l675_67524

theorem rationalize_denominator :
  (1 : ℝ) / ((3 : ℝ)^((1 : ℝ)/3) + (27 : ℝ)^((1 : ℝ)/3)) = (9 : ℝ)^((1 : ℝ)/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l675_67524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l675_67509

/-- Calculates the simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates the compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating that if the difference between compound and simple interest
    is 51 for a 10% rate over 2 years, then the principal is 5100 -/
theorem interest_difference_implies_principal :
  ∀ (principal : ℝ),
  compoundInterest principal 10 2 - simpleInterest principal 10 2 = 51 →
  principal = 5100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l675_67509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l675_67564

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem max_value_of_f :
  (∀ x > 0, x * (deriv f x) + 2 * f x = 1 / x^2) →
  f 1 = 1 →
  ∃ x > 0, ∀ y > 0, f y ≤ f x ∧ f x = exp (-1/2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l675_67564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_equality_repeating_decimal_product_equality_repeating_decimal_difference_equality_l675_67593

theorem repeating_decimal_sum_equality : 
  (12 : ℚ) / 99 + (122 : ℚ) / 999 = (4158 : ℚ) / 1089 := by sorry

theorem repeating_decimal_product_equality :
  (1 : ℚ) / 3 * (4 : ℚ) / 9 = (4 : ℚ) / 27 := by sorry

theorem repeating_decimal_difference_equality :
  1 - (85 : ℚ) / 99 = (14 : ℚ) / 99 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_equality_repeating_decimal_product_equality_repeating_decimal_difference_equality_l675_67593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_problem_l675_67514

theorem book_price_problem (price1 price2 : ℝ) : 
  60 * price1 + 75 * price2 = 2700 →
  60 * (0.85 * price1) + 75 * (0.90 * price2) = 2370 →
  price1 = 20 ∧ price2 = 20 := by
  intro h1 h2
  -- The proof steps would go here
  sorry

#check book_price_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_problem_l675_67514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_bisector_area_inequality_l675_67503

/-- A triangle type with necessary properties and operations. -/
structure Triangle where
  area : ℝ
  longestAngleBisector : ℝ
  shortestAngleBisector : ℝ
  isScalene : Prop

/-- For any scalene triangle, the square of the longest angle bisector is greater than √3 times the area,
    which is greater than the square of the shortest angle bisector. -/
theorem scalene_triangle_bisector_area_inequality (T : Triangle) (h : T.isScalene) :
  T.longestAngleBisector^2 > Real.sqrt 3 * T.area ∧
  Real.sqrt 3 * T.area > T.shortestAngleBisector^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_bisector_area_inequality_l675_67503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_extrema_l675_67552

-- Define the vectors
noncomputable def a (x : Real) : Real × Real := (Real.cos x, Real.sin x)
noncomputable def b : Real × Real := (3, -Real.sqrt 3)

-- Define the dot product function
noncomputable def f (x : Real) : Real := (a x).1 * b.1 + (a x).2 * b.2

-- Theorem statement
theorem vector_parallel_and_extrema :
  (∃ x : Real, x ∈ Set.Icc 0 Real.pi ∧ ∃ k : Real, a x = k • b) ∧
  (∃ x : Real, x ∈ Set.Icc 0 Real.pi ∧ f x = 3) ∧
  (∃ x : Real, x ∈ Set.Icc 0 Real.pi ∧ f x = -2 * Real.sqrt 3) ∧
  (∀ x : Real, x ∈ Set.Icc 0 Real.pi → f x ≤ 3) ∧
  (∀ x : Real, x ∈ Set.Icc 0 Real.pi → f x ≥ -2 * Real.sqrt 3) := by
  sorry

#check vector_parallel_and_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_extrema_l675_67552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_before_zero_l675_67560

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

-- State the theorem
theorem f_positive_before_zero (x₀ x₁ : ℝ) (h₀ : f x₀ = 0) (h₁ : 0 < x₁) (h₂ : x₁ < x₀) : f x₁ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_before_zero_l675_67560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l675_67569

/-- Proves that the father's age is 45 given the problem conditions -/
theorem fathers_age (son_age : ℕ) (father_age : ℕ) : 
  father_age = 3 * son_age ∧ 
  father_age + 15 = 2 * (son_age + 15) → 
  father_age = 45 := by
  intro h
  cases h with
  | intro h1 h2 =>
    have son_eq : son_age = 15 := by
      rw [h1] at h2
      ring_nf at h2
      linarith
    rw [son_eq] at h1
    exact h1

#check fathers_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l675_67569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l675_67586

-- Define the teams
inductive Team : Type
| A | B | C | D | E

-- Define the possible places
inductive Place : Type
| first | second | third | fourth | fifth

-- Define a function to represent the final ranking
variable (finalRanking : Team → Place)

-- Define the predictions
def prediction1 : (Team → Place) → Prop := 
  λ r => r Team.D = Place.first ∨ r Team.C = Place.second

def prediction2 : (Team → Place) → Prop := 
  λ r => r Team.A = Place.second

def prediction3 : (Team → Place) → Prop := 
  λ r => r Team.C = Place.third

def prediction4 : (Team → Place) → Prop := 
  λ r => r Team.C = Place.first ∨ r Team.D = Place.fourth

def prediction5 : (Team → Place) → Prop := 
  λ r => r Team.A = Place.second ∨ r Team.C = Place.third

-- Define the condition that each prediction has exactly one correct part
def oneCorrectPart (r : Team → Place) (p : (Team → Place) → Prop) : Prop :=
  (p r ∧ ¬(p r)) ∨ (¬(p r) ∧ p r)

-- Theorem statement
theorem correct_ranking : 
  ∃! r : Team → Place, 
    (oneCorrectPart r prediction1) ∧
    (oneCorrectPart r prediction2) ∧
    (oneCorrectPart r prediction3) ∧
    (oneCorrectPart r prediction4) ∧
    (oneCorrectPart r prediction5) ∧
    (r Team.D = Place.first) ∧
    (r Team.B = Place.second) ∧
    (r Team.C = Place.third) ∧
    (r Team.E = Place.fourth) ∧
    (r Team.A = Place.fifth) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l675_67586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l675_67576

/-- The function f satisfying the given differential equation -/
noncomputable def f : ℝ → ℝ := sorry

/-- The domain of f -/
def domain : Set ℝ := { x | 1/2 ≤ x }

/-- The differential equation for f -/
axiom f_eq : ∀ x ∈ domain, (deriv (deriv f)) x = f x + (Real.exp x) / x

/-- The initial condition for f -/
axiom f_init : f 1 = -Real.exp 1

/-- The existence of a satisfying the inequality -/
axiom exists_a : ∃ a ∈ Set.Icc (-2) 1, 
  ∀ m ∈ Set.Icc (2/3) 1, f (2 - 1/m) ≤ a^3 - 3*a - 2 - Real.exp 1

/-- The main theorem: the range of m -/
theorem range_of_m : 
  { m : ℝ | ∃ a ∈ Set.Icc (-2) 1, f (2 - 1/m) ≤ a^3 - 3*a - 2 - Real.exp 1 } = Set.Icc (2/3) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l675_67576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_simplification_l675_67582

theorem exponent_simplification :
  (625 : ℝ) ^ (0.24 : ℝ) * (625 : ℝ) ^ (0.06 : ℝ) = 5 ^ (6/5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_simplification_l675_67582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l675_67522

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 3 * n - 5

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(a n)

-- Define the sum of the first n terms of b_n
noncomputable def S (n : ℕ) : ℝ := (8^n - 1) / 28

-- Theorem statement
theorem arithmetic_geometric_sequence_properties :
  -- The sequence a_n is arithmetic with non-zero common difference
  (∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d) ∧
  -- The sum of the first 4 terms of a_n is 10
  (a 1 + a 2 + a 3 + a 4 = 10) ∧
  -- {a_2, a_5, a_7} form a geometric sequence
  (∃ r : ℝ, a 5 = a 2 * r ∧ a 7 = a 5 * r) →
  -- Prove that a_n = 3n - 5
  (∀ n : ℕ, a n = 3 * n - 5) ∧
  -- Prove that the sum of the first n terms of b_n is (8^n - 1) / 28
  (∀ n : ℕ, (Finset.range n).sum b = S n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l675_67522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l675_67570

theorem expression_equality : 
  (27 : ℝ) ^ (1/3) + |-(Real.sqrt 2)| + 2 * Real.sqrt 2 - (-(2 : ℝ)) = 5 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l675_67570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l675_67550

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m^2 * x^2 + 4 * m * x - 3 * Real.log x

-- State the theorem
theorem f_properties (m : ℝ) :
  -- Part 1: If x = 1 is an extreme point, then m = -3/2 or m = 1/2
  (∀ x, x > 0 → (deriv (f m)) x = 0 → x = 1) →
  (m = -3/2 ∨ m = 1/2) ∧
  -- Part 2: Monotonicity and extreme values depend on m
  (m > 0 →
    (∃ x₀, x₀ > 0 ∧
      (∀ x, 0 < x ∧ x < x₀ → (deriv (f m)) x < 0) ∧
      (∀ x, x > x₀ → (deriv (f m)) x > 0) ∧
      (∀ x, x > 0 → f m x ≥ f m x₀))) ∧
  (m < 0 →
    (∃ x₀, x₀ > 0 ∧
      (∀ x, 0 < x ∧ x < x₀ → (deriv (f m)) x < 0) ∧
      (∀ x, x > x₀ → (deriv (f m)) x > 0) ∧
      (∀ x, x > 0 → f m x ≥ f m x₀))) ∧
  (m = 0 →
    (∀ x, x > 0 → (deriv (f m)) x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l675_67550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_three_l675_67599

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a horizontal slice of a cone -/
structure ConeSlice where
  lowerHeight : ℝ
  upperHeight : ℝ
  cone : RightCircularCone

/-- Calculates the volume of a cone slice -/
noncomputable def volumeOfConeSlice (slice : ConeSlice) : ℝ :=
  sorry

/-- The ratio of volumes of two specific slices of a cone -/
noncomputable def volumeRatio (cone : RightCircularCone) : ℝ :=
  let smallestSlice : ConeSlice := { lowerHeight := 0, upperHeight := cone.height / 3, cone := cone }
  let middleSlice : ConeSlice := { lowerHeight := cone.height / 3, upperHeight := 2 * cone.height / 3, cone := cone }
  (volumeOfConeSlice middleSlice) / (volumeOfConeSlice smallestSlice)

/-- Theorem stating that the volume ratio of the middle piece to the smallest piece is 3 -/
theorem volume_ratio_is_three (cone : RightCircularCone) : volumeRatio cone = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_three_l675_67599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_l675_67548

theorem min_side_length (a b c : ℝ) (h1 : a^2 = b^2 + c^2 - b*c) 
  (h2 : (1/2) * b * c * Real.sqrt 3 / 2 = 3 * Real.sqrt 3 / 4) : 
  a ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_l675_67548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slopes_l675_67557

/-- Given a hyperbola with equation (x^2/144) - (y^2/81) = 1, 
    the slopes of its asymptotes are ±3/4 -/
theorem hyperbola_asymptote_slopes :
  ∃ (m : ℝ), m = 3/4 ∧ 
    (∀ (x y : ℝ), x^2 / 144 - y^2 / 81 = 1 → (y = m * x ∨ y = -m * x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slopes_l675_67557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l675_67526

noncomputable def solutions : Set (ℝ × ℝ) :=
  {(2, Real.sqrt 3), (2, -Real.sqrt 3), (-2, Real.sqrt 3), (-2, -Real.sqrt 3),
   (Real.sqrt 3, 2), (Real.sqrt 3, -2), (-Real.sqrt 3, 2), (-Real.sqrt 3, -2)}

theorem system_solution :
  ∀ x y : ℝ, (2*x^2 + 2*y^2 - x^2*y^2 = 2 ∧ x^4 + y^4 - 1/2*x^2*y^2 = 19) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l675_67526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folk_festival_theorem_l675_67532

noncomputable section

-- Define the statistical values
def sum_x_squared : ℝ := 80
def sum_y_squared : ℝ := 9000
def sum_xy : ℝ := 800

-- Define the correlation coefficient
def correlation_coefficient : ℝ := sum_xy / Real.sqrt (sum_x_squared * sum_y_squared)

-- Define the probabilities for each family
def family_A_prob : ℝ := 3/10
def family_B_prob1 : ℝ := 1/3
def family_B_prob2 : ℝ := 1/4
def family_B_prob3 : ℝ := 1/6

-- Define the costs and values
def cost_per_person : ℝ := 20
def value_per_rabbit : ℝ := 40

-- Define the expected profits
def expected_profit_A : ℝ := value_per_rabbit * (3 * family_A_prob) - 3 * cost_per_person
def expected_profit_B : ℝ := value_per_rabbit * (family_B_prob1 + family_B_prob2 + family_B_prob3) - 3 * cost_per_person

end noncomputable section

-- State the theorem
theorem folk_festival_theorem :
  correlation_coefficient = (2 * Real.sqrt 2) / 3 ∧
  expected_profit_A > expected_profit_B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folk_festival_theorem_l675_67532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_cost_l675_67533

/-- The cost of Charles' purchases before and after discount --/
structure CharlesPurchase where
  tshirt_cost : ℕ
  backpack_cost : ℕ
  cap_cost : ℕ
  discount : ℕ
  total_after_discount : ℕ

/-- Theorem stating the cost of the t-shirt before discount --/
theorem tshirt_cost 
  (purchase : CharlesPurchase)
  (h1 : purchase.backpack_cost = 10)
  (h2 : purchase.cap_cost = 5)
  (h3 : purchase.total_after_discount = 43)
  (h4 : purchase.discount = 2) :
  purchase.tshirt_cost = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_cost_l675_67533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_proof_l675_67554

theorem simplification_proof :
  ((2/9 - 1/6 + 1/18) * (-18) = -2) ∧
  (54 * (3/4) - (-54) * (1/2) + 54 * (-1/4) = 54) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_proof_l675_67554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greeting_cards_probability_l675_67580

theorem greeting_cards_probability :
  let total_distributions := (3 : ℕ) ^ 4
  let distributions_with_empty := total_distributions - (3 * Nat.choose 4 2)
  (distributions_with_empty : ℚ) / total_distributions = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greeting_cards_probability_l675_67580
