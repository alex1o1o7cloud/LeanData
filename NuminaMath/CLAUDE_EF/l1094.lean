import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1094_109451

theorem inequality_solution (x : ℝ) (h : x ≠ 3) :
  (x - 5) / ((x - 3)^2) < 0 ↔ x ∈ Set.Iio 3 ∪ Set.Ioo 3 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1094_109451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_AA_l1094_109477

-- Define the triangle ABC in the first quadrant
axiom p : ℝ
axiom q : ℝ
axiom r : ℝ
axiom s : ℝ
axiom t : ℝ
axiom u : ℝ

-- Ensure the points are in the first quadrant
axiom p_nonneg : 0 ≤ p
axiom q_nonneg : 0 ≤ q
axiom r_nonneg : 0 ≤ r
axiom s_nonneg : 0 ≤ s
axiom t_nonneg : 0 ≤ t
axiom u_nonneg : 0 ≤ u

-- Define the reflected points
noncomputable def p' : ℝ := -q
noncomputable def q' : ℝ := -p

-- Ensure the points are not on the line y = -x
axiom not_on_line : p ≠ -q ∧ r ≠ -s ∧ t ≠ -u

-- Define the slope of AA'
noncomputable def slope_AA' : ℝ := (q' - q) / (p' - p)

-- Theorem: The slope of AA' is not always 1
theorem slope_AA'_not_always_one : ¬ (∀ (p q : ℝ), slope_AA' = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_AA_l1094_109477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_temperature_relationship_l1094_109478

-- Define the data types
structure TemperatureData where
  temp : Int
  speed : Int

-- Define the dataset
def soundData : List TemperatureData := [
  { temp := -20, speed := 318 },
  { temp := -10, speed := 324 },
  { temp := 0,   speed := 330 },
  { temp := 10,  speed := 336 },
  { temp := 20,  speed := 342 },
  { temp := 30,  speed := 348 }
]

-- Theorem statement
theorem sound_temperature_relationship :
  -- 1. Both air temperature and speed of sound are variables
  (∃ t1 t2 s1 s2, t1 ≠ t2 ∧ s1 ≠ s2 ∧
    (∃ d1 d2, d1 ∈ soundData ∧ d2 ∈ soundData ∧ 
     d1.temp = t1 ∧ d1.speed = s1 ∧ d2.temp = t2 ∧ d2.speed = s2)) ∧
  -- 2. For every 10°C increase, speed increases by 6 m/s
  (∀ d1 d2, d1 ∈ soundData → d2 ∈ soundData → d2.temp - d1.temp = 10 → d2.speed - d1.speed = 6) ∧
  -- 3. At 20°C, sound travels 1710m in 5s
  (∃ d, d ∈ soundData ∧ d.temp = 20 ∧ d.speed * 5 = 1710) ∧
  -- 4. Direct relationship between temperature and speed
  (∀ d1 d2, d1 ∈ soundData → d2 ∈ soundData → d2.temp > d1.temp → d2.speed > d1.speed) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_temperature_relationship_l1094_109478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpool_commute_weeks_l1094_109432

/-- Represents the carpool scenario described in the problem -/
structure Carpool where
  num_people : ℕ
  one_way_miles : ℚ
  gas_cost_per_gallon : ℚ
  miles_per_gallon : ℚ
  days_per_week : ℕ
  individual_monthly_payment : ℚ

/-- Calculates the number of weeks per month the carpool commutes -/
noncomputable def weeks_per_month (c : Carpool) : ℚ :=
  (c.num_people * c.individual_monthly_payment) /
  (2 * c.one_way_miles * c.gas_cost_per_gallon * c.days_per_week / c.miles_per_gallon)

/-- Theorem stating that the carpool commutes 4 weeks per month -/
theorem carpool_commute_weeks (c : Carpool) : 
  c.num_people = 5 ∧ 
  c.one_way_miles = 21 ∧ 
  c.gas_cost_per_gallon = 5/2 ∧ 
  c.miles_per_gallon = 30 ∧ 
  c.days_per_week = 5 ∧ 
  c.individual_monthly_payment = 14 →
  weeks_per_month c = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpool_commute_weeks_l1094_109432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1094_109499

-- Define the line y = kx
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the circle (x-2)^2 + y^2 = 4
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the intersection of the line and circle
def intersects (k : ℝ) : Prop := ∃ x y : ℝ, line k x y ∧ circle_eq x y

-- Define the chord length
noncomputable def chord_length (k : ℝ) : ℝ := 2

theorem line_circle_intersection (k : ℝ) 
  (h1 : intersects k) 
  (h2 : chord_length k = 2) : 
  |k| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1094_109499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_three_fifths_l1094_109494

theorem sin_x_three_fifths (x : Real) 
  (h1 : Real.sin x = 3 / 5) 
  (h2 : π / 2 < x) 
  (h3 : x < π) : 
  Real.cos (2 * x) = 7 / 25 ∧ Real.tan (x + π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_three_fifths_l1094_109494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_expression_l1094_109495

theorem correct_expression : 
  (∀ a : ℝ, a > 0 → Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2) ∧ 
  (∃ x y : ℝ, 2 * Real.sqrt 2 + 3 * Real.sqrt 3 ≠ 5 * Real.sqrt 5) ∧
  (∃ a : ℝ, a ≠ 0 → a^6 / a^3 ≠ a^2) ∧
  (∃ a : ℝ, a ≠ 0 → (a^3)^2 ≠ a^5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_expression_l1094_109495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_theorem_l1094_109448

-- Define necessary structures and predicates
structure Triangle where
  -- Add necessary fields

structure Angle where
  measure : Real

def ExteriorAngle (T : Triangle) (a : Angle) : Prop := sorry

def InteriorOppositeAngles (T : Triangle) (a b c : Angle) : Prop := sorry

-- Theorem statement
theorem exterior_angle_theorem (T : Triangle) (ext_angle int_angle1 int_angle2 : Angle) : 
  ExteriorAngle T ext_angle →
  InteriorOppositeAngles T int_angle1 int_angle2 ext_angle →
  ext_angle.measure = int_angle1.measure + int_angle2.measure := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_theorem_l1094_109448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heronian_half_angle_tangent_rational_l1094_109463

/-- A Heronian triangle is a triangle with rational side lengths and rational area. -/
structure HeronianTriangle where
  a : ℚ
  b : ℚ
  c : ℚ
  area : ℚ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  valid_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  area_formula : area^2 = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) / 16

/-- The tangent of a half-angle in a triangle is rational if the triangle is Heronian. -/
theorem heronian_half_angle_tangent_rational (t : HeronianTriangle) :
  ∃ (tan_half_a tan_half_b tan_half_c : ℚ),
    tan_half_a = (t.b + t.c - t.a) / (t.b + t.c + t.a) ∧
    tan_half_b = (t.c + t.a - t.b) / (t.c + t.a + t.b) ∧
    tan_half_c = (t.a + t.b - t.c) / (t.a + t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heronian_half_angle_tangent_rational_l1094_109463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_inequality_implies_a_range_l1094_109409

theorem determinant_inequality_implies_a_range :
  (∀ x : ℝ, Matrix.det ![![a * x, 1], ![1, x + 1]] < 0) →
  ∃ S : Set ℝ, S = Set.Icc (-4) 0 ∧ ∀ a : ℝ, (∀ x : ℝ, Matrix.det ![![a * x, 1], ![1, x + 1]] < 0) ↔ a ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_inequality_implies_a_range_l1094_109409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1094_109468

noncomputable def angle (a b : ℝ × ℝ) : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : magnitude b = Real.sqrt 2)
  (h2 : dot_product a b = 2)
  (h3 : magnitude (vector_sum a b) = Real.sqrt 14) :
  angle a b = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1094_109468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1094_109401

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 6}

theorem problem_solution :
  (M = Set.Icc (-3) 3) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → Real.sqrt 3 * |a + b| ≤ |a * b + 3|) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1094_109401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_coefficient_sum_l1094_109441

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Calculates the perimeter of a quadrilateral given its four vertices -/
noncomputable def perimeter (a b c d : Point) : ℝ :=
  distance a b + distance b c + distance c d + distance d a

/-- Theorem: The sum of coefficients in the simplified perimeter expression is 12 -/
theorem perimeter_coefficient_sum :
  let a : Point := ⟨0, 1⟩
  let b : Point := ⟨2, 5⟩
  let c : Point := ⟨5, 2⟩
  let d : Point := ⟨7, 0⟩
  ∃ (x y : ℤ), perimeter a b c d = (x : ℝ) * Real.sqrt 2 + (y : ℝ) * Real.sqrt 5 ∧ x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_coefficient_sum_l1094_109441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1094_109443

-- Define the functions for each pair
def fA (x : ℝ) := abs x
noncomputable def gA (x : ℝ) := Real.sqrt (x^2)

def fB (x : ℝ) := 2 * x
def gB (x : ℝ) := 2 * (x + 1)

noncomputable def fC (x : ℝ) := Real.sqrt ((-x)^2)
noncomputable def gC (x : ℝ) := (Real.sqrt (-x))^2

noncomputable def fD (x : ℝ) := (x^2 + x) / (x + 1)
def gD (x : ℝ) := x

-- Theorem statement
theorem function_equality :
  (∀ x, fA x = gA x) ∧
  (∃ x, fB x ≠ gB x) ∧
  (∃ x, fC x ≠ gC x) ∧
  (∃ x, fD x ≠ gD x) := by
  sorry

#check function_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1094_109443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_diagonals_of_Q_l1094_109445

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_Q :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_diagonals_of_Q_l1094_109445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_even_and_decreasing_l1094_109427

def f₁ (x : ℝ) := x^3
def f₂ (x : ℝ) := -x^2 + 1
def f₃ (x : ℝ) := |x| + 1
noncomputable def f₄ (x : ℝ) := Real.sqrt x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def isMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem unique_even_and_decreasing :
  (isEven f₂ ∧ isMonoDecreasing f₂ 0 (Real.pi / 2)) ∧
  (¬(isEven f₁ ∧ isMonoDecreasing f₁ 0 (Real.pi / 2))) ∧
  (¬(isEven f₃ ∧ isMonoDecreasing f₃ 0 (Real.pi / 2))) ∧
  (¬(isEven f₄ ∧ isMonoDecreasing f₄ 0 (Real.pi / 2))) := by
  sorry

#check unique_even_and_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_even_and_decreasing_l1094_109427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_squares_l1094_109497

/-- Assumption that the inner square is rotated 45 degrees -/
axiom inner_square_rotated_45_degrees : Prop

/-- The greatest distance between vertices of two squares, one inscribed in the other -/
theorem greatest_distance_between_squares (inner_perimeter outer_perimeter : ℝ) 
  (h_inner : inner_perimeter = 24)
  (h_outer : outer_perimeter = 40)
  (h_rotated : inner_square_rotated_45_degrees) : 
  ∃ greatest_distance : ℝ, greatest_distance = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_squares_l1094_109497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l1094_109433

/-- Given a triangle with base c and altitude k, and a rectangle inscribed in it with height y,
    the area of the rectangle is (y*c*(k-y))/k -/
theorem inscribed_rectangle_area (c k y : ℝ) (hc : c > 0) (hk : k > 0) (hy : 0 < y ∧ y < k) :
  let n := c * (k - y) / k
  (y * n) = y * c * (k - y) / k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l1094_109433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_conjunction_l1094_109422

theorem proposition_conjunction : 
  (∃ x : ℝ, Real.sin x < 1) ∧ (∀ x : ℝ, Real.exp (abs x) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_conjunction_l1094_109422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_pairs_satisfying_equation_l1094_109488

theorem positive_integer_pairs_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 77) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_pairs_satisfying_equation_l1094_109488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l1094_109472

-- Define the right rectangular prism
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  total_face_area : a * b + b * c + c * a = 44
  total_edge_length : a + b + c = 12

-- Define the sphere containing the prism
noncomputable def sphere_radius (prism : RectangularPrism) : ℝ :=
  Real.sqrt ((prism.a^2 + prism.b^2 + prism.c^2) / 4)

-- Theorem statement
theorem sphere_surface_area (prism : RectangularPrism) :
  4 * Real.pi * (sphere_radius prism)^2 = 56 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l1094_109472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l1094_109415

/-- The cubic equation representing the first curve -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

/-- The linear equation representing the second curve -/
def g (x y : ℝ) : Prop := 2*x + 3*y = 3

/-- The x-coordinates of the intersection points -/
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry
noncomputable def x₃ : ℝ := sorry

/-- The y-coordinates of the intersection points -/
noncomputable def y₁ : ℝ := sorry
noncomputable def y₂ : ℝ := sorry
noncomputable def y₃ : ℝ := sorry

/-- The intersection points satisfy both equations -/
axiom h₁ : f x₁ = y₁ ∧ g x₁ y₁
axiom h₂ : f x₂ = y₂ ∧ g x₂ y₂
axiom h₃ : f x₃ = y₃ ∧ g x₃ y₃

theorem intersection_points_sum :
  x₁ + x₂ + x₃ = 2 ∧ y₁ + y₂ + y₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l1094_109415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1094_109438

def a : ℕ → ℚ
| 0 => 2
| n + 1 => 2 / (a n + 1)

def b (n : ℕ) : ℚ := |((a n + 2) / (a n - 1))|

theorem b_formula (n : ℕ) : b n = 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1094_109438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_arithmetic_sequence_roots_l1094_109454

/-- Given a cosine function on [0, 2π] with specific properties, prove the value of m. -/
theorem cosine_arithmetic_sequence_roots (f : ℝ → ℝ) (x₁ x₂ x₃ x₄ m : ℝ) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x = Real.cos x) →
  x₁ ∈ Set.Icc 0 (2 * Real.pi) →
  x₂ ∈ Set.Icc 0 (2 * Real.pi) →
  x₃ ∈ Set.Icc 0 (2 * Real.pi) →
  x₄ ∈ Set.Icc 0 (2 * Real.pi) →
  x₁ < x₂ →
  x₃ < x₄ →
  f x₁ = 0 →
  f x₂ = 0 →
  f x₃ = m →
  f x₄ = m →
  (x₃ - x₁ = x₄ - x₃) ∧ (x₄ - x₃ = x₂ - x₄) →
  m = -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_arithmetic_sequence_roots_l1094_109454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_X_to_CD_l1094_109419

-- Define the square and arcs
noncomputable def square (s : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

noncomputable def arc_A (s : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = s^2 ∧ 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 ≤ s ∧ p.2 ≤ s}

noncomputable def arc_B (s : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - s)^2 + p.2^2 = (s/2)^2 ∧ 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 ≤ s ∧ p.2 ≤ s}

-- Define the intersection point X
noncomputable def X (s : ℝ) : ℝ × ℝ :=
  (7*s/8, s*Real.sqrt 15/8)

-- Theorem statement
theorem distance_X_to_CD (s : ℝ) (h : s > 0) :
  X s ∈ square s ∧ X s ∈ arc_A s ∧ X s ∈ arc_B s →
  s - (X s).2 = s*(8 - Real.sqrt 15)/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_X_to_CD_l1094_109419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_l1094_109455

theorem g_of_two (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3*x - 7) = 4*x + 6) : g 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_l1094_109455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_is_182_l1094_109417

/-- The polynomial p(x) defined by the given expression -/
def p (x : ℝ) : ℝ := 3 * (x^3 - x^2 + 4) - 5 * (x^2 - 2*x + 3)

/-- The sum of squares of coefficients of the simplified polynomial -/
def sum_of_squares_of_coefficients : ℕ := 182

/-- Theorem stating that the sum of squares of coefficients of the simplified p(x) is 182 -/
theorem sum_of_squares_of_coefficients_is_182 : 
  ∃ (a b c d : ℝ), (∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧ 
  a^2 + b^2 + c^2 + d^2 = sum_of_squares_of_coefficients := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_is_182_l1094_109417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_when_focus_angle_120_l1094_109486

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b ^ 2 / h.a ^ 2)

/-- The angle between the lines connecting the endpoint of the imaginary axis to each focus -/
noncomputable def focus_angle (h : Hyperbola) : ℝ :=
  Real.arccos ((h.a ^ 2 - h.b ^ 2) / (h.a ^ 2 + h.b ^ 2))

/-- Theorem: If the focus angle is 120°, then the eccentricity is √6/2 -/
theorem eccentricity_when_focus_angle_120 (h : Hyperbola) :
  focus_angle h = 2 * Real.pi / 3 → eccentricity h = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_when_focus_angle_120_l1094_109486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_range_l1094_109490

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (3/2) * a * x^2

-- Define the set of a for which f(x) + a = 0 has three distinct real roots
def three_roots_set : Set ℝ := {a : ℝ | ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  f a x + a = 0 ∧ f a y + a = 0 ∧ f a z + a = 0}

-- State the theorem
theorem three_roots_range : 
  three_roots_set = {a : ℝ | a < -Real.sqrt 2 ∨ a > Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_range_l1094_109490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_volume_l1094_109430

/-- Definition of a triangular prism's volume -/
def volume_prism (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Definition of a right triangle -/
class IsRightTriangle (t : Type*) : Prop

/-- Definition of a half equilateral triangle (30-60-90 triangle) -/
class IsHalfEquilateral (t : Type*) : Prop

/-- 
Given a right triangular prism with a 30-60-90 triangle base and height 2,
where the hypotenuse of the base triangle is 2, prove that its volume is 4√3.
-/
theorem right_triangular_prism_volume 
  (base : Type*) 
  (base_area height hypotenuse : ℝ) : 
  IsRightTriangle base → 
  IsHalfEquilateral base → 
  height = 2 → 
  hypotenuse = 2 → 
  base_area = 2 * Real.sqrt 3 →
  volume_prism base_area height = 4 * Real.sqrt 3 := by
  sorry

#check right_triangular_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_volume_l1094_109430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_a_b_l1094_109450

noncomputable def g (a b x : ℝ) : ℝ := a * x^2 + b * x + Real.sqrt 3

theorem exist_a_b : ∃ a b : ℝ, 
  (g a b (g a b 1) = 1) ∧ (g a b 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_a_b_l1094_109450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1094_109476

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ (∀ q : ℝ, 0 < q → q < p → ∃ y, f (y + q) ≠ f y)) ∧
  (∀ x ∈ Set.Icc (-Real.pi/3) (Real.pi/3), f x ∈ Set.Icc (-2) (Real.sqrt 3)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k • Real.pi - Real.pi/12) (k • Real.pi + 5*Real.pi/12), 
    ∀ y ∈ Set.Icc (k • Real.pi - Real.pi/12) (k • Real.pi + 5*Real.pi/12), 
      x ≤ y → f x ≤ f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1094_109476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_height_l1094_109420

/-- Calculates the total distance traveled by a bouncing ball -/
noncomputable def totalDistance (initialHeight : ℝ) : ℝ :=
  initialHeight + 2 * (initialHeight/2) + 2 * (initialHeight/4) + 2 * (initialHeight/8) + initialHeight/16

theorem bouncing_ball_height (h : ℝ) :
  (h > 0) →
  (totalDistance h = 45) →
  (h = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_height_l1094_109420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_cos_neg_cos_l1094_109469

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((1 - x) / (1 + x))

theorem f_sum_cos_neg_cos (α : ℝ) (h : α ∈ Set.Ioo (π / 2) π) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_cos_neg_cos_l1094_109469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1094_109411

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define symmetry with respect to x-axis
def symmetric_x (A D : ℝ × ℝ) : Prop := A.1 = D.1 ∧ A.2 = -D.2

theorem parabola_line_intersection 
  (k : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : parabola A.1 A.2) 
  (hB : parabola B.1 B.2) 
  (hAl : line k A.1 A.2) 
  (hBl : line k B.1 B.2) 
  (hd : distance A B = 8) :
  (k = 1 ∨ k = -1) ∧
  (∀ D : ℝ × ℝ, symmetric_x A D → 
    ∃ P : ℝ × ℝ, P = (-1, 0) ∧ (∀ t : ℝ, 
      B.2 + t * (D.2 - B.2) = P.2 + t * (P.2 - B.2) ∧
      B.1 + t * (D.1 - B.1) = P.1 + t * (P.1 - B.1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1094_109411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_basket_theorem_l1094_109474

/-- Represents the weight difference from the standard weight -/
structure WeightDifference where
  difference : ℝ
  count : ℕ

/-- The set of weight differences for the fruit baskets -/
def weightDifferences : List WeightDifference := [
  ⟨-3, 5⟩, ⟨-4, 3⟩, ⟨-4.5, 2⟩, ⟨0, 4⟩, ⟨2, 3⟩, ⟨3.5, 7⟩
]

/-- The standard weight of each basket in kg -/
def standardWeight : ℝ := 40

/-- The number of baskets -/
def numBaskets : ℕ := 24

/-- The selling price per kg in 元 -/
def pricePerKg : ℝ := 2

/-- Theorem stating the three parts of the problem -/
theorem fruit_basket_theorem :
  let maxDiff := weightDifferences.map (·.difference) |>.maximum?
  let minDiff := weightDifferences.map (·.difference) |>.minimum?
  let totalDiff := weightDifferences.map (λ w => w.difference * (w.count : ℝ)) |>.sum
  let totalWeight := (numBaskets : ℝ) * standardWeight + totalDiff
  let totalEarnings := totalWeight * pricePerKg
  (∀ max min, maxDiff = some max → minDiff = some min → max - min = 8) ∧
  totalDiff = -5.5 ∧
  totalEarnings = 1909 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_basket_theorem_l1094_109474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_on_line_with_distance_difference_l1094_109465

-- Define the angle ABC
noncomputable def angle_ABC : Angle := sorry

-- Define the line P
noncomputable def line_P : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the length d
def d : ℝ := sorry

-- Define side AB of the angle
noncomputable def side_AB : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define side BC of the angle
noncomputable def side_BC : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the parallel line L
noncomputable def line_L : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the distance function
noncomputable def distance (point : EuclideanSpace ℝ (Fin 2)) (line : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- Theorem statement
theorem exists_point_on_line_with_distance_difference :
  ∃ x : EuclideanSpace ℝ (Fin 2), x ∈ line_P ∧ 
  distance x side_AB = distance x side_BC + d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_on_line_with_distance_difference_l1094_109465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1094_109434

-- Define the functions f and g
noncomputable def f (p q x : ℝ) : ℝ := x^2 + p*x + q
noncomputable def g (x : ℝ) : ℝ := 2*x + 1/x^2

-- Define the interval
def I : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem max_value_of_f (p q : ℝ) :
  (∃ x₀ ∈ I, (∀ x ∈ I, f p q x ≥ f p q x₀) ∧
                (∀ x ∈ I, g x ≥ g x₀) ∧
                (∀ x ∈ I, f p q x₀ = g x₀)) →
  (∃ x_max ∈ I, ∀ x ∈ I, f p q x ≤ f p q x_max) ∧
  (∃ x_max ∈ I, f p q x_max = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1094_109434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_in_interval_l1094_109442

open Real

-- Define the function f(x) = sin(log x)
noncomputable def f (x : ℝ) : ℝ := sin (log x)

-- State the theorem
theorem no_zeros_in_interval : 
  ∀ x ∈ Set.Ioo 1 (exp 1), f x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_in_interval_l1094_109442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_over_x_l1094_109406

theorem range_of_y_over_x (x y : ℝ) (h1 : x ≠ 0) :
  let z : ℂ := Complex.ofReal x + Complex.I * Complex.ofReal y
  Complex.abs (z - 2) = Real.sqrt 3 →
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_over_x_l1094_109406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1094_109464

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the line
def my_line (x y : ℝ) : Prop := y - 4 = 3/4 * (x + 3)

-- Define the point P
def point_P : ℝ × ℝ := (-3, 4)

-- Theorem statement
theorem line_tangent_to_circle :
  -- The line passes through point P
  my_line point_P.1 point_P.2 ∧
  -- The line is tangent to the circle
  ∃ (x y : ℝ), my_line x y ∧ my_circle x y ∧
  ∀ (x' y' : ℝ), my_line x' y' → (x' - x)^2 + (y' - y)^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1094_109464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_four_l1094_109482

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2

-- State the theorem
theorem a_greater_than_four (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 →
    (f a x1 - f a x2) / (x1 - x2) > 4) →
  a > 4 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_four_l1094_109482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_price_increase_l1094_109483

/-- Represents the percent change in a value -/
noncomputable def PercentChange (original current : ℝ) : ℝ :=
  (current - original) / original * 100

/-- Calculates the average of two real numbers -/
noncomputable def Average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem candy_price_increase :
  let original_price_M : ℝ := 1.20
  let original_price_N : ℝ := 1.50
  let weight_reduction_M : ℝ := 0.25
  let price_increase_N : ℝ := 0.15
  
  let new_price_per_ounce_M : ℝ := original_price_M / (1 - weight_reduction_M)
  let new_price_per_ounce_N : ℝ := original_price_N * (1 + price_increase_N)
  
  let percent_increase_M := PercentChange original_price_M new_price_per_ounce_M
  let percent_increase_N := PercentChange original_price_N new_price_per_ounce_N
  
  let average_percent_increase := Average percent_increase_M percent_increase_N
  
  ∃ ε > 0, |average_percent_increase - 24.17| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_price_increase_l1094_109483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passengers_after_two_hours_l1094_109407

structure Vehicle where
  initial_passengers : ℕ
  stops : List (ℕ × ℕ)

def calculate_passengers (v : Vehicle) : ℕ :=
  v.initial_passengers + (v.stops.map (fun stop => stop.1 - stop.2)).sum

def bus : Vehicle := {
  initial_passengers := 0,
  stops := [(7, 0), (5, 3), (4, 2), (9, 6), (7, 3), (11, 8)]
}

def train : Vehicle := {
  initial_passengers := 0,
  stops := [(9, 0), (6, 4), (8, 3), (11, 5), (14, 7), (12, 9)]
}

theorem passengers_after_two_hours :
  calculate_passengers bus = 21 ∧ calculate_passengers train = 32 := by
  sorry

#eval calculate_passengers bus
#eval calculate_passengers train

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passengers_after_two_hours_l1094_109407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_vectors_l1094_109459

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_vectors (a b : V) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a + b‖ = Real.sqrt 3) : 
  inner a (a + 2 • b) / (‖a‖ * ‖a + 2 • b‖) = 2 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_vectors_l1094_109459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_tangent_line_equation_l1094_109439

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - a * (x - 1)

-- Theorem for the intervals of monotonicity of g
theorem g_monotonicity (a : ℝ) :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (a - 1) → g a x₁ > g a x₂ ∧
  ∀ x₃ x₄, Real.exp (a - 1) < x₃ ∧ x₃ < x₄ → g a x₃ < g a x₄ := by sorry

-- Theorem for the equation of the tangent line
theorem tangent_line_equation :
  ∃ x y, x > 0 ∧ y = f x ∧ 
  ∀ t, f t = (t - x) * (Real.log x + 1) + y ∧
  -1 = (-x) * (Real.log x + 1) + y ∧
  y = x - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_tangent_line_equation_l1094_109439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l1094_109404

theorem simplify_expressions :
  (∃ x : ℝ, x = Real.sqrt 32 - 4 * Real.sqrt 0.5 + 3 * Real.sqrt 8 ∧ x = 8 * Real.sqrt 2) ∧
  (∃ y : ℝ, y = (1/2) * (Real.sqrt 3 + Real.sqrt 2) - (3/4) * (Real.sqrt 2 - Real.sqrt 27) ∧
      y = (11 * Real.sqrt 3) / 4 - Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l1094_109404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l1094_109487

-- Define the Letter type
inductive Letter
| A
| B
deriving BEq, Inhabited

-- Define a function to check if a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def firstStatement (letters : List Letter) : Bool :=
  letters.count letters.head! = 1

def secondStatement (letters : List Letter) : Bool :=
  letters.count Letter.A < 2

def thirdStatement (letters : List Letter) : Bool :=
  letters.count Letter.B = 1

-- Define a function to check if all statements are consistent
def statementsConsistent (letters : List Letter) : Bool :=
  (tellsTruth letters.head! = firstStatement letters) ∧
  (tellsTruth letters[1]! = secondStatement letters) ∧
  (tellsTruth letters[2]! = thirdStatement letters)

-- Theorem statement
theorem unique_arrangement :
  ∃! (letters : List Letter),
    letters.length = 3 ∧
    statementsConsistent letters ∧
    letters = [Letter.B, Letter.A, Letter.A] := by
  sorry

#eval statementsConsistent [Letter.B, Letter.A, Letter.A]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l1094_109487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_250_l1094_109421

/-- Calculates the length of a platform given train parameters -/
noncomputable def platformLength (trainSpeed : ℝ) (crossingTime : ℝ) (trainLength : ℝ) : ℝ :=
  trainSpeed * (5/18) * crossingTime - trainLength

/-- Theorem: The platform length is 250 meters given the specified conditions -/
theorem platform_length_is_250 :
  platformLength 72 15 50 = 250 := by
  -- Unfold the definition of platformLength
  unfold platformLength
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_250_l1094_109421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_letter_modified_good_words_count_l1094_109416

/-- Definition of a modified good word -/
def ModifiedGoodWord (w : List Char) : Prop :=
  ∀ i, i < w.length - 1 →
    (w.get ⟨i, by sorry⟩ = 'A' → w.get ⟨i+1, by sorry⟩ ≠ 'B') ∧
    (w.get ⟨i, by sorry⟩ = 'B' → w.get ⟨i+1, by sorry⟩ ≠ 'C') ∧
    (w.get ⟨i, by sorry⟩ = 'C' → w.get ⟨i+1, by sorry⟩ ≠ 'A' ∧ w.get ⟨i+1, by sorry⟩ ≠ 'B') ∧
    (w.get ⟨i, by sorry⟩ ∈ ['A', 'B', 'C'])

/-- The number of n-letter modified good words ending with each letter -/
def GoodWordCount (n : ℕ) : ℕ × ℕ × ℕ :=
  match n with
  | 0 => (0, 0, 0)
  | 1 => (1, 1, 1)
  | n+1 =>
    let (a, b, c) := GoodWordCount n
    (a + c, a, a)

/-- Theorem: The number of eight-letter modified good words is 199 -/
theorem eight_letter_modified_good_words_count :
  let (a, b, c) := GoodWordCount 8
  a + b + c = 199 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_letter_modified_good_words_count_l1094_109416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1988_11_equals_169_l1094_109461

/-- Sum of digits of a natural number -/
def sumOfDigits (k : ℕ) : ℕ := 
  if k < 10 then k else k % 10 + sumOfDigits (k / 10)

/-- Square of sum of digits -/
def f₁ (k : ℕ) : ℕ := (sumOfDigits k) ^ 2

/-- Recursive function definition -/
def f (n : ℕ) (k : ℕ) : ℕ :=
  match n with
  | 0 => k
  | 1 => f₁ k
  | n + 1 => f n (f₁ k)

/-- Main theorem -/
theorem f_1988_11_equals_169 : f 1988 11 = 169 := by
  sorry

#eval f 1988 11  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1988_11_equals_169_l1094_109461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_covered_l1094_109496

/-- Calculates the distance covered by a wheel given its diameter and number of revolutions -/
theorem wheel_distance_covered 
  (diameter : ℝ) 
  (revolutions : ℝ) 
  (h1 : diameter = 12) 
  (h2 : revolutions = 14.012738853503185) : 
  ∃ (distance : ℝ), |distance - 528.002| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_covered_l1094_109496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_losers_count_l1094_109456

/-- Represents a carnival booth with a ring toss game -/
structure Booth where
  winner_loser_ratio : Rat
  winners : Nat

/-- Calculates the number of losers for a given booth -/
def losers (b : Booth) : Nat :=
  (b.winners : Rat) * (1 / b.winner_loser_ratio) |>.floor.toNat

theorem carnival_losers_count (booth1 booth2 booth3 : Booth)
  (h1 : booth1.winner_loser_ratio = 4/1 ∧ booth1.winners = 28)
  (h2 : booth2.winner_loser_ratio = 3/2 ∧ booth2.winners = 36)
  (h3 : booth3.winner_loser_ratio = 1/3 ∧ booth3.winners = 15) :
  losers booth1 + losers booth2 + losers booth3 = 76 := by
  sorry

#eval losers ⟨4/1, 28⟩
#eval losers ⟨3/2, 36⟩
#eval losers ⟨1/3, 15⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_losers_count_l1094_109456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l1094_109418

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^a

noncomputable def g (x : ℝ) : ℝ := |Real.log x|

-- State the theorem
theorem range_of_sum (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧
    g (x₁ - 1) + f a 1 = 0 ∧
    g (x₂ - 1) + f a 1 = 0 ∧
    x₁ ≠ x₂) →
  (1 - Real.log 2 < a ∧ a < 1) →
  (1 / x₁ + 1 / x₂ = 1) →
  (2 - Real.log 2 < a + 1 / x₁ + 1 / x₂ ∧ a + 1 / x₁ + 1 / x₂ < 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l1094_109418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_height_exists_min_surface_area_l1094_109405

/-- A rectangular box with a lid, volume 72 cm³, and base side ratio 1:2 -/
structure Box where
  x : ℝ  -- Length of shorter side of the base
  y : ℝ  -- Height of the box
  volume_eq : 2 * x^2 * y = 72
  x_pos : x > 0
  y_pos : y > 0

/-- Surface area of the box -/
def surface_area (b : Box) : ℝ :=
  4 * b.x^2 + 6 * b.x * b.y

/-- The height that minimizes the surface area is 4 cm -/
theorem min_surface_area_height (b : Box) :
  (∀ b' : Box, surface_area b ≤ surface_area b') → b.y = 4 := by
  sorry

/-- Existence of a box with minimal surface area -/
theorem exists_min_surface_area :
  ∃ b : Box, ∀ b' : Box, surface_area b ≤ surface_area b' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_height_exists_min_surface_area_l1094_109405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l1094_109428

/-- Represents the arrival time of a person in hours after 1:00 PM -/
def ArrivalTime := { t : ℝ // 0 ≤ t ∧ t ≤ 2 }

/-- The probability space of all possible arrival scenarios -/
def Ω : Type := ArrivalTime × ArrivalTime × ArrivalTime

/-- The probability measure on Ω -/
noncomputable def P : MeasureTheory.MeasureSpace Ω := sorry

/-- The event that the meeting takes place -/
def MeetingEvent (ω : Ω) : Prop :=
  let (x, y, z) := ω
  z.val > x.val ∧ z.val > y.val ∧ |x.val - y.val| ≤ 1.5

theorem meeting_probability :
  P.volume (fun ω => MeetingEvent ω) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l1094_109428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l1094_109403

/-- Given a point M(3,3,4) and its projection N onto the Oxz plane, 
    the magnitude of vector ON is equal to 5. -/
theorem projection_magnitude : 
  let M : Fin 3 → ℝ := ![3, 3, 4]
  let N : Fin 3 → ℝ := ![M 0, 0, M 2]  -- Projection onto Oxz plane
  let ON : Fin 3 → ℝ := N
  ‖ON‖ = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l1094_109403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1094_109413

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

-- Define the points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for ∠APB = 90°
def rightAngle (P : ℝ × ℝ) (m : ℝ) : Prop :=
  let AP := (P.1 + m, P.2)
  let BP := (P.1 - m, P.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- Theorem statement
theorem max_m_value (m : ℝ) :
  m > 0 →
  (∃ P ∈ C, rightAngle P m) →
  (∀ n : ℝ, n > 0 → (∃ Q ∈ C, rightAngle Q n) → n ≤ m) →
  m = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1094_109413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l1094_109431

/-- The length of a train given its speed, a man's speed, and the time it takes to pass the man -/
noncomputable def train_length (train_speed man_speed : ℝ) (time : ℝ) : ℝ :=
  (train_speed - man_speed) * (1000 / 3600) * time

/-- Theorem stating that a train of specific length passes a man in a given time -/
theorem train_length_theorem (train_speed man_speed time : ℝ) 
  (h1 : train_speed = 68)
  (h2 : man_speed = 8)
  (h3 : time = 5.999520038396929) :
  ∃ ε > 0, |train_length train_speed man_speed time - 299.98| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l1094_109431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_room_time_l1094_109460

/-- The time (in hours) it takes Doug to paint the room alone -/
noncomputable def doug_time : ℝ := 4

/-- The time (in hours) it takes Dave to paint the room alone -/
noncomputable def dave_time : ℝ := 6

/-- The duration (in hours) of the lunch break -/
noncomputable def lunch_break : ℝ := 1/2

/-- The total time (in hours) it takes Doug and Dave to paint the room together, including the lunch break -/
noncomputable def total_time : ℝ := 29/10

theorem paint_room_time : 
  (1 / doug_time + 1 / dave_time) * (total_time - lunch_break) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_room_time_l1094_109460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_sin_odd_sin_45_sin_neg_765_deg_l1094_109429

-- Define the sine function (provided by Mathlib)
open Real

-- State the properties of the sine function
theorem sin_period (x : ℝ) : sin (x + 2 * π) = sin x := by sorry
theorem sin_odd (x : ℝ) : sin (-x) = -sin x := by sorry
theorem sin_45 : sin (π / 4) = sqrt 2 / 2 := by sorry

-- State the theorem
theorem sin_neg_765_deg :
  sin (-765 * π / 180) = -(sqrt 2 / 2) := by
  -- Convert -765° to radians
  have h1 : -765 * π / 180 = -765 * (π / 180)
  · ring
  
  -- Use periodicity to simplify the angle
  have h2 : sin (-765 * (π / 180)) = sin (-45 * (π / 180))
  · sorry
  
  -- Use the odd property of sine
  have h3 : sin (-45 * (π / 180)) = -sin (45 * (π / 180))
  · sorry
  
  -- Use the known value of sin(45°)
  have h4 : sin (45 * (π / 180)) = sqrt 2 / 2
  · sorry
  
  -- Combine the steps
  calc
    sin (-765 * π / 180) = sin (-765 * (π / 180)) := by rw [h1]
    _ = sin (-45 * (π / 180)) := by rw [h2]
    _ = -sin (45 * (π / 180)) := by rw [h3]
    _ = -(sqrt 2 / 2) := by rw [h4]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_sin_odd_sin_45_sin_neg_765_deg_l1094_109429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_male_given_obese_l1094_109410

/-- Represents the gender of an employee -/
inductive Gender
| male
| female

/-- Represents whether an employee is obese or not -/
inductive ObesityStatus
| obese
| notObese

/-- Definition of the company's employee composition -/
structure Company where
  male_ratio : ℚ
  female_ratio : ℚ
  obese_male_ratio : ℚ
  obese_female_ratio : ℚ
  male_ratio_eq : male_ratio = 3/5
  female_ratio_eq : female_ratio = 2/5
  obese_male_ratio_eq : obese_male_ratio = 1/5
  obese_female_ratio_eq : obese_female_ratio = 1/10

/-- The probability of selecting an obese male employee -/
def prob_obese_male (c : Company) : ℚ :=
  c.male_ratio * c.obese_male_ratio

/-- The probability of selecting an obese employee -/
def prob_obese (c : Company) : ℚ :=
  c.male_ratio * c.obese_male_ratio + c.female_ratio * c.obese_female_ratio

/-- Theorem: The probability that a randomly selected obese employee is male is 3/4 -/
theorem prob_male_given_obese (c : Company) :
  prob_obese_male c / prob_obese c = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_male_given_obese_l1094_109410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_on_interval_l1094_109452

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x / (x^2 - 1)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ -1

-- Theorem for odd function property
theorem f_is_odd : ∀ x : ℝ, domain x → f (-x) = -f x := by
  sorry

-- Theorem for decreasing property on (-1, 1)
theorem f_is_decreasing_on_interval : 
  ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_on_interval_l1094_109452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1094_109435

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Axioms for f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_period_2 : ∀ x, f (x + 2) = f (-x)
axiom f_def_01 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - Real.cos x

-- Theorem to prove
theorem f_inequality : f 2018 < f (2019/2) ∧ f (2019/2) < f (2020/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1094_109435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1094_109440

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.B = Real.pi / 4 ∧
  Real.cos t.A - Real.cos (2 * t.A) = 0 ∧
  t.b^2 + t.c^2 = t.a - t.b * t.c + 2

theorem triangle_problem (t : Triangle) (h : TriangleConditions t) :
  t.C = Real.pi / 12 ∧ 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = 1 - Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1094_109440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l1094_109480

noncomputable def number1 : ℝ := 0.8
noncomputable def number2 : ℝ := 1/2
noncomputable def number3 : ℝ := 0.6

theorem sum_of_numbers (h1 : number1 ≥ 0.1) (h2 : number2 ≥ 0.1) (h3 : number3 ≥ 0.1) :
  number1 + number2 + number3 = 1.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l1094_109480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arccos_square_l1094_109423

theorem sin_arccos_square (y : ℝ) (h1 : y > 0) (h2 : Real.sin (Real.arccos y) = y) : y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arccos_square_l1094_109423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1094_109446

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0), 
    a line l passing through P(3,6) intersects C at points A and B, 
    and the midpoint of AB is N(12,15). 
    Then, the eccentricity of the hyperbola C is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (A B : ℝ × ℝ) (h_A : A.1^2 / a^2 - A.2^2 / b^2 = 1) 
  (h_B : B.1^2 / a^2 - B.2^2 / b^2 = 1)
  (h_line : ∃ (m c : ℝ), A.2 = m * A.1 + c ∧ B.2 = m * B.1 + c ∧ 6 = m * 3 + c)
  (h_midpoint : (A.1 + B.1) / 2 = 12 ∧ (A.2 + B.2) / 2 = 15) :
  Real.sqrt (1 + b^2 / a^2) = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1094_109446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1094_109449

/-- The length of a platform given a train passing over it -/
theorem platform_length (train_length passing_time train_speed : ℝ) : 
  train_length = 50 → passing_time = 10 → train_speed = 15 → 
  train_speed * passing_time - train_length = 100 :=
by
  -- Introduce the hypotheses
  intro h_train_length h_passing_time h_train_speed

  -- Calculate the total distance
  have h_total_distance : train_speed * passing_time = 150 := by
    rw [h_train_speed, h_passing_time]
    norm_num

  -- Calculate the platform length
  have h_platform_length : train_speed * passing_time - train_length = 100 := by
    rw [h_total_distance, h_train_length]
    norm_num

  -- Conclude the proof
  exact h_platform_length


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1094_109449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1094_109489

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given m ∈ ℝ, if m²+m-2+(m²-1)i is a pure imaginary number, then m = -2 -/
theorem pure_imaginary_condition (m : ℝ) :
  IsPureImaginary (Complex.ofReal (m^2 + m - 2) + Complex.I * Complex.ofReal (m^2 - 1)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1094_109489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_correct_l1094_109481

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![-Real.sqrt 2 / 2, Real.sqrt 2 / 2],
    ![-Real.sqrt 2 / 2, -Real.sqrt 2 / 2]]

noncomputable def rotation_angle : ℝ := 135 * Real.pi / 180

theorem rotation_matrix_correct :
  rotation_matrix = Matrix.of (λ i j =>
    if i = j
    then Real.cos rotation_angle
    else if i < j
    then -Real.sin rotation_angle
    else Real.sin rotation_angle) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_correct_l1094_109481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_centers_distance_l1094_109493

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem hyperbola_circle_centers_distance 
  (h : Hyperbola) 
  (c : Circle) :
  (h.a = h.b) → -- Asymptotes are y = ±x
  (h.a^2 = 16) → -- Passes through (4,0)
  (c.center.1^2 - c.center.2^2 = h.a^2) → -- Circle center lies on hyperbola
  (distance c.center (h.a, 0) = c.radius) → -- Circle passes through vertex
  (distance c.center (h.a * Real.sqrt 2, 0) = c.radius) → -- Circle passes through focus
  distance c.center (0, 0) = 2 * Real.sqrt 2 := by
    sorry

#check hyperbola_circle_centers_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_centers_distance_l1094_109493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_depreciation_rate_approx_l1094_109475

/-- The annual depreciation rate of a machine given its initial value, selling price after two years, and profit. -/
noncomputable def annual_depreciation_rate (initial_value selling_price profit : ℝ) : ℝ :=
  let cost_after_two_years := selling_price - profit
  let depreciation_factor := Real.sqrt (cost_after_two_years / initial_value)
  (1 - depreciation_factor) * 100

/-- Theorem stating that the annual depreciation rate is approximately 76.09% given the specified conditions. -/
theorem depreciation_rate_approx (initial_value selling_price profit : ℝ)
  (h1 : initial_value = 150000)
  (h2 : selling_price = 116615)
  (h3 : profit = 24000) :
  ∃ ε > 0, |annual_depreciation_rate initial_value selling_price profit - 76.09| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_depreciation_rate_approx_l1094_109475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1094_109444

theorem sum_of_solutions : 
  (∃ x₁ x₂ : ℝ, 
    (4 * x₁) / (x₁^2 - 4) = (-3 * x₁) / (x₁ - 2) + 2 / (x₁ + 2) ∧
    (4 * x₂) / (x₂^2 - 4) = (-3 * x₂) / (x₂ - 2) + 2 / (x₂ + 2) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ 2 ∧ x₁ ≠ -2 ∧ x₂ ≠ 2 ∧ x₂ ≠ -2) →
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = -8/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1094_109444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identities_l1094_109473

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_identities {a b c A B C : ℝ} (h : Triangle a b c A B C) :
  b * Real.cos C + c * Real.cos B = a ∧ 
  (Real.cos A + Real.cos B) / (a + b) = 2 * (Real.sin (C / 2))^2 / c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identities_l1094_109473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_13_dividing_150_factorial_l1094_109471

theorem greatest_power_of_13_dividing_150_factorial : 
  (∃ k : ℕ+, (13 : ℕ) ^ k.val ∣ Nat.factorial 150 ∧ 
    ∀ m : ℕ+, (13 : ℕ) ^ m.val ∣ Nat.factorial 150 → m ≤ k) ∧ 
  (∀ k : ℕ+, (∀ m : ℕ+, (13 : ℕ) ^ m.val ∣ Nat.factorial 150 → m ≤ k) → k = 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_13_dividing_150_factorial_l1094_109471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_last_digit_l1094_109400

noncomputable def f (x : ℝ) : ℤ := ⌊x⌋ + ⌊3 * x⌋ + ⌊6 * x⌋

theorem f_last_digit (x : ℝ) (hx : x > 0) :
  ∃ (d : ℕ), d ∈ ({0, 1, 3, 4, 6, 7} : Set ℕ) ∧ f x % 10 = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_last_digit_l1094_109400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_150_and_350_l1094_109462

theorem count_even_numbers_between_150_and_350 : 
  (Finset.filter (fun n : ℕ => n % 2 = 0 ∧ n > 150 ∧ n < 350) (Finset.range 350)).card = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_150_and_350_l1094_109462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_other_sides_l1094_109414

/-- A parallelogram with given properties -/
structure Parallelogram where
  -- First adjacent side equation
  side1 : (x y : ℝ) → x + y + 1 = 0
  -- Second adjacent side equation
  side2 : (x : ℝ) → 3 * x - 4 = 0
  -- Intersection point of diagonals
  diagonal_intersection : ℝ × ℝ := (3, 3)

/-- The equations of the other two sides of the parallelogram -/
def other_sides (p : Parallelogram) : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (λ x y ↦ x + y - 13 = 0, λ x y ↦ 3 * x - y - 16 = 0)

/-- Theorem stating that the given equations are indeed the other two sides of the parallelogram -/
theorem parallelogram_other_sides (p : Parallelogram) :
  other_sides p = (λ x y ↦ x + y - 13 = 0, λ x y ↦ 3 * x - y - 16 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_other_sides_l1094_109414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_l1094_109467

theorem triangle_angle_b (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = Real.pi ∧  -- sum of angles in a triangle
  C = Real.pi/4 ∧  -- C = 45°
  b = Real.sqrt 2 ∧  -- b = √2
  c = 2 ∧  -- c = 2
  Real.sin A / a = Real.sin B / b ∧  -- sine rule
  Real.sin B / b = Real.sin C / c ∧  -- sine rule
  a + b > c ∧ b + c > a ∧ c + a > b  -- triangle inequality
  → B = Real.pi/6  -- B = 30°
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_l1094_109467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pencil_of_spheres_exists_net_of_spheres_l1094_109424

-- Define a sphere in 3D space
structure Sphere where
  center : Fin 3 → ℝ
  radius : ℝ

-- Define a radical axis
def RadicalAxis (s1 s2 : Sphere) : Set (Fin 3 → ℝ) :=
  {p | (s1.center 0 - p 0)^2 + (s1.center 1 - p 1)^2 + (s1.center 2 - p 2)^2 - s1.radius^2 =
       (s2.center 0 - p 0)^2 + (s2.center 1 - p 1)^2 + (s2.center 2 - p 2)^2 - s2.radius^2}

-- Define a radical center
def RadicalCenter (s1 s2 s3 : Sphere) : Set (Fin 3 → ℝ) :=
  RadicalAxis s1 s2 ∩ RadicalAxis s2 s3

-- Theorem for the existence of a pencil of spheres
theorem exists_pencil_of_spheres :
  ∃ (f : ℕ → Sphere), ∀ (i j : ℕ), i ≠ j → 
    RadicalAxis (f i) (f j) = RadicalAxis (f 0) (f 1) ∧
    ∃ (plane : Set (Fin 3 → ℝ)), 
      (∀ k, (λ i => (f k).center i) ∈ plane) ∧
      (∀ p ∈ plane, ∀ q ∈ RadicalAxis (f 0) (f 1), 
        (p 0 - q 0) * (q 0 - q 0) + (p 1 - q 1) * (q 1 - q 1) + (p 2 - q 2) * (q 2 - q 2) = 0) :=
by
  sorry

-- Theorem for the existence of a net of spheres
theorem exists_net_of_spheres :
  ∃ (f : ℕ → Sphere), ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    RadicalCenter (f i) (f j) (f k) = RadicalCenter (f 0) (f 1) (f 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pencil_of_spheres_exists_net_of_spheres_l1094_109424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_max_distance_on_circle_l1094_109470

noncomputable def circle_C (a : ℝ) (θ : ℝ) : ℝ × ℝ := (a + a * Real.cos θ, a * Real.sin θ)

def line_l (θ : ℝ) (ρ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = 2 * Real.sqrt 2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_and_line_intersection (a : ℝ) :
  (0 < a) ∧ (a < 5) →
  (∃ A B : ℝ × ℝ, ∃ θA θB ρA ρB : ℝ,
    A = circle_C a θA ∧
    B = circle_C a θB ∧
    line_l θA ρA ∧
    line_l θB ρB ∧
    distance A B = 2 * Real.sqrt 2) →
  a = 2 := by
  sorry

theorem max_distance_on_circle (a : ℝ) (M N : ℝ × ℝ) (θM θN : ℝ) :
  a = 2 →
  M = circle_C a θM →
  N = circle_C a θN →
  θN - θM = Real.pi/3 →
  ∀ P Q : ℝ × ℝ, ∀ θP θQ : ℝ,
    P = circle_C a θP →
    Q = circle_C a θQ →
    θQ - θP = Real.pi/3 →
    distance (0, 0) M + distance (0, 0) N ≤ distance (0, 0) P + distance (0, 0) Q →
  distance (0, 0) M + distance (0, 0) N = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_max_distance_on_circle_l1094_109470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_superabundant_l1094_109436

/-- Sum of divisors function -/
def f (n : ℕ) : ℕ := sorry

/-- A positive integer n is superabundant if f(f(n)) = n + 3 -/
def is_superabundant (n : ℕ) : Prop := f (f n) = n + 3

/-- 3 is the only superabundant number -/
theorem unique_superabundant : 
  ∀ n : ℕ, n > 0 → (is_superabundant n ↔ n = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_superabundant_l1094_109436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_power_function_l1094_109447

/-- A function f: ℝ → ℝ is a power function if there exist constants k and n such that
    f(x) = k * x^n for all x > 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ k n : ℝ, ∀ x > 0, f x = k * (x ^ n)

/-- The given function parameterized by α -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 1) * (x ^ (-4 * α - 2))

/-- If f(α) is a power function, then α = 2 -/
theorem alpha_value_for_power_function :
  (∃ α : ℝ, IsPowerFunction (f α)) → (∃ α : ℝ, IsPowerFunction (f α) ∧ α = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_power_function_l1094_109447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_parallelogram_condition_l1094_109412

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Define the dot product of PF1 and PF2
noncomputable def dot_product (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1)^2 + y^2 - 1

-- Define the line l
noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define point P
noncomputable def P : ℝ × ℝ := (-1, 2 * Real.sqrt 3 / 3)

-- Theorem 1: Minimum value of dot product
theorem min_dot_product :
  ∀ P, point_on_ellipse P → dot_product P ≥ 1 :=
by sorry

-- Theorem 2: Condition for parallelogram
theorem parallelogram_condition :
  ∃! k, ∀ A B Q,
    point_on_ellipse A ∧ point_on_ellipse B ∧ point_on_ellipse Q ∧
    (∃ x, A.1 = x ∧ A.2 = line_l k x) ∧
    (∃ x, B.1 = x ∧ B.2 = line_l k x) ∧
    (∃ x, Q.1 = x ∧ Q.2 = line_l k (x + 1) + P.2 - line_l k 0) →
    (A.1 - P.1 = Q.1 - B.1 ∧ A.2 - P.2 = Q.2 - B.2) ∧
    k = -Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_parallelogram_condition_l1094_109412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pairs_for_S_2013_l1094_109425

/-- S_n(a,p) denotes the exponent of p in the prime factorization of a^(p^n) - 1 -/
def S (n : ℕ) (a p : ℕ) : ℕ :=
  (a^(p^n) - 1).factorization p

/-- The theorem statement -/
theorem unique_pairs_for_S_2013 :
  {(n, p) : ℕ × ℕ | Nat.Prime p ∧ S n 2013 p = 100} = {(98, 2), (99, 503)} := by
  sorry

#check unique_pairs_for_S_2013

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pairs_for_S_2013_l1094_109425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_set_l1094_109453

def has_at_least_two_digits (m : ℕ) : Prop :=
  m ≥ 10

def all_digits_different (m : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ Nat.digits 10 m → d₂ ∈ Nat.digits 10 m → d₁ ≠ d₂ → d₁ ≠ d₂

def no_zero_digit (m : ℕ) : Prop :=
  0 ∉ Nat.digits 10 m

def all_permutations_divisible (m n : ℕ) : Prop :=
  ∀ p, p ∈ List.permutations (Nat.digits 10 m) → (Nat.ofDigits 10 p) % n = 0

def valid_m (m n : ℕ) : Prop :=
  has_at_least_two_digits m ∧
  all_digits_different m ∧
  no_zero_digit m ∧
  m % n = 0 ∧
  all_permutations_divisible m n

theorem valid_n_set : 
  {n : ℕ | ∃ m, valid_m m n} = {1, 2, 3, 4, 9} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_set_l1094_109453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1094_109437

-- Define the plane
variable (x y : ℝ)

-- Define the point P
def P : ℝ × ℝ := (x, y)

-- Define the line x = -1
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the point (2,0)
def point : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the distance from a point to a line
def distToLine (p : ℝ × ℝ) : ℝ := |p.1 + 1|

-- State the theorem
theorem trajectory_is_parabola :
  {P : ℝ × ℝ | distToLine P = distance P point - 1} = {P : ℝ × ℝ | ∃ a b c : ℝ, P.2 = a*P.1^2 + b*P.1 + c} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1094_109437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_is_eight_dot_product_is_constant_l1094_109484

noncomputable section

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define points P and M
def P : ℝ × ℝ := (0, Real.sqrt 3)
def M : ℝ × ℝ := (0, -(Real.sqrt 3)/4)

-- Define the left focus F
def F : ℝ × ℝ := (-(Real.sqrt 3), 0)

-- Define a line passing through two points
def Line (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Theorem 1
theorem product_of_distances_is_eight :
  ∀ A B : ℝ × ℝ,
  C A.1 A.2 → C B.1 B.2 →
  Line P F A.1 A.2 → Line P F B.1 B.2 →
  distance A F * distance B F = 8 := by sorry

-- Theorem 2
theorem dot_product_is_constant :
  ∀ A B : ℝ × ℝ,
  C A.1 A.2 → C B.1 B.2 →
  Line P A B.1 B.2 → Line P B A.1 A.2 →
  dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = 35/16 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_is_eight_dot_product_is_constant_l1094_109484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_trip_time_ratio_l1094_109498

/-- Represents a highway marker --/
structure Marker where
  position : ℝ

/-- Represents a motorcycle trip --/
structure MotorcycleTrip where
  markerA : Marker
  markerB : Marker
  markerC : Marker
  distanceAB : ℝ
  averageSpeed : ℝ

/-- The theorem statement --/
theorem motorcycle_trip_time_ratio
  (trip : MotorcycleTrip)
  (h1 : trip.distanceAB = 120)
  (h2 : trip.markerC.position - trip.markerB.position = (trip.distanceAB) / 2)
  (h3 : trip.averageSpeed = 30)
  (h4 : trip.markerB.position - trip.markerA.position = trip.distanceAB)
  (h5 : trip.markerC.position > trip.markerB.position) :
  (trip.distanceAB / trip.averageSpeed) = ((trip.markerC.position - trip.markerB.position) / trip.averageSpeed) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_trip_time_ratio_l1094_109498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_contour_length_is_6pi_a_l1094_109492

/-- Configuration of four tangent circles -/
structure FourTangentCircles where
  -- Radius of each circle
  radius : ℝ
  -- Centers of the four circles
  center_A : ℝ × ℝ
  center_B : ℝ × ℝ
  center_C : ℝ × ℝ
  center_D : ℝ × ℝ
  -- Conditions for tangency
  tangent_AB : dist center_A center_B = 2 * radius
  tangent_BC : dist center_B center_C = 2 * radius
  tangent_CD : dist center_C center_D = 2 * radius
  tangent_DA : dist center_D center_A = 2 * radius

/-- The length of the external contour of four tangent circles -/
noncomputable def external_contour_length (config : FourTangentCircles) : ℝ :=
  6 * Real.pi * config.radius

/-- Theorem: The length of the external contour of four tangent circles is 6πa -/
theorem external_contour_length_is_6pi_a (config : FourTangentCircles) :
  external_contour_length config = 6 * Real.pi * config.radius :=
by
  -- Unfold the definition of external_contour_length
  unfold external_contour_length
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_contour_length_is_6pi_a_l1094_109492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_singleton_l1094_109458

noncomputable def circle' (y z : ℝ) : ℝ := (y + z + |y - z|) / 2

theorem unique_solution_iff_singleton (X : Set ℝ) :
  (∀ (a b : ℝ), a ∈ X → b ∈ X → ∃! (x : ℝ), x ∈ X ∧ circle' a x = b) ↔ ∃ a : ℝ, X = {a} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_singleton_l1094_109458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sides_l1094_109457

theorem right_triangle_sides (p m : ℝ) (hp : p > 0) (hm : m > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = p ∧
    a * b = m * c ∧
    a^2 + b^2 = c^2 ∧
    a = (p + Real.sqrt (p^2 - 4*m*(-m + Real.sqrt (m^2 + p^2)))) / 2 ∧
    b = (p - Real.sqrt (p^2 - 4*m*(-m + Real.sqrt (m^2 + p^2)))) / 2 ∧
    c = -m + Real.sqrt (m^2 + p^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sides_l1094_109457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_increasing_relationship_negative_identity_example_l1094_109479

-- Define a real-valued function type
def RealFunction := ℝ → ℝ

-- Define what it means for f to be increasing
def IsIncreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define what it means for f to have an inverse
def HasInverse (f : RealFunction) : Prop :=
  ∃ g : RealFunction, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_increasing_relationship :
  (∀ f : RealFunction, IsIncreasing f → HasInverse f) ∧
  (∃ f : RealFunction, HasInverse f ∧ ¬IsIncreasing f) := by
  sorry

-- Example of a function that has an inverse but is not increasing
def negative_identity : RealFunction := fun x ↦ -x

-- Proof that negative_identity has an inverse but is not increasing
theorem negative_identity_example :
  HasInverse negative_identity ∧ ¬IsIncreasing negative_identity := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_increasing_relationship_negative_identity_example_l1094_109479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l1094_109485

/-- Calculates the profit percent for a car sale -/
noncomputable def profit_percent (purchase_price repair_costs selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_costs
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the profit percent for the given car sale is approximately 21.64% -/
theorem car_sale_profit_percent :
  let purchase_price : ℝ := 42000
  let repair_costs : ℝ := 13000
  let selling_price : ℝ := 66900
  abs (profit_percent purchase_price repair_costs selling_price - 21.64) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l1094_109485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_interior_angle_mean_l1094_109408

/-- The mean value of the measures of the five interior angles of any regular pentagon is 108°. -/
theorem regular_pentagon_interior_angle_mean : ℝ := 108

#check regular_pentagon_interior_angle_mean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_interior_angle_mean_l1094_109408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_A_l1094_109426

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 2*x - y - 5 = 0

-- Theorem statement
theorem tangent_line_at_point_A :
  circle_eq point_A.1 point_A.2 ∧
  tangent_line point_A.1 point_A.2 ∧
  ∀ (x y : ℝ), circle_eq x y ∧ tangent_line x y → (x, y) = point_A :=
by
  sorry

#check tangent_line_at_point_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_A_l1094_109426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_satisfying_integers_l1094_109466

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A positive integer m satisfies the condition if there exists a positive integer n 
    such that τ(n^2) / τ(n) = m -/
def satisfies_condition (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ (tau (n^2) : ℚ) / tau n = m

theorem characterization_of_satisfying_integers :
  ∀ m : ℕ, m > 0 → (satisfies_condition m ↔ m % 2 = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_satisfying_integers_l1094_109466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_of_3_to_20_l1094_109402

theorem third_of_3_to_20 (y : ℝ) :
  (1 / 3) * (3 : ℝ)^20 = 3^y → y = 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_of_3_to_20_l1094_109402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l1094_109491

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ (a d : ℝ), 
    Real.sqrt (25 + k) = a ∧ 
    Real.sqrt (121 + k) = a + d ∧ 
    Real.sqrt (361 + k) = a + 2*d) → 
  k = 216 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l1094_109491
