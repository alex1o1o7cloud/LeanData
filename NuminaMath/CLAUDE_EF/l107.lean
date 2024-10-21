import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_sum_l107_10774

/-- Two vectors are colinear if one is a scalar multiple of the other -/
def Colinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem colinear_vectors_sum (m n : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, 3, m)
  let b : ℝ × ℝ × ℝ := (2*n, 6, 8)
  Colinear a b → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_sum_l107_10774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_prediction_may_differ_l107_10764

/-- Represents a linear regression model -/
structure LinearRegression where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Predicted value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.m * x + model.b

/-- Actual observed value, which may differ from the predicted value -/
noncomputable def observe : ℝ → ℝ := sorry

/-- Statement: The actual observed value may not always equal the predicted value -/
theorem regression_prediction_may_differ (model : LinearRegression) (x : ℝ) :
  ∃ (y : ℝ), observe x ≠ predict model x := by
  sorry

#check regression_prediction_may_differ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_prediction_may_differ_l107_10764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supply_cost_l107_10787

/-- Calculates the total cost of pencils and pens with discounts applied --/
def total_cost_with_discounts (pencil_price pen_price : ℚ) 
                               (pencil_discount pen_discount : ℚ) 
                               (pencil_discount_threshold pen_discount_threshold : ℕ) 
                               (num_pencils num_pens : ℕ) : ℚ :=
  let pencil_cost := pencil_price * num_pencils
  let pen_cost := pen_price * num_pens
  let pencil_discount_amount := if num_pencils > pencil_discount_threshold 
                                then pencil_discount * pencil_cost 
                                else 0
  let pen_discount_amount := if num_pens > pen_discount_threshold 
                             then pen_discount * pen_cost 
                             else 0
  pencil_cost + pen_cost - pencil_discount_amount - pen_discount_amount

/-- Theorem stating that the total cost of 38 pencils and 56 pens is $252.10 --/
theorem school_supply_cost : 
  total_cost_with_discounts (5/2) (7/2) (1/10) (3/20) 30 50 38 56 = 12605/50 := by
  sorry

#eval total_cost_with_discounts (5/2) (7/2) (1/10) (3/20) 30 50 38 56

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supply_cost_l107_10787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_15_l107_10761

/-- The probability that no two of three randomly chosen numbers from [0, m] are within 2 units of each other -/
noncomputable def probability (m : ℝ) : ℝ := (m - 4)^3 / m^3

/-- 15 is the smallest positive integer m such that the probability is greater than 1/2 -/
theorem smallest_m_is_15 :
  ∀ m : ℕ+, 
  (∀ k : ℕ+, k < m → probability k ≤ 1/2) ∧
  probability m > 1/2 →
  m = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_15_l107_10761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_a_value_three_distinct_solutions_range_l107_10775

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 6*x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 - 6

-- Theorem for Part I
theorem tangent_line_a_value :
  ∃ (a : ℝ), 
    (f' 0 = -6) ∧ 
    (∃ (m : ℝ), m = -6 ∧ a = (1/2) * m^2) := by
  sorry

-- Theorem for Part II
theorem three_distinct_solutions_range (a : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = g a x ∧ f y = g a y ∧ f z = g a z) ↔
  (9/2 < a ∧ a < 22/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_a_value_three_distinct_solutions_range_l107_10775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_decrease_l107_10773

/-- Proves that given an article with an original cost of 75 Rs, 
    after a 20% increase and then a decrease to 72 Rs, 
    the percentage decrease is 20%. -/
theorem article_cost_decrease (original_cost : ℝ) (initial_increase_percent : ℝ) 
  (final_cost : ℝ) (h1 : original_cost = 75) 
  (h2 : initial_increase_percent = 20) (h3 : final_cost = 72) : 
  (original_cost * (1 + initial_increase_percent / 100) - final_cost) / 
  (original_cost * (1 + initial_increase_percent / 100)) * 100 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_decrease_l107_10773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_of_triangle_l107_10714

/-- Given a triangle with vertices A(-1, 2), B(3, -1), and C(-1, -3),
    the equation of the line on which the median to side BC lies is y = -2x. -/
theorem median_equation_of_triangle (A B C : ℝ × ℝ) : 
  A = (-1, 2) → B = (3, -1) → C = (-1, -3) →
  ∃ (m b : ℝ), m = -2 ∧ b = 0 ∧
  ∀ (x y : ℝ), (x, y) ∈ Set.range (λ t : ℝ ↦ (t, m * t + b)) ↔
  (∃ (t : ℝ), x = t ∧ y = m * t + b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_of_triangle_l107_10714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_halves_l107_10703

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2

/-- Function to calculate the angle at a vertex of a triangle -/
noncomputable def AngleAt (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a triangle ABC and three points A', B', C' satisfying certain conditions,
    prove that the angles of triangle A'B'C' are half of the angles α, β, γ. -/
theorem triangle_angle_halves
  (A B C A' B' C' : ℝ × ℝ)  -- Points in 2D plane
  (α β γ : ℝ)               -- Angles in radians
  (h1 : IsIsosceles A' B C) -- A'BC is isosceles
  (h2 : IsIsosceles B' A C) -- AB'C is isosceles
  (h3 : IsIsosceles C' A B) -- ABC' is isosceles
  (h4 : AngleAt A' B C = α) -- Angle at A' in A'BC is α
  (h5 : AngleAt B' A C = β) -- Angle at B' in AB'C is β
  (h6 : AngleAt C' A B = γ) -- Angle at C' in ABC' is γ
  (h7 : α + β + γ = 2 * Real.pi)  -- Sum of angles is 2π
  : AngleAt A' B' C' = α / 2 ∧
    AngleAt B' C' A' = β / 2 ∧
    AngleAt C' A' B' = γ / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_halves_l107_10703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l107_10744

-- Define the radius of the original ball
noncomputable def original_radius : ℝ := 1

-- Define the side length of the cubic container
noncomputable def cube_side_length : ℝ := 4

-- Define the volume of a sphere given its radius
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * (radius ^ 3)

-- Define the volume of a cube given its side length
noncomputable def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

-- Theorem: The maximum number of balls that can fit in the cube is 16
theorem max_balls_in_cube : 
  ⌊(cube_volume cube_side_length) / (sphere_volume original_radius)⌋ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l107_10744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_and_simplification_l107_10756

theorem calculation_and_simplification :
  (16 ^ (1 / 2 : ℝ) + (1 / 81 : ℝ) ^ (-1 / 4 : ℝ) - (-1 / 2 : ℝ) ^ (0 : ℝ) = 6) ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 
    (2 * a ^ (1 / 4 : ℝ) * b ^ (-1 / 3 : ℝ)) * 
    (-3 * a ^ (-1 / 2 : ℝ) * b ^ (2 / 3 : ℝ)) / 
    (-1 / 4 * a ^ (-1 / 4 : ℝ) * b ^ (-2 / 3 : ℝ)) = 
    24 / (b ^ (1 / 3 : ℝ))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_and_simplification_l107_10756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_sum_is_14_l107_10728

-- Define a cube with 6 faces
def Cube := Fin 6

-- Define a numbering of the cube faces
def Numbering := Cube → Fin 6

-- Define adjacency of faces (this is a simplified version, actual implementation would be more complex)
def Adjacent (f1 f2 : Cube) : Prop := sorry

-- Define a predicate for valid numbering (no adjacent faces have consecutive numbers)
def ValidNumbering (n : Numbering) : Prop :=
  ∀ (f1 f2 : Cube), Adjacent f1 f2 → |n f1 - n f2| ≠ 1

-- Define a function to get the sum of numbers on three faces meeting at a vertex
noncomputable def VertexSum (n : Numbering) (v : Fin 8) : Nat := sorry

-- The main theorem
theorem max_vertex_sum_is_14 (n : Numbering) (h : ValidNumbering n) :
  (∃ v : Fin 8, VertexSum n v = 14) ∧ 
  (∀ v : Fin 8, VertexSum n v ≤ 14) := by
  sorry

#check max_vertex_sum_is_14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_sum_is_14_l107_10728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l107_10705

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ (a d : ℝ), 
    Real.sqrt (49 + k) = a ∧ 
    Real.sqrt (225 + k) = a + d ∧ 
    Real.sqrt (400 + k) = a + 2*d) → 
  k = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l107_10705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_data_stats_l107_10755

noncomputable def average (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := average xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / xs.length

theorem transformed_data_stats (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_avg : average [x₁, x₂, x₃, x₄, x₅] = 2)
  (h_var : variance [x₁, x₂, x₃, x₄, x₅] = 3) :
  let new_data := [2*x₁+1, 2*x₂+1, 2*x₃+1, 2*x₄+1, 2*x₅+1, 1, 2, 3, 4, 5]
  average new_data = 4 ∧ variance new_data = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_data_stats_l107_10755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l107_10712

def word : String := "BALLOON"

def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

def total_letters : Nat := word.length

theorem balloon_arrangements :
  (List.foldl (· * ·) 1 (List.map Nat.factorial (List.range total_letters))) /
  (List.foldl (· * ·) 1 (List.map (Nat.factorial ∘ Prod.snd) letter_frequencies)) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l107_10712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_triangle_area_l107_10717

/-- Given a hyperbola and a parabola, prove the area of the triangle formed by their intersections -/
theorem hyperbola_parabola_intersection_triangle_area 
  (m : ℝ) 
  (hyperbola : ℝ → ℝ → Prop) 
  (parabola : ℝ → ℝ → Prop)
  (e : ℝ) :
  (∀ x y, hyperbola x y ↔ y^2 - x^2/m = 1) →
  (∀ x y, parabola x y ↔ y^2 = m*x) →
  e = Real.sqrt 3 →
  (1 + m) / 1 = 3 →
  let asymptote₁ := λ x : ℝ => (Real.sqrt 2 / 2) * x
  let asymptote₂ := λ x : ℝ => -(Real.sqrt 2 / 2) * x
  let intersection₁ := (4, 2 * Real.sqrt 2)
  let intersection₂ := (4, -2 * Real.sqrt 2)
  let intersection₃ := (0, 0)
  let triangle_area := abs ((intersection₁.1 - intersection₃.1) * (intersection₂.2 - intersection₃.2) -
                            (intersection₂.1 - intersection₃.1) * (intersection₁.2 - intersection₃.2)) / 2
  triangle_area = 8 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_triangle_area_l107_10717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_intersection_point_property_l107_10758

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Check if a point lies on a line defined by two other points -/
def pointOnLine (p q r : Point3D) : Prop :=
  ∃ t : ℝ, r.x = p.x + t * (q.x - p.x) ∧
            r.y = p.y + t * (q.y - p.y) ∧
            r.z = p.z + t * (q.z - p.z)

/-- Find the intersection point of two lines -/
noncomputable def lineIntersection (p₁ q₁ p₂ q₂ : Point3D) : Point3D :=
  sorry

/-- Main theorem -/
theorem cube_intersection_point_property (cube : Cube) 
  (K : Point3D) (P : Point3D) (M : Point3D) (X : Point3D) :
  distance cube.A cube.B = 1 →
  distance cube.B₁ K = distance K cube.C₁ →
  distance cube.C P = 0.25 * distance cube.C cube.D →
  distance cube.C₁ M = 0.25 * distance cube.C₁ cube.D₁ →
  pointOnLine cube.B cube.D₁ X →
  let Y := lineIntersection K X M P
  distance P Y = 1.5 * distance P M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_intersection_point_property_l107_10758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l107_10793

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x - a * x^2
  else -x^2 + (a - 2) * x + 2 * a

-- Define the property that the solution set of f(x) ≥ 0 is [-2, +∞)
def has_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≥ 0 ↔ x ≥ -2

-- Theorem stating the range of a
theorem range_of_a :
  ∀ a : ℝ, has_solution_set a ↔ 0 ≤ a ∧ a ≤ Real.exp 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l107_10793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_triangle_l107_10729

/-- The volume of a solid formed by revolving a right triangle about its hypotenuse -/
noncomputable def volume_of_revolved_triangle (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let r := a * b / c
  (1/3) * Real.pi * r^2 * c

/-- Theorem: The volume of the solid formed by revolving a right triangle 
    with legs of lengths 3 and 4 about its hypotenuse is equal to 48π/5 -/
theorem volume_of_specific_triangle : 
  volume_of_revolved_triangle 3 4 = (48 * Real.pi) / 5 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_triangle_l107_10729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_diameter_l107_10767

-- Define the central angle in radians
noncomputable def central_angle : ℝ := Real.pi / 2

-- Define the radius of the sector
def sector_radius : ℝ := 8

-- Define the arc length of the sector
noncomputable def arc_length : ℝ := central_angle * sector_radius

-- Define the circumference of the base of the cone
noncomputable def base_circumference : ℝ := arc_length

-- Define the diameter of the base of the cone
noncomputable def base_diameter : ℝ := base_circumference / Real.pi

-- Theorem statement
theorem cone_base_diameter :
  base_diameter = 4 := by
  -- Expand definitions
  unfold base_diameter base_circumference arc_length central_angle
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_diameter_l107_10767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l107_10757

/-- The value of n for which the parabola y = x^2 + 9 and the hyperbola y^2 - nx^2 = 1 are tangent -/
noncomputable def tangency_value : ℝ := 18 + 20 * Real.sqrt 2

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 9

/-- Hyperbola equation -/
def hyperbola (n x y : ℝ) : Prop := y^2 - n * x^2 = 1

/-- Theorem stating that the parabola and hyperbola are tangent when n equals the tangency_value -/
theorem parabola_hyperbola_tangent :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangency_value x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola tangency_value x' y' → (x', y') = (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l107_10757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_given_coefficient_l107_10723

/-- The coefficient of x^6 in the expansion of ((x^2 - a)(x + 1/x)^10) -/
def coefficient_x6 (a : ℝ) : ℝ := sorry

/-- The definite integral of (3x^2 + 1) from 0 to a -/
noncomputable def integral_result (a : ℝ) : ℝ := ∫ x in Set.Icc 0 a, (3 * x^2 + 1)

theorem integral_value_given_coefficient :
  ∀ a : ℝ, coefficient_x6 a = 30 → integral_result a = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_given_coefficient_l107_10723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_surjective_function_l107_10788

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, (f a = f b ∧ f (a + b) ≠ min (f a) (f b)) ∨
             (f a ≠ f b ∧ f (a + b) = min (f a) (f b))

theorem unique_surjective_function :
  ∀ f : ℕ → ℕ, is_surjective f → satisfies_conditions f →
  ∀ n : ℕ, f n = Nat.log2 n + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_surjective_function_l107_10788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_and_m_value_l107_10710

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))
noncomputable def b : ℝ × ℝ := (Real.sqrt 3, 1)

noncomputable def f (x m : ℝ) : ℝ := (a x).1 * b.1 + (a x).2 * b.2 + m

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  T > 0 ∧ is_period T f ∧ ∀ T' > 0, is_period T' f → T ≤ T'

theorem smallest_period_and_m_value :
  (∃ m : ℝ, is_smallest_positive_period π (f · m)) ∧
  (∃ m : ℝ, (∀ x ∈ Set.Icc 0 (π / 2), f x m ≥ 5) ∧
            (∃ x ∈ Set.Icc 0 (π / 2), f x m = 5) ∧
            m = 5 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_and_m_value_l107_10710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_C_x_coordinate_range_l107_10734

-- Define the problem setup
def point_A : ℝ × ℝ := (0, 3)
def line_L (x : ℝ) : ℝ := 2 * x - 4
def radius_C : ℝ := 1

-- Define the center of circle C
def center_C (a : ℝ) : ℝ × ℝ := (a, line_L a)

-- Define a point on circle C
noncomputable def point_on_C (a t : ℝ) : ℝ × ℝ := 
  (a + radius_C * Real.cos t, line_L a + radius_C * Real.sin t)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define point D
def point_D : ℝ × ℝ := (0, 1)

-- State the theorem
theorem center_C_x_coordinate_range :
  ∀ a : ℝ, (∃ t : ℝ, distance (point_on_C a t) point_A = 2 * distance (point_on_C a t) point_D) →
  0 ≤ a ∧ a ≤ 12/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_C_x_coordinate_range_l107_10734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l107_10776

/-- An ellipse with one focus at (√2, 0) and chord length 4/3√6 intercepted by x = √2 has equation x²/6 + y²/4 = 1 -/
theorem ellipse_equation (a b : ℝ) : 
  (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) →  -- Ellipse equation
  (Real.sqrt 2)^2 = a^2 - b^2 →  -- Focus at (√2, 0)
  2 * b^2 / a = 4 * Real.sqrt 6 / 3 →  -- Chord length condition
  a^2 = 6 ∧ b^2 = 4 :=
by
  intro h1 h2 h3
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l107_10776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_calculation_l107_10781

/-- Calculates the height of a tree given the amount of wood for rungs, rung length, and space between rungs -/
noncomputable def treeHeight (woodLength : ℝ) (rungLength : ℝ) (rungSpace : ℝ) : ℝ :=
  let totalInches : ℝ := woodLength * 12
  let numRungs : ℝ := totalInches / rungLength
  let heightInInches : ℝ := (numRungs - 1) * (rungLength + rungSpace) + rungLength
  heightInInches / 12

/-- The theorem stating the height of the tree -/
theorem tree_height_calculation :
  treeHeight 150 (18/12) (6/12) = 199.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval treeHeight 150 (18/12) (6/12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_calculation_l107_10781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portland_probability_l107_10795

def word_dock : Finset Char := {'D', 'O', 'C', 'K'}
def word_plants : Finset Char := {'P', 'L', 'A', 'N', 'T', 'S'}
def word_hero : Finset Char := {'H', 'E', 'R', 'O'}
def word_portland : Finset Char := {'P', 'O', 'R', 'T', 'L', 'A', 'N', 'D'}

def select_dock := 2
def select_plants := 4
def select_hero := 3

theorem portland_probability :
  (Finset.card (word_dock.powerset.filter (λ s => s.card = select_dock ∧ s ⊆ word_portland)) /
   Finset.card (word_dock.powerset.filter (λ s => s.card = select_dock))) *
  (Finset.card (word_plants.powerset.filter (λ s => s.card = select_plants ∧ s ⊆ word_portland)) /
   Finset.card (word_plants.powerset.filter (λ s => s.card = select_plants))) *
  (Finset.card (word_hero.powerset.filter (λ s => s.card = select_hero ∧ s ⊆ word_portland)) /
   Finset.card (word_hero.powerset.filter (λ s => s.card = select_hero))) = 1 / 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_portland_probability_l107_10795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_splendid_one_l107_10753

-- Define a splendid number
def is_splendid (x : ℝ) : Prop :=
  x > 0 ∧ ∀ n : ℕ, (Int.floor (x * 10^n) % 10 = 0 ∨ Int.floor (x * 10^n) % 10 = 9)

-- Theorem statement
theorem exists_splendid_one : ∃ x : ℝ, is_splendid x ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_splendid_one_l107_10753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_l107_10782

/-- The parametric equations of the curve --/
noncomputable def x (t : Real) : Real := Real.sqrt 2 * Real.cos t
noncomputable def y (t : Real) : Real := 2 * Real.sqrt 2 * Real.sin t

/-- The upper bound of y --/
def upper_bound : Real := 2

/-- The area of the figure --/
noncomputable def area : Real := Real.pi - 2

theorem figure_area :
  (∀ t, x t = Real.sqrt 2 * Real.cos t) ∧
  (∀ t, y t = 2 * Real.sqrt 2 * Real.sin t) ∧
  (∀ t, y t ≥ upper_bound → y t = upper_bound) →
  area = Real.pi - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_l107_10782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l107_10741

-- Define the line equation
noncomputable def line (x : ℝ) (b : ℝ) : ℝ := Real.sqrt 3 * x + b

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the intersection condition
def intersects (b : ℝ) : Prop := ∃ x y : ℝ, line x b = y ∧ circle_eq x y

-- Theorem statement
theorem intersection_condition :
  (∀ b : ℝ, |b| < 2 → intersects b) ∧
  (∃ b : ℝ, ¬(|b| < 2) ∧ intersects b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l107_10741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l107_10709

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/4)

theorem g_properties :
  (∀ x, g x = -Real.cos (2 * x)) ∧
  (∃ x, g x = 1) ∧
  (∀ x, g (Real.pi/2 + x) = g (Real.pi/2 - x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l107_10709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_inequality_holds_l107_10747

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log (x + 1)

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m b : ℝ), m = 1/2 + Real.log 2 ∧ b = -1/2 ∧
  ∀ (x : ℝ), (m * x + b) = (deriv f) 1 * (x - 1) + f 1 := by sorry

-- Theorem for the inequality
theorem inequality_holds (x : ℝ) (h : x > -1) :
  f x + 1/2 * x^3 ≥ x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_inequality_holds_l107_10747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_ln_l107_10778

/-- Given that f(x) = ln(x-a) is increasing on (1, +∞), the range of a is (-∞, 1] -/
theorem range_of_a_for_increasing_ln (a : ℝ) : 
  (∀ x y : ℝ, 1 < x ∧ x < y → Real.log (x - a) < Real.log (y - a)) → 
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_ln_l107_10778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_base_ratio_l107_10735

/-- Represents a trapezoid with bases a and b, and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  ha : a > 0
  hb : b > 0
  hh : h > 0
  hab : a > b

/-- The area of a trapezoid -/
noncomputable def Trapezoid.area (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

/-- The area of the quadrilateral formed by midpoints -/
noncomputable def Trapezoid.midpointQuadArea (t : Trapezoid) : ℝ := (t.a - t.b) * t.h / 4

theorem trapezoid_base_ratio (t : Trapezoid) :
  t.midpointQuadArea = t.area / 4 → t.a / t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_base_ratio_l107_10735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l107_10738

-- Define the equation
def equation (θ : Real) : Prop :=
  Real.tan (4 * Real.pi * Real.sin θ) = 1 / Real.tan (4 * Real.pi * Real.cos θ)

-- Define the solution set
noncomputable def solution_set : Set Real :=
  {θ | θ ∈ Set.Ioo 0 (2 * Real.pi) ∧ equation θ}

-- State the theorem
theorem equation_solutions :
  ∃ (s : Finset Real), s.card = 20 ∧ ∀ θ, θ ∈ s ↔ θ ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l107_10738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_circles_externally_tangent_implies_no_intersection_l107_10749

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 16

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, 0)
def center2 : ℝ × ℝ := (0, -4)
def radius1 : ℝ := 1
def radius2 : ℝ := 4

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem stating that the circles are externally tangent
theorem circles_externally_tangent : distance_between_centers = radius1 + radius2 := by
  -- We use 'sorry' to skip the proof for now
  sorry

-- Additional theorem to demonstrate the relationship
theorem circles_externally_tangent_implies_no_intersection :
  distance_between_centers = radius1 + radius2 →
  ∀ x y : ℝ, ¬(circle1 x y ∧ circle2 x y) := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_circles_externally_tangent_implies_no_intersection_l107_10749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_logarithm_simplification_l107_10797

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Part 1
theorem expression_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  4 * x^(1/4 : ℝ) * (-3 * x^(1/4 : ℝ) * y^(-(1/3) : ℝ)) / (-6 * x^(-(1/2) : ℝ) * y^(-(2/3) : ℝ)) = 2 * y^((1/3) : ℝ) := by
  sorry

-- Part 2
theorem logarithm_simplification :
  (1/2) * lg (32/49) - (4/3) * lg (Real.sqrt 8) + lg (Real.sqrt 245) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_logarithm_simplification_l107_10797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_container_difference_l107_10799

theorem marble_container_difference : 
  ∀ (total_marbles_A total_marbles_B red_A yellow_A red_B yellow_B : ℤ),
  -- Conditions
  total_marbles_A = total_marbles_B →
  total_marbles_A = red_A + yellow_A →
  total_marbles_B = red_B + yellow_B →
  7 * yellow_A = 3 * red_A →
  4 * yellow_B = red_B →
  yellow_A + yellow_B = 120 →
  -- Conclusion
  red_A - red_B = -24 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_container_difference_l107_10799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l107_10745

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2 - 1
  else if -1 ≤ x ∧ x < 0 then x^2
  else 0  -- undefined for other x values

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then Real.sqrt (x + 1)
  else if 0 < x ∧ x ≤ 1 then -Real.sqrt x
  else 0  -- undefined for other x values

-- Theorem statement
theorem f_inverse_correct :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f (f_inv x) = x ∧ f_inv (f x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l107_10745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdwatching_problem_l107_10790

/-- Proves that given the conditions from the birdwatching problem, 
    the average number of birds at each site on Wednesday was 8. -/
theorem birdwatching_problem 
  (sites_monday : ℕ) 
  (sites_tuesday : ℕ) 
  (sites_wednesday : ℕ)
  (avg_birds_monday : ℚ)
  (avg_birds_tuesday : ℚ)
  (avg_birds_total : ℚ)
  (h1 : sites_monday = 5)
  (h2 : sites_tuesday = 5)
  (h3 : sites_wednesday = 10)
  (h4 : avg_birds_monday = 7)
  (h5 : avg_birds_tuesday = 5)
  (h6 : avg_birds_total = 7) :
  (avg_birds_total * (sites_monday + sites_tuesday + sites_wednesday : ℚ) - 
   avg_birds_monday * (sites_monday : ℚ) - 
   avg_birds_tuesday * (sites_tuesday : ℚ)) / (sites_wednesday : ℚ) = 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdwatching_problem_l107_10790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_sin_value_l107_10726

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + Real.sin (2 * x + Real.pi / 3)

theorem f_range_and_sin_value :
  (∃ (a b : ℝ), a = -Real.sqrt 6 / 2 ∧ b = Real.sqrt 2 ∧
    ∀ x ∈ Set.Icc 0 (5 * Real.pi / 8), a ≤ f x ∧ f x ≤ b) ∧
  (∀ α : ℝ, α ∈ Set.Ioo (Real.pi / 6) Real.pi →
    f (α / 2) = 1 / 2 →
      Real.sin (2 * α + Real.pi / 6) = -Real.sqrt 7 / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_sin_value_l107_10726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_smaller_spheres_l107_10783

/-- The radius of the larger sphere -/
def R : ℝ := sorry

/-- The radius of the smaller spheres -/
def r : ℝ := sorry

/-- The number of smaller spheres -/
def n : ℕ := 4

/-- Theorem stating the maximum radius of smaller spheres inside a larger sphere -/
theorem max_radius_smaller_spheres :
  r ≤ (Real.sqrt 6 / (3 + Real.sqrt 6)) * R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_smaller_spheres_l107_10783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_value_l107_10777

/-- Represents a 2n × 2n grid with entries of 1 or -1 -/
def Grid (n : ℕ) := Fin (2*n) → Fin (2*n) → Int

/-- Predicate to check if a grid is valid according to the problem conditions -/
def is_valid_grid (n : ℕ) (g : Grid n) : Prop :=
  (∀ i j, g i j = 1 ∨ g i j = -1) ∧
  (Finset.sum (Finset.univ : Finset (Fin (2*n) × Fin (2*n))) (fun ij ↦ if g ij.1 ij.2 = 1 then 1 else 0) = 2*n^2)

/-- Sum of a row in the grid -/
def row_sum (n : ℕ) (g : Grid n) (i : Fin (2*n)) : Int :=
  Finset.sum Finset.univ (fun j ↦ g i j)

/-- Sum of a column in the grid -/
def col_sum (n : ℕ) (g : Grid n) (j : Fin (2*n)) : Int :=
  Finset.sum Finset.univ (fun i ↦ g i j)

/-- Definition of M for a given grid -/
def M (n : ℕ) (g : Grid n) : ℕ :=
  (Finset.sum Finset.univ (fun i ↦ Int.natAbs (row_sum n g i))) +
  (Finset.sum Finset.univ (fun j ↦ Int.natAbs (col_sum n g j)))

/-- The main theorem stating the maximum possible value of M -/
theorem max_M_value (n : ℕ) :
  ∃ (g : Grid n), is_valid_grid n g ∧ 
    ∀ (h : Grid n), is_valid_grid n h → M n h ≤ M n g ∧ M n g = 4*n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_value_l107_10777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_A_location_l107_10707

/-- The angle (in degrees) that the Sun's rays make with the plane of the Equator -/
def sun_angle : ℝ := 15

/-- The ratio of the shadow length in city A to the shadow length in city B -/
def shadow_ratio : ℝ := 3

/-- The latitude of city A (in degrees, negative for southern hemisphere) -/
def city_A_latitude : ℝ → Prop := λ φ => True

/-- The tangent of 15 degrees -/
noncomputable def tan_15 : ℝ := 2 - Real.sqrt 3

/-- The tangent function for angles in degrees -/
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

/-- Approximately equal -/
def approx_eq (x y : ℝ) : Prop := abs (x - y) < 0.01

theorem city_A_location :
  ∀ φ : ℝ,
  city_A_latitude φ →
  (φ = -45 ∨ approx_eq φ (-7.23333)) ∧
  tan_deg (φ + sun_angle) = shadow_ratio * tan_deg (abs φ - sun_angle) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_A_location_l107_10707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_sum_special_form_l107_10780

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  radius : ℝ
  sidesCount : ℕ
  sidesCount_eq : sidesCount = n
  sidesCount_ge_3 : sidesCount ≥ 3

/-- The sum of lengths of sides and diagonals of a regular polygon --/
noncomputable def sumOfLengths (p : RegularPolygon n) : ℝ :=
  sorry

/-- Representation of a real number in the form a + b√2 + c√3 + d√5 --/
structure SpecialForm where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  value : ℝ
  value_eq : value = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5

theorem decagon_sum_special_form :
  ∃ (sf : SpecialForm),
    sumOfLengths ⟨10, 10, rfl, by norm_num⟩ = sf.value ∧
    sf.a + sf.b + sf.c + sf.d = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_sum_special_form_l107_10780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_rung_spacing_l107_10740

/-- Calculates the space between rungs on a ladder. -/
noncomputable def spaceBetweenRungs (totalHeight : ℝ) (numRungs : ℕ) (rungLength : ℝ) : ℝ :=
  totalHeight / (numRungs - 1 : ℝ)

/-- Theorem stating the space between rungs for the given ladder specifications. -/
theorem ladder_rung_spacing :
  let totalHeight : ℝ := 600
  let numRungs : ℕ := 100
  let rungLength : ℝ := 18
  abs (spaceBetweenRungs totalHeight numRungs rungLength - 6.06) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_rung_spacing_l107_10740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l107_10742

open Real

-- Define the function y(x)
noncomputable def y (x : ℝ) (c : ℝ) : ℝ := -1 / (3 * x + c)

-- State the theorem
theorem y_satisfies_differential_equation (x : ℝ) (c : ℝ) :
  deriv (fun x => y x c) x = 3 * (y x c)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l107_10742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l107_10731

/-- The slope of the asymptotes of the hyperbola x²/16 - y²/25 = 1 -/
noncomputable def asymptote_slope : ℝ := 5/4

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2/16 - y^2/25 = 1

/-- The equation of the asymptotes -/
def is_on_asymptote (x y : ℝ) : Prop := y = asymptote_slope * x ∨ y = -asymptote_slope * x

/-- Theorem stating that points on the hyperbola approach the asymptotes as x approaches infinity -/
theorem hyperbola_asymptote_slope :
  ∀ ε > 0, ∃ x₀ > 0, ∀ x y, x ≥ x₀ → is_on_hyperbola x y →
    ∃ y', is_on_asymptote x y' ∧ |y - y'| < ε * |x| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l107_10731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l107_10785

/-- A curve defined by y = sin(2x + φ) -/
noncomputable def curve (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

/-- The curve passes through the origin -/
def passes_through_origin (φ : ℝ) : Prop :=
  ∃ x : ℝ, curve φ x = 0 ∧ x = 0

theorem sufficient_not_necessary :
  (∀ φ : ℝ, φ = π → passes_through_origin φ) ∧
  (∃ φ : ℝ, φ ≠ π ∧ passes_through_origin φ) := by
  sorry

#check sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l107_10785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l107_10759

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + 2*c = 2) :
  a + Real.sqrt (a*b) + (a^2*b*c)^(1/3) ≤ 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + 2*c' = 2 ∧ 
    a' + Real.sqrt (a'*b') + (a'^2*b'*c')^(1/3) = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l107_10759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_3967149_8654321_l107_10706

noncomputable def roundToNearestInteger (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem nearest_integer_to_3967149_8654321 :
  roundToNearestInteger 3967149.8654321 = 3967150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_3967149_8654321_l107_10706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_half_time_l107_10725

/-- Represents a burning candle with initial height and burn time -/
structure Candle where
  initialHeight : ℝ
  burnTime : ℝ

/-- Calculates the burn rate of a candle -/
noncomputable def burnRate (c : Candle) : ℝ := c.initialHeight / c.burnTime

/-- Calculates the height of a candle after a given time -/
noncomputable def heightAfterTime (c : Candle) (t : ℝ) : ℝ := c.initialHeight - (burnRate c) * t

/-- The time when the first candle's height is half the second candle's height -/
noncomputable def equalHeightTime (c1 c2 : Candle) : ℝ :=
  (c1.initialHeight - 0.5 * c2.initialHeight) / (0.5 * burnRate c2 - burnRate c1)

theorem candle_height_half_time :
  let c1 : Candle := { initialHeight := 12, burnTime := 6 }
  let c2 : Candle := { initialHeight := 15, burnTime := 5 }
  equalHeightTime c1 c2 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_half_time_l107_10725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_of_two_clubs_l107_10708

theorem average_score_of_two_clubs 
  (members_A members_B : ℕ) 
  (avg_A avg_B : ℝ) 
  (members_A_pos : members_A > 0)
  (members_B_pos : members_B > 0) :
  let total_members := members_A + members_B
  let total_score := (members_A : ℝ) * avg_A + (members_B : ℝ) * avg_B
  total_score / (total_members : ℝ) = 
    ((members_A : ℝ) * avg_A + (members_B : ℝ) * avg_B) / ((members_A + members_B) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_of_two_clubs_l107_10708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_properties_l107_10752

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 
    if -4 ≤ x then x + 2 else 0
  else if x ≤ 6 then 
    Real.sqrt (9 - (x - 3)^2) - 3
  else if x ≤ 7 then 
    -x + 6
  else 0  -- Define a default value for x outside the given range

-- Theorem statement
theorem abs_g_properties :
  (∀ x : ℝ, -4 ≤ x ∧ x < 0 ∧ x ≠ -2 → |g x| = g x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 6 → |g x| = -(g x)) ∧
  (∀ x : ℝ, 6 < x ∧ x ≤ 7 → |g x| = -(g x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_properties_l107_10752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2500th_term_l107_10720

/-- Represents the sequence where the nth positive integer appears n times -/
def sequenceElem (n : ℕ) : ℕ := n

/-- The 2500th term of the sequence -/
def term_2500 : ℕ := 71

theorem sequence_2500th_term :
  (∀ n : ℕ, ∀ k : ℕ, k > (n * (n - 1)) / 2 ∧ k ≤ (n * (n + 1)) / 2 → sequenceElem k = n) →
  term_2500 = 71 ∧ term_2500 % 5 = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2500th_term_l107_10720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l107_10751

/-- Calculates the speed of a train given its length and time to pass a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

theorem train_speed_calculation (length time : ℝ) 
  (h1 : length = 120) 
  (h2 : time = 6) : 
  train_speed length time = 20 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l107_10751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_properties_l107_10762

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem cosine_properties (ω φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2) 
  (h3 : ∀ x : ℝ, f ω φ (x + π/(2*ω)) = f ω φ x) 
  (h4 : f ω φ (π/6) = 1) :
  (∀ k : ℤ, StrictMonoOn (f ω φ) (Set.Icc (k*π + π/6) (k*π + 2*π/3))) ∧ 
  (Set.range (f ω φ) ∩ Set.Icc (-1/2) 1 = Set.Icc (-1/2) 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_properties_l107_10762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l107_10737

theorem cos_alpha_value (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.pi / 2 < β) (h4 : β < Real.pi)
  (h5 : Real.cos β = -3/5) (h6 : Real.sin (α + β) = 5/13) : Real.cos α = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l107_10737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_to_raise_average_l107_10704

def current_scores : List ℕ := [92, 82, 75, 65, 88]
def target_increase : ℚ := 4

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

noncomputable def min_next_score (scores : List ℕ) (target_increase : ℚ) : ℕ :=
  Int.toNat ⌈((average scores + target_increase) * (scores.length + 1 : ℚ) - scores.sum)⌉

theorem min_score_to_raise_average :
  min_next_score current_scores target_increase = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_to_raise_average_l107_10704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l107_10733

/-- The circle C with equation x^2 + (y-4)^2 = 4 -/
def circleC (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

/-- The hyperbola E with equation x^2/a^2 - y^2/b^2 = 1 -/
def hyperbolaE (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptotes (x y a b : ℝ) : Prop := y = b/a * x ∨ y = -b/a * x

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The theorem stating that the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, asymptotes x y a b → ∃ t, circleC (a * t) (b * t + 4)) →
  eccentricity a b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l107_10733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dozen_pens_l107_10746

/-- The cost of one dozen pens given specific conditions -/
theorem cost_of_dozen_pens (A : ℝ) : ℝ := by
  -- Define variables
  let cost_3_pens_5_pencils : ℝ := A
  let pen_pencil_ratio : ℝ := 5
  let total_cost_dozen_pens : ℝ := 300

  -- Theorem: The cost of one dozen pens is Rs. 300
  have cost_dozen_pens_is_300 : total_cost_dozen_pens = 300 := by rfl

  -- Return the cost of one dozen pens
  exact total_cost_dozen_pens

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dozen_pens_l107_10746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l107_10719

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2 * (x - 3)^2
def curve2 (x : ℝ) : ℝ := (x^2 - 1) * (x - 2)

-- Theorem statement
theorem intersection_sum : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, curve1 x = curve2 x) ∧ (S.sum id = 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l107_10719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_mean_score_l107_10721

/-- Represents the distribution of scores in a quiz -/
structure ScoreDistribution where
  zero_correct : ℚ
  one_correct : ℚ
  two_correct : ℚ
  three_correct : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (dist : ScoreDistribution) : ℚ :=
  (0 * dist.zero_correct + 1 * dist.one_correct + 2 * dist.two_correct + 3 * dist.three_correct) / 100

/-- The quiz score distribution -/
def quiz_distribution : ScoreDistribution :=
  { zero_correct := 20
  , one_correct := 5
  , two_correct := 40
  , three_correct := 35 }

theorem quiz_mean_score :
  mean_score quiz_distribution = 19/10 := by
  -- Unfold definitions and simplify
  unfold mean_score quiz_distribution
  -- Perform arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_mean_score_l107_10721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_sum_difference_l107_10713

def first_n_even_sum (n : ℕ) : ℕ := n * (n - 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference :
  (first_n_odd_sum 1000 : ℤ) - (first_n_even_sum 1000 : ℤ) = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_sum_difference_l107_10713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_ray_l107_10768

-- Define the curves in polar coordinates
noncomputable def C₁ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 4

noncomputable def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the intersection points
noncomputable def M (α : ℝ) : ℝ := 4 / (Real.sin α + Real.cos α)

noncomputable def N (α : ℝ) : ℝ := 2 * Real.sin α

-- Define the ratio function
noncomputable def ratio (α : ℝ) : ℝ := N α / M α

-- State the theorem
theorem max_ratio_on_ray :
  ∀ α, 0 < α → α < Real.pi / 2 →
  ratio α ≤ (Real.sqrt 2 + 1) / 4 ∧
  ∃ α₀, 0 < α₀ ∧ α₀ < Real.pi / 2 ∧ ratio α₀ = (Real.sqrt 2 + 1) / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_ray_l107_10768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_nabla_2016_solution_l107_10711

noncomputable def nabla (x y : ℝ) : ℝ := x - 1 / y

noncomputable def nested_nabla : ℕ → ℝ
  | 0 => 2
  | n + 1 => nabla 2 (nested_nabla n)

theorem nested_nabla_2016 :
  nested_nabla 2016 = 2017 / 2016 := by
  sorry

theorem solution :
  100 * 2017 + 2016 = 203716 := by
  norm_num

#eval 100 * 2017 + 2016

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_nabla_2016_solution_l107_10711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tortoise_speed_proof_l107_10750

/-- The distance from point A to the amusement park in meters -/
def distance : ℚ := 2640

/-- The speed of the Rabbit in meters per minute -/
def rabbit_speed : ℚ := 36

/-- The interval between Rabbit's breaks in minutes -/
def break_interval : ℚ := 3

/-- The duration of the k-th break for the Rabbit in minutes -/
def break_duration (k : ℕ) : ℚ := 1/2 * k

/-- The time difference between Tortoise and Rabbit arrival in minutes -/
def time_difference : ℚ := 10/3

/-- The speed of the Tortoise in meters per minute -/
def tortoise_speed : ℚ := 12

theorem tortoise_speed_proof :
  let rabbit_travel_time := distance / rabbit_speed
  let num_breaks := ⌊(rabbit_travel_time / break_interval : ℚ)⌋
  let total_break_time := (↑num_breaks * (↑num_breaks + 1) / 4 : ℚ)
  let rabbit_total_time := rabbit_travel_time + total_break_time
  let tortoise_time := rabbit_total_time - time_difference
  tortoise_speed = distance / tortoise_time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tortoise_speed_proof_l107_10750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l107_10732

/-- Represents the time in minutes to fill a pool with two valves -/
noncomputable def fill_time (pool_capacity : ℝ) (first_valve_time : ℝ) (valve_difference : ℝ) : ℝ :=
  let first_valve_rate := pool_capacity / (first_valve_time * 60)
  let second_valve_rate := first_valve_rate + valve_difference
  let combined_rate := first_valve_rate + second_valve_rate
  pool_capacity / combined_rate

/-- Theorem stating that under given conditions, the pool will be filled in 48 minutes -/
theorem pool_fill_time :
  fill_time 12000 2 50 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l107_10732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l107_10794

noncomputable def given_numbers : List ℝ := [10, -2.5, 0.8, 0, -Real.pi, 11, -9, -4.2, -2]

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

def is_negative (x : ℝ) : Prop := x < 0

def integer_set : Set ℝ := {x ∈ given_numbers | is_integer x}
def negative_set : Set ℝ := {x ∈ given_numbers | is_negative x}

theorem number_classification :
  integer_set = {10, 0, 11, -9, -2} ∧
  negative_set = {-2.5, -Real.pi, -9, -4.2, -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l107_10794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l107_10743

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance 2c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_sq : c^2 = a^2 - b^2

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- The theorem statement -/
theorem ellipse_eccentricity_theorem (E : Ellipse) (M : PointOnEllipse E)
  (h_line : M.y = Real.sqrt 3 * (M.x + E.c))
  (h_angle : ∃ θ₁ θ₂ : ℝ, θ₁ = 2 * θ₂ ∧ 
    Real.tan θ₁ = Real.sqrt 3 ∧
    Real.tan θ₂ = (M.y + E.c) / (M.x + E.c)) :
  eccentricity E = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l107_10743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l107_10715

-- Define the parametric equations
noncomputable def x (θ : Real) : Real := 1 + Real.cos θ
noncomputable def y (θ : Real) : Real := Real.sqrt 3 + Real.sin θ

-- Define the distance function from a point on the curve to the origin
noncomputable def distance_to_origin (θ : Real) : Real :=
  Real.sqrt ((x θ)^2 + (y θ)^2)

-- Theorem statement
theorem min_distance_to_origin :
  ∃ (d : Real), d = 1 ∧ ∀ (θ : Real), distance_to_origin θ ≥ d := by
  sorry

#check min_distance_to_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l107_10715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_minus_l107_10739

theorem tan_double_minus (θ φ : ℝ) (h1 : Real.tan θ = 3) (h2 : Real.tan φ = 2) :
  Real.tan (2 * θ - φ) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_minus_l107_10739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_and_bounds_l107_10771

open Real

theorem unique_zero_and_bounds (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 2) :
  ∃! x₀ : ℝ, x₀ > 0 ∧ (exp x₀ - x₀ - a = 0) ∧
  Real.sqrt (a - 1) ≤ x₀ ∧ x₀ ≤ Real.sqrt (2 * (a - 1)) ∧
  x₀ * (exp (exp x₀) - exp x₀ - a) ≥ (exp 1 - 1) * (a - 1) * a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_and_bounds_l107_10771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_zero_implies_sum_l107_10724

def matrix (a b c k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, k*b, k*c],
    ![k*b, c, k*a],
    ![k*c, k*a, b]]

theorem matrix_determinant_zero_implies_sum (a b c k : ℝ) :
  Matrix.det (matrix a b c k) = 0 →
  (a / (k*b + k*c) + b / (k*a + c) + c / (k*a + b) = -1) ∨
  (a / (k*b + k*c) + b / (k*a + c) + c / (k*a + b) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_zero_implies_sum_l107_10724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_profit_cost_properties_l107_10784

-- Define the cost function
noncomputable def cost (x : ℝ) : ℝ := (1/10) * (x - 15)^2 + 17.5

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := 1.6 * x - cost x

-- State the theorem
theorem optimal_production_and_profit :
  ∃ (x : ℝ), 
    10 ≤ x ∧ x ≤ 25 ∧
    (∀ y : ℝ, 10 ≤ y ∧ y ≤ 25 → profit y ≤ profit x) ∧
    x = 23 ∧ profit x = 12.9 := by
  sorry

-- Additional properties to ensure equivalence with the original problem
theorem cost_properties :
  cost 10 = 20 ∧ cost 15 = 17.5 ∧
  (∀ x : ℝ, 10 ≤ x ∧ x ≤ 25 → cost x ≥ 17.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_profit_cost_properties_l107_10784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l107_10760

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  h0 : a > b
  h1 : b > 0

/-- Represents a line passing through a vertex and a focus of the ellipse -/
structure EllipseLine (e : Ellipse) where
  c : ℝ  -- x-intercept of the line
  h0 : c^2 = e.a^2 - e.b^2  -- condition for passing through a focus

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center of the ellipse to the line -/
noncomputable def distance_to_line (e : Ellipse) (l : EllipseLine e) : ℝ :=
  (e.b * l.c) / Real.sqrt (l.c^2 + e.b^2)

theorem ellipse_eccentricity_theorem (e : Ellipse) (l : EllipseLine e) 
  (h : distance_to_line e l = e.b / 4) : 
  eccentricity e = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l107_10760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_third_term_is_three_l107_10792

def T : Finset (Fin 6 → Fin 6) :=
  (Finset.univ : Finset (Fin 6 → Fin 6)).filter (fun p => Function.Injective p ∧ p 0 ≠ 0)

theorem probability_third_term_is_three :
  (T.filter (fun p => p 2 = 2)).card / T.card = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_third_term_is_three_l107_10792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_correct_factorial_sequence_correct_l107_10754

/-- Geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometricSequence a q n * q

/-- Factorial sequence -/
def factorialSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => factorialSequence n * (n + 1)

theorem geometric_sequence_correct (a q : ℝ) (n : ℕ) :
  geometricSequence a q n = a * q^n := by
  sorry

theorem factorial_sequence_correct (n : ℕ) :
  factorialSequence n = Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_correct_factorial_sequence_correct_l107_10754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_base_angle_theorem_l107_10791

/-- Regular triangular pyramid with lateral edge twice the base side length -/
structure RegularTriangularPyramid where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = 2 * base_side

/-- Angle between slant height and base height in a regular triangular pyramid -/
noncomputable def slant_base_angle (p : RegularTriangularPyramid) : ℝ :=
  Real.arccos (Real.sqrt 5 / 30)

/-- Theorem stating the angle between slant height and base height -/
theorem slant_base_angle_theorem (p : RegularTriangularPyramid) :
  slant_base_angle p = Real.arccos (Real.sqrt 5 / 30) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_base_angle_theorem_l107_10791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_is_parallelogram_l107_10769

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the properties
def is_rectangle (rect : Rectangle) : Prop := sorry

def diagonals_intersect (rect : Rectangle) (E : Point) : Prop := sorry

def is_circumcenter (center : Point) (A B C : Point) : Prop := sorry

def is_parallelogram (P Q R S : Point) : Prop := sorry

-- Main theorem
theorem PQRS_is_parallelogram 
  (A B C D E P Q R S : Point) 
  (rect : Rectangle)
  (h_rect : is_rectangle rect)
  (h_diag : diagonals_intersect rect E)
  (h_P : is_circumcenter P A B E)
  (h_Q : is_circumcenter Q B C E)
  (h_R : is_circumcenter R C D E)
  (h_S : is_circumcenter S A D E) :
  is_parallelogram P Q R S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_is_parallelogram_l107_10769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_exists_l107_10716

theorem unique_divisor_exists : ∃! D : ℕ, 
  D > 0 ∧
  242 % D = 6 ∧
  698 % D = 13 ∧
  940 % D = 5 ∧
  D = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_exists_l107_10716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l107_10722

-- Define the angles
variable (α β : Real)

-- Define the symmetry condition
axiom symmetry : ∃ (k : Real), Real.tan β = -Real.tan α

-- Define the condition for α
axiom α_condition : Real.tan α = 4/3

-- Theorem to prove
theorem tan_difference : Real.tan (α - β) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l107_10722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_equals_sqrt_a_plus_one_l107_10727

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ a : ℝ) : 
  0 ≤ θ → θ ≤ π / 2 → Real.sin (2 * θ) = a → Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_equals_sqrt_a_plus_one_l107_10727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l107_10789

/-- A function f(x) = 3^x + b where b is a real number. -/
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x ↦ Real.exp (x * Real.log 3) + b

/-- The function g(b) = f(b) - f(b-1) where f is defined as above. -/
noncomputable def g (b : ℝ) : ℝ := f b b - f b (b - 1)

/-- The set of quadrants that the graph of f passes through. -/
def passes_through_quadrants (f : ℝ → ℝ) : Prop :=
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Fourth quadrant
  (∃ x y, x < 0 ∧ y < 0 ∧ f x = y)    -- Third quadrant

/-- The main theorem stating the range of g(b) given the conditions on f. -/
theorem range_of_g :
  ∀ b : ℝ, passes_through_quadrants (f b) →
  Set.range g = Set.Ioo 0 (2/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l107_10789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_membership_is_better_l107_10772

/-- Calculates the store's tiered discount --/
noncomputable def storeDiscount (amount : ℝ) : ℝ :=
  if amount ≤ 1000 then 0
  else if amount ≤ 2000 then 0.1 * (amount - 1000)
  else if amount ≤ 3000 then 0.1 * 1000 + 0.15 * (amount - 2000)
  else 0.1 * 1000 + 0.15 * 1000 + 0.2 * (amount - 3000)

/-- Calculates the total cost with gold membership discount --/
noncomputable def totalCostWithMembership (initialAmount : ℝ) : ℝ :=
  let afterStoreDiscount := initialAmount - storeDiscount initialAmount
  let afterMembershipDiscount := afterStoreDiscount * 0.95
  afterMembershipDiscount * 1.08

/-- Calculates the total cost with special discount code --/
noncomputable def totalCostWithCode (initialAmount : ℝ) : ℝ :=
  let afterStoreDiscount := initialAmount - storeDiscount initialAmount
  let afterSpecialDiscount := afterStoreDiscount - 200
  afterSpecialDiscount * 1.08

/-- The initial amount of John's purchase --/
def johnsPurchase : ℝ := 3 * 250 + 4 * 350 + 5 * 500

theorem gold_membership_is_better :
  totalCostWithMembership johnsPurchase < totalCostWithCode johnsPurchase :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_membership_is_better_l107_10772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l107_10748

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 4) * Real.cos (x + Real.pi / 4) + 1 / 2

theorem f_is_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l107_10748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l107_10798

/-- The distance covered by a wheel with a given diameter and number of revolutions -/
noncomputable def distance_covered (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  Real.pi * diameter * revolutions

/-- Theorem stating that a wheel with diameter 14 cm making 19.017288444040037 revolutions
    covers a distance of approximately 836.103 cm -/
theorem wheel_distance_theorem :
  let d := (14 : ℝ)
  let r := 19.017288444040037
  abs (distance_covered d r - 836.103) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l107_10798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l107_10766

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)
noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem sequence_problem (q d : ℝ) : 
  (arithmetic_sequence 1 d 8 = geometric_sequence 1 q 4) ∧ 
  (geometric_sum 1 q 4 = arithmetic_sequence 1 d 8 + 21) →
  ((q = 4 ∧ d = 9) ∨ (q = -5 ∧ d = -18)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l107_10766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l107_10770

-- Define the propositions
def P₁ : Prop := ∀ n : ℕ, n^2 > 2^n → ∃ n₀ : ℕ, n₀^2 ≤ 2^n₀

def P₂ : Prop := ∀ m n : ℝ, (m * 1 + 1 * (-n) = 0) ↔ m = n

def P₃ : Prop := ∀ A B : ℝ, (0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi) → 
  (Real.sin A ≤ Real.sin B → A ≤ B)

def P₄ : Prop := ∀ p q : Prop, ¬(p ∧ q) → ¬p

theorem proposition_truth : ¬P₁ ∧ P₂ ∧ P₃ ∧ ¬P₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l107_10770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_central_symmetry_f_axial_symmetry_f_odd_f_periodic_l107_10779

-- Define the function f(x) = cos(x) * sin(2x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (2 * x)

-- State the theorems to be proved
theorem f_central_symmetry : ∀ x, f (2 * Real.pi - x) + f x = 0 := by sorry

theorem f_axial_symmetry : ∀ x, f (Real.pi - x) = f x := by sorry

theorem f_odd : ∀ x, f (-x) = -f x := by sorry

theorem f_periodic : ∀ x, f (x + 2 * Real.pi) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_central_symmetry_f_axial_symmetry_f_odd_f_periodic_l107_10779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_vertical_asymptote_l107_10702

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 9*x + 20)

def has_one_vertical_asymptote (c : ℝ) : Prop :=
  (∃! x : ℝ, x^2 - 9*x + 20 = 0 ∧ x^2 - x + c ≠ 0)

theorem f_one_vertical_asymptote :
  ∀ c : ℝ, has_one_vertical_asymptote c ↔ (c = -12 ∨ c = -20) := by
  sorry

#check f_one_vertical_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_vertical_asymptote_l107_10702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resulting_solid_surface_area_l107_10730

/-- Represents the dimensions of a solid --/
structure Dimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the surface area of a rectangular solid --/
def surfaceArea (d : Dimensions) : Real :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Theorem: The surface area of the resulting solid is 26 square feet --/
theorem resulting_solid_surface_area :
  let cubeVolume : Real := 8
  let cubeSide : Real := cubeVolume ^ (1/3)
  let cuts : List Real := [1, 1.5, 2, 2.5]
  let resultingDimensions : Dimensions := {
    length := cuts[3],  -- Changed from cuts.last to cuts[3]
    width := cubeSide,
    height := cubeSide
  }
  surfaceArea resultingDimensions = 26 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_resulting_solid_surface_area_l107_10730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l107_10718

-- Define the sets M and N
def M : Set ℝ := {x | 1 < x + 1 ∧ x + 1 ≤ 3}
def N : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Define the theorem
theorem complement_intersection_theorem :
  (Set.univ \ M) ∩ (Set.univ \ N) = Set.Icc (-1 : ℝ) 0 ∪ Set.Ioc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l107_10718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_analysis_l107_10700

noncomputable section

/-- A function representing direct or inverse proportionality between x and y --/
def IsProportional (f : ℝ → ℝ) : Prop :=
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x) ∨ (∃ k : ℝ, ∀ x : ℝ, f x = k / x)

/-- Equation A: x^2 + y = 1 --/
noncomputable def EquationA (x : ℝ) : ℝ := 1 - x^2

/-- Equation B: 2xy = 5 --/
noncomputable def EquationB (x : ℝ) : ℝ := 5 / (2 * x)

/-- Equation C: x = 3y --/
noncomputable def EquationC (x : ℝ) : ℝ := x / 3

/-- Equation D: x^2 + 3x + y = 5 --/
noncomputable def EquationD (x : ℝ) : ℝ := 5 - x^2 - 3 * x

/-- Equation E: x/y = 5 --/
noncomputable def EquationE (x : ℝ) : ℝ := x / 5

theorem proportionality_analysis :
  ¬(IsProportional EquationA) ∧
  ¬(IsProportional EquationD) ∧
  (IsProportional EquationB) ∧
  (IsProportional EquationC) ∧
  (IsProportional EquationE) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_analysis_l107_10700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_right_branch_of_hyperbola_l107_10786

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the condition for point P
def is_on_trajectory (P : ℝ × ℝ) : Prop :=
  distance P F₁ - distance P F₂ = 8

-- Theorem statement
theorem trajectory_is_right_branch_of_hyperbola :
  ∀ P : ℝ × ℝ, is_on_trajectory P → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  (P.1 / a)^2 - (P.2 / b)^2 = 1 ∧ P.1 > 0 := by
  sorry

#check trajectory_is_right_branch_of_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_right_branch_of_hyperbola_l107_10786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l107_10701

theorem complex_number_opposite_parts (a : ℝ) : 
  let z : ℂ := a / (1 - 2*Complex.I) + Complex.I
  (z.re = -z.im) → a = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l107_10701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_iff_average_arithmetic_l107_10763

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The partial sum of a sequence up to the nth term -/
noncomputable def PartialSum (a : Sequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => a i)

/-- A sequence is arithmetic -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence of partial sum averages -/
noncomputable def PartialSumAverage (a : Sequence) (n : ℕ) : ℝ :=
  (PartialSum a n) / n

/-- Theorem stating the equivalence between an arithmetic sequence and its partial sum average sequence being arithmetic -/
theorem arithmetic_sequence_iff_average_arithmetic (a : Sequence) :
  IsArithmetic a ↔ IsArithmetic (PartialSumAverage a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_iff_average_arithmetic_l107_10763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_four_seven_l107_10765

-- Define the custom operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_four_seven : nabla 4 7 = 11 / 29 := by
  -- Unfold the definition of nabla
  unfold nabla
  -- Simplify the expression
  simp [add_div]
  -- Perform arithmetic calculations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_four_seven_l107_10765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_same_suit_l107_10736

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of suits in a standard deck -/
def numberOfSuits : ℕ := 4

/-- The number of cards in each suit -/
def cardsPerSuit : ℕ := standardDeckSize / numberOfSuits

/-- The probability of drawing four cards of the same suit from the top of a randomly arranged standard deck -/
def probabilityFourSameSuit : ℚ :=
  (numberOfSuits * (Nat.choose cardsPerSuit 4)) / (Nat.choose standardDeckSize 4)

/-- Theorem stating that the probability of drawing four cards of the same suit
    from the top of a randomly arranged standard 52-card deck is 2860/270725 -/
theorem probability_four_same_suit :
  probabilityFourSameSuit = 2860 / 270725 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_same_suit_l107_10736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_properties_l107_10796

-- Define a point
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a face
structure Face where
  vertices : Set Point
  is_polygon : Bool

-- Define a prism
structure Prism where
  base_faces : Set Face
  lateral_faces : Set Face
  is_prism : Bool

-- Define properties of a prism
def parallel (f1 f2 : Face) : Prop := sorry

def is_parallelogram (f : Face) : Prop := sorry

def congruent (f1 f2 : Face) : Prop := sorry

def Prism.has_parallel_non_base_planes (p : Prism) : Prop :=
  ∃ (f1 f2 : Face), f1 ∈ p.lateral_faces ∧ f2 ∈ p.lateral_faces ∧ f1 ≠ f2 ∧ parallel f1 f2

def Prism.lateral_faces_are_parallelograms (p : Prism) : Prop :=
  ∀ f ∈ p.lateral_faces, is_parallelogram f

def Prism.base_faces_are_congruent (p : Prism) : Prop :=
  ∃ (f1 f2 : Face), f1 ∈ p.base_faces ∧ f2 ∈ p.base_faces ∧ f1 ≠ f2 ∧ congruent f1 f2

def Prism.has_at_least_two_parallel_faces (p : Prism) : Prop :=
  ∃ (f1 f2 : Face), (f1 ∈ p.base_faces ∨ f1 ∈ p.lateral_faces) ∧ 
                    (f2 ∈ p.base_faces ∨ f2 ∈ p.lateral_faces) ∧ 
                    f1 ≠ f2 ∧ parallel f1 f2

-- Theorem stating the properties of a prism
theorem prism_properties (p : Prism) (h : p.is_prism) : 
  p.has_parallel_non_base_planes ∧
  p.lateral_faces_are_parallelograms ∧
  p.base_faces_are_congruent ∧
  p.has_at_least_two_parallel_faces :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_properties_l107_10796
