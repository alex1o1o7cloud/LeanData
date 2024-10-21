import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_l19_1941

/-- Calculates the profit amount given the selling price and profit percentage. -/
noncomputable def profit_amount (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  (profit_percentage / 100) * selling_price

/-- Theorem: The profit amount for a cricket bat sold for $900 with a 12.5% profit is $112.50. -/
theorem cricket_bat_profit :
  profit_amount 900 12.5 = 112.50 := by
  -- Unfold the definition of profit_amount
  unfold profit_amount
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_l19_1941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l19_1928

theorem inscribed_square_area (AB AC : ℝ) (h_right_angle : AB^2 + AC^2 = (AB + AC)^2 / 4) :
  let BC := Real.sqrt (AB^2 + AC^2)
  let s := BC^2 / (AB + AC)
  AB = 63 ∧ AC = 16 → s^2 = 2867 := by
    intro h
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l19_1928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accompanying_tangent_line_exists_l19_1964

noncomputable section

-- Define the function f(x) = ax + ln(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Theorem statement
theorem accompanying_tangent_line_exists (a : ℝ) :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ →
  ∃ (x₀ : ℝ), x₁ < x₀ ∧ x₀ < x₂ ∧
  (deriv (f a)) x₀ = (f a x₂ - f a x₁) / (x₂ - x₁) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_accompanying_tangent_line_exists_l19_1964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l19_1929

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

noncomputable def line_eq (x y : ℝ) (m : ℝ) : Prop := x - m*y + 1 = 0

noncomputable def intersects (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq x₁ y₁ m ∧ line_eq x₂ y₂ m

noncomputable def triangle_area (m : ℝ) : ℝ := 8/5

theorem line_circle_intersection (m : ℝ) :
  intersects m → triangle_area m = 8/5 → (m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l19_1929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersection_l19_1974

-- Define the structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the structure for a tetrahedron
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

-- Define the structure for a plane
structure Plane where
  normal : Point3D
  point : Point3D

-- Define a line
structure Line where
  direction : Point3D
  point : Point3D

-- Define the given conditions
def given_conditions 
  (tetra : Fin 3 → Tetrahedron)
  (α β γ : Fin 3 → Plane)
  (E : Point3D)
  (l : Line) : Prop :=
  ∀ i : Fin 3,
    (α i).point = (tetra i).B ∧
    (β i).point = (tetra i).C ∧
    (γ i).point = (tetra i).D ∧
    ((α i).normal.x * ((tetra i).B.x - (tetra i).A.x) +
     (α i).normal.y * ((tetra i).B.y - (tetra i).A.y) +
     (α i).normal.z * ((tetra i).B.z - (tetra i).A.z) = 0) ∧
    ((β i).normal.x * ((tetra i).C.x - (tetra i).A.x) +
     (β i).normal.y * ((tetra i).C.y - (tetra i).A.y) +
     (β i).normal.z * ((tetra i).C.z - (tetra i).A.z) = 0) ∧
    ((γ i).normal.x * ((tetra i).D.x - (tetra i).A.x) +
     (γ i).normal.y * ((tetra i).D.y - (tetra i).A.y) +
     (γ i).normal.z * ((tetra i).D.z - (tetra i).A.z) = 0) ∧
    (∃ t : ℝ, E = Point3D.mk ((α i).point.x + t * (α i).normal.x)
                              ((α i).point.y + t * (α i).normal.y)
                              ((α i).point.z + t * (α i).normal.z)) ∧
    (∃ t : ℝ, E = Point3D.mk ((β i).point.x + t * (β i).normal.x)
                              ((β i).point.y + t * (β i).normal.y)
                              ((β i).point.z + t * (β i).normal.z)) ∧
    (∃ t : ℝ, E = Point3D.mk ((γ i).point.x + t * (γ i).normal.x)
                              ((γ i).point.y + t * (γ i).normal.y)
                              ((γ i).point.z + t * (γ i).normal.z)) ∧
    (∃ t : ℝ, (tetra i).A = Point3D.mk (l.point.x + t * l.direction.x)
                                       (l.point.y + t * l.direction.y)
                                       (l.point.z + t * l.direction.z))

-- Define a circumsphere
def circumsphere (t : Tetrahedron) : Set Point3D :=
  {p : Point3D | ∃ r : ℝ, 
    (p.x - t.A.x)^2 + (p.y - t.A.y)^2 + (p.z - t.A.z)^2 = r^2 ∧
    (p.x - t.B.x)^2 + (p.y - t.B.y)^2 + (p.z - t.B.z)^2 = r^2 ∧
    (p.x - t.C.x)^2 + (p.y - t.C.y)^2 + (p.z - t.C.z)^2 = r^2 ∧
    (p.x - t.D.x)^2 + (p.y - t.D.y)^2 + (p.z - t.D.z)^2 = r^2}

-- Define the theorem
theorem tetrahedron_intersection
  (tetra : Fin 3 → Tetrahedron)
  (α β γ : Fin 3 → Plane)
  (E : Point3D)
  (l : Line)
  (h : given_conditions tetra α β γ E l) :
  ∃ (S : Set Point3D),
    S = (⋂ i : Fin 3, circumsphere (tetra i)) ∧
    ((∃ L : Line, (∀ p : Point3D, p ∈ S ↔ 
      ∃ t : ℝ, p = Point3D.mk (L.point.x + t * L.direction.x)
                               (L.point.y + t * L.direction.y)
                               (L.point.z + t * L.direction.z)) ∧
      (L.direction.x * l.direction.x + 
       L.direction.y * l.direction.y + 
       L.direction.z * l.direction.z = 0) ∧
      E ∈ S) ∨ 
     S = {E}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersection_l19_1974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_range_a_l19_1918

-- Define the functions g and f
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

-- Define the derivative of g with respect to x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := a - 2 * Real.sin x

-- Define the derivative of f with respect to x
noncomputable def f' (x : ℝ) : ℝ := -Real.exp x - 1

-- State the theorem
theorem parallel_tangents_range_a :
  (∀ a : ℝ, ∀ x₁ : ℝ, ∃ x₂ : ℝ, g' a x₁ = f' x₂) →
  (∀ a : ℝ, a ∈ Set.Ioi (-3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_range_a_l19_1918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_theorem_l19_1938

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define a line
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the set of lines that maximize the area of triangle AOB
def L : Set (ℝ → ℝ → Prop) :=
  {l | ∃ k m, (l = line k m ∧ k ≠ 0 ∧ m^2 = k^2 + 1/2) ∨ (l = line 0 1 ∨ l = line 0 (-1))}

-- Helper definitions
def area_triangle : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ := sorry
def area_quadrilateral : (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → ℝ := sorry
def parallel : (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → Prop := sorry

-- Main theorem
theorem ellipse_line_intersection_theorem :
  -- Maximum area of triangle AOB
  (∀ l : ℝ → ℝ → Prop, ∃ A B : ℝ × ℝ,
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 →
    area_triangle A B (0, 0) ≤ Real.sqrt 2 / 2) ∧
  -- Minimum area of quadrilateral formed by four lines from L
  (∀ l₁ l₂ l₃ l₄ : ℝ → ℝ → Prop,
    l₁ ∈ L ∧ l₂ ∈ L ∧ l₃ ∈ L ∧ l₄ ∈ L →
    ∃ k₁ k₂ k₃ k₄ : ℝ,
      parallel l₁ l₂ ∧ parallel l₃ l₄ ∧
      k₁ + k₂ + k₃ + k₄ = 0 →
      area_quadrilateral l₁ l₂ l₃ l₄ ≥ 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_theorem_l19_1938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_or_diagonal_not_exceeding_one_l19_1930

-- Define a convex hexagon
structure ConvexHexagon where
  -- Assume the existence of vertices, sides, and diagonals
  vertices : Finset (ℝ × ℝ)
  sides : Finset (ℝ × ℝ)
  diagonals : Finset (ℝ × ℝ)
  -- Ensure the hexagon is convex
  is_convex : Bool
  -- Ensure there are exactly 6 vertices, 6 sides, and 9 diagonals
  vertex_count : vertices.card = 6
  side_count : sides.card = 6
  diagonal_count : diagonals.card = 9

-- Define a function to get the length of a line segment
noncomputable def segmentLength (segment : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hexagon_side_or_diagonal_not_exceeding_one 
  (h : ConvexHexagon) 
  (longest_diagonal : ∃ d ∈ h.diagonals, segmentLength d = 2) :
  ∃ s ∈ h.sides ∪ h.diagonals, segmentLength s ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_or_diagonal_not_exceeding_one_l19_1930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l19_1933

theorem perpendicular_line_plane (e n : ℝ × ℝ × ℝ) (x : ℝ) :
  e = (1, 2, 1) →
  n = (1/2, x, 1/2) →
  (∃ (k : ℝ), e.1 = k * n.1 ∧ e.2.1 = k * n.2.1 ∧ e.2.2 = k * n.2.2) →
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l19_1933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l19_1957

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields here, e.g.:
  -- a : ℝ
  -- b : ℝ
  -- c : ℝ

/-- Represents that a triangle is isosceles -/
class Triangle.IsIsosceles (t : Triangle) where
  -- Add necessary conditions here

/-- Represents that a triangle is right-angled -/
class Triangle.IsRight (t : Triangle) where
  -- Add necessary conditions here

/-- Represents the two different diagrams for inscribing squares -/
inductive Diagram where
  | Diagram1
  | Diagram2

/-- Represents the area of an inscribed square in a triangle for a given diagram -/
def SquareArea (t : Triangle) (d : Diagram) : ℝ :=
  sorry -- Placeholder implementation

/-- Given an isosceles right triangle ABC with an inscribed square in diagram (1) 
    having an area of 441 cm², prove that the area of the inscribed square in 
    diagram (2) is 392 cm². -/
theorem inscribed_square_area (ABC : Triangle) 
  (h_isosceles : ABC.IsIsosceles)
  (h_right : ABC.IsRight)
  (h_square1_area : SquareArea ABC Diagram.Diagram1 = 441) : 
  SquareArea ABC Diagram.Diagram2 = 392 := by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l19_1957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_y_intercept_sum_l19_1931

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 4)
noncomputable def B : ℝ × ℝ := (6, 0)
noncomputable def C : ℝ × ℝ := (0, -2)

-- Define midpoints
noncomputable def E : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def F : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the slope of line BF
noncomputable def slope_BF : ℝ := (F.2 - B.2) / (F.1 - B.1)

-- Define the y-intercept of line BF
noncomputable def y_intercept_BF : ℝ := F.2 - slope_BF * F.1

-- Theorem statement
theorem slope_y_intercept_sum : slope_BF + y_intercept_BF = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_y_intercept_sum_l19_1931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_coordinate_l19_1907

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The x-coordinate of the vertex of the quadratic function -/
noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

theorem vertex_x_coordinate (a b c : ℝ) :
  quadratic_function a b c 0 = 1 ∧
  quadratic_function a b c 4 = 1 ∧
  quadratic_function a b c 7 = 5 →
  vertex_x a b c = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_coordinate_l19_1907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_on_cone_surface_l19_1956

-- Define the cone parameters
def baseRadius : ℝ := 400
def coneHeight : ℝ := 300  -- Changed variable name to avoid conflict

-- Define the distances of the points from the vertex
def startDistance : ℝ := 150
def endDistance : ℝ := 450

-- Define the slant height of the cone
noncomputable def slantHeight : ℝ := Real.sqrt (baseRadius^2 + coneHeight^2)

-- Define the angle of the sector when the cone is unwrapped
noncomputable def sectorAngle : ℝ := (2 * Real.pi * baseRadius) / slantHeight

-- The theorem to prove
theorem shortest_distance_on_cone_surface :
  let startPoint : ℝ × ℝ := (startDistance, 0)
  let endPoint : ℝ × ℝ := (-endDistance, 0)
  let distance := Real.sqrt ((endPoint.1 - startPoint.1)^2 + (endPoint.2 - startPoint.2)^2)
  distance = 600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_on_cone_surface_l19_1956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l19_1942

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem problem_solution (ω φ α : ℝ) 
  (h1 : ω > 0)
  (h2 : -π/2 < φ ∧ φ < π/2)
  (h3 : -3*π/4 < α ∧ α < -π/4)
  (h4 : f ω φ α = -4/5)
  (h5 : ∀ x, f ω φ (x + π/4) = f ω φ (π/4 - x))  -- symmetry about x = π/4
  (h6 : ∀ x, f ω φ (x + 5*π/4) = f ω φ (5*π/4 - x))  -- symmetry about x = 5π/4
  : ω = 1 ∧ φ = π/4 ∧ Real.sin α = -7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l19_1942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l19_1921

/-- Calculates the length of a train given its speed and time to cross a pole. -/
noncomputable def train_length (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_seconds

/-- Theorem stating that a train with a speed of 60 km/hr crossing a pole in 36 seconds
    has a length of approximately 600.12 meters. -/
theorem train_length_calculation :
  let speed := (60 : ℝ)
  let time := (36 : ℝ)
  abs (train_length speed time - 600.12) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l19_1921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_of_f_l19_1975

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

-- State the theorem
theorem min_point_of_f :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ = 2 ∧ 
  (∀ x > 0, f x ≥ f x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_of_f_l19_1975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_expression_equals_three_seventeenths_l19_1977

theorem inverse_expression_equals_three_seventeenths :
  (4 - 5 * (4 - 7 : ℝ)⁻¹)⁻¹ = 3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_expression_equals_three_seventeenths_l19_1977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_total_animals_l19_1997

def zoo_animals (snakes : ℕ) : ℕ :=
  let monkeys := 2 * snakes
  let lions := monkeys - 5
  let pandas := lions + 8
  let dogs := pandas / 3
  let elephants := 2 * (Nat.sqrt pandas)
  let birds := (3 * (monkeys + lions)) / 4
  let tigers := snakes / 2 + 1
  snakes + monkeys + lions + pandas + dogs + elephants + birds + tigers

theorem zoo_total_animals :
  zoo_animals 15 = 175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_total_animals_l19_1997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_functions_are_odd_l19_1989

noncomputable section

-- Function 1
def f1 (x : ℝ) : ℝ := 4 * Real.log (x + Real.sqrt (x^2 + 1))

-- Function 2
def f2 (x : ℝ) : ℝ := (1 + Real.sin (2*x) + Real.cos (2*x)) / (1 + Real.sin (2*x) - Real.cos (2*x))

-- Function 3
def f3 (x : ℝ) : ℝ := (1 + 2*Real.sin (x/2) + Real.sin x - Real.cos x) / (1 + 2*Real.cos (x/2) + Real.sin x + Real.cos x)

-- Function 4
def f4 (x : ℝ) : ℝ := if x ≥ 0 then 1 - Real.exp (-x * Real.log 2) else Real.exp (x * Real.log 2) - 1

-- Function 5
def f5 (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x^2 - 1)

-- Definition of odd function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem all_functions_are_odd :
  isOdd f1 ∧ isOdd f2 ∧ isOdd f3 ∧ isOdd f4 ∧ ∀ a, isOdd (f5 a) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_functions_are_odd_l19_1989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_existence_999_pairs_l19_1949

/-- The set of integers from which pairs are chosen -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2500}

/-- A structure representing a pair of integers -/
structure IntPair where
  a : ℕ
  b : ℕ
  h1 : a ∈ S
  h2 : b ∈ S
  h3 : a < b

/-- The property that all pairs are distinct -/
def distinct_pairs (pairs : List IntPair) : Prop :=
  ∀ i j, i ≠ j → 
    (pairs.get i).a ≠ (pairs.get j).a ∧ (pairs.get i).a ≠ (pairs.get j).b ∧
    (pairs.get i).b ≠ (pairs.get j).a ∧ (pairs.get i).b ≠ (pairs.get j).b

/-- The property that all sums are distinct and ≤ 2500 -/
def distinct_sums (pairs : List IntPair) : Prop :=
  ∀ i j, i ≠ j → 
    (pairs.get i).a + (pairs.get i).b ≠ (pairs.get j).a + (pairs.get j).b ∧
    (pairs.get i).a + (pairs.get i).b ≤ 2500

/-- The main theorem stating the maximum number of pairs -/
theorem max_pairs :
  ∀ pairs : List IntPair,
    distinct_pairs pairs →
    distinct_sums pairs →
    pairs.length ≤ 999 :=
by sorry

/-- The existence of 999 pairs satisfying the conditions -/
theorem existence_999_pairs :
  ∃ pairs : List IntPair,
    pairs.length = 999 ∧
    distinct_pairs pairs ∧
    distinct_sums pairs :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_existence_999_pairs_l19_1949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_l19_1920

def IsSubsetWithProperties (A : Set ℕ) : Prop :=
  (∀ a, a ∈ A → ∀ d : ℕ, d ∣ a → d ∈ A) ∧
  (∀ a b, a ∈ A → b ∈ A → 1 < a → a < b → 1 + a * b ∈ A)

theorem all_positive_integers (A : Set ℕ) 
  (h_subset : IsSubsetWithProperties A) 
  (h_size : ∃ x y z : ℕ, x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) : 
  A = {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_l19_1920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_2200_l19_1940

noncomputable def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

theorem count_ordered_pairs_2200 :
  let factorization := prime_factorization 2200
  ∀ (f : List (ℕ × ℕ)), f = factorization →
    (f.length = 3 ∧
     f.any (λ p => p = (2, 3)) ∧
     f.any (λ p => p = (5, 2)) ∧
     f.any (λ p => p = (11, 1))) →
    (Finset.card (Finset.filter (λ p : ℕ × ℕ => p.1 * p.2 = 2200 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.range 2201 ×ˢ Finset.range 2201)) = 24) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_2200_l19_1940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_days_B_is_15_l19_1954

noncomputable def work_days_B (days_A : ℝ) (total_wages : ℝ) (A_wages : ℝ) : ℝ :=
  (3 * days_A * total_wages) / (2 * A_wages)

theorem work_days_B_is_15 :
  work_days_B 10 3300 1980 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_days_B_is_15_l19_1954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qz_length_l19_1955

/-- A quadrilateral with specific properties -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  ab_parallel_yz : (A.2 - B.2) / (A.1 - B.1) = (Y.2 - Z.2) / (Y.1 - Z.1)
  az_length : Real.sqrt ((A.1 - Z.1)^2 + (A.2 - Z.2)^2) = 56
  bq_length : Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) = 18
  qy_length : Real.sqrt ((Q.1 - Y.1)^2 + (Q.2 - Y.2)^2) = 36
  q_on_az : (Q.2 - A.2) / (Q.1 - A.1) = (Z.2 - A.2) / (Z.1 - A.1)
  q_on_by : (Q.2 - B.2) / (Q.1 - B.1) = (Y.2 - B.2) / (Y.1 - B.1)

/-- The length of QZ in the quadrilateral is 112/3 -/
theorem qz_length (quad : Quadrilateral) : 
  Real.sqrt ((quad.Q.1 - quad.Z.1)^2 + (quad.Q.2 - quad.Z.2)^2) = 112/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_qz_length_l19_1955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_division_ratio_l19_1986

/-- Represents the ratio in which a plane divides the base circle of a cone -/
noncomputable def coneDivisionRatio (α β : ℝ) : ℝ :=
  Real.arccos (Real.tan β / Real.tan α) / (Real.pi - Real.arccos (Real.tan β / Real.tan α))

/-- Theorem stating the ratio in which a plane divides the base circle of a cone -/
theorem cone_division_ratio (α β : ℝ) (h1 : 0 < α) (h2 : α < Real.pi/2) (h3 : 0 < β) (h4 : β < α) :
  coneDivisionRatio α β = Real.arccos (Real.tan β / Real.tan α) / (Real.pi - Real.arccos (Real.tan β / Real.tan α)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_division_ratio_l19_1986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_statistics_l19_1925

noncomputable def sample : List ℝ := [10, 11, 9, 13, 12]

noncomputable def sample_mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def sample_variance (xs : List ℝ) : ℝ :=
  let mean := sample_mean xs
  (xs.map (λ x => (x - mean) ^ 2)).sum / xs.length

theorem sample_statistics :
  sample_mean sample = 11 ∧ sample_variance sample = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_statistics_l19_1925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rational_distance_set_l19_1993

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℚ
  y : ℚ

/-- The set of points we're constructing -/
def PointSet : Set Point :=
  {p | ∃ k : ℕ, k > 0 ∧ p.x = 1 ∧ p.y = (k^2 - 1) / (2*k)}

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Three points are collinear if the area of the triangle formed by them is zero -/
def collinear (p q r : Point) : Prop :=
  (p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)) = 0

theorem infinite_rational_distance_set :
  ∃ S : Set Point,
    Set.Infinite S ∧
    (∀ p q, p ∈ S → q ∈ S → p ≠ q → ∃ r : ℚ, distance p q = r) ∧
    (∀ p q r, p ∈ S → q ∈ S → r ∈ S → p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬collinear p q r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rational_distance_set_l19_1993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_ratio_proof_l19_1979

/-- The ratio of collapsing buildings between subsequent earthquakes -/
noncomputable def ratio : ℝ := 2

/-- The number of earthquakes -/
def num_earthquakes : ℕ := 4

/-- The number of buildings collapsed in the initial earthquake -/
def initial_collapse : ℕ := 4

/-- The total number of collapsed buildings after all earthquakes -/
def total_collapse : ℕ := 60

/-- The sum of a geometric series with first term a, ratio r, and n terms -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem earthquake_ratio_proof :
  geometric_sum (initial_collapse : ℝ) ratio num_earthquakes = total_collapse := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_ratio_proof_l19_1979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_stack_height_l19_1952

/-- Represents the stack of metal discs -/
structure DiscStack where
  top_diameter : ℝ
  bottom_diameter : ℝ
  diameter_decrease : ℝ
  disc_thickness : ℝ
  gap_between_discs : ℝ

/-- Calculates the number of discs in the stack -/
noncomputable def num_discs (stack : DiscStack) : ℕ :=
  Nat.floor ((stack.top_diameter - stack.bottom_diameter) / stack.diameter_decrease + 1)

/-- Calculates the total vertical distance of the disc stack -/
noncomputable def total_vertical_distance (stack : DiscStack) : ℝ :=
  let n := num_discs stack
  (n : ℝ) * stack.disc_thickness + ((n : ℝ) - 1) * stack.gap_between_discs

/-- Theorem stating that the total vertical distance is 22 cm for the given conditions -/
theorem disc_stack_height : 
  let stack := DiscStack.mk 30 2 2 1 0.5
  total_vertical_distance stack = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_stack_height_l19_1952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l19_1903

theorem sin_cos_equation_solution : ∃ x : ℝ, 
  x = π / 16 ∧ Real.sin (2 * x) * Real.sin (4 * x) = Real.cos (2 * x) * Real.cos (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l19_1903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ruffy_orlie_age_ratio_l19_1978

/-- Given:
  - Ruffy is 9 years old now
  - Four years ago, Ruffy was 1 year more than half as old as Orlie
Prove that Ruffy's age is 3/4 of Orlie's age -/
theorem ruffy_orlie_age_ratio : 
  ∀ (ruffy_age orlie_age : ℕ),
  ruffy_age = 9 →
  ruffy_age - 4 = 1 + (orlie_age - 4) / 2 →
  ruffy_age * 4 = orlie_age * 3 := by
  intros ruffy_age orlie_age h1 h2
  -- The proof steps would go here
  sorry

#check ruffy_orlie_age_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ruffy_orlie_age_ratio_l19_1978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l19_1982

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (-3, -2)

-- Define the line equation
def line_BC (x y : ℝ) : Prop := x - y + 1 = 0

-- Define right-angled triangle
def is_right_angled (a b c : ℝ × ℝ) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let bc := (c.1 - b.1, c.2 - b.2)
  ab.1 * bc.1 + ab.2 * bc.2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 5

theorem triangle_properties :
  (∀ x y, line_BC x y ↔ (y - B.2) = ((C.2 - B.2) / (C.1 - B.1)) * (x - B.1)) ∧
  is_right_angled A B C ∧
  (∀ x y, circle_eq x y ↔ (x - (-1))^2 + (y - (-1))^2 = 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l19_1982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_product_sum_l19_1943

def CubeFaces := Fin 6 → Nat

def is_valid_assignment (f : CubeFaces) : Prop :=
  (∀ i j : Fin 6, i ≠ j → f i ≠ f j) ∧
  (∀ i : Fin 6, f i ∈ Finset.range 7 \ {0})

def vertex_product_sum (f : CubeFaces) : Nat :=
  let a := f 0
  let b := f 1
  let c := f 2
  let d := f 3
  let e := f 4
  let f := f 5
  (a + b) * (c + d) * (e + f)

theorem max_vertex_product_sum :
  ∀ f : CubeFaces, is_valid_assignment f →
  vertex_product_sum f ≤ 343 :=
by
  sorry

#check max_vertex_product_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_product_sum_l19_1943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l19_1972

-- Define the universal set U
def U : Set ℕ := {0, 1, 3, 5, 6, 8}

-- Define set A
def A : Set ℕ := {1, 5, 8}

-- Define set B
def B : Set ℕ := {2}

-- Theorem statement
theorem complement_union_problem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l19_1972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_angle_proof_l19_1985

theorem polygon_angle_proof (a b c d : ℝ) : 
  -- Sum of three marked angles in a polygon is a°
  a = 900 →
  -- Sum of interior angles of a convex b-sided polygon is a°
  a = 180 * (b - 2) →
  -- Number of sides of the polygon
  b = 7 →
  -- Exponential equation
  27^(b - 1) = c^18 →
  -- Logarithmic equation
  c = Real.logb d 125 →
  -- Conclusion
  d = 5 := by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_angle_proof_l19_1985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_range_l19_1923

/-- The function f(x) = xe^x - a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem two_zeros_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ∧
  (∀ z w v : ℝ, z ≠ w ∧ z ≠ v ∧ w ≠ v → ¬(f a z = 0 ∧ f a w = 0 ∧ f a v = 0)) →
  -1 / Real.exp 1 < a ∧ a < 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_range_l19_1923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_to_line_l_l19_1911

noncomputable section

-- Define the curve C
def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define the line l in polar form
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi / 4) = 2 * Real.sqrt 2

-- Cartesian equation of line l
def line_l_cartesian (x y : ℝ) : Prop := x + y = 4

-- Distance function from a point to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  abs (x + y - 4) / Real.sqrt 2

-- Maximum distance from curve C to line l
def max_distance : ℝ := 3 * Real.sqrt 2

-- Point on curve C with maximum distance to line l
def max_distance_point : ℝ × ℝ := (-3/2, -1/2)

theorem curve_C_to_line_l :
  (∀ ρ θ, line_l ρ θ ↔ line_l_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∀ α, distance_to_line (curve_C α).1 (curve_C α).2 ≤ max_distance) ∧
  (distance_to_line (max_distance_point.1) (max_distance_point.2) = max_distance) ∧
  (∃ α, curve_C α = max_distance_point) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_to_line_l_l19_1911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hybrid_car_journey_theorem_l19_1904

/-- Represents a hybrid car journey between two locations -/
structure HybridCarJourney where
  oil_cost : ℝ  -- Total cost when using only oil
  electricity_cost : ℝ  -- Total cost when using only electricity
  oil_electricity_diff : ℝ  -- Difference in cost per km between oil and electricity

/-- Calculates properties of a hybrid car journey -/
noncomputable def journey_properties (j : HybridCarJourney) : ℝ × ℝ × ℝ :=
  let electricity_cost_per_km := (j.oil_cost * j.electricity_cost) / (j.oil_cost - j.electricity_cost * (1 + j.oil_electricity_diff))
  let distance := j.electricity_cost / electricity_cost_per_km
  let min_electric_distance := (j.oil_cost - 50) / j.oil_electricity_diff
  (electricity_cost_per_km, distance, min_electric_distance)

theorem hybrid_car_journey_theorem (j : HybridCarJourney) 
  (h1 : j.oil_cost = 80)
  (h2 : j.electricity_cost = 30)
  (h3 : j.oil_electricity_diff = 0.5) :
  journey_properties j = (0.3, 100, 60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hybrid_car_journey_theorem_l19_1904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l19_1922

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = 2 * c.p * x

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem about parabola properties -/
theorem parabola_properties (c : Parabola) 
  (B : ParabolaPoint c) 
  (hB : B.x = 1/2) 
  (hBl : distance B.x B.y (-c.p) 0 = distance B.x B.y 0 0) :
  (c.p = 2) ∧
  (∀ (A : ParabolaPoint c) (hA : A.x ≠ c.p/2),
    ∃ (k b : ℝ),
      let N := (-1, 1/k - k)
      let M := (-1, -2*k)
      (distance (1) (0) N.1 N.2)^2 + (distance (-3) (0) N.1 N.2)^2 = 
      (distance M.1 M.2 N.1 N.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l19_1922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l19_1934

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  -- Define triangle ABC
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Define sides a, b, c opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Radius of circumcircle is 1
  a / (2 * Real.sin A) = 1 ∧ b / (2 * Real.sin B) = 1 ∧ c / (2 * Real.sin C) = 1 →
  -- Given condition
  Real.tan A / Real.tan B = (2 * c - b) / b →
  -- Maximum area of triangle ABC
  (∃ (S : ℝ), S = a * b * Real.sin C / 2 ∧ 
    ∀ (S' : ℝ), S' = a * b * Real.sin C / 2 → S' ≤ Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l19_1934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_together_arrangements_girls_separated_arrangements_girls_not_at_ends_arrangements_fixed_boys_order_arrangements_girls_front_boys_back_arrangements_l19_1996

-- Define the number of girls and boys
def num_girls : ℕ := 3
def num_boys : ℕ := 5

-- (1) Girls must stand together
theorem girls_together_arrangements (n m : ℕ) : 
  n.factorial * (m + 1).factorial = 4320 ∧ n = num_girls ∧ m = num_boys :=
sorry

-- (2) Girls must be separated
theorem girls_separated_arrangements (n m : ℕ) : 
  m.factorial * Nat.choose (m + 1) n = 14400 ∧ n = num_girls ∧ m = num_boys :=
sorry

-- (3) Girls cannot stand at either end
theorem girls_not_at_ends_arrangements (n m : ℕ) : 
  (Nat.factorial m / Nat.factorial (m - 2)) * (n + m - 2).factorial = 14400 ∧ n = num_girls ∧ m = num_boys :=
sorry

-- (4) Boys stand in a fixed order
theorem fixed_boys_order_arrangements (n m : ℕ) : 
  (n + m).factorial / (n + m - n).factorial = 336 ∧ n = num_girls ∧ m = num_boys :=
sorry

-- (5) Girls in front row, boys in back row
theorem girls_front_boys_back_arrangements (n m : ℕ) : 
  n.factorial * m.factorial = 720 ∧ n = num_girls ∧ m = num_boys :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_together_arrangements_girls_separated_arrangements_girls_not_at_ends_arrangements_fixed_boys_order_arrangements_girls_front_boys_back_arrangements_l19_1996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caffeine_amount_l19_1937

/-- The amount of caffeine in John's first energy drink --/
noncomputable def caffeine_in_first_drink (first_drink_oz : ℝ) (second_drink_oz : ℝ) 
  (caffeine_ratio : ℝ) (total_caffeine : ℝ) : ℝ :=
  (first_drink_oz * total_caffeine) / (first_drink_oz + second_drink_oz * caffeine_ratio + 
  (first_drink_oz + second_drink_oz * caffeine_ratio))

/-- Theorem stating the amount of caffeine in the first drink --/
theorem caffeine_amount : 
  let first_drink_oz : ℝ := 12
  let second_drink_oz : ℝ := 2
  let caffeine_ratio : ℝ := 3
  let total_caffeine : ℝ := 750
  ∃ ε > 0, |caffeine_in_first_drink first_drink_oz second_drink_oz caffeine_ratio total_caffeine - 250| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_caffeine_amount_l19_1937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_positive_integers_l19_1970

def first_twelve_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

theorem median_of_first_twelve_positive_integers :
  let sorted_list := first_twelve_positive_integers
  let n := sorted_list.length
  let middle1 := sorted_list[n / 2 - 1]
  let middle2 := sorted_list[n / 2]
  (middle1 + middle2) / 2 = (13 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_positive_integers_l19_1970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l19_1971

theorem smallest_positive_z (x y z : ℝ) 
  (hx : Real.cos x = 0) 
  (hy : Real.sin y = 1) 
  (hxz : Real.cos (x + z) = -1/2) : 
  ∃ (w : ℝ), w > 0 ∧ w = 5*Real.pi/6 ∧ ∀ (z' : ℝ), z' > 0 → Real.cos (x + z') = -1/2 → z' ≥ w :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l19_1971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_selection_l19_1927

/-- A pair of elements -/
structure Pair where
  first : ℕ
  second : ℕ

/-- A configuration of 32 pairs in an 8x8 grid -/
def Configuration := Fin 8 → Fin 8 → Pair

/-- A selection of one element from each pair -/
def Selection := Fin 8 → Fin 8 → Bool

/-- Check if a selection is valid (one from each pair, at least one per row and column) -/
def is_valid_selection (config : Configuration) (sel : Selection) : Prop :=
  (∀ i j, sel i j = true ∨ sel i j = false) ∧
  (∀ i, ∃ j, sel i j = true) ∧
  (∀ j, ∃ i, sel i j = true) ∧
  (∀ i j, sel i j = true → sel i j ≠ false)

theorem exists_valid_selection (config : Configuration) :
  ∃ sel : Selection, is_valid_selection config sel := by
  sorry

#check exists_valid_selection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_selection_l19_1927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_decreasing_range_l19_1947

/-- Given a cubic function with specific properties, prove the range of t for which the function is monotonically decreasing on [t, t+1] -/
theorem cubic_function_decreasing_range (f : ℝ → ℝ) (t : ℝ) : 
  (∀ x, f x = x^3 + 3*x^2) → 
  (HasDerivAt f (-3) (-1)) →
  (f (-1) = 2) →
  (StrictMonoOn f (Set.Icc t (t+1))) →
  t ∈ Set.Icc (-2) (-1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_decreasing_range_l19_1947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_children_count_l19_1951

theorem farmer_children_count :
  ∀ (n : ℕ) 
    (apples_per_child : ℕ) 
    (eaten_per_child : ℕ) 
    (eaten_children : ℕ) 
    (sold_apples : ℕ) 
    (remaining_apples : ℕ),
  apples_per_child = 15 →
  eaten_per_child = 4 →
  eaten_children = 2 →
  sold_apples = 7 →
  remaining_apples = 60 →
  n * apples_per_child = 
    remaining_apples + sold_apples + (eaten_children * eaten_per_child) →
  n = 5 := by
  intros n apples_per_child eaten_per_child eaten_children sold_apples remaining_apples
  intros h1 h2 h3 h4 h5 h6
  sorry

#check farmer_children_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_children_count_l19_1951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_good_fruits_approx_l19_1994

def total_fruits : ℕ := 600 + 400 + 300 + 200

def rotten_oranges : ℕ := (600 * 15) / 100
def rotten_bananas : ℕ := (400 * 5) / 100
def rotten_apples : ℕ := (300 * 8) / 100
def rotten_avocados : ℕ := (200 * 10) / 100

def total_rotten_fruits : ℕ := rotten_oranges + rotten_bananas + rotten_apples + rotten_avocados

def good_fruits : ℕ := total_fruits - total_rotten_fruits

theorem percentage_good_fruits_approx (ε : ℚ) (h : ε > 0) :
  ∃ (p : ℚ), |p - (good_fruits : ℚ) / (total_fruits : ℚ) * 100| < ε ∧ |p - 89.73| < 0.01 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_good_fruits_approx_l19_1994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geography_marks_calculation_l19_1950

/-- Calculates the marks in geography given the marks in other subjects and the average --/
def geography_marks (history_govt : ℕ) (art : ℕ) (comp_sci : ℕ) (literature : ℕ) (average : ℚ) : ℕ :=
  (average * 5).floor.toNat - (history_govt + art + comp_sci + literature)

/-- Theorem stating that given the marks in other subjects and the average, the marks in geography are 56 --/
theorem geography_marks_calculation :
  geography_marks 60 72 85 80 (70.6 : ℚ) = 56 := by
  -- Unfold the definition of geography_marks
  unfold geography_marks
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geography_marks_calculation_l19_1950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_sphere_volume_ratio_proof_l19_1908

/-- The ratio of the volume of a regular icosahedron inscribed in a sphere
    to the volume of the sphere. -/
noncomputable def icosahedron_sphere_volume_ratio : ℝ :=
  (500 * (3 + Real.sqrt 5) * (5 + 2 * Real.sqrt 5) ^ (3/2)) /
  (72 * Real.sqrt 3375 * Real.pi)

/-- Theorem stating that the ratio of the volume of a regular icosahedron
    inscribed in a sphere to the volume of the sphere is equal to the
    calculated value. -/
theorem icosahedron_sphere_volume_ratio_proof (r : ℝ) (r_pos : r > 0) :
  let sphere_volume := (4/3) * Real.pi * r^3
  let icosahedron_side := r * Real.sqrt ((10 * (5 + 2 * Real.sqrt 5)) / 15)
  let icosahedron_volume := (5/12) * (3 + Real.sqrt 5) * icosahedron_side^3
  icosahedron_volume / sphere_volume = icosahedron_sphere_volume_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_sphere_volume_ratio_proof_l19_1908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l19_1902

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2)
  (hb : b % 7 = 3)
  (hc : c % 7 = 4) :
  (a * b * c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l19_1902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_toy_cars_l19_1926

/-- Calculates the number of toy cars bought given the initial amount,
    cost of items, and remaining amount. -/
def toy_cars_bought (initial_amount race_track_cost car_cost remaining_amount : ℚ) : ℕ :=
  let total_spent := initial_amount - remaining_amount
  let spent_on_cars := total_spent - race_track_cost
  (spent_on_cars / car_cost).floor.toNat

/-- Proves that Edward bought 4 toy cars given the problem conditions. -/
theorem edward_toy_cars :
  toy_cars_bought 17.80 6.00 0.95 8.00 = 4 := by
  sorry

#eval toy_cars_bought 17.80 6.00 0.95 8.00

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_toy_cars_l19_1926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_2_supports_safety_statement_1_insufficient_info_l19_1995

-- Define the data from the problem
def car_deaths_per_year : ℕ := 1240000
noncomputable def plane_accidents_per_million_flights : ℝ := 2.1
def car_deaths_per_million_travelers : ℕ := 100

-- Define a function to calculate plane deaths per million travelers
noncomputable def plane_deaths_per_million_travelers (avg_passengers_per_flight : ℝ) : ℝ :=
  plane_accidents_per_million_flights * avg_passengers_per_flight / 1000000

-- Theorem stating that statement ② supports the claim of airplane safety
theorem statement_2_supports_safety (avg_passengers_per_flight : ℝ) 
  (h : avg_passengers_per_flight > 0) :
  plane_deaths_per_million_travelers avg_passengers_per_flight < car_deaths_per_million_travelers :=
by sorry

-- Theorem stating that statement ① does not provide sufficient information
theorem statement_1_insufficient_info : 
  ∀ (total_car_travelers : ℕ) (total_plane_travelers : ℕ),
  (car_deaths_per_year : ℝ) / total_car_travelers ≠ plane_accidents_per_million_flights / total_plane_travelers :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_2_supports_safety_statement_1_insufficient_info_l19_1995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_years_l19_1976

/-- The number of years for a machine to depreciate from initial value to final value at a given rate -/
noncomputable def depreciation_years (initial_value : ℝ) (final_value : ℝ) (rate : ℝ) : ℝ :=
  Real.log (final_value / initial_value) / Real.log (1 - rate)

/-- Theorem: The number of years for a machine with initial value $128,000 to depreciate
    to $54,000 at a rate of 25% per annum is 3 years -/
theorem machine_depreciation_years :
  depreciation_years 128000 54000 0.25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_years_l19_1976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_l19_1945

/-- Predicate to check if four points form a square -/
def is_square (A B C D : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if each point of IJKL is on a side of WXYZ -/
def on_sides (I J K L W X Y Z : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the area of a square given its four vertices -/
def area (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Given two squares WXYZ and IJKL, where IJKL has one vertex on each side of WXYZ,
    and point I is on WZ such that WI = 3 · IZ, prove that the ratio of the area of IJKL
    to the area of WXYZ is 1/8. -/
theorem square_area_ratio (W X Y Z I J K L : ℝ × ℝ) : 
  is_square W X Y Z →
  is_square I J K L →
  on_sides I J K L W X Y Z →
  I.1 = W.1 + 3 * (Z.1 - I.1) →
  I.2 = W.2 →
  area I J K L / area W X Y Z = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_l19_1945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_work_proof_l19_1917

/-- Work done in lifting a satellite from Earth's surface to a given height -/
noncomputable def work_done (m : ℝ) (g : ℝ) (R : ℝ) (H : ℝ) : ℝ :=
  m * g * R * (1 - 1 / (1 + H / R))

/-- Proof that the work done in lifting a satellite is approximately 1.72 × 10^10 J -/
theorem satellite_work_proof :
  let m : ℝ := 7000  -- mass in kg
  let g : ℝ := 10    -- acceleration due to gravity in m/s²
  let R : ℝ := 6380000  -- Earth's radius in m
  let H : ℝ := 250000   -- height in m
  ∃ ε > 0, |work_done m g R H - 1.72e10| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval work_done 7000 10 6380000 250000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_work_proof_l19_1917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_and_logarithm_inequalities_l19_1961

theorem exponential_and_logarithm_inequalities :
  (∀ x : ℝ, x ≠ 0 → Real.exp x > 1 + x) ∧
  (∀ n : ℕ, 1 / (n + 1 : ℝ) < Real.log ((n + 1 : ℝ) / n) ∧ Real.log ((n + 1 : ℝ) / n) < 1 / n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_and_logarithm_inequalities_l19_1961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l19_1983

-- Define the hyperbola Γ
def hyperbola (t : ℝ) (x y : ℝ) : Prop :=
  x^2 / t^2 - y^2 = 1 ∧ t > 0

-- Define the eccentricity
noncomputable def eccentricity (t : ℝ) : ℝ :=
  Real.sqrt (t^2 + 1) / t

-- Define the focal length
noncomputable def focal_length (t : ℝ) : ℝ :=
  2 * Real.sqrt (t^2 + 1)

-- Define the left focus
noncomputable def left_focus (t : ℝ) : ℝ × ℝ :=
  (-Real.sqrt (t^2 + 1), 0)

-- Define the asymptote
def asymptote (t : ℝ) (x y : ℝ) : Prop :=
  y = x / t ∨ y = -x / t

-- Define the line through F₁
noncomputable def line_through_focus (t : ℝ) (x y : ℝ) : Prop :=
  y = t * x + t * Real.sqrt (t^2 + 1)

-- Define the intersection point M
noncomputable def intersection_point (t : ℝ) : ℝ × ℝ :=
  (-(t^2 * Real.sqrt (t^2 + 1)) / (t + 1), (t * Real.sqrt (t^2 + 1)) / (t + 1))

-- Define the area of triangle MOF₁
noncomputable def triangle_area (t : ℝ) : ℝ :=
  abs (t * (t^2 + 1) / (2 * (t + 1)))

-- Define the line l
def line_l (k m : ℝ) (x y : ℝ) : Prop :=
  k * x - y + m = 0 ∧ k > 0

theorem hyperbola_properties (t : ℝ) :
  (∀ x y, hyperbola t x y) ∧ eccentricity t = Real.sqrt 10 / 3 →
  (focal_length t = 2 * Real.sqrt 10) ∧
  (triangle_area t = 1/2 → t = 1) ∧
  (t = Real.sqrt 2 →
    ∀ k m : ℝ,
    (∃ P Q : ℝ × ℝ,
      hyperbola t P.1 P.2 ∧
      hyperbola t Q.1 Q.2 ∧
      line_l k m P.1 P.2 ∧
      line_l k m Q.1 Q.2 ∧
      Real.sqrt ((P.1 + Q.1)^2 + (P.2 + Q.2)^2) = 4) →
    0 < k ∧ k < Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l19_1983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l19_1967

theorem calculation_proof : (3.14 - Real.pi) ^ (0 : ℤ) + |Real.sqrt 2 - 1| + (1/2)^(-1 : ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l19_1967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_similarity_proof_l19_1968

-- Define the two parabolas
noncomputable def parabola1 (x : ℝ) : ℝ := -x^2 + 6*x - 10
noncomputable def parabola2 (x : ℝ) : ℝ := (1/2)*(x^2 + 6*x + 13)

-- Define the center of similarity
def center_of_similarity : ℝ × ℝ := (1, 0)

-- Define centrally_similar (placeholder definition)
def centrally_similar (f g : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem center_of_similarity_proof :
  let p1 := parabola1
  let p2 := parabola2
  centrally_similar p1 p2 →
  center_of_similarity = (1, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_similarity_proof_l19_1968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l19_1959

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f passes through (1, 0)
axiom f_at_one : f 1 = 0

-- f'(x) is the derivative of f(x)
def f' : ℝ → ℝ := deriv f

-- For x > 0, xf'(x) > 1
axiom xf'_gt_one : ∀ x : ℝ, x > 0 → x * (f' f x) > 1

theorem solution_set_of_inequality (x : ℝ) :
  (0 < x ∧ x ≤ 1) ↔ f x ≤ Real.log x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l19_1959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_implies_a_in_range_l19_1914

/-- The function f(x) defined as 2x^2 + (x-a)|x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 + (x - a) * |x - a|

/-- The property of f not being monotonic on the interval [-3,0] -/
def not_monotonic (a : ℝ) : Prop :=
  ¬(∀ x y, x ∈ Set.Icc (-3 : ℝ) 0 → y ∈ Set.Icc (-3 : ℝ) 0 → x ≤ y → f a x ≤ f a y)
  ∧ ¬(∀ x y, x ∈ Set.Icc (-3 : ℝ) 0 → y ∈ Set.Icc (-3 : ℝ) 0 → x ≤ y → f a x ≥ f a y)

/-- The theorem stating that if f is not monotonic on [-3,0], then a is in (-9,0) ∪ (0,3) -/
theorem f_not_monotonic_implies_a_in_range :
  ∀ a : ℝ, not_monotonic a → a ∈ Set.union (Set.Ioo (-9 : ℝ) 0) (Set.Ioo 0 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_implies_a_in_range_l19_1914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l19_1998

/-- Given a train's speed in km/h, time to cross a platform, and time to cross a man,
    calculate the length of the platform in meters. -/
noncomputable def platform_length (train_speed_kmph : ℝ) (time_platform : ℝ) (time_man : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * time_man
  let total_length := train_speed_mps * time_platform
  total_length - train_length

/-- Theorem stating that for a train traveling at 72 km/h, crossing a platform in 30 seconds
    and a man in 20 seconds, the platform length is 200 meters. -/
theorem platform_length_calculation :
  platform_length 72 30 20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l19_1998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_b_zero_l19_1919

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (bx + 2) / (x + a) --/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  (b * x + 2) / (x + a)

/-- If f(x) = (bx + 2) / (x + a) is an odd function, then a = 0 and b = 0 --/
theorem odd_function_implies_a_b_zero (a b : ℝ) :
  IsOdd (f a b) → a = 0 ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_b_zero_l19_1919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sphere_radius_l19_1973

/-- The weight of a hollow sphere is directly proportional to its surface area -/
def weight_proportional_to_area (r w : ℝ) : Prop := ∃ k, k > 0 ∧ w = k * (4 * Real.pi * r^2)

/-- Given: A hollow sphere of unknown radius r₁ weighs 8 grams -/
def sphere1_weight : ℝ := 8

/-- Given: A hollow sphere of radius 0.3 cm weighs 32 grams -/
def sphere2_radius : ℝ := 0.3
def sphere2_weight : ℝ := 32

/-- The theorem to prove -/
theorem first_sphere_radius :
  ∃ (r₁ : ℝ), r₁ > 0 ∧ 
  weight_proportional_to_area r₁ sphere1_weight ∧
  weight_proportional_to_area sphere2_radius sphere2_weight ∧
  r₁ = 0.15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sphere_radius_l19_1973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_term_l19_1916

noncomputable def a (n : ℕ) : ℝ := n / (n^2 + 81)

theorem sequence_max_term :
  ∃ (max : ℝ) (n_max : ℕ),
    (∀ n : ℕ, a n ≤ max) ∧
    (a n_max = max) ∧
    (max = 1/18) ∧
    (n_max = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_term_l19_1916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_work_time_l19_1906

theorem machine_work_time (x : ℝ) : x = 1 :=
  by
  -- Define the time taken by each machine to complete the job alone
  let time_R := x + 4
  let time_Q := time_R + 5
  let time_P := time_Q + 3

  -- Define the equation for combined work
  have combined_work_eq : 1 / time_P + 1 / time_Q + 1 / time_R = 1 / x := by
    sorry -- The proof of this equality is omitted

  -- Expand and simplify the equation
  have expanded_eq : x^4 + 25*x^3 + 187*x^2 + 526*x + 240 = 0 := by
    sorry -- The proof of this expansion is omitted

  -- Show that x = 1 is a solution
  have x_is_one : x = 1 := by
    sorry -- The proof that x = 1 satisfies the equation is omitted

  exact x_is_one


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_work_time_l19_1906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_percentage_l19_1939

/-- Calculates the discount percentage for a merchant's pricing strategy -/
theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) : 
  markup_percentage = 0.5 → profit_percentage = 0.2 → 
  (let cost_price := 100
   let marked_price := cost_price * (1 + markup_percentage)
   let selling_price := cost_price * (1 + profit_percentage)
   let discount_amount := marked_price - selling_price
   let discount_percentage := (discount_amount / marked_price) * 100
   discount_percentage) = 20 := by
  intros h1 h2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_percentage_l19_1939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_l19_1991

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference
  d_neq_0 : d ≠ 0
  geometric : (a 3) * (a 15) = (a 5)^2  -- a₃, a₅, a₁₅ form a geometric sequence
  a_1 : a 1 = 3
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The theorem stating the maximum value of S_n -/
theorem max_S_n (seq : ArithmeticSequence) :
  ∃ (max : ℚ), max = 4 ∧ ∀ (n : ℕ), S_n seq n ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_l19_1991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l19_1988

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 3*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (3/4, 0)

-- Define the line passing through the focus with inclination 30°
noncomputable def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 3/4)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B → Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l19_1988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_27m_squared_eq_300_l19_1936

/-- A prime number other than 2 -/
noncomputable def p : ℕ := sorry

/-- An even integer with exactly 13 positive divisors -/
noncomputable def m : ℕ := sorry

/-- The exponent of 2 in the prime factorization of m -/
noncomputable def a : ℕ := sorry

/-- The exponent of p in the prime factorization of m -/
noncomputable def b : ℕ := sorry

axiom m_form : m = 2^a * p^b

axiom m_even : Even m

axiom m_divisors : (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 13

axiom p_prime : Nat.Prime p

axiom p_not_two : p ≠ 2

/-- The number of positive divisors of 27m^2 -/
noncomputable def divisors_27m_squared : ℕ := 
  (Finset.filter (· ∣ (27 * m^2)) (Finset.range ((27 * m^2) + 1))).card

theorem divisors_27m_squared_eq_300 : divisors_27m_squared = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_27m_squared_eq_300_l19_1936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_triangle_area_l19_1912

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 2 * (Real.cos x) ^ 2 - 2

/-- Theorem stating the intervals where f(x) is increasing -/
theorem f_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (-Real.pi/3 + k * Real.pi) (Real.pi/6 + k * Real.pi)) := by
  sorry

/-- Theorem about the area of triangle ABC with given conditions -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 3 →
  2 * f C = -1 →
  2 * Real.sin A = Real.sin B →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_triangle_area_l19_1912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_on_circle_l19_1946

-- Define the circle
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Vector from A to O
def AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)

-- Vector from A to P
def AP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)

-- Theorem statement
theorem max_dot_product_on_circle :
  ∀ P : ℝ × ℝ, is_on_circle P → dot_product AO (AP P) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_on_circle_l19_1946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l19_1910

-- Define the function g as noncomputable due to dependency on Real.sqrt
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x - 5)

-- State the theorem
theorem smallest_x_in_domain_of_g_composed : 
  ∀ x : ℝ, (∀ y, y ∈ Set.range (g ∘ g) → x ≤ y) ↔ x = 30 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l19_1910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_balance_proof_l19_1969

/-- Proves that the initial balance was $150 given the conditions of the problem -/
theorem initial_balance_proof (deposit : ℝ) (new_balance : ℝ) (initial_balance : ℝ) : 
  deposit = 50 ∧ 
  deposit = (1/4) * new_balance ∧
  new_balance = deposit + initial_balance →
  initial_balance = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_balance_proof_l19_1969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_l19_1966

/-- A palindrome is a positive integer that reads the same from left to right and from right to left. -/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- The number of palindromes with a given number of digits. -/
def NumPalindromes (digits : ℕ) : ℕ := sorry

/-- There are 9 two-digit palindromes. -/
axiom two_digit_palindromes : NumPalindromes 2 = 9

/-- There are 90 three-digit palindromes. -/
axiom three_digit_palindromes : NumPalindromes 3 = 90

theorem palindrome_count :
  (NumPalindromes 4 = 90) ∧
  (∀ n : ℕ, n > 0 → NumPalindromes (2 * n + 1) = 9 * (10 ^ n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_l19_1966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l19_1987

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 * Real.sqrt 2 →
  b = Real.sqrt 6 →
  Real.cos A = 1 / 3 →
  let R := a / (2 * Real.sin A)
  R = 3 / 2 ∧ c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l19_1987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_total_miles_l19_1984

/-- Jerry's daily miles over five days -/
def jerry_miles : Fin 5 → ℕ
  | 0 => 15  -- Monday
  | 1 => 18  -- Tuesday
  | 2 => 25  -- Wednesday
  | 3 => 12  -- Thursday
  | 4 => 10  -- Friday

/-- Theorem: The sum of Jerry's daily miles over five days equals 80 -/
theorem jerry_total_miles :
  (Finset.univ : Finset (Fin 5)).sum jerry_miles = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_total_miles_l19_1984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_area_theorem_l19_1962

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line -/
structure Line where
  -- We'll represent a line by its y-intercept and slope
  yIntercept : ℝ
  slope : ℝ

noncomputable def isExternallyTangent (c1 c2 : Circle) : Prop :=
  (c1.radius + c2.radius)^2 = 
    (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2

noncomputable def isTangentToLine (c : Circle) (l : Line) : Prop :=
  -- A simplified condition for tangency
  abs (c.center.2 - (l.slope * c.center.1 + l.yIntercept)) = c.radius

def pointBetween (p1 p2 p3 : ℝ × ℝ) : Prop :=
  p1.1 ≤ p2.1 ∧ p2.1 ≤ p3.1 ∧ p1.2 = p2.2 ∧ p2.2 = p3.2

noncomputable def areaOfTriangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

theorem circle_tangency_area_theorem (P Q R : Circle) (l : Line) 
    (P' Q' R' : ℝ × ℝ) :
  P.radius = 2 →
  Q.radius = 3 →
  R.radius = 4 →
  isTangentToLine P l →
  isTangentToLine Q l →
  isTangentToLine R l →
  pointBetween P' Q' R' →
  isExternallyTangent Q P →
  isExternallyTangent Q R →
  areaOfTriangle P.center Q.center R.center = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_area_theorem_l19_1962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interchanges_equals_n_l19_1944

/-- Represents the state of coin distribution among kids -/
structure CoinDistribution where
  n : ℕ
  coins : List ℕ
  deriving Repr

/-- Checks if an interchange is possible given a coin distribution -/
def canInterchange (dist : CoinDistribution) : Bool :=
  ∃ k ∈ dist.coins, 2 * k ≥ List.sum dist.coins

/-- Performs an interchange on the given coin distribution -/
def interchange (dist : CoinDistribution) : CoinDistribution :=
  sorry

/-- Counts the maximum number of consecutive interchanges possible -/
def maxInterchanges (dist : CoinDistribution) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of interchanges is n -/
theorem max_interchanges_equals_n (n : ℕ) :
  ∀ (dist : CoinDistribution), 
    dist.n = n → 
    List.sum dist.coins = 2^n → 
    maxInterchanges dist = n :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interchanges_equals_n_l19_1944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l19_1963

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/6) + Real.cos x ^ 4 - Real.sin x ^ 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), -Real.pi/12 ≤ x ∧ x ≤ Real.pi/6 → f x ≤ Real.sqrt 3 + 1/2) ∧
  (∀ (x : ℝ), -Real.pi/12 ≤ x ∧ x ≤ Real.pi/6 → f x ≥ (Real.sqrt 3 + 1)/2) ∧
  f (-Real.pi/12) = (Real.sqrt 3 + 1)/2 ∧
  f (Real.pi/6) = Real.sqrt 3 + 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l19_1963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_less_than_diff_reverse_l19_1915

/-- A function that reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has an odd number of digits -/
def hasOddDigits (n : ℕ) : Prop := sorry

theorem sqrt_less_than_diff_reverse (N : ℕ) (h : N > reverseDigits N) :
  (hasOddDigits N → Real.sqrt (N : ℝ) < (N : ℝ) - (reverseDigits N : ℝ)) ∧
  (¬hasOddDigits N → ∃ (k : ℕ), N < 81 * 10^(2*k - 2) →
    Real.sqrt (N : ℝ) < (N : ℝ) - (reverseDigits N : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_less_than_diff_reverse_l19_1915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_increasing_interval_and_max_k_l19_1935

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 2 * a * Real.exp x + 2 * a^2

noncomputable def g (a : ℝ) (k : ℕ) (x : ℝ) : ℝ := 2 * a * Real.log x - (Real.log x)^2 + k^2 / 8

theorem functions_increasing_interval_and_max_k :
  (∀ a : ℝ, StrictMono (f a) ↔ StrictMono (g a 4)) ∧
  (∃ k : ℕ, k > 0 ∧ k ≤ 4 ∧ ∀ a : ℝ, ∀ x : ℝ, x > 0 → f a x > g a k x) ∧
  (∀ k : ℕ, k > 4 → ∃ a : ℝ, ∃ x : ℝ, x > 0 ∧ f a x ≤ g a k x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_increasing_interval_and_max_k_l19_1935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_difference_l19_1992

/-- Given vectors a and b in a 2D plane satisfying certain conditions, 
    prove that the magnitude of a - 2b is √21. -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  ‖b‖ = 2 * ‖a‖ ∧ ‖a‖ = 1 ∧ 
  a.1 * b.1 + a.2 * b.2 = -‖a‖ * ‖b‖ / 2 →
  ‖a - 2 • b‖ = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_difference_l19_1992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_period_f_monotone_interval_l19_1924

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) - 1/2

-- Statement 1
theorem f_at_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = Real.sqrt 2/2) :
  f α = 1/2 := by sorry

-- Statement 2
theorem f_period : 
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧ T = π := by sorry

-- Statement 3
theorem f_monotone_interval (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π - 3 * π / 8) (k * π + π / 8)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_period_f_monotone_interval_l19_1924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_l19_1932

-- Define the slopes of the lines as functions of a
noncomputable def slope_L1 (a : ℝ) : ℝ := a / (a - 1)
noncomputable def slope_L2 (a : ℝ) : ℝ := (1 - a) / (2 * a + 3)

-- Define the perpendicularity condition
def are_perpendicular (a : ℝ) : Prop :=
  (a = 1) ∨ 
  (a = -3) ∨
  (a ≠ 1 ∧ a ≠ -3/2 ∧ slope_L1 a * slope_L2 a = -1)

-- Theorem statement
theorem perpendicular_lines_condition :
  ∀ a : ℝ, are_perpendicular a ↔ (a = 1 ∨ a = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_l19_1932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomial_product_l19_1958

theorem constant_term_of_polynomial_product (p q : Polynomial ℝ) : 
  Polynomial.Monic p → Polynomial.Monic q → 
  p.degree = 3 → q.degree = 3 →
  (∃ (c : ℝ), c > 0 ∧ p.coeff 0 = c ∧ q.coeff 0 = c) →
  (∃ (a : ℝ), p.coeff 2 = a ∧ q.coeff 2 = a) →
  p * q = X^6 + 2•X^5 + 3•X^4 + 4•X^3 + 3•X^2 + 2•X + 9 →
  p.coeff 0 = 3 ∧ q.coeff 0 = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomial_product_l19_1958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l19_1981

/-- IsArithmeticSequence predicate -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : IsArithmeticSequence a) 
  (h_sum : a 2 + a 8 = 6) : a 4 + a 5 + a 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l19_1981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_argument_of_z_approx_pi_over_4_l19_1990

-- Define θ
noncomputable def θ : Real := Real.arctan (5/12)

-- Define the complex number z
noncomputable def z : ℂ := (Complex.cos (2*θ) + Complex.I * Complex.sin (2*θ)) / (239 + Complex.I)

-- Theorem statement
theorem argument_of_z_approx_pi_over_4 :
  ∃ (ε : Real), ε > 0 ∧ Complex.abs (Complex.arg z - π/4) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_argument_of_z_approx_pi_over_4_l19_1990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_circle_l19_1960

/-- A convex figure in 2D space -/
structure ConvexFigure where
  -- Add necessary fields and properties to define a convex figure
  -- This is a simplified representation
  is_convex : Bool

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  -- Add necessary fields to define an equilateral triangle
  side_length : ℝ
  is_equilateral : side_length = 1

/-- Predicate to check if a translated triangle lies on the boundary of the figure -/
def translated_triangle_on_boundary (F : ConvexFigure) (T : EquilateralTriangle) (translation : ℝ × ℝ) : Prop :=
  -- Define the condition for a translated triangle to lie on the boundary
  sorry

/-- The property that any equilateral triangle can be translated to lie on the boundary -/
def has_triangle_property (F : ConvexFigure) : Prop :=
  ∀ (T : EquilateralTriangle), ∃ (translation : ℝ × ℝ), 
    (translated_triangle_on_boundary F T translation)

/-- Predicate to check if a figure is a circle -/
def is_circle (F : ConvexFigure) : Prop :=
  -- Add necessary conditions for a figure to be a circle
  sorry

/-- The main theorem -/
theorem not_necessarily_circle : 
  ∃ (F : ConvexFigure), has_triangle_property F ∧ ¬(is_circle F) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_circle_l19_1960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_mod_100_l19_1948

/-- Definition of the triangle function -/
def triangle (n k : ℕ) : ℕ :=
  match n, k with
  | 0, _ => 1
  | n+1, 0 => 1
  | n+1, k+1 => triangle n k + triangle n (k+1)

/-- Definition of f(n) as the sum of the n-th row -/
def f (n : ℕ) : ℕ := Finset.sum (Finset.range (n+1)) (λ k => triangle n k)

/-- The main theorem to prove -/
theorem f_100_mod_100 : f 100 % 100 = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_mod_100_l19_1948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sequences_same_elements_l19_1999

/-- A p-arithmetic Fibonacci sequence -/
def pArithmeticFibonacci (p : ℕ → ℕ → ℕ) : ℕ → ℕ := sorry

/-- The sequence of ratios derived from a p-arithmetic Fibonacci sequence -/
def ratioSequence (v : ℕ → ℕ) : ℕ → ℚ :=
  λ n => if n = 0 then 0 else (v n : ℚ) / (v (n-1) : ℚ)

/-- Two ratio sequences have a common element -/
def hasCommonElement (t1 t2 : ℕ → ℚ) : Prop :=
  ∃ k l, t1 k = t2 l

/-- Two sequences consist of the same elements -/
def sameElements (t1 t2 : ℕ → ℚ) : Prop :=
  ∀ x, (∃ n, t1 n = x) ↔ (∃ m, t2 m = x)

theorem ratio_sequences_same_elements
  (p1 p2 : ℕ → ℕ → ℕ)
  (v1 : ℕ → ℕ)
  (v2 : ℕ → ℕ)
  (h1 : v1 = pArithmeticFibonacci p1)
  (h2 : v2 = pArithmeticFibonacci p2)
  (t1 : ℕ → ℚ)
  (t2 : ℕ → ℚ)
  (ht1 : t1 = ratioSequence v1)
  (ht2 : t2 = ratioSequence v2)
  (h : hasCommonElement t1 t2) :
  sameElements t1 t2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sequences_same_elements_l19_1999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_half_l19_1900

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := Real.exp (x - a) - Real.log (x + a) - 1

-- State the theorem
theorem minimum_value_implies_a_half (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f a x ≥ 0) ∧ (∃ x > 0, f a x = 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_half_l19_1900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_9_l19_1913

-- Define the revenue function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 13.5 - (1/30) * x^2
  else if x > 10 then 168/x - 2000/(3*x^2)
  else 0

-- Define the profit function y(x)
noncomputable def y (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then x * g x - 20 - 5.4 * x
  else if x > 10 then x * g x - 20 - 5.4 * x
  else 0

-- Theorem statement
theorem max_profit_at_9 :
  ∃ (max_x : ℝ), max_x = 9 ∧
  ∀ (x : ℝ), x > 0 → y x ≤ y max_x ∧
  y max_x = 28.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_9_l19_1913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l19_1980

theorem root_difference (a : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : x₃ ≠ x₄)
  (h3 : x₁^2 + a*x₁ + 2 = x₁)
  (h4 : x₂^2 + a*x₂ + 2 = x₂)
  (h5 : (x₃ - a)^2 + a*(x₃ - a) + 2 = x₃)
  (h6 : (x₄ - a)^2 + a*(x₄ - a) + 2 = x₄)
  (h7 : x₃ - x₁ = 3*(x₄ - x₂)) :
  x₄ - x₂ = a/2 := by
  sorry

#check root_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l19_1980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l19_1965

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 4 * x - 1
noncomputable def g (x : ℝ) : ℝ := 3 * x + 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x - 7) / 12

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x := by
  sorry

#check h_inverse_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l19_1965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_seven_sixths_l19_1901

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := x
def curve3 (x : ℝ) : ℝ := 2*x

-- Define the area of the plane figure
noncomputable def area_of_figure : ℝ :=
  ∫ x in Set.Icc 0 1, (curve3 x - curve2 x) +
  ∫ x in Set.Icc 1 2, (curve3 x - curve1 x)

-- Theorem statement
theorem area_is_seven_sixths : area_of_figure = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_seven_sixths_l19_1901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_ln_l19_1953

theorem tangent_to_ln (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = deriv Real.log x) → 
  k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_ln_l19_1953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_trigonometric_ratios_l19_1909

-- Part 1
theorem trigonometric_expression_equality :
  (Real.sqrt 3 * Real.sin (-20 * π / 3)) / Real.tan (11 * π / 3) -
  Real.cos (13 * π / 4) * Real.tan (-37 * π / 4) =
  (Real.sqrt 3 - Real.sqrt 2) / 2 := by sorry

-- Part 2
theorem trigonometric_ratios (α : Real) (h : Real.tan α = 4/3) :
  ((Real.sin (2*α) + 2 * Real.sin α * Real.cos α) /
   (2 * Real.cos (2*α) - Real.sin (2*α)) = -24/19) ∧
  (Real.sin α * Real.cos α = 12/25) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_trigonometric_ratios_l19_1909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_2x1_minus_x2_l19_1905

open Real

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 3)

noncomputable def g (x : ℝ) : ℝ := 3 * sin (2 * x + 2 * π / 3) + 1

def domain : Set ℝ := Set.Icc (-3 * π / 2) (3 * π / 2)

theorem max_value_of_2x1_minus_x2 :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ domain ∧ x₂ ∈ domain ∧ g x₁ * g x₂ = 16 ∧
  (∀ (y₁ y₂ : ℝ), y₁ ∈ domain → y₂ ∈ domain → g y₁ * g y₂ = 16 →
    2 * x₁ - x₂ ≥ 2 * y₁ - y₂) ∧
  2 * x₁ - x₂ = 35 * π / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_2x1_minus_x2_l19_1905
