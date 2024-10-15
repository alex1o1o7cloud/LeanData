import Mathlib

namespace NUMINAMATH_CALUDE_product_mod_seventeen_l4109_410975

theorem product_mod_seventeen :
  (5007 * 5008 * 5009 * 5010 * 5011) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l4109_410975


namespace NUMINAMATH_CALUDE_interior_point_is_center_of_gravity_l4109_410994

/-- A lattice point represented by its x and y coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle represented by its three vertices -/
structure LatticeTriangle where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint

/-- Checks if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Checks if a point is in the interior of a triangle -/
def isInterior (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Calculates the center of gravity of a triangle -/
def centerOfGravity (t : LatticeTriangle) : LatticePoint := sorry

/-- The main theorem -/
theorem interior_point_is_center_of_gravity 
  (t : LatticeTriangle) 
  (h1 : t.v1 = ⟨0, 0⟩) 
  (h2 : ∀ p : LatticePoint, p ≠ t.v1 ∧ p ≠ t.v2 ∧ p ≠ t.v3 → ¬isOnBoundary p t) 
  (p : LatticePoint) 
  (h3 : isInterior p t) 
  (h4 : ∀ q : LatticePoint, q ≠ p → ¬isInterior q t) : 
  p = centerOfGravity t := by
  sorry

end NUMINAMATH_CALUDE_interior_point_is_center_of_gravity_l4109_410994


namespace NUMINAMATH_CALUDE_angle_equivalence_l4109_410941

theorem angle_equivalence (α θ : Real) (h1 : α = 1690) (h2 : 0 < θ ∧ θ < 360) 
  (h3 : ∃ k : Int, α = k * 360 + θ) : θ = 250 := by
  sorry

end NUMINAMATH_CALUDE_angle_equivalence_l4109_410941


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4109_410911

/-- Given a hyperbola and a parabola with specific conditions, prove that the standard equation of the hyperbola is x²/16 - y²/16 = 1 -/
theorem hyperbola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, y^2 = 2*p*x) →
  (∃ x : ℝ, |x + a| = 3) →
  (b*(-1) + a*1 = 0) →
  (a = 4 ∧ b = 4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4109_410911


namespace NUMINAMATH_CALUDE_distance_to_origin_l4109_410939

/-- Given that point A has coordinates (√3, 2, 5) and its projection on the x-axis is (√3, 0, 0),
    prove that the distance from A to the origin is 4√2. -/
theorem distance_to_origin (A : ℝ × ℝ × ℝ) (h : A = (Real.sqrt 3, 2, 5)) :
  Real.sqrt ((Real.sqrt 3)^2 + 2^2 + 5^2) = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_l4109_410939


namespace NUMINAMATH_CALUDE_aaron_final_position_l4109_410949

/-- Represents a point on the coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Represents Aaron's state -/
structure AaronState where
  position : Point
  direction : Direction
  steps : Nat

/-- Defines the rules for Aaron's movement -/
def move (state : AaronState) : AaronState :=
  sorry

/-- Theorem stating Aaron's final position after 100 steps -/
theorem aaron_final_position :
  (move^[100] { position := { x := 0, y := 0 }, direction := Direction.East, steps := 0 }).position = { x := 10, y := 0 } :=
sorry

end NUMINAMATH_CALUDE_aaron_final_position_l4109_410949


namespace NUMINAMATH_CALUDE_complex_symmetry_quotient_l4109_410971

theorem complex_symmetry_quotient (z₁ z₂ : ℂ) : 
  (z₁.im = -z₂.im) → (z₁.re = z₂.re) → z₁ = 1 + I → z₁ / z₂ = I := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_quotient_l4109_410971


namespace NUMINAMATH_CALUDE_prob_through_C_value_l4109_410913

/-- Represents a grid of city blocks -/
structure CityGrid where
  width : ℕ
  height : ℕ

/-- Represents a position on the grid -/
structure Position where
  x : ℕ
  y : ℕ

/-- Probability of moving east at an intersection -/
def prob_east : ℚ := 2/3

/-- Probability of moving south at an intersection -/
def prob_south : ℚ := 1/3

/-- The starting position A -/
def start_pos : Position := ⟨0, 0⟩

/-- The ending position D -/
def end_pos : Position := ⟨5, 5⟩

/-- The intermediate position C -/
def mid_pos : Position := ⟨3, 2⟩

/-- Calculate the probability of reaching position C when moving from A to D -/
def prob_through_C (grid : CityGrid) (A B C : Position) : ℚ := sorry

/-- Theorem stating that the probability of passing through C is 25/63 -/
theorem prob_through_C_value :
  prob_through_C ⟨5, 5⟩ start_pos end_pos mid_pos = 25/63 := by sorry

end NUMINAMATH_CALUDE_prob_through_C_value_l4109_410913


namespace NUMINAMATH_CALUDE_boundary_length_square_l4109_410955

/-- The length of the boundary formed by semi-circle arcs and line segments on a square with area 144 square units, where each side is divided into four equal parts -/
theorem boundary_length_square (square_area : ℝ) (side_divisions : ℕ) : square_area = 144 ∧ side_divisions = 4 → ∃ (boundary_length : ℝ), boundary_length = 12 * Real.pi + 24 := by
  sorry

end NUMINAMATH_CALUDE_boundary_length_square_l4109_410955


namespace NUMINAMATH_CALUDE_platform_length_l4109_410914

/-- Given a train of length 300 meters that takes 18 seconds to cross a post
    and 39 seconds to cross a platform, the length of the platform is 350 meters. -/
theorem platform_length (train_length : ℝ) (post_time : ℝ) (platform_time : ℝ) :
  train_length = 300 →
  post_time = 18 →
  platform_time = 39 →
  ∃ (platform_length : ℝ),
    platform_length = 350 ∧
    (train_length / post_time) * platform_time = train_length + platform_length :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l4109_410914


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l4109_410940

/-- A line ax + by + c = 0 is parallel to the x-axis if and only if a = 0, b ≠ 0, and c ≠ 0 -/
def parallel_to_x_axis (a b c : ℝ) : Prop :=
  a = 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of the line in question -/
def line_equation (a x y : ℝ) : Prop :=
  (6 * a^2 - a - 2) * x + (3 * a^2 - 5 * a + 2) * y + a - 1 = 0

/-- The theorem to be proved -/
theorem line_parallel_to_x_axis (a : ℝ) :
  (∃ x y, line_equation a x y) ∧ 
  parallel_to_x_axis (6 * a^2 - a - 2) (3 * a^2 - 5 * a + 2) (a - 1) →
  a = -1/2 :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l4109_410940


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l4109_410988

/-- The equation of a circle symmetric to another circle with respect to a line. -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = 7 ∧ x₀ + y₀ = 4) → 
  (x^2 + y^2 = 7) := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l4109_410988


namespace NUMINAMATH_CALUDE_train_speed_l4109_410985

/-- Calculates the speed of a train given its composition and the time it takes to cross a bridge. -/
theorem train_speed (num_carriages : ℕ) (carriage_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  num_carriages = 24 →
  carriage_length = 60 →
  bridge_length = 3.5 →
  crossing_time = 5 / 60 →
  let total_train_length := (num_carriages + 1 : ℝ) * carriage_length
  let total_distance := total_train_length / 1000 + bridge_length
  let speed := total_distance / crossing_time
  speed = 60 := by
    sorry


end NUMINAMATH_CALUDE_train_speed_l4109_410985


namespace NUMINAMATH_CALUDE_williams_land_ratio_l4109_410918

/-- The ratio of an individual's tax payment to the total tax collected equals the ratio of their taxable land to the total taxable land -/
axiom tax_ratio_equals_land_ratio {total_tax individual_tax total_land individual_land : ℚ} :
  individual_tax / total_tax = individual_land / total_land

/-- Given the total farm tax and an individual's farm tax, prove that the ratio of the individual's
    taxable land to the total taxable land is 1/8 -/
theorem williams_land_ratio (total_tax individual_tax : ℚ)
    (h1 : total_tax = 3840)
    (h2 : individual_tax = 480) :
    ∃ (total_land individual_land : ℚ),
      individual_land / total_land = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_williams_land_ratio_l4109_410918


namespace NUMINAMATH_CALUDE_river_road_cars_l4109_410943

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 17 →  -- ratio of buses to cars is 1:17
  cars = buses + 80 →            -- 80 fewer buses than cars
  cars = 85 :=                   -- prove that there are 85 cars
by sorry

end NUMINAMATH_CALUDE_river_road_cars_l4109_410943


namespace NUMINAMATH_CALUDE_min_sum_abcd_l4109_410960

theorem min_sum_abcd (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) :
  ∃ (m : ℕ), (∀ (a' b' c' d' : ℕ), a' * b' + b' * c' + c' * d' + d' * a' = 707 →
    a' + b' + c' + d' ≥ m) ∧ a + b + c + d = m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abcd_l4109_410960


namespace NUMINAMATH_CALUDE_checkered_square_covering_l4109_410992

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a 2x2 checkered square -/
def CheckeredSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- Represents a 1x1 square -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- Checks if a point is inside a triangle -/
def isInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Checks if a set is completely covered by a triangle -/
def isCompletelyCovered (s : Set (ℝ × ℝ)) (t : Triangle) : Prop :=
  ∀ p ∈ s, isInTriangle p t

/-- Checks if a set can be placed inside a triangle -/
def canBePlacedInside (s : Set (ℝ × ℝ)) (t : Triangle) : Prop := sorry

theorem checkered_square_covering (t1 t2 : Triangle) 
  (h : ∀ p ∈ CheckeredSquare, isInTriangle p t1 ∨ isInTriangle p t2) :
  (∃ cell : Set (ℝ × ℝ), cell ⊆ CheckeredSquare ∧ 
    ¬(isCompletelyCovered cell t1 ∨ isCompletelyCovered cell t2)) ∧
  (canBePlacedInside UnitSquare t1 ∨ canBePlacedInside UnitSquare t2) := by
  sorry

end NUMINAMATH_CALUDE_checkered_square_covering_l4109_410992


namespace NUMINAMATH_CALUDE_speed_difference_l4109_410921

/-- The speed difference between a cyclist and a car -/
theorem speed_difference (cyclist_distance car_distance : ℝ) (time : ℝ) 
  (h_cyclist : cyclist_distance = 88)
  (h_car : car_distance = 48)
  (h_time : time = 8)
  (h_time_pos : time > 0) :
  cyclist_distance / time - car_distance / time = 5 := by
sorry

end NUMINAMATH_CALUDE_speed_difference_l4109_410921


namespace NUMINAMATH_CALUDE_convex_hull_of_37gons_has_at_least_37_sides_l4109_410998

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of regular polygons -/
def SetOfPolygons (n : ℕ) := Set (RegularPolygon n)

/-- The convex hull of a set of points in ℝ² -/
def ConvexHull (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The number of sides in the convex hull of a set of points -/
def NumSides (S : Set (ℝ × ℝ)) : ℕ := sorry

/-- The vertices of all polygons in a set -/
def AllVertices (S : SetOfPolygons n) : Set (ℝ × ℝ) := sorry

/-- Theorem: The convex hull of any set of regular 37-gons has at least 37 sides -/
theorem convex_hull_of_37gons_has_at_least_37_sides (S : SetOfPolygons 37) :
  NumSides (ConvexHull (AllVertices S)) ≥ 37 := by sorry

end NUMINAMATH_CALUDE_convex_hull_of_37gons_has_at_least_37_sides_l4109_410998


namespace NUMINAMATH_CALUDE_range_of_S_l4109_410990

theorem range_of_S (x₁ x₂ x₃ x₄ : ℝ) 
  (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0) 
  (h_sum : x₁ + x₂ - x₃ + x₄ = 1) : 
  let S := 1 - (x₁^4 + x₂^4 + x₃^4 + x₄^4) - 
    6 * (x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄)
  0 ≤ S ∧ S ≤ 3/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_S_l4109_410990


namespace NUMINAMATH_CALUDE_cut_cube_edge_count_l4109_410970

/-- A cube with corners cut off -/
structure CutCube where
  /-- The number of vertices in the original cube -/
  original_vertices : Nat
  /-- The number of edges in the original cube -/
  original_edges : Nat
  /-- The number of new edges created by each vertex cut -/
  new_edges_per_vertex : Nat

/-- The theorem stating that a cube with corners cut off has 56 edges -/
theorem cut_cube_edge_count (c : CutCube) 
  (h1 : c.original_vertices = 8)
  (h2 : c.original_edges = 12)
  (h3 : c.new_edges_per_vertex = 4) :
  c.new_edges_per_vertex * c.original_vertices + 2 * c.original_edges = 56 := by
  sorry

#check cut_cube_edge_count

end NUMINAMATH_CALUDE_cut_cube_edge_count_l4109_410970


namespace NUMINAMATH_CALUDE_geometry_propositions_l4109_410929

-- Define the concepts
def Plane : Type := sorry
def Line : Type := sorry
def perpendicular (a b : Plane) : Prop := sorry
def parallel (a b : Plane) : Prop := sorry
def passes_through (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line (p : Plane) : Line := sorry
def in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection_line (p q : Plane) : Line := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ (p q : Plane) (l : Line),
    passes_through l q ∧ l = perpendicular_line p → perpendicular p q

def proposition_2 : Prop :=
  ∀ (p q : Plane) (l m : Line),
    in_plane l p ∧ in_plane m p ∧ parallel l q ∧ parallel m q → parallel p q

def proposition_3 : Prop :=
  ∀ (p q : Plane) (l : Line),
    perpendicular p q ∧ in_plane l p ∧ ¬perpendicular l (intersection_line p q) →
    ¬perpendicular l q

def proposition_4 : Prop :=
  ∀ (p : Plane) (l m : Line),
    parallel l p ∧ parallel m p → parallel_lines l m

-- State the theorem
theorem geometry_propositions :
  proposition_1 ∧ proposition_3 ∧ ¬proposition_2 ∧ ¬proposition_4 := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l4109_410929


namespace NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l4109_410987

theorem factorization_of_2a2_minus_8b2 (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l4109_410987


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l4109_410980

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_q_positive : q > 0
  h_q_not_one : q ≠ 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- 
For a geometric sequence with positive terms and common ratio q where q > 0 and q ≠ 1,
a_n + a_{n+3} > a_{n+1} + a_{n+2} for all n
-/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  ∀ n, seq.a n + seq.a (n + 3) > seq.a (n + 1) + seq.a (n + 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l4109_410980


namespace NUMINAMATH_CALUDE_exactly_one_valid_set_l4109_410947

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A set of consecutive integers is valid if it contains at least two integers and sums to 18 -/
def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 2 ∧ sum_consecutive a n = 18

theorem exactly_one_valid_set :
  ∃! p : ℕ × ℕ, is_valid_set p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_valid_set_l4109_410947


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l4109_410934

/-- The area of a square with adjacent vertices at (1, -2) and (4, 1) on a Cartesian coordinate plane is 18. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (4, 1)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l4109_410934


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l4109_410969

theorem shaded_area_percentage (side_length : ℝ) (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℝ) :
  side_length = 6 ∧
  rect1_width = 2 ∧ rect1_height = 2 ∧
  rect2_width = 4 ∧ rect2_height = 1 ∧
  rect3_width = 6 ∧ rect3_height = 1 →
  (rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height) / (side_length * side_length) = 22 / 36 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l4109_410969


namespace NUMINAMATH_CALUDE_area_of_sliced_quadrilateral_l4109_410916

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a quadrilateral formed by slicing a rectangular prism -/
structure SlicedQuadrilateral where
  prism : RectangularPrism
  A : Point3D -- vertex
  B : Point3D -- midpoint on length edge
  C : Point3D -- midpoint on width edge
  D : Point3D -- midpoint on height edge

/-- Calculate the area of the sliced quadrilateral -/
def areaOfSlicedQuadrilateral (quad : SlicedQuadrilateral) : ℝ :=
  sorry -- Placeholder for the actual calculation

/-- Theorem: The area of the sliced quadrilateral is 1.5 square units -/
theorem area_of_sliced_quadrilateral :
  let prism := RectangularPrism.mk 2 3 4
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 1 0 0
  let C := Point3D.mk 0 1.5 0
  let D := Point3D.mk 0 0 2
  let quad := SlicedQuadrilateral.mk prism A B C D
  areaOfSlicedQuadrilateral quad = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_area_of_sliced_quadrilateral_l4109_410916


namespace NUMINAMATH_CALUDE_parabolas_similar_l4109_410991

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := x^2
def parabola2 (x : ℝ) : ℝ := 2 * x^2

-- Define a homothety transformation
def homothety (scale : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (scale * p.1, scale * p.2)

-- Theorem statement
theorem parabolas_similar :
  ∀ x : ℝ, homothety 2 (x, parabola2 x) = (2*x, parabola1 (2*x)) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_similar_l4109_410991


namespace NUMINAMATH_CALUDE_intersection_A_B_l4109_410950

def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4109_410950


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l4109_410904

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0) : 
  x₁^3 + x₂^3 = 18 ∧ x₂/x₁ + x₁/x₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l4109_410904


namespace NUMINAMATH_CALUDE_rabbits_ate_four_potatoes_l4109_410901

/-- The number of potatoes eaten by rabbits -/
def potatoes_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the number of potatoes eaten by rabbits is 4 -/
theorem rabbits_ate_four_potatoes (h1 : initial = 7) (h2 : remaining = 3) :
  potatoes_eaten initial remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_ate_four_potatoes_l4109_410901


namespace NUMINAMATH_CALUDE_ribbon_length_l4109_410948

/-- The original length of two ribbons with specific cutting conditions -/
theorem ribbon_length : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (x - 12 = 2 * (x - 18)) ∧ 
  (x = 24) := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_l4109_410948


namespace NUMINAMATH_CALUDE_existence_of_odd_digit_multiple_of_power_of_five_l4109_410919

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d < 10

def all_digits_odd (x : ℕ) : Prop :=
  ∀ d, d ∈ x.digits 10 → is_odd_digit d

theorem existence_of_odd_digit_multiple_of_power_of_five (n : ℕ) :
  n > 0 →
  ∃ x : ℕ,
    (x.digits 10).length = n ∧
    all_digits_odd x ∧
    x % (5^n) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_odd_digit_multiple_of_power_of_five_l4109_410919


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4109_410932

theorem trigonometric_identity : 
  Real.sin (-1200 * π / 180) * Real.cos (1290 * π / 180) + 
  Real.cos (-1020 * π / 180) * Real.sin (-1050 * π / 180) + 
  Real.tan (945 * π / 180) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4109_410932


namespace NUMINAMATH_CALUDE_circle_area_increase_l4109_410937

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.12 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  new_area = 1.2544 * original_area := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l4109_410937


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l4109_410908

/-- Represents the configuration of rectangles around a square -/
structure SquareWithRectangles where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The conditions of the problem -/
def problem_conditions (config : SquareWithRectangles) : Prop :=
  -- The area of the outer square is 9 times that of the inner square
  (config.inner_square_side + 2 * config.rectangle_short_side) ^ 2 = 9 * config.inner_square_side ^ 2 ∧
  -- The outer square's side length is composed of the inner square and two short sides of rectangles
  config.inner_square_side + 2 * config.rectangle_short_side = 
    config.rectangle_long_side + config.rectangle_short_side

/-- The theorem to prove -/
theorem rectangle_ratio_is_two (config : SquareWithRectangles) 
  (h : problem_conditions config) : 
  config.rectangle_long_side / config.rectangle_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l4109_410908


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4109_410963

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4109_410963


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l4109_410968

theorem smallest_six_digit_divisible_by_111 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 → n ≥ 100011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l4109_410968


namespace NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l4109_410993

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem max_k_for_f_geq_kx :
  ∃ (k : ℝ), k = 1 ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ k * x) ∧
  (∀ k' : ℝ, k' > k → ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x < k' * x) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l4109_410993


namespace NUMINAMATH_CALUDE_proposition_p_is_false_l4109_410977

theorem proposition_p_is_false : ¬(∀ x : ℝ, 2 * x^2 + 2 * x + (1/2 : ℝ) < 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_is_false_l4109_410977


namespace NUMINAMATH_CALUDE_consecutive_cubes_inequality_l4109_410942

theorem consecutive_cubes_inequality (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cubes_inequality_l4109_410942


namespace NUMINAMATH_CALUDE_sequence_properties_l4109_410952

def a (n : ℕ) : ℚ := 3 - 2^n

theorem sequence_properties :
  (∀ n : ℕ, a (2*n) = 3 - 4^n) ∧ (a 2 / a 3 = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l4109_410952


namespace NUMINAMATH_CALUDE_subtract_fifteen_from_number_l4109_410957

theorem subtract_fifteen_from_number (x : ℝ) : x / 10 = 6 → x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fifteen_from_number_l4109_410957


namespace NUMINAMATH_CALUDE_new_student_weight_l4109_410912

/-- Theorem: Weight of the new student when average weight decreases --/
theorem new_student_weight
  (n : ℕ) -- number of students
  (w : ℕ) -- weight of the replaced student
  (d : ℕ) -- decrease in average weight
  (h1 : n = 8)
  (h2 : w = 86)
  (h3 : d = 5)
  : ∃ (new_weight : ℕ), 
    (n : ℝ) * d = w - new_weight ∧ new_weight = 46 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l4109_410912


namespace NUMINAMATH_CALUDE_triangle_ratio_sine_relation_l4109_410983

theorem triangle_ratio_sine_relation (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6 →
  (Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) + 
   Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) /
  Real.sin (Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_sine_relation_l4109_410983


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4109_410930

open Set
open Function
open Real

def f (x : ℝ) := 3 * x^2 - 12 * x + 9

theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = Iio 1 ∪ Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4109_410930


namespace NUMINAMATH_CALUDE_banana_theorem_l4109_410995

/-- The number of pounds of bananas purchased by a grocer -/
def banana_problem (buy_rate : ℚ) (sell_rate : ℚ) (total_profit : ℚ) : ℚ :=
  total_profit / (sell_rate - buy_rate)

theorem banana_theorem :
  banana_problem (1/6) (1/4) 11 = 132 := by
  sorry

#eval banana_problem (1/6) (1/4) 11

end NUMINAMATH_CALUDE_banana_theorem_l4109_410995


namespace NUMINAMATH_CALUDE_hexagon_walk_distance_l4109_410979

theorem hexagon_walk_distance (side_length : ℝ) (walk_distance : ℝ) (end_distance : ℝ) : 
  side_length = 3 →
  walk_distance = 11 →
  end_distance = 2 * Real.sqrt 3 →
  ∃ (x y : ℝ), x^2 + y^2 = end_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_walk_distance_l4109_410979


namespace NUMINAMATH_CALUDE_like_terms_sum_l4109_410926

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, a i j ≠ 0 ∧ b i j ≠ 0 → i = j

theorem like_terms_sum (m n : ℕ) :
  like_terms (fun i j => if i = 2 ∧ j = n then 7 else 0)
             (fun i j => if i = m ∧ j = 3 then -5 else 0) →
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_like_terms_sum_l4109_410926


namespace NUMINAMATH_CALUDE_vasya_late_l4109_410900

/-- Proves that Vasya did not arrive on time given the conditions of his journey -/
theorem vasya_late (v : ℝ) (h : v > 0) : 
  (10 / v + 16 / (v / 2.5) + 24 / (6 * v)) > (50 / v) := by
  sorry

#check vasya_late

end NUMINAMATH_CALUDE_vasya_late_l4109_410900


namespace NUMINAMATH_CALUDE_all_negative_k_purely_imaginary_roots_l4109_410906

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_eq (z k : ℂ) : Prop := 10 * z^2 - 3 * i * z - k = 0

-- Define a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem all_negative_k_purely_imaginary_roots :
  ∀ k : ℝ, k < 0 →
    ∃ z₁ z₂ : ℂ, quadratic_eq z₁ k ∧ quadratic_eq z₂ k ∧
               is_purely_imaginary z₁ ∧ is_purely_imaginary z₂ :=
sorry

end NUMINAMATH_CALUDE_all_negative_k_purely_imaginary_roots_l4109_410906


namespace NUMINAMATH_CALUDE_angle_A_measure_l4109_410961

/-- Given a geometric configuration with connected angles, prove that angle A measures 70°. -/
theorem angle_A_measure (B C D : ℝ) (hB : B = 120) (hC : C = 30) (hD : D = 110) : ∃ A : ℝ,
  A = 70 ∧ 
  A + B + C = 180 ∧  -- Sum of angles at a point
  A + C + (D - C) = 180  -- Sum of angles in the triangle formed by A, C, and the complement of D
  := by sorry

end NUMINAMATH_CALUDE_angle_A_measure_l4109_410961


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l4109_410984

/-- Represents a configuration of seven consecutively tangent circles between two parallel lines -/
structure CircleConfiguration where
  radii : Fin 7 → ℝ
  largest_radius : radii 6 = 24
  smallest_radius : radii 0 = 6
  tangent : ∀ i : Fin 6, radii i < radii (i.succ)

/-- The theorem stating that the radius of the fourth circle is 12√2 -/
theorem fourth_circle_radius (config : CircleConfiguration) : config.radii 3 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l4109_410984


namespace NUMINAMATH_CALUDE_bill_difference_l4109_410974

theorem bill_difference : 
  ∀ (sarah_bill linda_bill : ℝ),
  sarah_bill * 0.15 = 3 →
  linda_bill * 0.25 = 3 →
  sarah_bill - linda_bill = 8 := by
sorry

end NUMINAMATH_CALUDE_bill_difference_l4109_410974


namespace NUMINAMATH_CALUDE_consecutive_prints_probability_l4109_410909

/-- The number of pieces of art -/
def total_pieces : ℕ := 12

/-- The number of Escher prints -/
def escher_prints : ℕ := 3

/-- The number of Dali prints -/
def dali_prints : ℕ := 2

/-- The probability of Escher and Dali prints being consecutive -/
def consecutive_probability : ℚ := 336 / (Nat.factorial total_pieces)

/-- Theorem stating the probability of Escher and Dali prints being consecutive -/
theorem consecutive_prints_probability :
  consecutive_probability = 336 / (Nat.factorial total_pieces) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_prints_probability_l4109_410909


namespace NUMINAMATH_CALUDE_square_of_1307_l4109_410973

theorem square_of_1307 : 1307 * 1307 = 1709849 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1307_l4109_410973


namespace NUMINAMATH_CALUDE_students_shorter_than_yoongi_l4109_410966

theorem students_shorter_than_yoongi (total_students : ℕ) (taller_than_yoongi : ℕ) :
  total_students = 20 →
  taller_than_yoongi = 11 →
  total_students - (taller_than_yoongi + 1) = 8 := by
sorry

end NUMINAMATH_CALUDE_students_shorter_than_yoongi_l4109_410966


namespace NUMINAMATH_CALUDE_loop_condition_correct_l4109_410922

/-- A program for calculating the average of 20 numbers -/
structure AverageProgram where
  numbers : Fin 20 → ℝ
  loop_var : ℕ
  sum : ℝ

/-- The loop condition for the average calculation program -/
def loop_condition (p : AverageProgram) : Prop :=
  p.loop_var ≤ 20

/-- The correctness of the loop condition -/
theorem loop_condition_correct (p : AverageProgram) : 
  loop_condition p ↔ p.loop_var ≤ 20 := by sorry

end NUMINAMATH_CALUDE_loop_condition_correct_l4109_410922


namespace NUMINAMATH_CALUDE_stratified_sampling_used_l4109_410935

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | SamplingByLot
  | RandomNumberTable
  | Stratified

/-- Represents a population with two strata -/
structure Population where
  total : Nat
  stratum1 : Nat
  stratum2 : Nat
  h_sum : stratum1 + stratum2 = total

/-- Represents a sample from a population with two strata -/
structure Sample where
  total : Nat
  stratum1 : Nat
  stratum2 : Nat
  h_sum : stratum1 + stratum2 = total

/-- Determines if the sampling method is stratified based on population and sample data -/
def isStratifiedSampling (pop : Population) (sample : Sample) : Prop :=
  (pop.stratum1 : Rat) / pop.total = (sample.stratum1 : Rat) / sample.total ∧
  (pop.stratum2 : Rat) / pop.total = (sample.stratum2 : Rat) / sample.total

/-- The theorem to be proved -/
theorem stratified_sampling_used
  (pop : Population)
  (sample : Sample)
  (h_pop : pop = { total := 900, stratum1 := 500, stratum2 := 400, h_sum := rfl })
  (h_sample : sample = { total := 45, stratum1 := 25, stratum2 := 20, h_sum := rfl }) :
  isStratifiedSampling pop sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_used_l4109_410935


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4109_410907

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 + 15 - 17*x + 19*x^2 + 2*x^3 = 2*x^3 - x^2 - 11*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4109_410907


namespace NUMINAMATH_CALUDE_negation_equivalence_l4109_410923

theorem negation_equivalence (x : ℝ) : 
  ¬(x ≥ 1 → x^2 - 4*x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4*x + 2 < -1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4109_410923


namespace NUMINAMATH_CALUDE_lcm_of_4_9_10_27_l4109_410967

theorem lcm_of_4_9_10_27 : Nat.lcm 4 (Nat.lcm 9 (Nat.lcm 10 27)) = 540 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_9_10_27_l4109_410967


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l4109_410933

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : 1 / (a - 1) + 1 / (b - 1) = 1) : 
  ∀ x y, x > 1 → y > 1 → 1 / (x - 1) + 1 / (y - 1) = 1 → a + 4 * b ≤ x + 4 * y ∧ 
  ∃ a₀ b₀, a₀ > 1 ∧ b₀ > 1 ∧ 1 / (a₀ - 1) + 1 / (b₀ - 1) = 1 ∧ a₀ + 4 * b₀ = 14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l4109_410933


namespace NUMINAMATH_CALUDE_probability_is_point_six_l4109_410944

/-- Represents a company with a number of representatives -/
structure Company where
  representatives : ℕ

/-- Represents the meeting setup -/
structure Meeting where
  companies : Finset Company
  total_representatives : ℕ

/-- Calculates the probability of selecting 3 individuals from 3 different companies -/
def probability_three_different_companies (m : Meeting) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem probability_is_point_six (m : Meeting) 
  (h1 : m.companies.card = 4)
  (h2 : ∃ a ∈ m.companies, a.representatives = 2)
  (h3 : (m.companies.filter (λ c : Company => c.representatives = 1)).card = 3)
  (h4 : m.total_representatives = 5) :
  probability_three_different_companies m = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_probability_is_point_six_l4109_410944


namespace NUMINAMATH_CALUDE_emmy_and_rosa_ipods_l4109_410903

/-- 
Given that Emmy originally had 14 iPods, lost 6, and has twice as many as Rosa,
prove that Emmy and Rosa have 12 iPods together.
-/
theorem emmy_and_rosa_ipods :
  ∀ (emmy_original emmy_lost emmy_current rosa : ℕ),
  emmy_original = 14 →
  emmy_lost = 6 →
  emmy_current = emmy_original - emmy_lost →
  emmy_current = 2 * rosa →
  emmy_current + rosa = 12 := by
  sorry

end NUMINAMATH_CALUDE_emmy_and_rosa_ipods_l4109_410903


namespace NUMINAMATH_CALUDE_probability_small_area_is_two_thirds_l4109_410927

/-- A right triangle XYZ with vertices X=(0,8), Y=(0,0), Z=(10,0) -/
structure RightTriangle where
  X : ℝ × ℝ := (0, 8)
  Y : ℝ × ℝ := (0, 0)
  Z : ℝ × ℝ := (10, 0)

/-- The area of a triangle given three points -/
def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- The probability that a randomly chosen point Q in the interior of XYZ
    satisfies area(QYZ) < 1/3 * area(XYZ) -/
def probabilitySmallArea (t : RightTriangle) : ℝ := sorry

/-- Theorem: The probability that area(QYZ) < 1/3 * area(XYZ) is 2/3 -/
theorem probability_small_area_is_two_thirds (t : RightTriangle) :
  probabilitySmallArea t = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_small_area_is_two_thirds_l4109_410927


namespace NUMINAMATH_CALUDE_book_cost_is_16_l4109_410936

/-- Represents the cost of Léa's purchases -/
def total_cost : ℕ := 28

/-- Represents the number of binders Léa bought -/
def num_binders : ℕ := 3

/-- Represents the cost of each binder -/
def binder_cost : ℕ := 2

/-- Represents the number of notebooks Léa bought -/
def num_notebooks : ℕ := 6

/-- Represents the cost of each notebook -/
def notebook_cost : ℕ := 1

/-- Proves that the cost of the book is $16 -/
theorem book_cost_is_16 : 
  total_cost - (num_binders * binder_cost + num_notebooks * notebook_cost) = 16 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_is_16_l4109_410936


namespace NUMINAMATH_CALUDE_cost_of_four_birdhouses_l4109_410959

/-- The cost to build a given number of birdhouses -/
def cost_of_birdhouses (num_birdhouses : ℕ) : ℚ :=
  let planks_per_house : ℕ := 7
  let nails_per_house : ℕ := 20
  let cost_per_nail : ℚ := 5 / 100
  let cost_per_plank : ℕ := 3
  let cost_per_house : ℚ := (planks_per_house * cost_per_plank) + (nails_per_house * cost_per_nail)
  num_birdhouses * cost_per_house

/-- Theorem stating the cost of building 4 birdhouses is $88.00 -/
theorem cost_of_four_birdhouses :
  cost_of_birdhouses 4 = 88 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_four_birdhouses_l4109_410959


namespace NUMINAMATH_CALUDE_absent_students_probability_l4109_410956

theorem absent_students_probability 
  (p_absent : ℝ) 
  (h_p_absent : p_absent = 1 / 10) 
  (p_present : ℝ) 
  (h_p_present : p_present = 1 - p_absent) 
  (n_students : ℕ) 
  (h_n_students : n_students = 3) :
  (n_students.choose 2 : ℝ) * p_absent^2 * p_present = 27 / 1000 := by
sorry

end NUMINAMATH_CALUDE_absent_students_probability_l4109_410956


namespace NUMINAMATH_CALUDE_speedster_convertibles_l4109_410920

theorem speedster_convertibles (total : ℕ) 
  (h1 : 2 * total = 3 * (total - 40))  -- 2/3 of total are Speedsters, so 1/3 is 40
  (h2 : 5 * (2 * total / 3) = 4 * total)  -- 4/5 of Speedsters (2/3 of total) are convertibles
  : 4 * total / 5 = 64 := by sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l4109_410920


namespace NUMINAMATH_CALUDE_correct_division_l4109_410982

theorem correct_division (x : ℝ) (h : x / 1.5 = 3.8) : x / 6 = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l4109_410982


namespace NUMINAMATH_CALUDE_license_plate_count_l4109_410915

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of digits (0-9) --/
def num_digits : ℕ := 10

/-- The number of odd digits --/
def num_odd_digits : ℕ := 5

/-- The number of even digits --/
def num_even_digits : ℕ := 5

/-- The number of positions for digits --/
def num_digit_positions : ℕ := 3

/-- The number of valid license plates --/
def num_valid_plates : ℕ := 6591000

theorem license_plate_count :
  num_letters ^ 3 * (num_digit_positions * num_odd_digits * num_even_digits ^ 2) = num_valid_plates :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l4109_410915


namespace NUMINAMATH_CALUDE_wheel_probability_l4109_410962

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → 
  p_A + p_B + p_C + p_D = 1 →
  p_D = 1/4 := by sorry

end NUMINAMATH_CALUDE_wheel_probability_l4109_410962


namespace NUMINAMATH_CALUDE_inequalities_proof_l4109_410996

theorem inequalities_proof (a b c d : ℝ) : 
  ((a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧ 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l4109_410996


namespace NUMINAMATH_CALUDE_binomial_series_expansion_l4109_410989

theorem binomial_series_expansion (x : ℝ) (n : ℕ) (h : |x| < 1) :
  (1 / (1 - x))^n = 1 + ∑' k, (n + k - 1).choose (n - 1) * x^k :=
sorry

end NUMINAMATH_CALUDE_binomial_series_expansion_l4109_410989


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4109_410953

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 + 5 * x^2 + d * x + 7
  (g 2 = 11) ∧ (g (-3) = 134) → c = -35/13 ∧ d = 16/13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4109_410953


namespace NUMINAMATH_CALUDE_marys_income_percentage_l4109_410905

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.9)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.44 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l4109_410905


namespace NUMINAMATH_CALUDE_taco_cheese_amount_l4109_410946

/-- The amount of cheese (in ounces) needed for a burrito -/
def cheese_per_burrito : ℝ := 4

/-- The total amount of cheese (in ounces) needed for 7 burritos and 1 taco -/
def total_cheese : ℝ := 37

/-- The amount of cheese (in ounces) needed for a taco -/
def cheese_per_taco : ℝ := total_cheese - 7 * cheese_per_burrito

theorem taco_cheese_amount : cheese_per_taco = 9 := by
  sorry

end NUMINAMATH_CALUDE_taco_cheese_amount_l4109_410946


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_l4109_410999

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def line_equation (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.m

def perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

theorem ellipse_circle_tangent (e : Ellipse) (a : Point) (l : Line) :
  ellipse_equation e a ∧ 
  a.x = 2 ∧ a.y = Real.sqrt 2 ∧
  ∃ (p q : Point),
    ellipse_equation e p ∧
    ellipse_equation e q ∧
    line_equation l p ∧
    line_equation l q ∧
    perpendicular p q →
  ∃ (r : ℝ), r = Real.sqrt (8/3) ∧
    ∀ (x y : ℝ), x^2 + y^2 = r^2 →
    ∃ (t : Point), line_equation l t ∧ t.x^2 + t.y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_l4109_410999


namespace NUMINAMATH_CALUDE_share_distribution_l4109_410951

theorem share_distribution (a b c : ℕ) : 
  a + b + c = 1010 →
  (a - 25) * 2 = (b - 10) * 3 →
  (a - 25) * 5 = (c - 15) * 3 →
  c = 495 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l4109_410951


namespace NUMINAMATH_CALUDE_x_12_equals_439_l4109_410958

theorem x_12_equals_439 (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^12 = 439 := by
  sorry

end NUMINAMATH_CALUDE_x_12_equals_439_l4109_410958


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l4109_410945

/-- A line parallel to y = 3x + 1 passing through (3,6) has y-intercept -3 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b x = y ↔ ∃ k, y = 3 * x + k) →  -- b is parallel to y = 3x + 1
  b 3 = 6 →                               -- b passes through (3,6)
  ∃ c, ∀ x, b x = 3 * x + c ∧ c = -3      -- b has equation y = 3x + c with c = -3
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l4109_410945


namespace NUMINAMATH_CALUDE_count_even_factors_l4109_410965

def n : ℕ := 2^4 * 3^3 * 7

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 32 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l4109_410965


namespace NUMINAMATH_CALUDE_assignment_theorem_l4109_410902

/-- The number of ways to assign 4 distinct objects to 3 distinct groups, 
    with at least one object in each group -/
def assignment_ways : ℕ := 36

/-- The number of ways to choose 2 objects from 4 distinct objects -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- The number of ways to arrange 3 distinct objects -/
def arrange_three : ℕ := Nat.factorial 3

theorem assignment_theorem : 
  assignment_ways = choose_two_from_four * arrange_three := by
  sorry

end NUMINAMATH_CALUDE_assignment_theorem_l4109_410902


namespace NUMINAMATH_CALUDE_custom_op_value_l4109_410981

-- Define the custom operation *
def custom_op (a b : ℤ) : ℚ := 1 / a + 1 / b

-- State the theorem
theorem custom_op_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 9) (prod_eq : a * b = 20) : 
  custom_op a b = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_value_l4109_410981


namespace NUMINAMATH_CALUDE_vermont_ads_clicked_l4109_410954

theorem vermont_ads_clicked (page1 page2 page3 page4 page5 page6 : ℕ) : 
  page1 = 18 →
  page2 = 2 * page1 →
  page3 = page2 + 32 →
  page4 = (5 * page2 + 4) / 8 →  -- Rounding up (5/8 * page2)
  page5 = page3 + 15 →
  page6 = page1 + page2 + page3 - 42 →
  ((3 * (page1 + page2 + page3 + page4 + page5 + page6) + 2) / 5 : ℕ) = 185 := by
  sorry

end NUMINAMATH_CALUDE_vermont_ads_clicked_l4109_410954


namespace NUMINAMATH_CALUDE_quarterback_passes_l4109_410931

theorem quarterback_passes (left right center : ℕ) : 
  left = 12 →
  right = 2 * left →
  center = left + 2 →
  left + right + center = 50 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l4109_410931


namespace NUMINAMATH_CALUDE_meaningful_fraction_l4109_410924

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l4109_410924


namespace NUMINAMATH_CALUDE_words_with_A_count_l4109_410997

def letter_set : Finset Char := {'A', 'B', 'C', 'D', 'E'}

/-- The number of 4-letter words using letters A, B, C, D, E with repetition allowed -/
def total_words : ℕ := (Finset.card letter_set) ^ 4

/-- The number of 4-letter words using only B, C, D, E with repetition allowed -/
def words_without_A : ℕ := ((Finset.card letter_set) - 1) ^ 4

/-- The number of 4-letter words using A, B, C, D, E with repetition, containing at least one A -/
def words_with_A : ℕ := total_words - words_without_A

theorem words_with_A_count : words_with_A = 369 := by sorry

end NUMINAMATH_CALUDE_words_with_A_count_l4109_410997


namespace NUMINAMATH_CALUDE_function_inequality_l4109_410964

/-- Given a real-valued function f(x) = e^x / x, prove that for all real x ≠ 0, 
    1 / (x * f(x)) > 1 - x -/
theorem function_inequality (x : ℝ) (hx : x ≠ 0) : 
  let f : ℝ → ℝ := fun x => Real.exp x / x
  1 / (x * f x) > 1 - x := by sorry

end NUMINAMATH_CALUDE_function_inequality_l4109_410964


namespace NUMINAMATH_CALUDE_vector_collinearity_angle_l4109_410917

theorem vector_collinearity_angle (θ : Real) :
  let a : Fin 2 → Real := ![2 * Real.cos θ, 2 * Real.sin θ]
  let b : Fin 2 → Real := ![3, Real.sqrt 3]
  (∃ (k : Real), a = k • b) →
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_angle_l4109_410917


namespace NUMINAMATH_CALUDE_min_value_theorem_l4109_410986

open Real

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  let f := fun x => 1/x + 9/(1-x)
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 16) ∧ (∃ z, 0 < z ∧ z < 1 ∧ f z = 16) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4109_410986


namespace NUMINAMATH_CALUDE_problem_solution_l4109_410978

theorem problem_solution : 2^2 + (-3)^2 - 1^2 + 4*2*(-3) = -12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4109_410978


namespace NUMINAMATH_CALUDE_eight_digit_repeating_divisible_by_10001_l4109_410938

/-- An 8-digit positive integer whose first four digits are the same as its last four digits -/
def EightDigitRepeating (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
        1000 * a + 100 * b + 10 * c + d

theorem eight_digit_repeating_divisible_by_10001 (n : ℕ) (h : EightDigitRepeating n) :
  10001 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_repeating_divisible_by_10001_l4109_410938


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_l4109_410976

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six :
  let a : ℚ := 1/2
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/6144 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_l4109_410976


namespace NUMINAMATH_CALUDE_geraldine_jazmin_doll_difference_l4109_410972

theorem geraldine_jazmin_doll_difference : 
  let geraldine_dolls : ℝ := 2186.0
  let jazmin_dolls : ℝ := 1209.0
  geraldine_dolls - jazmin_dolls = 977.0 := by
  sorry

end NUMINAMATH_CALUDE_geraldine_jazmin_doll_difference_l4109_410972


namespace NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_is_nine_l4109_410910

/-- The cost of a cassette tape given Josie's shopping scenario -/
theorem cassette_tape_cost : ℝ → Prop :=
  fun x =>
    let initial_amount : ℝ := 50
    let headphone_cost : ℝ := 25
    let remaining_amount : ℝ := 7
    let num_tapes : ℝ := 2
    initial_amount - (num_tapes * x + headphone_cost) = remaining_amount →
    x = 9

/-- Proof that the cost of each cassette tape is $9 -/
theorem cassette_tape_cost_is_nine : cassette_tape_cost 9 := by
  sorry

end NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_is_nine_l4109_410910


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l4109_410928

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l4109_410928


namespace NUMINAMATH_CALUDE_triangle_cosine_proof_l4109_410925

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) (D : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  f A = 3 / 2 →
  ∃ (AD BD : ℝ), AD = Real.sqrt 2 * BD ∧ AD = 2 →
  Real.cos C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_proof_l4109_410925
