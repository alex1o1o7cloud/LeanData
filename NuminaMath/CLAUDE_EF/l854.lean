import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_existence_l854_85478

-- Define a structure for a point in a plane
structure Point : Type :=
  (x : ℝ) (y : ℝ)

-- Define a distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define a predicate for non-collinearity of three points
def nonCollinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- State the theorem
theorem unique_point_existence (A B C : Point) (h : nonCollinear A B C) :
  ∃! X : Point, 
    (distance X A)^2 + (distance X B)^2 + (distance A B)^2 = 
    (distance X B)^2 + (distance X C)^2 + (distance B C)^2 ∧
    (distance X B)^2 + (distance X C)^2 + (distance B C)^2 = 
    (distance X C)^2 + (distance X A)^2 + (distance C A)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_existence_l854_85478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l854_85421

/-- The area of the largest inscribed right-angled triangle in a circle -/
theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 12) :
  ∃ (A : ℝ),
    A = (r * (2 * r)) / 2 ∧
    ∀ (t : ℝ),
      (∃ (α β γ : ℝ),
        α + β + γ = Real.pi ∧
        0 < α ∧ α < Real.pi/2 ∧
        0 < β ∧ β < Real.pi/2 ∧
        0 < γ ∧ γ < Real.pi/2 ∧
        α = Real.pi/2 ∧
        β ≠ Real.pi/4 ∧
        t = (r * r * Real.sin β * Real.sin γ) / Real.sin α) →
      t ≤ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l854_85421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_l854_85437

/-- A line with inclination angle α passing through (-2, -4) -/
def line (α : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t, x = -2 + t * Real.cos α ∧ y = -4 + t * Real.sin α}

/-- The curve y² = 2x -/
def curve : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 2*x}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The point M(-2, -4) -/
def M : ℝ × ℝ := (-2, -4)

theorem line_intersects_curve (α : ℝ) :
  ∃ A B : ℝ × ℝ,
    A ∈ line α ∧ A ∈ curve ∧
    B ∈ line α ∧ B ∈ curve ∧
    A ≠ B ∧
    distance M A * distance M B = 40 →
  α = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_l854_85437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l854_85482

/-- The distance from a point to a plane in 3D space -/
noncomputable def distance_point_to_plane (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

/-- Theorem: The distance from the point (2,4,1) to the plane x+2y+2z+3=0 is 5 -/
theorem distance_to_specific_plane :
  distance_point_to_plane 2 4 1 1 2 2 3 = 5 := by
  sorry

#check distance_to_specific_plane

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l854_85482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l854_85407

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, (2 : ℤ)^(2*n) + 15*n - 1 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l854_85407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l854_85488

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = Real.sqrt 2 / 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem statement
theorem propositions_truth : 
  p ∧ 
  q ∧ 
  (p ∧ q) ∧ 
  ¬(p ∧ ¬q) ∧ 
  (¬p ∨ q) ∧ 
  ¬(¬p ∨ ¬q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l854_85488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_locus_ratio_l854_85483

/-- A parabola with its vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- Points on a parabola -/
def parabola_points (P : Parabola) : Set (ℝ × ℝ) :=
  sorry

/-- Angle between two points at the vertex of a parabola -/
def angle_at_vertex (P : Parabola) (A B : ℝ × ℝ) : ℝ :=
  sorry

/-- The locus of midpoints of chords of a parabola -/
def midpoint_locus (P : Parabola) (angle : ℝ) : Set (ℝ × ℝ) :=
  {m | ∃ (A B : ℝ × ℝ), A ∈ parabola_points P ∧ B ∈ parabola_points P ∧ 
       angle_at_vertex P A B = angle ∧ m = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)}

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem parabola_midpoint_locus_ratio (P : Parabola) :
  ∃ (Q : Parabola), midpoint_locus P (π/2) = parabola_points Q ∧
  distance Q.focus P.focus / distance Q.vertex P.vertex = Real.sqrt (7/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_locus_ratio_l854_85483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_equations_segment_length_l854_85430

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define a point on the directrix below the x-axis
def point_on_directrix (M : ℝ × ℝ) : Prop :=
  directrix M.1 ∧ M.2 < 0

-- Define the focus F
def focus (F : ℝ × ℝ) (p : ℝ) : Prop :=
  F.1 = p/2 ∧ F.2 = 0

-- Define the line l passing through M and F
def line_through_points (l : ℝ → ℝ) (M F : ℝ × ℝ) : Prop :=
  ∀ x, l x - M.2 = ((F.2 - M.2) / (F.1 - M.1)) * (x - M.1)

-- Define the intersection of line l and parabola C
def intersection (N : ℝ × ℝ) (l : ℝ → ℝ) (p : ℝ) : Prop :=
  parabola p N.1 N.2 ∧ l N.1 = N.2

-- F is the midpoint of MN
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  F.1 = (M.1 + N.1) / 2 ∧ F.2 = (M.2 + N.2) / 2

theorem parabola_and_line_equations
  (p : ℝ) (M F N : ℝ × ℝ) (l : ℝ → ℝ) :
  parabola p F.1 F.2 →
  directrix (-1) →
  point_on_directrix M →
  focus F p →
  line_through_points l M F →
  intersection N l p →
  is_midpoint F M N →
  (∀ x y, parabola p x y ↔ y^2 = 4*x) ∧
  (∀ x, l x = Real.sqrt 3 * (x - 1)) :=
sorry

theorem segment_length
  (p : ℝ) (M F N P : ℝ × ℝ) (l : ℝ → ℝ) :
  parabola p F.1 F.2 →
  directrix (-1) →
  point_on_directrix M →
  focus F p →
  line_through_points l M F →
  intersection N l p →
  intersection P l p →
  N ≠ P →
  is_midpoint F M N →
  Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = 16/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_equations_segment_length_l854_85430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_quadrant_check_l854_85426

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Determines if a point is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem distance_and_quadrant_check :
  distance 0 0 16 (-9) = Real.sqrt 337 ∧
  in_fourth_quadrant 16 (-9) := by
  sorry

#check distance_and_quadrant_check

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_quadrant_check_l854_85426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_distribution_l854_85468

theorem bread_distribution (men women children : ℕ) : 
  men + women + children = 12 →
  8 * men + 2 * women + children = 48 →
  men = 5 ∧ women = 1 ∧ children = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_distribution_l854_85468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_properties_l854_85495

/-- Represents the number of articles read by a student -/
def ArticleCount := Nat

/-- Represents the frequency of students reading a certain number of articles -/
def Frequency := Nat

/-- The data set of article counts -/
def data : List Nat := [15, 12, 15, 13, 15, 15, 12, 18, 13, 18, 18, 15, 13, 15, 12, 15, 13, 15, 18, 18]

/-- The organized data as a list of tuples (article count, frequency) -/
def organizedData : List (Nat × Nat) := [(12, 3), (13, 4), (15, 8), (18, 5)]

/-- The total number of sampled students -/
def totalStudents : Nat := 20

/-- The number of ninth-grade students in the school -/
def ninthGradeStudents : Nat := 300

/-- Theorem stating the properties of the data set -/
theorem data_properties :
  (∃ m : Nat, m = 4 ∧
              (List.sum (List.map (λ (x : Nat × Nat) => x.1 * x.2) organizedData) : Rat) / totalStudents = 149/10 ∧
              (149/10 : Rat) * ninthGradeStudents = 4470) ∧
  (∃ median : Nat, median = 15) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_properties_l854_85495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_of_equilateral_triangles_l854_85499

/-- The median of a trapezoid formed by two equilateral triangles -/
theorem trapezoid_median_of_equilateral_triangles :
  let large_side : ℝ := 4
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side^2
  let small_area : ℝ := (1 / 3) * large_area
  let small_side : ℝ := Real.sqrt ((4 * small_area) / Real.sqrt 3)
  let median : ℝ := (large_side + small_side) / 2
  median = 2 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_of_equilateral_triangles_l854_85499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_sum_l854_85413

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola x^2 - y^2 = 1 -/
def Hyperbola : Set Point :=
  {p : Point | p.x^2 - p.y^2 = 1}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- States that two vectors are perpendicular -/
def perpendicular (v1 v2 : Point × Point) : Prop :=
  (v1.2.x - v1.1.x) * (v2.2.x - v2.1.x) + (v1.2.y - v1.1.y) * (v2.2.y - v2.1.y) = 0

theorem hyperbola_focal_sum (F1 F2 P : Point) :
  P ∈ Hyperbola →
  F1 ≠ F2 →
  perpendicular (P, F1) (P, F2) →
  distance P F1 + distance P F2 = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_sum_l854_85413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cory_richard_time_difference_l854_85402

/-- Proves that Cory takes 37 minutes more than Richard to clean her room -/
theorem cory_richard_time_difference 
  (richard_time cory_time blake_time total_time : ℕ) : 
  richard_time = 22 →
  cory_time > richard_time →
  blake_time = cory_time - 4 →
  richard_time + cory_time + blake_time = total_time →
  total_time = 136 →
  cory_time - richard_time = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cory_richard_time_difference_l854_85402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_difference_l854_85498

theorem crayon_difference (willy lucy jake : ℕ) 
  (h_willy : willy = 5092)
  (h_lucy : lucy = 3971)
  (h_jake : jake = 2435) :
  Int.ofNat willy - Int.ofNat (lucy + jake) = -1314 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_difference_l854_85498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_capacity_no_jackets_l854_85497

/-- The number of people that can fit on a raft --/
def raft_capacity (with_jackets : ℕ) (total : ℕ) : ℕ := total

/-- The theorem stating the raft capacity when no one wears a life jacket --/
theorem raft_capacity_no_jackets :
  (∀ n, raft_capacity n n = raft_capacity 0 0 - 7) →
  (raft_capacity 8 17 = 17) →
  raft_capacity 0 0 = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_capacity_no_jackets_l854_85497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l854_85446

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_eccentricity_range (e : Ellipse) :
  ∃ (p : PointOnEllipse e), 
    let f1 := (-Real.sqrt (e.a^2 - e.b^2), 0)
    let f2 := (Real.sqrt (e.a^2 - e.b^2), 0)
    let d1 := Real.sqrt ((p.x - f1.1)^2 + p.y^2)
    let d2 := Real.sqrt ((p.x - f2.1)^2 + p.y^2)
    d1 = 3 * d2 →
    1/2 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l854_85446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_t_squared_l854_85436

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  /-- The equation of the hyperbola -/
  satisfies_equation : ℝ → ℝ → Prop
  /-- The hyperbola passes through (-3, 4) -/
  passes_through_minus3_4 : satisfies_equation (-3) 4
  /-- The hyperbola passes through (-2, 0) -/
  passes_through_minus2_0 : satisfies_equation (-2) 0
  /-- The hyperbola passes through (t, 2) for some t -/
  passes_through_t_2 : ∃ t, satisfies_equation t 2

/-- The theorem stating that for a hyperbola with the given properties, t² = 21/4 -/
theorem hyperbola_t_squared (h : Hyperbola) : 
  ∃ (t : ℝ), h.satisfies_equation t 2 ∧ t^2 = 21/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_t_squared_l854_85436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_same_color_l854_85463

def red_marbles : ℕ := 3
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 9
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles
def drawn_marbles : ℕ := 4

def probability_same_color : ℚ := 9 / 170

theorem probability_four_same_color :
  (Nat.choose white_marbles drawn_marbles : ℚ) / (Nat.choose total_marbles drawn_marbles) +
  (Nat.choose blue_marbles drawn_marbles : ℚ) / (Nat.choose total_marbles drawn_marbles) =
  probability_same_color :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_same_color_l854_85463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_isosceles_distance_l854_85404

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define an isosceles triangle
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∨
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∨
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem parabola_isosceles_distance (P : ℝ × ℝ) :
  point_on_parabola P →
  is_isosceles origin focus P →
  distance origin P = 3/2 ∨ distance origin P = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_isosceles_distance_l854_85404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l854_85465

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.pi * x / 6 + Real.pi / 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ 1 + 6 * ↑k} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l854_85465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l854_85460

/-- Triangle with vertices A, B, and C -/
structure Triangle (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  A : P
  B : P
  C : P

/-- The area of a triangle -/
noncomputable def Triangle.area {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (t : Triangle P) : ℝ := sorry

/-- The centroid of a triangle -/
noncomputable def Triangle.centroid {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (t : Triangle P) : P := sorry

/-- Check if a point is on a line segment -/
def onSegment {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (A B X : P) : Prop := sorry

/-- Check if a point is on the opposite side of a line to two other points -/
def oppositeSide {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (X Y G B C : P) : Prop := sorry

/-- The area of a quadrilateral -/
noncomputable def quadArea {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (A B C D : P) : ℝ := sorry

/-- Check if two line segments are parallel -/
def parallel {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (A B C D : P) : Prop := sorry

/-- Main theorem -/
theorem triangle_area_inequality {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (t : Triangle P) 
  (X Y : P) (h1 : t.area = 1) 
  (h2 : onSegment t.A t.B X) (h3 : onSegment t.A t.C Y)
  (h4 : oppositeSide X Y (t.centroid) t.B t.C) :
  quadArea X (t.centroid) Y t.B + quadArea Y (t.centroid) X t.C ≥ 4/9 ∧
  (quadArea X (t.centroid) Y t.B + quadArea Y (t.centroid) X t.C = 4/9 ↔ 
   parallel X Y t.B t.C ∧ onSegment X Y (t.centroid)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l854_85460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_bound_l854_85425

/-- A line passing through a point with a given inclination angle -/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- A curve defined by a polar equation -/
structure Curve where
  equation : ℝ → ℝ

/-- Represents an intersection point between a line and a curve -/
structure IntersectionPoint where
  point : ℝ × ℝ

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_bound 
  (l : Line) 
  (c : Curve) 
  (m n : IntersectionPoint) :
  l.point = (4, 2) →
  0 < l.angle →
  l.angle < Real.pi / 2 →
  c.equation = (fun θ ↦ 4 * Real.cos θ) →
  m ≠ n →
  4 < distance l.point m.point + distance l.point n.point →
  distance l.point m.point + distance l.point n.point ≤ 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_bound_l854_85425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_people_round_table_l854_85422

def number_of_people : ℕ := 7

-- The number of unique seating arrangements for n people around a round table
-- where rotations are considered identical
def unique_circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem seven_people_round_table :
  unique_circular_arrangements number_of_people = 720 := by
  -- Unfold the definition of unique_circular_arrangements
  unfold unique_circular_arrangements
  -- Unfold the definition of number_of_people
  unfold number_of_people
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_people_round_table_l854_85422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l854_85431

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l854_85431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l854_85417

/-- The volume of a sphere -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The radius of a sphere given its volume -/
noncomputable def sphereRadius (v : ℝ) : ℝ := (3 * v / (4 * Real.pi))^(1/3)

theorem sphere_diameter_theorem :
  let r₁ : ℝ := 6
  let v₁ : ℝ := sphereVolume r₁
  let v₂ : ℝ := 3 * v₁
  let r₂ : ℝ := sphereRadius v₂
  let d₂ : ℝ := 2 * r₂
  d₂ = 12 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l854_85417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_relation_l854_85401

theorem triangle_sine_relation (a b c : ℝ) (A B C : ℝ) (m : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  (1 / 2) * b * c * Real.sin A = (Real.sqrt 3 / 12) * a^2 ∧
  Real.sin B ^ 2 + Real.sin C ^ 2 = m * Real.sin B * Real.sin C →
  2 ≤ m ∧ m ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_relation_l854_85401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l854_85454

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : (Real × Real × Real) :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_specific :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l854_85454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approximation_l854_85408

noncomputable def initial_loan : ℝ := 15000
noncomputable def interest_rate : ℝ := 0.08
def loan_duration : ℕ := 8
def mid_point : ℕ := 4

noncomputable def plan1_compound (t : ℝ) : ℝ :=
  initial_loan * (1 + interest_rate / 4) ^ (4 * t)

noncomputable def plan2_compound (t : ℝ) : ℝ :=
  initial_loan * (1 + interest_rate) ^ t

noncomputable def plan1_payment : ℝ :=
  let mid_balance := plan1_compound mid_point
  let first_payment := mid_balance / 3
  let remaining_balance := mid_balance - first_payment
  let final_balance := remaining_balance * (1 + interest_rate / 4) ^ (4 * mid_point)
  first_payment + final_balance

noncomputable def plan2_payment : ℝ :=
  5000 + (initial_loan - 5000) * (1 + interest_rate) ^ mid_point

theorem payment_difference_approximation :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ 
  |plan1_payment - plan2_payment - 6967| ≤ ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approximation_l854_85408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l854_85448

-- Define the two lines
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define the slope of a line in the form ax + by + c = 0
noncomputable def lineSlope (a b : ℝ) : ℝ := -a / b

-- Define parallel lines
def parallel (m : ℝ) : Prop := 
  lineSlope 1 m = lineSlope (m - 2) 3

-- Theorem statement
theorem parallel_lines_m_values :
  ∀ m : ℝ, parallel m → m = -1 ∨ m = 3 := by
  intro m h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l854_85448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l854_85458

theorem range_of_a_for_false_proposition :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l854_85458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_l854_85459

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem g_composition (x : ℝ) (h : -1 < x ∧ x < 1) : 
  g ((4*x + x^3) / (1 + 4*x^2)) = 2 * g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_l854_85459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_correct_answer_l854_85469

/-- Represents the animals in the puzzle --/
inductive Animal : Type
| Cat
| Hen
| Crab
| Bear
| Goat

/-- Assigns a digit to each animal --/
def animal_value : Animal → Nat := sorry

/-- Represents a row or column in the puzzle --/
def PuzzleLine := List Animal

/-- Calculates the sum of a puzzle line given the animal values --/
def line_sum (line : PuzzleLine) (values : Animal → Nat) : Nat :=
  line.map values |> List.sum

/-- The puzzle lines with their expected sums --/
def puzzle_lines : List (PuzzleLine × Nat) := sorry

/-- The order of animals under the table --/
def table_order : List Animal :=
  [Animal.Cat, Animal.Hen, Animal.Crab, Animal.Bear, Animal.Goat]

theorem puzzle_solution :
  ∃! (values : Animal → Nat),
    (∀ a b, a ≠ b → values a ≠ values b) ∧
    (∀ (line : PuzzleLine) (sum : Nat), (line, sum) ∈ puzzle_lines → line_sum line values = sum) ∧
    (values Animal.Cat = 1 ∧
     values Animal.Hen = 5 ∧
     values Animal.Crab = 2 ∧
     values Animal.Bear = 4 ∧
     values Animal.Goat = 3) :=
by sorry

/-- The solution to the puzzle --/
def puzzle_answer : Nat :=
  table_order.map animal_value |> List.foldl (fun acc d => acc * 10 + d) 0

theorem correct_answer : puzzle_answer = 15243 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_correct_answer_l854_85469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l854_85442

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_equality 
  (h1 : ∀ x, f x < (1 / 2 : ℝ))
  (h2 : f 1 = 1)
  (h3 : ∀ x, HasDerivAt f (f' x) x) :
  {x : ℝ | f x < x / 2 + 1 / 2} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l854_85442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_four_two_l854_85443

/-- Power function passing through (4, 2) has exponent 1/2 -/
theorem power_function_through_four_two (a : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = x^a) → -- Power function exists for positive x
  (4 : ℝ)^a = 2 →                       -- Function passes through (4, 2)
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_four_two_l854_85443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_problem_l854_85493

theorem squares_problem (x y : ℝ) 
  (h1 : x^2 + y^2 = 145) 
  (h2 : x^2 - y^2 = 105) : 
  (4*x + 4*y = 28 * Real.sqrt 5) ∧ 
  ((x + y) * max x y = 175) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_problem_l854_85493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_of_reversed_base20_difference_l854_85427

/-- Two-digit number in base 20 -/
def base20Number (a b : ℤ) : ℤ := 20 * a + b

theorem prime_factor_of_reversed_base20_difference 
  (A B : ℤ) 
  (h1 : 0 ≤ A ∧ A < 20) 
  (h2 : 0 ≤ B ∧ B < 20) 
  (h3 : A ≠ B) : 
  19 ∣ |base20Number A B - base20Number B A| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_of_reversed_base20_difference_l854_85427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_count_l854_85490

theorem sum_digits_count (X Y : ℕ) : 
  10 ≤ X ∧ X ≤ 99 → 10 ≤ Y ∧ Y ≤ 99 → 
  let sum := 1234 + (100 * X + 65) + (100 * Y + 2)
  4 ≤ (Nat.digits 10 sum).length ∧ (Nat.digits 10 sum).length ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_count_l854_85490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_of_tangency_proof_l854_85419

/-- The point of tangency for two parabolas -/
noncomputable def point_of_tangency : ℝ × ℝ := (-9/2, -35/2)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 10*x + 20

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 36*y + 380

/-- Theorem stating that the given point is the point of tangency for the two parabolas -/
theorem point_of_tangency_proof :
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_of_tangency_proof_l854_85419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_point_l854_85486

/-- Point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Distance between two polar points -/
noncomputable def polarDistance (p1 p2 : PolarPoint) : ℝ :=
  Real.sqrt ((p1.r * Real.cos p1.θ - p2.r * Real.cos p2.θ)^2 + 
             (p1.r * Real.sin p1.θ - p2.r * Real.sin p2.θ)^2)

/-- The line ρ cos θ + ρ sin θ = 0 -/
def onLine (p : PolarPoint) : Prop :=
  p.r * Real.cos p.θ + p.r * Real.sin p.θ = 0

theorem shortest_distance_point :
  let A : PolarPoint := ⟨2, Real.pi / 2⟩
  let B : PolarPoint := ⟨Real.sqrt 2, 3 * Real.pi / 4⟩
  onLine B ∧ 
  (∀ C : PolarPoint, onLine C → 0 ≤ C.θ ∧ C.θ ≤ 2 * Real.pi → 
    polarDistance A B ≤ polarDistance A C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_point_l854_85486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_circular_arrangement_l854_85494

theorem impossibility_of_circular_arrangement : ¬ ∃ (arr : List Nat), 
  (arr.length = 2022) ∧ 
  (∀ n, n ∈ arr → 1 ≤ n ∧ n ≤ 2022) ∧
  (arr.Nodup) ∧
  (∀ i, i < arr.length → arr[i]! ∣ (arr[(i+1) % arr.length]! - arr[(i-1) % arr.length]!)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_circular_arrangement_l854_85494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_right_triangle_side_relation_right_triangle_l854_85429

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Statement 2
theorem angle_sum_right_triangle (t : Triangle) :
  t.A = t.B + t.C → t.A = 90 := by sorry

-- Statement 3
theorem side_relation_right_triangle (t : Triangle) :
  t.b^2 = t.a^2 - t.c^2 → t.A = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_right_triangle_side_relation_right_triangle_l854_85429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ballHittingGroundTime_l854_85476

/-- The equation describing the height of the ball -/
def ballHeight (t : ℝ) : ℝ := -20 * t^2 + 30 * t + 60

/-- The initial velocity of the ball -/
def initialVelocity : ℝ := 30

/-- The initial height of the ball -/
def initialHeight : ℝ := 60

/-- Theorem stating that the positive solution to the height equation is (3 + √57) / 4 -/
theorem ballHittingGroundTime :
  ∃ t : ℝ, t > 0 ∧ ballHeight t = 0 ∧ t = (3 + Real.sqrt 57) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ballHittingGroundTime_l854_85476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l854_85474

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem inequality_holds (n : ℕ) : 2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l854_85474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_triangle_colorings_l854_85467

/- Define the colors -/
inductive Color
| Red
| White
| Blue

/- Define a type for the figure -/
structure DoubleTriangle :=
  (vertices : Fin 6 → Color)

/- Define the adjacency relation -/
def are_adjacent (i j : Fin 6) : Prop :=
  (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 0) ∨
  (i = 2 ∧ j = 3) ∨ (i = 3 ∧ j = 4) ∨ (i = 4 ∧ j = 5) ∨ (i = 5 ∧ j = 2)

/- Define a predicate for valid colorings -/
def is_valid_coloring (dt : DoubleTriangle) : Prop :=
  ∀ i j, i ≠ j → are_adjacent i j → dt.vertices i ≠ dt.vertices j

/- Provide instances for Fintype and DecidablePred -/
instance : Fintype DoubleTriangle := by sorry

instance : DecidablePred is_valid_coloring := by sorry

/- State the theorem -/
theorem double_triangle_colorings :
  (Finset.filter is_valid_coloring (Finset.univ : Finset DoubleTriangle)).card = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_triangle_colorings_l854_85467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_part2_range_of_a_part3_l854_85416

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

-- Part 1: Tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ x + 2 * y - Real.log 2 - 1 = 0 := by
  sorry

-- Part 2: Range of a for f(x) > ax when x ∈ [0, 2]
theorem range_of_a_part2 (a : ℝ) :
  (∀ x, x ∈ Set.Icc 0 2 → f x > a * x) ↔ a < Real.exp 1 - 1 := by
  sorry

-- Part 3: Range of a for g(x) = f(x) - ax to have exactly one zero point
theorem range_of_a_part3 (a : ℝ) :
  (∃! x, f x = a * x) ↔ (a < -1 ∨ a = Real.exp 1 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_part2_range_of_a_part3_l854_85416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_reciprocal_on_unit_interval_l854_85424

theorem decreasing_reciprocal_on_unit_interval :
  ∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → (1 : ℝ) / y < (1 : ℝ) / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_reciprocal_on_unit_interval_l854_85424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_training_cost_l854_85455

/-- Represents the number of different car types in the fleet -/
def num_car_types : Nat := 5

/-- Represents the total number of drivers to be trained -/
def num_drivers : Nat := 8

/-- Represents the cost in rubles to train one driver for one car type -/
def training_cost_per_driver_per_car : Nat := 10000

/-- Represents the maximum number of drivers that can be absent -/
def max_absent_drivers : Nat := 3

/-- Represents a valid training plan -/
structure TrainingPlan where
  driver_car_assignments : (Fin num_drivers) → Finset (Fin num_car_types)
  all_cars_covered : ∀ (absent : Finset (Fin num_drivers)), absent.card ≤ max_absent_drivers →
    ∀ car : Fin num_car_types, ∃ driver : Fin num_drivers, driver ∉ absent ∧ car ∈ driver_car_assignments driver

/-- Calculates the total cost of a training plan -/
def total_cost (plan : TrainingPlan) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin num_drivers)) (λ driver => (plan.driver_car_assignments driver).card * training_cost_per_driver_per_car)

/-- The main theorem stating the minimum training cost -/
theorem min_training_cost :
  ∃ (plan : TrainingPlan), total_cost plan = 200000 ∧
  ∀ (other_plan : TrainingPlan), total_cost other_plan ≥ 200000 := by
  sorry

#check min_training_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_training_cost_l854_85455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_proof_l854_85420

theorem sine_value_proof (α : ℝ) 
  (h1 : Real.cos (α - π/6) + Real.sin α = 4/5 * Real.sqrt 3)
  (h2 : α ∈ Set.Ioo 0 (π/3)) :
  Real.sin (α + 5/12 * π) = 7/10 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_proof_l854_85420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_movement_l854_85414

/-- Robot's x-coordinate as a function of time -/
def x (t : ℝ) : ℝ := t * (t - 6)^2

/-- Robot's y-coordinate as a function of time -/
noncomputable def y (t : ℝ) : ℝ := if t ≤ 7 then 0 else (t - 7)^2

/-- Distance traveled by the robot in the first 7 minutes -/
def distance_7min : ℝ := 71

/-- Velocity vector at time t -/
noncomputable def velocity (t : ℝ) : ℝ × ℝ :=
  (3 * (t - 2) * (t - 6), if t ≤ 7 then 0 else 2 * (t - 7))

/-- Change in velocity vector from t=7 to t=8 -/
def velocity_change : ℝ × ℝ := (21, 2)

theorem robot_movement :
  (distance_7min = 71) ∧
  (Real.sqrt ((velocity_change.1)^2 + (velocity_change.2)^2) = Real.sqrt 445) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_movement_l854_85414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l854_85462

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11)
  (hb : b % 15 = 13)
  (hc : c % 15 = 9) :
  (a + b + c) % 15 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l854_85462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keystone_arch_angle_l854_85473

/-- Represents a keystone arch with congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoid_angle : ℚ

/-- The larger interior angle of a trapezoid in the keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℚ :=
  180 - (arch.trapezoid_angle / 2)

/-- Theorem stating the larger interior angle of trapezoids in a 12-piece keystone arch -/
theorem keystone_arch_angle :
  ∀ (arch : KeystoneArch),
    arch.num_trapezoids = 12 →
    arch.trapezoid_angle = 360 / arch.num_trapezoids →
    larger_interior_angle arch = 97.5 :=
by
  sorry

#eval larger_interior_angle { num_trapezoids := 12, trapezoid_angle := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keystone_arch_angle_l854_85473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l854_85456

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + 2 * x - 8

-- State the theorem
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 3 4, f c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l854_85456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_probability_relatively_prime_probability_sum_of_num_denom_main_result_l854_85435

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := num_guests * num_roll_types

/-- Represents the number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_roll_types

/-- Calculates the probability of each guest getting one roll of each type -/
def probability_one_of_each : ℚ := 10 / 8773

/-- Theorem stating that the calculated probability is correct -/
theorem correct_probability :
  probability_one_of_each = 10 / 8773 := by
  rfl

/-- Theorem stating that the numerator and denominator of the probability are relatively prime -/
theorem relatively_prime_probability :
  Nat.Coprime 10 8773 := by
  sorry

/-- Theorem stating the sum of numerator and denominator -/
theorem sum_of_num_denom :
  10 + 8773 = 8783 := by
  rfl

/-- The main theorem combining all the results -/
theorem main_result :
  ∃ (m n : ℕ), probability_one_of_each = m / n ∧ 
               Nat.Coprime m n ∧ 
               m + n = 8783 := by
  use 10, 8773
  constructor
  · exact correct_probability
  constructor
  · exact relatively_prime_probability
  · exact sum_of_num_denom

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_probability_relatively_prime_probability_sum_of_num_denom_main_result_l854_85435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonessas_take_home_pay_l854_85451

def calculate_take_home_pay (gross_pay : ℚ) : ℚ :=
  let tax_rate := if gross_pay < 1000 then 1/10 else 3/25
  let tax_deduction := tax_rate * gross_pay
  let insurance_deduction := 1/20 * gross_pay
  let pension_deduction := 3/100 * gross_pay
  let union_dues := 30
  let investment_fund := 50
  let total_deductions := tax_deduction + insurance_deduction + pension_deduction + union_dues + investment_fund
  gross_pay - total_deductions

theorem jonessas_take_home_pay :
  calculate_take_home_pay 500 = 330 := by
  sorry

#eval calculate_take_home_pay 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonessas_take_home_pay_l854_85451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l854_85471

open Real

theorem trig_problem (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : sin β / sin α = cos (α + β)) :
  (α = π/6 → tan β = Real.sqrt 3/5) ∧ 
  (∀ γ, 0 < γ ∧ γ < π/2 ∧ sin γ / sin α = cos (α + γ) → tan γ ≤ Real.sqrt 2/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l854_85471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_l854_85411

noncomputable def f (n : ℕ) : ℝ := (Finset.range n).sum (fun i => 1 / Real.sqrt (i + 1))

noncomputable def g (n : ℕ) : ℝ := 2 * (Real.sqrt (n + 1) - 1)

theorem f_greater_than_g : ∀ n : ℕ+, f n.val > g n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_l854_85411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_tangent_line_values_l854_85438

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x + 1 / (a * x) + b

-- Theorem for the minimum value of f
theorem min_value_of_f (a b : ℝ) (ha : a > 0) :
  ∃ (m : ℝ), ∀ (x : ℝ), x > 0 → f a b x ≥ m ∧ ∃ (x₀ : ℝ), x₀ > 0 ∧ f a b x₀ = m :=
sorry

-- Theorem for the values of a and b when the tangent line is y = (3/2)x
theorem tangent_line_values (a b : ℝ) (ha : a > 0) :
  (f a b 1 = 3/2 ∧ (deriv (f a b)) 1 = 3/2) → a = 2 ∧ b = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_tangent_line_values_l854_85438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l854_85479

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (x y : ℝ), 
    (6 * x - 5 * y - 28 = 0) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    (∃ (A C : ℝ × ℝ), 
      (6 * A.1 - 5 * A.2 - 28 = 0) ∧ 
      (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
      (6 * C.1 - 5 * C.2 - 28 = 0) ∧ 
      (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
      (∃ (B : ℝ × ℝ), B = (0, b) ∧
        (∃ (F2 : ℝ × ℝ), F2 = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) ∧
          (F2.1^2 + F2.2^2 = (a^2 - b^2))))) →
  a^2 = 5 * b^2 ∨ a^2 = 5 * b^2 / 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l854_85479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l854_85440

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (oplus 1 x) * x + (oplus 2 x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 10 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ M := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l854_85440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l854_85481

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) :=
  ∀ n : ℕ, n > 0 → a (n + 1) - a n = d

theorem sequence_formula (a : ℕ → ℕ) :
  a 1 = 1 →
  arithmetic_sequence (λ n ↦ (a n : ℚ) / n) 2 →
  ∀ n : ℕ, n > 0 → a n = 2 * n^2 - n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l854_85481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l854_85441

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem smallest_rotation_power (n : ℕ) :
  (n > 0) →
  (rotation_matrix (2 * Real.pi / 3))^n = 1 →
  (∀ m : ℕ, m > 0 → m < n → (rotation_matrix (2 * Real.pi / 3))^m ≠ 1) →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l854_85441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l854_85428

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

theorem extreme_value_condition (a b : ℝ) : 
  (f a b 1 = 10) ∧ 
  (f_derivative a b 1 = 0) → 
  a = -4 ∧ b = 11 := by
  sorry

#check extreme_value_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l854_85428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_distance_l854_85489

/-- Represents the journey to the pass -/
structure Journey where
  total_distance : ℕ
  days : ℕ
  first_day_distance : ℕ

/-- Calculates the sum of a geometric series -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: Given the conditions of the journey, the distance walked on the second day is 96 miles -/
theorem second_day_distance (j : Journey) 
  (h1 : j.total_distance = 378)
  (h2 : j.days = 6)
  (h3 : geometricSum (j.first_day_distance : ℚ) (1/2) j.days = j.total_distance) :
  j.first_day_distance / 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_distance_l854_85489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l854_85412

-- Define a decreasing function f on ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set
def solution_set : Set ℝ := {x | Real.exp (-1) < x ∧ x < Real.exp 2}

theorem inequality_solution :
  (∀ x y : ℝ, x < y → f y < f x) →  -- f is decreasing
  f 3 = -1 →                        -- Point A
  f 0 = 1 →                         -- Point B
  ∀ x : ℝ, |f (1 + Real.log x)| < 1 ↔ x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l854_85412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_half_l854_85485

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- Represents a hyperbola whose vertices and foci are the foci and vertices of the given ellipse -/
def Hyperbola (e : Ellipse) : Type :=
  {h : ℝ × ℝ // ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ h.1^2 = e.a^2 - e.b^2 ∧ h.2^2 = e.b^2}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The theorem to be proved -/
theorem ellipse_eccentricity_sqrt_half (e : Ellipse) (h : Hyperbola e) :
  (∃ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 ∧ 
   ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ x^2 / m^2 - y^2 / n^2 = 1 ∧
   y = x ∧ y = -x) →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_half_l854_85485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_survey_l854_85415

theorem traffic_survey (N : ℕ) : 
  -- Total number of drivers in four communities is N
  -- Number of drivers in community A is 96
  -- Number of drivers sampled from A, B, C, D are 12, 21, 25, 43 respectively
  (∃ y z w, 96 + y + z + w = N) ∧
  12 + 21 + 25 + 43 = 101 ∧
  (101 : ℚ) / N = 12 / 96 →
  N = 808 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_survey_l854_85415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_saving_time_l854_85487

def monthly_income : ℚ := 4000
def vehicle_cost : ℚ := 16000
def saving_ratio : ℚ := 1/2

def months_to_save : ℚ := vehicle_cost / (monthly_income * saving_ratio)

theorem vehicle_saving_time : months_to_save = 8 := by
  -- Unfold the definition of months_to_save
  unfold months_to_save
  -- Simplify the expression
  simp [monthly_income, vehicle_cost, saving_ratio]
  -- Perform the arithmetic
  norm_num

#eval months_to_save

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_saving_time_l854_85487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l854_85477

/-- The circle C₀ with equation x² + y² = r² -/
def C₀ (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

/-- A point (x, y) lies on a given set -/
def lies_on (p : ℝ × ℝ) (S : Set (ℝ × ℝ)) : Prop := p ∈ S

/-- The tangent line to a set S at a point p -/
noncomputable def tangent_line_at (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Given a circle C₀ with equation x² + y² = r² and a point M(x₀, y₀) on C₀,
    the equation x₀x + y₀y = r² represents the tangent line to C₀ at M. -/
theorem tangent_line_to_circle (r x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 = r^2) :
  ∀ x y : ℝ, (x₀*x + y₀*y = r^2) ↔ (x, y) ∈ tangent_line_at (C₀ r) (x₀, y₀) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l854_85477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l854_85444

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (-1, 2, 4)
  A₂ : ℝ × ℝ × ℝ := (-1, -2, -4)
  A₃ : ℝ × ℝ × ℝ := (3, 0, -1)
  A₄ : ℝ × ℝ × ℝ := (7, -3, 1)

/-- Volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Height from A₄ to face A₁A₂A₃ -/
def heightA₄ (t : Tetrahedron) : ℝ :=
  sorry

theorem tetrahedron_properties (t : Tetrahedron) :
  volume t = 24 ∧ heightA₄ t = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l854_85444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_sum_l854_85405

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/16 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the foci (we don't need to specify their exact coordinates)
variable (F₁ F₂ : ℝ × ℝ)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem ellipse_foci_distance_sum (P : PointOnEllipse) :
  distance (P.x, P.y) F₁ + distance (P.x, P.y) F₂ = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_sum_l854_85405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_specific_from_five_l854_85492

/-- The probability of selecting a specific person when choosing 3 from 5 -/
theorem probability_select_specific_from_five : 
  (Nat.choose 5 3 - Nat.choose 4 2) / Nat.choose 5 3 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_specific_from_five_l854_85492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lavender_to_coconut_ratio_l854_85434

/-- Represents the amount of scent used for each candle -/
def scent_per_candle : ℝ := 1

/-- Represents the number of almond candles made -/
def almond_candles : ℕ := 10

/-- Represents the total amount of almond scent used -/
def total_almond_scent : ℝ := scent_per_candle * almond_candles

/-- Represents the total amount of coconut scent available -/
def total_coconut_scent : ℝ := 1.5 * total_almond_scent

/-- Represents the number of coconut candles made -/
def coconut_candles : ℕ := 15

/-- Represents the number of lavender candles made -/
def lavender_candles : ℕ := sorry

/-- Theorem stating the ratio of lavender to coconut candles -/
theorem lavender_to_coconut_ratio :
  (lavender_candles : ℝ) / coconut_candles = lavender_candles / 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lavender_to_coconut_ratio_l854_85434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l854_85464

/-- Area of a quadrilateral given by four points -/
def Area_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of a hyperbola given a and b -/
def Eccentricity (a b : ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (2 * Real.sqrt 2 : ℝ) > 0 →  -- Right focus x-coordinate is positive
  (∃ A B : ℝ × ℝ, Area_quadrilateral (0, 0) A (2 * Real.sqrt 2, 0) B = 4) →  -- Area condition
  Eccentricity a b = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l854_85464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l854_85470

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (AM : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  2 * b * Real.cos A - Real.sqrt 3 * c * Real.cos A = Real.sqrt 3 * a * Real.cos C →
  B = π / 6 →
  AM = Real.sqrt 7 →
  AM ^ 2 = b ^ 2 + (b / 2) ^ 2 - 2 * b * (b / 2) * Real.cos (2 * π / 3) →
  A = π / 6 ∧ 
  (1 / 2) * b ^ 2 * Real.sin C = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l854_85470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l854_85491

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The point (a, b) lies on the given line -/
def LiesOnLine (t : Triangle) : Prop :=
  t.a * (Real.sin t.A - Real.sin t.B) + t.b * Real.sin t.B = t.c * Real.sin t.C

/-- The equation relating a and b -/
def SidesRelation (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = 6 * (t.a + t.b) - 18

theorem triangle_properties (t : Triangle) 
  (h1 : LiesOnLine t) (h2 : SidesRelation t) : 
  t.C = π/3 ∧ (1/4 : Real) * t.a * t.b * Real.sin t.C = 9 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l854_85491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_and_uniqueness_of_pair_l854_85450

theorem existence_and_uniqueness_of_pair (a : ℕ) :
  ∃! (x y : ℕ), x + ((x + y - 1) * (x + y - 2)) / 2 = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_and_uniqueness_of_pair_l854_85450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l854_85496

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 1 - y^2 / 3 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define a point on the hyperbola
variable (P : ℝ × ℝ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem hyperbola_triangle_area :
  hyperbola P.1 P.2 →
  3 * distance P F1 = 4 * distance P F2 →
  (1 / 2) * distance P F1 * distance P F2 * Real.sqrt (1 - ((distance P F1^2 + distance P F2^2 - distance F1 F2^2) / (2 * distance P F1 * distance P F2))^2) = 3 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l854_85496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_three_solution_is_three_l854_85406

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

-- State the theorem
theorem unique_solution_is_three :
  ∃! x, f x = 1/4 ∧ x > 1 := by
  -- The proof goes here
  sorry

-- Additional theorem to show that the solution is 3
theorem solution_is_three :
  f 3 = 1/4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_three_solution_is_three_l854_85406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_makes_fewest_cookies_l854_85423

-- Define the shapes of cookies
inductive CookieShape
  | Circle
  | Square
  | Hexagon
  | Kite

-- Define a function to calculate the area of each cookie shape
noncomputable def cookieArea (shape : CookieShape) : ℝ :=
  match shape with
  | .Circle => 4 * Real.pi
  | .Square => 9
  | .Hexagon => 6 * Real.sqrt 3
  | .Kite => 6

-- Define a function to calculate the number of cookies for each shape
noncomputable def numCookies (shape : CookieShape) : ℝ :=
  (15 * cookieArea CookieShape.Circle) / cookieArea shape

-- Theorem statement
theorem charlie_makes_fewest_cookies :
  numCookies CookieShape.Hexagon < numCookies CookieShape.Circle ∧
  numCookies CookieShape.Hexagon < numCookies CookieShape.Square ∧
  numCookies CookieShape.Hexagon < numCookies CookieShape.Kite :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_makes_fewest_cookies_l854_85423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_2_sqrt_3_l854_85400

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line passing through the focus of the parabola
def focus_line (x : ℝ) : Prop := x = 1

-- Define the asymptotes of the hyperbola
def asymptote_pos (x y : ℝ) : Prop := y = Real.sqrt 3 * x
def asymptote_neg (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Define points A and B
noncomputable def point_A : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def point_B : ℝ × ℝ := (1, -Real.sqrt 3)

-- Theorem statement
theorem length_AB_is_2_sqrt_3 :
  let d := Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2)
  d = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_2_sqrt_3_l854_85400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_identity_l854_85453

theorem sine_sum_identity (α β : Real) (h : α + β ≤ π) :
  Real.sin α + Real.sin β = 2 * Real.sin ((α + β) / 2) * Real.cos ((α - β) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_identity_l854_85453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l854_85445

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2*x - 3) / (6*x + 1)

-- Theorem statement
theorem vertical_asymptote_of_f :
  ∃ (x : ℝ), x = -1/6 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
  ∀ (y : ℝ), 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l854_85445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_reversed_difference_l854_85439

/-- Predicate to check if y is the reverse of x -/
def is_reverse (x y : ℤ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧
    x = 100 * a + 10 * b + c ∧
    y = 100 * c + 10 * b + a

/-- Given two integers x and y between 0 and 999 inclusive, where y is the reverse of x,
    the number of distinct values of |x - y| is 10. -/
theorem distinct_values_of_reversed_difference : ∃ (S : Finset ℤ),
  (∀ x y : ℤ, 0 ≤ x ∧ x ≤ 999 ∧ 0 ≤ y ∧ y ≤ 999 →
    is_reverse x y →
      (|x - y| ∈ S)) ∧
  S.card = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_reversed_difference_l854_85439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_derivative_zero_l854_85432

theorem constant_derivative_zero
  (f : ℝ → ℝ) (a b : ℝ) (h : a ≤ b)
  (hf : DifferentiableOn ℝ f (Set.Icc a b))
  (hM : ∃ M, ∀ x ∈ Set.Icc a b, f x ≤ M)
  (hm : ∃ m, ∀ x ∈ Set.Icc a b, m ≤ f x)
  (heq : ∃ c, (∀ x ∈ Set.Icc a b, f x ≤ c) ∧ (∀ x ∈ Set.Icc a b, c ≤ f x)) :
  ∀ x ∈ Set.Icc a b, deriv f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_derivative_zero_l854_85432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_representation_l854_85452

/-- Given two vectors e₁ and e₂ in ℝ², prove that the vector a can be represented as their linear combination -/
theorem vector_representation (e₁ e₂ a : ℝ × ℝ) : 
  e₁ = (-5, 3) → e₂ = (-2, 1) → a = (-3, 7) → 
  ∃ (l m : ℝ), a = l • e₁ + m • e₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_representation_l854_85452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_iff_a_in_range_l854_85433

/-- The function f(x) = lg |ax^2 + ax + 1| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (|a * x^2 + a * x + 1|) / Real.log 10

/-- The domain of f is ℝ if and only if a ∈ [0, 4) -/
theorem domain_f_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ∈ Set.Ico 0 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_iff_a_in_range_l854_85433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l854_85472

noncomputable section

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 8 = 1

-- Define the given points
def point1 : ℝ × ℝ := (3, -4 * Real.sqrt 2)
def point2 : ℝ × ℝ := (9/4, 5)
def point3 : ℝ × ℝ := (3 * Real.sqrt 2, 2)

-- Define the shared hyperbola
def shared_hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 4 = 1

-- Theorem statement
theorem hyperbola_satisfies_conditions :
  -- The hyperbola passes through point1
  hyperbola point1.1 point1.2 ∧
  -- The hyperbola passes through point2
  hyperbola point2.1 point2.2 ∧
  -- The hyperbola passes through point3
  hyperbola point3.1 point3.2 ∧
  -- The hyperbola has foci on the y-axis
  (∃ c : ℝ, c > 0 ∧ hyperbola 0 c ∧ hyperbola 0 (-c)) ∧
  -- The hyperbola shares a common focus with the shared hyperbola
  (∃ f : ℝ, (hyperbola 0 f ∨ hyperbola 0 (-f)) ∧
            (shared_hyperbola 0 f ∨ shared_hyperbola 0 (-f))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l854_85472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_in_square_l854_85484

/-- Given a square ABCD with side length 2 and E as the midpoint of AB, 
    prove that the dot product of vectors EC and ED is equal to 3. -/
theorem dot_product_in_square (A B C D E : ℝ × ℝ) : 
  (‖B - A‖ = 2) →  -- Side length of square is 2
  (‖C - B‖ = 2) → 
  (‖D - C‖ = 2) → 
  (‖A - D‖ = 2) → 
  ((B - A) • (C - B) = 0) →  -- Perpendicular sides
  ((C - B) • (D - C) = 0) → 
  ((D - C) • (A - D) = 0) → 
  ((A - D) • (B - A) = 0) → 
  (E = (A + B) / 2) →  -- E is midpoint of AB
  ((C - E) • (D - E) = 3) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_in_square_l854_85484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_improper_fraction_count_l854_85475

theorem improper_fraction_count : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2000) ∧ 
    (∀ n ∈ S, Int.gcd (n^2 + 11) (n + 5) > 1) ∧
    S.card = 55 ∧
    (∀ n : ℤ, 1 ≤ n ∧ n ≤ 2000 ∧ Int.gcd (n^2 + 11) (n + 5) > 1 → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_improper_fraction_count_l854_85475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l854_85466

def sequence_a (n : ℕ) : ℚ := (1 / 3) ^ n.succ

def sum_S (n : ℕ) : ℚ := (1 / 2) * (1 - (1 / 3) ^ n.succ)

def arithmetic_sequence (a b c : ℚ) : Prop := b - a = c - b

theorem sequence_properties :
  ∀ (n : ℕ),
  (∀ (k : ℕ), sequence_a k = (1 / 3) ^ k.succ) ∧
  (∀ (k : ℕ), sum_S (k + 1) - sum_S k = (1 / 3) ^ (k + 1).succ) ∧
  (∀ (k : ℕ), sum_S k = (1 / 2) * (1 - (1 / 3) ^ k.succ)) ∧
  (∃ (t : ℚ), arithmetic_sequence (sum_S 0) (t * (sum_S 0 + sum_S 1)) (3 * (sum_S 1 + sum_S 2)) ∧ t = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l854_85466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l854_85403

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^2 = b^2 + c^2 - b*c → ∃ A : ℝ, 0 < A ∧ A < Real.pi ∧ Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧ A = Real.pi/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l854_85403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_pot_profit_l854_85410

theorem flower_pot_profit (x : ℝ) : 
  (∀ n : ℕ, n > 3 → (n - 3) * (10 - (n - 3)) = n * (10 - (n - 3))) →
  (3 * 10 = 30) →
  ((x + 3) * (10 - x) = 40) ↔ 
  (x ≥ 0 ∧ x = (40 / (10 - x)) - 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_pot_profit_l854_85410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_general_term_correct_l854_85449

def sequenceA (n : ℕ+) : ℕ :=
  2 * n.val - 1

theorem sequence_correct : ∀ n : ℕ+,
  (n = 1 → sequenceA n = 1) ∧
  (n = 2 → sequenceA n = 3) ∧
  (n = 3 → sequenceA n = 5) ∧
  (n = 4 → sequenceA n = 7) :=
by
  intro n
  constructor
  · intro h; simp [sequenceA, h]
  constructor
  · intro h; simp [sequenceA, h]
  constructor
  · intro h; simp [sequenceA, h]
  · intro h; simp [sequenceA, h]

theorem general_term_correct : ∀ n : ℕ+,
  sequenceA n = 2 * n.val - 1 :=
by
  intro n
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_general_term_correct_l854_85449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l854_85457

/-- The projection vector of a onto b is (-4/5, -8/5) -/
theorem projection_vector (a b c : ℝ × ℝ) : 
  a = (-2, -1) → b = (1, 2) → 
  c = ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b → 
  c = (-4/5, -8/5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l854_85457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophomore_count_l854_85447

theorem sophomore_count (total_students : ℕ) 
  (sophomore_chess_percent : ℚ) (senior_chess_percent : ℚ) :
  total_students = 36 →
  sophomore_chess_percent = 30 / 100 →
  senior_chess_percent = 25 / 100 →
  ∃ (sophomores seniors : ℕ),
    sophomores + seniors = total_students ∧
    (sophomore_chess_percent * sophomores = senior_chess_percent * seniors) ∧
    sophomores = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophomore_count_l854_85447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_triangle_abc_sides_l854_85480

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  (∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2), f x ∈ Set.Icc (-1) 3) ∧
  (∀ x ∈ Set.Ioo (-Real.pi/2) (-Real.pi/3), ∀ y ∈ Set.Ioo (-Real.pi/2) (-Real.pi/3), x < y → f y < f x) ∧
  (∀ x ∈ Set.Ioo (Real.pi/6) (Real.pi/2), ∀ y ∈ Set.Ioo (Real.pi/6) (Real.pi/2), x < y → f y < f x) ∧
  (∀ x ∈ Set.Ioo (-Real.pi/3) (Real.pi/6), ∀ y ∈ Set.Ioo (-Real.pi/3) (Real.pi/6), x < y → f x < f y) :=
by sorry

theorem triangle_abc_sides :
  ∀ A B C a b c : ℝ,
  0 < A ∧ A < Real.pi ∧
  f A = 2 ∧
  a = Real.sqrt 3 ∧
  b + c = 3 ∧
  b > c →
  b = 2 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_triangle_abc_sides_l854_85480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_ratio_l854_85418

/-- Given real numbers p, q, r and points P, Q, R in ℝ³ such that
    the midpoint of QR is (p,0,0),
    the midpoint of PR is (0,q,0),
    and the midpoint of PQ is (0,0,r),
    prove that (PQ² + PR² + QR²) / (p² + q² + r²) = 8 -/
theorem midpoint_ratio (p q r : ℝ) (P Q R : Fin 3 → ℝ) 
    (h1 : (Q 0 + R 0) / 2 = p ∧ (Q 1 + R 1) / 2 = 0 ∧ (Q 2 + R 2) / 2 = 0)
    (h2 : (P 0 + R 0) / 2 = 0 ∧ (P 1 + R 1) / 2 = q ∧ (P 2 + R 2) / 2 = 0)
    (h3 : (P 0 + Q 0) / 2 = 0 ∧ (P 1 + Q 1) / 2 = 0 ∧ (P 2 + Q 2) / 2 = r) :
  (dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2) / (p^2 + q^2 + r^2) = 8 := by
  sorry

#check midpoint_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_ratio_l854_85418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_exists_2x2_square_l854_85461

/-- Represents a rectangle on the grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents the initial 8x8 grid -/
def initialGrid : Nat := 8

/-- Represents the number of 2x1 rectangles removed -/
def removedRectangles : Nat := 8

/-- Represents the dimensions of the removed rectangles -/
def removedRectangleType : Rectangle := { width := 2, height := 1 }

/-- Represents the dimensions of the square we're trying to find -/
def targetSquare : Rectangle := { width := 2, height := 2 }

/-- 
  Theorem: After removing eight 2x1 rectangles from an 8x8 grid, 
  there always exists at least one 2x2 square in the remaining area.
-/
theorem always_exists_2x2_square : 
  ∃ (remaining_area : Finset (Nat × Nat)), 
    (∀ x y, (x, y) ∈ remaining_area → x < initialGrid ∧ y < initialGrid) ∧
    (remaining_area.card = initialGrid^2 - removedRectangles * removedRectangleType.width * removedRectangleType.height) ∧
    (∃ x y, ∀ i j, i < targetSquare.width ∧ j < targetSquare.height → 
      (x + i, y + j) ∈ remaining_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_exists_2x2_square_l854_85461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l854_85409

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem complement_of_A : Set.univ \ A = {y : ℝ | y < 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l854_85409
