import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_medians_side_length_l796_79626

/-- Triangle DEF with perpendicular medians -/
structure TriangleWithPerpendicularMedians where
  /-- Point D of the triangle -/
  D : ℝ × ℝ
  /-- Point E of the triangle -/
  E : ℝ × ℝ
  /-- Point F of the triangle -/
  F : ℝ × ℝ
  /-- Point P on side EF (endpoint of median from D) -/
  P : ℝ × ℝ
  /-- Point Q on side DF (endpoint of median from E) -/
  Q : ℝ × ℝ
  /-- Centroid of the triangle -/
  G : ℝ × ℝ
  /-- Median DP is perpendicular to median EQ -/
  medians_perpendicular : (P.1 - D.1) * (Q.1 - E.1) + (P.2 - D.2) * (Q.2 - E.2) = 0
  /-- Length of median DP is 27 -/
  dp_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 27
  /-- Length of median EQ is 36 -/
  eq_length : Real.sqrt ((Q.1 - E.1)^2 + (Q.2 - E.2)^2) = 36
  /-- Centroid G divides median DP in ratio 2:1 -/
  g_divides_dp : (G.1 - D.1) / (P.1 - G.1) = 2 ∧ (G.2 - D.2) / (P.2 - G.2) = 2
  /-- Centroid G divides median EQ in ratio 2:1 -/
  g_divides_eq : (G.1 - E.1) / (Q.1 - G.1) = 2 ∧ (G.2 - E.2) / (Q.2 - G.2) = 2

/-- The length of side DE in a triangle with perpendicular medians -/
noncomputable def side_length_DE (t : TriangleWithPerpendicularMedians) : ℝ :=
  Real.sqrt ((t.E.1 - t.D.1)^2 + (t.E.2 - t.D.2)^2)

/-- Theorem: In a triangle with perpendicular medians satisfying the given conditions, 
    the length of side DE is 27 -/
theorem perpendicular_medians_side_length 
  (t : TriangleWithPerpendicularMedians) : side_length_DE t = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_medians_side_length_l796_79626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_extended_parallelepiped_sum_of_m_n_p_n_and_p_relatively_prime_l796_79651

/-- Represents the dimensions of the rectangular parallelepiped -/
structure ParallelepipedDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a rectangular parallelepiped -/
noncomputable def extendedVolume (d : ParallelepipedDimensions) : ℝ :=
  d.length * d.width * d.height +
  2 * (d.length * d.width + d.length * d.height + d.width * d.height) +
  Real.pi * (d.length + d.width + d.height) +
  4 * Real.pi / 3

/-- Theorem stating the volume of the set of points for the given parallelepiped dimensions -/
theorem volume_of_extended_parallelepiped :
  let dimensions : ParallelepipedDimensions := ⟨2, 3, 4⟩
  extendedVolume dimensions = (228 + 85 * Real.pi) / 3 := by
  sorry

/-- Theorem stating the sum of m, n, and p -/
theorem sum_of_m_n_p :
  let m : ℕ := 228
  let n : ℕ := 85
  let p : ℕ := 3
  m + n + p = 316 := by
  sorry

/-- Theorem stating that n and p are relatively prime -/
theorem n_and_p_relatively_prime :
  let n : ℕ := 85
  let p : ℕ := 3
  Nat.Coprime n p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_extended_parallelepiped_sum_of_m_n_p_n_and_p_relatively_prime_l796_79651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swim_path_y_l796_79667

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the triangle ABC -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the path of the boy's swim -/
structure SwimPath where
  x : ℕ
  y : ℕ

/-- Circular chord predicate -/
def is_on_circular_chord (P : Point) : Prop := sorry

/-- Equilateral triangle predicate -/
def is_equilateral_triangle (t : Triangle) : Prop := sorry

/-- Triangle side length function -/
def triangle_side_length (t : Triangle) : ℝ := sorry

/-- Valid swim path predicate -/
def is_valid_swim_path (t : Triangle) (p : SwimPath) : Prop := sorry

/-- Main theorem -/
theorem swim_path_y (t : Triangle) (p : SwimPath) : 
  -- A, B, C are points on the edge of a circular chord
  -- B is due west of C
  -- ABC is an equilateral triangle
  (∀ (P : Point), (P = t.A ∨ P = t.B ∨ P = t.C) → is_on_circular_chord P) →
  (t.B.x < t.C.x ∧ t.B.y = t.C.y) →
  is_equilateral_triangle t →
  -- Side length of ABC is 86 meters
  triangle_side_length t = 86 →
  -- A boy swims from A towards B for x meters, then turns west for y meters to reach shore
  is_valid_swim_path t p →
  -- x and y are positive integers (ensured by SwimPath structure)
  -- Conclusion: y = 12
  p.y = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swim_path_y_l796_79667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_area_l796_79648

/-- Hyperbola equation: x²/4 - y² = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

/-- Circle equation: x² + y² = 5 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- Left focus of the hyperbola -/
noncomputable def leftFocus : ℝ × ℝ := (-Real.sqrt 5, 0)

/-- Right focus of the hyperbola -/
noncomputable def rightFocus : ℝ × ℝ := (Real.sqrt 5, 0)

/-- Intersection points of the hyperbola and circle -/
noncomputable def intersectionPoints : List (ℝ × ℝ) := [
  (2 * Real.sqrt 30 / 5, Real.sqrt 5 / 5),
  (-2 * Real.sqrt 30 / 5, Real.sqrt 5 / 5),
  (-2 * Real.sqrt 30 / 5, -Real.sqrt 5 / 5),
  (2 * Real.sqrt 30 / 5, -Real.sqrt 5 / 5)
]

/-- Area of the quadrilateral formed by the intersection points -/
noncomputable def quadrilateralArea : ℝ := 8 * Real.sqrt 6 / 5

theorem hyperbola_circle_intersection_area :
  ∀ (p : ℝ × ℝ), p ∈ intersectionPoints →
    hyperbola p.1 p.2 ∧ circleEq p.1 p.2 →
    quadrilateralArea = 8 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_area_l796_79648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_on_hyperbola_l796_79696

/-- The hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The first asymptote -/
def asymptote1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0

/-- The second asymptote -/
def asymptote2 (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

/-- Distance from a point to a line ax + by + c = 0 -/
noncomputable def distanceToLine (x y a b c : ℝ) : ℝ := 
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem distance_product_on_hyperbola (x y : ℝ) :
  hyperbola x y →
  (distanceToLine x y (Real.sqrt 3) (-1) 0) * (distanceToLine x y (Real.sqrt 3) 1 0) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_on_hyperbola_l796_79696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l796_79641

/-- Yan's position between his house and a park -/
structure Position where
  distanceToHouse : ℝ
  distanceToPark : ℝ

/-- Yan's walking speed -/
noncomputable def walkingSpeed : ℝ := 1

/-- Yan's scooter speed relative to walking speed -/
noncomputable def scooterSpeedMultiplier : ℝ := 10

/-- Time taken for Yan to walk directly to the park -/
noncomputable def directWalkTime (p : Position) : ℝ := p.distanceToPark / walkingSpeed

/-- Time taken for Yan to walk back home and scooter to the park -/
noncomputable def homeScooterTime (p : Position) : ℝ :=
  p.distanceToHouse / walkingSpeed + (p.distanceToHouse + p.distanceToPark) / (scooterSpeedMultiplier * walkingSpeed)

/-- The theorem stating that the ratio of Yan's distance from his house to his distance to the park is 9/11 -/
theorem yan_distance_ratio (p : Position) 
    (h : directWalkTime p = homeScooterTime p) : 
    p.distanceToHouse / p.distanceToPark = 9 / 11 := by
  sorry

#check yan_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l796_79641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l796_79694

theorem count_integer_solutions : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^2 < 7*x) ∧ (∀ x : ℤ, x^2 < 7*x → x ∈ S) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l796_79694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sets_l796_79685

noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

noncomputable def set1 : List ℝ := [1, 2, 2]
noncomputable def set2 : List ℝ := [1, Real.sqrt 3, 2]
noncomputable def set3 : List ℝ := [4, 5, 6]
noncomputable def set4 : List ℝ := [1, 1, Real.sqrt 3]

theorem right_triangle_sets :
  (¬ is_right_triangle set1[0]! set1[1]! set1[2]!) ∧
  (is_right_triangle set2[0]! set2[1]! set2[2]!) ∧
  (¬ is_right_triangle set3[0]! set3[1]! set3[2]!) ∧
  (¬ is_right_triangle set4[0]! set4[1]! set4[2]!) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sets_l796_79685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_employed_is_sixty_percent_l796_79681

/-- Represents the employment statistics of a town -/
structure TownEmployment where
  /-- Percentage of the population that are employed males -/
  employed_males_percentage : ℚ
  /-- Percentage of employed people that are females -/
  employed_females_percentage : ℚ

/-- Calculates the total percentage of employed people in the population -/
def total_employed_percentage (stats : TownEmployment) : ℚ :=
  stats.employed_males_percentage / (1 - stats.employed_females_percentage / 100)

/-- Theorem: The total percentage of employed people in the population is 60% -/
theorem total_employed_is_sixty_percent (stats : TownEmployment)
  (h1 : stats.employed_males_percentage = 45)
  (h2 : stats.employed_females_percentage = 25) :
  total_employed_percentage stats = 60 := by
  sorry

#eval total_employed_percentage { employed_males_percentage := 45, employed_females_percentage := 25 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_employed_is_sixty_percent_l796_79681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_phi_equals_two_l796_79625

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem f_phi_equals_two (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) (h3 : f 0 φ = 1) : f φ φ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_phi_equals_two_l796_79625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_volume_cut_correct_l796_79670

/-- Represents a square-based truncated pyramid -/
structure TruncatedPyramid where
  m : ℝ  -- height
  a : ℝ  -- base edge length
  b : ℝ  -- top edge length
  h_positive : 0 < m
  h_base_gt_top : a > b
  h_positive_edges : 0 < b

/-- The distance from the base where a plane cuts the truncated pyramid into two equal volumes -/
noncomputable def equal_volume_cut (p : TruncatedPyramid) : ℝ :=
  (p.m / (p.a - p.b)) * (p.a - ((p.a^3 + p.b^3) / 2) ^ (1/3 : ℝ))

/-- Theorem stating that the equal_volume_cut function gives the correct distance -/
theorem equal_volume_cut_correct (p : TruncatedPyramid) :
  let x := equal_volume_cut p
  let y := ((p.a^3 + p.b^3) / 2) ^ (1/3 : ℝ)
  x * (p.a^2 + p.a * y + y^2) = (p.m - x) * (y^2 + y * p.b + p.b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_volume_cut_correct_l796_79670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_product_l796_79668

/-- An ellipse with equation x²/5 + y² = 1 -/
structure Ellipse where
  eq : ℝ → ℝ → Prop
  h_eq : ∀ (x y : ℝ), eq x y ↔ x^2 / 5 + y^2 = 1

/-- The foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  P : ℝ × ℝ
  on_ellipse : e.eq P.1 P.2

/-- Vector from a point to another point -/
def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

/-- Theorem statement -/
theorem ellipse_foci_product (e : Ellipse) (f : Foci e) (p : PointOnEllipse e) 
  (h : dot_product (vector p.P f.F₁) (vector p.P f.F₂) = 0) :
  distance p.P f.F₁ * distance p.P f.F₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_product_l796_79668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_square_of_linear_implies_m_value_l796_79633

/-- G is defined as (5x^2 + 12x + 2m) / 5 -/
noncomputable def G (x m : ℝ) : ℝ := (5 * x^2 + 12 * x + 2 * m) / 5

/-- An expression linear in x -/
noncomputable def linear_expr (x c d : ℝ) : ℝ := c * x + d

/-- Theorem: If G is the square of a linear expression in x, then m = 3.6 -/
theorem G_square_of_linear_implies_m_value (m : ℝ) :
  (∃ c d : ℝ, ∀ x : ℝ, G x m = (linear_expr x c d)^2) →
  m = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_square_of_linear_implies_m_value_l796_79633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_probability_part_two_probability_range_l796_79643

/-- Represents a shooter in the marksmanship competition -/
structure Shooter where
  hitRate : ℝ

/-- Represents the shooting team -/
structure ShootingTeam where
  shooterA : Shooter
  shooterB : Shooter

/-- Calculates the probability of being recognized as an "Advanced Harmonious Group" in a single test -/
def advancedHarmoniousGroupProbability (team : ShootingTeam) : ℝ :=
  let p1 := team.shooterA.hitRate
  let p2 := team.shooterB.hitRate
  2 * p1 * (1 - p1) * p2 * (1 - p2) + p1^2 * p2^2

/-- Theorem for part (1) of the problem -/
theorem part_one_probability (team : ShootingTeam) 
    (h1 : team.shooterA.hitRate = 2/3) 
    (h2 : team.shooterB.hitRate = 1/2) : 
  advancedHarmoniousGroupProbability team = 2/9 := by
  sorry

/-- Theorem for part (2) of the problem -/
theorem part_two_probability_range (team : ShootingTeam) 
    (h1 : team.shooterA.hitRate = 2/3)
    (h2 : 12 * advancedHarmoniousGroupProbability team ≥ 5) :
  3/4 ≤ team.shooterB.hitRate ∧ team.shooterB.hitRate ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_probability_part_two_probability_range_l796_79643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_terms_l796_79662

/-- Predicate to check if a sequence of four terms forms an arithmetic progression. -/
def is_arithmetic_seq (a b c d : ℝ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c)

/-- Predicate to check if a sequence of five terms forms a geometric progression. -/
def is_geometric_seq (a b c d e : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c) ∧ (d / c = e / d) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (d ≠ 0)

/-- Given two sequences, one arithmetic and one geometric, both starting with 1 and ending with 4,
    prove that the product of the second term of the arithmetic sequence and the third term of the
    geometric sequence is 6. -/
theorem product_of_terms (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (is_arithmetic_seq 1 a₁ a₂ 4) → 
  (is_geometric_seq 1 b₁ b₂ b₃ 4) → 
  a₂ * b₂ = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_terms_l796_79662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l796_79652

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3*θ) = -117/125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l796_79652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l796_79621

-- Define the cone
structure Cone where
  volume : ℝ
  baseArea : ℝ

-- Define the given cone
noncomputable def givenCone : Cone where
  volume := (2 * Real.sqrt 3 / 3) * Real.pi
  baseArea := 2 * Real.pi

-- Theorem statement
theorem cone_lateral_surface_area (c : Cone) 
  (h1 : c.volume = (2 * Real.sqrt 3 / 3) * Real.pi)
  (h2 : c.baseArea = 2 * Real.pi) :
  ∃ (lateralSurfaceArea : ℝ), lateralSurfaceArea = Real.sqrt 10 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l796_79621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_23_18_l796_79679

/-- The sum of the infinite series Σ(3n + 2) / (n(n+1)(n+3)) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3 * n + 2) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the sum of the infinite series is equal to 23/18 -/
theorem infinite_series_sum_equals_23_18 : infinite_series_sum = 23 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_23_18_l796_79679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_correct_problem_instance_l796_79687

/-- Calculate the simple interest rate given principal, amount, and time -/
noncomputable def simple_interest_rate (P A T : ℝ) : ℝ :=
  (A - P) * 100 / (P * T)

theorem simple_interest_rate_correct (P A T : ℝ) (h_positive : P > 0 ∧ T > 0) :
  let R := simple_interest_rate P A T
  A = P * (1 + R * T / 100) := by
  sorry

/-- The specific problem instance -/
theorem problem_instance :
  let P : ℝ := 1750
  let A : ℝ := 2000
  let T : ℝ := 4
  let R := simple_interest_rate P A T
  abs (R - 3.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_correct_problem_instance_l796_79687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l796_79673

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 25) (h2 : zero_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : ℕ :=
by
  -- Proof goes here
  sorry

-- Remove the #eval line as it was causing the error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l796_79673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l796_79614

/-- Definition of the ellipse C -/
noncomputable def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- Definition of a point on the major axis -/
def on_major_axis (P : ℝ × ℝ) : Prop :=
  P.2 = 0 ∧ -2 ≤ P.1 ∧ P.1 ≤ 2

/-- Definition of a line with slope 1/2 passing through a point -/
noncomputable def line_through (P : ℝ × ℝ) (x : ℝ) : ℝ :=
  (1/2) * (x - P.1) + P.2

/-- Theorem statement -/
theorem constant_sum_of_squares (P : ℝ × ℝ) (A B : ℝ × ℝ) 
    (h_P : on_major_axis P) 
    (h_A : A ∈ C ∧ A.2 = line_through P A.1) 
    (h_B : B ∈ C ∧ B.2 = line_through P B.1) : 
  (A.1 - P.1)^2 + (A.2 - P.2)^2 + (B.1 - P.1)^2 + (B.2 - P.2)^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l796_79614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_l796_79632

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parameters of an ellipse in standard form -/
structure EllipseParams where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is on the ellipse given its parameters -/
def isOnEllipse (p : Point) (params : EllipseParams) : Prop :=
  (p.x - params.h)^2 / params.a^2 + (p.y - params.k)^2 / params.b^2 = 1

/-- Theorem: The ellipse with given foci and point has the specified parameters -/
theorem ellipse_parameters :
  let f1 : Point := ⟨1, 3⟩
  let f2 : Point := ⟨5, 3⟩
  let p : Point := ⟨10, 2⟩
  let params : EllipseParams := ⟨
    (Real.sqrt 82 + Real.sqrt 26) / 2,
    Real.sqrt (((Real.sqrt 82 + Real.sqrt 26) / 2)^2 - 4),
    3,
    3
  ⟩
  (distance f1 p + distance f2 p = 2 * params.a) ∧
  (distance f1 f2 = 2 * Real.sqrt (params.a^2 - params.b^2)) ∧
  isOnEllipse p params ∧
  params.a > 0 ∧
  params.b > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_l796_79632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_angle_cosine_l796_79618

def a : Fin 3 → ℝ := ![3, 0, 1]
def b : Fin 3 → ℝ := ![1, 2, -1]

def diagonal1 : Fin 3 → ℝ := ![a 0 + b 0, a 1 + b 1, a 2 + b 2]
def diagonal2 : Fin 3 → ℝ := ![b 0 - a 0, b 1 - a 1, b 2 - a 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem parallelogram_diagonal_angle_cosine :
  dot_product diagonal1 diagonal2 / (magnitude diagonal1 * magnitude diagonal2) = -1 / Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_angle_cosine_l796_79618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_q_in_product_l796_79691

theorem power_of_q_in_product (p q : ℕ) (n : ℕ) : 
  Prime p → Prime q → p ≠ q → (Finset.card (Nat.divisors (p^2 * q^n)) = 18) → n = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_q_in_product_l796_79691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l796_79655

/-- Given points A, B, C, and D in a 2D plane, where B is the midpoint of AC,
    prove that the midpoint of CD has specific coordinates. -/
theorem midpoint_coordinates (A B C D : ℝ × ℝ) : 
  A = (0, 0) →
  B = (2, 3) →
  D = (10, 0) →
  B = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = (7, 3) := by
  sorry

#check midpoint_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l796_79655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_implies_solution_set_l796_79616

/-- Direct proportion function -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Inverse proportion function -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := c / x

/-- The set of x values where f(x) > g(x) -/
def solution_set (k c : ℝ) : Set ℝ :=
  {x | x < -2 ∨ (0 < x ∧ x < 2)}

theorem intersection_point_implies_solution_set (k c : ℝ) :
  f k 2 = g c 2 ∧ f k 2 = -1/3 →
  ∀ x, x ∈ solution_set k c ↔ f k x > g c x := by
  sorry

#check intersection_point_implies_solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_implies_solution_set_l796_79616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_westford_carnival_savings_l796_79608

/-- Represents the carnival hat discount offer -/
structure CarnivalOffer where
  regularPrice : ℚ
  secondHatDiscount : ℚ
  thirdHatDiscount : ℚ
  fourthHatDiscount : ℚ

/-- Calculate the total cost of four hats with the given carnival offer -/
def totalCostWithDiscount (offer : CarnivalOffer) : ℚ :=
  offer.regularPrice +
  offer.regularPrice * (1 - offer.secondHatDiscount) +
  offer.regularPrice * (1 - offer.thirdHatDiscount) +
  offer.regularPrice * (1 - offer.fourthHatDiscount)

/-- Calculate the percentage saved when buying four hats with the given carnival offer -/
def percentageSaved (offer : CarnivalOffer) : ℚ :=
  (1 - totalCostWithDiscount offer / (4 * offer.regularPrice)) * 100

/-- The 2023 Westford Village Carnival hat offer -/
def westfordCarnivalOffer : CarnivalOffer :=
  { regularPrice := 60
  , secondHatDiscount := 1/5
  , thirdHatDiscount := 2/5
  , fourthHatDiscount := 1/2 }

theorem westford_carnival_savings :
  percentageSaved westfordCarnivalOffer = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_westford_carnival_savings_l796_79608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_equation_l796_79639

/-- Given a regression line with slope 1.23 passing through the point (4, 5),
    prove that its equation is ŷ = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 → center_x = 4 → center_y = 5 →
  ∃ (intercept : ℝ), intercept = center_y - slope * center_x ∧
                     intercept = 0.08 ∧
                     (fun x ↦ slope * x + intercept) = (fun x ↦ 1.23 * x + 0.08) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_equation_l796_79639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l796_79653

theorem cosine_identity (α : ℝ) : 
  Real.cos (2 * α) - Real.cos (3 * α) - Real.cos (4 * α) + Real.cos (5 * α) = 
  -4 * Real.sin (α / 2) * Real.sin α * Real.cos (7 * α / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l796_79653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l796_79699

/-- An arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence ℝ) :
  (seq.a 6 - 1)^3 + 2013 * (seq.a 6 - 1)^3 = 1 →
  (seq.a 2008 - 1)^3 + 2013 * (seq.a 2008 - 1)^3 = -1 →
  sum_n seq 2013 = 2013 ∧ seq.a 2008 < seq.a 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l796_79699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_reciprocal_sum_constant_l796_79671

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  h : f = fun x ↦ 4 * x^2

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  h : y = p.f x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement -/
theorem parabola_chord_reciprocal_sum_constant
  (p : Parabola)
  (C : PointOnParabola p)
  (h_C : C.x = 0 ∧ C.y = 16) :
  ∀ (A B : PointOnParabola p),
  (A.x ≠ C.x ∨ A.y ≠ C.y) →
  (B.x ≠ C.x ∨ B.y ≠ C.y) →
  (A.y - C.y) * (B.x - C.x) = (B.y - C.y) * (A.x - C.x) →
  1 / distance (A.x, A.y) (C.x, C.y) + 1 / distance (B.x, B.y) (C.x, C.y) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_reciprocal_sum_constant_l796_79671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_range_l796_79689

-- Define the function f(x) = x^2 * e^x
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

-- Define the property of having an extreme point in an interval
def has_extreme_point_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, f y ≤ f x ∨ f y ≥ f x

-- Theorem statement
theorem extreme_point_range (a : ℝ) :
  has_extreme_point_in_interval f a (a + 1) ↔ a ∈ Set.Ioo (-3) (-2) ∪ Set.Ioo (-1) 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_range_l796_79689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_80_factorial_l796_79684

theorem last_two_nonzero_digits_80_factorial (n : ℕ) : n = 92 → 
  ∃ k m : ℕ, 80 * Nat.factorial 79 = k * 10^19 + n * 10^18 + m * 10^17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_80_factorial_l796_79684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arguments_l796_79647

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z^4 = -32i
def equation (z : ℂ) : Prop := z^4 = -32 * i

-- Define the four roots in polar form
noncomputable def root (k : Fin 4) (r : ℝ) (θ : ℝ) : ℂ := r * Complex.exp (θ * i)

-- Helper function to choose the kth element
def choose (k : Fin 4) (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  match k with
  | ⟨0, _⟩ => x₁
  | ⟨1, _⟩ => x₂
  | ⟨2, _⟩ => x₃
  | ⟨3, _⟩ => x₄

-- State the theorem
theorem sum_of_arguments (z₁ z₂ z₃ z₄ : ℂ) (r₁ r₂ r₃ r₄ θ₁ θ₂ θ₃ θ₄ : ℝ) :
  equation z₁ ∧ equation z₂ ∧ equation z₃ ∧ equation z₄ →
  z₁ = root 0 r₁ θ₁ ∧ z₂ = root 1 r₂ θ₂ ∧ z₃ = root 2 r₃ θ₃ ∧ z₄ = root 3 r₄ θ₄ →
  (∀ k : Fin 4, (root k (choose k r₁ r₂ r₃ r₄) (choose k θ₁ θ₂ θ₃ θ₄)).re > 0) →
  (∀ k : Fin 4, 0 ≤ choose k θ₁ θ₂ θ₃ θ₄ ∧ choose k θ₁ θ₂ θ₃ θ₄ < 2 * Real.pi) →
  θ₁ + θ₂ + θ₃ + θ₄ = (810 * Real.pi) / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arguments_l796_79647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_slope_of_parallel_line_l796_79623

theorem parallel_line_slope (a b c : ℝ) (h : b ≠ 0) :
  let original_line := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let slope := -a / b
  ∀ m : ℝ, (∃ k, {(x, y) : ℝ × ℝ | y = m * x + k} ⊆ original_line) → m = slope :=
by sorry

theorem slope_of_parallel_line :
  let original_eq := {(x, y) : ℝ × ℝ | 3 * x - 6 * y = 9}
  ∀ m : ℝ, (∃ k, {(x, y) : ℝ × ℝ | y = m * x + k} ⊆ original_eq) → m = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_slope_of_parallel_line_l796_79623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l796_79661

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define line l passing through P and intersecting circle O at A and B
def line_l (x y : ℝ) : Prop := (x = -1) ∨ (y = 1)

-- Define the distance between A and B
noncomputable def distance_AB : ℝ := 2 * Real.sqrt 3

-- Define the midpoint M of chord AB
def midpoint_M (x y : ℝ) : Prop := x^2 + y^2 + x - y = 0

theorem circle_intersection_theorem :
  ∀ (A B : ℝ × ℝ),
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    ‖A - B‖ = distance_AB →
    (∀ (x y : ℝ), line_l x y ↔ (x = -1 ∨ y = 1)) ∧
    (∀ (M : ℝ × ℝ), M = (A + B) / 2 → midpoint_M M.1 M.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l796_79661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_solution_set_l796_79629

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else -x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem f_geq_x_squared_solution_set :
  {x : ℝ | f x ≥ x^2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_solution_set_l796_79629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_l796_79635

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the rationalized expression
noncomputable def rationalizedExpression : ℝ := 
  (cubeRoot 25 + cubeRoot 10 + cubeRoot 4) / 3

-- Theorem statement
theorem rationalize_denominator_sum :
  (cubeRoot 25 + cubeRoot 10 + cubeRoot 4 + 3 = 42) ∧
  (rationalizedExpression = 1 / (cubeRoot 5 - cubeRoot 2)) :=
by
  sorry

#check rationalize_denominator_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_l796_79635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l796_79628

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + (m^2 - 1) * x

-- State the theorem
theorem f_properties (m : ℝ) (h : m > 0) :
  -- Part 1: Maximum and minimum values when m = 1
  (∀ x ∈ Set.Icc (-3) 2, f 1 x ≤ 18) ∧
  (∃ x ∈ Set.Icc (-3) 2, f 1 x = 18) ∧
  (∀ x ∈ Set.Icc (-3) 2, f 1 x ≥ 0) ∧
  (∃ x ∈ Set.Icc (-3) 2, f 1 x = 0) ∧
  -- Part 2: Monotonically increasing interval
  (∀ x y, 1 - m < x ∧ x < y ∧ y < m + 1 → f m x < f m y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l796_79628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l796_79606

theorem constant_term_expansion (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → ∃ b c d : ℝ, (2 + a * x) * (1 + 1/x)^5 = 12 + b * x + c * (1/x) + d) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l796_79606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_eq_14_l796_79676

/-- The number of integers satisfying the given conditions -/
def count_integers : ℕ :=
  (Finset.filter (fun n : ℕ =>
    150 < n ∧ n < 300 ∧ n % 7 = n % 9) (Finset.range 300)).card

/-- Theorem stating that there are exactly 14 integers satisfying the conditions -/
theorem count_integers_eq_14 : count_integers = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_eq_14_l796_79676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_f_composed_l796_79634

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x - 1/x)^8 else -Real.sqrt x

-- Define the composition f[f(x)]
noncomputable def f_composed (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem constant_term_f_composed (x : ℝ) (h : x > 0) :
  ∃ (expansion : ℝ → ℝ),
    (f_composed x = expansion x) ∧
    (∃ (constant_term : ℝ),
      constant_term = 70 ∧
      (∀ ε > 0, ∃ δ > 0, ∀ y, |y| < δ → |expansion y - constant_term| < ε)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_f_composed_l796_79634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l796_79612

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 2)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x y, x ∈ Set.Icc 0 (Real.pi / 2) → y ∈ Set.Icc 0 (Real.pi / 2) → x < y → f x < f y) ∧
  (∀ x, f x = f (-x)) ∧
  ¬(∀ x, f (-x) = -f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l796_79612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sammy_math_homework_pages_l796_79622

theorem sammy_math_homework_pages 
  (total_pages : ℕ) 
  (science_project_percentage : ℚ) 
  (remaining_pages : ℕ) 
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 1/4)
  (h3 : remaining_pages = 80) :
  total_pages - (science_project_percentage * ↑total_pages).floor - remaining_pages = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sammy_math_homework_pages_l796_79622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l796_79610

/-- Sequence a_n with sum of first n terms S_n = (1/2)(3^n - 1) -/
def S (n : ℕ+) : ℚ := (1/2) * ((3 : ℚ)^(n : ℕ) - 1)

/-- Arithmetic sequence b_n -/
def b (n : ℕ+) : ℚ := 2 * (n : ℚ) + 1

/-- Sum of first n terms of sequence a_n + b_n -/
def T (n : ℕ+) : ℚ := ((3 : ℚ)^(n : ℕ))/2 + (n : ℚ)^2 + 2*(n : ℚ) - 1/2

theorem sequence_sum_theorem (n : ℕ+) :
  (∀ k : ℕ+, b k > 0) ∧
  (b 1 + b 2 + b 3 = 15) ∧
  (∃ r : ℚ, r > 0 ∧ (S 1 + b 1) * r = S 2 + b 2 ∧ (S 2 + b 2) * r = S 3 + b 3) →
  T n = ((3 : ℚ)^(n : ℕ))/2 + (n : ℚ)^2 + 2*(n : ℚ) - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l796_79610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_remove_all_pieces_l796_79646

/-- Represents a position on the triangular board -/
inductive Position
  | Vertex : Position
  | Edge : Position
  | Center : Position

/-- Represents the color of a piece -/
inductive Color
  | Black : Color
  | White : Color

/-- Represents the game state -/
structure GameState where
  pieces : Position → Option Color

/-- Initial game state -/
def initialState : GameState :=
  { pieces := fun pos => match pos with
    | Position.Vertex => some Color.Black
    | _ => some Color.White }

/-- Checks if two positions are neighbors -/
def isNeighbor : Position → Position → Bool :=
  sorry

/-- Performs a move in the game -/
def makeMove (state : GameState) (pos : Position) : Option GameState :=
  sorry

/-- Theorem stating that it's impossible to remove all pieces -/
theorem cannot_remove_all_pieces :
  ∀ (finalState : GameState),
  (∃ (moves : List Position), List.foldl (fun s p => (makeMove s p).getD s) initialState moves = finalState) →
  (∃ (pos : Position), finalState.pieces pos ≠ none) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_remove_all_pieces_l796_79646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l796_79627

open Real

theorem problem_statement (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : cos α * sqrt ((1 - sin α) / (1 + sin α)) + sin α * sqrt ((1 - cos α) / (1 + cos α)) = 3 / 5) :
  (sin α) / (1 + cos α) + (cos α) / (1 + sin α) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l796_79627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l796_79650

theorem angle_values (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.cos β = -1/3)
  (h4 : Real.sin (α + β) = (4 - Real.sqrt 2) / 6) :
  Real.tan β ^ 2 = 4 * Real.sqrt 2 / 7 ∧ α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l796_79650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_equals_23_l796_79686

open Real

-- Define the sum of sin^2 from 0° to 90° in 2° increments
noncomputable def sin_squared_sum : ℝ :=
  (Finset.range 46).sum (fun i => (sin (2 * i * π / 180)) ^ 2)

-- Theorem statement
theorem sin_squared_sum_equals_23 : sin_squared_sum = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_equals_23_l796_79686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_distance_problem_l796_79640

-- Define polar coordinates
structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the line equation
def lineEquation (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = 3

-- Define the problem statement
theorem polar_distance_problem 
  (A : PolarPoint) 
  (B : PolarPoint) 
  (h1 : A.r = 3 ∧ A.θ = Real.pi/4) 
  (h2 : B.r = Real.sqrt 2 ∧ B.θ = Real.pi/2) :
  -- Part 1: Distance between A and B
  Real.sqrt ((A.r^2 + B.r^2) - 2*A.r*B.r*(Real.cos (B.θ - A.θ))) = Real.sqrt 5 ∧
  -- Part 2: Distance from B to the line
  (3*Real.sqrt 2 - Real.sqrt 2) * Real.sin (3*Real.pi/4 - Real.pi/2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_distance_problem_l796_79640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l796_79631

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 2*x + 3*y = 0

-- Define the foci
def left_focus (F₁ : ℝ × ℝ) : Prop := True
def right_focus (F₂ : ℝ × ℝ) : Prop := True

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a P.1 P.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem hyperbola_foci_distance 
  (a : ℝ) 
  (F₁ F₂ P : ℝ × ℝ) :
  left_focus F₁ →
  right_focus F₂ →
  point_on_hyperbola P a →
  asymptote P.1 P.2 →
  distance P F₁ = 7 →
  (distance P F₂ = 1 ∨ distance P F₂ = 13) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l796_79631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_portion_approx_l796_79617

/-- The circle defined by the equation x^2 - 16x + y^2 - 8y = 32 -/
def circle_equation (x y : ℝ) : Prop := x^2 - 16*x + y^2 - 8*y = 32

/-- The line defined by the equation y = 2x - 20 -/
def line_equation (x y : ℝ) : Prop := y = 2*x - 20

/-- The area of the portion of the circle below the x-axis and to the left of the line -/
noncomputable def area_portion : ℝ := sorry

theorem area_portion_approx : ∃ (ε : ℝ), ε > 0 ∧ |area_portion - 20 * Real.pi| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_portion_approx_l796_79617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l796_79688

/-- Given four circles A, B, C, and D, where:
    - A, B, and C are externally tangent to each other
    - A, B, and C are internally tangent to D
    - B and C are congruent
    - A has radius 2
    - A is tangent to an external circle E
    - The center of E is the same as the center of D
    This theorem proves that the radius of B is as given by the formula. -/
theorem circle_radius_problem (r : ℝ) (h : r ≠ 2) :
  let radius_B := (-1 + Real.sqrt (1 + 2*r) + ((-1 + Real.sqrt (1 + 2*r))^2 / 4)) / 4
  ∃ (A B C D E : Set (ℝ × ℝ)) (center_D center_E : ℝ × ℝ),
    (∀ p : ℝ × ℝ, p ∈ A → (p.1 - 0)^2 + (p.2 - 0)^2 = 2^2) ∧
    (∀ p : ℝ × ℝ, p ∈ B → (p.1 - center_D.1)^2 + (p.2 - center_D.2)^2 = radius_B^2) ∧
    (∀ p : ℝ × ℝ, p ∈ C → (p.1 - center_D.1)^2 + (p.2 - center_D.2)^2 = radius_B^2) ∧
    (∀ p : ℝ × ℝ, p ∈ D → (p.1 - center_D.1)^2 + (p.2 - center_D.2)^2 = r^2) ∧
    (∀ p : ℝ × ℝ, p ∈ E → (p.1 - center_E.1)^2 + (p.2 - center_E.2)^2 = (r + 2)^2) ∧
    center_D = center_E ∧
    A ∩ B = {(2 + radius_B, 0)} ∧
    A ∩ C = {(-2 - radius_B, 0)} ∧
    B ∩ C = {(0, 2 * radius_B)} ∧
    A ∩ D = {(r - 2, 0)} ∧
    B ∩ D = {(r - radius_B, 0)} ∧
    C ∩ D = {(-r + radius_B, 0)} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l796_79688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_increases_with_n_l796_79630

-- Define the variables and their properties
variable (d R r : ℝ)
variable (n : ℝ)

-- Define the conditions
axiom d_pos : d > 0
axiom R_pos : R > 0
axiom r_pos : r > 0
axiom n_pos : n > 0
axiom R_gt_nr : R > n * r

-- Define P as a function of n
noncomputable def P (n : ℝ) : ℝ := (2 * d * n) / (R - n * r)

-- State the theorem
theorem P_increases_with_n : 
  ∀ n₁ n₂, n₁ > 0 → n₂ > 0 → R > n₁ * r → R > n₂ * r → n₁ < n₂ → P d R r n₁ < P d R r n₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_increases_with_n_l796_79630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l796_79666

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (x + φ)

theorem f_decreasing_in_interval (φ : ℝ) 
  (h1 : 0 < |φ| ∧ |φ| < Real.pi / 2) 
  (h2 : ∀ x, f (x + Real.pi / 4) φ = - f (-x - Real.pi / 4) φ) :
  ∀ x ∈ Set.Ioo 0 (Real.pi / 4), 
    ∀ y ∈ Set.Ioo 0 (Real.pi / 4), 
      x < y → f x φ > f y φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l796_79666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_4x4_l796_79615

/-- A coloring of a 4x4 grid -/
def Coloring := Fin 4 → Fin 4 → ℕ

/-- A valid coloring satisfies the condition that for each pair of different colors,
    there exist two cells of these colors in the same row or column -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ color1 color2, color1 ≠ color2 →
    (∃ i j1 j2, j1 ≠ j2 ∧ c i j1 = color1 ∧ c i j2 = color2) ∨
    (∃ j i1 i2, i1 ≠ i2 ∧ c i1 j = color1 ∧ c i2 j = color2)

/-- The number of colors used in a coloring -/
def num_colors (c : Coloring) : ℕ :=
  Finset.card (Finset.image (fun p => c p.1 p.2) (Finset.univ.product Finset.univ))

/-- The main theorem: The maximum number of colors in a valid 4x4 coloring is 8 -/
theorem max_colors_4x4 :
  (∃ c : Coloring, is_valid_coloring c ∧ num_colors c = 8) ∧
  (∀ c : Coloring, is_valid_coloring c → num_colors c ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_4x4_l796_79615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_minus_f_condition_implies_a_bound_l796_79692

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x + a * x^2 - x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.exp x + 2 * a * x - 1

-- State the theorem
theorem f_derivative_minus_f_condition_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f_derivative a x - f a x ≥ (4 * a + 1) * x) →
  a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_minus_f_condition_implies_a_bound_l796_79692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_second_quadrant_tan_half_l796_79637

theorem cos_second_quadrant_tan_half (α : Real) :
  α ∈ Set.Icc (π / 2) π → Real.tan α = 1 / 2 → Real.cos α = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_second_quadrant_tan_half_l796_79637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_study_session_l796_79645

structure Student where
  id : ℕ
deriving Repr, DecidableEq

structure FriendshipGraph where
  students : Finset Student
  friendships : Finset (Student × Student)

def directFriends (g : FriendshipGraph) (s : Student) : Finset Student :=
  g.students.filter (fun t => (s, t) ∈ g.friendships ∨ (t, s) ∈ g.friendships)

def friendsOfFriends (g : FriendshipGraph) (s : Student) : Finset Student :=
  (directFriends g s).biUnion (fun f => directFriends g f)

def invitedStudents (g : FriendshipGraph) (mia : Student) : Finset Student :=
  {mia} ∪ directFriends g mia ∪ friendsOfFriends g mia

theorem mia_study_session (g : FriendshipGraph) (mia : Student) :
  g.students.card = 25 →
  (invitedStudents g mia).card = 17 →
  (g.students \ invitedStudents g mia).card = 8 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_study_session_l796_79645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l796_79603

theorem coefficient_x_cubed_in_expansion : ∃ (c : ℕ), c = 160 ∧ 
  (Polynomial.expand ℕ 6 (1 + 2 * Polynomial.X)).coeff 3 = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l796_79603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_a_ln_b_l796_79695

theorem max_value_a_ln_b (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b = Real.exp 2) :
  ∃ (max : ℝ), max = Real.exp 1 ∧ ∀ x, x = a ^ Real.log b → x ≤ max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_a_ln_b_l796_79695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_asymptotes_period_l796_79649

theorem csc_asymptotes_period (b : ℝ) : 
  (∀ x : ℝ, x ∈ ({-2 * Real.pi, 2 * Real.pi} : Set ℝ) → ∃ k : ℤ, b * x = k * Real.pi) →
  b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_asymptotes_period_l796_79649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sansa_small_portraits_count_l796_79604

/-- Represents the daily sales and pricing of Sansa's portraits --/
structure PortraitSales where
  small_price : ℚ
  large_price : ℚ
  large_count : ℕ
  total_earnings : ℚ
  days : ℕ

/-- Calculates the number of small portraits sold per day --/
def small_portraits_per_day (s : PortraitSales) : ℚ :=
  ((s.total_earnings / s.days) - (s.large_price * s.large_count)) / s.small_price

/-- Theorem stating that Sansa sells 3 small portraits per day --/
theorem sansa_small_portraits_count :
  let s : PortraitSales := {
    small_price := 5,
    large_price := 10,
    large_count := 5,
    total_earnings := 195,
    days := 3
  }
  small_portraits_per_day s = 3 := by
  -- The proof goes here
  sorry

#eval small_portraits_per_day {
  small_price := 5,
  large_price := 10,
  large_count := 5,
  total_earnings := 195,
  days := 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sansa_small_portraits_count_l796_79604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_center_l796_79690

/-- Apollonius Circle Theorem -/
theorem apollonius_circle_center :
  ∀ (M : ℝ × ℝ),
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (3, 0)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M O / dist M A = 1 / 2) →
  ∃ (center : ℝ × ℝ) (r : ℝ),
    center = (-1, 0) ∧
    r > 0 ∧
    ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 ↔
      dist (x, y) O / dist (x, y) A = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_center_l796_79690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l796_79654

-- Define the function f(x) = log(3-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (3 - x)

-- Theorem stating that the domain of f is (-∞, 3)
theorem domain_of_f : Set.Iio 3 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l796_79654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridget_final_skittles_count_l796_79669

/-- Represents the number of Skittles a person has -/
structure SkittlesCount where
  count : Nat

/-- Calculates the final number of Skittles Bridget has after Henry gives her his Skittles -/
def final_skittles_count (bridget_initial : SkittlesCount) (henry_initial : SkittlesCount) : SkittlesCount :=
  ⟨bridget_initial.count + henry_initial.count⟩

/-- Theorem stating that Bridget's final Skittles count is the sum of their initial counts -/
theorem bridget_final_skittles_count 
  (bridget_initial : SkittlesCount) 
  (henry_initial : SkittlesCount) : 
  (final_skittles_count bridget_initial henry_initial).count = bridget_initial.count + henry_initial.count := by
  rfl

def bridget_initial : SkittlesCount := ⟨4⟩
def henry_initial : SkittlesCount := ⟨4⟩

#eval (final_skittles_count bridget_initial henry_initial).count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridget_final_skittles_count_l796_79669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l796_79642

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (5, f 5)

/-- The distance from the vertex to the origin -/
noncomputable def vertex_to_origin_distance : ℝ := Real.sqrt ((vertex.1)^2 + (vertex.2)^2)

/-- The side length of the inscribed square -/
noncomputable def side_length : ℝ := vertex_to_origin_distance

/-- The x-coordinate of the left side of the square -/
noncomputable def left_x : ℝ := 5 - side_length / 2

/-- The x-coordinate of the right side of the square -/
noncomputable def right_x : ℝ := 5 + side_length / 2

/-- Theorem: The area of the inscribed square is 41 -/
theorem inscribed_square_area : 
  side_length^2 = 41 ∧ 
  f left_x ≥ 0 ∧ 
  f right_x ≥ 0 ∧ 
  f left_x = f right_x ∧
  f left_x = side_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l796_79642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_special_l796_79609

theorem sine_double_angle_special (α : ℝ) :
  Real.cos (π / 3 - α) = 2 * Real.cos (α + π / 6) →
  Real.sin (2 * α + π / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_special_l796_79609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l796_79620

/-- Proves that given a journey of 120 miles in 120 minutes, where the average speed for the first 40 minutes is 50 mph and the second 40 minutes is 45 mph, the average speed for the last 40 minutes is 85 mph. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
    (speed_first_segment : ℝ) (speed_second_segment : ℝ) 
    (h1 : total_distance = 120)
    (h2 : total_time = 120)
    (h3 : speed_first_segment = 50)
    (h4 : speed_second_segment = 45)
    : (3 * (total_distance / total_time)) - speed_first_segment - speed_second_segment = 85 := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l796_79620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_cube_root_of_28_l796_79678

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (x : ℝ) : Prop :=
  log 5 (x - 3) + log (Real.sqrt 5) (x^3 - 3) + log (1/5) (x - 3) = 4

-- Theorem statement
theorem solution_is_cube_root_of_28 :
  ∃ x : ℝ, x > 0 ∧ equation x ∧ x = Real.rpow 28 (1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_cube_root_of_28_l796_79678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_over_a_range_l796_79664

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (x - a)^3 * (x - b)

-- Define the function g_k
noncomputable def g_k (a b k x : ℝ) : ℝ := (f a b x - f a b k) / (x - k)

-- Theorem statement
theorem b_over_a_range (a b : ℝ) :
  (0 < a) →
  (a < b) →
  (b < 1) →
  (∀ (k : ℤ), StrictMono (fun x => g_k a b (k : ℝ) x)) →
  (1 < b / a) ∧ (b / a ≤ 3) := by
  sorry

#check b_over_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_over_a_range_l796_79664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l796_79613

-- Define the function f(x) = (x - 1) / (x + 2)
noncomputable def f (x : ℝ) := (x - 1) / (x + 2)

-- Define the closed interval [3, 5]
def I : Set ℝ := Set.Icc 3 5

-- Theorem statement
theorem f_properties :
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x < f y) ∧
  (∀ x ∈ I, f x ≤ f 5) ∧
  (∀ x ∈ I, f 3 ≤ f x) ∧
  f 5 = 4/7 ∧
  f 3 = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l796_79613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_2theta_l796_79683

theorem tan_pi_4_minus_2theta (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.tan (π / 4 - 2 * θ) = - (1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_2theta_l796_79683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l796_79636

/-- Two lines are parallel -/
def parallel (l₁ l₂ : ℝ → ℝ) : Prop := ∃ c, ∀ x, l₂ x = l₁ x + c

/-- A line passes through a point -/
def passes_through (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop := l p.1 = p.2

/-- The y-intercept of a line -/
def y_intercept (l : ℝ → ℝ) : ℝ := l 0

theorem line_intersection_y_axis 
  (l₁ l₂ : ℝ → ℝ) 
  (h₁ : ∀ x, l₁ x = 2 * x + b₁) -- slope of l₁ is 2
  (h₂ : parallel l₁ l₂)
  (h₃ : passes_through l₂ (-1, 1)) :
  y_intercept l₂ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l796_79636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_union_l796_79697

/-- The function f_n for a given positive integer n -/
noncomputable def f (n : ℕ+) (x : ℝ) : ℝ := (n + x + 1 / (n + x)) / (n + 1)

/-- The range of f_n for a given positive integer n -/
def I (n : ℕ+) : Set ℝ := {y | ∃ x ∈ Set.Ioo 0 1, f n x = y}

/-- The union of all I_n for n from 1 to ∞ -/
def I_union : Set ℝ := ⋃ n : ℕ+, I n

/-- The main theorem stating the range of the union of all f_n -/
theorem range_of_f_union :
  I_union = Set.Ioo (5/6 : ℝ) (5/4 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_union_l796_79697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_papers_to_contents_ratio_l796_79611

/-- The weight of Karen's tote bag in pounds -/
noncomputable def karens_tote_weight : ℝ := 8

/-- The weight of Kevin's empty briefcase in pounds -/
noncomputable def kevins_empty_briefcase_weight : ℝ := karens_tote_weight / 2

/-- The weight of Kevin's laptop in pounds -/
noncomputable def kevins_laptop_weight : ℝ := karens_tote_weight + 2

/-- The weight of Kevin's full briefcase in pounds -/
noncomputable def kevins_full_briefcase_weight : ℝ := 2 * karens_tote_weight

/-- The weight of Kevin's work papers in pounds -/
noncomputable def kevins_work_papers_weight : ℝ := kevins_full_briefcase_weight - kevins_laptop_weight

/-- The weight of the contents of Kevin's full briefcase in pounds -/
noncomputable def kevins_full_briefcase_contents_weight : ℝ := kevins_laptop_weight + kevins_work_papers_weight

theorem work_papers_to_contents_ratio :
  kevins_work_papers_weight / kevins_full_briefcase_contents_weight = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_papers_to_contents_ratio_l796_79611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_equals_three_l796_79638

-- Define the square and circles
def square_side_length : ℝ := 3

-- Define the circle radius
def circle_radius : ℝ → Prop := λ r => 1.5 < r ∧ r < 2.5

-- Define the centers of the circles
def center_A : ℝ × ℝ := (0, 0)
def center_B : ℝ × ℝ := (square_side_length, 0)
def center_D : ℝ × ℝ := (0, square_side_length)

-- Define the circles
def circle_A (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - center_A.1)^2 + (p.2 - center_A.2)^2 = r^2}
def circle_B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - center_B.1)^2 + (p.2 - center_B.2)^2 = r^2}
def circle_D (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - center_D.1)^2 + (p.2 - center_D.2)^2 = r^2}

-- Define points P and Q
noncomputable def point_P (r : ℝ) : ℝ × ℝ := (square_side_length / 2, Real.sqrt (r^2 - (square_side_length / 2)^2))
noncomputable def point_Q (r : ℝ) : ℝ × ℝ := (Real.sqrt (r^2 - (square_side_length / 2)^2), square_side_length / 2)

-- Theorem statement
theorem length_PQ_equals_three (r : ℝ) (hr : circle_radius r) :
  Real.sqrt ((point_P r).1 - (point_Q r).1)^2 + ((point_P r).2 - (point_Q r).2)^2 = square_side_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_equals_three_l796_79638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_max_product_l796_79693

noncomputable def rope_length : ℝ := 10

noncomputable def square_area (x : ℝ) : ℝ := x^2 / 16

noncomputable def triangle_area (y : ℝ) : ℝ := y^2 / 24

noncomputable def sum_of_areas (x : ℝ) : ℝ := square_area x + triangle_area (rope_length - x)

noncomputable def product_of_areas (x : ℝ) : ℝ := square_area x * triangle_area (rope_length - x)

theorem min_sum_max_product :
  (∃ x : ℝ, x ≥ 0 ∧ x ≤ rope_length ∧ sum_of_areas x = (5 : ℝ) / 2) ∧
  (∃ x : ℝ, x ≥ 0 ∧ x ≤ rope_length ∧ product_of_areas x = 625 / 384) ∧
  (∀ x : ℝ, x ≥ 0 → x ≤ rope_length → sum_of_areas x ≥ (5 : ℝ) / 2) ∧
  (∀ x : ℝ, x ≥ 0 → x ≤ rope_length → product_of_areas x ≤ 625 / 384) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_max_product_l796_79693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quote_calculation_l796_79659

/-- Calculate the stock quote given the specified parameters --/
noncomputable def calculate_stock_quote (stock_yield : ℝ) (dividend_yield : ℝ) (dividend_tax : ℝ) 
                          (inflation_rate : ℝ) (risk_premium : ℝ) : ℝ :=
  let after_tax_yield := dividend_yield * (1 - dividend_tax)
  let real_yield := (1 + after_tax_yield) / (1 + inflation_rate) - 1
  let required_yield := real_yield + risk_premium
  let face_value := 100
  let dividend_per_share := face_value * stock_yield
  dividend_per_share / required_yield

/-- Theorem stating that the calculated stock quote is approximately $160.64 --/
theorem stock_quote_calculation :
  let stock_yield := 0.16
  let dividend_yield := 0.14
  let dividend_tax := 0.20
  let inflation_rate := 0.03
  let risk_premium := 0.02
  abs (calculate_stock_quote stock_yield dividend_yield dividend_tax inflation_rate risk_premium - 160.64) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quote_calculation_l796_79659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l796_79644

/-- A quadrilateral with side lengths a, b, c, and d in sequence is a parallelogram 
    if a^2 + b^2 + c^2 + d^2 - 2ac - 2bd = 0 -/
theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) : 
  a = c ∧ b = d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l796_79644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_f_range_l796_79658

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 4), 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 4), (Real.cos (x / 4))^2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Theorem 1
theorem cos_value (x : ℝ) (h : f x = 1) : 
  Real.cos (2 * Real.pi / 3 - x) = -1/2 := by
  sorry

-- Theorem 2
theorem f_range (A B C : ℝ) (a b c : ℝ) 
  (h : a * Real.cos C + c / 2 = b) :
  ∃ (l u : ℝ), l = 1 ∧ u = 3/2 ∧ l < f B ∧ f B < u := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_f_range_l796_79658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l796_79665

/-- Given complex numbers z₁, z₂, z₃ satisfying specific conditions, prove |z₃| = (40√3)/13 -/
theorem complex_magnitude (z₁ z₂ z₃ : ℂ) (k : ℝ) 
  (h1 : z₁ = -k • z₂)
  (h2 : k > 0)
  (h3 : Complex.abs (z₁ - z₂) = 13)
  (h4 : Complex.abs z₁ ^ 2 + Complex.abs z₃ ^ 2 + Complex.abs (z₁ * z₃) = 144)
  (h5 : Complex.abs z₂ ^ 2 + Complex.abs z₃ ^ 2 - Complex.abs (z₂ * z₃) = 25) :
  Complex.abs z₃ = 40 * Real.sqrt 3 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l796_79665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l796_79602

/-- The time it takes for two trains to cross each other -/
noncomputable def train_crossing_time (train_length : ℝ) (faster_speed : ℝ) : ℝ :=
  let slower_speed := faster_speed / 2
  let total_distance := 2 * train_length
  let relative_speed := slower_speed + faster_speed
  total_distance / relative_speed

/-- Theorem: Given two trains of length 100 m each, moving in opposite directions, 
    with one train moving twice as fast as the other, and the faster train moving 
    at 60.00000000000001 m/s, the time it takes for the trains to cross each other 
    is approximately 2.2222222222222223 seconds. -/
theorem train_crossing_theorem : 
  ∀ ε > 0, |train_crossing_time 100 60.00000000000001 - 2.2222222222222223| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l796_79602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_problem_l796_79663

theorem function_value_problem (f : ℝ → ℝ) :
  (∀ x, f (Real.cos x) = Real.cos (2 * x)) →
  f (-1/2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_problem_l796_79663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_exponent_equation_l796_79682

theorem integer_exponent_equation (m : ℤ) : ((-2 : ℚ) ^ (2 * m) = 2 ^ (18 - m)) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_exponent_equation_l796_79682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l796_79698

-- Define the line
def line (a b x y : ℝ) : Prop := x / a + y / b = 1

-- Define the circle (renamed to avoid conflict)
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the distance from a point to the origin
noncomputable def distance_to_origin (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- State the theorem
theorem min_distance_to_origin (a b : ℝ) :
  (∃ x y, line a b x y ∧ unit_circle x y) →
  distance_to_origin a b ≥ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l796_79698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l796_79601

/-- Iterated function application -/
def iter (f : ℕ+ → ℕ+) : ℕ → ℕ+ → ℕ+
  | 0, n => n
  | k+1, n => f (iter f k n)

/-- The property that f and g must satisfy -/
def satisfies (f g : ℕ+ → ℕ+) :=
  ∀ n : ℕ+, iter f (g n) n + iter g (f n) n = f (n + 1) - g (n + 1) + 1

/-- The main theorem -/
theorem unique_solution (f g : ℕ+ → ℕ+) (h : satisfies f g) :
  (∀ n : ℕ+, f n = n) ∧ (∀ n : ℕ+, g n = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l796_79601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l796_79619

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  x = 15/256 * y^2 - 381/128

-- Define the conditions
theorem parabola_properties :
  -- Passes through (2,8)
  parabola_equation 2 8 ∧
  -- Focus x-coordinate is 3
  (∃ (y : ℝ), parabola_equation 3 y ∧ 
    ∀ (x' y' : ℝ), parabola_equation x' y' → (x' - 3)^2 + y'^2 ≥ y^2) ∧
  -- Axis of symmetry parallel to y-axis
  (∀ (x y₁ y₂ : ℝ), parabola_equation x y₁ → parabola_equation x y₂ → y₁ = y₂ ∨ y₁ = -y₂) ∧
  -- Vertex on x-axis
  (∃ (x : ℝ), parabola_equation x 0 ∧ 
    ∀ (y : ℝ), y ≠ 0 → ¬parabola_equation x y) ∧
  -- Equation in standard form with integer coefficients and gcd = 1
  (∃ (a b c d e f : ℤ), c ≠ 0 ∧
    (∀ (x y : ℝ), parabola_equation x y ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0) ∧
    Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l796_79619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_g_3_singleton_l796_79672

-- Define the sequence of functions g_n
noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.sqrt (1 - x^2)
  | n + 1 => λ x => g n (Real.sqrt ((n + 2)^2 - x^2))

-- Define the domain of g_n
def domain (n : ℕ) : Set ℝ :=
  {x : ℝ | ∃ y, g n x = y}

-- State the theorem
theorem domain_g_3_singleton :
  (∃ M : ℕ, (∀ n > M, domain n = ∅) ∧
             domain M = {3} ∧
             (∀ n < M, domain n ≠ ∅)) ∧
  (∀ M : ℕ, (∀ n > M, domain n = ∅) ∧
             domain M = {3} ∧
             (∀ n < M, domain n ≠ ∅) →
             M = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_g_3_singleton_l796_79672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pork_price_increase_l796_79607

/-- Represents the monthly rate of increase in pork prices -/
def x : ℝ := sorry

/-- The initial price of pork in August -/
def initial_price : ℝ := 32

/-- The final price of pork in October -/
def final_price : ℝ := 64

/-- Theorem stating that the equation 32(1+x)^2 = 64 correctly represents
    the scenario where the price of pork doubles over two months with a
    consistent monthly rate of increase x -/
theorem pork_price_increase : initial_price * (1 + x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pork_price_increase_l796_79607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_parabola_intersection_l796_79675

/-- The line parameterized by φ --/
def line (φ : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t, x = t * Real.sin φ ∧ y = 1 + t * Real.cos φ}

/-- The parabola x^2 = 4y --/
def parabola : Set (ℝ × ℝ) :=
  {(x, y) | x^2 = 4*y}

/-- The minimum distance between intersection points of a line and a parabola --/
theorem min_distance_line_parabola_intersection :
  ∃ (min_dist : ℝ),
    min_dist = 4 ∧
    ∀ (φ : ℝ) (A B : ℝ × ℝ),
      0 < φ ∧ φ < π →
      (∀ (x y : ℝ), x * Real.cos φ - y * Real.sin φ + Real.sin φ = 0 ↔ (x, y) ∈ line φ) →
      (∀ (x y : ℝ), x^2 = 4*y ↔ (x, y) ∈ parabola) →
      A ∈ line φ ∧ A ∈ parabola →
      B ∈ line φ ∧ B ∈ parabola →
      A ≠ B →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_parabola_intersection_l796_79675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_is_46_l796_79657

/-- The speed of the faster train given the conditions of the problem -/
noncomputable def faster_train_speed (train_length : ℝ) (speed_difference : ℝ) (passing_time : ℝ) : ℝ :=
  speed_difference + (2 * train_length * 3600) / (speed_difference * passing_time)

/-- Theorem stating that under the given conditions, the speed of the faster train is 46 km/hr -/
theorem faster_train_speed_is_46 :
  faster_train_speed 0.025 36 18 = 46 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_is_46_l796_79657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_minimum_value_minimum_value_achieved_l796_79674

-- Part 1: Inequality solution
def solution_set (x : ℝ) : Prop := x ≤ -6 ∨ x ≥ 1

theorem inequality_solution :
  ∀ x : ℝ, |2*x + 5| ≥ 7 ↔ solution_set x := by sorry

-- Part 2: Minimum value
noncomputable def Z (x : ℝ) : ℝ := (x^2 + 5*x + 3) / x

theorem minimum_value :
  ∀ x : ℝ, x > 0 → Z x ≥ 2 * Real.sqrt 3 + 5 := by sorry

theorem minimum_value_achieved :
  ∃ x : ℝ, x > 0 ∧ Z x = 2 * Real.sqrt 3 + 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_minimum_value_minimum_value_achieved_l796_79674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfactory_percentage_approx_70_l796_79656

/-- Represents the grade distribution in a classroom --/
structure GradeDistribution where
  total : ℕ
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  f : ℕ

/-- Calculates the percentage of satisfactory grades --/
def satisfactoryPercentage (g : GradeDistribution) : ℚ :=
  (g.a + g.b + g.c + g.d : ℚ) / g.total * 100

/-- The classroom grade distribution --/
def classroomGrades : GradeDistribution :=
  { total := 36
  , a := 9
  , b := 7
  , c := 5
  , d := 4
  , f := 11 }

/-- Approximation tolerance --/
def ε : ℚ := 1/100

theorem satisfactory_percentage_approx_70 :
  abs (satisfactoryPercentage classroomGrades - 70) < ε :=
sorry

#eval satisfactoryPercentage classroomGrades

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfactory_percentage_approx_70_l796_79656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_focus_l796_79680

/-- A hyperbola with real-axis length m and a point P on it -/
structure Hyperbola (m : ℝ) where
  /-- P is a point on the hyperbola -/
  P : ℝ × ℝ
  /-- Right focus of the hyperbola -/
  right_focus : ℝ × ℝ
  /-- Left focus of the hyperbola -/
  left_focus : ℝ × ℝ
  /-- Distance from P to the right focus is m -/
  dist_right_focus : Real.sqrt ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) = m
  /-- The difference in distances from P to the foci is equal to the real-axis length -/
  focal_property : Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) - 
                   Real.sqrt ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) = m

/-- The distance from P to the left focus of the hyperbola is 2m -/
theorem distance_to_left_focus (m : ℝ) (h : Hyperbola m) : 
  Real.sqrt ((h.P.1 - h.left_focus.1)^2 + (h.P.2 - h.left_focus.2)^2) = 2 * m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_focus_l796_79680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisibility_l796_79660

open Set

theorem smallest_k_divisibility (S : Finset ℤ) : 
  (S.card = 1005) → 
  ∃ (x y : ℤ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ ((2007 ∣ (x + y)) ∨ (2007 ∣ (x - y))) ∧
  ∀ (T : Finset ℤ), T.card < 1005 → 
    ∃ (T' : Finset ℤ), T'.card = T.card ∧ 
      ∀ (a b : ℤ), a ∈ T' ∧ b ∈ T' ∧ a ≠ b → ¬(2007 ∣ (a + b)) ∧ ¬(2007 ∣ (a - b)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisibility_l796_79660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l796_79605

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 and 18, and a distance of 12 between them, is 228. -/
theorem trapezium_area_example : trapezium_area 20 18 12 = 228 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [add_mul, mul_div_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l796_79605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l796_79624

theorem complex_division_result : 
  (1 + Complex.I) / Complex.I = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l796_79624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_rolls_probability_l796_79600

def roll_probability : ℚ := 5 / 648

theorem six_rolls_probability : 
  (Nat.choose 6 4 * Nat.choose 2 1 : ℚ) * 
  (1 / 6 : ℚ)^4 * (1 / 6 : ℚ)^1 * ((4 / 6) : ℚ)^1 = roll_probability :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_rolls_probability_l796_79600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_possible_rankings_l796_79677

-- Define the players
inductive Player : Type
| P : Player
| Q : Player
| R : Player
| S : Player

-- Define a ranking sequence
def RankingSequence := List Player

-- Define the tournament structure
structure Tournament :=
  (saturday_match1 : Player × Player)
  (saturday_match2 : Player × Player)
  (sunday_winners_match : Player × Player)
  (sunday_losers_match : Player × Player)

def chess_tournament : Tournament :=
  { saturday_match1 := (Player.P, Player.Q),
    saturday_match2 := (Player.R, Player.S),
    sunday_winners_match := (Player.P, Player.R),  -- Example winners
    sunday_losers_match := (Player.Q, Player.S) }  -- Example losers

-- Function to generate all possible ranking sequences
def generate_ranking_sequences (t : Tournament) : List RankingSequence :=
  sorry

-- Theorem stating that there are exactly 8 possible ranking sequences
theorem eight_possible_rankings :
  (generate_ranking_sequences chess_tournament).length = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_possible_rankings_l796_79677
