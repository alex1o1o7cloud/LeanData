import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_divides_triangles_equal_area_l1089_108996

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

-- Define a line from a point through another point
noncomputable def lineThrough (start through : Point) : Line := sorry

-- Define the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

theorem centroid_divides_triangles_equal_area (ABC : Triangle) :
  let G := centroid ABC
  let ABG := Triangle.mk ABC.A ABC.B G
  let BAG := Triangle.mk ABC.B ABC.A G
  let BCG := Triangle.mk ABC.B ABC.C G
  let CBG := Triangle.mk ABC.C ABC.B G
  let CAG := Triangle.mk ABC.C ABC.A G
  let ACG := Triangle.mk ABC.A ABC.C G
  triangleArea ABG = triangleArea CAG ∧
  triangleArea BAG = triangleArea ACG ∧
  triangleArea BCG = triangleArea CBG :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_divides_triangles_equal_area_l1089_108996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_l1089_108939

/-- Definition of an ellipse with semi-major axis 4 and semi-minor axis 3 -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 9 = 1

/-- Definition of the foci of the ellipse -/
noncomputable def foci : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt 7
  (-c, 0, c, 0)

/-- Definition of a right-angled triangle formed by P, F1, and F2 -/
def forms_right_triangle (px py : ℝ) : Prop :=
  let (f1x, f1y, f2x, f2y) := foci
  (px - f1x)^2 + (py - f1y)^2 + (px - f2x)^2 + (py - f2y)^2 = (f2x - f1x)^2 + (f2y - f1y)^2

/-- Theorem: The distance from P to the x-axis is 9/4 -/
theorem distance_to_x_axis (px py : ℝ) :
  is_on_ellipse px py →
  forms_right_triangle px py →
  |py| = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_l1089_108939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_evaluation_l1089_108914

theorem complex_fraction_evaluation : 
  (⌈(18 : ℚ) / 8 - ⌈(28 : ℚ) / 18⌉⌉) / (⌈(28 : ℚ) / 8 + ⌈8 * (18 : ℚ) / 28⌉⌉) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_evaluation_l1089_108914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1089_108944

-- Define the slope of line m
noncomputable def slope_m : ℝ := -3

-- Define the point P that line l passes through
def point_P : ℝ × ℝ := (-1, 3)

-- Define the slope of line l (perpendicular to line m)
noncomputable def slope_l : ℝ := -1 / slope_m

-- Define the general form equation of line l
def line_l_equation (x y : ℝ) : Prop :=
  x - 3 * y + 10 = 0

-- Define the x-intercept of line l
noncomputable def x_intercept : ℝ := 10

-- Define the y-intercept of line l
noncomputable def y_intercept : ℝ := 10 / 3

-- Theorem statement
theorem triangle_area : 
  ∀ (x y : ℝ), 
  line_l_equation x y → 
  (1/2 : ℝ) * x_intercept * y_intercept = 50/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1089_108944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1089_108924

theorem remainder_problem (k : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k < 42)
  (h3 : k % 7 = 3) :
  k % 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1089_108924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1089_108940

-- Define the polar coordinate system
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

-- Define the line l
def line_l (α : ℝ) (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = ρ * Real.cos θ + 2

-- Define the curve C
def curve_C (ρ θ : ℝ) : Prop :=
  ρ = ρ * Real.cos θ + 2

-- Theorem statement
theorem intersection_point :
  ∃ (p : PolarCoord),
    line_l (π / 4) p.ρ p.θ ∧
    curve_C p.ρ p.θ ∧
    p.ρ = 2 ∧
    p.θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1089_108940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_outside_circle_l1089_108946

def circle_center : ℝ × ℝ := (0, 1)
def circle_radius : ℝ := 5

def point_A : ℝ × ℝ := (3, 4)
def point_B : ℝ × ℝ := (4, 5)
def point_C : ℝ × ℝ := (5, 1)
def point_D : ℝ × ℝ := (1, 5)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem only_B_outside_circle :
  distance point_B circle_center > circle_radius ∧
  distance point_A circle_center ≤ circle_radius ∧
  distance point_C circle_center ≤ circle_radius ∧
  distance point_D circle_center ≤ circle_radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_outside_circle_l1089_108946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1089_108973

def given_numbers : Set ℝ := {7, -3.14, -5, 1/8, 0, -5/3, 4.6, 3/4, -2}

def positive_numbers : Set ℝ := {7, 1/8, 4.6, 3/4}
def negative_numbers : Set ℝ := {-3.14, -5, -5/3, -2}
def integers : Set ℝ := {7, -5, 0, -2}
def fractions : Set ℝ := {-3.14, 1/8, -5/3, 4.6, 3/4}

theorem number_categorization :
  (∀ x ∈ positive_numbers, x > 0) ∧
  (∀ x ∈ negative_numbers, x < 0) ∧
  (∀ x ∈ integers, ∃ n : ℤ, x = n) ∧
  (∀ x ∈ fractions, ∃ p q : ℤ, q ≠ 0 ∧ x = p / q) ∧
  (positive_numbers ∪ negative_numbers ∪ integers = given_numbers) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1089_108973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l1089_108984

/-- Given vectors a, b, and c in ℝ², prove that if λa + b is collinear with c, then λ = -1 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, 0)) (h3 : c = (1, -2)) :
  (∃ l : ℝ, ∃ k : ℝ, k ≠ 0 ∧ (l • a.1 + b.1, l • a.2 + b.2) = k • c) → 
  (∃ l : ℝ, l = -1 ∧ ∃ k : ℝ, k ≠ 0 ∧ (l • a.1 + b.1, l • a.2 + b.2) = k • c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l1089_108984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_difference_l1089_108949

/-- Given a dress with an unknown original price, which is discounted by 15% to $78.2,
    and then increased by 25%, prove that the difference between the original price
    and the final price is $5.75. -/
theorem dress_price_difference :
  ∀ (original_price : ℝ),
  original_price * (1 - 0.15) = 78.2 →
  (78.2 * (1 + 0.25)) - original_price = 5.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_difference_l1089_108949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_through_focus_l1089_108930

/-- Given an ellipse with equation x²/16 + y²/9 = 1, a focus F at (c, 0) where c = √7, 
    and a chord AB passing through F such that AF = 2, there exists a point B on the ellipse 
    such that BF = √((x₂ - c)² + y₂²), where (x₂, y₂) are the coordinates of B. -/
theorem ellipse_chord_through_focus (A B : ℝ × ℝ) (F : ℝ × ℝ) :
  let c : ℝ := Real.sqrt 7
  let on_ellipse (p : ℝ × ℝ) := p.1^2/16 + p.2^2/9 = 1
  F = (c, 0) →
  on_ellipse A →
  on_ellipse B →
  (A.1 - F.1)^2 + A.2^2 = 4 →
  ∃ x₂ y₂ : ℝ, B = (x₂, y₂) ∧ 
    Real.sqrt ((x₂ - c)^2 + y₂^2) = 
      Real.sqrt ((B.1 - F.1)^2 + B.2^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_through_focus_l1089_108930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1089_108900

-- Define the parabola C
def Parabola (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ, C = {(x, y) : ℝ × ℝ | y^2 = 2*p*x}

-- Define the line y = x
def Line (L : Set (ℝ × ℝ)) : Prop :=
  L = {(x, y) : ℝ × ℝ | y = x}

-- Define the intersection points A and B
def Intersection (C L : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ C ∧ A ∈ L ∧ B ∈ C ∧ B ∈ L ∧ A ≠ B

-- Define the midpoint P
def Midpoint (A B P : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

theorem parabola_equation (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) (A B P : ℝ × ℝ) :
  Parabola C →
  Line L →
  Intersection C L A B →
  Midpoint A B P →
  P = (2, 2) →
  C = {(x, y) : ℝ × ℝ | y^2 = 4*x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1089_108900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rational_values_l1089_108937

theorem count_rational_values (p q : ℕ) (x : ℚ) : 
  (∃ (s : Finset ℚ), 
    s.card = 5 ∧ 
    (∀ y ∈ s, 0 ≤ y ∧ y ≤ 1 ∧ 
      (∃ (p q : ℕ), y = q / p ∧ Nat.Coprime q p ∧ p ≥ 2 ∧ 1 / p > 1 / 5))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rational_values_l1089_108937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1089_108962

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)

theorem triangle_properties (t : Triangle) 
  (h_area : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2))
  (h_angles : t.A + t.B + t.C = π)
  (h_positive : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h_angle_range : 0 < t.A ∧ t.A < 2*π/3 ∧ 0 < t.B ∧ t.B < π ∧ 0 < t.C ∧ t.C < π) :
  t.B = π/3 ∧ 
  (t.b = Real.sqrt 3 → Real.sqrt 3 < t.a + 2*t.c ∧ t.a + 2*t.c ≤ 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1089_108962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l1089_108913

theorem greatest_integer_fraction : 
  ⌊(5^50 + 4^50 : ℝ) / (5^48 + 4^48 : ℝ)⌋ = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l1089_108913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_AD_length_l1089_108969

/-- A regular octagon inscribed in a circle with side length s -/
structure RegularOctagon :=
  (s : ℝ)
  (s_pos : s > 0)

/-- The length of diagonal AD in a regular octagon -/
noncomputable def diagonalAD (octagon : RegularOctagon) : ℝ :=
  octagon.s * (1 + Real.sqrt 2 / 2)^(1/2) / Real.sqrt (2 - Real.sqrt 2)

/-- Theorem stating the length of diagonal AD in a regular octagon -/
theorem diagonal_AD_length (octagon : RegularOctagon) :
  diagonalAD octagon = octagon.s * (1 + Real.sqrt 2 / 2)^(1/2) / Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_AD_length_l1089_108969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l1089_108929

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations and operations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (plane_intersect : Plane → Plane → Line → Prop)

-- Theorem statement
theorem geometry_propositions 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) :
  (¬ ∀ (α β : Plane) (m n : Line), 
    subset m α ∧ subset n α ∧ parallel m n ∧ parallel m n → parallel_plane α β) ∧
  (¬ ∀ (α : Plane) (m n : Line), 
    subset m α ∧ ¬ subset n α ∧ skew m n → intersect n α) ∧
  (∀ (α β : Plane) (m n : Line), 
    plane_intersect α β m ∧ parallel n m ∧ ¬ subset n α ∧ ¬ subset n β → 
    parallel_plane α β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l1089_108929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacks_savings_l1089_108981

/-- Represents the fraction of allowance Jack puts into his piggy bank -/
def fraction_saved : ℚ → Prop := λ _ => True

/-- The initial amount in Jack's piggy bank -/
def initial_amount : ℕ := 43

/-- Jack's weekly allowance -/
def weekly_allowance : ℕ := 10

/-- The number of weeks -/
def num_weeks : ℕ := 8

/-- The final amount in Jack's piggy bank -/
def final_amount : ℕ := 83

theorem jacks_savings (f : ℚ) :
  fraction_saved f →
  (f * (weekly_allowance * num_weeks : ℚ) + initial_amount : ℚ) = final_amount →
  f = 1/2 := by
  sorry

#check jacks_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacks_savings_l1089_108981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_enclosure_l1089_108909

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set in ℝ²
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define what it means for one set to enclose another
def encloses (A B : Set (ℝ × ℝ)) : Prop := sorry

-- Define a parallelogram
def Parallelogram (P : Set (ℝ × ℝ)) : Prop := sorry

theorem convex_polygon_enclosure :
  ∀ (M : Set (ℝ × ℝ)),
  ConvexPolygon M →
  area M = 1 →
  ∃ (P : Set (ℝ × ℝ)),
    Parallelogram P ∧
    encloses P M ∧
    area P ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_enclosure_l1089_108909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_iff_valid_a_l1089_108918

/-- The function f(x) = log_a(x^2 + ax + 4) has no minimum value. -/
def has_no_minimum (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, Real.log (x^2 + a*x + 4) / Real.log a < y

/-- The set of all possible values of a such that f(x) has no minimum value. -/
def valid_a_set : Set ℝ :=
  {a : ℝ | 0 < a ∧ a ≠ 1 ∧ (0 < a ∧ a < 1 ∨ a ≥ 4)}

theorem no_minimum_iff_valid_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  has_no_minimum a ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_iff_valid_a_l1089_108918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circle_coverage_l1089_108933

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The length of side AB
  α : ℝ
  -- The measure of angle DAB in radians
  angleDAB : ℝ
  -- Assertion that ABCD is a parallelogram
  is_parallelogram : Bool
  -- Assertion that AD = 1
  ad_equals_one : Bool
  -- Assertion that AB = α
  ab_equals_α : Bool
  -- Assertion that α is the measure of angle DAB
  α_is_angleDAB : Bool
  -- Assertion that the three angles of triangle ABD are acute
  abd_angles_acute : Bool

/-- Definition of circle coverage for a parallelogram -/
def circles_cover_parallelogram (p : SpecialParallelogram) : Prop :=
  ∃ (KA KB KC KD : Set (ℝ × ℝ)),
    (∀ point : ℝ × ℝ, point ∈ KA ∪ KB ∪ KC ∪ KD → 
      point.1^2 + point.2^2 ≤ 1) ∧
    (∀ point : ℝ × ℝ, (point ∈ Set.univ : Prop) → point ∈ KA ∪ KB ∪ KC ∪ KD)

/-- The main theorem to be proved -/
theorem parallelogram_circle_coverage (p : SpecialParallelogram) :
  circles_cover_parallelogram p ↔ p.α ≤ Real.cos p.α + Real.sqrt 3 * Real.sin p.α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circle_coverage_l1089_108933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_negative_one_l1089_108950

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2 else -x + 1

-- Theorem statement
theorem f_composite_negative_one : f (f (-1)) = 6 := by
  -- Evaluate f(-1)
  have h1 : f (-1) = 2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(2)
  have h2 : f 2 = 6 := by
    simp [f]
    norm_num
  
  -- Combine the steps
  calc
    f (f (-1)) = f 2 := by rw [h1]
    _          = 6   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_negative_one_l1089_108950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l1089_108921

theorem rhombus_side_length (R : ℝ) (x : ℝ) :
  (x^2 * (Real.sqrt 3 / 2) = π * R^2) →
  x = R * Real.sqrt (2 * π / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l1089_108921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_specific_line_l1089_108905

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂) is (y₂ - y₁) / (x₂ - x₁) -/
def lineSlope (x₁ y₁ x₂ y₂ : ℚ) : ℚ := (y₂ - y₁) / (x₂ - x₁)

/-- The slope of the line passing through (2, -3) and (-3, 4) is -7/5 -/
theorem slope_of_specific_line :
  lineSlope 2 (-3) (-3) 4 = -7/5 := by
  -- Unfold the definition of lineSlope
  unfold lineSlope
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_specific_line_l1089_108905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_face_product_l1089_108925

-- Define a cube as a set of 12 edges
def Cube := Fin 12

-- Define a valid arrangement of numbers on a cube
def ValidArrangement (arr : Cube → ℕ) : Prop :=
  (∀ i : Cube, arr i ∈ Finset.range 13 \ {0}) ∧ 
  (∀ i j : Cube, i ≠ j → arr i ≠ arr j)

-- Define the product of numbers on a face (assuming top face is 0,1,2,3 and bottom face is 4,5,6,7)
def FaceProduct (arr : Cube → ℕ) (face : Bool) : ℕ :=
  if face then 
    (arr ⟨0, by norm_num⟩) * (arr ⟨1, by norm_num⟩) * (arr ⟨2, by norm_num⟩) * (arr ⟨3, by norm_num⟩)
  else 
    (arr ⟨4, by norm_num⟩) * (arr ⟨5, by norm_num⟩) * (arr ⟨6, by norm_num⟩) * (arr ⟨7, by norm_num⟩)

-- The main theorem
theorem exists_equal_face_product : 
  ∃ arr : Cube → ℕ, ValidArrangement arr ∧ FaceProduct arr true = FaceProduct arr false := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_face_product_l1089_108925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1089_108980

theorem hyperbola_asymptote (x y : ℝ) :
  x^2 - y^2 / 2 = 1 → ∃ k : ℝ, k = Real.sqrt 2 ∧ (k * x = y ∨ k * x = -y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1089_108980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1089_108995

/-- Given vectors a, b, and c in ℝ², prove that if c is parallel to 2a + b, then λ = 1/2 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, 2) →
  b = (2, -2) →
  c = (1, lambda) →
  ∃ (k : ℝ), c = k • (2 • a + b) →
  lambda = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1089_108995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_less_than_neg_e_l1089_108959

/-- A function f(x) = e^(-x) + ax, where x is a real number and a is a parameter. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) + a * x

/-- The theorem states that if f has two distinct zeros, then a < -e. -/
theorem two_zeros_implies_a_less_than_neg_e (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) → a < -Real.exp 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_less_than_neg_e_l1089_108959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equations_solvable_l1089_108972

variable (n : ℕ)

def is_solution_cubic (A : Matrix (Fin n) (Fin n) ℚ) : Prop :=
  A^3 + 6 • A^2 - 2 • (1 : Matrix (Fin n) (Fin n) ℚ) = 0

def is_solution_quartic (A : Matrix (Fin n) (Fin n) ℚ) : Prop :=
  A^4 + 6 • A^3 - 2 • (1 : Matrix (Fin n) (Fin n) ℚ) = 0

theorem matrix_equations_solvable (h : n = 2019) :
  (∃ A : Matrix (Fin n) (Fin n) ℚ, is_solution_cubic n A) ∧
  (∃ A : Matrix (Fin n) (Fin n) ℚ, is_solution_quartic n A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equations_solvable_l1089_108972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_difference_approx_l1089_108932

noncomputable def dress_price_difference (initial_sale_price : ℝ) (coupon_value : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let original_price := initial_sale_price / 0.85
  let price_after_increase := initial_sale_price * 1.25
  let price_after_first_discount := price_after_increase * 0.9
  let price_after_second_increase := price_after_first_discount * 1.05
  let price_after_final_discount := price_after_second_increase * 0.8
  let price_after_coupon := price_after_final_discount - coupon_value
  let final_price := price_after_coupon * (1 + sales_tax_rate)
  original_price - final_price

theorem dress_price_difference_approx :
  ∃ ε > 0, ε < 0.01 ∧ |dress_price_difference 85 10 0.08 - 24.05| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_difference_approx_l1089_108932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_transformation_theorem_l1089_108983

noncomputable section

-- Define the circle
def circle_param (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 / 2)

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the polar equation of the perpendicular line
def polar_perp_line (ρ θ : ℝ) : Prop := ρ = 3 / (4 * Real.cos θ - 2 * Real.sin θ)

theorem circle_transformation_theorem :
  (∀ x y : ℝ, (∃ θ : ℝ, (x, y) = transform (circle_param θ)) ↔ curve_C x y) ∧
  (∃ ρ θ : ℝ, polar_perp_line ρ θ ∧
    (∀ x y : ℝ, line_l x y ∧ curve_C x y →
      ρ * Real.cos θ = (x + 1) / 2 ∧ ρ * Real.sin θ = (y + 1/2) / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_transformation_theorem_l1089_108983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1089_108979

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 40, and 36, the distance between two adjacent parallel lines is approximately 16.87. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (40 * 40 * 20 + (d / 2) * 40 * (d / 2) = 40 * r^2) ∧ 
  (36 * 36 * 18 + (3 * d / 2) * 36 * (3 * d / 2) = 36 * r^2) → 
  abs (d - 16.87) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1089_108979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l1089_108955

/-- A triangle is acute if all angles are less than 90 degrees -/
def IsAcute (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- A triangle is right if one angle is exactly 90 degrees -/
def IsRight (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

/-- A triangle is obtuse if one angle is greater than 90 degrees -/
def IsObtuse (a b c : ℝ) : Prop :=
  a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2

/-- Triangle classification based on side lengths and circumradius -/
theorem triangle_classification (a b c R : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let expr := a^2 + b^2 + c^2 - 8*R^2
  (expr > 0 ↔ IsAcute a b c) ∧
  (expr = 0 ↔ IsRight a b c) ∧
  (expr < 0 ↔ IsObtuse a b c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l1089_108955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1089_108975

/-- A digit is a natural number from 0 to 9. -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- A sequence of 8 digits where no two adjacent digits have the same parity. -/
def ValidSequence : Type := { seq : Fin 8 → Digit // ∀ i : Fin 7, (seq i).val % 2 ≠ (seq (i.succ)).val % 2 }

/-- The number of valid sequences of 8 digits where no two adjacent digits have the same parity. -/
theorem count_valid_sequences : Fintype.card (Fin 10) * (Fintype.card (Fin 5) ^ 7) = 781250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1089_108975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_power_eight_l1089_108911

open Matrix Real

theorem rotation_matrix_power_eight :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos (π/4), -Real.sin (π/4); Real.sin (π/4), Real.cos (π/4)]
  A^8 = !![1, 0; 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_power_eight_l1089_108911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_numbers_l1089_108904

theorem existence_of_numbers : ∃ (m n p q : ℕ), 
  (m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q) ∧
  (m + n = p + q) ∧
  (Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/3) = Real.sqrt (p : ℝ) + (q : ℝ) ^ (1/3)) ∧
  (Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/3) > 2004) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_numbers_l1089_108904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_may_be_simple_random_sample_l1089_108967

/-- Represents a class of students -/
structure MyClass where
  total_students : ℕ
  num_boys : ℕ
  num_girls : ℕ
  boys_girls_sum : num_boys + num_girls = total_students

/-- Represents a sample of students -/
structure MySample where
  size : ℕ
  num_boys : ℕ
  num_girls : ℕ
  boys_girls_sum : num_boys + num_girls = size

/-- Defines what it means for a sample to be possible as a simple random sample -/
def isPossibleSimpleRandomSample (c : MyClass) (s : MySample) : Prop :=
  s.size ≤ c.total_students ∧
  s.num_boys ≤ c.num_boys ∧
  s.num_girls ≤ c.num_girls

/-- The main theorem to prove -/
theorem sample_may_be_simple_random_sample (c : MyClass) (s : MySample)
  (h_class : c.total_students = 50 ∧ c.num_boys = 20 ∧ c.num_girls = 30)
  (h_sample : s.size = 10 ∧ s.num_boys = 4 ∧ s.num_girls = 6) :
  isPossibleSimpleRandomSample c s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_may_be_simple_random_sample_l1089_108967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_and_choose_fair_l1089_108907

/-- Represents a piece of meat -/
structure Meat where
  weight : ℝ

/-- Represents a housewife's perception of meat weight -/
structure Perception where
  housewife : ℕ
  perceived_weight : Meat → ℝ

/-- Represents the division of meat -/
structure Division where
  part1 : Meat
  part2 : Meat

/-- Represents the divide and choose method -/
def divideAndChoose (m : Meat) (p1 p2 : Perception) : Prop :=
  ∃ (d : Division),
    (p1.perceived_weight d.part1 = p1.perceived_weight d.part2) ∧
    (p2.perceived_weight (if p2.perceived_weight d.part1 ≥ p2.perceived_weight d.part2 then d.part1 else d.part2) ≥ p2.perceived_weight m / 2)

/-- Theorem stating that the divide and choose method ensures fairness -/
theorem divide_and_choose_fair (m : Meat) (p1 p2 : Perception) (d : Division) :
  p1.housewife ≠ p2.housewife →
  divideAndChoose m p1 p2 →
  (p1.perceived_weight (if p2.perceived_weight d.part1 ≥ p2.perceived_weight d.part2 then d.part1 else d.part2) ≥ p1.perceived_weight m / 2) ∧
  (p2.perceived_weight (if p2.perceived_weight d.part1 ≥ p2.perceived_weight d.part2 then d.part1 else d.part2) ≥ p2.perceived_weight m / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_and_choose_fair_l1089_108907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_superfruit_cocktail_cost_l1089_108960

/-- The cost per litre of mixed fruit juice -/
noncomputable def mixed_fruit_cost : ℝ := 262.85

/-- The cost per litre of açaí berry juice -/
noncomputable def acai_berry_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
noncomputable def mixed_fruit_volume : ℝ := 37

/-- The volume of açaí berry juice used -/
noncomputable def acai_berry_volume : ℝ := 24.666666666666668

/-- The cost per litre of the superfruit juice cocktail -/
noncomputable def cocktail_cost_per_litre : ℝ :=
  (mixed_fruit_cost * mixed_fruit_volume + acai_berry_cost * acai_berry_volume) /
  (mixed_fruit_volume + acai_berry_volume)

theorem superfruit_cocktail_cost :
  cocktail_cost_per_litre = 1399.9999999999998 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_superfruit_cocktail_cost_l1089_108960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_2N_plus_1_l1089_108923

theorem divisors_of_2N_plus_1 (n : ℕ) (primes : Finset ℕ) 
  (h_prime : ∀ p ∈ primes, Nat.Prime p)
  (h_distinct : primes.card = n)
  (h_greater_than_3 : ∀ p ∈ primes, p > 3)
  (N : ℕ) (h_N : N = primes.prod id) :
  (Finset.filter (· ∣ 2 * N + 1) (Finset.range (2 * N + 2))).card ≥ 4 * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_2N_plus_1_l1089_108923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l1089_108999

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through a point -/
structure Line where
  point : Point
  slope : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Hyperbola equation -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Point on a line given parameter t -/
def point_on_line (l : Line) (t : ℝ) : Point :=
  { x := l.point.x + t, y := l.point.y + l.slope * t }

/-- Predicate for a point being a focus of the hyperbola -/
def is_focus (p : Point) (h : Hyperbola) : Prop := sorry

/-- Predicate for three points forming an isosceles right triangle -/
def is_isosceles_right_triangle (p1 p2 p3 : Point) : Prop := sorry

/-- Predicate for the angle at the second point being a right angle -/
def right_angle_at (p1 p2 p3 : Point) : Prop := sorry

/-- The theorem statement -/
theorem hyperbola_eccentricity_theorem (h : Hyperbola) 
  (l : Line) (P Q : Point) (F₁ F₂ : Point) :
  (∀ t : ℝ, hyperbola_equation h (point_on_line l t) ↔ t = 0 ∨ point_on_line l t = P ∨ point_on_line l t = Q) →
  is_focus F₁ h →
  is_focus F₂ h →
  point_on_line l 0 = F₁ →
  is_isosceles_right_triangle P Q F₂ →
  right_angle_at Q P F₂ →
  (eccentricity h)^2 = 5 + 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l1089_108999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alfred_sixth_game_prob_l1089_108927

/-- Probability of winning when going first in a single game -/
def p_first_win : ℚ := 2/3

/-- Probability of winning when going second in a single game -/
def p_second_win : ℚ := 1/3

/-- Probability of Alfred winning the n-th game -/
def alfred_win_prob : ℕ → ℚ
  | 0 => 0  -- Add a case for 0 to cover all natural numbers
  | 1 => p_first_win
  | n + 1 => p_second_win * alfred_win_prob n + p_first_win * (1 - alfred_win_prob n)

theorem alfred_sixth_game_prob :
  alfred_win_prob 6 = 364/729 := by sorry

#eval alfred_win_prob 6  -- This will compute and display the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alfred_sixth_game_prob_l1089_108927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_function_minimum_value_l1089_108964

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  (4 / (x - 1) ≤ x - 1) ↔ (x ≥ 3 ∨ (-1 ≤ x ∧ x < 1)) :=
sorry

-- Problem 2
noncomputable def f (x : ℝ) : ℝ := 2 / x + 9 / (1 - 2 * x)

theorem function_minimum_value :
  ∃ (min : ℝ), min = 25 ∧ ∀ x, 0 < x ∧ x < 1/2 → f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_function_minimum_value_l1089_108964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_pond_difference_l1089_108941

theorem duck_pond_difference (lake_michigan_ducks : ℤ) (north_pond_ducks : ℤ) : 
  lake_michigan_ducks = 100 →
  north_pond_ducks * 2 = lake_michigan_ducks →
  lake_michigan_ducks * 2 - north_pond_ducks = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_pond_difference_l1089_108941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_line_n_equation_l1089_108938

noncomputable section

-- Define the two parallel lines
def l₁ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 = 0

-- Define line m
def m_passes_through (x y : ℝ) : Prop := x = Real.sqrt 3 ∧ y = 4

-- Define the intercepted segment length
def intercepted_segment_length : ℝ := 2

-- Define line n
def n_perpendicular (l : ℝ → ℝ → Prop) : Prop := True  -- Placeholder

-- Define the area of the triangle formed by n and coordinate axes
def triangle_area : ℝ := 2 * Real.sqrt 3

-- Theorem for line m
theorem line_m_equation :
  ∀ m : ℝ → ℝ → Prop,
  (∃ x y, m x y ∧ m_passes_through x y) →
  (∃ x₁ y₁ x₂ y₂, l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧ m x₁ y₁ ∧ m x₂ y₂ ∧ 
   ((x₂ - x₁)^2 + (y₂ - y₁)^2 = intercepted_segment_length^2)) →
  ((∀ x y, m x y ↔ x = Real.sqrt 3) ∨ (∀ x y, m x y ↔ y = Real.sqrt 3 / 3 * x + 3)) :=
by
  sorry

-- Theorem for line n
theorem line_n_equation :
  ∀ n : ℝ → ℝ → Prop,
  n_perpendicular l₁ →
  n_perpendicular l₂ →
  (∃ x y, x > 0 ∧ y > 0 ∧ n x 0 ∧ n 0 y ∧ x * y / 2 = triangle_area) →
  ((∀ x y, n x y ↔ y = -(Real.sqrt 3 / 3) * x + 2) ∨ 
   (∀ x y, n x y ↔ y = -(Real.sqrt 3 / 3) * x - 2)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_line_n_equation_l1089_108938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_laurent_greater_chloe_prob_laurent_greater_chloe_proof_l1089_108986

/-- The probability that a uniformly random number from [0, 3000] is greater than
    a uniformly random number from [0, 1000], assuming independence -/
theorem prob_laurent_greater_chloe : ℝ := 5/6

/-- Proof of the theorem -/
theorem prob_laurent_greater_chloe_proof : prob_laurent_greater_chloe = 5/6 := by
  -- We'll use this opportunity to outline the proof steps
  -- Step 1: Define the random variables and their ranges
  -- Step 2: Set up the geometry of the problem
  -- Step 3: Calculate the area where Laurent's number is greater
  -- Step 4: Calculate the total area of possible outcomes
  -- Step 5: Compute the probability as the ratio of favorable outcomes to total outcomes
  sorry -- This skips the actual proof implementation

-- Check that our definitions are recognized
#check prob_laurent_greater_chloe
#check prob_laurent_greater_chloe_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_laurent_greater_chloe_prob_laurent_greater_chloe_proof_l1089_108986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_shift_count_l1089_108903

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem sin_period_shift_count (ω φ : ℝ) : 
  ω > 0 → 
  |φ| < 2016 * Real.pi → 
  (∀ x, f ω φ (x + Real.pi / ω) = f ω φ x) →
  (∀ x, f ω φ (x + Real.pi / 6) = g ω x) →
  (∃! n : ℕ, n = 2016 ∧ ∃ S : Finset ℝ, S.card = n ∧ ∀ φ' ∈ S, 
    |φ'| < 2016 * Real.pi ∧ 
    (∀ x, f ω φ' (x + Real.pi / 6) = g ω x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_shift_count_l1089_108903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_measure_l1089_108942

/-- Given a dihedral angle intersected by a plane, where:
    - The projection of the edge onto the plane coincides with the angle bisector of the intersection
    - The edge forms an angle α with the plane
    - The angle formed in the intersection is β
    This theorem states the relationship between these angles and the measure of the dihedral angle γ -/
theorem dihedral_angle_measure (α β γ : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 0 < β ∧ β < Real.pi) 
  (h3 : 0 < γ ∧ γ < Real.pi) : 
  Real.tan (γ / 2) = Real.tan (β / 2) / Real.sin α := by
  sorry

#check dihedral_angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_measure_l1089_108942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1089_108998

variable (f g : ℝ → ℝ)

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- g is an even function
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- f'(x)g(x) + f(x)g'(x) > 0 for x < 0
def condition_derivative (f g : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → (deriv f x) * g x + f x * (deriv g x) > 0

-- g(-3) = 0
def condition_g_at_minus_three (g : ℝ → ℝ) : Prop := g (-3) = 0

-- The solution set of f(x)g(x) < 0
def solution_set (f g : ℝ → ℝ) : Set ℝ :=
  {x | f x * g x < 0}

theorem solution_set_theorem
  (hf : odd_function f)
  (hg : even_function g)
  (hd : condition_derivative f g)
  (hg3 : condition_g_at_minus_three g) :
  solution_set f g = Set.Ioo (-(Real.pi)) (-3) ∪ Set.Ioo 0 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1089_108998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l1089_108916

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point at a given angle -/
structure Line where
  point : Point2D
  angle : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (a b : Point2D) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

theorem parabola_line_intersection_ratio 
  (para : Parabola) 
  (l : Line) 
  (h_focus : l.point = Point2D.mk (para.p / 2) 0)
  (h_angle : l.angle = π / 6)
  (A B : Point2D)
  (h_A : A.y^2 = 2 * para.p * A.x)
  (h_B : B.y^2 = 2 * para.p * B.x)
  (h_A_on_line : A.y - l.point.y = Real.tan l.angle * (A.x - l.point.x))
  (h_B_on_line : B.y - l.point.y = Real.tan l.angle * (B.x - l.point.x)) :
  (distance A l.point) / (distance B l.point) = 7 + 4 * Real.sqrt 3 ∨
  (distance A l.point) / (distance B l.point) = 7 - 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l1089_108916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1089_108928

theorem tan_theta_value (θ : ℝ) (h1 : Real.tan (2 * θ) = -2) (h2 : π < 2 * θ) (h3 : 2 * θ < 2 * π) :
  Real.tan θ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1089_108928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_2sqrt5_l1089_108991

noncomputable def my_sequence (n : ℕ) : ℝ := Real.sqrt (2 + 3 * (n - 1))

theorem position_of_2sqrt5 : ∃ n : ℕ, n = 7 ∧ my_sequence n = 2 * Real.sqrt 5 := by
  use 7
  constructor
  · rfl
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_2sqrt5_l1089_108991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1089_108954

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x) * (Real.cos (ω * x) - Real.sqrt 3 * Real.sin (ω * x)) + Real.sqrt 3 / 2

def smallest_positive_period (ω : ℝ) : Prop :=
  ∀ x, f ω (x + Real.pi / 2) = f ω x ∧
  ∀ T, 0 < T ∧ T < Real.pi / 2 → ∃ x, f ω (x + T) ≠ f ω x

def monotone_decreasing_intervals (ω : ℝ) : Set ℝ :=
  {x | ∃ k : ℤ, k * Real.pi / 2 + Real.pi / 24 ≤ x ∧ x ≤ k * Real.pi / 2 + 7 * Real.pi / 24}

theorem function_properties (ω : ℝ) (h : ω > 0) (h_period : smallest_positive_period ω) :
  ω = 2 ∧ ∀ x ∈ monotone_decreasing_intervals ω, 
    ∀ y ∈ monotone_decreasing_intervals ω, x < y → f ω y < f ω x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1089_108954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l1089_108977

/-- The distance between two planes in 3D space -/
noncomputable def plane_distance (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  |a₁ * d₂ - a₂ * d₁| / Real.sqrt (a₁^2 + b₁^2 + c₁^2)

/-- Theorem: The distance between the planes 2x + 4y - 4z = 10 and 4x + 8y - 8z = 18 is √36/36 -/
theorem distance_between_planes : 
  plane_distance 2 4 (-4) 10 4 8 (-8) 18 = Real.sqrt 36 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l1089_108977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_sum_ellipse_equation_exists_l1089_108936

noncomputable def ellipse_parametric (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t), (4 * (Real.cos t - 4)) / (3 - Real.cos t))

theorem ellipse_equation_sum (A B C D E F : ℤ) : Prop :=
  (∀ t : ℝ, let (x, y) := ellipse_parametric t
   (A : ℝ) * x^2 + (B : ℝ) * x * y + (C : ℝ) * y^2 + (D : ℝ) * x + (E : ℝ) * y + (F : ℝ) = 0) ∧
  Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) 
    (Nat.gcd (Int.natAbs D) (Nat.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
  Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1226

theorem ellipse_equation_exists : ∃ A B C D E F : ℤ, ellipse_equation_sum A B C D E F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_sum_ellipse_equation_exists_l1089_108936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_terms_l1089_108951

/-- Sequence defined by the given recursive formula -/
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => 3
  | 2 => 46
  | (n + 3) => Real.sqrt (a (n + 2) * a (n + 1) - Real.pi / a (n + 2))

/-- The maximum number of terms in the sequence -/
def max_terms : ℕ := 2022

/-- Theorem stating that the maximum number of terms in the sequence is 2022 -/
theorem sequence_max_terms :
  ∀ n : ℕ, n > max_terms → ¬ (a n ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_terms_l1089_108951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1089_108966

/-- Represents the tank and its properties -/
structure Tank where
  capacity : ℚ
  initialLevel : ℚ
  inflow : ℚ
  outflow1 : ℚ
  outflow2 : ℚ

/-- Calculates the time required to fill the tank completely -/
noncomputable def timeTofillTank (tank : Tank) : ℚ :=
  let netFlow := tank.inflow - (tank.outflow1 + tank.outflow2)
  let volumeToFill := tank.capacity - tank.initialLevel
  volumeToFill / netFlow

/-- Theorem stating that the time to fill the tank is 36 minutes -/
theorem tank_fill_time :
  let tank : Tank := {
    capacity := 6000,
    initialLevel := 3000,
    inflow := 1/2,  -- 1 kiloliter per 2 minutes
    outflow1 := 1/4,  -- 1 kiloliter per 4 minutes
    outflow2 := 1/6  -- 1 kiloliter per 6 minutes
  }
  timeTofillTank tank = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1089_108966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_different_greater_than_ten_of_suit_l1089_108952

/-- Represents a standard deck of 52 French playing cards. -/
structure Deck where
  cards : Finset (Fin 4 × Fin 13)
  card_count : cards.card = 52

/-- Represents a hand of 13 dealt cards. -/
def Hand := Finset (Fin 4 × Fin 13)

/-- The probability of dealing a specific hand from a deck. -/
def probability (deck : Deck) (hand : Hand) : ℚ :=
  (hand.card : ℚ) / (deck.cards.card : ℚ)

/-- A hand where all 13 cards have different values. -/
def all_different_values (hand : Hand) : Prop :=
  hand.card = 13 ∧ (hand.image Prod.snd).card = 13

/-- A hand where 10 out of 13 cards are of a specified suit. -/
def ten_of_specified_suit (hand : Hand) (suit : Fin 4) : Prop :=
  hand.card = 13 ∧ (hand.filter (λ c ↦ c.1 = suit)).card = 10

theorem probability_all_different_greater_than_ten_of_suit (deck : Deck) :
  ∃ (h₁ h₂ : Hand) (suit : Fin 4),
    all_different_values h₁ ∧
    ten_of_specified_suit h₂ suit ∧
    probability deck h₁ > probability deck h₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_different_greater_than_ten_of_suit_l1089_108952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coastal_city_spending_l1089_108902

/-- The accumulated spending (in million dollars) at the beginning of May -/
def spending_may_start : ℝ := 1.2

/-- The accumulated spending (in million dollars) at the end of September -/
def spending_sept_end : ℝ := 4.5

/-- The spending during May, June, July, August, and September (in million dollars) -/
def spending_may_to_sept : ℝ := spending_sept_end - spending_may_start

theorem coastal_city_spending :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |spending_may_to_sept - 3.3| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coastal_city_spending_l1089_108902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_upper_bound_l1089_108985

theorem sin_cos_sum_upper_bound (x y z : ℝ) :
  Real.sin x * Real.cos y + Real.sin y * Real.cos z + Real.sin z * Real.cos x ≤ 3/2 ∧
  ∃ a b c : ℝ, Real.sin a * Real.cos b + Real.sin b * Real.cos c + Real.sin c * Real.cos a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_upper_bound_l1089_108985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_64_l1089_108943

theorem sqrt_of_sqrt_64 : ∀ x : ℝ, x^2 = Real.sqrt 64 → x = 2 * Real.sqrt 2 ∨ x = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_64_l1089_108943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1089_108958

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (5 - 4*x - x^2)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Icc (-5 : ℝ) (-2 : ℝ) ∩ domain) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1089_108958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_calculation_l1089_108976

theorem class_mean_calculation (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = 28 →
  first_day_students = 24 →
  second_day_students = 4 →
  first_day_mean = 85/100 →
  second_day_mean = 90/100 →
  let new_mean := (first_day_mean * first_day_students + second_day_mean * second_day_students) / total_students
  ⌊new_mean * 100 + 1/2⌋ / 100 = 86/100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_calculation_l1089_108976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1089_108953

/-- The expression to be evaluated -/
noncomputable def expression : ℝ :=
  2 * ((3.6 * 0.48^2 * 2.50) / (Real.sqrt 0.12 * 0.09^3 * 0.5^2))^2 * Real.exp (-0.3)

/-- The approximate value of the expression -/
def approximateValue : ℝ := 9964154400

/-- Theorem stating that the expression is approximately equal to the given value -/
theorem expression_approximation : 
  ‖expression - approximateValue‖ < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1089_108953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_remainder_l1089_108992

/-- The prime number given in the problem -/
def p : ℕ := 2017

/-- The function to count the number of ordered triples (a,b,c) satisfying the conditions -/
def count_triples : ℕ :=
  (Finset.range (p * (p - 1))).sum (λ a ↦
    (Finset.range (p * (p - 1))).sum (λ b ↦
      if 1 ≤ a ∧ 1 ≤ b ∧ (a^b - b^a) % p = 0
      then 1
      else 0))

/-- The main theorem stating that the remainder of count_triples divided by 1000000 is 2016 -/
theorem count_triples_remainder :
  count_triples % 1000000 = 2016 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_remainder_l1089_108992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raven_brothers_ages_l1089_108908

/-- The number of brothers -/
def n : ℕ := 7

/-- The common difference between the ages of consecutive brothers -/
def d : ℚ := 3/2

/-- The ages of the brothers form an arithmetic sequence -/
def ages (x : ℚ) : Fin n → ℚ := fun i => x + i.val * d

/-- The oldest brother is 4 times as old as the youngest -/
def oldest_youngest_ratio (x : ℚ) : Prop := ages x ⟨n-1, Nat.lt_succ_self (n-1)⟩ = 4 * ages x ⟨0, Nat.zero_lt_succ _⟩

/-- The ages of the brothers when cursed -/
def cursed_ages : Fin n → ℚ := fun i => (3 : ℚ) + i.val * d

theorem raven_brothers_ages :
  ∃ x : ℚ, oldest_youngest_ratio x ∧ ages x = cursed_ages :=
sorry

#eval ages 3 ⟨0, Nat.zero_lt_succ _⟩
#eval ages 3 ⟨n-1, Nat.lt_succ_self (n-1)⟩
#eval (cursed_ages ⟨0, Nat.zero_lt_succ _⟩ : ℚ)
#eval (cursed_ages ⟨n-1, Nat.lt_succ_self (n-1)⟩ : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raven_brothers_ages_l1089_108908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_earnings_l1089_108910

/-- Calculates the new weekly earnings after a percentage raise -/
noncomputable def new_weekly_earnings (original_earnings : ℝ) (percentage_increase : ℝ) : ℝ :=
  original_earnings * (1 + percentage_increase / 100)

/-- Theorem: John's new weekly earnings after a 37.5% raise from $40 is $55 -/
theorem johns_new_earnings :
  new_weekly_earnings 40 37.5 = 55 := by
  -- Unfold the definition of new_weekly_earnings
  unfold new_weekly_earnings
  -- Simplify the arithmetic
  simp [mul_add, mul_div_cancel']
  -- The proof is completed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_earnings_l1089_108910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_key_arrangements_l1089_108931

/-- Represents the number of keys on the keychain -/
def num_keys : ℕ := 6

/-- Represents the number of key pairs that must be adjacent -/
def num_adjacent_pairs : ℕ := 2

/-- Represents the number of distinct units after grouping adjacent pairs -/
def num_units : ℕ := num_keys - num_adjacent_pairs

/-- Calculates the number of circular permutations for the units -/
def circular_permutations : ℕ := (num_units - 1).factorial

/-- Represents the number of internal arrangements for each adjacent pair -/
def internal_arrangements : ℕ := 2

/-- Theorem stating the number of distinct key arrangements -/
theorem distinct_key_arrangements :
  circular_permutations * internal_arrangements^num_adjacent_pairs = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_key_arrangements_l1089_108931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1089_108957

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (1 - x))

-- Define the region
noncomputable def region (x : ℝ) : ℝ := x * g x

-- Theorem statement
theorem area_of_region : 
  (∫ x in (0)..(1), region x) = π / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1089_108957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1089_108926

noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

theorem f_properties :
  let a := (2 : ℝ)
  let b := (5 : ℝ)
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y) ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f a) ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ≥ f b) ∧
  f a = 2 ∧
  f b = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1089_108926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_relation_l1089_108988

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 2B = A + C and a + √2b = 2c, then sin C = (√6 + √2) / 4 -/
theorem triangle_special_angle_relation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  2 * B = A + C ∧
  a + Real.sqrt 2 * b = 2 * c ∧
  Real.sin A / a = Real.sin B / b ∧
  Real.sin B / b = Real.sin C / c →
  Real.sin C = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_relation_l1089_108988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_trig_cos_periodic_l1089_108965

/-- A function is periodic if there exists a non-zero real number p such that f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- A function is trigonometric if it is either sine, cosine, or derived from them -/
def IsTrigonometric (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (g = Real.sin ∨ g = Real.cos) ∧ ∃ a b c d : ℝ, ∀ x : ℝ, f x = a * g (b * x + c) + d

/-- All trigonometric functions are periodic -/
axiom trig_periodic : ∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f

/-- Cosine is a trigonometric function -/
theorem cos_is_trig : IsTrigonometric Real.cos := by
  sorry

/-- Proof that cosine is periodic using the syllogism model -/
theorem cos_periodic : IsPeriodic Real.cos := by
  apply trig_periodic Real.cos
  exact cos_is_trig

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_trig_cos_periodic_l1089_108965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1089_108919

-- Define f as an odd function
noncomputable def f (x : ℝ) : ℝ := (4:ℝ)^x - 1 / (2:ℝ)^x

-- Define g
def g (x b : ℝ) : ℝ := (x - b) * (x - 2*b)

-- Define h
noncomputable def h (x b : ℝ) : ℝ :=
  if x < 1 then f x - b else g x b

theorem problem_solution :
  -- 1. Prove f(x) = 2^x - 2^(-x)
  (∀ x : ℝ, f x = (2:ℝ)^x - (2:ℝ)^(-x)) ∧
  -- 2. Prove k < -1/3 given the condition
  (∀ k : ℝ, (∀ t : ℝ, t ≥ 0 → f (t^2 - 2*t) + f (2*t^2 - k) > 0) → k < -1/3) ∧
  -- 3. Prove the range of b given h has exactly two zeros
  (∀ b : ℝ, (∃! x y : ℝ, h x b = 0 ∧ h y b = 0 ∧ x ≠ y) →
    ((1/2 < b ∧ b < 1) ∨ b ≥ 3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1089_108919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_m_for_symmetry_l1089_108968

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x - Real.sin x

noncomputable def g (m x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (x + m) - Real.sin (x + m)

def is_symmetric_about_y_axis (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem smallest_positive_m_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ 
    is_symmetric_about_y_axis (g m) ∧ 
    (∀ m' : ℝ, m' > 0 → is_symmetric_about_y_axis (g m') → m ≤ m') ∧ 
    m = 5 * Real.pi / 6 := by
  sorry

#check smallest_positive_m_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_m_for_symmetry_l1089_108968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_2022_l1089_108920

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 4 then 2 - |x - 3| else 0  -- We use 0 as a placeholder for other values

-- State the properties of f
axiom f_scale (x : ℝ) (h : 0 < x) : f (4 * x) = 4 * f x

-- State the theorem
theorem smallest_x_equals_f_2022 :
  ∃ (x : ℝ), x > 0 ∧ f x = f 2022 ∧ ∀ (y : ℝ), 0 < y ∧ y < x → f y ≠ f 2022 :=
sorry

-- Add a simple lemma to check if the definition works
lemma f_at_three : f 3 = 2 := by
  simp [f]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_2022_l1089_108920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_face_cross_section_properties_l1089_108989

/-- Regular octahedron with edge length e -/
structure RegularOctahedron :=
  (e : ℝ)
  (e_pos : 0 < e)

/-- Cross-section of a regular octahedron -/
structure OctahedronCrossSection (oct : RegularOctahedron) :=
  (perimeter : ℝ)
  (area : ℝ)

/-- The cross-section of a regular octahedron when intersected by a plane parallel to one of its faces -/
noncomputable def parallel_face_cross_section (oct : RegularOctahedron) : OctahedronCrossSection oct :=
  { perimeter := 3 * oct.e,
    area := (3 * Real.sqrt 3 / 8) * oct.e ^ 2 }

/-- Theorem stating the properties of the cross-section -/
theorem parallel_face_cross_section_properties (oct : RegularOctahedron) :
  let cs := parallel_face_cross_section oct
  cs.perimeter = 3 * oct.e ∧ cs.area = (3 * Real.sqrt 3 / 8) * oct.e ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_face_cross_section_properties_l1089_108989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1089_108935

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem f_properties :
  let f := f
  -- Smallest positive period is π
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
  -- Intervals of monotonic increase
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) ∧
  -- Triangle property
  ∀ A B C : ℝ, ∀ a b c : ℝ,
    f A = 1 / 2 →
    b + c = 2 * a →
    b * c * Real.cos A = 6 →
    a = 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1089_108935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l1089_108917

theorem tv_price_change (P : ℝ) (h : P > 0) : 
  let price_after_discount := P * (1 - 0.20)
  let price_after_promotion := price_after_discount * (1 + 0.55)
  let final_price := price_after_promotion * (1 + 0.12)
  (final_price - P) / P = 0.3888 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l1089_108917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_is_150_l1089_108961

/-- The smaller angle formed by the hands of a clock at 7 o'clock -/
noncomputable def clock_angle_at_7 : ℝ :=
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let degrees_per_hour : ℝ := total_degrees / total_hours
  let hours_from_12_to_7 : ℕ := 7
  degrees_per_hour * hours_from_12_to_7

theorem clock_angle_at_7_is_150 :
  clock_angle_at_7 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_is_150_l1089_108961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_teams_is_40_l1089_108970

/-- Represents the number of teams in the "3-Legged Race" -/
def x : ℕ := sorry

/-- Represents the number of teams in the "8-Legged Race" -/
def y : ℕ := sorry

/-- The total number of students participating in the events -/
def total_students : ℕ := 200

/-- The total number of legs involved in all events -/
def total_legs : ℕ := 240

/-- The number of people in a "3-Legged Race" team -/
def people_in_3legged : ℕ := 3

/-- The number of legs in a "3-Legged Race" team -/
def legs_in_3legged : ℕ := 4

/-- The number of people in an "8-Legged Race" team -/
def people_in_8legged : ℕ := 8

/-- The number of legs in an "8-Legged Race" team -/
def legs_in_8legged : ℕ := 9

theorem total_teams_is_40 : x + y = 40 :=
  by
    have h1 : people_in_3legged * x + people_in_8legged * y = total_students := by sorry
    have h2 : legs_in_3legged * x + legs_in_8legged * y = total_legs := by sorry
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_teams_is_40_l1089_108970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_l1089_108948

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The equation of the found ellipse -/
def found_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The foci of an ellipse given its equation coefficients -/
def foci (a b : ℝ) : Set (ℝ × ℝ) := {(-Real.sqrt (a^2 - b^2), 0), (Real.sqrt (a^2 - b^2), 0)}

theorem ellipse_proof :
  (∀ x y, given_ellipse x y ↔ found_ellipse x y) ∧
  found_ellipse (-3) 2 ∧
  foci 3 2 = foci (Real.sqrt 15) (Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_l1089_108948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investor_total_amount_l1089_108978

/-- Calculates the compound interest for a given principal, rate, compounding frequency, and time --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Theorem: The total amount received by the investor after two years is approximately $10,713.54 --/
theorem investor_total_amount : 
  let initial_investment : ℝ := 7000
  let additional_investment : ℝ := 2000
  let annual_rate : ℝ := 0.10
  let compounding_frequency : ℝ := 2
  let total_time : ℝ := 2
  let initial_amount := compound_interest initial_investment annual_rate compounding_frequency total_time
  let additional_amount := compound_interest additional_investment annual_rate compounding_frequency 1
  ∃ ε > 0, |initial_amount + additional_amount - 10713.54| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investor_total_amount_l1089_108978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_specific_perimeter_and_area_l1089_108947

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Perimeter of a triangle given by three points -/
noncomputable def perimeter (a b c : Point) : ℝ :=
  distance a b + distance b c + distance c a

/-- Area of a triangle given by three points -/
noncomputable def area (a b c : Point) : ℝ :=
  abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2

/-- The main theorem -/
theorem triangle_with_specific_perimeter_and_area :
  ∃! (s : Finset Point), 
    s.card = 4 ∧ 
    (∀ c ∈ s, 
      let a : Point := ⟨0, 0⟩
      let b : Point := ⟨12, 0⟩
      perimeter a b c = 60 ∧ 
      area a b c = 72) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_specific_perimeter_and_area_l1089_108947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l1089_108990

theorem distinct_pairs_count : 
  let count := Finset.card (Finset.filter 
    (fun p : ℕ × ℕ => let (x, y) := p; 0 < x ∧ x < y ∧ Real.sqrt 2500 = Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ))
    (Finset.range 51 ×ˢ Finset.range 51))
  count = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l1089_108990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_at_two_l1089_108994

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_prime_at_two (a b : ℝ) :
  (f a b 1 = -2) →  -- f(1) = -2
  (∀ x, x > 0 → (deriv (f a b)) x = (a * x + b) / (x^2)) →  -- f'(x) = (ax + b) / x^2
  (deriv (f a b) 1 = 0) →  -- f'(1) = 0 (maximum at x=1)
  deriv (f a b) 2 = -1/2 :=  -- f'(2) = -1/2
by
  sorry

#check f_prime_at_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_at_two_l1089_108994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_when_double_l1089_108997

theorem age_ratio_when_double (julio_initial_age james_initial_age : ℕ) 
  (h1 : julio_initial_age = 36)
  (h2 : james_initial_age = 11) :
  ∃ (years : ℕ), 
    (julio_initial_age + years = 2 * (james_initial_age + years)) ∧
    (julio_initial_age + years) / (james_initial_age + years) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_when_double_l1089_108997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_wheels_count_race_wheels_count_is_96_l1089_108993

theorem race_wheels_count (total_people : ℕ) (ratio : List ℕ) (h : ratio.length = 4) : ℕ :=
  let unicycles := (total_people / ratio.sum) * ratio[0]!
  let bicycles := (total_people / ratio.sum) * ratio[1]!
  let tricycles := (total_people / ratio.sum) * ratio[2]!
  let quadricycles := (total_people / ratio.sum) * ratio[3]!
  unicycles * 1 + bicycles * 2 + tricycles * 3 + quadricycles * 4

theorem race_wheels_count_is_96 :
  race_wheels_count 40 [2, 3, 4, 1] (by rfl) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_wheels_count_race_wheels_count_is_96_l1089_108993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_c_speed_calculation_l1089_108915

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  withStream : ℝ
  againstStream : ℝ

/-- Calculates the speed against the stream with wind resistance -/
noncomputable def speedAgainstWithWind (speed : BoatSpeed) (windResistance : ℝ) : ℝ :=
  speed.againstStream - windResistance

/-- Calculates the average speed of two boats -/
noncomputable def averageSpeed (boat1 : BoatSpeed) (boat2 : BoatSpeed) : BoatSpeed where
  withStream := (boat1.withStream + boat2.withStream) / 2
  againstStream := (boat1.againstStream + boat2.againstStream) / 2

theorem boat_c_speed_calculation 
  (boatA : BoatSpeed)
  (boatB : BoatSpeed)
  (windResistance : ℝ)
  (h1 : boatA.withStream = 20)
  (h2 : boatA.againstStream = 6)
  (h3 : boatB.withStream = 24)
  (h4 : boatB.againstStream = 8)
  (h5 : windResistance = 2) :
  let boatC := averageSpeed boatA boatB
  (boatC.withStream = 22 ∧
   boatC.againstStream = 7 ∧
   speedAgainstWithWind boatC windResistance = 5) := by
  sorry

#check boat_c_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_c_speed_calculation_l1089_108915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_cubic_minus_linear_minus_six_l1089_108963

theorem largest_divisor_of_cubic_minus_linear_minus_six :
  ∃ (k : ℕ), k > 0 ∧ (∀ (n : ℤ), (k : ℤ) ∣ (n^3 - n - 6)) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℤ), ¬((m : ℤ) ∣ (n^3 - n - 6))) ∧
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_cubic_minus_linear_minus_six_l1089_108963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passing_through_point_l1089_108912

-- Define the function f
noncomputable def f (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

-- State the theorem
theorem function_passing_through_point (n : ℝ) :
  f n 3 = Real.sqrt 3 → n = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry

#check function_passing_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passing_through_point_l1089_108912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l1089_108901

/-- 
Given two people A and B who can complete a task individually in 24 and 48 days respectively,
this theorem proves that they can complete the task together in 16 days.
-/
theorem task_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 24) 
  (hb : b_time = 48) 
  (hc : combined_time = 16) : 
  1 / combined_time = 1 / a_time + 1 / b_time := by
  sorry

#check task_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l1089_108901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_modified_product_l1089_108934

theorem min_value_of_modified_product (N : ℕ) : 
  (∃ (a b c d e f g h : ℚ), 
    (a = 9 ∨ a = 1/9) ∧ (b = 8 ∨ b = 1/8) ∧ (c = 7 ∨ c = 1/7) ∧ 
    (d = 6 ∨ d = 1/6) ∧ (e = 5 ∨ e = 1/5) ∧ (f = 4 ∨ f = 1/4) ∧ 
    (g = 3 ∨ g = 1/3) ∧ (h = 2 ∨ h = 1/2) ∧
    N = Int.floor (a * b * c * d * e * f * g * h)) →
  N ≥ 70 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_modified_product_l1089_108934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l1089_108922

/-- Given a triangle PQR with PQ = QR = 10, and a point S on PR such that PS = 12 and QS = 5, 
    prove that RS = 29/12. -/
theorem triangle_segment_length (P Q R S : EuclideanSpace ℝ (Fin 2)) : 
  ‖P - Q‖ = 10 →
  ‖Q - R‖ = 10 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • P + t • R →
  ‖P - S‖ = 12 →
  ‖Q - S‖ = 5 →
  ‖R - S‖ = 29/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l1089_108922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_rate_is_70_verify_solution_l1089_108945

/-- The rate per kg for apples that Tom purchased -/
def apple_rate (n : ℕ) : Prop := n = 70

/-- The total amount Tom paid to the shopkeeper -/
def total_paid : ℕ := 1145

/-- The amount of apples Tom purchased in kg -/
def apple_amount : ℕ := 8

/-- The amount of mangoes Tom purchased in kg -/
def mango_amount : ℕ := 9

/-- The rate per kg for mangoes -/
def mango_rate : ℕ := 65

theorem apple_rate_is_70 : apple_rate 70 := by
  rfl

theorem verify_solution : 
  apple_amount * 70 + mango_amount * mango_rate = total_paid := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_rate_is_70_verify_solution_l1089_108945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_journey_time_l1089_108906

/-- The time taken for a journey given distance and speed -/
noncomputable def journey_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Theorem: James' journey time is 5 hours -/
theorem james_journey_time :
  let distance : ℝ := 80
  let speed : ℝ := 16
  journey_time distance speed = 5 := by
  -- Unfold the definition of journey_time
  unfold journey_time
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_journey_time_l1089_108906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_water_ratio_in_pitcher_l1089_108974

/-- Given a glass and a mug with specific juice and water ratios,
    prove the ratio of juice to water when combined in a pitcher -/
theorem juice_water_ratio_in_pitcher (V : ℚ) (V_pos : V > 0) :
  (2 / 3 * V + 8 / 5 * V) / (1 / 3 * V + 2 / 5 * V) = 34 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_water_ratio_in_pitcher_l1089_108974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_exists_l1089_108982

/-- Number of positive divisors of m -/
def d (m : ℕ+) : ℕ := sorry

/-- Number of distinct prime divisors of m -/
def ω (m : ℕ+) : ℕ := sorry

/-- Main theorem -/
theorem infinite_n_exists (k : ℕ+) : 
  ∃ S : Set ℕ+, (Set.Infinite S) ∧ 
  (∀ n ∈ S, (ω n = k) ∧ 
    (∀ a b : ℕ+, a + b = n → ¬(d n ∣ d (a^2 + b^2)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_exists_l1089_108982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_zero_in_expansion_l1089_108956

theorem coefficient_zero_in_expansion (x : ℝ) (r : ℤ) : 
  r = 2 → 
  -1 ≤ r ∧ r ≤ 5 → 
  ∃ k : ℝ, (1 - 1/x) * (1 + x)^5 = k * x^r + (fun y ↦ (1 - 1/y) * (1 + y)^5 - k * y^r) x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_zero_in_expansion_l1089_108956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lattice_points_on_line_max_a_is_maximum_l1089_108987

/-- A lattice point in an xy-coordinate system. -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The maximum value of a for which the line y = mx + 3 does not pass through
    any lattice point with 0 < x ≤ 150 for all m such that 2/3 < m < a. -/
def max_a : ℚ := 152 / 151

theorem no_lattice_points_on_line (m : ℚ) (h1 : 2/3 < m) (h2 : m < max_a) :
  ∀ (p : LatticePoint), 0 < p.x → p.x ≤ 150 → ↑p.y ≠ m * ↑p.x + 3 := by
  sorry

theorem max_a_is_maximum :
  ∀ (a : ℚ), a > max_a →
  ∃ (m : ℚ) (p : LatticePoint), 2/3 < m ∧ m < a ∧ 0 < p.x ∧ p.x ≤ 150 ∧ ↑p.y = m * ↑p.x + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lattice_points_on_line_max_a_is_maximum_l1089_108987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_five_range_l1089_108971

/-- The range of exact values for the approximate number 5.0 -/
theorem approximate_five_range :
  ∀ x : ℝ, (round x = (5 : ℝ)) ↔ (4.95 ≤ x ∧ x < 5.05) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_five_range_l1089_108971
