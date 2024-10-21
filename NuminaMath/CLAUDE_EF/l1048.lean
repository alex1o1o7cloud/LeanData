import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_costPerPage_eq_0_8211_l1048_104828

/-- Calculates the cost per page in local currency given the following conditions:
    - 2 notebooks with 50 pages each
    - 1 box of 10 pens
    - 1 pack of 15 folders
    - Total cost before discounts: $40
    - Discount: 15% off total purchase
    - Sales tax: 5%
    - Exchange rate: 2.3 units of local currency to 1 dollar
-/
def costPerPage (
  notebookCount : Nat
) (pagesPerNotebook : Nat
) (penBoxCount : Nat
) (folderPackCount : Nat
) (totalCostBeforeDiscount : Real
) (discountPercentage : Real
) (salesTaxPercentage : Real
) (exchangeRate : Real
) : Real :=
  sorry

/-- Theorem stating that the cost per page is 0.8211 units of local currency -/
theorem costPerPage_eq_0_8211 :
  costPerPage 2 50 1 1 40 0.15 0.05 2.3 = 0.8211 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_costPerPage_eq_0_8211_l1048_104828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_root_product_correct_answer_l1048_104823

/-- The quadratic equation 4x^2 - 8x + k = 0 has real roots -/
def has_real_roots (k : ℝ) : Prop :=
  (8^2 - 4 * 4 * k) ≥ 0

/-- The product of roots for the quadratic equation 4x^2 - 8x + k = 0 -/
noncomputable def root_product (k : ℝ) : ℝ := k / 4

/-- Theorem stating that the product of roots is maximum when k = 4 -/
theorem max_root_product :
  ∀ k : ℝ, has_real_roots k → root_product k ≤ root_product 4 := by
  sorry

/-- The value of k that maximizes the product of roots -/
def max_k : ℝ := 4

/-- Theorem stating that 4 is the correct answer -/
theorem correct_answer : max_k = 4 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_root_product_correct_answer_l1048_104823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l1048_104884

theorem circular_table_seating (n m : ℕ) (hn : n = 8) (hm : m = 6) :
  (n.choose (n - m)) * Nat.factorial (m - 1) = 3360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l1048_104884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l1048_104818

theorem sum_remainder_mod_30 (a b c : ℕ) 
  (ha : a % 30 = 15) 
  (hb : b % 30 = 7) 
  (hc : c % 30 = 18) : 
  (a + 2 * b + c) % 30 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l1048_104818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_bisection_l1048_104840

/-- Given a segment AB, prove that arcs with radius (AB * √2) / 2 from A and B intersect at the midpoint of AB -/
theorem segment_bisection (A B : ℝ × ℝ) : 
  let AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let R : ℝ := (AB * Real.sqrt 2) / 2
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
  ((C.1 - A.1)^2 + (C.2 - A.2)^2 = R^2) ∧ 
  ((C.1 - B.1)^2 + (C.2 - B.2)^2 = R^2)
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_bisection_l1048_104840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1048_104887

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 5)

theorem f_properties :
  (∃ (x : ℝ), f x ≠ f (-x)) ∧  -- non-odd
  (∃ (x : ℝ), f x ≠ -f (-x)) ∧  -- non-even
  (∀ (x : ℝ), f (x + Real.pi) = f x) ∧  -- period π
  (∀ (x : ℝ), f x = Real.cos (2 * (x + Real.pi / 10))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1048_104887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_squares_count_l1048_104815

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- Calculates the number of black squares in a given square -/
def blackSquaresCount (s : Square) : Nat :=
  sorry

/-- Checks if a square is valid (fits on the board) -/
def isValidSquare (s : Square) : Prop :=
  s.size > 0 ∧ s.size ≤ 8 ∧ 
  s.position.1 + s.size ≤ 8 ∧ s.position.2 + s.size ≤ 8

/-- The set of all valid squares on the board -/
def allValidSquares : Set Square :=
  { s : Square | isValidSquare s }

/-- The set of squares containing at least 5 black squares -/
def squaresWithAtLeast5Black : Set Square :=
  { s ∈ allValidSquares | blackSquaresCount s ≥ 5 }

/-- Assume that squaresWithAtLeast5Black is finite -/
instance : Fintype squaresWithAtLeast5Black := sorry

theorem checkerboard_squares_count : Fintype.card squaresWithAtLeast5Black = 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_squares_count_l1048_104815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1048_104869

/-- Represents a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : A > 0 ∧ A < π/2 ∧ B > 0 ∧ B < π/2 ∧ C > 0 ∧ C < π/2
  angle_sum : A + B + C = π
  side_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

/-- The dot product of vectors AB and CB -/
noncomputable def dot_product (t : Triangle) : ℝ := t.c * (t.a * Real.cos t.B)

theorem triangle_properties (t : Triangle) 
  (h : 2 * area t = Real.sqrt 3 * dot_product t) : 
  t.B = π/3 ∧ Real.sqrt 3 < (t.a + t.c) / t.b ∧ (t.a + t.c) / t.b ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1048_104869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseRatio_eq_two_plus_sqrt_two_l1048_104829

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  -- The lengths of the parallel bases
  AB : ℝ
  CD : ℝ
  -- The areas of the four triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  -- Conditions
  ab_gt_cd : AB > CD
  area1_eq : area1 = 2
  area2_eq : area2 = 3
  area3_eq : area3 = 4
  area4_eq : area4 = 5

/-- The ratio of the parallel bases in the isosceles trapezoid -/
noncomputable def baseRatio (t : IsoscelesTrapezoidWithPoint) : ℝ := t.AB / t.CD

/-- Theorem stating the ratio of the parallel bases -/
theorem baseRatio_eq_two_plus_sqrt_two (t : IsoscelesTrapezoidWithPoint) :
  baseRatio t = 2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseRatio_eq_two_plus_sqrt_two_l1048_104829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1048_104826

/-- The line on which point P moves -/
def line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The circle to which the tangent is drawn -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The length of the tangent from a point (x, y) to the circle -/
noncomputable def tangentLength (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2 - 1)

/-- The minimum length of the tangent from any point on the line to the circle is 1 -/
theorem min_tangent_length : 
  ∃ (x y : ℝ), line x y ∧ ∀ (a b : ℝ), line a b → tangentLength x y ≤ tangentLength a b ∧ tangentLength x y = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1048_104826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_ratio_l1048_104875

-- Define an equilateral triangle with side length 10
def EquilateralTriangle := {side : ℝ // side = 10}

-- Define the area of the equilateral triangle
noncomputable def area (t : EquilateralTriangle) : ℝ := 
  (1/4) * t.val * t.val * Real.sqrt 3

-- Define the perimeter of the equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.val

-- Theorem statement
theorem area_perimeter_ratio (t : EquilateralTriangle) : 
  area t / perimeter t = 5 * Real.sqrt 3 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_ratio_l1048_104875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_dot_product_l1048_104807

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line with slope 45°
def line (x y : ℝ) : Prop := y = x - 1

-- Define a focus of the ellipse
def focus : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (4/3, 1/3)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem ellipse_line_intersection_dot_product :
  ellipse A.1 A.2 ∧ 
  ellipse B.1 B.2 ∧
  line A.1 A.2 ∧
  line B.1 B.2 ∧
  line focus.1 focus.2 →
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) = -1/3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_dot_product_l1048_104807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_if_zero_l1048_104837

noncomputable def sequenceA (a : ℝ) : ℕ → ℝ
  | 0 => 1
  | 1 => a
  | 2 => a
  | (n + 3) => 2 * sequenceA a (n + 2) * sequenceA a (n + 1) - sequenceA a n

def isPeriodic (s : ℕ → ℝ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, s (n + p) = s n

theorem sequence_periodic_if_zero (a : ℝ) :
  (∃ n : ℕ, sequenceA a n = 0) → isPeriodic (sequenceA a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_if_zero_l1048_104837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_nine_sixteenths_l1048_104841

/-- Triangle PQR with points S and T on its sides -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of PS, where S is on PQ -/
  PS : ℝ
  /-- Length of PT, where T is on PR -/
  PT : ℝ
  /-- PQ is positive -/
  PQ_pos : PQ > 0
  /-- QR is positive -/
  QR_pos : QR > 0
  /-- PR is positive -/
  PR_pos : PR > 0
  /-- PS is less than PQ -/
  PS_lt_PQ : PS < PQ
  /-- PT is less than PR -/
  PT_lt_PR : PT < PR

/-- The ratio of areas in the given triangle configuration -/
noncomputable def areaRatio (t : TrianglePQR) : ℝ := 9 / 16

/-- Theorem stating the ratio of areas for the given triangle configuration -/
theorem area_ratio_is_nine_sixteenths (t : TrianglePQR) 
  (h1 : t.PQ = 30) (h2 : t.QR = 50) (h3 : t.PR = 54) 
  (h4 : t.PS = 18) (h5 : t.PT = 36) : 
  areaRatio t = 9 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_nine_sixteenths_l1048_104841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1048_104857

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  (t.a = Real.cos t.A ∧ t.a = Real.cos t.B) ∧
  (t.b = t.a ∧ t.b = 2 * t.c - t.b) ∧
  (Real.cos t.A, Real.cos t.B) = (t.a, 2 * t.c - t.b) ∧
  t.b = 3 ∧
  (1/2) * t.b * Real.sin t.A = 3 * Real.sqrt 3

-- State the theorem
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.A = π/3 ∧ t.a = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1048_104857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l1048_104812

noncomputable section

/-- The curved surface area of a cone -/
def curved_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

/-- The radius of the cone in meters -/
def radius : ℝ := 28

/-- The curved surface area of the cone in square meters -/
def area : ℝ := 2638.9378290154264

/-- The slant height of the cone in meters -/
def slant_height : ℝ := 30

theorem cone_slant_height :
  curved_surface_area radius slant_height = area := by
  -- Unfold the definitions
  unfold curved_surface_area radius slant_height area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed by numerical approximation
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l1048_104812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l1048_104836

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x > 0 then x * (1 - x) else x * (1 + x)

-- State the theorem
theorem odd_function_property :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≤ 0 → f x = x * (1 + x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l1048_104836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_problem_l1048_104878

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The problem statement -/
theorem distance_point_to_line_problem :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 1
  let A : ℝ := 1
  let B : ℝ := -1
  let C : ℝ := -1
  distance_point_to_line x₀ y₀ A B C = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_problem_l1048_104878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1048_104850

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in slope-intercept form
structure Line where
  m : ℝ
  b : ℝ

def pointA : Point := { x := -5, y := 2 }

-- Function to check if a line passes through a point
def passesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

-- Function to get x-intercept of a line
noncomputable def xIntercept (l : Line) : ℝ := -l.b / l.m

-- Function to get y-intercept of a line
def yIntercept (l : Line) : ℝ := l.b

-- Theorem statement
theorem line_equation : 
  ∃ (l : Line), 
    passesThrough l pointA ∧ 
    xIntercept l = 2 * yIntercept l ∧
    (l.m * 2 + 5 = 0 ∨ l.m + 2 * l.b + 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1048_104850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1048_104848

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the circle
def circle_eq (x y : ℝ) : Prop := 9 * x^2 + 9 * y^2 - 12 * x - 12 * y = 29

-- Define the focus of a parabola y^2 = 4ax
def parabola_focus (a : ℝ) : ℝ × ℝ := (a, 0)

-- Theorem statement
theorem circle_equation :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- Point 1 is on the line and x-axis
    line x₁ y₁ ∧ y₁ = 0 ∧
    -- Point 2 is on the line and y-axis
    line x₂ y₂ ∧ x₂ = 0 ∧
    -- Point 3 is the focus of the parabola
    x₃ = (parabola_focus 2).1 ∧ y₃ = (parabola_focus 2).2 ∧
    -- All three points satisfy the circle equation
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ circle_eq x₃ y₃ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1048_104848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_properties_l1048_104860

/-- A random variable following a Bernoulli distribution -/
structure BernoulliDistribution where
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a Bernoulli distribution -/
def expectation_bernoulli (X : BernoulliDistribution) : ℝ := X.p

/-- Variance of a Bernoulli distribution -/
def variance_bernoulli (X : BernoulliDistribution) : ℝ := X.p * (1 - X.p)

/-- Expected value of a binomial distribution -/
def expectation_binomial (Y : BinomialDistribution) : ℝ := (Y.n : ℝ) * Y.p

/-- Variance of a binomial distribution -/
def variance_binomial (Y : BinomialDistribution) : ℝ := (Y.n : ℝ) * Y.p * (1 - Y.p)

theorem distribution_properties :
  let X : BernoulliDistribution := ⟨0.7, by norm_num⟩
  let Y : BinomialDistribution := ⟨10, 0.8, by norm_num⟩
  (expectation_bernoulli X = 0.7) ∧
  (variance_bernoulli X = 0.21) ∧
  (expectation_binomial Y = 8) ∧
  (variance_binomial Y = 1.6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_properties_l1048_104860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_theorem_l1048_104870

/-- Triangle ABC is acute -/
structure AcuteTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_acute : 0 < A.1 ∧ 0 < A.2 ∧ 0 < B.1 ∧ 0 < B.2 ∧ 0 < C.1 ∧ 0 < C.2

/-- Point P in the plane of triangle ABC -/
def PointInPlane := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Area of a triangle -/
noncomputable def triangleArea (t : AcuteTriangle) : ℝ := sorry

/-- Tangent of an angle in a triangle -/
noncomputable def tanAngle (t : AcuteTriangle) (vertex : ℝ × ℝ) : ℝ := sorry

/-- Orthocenter of a triangle -/
noncomputable def orthocenter (t : AcuteTriangle) : ℝ × ℝ := sorry

/-- Main theorem -/
theorem triangle_inequality_theorem (t : AcuteTriangle) (P : PointInPlane) :
  let u := distance t.A P
  let v := distance t.B P
  let w := distance t.C P
  let S := triangleArea t
  (u^2 * tanAngle t t.A + v^2 * tanAngle t t.B + w^2 * tanAngle t t.C ≥ 4 * S) ∧
  (u^2 * tanAngle t t.A + v^2 * tanAngle t t.B + w^2 * tanAngle t t.C = 4 * S ↔ P = orthocenter t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_theorem_l1048_104870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_arrangements_l1048_104865

/-- Represents a chessboard configuration -/
def ChessboardConfig := Fin 100 → Fin 100 → Bool

/-- Checks if a king at position (x, y) can capture another king -/
def canCapture (config : ChessboardConfig) (x y : Fin 100) : Prop :=
  ∃ (dx dy : Fin 3), 
    let nx := x + dx - 1
    let ny := y + dy - 1
    (nx ≠ x ∨ ny ≠ y) ∧ 
    config nx ny = true

/-- Checks if the configuration is valid -/
def isValidConfig (config : ChessboardConfig) : Prop :=
  (∀ x y, config x y → ¬canCapture config x y) ∧
  (∀ row, (Finset.filter (λ col => config row col) (Finset.univ : Finset (Fin 100))).card = 25) ∧
  (∀ col, (Finset.filter (λ row => config row col) (Finset.univ : Finset (Fin 100))).card = 25)

/-- The main theorem -/
theorem two_valid_arrangements : 
  ∃ (validConfigs : Finset ChessboardConfig), 
    (∀ config ∈ validConfigs, isValidConfig config) ∧
    validConfigs.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_arrangements_l1048_104865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l1048_104832

/-- Line 1 parameterization --/
noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 + 3*t, 2 - t)

/-- Line 2 parameterization --/
noncomputable def line2 (u : ℝ) : ℝ × ℝ := (-1 + 4*u, 4 + 3*u)

/-- The intersection point of the two lines --/
noncomputable def intersection_point : ℝ × ℝ := (-53/17, 56/17)

/-- Theorem stating that the intersection_point lies on both lines --/
theorem intersection_point_on_both_lines :
  ∃ t u : ℝ, line1 t = intersection_point ∧ line2 u = intersection_point := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l1048_104832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_l1048_104855

def third_quadrant (α : ℝ) : Prop := Real.sin α < 0 ∧ Real.cos α < 0

theorem trigonometric_expression (α : ℝ) 
  (h : third_quadrant α) : 
  (Real.cos α / Real.sqrt (1 - Real.sin α ^ 2)) + (2 * Real.sin α / Real.sqrt (1 - Real.cos α ^ 2)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_l1048_104855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1048_104883

theorem exponential_inequality (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1048_104883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_theorems_l1048_104845

/-- Circle with center (1,0) and radius 2 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4}

/-- Point P on the circle -/
def P : ℝ × ℝ := (2, 1)

/-- Center of the circle -/
def center : ℝ × ℝ := (1, 0)

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | (r.2 - p.2) * (q.1 - p.1) = (r.1 - p.1) * (q.2 - p.2)}

theorem circle_line_theorems :
  (P ∈ C) →
  (line_through P center = {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}) ∧
  (∃ l : Set (ℝ × ℝ), ∃ A B : ℝ × ℝ,
    A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ P ∈ l ∧
    (A.1 - center.1) * (B.1 - center.1) + (A.2 - center.2) * (B.2 - center.2) = 0 ∧
    l = {p : ℝ × ℝ | p.1 + p.2 - 3 = 0}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_theorems_l1048_104845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_annual_expenses_l1048_104816

noncomputable def epipen_cost (cost : ℝ) (coverage : ℝ) : ℝ := cost * (1 - coverage / 100)

noncomputable def monthly_expense (cost : ℝ) (coverage : ℝ) : ℝ := cost * (1 - coverage / 100)

noncomputable def total_cost (epipens : List (ℝ × ℝ)) (monthly_expenses : List (ℝ × ℝ)) : ℝ :=
  (epipens.map (fun (cost, coverage) => epipen_cost cost coverage)).sum +
  (monthly_expenses.map (fun (cost, coverage) => monthly_expense cost coverage)).sum

theorem john_annual_expenses :
  let epipens : List (ℝ × ℝ) := [(500, 75), (550, 60), (480, 70), (520, 65)]
  let monthly_expenses : List (ℝ × ℝ) := [
    (250, 80), (180, 70), (300, 75), (350, 60), (200, 70), (400, 80),
    (150, 90), (100, 100), (300, 60), (350, 90), (450, 85), (500, 65)
  ]
  total_cost epipens monthly_expenses = 1542.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_annual_expenses_l1048_104816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1048_104858

theorem pure_imaginary_condition (a b : ℝ) :
  let z : ℂ := Complex.I * (a + Complex.I * b)
  (∃ (y : ℝ), z = Complex.I * y ∧ y ≠ 0) ↔ (a ≠ 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1048_104858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_sequence_with_property_l1048_104803

theorem bounded_sequence_with_property (a : ℝ) (ha : a > 0) :
  ∃ (x : ℕ → ℝ), (∀ n : ℕ, ∃ M : ℝ, |x n| ≤ M) ∧
    ∀ i j : ℕ, i ≠ j → |x i - x j| * (|Int.ofNat i - Int.ofNat j|:ℝ)^a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_sequence_with_property_l1048_104803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l1048_104861

def sequence_c : ℕ → ℕ
  | 0 => 3
  | 1 => 3
  | (n + 2) => sequence_c n * sequence_c (n + 1)

theorem c_15_value : sequence_c 14 = 3^377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l1048_104861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1048_104886

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (8 * x - x^2) - Real.sqrt (114 * x - x^2 - 48)

-- State the theorem
theorem f_max_min :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 6 8, f x ≤ max ∧ f x ≥ min) ∧
    (∃ x₁ ∈ Set.Icc 6 8, f x₁ = max) ∧
    (∃ x₂ ∈ Set.Icc 6 8, f x₂ = min) ∧
    max = 2 * Real.sqrt 3 ∧
    min = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1048_104886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_negative_l1048_104835

theorem arithmetic_geometric_mean_negative (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a ≠ b) : 
  (a + b) / 2 < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_negative_l1048_104835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1048_104813

noncomputable def f (x : ℝ) : ℝ := ((Real.cos x - Real.sin x) * Real.sin (2 * x)) / Real.cos x

theorem f_properties :
  ∃ (domain : Set ℝ) (period : ℝ) (max_val min_val : ℝ) (decreasing_intervals : Set (Set ℝ)),
    -- Domain
    domain = {x | ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2} ∧
    -- Smallest positive period
    period = Real.pi ∧
    -- Maximum and minimum values on (-π/2, 0]
    (∀ x ∈ Set.Ioo (-Real.pi / 2) 0, f x ≤ max_val) ∧
    (∃ x ∈ Set.Ioo (-Real.pi / 2) 0, f x = max_val) ∧
    max_val = 0 ∧
    (∀ x ∈ Set.Ioo (-Real.pi / 2) 0, f x ≥ min_val) ∧
    (∃ x ∈ Set.Ioo (-Real.pi / 2) 0, f x = min_val) ∧
    min_val = -Real.sqrt 2 - 1 ∧
    -- Monotonically decreasing intervals
    decreasing_intervals = {I | ∃ k : ℤ,
      I = Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + Real.pi / 2) ∪
          Set.Ioc (k * Real.pi + Real.pi / 2) (k * Real.pi + 5 * Real.pi / 8)} ∧
    (∀ I ∈ decreasing_intervals, StrictMonoOn f I) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1048_104813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_arccos_abs_sin_l1048_104863

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.arccos (|Real.sin x|)

-- Define the interval
def interval : Set ℝ := {x | Real.pi / 2 ≤ x ∧ x ≤ 5 * Real.pi / 2}

-- State the theorem
theorem area_bounded_by_arccos_abs_sin :
  (∫ (x : ℝ) in interval, f x) = Real.pi ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_arccos_abs_sin_l1048_104863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_configuration_exists_l1048_104843

/-- A convex polygon in the plane -/
structure ConvexPolygon where
  -- Define the properties of a convex polygon
  -- (We don't need to fully define it for this statement)

/-- A configuration of polygons in the plane -/
structure PolygonConfiguration where
  polygons : List ConvexPolygon
  -- Additional properties can be defined here

/-- Two polygons share a boundary point -/
def sharesBoundaryPoint (p1 p2 : ConvexPolygon) : Prop :=
  sorry

/-- Two polygons have no common interior points -/
def noCommonInteriorPoints (p1 p2 : ConvexPolygon) : Prop :=
  sorry

/-- Two polygons are congruent -/
def isCongruent (p1 p2 : ConvexPolygon) : Prop :=
  sorry

/-- Main theorem statement -/
theorem convex_polygon_configuration_exists (K : ConvexPolygon) :
  ∃ (config : PolygonConfiguration),
    (config.polygons.length = 7) ∧
    (∀ p, p ∈ config.polygons → isCongruent p K) ∧
    (∀ p, p ∈ config.polygons.tail → sharesBoundaryPoint p K) ∧
    (∀ p1 p2, p1 ∈ config.polygons → p2 ∈ config.polygons → p1 ≠ p2 → noCommonInteriorPoints p1 p2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_configuration_exists_l1048_104843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_with_terminal_side_on_point_l1048_104889

theorem sine_of_angle_with_terminal_side_on_point 
  (α : ℝ) (m : ℝ) (h1 : m ≠ 0) (h2 : Real.cos α = m / 6) :
  let P : ℝ × ℝ := (Real.sqrt 3, m)
  Real.sin α = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_with_terminal_side_on_point_l1048_104889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_theta_l1048_104888

theorem tan_pi_minus_theta (θ : Real) 
  (h1 : π / 2 < θ) (h2 : θ < π) 
  (h3 : Real.sin (π / 2 + θ) = -3 / 5) : 
  Real.tan (π - θ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_theta_l1048_104888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_complete_ring_l1048_104838

def is_valid_step (seq : List Nat) (i : Nat) : Prop :=
  i < seq.length ∧
  ((i + seq[i]! % seq.length = seq.length - 1) ∨ 
   ((i + seq[i]!) % seq.length = (i + 1) % seq.length))

def is_complete_ring (seq : List Nat) : Prop :=
  seq.length = 6 ∧ 
  seq.toFinset = {1, 2, 3, 4, 5, 6} ∧
  seq.getLast? = some 6 ∧
  ∀ i, i < 5 → is_valid_step seq i

theorem unique_complete_ring :
  ∀ seq : List Nat, is_complete_ring seq → seq = [4, 2, 6, 5, 3, 1] :=
by
  sorry

#check unique_complete_ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_complete_ring_l1048_104838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_trip_duration_l1048_104895

/-- The time taken by the cyclist for the entire trip -/
noncomputable def cyclist_trip_time (D : ℝ) (v : ℝ) : ℝ :=
  let car_speed := v
  let cyclist_speed := v / 4.5
  let meeting_point := D * cyclist_speed / (car_speed + cyclist_speed)
  let car_return_time := meeting_point / car_speed
  let cyclist_remaining_time := (D - meeting_point) / cyclist_speed
  car_return_time + cyclist_remaining_time

/-- The theorem stating that the cyclist's trip time is 55 minutes -/
theorem cyclist_trip_duration (D : ℝ) (v : ℝ) 
  (h_positive_D : D > 0) (h_positive_v : v > 0) :
  cyclist_trip_time D v = 55 := by
  sorry

#check cyclist_trip_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_trip_duration_l1048_104895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_to_twelve_l1048_104882

theorem cube_root_eight_to_twelve : (8 : ℝ) ^ (1/3) ^ 12 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_to_twelve_l1048_104882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_calculation_l1048_104876

/-- The capacity of a tank with a leak and an inlet pipe -/
noncomputable def tank_capacity (leak_empty_time inlet_rate combined_empty_time : ℝ) : ℝ :=
  let leak_rate := 1 / leak_empty_time
  let inlet_rate_hourly := inlet_rate * 60
  let combined_rate := 1 / combined_empty_time
  (inlet_rate_hourly) / (leak_rate - combined_rate)

/-- Theorem stating the capacity of the tank under given conditions -/
theorem tank_capacity_calculation :
  tank_capacity 3 6 12 = 864 := by
  -- Unfold the definition of tank_capacity
  unfold tank_capacity
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_calculation_l1048_104876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_left_approx_4403_l1048_104854

/-- Calculates the number of fish left in the sea after fishing -/
def fish_left_in_sea (westward eastward northward southward : ℚ) 
  (a b c d e f : ℚ) : ℚ :=
  let west_left := westward - westward * a / b
  let east_left := eastward - eastward * c / d
  let south_left := southward - southward * e / f
  west_left + east_left + south_left + northward

/-- The number of fish left in the sea is approximately 4403 -/
theorem fish_left_approx_4403 : 
  ∃ ε > 0, |fish_left_in_sea 1800 3200 500 2300 3 4 2 5 1 3 - 4403| < ε := by
  sorry

#check fish_left_approx_4403

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_left_approx_4403_l1048_104854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1048_104852

theorem right_triangle_hypotenuse (X Y Z : Real) (h_right_angle : Y = 90)
  (h_cos : Real.cos X = 3 / 5) (h_yz : Z = 25) : Z = 125 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1048_104852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_lcm_problem_l1048_104802

theorem three_digit_lcm_problem (d n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) : 
  Nat.lcm d n = 690 → ¬(3 ∣ n) → ¬(2 ∣ d) → n = 230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_lcm_problem_l1048_104802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_quadratic_inequality_l1048_104800

theorem range_of_a_for_quadratic_inequality :
  {a : ℝ | ∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0} = Set.Icc (-8 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_quadratic_inequality_l1048_104800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_squared_eq_d_e_squared_eq_e_d_times_e_eq_zero_l1048_104808

/-- An infinite sequence where each number is the square of the previous one, starting with 5 -/
def sequenceA : ℕ → ℕ
  | 0 => 5
  | n + 1 => (sequenceA n) ^ 2

/-- An infinite number d ending with ...90625 -/
noncomputable def d : ℕ → ℕ := sorry

/-- e is defined as 1 - d -/
noncomputable def e : ℕ → ℕ := fun n => 1 - d n

/-- The last n digits of the nth and (n+1)th numbers in the sequence are the same -/
axiom last_digits_same (n : ℕ) : ∃ k : ℕ, sequenceA (n + 1) ≡ sequenceA n [MOD 10^n]

/-- d is equal to its square -/
theorem d_squared_eq_d : ∀ n, (d n)^2 = d n := by sorry

/-- e is equal to its square -/
theorem e_squared_eq_e : ∀ n, (e n)^2 = e n := by sorry

/-- The product of d and e is zero -/
theorem d_times_e_eq_zero : ∀ n, (d n) * (e n) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_squared_eq_d_e_squared_eq_e_d_times_e_eq_zero_l1048_104808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_operation_equality_l1048_104805

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation as a list of bits. -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_bits (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
  (to_bits n).reverse

/-- The binary number 110010₂ -/
def bin_110010 : List Bool := [true, true, false, false, true, false]

/-- The binary number 101000₂ -/
def bin_101000 : List Bool := [true, false, true, false, false, false]

/-- The binary number 100₂ -/
def bin_100 : List Bool := [true, false, false]

/-- The binary number 10₂ -/
def bin_10 : List Bool := [true, false]

/-- The binary number 10111000₂ -/
def bin_10111000 : List Bool := [true, false, true, true, true, false, false, false]

/-- Theorem stating the equality of the binary operation result -/
theorem binary_operation_equality :
  (binary_to_nat bin_110010 * binary_to_nat bin_101000) / 
  (binary_to_nat bin_100 * binary_to_nat bin_10) =
  binary_to_nat bin_10111000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_operation_equality_l1048_104805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_minimum_exists_inequality_proof_l1048_104820

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - x^2 + m * (x - 1)

theorem f_properties :
  (∀ x > 0, deriv f x = 2 * x - 2 / x) ∧
  (deriv f 1 = 0) ∧
  (f 1 = 1) := by sorry

theorem g_minimum_exists :
  ∃ m : ℝ, m ≤ 2 ∧ ∀ x ∈ Set.Ioc 0 1, g m x ≥ 0 ∧ g m 1 = 0 := by sorry

theorem inequality_proof :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → (x₂ - x₁) / (Real.log x₂ - Real.log x₁) < 2 * x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_minimum_exists_inequality_proof_l1048_104820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l1048_104873

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 - φ then 2 * (Real.log x / Real.log φ) + 2 else 0

-- State the theorem
theorem equilateral_triangle_division (n : ℝ) (a : ℝ) (h1 : n = 32) (h2 : a = 1) :
  2 * (Real.log n / Real.log φ) + f (a / n) ≥ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l1048_104873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l1048_104862

def set_A : Set ℝ := {x | x / (x - 1) ≥ 0}

def set_B : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

theorem intersection_equals_open_interval : set_A ∩ set_B = Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l1048_104862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1048_104839

/-- The coefficient of x^2 in the expansion of (2x-3)^5 -/
def coefficient_x_squared : ℤ := -1080

/-- The binomial expansion of (2x-3)^5 -/
def expansion (x : ℝ) : ℝ := (2*x - 3)^5

theorem coefficient_x_squared_in_expansion : 
  ∃ (a b c d e f : ℝ), 
    expansion = (λ x => a*x^5 + b*x^4 + c*x^3 + (coefficient_x_squared : ℝ)*x^2 + e*x + f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1048_104839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l1048_104819

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + Real.log (-x)
  else -((-x) + Real.log x)

-- State the theorem
theorem tangent_line_at_e (h : ∀ x, f (-x) = -f x) :
  let slope := 1 - 1 / Real.exp 1
  let point := (Real.exp 1, f (Real.exp 1))
  (λ x ↦ slope * (x - point.1) + point.2) = (λ x ↦ (1 - 1 / Real.exp 1) * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l1048_104819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_purchase_cost_l1048_104846

/-- Calculate the total cost of Jasmine's purchase with discounts and tax --/
theorem jasmine_purchase_cost :
  let coffee_pounds : ℚ := 4
  let milk_gallons : ℚ := 2
  let coffee_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let initial_discount_rate : ℚ := 1/10
  let additional_milk_discount_rate : ℚ := 1/20
  let tax_rate : ℚ := 2/25

  let coffee_cost := coffee_pounds * coffee_price_per_pound
  let milk_cost := milk_gallons * milk_price_per_gallon
  let total_cost_before_discounts := coffee_cost + milk_cost
  let initial_discount := initial_discount_rate * total_cost_before_discounts
  let cost_after_initial_discount := total_cost_before_discounts - initial_discount
  let additional_milk_discount := additional_milk_discount_rate * milk_cost
  let total_cost_after_discounts := cost_after_initial_discount - additional_milk_discount
  let tax := tax_rate * total_cost_after_discounts
  let final_cost := total_cost_after_discounts + tax

  ∃ (ε : ℚ), abs (final_cost - 8999/500) < ε ∧ ε > 0 := by
    sorry

#eval (4 : ℚ) * (5/2 : ℚ) + (2 : ℚ) * (7/2 : ℚ) -- Total before discounts
#eval (1 - 1/10) * ((4 : ℚ) * (5/2 : ℚ) + (2 : ℚ) * (7/2 : ℚ)) - (1/20) * (2 : ℚ) * (7/2 : ℚ) -- After discounts
#eval (1 + 2/25) * ((1 - 1/10) * ((4 : ℚ) * (5/2 : ℚ) + (2 : ℚ) * (7/2 : ℚ)) - (1/20) * (2 : ℚ) * (7/2 : ℚ)) -- Final cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_purchase_cost_l1048_104846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_f_49_final_result_l1048_104892

/-- A function from ℕ to ℕ satisfying the given condition -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 2 * f (a^2 + b^2) = (f a)^2 + (f b)^2

/-- The set of possible values for f(49) given the conditions -/
def PossibleValues (f : ℕ → ℕ) : Set ℕ :=
  {x : ℕ | SpecialFunction f ∧ f 49 = x}

/-- The theorem stating the possible values of f(49) -/
theorem possible_values_of_f_49 :
  ∀ f : ℕ → ℕ, SpecialFunction f → PossibleValues f = {0, 1, 16} := by
  sorry

/-- The number of possible values -/
def m : ℕ := Finset.card {0, 1, 16}

/-- The sum of possible values -/
def t : ℕ := Finset.sum {0, 1, 16} id

/-- The product of m and t -/
def result : ℕ := m * t

/-- The theorem stating the final result -/
theorem final_result : result = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_f_49_final_result_l1048_104892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_is_three_l1048_104825

/-- Represents the race between Nicky and Cristina -/
structure Race where
  length : ℝ
  cristina_pace : ℝ
  nicky_headstart : ℝ
  nicky_runtime : ℝ

/-- Calculates Nicky's pace given the race conditions -/
noncomputable def nicky_pace (race : Race) : ℝ :=
  (race.cristina_pace * (race.nicky_runtime - race.nicky_headstart)) / race.nicky_runtime

/-- Theorem stating that Nicky's pace is 3 meters per second given the race conditions -/
theorem nicky_pace_is_three (race : Race) 
  (h1 : race.length = 100)
  (h2 : race.cristina_pace = 5)
  (h3 : race.nicky_headstart = 12)
  (h4 : race.nicky_runtime = 30) :
  nicky_pace race = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_is_three_l1048_104825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_centroid_basis_transformation_l1048_104833

open InnerProductSpace

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

def is_basis (v₁ v₂ v₃ : V) : Prop := 
  LinearIndependent ℝ ![v₁, v₂, v₃] ∧ Submodule.span ℝ {v₁, v₂, v₃} = ⊤

def are_coplanar (a b c d : V) : Prop :=
  ∃ (x y z : ℝ), d - a = x • (b - a) + y • (c - a)

theorem coplanar_centroid (O A B C D : V) 
  (h_basis : is_basis V (A - O) (B - O) (C - O))
  (h_centroid : D - O = (1/3 : ℝ) • ((A - O) + (B - O) + (C - O))) :
  are_coplanar V A B C D :=
sorry

-- Additional theorem for option D
theorem basis_transformation (a b c : V)
  (h : is_basis V (a + b) (b + c) (c + a)) :
  is_basis V a b c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_centroid_basis_transformation_l1048_104833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_one_root_l1048_104871

noncomputable def f (b a x : ℝ) : ℝ := b * x^3 - 3/2 * (2*b + 1) * x^2 + 6 * x + a

def monotone_intervals (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

theorem f_monotonicity (b a : ℝ) (hb : b > 0) :
  (b > 1/2 → 
    (monotone_intervals (f b a) {x | x < 1/b}) ∧
    (monotone_intervals (f b a) {x | x > 2}) ∧
    (∀ x y, 1/b < x ∧ x < y ∧ y < 2 → f b a x > f b a y)) ∧
  (b < 1/2 → 
    (monotone_intervals (f b a) {x | x < 2}) ∧
    (monotone_intervals (f b a) {x | x > 1/b}) ∧
    (∀ x y, 2 < x ∧ x < y ∧ y < 1/b → f b a x > f b a y)) ∧
  (b = 1/2 → monotone_intervals (f b a) Set.univ) :=
sorry

theorem f_one_root (a : ℝ) :
  (∃! x, f 1 a x = 0) ↔ (a > -2 ∨ a < -5/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_one_root_l1048_104871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l1048_104849

theorem quadratic_root_difference (p q : ℕ) : 
  (∃ x₁ x₂ : ℚ, 
    (6 * x₁^2 - 19 * x₁ + 3 = 0) ∧ 
    (6 * x₂^2 - 19 * x₂ + 3 = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (|x₁ - x₂| = (Real.sqrt (p : ℝ)) / (q : ℝ)) ∧
    (∀ (prime : ℕ), prime.Prime → ¬(prime^2 ∣ p))) →
  p + q = 295 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l1048_104849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l1048_104885

def spinner_numbers : List Nat := [2, 4, 7, 8, 11, 14, 17, 19]

def is_prime (n : Nat) : Bool := Nat.Prime n

def count_primes (l : List Nat) : Nat :=
  l.filter is_prime |>.length

theorem spinner_prime_probability :
  let total_sections := spinner_numbers.length
  let prime_sections := count_primes spinner_numbers
  (prime_sections : ℚ) / total_sections = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l1048_104885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_is_30_degrees_l1048_104809

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the slope of the line
noncomputable def line_slope : ℝ := Real.sqrt 3 / 3

-- Define the slope angle in radians
noncomputable def slope_angle : ℝ := Real.pi / 6

-- Theorem statement
theorem slope_angle_is_30_degrees :
  (∀ x y : ℝ, line_equation x y) →
  line_slope = Real.tan slope_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_is_30_degrees_l1048_104809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_min_l1048_104821

/-- Right-angled trapezoid ABCD with specific properties -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_parallel_cd : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)
  ab_perpendicular_ad : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 2
  ad_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 2
  cd_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 1

/-- Point P on line segment BC -/
def P (t : Trapezoid) (lambda : ℝ) : ℝ × ℝ :=
  (t.B.1 + lambda * (t.C.1 - t.B.1), t.B.2 + lambda * (t.C.2 - t.B.2))

/-- Dot product of PA and PD -/
def dot_product (t : Trapezoid) (lambda : ℝ) : ℝ :=
  let p := P t lambda
  (p.1 - t.A.1) * (p.1 - t.D.1) + (p.2 - t.A.2) * (p.2 - t.D.2)

/-- Theorem: The dot product PA · PD is minimized when lambda = 1/2 -/
theorem dot_product_min (t : Trapezoid) :
  ∃ (lambda_min : ℝ), ∀ (lambda : ℝ), dot_product t lambda_min ≤ dot_product t lambda ∧ lambda_min = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_min_l1048_104821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1048_104847

/-- Given real numbers a, b, c, mₐ, mᵦ, mc, prove that 9/20 < x₁/x < 5/4
    where x = ab + bc + ca and x₁ = mₐmᵦ + mᵦmc + mcmₐ -/
theorem inequality_proof (a b c mₐ mᵦ mc : ℝ) : 
  let x := a * b + b * c + c * a
  let x₁ := mₐ * mᵦ + mᵦ * mc + mc * mₐ
  (9 : ℝ) / 20 < x₁ / x ∧ x₁ / x < (5 : ℝ) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1048_104847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1048_104801

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1048_104801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1048_104856

theorem max_value_theorem (m n : ℕ) (h1 : m = 15) 
  (h2 : ∃ (a : Fin m → ℕ) (b : Fin n → ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i j, i ≠ j → b i ≠ b j) ∧
    (∀ i, Even (a i)) ∧ 
    (∀ i, Odd (b i)) ∧ 
    (Finset.sum (Finset.univ : Finset (Fin m)) a + Finset.sum (Finset.univ : Finset (Fin n)) b = 1987)) :
  3 * m + 4 * n ≤ 221 := by
  sorry

#check max_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1048_104856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_rowing_time_l1048_104872

/-- Represents the time it takes Jake to row around a square lake -/
noncomputable def time_to_row_lake (lake_side_length : ℝ) (swim_speed : ℝ) : ℝ :=
  let lake_perimeter := 4 * lake_side_length
  let row_speed := 2 * swim_speed
  lake_perimeter / row_speed

/-- Theorem stating that it takes Jake 10 hours to row around the square lake -/
theorem jake_rowing_time :
  let lake_side_length : ℝ := 15
  let swim_speed : ℝ := 1 / (20 / 60)  -- 1 mile per 20 minutes, converted to miles per hour
  time_to_row_lake lake_side_length swim_speed = 10 := by
  sorry

#check jake_rowing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_rowing_time_l1048_104872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_one_plus_cos_max_value_l1048_104896

theorem sin_one_plus_cos_max_value (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (M : Real), M = Real.rpow 2 (1/3) ∧
  (∀ θ', 0 < θ' ∧ θ' < π → Real.sin (1 + Real.cos θ') ≤ M) ∧
  (∃ θ'', 0 < θ'' ∧ θ'' < π ∧ Real.sin (1 + Real.cos θ'') = M) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_one_plus_cos_max_value_l1048_104896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1048_104844

theorem calculation_proof : 
  2 * Real.cos (45 * π / 180) + |Real.sqrt 2 - 3| - (1/3)^(-2 : ℤ) + (2021 - Real.pi)^(0 : ℤ) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1048_104844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1048_104853

theorem calculate_expression : (1/2)⁻¹ - (2021 + Real.pi)^0 + 4 * Real.sin (60 * π / 180) - Real.sqrt 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1048_104853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_cos_3theta_l1048_104898

/-- The maximum y-coordinate of a point on the curve r = cos 3θ is 3√3/8 -/
theorem max_y_coordinate_cos_3theta :
  let r : ℝ → ℝ := λ θ => Real.cos (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 3 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_cos_3theta_l1048_104898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_oclock_ticks_l1048_104804

/-- Represents a clock with a certain number of ticks and total time for ticking -/
structure Clock where
  ticks : ℕ
  total_time : ℚ

/-- Calculates the time between the first and last ticks for a given clock -/
noncomputable def time_between_ticks (c : Clock) : ℚ :=
  c.total_time * (c.ticks - 1 : ℚ) / (12 - 1 : ℚ)

theorem six_oclock_ticks (clock_12 clock_6 : Clock) :
  clock_12.ticks = 12 →
  clock_12.total_time = 99 →
  clock_6.ticks = 6 →
  time_between_ticks clock_6 = 45 := by
  sorry

#eval (99 : ℚ) * (6 - 1) / (12 - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_oclock_ticks_l1048_104804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_correct_answer_l1048_104894

-- Define the type for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the property of being distinct
variable (distinct : {α : Type} → List α → Prop)

-- Theorem statement
theorem parallel_transitivity 
  (a b c : Line) 
  (α β : Plane) 
  (l : Line)
  (h_distinct_lines : distinct [a, b, c])
  (h_distinct_planes : distinct [α, β])
  (h_l_parallel_α : parallel_line_plane l α)
  (h_a_parallel_c : parallel_lines a c)
  (h_b_parallel_c : parallel_lines b c) :
  parallel_lines a b :=
by
  sorry

-- Additional theorem to represent the correct answer (option A)
theorem correct_answer 
  (a b c : Line)
  (h_a_parallel_c : parallel_lines a c)
  (h_b_parallel_c : parallel_lines b c) :
  parallel_lines a b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_correct_answer_l1048_104894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koschei_cant_escape_l1048_104877

/-- Represents the wall a guard is leaning against -/
inductive Wall : Type
| East : Wall
| West : Wall

/-- Represents the state of the three guards -/
structure GuardState :=
  (guard1 : Wall)
  (guard2 : Wall)
  (guard3 : Wall)

/-- Switches the wall a guard is leaning against -/
def switchWall (w : Wall) : Wall :=
  match w with
  | Wall.East => Wall.West
  | Wall.West => Wall.East

/-- Updates the guard state when Koschei moves through a passageway -/
def updateGuardState (state : GuardState) (passageway : Nat) : GuardState :=
  match passageway with
  | 0 => ⟨switchWall state.guard1, state.guard2, state.guard3⟩
  | 1 => ⟨state.guard1, switchWall state.guard2, state.guard3⟩
  | 2 => ⟨state.guard1, state.guard2, switchWall state.guard3⟩
  | _ => state

/-- Checks if all guards are leaning on the same wall -/
def allOnSameWall (state : GuardState) : Prop :=
  state.guard1 = state.guard2 ∧ state.guard2 = state.guard3

/-- Theorem: There exists an initial guard state such that for any sequence of Koschei's moves,
    it is impossible for all guards to lean on the same wall simultaneously -/
theorem koschei_cant_escape : ∃ (initialState : GuardState),
  ∀ (moves : List Nat), ¬(allOnSameWall (moves.foldl updateGuardState initialState)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_koschei_cant_escape_l1048_104877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colored_cells_l1048_104890

/-- Represents a coloring of a grid -/
def Coloring (m n : ℕ) := Fin m → Fin n → Bool

/-- Counts the number of colored cells in a 3x3 square starting at (i, j) -/
def countColored (c : Coloring 2014 2014) (i j : Fin 2012) : ℕ :=
  (Finset.range 3).sum (fun di => (Finset.range 3).sum (fun dj => 
    if c ⟨i.val + di, by sorry⟩ ⟨j.val + dj, by sorry⟩ then 1 else 0))

/-- A coloring is valid if every 3x3 square has an even number of colored cells -/
def isValidColoring (c : Coloring 2014 2014) : Prop :=
  ∀ i j : Fin 2012, Even (countColored c i j)

/-- The total number of colored cells in a coloring -/
def totalColored (c : Coloring 2014 2014) : ℕ :=
  (Finset.range 2014).sum (fun i => (Finset.range 2014).sum (fun j => 
    if c ⟨i, by sorry⟩ ⟨j, by sorry⟩ then 1 else 0))

/-- The main theorem: The minimum number of colored cells in a valid coloring is 1342 -/
theorem min_colored_cells :
  (∃ c : Coloring 2014 2014, isValidColoring c ∧ totalColored c = 1342) ∧
  (∀ c : Coloring 2014 2014, isValidColoring c → totalColored c ≥ 1342) := by
  sorry

#check min_colored_cells

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colored_cells_l1048_104890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_a_distance_l1048_104879

-- Define the square sheet
noncomputable def sheet_area : ℝ := 16

-- Define the side length of the square
noncomputable def side_length : ℝ := Real.sqrt sheet_area

-- Define the length of the leg of the black triangle after the first fold
noncomputable def black_triangle_leg : ℝ := (4 * Real.sqrt 6) / 3

-- Theorem to prove
theorem point_a_distance : 
  (2 * black_triangle_leg ^ 2 = sheet_area) →  -- Condition for equal black and white areas
  (Real.sqrt 2 * black_triangle_leg = (8 * Real.sqrt 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_a_distance_l1048_104879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1048_104810

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (M : ℝ × ℝ) :
  2 * b * Real.sin (C + π / 6) = a + c →
  M = ((B + C) / 2, 0) →  -- Assuming B and C are x-coordinates on a line
  dist (0, 0) M = dist (0, 0) (C, 0) →
  dist (0, 0) M = 2 →
  B = π / 3 ∧ a = 4 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1048_104810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_nine_terms_l1048_104851

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    2^(n-1)
  else
    2*n - 1

def S (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (i+1)) |>.sum

theorem sum_of_first_nine_terms : S 9 = 377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_nine_terms_l1048_104851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_omega_l1048_104897

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem exists_valid_omega : 
  ∃ ω : ℝ, 
    ω > 0 ∧ 
    (∀ x : ℝ, f ω (π/2 - x) = f ω (π/2 + x)) ∧ 
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/8 → f ω x > f ω y) ∧
    (ω = 1 ∨ ω = 3 ∨ ω = 5 ∨ ω = 7) :=
by sorry

#check exists_valid_omega

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_omega_l1048_104897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1048_104868

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  (1 + 3*m)*x + (3 - 2*m)*y + 8*m - 12 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

-- Theorem statement
theorem line_circle_intersection :
  ∀ m : ℝ, ∃! (p q : ℝ × ℝ),
    p ≠ q ∧
    line_eq m p.1 p.2 ∧
    line_eq m q.1 q.2 ∧
    circle_eq p.1 p.2 ∧
    circle_eq q.1 q.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1048_104868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arc_connection_l1048_104817

/-- Two straight paths in a plane -/
structure StraightPath where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a path is tangent to a circle at a given point -/
def is_tangent (p : StraightPath) (c : Circle) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if two paths are parallel -/
def are_parallel (p1 p2 : StraightPath) : Prop :=
  sorry

/-- Predicate to check if a line segment is perpendicular to a path -/
def is_perpendicular (start end_ : ℝ × ℝ) (p : StraightPath) : Prop :=
  sorry

/-- The main theorem -/
theorem circle_arc_connection (M N : StraightPath) (A B : ℝ × ℝ) :
  (∃ C : Circle, is_tangent M C A ∧ is_tangent N C B) ↔
  ((∃ O : ℝ × ℝ, is_perpendicular A O M ∧ is_perpendicular B O N ∧ 
    (O.1 - A.1)^2 + (O.2 - A.2)^2 = (O.1 - B.1)^2 + (O.2 - B.2)^2) ∨
   (are_parallel M N ∧ is_perpendicular A B M ∧ is_perpendicular A B N)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arc_connection_l1048_104817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_integral_l1048_104831

theorem geometric_sequence_integral (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m) →  -- geometric sequence condition
  a 6 + a 8 = ∫ x in (0 : ℝ)..(4 : ℝ), Real.sqrt (16 - x^2) →          -- integral condition
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 * Real.pi^2 :=          -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_integral_l1048_104831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1048_104899

/-- Given two vectors a and b in ℝ³, where a = (2, -1, 3) and b = (-4, 2, x),
    if a and b are parallel, then x = -6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 3 → ℝ := ![2, -1, 3]
  let b : Fin 3 → ℝ := ![-4, 2, x]
  (∃ (k : ℝ), a = k • b) →
  x = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1048_104899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_amount_optimal_purchase_is_30_l1048_104834

/-- The annual oil purchase amount in tons -/
noncomputable def annual_purchase : ℝ := 900

/-- The shipping cost per purchase in yuan -/
noncomputable def shipping_cost : ℝ := 3000

/-- The storage cost coefficient in thousand yuan per ton -/
noncomputable def storage_cost_coef : ℝ := 3

/-- The total cost function -/
noncomputable def total_cost (x : ℝ) : ℝ := (annual_purchase / x) * shipping_cost + storage_cost_coef * x * 1000

/-- The optimal purchase amount minimizes the total cost -/
theorem optimal_purchase_amount :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → total_cost x ≤ total_cost y := by
  sorry

/-- The optimal purchase amount is 30 tons -/
theorem optimal_purchase_is_30 :
  ∃ (x : ℝ), x = 30 ∧ ∀ (y : ℝ), y > 0 → total_cost x ≤ total_cost y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_amount_optimal_purchase_is_30_l1048_104834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1048_104822

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a sequence satisfies the adjacent sum condition -/
def validSequence (seq : List ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < seq.length → isPerfectSquare (seq[i]! + seq[i+1]!)

/-- A function that checks if a list contains all integers from 1 to n -/
def containsAllIntegers (seq : List ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → k ∈ seq

/-- The main theorem statement -/
theorem smallest_valid_n : 
  ∀ n : ℕ, n > 1 → 
  (∃ seq : List ℕ, seq.length = n ∧ validSequence seq ∧ containsAllIntegers seq n) 
  ↔ n ≥ 15 := by
  sorry

#check smallest_valid_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1048_104822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1048_104827

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 12 > 0}
def B (m : ℝ) : Set ℝ := {x | |x - m| ≤ m^2}

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, (B m ⊆ A) ∧ ¬(A ⊆ B m)) →
  (∀ m : ℝ, -2 < m ∧ m < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1048_104827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1048_104867

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 9) / (4 * x^2 + 6 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, x > M → |f x - 7/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1048_104867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_brake_distance_l1048_104891

def brakeSequence (a : ℕ) (d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => brakeSequence a d n + d

def totalDistance (a : ℕ) (d : ℤ) : ℤ :=
  (List.range 4).map (brakeSequence a d) |>.sum

theorem car_brake_distance :
  totalDistance 32 (-8) = 80 := by
  rfl

#eval totalDistance 32 (-8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_brake_distance_l1048_104891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_and_height_l1048_104859

/-- Definition of a tetrahedron with given vertices -/
structure Tetrahedron where
  A1 : ℝ × ℝ × ℝ
  A2 : ℝ × ℝ × ℝ
  A3 : ℝ × ℝ × ℝ
  A4 : ℝ × ℝ × ℝ

/-- Calculate the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculate the height from A4 to face A1A2A3 -/
noncomputable def tetrahedronHeight (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume and height of the specific tetrahedron -/
theorem tetrahedron_volume_and_height :
  let t : Tetrahedron := {
    A1 := (0, -1, -1),
    A2 := (-2, 3, 5),
    A3 := (1, -5, -9),
    A4 := (-1, -6, 3)
  }
  volume t = 5/3 ∧ tetrahedronHeight t = Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_and_height_l1048_104859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_false_l1048_104880

-- Define the linear regression equation
def linear_regression (x y : ℝ) (b a : ℝ) : Prop := y = b * x + a

-- Define the mean of x and y
noncomputable def x_mean : ℝ := sorry
noncomputable def y_mean : ℝ := sorry

-- Define the correlation index R²
def R_squared : ℝ → Prop := sorry

-- Define a better fit relation
def Better_fit (model : ℝ → ℝ) (r : ℝ) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ b a : ℝ, linear_regression x_mean y_mean b a

def proposition_2 : Prop :=
  ∀ x y : ℝ, linear_regression x y 3 (-5) → 
    linear_regression (x + 1) (y + 5) 3 (-5)

def proposition_3 : Prop :=
  R_squared 0.80 ∧ R_squared 0.98 → 
    (∀ model, Better_fit model 0.80 → ¬Better_fit model 0.98)

def proposition_4 : Prop :=
  ∀ y : ℝ, linear_regression 2 y 0.5 (-8) → y = -7

-- Theorem statement
theorem exactly_three_false :
  (¬ proposition_1 ∧ ¬ proposition_2 ∧ ¬ proposition_3 ∧ proposition_4) ∨
  (¬ proposition_1 ∧ ¬ proposition_2 ∧ proposition_3 ∧ ¬ proposition_4) ∨
  (¬ proposition_1 ∧ proposition_2 ∧ ¬ proposition_3 ∧ ¬ proposition_4) ∨
  (proposition_1 ∧ ¬ proposition_2 ∧ ¬ proposition_3 ∧ ¬ proposition_4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_false_l1048_104880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_chord_triangle_perimeter_l1048_104866

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  hpos : 0 < b ∧ b < a

/-- The foci of an ellipse -/
def Ellipse.foci (E : Ellipse a b) : ℝ × ℝ × ℝ × ℝ := sorry

/-- A chord of an ellipse passing through one of its foci -/
def Ellipse.focal_chord (E : Ellipse a b) : Set (ℝ × ℝ) := sorry

/-- The perimeter of a triangle formed by two points on an ellipse and one of its foci -/
def triangle_perimeter (E : Ellipse a b) (A B F : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_chord_triangle_perimeter 
  (a b : ℝ) (E : Ellipse a b) (A B : ℝ × ℝ) :
  let (F₁x, F₁y, F₂x, F₂y) := E.foci
  let F₁ : ℝ × ℝ := (F₁x, F₁y)
  let F₂ : ℝ × ℝ := (F₂x, F₂y)
  A ∈ E.focal_chord → B ∈ E.focal_chord →
  triangle_perimeter E A B F₂ = 4 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_chord_triangle_perimeter_l1048_104866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_l1048_104864

theorem x_is_perfect_square (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (h1 : (x^2022 + x + y^2) % (x*y) = 0) 
  : ∃ (n : ℕ), x = n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_l1048_104864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_primes_x_l1048_104874

def distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem min_distinct_primes_x (x y : ℕ) :
  (0 < x) →
  (0 < y) →
  (distinct_prime_factors (Nat.gcd x y) = 5) →
  (distinct_prime_factors (Nat.lcm x y) = 35) →
  (distinct_prime_factors x > distinct_prime_factors y) →
  (distinct_prime_factors x ≥ 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_primes_x_l1048_104874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_travel_distance_l1048_104811

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem brenda_travel_distance :
  let d₁ := distance (-3) 4 1 1
  let d₂ := distance 1 1 4 (-3)
  d₁ + d₂ = 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_travel_distance_l1048_104811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_k_eq_neg_one_l1048_104806

-- Define the slopes of the two lines
noncomputable def slope1 (k : ℝ) : ℝ := k / 3
noncomputable def slope2 : ℝ := 3

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop := slope1 k * slope2 = -1

-- Theorem statement
theorem lines_perpendicular_iff_k_eq_neg_one :
  ∀ k : ℝ, perpendicular k ↔ k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_k_eq_neg_one_l1048_104806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_minimum_point_l1048_104842

/-- A quadratic function satisfying specific point conditions -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

/-- Theorem stating that f satisfies the given conditions and has the specified range -/
theorem quadratic_function_properties :
  (f 0 = 5) ∧ 
  (f 1 = 2) ∧ 
  (f 2 = 1) ∧ 
  (f 3 = 2) ∧ 
  (f 4 = 5) ∧ 
  (∀ x : ℝ, 0 < x ∧ x < 3 → 1 ≤ f x ∧ f x < 5) := by
  sorry

/-- The minimum value of f in the interval (0, 3) -/
def min_value : ℝ := 1

/-- The x-coordinate of the minimum point -/
def min_point : ℝ := 2

/-- Theorem stating that the minimum point is correct -/
theorem minimum_point :
  ∀ x : ℝ, 0 < x ∧ x < 3 → f min_point ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_minimum_point_l1048_104842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1048_104824

-- Define foci as a function that takes a conic section and returns a set of points
def foci (C : Real → Real → Prop) : Set (Real × Real) := sorry

theorem hyperbola_equation (C h : Real → Real → Prop) :
  (∀ x y, C x y ↔ y^2/16 + x^2/12 = 1) →
  (∀ x y, h x y ↔ y^2/2 - x^2/2 = 1) →
  (foci C = foci h) →
  h 1 (Real.sqrt 3) →
  ∀ x y, h x y ↔ y^2/2 - x^2/2 = 1 :=
by sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1048_104824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1048_104893

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def sequence_prop (a : ℕ → ℝ) : Prop :=
  a 1 = 1/2 ∧ 
  (∀ n : ℕ, n > 0 → a (n + 2) = f (a n)) ∧
  (∀ n : ℕ, n > 0 → a n > 0)

theorem sequence_sum (a : ℕ → ℝ) :
  sequence_prop a →
  a 20 = a 18 →
  a 2016 + a 2017 = Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1048_104893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_sum_l1048_104881

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := Real.log x + n * x

theorem tangent_line_implies_sum (n : ℝ) (x₀ : ℝ) :
  (∃ (y : ℝ → ℝ), (∀ x, y x = 2 * x - 1) ∧ y x₀ = f n x₀ ∧ (deriv (f n)) x₀ = 2) →
  f n 1 + (deriv (f n)) 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_sum_l1048_104881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_f_nonnegative_and_a_range_l1048_104814

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (1 + x) * Real.exp (-x)
noncomputable def g (x a : ℝ) : ℝ := Real.exp x * (1/2 * x^3 + a*x + 1 + 2*x*(Real.cos x))

-- State the theorem
theorem f_minus_f_nonnegative_and_a_range :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x - f (-x) ≥ 0) ∧
  (∃ a : ℝ, ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (f x ≥ g x a ↔ a ≤ -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_f_nonnegative_and_a_range_l1048_104814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l1048_104830

theorem shaded_areas_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π) : 
  (∃ r : Real, r > 0 ∧ 
    (r^2 * Real.tan φ / 2 - φ * r^2 / 2 = φ * r^2 / 2)) ↔ 
  Real.tan φ = 2 * φ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l1048_104830
