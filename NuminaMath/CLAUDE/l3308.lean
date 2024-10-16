import Mathlib

namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l3308_330826

/-- Given two lines that intersect at (2, 3), prove the equation of the line
    passing through the points formed by their coefficients. -/
theorem line_through_coefficient_points
  (A₁ B₁ A₂ B₂ : ℝ) 
  (h₁ : A₁ * 2 + B₁ * 3 = 1)
  (h₂ : A₂ * 2 + B₂ * 3 = 1) :
  ∀ x y : ℝ, (x = A₁ ∧ y = B₁) ∨ (x = A₂ ∧ y = B₂) → 2*x + 3*y = 1 :=
sorry

end NUMINAMATH_CALUDE_line_through_coefficient_points_l3308_330826


namespace NUMINAMATH_CALUDE_diagonal_length_in_specific_kite_l3308_330808

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length -/
structure Kite :=
  (A B C D : ℝ × ℝ)
  (is_kite : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 ∧
             (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2)

/-- The theorem about the diagonal length in a specific kite -/
theorem diagonal_length_in_specific_kite (k : Kite) 
  (ab_length : (k.A.1 - k.B.1)^2 + (k.A.2 - k.B.2)^2 = 100)
  (bc_length : (k.B.1 - k.C.1)^2 + (k.B.2 - k.C.2)^2 = 225)
  (sin_B : Real.sin (Real.arcsin ((k.A.2 - k.B.2) / Real.sqrt ((k.A.1 - k.B.1)^2 + (k.A.2 - k.B.2)^2))) = 4/5)
  (angle_ADB : Real.cos (Real.arccos ((k.A.1 - k.D.1) * (k.B.1 - k.D.1) + 
                                      (k.A.2 - k.D.2) * (k.B.2 - k.D.2)) / 
                        (Real.sqrt ((k.A.1 - k.D.1)^2 + (k.A.2 - k.D.2)^2) * 
                         Real.sqrt ((k.B.1 - k.D.1)^2 + (k.B.2 - k.D.2)^2))) = -1/2) :
  (k.A.1 - k.C.1)^2 + (k.A.2 - k.C.2)^2 = 150 := by sorry

end NUMINAMATH_CALUDE_diagonal_length_in_specific_kite_l3308_330808


namespace NUMINAMATH_CALUDE_cube_dot_path_length_l3308_330830

theorem cube_dot_path_length (cube_edge : ℝ) (h_edge : cube_edge = 2) :
  let face_diagonal := cube_edge * Real.sqrt 2
  let dot_path_radius := face_diagonal / 2
  let dot_path_length := 2 * Real.pi * dot_path_radius
  dot_path_length = 2 * Real.sqrt 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cube_dot_path_length_l3308_330830


namespace NUMINAMATH_CALUDE_multiplication_proof_l3308_330801

theorem multiplication_proof : 287 * 23 = 6601 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l3308_330801


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3308_330807

theorem polynomial_remainder (m : ℚ) : 
  (∃ (f g : ℚ → ℚ) (R : ℚ), 
    (∀ y : ℚ, y^2 + m*y + 2 = (y - 1) * f y + R) ∧
    (∀ y : ℚ, y^2 + m*y + 2 = (y + 1) * g y + R)) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3308_330807


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l3308_330890

theorem ratio_of_fractions : (1 : ℚ) / 6 / ((5 : ℚ) / 8) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l3308_330890


namespace NUMINAMATH_CALUDE_probability_at_least_one_of_each_color_l3308_330813

theorem probability_at_least_one_of_each_color (white red yellow drawn : ℕ) 
  (hw : white = 5) (hr : red = 4) (hy : yellow = 3) (hd : drawn = 4) :
  let total := white + red + yellow
  let total_ways := Nat.choose total drawn
  let favorable_ways := 
    Nat.choose white 2 * Nat.choose red 1 * Nat.choose yellow 1 +
    Nat.choose white 1 * Nat.choose red 2 * Nat.choose yellow 1 +
    Nat.choose white 1 * Nat.choose red 1 * Nat.choose yellow 2
  (favorable_ways : ℚ) / total_ways = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_of_each_color_l3308_330813


namespace NUMINAMATH_CALUDE_definite_integral_tan_cos_sin_l3308_330869

theorem definite_integral_tan_cos_sin : 
  ∫ x in (π / 4)..(Real.arcsin (2 / Real.sqrt 5)), (4 * Real.tan x - 5) / (4 * Real.cos x ^ 2 - Real.sin (2 * x) + 1) = 2 * Real.log (5 / 4) - (1 / 2) * Real.arctan (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_tan_cos_sin_l3308_330869


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_nine_eq_three_l3308_330896

theorem sqrt_of_sqrt_nine_eq_three : Real.sqrt (Real.sqrt 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_nine_eq_three_l3308_330896


namespace NUMINAMATH_CALUDE_largest_a_when_b_equals_c_l3308_330852

theorem largest_a_when_b_equals_c (A B C : ℕ) 
  (h1 : A = 5 * B + C) 
  (h2 : B = C) : 
  A ≤ 24 ∧ ∃ (A₀ : ℕ), A₀ = 24 ∧ ∃ (B₀ C₀ : ℕ), A₀ = 5 * B₀ + C₀ ∧ B₀ = C₀ :=
by sorry

end NUMINAMATH_CALUDE_largest_a_when_b_equals_c_l3308_330852


namespace NUMINAMATH_CALUDE_tangent_problem_l3308_330809

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (1 + Real.tan α) / (1 - Real.tan α) = 3/22 := by
sorry

end NUMINAMATH_CALUDE_tangent_problem_l3308_330809


namespace NUMINAMATH_CALUDE_closer_to_origin_l3308_330828

theorem closer_to_origin : abs (-2 : ℝ) < abs (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_closer_to_origin_l3308_330828


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3308_330803

/-- A point in a 3D rectangular coordinate system -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The symmetric point about the z-axis -/
def symmetricAboutZAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

/-- Theorem: The symmetric point about the z-axis has coordinates (-a, -b, c) -/
theorem symmetric_point_coordinates (p : Point3D) :
  symmetricAboutZAxis p = { x := -p.x, y := -p.y, z := p.z } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3308_330803


namespace NUMINAMATH_CALUDE_sum_of_roots_is_3pi_l3308_330876

-- Define the equation
def tanEquation (x : ℝ) : Prop := Real.tan x ^ 2 - 12 * Real.tan x + 4 = 0

-- Define the interval
def inInterval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem sum_of_roots_is_3pi :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, tanEquation x ∧ inInterval x) ∧
    (∀ x, tanEquation x ∧ inInterval x → x ∈ roots) ∧
    (Finset.sum roots id = 3 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_3pi_l3308_330876


namespace NUMINAMATH_CALUDE_alicia_singles_stats_l3308_330883

/-- Represents a baseball player's hit statistics -/
structure HitStats where
  total : ℕ
  homeRuns : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the number of singles and their percentage of total hits -/
def singlesStats (stats : HitStats) : (ℕ × ℚ) :=
  let singles := stats.total - (stats.homeRuns + stats.triples + stats.doubles)
  let percentage := (singles : ℚ) / (stats.total : ℚ) * 100
  (singles, percentage)

/-- Theorem: Given Alicia's hit statistics, prove that she had 38 singles
    which constitute 76% of her total hits -/
theorem alicia_singles_stats :
  let alicia : HitStats := ⟨50, 2, 3, 7⟩
  singlesStats alicia = (38, 76) := by sorry

end NUMINAMATH_CALUDE_alicia_singles_stats_l3308_330883


namespace NUMINAMATH_CALUDE_necklace_packing_condition_l3308_330894

/-- Represents a necklace of cubes -/
structure CubeNecklace where
  n : ℕ
  numCubes : ℕ
  isLooped : Bool

/-- Represents a cubic box -/
structure CubicBox where
  edgeLength : ℕ

/-- Predicate to check if a necklace can be packed into a box -/
def canBePacked (necklace : CubeNecklace) (box : CubicBox) : Prop :=
  necklace.numCubes = box.edgeLength ^ 3 ∧
  necklace.isLooped = true

/-- Theorem stating the condition for packing the necklace -/
theorem necklace_packing_condition (n : ℕ) :
  let necklace := CubeNecklace.mk n (n^3) true
  let box := CubicBox.mk n
  canBePacked necklace box ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_necklace_packing_condition_l3308_330894


namespace NUMINAMATH_CALUDE_f_expression_on_negative_interval_l3308_330863

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_expression_on_negative_interval
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_known : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end NUMINAMATH_CALUDE_f_expression_on_negative_interval_l3308_330863


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3308_330879

theorem angle_measure_proof (x : Real) : 
  (x + (3 * x + 3) = 90) → x = 21.75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3308_330879


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l3308_330867

/-- A conic section type -/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 - 8 * x + 36 = 0

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : 
  ∃ (a b h k : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ x y, conic_equation x y ↔ (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l3308_330867


namespace NUMINAMATH_CALUDE_bus_passengers_specific_case_l3308_330835

def passengers (m n : ℕ) : ℕ := m - 12 + n

theorem bus_passengers (m n : ℕ) (h : m ≥ 12) : 
  passengers m n = m - 12 + n :=
by sorry

theorem specific_case : passengers 26 6 = 20 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_specific_case_l3308_330835


namespace NUMINAMATH_CALUDE_quadrilateral_area_main_theorem_l3308_330859

/-- A line with slope -1 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  xIntercept : ℝ
  yIntercept : ℝ

/-- A line passing through (10,0) and intersecting y-axis -/
structure Line2 where
  xIntercept : ℝ
  yIntercept : ℝ

/-- The intersection point of the two lines -/
def intersectionPoint : ℝ × ℝ := (5, 5)

/-- The theorem stating the area of the quadrilateral -/
theorem quadrilateral_area 
  (l1 : Line1) 
  (l2 : Line2) : ℝ :=
  let o := (0, 0)
  let b := (0, l1.yIntercept)
  let e := intersectionPoint
  let c := (l2.xIntercept, 0)
  87.5

/-- Main theorem to prove -/
theorem main_theorem 
  (l1 : Line1) 
  (l2 : Line2) 
  (h1 : l1.slope = -1)
  (h2 : l1.xIntercept > 0)
  (h3 : l1.yIntercept > 0)
  (h4 : l2.xIntercept = 10)
  (h5 : l2.yIntercept > 0) :
  quadrilateral_area l1 l2 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_main_theorem_l3308_330859


namespace NUMINAMATH_CALUDE_convex_polygon_contains_half_homothety_l3308_330892

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)

/-- A homothety transformation -/
def homothety (center : ℝ × ℝ) (k : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + k * (p.1 - center.1), center.2 + k * (p.2 - center.2))

/-- The theorem stating that a convex polygon contains its image under a 1/2 homothety -/
theorem convex_polygon_contains_half_homothety (P : ConvexPolygon) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ P.vertices →
    homothety center (1/2) p ∈ convexHull ℝ P.vertices :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_contains_half_homothety_l3308_330892


namespace NUMINAMATH_CALUDE_set_operations_l3308_330851

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Aᶜ : Set ℝ) = {x | x ≥ 3 ∨ x ≤ -2} ∧
  (A ∩ B : Set ℝ) = {x | -2 < x ∧ x < 3} ∧
  ((A ∩ B)ᶜ : Set ℝ) = {x | x ≥ 3 ∨ x ≤ -2} ∧
  (Aᶜ ∩ B : Set ℝ) = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3308_330851


namespace NUMINAMATH_CALUDE_parallel_line_through_point_in_plane_l3308_330880

/-- A plane in 3D space -/
structure Plane where
  -- Define plane properties here (omitted for brevity)

/-- A line in 3D space -/
structure Line where
  -- Define line properties here (omitted for brevity)

/-- A point in 3D space -/
structure Point where
  -- Define point properties here (omitted for brevity)

/-- Predicate to check if a line is parallel to a plane -/
def isParallelToPlane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Predicate to check if a point lies on a plane -/
def isOnPlane (P : Point) (α : Plane) : Prop :=
  sorry

/-- Predicate to check if two lines are parallel -/
def areLinesParallel (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def linePassesThroughPoint (l : Line) (P : Point) : Prop :=
  sorry

/-- Predicate to check if a line lies entirely in a plane -/
def lineInPlane (l : Line) (α : Plane) : Prop :=
  sorry

theorem parallel_line_through_point_in_plane 
  (l : Line) (α : Plane) (P : Point)
  (h1 : isParallelToPlane l α)
  (h2 : isOnPlane P α) :
  ∃! m : Line, 
    linePassesThroughPoint m P ∧ 
    areLinesParallel m l ∧ 
    lineInPlane m α :=
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_in_plane_l3308_330880


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3308_330887

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem statement -/
theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, f a b x ≥ 0) →
  (∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < 9) →
  (∃ c, ∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < c) →
  (∃! c, ∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < c) ∧
  (∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < 9) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3308_330887


namespace NUMINAMATH_CALUDE_geometric_sequence_S24_l3308_330836

/-- A geometric sequence with partial sums S_n -/
def geometric_sequence (S : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, S (n + 1) - S n = r * (S n - S (n - 1))

/-- Theorem: For a geometric sequence with given partial sums, S_24 can be determined -/
theorem geometric_sequence_S24 (S : ℕ → ℚ) 
  (h_geom : geometric_sequence S) 
  (h_S6 : S 6 = 48)
  (h_S12 : S 12 = 60) : 
  S 24 = 255 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_S24_l3308_330836


namespace NUMINAMATH_CALUDE_lucas_age_l3308_330888

/-- Given the ages of Lucas, Mia, and Noah, prove that Lucas is 11 years old -/
theorem lucas_age : 
  ∀ (lucas_age mia_age noah_age : ℕ),
  lucas_age = mia_age - 6 →
  mia_age = noah_age + 5 →
  noah_age = 12 →
  lucas_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_lucas_age_l3308_330888


namespace NUMINAMATH_CALUDE_train_length_l3308_330848

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 360 → time = 30 → speed * time * (1000 / 3600) = 3000 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3308_330848


namespace NUMINAMATH_CALUDE_probability_log_equals_one_l3308_330810

def set_A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def set_B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) := 
  {(2, 1), (4, 2), (6, 3)}

def total_outcomes : ℕ := Finset.card set_A * Finset.card set_B

theorem probability_log_equals_one :
  (Finset.card favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_log_equals_one_l3308_330810


namespace NUMINAMATH_CALUDE_quotient_invariance_l3308_330861

theorem quotient_invariance (a b k : ℝ) (hb : b ≠ 0) (hk : k ≠ 0) :
  (a * k) / (b * k) = a / b := by
  sorry

end NUMINAMATH_CALUDE_quotient_invariance_l3308_330861


namespace NUMINAMATH_CALUDE_simplify_expression_l3308_330831

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 3*(a + b)) :
  a/b + b/a - 3/(a*b) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3308_330831


namespace NUMINAMATH_CALUDE_unique_sum_preceding_numbers_l3308_330862

theorem unique_sum_preceding_numbers : 
  ∃! n : ℕ, n > 0 ∧ n = (n * (n - 1)) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_sum_preceding_numbers_l3308_330862


namespace NUMINAMATH_CALUDE_ratio_problem_l3308_330889

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) :
  x / y = 22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3308_330889


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3308_330814

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t, plane.A * line.x t + plane.B * line.y t + plane.C * line.z t + plane.D = 0

/-- The given point (1,2,-3) -/
def givenPoint : Point3D := ⟨1, 2, -3⟩

/-- The given line (x - 2)/4 = (y + 3)/(-6) = (z - 4)/2 -/
def givenLine : Line3D :=
  ⟨λ t => 4*t + 2, λ t => -6*t - 3, λ t => 2*t + 4⟩

/-- The plane we want to prove is correct -/
def resultPlane : Plane := ⟨3, 1, -3, 2⟩

theorem plane_equation_correct :
  pointOnPlane resultPlane givenPoint ∧
  lineInPlane resultPlane givenLine ∧
  resultPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
          (Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3308_330814


namespace NUMINAMATH_CALUDE_area_of_right_triangle_abc_l3308_330811

/-- Right triangle ABC with specific properties -/
structure RightTriangleABC where
  -- A, B, C are points in the plane
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- ABC is a right triangle with right angle at C
  is_right_triangle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  -- Length of AB is 50
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 50^2
  -- Median through A lies on y = x + 2
  median_A : ∃ t : ℝ, (A.1 + C.1) / 2 = t ∧ (A.2 + C.2) / 2 = t + 2
  -- Median through B lies on y = 3x + 1
  median_B : ∃ t : ℝ, (B.1 + C.1) / 2 = t ∧ (B.2 + C.2) / 2 = 3 * t + 1

/-- The area of the right triangle ABC is 250/3 -/
theorem area_of_right_triangle_abc (t : RightTriangleABC) : 
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_abc_l3308_330811


namespace NUMINAMATH_CALUDE_emily_holidays_l3308_330858

/-- The number of days Emily takes off each month -/
def days_off_per_month : ℕ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Emily takes in a year -/
def total_holidays : ℕ := days_off_per_month * months_in_year

theorem emily_holidays : total_holidays = 24 := by
  sorry

end NUMINAMATH_CALUDE_emily_holidays_l3308_330858


namespace NUMINAMATH_CALUDE_count_three_digit_Q_equal_l3308_330877

def Q (n : ℕ) : ℕ := 
  n % 3 + n % 5 + n % 7 + n % 11

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_Q_equal : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_three_digit n ∧ Q n = Q (n + 1)) ∧ 
    S.card = 9 ∧
    (∀ n, is_three_digit n → Q n = Q (n + 1) → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_three_digit_Q_equal_l3308_330877


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3308_330817

-- Problem 1
theorem problem_1 (a : ℝ) : (-a^2)^3 + 9*a^4*a^2 = 8*a^6 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2*a*b^2 + a^2*b + b^3 = b*(a+b)^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 2*y) :
  (1/(x-y) - 1/(x+y)) / ((x-2*y)/((x^2)-(y^2))) = 2*y/(x-2*y) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3308_330817


namespace NUMINAMATH_CALUDE_root_sum_ratio_l3308_330864

theorem root_sum_ratio (m₁ m₂ : ℝ) (a b : ℝ → ℝ) : 
  (∀ m, m * (a m)^2 - (3 * m - 2) * (a m) + 7 = 0) →
  (∀ m, m * (b m)^2 - (3 * m - 2) * (b m) + 7 = 0) →
  (a m₁ / b m₁ + b m₁ / a m₁ = 2) →
  (a m₂ / b m₂ + b m₂ / a m₂ = 2) →
  m₁ / m₂ + m₂ / m₁ = 194 / 9 := by
  sorry


end NUMINAMATH_CALUDE_root_sum_ratio_l3308_330864


namespace NUMINAMATH_CALUDE_total_spider_legs_l3308_330886

/-- The number of spiders in the room -/
def num_spiders : ℕ := 5

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in the room is 40 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l3308_330886


namespace NUMINAMATH_CALUDE_equal_probability_for_all_probability_independent_of_method_l3308_330823

/-- The probability of selecting a product given the described selection method -/
def selection_probability (total : ℕ) (remove : ℕ) (select : ℕ) : ℚ :=
  select / total

/-- The selection method ensures equal probability for all products -/
theorem equal_probability_for_all (total : ℕ) (remove : ℕ) (select : ℕ) 
  (h1 : total = 2003)
  (h2 : remove = 3)
  (h3 : select = 50) :
  selection_probability total remove select = 50 / 2003 := by
  sorry

/-- The probability is independent of the specific selection method -/
theorem probability_independent_of_method 
  (simple_random_sampling : (ℕ → ℕ → ℕ → ℚ))
  (systematic_sampling : (ℕ → ℕ → ℕ → ℚ))
  (total : ℕ) (remove : ℕ) (select : ℕ)
  (h1 : total = 2003)
  (h2 : remove = 3)
  (h3 : select = 50) :
  simple_random_sampling total remove select = systematic_sampling (total - remove) select select ∧
  simple_random_sampling total remove select = selection_probability total remove select := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_for_all_probability_independent_of_method_l3308_330823


namespace NUMINAMATH_CALUDE_parabola_normals_intersection_l3308_330875

/-- The condition for three distinct points on a parabola to have intersecting normals -/
theorem parabola_normals_intersection
  (a b c : ℝ)
  (h_distinct : (a - b) * (b - c) * (c - a) ≠ 0)
  (h_parabola : ∀ (x : ℝ), (x = a ∨ x = b ∨ x = c) → ∃ (y : ℝ), y = x^2) :
  (∃ (p : ℝ × ℝ),
    (∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c) →
      (y - x^2) = -(1 / (2*x)) * (p.1 - x) ∧ p.2 = y)) ↔
  a + b + c = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_normals_intersection_l3308_330875


namespace NUMINAMATH_CALUDE_triangle_angle_from_area_and_dot_product_l3308_330865

theorem triangle_angle_from_area_and_dot_product 
  (A B C : ℝ × ℝ) -- Points in 2D plane
  (area : ℝ) 
  (dot_product : ℝ) :
  area = Real.sqrt 3 / 2 →
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = dot_product →
  dot_product = -3 →
  let angle := Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))
  angle = 5 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_from_area_and_dot_product_l3308_330865


namespace NUMINAMATH_CALUDE_smallest_odd_polygon_is_seven_l3308_330805

/-- A polygon with an odd number of sides that can be divided into parallelograms -/
structure OddPolygon where
  sides : ℕ
  is_odd : Odd sides
  divisible_into_parallelograms : Bool

/-- The smallest number of sides for an OddPolygon -/
def smallest_odd_polygon_sides : ℕ := 7

/-- Theorem stating that the smallest number of sides for an OddPolygon is 7 -/
theorem smallest_odd_polygon_is_seven :
  ∀ (p : OddPolygon), p.divisible_into_parallelograms → p.sides ≥ smallest_odd_polygon_sides :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_polygon_is_seven_l3308_330805


namespace NUMINAMATH_CALUDE_triangle_DEF_angle_D_l3308_330824

theorem triangle_DEF_angle_D (D E F : ℝ) : 
  E = 3 * F → F = 15 → D + E + F = 180 → D = 120 := by sorry

end NUMINAMATH_CALUDE_triangle_DEF_angle_D_l3308_330824


namespace NUMINAMATH_CALUDE_P_value_at_7_l3308_330897

def P (a b c d e f g h : ℝ) (x : ℂ) : ℂ :=
  (3 * x^5 - 45 * x^4 + a * x^3 + b * x^2 + c * x + d) *
  (4 * x^5 - 100 * x^4 + e * x^3 + f * x^2 + g * x + h)

theorem P_value_at_7 (a b c d e f g h : ℝ) :
  (∃ (r : Multiset ℂ), r = {1, 2, 2, 3, 4, 4, 5, 5, 5} ∧ 
   (∀ z : ℂ, z ∈ r ↔ P a b c d e f g h z = 0)) →
  P a b c d e f g h 7 = 172800 := by
sorry

end NUMINAMATH_CALUDE_P_value_at_7_l3308_330897


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3308_330872

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + x) / 3 = (20 + 40 + 6) / 3 + 8 ↔ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3308_330872


namespace NUMINAMATH_CALUDE_orange_calorie_distribution_l3308_330846

theorem orange_calorie_distribution :
  ∀ (num_oranges : ℕ) 
    (pieces_per_orange : ℕ) 
    (num_people : ℕ) 
    (calories_per_orange : ℕ),
  num_oranges = 5 →
  pieces_per_orange = 8 →
  num_people = 4 →
  calories_per_orange = 80 →
  (num_oranges * pieces_per_orange / num_people) * (calories_per_orange / pieces_per_orange) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_calorie_distribution_l3308_330846


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3308_330844

theorem power_fraction_simplification :
  (3^2023 + 3^2021) / (3^2023 - 3^2021) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3308_330844


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l3308_330868

/-- Given two circles with centers 50 inches apart, with radii 7 inches and 10 inches respectively,
    the length of their common internal tangent is equal to the square root of the difference between
    the square of the distance between their centers and the square of the sum of their radii. -/
theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : center_distance = 50) 
  (h2 : radius1 = 7) 
  (h3 : radius2 = 10) : 
  ∃ (tangent_length : ℝ), tangent_length = Real.sqrt (center_distance^2 - (radius1 + radius2)^2) :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l3308_330868


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l3308_330812

def num_items : ℕ := 4
def num_colors : ℕ := 8

def total_combinations : ℕ := num_colors ^ num_items

def same_color_two_items : ℕ := (num_items.choose 2) * num_colors * (num_colors - 1) * (num_colors - 2)

def same_color_three_items : ℕ := (num_items.choose 3) * num_colors * (num_colors - 1)

def same_color_four_items : ℕ := num_colors

def two_pairs_same_color : ℕ := (num_items.choose 2) * 1 * num_colors * (num_colors - 1)

def invalid_combinations : ℕ := same_color_two_items + same_color_three_items + same_color_four_items + two_pairs_same_color

theorem valid_outfit_choices : 
  total_combinations - invalid_combinations = 1512 :=
sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l3308_330812


namespace NUMINAMATH_CALUDE_factorization_equality_l3308_330833

theorem factorization_equality (c : ℤ) : 
  (∀ x : ℤ, x^2 - x + c = (x + 2) * (x - 3)) → c = -6 := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l3308_330833


namespace NUMINAMATH_CALUDE_vacuum_cleaner_price_difference_l3308_330838

/-- The in-store price of the vacuum cleaner in dollars -/
def in_store_price : ℚ := 150

/-- The cost of each online payment in dollars -/
def online_payment : ℚ := 35

/-- The number of online payments -/
def num_payments : ℕ := 4

/-- The one-time processing fee for online purchase in dollars -/
def processing_fee : ℚ := 12

/-- The difference in cents between online and in-store purchase -/
def price_difference_cents : ℤ := 200

theorem vacuum_cleaner_price_difference :
  (num_payments * online_payment + processing_fee - in_store_price) * 100 = price_difference_cents := by
  sorry

end NUMINAMATH_CALUDE_vacuum_cleaner_price_difference_l3308_330838


namespace NUMINAMATH_CALUDE_gcd_difference_perfect_square_l3308_330891

theorem gcd_difference_perfect_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_difference_perfect_square_l3308_330891


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3308_330834

theorem complex_fraction_sum (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3308_330834


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3308_330878

theorem quadratic_form_ratio (j : ℝ) : ∃ (c p q : ℝ),
  (8 * j^2 - 6 * j + 16 = c * (j + p)^2 + q) ∧ (q / p = -119 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3308_330878


namespace NUMINAMATH_CALUDE_floor_plus_self_equals_twenty_l3308_330837

theorem floor_plus_self_equals_twenty (s : ℝ) : ⌊s⌋ + s = 20 ↔ s = 10 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_equals_twenty_l3308_330837


namespace NUMINAMATH_CALUDE_alyssa_bought_224_cards_l3308_330853

/-- The number of Pokemon cards Jason initially had -/
def initial_cards : ℕ := 676

/-- The number of Pokemon cards Jason has after Alyssa bought some -/
def remaining_cards : ℕ := 452

/-- The number of Pokemon cards Alyssa bought -/
def cards_bought : ℕ := initial_cards - remaining_cards

theorem alyssa_bought_224_cards : cards_bought = 224 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_bought_224_cards_l3308_330853


namespace NUMINAMATH_CALUDE_angle_set_impossibility_l3308_330854

/-- Represents a set of angles formed by lines through a single point -/
structure AngleSet where
  odd : ℕ  -- number of angles with odd integer measures
  even : ℕ -- number of angles with even integer measures

/-- The property that the number of odd-measure angles is 15 more than even-measure angles -/
def has_15_more_odd (as : AngleSet) : Prop :=
  as.odd = as.even + 15

/-- The property that both odd and even counts are even numbers due to vertical angles -/
def vertical_angle_property (as : AngleSet) : Prop :=
  Even as.odd ∧ Even as.even

theorem angle_set_impossibility : 
  ¬∃ (as : AngleSet), has_15_more_odd as ∧ vertical_angle_property as :=
sorry

end NUMINAMATH_CALUDE_angle_set_impossibility_l3308_330854


namespace NUMINAMATH_CALUDE_trapezoid_area_l3308_330860

/-- The area of a trapezoid bounded by y = ax, y = bx, x = c, and x = d in the first quadrant -/
theorem trapezoid_area (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hcd : c < d) :
  let area := 0.5 * ((a * c + a * d + b * c + b * d) * (d - c))
  ∃ (trapezoid_area : ℝ), trapezoid_area = area := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3308_330860


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3308_330840

theorem nested_square_root_value :
  ∃ y : ℝ, y = Real.sqrt (2 - Real.sqrt (2 + y)) → y = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3308_330840


namespace NUMINAMATH_CALUDE_rectangle_enumeration_l3308_330825

/-- Represents a rectangle in the Cartesian plane with sides parallel to the axes. -/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Defines when one rectangle is below another. -/
def is_below (r1 r2 : Rectangle) : Prop :=
  r1.y_max < r2.y_min

/-- Defines when one rectangle is to the right of another. -/
def is_right_of (r1 r2 : Rectangle) : Prop :=
  r1.x_min > r2.x_max

/-- Defines when two rectangles are disjoint. -/
def are_disjoint (r1 r2 : Rectangle) : Prop :=
  r1.x_max ≤ r2.x_min ∨ r2.x_max ≤ r1.x_min ∨
  r1.y_max ≤ r2.y_min ∨ r2.y_max ≤ r1.y_min

/-- The main theorem stating that any finite set of pairwise disjoint rectangles
    can be enumerated such that each rectangle is to the right of or below all
    subsequent rectangles in the enumeration. -/
theorem rectangle_enumeration (n : ℕ) (rectangles : Fin n → Rectangle)
    (h_disjoint : ∀ i j : Fin n, i ≠ j → are_disjoint (rectangles i) (rectangles j)) :
    ∃ σ : Equiv.Perm (Fin n),
      ∀ i j : Fin n, i < j →
        is_right_of (rectangles (σ i)) (rectangles (σ j)) ∨
        is_below (rectangles (σ i)) (rectangles (σ j)) :=
  sorry

end NUMINAMATH_CALUDE_rectangle_enumeration_l3308_330825


namespace NUMINAMATH_CALUDE_power_four_remainder_l3308_330802

theorem power_four_remainder (a : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) : 4^a % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_four_remainder_l3308_330802


namespace NUMINAMATH_CALUDE_sum_inequality_l3308_330870

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c / a + a / (b + c) + b / c ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3308_330870


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l3308_330843

/-- 
Given a parallelogram with area 98 square meters and base length 7 meters,
prove that the ratio of its altitude to its base is 2.
-/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
  area = 98 →
  base = 7 →
  area = base * altitude →
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l3308_330843


namespace NUMINAMATH_CALUDE_shoes_lost_l3308_330895

theorem shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : 
  initial_pairs = 26 → remaining_pairs = 21 → initial_pairs * 2 - remaining_pairs * 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_shoes_lost_l3308_330895


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l3308_330822

/-- The product of the coordinates of the midpoint of a line segment with endpoints (5, -3) and (-7, 11) is -4. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 5
  let y1 : ℝ := -3
  let x2 : ℝ := -7
  let y2 : ℝ := 11
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l3308_330822


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l3308_330898

def GCF (a b : ℕ) : ℕ := Nat.gcd a b

def LCM (c d : ℕ) : ℕ := Nat.lcm c d

theorem gcf_of_lcms : GCF (LCM 18 30) (LCM 10 45) = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l3308_330898


namespace NUMINAMATH_CALUDE_monotonically_decreasing_condition_l3308_330893

/-- A function f is monotonically decreasing on an interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

/-- The function f(x) = x^3 - ax^2 + 4d -/
def f (a d : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4*d

theorem monotonically_decreasing_condition (a d : ℝ) :
  MonotonicallyDecreasing (f a d) 0 2 → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_condition_l3308_330893


namespace NUMINAMATH_CALUDE_revenue_difference_l3308_330884

def viewers_game2 : ℕ := 80
def viewers_game1 : ℕ := viewers_game2 - 20
def viewers_game3 : ℕ := viewers_game2 + 15
def viewers_game4 : ℕ := viewers_game3 + (viewers_game3 / 10) + 1 -- Rounded up

def price_game1 : ℕ := 15
def price_game2 : ℕ := 20
def price_game3 : ℕ := 25
def price_game4 : ℕ := 30

def viewers_last_week : ℕ := 350
def price_last_week : ℕ := 18

def revenue_this_week : ℕ := 
  viewers_game1 * price_game1 + 
  viewers_game2 * price_game2 + 
  viewers_game3 * price_game3 + 
  viewers_game4 * price_game4

def revenue_last_week : ℕ := viewers_last_week * price_last_week

theorem revenue_difference : 
  revenue_this_week - revenue_last_week = 1725 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_l3308_330884


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3308_330873

theorem complex_number_quadrant (z : ℂ) (h : z / Complex.I = 2 - 3 * Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3308_330873


namespace NUMINAMATH_CALUDE_three_X_four_equals_31_l3308_330819

-- Define the operation X
def X (a b : ℤ) : ℤ := b + 12 * a - a^2

-- Theorem statement
theorem three_X_four_equals_31 : X 3 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_three_X_four_equals_31_l3308_330819


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3308_330882

theorem expression_simplification_and_evaluation :
  ∀ x : ℤ, -2 < x → x < 2 → x ≠ -1 → x ≠ 0 →
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1) = (1 - x) / x) ∧
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3308_330882


namespace NUMINAMATH_CALUDE_total_groom_time_is_210_l3308_330818

/-- The time it takes to groom a poodle, in minutes. -/
def poodle_groom_time : ℕ := 30

/-- The time it takes to groom a terrier, in minutes. -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- The number of poodles to be groomed. -/
def num_poodles : ℕ := 3

/-- The number of terriers to be groomed. -/
def num_terriers : ℕ := 8

/-- The total grooming time for all dogs. -/
def total_groom_time : ℕ := num_poodles * poodle_groom_time + num_terriers * terrier_groom_time

theorem total_groom_time_is_210 : total_groom_time = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_is_210_l3308_330818


namespace NUMINAMATH_CALUDE_binary_calculation_l3308_330849

def binary_to_decimal (b : ℕ) : ℕ := sorry

theorem binary_calculation : 
  (binary_to_decimal 0b111111111 + binary_to_decimal 0b11111) * binary_to_decimal 0b11 = 1626 := by
  sorry

end NUMINAMATH_CALUDE_binary_calculation_l3308_330849


namespace NUMINAMATH_CALUDE_nathaniel_tickets_l3308_330847

/-- The number of tickets Nathaniel gives to each friend -/
def tickets_per_friend : ℕ := 2

/-- The number of Nathaniel's best friends -/
def num_friends : ℕ := 4

/-- The number of tickets Nathaniel has left after giving away -/
def tickets_left : ℕ := 3

/-- The initial number of tickets Nathaniel had -/
def initial_tickets : ℕ := tickets_per_friend * num_friends + tickets_left

theorem nathaniel_tickets : initial_tickets = 11 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_tickets_l3308_330847


namespace NUMINAMATH_CALUDE_color_film_fraction_l3308_330885

theorem color_film_fraction (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) : 
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 30 / 31 := by
sorry


end NUMINAMATH_CALUDE_color_film_fraction_l3308_330885


namespace NUMINAMATH_CALUDE_exam_count_proof_l3308_330857

theorem exam_count_proof (prev_avg : ℝ) (desired_avg : ℝ) (next_score : ℝ) :
  prev_avg = 84 →
  desired_avg = 86 →
  next_score = 100 →
  ∃ n : ℕ, n > 0 ∧ (n * desired_avg - (n - 1) * prev_avg = next_score) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_exam_count_proof_l3308_330857


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3308_330816

theorem sufficient_not_necessary (x : ℝ) (h : x ≠ 0) :
  (∀ x > 1, x + 1/x > 2) ∧
  (∃ x, 0 < x ∧ x < 1 ∧ x + 1/x > 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3308_330816


namespace NUMINAMATH_CALUDE_length_breadth_difference_is_32_l3308_330855

/-- Represents a rectangular plot with given dimensions and fencing costs. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- The difference between length and breadth of a rectangular plot. -/
def length_breadth_difference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- The perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

theorem length_breadth_difference_is_32 (plot : RectangularPlot)
  (h1 : plot.length = 66)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : perimeter plot = plot.total_fencing_cost / plot.fencing_cost_per_meter) :
  length_breadth_difference plot = 32 := by
  sorry


end NUMINAMATH_CALUDE_length_breadth_difference_is_32_l3308_330855


namespace NUMINAMATH_CALUDE_factor_tree_value_l3308_330881

/-- Represents a node in the modified factor tree -/
inductive TreeNode
| Prime (value : ℕ)
| Composite (value : ℕ) (left : TreeNode) (middle : TreeNode) (right : TreeNode)

/-- Calculates the value of a node in the modified factor tree -/
def nodeValue : TreeNode → ℕ
| TreeNode.Prime n => n
| TreeNode.Composite _ left middle right => nodeValue left * nodeValue middle * nodeValue right

/-- The modified factor tree structure -/
def factorTree : TreeNode :=
  TreeNode.Composite 0  -- A
    (TreeNode.Composite 0  -- B
      (TreeNode.Prime 3)
      (TreeNode.Composite 0  -- D
        (TreeNode.Prime 3)
        (TreeNode.Prime 2)
        (TreeNode.Prime 2))
      (TreeNode.Prime 3))
    (TreeNode.Prime 3)
    (TreeNode.Composite 0  -- C
      (TreeNode.Prime 5)
      (TreeNode.Composite 0  -- E
        (TreeNode.Prime 5)
        (TreeNode.Prime 2)
        (TreeNode.Prime 1))  -- Using 1 as a placeholder for the missing third child
      (TreeNode.Prime 5))

theorem factor_tree_value :
  nodeValue factorTree = 1800 := by sorry

end NUMINAMATH_CALUDE_factor_tree_value_l3308_330881


namespace NUMINAMATH_CALUDE_matching_pair_guarantee_l3308_330874

/-- The number of different colors of plates -/
def num_colors : ℕ := 5

/-- The total number of plates to be pulled out -/
def total_plates : ℕ := 6

/-- The minimum number of plates needed to guarantee a matching pair -/
def min_matching_pair : ℕ := total_plates

theorem matching_pair_guarantee :
  min_matching_pair = total_plates :=
sorry

end NUMINAMATH_CALUDE_matching_pair_guarantee_l3308_330874


namespace NUMINAMATH_CALUDE_arc_measures_l3308_330856

-- Define the circle and angles
def Circle : Type := ℝ × ℝ
def CentralAngle (c : Circle) : ℝ := 60
def InscribedAngle (c : Circle) : ℝ := 30

-- Define the theorem
theorem arc_measures (c : Circle) :
  (2 * CentralAngle c = 120) ∧ (2 * InscribedAngle c = 60) :=
by sorry

end NUMINAMATH_CALUDE_arc_measures_l3308_330856


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3308_330899

theorem min_value_abs_sum (x : ℝ) : |x - 2| + |5 - x| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3308_330899


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l3308_330842

theorem probability_of_black_ball (p_red p_white p_black : ℝ) :
  p_red = 0.41 →
  p_white = 0.27 →
  p_red + p_white + p_black = 1 →
  p_black = 0.32 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l3308_330842


namespace NUMINAMATH_CALUDE_rectangle_area_relation_l3308_330850

/-- For a rectangle with area 12 and sides of length x and y, 
    the relationship between y and x is y = 12/x -/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relation_l3308_330850


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3308_330821

theorem triangle_side_lengths 
  (a b c : ℚ) 
  (perimeter : a + b + c = 24)
  (relation : a + 2 * b = 2 * c)
  (ratio : a = (1 / 2) * b) :
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3308_330821


namespace NUMINAMATH_CALUDE_sum_five_consecutive_odd_numbers_l3308_330839

theorem sum_five_consecutive_odd_numbers (n : ℤ) :
  let middle := 2 * n + 1
  let sum := (2 * n - 3) + (2 * n - 1) + (2 * n + 1) + (2 * n + 3) + (2 * n + 5)
  sum = 5 * middle :=
by sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_odd_numbers_l3308_330839


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3308_330871

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset Int),
    (∀ n ∈ S, -15 ≤ n ∧ n ≤ 7 ∧ (n - 4) * (n + 2) * (n + 6) < -1) ∧
    (∀ n : Int, -15 ≤ n ∧ n ≤ 7 ∧ (n - 4) * (n + 2) * (n + 6) < -1 → n ∈ S) ∧
    Finset.card S = 12 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3308_330871


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3308_330806

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 where a > 0,
    if one of its asymptote equations is y = -2x, then a = 1 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/4 = 1 ∧ y = -2*x) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3308_330806


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3308_330845

theorem quadratic_equal_roots (m n : ℝ) : 
  (m = 2 ∧ n = 1) → 
  ∃ x : ℝ, x^2 - m*x + n = 0 ∧ 
  ∀ y : ℝ, y^2 - m*y + n = 0 → y = x :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3308_330845


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3308_330829

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem states that for an arithmetic sequence satisfying given conditions, the 9th term equals 7. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 2)
  (h_fifth : a 5 = 3) :
  a 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3308_330829


namespace NUMINAMATH_CALUDE_girls_in_class_l3308_330841

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_nonbinary : ℕ) 
  (h1 : ratio_girls = 3)
  (h2 : ratio_boys = 2)
  (h3 : ratio_nonbinary = 1)
  (h4 : total = 72) :
  (total * ratio_girls) / (ratio_girls + ratio_boys + ratio_nonbinary) = 36 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3308_330841


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_l3308_330866

/-- Calculates the cost of a purchase at Joe's Fast Food -/
def calculate_cost (sandwich_price : ℕ) (soda_price : ℕ) (sandwich_count : ℕ) (soda_count : ℕ) (bulk_discount : ℕ) (bulk_threshold : ℕ) : ℕ :=
  let total_items := sandwich_count + soda_count
  let subtotal := sandwich_price * sandwich_count + soda_price * soda_count
  if total_items > bulk_threshold then subtotal - bulk_discount else subtotal

/-- The cost of purchasing 6 sandwiches and 6 sodas at Joe's Fast Food is 37 dollars -/
theorem joes_fast_food_cost : calculate_cost 4 3 6 6 5 10 = 37 := by
  sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_l3308_330866


namespace NUMINAMATH_CALUDE_system_solution_range_l3308_330815

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = 2 * k - 1) → 
  (x + 2 * y = -4) → 
  (x + y > 1) → 
  (k > 4) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l3308_330815


namespace NUMINAMATH_CALUDE_abie_chips_count_l3308_330800

theorem abie_chips_count (initial : Nat) (given : Nat) (bought : Nat) (final : Nat) : 
  initial = 20 → given = 4 → bought = 6 → final = initial - given + bought → final = 22 := by
  sorry

end NUMINAMATH_CALUDE_abie_chips_count_l3308_330800


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3308_330820

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^607 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3308_330820


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3308_330832

theorem greatest_value_quadratic_inequality :
  ∀ b : ℝ, -b^2 + 8*b - 15 ≥ 0 → b ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3308_330832


namespace NUMINAMATH_CALUDE_course_class_duration_l3308_330804

/-- Proves the duration of each class in a course given the total course duration and other parameters. -/
theorem course_class_duration 
  (weeks : ℕ) 
  (unknown_classes_per_week : ℕ) 
  (known_class_duration : ℕ) 
  (homework_duration : ℕ) 
  (total_course_time : ℕ) 
  (h1 : weeks = 24)
  (h2 : unknown_classes_per_week = 2)
  (h3 : known_class_duration = 4)
  (h4 : homework_duration = 4)
  (h5 : total_course_time = 336) :
  ∃ x : ℕ, x * unknown_classes_per_week * weeks + known_class_duration * weeks + homework_duration * weeks = total_course_time ∧ x = 3 := by
  sorry

#check course_class_duration

end NUMINAMATH_CALUDE_course_class_duration_l3308_330804


namespace NUMINAMATH_CALUDE_line_through_point_forming_triangle_l3308_330827

theorem line_through_point_forming_triangle : ∃ (a b : ℝ), 
  (∀ x y : ℝ, (x / a + y / b = 1) → ((-2) / a + 2 / b = 1)) ∧ 
  (1/2 * |a * b| = 1) ∧
  ((a = -1 ∧ b = -2) ∨ (a = 2 ∧ b = 1)) := by sorry

end NUMINAMATH_CALUDE_line_through_point_forming_triangle_l3308_330827
