import Mathlib

namespace NUMINAMATH_CALUDE_tiling_polygons_are_3_4_6_l4126_412639

/-- A regular polygon can tile the plane if there exists a positive integer number of polygons that can meet at a vertex to form a complete 360° angle. -/
def can_tile (n : ℕ) : Prop :=
  n > 2 ∧ ∃ x : ℕ, x > 0 ∧ x * ((n - 2) * 180 / n) = 360

/-- The set of numbers of sides for regular polygons that can tile the plane. -/
def tiling_polygons : Set ℕ := {n : ℕ | can_tile n}

/-- Theorem stating that the only regular polygons that can tile the plane have 3, 4, or 6 sides. -/
theorem tiling_polygons_are_3_4_6 : tiling_polygons = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_tiling_polygons_are_3_4_6_l4126_412639


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l4126_412616

theorem quadratic_no_real_roots :
  ∀ x : ℝ, 2 * (x - 1)^2 + 2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l4126_412616


namespace NUMINAMATH_CALUDE_patanjali_speed_l4126_412663

/-- Represents Patanjali's walking data over three days -/
structure WalkingData where
  speed_day1 : ℝ
  hours_day1 : ℝ
  total_distance : ℝ

/-- Conditions for Patanjali's walking problem -/
def walking_conditions (data : WalkingData) : Prop :=
  data.speed_day1 * data.hours_day1 = 18 ∧
  (data.speed_day1 + 1) * (data.hours_day1 - 1) + (data.speed_day1 + 1) * data.hours_day1 = data.total_distance - 18 ∧
  data.total_distance = 62

/-- Theorem stating that Patanjali's speed on the first day was 9 miles per hour -/
theorem patanjali_speed (data : WalkingData) 
  (h : walking_conditions data) : data.speed_day1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_patanjali_speed_l4126_412663


namespace NUMINAMATH_CALUDE_reciprocal_plus_x_eq_three_implies_fraction_l4126_412698

theorem reciprocal_plus_x_eq_three_implies_fraction (x : ℝ) (h : 1/x + x = 3) :
  x^2 / (x^4 + x^2 + 1) = 1/8 := by sorry

end NUMINAMATH_CALUDE_reciprocal_plus_x_eq_three_implies_fraction_l4126_412698


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l4126_412637

theorem sum_of_squares_and_square_of_sum : (3 + 5 + 7)^2 + (3^2 + 5^2 + 7^2) = 308 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l4126_412637


namespace NUMINAMATH_CALUDE_twelve_point_sphere_l4126_412665

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a tetrahedron is equifacial -/
def isEquifacial (t : Tetrahedron) : Prop := sorry

/-- Calculates the base of an altitude for a face of the tetrahedron -/
def altitudeBase (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

/-- Calculates the midpoint of an altitude of the tetrahedron -/
def altitudeMidpoint (t : Tetrahedron) (vertex : Fin 4) : Point3D := sorry

/-- Calculates the intersection point of the altitudes of a face -/
def faceAltitudeIntersection (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

/-- Checks if a point lies on a sphere -/
def pointOnSphere (p : Point3D) (s : Sphere) : Prop := sorry

/-- Main theorem: For an equifacial tetrahedron, there exists a sphere containing
    the bases of altitudes, midpoints of altitudes, and face altitude intersections -/
theorem twelve_point_sphere (t : Tetrahedron) (h : isEquifacial t) : 
  ∃ s : Sphere, 
    (∀ face : Fin 4, pointOnSphere (altitudeBase t face) s) ∧ 
    (∀ vertex : Fin 4, pointOnSphere (altitudeMidpoint t vertex) s) ∧
    (∀ face : Fin 4, pointOnSphere (faceAltitudeIntersection t face) s) := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_sphere_l4126_412665


namespace NUMINAMATH_CALUDE_calculation_proof_l4126_412654

theorem calculation_proof : (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + 1.08 + (0.9 : ℝ)^2 = 3.11175 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4126_412654


namespace NUMINAMATH_CALUDE_quadratic_surds_problem_l4126_412682

-- Define the variables and equations
theorem quadratic_surds_problem (x y : ℝ) 
  (hA : 5 * Real.sqrt (2 * x + 1) = 5 * Real.sqrt 5)
  (hB : 3 * Real.sqrt (x + 3) = 3 * Real.sqrt 5)
  (hC : Real.sqrt (10 * x + 3 * y) = Real.sqrt 320)
  (hAB : 5 * Real.sqrt (2 * x + 1) + 3 * Real.sqrt (x + 3) = Real.sqrt (10 * x + 3 * y)) :
  Real.sqrt (2 * y - x^2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_surds_problem_l4126_412682


namespace NUMINAMATH_CALUDE_curve_equation_l4126_412609

/-- Given a curve parameterized by (x,y) = (3t + 5, 6t - 8) where t is a real number,
    prove that the equation of the line is y = 2x - 18 -/
theorem curve_equation (t : ℝ) (x y : ℝ) 
    (h1 : x = 3 * t + 5) 
    (h2 : y = 6 * t - 8) : 
  y = 2 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_l4126_412609


namespace NUMINAMATH_CALUDE_cotangent_sum_equality_l4126_412630

/-- Given a triangle ABC, A'B'C' is the triangle formed by its medians -/
def MedianTriangle (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The sum of cotangents of angles in a triangle -/
def SumOfCotangents (A B C : ℝ × ℝ) : ℝ := sorry

theorem cotangent_sum_equality (A B C : ℝ × ℝ) :
  let (A', B', C') := MedianTriangle A B C
  SumOfCotangents A B C = SumOfCotangents A' B' C' := by sorry

end NUMINAMATH_CALUDE_cotangent_sum_equality_l4126_412630


namespace NUMINAMATH_CALUDE_tim_income_percentage_l4126_412699

theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 1.44 * juan) : 
  tim = 0.9 * juan := by
  sorry

end NUMINAMATH_CALUDE_tim_income_percentage_l4126_412699


namespace NUMINAMATH_CALUDE_local_extremum_and_inequality_l4126_412608

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem local_extremum_and_inequality (a b : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (-1 - δ) (-1 + δ), f a b x ≥ f a b (-1)) ∧
  (f a b (-1) = 0) ∧
  (∀ x ∈ Set.Icc (-2) 1, f a b x ≤ 20) →
  a = 2 ∧ b = 9 ∧ (∀ m : ℝ, (∀ x ∈ Set.Icc (-2) 1, f a b x ≤ m) ↔ m ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_local_extremum_and_inequality_l4126_412608


namespace NUMINAMATH_CALUDE_angle_C_is_105_degrees_l4126_412613

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the condition
def condition (t : Triangle) : Prop :=
  |2 * Real.sin t.A - 1| + |Real.sqrt 2 / 2 - Real.cos t.B| = 0

-- State the theorem
theorem angle_C_is_105_degrees (t : Triangle) (h : condition t) :
  t.C = 7 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_105_degrees_l4126_412613


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l4126_412602

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (p^3 + 4*p^2 + 5*p - 13 = 0) →
  (q^3 + 4*q^2 + 5*q - 13 = 0) →
  (r^3 + 4*r^2 + 5*r - 13 = 0) →
  ((p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0) →
  ((q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0) →
  ((r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0) →
  w = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l4126_412602


namespace NUMINAMATH_CALUDE_four_good_numbers_l4126_412635

/-- A real number k is a "good number" if the equation (x^2 - 1)(kx^2 - 6x - 8) = 0 
    has exactly three distinct real roots. -/
def is_good_number (k : ℝ) : Prop :=
  ∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    ∀ x : ℝ, (x^2 - 1) * (k * x^2 - 6 * x - 8) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃

/-- There are exactly 4 "good numbers". -/
theorem four_good_numbers : ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ k : ℝ, k ∈ s ↔ is_good_number k :=
  sorry

end NUMINAMATH_CALUDE_four_good_numbers_l4126_412635


namespace NUMINAMATH_CALUDE_diagonalSum_is_377_l4126_412661

/-- A hexagon inscribed in a circle with given side lengths -/
structure InscribedHexagon where
  -- Define the side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ
  -- Conditions on side lengths
  AB_length : AB = 41
  other_sides : BC = 91 ∧ CD = 91 ∧ DE = 91 ∧ EF = 91 ∧ FA = 91
  -- Ensure it's inscribed in a circle (this is implicit and we don't prove it)
  inscribed : True

/-- The sum of diagonal lengths from vertex A in the inscribed hexagon -/
def diagonalSum (h : InscribedHexagon) : ℝ :=
  let AC := sorry
  let AD := sorry
  let AE := sorry
  AC + AD + AE

/-- Theorem stating that the sum of diagonal lengths from A is 377 -/
theorem diagonalSum_is_377 (h : InscribedHexagon) : diagonalSum h = 377 := by
  sorry

end NUMINAMATH_CALUDE_diagonalSum_is_377_l4126_412661


namespace NUMINAMATH_CALUDE_fraction_of_juniors_l4126_412683

theorem fraction_of_juniors (J S : ℕ) : 
  J > 0 → -- There is at least one junior
  S > 0 → -- There is at least one senior
  (J : ℚ) / 2 = (S : ℚ) * 2 / 3 → -- Half the number of juniors equals two-thirds the number of seniors
  (J : ℚ) / (J + S) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_juniors_l4126_412683


namespace NUMINAMATH_CALUDE_product_lower_bound_l4126_412685

theorem product_lower_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h₄ : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_lower_bound_l4126_412685


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l4126_412603

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a + (n - 1) * seq.d) / 2

theorem fifteenth_term_ratio
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, (sum_n seq1 n) / (sum_n seq2 n) = (9 * n + 3) / (5 * n + 35)) :
  (seq1.a + 14 * seq1.d) / (seq2.a + 14 * seq2.d) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l4126_412603


namespace NUMINAMATH_CALUDE_player_one_wins_l4126_412623

/-- A cubic polynomial with integer coefficients -/
def CubicPolynomial (a b c : ℤ) : ℤ → ℤ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- A proposition stating that a cubic polynomial has three integer roots -/
def HasThreeIntegerRoots (p : ℤ → ℤ) : Prop :=
  ∃ x y z : ℤ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ p x = 0 ∧ p y = 0 ∧ p z = 0

theorem player_one_wins :
  ∀ a b : ℤ, ∃ c : ℤ, HasThreeIntegerRoots (CubicPolynomial a b c) :=
by sorry

end NUMINAMATH_CALUDE_player_one_wins_l4126_412623


namespace NUMINAMATH_CALUDE_meaningful_expression_l4126_412649

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x + 1)) ↔ x > -1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l4126_412649


namespace NUMINAMATH_CALUDE_division_problem_l4126_412632

theorem division_problem : ∃ (n : ℕ), n = 12401 ∧ n / 163 = 76 ∧ n % 163 = 13 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4126_412632


namespace NUMINAMATH_CALUDE_min_distinct_terms_scalene_triangle_l4126_412615

/-- Represents a scalene triangle with side lengths and angles -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ α ≠ β ∧ β ≠ γ ∧ α ≠ γ
  angle_sum : α + β + γ = π
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ
  law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ

/-- The minimum number of distinct terms in the 6-tuple (a,b,c,α,β,γ) for a scalene triangle is 4 -/
theorem min_distinct_terms_scalene_triangle (t : ScaleneTriangle) :
  ∃ (s : Finset ℝ), s.card = 4 ∧ {t.a, t.b, t.c, t.α, t.β, t.γ} ⊆ s :=
sorry

end NUMINAMATH_CALUDE_min_distinct_terms_scalene_triangle_l4126_412615


namespace NUMINAMATH_CALUDE_angle_properties_l4126_412672

def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180

def isSecondQuadrantAngle (β : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 90 < β ∧ β < k * 360 + 180

def isFirstQuadrantAngle (γ : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < γ ∧ γ < k * 360 + 90

theorem angle_properties :
  (∀ α : ℝ, isObtuseAngle α → isSecondQuadrantAngle α) ∧
  (∃ β γ : ℝ, isSecondQuadrantAngle β ∧ isFirstQuadrantAngle γ ∧ β < γ) ∧
  (∃ δ : ℝ, 90 < δ ∧ ¬ isObtuseAngle δ) ∧
  ¬ isSecondQuadrantAngle (-165) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l4126_412672


namespace NUMINAMATH_CALUDE_susan_chair_count_l4126_412611

/-- The number of chairs in Susan's house -/
def total_chairs (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ := red + yellow + blue

/-- Susan's chair collection -/
structure SusanChairs where
  red : ℕ
  yellow : ℕ
  blue : ℕ
  red_count : red = 5
  yellow_count : yellow = 4 * red
  blue_count : blue = yellow - 2

theorem susan_chair_count (s : SusanChairs) : total_chairs s.red s.yellow s.blue = 43 := by
  sorry

end NUMINAMATH_CALUDE_susan_chair_count_l4126_412611


namespace NUMINAMATH_CALUDE_area_of_specific_configuration_l4126_412670

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- Represents the stacked triangles configuration -/
structure StackedTriangles where
  base_triangle : EquilateralTriangle
  rotation_angles : Fin 4 → ℝ

/-- Calculates the area of the resulting geometric figure -/
def area_of_stacked_triangles (st : StackedTriangles) : ℝ :=
  sorry

/-- The main theorem stating the area of the specific configuration -/
theorem area_of_specific_configuration :
  let base_triangle := EquilateralTriangle.mk 8
  let rotation_angles := λ i => match i with
    | 0 => 0
    | 1 => π/4
    | 2 => π/2
    | 3 => 3*π/4
  let stacked_triangles := StackedTriangles.mk base_triangle rotation_angles
  area_of_stacked_triangles stacked_triangles = 52 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_specific_configuration_l4126_412670


namespace NUMINAMATH_CALUDE_hexagon_extended_side_length_l4126_412673

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y on the extension of side CD such that CY = 4CD -/
def extend_side (h : RegularHexagon) (CD : ℝ) (Y : ℝ) : Prop :=
  CD = h.side_length ∧ Y = 4 * CD

/-- The length of segment FY in the described configuration -/
def segment_FY_length (h : RegularHexagon) (Y : ℝ) : ℝ := sorry

/-- Theorem stating the length of FY is 5.5√3 -/
theorem hexagon_extended_side_length (h : RegularHexagon) (CD Y : ℝ) 
  (h_extend : extend_side h CD Y) : 
  segment_FY_length h Y = 5.5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_extended_side_length_l4126_412673


namespace NUMINAMATH_CALUDE_intersection_S_T_l4126_412693

def S : Set ℝ := {x | (x - 3) / (x - 6) ≤ 0}

def T : Set ℝ := {2, 3, 4, 5, 6}

theorem intersection_S_T : S ∩ T = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_S_T_l4126_412693


namespace NUMINAMATH_CALUDE_floor_difference_inequality_l4126_412692

theorem floor_difference_inequality (x y : ℝ) : 
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by sorry

end NUMINAMATH_CALUDE_floor_difference_inequality_l4126_412692


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l4126_412617

-- Define the complex numbers p, q, r
variable (p q r : ℂ)

-- Define the conditions
def is_equilateral_triangle (p q r : ℂ) : Prop :=
  Complex.abs (q - p) = 24 ∧ Complex.abs (r - q) = 24 ∧ Complex.abs (p - r) = 24

-- State the theorem
theorem equilateral_triangle_sum_product (h1 : is_equilateral_triangle p q r) 
  (h2 : Complex.abs (p + q + r) = 48) : 
  Complex.abs (p*q + p*r + q*r) = 768 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l4126_412617


namespace NUMINAMATH_CALUDE_triangle_inequality_l4126_412629

theorem triangle_inequality (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = (a * b * c) / (4 * area))
  (h_inradius : r = (2 * area) / (a + b + c))
  (h_area : area = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) / 16)) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4126_412629


namespace NUMINAMATH_CALUDE_matrix_inverse_l4126_412634

theorem matrix_inverse (x : ℝ) (h : x ≠ -12) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, x; -2, 6]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![6 / (24 + 2*x), -x / (24 + 2*x); 2 / (24 + 2*x), 4 / (24 + 2*x)]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry


end NUMINAMATH_CALUDE_matrix_inverse_l4126_412634


namespace NUMINAMATH_CALUDE_partner_c_investment_l4126_412624

/-- Calculates the investment of partner C in a partnership business --/
theorem partner_c_investment 
  (a_investment : ℕ) 
  (b_investment : ℕ) 
  (total_profit : ℕ) 
  (a_profit_share : ℕ) 
  (h1 : a_investment = 6300)
  (h2 : b_investment = 4200)
  (h3 : total_profit = 12300)
  (h4 : a_profit_share = 3690) :
  ∃ c_investment : ℕ, 
    c_investment = 10500 ∧ 
    (a_investment : ℚ) / (a_investment + b_investment + c_investment : ℚ) = 
    (a_profit_share : ℚ) / (total_profit : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_partner_c_investment_l4126_412624


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l4126_412686

theorem sum_of_cyclic_equations (x y z : ℝ) 
  (h1 : x + y = 1) 
  (h2 : y + z = 1) 
  (h3 : z + x = 1) : 
  x + y + z = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l4126_412686


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l4126_412675

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + (2*k - 1)*x₁ - k - 1 = 0) → 
  (x₂^2 + (2*k - 1)*x₂ - k - 1 = 0) → 
  (x₁ + x₂ - 4*x₁*x₂ = 2) → 
  (k = -3/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l4126_412675


namespace NUMINAMATH_CALUDE_simplify_expression_l4126_412688

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = 12/5 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4126_412688


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_three_l4126_412655

def is_divisible_by_three (n : ℕ) : Prop :=
  ∃ k : ℤ, 9 * (n - 1)^3 - 3 * n^3 + 19 * n + 27 = 3 * k

theorem largest_n_divisible_by_three :
  (∀ m : ℕ, m < 50000 → is_divisible_by_three m → m ≤ 49998) ∧
  (49998 < 50000) ∧
  is_divisible_by_three 49998 :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_three_l4126_412655


namespace NUMINAMATH_CALUDE_cantor_set_cardinality_cantor_set_operations_l4126_412679

-- Define the Cantor set
def CantorSet : Set ℝ := sorry

-- Theorem for part (a)
theorem cantor_set_cardinality : Cardinal.mk CantorSet = Cardinal.mk (Set.Icc 0 1) := by sorry

-- Define the sum and difference operations on sets
def setSum (A B : Set ℝ) : Set ℝ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a + b}
def setDiff (A B : Set ℝ) : Set ℝ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a - b}

-- Theorem for part (b)
theorem cantor_set_operations :
  (setSum CantorSet CantorSet = Set.Icc 0 2) ∧
  (setDiff CantorSet CantorSet = Set.Icc (-1) 1) := by sorry

end NUMINAMATH_CALUDE_cantor_set_cardinality_cantor_set_operations_l4126_412679


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l4126_412647

/-- Base prime representation of a natural number -/
def BasePrimeRepr : ℕ → List ℕ := sorry

/-- Check if a list represents a valid base prime representation -/
def IsValidBasePrimeRepr (l : List ℕ) : Prop := sorry

theorem base_prime_repr_360 :
  let repr := BasePrimeRepr 360
  IsValidBasePrimeRepr repr ∧ repr = [3, 2, 1] := by sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l4126_412647


namespace NUMINAMATH_CALUDE_tony_errands_halfway_distance_l4126_412695

theorem tony_errands_halfway_distance (groceries haircut doctor : ℕ) 
  (h1 : groceries = 10)
  (h2 : haircut = 15)
  (h3 : doctor = 5) :
  (groceries + haircut + doctor) / 2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_tony_errands_halfway_distance_l4126_412695


namespace NUMINAMATH_CALUDE_hair_group_existence_l4126_412625

theorem hair_group_existence (population : ℕ) (max_hairs : ℕ) 
  (h1 : population ≥ 8000000) 
  (h2 : max_hairs = 400000) : 
  ∃ (hair_count : ℕ), hair_count ≤ max_hairs ∧ 
  (∃ (group : Finset (Fin population)), 
    group.card ≥ 20 ∧ 
    ∀ (person : Fin population), person ∈ group → 
      (∃ (f : Fin population → ℕ), f person = hair_count ∧ f person ≤ max_hairs)) :=
sorry

end NUMINAMATH_CALUDE_hair_group_existence_l4126_412625


namespace NUMINAMATH_CALUDE_range_of_a_l4126_412658

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 7)
  (h_f2 : f 2 > 1)
  (h_f2014 : f 2014 = (a + 3) / (a - 3)) :
  0 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4126_412658


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely_l4126_412622

theorem abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely (a b : ℝ) :
  (∀ a b : ℝ, |a - b| = |a| - |b| → a * b ≥ 0) ∧
  (∃ a b : ℝ, a * b ≥ 0 ∧ |a - b| ≠ |a| - |b|) :=
by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely_l4126_412622


namespace NUMINAMATH_CALUDE_expected_socks_theorem_l4126_412656

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ := 2 * n

/-- Theorem: The expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_theorem (n : ℕ) : expected_socks n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_expected_socks_theorem_l4126_412656


namespace NUMINAMATH_CALUDE_unique_increasing_function_l4126_412627

def f (x : ℕ) : ℤ := x^3 - 1

theorem unique_increasing_function :
  (∀ x y : ℕ, x < y → f x < f y) ∧
  f 2 = 7 ∧
  (∀ m n : ℕ, f (m * n) = f m + f n + f m * f n) ∧
  (∀ g : ℕ → ℤ, 
    (∀ x y : ℕ, x < y → g x < g y) →
    g 2 = 7 →
    (∀ m n : ℕ, g (m * n) = g m + g n + g m * g n) →
    ∀ x : ℕ, g x = f x) :=
by sorry

end NUMINAMATH_CALUDE_unique_increasing_function_l4126_412627


namespace NUMINAMATH_CALUDE_inequality_proof_l4126_412678

theorem inequality_proof (p q : ℝ) (m n : ℕ+) 
  (h1 : p ≥ 0) (h2 : q ≥ 0) (h3 : p + q = 1) : 
  (1 - p ^ (m : ℝ)) ^ (n : ℝ) + (1 - q ^ (n : ℝ)) ^ (m : ℝ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4126_412678


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l4126_412691

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  1/7 < a ∧ a < 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l4126_412691


namespace NUMINAMATH_CALUDE_amulet_price_is_40_l4126_412643

/-- Calculates the selling price of amulets given the following conditions:
  * Dirk sells amulets for 2 days
  * Each day he sells 25 amulets
  * Each amulet costs him $30 to make
  * He gives 10% of his revenue to the faire
  * He made a profit of $300
-/
def amulet_price (days : ℕ) (amulets_per_day : ℕ) (cost_per_amulet : ℕ) 
                 (faire_percentage : ℚ) (profit : ℕ) : ℚ :=
  let total_amulets := days * amulets_per_day
  let total_cost := total_amulets * cost_per_amulet
  let x := (profit + total_cost) / (total_amulets * (1 - faire_percentage))
  x

theorem amulet_price_is_40 :
  amulet_price 2 25 30 (1/10) 300 = 40 := by
  sorry

end NUMINAMATH_CALUDE_amulet_price_is_40_l4126_412643


namespace NUMINAMATH_CALUDE_opposite_and_abs_of_2_minus_sqrt_3_l4126_412610

theorem opposite_and_abs_of_2_minus_sqrt_3 :
  let x : ℝ := 2 - Real.sqrt 3
  (- x = Real.sqrt 3 - 2) ∧ (abs x = 2 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_opposite_and_abs_of_2_minus_sqrt_3_l4126_412610


namespace NUMINAMATH_CALUDE_maggie_yellow_packs_l4126_412690

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := 4

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := (total_balls - (red_packs + green_packs) * balls_per_pack) / balls_per_pack

theorem maggie_yellow_packs : yellow_packs = 8 := by
  sorry

end NUMINAMATH_CALUDE_maggie_yellow_packs_l4126_412690


namespace NUMINAMATH_CALUDE_at_least_one_negative_l4126_412641

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l4126_412641


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l4126_412671

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  carbon_count * carbon_weight + 
  hydrogen_count * hydrogen_weight + 
  oxygen_count * oxygen_weight

theorem compound_molecular_weight : 
  molecular_weight = 65.048 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l4126_412671


namespace NUMINAMATH_CALUDE_volume_added_equals_expression_l4126_412667

/-- Represents a cylindrical tank lying on its side -/
structure CylindricalTank where
  radius : ℝ
  length : ℝ

/-- Calculates the volume of water added to the tank -/
def volumeAdded (tank : CylindricalTank) (initialDepth finalDepth : ℝ) : ℝ := sorry

/-- The main theorem to prove -/
theorem volume_added_equals_expression (tank : CylindricalTank) :
  tank.radius = 10 →
  tank.length = 30 →
  volumeAdded tank 5 (10 + 5 * Real.sqrt 2) = 1250 * Real.pi + 1500 + 750 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_added_equals_expression_l4126_412667


namespace NUMINAMATH_CALUDE_zeros_of_f_range_of_b_l4126_412636

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

-- Part 1
theorem zeros_of_f (b : ℝ) :
  f b 0 = f b 4 → (∃ x : ℝ, f b x = 0) ∧ (∀ x : ℝ, f b x = 0 → x = 3 ∨ x = 1) :=
sorry

-- Part 2
theorem range_of_b :
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ f b x = 0 ∧ f b y = 0) → b > 4 :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_range_of_b_l4126_412636


namespace NUMINAMATH_CALUDE_integer_fraction_count_l4126_412660

theorem integer_fraction_count : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ ∃ k : ℕ, k > 0 ∧ 1722 = k * (m^2 - 3)) ∧ 
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_count_l4126_412660


namespace NUMINAMATH_CALUDE_jade_lego_tower_remaining_pieces_l4126_412669

/-- Calculates the number of remaining Lego pieces after building a tower. -/
def remaining_pieces (initial_pieces : ℕ) (pieces_per_level : ℕ) (levels : ℕ) : ℕ :=
  initial_pieces - pieces_per_level * levels

/-- Proves that given 100 initial pieces, 7 pieces per level, and 11 levels built,
    the number of remaining pieces is equal to 23. -/
theorem jade_lego_tower_remaining_pieces :
  remaining_pieces 100 7 11 = 23 := by
  sorry

end NUMINAMATH_CALUDE_jade_lego_tower_remaining_pieces_l4126_412669


namespace NUMINAMATH_CALUDE_cubes_form_larger_cube_l4126_412640

/-- A function that determines if n cubes can form a larger cube -/
def can_form_cube (n : ℕ) : Prop :=
  ∃ (side : ℕ), side^3 = n

/-- The theorem stating that for all natural numbers greater than 70,
    it's possible to select that many cubes to form a larger cube -/
theorem cubes_form_larger_cube :
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → can_form_cube n :=
sorry

end NUMINAMATH_CALUDE_cubes_form_larger_cube_l4126_412640


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4126_412659

/-- The function f(x) = x^3 + 2x -/
def f (x : ℝ) : ℝ := x^3 + 2*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, f 1)

/-- Theorem: The equation of the tangent line to y = f(x) at (1, f(1)) is 5x - y - 2 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 5*x - y - 2 = 0} ↔ 
  y - (f 1) = (f_derivative 1) * (x - 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4126_412659


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l4126_412620

/-- The curve function f(x) = x^2 + 11 -/
def f (x : ℝ) : ℝ := x^2 + 11

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at P -/
def m : ℝ := 2 * P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - m * P.1

theorem tangent_y_intercept :
  b = 10 := by sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l4126_412620


namespace NUMINAMATH_CALUDE_girls_average_score_l4126_412652

theorem girls_average_score 
  (boys_avg : ℝ)
  (class_avg : ℝ)
  (boy_girl_ratio : ℝ)
  (h1 : boys_avg = 90)
  (h2 : class_avg = 94)
  (h3 : boy_girl_ratio = 0.5)
  : ∃ girls_avg : ℝ, girls_avg = 96 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_score_l4126_412652


namespace NUMINAMATH_CALUDE_six_n_divisors_l4126_412626

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem six_n_divisors (n : ℕ) 
  (h1 : divisor_count n = 10)
  (h2 : divisor_count (2 * n) = 20)
  (h3 : divisor_count (3 * n) = 15) :
  divisor_count (6 * n) = 30 := by
  sorry

end NUMINAMATH_CALUDE_six_n_divisors_l4126_412626


namespace NUMINAMATH_CALUDE_fraction_equality_l4126_412645

theorem fraction_equality (n : ℚ) : (4 + n) / (7 + n) = 3 / 4 ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4126_412645


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l4126_412689

theorem six_digit_numbers_with_zero (total_six_digit : ℕ) (no_zero_six_digit : ℕ) :
  total_six_digit = 900000 →
  no_zero_six_digit = 531441 →
  total_six_digit - no_zero_six_digit = 368559 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l4126_412689


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l4126_412653

theorem smallest_third_term_geometric_progression 
  (a : ℝ) -- Common difference of the arithmetic progression
  (h1 : (5 : ℝ) < 9 + a) -- Ensure the second term of GP is positive
  (h2 : 9 + a < 37 + 2*a) -- Ensure the third term of GP is greater than the second
  (h3 : (9 + a)^2 = 5*(37 + 2*a)) -- Condition for geometric progression
  : 
  ∃ (x : ℝ), x = 29 - 20*Real.sqrt 6 ∧ 
  x ≤ 37 + 2*a ∧
  ∀ (y : ℝ), y = 37 + 2*a → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l4126_412653


namespace NUMINAMATH_CALUDE_negation_of_existence_cube_positive_negation_l4126_412621

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem cube_positive_negation : 
  (¬∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cube_positive_negation_l4126_412621


namespace NUMINAMATH_CALUDE_expression_bounds_l4126_412680

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) : 
  let expr := Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
              Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2)
  2 * Real.sqrt 2 ≤ expr ∧ expr ≤ 4 ∧ 
  (∃ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
              expr = 2 * Real.sqrt 2) ∧
  (∃ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
              expr = 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l4126_412680


namespace NUMINAMATH_CALUDE_least_integer_with_two_prime_factors_l4126_412676

/-- A function that returns true if a number has exactly two prime factors -/
def has_two_prime_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q

/-- The main theorem stating that 33 is the least positive integer satisfying the condition -/
theorem least_integer_with_two_prime_factors :
  (∀ m : ℕ, m > 0 ∧ m < 33 → ¬(has_two_prime_factors m ∧ has_two_prime_factors (m + 1) ∧ has_two_prime_factors (m + 2))) ∧
  (has_two_prime_factors 33 ∧ has_two_prime_factors 34 ∧ has_two_prime_factors 35) :=
by sorry

#check least_integer_with_two_prime_factors

end NUMINAMATH_CALUDE_least_integer_with_two_prime_factors_l4126_412676


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4126_412666

-- Define the sets A and B
def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4126_412666


namespace NUMINAMATH_CALUDE_unit_digit_of_seven_power_ten_l4126_412601

theorem unit_digit_of_seven_power_ten (n : ℕ) : n = 10 → (7^n) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_seven_power_ten_l4126_412601


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l4126_412605

/-- The analysis method for proving inequalities -/
def analysis_method : Type := Unit

/-- A condition that makes an inequality hold -/
def condition : Type := Unit

/-- Predicate indicating if a condition is sufficient -/
def is_sufficient (c : condition) : Prop := sorry

/-- The condition sought by the analysis method -/
def sought_condition (m : analysis_method) : condition := sorry

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (m : analysis_method), is_sufficient (sought_condition m) := by
  sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l4126_412605


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l4126_412644

-- Define the matrix evaluation rule
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define our specific matrix as a function of x
def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3*x+1, 2; 2*x, x+1]

-- State the theorem
theorem matrix_equation_solution :
  ∀ x : ℝ, matrix_value (M x 0 0) (M x 1 1) (M x 0 1) (M x 1 0) = 5 ↔ 
  x = 2 * Real.sqrt 3 / 3 ∨ x = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l4126_412644


namespace NUMINAMATH_CALUDE_complex_distance_sum_constant_l4126_412697

theorem complex_distance_sum_constant (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_constant_l4126_412697


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l4126_412651

theorem factorial_difference_quotient : (Nat.factorial 15 - Nat.factorial 14 - Nat.factorial 13) / Nat.factorial 11 = 30420 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l4126_412651


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l4126_412614

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  ∃ (k : ℤ), (12*x + 2) * (12*x + 6) * (12*x + 10) * (6*x + 3) = 864 * k ∧
  ∀ (m : ℤ), m > 864 → ¬(∀ (y : ℤ), Odd y →
    ∃ (l : ℤ), (12*y + 2) * (12*y + 6) * (12*y + 10) * (6*y + 3) = m * l) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l4126_412614


namespace NUMINAMATH_CALUDE_sophie_and_hannah_fruits_l4126_412664

/-- The number of fruits eaten by Sophie and Hannah in 30 days -/
def total_fruits (sophie_oranges_per_day : ℕ) (hannah_grapes_per_day : ℕ) : ℕ :=
  30 * (sophie_oranges_per_day + hannah_grapes_per_day)

/-- Theorem stating that Sophie and Hannah eat 1800 fruits in 30 days -/
theorem sophie_and_hannah_fruits :
  total_fruits 20 40 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sophie_and_hannah_fruits_l4126_412664


namespace NUMINAMATH_CALUDE_total_people_is_36_l4126_412638

/-- A circular arrangement of people shaking hands -/
structure HandshakeCircle where
  people : ℕ
  handshakes : ℕ
  smallest_set : ℕ

/-- The number of people in the circle equals the number of handshakes -/
def handshakes_equal_people (circle : HandshakeCircle) : Prop :=
  circle.people = circle.handshakes

/-- The smallest set size plus the remaining people equals the total people -/
def smallest_set_property (circle : HandshakeCircle) : Prop :=
  circle.smallest_set + (circle.people - circle.smallest_set) = circle.people

/-- The main theorem: given the conditions, prove the total number of people is 36 -/
theorem total_people_is_36 (circle : HandshakeCircle) 
    (h1 : circle.handshakes = 36)
    (h2 : circle.smallest_set = 12)
    (h3 : handshakes_equal_people circle)
    (h4 : smallest_set_property circle) : 
  circle.people = 36 := by
  sorry

#check total_people_is_36

end NUMINAMATH_CALUDE_total_people_is_36_l4126_412638


namespace NUMINAMATH_CALUDE_barrel_capacity_l4126_412600

theorem barrel_capacity (total_capacity : ℝ) (increase : ℝ) (decrease : ℝ)
  (h1 : total_capacity = 7000)
  (h2 : increase = 1000)
  (h3 : decrease = 4000) :
  ∃ (x y : ℝ),
    x + y = total_capacity ∧
    x = 6400 ∧
    y = 600 ∧
    x / (total_capacity + increase) + y / (total_capacity - decrease) = 1 :=
by sorry

end NUMINAMATH_CALUDE_barrel_capacity_l4126_412600


namespace NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l4126_412606

-- Define the condition for a hyperbola
def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x^2 + b * y^2 = c ∧ c ≠ 0 ∧ a * b < 0

-- State the theorem
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ a b c : ℝ, is_hyperbola a b c → a * b < 0) ∧
  (∃ a b c : ℝ, a * b < 0 ∧ ¬(is_hyperbola a b c)) :=
sorry

end NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l4126_412606


namespace NUMINAMATH_CALUDE_inequality_proof_l4126_412633

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : 1/a + 1/b + 1/c = 1) :
  a^(a*b*c) + b^(b*c*a) + c^(c*a*b) ≥ 27*b*c + 27*c*a + 27*a*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4126_412633


namespace NUMINAMATH_CALUDE_lily_cost_is_four_l4126_412677

/-- Represents the cost structure for wedding reception decorations --/
structure WeddingDecoration where
  numTables : Nat
  tableclothCost : Nat
  placeSettingCost : Nat
  placeSettingsPerTable : Nat
  rosesPerCenterpiece : Nat
  roseCost : Nat
  liliesPerCenterpiece : Nat
  totalCost : Nat

/-- Calculates the cost of each lily given the wedding decoration details --/
def lilyCost (d : WeddingDecoration) : Rat :=
  let tableCostWithoutLilies := d.tableclothCost + 
                                d.placeSettingCost * d.placeSettingsPerTable + 
                                d.rosesPerCenterpiece * d.roseCost
  let totalCostWithoutLilies := d.numTables * tableCostWithoutLilies
  let totalLilyCost := d.totalCost - totalCostWithoutLilies
  let totalLilies := d.numTables * d.liliesPerCenterpiece
  totalLilyCost / totalLilies

/-- Theorem stating that the lily cost for the given wedding decoration is $4 --/
theorem lily_cost_is_four (d : WeddingDecoration) 
  (h1 : d.numTables = 20)
  (h2 : d.tableclothCost = 25)
  (h3 : d.placeSettingCost = 10)
  (h4 : d.placeSettingsPerTable = 4)
  (h5 : d.rosesPerCenterpiece = 10)
  (h6 : d.roseCost = 5)
  (h7 : d.liliesPerCenterpiece = 15)
  (h8 : d.totalCost = 3500) : 
  lilyCost d = 4 := by
  sorry

#eval lilyCost {
  numTables := 20,
  tableclothCost := 25,
  placeSettingCost := 10,
  placeSettingsPerTable := 4,
  rosesPerCenterpiece := 10,
  roseCost := 5,
  liliesPerCenterpiece := 15,
  totalCost := 3500
}

end NUMINAMATH_CALUDE_lily_cost_is_four_l4126_412677


namespace NUMINAMATH_CALUDE_lasso_probability_l4126_412648

theorem lasso_probability (p : ℝ) (n : ℕ) (h1 : p = 1 / 2) (h2 : n = 4) :
  1 - (1 - p) ^ n = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_lasso_probability_l4126_412648


namespace NUMINAMATH_CALUDE_set_empty_properties_l4126_412618

theorem set_empty_properties (A : Set α) :
  let p := A ∩ ∅ = ∅
  let q := A ∪ ∅ = A
  (p ∧ q) ∧ (¬p ∨ q) := by sorry

end NUMINAMATH_CALUDE_set_empty_properties_l4126_412618


namespace NUMINAMATH_CALUDE_fraction_evaluation_l4126_412684

theorem fraction_evaluation : 
  (1 / 5 - 1 / 7) / (3 / 8 + 2 / 9) = 144 / 1505 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l4126_412684


namespace NUMINAMATH_CALUDE_clara_lego_count_l4126_412612

/-- The number of legos each person has -/
structure LegoCount where
  kent : ℕ
  bruce : ℕ
  simon : ℕ
  clara : ℕ

/-- The conditions of the lego problem -/
def lego_problem (l : LegoCount) : Prop :=
  l.kent = 80 ∧
  l.bruce = l.kent + 30 ∧
  l.simon = l.bruce + (l.bruce / 4) ∧
  l.clara = (l.simon + l.kent) - ((l.simon + l.kent) / 10)

/-- The theorem stating Clara's lego count -/
theorem clara_lego_count (l : LegoCount) (h : lego_problem l) : l.clara = 197 := by
  sorry


end NUMINAMATH_CALUDE_clara_lego_count_l4126_412612


namespace NUMINAMATH_CALUDE_percentage_calculation_l4126_412607

theorem percentage_calculation (N : ℚ) (h : (1/2) * N = 16) : (3/4) * N = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l4126_412607


namespace NUMINAMATH_CALUDE_catch_up_equation_correct_l4126_412604

/-- Represents the problem of two horses racing, where one starts earlier than the other. -/
structure HorseRace where
  fast_speed : ℕ  -- Speed of the faster horse in miles per day
  slow_speed : ℕ  -- Speed of the slower horse in miles per day
  head_start : ℕ  -- Number of days the slower horse starts earlier

/-- The equation representing when the faster horse catches up to the slower horse -/
def catch_up_equation (race : HorseRace) (x : ℝ) : Prop :=
  (race.fast_speed : ℝ) * x = (race.slow_speed : ℝ) * (x + race.head_start)

/-- The specific race described in the problem -/
def zhu_shijie_race : HorseRace :=
  { fast_speed := 240
  , slow_speed := 150
  , head_start := 12 }

/-- Theorem stating that the given equation correctly represents the race situation -/
theorem catch_up_equation_correct :
  catch_up_equation zhu_shijie_race = fun x => 240 * x = 150 * (x + 12) :=
by sorry


end NUMINAMATH_CALUDE_catch_up_equation_correct_l4126_412604


namespace NUMINAMATH_CALUDE_difference_half_and_sixth_l4126_412657

theorem difference_half_and_sixth (x : ℝ) (hx : x = 1/2 - 1/6) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_difference_half_and_sixth_l4126_412657


namespace NUMINAMATH_CALUDE_solution_set_min_value_g_inequality_proof_l4126_412694

-- Define the absolute value function
def f (x : ℝ) : ℝ := |x|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Statement for part 1
theorem solution_set (x : ℝ) :
  f ((1 / 2^x) - 2) ≤ 1 ↔ x ∈ Set.Icc (Real.log 3 / Real.log 2) 0 :=
sorry

-- Statement for part 2
theorem min_value_g :
  ∃ (m : ℝ), m = 1 ∧ ∀ x, g x ≥ m :=
sorry

-- Statement for part 3
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_min_value_g_inequality_proof_l4126_412694


namespace NUMINAMATH_CALUDE_trio_ball_theorem_l4126_412631

/-- The number of minutes each child plays in the trio-ball game -/
def trio_ball_play_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) : ℕ :=
  (total_time * players_per_game) / num_children

/-- Theorem stating that each child plays for 60 minutes in the given scenario -/
theorem trio_ball_theorem :
  trio_ball_play_time 120 6 3 = 60 := by
  sorry

#eval trio_ball_play_time 120 6 3

end NUMINAMATH_CALUDE_trio_ball_theorem_l4126_412631


namespace NUMINAMATH_CALUDE_sphere_surface_area_l4126_412650

theorem sphere_surface_area (V : ℝ) (r : ℝ) (h : V = 72 * Real.pi) :
  4 * Real.pi * r^2 = 36 * Real.pi * (2^(1/3))^2 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l4126_412650


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l4126_412674

theorem diophantine_equation_solution :
  ∀ a b c : ℕ+,
  a + b = c - 1 →
  a^3 + b^3 = c^2 - 1 →
  ((a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l4126_412674


namespace NUMINAMATH_CALUDE_mod_inverse_sum_five_l4126_412681

theorem mod_inverse_sum_five : ∃ (a b : ℤ), 
  (5 * a) % 17 = 1 ∧ 
  (5^2 * b) % 17 = 1 ∧ 
  (a + b) % 17 = 14 := by
sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_five_l4126_412681


namespace NUMINAMATH_CALUDE_not_power_of_two_concat_l4126_412687

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def concat_numbers (nums : List ℕ) : ℕ := sorry

theorem not_power_of_two_concat :
  ∀ (perm : List ℕ),
    (∀ n ∈ perm, is_five_digit n) →
    (perm.length = 88889) →
    (∀ n, is_five_digit n → n ∈ perm) →
    ¬ ∃ k : ℕ, concat_numbers perm = 2^k :=
by sorry

end NUMINAMATH_CALUDE_not_power_of_two_concat_l4126_412687


namespace NUMINAMATH_CALUDE_cone_lateral_area_l4126_412668

/-- Given a cone with base circumference 4π and slant height 3, its lateral area is 6π. -/
theorem cone_lateral_area (c : ℝ) (l : ℝ) (h1 : c = 4 * Real.pi) (h2 : l = 3) :
  let r := c / (2 * Real.pi)
  π * r * l = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l4126_412668


namespace NUMINAMATH_CALUDE_modulus_of_z_l4126_412662

theorem modulus_of_z (z : ℂ) (h : z * (1 - Complex.I) = -1 - Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4126_412662


namespace NUMINAMATH_CALUDE_train_passing_time_l4126_412696

/-- The time taken for two trains to completely pass each other -/
theorem train_passing_time (length_A length_B : ℝ) (speed_A speed_B : ℝ) 
  (h1 : length_A = 150)
  (h2 : length_B = 150)
  (h3 : speed_A = 54 * (5/18))
  (h4 : speed_B = 36 * (5/18))
  (h5 : speed_A > 0)
  (h6 : speed_B > 0) :
  (length_A + length_B) / (speed_A + speed_B) = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l4126_412696


namespace NUMINAMATH_CALUDE_f_bounded_g_bounded_l4126_412619

-- Define the functions f and g
def f (x : ℝ) := 3 * x - 4 * x^3
def g (x : ℝ) := 3 * x - 4 * x^3

-- Theorem for function f
theorem f_bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |f x| ≤ 1 := by
  sorry

-- Theorem for function g
theorem g_bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |g x| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_bounded_g_bounded_l4126_412619


namespace NUMINAMATH_CALUDE_cube_volume_decomposition_l4126_412628

theorem cube_volume_decomposition (x : ℝ) (hx : x > 0) :
  ∃ (y z : ℝ),
    y = (3/2 + Real.sqrt (3/2)) * x ∧
    z = (3/2 - Real.sqrt (3/2)) * x ∧
    y^3 + z^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_decomposition_l4126_412628


namespace NUMINAMATH_CALUDE_rectangle_fold_l4126_412646

/-- Given a rectangle ABCD with side lengths AB = CD = 2 and BC = AD = 1,
    and points E on AB and F on CD such that AE = DF = x,
    if A and D coincide at point G on diagonal BD when folded along DE and BF,
    then x = 0. -/
theorem rectangle_fold (A B C D E F G : ℝ × ℝ) (x : ℝ) : 
  (B.1 - A.1 = 2) →  -- AB = 2
  (C.1 - D.1 = 2) →  -- CD = 2
  (C.2 - B.2 = 1) →  -- BC = 1
  (A.2 - D.2 = 1) →  -- AD = 1
  (E.1 - A.1 = x) →  -- AE = x
  (D.1 - F.1 = x) →  -- DF = x
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = 5 →  -- G is on diagonal BD
  (G = A) →  -- A and D coincide at G when folded
  (G = D) →
  x = 0 := by sorry

end NUMINAMATH_CALUDE_rectangle_fold_l4126_412646


namespace NUMINAMATH_CALUDE_parallel_lines_to_plane_not_always_parallel_l4126_412642

structure Plane where

structure Line where

def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

theorem parallel_lines_to_plane_not_always_parallel (m n : Line) (α : Plane) : 
  m ≠ n → 
  ¬(parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_to_plane_not_always_parallel_l4126_412642
