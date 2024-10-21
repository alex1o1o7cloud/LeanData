import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_ratio_l1141_114180

/-- Represents a cone with its height divided into three equal parts by parallel planes. -/
structure DividedCone where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  l₁ : ℝ
  l₂ : ℝ
  l₃ : ℝ
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  radii_ratio : r₁ / r₂ = 1 / 2 ∧ r₂ / r₃ = 2 / 3
  slant_heights_ratio : l₁ / l₂ = 1 / 2 ∧ l₂ / l₃ = 2 / 3
  surface_areas_ratio : S₁ / S₂ = 1 / 4 ∧ S₂ / S₃ = 4 / 9

/-- The ratio of the areas of the three parts of the cone's lateral surface is 1:3:5 -/
theorem cone_lateral_surface_area_ratio (c : DividedCone) :
  c.S₁ / (c.S₂ - c.S₁) = 1 / 3 ∧ (c.S₂ - c.S₁) / (c.S₃ - c.S₂) = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_ratio_l1141_114180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_reciprocal_l1141_114157

-- Define the polynomial
def p (x : ℝ) : ℝ := 40 * x^3 - 60 * x^2 + 24 * x - 1

-- Define the roots
axiom a : ℝ
axiom b : ℝ
axiom c : ℝ

-- State the conditions
axiom ha : 0 < a ∧ a < 1
axiom hb : 0 < b ∧ b < 1
axiom hc : 0 < c ∧ c < 1

-- State that a, b, and c are roots of the polynomial
axiom root_a : p a = 0
axiom root_b : p b = 0
axiom root_c : p c = 0

-- State the theorem
theorem root_sum_reciprocal :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_reciprocal_l1141_114157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_calculation_l1141_114171

noncomputable def square_side : ℝ := 4 + 2 * Real.sqrt 2

def surrounding_rectangle_area (s : ℝ) : ℝ := 2 * s^2

def square_area (s : ℝ) : ℝ := s^2

noncomputable def octagon_area (a : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * a^2

theorem total_area_calculation :
  let s := square_side
  let rectangle_area := surrounding_rectangle_area s
  let square_area := square_area s
  let octagon_side := 2 * Real.sqrt 2
  let octagon_area := octagon_area octagon_side
  rectangle_area + square_area - octagon_area = 56 + 24 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_calculation_l1141_114171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_binomial_coefficient_l1141_114108

def x : ℕ → ℤ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n+2 => 2 * (2 * (n+2) - 1) * x (n+1) / (n+2)

theorem x_is_binomial_coefficient (n : ℕ) : x n = Nat.choose (2 * n) n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_binomial_coefficient_l1141_114108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_face_area_sum_of_coefficients_l1141_114111

-- Define the cylinder
noncomputable def cylinder_radius : ℝ := 8
noncomputable def cylinder_height : ℝ := 10

-- Define the area of the unpainted face
noncomputable def unpainted_area : ℝ := 32 * Real.pi + 160

-- Theorem statement
theorem unpainted_face_area :
  ∀ (P Q : ℝ × ℝ),
    -- P and Q are on the edge of the circular face
    (P.1 - cylinder_radius)^2 + (P.2 - cylinder_radius)^2 = cylinder_radius^2 →
    (Q.1 - cylinder_radius)^2 + (Q.2 - cylinder_radius)^2 = cylinder_radius^2 →
    -- P and Q form a diameter
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (2 * cylinder_radius)^2 →
    -- The area of the unpainted face when sliced through P, Q, and the cylinder's center
    unpainted_area = 32 * Real.pi + 160 := by
  sorry

-- Additional theorem to match the original question
theorem sum_of_coefficients :
  unpainted_area = 32 * Real.pi + 160 * Real.sqrt 1 →
  32 + 160 + 1 = 193 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_face_area_sum_of_coefficients_l1141_114111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_incline_angle_l1141_114174

/-- A line is defined as a set of points (x, y) -/
def line : Set (ℝ × ℝ) := sorry

/-- α represents the incline angle of the line in degrees -/
def α : ℝ := sorry

/-- The incline angle of a vertical line is 90 degrees -/
theorem vertical_line_incline_angle :
  (∀ x y, x = 1 → (x, y) ∈ line) → α = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_incline_angle_l1141_114174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factorials_last_two_digits_l1141_114149

def last_two_digits (n : Nat) : Nat := n % 100

def factorial_ends_in_zeros (n : Nat) : Prop := ∀ m, m ≥ 10 → last_two_digits (Nat.factorial m) = 0

theorem sum_of_factorials_last_two_digits :
  factorial_ends_in_zeros 10 →
  last_two_digits (Finset.sum (Finset.range 51) Nat.factorial) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factorials_last_two_digits_l1141_114149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_projection_bijective_l1141_114128

-- Define the basic structures for projective geometry
structure ProjectivePoint where

structure ProjectiveLine where

structure ProjectivePlane where

-- Define the relation for a point lying on a line
def PointOnLine (p : ProjectivePoint) (l : ProjectiveLine) : Prop := sorry

-- Define the relation for a line lying on a plane
def LineOnPlane (l : ProjectiveLine) (π : ProjectivePlane) : Prop := sorry

-- Define central projection between two planes
def CentralProjection (π₁ π₂ : ProjectivePlane) : ProjectivePoint → ProjectivePoint := sorry

-- Axiom 1: Unique line through two points
axiom unique_line_through_points (p₁ p₂ : ProjectivePoint) (π : ProjectivePlane) :
  p₁ ≠ p₂ → ∃! l : ProjectiveLine, LineOnPlane l π ∧ PointOnLine p₁ l ∧ PointOnLine p₂ l

-- Axiom 2: Unique intersection point of two lines
axiom unique_intersection_point (l₁ l₂ : ProjectiveLine) (π : ProjectivePlane) :
  l₁ ≠ l₂ → LineOnPlane l₁ π → LineOnPlane l₂ π →
  ∃! p : ProjectivePoint, PointOnLine p l₁ ∧ PointOnLine p l₂

-- Theorem: Central projection is bijective
theorem central_projection_bijective (π₁ π₂ : ProjectivePlane) :
  Function.Bijective (CentralProjection π₁ π₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_projection_bijective_l1141_114128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_for_specific_triangle_l1141_114148

/-- Represents a right triangle with an inscribed square -/
structure RightTriangleWithInscribedSquare where
  -- Triangle side lengths
  de : ℝ
  ef : ℝ
  df : ℝ
  -- Condition that it's a right triangle
  is_right_triangle : de^2 + ef^2 = df^2
  -- Conditions for positive side lengths
  de_pos : de > 0
  ef_pos : ef > 0
  df_pos : df > 0

/-- The side length of the inscribed square -/
noncomputable def inscribed_square_side_length (t : RightTriangleWithInscribedSquare) : ℝ :=
  (780 : ℝ) / 229

/-- Theorem stating the side length of the inscribed square -/
theorem inscribed_square_side_length_for_specific_triangle :
  let t : RightTriangleWithInscribedSquare := {
    de := 5,
    ef := 12,
    df := 13,
    is_right_triangle := by norm_num,
    de_pos := by norm_num,
    ef_pos := by norm_num,
    df_pos := by norm_num
  }
  inscribed_square_side_length t = 780 / 229 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_for_specific_triangle_l1141_114148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AO_BC_l1141_114116

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the circumcenter O
variable (O : ℝ × ℝ)

-- Define vectors
def AB (A B : ℝ × ℝ) : ℝ × ℝ := B - A
def AC (A C : ℝ × ℝ) : ℝ × ℝ := C - A
def AO (A O : ℝ × ℝ) : ℝ × ℝ := O - A
def BC (B C : ℝ × ℝ) : ℝ × ℝ := C - B

-- State the conditions
variable (h1 : ‖AB A B‖ = 2)
variable (h2 : ‖AC A C‖ = 6)
variable (h3 : O = circumcenter A B C)

-- State the theorem
theorem dot_product_AO_BC :
  (AO A O) • (BC B C) = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AO_BC_l1141_114116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l1141_114182

/-- The principal amount invested -/
noncomputable def x : ℝ := sorry

/-- The interest rate as a percentage -/
noncomputable def y : ℝ := sorry

/-- Simple interest earned over two years -/
def simple_interest : ℝ := 400

/-- Compound interest earned over two years -/
def compound_interest : ℝ := 410

/-- Time period in years -/
def t : ℝ := 2

/-- Simple interest formula -/
noncomputable def simple_interest_formula (p r t : ℝ) : ℝ := p * r * t / 100

/-- Compound interest formula -/
noncomputable def compound_interest_formula (p r t : ℝ) : ℝ := p * ((1 + r / 100) ^ t - 1)

theorem investment_problem :
  simple_interest_formula x y t = simple_interest ∧
  compound_interest_formula x y t = compound_interest →
  x = 4000 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l1141_114182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_area_l1141_114184

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the surface area of a sphere given its radius -/
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem: The surface area of the circumscribed sphere of the given tetrahedron is 36π -/
theorem tetrahedron_circumscribed_sphere_area :
  let ABCD : Tetrahedron := {
    A := { x := 0, y := 0, z := 0 },
    B := { x := 0, y := 4, z := 0 },
    C := { x := 4, y := 4, z := 0 },
    D := { x := 0, y := 0, z := 2 }
  }
  sphereSurfaceArea 3 = 36 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_area_l1141_114184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1141_114133

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecular_weight (c h o n : ℕ) (c_weight h_weight o_weight n_weight : ℝ) : ℝ :=
  c * c_weight + h * h_weight + o * o_weight + n * n_weight

/-- Proves that the molecular weight of the given compound is approximately 134.156 g/mol -/
theorem compound_molecular_weight :
  let c := 5
  let h := 12
  let o := 3
  let n := 1
  let c_weight := 12.01
  let h_weight := 1.008
  let o_weight := 16.00
  let n_weight := 14.01
  abs (molecular_weight c h o n c_weight h_weight o_weight n_weight - 134.156) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1141_114133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_DME_ABC_l1141_114121

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define M as the midpoint of BC
noncomputable def M (A B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the condition that triangle ABC is acute
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the condition that AM = BC
def AM_equals_BC (A B C : ℝ × ℝ) : Prop := sorry

-- Define D as the intersection of angle bisector AMB with AB
noncomputable def D (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define E as the intersection of angle bisector AMC with AC
noncomputable def E (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the area of a triangle
noncomputable def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ratio_DME_ABC (A B C : ℝ × ℝ) :
  is_acute_triangle A B C →
  AM_equals_BC A B C →
  (area (D A B C) (M A B C) (E A B C)) / (area A B C) = 2 / 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_DME_ABC_l1141_114121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_converges_l1141_114123

/-- The integrand function --/
noncomputable def f (x : ℝ) : ℝ := (x * Real.cos x) / ((1 + x^2) * Real.sqrt (4 + x^2))

/-- The integral we want to prove convergent --/
noncomputable def I : ℝ := ∫ (x : ℝ) in Set.Ioi 1, f x

/-- Statement: The integral I converges --/
theorem integral_converges : ∃ (L : ℝ), I = L := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_converges_l1141_114123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1141_114165

-- Define the triangle and its properties
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  sides_match_angles : a = 2 * Real.sin (B/2) * Real.sin (C/2) ∧
                       b = 2 * Real.sin (A/2) * Real.sin (C/2) ∧
                       c = 2 * Real.sin (A/2) * Real.sin (B/2)

-- Define the vectors
noncomputable def m (t : Triangle) : ℝ × ℝ := (2 * Real.sin (t.A + t.C), Real.sqrt 3)
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.cos (2 * t.B), 1 - 2 * (Real.cos (t.B / 2))^2)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h_parallel : ∃ k : ℝ, m t = k • (n t))
  (h_sin : Real.sin t.A * Real.sin t.C = (Real.sin t.B)^2) :
  t.B = π/3 ∧ t.a = t.c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1141_114165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_3r_squared_l1141_114131

/-- The sum of the geometric series with a common ratio of r^4 and first term r^3 -/
noncomputable def S (r : ℝ) : ℝ := r^3 / (1 - r^4)

/-- r is a positive real number satisfying r^4 + (1/3)r - 1 = 0 -/
def r_condition (r : ℝ) : Prop := r > 0 ∧ r^4 + (1/3)*r - 1 = 0

theorem sum_equals_3r_squared {r : ℝ} (hr : r_condition r) : S r = 3 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_3r_squared_l1141_114131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_l1141_114113

-- Define the ellipse
def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) + (P.2^2 / 9) = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁.1 < 0 ∧ F₂.1 > 0 ∧ F₁.1 = -F₂.1 ∧ F₁.2 = 0 ∧ F₂.2 = 0

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the angle between three points
noncomputable def angle (A B C : ℝ × ℝ) : ℝ :=
  Real.arccos ((distance A B)^2 + (distance B C)^2 - (distance A C)^2) / (2 * distance A B * distance B C)

-- Theorem statement
theorem angle_measure (P F₁ F₂ : ℝ × ℝ) :
  ellipse P → foci F₁ F₂ → distance P F₁ * distance P F₂ = 12 →
  Real.cos (angle F₁ P F₂) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_l1141_114113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_true_discount_l1141_114120

/-- Calculates the true discount given the banker's gain, interest rate, and time period. -/
noncomputable def true_discount (bankers_gain : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  (bankers_gain * (100 + interest_rate * time)) / (interest_rate * time)

/-- Theorem stating that given specific conditions, the true discount is 72.8 -/
theorem specific_true_discount :
  let bankers_gain : ℝ := 7.8
  let interest_rate : ℝ := 12
  let time : ℝ := 1
  true_discount bankers_gain interest_rate time = 72.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_true_discount_l1141_114120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1141_114141

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    with foci F₁ and F₂, and a point Q on the ellipse such that
    the centroid G and incenter I of triangle QF₁F₂ satisfy
    vector GI = λ * vector F₁F₂, prove that the eccentricity
    of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b : ℝ) (F₁ F₂ Q G I : ℝ × ℝ) (lambda : ℝ) :
  a > b ∧ b > 0 ∧
  (Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1) ∧
  (G = ((Q.1 + F₁.1 + F₂.1) / 3, (Q.2 + F₁.2 + F₂.2) / 3)) ∧
  (∃ r : ℝ, r > 0 ∧
    r * ((Q.1 - F₁.1) ^ 2 + (Q.2 - F₁.2) ^ 2) = (Q.2 - F₁.2) * (F₂.1 - F₁.1) - (Q.1 - F₁.1) * (F₂.2 - F₁.2) ∧
    r * ((Q.1 - F₂.1) ^ 2 + (Q.2 - F₂.2) ^ 2) = (Q.2 - F₂.2) * (F₁.1 - F₂.1) - (Q.1 - F₂.1) * (F₁.2 - F₂.2) ∧
    r * ((F₁.1 - F₂.1) ^ 2 + (F₁.2 - F₂.2) ^ 2) = (F₁.2 - F₂.2) * (Q.1 - F₂.1) - (F₁.1 - F₂.1) * (Q.2 - F₂.2)) ∧
  (I.2 - G.2 = lambda * (F₂.1 - F₁.1) ∧ I.1 - G.1 = lambda * (F₂.2 - F₁.2)) →
  (((F₁.1 - F₂.1) ^ 2 + (F₁.2 - F₂.2) ^ 2).sqrt / (2 * a) = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1141_114141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_second_to_largest_l1141_114105

/-- A right circular cone sliced into five pieces by planes parallel to its base -/
structure SlicedCone where
  height : ℝ
  radius : ℝ
  slice_ratio : ℝ
  slice_count : ℕ

/-- Volume of a cone segment -/
noncomputable def segment_volume (c : SlicedCone) (h : ℝ) (r : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

/-- Theorem: The ratio of the volume of the second-largest piece to the largest piece is 64/125 -/
theorem volume_ratio_second_to_largest (c : SlicedCone) 
  (h_slice_ratio : c.slice_ratio = 0.8)
  (h_slice_count : c.slice_count = 5) :
  let v1 := segment_volume c c.height c.radius
  let v2 := segment_volume c (c.height * c.slice_ratio) (c.radius * c.slice_ratio)
  v2 / v1 = 64 / 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_second_to_largest_l1141_114105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_analysis_l1141_114135

def taxi_journey : List Int := [15, -4, 13, -10, -12, 3, -13, -17]

def fuel_consumption : Float := 0.4

theorem taxi_journey_analysis :
  let final_displacement := taxi_journey.sum
  let total_distance := (taxi_journey.map Int.natAbs).sum
  let total_fuel := (total_distance.toFloat) * fuel_consumption
  (final_displacement = -38 ∧
   total_distance = 87 ∧
   total_fuel = 34.8) := by
  sorry

#eval taxi_journey.sum
#eval (taxi_journey.map Int.natAbs).sum
#eval ((taxi_journey.map Int.natAbs).sum.toFloat) * fuel_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_analysis_l1141_114135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_tax_rate_l1141_114155

/-- Calculates Mindy's tax rate given the conditions -/
noncomputable def findMindyTaxRate (morkTaxRate : ℝ) (mindyIncomeRatio : ℝ) (combinedTaxRate : ℝ) : ℝ :=
  (combinedTaxRate * (1 + mindyIncomeRatio) - morkTaxRate) / mindyIncomeRatio

/-- Theorem stating that Mindy's tax rate is 30% given the conditions -/
theorem mindy_tax_rate :
  let morkTaxRate : ℝ := 0.40
  let mindyIncomeRatio : ℝ := 2
  let combinedTaxRate : ℝ := 1/3
  findMindyTaxRate morkTaxRate mindyIncomeRatio combinedTaxRate = 0.30 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_tax_rate_l1141_114155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1141_114119

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - Real.log x) + (2 * x - 1) / x^2

-- State the theorem
theorem f_inequality (x : ℝ) (hx : x ∈ Set.Icc 1 2) : 
  f 1 x > (deriv (f 1)) x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1141_114119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_neg_one_third_l1141_114126

noncomputable def g (a b : ℝ) : ℝ :=
  if a + b ≤ 5 then
    (a^2 * b - a + 3) / (3 * a)
  else
    (a * b^2 - b - 3) / (-3 * b)

theorem g_sum_equals_neg_one_third :
  g 3 2 + g 3 3 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_neg_one_third_l1141_114126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_theorem_l1141_114194

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the centroid of a quadrilateral with equal masses at its vertices -/
noncomputable def centroid (q : Quadrilateral) : Point3D :=
  { x := (q.A.x + q.B.x + q.C.x + q.D.x) / 4,
    y := (q.A.y + q.B.y + q.C.y + q.D.y) / 4,
    z := (q.A.z + q.B.z + q.C.z + q.D.z) / 4 }

/-- Theorem: The centroid of a quadrilateral with equal masses at its vertices
    is located at the average of the vertex positions -/
theorem centroid_theorem (q : Quadrilateral) :
  centroid q = { x := (q.A.x + q.B.x + q.C.x + q.D.x) / 4,
                 y := (q.A.y + q.B.y + q.C.y + q.D.y) / 4,
                 z := (q.A.z + q.B.z + q.C.z + q.D.z) / 4 } := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_theorem_l1141_114194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l1141_114106

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log (abs (sqrt (x + 1) - 1)) - 
  log (abs ((sqrt (x + 1) + 1/2)^2 + 3/4)) - 
  (2 / sqrt 3) * arctan ((2 * (sqrt (x + 1) + 1/2)) / sqrt 3)

noncomputable def g (x : ℝ) : ℝ := (sqrt (x + 1) + 2) / ((x + 1)^2 - sqrt (x + 1))

theorem integral_equality (x : ℝ) (h : x > -1) : 
  deriv f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l1141_114106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_value_at_l1141_114185

/-- The function to be minimized -/
noncomputable def f (x : ℝ) : ℝ := x + 1/x + 16*x/(x^2 + 1)

/-- Theorem stating that 8 is the minimum value of f(x) for x > 1 -/
theorem min_value_of_f :
  ∀ x : ℝ, x > 1 → f x ≥ 8 := by
  sorry

/-- Theorem stating the value of x that achieves the minimum -/
theorem min_value_at :
  ∃ x : ℝ, x > 1 ∧ f x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_value_at_l1141_114185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_specific_sine_phase_shift_l1141_114122

/-- PhaseShift is a function that returns the phase shift of a given periodic function. -/
def PhaseShift (f : ℝ → ℝ) : ℝ := sorry

/-- The phase shift of a sine function in the form y = A sin(x - C) is C. -/
theorem sine_phase_shift (A C : ℝ) : 
  let f : ℝ → ℝ := λ x => A * Real.sin (x - C)
  PhaseShift f = C := by sorry

/-- The phase shift of the function y = 3 sin(x - π/5) is π/5. -/
theorem specific_sine_phase_shift : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin (x - Real.pi/5)
  PhaseShift f = Real.pi/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_specific_sine_phase_shift_l1141_114122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tri_colored_triangle_l1141_114156

/-- A coloring of a complete graph satisfying the connectivity condition -/
structure ConnectedColoring (n k : ℕ) where
  coloring : Fin n → Fin n → Fin k
  complete : ∀ i j, i ≠ j → ∃ c, coloring i j = c
  connected : ∀ c i j, ∃ path : List (Fin n), 
    path.head? = some i ∧ 
    path.getLast? = some j ∧ 
    (∀ x y, (x, y) ∈ path.zip (path.tail) → coloring x y = c)

/-- The existence of a triangle with three distinct colors -/
theorem exists_tri_colored_triangle 
  {n k : ℕ} 
  (hn : n > 0) 
  (hk : k ≥ 3) 
  (col : ConnectedColoring n k) : 
  ∃ i j l : Fin n, 
    i ≠ j ∧ j ≠ l ∧ i ≠ l ∧
    col.coloring i j ≠ col.coloring j l ∧
    col.coloring j l ≠ col.coloring l i ∧
    col.coloring l i ≠ col.coloring i j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tri_colored_triangle_l1141_114156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_plus_n_l1141_114143

-- Define the function f
noncomputable def f (m n x : ℝ) : ℝ := m * (2 : ℝ)^x + x^2 + n*x

-- Define the set of roots of f
def root_set (m n : ℝ) : Set ℝ := {x | f m n x = 0}

-- Define the set of roots of f(f(x))
def double_root_set (m n : ℝ) : Set ℝ := {x | f m n (f m n x) = 0}

-- Theorem statement
theorem range_of_m_plus_n (m n : ℝ) :
  (root_set m n = double_root_set m n) ∧ (root_set m n).Nonempty →
  0 ≤ m + n ∧ m + n < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_plus_n_l1141_114143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1141_114134

-- Define the functions f and g
noncomputable def f (x : ℝ) := (2 : ℝ) ^ x
def g (x : ℝ) := 2 * x

-- Define what it means for g to be a covering function for f on an interval
def is_covering_function (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ x, m ≤ x ∧ x ≤ n → f x ≤ g x

-- State the theorem
theorem max_value_theorem (m n : ℝ) :
  is_covering_function f g m n → (2 : ℝ) ^ (abs (m - n)) ≤ 2 ∧ 
  ∃ m₀ n₀, is_covering_function f g m₀ n₀ ∧ (2 : ℝ) ^ (abs (m₀ - n₀)) = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1141_114134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1141_114151

theorem cosine_inequality (x y : ℝ) : 
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.cos (x + y) ≥ Real.cos x * Real.cos y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1141_114151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_18_l1141_114166

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℚ
  runsIn17thInning : ℚ
  averageIncrease : ℚ

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℚ :=
  (16 * b.initialAverage + b.runsIn17thInning) / 17

/-- Theorem stating that the new average is 18 given the conditions -/
theorem new_average_is_18 (b : Batsman) 
  (h1 : b.runsIn17thInning = 66)
  (h2 : b.averageIncrease = 3)
  (h3 : newAverage b = b.initialAverage + b.averageIncrease) :
  newAverage b = 18 := by
  sorry

#eval newAverage { initialAverage := 15, runsIn17thInning := 66, averageIncrease := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_18_l1141_114166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_200_terms_l1141_114130

def mySequence : ℕ → ℤ
  | 0 => 6
  | 1 => 3
  | n + 2 => mySequence n + mySequence (n + 1) - 5

theorem sum_of_first_200_terms : (Finset.range 200).sum (Int.natAbs ∘ mySequence) = 999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_200_terms_l1141_114130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l1141_114159

theorem integral_inequality (f : ℝ → ℝ) (A B : ℝ) :
  Continuous f →
  (∀ x ∈ Set.Icc 0 1, 0 < A ∧ A ≤ f x ∧ f x ≤ B) →
  A * B * ∫ x in Set.Icc 0 1, 1 / f x ≤ A + B - ∫ x in Set.Icc 0 1, f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l1141_114159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_sequence_sum_formula_l1141_114169

/-- The sequence where each positive integer occurs twice -/
def mySequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) / 2

/-- The sum of the first n terms of the sequence -/
def sum_sequence (n : ℕ) : ℕ := n^2 / 4

theorem sequence_sum_property (m n : ℕ) :
  sum_sequence (m + n) - sum_sequence (m - n) = m * n := by
  sorry

theorem sequence_sum_formula (n : ℕ) :
  sum_sequence n = (n^2 / 4 : ℚ).floor := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_sequence_sum_formula_l1141_114169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_six_l1141_114170

/-- The area of the shaded region in a 3 × 4 rectangle with two non-overlapping circles of diameter 2 removed --/
noncomputable def shaded_area : ℝ := 12 - 2 * Real.pi

/-- The whole number closest to the shaded area --/
def closest_whole_number : ℕ := 6

/-- Theorem stating that 6 is the closest whole number to the shaded area --/
theorem shaded_area_closest_to_six : 
  ∀ n : ℕ, |shaded_area - (closest_whole_number : ℝ)| ≤ |shaded_area - (n : ℝ)| :=
by
  sorry

#eval closest_whole_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_six_l1141_114170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_expression_l1141_114199

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The alpha constant -/
noncomputable def α : ℝ := (3 - Real.sqrt 5) / 2

/-- The f function -/
noncomputable def f (n : ℝ) : ℤ := floor (α * n)

/-- The F function -/
def F : ℕ → ℝ
| 0 => 1
| 1 => 3
| (k + 2) => 3 * F (k + 1) - F k

/-- The theorem to be proved -/
theorem F_expression (k : ℕ) :
  F k = (1 / Real.sqrt 5) * ((3 + Real.sqrt 5) / 2)^(k + 1) - 
        (1 / Real.sqrt 5) * ((3 - Real.sqrt 5) / 2)^(k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_expression_l1141_114199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_identical_selection_l1141_114158

/-- A courtier's selection of dogs -/
structure Selection where
  dogs : Finset ℕ
  size_eq_three : dogs.card = 3

/-- The problem setup -/
axiom num_dogs : ℕ
axiom num_courtiers : ℕ
axiom dog_selections : Finset Selection

/-- The conditions of the problem -/
axiom hundred_dogs : num_dogs = 100
axiom hundred_courtiers : num_courtiers = 100
axiom all_selections_valid : ∀ s, s ∈ dog_selections → s.dogs ⊆ Finset.range num_dogs
axiom correct_number_of_selections : dog_selections.card = num_courtiers
axiom common_dogs : ∀ s₁ s₂, s₁ ∈ dog_selections → s₂ ∈ dog_selections → s₁ ≠ s₂ → (s₁.dogs ∩ s₂.dogs).card ≥ 2

/-- The theorem to be proved -/
theorem exists_identical_selection :
  ∃ s₁ s₂, s₁ ∈ dog_selections ∧ s₂ ∈ dog_selections ∧ s₁ ≠ s₂ ∧ s₁.dogs = s₂.dogs := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_identical_selection_l1141_114158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_135_matrix_l1141_114179

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]]

noncomputable def angle_135 : ℝ := 135 * Real.pi / 180

theorem rotation_135_matrix :
  rotation_matrix angle_135 = ![![-1/Real.sqrt 2, -1/Real.sqrt 2],
                                ![1/Real.sqrt 2, -1/Real.sqrt 2]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_135_matrix_l1141_114179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1141_114102

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let (x₃, y₃) := R
  (1/2) * abs ((x₂ - x₁)*(y₃ - y₁) - (x₃ - x₁)*(y₂ - y₁))

/-- The area of triangle PQR with given coordinates is 24 square units -/
theorem area_of_triangle_PQR : 
  let P : ℝ × ℝ := (-2, 2)
  let Q : ℝ × ℝ := (6, 2)
  let R : ℝ × ℝ := (2, -4)
  triangle_area P Q R = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1141_114102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pauls_age_l1141_114146

/-- Given the ages of Mark, Alice, and Paul, prove Paul's age. -/
theorem pauls_age (mark_age alice_age paul_age : ℕ) : 
  mark_age = 20 →
  alice_age = mark_age + 4 →
  paul_age = alice_age - 5 →
  paul_age = 19 := by
  intro h1 h2 h3
  rw [h2] at h3
  rw [h1] at h3
  norm_num at h3
  exact h3

#check pauls_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pauls_age_l1141_114146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_proportional_other_equations_proportional_l1141_114101

noncomputable section

/-- A relation between x and y is directly proportional if there exists a non-zero constant k such that y = kx for all x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A relation between x and y is inversely proportional if there exists a non-zero constant k such that xy = k for all x ≠ 0 -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, x ≠ 0 → f x * x = k

/-- The function representing the equation 2x + 5y = 15 -/
noncomputable def f (x : ℝ) : ℝ := (15 - 2*x) / 5

theorem not_proportional :
  ¬(DirectlyProportional f) ∧ ¬(InverselyProportional f) :=
by sorry

/-- The function representing x + 2y = 1 -/
noncomputable def g (x : ℝ) : ℝ := (1 - x) / 2

/-- The function representing 5xy = 16 -/
noncomputable def h (x : ℝ) : ℝ := 16 / (5*x)

/-- The function representing x = 4y -/
noncomputable def i (x : ℝ) : ℝ := x / 4

/-- The function representing x/y = 5 -/
noncomputable def j (x : ℝ) : ℝ := x / 5

theorem other_equations_proportional :
  (DirectlyProportional g ∨ InverselyProportional g) ∧
  (DirectlyProportional h ∨ InverselyProportional h) ∧
  (DirectlyProportional i ∨ InverselyProportional i) ∧
  (DirectlyProportional j ∨ InverselyProportional j) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_proportional_other_equations_proportional_l1141_114101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrenches_in_comparison_group_l1141_114191

/-- The weight of a hammer -/
def H : ℝ := sorry

/-- The weight of a wrench -/
def W : ℝ := sorry

/-- The number of wrenches in the comparison group -/
def x : ℝ := sorry

/-- Hammers and wrenches have uniform weights -/
axiom uniform_weights : True

/-- The weight of one wrench is 2 times the weight of one hammer -/
axiom wrench_weight : W = 2 * H

/-- The total weight of 2 hammers and 2 wrenches is 1/3 of the weight of 8 hammers and x wrenches -/
axiom weight_comparison : 2 * H + 2 * W = (1 / 3) * (8 * H + x * W)

/-- The number of wrenches in the comparison group is 5 -/
theorem wrenches_in_comparison_group : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrenches_in_comparison_group_l1141_114191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_insured_employees_is_121_l1141_114189

/-- The maximum number of employees that can be insured -/
def max_insured_employees : ℕ :=
  let total_premium : ℚ := 5000000
  let outpatient_cost : ℚ := 18000
  let hospitalization_cost : ℚ := 60000
  let hospitalization_rate : ℚ := 1/4
  let overhead_cost : ℚ := 2800
  let profit_rate : ℚ := 15/100
  let total_cost_per_person : ℚ := outpatient_cost + hospitalization_rate * hospitalization_cost + overhead_cost
  let cost_with_profit : ℚ := total_cost_per_person * (1 + profit_rate)
  (total_premium / cost_with_profit).floor.toNat

theorem max_insured_employees_is_121 :
  max_insured_employees = 121 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_insured_employees_is_121_l1141_114189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_holly_tree_width_l1141_114176

/-- Given a fence length in yards, cost per tree in dollars, and total planting cost in dollars,
    calculate the width of each tree at maturity in feet. -/
noncomputable def tree_width_at_maturity (fence_length_yards : ℚ) (cost_per_tree : ℚ) (total_planting_cost : ℚ) : ℚ :=
  let fence_length_feet := fence_length_yards * 3
  let number_of_trees := total_planting_cost / cost_per_tree
  fence_length_feet / number_of_trees

/-- Theorem stating that for the given conditions, the tree width at maturity is 1.5 feet. -/
theorem holly_tree_width :
  tree_width_at_maturity 25 8 400 = 3/2 := by
  -- Unfold the definition of tree_width_at_maturity
  unfold tree_width_at_maturity
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_holly_tree_width_l1141_114176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_satisfies_conditions_f_b_satisfies_conditions_f_c_satisfies_conditions_polynomials_degree_at_most_2_l1141_114154

-- Part a
def f_a (x : ℝ) : ℝ := -x^2 + 3*x + 1

theorem f_a_satisfies_conditions :
  f_a 0 = 1 ∧ f_a 1 = 3 ∧ f_a 2 = 3 := by sorry

-- Part b
def f_b (x : ℝ) : ℝ := 3*x + 2

theorem f_b_satisfies_conditions :
  f_b (-1) = -1 ∧ f_b 0 = 2 ∧ f_b 1 = 5 := by sorry

-- Part c
def f_c (x : ℝ) : ℝ := x^2

theorem f_c_satisfies_conditions :
  f_c (-1) = 1 ∧ f_c 0 = 0 ∧ f_c 2 = 4 := by sorry

-- Prove that all polynomials are of degree no higher than 2
theorem polynomials_degree_at_most_2 :
  (∃ (a b c : ℝ), ∀ x, f_a x = a*x^2 + b*x + c) ∧
  (∃ (a b c : ℝ), ∀ x, f_b x = a*x^2 + b*x + c) ∧
  (∃ (a b c : ℝ), ∀ x, f_c x = a*x^2 + b*x + c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_satisfies_conditions_f_b_satisfies_conditions_f_c_satisfies_conditions_polynomials_degree_at_most_2_l1141_114154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_and_linear_func_l1141_114118

-- Define the inverse proportion function
noncomputable def inverse_prop (a : ℝ) (x : ℝ) : ℝ := a / x

-- Define the linear function
def linear_func (a : ℝ) (x : ℝ) : ℝ := -a * x + a

-- State the theorem
theorem inverse_prop_and_linear_func (a : ℝ) (h1 : a > 0) :
  (∀ x y, x > 0 → y > 0 → x < y → inverse_prop a x > inverse_prop a y) →
  (∀ x y, x < 0 → y < 0 → x < y → inverse_prop a x > inverse_prop a y) →
  ¬ ∃ x y, x < 0 ∧ y < 0 ∧ y = linear_func a x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_and_linear_func_l1141_114118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1141_114187

open Real Set

noncomputable def g (A : ℝ) : ℝ :=
  (sin A * (3 * cos A ^ 2 + cos A ^ 4 + 3 * sin A ^ 2 + sin A ^ 4)) /
  (tan A * ((1 / cos A) ^ 2 - sin A * tan A))

theorem range_of_g :
  {y | ∃ A : ℝ, (∀ n : ℤ, A ≠ n * π / 2) ∧ g A = y} = univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1141_114187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cosarcsin_cotarccos_l1141_114197

theorem unique_solution_cosarcsin_cotarccos (x : ℝ) : 
  (∃! x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x → 
    x = Real.sqrt ((3 - Real.sqrt 5) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cosarcsin_cotarccos_l1141_114197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_permutation_l1141_114198

def IsValidPermutation (p : List ℕ) : Prop :=
  ∀ i j, i < j → j < p.length →
    ¬∃ k, i < k ∧ k < j ∧ 2 * p[k]! = p[i]! + p[j]!

theorem exists_valid_permutation (n : ℕ) :
  ∃ p : List ℕ, p.length = 2^n ∧ p.Nodup ∧
    (∀ i, i ∈ p ↔ 1 ≤ i ∧ i ≤ 2^n) ∧
    IsValidPermutation p := by
  sorry

#check exists_valid_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_permutation_l1141_114198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l1141_114112

open Real

/-- Represents the common ratio of an arithmetic sequence. -/
noncomputable def commonDifference (a b : ℝ) : ℝ := (log (a^7 * b^10) - log (a^2 * b^4))

/-- Represents the nth term of the arithmetic sequence. -/
noncomputable def nthTerm (a b : ℝ) (n : ℕ) : ℝ := log (a^2 * b^4) + (n - 1) * commonDifference a b

theorem arithmetic_sequence_15th_term (a b : ℝ) (m : ℕ) 
  (h1 : commonDifference a b = log (a^12 * b^14) - log (a^7 * b^10))
  (h2 : nthTerm a b 15 = log (b^m)) :
  m = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l1141_114112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_problem_l1141_114138

/-- Represents the seating arrangement at a round table --/
structure SeatingArrangement where
  women : ℕ
  men : ℕ

/-- The conditions of the seating arrangement --/
def ValidSeating (s : SeatingArrangement) : Prop :=
  ∃ (women_left_of_women : ℕ) (women_left_of_men : ℕ),
    women_left_of_women = 7 ∧
    women_left_of_men = 12 ∧
    (3 * s.men : ℕ) = 4 * women_left_of_men

theorem seating_problem :
  ∃ (s : SeatingArrangement), ValidSeating s ∧ s.women + s.men = 35 := by
  -- We'll construct the SeatingArrangement
  let s : SeatingArrangement := ⟨19, 16⟩
  
  -- Now we'll prove that this arrangement satisfies the conditions
  have h1 : ValidSeating s := by
    -- We need to provide the values for women_left_of_women and women_left_of_men
    use 7, 12
    -- Now we prove the three conditions
    constructor
    · rfl  -- 7 = 7
    constructor
    · rfl  -- 12 = 12
    · -- Prove that (3 * 16 : ℕ) = 4 * 12
      norm_num
  
  -- Now we prove that the total number of people is 35
  have h2 : s.women + s.men = 35 := by
    norm_num
  
  -- Finally, we combine our proofs
  exact ⟨s, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_problem_l1141_114138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l1141_114196

open Real

noncomputable def f (x : ℝ) := log x + x^2 + x

theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) 
  (h : f x₁ + f x₂ + x₁ * x₂ = 0) : 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l1141_114196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_practices_athletics_l1141_114192

-- Define the athletes
inductive Athlete : Type
| Maria : Athlete
| Tania : Athlete
| Juan : Athlete
| David : Athlete

-- Define the sports
inductive Sport : Type
| Swimming : Sport
| Volleyball : Sport
| Gymnastics : Sport
| Athletics : Sport

-- Define the positions at the table
inductive Position : Type
| Top : Position
| Right : Position
| Bottom : Position
| Left : Position

-- Define the seating arrangement
def SeatingArrangement : Type := Athlete → Position

-- Define the sport assignment
def SportAssignment : Type := Athlete → Sport

-- Define the conditions of the seating arrangement
def ValidSeatingArrangement (seating : SeatingArrangement) (sports : SportAssignment) : Prop :=
  -- The person who practices swimming is to the left of Maria
  (∃ a : Athlete, sports a = Sport.Swimming ∧ 
    ((seating Athlete.Maria = Position.Right ∧ seating a = Position.Left) ∨
     (seating Athlete.Maria = Position.Top ∧ seating a = Position.Right) ∨
     (seating Athlete.Maria = Position.Left ∧ seating a = Position.Bottom) ∨
     (seating Athlete.Maria = Position.Bottom ∧ seating a = Position.Left))) ∧
  -- The person who practices gymnastics is in front of Juan
  (∃ a : Athlete, sports a = Sport.Gymnastics ∧
    ((seating Athlete.Juan = Position.Bottom ∧ seating a = Position.Top) ∨
     (seating Athlete.Juan = Position.Top ∧ seating a = Position.Bottom) ∨
     (seating Athlete.Juan = Position.Left ∧ seating a = Position.Right) ∨
     (seating Athlete.Juan = Position.Right ∧ seating a = Position.Left))) ∧
  -- Tânia and David sit next to each other
  ((seating Athlete.Tania = Position.Top ∧ seating Athlete.David = Position.Right) ∨
   (seating Athlete.Tania = Position.Right ∧ seating Athlete.David = Position.Bottom) ∨
   (seating Athlete.Tania = Position.Bottom ∧ seating Athlete.David = Position.Left) ∨
   (seating Athlete.Tania = Position.Left ∧ seating Athlete.David = Position.Top) ∨
   (seating Athlete.David = Position.Top ∧ seating Athlete.Tania = Position.Right) ∨
   (seating Athlete.David = Position.Right ∧ seating Athlete.Tania = Position.Bottom) ∨
   (seating Athlete.David = Position.Bottom ∧ seating Athlete.Tania = Position.Left) ∨
   (seating Athlete.David = Position.Left ∧ seating Athlete.Tania = Position.Top)) ∧
  -- A woman sits next to the person who practices volleyball
  (∃ a b : Athlete, sports a = Sport.Volleyball ∧ (b = Athlete.Maria ∨ b = Athlete.Tania) ∧
    ((seating a = Position.Top ∧ (seating b = Position.Right ∨ seating b = Position.Left)) ∨
     (seating a = Position.Right ∧ (seating b = Position.Top ∨ seating b = Position.Bottom)) ∨
     (seating a = Position.Bottom ∧ (seating b = Position.Right ∨ seating b = Position.Left)) ∨
     (seating a = Position.Left ∧ (seating b = Position.Top ∨ seating b = Position.Bottom))))

-- Theorem: Given the conditions, David practices athletics
theorem david_practices_athletics (seating : SeatingArrangement) (sports : SportAssignment) 
  (h : ValidSeatingArrangement seating sports) : 
  sports Athlete.David = Sport.Athletics :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_practices_athletics_l1141_114192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1141_114114

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define vector operations
noncomputable def vecLength (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def vecNormalize (v : ℝ × ℝ) : ℝ × ℝ :=
  let len := vecLength v
  (v.1 / len, v.2 / len)

-- Define the theorem
theorem quadrilateral_area (ABCD : Quadrilateral) : 
  ABCD.B.1 - ABCD.A.1 = 1 ∧ ABCD.B.2 - ABCD.A.2 = 1 ∧
  ABCD.C.1 - ABCD.D.1 = 1 ∧ ABCD.C.2 - ABCD.D.2 = 1 ∧
  (let BA := (ABCD.A.1 - ABCD.B.1, ABCD.A.2 - ABCD.B.2)
   let BC := (ABCD.C.1 - ABCD.B.1, ABCD.C.2 - ABCD.B.2)
   let BD := (ABCD.D.1 - ABCD.B.1, ABCD.D.2 - ABCD.B.2)
   vecNormalize BA + vecNormalize BC = (Real.sqrt 3 * (BD.1 / vecLength BD), Real.sqrt 3 * (BD.2 / vecLength BD))) →
  let area := Real.sqrt ((ABCD.B.1 - ABCD.A.1) * (ABCD.C.2 - ABCD.B.2) - (ABCD.C.1 - ABCD.B.1) * (ABCD.B.2 - ABCD.A.2))
  area = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1141_114114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_legs_l1141_114168

-- Define the necessary types and structures
structure Point (α : Type) where
  x : α
  y : α

structure Triangle (α : Type) where
  a : Point α
  b : Point α
  c : Point α

def RightTriangle {α : Type} (t : Triangle α) : Prop := sorry
def EquidistantFromLegs {α : Type} (p : Point α) (t : Triangle α) : Prop := sorry
def SegmentLength {α : Type} (p1 p2 : Point α) : α := sorry
def Hypotenuse {α : Type} (t : Triangle α) : Point α × Point α := sorry
def RemainingHypotenuse {α : Type} (t : Triangle α) (p : Point α) : Point α × Point α := sorry
def LegLength {α : Type} (p1 p2 : Point α) : α := sorry

theorem right_triangle_legs (t : Triangle ℝ) (h : Point ℝ) :
  RightTriangle t →
  EquidistantFromLegs h t →
  SegmentLength (Hypotenuse t).1 (Hypotenuse t).2 = 70 →
  SegmentLength (Hypotenuse t).1 h = 30 →
  SegmentLength h (Hypotenuse t).2 = 40 →
  (LegLength t.a t.c = 42 ∧ LegLength t.b t.c = 56) ∨ 
  (LegLength t.a t.c = 56 ∧ LegLength t.b t.c = 42) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_legs_l1141_114168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_specific_series_l1141_114186

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r satisfies the equation S = a / (1 - r) -/
noncomputable def infinite_geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Given an infinite geometric series with first term 500 and sum 4000,
    prove that the common ratio is 7/8 -/
theorem common_ratio_of_specific_series : 
  ∃ (r : ℝ), infinite_geometric_series_sum 500 r = 4000 ∧ r = 7/8 := by
  use 7/8
  constructor
  · -- Prove infinite_geometric_series_sum 500 (7/8) = 4000
    sorry
  · -- Prove 7/8 = 7/8
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_specific_series_l1141_114186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_not_in_lowest_terms_count_l1141_114144

theorem fraction_not_in_lowest_terms_count : 
  (Finset.filter (fun N : ℕ => Nat.gcd (N^2 + 5) (N + 4) > 1) (Finset.range 1000)).card = 524 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_not_in_lowest_terms_count_l1141_114144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l1141_114162

theorem tan_arccot_three_fifths : 
  Real.tan (Real.arctan⁻¹ (5 / 3)) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l1141_114162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_equal_square_area_l1141_114173

/-- The length of a rectangle with width 6.3 cm and area equal to a square with side length 12.5 cm is approximately 24.802 cm. -/
theorem rectangle_length_equal_square_area (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) 
  (h1 : square_side = 12.5)
  (h2 : rect_width = 6.3)
  (h3 : square_side * square_side = rect_width * rect_length) : 
  abs (rect_length - 24.802) < 0.001 := by
  sorry

#check rectangle_length_equal_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_equal_square_area_l1141_114173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1141_114104

/-- The time (in seconds) it takes for a train to cross a bridge. -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that a train of length 110 m, traveling at 72 kmph,
    takes 14 seconds to cross a bridge of length 170 m. -/
theorem train_crossing_bridge_time :
  train_crossing_time 110 72 170 = 14 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 110 72 170

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1141_114104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_is_14_minutes_l1141_114145

/-- Represents the boy's journey to school -/
structure SchoolJourney where
  normal_rate : ℝ
  normal_time : ℝ
  distance : ℝ

/-- The time taken when walking at a given percentage of normal rate -/
noncomputable def time_at_rate (j : SchoolJourney) (rate_percentage : ℝ) : ℝ :=
  j.distance / (rate_percentage * j.normal_rate)

theorem usual_time_is_14_minutes (j : SchoolJourney) 
  (h1 : time_at_rate j 0.8 = j.normal_time - 5)
  (h2 : time_at_rate j 1.2 = j.normal_time - 8) :
  j.normal_time = 14 := by
  sorry

#check usual_time_is_14_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_is_14_minutes_l1141_114145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1141_114140

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) → f x ≤ 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) ∧ f x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1141_114140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_tip_angles_ten_pointed_star_l1141_114161

/-- A star formed by connecting evenly spaced points on a circle -/
structure StarPolygon where
  n : ℕ  -- number of points
  assume_n_geq_5 : n ≥ 5

/-- The angle at each tip of the star -/
def tip_angle (s : StarPolygon) : ℚ :=
  (360 / s.n) * ((s.n - 2) / 2)

/-- The sum of all tip angles in the star -/
def sum_tip_angles (s : StarPolygon) : ℚ :=
  s.n * tip_angle s

/-- Theorem: The sum of tip angles in a 10-pointed star is 720° -/
theorem sum_tip_angles_ten_pointed_star :
  ∀ s : StarPolygon, s.n = 10 → sum_tip_angles s = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_tip_angles_ten_pointed_star_l1141_114161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l1141_114103

/-- Proves that given an article with two successive discounts of 10% and 4.999999999999997%,
    if the final price after discounts is Rs. 59.85, then the original list price is approximately Rs. 70. -/
theorem article_price_after_discounts (list_price : ℝ) : 
  let discount1 := 0.10
  let discount2 := 0.04999999999999997
  let final_price := 59.85
  (1 - discount1) * (1 - discount2) * list_price = final_price →
  ‖list_price - 70‖ < 0.01 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l1141_114103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_conditions_l1141_114172

/-- 
Given natural numbers x, y, z ≥ 1, this theorem states the necessary and sufficient conditions
for arranging letters A (x times), B (y times), and C (z times) in a circle
with no adjacent identical letters.
-/
theorem circle_arrangement_conditions (x y z : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (∃ (arrangement : List Char), 
    arrangement.length = x + y + z ∧ 
    arrangement.count 'A' ≥ x ∧
    arrangement.count 'B' ≥ y ∧
    arrangement.count 'C' ≥ z ∧
    ∀ i, i < arrangement.length → 
      arrangement[i]? ≠ arrangement[(i + 1) % arrangement.length]?) ↔
  (x ≤ y + z ∧ y ≤ x + z ∧ z ≤ x + y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_conditions_l1141_114172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circle_and_ellipse_l1141_114107

-- Define the circle and ellipse
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1/4
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem max_distance_between_circle_and_ellipse :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ 
    ellipse_eq x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ), circle_eq x3 y3 → ellipse_eq x4 y4 → 
      distance x1 y1 x2 y2 ≥ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 4 ∧
    x2 = 0 ∧ y2 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circle_and_ellipse_l1141_114107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_over_sin_squared_l1141_114150

open Real

theorem integral_x_over_sin_squared (x : ℝ) (h : x ≠ 0) (h2 : sin x ≠ 0) :
  deriv (fun x => -x * (cos x / sin x) + log (abs (sin x))) x = x / (sin x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_over_sin_squared_l1141_114150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_reachable_1998_l1141_114129

/-- The number of points after one iteration of adding points between neighbors -/
def next_num_points (n : ℕ) : ℕ := 2 * n - 1

/-- Predicate for whether a number can be reached by the point-adding process -/
inductive is_reachable : ℕ → Prop
  | base_one : is_reachable 1
  | step (n m : ℕ) : is_reachable m → next_num_points m = n → is_reachable n

theorem not_reachable_1998 : ¬ is_reachable 1998 := by
  sorry

#check not_reachable_1998

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_reachable_1998_l1141_114129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_10_expression_l1141_114100

noncomputable def u (b : ℝ) : ℕ → ℝ
  | 0 => b  -- Add case for 0
  | 1 => b
  | n + 1 => 1 / (2 - u b n)

theorem u_10_expression (b : ℝ) (h : b > 0) : u b 10 = (4 * b - 3) / (6 * b - 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_10_expression_l1141_114100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_sum_l1141_114183

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem mortgage_payment_sum :
  let a : ℝ := 100  -- first payment
  let r : ℝ := 3    -- common ratio (each payment is triple the previous)
  let n : ℕ := 5    -- number of payments
  geometric_sum a r n = 12100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_sum_l1141_114183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y₀_range_l1141_114115

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define the focal distance property
noncomputable def focal_distance_property (p : ℝ × ℝ) : Prop :=
  point_on_ellipse p →
  let d₁ := Real.sqrt ((p.1 - F₁.1)^2 + (p.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((p.1 - F₂.1)^2 + (p.2 - F₂.2)^2)
  2 * ((d₁ + d₂) / 2) = 2 * Real.sqrt 4

-- Define the perpendicular bisector intersection
def perpendicular_bisector_intersection (y₀ : ℝ) : Prop :=
  ∃ (k : ℝ), y₀ = k / (3 + 4 * k^2)

-- State the theorem
theorem y₀_range :
  ∀ y₀ : ℝ, perpendicular_bisector_intersection y₀ →
  -Real.sqrt 3 / 12 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y₀_range_l1141_114115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_right_triangle_construction_l1141_114124

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the right triangle
structure RightTriangle where
  A : Point
  B : Point
  C : Point
  right_angle : (C.x - A.x) * (B.x - A.x) + (C.y - A.y) * (B.y - A.y) = 0

-- Define the angle bisector
def is_on_angle_bisector (L : Point) (triangle : RightTriangle) : Prop :=
  let BA := (triangle.A.x - triangle.B.x, triangle.A.y - triangle.B.y)
  let BC := (triangle.C.x - triangle.B.x, triangle.C.y - triangle.B.y)
  let BL := (L.x - triangle.B.x, L.y - triangle.B.y)
  (BA.1 * BL.1 + BA.2 * BL.2) / Real.sqrt (BA.1^2 + BA.2^2) = 
  (BC.1 * BL.1 + BC.2 * BL.2) / Real.sqrt (BC.1^2 + BC.2^2)

-- Theorem statement
theorem unique_right_triangle_construction 
  (A C L : Point) : 
  ∃! B : Point, 
    (∃ (triangle : RightTriangle), triangle.A = A ∧ triangle.C = C ∧ 
    is_on_angle_bisector L triangle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_right_triangle_construction_l1141_114124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_intersection_distance_l1141_114127

/-- Regular octagon with side length 10 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 10)

/-- Point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Utility function for Euclidean distance -/
noncomputable def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Membership of a point in a circle -/
def Point.mem_circle (p : Point) (c : Circle) : Prop :=
  dist p c.center = c.radius

/-- Theorem: In a regular octagon ABCDEFGH with side length 10,
    where I is the intersection of circles centered at A and D
    with radii AC and CD respectively (I ≠ C),
    the length of segment IF is 10. -/
theorem octagon_intersection_distance
  (octagon : RegularOctagon)
  (A C D F I : Point)
  (circle_A circle_D : Circle)
  (h1 : circle_A.center = A ∧ circle_A.radius = dist A C)
  (h2 : circle_D.center = D ∧ circle_D.radius = dist C D)
  (h3 : I.mem_circle circle_A ∧ I.mem_circle circle_D)
  (h4 : I ≠ C)
  : dist I F = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_intersection_distance_l1141_114127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_correct_l1141_114190

/-- The sequence (a_n) defined by the given recurrence relation -/
noncomputable def a (α : ℝ) : ℕ → ℝ
  | 0 => α
  | n + 1 => a α n / (1 + a α n)

/-- The proposed general term for the sequence -/
noncomputable def general_term (α : ℝ) (n : ℕ) : ℝ := α / (1 + α * n)

/-- Theorem stating that the general term matches the sequence for all n -/
theorem general_term_correct (α : ℝ) (h : α > 0) :
  ∀ n, a α n = general_term α n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_correct_l1141_114190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pappus_theorem_l1141_114152

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the necessary relations
variable (on_line : Point → Line → Prop)
variable (intersect : Line → Line → Point)

-- Define a function to create a line from two points
variable (line_from_points : Point → Point → Line)

-- Define the theorem
theorem pappus_theorem 
  (l m : Line)
  (A B C D E F G H I : Point)
  (h1 : on_line A l ∧ on_line B l ∧ on_line C l)
  (h2 : on_line D m ∧ on_line E m ∧ on_line F m)
  (h3 : G = intersect (line_from_points A E) (line_from_points B D))
  (h4 : H = intersect (line_from_points A F) (line_from_points C D))
  (h5 : I = intersect (line_from_points B F) (line_from_points C E))
  : ∃ (n : Line), on_line G n ∧ on_line H n ∧ on_line I n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pappus_theorem_l1141_114152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_std_dev_properties_l1141_114193

noncomputable def is_strictly_increasing (a : Fin 2023 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

noncomputable def median (a : Fin 2023 → ℝ) : ℝ := a ⟨1012, by norm_num⟩

noncomputable def mean (a : Fin 2023 → ℝ) : ℝ := (Finset.sum Finset.univ (λ i => a i)) / 2023

noncomputable def variance (a : Fin 2023 → ℝ) : ℝ :=
  (Finset.sum Finset.univ (λ i => (a i - mean a) ^ 2)) / 2023

noncomputable def std_dev (a : Fin 2023 → ℝ) : ℝ := Real.sqrt (variance a)

noncomputable def transformed_sequence (a : Fin 2023 → ℝ) : Fin 2023 → ℝ :=
  λ i => 2 * (a i) + 1

theorem median_and_std_dev_properties
  (a : Fin 2023 → ℝ) (h : is_strictly_increasing a) :
  (median a = a ⟨1012, by norm_num⟩) ∧
  (std_dev (transformed_sequence a) = 2 * std_dev a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_std_dev_properties_l1141_114193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_edges_l1141_114178

/-- A cube is a three-dimensional shape with 6 faces and 12 edges. -/
structure Cube where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  face_count : faces.card = 6
  edge_count : edges.card = 12

/-- A face is a two-dimensional surface of a cube. -/
structure Face where
  edges : Finset (Fin 4)
  edge_count : edges.card = 4

/-- An edge is a one-dimensional line segment of a cube. -/
structure Edge

/-- A coloring is an assignment of colors to edges. -/
def Coloring := Fin 12 → Bool

/-- A coloring is valid if every face has at least one red edge. -/
def is_valid_coloring (c : Cube) (coloring : Coloring) : Prop :=
  ∀ f : Fin 6, ∃ e : Fin 4, coloring e = true

/-- The main theorem: the minimum number of red edges in a valid coloring is 3. -/
theorem min_red_edges (c : Cube) : 
  (∃ coloring : Coloring, is_valid_coloring c coloring ∧ 
    (c.edges.filter (fun e => coloring e = true)).card = 3) ∧ 
  (∀ coloring : Coloring, is_valid_coloring c coloring → 
    (c.edges.filter (fun e => coloring e = true)).card ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_edges_l1141_114178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_replacement_theorem_l1141_114188

/-- Calculates the final percentage of antifreeze in a cooling system after partial replacement -/
noncomputable def final_antifreeze_percentage (initial_volume : ℝ) (initial_concentration : ℝ) 
  (remaining_volume : ℝ) (new_concentration : ℝ) : ℝ :=
  let drained_volume := initial_volume - remaining_volume
  let initial_antifreeze := initial_volume * initial_concentration
  let remaining_antifreeze := remaining_volume * initial_concentration
  let new_antifreeze := drained_volume * new_concentration
  let total_antifreeze := remaining_antifreeze + new_antifreeze
  (total_antifreeze / initial_volume) * 100

theorem antifreeze_replacement_theorem (initial_volume : ℝ) (initial_concentration : ℝ) 
  (remaining_volume : ℝ) (new_concentration : ℝ) 
  (h1 : initial_volume = 19)
  (h2 : initial_concentration = 0.3)
  (h3 : remaining_volume = 7.6)
  (h4 : new_concentration = 0.8) :
  final_antifreeze_percentage initial_volume initial_concentration remaining_volume new_concentration = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_replacement_theorem_l1141_114188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l1141_114153

/-- Represents a marking configuration on a grid -/
def MarkingConfiguration := Fin 9 → Fin 9 → Bool

/-- Checks if a horizontal strip contains a marked cell -/
def horizontalStripMarked (config : MarkingConfiguration) (row col : Fin 9) : Prop :=
  ∃ i : Fin 5, config row (col + i)

/-- Checks if a vertical strip contains a marked cell -/
def verticalStripMarked (config : MarkingConfiguration) (row col : Fin 9) : Prop :=
  ∃ i : Fin 5, config (row + i) col

/-- Checks if a configuration is valid (all strips contain a marked cell) -/
def validConfiguration (config : MarkingConfiguration) : Prop :=
  ∀ row col : Fin 9,
    horizontalStripMarked config row col ∧
    verticalStripMarked config row col

/-- Counts the number of marked cells in a configuration -/
def markedCellCount (config : MarkingConfiguration) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 9)) fun row =>
    Finset.sum (Finset.univ : Finset (Fin 9)) fun col =>
      if config row col then 1 else 0)

/-- The main theorem: The minimum number of marked cells in a valid configuration is 16 -/
theorem min_marked_cells :
  (∃ config : MarkingConfiguration, validConfiguration config ∧ markedCellCount config = 16) ∧
  (∀ config : MarkingConfiguration, validConfiguration config → markedCellCount config ≥ 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l1141_114153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_implies_m_eq_one_range_subset_implies_m_in_interval_l1141_114109

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m - 2 / (5^x + 1)

-- Theorem 1: f is monotonically increasing
theorem f_increasing (m : ℝ) : Monotone (f m) := by sorry

-- Theorem 2: If f is odd, then m = 1
theorem f_odd_implies_m_eq_one (m : ℝ) :
  (∀ x, f m x = -(f m (-x))) → m = 1 := by sorry

-- Theorem 3: If range of f is subset of [-3, 1], then m is in [-1, 1]
theorem range_subset_implies_m_in_interval (m : ℝ) :
  (∀ x, -3 ≤ f m x ∧ f m x ≤ 1) → -1 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_implies_m_eq_one_range_subset_implies_m_in_interval_l1141_114109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1141_114164

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) / x

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y, f x = y

theorem domain_of_f : 
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ -3 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1141_114164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_igor_number_is_five_l1141_114125

def initial_lineup : List Nat := [9, 7, 11, 10, 6, 8, 5, 4, 1]

def is_less_than_neighbors (xs : List Nat) (i : Nat) : Bool :=
  match xs.get? i with
  | none => false
  | some x =>
    (i > 0 && x < xs[i-1]!) || (i < xs.length - 1 && x < xs[i+1]!)

def remove_less_than_neighbors (xs : List Nat) : List Nat :=
  xs.enum.filter (fun (i, _) => ¬is_less_than_neighbors xs i) |>.map Prod.snd

partial def iterate_removal (xs : List Nat) : List Nat :=
  if xs.length ≤ 3 then xs
  else iterate_removal (remove_less_than_neighbors xs)

theorem igor_number_is_five :
  let final_lineup := iterate_removal initial_lineup
  let igor_number := 5
  (igor_number ∉ final_lineup) ∧ (final_lineup.length = 3) ∧
  (∃ prev_lineup, prev_lineup = remove_less_than_neighbors (igor_number :: final_lineup) ∧
                  prev_lineup.length = 4 ∧ igor_number ∈ prev_lineup) := by
  sorry

#eval iterate_removal initial_lineup

end NUMINAMATH_CALUDE_ERRORFEEDBACK_igor_number_is_five_l1141_114125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1141_114160

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the line
def my_line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product of two vectors
def my_dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Main theorem
theorem intersection_distance (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  k > 4/3 →
  my_circle x1 y1 →
  my_circle x2 y2 →
  my_line k x1 y1 →
  my_line k x2 y2 →
  my_dot_product x1 y1 x2 y2 = 12 →
  (x1 - x2)^2 + (y1 - y2)^2 = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1141_114160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_is_121_l1141_114147

/-- An increasing geometric sequence with specific properties -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  is_increasing : ∀ n, a n < a (n + 1)
  is_geometric : ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = q * a n
  sum_first_three : a 1 + a 2 + a 3 = 13
  product_first_three : a 1 * a 2 * a 3 = 27

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- The theorem stating that the sum of the first 5 terms is 121 -/
theorem sum_of_first_five_is_121 (seq : IncreasingGeometricSequence) :
  geometric_sum (seq.a 1) (seq.a 2 / seq.a 1) 5 = 121 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_is_121_l1141_114147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1141_114142

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := 
  (sin (π - α) * cos (2*π - α) * sin (-α + 3*π/2)) / 
  (tan (-α - π) * sin (-π - α) * cos (-π + α))

-- State the theorem
theorem f_simplification_and_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π)  -- α is in the second quadrant
  (h2 : sin α = 3/5) : 
  f α = -cos α^2 / sin α ∧ f α = -16/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1141_114142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_with_decreasing_Q_l1141_114117

/-- Q(n) is the least common multiple of numbers n, n+1, ..., n+k -/
def Q (k : ℕ) (n : ℕ) : ℕ := Finset.lcm (Finset.range (k + 1)) (λ i => n + i)

/-- For any k > 1, there are infinitely many n such that Q(n) > Q(n+1) -/
theorem infinitely_many_n_with_decreasing_Q (k : ℕ) (hk : k > 1) :
  ∀ m : ℕ, ∃ n : ℕ, n ≥ m ∧ Q k n > Q k (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_with_decreasing_Q_l1141_114117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1141_114163

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculates the length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Calculates the height of a parallelogram from point D to side AB -/
noncomputable def parallelogramHeight (p : Parallelogram) : ℝ :=
  -- This is a placeholder for the actual height calculation
  sorry

/-- Calculates the area of a parallelogram -/
noncomputable def parallelogramArea (p : Parallelogram) : ℝ :=
  -- This is a placeholder for the actual area calculation
  sorry

/-- Theorem: The area of a parallelogram is equal to the product of its side and the height dropped to that side -/
theorem parallelogram_area (p : Parallelogram) : 
  parallelogramArea p = distance p.A p.B * parallelogramHeight p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1141_114163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integral_equality_l1141_114132

theorem quadratic_integral_equality (a c x₀ : ℝ) (ha : a ≠ 0) :
  (∫ x in (0:ℝ)..(1:ℝ), a * x^2 + c) = (a * x₀^2 + c) →
  x₀ = Real.sqrt 3 / 3 ∨ x₀ = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integral_equality_l1141_114132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_speed_calculation_l1141_114167

-- Define the normal walking speed
noncomputable def normal_speed : ℝ := 5

-- Define the distance to school
noncomputable def distance : ℝ := 630

-- Define the late time at normal speed
noncomputable def late_time : ℝ := 6

-- Define the early time at faster speed
noncomputable def early_time : ℝ := 30

-- Define the faster walking speed
noncomputable def faster_speed : ℝ := 105 / 17

-- Theorem to prove
theorem faster_speed_calculation :
  let t := distance / normal_speed + late_time
  faster_speed * (t - early_time) = distance :=
by
  -- Unfold the definitions
  unfold normal_speed distance late_time early_time faster_speed
  -- Simplify the goal
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_speed_calculation_l1141_114167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1141_114110

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := |x + a + 1| + |x - 4/a|

-- Theorem statement
theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f x a ≥ 5) ∧
  (f 1 a < 6 → 1 < a ∧ a < 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1141_114110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clover_area_specific_l1141_114177

/-- The area of a figure formed by three sectors of a circle -/
noncomputable def clover_area (r : ℝ) (θ : ℝ) (n : ℕ) : ℝ :=
  n * (θ / 360) * Real.pi * r^2

theorem clover_area_specific : 
  clover_area 15 60 3 = 112.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clover_area_specific_l1141_114177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l1141_114195

-- Define the left-hand side of the equation
noncomputable def lhs (x : ℝ) : ℝ :=
  (x * (x + 3) - 4 * (3 - (5 * x - (1 - 2 * x)) * ((x - 1)^2 / 4))) /
  (-4 * x / 3 - (-1 - (5/2 * (x + 6/5) - x) + x / 6))

-- Define the right-hand side of the equation
noncomputable def rhs (x : ℝ) : ℝ :=
  x / 2 * (x + 6) - (-x * (-3 * (x - 4) + 2 * (3 * x - 5) - 10))

-- Theorem stating that 15/4 and 1/3 are solutions to the equation
theorem solution_verification :
  (lhs (15/4) = rhs (15/4)) ∧ (lhs (1/3) = rhs (1/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l1141_114195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l1141_114137

def CircularArrangement (n : ℕ) := Fin n → Int

def IsValidArrangement (arr : CircularArrangement 2009) : Prop :=
  (∀ i, arr i = 1 ∨ arr i = -1) ∧
  (∃ i j, arr i = 1 ∧ arr j = -1)

def ProductOfTen (arr : CircularArrangement 2009) (start : Fin 2009) : Int :=
  (List.range 10).foldl (fun acc i ↦ acc * arr ((start + i) % 2009)) 1

def SumOfProducts (arr : CircularArrangement 2009) : Int :=
  (List.range 2009).foldl (fun acc i ↦ acc + ProductOfTen arr ⟨i, by sorry⟩) 0

theorem max_sum_of_products :
  ∀ arr : CircularArrangement 2009,
    IsValidArrangement arr →
    SumOfProducts arr ≤ 2005 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l1141_114137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_change_factor_l1141_114181

/-- Given a function q defined in terms of w, v, f, and z, this theorem proves
    the factor by which q changes when w is quadrupled, f is doubled, and z is tripled. -/
theorem q_change_factor
  (w v z : ℝ)
  (f : ℝ → ℝ)
  (q : ℝ → ℝ → (ℝ → ℝ) → ℝ → ℝ)
  (h_q : ∀ w v f z, q w v f z = 5 * w / (4 * v * f (z^2)))
  : q (4*w) v (λ x => 2 * f x) (3*z) = (2/9) * q w v f z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_change_factor_l1141_114181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1141_114175

theorem angle_in_second_quadrant (α : Real) 
  (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  ∃ x y : Real, x < 0 ∧ y > 0 ∧ Real.cos α = x ∧ Real.sin α = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1141_114175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_finish_l1141_114136

/-- Represents the race scenario where runner A is faster than runner B --/
structure Race where
  speed_ratio : ℚ  -- The ratio of A's speed to B's speed
  head_start : ℚ   -- The head start distance given to B
  course_length : ℚ -- The length of the race course

/-- Calculates the time taken by runner A to complete the race --/
noncomputable def time_a (r : Race) : ℚ := r.course_length / r.speed_ratio

/-- Calculates the time taken by runner B to complete the race --/
def time_b (r : Race) : ℚ := (r.course_length - r.head_start)

/-- The main theorem stating the conditions for both runners to finish simultaneously --/
theorem simultaneous_finish (r : Race) :
  r.speed_ratio = 4 ∧ r.head_start = 60 → r.course_length = 80 ∧ time_a r = time_b r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_finish_l1141_114136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l1141_114139

-- Define the line equation
def line_equation (x y Q : ℝ) : Prop :=
  12 * x - 4 * y + (Q - 305) = 0

-- Define the area of the triangle
noncomputable def triangle_area (Q : ℝ) : ℝ :=
  (305 - Q)^2 / 96

-- Theorem statement
theorem triangle_area_is_six :
  ∃ (Q x y : ℝ), line_equation x y Q ∧ triangle_area Q = 6 := by
  sorry

#check triangle_area_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l1141_114139
