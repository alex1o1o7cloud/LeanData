import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_cosine_l1281_128199

/-- A cone with an inscribed sphere -/
structure ConeWithSphere where
  R_w : ℝ  -- radius of the inscribed sphere
  R_k : ℝ  -- radius of the base of the cone
  SA : ℝ   -- slant height of the cone
  α : ℝ    -- angle at the vertex in the axial section

/-- The surface area of the sphere is equal to the area of the base of the cone -/
def surface_area_equal (c : ConeWithSphere) : Prop :=
  4 * Real.pi * c.R_w^2 = Real.pi * c.R_k^2

/-- The radius of the base is twice the radius of the sphere -/
def radius_relation (c : ConeWithSphere) : Prop :=
  c.R_k = 2 * c.R_w

/-- The slant height relates to the radius of the sphere -/
def slant_height_relation (c : ConeWithSphere) : Prop :=
  c.SA = (10 * c.R_w) / 3

/-- Theorem: If the surface area of the sphere is equal to the area of the base of the cone,
    then the cosine of the angle at the vertex in the axial section of the cone is 7/25 -/
theorem cone_sphere_cosine (c : ConeWithSphere) 
  (h1 : surface_area_equal c) 
  (h2 : radius_relation c) 
  (h3 : slant_height_relation c) : 
  Real.cos c.α = 7/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_cosine_l1281_128199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l1281_128106

/-- Calculates the actual profit percentage for a retailer who marks up goods by 50% 
    above cost price and then offers a 15% discount on the marked price. -/
theorem retailer_profit_percentage 
  (cost : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 50) 
  (h2 : discount_percentage = 15) 
  (h3 : cost > 0) : 
  (((cost * (1 + markup_percentage / 100) * (1 - discount_percentage / 100)) - cost) / cost) * 100 = 27.5 := by
  sorry

#check retailer_profit_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l1281_128106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_surface_area_approx_l1281_128133

noncomputable def pyramid_surface_area (base_side_length : ℝ) (height : ℝ) : ℝ :=
  let base_area := 5 * base_side_length^2 / (4 * Real.tan (Real.pi / 5))
  let slant_height := Real.sqrt (height^2 + (base_side_length / (2 * Real.sin (Real.pi / 5)))^2)
  let lateral_area := 5 * 0.5 * base_side_length * slant_height
  base_area + lateral_area

theorem right_pyramid_surface_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |pyramid_surface_area 6 15 - 299.16| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_surface_area_approx_l1281_128133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_range_l1281_128161

/-- An inverse proportion function with parameter m -/
noncomputable def inverse_proportion (m : ℝ) : ℝ → ℝ := fun x ↦ (1 + 2*m) / x

theorem inverse_proportion_m_range 
  (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ < 0) (h₂ : 0 < x₂) (h₃ : y₁ < y₂)
  (h₄ : inverse_proportion m x₁ = y₁)
  (h₅ : inverse_proportion m x₂ = y₂) :
  m > -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_range_l1281_128161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constructible_angles_characterization_l1281_128180

-- Define the set of angles constructible with compass and straightedge alone
def ConstructibleAngles : Set ℝ := sorry

-- Define the set of constructible angles given α
def ConstructibleAnglesWithAlpha (α : ℝ) : Set ℝ :=
  {θ | ∃ (k n : ℤ) (β : ℝ), β ∈ ConstructibleAngles ∧ θ = (k : ℝ) / (2 : ℝ) ^ n * α + β}

-- Define the predicate for constructible angles
def IsConstructibleAngle (α θ : ℝ) : Prop := sorry

-- Theorem statement
theorem constructible_angles_characterization (α : ℝ) :
  {θ | IsConstructibleAngle α θ} = ConstructibleAnglesWithAlpha α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constructible_angles_characterization_l1281_128180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_prism_surface_area_equals_88_l1281_128113

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The volume of a rectangular prism -/
def rect_prism_volume (l w h : ℝ) : ℝ := l * w * h

/-- The surface area of a rectangular prism -/
def rect_prism_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

theorem rect_prism_surface_area_equals_88 (r l w : ℝ) (h : ℝ) :
  r = 3 * (36 / Real.pi) →
  l = 6 →
  w = 4 →
  sphere_volume r = rect_prism_volume l w h →
  rect_prism_surface_area l w h = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_prism_surface_area_equals_88_l1281_128113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_zeros_in_unit_circle_l1281_128174

open Complex

-- Define the complex polynomial function F(z)
def F (z : ℂ) : ℂ := z^8 - 4*z^5 + z^2 - 1

-- Define the unit circle
def unit_circle : Set ℂ := {z : ℂ | abs z < 1}

-- Theorem statement
theorem F_zeros_in_unit_circle :
  ∃ (S : Finset ℂ), S.card = 5 ∧ ∀ z ∈ S, F z = 0 ∧ z ∈ unit_circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_zeros_in_unit_circle_l1281_128174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_properties_l1281_128153

def has_exactly_n_factors (x n : ℕ) : Prop :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = n

def is_smallest_with_properties (x : ℕ) : Prop :=
  has_exactly_n_factors x 8 ∧ 14 ∣ x ∧ 18 ∣ x ∧
  ∀ y : ℕ, y < x → ¬(has_exactly_n_factors y 8 ∧ 14 ∣ y ∧ 18 ∣ y)

theorem smallest_integer_with_properties :
  is_smallest_with_properties 1134 := by
  sorry

#check smallest_integer_with_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_properties_l1281_128153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_RST_value_l1281_128175

-- Define the points
variable (P Q R S T : EuclideanSpace ℝ (Fin 2))

-- Define the angles
def angle_PQR : ℝ := 40
variable (x : ℝ)
def angle_RST : ℝ := x

-- Define the properties
axiom PRT_straight : Collinear ℝ P R T
axiom QRS_straight : Collinear ℝ Q R S
axiom PQ_eq_QR : dist P Q = dist Q R
axiom RS_eq_RT : dist R S = dist R T

-- State the theorem
theorem angle_RST_value :
  angle_RST = 55 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_RST_value_l1281_128175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1281_128114

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (a^2 - b^2) / a^2 = 3/4  -- eccentricity squared
  h4 : b = 1  -- passing through (0,1)

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  e.a = 2 ∧ e.b = 1

/-- Two points on the ellipse -/
structure EllipsePoints (e : Ellipse) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  h1 : (M.1^2 / e.a^2) + (M.2^2 / e.b^2) = 1
  h2 : (N.1^2 / e.a^2) + (N.2^2 / e.b^2) = 1
  h3 : (M.1 - 0) * (N.1 - 0) + (M.2 - 1) * (N.2 - 1) = 0  -- perpendicular lines

/-- The fixed point that MN passes through -/
noncomputable def fixed_point : ℝ × ℝ := (0, -3/5)

/-- Main theorem -/
theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧
  ∀ (p : EllipsePoints e), ∃ (t : ℝ),
    fixed_point.1 = (1 - t) * 0 + t * ((p.M.1 + p.N.1) / 2) ∧
    fixed_point.2 = (1 - t) * (-3/5) + t * ((p.M.2 + p.N.2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1281_128114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1281_128121

/-- Given a hyperbola with equation y²/a² - x²/b² = 1, where a > 0 and b > 0,
    and eccentricity 3, prove that its asymptotes are given by x ± 2√2y = 0 -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (Real.sqrt (a^2 + b^2) / a = 3) →
  (∀ x y : ℝ, x = 2 * Real.sqrt 2 * y ∨ x = -2 * Real.sqrt 2 * y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1281_128121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1281_128131

-- Define the inclination angle function
noncomputable def inclination_angle (θ : ℝ) : ℝ := 
  Real.arctan (Real.cos θ)

-- State the theorem
theorem inclination_angle_range :
  Set.range inclination_angle = Set.union (Set.Icc 0 (Real.pi / 4)) (Set.Ico (3 * Real.pi / 4) Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1281_128131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_for_given_sine_l1281_128171

theorem cosine_symmetry_for_given_sine (k : ℝ) (h : 0 < k ∧ k ≤ 1) :
  ∃ (α β : ℝ), 0 ≤ α ∧ α < 2 * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧
    Real.sin α = k ∧ Real.sin β = k ∧
    Real.cos α = -Real.cos β ∧
    α + β = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_for_given_sine_l1281_128171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sunday_miles_l1281_128134

/-- Represents the number of miles Bill ran on Saturday -/
def bill_saturday : ℕ := sorry

/-- Represents the number of miles Bill ran on Sunday -/
def bill_sunday : ℕ := bill_saturday + 4

/-- Represents the number of miles Julia ran on Sunday -/
def julia_sunday : ℕ := 2 * bill_sunday

/-- Represents the number of miles Alex ran in total -/
def alex_total : ℕ := (bill_saturday + bill_sunday) / 2

theorem bill_sunday_miles :
  bill_saturday + bill_sunday + julia_sunday + alex_total = 54 →
  bill_sunday = 14 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sunday_miles_l1281_128134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athletes_scores_comparison_l1281_128105

noncomputable def scores_A : List ℝ := [8, 9, 14, 15, 15, 16, 21, 22]
noncomputable def scores_B : List ℝ := [7, 8, 13, 15, 15, 17, 22, 23]

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum) / (scores.length : ℝ)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (fun x => (x - m) ^ 2)).sum / (scores.length : ℝ)

theorem athletes_scores_comparison :
  (mean scores_A = mean scores_B) ∧
  (variance scores_A < variance scores_B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athletes_scores_comparison_l1281_128105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1281_128149

/-- The function f(x) = sin(ωx + π/3) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

/-- The theorem stating the monotonic decreasing interval of f(x) -/
theorem monotonic_decreasing_interval (ω : ℝ) (h1 : ω > 0) 
  (h2 : (Real.pi / ω) = Real.pi / 2) : 
  ∃ (a b : ℝ), a = Real.pi / 12 ∧ b = Real.pi / 2 ∧ 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f ω x > f ω y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1281_128149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_in_intervals_l1281_128129

open Real

noncomputable def f (x : ℝ) : ℝ := (1/3) * x * log x

theorem no_zeros_in_intervals (h : ∀ x > 0, f x = 0 ↔ x = 1) :
  (∀ x ∈ Set.Ioo (1/Real.exp 1) 1, f x ≠ 0) ∧ 
  (∀ x ∈ Set.Ioo 1 (Real.exp 1), f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_in_intervals_l1281_128129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_square_of_linear_implies_m_range_l1281_128181

/-- G is a function of x and m -/
noncomputable def G (x m : ℝ) : ℝ := (9 * x^2 + 27 * x + 4 * m) / 9

/-- LinearExpression represents a linear expression in x -/
structure LinearExpression (x : ℝ) where
  a : ℝ
  b : ℝ
  expr : ℝ := a * x + b

/-- Theorem stating that if G is the square of a linear expression in x, then 5 < m < 6 -/
theorem G_square_of_linear_implies_m_range (x m : ℝ) :
  (∃ (L : LinearExpression x), G x m = L.expr ^ 2) → 5 < m ∧ m < 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_square_of_linear_implies_m_range_l1281_128181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_eccentricity_l1281_128152

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_parabola_intersection_eccentricity 
  (h : Hyperbola) (p : Parabola) (A B : Point) :
  (A.x^2 / h.a^2 - A.y^2 / h.b^2 = 1) →  -- A is on the hyperbola
  (B.x^2 / h.a^2 - B.y^2 / h.b^2 = 1) →  -- B is on the hyperbola
  (A.y^2 = 2 * p.p * A.x) →              -- A is on the parabola
  (B.y^2 = 2 * p.p * B.x) →              -- B is on the parabola
  (∃ (t : ℝ), A.x + t * (B.x - A.x) = p.p ∧ 
              A.y + t * (B.y - A.y) = 0) →  -- AB passes through parabola focus (p, 0)
  ((B.x - A.x)^2 + (B.y - A.y)^2 = 4 * h.b^2) →  -- Length of AB equals conjugate axis
  eccentricity h = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_eccentricity_l1281_128152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_count_l1281_128151

/-- The maximum number of pairs (m,n) where m ∈ A, n ∈ B, and |m-n| ≤ 1000,
    given that A and B are sets of distinct integers with |A| = 2000 and |B| = 2016 -/
theorem max_pairs_count (A B : Finset ℤ) (hA : A.card = 2000) (hB : B.card = 2016) :
  (Finset.sum A fun m => (B.filter (fun n => |m - n| ≤ 1000)).card) ≤ 3015636 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_count_l1281_128151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_inequality_l1281_128127

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (16 * x) / (x^2 + 8)

-- Theorem for the maximum value of f(x) when x > 0
theorem f_max_value : ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), x > 0 → f x ≤ M := by
  sorry

-- Theorem for the inequality
theorem f_inequality (a b : ℝ) : f a < b^2 - 3*b + 21/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_inequality_l1281_128127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l1281_128145

/-- Predicate to represent that two angles have the same terminal side -/
def HasSameTerminalSide (angle1 : ℤ) (angle2 : ℤ) : Prop :=
  (angle1 % 360 = angle2 % 360) ∨ (angle1 % 360 = -angle2 % 360)

/-- For any integer k, the angles (2k+1)180° and (4k±1)180° have the same terminal side. -/
theorem same_terminal_side (k : ℤ) : 
  ∀ (sign : Bool), 
  HasSameTerminalSide ((2*k + 1) * 180) ((4*k + (if sign then 1 else -1)) * 180) := by
  intro sign
  unfold HasSameTerminalSide
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l1281_128145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_quartic_roots_l1281_128142

theorem determinant_of_quartic_roots (a b c d p q r : ℂ) : 
  (a^4 + p*a^2 + q*a + r = 0) →
  (b^4 + p*b^2 + q*b + r = 0) →
  (c^4 + p*c^2 + q*c + r = 0) →
  (d^4 + p*d^2 + q*d + r = 0) →
  Matrix.det (
    !![1+a, 1, 1, 1;
       1, 1+b, 1, 1;
       1, 1, 1+c, 1;
       1, 1, 1, 1+d]
  ) = r + q + p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_quartic_roots_l1281_128142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calligraphy_recitation_ratio_l1281_128119

theorem calligraphy_recitation_ratio : 
  ∀ (x : ℚ), x > 0 → 
  (x + (2/7 : ℚ) * x) / (x + (1/5 : ℚ) * x) = 3/4 := by
  intro x hx
  -- Simplify the numerator and denominator
  have h1 : x + (2/7 : ℚ) * x = (9/7 : ℚ) * x := by
    ring
  have h2 : x + (1/5 : ℚ) * x = (6/5 : ℚ) * x := by
    ring
  -- Rewrite the fraction
  rw [h1, h2]
  -- Simplify the fraction
  field_simp
  ring
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calligraphy_recitation_ratio_l1281_128119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_geometric_number_starting_with_2_l1281_128187

/-- A number is geometric if its digits form a geometric sequence -/
def IsGeometric (n : ℕ) : Prop :=
  ∃ a r : ℚ, r ≠ 1 ∧ n = (a * 100).floor + ((a * r) * 10).floor + (a * r * r).floor

/-- Check if a number has 3 distinct non-zero digits -/
def HasThreeDistinctNonZeroDigits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10) ∧
  (n / 100) ≠ 0 ∧ ((n / 10) % 10) ≠ 0 ∧ (n % 10) ≠ 0

theorem smallest_geometric_number_starting_with_2 :
  (∀ m : ℕ, m < 261 → ¬(IsGeometric m ∧ HasThreeDistinctNonZeroDigits m ∧ m / 100 = 2)) ∧
  (IsGeometric 261 ∧ HasThreeDistinctNonZeroDigits 261 ∧ 261 / 100 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_geometric_number_starting_with_2_l1281_128187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_l1281_128159

theorem cube_root_plus_abs_plus_power : (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^(2023 : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_l1281_128159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_identity_transformation_l1281_128195

noncomputable section

open Real

/-- Angle of line l₁ with positive x-axis -/
def θ₁ : ℝ := π / 60

/-- Angle of line l₂ with positive x-axis -/
def θ₂ : ℝ := π / 48

/-- Slope of line l -/
def k : ℝ := 17 / 89

/-- Angle change after one application of R -/
def δθ : ℝ := 7 * π / 240

/-- The transformation R applied n times -/
def R (n : ℕ) (θ : ℝ) : ℝ := θ + n * δθ

/-- The proposition to be proved -/
theorem smallest_m_for_identity_transformation :
  (∀ n < 480, R n (arctan k) ≠ arctan k) ∧
  R 480 (arctan k) = arctan k := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_identity_transformation_l1281_128195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1281_128130

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ := λ n => a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Part 1
theorem part1 (a : ℕ → ℚ) (n : ℕ) :
  a 4 = 10 → a 10 = -2 → sum_arithmetic_sequence (a 1) ((a 2) - (a 1)) n = 60 →
  n = 5 ∨ n = 6 := by sorry

-- Part 2
theorem part2 (a : ℕ → ℚ) :
  a 1 = -7 → (∀ n : ℕ, a (n + 1) = a n + 2) →
  sum_arithmetic_sequence (-7) 2 17 = 153 := by sorry

-- Part 3
theorem part3 (a : ℕ → ℚ) :
  a 2 + a 7 + a 12 = 24 →
  sum_arithmetic_sequence (a 1) ((a 2) - (a 1)) 13 = 104 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1281_128130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_triangle_l1281_128157

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) - 1

theorem f_range_in_triangle (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < 2 * Real.pi / 3) →
  (0 < B ∧ B < Real.pi) →
  (0 < C ∧ C < Real.pi) →
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a * c * Real.cos (Real.pi - B) = -1/2 * a * c) →
  (∀ x, f (Real.pi/3) ≥ f x) →
  (∀ x, f (-Real.pi/6) ≤ f x) →
  (f (Real.pi/3) = 1) →
  (f (-Real.pi/6) = -3) →
  ∀ y ∈ Set.Ioo (-2 : ℝ) 1, ∃ A', f A' = y ∧ 0 < A' ∧ A' < 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_triangle_l1281_128157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_contributors_l1281_128117

/-- The maximum number of contributors to a gift -/
def max_contributors (total : ℚ) (min_contribution : ℚ) (max_contribution : ℚ) : ℕ :=
  (((total - max_contribution) / min_contribution).floor + 1).toNat

/-- Theorem stating the maximum number of contributors for the given conditions -/
theorem gift_contributors :
  max_contributors 30 1 16 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_contributors_l1281_128117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neg_135_deg_to_rad_eleven_pi_thirds_to_deg_l1281_128124

-- Define the conversion factor between radians and degrees
noncomputable def rad_to_deg : ℝ := 180 / Real.pi

-- Theorem 1: -135 degrees is equal to -3π/4 radians
theorem neg_135_deg_to_rad :
  -135 / rad_to_deg = -3 * Real.pi / 4 := by sorry

-- Theorem 2: 11π/3 radians is equal to 660 degrees
theorem eleven_pi_thirds_to_deg :
  11 * Real.pi / 3 * rad_to_deg = 660 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neg_135_deg_to_rad_eleven_pi_thirds_to_deg_l1281_128124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rogers_cookie_price_l1281_128179

/-- Represents the shape of a cookie -/
inductive CookieShape
| Trapezoid
| Rectangle
| Circle
| Triangle

/-- Represents a friend who bakes cookies -/
structure Friend where
  name : String
  shape : CookieShape
  count : ℕ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (base1 base2 height : ℝ) : ℝ :=
  (base1 + base2) * height / 2

/-- Theorem: Roger's cookies should cost 56 cents each -/
theorem rogers_cookie_price
  (art roger : Friend)
  (h_art_shape : art.shape = CookieShape.Trapezoid)
  (h_roger_shape : roger.shape = CookieShape.Rectangle)
  (h_art_count : art.count = 15)
  (h_roger_count : roger.count = 20)
  (h_art_base1 h_art_base2 h_art_height : ℝ)
  (h_art_dimensions : h_art_base1 = 3 ∧ h_art_base2 = 7 ∧ h_art_height = 4)
  (h_art_price : ℝ) (h_art_price_value : h_art_price = 75)
  (h_same_dough : ℝ) -- Total amount of dough used by each friend
  : ∃ (roger_price : ℝ), roger_price = 56 := by
  sorry

#check rogers_cookie_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rogers_cookie_price_l1281_128179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_calculation_l1281_128191

/-- Represents the setup of a sphere and cone casting shadows -/
structure ShadowSetup where
  sphere_shadow : ℝ
  cone_height : ℝ
  cone_shadow : ℝ

/-- Calculates the radius of a sphere given a shadow setup -/
noncomputable def sphere_radius (setup : ShadowSetup) : ℝ :=
  setup.sphere_shadow * (setup.cone_height / setup.cone_shadow)

/-- Theorem stating that under given conditions, the sphere radius is 11.25 m -/
theorem sphere_radius_calculation (setup : ShadowSetup) 
  (h1 : setup.sphere_shadow = 15)
  (h2 : setup.cone_height = 3)
  (h3 : setup.cone_shadow = 4) :
  sphere_radius setup = 11.25 := by
  sorry

#check sphere_radius_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_calculation_l1281_128191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_norm_is_16_l1281_128156

noncomputable def z : ℂ := ((1 + Complex.I * Real.sqrt 3) * (3 - Complex.I)^2) / (3 - 4 * Complex.I)

theorem z_squared_norm_is_16 : Complex.normSq z = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_norm_is_16_l1281_128156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raghu_investment_l1281_128190

def investment_problem (geeta raghu trishul vishal suresh : ℝ) : Prop :=
  raghu = geeta * 1.05 ∧
  trishul = raghu * 0.9 ∧
  vishal = trishul * 1.1 ∧
  suresh = geeta * 0.8 ∧
  geeta + raghu + trishul + vishal + suresh = 21824

theorem raghu_investment :
  ∃ (geeta raghu trishul vishal suresh : ℝ),
    investment_problem geeta raghu trishul vishal suresh ∧
    (abs (raghu - 4742.40) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raghu_investment_l1281_128190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rates_l1281_128128

/-- Represents the work rate of a person or group of people in units of work per day -/
structure WorkRate where
  value : ℝ
  pos : value > 0

/-- Represents the time taken to complete a task in days -/
structure Time where
  value : ℝ
  pos : value > 0

/-- The total amount of work to be done -/
def totalWork : ℝ := 1

variable (M P S : WorkRate)

/-- Matt, Peter, and Sarah's combined work rate -/
def combinedRate (M P S : WorkRate) : WorkRate where
  value := M.value + P.value + S.value
  pos := by
    simp
    exact add_pos (add_pos M.pos P.pos) S.pos

/-- Peter and Sarah's combined work rate -/
def peterSarahRate (P S : WorkRate) : WorkRate where
  value := P.value + S.value
  pos := by
    simp
    exact add_pos P.pos S.pos

/-- Time taken by Matt, Peter, and Sarah together -/
def totalTime : Time where
  value := 15
  pos := by norm_num

/-- Time worked together before Matt stops -/
def workTimeTogether : Time where
  value := 8
  pos := by norm_num

/-- Time taken by Peter and Sarah to complete the remaining work -/
def remainingTime : Time where
  value := 10
  pos := by norm_num

theorem work_rates (h1 : (combinedRate M P S).value = 1 / totalTime.value)
                   (h2 : (peterSarahRate P S).value = (totalWork - workTimeTogether.value * (combinedRate M P S).value) / remainingTime.value) :
  M.value = 1 / 50 ∧ (peterSarahRate P S).value = 7 / 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rates_l1281_128128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1281_128193

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem f_minimum_value :
  (∀ x > 2, f x ≥ 4) ∧ (∃ x > 2, f x = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1281_128193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_pay_approx_l1281_128101

/-- The weekly pay of employee Y given the total pay and the pay ratio between X and Y -/
noncomputable def employee_y_pay (total_pay : ℝ) (x_ratio : ℝ) : ℝ :=
  total_pay / (1 + x_ratio)

theorem employee_y_pay_approx :
  let total_pay := (590 : ℝ)
  let x_ratio := (1.2 : ℝ)
  let y_pay := employee_y_pay total_pay x_ratio
  ∃ ε > 0, |y_pay - 268.18| < ε ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_pay_approx_l1281_128101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_Q_zeros_l1281_128150

-- Define the polynomial Q(z)
noncomputable def Q (z : ℂ) : ℂ := z^8 + (6 * Real.sqrt 2 + 8) * z^4 - (6 * Real.sqrt 2 + 9)

-- Define the set of zeros of Q(z)
def zeros_of_Q : Set ℂ := {z : ℂ | Q z = 0}

-- Define the concept of an 8-sided polygon in the complex plane
def is_octagon (vertices : Finset ℂ) : Prop := vertices.card = 8

-- Define the perimeter of a polygon given its vertices
noncomputable def perimeter (vertices : Finset ℂ) : ℝ := sorry

-- Theorem statement
theorem min_perimeter_of_Q_zeros :
  ∃ (vertices : Finset ℂ), is_octagon vertices ∧ 
    ↑vertices = zeros_of_Q ∧ 
    perimeter vertices = 8 * Real.sqrt 2 ∧
    ∀ (other_vertices : Finset ℂ), is_octagon other_vertices → 
      ↑other_vertices = zeros_of_Q → 
      perimeter other_vertices ≥ 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_Q_zeros_l1281_128150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_positivity_l1281_128132

theorem sine_positivity (x : ℝ) : (x^2 + 5*x + 6) * (x^2 + 11*x + 30) < 0 → Real.sin (2*x) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_positivity_l1281_128132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bamboo_problem_l1281_128196

-- Define a geometric sequence of 9 terms
def geometric_sequence (a : ℝ) (r : ℝ) : Fin 9 → ℝ := fun n ↦ a * r ^ (n : ℕ)

theorem bamboo_problem (a : ℝ) (r : ℝ) (h1 : r > 0) 
  (h2 : geometric_sequence a r 0 + geometric_sequence a r 1 + geometric_sequence a r 2 = 2)
  (h3 : geometric_sequence a r 6 + geometric_sequence a r 7 + geometric_sequence a r 8 = 128) :
  geometric_sequence a r 4 = 32/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bamboo_problem_l1281_128196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l1281_128192

/-- The volume of a cone in cubic inches -/
noncomputable def cone_volume : ℝ := 12288 * Real.pi

/-- The vertex angle of the vertical cross-section in radians -/
noncomputable def vertex_angle : ℝ := Real.pi / 3

/-- The height of the cone in inches -/
def cone_height : ℝ := 48

theorem cone_height_calculation :
  cone_height = 48 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#eval cone_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l1281_128192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_theorem_l1281_128102

def is_sum_of_three_subset (A : Finset Int) (s : Int) : Prop :=
  ∃ (x y z : Int), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = s

theorem subset_sum_theorem (A : Finset Int) :
  A.card = 4 →
  (∀ s, s ∈ ({-1, 3, 5, 8} : Finset Int) ↔ is_sum_of_three_subset A s) →
  A = {-3, 0, 2, 6} := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_theorem_l1281_128102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_tip_percentage_l1281_128178

/-- Represents the dinner scenario with John and Jane --/
structure DinnerScenario where
  originalPrice : ℚ
  discountRate : ℚ
  johnExtraPay : ℚ

/-- Calculates the tip percentage John paid --/
def calculateJohnTipPercentage (scenario : DinnerScenario) : ℚ :=
  let discountedPrice := scenario.originalPrice * (1 - scenario.discountRate)
  let tipDifference := scenario.johnExtraPay
  tipDifference / (scenario.originalPrice - discountedPrice)

/-- Theorem stating that John's tip percentage is 15% given the scenario --/
theorem john_tip_percentage (scenario : DinnerScenario) 
  (h1 : scenario.originalPrice = 40)
  (h2 : scenario.discountRate = 1/10)
  (h3 : scenario.johnExtraPay = 3/5) :
  calculateJohnTipPercentage scenario = 15/100 := by
  sorry

#eval calculateJohnTipPercentage { originalPrice := 40, discountRate := 1/10, johnExtraPay := 3/5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_tip_percentage_l1281_128178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_x_2_minus_x_l1281_128115

theorem integral_sqrt_x_2_minus_x : 
  ∫ x in (0:ℝ)..(1:ℝ), Real.sqrt (x * (2 - x)) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_x_2_minus_x_l1281_128115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlas_car_mpg_l1281_128183

/-- Calculates the miles per gallon of a car given the total distance, gas cost, and total spent on gas. -/
noncomputable def miles_per_gallon (total_distance : ℝ) (gas_cost_per_gallon : ℝ) (total_spent : ℝ) : ℝ :=
  total_distance / (total_spent / gas_cost_per_gallon)

/-- Theorem stating that Carla's car gets 25 miles per gallon. -/
theorem carlas_car_mpg :
  let total_distance : ℝ := 50
  let gas_cost_per_gallon : ℝ := 2.5
  let total_spent : ℝ := 5
  miles_per_gallon total_distance gas_cost_per_gallon total_spent = 25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlas_car_mpg_l1281_128183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l1281_128148

/-- Triangle with vertices A(2,3), B(2,9), and C(6,6) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (2, 9)
  C : ℝ × ℝ := (6, 6)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate the perimeter of the triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Calculate the area of the triangle using the coordinate formula -/
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * abs (t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2))

/-- Theorem: The perimeter of the triangle is 16 and its area is 12 -/
theorem triangle_perimeter_and_area (t : Triangle) : perimeter t = 16 ∧ area t = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l1281_128148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_path_length_l1281_128165

/-- The length of the path traversed by a vertex of an equilateral triangle rotating around a square --/
theorem triangle_rotation_path_length 
  (square_side : ℝ) 
  (triangle_side : ℝ) 
  (h_square : square_side = 6) 
  (h_triangle : triangle_side = 3) : 
  (4 * 3 * π : ℝ) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_path_length_l1281_128165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_reduction_l1281_128168

/-- Represents the fruit selling scenario -/
structure FruitSeller where
  initialPrice : ℚ
  initialSales : ℚ
  priceReductionEffect : ℚ
  funds : ℚ
  costPrice : ℚ
  desiredProfit : ℚ

/-- Calculates the daily sales volume based on price reduction -/
def dailySales (fs : FruitSeller) (priceReduction : ℚ) : ℚ :=
  fs.initialSales + (priceReduction / fs.priceReductionEffect) * 40

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (fs : FruitSeller) (priceReduction : ℚ) : ℚ :=
  (fs.initialPrice - fs.costPrice - priceReduction) * (dailySales fs priceReduction)

/-- Checks if the purchase is within budget -/
def withinBudget (fs : FruitSeller) (priceReduction : ℚ) : Prop :=
  fs.funds ≥ fs.costPrice * (dailySales fs priceReduction)

/-- Theorem stating that a 0.5 yuan price reduction achieves the desired profit within budget -/
theorem optimal_price_reduction (fs : FruitSeller) 
    (h1 : fs.initialPrice = 4) 
    (h2 : fs.initialSales = 100) 
    (h3 : fs.priceReductionEffect = 0.2) 
    (h4 : fs.funds = 500) 
    (h5 : fs.costPrice = 2) 
    (h6 : fs.desiredProfit = 300) : 
    dailyProfit fs (1/2) = fs.desiredProfit ∧ withinBudget fs (1/2) := by
  sorry

#check optimal_price_reduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_reduction_l1281_128168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1281_128112

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then Real.log (-x) + 2 * x else Real.log x + 2 * x

-- State the theorem
theorem tangent_line_equation 
  (h1 : even_function f)
  (h2 : ∀ x < 0, f x = Real.log (-x) + 2 * x)
  : let df := deriv f
    (df 1 = 3) ∧ (f 1 = -2) ∧ 
    (∀ x y, y = 3 * x - 5 ↔ y - (-2) = 3 * (x - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1281_128112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l1281_128126

theorem triangle_area_formula (S r rc p α β γ : ℝ) :
  rc > 0 →
  α > 0 → α < π →
  β > 0 → β < π →
  γ > 0 → γ < π →
  r = rc * Real.tan (α / 2) * Real.tan (β / 2) →
  p = rc * (Real.tan (γ / 2))⁻¹ →
  S = r * p →
  S = rc^2 * Real.tan (α / 2) * Real.tan (β / 2) * (Real.tan (γ / 2))⁻¹ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l1281_128126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_excluding_extremes_is_58_l1281_128110

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  average : ℚ
  innings : ℕ
  highestScore : ℕ
  scoreDifference : ℕ

/-- Calculates the average score excluding highest and lowest scores -/
def averageExcludingExtremes (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.average * stats.innings
  let lowestScore := stats.highestScore - stats.scoreDifference
  let runsExcludingExtremes := totalRuns - (stats.highestScore + lowestScore)
  runsExcludingExtremes / (stats.innings - 2)

/-- Theorem stating the average excluding extremes for given stats -/
theorem average_excluding_extremes_is_58 (stats : BatsmanStats)
  (h1 : stats.average = 61)
  (h2 : stats.innings = 46)
  (h3 : stats.highestScore = 202)
  (h4 : stats.scoreDifference = 150) :
  averageExcludingExtremes stats = 58 := by
  sorry

#eval averageExcludingExtremes ⟨61, 46, 202, 150⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_excluding_extremes_is_58_l1281_128110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_convergence_l1281_128162

def y : ℕ → ℕ
  | 0 => 50  -- Add this case to handle Nat.zero
  | 1 => 50
  | k + 2 => y (k + 1) * y (k + 1) - y (k + 1)

theorem sequence_sum_convergence :
  let series := fun n => 1 / (y n + 1 : ℝ)
  (∑' n, series n) = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_convergence_l1281_128162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l1281_128120

theorem absolute_value_nested_expression : 
  abs (abs (abs (-3 + 2) - 2) + 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l1281_128120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_theorem_l1281_128188

theorem root_product_theorem (a b : ℚ) : 
  (Real.sqrt (28 - 10 * Real.sqrt 3))^2 + a * (Real.sqrt (28 - 10 * Real.sqrt 3)) + b = 0 →
  a * b = -220 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_theorem_l1281_128188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_moles_in_reaction_l1281_128135

-- Define the chemical reaction
noncomputable def reaction (li3n : ℝ) (h2o : ℝ) : ℝ × ℝ :=
  (min li3n h2o, min li3n (h2o / 3))

-- Theorem statement
theorem water_moles_in_reaction 
  (li3n_moles : ℝ) 
  (h2o_moles : ℝ) 
  (nh3_produced : ℝ) :
  li3n_moles = 3 →
  (reaction li3n_moles h2o_moles).2 = 3 →
  h2o_moles = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_moles_in_reaction_l1281_128135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1281_128107

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from P to a point
def vector_PA (P A : ℝ × ℝ) : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)

theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e : Set (ℝ × ℝ) := Ellipse a b
  let M : ℝ × ℝ := (Real.sqrt 6, 1)
  let P : ℝ × ℝ := (Real.sqrt 6, 0)
  (M ∈ e) →
  ((a^2 - b^2) / a^2 = 1/2) →
  (∃ (A B : ℝ × ℝ), A ∈ e ∧ B ∈ e ∧
    dot_product (vector_PA P A) (vector_PA P B) = -2) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∀ (A B : ℝ × ℝ), A ∈ e → B ∈ e →
    dot_product (vector_PA P A) (vector_PA P B) = -2 →
    ∃ (t : ℝ), A.2 - B.2 = (A.1 - B.1) * t ∧ 
    A.1 + t * A.2 = 2 * Real.sqrt 6 / 3 ∧
    B.1 + t * B.2 = 2 * Real.sqrt 6 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1281_128107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brocard_and_interior_angle_bound_l1281_128198

-- Define a triangle
structure Triangle where
  A : Real × Real
  B : Real × Real
  C : Real × Real

-- Define the Brocard angle of a triangle
noncomputable def brocardAngle (t : Triangle) : Real := sorry

-- Define a point inside a triangle
def isInside (p : Real × Real) (t : Triangle) : Prop := sorry

-- Define the angle between three points
noncomputable def angle (p1 p2 p3 : Real × Real) : Real := sorry

-- Theorem statement
theorem brocard_and_interior_angle_bound (t : Triangle) (M : Real × Real) 
  (h : isInside M t) : 
  brocardAngle t ≤ 30 * Real.pi / 180 ∧ 
  min (angle t.A t.B M) (min (angle t.B t.C M) (angle t.C t.A M)) ≤ 30 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brocard_and_interior_angle_bound_l1281_128198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_cone_lateral_area_l1281_128167

theorem sphere_volume_equals_cone_lateral_area (h : ℝ) (r : ℝ) :
  h = 3 →
  r = 4 →
  let cone_lateral_area := π * r * Real.sqrt (h^2 + r^2)
  let sphere_radius := Real.sqrt (cone_lateral_area / (4 * π))
  let sphere_volume := (4/3) * π * sphere_radius^3
  sphere_volume = (20 * Real.sqrt 5 * π) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_cone_lateral_area_l1281_128167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_and_limit_l1281_128189

open Real Set

/-- The polynomial P_n(x) = x^(n+2) - 2x + 1 -/
def P (n : ℕ) (x : ℝ) : ℝ := x^(n+2) - 2*x + 1

/-- The unique root c_n of P_n(x) in (0,1) -/
noncomputable def c (n : ℕ) : ℝ := sorry

theorem unique_root_and_limit :
  (∀ n : ℕ, ∃! c_n : ℝ, c_n ∈ Ioo 0 1 ∧ P n c_n = 0) ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |c n - 1/2| < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_and_limit_l1281_128189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_obtuse_angle_l1281_128125

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the hexagon vertices -/
noncomputable def hexagonVertices : List Point := [
  ⟨0, 0⟩,
  ⟨3, 0⟩,
  ⟨4.5, Real.sqrt 3⟩,
  ⟨3, 2 * Real.sqrt 3⟩,
  ⟨0, 2 * Real.sqrt 3⟩,
  ⟨-1.5, Real.sqrt 3⟩
]

/-- Calculates the area of the hexagon -/
noncomputable def hexagonArea : ℝ := (27 * Real.sqrt 3) / 2

/-- Calculates the area of the quarter circle -/
noncomputable def quarterCircleArea : ℝ := (9 * Real.pi) / 4

/-- Theorem: The probability of ∠APB being obtuse when P is randomly selected
    from the interior of the hexagon is (6√3 - π) / (6√3) -/
theorem probability_obtuse_angle :
  (hexagonArea - quarterCircleArea) / hexagonArea = (6 * Real.sqrt 3 - Real.pi) / (6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_obtuse_angle_l1281_128125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l1281_128144

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  f₁ : Point -- Left focus
  f₂ : Point -- Right focus
  a : ℝ      -- Half of the constant difference

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Perimeter of triangle ABF₂ in a hyperbola -/
theorem hyperbola_triangle_perimeter
  (h : Hyperbola)
  (a b : Point)
  (on_hyperbola : ∀ p : Point, p = a ∨ p = b → 
    distance p h.f₂ - distance p h.f₁ = 2 * h.a)
  (passes_through_f₁ : distance a h.f₁ + distance b h.f₁ = distance a b)
  (chord_length : distance a b = 5)
  (h_2a : 2 * h.a = 8) :
  distance a h.f₂ + distance b h.f₂ + distance a b = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l1281_128144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1281_128169

/-- The hyperbola defined by x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The focus of the hyperbola -/
def focus : ℝ × ℝ := (2, 0)

/-- One of the asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y + x = 0

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

theorem distance_focus_to_asymptote :
  distance_point_to_line focus.1 focus.2 1 1 0 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1281_128169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1281_128170

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m^2 + m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-(2*m + 1))^2 - 4*(m^2 + m)

-- Define the condition for the roots
def root_condition (a b : ℝ) : Prop := (2*a + b) * (a + 2*b) = 20

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x : ℝ, ∃ y z : ℝ, y ≠ z ∧ quadratic_equation m y = 0 ∧ quadratic_equation m z = 0) ∧
  (∀ a b : ℝ, root_condition a b → quadratic_equation m a = 0 → quadratic_equation m b = 0 → m = -2 ∨ m = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1281_128170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_axis_length_asymptotes_equation_l1281_128173

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Statement for the length of the real axis
theorem real_axis_length : 
  ∃ (a : ℝ), a > 0 ∧ (∀ x y, hyperbola x y → x^2 ≤ a^2) ∧ 2*a = 2 :=
sorry

-- Statement for the equation of the asymptotes
theorem asymptotes_equation :
  ∃ (k : ℝ), k > 0 ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x y, 
      hyperbola x y → x > δ → |y - k*x| < ε*|x| ∧ |y + k*x| < ε*|x|) ∧ 
    k^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_axis_length_asymptotes_equation_l1281_128173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1281_128139

/-- The parabola equation x² = 2py where p > 0 -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

/-- The line equation 2x - y - 1 = 0 -/
def line (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- The distance between two points is 4√15 -/
def distance_AB (xA yA xB yB : ℝ) : Prop := 
  (xA - xB)^2 + (yA - yB)^2 = 240

/-- The dot product of two vectors is zero -/
def orthogonal_vectors (x1 y1 x2 y2 : ℝ) : Prop := 
  x1 * x2 + y1 * y2 = 0

/-- The main theorem -/
theorem parabola_intersection_theorem 
  (p : ℝ) (xA yA xB yB xM yM xN yN xF yF : ℝ) :
  parabola p xA yA ∧ parabola p xB yB ∧ 
  parabola p xM yM ∧ parabola p xN yN ∧
  line xA yA ∧ line xB yB ∧
  distance_AB xA yA xB yB ∧
  orthogonal_vectors (xM - xF) (yM - yF) (xN - xF) (yN - yF) →
  p = 2 ∧ 
  ∃ (S : ℝ), S = 12 - 8 * Real.sqrt 2 ∧ 
    ∀ (S' : ℝ), S' ≥ S ∧ 
      S' = 1/2 * abs ((xM - xF) * (yN - yF) - (yM - yF) * (xN - xF)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1281_128139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_less_than_10_8_div_by_36_l1281_128166

def count_squares (m : ℕ) (n : ℕ) : ℕ :=
  (Nat.sqrt m / n)

theorem count_squares_less_than_10_8_div_by_36 :
  count_squares 100000000 36 = 277 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_less_than_10_8_div_by_36_l1281_128166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_children_l1281_128184

-- Define the arithmetic sequence of ages
def age_sequence : ℕ → ℕ → ℝ 
  | n, i => 1.5 + (i - 1)

-- Define the sum of ages
def sum_of_ages (n : ℕ) : ℝ := 
  Finset.sum (Finset.range n) (λ i => age_sequence n i)

-- Theorem statement
theorem number_of_children : ∃ n : ℕ, n > 0 ∧ sum_of_ages n = 12 := by
  -- We know from the solution that n = 4
  let n := 4
  have h1 : n > 0 := by norm_num
  have h2 : sum_of_ages n = 12 := by
    -- The actual proof would go here
    sorry
  exact ⟨n, h1, h2⟩

#check number_of_children

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_children_l1281_128184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l1281_128111

theorem triangle_trigonometric_identity 
  (A B C : ℝ) 
  (h : A + B + C = Real.pi) : 
  Real.sin (A + B/2) + Real.sin (B + C/2) + Real.sin (C + A/2) = 
    -1 + 4 * Real.cos ((A-B)/4) * Real.cos ((B-C)/4) * Real.cos ((C-A)/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l1281_128111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_eq_2_l1281_128143

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_invertible : Function.Injective f

-- Define a and b
def a : ℝ := sorry
def b : ℝ := sorry

-- State the conditions
axiom f_a_eq_b : f a = b
axiom f_b_eq_6 : f b = 6

-- Theorem to prove
theorem a_minus_b_eq_2 : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_eq_2_l1281_128143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_G_l1281_128172

/-- The set G of points with integer coordinates (x, y) where 1 ≤ |x| ≤ 7 and 1 ≤ |y| ≤ 7 -/
def G : Set (ℤ × ℤ) :=
  {p | 1 ≤ |p.1| ∧ |p.1| ≤ 7 ∧ 1 ≤ |p.2| ∧ |p.2| ≤ 7}

/-- A square with vertices in G -/
structure SquareInG where
  vertices : Fin 4 → ℤ × ℤ
  in_G : ∀ i, vertices i ∈ G
  is_square : ∃ s : ℤ, s ≥ 6 ∧ 
    (vertices 1).1 - (vertices 0).1 = s ∧
    (vertices 1).2 - (vertices 0).2 = 0 ∧
    (vertices 2).1 - (vertices 1).1 = 0 ∧
    (vertices 2).2 - (vertices 1).2 = s ∧
    (vertices 3).1 - (vertices 2).1 = -s ∧
    (vertices 3).2 - (vertices 2).2 = 0 ∧
    (vertices 0).1 - (vertices 3).1 = 0 ∧
    (vertices 0).2 - (vertices 3).2 = -s

/-- The number of squares with side length at least 6 and all vertices in set G is 20 -/
theorem count_squares_in_G : ∃ squares : Finset SquareInG, squares.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_G_l1281_128172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_profit_calculation_l1281_128194

/-- Calculates the profit from bread sales given the specified conditions --/
theorem bread_profit_calculation (total_loaves : ℕ) 
  (morning_price afternoon_price_ratio late_afternoon_price production_cost : ℚ) : 
  total_loaves = 60 ∧ 
  morning_price = 3 ∧ 
  afternoon_price_ratio = 3/5 ∧ 
  late_afternoon_price = 3/2 ∧ 
  production_cost = 1 → 
  (let morning_sales := total_loaves / 3
   let afternoon_sales := (total_loaves - morning_sales) / 2
   let late_afternoon_sales := total_loaves - morning_sales - afternoon_sales
   let morning_revenue := morning_sales * morning_price
   let afternoon_revenue := afternoon_sales * (morning_price * afternoon_price_ratio)
   let late_afternoon_revenue := late_afternoon_sales * late_afternoon_price
   let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
   let total_cost := total_loaves * production_cost
   let profit := total_revenue - total_cost
   profit) = 66 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_profit_calculation_l1281_128194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1281_128147

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.sqrt 3 / 2) * Real.sin (ω * x) - (1 / 2) * (Real.sin (ω * x / 2))^2

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem f_properties (ω : ℝ) (h : ω > 0) 
  (h_period : is_smallest_positive_period (f ω) Real.pi) :
  (ω = 2) ∧ 
  (∀ k : ℤ, monotone_increasing_on (f ω) 
    (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (Set.range (f ω ∘ (fun x ↦ x * Real.pi / 2)) = Set.Icc (-1/2) 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1281_128147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1281_128108

/-- Calculates the time (in seconds) for a train to cross a post -/
noncomputable def time_to_cross (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem: The time for a train with length 250.02 meters, traveling at 40 km/hr, to cross a post is approximately 22.502 seconds -/
theorem train_crossing_time :
  ∃ ε > 0, |time_to_cross 250.02 40 - 22.502| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1281_128108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_minimum_value_l1281_128123

/-- The integral function to be minimized -/
noncomputable def I (a b : ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..(1 : ℝ), (Real.sqrt x - (a + b * x))^2

/-- Theorem stating the minimum value of the integral -/
theorem integral_minimum_value :
  ∃ (a b : ℝ), I a b = (1 : ℝ) / 450 ∧ ∀ (x y : ℝ), I x y ≥ (1 : ℝ) / 450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_minimum_value_l1281_128123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_max_area_l1281_128109

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line passing through (1,0)
def line_equation (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ k : ℝ) : ℝ :=
  abs (k * x₀ - y₀ - k) / Real.sqrt (k^2 + 1)

-- Define the area of triangle CPQ
noncomputable def triangle_area (r d : ℝ) : ℝ := 
  Real.sqrt (r^2 - d^2) * d

theorem circle_tangent_and_max_area :
  -- Part 1: Tangent line
  (∃ k : ℝ, ∀ x y : ℝ, line_equation k x y ↔ y = (3/4) * x - 3/4) ∧
  -- Part 2: Maximum area
  (∀ k : ℝ, triangle_area 2 (distance_point_to_line 3 4 k) ≤ 2) ∧
  -- Part 3: Lines achieving maximum area
  (∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧
    triangle_area 2 (distance_point_to_line 3 4 k₁) = 2 ∧
    triangle_area 2 (distance_point_to_line 3 4 k₂) = 2 ∧
    (∀ x y : ℝ, line_equation k₁ x y ↔ y = 7 * x - 7) ∧
    (∀ x y : ℝ, line_equation k₂ x y ↔ y = x - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_max_area_l1281_128109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_implies_vertex_c_coordinates_l1281_128136

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the Euler line
def euler_line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ :=
  let midAB := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)
  let slopeAB := (t.B.2 - t.A.2) / (t.B.1 - t.A.1)
  let perpSlope := -1 / slopeAB
  (midAB.1 + 1, midAB.2 + perpSlope)

-- Theorem statement
theorem euler_line_implies_vertex_c_coordinates :
  ∀ (m n : ℝ),
  let t := Triangle.mk (2, 0) (0, 4) (m, n)
  euler_line (centroid t).1 (centroid t).2 →
  euler_line (circumcenter t).1 (circumcenter t).2 →
  m = -4 ∧ n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_implies_vertex_c_coordinates_l1281_128136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisector_intersection_equilateral_triangle_l1281_128155

/-- Exterior angle of a triangle -/
def ExteriorAngle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
sorry

/-- Predicate to check if a point is the intersection of trisectors -/
def IsTrisectorIntersection (P A B C : EuclideanSpace ℝ (Fin 2)) (angle : ℝ) : Prop :=
sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
sorry

/-- Given a triangle ABC with exterior angles 3α, 3β, and 3γ, where α + β + γ = 120°,
    and trisectors of these exterior angles intersecting the adjacent sides,
    the triangle formed by the intersection points of these trisectors is equilateral. -/
theorem trisector_intersection_equilateral_triangle 
  (A B C : EuclideanSpace ℝ (Fin 2))
  (α β γ : ℝ)
  (h_sum : α + β + γ = 120 * (π / 180))
  (h_ext_A : ExteriorAngle A B C = 3 * α)
  (h_ext_B : ExteriorAngle B C A = 3 * β)
  (h_ext_C : ExteriorAngle C A B = 3 * γ)
  (P : EuclideanSpace ℝ (Fin 2)) -- Intersection of trisectors near A
  (Q : EuclideanSpace ℝ (Fin 2)) -- Intersection of trisectors near B
  (R : EuclideanSpace ℝ (Fin 2)) -- Intersection of trisectors near C
  (h_trisect_P : IsTrisectorIntersection P A B C α)
  (h_trisect_Q : IsTrisectorIntersection Q B C A β)
  (h_trisect_R : IsTrisectorIntersection R C A B γ) :
  IsEquilateral P Q R :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisector_intersection_equilateral_triangle_l1281_128155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1281_128100

-- Define the side lengths of the squares
def square_sides (a₁ : ℕ) : Fin 9 → ℕ
| 0 => a₁
| 1 => a₁
| 2 => 2 * a₁
| 3 => 4 * a₁
| 4 => 7 * a₁
| 5 => 9 * a₁
| 6 => 11 * a₁
| 7 => 19 * a₁
| 8 => 22 * a₁

-- Define the theorem
theorem rectangle_perimeter :
  ∀ (a₁ : ℕ) (h : a₁ = 1),
  let sides := square_sides a₁
  -- Conditions from the problem
  (sides 0 + sides 1 = sides 2) ∧
  (sides 0 + sides 1 + sides 2 = sides 3) ∧
  (sides 1 + sides 2 + sides 3 = sides 4) ∧
  (sides 2 + sides 4 = sides 5) ∧
  (sides 0 + sides 5 + sides 1 = sides 6) ∧
  (sides 4 + sides 0 + sides 6 = sides 7) ∧
  (sides 2 + sides 3 + sides 4 + sides 5 = sides 8) →
  -- Width and height are relatively prime
  Nat.Coprime (sides 8) (sides 7) →
  -- Perimeter is 82
  2 * (sides 8 + sides 7) = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1281_128100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1281_128158

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y = 0

/-- The intersection points of the parabola and circle -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ circle_eq p.1 p.2}

/-- The theorem stating that the distance between the intersection points is 4√5 -/
theorem intersection_distance :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
    p1 ≠ p2 ∧ Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1281_128158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1281_128164

/-- Calculates the speed of a train in km/h given its length, the bridge length it passes, and the time it takes to pass the bridge. -/
noncomputable def train_speed (train_length bridge_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  speed_ms * 3.6

/-- Theorem stating that a train of length 460 meters passing a bridge of length 140 meters in 48 seconds has a speed of 45 km/h. -/
theorem train_speed_theorem :
  train_speed 460 140 48 = 45 := by
  unfold train_speed
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1281_128164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_score_proof_l1281_128103

theorem highest_score_proof (total_innings : ℕ) (overall_average : ℚ) 
  (score_difference : ℕ) (average_excluding_extremes : ℚ) 
  (h1 : total_innings = 46)
  (h2 : overall_average = 60)
  (h3 : score_difference = 180)
  (h4 : average_excluding_extremes = 58)
  : ∃ (highest_score : ℕ), highest_score = 194 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_score_proof_l1281_128103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_p_pow_p_plus_one_l1281_128177

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that calculates the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log n 10 + 1

/-- The main theorem -/
theorem prime_p_pow_p_plus_one :
  ∀ p : ℕ, p ≥ 2 →
    (isPrime (p^p + 1) ∧ numDigits (p^p + 1) ≤ 19) ↔ p = 2 ∨ p = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_p_pow_p_plus_one_l1281_128177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l1281_128104

/-- Polar coordinate equation of a circle with center (1,1) and radius √2 -/
theorem circle_polar_equation (ρ θ : ℝ) : 
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ 
  ρ = 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l1281_128104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_roots_condition_l1281_128160

/-- The function f(x) defined as x^2 + a*x + b*cos(x) -/
noncomputable def f (a b x : ℝ) : ℝ := x^2 + a*x + b*(Real.cos x)

/-- Theorem stating the conditions for f(x) = 0 and f(f(x)) = 0 to have identical non-empty sets of real roots -/
theorem identical_roots_condition (a b : ℝ) :
  (∃ x : ℝ, f a b x = 0) ∧ 
  (∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) ↔ 
  b = 0 ∧ 0 ≤ a ∧ a < 4 := by
  sorry

#check identical_roots_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_roots_condition_l1281_128160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_two_identical_l1281_128116

def roll_seven_dice : ℕ := 7
def die_faces : ℕ := 6

-- Total number of possible outcomes
def total_outcomes : ℕ := die_faces ^ roll_seven_dice

-- Number of ways to choose 2 dice from 7
def choose_pair : ℕ := Nat.choose roll_seven_dice 2

-- Number of ways to arrange 5 distinct numbers
def arrange_distinct : ℕ := Nat.factorial 5

-- Total favorable outcomes
def favorable_outcomes : ℕ := choose_pair * die_faces * arrange_distinct

theorem probability_at_most_two_identical : 
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_two_identical_l1281_128116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_properties_l1281_128140

/-- The hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The foci of the hyperbola -/
noncomputable def foci (h : Hyperbola) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  ((-c, 0), (c, 0))

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  1/2 * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

/-- Main theorem -/
theorem hyperbola_point_properties (h : Hyperbola) (p : PointOnHyperbola h) :
  let (f₁, f₂) := foci h
  let p_coords := (p.x, p.y)
  triangleArea p_coords f₁ f₂ = 20 →
  (distance p_coords f₁ + distance p_coords f₂ = 50/3 ∧ abs p.y = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_properties_l1281_128140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_φ_is_even_u_is_neither_y_is_odd_l1281_128118

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.sin (2 * x)
noncomputable def φ (x : ℝ) : ℝ := 4 - 2 * x^4 + Real.sin x ^ 2
def u (x : ℝ) : ℝ := x^3 + 2 * x - 1
noncomputable def y (a k x : ℝ) : ℝ := (1 + a^(k * x)) / (1 - a^(k * x))

-- Define even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorems to prove
theorem f_is_odd : is_odd f := by sorry

theorem φ_is_even : is_even φ := by sorry

theorem u_is_neither : ¬(is_even u) ∧ ¬(is_odd u) := by sorry

theorem y_is_odd : ∀ a k, is_odd (y a k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_φ_is_even_u_is_neither_y_is_odd_l1281_128118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_2023_divisors_theorem_l1281_128186

def is_least_positive_integer_with_2023_divisors (x : ℕ) : Prop :=
  (∀ y : ℕ, y < x → (Finset.filter (λ d ↦ d ∣ y) (Finset.range y.succ)).card ≠ 2023) ∧
  (Finset.filter (λ d ↦ d ∣ x) (Finset.range x.succ)).card = 2023

theorem least_positive_integer_with_2023_divisors_theorem :
  ∃ (n j : ℕ),
    is_least_positive_integer_with_2023_divisors (n * 6^j) ∧
    ¬(6 ∣ n) ∧
    n + j = 60466182 :=
by
  sorry

#check least_positive_integer_with_2023_divisors_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_2023_divisors_theorem_l1281_128186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_allocation_l1281_128163

def bonus : ℚ := 1496

def kitchen_fraction : ℚ := 1/22
def renovation_fraction : ℚ := 2/5
def holiday_fraction : ℚ := 1/4
def charity_fraction : ℚ := 1/6
def gift_fraction : ℚ := 1/8

def family_members : ℕ := 6

def remaining_amount : ℚ := 
  bonus - (kitchen_fraction * bonus + 
           renovation_fraction * bonus + 
           holiday_fraction * bonus + 
           charity_fraction * bonus + 
           gift_fraction * bonus)

theorem bonus_allocation :
  abs (remaining_amount - 19.27) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_allocation_l1281_128163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_A_B_l1281_128176

def A : ℕ → ℤ
  | 0 => 39
  | n + 1 => (2 * n + 1) * (2 * n + 2) + A n

def B : ℕ → ℤ
  | 0 => 1
  | n + 1 => (2 * n) * (2 * n + 1) + B n

theorem positive_difference_A_B : |A 19 - B 19| = 722 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_A_B_l1281_128176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1281_128197

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focus
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 3

-- Define the asymptotes
def asymptotes (a b : ℝ) (x y : ℝ) : Prop :=
  (b * x + a * y = 0) ∨ (b * x - a * y = 0)

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y, focus x y) →
  (∃ x y, asymptotes a b x y ∧ circle_eq x y) →
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 3 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1281_128197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1281_128182

/-- The equation of a line with inclination angle 135° and y-intercept -1 -/
theorem line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), m = Real.tan (135 * π / 180) ∧ b = -1 ∧ y = m * x + b) ↔ 
  x + y + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1281_128182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_theorem_l1281_128146

/-- The distance from the origin to the midpoint of a line segment -/
noncomputable def midpoint_distance (a c m k : ℝ) : ℝ :=
  (((1 + m^2) * (a + c)^2 + 4 * k^2 + 4 * m * (a + c) * k).sqrt) / 2

/-- Theorem: The distance from the origin to the midpoint of a line segment
    on a line y = mx + k, given by two x-coordinates a and c, is equal to
    the formula for midpoint_distance. -/
theorem midpoint_distance_theorem (a b c d m k : ℝ) :
  b = m * a + k →
  d = m * c + k →
  let x_m := (a + c) / 2
  let y_m := (b + d) / 2
  (x_m^2 + y_m^2).sqrt = midpoint_distance a c m k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_theorem_l1281_128146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_direction_b_l1281_128141

noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b : ℝ × ℝ := (-4, 7)

noncomputable def projection (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / Real.sqrt (w.1^2 + w.2^2)

theorem projection_c_direction_b : 
  let c : ℝ × ℝ := (-a.1, -a.2)
  projection c b = -Real.sqrt 65 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_direction_b_l1281_128141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_minuend_l1281_128154

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def digits_used_once (a b c d e f g h : ℕ) : Prop :=
  (Finset.card {2, a, 1, b, c, 3, d, e, 0, f, g, h} = 10) ∧
  (∀ x ∈ Finset.range 10, is_valid_digit x)

theorem unique_minuend : 
  ∃! (a b c d e f g h : ℕ),
    digits_used_once a b c d e f g h ∧
    2196 - (300 + 10 * d + e) * (f + g / 10 + h / 100) = 2016 :=
sorry

#check unique_minuend

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_minuend_l1281_128154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_a_minus_b_l1281_128137

/-- The number of pieces of fruit in Bucket A -/
def bucketA : ℕ := sorry

/-- The number of pieces of fruit in Bucket B -/
def bucketB : ℕ := sorry

/-- The number of pieces of fruit in Bucket C -/
def bucketC : ℕ := sorry

/-- Bucket C has 9 pieces of fruit -/
axiom c_count : bucketC = 9

/-- Bucket B has 3 more pieces of fruit than Bucket C -/
axiom b_more_than_c : bucketB = bucketC + 3

/-- Bucket A has more fruit than Bucket B -/
axiom a_more_than_b : bucketA > bucketB

/-- Total fruit in all 3 buckets is 37 -/
axiom total_fruit : bucketA + bucketB + bucketC = 37

/-- Theorem: Bucket A has 4 more pieces of fruit than Bucket B -/
theorem bucket_a_minus_b : bucketA - bucketB = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_a_minus_b_l1281_128137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sum_positive_l1281_128185

noncomputable def calculator_operation (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (Real.sqrt a, b ^ 2, (c ^ 2) ^ 2)

noncomputable def iterate_operation (n : ℕ) (initial : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match n with
  | 0 => initial
  | n + 1 => calculator_operation (iterate_operation n initial).1 (iterate_operation n initial).2.1 (iterate_operation n initial).2.2

theorem final_sum_positive :
  let initial := (2, -2, 0)
  let final := iterate_operation 50 initial
  final.1 + final.2.1 + final.2.2 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sum_positive_l1281_128185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_calculation_l1281_128138

theorem triangle_area_calculation (OA OB : ℝ) (angle_BAC : ℝ) 
  (h_OA : OA = 3) 
  (h_OB : OB = 4) 
  (h_angle : angle_BAC = Real.pi / 4) : 
  ∃ (area : ℝ), area = (5 * Real.sqrt 17) / 2 := by
  sorry  -- Proof to be filled in

#check triangle_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_calculation_l1281_128138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1281_128122

noncomputable section

/-- Floor function -/
def floor (x : ℝ) : ℤ := Int.floor x

/-- Fractional part of a real number -/
def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

/-- Sequence definition -/
def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => (floor (a n) : ℝ) + 1 / frac (a n)

/-- Main theorem -/
theorem a_2019_value : a 2019 = 3029 + (Real.sqrt 3 - 1) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1281_128122
