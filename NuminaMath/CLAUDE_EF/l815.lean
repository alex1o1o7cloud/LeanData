import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_ln5_over_32_l815_81535

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ :=
  (4 * Real.sqrt (2 - x) - Real.sqrt (3 * x + 2)) / 
  ((Real.sqrt (3 * x + 2) + 4 * Real.sqrt (2 - x)) * (3 * x + 2)^2)

theorem integral_f_equals_ln5_over_32 :
  ∫ x in Set.Icc 0 2, f x = (1/32) * Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_ln5_over_32_l815_81535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l815_81515

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  ∃ (p : ℝ), 
    (∀ x, f (x + p) = f x) ∧ 
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
    (∀ x, -Real.pi / 6 < x → x < Real.pi / 6 → f x = 5 / 3 → 
      Real.sin (2 * x) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l815_81515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_A_and_B_y_value_when_A_minus_3B_constant_l815_81541

-- Define polynomials A and B
variable (x y : ℝ)
def B (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y + x + 2
def A (x y : ℝ) : ℝ := B x y + (6 * x^2 * y + 4 * x * y - 2 * x - 1)

-- Theorem 1: A + B = 12x^2y + 3
theorem sum_of_A_and_B (x y : ℝ) : A x y + B x y = 12 * x^2 * y + 3 := by
  sorry

-- Theorem 2: When A - 3B is constant for any x, y = 1/2
theorem y_value_when_A_minus_3B_constant (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, A x y - 3 * B x y = c) → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_A_and_B_y_value_when_A_minus_3B_constant_l815_81541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_on_dam_l815_81506

/-- Represents a trapezoidal dam -/
structure TrapezoidalDam where
  upperBase : ℝ
  lowerBase : ℝ
  height : ℝ

/-- Calculates the force of water pressure on a trapezoidal dam -/
noncomputable def waterPressureForce (dam : TrapezoidalDam) (waterDensityGravity : ℝ) : ℝ :=
  waterDensityGravity * (dam.upperBase + 2 * dam.lowerBase) * dam.height^2 / 6

/-- Theorem stating that the force of water pressure on the given trapezoidal dam is 735750 N -/
theorem water_pressure_force_on_dam :
  let dam : TrapezoidalDam := { upperBase := 10, lowerBase := 20, height := 3 }
  let waterDensityGravity : ℝ := 9810
  waterPressureForce dam waterDensityGravity = 735750 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_on_dam_l815_81506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l815_81536

/-- The length of the arc of the curve ρ = 2 sin φ from φ = 0 to φ = π/6 -/
noncomputable def arcLength : ℝ := Real.pi / 3

/-- The polar equation of the curve -/
noncomputable def ρ (φ : ℝ) : ℝ := 2 * Real.sin φ

/-- The range of φ -/
def φ_range : Set ℝ := { φ | 0 ≤ φ ∧ φ ≤ Real.pi / 6 }

/-- Theorem stating that the arc length of the curve is π/3 -/
theorem arc_length_of_curve :
  ∫ φ in φ_range, Real.sqrt ((ρ φ)^2 + ((2 * Real.cos φ)^2)) = arcLength := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l815_81536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l815_81527

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateralTriangleArea (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The ratio of areas as described in the problem -/
noncomputable def areaRatio : ℝ :=
  let largeTriangleArea := equilateralTriangleArea 12
  let smallTriangleArea := equilateralTriangleArea 6
  let remainingArea := largeTriangleArea - smallTriangleArea
  smallTriangleArea / remainingArea

theorem area_ratio_is_one_third :
  areaRatio = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l815_81527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_128_l815_81513

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (right_angle_ABC : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)
variable (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3)
variable (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4)

variable (right_angle_ABD : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
variable (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 12)

-- C and D on opposite sides of AB
variable (opposite_sides : (C.2 - A.2) * (B.1 - A.1) - (C.1 - A.1) * (B.2 - A.2) *
                           (D.2 - A.2) * (B.1 - A.1) - (D.1 - A.1) * (B.2 - A.2) < 0)

-- DE parallel to AC
variable (DE_parallel_AC : (E.2 - D.2) * (C.1 - A.1) = (E.1 - D.1) * (C.2 - A.2))

-- DE/DB = m/n
variable (m n : ℕ)
variable (m_n_coprime : Nat.Coprime m n)
variable (DE_DB_ratio : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / 
                        Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = m / n)

-- Theorem to prove
theorem sum_m_n_equals_128 : m + n = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_128_l815_81513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_l815_81561

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 2

-- Part 1: Minimum value of f on [t, t+1]
theorem min_value_f (t : ℝ) (h : t > 0) :
  (∃ (x : ℝ), x ∈ Set.Icc t (t + 1) ∧
    ∀ (y : ℝ), y ∈ Set.Icc t (t + 1) → f x ≤ f y) ∧
  (t = 1 → ∃ (x : ℝ), x ∈ Set.Icc t (t + 1) ∧ f x = 0) ∧
  (t > 1 → ∃ (x : ℝ), x ∈ Set.Icc t (t + 1) ∧ f x = t * Real.log t) :=
by sorry

-- Part 2: Range of a for which f(x₀) ≥ g(x₀) for some x₀ ∈ [1, e]
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x₀ : ℝ), x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f x₀ ≥ g a x₀) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_l815_81561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_weight_l815_81559

/-- Represents a person with a weight in pounds -/
structure Person where
  weight : ℚ
  deriving Repr

theorem jakes_weight (jake sister : Person) 
  (h1 : jake.weight - 15 = 2 * sister.weight) 
  (h2 : jake.weight + sister.weight = 132) : 
  jake.weight = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_weight_l815_81559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l815_81550

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l * 1.2 * w - l * w) / (l * w) = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l815_81550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l815_81552

def IsTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b c : ℝ) : 
  a = 4 → b = 7 → (0 < a ∧ 0 < b ∧ 0 < c) → IsTriangle a b c → 3 < c ∧ c < 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l815_81552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l815_81567

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = 4*x

/-- Point type representing a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Fixed point P(3,1) -/
def P : Point := ⟨3, 1⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the minimum value of |MP| + |MF| -/
theorem min_distance_sum (para : Parabola) :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (M : Point), para.eq M.x M.y →
    distance M P + distance M focus ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l815_81567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_one_l815_81544

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - 2 * x - a

-- State the theorem
theorem max_value_implies_a_equals_one :
  (∃ (a : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → f a x ≤ -1) ∧
  (∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 Real.pi ∧ f 1 x₀ = -1) →
  (∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → f a x ≤ -1) →
              (∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 Real.pi ∧ f a x₀ = -1) →
              a = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_one_l815_81544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_problem_l815_81512

theorem sine_difference_problem (θ : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : Real.sin θ = 2 * Real.sqrt 5 / 5) : 
  Real.sin (θ - π/4) = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_problem_l815_81512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_change_l815_81588

theorem garden_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  (L * 1.6) * (W * 0.7) = 1.12 * (L * W) := by
  -- Expand the left side of the equation
  calc (L * 1.6) * (W * 0.7)
    = L * W * (1.6 * 0.7) := by ring
  -- Simplify the right side
  _ = L * W * 1.12 := by norm_num
  -- The equality is now proven
  _ = 1.12 * (L * W) := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_change_l815_81588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_on_circle_l815_81573

-- Define the circle Ω
variable (Ω : Set (ℝ × ℝ))

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define that ABCD is inscribed in Ω
variable (inscribed : A ∈ Ω ∧ B ∈ Ω ∧ C ∈ Ω ∧ D ∈ Ω)

-- Define that ABCD has no parallel sides
variable (no_parallel : ¬∃ (m₁ m₂ : ℝ), (A.2 - B.2 = m₁ * (A.1 - B.1) ∧ C.2 - D.2 = m₁ * (C.1 - D.1)) ∨
                                       (A.2 - D.2 = m₂ * (A.1 - D.1) ∧ B.2 - C.2 = m₂ * (B.1 - C.1)))

-- Define O as the intersection of AB and CD
noncomputable def O : ℝ × ℝ := sorry

-- Define the set of all points X that are tangency points of circles containing AB and CD
def X : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem tangency_points_on_circle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center = O) ∧
    (radius = Real.sqrt ((O.1 - A.1) * (O.1 - B.1) + (O.2 - A.2) * (O.2 - B.2))) ∧
    (∀ x ∈ X, (x.1 - center.1)^2 + (x.2 - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_on_circle_l815_81573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l815_81505

/-- The function f(x) = (m+1)x^2 - mx + m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

theorem function_properties (m : ℝ) :
  (∀ x, f m x ≥ 0) ∧
  (∀ x ∈ Set.Icc (-1) 1, f m x ≥ 0) ∧
  (m ≥ 2 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l815_81505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_volume_60_litres_l815_81553

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The ratio of milk to water in a mixture -/
def ratio (m : Mixture) : ℚ := m.milk / m.water

theorem initial_volume_60_litres
  (initial : Mixture)
  (final : Mixture)
  (h1 : ratio initial = 2)
  (h2 : ratio final = 1/2)
  (h3 : final.water = initial.water + 60)
  (h4 : final.milk = initial.milk) :
  initial.milk + initial.water = 60 := by
  sorry

#eval ratio { milk := 40, water := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_volume_60_litres_l815_81553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l815_81574

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the point of tangency
def p : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let m := deriv f p.fst  -- Slope of the tangent line
  let b := p.snd - m * p.fst  -- y-intercept of the tangent line
  (λ x : ℝ ↦ m * x + b) = (λ x : ℝ ↦ 0) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l815_81574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_f_converse_l815_81569

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

-- Theorem statement
theorem range_of_f :
  ∀ y ∈ Set.range f, 0 < y ∧ y ≤ 1 :=
by sorry

-- Theorem statement for the converse (to establish equality with (0, 1])
theorem range_of_f_converse :
  ∀ y, 0 < y ∧ y ≤ 1 → ∃ x, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_f_converse_l815_81569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_palindromic_binary_year_before_2015_l815_81581

def to_binary : ℕ → List Bool :=
  sorry

def is_palindrome {α : Type*} : List α → Prop :=
  sorry

theorem last_palindromic_binary_year_before_2015 : 
  ∀ n : ℕ, n < 2015 → is_palindrome (to_binary n) → n ≤ 1967 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_palindromic_binary_year_before_2015_l815_81581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_for_729_l815_81537

/-- The length of a side of a cube given its volume -/
noncomputable def cube_side_length (volume : ℝ) : ℝ := Real.rpow volume (1/3)

/-- Theorem: For a cube with volume 729 cm³, the length of one side is 9 cm -/
theorem cube_side_length_for_729 : cube_side_length 729 = 9 := by
  -- Unfold the definition of cube_side_length
  unfold cube_side_length
  -- Simplify the expression
  simp
  -- The proof is complete, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_for_729_l815_81537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l815_81597

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ :=
  width * height

theorem triangle_fraction_of_grid :
  let triangle_area := triangleArea 1 3 5 1 4 4
  let grid_area := rectangleArea 6 5
  triangle_area / grid_area = 1/6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l815_81597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_calculation_l815_81549

theorem exponent_calculation (c d : ℝ) (h1 : (90 : ℝ)^c = 4) (h2 : (90 : ℝ)^d = 7) :
  (18 : ℝ)^((1 - c - d) / (2 * (1 - d))) = 45 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_calculation_l815_81549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_collinearity_l815_81594

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- Collinearity condition for three points -/
def isCollinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂) = 0)

/-- The theorem statement -/
theorem polynomial_collinearity (P : RealPolynomial) :
  (∀ x y z : ℝ, x + y + z = 0 →
    isCollinear (x, P x) (y, P y) (z, P z)) →
  ∃ a b c : ℝ, ∀ x : ℝ, P x = a * x^3 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_collinearity_l815_81594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BJ_l815_81555

-- Define the equilateral triangle and its points
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ‖B - A‖ = 23 ∧ ‖C - B‖ = 23 ∧ ‖A - C‖ = 23

-- Define points D, E, F
def PointOnSide (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • Q + (1 - t) • R

-- Define midpoints
def Midpoint (M P Q : ℝ × ℝ) : Prop :=
  M = (P + Q) / 2

-- Main theorem
theorem length_of_BJ (A B C D E F G H J : ℝ × ℝ) :
  Triangle A B C →
  PointOnSide D A B →
  PointOnSide E A B →
  PointOnSide F B C →
  Midpoint G A D →
  Midpoint H D E →
  Midpoint J E B →
  ‖G - A‖ = 2 →
  ‖F - G‖ = 13 →
  ‖J - H‖ = 7 →
  ‖C - F‖ = 1 →
  ‖B - J‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BJ_l815_81555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetric_to_A_l815_81509

/-- Two points in polar coordinates are symmetric with respect to the polar axis if their radial distances are equal and the sum of their polar angles is a multiple of 2π. -/
def symmetric_polar (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : Prop :=
  r1 = r2 ∧ ∃ k : ℤ, θ1 + θ2 = 2 * k * Real.pi

/-- The point A in polar coordinates -/
noncomputable def point_A : ℝ × ℝ := (3, -Real.pi/3)

/-- The point D in polar coordinates -/
noncomputable def point_D : ℝ × ℝ := (3, 5*Real.pi/6)

theorem not_symmetric_to_A : ¬(symmetric_polar point_A.1 point_A.2 point_D.1 point_D.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetric_to_A_l815_81509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_intersection_convex_polygon_l815_81554

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a Γ-shaped letter -/
structure GammaLetter where
  shortEnd : Point
  longPoints : List Point

/-- Represents a line in 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two identical Γ-shaped letters and their intersection points, 
    the intersection points form a convex polygon -/
theorem gamma_intersection_convex_polygon 
  (n : ℕ) 
  (gamma1 gamma2 : GammaLetter) 
  (h_identical : gamma1 = gamma2) 
  (h_n_points : gamma1.longPoints.length = n - 1 ∧ gamma2.longPoints.length = n - 1)
  (intersection_points : List Point) 
  (h_intersections : ∀ i, i ∈ Finset.range (n - 1) → 
    ∃ X, X ∈ intersection_points ∧ 
        (∃ l1 l2 : Line, 
          (l1.a * gamma1.shortEnd.x + l1.b * gamma1.shortEnd.y + l1.c = 0 ∧
           l1.a * (gamma1.longPoints.nthLe i sorry).x + l1.b * (gamma1.longPoints.nthLe i sorry).y + l1.c = 0) ∧
          (l2.a * gamma2.shortEnd.x + l2.b * gamma2.shortEnd.y + l2.c = 0 ∧
           l2.a * (gamma2.longPoints.nthLe i sorry).x + l2.b * (gamma2.longPoints.nthLe i sorry).y + l2.c = 0) ∧
          (l1.a * X.x + l1.b * X.y + l1.c = 0 ∧ l2.a * X.x + l2.b * X.y + l2.c = 0))) :
  -- The conclusion states that the intersection points form a convex polygon
  -- We'll use a placeholder definition for ConvexPolygon
  ∃ (vertices : List Point), vertices = intersection_points ∧ vertices.length ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_intersection_convex_polygon_l815_81554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_of_five_integers_l815_81507

theorem max_average_of_five_integers (a b c d e : ℕ) : 
  (∃ (min max : ℕ), {a, b, c, d, e} ⊆ Finset.Icc min max ∧ max - min = 10 ∧ max ≤ 53) →
  (a + b + c + d + e) / 5 ≤ 51 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_of_five_integers_l815_81507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_b_l815_81530

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def num_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem least_possible_b (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧
  num_factors a = 3 ∧
  num_factors b = a ∧
  b % a = 0 ∧
  ∃ p : ℕ, is_prime p ∧ p > 2 ∧ a = p^2 ∧ 
  (∀ q : ℕ, is_prime q ∧ q > 2 → p ≤ q) →
  b ≥ 256 ∧ (∀ c : ℕ, c < 256 → ¬(num_factors c = a ∧ c % a = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_b_l815_81530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_participants_l815_81580

theorem tournament_participants (n : ℕ) (m : ℕ) : 
  198 < n ∧ n < 230 ∧ 
  (n - 2 * m) ^ 2 = n ∧
  (∀ k, k = m ∨ k = n - m → k ≥ 105) →
  n - m ≥ 105 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_participants_l815_81580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_is_two_units_left_l815_81546

/-- A parabola is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The horizontal shift of a parabola -/
noncomputable def horizontal_shift (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- The original parabola y = x^2 + 1 -/
def original : Parabola := ⟨1, 0, 1, by norm_num⟩

/-- The translated parabola y = x^2 + 4x + 5 -/
def translated : Parabola := ⟨1, 4, 5, by norm_num⟩

theorem translation_is_two_units_left : 
  horizontal_shift original - horizontal_shift translated = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_is_two_units_left_l815_81546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_phi_l815_81534

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ) - Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

theorem even_function_phi (φ : ℝ) : 
  (∀ x, f x φ = f (-x) φ) → ∃ k : ℤ, φ = -π/3 + k * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_phi_l815_81534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_period_l815_81520

/-- Given a function y = A sin(Bx + C) + D + E cos(Fx + G) + H, 
    where the graph covers six periods from 0 to 2π, 
    A = 3, E = 2, F = B, and A ≠ E, prove that B = 6 -/
theorem sinusoidal_function_period (A B C D E F G H : ℝ) : 
  A = 3 → E = 2 → F = B → A ≠ E → 
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 * Real.pi → 
    ∃ k : ℕ, k = 6 ∧ 
    (λ x ↦ A * Real.sin (B * x + C) + D + E * Real.cos (F * x + G) + H) x = 
    (λ x ↦ A * Real.sin (B * x + C) + D + E * Real.cos (F * x + G) + H) (x + 2 * Real.pi / k)) →
  B = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_period_l815_81520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l815_81599

/-- The eccentricity of a hyperbola with equation x^2 - y^2/m = 1 -/
noncomputable def hyperbola_eccentricity (m : ℝ) : ℝ := Real.sqrt (1 + m)

theorem hyperbola_m_value (m : ℝ) (h1 : m > 0) (h2 : hyperbola_eccentricity m = Real.sqrt 3) : 
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l815_81599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_for_regular_pyramid_l815_81551

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Triangle where
  a : Point
  b : Point
  c : Point

structure Pyramid where
  base : Set Point
  lateral_faces : Set Triangle
  vertex : Point

-- Define the necessary predicates
def is_square (s : Set Point) : Prop := sorry

def is_lateral_edge (p : Pyramid) (e : Point × Point) : Prop := sorry

def lateral_edges (p : Pyramid) : Set (Point × Point) := sorry

def all_equal (s : Set (Point × Point)) : Prop := sorry

def is_equilateral (t : Triangle) : Prop := sorry

-- Define the main concepts
def is_regular_pyramid (p : Pyramid) : Prop :=
  is_square p.base ∧ 
  ∀ e : Point × Point, is_lateral_edge p e → all_equal (lateral_edges p)

def all_lateral_faces_equilateral (p : Pyramid) : Prop :=
  ∀ f ∈ p.lateral_faces, is_equilateral f

-- State the theorem
theorem sufficient_not_necessary_condition_for_regular_pyramid :
  (∀ p : Pyramid, all_lateral_faces_equilateral p → is_regular_pyramid p) ∧
  (∃ p : Pyramid, is_regular_pyramid p ∧ ¬all_lateral_faces_equilateral p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_for_regular_pyramid_l815_81551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_doubling_growth_rate_l815_81585

/-- The average annual growth rate that doubles a value over 10 years -/
noncomputable def average_annual_growth_rate : ℝ :=
  (2 ^ (1/10) - 1) * 100

/-- Assertion that the average annual growth rate is approximately 7.1773% -/
theorem gdp_doubling_growth_rate :
  abs (average_annual_growth_rate - 7.1773) < 0.00005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_doubling_growth_rate_l815_81585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cutOffPyramidVolumeTheorem_l815_81598

/-- Represents a right square pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  height : ℝ

/-- Calculates the volume of a cut-off portion of a right square pyramid -/
noncomputable def cutOffPyramidVolume (pyramid : RightSquarePyramid) (cutHeight : ℝ) : ℝ :=
  let remainingHeight := pyramid.height - cutHeight
  let newBaseEdge := (remainingHeight / pyramid.height) * pyramid.baseEdge
  (1 / 3) * (newBaseEdge ^ 2) * remainingHeight

/-- Theorem stating the volume of the cut-off pyramid -/
theorem cutOffPyramidVolumeTheorem (pyramid : RightSquarePyramid) 
    (h1 : pyramid.baseEdge = 10)
    (h2 : pyramid.height = 12)
    (h3 : 0 < pyramid.baseEdge)
    (h4 : 0 < pyramid.height)
    (cutHeight : ℝ)
    (h5 : cutHeight = 4)
    (h6 : 0 < cutHeight)
    (h7 : cutHeight < pyramid.height) :
  cutOffPyramidVolume pyramid cutHeight = 3200 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cutOffPyramidVolumeTheorem_l815_81598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_play_area_specific_l815_81526

/-- The area accessible to a llama tied to the corner of a rectangular shed --/
noncomputable def llamaPlayArea (shedLength shedWidth leashLength : ℝ) : ℝ :=
  let fullCircleArea := Real.pi * leashLength ^ 2
  let extendedArea := Real.pi * (leashLength - shedLength) ^ 2 / 4
  (3 * fullCircleArea / 4) + extendedArea

/-- Theorem stating the area a llama can access when tied to a 3m by 4m shed with a 4m leash --/
theorem llama_play_area_specific : llamaPlayArea 4 3 4 = 12.25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_play_area_specific_l815_81526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_integer_l815_81560

theorem round_to_nearest_integer (x : ℝ) (h : x = 5738291.4982) :
  round x = 5738291 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_integer_l815_81560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_n_l815_81584

/-- A function that checks if a polynomial x^2 + x - n can be factored into two linear factors with integer coefficients -/
def is_factorable (n : ℤ) : Prop :=
  ∃ a b : ℤ, ∀ x : ℤ, x^2 + x - n = (x - a) * (x - b)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^2

/-- The main theorem stating that there are exactly 2 integers n satisfying all conditions -/
theorem exactly_two_valid_n : 
  (∃! (s : Finset ℤ), 
    (∀ n ∈ s, 1 ≤ n ∧ n ≤ 2000 ∧ is_factorable n ∧ is_perfect_square n) ∧ 
    s.card = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_n_l815_81584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_and_min_g_l815_81522

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2*x^2 + 4/x

-- Theorem statement
theorem min_f_and_min_g :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ m = 4) ∧
  (∀ (x : ℝ), x > 0 → g x ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_and_min_g_l815_81522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2023_l815_81593

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + (-x) = 0) → 
  -2023 = -2023 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2023_l815_81593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_problem_l815_81595

theorem certain_number_problem (a : ℝ) :
  a^(-(1/4 : ℝ)) + 25^(-(1/2 : ℝ)) + 5^(-1 : ℝ) = 11 → a = (5/53)^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_problem_l815_81595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_37_equals_2_l815_81508

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem using a more general approach
theorem inverse_f_at_37_equals_2 : ∃ y : ℝ, f y = 37 ∧ y = 2 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_37_equals_2_l815_81508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l815_81548

noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem loan_difference_theorem (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) 
  (compound_frequency : ℝ) (total_time : ℝ) (partial_time : ℝ) (partial_payment_ratio : ℝ) :
  principal = 20000 →
  compound_rate = 0.08 →
  simple_rate = 0.09 →
  compound_frequency = 2 →
  total_time = 15 →
  partial_time = 10 →
  partial_payment_ratio = 1/3 →
  let partial_amount := compoundInterest principal compound_rate compound_frequency partial_time
  let partial_payment := partial_amount * partial_payment_ratio
  let remaining_principal := partial_amount - partial_payment
  let final_compound_amount := compoundInterest remaining_principal compound_rate compound_frequency (total_time - partial_time)
  let total_compound_payment := partial_payment + final_compound_amount
  let total_simple_payment := simpleInterest principal simple_rate total_time
  let difference := total_compound_payment - total_simple_payment
  ⌊difference⌋₊ = 11520 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l815_81548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_matrix_elements_l815_81590

theorem max_sum_matrix_elements (a b c : ℤ) : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := ![![-5/7, a/7], ![b/7, c/7]]
  (A ^ 2 = 1) → (∀ x y z : ℤ, x + y + z ≤ 30 ∨ 
    ¬(∃ B : Matrix (Fin 2) (Fin 2) ℚ, B = ![![-5/7, (x:ℚ)/7], ![(y:ℚ)/7, (z:ℚ)/7]] ∧ B ^ 2 = 1)) :=
by
  intro hA
  intro x y z
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_matrix_elements_l815_81590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_candies_remaining_l815_81545

/-- Represents the number of candies of each color initially in the bowl -/
def n : ℕ := 100  -- We use a concrete value for simplicity

/-- Represents the total number of candies after each step -/
def remaining_candies : ℕ → ℕ
| 0 => 6 * n  -- Initial state
| 1 => 5 * n  -- After eating all green
| 2 => (9 * n) / 2  -- After eating half of orange
| 3 => (23 * n) / 6  -- After eating two-thirds of purple
| 4 => (11 * n) / 6  -- After eating half of remaining except purple
| _ => (32 * n) / 100  -- Final state (32% of original)

/-- Represents the number of red candies after each step -/
def red_candies : ℕ → ℕ
| 0 => n  -- Initial state
| 1 => n  -- After eating all green
| 2 => n  -- After eating half of orange
| 3 => n  -- After eating two-thirds of purple
| 4 => n / 2  -- After eating half of remaining except purple
| _ => n / 2  -- Final state (no more eaten)

/-- The main theorem stating that 50% of red candies remain -/
theorem red_candies_remaining :
  red_candies 5 * 100 / red_candies 0 = 50 := by
  -- Proof goes here
  sorry

#eval red_candies 5 * 100 / red_candies 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_candies_remaining_l815_81545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l815_81524

open Real

theorem expression_simplification (α : ℝ) :
  (π/2 < α ∧ α < π) ∨ (3*π/2 < α ∧ α < 2*π) →
  let expr := sqrt ((1 + sin α) / (1 - sin α)) - sqrt ((1 - sin α) / (1 + sin α))
  ((π/2 < α ∧ α < π) → expr = -2 * tan α) ∧
  ((3*π/2 < α ∧ α < 2*π) → expr = 2 * tan α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l815_81524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_a_bounded_l815_81578

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x^2

-- State the theorem
theorem f_nonpositive_implies_a_bounded (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ 0) → a ≤ 2 * exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_a_bounded_l815_81578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l815_81591

/-- Represents a 6x6 table of integers -/
def Table := Fin 6 → Fin 6 → ℤ

/-- Checks if all elements in the table are unique -/
def all_unique (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

/-- Checks if the sum of a 1x5 rectangle in the table is 2022 or 2023 -/
def valid_sum (t : Table) (row col : Fin 6) (horizontal : Bool) : Prop :=
  let sum := if horizontal then
               (Finset.range 5).sum (λ i ↦ t row ⟨(col + i : ℕ) % 6, by sorry⟩)
             else
               (Finset.range 5).sum (λ i ↦ t ⟨(row + i : ℕ) % 6, by sorry⟩ col)
  sum = 2022 ∨ sum = 2023

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement : ¬ ∃ (t : Table), 
  all_unique t ∧ 
  (∀ row col, valid_sum t row col true) ∧ 
  (∀ row col, valid_sum t row col false) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l815_81591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l815_81577

theorem sin_x_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.cos (2 * x) = 7 / 25) : 
  Real.sin x = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l815_81577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_distinct_elements_l815_81570

theorem sequence_distinct_elements (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) : 
  ∀ k m : ℕ, k ≠ m → f k ≠ f m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_distinct_elements_l815_81570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l815_81582

noncomputable section

variable (f : ℝ → ℝ)

axiom f_odd : ∀ (x : ℝ), x ≠ 0 → f (-x) = -f x
axiom f_domain : ∀ (x : ℝ), x ≠ 0 → f x ≠ 0
axiom f_increasing : ∀ (x y : ℝ), 0 < x → x < y → f x < f y
axiom f_zero_at_three : f 3 = 0

theorem solution_range (x : ℝ) :
  x * (f x - f (-x)) < 0 ↔ (x ∈ Set.Ioo (-3) 0 ∨ x ∈ Set.Ioo 0 3) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l815_81582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l815_81529

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The distance from the center to a focus of the hyperbola -/
noncomputable def focal_distance (h : Hyperbola a b) : ℝ := Real.sqrt (a^2 + b^2)

/-- The equation of an asymptote of the hyperbola -/
noncomputable def asymptote_equation (h : Hyperbola a b) (x : ℝ) : ℝ := (b / a) * x

theorem hyperbola_equation (a b : ℝ) (h : Hyperbola a b) 
  (asymptote_intersect : asymptote_equation h 1 = 2) 
  (focal_distance_eq : focal_distance h = Real.sqrt 5) :
  a = 1 ∧ b = 2 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l815_81529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2024_is_half_l815_81557

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / my_sequence n

theorem sequence_2024_is_half :
  my_sequence 2023 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2024_is_half_l815_81557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l815_81565

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that a train of length 110 m, traveling at 60 kmph,
    takes approximately 16.79 seconds to cross a bridge of length 170 m -/
theorem train_crossing_bridge_time :
  ∃ ε > 0, |train_crossing_time 110 170 60 - 16.79| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l815_81565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l815_81562

/-- Represents a 3x3 grid of unit squares with the lower left corner at the origin -/
structure Grid :=
  (size : ℕ)
  (origin : ℝ × ℝ)

/-- Represents a line from (c,0) to (4,4) -/
structure DividingLine :=
  (c : ℝ)

/-- Calculates the area of the triangle formed by the dividing line and the x-axis -/
def triangleArea (line : DividingLine) : ℝ :=
  (4 - line.c) * 2

/-- Checks if the line divides the grid into two equal areas -/
def dividesEqually (grid : Grid) (line : DividingLine) : Prop :=
  triangleArea line = (grid.size * grid.size : ℝ) / 2

theorem dividing_line_theorem (grid : Grid) (line : DividingLine) :
  grid.size = 3 ∧ grid.origin = (0, 0) →
  dividesEqually grid line ↔ line.c = 1.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l815_81562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_to_read_tomorrow_l815_81532

def book_pages : ℕ := 120
def pages_read_yesterday : ℕ := 12
def pages_read_today : ℕ := 2 * pages_read_yesterday

theorem pages_to_read_tomorrow :
  (book_pages - (pages_read_yesterday + pages_read_today)) / 2 = 42 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_to_read_tomorrow_l815_81532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangentLinesToCircle_l815_81511

/-- Circle with center (1, -2) and radius √5 -/
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

/-- Line with equation x + 2y - 9 = 0 -/
def myLine (x y : ℝ) : Prop := x + 2*y - 9 = 0

/-- A line parallel to the given line -/
def parallelLine (c : ℝ) (x y : ℝ) : Prop := x + 2*y + c = 0

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (x₀ y₀ c : ℝ) : ℝ :=
  |x₀ + 2*y₀ + c| / Real.sqrt 5

theorem tangentLinesToCircle : 
  ∀ c : ℝ, (∀ x y : ℝ, parallelLine c x y → 
    (distancePointToLine 1 (-2) c = Real.sqrt 5)) ↔ (c = 8 ∨ c = -2) :=
by sorry

#check tangentLinesToCircle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangentLinesToCircle_l815_81511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_max_l815_81519

/-- Triangle perimeter function and its maximum value -/
theorem triangle_perimeter_and_max (A B C : ℝ) (x y : ℝ) :
  A = π / 3 →
  0 < x →
  x < 2 * π / 3 →
  A + x + C = π →
  Real.sin A ≠ 0 →
  y = 4 * Real.sin x + 4 * Real.sin (2 * π / 3 - x) + 2 * Real.sqrt 3 →
  ∃ (max_y : ℝ), max_y = 6 * Real.sqrt 3 ∧ y ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_max_l815_81519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l815_81587

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ + 2 * Real.pi / 3)

theorem min_phi_for_odd_function :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), f x φ = -f (-x) φ) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), f x ψ = -f (-x) ψ) → φ ≤ ψ) ∧
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l815_81587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_nine_l815_81528

/-- Represents the sequence of digits formed by concatenating the squares of natural numbers from 1 to 99 -/
def squareSequence : ℕ → ℕ := sorry

/-- Returns the nth digit in the squareSequence -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The 100th digit in the squareSequence is 9 -/
theorem hundredth_digit_is_nine : nthDigit 100 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_nine_l815_81528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_effect_l815_81531

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem salary_increase_effect (salaries : List ℝ) (increase : ℝ) :
  let new_salaries := salaries.map (· + increase)
  mean new_salaries = mean salaries + increase ∧
  variance new_salaries = variance salaries := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_effect_l815_81531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_graph_theorem_l815_81558

structure Graph :=
  (V : Type) -- Vertex set
  (E : V → V → Prop) -- Edge relation
  (symmetric : ∀ u v, E u v → E v u)
  (irreflexive : ∀ v, ¬E v v)

def ColoredGraph (G : Graph) :=
  G.V → G.V → Option Bool

def is_red (color : Option Bool) : Prop :=
  color = some true

def is_black (color : Option Bool) : Prop :=
  color = some false

def triangle (G : Graph) (a b c : G.V) : Prop :=
  G.E a b ∧ G.E b c ∧ G.E c a

def has_red_edge (G : Graph) (CG : ColoredGraph G) (a b c : G.V) : Prop :=
  (is_red (CG a b) ∨ is_red (CG b c) ∨ is_red (CG c a))

def all_red_between (G : Graph) (CG : ColoredGraph G) (vertices : Finset G.V) : Prop :=
  ∀ u v, u ∈ vertices → v ∈ vertices → u ≠ v → is_red (CG u v)

theorem colored_graph_theorem (G : Graph) (CG : ColoredGraph G) 
  (vertex_count : Fintype G.V) (edge_count : Fintype {p : G.V × G.V // G.E p.1 p.2}) :
  (Fintype.card G.V = 9) →
  (Fintype.card {p : G.V × G.V // G.E p.1 p.2} = 36) →
  (∀ a b c, triangle G a b c → has_red_edge G CG a b c) →
  ∃ vertices : Finset G.V, Finset.card vertices = 4 ∧ all_red_between G CG vertices :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_graph_theorem_l815_81558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_is_cone_l815_81540

/-- Spherical coordinates in 3D space -/
structure SphericalCoord where
  r : ℝ
  θ : ℝ
  φ : ℝ

/-- A constant angle -/
def c : ℝ := 0  -- We define c as a constant, for example 0

/-- Definition of a cone in spherical coordinates -/
def is_cone (s : Set SphericalCoord) : Prop :=
  ∃ k, ∀ p ∈ s, p.φ = k

/-- The set of points satisfying φ = c -/
def constant_phi_set : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_is_cone : is_cone constant_phi_set := by
  -- We use 'c' as the constant for the cone
  exists c
  -- For all points in the set, their φ coordinate equals c
  intro p hp
  -- This is true by the definition of constant_phi_set
  exact hp

-- Note: The proof is now complete, so we don't need 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_is_cone_l815_81540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_6s_l815_81514

/-- The time taken for a train to cross a telegraph post -/
noncomputable def train_crossing_time (length : ℝ) (speed : ℝ) : ℝ :=
  length / (speed * 1000 / 3600)

/-- Proof that the train crossing time is approximately 6 seconds -/
theorem train_crossing_approx_6s (length : ℝ) (speed : ℝ) 
  (h1 : length = 80) 
  (h2 : speed = 48) : 
  ∃ ε > 0, |train_crossing_time length speed - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_6s_l815_81514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a4_lt_b4_l815_81533

-- Define the sequences
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define the common ratio for the geometric sequence
def r : ℝ := sorry

-- Define the common difference for the arithmetic sequence
def d : ℝ := sorry

-- Axioms
axiom a_positive : ∀ n, a n > 0
axiom a_geometric : ∀ n, a (n + 1) = r * a n
axiom b_arithmetic : ∀ n, b (n + 1) = b n + d
axiom a1_eq_b1 : a 1 = b 1
axiom a7_eq_b7 : a 7 = b 7
axiom a1_ne_a7 : a 1 ≠ a 7

-- Theorem to prove
theorem a4_lt_b4 : a 4 < b 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a4_lt_b4_l815_81533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l815_81516

/-- Geometric sequence with first term 2 and common ratio q -/
def a (q : ℕ) : ℕ → ℝ
  | 0 => 2
  | n + 1 => q * a q n

/-- Sequence b_n defined by the equation 2n^2 - (t + b_n)n + (3/2)b_n = 0 -/
noncomputable def b (t : ℝ) : ℕ+ → ℝ :=
  fun n => (2 * n.val^2 - t * n.val) / (n.val - 3/2)

theorem geometric_sequence_properties (q : ℕ) (t : ℝ) (h_q : q > 0) :
  (3 * a q 3 = (8 * a q 1 + a q 5) / 2) →
  (∀ n : ℕ+, 2 * n.val^2 - (t + b t n) * n.val + (3/2) * b t n = 0) →
  (∀ n : ℕ, a q n = 2^n) ∧
  (∀ n : ℕ+, b 3 (n + 1) - b 3 n = b 3 2 - b 3 1) := by
  sorry

#check geometric_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l815_81516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_FCG_measure_l815_81575

-- Define the circle and points
variable (circle : Set (EuclideanSpace ℝ (Fin 2)))
variable (A B C D E F G : EuclideanSpace ℝ (Fin 2))

-- Define the clockwise arrangement
def clockwise_arrangement (circle : Set (EuclideanSpace ℝ (Fin 2))) (A B C D E F G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define that AE is a diameter
def is_diameter (circle : Set (EuclideanSpace ℝ (Fin 2))) (A E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define angle measure
noncomputable def angle_measure (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- State the theorem
theorem angle_FCG_measure
  (h_clockwise : clockwise_arrangement circle A B C D E F G)
  (h_diameter : is_diameter circle A E)
  (h_ABF : angle_measure A B F = 81)
  (h_EDG : angle_measure E D G = 76) :
  angle_measure F C G = 67 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_FCG_measure_l815_81575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_counts_SDR_l815_81583

/-- The number of SDR (System of Distinct Representatives) for the array II -/
noncomputable def z (n : ℕ) : ℝ :=
  ((1 + Real.sqrt 5) / 2) ^ n + ((1 - Real.sqrt 5) / 2) ^ n + 2

/-- The array II as described in the problem -/
def array_II (n : ℕ) : List (List ℕ) :=
  [List.range n,
   List.map (λ i => if i = n - 1 then 1 else i + 2) (List.range n),
   List.map (λ i => if i ≥ n - 2 then i - n + 3 else i + 3) (List.range n)]

/-- Number of SDR for a given array -/
def number_of_SDR (array : List (List ℕ)) : ℕ := sorry

/-- Theorem stating that z(n) is the number of SDR for array_II(n) -/
theorem z_counts_SDR (n : ℕ) : 
  z n = (number_of_SDR (array_II n)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_counts_SDR_l815_81583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_to_red_ratio_is_three_to_one_l815_81563

/-- Represents the collection of mugs -/
structure MugCollection where
  total : ℕ
  colors : ℕ
  yellow : ℕ
  other : ℕ
  red : ℕ
  blue : ℕ

/-- The ratio of blue mugs to red mugs -/
def blueToRedRatio (mc : MugCollection) : ℚ :=
  mc.blue / mc.red

/-- Theorem stating the ratio of blue mugs to red mugs is 3:1 -/
theorem blue_to_red_ratio_is_three_to_one (mc : MugCollection)
  (h_total : mc.total = 40)
  (h_colors : mc.colors = 4)
  (h_yellow : mc.yellow = 12)
  (h_other : mc.other = 4)
  (h_red_half_yellow : mc.red = mc.yellow / 2)
  (h_sum : mc.total = mc.blue + mc.red + mc.yellow + mc.other) :
  blueToRedRatio mc = 3 / 1 := by
  sorry

#check blue_to_red_ratio_is_three_to_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_to_red_ratio_is_three_to_one_l815_81563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l815_81572

/-- The area of a triangle bounded by the coordinate axes and the line 3x + 2y = 12 is 12 square units. -/
theorem triangle_area : ∃ A : Set (ℝ × ℝ), MeasureTheory.volume A = 12 := by
  -- Define the line equation
  let line_eq : ℝ → ℝ → Prop := λ x y => 3 * x + 2 * y = 12

  -- Define the triangle
  let triangle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line_eq p.1 p.2}

  -- State that the area of this triangle is 12
  use triangle
  
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l815_81572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l815_81525

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a * Real.tan t.B = 2 * t.b * Real.sin t.A)
  (h2 : t.b = Real.sqrt 3)
  (h3 : t.A = 5 * Real.pi / 12) :
  t.B = Real.pi / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l815_81525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_to_negative_two_to_negative_three_l815_81589

theorem sixteen_to_negative_two_to_negative_three :
  (16 : ℝ)^(-(2 : ℝ)^(-(3 : ℝ))) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_to_negative_two_to_negative_three_l815_81589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_problem_l815_81510

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line given by ax + by + c = 0 -/
def pointOnLine (p : Point2D) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y + c = 0

/-- Check if two lines are parallel -/
def linesParallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

theorem line_problem :
  let l_slope := -4/5
  let m : ℝ → ℝ → ℝ → Prop := fun x y c ↦ 4 * x + 5 * y + c = 0
  (l_slope = -4/5) ∧
  (∃ c, m 0 2 c ∧ linesParallel 4 5 4 5) := by
  sorry

#check line_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_problem_l815_81510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l815_81576

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- State the theorem
theorem f_monotone_decreasing_on_interval :
  StrictMonoOn f (Set.Ioo (0 : ℝ) (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l815_81576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_quiz_problem_l815_81518

/-- Represents the quiz problem with given parameters -/
structure QuizProblem where
  total_quizzes : ℕ
  goal_percentage : ℚ
  current_As : ℕ
  quizzes_taken : ℕ
  max_non_As : ℕ
  h_goal : (current_As + (total_quizzes - quizzes_taken - max_non_As) : ℚ) / total_quizzes ≥ goal_percentage
  h_remaining : max_non_As ≤ total_quizzes - quizzes_taken

/-- The theorem statement for Lisa's quiz problem -/
theorem lisa_quiz_problem :
  let p : QuizProblem := {
    total_quizzes := 60,
    goal_percentage := 85 / 100,
    current_As := 28,
    quizzes_taken := 40,
    max_non_As := 0,
    h_goal := by sorry,
    h_remaining := by sorry
  }
  p.max_non_As = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_quiz_problem_l815_81518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l815_81556

/-- Given two vectors a and b in ℝ², prove that the angle between them is π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  (norm a = 1) →
  (b = (1, Real.sqrt 3)) →
  ((b.1 - a.1, b.2 - a.2) • a = 0) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (norm a * norm b)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l815_81556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_315_l815_81502

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial Q -/
def SatisfiesConditions (Q : IntPolynomial) (b : ℤ) : Prop :=
  (Q.eval 2 = b) ∧ (Q.eval 4 = b) ∧ (Q.eval 6 = b) ∧ (Q.eval 8 = b) ∧
  (Q.eval 1 = -b) ∧ (Q.eval 3 = -b) ∧ (Q.eval 5 = -b) ∧ (Q.eval 7 = -b)

/-- The main theorem: 315 is the smallest positive integer satisfying the conditions -/
theorem smallest_b_is_315 :
  ∀ b : ℕ+, (∃ Q : IntPolynomial, SatisfiesConditions Q b) → b ≥ 315 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_315_l815_81502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base10_to_base7_conversion_l815_81571

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- The base-10 number we're converting --/
def base10Number : ℕ := 1023

/-- The proposed base-7 representation --/
def base7Representation : Fin 4 → ℕ := ![2, 6, 6, 1]

theorem base10_to_base7_conversion :
  base7ToBase10 (base7Representation 0) (base7Representation 1) (base7Representation 2) (base7Representation 3) = base10Number :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base10_to_base7_conversion_l815_81571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_triangle_area_12_13_l815_81503

/-- The maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle -/
noncomputable def max_inscribed_triangle_area (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  338 * Real.sqrt 3 - 507

/-- Theorem stating the maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle -/
theorem max_inscribed_triangle_area_12_13 :
  max_inscribed_triangle_area 12 13 = 338 * Real.sqrt 3 - 507 := by
  sorry

#check max_inscribed_triangle_area_12_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_triangle_area_12_13_l815_81503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_log3_243_l815_81568

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the logarithm base 3 function
noncomputable def log3 (x : ℝ) : ℝ :=
  Real.log x / Real.log 3

-- Define the sum of floor of log3 from 1 to n
noncomputable def sum_floor_log3 (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i => floor (log3 ((i + 1 : ℕ) : ℝ)))

-- Theorem statement
theorem sum_floor_log3_243 : sum_floor_log3 243 = 857 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_log3_243_l815_81568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l815_81517

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 300)
  (h2 : profit_percentage = 0.20) : 
  selling_price / (1 + profit_percentage) = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l815_81517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_eq_one_inequality_for_x_between_one_and_two_l815_81521

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Theorem 1: If f is tangent to y = x - 1, then a = 1
theorem tangent_implies_a_eq_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ - 1 ∧ (deriv (f a)) x₀ = 1) →
  a = 1 := by sorry

-- Theorem 2: Inequality for 1 < x < 2
theorem inequality_for_x_between_one_and_two (x : ℝ) :
  1 < x → x < 2 →
  (1 / Real.log x) - (1 / Real.log (x - 1)) < 1 / ((x - 1) * (2 - x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_eq_one_inequality_for_x_between_one_and_two_l815_81521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l815_81501

-- Define the volume of the sphere
noncomputable def sphere_volume : ℝ := 4 * Real.sqrt 3 * Real.pi

-- Define the relationship between the sphere and the cube
def cube_inscribed_in_sphere (s : ℝ) (r : ℝ) : Prop :=
  s * Real.sqrt 3 = 2 * r

-- Theorem statement
theorem cube_surface_area (s : ℝ) (r : ℝ) : 
  sphere_volume = (4 / 3) * Real.pi * r^3 →
  cube_inscribed_in_sphere s r →
  6 * s^2 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l815_81501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l815_81500

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin (x - 7 * π / 12)

-- State the theorem
theorem f_monotone_increasing : 
  StrictMonoOn f (Set.Icc (π / 12) (13 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l815_81500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l815_81538

-- Define vectors in 2D space
def Vector2D := ℝ × ℝ

-- Define dot product
def dot_product (a b : Vector2D) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define vector magnitude
noncomputable def magnitude (v : Vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define parallel vectors
def parallel (a b : Vector2D) : Prop := ∃ (k : ℝ), a = (k * b.1, k * b.2)

theorem vector_problems :
  -- Part 1
  ∀ (a b : Vector2D),
  magnitude a = 3 →
  magnitude b = 4 →
  dot_product a b = 3 * 4 * Real.cos (π / 3) →
  dot_product a b = 6 ∧ 
  magnitude (a.1 - b.1, a.2 - b.2) = Real.sqrt 13 ∧
  -- Part 2
  ∀ (a : Vector2D),
  magnitude a = 3 →
  parallel a (1, 2) →
  (a = (3 * Real.sqrt 5 / 5, 6 * Real.sqrt 5 / 5) ∨ 
   a = (-3 * Real.sqrt 5 / 5, -6 * Real.sqrt 5 / 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l815_81538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_of_equations_solution_l815_81504

theorem system_of_equations_solution :
  ∃ (x y : ℝ) (z : ℝ),
    (3 * x = 20 + (20 - x)) ∧
    (y = 2 * x - 5) ∧
    (z = Real.sqrt (x + 4)) ∧
    (x = 10) ∧
    (y = 15) ∧
    (z = Real.sqrt 14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_of_equations_solution_l815_81504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l815_81566

/-- A geometric sequence with first term a and common ratio q -/
def GeometricSequence (a : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a * q ^ (n - 1)

theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) :
  let a_n := GeometricSequence a q
  (a_n 1 + 3 * a_n 3) / (a_n 2 + 3 * a_n 4) = 1 / 2 →
  (a_n 4 * a_n 6 + a_n 6 * a_n 8) / (a_n 6 * a_n 8 + a_n 8 * a_n 10) = 1 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l815_81566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_radius_for_flower_bed_l815_81543

/-- The radius that maximizes the area of a circular sector with given constraints -/
noncomputable def optimal_radius (perimeter : ℝ) (min_angle : ℝ) (max_angle : ℝ) : ℝ :=
  perimeter / (2 + max_angle)

theorem optimal_radius_for_flower_bed :
  let perimeter : ℝ := 16
  let min_angle : ℝ := π / 4  -- 45 degrees in radians
  let max_angle : ℝ := 3 * π / 4  -- 135 degrees in radians
  optimal_radius perimeter min_angle max_angle = 16 / (2 + 3 * π / 4) :=
by
  -- Unfold the definition of optimal_radius
  unfold optimal_radius
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_radius_for_flower_bed_l815_81543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l815_81547

theorem inequality_proof (n : ℕ) (h : n > 1) :
  (1 : ℝ) / (2 * ↑n * Real.exp 1) < 
  (1 : ℝ) / Real.exp 1 - (1 - 1 / ↑n) ^ (↑n) ∧
  (1 : ℝ) / Real.exp 1 - (1 - 1 / ↑n) ^ (↑n) < 
  (1 : ℝ) / (↑n * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l815_81547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_log10_2013_l815_81579

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Logarithm base 10 -/
noncomputable def log10 (x : ℝ) : ℝ :=
  Real.log x / Real.log 10

/-- Sum of floor of log10 for integers from 1 to n -/
noncomputable def sum_floor_log10 (n : ℕ) : ℤ :=
  (Finset.range n).sum (fun i => floor (log10 (i + 1 : ℝ)))

/-- The main theorem -/
theorem sum_floor_log10_2013 :
  sum_floor_log10 2013 = 4932 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_log10_2013_l815_81579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_positive_factors_count_l815_81539

def n : ℕ := 2^3 * 3^3 * 5

theorem even_positive_factors_count : 
  (Finset.filter (fun d => n % d = 0 ∧ Even d ∧ d > 0) (Finset.range (n + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_positive_factors_count_l815_81539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_x_minus_2y_is_closed_open_interval_l815_81592

def range_of_x_minus_2y : Set ℝ :=
  {z | ∃ (x y : ℝ), -1 ≤ x ∧ x < 2 ∧ 0 < y ∧ y ≤ 1 ∧ z = x - 2*y}

theorem range_x_minus_2y_is_closed_open_interval :
  range_of_x_minus_2y = Set.Ioc (-3 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_x_minus_2y_is_closed_open_interval_l815_81592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_people_who_purchased_only_book_A_l815_81523

/-- Represents the number of people who purchased only Book A -/
def A : ℕ := sorry

/-- Represents the number of people who purchased only Book B -/
def B : ℕ := sorry

/-- Represents the number of people who purchased only Book C -/
def C : ℕ := sorry

/-- Represents the number of people who purchased both Books A and B -/
def AB : ℕ := sorry

/-- Represents the number of people who purchased both Books B and C -/
def BC : ℕ := sorry

/-- Represents the number of people who purchased both Books A and C -/
def AC : ℕ := sorry

/-- The price of Book A -/
def priceA : ℕ := 25

/-- The price of Book B -/
def priceB : ℕ := 20

/-- The price of Book C -/
def priceC : ℕ := 15

theorem number_of_people_who_purchased_only_book_A :
  (A + AB = 2 * (B + AB)) →
  (AB = 500) →
  (AB = 2 * B) →
  (BC = 300) →
  (priceA * (A + AB + AC) + priceB * (B + AB + BC) + priceC * (C + AC + BC) = 15000) →
  A = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_people_who_purchased_only_book_A_l815_81523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_speed_l815_81564

/-- The speed of train B given the conditions of the problem -/
theorem train_b_speed 
  (length_a : ℝ) 
  (length_b : ℝ) 
  (speed_a : ℝ) 
  (crossing_time : ℝ) 
  (h1 : length_a = 175) 
  (h2 : length_b = 150) 
  (h3 : speed_a = 54) 
  (h4 : crossing_time = 13) : 
  (length_a + length_b) / crossing_time * 3.6 - speed_a = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_speed_l815_81564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_transformation_l815_81596

/-- Represents a pyramid with a rectangular base -/
structure Pyramid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := (1/3) * p.length * p.width * p.height

/-- Transforms a pyramid by quadrupling its base dimensions and doubling its height -/
def transform (p : Pyramid) : Pyramid :=
  { length := 4 * p.length,
    width := 4 * p.width,
    height := 2 * p.height }

theorem pyramid_volume_transformation (p : Pyramid) 
  (h : volume p = 120) : 
  volume (transform p) = 3840 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_transformation_l815_81596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l815_81542

/-- The repeating decimal 0.7373... expressed as a real number -/
noncomputable def repeating_decimal : ℝ := 0.73737373

/-- The theorem stating that the repeating decimal 0.7373... is equal to 73/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 73 / 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l815_81542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l815_81586

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes has a slope of √2 and its right focus is at (√3, 0),
    then its eccentricity is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), m = Real.sqrt 2 ∧ b / a = m) →  -- One asymptote has slope √2
  (∃ (c : ℝ), c = Real.sqrt 3) →              -- Right focus is at (√3, 0)
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 3 :=   -- Eccentricity is √3
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l815_81586
