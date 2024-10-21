import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l565_56585

-- Define the cone
structure Cone where
  base_radius : ℝ
  lateral_surface_is_semicircle : Bool

-- Define the properties we want to prove
def angle_generatrix_height (c : Cone) : ℝ := 30

noncomputable def max_triangle_area (c : Cone) : ℝ := Real.sqrt 3

-- State the theorem
theorem cone_properties (c : Cone) 
  (h1 : c.base_radius = 1) 
  (h2 : c.lateral_surface_is_semicircle = true) : 
  angle_generatrix_height c = 30 ∧ max_triangle_area c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l565_56585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_problem_l565_56526

theorem urn_problem (M : ℝ) : 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 →
  M = 111 :=
by
  intro h
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_problem_l565_56526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_value_l565_56574

theorem like_terms_imply_value (a b : ℝ) (m n : ℕ) :
  (2 * a^2 * b^(m + 1) = -3 * a^n * b^2) → ((-1:ℤ) * m)^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_value_l565_56574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l565_56560

theorem min_value_theorem (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * (Real.log b / Real.log a) + 6 * (Real.log a / Real.log b) = 11) : 
  ∀ x y : ℝ, x > 1 ∧ y > x → a^3 + 2/(b-1) ≥ 2*Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l565_56560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zero_composition_l565_56579

-- Define a polynomial with integer coefficients
def IntPolynomial := Polynomial ℤ

-- Define n-fold composition of a function
def nFoldComposition (f : α → α) : ℕ → (α → α)
  | 0 => id
  | n + 1 => f ∘ nFoldComposition f n

theorem polynomial_zero_composition (P : IntPolynomial) :
  (∀ n : ℕ, nFoldComposition (λ x ↦ P.eval x) n 0 = 0) →
  P.eval 0 = 0 ∨ P.eval (P.eval 0) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zero_composition_l565_56579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hawaii_rainfall_days_left_l565_56528

/-- The number of days left in the year given the current rainfall and required average -/
def days_left (total_days : ℕ) (normal_avg : ℚ) (current_rain : ℕ) (required_avg : ℚ) : ℚ :=
  ((total_days : ℚ) * normal_avg - current_rain) / (required_avg - normal_avg)

theorem hawaii_rainfall_days_left :
  ⌊days_left 365 2 430 3⌋ = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hawaii_rainfall_days_left_l565_56528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_l565_56597

theorem min_value_trigonometric_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  (∀ φ, 0 < φ ∧ φ < π/2 → 3 * Real.sin φ + 2 / Real.cos φ + Real.sqrt 3 * (Real.cos φ / Real.sin φ) ≥ 7) ∧
  (∃ φ, 0 < φ ∧ φ < π/2 ∧ 3 * Real.sin φ + 2 / Real.cos φ + Real.sqrt 3 * (Real.cos φ / Real.sin φ) = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_l565_56597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_symmetric_xoy_l565_56552

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Symmetry with respect to xoy plane -/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem segment_length_symmetric_xoy :
  let A : Point3D := { x := 1, y := 2, z := -1 }
  let B : Point3D := symmetricXOY A
  distance A B = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_symmetric_xoy_l565_56552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l565_56506

/-- The projection of vector u onto vector v -/
noncomputable def proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (v.1^2 + v.2^2)
  (scalar * v.1, scalar * v.2)

/-- The condition that the projection of u onto (3, -4) equals (9/2, -6) -/
def projection_condition (u : ℝ × ℝ) : Prop :=
  proj (3, -4) u = (9/2, -6)

theorem projection_line_equation :
  ∀ u : ℝ × ℝ, projection_condition u →
  ∃ x y : ℝ, u = (x, y) ∧ y = -3/4 * x + 75/8 := by
  sorry

#check projection_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l565_56506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exists_with_slope_l565_56511

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (1, -3)

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- The circle function -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

/-- The slope of the tangent line -/
def tangent_slope : ℝ := -2

theorem tangent_line_exists_with_slope :
  ∃ (m b : ℝ), 
    (∃ (x y : ℝ), y = m * x + b ∧ y = parabola x) ∧  -- tangent to parabola
    (∃ (x y : ℝ), y = m * x + b ∧ circle_eq x y) ∧   -- tangent to circle
    (P.2 = m * P.1 + b) ∧                            -- passes through P
    m = tangent_slope :=                             -- has the specified slope
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exists_with_slope_l565_56511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_medians_theorem_l565_56549

/-- A triangle with two perpendicular medians -/
structure TriangleWithPerpendicularMedians where
  /-- The triangle vertices -/
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  /-- The point where median DP intersects EF -/
  P : ℝ × ℝ
  /-- The point where median EQ intersects DF -/
  Q : ℝ × ℝ
  /-- DP is a median -/
  dp_is_median : P = ((F.1 + E.1) / 2, (F.2 + E.2) / 2)
  /-- EQ is a median -/
  eq_is_median : Q = ((F.1 + D.1) / 2, (F.2 + D.2) / 2)
  /-- DP and EQ are perpendicular -/
  medians_perpendicular : (P.1 - D.1) * (Q.1 - E.1) + (P.2 - D.2) * (Q.2 - E.2) = 0
  /-- Length of DP is 15 -/
  dp_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 15
  /-- Length of EQ is 20 -/
  eq_length : Real.sqrt ((Q.1 - E.1)^2 + (Q.2 - E.2)^2) = 20

/-- The main theorem -/
theorem perpendicular_medians_theorem (t : TriangleWithPerpendicularMedians) :
  Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2) = 50 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_medians_theorem_l565_56549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_land_plot_area_l565_56569

/-- The area of a triangle given two sides and the angle between them -/
noncomputable def triangle_area (a b : ℝ) (θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

/-- The triangle PQR representing the plot of land -/
structure LandPlot where
  PR : ℝ  -- Length of Pine Road
  QR : ℝ  -- Length of Maple Road
  θ : ℝ   -- Angle between PR and QR

/-- The theorem stating that the area of the land plot is 4 square miles -/
theorem land_plot_area (plot : LandPlot) 
  (h1 : plot.PR = 2)  -- Pine Road is 2 miles
  (h2 : plot.QR = 4)  -- Maple Road is 4 miles
  (h3 : plot.θ = Real.pi/2)  -- The angle between PR and QR is 90 degrees
  : triangle_area plot.PR plot.QR plot.θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_land_plot_area_l565_56569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_numbers_approx_l565_56510

/-- The number of dice being rolled -/
def n : ℕ := 10

/-- The number of faces on each die -/
def k : ℕ := 6

/-- The probability of rolling 10 fair dice and getting each number from 1 to 6 at least once -/
noncomputable def prob_all_numbers : ℝ :=
  1 - (k : ℝ) * (k - 1 : ℝ) ^ n / k ^ n
    + (k.choose 2 : ℝ) * (k - 2 : ℝ) ^ n / k ^ n
    - (k.choose 3 : ℝ) * (k - 3 : ℝ) ^ n / k ^ n
    + (k.choose 4 : ℝ) * (k - 4 : ℝ) ^ n / k ^ n
    - (k.choose 5 : ℝ) * (k - 5 : ℝ) ^ n / k ^ n

/-- The theorem stating that the probability is approximately 0.272 -/
theorem prob_all_numbers_approx :
  ∃ ε > 0, abs (prob_all_numbers - 0.272) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_numbers_approx_l565_56510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l565_56537

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Definition of the line l passing through P(m, 0) with slope √2/2 -/
def line_l (m x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * (x - m)

/-- Points A and B are the intersections of line l and ellipse C -/
def intersection_points (m x y : ℝ) : Prop :=
  ellipse_C x y ∧ line_l m x y

/-- The sum of squared distances from P to A and B is constant -/
theorem constant_sum_of_squares (m : ℝ) (h : -2 ≤ m ∧ m ≤ 2) :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection_points m x1 y1 ∧
    intersection_points m x2 y2 ∧
    (x1 - m)^2 + y1^2 + (x2 - m)^2 + y2^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l565_56537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_mixing_l565_56557

/-- Represents a sugar solution with a given weight and sugar content -/
structure SugarSolution where
  totalWeight : ℝ
  sugarWeight : ℝ
  sugarPercentage : ℝ
  h_percentage : sugarPercentage = sugarWeight / totalWeight * 100

/-- The result of mixing two sugar solutions -/
noncomputable def mixSolutions (s1 s2 : SugarSolution) (replacementRatio : ℝ) : SugarSolution where
  totalWeight := s1.totalWeight
  sugarWeight := s1.sugarWeight * (1 - replacementRatio) + s2.sugarWeight * replacementRatio
  sugarPercentage := (s1.sugarWeight * (1 - replacementRatio) + s2.sugarWeight * replacementRatio) / s1.totalWeight * 100
  h_percentage := by sorry

theorem sugar_solution_mixing :
  ∀ (initialSolution replacingSolution : SugarSolution),
    initialSolution.sugarPercentage = 10 →
    replacingSolution.sugarPercentage = 50 →
    (mixSolutions initialSolution replacingSolution 0.25).sugarPercentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_mixing_l565_56557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l565_56566

def total_amount : ℝ := 12000

theorem money_distribution (p q r s : ℝ) : 
  p + q + r + s = total_amount →
  r = (2/3) * (p + q) →
  s = (1/4) * (p + q) →
  (abs (p + q - 6260.87) < 0.01 ∧ 
   abs (r - 4173.91) < 0.01 ∧ 
   abs (s - 1565.22) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l565_56566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l565_56583

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (|p.1| - p.1)^2 + (|p.2| - p.2)^2 ≤ 16 ∧ 2*p.2 + p.1 ≤ 0}

-- State the theorem
theorem area_of_S : MeasureTheory.volume S = 5 + π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l565_56583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_params_correct_l565_56542

/-- Definition of an ellipse with given foci and a point on the curve -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  point : ℝ × ℝ

/-- Parameters of the standard form equation of an ellipse -/
structure EllipseParams where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating the correctness of the ellipse parameters -/
theorem ellipse_params_correct (e : Ellipse) (p : EllipseParams) : 
  e.focus1 = (-3, 1) → 
  e.focus2 = (3, 1) → 
  e.point = (14, 9) → 
  p.a > 0 → 
  p.b > 0 → 
  (p.a = 16 ∧ 
   p.b^2 = 247 ∧ 
   p.h = 0 ∧ 
   p.k = 1) → 
  (∀ (x y : ℝ), (x - p.h)^2 / p.a^2 + (y - p.k)^2 / p.b^2 = 1 ↔ 
    distance (x, y) e.focus1 + distance (x, y) e.focus2 = 2 * p.a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_params_correct_l565_56542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_and_expectation_l565_56505

/-- Represents the number of teachers of each subject -/
def total_teachers : ℕ := 6
def chinese_teachers : ℕ := 2
def math_teachers : ℕ := 2
def english_teachers : ℕ := 2

/-- Number of teachers to be selected -/
def selected_teachers : ℕ := 3

/-- Random variable X representing the number of math teachers selected -/
def X : ℕ → ℝ := sorry

/-- Probability mass function of X -/
def pmf_X : ℕ → ℝ := sorry

/-- The probability of selecting more math teachers than Chinese teachers -/
def prob_more_math_than_chinese : ℝ := sorry

/-- The expectation of X -/
def expectation_X : ℝ := sorry

theorem prob_and_expectation :
  prob_more_math_than_chinese = 3/10 ∧ expectation_X = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_and_expectation_l565_56505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_condition_l565_56507

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (|x + a|) / Real.log (1/2)

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x < 0, f a x ≤ 0

-- State the theorem
theorem graph_condition (a : ℝ) :
  not_in_second_quadrant a ↔ a ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_condition_l565_56507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l565_56518

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Parabola y^2 = 2x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = 2 * p.x

/-- Projection of a point on y-axis -/
def projectionOnYAxis (p : Point) : Point :=
  ⟨0, p.y⟩

/-- Theorem: Minimum value of |PA| + |PM| is 9/2 -/
theorem min_distance_sum (p : Point) (h : onParabola p) : 
  let a : Point := ⟨7/2, 4⟩
  let m : Point := projectionOnYAxis p
  9/2 ≤ distance p a + distance p m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l565_56518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_l565_56561

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

-- Define proposition q
def q : Prop := ∀ A B C : ℝ, 
  (0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) → 
  (A > B → Real.sin A > Real.sin B)

-- Theorem statement
theorem p_or_q : p ∨ q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_l565_56561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_26_l565_56504

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its four vertices -/
noncomputable def trapezoidArea (e f g h : Point) : ℝ :=
  let height := h.x - e.x
  let base1 := f.y - e.y
  let base2 := g.y - h.y
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid EFGH with given vertices is 26 square units -/
theorem trapezoid_area_is_26 :
  let e : Point := ⟨2, -3⟩
  let f : Point := ⟨2, 2⟩
  let g : Point := ⟨6, 8⟩
  let h : Point := ⟨6, 0⟩
  trapezoidArea e f g h = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_26_l565_56504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l565_56513

-- Define the function f(x) = 2^(x-1) + x - 5
noncomputable def f (x : ℝ) : ℝ := 2^(x-1) + x - 5

-- Theorem statement
theorem solution_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l565_56513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_5292_l565_56524

theorem largest_prime_factor_of_5292 :
  (Nat.factors 5292).maximum? = some 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_5292_l565_56524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l565_56590

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ a < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.b^2 / h.a^2) + 1)

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point -/
def O : Point := ⟨0, 0⟩

/-- Distance between two points -/
noncomputable def dist (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Predicate for three lengths forming an arithmetic progression -/
def arithmetic_progression (a b c : ℝ) : Prop := b - a = c - b

theorem hyperbola_eccentricity (h : Hyperbola) 
  (A B : Point) 
  (h_line : ∃ (F : Point), dist F O = h.a * eccentricity h ∧ 
    (A.y - F.y) * (B.x - A.x) = (B.y - A.y) * (A.x - F.x))
  (h_arithmetic : arithmetic_progression (dist O A) (dist A B) (dist O B)) :
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l565_56590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_implies_sum_l565_56588

theorem determinant_implies_sum (x y : ℝ) (h1 : x ≠ y) 
  (h2 : Matrix.det ![![2, 3, 7], ![4, x, y], ![4, y, x + 1]] = 0) : 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_implies_sum_l565_56588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l565_56529

theorem integer_divisible_by_18_with_sqrt_between_26_and_26_2 :
  ∃ n : ℕ, (n : ℝ).sqrt > 26 ∧ (n : ℝ).sqrt < 26.2 ∧ n % 18 = 0 ∧ n = 684 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l565_56529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_max_area_theorem_l565_56500

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the upper vertex A
def A : ℝ × ℝ := (0, 1)

-- Define a line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the condition AP · AQ = 0
def perpendicular_vectors (P Q : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) = 0

-- Helper function to calculate triangle area
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem 1: The line always passes through a fixed point
theorem fixed_point_theorem (k : ℝ) (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 → ellipse Q.1 Q.2 →
  line_through_A k P.1 P.2 → line_through_A k Q.1 Q.2 →
  perpendicular_vectors P Q →
  ∃ (F : ℝ × ℝ), F = (0, -1/2) ∧ line_through_A k F.1 F.2 :=
by
  sorry

-- Theorem 2: The maximum area of triangle APQ
theorem max_area_theorem :
  ∃ (max_area : ℝ), max_area = 9/4 ∧
  ∀ (k : ℝ) (P Q : ℝ × ℝ),
    ellipse P.1 P.2 → ellipse Q.1 Q.2 →
    line_through_A k P.1 P.2 → line_through_A k Q.1 Q.2 →
    perpendicular_vectors P Q →
    area_triangle A P Q ≤ max_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_max_area_theorem_l565_56500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l565_56589

-- Define the circle C
noncomputable def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y = 0

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop :=
  y = x

-- Define the center of the circle
noncomputable def center_C (m : ℝ) : ℝ × ℝ :=
  (1, -m/2)

-- Define the tangent line
def tangent_line (k : ℝ) (x y : ℝ) : Prop :=
  k*x - y + k + 1 = 0

theorem circle_and_tangent_line :
  ∃ (m : ℝ),
    (∀ x y : ℝ, circle_C m x y → line_y_eq_x (center_C m).1 (center_C m).2) ∧
    (m = -2) ∧
    (∃ k : ℝ, (k = 1 ∨ k = -1) ∧
      (∀ x y : ℝ, tangent_line k x y →
        ((x + 1)^2 + (y - 1)^2 = 2 → x = -1 ∧ y = 1) ∨
        ((x + 1)^2 + (y - 1)^2 > 2))) :=
by sorry

#check circle_and_tangent_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l565_56589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_100_terms_equals_50_l565_56547

noncomputable def a (n : ℕ+) : ℝ := n * Real.cos (n * Real.pi)

theorem sum_first_100_terms_equals_50 :
  Finset.sum (Finset.range 100).attach (fun i => a ⟨i.val + 1, Nat.succ_pos i.val⟩) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_100_terms_equals_50_l565_56547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_number_equals_sum_of_white_numbers_l565_56554

-- Define white numbers
noncomputable def WhiteNumber (a b : ℤ) : ℝ := Real.sqrt (a + b * Real.sqrt 2)

-- Define black numbers
noncomputable def BlackNumber (c d : ℤ) : ℝ := Real.sqrt (c + d * Real.sqrt 7)

-- Theorem statement
theorem black_number_equals_sum_of_white_numbers :
  ∃ (a b c d a' b' : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    BlackNumber c d = WhiteNumber a b + WhiteNumber a' b' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_number_equals_sum_of_white_numbers_l565_56554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_triangle_inequality_l565_56534

/-- Given four distinct collinear points P, Q, R, S in that order,
    prove that PQ < PS/3 must be satisfied for a triangle to be formed
    when PQ and RS are rotated about Q and R respectively. -/
theorem collinear_points_triangle_inequality (P Q R S : EuclideanSpace ℝ (Fin 2)) (a b c : ℝ) :
  P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧  -- Points are distinct
  (∃ (t : ℝ), t ∈ Set.Ioo 0 1 ∧ Q = (1 - t) • P + t • S ∧
               R = (1 - t) • Q + t • S) ∧  -- Points are collinear
  dist P Q = a ∧
  dist P R = b ∧
  dist P S = c ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- Distances are positive
  (∃ (T : EuclideanSpace ℝ (Fin 2)), dist Q T = a ∧ dist R T = c - b ∧ dist T S = a) →  -- Triangle can be formed
  a < c / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_triangle_inequality_l565_56534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l565_56546

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 6) = 4 / 5) : 
  Real.sin α = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l565_56546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bulls_win_in_five_games_l565_56527

-- Define the probability of the Heat winning a single game
def p_heat_win : ℚ := 3/4

-- Define the probability of the Bulls winning a single game
def p_bulls_win : ℚ := 1 - p_heat_win

-- Define the number of games needed to win the series
def games_to_win : ℕ := 3

-- Define the maximum number of games in the series
def max_games : ℕ := 5

-- Theorem statement
theorem bulls_win_in_five_games :
  (Nat.choose 4 2 : ℚ) * p_bulls_win^2 * p_heat_win^2 * p_bulls_win = 27/512 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bulls_win_in_five_games_l565_56527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_alpha_l565_56572

theorem tan_double_alpha (α : ℝ) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) :
  Real.tan (2 * α) = -3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_alpha_l565_56572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l565_56553

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two :
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l565_56553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toot_only_symmetric_l565_56531

-- Define the set of letters with vertical symmetry
def symmetricLetters : Set Char := {'M', 'A', 'T', 'O'}

-- Define the set of words to consider
def words : List String := ["ROOT", "BOOM", "BOOT", "LOOT", "TOOT"]

-- Function to check if a word has vertical symmetry
def hasVerticalSymmetry (word : String) : Prop :=
  ∀ c, c ∈ word.data → c ∈ symmetricLetters

-- Theorem stating that TOOT is the only word with vertical symmetry
theorem toot_only_symmetric :
  ∃! word, word ∈ words ∧ hasVerticalSymmetry word ∧ word = "TOOT" :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toot_only_symmetric_l565_56531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l565_56596

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

def Parabola.contains (c : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * c.p * p.x

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

noncomputable def geometric_mean (a b : ℝ) : ℝ :=
  Real.sqrt (a * b)

def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

theorem parabola_theorem (c : Parabola) (P F Q : Point) (l1 l2 : Line) 
    (h1 : c.contains P)
    (h2 : P.x = 2)
    (h3 : distance P F = 4)
    (h4 : Q.x = 3 ∧ Q.y = 2)
    (h5 : ∃ A B, c.contains A ∧ c.contains B ∧ A ≠ B ∧ l1.contains A ∧ l1.contains B)
    (h6 : l2.contains F)
    (h7 : Line.perpendicular l1 l2)
    (h8 : ∃ M N, c.contains M ∧ c.contains N ∧ M ≠ N ∧ l2.contains M ∧ l2.contains N)
    (h9 : ∃ A B, l1.contains A ∧ l1.contains B ∧ A ≠ B ∧ 
         distance M N = geometric_mean (distance Q A) (distance Q B)) :
  distance M N = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l565_56596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_remaining_pencils_l565_56548

def jeff_initial_pencils : ℕ := 300
def jeff_donation_rate : ℚ := 30 / 100
def vicki_initial_multiplier : ℕ := 2
def vicki_donation_rate : ℚ := 3 / 4

def remaining_pencils : ℕ :=
  let jeff_remaining := jeff_initial_pencils - Int.floor (jeff_initial_pencils * jeff_donation_rate)
  let vicki_initial := jeff_initial_pencils * vicki_initial_multiplier
  let vicki_remaining := vicki_initial - Int.floor (vicki_initial * vicki_donation_rate)
  (jeff_remaining + vicki_remaining).toNat

theorem total_remaining_pencils :
  remaining_pencils = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_remaining_pencils_l565_56548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_6_sqrt_2_l565_56575

/-- The polynomial whose roots form the rhombus -/
def f (z : ℂ) : ℂ := z^4 + 4*Complex.I*z^3 + (2 + 2*Complex.I)*z^2 + (4 - 4*Complex.I)*z - (1 + 4*Complex.I)

/-- The roots of the polynomial f -/
noncomputable def roots : Finset ℂ := sorry

/-- The rhombus formed by the roots of f -/
noncomputable def rhombus : Set (ℂ × ℂ) := sorry

/-- The area of a rhombus -/
noncomputable def rhombusArea (r : Set (ℂ × ℂ)) : ℝ := sorry

/-- Theorem: The area of the rhombus formed by the roots of f is 6√2 -/
theorem rhombus_area_is_6_sqrt_2 : rhombusArea rhombus = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_6_sqrt_2_l565_56575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_equivalence_l565_56541

/-- A line in 3D space defined by a point and a direction vector -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Condition for a point to lie on a line in 3D space -/
def lieOnLine (l : Line3D) (p : Fin 3 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 3, p i = l.point i + t * l.direction i

theorem line_equation_equivalence (l : Line3D) (p : Fin 3 → ℝ) 
    (h_dir : ∀ i : Fin 3, l.direction i ≠ 0) :
  lieOnLine l p ↔ 
    ∀ i j : Fin 3, (p i - l.point i) / l.direction i = (p j - l.point j) / l.direction j :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_equivalence_l565_56541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l565_56578

-- Define λ as the definite integral of x^2 from 0 to 3
noncomputable def lambda : ℝ := ∫ x in (0:ℝ)..(3:ℝ), x^2

-- Define a geometric sequence with positive terms
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem min_value_of_expression (a : ℕ → ℝ) 
  (h : isGeometricSequence a) :
  ∃ m : ℝ, m = 6 ∧ ∀ x : ℝ, m ≤ (a 4 + lambda * a 2) / a 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l565_56578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_2cos4_l565_56586

theorem min_sin4_2cos4 :
  (∃ x : ℝ, Real.sin x ^ 4 + 2 * Real.cos x ^ 4 = 2 / 3) ∧
  (∀ x : ℝ, Real.sin x ^ 4 + 2 * Real.cos x ^ 4 ≥ 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_2cos4_l565_56586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l565_56515

theorem cube_root_54880000 : (54880000 : ℝ) ^ (1/3 : ℝ) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l565_56515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l565_56595

theorem problem_solution (x : ℝ) (p q r : ℕ) : 
  (1 + Real.sin x) * (1 + Real.cos x) = 9/4 →
  (1 - Real.sin x) * (1 - Real.cos x) = p/q - Real.sqrt r →
  Nat.Coprime p q →
  p + q + r = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l565_56595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1000_l565_56503

/-- Represents the sequence described in the problem -/
def sequenceElement : ℕ → ℕ
| 0 => 1
| n + 1 => sorry

/-- The sum of the first n terms of the sequence -/
def sequenceSum (n : ℕ) : ℕ := sorry

/-- The number of complete sets in the first n terms -/
def completeSets (n : ℕ) : ℕ := sorry

/-- The sum of a complete set starting at the k-th set -/
def setSum (k : ℕ) : ℕ := sorry

theorem sequence_sum_1000 :
  ∃ (S : ℕ), sequenceSum 1000 = S ∧ 2000 ≤ S ∧ S ≤ 2200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1000_l565_56503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2019_l565_56550

def sequence_a : ℕ → ℤ
  | 0 => 1  -- We need to handle the case for 0
  | 1 => 1
  | 2 => -1
  | n+3 => sequence_a (n+2) * sequence_a n

theorem sequence_a_2019 : sequence_a 2019 = -1 := by
  sorry

#eval sequence_a 2019  -- This line is optional, for checking the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2019_l565_56550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dimensions_volume_condition_l565_56587

/-- Given an isosceles triangle with volume of rotation v and inscribed circle radius ρ,
    this theorem states the relationship between these values and the triangle's height and base. -/
theorem isosceles_triangle_dimensions (v ρ : ℝ) (hv : v > 0) (hρ : ρ > 0) :
  ∃ (m a : ℝ),
    m > 0 ∧ a > 0 ∧
    m = (3 * v + Real.sqrt (9 * v^2 - 24 * ρ^3 * Real.pi * v)) / (2 * ρ^2 * Real.pi) ∧
    2 * a = (3 * v - Real.sqrt (9 * v^2 - 24 * ρ^3 * Real.pi * v)) / (ρ * Real.pi) ∧
    v > 8 * ρ^3 * Real.pi / 3 :=
by
  sorry

/-- The volume condition for the existence of the triangle -/
theorem volume_condition (v ρ : ℝ) (hv : v > 0) (hρ : ρ > 0) :
  v > 8 * ρ^3 * Real.pi / 3 →
  9 * v^2 > 24 * ρ^3 * Real.pi * v :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dimensions_volume_condition_l565_56587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_price_decrease_l565_56502

/-- Calculates the percent decrease in price per notebook during a sale -/
theorem notebook_price_decrease (original_price original_quantity sale_price sale_quantity : ℚ) :
  original_price = 10 →
  original_quantity = 6 →
  sale_price = 9 →
  sale_quantity = 8 →
  let original_unit_price := original_price / original_quantity
  let sale_unit_price := sale_price / sale_quantity
  let percent_decrease := (original_unit_price - sale_unit_price) / original_unit_price * 100
  (percent_decrease ≥ 32 ∧ percent_decrease ≤ 34) := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_price_decrease_l565_56502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spaceship_passengers_at_125_speed_l565_56522

/-- Represents the speed of a spaceship given the number of people on board. -/
noncomputable def spaceshipSpeed (people : ℕ) : ℝ :=
  500 * (1/2) ^ ((people - 200) / 100 : ℝ)

/-- Theorem stating that when the spaceship's speed is 125 km/hr, there are 400 people on board. -/
theorem spaceship_passengers_at_125_speed :
  spaceshipSpeed 400 = 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spaceship_passengers_at_125_speed_l565_56522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_MN_l565_56501

-- Define the circle passing through points A, B, and C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (1, -7)

-- Define the y-coordinates of points M and N
noncomputable def M : ℝ := -2 + 2 * Real.sqrt 6
noncomputable def N : ℝ := -2 - 2 * Real.sqrt 6

-- Theorem statement
theorem length_of_MN :
  abs (M - N) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_MN_l565_56501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_fraction_l565_56580

/-- Proves that a train moving at a reduced speed resulting in a 30-minute delay 
    on a journey that usually takes 3 hours is traveling at 6/7 of its usual speed. -/
theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) :
  usual_time = 3 →
  delay = 0.5 →
  (usual_time / (usual_time + delay)) = 6 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_fraction_l565_56580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l565_56599

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the eccentricity
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (he : Eccentricity a b = Real.sqrt 3 / 2)
  (hf : 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 3) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / 4 + (y t)^2 = 1) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁^2 / a^2 + y₁^2 / b^2 = 1) →
    (x₂^2 / a^2 + y₂^2 / b^2 = 1) →
    (y₁ / x₁) * (y₂ / x₂) = -1/4 →
    x₁^2 + x₂^2 = 4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l565_56599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_y_in_terms_of_a_and_b_l565_56512

theorem tan_y_in_terms_of_a_and_b 
  (a b : ℝ) 
  (y : ℝ)
  (h1 : Real.sin y = 2 * a * b / (a^2 + b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < y)
  (h5 : y < Real.pi/2) :
  Real.tan y = 2 * a * b / (a^2 - b^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_y_in_terms_of_a_and_b_l565_56512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l565_56551

/-- The function f(x, k) defined in the problem -/
noncomputable def f (x k : ℝ) : ℝ := x + 1/x + 1/(x + k/x)

/-- Theorem stating the minimum value of f(x, k) -/
theorem min_value_of_f (k : ℝ) (hk : k > 0) :
  ∃ (min : ℝ), min = (4*k + 1) / (2 * Real.sqrt k) ∧
    ∀ (x : ℝ), x > 0 → f x k ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l565_56551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_a_values_l565_56521

theorem max_min_a_values (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares_eq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  (∀ x, x ≤ 2 ∧ (x = a → (∃ b' c' d', a + b' + c' + d' = 3 ∧ a^2 + 2*b'^2 + 3*c'^2 + 6*d'^2 = 5))) ∧
  (∀ x, x ≥ 1 ∧ (x = a → (∃ b' c' d', a + b' + c' + d' = 3 ∧ a^2 + 2*b'^2 + 3*c'^2 + 6*d'^2 = 5))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_a_values_l565_56521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l565_56523

def initial_speed : ℕ := 55
def speed_increase : ℕ := 2
def total_hours : ℕ := 12

def speed (hour : ℕ) : ℕ := initial_speed + speed_increase * (hour - 1)

def total_distance : ℕ := Finset.sum (Finset.range total_hours) (fun hour => speed (hour + 1))

theorem car_distance_theorem : total_distance = 792 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l565_56523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l565_56563

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi)
  (h3 : ∀ x, f x φ = f (4*Real.pi/3 - x) φ) :
  (∀ x ∈ Set.Ioo 0 (5*Real.pi/12), 
    StrictMonoOn (fun x => -(f x φ)) (Set.Ioo 0 (5*Real.pi/12))) ∧
  (∃ x, HasDerivAt (fun x => f x φ) (-1) x ∧ 
        f x φ = Real.sqrt 3 / 2 - x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l565_56563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_distance_property_l565_56509

/-- A convex pentagon in 2D space -/
structure ConvexPentagon where
  vertices : Fin 5 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- The distance from a point to a line segment -/
noncomputable def distanceToSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem convex_pentagon_distance_property (P : ConvexPentagon) :
  ∃ i : Fin 5,
    let v := P.vertices
    let opposite_side := (v ((i + 2) % 5), v ((i + 3) % 5))
    distanceToSegment (v i) opposite_side.1 opposite_side.2 <
      distanceToSegment (v ((i + 1) % 5)) opposite_side.1 opposite_side.2 +
      distanceToSegment (v ((i + 4) % 5)) opposite_side.1 opposite_side.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_distance_property_l565_56509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l565_56565

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 4) > 4 / x + 27 / 10) ↔ x ∈ Set.Ioo (-5/8 : ℝ) 0 ∪ Set.Ioo (0 : ℝ) (2/5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l565_56565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l565_56559

-- Define the curves in polar coordinates
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 4
def C₂ (ρ θ : ℝ) : Prop := ρ^2 - 2*ρ*(Real.cos θ) - 4*ρ*(Real.sin θ) + 1 = 0
def C₃ (ρ θ : ℝ) : Prop := θ = Real.pi/4

-- Define the intersection points
noncomputable def P : ℝ × ℝ := (1, 4)
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State that P is on C₁ and C₂
axiom P_on_C₁ : C₁ P.1 P.2
axiom P_on_C₂ : C₂ P.1 P.2

-- State that A and B are on C₂ and C₃
axiom A_on_C₂ : C₂ A.1 A.2
axiom A_on_C₃ : C₃ A.1 A.2
axiom B_on_C₂ : C₂ B.1 B.2
axiom B_on_C₃ : C₃ B.1 B.2

-- Theorem statement
theorem area_of_triangle_PAB : 
  Real.sqrt 7 * (3/2) = (1/2) * Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) * 
                               Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) * 
                               Real.sin (Real.arctan ((B.2 - A.2)/(B.1 - A.1)) - 
                                         Real.arctan ((P.2 - A.2)/(P.1 - A.1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l565_56559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l565_56576

/-- The time taken for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_mph : ℝ) : ℝ :=
  let train_speed_fps := train_speed_mph * 5280 / 3600
  train_length / train_speed_fps

/-- Theorem: A train 2500 feet long traveling at 120 mph takes approximately 14.2045 seconds to pass a stationary point -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 2500 120 - 14.2045| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l565_56576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inequality_l565_56593

theorem cyclic_inequality (x y z l : ℝ) (k : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_l : 0 < l) 
  (h_k : k ≥ 1) (h_sum : x + y + z = 3) :
  (x^k / ((l + y) * (l + z))) + (y^k / ((l + z) * (l + x))) + (z^k / ((l + x) * (l + y))) ≥ 3 / (l + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inequality_l565_56593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_lower_bound_l565_56544

theorem sin_lower_bound (x : Real) (h : 0 < x ∧ x < Real.pi / 2) : Real.sin x > x - x^3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_lower_bound_l565_56544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_property_l565_56555

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 4 * x

theorem g_inverse_property (c d : ℝ) : 
  (∀ x, g c d (g c d x) = x) → c + d = 2.25 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_property_l565_56555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l565_56538

open Set
open Function
open Real

/-- The function f(x) = 1/x^2 -/
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2)

/-- The domain of f(x) -/
def domain : Set ℝ := {x | x < 0 ∨ x > 0}

theorem f_monotone_increasing :
  MonotoneOn f (Iio 0) := by
  sorry

#check f_monotone_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l565_56538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_xg_l565_56582

-- Define the function g(x) = √(x(1-x)) on the interval [0,1]
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (1 - x))

-- Define the domain of the function
def I : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem area_enclosed_by_xg :
  ∫ x in I, x * g x = π / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_xg_l565_56582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_area_theorem_l565_56556

/-- Represents a circle on a unit grid -/
structure GridCircle where
  radius : ℝ
  center : ℝ × ℝ

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : GridCircle) : ℝ := Real.pi * c.radius^2

/-- Counts the number of complete grid cells inside the circle -/
def completeGridCellsCount (c : GridCircle) : ℕ := sorry

/-- Calculates the area of complete grid cells inside the circle -/
def completeGridCellsArea (c : GridCircle) : ℝ := 
  (completeGridCellsCount c : ℝ)

/-- Main theorem: The area of complete grid cells inside a circle with 
    radius 1000 is at least 99% of the circle's area -/
theorem grid_circle_area_theorem (c : GridCircle) 
  (h : c.radius = 1000) : 
  completeGridCellsArea c ≥ 0.99 * circleArea c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_area_theorem_l565_56556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_6474_l565_56517

def concatenate (a b c : ℕ) : List ℕ :=
  (toString a ++ toString b ++ toString c).data.map (λ c => c.toNat - '0'.toNat)

def contains_subseq (l s : List ℕ) : Prop :=
  ∃ i, l.drop i = s ++ l.drop (i + s.length)

theorem smallest_n_with_6474 :
  ∀ n : ℕ, n < 46 →
    ¬(contains_subseq (concatenate n (n+1) (n+2)) [6,4,7,4]) ∧
    (contains_subseq (concatenate 46 47 48) [6,4,7,4]) :=
by sorry

#eval concatenate 46 47 48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_6474_l565_56517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_graph_intermediate_value_theorem_l565_56519

-- Define the function for proposition ④
def f (x : ℝ) : ℝ := 3 * x^2

-- Define the shifted function for proposition ④
def g (x : ℝ) : ℝ := 3 * (x - 1)^2

-- Theorem for proposition ④
theorem shift_graph : ∀ x : ℝ, g x = f (x - 1) := by sorry

-- Theorem for proposition ⑤
theorem intermediate_value_theorem {f : ℝ → ℝ} {a b : ℝ} 
  (h1 : ContinuousOn f (Set.Icc a b)) (h2 : a < b) (h3 : f a * f b < 0) :
  ∃ c ∈ Set.Icc a b, f c = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_graph_intermediate_value_theorem_l565_56519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grunters_win_probability_l565_56508

/-- The probability of winning a single game -/
noncomputable def single_game_probability : ℝ := 3 / 5

/-- The number of games played -/
def number_of_games : ℕ := 5

/-- The probability of winning all games -/
noncomputable def win_all_games : ℝ := single_game_probability ^ number_of_games

theorem grunters_win_probability :
  win_all_games = (3 / 5 : ℝ) ^ 5 := by
  -- Unfold the definitions
  unfold win_all_games
  unfold single_game_probability
  -- The rest of the proof
  sorry

#eval (3 / 5 : ℚ) ^ 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grunters_win_probability_l565_56508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drama_club_skills_l565_56540

theorem drama_club_skills (total_members : ℕ) 
  (cannot_paint cannot_write cannot_direct : ℕ) : 
  total_members = 120 →
  cannot_paint = 50 →
  cannot_write = 75 →
  cannot_direct = 40 →
  (total_members - cannot_paint) + (total_members - cannot_write) + (total_members - cannot_direct) - total_members = 75 := by
  intro h_total h_paint h_write h_direct
  -- Define intermediate calculations
  let can_paint := total_members - cannot_paint
  let can_write := total_members - cannot_write
  let can_direct := total_members - cannot_direct
  let total_skills := can_paint + can_write + can_direct
  let exactly_two_skills := total_skills - total_members
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drama_club_skills_l565_56540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_conversion_l565_56570

/-- The volume of a cube in cubic feet -/
noncomputable def cube_volume_ft : ℝ := 64

/-- The conversion factor from cubic feet to cubic yards -/
noncomputable def ft_to_yd_conversion : ℝ := 27

/-- The volume of the cube in cubic yards -/
noncomputable def cube_volume_yd : ℝ := cube_volume_ft / ft_to_yd_conversion

theorem cube_volume_conversion :
  cube_volume_yd = 64 / 27 := by
  -- Unfold the definitions
  unfold cube_volume_yd
  unfold cube_volume_ft
  unfold ft_to_yd_conversion
  -- The goal is now to prove 64 / 27 = 64 / 27
  rfl  -- reflexivity solves this trivial equality


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_conversion_l565_56570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_five_l565_56535

/-- A geometric sequence with sum of first n terms S_n -/
structure GeometricSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)

/-- The common ratio of a geometric sequence -/
def common_ratio (seq : GeometricSequence) : ℚ := seq.a 1 / seq.a 0

theorem common_ratio_is_five (seq : GeometricSequence) 
  (h1 : seq.a 5 = 4 * seq.S 4 + 3)
  (h2 : seq.a 6 = 4 * seq.S 5 + 3) : 
  common_ratio seq = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_five_l565_56535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l565_56584

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- A special triangle with the given condition -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 + Real.sqrt 2 * t.a * t.c

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = π/4 ∧ (∀ x : ℝ, Real.sqrt 2 * Real.cos t.A + Real.cos t.C ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l565_56584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_height_problem_l565_56591

/-- The circumference of a circle -/
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem tank_height_problem (h_a : ℝ) :
  let r_a := (circumference⁻¹ 7 : ℝ)
  let r_b := (circumference⁻¹ 10 : ℝ)
  let h_b := (7 : ℝ)
  cylinderVolume r_a h_a = 0.7 * cylinderVolume r_b h_b →
  h_a = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_height_problem_l565_56591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_f_has_two_zeros_l565_56525

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + a + 1

/-- The function g(x) -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - 2 * b * x^2 + b * x - 4 / 27

/-- The composition of g and f -/
noncomputable def g_of_f (a b : ℝ) (x : ℝ) : ℝ := g b (f a x)

/-- Main theorem: g(f(x)) has exactly two zeros -/
theorem g_of_f_has_two_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, g_of_f a b x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_f_has_two_zeros_l565_56525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unsafe_actions_l565_56568

-- Define the basic types
def Agent : Type := String
def Information : Type := String
def Website : Type := String
def File : Type := String

-- Define the properties
axiom is_sensitive : Information → Prop
axiom is_untrusted : Website → Prop
axiom is_unsafe_file : File → Prop

-- Define the actions
axiom shares_on : Agent → Information → Website → Prop
axiom downloads : Agent → File → Website → Prop

-- Define the safety property
axiom is_unsafe_action : Prop

-- State the theorem
theorem unsafe_actions 
  (a : Agent) (i : Information) (w : Website) (f : File) :
  (is_sensitive i ∧ is_untrusted w ∧ shares_on a i w) ∨
  (is_untrusted w ∧ is_unsafe_file f ∧ downloads a f w) →
  is_unsafe_action :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unsafe_actions_l565_56568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l565_56516

/-- The interest rate at which B lends money to C, given the conditions of the problem -/
noncomputable def interest_rate_B_to_C (principal : ℝ) (rate_A_to_B : ℝ) (years : ℕ) (gain_B : ℝ) : ℝ :=
  let interest_A_to_B := principal * rate_A_to_B * (years : ℝ)
  let total_interest_B_to_C := interest_A_to_B + gain_B
  total_interest_B_to_C / (principal * (years : ℝ))

/-- Theorem stating that under the given conditions, B lends money to C at 11% per annum -/
theorem interest_rate_problem (principal : ℝ) (rate_A_to_B : ℝ) (years : ℕ) (gain_B : ℝ)
    (h1 : principal = 3500)
    (h2 : rate_A_to_B = 0.1)
    (h3 : years = 3)
    (h4 : gain_B = 105) :
    interest_rate_B_to_C principal rate_A_to_B years gain_B = 0.11 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues
-- #eval interest_rate_B_to_C 3500 0.1 3 105

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l565_56516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l565_56545

/-- Represents the daily average growth rate of packages received by the express delivery store -/
def x : ℝ := sorry

/-- The number of packages received on the first day -/
def first_day_packages : ℝ := 200

/-- The number of packages received on the third day -/
def third_day_packages : ℝ := 242

/-- The number of days between the first and third day -/
def days_between : ℕ := 2

/-- Theorem stating that the equation 200(1+x)^2 = 242 correctly represents the relationship
    between the number of packages received on the first day, the daily average growth rate,
    and the number of packages received on the third day -/
theorem growth_rate_equation :
  first_day_packages * (1 + x) ^ days_between = third_day_packages :=
by
  sorry

#check growth_rate_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l565_56545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raise_calculation_l565_56564

/-- Calculates the new weekly earnings after a percentage raise --/
noncomputable def new_weekly_earnings (initial_earnings percentage_increase : ℝ) : ℝ :=
  initial_earnings * (1 + percentage_increase / 100)

/-- Proves that given an initial weekly earnings of $60 and a percentage increase of 16.666666666666664%, 
    the new weekly earnings after the raise is $70 --/
theorem raise_calculation :
  new_weekly_earnings 60 16.666666666666664 = 70 := by
  sorry

/-- Evaluates the new weekly earnings for the given values --/
def raise_calculation_eval : ℚ :=
  60 * (1 + 16.666666666666664 / 100)

#eval raise_calculation_eval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raise_calculation_l565_56564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_divisible_l565_56520

theorem polynomial_not_divisible (A : ℤ) (n m : ℕ) :
  ¬ (∃ (p : Polynomial ℤ), (3 * X^(2*n) + A * X^n + 2) = p * (2 * X^(2*m) + A * X^m + 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_divisible_l565_56520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_explicit_l565_56573

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define the center and radius of each circle
def center1 : ℝ × ℝ := (1, 0)
def radius1 : ℝ := 2
def center2 : ℝ × ℝ := (2, -1)
def radius2 : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 2

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > (radius1 - radius2) ∧
  distance_between_centers < (radius1 + radius2) := by
  -- The proof goes here
  sorry

-- Additional theorem to show the circles are indeed intersecting
theorem circles_intersect_explicit :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_explicit_l565_56573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l565_56532

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area S of triangle ABC satisfies S = (√3/4)(a² + c² - b²) -/
noncomputable def areaFormula (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)

theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : areaFormula t = (1/2) * t.a * t.c * Real.sin t.B) : 
  (t.B = π/3) ∧ 
  (∀ (a c : ℝ), (Real.sqrt 3 - 1) * a + 2 * c ≤ 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l565_56532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l565_56598

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem ellipse_triangle_area 
  (e : Ellipse) 
  (m f1 f2 : Point) 
  (h1 : e.a = 5 ∧ e.b = 4) 
  (h2 : isOnEllipse m e) 
  (h3 : distance f1 f2 = 6) 
  (h4 : ∀ p, isOnEllipse p e → distance p f1 + distance p f2 = 2 * e.a) 
  (h5 : Real.arccos ((distance m f1)^2 + (distance m f2)^2 - (distance f1 f2)^2) / (2 * distance m f1 * distance m f2) = π / 6) :
  triangleArea m f1 f2 = 16 * (2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l565_56598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotating_triangle_path_length_l565_56571

/-- The path length of a rotating isosceles right triangle in a rectangle --/
theorem rotating_triangle_path_length :
  ∀ (triangle_side : ℝ) (rect_length rect_width : ℝ),
  triangle_side = 3 →
  rect_length = 6 →
  rect_width = 3 →
  let hypotenuse := Real.sqrt (2 * triangle_side ^ 2)
  let rotation_arc_length := hypotenuse * (π / 2)
  let total_rotations := 6
  total_rotations * rotation_arc_length = 9 * Real.sqrt 2 * π :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotating_triangle_path_length_l565_56571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_even_and_decreasing_l565_56536

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Theorem statement
theorem cos_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f y < f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_even_and_decreasing_l565_56536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l565_56514

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1/4 + Real.log x / Real.log 4 else 2^(-x) - 1/4

-- Theorem statement
theorem f_properties :
  (∀ x, f x ≥ 1/4) ∧ 
  (f 0 = 3/4 ∧ f 2 = 3/4) ∧
  (∀ x, f x = 3/4 → x = 0 ∨ x = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l565_56514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_C_l565_56594

def total_length : ℕ := 45
def total_price : ℕ := 4500
def profit_per_meter_A : ℕ := 14
def profit_per_meter_B : ℕ := 18
def profit_per_meter_C : ℕ := 22
def length_A : ℕ := 15
def length_B : ℕ := 10
def length_C : ℕ := 20

theorem cost_price_C : 
  (total_price / total_length : ℚ) - profit_per_meter_C = 78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_C_l565_56594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_slope_l565_56543

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Area of the part of a circle above a line -/
noncomputable def area_above_line (c : Circle) (x y : ℝ) : ℝ := sorry

/-- Area of the part of a circle below a line -/
noncomputable def area_below_line (c : Circle) (x y : ℝ) : ℝ := sorry

/-- The line that equally divides the areas of the given circles -/
def dividingLine (c1 c2 c3 : Circle) : ℝ × ℝ → Prop :=
  fun p => ∃ m : ℝ, 
    (p.2 - 64 = m * (p.1 - 13)) ∧ 
    (abs m = 8/5) ∧
    (∀ x y : ℝ, y = m * (x - 13) + 64 → 
      (area_above_line c1 x y + area_above_line c2 x y + area_above_line c3 x y = 
       area_below_line c1 x y + area_below_line c2 x y + area_below_line c3 x y))

theorem dividing_line_slope :
  let c1 : Circle := { center := (10, 80), radius := 4 }
  let c2 : Circle := { center := (13, 64), radius := 4 }
  let c3 : Circle := { center := (15, 72), radius := 4 }
  ∃ (m : ℝ), dividingLine c1 c2 c3 (13, 64) ∧ abs m = 8/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_slope_l565_56543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_across_y_axis_l565_56577

/-- Given a point (x, y, z) in a three-dimensional Cartesian coordinate system,
    its reflection across the y-axis is (-x, y, -z). -/
def reflect_y (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-(p.1), p.2.1, -(p.2.2))

/-- The point (3, -4, 1) reflected across the y-axis is (-3, -4, -1). -/
theorem reflection_across_y_axis : reflect_y (3, -4, 1) = (-3, -4, -1) := by
  sorry

#check reflection_across_y_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_across_y_axis_l565_56577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l565_56562

def satisfies_equation (X Y : ℕ+) : Prop :=
  (X : ℚ) / 8 = 8 / (Y : ℚ)

theorem count_ordered_pairs : 
  ∃ s : Finset (ℕ+ × ℕ+), s.card = 7 ∧ ∀ p ∈ s, satisfies_equation p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l565_56562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_division_problem_l565_56592

theorem number_division_problem :
  ∀ a b : ℕ,
    (1 ≤ a ∧ a ≤ 9) →
    (1 ≤ b ∧ b ≤ 9) →
    (140 : ℚ) ≤ (1111 * a : ℚ) / (11 * b) ∧ (1111 * a : ℚ) / (11 * b) < 160 →
    (1111 * a) % (11 * b) = 11 * (a - b) →
    ((a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 4) ∨ (a = 7 ∧ b = 5) ∨ (a = 9 ∧ b = 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_division_problem_l565_56592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_determine_parameters_max_k_for_inequality_l565_56567

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

-- Define the function g
def g (x : ℝ) : ℝ := x - 2

theorem function_values_determine_parameters :
  ∀ a b : ℝ, a > 0 →
  (∀ x ∈ Set.Icc 0 3, f a b x ≤ 3) →
  (∃ x ∈ Set.Icc 0 3, f a b x = 3) →
  (∀ x ∈ Set.Icc 0 3, f a b x ≥ -1) →
  (∃ x ∈ Set.Icc 0 3, f a b x = -1) →
  a = 1 ∧ b = 0 :=
by
  sorry

theorem max_k_for_inequality :
  ∀ k : ℝ,
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, g (3^x) - k * 3^x ≥ 0) ↔
  k ≤ -5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_determine_parameters_max_k_for_inequality_l565_56567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_foci_product_l565_56530

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- F₁ and F₂ are the foci of both the ellipse and hyperbola -/
axiom foci_shared (F₁ F₂ : ℝ × ℝ) : Prop

/-- The distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- The theorem to be proved -/
theorem intersection_point_foci_product (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 → is_on_hyperbola P.1 P.2 → foci_shared F₁ F₂ →
  distance P F₁ * distance P F₂ = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_foci_product_l565_56530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimized_at_neg_21_div_2_l565_56558

/-- The function to be minimized -/
noncomputable def f (c : ℝ) : ℝ := (1/3) * c^2 + 7*c - 4

/-- The theorem stating that f is minimized at c = -21/2 -/
theorem f_minimized_at_neg_21_div_2 :
  ∃ (c_min : ℝ), c_min = -21/2 ∧ ∀ (c : ℝ), f c ≥ f c_min :=
by
  -- The proof is omitted for now
  sorry

#check f_minimized_at_neg_21_div_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimized_at_neg_21_div_2_l565_56558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l565_56581

/-- Circle C with radius 5 centered at the origin -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 25}

/-- Point P -/
def point_P : ℝ × ℝ := (3, 2)

/-- Line l passing through point P with inclination angle π/3 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (3 + t/2, 2 + t*(3^(1/2))/2)}

/-- Intersection points of line l and circle C -/
def intersection_points : Set (ℝ × ℝ) :=
  circle_C ∩ line_l

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2)^(1/2)

/-- Theorem: Product of distances from P to intersection points is 12 -/
theorem intersection_distance_product :
  ∀ A B, A ∈ intersection_points → B ∈ intersection_points → A ≠ B →
  distance point_P A * distance point_P B = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l565_56581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l565_56539

def f (x : ℝ) := x^3

theorem x_zero_value (x₀ : ℝ) (h : (deriv (deriv f)) x₀ = 6) : x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l565_56539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l565_56533

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 3 - ω * x)

theorem omega_value (ω : ℝ) :
  (∃ (p : ℝ), p > 0 ∧ p = 4 * Real.pi ∧ ∀ (x : ℝ), f ω x = f ω (x + p)) →
  ω = 1/2 ∨ ω = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l565_56533
