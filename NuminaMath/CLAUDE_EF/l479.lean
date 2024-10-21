import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_cone_properties_l479_47974

/-- Given a cone with height equal to its base radius and volume 8π/3, and a sphere with surface area
    equal to the lateral surface area of the cone, the volume of the sphere is (4/3)π * ⁴√8. -/
theorem sphere_volume_from_cone_properties (r : ℝ) (R : ℝ) :
  r > 0 →
  (1/3) * π * r^3 = (8*π)/3 →
  π * r * Real.sqrt (2 * r^2) = 4 * π * R^2 →
  (4/3) * π * R^3 = (4/3) * π * (2 : ℝ)^(3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_cone_properties_l479_47974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAM_l479_47915

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define point M
def M : ℝ × ℝ := (1, 0)

-- Define point O (origin)
def O : ℝ × ℝ := (0, 0)

-- Define point A as the intersection of FM and the parabola
noncomputable def A : ℝ × ℝ := (-2 + 2*(Real.sqrt 2), 3 - 2*(Real.sqrt 2))

-- Theorem statement
theorem area_of_triangle_OAM :
  let triangle_area := (1/2) * abs ((A.1 * M.2) - (M.1 * A.2))
  triangle_area = 3/2 - Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAM_l479_47915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_for_specific_widths_l479_47961

/-- The area of overlap between two strips -/
noncomputable def area_of_overlap (α : ℝ) : ℝ := 1 / Real.sin α

/-- Two strips overlapping at an angle -/
structure OverlappingStrips :=
  (α : ℝ)
  (width1 : ℝ)
  (width2 : ℝ)

/-- Theorem stating the area of overlap for specific strip widths -/
theorem overlap_area_for_specific_widths (strips : OverlappingStrips) 
  (h1 : strips.width1 = 1) 
  (h2 : strips.width2 = 2) 
  (h3 : strips.α ≠ 0) 
  (h4 : strips.α ≠ π) :
  area_of_overlap strips.α = 1 / Real.sin strips.α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_for_specific_widths_l479_47961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l479_47905

/-- An ellipse with equation x²/16 + y²/7 = 1 -/
structure Ellipse where
  equation : ℝ → ℝ → Prop
  eq_def : ∀ x y : ℝ, equation x y ↔ x^2/16 + y^2/7 = 1

/-- The left and right foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Points A and B where a line through F₁ intersects the ellipse -/
structure IntersectionPoints (e : Ellipse) (f : Foci e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_ellipse : e.equation A.1 A.2 ∧ e.equation B.1 B.2
  on_line : ∃ t : ℝ, A = f.F₁ + t • (B - f.F₁)

/-- The perimeter of triangle ABF₂ -/
noncomputable def trianglePerimeter (e : Ellipse) (f : Foci e) (p : IntersectionPoints e f) : ℝ :=
  dist p.A p.B + dist p.B f.F₂ + dist f.F₂ p.A

/-- Theorem: The perimeter of triangle ABF₂ is 16 -/
theorem ellipse_triangle_perimeter (e : Ellipse) (f : Foci e) (p : IntersectionPoints e f) :
  trianglePerimeter e f p = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l479_47905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_theta_equal_magnitude_theta_value_l479_47949

open Real Vector

-- Define the vectors a and b
noncomputable def a (θ : ℝ) : Fin 2 → ℝ := ![sin θ, cos θ - 2 * sin θ]
def b : Fin 2 → ℝ := ![2, 1]

-- Theorem for the first part
theorem parallel_vectors_tan_theta :
  ∀ θ : ℝ, (∃ k : ℝ, a θ = k • b) → tan θ = 2/5 := by sorry

-- Theorem for the second part
theorem equal_magnitude_theta_value :
  ∀ θ : ℝ, ‖a θ‖ = ‖b‖ → π/4 < θ → θ < π → θ = 3*π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_theta_equal_magnitude_theta_value_l479_47949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_a_for_non_negative_l479_47997

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

theorem tangent_line_at_zero (a : ℝ) (h : a = 1) :
  let f' := λ x ↦ Real.exp x - Real.cos x
  (f' 0) * 0 + (f 1 0) = 0 := by sorry

theorem max_a_for_non_negative (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_a_for_non_negative_l479_47997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l479_47999

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-3/5 * t + 2, 4/5 * t)

-- Define point M (intersection of line l and x-axis)
def point_M : ℝ × ℝ := (2, 0)

-- Define the maximum distance between M and any point on C
noncomputable def max_distance : ℝ := Real.sqrt 5 + 1

theorem max_distance_MN :
  ∀ θ : ℝ, dist point_M (curve_C θ) ≤ max_distance :=
by sorry

#check max_distance_MN

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l479_47999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_l479_47932

/-- Represents the traffic speed as a function of density -/
noncomputable def v (x : ℝ) : ℝ :=
  if x ≤ 20 then 60
  else if 20 < x ∧ x < 140 then -1/2 * x + 70
  else 0

/-- Represents the traffic flow rate -/
noncomputable def f (x : ℝ) : ℝ := x * v x

/-- Theorem stating the maximum traffic flow rate -/
theorem max_traffic_flow :
  ∃ (x_max : ℝ), x_max = 70 ∧ 
  ∀ (x : ℝ), f x ≤ f x_max ∧ f x_max = 2450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_l479_47932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_grid_coverage_l479_47953

/-- The area of a triangle given its vertices using the Shoelace formula -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- The fraction of an 8x10 grid covered by a triangle with given vertices -/
theorem triangle_grid_coverage (x1 y1 x2 y2 x3 y3 : ℝ) :
  x1 = -2 → y1 = 4 →
  x2 = 3 → y2 = -1 →
  x3 = 6 → y3 = 5 →
  (triangle_area x1 y1 x2 y2 x3 y3) / (8 * 10) = 9 / 32 := by
  sorry

#check triangle_grid_coverage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_grid_coverage_l479_47953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_cut_perimeter_difference_l479_47954

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
noncomputable def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Represents the possible ways to cut the plywood -/
inductive CutDirection
| Vertical
| Horizontal

/-- Calculates the dimensions of a single piece after cutting -/
noncomputable def cutPiece (original : Rectangle) (dir : CutDirection) : Rectangle :=
  match dir with
  | CutDirection.Vertical => { width := original.width / 3, length := original.length }
  | CutDirection.Horizontal => { width := original.width, length := original.length / 3 }

theorem plywood_cut_perimeter_difference :
  let original : Rectangle := { width := 3, length := 9 }
  let verticalCut := cutPiece original CutDirection.Vertical
  let horizontalCut := cutPiece original CutDirection.Horizontal
  let maxPerimeter := max (perimeter verticalCut) (perimeter horizontalCut)
  let minPerimeter := min (perimeter verticalCut) (perimeter horizontalCut)
  maxPerimeter - minPerimeter = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_cut_perimeter_difference_l479_47954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_coordinate_sum_l479_47910

theorem triangle_max_coordinate_sum :
  ∀ (x y : ℝ),
  let D : ℝ × ℝ := (10, 17)
  let E : ℝ × ℝ := (25, 22)
  let F : ℝ × ℝ := (x, y)
  let M : ℝ × ℝ := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  let area := abs ((E.1 - D.1) * (F.2 - D.2) - (F.1 - D.1) * (E.2 - D.2)) / 2
  area = 84 ∧
  (y - M.2) / (x - M.1) = -3 →
  x + y ≤ 943 / 21 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_coordinate_sum_l479_47910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l479_47958

/-- The solution to the equation (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) * x = 3200.0000000000005 -/
noncomputable def solution : ℝ := 1.6

/-- The left-hand side of the equation -/
noncomputable def left_hand_side (x : ℝ) : ℝ := (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) * x

/-- The right-hand side of the equation -/
noncomputable def right_hand_side : ℝ := 3200.0000000000005

/-- Theorem stating that the solution satisfies the equation within any positive error margin -/
theorem solution_satisfies_equation : 
  ∀ ε > 0, |left_hand_side solution - right_hand_side| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l479_47958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l479_47964

/-- An arithmetic sequence with common difference d and sum of first n terms S_n -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d ∧ S (n + 1) = S n + a (n + 1)

/-- The theorem stating the general term of the arithmetic sequence under given conditions -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h1 : ArithmeticSequence a d S)
  (h2 : ∀ n, Real.sqrt (8 * S (n + 1) + 2 * (n + 1 : ℝ)) - Real.sqrt (8 * S n + 2 * n) = d) :
  ∀ n, a n = 4 * n - 9 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l479_47964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_cosine_sum_l479_47993

theorem intersection_points_cosine_sum : ∀ x₁ x₂ x₃ : ℝ,
  0 < x₁ ∧ x₁ < π/2 →
  0 < x₂ ∧ x₂ < π/2 →
  0 < x₃ ∧ x₃ < π/2 →
  Real.sin x₁ = Real.sqrt 3 / 3 →
  Real.cos x₂ = Real.sqrt 3 / 3 →
  Real.tan x₃ = Real.sqrt 3 / 3 →
  Real.cos (x₁ + x₂ + x₃) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_cosine_sum_l479_47993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_color_probability_l479_47902

def num_balls : ℕ := 3

def total_arrangements : ℕ := Nat.factorial (3 * num_balls) / (Nat.factorial num_balls * Nat.factorial num_balls * Nat.factorial num_balls)

def successful_arrangements : ℕ := Nat.factorial 3

theorem alternating_color_probability :
  (successful_arrangements : ℚ) / total_arrangements = 1 / 280 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_color_probability_l479_47902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_of_f_l479_47986

-- Define the function f(x) = 2^(-|x|) - k
noncomputable def f (x k : ℝ) : ℝ := 2^(-abs x) - k

-- Theorem statement
theorem root_range_of_f (k : ℝ) :
  (∃ x, f x k = 0) → k ∈ Set.Ioo 0 1 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_of_f_l479_47986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l479_47971

theorem coefficient_x4_in_expansion : ∃ c : ℝ, 
  c = 122472 * Real.sqrt 2 ∧
  c = (Finset.range 10).sum (λ k ↦ if k = 5 then (Nat.choose 9 k : ℝ) * (3 * Real.sqrt 2) ^ k else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l479_47971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_tenth_l479_47903

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The ratio in question -/
noncomputable def ratio : ℝ := 8 / 12

theorem ratio_rounded_to_tenth :
  round_to_tenth ratio = 0.7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_tenth_l479_47903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_for_composite_sequence_l479_47950

theorem existence_of_k_for_composite_sequence : ∃ k : ℕ, ∀ n : ℕ, n > 0 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = k * 2^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_for_composite_sequence_l479_47950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l479_47919

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (3 * x^2 - a * x + 5) / Real.log (1/2)

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y → f a y < f a x) →
  -8 ≤ a ∧ a ≤ -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l479_47919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_guard_nights_l479_47977

/-- Represents a set of sons -/
def Sons := Fin 8

/-- Represents a night's guard duty -/
def Guard := Finset Sons

/-- The property that no two sons guard together more than once -/
def NoRepeatGuards (schedule : List Guard) : Prop :=
  ∀ (g1 g2 : Guard) (s1 s2 : Sons), 
    g1 ∈ schedule → g2 ∈ schedule → g1 ≠ g2 → 
    s1 ∈ g1.val → s2 ∈ g1.val → s1 ∈ g2.val → s2 ∈ g2.val → s1 ≠ s2 → False

/-- The theorem stating the maximum number of nights -/
theorem max_guard_nights :
  ∃ (schedule : List Guard), 
    (∀ g ∈ schedule, g.card = 3) ∧ 
    NoRepeatGuards schedule ∧
    schedule.length = 8 ∧
    (∀ (schedule' : List Guard), 
      (∀ g ∈ schedule', g.card = 3) → 
      NoRepeatGuards schedule' → 
      schedule'.length ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_guard_nights_l479_47977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_parametric_equation_l479_47946

-- Define the polar equations of C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def C₂ (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/4) = 2 * Real.sqrt 2

-- Define the parametric equation of line PQ
noncomputable def PQ (t a b : ℝ) : ℝ × ℝ := (t^3 + a, b/2 * t^3 + 1)

-- Define P as the center of C₁
def P : ℝ × ℝ := (0, 2)

-- Statement of the theorem
theorem intersection_and_parametric_equation :
  ∃ (a b : ℝ),
    -- Intersection points of C₁ and C₂
    (∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
      C₁ ρ₁ θ₁ ∧ C₂ ρ₁ θ₁ ∧
      C₁ ρ₂ θ₂ ∧ C₂ ρ₂ θ₂ ∧
      ((ρ₁, θ₁) = (4, Real.pi/2) ∧ (ρ₂, θ₂) = (2 * Real.sqrt 2, Real.pi/4)) ∨
      ((ρ₁, θ₁) = (2 * Real.sqrt 2, Real.pi/4) ∧ (ρ₂, θ₂) = (4, Real.pi/2))) ∧
    -- Values of a and b in the parametric equation
    a = -1 ∧ b = 2 ∧
    -- Q is the midpoint of the intersection points
    (∃ (Q : ℝ × ℝ), Q.1 = 1 ∧ Q.2 = 3) ∧
    -- PQ is the line through P and Q
    (∃ (t : ℝ), PQ t a b = (1, 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_parametric_equation_l479_47946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_store_analysis_l479_47980

/-- Represents the relationship between price increase and operating income -/
noncomputable def income_ratio (x : ℝ) : ℝ := 1 - (1/1500) * x^2 + (1/300) * x + 1

/-- The condition for non-decreasing operating income -/
def non_decreasing_income (x : ℝ) : Prop := income_ratio x ≥ 1

theorem clothing_store_analysis :
  ∀ (x : ℝ),
  (∀ (p : ℝ), income_ratio p = 1 - (1/1500) * p^2 + (1/300) * p + 1) ∧
  (non_decreasing_income x ↔ 0 ≤ x ∧ x ≤ 5) ∧
  (∀ (p : ℝ), income_ratio p ≤ income_ratio 2.5) :=
by sorry

#check clothing_store_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_store_analysis_l479_47980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_vertical_asymptote_l479_47990

/-- The function f(x) parameterized by c -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - x + c) / (x^2 + 3*x - 18)

/-- Theorem stating that f(x) has exactly one vertical asymptote iff c = -6 or c = -42 -/
theorem f_one_vertical_asymptote (c : ℝ) :
  (∃! x, (x^2 + 3*x - 18 = 0 ∧ x^2 - x + c ≠ 0)) ↔ (c = -6 ∨ c = -42) := by
  sorry

#check f_one_vertical_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_vertical_asymptote_l479_47990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_674m_value_l479_47944

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def sum_floor (m : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_floor m n + floor (m + (n + 1) / 2022)

theorem floor_674m_value (m : ℝ) 
  (h1 : 0 < m) (h2 : m < 1) 
  (h3 : sum_floor m 2021 = 21) : 
  floor (674 * m) = 6 ∨ floor (674 * m) = 7 := by
  sorry

#check floor_674m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_674m_value_l479_47944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_circle_equation_l479_47991

noncomputable section

-- Define the slope of the reference line x-2y=0
def reference_slope : ℝ := 1/2

-- Define the point that the line passes through
def point : ℝ × ℝ := (2, 3)

-- Define the three points for the circle
def point_O : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (4, 2)

-- Theorem for the line equation
theorem line_equation : 
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, A*x + B*y + C = 0 ↔ 
    (y - point.2 = reference_slope * (x - point.1))) ∧
  A = 1 ∧ B = -2 ∧ C = 4 :=
sorry

-- Theorem for the circle equation
theorem circle_equation :
  ∃ (h k r : ℝ), 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔
    ((x = point_O.1 ∧ y = point_O.2) ∨
     (x = point_A.1 ∧ y = point_A.2) ∨
     (x = point_B.1 ∧ y = point_B.2))) ∧
  h = 4 ∧ k = -3 ∧ r^2 = 25 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_circle_equation_l479_47991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l479_47912

noncomputable def equilateralTriangleArea (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

noncomputable def areaRatio (largeSide smallSide : ℝ) : ℝ :=
  let smallArea := equilateralTriangleArea smallSide
  let largeArea := equilateralTriangleArea largeSide
  let trapezoidArea := largeArea - smallArea
  smallArea / trapezoidArea

theorem equilateral_triangle_area_ratio :
  areaRatio 12 6 = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l479_47912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_five_in_thirty_factorial_l479_47957

theorem exponent_of_five_in_thirty_factorial :
  (Finset.prod (Finset.range 31) (λ k => k)).factorization 5 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_five_in_thirty_factorial_l479_47957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approx_l479_47992

/-- Calculates the speed in miles per hour given a distance in feet and time in seconds -/
noncomputable def speed_mph (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_feet / 5280) / (time_seconds / 3600)

/-- Theorem stating that an object traveling 300 feet in 5 seconds has a speed of approximately 40.9091 mph -/
theorem object_speed_approx : 
  let ε := 0.0001
  |speed_mph 300 5 - 40.9091| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approx_l479_47992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_seven_pi_sixths_l479_47951

theorem sin_alpha_plus_seven_pi_sixths (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = 4*Real.sqrt 3/5) : 
  Real.sin (α + 7*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_seven_pi_sixths_l479_47951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l479_47908

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 160

/-- The time taken for the train to pass a stationary point, in seconds -/
noncomputable def crossing_time : ℝ := 12

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := train_length / crossing_time

theorem train_speed_calculation : 
  ∃ ε > 0, |train_speed - 13.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l479_47908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_sum_l479_47937

theorem root_interval_sum (f : ℝ → ℝ) (a b : ℕ) (x₀ : ℝ) : 
  (∀ x, f x = Real.log x + 3 * x - 8) →
  (∃ x₀, f x₀ = 0 ∧ x₀ ∈ Set.Icc (a : ℝ) (b : ℝ)) →
  b - a = 1 →
  0 < a →
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_sum_l479_47937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l479_47987

theorem cos_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : Real.cos α = Real.sqrt 2 / 10)
  (h2 : α ∈ Set.Ioo (-Real.pi) 0) : 
  Real.cos (α - Real.pi/4) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l479_47987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l479_47939

theorem cos_theta_value (θ : ℝ) 
  (h1 : 0 ≤ θ ∧ θ ≤ π/2) 
  (h2 : Real.sin (θ - π/6) = 1/3) : 
  Real.cos θ = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l479_47939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l479_47948

noncomputable section

def a (α : Real) : Fin 2 → Real := fun i => if i = 0 then Real.cos α else Real.sin α
def b : Fin 2 → Real := fun i => if i = 0 then -1/2 else Real.sqrt 3/2

theorem vector_problem (α : Real) (h : 0 ≤ α ∧ α < 2*Real.pi) :
  (∃ k : Real, (∀ i, a α i = k * b i) → α = 2*Real.pi/3 ∨ α = 5*Real.pi/3) ∧
  ((∀ i, (Real.sqrt 3 * a α i + b i) * (a α i - Real.sqrt 3 * b i) = 0) → Real.tan α = Real.sqrt 3/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l479_47948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_percentage_calculation_l479_47989

/-- Calculate the interest percentage for a purchase with a payment plan -/
theorem interest_percentage_calculation (purchase_price down_payment monthly_payment_amount number_of_payments : ℚ)
  (purchase_price_pos : purchase_price > 0)
  (down_payment_pos : down_payment ≥ 0)
  (monthly_payment_amount_pos : monthly_payment_amount > 0)
  (number_of_payments_pos : number_of_payments > 0)
  (h_purchase_price : purchase_price = 112)
  (h_down_payment : down_payment = 12)
  (h_monthly_payment_amount : monthly_payment_amount = 10)
  (h_number_of_payments : number_of_payments = 12) :
  ∃ (interest_percentage : ℚ), 
    abs (interest_percentage - (((down_payment + number_of_payments * monthly_payment_amount - purchase_price) / purchase_price) * 100)) < 1/20 ∧
    abs (interest_percentage - 179/10) < 1/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_percentage_calculation_l479_47989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_with_circles_l479_47906

/-- The inequalities describe a rectangle with circumscribed and inscribed circles -/
theorem rectangle_with_circles (x y : ℝ) :
  (|x| + |y| ≤ Real.sqrt (2 * (x^2 + y^2))) ∧
  (Real.sqrt (2 * (x^2 + y^2)) ≤ Real.sqrt 2 * max (|x|) (|y|)) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (x^2 / a^2 + y^2 / b^2 ≤ 1) ∧
    (∃ (r : ℝ), r > 0 ∧ x^2 + y^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_with_circles_l479_47906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_C1_and_intersection_l479_47926

-- Define the curve C in polar coordinates
noncomputable def C (θ : Real) : Real × Real :=
  (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

-- Define point A
def A : Real × Real := (1, 0)

-- Define the transformation from M to P
noncomputable def M_to_P (M : Real × Real) : Real × Real :=
  (Real.sqrt 2 * (M.1 - A.1) + A.1, Real.sqrt 2 * (M.2 - A.2) + A.2)

-- State the theorem
theorem locus_C1_and_intersection : 
  ∀ θ : Real, 
  let P := M_to_P (C θ)
  (P.1 - (3 - Real.sqrt 2))^2 + P.2^2 = 4 ∧
  ¬∃ x y : Real, (x - Real.sqrt 2)^2 + y^2 = 2 ∧ (x - (3 - Real.sqrt 2))^2 + y^2 = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_C1_and_intersection_l479_47926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_infinite_l479_47930

-- Define the set A
def A : Set ℝ := sorry

-- Define the properties of A
axiom A_subset : A ⊆ {x : ℝ | 0 ≤ x ∧ x < 1}
axiom A_has_four_elements : ∃ (a b c d : ℝ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom A_closed_under_product_sum : ∀ (a b c d : ℝ), a ∈ A → b ∈ A → c ∈ A → d ∈ A → a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → (a * b + c * d) ∈ A

-- Theorem statement
theorem A_is_infinite : Set.Infinite A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_infinite_l479_47930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_in_second_car_l479_47901

-- Define the type for passengers
inductive Passenger : Type where
  | Jess
  | Lisa
  | Mark
  | Nina
  | Owen

-- Define the type for car positions
inductive Position : Type where
  | First
  | Second
  | Third
  | Fourth
  | Fifth

-- Function to represent the seating arrangement
def seating : Position → Option Passenger := sorry

-- Conditions
axiom five_cars : ∀ p : Position, p = Position.First ∨ p = Position.Second ∨ p = Position.Third ∨ p = Position.Fourth ∨ p = Position.Fifth

axiom nina_last : seating Position.Fifth = some Passenger.Nina

axiom jess_before_owen : ∃ p : Position, 
  (seating p = some Passenger.Jess) ∧ 
  (seating (match p with
    | Position.First => Position.Second
    | Position.Second => Position.Third
    | Position.Third => Position.Fourth
    | Position.Fourth => Position.Fifth
    | Position.Fifth => Position.Fifth
  ) = some Passenger.Owen)

axiom mark_before_jess : ∃ p₁ p₂ : Position,
  (seating p₁ = some Passenger.Mark) ∧
  (seating p₂ = some Passenger.Jess) ∧
  (match p₁, p₂ with
    | Position.First, Position.Second => True
    | Position.First, Position.Third => True
    | Position.First, Position.Fourth => True
    | Position.Second, Position.Third => True
    | Position.Second, Position.Fourth => True
    | Position.Third, Position.Fourth => True
    | _, _ => False
  )

axiom lisa_mark_gap : ∃ p₁ p₂ p₃ : Position,
  (seating p₁ = some Passenger.Lisa) ∧
  (seating p₂ = none) ∧
  (seating p₃ = some Passenger.Mark) ∧
  (match p₁, p₂, p₃ with
    | Position.First, Position.Second, Position.Third => True
    | Position.First, Position.Second, Position.Fourth => True
    | Position.First, Position.Third, Position.Fourth => True
    | Position.Second, Position.Third, Position.Fourth => True
    | Position.Second, Position.Third, Position.Fifth => True
    | Position.Third, Position.Fourth, Position.Fifth => True
    | _, _, _ => False
  )

-- Theorem to prove
theorem mark_in_second_car : seating Position.Second = some Passenger.Mark := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_in_second_car_l479_47901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_languages_l479_47931

/-- Represents a set of languages for a country -/
def LanguageSet := Finset ℕ

/-- The number of countries participating in the olympiad -/
def num_countries : ℕ := 100

/-- The size of subsets that must have a common language -/
def subset_size : ℕ := 20

/-- Checks if the given collection of language sets satisfies the conditions -/
def satisfies_conditions (n : ℕ) (languages : Finset LanguageSet) : Prop :=
  languages.card = num_countries ∧
  (∀ s : LanguageSet, s ∈ languages → s.card = n) ∧
  (∀ subset : Finset LanguageSet, subset ⊆ languages → subset.card = subset_size →
    (∃! lang : ℕ, ∀ s ∈ subset, lang ∈ s.val)) ∧
  (¬ ∃ lang : ℕ, ∀ s ∈ languages, lang ∈ s.val)

/-- The main theorem stating the minimum value of n -/
theorem min_languages :
  ∃ n : ℕ, n > 0 ∧ 
  (∃ languages : Finset LanguageSet, satisfies_conditions n languages) ∧
  (∀ m : ℕ, m < n → ¬∃ languages : Finset LanguageSet, satisfies_conditions m languages) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_languages_l479_47931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_is_225_l479_47927

/-- Marguerite's initial driving distance in miles -/
noncomputable def marguerite_initial_distance : ℝ := 150

/-- Marguerite's initial driving time in hours -/
noncomputable def marguerite_initial_time : ℝ := 3

/-- Marguerite's additional driving time in hours -/
noncomputable def marguerite_additional_time : ℝ := 2

/-- Speed increase after break in miles per hour -/
noncomputable def speed_increase : ℝ := 10

/-- Sam's driving time in hours -/
noncomputable def sam_driving_time : ℝ := 4.5

/-- Calculates the initial speed in miles per hour -/
noncomputable def initial_speed : ℝ := marguerite_initial_distance / marguerite_initial_time

/-- Theorem stating that Sam drove 225 miles -/
theorem sam_distance_is_225 : initial_speed * sam_driving_time = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_is_225_l479_47927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_line_cd_l479_47917

/-- Parabola struct representing y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line represented by x = my + b -/
structure Line where
  m : ℝ
  b : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem parabola_equation_and_line_cd 
  (para : Parabola) 
  (F : Point) -- Focus of the parabola
  (A : Point) 
  (hA : A.y = 2 ∧ A.x > 1) -- A is on the parabola with y = 2 and x > 1
  (hAF : distance A F = 5/2) -- |AF| = 5/2
  (M N : Point) 
  (hM : M.x = -2 ∧ M.y = 0) -- M(-2, 0)
  (hN : N.x = 2 ∧ N.y = 0) -- N(2, 0)
  (C D : Point) -- C and D are intersection points of line CD and the parabola
  (hMCD : triangleArea M C D = 16) -- Area of triangle MCD is 16
  : 
  (∀ (x y : ℝ), y^2 = 2*x ↔ y^2 = 2*para.p*x) ∧ -- Parabola equation is y² = 2x
  (∃ (l : Line), (l.m = 2*Real.sqrt 3 ∨ l.m = -2*Real.sqrt 3) ∧ l.b = 2 ∧ 
    (∀ (x y : ℝ), x = l.m * y + l.b ↔ (x = C.x ∧ y = C.y) ∨ (x = D.x ∧ y = D.y))) -- Line CD equation
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_line_cd_l479_47917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l479_47934

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem point_distance_to_line (m : ℝ) : 
  distance_point_to_line 3 m 1 1 (-4) = Real.sqrt 2 → m = 3 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l479_47934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_999_9951_to_hundredth_l479_47914

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_999_9951_to_hundredth :
  roundToHundredth 999.9951 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_999_9951_to_hundredth_l479_47914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_contained_in_plane_l479_47941

/-- A structure representing a 3D space with planes, lines, and points -/
structure Space3D where
  Plane : Type
  Line : Type
  Point : Type

/-- Parallel relation between planes -/
def parallel_planes (S : Space3D) (α β : S.Plane) : Prop := sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (S : Space3D) (a : S.Line) (α : S.Plane) : Prop := sorry

/-- Containment relation of a point in a plane -/
def point_in_plane (S : Space3D) (P : S.Point) (α : S.Plane) : Prop := sorry

/-- Containment relation of a point on a line -/
def point_on_line (S : Space3D) (P : S.Point) (a : S.Line) : Prop := sorry

/-- Containment relation of a line in a plane -/
def line_in_plane (S : Space3D) (a : S.Line) (α : S.Plane) : Prop := sorry

/-- Theorem: If plane α is parallel to plane β, point P is in α, 
    line a is parallel to β, and P is on a, then a is contained in α -/
theorem line_contained_in_plane 
  (S : Space3D) 
  (α β : S.Plane) 
  (a : S.Line) 
  (P : S.Point) 
  (h1 : parallel_planes S α β) 
  (h2 : point_in_plane S P α) 
  (h3 : parallel_line_plane S a β) 
  (h4 : point_on_line S P a) : 
  line_in_plane S a α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_contained_in_plane_l479_47941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_min_t_value_l479_47916

noncomputable section

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the distance from focus to directrix
def focus_directrix_distance : ℝ := 1/2

-- Define the parameter p
def p : ℝ := focus_directrix_distance

-- Define the point P on the parabola
def point_P (t : ℝ) : ℝ × ℝ := (t, t^2)

-- Define the point Q on the parabola
def point_Q (x : ℝ) : ℝ × ℝ := (x, x^2)

-- Define the point M on the x-axis
def point_M (x₀ : ℝ) : ℝ × ℝ := (x₀/2, 0)

-- Define the point N on the parabola
def point_N (x₀ : ℝ) : ℝ × ℝ := (x₀, x₀^2)

-- Theorem for the equation of the parabola
theorem parabola_equation : parabola p x y ↔ x^2 = y := by sorry

-- Theorem for the minimum value of t
theorem min_t_value : 
  ∀ t : ℝ, t > 0 → (∃ x x₀ : ℝ, 
    parabola p (point_P t).1 (point_P t).2 ∧
    parabola p (point_Q x).1 (point_Q x).2 ∧
    parabola p (point_N x₀).1 (point_N x₀).2 ∧
    -- MN is tangent to C
    (point_N x₀).2 - (point_M x₀).2 = 2*x₀*((point_N x₀).1 - (point_M x₀).1)) →
  t ≥ 2/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_min_t_value_l479_47916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_common_point_min_a_for_common_tangent_min_a_is_one_l479_47938

open Real

-- Define the functions
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b * log x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

-- Part 1
theorem tangent_line_at_common_point (a b : ℝ) :
  (f b 1 = 0) ∧ (g a 1 = 0) ∧ 
  (deriv (f b) 1 = deriv (g a) 1) →
  a = 1 ∧ b = 1 := by sorry

-- Part 2
theorem min_a_for_common_tangent (a : ℝ) :
  a > 0 →
  (∀ t, 0 < t ∧ t < exp 1 → 
    ∃ x, deriv (f 1) t * (x - t) + f 1 t = g a x) →
  a ≥ 1 := by sorry

theorem min_a_is_one :
  ∃ a, a > 0 ∧
  (∀ t, 0 < t ∧ t < exp 1 → 
    ∃ x, deriv (f 1) t * (x - t) + f 1 t = g a x) ∧
  (∀ a', a' > 0 ∧
    (∀ t, 0 < t ∧ t < exp 1 → 
      ∃ x, deriv (f 1) t * (x - t) + f 1 t = g a' x) →
    a ≤ a') := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_common_point_min_a_for_common_tangent_min_a_is_one_l479_47938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_range_l479_47945

noncomputable section

-- Define the points A, B, and F
def A : ℝ × ℝ := (-Real.sqrt 2, 0)
def B : ℝ × ℝ := (Real.sqrt 2, 0)
def F : ℝ × ℝ := (1, 0)

-- Define the trajectory C
def C (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1 ∧ x ≠ Real.sqrt 2 ∧ x ≠ -Real.sqrt 2

-- Define the condition for the moving point E
def E_condition (x y : ℝ) : Prop :=
  (y / (x + Real.sqrt 2)) * (y / (x - Real.sqrt 2)) = -1/2

-- Define the theorem
theorem trajectory_and_range :
  (∀ x y : ℝ, E_condition x y ↔ C x y) ∧
  (∀ l : ℝ → ℝ → Prop,
    (∃ M N : ℝ × ℝ, M ≠ N ∧ l M.1 M.2 ∧ l N.1 N.2 ∧ C M.1 M.2 ∧ C N.1 N.2 ∧ l F.1 F.2) →
    (∀ P : ℝ × ℝ, (P = F ∨
      (P.1 = 0 ∧ ∃ M N : ℝ × ℝ, M ≠ N ∧ C M.1 M.2 ∧ C N.1 N.2 ∧ l M.1 M.2 ∧ l N.1 N.2 ∧
        Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) = Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2))) →
      -Real.sqrt 2 / 4 ≤ P.2 ∧ P.2 ≤ Real.sqrt 2 / 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_range_l479_47945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l479_47984

theorem angle_between_median_and_bisector (β : ℝ) (φ : ℝ) : 
  -- Given conditions
  0 < β ∧ β < π/2 ∧  -- β is an acute angle
  Real.tan (β/2) = 1 / Real.sqrt (Real.sqrt 2) →
  -- Theorem to prove
  Real.tan φ = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l479_47984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l479_47923

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the asymptote of the hyperbola
def asymptote (a b x y : ℝ) : Prop := b * x - a * y = 0

-- Define the chord length
def chord_length (l : ℝ) : Prop := l = 2 * Real.sqrt 3

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ 
              asymptote a b x y ∧ 
              circle_eq x y ∧ 
              (∃ l : ℝ, chord_length l)) →
  (∃ e : ℝ, eccentricity e) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l479_47923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_inequality_solution_set_l479_47967

-- Problem 1
theorem calculation_proof : 
  (8 : Real)^(2/3) - (-7/8 : Real)^0 + (3 - Real.pi)^(1/4) + (((-2) : Real)^6)^(1/2) = Real.pi + 8 := by
  sorry

-- Problem 2
theorem inequality_solution_set (x : Real) : 
  (2*x - 1) / (3 - 4*x) ≥ 1 ↔ x ∈ Set.Icc (2/3) (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_inequality_solution_set_l479_47967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_cyclist_meeting_point_l479_47920

/-- Represents a point in the journey -/
inductive Point
| A
| B
| C

/-- Represents a participant in the journey -/
inductive Participant
| Car
| Bus
| Cyclist

/-- Represents a time in hours since midnight -/
noncomputable def Time := ℝ

structure Journey where
  distance_AB : ℝ
  car_start_time : Time
  bus_arrival_time : Time
  cyclist_arrival_time : Time
  car_passes_C : ℝ  -- Fraction of journey where car passes C

/-- Calculate the speed of a participant given distance and time -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Calculate the distance traveled given speed and time -/
noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- The main theorem to prove -/
theorem bus_cyclist_meeting_point (j : Journey) 
  (h1 : j.distance_AB = 10)
  (h2 : j.car_start_time = (7 : ℝ))
  (h3 : j.bus_arrival_time = (9 : ℝ))
  (h4 : j.cyclist_arrival_time = (10 : ℝ))
  (h5 : j.car_passes_C = 2/3) :
  ∃ (meeting_point : ℝ), 
    meeting_point = 180/26 ∧ 
    0 ≤ meeting_point ∧ 
    meeting_point ≤ j.distance_AB := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_cyclist_meeting_point_l479_47920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_binomial_coefficient_l479_47900

theorem max_binomial_coefficient (a b : ℝ) :
  let n : ℕ := 41
  let ratio := (n.choose 13) / (n.choose 14)
  ratio = 1/2 →
  ∃ k ∈ ({21, 22} : Finset ℕ), ∀ i ∈ Finset.range (n + 1), i ≠ k → n.choose i ≤ n.choose k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_binomial_coefficient_l479_47900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ping_pong_table_area_xiu_zhou_district_area_grand_canal_length_farmers_painting_exhibition_hall_area_l479_47940

-- Define the units
inductive MeasurementUnit
| SquareMeter
| SquareKilometer
| Kilometer

-- Define a function to determine the appropriate unit
def appropriateUnit (measurement : ℝ) (context : String) : MeasurementUnit :=
  sorry

-- Theorem statements
theorem ping_pong_table_area :
  appropriateUnit 4 "ping-pong table area" = MeasurementUnit.SquareMeter :=
by sorry

theorem xiu_zhou_district_area :
  appropriateUnit 580 "Xiu Zhou district area" = MeasurementUnit.SquareKilometer :=
by sorry

theorem grand_canal_length :
  appropriateUnit 1797 "Grand Canal length" = MeasurementUnit.Kilometer :=
by sorry

theorem farmers_painting_exhibition_hall_area :
  appropriateUnit 3000 "Xiu Zhou Farmers' Painting Exhibition Hall area" = MeasurementUnit.SquareMeter :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ping_pong_table_area_xiu_zhou_district_area_grand_canal_length_farmers_painting_exhibition_hall_area_l479_47940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_positivity_condition_l479_47985

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x + a) - Real.log x

-- State the theorems
theorem extreme_value_condition (a : ℝ) :
  (∀ x > 0, HasDerivAt (λ x => f a x) ((deriv (λ x => f a x)) x) x) →
  deriv (λ x => f a x) 1 = 0 → a = -1 := by sorry

theorem positivity_condition (a : ℝ) :
  a ≥ -2 → ∀ x > 0, f a x > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_positivity_condition_l479_47985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l479_47956

-- Define the slope of a line given its equation ax + by = c
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

-- Define the angle from the slope
noncomputable def angle_from_slope (m : ℝ) : ℝ := Real.arctan m

-- State the theorem
theorem line_slope_angle : 
  angle_from_slope (line_slope 1 (-Real.sqrt 3)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l479_47956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pair_solution_l479_47907

theorem ordered_pair_solution (x y : ℤ) : 
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = x + y * (1 / Real.cos (40 * π / 180)) →
  x = 2 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pair_solution_l479_47907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_zero_to_three_l479_47911

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) ≥ 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem stating that A ∩ B is equal to [0, 3)
theorem intersection_equals_zero_to_three :
  A_intersect_B = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_zero_to_three_l479_47911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_product_degree_l479_47943

/-- The degree of a polynomial product (x^5 + px^8 + qx + r)(x^4 + sx^3 + tx + u)(x^2 + vx + w) -/
theorem polynomial_product_degree
  (p q r s t u v w : ℚ)
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (hv : v ≠ 0) (hw : w ≠ 0) :
  Polynomial.degree ((X ^ 5 + p • X ^ 8 + q • X + Polynomial.C r) *
    (X ^ 4 + s • X ^ 3 + t • X + Polynomial.C u) *
    (X ^ 2 + v • X + Polynomial.C w)) = 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_product_degree_l479_47943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_distribution_l479_47918

/-- Represents the number of questionnaires drawn from each unit -/
structure QuestionnaireDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- Convert QuestionnaireDistribution to a function Fin 4 → ℕ -/
def QuestionnaireDistribution.toFin4 (dist : QuestionnaireDistribution) : Fin 4 → ℕ
  | 0 => dist.a
  | 1 => dist.b
  | 2 => dist.c
  | 3 => dist.d

/-- The main theorem statement -/
theorem questionnaire_distribution
  (seq : Fin 4 → ℕ)
  (h_arithmetic : ∀ i : Fin 3, seq (i + 1) - seq i = seq (i + 2) - seq (i + 1))
  (dist : QuestionnaireDistribution)
  (h_total : dist.a + dist.b + dist.c + dist.d = 100)
  (h_b : dist.b = 20)
  (h_prop : ∀ i : Fin 4, dist.toFin4 i * (seq 3 - seq 0) = seq i * 100) :
  dist.d = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_distribution_l479_47918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_with_plane_l479_47976

/-- An oblique line intersecting a plane -/
structure ObliqueIntersection where
  L : Type*  -- The type of lines
  P : Type*  -- The type of planes
  l : L      -- The oblique line
  p : P      -- The plane
  A : Type*  -- The type of points
  intersectionPoint : A

/-- The angle between a line and a plane -/
def angleBetweenLineAndPlane (oi : ObliqueIntersection) : ℝ := sorry

/-- The angle between two lines -/
def angleBetweenLines (oi : ObliqueIntersection) (m : oi.L) : ℝ := sorry

/-- A line in the plane -/
def lineInPlane (oi : ObliqueIntersection) (m : oi.L) : Prop := sorry

theorem smallest_angle_with_plane (oi : ObliqueIntersection) :
  ∀ m : oi.L, lineInPlane oi m →
    angleBetweenLineAndPlane oi ≤ angleBetweenLines oi m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_with_plane_l479_47976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_collection_total_l479_47982

/-- Given a collection of red, blue, and green balls with a specific ratio and number of green balls,
    prove the total number of balls. -/
theorem ball_collection_total (red blue green total : ℕ) : 
  red + blue + green = total →
  (red : ℚ) / green = 1 / 3 →
  (blue : ℚ) / green = 2 / 3 →
  green = 36 →
  total = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_collection_total_l479_47982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_B_l479_47960

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ :=
  Real.sqrt ((r₁ * Real.cos θ₁ - r₂ * Real.cos θ₂)^2 + (r₁ * Real.sin θ₁ - r₂ * Real.sin θ₂)^2)

/-- The theorem stating the distance between points A and B -/
theorem distance_A_B :
  polar_distance 1 (3 * Real.pi / 4) 2 (Real.pi / 4) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_B_l479_47960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_cosine_value_l479_47925

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) (h1 : ω > 0) (h2 : -π/2 ≤ φ ∧ φ < π/2)
  (h3 : ∀ x, f ω φ (x - π/3) = f ω φ (π/3 - x))
  (h4 : ∀ x, f ω φ x = f ω φ (x + π)) :
  ω = 2 ∧ φ = -π/6 := by
  sorry

theorem cosine_value (ω φ α : ℝ) (h1 : ω = 2) (h2 : φ = -π/6)
  (h3 : f ω φ (α/2) = Real.sqrt 3 / 4) (h4 : π/6 < α ∧ α < 2*π/3) :
  Real.cos (α + 3*π/2) = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_cosine_value_l479_47925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l479_47978

noncomputable def is_increasing_sequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≤ a (n + 1)

noncomputable def sequence_definition (θ : ℝ) (n : ℕ+) : ℝ :=
  (n : ℝ)^2 + 2 * Real.sqrt 3 * Real.sin θ * n

theorem theta_range (a : ℕ+ → ℝ) (θ : ℝ) :
  is_increasing_sequence a ∧
  (∀ n : ℕ+, a n = sequence_definition θ n) ∧
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  θ ∈ Set.Icc 0 ((4 / 3) * Real.pi) ∪ Set.Icc ((5 / 3) * Real.pi) (2 * Real.pi) :=
by
  sorry

#check theta_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l479_47978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l479_47995

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- The focus of a parabola -/
noncomputable def focus (C : Parabola) : ℝ × ℝ := (C.p / 2, 0)

/-- A line with slope 1 passing through a point -/
def line_through (point : ℝ × ℝ) (x : ℝ) : ℝ := x - point.1 + point.2

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The main theorem -/
theorem parabola_chord_theorem (C : Parabola) 
  (A B : ParabolaPoint C)
  (h_line : ∀ x, line_through (focus C) x = A.y ∨ line_through (focus C) x = B.y)
  (h_dist : distance (A.x, A.y) (B.x, B.y) = 4) :
  C.p = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l479_47995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l479_47979

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

theorem min_translation_for_symmetry :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, g m x = -g m (-x)) →
  (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, g m' x = -g m' (-x)) → m' ≥ m) →
  m = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l479_47979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_approximation_l479_47904

-- Define the cyclist's speed in km/h
noncomputable def cyclist_speed : ℝ := 12

-- Define the time the cyclist travels after passing the hiker in hours
noncomputable def cyclist_travel_time : ℝ := 5 / 60

-- Define the time the cyclist waits for the hiker in hours
noncomputable def cyclist_wait_time : ℝ := 10.000000000000002 / 60

-- Define the distance traveled by the cyclist
noncomputable def distance : ℝ := cyclist_speed * cyclist_travel_time

-- Define the hiker's speed
noncomputable def hiker_speed : ℝ := distance / cyclist_wait_time

-- Theorem statement
theorem hiker_speed_approximation : 
  ∃ ε > 0, |hiker_speed - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_approximation_l479_47904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_implication_l479_47952

def i : ℂ := Complex.I

theorem complex_real_implication (a : ℝ) : 
  (a * i - 10 / (3 - i)).im = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_implication_l479_47952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_norm_value_l479_47968

noncomputable section

def vector_2d : Type := ℝ × ℝ

def norm (v : vector_2d) : ℝ := Real.sqrt ((v.1 * v.1) + (v.2 * v.2))

def add_vector (v w : vector_2d) : vector_2d := (v.1 + w.1, v.2 + w.2)

theorem smallest_norm_value (v : vector_2d) : 
  norm (add_vector v (3, -1)) = 8 → norm v ≥ 8 - Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_norm_value_l479_47968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_roots_l479_47963

/-- Given a quadratic equation x^2 - (α - 2)x - α - 1 = 0 with roots p and q,
    the minimum value of p^2 + q^2 is 5. -/
theorem min_sum_squares_roots :
  ∃ (m : ℝ), (∀ (α p q : ℝ), p^2 - (α - 2) * p - (α + 1) = 0 →
                              q^2 - (α - 2) * q - (α + 1) = 0 →
                              p^2 + q^2 ≥ m) ∧
             (∃ (α p q : ℝ), p^2 - (α - 2) * p - (α + 1) = 0 ∧
                              q^2 - (α - 2) * q - (α + 1) = 0 ∧
                              p^2 + q^2 = m) ∧
             m = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_roots_l479_47963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_impossible_case_l479_47959

-- Define the periodic functions and their properties
def periodic_function (f : ℝ → ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x

def minimal_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  periodic_function f ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬ (∀ x, f (x + q) = f x)

-- Define the problem setup
variable (y₁ y₂ y₃ : ℝ → ℝ)
variable (a b t : ℝ)
variable (n : ℕ)

-- State the theorem
theorem one_impossible_case
  (h1 : minimal_positive_period y₁ a)
  (h2 : minimal_positive_period y₂ b)
  (h3 : b = n * a)
  (h4 : n ≥ 2)
  (h5 : y₃ = λ x ↦ y₁ x + y₂ x)
  (h6 : minimal_positive_period y₃ t) :
  ∃! case, case ∈ [t < a, t = a, a < t ∧ t < b, t = b, t > b] ∧ ¬case := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_impossible_case_l479_47959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_zeros_in_reciprocal_of_25_to_25_l479_47962

/-- The number of leading zeros in the decimal representation of 1/(25^25) is 33. -/
theorem leading_zeros_in_reciprocal_of_25_to_25 :
  let n : ℕ := 25
  let x : ℚ := 1 / (n^n : ℚ)
  ∃ (s : String), (∃ (d : ℕ), d ≠ 0 ∧ s.length = 33 ∧ 
    x = (s.toList.map (λ c => 0) ++ [d] ++ (s.toList.map (λ c => 0))).foldl (λ acc d => acc / 10 + d) 0 / 10^(34 + s.length)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_zeros_in_reciprocal_of_25_to_25_l479_47962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_is_one_half_l479_47924

def largest_number_less_than_or_equal_to_point_seven (x : ℝ) : Prop :=
  x ≤ 0.7 ∧ x ∈ ({0.8, 1/2, 0.9, 1/3} : Set ℝ) ∧
  ∀ y ∈ ({0.8, 1/2, 0.9, 1/3} : Set ℝ), y ≤ 0.7 → y ≤ x

theorem largest_number_is_one_half :
  largest_number_less_than_or_equal_to_point_seven (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_is_one_half_l479_47924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l479_47994

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 1)) / (Real.sqrt (6 - x))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 1 ≤ x ∧ x < 6} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l479_47994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_intersection_range_l479_47955

/-- The range of k for which the line y = kx + 2 intersects the right branch
    of the hyperbola x^2 - y^2 = 6 at two distinct points -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 6 ∧
    x₂^2 - y₂^2 = 6) ↔
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_intersection_range_l479_47955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l479_47935

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, -Real.sqrt 3 * t)

-- Define curve C1
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

-- Define curve C2 in polar coordinates
noncomputable def curve_C2_polar (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

-- Define curve C2 in Cartesian coordinates
def curve_C2_cartesian (x y : ℝ) : Prop := x^2 + y^2 = -2*x + 2*Real.sqrt 3*y

-- Define the intersection points A and B
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3 / 2, 1/2)
noncomputable def point_B : ℝ × ℝ := (2, -2*Real.sqrt 3)

-- State the theorem
theorem length_AB : 
  let dist := Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2)
  dist = 4 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l479_47935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_m_sum_l479_47921

-- Define the piecewise function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x^2 + 2*x + 1 else 3*x + 6

-- Theorem statement
theorem continuous_piecewise_function_m_sum :
  ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧
  (∀ x, ContinuousAt (f m₁) x) ∧
  (∀ x, ContinuousAt (f m₂) x) ∧
  m₁ + m₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_m_sum_l479_47921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l479_47928

theorem eigenvalues_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 6; 3, 2]
  ∀ k : ℝ, (∃ v : Fin 2 → ℝ, v ≠ 0 ∧ A.mulVec v = k • v) ↔ (k = 2 + 3 * Real.sqrt 2 ∨ k = 2 - 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l479_47928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l479_47942

def is_valid_number (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧  -- 3-digit positive integer
  (n % 10 ≠ (n / 10) % 10) ∧ 
  (n % 10 ≠ n / 100) ∧ 
  ((n / 10) % 10 ≠ n / 100) ∧  -- all digits are different
  (n / 100 ≠ 0) ∧  -- leading digit is not zero
  (n % 5 = 0) ∧  -- multiple of 5
  (n % 10 ≤ 6) ∧ ((n / 10) % 10 ≤ 6) ∧ (n / 100 ≤ 6) ∧  -- 6 is the largest digit
  (n % 10 = 6 ∨ (n / 10) % 10 = 6 ∨ n / 100 = 6)  -- 6 appears as a digit

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_number n = true) (Finset.range 1000)).card = 45 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n = true) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l479_47942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l479_47970

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (5 * a + 2) ^ (1/3 : ℝ) = 3
def condition2 : Prop := Real.sqrt (3 * a + b - 1) = 4
def condition3 : Prop := c = Int.floor (Real.sqrt 13)

-- Define the theorem
theorem problem_solution (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 c) :
  (a = 5 ∧ b = 2 ∧ c = 3) ∧ (Real.sqrt (3 * a - b + c) = 4 ∨ Real.sqrt (3 * a - b + c) = -4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l479_47970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_computation_l479_47969

theorem vector_computation :
  (3 : ℝ) • (![2, -8] : Fin 2 → ℝ) - (2 : ℝ) • (![1, -7] : Fin 2 → ℝ) = ![4, -10] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_computation_l479_47969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_diff_l479_47972

/-- The ellipse Γ in the Cartesian coordinate system -/
def Γ : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- A point P on Γ in the first quadrant -/
noncomputable def P : ℝ × ℝ := sorry

/-- Q₁ is the intersection of PF₁ and Γ -/
noncomputable def Q₁ : ℝ × ℝ := sorry

/-- Q₂ is the intersection of PF₂ and Γ -/
noncomputable def Q₂ : ℝ × ℝ := sorry

/-- P is on the ellipse Γ -/
axiom hP : P ∈ Γ

/-- P is in the first quadrant -/
axiom hP_quad : P.1 > 0 ∧ P.2 > 0

/-- Q₁ is on the ellipse Γ -/
axiom hQ₁ : Q₁ ∈ Γ

/-- Q₂ is on the ellipse Γ -/
axiom hQ₂ : Q₂ ∈ Γ

/-- The maximum value of y₁ - y₂ is 2√2/3 -/
theorem max_y_diff : 
  ∀ P Q₁ Q₂, P ∈ Γ → P.1 > 0 → P.2 > 0 → Q₁ ∈ Γ → Q₂ ∈ Γ → 
  Q₁.2 - Q₂.2 ≤ 2 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_diff_l479_47972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_paths_l479_47973

/-- The number of paths in an isosceles trapezoid composed of equilateral triangles -/
theorem isosceles_trapezoid_paths (n : ℕ) (h : n > 3) :
  let total_paths := 16*n^3 - 92*n^2 + 166*n - 90
  ∃ (paths_to_E paths_to_Δ paths_to_Γ paths_to_B : ℕ),
    paths_to_E = 8*n^3 - 48*n^2 + 88*n - 48 ∧
    paths_to_Δ = 8*n^3 - 48*n^2 + 88*n - 48 ∧
    paths_to_Γ = 4*n^2 - 12*n + 8 ∧
    paths_to_B = 2*n - 2 ∧
    total_paths = paths_to_E + paths_to_Δ + paths_to_Γ + paths_to_B := by
  sorry

#check isosceles_trapezoid_paths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_paths_l479_47973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_is_four_l479_47929

/-- Represents the food order details and bill splitting information -/
structure FoodOrder where
  oyster_dozens : ℕ
  oyster_price_per_dozen : ℚ
  shrimp_pounds : ℕ
  shrimp_price_per_pound : ℚ
  clam_pounds : ℕ
  clam_price_per_pound : ℚ
  amount_per_person : ℚ

/-- Calculates the number of people in the group based on the food order -/
def calculate_group_size (order : FoodOrder) : ℕ :=
  let total_cost := order.oyster_dozens * order.oyster_price_per_dozen +
                    order.shrimp_pounds * order.shrimp_price_per_pound +
                    order.clam_pounds * order.clam_price_per_pound
  (total_cost / order.amount_per_person).floor.toNat

/-- Theorem stating that the group size is 4 for the given food order -/
theorem group_size_is_four (order : FoodOrder)
  (h1 : order.oyster_dozens = 3)
  (h2 : order.oyster_price_per_dozen = 15)
  (h3 : order.shrimp_pounds = 2)
  (h4 : order.shrimp_price_per_pound = 14)
  (h5 : order.clam_pounds = 2)
  (h6 : order.clam_price_per_pound = 13.5)
  (h7 : order.amount_per_person = 25) :
  calculate_group_size order = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_is_four_l479_47929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersecting_l479_47975

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 9 = 0

-- Define the center and radius of C1
def center_C1 : ℝ × ℝ := (0, 0)
def radius_C1 : ℝ := 3

-- Define the center and radius of C2
def center_C2 : ℝ × ℝ := (4, -3)
def radius_C2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Theorem: The circles are intersecting
theorem circles_intersecting :
  distance_between_centers > abs (radius_C1 - radius_C2) ∧
  distance_between_centers < radius_C1 + radius_C2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersecting_l479_47975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_solution_set_l479_47936

theorem function_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m - |x| ≥ 0 ↔ x ∈ Set.Icc (-3) 3)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_solution_set_l479_47936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l479_47913

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

-- Define the line l
def l (k : ℤ) (x : ℝ) : ℝ := (k - 2 : ℝ) * x - k + 1

theorem problem_statement :
  -- Part 1
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f a x₀ > 0) →
  a < 2 / Real.exp 1
  ∧
  -- Part 2
  (a = 0 ∧ ∀ x > 1, x * Real.log x > l k x) →
  (∀ m : ℤ, m > k → m ≤ 4) :=
by
  sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l479_47913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_expression_zero_l479_47996

theorem angle_on_line_expression_zero (α : Real) :
  (∃ (x y : Real), x + y = 0 ∧ x = Real.cos α ∧ y = Real.sin α) →
  (Real.sin α / Real.sqrt (1 - Real.sin α ^ 2)) + (Real.sqrt (1 - Real.cos α ^ 2) / Real.cos α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_expression_zero_l479_47996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_smallest_solutions_l479_47966

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def equation (x : ℝ) : Prop := x - (floor x : ℝ) = 1 / (floor x : ℝ)^2

def is_solution (x : ℝ) : Prop := x > 0 ∧ equation x

theorem sum_of_three_smallest_solutions :
  ∃ (a b c : ℝ), is_solution a ∧ is_solution b ∧ is_solution c ∧
  (∀ x, is_solution x → x ≥ a) ∧
  (∀ x, is_solution x ∧ x ≠ a → x ≥ b) ∧
  (∀ x, is_solution x ∧ x ≠ a ∧ x ≠ b → x ≥ c) ∧
  a + b + c = 9 + 73 / 144 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_smallest_solutions_l479_47966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l479_47922

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- State the theorem
theorem tangent_line_at_one :
  ∃ (A B C : ℝ), 
    (A ≠ 0 ∨ B ≠ 0) ∧
    (∀ x y : ℝ, y = f x → (deriv f) x * (x - 1) = y - f 1 → A * x + B * y + C = 0) ∧
    A = 1 ∧ B = -1 ∧ C = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l479_47922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₀_range_l479_47988

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point M
noncomputable def M (x₀ : ℝ) : ℝ × ℝ := (x₀, Real.sqrt 3)

-- Define the angle condition
noncomputable def angle_condition (O M N : ℝ × ℝ) : Prop :=
  Real.arccos ((O.1 - M.1) * (N.1 - M.1) + (O.2 - M.2) * (N.2 - M.2)) / 
  (Real.sqrt ((O.1 - M.1)^2 + (O.2 - M.2)^2) * Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)) ≥ Real.pi/6

-- State the theorem
theorem x₀_range (x₀ : ℝ) :
  (∃ (N : ℝ × ℝ), circleEq N.1 N.2 ∧ 
   (∃ (line : ℝ → ℝ), line x₀ = Real.sqrt 3 ∧ line N.1 = N.2) ∧
   angle_condition (0, 0) (M x₀) N) →
  -1 ≤ x₀ ∧ x₀ ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₀_range_l479_47988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_circle_plus_two_l479_47965

noncomputable def circle_plus (a b : ℝ) : ℝ := a + a / b + a * b

theorem eight_circle_plus_two : circle_plus 8 2 = 28 := by
  unfold circle_plus
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_circle_plus_two_l479_47965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l479_47998

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function y = x - [x]
noncomputable def f (x : ℝ) : ℝ := x - (floor x)

-- State the theorem
theorem range_of_f : 
  (∀ y ∈ Set.Icc (0 : ℝ) 1, ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f x = y) ∧
  (∀ x : ℝ, -1 < x → x < 1 → 0 ≤ f x ∧ f x < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l479_47998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_value_circle_tangent_equations_l479_47947

-- Define the circle
def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + m = 0

-- Define the condition for the chord on x-axis
def chord_condition (m : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ (r/2)^2 + 2^2 = r^2 ∧ r^2 = 5 - m

-- Define the tangent line with equal intercepts
def tangent_equal_intercepts (a : ℝ) (x y : ℝ) : Prop :=
  x + y - a = 0

-- Theorem for part 1
theorem circle_m_value :
  (∃ m : ℝ, chord_condition m) → (∃ m : ℝ, m = 11/15) :=
by
  sorry

-- Theorem for part 2
theorem circle_tangent_equations (m : ℝ) :
  m = 3 →
  (∃ (k₁ k₂ a₁ a₂ : ℝ),
    (k₁ = 2 + Real.sqrt 6 ∧ k₂ = 2 - Real.sqrt 6 ∧ a₁ = -1 ∧ a₂ = 3) ∧
    (∀ x y : ℝ, circle_eq m x y →
      (y = k₁ * x ∨ y = k₂ * x ∨ tangent_equal_intercepts a₁ x y ∨ tangent_equal_intercepts a₂ x y))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_value_circle_tangent_equations_l479_47947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_with_specific_b_exists_l479_47933

theorem sin_range_with_specific_b_exists :
  ∃ (a b : ℝ) (k : ℤ), 
    (∀ x ∈ Set.Icc a b, Real.sin x ∈ Set.Icc (-1) (1/2)) ∧ 
    b = 2 * Real.pi * (k : ℝ) - Real.pi/6 ∧ 
    b > a :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_with_specific_b_exists_l479_47933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_task_sequences_l479_47981

theorem cleaning_task_sequences (n m : ℕ) (hn : n = 15) (hm : m = 5) :
  (Finset.range n).card.factorial / (Finset.range n).card.factorial.sub m.factorial = 360360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_task_sequences_l479_47981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_square_times_60_l479_47909

/-- The radius of the inscribed circle of a triangle --/
noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- Theorem: For a triangle with sides 5, 6, and 7, 60 times the square of the inradius equals 160 --/
theorem inradius_square_times_60 :
  60 * (inradius 5 6 7)^2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_square_times_60_l479_47909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l479_47983

/-- The distance of the race track -/
def d : ℝ := sorry

/-- The speed of racer X -/
def x : ℝ := sorry

/-- The speed of racer Y -/
def y : ℝ := sorry

/-- The speed of racer Z -/
def z : ℝ := sorry

/-- When X finishes, Y is 15 yards behind -/
axiom xy_relation : d / x = (d - 15) / y

/-- When Y finishes, Z is 5 yards behind -/
axiom yz_relation : d / y = (d - 5) / z

/-- When X finishes, Z is 18 yards behind -/
axiom xz_relation : d / x = (d - 18) / z

/-- The race distance is 37.5 yards -/
theorem race_distance : d = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l479_47983
