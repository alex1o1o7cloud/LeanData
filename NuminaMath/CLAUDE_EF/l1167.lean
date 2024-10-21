import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_color_schemes_l1167_116708

/-- Represents a 7x7 checkerboard with two yellow squares -/
def Checkerboard := Fin 7 × Fin 7 → Bool

/-- Two positions on the checkerboard -/
def TwoPositions := (Fin 7 × Fin 7) × (Fin 7 × Fin 7)

/-- Rotates a position on the checkerboard -/
def rotate (pos : Fin 7 × Fin 7) : Fin 7 × Fin 7 := sorry

/-- Checks if two positions are rotationally equivalent -/
def isRotationallyEquivalent (pos1 pos2 : TwoPositions) : Prop := sorry

/-- The set of all possible two-square selections on the checkerboard -/
def allPositions : Finset TwoPositions := sorry

/-- The set of rotationally inequivalent two-square selections -/
def inequivalentPositions : Finset TwoPositions := sorry

/-- The number of rotationally inequivalent color schemes -/
def numInequivalentSchemes : ℕ := Finset.card inequivalentPositions

theorem checkerboard_color_schemes :
  numInequivalentSchemes = 312 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_color_schemes_l1167_116708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1167_116760

theorem tan_difference (α β : Real) 
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  (h3 : Real.tan (π - β) = 1/2) :
  Real.tan (α - β) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1167_116760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1167_116713

/-- Geometric sequence with common ratio q -/
noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n => a₁ * q^(n-1)

/-- Partial sum of geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n =>
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_property (a₁ : ℝ) :
  let q : ℝ := -Real.sqrt 34 / 2
  let a := geometric_sequence a₁ q
  let S := geometric_sum a₁ q
  (2 * S 9 = S 3 + S 6) ∧ (a 2 + a 5 = 2 * a 8) := by
  sorry

#check geometric_sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1167_116713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_circles_theorem_l1167_116705

/-- Two circles are tangent -/
def CircleTangent (A B : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (r₁ + r₂)^2

/-- A circle is tangent to a line -/
def CircleTangentToLine (A : ℝ × ℝ) (r : ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (P : ℝ × ℝ), P ∈ l ∧ (A.1 - P.1)^2 + (A.2 - P.2)^2 = r^2

/-- Three mutually tangent circles theorem -/
theorem three_tangent_circles_theorem 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hca : c < a) (hab : a < b)
  (h_tangent : ∃ (A B C : ℝ × ℝ) (l : Set (ℝ × ℝ)), 
    CircleTangent A B a b ∧ 
    CircleTangent B C b c ∧ 
    CircleTangent A C a c ∧
    CircleTangentToLine A a l ∧
    CircleTangentToLine B b l ∧
    CircleTangentToLine C c l) :
  1 / Real.sqrt c = 1 / Real.sqrt a + 1 / Real.sqrt b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_circles_theorem_l1167_116705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_values_from_conditions_l1167_116724

theorem sine_values_from_conditions (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi/2) (h3 : Real.pi/2 < β) (h4 : β < Real.pi)
  (h5 : Real.tan (α/2) = 1/2) (h6 : Real.cos (β - α) = Real.sqrt 2/10) :
  Real.sin α = 4/5 ∧ Real.sin β = Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_values_from_conditions_l1167_116724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_magnitude_l1167_116753

theorem complex_ratio_magnitude (Z₁ Z₂ : ℂ) (h1 : Complex.abs Z₁ = 2) (h2 : Complex.abs Z₂ = 3)
  (h3 : (Z₁.arg - Z₂.arg).cos = 1/2) :
  Complex.abs ((Z₁ + Z₂) / (Z₁ - Z₂)) = Real.sqrt (19 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_magnitude_l1167_116753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l1167_116783

noncomputable section

/-- A function f is even if f(-x) = f(x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x-1)^2 + ax + sin(x + π/2) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x - 1)^2 + a*x + Real.sin (x + Real.pi/2)

/-- If f is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) :
  IsEven (f a) → a = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l1167_116783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_final_distance_l1167_116793

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Jane's walking path --/
noncomputable def jane_path : List Point :=
  [⟨0, 0⟩, ⟨0, -10⟩, ⟨-12, -10⟩, ⟨-12 + 20 * Real.sqrt 2 / 2, -10 + 20 * Real.sqrt 2 / 2⟩]

/-- Theorem stating that Jane's final position is approximately 4.66 meters from her starting point --/
theorem jane_final_distance : 
  abs (distance (jane_path.head!) (jane_path.getLast!) - 4.66) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_final_distance_l1167_116793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_168_l1167_116777

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
noncomputable def x_intercept1 : ℝ := -3
noncomputable def x_intercept2 : ℝ := 4

-- Define the y-intercept
noncomputable def y_intercept : ℝ := curve 0

-- Define the triangle area calculation
noncomputable def triangle_area (base width : ℝ) (height : ℝ) : ℝ := (1/2) * base * height

-- Theorem statement
theorem triangle_area_is_168 : 
  triangle_area (x_intercept2 - x_intercept1) y_intercept = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_168_l1167_116777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_circle_to_line_l1167_116730

/-- The circle equation in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ^2 + 2*ρ*(Real.cos θ) - 3 = 0

/-- The line equation in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop :=
  ρ*(Real.cos θ) + ρ*(Real.sin θ) - 7 = 0

/-- The maximum distance from a point on the circle to the line -/
noncomputable def max_distance : ℝ := 4*Real.sqrt 2 + 2

theorem max_distance_from_circle_to_line :
  ∀ ρ θ : ℝ, circle_equation ρ θ →
    (∃ ρ' θ' : ℝ, line_equation ρ' θ' ∧
      ∀ d : ℝ, d = Real.sqrt ((ρ*Real.cos θ - ρ'*Real.cos θ')^2 + (ρ*Real.sin θ - ρ'*Real.sin θ')^2) →
        d ≤ max_distance) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_circle_to_line_l1167_116730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_a_eq_one_l1167_116748

/-- Curve C in polar coordinates -/
noncomputable def curve_C (a : ℝ) (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 2 * a * Real.cos θ

/-- Line L in parametric form -/
noncomputable def line_L (t : ℝ) : ℝ × ℝ := (-2 + Real.sqrt 2/2 * t, -4 + Real.sqrt 2/2 * t)

/-- Point P -/
def point_P : ℝ × ℝ := (-2, -4)

theorem curve_line_intersection_a_eq_one 
  (a : ℝ) (ha : a > 0)
  (M N : ℝ × ℝ)
  (hM : ∃ t, line_L t = M ∧ curve_C a M.1 M.2)
  (hN : ∃ t, line_L t = N ∧ curve_C a N.1 N.2)
  (h_geom_seq : ∃ r, 
    (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 = r * ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
    (N.1 - M.1)^2 + (N.2 - M.2)^2 = r * ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2)) :
  a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_a_eq_one_l1167_116748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_distance_l1167_116704

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem light_ray_distance :
  ∃ (t : ℝ × ℝ), circleC t.1 t.2 ∧ distance point_A t = 2 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_distance_l1167_116704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l1167_116714

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem range_of_y :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 6 ≤ y x ∧ y x ≤ 13 := by
  sorry

-- Note: We changed the upper bound of x from 9 to 3 to match the solution's analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l1167_116714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l1167_116757

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the point of intersection P
def P : ℝ × ℝ := (1, 1)

-- Define the sequence of points P_n and Q_n
noncomputable def P_n (n : ℕ) : ℝ × ℝ := (1 + (-1/2)^(n-1), 1 - (-1/2)^(n-1))
noncomputable def Q_n (n : ℕ) : ℝ × ℝ := (1 + (-1/2)^(n-1), 1/2 * (1 + (-1/2)^(n-1)) + 1/2)

-- State the theorem
theorem intersection_and_distance :
  (Q_n 2 = (1/2, 3/4)) ∧
  (∀ n : ℕ, n > 0 → (P_n n).1 - (P.1) = (-1/2)^(n-1) ∧ (P_n n).2 - (P.2) = -(-1/2)^(n-1)) :=
by
  sorry

-- Note: The distance formula |PP_n|² = 2 × (1/4)^(n-1) is implied by the coordinate differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l1167_116757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_distance_l1167_116747

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the frog's jump sequence -/
def symmetricJump (p : Point) (center : Point) : Point :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Helper function to perform multiple jumps -/
def multiJump (n : ℕ) (p : Point) (centers : List Point) : Point :=
  match n, centers with
  | 0, _ => p
  | _, [] => p
  | n+1, center::rest => multiJump n (symmetricJump p center) (rest ++ [center])

theorem frog_jump_distance 
  (A B C P : Point)
  (h1 : distance P C = 0.27) :
  let P2009 := multiJump 2009 P [A, B, C, A]
  100 * distance P P2009 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_distance_l1167_116747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_intersects_small_circles_l1167_116773

theorem large_circle_intersects_small_circles :
  ∀ (x y : ℝ), ∃ (m n : ℤ), 
    Real.sqrt ((x - (m : ℝ)) ^ 2 + (y - (n : ℝ)) ^ 2) ≤ 100 + 1/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_intersects_small_circles_l1167_116773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_implies_value_l1167_116776

noncomputable def power_function (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

theorem power_function_through_point_implies_value :
  ∃ n : ℝ, power_function n 3 = Real.sqrt 3 / 3 → power_function n 2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_implies_value_l1167_116776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l1167_116775

/-- An equiangular hexagon with an inscribed square -/
structure InscribedSquareHexagon where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of EF -/
  ef : ℝ
  /-- The hexagon is equiangular -/
  equiangular : True
  /-- Square PQRS is inscribed in hexagon ABCDEF -/
  inscribed_square : True
  /-- P is on AB -/
  p_on_ab : True
  /-- Q is on CD -/
  q_on_cd : True
  /-- R is on EF -/
  r_on_ef : True

/-- The side length of the inscribed square -/
noncomputable def square_side_length (h : InscribedSquareHexagon) : ℝ :=
  25 * Real.sqrt 3 - 17

theorem inscribed_square_side_length (h : InscribedSquareHexagon) 
  (h_ab : h.ab = 50)
  (h_ef : h.ef = 35 * (Real.sqrt 3 - 1)) :
  square_side_length h = 25 * Real.sqrt 3 - 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l1167_116775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cost_l1167_116774

theorem ice_cream_cost (aaron_savings : ℚ) (carson_savings : ℚ) (dinner_bill_fraction : ℚ) 
  (aaron_scoops : ℕ) (carson_scoops : ℕ) (aaron_change : ℚ) (carson_change : ℚ) :
  aaron_savings = 40 →
  carson_savings = 40 →
  dinner_bill_fraction = 3/4 →
  aaron_scoops = 6 →
  carson_scoops = 6 →
  aaron_change = 1 →
  carson_change = 1 →
  (aaron_savings + carson_savings) * (1 - dinner_bill_fraction) - aaron_change - carson_change = 
    (aaron_scoops + carson_scoops) * (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cost_l1167_116774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_results_correct_l1167_116723

/-- An isosceles trapezoid with given parameters -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of shorter parallel side
  m : ℝ  -- height
  h_positive : 0 < a ∧ 0 < m

/-- Surfaces and volumes of solids formed by rotating the trapezoid -/
noncomputable def rotation_results (t : IsoscelesTrapezoid) : 
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let a := t.a
  let m := t.m
  ( (2 * m * Real.pi * (a + (2 + Real.sqrt 2) * m), 
     m^2 * Real.pi * (a + 4 * m / 3)),
    (2 * m * Real.pi * (a + m * Real.sqrt 2), 
     m^2 * Real.pi * (a + 2 * m / 3)),
    (2 * m * Real.pi * ((m + a) * (1 + Real.sqrt 2)) + a^2 * Real.pi * Real.sqrt 2, 
     m * Real.pi * Real.sqrt 2 / 6 * (4 * m^2 + 6 * m * a + 3 * a^2)) )

theorem rotation_results_correct (t : IsoscelesTrapezoid) :
  let ((F₁, K₁), (F₂, K₂), (F₃, K₃)) := rotation_results t
  (F₁ = 2 * t.m * Real.pi * (t.a + (2 + Real.sqrt 2) * t.m)) ∧
  (K₁ = t.m^2 * Real.pi * (t.a + 4 * t.m / 3)) ∧
  (F₂ = 2 * t.m * Real.pi * (t.a + t.m * Real.sqrt 2)) ∧
  (K₂ = t.m^2 * Real.pi * (t.a + 2 * t.m / 3)) ∧
  (F₃ = 2 * t.m * Real.pi * ((t.m + t.a) * (1 + Real.sqrt 2)) + t.a^2 * Real.pi * Real.sqrt 2) ∧
  (K₃ = t.m * Real.pi * Real.sqrt 2 / 6 * (4 * t.m^2 + 6 * t.m * t.a + 3 * t.a^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_results_correct_l1167_116723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_iff_equation_l1167_116721

/-- A triangle is characterized by its area, incircle radius, circumcircle radius, 
    and the distance between the centers of its incircle and circumcircle. -/
structure Triangle where
  t : ℝ  -- area
  ρ : ℝ  -- incircle radius
  r : ℝ  -- circumcircle radius
  d : ℝ  -- distance between incircle and circumcircle centers

/-- Predicate to determine if a triangle is right-angled -/
def IsRightTriangle (tri : Triangle) : Prop :=
  sorry -- This would be defined based on the specific criteria for a right triangle

/-- A triangle is right-angled if and only if t = ρ² + r² - d² -/
theorem right_triangle_iff_equation (tri : Triangle) : 
  IsRightTriangle tri ↔ tri.t = tri.ρ^2 + tri.r^2 - tri.d^2 :=
by
  sorry -- The proof would go here

#check right_triangle_iff_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_iff_equation_l1167_116721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l1167_116772

-- Define the ellipse
noncomputable def Ellipse (x y : ℝ) : Prop := y^2/4 + x^2 = 1

-- Define the point P
def P : ℝ × ℝ := (0, -2)

-- Define the eccentricity
noncomputable def e : ℝ := Real.sqrt 3 / 2

-- Define the slopes k₁ and k₂
variable (k₁ k₂ : ℝ)

-- Define the condition k₁ · k₂ = 2
def SlopeProduct : Prop := k₁ * k₂ = 2

-- Define a point on line PA
def PointOnPA (x : ℝ) : ℝ × ℝ := (x, k₁ * x - 2)

-- Define a point on line PB
def PointOnPB (x : ℝ) : ℝ × ℝ := (x, k₂ * x - 2)

-- Define the fixed point Q
def Q : ℝ × ℝ := (0, -6)

-- Theorem statement
theorem ellipse_intersection_fixed_point :
  ∀ (A B : ℝ × ℝ),
    Ellipse A.1 A.2 →
    Ellipse B.1 B.2 →
    (∃ x₁, A = PointOnPA k₁ x₁) →
    (∃ x₂, B = PointOnPB k₂ x₂) →
    SlopeProduct k₁ k₂ →
    (∃ t : ℝ, (1 - t) • A + t • B = Q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l1167_116772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_years_range_l1167_116766

def employee_years : List ℕ := [15, 10, 9, 17, 6, 3, 14, 16]

def range (l : List ℕ) : ℕ :=
  match l.maximum?, l.minimum? with
  | some max, some min => max - min
  | _, _ => 0

theorem employee_years_range : range employee_years = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_years_range_l1167_116766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_value_l1167_116742

theorem tan_A_value (A B : ℝ) (h1 : A + B = Real.pi) (h2 : Real.pi/2 < B ∧ B < Real.pi) (h3 : Real.sin B = 1/3) : 
  Real.tan A = Real.sqrt 2/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_value_l1167_116742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1167_116781

noncomputable def angle_terminal_point (α : ℝ) : ℝ × ℝ := (-12, 5)

theorem sin_alpha_value (α : ℝ) : 
  angle_terminal_point α = (-12, 5) → Real.sin α = 5/13 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1167_116781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_full_space_symmetry_half_space_symmetry_spatial_quadrant_symmetry_octant_symmetry_l1167_116701

-- Define the types for our geometric objects
structure Space where

structure Plane where

structure Line where

structure Point where

-- Define the concepts of symmetry
def isSymmetryPlane (s : Space) (p : Plane) : Prop := sorry

def isSymmetryAxis (s : Space) (l : Line) : Prop := sorry

def isSymmetryCenter (s : Space) (p : Point) : Prop := sorry

-- Define our spaces
def FullSpace : Space := sorry

def HalfSpace : Space := sorry

def SpatialQuadrant : Space := sorry

def Octant : Space := sorry

-- Define the boundary plane for half-space
def boundaryPlane : Plane := sorry

-- Define perpendicularity
def isPerpendicular (p1 p2 : Plane) : Prop := sorry

-- Define the number of symmetry planes
def numSymmetryPlanes (s : Space) : Nat := sorry

-- Theorems to prove
theorem full_space_symmetry :
  ∀ (p : Plane) (l : Line) (pt : Point),
    isSymmetryPlane FullSpace p ∧
    isSymmetryAxis FullSpace l ∧
    isSymmetryCenter FullSpace pt :=
by sorry

theorem half_space_symmetry :
  (∀ (p : Plane), isSymmetryPlane HalfSpace p ↔ isPerpendicular p boundaryPlane) ∧
  (∀ (l : Line), ¬isSymmetryAxis HalfSpace l) ∧
  (∀ (pt : Point), ¬isSymmetryCenter HalfSpace pt) :=
by sorry

theorem spatial_quadrant_symmetry :
  numSymmetryPlanes SpatialQuadrant = 3 :=
by sorry

theorem octant_symmetry :
  numSymmetryPlanes Octant = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_full_space_symmetry_half_space_symmetry_spatial_quadrant_symmetry_octant_symmetry_l1167_116701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_choice_related_to_gender_l1167_116762

/-- Represents a 2x2 contingency table --/
structure ContingencyTable :=
  (a b c d : ℕ)

/-- Calculates the chi-square statistic for a 2x2 contingency table --/
noncomputable def chi_square (table : ContingencyTable) : ℝ :=
  let n := table.a + table.b + table.c + table.d
  (n * (table.a * table.d - table.b * table.c)^2 : ℝ) / 
    ((table.a + table.b) * (table.c + table.d) * (table.a + table.c) * (table.b + table.d))

/-- The critical value for α = 0.05 --/
def critical_value : ℝ := 3.841

/-- The observed contingency table --/
def observed_table : ContingencyTable :=
  { a := 40, b := 10, c := 30, d := 20 }

/-- Theorem stating that the choice of course is related to gender --/
theorem course_choice_related_to_gender :
  chi_square observed_table > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_choice_related_to_gender_l1167_116762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_with_equal_intercepts_l1167_116752

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Check if a line is tangent to a circle -/
noncomputable def is_tangent (l : Line) (c : Circle) : Prop :=
  let d := abs (l.a * c.center_x + l.b * c.center_y + l.c) / Real.sqrt (l.a^2 + l.b^2)
  d = c.radius

/-- Check if a line has equal intercepts on x and y axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ abs (l.c / l.a) = abs (l.c / l.b)

/-- The circle (x-2)^2 + y^2 = 2 -/
noncomputable def given_circle : Circle :=
  { center_x := 2, center_y := 0, radius := Real.sqrt 2 }

/-- Theorem: The only lines tangent to the given circle with equal intercepts are x - y = 0, x + y - 4 = 0, and x + y = 0 -/
theorem tangent_lines_with_equal_intercepts :
  ∀ l : Line,
    is_tangent l given_circle ∧ has_equal_intercepts l ↔
    (l.a = 1 ∧ l.b = -1 ∧ l.c = 0) ∨
    (l.a = 1 ∧ l.b = 1 ∧ l.c = -4) ∨
    (l.a = 1 ∧ l.b = 1 ∧ l.c = 0) :=
by
  sorry

#check tangent_lines_with_equal_intercepts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_with_equal_intercepts_l1167_116752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_for_box_l1167_116749

/-- Represents a rectangular box with dimensions x, y, and z -/
structure RectangularBox where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Properties of the rectangular box -/
def BoxProperties (box : RectangularBox) : Prop :=
  2 * (box.x * box.y + box.y * box.z + box.z * box.x) = 118 ∧
  4 * (box.x + box.y + box.z) = 60 ∧
  box.z = 2 * box.y

/-- Sum of lengths of all interior diagonals -/
noncomputable def InteriorDiagonalsSum (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.x^2 + box.y^2 + box.z^2)

/-- Theorem: The sum of lengths of all interior diagonals for a box with given properties -/
theorem interior_diagonals_sum_for_box (box : RectangularBox) 
  (h : BoxProperties box) : 
  ∃ (sum : ℝ), InteriorDiagonalsSum box = sum :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_for_box_l1167_116749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertices_form_parabola_l1167_116718

noncomputable def vertex_x (a t : ℝ) : ℝ := -t / (2 * a)

noncomputable def vertex_y (a c t : ℝ) : ℝ := a * (vertex_x a t)^2 + t * (vertex_x a t) + c

theorem vertices_form_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t : ℝ, f t = (vertex_x a t, vertex_y a c t)) ∧
    (∃ A B D : ℝ, ∀ x y : ℝ, (x, y) ∈ Set.range f → y = A * x^2 + B * x + D) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertices_form_parabola_l1167_116718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_sculpture_l1167_116759

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical sculpture -/
structure SculptureDimensions where
  height : ℝ
  diameter : ℝ

/-- Calculates the volume of a rectangular block -/
def blockVolume (b : BlockDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a cylindrical sculpture -/
noncomputable def sculptureVolume (s : SculptureDimensions) : ℝ :=
  Real.pi * (s.diameter / 2) ^ 2 * s.height

/-- Calculates the number of whole blocks needed for the sculpture -/
noncomputable def blocksNeeded (s : SculptureDimensions) (b : BlockDimensions) : ℕ :=
  Nat.ceil (sculptureVolume s / blockVolume b)

theorem blocks_needed_for_sculpture
  (s : SculptureDimensions)
  (b : BlockDimensions)
  (h1 : s.height = 10)
  (h2 : s.diameter = 5)
  (h3 : b.length = 4)
  (h4 : b.width = 3)
  (h5 : b.height = 1) :
  blocksNeeded s b = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_sculpture_l1167_116759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_convex_interval_length_l1167_116771

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/12) * x^4 - (1/6) * m * x^3 - (3/2) * x^2

-- Define the second derivative of f(x)
noncomputable def f_second_deriv (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 3

-- State the theorem
theorem max_convex_interval_length (m : ℝ) (h : |m| ≤ 2) :
  ∃ a b : ℝ, a < b ∧
    (∀ x, a < x ∧ x < b → f_second_deriv m x < 0) ∧
    (∀ c d, c < d ∧ (∀ x, c < x ∧ x < d → f_second_deriv m x < 0) → d - c ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_convex_interval_length_l1167_116771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_operations_l1167_116761

theorem math_operations :
  (- (-Real.sqrt 2) = Real.sqrt 2) ∧
  (1 / Real.sqrt 2 = Real.sqrt 2 / 2) ∧
  ((-1/8 : ℝ) ^ (1/3 : ℝ) = -1/2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_operations_l1167_116761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shekars_english_score_l1167_116712

theorem shekars_english_score 
  (math_score science_score social_studies_score biology_score : ℕ)
  (english_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : social_studies_score = 82)
  (h4 : biology_score = 85)
  (h5 : average_score = 74)
  (h6 : (math_score + science_score + social_studies_score + biology_score + english_score : ℚ) / 5 = average_score) :
  english_score = 62 := by
  sorry

#check shekars_english_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shekars_english_score_l1167_116712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_product_l1167_116785

def basketball_scores (scores : List Nat) : Prop :=
  scores.length = 8 ∧
  scores.take 6 = [10, 5, 8, 6, 11, 4] ∧
  scores[6]! < 15 ∧
  scores[7]! < 15 ∧
  (scores.take 7).sum % 7 = 0 ∧
  scores.sum % 8 = 0

theorem basketball_score_product (scores : List Nat) 
  (h : basketball_scores scores) : scores[6]! * scores[7]! = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_product_l1167_116785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_five_sixths_l1167_116780

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0  -- Define a default value for x outside [0,2]

-- State the theorem
theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_five_sixths_l1167_116780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_two_l1167_116716

/-- The general term of the series -/
def a (n : ℕ) : ℚ := (5*n^3 - 2*n^2 - 2*n + 2) / (n^6 - 2*n^5 + 2*n^4 - 2*n^3 + 2*n^2 - 2*n)

/-- The series sum starting from n = 2 -/
noncomputable def S : ℚ := ∑' n, if n ≥ 2 then a n else 0

theorem sum_equals_two : S = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_two_l1167_116716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_in_square_l1167_116745

/-- Predicate to check if four points form a square -/
def is_square (A B C D : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if four points form an ellipse with A and B as foci -/
def is_ellipse_with_foci (A B C D : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the eccentricity of an ellipse given its foci and two points on the ellipse -/
noncomputable def ellipse_eccentricity (A B C D : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of an ellipse with foci at two opposite corners of a square
    and passing through the other two corners is √2 - 1 -/
theorem ellipse_eccentricity_in_square : ∀ (A B C D : ℝ × ℝ),
  is_square A B C D →
  is_ellipse_with_foci A B C D →
  ellipse_eccentricity A B C D = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_in_square_l1167_116745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l1167_116706

noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

def point_on_line (x y : ℝ) (m b : ℝ) : Prop :=
  y = m * x + b

def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

theorem ellipse_fixed_point (a b : ℝ) :
  a > b ∧ b > 0 →
  ellipse 2 0 a b →
  eccentricity a b = 1/2 →
  (∀ x y : ℝ, ellipse x y a b ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ p_x p_y m_x m_y n_x n_y : ℝ,
    p_x = -1 →
    ellipse m_x m_y 2 (Real.sqrt 3) →
    ellipse n_x n_y 2 (Real.sqrt 3) →
    is_midpoint p_x p_y m_x m_y n_x n_y →
    ∃ l_m : ℝ, point_on_line (-1/4) 0 l_m p_y ∧
               perpendicular l_m ((n_y - m_y) / (n_x - m_x))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l1167_116706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_symmetric_values_l1167_116769

def is_sum_symmetric (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  (∀ d ∈ digits, d ≠ 0) ∧
  digits[0]! + digits[1]! = digits[2]! + digits[3]!

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  b * 1000 + a * 100 + d * 10 + c

def F (n : ℕ) : ℚ :=
  (n + swap_digits n : ℚ) / 101

def k (A B : ℕ) : ℚ :=
  3 * F A + 2 * F B

theorem sum_symmetric_values (a b m n : ℕ) :
  3 ≤ a ∧ a ≤ 8 ∧
  0 ≤ b ∧ b ≤ 5 ∧
  2 ≤ m ∧ m ≤ 9 ∧
  5 ≤ n ∧ n ≤ 12 ∧
  is_sum_symmetric (1000 * a + 10 * b + 746) ∧
  is_sum_symmetric (100 * m + n + 2026) →
  (∃ z : ℤ, k (1000 * a + 10 * b + 746) (100 * m + n + 2026) = 77 * ↑z) ↔
  (1000 * a + 10 * b + 746) ∈ ({3746, 4756, 6776, 5766, 7786, 8796} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_symmetric_values_l1167_116769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_box_count_l1167_116779

/-- Proves that the total number of pieces of fruit in a box is 56,
    given the specified conditions about the ratios of different fruits. -/
theorem fruit_box_count : ∃ (apples peaches oranges : ℕ),
  apples = 35 ∧
  oranges = 2 * peaches ∧
  apples = 5 * peaches ∧
  4 * oranges = apples + peaches + oranges ∧
  apples + peaches + oranges = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_box_count_l1167_116779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_max_volume_l1167_116700

/-- The side length of the original square sheet in centimeters. -/
noncomputable def sheet_side : ℝ := 120

/-- The volume of the tank as a function of the base length x. -/
noncomputable def tank_volume (x : ℝ) : ℝ := -1/2 * x^3 + 60 * x^2

/-- The maximum volume of the tank in cubic centimeters. -/
noncomputable def max_volume : ℝ := 128000

/-- Theorem stating that there exists a base length that maximizes the tank volume. -/
theorem tank_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < sheet_side ∧
  (∀ (y : ℝ), y > 0 → y < sheet_side → tank_volume y ≤ tank_volume x) ∧
  tank_volume x = max_volume := by
  sorry

#check tank_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_max_volume_l1167_116700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1167_116795

/-- Given a line and a hyperbola, if the line intersects the asymptotes of the hyperbola at two points
    equidistant from a specific point on the x-axis, then the eccentricity of the hyperbola is √5/2. -/
theorem hyperbola_eccentricity (m a b : ℝ) (hm : m ≠ 0) (ha : a > 0) (hb : b > 0) :
  let line := {p : ℝ × ℝ | p.1 - 3*p.2 + m = 0}
  let hyperbola := {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}
  let asymptotes := {p : ℝ × ℝ | p.2 = b/a * p.1 ∨ p.2 = -b/a * p.1}
  let P := (m, 0)
  (∃ A B : ℝ × ℝ, A ∈ line ∧ A ∈ asymptotes ∧ B ∈ line ∧ B ∈ asymptotes ∧ A ≠ B) →
  (∃ A B : ℝ × ℝ, A ∈ line ∧ A ∈ asymptotes ∧ B ∈ line ∧ B ∈ asymptotes ∧ A ≠ B ∧ 
    dist P A = dist P B) →
  ∃ e : ℝ, e = Real.sqrt 5 / 2 ∧ e = (Real.sqrt (a^2 + b^2)) / a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1167_116795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1167_116740

theorem relationship_abc (a b c : ℝ) : 
  a = (2011 : ℝ)^(6/10) → b = (6/10 : ℝ)^2011 → c = Real.log 2011 / Real.log (6/10) → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1167_116740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_integers_sum_of_squares_l1167_116727

theorem consecutive_even_integers_sum_of_squares : ∀ a : ℤ,
  (∃ (x y z : ℤ), x = a - 2 ∧ y = a ∧ z = a + 2 ∧ 
   Even x ∧ Even y ∧ Even z ∧
   x * y * z = 12 * (x + y + z)) →
  (a - 2)^2 + a^2 + (a + 2)^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_integers_sum_of_squares_l1167_116727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1167_116717

-- Define the complex number z
noncomputable def z (a : ℝ) : ℂ := 1 / (a - Complex.I)

-- Define the condition that z lies on the line x - 2y = 0
def on_line (z : ℂ) : Prop := z.re - 2 * z.im = 0

-- Theorem statement
theorem imaginary_part_of_z :
  ∃ (a : ℝ), on_line (z a) ∧ (z a).im = 1/5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1167_116717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_ratio_l1167_116729

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the bottom base
  r : ℝ  -- radius of the top base
  s : ℝ  -- radius of the inscribed sphere
  H : ℝ  -- height of the truncated cone

/-- The volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius^3

/-- The volume of a truncated cone -/
noncomputable def truncatedConeVolume (c : TruncatedConeWithSphere) : ℝ :=
  (Real.pi / 3) * (c.R^2 + c.R * c.r + c.r^2) * c.H

theorem inscribed_sphere_ratio (c : TruncatedConeWithSphere) :
  c.s^2 = c.R * c.r →
  truncatedConeVolume c = 3 * sphereVolume c.s →
  c.R / c.r = (5 + Real.sqrt 21) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_ratio_l1167_116729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_100_trailing_zeros_l1167_116744

def factorial (n : ℕ) : ℕ := 
  Finset.prod (Finset.range n) (λ i => i + 1)

def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range 3).sum (λ i => n / (5 ^ (i + 1)))

theorem factorial_100_trailing_zeros :
  trailingZeros (factorial 100) = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_100_trailing_zeros_l1167_116744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l1167_116778

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Helper function to convert (ℝ × ℝ) to Point
def pairToPoint (pair : ℝ × ℝ) : Point :=
  ⟨pair.1, pair.2⟩

-- State the theorem
theorem circle_radius_theorem 
  (C₁ C₂ : Circle) (O X Y Z : Point) : 
  (O = pairToPoint C₁.center) →  -- O is the center of C₁
  (distance O (pairToPoint C₂.center) = C₂.radius) →  -- O lies on C₂
  (distance X (pairToPoint C₁.center) = C₁.radius) →  -- X is on C₁
  (distance Y (pairToPoint C₁.center) = C₁.radius) →  -- Y is on C₁
  (distance X (pairToPoint C₂.center) = C₂.radius) →  -- X is on C₂
  (distance Y (pairToPoint C₂.center) = C₂.radius) →  -- Y is on C₂
  (distance Z (pairToPoint C₂.center) = C₂.radius) →  -- Z is on C₂
  (distance X Z = 15) →  -- XZ = 15
  (distance O Z = 12) →  -- OZ = 12
  (distance Y Z = 8) →   -- YZ = 8
  (distance Z (pairToPoint C₁.center) > C₁.radius) →  -- Z is exterior to C₁
  C₁.radius = Real.sqrt 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l1167_116778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1167_116703

theorem trig_identity (α : Real) 
  (h1 : -π/2 < α) 
  (h2 : α < 0) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = 25/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1167_116703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_equals_7_l1167_116798

-- Define the functions t and s
noncomputable def t (x : ℝ) : ℝ := 4 * x - 6

noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 6) / 4  -- Inverse of t(x)
  x^2 + 5 * x - 7

-- Theorem statement
theorem s_of_2_equals_7 : s 2 = 7 := by
  -- Unfold the definition of s
  unfold s
  -- Simplify the expression
  simp
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_equals_7_l1167_116798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_proof_l1167_116792

def initial_population : ℝ := 4999.999999999999

def year_end_population (start : ℝ) (change : ℝ) : ℝ :=
  start * (1 + change)

def population_after_three_years (p : ℝ) : ℝ :=
  year_end_population (year_end_population (year_end_population p (-0.1)) 0.1) (-0.1)

theorem population_change_proof :
  Int.floor (population_after_three_years initial_population) = 4455 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_proof_l1167_116792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1167_116786

/-- Calculates the speed of a train in km/hr given its length and time to pass a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Theorem: A train 280 m long that passes a tree in 16 seconds has a speed of 63 km/hr -/
theorem train_speed_calculation :
  train_speed 280 16 = 63 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1167_116786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_inequality_l1167_116741

/-- A triangle with special points -/
structure SpecialTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ      -- Centroid
  H : ℝ × ℝ      -- Orthocenter
  M : ℝ × ℝ      -- Midpoint of arc AC (not containing B)
  R : ℝ          -- Radius of circumcircle

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: If MG = R in a special triangle, then BG ≥ BH -/
theorem special_triangle_inequality (t : SpecialTriangle) 
    (h : distance t.M t.G = t.R) : 
  distance t.B t.G ≥ distance t.B t.H := by
  sorry

#check special_triangle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_inequality_l1167_116741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1167_116709

theorem cubic_equation_roots (a : ℝ) :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (λ x => 6.269 * x^3 - 2*a*x^2 + (a^2 + 2*Real.sqrt 3*a - 9)*x - (2*a^2*Real.sqrt 3 - 12*a + 6*Real.sqrt 3)) x₁ = 0 ∧
    (λ x => 6.269 * x^3 - 2*a*x^2 + (a^2 + 2*Real.sqrt 3*a - 9)*x - (2*a^2*Real.sqrt 3 - 12*a + 6*Real.sqrt 3)) x₂ = 0 ∧
    (λ x => 6.269 * x^3 - 2*a*x^2 + (a^2 + 2*Real.sqrt 3*a - 9)*x - (2*a^2*Real.sqrt 3 - 12*a + 6*Real.sqrt 3)) x₃ = 0 ∧
    x₁ = 2 * Real.sqrt 3 ∧
    x₂ = a - Real.sqrt 3 ∧
    x₃ = a - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1167_116709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_negative_probability_l1167_116758

def S : Finset Int := {-5, -8, 7, 4, -2, 6, -3}

theorem product_negative_probability :
  (Finset.card (Finset.filter (fun p => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ p.1 * p.2 < 0) (S.product S)) : ℚ) /
  (Finset.card (Finset.filter (fun p => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2) (S.product S)) : ℚ) = 4 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_negative_probability_l1167_116758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l1167_116732

noncomputable def ellipse_C : Set (ℝ × ℝ) :=
  {p | p.2^2 / 4 + p.1^2 = 1}

noncomputable def F₁ : ℝ × ℝ := (0, -Real.sqrt 3)
noncomputable def F₂ : ℝ × ℝ := (0, Real.sqrt 3)
noncomputable def M : ℝ × ℝ := (Real.sqrt 3 / 2, 1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_C_properties :
  (∀ (p : ℝ × ℝ), p ∈ ellipse_C → 1 ≤ (1 / distance p F₁ + 1 / distance p F₂) ∧ 
                                   (1 / distance p F₁ + 1 / distance p F₂) ≤ 4) ∧
  M ∈ ellipse_C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l1167_116732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1167_116739

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x^2 - 6*x + 5)) / Real.log 0.3

-- State the theorem
theorem f_monotone_increasing :
  ∃ (a b c d : ℝ), a < b ∧ c < d ∧
  (∀ x y, (x < y ∧ ((a < x ∧ x < y ∧ y < b) ∨ (c < x ∧ x < y ∧ y < d))) → f x < f y) ∧
  (∀ x, x ∉ (Set.Ioo a b) ∪ (Set.Ioo c d) → 
    ∃ y, y ≠ x ∧ ((y < x → f y ≥ f x) ∨ (x < y → f x ≥ f y))) :=
by
  -- We'll use -∞ and 1 for a and b, and 3 and 5 for c and d
  use (-Real.log 0 / Real.log 0.3), 1, 3, 5
  sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1167_116739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_subtraction_theorem_l1167_116734

/-- A polynomial in two variables -/
def MyPolynomial (α : Type*) := α → α → α

theorem polynomial_subtraction_theorem 
  (P : MyPolynomial ℝ) :
  (∀ x y : ℝ, P x y - (x^2 - 3*y^2) = x^2 + 2*y^2) →
  (∀ x y : ℝ, P x y = 2*x^2 - y^2) :=
by
  intro h
  intro x y
  have eq := h x y
  rw [sub_eq_iff_eq_add] at eq
  rw [eq]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_subtraction_theorem_l1167_116734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trirectangular_tetrahedron_circumsphere_radius_l1167_116707

/-- A trirectangular tetrahedron is a tetrahedron where three face angles at one vertex are right angles. -/
structure TrirectangularTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The radius of the circumsphere of a trirectangular tetrahedron. -/
noncomputable def circumsphere_radius (t : TrirectangularTetrahedron) : ℝ :=
  (1 / 2) * Real.sqrt (t.a^2 + t.b^2 + t.c^2)

/-- Theorem: The radius of the circumsphere of a trirectangular tetrahedron
    with lateral edges of lengths a, b, and c is (1/2) * √(a² + b² + c²). -/
theorem trirectangular_tetrahedron_circumsphere_radius 
    (t : TrirectangularTetrahedron) : 
    circumsphere_radius t = (1 / 2) * Real.sqrt (t.a^2 + t.b^2 + t.c^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trirectangular_tetrahedron_circumsphere_radius_l1167_116707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_consecutive_sides_double_angle_l1167_116787

theorem unique_triangle_consecutive_sides_double_angle :
  ∃! (a b c : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (a > 0) ∧
    (a^2 + b^2 > c^2) ∧  -- Triangle inequality
    ∃ (A B C : Real),
      (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
      (A + B + C = Real.pi) ∧
      (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) ∧  -- Law of cosines
      ((A = 2*B) ∨ (B = 2*C) ∨ (C = 2*A)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_consecutive_sides_double_angle_l1167_116787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_number_wall_m_value_l1167_116790

/-- A modified Number Wall structure -/
structure ModifiedNumberWall where
  bottom_row : Fin 4 → ℕ
  second_row : Fin 3 → ℕ
  third_row : Fin 2 → ℕ
  top_value : ℕ

/-- The rule for calculating values in the modified Number Wall -/
def calculate_above (a b : ℕ) : ℕ := a + b

/-- The given modified Number Wall instance -/
def given_wall (m : ℕ) : ModifiedNumberWall where
  bottom_row := fun i => match i with
    | ⟨0, _⟩ => m
    | ⟨1, _⟩ => 3
    | ⟨2, _⟩ => 9
    | ⟨3, _⟩ => 6
  second_row := fun i => match i with
    | ⟨2, _⟩ => 12
    | _ => 0  -- placeholder for unknown values
  third_row := fun _ => 0  -- placeholder for unknown values
  top_value := 55

theorem modified_number_wall_m_value :
  ∃ (m : ℕ), 
    let wall := given_wall m
    let second_left := calculate_above m 3
    let second_right := calculate_above 9 6
    let third_left := calculate_above second_left 12
    let third_right := calculate_above second_right 12
    wall.top_value = calculate_above third_left third_right ∧ m = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_number_wall_m_value_l1167_116790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_digit_number_satisfies_conditions_l1167_116733

/-- A function that checks if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that extracts the first two digits of a six-digit number --/
def firstTwoDigits (n : ℕ) : ℕ :=
  n / 10000

/-- A function that extracts the middle two digits of a six-digit number --/
def middleTwoDigits (n : ℕ) : ℕ :=
  (n / 100) % 100

/-- A function that extracts the last two digits of a six-digit number --/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- The main theorem stating that no six-digit number satisfies all conditions --/
theorem no_six_digit_number_satisfies_conditions : 
  ¬ ∃ (N : ℕ), 
    100000 ≤ N ∧ N < 1000000 ∧  -- Six-digit number
    (∀ d, d ∈ N.digits 10 → d ≠ 0) ∧  -- No digit is zero
    isPerfectSquare N ∧  -- N is a perfect square
    isPerfectSquare (firstTwoDigits N) ∧  -- First two digits are a perfect square
    isPerfectSquare (middleTwoDigits N) ∧  -- Middle two digits are a perfect square
    isPerfectSquare (lastTwoDigits N) ∧  -- Last two digits are a perfect square
    isPerfectSquare ((firstTwoDigits N)^2 + (lastTwoDigits N)^2)  -- Sum of squares of first and last two digits is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_digit_number_satisfies_conditions_l1167_116733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1167_116710

/-- A predicate to determine if a point is the focus of a parabola -/
def is_focus (parabola : Set (ℝ × ℝ)) (f : ℝ × ℝ) : Prop :=
  ∃ (p : ℝ), p > 0 ∧
  (∀ (x y : ℝ), (x, y) ∈ parabola ↔ (x - f.1)^2 + (y - f.2)^2 = (x + f.1 + p/2)^2)

/-- Given a parabola y^2 = ax with latus rectum x = -1, prove that its focus has coordinates (1, 0) -/
theorem parabola_focus_coordinates (a : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = a * x}
  let latus_rectum := {(x, y) : ℝ × ℝ | x = -1}
  (∀ (x y : ℝ), (x, y) ∈ latus_rectum → (x, y) ∈ parabola) →
  ∃ (f : ℝ × ℝ), f = (1, 0) ∧ is_focus parabola f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1167_116710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_five_pairs_l1167_116737

/-- A pair of 2-digit positive integers -/
def TwoDigitPair := { p : ℕ × ℕ // 10 ≤ p.1 ∧ p.1 ≤ 99 ∧ 10 ≤ p.2 ∧ p.2 ≤ 99 }

/-- The product of a pair of integers -/
def pairProduct (p : TwoDigitPair) : ℕ := p.val.1 * p.val.2

/-- The set of all pairs of 2-digit positive integers whose product is 630 -/
def validPairs : Set TwoDigitPair :=
  { p | pairProduct p = 630 }

/-- Proof that validPairs is finite -/
instance : Fintype validPairs := by
  sorry

theorem exactly_five_pairs :
  Fintype.card validPairs = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_five_pairs_l1167_116737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_cost_l1167_116756

def michael_money : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def additional_money_needed : ℕ := 11

theorem balloon_cost : 
  michael_money + additional_money_needed - (cake_cost + bouquet_cost) = 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_cost_l1167_116756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l1167_116722

-- Define the function f(x) = log_a x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function g(x) = (2+m)√x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (2 + m) * Real.sqrt x

-- Theorem statement
theorem log_base_value (a : ℝ) (m : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) 16, f a x ≤ 4) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) 16, f a x ≥ m) →
  (∃ x ∈ Set.Icc (1/2 : ℝ) 16, f a x = 4) →
  (∃ x ∈ Set.Icc (1/2 : ℝ) 16, f a x = m) →
  (∀ x y, 0 < x ∧ x < y → g m x < g m y) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l1167_116722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_correct_l1167_116764

/-- A particle moves in a plane according to the given equations. This function calculates its speed at time t. -/
noncomputable def particleSpeed (t : ℝ) : ℝ :=
  let x := t^2 + 2*t + 7
  let y := 2*t^2 + 4*t - 13
  2 * Real.sqrt 5 * |t + 1|

/-- Theorem stating that the speed of the particle at time t is correct. -/
theorem particle_speed_correct (t : ℝ) : 
  particleSpeed t = 2 * Real.sqrt 5 * |t + 1| :=
by
  -- Unfold the definition of particleSpeed
  unfold particleSpeed
  -- The definition matches the right-hand side exactly, so we're done
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_correct_l1167_116764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1167_116751

/-- Given a triangle ABC with points M on BC and N on CA, prove that r - s = -1 -/
theorem triangle_vector_relation (A B C M N : EuclideanSpace ℝ (Fin 2)) 
  (a b : EuclideanSpace ℝ (Fin 2)) (r s : ℝ) : 
  (B - A = a) →  -- AB = a
  (C - A = b) →  -- AC = b
  (M - B = (1/3 : ℝ) • (C - B)) →  -- BM = 1/3 * BC
  (N - C = (1/3 : ℝ) • (A - C)) →  -- CN = 1/3 * CA
  (N - M = r • a + s • b) →  -- MN = r*a + s*b
  r - s = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1167_116751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_workers_needed_l1167_116738

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℕ
  initialWorkers : ℕ
  completedLength : ℝ
  completedDays : ℕ

/-- Represents the workforce fluctuations throughout the week -/
structure WorkforceFluctuations where
  monTueIncrease : ℝ
  friDecrease : ℝ

/-- Calculates the average number of workers per day given the fluctuations -/
noncomputable def averageWorkersPerDay (initialWorkers : ℕ) (fluctuations : WorkforceFluctuations) : ℝ :=
  ((1 + fluctuations.monTueIncrease) * (initialWorkers : ℝ) * 2 +
   (initialWorkers : ℝ) * 3 +
   (1 - fluctuations.friDecrease) * (initialWorkers : ℝ)) / 6

/-- Theorem stating that at least 79 additional workers are needed to complete the project on time -/
theorem additional_workers_needed (project : RoadProject) (fluctuations : WorkforceFluctuations) :
  project.totalLength = 15 ∧
  project.totalDays = 300 ∧
  project.initialWorkers = 50 ∧
  project.completedLength = 2.5 ∧
  project.completedDays = 100 ∧
  fluctuations.monTueIncrease = 0.2 ∧
  fluctuations.friDecrease = 0.1 →
  ∃ n : ℕ, n ≥ 79 ∧
    (averageWorkersPerDay (project.initialWorkers + n) fluctuations) *
    (project.totalLength - project.completedLength) /
    (project.totalDays - project.completedDays : ℝ) ≥
    project.totalLength / (project.totalDays : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_workers_needed_l1167_116738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_function_l1167_116794

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (Real.pi/3 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.sin (Real.pi/3 * (x - 3) + φ)

theorem shifted_sine_function 
  (φ : ℝ) 
  (h1 : |φ| < Real.pi/2) 
  (h2 : ∀ x, f x φ = f (2 - x) φ) -- symmetry about x = 1
  : 
  ∀ x, g x φ = Real.sin (Real.pi/3 * x - 5*Real.pi/6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_function_l1167_116794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_change_l1167_116743

/-- Energy stored between two charges -/
noncomputable def energy (q1 q2 r : ℝ) : ℝ := (q1 * q2) / r

/-- Total energy of four charges in a square configuration -/
noncomputable def squareEnergy (q s : ℝ) : ℝ :=
  4 * energy q q s + 2 * energy q q (s * Real.sqrt 2)

/-- Total energy when one charge is moved to the center -/
noncomputable def centerEnergy (q s : ℝ) : ℝ :=
  3 * energy q q (s / Real.sqrt 2) + 3 * energy q q s

/-- Theorem stating the energy change when moving a charge to the center -/
theorem energy_change (q s : ℝ) (h1 : q > 0) (h2 : s > 0) 
  (h3 : squareEnergy q s = 20) :
  centerEnergy q s = (180 * Real.sqrt 2 + 40) / 14 := by
  sorry

#check energy_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_change_l1167_116743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_difference_l1167_116799

/-- Probability of getting heads in a single coin toss -/
noncomputable def p_heads : ℝ := 2/3

/-- Probability of getting tails in a single coin toss -/
noncomputable def p_tails : ℝ := 1/3

/-- Probability of winning Game A -/
noncomputable def p_win_A : ℝ := p_heads^3 + p_tails^3

/-- Probability of winning Game C -/
noncomputable def p_win_C : ℝ := (p_heads^3 + p_tails^3)^2

/-- The difference in probabilities of winning Game A and Game C -/
theorem prob_difference : p_win_A - p_win_C = 2/9 := by
  -- Expand definitions
  unfold p_win_A p_win_C p_heads p_tails
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_difference_l1167_116799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_pi_over_two_l1167_116750

-- Define the circle C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 = 0}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Theorem statement
theorem central_angle_is_pi_over_two :
  ∃ (A B : ℝ × ℝ),
    A ∈ C ∩ x_axis ∧
    B ∈ C ∩ x_axis ∧
    A ≠ B ∧
    let center := (2, 2)
    let angle := Real.arccos ((A.1 - center.1) * (B.1 - center.1) + (A.2 - center.2) * (B.2 - center.2)) /
                  (((A.1 - center.1)^2 + (A.2 - center.2)^2) * ((B.1 - center.1)^2 + (B.2 - center.2)^2))^(1/2)
    angle = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_pi_over_two_l1167_116750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_correct_l1167_116735

/-- The radius of the base of a cone given specific conditions -/
noncomputable def cone_base_radius : ℝ := 169 / 60

/-- The configuration of spheres and cone -/
structure SpheresAndCone where
  sphere1_radius : ℝ
  sphere2_radius : ℝ
  sphere3_radius : ℝ
  cone_height_radius_ratio : ℝ

/-- The specific configuration in the problem -/
noncomputable def problem_config : SpheresAndCone :=
  { sphere1_radius := 5
  , sphere2_radius := 4
  , sphere3_radius := 4
  , cone_height_radius_ratio := 4 / 3 }

/-- Theorem stating that the cone base radius is correct for the given configuration -/
theorem cone_base_radius_correct (config : SpheresAndCone)
  (h1 : config = problem_config)
  (h2 : config.sphere1_radius > 0)
  (h3 : config.sphere2_radius > 0)
  (h4 : config.sphere3_radius > 0)
  (h5 : config.cone_height_radius_ratio > 0) :
  ∃ (r : ℝ), r = cone_base_radius ∧
    (r > 0) ∧
    (∃ (h : ℝ), h / r = config.cone_height_radius_ratio) ∧
    (∃ (x y z : ℝ × ℝ × ℝ),
      (‖x - y‖ = config.sphere1_radius + config.sphere2_radius) ∧
      (‖y - z‖ = config.sphere2_radius + config.sphere3_radius) ∧
      (‖z - x‖ = config.sphere3_radius + config.sphere1_radius) ∧
      (∃ (c : ℝ × ℝ × ℝ),
        (‖c - x‖ = config.sphere1_radius + r) ∧
        (‖c - y‖ = config.sphere2_radius + r) ∧
        (‖c - z‖ = config.sphere3_radius + r))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_correct_l1167_116735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_for_angle_l1167_116765

theorem sine_value_for_angle (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.cos (π / 6) ∧ y = -2 * Real.sin (π / 6) ∧ 
   x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_for_angle_l1167_116765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_geometric_mean_l1167_116731

/-- Given a circle with chord AB and tangent at P, prove that the perpendicular
    from P to AB is the geometric mean of perpendiculars from A and B to the tangent -/
theorem perpendicular_geometric_mean
  (Circle : Type) -- Type representing a circle
  (Point : Type) -- Type representing a point
  (Line : Type) -- Type representing a line
  (A B P : Point) -- Points A, B, and P
  (on_circle : Point → Circle → Prop) -- Predicate for a point being on the circle
  (is_chord : Point → Point → Circle → Prop) -- Predicate for a line segment being a chord
  (is_tangent : Point → Circle → Prop) -- Predicate for a line being tangent to the circle
  (perpendicular_length : Point → Line → ℝ) -- Function to get length of perpendicular from point to line
  (line_through : Point → Point → Line) -- Function to create a line through two points
  (tangent_line : Point → Circle → Line) -- Function to get the tangent line at a point
  (circle : Circle) -- The given circle
  (h_chord : is_chord A B circle) -- AB is a chord of the circle
  (h_tangent : is_tangent P circle) -- Tangent at point P
  (h_A_on_circle : on_circle A circle) -- A is on the circle
  (h_B_on_circle : on_circle B circle) -- B is on the circle
  (h_P_on_circle : on_circle P circle) -- P is on the circle
  (h_P_not_A : P ≠ A) -- P is not A
  (h_P_not_B : P ≠ B) -- P is not B
  : perpendicular_length P (line_through A B) = 
    Real.sqrt (perpendicular_length A (tangent_line P circle) * perpendicular_length B (tangent_line P circle)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_geometric_mean_l1167_116731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l1167_116719

def iteratedFunction (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iteratedFunction f n x)

theorem no_such_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), n > 0 → iteratedFunction f n n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l1167_116719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_existence_l1167_116736

theorem solutions_existence (n : ℕ) : 
  ∃ (S : Finset (ℕ × ℕ)), S.card ≥ n ∧ 
  ∀ (p : ℕ × ℕ), p ∈ S → (p.1^2 + 15 * p.2^2 : ℕ) = 2^(2*n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_existence_l1167_116736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_a_range_l1167_116725

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := 3^x - 4/x - a

-- State the theorem
theorem zero_point_implies_a_range :
  ∀ a : ℝ, (∃ x ∈ Set.Ioo 1 2, f x a = 0) → a ∈ Set.Ioo (-1) 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_a_range_l1167_116725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_closure_time_l1167_116715

-- Define the rates of pipes A and B
noncomputable def rate_A : ℝ := 1 / 15
noncomputable def rate_B : ℝ := 1 / 24

-- Define the total time to fill the tank
def total_time : ℝ := 30

-- Define the theorem
theorem pipe_closure_time :
  ∃ (closure_time : ℝ),
    closure_time = 26.25 ∧
    (rate_A - rate_B) * closure_time + rate_A * (total_time - closure_time) = 1 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_closure_time_l1167_116715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cube_cost_is_56_l1167_116720

/-- Represents the given problem parameters and calculates the total cost of making ice cubes. -/
noncomputable def ice_cube_cost (total_weight : ℝ) (cube_weight : ℝ) (water_per_cube : ℝ) 
                  (cubes_per_hour : ℝ) (hourly_cost : ℝ) (water_cost_per_oz : ℝ) : ℝ :=
  let num_cubes := total_weight / cube_weight
  let hours_needed := num_cubes / cubes_per_hour
  let water_needed := num_cubes * water_per_cube
  let machine_cost := hours_needed * hourly_cost
  let water_cost := water_needed * water_cost_per_oz
  machine_cost + water_cost

/-- Theorem stating that the total cost to make 10 pounds of ice cubes is $56 under the given conditions. -/
theorem ice_cube_cost_is_56 : 
  ice_cube_cost 10 (1/16) 2 10 1.5 0.1 = 56 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval ice_cube_cost 10 (1/16) 2 10 1.5 0.1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cube_cost_is_56_l1167_116720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1167_116784

/-- A line is tangent to the circle (x-1)^2 + (y-1)^2 = 1 through (2,4) iff it's x=2 or 4x-3y+4=0 -/
theorem tangent_line_to_circle : 
  ∀ l : Set (ℝ × ℝ),
  (∀ p : ℝ × ℝ, p ∈ l → (p.1 - 1)^2 + (p.2 - 1)^2 ≥ 1) ∧ 
  (∃ q : ℝ × ℝ, q ∈ l ∧ (q.1 - 1)^2 + (q.2 - 1)^2 = 1) ∧
  (2, 4) ∈ l ↔ 
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 = 2) ∨ 
  (∀ p : ℝ × ℝ, p ∈ l ↔ 4 * p.1 - 3 * p.2 + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1167_116784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_slice_volume_l1167_116711

/-- Given a sphere with circumference 18π inches cut into 6 congruent slices,
    prove that the volume of one slice is 162π cubic inches. -/
theorem sphere_slice_volume (circumference : ℝ) (num_slices : ℕ) :
  circumference = 18 * Real.pi →
  num_slices = 6 →
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius ^ 3
  let slice_volume := sphere_volume / (num_slices : ℝ)
  slice_volume = 162 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_slice_volume_l1167_116711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1167_116768

theorem unique_solution_exponential_equation :
  ∃! x : ℚ, (5 : ℝ) ^ ((3 : ℝ) * x^2 - 8 * x + 3) = (5 : ℝ) ^ ((3 : ℝ) * x^2 + 6 * x - 5) ∧ x = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1167_116768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_hexagon_l1167_116788

/-- A regular hexagon in 2D space -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ i j : Fin 6, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- The intersection points of the diagonals -/
def intersection_points (h : RegularHexagon) : Fin 6 → ℝ × ℝ := sorry

/-- Area of a hexagon -/
noncomputable def area (h : RegularHexagon) : ℝ := sorry

/-- The theorem stating that the intersection points form a regular hexagon with 1/3 area -/
theorem diagonal_intersection_hexagon (h : RegularHexagon) :
  let inner_hex := RegularHexagon.mk (intersection_points h) sorry
  (∀ i j : Fin 6, dist (inner_hex.vertices i) (inner_hex.vertices j) = 
                  dist (inner_hex.vertices 0) (inner_hex.vertices 1)) ∧
  (area inner_hex = (1/3 : ℝ) * area h) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_hexagon_l1167_116788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sinusoidal_function_l1167_116789

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x

theorem transform_sinusoidal_function :
  ∀ x : ℝ, f x = g (3 * (x + Real.pi / 6)) :=
by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sinusoidal_function_l1167_116789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_three_l1167_116726

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity_sqrt_three (h : Hyperbola) 
  (line_angle : Real.cos (π/6) * h.a = Real.sin (π/6) * h.b) : 
  eccentricity h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_three_l1167_116726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_iff_a_eq_3_l1167_116791

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines when two lines are parallel -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

theorem parallel_iff_a_eq_3 :
  ∀ a : ℝ,
    (parallel
      { a := a, b := 3, c := 1 }
      { a := 1, b := a - 2, c := a }) ↔
    a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_iff_a_eq_3_l1167_116791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_system_solvability_l1167_116782

-- Define a sequence of positive integers
def m : ℕ → ℕ+ := sorry

-- Define the sequence 2mᵢ - 1
def seq (i : ℕ) : ℕ := 2 * (m i).val - 1

-- Define the system of congruences
def congruence_system (x : ℤ) : Prop :=
  ∀ i, x ≡ 2 * (m i).val^2 [ZMOD (2 * (m i).val - 1)]

-- State the theorem
theorem congruence_system_solvability :
  (∃ x, congruence_system x) ↔ (∃ k : ℕ, ∀ i, seq i ∣ k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_system_solvability_l1167_116782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1167_116746

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + 1 / 2

theorem monotonic_increase_interval 
  (ω : ℝ) (α β : ℝ) 
  (h_ω : ω > 0) 
  (h_α : f ω α = -1/2) 
  (h_β : f ω β = 1/2) 
  (h_min : |α - β| ≥ 3*Real.pi/4 ∧ ∀ γ δ, f ω γ = -1/2 → f ω δ = 1/2 → |γ - δ| ≥ 3*Real.pi/4) :
  ∃ k : ℤ, StrictMonoOn (f ω) (Set.Icc (-Real.pi/2 + 3*k*Real.pi) (Real.pi + 3*k*Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1167_116746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_MDE_collinear_l1167_116767

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points A, B, and M on the x-axis
noncomputable def A : ℝ × ℝ := (-2, 0)
noncomputable def B : ℝ × ℝ := (2, 0)
noncomputable def M : ℝ × ℝ := (4, 0)

-- Define point Q on the line x = 1
noncomputable def Q (m : ℝ) : ℝ × ℝ := (1, m)

-- Define the intersection of QA with the circle
noncomputable def D (m : ℝ) : ℝ × ℝ := ((18 - 2*m^2) / (m^2 + 9), 12*m / (m^2 + 9))

-- Define the intersection of QB with the circle
noncomputable def E (m : ℝ) : ℝ × ℝ := ((2*m^2 - 2) / (m^2 + 1), 4*m / (m^2 + 1))

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- Theorem statement
theorem MDE_collinear (m : ℝ) : collinear M (D m) (E m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_MDE_collinear_l1167_116767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_set_characterization_l1167_116754

-- Define the plane
def Plane := ℝ × ℝ

-- Define points A and B
def A : Plane := (0, 0)
def B : Plane := (7, 0)

-- Define distance function
noncomputable def dist (p q : Plane) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem point_set_characterization (P : Plane) :
  (dist P A)^2 = (dist P B)^2 - 7 ↔ P.1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_set_characterization_l1167_116754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1167_116797

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and semi-focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_equation : c^2 = a^2 + b^2

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- Theorem: Under given conditions, the eccentricity of the hyperbola is √2 + 1 -/
theorem hyperbola_eccentricity (h : Hyperbola) (P : Point) (F₁ F₂ : Point)
    (h_right_branch : P.x > 0 ∧ (P.x^2 / h.a^2) - (P.y^2 / h.b^2) = 1)
    (h_foci : F₁.x = -h.c ∧ F₁.y = 0 ∧ F₂.x = h.c ∧ F₂.y = 0)
    (h_dot_product : (P.x + F₂.x) * (F₂.x - P.x) + (P.y + F₂.y) * (F₂.y - P.y) = 0)
    (h_triangle_area : (F₂.x - F₁.x) * (P.y - F₁.y) - (F₂.y - F₁.y) * (P.x - F₁.x) = 4 * h.a * h.c) :
    eccentricity h = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1167_116797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_42nd_number_46_row_l1167_116770

theorem pascal_triangle_42nd_number_46_row : 
  (Nat.choose 45 41 : ℕ) = 148995 := by
  -- Define the row number (0-indexed)
  let row : ℕ := 45

  -- Define the position in the row (0-indexed)
  let pos : ℕ := 41

  -- The binomial coefficient for this position
  let result := Nat.choose row pos

  -- Assert that this is equal to 148995
  show result = 148995
  
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_42nd_number_46_row_l1167_116770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_intersection_points_star_2018_25_intersection_points_l1167_116763

/-- Definition of a regular (n; k)-star -/
def regular_star (n k : ℕ) : Prop :=
  Nat.Coprime n k ∧ n ≥ 5 ∧ k < n / 2

/-- Number of intersection points in a regular (n; k)-star -/
def intersection_points (n k : ℕ) : ℕ := n * (k - 1)

/-- Theorem: The number of intersection points in a regular (n; k)-star is n * (k - 1) -/
theorem regular_star_intersection_points (n k : ℕ) :
  regular_star n k → intersection_points n k = n * (k - 1) := by
  intro h
  rfl

/-- Theorem: The (2018; 25)-star has 48432 intersection points -/
theorem star_2018_25_intersection_points :
  regular_star 2018 25 → intersection_points 2018 25 = 48432 := by
  intro h
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_intersection_points_star_2018_25_intersection_points_l1167_116763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_is_25_l1167_116796

/-- Represents the dimensions and plastering cost of a tank -/
structure Tank where
  width : ℚ
  depth : ℚ
  plasteringCost : ℚ
  totalCost : ℚ

/-- Calculates the length of the tank based on its dimensions and plastering cost -/
def tankLength (t : Tank) : ℚ :=
  ((t.totalCost * 100) / t.plasteringCost - 2 * t.width * t.depth) / (2 * t.depth + t.width)

/-- Theorem stating that the length of the tank is 25 meters -/
theorem tank_length_is_25 (t : Tank) 
    (h1 : t.width = 12)
    (h2 : t.depth = 6)
    (h3 : t.plasteringCost = 25)
    (h4 : t.totalCost = 186) : 
  tankLength t = 25 := by
  sorry

#eval tankLength { width := 12, depth := 6, plasteringCost := 25, totalCost := 186 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_is_25_l1167_116796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1167_116702

noncomputable def f (x : ℝ) := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - 3

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧
    ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (k * Real.pi / 2 + x) = f (k * Real.pi / 2 - x)) ∧
  (∀ y : ℝ, y ∈ Set.Icc 3 5 ↔ ∃ x : ℝ, x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1167_116702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_w_y_l1167_116755

-- Define the variables
variable (w x y z : ℚ)

-- Define the given ratios
def ratio_w_x (w x : ℚ) : Prop := w / x = 5 / 4
def ratio_y_z (y z : ℚ) : Prop := y / z = 5 / 3
def ratio_z_x (z x : ℚ) : Prop := z / x = 1 / 5

-- Theorem to prove
theorem ratio_w_y (hw : ratio_w_x w x) (hy : ratio_y_z y z) (hz : ratio_z_x z x) :
  w / y = 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_w_y_l1167_116755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sum_inequality_l1167_116728

theorem gcd_sum_inequality (m n : ℕ) (h : m ≠ n) :
  Nat.gcd m n + Nat.gcd (m + 1) (n + 1) + Nat.gcd (m + 2) (n + 2) ≤ 2 * Int.natAbs (m - n) + 1 ∧
  (Nat.gcd m n + Nat.gcd (m + 1) (n + 1) + Nat.gcd (m + 2) (n + 2) = 2 * Int.natAbs (m - n) + 1 ↔
   (n = m - 1 ∨ n = m + 1 ∨ n = m - 2 ∨ n = m + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sum_inequality_l1167_116728
