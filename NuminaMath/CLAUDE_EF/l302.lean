import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l302_30282

/-- A rectangle with dimensions 6 by 4 units -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_eq : width = 6
  height_eq : height = 4

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle DEF defined by its vertices -/
structure TriangleDEF where
  D : Point
  E : Point
  F : Point
  D_eq : D = ⟨0, 2⟩
  E_eq : E = ⟨6, 0⟩
  F_eq : F = ⟨3, 4⟩

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- Theorem stating that the area of triangle DEF is 6 square units -/
theorem triangle_DEF_area (r : Rectangle) (t : TriangleDEF) :
  triangleArea t.D t.E t.F = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l302_30282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seats_per_row_l302_30224

/-- Proves the number of seats in each row of an auditorium --/
theorem seats_per_row (rows : ℕ) (ticket_price : ℚ) (sold_fraction : ℚ) (total_earnings : ℚ) 
  (seats : ℕ) :
  rows = 20 →
  ticket_price = 10 →
  sold_fraction = 3/4 →
  total_earnings = 1500 →
  (sold_fraction * (rows * seats) * ticket_price = total_earnings) →
  seats = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seats_per_row_l302_30224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_on_line_l302_30200

noncomputable def z (a : ℝ) : ℂ := (1 - a * Complex.I) / Complex.I

theorem complex_on_line (a : ℝ) : 
  (z a).re + 2 * (z a).im + 5 = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_on_line_l302_30200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_theorem_l302_30235

theorem fraction_sum_theorem : ∃ (a₂ a₃ a₄ a₅ a₆ : ℤ),
  (3 : ℚ) / 5 = a₂ / 2 + a₃ / 6 + a₄ / 24 + a₅ / 120 + a₆ / 720 ∧
  (0 ≤ a₂ ∧ a₂ < 3) ∧
  (0 ≤ a₃ ∧ a₃ < 4) ∧
  (0 ≤ a₄ ∧ a₄ < 5) ∧
  (0 ≤ a₅ ∧ a₅ < 6) ∧
  (0 ≤ a₆ ∧ a₆ < 7) ∧
  a₂ + a₃ + a₄ + a₅ + a₆ = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_theorem_l302_30235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l302_30241

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 - 1/2

theorem f_properties :
  ∀ x : ℝ,
  (f x = Real.sin (2*x + π/6)) ∧
  (∃ p : ℝ, p > 0 ∧ ∀ y : ℝ, f (y + p) = f y ∧ ∀ q : ℝ, q > 0 ∧ (∀ z : ℝ, f (z + q) = f z) → p ≤ q) ∧
  (f x = Real.sin (2*(x + π/12))) ∧
  (∀ y z : ℝ, -π/3 < y ∧ y < z ∧ z < π/6 → f y < f z) ∧
  (f (π/6) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l302_30241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_F_l302_30275

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

def g (k : ℝ) (x : ℝ) : ℝ := k * x

noncomputable def F (k : ℝ) (x : ℝ) : ℝ := max (f x) (g k x)

theorem max_value_F (k : ℝ) (h_k : k > 0) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), F k x ≤ k * (Real.exp 1)) ∧ 
  (∃ x ∈ Set.Icc 1 (Real.exp 1), F k x = k * (Real.exp 1)) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), F k x ≤ 1 / (Real.exp 1)) ∧ 
  (∃ x ∈ Set.Icc 1 (Real.exp 1), F k x = 1 / (Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_F_l302_30275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l302_30237

-- Define the circle C
def circleC (x y r : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = r^2

-- Define the condition |MA| = √2 |MO|
def condition (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 2 * (x^2 + y^2)

-- Define the range of r
def r_range (r : ℝ) : Prop := 3 * Real.sqrt 5 - 3 * Real.sqrt 2 ≤ r ∧ r ≤ 3 * Real.sqrt 5 + 3 * Real.sqrt 2

-- Theorem statement
theorem circle_intersection_range :
  ∀ r : ℝ, r > 0 →
  (∃ x y : ℝ, circleC x y r ∧ condition x y) ↔ r_range r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l302_30237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l302_30296

-- Define the function f(x) = 2 - xe^x
noncomputable def f (x : ℝ) : ℝ := 2 - x * Real.exp x

-- Define the point of tangency
def point : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem tangent_line_equation :
  let df := λ x => -(1 + x) * Real.exp x  -- f'(x)
  let slope := df point.fst
  let tangent_eq := λ x y => x + y - 2 = 0
  tangent_eq point.fst point.snd ∧ 
  ∀ x y, tangent_eq x y ↔ y - point.snd = slope * (x - point.fst) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l302_30296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l302_30246

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (1, -2)
def center2 : ℝ × ℝ := (3, -1)

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 1

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > abs (radius1 - radius2) ∧
  distance_between_centers < radius1 + radius2 :=
by
  apply And.intro
  · -- Proof that distance_between_centers > abs (radius1 - radius2)
    sorry
  · -- Proof that distance_between_centers < radius1 + radius2
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l302_30246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_of_sequence_l302_30234

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem sixth_term_of_sequence
  (a₁ : ℝ) (d : ℝ)
  (h_sum : sum_arithmetic_sequence a₁ d 5 = 15)
  (h_fourth : arithmetic_sequence a₁ d 4 = 4) :
  arithmetic_sequence a₁ d 6 = 6 := by
  sorry

#check sixth_term_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_of_sequence_l302_30234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_D_to_D_l302_30212

-- Define the points D and D'
def D : ℝ × ℝ := (2, 5)
def D' : ℝ × ℝ := (2, -5)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_D_to_D'_is_10 :
  distance D D' = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_D_to_D_l302_30212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_num_solutions_correct_l302_30293

/-- The number of solutions to the equation a^x = x, where a is a positive real number -/
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if a > Real.exp (1 / Real.exp 1) then 0
  else if 0 < a ∧ a ≤ 1 ∨ a = Real.exp (1 / Real.exp 1) then 1
  else if 1 < a ∧ a < Real.exp (1 / Real.exp 1) then 2
  else 0

theorem solutions_count (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, a^x = x) ↔ num_solutions a > 0 := by
  sorry

theorem num_solutions_correct (a : ℝ) (h : a > 0) :
  (num_solutions a = 0 ↔ a > Real.exp (1 / Real.exp 1)) ∧
  (num_solutions a = 1 ↔ (0 < a ∧ a ≤ 1 ∨ a = Real.exp (1 / Real.exp 1))) ∧
  (num_solutions a = 2 ↔ (1 < a ∧ a < Real.exp (1 / Real.exp 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_num_solutions_correct_l302_30293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_is_half_l302_30256

-- Define the type of functions from [0,1] to ℝ
def UnitIntervalFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ∈ Set.Icc 0 1}

-- Define the property that f(0) = f(1)
def HasEqualEndpoints (f : UnitIntervalFunction) : Prop :=
  f.val 0 = f.val 1

-- Define the contraction property
def IsContraction (f : UnitIntervalFunction) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → |f.val x - f.val y| < |x - y|

-- Define the theorem
theorem smallest_upper_bound_is_half :
  ∃ m : ℝ, m = 1/2 ∧
  (∀ f : UnitIntervalFunction, HasEqualEndpoints f → IsContraction f →
    ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → |f.val x - f.val y| < m) ∧
  (∀ ε > 0, ∃ f : UnitIntervalFunction, HasEqualEndpoints f ∧ IsContraction f ∧
    ∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ |f.val x - f.val y| > m - ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_is_half_l302_30256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_350_units_l302_30203

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 5 then -0.5 * x^2 + 3.5 * x - 0.5
  else if x > 5 then 17 - 2.5 * x
  else 0

-- State the theorem
theorem max_profit_at_350_units :
  ∃ (max_profit : ℝ),
    (∀ x, profit x ≤ max_profit) ∧
    profit 3.5 = max_profit ∧
    max_profit = 5.625 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_350_units_l302_30203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_from_chord_ratio_l302_30223

/-- Two lines with slopes that are negative reciprocals of each other -/
def are_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A line with slope m passing through point (1,1) -/
def line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - 1 = m * (p.1 - 1)}

/-- The circle x^2 + y^2 = 4 -/
def unit_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The chord length for a line intersecting the circle -/
noncomputable def chord_length (m : ℝ) : ℝ :=
  2 * Real.sqrt (4 - (m - 1)^2 / (m^2 + 1))

/-- The theorem stating the relationship between the slopes and chord lengths -/
theorem slopes_product_from_chord_ratio (m₁ m₂ : ℝ) :
  are_perpendicular m₁ m₂ →
  (1, 1) ∈ line m₁ ∩ line m₂ →
  chord_length m₁ / chord_length m₂ = Real.sqrt 6 / 2 →
  m₁ * m₂ = -9 ∨ m₁ * m₂ = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_from_chord_ratio_l302_30223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_sum_l302_30288

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 6 then x^2 - 4*x + 4
  else if x > 0 then -x^2 + 3*x + 4
  else 4*x + 8

theorem piecewise_function_sum : f (-5) + f 3 + f 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_sum_l302_30288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_a_triangle_inequality_b_l302_30295

-- Part (a)
theorem triangle_inequality_a (X A B C : ℝ × ℝ) (a b c : ℝ) :
  let XA := Real.sqrt ((X.1 - A.1)^2 + (X.2 - A.2)^2)
  let XB := Real.sqrt ((X.1 - B.1)^2 + (X.2 - B.2)^2)
  let XC := Real.sqrt ((X.1 - C.1)^2 + (X.2 - C.2)^2)
  a > 0 ∧ b > 0 ∧ c > 0 →
  (XB / b) * (XC / c) + (XC / c) * (XA / a) + (XA / a) * (XB / b) ≥ 1 :=
by sorry

-- Part (b)
theorem triangle_inequality_b (A B C A1 B1 C1 : ℝ × ℝ) (a b c a1 b1 c1 S : ℝ) :
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CA := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  a = BC ∧ b = CA ∧ c = AB ∧
  A1 ∈ Set.Icc B C ∧ B1 ∈ Set.Icc C A ∧ C1 ∈ Set.Icc A B ∧
  S = (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) →
  4 * S^2 ≤ a^2 * b1 * c1 + b^2 * a1 * c1 + c^2 * a1 * b1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_a_triangle_inequality_b_l302_30295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l302_30226

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define a point on the ellipse
def point_on_ellipse (M : ℝ × ℝ) : Prop := ellipse M.1 M.2

-- Define tangent points on the circle
def tangent_points (M A B : ℝ × ℝ) : Prop :=
  my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧
  (A.1 - M.1)^2 + (A.2 - M.2)^2 = (B.1 - M.1)^2 + (B.2 - M.2)^2

-- Define the line passing through A and B
def line_through (A B P Q : ℝ × ℝ) : Prop :=
  (P.2 = 0 ∨ Q.1 = 0) ∧
  (B.2 - A.2) * (P.1 - A.1) = (B.1 - A.1) * (P.2 - A.2) ∧
  (B.2 - A.2) * (Q.1 - A.1) = (B.1 - A.1) * (Q.2 - A.2)

-- Define the area of triangle POQ
def triangle_area (P Q : ℝ × ℝ) : ℝ := (1/2) * abs (P.1 * Q.2)

-- Theorem statement
theorem min_triangle_area :
  ∀ (M A B P Q : ℝ × ℝ),
    point_on_ellipse M →
    tangent_points M A B →
    line_through A B P Q →
    triangle_area P Q ≥ 2/3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l302_30226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_properties_l302_30267

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- Length of the median to the hypotenuse -/
  median_length : ℝ
  /-- Assumption that the median length is positive -/
  median_positive : median_length > 0

/-- Calculate the length of a leg of an isosceles right triangle -/
noncomputable def leg_length (t : IsoscelesRightTriangle) : ℝ :=
  t.median_length * Real.sqrt 2

/-- Calculate the area of an isosceles right triangle -/
def triangle_area (t : IsoscelesRightTriangle) : ℝ :=
  (t.median_length^2) * 2

/-- Theorem stating the properties of an isosceles right triangle with median length 12 -/
theorem isosceles_right_triangle_properties :
  let t : IsoscelesRightTriangle := ⟨12, by norm_num⟩
  (leg_length t = 12 * Real.sqrt 2) ∧ (triangle_area t = 144) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_properties_l302_30267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l302_30215

-- Define the function f(x) = 2^(-x) - 3x
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x * Real.log 2) - 3*x

-- State the theorem
theorem range_of_a_for_inequality (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l302_30215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l302_30286

-- Define the right prism with isosceles trapezoid base
structure RightPrism where
  l : ℝ  -- length of the segment connecting upper vertex to center of circumscribed circle
  α : ℝ  -- angle between diagonals of the trapezoid
  β : ℝ  -- angle between l and the base plane

-- Define the volume function
noncomputable def volume (p : RightPrism) : ℝ :=
  2 * p.l^3 * (Real.cos p.β)^2 * (Real.cos (p.α/2))^2 * Real.sin p.α * Real.sin p.β

-- State the theorem
theorem right_prism_volume (p : RightPrism) :
  volume p = 2 * p.l^3 * (Real.cos p.β)^2 * (Real.cos (p.α/2))^2 * Real.sin p.α * Real.sin p.β :=
by
  -- Unfold the definition of volume
  unfold volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l302_30286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ADEC_is_135_l302_30247

-- Define the triangle and points
structure Triangle :=
  (A B C : ℝ × ℝ)

noncomputable def D (t : Triangle) : ℝ × ℝ := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

noncomputable def E (t : Triangle) : ℝ × ℝ := sorry
noncomputable def F (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions
def is_right_triangle (t : Triangle) : Prop :=
  (t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2) = 0

noncomputable def length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_perpendicular (p q r s : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (s.1 - r.1) + (q.2 - p.2) * (s.2 - r.2) = 0

noncomputable def area_quadrilateral (p q r s : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ADEC_is_135 (t : Triangle) :
  is_right_triangle t →
  length t.A t.B = 24 →
  length t.A t.C = 15 →
  is_perpendicular (D t) (E t) t.A t.B →
  is_perpendicular t.C (F t) t.A t.B →
  F t ≠ D t →
  area_quadrilateral t.A (D t) (E t) t.C = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ADEC_is_135_l302_30247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l302_30276

theorem trigonometric_identity (θ : ℝ) : 
  Real.tan (-2019 * Real.pi + θ) = -2 → 
  2 * Real.sqrt 2 * Real.sin (θ - Real.pi / 6) * Real.sin (θ + Real.pi / 4) = (2 * Real.sqrt 3 + 1) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l302_30276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_sum_of_distances_l302_30262

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Sum of distances from a point to two fixed points -/
noncomputable def sumOfDistances (a b c : Point) : ℝ :=
  distance a c + distance b c

/-- The point that minimizes the sum of distances -/
def minimalPoint (a b : Point) : Point :=
  Point.mk 0 (-1)

theorem minimal_sum_of_distances :
  let a := Point.mk 7 6
  let b := Point.mk 1 (-2)
  let c := minimalPoint a b
  ∀ k : ℝ, sumOfDistances a b c ≤ sumOfDistances a b (Point.mk 0 k) := by
  sorry

#check minimal_sum_of_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_sum_of_distances_l302_30262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_remaining_l302_30263

def book_length : ℕ := 600
def first_week_fraction : ℚ := 1/2
def second_week_fraction : ℚ := 3/10
def third_week_hours : ℕ := 10
def reading_speed : ℕ := 15

theorem pages_remaining : 
  book_length - 
  (first_week_fraction * book_length).floor - 
  (second_week_fraction * (book_length - (first_week_fraction * book_length).floor)).floor - 
  (third_week_hours * reading_speed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_remaining_l302_30263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_correct_l302_30277

/-- The number of 6-digit numbers composed of digits 1, 2, and 3, where each digit appears at least once -/
def count_special_numbers : ℕ := 390

theorem count_special_numbers_correct :
  count_special_numbers = 390 :=
by
  -- Unfold the definition of count_special_numbers
  unfold count_special_numbers
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_correct_l302_30277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_one_root_range_l302_30232

/-- The logarithm function (base 10) -/
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The function representing the left side of the equation -/
noncomputable def f (x : ℝ) : ℝ := log (x - 1) + log (3 - x)

/-- The function representing the right side of the equation -/
noncomputable def g (a x : ℝ) : ℝ := log (a - x)

/-- The theorem stating the range of a for which the equation has exactly one real root -/
theorem equation_one_root_range (a : ℝ) :
  (∃! x, f x = g a x ∧ 1 < x ∧ x < 3) ↔ 1 < a ∧ a ≤ 13/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_one_root_range_l302_30232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_sum_l302_30264

def cards : Finset Int := {-2, -1, 0, 3, 4}

def is_odd (n : Int) : Bool := n % 2 ≠ 0

def sum_is_odd (a b : Int) : Bool := is_odd (a + b)

def total_outcomes : Nat := cards.card * (cards.card - 1)

def favorable_outcomes : Nat := 
  cards.card * (cards.card - 1) - 
  (cards.filter (λ x => is_odd x)).card * ((cards.filter (λ x => is_odd x)).card - 1) - 
  (cards.filter (λ x => ¬is_odd x)).card * ((cards.filter (λ x => ¬is_odd x)).card - 1)

theorem probability_of_odd_sum : 
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_sum_l302_30264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_distribution_problem_l302_30258

/-- The price of a type A pen in yuan -/
noncomputable def price_A : ℚ := 7/2

/-- The price of a type B pen in yuan -/
noncomputable def price_B : ℚ := 53/20

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The total number of pens -/
def total_pens : ℕ := 3 * num_students + 2

/-- The number of type A pens (minimum possible) -/
def num_A_pens : ℕ := 1

/-- The number of type B pens -/
def num_B_pens : ℕ := total_pens - num_A_pens

/-- The minimum contribution per student in yuan -/
noncomputable def min_contribution : ℚ := (price_A * num_A_pens + price_B * num_B_pens) / num_students

theorem pen_distribution_problem :
  num_students = 15 ∧
  3 * num_students + 2 = total_pens ∧
  4 * num_students - 13 = total_pens ∧
  min_contribution = 209/25 := by
  sorry

#eval num_students
#eval total_pens
#eval num_B_pens

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_distribution_problem_l302_30258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_section_area_l302_30270

-- Define the area of the equidistant section
noncomputable def area_of_equidistant_section (B b : ℝ) : ℝ :=
  (B + 2 * Real.sqrt B * Real.sqrt b + b) / 4

theorem truncated_pyramid_section_area 
  (B b : ℝ) 
  (hB : B > 0) 
  (hb : b > 0) : 
  ∃ x : ℝ, x = area_of_equidistant_section B b ∧ 
  x = (B + 2 * Real.sqrt B * Real.sqrt b + b) / 4 :=
by
  use area_of_equidistant_section B b
  constructor
  · rfl
  · rfl

#check truncated_pyramid_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_section_area_l302_30270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_theorem_l302_30259

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (20, 0)
def C (y : ℝ) : ℝ × ℝ := (0, y)

-- Define the volumes of cones M and N
noncomputable def volume_M (y : ℝ) : ℝ := (1/3) * Real.pi * y^2 * 20
noncomputable def volume_N (y : ℝ) : ℝ := (1/3) * Real.pi * 20^2 * y

-- Define the length of BC
noncomputable def length_BC (y : ℝ) : ℝ := Real.sqrt ((20 - 0)^2 + (y - 0)^2)

-- Theorem statement
theorem triangle_rotation_theorem (y : ℝ) (h1 : y > 0) 
  (h2 : volume_M y - volume_N y = 140 * Real.pi) : 
  length_BC y = 29 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_theorem_l302_30259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_sum_of_squares_l302_30294

theorem triangle_minimum_sum_of_squares (a b c : ℝ) (A B C : Real) :
  (a + b)^2 = 10 + c^2 →
  Real.cos C = 2/3 →
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), x^2 + y^2 ≥ min := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_sum_of_squares_l302_30294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_impact_point_l302_30278

/-- Free fall problem -/
theorem free_fall_impact_point (g R r : ℝ) (α : ℝ) :
  let V := Real.sqrt (2 * g * r * Real.cos α)
  let x (t : ℝ) := R * Real.sin α + V * Real.cos α * t
  let y (t : ℝ) := R * (1 - Real.cos α) + V * Real.sin α * t - g * t^2 / 2
  let T := Real.sqrt (2 * R / g) * (Real.sin α * Real.sqrt (Real.cos α) + Real.sqrt (1 - Real.cos α^3))
  x T = R * (Real.sin α + Real.sin (2 * α) + Real.sqrt (Real.cos α * (1 - Real.cos α^3))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_impact_point_l302_30278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_A_l302_30245

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 0, 4],
    ![0, 2, 2],
    ![4, 2, 2]]

theorem eigenvalues_of_A :
  ∃ (v1 v2 v3 : Fin 3 → ℝ),
    v1 ≠ 0 ∧ v2 ≠ 0 ∧ v3 ≠ 0 ∧
    A.mulVec v1 = (2 : ℝ) • v1 ∧
    A.mulVec v2 = (4 + 2 * Real.sqrt 5 : ℝ) • v2 ∧
    A.mulVec v3 = (4 - 2 * Real.sqrt 5 : ℝ) • v3 ∧
    ∀ (k : ℝ) (v : Fin 3 → ℝ),
      v ≠ 0 → A.mulVec v = k • v →
        k = 2 ∨ k = 4 + 2 * Real.sqrt 5 ∨ k = 4 - 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_A_l302_30245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l302_30244

/-- Given vectors a, b, and c in ℝ², prove that if λa + b is collinear with c, then λ = -1 -/
theorem collinear_vectors_lambda (a b c : ℝ × ℝ) (l : ℝ) : 
  a = (1, 2) → b = (2, 0) → c = (1, -2) → 
  (∃ (k : ℝ), l • a + b = k • c) → 
  l = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l302_30244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l302_30271

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2*x

noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - x - 2

theorem tangent_line_at_2 : 
  ∃ (m b : ℝ), m = f' 2 ∧ b = f 2 - m * 2 ∧ 
  ∀ (x y : ℝ), y = m * x + b ↔ y - f 2 = m * (x - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l302_30271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l302_30225

theorem sum_of_roots_quadratic (x : ℝ) : 
  (3 * x^2 - 15 * x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, 3 * y^2 - 15 * y = 0) ∧ (s.sum id = 5)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l302_30225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l302_30218

/-- The area of a triangle with vertices at (1,1), (1,6), and (8,9) is 17.5 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 17.5 := by
  -- Define the vertices of the triangle
  let x₁ : ℝ := 1
  let y₁ : ℝ := 1
  let x₂ : ℝ := 1
  let y₂ : ℝ := 6
  let x₃ : ℝ := 8
  let y₃ : ℝ := 9

  -- Define the area calculation
  let area := (1/2 : ℝ) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|

  -- State the existence of the area and its value
  use area
  
  -- Prove that the area equals 17.5
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l302_30218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_with_digit_sum_22_l302_30238

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_prime_with_digit_sum_22 :
  Nat.Prime 499 ∧
  digit_sum 499 = 22 ∧
  ∀ p : ℕ, p < 499 → Nat.Prime p → digit_sum p ≠ 22 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_with_digit_sum_22_l302_30238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_element_ring_field_equiv_l302_30291

/-- A ring with 9 elements -/
structure NineElementRing (A : Type) [Ring A] :=
  (nine_elements : ∃ (s : Finset A), s.card = 9 ∧ ∀ a : A, a ∈ s)

/-- Property that for all non-zero x, there exist a and b satisfying the equation -/
def SatisfiesQuadratic (A : Type) [Ring A] : Prop :=
  ∀ x : A, x ≠ 0 → ∃ a b : A, a ∈ ({-1, 0, 1} : Set A) ∧ b ∈ ({-1, 1} : Set A) ∧ x^2 + a*x + b = 0

/-- The main theorem -/
theorem nine_element_ring_field_equiv {A : Type} [Ring A] (h : NineElementRing A) :
  SatisfiesQuadratic A ↔ IsField A :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_element_ring_field_equiv_l302_30291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_18_consecutive_good_numbers_l302_30221

/-- A natural number is good if it has exactly two prime divisors -/
def is_good (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n ∧
  ∀ r : ℕ, Nat.Prime r → r ∣ n → (r = p ∨ r = q)

/-- Theorem: There does not exist a sequence of 18 consecutive natural numbers where each number is good -/
theorem no_18_consecutive_good_numbers :
  ¬ ∃ k : ℕ, ∀ i : ℕ, i < 18 → is_good (k + i) :=
by
  sorry

#check no_18_consecutive_good_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_18_consecutive_good_numbers_l302_30221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_contribution_sum_l302_30289

/-- Calculates the sum of a finite geometric series -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Proves that the total charity contribution over a year is approximately $21839.99 -/
theorem charity_contribution_sum : 
  let initial_donation : ℝ := 1453
  let monthly_increase : ℝ := 1.04
  let months : ℕ := 12
  abs (geometric_sum initial_donation monthly_increase months - 21839.99) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_contribution_sum_l302_30289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_properties_l302_30272

-- Define the golden ratio
noncomputable def golden_ratio : ℝ := (Real.sqrt 5 - 1) / 2

-- Define an isosceles triangle with base angle 72°
structure IsoscelesTriangle72 where
  base : ℝ
  side : ℝ
  base_angle : ℝ
  base_angle_eq : base_angle = 72 * Real.pi / 180

-- Theorem statement
theorem golden_ratio_properties (t : IsoscelesTriangle72) : 
  -- The ratio of the smaller part to the larger part in a golden ratio division
  golden_ratio = (Real.sqrt 5 - 1) / 2 ∧ 
  -- The segments a and b divide (a + b) in the golden ratio
  (t.base + t.side) / t.side = t.side / t.base := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_properties_l302_30272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_avg_profit_maximization_reduced_avg_profit_maximization_l302_30297

noncomputable section

/-- Revenue function --/
def R (x : ℝ) : ℝ := -1/2 * x^2 + 400 * x

/-- Daily profit function --/
def L (x : ℝ) : ℝ := R x - (100 * x + 20000)

/-- Average profit function --/
def P (x : ℝ) : ℝ := L x / x

/-- Maximum daily production capacity --/
def max_capacity : ℝ := 360

/-- Reduced maximum daily production capacity --/
def reduced_max_capacity : ℝ := 160

theorem profit_maximization (x : ℝ) (h : 0 < x ∧ x ≤ max_capacity) :
  (∀ y, 0 < y ∧ y ≤ max_capacity → L y ≤ L 300) ∧
  L 300 = 25000 := by
  sorry

theorem avg_profit_maximization (x : ℝ) (h : 0 < x ∧ x ≤ max_capacity) :
  (∀ y, 0 < y ∧ y ≤ max_capacity → P y ≤ P 200) ∧
  P 200 = 100 := by
  sorry

theorem reduced_avg_profit_maximization (x : ℝ) (h : 0 < x ∧ x ≤ reduced_max_capacity) :
  (∀ y, 0 < y ∧ y ≤ reduced_max_capacity → P y ≤ P 160) ∧
  P 160 = 95 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_avg_profit_maximization_reduced_avg_profit_maximization_l302_30297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_finite_set_with_integer_geometric_mean_no_infinite_set_with_integer_geometric_mean_l302_30250

-- Define the geometric mean
noncomputable def geometric_mean (S : Finset ℕ+) : ℝ :=
  (S.prod id) ^ (1 / S.card : ℝ)

-- Theorem 1: For any positive integer n, there exists a set S_n
theorem exists_finite_set_with_integer_geometric_mean (n : ℕ+) :
  ∃ (S : Finset ℕ+), S.card = n ∧ ∀ (T : Finset ℕ+), T ⊆ S → ∃ (m : ℕ+), geometric_mean T = m := by
  sorry

-- Theorem 2: No such infinite set exists
theorem no_infinite_set_with_integer_geometric_mean :
  ¬∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ (T : Finset ℕ+), ↑T ⊆ S → ∃ (m : ℕ+), geometric_mean T = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_finite_set_with_integer_geometric_mean_no_infinite_set_with_integer_geometric_mean_l302_30250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_half_l302_30208

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁))

/-- The area of a triangle ABC with given coordinates -/
theorem triangle_area_half (A B C : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (1424233, 2848467) → 
  C = (1424234, 2848469) → 
  area_triangle A B C = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_half_l302_30208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_profit_l302_30252

/-- Given a manufacturing firm with the following conditions:
  - Daily maintenance cost is $800
  - Each worker is paid $20 per hour
  - Each worker makes 4 widgets per hour
  - Each widget is sold for $4.00
  - Workday is 10 hours long
  This theorem proves that the minimum number of workers needed to turn a profit is 21. -/
theorem min_workers_for_profit : ℕ := by
  -- Define constants
  let maintenance_cost : ℝ := 800
  let hourly_wage : ℝ := 20
  let widgets_per_hour : ℝ := 4
  let widget_price : ℝ := 4
  let work_hours : ℝ := 10

  -- Define profit condition
  let profit_condition (n : ℕ) : Prop :=
    work_hours * widgets_per_hour * widget_price * n > maintenance_cost + work_hours * hourly_wage * n

  -- Define minimum number of workers
  let min_workers : ℕ := 21

  -- Prove that min_workers satisfies the profit condition
  have h1 : profit_condition min_workers := by sorry

  -- Prove that no smaller number of workers satisfies the profit condition
  have h2 : ∀ m : ℕ, m < min_workers → ¬profit_condition m := by sorry

  -- Return the result
  exact min_workers

#check min_workers_for_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_profit_l302_30252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_w_l302_30253

/-- The function w(x) -/
noncomputable def w (x : ℝ) : ℝ := (x - 1) ^ (1/3) + Real.sqrt (8 - x)

/-- The domain of w(x) -/
def domain_w : Set ℝ := {x | x ≤ 8}

/-- Theorem: The domain of w(x) is (-∞, 8] -/
theorem domain_of_w : {x : ℝ | ∃ y, w x = y} = domain_w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_w_l302_30253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_4334_cents_l302_30231

def rose_price : ℚ := 20
def lily_price : ℚ := 15
def sunflower_price : ℚ := 10

def rose_discount : ℚ := 15 / 100
def lily_discount : ℚ := 10 / 100

def sales_tax : ℚ := 7 / 100

def discounted_rose_price : ℚ := rose_price * (1 - rose_discount)
def discounted_lily_price : ℚ := lily_price * (1 - lily_discount)

def total_before_tax : ℚ := discounted_rose_price + discounted_lily_price + sunflower_price

def total_with_tax : ℚ := total_before_tax * (1 + sales_tax)

theorem total_cost_is_4334_cents : 
  (⌊total_with_tax * 100⌋ : ℚ) / 100 = 4334 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_4334_cents_l302_30231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l302_30236

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = Real.sin θ + 2 * Real.cos θ

-- Define what it means for a curve to be a circle in Cartesian coordinates
def is_circle (f : ℝ × ℝ → Prop) : Prop :=
  ∃ h k r, ∀ x y, f (x, y) ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  is_circle (λ (p : ℝ × ℝ) => ∃ θ, polar_equation (Real.sqrt (p.1^2 + p.2^2)) θ ∧ 
                                   p.1 = Real.sqrt (p.1^2 + p.2^2) * Real.cos θ ∧ 
                                   p.2 = Real.sqrt (p.1^2 + p.2^2) * Real.sin θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l302_30236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_returned_balls_is_correct_l302_30204

/-- Represents a circle of balls -/
def CircleOfBalls := Fin 8

/-- Probability of choosing adjacent balls -/
noncomputable def probAdjacent : ℝ := 2/3

/-- Probability of choosing balls two apart -/
noncomputable def probTwoApart : ℝ := 1/3

/-- Probability of a specific ball being involved in a swap -/
noncomputable def probInvolved : ℝ := 1/2

/-- Expected number of balls returning to their original positions -/
noncomputable def expectedReturnedBalls : ℝ := 2.5

/-- Theorem stating the expected number of balls returning to their original positions -/
theorem expected_returned_balls_is_correct : 
  expectedReturnedBalls = 
    (8 : ℝ) * (probInvolved^2 / 8 + (1 - probInvolved)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_returned_balls_is_correct_l302_30204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_l302_30299

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) := by sorry

-- Helper lemmas
lemma f_lower_bound (a x : ℝ) : f a x ≥ |a + 3| := by sorry

lemma case_analysis (a : ℝ) :
  (∀ x, f a x > -a) ↔ (a ≥ 0 ∨ (a < 0 ∧ a > -3/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_l302_30299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l302_30216

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 32 / (x + 4)

-- Define the function h in terms of g⁻¹
noncomputable def h (x : ℝ) : ℝ := 4 * (Function.invFun g x)

-- Theorem statement
theorem h_equality : h (32/9) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l302_30216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_valid_subset_l302_30205

def is_valid_subset (M : Finset ℕ) : Prop :=
  M ⊆ Finset.range 26 ∧
  ∀ A B : Finset ℕ, A ⊆ M → B ⊆ M → A ∩ B = ∅ → A ≠ B → (A.sum id) ≠ (B.sum id)

theorem max_sum_of_valid_subset :
  ∀ M : Finset ℕ, is_valid_subset M → (M.sum id) ≤ 123 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_valid_subset_l302_30205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_to_playground_distance_l302_30283

-- Define the step length in centimeters
def step_length : ℝ := 52

-- Define the number of steps
def num_steps : ℕ := 176

-- Define the conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem classroom_to_playground_distance :
  (step_length * (num_steps : ℝ) * cm_to_m) = 91.52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_to_playground_distance_l302_30283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l302_30290

/-- The time taken for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length > 0 →
  train_speed > 0 →
  man_speed > 0 →
  let relative_speed := train_speed + man_speed
  let time := train_length / (relative_speed * 1000 / 3600)
  ∃ ε > 0, |time - 6| < ε :=
by
  intros h_length h_train_speed h_man_speed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l302_30290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l302_30209

open Real Set

theorem trigonometric_equation_solutions :
  ∃! (n : ℕ), n = (({θ : ℝ | θ ∈ Ioo 0 (2*π) ∧ tan (7 * π * cos θ) = (tan (7 * π * sin θ))⁻¹}).ncard) ∧ n = 36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l302_30209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l302_30222

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

def Triangle := Point → Point → Point → Prop

def Circumcenter (t : Triangle) (o : Point) : Prop := sorry
def Orthocenter (t : Triangle) (m : Point) : Prop := sorry
def OnCircumcircle (t : Triangle) (p : Point) : Prop := sorry
def Perpendicular (l1 l2 : Point → Point → Prop) : Prop := sorry

-- Define the theorem
theorem triangle_construction 
  (a o m : Point) 
  (t : Triangle) 
  (h1 : Circumcenter t o) 
  (h2 : Orthocenter t m) 
  (h3 : t a a a) :
  ∃ (b c p : Point),
    t a b c ∧
    OnCircumcircle t b ∧ 
    OnCircumcircle t c ∧
    Perpendicular (λ x y => x = a ∧ y = m) (λ x y => x = b ∧ y = c) ∧
    (p.x - a.x) * (m.x - a.x) + (p.y - a.y) * (m.y - a.y) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l302_30222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_k_range_l302_30257

-- Define the circle and points
def circleEq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

def C : ℝ × ℝ := (-1, 0)

def A : ℝ × ℝ := (1, 0)

noncomputable def P (t : ℝ) : ℝ × ℝ := (Real.sqrt 8 * Real.cos t - 1, Real.sqrt 8 * Real.sin t)

noncomputable def Q (t : ℝ) : ℝ × ℝ := sorry

noncomputable def M (t : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
def cond1 (t : ℝ) : Prop := circleEq (P t).1 (P t).2

def cond2 (t : ℝ) : Prop := ∃ r : ℝ, Q t = C + r • (P t - C)

def cond3 (t : ℝ) : Prop := (M t - Q t) • (A - P t) = 0

def cond4 (t : ℝ) : Prop := A - P t = 2 • (A - M t)

-- Define the trajectory of Q
def trajectory_Q (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the tangent line and intersection points
def tangent_line (k b x y : ℝ) : Prop := y = k * x + b ∧ b^2 = k^2 + 1

noncomputable def F (k b : ℝ) : ℝ × ℝ := sorry

noncomputable def H (k b : ℝ) : ℝ × ℝ := sorry

def intersection_condition (k b : ℝ) : Prop :=
  trajectory_Q (F k b).1 (F k b).2 ∧
  trajectory_Q (H k b).1 (H k b).2 ∧
  tangent_line k b (F k b).1 (F k b).2 ∧
  tangent_line k b (H k b).1 (H k b).2 ∧
  F k b ≠ H k b

def dot_product_condition (k b : ℝ) : Prop :=
  3/4 ≤ ((F k b).1 * (H k b).1 + (F k b).2 * (H k b).2) ∧
  ((F k b).1 * (H k b).1 + (F k b).2 * (H k b).2) ≤ 4/5

-- Main theorem
theorem trajectory_and_k_range :
  (∀ t : ℝ, cond1 t → cond2 t → cond3 t → cond4 t →
    trajectory_Q (Q t).1 (Q t).2) ∧
  (∀ k : ℝ, (∃ b : ℝ, intersection_condition k b ∧ dot_product_condition k b) ↔
    (k ∈ Set.Icc (-Real.sqrt 2 / 2) (-Real.sqrt 3 / 3) ∨
     k ∈ Set.Icc (Real.sqrt 3 / 3) (Real.sqrt 2 / 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_k_range_l302_30257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l302_30284

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (2*x - 1) + 1 / (x^2)

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x ↦ -2 * Real.sin (2*x - 1) - 2 / (x^3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l302_30284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_ratio_bound_l302_30240

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A collection of n points in a plane --/
def PointSet (n : ℕ) := Fin n → Point

/-- Predicate to check if three points are collinear --/
def areCollinear (p q r : Point) : Prop := sorry

/-- Predicate to check if a triangle is acute-angled --/
def isAcuteTriangle (p q r : Point) : Prop := sorry

/-- The total number of triangles formed by n points --/
def totalTriangles (n : ℕ) : ℕ := Nat.choose n 3

/-- The number of acute-angled triangles formed by n points --/
noncomputable def acuteTriangles (points : PointSet n) : ℕ := sorry

theorem acute_triangle_ratio_bound 
  (n : ℕ) 
  (h_n : n > 3) 
  (points : PointSet n)
  (h_not_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ¬areCollinear (points i) (points j) (points k)) :
  (acuteTriangles points : ℚ) / (totalTriangles n : ℚ) ≤ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_ratio_bound_l302_30240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_fifth_l302_30233

/-- Three unit squares arranged adjacently --/
structure UnitSquares :=
  (square1 : Set (ℝ × ℝ))
  (square2 : Set (ℝ × ℝ))
  (square3 : Set (ℝ × ℝ))

/-- Triangle formed by connecting vertices of the unit squares --/
structure Triangle (squares : UnitSquares) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle --/
def triangleArea {squares : UnitSquares} (t : Triangle squares) : ℝ := sorry

/-- Theorem stating that the area of the specific triangle is 1/5 --/
theorem triangle_area_is_one_fifth (squares : UnitSquares) (t : Triangle squares) :
  triangleArea t = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_fifth_l302_30233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_l302_30285

noncomputable def original_function (x : ℝ) : ℝ := Real.log (1 / x)

def reflected_function (f : ℝ → ℝ) : ℝ → ℝ := fun x => -f x

def translated_function (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x => f (x - h)

noncomputable def C1 : ℝ → ℝ := reflected_function original_function

noncomputable def C2 : ℝ → ℝ := translated_function C1 1

theorem C2_equation : C2 = fun x => Real.log (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_l302_30285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_space_l302_30239

-- Define the cube edge length in meters
noncomputable def cubeEdgeLength : ℝ := 3000

-- Define the Earth's population
noncomputable def earthPopulation : ℝ := 7000000000

-- Calculate the volume of the cube
noncomputable def cubeVolume : ℝ := cubeEdgeLength ^ 3

-- Calculate the volume per person
noncomputable def volumePerPerson : ℝ := cubeVolume / earthPopulation

-- Define a minimum required volume per person (including space for buildings and structures)
noncomputable def minRequiredVolumePerPerson : ℝ := 10 -- This is an assumption, adjust as needed

-- Theorem stating that the available volume per person is insufficient
theorem insufficient_space : volumePerPerson < minRequiredVolumePerPerson := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_space_l302_30239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_replacement_l302_30261

def original_number : ℚ := -142800 / 1000000

def replace_digit (n : ℚ) (old_digit new_digit : ℕ) (position : ℕ) : ℚ :=
  sorry

def is_largest_replacement (n : ℚ) (replaced_digit : ℕ) (new_digit : ℕ) : Prop :=
  ∀ (d : ℕ), d ≠ 0 → d ≠ replaced_digit →
    replace_digit n replaced_digit new_digit 2 ≥ replace_digit n d new_digit (Nat.digits 10 d).length

theorem largest_replacement :
  is_largest_replacement original_number 4 3 :=
sorry

#eval original_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_replacement_l302_30261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_l302_30265

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (2, 5)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem equidistant_point_x_coordinate :
  ∃ x : ℝ, distance A (x, 0) = distance B (x, 0) ∧ x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_l302_30265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_function_properties_second_function_properties_third_function_properties_l302_30213

-- Define a fractional-linear function
noncomputable def fractionalLinear (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- Define the inverse of a fractional-linear function
noncomputable def inverseFractionalLinear (a b c d : ℝ) (x : ℝ) : ℝ := (-d * x + b) / (c * x - a)

-- Define a fixed point
def isFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Define a cycle
def isCycle (f : ℝ → ℝ) (xs : List ℝ) : Prop :=
  xs.length > 1 ∧ (∀ i, f (xs.get! i) = xs.get! ((i + 1) % xs.length))

-- Theorem for the first function
theorem first_function_properties :
  ∃ (fp1 fp2 : ℝ) (c1 c2 c3 : List ℝ),
    isFixedPoint (fractionalLinear 4 1 2 3) fp1 ∧
    isFixedPoint (fractionalLinear 4 1 2 3) fp2 ∧
    fp1 ≠ fp2 ∧
    isCycle (fractionalLinear 4 1 2 3) c1 ∧
    isCycle (fractionalLinear 4 1 2 3) c2 ∧
    isCycle (fractionalLinear 4 1 2 3) c3 ∧
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 := by sorry

-- Theorem for the second function
theorem second_function_properties :
  ∃ (c1 c2 : List ℝ),
    isCycle (fractionalLinear 2 1 3 2) c1 ∧
    isCycle (fractionalLinear 2 1 3 2) c2 ∧
    c1 ≠ c2 ∧
    (∀ x : ℝ, ¬isFixedPoint (fractionalLinear 2 1 3 2) x) := by sorry

-- Theorem for the third function
theorem third_function_properties :
  ∃ (fp : ℝ) (c : List ℝ),
    isFixedPoint (fractionalLinear 3 (-1) 1 1) fp ∧
    isCycle (fractionalLinear 3 (-1) 1 1) c ∧
    (∀ x : ℝ, isFixedPoint (fractionalLinear 3 (-1) 1 1) x → x = fp) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_function_properties_second_function_properties_third_function_properties_l302_30213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_at_six_l302_30255

-- Define the divisor function d(n)
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

-- Define the function f(n)
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val^(1/4 : ℝ)

-- Theorem statement
theorem max_f_at_six : ∃ (N : ℕ+), ∀ (n : ℕ+), n ≠ N → f n < f N ∧ N = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_at_six_l302_30255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_pentagon_area_l302_30268

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure TangentialTrapezoid where
  top_base : ℝ
  side_length : ℝ
  bottom_base : ℝ
  height : ℝ
  is_isosceles : side_length > 0
  has_inscribed_circle : top_base + bottom_base + 2 * side_length = 4 * side_length
  has_circumscribed_circle : top_base * bottom_base = 4 * height^2

/-- The area of the pentagon formed in a tangential trapezoid -/
noncomputable def pentagon_area (t : TangentialTrapezoid) : ℝ :=
  (7 / 2) * Real.sqrt 35

/-- Theorem: The area of the pentagon in a specific tangential trapezoid is (7/2) * √35 -/
theorem specific_trapezoid_pentagon_area :
  ∃ (t : TangentialTrapezoid),
    t.top_base = 5 ∧
    t.side_length = 6 ∧
    pentagon_area t = (7 / 2) * Real.sqrt 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_pentagon_area_l302_30268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_circle_tangent_to_directrix_l302_30248

-- Define the parabola C
structure Parabola where
  vertex : ℝ × ℝ
  focusX : ℝ
  passingPoint : ℝ × ℝ

-- Define the given parabola
noncomputable def C : Parabola where
  vertex := (0, 0)
  focusX := 1/2
  passingPoint := (2, 2)

-- Define the equation of the parabola
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = 2 * (x - p.vertex.1)

-- Define the directrix of the parabola
def directrix (p : Parabola) : ℝ → Prop :=
  fun x => x = -p.focusX

-- Define the circle with diameter AB
def circle_AB (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => (x - 2)^2 + y^2 = 1

-- Theorem 1: The standard equation of C is y^2 = 2x
theorem parabola_equation_correct :
  ∀ x y, parabola_equation C x y ↔ y^2 = 2*x := by sorry

-- Theorem 2: The circle with diameter AB is tangent to the directrix
theorem circle_tangent_to_directrix :
  ∃ x, directrix C x ∧ circle_AB C x 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_circle_tangent_to_directrix_l302_30248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_poly_gen_correct_lucas_poly_gen_correct_l302_30287

/-- Fibonacci polynomial sequence -/
noncomputable def FibPoly (x : ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => x * FibPoly x (n + 1) + FibPoly x n

/-- Lucas polynomial sequence -/
noncomputable def LucasPoly (x : ℝ) : ℕ → ℝ
  | n => FibPoly x (n - 1) + FibPoly x (n + 1)

/-- Generating function for Fibonacci polynomials -/
noncomputable def FibPolyGen (x z : ℝ) : ℝ := z / (1 - x * z - z^2)

/-- Generating function for Lucas polynomials -/
noncomputable def LucasPolyGen (x z : ℝ) : ℝ := (2 - x * z) / (1 - x * z - z^2)

/-- Theorem stating that FibPolyGen is the generating function for FibPoly -/
theorem fib_poly_gen_correct (x : ℝ) :
  ∀ z, abs z < (1 - abs x) / 2 → FibPolyGen x z = ∑' n, FibPoly x n * z^n := by sorry

/-- Theorem stating that LucasPolyGen is the generating function for LucasPoly -/
theorem lucas_poly_gen_correct (x : ℝ) :
  ∀ z, abs z < (1 - abs x) / 2 → LucasPolyGen x z = ∑' n, LucasPoly x n * z^n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_poly_gen_correct_lucas_poly_gen_correct_l302_30287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_heads_in_eight_tosses_l302_30211

theorem probability_three_heads_in_eight_tosses : 
  let n : ℕ := 8  -- total number of tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible sequences
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of sequences with exactly k heads
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_heads_in_eight_tosses_l302_30211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_equilateral_triangle_l302_30228

/-- The circle defined by x^2 + (y - 1)^2 = 1 -/
def myCircle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- The ellipse defined by 9x^2 + (y + 1)^2 = 9 -/
def myEllipse (x y : ℝ) : Prop := 9 * x^2 + (y + 1)^2 = 9

/-- A point (x, y) is an intersection point if it satisfies both circle and ellipse equations -/
def intersection_point (x y : ℝ) : Prop := myCircle x y ∧ myEllipse x y

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Three points form an equilateral triangle if the distances between all pairs of points are equal -/
def is_equilateral_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  distance x1 y1 x2 y2 = distance x2 y2 x3 y3 ∧
  distance x2 y2 x3 y3 = distance x3 y3 x1 y1

theorem intersection_forms_equilateral_triangle :
  ∃ x1 y1 x2 y2 x3 y3 : ℝ,
    intersection_point x1 y1 ∧
    intersection_point x2 y2 ∧
    intersection_point x3 y3 ∧
    is_equilateral_triangle x1 y1 x2 y2 x3 y3 :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_equilateral_triangle_l302_30228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equivalence_l302_30227

-- Define the parametric equations
def equation1 (t : ℝ) : ℝ × ℝ := (t, t^2)

noncomputable def equation2 (t : ℝ) : ℝ × ℝ := (Real.tan t, (Real.tan t)^2)

noncomputable def equation3 (t : ℝ) : ℝ × ℝ := (Real.sin t, (Real.sin t)^2)

-- Define the standard form of the curve
def standard_curve (x : ℝ) : ℝ := x^2

-- Theorem stating that equations 1 and 2 represent the same curve, but 3 is different
theorem curve_equivalence :
  (∀ x : ℝ, ∃ t : ℝ, equation1 t = (x, standard_curve x)) ∧
  (∀ x : ℝ, ∃ t : ℝ, equation2 t = (x, standard_curve x)) ∧
  ¬(∀ x : ℝ, ∃ t : ℝ, equation3 t = (x, standard_curve x)) := by
  sorry

#check curve_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equivalence_l302_30227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_neg_one_no_a_for_strictly_increasing_f_l302_30206

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

-- Part 1: Range of f when a = -1
theorem range_of_f_when_a_neg_one :
  Set.range (f (-1)) = Set.Iic (-1) :=
sorry

-- Part 2: Non-existence of a for strictly increasing f
theorem no_a_for_strictly_increasing_f :
  ¬∃ (a : ℝ), StrictMono (fun x => f a x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_neg_one_no_a_for_strictly_increasing_f_l302_30206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_tiling_l302_30201

/-- Represents the dimensions of a rectangular object. -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular object given its dimensions. -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Calculates the number of whole tiles that fit along one side. -/
def tilesAlongSide (roomSide : ℚ) (tileSide : ℚ) : ℕ :=
  (roomSide / tileSide).floor.toNat

/-- The main theorem about tiling the room. -/
theorem room_tiling (room : Dimensions) (tile : Dimensions) : 
  room.length = 10 ∧ room.width = 15 ∧ tile.length = 1/3 ∧ tile.width = 1/2 →
  (tilesAlongSide room.length tile.length * tilesAlongSide room.width tile.width = 900) ∧
  (area room = 900 * area tile) := by
  sorry

#check room_tiling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_tiling_l302_30201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_specific_parallel_lines_l302_30273

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ  -- coefficient of x
  b : ℝ  -- coefficient of y
  c1 : ℝ  -- constant term for line 1
  c2 : ℝ  -- constant term for line 2
  parallel : a ≠ 0 ∨ b ≠ 0  -- ensure lines are not degenerate

/-- Distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l : ParallelLines) : ℝ :=
  abs (l.c1 - l.c2) / Real.sqrt (l.a^2 + l.b^2)

/-- The given parallel lines and their distance -/
theorem distance_of_specific_parallel_lines :
  let l : ParallelLines := {
    a := 2,
    b := 1,
    c1 := -1,
    c2 := 1,
    parallel := Or.inl (by norm_num)
  }
  distance_between_parallel_lines l = 2 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_specific_parallel_lines_l302_30273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l302_30280

theorem sin_double_angle (α : ℝ) :
  Real.sin (α - π/4) = 3/5 → Real.sin (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l302_30280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_B_to_C_value_l302_30254

/-- The interest rate at which B lent money to C -/
noncomputable def interest_rate_B_to_C (principal : ℝ) (rate_A_to_B : ℝ) (years : ℝ) (gain_B : ℝ) : ℝ :=
  let interest_A_from_B := principal * rate_A_to_B * years
  let total_interest_B_from_C := interest_A_from_B + gain_B
  (total_interest_B_from_C * 100) / (principal * years)

/-- Theorem stating the interest rate at which B lent money to C -/
theorem interest_rate_B_to_C_value :
  interest_rate_B_to_C 3200 0.12 5 400 = 14.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval interest_rate_B_to_C 3200 0.12 5 400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_B_to_C_value_l302_30254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_factorial_sum_l302_30274

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumOfFactorials : ℕ := 
  (List.range 40).map (fun i => factorial ((i + 1) * 3)) |> List.sum

theorem last_two_digits_of_factorial_sum :
  lastTwoDigits sumOfFactorials = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_factorial_sum_l302_30274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_dot_product_extrema_l302_30266

noncomputable def a (x : ℝ) : Fin 2 → ℝ := ![Real.cos x, Real.sin x]
noncomputable def b : Fin 2 → ℝ := ![3, -Real.sqrt 3]

theorem vector_parallel_and_dot_product_extrema (x : ℝ) 
  (h : x ∈ Set.Icc 0 Real.pi) :
  (∃ k : ℝ, a x = k • b → x = 5 * Real.pi / 6) ∧
  (∀ y ∈ Set.Icc 0 Real.pi, Finset.sum (Finset.range 2) (λ i => (a y i) * (b i)) ≤ 3) ∧
  (∀ y ∈ Set.Icc 0 Real.pi, Finset.sum (Finset.range 2) (λ i => (a y i) * (b i)) ≥ -2 * Real.sqrt 3) ∧
  (Finset.sum (Finset.range 2) (λ i => (a 0 i) * (b i)) = 3) ∧
  (Finset.sum (Finset.range 2) (λ i => (a (5 * Real.pi / 6) i) * (b i)) = -2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_dot_product_extrema_l302_30266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l302_30249

theorem system_solution (x y : ℝ) 
  (eq1 : Real.cos x = 2 * (Real.cos y)^3)
  (eq2 : Real.sin x = 2 * (Real.sin y)^3) :
  ∃ (l k : ℤ), x = 2 * l * Real.pi + k * Real.pi / 2 + Real.pi / 4 ∧ 
               y = k * Real.pi / 2 + Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l302_30249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l302_30217

/-- A cube ABCDEFGH with edge length 2 --/
structure Cube where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ
  E : ℝ × ℝ × ℝ
  F : ℝ × ℝ × ℝ
  G : ℝ × ℝ × ℝ
  H : ℝ × ℝ × ℝ
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- The volume of a pyramid --/
noncomputable def pyramid_volume (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

/-- Theorem: The volume of pyramid ABFG in the cube is 4/3 --/
theorem pyramid_volume_in_cube (c : Cube) : 
  ∃ (base_area height : ℝ), pyramid_volume base_area height = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l302_30217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_sum_a_c_l302_30207

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.b * Real.cos t.A = (2 * t.c + t.a) * Real.cos (Real.pi - t.B)

-- Theorem 1: Measure of angle B
theorem angle_B_measure (t : Triangle) (h : satisfiesConditions t) : t.B = 2 * Real.pi / 3 :=
sorry

-- Theorem 2: Sum of a and c
theorem sum_a_c (t : Triangle) (h1 : satisfiesConditions t) (h2 : t.b = 4) 
  (h3 : 1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3) : t.a + t.c = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_sum_a_c_l302_30207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l302_30251

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the main property of g
axiom g_property : ∀ (x y : ℝ), g ((x - y)^2) = g x^2 - 3*x*g y + y^2

-- Theorem stating the properties we want to prove
theorem g_properties :
  (∃ (s : Set ℝ), s = {1, 2} ∧ ∀ (z : ℝ), g 1 = z → z ∈ s) ∧
  (Finset.sum {1, 2} id = 3) ∧
  (Finset.card {1, 2} * Finset.sum {1, 2} id = 6) := by
  sorry

#check g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l302_30251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l302_30219

-- Define the curve C'
noncomputable def curve_C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function to minimize
noncomputable def f (x y : ℝ) : ℝ := x^2 - Real.sqrt 3 * x * y + 2 * y^2

-- Theorem statement
theorem min_value_on_curve (x y : ℝ) :
  curve_C' x y → f x y ≥ 1 ∧ 
  (f x y = 1 ↔ (x = 1 ∧ y = Real.sqrt 3 / 2) ∨ (x = -1 ∧ y = -Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l302_30219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l302_30220

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

-- Define the horizontal asymptote
def horizontal_asymptote : ℝ := 3

-- Theorem statement
theorem g_crosses_asymptote :
  ∃ x : ℝ, g x = horizontal_asymptote ∧ x = 13/4 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l302_30220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_quantity_after_two_replacements_l302_30279

/-- Represents the state of the vessel -/
structure VesselState where
  milk : ℝ  -- Amount of milk in litres
  water : ℝ  -- Amount of water in litres

/-- Calculates the new state after removing and replacing liquid -/
noncomputable def removeAndReplace (state : VesselState) (amount : ℝ) : VesselState :=
  let totalLiquid := state.milk + state.water
  let milkRatio := state.milk / totalLiquid
  let removedMilk := milkRatio * amount
  { milk := state.milk - removedMilk,
    water := totalLiquid - (state.milk - removedMilk) }

theorem milk_quantity_after_two_replacements 
  (initialMilk : ℝ) 
  (vesselCapacity : ℝ) 
  (replacementAmount : ℝ) :
  initialMilk = vesselCapacity →
  vesselCapacity = 30 →
  replacementAmount = 9 →
  let initialState : VesselState := { milk := initialMilk, water := 0 }
  let afterFirstReplacement := removeAndReplace initialState replacementAmount
  let finalState := removeAndReplace afterFirstReplacement replacementAmount
  finalState.milk = 14.7 := by
  sorry

#check milk_quantity_after_two_replacements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_quantity_after_two_replacements_l302_30279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_group_size_l302_30292

/-- Calculates the total number of people in a movie theater group given ticket prices and total amount paid. -/
def total_people (adult_price child_price total_paid : ℚ) (num_adults : ℕ) : ℕ :=
  let num_children := ((total_paid - (adult_price * num_adults)) / child_price).floor
  num_adults + num_children.toNat

/-- Theorem stating that under the given conditions, the total number of people in the group is 7. -/
theorem movie_group_size :
  total_people (9.5) (6.5) (54.5) 3 = 7 := by
  sorry

#eval total_people (9.5) (6.5) (54.5) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_group_size_l302_30292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_diagonal_divisibility_l302_30202

/-- Represents a vertex of the heptagon -/
structure Vertex where
  value : ℤ

/-- Represents the heptagon -/
structure Heptagon where
  vertices : Fin 7 → Vertex
  side_divisibility : ∀ i : Fin 7, (vertices i).value ∣ (vertices (i + 1)).value ∨ (vertices (i + 1)).value ∣ (vertices i).value

/-- Represents a diagonal in the heptagon -/
structure Diagonal where
  start : Fin 7
  endpoint : Fin 7
  is_diagonal : start ≠ endpoint ∧ start ≠ (endpoint + 1) ∧ endpoint ≠ (start + 1)

/-- The main theorem -/
theorem heptagon_diagonal_divisibility (h : Heptagon) (diagonals : List Diagonal) :
  ¬(∀ d : Diagonal, d ∈ diagonals →
    ¬((h.vertices d.start).value ∣ (h.vertices d.endpoint).value) ∧
    ¬((h.vertices d.endpoint).value ∣ (h.vertices d.start).value)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_diagonal_divisibility_l302_30202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l302_30242

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = |x| + 1}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l302_30242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l302_30229

-- Problem 1
theorem problem_1 : Real.sqrt 9 * 3⁻¹ + ((-8) ^ (1/3 : ℝ)) + (-Real.sqrt 2)^0 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) : 
  (1 - a⁻¹) / ((a^2 - 1) / (2 * a)) = 2 / (a + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l302_30229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l302_30243

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : 0 < β) (h2 : β < α) (h3 : α < π/4)
  (h4 : Real.cos (α - β) = 12/13) (h5 : Real.sin (α + β) = 4/5) :
  Real.sin (2*α) = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l302_30243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_vertex_counts_l302_30214

/-- The number of vertices of a regular polygon -/
def vertices (n : ℕ) : ℕ := n

/-- The set of possible numbers of distinct vertices when drawing
    regular 15-gon, 21-gon, and 35-gon on a circle -/
def possible_vertex_counts : Finset ℕ :=
  { 15 + 21 + 35,
    15 + 35 - Nat.gcd 15 35 + 21,
    15 + 21 - Nat.gcd 15 21 + 35,
    21 + 35 - Nat.gcd 21 35 + 15,
    15 + 21 + 35 - Nat.gcd 21 35 - Nat.gcd 15 35 - Nat.gcd 15 21 + 2,
    21 + 35 - Nat.gcd 21 35 - Nat.gcd 15 35 + 15,
    15 + 21 - Nat.gcd 15 21 - Nat.gcd 15 35 + 35,
    15 + 21 - Nat.gcd 15 21 - Nat.gcd 21 35 + 35 }

theorem sum_of_possible_vertex_counts :
  (possible_vertex_counts.sum id) = 510 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_vertex_counts_l302_30214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_tan_alpha_complex_trig_expression_l302_30269

-- Define the angle α based on the given point
noncomputable def α : Real := Real.arctan (-4 / -3)

-- Theorem 1
theorem sin_over_tan_alpha : Real.sin α / Real.tan α = -3/5 := by sorry

-- Theorem 2
theorem complex_trig_expression :
  (Real.sin (α + π/2) * Real.cos (9*π/2 - α) * Real.tan (2*π - α) * Real.cos (-3*π/2 + α)) /
  (Real.sin (2*π - α) * Real.tan (-α - π) * Real.sin (π + α)) = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_tan_alpha_complex_trig_expression_l302_30269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_a_range_l302_30210

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (3 - a) * x + 2 else a^(2 * x^2 - 9 * x + 11)

noncomputable def sequence_a (a : ℝ) (n : ℕ) : ℝ := f a n

theorem increasing_sequence_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ n : ℕ, sequence_a a n < sequence_a a (n + 1)) → 2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_a_range_l302_30210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BE_l302_30298

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 12 ∧ d B C = 12 ∧ d C A = 12

-- Define point D as the foot of the perpendicular from A to BC
def FootOfPerpendicular (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - B.1) * (C.1 - B.1) + (D.2 - B.2) * (C.2 - B.2) =
    ((C.1 - B.1)^2 + (C.2 - B.2)^2) * ((D.1 - B.1) * (A.1 - B.1) + (D.2 - B.2) * (A.2 - B.2)) /
      ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2))

-- Define point E as the midpoint of AD
def Midpoint (A D E : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + D.1) / 2 ∧ E.2 = (A.2 + D.2) / 2

-- Theorem statement
theorem length_BE (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  FootOfPerpendicular A B C D →
  Midpoint A D E →
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d B E = Real.sqrt 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BE_l302_30298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pasture_area_l302_30260

/-- A rectangular pasture with one side along a barn --/
structure Pasture where
  barn_length : ℝ
  fence_length : ℝ

/-- The area of the pasture as a function of the perpendicular side length --/
noncomputable def pasture_area (p : Pasture) (x : ℝ) : ℝ := x * (p.fence_length - 2 * x)

/-- The optimal perpendicular side length that maximizes the area --/
noncomputable def optimal_side_length (p : Pasture) : ℝ := p.fence_length / 4

/-- The length of the side parallel to the barn that maximizes the area --/
noncomputable def optimal_parallel_length (p : Pasture) : ℝ := p.fence_length / 2

theorem optimal_pasture_area (p : Pasture) (h1 : p.barn_length = 400) (h2 : p.fence_length = 240) :
  optimal_parallel_length p = 120 := by
  -- Unfold the definition of optimal_parallel_length
  unfold optimal_parallel_length
  -- Use the hypothesis h2 to rewrite the fence_length
  rw [h2]
  -- Simplify the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pasture_area_l302_30260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_half_l302_30281

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_2019_equals_half :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 4) = f x) →  -- f has period 4
  (∃ a b, ∀ x, -2 ≤ x ∧ x < 0 → f x = a * x + b) →  -- f(x) = ax + b for -2 ≤ x < 0
  (∃ a, ∀ x, 0 < x ∧ x ≤ 2 → f x = a * x - 1) →  -- f(x) = ax - 1 for 0 < x ≤ 2
  f 2019 = 1/2 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_half_l302_30281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_integers_count_l302_30230

def T : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 100) (Finset.range 101)

def multiples_of_2 (S : Finset ℕ) : Finset ℕ := S.filter (fun n => n % 2 = 0)
def multiples_of_5 (S : Finset ℕ) : Finset ℕ := S.filter (fun n => n % 5 = 0)

def remaining_set (S : Finset ℕ) : Finset ℕ :=
  S \ (multiples_of_2 S ∪ multiples_of_5 S)

theorem remaining_integers_count :
  Finset.card (remaining_set T) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_integers_count_l302_30230
