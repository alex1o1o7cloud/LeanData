import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_theorem_l1335_133596

/-- A segment on a line --/
structure Segment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- The property that any subset of 1998 segments contains two intersecting segments --/
def IntersectionProperty (S : Set Segment) : Prop :=
  ∀ T : Set Segment, T ⊆ S → T.Finite → T.ncard = 1998 →
    ∃ s1 s2, s1 ∈ T ∧ s2 ∈ T ∧ s1 ≠ s2 ∧ (s1.left ≤ s2.right ∧ s2.left ≤ s1.right)

/-- Main theorem --/
theorem segment_intersection_theorem (S : Set Segment) (h : S.Finite) (h_prop : IntersectionProperty S) :
  ∃ P : Set ℝ, P.Finite ∧ P.ncard ≤ 1997 ∧ ∀ s ∈ S, ∃ p ∈ P, s.left ≤ p ∧ p ≤ s.right := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_theorem_l1335_133596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1335_133542

-- Define the parabola and its properties
noncomputable def Parabola (p : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 = 2*p*y ∧ p > 0}

-- Define the distance between a point on the parabola and its focus
noncomputable def DistanceToFocus (p : ℝ) (x y : ℝ) : ℝ := 1 + p/2

-- Define the distance between the focus and the directrix
noncomputable def FocusDirectrixDistance (p : ℝ) : ℝ := p

-- Theorem statement
theorem parabola_focus_directrix_distance 
  (p : ℝ) (x : ℝ) 
  (h1 : (x, 1) ∈ Parabola p) 
  (h2 : DistanceToFocus p x 1 = 3) : 
  FocusDirectrixDistance p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1335_133542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_removed_volume_l1335_133575

/-- The length of the side of the cube -/
noncomputable def cube_side : ℝ := 2

/-- The volume of a single removed tetrahedron -/
noncomputable def tetrahedron_volume : ℝ := (6 - 4 * Real.sqrt 2) / 3

/-- The number of corners in a cube -/
def num_corners : ℕ := 8

/-- Theorem stating the total volume of removed tetrahedra -/
theorem total_removed_volume :
  (num_corners : ℝ) * tetrahedron_volume = (48 - 32 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_removed_volume_l1335_133575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_quadrants_l1335_133559

/-- Represents a quadrant in the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Determines if a line passes through a given quadrant -/
def passes_through (l : Line) (q : Quadrant) : Prop := sorry

/-- The original line y = kx + b -/
def original_line (k b : ℝ) : Line := { m := k, c := b }

/-- The transformed line y = -bx + k -/
def transformed_line (k b : ℝ) : Line := { m := -b, c := k }

theorem line_quadrants (k b : ℝ) :
  (passes_through (original_line k b) Quadrant.I) ∧
  (passes_through (original_line k b) Quadrant.III) ∧
  (passes_through (original_line k b) Quadrant.IV) →
  (passes_through (transformed_line k b) Quadrant.I) ∧
  (passes_through (transformed_line k b) Quadrant.II) ∧
  (passes_through (transformed_line k b) Quadrant.IV) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_quadrants_l1335_133559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_eq_one_l1335_133508

-- Define the constants and functions
noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def β : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

-- Theorem statement
theorem x_times_one_minus_f_eq_one : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_eq_one_l1335_133508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l1335_133585

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Line structure -/
structure Line where
  k : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and a line, find the equation of the parabola and the length of the chord -/
def parabola_chord_problem (E : Parabola) (l : Line) (A B : Point) : Prop :=
  let p := E.p
  let k := l.k
  -- Parabola equation
  (∀ x y, y = x^2 / (2 * p)) ∧
  -- Line equation
  (∀ x, l.k * x + 2 = A.y ∧ l.k * x + 2 = B.y) ∧
  -- A and B are on the parabola
  (A.y = A.x^2 / (2 * p) ∧ B.y = B.x^2 / (2 * p)) ∧
  -- Dot product condition
  (A.x * B.x + A.y * B.y = 2) →
  -- Conclusion 1: Equation of parabola E
  (p = 1/2) ∧
  -- Conclusion 2: Length of chord AB when k = 1
  (l.k = 1 → Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 3 * Real.sqrt 2)

theorem parabola_chord_theorem (E : Parabola) (l : Line) (A B : Point) :
  parabola_chord_problem E l A B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l1335_133585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_quadratic_plus_constant_l1335_133546

open Set
open MeasureTheory
open Real

theorem integral_of_quadratic_plus_constant (f : ℝ → ℝ) :
  (∀ x, f x = x^2 + 2 * ∫ y in (0:ℝ)..(1:ℝ), f y) →
  ∫ x in (0:ℝ)..(1:ℝ), f x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_quadratic_plus_constant_l1335_133546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_unique_l1335_133547

/-- A monic quartic polynomial with complex coefficients -/
def MonicQuarticPolynomial (a b c d : ℂ) : ℂ → ℂ :=
  fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_unique 
  (q : ℂ → ℂ) 
  (h_monic : q = fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d) 
  (h_real_coeff : a.im = 0 ∧ b.im = 0 ∧ c.im = 0 ∧ d.im = 0)
  (h_root : q (2 - I) = 0)
  (h_value : q 0 = 32) :
  q = fun x ↦ x^4 - 5.6*x^3 + 22.4*x^2 - 28*x + 25.6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_unique_l1335_133547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1335_133528

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Calculate the distance between two points -/
noncomputable def Point.distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Find the intersection point of two lines -/
noncomputable def Line.intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
    y := l1.slope * (l2.intercept - l1.intercept) / (l1.slope - l2.slope) + l1.intercept }

theorem line_equation_proof (l : Line) (P : Point) (l1 l2 : Line) :
  P.x = 3 ∧ P.y = 1 ∧
  l1.slope = -1 ∧ l1.intercept = -1 ∧
  l2.slope = -1 ∧ l2.intercept = -6 ∧
  P.onLine l ∧
  (Line.intersectionPoint l l1).distance (Line.intersectionPoint l l2) = 5 →
  l.slope = -1 ∧ l.intercept = 4 := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1335_133528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_water_management_l1335_133525

/-- Represents the water outflow function -/
noncomputable def outflow (t : ℝ) : ℝ := 120 * Real.sqrt (6 * t)

/-- Represents the water volume at time t -/
noncomputable def water_volume (t : ℝ) : ℝ := 400 + 60 * t - outflow t

theorem reservoir_water_management :
  let initial_volume : ℝ := 400
  let inflow_rate : ℝ := 60
  let time_range : Set ℝ := Set.Icc 0 24
  -- The reservoir contains 400 tons after 24 hours
  (water_volume 24 = initial_volume) ∧
  -- The minimum water volume occurs at t = 6 hours
  (∀ t ∈ time_range, water_volume 6 ≤ water_volume t) ∧
  -- The minimum water volume is 40 tons
  (water_volume 6 = 40) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_water_management_l1335_133525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1335_133582

-- Define the function f(x) = sin x
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem sin_cos_product (α : ℝ) :
  (∀ x, x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → f x = Real.sin x) →
  f (Real.sin α) + f (Real.cos α - 1/2) = 0 →
  Real.sin α * Real.cos α = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1335_133582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_is_arcsin_third_max_angle_achieved_when_cube_l1335_133548

/-- A rectangular parallelepiped with a square base -/
structure RectParallelepiped where
  a : ℝ  -- length of the square base side
  b : ℝ  -- height of the parallelepiped
  a_pos : 0 < a
  b_pos : 0 < b

/-- The angle between line BD₁ and plane BDC₁ in a rectangular parallelepiped -/
noncomputable def angle (p : RectParallelepiped) : ℝ :=
  Real.arcsin (p.a * p.b / Real.sqrt (2 * p.a^4 + 2 * p.b^4 + 5 * p.a^2 * p.b^2))

/-- The theorem stating the maximum angle -/
theorem max_angle_is_arcsin_third (p : RectParallelepiped) : 
  angle p ≤ Real.arcsin (1/3) := by
  sorry

/-- The theorem stating when the maximum angle is achieved -/
theorem max_angle_achieved_when_cube (p : RectParallelepiped) : 
  angle p = Real.arcsin (1/3) ↔ p.a = p.b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_is_arcsin_third_max_angle_achieved_when_cube_l1335_133548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_equals_radius_at_half_fill_l1335_133567

/-- Represents a horizontal cylindrical tank -/
structure HorizontalCylindricalTank where
  radius : ℝ
  length : ℝ

/-- Calculates the volume of a horizontal cylindrical tank -/
noncomputable def tankVolume (tank : HorizontalCylindricalTank) : ℝ :=
  Real.pi * tank.radius^2 * tank.length

/-- Calculates the depth of water in a horizontal cylindrical tank when filled to a given percentage -/
noncomputable def waterDepth (tank : HorizontalCylindricalTank) (fillPercentage : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a specific tank with 50% fill, the water depth equals the radius -/
theorem water_depth_equals_radius_at_half_fill
  (tank : HorizontalCylindricalTank)
  (h_radius : tank.radius = 5)
  (h_length : tank.length = 10)
  (h_fill : fillPercentage = 0.5) :
  waterDepth tank fillPercentage = tank.radius := by
  sorry

#check water_depth_equals_radius_at_half_fill

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_equals_radius_at_half_fill_l1335_133567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_range_l1335_133540

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * Real.cos x ^ 2 + 1

theorem f_monotone_and_range :
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Ioo (Real.pi / 4) Real.pi, f x > m - 3) ↔ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_range_l1335_133540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_is_40_l1335_133581

/-- Represents the fuel efficiency of a car in different driving conditions. -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℚ
  city_miles_per_tankful : ℚ
  city_mpg_reduction : ℚ

/-- Calculates the city miles per gallon given the car's fuel efficiency data. -/
noncomputable def city_mpg (car : CarFuelEfficiency) : ℚ :=
  let highway_mpg := car.highway_miles_per_tankful / (car.highway_miles_per_tankful / (car.city_miles_per_tankful / (car.highway_miles_per_tankful - car.city_mpg_reduction)))
  highway_mpg - car.city_mpg_reduction

/-- Theorem stating that for the given car specifications, the city miles per gallon is 40. -/
theorem car_city_mpg_is_40 :
  let car := CarFuelEfficiency.mk 462 336 15
  city_mpg car = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_is_40_l1335_133581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosB_special_triangle_l1335_133588

/-- Triangle with special properties -/
structure SpecialTriangle where
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Sides of the triangle opposite to respective angles
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle condition
  angle_condition : A - C = π / 2
  -- Side condition (arithmetic progression)
  side_condition : 2 * b = a + c
  -- Triangle angle sum
  angle_sum : A + B + C = π
  -- Positive side lengths
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- Theorem stating that cos B = 3/4 for the special triangle -/
theorem cosB_special_triangle (t : SpecialTriangle) : Real.cos t.B = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosB_special_triangle_l1335_133588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1335_133572

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 else 2^x - 2

-- Theorem statement
theorem f_properties :
  f (f (-2)) = 14 ∧ (∃! x, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1335_133572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_bound_tangent_line_at_two_l1335_133553

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x - Real.log x - 1

-- State the theorem
theorem tangent_line_and_inequality_bound {a : ℝ} :
  (∀ x > 0, f x ≥ a * x - 2) ↔ a ≤ 1 - 1 / Real.exp 2 :=
by sorry

-- Define the tangent line function
noncomputable def tangent_line (x y : ℝ) : ℝ := x - 2 * y - Real.log 4

theorem tangent_line_at_two :
  ∀ x > 0, f x = tangent_line x (f 2) + (f x - tangent_line x (f 2)) ∧
  |f x - tangent_line x (f 2)| ≤ |((x - 2) ^ 2)| / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_bound_tangent_line_at_two_l1335_133553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1335_133522

noncomputable section

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_ratio (A B C P Q R : ℝ × ℝ) : 
  (P.1 - A.1) / (B.1 - A.1) = 1/3 →
  (R.1 - P.1) / (B.1 - P.1) = 1/3 →
  Q.2 = B.2 + (C.2 - B.2) * ((Q.1 - B.1) / (C.1 - B.1)) →
  (C.2 - P.2) / (C.1 - P.1) = (B.2 - R.2) / (B.1 - R.1) →
  (triangle_area A B C) / (triangle_area P Q C) = 9/2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1335_133522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1335_133534

/-- Given an ellipse with equation (x^2 / 56) + (y^2 / 14) = 8, 
    the distance between its foci is 8√21 -/
theorem ellipse_foci_distance : 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x^2 / 56 + y^2 / 14 = 8 →
      (f₁.1 = -f₂.1 ∧ f₁.2 = 0 ∧ f₂.2 = 0)) ∧
    dist f₁ f₂ = 8 * Real.sqrt 21 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1335_133534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_BQW_l1335_133502

/-- Rectangle ABCD with given properties --/
structure Rectangle where
  AB : ℝ
  AZ : ℝ
  WC : ℝ
  area_ZWCD : ℝ
  AB_eq : AB = 20
  AZ_eq : AZ = 8
  WC_eq : WC = 8
  area_ZWCD_eq : area_ZWCD = 192

/-- Point Q divides ZW in 2:1 ratio --/
noncomputable def point_Q (r : Rectangle) : ℝ := 2 / 3 * (r.AB - 2 * r.AZ)

/-- Theorem: Area of triangle BQW is approximately 1.78 square units --/
theorem area_triangle_BQW (r : Rectangle) : 
  abs (1 / 2 * (point_Q r / 3) * (r.area_ZWCD / (1 / 2 * (r.AB + r.WC)) - r.WC) - 1.78) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_BQW_l1335_133502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_arithmetic_sequence_l1335_133536

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem first_term_of_arithmetic_sequence 
  (a d : ℝ) 
  (h1 : sum_arithmetic_sequence a d 50 = 200)
  (h2 : sum_arithmetic_sequence (arithmetic_sequence a d 51) d 50 = 2700) :
  a = -20.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_arithmetic_sequence_l1335_133536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_consecutive_square_digits_l1335_133599

/-- A function that checks if a two-digit number is a perfect square -/
def is_two_digit_perfect_square (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ ∃ k : ℕ, k * k = n

/-- A function that checks if each pair of consecutive digits in a number forms a perfect square -/
def consecutive_digits_are_squares (n : ℕ) : Prop :=
  ∀ i : ℕ, i < (n.digits 10).length - 1 →
    is_two_digit_perfect_square ((n.digits 10).get ⟨i, by sorry⟩ * 10 + (n.digits 10).get ⟨i+1, by sorry⟩)

/-- The theorem stating that the leftmost three digits of the largest number with the given property are 816 -/
theorem largest_number_with_consecutive_square_digits :
  ∃ N : ℕ, (∀ m : ℕ, consecutive_digits_are_squares m → m ≤ N) ∧
           consecutive_digits_are_squares N ∧
           (N.digits 10).take 3 = [8, 1, 6] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_consecutive_square_digits_l1335_133599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_after_decimal_l1335_133563

theorem first_three_digits_after_decimal (x : ℝ) : x = (10^1001 + 1)^(9/8) →
  ∃ (y : ℝ), 0 < y ∧ y < 1 ∧ x = 125 * 10^(-3 : ℤ) + y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_after_decimal_l1335_133563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_value_l1335_133537

noncomputable def f (x m : ℝ) : ℝ := Real.cos (x - Real.pi/6) + Real.cos (x + Real.pi/6) + Real.sin x + m

theorem max_value_implies_m_value :
  (∃ (max_val : ℝ), max_val = 1 ∧ ∀ (x : ℝ), f x m ≤ max_val) →
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_value_l1335_133537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l1335_133527

/-- For real numbers x and y where x > 1 and y > 1, 
    the sum of 1 / (x^(3^n) - y^(-3^n)) from n = 0 to infinity
    is equal to 1 / (x*y - 1) -/
theorem infinite_sum_equality (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ∑' n, 1 / (x^(3^n : ℕ) - (1/y)^(3^n : ℕ)) = 1 / (x*y - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l1335_133527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1335_133551

noncomputable def line_point (t : ℝ) : ℝ × ℝ × ℝ := (5 - 3*t, -1 + 4*t, 2 - 2*t)

def given_point : ℝ × ℝ × ℝ := (3, -2, 5)

noncomputable def closest_point : ℝ × ℝ × ℝ := (163/29, -45/29, 66/29)

theorem closest_point_on_line :
  let vector_to_closest := (closest_point.1 - given_point.1, 
                            closest_point.2.1 - given_point.2.1, 
                            closest_point.2.2 - given_point.2.2)
  let direction_vector := (-3, 4, -2)
  (∃ t : ℝ, line_point t = closest_point) ∧ 
  (vector_to_closest.1 * direction_vector.1 + 
   vector_to_closest.2.1 * direction_vector.2.1 + 
   vector_to_closest.2.2 * direction_vector.2.2 = 0) ∧
  (∀ p : ℝ × ℝ × ℝ, (∃ t : ℝ, line_point t = p) → 
    (p.1 - given_point.1)^2 + (p.2.1 - given_point.2.1)^2 + (p.2.2 - given_point.2.2)^2 ≥
    (closest_point.1 - given_point.1)^2 + (closest_point.2.1 - given_point.2.1)^2 + (closest_point.2.2 - given_point.2.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1335_133551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l1335_133597

theorem tan_theta_minus_pi_fourth (θ : Real) 
  (h : Real.cos θ - 3 * Real.sin θ = 0) : 
  Real.tan (θ - Real.pi/4) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l1335_133597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_never_terminates_l1335_133557

-- Define a triangle as a triple of real numbers representing its angles in radians
def Triangle := ℝ × ℝ × ℝ

-- Define a non-degenerate triangle
def NonDegenerateTriangle (t : Triangle) : Prop :=
  let (α, β, γ) := t
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = Real.pi

-- Define the next triangle in the sequence
def NextTriangle (t : Triangle) : Triangle :=
  t

-- Define a valid triangle (satisfies triangle inequality)
def ValidTriangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem sequence_never_terminates (t₀ : Triangle) (h : NonDegenerateTriangle t₀) :
  ∀ n : ℕ, ValidTriangle (((fun t => NextTriangle t)^[n]) t₀) := by
  sorry

#check sequence_never_terminates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_never_terminates_l1335_133557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_from_tan_trig_sum_equals_negative_one_l1335_133561

-- Part 1
theorem sin_cos_from_tan (α : Real) (h1 : Real.tan α = -4/3) (h2 : α ∈ Set.Icc (3*π/2) (2*π)) :
  Real.sin α = -4/5 ∧ Real.cos α = 3/5 := by sorry

-- Part 2
theorem trig_sum_equals_negative_one :
  Real.sin (25*π/6) + Real.cos (26*π/3) + Real.tan (-25*π/4) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_from_tan_trig_sum_equals_negative_one_l1335_133561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_equation_l1335_133520

noncomputable def team_a_rate : ℝ := 1 / 12
noncomputable def team_b_rate : ℝ := 1 / 16
def initial_days : ℝ := 5

theorem project_completion_equation (x : ℝ) : 
  (x * team_b_rate) + ((initial_days + x) * team_a_rate) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_equation_l1335_133520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1335_133573

-- Define the complex number
def z (a : ℝ) : ℂ := Complex.ofReal (a + 3) + Complex.I * Complex.ofReal (3*a - 1)

-- Define the condition for the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem a_range : 
  (∀ a : ℝ, in_fourth_quadrant (z a)) ↔ a ∈ Set.Ioo (-3 : ℝ) (1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1335_133573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l1335_133505

-- Define the linear function
def linear_function (m n x : ℝ) : ℝ := (3*m - 2)*x + (1 - n)

-- Theorem for the three parts of the problem
theorem linear_function_properties (m n : ℝ) :
  -- Part 1: Passes through origin
  ((∀ x, linear_function m n x = 0 ↔ x = 0) ↔ (m ≠ 2/3 ∧ n = 1))
  ∧
  -- Part 2: Increasing function
  ((∀ x y, x < y → linear_function m n x < linear_function m n y) ↔ m > 2/3)
  ∧
  -- Part 3: Not passing through third quadrant
  ((∀ x y, x < 0 ∧ y < 0 → linear_function m n x ≠ y) ↔ (m < 2/3 ∧ n ≤ 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l1335_133505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1335_133595

noncomputable section

-- Define the areas of the lawns
def andy_area : ℝ → ℝ := λ x => x
def beth_area : ℝ → ℝ := λ x => x / 3
def carlos_area : ℝ → ℝ := λ x => x / 4

-- Define the mowing rates
def andy_rate : ℝ → ℝ := λ r => r
def beth_rate : ℝ → ℝ := andy_rate
def carlos_rate : ℝ → ℝ := λ r => r / 2

-- Define the mowing times
def andy_time (x r : ℝ) : ℝ := (andy_area x) / (andy_rate r)
def beth_time (x r : ℝ) : ℝ := (beth_area x) / (beth_rate r) + 0.5
def carlos_time (x r : ℝ) : ℝ := (carlos_area x) / (carlos_rate r) + 0.5

-- Theorem statement
theorem beth_finishes_first (x r : ℝ) (hx : x > 0) (hr : r > 0) :
  beth_time x r < min (andy_time x r) (carlos_time x r) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1335_133595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1335_133504

open Real

theorem beta_value (α β : ℝ) (h1 : tan (α + β) = 7) (h2 : tan α = 3/4) (h3 : β ∈ Set.Ioo 0 π) :
  β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1335_133504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_is_exp_minus_one_l1335_133566

-- Define the logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the symmetry condition
def symmetric_to_log2 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ x = log2 (y + 1)

-- Define the symmetry with respect to x - y = 0
def symmetric_wrt_xy (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem symmetric_function_is_exp_minus_one 
  (f : ℝ → ℝ) 
  (h1 : symmetric_to_log2 f) 
  (h2 : symmetric_wrt_xy f (λ x ↦ log2 (x + 1))) :
  ∀ x, f x = 2^x - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_is_exp_minus_one_l1335_133566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smoothie_containers_correct_l1335_133513

/-- Proves that given the ingredients and conversion rates, 
    the total volume of smoothie for 5 servings requires 5 containers of 1.5-cup capacity. -/
def smoothie_containers (strawberries yogurt orange_juice spinach : ℚ)
                        (honey chia_seeds : ℚ)
                        (servings : ℕ)
                        (container_capacity : ℚ) : ℕ :=
  let total_per_serving := strawberries + yogurt + orange_juice + spinach + honey + chia_seeds
  let total_volume := total_per_serving * servings
  (total_volume / container_capacity).ceil.toNat

theorem smoothie_containers_correct :
  smoothie_containers 0.2 0.1 0.2 0.5 0.125 0.125 5 1.5 = 5 := by
  -- Unfold the definition and simplify
  unfold smoothie_containers
  simp
  -- The rest of the proof
  sorry

#eval smoothie_containers 0.2 0.1 0.2 0.5 0.125 0.125 5 1.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smoothie_containers_correct_l1335_133513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_bus_visible_length_l1335_133558

/-- The length of the red bus in feet -/
noncomputable def red_bus_length : ℝ := 48

/-- The length of the orange car in feet -/
noncomputable def orange_car_length : ℝ := red_bus_length / 4

/-- The length of the yellow bus in feet -/
noncomputable def yellow_bus_length : ℝ := orange_car_length * 3.5

/-- The length of the red bus that the yellow bus driver sees in feet -/
noncomputable def visible_red_bus_length : ℝ := red_bus_length

theorem red_bus_visible_length :
  visible_red_bus_length = red_bus_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_bus_visible_length_l1335_133558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_area_value_l1335_133531

/-- The area bounded by the curves y = √(4 - x²), y = 0, x = 0, and x = 1 -/
noncomputable def bounded_area : ℝ := ∫ x in Set.Icc 0 1, Real.sqrt (4 - x^2)

/-- Theorem stating that the bounded area is equal to π/3 + √3/2 -/
theorem bounded_area_value : bounded_area = π / 3 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_area_value_l1335_133531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l1335_133584

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 3)

theorem symmetry_of_sine_function (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∀ x, f ω (13 * Real.pi / 12 + x) = f ω (13 * Real.pi / 12 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l1335_133584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_relief_work_l1335_133577

/-- Represents the time and cost for a team to complete the work alone -/
structure TeamData where
  months : ℚ
  cost_per_month : ℚ

/-- Calculates the time and total cost for two teams working together -/
def cooperative_work (team_a team_b : TeamData) : (ℚ × ℚ) :=
  let combined_rate := 1 / team_a.months + 1 / team_b.months
  let time := 1 / combined_rate
  let total_cost := (team_a.cost_per_month + team_b.cost_per_month) * time
  (time, total_cost)

/-- Theorem stating the cooperative work time and cost for the given scenario -/
theorem earthquake_relief_work : 
  let team_a : TeamData := ⟨3, 12000⟩
  let team_b : TeamData := ⟨6, 5000⟩
  cooperative_work team_a team_b = (2, 34000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_relief_work_l1335_133577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_for_x_less_than_neg_one_l1335_133524

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_increases_for_x_less_than_neg_one
  (k : ℝ) (h_k : k ≠ 0) (h_point : inverse_proportion k (-1) = 2) :
  ∀ x₁ x₂, x₁ < -1 → x₂ < -1 → x₁ < x₂ → inverse_proportion k x₁ < inverse_proportion k x₂ :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_for_x_less_than_neg_one_l1335_133524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_five_percentile_l1335_133511

def satisfaction_data : List ℕ := [8, 4, 5, 6, 9, 8, 9, 7, 10, 10]

def is_percentile (p : ℚ) (x : ℕ) (data : List ℕ) : Prop :=
  (data.filter (λ y => y ≤ x)).length / data.length ≥ p / 100 ∧
  (data.filter (λ y => y ≥ x)).length / data.length ≥ (100 - p) / 100

theorem twenty_five_percentile :
  is_percentile 25 6 satisfaction_data :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_five_percentile_l1335_133511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1335_133587

-- Define the function f with domain [-2, 2]
noncomputable def f : ℝ → ℝ := sorry

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (2 * x + 1)

-- State the theorem about the domain of g
theorem domain_of_g :
  Set.Icc (-2 : ℝ) 2 = {x | f x ≠ 0} →
  {x : ℝ | g x ≠ 0} = Set.Ioo (-1/2 : ℝ) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1335_133587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_squared_l1335_133543

noncomputable section

/-- The parabola function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 4

/-- The x-coordinate of point A -/
def x_A : ℝ := 7/6

/-- The y-coordinate of point A -/
def y_A : ℝ := f x_A

/-- The square of the length of AB -/
def AB_squared : ℝ := (x_A - (2 - x_A))^2 + (y_A - (4 - y_A))^2

/-- Theorem stating the relationship between the square of AB's length and the given points -/
theorem AB_length_squared :
  ∀ (A B : ℝ × ℝ),
  A.1 ≠ B.1 →
  A.2 = f A.1 →
  B.2 = f B.1 →
  (1, 2) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB_squared :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_squared_l1335_133543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1335_133579

theorem problem_statement :
  (¬(∃ x₀ : ℝ, x₀ < 0 ∧ (2 : ℝ)^x₀ < (3 : ℝ)^x₀)) ∧
  (∀ x : ℝ, 0 < x ∧ x < Real.pi/2 → Real.sin x < x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1335_133579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1335_133569

/-- Calculates the cost of fencing a rectangular park -/
noncomputable def fencing_cost (area : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let length := Real.sqrt ((ratio_length * area) / ratio_width)
  let width := (ratio_width * length) / ratio_length
  let perimeter := 2 * (length + width)
  perimeter * cost_per_meter

/-- Proves that the fencing cost for the given park is 140 rupees -/
theorem park_fencing_cost :
  fencing_cost 4704 3 2 0.5 = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1335_133569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_range_m_l1335_133529

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if -1 < x ∧ x ≤ 0 then 1 / (x + 1) - 3
  else if 0 < x ∧ x ≤ 1 then x^2 - 3*x + 2
  else 0  -- Undefined for other x values

-- Define the set of m values
def M : Set ℝ := Set.Ioc (-9/4) (-2) ∪ Set.Ico 0 2

-- State the theorem
theorem two_roots_range_m :
  ∀ m : ℝ, (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = m * (x₁ + 1) ∧ g x₂ = m * (x₂ + 1)) ↔ m ∈ M :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_range_m_l1335_133529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_condition_l1335_133570

def b (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => if (b n i) % 3 = 0 then (b n i) / 3 else 2 * (b n i) + 2

def satisfies_condition (b₁ : ℕ) : Bool :=
  b₁ > 0 && b₁ ≤ 3000 && b₁ < b b₁ 1 && b₁ < b b₁ 2 && b₁ < b b₁ 3

theorem count_satisfying_condition :
  (Finset.filter (fun x => satisfies_condition x) (Finset.range 3001)).card = 2000 := by
  sorry

#eval (Finset.filter (fun x => satisfies_condition x) (Finset.range 3001)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_condition_l1335_133570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_not_below_l1335_133503

theorem graph_not_below (a : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x < 1 → Real.exp x - 1 ≥ x^2 - a*x) → 
  a ≥ 2 - Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_not_below_l1335_133503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rounded_to_10_blank_l1335_133545

-- Define a function to round a number to a specified number of decimal places
noncomputable def roundToDecimalPlaces (x : ℝ) (places : ℕ) : ℝ :=
  (⌊x * 10^places + 0.5⌋ : ℝ) / 10^places

-- Define the property we want to prove
def roundedTo10Blank (x : ℝ) : Prop :=
  ∃ n : ℕ, roundToDecimalPlaces x 2 = 10 + n / 100

-- State the theorem
theorem not_rounded_to_10_blank :
  ¬ roundedTo10Blank 9.996 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rounded_to_10_blank_l1335_133545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1335_133552

/-- Given a positive real s satisfying s^3 - (3/7)s + 1 = 0, 
    the sum of the infinite series s^3 + 2s^6 + 3s^9 + 4s^12 + ... is equal to 49s/9 -/
theorem infinite_series_sum (s : ℝ) (hs : s > 0) (heq : s^3 - 3/7 * s + 1 = 0) :
  ∃ (sum : ℝ), sum = (∑' n, n * s^(3*n)) ∧ sum = 49/9 * s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1335_133552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equations_subset_eq4_range_not_subset_eq_range_l1335_133568

-- Define the concept of a subset equation
def is_subset_equation (eq : ℝ → Prop) (ineq : ℝ → Prop) : Prop :=
  ∀ x, eq x → ineq x

-- Define the given equations and inequalities
def eq1 (x : ℝ) : Prop := 3*x - 2 = 0
def eq2 (x : ℝ) : Prop := 2*x - 3 = 0
def eq3 (x : ℝ) : Prop := x - (3*x + 1) = -7
def ineq1 (x : ℝ) : Prop := -x + 2 > x - 5
def ineq2 (x : ℝ) : Prop := 3*x - 1 > -x + 2

-- Define the system of inequalities
def system (m : ℝ) (x : ℝ) : Prop := x + m < 2*x ∧ x - 2 < m

-- Define the equation (2x-1)/3 = -3
def eq4 (x : ℝ) : Prop := (2*x - 1)/3 = -3

theorem subset_equations :
  (is_subset_equation eq2 (λ x ↦ ineq1 x ∧ ineq2 x)) ∧
  (is_subset_equation eq3 (λ x ↦ ineq1 x ∧ ineq2 x)) ∧
  ¬(is_subset_equation eq1 (λ x ↦ ineq1 x ∧ ineq2 x)) := by sorry

theorem subset_eq4_range (m : ℝ) :
  (∀ x, eq4 x → system m x) → -6 < m ∧ m < -4 := by sorry

theorem not_subset_eq_range (m : ℝ) :
  (¬∀ x, x = 1 → system m x) ∧ (¬∀ x, x = 2 → system m x) →
  m ≥ 2 ∨ m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equations_subset_eq4_range_not_subset_eq_range_l1335_133568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_possible_n_values_l1335_133501

theorem arithmetic_progression_possible_n_values : 
  ∃! k : ℕ, k = (Finset.filter 
    (λ n : ℕ ↦ n > 1 ∧ 153 % n = 0 ∧ (153 + n * (n - 1)) % n = 0) 
    (Finset.range 154)).card ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_possible_n_values_l1335_133501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_construction_l1335_133560

/-- Represents a point on a sphere -/
structure SpherePoint where
  x : ℝ
  y : ℝ
  z : ℝ
  on_sphere : x^2 + y^2 + z^2 = 1

/-- Represents a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a compass construction on a sphere or plane -/
inductive CompassConstruction
  | DrawCircle : SpherePoint → ℝ → CompassConstruction
  | MeasureDistance : SpherePoint → SpherePoint → CompassConstruction
  | PlanarConstruction : CompassConstruction

/-- Represents the vertices of a cube -/
structure CubeVertices where
  vertices : Fin 8 → SpherePoint

/-- Predicate to check if a point is on the surface of a sphere -/
def on_sphere_surface (s : Sphere) (p : SpherePoint) : Prop :=
  let (cx, cy, cz) := s.center
  (p.x - cx)^2 + (p.y - cy)^2 + (p.z - cz)^2 = s.radius^2

/-- Predicate to check if a cube is inscribed in a sphere -/
def is_inscribed_cube (cube : CubeVertices) (s : Sphere) : Prop :=
  ∀ v : Fin 8, on_sphere_surface s (cube.vertices v)

/-- Main theorem: It is possible to construct the vertices of an inscribed cube on a sphere
    using compass constructions on the sphere and planar Euclidean constructions -/
theorem inscribed_cube_construction (s : Sphere) :
  ∃ (constructions : List CompassConstruction) (cube : CubeVertices),
    (∀ v : Fin 8, on_sphere_surface s (cube.vertices v)) ∧
    (is_inscribed_cube cube s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_construction_l1335_133560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1335_133589

-- Define variables
variable (a b : ℝ)

-- Define the complex number z as a function
def z (a b : ℝ) : ℂ := (a + 2 * Complex.I) * (1 - b * Complex.I)

-- State the theorem
theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : Complex.re (z a b) = 2) :
  2 / a + 1 / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1335_133589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winnable_symmetry_l1335_133593

/-- Represents a bead color -/
inductive Color
| Sapphire
| Turquoise
deriving BEq, Repr

/-- Represents a necklace as a list of bead colors -/
def Necklace := List Color

/-- Checks if three consecutive beads have the same color -/
def hasThreeConsecutive (n : Necklace) : Bool :=
  sorry

/-- Checks if a necklace is valid according to the problem conditions -/
def isValidNecklace (n : Necklace) : Prop :=
  n.length = 6 * (n.length / 6) ∧ 
  n.count Color.Sapphire = 3 * (n.length / 6) ∧
  n.count Color.Turquoise = 3 * (n.length / 6) ∧
  ¬hasThreeConsecutive n

/-- Represents a player's move -/
inductive Move
| Tasty
| Stacy

/-- Applies a move to a necklace -/
def applyMove (m : Move) (n : Necklace) : Necklace :=
  sorry

/-- Checks if a necklace can be won with a given starting player -/
def canWin (startingPlayer : Move) (n : Necklace) : Prop :=
  sorry

/-- The main theorem to be proven -/
theorem winnable_symmetry (n : Necklace) :
  isValidNecklace n →
  canWin Move.Tasty n →
  canWin Move.Stacy n :=
by
  sorry

#check winnable_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winnable_symmetry_l1335_133593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_base_l1335_133521

/-- Represents a number in base b -/
structure BaseNumber (b : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < b

/-- Converts a BaseNumber to its decimal representation -/
def BaseNumber.toNat {b : ℕ} (n : BaseNumber b) : ℕ := sorry

/-- Theorem stating the base of the magic square is 5 -/
theorem magic_square_base : 
  ∀ b : ℕ, b > 1 →
  (let row1_21 : BaseNumber b := ⟨[1, 2], sorry⟩
   let col1_14 : BaseNumber b := ⟨[4, 1], sorry⟩
   (1 + row1_21.toNat + 2 = col1_14.toNat + 1 + 4)) → b = 5 := by
  sorry

#check magic_square_base

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_base_l1335_133521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_for_given_radius_and_area_l1335_133583

/-- The height of a right cylinder with given radius and surface area -/
noncomputable def cylinder_height (r : ℝ) (sa : ℝ) : ℝ :=
  (sa - 2 * Real.pi * r^2) / (2 * Real.pi * r)

/-- Theorem: A right cylinder with radius 3 meters and surface area 36π square meters has height 3 meters -/
theorem cylinder_height_for_given_radius_and_area :
  cylinder_height 3 (36 * Real.pi) = 3 := by
  -- Unfold the definition of cylinder_height
  unfold cylinder_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_for_given_radius_and_area_l1335_133583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_after_correction_l1335_133574

theorem correct_average_after_correction (n : ℕ) (initial_avg : ℚ) 
  (incorrect_numbers : List ℚ) (correct_numbers : List ℚ) 
  (hn : n = 20)
  (hinitial : initial_avg = 24)
  (hincorrect : incorrect_numbers = [35, 20, 50])
  (hcorrect : correct_numbers = [45, 30, 60])
  : (n : ℚ) * initial_avg - incorrect_numbers.sum + correct_numbers.sum = n * 25.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_after_correction_l1335_133574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_lock_combinations_correct_l1335_133592

/-- Calculates the number of possible combinations for a suitcase lock with 4 dials,
    where each dial has digits 0-9 and no two dials can have the same digit. -/
def suitcase_lock_combinations : Nat :=
  let num_dials : Nat := 4
  let digit_range : Finset Nat := Finset.range 10
  Nat.descFactorial 10 4

theorem suitcase_lock_combinations_correct :
  suitcase_lock_combinations = 5040 := by
  -- Unfold the definition
  unfold suitcase_lock_combinations
  
  -- Calculate the descending factorial
  have h : Nat.descFactorial 10 4 = 5040 := by
    -- This is a direct calculation, but we'll use sorry for brevity
    sorry
  
  -- Apply the calculation
  exact h

#eval suitcase_lock_combinations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_lock_combinations_correct_l1335_133592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_power_of_two_permutation_l1335_133564

def is_permutation_of_digits (a b : ℕ) : Prop :=
  ∃ (p : List ℕ → List ℕ), Function.Bijective p ∧ 
    p (Nat.digits 10 a) = Nat.digits 10 b

theorem no_power_of_two_permutation (k n : ℕ) (h1 : k > 3) (h2 : n > k) :
  ¬ is_permutation_of_digits (2^k) (2^n) := by
  sorry

#check no_power_of_two_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_power_of_two_permutation_l1335_133564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1335_133507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (80 - x)) + Real.sqrt (x * (3 - x))

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ) (x₀ : ℝ),
    (∀ x, 0 ≤ x ∧ x ≤ 3 → f x ≤ M) ∧
    (0 ≤ x₀ ∧ x₀ ≤ 3) ∧
    f x₀ = M ∧
    M = 4 * Real.sqrt 15 ∧
    x₀ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1335_133507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_is_500_l1335_133555

/-- In a race, runner A beats runner B by either a distance or time. -/
structure Race where
  length : ℝ
  distance_diff : ℝ
  time_diff : ℝ

/-- The time taken by runner A to complete the race -/
noncomputable def race_time (r : Race) : ℝ :=
  r.length * r.time_diff / r.distance_diff

/-- Theorem stating that for the given race conditions, A's race time is 500 seconds -/
theorem race_time_is_500 (r : Race) 
  (h1 : r.length = 1000)
  (h2 : r.distance_diff = 20)
  (h3 : r.time_diff = 10) :
  race_time r = 500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_is_500_l1335_133555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_approx_l1335_133562

/-- The revolutions per minute of a bus wheel -/
noncomputable def wheel_rpm (wheel_radius : ℝ) (bus_speed : ℝ) : ℝ :=
  let cm_per_km := 100000
  let minutes_per_hour := 60
  let cm_per_minute := bus_speed * cm_per_km / minutes_per_hour
  let wheel_circumference := 2 * Real.pi * wheel_radius
  cm_per_minute / wheel_circumference

/-- Theorem: Given a bus wheel with radius 100 cm and bus speed 66 km/h, 
    the wheel makes approximately 175.04 revolutions per minute -/
theorem bus_wheel_rpm_approx :
  let wheel_radius := (100 : ℝ)
  let bus_speed := (66 : ℝ)
  abs (wheel_rpm wheel_radius bus_speed - 175.04) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_approx_l1335_133562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_water_bottles_l1335_133516

theorem camp_water_bottles (bottles_per_case : ℕ) (cases_purchased : ℕ) 
  (camp_days : ℕ) (group1 : ℕ) (group2 : ℕ) (group3 : ℕ) 
  (bottles_per_child_per_day : ℕ) :
  bottles_per_case = 24 →
  cases_purchased = 13 →
  camp_days = 3 →
  group1 = 14 →
  group2 = 16 →
  group3 = 12 →
  bottles_per_child_per_day = 3 →
  (let group4 := (group1 + group2 + group3) / 2
   let total_children := group1 + group2 + group3 + group4
   let total_bottles_needed := total_children * bottles_per_child_per_day * camp_days
   let total_bottles_purchased := bottles_per_case * cases_purchased
   let additional_bottles_needed := total_bottles_needed - total_bottles_purchased
   additional_bottles_needed) = 255 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_water_bottles_l1335_133516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_tangent_relationship_l1335_133550

theorem sine_cosine_tangent_relationship :
  let a := Real.sin (46 * π / 180)
  let b := Real.cos (46 * π / 180)
  let c := Real.tan (46 * π / 180)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_tangent_relationship_l1335_133550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steinbart_theorem_l1335_133512

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
inductive Line where
  | TwoPoints : Point → Point → Line
  | PointAndDirection : Point → Point → Line

/-- Checks if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Checks if a point is inside a circle -/
def is_inside_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Checks if a point is on a line -/
def is_on_line (p : Point) (l : Line) : Prop := sorry

/-- Checks if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if a circle is inscribed in a triangle -/
def is_inscribed_circle (c : Circle) (A B C : Point) : Prop := sorry

theorem steinbart_theorem 
  (A B C D E F P D' E' F' : Point) 
  (k : Circle) :
  (∃ (BC CA AB : Line),
    -- Triangle ABC exists
    Line.TwoPoints A B = AB ∧ 
    Line.TwoPoints B C = BC ∧ 
    Line.TwoPoints C A = CA ∧
    -- k is inscribed in triangle ABC
    is_inscribed_circle k A B C ∧
    -- k is tangent to BC, CA, AB at D, E, F respectively
    is_on_line D BC ∧ is_on_line E CA ∧ is_on_line F AB ∧
    is_on_circle D k ∧ is_on_circle E k ∧ is_on_circle F k ∧
    -- P is inside k
    is_inside_circle P k ∧
    -- D', E', F' are intersections of DP, EP, FP with k
    is_on_circle D' k ∧ is_on_circle E' k ∧ is_on_circle F' k ∧
    is_on_line D' (Line.TwoPoints D P) ∧
    is_on_line E' (Line.TwoPoints E P) ∧
    is_on_line F' (Line.TwoPoints F P)) →
  -- Then AD', BE', CF' are concurrent
  are_concurrent 
    (Line.TwoPoints A D') 
    (Line.TwoPoints B E') 
    (Line.TwoPoints C F') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steinbart_theorem_l1335_133512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_b_l1335_133598

def a : Fin 2 → ℝ
  | 0 => 5
  | 1 => -7

def b : Fin 2 → ℝ
  | 0 => -6
  | 1 => -4

theorem dot_product_a_b :
  (Finset.univ.sum fun i => a i * b i) = -2 := by
  simp [a, b]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_b_l1335_133598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1335_133576

noncomputable section

open Real

def f (α : ℝ) : ℝ := 
  (sin (π/2 - α) + sin (-π - α)) / (3 * cos (2*π + α) + cos (3*π/2 - α))

theorem problem_solution (α : ℝ) (h : f α = 3) :
  (sin α - 3 * cos α) / (sin α + cos α) = -1/3 ∧
  ∃ a : ℝ, (abs a = 5/2 ∧
    ∀ x y : ℝ, (x - a)^2 + y^2 = 7 ↔
      (∃ t : ℝ, y = 2*x ∧
        (x - a)^2 + (2*x)^2 = 5 ∧
        ((x - a)^2 + (2*x)^2) * ((x - (x - a))^2 + (2*x - 2*x)^2) = 8)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1335_133576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sequence_properties_l1335_133519

/-- A sequence defined by a quadratic expression -/
def quadratic_sequence (a b c : ℝ) (n : ℕ) (h : n ≥ 3) (ha : a > 0) : ℕ → ℝ :=
  λ k ↦ a * k^2 + b * k + c

theorem quadratic_sequence_properties
  (a b c : ℝ) (n : ℕ) (h : n ≥ 3) (ha : a > 0) :
  let f := quadratic_sequence a b c n h ha
  ∀ k, k ≥ 1 → k < n →
  (3 * a + b > 0 → f (k + 1) > f k) ∧
  (3 * a + b = 0 → f (k + 1) ≥ f k) ∧
  (-(2 * n - 1) * a < b ∧ b < -3 * a →
    ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n ∧ f i > f j ∧ f j < f k) ∧
  ((2 * n - 1) * a + b = 0 → f (k + 1) ≤ f k) ∧
  ((2 * n - 1) * a + b < 0 → f (k + 1) < f k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sequence_properties_l1335_133519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l1335_133532

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the line
def my_line (x y k : ℝ) : Prop := y = k*x + 2

-- Theorem statement
theorem chord_length_is_four :
  ∀ k : ℝ, ∃ chord_length : ℝ,
    (∀ x y : ℝ, my_circle x y → my_line x y k → 
      (∃ x1 y1 x2 y2 : ℝ, 
        my_circle x1 y1 ∧ my_circle x2 y2 ∧ 
        my_line x1 y1 k ∧ my_line x2 y2 k ∧
        chord_length = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))) ∧
    chord_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l1335_133532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1335_133554

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain (x : ℝ) : Prop := -Real.pi/2 < x ∧ x < Real.pi/2

-- State the condition on f
def condition (x : ℝ) : Prop := 
  domain x → (deriv (deriv f)) x * Real.cos x + f x * Real.sin x > 0

-- State the theorem
theorem f_inequality (hf : ∀ x, condition f x) : 
  Real.sqrt 2 * f (Real.pi/3) > f (Real.pi/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1335_133554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l1335_133556

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

theorem trapezium_properties (t : Trapezium) 
  (h1 : t.side1 = 20)
  (h2 : t.side2 = 18)
  (h3 : t.height = 25) :
  area t = 475 ∧ t.side1 = 20 := by
  sorry

#check trapezium_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l1335_133556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_f_l1335_133571

noncomputable def f (x : ℝ) := (1 - x) / Real.sqrt (x + 2)

theorem range_of_x_in_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x > -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_f_l1335_133571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l1335_133586

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - log x

-- State the theorem
theorem f_monotone_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l1335_133586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1335_133526

/-- A function that represents |x(mx+2)| --/
def f (m : ℝ) (x : ℝ) : ℝ := |x * (m * x + 2)|

/-- The statement that m > 0 is sufficient but not necessary for f to be increasing on (0,+∞) --/
theorem sufficient_not_necessary_condition :
  (∀ m > 0, StrictMonoOn (f m) (Set.Ioi 0)) ∧
  (∃ m ≤ 0, StrictMonoOn (f m) (Set.Ioi 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1335_133526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_comparison_l1335_133533

noncomputable def f (x : ℝ) : ℝ := 6 / x

theorem inverse_proportion_comparison : f (-3) > f (-1) := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expressions
  simp
  -- Use real number properties to prove the inequality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_comparison_l1335_133533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bond_rate_is_6_25_percent_l1335_133541

/-- Calculates the interest rate of the second bond given the total investment,
    investment in the first bond, interest rate of the first bond, and desired total interest income. -/
noncomputable def calculate_second_bond_rate (total_investment : ℝ) (first_bond_investment : ℝ) 
                               (first_bond_rate : ℝ) (desired_income : ℝ) : ℝ :=
  let second_bond_investment := total_investment - first_bond_investment
  let first_bond_income := first_bond_investment * first_bond_rate
  let second_bond_income := desired_income - first_bond_income
  second_bond_income / second_bond_investment

/-- The interest rate of the second bond is 6.25% given the specified conditions. -/
theorem second_bond_rate_is_6_25_percent :
  let total_investment : ℝ := 32000
  let first_bond_investment : ℝ := 20000
  let first_bond_rate : ℝ := 0.0575
  let desired_income : ℝ := 1900
  calculate_second_bond_rate total_investment first_bond_investment first_bond_rate desired_income = 0.0625 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bond_rate_is_6_25_percent_l1335_133541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_open_interval_one_two_l1335_133594

open Real

/-- The function g as defined in the problem -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

/-- The theorem statement -/
theorem g_range_open_interval_one_two :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → g a b c > 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → g a b c < 2) ∧
  (∀ ε : ℝ, ε > 0 → ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 1| < ε) ∧
  (∀ ε : ℝ, ε > 0 → ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 2| < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_open_interval_one_two_l1335_133594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elan_speed_l1335_133500

/-- Two people moving towards each other with increasing speeds -/
noncomputable def meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) (distance1 : ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  (speed1 * (2 ^ t - 1) / Real.log 2 = distance1) ∧
  (speed2 * (2 ^ t - 1) / Real.log 2 = initial_distance - distance1)

theorem elan_speed (initial_distance speed1 distance1 : ℝ) :
  initial_distance = 120 →
  speed1 = 10 →
  distance1 = 80 →
  ∃ speed2 : ℝ, speed2 = 40 / 3 ∧ meeting_time initial_distance speed1 speed2 distance1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elan_speed_l1335_133500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_small_digit_sum_l1335_133509

/-- Sums the digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem multiple_with_small_digit_sum (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : q = 2 * p + 1) :
  ∃ (n : ℕ), n > 0 ∧ q ∣ n ∧ (∃ (s : ℕ), s > 0 ∧ s ≤ 3 ∧ sum_of_digits n = s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_small_digit_sum_l1335_133509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_metal_beads_count_total_beads_equation_l1335_133539

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of bracelets that can be made -/
def total_bracelets : ℕ := 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Rose has -/
def rose_total_beads : ℕ := rose_crystal_beads + rose_stone_beads

/-- The total number of beads needed for all bracelets -/
def total_beads_needed : ℕ := total_bracelets * beads_per_bracelet

theorem nancy_metal_beads_count : nancy_metal_beads = 40 := by
  rfl

theorem total_beads_equation :
  nancy_metal_beads + nancy_pearl_beads + rose_total_beads = total_beads_needed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_metal_beads_count_total_beads_equation_l1335_133539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1335_133590

noncomputable def distance_hour1 : ℝ := 20
noncomputable def distance_hour2 : ℝ := 40
noncomputable def distance_hour3 : ℝ := 60
noncomputable def distance_hour4 : ℝ := 100
noncomputable def total_time : ℝ := 4

noncomputable def total_distance : ℝ := distance_hour1 + distance_hour2 + distance_hour3 + distance_hour4

noncomputable def average_speed : ℝ := total_distance / total_time

theorem car_average_speed : average_speed = 55 := by
  -- Unfold definitions
  unfold average_speed total_distance
  -- Simplify the expression
  simp [distance_hour1, distance_hour2, distance_hour3, distance_hour4, total_time]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1335_133590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_10_eq_401_l1335_133549

def climb (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | 4 => 8
  | n + 5 => climb (n + 4) + climb (n + 3) + climb (n + 2) + climb (n + 1)

theorem climb_10_eq_401 : climb 10 = 401 := by
  rfl

#eval climb 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_10_eq_401_l1335_133549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l1335_133580

/-- A nonconstant two-variable polynomial with real coefficients -/
def TwoVarPolynomial := ℝ → ℝ → ℝ

/-- The property that the polynomial satisfies -/
def SatisfiesProperty (P : TwoVarPolynomial) (c : ℝ) : Prop :=
  ∀ x y : ℝ, (P x y)^2 = P (c * x * y) (x^2 + y^2)

/-- The polynomial is nonconstant -/
def IsNonconstant (P : TwoVarPolynomial) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, P x₁ y₁ ≠ P x₂ y₂

/-- Theorem stating the existence of a nonconstant polynomial satisfying the property -/
theorem polynomial_existence (c : ℝ) (hc : c ≠ 0) :
  ∃ P : TwoVarPolynomial, IsNonconstant P ∧ SatisfiesProperty P c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l1335_133580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l1335_133515

noncomputable def f (x a : ℝ) : ℝ := 
  ((Real.exp x - a)^2 + (Real.exp (-x) + a)^2) / (Real.exp x - Real.exp (-x))

theorem minimum_value_of_f (a : ℝ) : 
  (∀ x : ℝ, x > 0 → f x a ≥ 6) ∧ 
  (∃ x : ℝ, x > 0 ∧ f x a = 6) → 
  a = -1 ∨ a = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l1335_133515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_true_l1335_133578

-- Define the initial equation
def initial_equation : ℕ → ℕ → Prop :=
  fun x y => x = y

-- Define the final equation
def final_equation : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun a b c d e => a = b + c - d - e

-- Define the constraint of moving only one matchstick
def one_matchstick_move (initial : ℕ → ℕ → Prop) (final : ℕ → ℕ → ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (x y : ℕ), initial x y ∧ ∃ (a b c d e : ℕ), final a b c d e

-- Theorem statement
theorem equation_true :
  initial_equation 3 2 →
  one_matchstick_move initial_equation final_equation →
  final_equation 3 11 1 2 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_true_l1335_133578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fifth_power_l1335_133506

theorem coefficient_x_fifth_power : 
  Polynomial.coeff ((X - 1 : Polynomial ℝ) * (X + 1)^8) 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fifth_power_l1335_133506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_THH_before_HHH_is_seven_eighths_l1335_133544

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
  | Heads
  | Tails

/-- A fair coin is a coin that has equal probability of landing heads or tails -/
def fair_coin (p : CoinFlip → ℝ) : Prop :=
  p CoinFlip.Heads = 1/2 ∧ p CoinFlip.Tails = 1/2

/-- The probability of seeing THH before HHH when flipping a fair coin -/
noncomputable def prob_THH_before_HHH (p : CoinFlip → ℝ) : ℝ := 7/8

/-- Theorem: The probability of seeing THH before HHH when flipping a fair coin is 7/8 -/
theorem prob_THH_before_HHH_is_seven_eighths (p : CoinFlip → ℝ) 
  (h : fair_coin p) : prob_THH_before_HHH p = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_THH_before_HHH_is_seven_eighths_l1335_133544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_score_average_l1335_133514

theorem quiz_score_average (initial_students : ℕ) (dropped_students : ℕ) 
  (remaining_average : ℚ) (dropped_score : ℚ) 
  (h1 : initial_students = 16)
  (h2 : dropped_students = 1)
  (h3 : remaining_average = 62)
  (h4 : dropped_score = 70)
  : (initial_students * remaining_average + dropped_students * dropped_score) / initial_students = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_score_average_l1335_133514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_properties_l1335_133535

structure Cube where
  vertices : Finset (Fin 8)

structure Tetrahedron where
  vertices : Finset (Fin 4)

def is_right_angled_triangle (face : Finset (Fin 3)) : Prop := sorry
def is_equilateral_triangle (face : Finset (Fin 3)) : Prop := sorry

def select_vertices (cube : Cube) (n : Nat) : Finset (Fin n) := sorry

def form_tetrahedron (cube : Cube) : Tetrahedron := sorry

def faces_of_tetrahedron (tetra : Tetrahedron) : Finset (Finset (Fin 3)) := sorry

theorem cube_tetrahedron_properties (cube : Cube) : 
  ∃ (t1 t2 t3 : Tetrahedron),
    (∀ face, face ∈ faces_of_tetrahedron t1 → is_right_angled_triangle face) ∧
    (∀ face, face ∈ faces_of_tetrahedron t2 → is_equilateral_triangle face) ∧
    (∃! face, face ∈ faces_of_tetrahedron t3 ∧ is_equilateral_triangle face) := by
  sorry

#check cube_tetrahedron_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_properties_l1335_133535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_running_ratio_l1335_133523

theorem walking_running_ratio (walk_speed run_speed total_distance total_time : ℝ)
  (h1 : walk_speed = 4)
  (h2 : run_speed = 8)
  (h3 : total_distance = 12)
  (h4 : total_time = 2.25)
  (h5 : walk_speed > 0)
  (h6 : run_speed > 0) :
  let walk_distance := (total_time * walk_speed * run_speed - total_distance * walk_speed) / (run_speed - walk_speed)
  let run_distance := total_distance - walk_distance
  walk_distance = run_distance :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_running_ratio_l1335_133523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_eight_l1335_133510

-- Define the square
structure Square where
  side_length : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration
structure Configuration where
  square : Square
  circles : Fin 4 → Circle

-- Define the tangency conditions
def is_tangent_to_sides (c : Circle) (s : Square) : Prop :=
  ∃ (x y : ℝ), (x = 0 ∨ x = s.side_length) ∧ (y = 0 ∨ y = s.side_length) ∧
  ((c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2)

def are_circles_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

-- Main theorem
theorem square_side_length_is_eight (conf : Configuration) 
  (h1 : ∀ i, (conf.circles i).radius = 2)
  (h2 : ∀ i, is_tangent_to_sides (conf.circles i) conf.square)
  (h3 : ∀ i, are_circles_tangent (conf.circles i) (conf.circles ((i + 1) % 4))) :
  conf.square.side_length = 8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_eight_l1335_133510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_schemes_difference_l1335_133517

/-- Calculates the compound interest amount after n years -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  principal * (1 + rate / 2) ^ (2 * n)

/-- Calculates the simple interest amount after n years -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  principal * (1 + rate * n)

/-- The loan schemes problem -/
theorem loan_schemes_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) 
  (h_principal : principal = 12000)
  (h_compound_rate : compound_rate = 0.08)
  (h_simple_rate : simple_rate = 0.10) :
  let compound_amount := 
    compound_interest (principal / 2) compound_rate 6 + 
    compound_interest (compound_interest principal compound_rate 6 / 2) compound_rate 6
  let simple_amount := simple_interest principal simple_rate 12
  ‖simple_amount - compound_amount - 1415‖ < 1 := by
  sorry

#eval println! "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_schemes_difference_l1335_133517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1335_133538

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_identity 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_pos : φ > 0) 
  (h_period : ∀ x : ℝ, f ω φ (x + π) = f ω φ x)
  (h_max : ∀ x : ℝ, f ω φ x ≤ f ω φ (π/8)) :
  ∀ x : ℝ, f ω φ x = Real.sin (2 * x + π/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1335_133538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l1335_133518

noncomputable def a : ℝ := (-1/2:ℝ)^(-1:ℝ)
noncomputable def b : ℝ := (2:ℝ)^(-1/2:ℝ)
noncomputable def c : ℝ := (1/2:ℝ)^(-1/2:ℝ)
noncomputable def d : ℝ := (2:ℝ)^(-1:ℝ)

theorem largest_number :
  c ≥ a ∧ c ≥ b ∧ c ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l1335_133518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_l1335_133591

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Define the set of possible ratios y/x
def ratio_set : Set ℝ := {r | ∃ (x y : ℝ), x ≠ 0 ∧ circle_equation x y ∧ r = y / x}

-- Theorem statement
theorem ratio_range : ratio_set = Set.Iic (-2 * Real.sqrt 2) ∪ Set.Ici (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_l1335_133591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1335_133530

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x + a)

-- Define the range of a function
def isRange (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y, y ∈ S ↔ ∃ x, f x = y

-- State the theorem
theorem range_of_f (a : ℝ) :
  (isRange (f a) (Set.Ici 0)) ↔ a ≤ 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1335_133530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_max_parts_l1335_133565

open Complex

theorem complex_product_max_parts :
  ∀ θ : ℝ,
  let z₁ : ℂ := ⟨Real.cos θ, -1⟩
  let z₂ : ℂ := ⟨Real.sin θ, 1⟩
  (∀ φ : ℝ, (z₁ * z₂).re ≤ (3 : ℝ) / 2) ∧
  (∃ φ : ℝ, (z₁ * z₂).re = (3 : ℝ) / 2) ∧
  (∀ φ : ℝ, (z₁ * z₂).im ≤ Real.sqrt 2) ∧
  (∃ φ : ℝ, (z₁ * z₂).im = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_max_parts_l1335_133565
