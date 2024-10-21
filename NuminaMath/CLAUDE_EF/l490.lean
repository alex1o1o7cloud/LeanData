import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l490_49074

/-- The line equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A right triangle bounded by coordinate axes and a line --/
structure AxisAlignedTriangle where
  boundingLine : Line

/-- Calculate the area of an axis-aligned triangle --/
noncomputable def areaOfAxisAlignedTriangle (t : AxisAlignedTriangle) : ℝ :=
  let x_intercept := t.boundingLine.c / t.boundingLine.a
  let y_intercept := t.boundingLine.c / t.boundingLine.b
  (x_intercept * y_intercept) / 2

theorem area_of_specific_triangle :
  let t : AxisAlignedTriangle := { boundingLine := { a := 3, b := 4, c := 12 } }
  areaOfAxisAlignedTriangle t = 6 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l490_49074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_calculation_l490_49067

-- Define the dimensions and specifications
def roof_base_width : ℝ := 8
def roof_top_width : ℝ := 4
def roof_height : ℝ := 5
def triangle_side : ℝ := 6
def paint_coverage : ℝ := 100
def paint_cost_per_gallon : ℝ := 15

-- Define the theorem
theorem paint_cost_calculation :
  let roof_area := 2 * ((roof_base_width + roof_top_width) / 2 * roof_height)
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  let total_area := roof_area + triangle_area
  let gallons_needed := total_area / paint_coverage
  let total_cost := Int.ceil gallons_needed * paint_cost_per_gallon
  total_cost = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_calculation_l490_49067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angle_cosine_l490_49045

theorem isosceles_triangle_base_angle_cosine (k : ℝ) (h1 : 0 < k) (h2 : k ≤ 1/2) :
  ∃ x : ℝ, (0 < x ∧ x < Real.pi/2) ∧
    (Real.cos x = (1 + Real.sqrt (1 - 2*k)) / 2 ∨ Real.cos x = (1 - Real.sqrt (1 - 2*k)) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angle_cosine_l490_49045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_equality_l490_49001

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the setup
noncomputable def larger_circle : Circle := sorry
noncomputable def smaller_circle : Circle := sorry
noncomputable def inscribed_triangle : Triangle := sorry

-- Define the conditions
def are_internally_tangent (c1 c2 : Circle) : Prop := sorry
def is_equilateral (t : Triangle) : Prop := sorry
def is_inscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define tangent calculation
noncomputable def tangent_length (point : ℝ × ℝ) (circle : Circle) : ℝ := sorry

-- Main theorem
theorem tangent_sum_equality :
  are_internally_tangent larger_circle smaller_circle →
  is_equilateral inscribed_triangle →
  is_inscribed inscribed_triangle larger_circle →
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    tangent_length (inscribed_triangle.vertices i) smaller_circle =
    tangent_length (inscribed_triangle.vertices j) smaller_circle +
    tangent_length (inscribed_triangle.vertices k) smaller_circle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_equality_l490_49001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_in_fourth_quadrant_l490_49059

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi / 2 - α) * Real.sin (-α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α) * Real.sin (Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_value_in_fourth_quadrant (α : Real)
  (h1 : Real.pi / 2 < α ∧ α < 2 * Real.pi)  -- α is in the fourth quadrant
  (h2 : Real.cos (3 * Real.pi / 2 - α) = 2 / 3) :
  f α = -(Real.sqrt 5) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_in_fourth_quadrant_l490_49059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_flower_arrangement_l490_49092

theorem candle_flower_arrangement (total_candles : ℕ) (total_flowers : ℕ) (flowers_chosen : ℕ) 
  (total_groupings : ℕ) (h1 : total_candles = 4) (h2 : total_flowers = 9) (h3 : flowers_chosen = 8) 
  (h4 : total_groupings = 54) : 
  (Nat.choose total_candles (Nat.choose total_flowers flowers_chosen) * Nat.choose total_flowers flowers_chosen = total_groupings) → 
  (Nat.choose total_candles (Nat.choose total_flowers flowers_chosen) = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_flower_arrangement_l490_49092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sum_of_sines_l490_49035

theorem max_value_sum_of_sines :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ 
    ∀ (x : ℝ), Real.sin x + Real.sin (x - π/3) ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sum_of_sines_l490_49035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l490_49070

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x * Real.cos x - Real.sqrt 3 * a * (Real.cos x)^2 + (Real.sqrt 3 / 2) * a + b

theorem function_properties (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x ≥ 2) ∧ 
  (∀ x, f a b x ≤ 4) ∧ 
  (∃ x, f a b x = 2) ∧ 
  (∃ x, f a b x = 4) →
  (∀ k : ℤ, ∃ x : ℝ, x = 5 * Real.pi / 12 + k * Real.pi / 2 ∧ 
    (∀ y : ℝ, f a b (x + y) = f a b (x - y))) ∧
  a = 1 ∧ 
  b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l490_49070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_odd_function_theorem_l490_49086

-- Define the function f
noncomputable def f (n : ℝ) : ℝ → ℝ := fun x ↦ (n^2 - 3*n + 3) * x^(n+1)

-- State the theorem
theorem power_function_odd_function_theorem (n : ℝ) 
  (h1 : ∀ x, f n x = (n^2 - 3*n + 3) * x^(n+1))
  (h2 : ∃ c, ∀ x, f n x = c * x^n)  -- f is a power function
  (h3 : ∀ x, f n (-x) = -(f n x))   -- f is an odd function
  : (∀ x, f n x = x^3) ∧ 
    {x : ℝ | f n (x+1) + f n (3-2*x) > 0} = {x : ℝ | x < 4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_odd_function_theorem_l490_49086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l490_49072

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) - 2 * Real.sin x ^ 2 + 1

/-- Theorem stating that the maximum value of f(x) is 2 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l490_49072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctic_circle_spherical_distance_l490_49028

/-- A point on a sphere with given latitude -/
structure Sphere (R : ℝ) where
  latitude : ℝ

/-- The arc length between two points on a latitude circle -/
def arc_length_latitude_circle (R : ℝ) (latitude : ℝ) (A B : Sphere R) : ℝ := 
  sorry

/-- The spherical distance between two points on a sphere -/
def spherical_distance (R : ℝ) (A B : Sphere R) : ℝ := 
  sorry

/-- The spherical distance between two points on the Arctic Circle -/
theorem arctic_circle_spherical_distance (R : ℝ) (A B : Sphere R) : 
  A.latitude = 60 → 
  B.latitude = 60 → 
  arc_length_latitude_circle R 60 A B = π * R / 2 → 
  spherical_distance R A B = π * R / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctic_circle_spherical_distance_l490_49028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_divisibility_l490_49062

theorem repunit_divisibility (m : ℕ) (h : Nat.Coprime m 10) :
  ∃ n : ℕ, m ∣ (10^n - 1) / 9 ∧
  ∀ k : ℕ, ∃ n' : ℕ, n' > k ∧ m ∣ (10^n' - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_divisibility_l490_49062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_min_distance_l490_49055

/-- The circle with center (2,2) and radius 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

/-- The line y = kx -/
def line_eq (k x y : ℝ) : Prop := y = k * x

/-- The minimum distance between a point on the circle and a point on the line -/
noncomputable def min_distance : ℝ := 2 * Real.sqrt 2 - 1

theorem circle_line_min_distance (k : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), circle_eq x₁ y₁ ∧ line_eq k x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ), circle_eq x₃ y₃ → line_eq k x₄ y₄ →
      Real.sqrt ((x₃ - x₄)^2 + (y₃ - y₄)^2) ≥ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = min_distance →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_min_distance_l490_49055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l490_49089

-- Define the function g
def g : ℝ → ℝ := λ x ↦ x

-- State the theorem
theorem inverse_function_value (f : ℝ → ℝ) (h : f ∘ g = id ∧ g ∘ f = id) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l490_49089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l490_49005

/-- Given a line l and a circle O, prove that the trajectory of point P satisfies x^2 + y^2 = 2 -/
theorem trajectory_equation (a b α : ℝ) : 
  let l : ℝ × ℝ → Prop := λ (x, y) ↦ ∃ t, x = a + t * Real.sin α ∧ y = b + t * Real.cos α
  let O : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 4
  let P : ℝ × ℝ := (a, b)
  a^2 + b^2 = 2 :=
by
  intro
  sorry

/-- Helper lemma: If P is inside O and l intersects O at A and B forming a geometric sequence, then a^2 + b^2 = 2 -/
lemma geometric_sequence_implies_trajectory (a b α : ℝ) (A B : ℝ × ℝ) :
  let l : ℝ × ℝ → Prop := λ (x, y) ↦ ∃ t, x = a + t * Real.sin α ∧ y = b + t * Real.cos α
  let O : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 4
  let P : ℝ × ℝ := (a, b)
  (P.1^2 + P.2^2 < 4) →
  (l A) → (l B) →
  (O A) → (O B) →
  (∃ r, (A.1 - P.1)^2 + (A.2 - P.2)^2 = r * (a^2 + b^2) ∧
        a^2 + b^2 = r * ((B.1 - P.1)^2 + (B.2 - P.2)^2)) →
  a^2 + b^2 = 2 :=
by
  intros l O P h_P_in_O h_A_on_l h_B_on_l h_A_on_O h_B_on_O h_geometric
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l490_49005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCD_is_180_l490_49016

/-- Represents a trapezoid ABCD with points E and F -/
structure Trapezoid where
  -- BC and AD are the parallel sides
  BC : ℝ
  AD : ℝ
  -- Height of the trapezoid
  height : ℝ
  -- Point F divides AD
  AF : ℝ
  FD : ℝ
  -- Point E divides DC
  CE : ℝ
  ED : ℝ
  -- Conditions
  parallel : BC > 0 ∧ AD > 0
  ratio_BCAD : BC / AD = 5 / 7
  ratio_AFFD : AF / FD = 4 / 3
  ratio_CEED : CE / ED = 2 / 3
  area_ABEF : ℝ

/-- The area of the trapezoid ABCD -/
noncomputable def area (t : Trapezoid) : ℝ :=
  (t.BC + t.AD) * t.height / 2

/-- Theorem stating that if the area of ABEF is 123, then the area of ABCD is 180 -/
theorem area_ABCD_is_180 (t : Trapezoid) (h : t.area_ABEF = 123) : area t = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCD_is_180_l490_49016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l490_49025

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_five (q : ℝ) :
  q > 0 →
  geometric_sum 1 q 4 = 5 * geometric_sum 1 q 2 →
  geometric_sum 1 q 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l490_49025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pt_length_l490_49053

noncomputable section

structure Triangle (P Q R : ℝ × ℝ) where
  right_angle : (R.1 - Q.1) * (P.1 - Q.1) + (R.2 - Q.2) * (P.2 - Q.2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem pt_length 
  (P Q R S T : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (pr_length : distance P R = 5)
  (rq_length : distance R Q = 12)
  (s_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2))
  (t_on_qr : ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ T = (u * Q.1 + (1 - u) * R.1, u * Q.2 + (1 - u) * R.2))
  (rts_right_angle : (R.1 - T.1) * (S.1 - T.1) + (R.2 - T.2) * (S.2 - T.2) = 0)
  (st_length : distance S T = 3) :
  distance P T = 39 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pt_length_l490_49053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l490_49052

/-- The function h(t) = (t^2 + 1/2 t) / (t^2 + 2) -/
noncomputable def h (t : ℝ) : ℝ := (t^2 + 1/2 * t) / (t^2 + 2)

/-- The range of h is {1/4} -/
theorem range_of_h : Set.range h = {1/4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l490_49052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_tangent_line_l490_49061

noncomputable section

open Real

/-- Curve C₁ with polar equation ρ = 1 -/
def C₁ (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

/-- Curve C₂ with polar equation ρ = 4cos θ -/
def C₂ (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

/-- The intersection points of C₁ and C₂ -/
def intersection_points : Set (ℝ × ℝ) :=
  {(1/4, Real.sqrt 15/4), (1/4, -Real.sqrt 15/4)}

/-- The equation of the line tangent to both C₁ and C₂ -/
def tangent_line (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y - 2 = 0 ∨ x - Real.sqrt 3 * y - 2 = 0

theorem intersection_and_tangent_line :
  (∀ p ∈ intersection_points, ∃ θ₁ θ₂, C₁ θ₁ = p ∧ C₂ θ₂ = p) ∧
  (∃ l : ℝ → ℝ → Prop, l = tangent_line ∧
    (∃ θ₁, l (C₁ θ₁).1 (C₁ θ₁).2) ∧
    (∃ θ₂, l (C₂ θ₂).1 (C₂ θ₂).2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_tangent_line_l490_49061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l490_49081

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train 50 m long traveling at 144 km/hr takes 1.25 seconds to cross an electric pole -/
theorem train_crossing_pole : train_crossing_time 50 144 = 1.25 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l490_49081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_l490_49058

open Real

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * sin (x / 3 - φ)

theorem cos_alpha_plus_beta (φ α β : ℝ) :
  0 < φ → φ < π / 2 →
  f φ 0 = -1 →
  α ∈ Set.Icc 0 (π / 2) →
  β ∈ Set.Icc 0 (π / 2) →
  f φ (3 * α + π / 2) = 10 / 13 →
  f φ (3 * β + 2 * π) = 6 / 5 →
  cos (α + β) = 16 / 65 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_l490_49058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_arithmetic_sequence_l490_49068

def ellipse (b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / 25 + y^2 / b^2 = 1}

def major_axis_length : ℝ := 10
def minor_axis_length (b : ℝ) := 2 * b
noncomputable def focal_length (b : ℝ) := 2 * Real.sqrt (25 - b^2)

def arithmetic_sequence (a b c : ℝ) := b - a = c - b

theorem ellipse_arithmetic_sequence (b : ℝ) 
  (h1 : 0 < b) (h2 : b < 5) 
  (h3 : arithmetic_sequence (minor_axis_length b) (focal_length b) major_axis_length) : 
  b^2 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_arithmetic_sequence_l490_49068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l490_49009

-- Define the ellipse
noncomputable def Ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}

-- Define eccentricity
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the perimeter of triangle ABF₁
noncomputable def TrianglePerimeter (a : ℝ) : ℝ := 4 * a

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : Eccentricity a b = Real.sqrt 3 / 3)
  (h4 : TrianglePerimeter a = 4 * Real.sqrt 6) :
  Ellipse 6 4 = Ellipse a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l490_49009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l490_49049

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

-- Theorem statement
theorem symmetry_about_origin : ∀ x, f (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l490_49049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_score_for_20_hours_l490_49047

-- Define the data points
def data : List (ℝ × ℝ) := [(15, 79), (23, 97), (16, 64), (24, 92), (12, 58)]

-- Define the slope of the regression line
def m : ℝ := 2.5

-- Define the function to calculate the mean of a list of real numbers
noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

-- Define the function to calculate the y-intercept
noncomputable def calculateYIntercept (xMean yMean : ℝ) (slope : ℝ) : ℝ :=
  yMean - slope * xMean

-- Define the regression line function
noncomputable def regressionLine (x : ℝ) (slope : ℝ) (yIntercept : ℝ) : ℝ :=
  slope * x + yIntercept

-- Theorem statement
theorem estimated_score_for_20_hours (xMean yMean : ℝ) 
  (h1 : xMean = mean (data.map Prod.fst))
  (h2 : yMean = mean (data.map Prod.snd))
  (yIntercept : ℝ) 
  (h3 : yIntercept = calculateYIntercept xMean yMean m) :
  regressionLine 20 m yIntercept = 83 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_score_for_20_hours_l490_49047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_in_geometric_triangle_l490_49008

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  sum_angles : A + B + C = π
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition that sides form a geometric progression
def geometric_progression (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

-- State the theorem
theorem cosine_sum_in_geometric_triangle (t : Triangle) 
  (h : geometric_progression t) : 
  Real.cos (t.A - t.C) + Real.cos t.B + Real.cos (2 * t.B) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_in_geometric_triangle_l490_49008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_net_percent_change_formula_l490_49013

/-- The net percent change of a quantity over three time periods. -/
noncomputable def netPercentChange (i j k : ℝ) : ℝ :=
  i + j - k + (i * j - i * k - j * k - i * j * k / 100) / 100

/-- 
Theorem stating that the net percent change of a quantity 
that increases by i%, then by j%, and finally decreases by k% 
is equal to i + j - k + (ij - ik - jk - ijk/100)/100.
-/
theorem net_percent_change_formula (i j k : ℝ) :
  let initial_quantity : ℝ := 1  -- Arbitrary initial quantity
  let quantity_t1 : ℝ := initial_quantity * (1 + i / 100)
  let quantity_t2 : ℝ := quantity_t1 * (1 + j / 100)
  let quantity_t3 : ℝ := quantity_t2 * (1 - k / 100)
  let total_percent_change : ℝ := (quantity_t3 - initial_quantity) / initial_quantity * 100
  total_percent_change = netPercentChange i j k := by
  sorry

#check net_percent_change_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_net_percent_change_formula_l490_49013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_nonempty_subsets_of_P_l490_49048

def P : Set (Nat × Nat) := {p | p.1 + p.2 = 5 ∧ p.1 > 0 ∧ p.2 > 0}

theorem number_of_nonempty_subsets_of_P :
  Finset.card (Finset.powerset (Finset.filter (fun p => p.1 + p.2 = 5 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.range 5 ×ˢ Finset.range 5)) \ {∅}) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_nonempty_subsets_of_P_l490_49048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l490_49065

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

/-- Configuration of geometric objects -/
structure GeometricConfiguration where
  largeCircle : Circle
  smallCircle : Circle
  line1 : Line
  line2 : Line
  smallCircleIntersectsLarge : (smallCircle.center.1 - largeCircle.center.1)^2 + 
                               (smallCircle.center.2 - largeCircle.center.2)^2 < 
                               (smallCircle.radius + largeCircle.radius)^2
  distinctLines : line1 ≠ line2

/-- Predicate to check if a point is on a circle -/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a point is on a line -/
def onLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points (config : GeometricConfiguration) : 
  ∃ (n : ℕ), n ≤ 11 ∧ 
  (∀ (m : ℕ), m > 11 → 
    ¬∃ (points : Finset (ℝ × ℝ)), points.card = m ∧ 
      (∀ p ∈ points, (onCircle p config.largeCircle ∧ onLine p config.line1) ∨
                     (onCircle p config.largeCircle ∧ onLine p config.line2) ∨
                     (onCircle p config.smallCircle ∧ onLine p config.line1) ∨
                     (onCircle p config.smallCircle ∧ onLine p config.line2) ∨
                     (onLine p config.line1 ∧ onLine p config.line2) ∨
                     (onCircle p config.largeCircle ∧ onCircle p config.smallCircle))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l490_49065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l490_49036

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

-- State the theorem
theorem f_properties :
  -- Part 1: f is monotonically increasing on (-π/2, π/2)
  (∀ x y : ℝ, -π/2 < x ∧ x < y ∧ y < π/2 → f x < f y) ∧
  -- Part 2: Characterization of m for which f(x) ≥ mx² on (0, π/2)
  (∀ m : ℝ, (∀ x : ℝ, 0 < x ∧ x < π/2 → f x ≥ m * x^2) ↔ m ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l490_49036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_properties_l490_49040

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields here
  n_ge_3 : n ≥ 3

/-- Represents a diagonal of a polygon -/
structure Diagonal (P : Polygon n) where
  -- Add necessary fields here

/-- Represents that two diagonals do not intersect -/
def NonIntersecting (P : Polygon n) (d1 d2 : Diagonal P) : Prop :=
  sorry

/-- Represents a triangulation of a polygon -/
structure Triangulation (P : Polygon n) where
  diagonals : List (Diagonal P)
  non_intersecting : ∀ d1 d2, d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → NonIntersecting P d1 d2
  covers_polygon : sorry -- This should represent that the diagonals divide the polygon into triangles

/-- The degree measure equivalent to 180° -/
noncomputable def d : ℝ := 180

/-- Function to represent the interior angles of a polygon -/
noncomputable def interior_angles (P : Polygon n) : List ℝ :=
  sorry

theorem polygon_properties (n : ℕ) (P : Polygon n) :
  (∃ T : Triangulation P, True) ∧ 
  (List.sum (interior_angles P) = 2 * d * (n - 2)) :=
sorry

#check polygon_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_properties_l490_49040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l490_49051

open Real

theorem phi_value (φ : ℝ) (h1 : |φ| < π/2) :
  let f := λ x => 2 * sin (x + 2*φ)
  (∀ x, f (x + π/2 + π/4) = f (π/2 - x + π/4)) →
  (2 * sin (2*φ) > 0) →
  φ = 3*π/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l490_49051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l490_49088

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem function_inequalities :
  (∃ m_min m_max : ℝ,
    m_min = 1 / (2 * Real.exp 1) ∧
    m_max = Real.exp 1 ∧
    (∀ x : ℝ, x > 0 → ∀ m : ℝ, m_min ≤ m ∧ m ≤ m_max ↔ f x ≤ m * x ∧ m * x ≤ g x)) ∧
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 →
    (x₁ * f x₁ - x₂ * f x₂) * (x₁^2 + x₂^2) > 2 * x₂ * (x₁ - x₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l490_49088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_curve_eccentricity_l490_49044

/-- The quadratic equation defining the curve -/
def quadratic_equation (x y : ℝ) : Prop :=
  10 * x - 2 * x * y - 2 * y + 1 = 0

/-- The eccentricity of the curve -/
noncomputable def eccentricity : ℝ := Real.sqrt 2

/-- Theorem stating that the eccentricity of the quadratic curve is √2 -/
theorem quadratic_curve_eccentricity :
  ∀ x y : ℝ, quadratic_equation x y → 
  ∃ e : ℝ, e = eccentricity ∧ 
  (∃ a b c d : ℝ, (x - a)^2 / b^2 + (y - c)^2 / d^2 = 1 ∧ 
  e^2 = 1 - (min b d / max b d)^2) :=
by
  sorry

#check quadratic_curve_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_curve_eccentricity_l490_49044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l490_49018

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively

-- Define the conditions of the problem
def TriangleConditions (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧  -- Positive sides
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧  -- Positive angles
  t.A + t.B + t.C = Real.pi ∧  -- Sum of angles in a triangle
  t.b ^ 2 = t.a * t.c ∧  -- Geometric progression condition
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C  -- Given equation

-- Theorem statement
theorem triangle_problem (t : Triangle) (h : TriangleConditions t) :
  t.B = Real.pi / 3 ∧ 1 / Real.tan t.A + 1 / Real.tan t.C = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l490_49018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_median_theorem_l490_49037

theorem pythagorean_median_theorem (a b x y : ℝ) : 
  a > 0 → b > 0 → x > 0 → y > 0 →
  x^2 + y^2 = 4 * b^2 →  -- Pythagorean theorem
  a^2 = 2 * b^2 + (x^2 + y^2) / 4 →  -- Median formula
  x * y = (1/4) * (a^2 + a * Real.sqrt (a^2 + 8*b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_median_theorem_l490_49037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_statements_l490_49066

-- Define the statements
def statement1 : Prop := ∀ x y : ℝ, (x > y ↔ Real.log x > Real.log y)
def statement2 : Prop := ∀ a b c : ℝ, ((a > b → a * c^2 > b * c^2) ∧ ¬(a * c^2 > b * c^2 → a > b))
def statement3 : Prop := ∃ k : ℝ, (k = Real.sqrt 3 → (∀ x y : ℝ, y = k * x + 2 → (x^2 + y^2 = 1 → (∀ ε > 0, ∃ x' y' : ℝ, x'^2 + y'^2 < 1 ∧ (y' - (k * x' + 2))^2 < ε^2)))) ∧ ¬(∀ x y : ℝ, y = k * x + 2 → (x^2 + y^2 = 1 → (∀ ε > 0, ∃ x' y' : ℝ, x'^2 + y'^2 < 1 ∧ (y' - (k * x' + 2))^2 < ε^2)) → k = Real.sqrt 3)
def statement4 : Prop := ∀ α β : ℝ, ¬(α > β → Real.sin α > Real.sin β) ∧ ¬(Real.sin α > Real.sin β → α > β)

-- Function to count true statements
def countTrueStatements (s1 s2 s3 s4 : Bool) : Nat :=
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0)

-- Theorem to prove
theorem number_of_correct_statements : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ statement4) ∧ 
  (countTrueStatements false true true true = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_statements_l490_49066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l490_49069

theorem sequence_inequality (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n, n > 0 → (a n)^2 ≤ a n - a (n-1)) :
  ∀ n : ℕ, n > 0 → a n < 1 / (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l490_49069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_side_a_value_l490_49087

-- Define the function f
noncomputable def f (x : Real) : Real := Real.sin x * Real.cos (x - Real.pi / 6) + 1/2 * Real.cos (2 * x)

-- Theorem for the maximum value of f
theorem f_max_value : ∃ (x : Real), f x = 3/4 ∧ ∀ (y : Real), f y ≤ 3/4 := by
  sorry

-- Define a triangle ABC
structure Triangle (a b c : Real) where
  area : Real
  area_formula : area = 1/2 * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))

-- Theorem for the value of side a
theorem side_a_value {a b c : Real} (t : Triangle a b c) :
  t.area = 4 * Real.sqrt 3 →
  f (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) = 1/2 →
  b + c = 10 →
  a = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_side_a_value_l490_49087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_reciprocal_minus_negative_l490_49056

-- Define s and t as real numbers
variable (s t : ℝ)

-- Define the expression
noncomputable def expression (s t : ℝ) : ℝ := (1/s)^2 - (-t)

-- State the theorem
theorem square_reciprocal_minus_negative (s t : ℝ) (h : s ≠ 0) :
  expression s t = (1/s)^2 - (-t) := by
  -- Unfold the definition of expression
  unfold expression
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_reciprocal_minus_negative_l490_49056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_is_four_l490_49020

def sequenceTerm (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 4

theorem twelfth_term_is_four :
  (∀ n : ℕ, n ≥ 1 → sequenceTerm n * sequenceTerm (n + 1) = 12) →
  sequenceTerm 1 = 3 →
  sequenceTerm 2 = 4 →
  sequenceTerm 12 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_is_four_l490_49020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_curve_l490_49098

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line parallel to the tangent -/
def m : ℝ := 4

theorem tangent_lines_to_curve :
  ∃ (a b : ℝ), 
    (f' a = m) ∧ 
    (b = f a) ∧
    ((fun x ↦ m*x - m*a + b) = (fun x ↦ 4*x - 4) ∨ (fun x ↦ m*x - m*a + b) = (fun x ↦ 4*x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_curve_l490_49098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_frame_width_l490_49095

-- Define the picture frame
structure PictureFrame where
  width : ℝ
  height : ℝ
  perimeter : ℝ

-- Theorem statement
theorem picture_frame_width
  (frame : PictureFrame)
  (height_condition : frame.height = 9)
  (perimeter_condition : frame.perimeter = 30)
  (perimeter_formula : frame.perimeter = 2 * frame.width + 2 * frame.height) :
  frame.width = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_frame_width_l490_49095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_2_area_implies_a_3_l490_49000

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * abs (x - 1) - abs (x + 1)

-- Part I
theorem solution_set_for_a_2 :
  {x : ℝ | f 2 x ≥ 3} = Set.Iic (-2/3) ∪ Set.Ici 6 := by sorry

-- Part II
theorem area_implies_a_3 (a : ℝ) (h1 : a > 1) :
  (∃ x1 x2 : ℝ, f a x1 = 1 ∧ f a x2 = 1 ∧ 
    (1/2) * (x2 - x1) * 3 = 27/8) → a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_2_area_implies_a_3_l490_49000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_log_problem_l490_49033

theorem log_sum_difference (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log x + Real.log y - Real.log z = Real.log ((x * y) / z) :=
by sorry

theorem log_problem :
  Real.log 32 + Real.log 50 - Real.log 8 = Real.log 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_log_problem_l490_49033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_properties_l490_49027

noncomputable section

variable (x : ℝ)

def a : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (x, 4)

def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vectors_properties :
  (x = 2 → are_parallel (a x) (b x)) ∧
  (x = 0 → angle (a x) (b x) = π / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_properties_l490_49027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l490_49050

/-- Calculates the final amount after compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_growth :
  let principal : ℝ := 15000
  let annual_rate : ℝ := 0.05
  let years : ℕ := 8
  let compoundings_per_year : ℕ := 2
  let periods : ℕ := years * compoundings_per_year
  let rate_per_period : ℝ := annual_rate / (compoundings_per_year : ℝ)
  let final_amount : ℝ := compound_interest principal rate_per_period periods
  round_to_nearest final_amount = 22333 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l490_49050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l490_49084

theorem imaginary_part_of_z (z : ℂ) : z = (1 + Complex.I)^2 * (2 + Complex.I) → z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l490_49084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_with_128_three_digit_remainders_of_5_l490_49075

theorem divisor_with_128_three_digit_remainders_of_5 :
  ∃! d : ℕ, 
    (d > 0) ∧ 
    (∃! s : Finset ℕ, 
      s.card = 128 ∧ 
      (∀ w ∈ s, 100 ≤ w ∧ w ≤ 999 ∧ w % d = 5) ∧
      (∀ w, 100 ≤ w ∧ w ≤ 999 ∧ w % d = 5 → w ∈ s)) ∧
    (∀ d' : ℕ, 0 < d' ∧ d' < d → 
      ¬∃ s : Finset ℕ, 
        s.card = 128 ∧ 
        (∀ w ∈ s, 100 ≤ w ∧ w ≤ 999 ∧ w % d' = 5) ∧
        (∀ w, 100 ≤ w ∧ w ≤ 999 ∧ w % d' = 5 → w ∈ s)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_with_128_three_digit_remainders_of_5_l490_49075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_equation_l490_49041

/-- Represents a hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- Represents a line with slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The theorem statement -/
theorem hyperbola_and_line_equation 
  (C : Hyperbola) 
  (h_eccentricity : C.a / Real.sqrt (C.a^2 + C.b^2) = 2)
  (h_point : C.a^2 * 2^2 - C.b^2 * 3^2 = C.a^2 * C.b^2)
  (l : Line)
  (h_slope : l.slope = Real.sqrt 5 / 5)
  (h_intersect : ∃ (P Q : ℝ × ℝ) (M : ℝ), 
    (P.1^2 / C.a^2 - P.2^2 / C.b^2 = 1) ∧
    (Q.1^2 / C.a^2 - Q.2^2 / C.b^2 = 1) ∧
    (P.1 = l.slope * P.2 + l.intercept) ∧
    (Q.1 = l.slope * Q.2 + l.intercept) ∧
    (M = l.intercept) ∧
    (Q = ((P.1 + M) / 2, P.2 / 2))) :
  (C.a = 1 ∧ C.b = Real.sqrt 3) ∧ 
  (l.intercept = Real.sqrt 21 ∨ l.intercept = -Real.sqrt 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_equation_l490_49041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_sales_increase_effect_l490_49085

theorem price_reduction_sales_increase_effect (P S : ℝ) (hP : P > 0) (hS : S > 0) : 
  let new_price := P * (1 - 0.15)
  let new_sales := S * (1 + 0.80)
  let original_revenue := P * S
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.53 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_sales_increase_effect_l490_49085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_equals_43_l490_49064

-- Define the function g as noncomputable
noncomputable def g : ℝ → ℝ := fun v => (v^2 + 10*v + 28) / 4

-- State the theorem
theorem g_of_8_equals_43 : g 8 = 43 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num

-- State the condition given in the problem
axiom g_condition : ∀ x : ℝ, g (2*x - 4) = x^2 + x + 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_equals_43_l490_49064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_balloons_theorem_l490_49038

def valid_n (n : ℕ) : Prop := n > 0 ∧ (n * (n + 1) / 2) % n = 0

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def shooting_sequences (n : ℕ) (k : ℕ) : Prop :=
  valid_n n → k ≥ double_factorial n

theorem disk_balloons_theorem (n : ℕ) (h : valid_n n) :
  Odd n ∧ ∃ k, shooting_sequences n k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_balloons_theorem_l490_49038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l490_49019

-- Define the equation for a
def equation (a : ℝ) : Prop := a^5 + a^3 - 1 = 0

-- Define the intersection points
def intersection_point (a : ℝ) : ℝ × ℝ := (a^5, a)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance (a : ℝ) (h : equation a) :
  distance (intersection_point a) (intersection_point (-a)) = Real.sqrt (4 * a^2) := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l490_49019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_a_probability_l490_49012

/-- The probability of an event occurring exactly k times in n trials -/
noncomputable def probability_exactly_k (n k : ℕ) (p : ℝ) : ℝ :=
  1 / Real.sqrt (n * p * (1 - p)) * Real.exp (-(((k : ℝ) - n * p)^2 / (2 * n * p * (1 - p))))

/-- The probability of event A occurring exactly 70 times in 243 trials -/
theorem event_a_probability : 
  abs (probability_exactly_k 243 70 0.25 - 0.0231) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_a_probability_l490_49012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacobs_tank_capacity_l490_49030

/-- Represents the capacity of Jacob's water tank in milliliters -/
def tank_capacity : ℕ → ℕ := sorry

/-- Represents the daily water collection from rain in milliliters -/
def rain_collection : ℕ := 800

/-- Represents the daily water collection from river in milliliters -/
def river_collection : ℕ := 1700

/-- Represents the number of days it takes to fill the tank -/
def days_to_fill : ℕ := 20

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℕ := 1000

theorem jacobs_tank_capacity :
  tank_capacity days_to_fill = (rain_collection + river_collection) * days_to_fill ∧
  (tank_capacity days_to_fill) / ml_to_l = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacobs_tank_capacity_l490_49030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l490_49090

/-- If the cost price is 92% of the selling price, then the profit percentage is approximately 8.70%. -/
theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.92 * selling_price) : 
  ∃ ε > 0, |((selling_price - cost_price) / cost_price * 100) - 8.70| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l490_49090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_tire_distance_l490_49007

/-- The wear rate of rear tires in wear per kilometer -/
noncomputable def rear_wear_rate : ℝ := 1 / 15000

/-- The wear rate of front tires in wear per kilometer -/
noncomputable def front_wear_rate : ℝ := 1 / 25000

/-- The total wear of a tire before it needs replacement -/
def total_wear : ℝ := 1

/-- The maximum distance a truck can travel without replacing tires -/
noncomputable def max_distance : ℝ := 18750

/-- Theorem: The maximum distance a truck can travel is correct given the wear rates -/
theorem truck_tire_distance :
  max_distance * (rear_wear_rate + front_wear_rate) = 2 * total_wear := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_tire_distance_l490_49007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_iff_m_geq_two_l490_49060

/-- A function f is monotonically increasing on an interval [a,b] if for any x, y in [a,b] with x ≤ y, f(x) ≤ f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- The function f(x) = x^2 - 4x + m*ln(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + m * Real.log x

theorem monotonically_increasing_iff_m_geq_two :
  ∀ m : ℝ, (MonotonicallyIncreasing (f m) 1 2) ↔ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_iff_m_geq_two_l490_49060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_for_line_in_plane_l490_49034

/-- Given a line l with direction vector s = (1, 2, x) and a plane α with normal vector n = (-2, y, 2),
    if l is contained in α, then the maximum value of x*y is 1/4. -/
theorem max_xy_for_line_in_plane (x y : ℝ) : 
  (1 * (-2) + 2 * y + x * 2 = 0) →  -- line l is contained in plane α
  x * y ≤ (1/4 : ℝ) := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_for_line_in_plane_l490_49034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_inequality_solution_l490_49022

def solution_set : Set ℝ := {x | (x - 1) / (x - 3) ≤ 0 ∧ x ≠ 3}

theorem fractional_inequality_solution : solution_set = Set.Icc 1 3 \ {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_inequality_solution_l490_49022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_slope_relationship_l490_49080

/-- A line in a 2D plane -/
structure Line : Type where
  angle : ℝ
  slope : ℝ

/-- The angle of inclination of a line -/
noncomputable def angle_of_inclination (l : Line) : ℝ := l.angle

/-- The slope of a line -/
noncomputable def slope_of_line (l : Line) : ℝ := l.slope

theorem angle_slope_relationship (l : Line) :
  (angle_of_inclination l > π/4) → 
  (¬ ((angle_of_inclination l > π/4) ↔ (slope_of_line l > 1))) ∧
  ((slope_of_line l > 1) → (angle_of_inclination l > π/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_slope_relationship_l490_49080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l490_49054

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}

-- Define set B
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

-- Define set C
def C (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 2 - a}

theorem set_operations_and_range :
  (∃ (a : ℝ),
    (A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5}) ∧
    (B ∪ (U \ A) = {x : ℝ | x ≤ 5 ∨ x ≥ 9}) ∧
    (C a ∪ (U \ B) = U → a ≤ -3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l490_49054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_formula_l490_49097

/-- The volume of a truncated cone (frustum) with height h and base radii R and r. -/
noncomputable def frustumVolume (h R r : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

/-- Theorem stating that the volume of a truncated cone (frustum) with height h and base radii R and r
    is equal to (1/3) * π * h * (R² + R*r + r²). -/
theorem frustum_volume_formula (h R r : ℝ) (h_pos : h > 0) (R_pos : R > 0) (r_pos : r > 0) :
  frustumVolume h R r = (1/3) * Real.pi * h * (R^2 + R*r + r^2) := by
  -- Unfold the definition of frustumVolume
  unfold frustumVolume
  -- The equation is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_formula_l490_49097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l490_49003

theorem angle_in_third_quadrant (θ : Real) 
  (h1 : Real.cos θ < 0) 
  (h2 : Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ)) : 
  θ ∈ Set.Ioo π (3 * π / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l490_49003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_cylinder_l490_49039

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The shape described by the equation ρ = c * sin(φ) in spherical coordinates -/
def cylinderShape (c : ℝ) (point : SphericalCoord) : Prop :=
  c > 0 ∧ point.ρ = c * Real.sin point.φ

/-- Predicate to check if a set of SphericalCoord points forms a cylinder -/
def isCylinder (shape : Set SphericalCoord) : Prop :=
  ∃ (c : ℝ), c > 0 ∧ ∀ point ∈ shape, point.ρ = c * Real.sin point.φ

theorem equation_describes_cylinder (c : ℝ) :
  ∃ (shape : Set SphericalCoord), 
    (∀ point ∈ shape, cylinderShape c point) ∧ 
    (isCylinder shape) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_cylinder_l490_49039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_2_equals_6_l490_49043

-- Define the function q
noncomputable def q (x : ℝ) : ℝ := 
  Real.sign (3*x - 3) * Real.sqrt (abs (3*x - 3)) + 
  3 * Real.sign (3*x - 3) * (abs (3*x - 3))^(1/6) + 
  (abs (3*x - 3))^(1/8)

-- Theorem statement
theorem q_2_equals_6 : 
  (∃ n : ℤ, q 2 = n) → q 2 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_2_equals_6_l490_49043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_cost_price_l490_49046

theorem watch_cost_price :
  ∃ (cost_price : ℚ),
  (cost_price * 75 / 100 = cost_price * 110 / 100 - 200) ∧
  cost_price = 571.43 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_cost_price_l490_49046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l490_49096

theorem cosine_of_angle_through_point (α : ℝ) (P : ℝ × ℝ) :
  P = (-5, 12) →
  Real.cos α = -5 / 13 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l490_49096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l490_49078

-- Define auxiliary functions
noncomputable def area : Set ℝ × Set ℝ × Set ℝ → ℝ := sorry
def congruent : (Set ℝ × Set ℝ × Set ℝ) → (Set ℝ × Set ℝ × Set ℝ) → Prop := sorry

-- Define the propositions
def proposition1 : Prop := ∀ x y : ℝ, x * y = 1 → (x = 1 / y ∧ y = 1 / x)
def proposition2 : Prop := ∀ t1 t2 : Set ℝ × Set ℝ × Set ℝ, area t1 = area t2 → congruent t1 t2
def proposition3 : Prop := ∀ x y : ℝ, x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2)
def proposition4 : Prop := ∃ x : ℝ, 4 * x^2 - 4 * x + 1 ≤ 0

-- Define the theorem
theorem proposition_truth : 
  (∀ x y : ℝ, x = 1 / y ∧ y = 1 / x → x * y = 1) ∧ 
  (∃ t1 t2 : Set ℝ × Set ℝ × Set ℝ, area t1 ≠ area t2 ∧ congruent t1 t2) ∧ 
  (∃ x y : ℝ, x + y ≠ 3 ∧ x = 1 ∧ y = 2) ∧
  (∀ x : ℝ, 4 * x^2 - 4 * x + 1 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l490_49078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_college_students_count_l490_49057

theorem college_students_count : ℕ := by
  -- Given conditions
  let boy_girl_ratio : Rat := 5 / 7
  let num_girls : ℕ := 140
  
  -- Calculate number of boys
  let num_boys : ℕ := (num_girls * 5) / 7
  
  -- Total number of students
  let total_students : ℕ := num_boys + num_girls
  
  -- Prove that the total number of students is 240
  have h : total_students = 240 := by
    -- Calculation steps
    sorry
  
  exact total_students


end NUMINAMATH_CALUDE_ERRORFEEDBACK_college_students_count_l490_49057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l490_49079

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) - Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), f x ≤ 2) ∧
  (f (-Real.pi / 12) = 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), f x ≥ -Real.sqrt 3) ∧
  (f (Real.pi / 3) = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l490_49079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l490_49014

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is positive if all its terms are positive -/
def IsPositive (a : Sequence) : Prop :=
  ∀ n, a n > 0

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

/-- Given sequence a, define sequence b as b_n = a_{n+2} / a_n -/
noncomputable def b (a : Sequence) : Sequence :=
  λ n => a (n + 2) / a n

/-- Given sequence a, define sequence c as c_n = a_n * a_{n+1}^2 -/
def c (a : Sequence) : Sequence :=
  λ n => a n * (a (n + 1))^2

theorem geometric_sequence_properties (a : Sequence) 
  (ha : IsPositive a) :
  (IsGeometric a → IsGeometric (c a)) ∧
  (IsGeometric (c a) → (∀ n, b a (n + 1) ≥ b a n) → IsGeometric a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l490_49014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l490_49042

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∀ x ∈ domain, f a b c x = f a b c x) →  -- This is to ensure f is well-defined on the domain
  (f a b c 0 = 0) →  -- The curve passes through the origin
  (HasDerivAt (f a b c) (-1) (-1) ∧ HasDerivAt (f a b c) (-1) 1) →  -- The slope of the tangent line at x = ±1 is -1
  (∃ f_final : ℝ → ℝ, 
    (∀ x ∈ domain, f a b c x = f_final x) ∧  -- f(x) = x^3 - 4x
    (∀ x ∈ domain, f_final x = x^3 - 4*x) ∧
    (∃ max min : ℝ, 
      (∀ x ∈ domain, f_final x ≤ max) ∧
      (∀ x ∈ domain, min ≤ f_final x) ∧
      (max + min = 0))) :=  -- The sum of the maximum and minimum values of f(x) is 0
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l490_49042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l490_49099

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 500

/-- The speeds of the three runners in meters per second -/
def runner_speeds : Fin 3 → ℝ
| 0 => 4.4
| 1 => 4.8
| 2 => 5.0
| _ => 0 -- This case should never occur due to Fin 3

/-- The time at which all runners meet again -/
def meeting_time : ℝ := 2500

/-- The set of all positive times when the runners' positions are congruent modulo the track length -/
def meeting_times : Set ℝ :=
  {t : ℝ | t > 0 ∧ ∀ i j : Fin 3, (runner_speeds i * t - runner_speeds j * t) % track_length = 0}

theorem runners_meet_time :
  meeting_time ∈ meeting_times ∧ ∀ t ∈ meeting_times, t ≥ meeting_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l490_49099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_fraction_l490_49024

/-- Floor function: greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- The expression (n^2 + 1) / (floor(√n)^2 + 2) is not an integer for any positive integer n -/
theorem not_integer_fraction (n : ℕ+) : 
  ¬ ∃ (k : ℤ), (n.val^2 + 1 : ℝ) / ((floor (Real.sqrt n.val))^2 + 2) = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_fraction_l490_49024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l490_49011

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (1, 3)
noncomputable def C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

theorem problem_solution :
  C = (0, 2) ∧
  dot_product (vector O A) (vector A B) = 0 ∧
  angle (vector O A) (vector O C) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l490_49011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_g_transformed_l490_49094

noncomputable section

-- Define the original function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -2 ∧ x < 1 then -1 - x
  else if x ≥ 1 ∧ x ≤ 3 then Real.sqrt (9 - (x - 3)^2) - 1
  else if x > 3 ∧ x ≤ 4 then 3 * (x - 3)
  else 0  -- undefined for other x values

-- Define the transformed function h
noncomputable def h (x : ℝ) : ℝ :=
  if x ≥ -8 ∧ x < -1 then -(x + 5) / 3
  else if x ≥ -1 ∧ x ≤ 7 then Real.sqrt (9 - ((x - 7) / 3)^2) - 1
  else if x > 7 ∧ x ≤ 10 then x - 7
  else 0  -- undefined for other x values

-- Theorem stating the equivalence of h(x) and g((x + 2)/3)
theorem h_equals_g_transformed (x : ℝ) : 
  (x ≥ -8 ∧ x ≤ 10) → h x = g ((x + 2) / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_g_transformed_l490_49094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l490_49023

-- Define the function f
noncomputable def f (b x : ℝ) : ℝ := (x^2 + b*x + b) * Real.sqrt (1 - 2*x)

-- State the theorem
theorem f_max_value (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f b x ≤ max b (Real.sqrt 3)) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 0, f b x = max b (Real.sqrt 3)) := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l490_49023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l490_49093

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y = -4/3 * abs x + 2
def C₂ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define when a point is on a curve
def on_curve (p : Point) (curve : ℝ → ℝ → Prop) : Prop :=
  curve p.x p.y

-- Theorem statement
theorem curves_intersection :
  ∃ (p₁ p₂ p₃ : Point),
    (on_curve p₁ C₁ ∧ on_curve p₁ C₂) ∧
    (on_curve p₂ C₁ ∧ on_curve p₂ C₂) ∧
    (on_curve p₃ C₁ ∧ on_curve p₃ C₂) ∧
    (∀ (p : Point), (on_curve p C₁ ∧ on_curve p C₂) → (p = p₁ ∨ p = p₂ ∨ p = p₃)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l490_49093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l490_49006

-- Define the ellipse E
noncomputable def E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line l
noncomputable def l (x : ℝ) : ℝ := x / 2

-- Define the condition for points A and B on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := E p.1 p.2

-- Define the vector product
def vec_product (p q : ℝ × ℝ) : ℝ := (p.1 - q.1) * (p.2 - q.2)

theorem ellipse_intersection_theorem :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    on_ellipse A ∧
    on_ellipse B ∧
    A.2 = l A.1 ∧
    B.2 = l B.1 ∧
    (P.1^2 + P.2^2) = 4 * vec_product P A * vec_product P B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l490_49006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_with_quarter_angle_inclination_l490_49026

noncomputable def L1 : ℝ → ℝ := λ x ↦ -Real.sqrt 3 * x + 1

noncomputable def angle_L1 : ℝ := Real.arctan (-Real.sqrt 3)

noncomputable def slope_L2_L3 : ℝ := Real.sqrt 3 / 3

noncomputable def L2 : ℝ → ℝ := λ x ↦ slope_L2_L3 * (x - Real.sqrt 3) - 1

noncomputable def L3 : ℝ → ℝ := λ x ↦ slope_L2_L3 * x - 5

theorem lines_with_quarter_angle_inclination :
  (∀ x, L2 x = -1 / 3 * (Real.sqrt 3 * x - 3 * L2 x - 6)) ∧
  (∀ x, L3 x = -1 / 3 * (Real.sqrt 3 * x - 3 * L3 x - 15)) ∧
  Real.arctan slope_L2_L3 = 1 / 4 * angle_L1 ∧
  L2 (Real.sqrt 3) = -1 ∧
  L3 0 = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_with_quarter_angle_inclination_l490_49026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_l490_49029

theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∀ n ∈ ({a, b, c, d, e} : Set ℤ), Odd n) →  -- all numbers are odd
  (b = a + 2) →                        -- consecutive
  (c = b + 2) →                        -- consecutive
  (d = c + 2) →                        -- consecutive
  (e = d + 2) →                        -- consecutive
  (a + c = 146) →                      -- sum of a and c
  (e = 79) →                           -- value of e
  d = 77 :=                            -- conclusion: value of d
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_l490_49029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_alone_time_c_alone_days_l490_49002

/-- The time taken for A and B to finish the job together -/
noncomputable def time_AB : ℝ := 15

/-- The time taken for A, B, and C to finish the job together -/
noncomputable def time_ABC : ℝ := 3

/-- The work rate of person X (fraction of job completed per day) -/
noncomputable def work_rate (X : ℝ) : ℝ := 1 / X

/-- The work rate of C alone -/
noncomputable def C : ℝ := 4 / 15

theorem c_alone_time (A B : ℝ) :
  work_rate time_AB = work_rate A + work_rate B →
  work_rate time_ABC = work_rate A + work_rate B + work_rate C →
  C = 4 / 15 :=
by sorry

theorem c_alone_days :
  1 / C = 3.75 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_alone_time_c_alone_days_l490_49002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_range_l490_49076

noncomputable def f (x : ℝ) := 2 * Real.sin (3 * x - Real.pi / 3)

theorem function_properties_and_inequality_range :
  (∃ ω φ : ℝ, ω > 0 ∧ -Real.pi/2 < φ ∧ φ < 0 ∧
    (∀ x, f x = 2 * Real.sin (ω * x + φ)) ∧
    Real.tan φ = -Real.sqrt 3) ∧
  (∀ x₁ x₂ : ℝ, |f x₁ - f x₂| = 4 → |x₁ - x₂| ≥ Real.pi/3) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/6) → ∀ m : ℝ, m ≥ 1/3 → m * f x + 2 * m ≥ f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/6) ∧ ∀ m : ℝ, m < 1/3 → m * f x + 2 * m < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_range_l490_49076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_properties_l490_49077

/-- Definition of foci for a conic section -/
noncomputable def foci (C : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (C : Set (ℝ × ℝ)) : ℝ := sorry

/-- Definition of asymptotes for a hyperbola -/
noncomputable def asymptotes (C : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Given an ellipse C₁ and a hyperbola C₂ that share the same foci, 
    prove the eccentricity of C₁ and the equations of the asymptotes of C₂ -/
theorem ellipse_hyperbola_properties (n : ℝ) :
  let C₁ := {(x, y) : ℝ × ℝ | x^2/3 + y^2/n = 1}
  let C₂ := {(x, y) : ℝ × ℝ | x^2 - y^2/n = 1}
  (∀ (F₁ F₂ : ℝ × ℝ), (F₁ ∈ foci C₁ ∧ F₂ ∈ foci C₁) ↔ (F₁ ∈ foci C₂ ∧ F₂ ∈ foci C₂)) →
  (∃ (e : ℝ), eccentricity C₁ = e ∧ e^2 = 2/3) ∧
  (∀ (x y : ℝ), (x, y) ∈ asymptotes C₂ ↔ y = x ∨ y = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_properties_l490_49077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_sharing_l490_49082

theorem money_sharing (total amanda ben carlos : ℝ) : 
  amanda + ben + carlos = total →
  amanda / 2 = ben / 3 →
  amanda / 2 = carlos / 5 →
  amanda = 30 →
  total = 150 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_sharing_l490_49082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_of_one_plus_i_sqrt_three_l490_49073

noncomputable def z : ℂ := 1 + Complex.I * Real.sqrt 3

theorem complex_argument_of_one_plus_i_sqrt_three :
  Complex.arg z = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_of_one_plus_i_sqrt_three_l490_49073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l490_49010

noncomputable def f (x : ℝ) := 1 / (2 * x - 1) + Real.sqrt (x + 1) + (3 * x - 1) ^ (1/3 : ℝ)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (2 * x - 1 ≠ 0) ∧ (x + 1 ≥ 0)

theorem domain_of_f :
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ -1 ∧ x ≠ 1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l490_49010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_range_l490_49091

-- Define the ellipse parameterization
noncomputable def ellipse (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 4 * Real.sin θ)

-- Define the condition for θ
def valid_θ (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Define the line
def line (x b : ℝ) : ℝ :=
  x + b

-- Define the intersection condition
def has_intersection (b : ℝ) : Prop :=
  ∀ θ, valid_θ θ → ∃ x, (x, line x b) = ellipse θ

-- State the theorem
theorem ellipse_line_intersection_range :
  ∀ b, has_intersection b ↔ b ∈ Set.Icc (-2) (2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_range_l490_49091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_adjustment_l490_49032

theorem price_adjustment (p : ℝ) (h : p > 0) : 
  p * (1 + 0.3) * (1 - 0.25) * (1 + 0.1) = p * 1.0725 :=
by 
  -- Algebraic manipulation
  calc p * (1 + 0.3) * (1 - 0.25) * (1 + 0.1)
       = p * 1.3 * 0.75 * 1.1 := by ring
  -- Simplify
  _    = p * 1.0725 := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_adjustment_l490_49032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l490_49017

/-- The distance between two points in 2D space -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Circle C₁ with equation x² + y² + 2x - 4y + 1 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1) = 0}

/-- Circle C₂ with equation (x - 3)² + (y + 1)² = 1 -/
def C₂ : Set (ℝ × ℝ) :=
  {p | ((p.1 - 3)^2 + (p.2 + 1)^2) = 1}

/-- The center of circle C₁ -/
def center_C₁ : ℝ × ℝ := (-1, 2)

/-- The center of circle C₂ -/
def center_C₂ : ℝ × ℝ := (3, -1)

/-- The radius of circle C₁ -/
def radius_C₁ : ℝ := 2

/-- The radius of circle C₂ -/
def radius_C₂ : ℝ := 1

theorem circles_are_separate : 
  distance center_C₁.1 center_C₁.2 center_C₂.1 center_C₂.2 > radius_C₁ + radius_C₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l490_49017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_schools_l490_49015

/-- Represents a student in the mathematics competition --/
structure Student where
  score : ℕ
  rank : ℕ

/-- Represents a high school team in the competition --/
structure Team where
  members : Fin 4 → Student

/-- The mathematics competition --/
structure Competition where
  schools : ℕ
  teams : Fin schools → Team
  allStudents : Fin (4 * schools) → Student
  andrea : Student
  beth : Student
  carla : Student

/-- The main theorem stating the number of schools in the competition --/
theorem number_of_schools (comp : Competition) : comp.schools = 23 :=
  by
  have andrea_highest : ∀ t : Fin comp.schools, ∀ s : Fin 4, 
    (comp.teams t).members s ≠ comp.andrea → 
    ((comp.teams t).members s).score < comp.andrea.score := sorry
  
  have andrea_median : comp.andrea.rank = (4 * comp.schools + 1) / 2 + 1 := sorry
  
  have beth_rank : comp.beth.rank = 48 := sorry
  
  have carla_rank : comp.carla.rank = 75 := sorry
  
  have andrea_rank_bound : comp.andrea.rank < comp.beth.rank := sorry
  
  sorry -- Complete the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_schools_l490_49015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_all_roots_real_smallest_c_is_minimal_l490_49083

/-- The smallest positive real number c such that all roots of x^3 - 4x^2 + cx - 4 are real -/
noncomputable def smallest_c : ℝ := 3 * Real.rpow 16 (1/3)

/-- The polynomial x^3 - 4x^2 + cx - 4 -/
def polynomial (x c : ℝ) : ℝ := x^3 - 4*x^2 + c*x - 4

theorem smallest_c_all_roots_real :
  ∀ c : ℝ, c ≥ smallest_c →
    ∃ p q r : ℝ, (∀ x : ℝ, polynomial x c = (x - p) * (x - q) * (x - r)) :=
by sorry

theorem smallest_c_is_minimal :
  ∀ c : ℝ, c > 0 → c < smallest_c →
    ¬(∃ p q r : ℝ, (∀ x : ℝ, polynomial x c = (x - p) * (x - q) * (x - r))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_all_roots_real_smallest_c_is_minimal_l490_49083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_least_diameter_half_time_l490_49031

open Real

/-- Two circular tracks with cars traveling in opposite directions -/
structure TwoTrackSystem where
  -- Radius of both tracks
  radius : ℝ
  -- Time for one full lap (in hours)
  lapTime : ℝ
  -- Initial angle of car A (in radians)
  initialAngleA : ℝ
  -- Initial angle of car B (in radians)
  initialAngleB : ℝ

/-- Position of a car on a circular track at a given time -/
noncomputable def carPosition (s : TwoTrackSystem) (t : ℝ) (isCarA : Bool) : ℝ × ℝ :=
  let angle := if isCarA then
    s.initialAngleA + 2 * π * t / s.lapTime
  else
    s.initialAngleB - 2 * π * t / s.lapTime
  (s.radius * cos angle, s.radius * sin angle)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The distance between cars is at least the diameter for half the lap time -/
theorem distance_at_least_diameter_half_time (s : TwoTrackSystem) :
  ∃ t₀ : ℝ, t₀ > 0 ∧ t₀ < s.lapTime ∧
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ t₀ →
    distance (carPosition s t true) (carPosition s t false) ≥ 2 * s.radius) ∧
  t₀ = s.lapTime / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_least_diameter_half_time_l490_49031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_with_replacement_not_independent_without_replacement_l490_49004

-- Define the sample space
def Ω : Type := Fin 4 × Fin 4

-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A and B_complement
def A : Ω → Prop := fun ω ↦ ω.1 < 3
def B_complement : Ω → Prop := fun ω ↦ ω.2 ≥ 3

-- With replacement scenario
theorem independence_with_replacement 
  (h_uniform : ∀ ω : Fin 4, P {(ω, ω') | ω' : Fin 4} = 1/4) :
  P {ω | A ω ∧ B_complement ω} = P {ω | A ω} * P {ω | B_complement ω} := by
  sorry

-- Without replacement scenario
theorem not_independent_without_replacement 
  (h_uniform : ∀ ω : Ω, ω.1 ≠ ω.2 → P {ω} = 1/12) :
  P {ω | A ω ∧ B_complement ω} ≠ P {ω | A ω} * P {ω | B_complement ω} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_with_replacement_not_independent_without_replacement_l490_49004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_quantity_is_20_l490_49021

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- The initial mixture before adding water -/
def initial_mixture (x : ℝ) : Mixture :=
  { milk := 3 * x,
    water := 2 * x }

/-- The mixture after adding 10 liters of water -/
def final_mixture (x : ℝ) : Mixture :=
  { milk := 3 * x,
    water := 2 * x + 10 }

/-- The theorem stating the initial total quantity -/
theorem initial_quantity_is_20 :
  ∀ x : ℝ,
  (initial_mixture x).milk / (initial_mixture x).water = 3 / 2 →
  (final_mixture x).milk / (final_mixture x).water = 2 / 3 →
  (initial_mixture x).milk + (initial_mixture x).water = 20 := by
  intro x h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_quantity_is_20_l490_49021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_l490_49063

theorem triangle_area_in_circle (r a b c : ℝ) (h1 : r = 4) 
  (h2 : a / b = 2 / 3 ∧ b / c = 3 / 4) (h3 : a^2 + b^2 = c^2) 
  (h4 : c = 2 * r) : 
  (1/2 : ℝ) * a * b = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_l490_49063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_k_neg_one_collinear_implies_lambda_two_magnitude_m_plus_2c_equals_sqrt_seven_l490_49071

noncomputable section

-- Define vectors a, b, c
def a (k : ℝ) : ℝ × ℝ := (Real.sqrt 3, k)
def b : ℝ × ℝ := (0, -1)
def c : ℝ × ℝ := (1, Real.sqrt 3)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Part I
theorem perpendicular_implies_k_neg_one :
  dot_product (a k) c = 0 → k = -1 := by sorry

-- Part II
theorem collinear_implies_lambda_two :
  ∃ (t : ℝ), (a 1 - 2 • b) = t • c := by sorry

-- Part III
theorem magnitude_m_plus_2c_equals_sqrt_seven :
  ∀ (m : ℝ × ℝ),
    magnitude m = Real.sqrt 3 →
    dot_product m c = -3 →
    magnitude (m + 2 • c) = Real.sqrt 7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_k_neg_one_collinear_implies_lambda_two_magnitude_m_plus_2c_equals_sqrt_seven_l490_49071
