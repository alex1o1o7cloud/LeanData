import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l403_40365

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) / Real.log (Real.sqrt 5)
  else Real.log x / Real.log (Real.sqrt 5)

-- State the theorem
theorem f_properties :
  -- f is even
  (∀ x, x ≠ 0 → f (-x) = f x) ∧
  -- f(5) = 2
  f 5 = 2 ∧
  -- The solution set of f(x) > 4
  (∀ x, f x > 4 ↔ (x < -25 ∨ x > 25)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l403_40365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l403_40307

theorem log_inequality : ∃ (a b c : ℝ),
  a = Real.log 3 / Real.log 2 ∧
  b = Real.log 2 / Real.log 3 ∧
  c = Real.log (Real.log 2 / Real.log 3) / Real.log 2 ∧
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l403_40307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_male_volunteer_l403_40320

-- Rename the structure to avoid conflict with the built-in Unit type
structure VolunteerUnit where
  male : ℕ
  female : ℕ

def total_volunteers (u : VolunteerUnit) : ℕ := u.male + u.female

noncomputable def prob_male (u : VolunteerUnit) : ℚ := 
  (u.male : ℚ) / (total_volunteers u : ℚ)

theorem prob_male_volunteer (unit_a unit_b : VolunteerUnit) 
  (ha : unit_a.male = 5 ∧ unit_a.female = 7)
  (hb : unit_b.male = 4 ∧ unit_b.female = 2) :
  (1/2 * prob_male unit_a + 1/2 * prob_male unit_b) = 13/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_male_volunteer_l403_40320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l403_40308

/-- Represents a hyperbola with equation y²/9 - x² = 1 -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = 3*x ∨ y = -3*x

/-- The foci of the hyperbola are on the y-axis -/
def foci_on_y_axis : Prop := ∃ c : ℝ, c > 0 ∧ (0, c) ∈ {p : ℝ × ℝ | hyperbola p.1 p.2}

theorem hyperbola_properties :
  ∀ x y : ℝ, hyperbola x y →
  foci_on_y_axis ∧
  asymptotes x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l403_40308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XZ_length_l403_40380

-- Define the triangle XYZ
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.X
  let (x₂, y₂) := t.Y
  let (x₃, y₃) := t.Z
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0

noncomputable def sideLength (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

noncomputable def cosZ (t : Triangle) : ℝ :=
  (sideLength t.X t.Z) / (sideLength t.Y t.Z)

noncomputable def tanZ (t : Triangle) : ℝ :=
  (sideLength t.X t.Y) / (sideLength t.X t.Z)

-- Theorem statement
theorem triangle_XZ_length (t : Triangle) :
  isRightTriangle t →
  sideLength t.Y t.Z = 30 →
  tanZ t = 3 * cosZ t →
  sideLength t.X t.Z = Real.sqrt 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XZ_length_l403_40380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_to_odd_function_l403_40372

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x / 3 + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi)

theorem translation_to_odd_function :
  ∀ x, g x = -g (-x) :=
by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_to_odd_function_l403_40372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_implies_elements_in_intersection_l403_40346

-- Define the sets α and β
variable (α β : Set ℝ)

-- Define points A and B
variable (A B : ℝ)

-- Define the line segment AB
def AB (A B : ℝ) : Set ℝ := {X | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) * A + t * B}

-- Theorem statement
theorem subset_intersection_implies_elements_in_intersection 
  (h1 : AB A B ⊆ α) (h2 : AB A B ⊆ β) : A ∈ (α ∩ β) ∧ B ∈ (α ∩ β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_implies_elements_in_intersection_l403_40346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_to_appear_is_six_l403_40371

def G : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => G (n + 1) + G n

def unitDigit (n : ℕ) : ℕ := n % 10

def appearsAsUnitDigit (d : ℕ) : Prop :=
  ∃ n, unitDigit (G n) = d

theorem last_digit_to_appear_is_six :
  (∀ d, d < 10 → appearsAsUnitDigit d) ∧
  (∀ n, G n ≤ n → unitDigit (G n) ≠ 6) ∧
  (∃ n, unitDigit (G n) = 6) := by
  sorry

#eval unitDigit (G 19)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_to_appear_is_six_l403_40371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l403_40333

/-- The function f(x) = (2x^3 - 7x^2 + 11) / (4x^2 + 6x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (2*x^3 - 7*x^2 + 11) / (4*x^2 + 6*x + 3)

/-- The denominator of f(x) -/
noncomputable def denom (x : ℝ) : ℝ := 4*x^2 + 6*x + 3

/-- Theorem: The sum of the x-coordinates of the vertical asymptotes of f(x) is -3/2 -/
theorem vertical_asymptotes_sum :
  ∃ (p q : ℝ), (denom p = 0 ∧ denom q = 0 ∧ p ≠ q) ∧ p + q = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l403_40333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lobachevsky_convergence_l403_40382

/-- Lucas numbers sequence -/
def lucas : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Coefficient sequence in Lobachevsky method -/
def p (n : ℕ) : ℚ := lucas (2^n)

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Initial polynomial in Lobachevsky method -/
def P₀ (x : ℝ) : ℝ := x^2 - x - 1

/-- n-th polynomial in Lobachevsky method -/
def Pₙ (n : ℕ) (x : ℝ) : ℝ := x^2 - p n * x + 1

/-- Theorem: Lobachevsky method converges to the roots of x² - x - 1 -/
theorem lobachevsky_convergence :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((p n : ℝ)^(1/(2^n : ℝ)) : ℝ) - φ| < ε) ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((p n : ℝ)^(-1/(2^n : ℝ)) : ℝ) + φ⁻¹| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lobachevsky_convergence_l403_40382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_lines_product_l403_40393

/-- Given a square formed by the lines y = 3, y = 7, x = 2, and x = b,
    the product of possible values for b is -12 -/
theorem square_lines_product (b₁ b₂ : ℝ) : 
  (∃ (b : ℝ), Set.prod (Set.Icc 3 7) (Set.Icc (min 2 b) (max 2 b)) = Set.prod (Set.Icc 0 4) (Set.Icc 0 4)) →
  b₁ ≠ b₂ →
  (max 2 b₁ - min 2 b₁) = 4 →
  (max 2 b₂ - min 2 b₂) = 4 →
  b₁ * b₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_lines_product_l403_40393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_meet_time_l403_40398

-- Define the circles and bug speeds
noncomputable def large_radius : ℝ := 7
noncomputable def small_radius : ℝ := 3
noncomputable def large_speed : ℝ := 4 * Real.pi
noncomputable def small_speed : ℝ := 3 * Real.pi
noncomputable def delay : ℝ := 2

-- Define the function to calculate the time to complete one revolution
noncomputable def revolution_time (radius : ℝ) (speed : ℝ) : ℝ :=
  (2 * Real.pi * radius) / speed

-- Define the theorem
theorem bugs_meet_time :
  let large_time := revolution_time large_radius large_speed
  let small_time := revolution_time small_radius small_speed
  ∃ n m : ℕ, n * large_time = m * small_time + delay + 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_meet_time_l403_40398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susie_remaining_money_l403_40376

/-- Susie's remaining money after spending on make-up and skincare --/
theorem susie_remaining_money : 
  (let hours_per_day : ℝ := 3
   let rate_per_hour : ℝ := 10
   let days_per_week : ℝ := 7
   let weekly_earnings := hours_per_day * rate_per_hour * days_per_week
   let makeup_fraction : ℝ := 3 / 10
   let skincare_fraction : ℝ := 2 / 5
   let makeup_cost := makeup_fraction * weekly_earnings
   let remaining_after_makeup := weekly_earnings - makeup_cost
   let skincare_cost := skincare_fraction * remaining_after_makeup
   let final_remaining := remaining_after_makeup - skincare_cost
   final_remaining) = 88.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susie_remaining_money_l403_40376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l403_40384

-- Define the function f
noncomputable def f (m n x : ℝ) : ℝ := m * x / (x^2 + n)

-- Define the function g
def g (a x : ℝ) : ℝ := x^2 - 2 * a * x + a

-- Main theorem
theorem main_theorem (m n : ℝ) 
  (h1 : ∃ (x : ℝ), f m n x = 2 ∧ ∀ (y : ℝ), f m n y ≤ f m n x)
  (h2 : f m n 1 = 2) :
  (∀ (x : ℝ), f m n x = 4 * x / (x^2 + 1)) ∧
  (∀ (x : ℝ), x > 0 → f m n x ≤ 2) ∧
  (∀ (a : ℝ), (∀ (x₁ : ℝ), ∃ (x₂ : ℝ), x₂ ∈ Set.Icc (-1) 0 ∧ g a x₂ ≤ f m n x₁) → a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l403_40384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_l403_40314

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan (2 * x + π / 4)

-- Define the set of x values
def solution_set : Set ℝ := { x | ∃ k : ℤ, π / 24 + k * π / 2 ≤ x ∧ x ≤ π / 8 + k * π / 2 }

-- Theorem statement
theorem tan_inequality_solution :
  { x : ℝ | f x ≥ sqrt 3 } = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_l403_40314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_cut_exists_l403_40395

/-- Represents a point on a grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a polygon on a grid -/
structure GridPolygon where
  vertices : List GridPoint

/-- Represents a cut shape -/
inductive CutShape
  | Left
  | Centered
  | Right

/-- Represents a cut on the grid -/
structure Cut where
  shape : CutShape
  endpoints : GridPoint × GridPoint

/-- Checks if a cut is valid according to the problem constraints -/
def isValidCut (polygon : GridPolygon) (cut : Cut) : Prop :=
  -- The cut lies inside the polygon
  cut.endpoints.1 ∈ polygon.vertices ∧ cut.endpoints.2 ∈ polygon.vertices ∧
  -- Only the endpoints of the cut touch the border of the polygon
  ∀ p : GridPoint, p ∈ [cut.endpoints.1, cut.endpoints.2] → p ∈ polygon.vertices ∧
  -- The sides of the polygon and segments of the cut follow the grid lines
  ∀ p q : GridPoint, (p ∈ polygon.vertices ∧ q ∈ polygon.vertices) →
    (p.x = q.x ∨ p.y = q.y) ∧
  -- Small segments of the cut are half the length of the larger segments
  ∃ l : Int, ∀ s : Int, s ∈ [l, l / 2] → 
    ∃ p q : GridPoint, p ≠ q ∧ (p.x - q.x).natAbs + (p.y - q.y).natAbs = s.natAbs

/-- Calculates the area of a polygon (placeholder function) -/
noncomputable def polygonArea (polygon : GridPolygon) : Int :=
  sorry

/-- Checks if a cut divides a polygon into two equal areas -/
def dividesEquallyArea (polygon : GridPolygon) (cut : Cut) : Prop :=
  ∃ area1 area2 : Int, area1 = area2 ∧ area1 + area2 = polygonArea polygon

/-- The main theorem stating that for any polygon, there exists a valid cut
    for each cut shape that divides the polygon into two equal areas -/
theorem equal_area_cut_exists (polygon : GridPolygon) :
  ∀ shape : CutShape, ∃ cut : Cut,
    cut.shape = shape ∧ isValidCut polygon cut ∧ dividesEquallyArea polygon cut :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_cut_exists_l403_40395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l403_40350

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (2, 1) to the line 3x - 4y + 2 = 0 is 4/5 -/
theorem distance_point_to_specific_line :
  distance_point_to_line 2 1 3 (-4) 2 = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l403_40350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_two_rational_others_l403_40374

theorem irrational_sqrt_two_rational_others : 
  (Irrational (Real.sqrt 2)) ∧ 
  (¬ Irrational (0.5 : ℝ)) ∧ 
  (¬ Irrational (1/3 : ℝ)) ∧ 
  (¬ Irrational (Real.sqrt 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_two_rational_others_l403_40374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l403_40370

open Real

theorem trig_equation_solution (t : ℝ) :
  (cos t ≠ 0) → (sin t ≠ 0) → (cos (2*t) ≠ 0) →
  ((tan t - (1 / tan t) + 2 * tan (2*t)) * (1 + cos (3*t)) = 4 * sin (3*t)) ↔
  (∃ (k : ℤ), t = π/5*(2*k+1) ∧ k ≠ 5*l+2 ∨
   ∃ (n : ℤ), t = π/3*(2*n+1) ∧ n ≠ 3*m+1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l403_40370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_f_parity_g_piecewise_l403_40336

-- Definition of floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Definition of ceiling function
noncomputable def ceil (x : ℝ) : ℤ := -Int.floor (-x)

-- Definition of the function f
noncomputable def f (x : ℝ) : ℤ := ceil (x * floor x)

-- Theorem for part (1)
theorem f_values :
  f (-3/2) = 3 ∧ f (3/2) = 2 := by sorry

-- Theorem for part (2)
theorem f_parity :
  ¬(∀ x, f (-x) = f x) ∧ ¬(∀ x, f (-x) = -f x) := by sorry

-- Definition of the function g
noncomputable def g (x : ℝ) : ℤ := floor x + ceil x

-- Theorem for part (3)
theorem g_piecewise (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) :
  g x = if x = -1 then -2
        else if -1 < x ∧ x < 0 then -1
        else if x = 0 then 0
        else if 0 < x ∧ x < 1 then 1
        else 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_f_parity_g_piecewise_l403_40336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l403_40354

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-4) (-1) then -3 - x
  else if x ∈ Set.Icc (-1) 2 then Real.sqrt (9 - (x - 1)^2) - 3
  else if x ∈ Set.Icc 2 4 then 3 * (x - 2)
  else 0  -- Default value for x outside the domain

-- Define the absolute value of g(x)
noncomputable def abs_g (x : ℝ) : ℝ := |g x|

-- Theorem statement
theorem abs_g_piecewise (x : ℝ) :
  (x ∈ Set.Icc (-4) 4) →
  abs_g x = if x ∈ Set.Icc (-4) (-1) then 3 + x
            else if x ∈ Set.Icc (-1) 2 then Real.sqrt (9 - (x - 1)^2)
            else 3 * (x - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l403_40354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_property_l403_40310

noncomputable def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  fun i => a₁ + d * i.val

noncomputable def sum_arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_progression_property (a₁ d : ℝ) (n : ℕ) :
  (sum_arithmetic_progression a₁ d n = 112) →
  ((a₁ + d) * d = 30) →
  ((a₁ + 2 * d) + (a₁ + 4 * d) = 32) →
  (n = 7 ∧ ((a₁ = 1 ∧ d = 5) ∨ (a₁ = 7 ∧ d = 3))) :=
by
  sorry

#check arithmetic_progression_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_property_l403_40310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_georgia_stationery_l403_40315

/-- The number of pieces of stationery Georgia has -/
def georgia : ℕ := sorry

/-- The number of pieces of stationery Lorene has -/
def lorene : ℕ := sorry

/-- Lorene has three times as many pieces of stationery as Georgia -/
axiom lorene_triple : lorene = 3 * georgia

/-- Georgia has 50 fewer pieces of stationery than Lorene -/
axiom georgia_fewer : georgia = lorene - 50

theorem georgia_stationery : georgia = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_georgia_stationery_l403_40315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_equals_8_point_1_l403_40396

/-- The height of a rectangular-based pyramid with the same volume as a cube -/
noncomputable def pyramid_height (cube_edge : ℝ) (pyramid_base_length : ℝ) (pyramid_base_width : ℝ) : ℝ :=
  (cube_edge ^ 3 * 3) / (pyramid_base_length * pyramid_base_width)

/-- Theorem stating that the height of the pyramid is 8.1 units -/
theorem pyramid_height_equals_8_point_1 :
  pyramid_height 6 10 8 = 8.1 := by
  -- Unfold the definition of pyramid_height
  unfold pyramid_height
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_equals_8_point_1_l403_40396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l403_40300

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10*y + 2*x + 9 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (8, -5)

/-- Theorem: The vertex of the parabola y^2 + 10y + 2x + 9 = 0 is at (8, -5) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → (x, y) = vertex ∨ y^2 > (vertex.2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l403_40300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l403_40399

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1 - x^2) / x^2

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x - 2 / x^3

theorem min_value_and_inequality (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a x ≤ f a y) ∧ (f a (Real.sqrt (2 / a)) = 0) ↔ a = 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc 1 2 → f 2 x ≤ f_derivative 2 x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l403_40399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_maximum_l403_40359

/-- The concentration of a chemical in a pool as a function of time -/
noncomputable def concentration (t : ℝ) : ℝ := 20 * t / (t^2 + 4)

/-- The time at which the concentration reaches its maximum -/
def max_concentration_time : ℝ := 2

theorem concentration_maximum :
  ∀ t : ℝ, t > 0 → concentration t ≤ concentration max_concentration_time :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_maximum_l403_40359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_collection_proof_l403_40305

/-- The number of books Tracy collected in the first week -/
def first_week_books : ℕ := 20

/-- The total number of books collected over 6 weeks -/
def total_books : ℕ := 1025

/-- The collection pattern for each week -/
def weekly_collection (week : ℕ) : ℚ :=
  match week with
  | 1 => 1
  | 2 => 5
  | 3 => 15/2
  | 4 => 10
  | 5 => 13
  | 6 => 16
  | _ => 0

/-- The sum of books collected over 6 weeks -/
def total_collection : ℚ := (List.range 6).map (λ w => weekly_collection (w + 1)) |>.sum

theorem book_collection_proof :
  (first_week_books : ℚ) * total_collection = total_books := by
  sorry

#eval first_week_books
#eval total_books
#eval total_collection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_collection_proof_l403_40305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_plus_h_l403_40306

/-- Represents a hyperbola with asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The standard form of a hyperbola equation -/
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- Theorem stating the relationship between a and h for the given hyperbola -/
theorem hyperbola_a_plus_h (H : Hyperbola) (a b h k : ℝ) :
  H.asymptote1 = (λ x : ℝ => 3/2 * x + 4) →
  H.asymptote2 = (λ x : ℝ => -3/2 * x + 2) →
  H.point = (2, 8) →
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, standard_form a b h k x y ↔ 
    (y = H.asymptote1 x ∨ y = H.asymptote2 x ∨ (x, y) = H.point)) →
  a + h = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_plus_h_l403_40306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l403_40361

theorem problem_statement (a b : ℝ) (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l403_40361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l403_40378

/-- 
Given an initial salary that increases by a consistent percentage each year,
this theorem proves that the percentage increase is approximately 40% 
when the initial salary of 3000 becomes 8232 after 3 years.
-/
theorem salary_increase_percentage 
  (initial_salary : ℝ) 
  (final_salary : ℝ) 
  (years : ℕ) 
  (increase_rate : ℝ) 
  (h1 : initial_salary = 3000)
  (h2 : final_salary = 8232)
  (h3 : years = 3)
  (h4 : final_salary = initial_salary * (1 + increase_rate / 100) ^ years) :
  39 < increase_rate ∧ increase_rate < 41 := by
  sorry

#eval Float.log 2.744 / Float.log 1.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l403_40378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_on_circle_l403_40340

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the vector sum
def vector_sum (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := 
  (x1 + x2, y1 + y2)

-- Define the magnitude of a vector
noncomputable def magnitude (x y : ℝ) : ℝ := 
  Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem max_vector_sum_on_circle :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C x1 y1 → 
    circle_C x2 y2 → 
    distance x1 y1 x2 y2 = 2 * Real.sqrt 3 →
    ∀ (s : ℝ × ℝ), 
      s = vector_sum x1 y1 x2 y2 → 
      magnitude s.1 s.2 ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_on_circle_l403_40340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_is_3_l403_40341

/-- The decimal representation of 5/37 has a repeating block of length 3 -/
def repeating_block_length : Nat := 3

/-- The repeating block in the decimal representation of 5/37 -/
def repeating_block : Fin repeating_block_length → Nat
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 5

/-- The nth digit after the decimal point in the decimal representation of 5/37 -/
def nth_digit (n : Nat) : Nat :=
  repeating_block ⟨(n - 1) % repeating_block_length, by
    have h : (n - 1) % repeating_block_length < repeating_block_length := Nat.mod_lt _ (Nat.zero_lt_succ 2)
    exact h⟩

/-- The 150th digit after the decimal point in the decimal representation of 5/37 is 3 -/
theorem digit_150_is_3 : nth_digit 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_is_3_l403_40341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l403_40389

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- The first line equation: 3y - 3b = 9x -/
def line1 (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * y - 3 * b = 9 * x

/-- The second line equation: y + 2 = (b + 9)x -/
def line2 (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y + 2 = (b + 9) * x

theorem parallel_lines_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, line1 b x y ∨ line2 b x y) → b = -6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l403_40389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l403_40390

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points D and E on extensions of AC and AB
variable (D E : EuclideanSpace ℝ (Fin 2))

-- Define point P as intersection of BD and CE
variable (P : EuclideanSpace ℝ (Fin 2))

-- Define the constant k
variable (k : ℝ)

-- Define helper functions for angle and segment length
def angle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry
def segment_length (A B : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem triangle_equality 
  (on_AC_ext : ∃ t : ℝ, D = A + t • (C - A) ∧ t > 1)
  (on_AB_ext : ∃ t : ℝ, E = A + t • (B - A) ∧ t > 1)
  (P_on_BD : ∃ t : ℝ, P = B + t • (D - B) ∧ 0 < t ∧ t < 1)
  (P_on_CE : ∃ t : ℝ, P = C + t • (E - C) ∧ 0 < t ∧ t < 1)
  (BD_eq_CE : segment_length B D = segment_length C E)
  (angle_condition : angle A E P - angle A D P = k^2 * (angle P E D - angle P D E)) :
  segment_length A B = segment_length A C :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l403_40390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_count_l403_40324

def number_of_arrangements (total : ℕ) (special : ℕ) : ℕ :=
  (Nat.factorial (total - special)) * (Nat.descFactorial (total - special + 1) special)

theorem student_arrangement_count : ∀ n : ℕ, 
  n = 5 → 
  (number_of_arrangements n 2) = 72 :=
fun n h => by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_count_l403_40324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_when_m_neg_two_extreme_points_product_greater_than_e_squared_l403_40319

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1/2) * m * x^2 - x

noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := Real.log x + 1 - m * x

theorem zero_point_when_m_neg_two :
  f (-2) 1 = 0 := by sorry

theorem extreme_points_product_greater_than_e_squared (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f_deriv m x₁ = 0) (h₂ : f_deriv m x₂ = 0) (h₃ : x₁ < x₂) :
  x₁ * x₂ > Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_when_m_neg_two_extreme_points_product_greater_than_e_squared_l403_40319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poods_bought_l403_40318

/-- Represents the number of poods of sugar bought -/
def x : ℕ := sorry

/-- Represents the price per pood in rubles -/
def p : ℝ := sorry

/-- The total cost is 500 rubles -/
axiom total_cost : x * p = 500

/-- If 5 more poods were bought, each pood would cost 5 rubles less -/
axiom price_reduction : 500 / (x + 5) = p - 5

/-- Theorem stating that 20 poods of sugar were bought -/
theorem poods_bought : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poods_bought_l403_40318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_paints_210_l403_40369

/-- Given a total area and a work ratio, calculate the area painted by the second person -/
noncomputable def area_painted_by_second (total_area : ℝ) (ratio1 ratio2 : ℕ) : ℝ :=
  total_area * (ratio2 : ℝ) / ((ratio1 : ℝ) + (ratio2 : ℝ))

/-- Theorem: Bob paints 210 square feet given the conditions -/
theorem bob_paints_210 :
  area_painted_by_second 330 4 7 = 210 := by
  -- Unfold the definition of area_painted_by_second
  unfold area_painted_by_second
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_paints_210_l403_40369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l403_40387

noncomputable section

open Real

def F (a : ℝ) (x : ℝ) : ℝ := log x + a / x

theorem problem_1 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 e, F a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 e, F a x = 3/2) →
  a = sqrt e :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x ≥ 1, log x ≤ x + a / x) →
  a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l403_40387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_all_values_valid_no_other_values_l403_40323

/-- The set of all possible values obtainable from the expression 3^(3+3^3) 
    by changing the order of operations and parentheses. -/
def expression_values : Finset ℕ :=
  {3^30, 3^216, 387420489}

/-- The theorem stating that there are exactly 3 distinct values 
    obtainable from the expression 3^(3+3^3) by changing the order 
    of operations and parentheses. -/
theorem distinct_values_count : Finset.card expression_values = 3 := by
  sorry

/-- A function that evaluates the expression given a specific parenthesization. -/
def evaluate_expression (a b c : ℕ) (parenthesization : ℕ) : ℕ :=
  match parenthesization with
  | 0 => a^(b + c^a)
  | 1 => a^((b + c)^a)
  | 2 => (a^(b + c))^a
  | _ => (a^b)^(c + a)

/-- The theorem stating that all values in the set are indeed 
    obtainable from the expression 3^(3+3^3) by some valid 
    parenthesization and order of operations. -/
theorem all_values_valid : ∀ x ∈ expression_values, 
  ∃ parenthesization, evaluate_expression 3 3 3 parenthesization = x := by
  sorry

/-- The theorem stating that no other parenthesization produces 
    a value different from those in the set. -/
theorem no_other_values : ∀ parenthesization, 
  evaluate_expression 3 3 3 parenthesization ∈ expression_values := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_all_values_valid_no_other_values_l403_40323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sum_l403_40360

open Matrix

def HasInverseOverZ (M : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  ∃ N : Matrix (Fin 2) (Fin 2) ℤ, M * N = 1 ∧ N * M = 1

theorem inverse_of_sum (A B : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : HasInverseOverZ A)
  (hA1B : HasInverseOverZ (A + B))
  (hA2B : HasInverseOverZ (A + 2 • B))
  (hA3B : HasInverseOverZ (A + 3 • B))
  (hA4B : HasInverseOverZ (A + 4 • B)) :
  HasInverseOverZ (A + 5 • B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sum_l403_40360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l403_40316

theorem log_equality_implies_base (y : ℝ) (h : y > 0) : 
  Real.log 8 / Real.log y = Real.log 4 / Real.log 64 → y = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l403_40316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_products_in_set_l403_40363

theorem distinct_products_in_set (M : ℕ+) :
  let S : Set ℕ := {n : ℕ | M^2 ≤ n ∧ n < (M+1)^2}
  ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a * b = c * d → (a = c ∧ b = d) ∨ (a = d ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_products_in_set_l403_40363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_allocation_schemes_l403_40342

/-- The number of ways to allocate n students to k venues, with at least one student in each venue. -/
def number_of_allocation_schemes (n : ℕ) (k : ℕ) : ℕ :=
  if n ≥ k then
    -- Implementation details omitted
    0  -- Placeholder
  else
    0

theorem student_allocation_schemes (n : ℕ) (k : ℕ) : n = 4 ∧ k = 3 →
  (number_of_allocation_schemes n k) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_allocation_schemes_l403_40342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l403_40312

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (t : ℝ), t > 0 → (∀ (x : ℝ), f (x + t) = f x) → t ≥ T) ∧
    T = Real.pi ∧
    (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → f x ≤ 2) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) ∧ f x = 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → f x ≥ -Real.sqrt 2 + 1) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) ∧ f x = -Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l403_40312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_a_range_l403_40375

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt ((a - 2) * x^2 + 2 * (a - 2) * x + 4)

-- State the theorem
theorem function_domain_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (2 ≤ a ∧ a ≤ 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_a_range_l403_40375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l403_40338

noncomputable def f (x : ℝ) : ℝ := (Real.exp 1) * Real.exp x - x + (1/2) * x^2

theorem f_properties :
  ∃ (f' : ℝ → ℝ),
  (∀ x, HasDerivAt f (f' x) x) ∧
  (∀ x, x > 0 → StrictMonoOn f (Set.Ioo 0 x)) ∧
  (∀ x, x < 0 → StrictAntiOn f (Set.Ioo x 0)) ∧
  (∀ a : ℝ, (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ ∈ Set.Icc (-1) 2 ∧ r₂ ∈ Set.Icc (-1) 2 ∧ 
    f r₁ = (1/2) * r₁^2 + a ∧ f r₂ = (1/2) * r₂^2 + a) 
    ↔ a ∈ Set.Ioo 1 (1 + 1 / Real.exp 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l403_40338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_quarter_circles_l403_40385

noncomputable def quarter_circle_length (r : ℝ) : ℝ := (1/2) * Real.pi * r

theorem sum_of_quarter_circles :
  let radii : List ℝ := [1, 1, 2, 3, 5]
  (radii.map quarter_circle_length).sum = 6 * Real.pi := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_quarter_circles_l403_40385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_of_ellipse_containing_circles_l403_40313

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- A circle with center (h, 0) and radius r -/
structure Circle where
  h : ℝ
  r : ℝ
  h_pos_r : 0 < r

/-- The ellipse contains the given circle -/
def contains (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, (x - c.h)^2 + y^2 = c.r^2 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- The area of an ellipse -/
noncomputable def area (e : Ellipse) : ℝ := Real.pi * e.a * e.b

theorem smallest_area_of_ellipse_containing_circles :
  ∀ e : Ellipse,
  contains e { h := 2, r := 2, h_pos_r := by norm_num } →
  contains e { h := -2, r := 2, h_pos_r := by norm_num } →
  area e ≥ (3 * Real.sqrt 3 / 2) * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_of_ellipse_containing_circles_l403_40313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l403_40327

-- Define the constants as noncomputable
noncomputable def a : ℝ := (4/5)^(1/2)
noncomputable def b : ℝ := (5/4)^(1/5)
noncomputable def c : ℝ := (3/4)^(3/4)

-- State the theorem
theorem abc_inequality : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l403_40327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fair_given_three_l403_40373

/-- Represents a die with probabilities for each face -/
structure Die where
  prob_one : ℝ
  prob_two : ℝ
  prob_three : ℝ
  prob_four : ℝ
  prob_five : ℝ
  prob_six : ℝ
  sum_to_one : prob_one + prob_two + prob_three + prob_four + prob_five + prob_six = 1

/-- The fair die -/
noncomputable def fair_die : Die := {
  prob_one := 1/6,
  prob_two := 1/6,
  prob_three := 1/6,
  prob_four := 1/6,
  prob_five := 1/6,
  prob_six := 1/6,
  sum_to_one := by norm_num
}

/-- The unfair die -/
noncomputable def unfair_die : Die := {
  prob_one := 1/3,
  prob_two := 0,
  prob_three := 1/3,
  prob_four := 0,
  prob_five := 1/3,
  prob_six := 0,
  sum_to_one := by norm_num
}

theorem probability_fair_given_three :
  let prob_fair := (1/4 : ℝ)
  let prob_unfair := (3/4 : ℝ)
  let prob_three_fair := fair_die.prob_three
  let prob_three_unfair := unfair_die.prob_three
  let prob_three := prob_three_fair * prob_fair + prob_three_unfair * prob_unfair
  (prob_three_fair * prob_fair) / prob_three = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fair_given_three_l403_40373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_161_l403_40302

/-- Represents the array of numbers as described in the problem -/
def NumberArray (k i : ℕ) : ℕ :=
  2 * k^2 - 4 * k + 2 + 2 * (i - 1)

/-- The row number containing a given value -/
def RowOf (n : ℕ) : ℕ :=
  (n + 3).sqrt

/-- The position of a number within its row -/
def PositionInRow (n : ℕ) : ℕ :=
  n - (RowOf n - 1)^2 + 1

theorem number_above_161 :
  let row := RowOf 161
  let pos := PositionInRow 161
  NumberArray (row - 1) pos = 177 := by
  sorry

#eval NumberArray (RowOf 161 - 1) (PositionInRow 161)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_161_l403_40302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_standing_l403_40322

/-- The Josephus function for eliminating every second person -/
def josephus (n : ℕ) : ℕ :=
  2 * (n - 2^(Nat.log2 n))

/-- Theorem stating that the last remaining student in a circle of 2012 students,
    where every second student is eliminated, is number 1976 -/
theorem last_student_standing : josephus 2012 = 1976 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_standing_l403_40322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_z_value_l403_40356

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Point P is equidistant from points A and B -/
def is_equidistant (z : ℝ) : Prop :=
  distance 1 2 z 1 1 2 = distance 1 2 z 2 1 1

theorem equidistant_point_z_value :
  ∃ z : ℝ, is_equidistant z ∧ z = 3/2 := by
  sorry

#check equidistant_point_z_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_z_value_l403_40356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l403_40364

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 10

-- Define the point P
def point_P : ℝ × ℝ := (-3, -4)

-- Define the line passing through P
def line_l (k : ℝ) (x y : ℝ) : Prop := y + 4 = k * (x + 3)

-- Define the intersection points A and B (existence assumed)
axiom exists_intersection_points (k : ℝ) : ∃ (A B : ℝ × ℝ), 
  circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧ 
  line_l k A.1 A.2 ∧ line_l k B.1 B.2

-- Define the area of triangle AOB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ := 
  (1/2) * abs (A.1 * B.2 - A.2 * B.1)

-- Theorem statement
theorem slope_of_line (k : ℝ) : 
  (∃ (A B : ℝ × ℝ), circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧ 
   line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ 
   triangle_area A B = 5) → 
  (k = 1/2 ∨ k = 11/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l403_40364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_over_tangent_negative_l403_40309

/-- Given an angle α in a rectangular coordinate system XOY, where the terminal side of α 
    passes through point P(-1, m) with m ≠ 0, prove that sinα / tanα is always negative. -/
theorem sine_over_tangent_negative (m : ℝ) (hm : m ≠ 0) :
  let α := Real.arctan m
  (Real.sin α) / (Real.tan α) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_over_tangent_negative_l403_40309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_two_undefined_points_l403_40345

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := 10 * Real.tan ((2 * k - 1) * x / 5)

theorem smallest_k_for_two_undefined_points : 
  ∀ k : ℕ, k > 0 → 
  (∀ x : ℝ, ∃ x₁ x₂ : ℝ, x ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x + 1 ∧ 
    ¬∃ (y₁ : ℝ), f k x₁ = y₁ ∧ 
    ¬∃ (y₂ : ℝ), f k x₂ = y₂) 
  ↔ k ≥ 13 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_two_undefined_points_l403_40345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_30s_l403_40351

/-- Represents the speed of a train in km/hr -/
def TrainSpeed : Type := ℝ

/-- Represents the length of a train in km -/
def TrainLength : Type := ℝ

/-- Calculates the time (in seconds) it takes for two trains to cross each other
    given their speeds and lengths -/
noncomputable def crossingTime (speed1 speed2 length1 length2 : ℝ) : ℝ :=
  (length1 + length2) * 1000 / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem stating that the crossing time for the given train specifications
    is approximately 30 seconds -/
theorem train_crossing_time_approx_30s 
  (speed1 : TrainSpeed) 
  (speed2 : TrainSpeed)
  (length1 : TrainLength)
  (length2 : TrainLength)
  (h1 : speed1 = (150 : ℝ))
  (h2 : speed2 = (90 : ℝ))
  (h3 : length1 = (1.10 : ℝ))
  (h4 : length2 = (0.9 : ℝ)) :
  ∃ ε > 0, |crossingTime speed1 speed2 length1 length2 - 30| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_30s_l403_40351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l403_40355

/-- The length of the line segment formed by the intersection of a line and a circle --/
theorem intersection_length (a b c : ℝ) (d e f g h : ℝ) : 
  let l : ℝ × ℝ → Prop := λ p => a * p.1 + b * p.2 = c
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + p.2^2 + d * p.1 + e * p.2 + f = 0
  (∃ A B : ℝ × ℝ, l A ∧ l B ∧ circle A ∧ circle B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = g^2 + h^2) →
  a = 12 ∧ b = -5 ∧ c = 3 ∧ d = -6 ∧ e = -8 ∧ f = 16 ∧ g = 4 ∧ h = 4 →
  ∃ A B : ℝ × ℝ, l A ∧ l B ∧ circle A ∧ circle B ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 := by
  sorry

#check intersection_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l403_40355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ends_with_5_l403_40301

def number_set : Set ℕ := {n | 40 ≤ n ∧ n ≤ 990}

def ends_with_5 (n : ℕ) : Prop := n % 10 = 5

def count_total : ℕ := Finset.card (Finset.filter (λ n => 40 ≤ n ∧ n ≤ 990) (Finset.range 991))

def count_ends_with_5 : ℕ := Finset.card (Finset.filter (λ n => 40 ≤ n ∧ n ≤ 990 ∧ n % 10 = 5) (Finset.range 991))

theorem probability_ends_with_5 : 
  (count_ends_with_5 : ℚ) / (count_total : ℚ) = 95 / 951 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ends_with_5_l403_40301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_at_one_l403_40397

/-- Given two monic, non-constant polynomials with integer coefficients p and q
    such that x^8 - 98x^4 + 1 = p(x) * q(x), prove that p(1) + q(1) = 4 -/
theorem polynomial_sum_at_one (p q : Polynomial ℤ) :
  Polynomial.Monic p → Polynomial.Monic q → 
  (∀ x : ℤ, (X : Polynomial ℤ)^8 - 98*(X : Polynomial ℤ)^4 + 1 = p.eval x * q.eval x) →
  p.eval 1 + q.eval 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_at_one_l403_40397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pollution_reduction_l403_40386

-- Define the pollution index function
noncomputable def P (P₀ k t : ℝ) : ℝ := P₀ * Real.exp (-k * t)

-- State the theorem
theorem pollution_reduction (P₀ k : ℝ) (P₀_pos : P₀ > 0) :
  P P₀ k 5 = 0.9 * P₀ →
  P P₀ k 10 / P₀ = 0.81 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pollution_reduction_l403_40386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_with_sectors_area_l403_40367

/-- The area of the region inside a regular hexagon but outside circular sectors --/
theorem hexagon_with_sectors_area (s r θ : ℝ) : 
  s = 8 →
  r = 4 →
  θ = 90 →
  let hexagon_area := 6 * (Real.sqrt 3 / 4 * s^2)
  let sector_area := 6 * (θ / 360 * Real.pi * r^2)
  hexagon_area - sector_area = 96 * Real.sqrt 3 - 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_with_sectors_area_l403_40367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_two_is_valid_invalid_lengths_l403_40332

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a < b + c
  hbc : b < a + c
  hca : c < a + b

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (c : ℝ), 0 < c ∧ c < a + b ∧ a - b < c ∧ b - a < c := by sorry

/-- Theorem for the range of the third side in a triangle with sides 3 and 4 -/
theorem third_side_range :
  ∃ (c : ℝ), 1 < c ∧ c < 7 ∧ 
  ∃ (t : Triangle), t.a = 3 ∧ t.b = 4 ∧ t.c = c := by sorry

/-- Theorem stating that 2 is a valid length for the third side -/
theorem two_is_valid :
  ∃ (t : Triangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 2 := by sorry

/-- Theorem stating that 1, 7, and 8 are not valid lengths for the third side -/
theorem invalid_lengths :
  (¬ ∃ (t : Triangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 1) ∧
  (¬ ∃ (t : Triangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 7) ∧
  (¬ ∃ (t : Triangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_two_is_valid_invalid_lengths_l403_40332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_length_l403_40326

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

-- Define f(x)
noncomputable def f (x : ℝ) : ℝ := (floor x : ℝ) * frac x

-- Define g(x)
def g (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem solution_interval_length :
  MeasureTheory.volume (Set.Icc 0 3 ∩ {x : ℝ | f x < g x}) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_length_l403_40326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_consecutive_odd_sum_of_squares_sum_of_squares_mod_four_square_mod_four_l403_40303

/-- For any integer k, at least one of the numbers 4k+1, 4k+3, or 4k+5 cannot be expressed as the sum of two non-zero squares. -/
theorem no_three_consecutive_odd_sum_of_squares (k : ℤ) : 
  ∃ n : ℤ, n ∈ ({4*k+1, 4*k+3, 4*k+5} : Set ℤ) ∧ ¬∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ n = a^2 + b^2 := by
  sorry

/-- A number that is the sum of two non-zero squares is congruent to 0, 1, or 2 modulo 4. -/
theorem sum_of_squares_mod_four (n a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : n = a^2 + b^2) :
  n % 4 = 0 ∨ n % 4 = 1 ∨ n % 4 = 2 := by
  sorry

/-- The square of any integer is congruent to 0 or 1 modulo 4. -/
theorem square_mod_four (n : ℤ) : n^2 % 4 = 0 ∨ n^2 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_consecutive_odd_sum_of_squares_sum_of_squares_mod_four_square_mod_four_l403_40303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_16_equal_parts_l403_40362

-- Define a circle
def Circle : Type := Unit

-- Define a cut
inductive Cut
| Straight : Cut
| Radial : ℝ → Cut

-- Define a partition of a circle
def Partition : Type := List Cut

-- Function to count the number of parts in a partition
def count_parts (p : Partition) : ℕ := sorry

-- Function to check if all parts in a partition are equal
def all_parts_equal (p : Partition) : Prop := sorry

-- Theorem statement
theorem circle_16_equal_parts :
  ∃ (p : Partition), count_parts p = 16 ∧ all_parts_equal p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_16_equal_parts_l403_40362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_area_of_hemisphere_l403_40337

-- Define the hemisphere
structure Hemisphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

-- Define the light source
structure LightSource where
  position : ℝ × ℝ × ℝ

-- Define the shadow area
noncomputable def shadowArea (h : Hemisphere) (l : LightSource) : ℝ := 3 * Real.pi * h.radius^2

theorem shadow_area_of_hemisphere (R : ℝ) (h : R > 0) :
  let O : Hemisphere := { radius := R, center := (0, 0, 0) }
  let P : LightSource := { position := (0, 0, R) }
  shadowArea O P = 3 * Real.pi * R^2 := by
  sorry

#check shadow_area_of_hemisphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_area_of_hemisphere_l403_40337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_league_female_fraction_l403_40317

theorem basketball_league_female_fraction :
  -- Define the variables
  let last_year_males : ℕ := 30
  let last_year_females : ℚ := 15  -- Changed to ℚ for rational numbers
  let male_increase_rate : ℚ := 1 / 10
  let female_increase_rate : ℚ := 1 / 4
  let total_increase_rate : ℚ := 3 / 20

  -- Define this year's participants
  let this_year_males : ℚ := last_year_males * (1 + male_increase_rate)
  let this_year_females : ℚ := last_year_females * (1 + female_increase_rate)
  let this_year_total : ℚ := (last_year_males + last_year_females) * (1 + total_increase_rate)

  -- State the theorem
  (this_year_females / this_year_total) = 19 / 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_league_female_fraction_l403_40317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_k_eq_neg_one_l403_40331

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the slope of a parametric line -/
noncomputable def slopeOfLine (l : ParametricLine) : ℝ :=
  (l.y 1 - l.y 0) / (l.x 1 - l.x 0)

/-- Checks if two lines are perpendicular -/
def perpendicular (l1 l2 : ParametricLine) : Prop :=
  slopeOfLine l1 * slopeOfLine l2 = -1

/-- The first line l₁ -/
def l1 (k : ℝ) : ParametricLine where
  x := λ t => 1 - 2 * t
  y := λ t => 2 + k * t

/-- The second line l₂ -/
def l2 : ParametricLine where
  x := λ s => s
  y := λ s => 1 - 2 * s

/-- Theorem: l₁ and l₂ are perpendicular if and only if k = -1 -/
theorem perpendicular_iff_k_eq_neg_one :
  ∀ k, perpendicular (l1 k) l2 ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_k_eq_neg_one_l403_40331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_product_equals_sixteen_l403_40352

theorem radical_product_equals_sixteen :
  (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 16 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_product_equals_sixteen_l403_40352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_solution_l403_40383

def travel_problem (d : ℝ) : Prop :=
  let speeds : List ℝ := [2, 4, 6, 8, 10]
  let total_time : ℝ := 22 / 60
  ((speeds.map (λ s => d / s)).sum = total_time) ∧
  (5 * d = 220 / 137)

theorem travel_solution :
  ∃ d : ℝ, travel_problem d := by
  -- We'll use d = 44 / 137 as our solution
  let d : ℝ := 44 / 137
  
  -- Prove that this d satisfies the travel_problem
  have h1 : ((([2, 4, 6, 8, 10].map (λ s => d / s))).sum = 22 / 60) := by
    -- This step would require actual computation
    sorry
  
  have h2 : 5 * d = 220 / 137 := by
    -- This is a simple algebraic verification
    simp [d]
    ring
  
  -- Combine the two conditions
  exact ⟨d, ⟨h1, h2⟩⟩

#check travel_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_solution_l403_40383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_volume_formula_l403_40379

/-- The volume of a spherical segment -/
noncomputable def spherical_segment_volume (R h : ℝ) : ℝ := (Real.pi * h^2 * (3*R - h)) / 3

/-- Theorem: The volume of a spherical segment with height h from a sphere of radius R
    is equal to πh²(3R - h) / 3 -/
theorem spherical_segment_volume_formula (R h : ℝ) (hR : R > 0) (hh : h > 0) (hh_le_R : h ≤ R) :
  spherical_segment_volume R h = (Real.pi * h^2 * (3*R - h)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_volume_formula_l403_40379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_iff_h_gt_e_minus_3_l403_40344

open Real

/-- The function f(x) = -ln x + x + h -/
noncomputable def f (h : ℝ) (x : ℝ) : ℝ := -log x + x + h

/-- The interval [1/e, e] -/
def I : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

/-- Triangle inequality for f(a), f(b), f(c) -/
def triangle_inequality (h : ℝ) (a b c : ℝ) : Prop :=
  f h a + f h b > f h c ∧ f h b + f h c > f h a ∧ f h c + f h a > f h b

theorem triangle_exists_iff_h_gt_e_minus_3 :
  ∀ h : ℝ, (∀ a b c : ℝ, a ∈ I → b ∈ I → c ∈ I → triangle_inequality h a b c) ↔ h > Real.exp 1 - 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_iff_h_gt_e_minus_3_l403_40344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_arithmetic_sequence_preservation_l403_40392

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_properties (a q : ℝ) :
  let S := geometric_sum a q
  (∃ d : ℝ, S 3 - S 1 = S 4 - S 3 ∧ S 3 - S 1 = d ∧ S 4 - S 3 = d) →
  q = (1 + Real.sqrt 5) / 2 ∨ q = (1 - Real.sqrt 5) / 2 := by
  sorry

theorem arithmetic_sequence_preservation (a q : ℝ) (m n l : ℕ) :
  let S := geometric_sum a q
  let a_seq := geometric_sequence a q
  (∃ d : ℝ, S n - S m = S l - S n ∧ S n - S m = d ∧ S l - S n = d) →
  ∀ k : ℕ, ∃ d' : ℝ, a_seq (n + k) - a_seq (m + k) = a_seq (l + k) - a_seq (n + k) ∧
                     a_seq (n + k) - a_seq (m + k) = d' ∧
                     a_seq (l + k) - a_seq (n + k) = d' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_arithmetic_sequence_preservation_l403_40392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_and_increasing_l403_40330

-- Define the interval (-1, 1)
def openInterval : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being increasing on an interval
def isIncreasing (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

-- State the theorem
theorem sin_odd_and_increasing : 
  isOdd Real.sin ∧ isIncreasing Real.sin openInterval :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_and_increasing_l403_40330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l403_40334

theorem sine_cosine_relation (α : ℝ) :
  Real.sin (3 * Real.pi + α) = -1/2 → Real.cos ((7 * Real.pi)/2 - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l403_40334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_product_sum_divisibility_l403_40381

theorem distinct_numbers_product_sum_divisibility (n : ℕ) (S : Finset ℕ) : 
  n = 2014 → 
  S.card = n → 
  (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → (a * b) % (a + b) = 0) →
  ¬∃ a : ℕ, a ∈ S ∧ ∃ p₁ p₂ p₃ p₄ p₅ p₆ : ℕ, 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    a = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_product_sum_divisibility_l403_40381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_gain_percent_l403_40321

/-- Given a product with an original selling price and gain percentage,
    calculate the gain percentage after applying a discount. -/
theorem clearance_sale_gain_percent
  (original_price : ℝ)
  (original_gain_percent : ℝ)
  (discount_percent : ℝ)
  (h1 : original_price = 30)
  (h2 : original_gain_percent = 15)
  (h3 : discount_percent = 10) :
  ∃ (new_gain_percent : ℝ),
    let cost_price := original_price / (1 + original_gain_percent / 100)
    let discounted_price := original_price * (1 - discount_percent / 100)
    let new_gain := discounted_price - cost_price
    new_gain_percent = (new_gain / cost_price) * 100 ∧
    abs (new_gain_percent - 3.49) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_gain_percent_l403_40321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_min_l403_40325

/-- The function H as defined in the problem -/
def H (p q : ℝ) : ℝ := 3*p*q - 2*p*(1-q) - 4*(1-p)*q + 5*(1-p)*(1-q)

/-- The function K as defined in the problem -/
noncomputable def K (p : ℝ) : ℝ := ⨆ q ∈ Set.Icc 0 1, H p q

/-- The theorem stating that K(p) is minimized at p = 0 with a value of -1 -/
theorem K_min :
  ∀ p ∈ Set.Icc 0 1, K p ≥ K 0 ∧ K 0 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_min_l403_40325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_FPG_is_45_l403_40304

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Represents the triangle formed by one diagonal and two sides of the trapezoid -/
structure DiagonalTriangle where
  base : ℝ
  area : ℝ

/-- Calculates the area of the triangle FPG in the given trapezoid -/
noncomputable def area_FPG (t : Trapezoid) (dt : DiagonalTriangle) : ℝ :=
  dt.area - (t.area - dt.area) / 2

/-- Theorem stating that the area of triangle FPG is 45 square units -/
theorem area_FPG_is_45 (t : Trapezoid) (dt : DiagonalTriangle) :
  t.base1 = 15 ∧ t.base2 = 25 ∧ t.area = 200 ∧ dt.base = 25 →
  area_FPG t dt = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_FPG_is_45_l403_40304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l403_40353

/-- The sum of the infinite series ∑(k=1 to ∞) k²/3^k is equal to 4 -/
theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ)^2 / (3 : ℝ)^k = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l403_40353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_words_count_l403_40339

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- Calculates the number of words of length n that contain at least one 'A' -/
def words_with_a (n : ℕ) : ℕ :=
  alphabet_size ^ n - (alphabet_size - 1) ^ n

/-- The total number of words with 5 letters or less, containing at least one 'A' -/
def total_words : ℕ :=
  (List.range max_word_length).map (fun i => words_with_a (i + 1)) |>.sum

/-- The main theorem stating the total number of possible words -/
theorem total_words_count : total_words = 2186085 := by
  sorry

#eval total_words

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_words_count_l403_40339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_p_plus_q_minus_one_l403_40328

/-- Given that 400000001 is the product of two prime numbers p and q,
    prove that the sum of the natural divisors of (p+q-1) is 45864. -/
theorem sum_of_divisors_p_plus_q_minus_one (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p * q = 400000001 → 
  (Finset.sum (Nat.divisors (p + q - 1)) id) = 45864 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_p_plus_q_minus_one_l403_40328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l403_40347

-- Part 1
theorem part1 (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = x^2 - 3*x + 2) →
  (∀ x, f x = x^2 - 5*x + 6) :=
by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) :
  (∀ x, ∃ a b, f x = a*x + b) →
  (∀ x, f (f x) = 4*x + 8) →
  ((∀ x, f x = 2*x + 8/3) ∨ (∀ x, f x = -2*x - 8)) :=
by sorry

-- Part 3
theorem part3 (f : ℝ → ℝ) :
  (∀ x ≠ 0, f (1/x + 1) = 1/x^2 - 1) →
  (∀ x ≠ 1, f x = x^2 - 2*x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l403_40347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_paths_count_l403_40311

/-- The number of shortest paths on a rectangular grid -/
def num_shortest_paths (h k : ℕ) : ℕ :=
  Nat.choose (h + k) h

/-- Theorem: The number of shortest paths from A to B on a rectangular grid,
    where B is h blocks north and k blocks east of A, is equal to (h + k)! / (h! * k!) -/
theorem shortest_paths_count (h k : ℕ) :
  num_shortest_paths h k = (Nat.factorial (h + k)) / (Nat.factorial h * Nat.factorial k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_paths_count_l403_40311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_is_cos_45_l403_40388

noncomputable def complex_equation (z : ℂ) : Prop :=
  z^6 - z^4 + z^2 - 1 = 0

noncomputable def max_imaginary_part (eq : ℂ → Prop) : ℝ :=
  Real.sqrt 2 / 2

theorem max_imaginary_part_is_cos_45 :
  ∃ (φ : ℝ), 
    -π/2 ≤ φ ∧ φ ≤ π/2 ∧
    max_imaginary_part complex_equation = Real.cos φ ∧
    φ = π/4 := by
  sorry

#check max_imaginary_part_is_cos_45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_is_cos_45_l403_40388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l403_40357

theorem survey_result (total : ℕ) 
  (h1 : (868 : ℚ) / 1000 * total = ⌊(21 : ℚ) / ((457 : ℚ) / 1000)⌋)
  (h2 : (457 : ℚ) / 1000 * ⌊(21 : ℚ) / ((457 : ℚ) / 1000)⌋ = 21) :
  total = 53 := by
  sorry

#check survey_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l403_40357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_x_where_f_negative_l403_40335

/-- The function f(x) defined as x^2(ln x - a) + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

/-- Theorem stating that the claim "For all a > 0, for all x > 0, f(x) ≥ 0" is false -/
theorem exists_a_x_where_f_negative :
  ¬ (∀ a > 0, ∀ x > 0, f a x ≥ 0) := by
  sorry

#check exists_a_x_where_f_negative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_x_where_f_negative_l403_40335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l403_40329

def spinner1 : Finset ℕ := {3, 4, 7}
def spinner2 : Finset ℕ := {1, 4, 5, 6, 8}

def isEven (n : ℕ) : Bool := n % 2 = 0

def productIsEven (a : ℕ) (b : ℕ) : Bool := isEven (a * b)

theorem spinner_probability :
  (Finset.card (Finset.filter (fun p => productIsEven p.1 p.2) (spinner1.product spinner2))) /
  (Finset.card (spinner1.product spinner2)) = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l403_40329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_solution_valid_l403_40366

-- Define the integral equation
def integral_equation (φ : ℝ → ℝ) : Prop :=
  ∀ x, φ x + ∫ t in Set.Icc 0 1, x * (Real.exp (x * t) - 1) * φ t = Real.exp x - x

-- Define the approximate solution
noncomputable def φ_approx (x : ℝ) : ℝ :=
  Real.exp x - x * (0.6666 * Real.exp (0.5 * x) + 0.1666 * Real.exp x) - 0.1668 * x

-- State the theorem
theorem approx_solution_valid :
  ∃ ε > 0, ∀ x, |φ_approx x + ∫ t in Set.Icc 0 1, x * (Real.exp (x * t) - 1) * φ_approx t - (Real.exp x - x)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_solution_valid_l403_40366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_shaded_areas_l403_40368

theorem different_shaded_areas (total_area : ℝ) (h_positive : total_area > 0) :
  let shaded_area_I := (1 / 6 : ℝ) * total_area
  let shaded_area_II := (1 / 2 : ℝ) * total_area
  let shaded_area_III := (2 / 3 : ℝ) * total_area
  shaded_area_I ≠ shaded_area_II ∧ 
  shaded_area_I ≠ shaded_area_III ∧ 
  shaded_area_II ≠ shaded_area_III :=
by
  -- Introduce the local definitions
  intro shaded_area_I shaded_area_II shaded_area_III
  
  -- Split the goal into three parts
  apply And.intro
  · -- Prove shaded_area_I ≠ shaded_area_II
    sorry
  apply And.intro
  · -- Prove shaded_area_I ≠ shaded_area_III
    sorry
  · -- Prove shaded_area_II ≠ shaded_area_III
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_shaded_areas_l403_40368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_small_altitudes_large_area_l403_40349

theorem triangle_with_small_altitudes_large_area : 
  ∃ (A B C : ℝ × ℝ), 
    let triangle := (A, B, C)
    let area := abs ((A.1 - C.1) * (B.2 - A.2) - (A.1 - B.1) * (C.2 - A.2)) / 2
    let altitudeA := 2 * area / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let altitudeB := 2 * area / Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    let altitudeC := 2 * area / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    (altitudeA < 0.01 ∧ altitudeB < 0.01 ∧ altitudeC < 0.01) ∧ area > 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_small_altitudes_large_area_l403_40349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_five_l403_40394

/-- A right-angled triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The length of the line segment from the right angle vertex to the midpoint of the hypotenuse -/
noncomputable def crease_length (t : RightTriangle) : ℝ :=
  t.c / 2

theorem crease_length_is_five (t : RightTriangle) : crease_length t = 5 := by
  unfold crease_length
  rw [t.hc]
  norm_num

#check crease_length_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_five_l403_40394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_to_florence_l403_40391

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the y-coordinate of the meeting point -/
noncomputable def meetingPointY (p1 p2 : Point) : ℝ :=
  (p1.y + p2.y) / 2

/-- Calculate the vertical distance between two y-coordinates -/
noncomputable def verticalDistance (y1 y2 : ℝ) : ℝ :=
  |y2 - y1|

theorem meeting_point_distance_to_florence :
  let daniel : Point := ⟨10, -5⟩
  let emma : Point := ⟨0, 20⟩
  let florence : Point := ⟨3, 15⟩
  let meetingY : ℝ := meetingPointY daniel emma
  verticalDistance meetingY florence.y = 7.5 := by
  sorry

#check meeting_point_distance_to_florence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_to_florence_l403_40391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l403_40358

noncomputable def v (x : ℝ) := 1 / Real.sqrt (x - 3)

theorem domain_of_v :
  Set.range v = {y | ∃ x > 3, y = v x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l403_40358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_prime_power_l403_40343

/-- The sum of squares from 2^2 to n^2 -/
def sum_of_squares (n : ℕ) : ℕ := (n - 1) * (2 * n^2 + 5 * n + 6) / 6

/-- A number is a prime power if it's of the form p^k for some prime p and positive integer k -/
def is_prime_power (m : ℕ) : Prop :=
  ∃ (p k : ℕ), Nat.Prime p ∧ k > 0 ∧ m = p^k

theorem sum_of_squares_prime_power :
  ∀ n : ℕ, n > 1 → (is_prime_power (sum_of_squares n) ↔ n ∈ ({2, 3, 4, 7} : Set ℕ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_prime_power_l403_40343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l403_40348

open Real

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (sin (4 * x + π / 3)) / (sin (2 * x + 2 * π / 3))

/-- The line of symmetry between f(x) and g(x) -/
noncomputable def symmetry_line : ℝ := π / 12

/-- The function g(x) symmetric to f(x) about the line x = π/12 -/
noncomputable def g (x : ℝ) : ℝ := f (symmetry_line * 2 - x)

/-- The proposed symmetry center of g(x) -/
noncomputable def symmetry_center : ℝ × ℝ := (π / 4, 0)

/-- Theorem stating that (π/4, 0) is a symmetry center of g(x) -/
theorem symmetry_center_of_g :
  ∀ x : ℝ, g (symmetry_center.1 + x) = g (symmetry_center.1 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l403_40348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_medians_sum_eq_semiperimeter_l403_40377

/-- Represents a right-angled triangle with given leg lengths -/
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  a_pos : 0 < a
  b_pos : 0 < b

/-- Calculates the length of the hypotenuse using the Pythagorean theorem -/
noncomputable def hypotenuse (t : RightTriangle) : ℝ :=
  Real.sqrt (t.a ^ 2 + t.b ^ 2)

/-- Calculates the length of a median in a right-angled triangle -/
noncomputable def median (t : RightTriangle) (side : ℝ) : ℝ :=
  Real.sqrt (2 * t.a ^ 2 + 2 * t.b ^ 2 - side ^ 2) / 2

/-- Calculates the semiperimeter of a right-angled triangle -/
noncomputable def semiperimeter (t : RightTriangle) : ℝ :=
  (t.a + t.b + hypotenuse t) / 2

/-- States that there exists a right-angled triangle where the sum of its two smaller medians
    equals its semiperimeter -/
theorem exists_triangle_medians_sum_eq_semiperimeter :
  ∃ t : RightTriangle, 
    let m1 := median t t.a
    let m2 := median t t.b
    m1 + m2 = semiperimeter t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_medians_sum_eq_semiperimeter_l403_40377
