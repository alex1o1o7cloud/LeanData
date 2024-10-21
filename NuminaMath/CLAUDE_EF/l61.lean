import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pre_tax_remuneration_for_280_tax_l61_6148

/-- Calculates the tax for a given pre-tax remuneration --/
noncomputable def calculate_tax (pre_tax_remuneration : ℝ) : ℝ :=
  let taxable_income := if pre_tax_remuneration ≤ 4000 then pre_tax_remuneration - 800 else pre_tax_remuneration * 0.8
  taxable_income * 0.2 * 0.7

/-- Theorem stating that if the calculated tax is 280, then the pre-tax remuneration is 2800 --/
theorem pre_tax_remuneration_for_280_tax :
  calculate_tax 2800 = 280 := by
  -- Unfold the definition of calculate_tax
  unfold calculate_tax
  -- Simplify the if-then-else expression
  simp [if_pos (show 2800 ≤ 4000 by norm_num)]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pre_tax_remuneration_for_280_tax_l61_6148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_with_constraints_l61_6186

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal distance of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_equation_with_constraints 
  (e : Ellipse) 
  (h_focal_distance : e.focalDistance = 2)
  (h_point : e.equation 3 (-2 * Real.sqrt 6)) :
  e.a = 6 ∧ e.b^2 = 32 := by sorry

#check ellipse_equation_with_constraints

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_with_constraints_l61_6186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l61_6118

/-- The length of segment AB given the specified conditions --/
theorem length_of_segment_AB 
  (k : ℝ) 
  (A B : ℝ × ℝ) 
  (h_A : A.2 = k * (A.1 - 2) ∧ A.2^2 = 8*A.1)
  (h_B : B.2 = k * (B.1 - 2) ∧ B.2^2 = 8*B.1)
  (h_distinct : A ≠ B)
  (h_orthogonal : (A.1 - (-2)) * (B.1 - (-2)) + (A.2 - 4) * (B.2 - 4) = 0)
  : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16^2 := by
  sorry

#check length_of_segment_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l61_6118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ZEN_l61_6144

/-- Triangle XYZ with right angle at Z, XZ = 10, YZ = 26, N midpoint of XY, E on same side of XY as Z, XE = YE = 20 -/
structure TriangleXYZ where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  N : ℝ × ℝ
  E : ℝ × ℝ
  right_angle_Z : (X.1 - Z.1) * (Y.1 - Z.1) + (X.2 - Z.2) * (Y.2 - Z.2) = 0
  XZ_length : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 10
  YZ_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 26
  N_midpoint : N = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)
  E_same_side : (E.2 - X.2) * (Y.1 - X.1) > (E.1 - X.1) * (Y.2 - X.2)
  XE_length : Real.sqrt ((X.1 - E.1)^2 + (X.2 - E.2)^2) = 20
  YE_length : Real.sqrt ((Y.1 - E.1)^2 + (Y.2 - E.2)^2) = 20

/-- The area of triangle ZEN is 65 -/
theorem area_ZEN (t : TriangleXYZ) : Real.sqrt (
    ((t.Z.1 - t.E.1) * (t.N.2 - t.E.2) - (t.Z.2 - t.E.2) * (t.N.1 - t.E.1))^2
  ) / 2 = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ZEN_l61_6144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_shirts_l61_6121

noncomputable def initial_investment : ℝ := 1500
noncomputable def cost_per_shirt : ℝ := 3
noncomputable def selling_price : ℝ := 20

noncomputable def profit_per_shirt : ℝ := selling_price - cost_per_shirt

noncomputable def break_even_point : ℕ := 
  (Int.ceil (initial_investment / profit_per_shirt)).toNat

theorem break_even_shirts : break_even_point = 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_shirts_l61_6121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_rate_calculation_l61_6102

/-- Hiking trip parameters and calculation of downhill rate -/
noncomputable def hiking_trip (total_time uphill_rate uphill_time : ℝ) : ℝ :=
  let downhill_time := total_time - uphill_time
  let distance := uphill_rate * uphill_time
  distance / downhill_time

/-- Theorem stating that the downhill rate is approximately 2.67 mph -/
theorem downhill_rate_calculation :
  ∀ (ε : ℝ), ε > 0 → |hiking_trip 3 4 1.2 - 2.67| < ε :=
by
  sorry

/-- Approximate evaluation of the hiking trip function -/
def approx_hiking_trip : ℚ :=
  (4 * 1.2) / (3 - 1.2)

#eval approx_hiking_trip

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_rate_calculation_l61_6102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_range_l61_6115

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else Real.exp (x * Real.log 2)

theorem f_composition_range :
  {a : ℝ | f (f a) = Real.exp ((f a) * Real.log 2)} = {a : ℝ | a ≥ 2/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_range_l61_6115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_axis_have_no_quadrant_l61_6100

/-- A point in the 2D coordinate system --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a quadrant --/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Function to determine if a point is on a coordinate axis --/
def isOnAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

/-- Function to determine which quadrant a point belongs to --/
noncomputable def quadrantOf (p : Point) : Option Quadrant :=
  if p.x > 0 ∧ p.y > 0 then some Quadrant.first
  else if p.x < 0 ∧ p.y > 0 then some Quadrant.second
  else if p.x < 0 ∧ p.y < 0 then some Quadrant.third
  else if p.x > 0 ∧ p.y < 0 then some Quadrant.fourth
  else none

/-- Theorem: Points on coordinate axes do not belong to any quadrant --/
theorem points_on_axis_have_no_quadrant (p : Point) :
  isOnAxis p → quadrantOf p = none := by
  intro h
  cases h with
  | inl h_x => 
    simp [quadrantOf, isOnAxis, h_x]
  | inr h_y => 
    simp [quadrantOf, isOnAxis, h_y]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_axis_have_no_quadrant_l61_6100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l61_6101

open Set

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axiom for the given inequality
axiom f_inequality (x : ℝ) (hx : x < 0) : 2 * f x + x * f' x > x^2

-- Theorem statement
theorem solution_set :
  {x : ℝ | (x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0} = Ioo (-2019) (-2016) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l61_6101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l61_6140

noncomputable def calculate_total_cost (coffee_price : ℚ) (cake_price : ℚ) (ice_cream_price : ℚ) 
  (sandwich_price : ℚ) (water_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let mell_order := 2 * coffee_price + cake_price + sandwich_price
  let friend_order := 2 * coffee_price + cake_price + sandwich_price + ice_cream_price + water_price
  let friend_order_with_promotion := friend_order - coffee_price / 2
  let total_before_discount := mell_order + 2 * friend_order_with_promotion
  let discounted_total := total_before_discount * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  final_total

theorem total_cost_calculation :
  calculate_total_cost 4 7 3 6 2 (15/100) (1/10) = 647/10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l61_6140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_f_inequality_iff_l61_6159

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x else 1 - 3*x

-- State the theorems
theorem f_composition_negative_one : f (f (-1)) = -5 := by sorry

theorem f_inequality_iff (a : ℝ) : f (2*a^2 - 3) > f (5*a) ↔ -1/2 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_f_inequality_iff_l61_6159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_x_value_l61_6198

theorem power_equality_implies_x_value (x : ℝ) : (512 : ℝ)^x = (64 : ℝ)^240 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_x_value_l61_6198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_transformed_sin_l61_6126

/-- A function obtained by translating sin x left by π/4 and compressing x-axis by 1/2 -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

/-- Theorem stating that f(x) is equivalent to the described transformation of sin x -/
theorem f_eq_transformed_sin (x : ℝ) : 
  f x = Real.sin (2 * x + Real.pi / 4) := by 
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_transformed_sin_l61_6126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_equals_5_sqrt_3_l61_6156

-- Define an equilateral triangle
structure EquilateralTriangle where
  sideLength : ℝ
  isPositive : sideLength > 0

-- Define a point inside the triangle
structure PointInTriangle (t : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  insideTriangle : True  -- Placeholder condition, replace with actual condition later

-- Define the sum of perpendicular distances
def sumOfPerpendiculars (t : EquilateralTriangle) (p : PointInTriangle t) : ℝ :=
  sorry  -- Placeholder for the actual calculation

-- Theorem statement
theorem sum_of_perpendiculars_equals_5_sqrt_3 (t : EquilateralTriangle) 
  (h : t.sideLength = 10) (p : PointInTriangle t) : 
  sumOfPerpendiculars t p = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_equals_5_sqrt_3_l61_6156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_tan_is_odd_l61_6127

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the given functions
noncomputable def f1 (x : ℝ) : ℝ := |Real.sin x|
noncomputable def f2 (x : ℝ) : ℝ := Real.cos x
noncomputable def f3 (x : ℝ) : ℝ := Real.tan x
noncomputable def f4 (x : ℝ) : ℝ := Real.sin |x|

-- Theorem statement
theorem only_tan_is_odd :
  ¬(IsOdd f1) ∧ ¬(IsOdd f2) ∧ (IsOdd f3) ∧ ¬(IsOdd f4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_tan_is_odd_l61_6127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_eight_thirds_l61_6182

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^3

-- Define the tangent line at (1, 1)
noncomputable def tangent_line (x : ℝ) : ℝ := 3*x - 2

-- Define the x-coordinate of the intersection of the tangent line with the x-axis
noncomputable def x_intercept : ℝ := 2/3

-- Define the area of the triangle
noncomputable def triangle_area : ℝ := (1/2) * (2 - x_intercept) * tangent_line 2

-- Theorem statement
theorem triangle_area_is_eight_thirds : triangle_area = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_eight_thirds_l61_6182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_plus_c_l61_6196

/-- The function f(x) = x^2 + bx + c * 3^x -/
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b*x + c * Real.exp (Real.log 3 * x)

/-- The set of real solutions to f(x) = 0 -/
def S (b c : ℝ) : Set ℝ := {x | f b c x = 0}

/-- The set of real solutions to f(f(x)) = 0 -/
def T (b c : ℝ) : Set ℝ := {x | f b c (f b c x) = 0}

theorem range_of_b_plus_c (b c : ℝ) :
  S b c = T b c → S b c ≠ ∅ → b + c ∈ Set.Icc 0 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_plus_c_l61_6196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_reciprocal_squared_l61_6171

open Complex

/-- The area of a parallelogram formed by complex numbers -/
noncomputable def parallelogramArea (z : ℂ) : ℝ :=
  2 * Complex.abs (z.im * (1/z).re - z.re * (1/z).im)

/-- The theorem statement -/
theorem smallest_sum_reciprocal_squared (z : ℂ) (h1 : z.im > 0) 
    (h2 : parallelogramArea z = 12/13) :
    ∃ d : ℝ, d^2 = 16/13 ∧ ∀ w : ℂ, w.im > 0 → parallelogramArea w = 12/13 → 
    Complex.normSq (w + 1/w) ≥ d^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_reciprocal_squared_l61_6171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_inscribed_volumes_l61_6116

/-- Given a cube with edge length 8 inches, containing a perfectly fitted cone and an inscribed sphere,
    prove the volumes of the sphere and cone. -/
theorem cube_inscribed_volumes (cube_edge : ℝ) (h_edge : cube_edge = 8) :
  (4 / 3) * π * (cube_edge / 2)^3 = (256 / 3) * π ∧
  (1 / 3) * π * (cube_edge / 2)^2 * cube_edge = (128 / 3) * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_inscribed_volumes_l61_6116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_drawings_l61_6110

/-- The number of additional drawings that can be made given the total number of markers,
    drawings per marker, and drawings already made. -/
def additional_drawings (total_markers : ℕ) (drawings_per_marker : ℚ) (drawings_made : ℕ) : ℕ :=
  (((total_markers : ℚ) * drawings_per_marker - drawings_made) : ℚ).floor.toNat

/-- Theorem stating that given 12 markers, each lasting for 1.5 drawings,
    and 8 drawings already made, 10 additional drawings can be made. -/
theorem anne_drawings : additional_drawings 12 (3/2) 8 = 10 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_drawings_l61_6110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_parabola_l61_6143

/-- A line is tangent to a parabola if and only if their intersection has exactly one solution -/
def is_tangent (a b c : ℝ) (p : ℝ → ℝ) :=
  ∃! x : ℝ, a * x + b * (p x) + c = 0

/-- The parabola y^2 = 16x -/
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (16 * x)

theorem line_tangent_parabola (k : ℝ) :
  is_tangent 4 7 k parabola ↔ k = 49 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_parabola_l61_6143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_shop_time_is_18_minutes_l61_6163

/-- Represents the cycling scenario with given distances and times -/
structure CyclingScenario where
  total_distance : ℚ
  total_time : ℚ
  coffee_shop_distance : ℚ

/-- Calculates the time to reach the coffee shop given a cycling scenario -/
def time_to_coffee_shop (scenario : CyclingScenario) : ℚ :=
  (scenario.coffee_shop_distance * scenario.total_time) / scenario.total_distance

/-- Theorem stating that the time to reach the coffee shop is 18 minutes -/
theorem coffee_shop_time_is_18_minutes (scenario : CyclingScenario)
  (h1 : scenario.total_distance = 5)
  (h2 : scenario.total_time = 30)
  (h3 : scenario.coffee_shop_distance = 3) :
  time_to_coffee_shop scenario = 18 := by
  sorry

/-- Example calculation -/
def example_scenario : CyclingScenario :=
  { total_distance := 5, total_time := 30, coffee_shop_distance := 3 }

#eval time_to_coffee_shop example_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_shop_time_is_18_minutes_l61_6163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_implies_m_equals_two_l61_6145

/-- If (m+2)x^(|m|-1)+8=0 is a linear equation, then m=2 -/
theorem linear_equation_implies_m_equals_two (m : ℤ) : 
  (∃ a b : ℝ, ∀ x, (m + 2 : ℝ) * x^(Int.natAbs m - 1) + 8 = a * x + b) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_implies_m_equals_two_l61_6145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l61_6187

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (5, 2, 0)
  A₂ : ℝ × ℝ × ℝ := (2, 5, 0)
  A₃ : ℝ × ℝ × ℝ := (1, 2, 4)
  A₄ : ℝ × ℝ × ℝ := (-1, 1, 1)

/-- Calculate the volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedron_height (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume and height of the tetrahedron -/
theorem tetrahedron_properties (t : Tetrahedron) :
  volume t = 12 ∧ tetrahedron_height t = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l61_6187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_b_part_d_l61_6173

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) * x + a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - 1/x) * Real.log x + a

-- Define the zeros of f and g
def zeros_f (a : ℝ) : Set ℝ := {x | f a x = 0}
def zeros_g (a : ℝ) : Set ℝ := {x | g a x = 0}

-- Theorem for part B
theorem part_b (a : ℝ) (x₃ x₄ : ℝ) (h₁ : x₃ ∈ zeros_g a) (h₂ : x₄ ∈ zeros_g a) (h₃ : x₃ < x₄) :
  a < -3 → x₃ + x₄ > 10/3 := by
  sorry

-- Theorem for part D
theorem part_d : ∃ a : ℝ, ∃ x₂ x₃ x₄ : ℝ,
  x₂ ∈ zeros_f a ∧ x₃ ∈ zeros_g a ∧ x₄ ∈ zeros_g a ∧
  x₃ < x₄ ∧ x₃^2 = x₂ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_b_part_d_l61_6173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_congruent_to_one_mod_four_count_l61_6114

theorem two_digit_congruent_to_one_mod_four_count : 
  (Finset.filter (fun n => n % 4 = 1) (Finset.range 90)).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_congruent_to_one_mod_four_count_l61_6114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l61_6199

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((x - 3) / (x + 3)) / Real.log a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log (x - 1) / Real.log a

-- Define the theorem
theorem range_of_a (a m n : ℝ) : 
  0 < a → a < 1 →
  m < n →
  (∀ x, x > 3 → f a x = g a x → (x = m ∨ x = n)) →
  (∀ x, m ≤ x ∧ x ≤ n → ∃ y, m ≤ y ∧ y ≤ n ∧ f a x = g a y) →
  (0 < a ∧ a < (2 - Real.sqrt 3) / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l61_6199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_is_5_l61_6134

/-- A parabola passing through three points -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, y = a * x^2 + b * x + c

/-- The parabola passes through the given points -/
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The x-coordinate of the vertex of a parabola -/
noncomputable def vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

theorem parabola_vertex_x_is_5 (p : Parabola) 
  (h1 : passes_through p 2 16)
  (h2 : passes_through p 8 16)
  (h3 : passes_through p 10 25) :
  vertex_x p = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_is_5_l61_6134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_on_discounted_items_l61_6160

/-- Calculate the total amount spent by friends on discounted items -/
theorem total_spent_on_discounted_items 
  (num_friends : ℕ) 
  (tshirt_price hat_price : ℚ) 
  (tshirt_discount hat_discount : ℚ) 
  (h1 : num_friends = 4)
  (h2 : tshirt_price = 20)
  (h3 : hat_price = 15)
  (h4 : tshirt_discount = 40 / 100)
  (h5 : hat_discount = 60 / 100)
  : num_friends * ((tshirt_price * (1 - tshirt_discount)) + (hat_price * (1 - hat_discount))) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_on_discounted_items_l61_6160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_l61_6130

/-- The sum of complex exponentials -/
noncomputable def complex_sum : ℂ :=
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (23 * Real.pi * Complex.I / 60) +
  Complex.exp (35 * Real.pi * Complex.I / 60) +
  Complex.exp (47 * Real.pi * Complex.I / 60) +
  Complex.exp (59 * Real.pi * Complex.I / 60)

/-- The theorem stating that the argument of the complex sum is 7π/12 -/
theorem complex_sum_arg :
  Complex.arg complex_sum = 7 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_l61_6130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_seven_fiftyone_base_sixteen_l61_6107

/-- Represents a repeating decimal in base k -/
structure RepeatingDecimal (k : ℕ) where
  whole : ℕ
  repeating : List ℕ
  repeating_nonzero : repeating ≠ []

/-- Converts a RepeatingDecimal to a rational number -/
noncomputable def RepeatingDecimal.toRational {k : ℕ} (d : RepeatingDecimal k) : ℚ :=
  sorry

theorem repeating_decimal_seven_fiftyone_base_sixteen :
  ∃ (d : RepeatingDecimal 16),
    d.whole = 0 ∧
    d.repeating = [2, 3] ∧
    d.toRational = 7 / 51 := by
  sorry

#check repeating_decimal_seven_fiftyone_base_sixteen

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_seven_fiftyone_base_sixteen_l61_6107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_count_proof_l61_6103

/-- The number of different bouquets that can be purchased for exactly $60,
    given that roses cost $4 each and carnations cost $2 each. -/
def bouquet_count : ℕ := 16

theorem bouquet_count_proof : bouquet_count = 16 := by
  -- Define the total amount to spend
  let total : ℕ := 60
  -- Define the cost of a rose
  let rose_cost : ℕ := 4
  -- Define the cost of a carnation
  let carnation_cost : ℕ := 2
  -- Define the set of valid bouquets
  let valid_bouquets := {(r, c) : ℕ × ℕ | rose_cost * r + carnation_cost * c = total}
  -- Prove that the number of elements in valid_bouquets is 16
  sorry

#eval bouquet_count -- Should output 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_count_proof_l61_6103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_chloride_formed_l61_6123

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between Hydrochloric acid and Ammonia to form Ammonium chloride -/
structure Reaction where
  hcl : Moles  -- moles of Hydrochloric acid
  nh3 : Moles  -- moles of Ammonia
  nh4cl : Moles  -- moles of Ammonium chloride

/-- The reaction is balanced when the moles of all reactants and products are equal -/
def is_balanced (r : Reaction) : Prop :=
  r.hcl = r.nh3 ∧ r.nh3 = r.nh4cl

/-- Theorem stating that 3 moles of Ammonium chloride are formed -/
theorem ammonium_chloride_formed (r : Reaction) 
  (h1 : r.hcl = (3 : ℝ)) 
  (h2 : r.nh3 = (3 : ℝ)) 
  (h3 : is_balanced r) : 
  r.nh4cl = (3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_chloride_formed_l61_6123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_square_midpoint_path_length_l61_6132

/-- Given a square ABCD with side length a, where S is the midpoint of AB,
    and the square rolls along a line segment of length 5a,
    the length of the curve traced by S is (πa/2)(1 + √5) -/
theorem rolling_square_midpoint_path_length (a : ℝ) (h : a > 0) :
  (π * a / 2) * (1 + Real.sqrt 5) = (π * a / 2) * (1 + Real.sqrt 5) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_square_midpoint_path_length_l61_6132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_sum_l61_6162

/-- Circle with center (-8, 5) and radius 15 -/
def my_circle (x y : ℝ) : Prop := (x + 8)^2 + (y - 5)^2 = 15^2

/-- The sum of y-coordinates of the intersection points of the circle with the y-axis is 10 -/
theorem circle_y_axis_intersection_sum : 
  ∃ y₁ y₂ : ℝ, my_circle 0 y₁ ∧ my_circle 0 y₂ ∧ y₁ + y₂ = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_sum_l61_6162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_mixture_problem_l61_6124

/-- Represents a solution of alcohol and water -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the amount of alcohol in a solution -/
def alcoholAmount (s : Solution) : ℝ := s.volume * s.alcoholPercentage

/-- The problem statement -/
theorem alcohol_water_mixture_problem 
  (initialSolution : Solution)
  (addedAlcohol : ℝ)
  (addedWater : ℝ)
  (finalSolution : Solution) :
  initialSolution.volume = 40 ∧
  initialSolution.alcoholPercentage = 0.05 ∧
  addedAlcohol = 6.5 ∧
  finalSolution.alcoholPercentage = 0.17 ∧
  finalSolution.volume = initialSolution.volume + addedAlcohol + addedWater ∧
  alcoholAmount finalSolution = alcoholAmount initialSolution + addedAlcohol →
  abs (addedWater - 3.5) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_mixture_problem_l61_6124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l61_6105

/-- Definition of ellipse M -/
def ellipse_M (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

/-- Right focus of ellipse M -/
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Symmetry condition -/
def symmetric_about (c : ℝ) (x y : ℝ) : Prop :=
  ellipse_M x y (2*c) ↔ ellipse_M (2*c - x) y (2*c)

/-- Passes through origin -/
def passes_through_origin (a : ℝ) : Prop :=
  ellipse_M 0 0 a

/-- Line through (4,0) intersecting ellipse M -/
def intersecting_line (k : ℝ) (x : ℝ) : ℝ := k * (x - 4)

/-- Point Q symmetric to E about x-axis -/
def symmetric_points (x y : ℝ) : Prop :=
  ellipse_M x y 2 ∧ ellipse_M x (-y) 2

/-- Main theorem -/
theorem ellipse_theorem (a c : ℝ) (h1 : a > Real.sqrt 3) (h2 : c > 0) 
  (h3 : ∀ x y, symmetric_about c x y)
  (h4 : passes_through_origin a)
  (h5 : ∀ k, k ≠ 0 → ∃ x1 y1 x2 y2, 
    ellipse_M x1 y1 a ∧ 
    ellipse_M x2 y2 a ∧
    y1 = intersecting_line k x1 ∧
    y2 = intersecting_line k x2 ∧
    symmetric_points x2 y2) :
  (∀ x y, ellipse_M x y a ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ k, k ≠ 0 → ∃ x1 y1 x2 y2,
    ellipse_M x1 y1 2 ∧
    ellipse_M x2 y2 2 ∧
    y1 = intersecting_line k x1 ∧
    y2 = intersecting_line k x2 ∧
    symmetric_points x2 y2 ∧
    (x1 * intersecting_line k x2 - x2 * intersecting_line k x1) / 
    (intersecting_line k x2 - intersecting_line k x1) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l61_6105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_complex_l61_6117

theorem equation_solution_complex (c d : ℂ) : 
  c^2 ≠ 0 → c + d^2 ≠ 0 → (c + d^2) / c^2 = 2 * d / (c + d^2) → 
  ¬(c.re = c ∧ d.re = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_complex_l61_6117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_over_ln2_eq_2n_l61_6157

/-- The sum of squares of elements in the nth row of Pascal's triangle -/
def S (n : ℕ) : ℕ := 2^(2*n)

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ := Real.log (S n)

/-- Theorem stating the relationship between g(n)/ln(2) and n -/
theorem g_over_ln2_eq_2n (n : ℕ) : g n / Real.log 2 = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_over_ln2_eq_2n_l61_6157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_27_l61_6128

/-- The angle between clock hands at a given time -/
noncomputable def clockAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hourAngle := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle := minutes * 6
  abs (hourAngle - minuteAngle)

/-- The acute angle between clock hands -/
noncomputable def acuteClockAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  min (clockAngle hours minutes) (360 - clockAngle hours minutes)

theorem clock_angle_at_3_27 :
  acuteClockAngle 3 27 = 58.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_27_l61_6128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_equals_92_point_4_l61_6170

-- Define the writing times for each author
noncomputable def woody_writing_time : ℝ := 18 -- 1.5 years * 12 months/year
noncomputable def ivanka_writing_time : ℝ := woody_writing_time + 3
noncomputable def alice_writing_time : ℝ := woody_writing_time / 2
noncomputable def tom_writing_time : ℝ := alice_writing_time * 2

-- Define the editing and revising percentages
def editing_percentage : ℝ := 0.25
def revising_percentage : ℝ := 0.15

-- Calculate total time for each author
noncomputable def author_total_time (writing_time : ℝ) : ℝ :=
  writing_time * (1 + editing_percentage + revising_percentage)

-- Calculate the total time for all authors
noncomputable def total_time_all_authors : ℝ :=
  author_total_time ivanka_writing_time +
  author_total_time woody_writing_time +
  author_total_time alice_writing_time +
  author_total_time tom_writing_time

-- Theorem statement
theorem total_time_equals_92_point_4 :
  total_time_all_authors = 92.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_equals_92_point_4_l61_6170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_fraction_l61_6164

def options : List ℝ := [100, 500, 1900, 2000, 2500]

noncomputable def closestTo (x : ℝ) (l : List ℝ) : ℝ :=
  (l.argmin (fun y => |y - x|)).getD 0

theorem closest_to_fraction :
  closestTo (410 / 0.21) options = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_fraction_l61_6164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_work_time_l61_6106

/-- Represents the time (in days) it takes for a person to complete a task alone -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents the rate at which a person completes a task -/
noncomputable def workRate (w : WorkTime) : ℚ := 1 / w.days

/-- Theorem stating that A can complete the work in 8 days -/
theorem a_work_time (b c : WorkTime) (h_b : b.days = 16) (h_c : c.days = 16)
  (h_abc : workRate b + workRate c + workRate ⟨8, by norm_num⟩ = 1 / 4) :
  ∃ (a : WorkTime), a.days = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_work_time_l61_6106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_line_distance_l61_6168

open Real

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / sqrt (l.a^2 + l.b^2)

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point where
  x := (t.A.x + t.B.x + t.C.x) / 3
  y := (t.A.y + t.B.y + t.C.y) / 3

/-- Theorem: In any triangle, for a line passing through its centroid but not through any vertex,
    the sum of distances from two vertices to the line equals the distance from the third vertex to the line -/
theorem triangle_centroid_line_distance (t : Triangle) (r : Line) :
  let G := centroid t
  (r.a * G.x + r.b * G.y + r.c = 0) →  -- r passes through the centroid
  (r.a * t.A.x + r.b * t.A.y + r.c ≠ 0) →  -- r doesn't pass through A
  (r.a * t.B.x + r.b * t.B.y + r.c ≠ 0) →  -- r doesn't pass through B
  (r.a * t.C.x + r.b * t.C.y + r.c ≠ 0) →  -- r doesn't pass through C
  (distancePointToLine t.A r = distancePointToLine t.B r + distancePointToLine t.C r) ∨
  (distancePointToLine t.B r = distancePointToLine t.A r + distancePointToLine t.C r) ∨
  (distancePointToLine t.C r = distancePointToLine t.A r + distancePointToLine t.B r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_line_distance_l61_6168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_theorem_l61_6166

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle. -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to a line ax + by + c = 0. -/
def Circle.tangentTo (c : Circle) (a b d : ℝ) : Prop :=
  |a * c.center.1 + b * c.center.2 + d| = c.radius * Real.sqrt (a^2 + b^2)

/-- The main theorem about the circle. -/
theorem circle_theorem (c : Circle) :
  c.contains (0, -1) ∧
  c.tangentTo 1 1 (-1) ∧
  c.center.2 = -2 * c.center.1 →
  ((c.center = (1, -2) ∧ c.radius^2 = 2) ∨
   (c.center = (1/9, -2/9) ∧ c.radius^2 = 50/81)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_theorem_l61_6166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l61_6161

def sequence_f (n : ℕ) : ℚ := n / (n + 1)

theorem sequence_value :
  (1 : ℚ) / 2 / (List.foldl (λ acc i => acc / sequence_f i) 1 (List.range 2020)) = 2021 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l61_6161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l61_6185

/-- The constant term in the expansion of (x - 1/x - 1)^4 -/
def constant_term : ℤ := -5

/-- The expression (x - 1/x - 1)^4 -/
def expression (x : ℚ) : ℚ := (x - 1/x - 1)^4

theorem constant_term_proof : 
  ∃ (f : ℚ → ℚ), (∀ x, x ≠ 0 → f x = expression x) ∧ 
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c = constant_term) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l61_6185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l61_6167

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem isosceles_triangle_area : 
  triangle_area 10 11 11 = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l61_6167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l61_6183

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  A := sorry
  B := 45 * Real.pi / 180  -- Convert 45° to radians
  C := sorry
  a := Real.sqrt 3
  b := Real.sqrt 2
  c := sorry

-- Define the theorem
theorem triangle_solution (t : Triangle) (h : t = given_triangle) :
  (t.A = 60 * Real.pi / 180 ∨ t.A = 120 * Real.pi / 180) ∧
  (t.C = 75 * Real.pi / 180 ∨ t.C = 15 * Real.pi / 180) ∧
  (t.c = (Real.sqrt 6 + Real.sqrt 2) / 2 ∨ t.c = (Real.sqrt 6 - Real.sqrt 2) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l61_6183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_abs_value_l61_6147

theorem quadratic_abs_value (x : ℂ) : 
  x^2 - 6*x + 20 = 0 → ∃! r : ℝ, r > 0 ∧ r^2 = 20 ∧ Complex.abs x = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_abs_value_l61_6147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_beta_plus_delta_l61_6137

variable (β δ : ℂ)

def g (z : ℂ) : ℂ := (3 - 2*Complex.I)*z^2 + β*z + δ

theorem min_abs_beta_plus_delta :
  (g β δ 1).im = 0 →
  (g β δ (-Complex.I)).im = 0 →
  ∀ β' δ' : ℂ, (g β' δ' 1).im = 0 → (g β' δ' (-Complex.I)).im = 0 → Complex.abs β + Complex.abs δ ≤ Complex.abs β' + Complex.abs δ' →
  Complex.abs β + Complex.abs δ = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_beta_plus_delta_l61_6137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_championship_pigeonhole_l61_6195

theorem football_championship_pigeonhole (n : ℕ) (h : n = 30) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ k < n ∧ 
    ∀ (function : Fin n → ℕ), function i = k ∧ function j = k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_championship_pigeonhole_l61_6195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incident_ray_equation_l61_6175

/-- The line of reflection --/
def reflection_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

/-- The point of emission --/
def M : ℝ × ℝ := (-1, 0)

/-- The point through which the reflected ray passes --/
def N : ℝ × ℝ := (0, 1)

/-- The equation of the incident ray --/
def incident_ray (x y : ℝ) : Prop :=
  x + 3*y + 1 = 0

/-- The reflection of point N with respect to the reflection line --/
noncomputable def N' : ℝ × ℝ := (2, -1)

theorem incident_ray_equation :
  ∀ x y : ℝ,
  (x - M.1) * (N'.2 - M.2) = (y - M.2) * (N'.1 - M.1) ↔
  incident_ray x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incident_ray_equation_l61_6175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_journey_time_l61_6109

noncomputable def unloaded_speed : ℝ := 20
noncomputable def loaded_speed : ℝ := 10

noncomputable def journey_part1 : ℝ := 180
noncomputable def journey_part2 : ℝ := 120
noncomputable def journey_part3 : ℝ := 80
noncomputable def journey_part4 : ℝ := 140

noncomputable def total_time : ℝ := journey_part1 / loaded_speed + 
                      journey_part2 / unloaded_speed + 
                      journey_part3 / loaded_speed + 
                      journey_part4 / unloaded_speed

theorem toby_journey_time : total_time = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_journey_time_l61_6109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_at_2_l61_6176

-- Define f'' as a function
noncomputable def f'' : ℝ → ℝ := sorry

-- Define f' as a function
noncomputable def f' : ℝ → ℝ := sorry

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (f'' 1) * x^3 - 2 * x^2 + 3

-- State the theorem
theorem f_second_derivative_at_2 (h : f' 1 = 2) : f'' 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_at_2_l61_6176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_problem_solution_l61_6194

/-- Represents a candle with a burn rate (fraction of total length burned per hour) -/
structure Candle where
  burnRate : ℚ
  deriving Repr

/-- The problem setup with three candles -/
structure CandleProblem where
  candle1 : Candle
  candle2 : Candle
  candle3 : Candle
  initialLength : ℚ
  deriving Repr

theorem candle_problem_solution (p : CandleProblem)
  (h1 : p.candle2.burnRate = 1 / 12)
  (h2 : p.candle3.burnRate = 1 / 8)
  (h3 : ∃ t : ℚ, 
    p.initialLength - p.candle1.burnRate * (t + 1) * p.initialLength = 
    p.initialLength - p.candle3.burnRate * t * p.initialLength)
  (h4 : ∃ t : ℚ, 
    p.initialLength - p.candle1.burnRate * (t + 3) * p.initialLength = 
    p.initialLength - p.candle2.burnRate * (t + 2) * p.initialLength)
  : p.candle1.burnRate = 1 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_problem_solution_l61_6194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l61_6133

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- The first, second, and fourth terms form a geometric sequence -/
def geometric_subseq (a : ℕ → ℚ) : Prop :=
  (a 2)^2 = a 1 * a 4

/-- Sum of the first n terms of the sequence -/
def seq_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  arithmetic_seq a → geometric_subseq a →
  (a 1 = 2 ∧ ∀ n, seq_sum a n = n^2 + n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l61_6133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_calculation_l61_6191

/-- Calculates the dividend received from an investment in shares with a premium and dividend rate -/
theorem dividend_calculation (investment : ℕ) (share_value : ℕ) (premium_rate : ℚ) (dividend_rate : ℚ) : 
  investment = 14400 →
  share_value = 100 →
  premium_rate = 1/5 →
  dividend_rate = 1/20 →
  (dividend_rate * ↑share_value * (↑investment / (↑share_value * (1 + premium_rate)))).floor = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_calculation_l61_6191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_6_7_l61_6135

def sequenceFunction (a : ℚ) : ℚ :=
  if 0 ≤ a ∧ a < 1/2 then 2*a
  else if 1/2 ≤ a ∧ a < 1 then 2*a - 1
  else a

def a : ℕ → ℚ
| 0 => 6/7
| n + 1 => sequenceFunction (a n)

theorem a_2017_equals_6_7 : a 2017 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_6_7_l61_6135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_g_odd_and_symmetric_l61_6169

open Real

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x - b * cos x

-- Define the function g
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a b (π/4 - x)

-- Theorem statement
theorem f_minimum_implies_g_odd_and_symmetric (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x ≥ f a b (3*π/4)) →
  (∀ x, g a b x = -g a b (-x)) ∧
  (∀ x, g a b (x + π) = -g a b (π - x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_g_odd_and_symmetric_l61_6169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_alone_completion_time_l61_6172

-- Define the work rates
def work_rate_y : ℝ → ℝ := id

noncomputable def work_rate_x (wy : ℝ) : ℝ := 3 * wy

noncomputable def work_rate_z (wy : ℝ) : ℝ := (1/2) * wy

-- Define the total work
noncomputable def total_work (wy : ℝ) : ℝ := 20 * (work_rate_x wy + wy + work_rate_z wy)

-- Theorem statement
theorem y_alone_completion_time (wy : ℝ) (wy_pos : wy > 0) :
  total_work wy = 90 * wy := by
  sorry

#check y_alone_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_alone_completion_time_l61_6172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_div_D_eq_sixteen_l61_6152

-- Define the series C
noncomputable def C : ℝ := ∑' n, if n % 4 ≠ 0 then ((-1) ^ ((n - 2) / 2)) / n^2 else 0

-- Define the series D
noncomputable def D : ℝ := ∑' n, if n % 4 = 0 ∧ n ≠ 0 then ((-1) ^ (n / 4 - 1)) / n^2 else 0

-- Theorem stating that C/D = 16
theorem C_div_D_eq_sixteen : C / D = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_div_D_eq_sixteen_l61_6152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l61_6120

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 6) = 1 / 3) :
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l61_6120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_ratio_theorem_l61_6125

/-- Represents the ratio of milk types sold --/
structure MilkRatio where
  regular : ℕ
  chocolate : ℕ
  strawberry : ℕ
deriving Repr

/-- Given the sales data, compute the ratio of milk types sold --/
def computeMilkRatio (totalSold : ℕ) (regularSold : ℕ) (chocolateSold : ℕ) : MilkRatio :=
  let strawberrySold := totalSold - regularSold - chocolateSold
  let gcd := regularSold.gcd (chocolateSold.gcd strawberrySold)
  { regular := regularSold / gcd
    chocolate := chocolateSold / gcd
    strawberry := strawberrySold / gcd }

theorem milk_ratio_theorem (totalSold : ℕ) (regularSold : ℕ) (chocolateSold : ℕ)
    (h1 : totalSold = 60)
    (h2 : regularSold = 12)
    (h3 : chocolateSold = 24) :
    computeMilkRatio totalSold regularSold chocolateSold = { regular := 1, chocolate := 2, strawberry := 2 } := by
  sorry

#eval computeMilkRatio 60 12 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_ratio_theorem_l61_6125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l61_6155

theorem sin_cos_difference (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/2) : Real.sin α - Real.cos α = Real.sqrt 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l61_6155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invisibility_cloak_change_l61_6179

/-- Represents the price and change for an invisibility cloak transaction -/
structure CloakTransaction where
  silver_cost : ℕ
  gold_change : ℕ

/-- Represents the exchange rate between silver and gold coins -/
def exchange_rate : ℚ := 5 / 3

/-- The cost of an invisibility cloak in gold coins -/
def cloak_cost : ℕ := 8

/-- Calculate the change in silver coins when buying a cloak with gold coins -/
def calculate_change (gold_paid : ℕ) : ℕ :=
  (((gold_paid - cloak_cost : ℤ) * exchange_rate).floor : ℤ).toNat

theorem invisibility_cloak_change 
  (transaction1 : CloakTransaction) 
  (transaction2 : CloakTransaction) 
  (h1 : transaction1.silver_cost = 20 ∧ transaction1.gold_change = 4)
  (h2 : transaction2.silver_cost = 15 ∧ transaction2.gold_change = 1)
  (h3 : transaction1.silver_cost - transaction1.gold_change * 5 / 3 = 
       transaction2.silver_cost - transaction2.gold_change * 5 / 3) :
  calculate_change 14 = 10 := by
  sorry

#eval calculate_change 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invisibility_cloak_change_l61_6179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l61_6142

/-- Calculates the total cost of fencing for an irregular polygon-shaped field -/
noncomputable def fencingCost (sides : List ℝ) (rate1 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  let perimeter := sides.sum
  let cost1 := min perimeter 50 * rate1
  let cost2 := min (max (perimeter - 50) 0) 50 * rate2
  let cost3 := max (perimeter - 100) 0 * rate3
  cost1 + cost2 + cost3

/-- Theorem stating the total cost of fencing for the given field -/
theorem fencing_cost_theorem (sides : List ℝ) (h1 : sides = [32, 45, 25, 52, 39, 60]) :
  fencingCost sides 1.5 1.75 2 = 468.5 := by
  sorry

-- Use #eval only for Int or Nat calculations
def fencingCostInt (sides : List Int) (rate1 : Int) (rate2 : Int) (rate3 : Int) : Int :=
  let perimeter := sides.sum
  let cost1 := min perimeter 50 * rate1
  let cost2 := min (max (perimeter - 50) 0) 50 * rate2
  let cost3 := max (perimeter - 100) 0 * rate3
  (cost1 + cost2 + cost3) / 100  -- Assuming rates are in cents

#eval fencingCostInt [32, 45, 25, 52, 39, 60] 150 175 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l61_6142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circles_l61_6184

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  vertices : Fin 4 → Point

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Function to check if two points are vertices of a rectangle -/
def are_vertices (r : Rectangle) (p q : Point) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ r.vertices i = p ∧ r.vertices j = q

/-- Function to create a circle from two points as diameter endpoints -/
noncomputable def circle_from_diameter (p q : Point) : Circle :=
  { center := ⟨(p.x + q.x) / 2, (p.y + q.y) / 2⟩,
    radius := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) / 2 }

/-- The main theorem -/
theorem rectangle_circles (r : Rectangle) :
  ∃! (circles : Finset Circle), 
    circles.card = 5 ∧ 
    (∀ c ∈ circles, ∃ p q, are_vertices r p q ∧ c = circle_from_diameter p q) ∧
    (∀ p q, are_vertices r p q → circle_from_diameter p q ∈ circles) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circles_l61_6184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_absolute_values_30_l61_6193

def sequenceA (n : ℕ) : ℤ :=
  -60 + 3 * (n.pred)

def sumAbsoluteValues (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => Int.natAbs (sequenceA (i + 1)))

theorem sum_absolute_values_30 :
  sumAbsoluteValues 30 = 765 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_absolute_values_30_l61_6193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_path_count_l61_6108

/-- Represents a face of the icosahedron -/
inductive Face
| Top
| TopRing
| BottomRing
| Bottom

/-- Represents a path from top to bottom of the icosahedron -/
def IcosahedronPath := List Face

/-- The regular icosahedron structure -/
structure Icosahedron where
  faces : Nat
  topFaces : Nat
  bottomFaces : Nat
  topRingFaces : Nat
  bottomRingFaces : Nat

/-- Checks if a path is valid according to the problem constraints -/
def isValidPath (p : IcosahedronPath) : Bool :=
  sorry

/-- Counts the number of valid paths from top to bottom -/
def countValidPaths (ico : Icosahedron) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem icosahedron_path_count :
  let ico : Icosahedron := {
    faces := 20,
    topFaces := 3,
    bottomFaces := 3,
    topRingFaces := 5,
    bottomRingFaces := 5
  }
  countValidPaths ico = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_path_count_l61_6108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_shaped_area_l61_6189

/-- The area of the football-shaped region formed by two quarter circles in a square -/
theorem football_shaped_area (side_length : ℝ) (h : side_length = 4) :
  let quarter_circle_area := π * side_length^2 / 4
  let triangle_area := side_length^2 / 2
  let football_area := 2 * (quarter_circle_area - triangle_area)
  football_area = 8 * π - 16 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_shaped_area_l61_6189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reroll_two_dice_l61_6104

/-- Represents a standard six-sided die --/
def Die := Fin 6

/-- The optimal strategy for rerolling dice to achieve a sum of 9 --/
def optimalRerollStrategy (initialRoll : Die × Die × Die) : Fin 4 := sorry

/-- The probability of an event occurring in the dice game --/
def probability (event : (Die × Die × Die) → Prop) : ℚ := sorry

/-- Theorem stating the probability of rerolling exactly two dice in the optimal strategy --/
theorem probability_reroll_two_dice :
  probability (λ roll => optimalRerollStrategy roll = 2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reroll_two_dice_l61_6104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_300_l61_6192

theorem perfect_squares_between_50_and_300 :
  (Finset.filter (fun n => 50 ≤ n^2 ∧ n^2 ≤ 300) (Finset.range 18)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_300_l61_6192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_difference_l61_6181

/-- Represents the price changes for a product -/
structure PriceChanges :=
  (change1 : ℝ)
  (change2 : ℝ)
  (change3 : ℝ)
  (change4 : ℝ)

/-- Calculates the final price after applying a series of percentage changes -/
def applyPriceChanges (initialPrice : ℝ) (changes : PriceChanges) : ℝ :=
  initialPrice * (1 + changes.change1) * (1 + changes.change2) * (1 + changes.change3) * (1 + changes.change4)

/-- Represents a customer's purchase amounts -/
structure CustomerPurchase :=
  (amountX : ℝ)
  (amountY : ℝ)

/-- Theorem statement -/
theorem expenditure_difference 
  (initialPrice : ℝ)
  (changesX : PriceChanges)
  (changesY : PriceChanges)
  (customerA : CustomerPurchase)
  (customerB : CustomerPurchase) :
  initialPrice > 0 →
  changesX = { change1 := 0.1, change2 := -0.12, change3 := 0.05, change4 := 0.07 } →
  changesY = { change1 := -0.07, change2 := 0.08, change3 := 0.05, change4 := -0.06 } →
  customerA = { amountX := 0.65, amountY := 0.80 } →
  customerB = { amountX := 0.75, amountY := 0.90 } →
  let finalPriceX := applyPriceChanges initialPrice changesX
  let finalPriceY := applyPriceChanges initialPrice changesY
  let expenditureA := customerA.amountX * finalPriceX + customerA.amountY * finalPriceY
  let expenditureB := customerB.amountX * finalPriceX + customerB.amountY * finalPriceY
  abs (expenditureB - expenditureA - 20.79) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_difference_l61_6181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l61_6158

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := 3*x - y = 0

-- Theorem statement
theorem circle_properties :
  -- The circle passes through point A(2, 3)
  my_circle 2 3 ∧
  -- The center of the circle lies on the line 3x - y = 0
  (∃ x y : ℝ, my_circle x y ∧ center_line x y) ∧
  -- The circle is tangent to the y-axis
  (∃ y : ℝ, my_circle 0 y ∧ (∀ x' y' : ℝ, x' > 0 → ¬my_circle x' y')) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l61_6158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l61_6122

/-- The volume of a regular triangular pyramid with height h and right angles at the apex -/
noncomputable def regularTriangularPyramidVolume (h : ℝ) : ℝ := (h^3 * Real.sqrt 3) / 2

/-- Theorem: The volume of a regular triangular pyramid with height h and right angles at the apex is (h³√3)/2 -/
theorem regular_triangular_pyramid_volume (h : ℝ) (h_pos : h > 0) :
  regularTriangularPyramidVolume h = (h^3 * Real.sqrt 3) / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l61_6122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l61_6119

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x < 2^x)) ↔ ∃ x : ℝ, x ≥ 2^x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l61_6119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_odd_terms_l61_6129

/-- The largest digit in a natural number -/
def largest_digit (n : ℕ) : ℕ :=
  (n.digits 10).maximum?.getD 0

/-- A sequence of natural numbers where each number except the first
    is obtained by adding the largest digit of the previous number to it. -/
def DigitAdditionSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a (n-1) + largest_digit (a (n-1))

/-- A predicate to check if a number is odd -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- The maximum number of consecutive odd terms in the sequence is 5 -/
theorem max_consecutive_odd_terms
  (a : ℕ → ℕ) (h : DigitAdditionSequence a) :
  ∀ k : ℕ, (∀ i : ℕ, i < 6 → is_odd (a (k + i))) →
    ¬(∀ i : ℕ, i < 7 → is_odd (a (k + i))) :=
by
  sorry

/-- Illustrative example of 5 consecutive odd terms -/
def example_sequence : ℕ → ℕ
  | 0 => 807
  | n + 1 => example_sequence n + largest_digit (example_sequence n)

#eval [example_sequence 0, example_sequence 1, example_sequence 2, example_sequence 3, example_sequence 4, example_sequence 5]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_odd_terms_l61_6129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_books_l61_6113

theorem special_collection_books (return_rate : ℚ) (end_count books_loaned : ℕ) : 
  return_rate = 65 / 100 →
  end_count = 244 →
  books_loaned = 160 →
  ∃ initial_count : ℕ, initial_count = 300 ∧ 
    initial_count = end_count + books_loaned - (return_rate * books_loaned).floor :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_books_l61_6113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_envelopes_need_extra_postage_l61_6141

/-- Represents an envelope with length and height -/
structure Envelope where
  length : ℚ
  height : ℚ

/-- Checks if an envelope requires extra postage -/
def requiresExtraPostage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 14/10 || ratio > 26/10

/-- The set of envelopes given in the problem -/
def envelopes : List Envelope := [
  ⟨7, 5⟩,  -- Envelope A
  ⟨8, 2⟩,  -- Envelope B
  ⟨7, 7⟩,  -- Envelope C
  ⟨12, 4⟩  -- Envelope D
]

/-- Theorem: Exactly 3 envelopes require extra postage -/
theorem three_envelopes_need_extra_postage :
  (envelopes.filter requiresExtraPostage).length = 3 := by
  -- Proof goes here
  sorry

#eval (envelopes.filter requiresExtraPostage).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_envelopes_need_extra_postage_l61_6141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_approx_l61_6153

open Real

-- Define the angle α in radians
noncomputable def α : ℝ := 38 * π / 180 + 40 * π / (180 * 60)

-- Define the angle bisector length
def fa : ℝ := 7.8

-- Define the area calculation function
noncomputable def triangle_area (α : ℝ) (fa : ℝ) : ℝ :=
  (fa^2 * (cos (α/2))^2 * tan α) / 2

-- State the theorem
theorem right_triangle_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |triangle_area α fa - 21.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_approx_l61_6153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_surface_area_l61_6174

/-- Theorem: The surface area of a cube inscribed in a sphere that is inscribed in another cube with surface area 24 square meters is 8 square meters. -/
theorem inscribed_cube_surface_area (outer_cube_surface_area : ℝ) 
  (h1 : outer_cube_surface_area = 24) : ℝ :=
by
  -- Define the side length of the outer cube
  let outer_side_length := Real.sqrt (outer_cube_surface_area / 6)
  
  -- Define the diameter of the inscribed sphere (equal to outer cube's side length)
  let sphere_diameter := outer_side_length
  
  -- Define the side length of the inner cube
  let inner_side_length := Real.sqrt ((sphere_diameter ^ 2) / 3)
  
  -- Calculate the surface area of the inner cube
  let inner_cube_surface_area := 6 * inner_side_length ^ 2
  
  -- Prove that the inner cube's surface area is 8
  sorry

-- Remove the #eval statement as it was causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_surface_area_l61_6174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l61_6136

def CircularArrangement := List Char

def writeEveryThird (arrangement : CircularArrangement) : List Char :=
  sorry

theorem correct_arrangement (arrangement : CircularArrangement) :
  arrangement ≠ [] →
  arrangement.head? = some 'L' →
  writeEveryThird arrangement = ['L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'] →
  arrangement = ['L', 'R', 'O', 'M', 'S', 'Q', 'N', 'P'] := by
  sorry

#check correct_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l61_6136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_walking_distance_l61_6190

/-- Tom's walking rate in miles per minute -/
def walking_rate : ℚ := 1 / 18

/-- The time Tom walks in minutes -/
def walking_time : ℚ := 15

/-- Function to round a rational number to the nearest tenth -/
def round_to_tenth (x : ℚ) : ℚ := 
  ⌊(x * 10 + 1/2)⌋ / 10

/-- Theorem stating that Tom walks 0.8 miles in 15 minutes, rounded to the nearest tenth -/
theorem tom_walking_distance : 
  round_to_tenth (walking_rate * walking_time) = 8 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_walking_distance_l61_6190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l61_6197

theorem system_solutions :
  let S := {(x, y) : ℝ × ℝ | x^2 + y = 12 ∧ y^2 + x = 12}
  S = {(3, 3), (-4, -4), 
       ((1 + 3 * Real.sqrt 5) / 2, (1 - 3 * Real.sqrt 5) / 2),
       ((1 - 3 * Real.sqrt 5) / 2, (1 + 3 * Real.sqrt 5) / 2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l61_6197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_value_l61_6178

/-- The area of an isosceles triangle with two sides of length 26 and base 30 -/
noncomputable def isosceles_triangle_area : ℝ :=
  let a : ℝ := 26  -- length of equal sides
  let b : ℝ := 30  -- length of base
  let h : ℝ := Real.sqrt (a^2 - (b/2)^2)  -- height of the triangle
  (1/2) * b * h

theorem isosceles_triangle_area_value : isosceles_triangle_area = 15 * Real.sqrt 451 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval isosceles_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_value_l61_6178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l61_6146

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

-- Define the line l in polar form
def line_l (ρ θ : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (θ + Real.pi / 4) = 3

-- Define the distance function from a point to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + y - 3| / Real.sqrt 2

-- Theorem statement
theorem max_distance_curve_to_line :
  ∃ d_max : ℝ, d_max = (Real.sqrt 10 + 3 * Real.sqrt 2) / 2 ∧
  ∀ α : ℝ, distance_to_line (curve_C α).1 (curve_C α).2 ≤ d_max :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l61_6146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunflower_percent_in_mix_l61_6154

/-- Represents a brand of birdseed -/
structure Birdseed where
  sunflowerPercent : ℚ
  milletPercent : ℚ
  sunflower_millet_sum : sunflowerPercent + milletPercent = 1

/-- Calculates the percentage of sunflower in a mix of two birdseed brands -/
def mixSunflowerPercent (brandA brandB : Birdseed) (brandAPercent : ℚ) : ℚ :=
  brandA.sunflowerPercent * brandAPercent + brandB.sunflowerPercent * (1 - brandAPercent)

theorem sunflower_percent_in_mix 
  (brandA : Birdseed)
  (brandB : Birdseed)
  (h1 : brandA.sunflowerPercent = 3/5)
  (h2 : brandB.sunflowerPercent = 7/20)
  (h3 : mixSunflowerPercent brandA brandB (3/5) = 1/2) :
  True :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunflower_percent_in_mix_l61_6154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_a_value_l61_6149

/-- The linear relationship between x and y --/
noncomputable def linear_relationship (x : ℝ) : ℝ := 2.1 * x - 0.3

/-- The mean of x values --/
noncomputable def mean_x : ℝ := (1 + 2 + 3 + 4 + 5) / 5

/-- The sum of known y values --/
noncomputable def sum_known_y : ℝ := 2 + 3 + 7 + 8

/-- The theorem stating that a = 10 satisfies the linear relationship --/
theorem population_a_value (a : ℝ) : 
  (linear_relationship mean_x = (sum_known_y + a) / 5) → a = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_a_value_l61_6149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_when_m_is_one_range_of_m_when_f_range_is_real_range_of_m_when_f_increasing_l61_6180

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x - m) / Real.log (1/2)

-- Theorem 1: Domain of f when m = 1
theorem domain_when_m_is_one :
  {x : ℝ | f 1 x ∈ Set.univ} = {x : ℝ | x < (1 - Real.sqrt 5) / 2 ∨ x > (1 + Real.sqrt 5) / 2} := by
  sorry

-- Theorem 2: Range of m when range of f is ℝ
theorem range_of_m_when_f_range_is_real :
  {m : ℝ | ∀ y : ℝ, ∃ x : ℝ, f m x = y} = {m : ℝ | m ≤ -4 ∨ m ≥ 0} := by
  sorry

-- Theorem 3: Range of m when f is increasing in (-∞, 1 - √3)
theorem range_of_m_when_f_increasing :
  {m : ℝ | ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 - Real.sqrt 3 → f m x₁ < f m x₂} =
  {m : ℝ | 2 - 2 * Real.sqrt 3 ≤ m ∧ m ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_when_m_is_one_range_of_m_when_f_range_is_real_range_of_m_when_f_increasing_l61_6180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_isosceles_set_l61_6165

def isValidSet (A : Finset ℕ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x ≠ y → x ≤ 2015 ∧ y ≤ 2015 ∧
    ((2*x > y ∧ y > x) ∨ (2*y > x ∧ x > y))

theorem largest_isosceles_set :
  ∃ (A : Finset ℕ), isValidSet A ∧ A.card = 10 ∧
    ∀ (B : Finset ℕ), isValidSet B → B.card ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_isosceles_set_l61_6165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_network_inequality_l61_6131

/-- Represents the number of paths between ports in a 6-port network -/
def f (i j : Fin 6) : ℕ := sorry

/-- The inequality holds for the given network configuration -/
theorem network_inequality :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_network_inequality_l61_6131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_equals_binomial_difference_l61_6150

def y (n : ℕ) : ℕ → ℚ
  | 0 => 1
  | 1 => n + 1
  | (k + 2) => ((n + 2) * y n (k + 1) - (n + 1 - k) * y n k) / (k + 2)

theorem sum_y_equals_binomial_difference (n : ℕ) :
  (Finset.range (n + 2)).sum (y n) = (2 : ℚ)^(2*n + 1) - (Nat.choose (2*n + 2) (n + 2) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_equals_binomial_difference_l61_6150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_region_l61_6188

/-- The volume of the solid generated by rotating the region bounded by y = x^2 - x and y = x
    about the line y = x -/
theorem volume_of_rotated_region : ∃ v : ℝ, v = (16 * Real.pi * Real.sqrt 2) / 15 := by
  -- Define the lower bound of the region
  let lower_bound (x : ℝ) := x^2 - x
  
  -- Define the upper bound of the region
  let upper_bound (x : ℝ) := x
  
  -- Define the line of rotation
  let rotation_line (x : ℝ) := x
  
  -- Define the integration limits
  let a : ℝ := 0
  let b : ℝ := 2

  -- State the volume formula using cylindrical shells method
  let volume_formula : ℝ := ∫ x in a..b, 2 * Real.pi * (((x - rotation_line x)^2 + (upper_bound x - lower_bound x)^2).sqrt / Real.sqrt 2) * (upper_bound x - lower_bound x)

  -- Assert that the volume is equal to 16π√2 / 15
  have volume_value : volume_formula = (16 * Real.pi * Real.sqrt 2) / 15 := by sorry

  -- Return the existence of the volume
  exact ⟨volume_formula, volume_value⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_region_l61_6188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l61_6151

def is_arithmetic (seq : List ℕ) : Prop :=
  ∃ d, ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = d

theorem arithmetic_sequence_sum (seq : List ℕ) :
  seq.length = 6 →
  seq[0]! = 3 →
  seq[1]! = 8 →
  seq[2]! = 13 →
  seq[5]! = 33 →
  is_arithmetic seq →
  seq[3]! + seq[4]! = 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l61_6151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l61_6177

/-- The function f(x) = 2/x + ln(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem f_properties :
  (∃! x : ℝ, x > 0 ∧ f x = x) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ > x₂ → f x₁ = f x₂ → x₁ + x₂ > 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l61_6177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l61_6111

def S (n : ℕ) : ℕ := n^2 + 3*n + 5

def a : ℕ → ℕ
  | 0 => 9  -- Adding this case to cover Nat.zero
  | 1 => 9
  | n+2 => 2*(n+2) + 2

theorem sequence_formula (n : ℕ) : 
  (n = 0 ∨ n = 1 ∧ a n = 9) ∨ 
  (n > 1 ∧ a n = 2*n + 2) ∧ 
  S n = n^2 + 3*n + 5 ∧
  (∀ k : ℕ, k > 1 → S k - S (k-1) = a k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l61_6111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rectangles_correct_l61_6139

/-- The number of rectangles with sides parallel to the axes and vertices
    at integer coordinates (a, b) where 0 ≤ a, b ≤ n -/
def count_rectangles (n : ℕ) : ℕ :=
  ((n + 1)^2 * n^2) / 4

/-- The actual number of valid rectangles for a given n -/
def number_of_valid_rectangles (n : ℕ) : ℕ :=
  sorry -- This would be defined based on the problem conditions

/-- Theorem stating that count_rectangles gives the correct number of rectangles -/
theorem count_rectangles_correct (n : ℕ) :
  count_rectangles n = number_of_valid_rectangles n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rectangles_correct_l61_6139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density_ratio_is_1_75_l61_6112

/-- Represents a cube composed of smaller cubes -/
structure Cube where
  smallCubes : Nat
  replacedCubes : Nat
  densityMultiplier : ℚ

/-- Calculates the density ratio of a cube after replacing some of its smaller cubes -/
def densityRatio (c : Cube) : ℚ :=
  let initialMass := c.smallCubes
  let finalMass := (c.smallCubes - c.replacedCubes) + c.replacedCubes * c.densityMultiplier
  finalMass / c.smallCubes

/-- Theorem stating that the density ratio of a specific cube configuration is 1.75 -/
theorem density_ratio_is_1_75 (c : Cube) 
    (h1 : c.smallCubes = 8)
    (h2 : c.replacedCubes = 3)
    (h3 : c.densityMultiplier = 3) :
    densityRatio c = 7/4 := by
  sorry

/-- Compute the density ratio for the given cube configuration -/
def computeDensityRatio : ℚ :=
  densityRatio { smallCubes := 8, replacedCubes := 3, densityMultiplier := 3 }

#eval computeDensityRatio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_density_ratio_is_1_75_l61_6112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l61_6138

-- Define the train length in meters
noncomputable def train_length : ℝ := 120

-- Define the train speed in kilometers per hour
noncomputable def train_speed_kmph : ℝ := 60

-- Convert km/h to m/s
noncomputable def speed_to_ms (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

-- Calculate the time needed for the train to pass the pole
noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

-- Theorem statement
theorem train_passing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧
  |time_to_pass train_length (speed_to_ms train_speed_kmph) - 7.2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l61_6138
