import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_l938_93865

/-- Represents a cylindrical boiler -/
structure Boiler where
  volume : ℝ
  base_cost : ℝ
  lateral_cost : ℝ

/-- The cost function for the boiler -/
noncomputable def cost (b : Boiler) (d h : ℝ) : ℝ :=
  2 * Real.pi * (d / 2)^2 * b.base_cost + Real.pi * d * h * b.lateral_cost

/-- The volume constraint for the boiler -/
def volume_constraint (b : Boiler) (d h : ℝ) : Prop :=
  b.volume = Real.pi * (d / 2)^2 * h

/-- The optimal ratio of base diameter to height minimizes cost -/
theorem optimal_ratio (b : Boiler) :
  ∃ (d h : ℝ), d > 0 ∧ h > 0 ∧ volume_constraint b d h ∧
  (∀ (d' h' : ℝ), d' > 0 → h' > 0 → volume_constraint b d' h' →
    cost b d h ≤ cost b d' h') ∧
  d / h = b.lateral_cost / b.base_cost := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_l938_93865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_catch_in_four_steps_l938_93878

-- Define the grid positions
inductive Position : Type
| A | B | C | D | E | F

-- Define the game state
structure GameState where
  policeman : Position
  thief : Position
  turn : Nat

-- Define valid moves
def validMove : Position → Position → Prop
| Position.A, Position.C => True
| Position.C, Position.A => True
| Position.C, Position.D => True
| Position.D, Position.C => True
| Position.D, Position.A => True
| Position.B, Position.E => True
| Position.E, Position.B => True
| Position.E, Position.F => True
| Position.F, Position.E => True
| Position.F, Position.C => True
| _, _ => False

-- Define a game step
def gameStep (state : GameState) (newPolicePos newThiefPos : Position) : GameState :=
  { policeman := newPolicePos,
    thief := newThiefPos,
    turn := state.turn + 1 }

-- Define the winning condition
def policeCaught (state : GameState) : Prop :=
  state.policeman = state.thief

-- The main theorem
theorem police_catch_in_four_steps :
  ∃ (strategy : GameState → Position),
    ∀ (thiefStrategy : GameState → Position),
      ∃ (finalState : GameState),
        finalState.turn ≤ 4 ∧
        policeCaught finalState ∧
        (∀ (state : GameState) (newPolicePos newThiefPos : Position),
          state.turn < 4 →
          validMove state.policeman newPolicePos →
          validMove state.thief newThiefPos →
          finalState = gameStep state newPolicePos newThiefPos) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_catch_in_four_steps_l938_93878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_negative_6_l938_93838

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We need to handle the case for 0
  | 1 => 3
  | 2 => 6
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_2009_equals_negative_6 : sequence_a 2009 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_negative_6_l938_93838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_comparison_breakeven_point_l938_93899

/-- Represents the cost calculation for a family trip -/
def family_trip_cost (m : ℝ) (x : ℕ) : (ℝ × ℝ) :=
  let cost_A := 0.5 * m * (x : ℝ) + 1.5 * m
  let cost_B := 0.6 * m * ((x : ℝ) + 2)
  (cost_A, cost_B)

/-- Theorem stating the relationship between costs and number of children -/
theorem cost_comparison (m : ℝ) (h_m : m > 0) :
  ∃ x₀ : ℕ, x₀ = 3 ∧
  (∀ x < x₀, (family_trip_cost m x).1 > (family_trip_cost m x).2) ∧
  (∀ x > x₀, (family_trip_cost m x).1 < (family_trip_cost m x).2) ∧
  ((family_trip_cost m x₀).1 = (family_trip_cost m x₀).2) :=
by
  sorry

/-- Corollary stating that 3 is the breakeven point -/
theorem breakeven_point (m : ℝ) (h_m : m > 0) :
  (family_trip_cost m 3).1 = (family_trip_cost m 3).2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_comparison_breakeven_point_l938_93899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l938_93805

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

-- State the theorem
theorem f_inequality_implies_a_range (a : ℝ) :
  f (2 - a^2) > f a → -2 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l938_93805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_xz_plane_l938_93853

theorem equidistant_point_xz_plane :
  let p : Fin 3 → ℝ := ![15/8, 0, 5/8]
  let p1 : Fin 3 → ℝ := ![1, -2, 0]
  let p2 : Fin 3 → ℝ := ![2, 2, 3]
  let p3 : Fin 3 → ℝ := ![3, 3, -2]
  let distance (a b : Fin 3 → ℝ) := 
    Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2 + (a 2 - b 2)^2)
  (distance p p1 = distance p p2) ∧ 
  (distance p p1 = distance p p3) ∧
  (p 0 + 2 * p 2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_xz_plane_l938_93853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l938_93832

theorem tan_double_angle (α : ℝ) :
  Real.tan (α + π / 4) = Real.sqrt 3 - 2 → Real.tan (2 * α) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l938_93832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_zero_sum_intercepts_line_equation_with_specific_triangle_area_l938_93806

/-- A line passing through point (2, 3) -/
structure Line where
  slope : ℝ
  intercept : ℝ
  passes_through_point : slope * 2 + intercept = 3

/-- The sum of x-intercept and y-intercept of a line -/
noncomputable def sum_of_intercepts (l : Line) : ℝ :=
  -l.intercept / l.slope - l.intercept

/-- The area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangle_area (l : Line) : ℝ :=
  (l.intercept / l.slope * l.intercept) / 2

theorem line_equation_with_zero_sum_intercepts (l : Line) :
  sum_of_intercepts l = 0 →
  (∀ x y, l.slope * x + l.intercept = y ↔ (3 * x - 2 * y = 0 ∨ x - y + 1 = 0)) :=
by sorry

theorem line_equation_with_specific_triangle_area (l : Line) :
  triangle_area l = 16 →
  (∀ x y, l.slope * x + l.intercept = y ↔ (x + 2 * y - 8 = 0 ∨ 9 * x + 2 * y - 24 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_zero_sum_intercepts_line_equation_with_specific_triangle_area_l938_93806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_ratio_l938_93885

/-- A rectangular prism with an inscribed sphere -/
structure PrismWithSphere where
  width : ℝ
  length : ℝ
  height : ℝ
  sphere_radius : ℝ
  width_positive : 0 < width
  length_eq_twice_width : length = 2 * width
  height_eq_twice_width : height = 2 * width
  sphere_diameter_eq_width : 2 * sphere_radius = width

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a rectangular prism -/
def prism_volume (p : PrismWithSphere) : ℝ := p.length * p.width * p.height

/-- The ratio of the volume of the inscribed sphere to the volume of the rectangular prism -/
noncomputable def volume_ratio (p : PrismWithSphere) : ℝ :=
  sphere_volume p.sphere_radius / prism_volume p

theorem inscribed_sphere_volume_ratio (p : PrismWithSphere) :
  volume_ratio p = Real.pi / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_ratio_l938_93885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l938_93823

/-- Represents a circle with parameter t -/
def circle_eq (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x + 2*(1-4*t^2)*y + 16*t^4 + 9 = 0

/-- The range of t for which the circle is defined -/
def t_range : Set ℝ := {t | -1/7 < t ∧ t < 1}

/-- The equation of the circle with the largest area -/
def largest_circle (x y : ℝ) : Prop :=
  (x - 24/7)^2 + (y + 13/49)^2 = 16/7

/-- The range of t when point P(3,4t^2) is always inside the circle -/
def t_range_inside : Set ℝ := {t | 0 < t ∧ t < 3/4}

/-- Point P as a function of t -/
def point_P (t : ℝ) : ℝ × ℝ := (3, 4*t^2)

theorem circle_properties :
  ∀ t : ℝ,
  (∃ x y : ℝ, circle_eq t x y) →
  (t ∈ t_range) ∧
  (∃ x y : ℝ, largest_circle x y) ∧
  (∀ t' ∈ t_range_inside, ∃ x y : ℝ, circle_eq t' x y ∧ point_P t' ∈ {p : ℝ × ℝ | ∃ x y : ℝ, circle_eq t' x y ∧ p = (x, y)}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l938_93823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_area_of_triangle_l938_93824

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 2 * (Real.cos x) ^ 2

-- Part I: Range of f(x)
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (1 / 2) 2 := by sorry

-- Part II: Area of triangle ABC
theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) :
  f A = 3 / 2 →
  Real.sqrt 2 * a = Real.sqrt 3 * b →
  c = 1 + Real.sqrt 3 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  (1 / 2) * b * c * Real.sin A = (3 + Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_area_of_triangle_l938_93824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_problem_l938_93826

/-- The theorem statement for the polynomial division problem -/
theorem polynomial_division_problem : 
  ∃ (Q R : ℂ → ℂ), 
    (∀ z : ℂ, z^2023 + 1 = (z^2 + 1) * Q z + R z) ∧ 
    (∃ (a b : ℂ), ∀ z, R z = a * z + b) ∧
    (∀ z, R z = z + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_problem_l938_93826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93855

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_properties :
  (∀ x, f x = x / (x^2 + 1)) →
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
  (∀ x, -1/2 ≤ f x ∧ f x ≤ 1/2) ∧
  (∃ x₁ x₂, f x₁ = -1/2 ∧ f x₂ = 1/2) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_two_thirds_l938_93880

/-- The probability that the sum of two randomly selected numbers from [0,1] is less than or equal to 2/3 -/
theorem probability_sum_less_than_two_thirds :
  let X : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
  let E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 ≤ 2/3} ∩ X
  (MeasureTheory.volume E) / (MeasureTheory.volume X) = 2/9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_two_thirds_l938_93880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l938_93852

/-- Given an acute triangle ABC with sin A = (4√5)/9 and side a = 4 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  0 < A ∧ A < Real.pi/2 →  -- A is acute
  0 < B ∧ B < Real.pi/2 →  -- B is acute
  0 < C ∧ C < Real.pi/2 →  -- C is acute
  Real.sin A = (4 * Real.sqrt 5) / 9 →
  a = 4 →
  a = b * Real.sin C →  -- Law of sines
  a = c * Real.sin B →  -- Law of sines
  b = c * Real.sin A →  -- Law of sines
  (
    Real.sin (2 * (B + C)) + (Real.sin ((B + C) / 2))^2 = (45 - 8 * Real.sqrt 5) / 81 ∧
    (1/2 * b * c * Real.sin A) ≤ 2 * Real.sqrt 5
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l938_93852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93849

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (2*x^2 + x) * Real.log x - (2*a + 1)*x^2 - (a + 1)*x + b

-- State the theorem
theorem f_properties (a b : ℝ) :
  -- Part 1: Monotonicity when a = 1
  (∀ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < Real.exp 1 ∧ x₂ < Real.exp 1 ∧ x₁ < x₂ → f 1 b x₁ > f 1 b x₂) ∧
  (∀ x₁ x₂, x₁ > Real.exp 1 ∧ x₂ > Real.exp 1 ∧ x₁ < x₂ → f 1 b x₁ < f 1 b x₂) ∧
  -- Part 2: Minimum value of b-a when f(x) ≥ 0
  ((∀ x, x > 0 → f a b x ≥ 0) → b - a ≥ 3/4 + Real.log 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93847

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

-- Theorem statement
theorem f_properties (a : ℝ) :
  (∀ x, x > 0 → (0 < a ∧ a < Real.exp 1) → f a x ≠ 0) ∧
  (a ≤ 0 → ∃! x, x > 0 ∧ f a x = 0) ∧
  (∀ x, x > 1 → f a x ≥ a * x^a * Real.log x - x * Real.exp x → a < Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_P_l938_93844

/-- Given an angle α whose terminal side passes through point P(t, 2t) where t > 0, 
    prove that cosα = √5/5 -/
theorem cos_alpha_for_point_P (t : ℝ) (h : t > 0) :
  let α := Real.arctan (2)
  Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_P_l938_93844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_4_5_l938_93870

-- Define the base triangle
def base_triangle (hypotenuse : ℝ) (acute_angle : ℝ) : Prop :=
  hypotenuse = 6 ∧ acute_angle = 15 * Real.pi / 180

-- Define the lateral edge inclination
def lateral_edge_inclination (angle : ℝ) : Prop :=
  angle = 45 * Real.pi / 180

-- Define the volume of the pyramid
noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1/3) * base_area * height

-- Theorem statement
theorem pyramid_volume_is_4_5 
  (hypotenuse : ℝ)
  (acute_angle : ℝ)
  (inclination : ℝ)
  (h1 : base_triangle hypotenuse acute_angle)
  (h2 : lateral_edge_inclination inclination) :
  ∃ (base_area height : ℝ),
    pyramid_volume base_area height = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_4_5_l938_93870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_is_composite_l938_93842

/-- Represents the number formed by n ones, followed by a 2, followed by n ones -/
def special_number (n : ℕ) : ℕ :=
  let ones := (10^(n+1) - 1) / 9  -- This represents n+1 ones in decimal
  ones * 10^n + 2 * 10^n + ones

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k, 1 < k ∧ k < m ∧ m % k = 0

theorem special_number_is_composite (n : ℕ) (h : 0 < n) : 
  is_composite (special_number n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_is_composite_l938_93842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_problem_l938_93802

/-- The amount of alloy B mixed with alloy A -/
def alloy_B_amount : ℝ → Prop := sorry

/-- The ratio of lead to tin in alloy A -/
def alloy_A_ratio : ℝ → ℝ → Prop := sorry

/-- The ratio of tin to copper in alloy B -/
def alloy_B_ratio : ℝ → ℝ → Prop := sorry

/-- The amount of tin in the new alloy -/
def new_alloy_tin_amount : ℝ → Prop := sorry

theorem alloy_mixture_problem (alloy_A_amount : ℝ) :
  alloy_A_amount = 60 →
  alloy_A_ratio 3 2 →
  alloy_B_ratio 1 4 →
  new_alloy_tin_amount 44 →
  alloy_B_amount 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_problem_l938_93802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l938_93817

/-- A plane in 3D space -/
structure Plane where
  -- Define plane properties here

/-- A point in 3D space -/
structure Point where
  -- Define point properties here

/-- A line in 3D space -/
structure Line where
  -- Define line properties here

/-- Defines the relationship between two lines -/
inductive LineRelationship
  | Skew
  | Intersecting

/-- Checks if a point is inside a plane -/
def Point.insidePlane (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Checks if a point is outside a plane -/
def Point.outsidePlane (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Creates a line connecting two points -/
def Line.connectPoints (p1 p2 : Point) : Line :=
  sorry

/-- Checks if a line is within a plane -/
def Line.withinPlane (l : Line) (plane : Plane) : Prop :=
  sorry

/-- The main theorem -/
theorem line_plane_relationship (plane : Plane) (p1 p2 : Point) (l : Line) :
  p1.insidePlane plane →
  p2.outsidePlane plane →
  l.withinPlane plane →
  let l_connect := Line.connectPoints p1 p2
  (LineRelationship.Skew : LineRelationship) = LineRelationship.Skew ∨ 
  (LineRelationship.Intersecting : LineRelationship) = LineRelationship.Intersecting :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l938_93817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_triangle_area_l938_93860

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_vertex : b = 1
  h_eccentricity : (a^2 - b^2).sqrt / a = Real.sqrt 3 / 2

/-- Point on the ellipse -/
def point_on_ellipse (e : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Left focus of the ellipse -/
noncomputable def left_focus (e : SpecialEllipse) : ℝ × ℝ :=
  (-(e.a^2 - e.b^2).sqrt, 0)

/-- Slope of a line -/
noncomputable def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (y₂ - y₁) / (x₂ - x₁)

/-- Area of a triangle given two side lengths and the angle between them -/
noncomputable def triangle_area (side1 side2 angle : ℝ) : ℝ :=
  1/2 * side1 * side2 * Real.sin angle

/-- Main theorem -/
theorem special_ellipse_triangle_area (e : SpecialEllipse) (x y : ℝ) :
  point_on_ellipse e x y →
  y < 0 →
  line_slope x y (left_focus e).1 (left_focus e).2 = Real.tan (π/6) →
  ∃ (x' y' : ℝ), point_on_ellipse e x' y' ∧ 
    triangle_area (2 * (e.a^2 - e.b^2).sqrt) (-y) (π/2) = Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_triangle_area_l938_93860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_ratio_is_one_half_l938_93897

/-- Represents the distribution of gummy bear candies -/
structure CandyDistribution where
  initial : ℕ
  siblings : ℕ
  candies_per_sibling : ℕ
  to_eat : ℕ
  remaining : ℕ

/-- Calculates the ratio of candies given to best friend to candies left after giving to siblings -/
def candy_ratio (d : CandyDistribution) : ℚ :=
  let given_to_siblings := d.siblings * d.candies_per_sibling
  let left_after_siblings := d.initial - given_to_siblings
  let to_best_friend := left_after_siblings - d.to_eat - d.remaining
  ↑to_best_friend / ↑left_after_siblings

/-- Theorem stating the ratio of candies given to best friend to candies left after giving to siblings -/
theorem candy_ratio_is_one_half (d : CandyDistribution) 
  (h1 : d.initial = 100)
  (h2 : d.siblings = 3)
  (h3 : d.candies_per_sibling = 10)
  (h4 : d.to_eat = 16)
  (h5 : d.remaining = 19) : 
  candy_ratio d = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_ratio_is_one_half_l938_93897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l938_93873

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x else (2 - 3 * a) * x + 1

theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) →
  a ∈ Set.Ioo (2/3) (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l938_93873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_theorem_l938_93840

-- Define the function g
def g : Fin 5 → Fin 5
| 0 => 3  -- represents g(1) = 4
| 1 => 4  -- represents g(2) = 5
| 2 => 0  -- represents g(3) = 1
| 3 => 2  -- represents g(4) = 3
| 4 => 1  -- represents g(5) = 2

-- Assume g is bijective (has an inverse)
axiom g_bijective : Function.Bijective g

-- Define g_inv as the inverse of g
noncomputable def g_inv : Fin 5 → Fin 5 := Function.invFun g

-- State the theorem
theorem inverse_composition_theorem :
  g_inv (g_inv (g_inv 3)) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_theorem_l938_93840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_commutative_identity_l938_93816

-- Define a non-commutative algebraic system
class NonCommutativeSystem (α : Type*) extends Add α, Mul α, One α, Inv α where
  -- Addition is associative and commutative
  add_assoc : ∀ a b c : α, (a + b) + c = a + (b + c)
  add_comm : ∀ a b : α, a + b = b + a
  -- Multiplication is associative but not necessarily commutative
  mul_assoc : ∀ a b c : α, (a * b) * c = a * (b * c)
  -- Distributive laws
  left_distrib : ∀ a b c : α, a * (b + c) = a * b + a * c
  right_distrib : ∀ a b c : α, (a + b) * c = a * c + b * c
  -- Identity element
  one_mul : ∀ a : α, 1 * a = a
  mul_one : ∀ a : α, a * 1 = a
  -- Inverse element
  mul_inv_cancel_left : ∀ a : α, a⁻¹ * a = 1
  mul_inv_cancel_right : ∀ a : α, a * a⁻¹ = 1

-- State the theorem
theorem non_commutative_identity {α : Type*} [NonCommutativeSystem α] (a b : α) :
  (a + a * b⁻¹ * a)⁻¹ + (a + b)⁻¹ = a⁻¹ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_commutative_identity_l938_93816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_generating_function_set_l938_93813

/-- An infinite sequence of polynomials -/
noncomputable def PolynomialSequence : ℕ → (ℝ → ℝ) := sorry

/-- A finite set of functions -/
noncomputable def FunctionSet : Finset (ℝ → ℝ) := sorry

/-- Composition of functions from the FunctionSet -/
noncomputable def ComposeFrom (fs : Finset (ℝ → ℝ)) : List (ℝ → ℝ) → (ℝ → ℝ)
  | [] => id
  | (f :: rest) => f ∘ (ComposeFrom fs rest)

/-- Theorem: For any infinite sequence of polynomials, there exists a finite set of functions
    whose compositions can generate any polynomial in the sequence -/
theorem exists_generating_function_set :
  ∃ (fs : Finset (ℝ → ℝ)), ∀ (n : ℕ), ∃ (comp : List (ℝ → ℝ)), 
    (∀ f ∈ comp, f ∈ fs) ∧ ComposeFrom fs comp = PolynomialSequence n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_generating_function_set_l938_93813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_points_l938_93808

theorem third_place_points : 
  ∃! x : ℕ, 
    x > 0 ∧ 
    (∃ (a b c d : ℕ), 
      a + b + c + d = 7 ∧ 
      11^a * 7^b * x^c * 2^d = 38500) ∧ 
    x = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_points_l938_93808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l938_93804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then
    -1/3 * x^3 + (1-a)/2 * x^2 + a*x - 4/3
  else
    (a-1) * Real.log x + 1/2 * x^2 - a*x

theorem monotonic_f_implies_a_range (a : ℝ) (h1 : a > 0) :
  (∀ x y, -a < x ∧ x < y ∧ y < 2*a → f a x ≤ f a y) →
  0 < a ∧ a ≤ 10/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l938_93804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_do_not_interfere_l938_93829

-- Define the speeds and distances
noncomputable def speed_car1 : ℝ := 60
noncomputable def speed_car2 : ℝ := speed_car1 * (1 + 1/5)
noncomputable def distance_car1 : ℝ := 120
noncomputable def distance_car2 : ℝ := 180
noncomputable def bridge_length : ℝ := 2

-- Define the time it takes for each car to reach the bridge
noncomputable def time_to_bridge_car1 : ℝ := distance_car1 / speed_car1
noncomputable def time_to_bridge_car2 : ℝ := distance_car2 / speed_car2

-- Define the distance traveled by car1 when car2 reaches the bridge
noncomputable def distance_traveled_car1 : ℝ := speed_car1 * time_to_bridge_car2

-- Theorem stating that car1 will have crossed the bridge before car2 arrives
theorem cars_do_not_interfere : 
  distance_traveled_car1 > distance_car1 + bridge_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_do_not_interfere_l938_93829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_independent_of_x_l938_93867

open Real

theorem f_independent_of_x (θ : ℝ) :
  let f (x : ℝ) := (cos (x - θ))^2 + (cos x)^2 - 2 * cos θ * cos (x - θ) * cos x
  ∀ x : ℝ, f x = (sin θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_independent_of_x_l938_93867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l938_93822

-- Define the lines
noncomputable def line1 (x y : ℝ) : Prop := x - y + 2 = 0
noncomputable def line2 (x y : ℝ) : Prop := 2*x + y + 1 = 0
noncomputable def line3 (x y : ℝ) : Prop := x - 3*y + 2 = 0

-- Define the intersection point of line1 and line2
noncomputable def intersection : ℝ × ℝ := (-1, 1)

-- Define the slope of line3
noncomputable def slope_line3 : ℝ := 1/3

-- Define the slope of line l (perpendicular to line3)
noncomputable def slope_l : ℝ := -3

-- Define the equation of line l
noncomputable def line_l (x y : ℝ) : Prop := 3*x + y + 2 = 0

-- Theorem statement
theorem line_l_equation :
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = intersection) →
  (slope_l = -1 / slope_line3) →
  (∀ x y : ℝ, line_l x y ↔ y - intersection.2 = slope_l * (x - intersection.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l938_93822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_term_l938_93843

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

noncomputable def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ) * (a 1 + a n) / 2

theorem smallest_absolute_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1_pos : a 1 > 0)
  (h_S12_pos : sum_of_terms a 12 > 0)
  (h_S13_neg : sum_of_terms a 13 < 0) :
  ∀ k ∈ Finset.range 13, |a 7| ≤ |a (k + 1)| :=
by
  sorry

#check smallest_absolute_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_term_l938_93843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l938_93839

theorem contrapositive_equivalence :
  (∀ (x y : ℝ), x * y = 0 → x = 0) ↔ (∀ (x y : ℝ), x ≠ 0 → x * y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l938_93839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sin_l938_93863

theorem integral_x_squared_plus_sin : ∫ (x : ℝ) in (-1)..1, x^2 + Real.sin x = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sin_l938_93863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l938_93874

/-- 
A line y = kx + b is tangent to a curve y = f(x) at point x if:
1. The line passes through the point (x, f(x))
2. The slope of the line equals the derivative of f at x
-/
def is_tangent_line (k b : ℝ) (f : ℝ → ℝ) (x : ℝ) : Prop :=
  k * x + b = f x ∧ k = deriv f x

theorem tangent_line_value (k b : ℝ) :
  (∃ x₁ > 0, is_tangent_line k b (fun x ↦ Real.log x + 2) x₁) →
  (∃ x₂ > -1, is_tangent_line k b (fun x ↦ Real.log (x + 1)) x₂) →
  b = 1 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l938_93874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_pairs_104_l938_93888

theorem sum_pairs_104 (selected : Finset ℕ) : 
  (∀ n ∈ selected, n ∈ Finset.range 34 ∧ n % 3 = 1) →
  selected.card = 20 →
  ∃ a b c d, a ∈ selected ∧ b ∈ selected ∧ c ∈ selected ∧ d ∈ selected ∧
             a ≠ c ∧ b ≠ d ∧ a + b = 104 ∧ c + d = 104 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_pairs_104_l938_93888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_function_extrema_l938_93833

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/6) * x^3 - (1/2) * a * x^2 + x

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x + 1

-- Define the second derivative of f
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := x - a

-- State the theorem
theorem convex_function_extrema (a : ℝ) (h1 : a ≤ 2) :
  (∀ x ∈ Set.Ioo (-1) 2, f'' a x < 0) →
  (∃ x ∈ Set.Ioo (-1) 2, ∀ y ∈ Set.Ioo (-1) 2, f a y ≤ f a x) ∧
  (∀ x ∈ Set.Ioo (-1) 2, ∃ y ∈ Set.Ioo (-1) 2, f a y < f a x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_function_extrema_l938_93833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l938_93892

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Always true for integers

def line_equation (m : ℚ) (x : ℚ) : ℚ :=
  m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℤ, 0 < x → x ≤ 150 →
    ¬∃ y : ℤ, is_lattice_point x y ∧ ↑y = line_equation m ↑x

theorem max_a_value :
  (∀ m : ℚ, 1/3 < m → m < 51/151 → no_lattice_points m) ∧
  (∀ ε > 0, ∃ m : ℚ, 51/151 < m ∧ m < 51/151 + ε ∧ ¬no_lattice_points m) :=
by
  sorry

#check max_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l938_93892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_squared_div_simplified_l938_93800

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

-- Theorem for part I
theorem z_purely_imaginary : z (-1/2) = Complex.I * (((-1/2)^2 - 3 * (-1/2) + 2) : ℝ) := by
  sorry

-- Theorem for part II
theorem z_squared_div_simplified : 
  (z 0)^2 / (z 0 + 5 + 2 * Complex.I) = -32/25 - 24/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_squared_div_simplified_l938_93800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_A_l938_93809

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Two points are parallel to the y-axis if their x-coordinates are equal -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

theorem coordinates_of_A (b : Point) (a : Point) 
    (h1 : b.x = 2 ∧ b.y = 1)  -- B has coordinates (2, 1)
    (h2 : parallelToYAxis a b)  -- AB is parallel to y-axis
    (h3 : distance a b = 4)  -- Length of AB is 4
    : (a.x = 2 ∧ a.y = -3) ∨ (a.x = 2 ∧ a.y = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_A_l938_93809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l938_93877

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * t, t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (4 / (1 + 3 * Real.sin θ ^ 2))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry

noncomputable def B : ℝ × ℝ := sorry

-- Define point E as the intersection of C₁ with the x-axis
def E : ℝ × ℝ := (1, 0)

-- State the theorem
theorem intersection_distance_sum : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d E A + d E B = 2 * Real.sqrt 19 / 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l938_93877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_cut_l938_93895

theorem pyramid_volume_cut (base_edge slant_edge cut_height : ℝ) :
  base_edge = 12 →
  slant_edge = 15 →
  cut_height = 4 →
  let original_height := Real.sqrt 153
  let new_height := original_height - cut_height
  let volume := (1/3) * (base_edge^2) * ((original_height - cut_height) / original_height)^3
  volume = (1/3) * 144 * ((Real.sqrt 153 - 4) / Real.sqrt 153)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_cut_l938_93895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_solution_set_f_positive_l938_93835

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((20 / (x + 10)) + a)

-- Define the property of being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Theorem 1: If f is an odd function, then a = -1
theorem odd_function_implies_a_eq_neg_one (a : ℝ) :
  is_odd_function (f a) → a = -1 := by sorry

-- Theorem 2: The solution set of f(x) > 0 is {x | -10 < x < 0}
theorem solution_set_f_positive (x : ℝ) :
  f (-1) x > 0 ↔ -10 < x ∧ x < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_solution_set_f_positive_l938_93835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_y_axis_l938_93825

theorem symmetry_y_axis (θ : ℝ) : 
  (Real.cos θ = -Real.cos (θ + π/6) ∧ Real.sin θ = Real.sin (θ + π/6)) → 
  θ = 5*π/12 ∨ θ = 5*π/12 + π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_y_axis_l938_93825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l938_93884

/-- Triangle DEF with sides DE = EF = 26 and DF = 38 -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  h_DE_EF : DE = EF
  h_DE : DE = 26
  h_DF : DF = 38

/-- The semi-perimeter of a triangle -/
noncomputable def semiPerimeter (t : Triangle) : ℝ := (t.DE + t.EF + t.DF) / 2

/-- The area of a triangle using Heron's formula -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let s := semiPerimeter t
  Real.sqrt (s * (s - t.DE) * (s - t.EF) * (s - t.DF))

/-- Theorem: The area of triangle DEF is equal to √(s(s-a)(s-b)(s-c)) -/
theorem area_of_triangle_DEF (t : Triangle) : 
  triangleArea t = Real.sqrt 114345 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l938_93884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_senate_committee_seating_arrangements_l938_93861

-- Define the number of Democrats and Republicans
def num_democrats : ℕ := 6
def num_republicans : ℕ := 5

-- Define the total number of politicians
def total_politicians : ℕ := num_democrats + num_republicans

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem senate_committee_seating_arrangements :
  factorial (total_politicians - 1) = 3628800 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_senate_committee_seating_arrangements_l938_93861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_distance_l938_93896

/-- The value of 'a' for a hyperbola with specific properties -/
theorem hyperbola_asymptote_distance (a : ℝ) : a > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 = 1 → 
    ∃ (m : ℝ), (|2 - m * 0|) / Real.sqrt (1 + m^2) = 1) →
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_distance_l938_93896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_floor_div_100_l938_93819

noncomputable def M : ℚ := (2 * Nat.factorial 17) * (
  1 / (Nat.factorial 3 * Nat.factorial 16) +
  1 / (Nat.factorial 4 * Nat.factorial 15) +
  1 / (Nat.factorial 5 * Nat.factorial 14) +
  1 / (Nat.factorial 6 * Nat.factorial 13) +
  1 / (Nat.factorial 7 * Nat.factorial 12) +
  1 / (Nat.factorial 8 * Nat.factorial 11) +
  1 / (Nat.factorial 9 * Nat.factorial 10) +
  1 / (Nat.factorial 10 * Nat.factorial 9)
)

theorem M_floor_div_100 : ⌊M / 100⌋ = 262 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_floor_div_100_l938_93819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_of_divisors_eight_l938_93864

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun i => n % i = 0) (Finset.range (n + 1))).sum id

/-- Theorem: There exists a positive integer A such that the sum of its divisors is 8 -/
theorem exists_sum_of_divisors_eight : ∃ A : ℕ, A > 0 ∧ sum_of_divisors A = 8 := by
  use 7
  constructor
  · exact Nat.zero_lt_succ 6
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_of_divisors_eight_l938_93864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pairing_exists_l938_93810

/-- A configuration of points on a circle -/
structure CircleConfiguration (n : ℕ) where
  points : Fin (2*n) → ℝ × ℝ
  distinct : ∀ i j, i ≠ j → points i ≠ points j
  on_circle : ∀ i, (points i).1^2 + (points i).2^2 = 1

/-- A pairing of points -/
def Pairing (n : ℕ) := Fin n → Fin (2*n) × Fin (2*n)

/-- Check if a pairing is valid (no intersections) -/
def ValidPairing (n : ℕ) (p : Pairing n) : Prop := 
  ∀ i j, i ≠ j → (p i).1 ≠ (p j).1 ∧ (p i).1 ≠ (p j).2 ∧ 
                 (p i).2 ≠ (p j).1 ∧ (p i).2 ≠ (p j).2

/-- The value of a connection -/
def ConnectionValue (n : ℕ) (a b : Fin (2*n)) : ℕ := 
  Int.natAbs (a.val - b.val)

/-- The sum of connection values for a pairing -/
def PairingSum (n : ℕ) (p : Pairing n) : ℕ := 
  Finset.sum (Finset.univ : Finset (Fin n)) (λ i => ConnectionValue n (p i).1 (p i).2)

/-- The main theorem -/
theorem optimal_pairing_exists (n : ℕ) : 
  ∃ (p : Pairing n), ValidPairing n p ∧ PairingSum n p = n^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pairing_exists_l938_93810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_perimeter_27_all_triangles_counted_l938_93831

/-- An isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  base : ℕ
  equal_side : ℕ
  is_valid : base < 2 * equal_side

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := t.base + 2 * t.equal_side

/-- The set of all isosceles triangles with integer side lengths and perimeter 27 -/
def isosceles_triangles_perimeter_27 : Set IsoscelesTriangle :=
  {t : IsoscelesTriangle | perimeter t = 27}

/-- Helper function to list all valid isosceles triangles with perimeter 27 -/
def list_isosceles_triangles_perimeter_27 : List IsoscelesTriangle :=
  List.filter
    (fun t => perimeter t = 27)
    [⟨1, 13, by norm_num⟩, ⟨3, 12, by norm_num⟩, ⟨5, 11, by norm_num⟩,
     ⟨7, 10, by norm_num⟩, ⟨9, 9, by norm_num⟩, ⟨11, 8, by norm_num⟩]

theorem count_isosceles_triangles_perimeter_27 :
  List.length list_isosceles_triangles_perimeter_27 = 6 := by
  rfl

theorem all_triangles_counted :
  ∀ t ∈ isosceles_triangles_perimeter_27, t ∈ list_isosceles_triangles_perimeter_27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_perimeter_27_all_triangles_counted_l938_93831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_with_same_side_as_negative_100_l938_93889

-- Define a function to normalize an angle to the range [0, 360)
noncomputable def normalizeAngle (angle : ℝ) : ℝ :=
  (angle % 360 + 360) % 360

-- Define a function to check if two angles have the same terminal side
def sameSide (angle1 angle2 : ℝ) : Prop :=
  normalizeAngle angle1 = normalizeAngle angle2

-- Theorem statement
theorem angle_with_same_side_as_negative_100 :
  ∃ (angle : ℝ), 0 ≤ angle ∧ angle < 360 ∧ sameSide angle (-100) ∧ angle = 260 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_with_same_side_as_negative_100_l938_93889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_l938_93812

noncomputable def angle : ℝ := Real.pi / 4  -- 45° in radians

def scale_factor : ℝ := 3

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor * Real.cos angle, -scale_factor * Real.sin angle],
    ![scale_factor * Real.sin angle, scale_factor * Real.cos angle]]

theorem det_S : Matrix.det S = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_l938_93812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_half_angle_relations_l938_93848

/-- Theorem about trigonometric functions of half-angles in a triangle -/
theorem triangle_half_angle_relations (α β γ r R p : ℝ) : 
  (α + β + γ = Real.pi) →  -- Sum of angles in a triangle
  (r > 0) →          -- Radius of inscribed circle is positive
  (R > 0) →          -- Radius of circumscribed circle is positive
  (p > 0) →          -- Semi-perimeter is positive
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) = r / (4*R)) ∧
  (Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2) = r / p) ∧
  (Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) = p / (4*R)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_half_angle_relations_l938_93848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_for_all_real_min_value_with_constraint_min_value_achieved_l938_93836

-- Part 1
noncomputable def f (a : ℝ) (x : ℝ) := Real.sqrt (|x + 1| + |x - 2| - a)

theorem f_defined_for_all_real (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ≤ 3 := by sorry

-- Part 2
theorem min_value_with_constraint (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  x^2 + y^2 + z^2 ≥ 1/14 := by sorry

theorem min_value_achieved (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  x^2 + y^2 + z^2 = 1/14 ↔ x = 1/14 ∧ y = 1/7 ∧ z = 3/14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_for_all_real_min_value_with_constraint_min_value_achieved_l938_93836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_trip_distance_l938_93830

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The total distance of Brenda's trip -/
noncomputable def total_distance : ℝ :=
  distance (-3) 6 1 1 + distance 1 1 6 (-3)

theorem brenda_trip_distance :
  total_distance = 2 * Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_trip_distance_l938_93830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_seat_ticket_price_l938_93872

theorem main_seat_ticket_price 
  (total_seats : ℕ) 
  (back_seat_price : ℚ) 
  (total_revenue : ℚ) 
  (back_seats_sold : ℕ) 
  (h1 : total_seats = 20000)
  (h2 : back_seat_price = 45)
  (h3 : total_revenue = 955000)
  (h4 : back_seats_sold = 14500)
  : ↑55 = (total_revenue - (↑back_seats_sold * back_seat_price)) / ↑(total_seats - back_seats_sold) := by
  sorry

#check main_seat_ticket_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_seat_ticket_price_l938_93872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_bet_profit_l938_93866

/-- Calculates the expected profit from a soccer bet with given odds and initial bet -/
def expected_profit (odds1 odds2 odds3 odds4 initial_bet : ℝ) : ℝ :=
  odds1 * odds2 * odds3 * odds4 * initial_bet - initial_bet

/-- Theorem stating the expected profit for the given soccer bet -/
theorem soccer_bet_profit :
  let odds1 : ℝ := 1.28
  let odds2 : ℝ := 5.23
  let odds3 : ℝ := 3.25
  let odds4 : ℝ := 2.05
  let initial_bet : ℝ := 5.00
  abs ((expected_profit odds1 odds2 odds3 odds4 initial_bet) - 212.822) < 0.001 := by
  sorry

#eval expected_profit 1.28 5.23 3.25 2.05 5.00

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_bet_profit_l938_93866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_range_l938_93828

-- Define the ellipse parameters
variable (a b c : ℝ)
variable (h_a_pos : a > 0)
variable (h_b_pos : b > 0)
variable (h_a_gt_b : a > b)

-- Define the eccentricity
variable (h_e : c/a = Real.sqrt 3 / 2)

-- Define the intercepted line segment length
variable (h_intercept : 2 * b^2 / a = Real.sqrt 2)

-- Define the ellipse equation
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the line with slope 1/2
def line (m : ℝ) (x y : ℝ) : Prop := y = 1/2 * x + m

-- State the theorem
theorem y_intercept_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_gt_b : a > b) :
  ∃ (m : ℝ) (A B : ℝ × ℝ),
    ellipse a b A.1 A.2 ∧ 
    ellipse a b B.1 B.2 ∧
    line m A.1 A.2 ∧
    line m B.1 B.2 ∧
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 < 0) →
    m ∈ {x | -Real.sqrt 2 < x ∧ x < 0 ∨ 0 < x ∧ x < Real.sqrt 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_range_l938_93828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_music_length_approx_l938_93876

/-- Represents the jogging scenario with music and without music --/
structure JoggingScenario where
  speed_with_music : ℝ
  speed_without_music : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- Calculates the length of the music played during the jog --/
noncomputable def music_length (scenario : JoggingScenario) : ℝ :=
  let t := (scenario.total_distance - scenario.speed_without_music * scenario.total_time) /
           (scenario.speed_with_music - scenario.speed_without_music)
  t * 60

/-- Theorem stating the length of the music played during the jog --/
theorem music_length_approx (scenario : JoggingScenario)
  (h1 : scenario.speed_with_music = 6)
  (h2 : scenario.speed_without_music = 4)
  (h3 : scenario.total_distance = 6)
  (h4 : scenario.total_time = 70 / 60) :
  ∃ ε > 0, |music_length scenario - 47.14| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_music_length_approx_l938_93876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dig_time_approx_l938_93834

/-- Represents the time taken to dig a hole under certain conditions -/
noncomputable def dig_time (ryan_rate : ℝ) (castel_rate : ℝ) (alex_rate : ℝ)
              (loose_percent : ℝ) (medium_percent : ℝ) (compact_percent : ℝ) : ℝ :=
  1 / ((loose_percent * ryan_rate) + (medium_percent * castel_rate) + (compact_percent * alex_rate))

/-- Theorem stating the approximate time taken to dig the hole together -/
theorem dig_time_approx :
  let ryan_rate : ℝ := 1 / 5
  let castel_rate : ℝ := 1 / 6
  let alex_rate : ℝ := 1 / 8
  let loose_percent : ℝ := 0.3
  let medium_percent : ℝ := 0.4
  let compact_percent : ℝ := 0.3
  abs (dig_time ryan_rate castel_rate alex_rate loose_percent medium_percent compact_percent - 6.09) < 0.01 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dig_time_approx_l938_93834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_a_minus_b_l938_93801

theorem set_equality_implies_a_minus_b (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a + b} → a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_a_minus_b_l938_93801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dubois_speed_ratio_l938_93854

/-- Represents the travel details of Monsieur and Madame Dubois -/
structure TravelDetails where
  madame_driving_time : ℚ
  monsieur_driving_time : ℚ
  madame_stop_time : ℚ
  monsieur_stop_time : ℚ

/-- The ratio of average speeds of Madame and Monsieur Dubois -/
def speed_ratio (t : TravelDetails) : ℚ :=
  t.monsieur_driving_time / t.madame_driving_time

theorem dubois_speed_ratio (t : TravelDetails) 
  (h1 : t.madame_driving_time + t.madame_stop_time = t.monsieur_driving_time + t.monsieur_stop_time)
  (h2 : t.monsieur_stop_time = t.madame_driving_time / 3)
  (h3 : t.madame_stop_time = t.monsieur_driving_time / 4)
  : speed_ratio t = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dubois_speed_ratio_l938_93854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_approximation_l938_93881

/-- Given that 102x = 25 and y - x ≈ 0.9992159937279498, prove that y ≈ 1.2443140329436361 -/
theorem certain_number_approximation (x y : ℝ) 
  (h1 : 102 * x = 25)
  (h2 : abs ((y - x) - 0.9992159937279498) < 1e-10) : 
  abs (y - 1.2443140329436361) < 1e-10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_approximation_l938_93881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_working_time_l938_93821

/-- The rate at which Machine A produces shirts per minute -/
noncomputable def machine_a_rate : ℚ := 3

/-- The rate at which Machine B produces shirts per minute -/
noncomputable def machine_b_rate : ℚ := 5

/-- The total number of shirts produced by both machines -/
noncomputable def total_shirts : ℚ := 54

/-- The time (in minutes) that both machines worked simultaneously -/
noncomputable def working_time : ℚ := total_shirts / (machine_a_rate + machine_b_rate)

theorem machines_working_time :
  working_time = 27/4 := by
  -- Expand the definition of working_time
  unfold working_time
  -- Simplify the fraction
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_working_time_l938_93821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_l938_93811

/-- Represents a warehouse with a supply of computers -/
structure Warehouse where
  supply : ℕ

/-- Represents a school with a demand for computers -/
structure School where
  demand : ℕ

/-- Represents the distance between a warehouse and a school -/
structure Distance where
  value : ℕ

/-- Represents the allocation of computers from a warehouse to schools -/
structure Allocation where
  to_school_a : ℕ
  to_school_b : ℕ
  to_school_c : ℕ

/-- Calculates the total shipping cost given the allocation and distances -/
def total_shipping_cost (a : ℝ) (alloc : Allocation) 
  (dist_a : Distance × Distance × Distance) 
  (dist_b : Distance × Distance × Distance) : ℝ :=
  a * (alloc.to_school_a * dist_a.1.value + 
       alloc.to_school_b * dist_a.2.1.value + 
       alloc.to_school_c * dist_a.2.2.value +
       (9 - alloc.to_school_a) * dist_b.1.value +
       (15 - alloc.to_school_b) * dist_b.2.1.value +
       (8 - alloc.to_school_c) * dist_b.2.2.value)

theorem optimal_allocation 
  (warehouse_a : Warehouse)
  (warehouse_b : Warehouse)
  (school_a : School)
  (school_b : School)
  (school_c : School)
  (dist_a : Distance × Distance × Distance)
  (dist_b : Distance × Distance × Distance)
  (a : ℝ)
  (h1 : warehouse_a.supply = 12)
  (h2 : warehouse_b.supply = 20)
  (h3 : school_a.demand = 9)
  (h4 : school_b.demand = 15)
  (h5 : school_c.demand = 8)
  (h6 : dist_a.1.value = 10 ∧ dist_a.2.1.value = 5 ∧ dist_a.2.2.value = 6)
  (h7 : dist_b.1.value = 4 ∧ dist_b.2.1.value = 8 ∧ dist_b.2.2.value = 15)
  (h8 : a > 0) :
  ∀ (alloc : Allocation), 
    alloc.to_school_a + alloc.to_school_b + alloc.to_school_c ≤ warehouse_a.supply →
    total_shipping_cost a alloc dist_a dist_b ≥ 
    total_shipping_cost a ⟨0, 4, 8⟩ dist_a dist_b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_l938_93811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_ap_zeros_l938_93883

/-- A polynomial of degree 4 with real coefficients -/
structure Polynomial4 where
  a : ℝ
  j : ℝ
  k : ℝ
  constant : ℝ := 100

/-- Four real numbers in arithmetic progression -/
structure ArithmeticProgression4 where
  a : ℝ
  d : ℝ
  seq : Fin 4 → ℝ
  is_ap : ∀ i : Fin 3, seq (i + 1) - seq i = d

/-- The zeros of a polynomial form an arithmetic progression -/
def has_ap_zeros (p : Polynomial4) (ap : ArithmeticProgression4) : Prop :=
  ∀ i : Fin 4, p.a * (ap.seq i)^4 + p.j * (ap.seq i)^2 + p.k * (ap.seq i) + p.constant = 0

theorem polynomial_with_ap_zeros (p : Polynomial4) (ap : ArithmeticProgression4) 
  (h : has_ap_zeros p ap) : p.j = -100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_ap_zeros_l938_93883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l938_93807

theorem subsets_containing_six (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5, 6}) :
  (Finset.filter (λ A => 6 ∈ A) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l938_93807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l938_93858

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The sum of the first n terms of a sequence -/
def PartialSum (a : Sequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => a i)

/-- Arithmetic progression predicate -/
def IsArithmeticProgression (a : Sequence) (start finish : ℕ) : Prop :=
  ∃ d : ℝ, ∀ i ∈ Finset.range (finish - start + 1), a (start + i) = a start + i * d

/-- Geometric progression predicate -/
def IsGeometricProgression (a : Sequence) (start : ℕ) : Prop :=
  ∃ r : ℝ, ∀ i : ℕ, a (start + i + 1) = a (start + i) * r

theorem sequence_existence : 
  ∃ a : Sequence, 
    (∀ k : ℕ, k > 2022 → |PartialSum a k| > |PartialSum a (k + 1)|) ∧
    IsArithmeticProgression a 1 2022 ∧
    IsGeometricProgression a 2022 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l938_93858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_consecutive_probability_l938_93886

def total_art_pieces : ℕ := 12
def escher_prints : ℕ := 4

theorem escher_prints_consecutive_probability :
  (Nat.factorial (total_art_pieces - escher_prints + 1) * Nat.factorial escher_prints) /
  Nat.factorial total_art_pieces = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_consecutive_probability_l938_93886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l938_93820

/-- The constant term in the expansion of (1/x + 2x)^6 -/
def constantTerm : ℕ := 160

/-- The binomial expression (1/x + 2x)^6 -/
noncomputable def binomialExpression (x : ℝ) : ℝ := (1/x + 2*x)^6

/-- Theorem stating that the constant term of the binomial expansion is 160 -/
theorem constant_term_of_binomial_expansion :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = binomialExpression x) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| → |x| < δ → |f x - constantTerm| < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l938_93820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_omega_l938_93871

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem max_value_implies_omega (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), f ω x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 3), f ω x = 1) →
  ω = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_omega_l938_93871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l938_93857

/-- Represents a parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ∀ x y : ℝ, y^2 = 4*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  λ x ↦ x = -1

/-- Distance from a point to the directrix -/
def distanceToDirectrix (p : Parabola) (point : ℝ × ℝ) : ℝ :=
  |point.1 - (-1)|

theorem parabola_properties (p : Parabola) :
  focus p = (1, 0) ∧ distanceToDirectrix p (4, 4) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l938_93857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l938_93803

-- Define the complex numbers
variable (a b c d e f : ℂ)

-- State the given conditions
theorem complex_sum_problem 
  (h1 : b = 2)
  (h2 : e = -2*a - 2*c)
  (h3 : a + b*Complex.I + c + d*Complex.I + e + f*Complex.I = 2 - 2*Complex.I) :
  d + f = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l938_93803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_wins_iff_n_gt_8_l938_93827

/-- The game interval --/
def GameInterval (n : ℕ) := Set.Icc (0 : ℝ) n

/-- A valid move in the game --/
def ValidMove (n : ℕ) (prev_moves : Finset ℝ) (x : ℝ) : Prop :=
  x ∈ GameInterval n ∧ ∀ y ∈ prev_moves, |x - y| > 2

/-- Bela wins the game --/
def BelaWins (n : ℕ) : Prop :=
  ∀ (moves : List ℝ),
    (∀ i : Fin moves.length, ValidMove n (moves.take i.val).toFinset (moves.get i)) →
    moves.length % 2 = 0 →
    ∃ x : ℝ, ValidMove n moves.toFinset x

theorem bela_wins_iff_n_gt_8 (n : ℕ) (h : n > 6) :
  BelaWins n ↔ n > 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_wins_iff_n_gt_8_l938_93827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_special_line_satisfies_conditions_l938_93879

/-- A line passing through (1,2) with vertical intercept twice the horizontal intercept -/
structure SpecialLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (1,2)
  point_condition : 2 = m * 1 + b
  -- The vertical intercept is twice the horizontal intercept
  intercept_condition : b = -2 * (-b / m)

/-- The equation of the special line is either 2x - y = 0 or 2x + y - 4 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -2 ∧ l.b = 4) := by
  sorry

/-- The equation of the special line in standard form -/
noncomputable def special_line_standard_form (l : SpecialLine) : ℝ × ℝ × ℝ :=
  if l.m = 2 ∧ l.b = 0 then (2, -1, 0) else (2, 1, -4)

/-- The special line equation satisfies the original conditions -/
theorem special_line_satisfies_conditions (l : SpecialLine) :
  let (a, b, c) := special_line_standard_form l
  a * 1 + b * 2 + c = 0 ∧ 
  (2 * c / a = -c / b ∨ 2 * c / a = c / b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_special_line_satisfies_conditions_l938_93879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_reconstruction_l938_93859

-- Define the quadrilateral and extended points
variable (A B C D A' B' C' D' : ℝ × ℝ)

-- Define the scaling factor
def r : ℝ := 2

-- Define the conditions
def condition1 (A B A' : ℝ × ℝ) (r : ℝ) : Prop := 
  A'.1 - B.1 = r * (A.1 - B.1) ∧ A'.2 - B.2 = r * (A.2 - B.2)

def condition2 (B C B' : ℝ × ℝ) (r : ℝ) : Prop := 
  B'.1 - C.1 = r * (B.1 - C.1) ∧ B'.2 - C.2 = r * (B.2 - C.2)

def condition3 (C D C' : ℝ × ℝ) (r : ℝ) : Prop := 
  C'.1 - D.1 = r * (C.1 - D.1) ∧ C'.2 - D.2 = r * (C.2 - D.2)

def condition4 (D A D' : ℝ × ℝ) (r : ℝ) : Prop := 
  D'.1 - A.1 = r * (D.1 - A.1) ∧ D'.2 - A.2 = r * (D.2 - A.2)

-- Define the theorem
theorem quadrilateral_reconstruction 
  (h1 : condition1 A B A' r)
  (h2 : condition2 B C B' r)
  (h3 : condition3 C D C' r)
  (h4 : condition4 D A D' r) :
  ∃ (p q s : ℝ), 
    A.1 = (1/26) * A'.1 + (6/26) * B'.1 + (8/26) * C'.1 + (16/26) * D'.1 ∧
    A.2 = (1/26) * A'.2 + (6/26) * B'.2 + (8/26) * C'.2 + (16/26) * D'.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_reconstruction_l938_93859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_length_l938_93851

theorem center_square_side_length 
  (large_square_side : ℝ) 
  (l_shape_area_fraction : ℝ) :
  large_square_side = 120 →
  l_shape_area_fraction = 2 / 9 →
  let total_area := large_square_side ^ 2
  let l_shape_total_area := 4 * l_shape_area_fraction * total_area
  let center_square_area := total_area - l_shape_total_area
  Real.sqrt center_square_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_length_l938_93851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l938_93891

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧ 0 < t.B ∧ t.B < Real.pi/2 ∧ 0 < t.C ∧ t.C < Real.pi/2

def satisfiesLawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h_acute : isAcute t)
  (h_law_of_sines : satisfiesLawOfSines t)
  (h_condition : 2 * t.b * Real.sin t.A = Real.sqrt 3 * t.a) :
  t.B = Real.pi/3 ∧ 
  (t.b = 6 ∧ t.a + t.c = 8 → t.a * t.c * Real.sin t.B / 2 = 7 * Real.sqrt 3 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l938_93891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93818

noncomputable def f (x : ℝ) : ℝ := 4 * Real.tan x * Real.sin (Real.pi / 2 - x) * Real.cos (x - Real.pi / 3) - Real.sqrt 3

def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi

def is_period (T : ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

def monotone_increasing_on (a b : ℝ) : Prop := ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y

theorem f_properties :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x - Real.pi / 3)) ∧
  domain = {x : ℝ | ∀ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi} ∧
  is_period Real.pi ∧
  (∀ T : ℝ, is_period T → Real.pi ≤ T) ∧
  monotone_increasing_on (-Real.pi / 12) (Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l938_93818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l938_93837

/-- Calculates the annual repayment for a loan with simple interest -/
noncomputable def annual_repayment_simple (loan_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  (loan_amount * (1 + years * interest_rate)) / (years + (years * (years - 1) / 2) * interest_rate)

/-- Calculates the annual repayment for a loan with compound interest -/
noncomputable def annual_repayment_compound (loan_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  (loan_amount * (1 + interest_rate) ^ years) * (interest_rate / ((1 + interest_rate) ^ years - 1))

theorem loan_repayment_theorem (loan_amount : ℝ) (simple_rate compound_rate : ℝ) (years : ℕ) 
    (h1 : loan_amount = 100000)
    (h2 : simple_rate = 0.05)
    (h3 : compound_rate = 0.04)
    (h4 : years = 10) :
  (abs (annual_repayment_simple loan_amount simple_rate years - 12245) < 1) ∧
  (abs (annual_repayment_compound loan_amount compound_rate years - 12330) < 1) := by
  sorry

-- Remove #eval statements as they are not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l938_93837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_sales_theorem_l938_93890

/-- Calculates the total money made from selling chocolate bars --/
def total_money_made (milk_price dark_price white_price : ℚ) 
  (milk_count dark_count white_count : ℚ) 
  (milk_sold_percent dark_sold_percent white_sold_percent : ℚ) : ℚ :=
  let milk_sold := ⌊milk_sold_percent * milk_count⌋
  let dark_sold := ⌊dark_sold_percent * dark_count⌋
  let white_sold := ⌊white_sold_percent * white_count⌋
  milk_sold * milk_price + dark_sold * dark_price + white_sold * white_price

/-- Theorem stating the total money made from selling chocolate bars --/
theorem chocolate_sales_theorem : 
  total_money_made 6 8 10 15 12 13 (70/100) (80/100) (65/100) = 212 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_sales_theorem_l938_93890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_has_46_chickens_l938_93841

/-- Represents the farmer's chicken and egg business -/
structure ChickenBusiness where
  chickens : ℕ
  eggs_per_week : ℕ
  price_per_dozen : ℚ
  weeks : ℕ
  total_revenue : ℚ

/-- Calculates the number of chickens given the business parameters -/
def calculate_chickens (b : ChickenBusiness) : ℚ :=
  (b.total_revenue * 12) / (b.eggs_per_week * b.weeks * b.price_per_dozen)

/-- Theorem stating that the farmer has 46 chickens -/
theorem farmer_has_46_chickens :
  let b : ChickenBusiness := {
    chickens := 46,
    eggs_per_week := 6,
    price_per_dozen := 3,
    weeks := 8,
    total_revenue := 552
  }
  calculate_chickens b = 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_has_46_chickens_l938_93841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_no_advice_formula_l938_93862

/-- The expected number of explorers who do not receive advice in a group of n explorers with friendship probability p -/
noncomputable def expected_no_advice (n : ℕ) (p : ℝ) : ℝ :=
  (1 - (1 - p)^n) / p

/-- Theorem: The expected number of explorers who do not receive advice
    in a group of n explorers with friendship probability p is (1 - (1-p)^n) / p -/
theorem expected_no_advice_formula (n : ℕ) (p : ℝ) 
  (hp : 0 < p ∧ p < 1) : 
  expected_no_advice n p = (1 - (1 - p)^n) / p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_no_advice_formula_l938_93862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l938_93845

def Q : ℕ := (List.range 67).filter Nat.Prime |>.map Nat.factorial |>.prod

theorem no_primes_in_sequence : 
  ∀ n ∈ Finset.range 60, ¬Nat.Prime (Q + (n + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l938_93845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_specific_field_l938_93898

/-- Calculates the fencing required for a rectangular field -/
noncomputable def fencing_required (area : ℝ) (uncovered_side : ℝ) : ℝ :=
  let width := area / uncovered_side
  uncovered_side + 2 * width

/-- Theorem: The fencing required for a rectangular field with area 720 sq. feet 
    and one uncovered side of 30 feet is 78 feet -/
theorem fencing_for_specific_field : fencing_required 720 30 = 78 := by
  -- Unfold the definition of fencing_required
  unfold fencing_required
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_specific_field_l938_93898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l938_93882

noncomputable def f (m : ℝ) (x : ℝ) := Real.sin (x + m) + Real.cos (x + m)

theorem min_shift_for_even_function :
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x : ℝ), f m x = f m (-x)) ∧
  (∀ (m' : ℝ), m' > 0 → (∀ (x : ℝ), f m' x = f m' (-x)) → m' ≥ m) ∧
  m = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l938_93882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_implies_nine_nine_implies_no_constant_term_l938_93814

/-- 
Given a natural number n, we define a polynomial P(x) as follows:
P(x) = (1 + x + x^2) * (x + 1/x^3)^n
-/
noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 + x + x^2) * (x + 1/x^3)^n

/-- 
A function that checks if P(x) has a constant term for a given n
-/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ ∀ (x : ℝ), x ≠ 0 → P n x - c * P n (1/x) = 0

/-- 
The main theorem: If P(x) has no constant term, then n must be 9
-/
theorem constant_term_implies_nine (n : ℕ) : 
  ¬(has_constant_term n) → n = 9 := by
  sorry

/-- 
Conversely, if n is 9, then P(x) has no constant term
-/
theorem nine_implies_no_constant_term : 
  ¬(has_constant_term 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_implies_nine_nine_implies_no_constant_term_l938_93814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_approximation_l938_93850

noncomputable def markup_percentage (tax_rate : ℝ) (initial_discount : ℝ) (further_discount : ℝ) 
  (profit : ℝ) (cost_price : ℝ) (additional_expense : ℝ) : ℝ :=
  let cp_after_tax := cost_price * (1 + tax_rate)
  let total_cost_price := cp_after_tax + additional_expense
  let selling_price := total_cost_price * (1 + profit)
  let initial_selling_price := (selling_price + further_discount) / (1 - initial_discount)
  let markup := initial_selling_price - total_cost_price
  (markup / total_cost_price) * 100

theorem markup_percentage_approximation :
  let tax_rate := (12 : ℝ) / 100
  let initial_discount := (15 : ℝ) / 100
  let further_discount := (65 : ℝ)
  let profit := (30 : ℝ) / 100
  let cost_price := (250 : ℝ)
  let additional_expense := (25 : ℝ)
  abs (markup_percentage tax_rate initial_discount further_discount profit cost_price additional_expense - 78.21) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_approximation_l938_93850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_l938_93868

/-- The center of the circle described by the equation x^2 + y^2 - x + 2y = 0 -/
noncomputable def circle_center : ℝ × ℝ := (1/2, -1)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 2*y = 0

theorem center_of_circle :
  let (h, k) := circle_center
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 1/4) := by
  sorry

#check center_of_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_l938_93868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l938_93815

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (x + Real.pi/4) - Real.tan (x + Real.pi/3) + Real.sin (x + Real.pi/3)

theorem max_value_of_f :
  ∃ (max : ℝ), max = -Real.sqrt 3 + 1 - Real.sqrt 2 / 2 ∧
  ∀ (x : ℝ), -Real.pi/2 ≤ x ∧ x ≤ -Real.pi/4 → f x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l938_93815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_correct_l938_93856

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original : ℝ := 12.8572

/-- The expected result after rounding to the nearest hundredth -/
def expected : ℝ := 12.86

theorem round_to_hundredth_correct :
  roundToHundredth original = expected := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_correct_l938_93856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_outside_circle_l938_93893

/-- The circle with equation x^2 + y^2 = 24 -/
def my_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 24}

/-- The point P with coordinates (2, 5) -/
def point_P : ℝ × ℝ := (2, 5)

/-- A point is outside the circle if its distance from the origin is greater than the radius -/
def is_outside (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 > 24

theorem point_P_outside_circle : is_outside point_P := by
  unfold is_outside point_P
  norm_num
  -- The proof is completed by norm_num, but we can add sorry if needed
  -- sorry

#check point_P_outside_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_outside_circle_l938_93893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_24_minutes_l938_93846

/-- Represents the state of filling a tub with water -/
structure TubFilling where
  capacity : ℚ
  flowRate : ℚ
  escapeRate : ℚ

/-- Calculates the time in minutes to fill the tub -/
noncomputable def timeToFill (t : TubFilling) : ℚ :=
  t.capacity / ((t.flowRate - t.escapeRate) / 2 - t.escapeRate / 2)

/-- Theorem stating the time to fill the tub under given conditions -/
theorem fill_time_is_24_minutes :
  let t : TubFilling := {
    capacity := 120,
    flowRate := 12,
    escapeRate := 1
  }
  timeToFill t = 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_24_minutes_l938_93846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_translation_l938_93887

theorem cosine_translation (φ : Real) : 
  (0 < φ) ∧ (φ < π) ∧ 
  (∀ x : Real, Real.cos (2*x - π/6) = Real.cos (2*(x + φ))) → 
  φ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_translation_l938_93887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_months_l938_93894

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem mortgage_payment_months (a : ℝ) (r : ℝ) (s : ℝ) (n : ℕ) 
  (h1 : a = 100)
  (h2 : r = 3)
  (h3 : s = 12100)
  (h4 : s = geometric_sum a r n)
  : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_months_l938_93894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_C₂_range_l938_93875

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := ((3/2) * Real.cos α, Real.sin α)

-- Define the curve C₂
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ α : ℝ, p = (3 * Real.cos α, 2 * Real.sin α)}

-- Theorem for the general equation of C₂
theorem C₂_equation : ∀ p : ℝ × ℝ, p ∈ C₂ ↔ p.1^2 / 9 + p.2^2 / 4 = 1 := by
  sorry

-- Theorem for the range of x + 2y on C₂
theorem C₂_range : ∀ p : ℝ × ℝ, p ∈ C₂ → -5 ≤ p.1 + 2*p.2 ∧ p.1 + 2*p.2 ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_C₂_range_l938_93875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l938_93869

/-- Represents a person in the arrangement -/
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person
| E : Person
deriving Repr, DecidableEq

/-- An arrangement is a list of people -/
def Arrangement := List Person

/-- Check if an arrangement is valid (A and B are not adjacent to C) -/
def isValidArrangement : Arrangement → Bool
  | [] => true
  | [_] => true
  | x :: y :: rest => 
    if x = Person.C then
      (y ≠ Person.A && y ≠ Person.B) && isValidArrangement (y :: rest)
    else if y = Person.C then
      (x ≠ Person.A && x ≠ Person.B) && isValidArrangement (y :: rest)
    else
      isValidArrangement (y :: rest)

/-- The main theorem: there are exactly 36 valid arrangements -/
theorem valid_arrangements_count :
  (List.filter isValidArrangement (List.permutations [Person.A, Person.B, Person.C, Person.D, Person.E])).length = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l938_93869
