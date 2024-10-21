import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_differences_in_T_l955_95537

def T : Finset ℕ := Finset.image (fun i => 3^i) (Finset.range 8)

def M : ℕ := Finset.sum (Finset.product T T) (fun (a, b) => max a b - min a b)

theorem sum_of_differences_in_T : M = 20317 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_differences_in_T_l955_95537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_15_value_l955_95597

/-- Given a geometric sequence {a_n}, S_n denotes the sum of the first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- The sequence {a_n} is geometric -/
axiom is_geometric_sequence : ∀ n : ℕ, ∃ r : ℝ, S (n + 1) - S n = r * (S n - S (n - 1))

/-- S_5 = 4 -/
axiom S_5 : S 5 = 4

/-- S_10 = 12 -/
axiom S_10 : S 10 = 12

/-- Theorem: S_15 = 28 -/
theorem S_15_value : S 15 = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_15_value_l955_95597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_quadratic_equation_prs_over_q_value_l955_95578

theorem largest_solution_quadratic_equation :
  ∀ (y p q r s : ℝ),
  (7 * y) / 8 + 2 = 4 / y →
  y = (p + q * Real.sqrt r) / s →
  p = -8 ∧ q = 12 ∧ r = 2 ∧ s = 7 →
  ∀ y' : ℝ, (7 * y') / 8 + 2 = 4 / y' → y' ≤ y :=
by
  sorry

#check largest_solution_quadratic_equation

theorem prs_over_q_value :
  ∀ (p q r s : ℝ),
  p = -8 ∧ q = 12 ∧ r = 2 ∧ s = 7 →
  (p * r * s) / q = -28 / 3 :=
by
  sorry

#check prs_over_q_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_quadratic_equation_prs_over_q_value_l955_95578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l955_95599

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (2 * ω * x + φ) - 1

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < Real.pi / 2) 
  (h_period : ∀ x, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x)
  (h_point : f ω φ 0 = -1/2) :
  ω = 2 ∧ φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l955_95599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_b_c_start_l955_95534

/-- Represents a racer in the kilometer race -/
structure Racer where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  a : Racer
  b : Racer
  c : Racer
  race_length : ℝ
  a_b_start : ℝ
  a_c_start : ℝ

/-- Theorem stating the relationship between racers' speeds based on given conditions -/
theorem race_theorem (rc : RaceConditions) (h1 : rc.race_length = 1000)
    (h2 : rc.a_b_start = 100) (h3 : rc.a_c_start = 150) :
    rc.b.speed / rc.c.speed = 900 / 850 := by sorry

/-- Main theorem proving B can give C a 100 meters start -/
theorem b_c_start (rc : RaceConditions) (h1 : rc.race_length = 1000)
    (h2 : rc.a_b_start = 100) (h3 : rc.a_c_start = 150) :
    ∃ start : ℝ, start = 100 ∧ 
    rc.b.speed * (rc.race_length - start) = rc.c.speed * rc.race_length := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_b_c_start_l955_95534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_is_8_4_l955_95571

/-- Represents the shooting results of an athlete -/
structure ShootingResult where
  total_shots : Nat
  ten_points : Nat
  nine_points : Nat
  eight_points : Nat
  seven_points : Nat

/-- Calculates the average score for a given ShootingResult -/
def averageScore (result : ShootingResult) : ℚ :=
  let total_points := 
    10 * result.ten_points + 
    9 * result.nine_points + 
    8 * result.eight_points + 
    7 * result.seven_points
  ↑total_points / ↑result.total_shots

/-- The specific shooting result from the problem -/
def athleteResult : ShootingResult := {
  total_shots := 10
  ten_points := 1
  nine_points := 4
  eight_points := 3
  seven_points := 2
}

/-- Theorem stating that the average score for the given shooting result is 8.4 -/
theorem average_score_is_8_4 : 
  averageScore athleteResult = 84 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_is_8_4_l955_95571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l955_95569

theorem sin_pi_minus_alpha (α : ℝ) 
  (h1 : Real.cos (2 * Real.pi - α) = Real.sqrt 5 / 3)
  (h2 : α > -Real.pi / 2 ∧ α < 0) : 
  Real.sin (Real.pi - α) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l955_95569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_metal_mass_l955_95596

/-- Represents the mass of each metal in the alloy -/
structure MetalMasses where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ
  m4 : ℝ

/-- Defines the properties of the alloy based on the given conditions -/
def AlloyProperties (masses : MetalMasses) : Prop :=
  masses.m1 + masses.m2 + masses.m3 + masses.m4 = 35 ∧
  masses.m1 = 1.5 * masses.m2 ∧
  masses.m2 / masses.m3 = 3 / 4 ∧
  masses.m3 / masses.m4 = 5 / 6

/-- Theorem stating that given the alloy properties, the mass of the third metal is approximately 6.73 kg -/
theorem third_metal_mass (masses : MetalMasses) 
  (h : AlloyProperties masses) : 
  ∃ ε > 0, |masses.m3 - 6.73| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_metal_mass_l955_95596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_101_l955_95511

def f (x : ℤ) : ℤ := x^2 - x + 2010

theorem gcd_f_100_101 : Int.gcd (f 100) (f 101) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_101_l955_95511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_10_eq_2_l955_95562

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x / Real.log 2

-- State the theorem
theorem f_f_10_eq_2 : f (f 10) = 2 := by
  -- Evaluate f(10)
  have h1 : f 10 = Real.log 10 / Real.log 2 := by sorry
  -- Show that Real.log 10 / Real.log 2 = 1
  have h2 : Real.log 10 / Real.log 2 = 1 := by sorry
  -- Evaluate f(f(10))
  have h3 : f (f 10) = f 1 := by sorry
  -- Evaluate f(1)
  have h4 : f 1 = 1^2 + 1 := by sorry
  -- Simplify
  have h5 : 1^2 + 1 = 2 := by sorry
  -- Combine steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_10_eq_2_l955_95562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l955_95535

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the x-coordinate of the vertex of a parabola -/
noncomputable def vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- Calculates the y-coordinate of the vertex of a parabola -/
noncomputable def vertex_y (p : Parabola) : ℝ := 
  p.a * (vertex_x p)^2 + p.b * (vertex_x p) + p.c

/-- Represents the problem setup -/
structure ProblemSetup where
  c : ℝ
  C₁ : Parabola
  C₂ : Parabola
  h₁ : C₁.a = 1 ∧ C₁.b = 3 ∧ C₁.c = c
  h₂ : C₂.a = C₁.a ∧ C₂.b = C₁.b ∧ C₂.c = 2 - C₁.c
  h₃ : abs (vertex_y C₁ - vertex_y C₂) = 5

theorem parabola_problem (setup : ProblemSetup) : setup.c = 3/4 ∨ setup.c = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l955_95535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_income_is_40000_tax_calculation_consistent_l955_95593

/-- Represents the tax rate for the first $30000 of income -/
noncomputable def q : ℝ := Real.pi  -- Using pi as a placeholder for an arbitrary real number

/-- Represents Emily's annual income -/
def emily_income : ℝ := 40000

/-- Calculates the tax amount for a given income -/
noncomputable def tax_amount (income : ℝ) : ℝ :=
  if income ≤ 30000 then
    (q / 100) * income
  else
    (q / 100) * 30000 + ((q + 3) / 100) * (income - 30000)

/-- States that Emily's tax paid is (q + 0.75)% of her total income -/
axiom emily_tax_condition : 
  tax_amount emily_income = ((q + 0.75) / 100) * emily_income

/-- Theorem: Emily's annual income is $40000 -/
theorem emily_income_is_40000 : emily_income = 40000 := by
  rfl

/-- Proof that the tax calculation is consistent with Emily's income -/
theorem tax_calculation_consistent : 
  tax_amount emily_income = ((q + 0.75) / 100) * emily_income := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_income_is_40000_tax_calculation_consistent_l955_95593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_150_l955_95552

/-- An isosceles triangle with a square inscribed in it. -/
structure TriangleWithSquare where
  base : ℝ
  height : ℝ
  squareSide : ℝ
  isIsosceles : base > 0 ∧ height > 0
  isInscribed : squareSide > 0 ∧ squareSide < base ∧ squareSide < height

/-- The area of the shaded region in a triangle with an inscribed square. -/
noncomputable def shadedArea (t : TriangleWithSquare) : ℝ :=
  (t.base * t.height / 2) - t.squareSide ^ 2

/-- Theorem stating that for a specific triangle with base 21 and height 28,
    the shaded area is 150 square units. -/
theorem shaded_area_is_150 :
  ∃ t : TriangleWithSquare,
    t.base = 21 ∧
    t.height = 28 ∧
    shadedArea t = 150 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_150_l955_95552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_squared_cos_cubed_l955_95581

theorem integral_sin_squared_cos_cubed (x : ℝ) :
  (deriv (fun x => (Real.sin x ^ 3) / 3 - (Real.sin x ^ 5) / 5)) x = Real.sin x ^ 2 * Real.cos x ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_squared_cos_cubed_l955_95581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_area_equation_l955_95531

/-- The length of the rectangular area in meters -/
def length_of_rectangle (x : ℝ) : ℝ := x

/-- The width of the rectangular area in meters -/
def width_of_rectangle (x : ℝ) : ℝ := x - 6

/-- The area of the rectangular space in square meters -/
def area_of_rectangle (x : ℝ) : ℝ := x * (x - 6)

/-- Proves that for a rectangular area with length x meters and width (x-6) meters, 
    if the area is 720 square meters, then the equation x(x-6) = 720 holds true. -/
theorem rectangular_area_equation (x : ℝ) :
  (x > 6) →  -- Ensure width is positive
  (x * (x - 6) = 720) ↔ 
  (x = length_of_rectangle x ∧ 
   x - 6 = width_of_rectangle x ∧ 
   720 = area_of_rectangle x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_area_equation_l955_95531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_range_theorem_l955_95594

noncomputable def P (x y : ℤ) : ℤ := x - (x^2 + 1) * (x^2 - 3*y^2 - 1)^2

noncomputable def u (n : ℕ) : ℤ := ⌈(1/2 : ℝ) * (2 + Real.sqrt 3)^n⌉

theorem polynomial_range_theorem :
  ∀ z : ℤ, (∃ x y : ℤ, P x y = z) ↔ 
    (z < 0 ∨ ∃ n : ℕ, z = u n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_range_theorem_l955_95594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_two_l955_95565

theorem tan_alpha_minus_two (α : ℝ) 
  (h1 : Real.tan α = -2) 
  (h2 : Real.cos α ≠ 0) : 
  (((Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)) = 5) ∧ 
  ((1 / (Real.sin α * Real.cos α)) = -5/2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_two_l955_95565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highrise_elevator_speed_l955_95500

/-- Represents an elevator in a building -/
structure Elevator where
  speed : ℝ
  buildingHeight : ℝ
  numStops : ℕ
  stopDuration : ℝ

/-- The time taken for an elevator to reach the top floor -/
noncomputable def travelTime (e : Elevator) : ℝ :=
  e.buildingHeight / e.speed + (e.numStops : ℝ) * e.stopDuration

/-- Theorem stating the high-rise elevator speed is 3 m/s -/
theorem highrise_elevator_speed :
  ∀ (regular highrise : Elevator),
    regular.buildingHeight = 33 →
    highrise.buildingHeight = 81 →
    regular.numStops = 2 →
    highrise.numStops = 1 →
    regular.stopDuration = 6 →
    highrise.stopDuration = 7 →
    travelTime regular = travelTime highrise →
    highrise.speed = regular.speed + 1.5 →
    highrise.speed < 5 →
    highrise.speed = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highrise_elevator_speed_l955_95500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_ring_area_l955_95522

/-- Represents a convex polygon circumscribed about a circle -/
structure CircumscribedPolygon where
  /-- Radius of the inscribed circle -/
  R : ℝ
  /-- Perimeter of the polygon -/
  P : ℝ
  /-- Assumption that the perimeter is positive -/
  P_pos : P > 0

/-- Represents the coin moving along the polygon -/
structure MovingCoin where
  /-- Radius of the coin -/
  r : ℝ
  /-- Assumption that the coin's radius is positive -/
  r_pos : r > 0

/-- The polygonal ring formed by the path of the coin -/
def PolygonalRing (polygon : CircumscribedPolygon) (coin : MovingCoin) : Prop :=
  polygon.R > coin.r

/-- Additional area correction due to geometric considerations -/
noncomputable def areaCorrection (polygon : CircumscribedPolygon) (coin : MovingCoin) : ℝ :=
  sorry

/-- The area of the polygonal ring -/
noncomputable def areaPolygonalRing (polygon : CircumscribedPolygon) (coin : MovingCoin) 
  (ring : PolygonalRing polygon coin) : ℝ := 
  polygon.P * coin.r + areaCorrection polygon coin

/-- Theorem stating the area of the polygonal ring -/
theorem polygonal_ring_area (polygon : CircumscribedPolygon) (coin : MovingCoin) 
  (ring : PolygonalRing polygon coin) :
  ∃ (A : ℝ), A = areaPolygonalRing polygon coin ring := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_ring_area_l955_95522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l955_95560

/-- Represents the tax system described in the problem -/
structure TaxSystem where
  /-- Tax rate in percentage, equal to income in hundreds of dollars -/
  rate : ℝ
  /-- Fixed deduction amount in dollars -/
  deduction : ℝ := 2000

/-- Calculates the take-home pay for a given income -/
noncomputable def takeHomePay (ts : TaxSystem) (income : ℝ) : ℝ :=
  let taxableIncome := income - ts.deduction
  let tax := (ts.rate / 100) * taxableIncome
  income - tax

/-- Theorem stating that the income maximizing take-home pay is approximately $26040 -/
theorem max_take_home_pay (ts : TaxSystem) :
  ∃ (max_income : ℝ), 
    (∀ (income : ℝ), takeHomePay ts income ≤ takeHomePay ts max_income) ∧
    (abs (max_income - 26040) < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l955_95560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_has_only_minimum_l955_95514

open Set
open Function

/-- Given a function h(x) = f(x) / g(x) on the interval (0, 3] where g(x) ≠ 0,
    and for all x in (0, 3], f(x)g'(x) > f'(x)g(x),
    prove that h(x) has only a minimum value on this interval. -/
theorem h_has_only_minimum
  (f g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (h_def : ∀ x ∈ Ioo 0 3, h x = f x / g x)
  (g_nonzero : ∀ x ∈ Ioo 0 3, g x ≠ 0)
  (fg_inequality : ∀ x ∈ Ioo 0 3, f x * (deriv g x) > (deriv f x) * g x)
  : ∃ (min : ℝ), IsMinOn h (Ioo 0 3) min ∧ ¬∃ (max : ℝ), IsMaxOn h (Ioo 0 3) max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_has_only_minimum_l955_95514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_overflow_l955_95592

/-- Represents a cylindrical container -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a spherical ball -/
structure Ball where
  radius : ℝ

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

/-- The volume of a ball -/
noncomputable def ballVolume (b : Ball) : ℝ :=
  (4/3) * Real.pi * b.radius^3

theorem water_overflow (c : Cylinder) (b : Ball) :
  c.radius = 5 ∧ c.height = 2 ∧ b.radius = 3 →
  cylinderVolume c = 50 * Real.pi ∧ ballVolume b = 36 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_overflow_l955_95592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_vendor_throwaway_l955_95586

/-- Represents the percentage of apples thrown away on the first day -/
def first_day_throwaway_percent : ℝ → Prop := sorry

/-- The total percentage of apples thrown away over two days -/
def total_throwaway_percent : ℝ → ℝ → Prop := sorry

theorem apple_vendor_throwaway 
  (x : ℝ) 
  (h1 : first_day_throwaway_percent x)
  (h2 : total_throwaway_percent x 28) : 
  x = 40 := by
  sorry

#check apple_vendor_throwaway

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_vendor_throwaway_l955_95586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kelly_weight_is_34_l955_95568

/-- The weight capacity of the bridge in kilograms -/
def bridge_capacity : ℝ := 100

/-- Megan's weight in kilograms -/
noncomputable def megan_weight : ℝ := 40

/-- Kelly's weight as a fraction of Megan's weight -/
def kelly_weight_fraction : ℝ := 0.85

/-- Mike's weight difference from Megan's weight in kilograms -/
def mike_weight_difference : ℝ := 5

/-- The amount by which the children exceed the bridge capacity in kilograms -/
def excess_weight : ℝ := 19

/-- Kelly's weight in kilograms -/
noncomputable def kelly_weight : ℝ := kelly_weight_fraction * megan_weight

theorem kelly_weight_is_34 :
  megan_weight + kelly_weight + (megan_weight + mike_weight_difference) = bridge_capacity + excess_weight ∧
  kelly_weight = 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kelly_weight_is_34_l955_95568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisected_by_line_l955_95547

-- Define the circle
def my_circle (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y + 1)^2 = 3

-- Define the line
def my_line (a : ℝ) (x y : ℝ) : Prop :=
  5*x + 4*y - a = 0

-- Define the center of the circle
def my_circle_center (a : ℝ) : ℝ × ℝ :=
  (a, -1)

-- State the theorem
theorem circle_bisected_by_line (a : ℝ) :
  (∀ x y : ℝ, my_circle a x y → my_line a x y → 
    (my_circle_center a).1 = x ∧ (my_circle_center a).2 = y) →
  a = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisected_by_line_l955_95547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_has_minimum_l955_95580

/-- A quadratic function g(x) with specific properties -/
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + b^2 / a

/-- Theorem: The graph of y = g(x) has a minimum -/
theorem quadratic_has_minimum (a b : ℝ) (h : a > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), g a b x_min ≤ g a b x := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_has_minimum_l955_95580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_inside_inscribed_square_l955_95584

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a square -/
structure Square where
  D : Point
  E : Point
  F : Point
  G : Point

/-- Predicate to check if a triangle is acute -/
def IsAcute (t : Triangle) : Prop := sorry

/-- Predicate to check if a square is inscribed in a triangle -/
def IsInscribed (t : Triangle) (s : Square) : Prop := sorry

/-- Predicate to check if the vertices of the square are on the sides of the triangle -/
def VerticesOnSides (t : Triangle) (s : Square) : Prop := sorry

/-- Represents the configuration of a square inscribed in a triangle -/
structure InscribedSquareTriangle where
  triangle : Triangle
  square : Square
  is_acute : IsAcute triangle
  is_inscribed : IsInscribed triangle square
  vertices_on_sides : VerticesOnSides triangle square

/-- The center of the inscribed circle of a triangle -/
noncomputable def incenter (t : Triangle) : Point := sorry

/-- Predicate to check if a point is inside a square -/
def is_inside_square (p : Point) (s : Square) : Prop := sorry

/-- Main theorem: The incenter of an acute triangle is inside its inscribed square -/
theorem incenter_inside_inscribed_square (config : InscribedSquareTriangle) :
  is_inside_square (incenter config.triangle) config.square := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_inside_inscribed_square_l955_95584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_two_pi_thirds_l955_95544

theorem cos_alpha_plus_two_pi_thirds (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = -4*Real.sqrt 3/5) 
  (h2 : -π/2 < α) (h3 : α < 0) : 
  Real.cos (α + 2*π/3) = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_two_pi_thirds_l955_95544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l955_95502

/-- If the terminal side of angle α passes through point P(4,-3), then 2sin α + 3cos α = 6/5 -/
theorem angle_terminal_side_point (α : Real) : 
  (let P : ℝ × ℝ := (4, -3)
   let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
   let sin_α : ℝ := P.2 / r
   let cos_α : ℝ := P.1 / r
   2 * sin_α + 3 * cos_α) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l955_95502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rad_to_deg_conversions_deg_to_rad_conversions_l955_95519

-- Define the conversion factor
noncomputable def π : Real := Real.pi

-- Define the conversion function from radians to degrees
noncomputable def radToDeg (x : Real) : Real := x * 180 / π

-- Define the conversion function from degrees to radians
noncomputable def degToRad (x : Real) : Real := x * π / 180

-- Theorem statements
theorem rad_to_deg_conversions :
  (radToDeg (π/12) = 15) ∧
  (radToDeg (13*π/6) = 390) ∧
  (radToDeg (-5*π/12) = -75) := by
  sorry

theorem deg_to_rad_conversions :
  (degToRad 36 = π/5) ∧
  (degToRad (-105) = -7*π/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rad_to_deg_conversions_deg_to_rad_conversions_l955_95519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_dough_calculation_l955_95585

/-- Pizza dough recipe calculation -/
theorem pizza_dough_calculation 
  (flour_base : ℝ) 
  (milk_base : ℝ) 
  (water_base : ℝ) 
  (flour_total : ℝ) 
  (h1 : flour_base = 250) 
  (h2 : milk_base = 50) 
  (h3 : water_base = 30) 
  (h4 : flour_total = 1250) : 
  (flour_total / flour_base * milk_base = 250) ∧ 
  (flour_total / flour_base * water_base = 150) := by
  sorry

#check pizza_dough_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_dough_calculation_l955_95585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_intercepts_l955_95546

-- Define the line equation
noncomputable def line_equation (x y a : ℝ) : Prop := x + a^2 * y - a = 0

-- Define the sum of intercepts function
noncomputable def sum_of_intercepts (a : ℝ) : ℝ := a + 1 / a^2

-- Theorem statement
theorem min_sum_of_intercepts :
  ∀ a : ℝ, a > 0 →
  sum_of_intercepts a ≥ 2 ∧
  (sum_of_intercepts a = 2 ↔ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_intercepts_l955_95546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ellipse_intersection_l955_95509

/-- The ellipse C with semi-major axis a and semi-minor axis b -/
def C (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- The triangle formed by points O, S, and Q -/
def triangle_OSQ (O S Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {T : ℝ × ℝ | ∃ (t1 t2 t3 : ℝ), t1 ≥ 0 ∧ t2 ≥ 0 ∧ t3 ≥ 0 ∧ t1 + t2 + t3 = 1 ∧
    T = (t1 * O.1 + t2 * S.1 + t3 * Q.1, t1 * O.2 + t2 * S.2 + t3 * Q.2)}

/-- The area of a triangle formed by three points -/
noncomputable def area (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given an ellipse and a line intersecting it, prove the maximum area of a specific triangle -/
theorem max_area_triangle_ellipse_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (P Q S : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (P ∈ C a b ∧ Q ∈ C a b) ∧
    (P ∈ l ∧ Q ∈ l) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (m : ℝ), x = m + k * y) ∧
    (S = (P.1, -P.2)) ∧
    (∀ (T : ℝ × ℝ), T ∈ triangle_OSQ O S Q → area (triangle_OSQ O S Q) ≤ (1/2) * a * b) ∧
    (∃ (T : ℝ × ℝ), T ∈ triangle_OSQ O S Q ∧ area (triangle_OSQ O S Q) = (1/2) * a * b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ellipse_intersection_l955_95509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_plus_3_l955_95559

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

-- State the theorem
theorem f_pi_plus_3 (a b : ℝ) :
  (f a b (-3) = 5) →
  (f a b 3 + f a b (-3) = 2) →
  (f a b (Real.pi + 3) = -3) :=
by
  intro h1 h2
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_plus_3_l955_95559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_min_value_of_g_l955_95504

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x * (x + 1) - Real.log x

def g (x : ℝ) : ℝ := f a x + Real.log x - a * x^2 + Real.exp x

theorem tangent_line_at_one (h : a = 1) :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ y = f 1 x + (deriv (f 1)) 1 * (x - 1) :=
sorry

theorem min_value_of_g (h : a < -1) :
  ∃ (x_min : ℝ), ∀ x, x > 0 → g a x ≥ g a x_min ∧ g a x_min = a * (Real.log (-a) - 1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_min_value_of_g_l955_95504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_triangle_with_centroid_l955_95507

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
variable (colorFunction : Point → Color)

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3
  , y := (t.A.y + t.B.y + t.C.y) / 3 }

-- The main theorem
theorem exists_monochromatic_triangle_with_centroid :
  ∃ (t : Triangle), 
    (colorFunction t.A = colorFunction t.B) ∧ 
    (colorFunction t.B = colorFunction t.C) ∧ 
    (colorFunction t.C = colorFunction (centroid t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_triangle_with_centroid_l955_95507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_four_fifteenths_l955_95573

/-- Represents the fraction of area shaded in each subdivision level -/
def shaded_fraction : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (1/16) * shaded_fraction n

/-- The sum of shaded areas across all subdivision levels -/
noncomputable def total_shaded_area : ℚ := ∑' n, shaded_fraction n

/-- Theorem stating that the total shaded area is 4/15 -/
theorem shaded_area_is_four_fifteenths : total_shaded_area = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_four_fifteenths_l955_95573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_median_minimizes_sum_distances_l955_95564

/-- The point that minimizes the sum of distances to four given points in ℝ² -/
def geometric_median : ℝ × ℝ := (6, 12)

/-- The four given points in ℝ² -/
def points : List (ℝ × ℝ) := [(0, 0), (10, 20), (5, 15), (12, -6)]

/-- The Euclidean distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The sum of distances from a point to all given points -/
noncomputable def sum_distances (p : ℝ × ℝ) : ℝ :=
  (points.map (distance p)).sum

/-- Theorem: The geometric median minimizes the sum of distances to the given points -/
theorem geometric_median_minimizes_sum_distances :
  ∀ p : ℝ × ℝ, sum_distances geometric_median ≤ sum_distances p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_median_minimizes_sum_distances_l955_95564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constant_difference_l955_95572

theorem triangle_constant_difference (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π)
  (h_sum : α + β + γ = π)
  (h_cos_law_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos α)
  (h_cos_law_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos β)
  (h_cos_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos γ) :
  a^2 + b^2 - a*b*Real.cos γ = a^2 + c^2 - a*c*Real.cos β ∧
  a^2 + c^2 - a*c*Real.cos β = b^2 + c^2 - b*c*Real.cos α ∧
  b^2 + c^2 - b*c*Real.cos α = a^2 + b^2 + c^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constant_difference_l955_95572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l955_95527

/-- Given a function f: ℝ → ℝ with a tangent line y = 3x - 2 at the point (1, f(1)),
    prove that f(1) + f'(1) = 4. -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = 3 * x - 2) : 
    f 1 + deriv f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l955_95527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_ratio_bounds_l955_95505

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the incenter I
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the angle bisector intersection points
noncomputable def angleBisectorIntersection (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incenter_ratio_bounds (t : Triangle) :
  let I := incenter t
  let (A', B', C') := angleBisectorIntersection t
  let ratio := (distance t.A I * distance t.B I * distance t.C I) /
               (distance t.A A' * distance t.B B' * distance t.C C')
  1/4 < ratio ∧ ratio ≤ 8/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_ratio_bounds_l955_95505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_order_l955_95567

-- Define the types of phenomena
inductive Phenomenon
  | VeryLikely
  | Certain
  | Possible
  | Impossible
  | NotVeryLikely

-- Define a function to represent the probability of a phenomenon
noncomputable def probability (p : Phenomenon) : ℝ := sorry

-- State the theorem about the order of probabilities
theorem probability_order :
  probability Phenomenon.Certain > probability Phenomenon.VeryLikely ∧
  probability Phenomenon.VeryLikely > probability Phenomenon.Possible ∧
  probability Phenomenon.Possible > probability Phenomenon.NotVeryLikely ∧
  probability Phenomenon.NotVeryLikely > probability Phenomenon.Impossible :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_order_l955_95567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_sum_l955_95512

theorem prime_divisor_sum (p : Nat) : 
  Nat.Prime p → 
  (∃ n u v : Nat, n > 0 ∧ u > 0 ∧ v > 0 ∧
    (Finset.card (Nat.divisors n) = p^u) ∧
    (Finset.sum (Nat.divisors n) id = p^v)) →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_sum_l955_95512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_configuration_theorem_l955_95518

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
structure LineSegment where
  p1 : Point
  p2 : Point

/-- A configuration of points and line segments in a plane -/
structure PlaneConfiguration where
  points : Finset Point
  segments : Finset LineSegment

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Predicate to check if a line segment connects two points -/
def connects (s : LineSegment) (p1 p2 : Point) : Prop :=
  (s.p1 = p1 ∧ s.p2 = p2) ∨ (s.p1 = p2 ∧ s.p2 = p1)

theorem plane_configuration_theorem (config : PlaneConfiguration) :
  (config.points.card = 6) →
  (∀ p1 p2 p3, p1 ∈ config.points → p2 ∈ config.points → p3 ∈ config.points →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) →
  (config.segments.card = 13) →
  ∃ p1 p2 p3 p4, p1 ∈ config.points ∧ p2 ∈ config.points ∧ p3 ∈ config.points ∧ p4 ∈ config.points ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    (∃ s12 ∈ config.segments, connects s12 p1 p2) ∧
    (∃ s13 ∈ config.segments, connects s13 p1 p3) ∧
    (∃ s14 ∈ config.segments, connects s14 p1 p4) ∧
    (∃ s23 ∈ config.segments, connects s23 p2 p3) ∧
    (∃ s24 ∈ config.segments, connects s24 p2 p4) ∧
    (∃ s34 ∈ config.segments, connects s34 p3 p4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_configuration_theorem_l955_95518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_symmetric_points_l955_95563

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Check if a point is symmetric to another point with respect to the y-axis -/
def isSymmetricToYAxis (p q : Point3D) : Prop :=
  p.x = -q.x ∧ p.y = q.y ∧ p.z = q.z

theorem distance_symmetric_points :
  let A : Point3D := ⟨-1, 2, -3⟩
  let B : Point3D := ⟨-1, 0, 2⟩
  ∀ M : Point3D, isSymmetricToYAxis A M → distance B M = 3 := by
  sorry

#check distance_symmetric_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_symmetric_points_l955_95563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pin_matching_probability_l955_95525

/-- Represents a credit card with a PIN code and attempt limit -/
structure CreditCard where
  pin : Nat
  attemptsLeft : Nat

/-- Represents the state of the PIN matching process -/
structure PINMatchingState where
  cards : List CreditCard
  pins : List Nat

/-- The probability of successfully matching all PINs to cards -/
noncomputable def probabilityOfSuccess (initialState : PINMatchingState) : ℚ :=
  1 - (1 / 4 * 1 / 3 * 1 / 2)

/-- Theorem stating the probability of successfully matching all PINs -/
theorem pin_matching_probability 
  (initialState : PINMatchingState) 
  (h1 : initialState.cards.length = 4)
  (h2 : initialState.pins.length = 4)
  (h3 : ∀ c ∈ initialState.cards, c.attemptsLeft = 3)
  (h4 : initialState.pins.Pairwise (· ≠ ·))
  : probabilityOfSuccess initialState = 23 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pin_matching_probability_l955_95525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l955_95550

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  (x * y / Real.sqrt (x * y + y * z) + 
   y * z / Real.sqrt (y * z + z * x) + 
   z * x / Real.sqrt (z * x + x * y)) ≤ Real.sqrt 2 / 2 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l955_95550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_eq_four_l955_95508

/-- The infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.exp (x * Real.log x)

/-- Theorem stating that √2 is the solution to the infinite power tower equation -/
theorem infinite_power_tower_eq_four :
  ∃ (x : ℝ), x > 0 ∧ infinitePowerTower x = 4 ∧ x = Real.sqrt 2 := by
  -- The proof is omitted for now
  sorry

#check infinite_power_tower_eq_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_eq_four_l955_95508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_prime_congruence_l955_95530

theorem fermat_prime_congruence (m k : ℕ) (hp : Prime (2^(2^m) + 1)) :
  let p := 2^(2^m) + 1
  (2^(2^(m+1) * p^k) ≡ 1 [MOD p^(k+1)]) ∧
  (∀ n : ℕ, n > 0 → (2^n ≡ 1 [MOD p^(k+1)] → n ≥ 2^(m+1) * p^k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_prime_congruence_l955_95530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_counting_not_rational_l955_95576

/-- The prime counting function -/
def prime_counting (n : ℕ) : ℕ := (Finset.range n).filter Nat.Prime |>.card

/-- Statement: The prime counting function cannot be represented as a rational function -/
theorem prime_counting_not_rational : 
  ¬∃ (P Q : ℕ → ℤ), ∀ x : ℕ, (prime_counting x : ℚ) = (P x : ℚ) / (Q x : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_counting_not_rational_l955_95576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_selection_count_l955_95526

/-- The number of teachers qualified to teach only English -/
def english_only : ℕ := 3

/-- The number of teachers qualified to teach only Japanese -/
def japanese_only : ℕ := 2

/-- The number of teachers qualified to teach both English and Japanese -/
def both : ℕ := 4

/-- The number of English teachers to be selected -/
def english_select : ℕ := 3

/-- The number of Japanese teachers to be selected -/
def japanese_select : ℕ := 3

/-- The total number of possible selections -/
def total_selections : ℕ := 420

theorem teacher_selection_count :
  (Nat.choose both english_select * Nat.choose (english_only + both) english_select) +
  (Nat.choose japanese_only 1 * Nat.choose both 2 * Nat.choose (english_only + both - 2) english_select) +
  (Nat.choose japanese_only 2 * Nat.choose both 1 * Nat.choose (english_only + both - 1) english_select) =
  total_selections := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_selection_count_l955_95526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_one_zero_f_nonnegative_iff_a_geq_one_l955_95501

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-a * x) + Real.sin x - Real.cos x

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := (deriv (f a)) x - (1/3) * x - 1

-- Part 1: F(x) has exactly one zero when a = -1 and x ≥ -π/4
theorem F_has_one_zero :
  ∃! x, x ≥ -Real.pi/4 ∧ F (-1) x = 0 := by
  sorry

-- Part 2: f(x) ≥ 0 for all x ≤ 0 if and only if a ≥ 1
theorem f_nonnegative_iff_a_geq_one (a : ℝ) :
  (∀ x ≤ 0, f a x ≥ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_one_zero_f_nonnegative_iff_a_geq_one_l955_95501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_g_l955_95523

/-- Given functions f and g, prove that f(x) < g(x) for x > 0 under certain conditions -/
theorem f_less_than_g (a b : ℝ) (h1 : a = 2 * b) (h2 : a ≥ 2/3) :
  ∀ x > 0, (λ x : ℝ => Real.sin x - a * x) x < (λ x : ℝ => b * x * Real.cos x) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_g_l955_95523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l955_95575

-- Define the function f and its derivative f'
def f (x : ℝ) : ℝ := x^3 - x^2 - 2*x

def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 2

-- Theorem statement
theorem tangent_line_theorem :
  (f' (-1) = 3) ∧
  (∃ (x₀ : ℝ), x₀ ≠ 0 ∧ f x₀ = f' x₀ * x₀ ∧ (f' x₀ = -2 ∨ f' x₀ = -9/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l955_95575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_minus_x_l955_95540

open Real

theorem min_value_sin_minus_x :
  let f : ℝ → ℝ := λ x ↦ sin x - x
  ∃ (min_val : ℝ), min_val = 1 - π / 2 ∧
    ∀ x, x ∈ Set.Icc 0 (π / 2) → f x ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_minus_x_l955_95540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_properties_l955_95577

-- Define the ellipse
def Ellipse : Set (ℝ × ℝ) := {p | (p.1^2 / 4) + p.2^2 = 1}

-- Define the quadrilateral ABCD
structure Quadrilateral (E : Set (ℝ × ℝ)) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  hA : A ∈ E
  hB : B ∈ E
  hC : C ∈ E
  hD : D ∈ E

-- Define the condition that diagonals pass through origin
def diagonalsThroughOrigin (q : Quadrilateral Ellipse) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ * q.A.1 + (1 - t₁) * q.C.1 = 0 ∧
              t₁ * q.A.2 + (1 - t₁) * q.C.2 = 0 ∧
              t₂ * q.B.1 + (1 - t₂) * q.D.1 = 0 ∧
              t₂ * q.B.2 + (1 - t₂) * q.D.2 = 0

-- Define the slope product condition
noncomputable def slopeProductCondition (q : Quadrilateral Ellipse) : Prop :=
  (q.A.2 - q.C.2) / (q.A.1 - q.C.1) * (q.B.2 - q.D.2) / (q.B.1 - q.D.1) = -1/4

-- Define dot product
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define area (placeholder)
noncomputable def area (q : Quadrilateral Ellipse) : ℝ := sorry

-- Main theorem
theorem ellipse_quadrilateral_properties
  (q : Quadrilateral Ellipse)
  (h₁ : diagonalsThroughOrigin q)
  (h₂ : slopeProductCondition q) :
  (∀ (p : ℝ × ℝ), p ∈ Ellipse →
    (dot_product p q.A) * (dot_product p q.B) ∈ Set.Icc (-3/2) (3/2) \ {0}) ∧
  (area q = 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_properties_l955_95577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2theta_problem_l955_95561

open Real

theorem tan_2theta_problem (θ : ℝ) 
  (h1 : tan (2 * θ) = -2 * sqrt 2) 
  (h2 : θ ∈ Set.Ioo (π/4) (π/2)) : 
  tan θ = sqrt 2 ∧ 
  (2 * (cos (θ/2))^2 - sin θ - 1) / (sqrt 2 * sin (π/4 + θ)) = 2 * sqrt 2 - 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2theta_problem_l955_95561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jen_ate_eleven_suckers_l955_95515

/-- The number of suckers Sienna initially had -/
def sienna_initial : ℕ := sorry

/-- The number of suckers Bailey received from Sienna -/
def bailey_received : ℕ := sienna_initial / 2

/-- The number of suckers Jen initially had -/
def jen_initial : ℕ := sorry

/-- The number of suckers Jen ate -/
def jen_ate : ℕ := jen_initial / 2

/-- The number of suckers Molly received from Jen -/
def molly_received : ℕ := jen_initial - jen_ate

/-- The number of suckers Molly ate -/
def molly_ate : ℕ := 2

/-- The number of suckers Harmony received from Molly -/
def harmony_received : ℕ := molly_received - molly_ate

/-- The number of suckers Harmony kept -/
def harmony_kept : ℕ := 3

/-- The number of suckers Taylor received from Harmony -/
def taylor_received : ℕ := harmony_received - harmony_kept

/-- The number of suckers Taylor ate -/
def taylor_ate : ℕ := 1

/-- The number of suckers Callie received from Taylor -/
def callie_received : ℕ := 5

theorem jen_ate_eleven_suckers : jen_ate = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jen_ate_eleven_suckers_l955_95515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_sector_area_l955_95555

/-- The area of the region inside a regular hexagon with side length 8 but outside 
    the circular sectors with radius 4 centered at each vertex. -/
theorem hexagon_sector_area : 
  (6 * (Real.sqrt 3 / 4 * 8^2)) - (6 * (1 / 6 * Real.pi * 4^2)) = 96 * Real.sqrt 3 - 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_sector_area_l955_95555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_vector_properties_l955_95543

-- Define complex numbers z₁ and z₂
def z₁ : ℂ := 1 - Complex.I
def z₂ (x y : ℝ) : ℂ := x + y * Complex.I

-- Define the condition for parallel vectors
def parallel (z₁ z₂ : ℂ) : Prop := ∃ (k : ℝ), z₂ = k * z₁

-- Define the condition for perpendicular vectors
def perpendicular (z₁ z₂ : ℂ) : Prop := (z₁.re * z₂.re + z₁.im * z₂.im = 0)

theorem complex_vector_properties (x y : ℝ) :
  (parallel z₁ (z₂ x y) → x + y = 0) ∧
  (parallel z₁ (z₂ x y) → ∃ (r : ℝ), (z₂ x y) / z₁ = r) ∧
  (perpendicular z₁ (z₂ x y) → Complex.abs (z₁ + z₂ x y) = Complex.abs (z₁ - z₂ x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_vector_properties_l955_95543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l955_95528

theorem exponential_equation_solution (x : ℝ) : (4 : ℝ) ^ (2 * x + 2) = (16 : ℝ) ^ (3 * x - 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l955_95528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l955_95582

/-- The time (in days) it takes for worker A to complete the work alone -/
noncomputable def time_A : ℝ := 24

/-- The time (in days) it takes for worker B to complete the work alone -/
noncomputable def time_B : ℝ := 24

/-- The time (in days) it takes for workers A and B to complete the work together -/
noncomputable def time_AB : ℝ := 12

/-- The work rate of a worker is the reciprocal of the time it takes them to complete the work -/
noncomputable def work_rate (time : ℝ) : ℝ := 1 / time

theorem work_completion_time :
  work_rate time_A + work_rate time_B = work_rate time_AB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l955_95582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_divisibility_l955_95591

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | 2 => 2
  | (n + 3) => sequence_a (n + 1) * (sequence_a (n + 2) + 1)

theorem sequence_a_divisibility (n : ℕ) (h : n ≥ 100) :
  (sequence_a n)^n ∣ sequence_a (sequence_a n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_divisibility_l955_95591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_weighted_sum_l955_95554

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the weighted sum of distances
noncomputable def weightedSum (X : Point) (t : Triangle) (m n p : ℝ) : ℝ :=
  m * distance X t.A + n * distance X t.B + p * distance X t.C

-- Main theorem
theorem minimize_weighted_sum (t : Triangle) (m n p : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  ∃ X : Point, ∀ Y : Point, 
    weightedSum X t m n p ≤ weightedSum Y t m n p ∧
    ((m ≥ n + p ∨ n ≥ m + p ∨ p ≥ m + n) → 
      (X = t.A ∨ X = t.B ∨ X = t.C)) ∧
    ((m < n + p ∧ n < m + p ∧ p < m + n) → 
      (∃ condition : Prop, condition)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_weighted_sum_l955_95554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_even_first_eight_primes_l955_95598

def first_eight_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_even (n : ℕ) : Bool := n % 2 = 0

def sum_is_even (pair : ℕ × ℕ) : Bool :=
  is_even (pair.fst + pair.snd)

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.filter (fun p => p.fst ≠ p.snd) (List.product l l)

theorem prob_sum_even_first_eight_primes :
  let pairs := distinct_pairs first_eight_primes
  let even_sum_pairs := pairs.filter sum_is_even
  (even_sum_pairs.length : ℚ) / (pairs.length : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_even_first_eight_primes_l955_95598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_720_l955_95541

theorem divisors_of_720 : 
  (Finset.filter (λ x ↦ 720 % x = 0) (Finset.range 721)).card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_720_l955_95541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_difference_l955_95532

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_difference_l955_95532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l955_95538

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that if sin(π/2 + C) = (2b + c) / (2a), a = 4, and the area is 4√3 / 3,
then angle A is 120° and the perimeter is 4 + 8√3 / 3.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  Real.sin (π / 2 + C) = (2 * b + c) / (2 * a) →
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 / 3 →
  A = 2 * π / 3 ∧ a + b + c = 4 + 8 * Real.sqrt 3 / 3 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l955_95538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_points_distance_l955_95510

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A regular hexagon with side length 2 -/
def RegularHexagon : Set Point := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Theorem: Given six points in a regular hexagon with side length 2,
    there always exists a pair of points with distance at most 2/√3 -/
theorem hexagon_points_distance (points : Finset Point) :
  points.card = 6 →
  (∀ p, p ∈ points → p ∈ RegularHexagon) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ 2 / Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_points_distance_l955_95510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_l955_95513

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- The fixed line -/
def fixed_line (x : ℝ) : Prop := x = 2

/-- The left focus of the ellipse -/
def left_focus : ℝ × ℝ := (-2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Distance from a point to the fixed line -/
def distance_to_line (p : ℝ × ℝ) : ℝ := |p.1 - 2|

/-- The locus equation -/
def locus (x y : ℝ) : Prop := y^2 = -8 * x

theorem locus_equation (x y : ℝ) :
  (∀ x₀ y₀, ellipse x₀ y₀ → fixed_line x₀) →
  (distance (x, y) left_focus = distance_to_line (x, y)) ↔ locus x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_l955_95513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l955_95524

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p₁ p₂ : Point) : ℝ :=
  Real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2)

/-- Calculate the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem about the ellipse and the maximum product of distances -/
theorem ellipse_theorem (E : Ellipse) (A₁ A₂ P : Point) (t : ℝ) :
  distance A₁ A₂ = 2 * Real.sqrt 2 →
  eccentricity E = Real.sqrt 2 / 2 →
  P.x = 2 * Real.sqrt 2 ∧ P.y = t ∧ t ≠ 0 →
  (∀ x y, x^2 / E.a^2 + y^2 / E.b^2 = 1 ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ d₁ d₂, d₁ * d₂ ≤ 3/4 ∧
    (∀ d₁' d₂', d₁' * d₂' ≤ d₁ * d₂)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l955_95524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l955_95533

/-- Represents the symbols in the grid --/
inductive GridSymbol : Type
| omega : GridSymbol
| square : GridSymbol
| triangle : GridSymbol

/-- Represents the 3x3 grid --/
def Grid := Fin 3 → Fin 3 → GridSymbol

/-- Assigns a value to each symbol --/
def SymbolValue := GridSymbol → ℕ

/-- The sum of values in a row --/
def rowSum (g : Grid) (sv : SymbolValue) (row : Fin 3) : ℕ :=
  (List.sum (List.map (fun col => sv (g row col)) (List.range 3)))

/-- The sum of values in a column --/
def colSum (g : Grid) (sv : SymbolValue) (col : Fin 3) : ℕ :=
  (List.sum (List.map (fun row => sv (g row col)) (List.range 3)))

/-- The theorem statement --/
theorem grid_sum_theorem (g : Grid) (sv : SymbolValue) 
  (h1 : ∀ s1 s2, s1 ≠ s2 → sv s1 ≠ sv s2)  -- Each symbol has a different value
  (h2 : colSum g sv 0 = 22)  -- Sum of first column
  (h3 : colSum g sv 1 = 12)  -- Sum of second column
  (h4 : colSum g sv 2 = 20)  -- Sum of third column
  (h5 : rowSum g sv 1 = 20)  -- Sum of second row
  (h6 : rowSum g sv 2 = 15)  -- Sum of third row
  : rowSum g sv 0 = 19 := by  -- The sum of the first row is 19
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l955_95533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l955_95506

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 7*x^2 - 4*x + 2) / (x^3 - 3*x^2 + 2*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ∨ 2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l955_95506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_bounds_l955_95589

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => sequence_a n + 1 / sequence_a n

theorem a_100_bounds : 14 < sequence_a 100 ∧ sequence_a 100 < 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_bounds_l955_95589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ratio_l955_95521

theorem right_triangle_ratio (m n a b c : ℝ) 
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_ratio : m < n) 
  (h_hypotenuse_division : m + n = c^2 / a^2) 
  (h_midpoint_angle : ∃ (x y : ℝ), x^2 + y^2 = (c^2 / 4) ∧ 
    (x - m * c / (2 * (m + n)))^2 + (y - a / 2)^2 = 
    (n * c / (2 * (m + n)))^2 + (b / 2)^2) :
  a / Real.sqrt n = b / Real.sqrt m ∧ a / Real.sqrt n = c / Real.sqrt (m + n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ratio_l955_95521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_minus_beta_l955_95539

theorem tan_two_alpha_minus_beta 
  (α β : ℝ) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = 1/2)
  (h3 : 0 < β ∧ β < π/2) -- β is in the first quadrant
  (h4 : Real.cos β = 3/5) :
  Real.tan (2*α - β) = (48 + 25*Real.sqrt 3) / 39 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_minus_beta_l955_95539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_conversion_error_joe_percentage_error_l955_95587

-- Define the correct conversion factor from pounds to kilograms
def correct_lb_to_kg (lb : ℝ) : ℝ := 0.45 * lb

-- Define Joe's method for converting kilograms to pounds
def joe_kg_to_lb (kg : ℝ) : ℝ := 2.2 * kg

-- State the theorem
theorem joe_conversion_error : 
  ∀ (lb : ℝ), lb > 0 → (joe_kg_to_lb (correct_lb_to_kg lb) / lb) = 0.99 := by
  sorry

-- Define the percentage error
noncomputable def percentage_error (actual value : ℝ) : ℝ := (actual - value) / actual * 100

-- State the theorem about the percentage error
theorem joe_percentage_error : 
  ∀ (lb : ℝ), lb > 0 → percentage_error lb (joe_kg_to_lb (correct_lb_to_kg lb)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_conversion_error_joe_percentage_error_l955_95587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_theorem_l955_95588

-- Define the points
variable (A B₀ B₁ C₀ C₁ P₀ P₁ P₂ Q₀ Q₁ : EuclideanSpace ℝ (Fin 2))

-- Define the ellipse
def is_ellipse_point (C B₀ B₁ C₀ : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist C B₀ + dist C B₁ = dist C₀ B₀ + dist C₀ B₁

-- Define the conditions
def conditions (A B₀ B₁ C₀ C₁ P₀ P₁ P₂ Q₀ Q₁ : EuclideanSpace ℝ (Fin 2)) : Prop :=
  is_ellipse_point C₀ B₀ B₁ C₀ ∧ is_ellipse_point C₁ B₀ B₁ C₀ ∧
  (∃ t : ℝ, t > 0 ∧ P₀ = A + t • (B₀ - A)) ∧
  (∃ s : ℝ, s > 0 ∧ Q₀ = C₁ + s • (B₀ - C₁)) ∧
  dist B₀ P₀ = dist B₀ Q₀ ∧
  (∃ r : ℝ, r > 0 ∧ P₁ = B₁ + r • (A - B₁)) ∧
  dist C₁ Q₀ = dist C₁ P₁ ∧
  (∃ u : ℝ, u > 0 ∧ Q₁ = B₁ + u • (C₀ - B₁)) ∧
  dist B₁ P₁ = dist B₁ Q₁ ∧
  (∃ v : ℝ, v > 0 ∧ P₂ = A + v • (B₀ - A)) ∧
  dist C₀ Q₁ = dist C₀ P₂

-- Define concyclic points
def is_concyclic (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    dist center P = radius ∧
    dist center Q = radius ∧
    dist center R = radius ∧
    dist center S = radius

-- Define the theorem
theorem ellipse_points_theorem (A B₀ B₁ C₀ C₁ P₀ P₁ P₂ Q₀ Q₁ : EuclideanSpace ℝ (Fin 2))
  (h : conditions A B₀ B₁ C₀ C₁ P₀ P₁ P₂ Q₀ Q₁) :
  P₀ = P₂ ∧ is_concyclic P₀ Q₀ Q₁ P₁ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_theorem_l955_95588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_30_prime_factors_l955_95553

theorem factorial_30_prime_factors :
  let n : ℕ := 30
  let factorial_n := n.factorial
  let prime_factors := Finset.filter (λ p => Nat.Prime p ∧ p ≤ n) (Finset.range (n + 1))
  (Finset.card prime_factors = 10) ∧
  (Finset.sum prime_factors id = 129) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_30_prime_factors_l955_95553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l955_95520

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 6) :
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) + 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) /
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) - 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) = -Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l955_95520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l955_95549

-- Define the triangle sides as functions of x
noncomputable def side_a (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def side_b (x : ℝ) : ℝ := x^2 - 1
noncomputable def side_c (x : ℝ) : ℝ := 2*x + 1

-- Define the cosine of the angle opposite to side c
noncomputable def cos_C (x : ℝ) : ℝ := (side_a x^2 + side_b x^2 - side_c x^2) / (2 * side_a x * side_b x)

-- Theorem stating that the largest angle is 120°
theorem largest_angle_is_120_degrees (x : ℝ) : 
  cos_C x = -1/2 := by
  sorry

#check largest_angle_is_120_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l955_95549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l955_95579

def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

def ellipse_eq (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

theorem max_distance_circle_ellipse :
  ∃ (max_dist : ℝ),
    max_dist = 1 + (3 * Real.sqrt 6) / 2 ∧
    ∀ (px py qx qy : ℝ),
      circle_eq px py → ellipse_eq qx qy →
      Real.sqrt ((px - qx)^2 + (py - qy)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l955_95579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_weight_approximation_l955_95570

/-- The number of squares on a chessboard -/
def num_squares : ℕ := 64

/-- The number of grains in 1 kg of wheat -/
def grains_per_kg : ℕ := 20000

/-- The logarithm base 10 of 2 -/
noncomputable def log_2 : ℝ := 0.3

/-- The geometric sequence of grains on the chessboard -/
def grain_sequence (n : ℕ) : ℕ := 2^n - 1

/-- The total weight of wheat on the chessboard in kg -/
noncomputable def total_weight : ℝ := (grain_sequence num_squares) / grains_per_kg

/-- The theorem stating the approximate weight of wheat needed -/
theorem wheat_weight_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |total_weight - 10^12| < ε * 10^12 := by
  sorry

#eval num_squares
#eval grains_per_kg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_weight_approximation_l955_95570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_top_four_l955_95542

/-- Represents a football league with the given conditions -/
structure FootballLeague where
  numTeams : Nat
  gamesPerPair : Nat
  pointsPerWin : Nat

/-- Calculates the total number of games played in the league -/
def totalGames (league : FootballLeague) : Nat :=
  (league.numTeams.choose 2) * league.gamesPerPair

/-- Calculates the total points distributed in the league -/
def totalPoints (league : FootballLeague) : Nat :=
  (totalGames league) * league.pointsPerWin

/-- Theorem: The maximum possible points for each of the top 4 teams is 45 -/
theorem max_points_top_four (league : FootballLeague) 
  (h1 : league.numTeams = 8)
  (h2 : league.gamesPerPair = 3)
  (h3 : league.pointsPerWin = 3) :
  ∃ (maxPoints : Nat), maxPoints = 45 ∧ 
  (∀ (topFourPoints : Nat), topFourPoints ≤ maxPoints ∧ 
  (4 * topFourPoints ≤ totalPoints league)) := by
  -- Proof goes here
  sorry

#eval totalPoints ⟨8, 3, 3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_top_four_l955_95542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l955_95590

/-- A geometric sequence with fourth term 720 and seventh term 5040 has first term 720/7 -/
theorem geometric_sequence_first_term : 
  ∀ (a : ℝ) (r : ℝ),
  (∃ (seq : ℕ → ℝ), 
    (∀ n, seq (n + 1) = seq n * r) ∧ 
    seq 0 = a ∧
    seq 3 = 720 ∧ 
    seq 6 = 5040) →
  a = 720 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l955_95590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l955_95557

theorem vector_angle_problem (α β : ℝ) 
  (h1 : π / 2 < α ∧ α < π)
  (h2 : π / 6 < β ∧ β < π / 2)
  (h3 : Real.sin α * 1 + Real.cos α * Real.sqrt 3 = 0)  -- a ⟂ b
  (h4 : Real.sin (α - β) = 3 / 5) :
  α = 2 * π / 3 ∧ β = Real.arcsin ((4 * Real.sqrt 3 + 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l955_95557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l955_95556

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2-x)

-- Theorem statement
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = 4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l955_95556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_response_rate_increase_l955_95583

def survey1_total : ℕ := 70
def survey1_responses : ℕ := 7
def survey2_total : ℕ := 63
def survey2_responses : ℕ := 9

noncomputable def response_rate (responses : ℕ) (total : ℕ) : ℝ :=
  (responses : ℝ) / (total : ℝ) * 100

noncomputable def percentage_increase (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  (new_rate - old_rate) / old_rate * 100

theorem response_rate_increase :
  let rate1 := response_rate survey1_responses survey1_total
  let rate2 := response_rate survey2_responses survey2_total
  abs (percentage_increase rate1 rate2 - 42.86) < 0.01 := by
  sorry

#eval survey1_total
#eval survey1_responses
#eval survey2_total
#eval survey2_responses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_response_rate_increase_l955_95583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_A_equal_pair_C_equal_same_functions_l955_95574

-- Define the functions for pair A
def f_A (x : ℝ) : ℝ := x^2 + 2*x
def g_A (x : ℝ) : ℝ := x^2 + 2*x

-- Define the functions for pair C
noncomputable def f_C (x : ℝ) : ℝ := |x| / x
noncomputable def g_C (x : ℝ) : ℝ := if x > 0 then 1 else -1

-- Theorem for pair A
theorem pair_A_equal : ∀ x : ℝ, f_A x = g_A x := by
  intro x
  simp [f_A, g_A]

-- Theorem for pair C
theorem pair_C_equal : ∀ x : ℝ, x ≠ 0 → f_C x = g_C x := by
  sorry

-- Main theorem combining both pairs
theorem same_functions : (∀ x : ℝ, f_A x = g_A x) ∧ (∀ x : ℝ, x ≠ 0 → f_C x = g_C x) := by
  constructor
  · exact pair_A_equal
  · exact pair_C_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_A_equal_pair_C_equal_same_functions_l955_95574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triangle_count_is_1130_l955_95503

/-- A regular decagon with marked points -/
structure MarkedDecagon where
  vertices : Finset (Fin 10)
  midpoints : Finset (Fin 10)
  total_points : (vertices ∪ midpoints).card = 20

/-- A triangle formed by three distinct points -/
structure Triangle where
  p1 : Fin 20
  p2 : Fin 20
  p3 : Fin 20
  distinct : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

/-- Collinear points on a side of the decagon -/
def collinear_set (d : MarkedDecagon) : Finset (Finset (Fin 20)) :=
  sorry

/-- The number of valid triangles formed by marked points of a regular decagon -/
def valid_triangle_count (d : MarkedDecagon) : ℕ :=
  Nat.choose 20 3 - (collinear_set d).card

/-- Theorem stating the number of valid triangles -/
theorem valid_triangle_count_is_1130 (d : MarkedDecagon) :
  valid_triangle_count d = 1130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triangle_count_is_1130_l955_95503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_l955_95551

/-- Given an integer triangle with two sides equal to 10 and an integer radius of the inscribed circle, 
    the third side is 12. -/
theorem inscribed_circle_triangle : ∀ (a r : ℤ),
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi → (20 + a : ℝ) / 100 * r = Real.sin θ) →
  1 ≤ a ∧ a ≤ 19 →
  r > 0 →
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_l955_95551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l955_95566

/-- The central angle of the sector in the unfolded lateral surface of a cone -/
noncomputable def central_angle (base_radius : ℝ) (slant_height : ℝ) : ℝ :=
  (2 * base_radius * Real.pi) * 180 / (slant_height * Real.pi)

/-- Theorem: For a cone with base radius 8 and slant height 15,
    the central angle of the sector in the unfolded lateral surface is 192° -/
theorem cone_central_angle :
  central_angle 8 15 = 192 := by
  -- Unfold the definition of central_angle
  unfold central_angle
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l955_95566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_c_l955_95516

theorem value_of_c (n : ℕ) (m : ℝ) (h : m^(2*n.succ) = 2) : 2*m^(6*n.succ) - 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_c_l955_95516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l955_95517

/-- Represents a pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Fin 5 → ℕ
  is_valid_sides : (∀ i, sides i ∈ ({17, 23, 18, 30, 34} : Set ℕ)) ∧ 
                   (∀ x ∈ ({17, 23, 18, 30, 34} : Set ℕ), ∃ i, sides i = x)

/-- The area of a CornerCutPentagon is 865 -/
def corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  865

/-- The main theorem stating that the area of the specific pentagon is 865 -/
theorem specific_pentagon_area : 
  ∃ p : CornerCutPentagon, corner_cut_pentagon_area p = 865 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l955_95517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_share_value_l955_95545

noncomputable def profit : ℝ := 41000

def ratios : List ℝ := [2, 3, 4, 4, 6]

noncomputable def max_share (profit : ℝ) (ratios : List ℝ) : ℝ :=
  let total_parts := ratios.sum
  let part_value := profit / total_parts
  (ratios.maximum.get sorry) * part_value

theorem max_share_value :
  max_share profit ratios = 12947.34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_share_value_l955_95545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_annulus_with_points_l955_95595

/-- Given a set of points in a circle, there exists an annulus containing at least a certain number of these points. -/
theorem existence_of_annulus_with_points 
  (points : Finset (ℝ × ℝ)) 
  (h_num_points : points.card = 650) 
  (h_in_circle : ∀ p ∈ points, Real.sqrt ((p.1)^2 + (p.2)^2) ≤ 16) :
  ∃ center : ℝ × ℝ, 
    ((points.filter (fun p => 
      2 ≤ Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ∧ 
      Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 3)).card ≥ 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_annulus_with_points_l955_95595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_cost_optimization_l955_95558

-- Define the pool parameters
noncomputable def pool_volume : ℝ := 8
noncomputable def pool_depth : ℝ := 2
noncomputable def bottom_cost_per_sqm : ℝ := 120
noncomputable def wall_cost_per_sqm : ℝ := 80

-- Define the total cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  320 * (x + 4 / x) + 480

-- Theorem statement
theorem pool_cost_optimization :
  ∀ x > 0,
  total_cost x ≥ 1760 ∧
  (total_cost x = 1760 ↔ x = 2) := by
  sorry

#check pool_cost_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_cost_optimization_l955_95558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_problem_l955_95529

theorem handshake_problem (n : ℕ) (handshakes : Fin n → ℕ) :
  n = 5 ∧
  (∀ i j : Fin n, i ≠ j → handshakes i + handshakes j ≤ n - 1) ∧
  (∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    handshakes i = 4 ∧ handshakes j = 3 ∧ handshakes k = 2 ∧ handshakes l = 1) →
  ∃ m : Fin n, handshakes m = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_problem_l955_95529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_plus_2b_l955_95536

-- Define the function f(x) = log_a x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function of f
noncomputable def f_inverse (a : ℝ) (y : ℝ) : ℝ := a ^ y

-- Theorem statement
theorem min_value_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f_inverse a (-1) = b) : 
  (∀ a' b' : ℝ, a' > 0 → a' ≠ 1 → f_inverse a' (-1) = b' → a + 2*b ≤ a' + 2*b') ∧ 
  a + 2*b = 2 * Real.sqrt 2 := by
  sorry

#check min_value_a_plus_2b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_plus_2b_l955_95536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l955_95548

theorem trigonometric_problem (α β : ℝ)
  (h1 : Real.cos α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = Real.sqrt 10 / 10)
  (h3 : 0 < α ∧ α < Real.pi)
  (h4 : 0 < β ∧ β < Real.pi) :
  Real.cos (2 * α - β) = Real.sqrt 2 / 10 ∧ β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l955_95548
