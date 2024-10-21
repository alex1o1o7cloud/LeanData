import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l858_85877

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the surface area of a cylinder -/
noncomputable def surfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.radius * (c.radius + c.height)

/-- Theorem: The surface area of a specific cylinder -/
theorem cylinder_surface_area :
  ∃ (c : Cylinder),
    c.height = 8 ∧
    surfaceArea { radius := c.radius, height := c.height + 2 } - surfaceArea c = 25.12 ∧
    surfaceArea c = 125.6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l858_85877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_sales_l858_85863

theorem restaurant_sales (meals_8 meals_10 meals_4 : ℕ) 
  (price_8 price_10 price_4 : ℚ) (discount_rate tax_rate : ℚ) :
  meals_8 = 10 ∧ 
  meals_10 = 5 ∧ 
  meals_4 = 20 ∧
  price_8 = 8 ∧ 
  price_10 = 10 ∧ 
  price_4 = 4 ∧
  discount_rate = 1/10 ∧ 
  tax_rate = 1/20 →
  (meals_8 * price_8 + meals_10 * price_10 + meals_4 * price_4) * 
  (1 - discount_rate) * (1 + tax_rate) = 198.45 := by
sorry

#eval (10 * 8 + 5 * 10 + 20 * 4 : ℚ) * (1 - 1/10) * (1 + 1/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_sales_l858_85863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parabola_intersection_l858_85882

noncomputable def circle_set (x y : ℝ) : Prop := x^2 + y^2 = 1

def parabola (x y : ℝ) : Prop := y = x^2 + 1

noncomputable def tangent_slope (a b : ℝ) : ℝ := -a / b

def tangent_line (a b x y : ℝ) : Prop :=
  y = tangent_slope a b * (x - a) + b

def single_intersection (a b : ℝ) : Prop :=
  ∃! x y, tangent_line a b x y ∧ parabola x y

theorem circle_tangent_parabola_intersection :
  ∀ a b : ℝ, circle_set a b ∧ single_intersection a b →
    ((a = 0 ∧ b = 1) ∨
     (a = 2 * Real.sqrt 6 / 5 ∧ b = -1/5) ∨
     (a = -2 * Real.sqrt 6 / 5 ∧ b = -1/5)) :=
by
  sorry

#check circle_tangent_parabola_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parabola_intersection_l858_85882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_meter_percentage_l858_85812

/-- The percentage of defective meters, given the total number of meters examined
    and the number of defective meters. -/
noncomputable def defective_percentage (total : ℝ) (defective : ℕ) : ℝ :=
  (defective : ℝ) / total * 100

/-- The theorem stating that the percentage of defective meters is approximately 0.07%
    when 2 out of 2857.142857142857 meters are rejected as defective. -/
theorem defective_meter_percentage :
  let total := 2857.142857142857
  let defective := 2
  abs (defective_percentage total defective - 0.07) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_meter_percentage_l858_85812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l858_85840

noncomputable def b : ℝ × ℝ := (1/2, Real.sqrt 3 / 2)

theorem projection_of_a_on_b (a : ℝ × ℝ) 
  (h : a.1 * b.1 + a.2 * b.2 = 1/2) :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l858_85840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l858_85851

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  s : ℝ  -- radius of the inscribed sphere
  H : ℝ  -- height of the truncated cone

/-- The sphere is inscribed in the truncated cone -/
def is_inscribed (cone : TruncatedConeWithSphere) : Prop :=
  cone.s = Real.sqrt (cone.R * cone.r) ∧ cone.H = 2 * cone.s

/-- The volume of the truncated cone is three times that of the sphere -/
def volume_relation (cone : TruncatedConeWithSphere) : Prop :=
  (Real.pi * (cone.R^2 + cone.r^2 + cone.R * cone.r) / 3) * cone.H = 3 * (4 * Real.pi * cone.s^3 / 3)

/-- The theorem to be proved -/
theorem truncated_cone_sphere_ratio (cone : TruncatedConeWithSphere) 
  (h_inscribed : is_inscribed cone) (h_volume : volume_relation cone) : 
  cone.R / cone.r = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l858_85851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l858_85878

/-- A line in 2D space represented by slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The perpendicular slope of a given slope -/
noncomputable def perpendicularSlope (m : ℝ) : ℝ := -1 / m

/-- The equation of a line passing through a point with a given slope -/
noncomputable def lineEquation (point : ℝ × ℝ) (slope : ℝ) : Line :=
  { slope := slope
  , intercept := point.2 - slope * point.1 }

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

theorem intersection_of_perpendicular_lines :
  let l1 : Line := { slope := 3, intercept := 4 }
  let l2 := lineEquation (3, 2) (perpendicularSlope l1.slope)
  intersectionPoint l1 l2 = (-3/10, 31/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l858_85878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_fraction_l858_85850

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop := Complex.re z = 0 ∧ Complex.im z ≠ 0

/-- Given that (a+i)/(1+i) is a pure imaginary number, prove that a = -1. -/
theorem pure_imaginary_fraction (a : ℝ) (h : IsPureImaginary ((a + Complex.I) / (1 + Complex.I))) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_fraction_l858_85850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l858_85865

/-- The standard equation of a hyperbola with one asymptote y = 2x and passing through the focus of the parabola y² = 4x is x² - y²/4 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ x^2 - y^2/4 = k) → -- Hyperbola equation with k
  (y = 2*x ∨ y = -2*x) → -- Asymptotes
  (∃ (x₀ y₀ : ℝ), y₀^2 = 4*x₀ ∧ x₀^2 - y₀^2/4 = 1) → -- Passes through focus of parabola
  x^2 - y^2/4 = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l858_85865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l858_85829

/-- Given a three-digit number and two two-digit numbers satisfying certain conditions,
    prove that their sum is 247. -/
theorem sum_of_numbers (A B C : ℕ) : 
  100 ≤ A ∧ A < 1000 ∧  -- A is a three-digit number
  10 ≤ B ∧ B < 100 ∧    -- B is a two-digit number
  10 ≤ C ∧ C < 100 ∧    -- C is a two-digit number
  (7 ∈ A.digits 10) ∨ (7 ∈ B.digits 10) ∨ (7 ∈ C.digits 10) ∧  -- At least one number contains 7
  (3 ∈ B.digits 10) ∧ (3 ∈ C.digits 10) ∧  -- Both B and C contain 3
  (if 7 ∈ A.digits 10 then A else 0) + 
  (if 7 ∈ B.digits 10 then B else 0) + 
  (if 7 ∈ C.digits 10 then C else 0) = 208 ∧  -- Sum of numbers with 7 is 208
  B + C = 76 →  -- Sum of numbers with 3 is 76
  A + B + C = 247 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l858_85829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_max_length_constraint_l858_85862

/-- Represents the properties of a spring -/
structure Spring where
  initialLength : ℝ
  stretchFactor : ℝ
  maxLength : ℝ

/-- Calculates the length of the spring given a mass -/
noncomputable def springLength (s : Spring) (mass : ℝ) : ℝ :=
  min s.maxLength (s.initialLength + s.stretchFactor * mass)

/-- Theorem: A spring with given properties cannot reach 23 cm when 30 kg is hung -/
theorem spring_max_length_constraint (s : Spring) 
  (h1 : s.initialLength = 8)
  (h2 : s.stretchFactor = 0.5)
  (h3 : s.maxLength = 20) :
  springLength s 30 < 23 := by
  sorry

#eval "Spring theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_max_length_constraint_l858_85862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_in_CaCO3_approx_l858_85831

/-- The mass percentage of calcium (Ca) in calcium carbonate (CaCO3) -/
noncomputable def mass_percentage_Ca_in_CaCO3 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_C : ℝ := 12.01
  let molar_mass_O : ℝ := 16.00
  let molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  (molar_mass_Ca / molar_mass_CaCO3) * 100

/-- The mass percentage of calcium in calcium carbonate is approximately 40.04% -/
theorem mass_percentage_Ca_in_CaCO3_approx :
  abs (mass_percentage_Ca_in_CaCO3 - 40.04) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_in_CaCO3_approx_l858_85831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_77_l858_85890

def digits : List Nat := [2, 0, 1, 3]

def is_valid_number (n : Nat) : Bool :=
  1000 ≤ n && n < 2013 && (Nat.digits 10 n).all (fun d => d ∈ digits)

def count_valid_numbers : Nat :=
  (List.range 2013).filter is_valid_number |>.length

theorem count_valid_numbers_eq_77 : count_valid_numbers = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_77_l858_85890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_two_heads_five_coins_l858_85814

/-- The probability of getting at least two heads when tossing five fair coins -/
theorem prob_at_least_two_heads_five_coins : ℝ := by
  let n : ℕ := 5  -- number of coins
  let p : ℝ := 1/2  -- probability of heads for a fair coin
  let prob_less_than_two : ℝ := (n.choose 0) * p^0 * (1-p)^(n-0) + (n.choose 1) * p^1 * (1-p)^(n-1)
  have h : 1 - prob_less_than_two = 13/16 := by sorry
  exact 13/16


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_two_heads_five_coins_l858_85814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_value_l858_85849

/-- The area of the region inside a regular hexagon with side length 4 but outside
    six inscribed semicircles and connecting segments -/
noncomputable def hexagon_semicircles_area : ℝ :=
  let s : ℝ := 4  -- side length of the hexagon
  let r : ℝ := s / 2  -- radius of each semicircle
  let hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area : ℝ := Real.pi * r^2 / 2
  let total_semicircle_area : ℝ := 6 * semicircle_area
  hexagon_area - total_semicircle_area

/-- Theorem stating that the calculated area is equal to 48√3 - 12π -/
theorem hexagon_semicircles_area_value : hexagon_semicircles_area = 48 * Real.sqrt 3 - 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_value_l858_85849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l858_85820

/-- Ben's current age -/
def B : ℕ := sorry

/-- Amy's current age -/
def A : ℕ := sorry

/-- Ben's age 4 years ago was twice Amy's age 4 years ago -/
axiom condition1 : B - 4 = 2 * (A - 4)

/-- Ben's age 8 years ago was three times Amy's age 8 years ago -/
axiom condition2 : B - 8 = 3 * (A - 8)

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- The ratio of their ages after x years is 3:2 -/
axiom future_ratio : (B + x) / (A + x) = 3 / 2

/-- Theorem stating that x equals 4 -/
theorem age_ratio_years : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l858_85820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l858_85804

/-- Line l defined by parametric equations -/
def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 2*t)

/-- Curve C defined by Cartesian equation -/
def curve_C (p : ℝ × ℝ) : Prop := p.2^2 = 8*p.1

/-- Intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ curve_C p}

/-- Length of segment AB -/
noncomputable def length_AB : ℝ := 2 * Real.sqrt 5

/-- Theorem: The length of segment AB is 2√5 -/
theorem length_of_segment_AB :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ dist A B = length_AB :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l858_85804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l858_85825

theorem existence_of_special_set : ∃ (S : Finset ℕ), 
  Finset.card S = 100 ∧ 
  (∀ (A : Finset ℕ), A ⊆ S → Finset.card A = 5 → 
    (A.prod id) % (A.sum id) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l858_85825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_in_first_quadrant_l858_85892

theorem angle_range_in_first_quadrant (α : Real) : 
  (0 < α ∧ α < Real.pi / 2) →  -- α is in the first quadrant
  (Real.cos α ≤ Real.sin α) →  -- given condition
  (Real.pi / 4 ≤ α ∧ α < Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_in_first_quadrant_l858_85892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_difference_l858_85847

/-- Given a circle of radius 3 centered at point A and an equilateral triangle
    with side length 6 and one vertex at A, the difference between the area of
    the region inside the circle but outside the triangle and the area of the
    region inside the triangle but outside the circle is 9(π - √3). -/
theorem circle_triangle_area_difference :
  let r : ℝ := 3
  let s : ℝ := 6
  let circle_area := π * r^2
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  circle_area - triangle_area = 9 * (π - Real.sqrt 3) := by
  sorry

#check circle_triangle_area_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_difference_l858_85847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_axis_intersection_is_0_4_l858_85886

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-axis intersection point of a line -/
noncomputable def y_axis_intersection (l : Line) : ℝ × ℝ :=
  (0, l.y₁ + (l.y₂ - l.y₁) / (l.x₂ - l.x₁) * (-l.x₁))

/-- The line passing through (2, 8) and (5, 14) -/
def given_line : Line := { x₁ := 2, y₁ := 8, x₂ := 5, y₂ := 14 }

theorem y_axis_intersection_is_0_4 : 
  y_axis_intersection given_line = (0, 4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_axis_intersection_is_0_4_l858_85886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_probability_theorem_l858_85891

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cylinder -/
structure Cylinder where
  baseRadius : ℝ
  height : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Check if a point is inside the cylinder -/
def isInsideCylinder (c : Cylinder) (p : Point3D) : Prop :=
  p.x^2 + p.y^2 ≤ c.baseRadius^2 ∧ 0 ≤ p.z ∧ p.z ≤ c.height

/-- The probability function (to be defined) -/
noncomputable def probability (c : Cylinder) : ℝ := sorry

/-- The main theorem to prove -/
theorem cylinder_probability_theorem (c : Cylinder) 
  (h1 : c.baseRadius = 1) 
  (h2 : c.height = 3) : 
  ∃ x, probability c = x ∧ x = x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_probability_theorem_l858_85891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_privateer_overtakes_after_five_hours_l858_85856

/-- Represents the chase between a privateer and a merchantman -/
structure ChaseScenario where
  initialDistance : ℚ
  initialPrivateerSpeed : ℚ
  initialMerchantmanSpeed : ℚ
  initialChaseDuration : ℚ
  laterPrivateerSpeed : ℚ
  laterMerchantmanSpeed : ℚ

/-- The time it takes for the privateer to overtake the merchantman -/
def overtakeTime (chase : ChaseScenario) : ℚ :=
  chase.initialChaseDuration +
  (chase.initialDistance + 
   chase.initialChaseDuration * (chase.initialMerchantmanSpeed - chase.initialPrivateerSpeed)) /
  (chase.laterPrivateerSpeed - chase.laterMerchantmanSpeed)

/-- The specific chase scenario described in the problem -/
def specificChase : ChaseScenario :=
  { initialDistance := 15
  , initialPrivateerSpeed := 10
  , initialMerchantmanSpeed := 7
  , initialChaseDuration := 3
  , laterPrivateerSpeed := 15
  , laterMerchantmanSpeed := 12 }

/-- Theorem stating that the privateer overtakes the merchantman after 5 hours -/
theorem privateer_overtakes_after_five_hours :
  overtakeTime specificChase = 5 := by
  -- Unfold the definitions and perform the calculation
  unfold overtakeTime specificChase
  -- Simplify the arithmetic expression
  simp [add_div, mul_div_assoc]
  -- The proof is complete
  rfl

#eval overtakeTime specificChase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_privateer_overtakes_after_five_hours_l858_85856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_is_three_l858_85835

/-- The cost of a dozen apples -/
def apple_dozen_cost : ℚ := sorry

/-- The cost of a bunch of bananas -/
def banana_bunch_cost : ℚ := sorry

/-- Tony's purchase: 2 dozen apples and 1 bunch of bananas cost $7 -/
axiom tony_purchase : 2 * apple_dozen_cost + banana_bunch_cost = 7

/-- Arnold's purchase: 1 dozen apples and 1 bunch of bananas cost $5 -/
axiom arnold_purchase : apple_dozen_cost + banana_bunch_cost = 5

/-- Theorem: The cost of a bunch of bananas is $3 -/
theorem banana_cost_is_three : banana_bunch_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_is_three_l858_85835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_grid_filling_exists_l858_85848

/-- A function representing the filling of an infinite grid of natural numbers -/
def grid_filling : ℕ → ℕ → ℕ := sorry

/-- Every row contains all natural numbers exactly once -/
axiom row_contains_all (m : ℕ) : 
  ∀ k : ℕ, ∃! n : ℕ, grid_filling m n = k

/-- Every column contains all natural numbers exactly once -/
axiom col_contains_all (n : ℕ) : 
  ∀ k : ℕ, ∃! m : ℕ, grid_filling m n = k

/-- Theorem: It is possible to fill an infinite grid with natural numbers
    such that every row and column contains all natural numbers exactly once -/
theorem infinite_grid_filling_exists : 
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, ∀ k : ℕ, ∃! n : ℕ, f m n = k) ∧ 
    (∀ n : ℕ, ∀ k : ℕ, ∃! m : ℕ, f m n = k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_grid_filling_exists_l858_85848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l858_85818

-- Define the radii of the circles
def R : ℝ := 12
def r : ℝ := 7

-- Define the area of a circle
noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

-- State the theorem
theorem area_between_circles :
  circle_area R - circle_area r = 95 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l858_85818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_magnitude_length_AD_l858_85801

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  D : Real × Real  -- Midpoint of BC

-- Define the given conditions
axiom triangle_condition (t : Triangle) : 
  2 * Real.sin t.A * Real.sin (t.C + Real.pi/6) = Real.sin (t.A + t.C) + Real.sin t.C

axiom side_lengths (t : Triangle) : t.b = 2 ∧ t.c = 1

axiom D_midpoint (t : Triangle) : t.D = ((t.b + t.c) / 2, 0)  -- Assuming BC is on x-axis

-- Theorem statements
theorem angle_A_magnitude (t : Triangle) : t.A = Real.pi/3 := by sorry

theorem length_AD (t : Triangle) : 
  Real.sqrt ((t.D.1 - 0)^2 + (t.D.2 - t.c)^2) = Real.sqrt 7 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_magnitude_length_AD_l858_85801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l858_85866

noncomputable def f (x : ℝ) : ℝ :=
  if -6 ≤ x ∧ x < -3 then (x + 5)^2 - 3
  else if -3 ≤ x ∧ x < 3 then -1/3 * x
  else if 3 ≤ x ∧ x ≤ 6 then -(x - 5)^2 + 3
  else 0

theorem f_properties :
  (∀ x ∈ Set.Icc (-6) 6, f (-x) = -f x) ∧ 
  (∀ x ∈ Set.Icc 0 3, ∃ k, f x = k * x) ∧
  (∀ x ∈ Set.Icc 3 6, ∃ a b c, f x = a * x^2 + b * x + c) ∧
  (∀ x ∈ Set.Icc 3 6, f x ≤ f 5) ∧
  f 5 = 3 ∧
  f 6 = 2 := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l858_85866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_endpoint_l858_85895

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x - 1 ∧ y + 3 ≤ 5) ∨
               (5 = y + 3 ∧ x - 1 ≤ 5) ∨
               (x - 1 = y + 3 ∧ 5 ≥ x - 1)}

-- Helper definition for a ray
def IsRay (r : Set (ℝ × ℝ)) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    r = {x : ℝ × ℝ | ∃ (t : ℝ), t ≥ 0 ∧ x = p + t • (q - p)}

-- Theorem statement
theorem T_is_three_rays_with_common_endpoint :
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Set (ℝ × ℝ)),
    IsRay r₁ ∧ IsRay r₂ ∧ IsRay r₃ ∧
    p ∈ r₁ ∧ p ∈ r₂ ∧ p ∈ r₃ ∧
    T = r₁ ∪ r₂ ∪ r₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_endpoint_l858_85895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_values_xy_and_x_plus_y_l858_85834

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 2 / y + 8 / x = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → 2 / b + 8 / a = 1 → x * y ≤ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 / b + 8 / a = 1 → x + y ≤ a + b) ∧
  x * y = 64 ∧ x + y = 18 := by
  sorry

#check min_values_xy_and_x_plus_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_values_xy_and_x_plus_y_l858_85834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l858_85832

/-- The quadratic equation x^2 - (2m-2)x + (m^2-2m) = 0 has two distinct real roots,
    and if x1 and x2 are its roots with x1^2 + x2^2 = 10, then m = -1 or m = 3 -/
theorem quadratic_equation_roots (m : ℝ) :
  let f := λ x : ℝ => x^2 - (2*m-2)*x + (m^2-2*m)
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  (x1^2 + x2^2 = 10 → m = -1 ∨ m = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l858_85832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l858_85899

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/4) * Real.cos (x + Real.pi/4) + 1/2

theorem f_properties : 
  (∀ x, f (-x) = -f x) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l858_85899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_corn_purchase_l858_85876

/-- Represents the cost of corn per pound -/
def corn_cost : ℝ := 1.10

/-- Represents the cost of beans per pound -/
def bean_cost : ℝ := 0.55

/-- Represents the total pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- Represents the total cost of the purchase -/
def total_cost : ℝ := 21.60

/-- Represents the number of pounds of corn bought -/
def corn_pounds : ℝ := 9.3

theorem jordan_corn_purchase :
  ∃ (bean_pounds : ℝ),
    bean_pounds + corn_pounds = total_pounds ∧
    corn_cost * corn_pounds + bean_cost * bean_pounds = total_cost ∧
    |corn_pounds - 9.3| < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_corn_purchase_l858_85876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l858_85868

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  (Real.sin α + Real.sin β > Real.sin γ) ∧ 
  (Real.cos (α/2)^2 + Real.cos (β/2)^2 > Real.cos (γ/2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l858_85868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_quadratic_inequality_l858_85841

theorem solution_set_of_quadratic_inequality :
  let f : ℝ → ℝ := λ x => 9 * x^2 + 6 * x + 1
  Set.range f ∩ Set.Ici 0 = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_quadratic_inequality_l858_85841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_in_list_l858_85813

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List Int) : List Int :=
  l.filter (λ x => x > 0)

def range (l : List Int) : Int :=
  match l.maximum, l.minimum with
  | some max, some min => max - min
  | _, _ => 0

theorem range_of_positive_integers_in_list :
  let F := consecutive_integers (-4) 12
  range (positive_integers F) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_in_list_l858_85813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_two_max_values_l858_85803

noncomputable def sin_period (x : ℝ) : ℝ := Real.sin (Real.pi * x / 3)

def has_two_max_values (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ n ∧
    (∀ x ∈ Set.Icc 0 n, f x ≤ f x₁) ∧
    (∀ x ∈ Set.Icc 0 n, f x ≤ f x₂)

theorem min_n_for_two_max_values :
  ∀ n : ℕ, (has_two_max_values sin_period n ∧ ∀ m < n, ¬has_two_max_values sin_period m) → n = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_two_max_values_l858_85803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_squared_sin_l858_85857

theorem integral_cos_squared_sin : ∫ x in (0 : Real)..(Real.pi/2), (Real.cos x) ^ 2 * (Real.sin x) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_squared_sin_l858_85857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_to_F_probability_is_five_sixths_l858_85872

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  right_angle : DE^2 + EF^2 = DF^2
  de_eq : DE = 6
  ef_eq : EF = 8
  df_eq : DF = 10

/-- The probability that a randomly selected point in the triangle is closer to F than to D or E -/
noncomputable def closer_to_F_probability (t : RightTriangle) : ℝ :=
  (t.DF / 2) * (t.EF / 2) / ((t.DE * t.EF) / 2)

/-- The theorem stating that the probability is 5/6 -/
theorem closer_to_F_probability_is_five_sixths (t : RightTriangle) :
  closer_to_F_probability t = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_to_F_probability_is_five_sixths_l858_85872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l858_85806

/-- Two circles are externally tangent -/
def ExternallyTangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A circle is internally tangent to another circle -/
def InternallyTangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A chord is a common external tangent to two circles inside a larger circle -/
def IsCommonExternalTangent (chord r₁ r₂ r₃ : ℝ) : Prop := sorry

/-- Given three circles with radii 4, 8, and 16, where the smaller circles are externally
    tangent to each other and internally tangent to the largest circle, the square of the
    length of a chord in the largest circle that is a common external tangent to the
    smaller circles is 7616/9. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 16)
    (h_ext_tangent : ExternallyTangent r₁ r₂)
    (h_int_tangent₁ : InternallyTangent r₁ r₃)
    (h_int_tangent₂ : InternallyTangent r₂ r₃)
    (chord : ℝ) (h_chord : IsCommonExternalTangent chord r₁ r₂ r₃) :
    chord^2 = 7616 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l858_85806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l858_85846

variable (x : ℝ)

def M (x : ℝ) : ℝ := 4 * x^2 - 2 * x - 1
def N (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 5

theorem problem_solution :
  (∀ x, 2 * M x - N x = 5 * x^2 - 2 * x + 3) ∧
  (2 * M (-1) - N (-1) = 10) ∧
  (∀ x, M x > N x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l858_85846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_allocation_schemes_count_l858_85827

/-- Represents the number of classes --/
def num_classes : ℕ := 4

/-- Represents the total number of spots --/
def total_spots : ℕ := 5

/-- Represents the minimum number of spots Class A must receive --/
def min_spots_class_a : ℕ := 2

/-- 
  Calculates the number of ways to distribute spots among classes
  n: number of spots to distribute
  k: number of classes to distribute spots to
--/
def distribute_spots (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

/-- 
  Calculates the total number of allocation schemes
--/
def total_allocation_schemes : ℕ :=
  Finset.sum (Finset.range (total_spots - min_spots_class_a + 1)) (fun i =>
    distribute_spots (total_spots - (min_spots_class_a + i)) (num_classes - 1))

/-- 
  Theorem stating that the total number of allocation schemes is 20
--/
theorem allocation_schemes_count :
  total_allocation_schemes = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_allocation_schemes_count_l858_85827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_equals_B_l858_85875

noncomputable def f (x : ℝ) : ℝ := 4^x - 3 * 2^x + 3

def A : Set ℝ := {x | ∃ y ∈ Set.Icc 1 7, f x = y}

def B : Set ℝ := Set.Iic 0 ∪ Set.Icc 1 2

theorem domain_of_f_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_equals_B_l858_85875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_seven_l858_85898

theorem sum_of_solutions_is_seven : 
  ∃ (x₁ x₂ : ℝ), 
    ((2 : ℝ)^(x₁^2 - 4*x₁ - 4) = (8 : ℝ)^(x₁ - 5)) ∧ 
    ((2 : ℝ)^(x₂^2 - 4*x₂ - 4) = (8 : ℝ)^(x₂ - 5)) ∧ 
    (x₁ ≠ x₂) ∧ 
    (x₁ + x₂ = 7) ∧ 
    (∀ x : ℝ, (2 : ℝ)^(x^2 - 4*x - 4) = (8 : ℝ)^(x - 5) → (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_seven_l858_85898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l858_85880

/-- Represents a train passing a platform -/
structure TrainPassing where
  train_length : ℝ
  time_to_pass_point : ℝ
  platform_length : ℝ

/-- Calculates the time it takes for a train to pass a platform -/
noncomputable def time_to_pass_platform (tp : TrainPassing) : ℝ :=
  let train_speed := tp.train_length / tp.time_to_pass_point
  let total_distance := tp.train_length + tp.platform_length
  total_distance / train_speed

/-- Theorem: A train 1200m long that takes 120s to pass a point will take 210s to pass a 900m platform -/
theorem train_passing_platform : 
  let tp : TrainPassing := { train_length := 1200, time_to_pass_point := 120, platform_length := 900 }
  time_to_pass_platform tp = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l858_85880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_can_avoid_wolf_l858_85809

-- Define the game board as an infinite plane
def InfinitePlane := ℝ × ℝ

-- Define the maximum distance a piece can move
def MaxMoveDistance : ℝ := 1

-- Define the number of sheep
def NumSheep : ℕ := 50

-- Define a game state
structure GameState where
  wolfPos : InfinitePlane
  sheepPos : Fin NumSheep → InfinitePlane

-- Define a move for a piece
def Move := InfinitePlane

-- Function to check if a move is valid (within MaxMoveDistance)
def isValidMove (start finish : InfinitePlane) : Prop :=
  Real.sqrt ((start.1 - finish.1)^2 + (start.2 - finish.2)^2) ≤ MaxMoveDistance

-- Define a strategy for sheep
def SheepStrategy := GameState → Fin NumSheep → Move

-- Theorem: There exists a strategy for sheep such that the wolf can never catch any sheep
theorem sheep_can_avoid_wolf :
  ∃ (strategy : SheepStrategy),
    ∀ (initialState : GameState) (wolfMoves : ℕ → Move),
      ∀ (n : ℕ) (i : Fin NumSheep),
        let finalState := initialState  -- Placeholder for the actual game progression
        ∀ (i : Fin NumSheep), finalState.wolfPos ≠ finalState.sheepPos i := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_can_avoid_wolf_l858_85809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_two_l858_85837

theorem det_B_equals_two (x y : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  (B + 2 * B⁻¹ = 0) → Matrix.det B = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_two_l858_85837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_from_trig_identity_l858_85836

theorem triangle_angles_from_trig_identity 
  (A B C : ℝ) 
  (h1 : 0 < A ∧ A < Real.pi) 
  (h2 : 0 < B ∧ B < Real.pi) 
  (h3 : 0 < C ∧ C < Real.pi) 
  (h4 : Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2)) : 
  A + B + C = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_from_trig_identity_l858_85836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_fan_usage_days_l858_85843

/-- Calculates the number of days an electric fan was used based on its power rating,
    daily usage hours, and total energy consumption. -/
noncomputable def calculateDays (power : ℝ) (hoursPerDay : ℝ) (totalEnergy : ℝ) : ℝ :=
  totalEnergy * 1000 / (power * hoursPerDay)

/-- Theorem stating that given the specific conditions of John's electric fan usage,
    the number of days calculated is 30. -/
theorem john_fan_usage_days :
  let power : ℝ := 75  -- Fan power in watts
  let hoursPerDay : ℝ := 8  -- Hours of usage per day
  let totalEnergy : ℝ := 18  -- Total energy consumption in kWh
  calculateDays power hoursPerDay totalEnergy = 30 := by
  -- Unfold the definition of calculateDays
  unfold calculateDays
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_fan_usage_days_l858_85843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_softball_purchase_count_l858_85815

def original_dodgeball_count : ℕ := 15
def dodgeball_price : ℚ := 5
def budget_increase_percentage : ℚ := 20 / 100
def softball_price : ℚ := 9

def original_budget : ℚ := original_dodgeball_count * dodgeball_price

def new_budget : ℚ := original_budget * (1 + budget_increase_percentage)

def softball_count : ℕ := (new_budget / softball_price).floor.toNat

theorem softball_purchase_count : softball_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_softball_purchase_count_l858_85815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_radius_sum_l858_85860

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle

/-- Represents the configuration of circles as described in the problem -/
structure CircleConfiguration where
  A : Circle
  B : Circle
  C : Circle
  D : Circle
  E : Circle
  T : EquilateralTriangle

/-- The main theorem statement -/
theorem circle_configuration_radius_sum (config : CircleConfiguration) :
  config.A.radius = 12 ∧
  config.B.radius = 4 ∧
  config.C.radius = 3 ∧
  config.D.radius = 3 ∧
  (∃ m n : ℕ, config.E.radius = m / n ∧ Nat.Coprime m n) →
  ∃ m n : ℕ, config.E.radius = m / n ∧ m + n = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_radius_sum_l858_85860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spend_on_books_l858_85867

def book_prices : List ℚ := [25, 18, 21, 35, 12, 10]

def discount_over_22 (price : ℚ) : ℚ :=
  if price > 22 then price * (1 - 30/100) else price

def discount_under_20 (price : ℚ) : ℚ :=
  if price < 20 then price * (1 - 20/100) else price

def apply_discount (price : ℚ) : ℚ :=
  if price > 22 then discount_over_22 price
  else if price < 20 then discount_under_20 price
  else price

theorem total_spend_on_books :
  (book_prices.map apply_discount).sum = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spend_on_books_l858_85867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_loss_percentage_l858_85842

/-- Represents the selling price of an article -/
def selling_price : ℝ → ℝ := λ x => x

/-- Represents the cost price of an article -/
def cost_price : ℝ := 250

/-- Calculates the profit percentage given selling price and cost price -/
noncomputable def profit_percentage (sp : ℝ) (cp : ℝ) : ℝ :=
  (sp - cp) / cp * 100

theorem initial_loss_percentage :
  ∃ (x : ℝ),
    selling_price x = x ∧
    profit_percentage (x + 50) cost_price = 10 ∧
    profit_percentage x cost_price = -10 :=
by
  -- The proof goes here
  sorry

#eval cost_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_loss_percentage_l858_85842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_problems_l858_85896

-- Define the set of all problems
def AllProblems : Set Nat := {1, 2, 3, 4, 5}

-- Define what makes a problem solvable by independence tests
def SolvableByIndependenceTest (p : Nat) : Prop :=
  ∃ (var1 var2 : Type) (h1 : Fintype var1) (h2 : Fintype var2),
  (p ∈ AllProblems) ∧ 
  (∃ (relation : var1 → var2 → Prop), True)

-- Define the set of problems solvable by independence tests
def IndependenceTestProblems : Set Nat :=
  {p ∈ AllProblems | SolvableByIndependenceTest p}

-- Theorem statement
theorem independence_test_problems :
  IndependenceTestProblems = {2, 4, 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_problems_l858_85896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l858_85845

-- Define the circles and their properties
structure CircleConfig where
  K : Set (ℝ × ℝ)  -- Circle K
  L : Set (ℝ × ℝ)  -- Circle L
  M : Set (ℝ × ℝ)  -- Circle M
  A : ℝ × ℝ        -- Point A
  B : ℝ × ℝ        -- Point B

-- Helper definitions
def IsDiameter (circle : Set (ℝ × ℝ)) (a b : ℝ × ℝ) : Prop := sorry
def IsTangent (circle1 circle2 : Set (ℝ × ℝ)) : Prop := sorry
def TouchesAtCenter (circle1 circle2 : Set (ℝ × ℝ)) : Prop := sorry
def Line (a b : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
noncomputable def area (circle : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the geometric conditions
def validConfig (c : CircleConfig) : Prop :=
  IsDiameter c.K c.A c.B ∧
  IsTangent c.L c.K ∧ IsTangent c.L (Line c.A c.B) ∧ TouchesAtCenter c.L c.K ∧
  IsTangent c.M c.K ∧ IsTangent c.M c.L ∧ IsTangent c.M (Line c.A c.B)

-- Define the theorem
theorem circle_area_ratio (c : CircleConfig) (h : validConfig c) :
  (area c.K) / (area c.M) = 6 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l858_85845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_C7H6O2_l858_85817

/-- Represents the atomic weights of elements --/
def AtomicWeight : Type := ℝ

/-- Represents the number of atoms of an element in a compound --/
def AtomCount : Type := ℕ

/-- Calculates the molecular weight of a compound --/
def molecular_weight (c_count h_count o_count : ℕ) 
  (c_weight h_weight o_weight : ℝ) : ℝ :=
  (c_count : ℝ) * c_weight + (h_count : ℝ) * h_weight + (o_count : ℝ) * o_weight

/-- Theorem: The molecular weight of C₇H₆O₂ is approximately 122 --/
theorem molecular_weight_C7H6O2 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |molecular_weight 7 6 2 12.01 1.008 16.00 - 122| < ε := by
  sorry

#eval molecular_weight 7 6 2 12.01 1.008 16.00

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_C7H6O2_l858_85817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_is_twelve_l858_85810

/-- Represents a triangle with a given base length -/
structure Triangle where
  base : ℝ

/-- Represents a trapezoid with two base lengths -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ

/-- The altitude of a shape -/
def altitude : ℝ := 1 -- Placeholder value, can be changed as needed

/-- Calculates the area of a triangle -/
noncomputable def triangle_area (t : Triangle) (h : ℝ) : ℝ := (1/2) * t.base * h

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) (h : ℝ) : ℝ := ((t.shorter_base + t.longer_base) / 2) * h

/-- Calculates the median of a trapezoid -/
noncomputable def trapezoid_median (t : Trapezoid) : ℝ := (t.shorter_base + t.longer_base) / 2

theorem trapezoid_median_is_twelve 
  (t : Triangle) 
  (z : Trapezoid) 
  (h : ℝ) 
  (h_triangle_base : t.base = 24) 
  (h_trapezoid_base_diff : z.longer_base = z.shorter_base + 10) 
  (h_equal_areas : triangle_area t h = trapezoid_area z h) :
  trapezoid_median z = 12 := by
  sorry

#check trapezoid_median_is_twelve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_is_twelve_l858_85810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_max_x_plus_2y_l858_85802

/-- Circle C with given properties -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  /-- C is tangent to the y-axis -/
  tangent_y_axis : ∃ (x : ℝ), x > 0 ∧ (x^2 = radius^2)
  /-- The center of C lies on the line x - 2y = 0 -/
  center_on_line : ∃ (m : ℝ), center = (2*m, m)
  /-- The chord length intercepted by the positive semiaxis of x is 2√3 -/
  chord_length : ∃ (x : ℝ), x > 0 ∧ (x^2 = radius^2) ∧ x = Real.sqrt 3

/-- The equation of circle C is (x-2)² + (y-1)² = 4 -/
theorem circle_equation (c : CircleC) :
  c.center = (2, 1) ∧ c.radius = 2 :=
sorry

/-- The maximum value of x + 2y for any point (x,y) on circle C is 4 + 2√5 -/
theorem max_x_plus_2y (c : CircleC) :
  ∃ (max : ℝ), max = 4 + 2 * Real.sqrt 5 ∧
  ∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 4 → x + 2*y ≤ max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_max_x_plus_2y_l858_85802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_l858_85826

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℚ
  width : ℚ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℚ := 2 * (r.length + r.width)

/-- The original rectangle -/
def original : Rectangle := { length := 8, width := 6 }

/-- The large rectangle after folding and cutting -/
def large : Rectangle := { length := original.length / 2, width := original.width }

/-- One of the small rectangles after folding and cutting -/
def small : Rectangle := { length := original.length / 2, width := original.width / 2 }

theorem perimeter_ratio :
  perimeter large / perimeter small = 10 / 7 := by
  -- Expand the definitions and simplify
  simp [perimeter, large, small, original]
  -- Perform the arithmetic
  norm_num

#eval perimeter large / perimeter small

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_l858_85826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_subset_stable_points_range_of_a_for_equal_nonempty_sets_l858_85819

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the sets A and B
def A (f : ℝ → ℝ) : Set ℝ := {x | f x = x}
def B (f : ℝ → ℝ) : Set ℝ := {x | f (f x) = x}

-- Part 1: Prove that A ⊆ B
theorem fixed_points_subset_stable_points : A f ⊆ B f := by
  sorry

-- Define the specific function f(x) = ax^2 - 1
def f_specific (a : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - 1

-- Part 2: Prove the range of a for which A = B ≠ ∅
theorem range_of_a_for_equal_nonempty_sets (a : ℝ) :
  (A (f_specific a) = B (f_specific a) ∧ (A (f_specific a)).Nonempty) ↔ 
  a ∈ Set.Icc (-1/4 : ℝ) (3/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_subset_stable_points_range_of_a_for_equal_nonempty_sets_l858_85819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l858_85844

/-- Calculates the speed of a bus including stoppages -/
noncomputable def bus_speed_with_stoppages (speed_without_stoppages : ℝ) (stop_time : ℝ) : ℝ :=
  let running_time := (60 - stop_time) / 60
  let distance := speed_without_stoppages * running_time
  distance

/-- Theorem stating that a bus with a speed of 82 kmph excluding stoppages
    and stopping for 5.12 minutes per hour has a speed of approximately 75 kmph
    including stoppages -/
theorem bus_speed_theorem :
  let speed_without_stoppages := (82 : ℝ)
  let stop_time := (5.12 : ℝ)
  let speed_with_stoppages := bus_speed_with_stoppages speed_without_stoppages stop_time
  (speed_with_stoppages ≥ 74.9 ∧ speed_with_stoppages ≤ 75.1) :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval bus_speed_with_stoppages 82 5.12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l858_85844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ellipse_problem_l858_85828

/-- Helper function to define a tangent line -/
def is_tangent_line (P F : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ Q ∈ C, ∀ R ∈ C, R ≠ Q → (P.1 - F.1) * (R.1 - Q.1) + (P.2 - F.2) * (R.2 - Q.2) ≠ 0

/-- Given a point P, a circle C, an ellipse E, and their properties, prove the values of m, 
the equation of E, and the range of the dot product AP · AQ. -/
theorem circle_ellipse_problem (m a b : ℝ) (P : ℝ × ℝ) (A : ℝ × ℝ) :
  P = (4, 4) →
  A = (3, 1) →
  m < 3 →
  a > b →
  b > 0 →
  (∀ x y, (x - m)^2 + y^2 = 5 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t)) →
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t)) →
  A ∈ Set.range (λ t : ℝ × ℝ ↦ t) →
  A ∈ Set.range (λ t : ℝ × ℝ ↦ t) →
  (∃ F₁ : ℝ × ℝ, F₁.1 < A.1 ∧ F₁.2 = 0 ∧ is_tangent_line P F₁ (Set.range (λ t : ℝ × ℝ ↦ t))) →
  m = 1 ∧
  a^2 = 18 ∧
  b^2 = 2 ∧
  (∀ Q ∈ Set.range (λ t : ℝ × ℝ ↦ t), 
    -12 ≤ (Q.1 - A.1) * (P.1 - A.1) + (Q.2 - A.2) * (P.2 - A.2) ∧
    (Q.1 - A.1) * (P.1 - A.1) + (Q.2 - A.2) * (P.2 - A.2) ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ellipse_problem_l858_85828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_quadratic_l858_85879

/-- Represents the number of intersection points between a function and the x-axis. -/
def NumberOfIntersectionPoints (f : ℝ → ℝ) : ℕ :=
sorry

/-- Given a < b < c and a + b + c = 0, the number of intersection points
    between y = ax^2 + bx + c and the x-axis depends on the discriminant's sign. -/
theorem intersection_points_quadratic (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) :
  ∃ (d : ℝ), NumberOfIntersectionPoints (λ x ↦ a*x^2 + b*x + c) = 
    if d > 0 then 2
    else if d = 0 then 1
    else 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_quadratic_l858_85879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_shift_period_l858_85858

noncomputable def f (x : ℝ) : ℝ := Real.tan (x - Real.pi/3)

theorem tan_shift_period (k : ℝ) :
  ∀ x₁ x₂ : ℝ, f x₁ = k ∧ f x₂ = k ∧ x₁ < x₂ ∧ ∀ x, x₁ < x ∧ x < x₂ → f x ≠ k →
  x₂ - x₁ = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_shift_period_l858_85858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l858_85871

/-- Given a hyperbola with asymptotes y = ± (1/3)x and one focus at (0, 2√5),
    its standard equation is y²/2 - x²/18 = 1 -/
theorem hyperbola_standard_equation 
  (h : Set (ℝ × ℝ)) 
  (asymptote : Set (ℝ × ℝ) → Set (ℝ → Set ℝ))
  (focus : (0, 2 * Real.sqrt 5) ∈ h) :
  ∃ (standard_equation : ℝ → ℝ → Prop),
    standard_equation = fun (x y : ℝ) => y^2 / 2 - x^2 / 18 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l858_85871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l858_85808

/-- The integer part of a real number -/
noncomputable def integerPart (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part of a real number -/
noncomputable def fractionalPart (x : ℝ) : ℝ :=
  x - (integerPart x : ℝ)

/-- Theorem: If {x}([x]-1) < x-2, then x ≥ 3 -/
theorem inequality_solution (x : ℝ) :
  fractionalPart x * ((integerPart x : ℝ) - 1) < x - 2 → x ≥ 3 := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l858_85808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_obtainable_l858_85894

def initial_set : List ℤ := [-1, 2, -3, 4, -5, 6]

def operation (a b : ℤ) : List ℤ := [2*a + b, 2*b + a]

def is_valid_sextuple (s : List ℤ) : Prop :=
  s.length = 6 ∧ s.sum = initial_set.sum

def can_obtain (s : List ℤ) : Prop :=
  ∃ (sequence : List (List ℤ)),
    sequence.head? = some initial_set ∧
    sequence.getLast? = some s ∧
    ∀ i < sequence.length - 1,
      ∃ (a b : ℤ) (rest : List ℤ),
        sequence[i]! = a :: b :: rest ∧
        sequence[i+1]! = (operation a b) ++ rest

theorem only_one_obtainable : 
  (can_obtain [0, 0, 0, 3, -9, 9]) ∧
  (¬ can_obtain [0, 1, 1, 3, 6, -6]) ∧
  (¬ can_obtain [0, 0, 0, 3, -6, 9]) ∧
  (¬ can_obtain [0, 1, 1, -3, 6, -9]) ∧
  (¬ can_obtain [0, 0, 2, 5, 5, 6]) :=
by sorry

#check only_one_obtainable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_obtainable_l858_85894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l858_85887

/-- The quadratic function y = -2x^2 + 5x - 2 -/
noncomputable def y (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 2

/-- The maximum value of y -/
noncomputable def max_y : ℝ := 9 / 8

/-- Theorem: If 9/m is the maximum value of y, then m = 8 -/
theorem quadratic_max_value (m : ℝ) : (∀ x, y x ≤ 9/m) → m = 8 := by
  sorry

#check quadratic_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l858_85887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_correct_sum_of_perimeters_formula_l858_85807

/-- Represents the sum of perimeters of a series of equilateral triangles -/
noncomputable def sum_of_perimeters (n : ℕ) : ℝ :=
  240 * (1 - (1/2)^n)

/-- Proves that the sum of perimeters formula is correct for the given triangle series -/
theorem sum_of_perimeters_correct (n : ℕ) (h : n > 4) :
  sum_of_perimeters n =
    (40 * 3) * (1 - (1/2)^n) / (1 - 1/2) :=
by sorry

/-- Proves that the sum of perimeters is equal to the given formula -/
theorem sum_of_perimeters_formula (n : ℕ) (h : n > 4) :
  sum_of_perimeters n = 240 * (1 - (1/2)^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_correct_sum_of_perimeters_formula_l858_85807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l858_85881

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 4)

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f a b x = -f a b (-x)) →
  f a b (1/2) = 2/17 →
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → g x = x / (x^2 + 4)) ∧
    (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f a b x = g x) ∧
    (∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → g x < g y) ∧
    {a : ℝ | g (a+1) + g (1-2*a) > 0} = Set.Ioo (-1/2 : ℝ) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l858_85881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_extra_chores_l858_85822

/-- Calculates the average number of extra chores per week given the total amount earned,
    number of weeks, fixed allowance, and extra chore pay rate. -/
def average_extra_chores (total_amount : ℚ) (num_weeks : ℚ) (fixed_allowance : ℚ) (extra_chore_pay : ℚ) : ℚ :=
  ((total_amount / num_weeks) - fixed_allowance) / extra_chore_pay

/-- Proves that given the specified conditions, the average number of extra chores per week is 15. -/
theorem carol_extra_chores :
  let total_amount : ℚ := 425
  let num_weeks : ℚ := 10
  let fixed_allowance : ℚ := 20
  let extra_chore_pay : ℚ := 3/2
  average_extra_chores total_amount num_weeks fixed_allowance extra_chore_pay = 15 := by
  -- Unfold the definition of average_extra_chores
  unfold average_extra_chores
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_extra_chores_l858_85822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_king_route_length_l858_85864

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a king's move on the chessboard --/
inductive KingMove
  | Diagonal
  | Orthogonal

/-- Represents a king's route on the chessboard --/
def KingRoute := List KingMove

/-- The length of a king's move --/
noncomputable def moveLength (m : KingMove) : ℝ :=
  match m with
  | KingMove.Diagonal => Real.sqrt 2
  | KingMove.Orthogonal => 1

/-- The total length of a king's route --/
noncomputable def routeLength (r : KingRoute) : ℝ :=
  r.map moveLength |>.sum

/-- Theorem: Maximum length of king's route on 9x9 chessboard --/
theorem max_king_route_length (board : Chessboard) (route : KingRoute) :
    board.size = 9 →
    route.length = board.size^2 - 1 →
    routeLength route ≤ 16 + 64 * Real.sqrt 2 := by
  sorry

#check max_king_route_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_king_route_length_l858_85864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_triangle_angle_sum_l858_85893

/-- The measure of an interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (180 * (n - 2) : ℝ) / n

/-- The sum of measures of an angle in a regular pentagon and an angle in a regular triangle -/
noncomputable def angle_sum : ℝ := interior_angle 5 + interior_angle 3

theorem pentagon_triangle_angle_sum : angle_sum = 168 := by
  -- Unfold the definitions
  unfold angle_sum interior_angle
  -- Simplify the expressions
  simp [Nat.cast_sub, Nat.cast_mul, Nat.cast_ofNat]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_triangle_angle_sum_l858_85893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equation_l858_85853

theorem power_sum_equation (a b c d : ℕ) (n : ℕ+) :
  (n : ℕ)^a + (n : ℕ)^b + (n : ℕ)^c = (n : ℕ)^d ↔ 
  (n = 2 ∧ b = a ∧ c = a + 1 ∧ d = a + 2) ∨
  (n = 3 ∧ b = a ∧ c = a ∧ d = a + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equation_l858_85853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_circumradii_l858_85859

-- Define a triangle in a plane
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define a function to calculate the radius of a circumcircle
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define a function to find the foot of an altitude
noncomputable def altitude_foot (t : Triangle) (vertex : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the sum of circumradii function
noncomputable def sum_of_circumradii (t : Triangle) (M : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  circumradius ⟨t.A, t.B, M⟩ + circumradius ⟨t.B, t.C, M⟩

-- State the theorem
theorem min_sum_circumradii (t : Triangle) :
  ∃ M : EuclideanSpace ℝ (Fin 2), ∀ P : EuclideanSpace ℝ (Fin 2),
    sum_of_circumradii t M ≤ sum_of_circumradii t P ∧
    M = altitude_foot t t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_circumradii_l858_85859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_neg_two_f_inverse_two_l858_85870

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 1 then x^2 + 1
  else if x < -1 then 2*x + 3
  else 0  -- This case should never occur in our problem

-- Theorem 1: f(f(-2)) = 2
theorem f_f_neg_two : f (f (-2)) = 2 := by sorry

-- Theorem 2: f(-1) = 2 and for all x ≠ -1, f(x) ≠ 2
theorem f_inverse_two :
  f (-1) = 2 ∧ ∀ x : ℝ, x ≠ -1 → f x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_neg_two_f_inverse_two_l858_85870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_k_l858_85823

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + by)^2 -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x y, p x y = (a * x + b * y) ^ 2

/-- The given polynomial -/
def P (k : ℝ) : ℝ → ℝ → ℝ := fun x y ↦ x^2 + k*x*y + 16*y^2

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (P k) → k = 8 ∨ k = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_k_l858_85823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_is_100_l858_85885

/-- A function that checks if a number is a palindrome in the form ABCBA -/
def isPalindrome (n : ℕ) : Bool :=
  if 1000 ≤ n ∧ n < 2000 then
    let a := n / 10000
    let b := (n / 1000) % 10
    let c := (n / 100) % 10
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a ∧ a = 1
  else
    false

/-- The count of palindromes between 1000 and 2000 -/
def palindromeCount : ℕ := (List.range 1000).map (· + 1000) |>.filter isPalindrome |>.length

/-- Theorem stating that the count of palindromes between 1000 and 2000 is 100 -/
theorem palindrome_count_is_100 : palindromeCount = 100 := by
  sorry

#eval palindromeCount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_is_100_l858_85885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l858_85888

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties 
  (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : -π/2 ≤ φ ∧ φ < π/2)
  (h_sym : ∀ x, f ω φ (2*π/3 - x) = f ω φ (2*π/3 + x))
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (α : ℝ)
  (h_α : π/6 < α ∧ α < 2*π/3)
  (h_f_α : f ω φ (α/2) = Real.sqrt 3 / 4) :
  ω = 2 ∧ 
  φ = -π/6 ∧ 
  Real.cos (α + 3*π/2) = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l858_85888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_cos_squared_l858_85861

theorem integral_sin_cos_squared : 
  (∫ x in (0 : ℝ)..Real.pi, 16 * (Real.sin (x/2))^2 * (Real.cos (x/2))^6) = (5*Real.pi)/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_cos_squared_l858_85861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l858_85839

/-- Represents a four-digit number as a list of its digits -/
def FourDigitNumber := List Nat

/-- Checks if all digits in the list are distinct -/
def distinct_digits (n : FourDigitNumber) : Prop :=
  n.length = 4 ∧ n.Nodup

/-- Checks if the sum of digits is 16 -/
def sum_16 (n : FourDigitNumber) : Prop :=
  n.sum = 16

/-- Checks if one digit is twice another -/
def has_double (n : FourDigitNumber) : Prop :=
  ∃ i j, i ∈ n.toFinset ∧ j ∈ n.toFinset ∧ i = 2 * j

/-- Checks if one digit is three times another -/
def has_triple (n : FourDigitNumber) : Prop :=
  ∃ i j, i ∈ n.toFinset ∧ j ∈ n.toFinset ∧ i = 3 * j

/-- Checks if a number satisfies all conditions -/
def satisfies_conditions (n : FourDigitNumber) : Prop :=
  distinct_digits n ∧ sum_16 n ∧ has_double n ∧ has_triple n

/-- Converts a list of digits to a natural number -/
def list_to_nat (n : FourDigitNumber) : Nat :=
  n.foldl (fun acc d => acc * 10 + d) 0

/-- Theorem: 1267 is the smallest four-digit number satisfying all conditions -/
theorem smallest_satisfying_number :
  satisfies_conditions [1, 2, 6, 7] ∧
  ∀ n : FourDigitNumber, satisfies_conditions n → 1267 ≤ list_to_nat n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l858_85839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similarity_theorem_l858_85816

open Matrix

theorem matrix_similarity_theorem (M N : Matrix (Fin 2) (Fin 2) ℂ) 
  (hM : M * M = 0)
  (hN : N * N = 0)
  (hMN : M * N + N * M = 1) :
  ∃ (A : Matrix (Fin 2) (Fin 2) ℂ), IsUnit A ∧ 
    (M = A * !![0, 1; 0, 0] * A⁻¹) ∧
    (N = A * !![0, 0; 1, 0] * A⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similarity_theorem_l858_85816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equality_l858_85897

noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3*x - 50

theorem h_triple_equality (b : ℝ) :
  b < 0 → (h (h (h 15)) = h (h (h b)) ↔ b = -55/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equality_l858_85897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_sum_diff_divisible_2009_l858_85821

theorem smallest_n_sum_diff_divisible_2009 : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (A : Finset ℤ), A.card = n → 
    ∃ (a b : ℤ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ 
    ((a + b) % 2009 = 0 ∨ (a - b) % 2009 = 0)) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (B : Finset ℤ), B.card = m ∧
    ∀ (a b : ℤ), a ∈ B → b ∈ B → a ≠ b → 
    ((a + b) % 2009 ≠ 0 ∧ (a - b) % 2009 ≠ 0)) ∧
  n = 1006 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_sum_diff_divisible_2009_l858_85821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toys_per_day_calculation_l858_85883

/-- A factory produces toys with the following parameters:
  * total_weekly_production: The total number of toys produced per week
  * working_days_per_week: The number of days workers work per week
-/
structure ToyFactory where
  total_weekly_production : ℕ
  working_days_per_week : ℕ

/-- Calculate the number of toys produced per day, rounded to the nearest integer -/
def toys_per_day (factory : ToyFactory) : ℕ :=
  (factory.total_weekly_production + factory.working_days_per_week / 2) / factory.working_days_per_week

/-- The theorem states that for a factory producing 6400 toys per week
    with workers working 3 days a week, the number of toys produced
    per day is 2133 -/
theorem toys_per_day_calculation (factory : ToyFactory)
    (h1 : factory.total_weekly_production = 6400)
    (h2 : factory.working_days_per_week = 3) :
    toys_per_day factory = 2133 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toys_per_day_calculation_l858_85883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l858_85824

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 2*x + 1)

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x, ∃ y, f a x = y

-- Define the proposition q
def q (a : ℝ) : Prop := ∀ x ∈ Set.Icc (1/2) 2, x + 1/x > a

-- Theorem statement
theorem range_of_a (a : ℝ) (h_p : p a) (h_q : q a) : 1 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l858_85824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_in_seat_one_l858_85833

-- Define the set of people
inductive Person : Type
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person

-- Define the seating arrangement
def seating : Person → Nat := sorry

-- Define the "next to" relation
def next_to (p1 p2 : Person) : Prop :=
  ∃ (n : Nat), (seating p1 = n ∧ seating p2 = n + 1) ∨ (seating p2 = n ∧ seating p1 = n + 1)

-- State the theorem
theorem dana_in_seat_one :
  (seating Person.Abby = 3) →                    -- Abby is in seat 3
  (¬ next_to Person.Dana Person.Abby) →          -- Dana is not next to Abby
  (next_to Person.Bret Person.Carl) →            -- Bret is next to Carl
  (∀ p : Person, 1 ≤ seating p ∧ seating p ≤ 4) →  -- All seats are between 1 and 4
  (∀ p1 p2 : Person, p1 ≠ p2 → seating p1 ≠ seating p2) →  -- Each person has a unique seat
  (seating Person.Dana = 1) :=                   -- Dana is in seat 1
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_in_seat_one_l858_85833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_b_729_equals_negative_three_halves_implies_b_equals_one_over_81_l858_85873

theorem log_b_729_equals_negative_three_halves_implies_b_equals_one_over_81 :
  ∀ b : ℝ, b > 0 → (Real.log 729 / Real.log b = -3/2) → b = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_b_729_equals_negative_three_halves_implies_b_equals_one_over_81_l858_85873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_white_balls_l858_85800

/-- The probability of drawing two white balls from a box containing 7 white balls
    and 8 black balls, without replacement, is 1/5. -/
theorem prob_two_white_balls (white_balls black_balls : ℕ) 
    (h_white : white_balls = 7)
    (h_black : black_balls = 8) : 
    (Nat.choose white_balls 2 : ℚ) / Nat.choose (white_balls + black_balls) 2 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_white_balls_l858_85800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_exponent_probability_l858_85869

noncomputable def binomial_expansion (x a : ℝ) := (x - a / Real.sqrt x) ^ 5

theorem integer_exponent_probability (x a : ℝ) :
  let expansion := binomial_expansion x a
  let total_terms := 6
  let integer_exponent_terms := 3
  (integer_exponent_terms : ℚ) / total_terms = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_exponent_probability_l858_85869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l858_85854

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.sqrt (1 + x^2))

-- State the theorem that f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l858_85854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_roots_l858_85811

/-- An odd function f defined on ℝ with specific behavior for positive x -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- Definition of f for positive x -/
axiom f_pos (x : ℝ) (h : x > 0) : f x = 2012^x + Real.log x / Real.log 2012

/-- Theorem: The equation f(x) = 0 has exactly 3 real roots -/
theorem f_three_roots : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_roots_l858_85811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_monotonicity_l858_85852

-- Define strict monotonicity on the whole real axis
def StrictlyMonotone (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem polynomial_monotonicity 
  (P : ℝ → ℝ) 
  (h1 : StrictlyMonotone (fun x ↦ P (P x)))
  (h2 : StrictlyMonotone (fun x ↦ P (P (P x))))
  : StrictlyMonotone P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_monotonicity_l858_85852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_assignment_exists_l858_85830

/-- A vertex or midpoint of a regular 2007-gon -/
inductive Point2007
| vertex (n : Fin 2007)
| midpoint (n : Fin 2007)

/-- The assignment function -/
def assignment : Point2007 → Fin 4014 := sorry

/-- The constant sum for each edge -/
def edgeSum : ℕ := 6022

/-- Theorem stating the existence of a valid assignment -/
theorem valid_assignment_exists :
  ∃ (f : Point2007 → Fin 4014),
    (∀ x : Fin 4014, ∃ y : Point2007, f y = x) ∧
    (∀ n : Fin 2007,
      (f (Point2007.vertex n)).val +
      (f (Point2007.vertex (n + 1))).val +
      (f (Point2007.midpoint n)).val = edgeSum) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_assignment_exists_l858_85830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_greater_than_max_perimeter_l858_85805

theorem smallest_whole_number_greater_than_max_perimeter : 
  ∃ (n : ℕ), n = 60 ∧
  (∀ c : ℝ, |7 - 23| ≤ c ∧ c < 7 + 23 → 
    (n : ℝ) > 7 + 23 + c ∧
    ∀ m : ℕ, m < n → ∃ c' : ℝ, |7 - 23| ≤ c' ∧ c' < 7 + 23 ∧ (m : ℝ) ≤ 7 + 23 + c') :=
by
  -- Define the two known side lengths
  let a : ℝ := 7
  let b : ℝ := 23

  -- Define the range for the third side c
  let c_lower_bound : ℝ := |a - b|
  let c_upper_bound : ℝ := a + b

  -- Define the perimeter function
  let perimeter (c : ℝ) : ℝ := a + b + c

  -- State and prove the theorem
  use 60
  apply And.intro
  · rfl -- Prove n = 60
  · intro c hc
    apply And.intro
    · -- Prove (60 : ℝ) > perimeter c
      sorry
    · -- Prove ∀ m < 60, ∃ c' satisfying the conditions
      sorry

-- The proof details are omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_greater_than_max_perimeter_l858_85805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_helens_cookies_l858_85874

/-- Helen's cookie baking problem -/
theorem helens_cookies (yesterday_chocolate : ℕ) (yesterday_raisin : ℕ) (today_raisin : ℕ) 
  (today_chocolate : ℕ)
  (h1 : yesterday_chocolate = 519)
  (h2 : yesterday_raisin = 300)
  (h3 : today_raisin = 280)
  (h4 : yesterday_raisin = today_raisin + 20)
  (h5 : yesterday_chocolate + yesterday_raisin = today_raisin + today_chocolate) :
  today_chocolate = 539 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_helens_cookies_l858_85874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l858_85838

theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter := 1.25 * P
  let second_quarter := 1.8 * P
  (second_quarter - first_quarter) / first_quarter * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l858_85838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_parabola_vertex_l858_85855

/-- The vertex of a parabola y = x^2 - 2x + 3 shifted 2 units right and 3 units up -/
theorem shifted_parabola_vertex :
  let f (x : ℝ) := x^2 - 2*x + 3
  let g (x : ℝ) := f (x - 2) + 3
  let vertex : ℝ × ℝ := (3, 5)
  (∀ x : ℝ, g x ≥ g vertex.1) ∧ g vertex.1 = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_parabola_vertex_l858_85855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l858_85889

/-- The area of a triangle with base 4 and height 8 is 16 -/
theorem triangle_area 
  (base : Real)
  (height : Real)
  (h1 : base = 4)
  (h2 : height = 8) :
  (base * height) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l858_85889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_zero_not_in_range_of_f_l858_85884

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)^3

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, f x = y :=
by
  -- The proof is omitted for now
  sorry

-- Additional theorem to state that 0 is not in the range of f
theorem zero_not_in_range_of_f :
  ¬ ∃ x : ℝ, f x = 0 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_zero_not_in_range_of_f_l858_85884
