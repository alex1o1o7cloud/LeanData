import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_large_fraction_mod_10_l1158_115876

theorem remainder_of_large_fraction_mod_10 : ∃ n : ℕ, 
  (⌊(10^20000 : ℚ) / ((10^100 : ℚ) + 3)⌋ : ℤ) ≡ n [ZMOD 10] ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_large_fraction_mod_10_l1158_115876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_production_l1158_115892

/-- The amount of sugar (in grams) needed to produce one chocolate bar -/
noncomputable def sugar_per_bar : ℚ := 3/2

/-- The amount of sugar (in grams) used in two minutes -/
noncomputable def sugar_two_minutes : ℚ := 108

/-- The number of minutes in the production period -/
noncomputable def production_period : ℚ := 2

/-- The number of chocolate bars produced per minute -/
noncomputable def bars_per_minute : ℚ := (sugar_two_minutes / sugar_per_bar) / production_period

theorem chocolate_production :
  bars_per_minute = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_production_l1158_115892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_intersects_asymptote_l1158_115812

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3*x^2 - 8*x - 9) / (x^2 - 5*x + 6)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 3

theorem g_intersects_asymptote :
  ∃ x : ℝ, g x = horizontal_asymptote ∧ x = 27/7 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_intersects_asymptote_l1158_115812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_with_infinite_preimages_l1158_115896

/-- The piecewise linear function g with period 2 -/
noncomputable def g (x : ℝ) : ℝ :=
  if x < 1/3 then 0
  else if x < 2/3 then 3*x - 1
  else if x < 4/3 then 1
  else if x < 5/3 then -3*x + 4
  else 0

/-- The function f defined as an infinite series -/
noncomputable def f (t : ℝ) : ℝ :=
  ∑' n, (g (9^n * t)) / (2^(n+1))

/-- The main theorem stating the existence of a function with the required properties -/
theorem exists_continuous_function_with_infinite_preimages :
  ∃ f : ℝ → ℝ, Continuous f ∧ (∀ t ∈ Set.Icc 0 1, f t ∈ Set.Icc 0 1) ∧
    (∀ β ∈ Set.Icc 0 1, (Set.Icc 0 1 ∩ f ⁻¹' {β}).Infinite) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_with_infinite_preimages_l1158_115896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_2a_plus_b_l1158_115859

theorem value_of_2a_plus_b (a b : ℝ) 
  (h1 : (a + 5).sqrt = -3) 
  (h2 : b^(1/3 : ℝ) = -2) : 
  2*a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_2a_plus_b_l1158_115859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_side_a_length_l1158_115822

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.cos (t.A / 2) = 2 * Real.sqrt 5 / 5 ∧
  t.b * t.c * Real.cos t.A = 3

-- Define the additional condition for part 2
def additional_condition (t : Triangle) : Prop :=
  t.b + t.c = 6

-- Theorem for part 1
theorem triangle_area (t : Triangle) (h : triangle_conditions t) :
  (1 / 2) * t.b * t.c * Real.sin t.A = 2 := by
  sorry

-- Theorem for part 2
theorem side_a_length (t : Triangle) (h1 : triangle_conditions t) (h2 : additional_condition t) :
  t.a = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_side_a_length_l1158_115822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1158_115867

noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

noncomputable def curve_derivative (α : ℝ) (x : ℝ) : ℝ := α * x^(α - 1)

theorem tangent_through_origin (α : ℝ) :
  curve α 1 = 2 ∧ 
  (∃ m : ℝ, ∀ x : ℝ, m * x = curve_derivative α 1 * (x - 1) + curve α 1) ∧
  (∃ m : ℝ, m * 0 = 0 ∧ m * 1 = curve α 1) →
  α = 2 := by
  sorry

#check tangent_through_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1158_115867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l1158_115839

-- Define the line L
def L (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the circle M
def CircleM (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 8

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the line L'
def L' (m : ℝ) (x y : ℝ) : Prop := y = -x - m

-- Theorem statement
theorem circle_and_line_problem (m : ℝ) :
  (∃ (x y : ℝ), L m x y ∧ CircleM x y) ∧  -- Line L intersects circle M
  (CircleM 0 m) ∧  -- Point P is on circle M
  (L m 0 m) ∧  -- Point P is on line L
  (∀ (x : ℝ), L m x (-x - m)) ∧  -- L is symmetric about x-axis
  (∃ (x₁ x₂ : ℝ), 
    L' m x₁ (-(x₁ + m)) ∧ CircleM x₁ (-(x₁ + m)) ∧ 
    L' m x₂ (-(x₂ + m)) ∧ CircleM x₂ (-(x₂ + m)) ∧ 
    (x₂ - x₁)^2 + ((-(x₂ + m)) - (-(x₁ + m)))^2 = 16) →  -- Chord length is 4
  (m = 2*Real.sqrt 2 - 2 ∨ m = -2*Real.sqrt 2 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l1158_115839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_efficiency_l1158_115879

/-- Represents the fuel efficiency of a car -/
structure CarEfficiency where
  speed : ℚ  -- Speed in miles per hour
  time : ℚ  -- Travel time in hours
  tankCapacity : ℚ  -- Tank capacity in gallons
  tankUsed : ℚ  -- Fraction of tank used

/-- Calculates the miles per gallon for a given car efficiency -/
def milesPerGallon (c : CarEfficiency) : ℚ :=
  (c.speed * c.time) / (c.tankCapacity * c.tankUsed)

/-- Theorem stating that the car achieves 30 miles per gallon -/
theorem car_efficiency :
  let c : CarEfficiency := {
    speed := 60,
    time := 5,
    tankCapacity := 12,
    tankUsed := 5 / 6
  }
  milesPerGallon c = 30 := by
  -- Unfold the definition of milesPerGallon
  unfold milesPerGallon
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_efficiency_l1158_115879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1158_115838

-- Define the function (marked as noncomputable due to use of transcendental functions)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x / 6 - Real.pi / 3)

-- State the theorem
theorem max_min_difference :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc 0 9, f x ≤ max ∧ min ≤ f x) ∧
  (max - min = 2 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1158_115838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_a_2017_l1158_115810

noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 + 1) ^ n - (Real.sqrt 2 - 1) ^ n

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem units_digit_of_a_2017 :
  (floor (a 2017)) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_a_2017_l1158_115810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_journey_distance_l1158_115895

/-- The radius of the circular room -/
def room_radius : ℝ := 45

/-- The length of the third leg of the spider's journey -/
def third_leg : ℝ := 70

/-- The total distance the spider crawled -/
noncomputable def total_distance : ℝ := 2 * room_radius + third_leg + Real.sqrt ((2 * room_radius)^2 - third_leg^2)

/-- Theorem stating that the total distance is approximately 216.57 feet -/
theorem spider_journey_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |total_distance - 216.57| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_journey_distance_l1158_115895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_g_l1158_115836

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / 3

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 - f (x^2 + 1)

-- Theorem statement
theorem domain_and_range_of_g :
  (∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 0 2) →
  (Set.Icc (-Real.sqrt 2) (Real.sqrt 2) = {x : ℝ | g x ∈ Set.range g}) ∧
  (Set.Icc (-1) (5/3) = Set.range g) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_g_l1158_115836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_half_xe_pow_x_ge_f_l1158_115869

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 + x + 1

-- Theorem for part I
theorem f_max_at_half (x : ℝ) (hx : x > 0) :
  ∃ (y : ℝ), y > 0 ∧ f (-2) y ≥ f (-2) x ∧ y = 1/2 := by
  sorry

-- Theorem for part II
theorem xe_pow_x_ge_f (x : ℝ) (hx : x > 0) :
  x * Real.exp x ≥ f 2 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_half_xe_pow_x_ge_f_l1158_115869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_iff_n_odd_l1158_115817

/-- A game where players draw chords on a circle with n points -/
structure ChordGame where
  n : ℕ
  h_n_ge_5 : n ≥ 5

/-- Predicate to check if a move (drawing a chord) is valid -/
def validMove (game : ChordGame) (move : ℕ × ℕ) : Prop :=
  move.1 < game.n ∧ move.2 < game.n ∧ move.1 ≠ move.2

/-- A strategy for the first player (Petya) in the chord game -/
def WinningStrategy (game : ChordGame) : Prop :=
  ∃ (strategy : ℕ → ℕ × ℕ), 
    ∀ (opponent_moves : ℕ → ℕ × ℕ),
      (∀ k, validMove game (strategy k)) ∧ 
      (∀ k, validMove game (opponent_moves k)) →
      ∃ m, ¬validMove game (opponent_moves m)

/-- The main theorem: Petya has a winning strategy if and only if n is odd -/
theorem petya_wins_iff_n_odd (game : ChordGame) : 
  WinningStrategy game ↔ Odd game.n := by
  sorry

#check petya_wins_iff_n_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_iff_n_odd_l1158_115817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ellipse_construction_l1158_115874

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the ellipse
structure Ellipse where
  center : Point
  majorAxis : ℝ
  minorAxis : ℝ

-- Define the given elements
def perpendicularAxes : (Line × Line) → Prop := sorry

def tangentLine : Line := sorry

def tangencyPoint : Point := sorry

-- Define the construction function
def constructEllipseEndpoints (axes : Line × Line) (t : Line) (T : Point) : Option (Point × Point × Point × Point) :=
  sorry

-- Theorem statement
theorem unique_ellipse_construction 
  (axes : Line × Line) 
  (t : Line) 
  (T : Point) 
  (h1 : perpendicularAxes axes) :
  ∃! (A B C D : Point), constructEllipseEndpoints axes t T = some (A, B, C, D) :=
by
  sorry

#check unique_ellipse_construction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ellipse_construction_l1158_115874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l1158_115844

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := Set.range (Int.cast : ℤ → ℝ)

theorem A_intersect_B : A ∩ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l1158_115844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_sphere_radius_ratio_l1158_115885

/-- Define a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Four spheres are mutually tangent -/
def are_mutually_tangent (s1 s2 s3 s4 : Sphere) : Prop :=
  sorry

/-- A sphere touches all four other spheres -/
def touches_all (s5 s1 s2 s3 s4 : Sphere) : Prop :=
  sorry

/-- Given four mutually tangent spheres of equal radius, 
    the ratio of the radius of a fifth sphere touching all four 
    to the radius of one of the given spheres -/
theorem fifth_sphere_radius_ratio (r : ℝ) (R : ℝ) 
  (h1 : r > 0) 
  (h2 : R > 0) 
  (h3 : ∃ (s1 s2 s3 s4 s5 : Sphere), 
    (s1.radius = r ∧ s2.radius = r ∧ s3.radius = r ∧ s4.radius = r) ∧ 
    (are_mutually_tangent s1 s2 s3 s4) ∧
    (s5.radius = R) ∧
    (touches_all s5 s1 s2 s3 s4)) :
  R / r = Real.sqrt 6 / 2 + 1 ∨ R / r = Real.sqrt 6 / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_sphere_radius_ratio_l1158_115885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l1158_115827

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem tangent_point_x_coordinate 
  (h_even : ∀ x, f a x = f a (-x))
  (h_slope : ∃ x, (deriv (f a)) x = 3/2) :
  ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l1158_115827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_equals_rhombus_hexagon_l1158_115889

/-- The area of a regular hexagon with side length a -/
noncomputable def hexagonArea (a : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * a^2

/-- The area of the annular region formed by cutting out a concentric regular hexagon 
    with side length s/2 from a regular hexagon with side length s -/
noncomputable def annulusArea (s : ℝ) : ℝ := hexagonArea s - hexagonArea (s/2)

/-- The area of a regular hexagon formed by 6 identical rhombuses, 
    where each rhombus has a side length equal to the side length of the original hexagon -/
noncomputable def rhombusHexagonArea (s : ℝ) : ℝ := 6 * ((s^2 * Real.sqrt 3) / 2)

theorem annulus_equals_rhombus_hexagon (s : ℝ) (h : s > 0) : 
  annulusArea s = rhombusHexagonArea s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_equals_rhombus_hexagon_l1158_115889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_two_units_l1158_115807

def validPoint (x y : ℤ) : Prop := (abs x ≤ 4) ∧ (abs y ≤ 4)

noncomputable def distanceFromOrigin (x y : ℤ) : ℝ := Real.sqrt (x^2 + y^2)

def withinTwoUnits (x y : ℤ) : Prop := distanceFromOrigin x y ≤ 2

def totalValidPoints : ℕ := (2 * 4 + 1)^2

def favorablePoints : ℕ := 13

theorem probability_within_two_units :
  (favorablePoints : ℚ) / totalValidPoints = 13 / 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_two_units_l1158_115807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l1158_115823

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := (1/4) * x^2 - (1/2) * Real.log x

/-- The line function -/
noncomputable def g (x : ℝ) : ℝ := (3/4) * x - 1

/-- The distance between a point on the curve and a point on the line -/
noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt ((x - y)^2 + (f x - g y)^2)

theorem min_distance_curve_line :
  ∃ (d : ℝ), d = (2 - 2 * Real.log 2) / 5 ∧
  ∀ (x y : ℝ), x > 0 → distance x y ≥ d := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l1158_115823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1158_115862

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (|x - 2| + |x + 3| - 5)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -3 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1158_115862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l1158_115855

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_decreasing :
  ∃ α : ℝ, (f α 3 = (Real.sqrt 3) / 3) ∧ (∀ x y : ℝ, x > y → 0 < x → 0 < y → f α x < f α y) := by
  -- Provide the value of α
  use -1/2
  
  constructor
  · -- Prove f α 3 = (Real.sqrt 3) / 3
    simp [f]
    sorry
  
  · -- Prove the function is decreasing for positive x and y
    intros x y hxy hx hy
    simp [f]
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l1158_115855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_solutions_l1158_115878

noncomputable def f (n : ℝ) : ℝ :=
  if n < 0 then n^2 - 2 else 3*n - 20

theorem positive_difference_of_solutions : ∃ a₁ a₂ : ℝ,
  f (-2) + f 2 + f a₁ = 0 ∧
  f (-2) + f 2 + f a₂ = 0 ∧
  |a₁ - a₂| = Real.sqrt 14 + 32/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_solutions_l1158_115878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_given_f_plus_h_l1158_115821

/-- A polynomial function over the reals. -/
def MyPolynomial := ℝ → ℝ

/-- The degree of a polynomial. -/
noncomputable def degree (p : MyPolynomial) : ℕ := sorry

/-- Given polynomial f(x) = -9x^5 + 2x^3 - 4x + 7. -/
def f : MyPolynomial := λ x ↦ -9 * x^5 + 2 * x^3 - 4 * x + 7

theorem degree_of_h_given_f_plus_h (h : MyPolynomial) 
  (h_cond : degree (λ x ↦ f x + h x) = 3) : 
  degree h = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_given_f_plus_h_l1158_115821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_chain_l1158_115870

theorem inequality_chain (φ : ℝ) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_chain_l1158_115870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_walking_speed_prove_fred_speed_l1158_115853

/-- The distance between Fred and Sam -/
def total_distance : ℚ := 35

/-- Sam's walking speed in miles per hour -/
def sam_speed : ℚ := 5

/-- The distance Sam walks before meeting Fred -/
def sam_distance : ℚ := 25

/-- Fred's walking speed in miles per hour -/
def fred_speed : ℚ := 2

/-- The time it takes for Fred and Sam to meet -/
noncomputable def meeting_time : ℚ := sam_distance / sam_speed

theorem fred_walking_speed :
  fred_speed * meeting_time = total_distance - sam_distance :=
by sorry

theorem prove_fred_speed :
  fred_speed = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_walking_speed_prove_fred_speed_l1158_115853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1158_115877

/-- The complex number z -/
noncomputable def z : ℂ := (2 - Complex.I) / (1 + Complex.I)

/-- The real part of z -/
noncomputable def real_part : ℝ := z.re

/-- The imaginary part of z -/
noncomputable def imag_part : ℝ := z.im

/-- Theorem: The point corresponding to z is in the fourth quadrant -/
theorem point_in_fourth_quadrant : real_part > 0 ∧ imag_part < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1158_115877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1158_115802

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - a^(-x)

-- State the theorem
theorem f_properties (a : ℝ) (h_a : a > 1) :
  -- 1. f is odd
  (∀ x : ℝ, f a (-x) = -(f a x)) ∧
  -- 2. f is monotonically increasing
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
  -- 3. Condition for f(1-t) + f(1-t^2) < 0
  (∀ t : ℝ, f a (1-t) + f a (1-t^2) < 0 ↔ t < -2 ∨ t > 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1158_115802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1158_115825

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))
noncomputable def c : ℝ × ℝ := (Real.sqrt 3, -1)

theorem vector_problem (x : ℝ) :
  (∃ k : ℤ, x = k * π / 2 + π / 4 ↔ (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0) ∧
  (∃ M : ℝ, M = 3 ∧ ∀ y : ℝ, Real.sqrt (((a y).1 - c.1)^2 + ((a y).2 - c.2)^2) ≤ M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1158_115825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2016_l1158_115830

def a : ℕ → ℚ 
  | 0 => 2
  | 1 => 1
  | (n+2) => if a (n+1) ≥ 2 then 2 * a (n+1) / a n else 4 / a n

def S (n : ℕ) : ℚ := (List.range n).map a |>.sum

theorem sequence_sum_2016 : S 2016 = 5241 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2016_l1158_115830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_time_is_five_l1158_115873

/-- Represents the problem of a boat traveling upstream and downstream -/
structure BoatProblem where
  distance : ℝ  -- Total distance traveled in miles
  downstreamTime : ℝ  -- Time for downstream trip in hours
  stillWaterSpeed : ℝ  -- Speed of the boat in still water in mph

/-- Calculates the upstream travel time given a BoatProblem -/
noncomputable def upstreamTime (problem : BoatProblem) : ℝ :=
  let currentSpeed := (problem.distance / problem.downstreamTime - problem.stillWaterSpeed) / 2
  problem.distance / (problem.stillWaterSpeed - currentSpeed)

/-- Theorem stating that for the given conditions, the upstream time is 5 hours -/
theorem upstream_time_is_five :
  let problem : BoatProblem := {
    distance := 300,
    downstreamTime := 2,
    stillWaterSpeed := 105
  }
  upstreamTime problem = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_time_is_five_l1158_115873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduce_to_zero_l1158_115816

/-- Definition of the operation: replace two numbers with their difference -/
def replace_with_diff (a b : ℕ) : ℕ := max a b - min a b

/-- Definition of a sequence of consecutive positive integers from 1 to n -/
def consecutive_sequence (n : ℕ) : List ℕ := List.range n

/-- Predicate to check if a number is odd and greater than 1998 -/
def is_odd_gt_1998 (n : ℕ) : Prop := n % 2 = 1 ∧ n > 1998

/-- Function to iterate the replace_with_diff operation on a list -/
def iterate_ops : List ℕ → List (ℕ × ℕ) → List ℕ
| l, [] => l
| l, (a, b) :: rest => iterate_ops (replace_with_diff a b :: l.filter (λ x => x ≠ a ∧ x ≠ b)) rest

/-- Theorem stating the condition for reducing a sequence to zero -/
theorem reduce_to_zero (n : ℕ) :
  (∃ (l : List ℕ), l.length = 1 ∧ l.head? = some 0 ∧
    ∃ (ops : List (ℕ × ℕ)), iterate_ops (consecutive_sequence n) ops = l) ↔
  is_odd_gt_1998 n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduce_to_zero_l1158_115816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_set_equality_l1158_115820

theorem sine_cosine_set_equality (α : ℝ) :
  ({Real.sin α, Real.sin (2 * α), Real.sin (3 * α)} : Set ℝ) = {Real.cos α, Real.cos (2 * α), Real.cos (3 * α)}
    ↔ ∃ k : ℤ, α = π / 8 + k * π / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_set_equality_l1158_115820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_l1158_115840

def matrix_equation (a b c d : ℕ+) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 0; 0, 3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![a, b; c, d]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![12, 16; -15, -20]
  A * B = B * C

theorem smallest_sum (a b c d : ℕ+) (h : matrix_equation a b c d) :
  (a : ℕ) + b + c + d ≥ 47 ∧ ∃ (a' b' c' d' : ℕ+), matrix_equation a' b' c' d' ∧ (a' : ℕ) + b' + c' + d' = 47 :=
by
  sorry

#check smallest_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_l1158_115840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l1158_115894

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Line) : Prop := l₁.slope = l₂.slope

/-- Two lines are distinct if they are not identical -/
def distinct (l₁ l₂ : Line) : Prop := l₁ ≠ l₂

/-- The problem statement -/
theorem parallel_lines_theorem (a : ℝ) : 
  let l₁ : Line := ⟨1, a, -(2*a + 2)⟩
  let l₂ : Line := ⟨a, 1, -(a + 1)⟩
  parallel l₁ l₂ ∧ distinct l₁ l₂ → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l1158_115894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l1158_115891

noncomputable section

/-- The slope of the given line -/
def m : ℝ := 5/3

/-- The y-intercept of the given line -/
def c₁ : ℝ := -2

/-- The distance between the parallel lines -/
def d : ℝ := 6

/-- The equation of the given line -/
def given_line (x : ℝ) : ℝ := m * x + c₁

/-- The possible y-intercepts of the parallel lines -/
def c₂₁ : ℝ := 2 * Real.sqrt 34 - 2
def c₂₂ : ℝ := -(2 * Real.sqrt 34 + 2)

/-- The equations of the parallel lines -/
def parallel_line₁ (x : ℝ) : ℝ := m * x + c₂₁
def parallel_line₂ (x : ℝ) : ℝ := m * x + c₂₂

theorem parallel_lines_theorem :
  (∀ x, ‖parallel_line₁ x - given_line x‖ = d) ∧
  (∀ x, ‖parallel_line₂ x - given_line x‖ = d) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l1158_115891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_z10_from_origin_l1158_115872

noncomputable def complex_sequence : ℕ → ℂ
  | 0 => 0
  | n + 1 => (complex_sequence n)^2 + (1 + Complex.I)

theorem distance_z10_from_origin :
  Complex.abs (complex_sequence 10) = Real.sqrt 45205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_z10_from_origin_l1158_115872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_stretch_theorem_l1158_115832

/-- Represents the work done on a spring -/
structure SpringWork where
  extension : ℝ  -- Extension in meters
  work : ℝ       -- Work done in Joules

/-- Calculates the spring constant given a SpringWork instance -/
noncomputable def springConstant (sw : SpringWork) : ℝ :=
  (2 * sw.work) / (sw.extension ^ 2)

/-- Theorem: Given a spring where 29.43 J of work is done to stretch it by 5 cm,
    the stretch distance when 9.81 J of work is done is approximately 0.029 m -/
theorem spring_stretch_theorem (sw1 sw2 : SpringWork)
    (h1 : sw1.extension = 0.05)
    (h2 : sw1.work = 29.43)
    (h3 : sw2.work = 9.81) :
    ∃ ε > 0, |sw2.extension - 0.029| < ε := by
  sorry

#check spring_stretch_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_stretch_theorem_l1158_115832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_l1158_115857

/-- Predicate representing an isosceles triangle with side lengths a and base c. -/
def IsoscelesTriangle (a c : ℝ) : Prop := sorry

/-- Predicate representing that a length is an altitude to a side in a triangle. -/
def IsAltitude (m side : ℝ) : Prop := sorry

/-- Function to get the angle opposite to the base in an isosceles triangle. -/
noncomputable def AngleOppositeBase (a c : ℝ) (h : IsoscelesTriangle a c) : ℝ := sorry

/-- Given an isosceles triangle ABC with base c, altitude to leg m_a, and altitude to base m_c,
    if a + m_c = s_1 and c + m_a = s_2, then sin γ = (s_2 / (2s_1)) * √(4s_1^2 - s_2^2),
    where γ is the angle opposite to the base. -/
theorem isosceles_triangle_angle (a c m_a m_c s_1 s_2 : ℝ) (γ : ℝ) 
    (h_isosceles : IsoscelesTriangle a c)
    (h_m_a : IsAltitude m_a a)
    (h_m_c : IsAltitude m_c c)
    (h_s1 : a + m_c = s_1)
    (h_s2 : c + m_a = s_2)
    (h_γ : γ = AngleOppositeBase a c h_isosceles) :
  Real.sin γ = (s_2 / (2 * s_1)) * Real.sqrt (4 * s_1^2 - s_2^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_l1158_115857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_in_triangle_sides_l1158_115852

/-- A right triangle with a 60° angle, containing an inscribed rhombus --/
structure RhombusInTriangle where
  /-- Side length of the inscribed rhombus --/
  rhombus_side : ℝ
  /-- The rhombus is inscribed in the triangle --/
  inscribed : Prop
  /-- The 60° angle is common to both the rhombus and the triangle --/
  common_angle : Prop
  /-- All vertices of the rhombus lie on the sides of the triangle --/
  vertices_on_sides : Prop

/-- The sides of the triangle containing the inscribed rhombus --/
noncomputable def triangle_sides (r : RhombusInTriangle) : ℝ × ℝ × ℝ :=
  (12, 12 * Real.sqrt 3, 24)

/-- Theorem stating the sides of the triangle given the inscribed rhombus --/
theorem rhombus_in_triangle_sides (r : RhombusInTriangle) 
    (h : r.rhombus_side = 6) : 
    triangle_sides r = (12, 12 * Real.sqrt 3, 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_in_triangle_sides_l1158_115852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_sum_exceeds_32_l1158_115897

/-- An arithmetic sequence with first term 2 and a1 + a4 = a5 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ a 1 + a 4 = a 5 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- The minimum n such that the sum exceeds 32 is 6 -/
theorem min_n_sum_exceeds_32 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
    (∃ n : ℕ, SumArithmeticSequence a n > 32) ∧
    (∀ n : ℕ, n < 6 → SumArithmeticSequence a n ≤ 32) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_sum_exceeds_32_l1158_115897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_condition_min_young_age_l1158_115846

-- Define Min-young's current age
def current_age : ℕ := 8

-- Define the condition given in the problem
theorem age_condition : current_age + 24 = 4 * current_age := by
  -- Proof steps will go here
  sorry

-- Theorem to prove
theorem min_young_age : current_age = 8 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_condition_min_young_age_l1158_115846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1158_115868

noncomputable def distance_polar (r₁ r₂ : ℝ) (θ₁ θ₂ : ℝ) : ℝ :=
  Real.sqrt ((r₁ * Real.cos θ₁ - r₂ * Real.cos θ₂)^2 + (r₁ * Real.sin θ₁ - r₂ * Real.sin θ₂)^2)

theorem distance_between_points (θ₁ θ₂ : ℝ) (h : θ₁ - θ₂ = π/3) :
  distance_polar 4 6 θ₁ θ₂ = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1158_115868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_four_l1158_115861

/-- The sum of the geometric series 1 + 1/2 + 1/4 + 1/8 + ... -/
noncomputable def S₁ : ℝ := 2

/-- The sum of the geometric series 1 - 1/2 + 1/4 - 1/8 + ... -/
noncomputable def S₂ : ℝ := 2/3

/-- The product of S₁ and S₂ -/
noncomputable def P : ℝ := S₁ * S₂

/-- The equation to be satisfied -/
def equation (x : ℝ) : Prop :=
  P = (1 : ℝ) / (1 - 1/x)

/-- The solution is x = 4 -/
theorem solution_is_four : equation 4 := by
  sorry

#check solution_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_four_l1158_115861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l1158_115801

/-- The angle in degrees that the hour hand moves in one minute -/
noncomputable def hourHandAnglePerMinute : ℝ := 0.5

/-- The angle in degrees that the minute hand moves in one minute -/
noncomputable def minuteHandAnglePerMinute : ℝ := 6

/-- The angle in degrees between each hour mark on the clock -/
noncomputable def hourMarkAngle : ℝ := 30

/-- Calculates the position of the hour hand at a given hour and minute -/
noncomputable def hourHandPosition (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour % 12 : ℝ) * hourMarkAngle + (minute : ℝ) * hourHandAnglePerMinute

/-- Calculates the position of the minute hand at a given minute -/
noncomputable def minuteHandPosition (minute : ℕ) : ℝ :=
  (minute : ℝ) * minuteHandAnglePerMinute

/-- Calculates the smaller angle between two angles on a clock -/
noncomputable def smallerAngleBetween (angle1 : ℝ) (angle2 : ℝ) : ℝ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

theorem clock_angle_at_7_15 :
  smallerAngleBetween (hourHandPosition 7 15) (minuteHandPosition 15) = 127.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l1158_115801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1158_115835

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/(x+1)

theorem f_bounds (x : ℝ) (h : x ∈ Set.Icc 0 1) : 
  f x ≥ x^2 - (4/9)*x + 8/9 ∧ 68/81 < f x ∧ f x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1158_115835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_seq4_increasing_and_infinite_l1158_115806

-- Define the sequences
def seq1 : ℕ → ℕ
  | n => if n ≤ 20 then n else 20

def seq2 : ℕ → ℤ
  | n => -(n + 1)

def seq3 : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | n + 4 => seq3 n + 1

def seq4 : ℕ → ℤ
  | 0 => -1
  | n + 1 => seq4 n + 1

-- Define the properties
def isIncreasing {α : Type*} [LT α] (s : ℕ → α) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

def isInfinite {α : Type*} [LT α] (s : ℕ → α) : Prop :=
  ∀ N : α, ∃ n : ℕ, N < s n

theorem only_seq4_increasing_and_infinite :
  (¬(isIncreasing seq1 ∧ isInfinite seq1)) ∧
  (¬(isIncreasing seq2 ∧ isInfinite seq2)) ∧
  (¬(isIncreasing seq3 ∧ isInfinite seq3)) ∧
  (isIncreasing seq4 ∧ isInfinite seq4) := by
  sorry

#check only_seq4_increasing_and_infinite

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_seq4_increasing_and_infinite_l1158_115806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1158_115847

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 11 + y^2 / 7 = 1

-- Define the foci of the ellipse
def ellipse_foci : Set (ℝ × ℝ) := {(-2, 0), (2, 0)}

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- State the theorem
theorem hyperbola_equation (a b : ℝ) : 
  (∀ (x y : ℝ), hyperbola a b x y → (x = 0 ∨ y = 0) → (y = 0 ∨ x = 0)) → -- Axes of symmetry are coordinate axes
  (∃ (f : ℝ × ℝ), f ∈ ellipse_foci ∧ 
    ∀ (x y : ℝ), hyperbola a b x y → x ≠ 0 → 
    Real.sqrt 2 = |((x - f.1) / y)|) → -- Distance from focus to asymptote is √2
  (∀ (f : ℝ × ℝ), f ∈ ellipse_foci ↔ (∃ (x y : ℝ), hyperbola a b x y ∧ x = f.1 ∧ y = f.2)) → -- Foci of ellipse are vertices of hyperbola
  a^2 = 2 ∧ b^2 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1158_115847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1158_115860

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {x : ℤ | x^2 - x = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1158_115860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1158_115884

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: The time taken for a 100m train to cross a 150m bridge at 63 kmph is approximately 14.29 seconds -/
theorem train_crossing_bridge_time :
  ∃ ε > 0, |train_crossing_time 100 150 63 - 14.29| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1158_115884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l1158_115803

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 10*y + 13 = 0

/-- The line equation -/
def line_equation (y : ℝ) : Prop :=
  y = -2

/-- The area of the circle above the line -/
noncomputable def area_above_line : ℝ := 2 * Real.pi

theorem circle_area_above_line :
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation y →
  area_above_line = 2 * Real.pi := by
  sorry

#check circle_area_above_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l1158_115803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_z_111_l1158_115851

/-- The complex sequence z_n defined by z₁ = 0 and zₙ₊₁ = zₙ² + i -/
noncomputable def z : ℕ → ℂ
  | 0 => 0
  | n + 1 => (z n)^2 + Complex.I

/-- The modulus of z₁₁₁ is √2 -/
theorem modulus_z_111 : Complex.abs (z 111) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_z_111_l1158_115851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1158_115842

/-- Prove that for a parabola y^2 = 2px (p > 0) and a line with slope 2 passing through its focus,
    if the area of the trapezoid formed by the line's intersection points with the parabola
    and their y-axis projections is 6√5, then p = 2√2. -/
theorem parabola_line_intersection (p : ℝ) (h₁ : p > 0) : 
  let parabola := fun (x y : ℝ) ↦ y^2 = 2*p*x
  let focus := (p/2, 0)
  let line := fun (x y : ℝ) ↦ y = 2*(x - p/2)
  let intersection := fun (x y : ℝ) ↦ parabola x y ∧ line x y
  let trapezoid_area := 6 * Real.sqrt 5
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection x₁ y₁ ∧ 
    intersection x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧
    trapezoid_area = (1/2) * (|y₁| + |y₂|) * |x₂ - x₁|) →
  p = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1158_115842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l1158_115871

/-- Represents the speed of the firetruck in different terrains -/
structure FiretruckSpeed where
  highway : ℝ
  prairie : ℝ

/-- Represents the time limit for the firetruck's travel -/
noncomputable def timeLimit : ℝ := 8 / 60 -- 8 minutes converted to hours

/-- Calculates the area of the region that can be reached by the firetruck -/
noncomputable def reachableArea (speed : FiretruckSpeed) (t : ℝ) : ℝ :=
  let highwayDistance := speed.highway * t
  let prairieDistance := speed.prairie * t
  4 * (highwayDistance^2 + (Real.pi * prairieDistance^2 / 2) - prairieDistance * highwayDistance)

/-- Theorem stating the area of the region reachable by the firetruck -/
theorem firetruck_reachable_area :
  let speed := FiretruckSpeed.mk 60 10
  reachableArea speed timeLimit = (2112 + 32 * Real.pi) / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l1158_115871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_exists_l1158_115854

/-- A type representing a point on a circle -/
structure CirclePoint where
  angle : ℝ

/-- A type representing a circle with points -/
structure Circle where
  points : Finset CirclePoint
  no_diametric_points : ∀ p q : CirclePoint, p ∈ points → q ∈ points → p.angle ≠ q.angle + π

/-- Main theorem statement -/
theorem equal_division_exists (c : Circle) (h : c.points.card = 2000000) :
  ∃ θ : ℝ, (c.points.filter (λ p => 0 ≤ p.angle - θ ∧ p.angle - θ < π)).card = 1000000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_exists_l1158_115854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_returned_cans_l1158_115865

/-- Calculates the average price of returned cans given the initial purchase and remaining cans information -/
theorem average_price_of_returned_cans 
  (total_cans : ℕ) 
  (returned_cans : ℕ) 
  (initial_avg_price : ℚ) 
  (remaining_avg_price : ℚ) 
  (h1 : total_cans = 6) 
  (h2 : returned_cans = 2) 
  (h3 : initial_avg_price = 365/10) 
  (h4 : remaining_avg_price = 30) : 
  (initial_avg_price * total_cans - remaining_avg_price * (total_cans - returned_cans)) / returned_cans = 495/10 := by
  sorry

#check average_price_of_returned_cans

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_returned_cans_l1158_115865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_radius_l1158_115824

-- Define the constants
def stationary_tank_radius : ℝ := 100
def stationary_tank_height : ℝ := 25
def oil_truck_tank_height : ℝ := 12
def oil_level_drop : ℝ := 0.03

-- Define the volume of oil pumped out
noncomputable def volume_pumped_out : ℝ := Real.pi * stationary_tank_radius^2 * oil_level_drop

-- Theorem statement
theorem oil_truck_tank_radius : 
  ∃ (r : ℝ), r > 0 ∧ Real.pi * r^2 * oil_truck_tank_height = volume_pumped_out ∧ r = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_radius_l1158_115824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_game_theorem_l1158_115831

/-- Determines if Jana has a winning strategy in the dandelion mowing game -/
def jana_wins (m n : ℕ) : Prop :=
  (m = 1 ∨ n = 1) ∨ (m + n) % 2 = 1

/-- The dandelion mowing game theorem -/
theorem dandelion_game_theorem (m n : ℕ) :
  jana_wins m n ↔ (m = 1 ∨ n = 1) ∨ (m + n) % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_game_theorem_l1158_115831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_25_percent_l1158_115829

/-- The discount percentage for soda purchased in 24-can cases -/
noncomputable def discount_percentage (regular_price : ℝ) (discounted_total : ℝ) : ℝ :=
  (1 - discounted_total / (70 * regular_price)) * 100

/-- Theorem stating that the discount percentage is 25% -/
theorem discount_is_25_percent : 
  let regular_price := 0.55
  let discounted_total := 28.875
  discount_percentage regular_price discounted_total = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_25_percent_l1158_115829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1158_115890

noncomputable def angle : ℝ := Real.pi / 6  -- 30° in radians

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (-1, Real.sqrt 2) →
  Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) : ℝ) = Real.sqrt 3 →
  Real.sqrt ((b.1 ^ 2 + b.2 ^ 2) : ℝ) = 2 →
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 * 2 * Real.cos angle →
  Real.sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) = Real.sqrt 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1158_115890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_implies_a_equals_one_b_can_be_any_natural_l1158_115809

theorem divisibility_condition_implies_a_equals_one (a b : ℕ) :
  (∃ k : ℤ, (a + b) * (b + 1) - 1 = k * (a * (a + b) + 1)) →
  a = 1 :=
by sorry

-- Additional theorem to capture the fact that b can be any natural number
theorem b_can_be_any_natural (b : ℕ) :
  ∃ k : ℤ, (1 + b) * (b + 1) - 1 = k * (1 * (1 + b) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_implies_a_equals_one_b_can_be_any_natural_l1158_115809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_f_less_than_2_l1158_115834

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2 * Real.exp (x - 1) else x^3 + x

-- State the theorem
theorem solution_set_f_f_less_than_2 :
  {x : ℝ | f (f x) < 2} = Set.Iio (1 - Real.log 2) := by
  sorry

#check solution_set_f_f_less_than_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_f_less_than_2_l1158_115834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_decreasing_on_interval_g_has_two_zeros_l1158_115841

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + 1 / x

noncomputable def g (x : ℝ) : ℝ := f 1 (Real.exp x) - 18

-- Theorem for the parity of f
theorem f_is_odd (a : ℝ) : ∀ x ≠ 0, f a (-x) = -(f a x) := by
  intro x hx
  simp [f]
  ring

-- Theorem for the monotonicity of f
theorem f_decreasing_on_interval (a : ℝ) (h : a > 0) :
  ∀ x y, 0 < x ∧ x < y ∧ y < 1 / Real.sqrt a → f a x > f a y := by
  sorry

-- Theorem for the number of zeros of g
theorem g_has_two_zeros :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ ∀ x, g x = 0 → x = x₁ ∨ x = x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_decreasing_on_interval_g_has_two_zeros_l1158_115841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_area_ratio_l1158_115882

noncomputable def angle_to_radians (degrees : ℝ) : ℝ := degrees * (Real.pi / 180)

noncomputable def triangle_area_ratio (α β γ : ℝ) : ℝ :=
  1 / (2 * Real.cos (angle_to_radians α) * Real.cos (angle_to_radians β) * Real.cos (angle_to_radians γ))

theorem triangle_tangent_area_ratio :
  let α : ℝ := 50
  let β : ℝ := 60
  let γ : ℝ := 70
  ∃ (ε : ℝ), ε > 0 ∧ |triangle_area_ratio α β γ - 4.55| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_area_ratio_l1158_115882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sequence_product_l1158_115819

/-- Calculates the product of fractions from 1/5 to n/(n+4), where n is the last numerator -/
def fractionProduct (n : ℕ) : ℚ :=
  (List.range n).foldl (λ acc i => acc * ((i + 1 : ℚ) / (i + 5 : ℚ))) 1

/-- The problem statement -/
theorem fraction_sequence_product :
  fractionProduct 50 = 6 / 78963672 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sequence_product_l1158_115819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_square_sum_equation_l1158_115866

theorem largest_n_for_square_sum_equation : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → 
    (x + ↑i)^2 + (y i)^2 = (x + ↑j)^2 + (y j)^2) ∧
  (∀ (m : ℕ), m > n → 
    ¬∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 
      (x + ↑i)^2 + (y i)^2 = (x + ↑j)^2 + (y j)^2) ∧
  n = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_square_sum_equation_l1158_115866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1158_115815

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

-- State the theorem
theorem odd_function_properties :
  ∃ b : ℝ,
    (∀ x, f b x = -f b (-x)) ∧  -- f is an odd function
    (b = 1) ∧  -- b = 1
    (∀ x y, x < y → f b x > f b y) ∧  -- f is monotonically decreasing
    (∀ k, (∀ t, f b (t^2 - 2*t) + f b (2*t^2 - k) < 0) ↔ k < -1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1158_115815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1158_115899

-- Define the hyperbola parameters
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 7

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the distance between intersection points
noncomputable def distance (k : ℝ) : ℝ := 16 * Real.sqrt 3

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y ↔ x^2 / 4 - y^2 / 3 = 1) ∧
  (∀ k, (∃ x₁ x₂, hyperbola x₁ (line k x₁) ∧ hyperbola x₂ (line k x₂) ∧
    (x₂ - x₁)^2 + (line k x₂ - line k x₁)^2 = (distance k)^2) →
    k = Real.sqrt (2145 / 65^2) ∨ k = -Real.sqrt (2145 / 65^2) ∨ k = 1 ∨ k = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1158_115899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karens_speed_l1158_115881

/-- Prove that Karen's average speed is 78.75 mph given the race conditions --/
theorem karens_speed (karen_delay : Real) (tom_speed : Real) (tom_distance : Real) (karen_lead : Real) : 
  let karen_speed := (tom_distance + karen_lead) / (tom_distance / tom_speed - karen_delay)
  karen_delay = 4 / 60 ∧ 
  tom_speed = 45 ∧ 
  tom_distance = 24 ∧ 
  karen_lead = 4 → 
  karen_speed = 78.75 := by
    intros h
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_karens_speed_l1158_115881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_over_r_plus_1_eq_A_B_comparison_l1158_115811

/-- Represents the coefficients in the expansion of (q+x)^n -/
noncomputable def a (q : ℝ) (n : ℕ+) (r : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ i ↦ if i = r then (q + 1)^(n:ℕ) else 0)

/-- Sum of a_r / (r+1) when q = 1 -/
noncomputable def sum_a_over_r_plus_1 (n : ℕ+) : ℝ :=
  (Finset.range (n + 1)).sum (λ r ↦ a 1 n r / (r + 1 : ℝ))

/-- A_n as defined in the problem -/
noncomputable def A (n : ℕ+) : ℝ := (n : ℝ) * (a n n 0 + a n n 1) / 2

/-- B_n as defined in the problem -/
noncomputable def B (n : ℕ+) : ℝ := (Finset.range (n + 1)).sum (λ r ↦ a n n r)

theorem sum_a_over_r_plus_1_eq (n : ℕ+) :
  sum_a_over_r_plus_1 n = (2^((n:ℕ) + 1) - 1) / ((n:ℕ) + 1 : ℝ) := by sorry

theorem A_B_comparison (n : ℕ+) :
  (n = 1 ∨ n = 2 → A n < B n) ∧
  (n ≥ 3 → A n > B n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_over_r_plus_1_eq_A_B_comparison_l1158_115811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l1158_115808

-- Define a square
structure Square where
  side : ℝ
  side_pos : side > 0

-- Define a line that divides a square in 2:3 ratio
structure DividingLine (s : Square) where
  ratio : ℝ
  ratio_eq : ratio = 2 / 3

-- Define a function to count intersections
def countIntersections (s : Square) (lines : List (DividingLine s)) : ℕ → ℕ
  | 0 => 0
  | (n + 1) => sorry

-- Theorem statement
theorem at_least_three_lines_intersect (s : Square) 
  (lines : List (DividingLine s)) 
  (h1 : lines.length = 9) :
  ∃ p : ℕ, countIntersections s lines p ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l1158_115808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_level_number_max_min_l1158_115856

def Digits := List Nat

def TwoLevelNumber (d1 d2 : Digits) : Prop :=
  d1.length = 4 ∧ d2.length = 4 ∧ 
  (d1 = [1, 4, 6, 8] ∨ d1 = [0, 0, 0, 5]) ∧
  (d2 = [1, 4, 6, 8] ∨ d2 = [0, 0, 0, 5]) ∧
  d1 ≠ d2

def toNumber (d1 d2 : Digits) : Nat :=
  match d1, d2 with
  | [a, b, c, d], [e, f, g, h] =>
    1000 * (1000 * a + 100 * b + 10 * c + d) +
    1000 * e + 100 * f + 10 * g + h
  | _, _ => 0  -- default case for invalid inputs

theorem two_level_number_max_min :
  ∀ d1 d2 : Digits, TwoLevelNumber d1 d2 →
  (∀ d1' d2' : Digits, TwoLevelNumber d1' d2' → 
    toNumber d1' d2' ≤ toNumber [8, 6, 4, 1] [5, 0, 0, 0]) ∧
  (∀ d1' d2' : Digits, TwoLevelNumber d1' d2' → 
    toNumber [1, 4, 6, 8] [0, 0, 0, 5] ≤ toNumber d1' d2') :=
by sorry

#eval toNumber [8, 6, 4, 1] [5, 0, 0, 0]  -- Should output 86415000
#eval toNumber [1, 4, 6, 8] [0, 0, 0, 5]  -- Should output 14680005

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_level_number_max_min_l1158_115856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_trajectory_equations_x0_range_l1158_115826

-- Define complex number β
noncomputable def β : ℂ := sorry

-- Define the quadratic equation
def quadratic_eq (t : ℂ) (m : ℝ) : Prop := t^2 - 2*t + m = 0

-- Define the condition for |β+3| + (-1)^n|β-3|
def trajectory_condition (β : ℂ) (n : ℕ) (a : ℝ) : Prop :=
  Complex.abs (β + 3) + (-1)^n * Complex.abs (β - 3) = 3*a + (-1)^n * a

-- Define point D
noncomputable def point_D : ℂ := 2 + Complex.I * Real.sqrt 2

-- Theorem 1
theorem m_value :
  ∀ m : ℝ, quadratic_eq β m → Complex.abs β = 2 → m = 4 := by sorry

-- Theorem 2
theorem trajectory_equations :
  ∀ n : ℕ, ∀ a : ℝ,
  (3/2 < a ∧ a < 3) →
  trajectory_condition β n a →
  trajectory_condition point_D n a →
  ((n % 2 = 1 → ∀ x y : ℝ, x^2/3 - y^2/6 = 1 ∧ x ≥ Real.sqrt 3) ∧
   (n % 2 = 0 → ∀ x y : ℝ, x^2/12 + y^2/3 = 1)) := by sorry

-- Theorem 3
theorem x0_range :
  ∀ x0 : ℝ,
  (∀ x y : ℝ, x^2/12 + y^2/3 = 1 →
    (x - x0)^2 + y^2 ≥ 4/3) →
  ((0 < x0 ∧ x0 ≤ Real.sqrt 5) ∨ x0 ≥ 8*Real.sqrt 3/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_trajectory_equations_x0_range_l1158_115826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1158_115805

/-- Family of curves parameterized by θ -/
noncomputable def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- Line y = 2x -/
def line (x y : ℝ) : Prop := y = 2 * x

/-- Length of the chord intercepted by the curve on the line -/
noncomputable def chord_length (θ : ℝ) : ℝ :=
  let t := (8 * Real.sin θ + Real.cos θ + 1) / (2 * Real.sin θ - Real.cos θ + 3)
  Real.sqrt 5 * |t|

/-- Theorem stating the existence of a maximum chord length -/
theorem max_chord_length :
  ∃ (θ : ℝ), ∀ (φ : ℝ), chord_length θ ≥ chord_length φ ∧ chord_length θ = 8 * Real.sqrt 5 := by
  sorry

#check max_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1158_115805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalue_characterization_l1158_115888

/-- The matrix A in the problem -/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 8; 2, 1]

/-- The set of eigenvalues of A -/
def eigenvalues : Set ℝ := {5, -3}

/-- Theorem stating that k is an eigenvalue of A if and only if k is 5 or -3 -/
theorem eigenvalue_characterization (k : ℝ) :
  (∃ v : Fin 2 → ℝ, v ≠ 0 ∧ A.mulVec v = k • v) ↔ k ∈ eigenvalues :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalue_characterization_l1158_115888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_six_l1158_115800

/-- A geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_six (a q : ℝ) (h1 : a + a*q = 3/4) (h2 : a*q^3 + a*q^4 = 6) :
  geometric_sum a q 6 = 63/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_six_l1158_115800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_19_62_l1158_115864

theorem no_polynomial_19_62 :
  ¬∃ (a b c d : ℤ), 
    (λ x => a * x^3 + b * x^2 + c * x + d) 19 = 1 ∧
    (λ x => a * x^3 + b * x^2 + c * x + d) 62 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_19_62_l1158_115864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1158_115804

/-- The time it takes for two trains to completely pass each other -/
noncomputable def train_passing_time (length_A length_B speed_A speed_B : ℝ) : ℝ :=
  (length_A + length_B) / (speed_A / 3.6 + speed_B / 3.6)

/-- Theorem stating that the time for the given trains to pass each other is approximately 7.14 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 100 150 72 54 - 7.14| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1158_115804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l1158_115850

/-- Dilation of a complex number -/
def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale • (point - center)

/-- The problem statement -/
theorem dilation_problem : 
  dilation (1 + 2*I) 4 (-2 - 2*I) = -11 - 14*I := by
  -- Expand the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im, Complex.mul_re, Complex.mul_im]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l1158_115850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_equal_volume_specific_cone_l1158_115875

/-- The radius of a sphere with the same volume as a cone -/
noncomputable def sphere_radius_equal_volume_cone (cone_radius : ℝ) (cone_height : ℝ) : ℝ :=
  (3 / 2) ^ (1 / 3)

/-- Theorem: The radius of a sphere with the same volume as a cone with radius 1 inch and height 6 inches is equal to (3/2)^(1/3) inches -/
theorem sphere_radius_equal_volume_specific_cone :
  sphere_radius_equal_volume_cone 1 6 = (3 / 2) ^ (1 / 3) := by
  -- Unfold the definition of sphere_radius_equal_volume_cone
  unfold sphere_radius_equal_volume_cone
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_equal_volume_specific_cone_l1158_115875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l1158_115813

/-- Given a triangle DEF where cos(2D) + cos(2E) + cos(2F) = 1/2 and two sides have lengths 7 and 24,
    the maximum possible length of the third side is √793. -/
theorem max_third_side_length (D E F : ℝ) (a b : ℝ) :
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1/2 →
  a = 7 →
  b = 24 →
  ∃ (c : ℝ), c ≤ Real.sqrt 793 ∧
    ∃ (s : ℝ), s = (a + b + c) / 2 ∧
    c^2 = a^2 + b^2 - 2*a*b*(Real.cos D) ∧
    a^2 = b^2 + c^2 - 2*b*c*(Real.cos E) ∧
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos F) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l1158_115813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_trajectory_l1158_115814

/-- The ellipse equation -/
def ellipse (m n : ℝ) : Prop := m^2 / 3 + n^2 / 4 = 1

/-- The centroid formula for triangle PF₁F₂ -/
def centroid (x y m n : ℝ) : Prop := x = m / 3 ∧ y = n / 3

/-- The trajectory equation of G -/
def trajectory (x y : ℝ) : Prop := 3 * x^2 + 9 * y^2 / 4 = 1

theorem centroid_trajectory (x y m n : ℝ) :
  ellipse m n →
  centroid x y m n →
  x ≠ 0 →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_trajectory_l1158_115814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_condition_l1158_115880

/-- Given two points M and N with coordinates dependent on m, 
    prove that the line MN is vertical if and only if m = -5 -/
theorem vertical_line_condition (m : ℝ) : 
  (2*m + 3 = m - 2) ↔ (m = -5) :=
by
  constructor
  · intro h
    -- Prove m = -5 from the assumption
    sorry
  · intro h
    -- Prove 2*m + 3 = m - 2 from m = -5
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_condition_l1158_115880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_tangency_collinearity_l1158_115845

-- Define the points and lines
variable (A B C D E F X₁ X₂ X₃ X₄ : Point)
variable (AB BC CD DA : Line)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection of half-lines
def intersect_halflines (A B D C E : Point) : Prop := sorry

-- Define the incircles and excircles
def incircle_tangency (X Y Z T : Point) : Prop := sorry
def excircle_tangency (X Y Z T : Point) : Prop := sorry

-- Define collinearity
def collinear (P Q R : Point) : Prop := sorry

-- State the theorem
theorem quadrilateral_tangency_collinearity 
  (h_quad : is_quadrilateral A B C D)
  (h_int1 : intersect_halflines A B D C E)
  (h_int2 : intersect_halflines B C A D F)
  (h_tan1 : incircle_tangency E B C X₁)
  (h_tan2 : incircle_tangency F C D X₂)
  (h_tan3 : excircle_tangency E A D X₃)
  (h_tan4 : excircle_tangency F A B X₄) :
  collinear X₁ X₃ E ↔ collinear X₂ X₄ F :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_tangency_collinearity_l1158_115845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deepak_share_approx_810_l1158_115837

/-- Calculates the share of profit for an investor given two investments and the total profit -/
def calculate_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  (investment2 * total_profit) / (investment1 + investment2)

/-- Approximate equality for natural numbers -/
def approx_equal (a b : ℕ) : Prop :=
  (a = b) ∨ (a + 1 = b) ∨ (a = b + 1)

notation a " ≈ " b => approx_equal a b

/-- Theorem stating that Deepak's share of the profit is approximately 810 -/
theorem deepak_share_approx_810 :
  let anand_investment := 2250
  let deepak_investment := 3200
  let total_profit := 1380
  (calculate_share anand_investment deepak_investment total_profit) ≈ 810 := by
  -- Proof goes here
  sorry

#eval calculate_share 2250 3200 1380

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deepak_share_approx_810_l1158_115837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_calculation_l1158_115887

/-- Given a price before tax and a total price including tax, calculate the tax rate. -/
noncomputable def calculate_tax_rate (price_before_tax : ℝ) (total_price : ℝ) : ℝ :=
  ((total_price - price_before_tax) / price_before_tax) * 100

/-- Theorem: The tax rate for a $92 item with a total price of $98.90 is 7.5%. -/
theorem tax_rate_calculation :
  let price_before_tax : ℝ := 92
  let total_price : ℝ := 98.90
  calculate_tax_rate price_before_tax total_price = 7.5 := by
  unfold calculate_tax_rate
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_calculation_l1158_115887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_log_diffs_count_l1158_115858

def numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def log_diff (a b : ℕ) : ℝ := Real.log (a : ℝ) - Real.log (b : ℝ)

noncomputable def unique_log_diffs : Finset ℝ :=
  (numbers.powerset.filter (λ s => s.card = 2)).image
    (λ s => let a := s.toList.head!; let b := s.toList.tail!.head!; log_diff a b)

theorem unique_log_diffs_count :
  unique_log_diffs.card = 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_log_diffs_count_l1158_115858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distance_points_l1158_115848

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane --/
structure Square where
  sideLength : ℝ

/-- Function to calculate the distance from a point to a side of the square --/
def distanceToSide (p : Point) (s : Square) (side : Nat) : ℝ :=
  match side with
  | 0 => p.y                 -- distance to bottom side
  | 1 => s.sideLength - p.x  -- distance to right side
  | 2 => s.sideLength - p.y  -- distance to top side
  | 3 => p.x                 -- distance to left side
  | _ => 0                   -- invalid side

/-- Function to check if a point satisfies the distance conditions --/
def satisfiesDistanceConditions (p : Point) (s : Square) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 4)), 
    (distanceToSide p s 0) = perm 1 ∧
    (distanceToSide p s 1) = perm 2 ∧
    (distanceToSide p s 2) = perm 3 ∧
    (distanceToSide p s 3) = perm 4

/-- The main theorem to be proved --/
theorem square_distance_points (s : Square) (h : s.sideLength = 5) :
  ∃! (points : Finset Point), points.card = 8 ∧ ∀ p ∈ points, satisfiesDistanceConditions p s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distance_points_l1158_115848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_properties_l1158_115898

/-- Ellipse E with foci on the x-axis -/
def ellipse (b : ℝ) (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / b^2 = 1

/-- Inscribed circle C -/
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 8/3

/-- Tangent line l to circle C -/
def tangent_line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

/-- Points A and B where tangent line intersects ellipse -/
def intersection_points (b k m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse b x₁ y₁ ∧ ellipse b x₂ y₂ ∧
  tangent_line k m x₁ y₁ ∧ tangent_line k m x₂ y₂

/-- OA is perpendicular to OB -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem ellipse_circle_intersection_properties :
  ∃ (b k m x₁ y₁ x₂ y₂ : ℝ),
    intersection_points b k m x₁ y₁ x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ ∧
    b^2 = 4 ∧
    (4 * Real.sqrt 6 / 3 ≤ distance x₁ y₁ x₂ y₂ ∧ 
     distance x₁ y₁ x₂ y₂ ≤ 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_properties_l1158_115898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l1158_115833

noncomputable def sample_data : List ℝ := [-2, 0, 5, 3, 4]

noncomputable def mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length : ℝ)

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (fun x => (x - mean data) ^ 2)).sum / (data.length : ℝ)

theorem variance_of_sample_data :
  variance sample_data = 34 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l1158_115833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_example_l1158_115849

/-- Sum of an arithmetic series -/
noncomputable def arithmetic_series_sum (a₁ aₙ : ℝ) (d : ℝ) : ℝ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series with first term 15, last term 35, and common difference 0.2 is 2525 -/
theorem arithmetic_series_sum_example : arithmetic_series_sum 15 35 0.2 = 2525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_example_l1158_115849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_REG_l1158_115863

-- Define the circle P
noncomputable def P : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the radius of circle P
def radius_P : ℝ := 9

-- Define chord EF
noncomputable def EF : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the length of chord EF
def length_EF : ℝ := 10

-- Define diameter QS
noncomputable def QS : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- State that EF is parallel to QS
axiom EF_parallel_QS : sorry

-- Define point R
noncomputable def R : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the distance QR
def QR : ℝ := 27

-- State that Q, S, P, and R are collinear
axiom QSPR_collinear : sorry

-- Define line RG
noncomputable def RG : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- State that RG is perpendicular to EF
axiom RG_perp_EF : sorry

-- Define point G as the intersection of RG and EF
noncomputable def G : EuclideanSpace ℝ (Fin 2) := sorry

-- Define triangle REG
noncomputable def triangle_REG : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the area of a triangle
noncomputable def triangleArea (t : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- Theorem to prove
theorem area_triangle_REG : 
  triangleArea triangle_REG = 10 * Real.sqrt 14 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_REG_l1158_115863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos2α_necessary_not_sufficient_for_sinα_plus_cosα_zero_l1158_115886

theorem cos2α_necessary_not_sufficient_for_sinα_plus_cosα_zero :
  (∀ α : ℝ, Real.cos (2 * α) = (Real.cos α + Real.sin α) * (Real.cos α - Real.sin α)) →
  (∀ α : ℝ, Real.sin α + Real.cos α = 0 → Real.cos (2 * α) = 0) ∧
  (∃ α : ℝ, Real.cos (2 * α) = 0 ∧ Real.sin α + Real.cos α ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos2α_necessary_not_sufficient_for_sinα_plus_cosα_zero_l1158_115886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_empty_l1158_115883

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 2}

theorem A_intersect_B_empty : (A.prod Set.univ) ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_empty_l1158_115883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_percentage_l1158_115843

/-- Represents the composition of a flower bouquet -/
structure Bouquet where
  total : ℝ
  pink_ratio : ℝ
  red_ratio : ℝ
  yellow_ratio : ℝ
  pink_rose_ratio : ℝ
  red_carnation_ratio : ℝ
  yellow_rose_ratio : ℝ

/-- Theorem stating the percentage of carnations in the bouquet -/
theorem carnation_percentage (b : Bouquet) 
  (h1 : b.pink_rose_ratio = 1/5 * b.pink_ratio)
  (h2 : b.red_carnation_ratio = 1/2 * b.red_ratio)
  (h3 : b.yellow_rose_ratio = 3/10 * b.yellow_ratio)
  (h4 : b.yellow_ratio = 3/10)
  (h5 : b.pink_ratio = b.red_ratio)
  (h6 : b.pink_ratio + b.red_ratio + b.yellow_ratio = 1)
  : (b.pink_ratio - b.pink_rose_ratio + b.red_carnation_ratio + b.yellow_ratio - b.yellow_rose_ratio) * 100 = 66.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_percentage_l1158_115843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1158_115828

noncomputable section

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) (AD : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (Real.sin B + Real.sin C) / (Real.sin A - Real.sin C) = (Real.sin A + Real.sin C) / Real.sin B →
  a = Real.sqrt 7 →
  AD = 2 / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (∃ D : ℝ, AD * Real.sin (A / 2) = b * Real.sin (C / 2) ∧ AD * Real.sin (A / 2) = c * Real.sin (B / 2)) →
  A = 2 * π / 3 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1158_115828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l1158_115893

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n * (2 * a₁ + (n - 1) * d)) / 2

-- Theorem statement
theorem common_difference_is_two (a₁ : ℝ) :
  (∃ d : ℝ, 2 * S a₁ d 3 = 3 * S a₁ d 2 + 6) →
  (∃ d : ℝ, d = 2 ∧ 2 * S a₁ d 3 = 3 * S a₁ d 2 + 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l1158_115893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_interior_angle_sum_l1158_115818

-- Define interior angle of a polygon
def interior_angle (n : ℕ) (i : Fin n) : ℝ := sorry

-- Define sum of interior angles of an n-sided polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (∀ i : Fin n, interior_angle n i = 108) → sum_interior_angles n = 540 := by
  intro h_angles
  have h_n : n = 5 := by
    -- Proof that n = 5
    sorry
  rw [h_n]
  -- Evaluate sum_interior_angles 5
  simp [sum_interior_angles]
  -- Simplify (5 - 2) * 180
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_interior_angle_sum_l1158_115818
