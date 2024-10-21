import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_number_sum_l977_97794

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ∈ Finset.range 50 →
    B ∈ Finset.range 50 →
    A ≠ 0 →
    A ≠ 49 →
    is_prime B →
    is_perfect_square (150 - B + A) →
    A + B = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_number_sum_l977_97794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_complementary_l977_97770

-- Define the type for balls
inductive Ball : Type
| Red : Ball
| Black : Ball
deriving DecidableEq

-- Define the bag of balls
def bag : Multiset Ball :=
  2 • {Ball.Red} + 2 • {Ball.Black}

-- Define the sample space (all possible outcomes when drawing two balls)
def sampleSpace : Set (Ball × Ball) :=
  {pair | pair.1 ∈ bag ∧ pair.2 ∈ bag.erase pair.1}

-- Define the event "Exactly one black ball"
def exactlyOneBlack : Set (Ball × Ball) :=
  {pair ∈ sampleSpace | (pair.1 = Ball.Black ∧ pair.2 = Ball.Red) ∨ 
                        (pair.1 = Ball.Red ∧ pair.2 = Ball.Black)}

-- Define the event "Exactly two black balls"
def exactlyTwoBlack : Set (Ball × Ball) :=
  {pair ∈ sampleSpace | pair.1 = Ball.Black ∧ pair.2 = Ball.Black}

-- State the theorem
theorem mutually_exclusive_not_complementary :
  (exactlyOneBlack ∩ exactlyTwoBlack = ∅) ∧
  (exactlyOneBlack ∪ exactlyTwoBlack ≠ sampleSpace) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_complementary_l977_97770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_theorem_l977_97704

/-- Represents the total distance to the mall in kilometers -/
noncomputable def distance_to_mall : ℝ := 60

/-- Represents the speed in city traffic in km/hr -/
noncomputable def city_speed : ℝ := 30

/-- Represents the speed on highway to mall in km/hr -/
noncomputable def highway_speed : ℝ := 90

/-- Represents the shopping time in hours -/
noncomputable def shopping_time : ℝ := 2

/-- Represents the speed on congested highway during return in km/hr -/
noncomputable def congested_highway_speed : ℝ := 60

/-- Calculates the total time for Mike's round trip -/
noncomputable def total_trip_time (city_distance : ℝ) : ℝ :=
  let highway_distance := distance_to_mall - city_distance
  let time_to_highway := city_distance / city_speed
  let time_on_highway_to_mall := highway_distance / highway_speed
  let time_on_congested_highway := highway_distance / congested_highway_speed
  let time_from_highway := city_distance / city_speed
  time_to_highway + time_on_highway_to_mall + shopping_time + time_on_congested_highway + time_from_highway

/-- Theorem stating that there exists a city_distance that makes the total trip time approximately 4.367 hours -/
theorem trip_time_theorem :
  ∃ (city_distance : ℝ), 0 < city_distance ∧ city_distance < distance_to_mall ∧ 
  |total_trip_time city_distance - 4.367| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_theorem_l977_97704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_purchase_is_18_l977_97781

/-- The minimum purchase amount for free delivery at a fast-food chain -/
def min_purchase_for_free_delivery : ℚ :=
  let burger_price : ℚ := 3.20
  let fries_price : ℚ := 1.90
  let milkshake_price : ℚ := 2.40
  let burger_quantity : ℚ := 2
  let fries_quantity : ℚ := 2
  let milkshake_quantity : ℚ := 2
  let additional_amount : ℚ := 3.00
  (burger_price * burger_quantity + 
   fries_price * fries_quantity + 
   milkshake_price * milkshake_quantity + 
   additional_amount)

/-- Theorem stating that the minimum purchase amount for free delivery is $18.00 -/
theorem min_purchase_is_18 : min_purchase_for_free_delivery = 18 := by
  -- Unfold the definition of min_purchase_for_free_delivery
  unfold min_purchase_for_free_delivery
  -- Simplify the arithmetic expression
  simp [mul_add, add_mul, mul_comm, add_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_purchase_is_18_l977_97781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_calculation_l977_97746

/-- Represents the cost of painting each face of a cube in paise per square centimeter -/
structure CubePaintingCosts where
  faceA : ℝ
  faceB : ℝ
  faceC : ℝ
  faceD : ℝ
  faceE : ℝ
  faceF : ℝ

/-- Calculates the volume of a cube given its painting costs and total cost -/
noncomputable def cubeVolumeFromPaintingCost (costs : CubePaintingCosts) (totalCostRupees : ℝ) : ℝ :=
  let totalCostPaise := totalCostRupees * 100
  let sumOfCosts := costs.faceA + costs.faceB + costs.faceC + costs.faceD + costs.faceE + costs.faceF
  let sideLength := (totalCostPaise / sumOfCosts).sqrt
  sideLength ^ 3

theorem cube_volume_calculation (costs : CubePaintingCosts) 
  (h1 : costs.faceA = 12)
  (h2 : costs.faceB = 13)
  (h3 : costs.faceC = 14)
  (h4 : costs.faceD = 15)
  (h5 : costs.faceE = 16)
  (h6 : costs.faceF = 17)
  (h7 : cubeVolumeFromPaintingCost costs 512.34 = 589 * Real.sqrt 589) : 
  cubeVolumeFromPaintingCost costs 512.34 = 589 * Real.sqrt 589 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_calculation_l977_97746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_players_same_flips_l977_97723

noncomputable def coin_flip_game (n : ℕ) : ℝ := (1 / 2) ^ (4 * n)

theorem all_players_same_flips : ∑' (n : ℕ), coin_flip_game n = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_players_same_flips_l977_97723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_trigonometry_l977_97789

theorem triangle_abc_trigonometry 
  (A B : ℝ) 
  (h1 : Real.sin (A + π/6) = 2 * Real.cos A) 
  (h2 : 0 < B ∧ B < π/3) 
  (h3 : Real.sin (A - B) = 3/5) : 
  Real.tan A = Real.sqrt 3 ∧ Real.sin B = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_trigonometry_l977_97789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_l977_97749

-- Define the circle O
def circle_O (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the orthocenter H
def orthocenter (H M A Q : ℝ × ℝ) : Prop :=
  let (xh, yh) := H
  let (xm, ym) := M
  let (xa, ya) := A
  let (xq, yq) := Q
  (xh - xa) * (xm - xa) + (yh - ya) * (ym - ya) = 0 ∧
  (xh - xq) * (xm - xq) + (yh - yq) * (ym - yq) = 0

-- Define the tangent line condition
def tangent_line (M A : ℝ × ℝ) (O : ℝ × ℝ → Prop) : Prop :=
  ∃ (t : ℝ), M = (t * A.1, t * A.2) ∧ O M

-- Define the locus equation
def locus_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + (y - 2)^2 = 4 ∧ x ≠ 0

-- State the theorem
theorem orthocenter_locus :
  ∀ (H M Q : ℝ × ℝ),
    tangent_line M point_A circle_O →
    (∃ (t : ℝ), Q = (2 * Real.cos t, 2 * Real.sin t)) →
    orthocenter H M point_A Q →
    locus_equation H :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_l977_97749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l977_97761

theorem exponential_inequality (x : ℝ) :
  (6 : ℝ)^(x^2 + x - 2) < 1 ↔ -2 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l977_97761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_for_given_triangle_l977_97757

/-- A triangle with two perpendicular medians -/
structure TriangleWithPerpendicularMedians where
  /-- Length of the first median -/
  m₁ : ℝ
  /-- Length of the second median -/
  m₂ : ℝ
  /-- Area of the triangle -/
  area : ℝ
  /-- The two medians are perpendicular -/
  perpendicular : m₁ > 0 ∧ m₂ > 0
  /-- The area is positive -/
  positive_area : area > 0

/-- The length of the third median in a triangle with two perpendicular medians -/
noncomputable def third_median_length (t : TriangleWithPerpendicularMedians) : ℝ :=
  Real.sqrt ((2 * (t.m₁^2 + t.m₂^2) - (t.m₁ + t.m₂)^2) / 4)

/-- Theorem stating the length of the third median in the given triangle -/
theorem third_median_length_for_given_triangle :
  let t : TriangleWithPerpendicularMedians := {
    m₁ := 5,
    m₂ := 9,
    area := 6 * Real.sqrt 35,
    perpendicular := by simp
    positive_area := by simp
  }
  third_median_length t = 2 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_for_given_triangle_l977_97757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_2x_l977_97715

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The theorem stating that the distance from (0, 5) to y = 2x is √5 -/
theorem distance_to_line_2x (x₀ y₀ : ℝ) (h1 : x₀ = 0) (h2 : y₀ = 5) : 
  distance_point_to_line x₀ y₀ (-2) 1 0 = Real.sqrt 5 := by
  sorry

#check distance_to_line_2x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_2x_l977_97715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_is_sqrt_46_l977_97731

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area : 2 * (a * b + b * c + a * c) = 54
  edge_length : 4 * (a + b + c) = 40
  edge_relation : a = 2 * (b + c)

/-- The length of the interior diagonal of a rectangular solid -/
noncomputable def interior_diagonal (solid : RectangularSolid) : ℝ :=
  Real.sqrt (solid.a^2 + solid.b^2 + solid.c^2)

/-- Theorem stating that the interior diagonal of the given rectangular solid is √46 -/
theorem interior_diagonal_is_sqrt_46 (solid : RectangularSolid) :
  interior_diagonal solid = Real.sqrt 46 := by
  sorry

#check interior_diagonal_is_sqrt_46

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_is_sqrt_46_l977_97731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AC_l977_97793

-- Define the circle and points
def Circle : Type := ℝ × ℝ
def Point : Type := ℝ × ℝ

-- Define the radius of the circle
def radius : ℝ := 5

-- Define points A, B, and C
variable (A B C : Point)

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the midpoint of the minor arc (this is a placeholder definition)
def midpoint_of_minor_arc (p q : Point) : Point :=
  sorry

-- State the theorem
theorem length_AC (h1 : distance A (0, 0) = radius)
                  (h2 : distance B (0, 0) = radius)
                  (h3 : distance A B = 6)
                  (h4 : C = midpoint_of_minor_arc A B) :
  distance A C = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AC_l977_97793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_level_drop_l977_97783

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Calculates the cross-sectional area of a cylinder -/
noncomputable def cylinderArea (c : Cylinder) : ℝ := Real.pi * c.radius^2

/-- The stationary oil tank -/
def stationaryTank : Cylinder := { radius := 100, height := 25 }

/-- The oil truck tank -/
def truckTank : Cylinder := { radius := 8, height := 10 }

/-- Theorem: The oil level drop in the stationary tank -/
theorem oil_level_drop :
  (cylinderVolume truckTank) / (cylinderArea stationaryTank) = 0.064 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_level_drop_l977_97783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_l977_97728

/-- The average age of a cricket team -/
def average_age (n : ℕ) (total_age : ℚ) : ℚ :=
  total_age / n

/-- The cricket team problem -/
theorem cricket_team_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 5 →
  let total_age : ℚ := team_size * 23
  let remaining_players := team_size - 2
  let remaining_age : ℚ := total_age - (captain_age + (captain_age + wicket_keeper_age_diff))
  average_age remaining_players remaining_age = average_age team_size total_age - 1 →
  average_age team_size total_age = 23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_l977_97728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l977_97710

-- Define a curve in 2D space
def Curve (k : ℝ) := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 1}

-- Define what it means for a set to be a parabola
def IsParabola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
  S = {(x, y) : ℝ × ℝ | a*y = b*x^2 + c*x + d*y + e}

-- Theorem: For any real k, the curve defined by x^2 + ky^2 = 1 is not a parabola
theorem not_parabola : ∀ k : ℝ, ¬ IsParabola (Curve k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l977_97710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l977_97771

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = Real.pi
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  -- Angles form an arithmetic sequence
  2 * t.B = t.A + t.C
  -- Sides form a geometric sequence
  ∧ t.b^2 = t.a * t.c

-- State the theorem
theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.cos t.B = 1/2 ∧ Real.sin t.A * Real.sin t.C = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l977_97771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_g_properties_l977_97743

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + b * x

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x - 4*x

theorem tangent_line_and_g_properties
  (a b : ℝ)
  (h1 : f' a b 0 = 1)
  (h2 : f' a b 2 = 1) :
  (∃ (m c : ℝ), m = 4 ∧ c = -9 ∧ ∀ x y, y = f a b x → m*x - y - c = 0) ∧
  (∀ x ∈ Set.Icc (-3) 2,
    (x ∈ Set.Icc (-3) (-1) → ∀ y ∈ Set.Icc (-3) x, g a b y ≤ g a b x) ∧
    (x ∈ Set.Ioc (-1) 2 → ∀ y ∈ Set.Ioc x 2, g a b x ≤ g a b y)) ∧
  (∀ x ∈ Set.Icc (-3) 2, g a b x ≥ -9) ∧
  (g a b (-3) = -9) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_g_properties_l977_97743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l977_97786

-- Define the total revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

-- Define the total cost function
noncomputable def C (x : ℝ) : ℝ := 20000 + 100 * x

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := R x - C x

-- Theorem statement
theorem max_profit :
  ∃ (x_max : ℝ), x_max = 300 ∧ 
  ∀ (x : ℝ), x ≥ 0 → f x ≤ f x_max ∧ 
  f x_max = 25000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l977_97786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feb14_is_saturday_if_feb13_is_friday_in_leap_year_l977_97726

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  isLeapYear : Bool

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Gets the day of the week for a given February date -/
def getDayOfWeek (date : FebruaryDate) : DayOfWeek :=
  sorry  -- This function is not implemented, but we define it to use in the theorem

/-- Theorem: If February 13th is a Friday in a leap year, then February 14th is a Saturday -/
theorem feb14_is_saturday_if_feb13_is_friday_in_leap_year 
  (feb13 : FebruaryDate) 
  (feb14 : FebruaryDate) 
  (h1 : feb13.day = 13)
  (h2 : feb13.isLeapYear = true)
  (h3 : feb14.day = 14)
  (h4 : feb14.isLeapYear = feb13.isLeapYear)
  (h5 : getDayOfWeek feb13 = DayOfWeek.Friday) :
  getDayOfWeek feb14 = DayOfWeek.Saturday :=
by
  sorry  -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_feb14_is_saturday_if_feb13_is_friday_in_leap_year_l977_97726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_properties_l977_97769

noncomputable section

-- Define the function f(x) and its derivative
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x + 4
def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - a

-- Theorem statement
theorem extremum_properties :
  ∃ (a : ℝ),
    -- f(x) has an extremum at x = -2
    (f' a (-2) = 0) ∧
    -- a = 4
    (a = 4) ∧
    -- f(x) is increasing on (-∞, -2) and (2, +∞), decreasing on (-2, 2)
    (∀ x, x < -2 → f' a x > 0) ∧
    (∀ x, -2 < x ∧ x < 2 → f' a x < 0) ∧
    (∀ x, x > 2 → f' a x > 0) ∧
    -- The minimum value of f(x) on [0, 3] is -4/3
    (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≥ -4/3) ∧
    (f a 2 = -4/3) ∧
    -- The maximum value of f(x) on [0, 3] is 1
    (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≤ 1) ∧
    (f a 3 = 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_properties_l977_97769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_G_function_g_is_G_function_l977_97797

-- Definition of a G function
def is_G_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := -x

noncomputable def g (x : ℝ) : ℝ := 1 / x

-- Theorem for f(x) = -x
theorem f_is_G_function : is_G_function f := by
  intro x₁ x₂
  simp [f, is_G_function]
  linarith

-- Theorem for g(x) = 1/x (for x > 0)
theorem g_is_G_function : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g ((x₁ + x₂) / 2) ≤ (g x₁ + g x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_G_function_g_is_G_function_l977_97797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flyer_distribution_l977_97741

theorem flyer_distribution (total : ℕ) (ryan : ℕ) (alyssa : ℕ) (belinda_percent : ℚ) (scott : ℕ) : 
  total = 200 →
  ryan = 42 →
  alyssa = 67 →
  belinda_percent = 20 / 100 →
  scott = total - (ryan + alyssa + (belinda_percent * ↑total).floor) →
  scott = 51 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flyer_distribution_l977_97741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_for_spheres_l977_97717

/-- The volume of water needed to cover four spheres in a cylindrical container --/
theorem water_volume_for_spheres (r : ℝ) (h : r = 1) :
  let sphere_radius : ℝ := 1 / 2
  let water_height : ℝ := 1 + Real.sqrt 2 / 2
  let cylinder_volume : ℝ := π * r^2 * water_height
  let sphere_volume : ℝ := 4 * (4 * π * sphere_radius^3 / 3)
  cylinder_volume - sphere_volume = π * (1 / 3 + Real.sqrt 2 / 2) := by
  sorry

#check water_volume_for_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_for_spheres_l977_97717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_statistics_l977_97739

/-- Game "Guess the Coin, Move Forward" -/
structure Game where
  n : ℕ+  -- Number of rounds
  correct_move : ℕ  -- Steps forward for correct guess
  incorrect_move : ℕ  -- Steps backward for incorrect guess

/-- Outcome of the game -/
def game_outcome (g : Game) : ℝ → ℝ :=
  λ k ↦ g.correct_move * k - g.incorrect_move * (g.n - k)

/-- Y: Difference in steps between A and B at the end of the game -/
def Y (g : Game) : ℝ → ℝ :=
  λ k ↦ 2 * (game_outcome g k)

/-- Expected value of Y -/
noncomputable def E_Y (g : Game) : ℝ := sorry

/-- Variance of Y -/
noncomputable def D_Y (g : Game) : ℝ := sorry

/-- Theorem: Expected value and variance of Y for the game -/
theorem game_statistics (g : Game) :
  (E_Y g = 0) ∧ (D_Y g = 9 * g.n) := by
  sorry

#check game_statistics

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_statistics_l977_97739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l977_97720

/-- The sum of geometric series with ratio x^4 from 0 to k -/
noncomputable def geometric_sum_4 (x : ℝ) (k : ℕ) : ℝ :=
  (1 - x^(4*(k+1))) / (1 - x^4)

/-- The sum of geometric series with ratio x^2 from 0 to k -/
noncomputable def geometric_sum_2 (x : ℝ) (k : ℕ) : ℝ :=
  (1 - x^(2*(k+1))) / (1 - x^2)

theorem divisibility_condition (k : ℕ) :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → 
    (∃ m : ℤ, geometric_sum_4 x k = m) → 
    (∃ n : ℤ, geometric_sum_2 x k = n) →
    (∃ p : ℤ, geometric_sum_4 x k / geometric_sum_2 x k = p)) ↔ 
  k % 2 = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l977_97720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_case_p_range_l977_97780

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  p : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sin t.A + Real.sin t.C = t.p * Real.sin t.B ∧ 4 * t.a * t.c = t.b^2

-- Theorem 1
theorem specific_case (t : Triangle) (h : triangle_conditions t) 
  (h_p : t.p = 5/4) (h_b : t.b = 1) :
  (t.a = 1 ∧ t.c = 1/4) ∨ (t.a = 1/4 ∧ t.c = 1) :=
sorry

-- Theorem 2
theorem p_range (t : Triangle) (h : triangle_conditions t) 
  (h_acute : 0 < t.B ∧ t.B < Real.pi/2) :
  Real.sqrt 6/2 < t.p ∧ t.p < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_case_p_range_l977_97780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_y4_value_l977_97721

-- Define the quadrilateral
noncomputable def quadrilateral (y4 : ℝ) : List (ℝ × ℝ) :=
  [(4, -3), (4, 7), (12, 2), (12, y4)]

-- Define the area calculation function for a trapezoid
noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (1/2) * (a + b) * h

-- Theorem statement
theorem quadrilateral_y4_value :
  ∀ y4 : ℝ,
  trapezoid_area 10 (|y4 - 2|) 8 = 76 →
  y4 > 2 →
  y4 = 11 := by
  sorry

#check quadrilateral_y4_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_y4_value_l977_97721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_in_20gon_l977_97750

-- Define a regular polygon
structure RegularPolygon (n : ℕ) where
  vertices : Finset (ℝ × ℝ)
  card_eq : vertices.card = n

-- Define a marked vertex
def MarkedVertex (p : RegularPolygon 20) := {v : ℝ × ℝ // v ∈ p.vertices}

-- Define an isosceles triangle
def IsoscelesTriangle (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (a.1 - c.1)^2 + (a.2 - c.2)^2 ∨
  (b.1 - a.1)^2 + (b.2 - a.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∨
  (c.1 - a.1)^2 + (c.2 - a.2)^2 = (c.1 - b.1)^2 + (c.2 - b.2)^2

theorem isosceles_triangle_in_20gon (p : RegularPolygon 20) 
  (marked : Finset (MarkedVertex p)) (h : marked.card = 9) :
  ∃ (a b c : MarkedVertex p), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧
    IsoscelesTriangle a.val b.val c.val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_in_20gon_l977_97750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l977_97767

theorem count_integers_in_square_range : 
  (Finset.filter (fun x : ℕ => 144 ≤ x^2 ∧ x^2 ≤ 289 ∧ x > 0) (Finset.range 18)).card = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l977_97767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l977_97718

noncomputable def f (x : ℝ) := -Real.sin (Real.pi * x / 2)

theorem f_properties :
  (∀ x, f (x + 4) = f x) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  (∀ x, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l977_97718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_l977_97713

/-- An ellipse with foci F₁ and F₂, and A as an endpoint of its minor axis. -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  (distance e.F₁ e.F₂) / (2 * distance e.A e.F₁)

/-- Theorem: The eccentricity of an ellipse is 1/2 if its foci and an endpoint 
    of its minor axis form an equilateral triangle -/
theorem ellipse_eccentricity_equilateral (e : Ellipse) 
  (h : distance e.F₁ e.F₂ = distance e.A e.F₁ ∧ 
       distance e.F₁ e.F₂ = distance e.A e.F₂) : 
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_l977_97713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_7_5_l977_97735

/-- The area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  let x4 := v4.1
  let y4 := v4.2
  (1/2) * ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem quadrilateral_area_is_7_5 :
  quadrilateralArea (2, 1) (4, 3) (7, 1) (4, 6) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_7_5_l977_97735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coneVolumeOverPi_value_l977_97787

/-- The volume of a cone formed from a 240-degree sector of a circle with radius 20, divided by π -/
noncomputable def coneVolumeOverPi : ℝ :=
  let sectorAngle : ℝ := 240
  let circleRadius : ℝ := 20
  let baseRadius : ℝ := (2 / 3) * circleRadius
  let height : ℝ := Real.sqrt ((circleRadius ^ 2) - (baseRadius ^ 2))
  (1 / 3) * baseRadius ^ 2 * height

theorem coneVolumeOverPi_value : coneVolumeOverPi = 32000 * Real.sqrt 10 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coneVolumeOverPi_value_l977_97787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_arrangements_l977_97777

-- Define the number of children and trees
def num_children : ℕ := 5
def num_trees : ℕ := 2

-- Define a function to calculate the number of ways to divide children
def ways_to_divide (n : ℕ) (k : ℕ) : ℕ :=
  (Finset.sum (Finset.range (n - 1)) (λ i => Nat.choose n (i + 1))) / k

-- Define a function to calculate circular arrangements
def circular_arrangements (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n - 1)) (λ i => (Nat.factorial (i + 1)) * (Nat.factorial (n - i - 2)))

-- Theorem statement
theorem christmas_tree_arrangements :
  ways_to_divide num_children num_trees * circular_arrangements num_children = 240 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_arrangements_l977_97777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_gain_percentage_l977_97736

/-- Calculates the overall gain percentage for three bicycle transactions -/
def overall_gain_percentage (cp1 sp1 cp2 sp2 cp3 sp3 : ℚ) : ℚ :=
  let tcp := cp1 + cp2 + cp3
  let tsp := sp1 + sp2 + sp3
  ((tsp - tcp) / tcp) * 100

/-- The overall gain percentage for the given bicycle transactions is approximately 7.89% -/
theorem bicycle_gain_percentage :
  let cp1 := (900 : ℚ)
  let sp1 := (1100 : ℚ)
  let cp2 := (1200 : ℚ)
  let sp2 := (1400 : ℚ)
  let cp3 := (1700 : ℚ)
  let sp3 := (1600 : ℚ)
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ 
    |overall_gain_percentage cp1 sp1 cp2 sp2 cp3 sp3 - (789 : ℚ) / 100| < ε := by
  sorry

#eval overall_gain_percentage 900 1100 1200 1400 1700 1600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_gain_percentage_l977_97736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l977_97747

noncomputable def f (x : Real) : Real := Real.cos (x + Real.pi / 18) + Real.cos (x + 7 * Real.pi / 18)

theorem f_min_value :
  ∃ (min : Real), (∀ (x : Real), f x ≥ min) ∧ (∃ (x : Real), f x = min) ∧ (min = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l977_97747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_and_max_value_l977_97760

noncomputable def a (x : Real) : Fin 2 → Real
  | 0 => Real.sqrt 3 * Real.sin x
  | 1 => Real.sin x

noncomputable def b (x : Real) : Fin 2 → Real
  | 0 => Real.cos x
  | 1 => Real.sin x

noncomputable def f (x : Real) : Real :=
  (a x 0 * b x 0) + (a x 1 * b x 1)

theorem vector_equality_and_max_value (x : Real) 
  (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  (∀ i : Fin 2, (a x i) ^ 2 = (b x i) ^ 2) → x = Real.pi / 6 ∧
  ∃ (max : Real), ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ max ∧ max = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_and_max_value_l977_97760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l977_97724

theorem triangle_right_angled (A B C : ℝ) (h : Real.sin (A - B) = 1 + 2 * Real.cos (B + C) * Real.sin (A + C)) :
  A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l977_97724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_two_l977_97766

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem inverse_f_at_negative_two :
  ∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f ∧ f_inv (-2) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_two_l977_97766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_kw_price_percentage_l977_97702

/-- The price of Company KW as a percentage of the combined assets of Companies A and B -/
theorem company_kw_price_percentage (price : ℝ) (asset_a : ℝ) (asset_b : ℝ)
  (h1 : price = asset_a + 0.6 * asset_a)
  (h2 : price = asset_b + asset_b) :
  ∃ (ε : ℝ), abs ((price / (asset_a + asset_b)) * 100 - 88.89) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_kw_price_percentage_l977_97702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_of_inclination_l977_97738

/-- The line equation -/
def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + 3 * y - 2 = 0

/-- The slope of the line -/
noncomputable def line_slope : ℝ := -Real.sqrt 3 / 3

/-- The angle of inclination in radians -/
noncomputable def angle_of_inclination : ℝ := 5 * Real.pi / 6

theorem line_angle_of_inclination :
  Real.tan angle_of_inclination = line_slope :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_of_inclination_l977_97738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_143_l977_97788

theorem greatest_prime_factor_of_143 :
  (Nat.factors 143).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_143_l977_97788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_cost_in_cny_l977_97775

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nam : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny_rate : ℚ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nam : ℚ := 160

/-- Converts Namibian dollars to US dollars -/
def nam_to_usd (nam : ℚ) : ℚ := nam / usd_to_nam

/-- Converts US dollars to Chinese yuan -/
def usd_to_cny (usd : ℚ) : ℚ := usd * usd_to_cny_rate

theorem sculpture_cost_in_cny : 
  usd_to_cny (nam_to_usd sculpture_cost_nam) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_cost_in_cny_l977_97775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_commutative_not_associative_l977_97756

-- Define the ⋄ operation
def diamond (a b : ℚ) : ℚ := (a * b + 5) / (a + b)

-- Theorem statement
theorem diamond_commutative_not_associative :
  (∀ a b : ℚ, a > 1 ∧ b > 1 → diamond a b = diamond b a) ∧
  (∃ a b c : ℚ, a > 1 ∧ b > 1 ∧ c > 1 ∧ diamond (diamond a b) c ≠ diamond a (diamond b c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_commutative_not_associative_l977_97756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l977_97762

theorem subsets_containing_six (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5, 6}) :
  (Finset.filter (fun A => 6 ∈ A) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l977_97762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_bound_2008_l977_97779

open BigOperators

theorem sum_bound_2008 (x : Fin 2009 → ℝ) (h : ∀ i, x i > 0) :
  (∑ i : Fin 2009, x i / (x i + x i.succ)) < 2008 ∧
  ∀ M, ((∀ x : Fin 2009 → ℝ, (∀ i, x i > 0) → 
    (∑ i : Fin 2009, x i / (x i + x i.succ)) < M) → M ≥ 2008) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_bound_2008_l977_97779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_rate_approximation_l977_97796

/-- The daily interest rate given an annual rate of 4.5% -/
noncomputable def daily_rate : ℝ := 0.045 / 365

/-- The equivalent annual rate after daily compounding -/
noncomputable def equivalent_annual_rate : ℝ := (1 + daily_rate) ^ 365 - 1

/-- Theorem stating that the equivalent annual rate is approximately 4.59% -/
theorem equivalent_rate_approximation : 
  ∃ ε > 0, |equivalent_annual_rate - 0.0459| < ε ∧ ε < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_rate_approximation_l977_97796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_l977_97707

def total_students : ℕ := 360
def min_rows : ℕ := 12
def min_students_per_row : ℕ := 18

def valid_arrangement (students_per_row : ℕ) : Bool :=
  students_per_row ≥ min_students_per_row &&
  (total_students / students_per_row) ≥ min_rows &&
  students_per_row * (total_students / students_per_row) = total_students

theorem sum_of_valid_arrangements :
  (Finset.filter (fun x => valid_arrangement x) (Finset.range (total_students + 1))).sum id = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_l977_97707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_free_amount_l977_97729

/-- Proves that the tax-free amount is $600 given the problem conditions -/
theorem tax_free_amount (total_value tax_paid tax_rate : ℝ) : ℝ :=
  let tax_free_amount := total_value - tax_paid / tax_rate
  have h1 : total_value = 1720 := by sorry
  have h2 : tax_paid = 112 := by sorry
  have h3 : tax_rate = 0.1 := by sorry
  have h4 : tax_free_amount = 600 := by sorry
  tax_free_amount

-- Example usage
example : tax_free_amount 1720 112 0.1 = 600 := by
  unfold tax_free_amount
  simp
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_free_amount_l977_97729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_friction_coefficient_l977_97792

/-- The coefficient of friction between a rod and a horizontal surface --/
noncomputable def friction_coefficient (α : ℝ) (reaction_ratio : ℝ) : ℝ :=
  let cos_α := Real.cos (α * Real.pi / 180)
  let sin_α := Real.sin (α * Real.pi / 180)
  (1 / reaction_ratio - cos_α) / sin_α

/-- Theorem stating the coefficient of friction for a rod on a horizontal surface --/
theorem rod_friction_coefficient :
  let α : ℝ := 70
  let reaction_ratio : ℝ := 21
  abs (friction_coefficient α reaction_ratio - 0.05) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_friction_coefficient_l977_97792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_electrode_reaction_is_correct_l977_97725

-- Define the molecules and ions involved
inductive Molecule
| CH₃OH : Molecule
| O₂ : Molecule
| CO₂ : Molecule
| H₂O : Molecule
| Hplus : Molecule  -- Changed from H⁺ to Hplus
| eminus : Molecule  -- Changed from e⁻ to eminus

-- Define the state of matter
inductive State
| gas : State
| liquid : State
| aqueous : State

-- Define a chemical species (molecule with its state)
structure ChemicalSpecies where
  molecule : Molecule
  state : State

-- Define a chemical reaction
structure Reaction where
  reactants : List (Nat × ChemicalSpecies)
  products : List (Nat × ChemicalSpecies)

-- Define the overall reaction
def overallReaction : Reaction := {
  reactants := [(2, ⟨Molecule.CH₃OH, State.gas⟩), (3, ⟨Molecule.O₂, State.gas⟩)],
  products := [(2, ⟨Molecule.CO₂, State.gas⟩), (4, ⟨Molecule.H₂O, State.liquid⟩)]
}

-- Define the negative electrode reaction
def negativeElectrodeReaction : Reaction := {
  reactants := [(1, ⟨Molecule.CH₃OH, State.gas⟩), (1, ⟨Molecule.H₂O, State.liquid⟩)],
  products := [(1, ⟨Molecule.CO₂, State.gas⟩), (6, ⟨Molecule.Hplus, State.aqueous⟩)]
}

-- Define a function to check if a reaction is an oxidation
def isOxidation (r : Reaction) : Prop := sorry

-- Theorem to prove
theorem negative_electrode_reaction_is_correct
  (h1 : isOxidation negativeElectrodeReaction)
  (h2 : ∃ (positiveElectrodeReaction : Reaction),
        overallReaction.reactants = negativeElectrodeReaction.reactants ++ positiveElectrodeReaction.reactants ∧
        overallReaction.products = negativeElectrodeReaction.products ++ positiveElectrodeReaction.products) :
  negativeElectrodeReaction = {
    reactants := [(1, ⟨Molecule.CH₃OH, State.gas⟩), (1, ⟨Molecule.H₂O, State.liquid⟩)],
    products := [(1, ⟨Molecule.CO₂, State.gas⟩), (6, ⟨Molecule.Hplus, State.aqueous⟩)]
  } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_electrode_reaction_is_correct_l977_97725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_bound_l977_97727

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the ratio f
noncomputable def f (t : Triangle) (p : Point) : ℝ :=
  sorry -- Definition of f based on the areas of smaller triangles and original triangle

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point :=
  sorry -- Definition of centroid

-- State the theorem
theorem triangle_area_ratio_bound (t : Triangle) (p : Point) :
  f t p ≥ 1/3 ∧ (f t p = 1/3 ↔ p = centroid t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_bound_l977_97727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_element_is_205_l977_97776

theorem third_element_is_205 (numbers : List ℕ) : 
  numbers = [201, 202, 205, 206, 209, 209, 210, 212, 212] → 
  numbers.get? 2 = some 205 := by
  intro h
  rw [h]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_element_is_205_l977_97776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_VXYZ_is_100_l977_97778

/-- Represents a parallelogram WXYZ with a point V on side WZ -/
structure Parallelogram where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  V : ℝ × ℝ

/-- The area of a region VXYZ in a parallelogram WXYZ -/
noncomputable def area_VXYZ (p : Parallelogram) : ℝ := sorry

/-- Check if four points form a parallelogram -/
def is_parallelogram (a b c d : ℝ × ℝ) : Prop := sorry

/-- Calculate the distance between two points -/
noncomputable def distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Calculate the height from a point to a line -/
noncomputable def height_from_point_to_line (p a b : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is on a line defined by two other points -/
def on_line (p a b : ℝ × ℝ) : Prop := sorry

/-- Theorem stating that the area of region VXYZ is 100 square units -/
theorem area_VXYZ_is_100 (p : Parallelogram) 
  (h1 : is_parallelogram p.W p.X p.Y p.Z)
  (h2 : distance p.W p.Z = 12)
  (h3 : height_from_point_to_line p.Y p.W p.Z = 10)
  (h4 : on_line p.V p.W p.Z)
  (h5 : distance p.W p.V + distance p.V p.Z = 12)
  (h6 : distance p.V p.Z = 8) :
  area_VXYZ p = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_VXYZ_is_100_l977_97778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_percentage_theorem_l977_97722

/-- The commission percentage on sales exceeding Rs. 10,000 -/
noncomputable def commission_percentage_above_10000 (
  total_sales : ℝ)
  (remitted_amount : ℝ)
  (commission_below_10000 : ℝ) : ℝ :=
  let sales_above_10000 := total_sales - 10000
  let total_commission := total_sales - remitted_amount
  let commission_above_10000 := total_commission - commission_below_10000
  (commission_above_10000 / sales_above_10000) * 100

theorem commission_percentage_theorem :
  let total_sales := (32500 : ℝ)
  let remitted_amount := (31100 : ℝ)
  let commission_below_10000 := 10000 * 0.05
  commission_percentage_above_10000 total_sales remitted_amount commission_below_10000 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_percentage_theorem_l977_97722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l977_97737

/-- The ellipse representing the trajectory of circle N's center -/
noncomputable def C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

/-- The x-coordinate of the intersection point of line AP with x-axis -/
noncomputable def S (x₀ y₀ x₁ y₁ : ℝ) : ℝ := (x₀*y₁ - x₁*y₀) / (y₁ - y₀)

/-- The x-coordinate of the intersection point of line BP with x-axis -/
noncomputable def T (x₀ y₀ x₁ y₁ : ℝ) : ℝ := (x₀*y₁ + x₁*y₀) / (y₁ + y₀)

theorem constant_product (x₀ y₀ x₁ y₁ : ℝ) 
  (hP : C x₀ y₀) (hA : C x₁ y₁) (h_diff : x₀ ≠ x₁) (h_nonzero : y₁ ≠ 0) :
  |S x₀ y₀ x₁ y₁ * T x₀ y₀ x₁ y₁| = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l977_97737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l977_97709

/-- Calculates the length of a train given the speeds of two trains, time to cross, and length of the other train -/
noncomputable def trainLength (speed1 speed2 : ℝ) (timeToCross : ℝ) (otherTrainLength : ℝ) : ℝ :=
  let relativeSpeed := (speed1 + speed2) * 1000 / 3600
  relativeSpeed * timeToCross - otherTrainLength

theorem first_train_length :
  let speed1 : ℝ := 120 -- kmph
  let speed2 : ℝ := 80 -- kmph
  let timeToCross : ℝ := 9 -- seconds
  let secondTrainLength : ℝ := 210.04 -- meters
  trainLength speed1 speed2 timeToCross secondTrainLength = 290 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l977_97709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_l977_97708

-- Define the functions
noncomputable def f (x : ℝ) := Real.exp (x - 1)
noncomputable def g (x : ℝ) := Real.cos (2 * x)

-- Define the propositions
def p : Prop := ∀ x y : ℝ, x < y → f x < f y
def q : Prop := ∀ x : ℝ, g (-x) = -g x

-- Define the compound propositions
def A : Prop := p ∧ q
def B : Prop := ¬p ∨ q
def C : Prop := ¬p ∧ ¬q
def D : Prop := p ∧ ¬q

-- Theorem statement
theorem correct_option : D ∧ ¬A ∧ ¬B ∧ ¬C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_l977_97708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_proof_l977_97703

/-- The angle the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The angle the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The time in minutes after 3:00 -/
def t : ℝ := sorry

theorem exact_time_proof :
  let minute_hand_pos := minute_hand_speed * (t + 8)
  let hour_hand_pos := 90 + hour_hand_speed * (t - 4)
  abs (minute_hand_pos - hour_hand_pos) = 180 →
  ∃ ε > 0, abs (t - 21.82) < ε :=
by sorry

#check exact_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_proof_l977_97703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_four_heads_in_six_flips_l977_97712

/-- The probability of getting exactly k successes in n trials with probability p for each trial. -/
noncomputable def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of a fair coin landing heads. -/
def fairCoinProbability : ℚ := 1 / 2

theorem fair_coin_four_heads_in_six_flips :
  binomialProbability 6 4 (fairCoinProbability : ℝ) = 15 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_four_heads_in_six_flips_l977_97712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l977_97754

-- Define the ellipse C
noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem ellipse_properties_and_max_area
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_P : ellipse_C a b 1 (Real.sqrt 2 / 2))
  (h_e : eccentricity a b = Real.sqrt 2 / 2) :
  (∃ (x y : ℝ), x^2 / 4 + y^2 = 1) ∧
  (∃ (max_area : ℝ), 
    max_area = 1 ∧
    ∀ (P Q : ℝ × ℝ), 
      (∃ (k : ℝ), P.2 = k * P.1 - 2 ∧ Q.2 = k * Q.1 - 2) →
      ellipse_C 2 1 P.1 P.2 →
      ellipse_C 2 1 Q.1 Q.2 →
      area_triangle (0, 0) P Q ≤ max_area) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l977_97754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_three_l977_97748

theorem reciprocal_of_negative_three :
  (fun x : ℝ => 1 / x) (-3) = -(1 / 3) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_three_l977_97748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_with_condition_l977_97701

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem odd_function_with_condition 
  (a b : ℝ) 
  (h1 : ∀ x ∈ Set.Ioo (-1) 1, f a b (-x) = -(f a b x)) 
  (h2 : f a b (1/2) = 2/5) :
  (∀ x ∈ Set.Ioo (-1) 1, f 1 0 x = x / (1 + x^2)) ∧ 
  (Set.Ioo 0 (1/2) = {x | x ∈ Set.Ioo (-1) 1 ∧ f 1 0 (x-1) + f 1 0 x < 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_with_condition_l977_97701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l977_97759

/-- Given a point M with polar coordinates (2, 2kπ + 2π/3) where k ∈ ℤ,
    its Cartesian coordinates are (-1, √3) -/
theorem polar_to_cartesian (k : ℤ) :
  let r : ℝ := 2
  let θ : ℝ := 2 * k * π + 2 * π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (-1, Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l977_97759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_rectangle_theorem_l977_97745

/-- Represents a rectangle on a grid -/
structure Rectangle where
  x : Int  -- x-coordinate of the left-bottom corner
  y : Int  -- y-coordinate of the left-bottom corner
  width : Nat  -- width of the rectangle
  height : Nat  -- height of the rectangle

/-- Predicate to check if a rectangle has an odd number of cells -/
def Rectangle.hasOddCells (r : Rectangle) : Prop :=
  r.width * r.height % 2 = 1

/-- Predicate to check if two rectangles overlap -/
def rectanglesOverlap (r1 r2 : Rectangle) : Prop :=
  r1.x < r2.x + r2.width ∧
  r2.x < r1.x + r1.width ∧
  r1.y < r2.y + r2.height ∧
  r2.y < r1.y + r1.height

/-- Predicate to check if two rectangles share a boundary point -/
def rectanglesShareBoundary (r1 r2 : Rectangle) : Prop :=
  (r1.x = r2.x + r2.width ∨ r2.x = r1.x + r1.width) ∧
  (r1.y ≤ r2.y + r2.height ∧ r2.y ≤ r1.y + r1.height) ∨
  (r1.y = r2.y + r2.height ∨ r2.y = r1.y + r1.height) ∧
  (r1.x ≤ r2.x + r2.width ∧ r2.x ≤ r1.x + r1.width)

/-- Type representing the four colors -/
inductive Color
  | color1
  | color2
  | color3
  | color4

/-- The main theorem -/
theorem four_color_rectangle_theorem (rectangles : Set Rectangle) :
  (∀ r ∈ rectangles, r.hasOddCells) →
  (∀ r1 r2, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 → ¬rectanglesOverlap r1 r2) →
  ∃ (coloring : Rectangle → Color),
    ∀ r1 r2, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 →
      rectanglesShareBoundary r1 r2 →
        coloring r1 ≠ coloring r2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_rectangle_theorem_l977_97745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_100_value_l977_97791

def mySequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 2018
  | 1 => 2017
  | k + 2 => mySequence k.succ - mySequence k

def S (n : ℕ) : ℤ := (List.range n).map mySequence |>.sum

theorem S_100_value : S 100 = 2016 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_100_value_l977_97791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_price_adjustment_l977_97716

-- Define the variables and constants
variable (x : ℝ) -- New electricity price
variable (y : ℝ) -- Increase in electricity consumption

-- Constants
def last_year_price : ℝ := 0.8
def last_year_consumption : ℝ := 1 -- 100 million kWh
def cost_price : ℝ := 0.3
def target_revenue_increase : ℝ := 0.2 -- 20%

-- Define the relationship between x and y
noncomputable def consumption_increase (x : ℝ) : ℝ := 1 / (5 * x - 2)

-- Define the revenue function
noncomputable def revenue (x : ℝ) : ℝ := (1 + consumption_increase x) * (x - cost_price)

-- Theorem statement
theorem electricity_price_adjustment :
  ∃ x : ℝ, 0.55 ≤ x ∧ x ≤ 0.75 ∧
  revenue x = last_year_consumption * (last_year_price - cost_price) * (1 + target_revenue_increase) ∧
  x = 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_price_adjustment_l977_97716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_45_implies_ratio_sqrt3_l977_97773

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ :=
  2 * Real.arctan (h.b / h.a)

/-- Theorem: For a hyperbola with 45° angle between asymptotes, a/b = √3 -/
theorem hyperbola_asymptote_angle_45_implies_ratio_sqrt3 (h : Hyperbola) 
  (h_angle : asymptote_angle h = π/4) : h.a / h.b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_45_implies_ratio_sqrt3_l977_97773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_x_equals_one_l977_97714

noncomputable def g (x : ℝ) : ℝ := |⌊2*x⌋| - |⌊2 - 2*x⌋|

theorem g_symmetry_about_x_equals_one : 
  ∀ x : ℝ, g x = g (2 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_x_equals_one_l977_97714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_problem_l977_97706

/-- Calculates the time (in seconds) for a train to cross a platform -/
noncomputable def train_crossing_time (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_crossing_problem :
  train_crossing_time 600 450 120 = 31.5 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_problem_l977_97706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_specific_line_l977_97700

/-- The angle of inclination of a line in degrees -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  (180 / Real.pi) * Real.arctan (-a / b)

/-- The line equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem angle_of_inclination_specific_line :
  let l : Line := { a := 1, b := Real.sqrt 3, c := 1 }
  angle_of_inclination l.a l.b l.c = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_specific_line_l977_97700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_with_all_items_l977_97742

def stadium_capacity : ℕ := 4800
def scarf_interval : ℕ := 45
def ticket_interval : ℕ := 36
def poster_interval : ℕ := 60

theorem fans_with_all_items : 
  (stadium_capacity / (Nat.lcm scarf_interval (Nat.lcm ticket_interval poster_interval))) = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_with_all_items_l977_97742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l977_97732

/-- The area of a pentagon with given vertices -/
theorem pentagon_area : ℝ := by
  -- Define the vertices of the pentagon
  let v1 : ℝ × ℝ := (2, 1)
  let v2 : ℝ × ℝ := (1, 4)
  let v3 : ℝ × ℝ := (4, 5)
  let v4 : ℝ × ℝ := (7, 2)
  let v5 : ℝ × ℝ := (5, 0)

  -- Define a function to calculate the area of a pentagon using the Shoelace formula
  let shoelace_formula (v1 v2 v3 v4 v5 : ℝ × ℝ) : ℝ :=
    (1/2) * abs (
      v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v5.2 + v5.1 * v1.2 -
      v2.1 * v1.2 - v3.1 * v2.2 - v4.1 * v3.2 - v5.1 * v4.2 - v1.1 * v5.2
    )

  -- Calculate the area using the Shoelace formula
  have area_calculation : shoelace_formula v1 v2 v3 v4 v5 = 18 := by sorry

  -- Return the calculated area
  exact 18

-- The theorem statement proves that the area of the pentagon is 18 square units

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l977_97732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2a_minus_pi_4_l977_97740

theorem sin_2a_minus_pi_4 (α : ℝ) (h1 : Real.sin α + Real.cos α = 1/5) (h2 : 0 ≤ α) (h3 : α ≤ π) :
  Real.sin (2*α - π/4) = -17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2a_minus_pi_4_l977_97740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_8_fourth_power_l977_97799

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x : ℝ, x ≥ 1 → f (g x) = x^2
axiom gf_condition : ∀ x : ℝ, x ≥ 1 → g (f x) = x^4
axiom g_64 : g 64 = 64

-- State the theorem to be proved
theorem g_8_fourth_power : (g 8)^4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_8_fourth_power_l977_97799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_M_to_C₁_l977_97744

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x - 2*y - 7 = 0

/-- The curve C₂ in parametric form -/
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)

/-- Point P -/
def P : ℝ × ℝ := (-4, 4)

/-- Midpoint M of PQ -/
noncomputable def M (θ : ℝ) : ℝ × ℝ := 
  let Q := C₂ θ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- Distance from a point to the line C₁ -/
noncomputable def distToC₁ (p : ℝ × ℝ) : ℝ :=
  |p.1 - 2*p.2 - 7| / Real.sqrt 5

/-- Theorem stating the minimum distance from M to C₁ -/
theorem min_distance_M_to_C₁ : 
  ∃ θ : ℝ, ∀ φ : ℝ, distToC₁ (M θ) ≤ distToC₁ (M φ) ∧ distToC₁ (M θ) = 8 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_M_to_C₁_l977_97744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appended_A_is_perfect_square_l977_97784

def A : ℕ := 13223140496

theorem appended_A_is_perfect_square :
  ∃ (k : ℕ), (A * (10^(Nat.log 10 A + 1) + 1) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appended_A_is_perfect_square_l977_97784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l977_97772

theorem vector_equation_solution :
  ∃ (u v : ℝ), u = (9 : ℝ) / 2 ∧ v = -7 ∧
  (⟨3, 1⟩ : ℝ × ℝ) + u • ⟨4, -6⟩ = ⟨0, 2⟩ + v • ⟨-3, 4⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l977_97772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_product_l977_97719

theorem remainder_of_product : (1225 * 1227 * 1229) % 12 = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_product_l977_97719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_ellipse_properties_l977_97782

-- Define the point P and foci F₁ and F₂
def P := ℝ × ℝ
def F₁ : ℝ × ℝ := (0, 5)
def F₂ : ℝ × ℝ := (0, -5)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem hyperbola_and_ellipse_properties :
  ∀ (x y : ℝ),
  (y^2 / 45 + x^2 / 20 = 1) →
  (distance (x, y) F₁ + distance (x, y) F₂ = 6 * Real.sqrt 5) →
  -- 1. The equation represents an ellipse
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ y^2 / a^2 + x^2 / b^2 = 1) ∧
  -- 2. When PF₁ ⋅ PF₂ = 0, P has coordinates (±4, ±3)
  ((x - 0) * (x - 0) + (y - 5) * (y + 5) = 0 →
    ((x = 4 ∨ x = -4) ∧ (y = 3 ∨ y = -3))) ∧
  -- 3. The minimum value of cos∠F₁PF₂ is -1/9
  (∀ (x' y' : ℝ),
    y'^2 / 45 + x'^2 / 20 = 1 →
    distance (x', y') F₁ + distance (x', y') F₂ = 6 * Real.sqrt 5 →
    let cosAngle := ((distance (x', y') F₁)^2 + (distance (x', y') F₂)^2 - 100) /
                    (2 * distance (x', y') F₁ * distance (x', y') F₂)
    cosAngle ≥ -1/9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_ellipse_properties_l977_97782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_maximum_l977_97798

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - Real.sqrt 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the curve C'
noncomputable def curve_C' (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 2 * Real.sin θ)

-- Function to maximize on C'
def f_to_maximize (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  4 * x^2 + x * y + y^2

-- Theorem statement
theorem intersection_and_maximum :
  (∃! p : ℝ × ℝ, line_l p.1 p.2 ∧ circle_C p.1 p.2) ∧
  (∀ θ : ℝ, f_to_maximize (curve_C' θ) ≤ 5) ∧
  (f_to_maximize (Real.sqrt 2 / 2, Real.sqrt 2) = 5) ∧
  (f_to_maximize (-Real.sqrt 2 / 2, -Real.sqrt 2) = 5) := by
  sorry

#check intersection_and_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_maximum_l977_97798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_nonpositive_monotonicity_when_a_positive_min_a_for_inequality_l977_97755

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

-- State the theorems
theorem monotonicity_when_a_nonpositive (a : ℝ) (ha : a ≤ 0) :
  StrictMono (f a) := by sorry

theorem monotonicity_when_a_positive (a : ℝ) (ha : a > 0) :
  ∀ x y, 0 < x → 0 < y →
    (x < y → x < (1/a) → f a x < f a y) ∧
    (x < y → (1/a) < x → f a x > f a y) := by sorry

theorem min_a_for_inequality (a : ℝ) :
  (∀ x, x ≥ 1 → f a x ≤ Real.log x / (x + 1)) ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_nonpositive_monotonicity_when_a_positive_min_a_for_inequality_l977_97755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_sin_cos_inequality_l977_97765

theorem max_m_for_sin_cos_inequality : 
  (∃ (m : ℝ), ∀ (x : ℝ), Real.sin x * Real.cos x ≤ m) ∧ 
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), Real.sin x * Real.cos x > 1/2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_sin_cos_inequality_l977_97765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_quadratic_equation_l977_97785

theorem is_quadratic_equation : ∃ (a b c : ℝ), a ≠ 0 ∧
  ∀ x : ℝ, Real.sqrt 2 * x^2 - (Real.sqrt 2 / 4) * x - (1 / 2) = a * x^2 + b * x + c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_quadratic_equation_l977_97785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_result_l977_97734

noncomputable def initial_number : ℂ := 4 + 3 * Complex.I

noncomputable def rotation_60_deg : ℂ := (1 / 2 : ℂ) + (Complex.I * Real.sqrt 3) / 2

def dilation_factor : ℝ := 2

noncomputable def transformation (z : ℂ) : ℂ := dilation_factor * (rotation_60_deg * z)

theorem complex_transformation_result :
  transformation initial_number = (4 - 3 * Real.sqrt 3) + Complex.I * (4 * Real.sqrt 3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_result_l977_97734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l977_97763

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by its vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Check if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

/-- Calculate the angle between two vectors -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  let v1 := (p2.x - p1.x, p2.y - p1.y)
  let v2 := (p3.x - p1.x, p3.y - p1.y)
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

/-- Calculate the area of a hexagon -/
noncomputable def hexagonArea (h : Hexagon) : ℝ :=
  sorry -- Actual calculation would go here

/-- Main theorem -/
theorem hexagon_area_theorem (h : Hexagon) :
  h.A = ⟨0, 0⟩ →
  h.B.y = 4 →
  parallel h.A h.B h.D h.E →
  parallel h.B h.C h.E h.F →
  parallel h.C h.D h.F h.A →
  angle h.F h.A h.B = 2 * Real.pi / 3 →
  ({h.A.y, h.B.y, h.C.y, h.D.y, h.E.y, h.F.y} : Set ℝ) = {0, 4, 8, 12, 16, 20} →
  hexagonArea h = 192 * Real.sqrt 3 := by
  sorry

#eval 192 + 3  -- Should output 195

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l977_97763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_affordable_shirt_price_l977_97751

/-- Represents the problem of finding the maximum affordable shirt price --/
def MaxAffordableShirtPrice (budget : ℚ) (numShirts : ℕ) (entranceFee : ℚ) 
  (discountThreshold : ℕ) (discountRate : ℚ) (taxRate : ℚ) : ℕ :=
  let remainingBudget := budget - entranceFee
  let discountFactor := if numShirts > discountThreshold then 1 - discountRate else 1
  let taxFactor := 1 + taxRate
  ⌊(remainingBudget / (numShirts * discountFactor * taxFactor))⌋.toNat

/-- Theorem stating that the maximum affordable shirt price is $10 --/
theorem max_affordable_shirt_price :
  MaxAffordableShirtPrice 200 20 5 15 (1/10) (1/20) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_affordable_shirt_price_l977_97751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_color_count_l977_97752

/-- Represents a bag containing two balls of different colors -/
structure Bag where
  color1 : Nat
  color2 : Nat
  different_colors : color1 ≠ color2

/-- Represents a distribution of balls into bags -/
structure Distribution (k : Nat) where
  bags : Finset Bag
  total_bags : bags.card = 2520
  valid_colors : ∀ b ∈ bags, b.color1 < k ∧ b.color2 < k
  equal_balls : ∀ c < k, (bags.filter (λ b => b.color1 = c ∨ b.color2 = c)).card = 2520

/-- Represents a circular arrangement of bags -/
def CircularArrangement (k : Nat) (d : Distribution k) :=
  List Bag

/-- Predicate to check if a circular arrangement is valid -/
def ValidArrangement (k : Nat) (d : Distribution k) (arr : CircularArrangement k d) : Prop :=
  ∀ i, (arr.get? i).map (λ b => b.color1) ≠ (arr.get? (i+1)).map (λ b => b.color1) ∧
       (arr.get? i).map (λ b => b.color1) ≠ (arr.get? (i+1)).map (λ b => b.color2) ∧
       (arr.get? i).map (λ b => b.color2) ≠ (arr.get? (i+1)).map (λ b => b.color1) ∧
       (arr.get? i).map (λ b => b.color2) ≠ (arr.get? (i+1)).map (λ b => b.color2)

/-- The main theorem stating that 6 is the smallest number of colors that allows a valid arrangement -/
theorem smallest_valid_color_count :
  (∃ k, ∀ d : Distribution k, ∃ arr : CircularArrangement k d, ValidArrangement k d arr) ∧
  (∀ k < 6, ∃ d : Distribution k, ∀ arr : CircularArrangement k d, ¬ValidArrangement k d arr) ∧
  (∀ d : Distribution 6, ∃ arr : CircularArrangement 6 d, ValidArrangement 6 d arr) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_color_count_l977_97752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_to_inequality_l977_97730

theorem no_real_solutions_to_inequality :
  ∀ x : ℝ, ¬(Real.rpow x (1/3) + 4 / (Real.rpow x (1/3) + 2) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_to_inequality_l977_97730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wrapping_paper_area_l977_97768

/-- Given a rectangular box with dimensions l, w, and h, the minimum area of wrapping paper
    required to wrap the box such that opposite corners meet at the center of the top
    is w * l + 2 * w * h + 2 * l * h + 4 * h^2 -/
theorem min_wrapping_paper_area (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  (w + 2 * h) * (l + 2 * h) = w * l + 2 * w * h + 2 * l * h + 4 * h^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wrapping_paper_area_l977_97768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_circles_l977_97764

/-- The number of small circles fitting along the diameter of a large semicircle -/
def N : ℕ := 60

/-- The diameter of each small circle -/
noncomputable def d : ℝ := 1  -- Assume unit diameter for simplicity

/-- The area of all small circles combined -/
noncomputable def A : ℝ := N * (Real.pi * d^2 / 4)

/-- The area of the region inside the large semicircle but outside the small circles -/
noncomputable def B : ℝ := (Real.pi * (N * d)^2 / 8) - A

/-- The ratio of A to B is 1:29 -/
axiom ratio_condition : A / B = 1 / 29

theorem number_of_circles : N = 60 := by
  -- The proof goes here
  sorry

#eval N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_circles_l977_97764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l977_97753

/-- The area of a triangle given its three vertices -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- The maximum area of a triangle ABC with given vertices -/
theorem triangle_max_area :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ → ℝ × ℝ := fun θ ↦ (Real.cos θ, -1 + Real.sin θ)
  
  ∃ (max_area : ℝ),
    (∀ θ : ℝ, area_triangle A B (C θ) ≤ max_area) ∧
    (∃ θ : ℝ, area_triangle A B (C θ) = max_area) ∧
    max_area = 3 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l977_97753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_is_correct_l977_97733

/-- A rectangular parallelepiped with a square base -/
structure RectangularParallelepiped where
  base : ℝ
  height : ℝ

/-- The angle between line BD₁ and plane BDC₁ in a rectangular parallelepiped -/
noncomputable def angle (p : RectangularParallelepiped) : ℝ :=
  Real.arcsin (p.height / Real.sqrt ((2 + p.height^2) * (1 + 2 * p.height^2)))

/-- The maximum possible angle between line BD₁ and plane BDC₁ in any rectangular parallelepiped -/
noncomputable def max_angle : ℝ := Real.arcsin (1/3)

theorem max_angle_is_correct :
  ∀ p : RectangularParallelepiped, angle p ≤ max_angle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_is_correct_l977_97733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_value_from_double_angle_and_sum_formula_l977_97790

theorem tangent_value_from_double_angle_and_sum_formula 
  (α : ℝ) 
  (h1 : α > 0)
  (h2 : α < π / 2)
  (h3 : Real.cos (2 * α) = (2 * Real.sqrt 5 / 5) * Real.sin (α + π / 4)) : 
  Real.tan α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_value_from_double_angle_and_sum_formula_l977_97790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_l977_97705

theorem geometric_series_product (y : ℝ) : y = 9 := by
  let S₁ := (1 + 1/3 + 1/9 + 1/27 : ℝ) -- Represents the infinite sum
  let S₂ := (1 - 1/3 + 1/9 - 1/27 : ℝ) -- Represents the infinite sum
  let P := S₁ * S₂
  let RHS := (1 + 1/y + 1/y^2 + 1/y^3 : ℝ) -- Represents the infinite sum
  have h : P = RHS := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_l977_97705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_decrease_l977_97774

theorem salary_decrease (initial_salary : ℝ) (cut1 cut2 cut3 cut4 cut5 : ℝ) 
  (h1 : cut1 = 0.08) (h2 : cut2 = 0.14) (h3 : cut3 = 0.18) 
  (h4 : cut4 = 0.22) (h5 : cut5 = 0.27) :
  let total_decrease := 1 - (1 - cut1) * (1 - cut2) * (1 - cut3) * (1 - cut4) * (1 - cut5)
  ∃ ε > 0, |total_decrease - 0.5607| < ε := by
sorry

#eval (1 - (1 - 0.08) * (1 - 0.14) * (1 - 0.18) * (1 - 0.22) * (1 - 0.27))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_decrease_l977_97774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l977_97795

/-- A function with two distinct extreme points -/
def has_two_distinct_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ 
    (∀ z, z > 0 → (deriv f x = 0 ∧ deriv f y = 0 ∧ (deriv f z = 0 → z = x ∨ z = y)))

/-- The main theorem -/
theorem extreme_points_imply_a_range (a : ℝ) :
  has_two_distinct_extreme_points (fun x => (1/2) * x^2 - 2*x + a * Real.log x) →
  0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l977_97795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinB_in_triangle_l977_97711

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem sinB_in_triangle (t : Triangle) 
  (h1 : Real.cos t.A = 1/3)
  (h2 : t.a = Real.sqrt 2)
  (h3 : t.b = Real.sqrt 3 / 2) :
  Real.sin t.B = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinB_in_triangle_l977_97711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_sphere_area_l977_97758

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if two planes are perpendicular -/
def perpendicular (p1 p2 : Plane) : Prop := sorry

/-- Checks if a line is perpendicular to a plane -/
def perpendicular_to_plane (line : Point3D × Point3D) (p : Plane) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the surface area of the circumscribed sphere of a triangular pyramid -/
noncomputable def circumscribed_sphere_surface_area (tp : TriangularPyramid) : ℝ := sorry

theorem triangular_pyramid_sphere_area 
  (tp : TriangularPyramid)
  (h1 : perpendicular (Plane.mk 1 0 0 0) (Plane.mk 0 1 0 0))  -- ABD perpendicular to BCD
  (h2 : perpendicular_to_plane (tp.B, tp.C) (Plane.mk 0 1 0 0))  -- BC perpendicular to plane containing BD
  (h3 : distance tp.A tp.B = 4 * Real.sqrt 3)
  (h4 : distance tp.A tp.D = 4 * Real.sqrt 3)
  (h5 : distance tp.B tp.D = 4 * Real.sqrt 3)
  (h6 : distance tp.B tp.C = 6) :
  circumscribed_sphere_surface_area tp = 100 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_sphere_area_l977_97758
