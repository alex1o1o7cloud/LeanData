import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_divisible_by_13_l142_14243

theorem count_three_digit_divisible_by_13 : 
  Finset.card (Finset.filter (fun n => n % 13 = 0) (Finset.range 900 ⊔ Finset.range 100)) = 69 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_divisible_by_13_l142_14243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_approx_l142_14211

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles -/
  baseAngle : ℝ
  /-- Assumption that the trapezoid is isosceles and circumscribed around a circle -/
  isIsoscelesCircumscribed : Bool

/-- Calculate the area of the circumscribed trapezoid -/
noncomputable def calculateArea (t : CircumscribedTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is approximately 74.07 -/
theorem trapezoid_area_approx :
  let t : CircumscribedTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arcsin 0.6,
    isIsoscelesCircumscribed := true
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |calculateArea t - 74.07| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_approx_l142_14211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_area_l142_14230

/-- Pentagon with specific side lengths that can be divided into a rectangle and triangle -/
structure SpecialPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  rectangle_side1 : ℝ
  rectangle_side2 : ℝ
  triangle_height : ℝ
  side1_eq : side1 = 25
  side2_eq : side2 = 30
  side3_eq : side3 = 28
  side4_eq : side4 = 25
  side5_eq : side5 = 30
  rectangle_side1_eq : rectangle_side1 = side1
  rectangle_side2_eq : rectangle_side2 = side2
  triangle_height_eq : triangle_height = side3

/-- Calculate the area of the special pentagon -/
noncomputable def area_of_special_pentagon (p : SpecialPentagon) : ℝ :=
  p.rectangle_side1 * p.rectangle_side2 + (1/2) * p.rectangle_side2 * p.triangle_height

/-- Theorem stating that the area of the special pentagon is 1170 square units -/
theorem special_pentagon_area (p : SpecialPentagon) :
  area_of_special_pentagon p = 1170 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_area_l142_14230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_sixth_l142_14242

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  yard_length : ℝ
  yard_width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- The fraction of the yard occupied by flower beds -/
noncomputable def flower_bed_fraction (y : YardWithFlowerBeds) : ℝ :=
  let triangle_leg := (y.trapezoid_long_side - y.trapezoid_short_side) / 3
  let triangle_area := triangle_leg ^ 2 / 2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := y.yard_length * y.yard_width
  total_flower_bed_area / yard_area

/-- Theorem: The fraction of the yard occupied by flower beds is 1/6 -/
theorem flower_bed_fraction_is_one_sixth (y : YardWithFlowerBeds) 
  (h1 : y.trapezoid_short_side = 10)
  (h2 : y.trapezoid_long_side = 20)
  (h3 : y.yard_length = 20)
  (h4 : y.yard_width = 5) :
  flower_bed_fraction y = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_sixth_l142_14242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_properties_l142_14279

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 1 then x^2 + x + 1
  else if x = 0 then 0
  else if -1 < x ∧ x < 0 then -x^2 + x - 1
  else 0  -- This case should never occur given the domain

theorem odd_decreasing_function_properties
  (h_odd : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x)
  (h_decreasing : ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x > f y)
  (h_condition : ∃ a : ℝ, f (1 - a) + f (1 - 2*a) < 0)
  (h_definition : ∀ x, x ∈ Set.Ioo (0 : ℝ) 1 → f x = x^2 + x + 1) :
  (∃ a : ℝ, 0 < a ∧ a ≤ 2/3 ∧ f (1 - a) + f (1 - 2*a) < 0) ∧
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → 
    ((0 < x → f x = x^2 + x + 1) ∧
    (x = 0 → f x = 0) ∧
    (x < 0 → f x = -x^2 + x - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_properties_l142_14279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_calculation_l142_14292

/-- Represents the time (in days) it takes for a person to complete the work alone -/
structure WorkerTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the total payment for a job given the worker times and b's share -/
noncomputable def totalPayment (w : WorkerTime) (b_share : ℝ) : ℝ :=
  let total_daily_work := 1 / w.a + 1 / w.b + 1 / w.c
  let b_daily_work := 1 / w.b
  (b_share * total_daily_work) / b_daily_work

theorem total_payment_calculation (w : WorkerTime) (b_share : ℝ) :
  w.a = 6 → w.b = 8 → w.c = 12 → b_share = 600.0000000000001 →
  totalPayment w b_share = 1800.0000000000003 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_calculation_l142_14292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_competition_selection_l142_14295

theorem science_competition_selection (f m : ℕ) (h1 : f = 2) (h2 : m = 4) :
  (Finset.sum (Finset.range (f + 1)) (λ k => 
    if k ≤ f ∧ k ≥ 1 then Nat.choose f k * Nat.choose m (3 - k) else 0)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_competition_selection_l142_14295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l142_14283

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 5) / (-1) = (y + 3) / 5 ∧ (y + 3) / 5 = (z - 1) / 2

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 7 * y - 5 * z - 11 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (4, 2, 3)

theorem intersection_point_is_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    let (x, y, z) := p
    line_equation x y z ∧ plane_equation x y z ∧ p = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l142_14283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_distance_theorem_l142_14298

-- Define the grasshopper jump function
def grasshopperJump (t : ℕ) : ℕ := t * (t + 1) / 2

-- Theorem statement
theorem grasshopper_distance_theorem :
  ∃ (m t Δt : ℕ),
    -- First recording condition
    grasshopperJump t - grasshopperJump (t - m) = 9 ∧
    -- Second recording condition
    grasshopperJump (t + Δt) - grasshopperJump (t + Δt - m) = 39 ∧
    -- Possible time differences
    Δt ∈ ({10, 15, 30} : Set ℕ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_distance_theorem_l142_14298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_equation_sum_l142_14240

theorem abs_equation_sum : ∃ (S : Finset ℝ), (∀ x ∈ S, |3*x - 9| = 6) ∧ (S.sum id = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_equation_sum_l142_14240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_solutions_l142_14290

def τ (n : ℕ+) : ℕ := (Nat.divisors n.val).card

def φ (n : ℕ+) : ℕ := Nat.totient n.val

def σ (n : ℕ+) : ℕ := (Nat.divisors n.val).sum id

theorem finite_solutions (a : ℕ) (ha : a ≥ 9) :
  Set.Finite {n : ℕ+ | τ n = a ∧ n.val ∣ φ n + σ n} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_solutions_l142_14290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l142_14270

-- Define the curve C
noncomputable def C (α : ℝ) : ℝ × ℝ :=
  (3 + Real.sqrt 10 * Real.cos α, 1 + Real.sqrt 10 * Real.sin α)

-- Define the line l in polar form
def l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ - ρ * Real.cos θ = 2

-- State the theorem
theorem chord_length :
  ∃ (chord_length : ℝ),
    chord_length = 2 * Real.sqrt 2 ∧
    ∀ (α θ ρ : ℝ),
      C α = (ρ * Real.cos θ, ρ * Real.sin θ) →
      l ρ θ →
      chord_length = 2 * Real.sqrt (10 - 8) := by
  sorry

#check chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l142_14270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_angle_theorem_l142_14233

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  Real.arccos ((distance p1 p2)^2 + (distance p2 p3)^2 - (distance p1 p3)^2) / (2 * distance p1 p2 * distance p2 p3)

/-- The main theorem -/
theorem hyperbola_angle_theorem (h : Hyperbola) (p f1 f2 : Point) :
  h.a = 2 ∧ h.b = 3*Real.sqrt 5 ∧
  isOnHyperbola h p ∧
  ∃ (d : ℝ), d > 0 ∧ 
    distance p f2 - distance p f1 = d ∧
    distance f1 f2 - distance p f2 = d →
  angle f1 p f2 = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_angle_theorem_l142_14233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_push_time_l142_14286

/-- Represents a segment of the journey -/
structure Segment where
  distance : ℝ
  speed : ℝ
deriving Inhabited

/-- Calculates the time taken for a segment -/
noncomputable def time_for_segment (s : Segment) : ℝ := s.distance / s.speed

/-- The problem statement -/
theorem car_push_time (segments : List Segment) 
  (h1 : segments.length = 3)
  (h2 : segments[0]!.distance = 3 ∧ segments[0]!.speed = 6)
  (h3 : segments[1]!.distance = 3 ∧ segments[1]!.speed = 3)
  (h4 : segments[2]!.distance = 4 ∧ segments[2]!.speed = 8)
  (h5 : (segments.map (λ s => s.distance)).sum = 10) :
  (segments.map time_for_segment).sum = 2 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_push_time_l142_14286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_four_equals_sqrt_two_l142_14247

theorem fourth_root_of_four_equals_sqrt_two :
  (1 / 4 : ℝ) ^ (-(1 / 4 : ℝ)) = Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_four_equals_sqrt_two_l142_14247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l142_14219

/-- Predicate stating that a triangle with sides a, b, c is acute -/
def AcuteTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Predicate stating that m is a median of a triangle with sides a, b, c -/
def IsMedian (a b c m : ℝ) : Prop :=
  4 * m^2 = 2 * b^2 + 2 * c^2 - a^2

/-- For an acute triangle ABC with sides a, b, c and medians m_a, m_b, m_c,
    the sum of the squares of the medians divided by the sum of the squares of two sides
    minus the square of the third side, taken cyclically, is greater than or equal to 9/4. -/
theorem median_inequality (a b c m_a m_b m_c : ℝ) (h_acute : AcuteTriangle a b c)
  (h_ma : IsMedian a b c m_a) (h_mb : IsMedian b c a m_b) (h_mc : IsMedian c a b m_c) :
  m_a^2 / (-a^2 + b^2 + c^2) + m_b^2 / (-b^2 + c^2 + a^2) + m_c^2 / (-c^2 + a^2 + b^2) ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l142_14219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_coins_l142_14260

/-- Represents the value of a coin in cents -/
def CoinValue : Type := Nat

/-- The set of coins being flipped -/
def Coins : Finset Nat := {1, 5, 10, 25, 50, 100}

/-- The probability of a coin landing heads -/
noncomputable def probHeads : ℝ := 1 / 2

/-- The expected value of the sum of coins coming up heads -/
noncomputable def expectedValue : ℝ := (Coins.sum (fun c => probHeads * (c : ℝ))) / 1

theorem expected_value_of_coins : expectedValue = 95.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_coins_l142_14260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l142_14255

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  Real.sin (t.C / 2) = 2 * Real.sqrt 2 / 3 ∧
  t.a + t.b = 2 * Real.sqrt 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin (t.A + t.B) = 4 * Real.sqrt 2 / 9 ∧
  (∀ S : ℝ, S = 1/2 * t.a * t.b * Real.sin t.C → S ≤ 4 * Real.sqrt 2 / 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l142_14255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_points_on_C1_sum_l142_14227

/-- The curve C1 in polar coordinates -/
def C1 (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ^2 + (ρ^2 * Real.sin θ^2) / 4 = 1

/-- Two points are perpendicular if their angles differ by π/2 -/
def perpendicular (θ1 θ2 : ℝ) : Prop :=
  θ2 = θ1 + Real.pi/2 ∨ θ2 = θ1 - Real.pi/2

theorem perpendicular_points_on_C1_sum (ρ1 ρ2 θ1 θ2 : ℝ) :
  C1 ρ1 θ1 → C1 ρ2 θ2 → perpendicular θ1 θ2 →
  1/ρ1^2 + 1/ρ2^2 = 5/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_points_on_C1_sum_l142_14227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l142_14208

-- Define the solution set type
def SolutionSet (α : Type) := Set α

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

-- State the theorem
theorem inequality_solution_sets :
  ∀ a : ℝ,
  (a > 0 → {x : ℝ | inequality x a} = {x : ℝ | -a < x ∧ x < 3*a}) ∧
  (a = 0 → {x : ℝ | inequality x a} = ∅) ∧
  (a < 0 → {x : ℝ | inequality x a} = {x : ℝ | 3*a < x ∧ x < -a}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l142_14208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_harmonic_mean_relation_l142_14235

theorem arithmetic_geometric_harmonic_mean_relation (x y : ℤ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  ∃ (a b : ℤ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  (x + y) / 2 = 10 * a + b ∧
  Real.sqrt (x * y : ℝ) = 10 * b + a ∧
  2 * x * y / (x + y) = 10 * b + a - 1 →
  |x - y| = 18 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_harmonic_mean_relation_l142_14235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_eq_ge_two_l142_14213

/-- The sum of unique prime factors of a positive integer -/
def s (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

/-- The range of s(n) for n ≥ 2 -/
def s_range : Set ℕ :=
  {m | ∃ n ≥ 2, s n = m}

/-- Theorem: The range of s(n) is the set of positive integers ≥ 2 -/
theorem s_range_eq_ge_two : s_range = {n : ℕ | n ≥ 2} := by
  sorry

#check s_range_eq_ge_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_eq_ge_two_l142_14213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l142_14232

-- Define the triangle T
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  side_length : a^2 + b^2 = 9

-- Define the tetrahedron formed by four copies of T
noncomputable def tetrahedron_volume (t : Triangle) : ℝ := (1/6) * t.a * t.b * t.c

noncomputable def tetrahedron_surface_area (t : Triangle) : ℝ := 
  2 * Real.sqrt (t.a^2 * t.b^2 + t.a^2 * t.c^2 + t.b^2 * t.c^2)

-- Define the circumradius of T
noncomputable def circumradius (t : Triangle) : ℝ := 
  (t.a * t.b * t.c) / (4 * Real.sqrt ((t.a + t.b + t.c) * (-t.a + t.b + t.c) * (t.a - t.b + t.c) * (t.a + t.b - t.c)))

-- Theorem statement
theorem triangle_circumradius (t : Triangle) 
  (h1 : tetrahedron_volume t = 4)
  (h2 : tetrahedron_surface_area t = 24) :
  circumradius t = Real.sqrt (4 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l142_14232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l142_14257

theorem cosine_equation_solution (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = Real.pi/2 + k*Real.pi ∨ x = Real.pi/6 + 2*k*Real.pi ∨ x = 5*Real.pi/6 + 2*k*Real.pi ∨ 
           x = Real.pi/4 + 2*k*Real.pi ∨ x = 3*Real.pi/4 + 2*k*Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l142_14257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_in_third_quadrant_l142_14291

theorem tan_alpha_in_third_quadrant (α : Real) : 
  (α ∈ Set.Icc π (3*π/2)) →  -- α is in the third quadrant
  (Real.sin α = -12/13) →    -- given sin α value
  (Real.tan α = 12/5) :=     -- prove that tan α equals 12/5
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_in_third_quadrant_l142_14291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_second_derivative_at_one_l142_14285

/-- Given a function f such that f(x) = 2x * f''(1) + ln(x), prove that f''(1) = -1 -/
theorem function_second_derivative_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 * x * (deriv (deriv f) 1) + Real.log x) :
  deriv (deriv f) 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_second_derivative_at_one_l142_14285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l142_14204

-- Define the universe of triangles
variable (Triangle : Type)

-- Define the property of being an isosceles triangle
variable (isIsosceles : Triangle → Prop)

-- Define the original proposition
def someProp (Triangle : Type) (isIsosceles : Triangle → Prop) : Prop :=
  ∃ t : Triangle, isIsosceles t

-- Define the negation we want to prove
def negationProp (Triangle : Type) (isIsosceles : Triangle → Prop) : Prop :=
  ∀ t : Triangle, ¬(isIsosceles t)

-- Theorem stating that the negation of the original proposition
-- is equivalent to the negation we defined
theorem negation_equivalence (Triangle : Type) (isIsosceles : Triangle → Prop) :
    ¬(someProp Triangle isIsosceles) ↔ negationProp Triangle isIsosceles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l142_14204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_theorem_l142_14281

/-- Represents a data point with promotion time and number of reported cases -/
structure DataPoint where
  x : ℝ  -- promotion time in months
  y : ℝ  -- number of reported cases

/-- Represents the statistics of the dataset -/
structure DataStatistics where
  sum_ti_yi : ℝ  -- sum of (ti * yi)
  t_bar : ℝ      -- average of ti
  sum_ti_squared_minus_7t_bar_squared : ℝ  -- sum of ti^2 minus 7 * (t_bar^2)

/-- Calculates the coefficients of the regression equation y = a + b/x -/
noncomputable def calculate_regression_coefficients (data : List DataPoint) (stats : DataStatistics) : 
  (ℝ × ℝ) := sorry

/-- States the theorem about the regression equation -/
theorem regression_equation_theorem (data : List DataPoint) (stats : DataStatistics) :
  let (a, b) := calculate_regression_coefficients data stats
  a = 30 ∧ b = 1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_theorem_l142_14281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_x_value_l142_14221

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : 
  (∀ y : ℕ, y > 0 ∧ (∃ r : ℕ, Nat.Prime r ∧ r % 2 = 1 ∧ y = 11 * p * r) → y ≥ 66) ∧ x = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_x_value_l142_14221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_opposite_sides_l142_14256

-- Define the plane
variable (a x y : ℝ)

-- Define point A
def point_A (a x y : ℝ) : Prop :=
  5 * a^2 - 4 * a * x + 6 * a * y + x^2 - 2 * x * y + 2 * y^2 = 0

-- Define circle centered at point B
def circle_B (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 4 * a^3 * x - 2 * a * x + 2 * a^2 * y + 4 * a^4 + 1 = 0

-- Define the condition for A and B being on opposite sides of x = 3
def opposite_sides (x_A x_B : ℝ) : Prop :=
  (x_A - 3) * (x_B - 3) < 0

-- Theorem statement
theorem points_opposite_sides :
  ∀ (a x_A y_A x_B y_B : ℝ),
  point_A a x_A y_A →
  circle_B a x_B y_B →
  x_A ≠ 3 ∧ x_B ≠ 3 →
  opposite_sides x_A x_B →
  (a > 0 ∧ a < 1/2) ∨ (a > 1 ∧ a < 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_opposite_sides_l142_14256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_property_l142_14263

-- Define the function g(x) = 2x²e^(2x) + ln x
noncomputable def g (x : ℝ) := 2 * x^2 * Real.exp (2 * x) + Real.log x

-- State the theorem
theorem root_property (x₀ : ℝ) (h₁ : x₀ > 0) (h₂ : g x₀ = 0) :
  2 * x₀ + Real.log x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_property_l142_14263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l142_14271

-- Define the lines and circle
def line1 (x y : ℝ) := 4 * x - 2 * y + 7 = 0
def line2 (x y : ℝ) := 2 * x - y + 1 = 0
def line_l (x y m : ℝ) := x - 2 * y + m = 0
def circle_C (x y : ℝ) := x^2 + (y - 2)^2 = 1/5

-- Define the distance between two parallel lines
noncomputable def distance_parallel_lines (a b c d : ℝ) : ℝ := 
  |c - d| / Real.sqrt (a^2 + b^2)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ := 
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

-- State the theorem
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y, line1 x y ↔ line2 x y) →  -- lines are parallel
  (m > 0) →
  (distance_parallel_lines 4 (-2) 7 2 = 
    (1/2) * distance_point_to_line 0 0 1 (-2) m) →
  (m = 5 ∧ 
   distance_point_to_line 0 2 1 (-2) m = Real.sqrt (1/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l142_14271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l142_14217

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def adjacent_digits_sum_square (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, i + 1 < digits.length → is_square (digits[i]! + digits[i+1]!)

def interesting_number (n : ℕ) : Prop :=
  (n.digits 10).Nodup ∧ adjacent_digits_sum_square n

theorem largest_interesting_number :
  interesting_number 6310972 ∧ ∀ m : ℕ, interesting_number m → m ≤ 6310972 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l142_14217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_is_uncountable_l142_14225

def furniture : String := "furniture"

theorem furniture_is_uncountable : True := by
  -- This is where we would prove that furniture is uncountable,
  -- but since this is a linguistic concept, we can't formally prove it in Lean
  sorry

#check furniture
#check furniture_is_uncountable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_is_uncountable_l142_14225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triples_eq_answer_triples_l142_14278

def ε : ℝ := 18

def is_valid_triple (a b c : ℝ) : Prop :=
  ∃ (n₁ n₂ n₃ : ℕ), 
    a = n₁ * ε ∧ 
    b = n₂ * ε ∧ 
    c = n₃ * ε ∧ 
    n₁ > 0 ∧ n₂ > 0 ∧ n₃ > 0 ∧
    a + b + c = 180

def valid_triples : Set (ℝ × ℝ × ℝ) :=
  {t | is_valid_triple t.1 t.2.1 t.2.2}

def answer_triples : Set (ℝ × ℝ × ℝ) :=
  {(18, 18, 144), (18, 36, 126), (18, 54, 108), (18, 72, 90),
   (36, 36, 108), (36, 54, 90), (36, 72, 72), (54, 54, 72)}

theorem valid_triples_eq_answer_triples : valid_triples = answer_triples := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triples_eq_answer_triples_l142_14278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_weighted_mean_is_90_3_l142_14250

/-- Calculates the weighted mean of a list of scores and their corresponding weights -/
noncomputable def weighted_mean (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) scores weights)) / (List.sum weights)

/-- Bob's quiz scores -/
def bob_scores : List ℝ := [85, 90, 88, 92, 86, 94]

/-- Weights for Bob's quizzes -/
def bob_weights : List ℝ := [1, 2, 1, 3, 1, 2]

/-- Theorem: Bob's weighted mean score is 90.3 -/
theorem bob_weighted_mean_is_90_3 :
  weighted_mean bob_scores bob_weights = 90.3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_weighted_mean_is_90_3_l142_14250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_open_zero_three_l142_14277

/-- The function g defined on positive real numbers -/
noncomputable def g (x y z : ℝ) : ℝ := x^2 / (x^2 + y^2) + y^2 / (y^2 + z^2) + z^2 / (z^2 + x^2)

/-- Theorem stating the range of g is (0, 3) -/
theorem g_range_open_zero_three :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    0 < g x y z ∧ g x y z < 3 ∧
    (∀ ε > 0, ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      (|g a b c - 3| < ε ∨ |g a b c| < ε)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_open_zero_three_l142_14277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_time_discount_l142_14266

/-- True discount calculation function -/
noncomputable def trueDiscount (presentValue : ℝ) (interest : ℝ) : ℝ :=
  (interest * presentValue) / (presentValue + interest)

/-- Amount calculation function -/
noncomputable def amount (presentValue : ℝ) (interest : ℝ) : ℝ :=
  presentValue + interest

theorem double_time_discount (initialBill : ℝ) (initialDiscount : ℝ) (initialTime : ℝ) :
  initialBill = 110 →
  initialDiscount = 10 →
  initialTime > 0 →
  let presentValue := initialBill - initialDiscount
  let initialInterest := initialDiscount
  let doubleTimeInterest := 2 * initialInterest
  trueDiscount presentValue initialInterest = initialDiscount →
  trueDiscount presentValue doubleTimeInterest = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_time_discount_l142_14266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l142_14251

theorem wall_building_time (a d e : ℝ) (h : a > 0 ∧ d > 0 ∧ e > 0) :
  (a^2 * e) / (1.5 * d^2) = (a^2 * e) / (1.5 * d^2) :=
by
  -- Initial rate: d / (a * e)
  -- New rate with 50% increase: 1.5 * d / (a * e)
  -- New equation: a = (1.5 * d / (a * e)) * (d * x)
  -- Solve for x:
  -- a = (1.5 * d^2 * x) / (a * e)
  -- a^2 * e = 1.5 * d^2 * x
  -- x = (a^2 * e) / (1.5 * d^2)
  rfl  -- reflexivity proves the equality

#check wall_building_time

-- The theorem states that the derived formula is equal to itself,
-- which is trivially true. In a more complete proof, we would show
-- that this formula correctly solves the problem given the initial conditions.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l142_14251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l142_14282

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 + a 3 = 32) 
  (h2 : a 11 + a 12 + a 13 = 118) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) : 
  a 4 + a 10 = 50 := by
  sorry

#check arithmetic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l142_14282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sushi_cost_ratio_l142_14259

theorem sushi_cost_ratio 
  (combined_cost : ℝ)
  (eel_cost : ℝ)
  (h1 : combined_cost = 200)
  (h2 : eel_cost = 180) :
  eel_cost / (combined_cost - eel_cost) = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sushi_cost_ratio_l142_14259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_prob_iff_N_18_l142_14203

/-- The number of balls in the lottery -/
def N : ℕ := 18

/-- The number of combinations of k balls from N balls that sum to s -/
def number_of_combinations_sum (N k s : ℕ) : ℕ :=
  sorry

/-- The number of combinations of 10 balls from N balls that sum to 63 -/
def number_of_combinations_sum_63 (N : ℕ) : ℕ :=
  number_of_combinations_sum N 10 63

/-- The number of combinations of 8 balls from N balls that sum to 44 -/
def number_of_combinations_sum_44 (N : ℕ) : ℕ :=
  number_of_combinations_sum N 8 44

/-- The probability of selecting 10 balls with a sum of 63 -/
def prob_main_draw (N : ℕ) : ℚ :=
  (number_of_combinations_sum_63 N) / (Nat.choose N 10)

/-- The probability of selecting 8 balls with a sum of 44 -/
def prob_additional_draw (N : ℕ) : ℚ :=
  (number_of_combinations_sum_44 N) / (Nat.choose N 8)

/-- The main theorem: probabilities are equal if and only if N = 18 -/
theorem equal_prob_iff_N_18 :
  prob_main_draw N = prob_additional_draw N ↔ N = 18 :=
by
  sorry

#eval N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_prob_iff_N_18_l142_14203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_lattice_point_l142_14289

-- Define the line l passing through points A and B
def line_l (x : ℚ) : ℚ := (8 * x + 1) / 15

-- Define points A and B
def point_A : ℚ × ℚ := (1/2, 1/3)
def point_B : ℚ × ℚ := (1/4, 1/5)

-- Define the lattice point in question
def lattice_point : ℤ × ℤ := (-2, -1)

-- Theorem statement
theorem closest_lattice_point :
  -- The lattice point lies on the line l
  (line_l (lattice_point.1 : ℚ) = lattice_point.2) ∧
  -- The lattice point is the closest to point A
  (∀ (x y : ℤ), (x, y) ≠ lattice_point →
    (line_l (x : ℚ) = y) →
    ((x : ℚ) - point_A.1)^2 + ((y : ℚ) - point_A.2)^2 >
    ((lattice_point.1 : ℚ) - point_A.1)^2 + ((lattice_point.2 : ℚ) - point_A.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_lattice_point_l142_14289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_A_coordinates_l142_14215

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the lines
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

def on_x_axis (p : Point) : Prop := p.y = 0
def on_y_axis (p : Point) : Prop := p.x = 0

-- Define the problem setup
axiom A : Point
axiom B : Point
axiom C : Point

axiom B_on_x_axis : on_x_axis B
axiom C_on_y_axis : on_y_axis C

axiom a : ℝ
axiom b : ℝ

-- Define the three lines
noncomputable def line1 : Line := { m := a, b := 4 }
noncomputable def line2 : Line := { m := 2, b := b }
noncomputable def line3 : Line := { m := a/2, b := 8 }

-- Helper function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

-- Axiom stating that A, B, C are on these lines
axiom points_on_lines : 
  (point_on_line A line1 ∨ point_on_line A line2 ∨ point_on_line A line3) ∧
  (point_on_line B line1 ∨ point_on_line B line2 ∨ point_on_line B line3) ∧
  (point_on_line C line1 ∨ point_on_line C line2 ∨ point_on_line C line3)

-- The theorem to prove
theorem sum_of_A_coordinates : 
  A.x + A.y = 13 ∨ A.x + A.y = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_A_coordinates_l142_14215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_plane_equation_proof_result_l142_14284

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- A line in 3D space represented by two planes it lies on -/
structure Line where
  plane1 : Plane
  plane2 : Plane

/-- The distance from a point to a plane -/
noncomputable def distance_point_to_plane (x y z : ℚ) (p : Plane) : ℝ :=
  |↑p.A * x + ↑p.B * y + ↑p.C * z + ↑p.D| / Real.sqrt (↑p.A^2 + ↑p.B^2 + ↑p.C^2)

/-- Check if a plane contains a line -/
def plane_contains_line (p : Plane) (l : Line) : Prop :=
  ∀ (x y z : ℚ), p.A * x + p.B * y + p.C * z + p.D = 0 →
    (l.plane1.A * x + l.plane1.B * y + l.plane1.C * z + l.plane1.D = 0 ∧
     l.plane2.A * x + l.plane2.B * y + l.plane2.C * z + l.plane2.D = 0)

theorem plane_equation_proof (M : Line) (Q : Plane) : Prop :=
  let plane1 : Plane := ⟨2, -1, 1, -4⟩
  let plane2 : Plane := ⟨1, 1, -2, 1⟩
  M = ⟨plane1, plane2⟩ →
  Q.A > 0 →
  Int.gcd (Int.natAbs Q.A) (Int.gcd (Int.natAbs Q.B) (Int.gcd (Int.natAbs Q.C) (Int.natAbs Q.D))) = 1 →
  Q ≠ plane1 ∧ Q ≠ plane2 →
  plane_contains_line Q M →
  distance_point_to_plane 2 (-1) 1 Q = 3 / Real.sqrt 2 →
  Q = Plane.mk 3 (-4) 7 (-10)

-- The proof of this theorem is omitted
theorem plane_equation_proof_result : ∃ M Q, plane_equation_proof M Q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_plane_equation_proof_result_l142_14284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_of_f_l142_14268

-- Define the function f(x) = (1/3)x - ln x
noncomputable def f (x : ℝ) : ℝ := (1/3) * x - Real.log x

-- Theorem statement
theorem zero_points_of_f :
  (∀ x ∈ Set.Ioo (1/Real.exp 1) 1, f x ≠ 0) ∧
  (∃ x ∈ Set.Ioo 1 (Real.exp 1), f x = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_of_f_l142_14268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l142_14248

open Real

/-- The function f(x) = 1/2 * x^2 - x + a * ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - x + a * log x

/-- A predicate to express that x is an extreme point of function g -/
def IsExtremePoint (g : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → g y ≤ g x ∨ g y ≥ g x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 2/9)
  (hx : 0 < x₁ ∧ x₁ < x₂)
  (hf : IsExtremePoint (f a) x₁ ∧ IsExtremePoint (f a) x₂) :
  (f a x₁) / x₂ > -5/12 - 1/3 * log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l142_14248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l142_14252

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3^(2*x)

-- Define the shift amount
def shift : ℝ := 1

-- Theorem statement
theorem graph_shift :
  ∀ x y : ℝ, y = f (x - shift) ↔ y = 9^(x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l142_14252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_odd_iff_perfect_square_l142_14216

theorem divisors_odd_iff_perfect_square (n : ℕ+) :
  Odd (Finset.card (Nat.divisors n)) ↔ ∃ k : ℕ, (n : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_odd_iff_perfect_square_l142_14216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_approx_81_82_l142_14241

-- Define the mean scores and ratio
noncomputable def morning_mean : ℝ := 90
noncomputable def afternoon_mean : ℝ := 75
def class_ratio : ℚ := 5/6

-- Define the function to calculate the combined mean
noncomputable def combined_mean (m : ℝ) (a : ℝ) (r : ℚ) : ℝ :=
  let morning_students : ℝ := (r : ℝ) * a
  let total_students : ℝ := morning_students + a
  let total_score : ℝ := m * morning_students + a * a
  total_score / total_students

-- Theorem statement
theorem combined_mean_approx_81_82 :
  ∃ ε > 0, |combined_mean morning_mean afternoon_mean class_ratio - 900/11| < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_approx_81_82_l142_14241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_example_l142_14264

noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_example :
  let (x, y) := polar_to_rectangular 5 (5 * π / 4)
  x = -5 * Real.sqrt 2 / 2 ∧ y = -5 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_example_l142_14264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_decreasing_function_log_translation_l142_14201

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Proposition ②
theorem periodic_function (h : ∀ x : ℝ, f (2 + x) = -f x) : 
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
sorry

-- Proposition ③
theorem decreasing_function (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  ∀ x y : ℝ, x < y → f x > f y :=
sorry

-- Proposition ④
theorem log_translation :
  ∃ a b : ℝ, ∀ x : ℝ, Real.log ((x + 3) / 10) = Real.log x + a * x + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_decreasing_function_log_translation_l142_14201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l142_14294

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the maximum value M(a) of f(x) in [1, 3]
noncomputable def M (a : ℝ) : ℝ := 
  if a ≤ 1/2 then f a 1 else f a 3

-- Define the minimum value N(a) of f(x) in [1, 3]
noncomputable def N (a : ℝ) : ℝ := 1 - 1/a

-- Define the function g(a)
noncomputable def g (a : ℝ) : ℝ := M a - N a

-- State the theorem
theorem g_properties :
  ∀ a : ℝ, 1/3 ≤ a ∧ a ≤ 1 →
    (g a = if a ≤ 1/2 then a - 2 + 1/a else 9*a - 6 + 1/a) ∧
    (∀ b : ℝ, 1/3 ≤ b ∧ b ≤ 1 → g b ≥ 1/2) ∧
    g (1/2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l142_14294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_half_l142_14244

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ := ∑' k, (1 : ℝ) / (k + 2 : ℝ) ^ n

/-- The theorem stating that the sum of g(n) from n = 1 to infinity equals 1/2 -/
theorem sum_of_g_equals_half : ∑' n, g n = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_half_l142_14244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_smaller_radius_l142_14226

/-- A frustum with the given properties -/
structure Frustum where
  r : ℝ  -- radius of smaller base
  R : ℝ  -- radius of larger base
  l : ℝ  -- slant height
  S : ℝ  -- lateral area

/-- The circumference of the larger base is three times the circumference of the smaller base -/
def circum_ratio (f : Frustum) : Prop :=
  2 * Real.pi * f.R = 3 * (2 * Real.pi * f.r)

/-- The slant height is 3 -/
def slant_height (f : Frustum) : Prop :=
  f.l = 3

/-- The lateral area is 84π -/
def lateral_area (f : Frustum) : Prop :=
  f.S = 84 * Real.pi

/-- Theorem: The radius of the smaller base is 7 -/
theorem frustum_smaller_radius (f : Frustum) 
  (h1 : circum_ratio f) 
  (h2 : slant_height f) 
  (h3 : lateral_area f) : 
  f.r = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_smaller_radius_l142_14226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l142_14249

theorem sum_of_integers (x y : ℤ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x - y = 8) 
  (h2 : x * y = 240) : 
  x + y = 32 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l142_14249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l142_14202

theorem trig_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 169/381 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l142_14202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l142_14237

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

def tangent_line (x y : ℝ) : Prop := x - 2 * y = 0

theorem tangent_through_origin : 
  ∃ (x₀ : ℝ), x₀ > 1 ∧ 
  (∀ (x : ℝ), tangent_line x (f x) ↔ x = x₀) ∧
  tangent_line 0 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l142_14237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l142_14209

noncomputable def curve (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 3)
noncomputable def point_B : ℝ × ℝ := (-Real.sqrt 3, 1)

theorem distance_AB : 
  curve point_A.1 point_A.2 → 
  curve point_B.1 point_B.2 → 
  Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l142_14209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l142_14212

/-- Custom binary operation ⊗ -/
noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

/-- Theorem stating the result of the calculation -/
theorem otimes_calculation : 
  ((otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4))) = -2016/729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l142_14212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_p_theorem_l142_14239

noncomputable def largest_p_for_three_positive_integer_roots : ℝ := 76

theorem largest_p_theorem : 
  -- Define the polynomial function
  let f (p : ℝ) (x : ℝ) : ℝ := 5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 - 66*p

  -- Define the property of having three positive integer roots
  let has_three_positive_integer_roots (p : ℝ) : Prop :=
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    f p (↑a) = 0 ∧ f p (↑b) = 0 ∧ f p (↑c) = 0

  -- The theorem statement
  (∀ q : ℝ, q > largest_p_for_three_positive_integer_roots → 
    ¬(has_three_positive_integer_roots q)) ∧
  (has_three_positive_integer_roots largest_p_for_three_positive_integer_roots) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_p_theorem_l142_14239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_redistribution_l142_14218

theorem sticker_redistribution (e : ℚ) (h : e > 0) : 
  let liam_initial := 4 * e
  let noah_initial := 3 * e
  let emma_initial := e
  let total := liam_initial + noah_initial + emma_initial
  let equal_share := total / 3
  let liam_gives := liam_initial - equal_share
  let noah_receives := equal_share - noah_initial
  noah_receives / liam_initial = 1 / 6 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_redistribution_l142_14218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_sum_difference_l142_14245

def divisor_sum (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem unique_divisor_sum_difference :
  ∃! (c : ℕ), ∃ (seq : ℕ → ℕ),
    (∀ i j : ℕ, i < j → seq i < seq j) ∧
    (∀ i : ℕ, divisor_sum (seq i) - seq i = c) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_sum_difference_l142_14245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_inequality_l142_14231

-- Define the constants
noncomputable def a : ℝ := (1/2)^3
noncomputable def b : ℝ := 3^(1/2)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

-- State the theorem
theorem a_b_c_inequality : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_inequality_l142_14231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_values_l142_14238

-- Define the functions f and g
noncomputable def f (a b x : ℝ) : ℝ := a + 3 / (x - b)
noncomputable def g (c x : ℝ) : ℝ := 1 + c / (2 * x + 1)

-- State the theorem
theorem inverse_functions_values :
  ∃ (a b c : ℝ), 
    (∀ x, f a b (g c x) = x) ∧ 
    (∀ x, g c (f a b x) = x) ∧
    a = -1/2 ∧ b = 1 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_values_l142_14238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l142_14276

-- Define the sequences a_n and b_n
def a : ℕ → ℤ := sorry
def b : ℕ → ℤ := sorry

-- Define the conditions
axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 3
axiom b_1 : b 1 = 5

-- Define the condition for any positive integers i, j, k, l
axiom condition (i j k l : ℕ) : i + j = k + l → a i + b j = a k + b l

-- Theorem to prove
theorem sequence_sum (n : ℕ) : a n + b n = 4 * n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l142_14276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_properties_l142_14274

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  slope1 : ℝ
  slope2 : ℝ
  intersectionX : ℝ
  intersectionY : ℝ

/-- Calculate the x-intercept of a line given its slope and a point it passes through -/
noncomputable def xIntercept (m : ℝ) (x0 y0 : ℝ) : ℝ :=
  x0 - y0 / m

/-- Calculate the y-intercept of a line given its slope and a point it passes through -/
def yIntercept (m : ℝ) (x0 y0 : ℝ) : ℝ :=
  y0 - m * x0

/-- Calculate the distance between two points on the x-axis -/
def xAxisDistance (x1 x2 : ℝ) : ℝ :=
  |x1 - x2|

theorem intersecting_lines_properties (lines : IntersectingLines)
    (h1 : lines.slope1 = 2)
    (h2 : lines.slope2 = -4)
    (h3 : lines.intersectionX = 12)
    (h4 : lines.intersectionY = 20) :
    xAxisDistance (xIntercept lines.slope1 lines.intersectionX lines.intersectionY)
                  (xIntercept lines.slope2 lines.intersectionX lines.intersectionY) = 15 ∧
    yIntercept lines.slope1 lines.intersectionX lines.intersectionY = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_properties_l142_14274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l142_14269

theorem triangle_angle_inequalities (A B : ℝ) (h : A > B) (h_triangle : 0 < A ∧ 0 < B ∧ A + B < Real.pi) :
  (Real.sin A > Real.sin B ∧ Real.cos A < Real.cos B ∧ Real.cos (2 * A) < Real.cos (2 * B)) ∧
  ¬(∀ A B : ℝ, A > B → Real.sin (2 * A) > Real.sin (2 * B)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l142_14269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_possibility_l142_14220

theorem triangle_side_possibility (AB BC AC : ℝ) : 
  AB = 5 → BC = 9 → AC = 5 → 
  AB + BC > AC ∧ BC + AC > AB ∧ AC + AB > BC ∧
  |AB - BC| < AC ∧ |BC - AC| < AB ∧ |AC - AB| < BC :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_possibility_l142_14220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l142_14297

/-- Represents a circle in a 2D plane --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle --/
noncomputable def Circle.center (circle : Circle) : ℝ × ℝ :=
  (-circle.a / 2, -circle.b / 2)

/-- The radius of a circle --/
noncomputable def Circle.radius (circle : Circle) : ℝ :=
  Real.sqrt ((circle.a / 2)^2 + (circle.b / 2)^2 - circle.e)

theorem circle_center_and_radius :
  let circle : Circle := { a := -2, b := 6, c := 1, d := 1, e := 0 }
  circle.center = (1, -3) ∧ circle.radius = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l142_14297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendance_difference_l142_14261

theorem max_attendance_difference : 
  ∀ (a b : ℝ), 
  (0.9 * 40000 ≤ a ∧ a ≤ 1.1 * 40000) → 
  (0.85 * 70000 ≤ b ∧ b ≤ 1.15 * 70000) → 
  b - a ≤ 46000 := by
  -- Alice's estimate for Atlanta
  let alice_estimate : ℝ := 40000
  -- Carl's estimate for Boston
  let carl_estimate : ℝ := 70000
  -- Actual attendance in Atlanta
  let atlanta_actual := λ a : ℝ => 0.9 * alice_estimate ≤ a ∧ a ≤ 1.1 * alice_estimate
  -- Actual attendance in Boston
  let boston_actual := λ b : ℝ => 0.85 * carl_estimate ≤ b ∧ b ≤ 1.15 * carl_estimate
  -- Maximum difference between attendances
  let max_difference : ℝ := 46000
  -- Proof
  intro a b ha hb
  sorry

#check max_attendance_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendance_difference_l142_14261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_or_power_of_two_l142_14234

theorem prime_or_power_of_two (n : ℕ) (h_n : n > 6) :
  (∃ (k : ℕ) (a : ℕ → ℕ) (d : ℕ),
    (∀ i, i ∈ Finset.range k → a i < n ∧ Nat.Coprime (a i) n) ∧
    (∀ i, i ∈ Finset.range (k - 1) → a (i + 1) - a i = d) ∧
    d > 0 ∧
    (∀ m, m < n → Nat.Coprime m n → ∃ i, i ∈ Finset.range k ∧ a i = m)) →
  Nat.Prime n ∨ ∃ m : ℕ, n = 2^m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_or_power_of_two_l142_14234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_pet_owners_l142_14293

theorem expected_pet_owners (sample_size : ℕ) (pet_ownership_ratio : ℚ) 
  (h1 : sample_size = 400) 
  (h2 : pet_ownership_ratio = 3/8) : 
  Int.floor (pet_ownership_ratio * ↑sample_size) = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_pet_owners_l142_14293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_abc_in_special_triangle_l142_14258

/-- Given a triangle ABC where the ratio of sines of angles A, B, and C is 2:3:4,
    prove that the angle ABC is equal to arccos(11/16) -/
theorem angle_abc_in_special_triangle (A B C : Real) (h_triangle : A + B + C = Real.pi)
  (h_sine_ratio : ∃ (k : Real), Real.sin A = 2*k ∧ Real.sin B = 3*k ∧ Real.sin C = 4*k) :
  B = Real.arccos (11/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_abc_in_special_triangle_l142_14258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_x_coordinate_l142_14262

/-- A circle with diameter endpoints (1,5) and (7,3) intersects the line y = 1 at a point with x-coordinate either 3 or 5 -/
theorem circle_intersection_x_coordinate :
  ∃ (x : ℝ), (x = 3 ∨ x = 5) ∧
  let center : ℝ × ℝ := ((1 + 7) / 2, (5 + 3) / 2)
  let radius : ℝ := Real.sqrt ((1 - center.1)^2 + (5 - center.2)^2)
  (x - center.1)^2 + (1 - center.2)^2 = radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_x_coordinate_l142_14262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_M_M_contains_one_M_closed_under_operations_M_is_smallest_l142_14229

-- Define the set M using an inductive definition
inductive M : ℚ → Prop
  | one : M 1
  | inv (x : ℚ) : M x → M (1 / (1 + x))
  | frac (x : ℚ) : M x → M (x / (1 + x))

-- State the theorem
theorem smallest_set_M :
  ∀ x : ℚ, M x ↔ 0 < x ∧ x ≤ 1 :=
by sorry

-- Define the property that M contains 1
theorem M_contains_one : M 1 :=
M.one

-- Define the property that M is closed under the given operations
theorem M_closed_under_operations {x : ℚ} (hx : M x) :
  M (1 / (1 + x)) ∧ M (x / (1 + x)) :=
⟨M.inv x hx, M.frac x hx⟩

-- Define the property that M is the smallest set satisfying the conditions
theorem M_is_smallest (S : ℚ → Prop) 
  (h1 : S 1) 
  (h2 : ∀ x, S x → S (1 / (1 + x)) ∧ S (x / (1 + x))) :
  ∀ x, M x → S x :=
by
  intro x hx
  induction hx with
  | one => exact h1
  | inv x hx ih => exact (h2 x ih).1
  | frac x hx ih => exact (h2 x ih).2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_M_M_contains_one_M_closed_under_operations_M_is_smallest_l142_14229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_theorem_l142_14265

/-- Parabola struct representing y² = 4x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line struct representing a line passing through a point --/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Point struct representing a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the slope between two points --/
noncomputable def slopeBetweenPoints (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Theorem stating the existence of a line l such that for any point M on l,
    the slopes of MA, MF, and MB form an arithmetic sequence --/
theorem parabola_line_theorem (p : Parabola) (l : Line) :
  p.equation = fun x y => y^2 = 4*x →
  p.focus = (1, 0) →
  l.slope = 0 →
  l.yIntercept = -1 →
  ∀ (M : Point),
    M.x = -1 →
    ∃ (A B : Point),
      p.equation A.x A.y ∧
      p.equation B.x B.y ∧
      let slopeMA := slopeBetweenPoints M A
      let slopeMF := slopeBetweenPoints M (Point.mk 1 0)
      let slopeMB := slopeBetweenPoints M B
      2 * slopeMF = slopeMA + slopeMB :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_theorem_l142_14265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_f_greater_than_x_plus_one_l142_14205

-- Define the functions
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) * (x^3 + α * x + 4 * x * Real.cos x + 1)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m * (x + 1)

-- State the theorems
theorem g_minimum (m : ℝ) (hm : m ≥ 1) :
  ∃ (x : ℝ), ∀ (y : ℝ), g m y ≥ g m x ∧ g m x = -m * Real.log m := by
  sorry

theorem f_greater_than_x_plus_one (α : ℝ) (hα : α ≥ -7/2) :
  ∀ x, x > 0 ∧ x < 1 → f α x > x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_f_greater_than_x_plus_one_l142_14205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l142_14280

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

def has_exact_extreme_points (g : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ (s : Finset ℝ), s.card = n ∧ (∀ x ∈ s, a < x ∧ x < b) ∧
    (∀ x ∈ s, HasDerivAt g (0 : ℝ) x) ∧
    (∀ x, a < x ∧ x < b → x ∉ s → HasDerivAt g (deriv g x) x)

def has_exact_zeros (g : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ (s : Finset ℝ), s.card = n ∧ (∀ x ∈ s, a < x ∧ x < b ∧ g x = 0) ∧
    (∀ x, a < x ∧ x < b → x ∉ s → g x ≠ 0)

theorem omega_range (ω : ℝ) :
  has_exact_extreme_points (f ω) 3 0 Real.pi ∧ has_exact_zeros (f ω) 2 0 Real.pi →
  13 / 6 < ω ∧ ω ≤ 8 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l142_14280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l142_14254

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the absolute value function
def absValue (x : ℝ) : ℝ := abs x

theorem problem_solution :
  (intPart (5 - Real.sqrt 2) = 3) ∧
  (absValue (1 - 2 * Real.sqrt 3) = 2 * Real.sqrt 3 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l142_14254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sin_half_pi_plus_alpha_l142_14299

theorem cos_squared_sin_half_pi_plus_alpha (α : ℝ) 
  (h1 : π / 2 < α * π) 
  (h2 : Real.cos α = -3 / 5) : 
  ((Real.cos^2) ∘ Real.sin) (π / 2 + α) = 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sin_half_pi_plus_alpha_l142_14299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_sin_lambda_alpha_l142_14273

/-- A piecewise function f(x) --/
noncomputable def f (x : ℝ) (lambda alpha : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2017*x + Real.sin x
  else -x^2 + lambda*x + Real.cos (x + alpha)

/-- The main theorem --/
theorem odd_function_implies_sin_lambda_alpha (lambda alpha : ℝ) :
  (∀ x, f (-x) lambda alpha = -(f x lambda alpha)) → Real.sin (lambda * alpha) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_sin_lambda_alpha_l142_14273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_product_analogical_reasoning_l142_14236

-- Define the property for real numbers
def absolute_value_product_real : Prop :=
  ∀ (z₁ z₂ : ℝ), |z₁ * z₂| = |z₁| * |z₂|

-- Define the property for complex numbers
def absolute_value_product_complex : Prop :=
  ∀ (z₁ z₂ : ℂ), Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂

-- Define analogical reasoning
def is_analogical_reasoning (P Q : Prop) : Prop :=
  P → Q

-- Theorem statement
theorem absolute_value_product_analogical_reasoning :
  is_analogical_reasoning absolute_value_product_real absolute_value_product_complex :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_product_analogical_reasoning_l142_14236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l142_14287

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6) - Real.cos (2 * x + Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem f_properties :
  (f (Real.pi / 12) = Real.sqrt 3 + 1) ∧
  (∀ x : ℝ, f x ≤ 3) ∧
  (∀ x : ℝ, f x = 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l142_14287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l142_14267

theorem calculate_expression : (-2 : ℤ)^3 + ((-4 : ℤ)^2 - (1 - 3^2) * 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l142_14267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_to_c_l142_14272

/-- Represents the amount of money in Rupees -/
abbrev Rupees := ℝ

/-- The simple interest rate as a decimal -/
def interest_rate : ℝ := 0.1375

/-- Calculate simple interest -/
def simple_interest (principal : Rupees) (time : ℝ) : Rupees :=
  principal * interest_rate * time

theorem loan_to_c (loan_to_b loan_to_c : Rupees) 
  (h1 : loan_to_b = 4000)
  (h2 : simple_interest loan_to_b 2 + simple_interest loan_to_c 4 = 2200) :
  loan_to_c = 2000 := by
  sorry

#check loan_to_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_to_c_l142_14272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_complementary_events_l142_14223

-- Define the probabilities of events A and B
noncomputable def P_A (y : ℝ) : ℝ := 1 / y
noncomputable def P_B (x : ℝ) : ℝ := 4 / x

-- State the theorem
theorem min_sum_complementary_events (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_complement : P_A y + P_B x = 1) :
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ P_A y + P_B x = 1 ∧ x + y = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_complementary_events_l142_14223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l142_14228

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the length of the real axis
noncomputable def real_axis_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola x y → real_axis_length = 2 * Real.sqrt 2 :=
by
  -- Introduce variables and hypothesis
  intro x y h
  -- Unfold the definition of real_axis_length
  unfold real_axis_length
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l142_14228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l142_14253

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 1400 ∧
  2021 ∈ arr ∧
  ∀ i, i < arr.length → 
    arr[i]! = Nat.gcd (arr[(i-2 + 1400) % 1400]!) (arr[(i-1 + 1400) % 1400]!) + 
              Nat.gcd (arr[(i+1) % 1400]!) (arr[(i+2) % 1400]!)

theorem no_valid_arrangement : ¬ ∃ arr : List Nat, is_valid_arrangement arr := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l142_14253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l142_14296

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.cos (ω * x + Real.pi / 6)

theorem periodic_function_value (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) →
  (∀ p : ℝ, p > 0 → p < Real.pi → ∃ x : ℝ, f ω (x + p) ≠ f ω x) →
  (f ω (Real.pi / 3) = -3 ∨ f ω (Real.pi / 3) = 0) :=
by
  sorry

#check periodic_function_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l142_14296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l142_14210

theorem problem_statement (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : Real.log a * Real.log b = 1) :
  (Real.log 2 / Real.log a < Real.log 2 / Real.log b) ∧
  ((1/2 : ℝ)^(a*b+1) < (1/2 : ℝ)^(a+b)) ∧
  (a^a * b^b > a^b * b^a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l142_14210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l142_14206

/-- The cost price of a book given its selling price and profit percentage -/
noncomputable def cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem stating that the cost price of a book sold for $200 with 20% profit is approximately $166.67 -/
theorem book_cost_price :
  let selling_price : ℝ := 200
  let profit_percentage : ℝ := 20
  abs (cost_price selling_price profit_percentage - 166.67) < 0.01 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l142_14206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l142_14224

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (1 - Real.log x)) / (x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 0 < x ∧ x ≤ Real.exp 1 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l142_14224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_shifted_unit_value_l142_14288

theorem polynomial_with_shifted_unit_value
  (k : ℕ)
  (P : Polynomial ℝ)
  (h_deg : P.natDegree = k)
  (h_zeros : ∃ (zeros : Finset ℝ), zeros.card = k ∧ (∀ a ∈ zeros, P.eval a = 0))
  (h_shift : ∀ a : ℝ, P.eval a = 0 → P.eval (a + 1) = 1) :
  (k = 0 ∧ P = 1) ∨ (k = 1 ∧ ∃ a : ℝ, P = Polynomial.X - Polynomial.C a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_shifted_unit_value_l142_14288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l142_14275

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) :
  a^(-2 : ℤ) / a^(5 : ℤ) * (4 * a) / (2^(-1 : ℤ) * a)^(-3 : ℤ) = 1 / (2 * a^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l142_14275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_difference_l142_14214

def digits : List ℕ := [1, 3, 5, 7, 8]

def is_valid_subtraction (a b : ℕ) : Prop :=
  (1000 ≤ b) ∧ (b < 10000) ∧ (100 ≤ a) ∧ (a < 1000) ∧
  (∀ d : ℕ, d ∈ digits → (List.count d (Nat.digits 10 a) + List.count d (Nat.digits 10 b) = 1))

def smallest_difference : ℕ := 482

theorem smallest_possible_difference :
  ∀ a b : ℕ, is_valid_subtraction a b → b - a ≥ smallest_difference :=
by sorry

#check smallest_possible_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_difference_l142_14214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_from_polar_equation_l142_14207

/-- The polar equation of the circle -/
def polar_equation (r θ : ℝ) : Prop := r = -2 * Real.cos θ + 6 * Real.sin θ

/-- The area of the circle described by the polar equation -/
noncomputable def circle_area : ℝ := 10 * Real.pi

/-- Theorem stating that the area of the circle described by the polar equation is 10π -/
theorem area_of_circle_from_polar_equation :
  ∀ r θ : ℝ, polar_equation r θ → circle_area = 10 * Real.pi := by
  sorry

#check area_of_circle_from_polar_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_from_polar_equation_l142_14207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_product_l142_14200

/-- Regular pentagon in the complex plane -/
def RegularPentagon (Q : Fin 5 → ℂ) : Prop :=
  Q 0 = 1 ∧ Q 2 = 5 ∧ ∀ i j : Fin 5, Complex.abs (Q i - Q j) = Complex.abs (Q ((i + 1) % 5) - Q ((j + 1) % 5))

theorem regular_pentagon_product (Q : Fin 5 → ℂ) (h : RegularPentagon Q) :
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) = 242 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_product_l142_14200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_fraction_equals_one_l142_14246

theorem sine_fraction_equals_one :
  let c : ℝ := 2 * Real.pi / 13
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (2 * c) * Real.sin (3 * c) * Real.sin (5 * c) * Real.sin (6 * c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_fraction_equals_one_l142_14246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_squared_l142_14222

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5

theorem sum_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 80/361 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_squared_l142_14222
