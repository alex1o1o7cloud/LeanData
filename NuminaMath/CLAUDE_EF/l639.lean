import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_proof_l639_63999

/-- The amount of alloy b in kg -/
noncomputable def alloy_b_amount : ℝ := 250

/-- The ratio of lead to tin in alloy a -/
noncomputable def alloy_a_ratio : ℝ := 1 / 3

/-- The ratio of tin to copper in alloy b -/
noncomputable def alloy_b_ratio : ℝ := 3 / 5

/-- The amount of tin in the new alloy in kg -/
noncomputable def total_tin : ℝ := 221.25

/-- The amount of alloy a used in kg -/
noncomputable def alloy_a_amount : ℝ := 170

theorem alloy_mixture_proof :
  alloy_a_amount * (alloy_a_ratio / (1 + alloy_a_ratio)) +
  alloy_b_amount * (alloy_b_ratio / (1 + alloy_b_ratio)) = total_tin :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_proof_l639_63999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_membership_l639_63957

def my_sequence (n : ℕ) : ℚ := 1 - 1 / (n + 1)

theorem sequence_membership : 
  (∃ n₁ n₂ : ℕ, my_sequence n₁ = 49/50 ∧ my_sequence n₂ = 24/25) ∧ 
  (∀ n : ℕ, my_sequence n ≠ 47/50) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_membership_l639_63957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_theorem_l639_63922

/-- The distance between a point in polar coordinates and a line in polar form --/
noncomputable def distance_point_to_line (ρ : ℝ) (θ : ℝ) (a : ℝ) : ℝ :=
  sorry

/-- The point P in polar coordinates --/
noncomputable def P : ℝ × ℝ := (2 * Real.sqrt 3, Real.pi / 6)

/-- The line l in polar form: ρ cos(θ + π/4) = 2√2 --/
def line_equation (ρ : ℝ) (θ : ℝ) : Prop :=
  ρ * Real.cos (θ + Real.pi / 4) = 2 * Real.sqrt 2

/-- The theorem stating the distance between point P and line l --/
theorem distance_point_to_line_theorem :
  distance_point_to_line P.1 P.2 (2 * Real.sqrt 2) = (Real.sqrt 2 + Real.sqrt 6) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_theorem_l639_63922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_purely_imaginary_z₁_fourth_quadrant_z₂_first_quadrant_l639_63953

-- Define complex numbers z₁ and z₂
def z₁ (x : ℝ) : ℂ := x^2 - 1 + (x^2 - 3*x + 2)*Complex.I
def z₂ (x : ℝ) : ℂ := x + (3 - 2*x)*Complex.I

-- Theorem for the first question
theorem z₁_purely_imaginary (x : ℝ) :
  (z₁ x).re = 0 ∧ (z₁ x).im ≠ 0 → x = -1 :=
by
  sorry

-- Theorem for the second question
theorem z₁_fourth_quadrant_z₂_first_quadrant (x : ℝ) :
  (z₁ x).re > 0 ∧ (z₁ x).im < 0 ∧ (z₂ x).re > 0 ∧ (z₂ x).im > 0 →
  1 < x ∧ x < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_purely_imaginary_z₁_fourth_quadrant_z₂_first_quadrant_l639_63953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l639_63966

noncomputable def a (m : ℝ) : ℝ × ℝ := (m, 1)
noncomputable def b : ℝ × ℝ := (1/2, Real.sqrt 3 / 2)

def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_relations :
  (∃ m : ℝ, parallel (a m) b ∧ m = -Real.sqrt 3 / 3) ∧
  (∃ m : ℝ, perpendicular (a m) b ∧ m = -Real.sqrt 3) ∧
  (∀ k t : ℝ, k ≠ 0 → t ≠ 0 →
    perpendicular (a (-Real.sqrt 3)) b →
    perpendicular (a (-Real.sqrt 3) + (t^2 - 3) • b) (-k • a (-Real.sqrt 3) + t • b) →
    (k + t^2) / t ≥ -7/4) ∧
  (∃ k t : ℝ, k ≠ 0 ∧ t ≠ 0 ∧
    perpendicular (a (-Real.sqrt 3)) b ∧
    perpendicular (a (-Real.sqrt 3) + (t^2 - 3) • b) (-k • a (-Real.sqrt 3) + t • b) ∧
    (k + t^2) / t = -7/4) := by
  sorry

#check vector_relations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l639_63966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l639_63984

/-- The distance between stations A and B given two trains meeting -/
theorem distance_between_stations (speed1 speed2 time1 time2 : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : time1 = 4)
  (h4 : time2 = 3) :
  speed1 * time1 + speed2 * time2 = 155 := by
  sorry

#check distance_between_stations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l639_63984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_z_pure_imaginary_l639_63937

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The complex number z as a function of x -/
noncomputable def z (x : ℝ) : ℂ := (2 + i) / (x - i)

/-- Main theorem: If z(x) is pure imaginary, then x = 1/2 -/
theorem x_value_when_z_pure_imaginary : 
  ∀ x : ℝ, isPureImaginary (z x) → x = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_z_pure_imaginary_l639_63937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_squared_alpha_plus_pi_fourth_l639_63918

theorem tan_squared_alpha_plus_pi_fourth (α : Real) 
  (h : Real.cos (α - Real.pi/4) = Real.sqrt 2/4) : 
  Real.tan (α + Real.pi/4) ^ 2 = 1/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_squared_alpha_plus_pi_fourth_l639_63918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_segment_l639_63925

/-- A rectangle in a 2D plane -/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- The center point of a rectangle -/
noncomputable def Rectangle.center (r : Rectangle) : ℝ × ℝ :=
  (r.x + r.width / 2, r.y + r.height / 2)

/-- A partition of a square into rectangles -/
structure SquarePartition where
  side : ℝ
  rectangles : List Rectangle

/-- A line segment between two points -/
structure Segment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Check if a segment intersects a rectangle -/
def Segment.intersectsRectangle (s : Segment) (r : Rectangle) : Prop :=
  sorry

/-- The main theorem -/
theorem exists_non_intersecting_segment (partition : SquarePartition) :
  ∃ (r1 r2 : Rectangle) (s : Segment),
    r1 ∈ partition.rectangles ∧
    r2 ∈ partition.rectangles ∧
    r1 ≠ r2 ∧
    s.start = r1.center ∧
    s.endpoint = r2.center ∧
    ∀ (r : Rectangle), r ∈ partition.rectangles → r ≠ r1 → r ≠ r2 →
      ¬s.intersectsRectangle r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_segment_l639_63925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_and_profit_l639_63921

-- Define the constants and variables
noncomputable def cost_per_kg : ℝ := 20
noncomputable def t : ℝ := 5
noncomputable def x : ℝ := 26

-- Define the daily sales volume function
noncomputable def q (x : ℝ) : ℝ := 100 * Real.exp 30 / Real.exp x

-- Define the daily profit function
noncomputable def y (x : ℝ) : ℝ := q x * (x - cost_per_kg - t)

-- State the theorem
theorem optimal_price_and_profit :
  (∀ x', 25 ≤ x' ∧ x' ≤ 40 → y x' ≤ y x) ∧
  y x = 100 * Real.exp 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_and_profit_l639_63921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l639_63991

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Theorem statement
theorem f_inequality : f 1 < f (Real.sqrt 3) ∧ f (Real.sqrt 3) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l639_63991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_area_iff_rectangle_l639_63939

/-- A convex quadrilateral with side lengths a, b, c, d in sequential order. -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The area of a quadrilateral calculated using the Egyptian formula. -/
noncomputable def egyptianArea (q : ConvexQuadrilateral) : ℝ :=
  (q.a + q.c) * (q.b + q.d) / 4

/-- The actual area of a quadrilateral. -/
noncomputable def actualArea (q : ConvexQuadrilateral) : ℝ := sorry

/-- A predicate to determine if a quadrilateral is a rectangle. -/
def isRectangle (q : ConvexQuadrilateral) : Prop := sorry

/-- Theorem stating that the Egyptian formula for area is correct if and only if the quadrilateral is a rectangle. -/
theorem egyptian_area_iff_rectangle (q : ConvexQuadrilateral) :
  egyptianArea q = actualArea q ↔ isRectangle q := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_area_iff_rectangle_l639_63939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l639_63960

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (1/2)^x}
def Q : Set ℝ := {x | ∃ y : ℝ, y = Real.log (2*x - x^2) / Real.log 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l639_63960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_properties_l639_63909

theorem subset_with_properties (A : Finset ℕ+) : 
  ∃ B : Finset ℕ+, B ⊆ A ∧ 
    (∀ b₁ b₂ : ℕ+, b₁ ∈ B → b₂ ∈ B → b₁ ≠ b₂ → 
      (¬(b₁ ∣ b₂) ∧ ¬(b₂ ∣ b₁) ∧ ¬((b₁ + 1) ∣ (b₂ + 1)) ∧ ¬((b₂ + 1) ∣ (b₁ + 1)))) ∧
    (∀ a : ℕ+, a ∈ A → ∃ b : ℕ+, b ∈ B ∧ ((a ∣ b) ∨ ((b + 1) ∣ (a + 1)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_properties_l639_63909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_min_value_of_a_l639_63956

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the tangent line
theorem tangent_line_at_one (x : ℝ) :
  (fun y => y = x - 1) = (fun y => ∃ k, y - f 1 = k * (x - 1) ∧ k = (deriv f) 1) := by
  sorry

-- Theorem for the minimum value of a
theorem min_value_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ a * x^2 + 2/a) ↔ a ≥ -Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_min_value_of_a_l639_63956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_winner_max_matches_l639_63994

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Definition of a valid tournament -/
def valid_tournament (n : ℕ) (max_matches : ℕ) : Prop :=
  fib (max_matches + 2) = n ∧
  ∀ m : ℕ, m ≤ max_matches → 
    ∃ tournament : Finset (Fin n), 
      tournament.card = fib (m + 2) ∧
      ∃ winner : Fin n, winner ∈ tournament ∧
        ∀ loser : Fin n, loser ∈ tournament → 
          (Int.ofNat winner.val - Int.ofNat loser.val).natAbs ≤ 1

theorem tournament_winner_max_matches :
  valid_tournament 55 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_winner_max_matches_l639_63994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_l639_63945

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  E : Point
  F : Point
  G : Point
  H : Point
  EF : ℝ
  FG : ℝ
  HE : ℝ
  GH : ℝ

/-- Checks if a circle is tangent to another circle -/
def isTangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x)^2 + (c1.center.y - c2.center.y)^2 = (c1.radius + c2.radius)^2

/-- The main theorem to be proved -/
theorem inner_circle_radius 
  (trapezoid : IsoscelesTrapezoid)
  (circleE circleF circleG circleH innerCircle : Circle) :
  trapezoid.EF = 8 →
  trapezoid.FG = 3 →
  trapezoid.HE = 3 →
  trapezoid.GH = 6 →
  circleE.radius = 4 →
  circleF.radius = 4 →
  circleG.radius = 2.5 →
  circleH.radius = 2.5 →
  isTangent innerCircle circleE →
  isTangent innerCircle circleF →
  isTangent innerCircle circleG →
  isTangent innerCircle circleH →
  ∃ (ε : ℝ), abs (innerCircle.radius - 1.673) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_l639_63945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l639_63993

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the line l
def L (x y : ℝ) : Prop := x + 3*y - 6 = 0

-- Define the angle between two vectors
noncomputable def angle (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.arccos ((x₁*x₂ + y₁*y₂) / (Real.sqrt (x₁^2 + y₁^2) * Real.sqrt (x₂^2 + y₂^2)))

-- Statement of the theorem
theorem circle_line_intersection_range :
  ∀ x₀ y₀ : ℝ,
  L x₀ y₀ →
  (∃ x y : ℝ, C x y ∧ angle x₀ y₀ x y = π/3) →
  0 ≤ x₀ ∧ x₀ ≤ 6/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l639_63993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_fibonacci_is_34_l639_63910

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => fibonacci (n+1) + fibonacci n

theorem ninth_fibonacci_is_34 : fibonacci 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_fibonacci_is_34_l639_63910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_properties_l639_63926

/-- A function f: ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function f: ℝ → ℝ satisfies the functional equation f(x+y) = f(x)f(y) for all x, y ∈ ℝ -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y

/-- The exponential function with base 3 -/
noncomputable def f (x : ℝ) : ℝ := 3^x

theorem exponential_properties :
  MonotonicallyIncreasing f ∧ SatisfiesFunctionalEquation f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_properties_l639_63926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_EFGH_product_l639_63914

noncomputable section

def grid_distance : ℝ := 1

structure Point where
  x : ℝ
  y : ℝ

def E : Point := ⟨4, 5⟩
def F : Point := ⟨5, 2⟩
def G : Point := ⟨2, 1⟩
def H : Point := ⟨1, 4⟩

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

noncomputable def side_length : ℝ := distance E F

noncomputable def area : ℝ := side_length^2

noncomputable def perimeter : ℝ := 4 * side_length

theorem square_EFGH_product :
  area * perimeter = 40 * Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_EFGH_product_l639_63914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_coordinates_l639_63976

/-- Given two vectors a and b in ℝ², prove that b = (3, -6) -/
theorem vector_b_coordinates (a b : ℝ × ℝ) (θ : ℝ) : 
  a = (-1, 2) → 
  Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 3 * Real.sqrt 5 →
  θ = Real.arccos (-(a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)))) →
  Real.cos θ = -1 →
  b = (3, -6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_coordinates_l639_63976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_positive_reals_l639_63946

-- Define a differentiable function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f is differentiable with derivative f'
axiom f_differentiable : Differentiable ℝ f
axiom f_has_derivative : ∀ x, HasDerivAt f (f' x) x

-- State the conditions given in the problem
axiom f_gt_f' : ∀ x, f x > f' x
axiom f_0_eq_1 : f 0 = 1

-- Define the theorem to be proved
theorem solution_set_is_positive_reals :
  {x : ℝ | f x / Real.exp x < 1} = Set.Ioi 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_positive_reals_l639_63946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_power_of_two_l639_63948

open BigOperators

def s (n : ℕ) : ℚ :=
  ∑ k in Finset.range ((n / 2) + 1), (1973 : ℚ)^k * (n.choose (2*k + 1))

theorem sum_divisible_by_power_of_two (n : ℕ) :
  ∃ m : ℤ, s n = m * (2 : ℚ)^(n - 1) := by
  sorry

#check sum_divisible_by_power_of_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_power_of_two_l639_63948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_integer_root_of_7776_l639_63978

theorem one_integer_root_of_7776 : ∃! (n : ℕ), n > 0 ∧ ∃ (m : ℕ), m > 0 ∧ (m : ℝ)^n = 7776 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_integer_root_of_7776_l639_63978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_99_l639_63981

def a : ℕ → ℕ
  | 0 => 10  -- Add a case for 0
  | 1 => 10  -- Add a case for 1
  | 2 => 10  -- Add cases for 2 to 9
  | 3 => 10
  | 4 => 10
  | 5 => 10
  | 6 => 10
  | 7 => 10
  | 8 => 10
  | 9 => 10
  | 10 => 10
  | n+1 => 100 * a n + (n+1)

theorem least_multiple_of_99 :
  (∀ k, 10 < k → k < 45 → a k % 99 ≠ 0) ∧ (a 45 % 99 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_99_l639_63981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_dice_game_l639_63974

def win_amount (roll : Nat) : ℚ :=
  if roll = 1 then -3
  else if Nat.Prime roll then roll
  else 0

def expected_value (win_func : Nat → ℚ) (max_roll : Nat) : ℚ :=
  (Finset.sum (Finset.range max_roll) (fun i => win_func (i + 1))) / max_roll

theorem expected_value_dice_game :
  expected_value win_amount 6 = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_dice_game_l639_63974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l639_63923

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem interval_of_increase 
  (φ : ℝ) 
  (h1 : |φ| < π) 
  (h2 : ∀ x : ℝ, f x φ ≤ |f (π/6) φ|) 
  (h3 : f (π/2) φ < f (π/3) φ) :
  ∃ k : ℤ, ∀ x : ℝ, 
    StrictMono (fun y => f y φ) ↔ 
      x ∈ Set.Icc (k * π - π/3) (k * π + π/6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l639_63923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l639_63901

-- Define the circles
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the point P
variable (P : ℝ × ℝ)

-- Define the condition that P is on the radical axis
def on_radical_axis (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 + y^2 - 1 = (x - 3)^2 + (y - 4)^2 - 4

-- Define the distance function
noncomputable def distance_to_origin (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem min_distance_to_origin :
  ∀ P, on_radical_axis P → 
  (∀ Q, on_radical_axis Q → distance_to_origin P ≤ distance_to_origin Q) →
  distance_to_origin P = 11/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l639_63901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_properties_l639_63995

-- Define the line l and circle C
noncomputable def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - (m^2 + 1) * y = 4 * m
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 4 * y + 16 = 0

-- Define the slope of line l
noncomputable def slope_l (m : ℝ) : ℝ := 1 / (m^2 + 1)

-- Theorem statement
theorem line_circle_properties (m : ℝ) :
  (∀ k : ℝ, slope_l m = k → -1 ≤ k ∧ k ≤ 1) ∧
  ¬∃ (θ : ℝ), 
    (∀ x y : ℝ, line_l m x y → circle_C x y → 
      (θ / (2 * Real.pi - θ) = Real.sqrt 3 ∨ 
       (2 * Real.pi - θ) / θ = Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_properties_l639_63995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bin_game_solution_l639_63961

/-- The number of green balls in the bin -/
def green_balls : ℕ := 8

/-- The number of purple balls in the bin -/
def k : ℕ → ℕ := id

/-- The win amount for drawing a green ball -/
def green_win : ℤ := 5

/-- The loss amount for drawing a purple ball -/
def purple_loss : ℤ := 3

/-- The expected value of the game -/
def expected_value : ℚ := 1

/-- Theorem stating that if the expected value is 1, then k must be 8 -/
theorem bin_game_solution (k_pos : k n > 0) (n : ℕ) :
  (green_balls : ℚ) / (green_balls + k n : ℚ) * green_win +
  (k n : ℚ) / (green_balls + k n : ℚ) * (-purple_loss) = expected_value →
  k n = 8 := by
    sorry

#check bin_game_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bin_game_solution_l639_63961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l639_63942

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} := by
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l639_63942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_first_second_instantaneous_velocity_at_one_time_velocity_14_l639_63977

-- Define the distance function
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3 + x^2 + 2*x

-- Define the velocity function (derivative of f)
noncomputable def v (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 2

-- Theorem for average velocity during the first second
theorem average_velocity_first_second :
  f 1 - f 0 = 3/2 := by sorry

-- Theorem for instantaneous velocity at t = 1
theorem instantaneous_velocity_at_one :
  v 1 = 6 := by sorry

-- Theorem for time when velocity reaches 14 m/s
theorem time_velocity_14 :
  ∃ t : ℝ, t = 2 ∧ v t = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_first_second_instantaneous_velocity_at_one_time_velocity_14_l639_63977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2020_less_than_reciprocal_l639_63998

/-- A sequence defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 1  -- We set a₀ = 1 as a placeholder; any positive real number would work
  | n + 1 => a n / Real.sqrt (1 + 2020 * (a n)^2)

/-- The theorem to prove -/
theorem a_2020_less_than_reciprocal : a 2020 < 1 / 2020 := by
  sorry

/-- Auxiliary lemma: a₀ is positive -/
lemma a0_pos : a 0 > 0 := by
  -- This follows directly from our definition of a 0
  exact Real.zero_lt_one


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2020_less_than_reciprocal_l639_63998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_for_bag1_is_three_l639_63931

/-- Represents a query about the parity of balls in 15 bags -/
def Query := Finset (Fin 100)

/-- The result of a query is either odd or even -/
inductive QueryResult
| odd
| even

/-- A function that determines the result of a query -/
def queryFunction : Query → QueryResult := sorry

/-- The minimum number of queries needed to determine the parity of bag 1 -/
def minQueriesForBag1 : ℕ := 3

/-- Theorem stating that at least 3 queries are needed to determine the parity of bag 1 -/
theorem min_queries_for_bag1_is_three :
  minQueriesForBag1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_for_bag1_is_three_l639_63931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l639_63911

/-- A tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- The given tetrahedron with specified edge lengths -/
def given_tetrahedron : Tetrahedron where
  PQ := 42
  PR := 8
  PS := 19
  QR := 37
  QS := 28
  RS := 14

theorem tetrahedron_edge_length (t : Tetrahedron) 
  (h1 : t.PQ = 42)
  (h2 : Finset.toSet {t.PQ, t.PR, t.PS, t.QR, t.QS, t.RS} = Finset.toSet {8, 14, 19, 28, 37, 42}) :
  t.RS = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l639_63911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l639_63965

-- Define the curves
noncomputable def curve1 (t : ℝ) : ℝ × ℝ := (2 - t * Real.sin (30 * Real.pi / 180), -1 + t * Real.sin (30 * Real.pi / 180))

noncomputable def curve2 (θ : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos θ, 2 * Real.sqrt 2 * Real.sin θ)

-- Define the intersection points
noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_distance : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l639_63965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l639_63907

-- Define the base of the logarithm
noncomputable def base : ℝ := 0.2

-- Define a, b, and c
noncomputable def a : ℝ := Real.log 2 / Real.log base
noncomputable def b : ℝ := Real.log 3 / Real.log base
noncomputable def c : ℝ := 2 ^ base

-- Theorem statement
theorem order_of_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l639_63907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_entrance_fee_l639_63973

/-- Park entrance fee problem -/
theorem park_entrance_fee (adult_fee child_fee total : ℕ) 
  (ha : adult_fee = 20)
  (hc : child_fee = 15)
  (ht : total = 2400)
  (h_at_least_one : ∃ (a c : ℕ), a ≥ 1 ∧ c ≥ 1 ∧ a * adult_fee + c * child_fee = total) :
  ∃ (a c : ℕ), a ≥ 1 ∧ c ≥ 1 ∧ a * adult_fee + c * child_fee = total ∧ 
    ∀ (a' c' : ℕ), a' ≥ 1 → c' ≥ 1 → a' * adult_fee + c' * child_fee = total →
      |((a : ℚ) / c) - 1| ≤ |((a' : ℚ) / c') - 1| ∧ a = 69 ∧ c = 68 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_entrance_fee_l639_63973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_interchange_l639_63924

theorem two_digit_number_interchange (x y : ℤ) : 
  0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ y = 2 * x ∧ (x + y) - (y - x) = 8 →
  abs ((10 * x + y) - (10 * y + x)) = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_interchange_l639_63924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_is_12_l639_63933

-- Define the ellipse
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  tangent_x : Bool
  tangent_y : Bool

-- Define our specific ellipse
noncomputable def our_ellipse : Ellipse :=
  { foci1 := (4, -6 + 2 * Real.sqrt 5)
    foci2 := (4, -6 - 2 * Real.sqrt 5)
    tangent_x := true
    tangent_y := true }

-- Define the length of the major axis
def major_axis_length (e : Ellipse) : ℝ := sorry

-- Theorem statement
theorem major_axis_length_is_12 : 
  major_axis_length our_ellipse = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_is_12_l639_63933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_condition_l639_63908

/-- A function f is a power function if it has the form f(x) = c * x^α for some constants c and α -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c α : ℝ), ∀ x, f x = c * (x ^ α)

/-- The function f(x) = (m^2 - m - 1)x^m -/
noncomputable def f (m : ℝ) : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * (x ^ m)

theorem power_function_condition (m : ℝ) :
  isPowerFunction (f m) → m^2 - m - 2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_condition_l639_63908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_arithmetic_progression_length_l639_63934

/-- A permutation of positive integers -/
def Permutation := ℕ+ → ℕ+

/-- Checks if a sequence forms an arithmetic progression with an odd common difference -/
def IsOddArithmeticProgression (seq : Finset ℕ+) : Prop :=
  ∃ (d : ℕ+), Odd d.val ∧ ∀ (i j : ℕ+), i < j → i ∈ seq → j ∈ seq →
    ∃ (k : ℕ), seq.toList.get! j.val.pred - seq.toList.get! i.val.pred = k * d

/-- The main theorem -/
theorem greatest_arithmetic_progression_length :
  (∀ (p : Permutation), ∃ (seq : Finset ℕ+), seq.card = 3 ∧ IsOddArithmeticProgression seq) ∧
  ¬(∀ (p : Permutation), ∃ (seq : Finset ℕ+), seq.card = 4 ∧ IsOddArithmeticProgression seq) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_arithmetic_progression_length_l639_63934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dynamics_value_is_7_l639_63929

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 2
  | 2 => 1
  | 3 => 2
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -1
  | _ => 0

def letter_position (c : Char) : ℕ :=
  match c with
  | 'd' => 4
  | 'y' => 25
  | 'n' => 14
  | 'a' => 1
  | 'm' => 13
  | 'i' => 9
  | 'c' => 3
  | 's' => 19
  | _ => 0

def word_value (word : List Char) : ℤ :=
  (word.map (λ c ↦ alphabet_value (letter_position c))).sum

theorem dynamics_value_is_7 : word_value ['d', 'y', 'n', 'a', 'm', 'i', 'c', 's'] = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dynamics_value_is_7_l639_63929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l639_63996

theorem congruence_solutions_count :
  (Finset.filter (fun x : ℕ => x > 0 ∧ x < 70 ∧ (x : ℤ) + 20 ≡ 45 [ZMOD 26]) (Finset.range 70)).card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l639_63996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_monotonic_intervals_impl_b_gt_one_l639_63941

/-- A cubic function with parameter b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -4/3 * x^3 + (b-1) * x

/-- The derivative of f with respect to x -/
noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := -4 * x^2 + (b-1)

/-- Theorem: If f has three monotonic intervals, then b > 1 -/
theorem three_monotonic_intervals_impl_b_gt_one (b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f' b x₁ = 0 ∧ f' b x₂ = 0) → b > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_monotonic_intervals_impl_b_gt_one_l639_63941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenters_concyclic_l639_63959

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the quadrilateral
def Quadrilateral (A B C D : Point) : Prop :=
  sorry

-- Define the inscribed property
def Inscribed (A B C D : Point) : Prop :=
  sorry

-- Define the orthocenter
noncomputable def Orthocenter (A B C : Point) : Point :=
  sorry

-- Define Pa, Pc based on orthocenters involving diagonal AC
noncomputable def Pa (A B C D : Point) : Point :=
  sorry

noncomputable def Pc (A B C D : Point) : Point :=
  sorry

-- Define Pb, Pd based on orthocenters involving diagonal BD
noncomputable def Pb (A B C D : Point) : Point :=
  sorry

noncomputable def Pd (A B C D : Point) : Point :=
  sorry

-- Define concyclicity
def Concyclic (P Q R S : Point) : Prop :=
  ∃ (circle : Circle), 
    (circle.center.x - P.x)^2 + (circle.center.y - P.y)^2 = circle.radius^2 ∧
    (circle.center.x - Q.x)^2 + (circle.center.y - Q.y)^2 = circle.radius^2 ∧
    (circle.center.x - R.x)^2 + (circle.center.y - R.y)^2 = circle.radius^2 ∧
    (circle.center.x - S.x)^2 + (circle.center.y - S.y)^2 = circle.radius^2

-- The main theorem
theorem orthocenters_concyclic 
  (A B C D : Point) 
  (h1 : Quadrilateral A B C D) 
  (h2 : Inscribed A B C D) : 
  Concyclic (Pa A B C D) (Pb A B C D) (Pc A B C D) (Pd A B C D) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenters_concyclic_l639_63959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_circle_center_l639_63950

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -4)

/-- The point we're calculating the distance to -/
def point : ℝ × ℝ := (5, 4)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_to_circle_center : 
  distance circle_center point = 2 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_circle_center_l639_63950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_value_l639_63920

/-- The coefficient of x^2 in the expansion of (x^2 - 3x + 2)^4 -/
def coefficient_x_squared : ℤ :=
  (Nat.choose 4 1) * 2^3 + (Nat.choose 4 2) * 3^2 * 2^2

theorem coefficient_x_squared_value : coefficient_x_squared = 248 := by
  rfl

#eval coefficient_x_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_value_l639_63920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l639_63972

/-- The circle centered at (-1, 2) with radius 5 -/
def myCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

/-- The line y = 3x -/
def myLine (x y : ℝ) : Prop := y = 3 * x

/-- The chord formed by the intersection of the line and the circle -/
def myChord : Set (ℝ × ℝ) := {p | myCircle p.1 p.2 ∧ myLine p.1 p.2}

theorem chord_length : 
  ∃ p q : ℝ × ℝ, p ∈ myChord ∧ q ∈ myChord ∧ p ≠ q ∧ 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l639_63972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l639_63947

theorem expression_evaluation : 
  -2^2 + 2 * Real.sin (π / 3) + (Real.sqrt 3 - Real.pi)^0 - abs (1 - Real.sqrt 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l639_63947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l639_63982

noncomputable def hyperbola_eccentricity (a b : ℝ) (lambda : ℝ) (A B : ℝ × ℝ) (m : ℝ × ℝ) : Set ℝ :=
  let equation := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = lambda
  let on_asymptote := fun (p : ℝ × ℝ) => ∃ (t : ℝ), equation (t * p.1) (t * p.2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AB_length := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let dot_product := AB.1 * m.1 + AB.2 * m.2
  let m_length := Real.sqrt (m.1^2 + m.2^2)
  {e : ℝ | (lambda ≠ 0) ∧
           (on_asymptote A) ∧
           (on_asymptote B) ∧
           (A ≠ B) ∧
           (m = (1, 0)) ∧
           (AB_length = 6) ∧
           (dot_product / m_length = 3) ∧
           (e = 2 ∨ e = 2 * Real.sqrt 3 / 3)}

theorem hyperbola_eccentricity_theorem (a b : ℝ) (lambda : ℝ) (A B : ℝ × ℝ) (m : ℝ × ℝ) :
  ∃ (e : ℝ), e ∈ hyperbola_eccentricity a b lambda A B m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l639_63982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_worth_after_two_years_l639_63962

/-- Represents a quarter's financial changes -/
structure QuarterChange where
  rate : ℝ
  transaction : ℝ

/-- Calculates the final portfolio worth after applying quarterly changes -/
def finalPortfolioWorth (initial : ℝ) (changes : List QuarterChange) : ℝ :=
  changes.foldl
    (fun acc change =>
      let newAmount := acc * (1 + change.rate)
      newAmount + change.transaction)
    initial

/-- Theorem stating the final portfolio worth after two years -/
theorem portfolio_worth_after_two_years :
  let initial : ℝ := 80
  let changes : List QuarterChange := [
    { rate := 0.15, transaction := 0 },
    { rate := 0.05, transaction := 28 },
    { rate := 0.06, transaction := -10 },
    { rate := -0.03, transaction := 0 },
    { rate := 0.10, transaction := 40 },
    { rate := -0.04, transaction := -20 },
    { rate := 0.02, transaction := 12 },
    { rate := -0.07, transaction := -15 }
  ]
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |finalPortfolioWorth initial changes - 132.23| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_worth_after_two_years_l639_63962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_rotation_theorem_example_arrangement_properties_l639_63955

/-- Represents a circular arrangement of seats, guests, and name cards. -/
structure CircularArrangement (n : ℕ) where
  seats : Fin n → Fin n
  guests : Fin n → Fin n
  namecards : Fin n → Fin n

/-- Predicate to check if a guest is sitting in front of their own name card. -/
def isCorrectlySeated (arr : CircularArrangement n) (i : Fin n) : Prop :=
  arr.namecards (arr.seats i) = arr.guests i

/-- Predicate to check if no guest is sitting in front of their own name card. -/
def noOneCorrectlySeated (arr : CircularArrangement n) : Prop :=
  ∀ i : Fin n, ¬(isCorrectlySeated arr i)

/-- Represents a rotation of the circular arrangement. -/
def rotate (arr : CircularArrangement n) (k : Fin n) : CircularArrangement n where
  seats := λ i ↦ (arr.seats i + k : Fin n)
  guests := arr.guests
  namecards := arr.namecards

/-- The main theorem to be proved. -/
theorem circular_arrangement_rotation_theorem :
  ∀ (arr : CircularArrangement 15),
    noOneCorrectlySeated arr →
    ∃ (k : Fin 15), ∃ (i j : Fin 15),
      i ≠ j ∧
      isCorrectlySeated (rotate arr k) i ∧
      isCorrectlySeated (rotate arr k) j :=
by
  sorry

/-- Example of an arrangement where exactly one person is correctly seated and rotation doesn't increase this number. -/
def exampleArrangement : CircularArrangement 15 where
  seats := λ i ↦ i
  guests := λ i ↦ i
  namecards := λ i ↦ (15 - i : Fin 15)

/-- Theorem stating that the example arrangement satisfies the required properties. -/
theorem example_arrangement_properties :
  ∃ (i : Fin 15), isCorrectlySeated exampleArrangement i ∧
  ∀ (k : Fin 15), ∀ (j : Fin 15), j ≠ i → ¬isCorrectlySeated (rotate exampleArrangement k) j :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_rotation_theorem_example_arrangement_properties_l639_63955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l639_63930

/-- A parabola with equation x² = 4y -/
structure Parabola where
  equation : ∀ x y, x^2 = 4*y

/-- A point P(m,1) on the parabola -/
structure PointOnParabola (p : Parabola) where
  m : ℝ
  on_parabola : m^2 = 4

/-- The directrix of the parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  λ y ↦ y = -1

/-- The focus of the parabola -/
def focus (p : Parabola) : ℝ × ℝ :=
  (0, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_properties (p : Parabola) (point : PointOnParabola p) :
  (directrix p (-1)) ∧ 
  (distance (point.m, 1) (focus p) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l639_63930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_101_l639_63949

theorem units_digit_factorial_101 : ∃ n : ℕ, Nat.factorial 101 = 10 * n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_101_l639_63949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_cardinality_l639_63997

/-- The symmetric difference of two sets -/
def symmetricDifference (x y : Finset ℤ) : Finset ℤ :=
  (x \ y) ∪ (y \ x)

/-- Theorem: Given sets x and y of integers, where |x| = 12, |y| = 18, and |x # y| = 18,
    the number of integers in both x and y is 12 -/
theorem intersection_cardinality
  (x y : Finset ℤ)
  (hx : x.card = 12)
  (hy : y.card = 18)
  (hxy : (symmetricDifference x y).card = 18) :
  (x ∩ y).card = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_cardinality_l639_63997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_two_l639_63970

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x) + 2

theorem tangent_line_at_zero_two :
  let P : ℝ × ℝ := (0, 2)
  let tangent_line (x y : ℝ) : Prop := x - y + 2 = 0
  tangent_line P.1 P.2 ∧
  ∀ x y : ℝ, tangent_line x y →
    (y - P.2) = (deriv f P.1) * (x - P.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_two_l639_63970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l639_63904

theorem quadratic_inequality_solution_set :
  {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {(-1/3 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l639_63904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l639_63971

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle x² + (y-4)² = 1 -/
def circle_eq (x y : ℝ) : Prop := x^2 + (y-4)^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The theorem statement -/
theorem min_distance_sum :
  ∃ (min_val : ℝ),
    min_val = Real.sqrt 17 - 1 ∧
    ∀ (x_p y_p x_q y_q : ℝ),
      parabola x_p y_p →
      circle_eq x_q y_q →
      distance x_p y_p x_q y_q + distance x_p y_p 1 0 ≥ min_val :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l639_63971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_are_perpendicular_l639_63935

-- Define the coefficients of the plane equations
def plane1_coeffs : Fin 3 → ℝ := ![2, 3, -4]
def plane2_coeffs : Fin 3 → ℝ := ![5, -2, 1]

-- Define the dot product of two vectors
def dot_product (v1 v2 : Fin 3 → ℝ) : ℝ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1) + (v1 2) * (v2 2)

-- Define the condition for perpendicular planes
def perpendicular_planes (v1 v2 : Fin 3 → ℝ) : Prop :=
  dot_product v1 v2 = 0

-- Theorem statement
theorem planes_are_perpendicular :
  perpendicular_planes plane1_coeffs plane2_coeffs :=
by
  -- Unfold the definitions
  unfold perpendicular_planes
  unfold dot_product
  -- Compute the dot product
  simp [plane1_coeffs, plane2_coeffs]
  -- The result should be 0
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_are_perpendicular_l639_63935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l639_63919

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) → m1 = m2

/-- The first line equation: 3y - 3b = 9x -/
def line1 (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * y - 3 * b = 9 * x

/-- The second line equation: y + 2 = (b + 9)x -/
def line2 (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y + 2 = (b + 9) * x

/-- If the two lines are parallel, then b = -6 -/
theorem parallel_lines_b_value :
  (∃ b : ℝ, ∀ x y : ℝ, (line1 b x y ↔ ∃ k, y = 3 * x + k) ∧ 
                       (line2 b x y ↔ ∃ k, y = (b + 9) * x + k)) →
  ∃ b : ℝ, b = -6 := by
  sorry

#check parallel_lines_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l639_63919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_implies_m_gt_one_l639_63917

noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m - 1) / x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem inverse_proportion_decreasing_implies_m_gt_one (m : ℝ) :
  decreasing_function (inverse_proportion m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_implies_m_gt_one_l639_63917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_bisection_l639_63954

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define membership for Point in Circle
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

instance : Membership Point Circle where
  mem := Point.inCircle

-- Define the problem setup
variable (A B C D M : Point)
variable (circle1 circle2 circle_AB : Circle)

-- Define the conditions
axiom equal_circles : circle1.radius = circle2.radius
axiom A_on_circles : A ∈ circle1 ∧ A ∈ circle2
axiom B_on_circles : B ∈ circle1 ∧ B ∈ circle2
axiom CD_secant : C ∈ circle1 ∧ D ∈ circle2 ∧ A.x < C.x ∧ C.x < D.x ∧ A.y < C.y ∧ C.y < D.y
axiom circle_AB_diameter : circle_AB.center = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩ ∧
                           circle_AB.radius = ((A.x - B.x)^2 + (A.y - B.y)^2)^(1/2) / 2
axiom M_on_CD_and_circle_AB : M ∈ circle_AB ∧ (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
                               M = ⟨C.x + t*(D.x - C.x), C.y + t*(D.y - C.y)⟩)

-- Define the theorem to be proved
theorem secant_bisection :
  M.x = (C.x + D.x) / 2 ∧ M.y = (C.y + D.y) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_bisection_l639_63954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_theorem_l639_63964

-- Define the complex numbers
variable (p q r s t u : ℝ)

-- Define the conditions
def complex_sum_condition (p q r s t u : ℝ) : Prop := p + r + t = 0 ∧ q + s + u = 7
def q_condition (q : ℝ) : Prop := q = 5
def p_condition (p r t : ℝ) : Prop := p = -r - t

-- Theorem statement
theorem complex_sum_theorem 
  (h1 : complex_sum_condition p q r s t u)
  (h2 : q_condition q)
  (h3 : p_condition p r t) :
  s + u = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_theorem_l639_63964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_m_range_l639_63985

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) + x^2 - m * x

-- State the theorem
theorem f_monotonicity_and_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f m x₁ > f m x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ > 0 → x₂ > 0 → f m x₁ < f m x₂) ∧
  (m ∈ Set.Icc (-1 : ℝ) 1 ↔ 
    ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1 : ℝ) 1 → x₂ ∈ Set.Icc (-1 : ℝ) 1 → 
      |f m x₁ - f m x₂| ≤ Real.exp 1 - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_m_range_l639_63985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l639_63951

theorem cos_beta_value (α β : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = -(Real.sqrt 10 / 10))
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  Real.cos β = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l639_63951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_abscissa_sine_l639_63952

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3)

noncomputable def g (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi/3)

theorem double_abscissa_sine (x : ℝ) : g (2 * x) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_abscissa_sine_l639_63952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_coloring_count_l639_63988

/-- The number of faces in a regular octahedron -/
def num_faces : ℕ := 8

/-- The number of colors available -/
def num_colors : ℕ := 8

/-- The number of rotations that produce indistinguishable configurations -/
def num_rotations : ℕ := 3

/-- The number of distinguishable ways to construct a colored regular octahedron -/
def distinguishable_octahedrons : ℕ := (num_colors - 1).factorial / num_rotations

theorem octahedron_coloring_count :
  distinguishable_octahedrons = 1680 := by
  -- Proof goes here
  sorry

#eval distinguishable_octahedrons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_coloring_count_l639_63988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l639_63906

-- Define the vector type
def Vec : Type := ℝ × ℝ

-- Define the dot product
def dot (v w : Vec) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : Vec) : ℝ := Real.sqrt (dot v v)

-- Define the angle between two vectors
noncomputable def angle (v w : Vec) : ℝ := Real.arccos ((dot v w) / (magnitude v * magnitude w))

-- State the theorem
theorem vector_sum_magnitude (a b c : Vec) :
  (magnitude a = 1) →
  (magnitude b = 2) →
  (magnitude c = 3) →
  (angle a b = angle b c) →
  (angle b c = angle c a) →
  (magnitude (a.1 + b.1 + c.1, a.2 + b.2 + c.2) = Real.sqrt 3 ∨
   magnitude (a.1 + b.1 + c.1, a.2 + b.2 + c.2) = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l639_63906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_coeffs_count_l639_63938

/-- Number of 1's in the binary representation of a natural number -/
def numOnes (n : ℕ) : ℕ :=
  (n.digits 2).count 1

/-- The polynomial (x^2 + x + 1)^n -/
noncomputable def u (n : ℕ) : Polynomial ℤ :=
  (Polynomial.X^2 + Polynomial.X + 1)^n

/-- Number of odd coefficients in a polynomial -/
def numOddCoeffs (p : Polynomial ℤ) : ℕ :=
  (p.support.filter (fun i => p.coeff i % 2 ≠ 0)).card

theorem odd_coeffs_count (n : ℕ+) :
  numOddCoeffs (u n) = 2^(numOnes n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_coeffs_count_l639_63938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l639_63958

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2*θ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l639_63958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_sqrt_equation_l639_63932

theorem integer_pair_sqrt_equation :
  ∀ a b : ℕ+,
    (Real.sqrt ((a.val * b.val : ℚ) / (2 * b.val^2 - a.val)) = (a.val + 2 * b.val : ℚ) / (4 * b.val)) ↔
    ((a = 72 ∧ b = 18) ∨ (a = 72 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_sqrt_equation_l639_63932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_identities_l639_63944

theorem algebraic_identities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt ((a + b + c) / (1/a + 1/b + 1/c)) * Real.sqrt ((b*c + a*c + a*b) / (1/(b*c) + 1/(a*c) + 1/(a*b))) = a * b * c) ∧
  (Real.sqrt ((a + b + c) / (1/a + 1/b + 1/c)) / Real.sqrt ((b*c + a*c + a*b) / (1/(b*c) + 1/(a*c) + 1/(a*b))) = (a + b + c) / (a*b + a*c + b*c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_identities_l639_63944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendance_l639_63902

-- Define the days of the week
inductive Day
  | Mon | Tues | Wed | Thurs | Fri | Sat

-- Define the team members
inductive Member
  | Alice | Bob | Charlie | Diana | Eva

-- Define the availability function
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Mon => true
  | Member.Alice, Day.Fri => true
  | Member.Bob, Day.Tues => true
  | Member.Bob, Day.Wed => true
  | Member.Bob, Day.Thurs => true
  | Member.Charlie, Day.Mon => true
  | Member.Charlie, Day.Tues => true
  | Member.Charlie, Day.Thurs => true
  | Member.Charlie, Day.Fri => true
  | Member.Diana, Day.Mon => true
  | Member.Diana, Day.Fri => true
  | Member.Eva, Day.Thurs => true
  | Member.Eva, Day.Fri => true
  | Member.Eva, Day.Sat => true
  | _, _ => false

-- Define the function to count available members on a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Charlie, Member.Diana, Member.Eva]).length

-- Theorem statement
theorem max_attendance :
  (∀ d : Day, availableCount d ≤ 3) ∧
  (availableCount Day.Mon = 3) ∧
  (availableCount Day.Thurs = 3) ∧
  (availableCount Day.Fri = 3) ∧
  (∀ d : Day, availableCount d = 3 → (d = Day.Mon ∨ d = Day.Thurs ∨ d = Day.Fri)) := by
  sorry

#eval availableCount Day.Mon
#eval availableCount Day.Tues
#eval availableCount Day.Wed
#eval availableCount Day.Thurs
#eval availableCount Day.Fri
#eval availableCount Day.Sat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendance_l639_63902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_nh3_oxidized_is_0_34_l639_63903

/-- Represents the reaction between chlorine and ammonia -/
structure ChlorineAmmoniaReaction where
  initial_volume : ℝ
  initial_cl2_fraction : ℝ
  escaping_volume : ℝ
  escaping_cl2_fraction : ℝ
  molar_volume : ℝ
  nh3_molar_mass : ℝ

/-- Calculates the mass of NH₃ oxidized in the reaction -/
noncomputable def mass_nh3_oxidized (r : ChlorineAmmoniaReaction) : ℝ :=
  let initial_cl2_moles := r.initial_volume * r.initial_cl2_fraction / r.molar_volume
  let escaping_cl2_moles := r.escaping_volume * r.escaping_cl2_fraction / r.molar_volume
  let reacted_cl2_moles := initial_cl2_moles - escaping_cl2_moles
  let nh3_moles := (2/3) * reacted_cl2_moles
  nh3_moles * r.nh3_molar_mass

/-- Theorem stating that the mass of NH₃ oxidized is 0.34 g -/
theorem mass_nh3_oxidized_is_0_34 (r : ChlorineAmmoniaReaction) 
    (h1 : r.initial_volume = 1.12)
    (h2 : r.initial_cl2_fraction = 0.9)
    (h3 : r.escaping_volume = 0.672)
    (h4 : r.escaping_cl2_fraction = 0.5)
    (h5 : r.molar_volume = 22.4)
    (h6 : r.nh3_molar_mass = 17) :
    mass_nh3_oxidized r = 0.34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_nh3_oxidized_is_0_34_l639_63903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equation_l639_63980

/-- Parabola C defined by y = x^2 + 1 -/
def C : ℝ → ℝ := λ x => x^2 + 1

/-- The area enclosed by parabola C and tangent lines from point (s, t) -/
noncomputable def enclosed_area (s t : ℝ) : ℝ :=
  let α := Real.sqrt (s^2 + 1 - t)
  4 * α * (s^2 - s + α^2 / 6)

/-- Theorem stating the relationship between enclosed area and parameters s, t, and a -/
theorem enclosed_area_equation (s t a : ℝ) (h1 : t < 0) (h2 : 0 < a) :
  enclosed_area s t = a ↔ 4 * Real.sqrt (s^2 + 1 - t) * (s^2 - s + (s^2 + 1 - t) / 6) = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equation_l639_63980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l639_63905

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = ↑n}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l639_63905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_audio_tape_thickness_l639_63912

/-- Represents the properties of an audio cassette tape -/
structure AudioTape where
  playTime : ℝ  -- in minutes
  smallestDiameter : ℝ  -- in cm
  largestDiameter : ℝ  -- in cm
  playbackSpeed : ℝ  -- in cm/s

/-- Calculates the thickness of an audio tape given its properties -/
noncomputable def calculateTapeThickness (tape : AudioTape) : ℝ :=
  let totalPlayTime := tape.playTime * 60  -- convert to seconds
  let tapeLength := totalPlayTime * tape.playbackSpeed
  let smallestRadius := tape.smallestDiameter / 2
  let largestRadius := tape.largestDiameter / 2
  ((largestRadius^2 - smallestRadius^2) * Real.pi) / tapeLength

/-- Theorem: The thickness of the described audio tape is approximately 0.00373 mm -/
theorem audio_tape_thickness : 
  let tape : AudioTape := {
    playTime := 60,  -- 2 × 30 minutes
    smallestDiameter := 2,
    largestDiameter := 4.5,
    playbackSpeed := 4.75
  }
  |calculateTapeThickness tape * 10 - 0.00373| < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_audio_tape_thickness_l639_63912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_l639_63900

noncomputable section

open Real

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define point P
def P : ℝ × ℝ := (1, -5)

-- Define line l passing through P with inclination angle π/3
def l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, -5 + (sqrt 3 / 2) * t)

-- Define point C in polar coordinates
def C_polar : ℝ × ℝ := (4, π/2)

-- Define circle C
def circle_C (θ : ℝ) : ℝ × ℝ := (8 * sin θ * cos θ, 8 * sin θ * sin θ)

-- Define the distance function between a point and a line
def distance_point_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

-- Theorem: Line l and circle C are disjoint
theorem line_circle_disjoint :
  distance_point_line (0, 4) (sqrt 3) (-1) (-5 - sqrt 3) > 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_l639_63900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l639_63987

-- Define the function f(x) with parameters a and b
noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∃ (x : ℝ), x = -3 ∧ IsLocalMax (f a b) x ∧ f a b x = 9) →
  (a = 1 ∧ b = -3 ∧
   (∀ x ∈ Set.Icc (-3 : ℝ) 3, f a b x ≤ 9) ∧
   (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = -5/3) ∧
   (∀ x ∈ Set.Icc (-3 : ℝ) 3, f a b x ≥ -5/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l639_63987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_with_product_property_l639_63968

-- Define the property for set A
def has_product_property (A : Set ℕ) : Prop :=
  ∀ S : Set ℕ,
    (Set.Infinite S ∧ ∀ p ∈ S, Nat.Prime p) →
    ∃ (k : ℕ) (m n : ℕ),
      k ≥ 2 ∧
      m ∈ A ∧
      n ∉ A ∧
      (∃ (factors_m factors_n : Finset ℕ),
        factors_m.card = k ∧
        factors_n.card = k ∧
        (∀ p ∈ factors_m, p ∈ S) ∧
        (∀ p ∈ factors_n, p ∈ S) ∧
        m = factors_m.prod id ∧
        n = factors_n.prod id)

-- State the theorem
theorem exists_set_with_product_property :
  ∃ A : Set ℕ, has_product_property A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_with_product_property_l639_63968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_for_two_intersections_l639_63986

-- Define the line and curve
def line (x b : ℝ) : ℝ := x + b

noncomputable def curve (x : ℝ) : ℝ := 2 - Real.sqrt (4 * x - x^2)

-- Define the intersection condition
def intersects_at_two_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ line x₁ b = curve x₁ ∧ line x₂ b = curve x₂

-- Theorem statement
theorem b_range_for_two_intersections :
  ∀ b : ℝ, intersects_at_two_points b ↔ 2 ≤ b ∧ b < 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_for_two_intersections_l639_63986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_A_l639_63967

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 < 4}

-- Define set B
def B : Set ℝ := {x : ℝ | Real.rpow 3 x < 9}

-- Theorem statement
theorem intersection_equals_A : A ∩ B = A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_A_l639_63967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l639_63915

/-- The line l: 2x - y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- The circle C: (x + 1)² + (y - 1)² = 4 -/
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 4

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2 * x - y - 2) / Real.sqrt 5

/-- The maximum distance from any point on circle C to line l -/
theorem max_distance_circle_to_line :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 5 + 2 ∧
  ∀ (x y : ℝ), circle_C x y →
    distance_to_line x y ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l639_63915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l639_63916

/-- Represents the journey with given conditions -/
structure Journey where
  flatRoadSpeed : ℚ
  upMountainSpeed : ℚ
  downMountainSpeed : ℚ
  totalTime : ℚ

/-- Calculates the average speed of the journey -/
def averageSpeed (j : Journey) : ℚ :=
  let flatRoadTime := j.totalTime / 2
  let mountainTime := j.totalTime / 2
  let totalDistance := 2 * (flatRoadTime * j.flatRoadSpeed + mountainTime * j.upMountainSpeed)
  totalDistance / j.totalTime

/-- Theorem: The average speed of the journey is 4 li per hour -/
theorem journey_average_speed :
  let j := Journey.mk 4 3 6 5
  averageSpeed j = 4 := by
  -- Proof goes here
  sorry

#eval averageSpeed (Journey.mk 4 3 6 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l639_63916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_circumcircle_radius_l639_63940

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 equal interior angles. -/
structure RegularPentagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The circumcircle of a polygon is the circle that passes through all of its vertices. -/
noncomputable def circumcircle_radius (p : RegularPentagon) : ℝ :=
  p.side_length / (2 * Real.sin (36 * Real.pi / 180))

/-- Theorem: The radius of the circumcircle of a regular pentagon is equal to its side length divided by 2 * sin(36°). -/
theorem regular_pentagon_circumcircle_radius (p : RegularPentagon) :
    ∃ (R : ℝ), R > 0 ∧ R = circumcircle_radius p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_circumcircle_radius_l639_63940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_reflection_theorem_l639_63969

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of a point on the ellipse -/
def on_ellipse (x y a b : ℝ) : Prop := ellipse x y a b

/-- Definition of a line passing through a point with a given slope -/
def line_through_point (x y x0 y0 k : ℝ) : Prop := y - y0 = k * (x - x0)

/-- Definition of reflection across x-axis -/
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem ellipse_reflection_theorem (a b k : ℝ) :
  a > 0 → b > 0 → a > b →
  on_ellipse 2 0 a b →
  2 * b = a →
  ∀ (x1 y1 x2 y2 : ℝ),
    on_ellipse x1 y1 a b →
    on_ellipse x2 y2 a b →
    line_through_point x1 y1 1 0 k →
    line_through_point x2 y2 1 0 k →
    let (x1', y1') := reflect_x x1 y1
    ∃ (t : ℝ), line_through_point 4 0 x1' y1' ((y2 - y1') / (x2 - x1')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_reflection_theorem_l639_63969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_m_bound_l639_63979

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_recursive : ∀ x : ℝ, f x = (1/3) * f (x + 1)

axiom f_interval : ∀ x : ℝ, -1 < x ∧ x ≤ 0 → f x = x * (x + 1)

theorem largest_m_bound :
  (∃ m : ℝ, (∀ x : ℝ, x ≤ m → f x ≥ -81/16) ∧
    (∀ m' : ℝ, m' > m → ∃ x : ℝ, x ≤ m' ∧ f x < -81/16)) →
  (∃! m : ℝ, (∀ x : ℝ, x ≤ m → f x ≥ -81/16) ∧
    (∀ m' : ℝ, m' > m → ∃ x : ℝ, x ≤ m' ∧ f x < -81/16) ∧
    m = 9/4) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_m_bound_l639_63979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_second_odd_given_first_odd_is_half_l639_63983

def numbers : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Prop := n % 2 = 1

instance : DecidablePred is_odd := fun n => decidable_of_iff
  (n % 2 = 1)
  (by simp [is_odd])

def probability_second_odd_given_first_odd (S : Finset ℕ) : ℚ :=
  let odd_numbers := S.filter is_odd
  let total_numbers := S.card
  let odd_count := odd_numbers.card
  if odd_count > 1 then
    (odd_count - 1) / (total_numbers - 1)
  else
    0

theorem probability_second_odd_given_first_odd_is_half :
  probability_second_odd_given_first_odd numbers = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_second_odd_given_first_odd_is_half_l639_63983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_items_probability_l639_63913

/-- The probability of a defective item -/
def p : ℝ := 0.01

/-- The number of items inspected -/
def n : ℕ := 1100

/-- The maximum number of defective items we're interested in -/
def k : ℕ := 17

/-- The probability that out of n items taken for inspection, no more than k will be defective,
    given a defect rate of p -/
noncomputable def probability_no_more_than_k_defective (p : ℝ) (n : ℕ) (k : ℕ) : ℝ :=
  sorry

/-- A tolerance for approximate equality -/
def tolerance : ℝ := 0.0001

theorem defective_items_probability :
  abs (probability_no_more_than_k_defective p n k - 0.9651) < tolerance := by
  sorry

#eval tolerance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_items_probability_l639_63913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l639_63990

noncomputable section

-- Define the circle O
def circle_O (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2 * Real.sqrt 3)^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a general point on the circle
def point_on_circle (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos t, 2 * Real.sqrt 3 + 2 * Real.sin t)

-- Define the tangency condition
def is_tangent (x y : ℝ) : Prop :=
  ∃ t : ℝ, point_on_circle t = (x, y) ∧
    ((x - 3) * (-2 * Real.sin t) = (y - 0) * (2 * Real.cos t))

-- Theorem statement
theorem tangent_line_equation :
  ∃ A B : ℝ × ℝ,
    is_tangent A.1 A.2 ∧
    is_tangent B.1 B.2 ∧
    ∀ x y : ℝ, (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) →
      x - Real.sqrt 3 * y + 3 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l639_63990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pleasant_park_average_running_time_l639_63927

/-- The average number of minutes run per day by students in Pleasant Park Elementary --/
theorem pleasant_park_average_running_time (g : ℕ) : 
  (3 * g * 18 + g * 12 + g * 9 : ℚ) / (3 * g + g + g) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pleasant_park_average_running_time_l639_63927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_product_permutation_proof_l639_63975

def cyclic_product (xs : List ℕ) : ℕ :=
  match xs with
  | [] => 0
  | x :: tail => List.foldl (λ acc (a, b) => acc + a * b) (x * xs.getLast!) (xs.zip tail)

def max_cyclic_product_permutation : Prop :=
  let perms := List.permutations [1, 2, 3, 6, 7]
  let max_value := perms.map cyclic_product |>.maximum?
  let max_count := (perms.filter (λ p => cyclic_product p = max_value.getD 0)).length
  (max_value = some 75) ∧ (max_count = 10)

theorem max_cyclic_product_permutation_proof : max_cyclic_product_permutation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_product_permutation_proof_l639_63975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_position_l639_63963

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

/-- The line x + y + 5 = 0 -/
def line (z : ℂ) : Prop := z.re + z.im + 5 = 0

theorem complex_number_position (m : ℝ) :
  ((z m).im > 0 ↔ m < -3 ∨ m > 5) ∧
  (line (z m) ↔ m = -3 ∨ m = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_position_l639_63963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_line_slope_l639_63936

noncomputable def slopeAngle (m : ℝ) : ℝ := Real.arctan m * (180 / Real.pi)

theorem line_slope_angle :
  slopeAngle (-1) = 135 := by
  sorry

-- Helper theorem to connect the line equation to its slope
theorem line_slope (x y : ℝ) :
  2 * x + 2 * y - 1 = 0 → -1 = -(2 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_line_slope_l639_63936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l639_63992

theorem factorial_ratio : (11 * 10 * 9 * Nat.factorial 8) / (Nat.factorial 8 * Nat.factorial 3) = 165 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l639_63992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_four_l639_63989

noncomputable def g (x : ℝ) : ℝ :=
  if x < 8 then x^2 - 6 else x - 15

theorem g_composition_equals_four : g (g (g 20)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_four_l639_63989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identity_l639_63943

/-- In a triangle ABC with sides a, b, and c, and angles A, B, and C, 
    the following equation holds. -/
theorem triangle_identity (a b c : ℝ) (A B C : ℝ) 
    (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_angle_sum : A + B + C = Real.pi)
    (h_cos_A : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c))
    (h_cos_B : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c))
    (h_cos_C : Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)) :
  (b - c) / a * (Real.cos (A / 2))^2 + 
  (c - a) / b * (Real.cos (B / 2))^2 + 
  (a - b) / c * (Real.cos (C / 2))^2 = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identity_l639_63943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_7th_term_l639_63928

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_7th_term 
  (seq : ArithmeticSequence) 
  (h1 : sum_n seq 5 = 25) 
  (h2 : seq.a 2 = 3) : 
  seq.a 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_7th_term_l639_63928
