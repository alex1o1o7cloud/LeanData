import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_perimeters_l1247_124775

/-- A circle inscribed in an angle -/
structure InscribedCircle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ
  vertex : EuclideanSpace ℝ (Fin 2)

/-- An arbitrary circle passing through a point and touching another circle externally -/
structure ArbitraryCircle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ
  passingPoint : EuclideanSpace ℝ (Fin 2)
  touchingCircle : InscribedCircle

/-- A triangle formed by the vertex of the angle and two points on its sides -/
structure ConstructedTriangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  inscribedCircle : InscribedCircle
  arbitraryCircle : ArbitraryCircle

/-- The perimeter of a triangle -/
noncomputable def trianglePerimeter (t : ConstructedTriangle) : ℝ :=
  dist t.A t.B + dist t.B t.C + dist t.C t.A

/-- Predicate to check if a circle is externally tangent to another circle -/
def circleExternallyTangent (c₁ c₂ : InscribedCircle) : Prop :=
  dist c₁.center c₂.center = c₁.radius + c₂.radius

/-- Predicate to check if a point lies on a circle -/
def pointOnCircle (p : EuclideanSpace ℝ (Fin 2)) (c : ArbitraryCircle) : Prop :=
  dist p c.center = c.radius

/-- The main theorem statement -/
theorem equal_perimeters (ω : InscribedCircle) (t₁ t₂ : ConstructedTriangle) 
  (h₁ : t₁.inscribedCircle = ω) (h₂ : t₂.inscribedCircle = ω)
  (h₃ : t₁.C = ω.vertex) (h₄ : t₂.C = ω.vertex)
  (h₅ : circleExternallyTangent t₁.arbitraryCircle.touchingCircle ω)
  (h₆ : circleExternallyTangent t₂.arbitraryCircle.touchingCircle ω)
  (h₇ : pointOnCircle t₁.A t₁.arbitraryCircle)
  (h₈ : pointOnCircle t₁.B t₁.arbitraryCircle)
  (h₉ : pointOnCircle t₂.A t₂.arbitraryCircle)
  (h₁₀ : pointOnCircle t₂.B t₂.arbitraryCircle) :
  trianglePerimeter t₁ = trianglePerimeter t₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_perimeters_l1247_124775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_and_average_l1247_124706

-- Define the set of positive numbers
variable (x₁ x₂ x₃ x₄ : ℝ)

-- Define the variance of the original set
noncomputable def variance : ℝ := (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16)

-- Define the average of the new set
noncomputable def new_average : ℝ := (1/4) * ((x₁ + 2) + (x₂ + 2) + (x₃ + 2) + (x₄ + 2))

-- Theorem statement
theorem variance_and_average (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) :
  new_average = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_and_average_l1247_124706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1247_124723

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3*x - 9) * (x - 4) / (x - 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Iic 1 ∪ Set.Ioo 1 3 ∪ Set.Ici 4

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | g x ≥ 0 ∧ x ≠ 1} = solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1247_124723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1247_124774

-- Define the function f(x) = 3^(-x)
noncomputable def f (x : ℝ) : ℝ := 3^(-x)

-- Define the domain
def domain : Set ℝ := { x | -2 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | 1/3 ≤ y ∧ y ≤ 9 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1247_124774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_special_list_l1247_124763

/-- The sum of integers from 1 to n -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n from 1 to 250 appears n times -/
def special_list : List ℕ := sorry

theorem median_of_special_list :
  let total_elements : ℕ := triangular_sum 250
  let median_positions : Fin 2 → ℕ := ![total_elements / 2, total_elements / 2 + 1]
  ∀ i : Fin 2, ∃ n : ℕ, 
    (n ≤ 250) ∧ 
    (triangular_sum (n - 1) < median_positions i) ∧ 
    (median_positions i ≤ triangular_sum n) ∧
    (special_list.get? (median_positions i - 1) = some n) ∧
    n = 177 :=
by sorry

#eval triangular_sum 250

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_special_list_l1247_124763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_square_vertex_l1247_124759

/-- Given a square ABCD with side length 1 and a point P in the same plane,
    if the distances u, v, and w from P to A, B, and C respectively satisfy u^2 + v^2 = w^2,
    then the maximum distance from P to D is 2 + √2. -/
theorem max_distance_to_square_vertex (A B C D P : ℝ × ℝ) (u v w : ℝ) :
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Square ABCD with side length 1
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C D = 1) ∧ (dist D A = 1) →
  -- Counterclockwise order
  (A.1 < B.1) ∧ (A.2 < C.2) ∧ (C.1 > D.1) ∧ (B.2 < D.2) →
  -- Distances from P to A, B, C
  (dist P A = u) ∧ (dist P B = v) ∧ (dist P C = w) →
  -- Given condition
  u^2 + v^2 = w^2 →
  -- Maximum distance from P to D
  (∀ Q : ℝ × ℝ, (dist Q A = u) ∧ (dist Q B = v) ∧ (dist Q C = w) → dist Q D ≤ 2 + Real.sqrt 2) ∧
  (∃ Q : ℝ × ℝ, (dist Q A = u) ∧ (dist Q B = v) ∧ (dist Q C = w) ∧ dist Q D = 2 + Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_square_vertex_l1247_124759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_over_a_l1247_124743

theorem min_b_over_a (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x = Real.log x + (Real.exp 1 - a) * x - b) →
  (∀ x > 0, f x ≤ 0) →
  ∃ m : ℝ, m = -1 / Real.exp 1 ∧ ∀ k : ℝ, k = b / a → k ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_over_a_l1247_124743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_volume_l1247_124789

/-- The volume of a regular triangular prism with a specific cross-section -/
theorem regular_triangular_prism_volume (S : ℝ) : 
  S > 0 →  -- Ensure S is positive
  ∃ V : ℝ, 
    -- V is the volume of a regular triangular prism where:
    -- - A plane passes through a side of the lower base and the opposite vertex of the upper base
    -- - The plane forms a 45° angle with the plane of the lower base
    -- - The area of the cross-section is S
    V = (S * Real.sqrt S * (6 ^ (1/4))) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_volume_l1247_124789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_7_l1247_124730

/-- Line l parameterized by t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + Real.sqrt 3 / 2 * t, 1 / 2 * t)

/-- Circle C -/
def circle_C (p : ℝ × ℝ) : Prop := (p.1 - 2)^2 + p.2^2 = 4

/-- The intersection points of line l and circle C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, line_l t = p ∧ circle_C p}

theorem intersection_distance_is_sqrt_7 :
  ∀ p q, p ∈ intersection_points → q ∈ intersection_points → p ≠ q →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_7_l1247_124730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_sum_l1247_124748

/-- Represents the initial sum of money in dollars -/
def P : ℝ := 0

/-- Represents the original interest rate as a percentage -/
def r : ℝ := 0

/-- Represents the time period in years -/
def t : ℝ := 3

/-- The sum after 3 years at the original interest rate -/
def sum1 : ℝ := 920

/-- The sum after 3 years with interest rate increased by 3% -/
def sum2 : ℝ := 992

/-- The equation for the first sum -/
axiom eq1 : sum1 = P + (P * r * t) / 100

/-- The equation for the second sum with increased interest rate -/
axiom eq2 : sum2 = P + (P * (r + 3) * t) / 100

theorem initial_sum : P = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_sum_l1247_124748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1247_124769

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

theorem intersection_range (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) →
  b > -4/3 ∧ b < 28/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1247_124769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_and_circle_l1247_124756

/-- Curve E: 4/x^2 + 9/y^2 = 1 -/
def E (x y : ℝ) : Prop := 4 / x^2 + 9 / y^2 = 1

/-- Point on curve E -/
structure PointOnE where
  x : ℝ
  y : ℝ
  on_curve : E x y

/-- Quadrilateral ABCD on curve E -/
structure QuadrilateralABCD where
  m : ℝ
  n : ℝ
  m_pos : m > 0
  n_pos : n > 0
  A : PointOnE
  B : PointOnE
  C : PointOnE
  D : PointOnE
  A_on_curve : A.x = m ∧ A.y = n
  B_on_curve : B.x = -m ∧ B.y = n
  C_on_curve : C.x = -m ∧ C.y = -n
  D_on_curve : D.x = m ∧ D.y = -n

/-- Area of quadrilateral ABCD -/
def area_ABCD (q : QuadrilateralABCD) : ℝ := 4 * q.m * q.n

/-- Radius of circumscribed circle of quadrilateral ABCD -/
noncomputable def radius_circumscribed (q : QuadrilateralABCD) : ℝ := Real.sqrt (q.m^2 + q.n^2)

/-- Area of circumscribed circle of quadrilateral ABCD -/
noncomputable def area_circumscribed (q : QuadrilateralABCD) : ℝ := Real.pi * (radius_circumscribed q)^2

theorem min_area_and_circle (q : QuadrilateralABCD) :
  (∀ q' : QuadrilateralABCD, area_ABCD q ≤ area_ABCD q') ∧
  (∀ q' : QuadrilateralABCD, area_circumscribed q ≤ area_circumscribed q') →
  area_ABCD q = 48 ∧ area_circumscribed q = 25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_and_circle_l1247_124756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l1247_124717

noncomputable def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sumFirstN (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_ratio
  (a : ℕ → ℚ)
  (h_arith : arithmeticSequence a)
  (h_ratio : a 8 / a 7 = 13 / 5) :
  sumFirstN a 15 / sumFirstN a 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l1247_124717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1247_124785

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi/3) * Real.cos x - 1/4

theorem f_properties :
  (f (Real.pi/3) = -1/2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 2*Real.pi/3) (k * Real.pi - Real.pi/6), 
    ∀ y ∈ Set.Icc (k * Real.pi - 2*Real.pi/3) (k * Real.pi - Real.pi/6),
    x ≤ y → f x ≤ f y) ∧
  (∀ θ ∈ Set.Icc 0 (Real.pi/2), 
    (∀ x, f (x + θ) = -f (-x + θ)) →
    Set.range (fun x => (f (x + θ))^2) = Set.Icc 0 (1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1247_124785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circular_sectors_area_l1247_124792

/-- The area of the region inside a regular hexagon but outside the circular sectors --/
theorem hexagon_circular_sectors_area (s r θ : ℝ) : 
  s = 8 → r = 4 → θ = 60 → 
  (6 * ((Real.sqrt 3 / 4) * s^2)) - (6 * (θ / 360) * Real.pi * r^2) = 96 * Real.sqrt 3 - 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circular_sectors_area_l1247_124792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_daily_round_trip_l1247_124786

/-- Represents Tony's driving situation --/
structure TonyDriving where
  efficiency : ℚ  -- miles per gallon (changed to ℚ for rational numbers)
  workDays : ℕ    -- days per week
  tankCapacity : ℚ -- gallons (changed to ℚ)
  gasPrice : ℚ     -- dollars per gallon (changed to ℚ)
  totalSpent : ℚ   -- dollars spent in 4 weeks (changed to ℚ)

/-- Calculates the daily round trip distance to work --/
def dailyRoundTrip (t : TonyDriving) : ℚ :=
  (t.totalSpent / t.gasPrice) * t.efficiency / (4 * t.workDays)

/-- Theorem stating that Tony's daily round trip is 50 miles --/
theorem tony_daily_round_trip :
  let t : TonyDriving := {
    efficiency := 25
    workDays := 5
    tankCapacity := 10
    gasPrice := 2
    totalSpent := 80
  }
  dailyRoundTrip t = 50 := by
  -- Proof goes here
  sorry

#eval dailyRoundTrip {
  efficiency := 25
  workDays := 5
  tankCapacity := 10
  gasPrice := 2
  totalSpent := 80
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_daily_round_trip_l1247_124786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1247_124729

-- Define the trapezoid ABCD and points X, Y
structure Trapezoid :=
  (A B C D X Y : ℝ × ℝ)

-- Define the properties of the trapezoid
def is_isosceles_trapezoid (t : Trapezoid) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2

def parallel_sides (t : Trapezoid) : Prop :=
  (t.B.2 - t.C.2) / (t.B.1 - t.C.1) = (t.A.2 - t.D.2) / (t.A.1 - t.D.1)

def x_between_b_and_y (t : Trapezoid) : Prop :=
  (t.B.1 < t.X.1 ∧ t.X.1 < t.Y.1) ∨ (t.B.1 > t.X.1 ∧ t.X.1 > t.Y.1)

def right_angle_at_x (t : Trapezoid) : Prop :=
  (t.B.1 - t.X.1) * (t.C.1 - t.X.1) + (t.B.2 - t.X.2) * (t.C.2 - t.X.2) = 0

def right_angle_at_y (t : Trapezoid) : Prop :=
  (t.A.1 - t.Y.1) * (t.D.1 - t.Y.1) + (t.A.2 - t.Y.2) * (t.D.2 - t.Y.2) = 0

noncomputable def segment_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def area (t : Trapezoid) : ℝ :=
  ((t.A.2 + t.D.2) / 2) * (t.D.1 - t.A.1)

-- State the theorem
theorem trapezoid_area (t : Trapezoid) :
  is_isosceles_trapezoid t →
  parallel_sides t →
  x_between_b_and_y t →
  right_angle_at_x t →
  right_angle_at_y t →
  segment_length t.B t.X = 2 →
  segment_length t.X t.Y = 4 →
  segment_length t.Y t.D = 3 →
  area t = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1247_124729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_domino_placement_l1247_124726

/-- A domino placement on an n × n chessboard is balanced if there exists a positive integer k
    such that each row and column intersects exactly k dominoes. -/
def IsBalancedPlacement (n : ℕ) (placement : Finset (Fin n × Fin n × Bool)) : Prop :=
  ∃ (k : ℕ+), ∀ (i : Fin n),
    (placement.filter (fun p => p.1 = i)).card = k ∧
    (placement.filter (fun p => p.2.1 = i)).card = k

/-- The minimum number of dominoes required for a balanced placement on an n × n chessboard. -/
def MinDominoes (n : ℕ) : ℕ :=
  if 3 ∣ n then 2 * n / 3 else 2 * n

/-- Theorem stating that for all n ≥ 3, there exists a balanced placement of dominoes
    with the minimum number of dominoes. -/
theorem balanced_domino_placement (n : ℕ) (h : n ≥ 3) :
  ∃ (placement : Finset (Fin n × Fin n × Bool)),
    IsBalancedPlacement n placement ∧
    placement.card = MinDominoes n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_domino_placement_l1247_124726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1247_124716

/-- The smallest positive period of f(x) = tan(x/2 - 2) -/
noncomputable def smallest_positive_period : ℝ := 2 * Real.pi

/-- The general form of a tangent function -/
noncomputable def tangent_function (A ω φ x : ℝ) : ℝ := A * Real.tan (ω * x + φ)

/-- The period of the general tangent function -/
noncomputable def tangent_period (ω : ℝ) : ℝ := Real.pi / ω

/-- The specific tangent function f(x) = tan(x/2 - 2) -/
noncomputable def f (x : ℝ) : ℝ := Real.tan (x/2 - 2)

theorem smallest_positive_period_of_f :
  ∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) →
  p ≥ smallest_positive_period :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1247_124716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_sixths_l1247_124700

theorem sec_seven_pi_sixths : 1 / Real.cos (7 * π / 6) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_sixths_l1247_124700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leaf_sides_connected_l1247_124771

open Set Metric

/-- A leaf is the intersection of a finite number of congruent closed circular discs. -/
def Leaf (α : Type*) [MetricSpace α] (centers : Finset α) (radius : ℝ) : Set α :=
  ⋂ c ∈ centers, closedBall c radius

/-- A side of a leaf is the intersection of the leaf with the boundary of any of its defining discs. -/
def LeafSide (α : Type*) [MetricSpace α] (centers : Finset α) (radius : ℝ) (c : α) : Set α :=
  (Leaf α centers radius) ∩ (sphere c radius)

/-- The sides of a leaf are connected. -/
theorem leaf_sides_connected
  {α : Type*} [MetricSpace α] (centers : Finset α) (radius : ℝ) :
  ∀ c ∈ centers, IsConnected (LeafSide α centers radius c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leaf_sides_connected_l1247_124771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1247_124754

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- Calculates the time taken for two trains to cross each other -/
noncomputable def timeToCross (train1 train2 : Train) : ℝ :=
  (train1.length + train2.length) / ((train1.speed + train2.speed) * (5/18))

/-- Theorem stating the time taken for the given trains to cross each other -/
theorem train_crossing_time :
  let train1 : Train := { length := 1500, speed := 90 }
  let train2 : Train := { length := 1000, speed := 75 }
  let crossingTime := timeToCross train1 train2
  abs (crossingTime - 54.55) < 0.01 := by
  sorry

-- Use #eval only for computable functions
-- #eval timeToCross { length := 1500, speed := 90 } { length := 1000, speed := 75 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1247_124754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_narration_distribution_l1247_124708

theorem narration_distribution (total_time : ℝ) (disc_capacity : ℝ) :
  total_time = 480 →
  disc_capacity = 70 →
  total_time > 0 →
  disc_capacity > 0 →
  let num_discs : ℕ := Nat.ceil (total_time / disc_capacity)
  let time_per_disc : ℝ := total_time / (num_discs : ℝ)
  time_per_disc = 480 / (Nat.ceil (480 / 70) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_narration_distribution_l1247_124708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l1247_124720

theorem max_value_x_plus_2y (x y : ℝ) (h : (2 : ℝ)^x + (4 : ℝ)^y = 4) :
  ∃ (max : ℝ), (∀ (a b : ℝ), (2 : ℝ)^a + (4 : ℝ)^b = 4 → a + 2*b ≤ max) ∧ (x + 2*y = max) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l1247_124720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1247_124736

/-- Given a quadratic function f(x) = -x^2 + cx + d where f(x) ≤ 0 
    has the solution [-5, 1] ∪ [7, ∞), the vertex of this parabola 
    is at the point (-2, 9). -/
theorem parabola_vertex (c d : ℝ) : 
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ∈ Set.Ici 7 ∪ Set.Icc (-5) 1) →
  ∃! v : ℝ × ℝ, (v.1 = -2 ∧ v.2 = 9) ∧ 
    ∀ x, -x^2 + c*x + d ≤ -(x - v.1)^2 + v.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1247_124736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_eq_108_l1247_124742

/-- An equilateral triangle with vertices on the hyperbola xy = 4 and centroid at (1, 4) -/
structure EquilateralTriangleOnHyperbola where
  /-- The vertices of the triangle -/
  vertices : Fin 3 → ℝ × ℝ
  /-- The vertices lie on the hyperbola xy = 4 -/
  on_hyperbola : ∀ i, (vertices i).1 * (vertices i).2 = 4
  /-- The triangle is equilateral -/
  equilateral : ∀ i j, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)
  /-- The centroid of the triangle is at (1, 4) -/
  centroid : (1, 4) = (
    ((vertices 0).1 + (vertices 1).1 + (vertices 2).1) / 3,
    ((vertices 0).2 + (vertices 1).2 + (vertices 2).2) / 3
  )

/-- Calculate the area of a triangle given its vertices -/
def area (vertices : Fin 3 → ℝ × ℝ) : ℝ := sorry

/-- The square of the area of the triangle is 108 -/
theorem area_squared_eq_108 (t : EquilateralTriangleOnHyperbola) :
  (area t.vertices) ^ 2 = 108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_eq_108_l1247_124742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1247_124744

theorem expression_equality : (Real.pi - 1) ^ 0 - Real.sqrt 9 + 2 * Real.cos (Real.pi / 4) + (1 / 5) ^ (-1 : ℤ) = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1247_124744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_comparison_theorem_l1247_124704

/-- Parameters for the car comparison problem -/
structure CarParams where
  fuelCapacity : ℝ
  fuelPrice : ℝ
  batteryCapacity : ℝ
  electricityPrice : ℝ
  range : ℝ
  fuelOtherExpenses : ℝ
  newEnergyOtherExpenses : ℝ

/-- Calculate the cost per kilometer for a car -/
noncomputable def costPerKm (params : CarParams) (isFuelCar : Bool) : ℝ :=
  if isFuelCar then
    (params.fuelCapacity * params.fuelPrice) / params.range
  else
    (params.batteryCapacity * params.electricityPrice) / params.range

/-- The main theorem for the car comparison problem -/
theorem car_comparison_theorem (params : CarParams) 
  (h1 : params.fuelCapacity = 40)
  (h2 : params.fuelPrice = 9)
  (h3 : params.batteryCapacity = 60)
  (h4 : params.electricityPrice = 0.6)
  (h5 : costPerKm params true = costPerKm params false + 0.54)
  (h6 : params.fuelOtherExpenses = 4800)
  (h7 : params.newEnergyOtherExpenses = 7500) :
  let newEnergyCost := costPerKm params false
  let fuelCost := costPerKm params true
  let breakEvenMileage := (params.newEnergyOtherExpenses - params.fuelOtherExpenses) / (fuelCost - newEnergyCost)
  (newEnergyCost = 36 / params.range) ∧
  (fuelCost = 0.6 ∧ newEnergyCost = 0.06) ∧
  (breakEvenMileage > 5000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_comparison_theorem_l1247_124704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_point_l1247_124765

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point : Type := ℝ × ℝ

-- Define a function to check if a point is outside a circle
def isOutside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define a function to check if a point is on the circle
def isOnCircle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Main theorem
theorem power_of_point (c : Circle) (A B C D E : Point) :
  isOutside A c →
  isOnCircle B c →
  isOnCircle C c →
  isOnCircle D c →
  isOnCircle E c →
  distance A B * distance A C = distance A D * distance A E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_point_l1247_124765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_alphas_l1247_124725

noncomputable def Q (x : ℂ) : ℂ := ((x^24 - 1) / (x - 1))^2 - x^23

theorem sum_of_first_five_alphas :
  ∀ (α : ℕ → ℝ) (r : ℕ → ℝ),
    (∀ k, k ≥ 1 → k ≤ 46 → 0 < α k ∧ α k < 1) →
    (∀ k, k ≥ 1 → k ≤ 46 → r k > 0) →
    (∀ k, k ≥ 1 → k < 46 → α k ≤ α (k+1)) →
    (∀ k, k ≥ 1 → k ≤ 46 → 
      Q (r k * (Complex.exp (2 * Real.pi * Complex.I * α k))) = 0) →
    α 1 + α 2 + α 3 + α 4 + α 5 = 121 / 575 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_alphas_l1247_124725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_usual_time_l1247_124703

/-- Proves that given a man who walks at 75% of his usual speed and takes 40 minutes more than usual to cover a distance, his usual time to cover this distance is 120 minutes. -/
theorem mans_usual_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → 
  usual_time > 0 → 
  (0.75 * usual_speed) * (usual_time + 40) = usual_speed * usual_time → 
  usual_time = 120 := by
  intros h_speed_pos h_time_pos h_equation
  -- The proof steps would go here
  sorry

#check mans_usual_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_usual_time_l1247_124703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cubed_l1247_124713

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 5 * x)

-- State the theorem
theorem inverse_g_cubed :
  (Function.invFun g 5) ^ (-3 : ℝ) = -125 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cubed_l1247_124713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_sum_inequality_l1247_124781

theorem lcm_sum_inequality (a b c d e : ℕ) 
  (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (1 : ℚ) / (Nat.lcm a b) + (1 : ℚ) / (Nat.lcm b c) + (1 : ℚ) / (Nat.lcm c d) + (1 : ℚ) / (Nat.lcm d e) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_sum_inequality_l1247_124781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l1247_124784

theorem circle_intersection (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x^2 + (y - 1)^2 = a^2 ∧ (x - 2)^2 + y^2 = 4 → y = 2*x) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l1247_124784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_mn_value_l1247_124755

/-- A circle tangent to positive half-axes with center at distance √2 from y = -x -/
def CircleC : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- A line with equation x/m + y/n = 1 -/
def LineL (m n : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 / m + p.2 / n = 1}

/-- The line is tangent to the circle -/
def IsTangent (m n : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ CircleC ∩ LineL m n ∧
    ∀ q : ℝ × ℝ, q ∈ CircleC ∩ LineL m n → q = p

theorem min_mn_value (m n : ℝ) (hm : m > 2) (hn : n > 2) (ht : IsTangent m n) :
    m * n ≥ 6 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_mn_value_l1247_124755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connection_characterization_l1247_124779

/-- Given m points, where each point is connected to l other points,
    this theorem characterizes the possible values of l. -/
theorem connection_characterization (m : ℕ) (l : ℕ) 
  (connect_count : Fin m → ℕ) : 
  (∀ (point : Fin m), connect_count point = l) →
  (l < m ∧ Even (l * m)) :=
by
  sorry

/-- The number of connections for a given point -/
def example_connect_count (m : ℕ) : Fin m → ℕ :=
  fun _ ↦ 0  -- Placeholder implementation

#check connection_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connection_characterization_l1247_124779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_lipstick_price_l1247_124795

/-- Represents the cost of beauty products purchased by Maddie -/
structure BeautyPurchase where
  palette_price : ℚ
  palette_count : ℕ
  lipstick_count : ℕ
  hair_color_price : ℚ
  hair_color_count : ℕ
  total_paid : ℚ

/-- Calculates the cost of each lipstick given the beauty purchase information -/
def lipstick_price (purchase : BeautyPurchase) : ℚ :=
  (purchase.total_paid - 
   (purchase.palette_price * purchase.palette_count + 
    purchase.hair_color_price * purchase.hair_color_count)) / 
   purchase.lipstick_count

/-- Theorem stating that given Maddie's purchase information, each lipstick costs $2.50 -/
theorem maddie_lipstick_price :
  let purchase : BeautyPurchase := {
    palette_price := 15,
    palette_count := 3,
    lipstick_count := 4,
    hair_color_price := 4,
    hair_color_count := 3,
    total_paid := 67
  }
  lipstick_price purchase = 5/2 := by sorry

#eval lipstick_price {
  palette_price := 15,
  palette_count := 3,
  lipstick_count := 4,
  hair_color_price := 4,
  hair_color_count := 3,
  total_paid := 67
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_lipstick_price_l1247_124795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_l1247_124719

theorem truncated_cone_volume (a α : ℝ) (ha : a > 0) (hα : 0 < α ∧ α < π / 2) :
  let R : ℝ := a * Real.sin α / 2
  let V : ℝ := (1/3) * Real.pi * (R * Real.sin α)^2 * (2 * R * (1 + Real.cos α))
  V = (Real.pi * a^3 * Real.sin α^5 * Real.cos (α/2)^2) / 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_l1247_124719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1247_124749

-- Define the function f(x) = a^(-x^2 + 3x + 2)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(-x^2 + 3*x + 2)

-- State the theorem
theorem monotone_increasing_interval 
  (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∃ (l : ℝ), l = 3/2 ∧ 
  ∀ (x y : ℝ), l < x ∧ x < y → f a x < f a y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1247_124749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_extrema_in_interval_l1247_124772

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 2

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem a_range_for_extrema_in_interval (a : ℝ) :
  a > 0 →
  (∀ x : ℝ, f_derivative a x = 0 → -1 < x ∧ x < 1) →
  Real.sqrt 3 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_extrema_in_interval_l1247_124772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_value_l1247_124738

theorem largest_value : 
  (1 / Real.exp 1 > Real.log (Real.sqrt 2)) ∧ 
  (1 / Real.exp 1 > Real.log Real.pi / Real.pi) ∧ 
  (1 / Real.exp 1 > Real.sqrt 10 * Real.log 10 / 20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_value_l1247_124738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_root_eight_over_log_eight_l1247_124796

theorem log_root_eight_over_log_eight :
  (Real.log (8^(1/3))) / (Real.log 8) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_root_eight_over_log_eight_l1247_124796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_midpoint_l1247_124793

-- Define the rotation matrix for 90 degrees clockwise
def rotate90Clockwise (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := (p.1 - center.1, p.2 - center.2)
  (center.1 + y, center.2 - x)

-- Define the translation
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

-- Theorem statement
theorem triangle_transformation_midpoint :
  let B : ℝ × ℝ := (1, 1)
  let I : ℝ × ℝ := (2, 4)
  let G : ℝ × ℝ := (5, 1)
  let B' := translate B (-5) 2
  let G' := translate (rotate90Clockwise G B) (-5) 2
  ((B'.1 + G'.1) / 2, (B'.2 + G'.2) / 2) = (-2, 3) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_midpoint_l1247_124793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_balls_radius_l1247_124702

-- Define the axioms
axiom balls_touch : ℝ → ℝ → ℝ → ℝ → Prop
axiom ball_touches_all : ℝ → ℝ → ℝ → ℝ → ℝ → Prop

theorem touching_balls_radius (r₁ r₂ r₃ r₄ : ℝ) 
  (h₁ : r₁ = 3) (h₂ : r₂ = 3) (h₃ : r₃ = 2) (h₄ : r₄ = 2) 
  (h_touch : balls_touch r₁ r₂ r₃ r₄) : 
  ∃ r : ℝ, r = 6/11 ∧ ball_touches_all r r₁ r₂ r₃ r₄ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_balls_radius_l1247_124702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_l1247_124760

theorem tan_double_angle_specific (α : Real) 
  (h1 : Real.sin α = 4/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.tan (2*α) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_l1247_124760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_13_is_constant_l1247_124767

/-- Arithmetic sequence with common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

/-- The sum a₂ + a₈ + a₁₁ is constant -/
axiom constant_sum (a₁ d : ℝ) : 
  ∃ k, ∀ a₁' d', arithmetic_sequence a₁' d' 2 + arithmetic_sequence a₁' d' 8 + arithmetic_sequence a₁' d' 11 = k

theorem S_13_is_constant :
  ∃ k, ∀ a₁ d, S a₁ d 13 = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_13_is_constant_l1247_124767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_eleven_power_l1247_124710

theorem ones_digit_of_eleven_power (n : ℕ) : (11^n : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_eleven_power_l1247_124710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_valentines_mrs_wong_valentines_l1247_124733

/-- Given the number of Valentines given away and the number left,
    prove that the initial number is their sum. -/
theorem initial_valentines (given_away left : ℕ) :
  given_away + left = given_away + left :=
by rfl

/-- Mrs. Wong's Valentine problem -/
theorem mrs_wong_valentines : 
  8 + 22 = 30 :=
by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_valentines_mrs_wong_valentines_l1247_124733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_sqrt3_l1247_124797

-- Define the angle of inclination for a line
noncomputable def angle_of_inclination (a b : ℝ) : ℝ := Real.arctan (a / b)

-- Theorem statement
theorem line_inclination_sqrt3 :
  angle_of_inclination (Real.sqrt 3) (-1) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_sqrt3_l1247_124797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_properties_l1247_124734

/-- A color type with n colors -/
def Color (n : ℕ) := Fin n

/-- A coloring function that assigns a color to each positive integer -/
def Coloring (n : ℕ) := ℕ → Color n

/-- The property that there are infinitely many numbers of each color -/
def InfinitelyMany (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (color : Color n), ∀ (k : ℕ), ∃ (m : ℕ), m > k ∧ c m = color

/-- The property that the arithmetic mean of two numbers with the same parity
    has a color determined only by the colors of the two numbers -/
def MeanColorDetermined (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (a b x y : ℕ), a % 2 = b % 2 → x % 2 = y % 2 →
    c a = c x → c b = c y → c ((a + b) / 2) = c ((x + y) / 2)

/-- The main theorem about the coloring properties -/
theorem coloring_properties (n : ℕ) (c : Coloring n)
    (h1 : InfinitelyMany n c) (h2 : MeanColorDetermined n c) :
  (∀ (a b : ℕ), a % 2 = b % 2 → c a = c b → c ((a + b) / 2) = c a) ∧
  (∃ (coloring : Coloring n), InfinitelyMany n coloring ∧ MeanColorDetermined n coloring ↔ n % 2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_properties_l1247_124734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_2lnx_at_1_0_l1247_124788

/-- The equation of the tangent line to y = 2ln(x) at (1,0) is y = 2x - 2 -/
theorem tangent_line_2lnx_at_1_0 :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.log x
  let tangent_line : ℝ → ℝ := fun x ↦ 2 * x - 2
  (∀ x, x > 0 → HasDerivAt f (2 / x) x) →
  f 1 = 0 →
  tangent_line = fun x ↦ f 1 + 2 * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_2lnx_at_1_0_l1247_124788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_l1247_124782

/-- The function f(x) = log_a(x^2 - ax + 1) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 1) / Real.log a

/-- The theorem stating the range of a for which f is monotonically increasing on [2, +∞) --/
theorem f_monotone_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  1 < a ∧ a < 5/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_l1247_124782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_theorem_l1247_124722

noncomputable def polynomial (x : ℝ) : ℝ := 1 + x^5 + x^7

theorem coefficient_theorem :
  ∃ (expanded : ℕ → ℝ),
    (expanded 17 = 190) ∧ 
    (expanded 18 = 0) := by
  -- We introduce a function 'expanded' that represents the coefficients
  -- of the expanded polynomial (1 + x^5 + x^7)^20
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_theorem_l1247_124722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_l1247_124766

def valid_subset (S : Finset ℤ) : Prop :=
  ∀ x ∈ S, ∀ y ∈ S, x ≠ 4 * y ∧ y ≠ 4 * x

theorem largest_valid_subset :
  ∃ S : Finset ℤ,
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 150) ∧
    valid_subset S ∧
    S.card = 122 ∧
    ∀ T : Finset ℤ, (∀ n ∈ T, 1 ≤ n ∧ n ≤ 150) → valid_subset T → T.card ≤ 122 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_l1247_124766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lower_than_b_assignments_l1247_124768

-- Define the problem parameters
def total_assignments : ℕ := 60
def goal_percentage : ℚ := 85 / 100
def completed_assignments : ℕ := 40
def current_b_grades : ℕ := 34

-- Define the theorem
theorem max_lower_than_b_assignments :
  let total_b_needed : ℕ := (goal_percentage * total_assignments).ceil.toNat
  let remaining_assignments : ℕ := total_assignments - completed_assignments
  let remaining_b_needed : ℕ := total_b_needed - current_b_grades
  remaining_assignments - remaining_b_needed = 3 :=
by
  -- Unfold the let bindings
  simp [total_assignments, goal_percentage, completed_assignments, current_b_grades]
  -- Perform the calculations
  norm_num
  -- The proof is complete
  rfl

#eval (goal_percentage * total_assignments).ceil.toNat -- Should output 51
#eval total_assignments - completed_assignments -- Should output 20
#eval (goal_percentage * total_assignments).ceil.toNat - current_b_grades -- Should output 17
#eval (total_assignments - completed_assignments) - ((goal_percentage * total_assignments).ceil.toNat - current_b_grades) -- Should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lower_than_b_assignments_l1247_124768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_true_l1247_124741

theorem propositions_true : 
  (∃ x : ℝ, Real.sin x < 1) ∧ (∀ x : ℝ, Real.exp (abs x) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_true_l1247_124741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_theorem_l1247_124745

-- Define the ellipse equation
def ellipse_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focus position
def focus_position : ℝ × ℝ := (3, 0)

-- Define the total distance property
def total_distance : ℝ := 10

-- Theorem statement
theorem ellipse_foci_theorem (a b : ℝ) :
  a > b ∧ b > 0 ∧
  (∀ x y, ellipse_equation a b x y →
    let (fx, fy) := focus_position
    let d1 := Real.sqrt ((x - fx)^2 + (y - fy)^2)
    let d2 := Real.sqrt ((x + fx)^2 + y^2)
    d1 + d2 = total_distance) →
  a = 5 ∧ b = 4 := by
  sorry

#check ellipse_foci_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_theorem_l1247_124745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_existence_l1247_124773

/-- A triangle can be constructed given its perimeter, one side, and the angle opposite to that side 
    if and only if certain conditions are met. -/
theorem triangle_construction_existence 
  (K c : ℝ) (γ : Real) : 
  (∃ (a b : ℝ), 
    a + b + c = K ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    Real.sin γ = c / (2 * (a * b * Real.sin γ / (2 * c))))
  ↔ 
  (K > c ∧ 0 < γ ∧ γ < Real.pi ∧ K > 2 * c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_existence_l1247_124773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_probability_l1247_124794

/-- The duration of the total event in minutes -/
noncomputable def total_duration : ℝ := 60

/-- The duration of the favorable outcome in minutes -/
noncomputable def favorable_duration : ℝ := 15

/-- The probability of an event occurring within the favorable duration -/
noncomputable def probability : ℝ := favorable_duration / total_duration

/-- Theorem: The probability of waiting no more than 15 minutes for an hourly event 
    that can occur at any time within a 60-minute period is equal to 1/4 -/
theorem waiting_probability : probability = 1 / 4 := by
  unfold probability
  unfold favorable_duration
  unfold total_duration
  norm_num
  -- The proof is completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_probability_l1247_124794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_and_monotonicity_l1247_124778

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a + 2) * x + 2 * a * Real.log x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - (a + 2) + 2 * a / x

theorem tangent_parallel_and_monotonicity (a : ℝ) :
  (f_derivative a 1 = 0 → a = 1) ∧
  (∀ x > 0,
    (a ≤ 0 → 
      (x < 2 → f_derivative a x < 0) ∧ 
      (x > 2 → f_derivative a x > 0)) ∧
    (a = 2 → f_derivative a x ≥ 0) ∧
    (0 < a ∧ a < 2 → 
      ((x < a ∨ x > 2) → f_derivative a x > 0) ∧ 
      (a < x ∧ x < 2 → f_derivative a x < 0)) ∧
    (a > 2 → 
      ((x < 2 ∨ x > a) → f_derivative a x > 0) ∧ 
      (2 < x ∧ x < a → f_derivative a x < 0))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_and_monotonicity_l1247_124778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_relations_l1247_124758

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (angle : Line → Plane → ℝ)

-- Theorem statement
theorem geometry_relations 
  (α β : Plane) (m n : Line) :
  -- Statement 1
  (∃ (α β : Plane) (m n : Line), 
    perpendicular m n ∧ perpendicular_plane m α ∧ parallel_line n m ∧ 
    ¬(perpendicular_plane m β)) ∧
  -- Statement 2
  (perpendicular_plane m α → parallel_line n m → perpendicular m n) ∧
  -- Statement 3
  (parallel_plane α β → contains α m → parallel_line m n) ∧
  -- Statement 4
  (parallel_line m n → parallel_plane α β → angle m α = angle n β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_relations_l1247_124758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_is_zero_l1247_124777

theorem coefficient_x4_is_zero :
  let expression := fun x : ℝ => x^3 / 3 - 3 / x^2
  let expansion := fun x => (expression x)^10
  (∀ c : ℝ, expansion = fun x => c * x^4 + (expansion x - c * x^4)) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_is_zero_l1247_124777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1247_124752

/-- The volume of the solid formed by rotating the region enclosed by y = 2^x - 1, x = 0, and y = 1 around the y-axis -/
noncomputable def rotationVolume : ℝ := Real.pi - (2 * Real.pi * (1 - Real.log 2)^2) / (Real.log 2)^2

/-- The curve defining the upper boundary of the region -/
noncomputable def curve (x : ℝ) : ℝ := 2^x - 1

theorem volume_of_rotation (x y : ℝ) :
  0 ≤ x ∧ x ≤ Real.log 2 ∧ 0 ≤ y ∧ y ≤ 1 ∧ y = curve x →
  rotationVolume = Real.pi * ∫ y in Set.Icc 0 1, (Real.log (y + 1) / Real.log 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1247_124752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l1247_124757

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x) + 1

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * (x + Real.pi / 4) - Real.pi / 4)

theorem g_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 8) (5 * Real.pi / 8),
  ∀ y ∈ Set.Icc (Real.pi / 8) (5 * Real.pi / 8),
  x < y → g y < g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l1247_124757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_approx_89_l1247_124761

/-- The side length of the large square in inches -/
noncomputable def large_square_side : ℝ := 120

/-- The total area of the large square in square inches -/
noncomputable def total_area : ℝ := large_square_side ^ 2

/-- The fraction of the total area occupied by each L-shaped region -/
noncomputable def l_shape_fraction : ℝ := 1 / 9

/-- The number of L-shaped regions -/
def num_l_shapes : ℕ := 4

/-- The side length of the center square in inches -/
noncomputable def center_square_side : ℝ := Real.sqrt ((1 - num_l_shapes * l_shape_fraction) * total_area)

theorem center_square_side_approx_89 : 
  ∃ ε > 0, |center_square_side - 89| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_approx_89_l1247_124761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_in_unit_interval_l1247_124712

/-- A polynomial with integer coefficients -/
def IntegerPolynomial (n : ℕ) := Fin (n + 1) → ℤ

/-- The degree of a polynomial -/
def degree (p : IntegerPolynomial n) : ℕ := n

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : IntegerPolynomial n) : ℤ := p (Fin.last n)

/-- Count of roots in an interval -/
noncomputable def rootCount (p : IntegerPolynomial n) (a b : ℝ) : ℕ :=
  sorry

/-- Theorem: Maximum roots of a degree 2022 integer polynomial with leading coefficient 1 in (0,1) -/
theorem max_roots_in_unit_interval
  (p : IntegerPolynomial 2022)
  (h1 : degree p = 2022)
  (h2 : leadingCoefficient p = 1) :
  rootCount p 0 1 ≤ 2021 :=
by
  sorry

#check max_roots_in_unit_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_in_unit_interval_l1247_124712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l1247_124776

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + x - 4

-- Theorem statement
theorem f_has_root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l1247_124776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l1247_124753

/-- Proves the true discount and the difference from claimed discount for a series of discounts --/
theorem discount_calculation (initial_discount additional_discount : ℚ) 
  (h1 : initial_discount = 40/100)
  (h2 : additional_discount = 10/100) :
  let true_discount := initial_discount + additional_discount * (1 - initial_discount)
  let claimed_discount := 1/2
  (true_discount = 46/100) ∧ (claimed_discount - true_discount = 4/100) := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l1247_124753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_f_implies_product_one_l1247_124770

-- Define the function f(x) = |log_3(x)|
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 3|

-- State the theorem
theorem equal_f_implies_product_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  f a = f b → a * b = 1 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_f_implies_product_one_l1247_124770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_trajectory_is_parabola_l1247_124715

/-- The equation of a moving circle -/
def circle_equation (x y θ : ℝ) : Prop :=
  x^2 + y^2 - x * Real.sin (2 * θ) + 2 * Real.sqrt 2 * y * Real.sin (θ + Real.pi/4) = 0

/-- The parametric equations of the circle's center -/
noncomputable def center_trajectory (θ : ℝ) : ℝ × ℝ :=
  (Real.sin θ * Real.cos θ, -(Real.sin θ + Real.cos θ))

/-- The trajectory of the circle's center is part of a parabola -/
theorem center_trajectory_is_parabola :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (∃ θ, (x, y) = center_trajectory θ) →
    (y^2 = a*x^2 + b*x + c ∧ x ∈ Set.Icc (-1/2) (1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_trajectory_is_parabola_l1247_124715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_with_stoppages_l1247_124737

/-- Calculates the speed of a bus including stoppages -/
noncomputable def speed_including_stoppages (speed_excluding_stoppages : ℝ) (stoppage_time : ℝ) : ℝ :=
  let effective_travel_time := 1 - stoppage_time / 60
  let distance := speed_excluding_stoppages * effective_travel_time
  distance

/-- Theorem: Given a bus with a speed of 75 kmph excluding stoppages and
    stopping for 24 minutes per hour, the speed including stoppages is 45 kmph -/
theorem bus_speed_with_stoppages :
  speed_including_stoppages 75 24 = 45 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_with_stoppages_l1247_124737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_l1247_124732

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

-- State the theorem
theorem t_of_f_3 : t (f 3) = Real.sqrt (37 - 5 * Real.sqrt 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_l1247_124732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_rate_ratio_l1247_124762

/-- Represents the rate of a machine in units of work per hour -/
structure Rate where
  value : ℝ

/-- The total amount of work to be done -/
structure TotalWork where
  value : ℝ

/-- Calculates the amount of work done given a rate and time -/
def workDone (rate : Rate) (time : ℝ) : TotalWork :=
  ⟨rate.value * time⟩

theorem machine_rate_ratio 
  (total_work : TotalWork)
  (machine_a_rate : Rate)
  (machine_b_rate : Rate)
  (h1 : workDone machine_a_rate 8 = total_work)
  (h2 : ⟨(workDone machine_a_rate 6).value + (workDone machine_b_rate 8).value⟩ = total_work) :
  machine_b_rate.value / machine_a_rate.value = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_rate_ratio_l1247_124762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l1247_124728

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => (1/2) * sequence_a (n + 1) + 1 / (4 * sequence_a (n + 1))

theorem sequence_a_property (n : ℕ) (h : n > 1) :
  ∃ k : ℕ+, Real.sqrt (2 / (2 * (sequence_a n)^2 - 1)) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l1247_124728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1247_124750

/-- Theorem: Asymptotes of a hyperbola with given properties -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imaginary_axis : 2 * b = 2) 
  (h_focal_distance : 2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 + b^2)) :
  let asymptote := fun (x : ℝ) ↦ b / a * x
  ∀ x : ℝ, (asymptote x = (Real.sqrt 2 / 2) * x) ∧ (asymptote (-x) = -(Real.sqrt 2 / 2) * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1247_124750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_approx_l1247_124721

-- Define the diameter of the circle
def diameter : ℝ := 10.37

-- Define pi (we'll use Lean's built-in pi constant)
noncomputable def π : ℝ := Real.pi

-- Define the function to calculate the area of a circle given its diameter
noncomputable def circle_area (d : ℝ) : ℝ := π * (d / 2)^2

-- Theorem statement
theorem circle_area_approx :
  ∃ ε > 0, |circle_area diameter - 84.448| < ε := by
  sorry

-- Additional lemma to show the actual calculation
lemma circle_area_calc :
  ∃ ε > 0, |circle_area diameter - 84.448| < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_approx_l1247_124721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1247_124727

-- Define the points and vectors
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, -3)
def N : ℝ × ℝ := (5, 1)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the trajectory of C
def trajectory_C (t : ℝ) : ℝ × ℝ :=
  (t * M.1 + (1 - t) * N.1, t * M.2 + (1 - t) * N.2)

-- Define A and B as the intersection points of trajectory_C and parabola
axiom A_def : ∃ t₁ : ℝ, parabola (trajectory_C t₁).1 (trajectory_C t₁).2
axiom B_def : ∃ t₂ : ℝ, ∃ t₁ : ℝ, t₂ ≠ t₁ ∧ parabola (trajectory_C t₂).1 (trajectory_C t₂).2

-- Define P on the x-axis
def P (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define D and E as intersection points of a line through P and the parabola
axiom D_E_def : ∀ m : ℝ, ∃ (k : ℝ) (y₁ y₂ : ℝ), 
  y₁ ≠ y₂ ∧ 
  parabola (k * y₁ + m) y₁ ∧ 
  parabola (k * y₂ + m) y₂

-- Main theorem to prove
theorem main_theorem : 
  -- Part 1: OA ⊥ OB
  (∀ (A B : ℝ × ℝ), A.1 * B.1 + A.2 * B.2 = 0) ∧
  -- Part 2: Existence of P(4,0)
  (∃ (D E : ℝ × ℝ), 
    parabola D.1 D.2 ∧ 
    parabola E.1 E.2 ∧ 
    D.1 * E.1 + D.2 * E.2 = 0 ∧
    P 4 = (4, 0)) ∧
  -- Part 3: Trajectory of circle center
  (∀ (x y : ℝ), (∃ (D E : ℝ × ℝ), 
    parabola D.1 D.2 ∧ 
    parabola E.1 E.2 ∧ 
    x = (D.1 + E.1) / 2 ∧ 
    y = (D.2 + E.2) / 2) → 
  y^2 = 2*x - 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1247_124727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1247_124711

/-- The function f(x) = x / ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

/-- Theorem stating the inequality for 0 < x < 1 -/
theorem f_inequality (x : ℝ) (h : 0 < x) (h' : x < 1) :
  f x < f (x^2) ∧ f (x^2) < (f x)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1247_124711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_probability_l1247_124764

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a red ball from a given container -/
def redProbability (c : Container) : ℚ :=
  c.red / (c.red + c.green)

/-- The set of containers -/
def containers : List Container := 
  [⟨3, 7⟩, ⟨7, 3⟩, ⟨5, 6⟩]

theorem red_ball_probability :
  (containers.map redProbability).sum / containers.length = 16 / 33 := by
  sorry

#eval (containers.map redProbability).sum / containers.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_probability_l1247_124764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_specific_case_l1247_124783

/-- The central angle corresponding to the chord intercepted by a line on a circle -/
noncomputable def central_angle (a b c : ℝ) (r : ℝ) : ℝ :=
  2 * Real.arccos (Real.sqrt (r^2 - ((c / Real.sqrt (a^2 + b^2))^2)) / r)

/-- The theorem stating that the central angle for the given line and circle is 2π/3 -/
theorem central_angle_specific_case :
  central_angle 3 4 5 2 = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_specific_case_l1247_124783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1247_124714

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- The problem setup -/
structure Problem where
  topRow : ArithmeticSequence
  column1 : ArithmeticSequence
  column2 : ArithmeticSequence
  topLeft : ℝ
  col1Second : ℝ
  col1Third : ℝ
  col2Last : ℝ

/-- The theorem statement -/
theorem find_x (p : Problem) 
  (h1 : p.topLeft = 30)
  (h2 : p.col1Second = 20)
  (h3 : p.col1Third = 24)
  (h4 : p.col2Last = -10)
  (h5 : p.topRow.first = p.topLeft)
  (h6 : p.column1.first = p.topLeft)
  (h7 : p.column2.first = p.topRow.first + 6 * p.topRow.diff) :
  p.column2.first = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1247_124714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garment_profit_l1247_124739

-- Define the piecewise function for the actual wholesale price
noncomputable def wholesale_price (x : ℕ) : ℝ :=
  if x ≤ 100 then 60
  else if x ≤ 500 then 62 - (x : ℝ) / 50
  else 0

-- Define the profit function
noncomputable def profit (x : ℕ) : ℝ :=
  (wholesale_price x - 40) * x

-- Theorem statement
theorem garment_profit :
  (∀ x : ℕ, x ≤ 500 → wholesale_price x = if x ≤ 100 then 60 else 62 - (x : ℝ) / 50) ∧
  profit 450 = 5850 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garment_profit_l1247_124739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_tank_volume_l1247_124751

/-- The volume of water in a cylindrical tank -/
noncomputable def water_volume (r h d : ℝ) : ℝ :=
  h * (2 * r^2 * Real.arccos ((r - d) / r) - (r - d) * (2 * r * d - d^2).sqrt)

theorem cylindrical_tank_volume :
  let r : ℝ := 6  -- radius of the tank
  let h : ℝ := 10 -- height of the tank
  let d : ℝ := 3  -- depth of water
  water_volume r h d = 120 * Real.pi - 90 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_tank_volume_l1247_124751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1247_124707

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

-- State the theorem
theorem tangent_line_at_one (x y : ℝ) :
  (∃ (m : ℝ), (y - f 1) = m * (x - 1) ∧ 
   ∀ (h : ℝ), h ≠ 0 → ((f (1 + h) - f 1) / h) = m) →
  x + y - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1247_124707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_is_ellipse_l1247_124701

/-- The circle C with equation (x+3)²+y²=100 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 3)^2 + p.2^2 = 100}

/-- Point B with coordinates (3,0) -/
def point_B : ℝ × ℝ := (3, 0)

/-- A point P on the circle C -/
def point_P : circle_C := sorry

/-- Point M on the perpendicular bisector of BP and on CP -/
def point_M (P : circle_C) : ℝ × ℝ := sorry

/-- The locus of point M is an ellipse -/
theorem locus_of_M_is_ellipse :
  ∀ P : circle_C, ∃ x y : ℝ,
    point_M P = (x, y) ∧ x^2 / 25 + y^2 / 16 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_is_ellipse_l1247_124701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_apples_at_least_15_l1247_124798

/-- Represents the number of apples Harold had initially -/
def initial_apples : ℕ := sorry

/-- Represents the number of people Harold split apples between -/
def num_people : ℕ := 3

/-- Represents the number of apples each person received -/
def apples_per_person : ℕ := 5

/-- Represents the number of leftover apples Harold kept -/
def leftovers : ℕ := sorry

/-- The theorem stating that Harold had at least 15 apples initially -/
theorem harold_apples_at_least_15 :
  initial_apples ≥ num_people * apples_per_person :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_apples_at_least_15_l1247_124798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1247_124787

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/3) * x^3 + x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := x^2 + 1

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 4/3)

/-- The slope of the tangent line at the given point -/
def tangent_slope : ℝ := f' point.1

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := point.2 - tangent_slope * point.1

/-- The x-intercept of the tangent line -/
def x_intercept : ℝ := -y_intercept / tangent_slope

/-- The area of the triangle -/
def triangle_area : ℝ := (1/2) * x_intercept * abs y_intercept

theorem tangent_triangle_area :
  triangle_area = 1/9 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1247_124787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_and_white_probability_l1247_124735

/-- Represents the possible gestures in the "Black and White" game -/
inductive Gesture
| White
| Black

/-- Represents the outcome of a single round in the "Black and White" game -/
structure GameRound where
  a : Gesture
  b : Gesture
  c : Gesture

/-- Determines if person A wins in a given game round -/
def a_wins (round : GameRound) : Prop :=
  (round.a = Gesture.White ∧ round.b = Gesture.Black ∧ round.c = Gesture.Black) ∨
  (round.a = Gesture.Black ∧ round.b = Gesture.White ∧ round.c = Gesture.White)

/-- The set of all possible game rounds -/
def all_rounds : Finset GameRound :=
  sorry

/-- The set of game rounds where A wins -/
def winning_rounds : Finset GameRound :=
  sorry

theorem black_and_white_probability :
  (Finset.card winning_rounds : ℚ) / (Finset.card all_rounds : ℚ) = 1/4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_and_white_probability_l1247_124735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cover_escalator_length_l1247_124705

/-- Calculates the time taken to cover a distance given the total speed and distance. -/
noncomputable def time_to_cover_distance (speed : ℝ) (distance : ℝ) : ℝ :=
  distance / speed

/-- Represents an escalator with a given length and speed. -/
structure Escalator where
  length : ℝ
  speed : ℝ

/-- Represents a person walking on an escalator with a given speed. -/
structure PersonOnEscalator where
  walking_speed : ℝ
  escalator : Escalator

/-- Theorem: The time taken to cover the escalator length is 8 seconds. -/
theorem time_to_cover_escalator_length 
  (e : Escalator) 
  (p : PersonOnEscalator) 
  (h1 : e.length = 160) 
  (h2 : e.speed = 12) 
  (h3 : p.walking_speed = 8) 
  (h4 : p.escalator = e) : 
  time_to_cover_distance (e.speed + p.walking_speed) e.length = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cover_escalator_length_l1247_124705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1247_124799

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + x - 2)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt ((2*x + 6) / (3 - x)) + (x + 2)^0

-- Define the domains of f and g
def A : Set ℝ := {x : ℝ | x^2 + x - 2 ≥ 0}
def B : Set ℝ := {x : ℝ | (2*x + 6) / (3 - x) ≥ 0 ∧ x + 2 ≠ 0}

-- State the theorem
theorem domain_intersection :
  A ∩ B = {x : ℝ | (1 ≤ x ∧ x < 3) ∨ (-3 ≤ x ∧ x < -2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1247_124799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_integer_terms_l1247_124740

def c (n : ℕ+) : ℚ := (4 * n + 31) / (2 * n - 1)

theorem exactly_four_integer_terms :
  ∃! (S : Finset ℕ+), S.card = 4 ∧ ∀ n, n ∈ S ↔ (c n).num % (c n).den = 0 ∧ c n > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_integer_terms_l1247_124740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1247_124718

/-- The function f(x) = (1 + ln x) / x -/
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem f_properties :
  /- 1. The interval of monotonic increase is (0, 1) -/
  (∃ (a b : ℝ), a = 0 ∧ b = 1 ∧ ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∧
  
  /- 2. If f(x) has an extreme value in (a, a + 1/2) where a > 0, then 1/2 < a < 1 -/
  (∀ a : ℝ, a > 0 →
    (∃ x : ℝ, a < x ∧ x < a + 1/2 ∧ 
      (∀ y : ℝ, a < y ∧ y < a + 1/2 → f y ≤ f x)) →
    1/2 < a ∧ a < 1) ∧
  
  /- 3. For all x ≥ 1, f(x) > (2 sin x) / (x + 1) -/
  (∀ x : ℝ, x ≥ 1 → f x > 2 * Real.sin x / (x + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1247_124718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_theorem_l1247_124791

def number_of_handshakes : Fin 11 → ℕ := sorry

theorem handshake_theorem (n : ℕ) (total_handshakes : ℕ) :
  n = 11 →
  total_handshakes = 55 →
  total_handshakes = n * (n - 1) / 2 →
  ∀ boy : Fin n, (n - 1 : ℕ) = number_of_handshakes boy :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_theorem_l1247_124791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l1247_124747

noncomputable def x (t : ℝ) : ℝ := 2 - 2 * Real.exp (2 * t) * Real.cos t + Real.exp (2 * t) * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 1 + 3 * Real.exp (2 * t) * Real.sin t - Real.exp (2 * t) * Real.cos t

noncomputable def x' (t : ℝ) : ℝ := x t + y t - 3
noncomputable def y' (t : ℝ) : ℝ := -2 * x t + 3 * y t + 1

theorem solution_satisfies_system :
  (∀ t, (x' t) = (x t + y t - 3)) ∧
  (∀ t, (y' t) = (-2 * x t + 3 * y t + 1)) ∧
  (x 0 = 0) ∧ (y 0 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l1247_124747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_c_grades_l1247_124746

theorem fraction_of_c_grades (total_students : ℕ) 
  (fraction_a : ℚ) (fraction_b : ℚ) (num_d : ℕ) 
  (h1 : total_students = 100)
  (h2 : fraction_a = 1 / 5)
  (h3 : fraction_b = 1 / 4)
  (h4 : num_d = 5) :
  (total_students : ℚ) - (fraction_a * total_students + fraction_b * total_students + num_d) = (1 / 2) * total_students := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_c_grades_l1247_124746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_seven_equals_four_l1247_124780

-- Define the function g as noncomputable
noncomputable def g : ℝ → ℝ := fun u => (u^2 - 10*u + 37) / 4

-- State the theorem
theorem g_of_seven_equals_four :
  (∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) → g 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_seven_equals_four_l1247_124780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1247_124724

noncomputable def f (x : ℝ) := Real.sqrt (Real.sin x - Real.cos x)

theorem domain_of_f :
  ∀ x : ℝ, (∃ k : ℤ, 2 * k * Real.pi + Real.pi / 4 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 4) ↔
  (0 ≤ Real.sin x - Real.cos x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1247_124724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_and_tan_2alpha_l1247_124790

theorem sin_alpha_and_tan_2alpha 
  (α : ℝ) 
  (h1 : Real.sin α = 4/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  (Real.sin (α - π/4) = (7 * Real.sqrt 2) / 10) ∧ 
  (Real.tan (2*α) = 24/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_and_tan_2alpha_l1247_124790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l1247_124731

theorem segment_length : 
  let endpoints := {x : ℝ | |x - Real.rpow 27 (1/3)| = 4}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |a - b| = 8 ∧ 
    ∀ x : ℝ, x ∈ endpoints → a ≤ x ∧ x ≤ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l1247_124731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_l1247_124709

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) := (x - 3) / Real.sqrt (x - 2)

-- Theorem stating the condition for f to be meaningful
theorem f_meaningful (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_l1247_124709
