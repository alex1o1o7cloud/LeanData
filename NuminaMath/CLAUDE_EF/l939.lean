import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l939_93956

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus
def focus : ℝ × ℝ := (0, 1)

-- Define point A
def point_A : ℝ × ℝ := (-1, 8)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ m : ℝ, m = 9 ∧
    ∀ P : ℝ × ℝ, parabola P.1 P.2 →
      distance P point_A + distance P focus ≥ m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l939_93956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l939_93915

-- Define the circles
def circle1 (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 4
def circle2 (b x y : ℝ) : Prop := x^2 + (y - b)^2 = 1

-- Define the intersection points
def intersect (a b : ℝ) (A B : ℝ × ℝ) : Prop :=
  circle1 a A.1 A.2 ∧ circle1 a B.1 B.2 ∧
  circle2 b A.1 A.2 ∧ circle2 b B.1 B.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem
theorem circle_intersection_range (a b : ℝ) (A B : ℝ × ℝ) :
  intersect a b A B →
  (∃ a', intersect a' b A B ∧ distance A B = 2) →
  -Real.sqrt 3 ≤ b ∧ b ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l939_93915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l939_93966

-- Define the piecewise function g(x)
noncomputable def g (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 5 then 4 * x^2 - 5 else b * x + 3

-- Theorem statement
theorem continuity_condition (b : ℝ) :
  Continuous (g b) ↔ b = 92 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l939_93966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_route_length_l939_93927

/-- The total length of a rectangular bike route -/
theorem bike_route_length 
  (upper_segment1 upper_segment2 upper_segment3 left_segment1 left_segment2 : ℝ) :
  upper_segment1 = 4 →
  upper_segment2 = 7 →
  upper_segment3 = 2 →
  left_segment1 = 6 →
  left_segment2 = 7 →
  2 * (upper_segment1 + upper_segment2 + upper_segment3) +
  2 * (left_segment1 + left_segment2) = 52 := by
  sorry

#check bike_route_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_route_length_l939_93927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_is_1_l939_93980

def concatenated_integers : List ℕ := List.reverse (List.range 100)

def digit_at_position (n : ℕ) : Option ℕ :=
  let digits := concatenated_integers.bind (λ i => i.repr.toList.map Char.toNat)
  digits.get? (n - 1)

theorem digit_150_is_1 : digit_at_position 150 = some 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_is_1_l939_93980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l939_93993

theorem ascending_order : Real.log 20.8 < 0.82 ∧ 0.82 < 20.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l939_93993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l939_93906

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - Real.exp x) / (1 + Real.exp x)

-- Define the function g
noncomputable def g (x : ℝ) : ℤ := floor (f x) + floor (f (-x))

-- State the theorem about the range of g
theorem range_of_g : Set.range g = {0, -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l939_93906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integer_points_can_be_marked_l939_93944

/-- Represents a marked point on the segment -/
structure MarkedPoint where
  coordinate : ℤ
  isMarked : Bool

/-- Represents the segment [0, 2002] with marked points -/
structure Segment where
  points : List MarkedPoint
  d : ℤ
  h_d_coprime : Nat.Coprime d.natAbs 1001

/-- Checks if all integer points on the segment are marked -/
def allIntegerPointsMarked (seg : Segment) : Prop :=
  ∀ i : ℤ, 0 ≤ i ∧ i ≤ 2002 → ∃ p ∈ seg.points, p.coordinate = i ∧ p.isMarked

/-- Represents the operation of marking the midpoint -/
def markMidpoint (seg : Segment) (a b : MarkedPoint) : Option Segment :=
  if a.isMarked ∧ b.isMarked ∧ (a.coordinate + b.coordinate) % 2 = 0 then
    some { seg with 
      points := { coordinate := (a.coordinate + b.coordinate) / 2, isMarked := true } :: seg.points
    }
  else
    none

/-- Helper function to check if a point is in the list -/
def pointInList (p : ℤ) (points : List MarkedPoint) : Prop :=
  ∃ mp ∈ points, mp.coordinate = p

/-- The main theorem to be proved -/
theorem all_integer_points_can_be_marked (seg : Segment) :
  (pointInList 0 seg.points) → (pointInList 2002 seg.points) → (pointInList seg.d seg.points) →
  ∃ (markingSequence : List (MarkedPoint × MarkedPoint)),
    allIntegerPointsMarked (markingSequence.foldl (λ s (a, b) => (markMidpoint s a b).getD s) seg) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integer_points_can_be_marked_l939_93944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l939_93979

/-- The speed formula for the nth mile -/
noncomputable def speed (k : ℝ) (n : ℕ) : ℝ :=
  k / ((n - 1) * 2^(n - 2))

/-- The time taken to traverse the nth mile -/
noncomputable def time (k : ℝ) (n : ℕ) : ℝ :=
  1 / speed k n

theorem nth_mile_time (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℝ, (time k 2 = 2) → (time k n = 2 * (n - 1) * 2^(n - 2)) := by
  sorry

#check nth_mile_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l939_93979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_even_difference_l939_93950

def CircularArrangement (n : ℕ) := Fin n → ℕ

def AlternatingArrangement (arr : CircularArrangement 2010) : Prop :=
  ∀ i : Fin 2010, 
    (arr i < arr (i.succ) ∧ arr (i.succ) > arr (i.succ.succ)) ∨
    (arr i > arr (i.succ) ∧ arr (i.succ) < arr (i.succ.succ))

def ValidArrangement (arr : CircularArrangement 2010) : Prop :=
  (∀ i : Fin 2010, 1 ≤ arr i ∧ arr i ≤ 2010) ∧
  (∀ i j : Fin 2010, i ≠ j → arr i ≠ arr j)

theorem adjacent_even_difference 
  (arr : CircularArrangement 2010) 
  (h1 : AlternatingArrangement arr) 
  (h2 : ValidArrangement arr) : 
  ∃ i : Fin 2010, Even (|Int.ofNat (arr i) - Int.ofNat (arr i.succ)|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_even_difference_l939_93950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_is_six_l939_93902

theorem missing_number_is_six :
  ∃! n : ℕ, (Nat.factorial n - Nat.factorial 4) / Nat.factorial 5 = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_is_six_l939_93902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l939_93943

/-- Hyperbola equation coefficients -/
def a : ℝ := 4
def b : ℝ := -1
def c : ℝ := 8
def d : ℝ := -4
def e : ℝ := -4

/-- Rotation angle in radians -/
noncomputable def θ : ℝ := Real.pi / 4

/-- Hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Rotation matrix -/
noncomputable def rotate (x y : ℝ) : ℝ × ℝ :=
  ((x * Real.cos θ - y * Real.sin θ) / Real.sqrt 2,
   (x * Real.sin θ + y * Real.cos θ) / Real.sqrt 2)

/-- Focus coordinates -/
noncomputable def focus : ℝ × ℝ :=
  ((1 + Real.sqrt 5) / Real.sqrt 2, -3 / Real.sqrt 2)

theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ rotate x y = focus := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l939_93943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_cost_minimum_l939_93960

/-- The cost function for the warehouse location problem -/
noncomputable def ω (x : ℝ) : ℝ := 48 / (x + 1) + 3 * x + 1

/-- The theorem stating the minimum value and location of the cost function -/
theorem warehouse_cost_minimum :
  ∃ (x_min : ℝ), x_min = 3 ∧ 
  (∀ x : ℝ, x > 0 → ω x ≥ ω x_min) ∧
  ω x_min = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_cost_minimum_l939_93960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_point_l939_93942

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem smallest_max_point 
  (A ω φ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 ≤ φ ∧ φ ≤ Real.pi / 2)
  (x₁ x₂ : ℝ)
  (h_adjacent : ∀ x, x₁ < x → x < x₂ → f A ω φ x ≤ f A ω φ x₁)
  (h_highest : f A ω φ x₁ = 4 ∧ f A ω φ x₂ = 4)
  (h_period : |x₁ - x₂| = Real.pi)
  (h_zero : f A ω φ (Real.pi / 3) = 0) :
  ∃ x₀ : ℝ, x₀ = Real.pi / 12 ∧ 
    x₀ > 0 ∧
    (∀ x, f A ω φ x ≤ f A ω φ x₀) ∧
    (∀ y, 0 < y → y < x₀ → ∃ z, f A ω φ z > f A ω φ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_point_l939_93942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_of_equation_l939_93978

/-- The equation 3^(x^2-x-y) + 3^(y^2-y-z) + 3^(z^2-z-x) = 1 has a unique solution (x, y, z) = (1, 1, 1) for real numbers x, y, and z. -/
theorem unique_solution_of_equation :
  ∃! (x y z : ℝ), (3 : ℝ)^(x^2 - x - y) + (3 : ℝ)^(y^2 - y - z) + (3 : ℝ)^(z^2 - z - x) = 1 ∧ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_of_equation_l939_93978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousand_fourth_power_equals_ten_to_x_l939_93963

theorem thousand_fourth_power_equals_ten_to_x (x : ℝ) : (1000 : ℝ)^4 = 10^x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousand_fourth_power_equals_ten_to_x_l939_93963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_polyhedra_l939_93926

-- Define the type for geometric bodies
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cube
  | HexagonalPyramid
  | Sphere
  | Cone
  | TruncatedCone
  | Hemisphere

-- Define what it means for a geometric body to be a polyhedron
def isPolyhedron : GeometricBody → Prop := sorry

-- Define the set of geometric bodies we want to prove are all polyhedra
def polyhedronSet : Set GeometricBody :=
  {GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, 
   GeometricBody.Cube, GeometricBody.HexagonalPyramid}

-- Theorem statement
theorem all_polyhedra : ∀ x ∈ polyhedronSet, isPolyhedron x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_polyhedra_l939_93926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l939_93961

theorem sin_2alpha_value (α : ℝ) 
  (h : (Real.cos (2 * α)) / (Real.cos (α - Real.pi/4)) = Real.sqrt 2 / 2) : 
  Real.sin (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l939_93961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_product_l939_93967

-- Define the two curves
noncomputable def y₁ (x : ℝ) : ℝ := 2 - 1/x
noncomputable def y₂ (x : ℝ) : ℝ := x^3 - x^2 + 2*x

-- Define the derivatives of the curves
noncomputable def y₁' (x : ℝ) : ℝ := 1/x^2
noncomputable def y₂' (x : ℝ) : ℝ := 3*x^2 - 2*x + 2

-- Theorem statement
theorem tangent_slope_product (x₀ : ℝ) : 
  x₀ ≠ 0 → y₁' x₀ * y₂' x₀ = 3 → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_product_l939_93967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_l939_93953

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the line l
def line_l (k : ℝ) (ρ θ : ℝ) : Prop := k*ρ*Real.cos θ - ρ*Real.sin θ - k = 0

-- Define the focus F of the curve C
def focus_F : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)

-- Define the condition that A and B are on both curve C and line l
def intersection_condition (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ ρ₁ ρ₂ θ₁ θ₂,
    curve_C t₁ = A ∧
    curve_C t₂ = B ∧
    line_l k ρ₁ θ₁ ∧
    line_l k ρ₂ θ₂

-- State the theorem
theorem intersection_sum_reciprocal (k : ℝ) (A B : ℝ × ℝ) 
  (h : intersection_condition k A B) :
  1 / dist A focus_F + 1 / dist B focus_F = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_l939_93953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l939_93955

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of a line Ax + By + C = 0 is -A/B when B ≠ 0 -/
noncomputable def line_slope (A B : ℝ) : ℝ := -A / B

theorem parallel_lines_condition (a : ℝ) :
  (∃ b, b ≠ 2 ∧ are_parallel (line_slope a 2) (line_slope 1 (b - 1))) ∧
  (are_parallel (line_slope a 2) (line_slope 1 (a - 1)) → a = 2) :=
by
  sorry

#check parallel_lines_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l939_93955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l939_93916

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 2)

-- Define the domain of the function
def domain : Set ℝ := {x | x ≥ 1 ∧ x ≠ 2}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ∈ domain :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l939_93916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_general_formula_l939_93989

def a : ℕ → ℚ
| 0 => 1
| n + 1 => a n / (2 * a n + 1)

theorem arithmetic_sequence_and_general_formula :
  (∀ n : ℕ, (1 / a (n + 1) - 1 / a n) = 2) ∧
  (∀ n : ℕ, a n = 1 / (2 * n.succ - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_general_formula_l939_93989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_nth_roots_for_reciprocal_exists_bijection_without_nth_root_l939_93909

-- Define the concept of n-th functional root
def is_nth_functional_root {A : Type} (f g : A → A) (n : ℕ) : Prop :=
  ∀ x, (f^[n]) x = g x

-- Part (a)
theorem infinite_nth_roots_for_reciprocal :
  ∀ n : ℕ+, ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧
    ∀ f ∈ S, is_nth_functional_root f (λ x ↦ 1/x) n.val :=
sorry

-- Part (b)
theorem exists_bijection_without_nth_root :
  ∃ h : ℝ → ℝ, Function.Bijective h ∧
    ∀ n : ℕ+, ∀ f : ℝ → ℝ, ¬(is_nth_functional_root f h n.val) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_nth_roots_for_reciprocal_exists_bijection_without_nth_root_l939_93909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_T_l939_93996

noncomputable section

/-- The function g(x) = (3x+4)/(x+3) -/
def g (x : ℝ) : ℝ := (3*x + 4) / (x + 3)

/-- The set T of values assumed by g(x) when x ≥ 0 -/
def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ g x = y}

/-- P is an upper bound of T -/
def P : ℝ := 3

/-- Q is a lower bound of T -/
def Q : ℝ := 4/3

theorem bounds_of_T :
  Q ∈ T ∧ P ∉ T := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_T_l939_93996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moments_of_inertia_l939_93910

/-- The circle defined by x^2 + y^2 ≤ -x -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ -p.1}

/-- The density of the circle -/
def density : ℝ := 1

/-- Moment of inertia relative to the x-axis -/
noncomputable def I_Ox (S : Set (ℝ × ℝ)) : ℝ :=
  ∫ p in S, p.2^2 * density

/-- Moment of inertia relative to the y-axis -/
noncomputable def I_Oy (S : Set (ℝ × ℝ)) : ℝ :=
  ∫ p in S, p.1^2 * density

/-- Moment of inertia relative to the origin -/
noncomputable def I_O (S : Set (ℝ × ℝ)) : ℝ :=
  ∫ p in S, (p.1^2 + p.2^2) * density

theorem moments_of_inertia :
  I_Ox Circle = π / 64 ∧
  I_Oy Circle = 5 * π / 64 ∧
  I_O Circle = 3 * π / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moments_of_inertia_l939_93910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l939_93945

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := 2 - sequence_a n

def T (n : ℕ) : ℝ := sorry

theorem sequence_inequality (l : ℝ) : 
  (∀ n : ℕ+, (S n)^2 - l * (-1)^(n : ℕ) * (T n) ≥ 0) ↔ l = 8/5 := by sorry

#check sequence_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l939_93945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_solution_is_unique_l939_93917

/-- Represents a package style with its total weight and content weights -/
structure PackageStyle where
  totalWeight : ℕ
  weightI : ℕ
  weightII : ℕ

/-- The transport problem setup -/
structure TransportProblem where
  packageA : PackageStyle := { totalWeight := 6, weightI := 3, weightII := 3 }
  packageB : PackageStyle := { totalWeight := 5, weightI := 3, weightII := 2 }
  packageC : PackageStyle := { totalWeight := 5, weightI := 2, weightII := 3 }
  truckCapacity : ℕ := 28
  totalPackages : ℕ := 5

/-- The solution to the transport problem -/
structure TransportSolution where
  numA : ℕ
  numB : ℕ
  numC : ℕ

/-- Theorem stating the uniqueness of the transport solution -/
theorem transport_solution_is_unique (p : TransportProblem) :
  ∃! (s : TransportSolution),
    s.numA + s.numB + s.numC = p.totalPackages ∧
    s.numA ≥ 1 ∧ s.numB ≥ 1 ∧ s.numC ≥ 1 ∧
    s.numA * p.packageA.totalWeight + s.numB * p.packageB.totalWeight + s.numC * p.packageC.totalWeight = p.truckCapacity ∧
    s.numA = 3 ∧ s.numB = 1 ∧ s.numC = 1 := by
  sorry

#check transport_solution_is_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_solution_is_unique_l939_93917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_b_l939_93920

theorem triangle_cosine_b (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b^2 = a * c →
  Real.sin A = (Real.sin (B - A) + Real.sin C) / 2 →
  Real.cos B = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_b_l939_93920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_three_solutions_l939_93970

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else Real.log (x + 1)

-- State the theorem
theorem range_of_m_for_three_solutions :
  ∀ m : ℝ, (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ = x₁ + m ∧ f x₂ = x₂ + m ∧ f x₃ = x₃ + m) →
  m ∈ Set.Ioo (-1/4 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_three_solutions_l939_93970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_minimum_at_pi_fourth_l939_93931

/-- The function y in terms of x and c -/
noncomputable def y (x c : ℝ) : ℝ := 3 * Real.sin (2 * x + c) + 1

/-- The derivative of y with respect to x -/
noncomputable def y_derivative (x c : ℝ) : ℝ := 6 * Real.cos (2 * x + c)

theorem smallest_c_for_minimum_at_pi_fourth (c : ℝ) : 
  c > 0 → -- c is positive
  (∀ x : ℝ, y (π/4) c ≤ y x c) → -- minimum occurs at x = π/4
  (∀ c' : ℝ, c' > 0 → (∀ x : ℝ, y (π/4) c' ≤ y x c') → c' ≥ c) → -- c is the smallest such constant
  c = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_minimum_at_pi_fourth_l939_93931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_reasonable_sample_survey_l939_93913

/-- Represents a sample survey method --/
inductive SampleSurvey
  | park : SampleSurvey
  | hospital : SampleSurvey
  | neighbors : SampleSurvey
  | randomHouseholdRegistry : SampleSurvey

/-- Indicates whether a sample survey has generality --/
def hasGenerality (s : SampleSurvey) : Prop := sorry

/-- Indicates whether a sample survey has representativeness --/
def hasRepresentativeness (s : SampleSurvey) : Prop := sorry

/-- Indicates whether a sample survey is reasonable --/
def isReasonable (s : SampleSurvey) : Prop :=
  hasGenerality s ∧ hasRepresentativeness s

/-- Indicates whether a sample survey uses random sampling --/
def usesRandomSampling (s : SampleSurvey) : Prop := sorry

/-- Indicates whether a sample survey covers a significant portion of the population --/
def coversMeaningfulPortion (s : SampleSurvey) : Prop := sorry

/-- Indicates whether a sample survey uses a comprehensive database --/
def usesComprehensiveDatabase (s : SampleSurvey) : Prop := sorry

theorem most_reasonable_sample_survey :
  (∀ s : SampleSurvey, isReasonable s ↔ (hasGenerality s ∧ hasRepresentativeness s)) →
  (∀ s : SampleSurvey, usesRandomSampling s ∧ coversMeaningfulPortion s ∧ usesComprehensiveDatabase s
    → hasGenerality s ∧ hasRepresentativeness s) →
  usesRandomSampling SampleSurvey.randomHouseholdRegistry →
  coversMeaningfulPortion SampleSurvey.randomHouseholdRegistry →
  usesComprehensiveDatabase SampleSurvey.randomHouseholdRegistry →
  isReasonable SampleSurvey.randomHouseholdRegistry :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_reasonable_sample_survey_l939_93913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_partition_l939_93982

def is_valid_partition (n : ℕ) (A B : Set ℕ) : Prop :=
  A ∪ B = Finset.range n.succ ∧ 
  A ∩ B = ∅ ∧
  (∀ x y, x ∈ A → y ∈ A → (x + y) / 2 ∉ A) ∧
  (∀ x y, x ∈ B → y ∈ B → (x + y) / 2 ∉ B)

theorem largest_valid_partition : 
  (∃ A B : Set ℕ, is_valid_partition 8 A B) ∧ 
  (∀ n > 8, ¬ ∃ A B : Set ℕ, is_valid_partition n A B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_partition_l939_93982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_bound_l939_93949

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q →
    ((a * Real.log (p + 2) - (p + 1)^2) - (a * Real.log (q + 2) - (q + 1)^2)) / (p - q) > 2) →
  a ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_bound_l939_93949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_radius_l939_93930

/-- The radius of a circle that intersects a specific line, given perpendicular radii -/
theorem circle_line_intersection_radius (r : ℝ) (A B : ℝ × ℝ) : 
  r > 0 →
  (3 * A.fst - 4 * A.snd - 1 = 0) →
  (3 * B.fst - 4 * B.snd - 1 = 0) →
  A.fst^2 + A.snd^2 = r^2 →
  B.fst^2 + B.snd^2 = r^2 →
  A.fst * B.fst + A.snd * B.snd = 0 →
  r = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_radius_l939_93930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_set_operation_l939_93954

theorem real_set_operation : 
  (Int.cast '' Set.univ : Set ℝ) ∪ (Set.univ \ (Nat.cast '' Set.univ : Set ℝ)) = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_set_operation_l939_93954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_64_equals_8_l939_93929

theorem power_of_64_equals_8 : (64 : ℝ) ^ (375/1000 : ℝ) * (64 : ℝ) ^ (125/1000 : ℝ) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_64_equals_8_l939_93929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_not_covered_fraction_l939_93952

/-- Floor composed of a square and a right triangle -/
structure Floor where
  square_area : ℚ
  triangle_leg1 : ℚ
  triangle_leg2 : ℚ

/-- Rectangular rug -/
structure Rug where
  length : ℚ
  width : ℚ

/-- Calculate the area of the floor -/
def floor_area (f : Floor) : ℚ :=
  f.square_area + (1/2) * f.triangle_leg1 * f.triangle_leg2

/-- Calculate the area of the rug -/
def rug_area (r : Rug) : ℚ :=
  r.length * r.width

/-- Theorem: The fraction of the floor not covered by the rug is 17/24 -/
theorem floor_not_covered_fraction (f : Floor) (r : Rug) :
  f.square_area = 36 ∧
  f.triangle_leg1 = 6 ∧
  f.triangle_leg2 = 4 ∧
  r.length = 2 ∧
  r.width = 7 ∧
  rug_area r ≤ floor_area f →
  (floor_area f - rug_area r) / floor_area f = 17 / 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_not_covered_fraction_l939_93952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_color_on_circle_l939_93932

/-- Represents a color --/
structure Color where

/-- Represents a point in the plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in the plane --/
structure Circle where
  center : Point
  radius : ℝ

/-- The angle between two points with respect to the origin, measured in radians --/
noncomputable def angle (O A X : Point) : ℝ := sorry

/-- The distance between two points --/
noncomputable def distance (P Q : Point) : ℝ := sorry

/-- Checks if a point lies on a line defined by two other points --/
def onLine (P Q R : Point) : Prop := sorry

/-- The circle C(X) as defined in the problem --/
noncomputable def circleC (O A X : Point) : Circle :=
  { center := O
    radius := distance O X + angle O A X / distance O X }

/-- A coloring of the plane --/
noncomputable def coloringOfPlane : Point → Color := sorry

/-- Check if a point is on the circumference of a circle --/
def onCircumference (P : Point) (C : Circle) : Prop := sorry

/-- The main theorem --/
theorem exists_point_with_color_on_circle
  (O A : Point) (n : ℕ) (h : n > 0) :
  ∃ X : Point, ¬onLine O A X ∧
    ∃ Y : Point, onCircumference Y (circleC O A X) ∧
      coloringOfPlane Y = coloringOfPlane X := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_color_on_circle_l939_93932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_sixth_l939_93957

theorem sin_double_angle_plus_pi_sixth (α : Real) 
  (h1 : Real.cos (α + Real.pi/3) = 1/3) 
  (h2 : 0 < α ∧ α < Real.pi/2) : 
  Real.sin (2*α + Real.pi/6) = 7/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_sixth_l939_93957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_line_max_distance_l939_93937

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 24/5 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y = 0

/-- The maximum distance from a point on the circle to the line -/
noncomputable def max_distance : ℝ := (2 + Real.sqrt 5) / 5

/-- Theorem stating the maximum distance from any point on the circle to the line -/
theorem circle_to_line_max_distance :
  ∀ (x y : ℝ), circle_eq x y →
  (∀ (x' y' : ℝ), line_eq x' y' →
    Real.sqrt ((x - x')^2 + (y - y')^2) ≤ max_distance) ∧
  (∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧ 
    ∃ (x₁ y₁ : ℝ), line_eq x₁ y₁ ∧
      Real.sqrt ((x₀ - x₁)^2 + (y₀ - y₁)^2) = max_distance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_line_max_distance_l939_93937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_parameter_range_l939_93969

theorem function_property_implies_parameter_range 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (hf : ∀ (x : ℝ), x > 0 → f x = a * Real.log x + (x + 1)^2) 
  (h : ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f x₁ - f x₂ ≥ 4 * (x₁ - x₂)) : 
  a ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_parameter_range_l939_93969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_length_is_five_l939_93904

/-- Represents a cistern with given dimensions and water level -/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  total_wet_area : ℝ

/-- Calculates the length of a cistern given its properties -/
noncomputable def calculate_cistern_length (c : Cistern) : ℝ :=
  (c.total_wet_area - 2 * c.width * c.depth) / (c.width + 2 * c.depth)

/-- Theorem stating that for a cistern with given properties, its length is 5 meters -/
theorem cistern_length_is_five :
  ∃ (c : Cistern),
    c.width = 4 ∧
    c.depth = 1.25 ∧
    c.total_wet_area = 42.5 ∧
    calculate_cistern_length c = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_length_is_five_l939_93904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l939_93935

/-- Sequence a_n -/
noncomputable def a (n : ℕ) (t : ℝ) : ℝ := -n + t

/-- Sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := 2^n

/-- Sequence c_n -/
noncomputable def c (n : ℕ) (t : ℝ) : ℝ := (a n t + b n) / 2 + |a n t - b n| / 2

/-- The theorem stating the range of t -/
theorem t_range :
  ∀ t : ℝ, (∀ n : ℕ, n ≥ 1 → c n t ≥ c 3 t) ↔ 10 ≤ t ∧ t ≤ 19 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l939_93935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l939_93986

theorem expansion_terms_count : 
  (Finset.card (Finset.filter (fun (a, b, c) => a + b + c = 11) (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12))))) = 78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l939_93986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_velocity_increase_is_210_l939_93998

/-- Represents the minimum velocity increase required for car A --/
noncomputable def min_velocity_increase (vA vB vC vD : ℝ) (distAC distCD : ℝ) : ℝ :=
  let new_vA := (distAC + distCD) / 2 - vD
  new_vA - vA

/-- Theorem stating the minimum velocity increase for car A --/
theorem min_velocity_increase_is_210 :
  let vA : ℝ := 80
  let vB : ℝ := 50
  let vC : ℝ := 70
  let vD : ℝ := 60
  let distAC : ℝ := 300
  let distCD : ℝ := 400
  min_velocity_increase vA vB vC vD distAC distCD = 210 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval min_velocity_increase 80 50 70 60 300 400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_velocity_increase_is_210_l939_93998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_age_l939_93974

/-- Given the average age of students in a class, the average age of boys, and the ratio of boys to girls,
    calculate the average age of girls. -/
theorem girls_average_age 
  (avg_age : ℝ) 
  (boys_avg_age : ℝ) 
  (ratio : ℝ) 
  (h_avg : avg_age = 15.8)
  (h_boys : boys_avg_age = 16.2)
  (h_ratio : ratio = 1.0000000000000044)
  : ∃ (girls_avg_age : ℝ), abs (girls_avg_age - 15.4) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_age_l939_93974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_with_property_l939_93988

/-- A function that checks if a set of 51 numbers from 1 to n contains a pair summing to 101 -/
def has_pair_sum_101 (n : ℕ) (s : Finset ℕ) : Prop :=
  s.card = 51 ∧ s ⊆ Finset.range n ∧
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ a + b = 101

/-- The property that any set of 51 numbers from 1 to n contains a pair summing to 101 -/
def property (n : ℕ) : Prop :=
  ∀ (s : Finset ℕ), has_pair_sum_101 n s

theorem max_n_with_property :
  (property 100) ∧ (∀ m : ℕ, m > 100 → ¬(property m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_with_property_l939_93988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l939_93921

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nested_radical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))

/-- The nested radical satisfies the equation x = sqrt(3 - x) -/
axiom nested_radical_eq : nested_radical = Real.sqrt (3 - nested_radical)

theorem nested_radical_value : nested_radical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l939_93921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_l939_93947

/-- The area of a regular octagon formed by cutting corners from a square --/
theorem octagon_area (m : ℝ) : 
  ∃ x : ℝ, 
    (2 * m - 2 * x = Real.sqrt 2 * x) ∧ 
    ((2 * m)^2 - 4 * (x^2 / 2)) = 4 * (Real.sqrt 2 - 1) * m^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_l939_93947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l939_93971

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children_per_family : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children_per_family = 3)
  (h3 : childless_families = 3) :
  (((total_families : ℚ) * average_children_per_family) / 
   ((total_families - childless_families) : ℚ)) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l939_93971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l939_93977

theorem set_equality : 
  {x : ℕ | ∃ a : ℕ, x = a^2 + 1} = {y : ℕ | ∃ b : ℕ, y = b^2 - 4*b + 5} := by
  sorry

#check set_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l939_93977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_of_containing_disk_l939_93997

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def R : Set (ℝ × ℝ) := {p : ℝ × ℝ | (floor p.1)^2 + (floor p.2)^2 = 25}

def disk (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 ≤ r^2}

theorem min_radius_of_containing_disk :
  ∃ (r : ℝ), (∀ (p : ℝ × ℝ), p ∈ R → p ∈ disk r) ∧
  (∀ (s : ℝ), s < r → ∃ (q : ℝ × ℝ), q ∈ R ∧ q ∉ disk s) ∧
  r = (9 * Real.sqrt 2) / 2 := by
  sorry

#check min_radius_of_containing_disk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_of_containing_disk_l939_93997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l939_93983

noncomputable def g (x : ℝ) : ℝ := |Int.floor x| - |Int.floor (2 - x)| + x

theorem g_symmetry : ∀ x : ℝ, g x = g (3 - x) := by
  intro x
  simp [g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l939_93983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_coefficients_sum_l939_93941

theorem quadratic_inequality_coefficients_sum (a b : ℝ) :
  (∀ x : ℝ, ax^2 + (a+b)*x + 2 > 0 ↔ x ∈ Set.Ioo (-3) 1) →
  a + b = -4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_coefficients_sum_l939_93941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_A_distance_l939_93968

/-- The distance traveled by Train A when it meets Train B -/
noncomputable def distance_traveled_by_train_A (route_length : ℝ) (time_A : ℝ) (time_B : ℝ) : ℝ :=
  let speed_A := route_length / time_A
  let speed_B := route_length / time_B
  let time_to_meet := route_length / (speed_A + speed_B)
  speed_A * time_to_meet

/-- Theorem stating that Train A travels 75 miles before meeting Train B -/
theorem train_A_distance (route_length : ℝ) (time_A : ℝ) (time_B : ℝ) 
  (h_route : route_length = 200)
  (h_time_A : time_A = 10)
  (h_time_B : time_B = 6) :
  distance_traveled_by_train_A route_length time_A time_B = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_A_distance_l939_93968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_objective_function_range_l939_93933

theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2*y ≥ 2) 
  (h2 : 2*x + y ≤ 4) 
  (h3 : 4*x - y ≥ 1) : 
  19/9 ≤ 3*x + y ∧ 3*x + y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_objective_function_range_l939_93933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fliers_remaining_l939_93962

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h_total : total = 1000)
  (h_morning : morning_fraction = 1 / 5)
  (h_afternoon : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * ↑total).floor - (afternoon_fraction * ↑(total - (morning_fraction * ↑total).floor)).floor = 600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fliers_remaining_l939_93962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_remaining_pairs_l939_93984

theorem max_sum_of_remaining_pairs (a b c d : ℝ) : 
  let sums : Set ℝ := {a + b, a + c, a + d, b + c, b + d, c + d}
  (189 ∈ sums) → (320 ∈ sums) → (287 ∈ sums) → (264 ∈ sums) →
  ∃ (x y : ℝ), x ∈ sums ∧ y ∈ sums ∧ 
  x ≠ 189 ∧ x ≠ 320 ∧ x ≠ 287 ∧ x ≠ 264 ∧
  y ≠ 189 ∧ y ≠ 320 ∧ y ≠ 287 ∧ y ≠ 264 ∧
  x + y ≤ 530 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_remaining_pairs_l939_93984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l939_93903

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x ≥ 2 * y + 16
def equation2 (x y : ℝ) : Prop := x^4 + 2*x^2*y^2 + y^4 + 25 - 26*x^2 - 26*y^2 = 72*x*y

-- Theorem statement
theorem unique_solution :
  ∃! p : ℝ × ℝ, 
    let (x, y) := p
    equation1 x y ∧ equation2 x y ∧ x = 6 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l939_93903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l939_93995

theorem max_value_of_function :
  ∃ (max : Real), max = 1 ∧ ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4 ≤ max :=
by
  -- We'll use 1 as our maximum value
  use 1
  
  -- Split the goal into two parts
  constructor
  
  -- Prove that max = 1
  · rfl
  
  -- Prove the inequality for all x in the interval
  · intro x hx
    
    -- Rewrite the left side of the inequality
    have h1 : Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4 =
              -(Real.cos x - Real.sqrt 3 / 2) ^ 2 + 1 := by
      -- This step requires algebraic manipulation
      sorry
    
    -- Use the fact that (a - b)^2 ≥ 0 for all real a, b
    have h2 : (Real.cos x - Real.sqrt 3 / 2) ^ 2 ≥ 0 := by
      apply pow_two_nonneg
    
    -- Combine the above facts
    calc
      Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4
        = -(Real.cos x - Real.sqrt 3 / 2) ^ 2 + 1 := h1
      _ ≤ -0 + 1 := by linarith [h2]
      _ = 1 := by norm_num

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l939_93995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_and_triangle_verify_perimeter_l939_93985

/-- Given a quadratic equation x^2 - (m+3)x + m + 1 = 0, prove it always has
    two distinct real roots and find the perimeter of an isosceles triangle
    formed by its roots when one root is 4. -/
theorem quadratic_roots_and_triangle (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (m+3)*x₁ + m + 1 = 0 ∧
    x₂^2 - (m+3)*x₂ + m + 1 = 0) ∧
  (∃ x : ℝ, x^2 - (m+3)*x + m + 1 = 0 ∧ x = 4 →
    ∃ y : ℝ, y^2 - (m+3)*y + m + 1 = 0 ∧ y ≠ 4 ∧
    2*4 + y = 26/3) :=
by sorry

/-- The perimeter of the isosceles triangle formed by the roots of the equation
    x^2 - (m+3)x + m + 1 = 0 when one root is 4. -/
noncomputable def triangle_perimeter (m : ℝ) : ℝ := 26/3

/-- Prove that the perimeter of the isosceles triangle is indeed 26/3. -/
theorem verify_perimeter (m : ℝ) :
  (∃ x : ℝ, x^2 - (m+3)*x + m + 1 = 0 ∧ x = 4) →
  triangle_perimeter m = 26/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_and_triangle_verify_perimeter_l939_93985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_length_l939_93912

/-- Represents the properties of a train -/
structure Train where
  speed : ℝ  -- Speed in kmph
  length : ℝ  -- Length in meters

/-- Calculates the relative speed of two trains moving in opposite directions -/
noncomputable def relativeSpeed (t1 t2 : Train) : ℝ :=
  (t1.speed + t2.speed) * 1000 / 3600

/-- Theorem: Given two trains with specific properties, the slower train has a length of 700 meters -/
theorem slower_train_length
  (t1 t2 : Train)
  (h1 : t1.speed = 100 ∨ t1.speed = 120)
  (h2 : t2.speed = 100 ∨ t2.speed = 120)
  (h3 : t1.speed ≠ t2.speed)
  (h4 : t1.length = 500 ∨ t1.length = 700)
  (h5 : t2.length = 500 ∨ t2.length = 700)
  (h6 : t1.length ≠ t2.length)
  (h7 : (t1.length + t2.length) / (relativeSpeed t1 t2) = 19.6347928529354) :
  (if t1.speed < t2.speed then t1.length else t2.length) = 700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_length_l939_93912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l939_93911

noncomputable section

def fixed_cost : ℝ := 20000
def variable_cost_per_unit : ℝ := 100

noncomputable def total_revenue (x : ℝ) : ℝ :=
  if x ≤ 400 then 400 * x - (1/2) * x^2 else 80000

noncomputable def total_cost (x : ℝ) : ℝ := fixed_cost + variable_cost_per_unit * x

noncomputable def profit (x : ℝ) : ℝ := total_revenue x - total_cost x

theorem max_profit_at_300 :
  ∀ x : ℝ, x ≥ 0 → profit 300 ≥ profit x :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l939_93911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_angles_sum_l939_93990

-- Define necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define necessary functions and relations
def DiametricallyOpposite (p q : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2 ∧
  (q.x - c.center.x)^2 + (q.y - c.center.y)^2 = c.radius^2 ∧
  (p.x - q.x)^2 + (p.y - q.y)^2 = 4 * c.radius^2

def ArcMeasure (p q : Point) (c : Circle) : ℝ := sorry

def AngleMeasure (p : Point) : ℝ := sorry

-- Define membership for Point in Circle
instance : Membership Point Circle where
  mem p c := (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Main theorem
theorem circle_angles_sum (A B R D C P : Point) (circle : Circle) 
  (on_circle : A ∈ circle ∧ B ∈ circle ∧ R ∈ circle ∧ D ∈ circle ∧ C ∈ circle)
  (diametric : DiametricallyOpposite B D circle)
  (arc_BR : ArcMeasure B R circle = 72)
  (arc_RD : ArcMeasure R D circle = 108) :
  AngleMeasure P + AngleMeasure R = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_angles_sum_l939_93990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_with_unit_product_roots_l939_93958

theorem infinite_pairs_with_unit_product_roots :
  ∃ f : ℕ → ℤ × ℤ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (a, b) := f n
      ∃ x₁ x₂ : ℝ,
        x₁ ≠ x₂ ∧
        x₁ * x₂ = 1 ∧
        x₁^2012 = a * x₁ + b ∧
        x₂^2012 = a * x₂ + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_with_unit_product_roots_l939_93958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l939_93900

theorem value_of_a (a b c : ℕ) (h : (18 ^ a) * (9 ^ (3*a - 1)) * (c ^ a) = (2 ^ 7) * (3 ^ b)) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l939_93900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_wins_l939_93918

def is_valid_move (old new : ℕ) : Prop :=
  new > old ∧ new - old < old

def winning_sequence (seq : List ℕ) : Prop :=
  seq.head? = some 2 ∧
  seq.getLast? = some 1987 ∧
  ∀ (i : ℕ), i < seq.length - 1 → is_valid_move (seq[i]!) (seq[i+1]!)

theorem starting_player_wins : ∃ (seq : List ℕ), winning_sequence seq ∧ seq.length % 2 = 1 := by
  sorry

#check starting_player_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_wins_l939_93918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_two_x_eq_neg_one_l939_93981

theorem cos_five_two_x_eq_neg_one (x θ : Real) 
  (h1 : Real.sin x = Real.sin θ + Real.cos θ) 
  (h2 : Real.cos x = Real.sin θ * Real.cos θ) : 
  (Real.cos (2 * x))^5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_two_x_eq_neg_one_l939_93981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_roots_l939_93939

-- Define a function to represent a quadratic equation
def quadratic_equation (p q : ℤ) (x : ℤ) : ℤ := x^2 + p*x + q

-- Define a function to check if a quadratic equation has integer roots
def has_integer_roots (p q : ℤ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ quadratic_equation p q x = 0 ∧ quadratic_equation p q y = 0

-- Define a function to increment p and q
def increment (p q : ℤ) : ℤ × ℤ := (p + 1, q + 1)

-- Theorem statement
theorem quadratic_integer_roots :
  has_integer_roots 3 2 ∧
  has_integer_roots 4 3 ∧
  has_integer_roots 5 4 ∧
  has_integer_roots 6 5 ∧
  has_integer_roots 7 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_roots_l939_93939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_subset_count_l939_93994

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 6}

def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2*m + 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ (m < -2 ∨ (0 ≤ m ∧ m ≤ 5/2)) := by sorry

def A_nat : Finset ℕ := Finset.filter (fun x => x ≤ 6) (Finset.range 7)

theorem subset_count : Finset.card (Finset.powerset A_nat) = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_subset_count_l939_93994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harolds_income_l939_93936

def monthly_income : ℝ → Prop := λ _ => True

noncomputable def rent : ℝ := 700
noncomputable def car_payment : ℝ := 300
noncomputable def groceries : ℝ := 50
noncomputable def utilities : ℝ := car_payment / 2
noncomputable def expenses : ℝ := rent + car_payment + utilities + groceries
noncomputable def remaining_after_expenses (income : ℝ) : ℝ := income - expenses
noncomputable def retirement_savings (income : ℝ) : ℝ := (remaining_after_expenses income) / 2
noncomputable def left_after_savings (income : ℝ) : ℝ := remaining_after_expenses income - retirement_savings income

theorem harolds_income :
  ∃ (income : ℝ), monthly_income income ∧ left_after_savings income = 650 := by
  use 2500
  constructor
  · trivial
  · sorry -- The actual calculation would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harolds_income_l939_93936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_plane_l939_93905

noncomputable def point_A : Fin 3 → ℝ := ![2, -1, 2]
noncomputable def normal_vector : Fin 3 → ℝ := ![3, 1, 2]
noncomputable def point_P : Fin 3 → ℝ := ![1, 3, 3/2]

noncomputable def vector_PA : Fin 3 → ℝ := λ i => point_A i - point_P i

def dot_product (v1 v2 : Fin 3 → ℝ) : ℝ :=
  (Finset.univ.sum λ i => v1 i * v2 i)

theorem point_in_plane :
  dot_product vector_PA normal_vector = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_plane_l939_93905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_D_is_periodic_l939_93924

/-- D is a given real-valued function -/
noncomputable def D : ℝ → ℝ := sorry

/-- D is a periodic function -/
theorem D_is_periodic : ∃ (p : ℝ), p ≠ 0 ∧ ∀ (x : ℝ), D (x + p) = D x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_D_is_periodic_l939_93924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l939_93951

-- Define the function f on the open interval (-1, 1)
variable (f : ℝ → ℝ)

-- Define the property that f is decreasing on (-1, 1)
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (h1 : isDecreasing f) (a : ℝ) 
    (h2 : f (1 + a) < f 0) : 
  -1 < a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l939_93951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l939_93964

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x / Real.log a else (2 - a) * x - a / 2

theorem monotonic_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4/3 ≤ a ∧ a < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l939_93964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_yards_lost_l939_93934

/-- Represents the yards lost by a football team -/
def yards_lost (x : ℤ) : Prop := True

/-- Represents the final progress of a football team -/
def final_progress (y : ℤ) : Prop := True

theorem football_team_yards_lost 
  (x : ℤ) 
  (h1 : yards_lost x) 
  (h2 : final_progress 2) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_yards_lost_l939_93934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_to_paint_l939_93946

/-- The area of the wall that needs to be painted, given the dimensions of the wall and the glass painting. -/
theorem area_to_paint (wall_height wall_length glass_height glass_width : ℝ) 
  (h_wall_height : wall_height = 8)
  (h_wall_length : wall_length = 15)
  (h_glass_height : glass_height = 3)
  (h_glass_width : glass_width = 5) :
  wall_height * wall_length - glass_height * glass_width = 105 := by
  -- Substitute the given values
  rw [h_wall_height, h_wall_length, h_glass_height, h_glass_width]
  -- Perform the calculation
  norm_num

#check area_to_paint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_to_paint_l939_93946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l939_93975

/-- Calculates the length of a train given the speeds of two trains, time to cross, and length of the other train -/
noncomputable def trainLength (speed1 speed2 : ℝ) (timeToCross : ℝ) (otherTrainLength : ℝ) : ℝ :=
  let relativeSpeed := (speed1 + speed2) * (5 / 18)
  let totalDistance := relativeSpeed * timeToCross
  totalDistance - otherTrainLength

/-- Theorem stating that under given conditions, the length of the first train is 300 meters -/
theorem first_train_length :
  let speed1 : ℝ := 36
  let speed2 : ℝ := 18
  let timeToCross : ℝ := 46.66293363197611
  let otherTrainLength : ℝ := 400
  trainLength speed1 speed2 timeToCross otherTrainLength = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l939_93975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l939_93999

/-- Given a circle with radius √10, center on y = 2x, and chord length 4√2 on x - y = 0 --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 2 * center.1
  radius_value : radius = Real.sqrt 10
  chord_length : 4 * Real.sqrt 2 = 2 * Real.sqrt (radius^2 - (center.1 - center.2)^2 / 2)

/-- The equation of the circle satisfies one of two possible forms --/
theorem circle_equation (c : Circle) :
  ((c.center.1 - 2)^2 + (c.center.2 - 4)^2 = 10) ∨ 
  ((c.center.1 + 2)^2 + (c.center.2 + 4)^2 = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l939_93999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_radius_and_sun_angle_l939_93919

/-- The radius of the spherical stone -/
noncomputable def stone_radius : ℝ := 5

/-- The angle of elevation of the sun in radians -/
noncomputable def sun_angle : ℝ := Real.arctan (1/4)

/-- The length of the shadow cast by the stone -/
def stone_shadow_length : ℝ := 20

/-- The height of the stick -/
def stick_height : ℝ := 1

/-- The length of the shadow cast by the stick -/
def stick_shadow_length : ℝ := 4

theorem stone_radius_and_sun_angle :
  (stone_radius / stone_shadow_length = stick_height / stick_shadow_length) ∧
  (sun_angle = Real.arctan (stick_height / stick_shadow_length)) ∧
  (stone_radius = 5) ∧
  (sun_angle = Real.arctan (1/4)) := by
  sorry

#check stone_radius_and_sun_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_radius_and_sun_angle_l939_93919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_l939_93923

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90 degrees clockwise around the origin -/
noncomputable def rotate90 (p : Point) : Point :=
  { x := p.y, y := -p.x }

/-- Returns the vertex of a parabola -/
noncomputable def vertex (p : Parabola) : Point :=
  { x := -p.b / (2 * p.a), y := p.c - p.b^2 / (4 * p.a) }

/-- Translates a point by a given vector -/
noncomputable def translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

/-- Theorem: Rotating the parabola y = x^2 - 4x + 3 by 90 degrees clockwise around its vertex
    results in the equation (y+1)^2 = x-2 -/
theorem parabola_rotation (p : Parabola) 
    (h1 : p.a = 1) (h2 : p.b = -4) (h3 : p.c = 3) :
    let v := vertex p
    let rotated (x y : ℝ) := 
      let p' := rotate90 { x := x - v.x, y := y - v.y }
      translate p' v
    ∀ x y, (rotated x y).x = x ∧ (rotated x y).y = y ↔ (y + 1)^2 = x - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_l939_93923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l939_93901

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  focus1 : Point
  focus2 : Point

/-- The length of the major axis of an ellipse -/
noncomputable def majorAxisLength (e : Ellipse) : ℝ :=
  2 * ((e.focus1.x - e.center.x)^2 + (e.focus1.y - e.center.y)^2).sqrt

theorem ellipse_major_axis_length :
  let e : Ellipse := {
    center := { x := 3, y := 0 },
    focus1 := { x := 3, y := 2 + Real.sqrt 2 },
    focus2 := { x := 3, y := -(2 + Real.sqrt 2) }
  }
  ∀ (y : ℝ), y = 0 ∨ y = 3 → ∃ (x : ℝ), (x - 3)^2 / (majorAxisLength e / 2)^2 + y^2 / (3)^2 = 1 →
  majorAxisLength e = 4 + 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l939_93901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_between_15k_and_20k_l939_93914

def total_cars : ℕ := 3000
def percent_less_than_15k : ℚ := 15 / 100
def percent_more_than_20k : ℚ := 40 / 100

theorem cars_between_15k_and_20k :
  let cars_less_than_15k := (percent_less_than_15k * total_cars).floor
  let cars_more_than_20k := (percent_more_than_20k * total_cars).floor
  let cars_outside_range := cars_less_than_15k + cars_more_than_20k
  total_cars - cars_outside_range = 1350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_between_15k_and_20k_l939_93914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circles_l939_93940

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define points on the circles
def M (x y : ℝ) : Prop := C₁ x y
def N (x y : ℝ) : Prop := C₂ x y

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem max_distance_between_circles :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), M x₁ y₁ ∧ N x₂ y₂ ∧
  (∀ (a b c d : ℝ), M a b → N c d → distance x₁ y₁ x₂ y₂ ≥ distance a b c d) ∧
  distance x₁ y₁ x₂ y₂ = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circles_l939_93940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_difference_divisibility_l939_93928

def T : ℕ → ℕ
  | 0 => 2  -- Add this case to handle zero
  | 1 => 2
  | n + 2 => 2^(T (n + 1))

theorem tower_difference_divisibility (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, T n - T (n - 1) = n! * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_difference_divisibility_l939_93928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_plot_cost_per_meter_l939_93987

/-- Represents a rectangular plot with fencing. -/
structure Plot where
  length : ℝ
  breadth : ℝ
  total_cost : ℝ

/-- Calculates the perimeter of a rectangular plot. -/
noncomputable def perimeter (p : Plot) : ℝ := 2 * (p.length + p.breadth)

/-- Calculates the cost per meter of fencing. -/
noncomputable def cost_per_meter (p : Plot) : ℝ := p.total_cost / perimeter p

/-- Theorem stating that for a specific plot, the cost per meter is 26.5. -/
theorem specific_plot_cost_per_meter :
  let p : Plot := { length := 66, breadth := 34, total_cost := 5300 }
  cost_per_meter p = 26.5 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_plot_cost_per_meter_l939_93987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l939_93922

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def SumArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1)) / 2

/-- Theorem: If {a_n} is an increasing arithmetic sequence and {S_n / a_n} is also an arithmetic sequence,
    then S_3 / a_3 = 2 -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_ratio_arith : ArithmeticSequence (fun n ↦ SumArithmeticSequence a n / a n)) :
  SumArithmeticSequence a 3 / a 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l939_93922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_calculation_l939_93938

/-- Calculate the true discount on a sum due in the future. -/
noncomputable def true_discount (sum_due : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  sum_due - sum_due / (1 + interest_rate) ^ years

/-- The true discount on Rs. 768 due in 3 years at 14% per annum is approximately Rs. 249.705. -/
theorem true_discount_calculation :
  let sum_due : ℝ := 768
  let years : ℕ := 3
  let interest_rate : ℝ := 0.14
  abs (true_discount sum_due years interest_rate - 249.705) < 0.001 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval true_discount 768 3 0.14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_calculation_l939_93938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l939_93925

/-- The distance between the foci of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: The distance between the foci of an ellipse with semi-major axis 10 and semi-minor axis 7 is 2√51 -/
theorem ellipse_focal_distance :
  focal_distance 10 7 = 2 * Real.sqrt 51 := by
  -- Unfold the definition of focal_distance
  unfold focal_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l939_93925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_correct_l939_93907

-- Define the propositions
def proposition1 (A B : Real) : Prop :=
  (Real.sin A ≤ Real.sin B) → (A ≤ B)

def proposition2 (x y : Real) : Prop :=
  ((x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5)) ∧ 
  ¬((x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3))

def proposition3 : Prop :=
  (∀ x : Real, x^3 - x^2 + 1 ≤ 0) ↔ 
  ¬(∃ x : Real, x^3 - x^2 + 1 > 0)

-- Theorem stating all propositions are correct
theorem all_propositions_correct : 
  (∀ A B : Real, 0 < A ∧ A < π ∧ 0 < B ∧ B < π → proposition1 A B) ∧ 
  (∀ x y : Real, proposition2 x y) ∧ 
  proposition3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_correct_l939_93907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_odds_l939_93991

/-- Represents the odds against an event occurring -/
structure Odds where
  against : ℕ
  inFavor : ℕ

/-- Calculates the probability of an event given its odds -/
def probability (o : Odds) : ℚ :=
  o.inFavor / (o.against + o.inFavor)

theorem horse_race_odds (oddsA oddsB : Odds) 
  (h1 : oddsA = Odds.mk 5 2) 
  (h2 : oddsB = Odds.mk 3 4) : 
  ∃ (oddsC : Odds), oddsC = Odds.mk 6 1 ∧ 
  probability oddsA + probability oddsB + probability oddsC = 1 := by
  sorry

#eval probability (Odds.mk 5 2)
#eval probability (Odds.mk 3 4)
#eval probability (Odds.mk 6 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_odds_l939_93991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l939_93992

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ
  last_term : ℝ

/-- The number of terms in an arithmetic sequence -/
noncomputable def num_terms (seq : ArithmeticSequence) : ℕ :=
  ⌊(seq.last_term - seq.first_term) / seq.common_difference⌋.toNat + 1

/-- The given arithmetic sequence -/
def given_sequence : ArithmeticSequence :=
  { first_term := 5
    common_difference := 7
    last_term := 126 }

theorem arithmetic_sequence_terms :
  num_terms given_sequence = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l939_93992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_score_domination_l939_93972

/-- Represents a student's scores for three problems -/
structure StudentScores where
  problem1 : Nat
  problem2 : Nat
  problem3 : Nat

/-- Checks if one student's scores dominate another's -/
def dominates (a b : StudentScores) : Prop :=
  a.problem1 ≥ b.problem1 ∧ a.problem2 ≥ b.problem2 ∧ a.problem3 ≥ b.problem3

theorem student_score_domination 
  (students : Finset StudentScores) 
  (h_count : students.card = 49)
  (h_range : ∀ s, s ∈ students → s.problem1 ≤ 7 ∧ s.problem2 ≤ 7 ∧ s.problem3 ≤ 7) :
  ∃ a b, a ∈ students ∧ b ∈ students ∧ a ≠ b ∧ dominates a b := by
  sorry

#check student_score_domination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_score_domination_l939_93972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2a_minus_cos_squared_half_a_l939_93965

theorem sin_2a_minus_cos_squared_half_a (a : Real) 
  (h1 : Real.sin (π - a) = 4/5) 
  (h2 : 0 < a ∧ a < π/2) : 
  Real.sin (2*a) - (Real.cos (a/2))^2 = 4/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2a_minus_cos_squared_half_a_l939_93965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l939_93908

-- Define the function f as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- State the theorem
theorem min_value_of_m (φ : ℝ) (h1 : |φ| < π/2) :
  let f_translated (x : ℝ) := f (x + π/6) φ
  (∀ x, f_translated x = f_translated (-x)) →  -- f_translated is even
  (∃ m : ℝ, m = -1/2 ∧ 
    (∃ x : ℝ, x ∈ Set.Icc 0 (π/2) ∧ f x φ ≤ m) ∧
    (∀ m' : ℝ, m' < m → 
      ∀ x : ℝ, x ∈ Set.Icc 0 (π/2) → f x φ > m')) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l939_93908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_ratio_l939_93976

/-- Proves that for an ellipse intersecting with a line, if the slope of the line passing through
    the origin and the midpoint of the intersection points is √2/2, then the ratio of the
    coefficients in the ellipse equation is √2/2. -/
theorem ellipse_line_intersection_ratio (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m ≠ n) :
  let C := {p : ℝ × ℝ | m * p.1^2 + n * p.2^2 = 1}
  let L := {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}
  ∀ (A B : ℝ × ℝ), A ∈ C → A ∈ L → B ∈ C → B ∈ L → A ≠ B →
  let M := (A + B) / 2
  (M.2 / M.1 = Real.sqrt 2 / 2) →
  m / n = Real.sqrt 2 / 2
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_ratio_l939_93976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_equal_surface_area_polyhedra_l939_93948

noncomputable def tetrahedron_surface_area (a : ℝ) : ℝ := a^2 * Real.sqrt 3

def hexahedron_surface_area (b : ℝ) : ℝ := 6 * b^2

noncomputable def octahedron_surface_area (c : ℝ) : ℝ := 2 * c^2 * Real.sqrt 3

noncomputable def tetrahedron_volume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 12

def hexahedron_volume (b : ℝ) : ℝ := b^3

noncomputable def octahedron_volume (c : ℝ) : ℝ := (c^3 * Real.sqrt 2) / 3

theorem volume_ratio_of_equal_surface_area_polyhedra (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : tetrahedron_surface_area a = hexahedron_surface_area b)
  (h5 : hexahedron_surface_area b = octahedron_surface_area c) :
  ∃ (k : ℝ), k > 0 ∧ 
    tetrahedron_volume a = k ∧
    hexahedron_volume b = k * Real.rpow 3 (1/4) ∧
    octahedron_volume c = k * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_equal_surface_area_polyhedra_l939_93948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l939_93973

/-- Circle defined by the equation x^2 + y^2 - 8x + 15 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 15 = 0

/-- Parabola defined by the equation y^2 = 4x -/
def Parabola (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- Distance between two points (x1, y1) and (x2, y2) -/
noncomputable def Distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating the minimum distance between the circle and parabola -/
theorem min_distance_circle_parabola :
  ∃ (d : ℝ), d = 2 * Real.sqrt 3 - 1 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
    Circle x1 y1 → Parabola x2 y2 →
    Distance x1 y1 x2 y2 ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l939_93973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l939_93959

noncomputable section

-- Define the functions
def f1 (a : ℝ) (x : ℝ) : ℝ := |x + a|
def f2 (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / x

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x < -1, Monotone (fun x => f1 a x)) →
  (a > 0) →
  (∀ x > 2, StrictMono (fun x => f2 a x)) →
  a ∈ Set.Ioc 0 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l939_93959
